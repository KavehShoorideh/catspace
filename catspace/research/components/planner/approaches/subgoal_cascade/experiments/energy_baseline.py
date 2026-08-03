#!/usr/bin/env python
"""
catspace/research/components/planner/approaches/subgoal_cascade/experiments/energy_baseline.py — the compute–strength Pareto instrument
(Kaveh 2026-07-18: planner objective = E_mu[score] - c*compute; before any
probe/cascade is built, measure what a decision COSTS today).

Plays one checkpoint (White) against the TABLEBASE-OPTIMAL defender on the
fixed toy set — the exact playout_ab protocol (deterministic both sides,
mate-within-budget scoring), so conversion numbers land on the same scale as
the 0.600 re-baseline — at one or more (policy, budget) configs, and prints a
VERDICT line per config:

  conversion       mate-rate (the strength axis)
  rows/move        TRUE energy: rows through fb.embed_F / fb.embed_B per
                   White move. Counts every fresh network forward by every
                   component (reach, committor, certainty, subgoal embeds),
                   skips cache hits, and is policy-agnostic — mcts / beam /
                   plan are directly comparable where their own budget
                   accounting is not.
  evals/move       the policy's internal budget counter where one exists
                   (mcts only), for cross-checking rows/move.
  ms/move          wall-clock (device-labeled; CPU numbers give shape, not
                   absolutes, while training holds the GPU).
  util             rows/move / nominal budget — how much of the fixed budget
                   a move actually consumes. MCTS has NO stopping rule (it
                   spends the budget every move by construction); this column
                   is the flatness the decision cascade exists to beat.

Protocol notes (matched to playout_ab for comparability):
  - ONE policy instance per config, shared across all n starts — the eval
    cache (and FBMCTSPolicy.path_counts) persist across starts, exactly as in
    mate_vector. Cache warming across games is part of the real energy story.
  - policy 'mcts' uses the phead committor readout (-ln P_win), the 0.600
    baseline config. 'beam' and 'plan' use the zgoals["MATE_W"] readout their
    historical runs used (plan = the shelved FBPlanPolicy, rounds 10-12: a
    strength wash, e=0.47, never priced on compute — priced here).

Usage (CPU while the GPU trains; rerun on MPS for absolute ms later):
  .venv/bin/python catspace/research/components/planner/approaches/subgoal_cascade/experiments/energy_baseline.py \
      --ckpt data/derived/sep/cert_base_full.pt \
      --phead data/derived/sep/cert_base_full_phead.pt \
      --policies mcts --budgets 200 800 1600 --n 100 --device cpu
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import chess
import numpy as np


from catspace.research.components.planner.approaches.committor_value.experiments.value_fixed_point import TB, tb_best_move
from catspace.io import paths


class EnergyMeter:
    """Counts rows through fb.embed_F / fb.embed_B (fresh network forwards).
    Instance-attribute shadowing of the bound methods; every policy built on
    this fb instance is metered automatically."""

    def __init__(self, fb):
        self.rows = 0
        for name in ("embed_F", "embed_B"):
            orig = getattr(fb, name)
            setattr(fb, name, self._wrap(orig))

    def _wrap(self, fn):
        def counted(planes, *a, **k):
            self.rows += len(planes)
            return fn(planes, *a, **k)
        return counted


def playout_profiled(pol, start, tb, rng, max_plies, meter, mcts=None):
    """playout_ab.playout with per-White-move (rows, evals, ms) capture."""
    b = start.copy(stack=False)
    seen = set()
    per_move = []
    for _ in range(max_plies):
        if b.is_game_over(claim_draw=True):
            break
        if b.turn == chess.WHITE:
            r0, t0 = meter.rows, time.perf_counter()
            m = pol.move(b, rng)
            ms = (time.perf_counter() - t0) * 1e3
            ev = mcts.evals_used if mcts is not None else None
            per_move.append((meter.rows - r0, ev, ms))
        else:
            m = tb_best_move(b, tb, seen)
            seen.add(b.board_fen())
        if m is None:
            break
        b.push(m)
    out = b.outcome(claim_draw=True)
    mated = 1.0 if (out and out.winner == chess.WHITE) else 0.0
    return mated, (b.ply() if mated else None), per_move


def build_policy(kind, fb, pay, phead, nodes, dev, plan_nodes, shallow_nodes,
                 early_stop=False, mate_stop=False):
    z = pay["zgoals"]["MATE_W"]
    if kind == "mcts":
        import torch
        from catspace.research.components.encoder.approaches.jepa_tokenizer.src.policy_fb import make_search_policy

        class Committor(torch.nn.Module):
            def forward(self, f):
                p = torch.softmax(phead(f), dim=1)
                return -torch.log(p[:, 0].clamp_min(1e-6)).unsqueeze(-1)

        pol = make_search_policy("mcts", fb, z, max_nodes=nodes, device=dev,
                                 committor_head=Committor(),
                                 decision_stop=early_stop,
                                 mate_stop=early_stop or mate_stop)
        return pol, pol.mcts
    if kind == "mctsplan":
        import torch
        from catspace.research.components.encoder.approaches.jepa_tokenizer.src.policy_fb import make_search_policy

        class Committor(torch.nn.Module):
            def forward(self, f):
                p = torch.softmax(phead(f), dim=1)
                return -torch.log(p[:, 0].clamp_min(1e-6)).unsqueeze(-1)

        pol = make_search_policy("mctsplan", fb, z, max_nodes=nodes, device=dev,
                                 committor_head=Committor())
        return pol, pol.mcts
    if kind == "beam":
        from catspace.research.components.encoder.approaches.jepa_tokenizer.src.policy_fb import make_search_policy
        return make_search_policy("beam", fb, z, max_nodes=nodes, beam=4,
                                  device=dev), None
    if kind == "plan":
        # factory, not instance: FBPlanPolicy carries game state (active
        # subgoal, plies-since-plan, reach-at-plan) that must NOT leak
        # across starts (review 2026-07-18 MED) -- a fresh policy per game,
        # exactly one plan lifecycle per playout
        from catspace.research.components.encoder.approaches.jepa_tokenizer.src.policy_fb import FBPlanPolicy

        def fresh():
            return FBPlanPolicy(fb, z, plan_nodes=plan_nodes,
                                shallow_nodes=shallow_nodes, device=dev)
        return fresh, None
    raise ValueError(kind)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default=paths.sep("cert_base_full.pt"))
    ap.add_argument("--phead", default=paths.sep("cert_base_full_phead.pt"))
    ap.add_argument("--fixed-set", default=paths.experiment("krrkbp_test_n200.json"))
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--policies", nargs="+", default=["mcts"],
                    choices=["mcts", "beam", "plan", "mctsplan"])
    ap.add_argument("--budgets", nargs="+", type=int, default=[200, 800, 1600],
                    help="node budgets (mcts/beam; 'plan' ignores these and "
                         "uses --plan-nodes/--shallow-nodes once)")
    ap.add_argument("--plan-nodes", type=int, default=2000)
    ap.add_argument("--shallow-nodes", type=int, default=60)
    ap.add_argument("--max-plies", type=int, default=120)
    ap.add_argument("--early-stop", action="store_true",
                    help="mcts only: BOTH stops (stability heuristic + certified "
                         "mate-stop) -- the v1/v2 measured configuration. "
                         "Stability measured harmful at 800n; see JOURNAL.")
    ap.add_argument("--mate-stop", action="store_true",
                    help="mcts only: certified mate-stop ALONE (provably "
                         "move-identical readout; pure energy saving).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--syzygy-dir", default=str(paths.syzygy_dir()))
    ap.add_argument("--out", default=paths.experiment("energy_baseline.jsonl"))
    args = ap.parse_args()

    import torch
    from catspace.research.components.encoder.approaches.jepa_tokenizer.src.eval_head import EvalHead
    from catspace.research.components.encoder.approaches.jepa_tokenizer.src.fb import load_ckpt, pick_device
    dev = pick_device(args.device)
    fb, pay = load_ckpt(Path(args.ckpt), dev)
    hp = torch.load(args.phead, map_location=dev, weights_only=False)
    phead = EvalHead(d_in=hp["d_in"]).to(dev)
    phead.load_state_dict(hp["state"])
    phead.eval()
    meter = EnergyMeter(fb)

    starts = json.loads(Path(args.fixed_set).read_text())["fens"][:args.n]
    tb = TB(args.syzygy_dir)
    records = []
    t_all = time.perf_counter()
    configs = [(k, b) for k in args.policies
               for b in ([None] if k == "plan" else args.budgets)]
    for kind, nodes in configs:
        pol, mcts = build_policy(kind, fb, pay, phead, nodes, dev,
                                 args.plan_nodes, args.shallow_nodes,
                                 early_stop=args.early_stop,
                                 mate_stop=args.mate_stop)
        mated, plies, rows_m, evals_m, ms_m, moves = [], [], [], [], [], 0
        t0 = time.perf_counter()
        for i, fen in enumerate(starts):
            rng = np.random.default_rng([args.seed, i])
            game_pol = pol() if callable(pol) else pol
            m, p, per_move = playout_profiled(game_pol, chess.Board(fen), tb, rng,
                                              args.max_plies, meter, mcts)
            mated.append(m)
            if p is not None:
                plies.append(p)
            rows_m += [r for r, _, _ in per_move]
            evals_m += [e for _, e, _ in per_move if e is not None]
            ms_m += [t for _, _, t in per_move]
            moves += len(per_move)
        elapsed = time.perf_counter() - t0
        conv = float(np.mean(mated))
        rows_arr, ms_arr = np.array(rows_m), np.array(ms_m)
        nominal = nodes if nodes else args.plan_nodes
        suffix = ("+stop" if args.early_stop else "+mstop" if args.mate_stop else "") \
            if kind == "mcts" else ""
        label = (f"{kind}{suffix}"
                 + (f"@{nodes}n" if nodes else f"@{args.plan_nodes}/{args.shallow_nodes}n"))
        rec = dict(policy=kind, early_stop=bool(args.early_stop and kind == "mcts"),
                   mate_stop=bool((args.mate_stop or args.early_stop) and kind == "mcts"),
                   nodes=nodes, plan_nodes=args.plan_nodes if kind == "plan" else None,
                   conversion=conv, n_starts=len(starts), moves=moves,
                   rows_per_move=float(rows_arr.mean()),
                   rows_p50=float(np.percentile(rows_arr, 50)),
                   rows_p90=float(np.percentile(rows_arr, 90)),
                   evals_per_move=(float(np.mean(evals_m)) if evals_m else None),
                   ms_per_move=float(ms_arr.mean()),
                   ms_p50=float(np.percentile(ms_arr, 50)),
                   util=float(rows_arr.mean() / nominal),
                   plies_to_mate=(float(np.mean(plies)) if plies else None),
                   device=str(dev), ckpt=args.ckpt, seed=args.seed,
                   elapsed_s=elapsed)
        records.append(rec)
        ev_txt = f"evals/move={np.mean(evals_m):.0f} " if evals_m else ""
        ptm_txt = f"{np.mean(plies):.0f}" if plies else "nan"
        print(f"VERDICT ENERGY {label} conversion={conv:.3f} "
              f"rows/move={rows_arr.mean():.0f} (p50={np.percentile(rows_arr, 50):.0f} "
              f"p90={np.percentile(rows_arr, 90):.0f}) {ev_txt}"
              f"ms/move={ms_arr.mean():.0f} util={rows_arr.mean() / nominal:.2f} "
              f"plies-to-mate={ptm_txt} "
              f"(n={len(starts)}, {moves} moves, {dev}, {elapsed:.0f}s)")
    tb.close()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"TOTAL {time.perf_counter() - t_all:.0f}s -> {out}")


if __name__ == "__main__":
    main()
