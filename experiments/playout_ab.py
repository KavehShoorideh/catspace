#!/usr/bin/env python
"""
experiments/playout_ab.py — paired PLAYOUT A/B with a DETERMINISTIC defender.

The lesson from move_ab: endgame play only diverges when each model drives its OWN
trajectory (fixed-position eval can't see it), and SF-conversion is too
high-variance (CI +-0.38 at n=200) because the opponent is stochastic. This plays
each model (White, hop search) against a TABLEBASE-OPTIMAL defender (Black,
deterministic -> zero opponent variance), from a set of winning starts, and scores
mate-within-budget. Because both the model (argmax) and the defender are
deterministic, the per-start result is exact and reproducible -- the paired diff
vs the incumbent has real power (variance only from which starts we sampled).

VERDICT: mate-rate A vs B, paired diff, bootstrap CI over starts, and mean
plies-to-mate among converted (lower = crisper conversion).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import chess
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.value_fixed_point import TB, tb_best_move


def playout(pol, start, tb, rng, max_plies):
    """White = model (hop search), Black = tablebase-optimal. Return (mated, plies)."""
    b = start.copy(stack=False)
    seen = set()
    for ply in range(max_plies):
        if b.is_game_over(claim_draw=True):
            break
        if b.turn == chess.WHITE:
            m = pol.move(b, rng)
        else:
            m = tb_best_move(b, tb, seen); seen.add(b.board_fen())
        if m is None:
            break
        b.push(m)
    out = b.outcome(claim_draw=True)
    mated = 1.0 if (out and out.winner == chess.WHITE) else 0.0
    return mated, (b.ply() if mated else None)


def mate_vector(ckpt, starts, tb, nodes, beam, max_plies, seed, device, bank_boards=None,
                search="beam", c_puct=1.5, s_head_path=None, g_sharp=0.0, rescue=False,
                committor_path=None, clearance_beta=0.0, phead_path=None,
                detect_threefold=True, coherence_k=0.0, certainty_stop=0.0):
    from catspace.nn.fb import load_ckpt, pick_device
    from catspace.nn.policy_fb import make_search_policy
    dev = pick_device(device)
    fb, pay = load_ckpt(Path(ckpt), dev)
    if bank_boards is not None:                          # region goal: soft-min over exemplars
        from catspace.goal_bank import embed_bank
        z = embed_bank(fb, bank_boards, dev)             # (m, d) -> soft_min_bank readout
    else:
        z = pay["zgoals"]["MATE_W"]                      # centroid goal
    s_head = None
    if s_head_path:
        import torch
        hp = torch.load(s_head_path, map_location=dev, weights_only=False)
        s_head = torch.nn.Sequential(torch.nn.Linear(hp["d_in"], 128), torch.nn.ReLU(),
                                     torch.nn.Linear(128, 1), torch.nn.Softplus()).to(dev)
        s_head.load_state_dict(hp["state"]); s_head.eval()
    kw = {}
    if phead_path:
        import torch
        from catspace.nn.eval_head import EvalHead
        hp = torch.load(phead_path, map_location=dev, weights_only=False)
        ph = EvalHead(d_in=hp["d_in"]).to(dev)
        ph.load_state_dict(hp["state"]); ph.eval()

        pcb = clearance_beta                         # phead-native draw clearance

        class PheadCommittor(torch.nn.Module):
            def forward(self, f):
                p = torch.softmax(ph(f), dim=1)
                pw = p[:, 0].clamp_min(1e-6)
                dW = -torch.log(pw)                  # committor to the win surface
                if pcb:
                    # reach = -d_W + beta*d_D = ln P_win - beta*ln P_draw: steer
                    # toward the win AND away from the draw basin (the drift the
                    # toy dies to). d_W head returns the DISTANCE the readout
                    # negates, so add -beta*d_D here = +beta*ln P_draw... encode
                    # by RETURNING d_W - beta*d_D (readout negates -> reach).
                    pd = p[:, 1].clamp_min(1e-6)
                    dW = dW - pcb * (-torch.log(pd))
                return dW.unsqueeze(-1)
        kw["committor_head"] = PheadCommittor()
        if certainty_stop > 0.0 and search == "mcts":
            # obvious-region recognizer uses the RAW phead softmax (W/D/L)
            kw["certainty_head"] = ph
            kw["certainty_stop"] = certainty_stop
            print(f"obvious-region soft-terminal: certainty_stop={certainty_stop}")
        if clearance_beta:
            print(f"phead committor + draw clearance beta={clearance_beta}")
    elif committor_path:
        import torch

        def load_head(p):
            hp = torch.load(p, map_location=dev, weights_only=False)
            h = torch.nn.Sequential(torch.nn.Linear(hp["d_in"], 128), torch.nn.ReLU(),
                                    torch.nn.Linear(128, 1), torch.nn.Softplus()).to(dev)
            h.load_state_dict(hp["state"]); h.eval()
            return h
        kw["committor_head"] = load_head(committor_path)
        if clearance_beta:
            dpath = committor_path.replace("_whead", "_dhead")
            kw["committor_dhead"] = load_head(dpath)
            kw["clearance_beta"] = clearance_beta
            print(f"clearance readout: beta={clearance_beta} dhead={dpath}")
    if rescue:
        import numpy as _np
        ev = {}
        for t in ("certainty_table_demo_tb", "certainty_table_eps05", "certainty_table_r2_K16"):
            path = Path(f"artifacts/experiments/{t}.json")
            if not path.exists():
                continue
            for r in json.loads(path.read_text())["rows"]:
                d_ev = ((r["plies"] if r["plies"] is not None else 100.0)
                        + 8.0 * (-_np.log(max(r["p_hat"], 1.0 / (r["n"] + 2))))) / 50.0
                n0, d0 = ev.get(r["fen"], (0.0, 0.0))
                ev[r["fen"]] = (n0 + r["n"], (n0 * d0 + r["n"] * d_ev) / (n0 + r["n"]))
        kw = dict(evidence=ev, rollout_on_flat=True, tree_reuse=True)
        print(f"rescue: {len(ev)} evidence states, rollouts+reuse ON")
    if search == "mcts" and not detect_threefold:
        kw["detect_threefold"] = False
    if search == "mcts" and coherence_k:
        kw["coherence_k"] = coherence_k
        print(f"coherence-length backup discount k={coherence_k}")
    pol = make_search_policy(search, fb, z, max_nodes=nodes, beam=beam,
                             c_puct=c_puct, device=dev, s_head=s_head, g_sharp=g_sharp, **kw)
    mated, plies = [], []
    for i, fen in enumerate(starts):
        rng = np.random.default_rng([seed, i])
        m, p = playout(pol, chess.Board(fen), tb, rng, max_plies)
        mated.append(m)
        if p is not None:
            plies.append(p)
    return np.array(mated), (float(np.mean(plies)) if plies else float("nan"))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt-a", required=True)
    ap.add_argument("--ckpt-b", required=True)
    ap.add_argument("--fixed-set", default="artifacts/experiments/krrkbp_test_n200.json")
    ap.add_argument("--n", type=int, default=100, help="number of starts to play")
    ap.add_argument("--nodes", type=int, default=200)
    ap.add_argument("--nodes-b", type=int, default=None,
                    help="hop-search node budget for ckpt-b (default = --nodes). Set higher to "
                         "A/B search DEPTH on the same checkpoint: is the ceiling search or embedding?")
    ap.add_argument("--beam", type=int, default=4)
    ap.add_argument("--max-plies", type=int, default=120)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--syzygy-dir", default="data/syzygy")
    ap.add_argument("--label", default="")
    ap.add_argument("--ckpt-b-goal", choices=("centroid", "bank"), default="centroid",
                    help="goal used by ckpt-b's planner: centroid (zgoals MATE_W) or a soft-min "
                         "BANK of mate exemplars (Kaveh's 'arrive anywhere in the mate region')")
    ap.add_argument("--bank-shards", nargs="+", default=["data/selfplay/krrkbp_sfsf"])
    ap.add_argument("--bank-max-pieces", type=int, default=6)
    ap.add_argument("--bank-size", type=int, default=128)
    ap.add_argument("--search-a", choices=("beam", "mcts", "anytime"), default="beam")
    ap.add_argument("--search-b", choices=("beam", "mcts", "anytime"), default="beam",
                    help="readout for each side: beam = FBSearchPolicy minimax, mcts = PUCT "
                         "(catspace/nn/mcts.py). Same node budget = matched compute.")
    ap.add_argument("--c-puct", type=float, default=1.5)
    ap.add_argument("--s-head-b", default=None, help="sharpness head for side B (two-channel readout)")
    ap.add_argument("--g-sharp", type=float, default=1.0, help="risk weight for side B's S penalty")
    ap.add_argument("--rescue-b", action="store_true",
                    help="side B: evidence blend + flat/low-conf rollouts + tree reuse")
    ap.add_argument("--no-threefold-a", action="store_true",
                    help="disable path-aware threefold detection on side A "
                         "(baseline for A/B: does seeing repetitions help?)")
    ap.add_argument("--committor-b", default=None,
                    help="committor head (*_whead.pt) for side B: reach = -d_W(s), "
                         "no goal vector (mcts only)")
    ap.add_argument("--committor-a", default=None,
                    help="committor head for side A (paired readout A/Bs on the "
                         "same checkpoint)")
    ap.add_argument("--clearance-beta", type=float, default=0.0,
                    help="side B: draw-surface clearance weight (reach = -d_W + "
                         "beta*d_D; needs the _dhead sibling of --committor-b)")
    ap.add_argument("--certainty-stop", type=float, default=0.0,
                    help="B-side only (MCTS+phead): obvious-region soft-terminal. A node "
                         "whose phead confidence (peak W/D/L prob) >= this is treated as a "
                         "RESOLVED region -- its committor value backs up and the search "
                         "stops there instead of recursing to mate. 0=off. Try 0.9.")
    ap.add_argument("--coherence-k", type=float, default=0.0,
                    help="B-side only: coherence-length backup discount strength "
                         "(MCTS). 0=off (flat backup); >0 trusts the best-case field "
                         "deep on FORCED lines, discounts it through DIVERGENT nodes.")
    ap.add_argument("--phead-b", default=None,
                    help="side B: full-board outcome head (*_phead.pt) as the "
                         "W-committor readout (d_W = -ln P_win from the 3-class "
                         "head; zero-training full-board->toy transfer)")
    ap.add_argument("--phead-a", default=None, help="side A phead readout (for "
                    "clearance A/B: phead both sides, clearance differs)")
    ap.add_argument("--clearance-a", type=float, default=0.0,
                    help="side A draw-clearance beta (phead readout)")
    args = ap.parse_args()

    import torch  # noqa: F401
    tb = TB(args.syzygy_dir)
    starts = json.loads(Path(args.fixed_set).read_text())["fens"][:args.n]
    bank_boards = None
    if args.ckpt_b_goal == "bank":
        from catspace.goal_bank import harvest_mate_finals
        bank_boards = harvest_mate_finals(args.bank_shards, want_result=1,
                                          max_pieces=args.bank_max_pieces, cap=args.bank_size)
        print(f"goal bank: {len(bank_boards)} white-mate exemplars (<= {args.bank_max_pieces} pieces)")
    a, pa = mate_vector(args.ckpt_a, starts, tb, args.nodes, args.beam, args.max_plies,
                        args.seed, args.device, search=args.search_a, c_puct=args.c_puct,
                        committor_path=args.committor_a,
                        detect_threefold=not args.no_threefold_a,
                        phead_path=args.phead_a, clearance_beta=args.clearance_a)
    b, pb = mate_vector(args.ckpt_b, starts, tb, args.nodes_b or args.nodes, args.beam,
                        args.max_plies, args.seed, args.device, bank_boards=bank_boards,
                        search=args.search_b, c_puct=args.c_puct,
                        s_head_path=args.s_head_b, g_sharp=args.g_sharp, rescue=args.rescue_b,
                        committor_path=args.committor_b, clearance_beta=args.clearance_beta,
                        phead_path=args.phead_b, coherence_k=args.coherence_k,
                        certainty_stop=args.certainty_stop)
    tb.close()
    n = len(starts)
    diff = float(b.mean() - a.mean())
    rng = np.random.default_rng(0)
    idx = rng.integers(0, n, size=(args.boot, n))
    boot = b[idx].mean(1) - a[idx].mean(1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    sig = (lo > 0 or hi < 0)
    # anytime-valid sign-test e-process over the paired per-start diffs
    # (catspace.abtest): e-values compose across sequential looks (e.g. the
    # data-scaling curve's repeated money tests), unlike bootstrap CIs
    from catspace.abtest import EValueTest
    ev = EValueTest()
    for d in (b - a):
        ev.update(float(d))
    print(f"PLAYOUT_AB {args.label} mate-rate A={a.mean():.3f} vs B={b.mean():.3f}  "
          f"diff={diff:+.3f} CI=[{lo:+.3f},{hi:+.3f}]  e={ev.e:.2f} "
          f"(n={n} starts, {ev.n} decisive, deterministic defender; "
          f"plies-to-mate A={pa:.0f} B={pb:.0f}) "
          f"[{'SIGNIFICANT' if sig else 'ns'}]")


if __name__ == "__main__":
    main()
