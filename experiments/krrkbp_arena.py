#!/usr/bin/env python
"""
experiments/krrkbp_arena.py — matched-pair comparison of two White policies
(baseline FBSearchPolicy vs FBPlanPolicy by default) on the fixed KRRvKBP
diagnostic set (catspace/diagnostic_krrkbp.py, artifacts/experiments/
krrkbp_fixed_set.json), Stockfish always Black (colors fixed per Kaveh).

Matched-seed pairing, not two independent unpaired batches: for each of the
20 fixed starting positions, BOTH candidate policies play the SAME position
against the SAME opponent with the SAME rng seed, isolating the effect of
the policy itself from opponent-randomness or position difficulty. The
per-position score DIFFERENCE (b - a) feeds catspace.abtest.EValueTest's
anytime-valid sign test (same e-process arena_real.py uses, but on paired
diffs instead of unpaired win/loss) plus a time-uniform confidence sequence
on the mean diff (abtest.confidence_sequence) -- both already built,
neither previously wired to a real-board comparison.

Syzygy tablebase DTZ (data/syzygy/) is looked up per starting position
purely for the printed readout -- observational only, never feeds the
score or the statistical test (Kaveh: "if it wins some other way, who am I
to penalize it? ... use the tablebase to tell me what was the actual
distance to mate so I can compare to my planner when inspecting visually").
"""
from __future__ import annotations

import argparse
from pathlib import Path

import chess
import chess.syzygy
import numpy as np

from catspace.abtest import EValueTest, confidence_sequence
from catspace.diagnostic_krrkbp import load_fixed_set
from catspace.io.paths import derived_dir
from catspace.realboard import play_board_game
from catspace.uci import UCIBoardPolicy

RESULT_SCORE = {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5, "*": 0.5}


def dtz_of(board: chess.Board, tablebase) -> int | None:
    if tablebase is None:
        return None
    try:
        return tablebase.probe_dtz(board)
    except (KeyError, chess.syzygy.MissingTableError):
        return None


def run_paired(policy_a_factory, policy_b_factory, name_a: str, name_b: str, positions: list,
               opponent: UCIBoardPolicy, max_plies: int, base_seed: int, alpha: float,
               tablebase=None, early_stop: bool = True, verbose: bool = True) -> dict:
    """opponent must already be inside its `with` block (this doesn't own
    its lifetime, matching arena_real.run_arena's convention)."""
    test = EValueTest()
    diffs: list[float] = []
    rows = []
    for i, start in enumerate(positions):
        dtz = dtz_of(start, tablebase)
        outcomes = {}
        for tag, factory in (("a", policy_a_factory), ("b", policy_b_factory)):
            rng = np.random.default_rng([base_seed, i])
            white = factory()
            rec = play_board_game(white, opponent, start=start.copy(stack=False),
                                  opening_plies=0, max_plies=max_plies, rng=rng)
            outcomes[tag] = dict(score=RESULT_SCORE[rec.result], result=rec.result,
                                 n_plies=rec.n_plies, termination=rec.termination)
        diff = outcomes["b"]["score"] - outcomes["a"]["score"]
        diffs.append(diff)
        e = test.update(diff)
        rows.append(dict(i=i, fen=start.fen(), dtz=dtz, a=outcomes["a"], b=outcomes["b"], diff=diff))
        if verbose:
            print(f"  pos {i:02d} dtz={dtz!s:>5}  {name_a}={outcomes['a']['result']:>7}"
                  f"({outcomes['a']['n_plies']:3d}pl)  {name_b}={outcomes['b']['result']:>7}"
                  f"({outcomes['b']['n_plies']:3d}pl)  diff={diff:+.1f}  e={e:.2f}", flush=True)
        if early_stop and test.reject_at(alpha) and i + 1 < len(positions):
            if verbose:
                print(f"  early stop: e-value crossed 1/alpha={1/alpha:.1f} after {i + 1} positions")
            break

    ci = confidence_sequence(np.array(diffs), alpha=alpha)
    a_mean = float(np.mean([r["a"]["score"] for r in rows]))
    b_mean = float(np.mean([r["b"]["score"] for r in rows]))
    return dict(name_a=name_a, name_b=name_b, n_positions=len(rows), rows=rows,
               a_score_mean=a_mean, b_score_mean=b_mean, mean_diff=float(np.mean(diffs)),
               diff_ci=ci, e_value=test.e, n_decisive=test.n, k_b_wins=test.k,
               reject_at_alpha=test.reject_at(alpha), alpha=alpha)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fixed-set", default="artifacts/experiments/krrkbp_fixed_set.json")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--opponent", default="sf:skill=0")
    ap.add_argument("--max-plies", type=int, default=150)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--syzygy-dir", default="data/syzygy")
    ap.add_argument("--no-early-stop", action="store_true")
    ap.add_argument("--baseline-nodes", type=int, default=200)
    ap.add_argument("--baseline-beam", type=int, default=4)
    ap.add_argument("--plan-nodes", type=int, default=2000)
    ap.add_argument("--plan-beam", type=int, default=4)
    ap.add_argument("--shallow-nodes", type=int, default=60)
    ap.add_argument("--shallow-beam", type=int, default=3)
    args = ap.parse_args()

    import torch  # noqa: F401  (fail early with a clear message if .[nn] absent)
    from catspace.nn.fb import load_ckpt, pick_device
    from catspace.nn.policy_fb import FBPlanPolicy, FBSearchPolicy

    device = pick_device(args.device)
    fb, payload = load_ckpt(Path(args.ckpt) if args.ckpt else derived_dir() / "lichess_fb.pt", device)
    if "MATE_W" not in payload.get("zgoals", {}):
        raise SystemExit("checkpoint has no zgoals -- finish a train_lichess_fb.py run first")
    z_white = payload["zgoals"]["MATE_W"]

    positions = load_fixed_set(args.fixed_set)
    print(f"{len(positions)} fixed KRRvKBP positions loaded from {args.fixed_set}")

    def make_baseline():
        return FBSearchPolicy(fb, z_white, max_nodes=args.baseline_nodes,
                              beam=args.baseline_beam, device=device)

    def make_plan():
        return FBPlanPolicy(fb, z_white, plan_nodes=args.plan_nodes, plan_beam=args.plan_beam,
                            shallow_nodes=args.shallow_nodes, shallow_beam=args.shallow_beam,
                            device=device)

    tablebase = None
    if args.syzygy_dir and Path(args.syzygy_dir).exists():
        tablebase = chess.syzygy.open_tablebase(args.syzygy_dir)

    if not args.opponent.startswith("sf:"):
        raise SystemExit("only sf:<elo>|sf:skill=<k> supported "
                          "(Stockfish always plays Black in this diagnostic)")
    arg = args.opponent[3:]
    opp = (UCIBoardPolicy(skill=int(arg[6:]), movetime=0.02) if arg.startswith("skill=")
          else UCIBoardPolicy(elo=int(arg), movetime=0.05))

    print(f"FBSearchPolicy(max_nodes={args.baseline_nodes}, beam={args.baseline_beam}) vs "
          f"FBPlanPolicy(plan_nodes={args.plan_nodes}/beam={args.plan_beam}, "
          f"shallow_nodes={args.shallow_nodes}/beam={args.shallow_beam}) "
          f"as White, vs {args.opponent} as Black, device={device}")

    with opp:
        result = run_paired(make_baseline, make_plan, "FBSearchPolicy", "FBPlanPolicy",
                            positions, opp, args.max_plies, args.seed, args.alpha,
                            tablebase=tablebase, early_stop=not args.no_early_stop, verbose=True)

    if tablebase is not None:
        tablebase.close()

    lo, hi = result["diff_ci"]
    print(f"\nVERDICT FBSearchPolicy={result['a_score_mean']:.3f} vs "
          f"FBPlanPolicy={result['b_score_mean']:.3f}  (n={result['n_positions']} positions, "
          f"mean_diff={result['mean_diff']:+.3f} CI=[{lo:+.3f},{hi:+.3f}], "
          f"e={result['e_value']:.2f}, "
          f"{'REJECT(diff!=0)' if result['reject_at_alpha'] else 'no rejection'} at alpha={args.alpha})")


if __name__ == "__main__":
    main()
