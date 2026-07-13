#!/usr/bin/env python
"""
experiments/sharpness_bench.py — GROUND-TRUTH SHARPNESS benchmark.

2026-07-13 (Kaveh's reframe): the tactical/positional boundary is not temporal
depth, it's LOCAL SHARPNESS of the value landscape. A sharp position is one where
one tempo flips the result (you must not prune); a smooth one is where many
move-orders converge (a coarse estimate suffices). Any handover keyed on ply is
mis-specified -- it should be driven by an UNCERTAINTY signal the model emits.

This benchmark is the measurement backbone for testing that: it gives an EXACT,
tablebase-derived ground-truth sharpness for a position, then scores any
candidate uncertainty signal by its rank-correlation (Spearman rho) with that
truth. It's the sharpness analogue of qm_fitness_probe's Syzygy distance
calibration -- and lets us rank the A/B/C/D uncertainty options as sharpness
DETECTORS on ground truth, before any of them touch the search.

Ground-truth sharpness of a WINNING position (side-to-move wins under perfect
play, tablebase WDL=+2): for each legal move, does it PRESERVE the win? A move
holds iff the resulting position is a loss for the opponent (WDL < 0 from their
POV) or delivers mate. sharpness = 1 - (fraction of moves that hold). Only 1 of
20 holds -> sharpness ~0.95 (must find THE move); 18/20 hold -> ~0.10 (quiet).

Candidate uncertainty signals (pluggable; scored by rho vs sharpness):
  score_spread       any checkpoint: std of the value head's move-scores.
                     A point estimate's own "one move dominates" signal.
  head_disagreement  two-horizon: 1 - rank-corr between the NEAR and FAR heads'
                     move orderings. High disagree = the heads can't agree on the
                     best move = the sharp regime (Option A, ~free from the
                     two-head run). This is the hypothesis-validating signal.
  (later) dist_sigma the distributional far head's predicted spread (Option B).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import chess
import chess.syzygy
import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # run as a plain script

from catspace.data.encode import encode_meta, encode_packed
from catspace.diagnostic_krrkbp import random_krrkbp
from catspace.io.paths import derived_dir
from catspace.nn.features import feature_planes, omega_ids
from experiments.qm_fitness_probe import random_krvk
from experiments.selfplay_generate import random_endgame_start


# ------------------------------------------------------------ ground truth
def _move_cost(b2: chess.Board, tb, big: float) -> float | None:
    """'Progress cost' of the position AFTER my move (opponent to move in b2):
    0 if I just mated; DTZ magnitude (plies toward the next zeroing move in the
    winning line) if the opponent is still lost -- smaller = I convert faster;
    `big` if I threw the win to a draw or worse. None if unprobeable."""
    if b2.is_checkmate():
        return 0.0
    if b2.is_stalemate() or b2.is_insufficient_material() or b2.can_claim_draw():
        return big
    try:
        wdl_opp = tb.probe_wdl(b2)
        if wdl_opp >= 0:                   # opponent not losing => I threw it
            return big
        return float(abs(tb.probe_dtz(b2)))
    except (KeyError, chess.syzygy.MissingTableError):
        return None


def ground_truth_sharpness(board: chess.Board, tb, margin: float = 2.0,
                           big: float = 1000.0) -> dict | None:
    """Sharpness of a position the SIDE-TO-MOVE wins (caller ensures WDL=+2),
    measured as value CURVATURE over the legal moves: how tightly does progress
    depend on the exact move? For each move, its progress cost (`_move_cost`):
    a SMOOTH position has many moves near the best cost (any reasonable move
    keeps converting); a SHARP one has few (one tempo matters). Two scalars:

      dtz_sharpness  = 1 - (fraction of moves within `margin` of the best cost)
                       -- the fine, conversion-relevant curvature (a rook hang
                       that still wins by WDL shows up here as a big cost jump).
      result_sharpness = 1 - (fraction of moves that preserve the WDL win)
                       -- the coarse, blunder-relevant version (flips only near
                       the win/draw boundary).

    None if too few moves or successors unprobeable."""
    moves = list(board.legal_moves)
    if len(moves) < 2:
        return None
    costs, holds = [], 0
    for m in moves:
        b2 = board.copy(stack=False)
        b2.push(m)
        c = _move_cost(b2, tb, big)
        if c is None:
            continue
        costs.append(c)
        if c < big:
            holds += 1
    if len(costs) < 2:
        return None
    costs = np.array(costs)
    best = costs.min()
    near_best = int((costs <= best + margin).sum())
    return dict(sharpness=float(1.0 - near_best / len(costs)),
                result_sharpness=float(1.0 - holds / len(costs)),
                cost_spread=float(costs.std()), best_cost=float(best),
                n_moves=len(costs), only_move=bool(near_best == 1))


def sample_winning_positions(rng, tb, n: int, kind: str) -> list:
    """Positions where the side to move wins (WDL=+2), from a chosen family."""
    gen = {"krvk": lambda: random_krvk(rng), "krrkbp": lambda: random_krrkbp(rng),
           "endgame": lambda: random_endgame_start(rng)}[kind]
    out, tries = [], 0
    while len(out) < n and tries < n * 80:
        tries += 1
        b = gen()
        if b is None or b.is_game_over():
            continue
        try:
            if tb.probe_wdl(b) == 2:       # side to move wins under perfect play
                out.append(b)
        except (KeyError, chess.syzygy.MissingTableError):
            continue
    return out


# ------------------------------------------------------- candidate signals
def _reach(fb, boards, z, device, near: bool):
    import torch
    packed = np.stack([encode_packed(b) for b in boards])
    meta = np.stack([encode_meta(b) for b in boards])
    planes = torch.from_numpy(feature_planes(packed, meta)).to(device)
    om = torch.from_numpy(np.tile(
        omega_ids(np.array([1800]), np.array([1800]), np.array([300.0]))[0],
        (len(boards), 1))).to(device)
    with torch.no_grad():
        if near:
            f = fb.embed_F_near(planes, om)
            return (f @ z).cpu().numpy()
        f = fb.embed_F(planes, om)
        return fb.score(f, z).cpu().numpy()


def signals_for_position(fb, board, zgoals, device) -> dict:
    """All candidate uncertainty signals available for this checkpoint."""
    import torch
    moves = list(board.legal_moves)
    if len(moves) < 2:
        return {}
    succ = [board.copy(stack=False) for _ in moves]
    for b2, m in zip(succ, moves):
        b2.push(m)
    z_far = torch.as_tensor(zgoals["MATE_W"], dtype=torch.float32, device=device)
    far = _reach(fb, succ, z_far, device, near=False)
    out = {"score_spread": float(np.std(far))}
    if getattr(fb, "two_horizon", False) and "MATE_W_NEAR" in zgoals:
        z_near = torch.as_tensor(zgoals["MATE_W_NEAR"], dtype=torch.float32, device=device)
        near = _reach(fb, succ, z_near, device, near=True)
        rho = spearmanr(far, near).statistic          # agreement of the two orderings
        out["head_disagreement"] = float(1.0 - (rho if np.isfinite(rho) else 0.0))
    if getattr(fb, "distributional", False):
        # dist_sigma (option B): entropy of THIS position's predicted
        # distance-to-mate distribution -- aleatoric uncertainty. Wide/bimodal
        # => volatile => should read as sharp. The signal the reframe wants.
        packed = encode_packed(board)[None]
        meta = encode_meta(board)[None]
        planes = torch.from_numpy(feature_planes(packed, meta)).to(device)
        om = torch.from_numpy(omega_ids(np.array([1800]), np.array([1800]),
                                        np.array([300.0]))).to(device)
        with torch.no_grad():
            f_s = fb.embed_F(planes, om)
            out["dist_sigma"] = float(fb.dist_entropy(f_s, z_far)[0])
    return out


# ---------------------------------------------------------------- driver
def run(fb, zgoals, device, tb, n_per_kind: int, kinds: list, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    rows = []
    for kind in kinds:
        boards = sample_winning_positions(rng, tb, n_per_kind, kind)
        for b in boards:
            gt = ground_truth_sharpness(b, tb)
            if gt is None:
                continue
            sig = signals_for_position(fb, b, zgoals, device) if fb is not None else {}
            rows.append(dict(kind=kind, fen=b.fen(), **gt, **sig))
    if not rows:
        return dict(error="no scorable positions")

    sharp = np.array([r["sharpness"] for r in rows])
    result = dict(n=len(rows), mean_sharpness=float(sharp.mean()),
                  frac_only_move=float(np.mean([r["only_move"] for r in rows])),
                  by_kind={k: int(sum(r["kind"] == k for r in rows)) for k in kinds})
    signal_names = sorted({k for r in rows for k in r
                           if k in ("score_spread", "head_disagreement", "dist_sigma")})
    result["signal_rho_vs_sharpness"] = {}
    for name in signal_names:
        pairs = [(r[name], r["sharpness"]) for r in rows if name in r]
        if len(pairs) >= 10:
            xs, ys = zip(*pairs)
            rho = spearmanr(xs, ys).statistic
            result["signal_rho_vs_sharpness"][name] = dict(
                rho=float(rho) if np.isfinite(rho) else None, n=len(pairs),
                note="rho>0 wanted: signal should rise with true sharpness")
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=None, help="checkpoint to score signals for; "
                    "omit to only characterize the ground-truth sharpness distribution")
    ap.add_argument("--syzygy-dir", default="data/syzygy")
    ap.add_argument("--kinds", nargs="+", default=["krvk", "krrkbp", "endgame"])
    ap.add_argument("--n-per-kind", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    tb = chess.syzygy.open_tablebase(args.syzygy_dir)
    fb, zgoals = None, {}
    if args.ckpt:
        import torch  # noqa: F401
        from catspace.nn.fb import load_ckpt, pick_device
        device = pick_device(args.device)
        fb, payload = load_ckpt(Path(args.ckpt) if args.ckpt else derived_dir() / "lichess_fb.pt",
                                device)
        zgoals = {k: v.cpu().numpy() for k, v in payload.get("zgoals", {}).items()}
    else:
        device = "cpu"

    print(f"ckpt={args.ckpt} two_horizon={getattr(fb, 'two_horizon', False)} "
          f"kinds={args.kinds} n_per_kind={args.n_per_kind}", flush=True)
    result = run(fb, zgoals, device, tb, args.n_per_kind, args.kinds, args.seed)
    tb.close()
    print(json.dumps(result, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
