#!/usr/bin/env python
"""
experiments/rim_staircase.py — WHY is the field flat near mate? (Kaveh
2026-07-16: "is there a path from mate-in-4 to mate-in-3? what do those
regions look like? why is the field flat there?")

Three measurements on KRRvk (pure rook endgame, tablebase = exact truth):
 1. PROGRESS-MOVE RANKING at a given position: rank all moves by the field's
    d_W; label each child PROGRESS (dtz strictly improves) or WASTE; report
    the best progress-move rank and the d_W gap.
 2. THE STAIRCASE: random legal KRRvk wins binned by DTZ; mean/std of d_W per
    bin + Spearman(d_W, dtz) per band -- where the learned distance stops
    ordering true distance.
 3. TARGET SATURATION: same positions, empirical P-hat via eps-noise rollouts
    (the training-data generator's own estimator) -- if P-hat is flat across
    the same band, the field is flat because its TARGET is flat (committor
    saturation near the goal), not because learning failed.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import chess
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from catspace.data.encode import encode_meta, encode_packed
from catspace.nn.features import feature_planes, omega_ids
from experiments.certainty_distill import spearman_ci
from experiments.value_fixed_point import TB, tb_best_move


def sample_krrk(n, tb, seed):
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        sq = rng.choice(64, size=4, replace=False)
        b = chess.Board(None)
        b.set_piece_at(int(sq[0]), chess.Piece(chess.KING, chess.WHITE))
        b.set_piece_at(int(sq[1]), chess.Piece(chess.KING, chess.BLACK))
        b.set_piece_at(int(sq[2]), chess.Piece(chess.ROOK, chess.WHITE))
        b.set_piece_at(int(sq[3]), chess.Piece(chess.ROOK, chess.WHITE))
        b.turn = chess.WHITE
        if not b.is_valid() or b.is_game_over():
            continue
        w, d = tb.wdl_dtz(b)
        if w == 2 and d is not None and d > 0:
            out.append((b.fen(), int(d)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="data/derived/sep/rootloop_r12.pt")
    ap.add_argument("--whead", default="data/derived/sep/rootloop_r12_whead.pt")
    ap.add_argument("--fen", default="8/8/8/4k3/8/K7/R7/R7 w - - 0 1")
    ap.add_argument("--n-staircase", type=int, default=400)
    ap.add_argument("--rollouts", type=int, default=24, help="per-state, for P-hat")
    ap.add_argument("--epsilon", type=float, default=0.15)
    ap.add_argument("--n-phat", type=int, default=120, help="staircase states to rollout")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    import torch
    from catspace.nn.fb import load_ckpt, pick_device

    dev = pick_device(args.device)
    fb, _ = load_ckpt(Path(args.ckpt), dev)
    fb.eval()
    hp = torch.load(args.whead, map_location=dev, weights_only=False)
    head = torch.nn.Sequential(torch.nn.Linear(hp["d_in"], 128), torch.nn.ReLU(),
                               torch.nn.Linear(128, 1), torch.nn.Softplus()).to(dev)
    head.load_state_dict(hp["state"]); head.eval()
    om_row = omega_ids(np.array([1800]), np.array([1800]), np.array([np.nan]))[0]

    def d_of(boards):
        with torch.no_grad():
            packed = np.stack([encode_packed(b) for b in boards])
            meta = np.stack([encode_meta(b) for b in boards])
            pl = torch.from_numpy(feature_planes(packed, meta)).to(dev)
            om = torch.from_numpy(np.tile(om_row, (len(boards), 1))).to(dev)
            return head(fb.embed_F(pl, om)).squeeze(-1).cpu().numpy()

    tb = TB("data/syzygy")

    # ---- 1. progress-move ranking at the probe position
    b0 = chess.Board(args.fen)
    _, dtz0 = tb.wdl_dtz(b0)
    moves = list(b0.legal_moves)
    kids = []
    for m in moves:
        b = b0.copy(stack=False); b.push(m)
        kids.append(b)
    d = d_of(kids)
    labels = []
    for b in kids:
        _, dz = tb.wdl_dtz(b)
        labels.append(("PROGRESS" if dz is not None and abs(dz) < dtz0 else "waste",
                       dz))
    order = np.argsort(d)
    best_prog_rank = next(i + 1 for i, j in enumerate(order) if labels[j][0] == "PROGRESS")
    prog_d = [d[i] for i in range(len(moves)) if labels[i][0] == "PROGRESS"]
    waste_d = [d[i] for i in range(len(moves)) if labels[i][0] == "waste"]
    print(f"position {args.fen} (DTZ {dtz0})")
    print(f"VERDICT PROGRESS_RANK best progress move ranks #{best_prog_rank}/{len(moves)}; "
          f"mean d_W progress {np.mean(prog_d):.4f} vs waste {np.mean(waste_d):.4f} "
          f"(gap {np.mean(waste_d) - np.mean(prog_d):+.4f}; positive = field prefers progress)")
    for rank, j in enumerate(order[:6], 1):
        print(f"  #{rank} {b0.san(moves[j]):>6} d_W={d[j]:.4f} {labels[j][0]} dtz={labels[j][1]}")

    # ---- 2. the staircase
    states = sample_krrk(args.n_staircase, tb, args.seed)
    ds = d_of([chess.Board(f) for f, _ in states])
    dtzs = np.array([z for _, z in states], dtype=float)
    print("\nSTAIRCASE  d_W (learned) per true DTZ bin:")
    for lo, hi in [(1, 2), (3, 4), (5, 6), (7, 8), (9, 12), (13, 30)]:
        m = (dtzs >= lo) & (dtzs <= hi)
        if m.sum() >= 5:
            print(f"  dtz {lo:>2}-{hi:<2}: n={int(m.sum()):>3}  d_W {ds[m].mean():.4f} "
                  f"+- {ds[m].std():.4f}")
    for lo, hi, tag in [(1, 6, "NEAR (dtz 1-6)"), (7, 30, "FAR (dtz 7-30)"),
                        (1, 30, "ALL")]:
        m = (dtzs >= lo) & (dtzs <= hi)
        r, l, h = spearman_ci(ds[m], dtzs[m])
        print(f"VERDICT STAIRCASE_{tag.split()[0]} Spearman(d_W, dtz) = {r:+.3f} "
              f"CI[{l:+.3f},{h:+.3f}] (n={int(m.sum())}) [{tag}]")

    # ---- 3. target saturation: empirical P-hat vs dtz under the generator's eps
    rng = np.random.default_rng(args.seed + 1)
    pick = rng.choice(len(states), size=min(args.n_phat, len(states)), replace=False)
    phats, pz = [], []
    for i in pick:
        fen, z = states[int(i)]
        wins = 0
        for r in range(args.rollouts):
            g = np.random.default_rng([args.seed, int(i), r])
            b = chess.Board(fen)
            seen = set()
            for _ in range(100):
                if b.is_game_over(claim_draw=True):
                    break
                if b.turn == chess.WHITE:
                    if g.random() < args.epsilon:
                        ms = list(b.legal_moves)
                        b.push(ms[int(g.integers(len(ms)))])
                        continue
                    m = tb_best_move(b, tb, seen)
                else:
                    m = tb_best_move(b, tb, seen)
                    seen.add(b.board_fen())
                b.push(m)
            out = b.outcome(claim_draw=True)
            wins += int(bool(out and out.winner == chess.WHITE))
        phats.append(wins / args.rollouts)
        pz.append(z)
    phats, pz = np.array(phats), np.array(pz, dtype=float)
    print(f"\nTARGET SATURATION  empirical P-hat (eps={args.epsilon}, tb-White) per DTZ bin:")
    for lo, hi in [(1, 2), (3, 4), (5, 6), (7, 8), (9, 12), (13, 30)]:
        m = (pz >= lo) & (pz <= hi)
        if m.sum() >= 5:
            print(f"  dtz {lo:>2}-{hi:<2}: n={int(m.sum()):>3}  P-hat {phats[m].mean():.3f} "
                  f"+- {phats[m].std():.3f}")
    for lo, hi, tag in [(1, 6, "NEAR"), (7, 30, "FAR")]:
        m = (pz >= lo) & (pz <= hi)
        if m.sum() >= 10:
            r, l, h = spearman_ci(-phats[m], pz[m])
            print(f"VERDICT TARGET_{tag} Spearman(-P-hat, dtz) = {r:+.3f} "
                  f"CI[{l:+.3f},{h:+.3f}] (n={int(m.sum())})")
    tb.close()


if __name__ == "__main__":
    main()
