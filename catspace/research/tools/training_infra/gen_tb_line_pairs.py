#!/usr/bin/env python
"""gen_tb_line_pairs.py -- EXACT ply-distance pairs from tablebase-optimal lines.

rollout_line() plays a won position to mate with both sides tablebase-optimal. Because the
play is optimal, position i on the line is EXACTLY (j - i) plies from position j -- so every
ordered pair on a line is a ground-truth quasimetric label, not an estimate (the docstring
has said so since 2026-07-26; it was never wired into training). These are the tightest
possible ceilings for the QRL constraint: our corpus ceilings come from GAME paths, which
wander (measured tortuosity 0.31, so they overestimate ~3x).

CAVEAT recorded in the output: Syzygy is DTZ-optimal, not DTM-optimal, so a line is a correct
win that respects the 50-move rule but is not guaranteed the SHORTEST mate. These are exact
distances along an optimal-play trajectory, which is what a planner needs, not theoretical DTM.

    .venv/bin/python -m ...gen_tb_line_pairs --n 4000 --out data/derived/tb_line_pairs.npz
"""
from __future__ import annotations
import argparse, time
import chess, numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=4000, help="lines to roll out")
    ap.add_argument("--max-pairs-per-line", type=int, default=24)
    ap.add_argument("--out", default="data/derived/tb_line_pairs.npz")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from catspace.research.components.planner.approaches.endgame_groundtruth.src.tb import (
        TB, rollout_line)
    from catspace.research.components.encoder.approaches.jepa_tokenizer.src.jepa import tokenize
    from catspace.io import paths
    tb = TB(str(paths.syzygy_dir()))
    rng = np.random.default_rng(args.seed)
    PIECES = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT, chess.PAWN]

    def random_won_position():
        """random legal <=5-piece position that the tablebase says is WON for white."""
        for _ in range(200):
            b = chess.Board(None)
            sq = list(rng.choice(64, size=rng.integers(3, 6), replace=False))
            b.set_piece_at(int(sq[0]), chess.Piece(chess.KING, chess.WHITE))
            b.set_piece_at(int(sq[1]), chess.Piece(chess.KING, chess.BLACK))
            for s in sq[2:]:
                col = chess.WHITE if rng.random() < 0.8 else chess.BLACK
                pt = PIECES[int(rng.integers(0, len(PIECES)))]
                if pt == chess.PAWN and chess.square_rank(int(s)) in (0, 7):
                    pt = chess.ROOK
                b.set_piece_at(int(s), chess.Piece(pt, col))
            b.turn = chess.WHITE
            if not b.is_valid():
                continue
            try:
                w, _ = tb.wdl_dtz(b)
            except Exception:
                continue
            if w is not None and w > 0:
                return b
        return None

    toks, globs, ia, ib, gaps = [], [], [], [], []
    t0, nline = time.time(), 0
    while nline < args.n and time.time() - t0 < 3600:
        b = random_won_position()
        if b is None:
            continue
        line = rollout_line(b, tb, cap=140)
        if not line or len(line) < 6:
            continue
        nline += 1
        base = len(toks)
        for p in line:
            tk, gl = tokenize(p)
            toks.append(np.asarray(tk, np.uint8)); globs.append(np.asarray(gl, np.float32))
        L = len(line)
        for _ in range(min(args.max_pairs_per_line, L * (L - 1) // 2)):
            i = int(rng.integers(0, L - 1)); j = int(rng.integers(i + 1, L))
            ia.append(base + i); ib.append(base + j); gaps.append(j - i)
        if nline % 250 == 0:
            print(f"  {nline} lines, {len(toks):,} positions, {len(ia):,} pairs "
                  f"[{time.time()-t0:.0f}s]", flush=True)
    np.savez_compressed(args.out, tok=np.array(toks, np.uint8),
                        glob=np.array(globs, np.float32), ia=np.array(ia, np.int64),
                        ib=np.array(ib, np.int64), gap=np.array(gaps, np.float32),
                        note=np.array(["EXACT ply distances along tablebase-optimal lines. "
                                       "Syzygy is DTZ-optimal so lines are correct wins, not "
                                       "guaranteed shortest mates."]))
    print(f"[tb-pairs] {nline} lines | {len(toks):,} positions | {len(ia):,} exact pairs "
          f"-> {args.out} [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
