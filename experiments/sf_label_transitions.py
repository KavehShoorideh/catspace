#!/usr/bin/env python
"""experiments/sf_label_transitions.py -- M2a: add OBJECTIVE crossing labels to the transition data.
For each position (fen) + the ACTUAL move played, Stockfish (near-perfect value oracle) gives the
white-POV committor c(s) and c(s.move); the mover-POV LOSS = how much the move moved c AGAINST the
mover = the realized self-blunder magnitude (a basin crossing when large). This is the gold label
T is trained to predict from phi + context. Parallel SF over fen chunks (batched tensor-op spirit;
SF is the bottleneck, run many engines).
"""
from __future__ import annotations

import argparse, os, shutil, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def worker(task):
    fens, moves, depth = task
    import chess, chess.engine
    eng = chess.engine.SimpleEngine.popen_uci(shutil.which("stockfish") or "/opt/homebrew/bin/stockfish")
    try:
        eng.configure({"UCI_ShowWDL": True})
    except Exception:
        pass

    def committor(board):                                    # white-POV P(win)
        info = eng.analyse(board, chess.engine.Limit(depth=depth))
        w = info["wdl"].white(); tot = max(1, w.wins + w.draws + w.losses)
        return w.wins / tot

    cb, ca, loss = [], [], []
    for fen, mv in zip(fens, moves):
        try:
            b = chess.Board(fen); white_to_move = (b.turn == chess.WHITE)
            c0 = committor(b)
            b.push(chess.Move.from_uci(mv)); c1 = committor(b)
            # mover-POV loss = c moved against the mover
            lm = (c0 - c1) if white_to_move else (c1 - c0)
            cb.append(c0); ca.append(c1); loss.append(max(0.0, lm))
        except Exception:
            cb.append(np.nan); ca.append(np.nan); loss.append(np.nan)
    eng.quit()
    return cb, ca, loss


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/derived/transition_data.npz")
    ap.add_argument("--out", default="")
    ap.add_argument("--depth", type=int, default=12); ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="0 = all positions")
    args = ap.parse_args()
    t0 = time.time()
    z = dict(np.load(args.data, allow_pickle=True))
    fens = z["fen"]; moves = z["move"]
    if args.limit:
        idx = np.arange(min(args.limit, len(fens)))
        for k in z:
            z[k] = z[k][idx]
        fens, moves = z["fen"], z["move"]
    W = args.workers or max(1, (os.cpu_count() or 4) - 1)
    print(f"[sf-label] {len(fens)} positions | SF depth {args.depth} x {W} workers", flush=True)
    chunks_f = np.array_split(fens, W); chunks_m = np.array_split(moves, W)
    cb = np.empty(len(fens)); ca = np.empty(len(fens)); loss = np.empty(len(fens))
    with ProcessPoolExecutor(max_workers=W) as ex:
        results = list(ex.map(worker, [(list(f), list(m), args.depth) for f, m in zip(chunks_f, chunks_m)]))
    k = 0
    for r in results:
        L = len(r[0]); cb[k:k+L], ca[k:k+L], loss[k:k+L] = r; k += L
    z["committor_before"] = cb; z["committor_after"] = ca; z["mover_loss"] = loss
    out = args.out or args.data.replace(".npz", "_labeled.npz")
    np.savez_compressed(out, **z)
    ok = ~np.isnan(loss)
    print(f"\n=== {out}: {ok.sum()} labeled [{time.time()-t0:.0f}s] ===")
    for thr in (0.10, 0.20, 0.30):
        print(f"  realized crossing rate (mover_loss >= {thr}): {np.mean(loss[ok] >= thr):.1%}")
    print(f"  mean mover_loss {np.nanmean(loss):.3f} | committor_before mean {np.nanmean(cb):.3f}")
    print("DONE sf_label_transitions", flush=True)


if __name__ == "__main__":
    main()
