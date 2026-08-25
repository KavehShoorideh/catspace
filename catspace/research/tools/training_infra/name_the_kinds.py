"""What ARE the discovered kinds of future?

The kinds fall out of clustering; nothing names them. This asks what actually HAPPENS
in the lines assigned to each kind, using only python-chess over the stored PVs -- no
model, no GPU. Every statistic is a plain fact about the moves in the line, so a kind
that is real should show a coherent signature and one that is noise should not.

Reported per kind, as a delta against the all-kinds average, so the numbers say what is
DISTINCTIVE rather than what is common:
  queens off   fraction of lines where both queens leave the board
  material     net centipawn swing over the line, from the mover's POV
  captures     captures per ply
  checks       checks per ply
  endgame      pieces remaining at the end of the line
  value        Stockfish eval of the root move (what the kind is worth)
"""
import argparse
import glob
import os
import numpy as np
import chess
from catspace.research.tools.npz_io import load_npz

VAL = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
       chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}


def line_stats(board, moves):
    """Facts about one principal variation, all verifiable by hand."""
    mover = board.turn
    cap = chk = plies = 0
    for m in moves:
        if board.is_capture(m):
            cap += 1
        board.push(m)
        plies += 1
        if board.is_check():
            chk += 1
        if board.is_game_over():
            break
    def mat(b, colour):
        return sum(VAL[p.piece_type] for p in b.piece_map().values()
                   if p.color == colour)
    return {
        "queens_off": float(not board.pieces(chess.QUEEN, chess.WHITE)
                            or not board.pieces(chess.QUEEN, chess.BLACK)),
        "material": float(mat(board, mover) - mat(board, not mover)),
        "captures": cap / max(plies, 1),
        "checks": chk / max(plies, 1),
        "pieces_left": float(len(board.piece_map())),
        "plies": float(plies),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fkind", default="data/derived/gko1_fkind")
    ap.add_argument("--shards", default="data/derived/gko1_shards")
    ap.add_argument("--max_shards", type=int, default=6)
    ap.add_argument("--max_rows", type=int, default=400)
    a = ap.parse_args()

    K = int(np.load(os.path.join(a.fkind, "centroids.npz"))["K"])
    acc = {k: [] for k in range(K)}
    vals = {k: [] for k in range(K)}
    used = 0
    for f in sorted(glob.glob(os.path.join(a.fkind, "shard_*.npz"))):
        z = load_npz(f)
        if "fkind" not in z.files:
            continue
        d = load_npz(os.path.join(a.shards, os.path.basename(f)))
        fk, rmv, pv, pvn, val = z["fkind"], z["rmv"], z["pv"], z["pvn"], z["val"]
        for i in range(min(a.max_rows, len(fk))):
            if not (fk[i] >= 0).any():
                continue
            try:
                b0 = chess.Board(str(d["fen"][i]))
            except Exception:
                continue
            for j in range(fk.shape[1]):
                k = int(fk[i, j])
                if k < 0 or pvn[i, j] == 0:
                    continue
                mv = []
                f_, t_, pr = (int(x) for x in rmv[i, j])
                mv.append(chess.Move(f_, t_, promotion=pr or None))
                for s in range(int(pvn[i, j])):
                    f2, t2, p2 = (int(x) for x in pv[i, j, s])
                    mv.append(chess.Move(f2, t2, promotion=p2 or None))
                try:
                    acc[k].append(line_stats(b0.copy(), mv))
                except Exception:
                    continue
                vals[k].append(float(np.clip(val[i, j], -10, 10)))
        used += 1
        if used >= a.max_shards:
            break

    keys = ["queens_off", "material", "captures", "checks", "pieces_left", "plies"]
    n = {k: len(acc[k]) for k in range(K)}
    tot = sum(n.values())
    base = {q: np.mean([s[q] for k in range(K) for s in acc[k]]) for q in keys}
    print(f"\n  {tot:,} lines over {used} shards | overall: "
          + "  ".join(f"{q} {base[q]:.2f}" for q in keys))
    print(f"\n  kind    n     queens-off   material   captures   checks   pieces-left"
          f"   SF value")
    for k in range(K):
        if not acc[k]:
            continue
        m = {q: np.mean([s[q] for s in acc[k]]) for q in keys}
        print(f"  {k:<4} {n[k]:>6}   {m['queens_off']:+8.2f}   {m['material']:+7.2f}   "
              f"{m['captures']:+7.2f}   {m['checks']:+6.2f}   {m['pieces_left']:+9.2f}   "
              f"{np.mean(vals[k]):+7.2f}",
              end="")
        # the single most distinctive axis for this kind, in SDs from the mean
        z = {q: (m[q] - base[q]) /
                (np.std([s[q] for kk in range(K) for s in acc[kk]]) + 1e-9)
             for q in keys}
        top = max(z, key=lambda q: abs(z[q]))
        print(f"   <- {top} {z[top]:+.2f} sd")
    print("\n  (columns are absolute means; the tag is the axis on which the kind is "
          "furthest\n   from the all-kinds average, in standard deviations)")


if __name__ == "__main__":
    main()
