#!/usr/bin/env python
"""experiments/gen_contrast_mate_tuples.py -- Kaveh 2026-07-22: the matched-anchor
contrastive data ("random play with similar material counts versus human/machine play
towards the goal ... same material count ... separate distance-to-mate from piece count").

Per tuple, from ONE anchor position (a won toy position with known DTM):
  POS branch   Stockfish plays j plies toward mate (the clean machine ladder; tb/DTZ play
               hangs rooks -- JOURNAL 2026-07-22), then on to the actual MATE exemplar M.
               Verified: tb rollout-DTM at branch end DECREASED vs the anchor.
  NEG branch   j plies of RANDOM legal play (both sides) from the SAME anchor. Kaveh's
               filter, rules+tb only: discard if it delivered mate / won material for
               White / tb rollout-DTM at the end got CLOSER to mate. Kept = drifting.
Both branches share the anchor -> material, phase, king placement all matched; the only
difference is purposeful vs random play. The trainer's hinge d(F(pos_t),B(M)) + m <
d(F(neg_t),B(M)) can then only be satisfied by STRUCTURE-of-progress, not piece count.

Output npz: packed/meta for all states + index arrays (tuple_id, role pos/neg, depth t)
+ mate exemplar per tuple + anchor DTM.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import chess
import chess.engine
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from catspace.research.tools.chess_specific.chessdata.encode import board_from_packed, encode_meta, encode_packed
from experiments.gen_dtm_data import rollout_dtm
from experiments.value_fixed_point import TB


def material_count(b: chess.Board, color) -> int:
    vals = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
    return sum(vals.get(p.piece_type, 0) for p in b.piece_map().values() if p.color == color)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dtm-npz", default="data/derived/dtm_endgame.npz")
    ap.add_argument("--n-tuples", type=int, default=2000)
    ap.add_argument("--j", type=int, default=6, help="branch depth (plies)")
    ap.add_argument("--dtm-min", type=int, default=8, help="anchor DTM >= this (room to make progress)")
    ap.add_argument("--sf-depth", type=int, default=14)
    ap.add_argument("--engine", default="stockfish")
    ap.add_argument("--out", default="data/derived/contrast_mate_tuples.npz")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time(); rng = np.random.default_rng(args.seed)
    tb = TB("data/syzygy")
    eng = chess.engine.SimpleEngine.popen_uci(args.engine)
    eng.configure({"Threads": 1})

    dz = np.load(args.dtm_npz)
    P, M, dtm = np.asarray(dz["packed"]), np.asarray(dz["meta"]), np.asarray(dz["dtm"])
    cand = np.flatnonzero(dtm >= args.dtm_min)
    rng.shuffle(cand)

    PK, MT = [], []                       # the state store
    tid, role, depth = [], [], []         # per-state tuple id / +1 pos -1 neg 0 mate / ply depth
    anchor_dtm_l = []
    made = tried = 0

    def add(b, t_, r_, d_):
        PK.append(encode_packed(b)); MT.append(encode_meta(b))
        tid.append(t_); role.append(r_); depth.append(d_)

    for ci in cand:
        if made >= args.n_tuples:
            break
        tried += 1
        anchor = board_from_packed(P[ci], M[ci])
        if anchor.turn != chess.WHITE or anchor.is_game_over():
            continue
        # ---- POS: Stockfish j plies, then to mate (cap 60)
        b = anchor.copy(stack=False); pos_states = []
        ok = True
        for t in range(args.j):
            if b.is_game_over(claim_draw=True):
                ok = False; break
            info = eng.analyse(b, chess.engine.Limit(depth=args.sf_depth))
            if not info.get("pv"):
                ok = False; break
            b.push(info["pv"][0]); pos_states.append(b.copy(stack=False))
        if not ok or b.is_game_over(claim_draw=True) and not b.is_checkmate():
            continue
        end_dtm = 0 if b.is_checkmate() else rollout_dtm(b, tb)
        if end_dtm is None or end_dtm >= dtm[ci]:          # no verified progress -> drop
            continue
        mate_b = b.copy(stack=False)
        for _ in range(60):                                 # continue to the actual mate exemplar
            if mate_b.is_checkmate():
                break
            if mate_b.is_game_over(claim_draw=True):
                mate_b = None; break
            info = eng.analyse(mate_b, chess.engine.Limit(depth=args.sf_depth))
            if not info.get("pv"):
                mate_b = None; break
            mate_b.push(info["pv"][0])
        if mate_b is None or not mate_b.is_checkmate():
            continue
        # ---- NEG: random legal play, same anchor; Kaveh's filter (rules+tb only)
        neg_states = None
        for _try in range(6):
            nb = anchor.copy(stack=False); states = []
            dead = False
            for t in range(args.j):
                moves = list(nb.legal_moves)
                if not moves:
                    dead = True; break
                nb.push(moves[int(rng.integers(len(moves)))]); states.append(nb.copy(stack=False))
            if dead or nb.is_checkmate():                   # random stumbled into mate -> reject
                continue
            if material_count(nb, chess.WHITE) > material_count(anchor, chess.WHITE):
                continue                                    # "ended good" (won material) -> reject
            nd = rollout_dtm(nb, tb) if not nb.is_game_over(claim_draw=True) else None
            if nd is not None and nd < dtm[ci]:             # random got CLOSER to mate -> reject
                continue
            neg_states = states
            break
        if neg_states is None:
            continue
        # ---- record tuple
        add(anchor, made, 0, 0)                             # role 0 = anchor
        for t, s in enumerate(pos_states):
            add(s, made, +1, t + 1)
        for t, s in enumerate(neg_states):
            add(s, made, -1, t + 1)
        add(mate_b, made, 2, 99)                            # role 2 = mate exemplar
        anchor_dtm_l.append(int(dtm[ci]))
        made += 1
        if made % 100 == 0:
            print(f"  {made}/{args.n_tuples} tuples ({tried} tried)  [{time.time()-t0:.0f}s]", flush=True)

    eng.quit(); tb.close()
    np.savez_compressed(args.out, packed=np.stack(PK), meta=np.stack(MT),
                        tuple_id=np.array(tid, np.int32), role=np.array(role, np.int8),
                        depth=np.array(depth, np.int16), anchor_dtm=np.array(anchor_dtm_l, np.int32))
    print(f"VERDICT CONTRAST_TUPLES made={made} tried={tried} states={len(PK)} j={args.j} "
          f"dtm_min={args.dtm_min} -> {args.out}  [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
