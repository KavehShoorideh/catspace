#!/usr/bin/env python
"""catspace/research/components/search/approaches/puct_mcts/experiments/search_retrieval_combined.py -- the END-TO-END outcome mechanism the merged
IQE + capture-strata architecture actually specifies (Kaveh 2026-07-20): recursive minimax that
bottoms out in the TABLEBASE at the solved frontier, with a RETRIEVAL leaf (kNN over labeled L1
embeddings) for the quiet positions search doesn't reach -- NOT the material stand-in.

Validation (frontier artificially lowered to <= `frontier`): from 5-6p positions, recover the
TRUE full-<=6 tablebase outcome. Forcing/tactical lines reach the frontier (search-exact); quiet
lines use the retrieval leaf. We report the combined accuracy and, head-to-head, the same search
with the MATERIAL leaf -- so the retrieval leaf's contribution (esp. recovering draws) is isolated.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import chess
import numpy as np
import torch


from catspace.research.tools.chess_specific.chessdata.encode import board_from_packed, encode_meta, encode_packed
from catspace.research.components.encoder.approaches.jepa_tokenizer.src.features import feature_planes, omega_ids
from catspace.research.components.encoder.approaches.jepa_tokenizer.src.fb import load_ckpt, pick_device
from catspace.research.components.planner.approaches.committor_value.experiments.value_fixed_point import TB, white_pov_value
from catspace.io import paths

BOARD_ONLY = (18, 19)
PIECE_VAL = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}


def pcount(b):
    return len(b.piece_map())


def material_stm(board):
    w = sum(PIECE_VAL.get(p.piece_type, 0) for p in board.piece_map().values() if p.color == chess.WHITE)
    b = sum(PIECE_VAL.get(p.piece_type, 0) for p in board.piece_map().values() if p.color == chess.BLACK)
    return float(np.sign((w - b) if board.turn == chess.WHITE else (b - w)))


def quiescence(board, tb, frontier, alpha, beta, leaf_fn, stats):
    if board.is_checkmate():
        return -1.0
    if board.is_stalemate() or board.is_insufficient_material() or board.can_claim_draw():
        return 0.0
    if pcount(board) <= frontier:
        v = white_pov_value(board, tb)
        if v is not None:
            stats[0] += 1
            stm = v if board.turn == chess.WHITE else (1.0 - v)
            return 2.0 * stm - 1.0
    caps = [m for m in board.legal_moves if board.is_capture(m)]
    if not caps:
        stats[1] += 1
        return leaf_fn(board)
    best = leaf_fn(board)
    for m in caps:
        c = board.copy(stack=False); c.push(m)
        v = -quiescence(c, tb, frontier, -beta, -alpha, leaf_fn, stats)
        if v > best:
            best = v
        alpha = max(alpha, v)
        if alpha >= beta:
            break
    return best


def negamax(board, tb, frontier, depth, alpha, beta, leaf_fn, stats):
    if board.is_checkmate():
        return -1.0
    if board.is_stalemate() or board.is_insufficient_material() or board.can_claim_draw():
        return 0.0
    if pcount(board) <= frontier:
        v = white_pov_value(board, tb)
        if v is not None:
            stats[0] += 1
            stm = v if board.turn == chess.WHITE else (1.0 - v)
            return 2.0 * stm - 1.0
    if depth == 0:
        return quiescence(board, tb, frontier, alpha, beta, leaf_fn, stats)
    best = -2.0
    for m in sorted(board.legal_moves, key=lambda m: not board.is_capture(m)):
        c = board.copy(stack=False); c.push(m)
        v = -negamax(c, tb, frontier, depth - 1, -beta, -alpha, leaf_fn, stats)
        if v > best:
            best = v
        alpha = max(alpha, v)
        if alpha >= beta:
            break
    return best if best > -2.0 else 0.0


def wpov_wdl(stm_val, turn):
    wp = stm_val if turn == chess.WHITE else -stm_val
    return 1 if wp > 0.5 else (-1 if wp < -0.5 else 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--l1", default=paths.sep("iqe_stratified.pt"))
    ap.add_argument("--data", default=paths.derived("stratified_perfect.npz"))
    ap.add_argument("--syzygy", default=str(paths.syzygy_dir()))
    ap.add_argument("--frontier", type=int, default=4)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--n", type=int, default=120, help="positions per source stratum")
    ap.add_argument("--ref-n", type=int, default=6000, help="labeled reference set for the kNN leaf")
    ap.add_argument("--knn-k", type=int, default=15)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time()
    dev = pick_device(args.device)
    fb, _ = load_ckpt(Path(args.l1), dev); fb.eval()
    om = omega_ids(np.array([1800]), np.array([1800]), np.array([300.0]))[0]
    tb = TB(args.syzygy)
    rng = np.random.default_rng(args.seed)

    nz = np.load(args.data, allow_pickle=True)
    P, M, WDL, PCNT = (np.asarray(nz["packed"]), np.asarray(nz["meta"]),
                       np.asarray(nz["wdl"]), np.asarray(nz["pcount"]).astype(int))
    k6 = PCNT <= 6

    def embed_boards(pk, mt, bs=4096):
        out = []
        for i in range(0, len(pk), bs):
            pl = feature_planes(pk[i:i+bs], mt[i:i+bs]); pl[:, BOARD_ONLY] = 0.0
            o = torch.from_numpy(np.tile(om, (len(pl), 1))).to(dev)
            with torch.no_grad():
                e = fb.embed_F(torch.from_numpy(pl).to(dev), o).cpu().numpy()
            out.append(e)
        e = np.concatenate(out).astype(np.float32)
        return e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-8)            # cosine

    # reference vector DB (labeled <=6 embeddings)
    ref = np.flatnonzero(k6); ref = ref[rng.permutation(len(ref))[: args.ref_n]]
    E_ref = embed_boards(P[ref], M[ref]); WDL_ref = WDL[ref]
    print(f"[stage] reference DB: {len(ref)} labeled <=6 embeddings; retrieval leaf k={args.knn_k}", flush=True)

    cache = {}
    def retrieval_leaf(board):
        key = board._transposition_key()
        v = cache.get(key)
        if v is None:
            pk = encode_packed(board)[None]; mt = encode_meta(board)[None]
            q = embed_boards(pk, mt)[0]
            sims = E_ref @ q
            top = np.argpartition(-sims, args.knn_k)[: args.knn_k]
            wdl = int(np.sign(WDL_ref[top].sum()))                              # majority White-POV WDL
            v = float(wdl if board.turn == chess.WHITE else -wdl)               # -> stm-POV
            cache[key] = v
        return v

    # source positions strictly above the lowered frontier
    src = np.flatnonzero((PCNT > args.frontier) & k6)
    sel = np.concatenate([src[PCNT[src] == pc][rng.permutation((PCNT[src] == pc).sum())[: args.n]]
                          for pc in sorted(set(PCNT[src].tolist()))])
    print(f"[stage] {len(sel)} source positions (strata {sorted(set(PCNT[sel].tolist()))}), "
          f"frontier<= {args.frontier}p, depth {args.depth}+quiescence", flush=True)

    res = {"retrieval": [0, [0, 0]], "material": [0, [0, 0]]}
    perclass = {"retrieval": {1: [0, 0], 0: [0, 0], -1: [0, 0]}, "material": {1: [0, 0], 0: [0, 0], -1: [0, 0]}}
    n = 0
    for j in sel:
        b = board_from_packed(P[j], M[j])
        if b.is_game_over():
            continue
        n += 1
        true = int(WDL[j])
        for name, leaf in (("retrieval", retrieval_leaf), ("material", material_stm)):
            stats = [0, 0]
            v = negamax(b, tb, args.frontier, args.depth, -1e9, 1e9, leaf, stats)
            ok = int(wpov_wdl(v, b.turn) == true)
            res[name][0] += ok; res[name][1][0] += stats[0]; res[name][1][1] += stats[1]
            perclass[name][true][0] += ok; perclass[name][true][1] += 1
        if n % 40 == 0:
            print(f"  {n}/{len(sel)}  ({time.time()-t0:.0f}s, cache {len(cache)})", flush=True)
    tb.close()

    for name in ("retrieval", "material"):
        acc = res[name][0] / n; reach = res[name][1][0] / max(sum(res[name][1]), 1)
        pc = perclass[name]
        wr = pc[1][0]/max(pc[1][1],1); dr = pc[0][0]/max(pc[0][1],1); lr = pc[-1][0]/max(pc[-1][1],1)
        print(f"  {name:9s} leaf: WDL acc {acc:.3f}  frontier-reach {reach:.3f}  "
              f"(win-rec {wr:.2f} draw-rec {dr:.2f} loss-rec {lr:.2f})", flush=True)
    ra = res["retrieval"][0] / n; ma = res["material"][0] / n
    print(f"VERDICT SEARCH_RETRIEVAL n={n} frontier={args.frontier} depth={args.depth} "
          f"retrieval_acc={ra:.3f} material_acc={ma:.3f} lift={ra-ma:+.3f} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
