#!/usr/bin/env python
"""catspace/research/components/planner/approaches/endgame_groundtruth/experiments/gen_stratified_perfect.py -- the PERFECT-PLAY STRATIFIED dataset for the
bottom-up curriculum (Kaveh 2026-07-20). Strata boundary = PIECE COUNT: only a capture changes
it (promotion is count-preserving), so the game is a strict DAG over piece-count strata and any
position is <= k captures from a fully-solved tablebase stratum. Labels are under PERFECT play
(optimal attacker AND optimal defender -- the tablebase's exact V*/DTM), so targets are exact and
reproducible, opponent stochasticity lives in the (separate) L3 playability model, and the failing
data (draws/losses) is GENUINE (lost against perfect play from lost starts), not manufactured.

Local Syzygy frontier = the KRRvKBP endgame and its capture-descendants (3-6 pieces):
  6p KRRvKBP ; 5p KRRvKB KRRvKP KRvKBP ; 4p KRRvK KRvKB KRvKP KBP-v-K ; 3p KRvK KP-v-K KB-v-K.
One stratum ABOVE the frontier (the extrapolation test): 7p KRRvKBPP (defender +pawn) and
KRRRvKBP (attacker +rook), each one capture from the solved root; labeled by NEGAMAX that bottoms
out in the tablebase at <= 6 pieces (grounded where lines convert within the depth cap).

CHECKPOINTING (Kaveh's rule): every chunk writes its own shard immediately to <out>.shards/, so a
kill never loses completed work and a rerun RESUMES (skips shards that exist). --merge-only just
assembles existing shards into the final .npz.

Emits (final .npz):
  positions:  packed/meta, sdtm (signed White-POV plies-to-mate; +White/-Black/0 draw), wdl
              (White-POV {+1,0,-1}), pcount, matid, grounded (7p: bottomed in TB)
  edges:      e_p*/e_c* + drop (child has FEWER pieces = a capture = a stratum boundary)
  pairs:      a*/b* + gap  (optimal-line pairs, EXACT perfect-play ply-gap, for DTM/mate anchoring)
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import chess
import numpy as np


from catspace.research.tools.chess_specific.chessdata.encode import encode_meta, encode_packed
from catspace.research.components.memory.approaches.experience_store.experiments import selfplay_generate as sg
from catspace.research.components.memory.approaches.experience_store.experiments.selfplay_generate import random_endgame_start
from catspace.research.components.planner.approaches.committor_value.experiments.value_fixed_point import TB, tb_best_move, white_pov_value
from catspace.io import paths

W, B, K, R, Bp, N, P, Q = (chess.WHITE, chess.BLACK, chess.KING, chess.ROOK,
                           chess.BISHOP, chess.KNIGHT, chess.PAWN, chess.QUEEN)

STRATA_MENUS = {
    "krrkbp": [(R, W), (R, W), (Bp, B), (P, B)],            # 6p root
    "krrkb":  [(R, W), (R, W), (Bp, B)],                    # 5p
    "krrkp":  [(R, W), (R, W), (P, B)],
    "krkbp":  [(R, W), (Bp, B), (P, B)],
    "krrk":   [(R, W), (R, W)],                             # 4p
    "krkb":   [(R, W), (Bp, B)],
    "krkp":   [(R, W), (P, B)],
    "kbpk":   [(Bp, B), (P, B)],                            # White bare K vs K+B+P (Black-favoured)
    "krk":    [(R, W)],                                     # 3p
    "kpk":    [(P, B)],                                     # White bare K vs K+P (draw/Black-win)
    "kbk":    [(Bp, B)],                                    # White bare K vs K+B (draw -- yields 0 rows)
}
ABOVE_FRONTIER = {                                          # 7p: one capture above the frontier
    "krrkbpp": [(R, W), (R, W), (Bp, B), (P, B), (P, B)],
    "krrrkbp": [(R, W), (R, W), (R, W), (Bp, B), (P, B)],
}
ALL_MENUS = {**STRATA_MENUS, **ABOVE_FRONTIER}
MATID = {name: i for i, name in enumerate(ALL_MENUS)}

SHORT_KEYS = ["pk", "mt", "sdtm", "wdl", "pcnt", "matid", "grnd",
              "ep", "em", "cp", "cm", "drop", "ap", "am", "bp", "bm", "gap"]
SHORT2LONG = {"pk": "packed", "mt": "meta", "sdtm": "sdtm", "wdl": "wdl", "pcnt": "pcount",
              "matid": "matid", "grnd": "grounded", "ep": "e_p_packed", "em": "e_p_meta",
              "cp": "e_c_packed", "cm": "e_c_meta", "drop": "e_drop", "ap": "a_packed",
              "am": "a_meta", "bp": "b_packed", "bm": "b_meta", "gap": "gap"}


def pcount(board: chess.Board) -> int:
    return len(board.piece_map())


def optimal_line(board: chess.Board, tb, cap: int = 250):
    """Perfect play (tb-optimal BOTH sides) to absorption. Returns (boards, winner) with
    winner in {+1 White, -1 Black, 0 draw}."""
    b = board.copy(stack=False)
    seen = set()
    boards = [b.copy(stack=False)]
    for _ in range(cap):
        if b.is_checkmate():
            return boards, (1 if b.turn == chess.BLACK else -1)
        if b.is_game_over(claim_draw=True):
            return boards, 0
        m = tb_best_move(b, tb, seen)
        if m is None:
            return boards, 0
        if b.turn == chess.BLACK:
            seen.add(b.board_fen())
        b.push(m)
        boards.append(b.copy(stack=False))
    return boards, 0


def negamax_tb(board: chess.Board, tb, depth: int, alpha: float, beta: float, budget: list):
    """Side-to-move-POV value in {-1,0,+1} with the tablebase as the leaf oracle at <= 6 pieces.
    Returns (value, grounded); grounded=True iff backed by a TB probe / real terminal."""
    budget[0] -= 1
    if board.is_checkmate():
        return -1.0, True
    if board.is_stalemate() or board.is_insufficient_material() or board.can_claim_draw():
        return 0.0, True
    if pcount(board) <= 6:
        v = white_pov_value(board, tb)
        if v is not None:
            stm = v if board.turn == chess.WHITE else (1.0 - v)
            return (2.0 * stm - 1.0), True
    if depth == 0 or budget[0] <= 0:
        return 0.0, False
    best, grounded = -2.0, False
    for m in board.legal_moves:
        c = board.copy(stack=False); c.push(m)
        v, g = negamax_tb(c, tb, depth - 1, -beta, -alpha, budget)
        v = -v
        if v > best:
            best, grounded = v, g
        alpha = max(alpha, v)
        if alpha >= beta or budget[0] <= 0:
            break
    return (best if best > -2.0 else 0.0), grounded


def _emit_edges(board, mat_now, edge_cap, rng):
    caps, noncaps = [], []
    for m in board.legal_moves:
        c = board.copy(stack=False); c.push(m)
        drop = pcount(c) < mat_now
        (caps if drop else noncaps).append((c, drop))
    if len(noncaps) > edge_cap:
        idx = rng.choice(len(noncaps), edge_cap, replace=False)
        noncaps = [noncaps[i] for i in idx]
    return caps + noncaps


def gen_chunk(task):
    (name, n, seed, syzygy_dir, edge_cap, pairs_per, is_7p, nm_depth, nm_budget) = task
    sg._ENDGAME_MENUS.update(ALL_MENUS)
    tb = TB(syzygy_dir)
    rng = np.random.default_rng(seed)
    mid = MATID[name]
    pk, mt, sdtm, wdl, pcnt, grnd = [], [], [], [], [], []
    ep, em, cp, cm, drop = [], [], [], [], []
    ap, am, bp, bm, gap = [], [], [], [], []
    got = tries = 0
    while got < n and tries < n * 400:
        tries += 1
        b = random_endgame_start(rng, name)
        if b is None:
            continue
        pc = pcount(b)
        if not is_7p:
            line, winner = optimal_line(b, tb)
            if winner == 0:
                sd, wd = 0.0, 0
            else:
                dtm = len(line) - 1
                sd, wd = float(winner * dtm), int(winner)
                if len(line) >= 3:
                    for _ in range(pairs_per):
                        i = int(rng.integers(0, len(line) - 1))
                        j = int(rng.integers(i + 1, len(line)))
                        ap.append(encode_packed(line[i])); am.append(encode_meta(line[i]))
                        bp.append(encode_packed(line[j])); bm.append(encode_meta(line[j]))
                        gap.append(float(j - i))
            g = True
        else:
            v, g = negamax_tb(b, tb, nm_depth, -1e9, 1e9, [nm_budget])
            stm_wdl = 1 if v > 0.5 else (-1 if v < -0.5 else 0)
            wd = stm_wdl if b.turn == chess.WHITE else -stm_wdl
            sd = float("nan")
        pk.append(encode_packed(b)); mt.append(encode_meta(b))
        sdtm.append(sd); wdl.append(int(wd)); pcnt.append(pc); grnd.append(bool(g))
        for c, dr in _emit_edges(b, pc, edge_cap, rng):
            ep.append(encode_packed(b)); em.append(encode_meta(b))
            cp.append(encode_packed(c)); cm.append(encode_meta(c)); drop.append(dr)
        got += 1
    tb.close()

    def stk(x, dt): return (np.stack(x).astype(dt) if x else np.zeros((0,), dt))
    return dict(
        name=name, got=got,
        pk=stk(pk, np.uint64), mt=stk(mt, np.uint8),
        sdtm=np.array(sdtm, np.float32), wdl=np.array(wdl, np.int8),
        pcnt=np.array(pcnt, np.int8), matid=np.full(got, mid, np.int16),
        grnd=np.array(grnd, bool),
        ep=stk(ep, np.uint64), em=stk(em, np.uint8),
        cp=stk(cp, np.uint64), cm=stk(cm, np.uint8), drop=np.array(drop, bool),
        ap=stk(ap, np.uint64), am=stk(am, np.uint8),
        bp=stk(bp, np.uint64), bm=stk(bm, np.uint8), gap=np.array(gap, np.float32),
    )


def _shard_path(shard_dir, name, seed):
    return shard_dir / f"{name}_{seed}.npz"


def _save_shard(path, r):
    tmp = Path(str(path) + ".tmp.npz")           # end in .npz so savez won't re-append
    np.savez_compressed(tmp, got=np.array([r["got"]]), name=np.array([r["name"]], dtype=object),
                        **{k: r[k] for k in SHORT_KEYS})
    os.replace(tmp, path)


def _merge(shard_dir, out, t0):
    shards = sorted(shard_dir.glob("*.npz"))
    agg = {k: [] for k in SHORT_KEYS}
    for s in shards:
        z = np.load(s, allow_pickle=True)
        for k in SHORT_KEYS:
            if len(z[k]):
                agg[k].append(z[k])
    for k in SHORT_KEYS:
        agg[k] = np.concatenate(agg[k], axis=0) if agg[k] else np.zeros((0,), np.float32)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, material_names=np.array(list(ALL_MENUS), dtype=object),
                        **{SHORT2LONG[k]: agg[k] for k in SHORT_KEYS})
    npos, nedge, npair, ndrop = len(agg["pk"]), len(agg["ep"]), len(agg["ap"]), int(agg["drop"].sum())
    print(f"[merge] {len(shards)} shards -> {npos} positions, {nedge} edges ({ndrop} captures), "
          f"{npair} pairs  ({time.time()-t0:.0f}s)")
    for pc in sorted(set(agg["pcnt"].tolist()), reverse=True):
        m = agg["pcnt"] == pc
        w, d, l = int((agg["wdl"][m] == 1).sum()), int((agg["wdl"][m] == 0).sum()), int((agg["wdl"][m] == -1).sum())
        print(f"    {pc}p: n={int(m.sum()):6d}  W={w} D={d} L={l}")
    print(f"VERDICT STRAT_DATA n={npos} edges={nedge} drops={ndrop} pairs={npair} "
          f"strata={sorted(set(agg['pcnt'].tolist()))} -> {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per", type=int, default=4000)
    ap.add_argument("--per-7p", type=int, default=2500)
    ap.add_argument("--edge-cap", type=int, default=6)
    ap.add_argument("--pairs-per", type=int, default=2)
    ap.add_argument("--nm-depth", type=int, default=6)
    ap.add_argument("--nm-budget", type=int, default=50000)
    ap.add_argument("--out", default=paths.derived("stratified_perfect.npz"))
    ap.add_argument("--shard-dir", default=None, help="default = <out>.shards/")
    ap.add_argument("--merge-only", action="store_true", help="just assemble existing shards")
    ap.add_argument("--syzygy-dir", default=str(paths.syzygy_dir()))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--only", default=None, help="comma list of material names (smoke)")
    args = ap.parse_args()

    shard_dir = Path(args.shard_dir) if args.shard_dir else Path(str(args.out) + ".shards")
    shard_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    if args.merge_only:
        _merge(shard_dir, args.out, t0)
        return

    names = args.only.split(",") if args.only else list(ALL_MENUS)
    Wk = max(1, args.workers)
    tasks, skipped = [], 0
    for name in names:
        is7 = name in ABOVE_FRONTIER
        total = args.per_7p if is7 else args.per
        base, rem = divmod(total, Wk)
        for w in range(Wk):
            nn = base + (1 if w < rem else 0)
            if not nn:
                continue
            seed = args.seed + 1000 * MATID[name] + w
            if _shard_path(shard_dir, name, seed).exists():          # RESUME: skip done chunks
                skipped += 1
                continue
            tasks.append((name, nn, seed, args.syzygy_dir, args.edge_cap, args.pairs_per,
                          is7, args.nm_depth, args.nm_budget))
    print(f"[stage] stratified perfect-play gen: {len(names)} materials, {Wk} workers, "
          f"{len(tasks)} chunks to run ({skipped} already done -> shards in {shard_dir})", flush=True)

    done = 0
    if Wk == 1:
        for t in tasks:
            r = gen_chunk(t); _save_shard(_shard_path(shard_dir, t[0], t[2]), r); done += r["got"]
            print(f"  {t[0]:8s} +{r['got']:5d}  total {done}  ({time.time()-t0:.0f}s)", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=Wk) as ex:
            futs = {ex.submit(gen_chunk, t): t for t in tasks}
            for fut in as_completed(futs):
                t = futs[fut]; r = fut.result()
                _save_shard(_shard_path(shard_dir, t[0], t[2]), r)     # CHECKPOINT per chunk
                done += r["got"]
                print(f"  {r['name']:8s} +{r['got']:5d}  total {done}  "
                      f"({done/max(time.time()-t0,1):.0f} pos/s, {time.time()-t0:.0f}s)", flush=True)

    _merge(shard_dir, args.out, t0)


if __name__ == "__main__":
    main()
