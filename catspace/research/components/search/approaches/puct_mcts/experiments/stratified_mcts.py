#!/usr/bin/env python
"""catspace/research/components/search/approaches/puct_mcts/experiments/stratified_mcts.py -- the mechanism Kaveh specified precisely: ADVERSARIAL MCTS on
the REACHABILITY field to get from n pieces to n-1 (the first capture), and at n-1 ESTIMATE THE
OUTCOME BY kNN LOOKUP over the labeled n-1 embeddings (the vector DB).

The field does only what it's good at -- REACHABILITY (guiding the search toward a winning capture,
the n-1 boundary). The OUTCOME is a kNN retrieval at the boundary: at/below the frontier the
labels are tablebase-exact; above it they're the distilled estimates (same lookup either way).
White maximizes the boundary outcome it can FORCE; Black minimizes (adversarial negamax backup).
The backed-up root value is the search-refined signal that distills into the n-piece embeddings.

Validation on <= 6 (frontier lowered): recover the TRUE tablebase WDL from n=5,6 positions, and
CONVERT won ones vs optimal defense -- vs the greedy field engine that failed (0.35) / material
(0.55). Here the reference DB is tablebase-labeled, so kNN-at-boundary IS the retrieval leaf
(0.75/0.79 standalone); the question is whether adversarial MCTS + that leaf ties n->n-1.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import chess
import numpy as np
import torch


from catspace.research.tools.chess_specific.chessdata.encode import board_from_packed, encode_meta, encode_packed
from catspace.research.components.encoder.approaches.jepa_tokenizer.src.features import feature_planes, omega_ids
from catspace.research.components.encoder.approaches.jepa_tokenizer.src.fb import load_ckpt, pick_device
from catspace.research.components.planner.approaches.committor_value.experiments.value_fixed_point import TB, tb_best_move, white_pov_value
from catspace.io import paths

BOARD_ONLY = (18, 19)
VAL = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}


def pcount(b):
    return len(b.piece_map())


class StratifiedMCTS:
    def __init__(self, l1, data, syzygy, frontier, ref_n=6000, knn_k=15, goal_n=1500,
                 device="auto", seed=0):
        self.dev = pick_device(device)
        self.fb, _ = load_ckpt(Path(l1), self.dev); self.fb.eval()
        self.om = omega_ids(np.array([1800]), np.array([1800]), np.array([300.0]))[0]
        self.tb = TB(syzygy); self.frontier = frontier; self.knn_k = knn_k
        nz = np.load(data, allow_pickle=True)
        P, M, WDL, PCNT = (np.asarray(nz["packed"]), np.asarray(nz["meta"]),
                           np.asarray(nz["wdl"]), np.asarray(nz["pcount"]).astype(int))
        rng = np.random.default_rng(seed); k6 = PCNT <= 6
        # vector DB for the OUTCOME lookup at the boundary (F-side, labeled)
        ref = np.flatnonzero(k6); ref = ref[rng.permutation(len(ref))[:ref_n]]
        self.E_ref = torch.nn.functional.normalize(self._embF(P[ref], M[ref]), dim=1)
        self.WDL_ref = torch.from_numpy(WDL[ref]).float().to(self.dev)
        # reachability subgoal region (B-side): WON positions at/below the frontier
        wl = np.flatnonzero((WDL == 1) & (PCNT <= frontier)); wl = wl[rng.permutation(len(wl))[:goal_n]]
        self.B_goal = self._embB(P[wl], M[wl])
        self._rcache = {}; self._vcache = {}; self._rng = np.random.default_rng(seed + 1)

    def _planes(self, pk, mt):
        pl = feature_planes(pk, mt); pl[:, BOARD_ONLY] = 0.0
        return torch.from_numpy(pl).to(self.dev)

    def _embF(self, pk, mt):
        with torch.no_grad():
            o = torch.from_numpy(np.tile(self.om, (len(pk), 1))).to(self.dev)
            return self.fb.embed_F(self._planes(pk, mt), o)

    def _embB(self, pk, mt):
        with torch.no_grad():
            return self.fb.embed_B(self._planes(pk, mt))

    def retrieval_value(self, board):
        """kNN over the labeled vector DB -> (soft White-POV WDL in [-1,1], neighbor VARIANCE).
        The mean navigates the search toward higher-WDL regions; the variance is the uncertainty
        signal (high = neighbors disagree = keep searching one layer below)."""
        key = board._transposition_key()
        r = self._vcache.get(key)
        if r is None:
            q = torch.nn.functional.normalize(self._embF(encode_packed(board)[None], encode_meta(board)[None]), dim=1)[0]
            sims = self.E_ref @ q
            top = sims.topk(self.knn_k).indices
            nb = self.WDL_ref[top]
            r = (float(nb.mean()), float(nb.var()))
            self._vcache[key] = r
        return r

    def leaf_value(self, board):
        """(White-POV value in [-1,1], uncertainty). Terminal/tablebase EXACT; else the kNN value."""
        if board.is_checkmate():
            return (1.0 if board.turn == chess.BLACK else -1.0), 0.0
        if board.is_stalemate() or board.is_insufficient_material() or board.can_claim_draw():
            return 0.0, 0.0
        if pcount(board) <= self.frontier:                         # EXACT tablebase anchor at the frontier
            v = white_pov_value(board, self.tb)
            if v is not None:
                return 2.0 * v - 1.0, 0.0
        return self.retrieval_value(board)

    def _exact(self, board):
        return board.is_game_over(claim_draw=True) or pcount(board) <= self.frontier

    def rollout(self, board, temp=0.6, max_plies=18):
        """NOISY gradient rollout (Kaveh): bias toward captures + the mover's material gain (a cheap
        proxy for 'toward higher WDL'), with softmax RANDOMNESS so hidden tactics that go against
        the local gradient still get explored. Ends EXACT at the tablebase, else the kNN value."""
        b = board.copy(stack=False)
        for _ in range(max_plies):
            if self._exact(b):
                return self.leaf_value(b)[0]
            kids = [(m, (lambda c: (c.push(m), c)[1])(b.copy(stack=False))) for m in b.legal_moves]
            caps = [(m, c) for m, c in kids if b.is_capture(m)]
            pool = caps if caps else kids
            s = np.array([self._wm(c) for _, c in pool], float) * (1.0 if b.turn == chess.WHITE else -1.0)
            p = np.exp((s - s.max()) / temp); p /= p.sum()
            b = pool[int(self._rng.choice(len(pool), p=p))][1]     # sample -> exploration
        return self.retrieval_value(b)[0]

    @staticmethod
    def _wm(b):
        return sum(VAL.get(p.piece_type, 0) for p in b.piece_map().values() if p.color == chess.WHITE) \
            - sum(VAL.get(p.piece_type, 0) for p in b.piece_map().values() if p.color == chess.BLACK)

    def field_reach(self, boards):
        """field distance from each board to the won-simplification region (reachability)."""
        pk = np.stack([encode_packed(b) for b in boards]); mt = np.stack([encode_meta(b) for b in boards])
        with torch.no_grad():
            return self.fb.distance_matrix(self._embF(pk, mt), self.B_goal).min(1).values.cpu().numpy()

    def search(self, root, iters=250, c=1.4):
        N = {}; Wv = {}; ch = {}
        def key(b): return b.board_fen() + (" w" if b.turn else " b")
        def kids_of(b):
            k = key(b)
            if k not in ch:
                ch[k] = [(m, (lambda cc: (cc.push(m), cc)[1])(b.copy(stack=False))) for m in b.legal_moves]
                N[k] = 0; Wv[k] = 0.0
            return ch[k]
        rk = key(root)
        for _ in range(iters):
            b = root.copy(stack=False); path = []
            while True:
                if b.is_game_over(claim_draw=True):
                    val = self.leaf_value(b)[0]; break
                kids = kids_of(b); k = key(b); path.append(k)
                logN = math.log(N[k] + 1); best, bchild = -1e9, None
                for m, cchild in kids:
                    ck = key(cchild); n = N.get(ck, 0)
                    q = (Wv.get(ck, 0.0) / n) if n else 0.0
                    q = q if b.turn == chess.WHITE else -q
                    u = q + c * math.sqrt(logN / (1 + n))
                    if u > best:
                        best, bchild = u, cchild
                if N.get(key(bchild), 0) == 0:
                    path.append(key(bchild))
                    val = self.leaf_value(bchild)[0] if self._exact(bchild) else self.rollout(bchild)
                    break                                          # exact leaves as-is; else noisy rollout
                b = bchild
            for k in path:
                N[k] = N.get(k, 0) + 1; Wv[k] = Wv.get(k, 0.0) + val
        kids = kids_of(root)
        def cscore(cc):
            ck = key(cc); n = N.get(ck, 0); q = (Wv.get(ck, 0.0) / n) if n else 0.0
            return ((q if root.turn == chess.WHITE else -q), n)
        best_m = max(kids, key=lambda mc: cscore(mc[1]))[0]
        return Wv.get(rk, 0.0) / max(N.get(rk, 1), 1), best_m

    def move(self, board, iters=250):
        if pcount(board) <= self.frontier:
            return tb_best_move(board, self.tb)
        return self.search(board, iters=iters)[1]

    def close(self):
        self.tb.close()


def play(mcts, start, iters, ply_cap):
    b = start.copy(stack=False)
    for _ in range(ply_cap):
        if b.is_game_over(claim_draw=True):
            return b.is_checkmate() and b.turn == chess.BLACK
        m = mcts.move(b, iters=iters) if b.turn == chess.WHITE else tb_best_move(b, mcts.tb)
        if m is None:
            return False
        b.push(m)
    return False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--l1", default=paths.sep("iqe_stratified.pt"))
    ap.add_argument("--data", default=paths.derived("stratified_perfect.npz"))
    ap.add_argument("--syzygy", default=str(paths.syzygy_dir()))
    ap.add_argument("--frontier", type=int, default=5)
    ap.add_argument("--iters", type=int, default=250)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--mode", choices=["value", "convert"], default="convert")
    ap.add_argument("--ply-cap", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time()
    mcts = StratifiedMCTS(args.l1, args.data, args.syzygy, args.frontier, seed=args.seed)
    nz = np.load(args.data, allow_pickle=True)
    P, M, SDTM, WDL, PCNT = (np.asarray(nz["packed"]), np.asarray(nz["meta"]), np.asarray(nz["sdtm"]),
                             np.asarray(nz["wdl"]), np.asarray(nz["pcount"]).astype(int))
    rng = np.random.default_rng(args.seed)

    if args.mode == "value":
        cand = np.flatnonzero((PCNT > args.frontier) & (PCNT <= 6)); sel = cand[rng.permutation(len(cand))[: args.n]]
        ok = n = 0
        for j in sel:
            b = board_from_packed(P[j], M[j])
            if b.is_game_over():
                continue
            n += 1; v, _ = mcts.search(b, iters=args.iters)
            pred = 1 if v > 0.33 else (-1 if v < -0.33 else 0)
            ok += int(pred == int(WDL[j]))
            if n % 10 == 0:
                print(f"  {n}  wdl_rec {ok/n:.3f} ({time.time()-t0:.0f}s)", flush=True)
        print(f"VERDICT MCTS_VALUE frontier={args.frontier} iters={args.iters} n={n} "
              f"wdl_recovery={ok/n:.3f} ({time.time()-t0:.0f}s)")
    else:
        cand = np.flatnonzero((SDTM > 0) & (PCNT > args.frontier) & (PCNT <= 6)); starts = []
        for j in rng.permutation(cand):
            b = board_from_packed(P[j], M[j])
            if b.turn == chess.WHITE and not b.is_game_over():
                starts.append(b)
            if len(starts) >= args.n:
                break
        print(f"[stage] {len(starts)} won starts, frontier<= {args.frontier}p, MCTS iters {args.iters}, "
              f"vs optimal defense", flush=True)
        wins = 0
        for i, b in enumerate(starts):
            wins += play(mcts, b, args.iters, args.ply_cap)
            if (i + 1) % 5 == 0:
                print(f"  {i+1}/{len(starts)}  wins {wins}  ({time.time()-t0:.0f}s)", flush=True)
        print(f"VERDICT MCTS_CONVERT frontier={args.frontier} iters={args.iters} n={len(starts)} "
              f"convert={wins/len(starts):.3f} (greedy-field 0.35, material 0.55) ({time.time()-t0:.0f}s)")
    mcts.close()


if __name__ == "__main__":
    main()
