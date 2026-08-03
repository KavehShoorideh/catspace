#!/usr/bin/env python
"""catspace/research/components/search/approaches/puct_mcts/experiments/mcts_field.py -- LAYER 2: wire the ClockField quasimetric into the existing MCTS
(catspace/nn/mcts.py). The low-level search that finds concrete moves following the DISTANCE-TO-MATE
gradient (Kaveh's preference over raw value), with the TABLEBASE HANDOFF at <=7 pieces via the MCTS
certainty_fn hook and mate_stop for immediate mates.

value_fn(board) -> white-POV value in [-1,1]:
    base   = 2*committor - 1                         (navigation: which way + how much, defined everywhere)
    sharpen (winning side) by DISTANCE-TO-MATE: among winning nodes prefer LOWER d_mate (faster mate).
      v = base + w_dm * (2c-1 clipped to the winning side) * (1 - min(d_mate,Dcap)/Dcap)
certainty_fn(boards) -> (white_pov_value, confidence): at <=7 pieces return the EXACT tablebase WDL
    (2*wdl-1) with confidence 1.0 -> the search treats it as a soft-terminal (the handoff).
Real-history lc0 planes are rebuilt from board.move_stack (the field is history-trained).
"""
from __future__ import annotations

import sys
from pathlib import Path

import chess
import numpy as np
import torch

from catspace.research.components.encoder.approaches.reachability_field.experiments.train_clock_field import ClockField
from catspace.research.tools.training_infra.train.scaffold import resolve_device
from catspace.research.components.search.approaches.puct_mcts.src.mcts import MCTS
from catspace.io import paths


def board_to_planes(board):
    """Rebuild the lc0 112-plane REAL-history tensor for `board` from its move stack."""
    from lczerolens import LczeroBoard
    start = board.copy()
    moves = []
    while start.move_stack:
        moves.append(start.pop())
    lb = LczeroBoard(start.fen())
    for m in reversed(moves):
        lb.push(m)
    return lb.to_input_tensor().to(torch.float32).numpy()


class FieldMCTS:
    def __init__(self, ckpt, tb=None, device="auto", nodes=200, w_dm=0.4, dcap=40.0,
                 c_puct=1.5, mate_stop=True):
        self.dev = resolve_device(device); self.tb = tb; self.nodes = nodes
        self.w_dm = w_dm; self.dcap = dcap; self.c_puct = c_puct; self.mate_stop = mate_stop
        p = torch.load(ckpt, map_location=self.dev, weights_only=False)
        cfg = p.get("cfg", {"d": 64, "ch": 128, "blocks": 8, "in_planes": 112})
        self.net = ClockField(cfg["d"], ch=cfg["ch"], blocks=cfg["blocks"], in_planes=cfg.get("in_planes", 112)).to(self.dev)
        self.net.load_state_dict(p["state_dict"]); self.net.eval()

    def _value(self, boards):
        x = torch.from_numpy(np.stack([board_to_planes(b) for b in boards])).to(self.dev)
        with torch.no_grad():
            c = self.net.committor(x).cpu().numpy()          # P(white win)
            dm = self.net.d_mate(x).cpu().numpy()            # distance to WHITE mate
        base = 2 * c - 1
        win_side = np.clip(2 * c - 1, 0, 1)                  # >0 only when white better
        sharp = self.w_dm * win_side * (1.0 - np.minimum(dm, self.dcap) / self.dcap)
        return np.clip(base + sharp, -1, 1)

    def _certainty(self, boards):
        # tablebase handoff at <=7 pieces -> exact white-POV value, confidence 1.0
        if self.tb is None:
            return [(0.0, 0.0) for _ in boards]
        from catspace.research.components.planner.approaches.committor_value.experiments.value_fixed_point import white_pov_value
        out = []
        for b in boards:
            if chess.popcount(b.occupied) <= 7 and not b.is_game_over():
                try:
                    out.append((2 * white_pov_value(b, self.tb) - 1, 1.0)); continue
                except Exception:
                    pass
            out.append((0.0, 0.0))
        return out

    def _policy(self, board):
        ms = list(board.legal_moves)
        return {m: 1.0 / len(ms) for m in ms} if ms else {}

    def select(self, board):
        # HANDOFF: at <=7 pieces the tablebase plays the DTZ-optimal MOVE directly (not the field).
        if self.tb is not None and chess.popcount(board.occupied) <= 7 and not board.is_game_over():
            from catspace.research.components.planner.approaches.endgame_groundtruth.src.tb import tb_best_move
            try:
                mv = tb_best_move(board, self.tb, set())
                if mv is not None:
                    return mv
            except Exception:
                pass
        m = MCTS(lambda bs: np.zeros(len(bs)), max_nodes=self.nodes, c_puct=self.c_puct,
                 mate_stop=self.mate_stop, value_fn=self._value, policy_fn=self._policy,
                 certainty_fn=self._certainty, certainty_stop=0.9, batch_leaves=16, eval_cache={})
        root = m.run(board.copy(stack=True))
        if root is None or not root.children:
            return next(iter(board.legal_moves), None)
        best = max(root.children, key=lambda c: (c.N, (c.terminal_v if c.terminal_v is not None else c.Q)))
        return best.move


def _conversion_test(nodes=200, N=20):
    """Does SEARCH fix the greedy 0/30? d_mate-MCTS (white) converts won endgames vs TB-optimal defender."""
    from catspace.research.components.planner.approaches.endgame_groundtruth.src.tb import TB, DEFAULT_SYZYGY, tb_best_move
    from catspace.research.components.planner.approaches.endgame_groundtruth.experiments.gen_dtm_data import random_class_start
    tb = TB(str(DEFAULT_SYZYGY), cache_db=None); rng = np.random.default_rng(0)
    eng = FieldMCTS(paths.experiment("field_fullgame_v3_final.pt"), tb=tb, nodes=nodes)
    for cls in ["KQvK", "KRvK"]:
        mated = 0
        for _ in range(N):
            b = None
            for _ in range(50):
                b = random_class_start(rng, cls)
                if b and not b.is_game_over() and b.turn == chess.WHITE:
                    break
            if b is None:
                continue
            for _ in range(100):
                if b.is_game_over():
                    break
                mv = eng.select(b) if b.turn == chess.WHITE else tb_best_move(b, tb, set())
                if mv is None:
                    break
                b.push(mv)
            mated += int(b.is_checkmate())
        print(f"  {cls}: d_mate-MCTS({nodes}n) vs TB-optimal defender -> mated {mated}/{N}", flush=True)
    tb.close()


def _vs_maia(nodes=100, elo=1100, games=6, max_plies=160, seed=0):
    """FieldMCTS vs Maia, full games (real history, in-distribution), <=7p tablebase handoff."""
    import time, chess.engine
    from lczerolens import LczeroBoard
    from catspace.research.components.planner.approaches.endgame_groundtruth.src.tb import TB, DEFAULT_SYZYGY
    tb = TB(str(DEFAULT_SYZYGY), cache_db=None); rng = np.random.default_rng(seed)
    eng = FieldMCTS(paths.experiment("field_fullgame_v3_final.pt"), tb=tb, nodes=nodes)
    maia = chess.engine.SimpleEngine.popen_uci(["lc0", f"--weights=data/engines/maia/maia-{elo}.pb.gz", "--backend=eigen"])
    t0 = time.time(); W = D = L = 0
    for g in range(games):
        fw = (g % 2 == 0); board = LczeroBoard()
        for _ in range(4):
            ms = list(board.legal_moves); board.push(ms[rng.integers(0, len(ms))])
        while not board.is_game_over(claim_draw=True) and board.ply() < max_plies:
            if board.turn == (chess.WHITE if fw else chess.BLACK):
                mv = eng.select(board)
            else:
                mv = maia.play(board, chess.engine.Limit(nodes=1)).move
            if mv is None: break
            board.push(mv)
        res = board.result(claim_draw=True)
        if res == "1/2-1/2": D += 1
        elif (res == "1-0") == fw: W += 1
        else: L += 1
        print(f"  game {g+1}/{games} field={'W' if fw else 'B'} -> {res} | W{W} D{D} L{L} [{time.time()-t0:.0f}s]", flush=True)
    maia.quit(); tb.close()
    n = W + D + L
    print(f"VERDICT FieldMCTS({nodes}n) vs maia-{elo}: {W}W {D}D {L}L | SCORE {(W+0.5*D)/n:.3f} "
          f"(shallow-search baseline 0.125) [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="convert", choices=["convert", "maia"])
    ap.add_argument("--nodes", type=int, default=200); ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--elo", type=int, default=1100); ap.add_argument("--games", type=int, default=6)
    a = ap.parse_args()
    if a.mode == "convert":
        _conversion_test(nodes=a.nodes, N=a.n)
    else:
        _vs_maia(nodes=a.nodes, elo=a.elo, games=a.games)
