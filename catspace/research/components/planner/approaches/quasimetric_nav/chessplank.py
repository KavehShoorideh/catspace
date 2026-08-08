#!/usr/bin/env python
"""chessplank.py -- THE most basic quasimetric-navigation engine (Kaveh 2026-08-08).

Move choice is THREAT-FIRST navigation on the outcome field, exactly as specified:
  'it's about holding back the draw and holding back the loss and going towards the win.'

For every legal move, embed the child and read d(child -> WIN/DRAW/LOSS poles), POV-flipped
(the child's mover is the opponent: our win is their loss). With d_bad = min(d_draw, d_loss):

  primary   maximise  d_bad - d_win      (push the nearest threat out AND pull the win in;
                                          a move that delays the draw but delays our win MORE
                                          worsens the margin and is rejected)
  tie-break maximise  d_bad              (among margin-equal moves, buy time vs the threat)
  <=5 pieces: Syzygy lookup outright     (the hybrid oracle -- no field consulted)

No search. One field readout per legal move. This is deliberately the dumbest possible planner:
its match results are the honest floor of what the geometry alone currently buys.
"""
from __future__ import annotations

import argparse
import random

import chess
import numpy as np
import torch

from catspace.io import paths
from catspace.research.components.encoder.approaches.jepa_tokenizer.src.jepa import tokenize
from catspace.research.components.encoder.approaches.reach_probability.experiments.interpret_reach import (
    load_net)
from catspace.research.components.planner.approaches.endgame_groundtruth.src.tb import TB


class ChessPlank:
    def __init__(self, ckpt, device="mps", cond_elo=None, use_tb=True):
        self.net, pay = load_net(ckpt, device)
        self.cfg = pay["cfg"]
        self.device = device
        self.cond_elo = cond_elo
        pn = self.cfg["pole_names"]
        self.poles = self.net.poles.poles.detach().float()
        self.pi = {n: pn.index(n) for n in ("WIN", "DRAW", "LOSS")}
        self.tb = TB() if use_tb else None
        if getattr(self.net, "split_head", False):
            self.dist = self.net.dB
        elif getattr(self.net, "dual", False):
            self.dist = self.net.qhead.d_base
        else:
            self.dist = self.net.iqe

    def _embed(self, toks, globs):
        tok_t = torch.from_numpy(np.array(toks).astype(np.int64)).to(self.device)
        glob_t = torch.from_numpy(np.array(globs).astype(np.float32)).to(self.device)
        if self.cond_elo is not None and getattr(self.net, "dual", False):
            cval = (self.cond_elo - 1500.0) / 500.0
            cond = torch.full((len(toks), self.net.qhead.proj_delta.in_features
                               - self.net.qhead.proj_base.in_features), cval, device=self.device)
            return self.net.encode_dual(tok_t, glob_t, cond)[1]
        return self.net.encode_q(tok_t, glob_t)

    def _tb_move(self, board):
        best, key = None, None
        for mv in board.legal_moves:
            board.push(mv)
            try:
                w, dz = self.tb.wdl_dtz(board)
            except Exception:
                w = None
            board.pop()
            if w is None:
                return None                      # any unprobeable child -> fall back to field
            # opponent POV: lower w better for us; among our wins prefer fast (small |dtz|),
            # among our losses prefer slow
            k = (w, (abs(dz) if w < 0 else -abs(dz)) if dz is not None else 0)
            if key is None or k < key:
                key, best = k, mv
        return best

    def choose(self, board):
        moves = list(board.legal_moves)
        if not moves:
            return None
        if self.tb is not None and len(board.piece_map()) <= 5:
            mv = self._tb_move(board)
            if mv is not None:
                return mv
        toks, globs = [], []
        for mv in moves:
            board.push(mv)
            tk, gl = tokenize(board)
            toks.append(tk); globs.append(gl)
            board.pop()
        with torch.no_grad():
            z = self._embed(toks, globs)
            D = {n: self.dist(z, self.poles[[k]].expand(len(z), -1).to(self.device))
                 .float().cpu().numpy() for n, k in self.pi.items()}
        # POV flip at the child: our win = child-mover's LOSS
        d_win, d_draw, d_loss = D["LOSS"], D["DRAW"], D["WIN"]
        d_bad = np.minimum(d_draw, d_loss)
        margin = d_bad - d_win
        order = np.lexsort((-d_bad, -margin))    # primary margin desc, then d_bad desc
        return moves[int(order[0])]


def play(engine, opponent, engine_white, max_plies=300, start_fen=None):
    b = chess.Board(start_fen) if start_fen else chess.Board()
    while not b.is_game_over(claim_draw=True) and b.ply() < max_plies:
        mv = engine.choose(b) if b.turn == engine_white else opponent(b)
        if mv is None:
            break
        b.push(mv)
    out = b.outcome(claim_draw=True)
    if out is None or out.winner is None:
        return 0.5
    return 1.0 if out.winner == engine_white else 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--opponent", default="random", choices=["random"])
    ap.add_argument("--cond-elo", type=float, default=None)
    ap.add_argument("--start", default="balanced", choices=["balanced", "piece-up"],
                    help="piece-up: opponent missing a knight -- the conversion test")
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    eng = ChessPlank(args.ckpt, args.device, args.cond_elo)
    rng = random.Random(0)

    def rand_opp(b):
        ms = list(b.legal_moves)
        return rng.choice(ms) if ms else None

    def start_fen(white):
        if args.start == "balanced":
            return None
        b = chess.Board()
        squares = [sq for sq, pc in b.piece_map().items()
                   if pc.piece_type == chess.KNIGHT and pc.color != white]
        b.remove_piece_at(rng.choice(squares))
        return b.fen()

    score, n = 0.0, 0
    for g in range(args.games):
        white = g % 2 == 0
        score += play(eng, rand_opp, white, start_fen=start_fen(white))
        n += 1
        if n % 10 == 0:
            print(f"[plank] {n}/{args.games}  score {score/n:.2f}", flush=True)
    print(f"[plank] FINAL vs {args.opponent} ({args.start}): {score/n:.3f} over {n} games "
          f"(0.5 = parity; random-vs-random ~0.5)")


if __name__ == "__main__":
    main()
