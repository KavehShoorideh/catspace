#!/usr/bin/env python
"""experiments/mcts_convert.py -- S2c (METASTABILITY_PLAN, Kaveh): demonstrate we can CONVERT
within a basin -- take a WON tablebase position and win it to the end via Monte Carlo Tree
Search that descends the field's value toward mate while AVOIDING surfaces with an interface to
the DRAW basin (the ∞-barrier makes draw/loss leaves value -BIG, so MCTS naturally steers off
them). White = MCTS on the learned field; Black = tablebase-optimal defense (hardest).

Value (White's perspective) V_white:
  checkmate & Black to move  -> +BIG  (White delivered mate)
  checkmate & White to move  -> -BIG  (White got mated)
  draw / stalemate           -> -BIG  (threw the win = draw-basin interface, AVOID)
  else                       -> -d_to_mate(node)   (closer to mate = better)
Negamax MCTS: White maximizes V_white, Black minimizes it. Reports conversion (mate) rate vs
sims, next to greedy (5%) and depth-3 minimax (17.5%).
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from catspace.research.tools.chess_specific.chessdata.encode import encode_meta, encode_packed
from catspace.research.components.encoder.approaches.jepa_tokenizer.src.fb import pick_device
from catspace.research.components.planner.approaches.endgame_groundtruth.src.tb import TB, DEFAULT_SYZYGY, rollout_dtm, tb_best_move
from experiments.arch_bakeoff import tokens
from experiments.mate_with_search import load
from experiments.value_fixed_point import white_pov_value

BIG = 1e4


class Node:
    __slots__ = ("board", "turn_white", "children", "untried", "N", "W", "terminal_v")
    def __init__(self, board):
        self.board = board
        self.turn_white = board.turn == chess.WHITE
        self.children = {}
        self.untried = None
        self.N = 0; self.W = 0.0
        self.terminal_v = None


SCALE = 40.0                                                 # committor length-scale in plies


@torch.no_grad()
def field_v(boards, net, dev):
    """Leaf EXPECTED-SCORE estimate (Kaveh: MCTS maximizes expected score, incl. the LOSS term).
    General form spanning [0,1]:
        score = 0.5 + 0.5*exp(-d_win/scale) - 0.5*exp(-d_loss/scale)
    -> ~1.0 near White-mate (win), 0.5 at the win/draw INTERFACE, -> 0.0 near White-getting-mated.
    The field currently has only d_win (distance-to-White-mate); loss-distance d_loss = inf for a
    won toy, so the loss term is 0 and this reduces to [0.5,1]. When the two-region WDL field
    (defect-2) adds d_loss, it plugs in here unchanged and the loss term activates."""
    pk = np.stack([encode_packed(b) for b in boards]); mt = np.stack([encode_meta(b) for b in boards])
    ids, stm = tokens(pk, mt)
    d_win = net.d_to_mate(torch.from_numpy(ids.astype(np.int64)).to(dev),
                          torch.from_numpy(stm.astype(np.int64)).to(dev)).cpu().numpy()
    d_loss = np.full_like(d_win, np.inf)                     # TODO: two-region field -> real d_loss
    return 0.5 + 0.5 * np.exp(-d_win / SCALE) - 0.5 * np.exp(-d_loss / SCALE)


def terminal_value(board):
    if board.is_checkmate():
        return 1.0 if board.turn == chess.BLACK else 0.0     # Black mated => White wins (score 1)
    if board.is_game_over(claim_draw=True):
        return 0.5                                           # draw score
    return None


def mcts_move(root_board, sims, net, dev, c=1.4):          # values now in [0,1] -> standard UCT c
    root = Node(root_board.copy(stack=False))
    for _ in range(sims):
        node = root; path = [node]
        # ---- selection ----
        while True:
            if node.terminal_v is not None:
                break
            if node.untried is None:
                node.terminal_v = terminal_value(node.board)
                if node.terminal_v is not None:
                    break
                node.untried = list(node.board.legal_moves)
            if node.untried:                                 # expand one
                m = node.untried.pop()
                nb = node.board.copy(stack=False); nb.push(m)
                child = Node(nb); node.children[m] = child
                path.append(child); node = child
                break
            # fully expanded -> UCB (negamax: White max V_white, Black min)
            lnN = math.log(node.N + 1)
            best, bm = None, None
            for m, ch in node.children.items():
                q = ch.W / ch.N if ch.N else 0.0
                u = c * math.sqrt(lnN / (ch.N + 1))
                score = (q + u) if node.turn_white else (-q + u)
                if best is None or score > best:
                    best, bm = score, ch
            path.append(bm); node = bm
        # ---- eval leaf ----
        v = node.terminal_v if node.terminal_v is not None else float(field_v([node.board], net, dev)[0])
        # ---- backup ----
        for n in path:
            n.N += 1; n.W += v
    # pick root child best for White (highest mean V_white)
    best, bm = None, None
    for m, ch in root.children.items():
        q = ch.W / ch.N if ch.N else -1e18
        if best is None or q > best:
            best, bm = q, m
    return bm


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="artifacts/experiments/mate_field_v1.pt")
    ap.add_argument("--sims", type=int, nargs="*", default=[64, 256, 800])
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--classes", nargs="*", default=["KQvK", "KRvK", "KRRvK"])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    t0 = time.time(); dev = pick_device(args.device); rng = np.random.default_rng(args.seed)
    net, ck = load(args.ckpt, dev)
    tb = TB(str(DEFAULT_SYZYGY), cache_db=None)
    print(f"[mcts-convert] greedy 5% / minimax-d3 17.5% baselines", flush=True)

    starts = []
    while len(starts) < args.n:
        cls = args.classes[rng.integers(0, len(args.classes))]
        from experiments.gen_dtm_data import random_class_start
        b = random_class_start(rng, cls)
        if b is None or b.turn != chess.WHITE or white_pov_value(b, tb) != 1.0:
            continue
        opt = rollout_dtm(b, tb)
        if opt and opt >= 1:
            starts.append((b.fen(), opt))

    for sims in args.sims:
        mated = drew = 0
        for fen, opt in starts:
            b = chess.Board(fen); cap = 3 * opt + 10; ok = False
            while len(b.move_stack) < cap:
                if b.is_checkmate(): ok = True; break
                if b.is_game_over(claim_draw=True): break
                if b.turn == chess.WHITE:
                    m = mcts_move(b, sims, net, dev)
                    if m is None: break
                    b.push(m)
                else:
                    m = tb_best_move(b, tb, set())
                    if m is None: break
                    b.push(m)
            if ok: mated += 1
            elif b.is_game_over(claim_draw=True) and not b.is_checkmate(): drew += 1
        print(f"  sims {sims}: CONVERT {100*mated/len(starts):.1f}% ({mated}/{len(starts)}) "
              f"| drew {drew} | [{time.time()-t0:.0f}s]", flush=True)
    tb.close()
    print("DONE mcts_convert", flush=True)


if __name__ == "__main__":
    main()
