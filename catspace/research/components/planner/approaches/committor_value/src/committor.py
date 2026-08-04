"""CommittorGreedy -- committor-net value policy (1-ply/2-ply expectimax; the 0.125
baseline and the probe's value oracle). ValueOracle component; extracted from
catspace/approaches/gauntlet_harness/experiments/play_vs_maia.py 2026-07-30."""
from __future__ import annotations

import chess
import numpy as np
import torch

from catspace.research.components.planner.approaches.committor_value.src.clock_field import ClockField


class CommittorGreedy:
    """1-ply committor-greedy field player. Keeps an lczero board mirror for real-history planes."""
    def __init__(self, ckpt, device, tau=0.0, opp_tau=0.0):
        self.dev = device; self.tau = tau; self.opp_tau = opp_tau
        p = torch.load(ckpt, map_location=device, weights_only=False)
        cfg = p.get("cfg", {"d": 64, "ch": 128, "blocks": 8, "in_planes": 112})
        self.net = ClockField(cfg["d"], ch=cfg["ch"], blocks=cfg["blocks"], in_planes=cfg.get("in_planes", 112)).to(device)
        self.net.load_state_dict(p["state_dict"]); self.net.eval()
        from lczerolens import LczeroBoard
        self._LB = LczeroBoard

    def _committor(self, planes_list):
        if not planes_list:
            return np.zeros(0)
        x = torch.from_numpy(np.stack(planes_list)).to(self.dev)
        with torch.no_grad():
            return self.net.committor(x).cpu().numpy()       # P(white win)

    def _term_myval(self, board, my_white):
        r = board.result(claim_draw=True)
        return 1.0 if ((r == "1-0") == my_white) else (0.5 if r == "1/2-1/2" else 0.0)

    def select(self, lcboard, rng, depth=1):
        moves = list(lcboard.legal_moves)
        if not moves:
            return None, 0.5
        my_white = (lcboard.turn == chess.WHITE)
        if depth <= 1:                                       # 1-ply: committor of each successor
            planes, term = [], {}
            for i, m in enumerate(moves):
                lcboard.push(m)
                if lcboard.is_game_over(claim_draw=True):
                    term[i] = self._term_myval(lcboard, my_white)
                else:
                    term[i] = ("leaf", len(planes)); planes.append(lcboard.to_input_tensor().to(torch.float32).numpy())
                lcboard.pop()
            c = self._committor(planes)
            vals = np.array([t if not isinstance(t, tuple) else (c[t[1]] if my_white else 1 - c[t[1]]) for t in term.values()])
        else:                                                # 2-ply MINIMAX: worst-case over Maia replies
            leaves = []; move_reply = []
            for m in moves:
                lcboard.push(m)
                if lcboard.is_game_over(claim_draw=True):
                    move_reply.append([("term", self._term_myval(lcboard, my_white))]); lcboard.pop(); continue
                rr = []
                for r_ in lcboard.legal_moves:
                    lcboard.push(r_)
                    if lcboard.is_game_over(claim_draw=True):
                        rr.append(("term", self._term_myval(lcboard, my_white)))
                    else:
                        rr.append(("leaf", len(leaves))); leaves.append(lcboard.to_input_tensor().to(torch.float32).numpy())
                    lcboard.pop()
                move_reply.append(rr); lcboard.pop()
            c = self._committor(leaves)
            def leafval(t): return t[1] if t[0] == "term" else (c[t[1]] if my_white else 1 - c[t[1]])
            def agg(rr):
                if not rr: return 0.5
                v = np.array([leafval(t) for t in rr])
                if self.opp_tau <= 0:                        # paranoid MINIMAX (opponent finds refutation)
                    return float(v.min())
                # EXPECTIMAX vs a FALLIBLE opponent: they prefer low-my-value replies with softmax temp
                w = np.exp(-(v - v.min()) / self.opp_tau); w /= w.sum()
                return float((w * v).sum())
            vals = np.array([agg(rr) for rr in move_reply])
        if self.tau > 0:
            p = np.exp((vals - vals.max()) / self.tau); p /= p.sum(); i = rng.choice(len(moves), p=p)
        else:
            i = int(np.argmax(vals))
        return moves[i], float(vals[i])
