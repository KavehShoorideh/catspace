"""catspace/transition.py -- realized CROSSING RISK: the expected objective committor swing under a
move-distribution. ONE primitive, both sides of the exploit (Kaveh 2026-07-27):
  * feed the OPPONENT's (Elo,z) move-model  -> their exploitable crossing risk (T-with-z; the M2 signal
    M3 turns into subgoals: "where is THIS opponent likely to cross a basin?"), and
  * feed OUR OWN move-model                 -> our blunder risk (the self-model / denial term: "where are
    WE likely to slip?" -- the defensive half of the planner).

The OBJECTIVE committor is the frozen trunk's own WDL head (a strong net, ~same forward that yields phi;
no per-move Stockfish). committor(s) = mover's win-prob = wdl(s)[win]; after the mover plays m the turn
flips, so the mover's win-prob at the child is wdl(s.m)[loss]. A move's self-inflicted committor loss
(a step toward crossing) is max(0, wdl(s)[win] - wdl(s.m)[loss]); the crossing RISK at s under a move
model is the p_model-weighted sum over the move's likely candidates. Weaker players cross more -> the
exploitable edge.
"""
from __future__ import annotations

import chess
import torch


class CrossingRisk:
    def __init__(self, field):
        self.field = field; self.trunk = field.trunk; self.dev = field.dev

    @torch.no_grad()
    def _wdl(self, boards):
        """(B,3) win/draw/loss for the SIDE TO MOVE, from the frozen trunk's WDL head."""
        x = torch.stack([b.to_input_tensor() for b in boards]).float().to(self.dev)
        return self.trunk(x)["wdl"]

    @torch.no_grad()
    def mover_winprob(self, board):
        return float(self._wdl([board])[0, 0])

    @torch.no_grad()
    def risk(self, board, move_probs, topk: int = 16):
        """Expected crossing risk at `board` under move-distribution `move_probs` {uci: prob}.
        Returns (risk, mover_winprob). risk = sum_m p(m) * max(0, committor(s) - committor(s.m)),
        the p-weighted objective committor loss the mover inflicts on themselves."""
        items = sorted(move_probs.items(), key=lambda kv: kv[1], reverse=True)[:topk]
        if not items:
            return 0.0, self.mover_winprob(board)
        cw_s = float(self._wdl([board])[0, 0])                 # mover's objective win-prob at s
        children = []
        for uci, _ in items:
            c = board.copy()
            c.push(chess.Move.from_uci(uci))
            children.append(c)
        mover_win_child = self._wdl(children)[:, 2]            # loss for new stm = mover's win at child
        swing = (cw_s - mover_win_child).clamp(min=0)          # (K,) self-inflicted committor loss
        p = torch.tensor([pr for _, pr in items], dtype=torch.float32, device=self.dev)
        p = p / p.sum()
        return float((p * swing).sum()), cw_s


def _demo():
    """Validation: under a WEAKER move-model the expected crossing risk is HIGHER (weaker players
    blunder more) -- the exploitable signal, computed on live Maia move-distributions."""
    from lczerolens import LczeroBoard
    from catspace.field import ReachabilityField
    from maia2 import model as maia_model, inference
    field = ReachabilityField(); cr = CrossingRisk(field)
    prepared = inference.prepare(); maia = maia_model.from_pretrained(type="rapid", device=str(field.dev))
    # sharp, double-edged middlegames where a weaker player can go wrong
    positions = {
        "Italian (sharp)": "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/2N2N2/PPPP1PPP/R1BQK2R w KQkq - 4 4",
        "tactical (open)": "r2q1rk1/ppp2ppp/2n1bn2/2bpp3/4P3/1PNP1N2/PBP2PPP/R2QKB1R w KQ - 0 8",
    }
    print(f"  {'position':<18} {'Maia-1100 risk':>16} {'Maia-1900 risk':>16}  (weaker should be higher)")
    for name, fen in positions.items():
        b = LczeroBoard(fen); row = []
        for elo in (1100, 1900):
            mp, _ = inference.inference_each(maia, prepared, fen, elo, elo)
            r, cw = cr.risk(b, mp)
            row.append(r)
        flag = "OK" if row[0] > row[1] else "??"
        print(f"  {name:<18} {row[0]:>16.4f} {row[1]:>16.4f}  {flag}")
    print("DONE crossing demo")


if __name__ == "__main__":
    _demo()
