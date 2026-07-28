"""catspace/transition.py -- realized CROSSING RISK: the expected OBJECTIVE committor swing under a
move-distribution. ONE primitive, both sides of the exploit (Kaveh 2026-07-27):
  * feed the OPPONENT's (Elo,z) move-model  -> their exploitable crossing risk (T-with-z; the M2 signal
    M3 turns into subgoals: "where is THIS opponent likely to cross a basin?"), and
  * feed OUR OWN move-model                 -> our blunder risk (the self-model / denial term: "where are
    WE likely to slip?" -- the defensive half of the planner).

The OBJECTIVE committor is adjudicated by the REFEREE = STOCKFISH (Kaveh: our near-perfect oracle,
consistent with the M0 basins and the M2a SF committor labels -- NOT our own trunk's WDL, which is a
strong opinion but not the ground-truth arbiter). committor_mover(s) = SF WDL win-fraction from the
MOVER's POV; after the mover plays m the turn flips but we keep the SAME mover's POV, so a move's
self-inflicted committor loss (a step toward crossing) is max(0, committor_mover(s) -
committor_mover(s.m)); the crossing RISK at s under a move model is the p_model-weighted sum over the
move's likely candidates. Weaker players cross more -> the exploitable edge.

(A fast play-time approximation of the committor -- a distilled head or the trunk WDL -- can be swapped
in for search, but it must be VALIDATED against this SF referee; the referee itself is SF.)
"""
from __future__ import annotations

import shutil

import chess
import chess.engine
import numpy as np


class CrossingRisk:
    def __init__(self, sf_path: str | None = None, depth: int = 12):
        self._path = sf_path or shutil.which("stockfish") or "/opt/homebrew/bin/stockfish"
        self.depth = depth
        self._open()

    def _open(self):
        self.eng = chess.engine.SimpleEngine.popen_uci(self._path)
        try:
            self.eng.configure({"UCI_ShowWDL": True})
        except Exception:
            pass

    def _analyse(self, fen):
        """Robust SF analyse on a CLEAN chess.Board(fen); restart the engine once on a protocol error
        (Stockfish occasionally desyncs under rapid reuse). Returns info or None."""
        b = chess.Board(fen)
        for attempt in (0, 1):
            try:
                return self.eng.analyse(b, chess.engine.Limit(depth=self.depth))
            except Exception:                              # EngineError OR async IllegalMoveError etc.
                try:
                    self.eng.quit()
                except Exception:
                    pass
                self._open()
        return None

    def committor(self, board, pov=None):
        """SF-refereed win-fraction from `pov` (default the side to move). (win_frac|None, pov_color)."""
        pov = board.turn if pov is None else pov
        info = self._analyse(board.fen())
        if info is None or "wdl" not in info:
            return None, pov
        w = info["wdl"].pov(pov); tot = max(1, w.wins + w.draws + w.losses)
        return w.wins / tot, pov

    def risk(self, board, move_probs, topk: int = 16):
        """Expected crossing risk at `board` under move-distribution `move_probs` {uci: prob}: SF-refereed.
        Returns (risk, mover_winfrac). risk = sum_m p(m) * max(0, committor_mover(s) - committor_mover(s.m))."""
        items = sorted(move_probs.items(), key=lambda kv: kv[1], reverse=True)[:topk]
        if not items:
            return 0.0, self.committor(board)[0]
        cw_s, mover = self.committor(board)                    # mover's objective win-frac at s
        if cw_s is None:
            return 0.0, 0.0
        swings, ps = [], []
        for uci, pr in items:
            c = board.copy()
            try:
                c.push(chess.Move.from_uci(uci))
            except Exception:                              # move illegal in this position -> skip
                continue
            cw_child, _ = self.committor(c, pov=mover)          # SAME mover's POV after the turn flips
            if cw_child is None:
                continue
            swings.append(max(0.0, cw_s - cw_child)); ps.append(pr)
        if not ps:
            return 0.0, cw_s
        p = np.asarray(ps, float); p /= p.sum()
        return float((p * np.asarray(swings)).sum()), cw_s

    def close(self):
        try:
            self.eng.quit()
        except Exception:
            pass


def _demo():
    """Validation: under a WEAKER move-model the expected SF-refereed crossing risk is HIGHER (weaker
    players blunder more) -- the exploitable signal, on live Maia move-distributions."""
    from lczerolens import LczeroBoard
    from catspace.train.scaffold import resolve_device
    from maia2 import model as maia_model, inference
    dev = resolve_device("auto")
    prepared = inference.prepare(); maia = maia_model.from_pretrained(type="rapid", device=str(dev))
    cr = CrossingRisk(depth=12)
    positions = {
        "Italian (sharp)": "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/2N2N2/PPPP1PPP/R1BQK2R w KQkq - 4 4",
        "tactical (open)": "r2q1rk1/ppp2ppp/2n1bn2/2bpp3/4P3/1PNP1N2/PBP2PPP/R2QKB1R w KQ - 0 8",
    }
    print(f"  referee = Stockfish (depth 12)")
    print(f"  {'position':<18} {'Maia-1100 risk':>16} {'Maia-1900 risk':>16}  (weaker should be higher)")
    for name, fen in positions.items():
        b = LczeroBoard(fen); row = []
        for elo in (1100, 1900):
            mp, _ = inference.inference_each(maia, prepared, fen, elo, elo)
            r, cw = cr.risk(b, mp)
            row.append(r)
        flag = "OK" if row[0] > row[1] else "??"
        print(f"  {name:<18} {row[0]:>16.4f} {row[1]:>16.4f}  {flag}")
    cr.close()
    print("DONE crossing demo")


if __name__ == "__main__":
    _demo()
