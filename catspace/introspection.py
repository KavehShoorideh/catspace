"""catspace/engine/introspection.py -- THE PLANNER'S SENSORIUM (Kaveh 2026-07-25: 'expose
the internals of our engine... what do we know about the position, how often have we seen
it, sharpness of the local field... available as stats that can be requested by the planner
so it knows where to spend its time').

Named probes over objects the engine already maintains -- banks, field geometry, rules
surfaces, experience store -- no new learning, no hand-coded concepts. `summary()` is the
flat observation dict; the RL PlanSelector consumes it later, and logging summaries at
plan-decision points collects its training observations from day one."""
from __future__ import annotations

import numpy as np
import chess


class ProbeKit:
    def __init__(self, fm, win_bank, loss_bank, draw_bank, exp_db=None,
                 game_ctx: dict | None = None, prior_fn=None):
        self.fm = fm
        self.banks = {"win": win_bank, "loss": loss_bank, "draw": draw_bank}
        self.exp_db = exp_db                    # sqlite connection or None
        self.game_ctx = game_ctx or {}
        self.prior_fn = prior_fn

    # -- memory: what do our banks know near here? --------------------------------
    def memory(self, b: chess.Board) -> dict:
        F = self.fm.embed_F_boards([b])
        out = {}
        for name, bk in self.banks.items():
            out[f"n_{name}"] = len(bk) if bk is not None else 0
            if bk is not None and len(bk) > 0:
                d = self.fm.d_to_bank(F, bk.embs)
                out[f"d_{name}"] = float(d[0])
            else:
                out[f"d_{name}"] = float("inf")
        wb = self.banks["win"]
        if wb is not None and hasattr(wb, "class_idx"):
            from catspace.research.components.planner.approaches.endgame_groundtruth.src.material import mat_sig
            out["class_density"] = int(len(wb.class_idx(mat_sig(b))))
        return out

    # -- familiarity: how often have we been here? --------------------------------
    def familiarity(self, b: chess.Board) -> dict:
        epd = b.epd()
        out = {"seen_in_game": int(self.game_ctx.get("hist", {}).get(epd, 0))}
        if self.exp_db is not None:
            try:
                out["seen_across_games"] = self.exp_db.execute(
                    "SELECT COUNT(*) FROM positions WHERE epd=?", (epd,)).fetchone()[0]
            except Exception:
                out["seen_across_games"] = -1
        return out

    # -- sharpness: does the local field discriminate? ----------------------------
    def sharpness(self, b: chess.Board) -> dict:
        wb = self.banks["win"]
        out = {}
        kids = []
        for m in b.legal_moves:
            c = b.copy(stack=False); c.push(m); kids.append(c)
        if wb is not None and len(wb) > 0 and kids:
            d = self.fm.d_boards_to_bank(kids, wb.embs)
            out["child_dwin_spread"] = float(d.max() - d.min())
            out["child_dwin_margin"] = float(np.median(d) - d.min())
        if self.prior_fn is not None:
            pri = np.array(list(self.prior_fn(b).values()))
            pri = pri[pri > 0]
            out["prior_entropy"] = float(-(pri * np.log(pri)).sum())
            out["prior_top1"] = float(pri.max())
        return out

    # -- surfaces: how close are the rules' absorbing surfaces? -------------------
    def surfaces(self, b: chess.Board) -> dict:
        h = self.game_ctx.get("hist", {})
        return {"clock": int(b.halfmove_clock),
                "clock_headroom": int(100 - b.halfmove_clock),
                "rep_max_nearby": int(max([h.get(b.epd(), 0)] or [0])),
                "n_pieces": len(b.piece_map())}

    def summary(self, b: chess.Board) -> dict:
        s = {}
        for part in (self.memory, self.familiarity, self.sharpness, self.surfaces):
            try:
                s.update(part(b))
            except Exception as e:
                s[f"err_{part.__name__}"] = str(e)[:60]
        return s
