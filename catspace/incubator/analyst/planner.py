"""planner.py -- THE PLANNING LAYER (Kaveh 2026-09-11): "a future board embedding distribution predicted from the
current state, quantized and interpretable; it must predict the sharpness and the value of what is coming, not the
details; and a signal from the tactics to the planner and back".

From the current interaction graph the knowledge network predicts a DISTRIBUTION over named regime codes ten and
twenty plies ahead (regimes.py: each code is a cell of a quantizer on the graph's own indicators, described by
its centroid). Read off the distribution: the expected VALUE (each code's mean result and the value head), the
expected SHARPNESS (the mass on codes with a big threat or a mate threat), and the uncertainty (its entropy).

The two signals. Tactics -> planner: the current ledgers, zones and threats are inputs of the prediction, and the
sharp codes are defined by them. Planner -> tactics: (1) a STEERING TARGET -- the reachable future code whose
value beats the expected one by the most -- is turned into NEEDS by comparing its centroid with the present (queens
off -> remove their queen; a passed pawn of ours -> the lever that makes one, or the race; a king under attack ->
the mate goal; material up -> win the dearest target), and the needs planner and the goal search take it from
there; (2) a SEARCH BUDGET from the expected sharpness: deep where tactics are coming, shallow where they are not.

    python -m catspace.incubator.analyst.planner --fen "<fen>" [--ckpt data/derived/checkpoints/kn_future/best.pt]
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import chess
import numpy as np
import torch
import torch.nn.functional as F

from .memo import Memo
from .graph import build
from .regimes import Codebook, CODE_KEYS, BIG
from . import knowledge_net as K
from .experts import net_sample

DEFAULT_CKPT = "data/derived/checkpoints/kn_future/best.pt"


@dataclass
class Future:
    code: int
    p: float
    value: float
    text: str
    sharp: bool


@dataclass
class Steer:
    code: int
    text: str
    value: float
    gain: float                          # over the expected value
    p: float                             # how likely it is already
    needs: list                          # (kind, square/piece, text)


@dataclass
class Plan:
    code_now: int
    now_text: str
    futures10: list[Future]
    futures20: list[Future]
    value_now: float                     # the mover's result rate in the present code
    value10: float                       # the value head at +10 (tanh cp)
    value20: float
    expected10: float                    # expected result from the distribution at +10
    expected20: float
    sharp10: float                       # mass on sharp codes
    sharp20: float
    entropy10: float
    sharp_window: float = 0.0            # a big threat within 10 plies (the sharpness head), else the code mass
    steer: list[Steer] = field(default_factory=list)
    budget: dict = field(default_factory=dict)
    lines: list[str] = field(default_factory=list)


class Planner:
    def __init__(self, ckpt: str = DEFAULT_CKPT, data_dir: str = "data/derived/knowledge"):
        self.model, self.cfg, self.device = K.load_model(ckpt)
        meta = json.load(open(Path(ckpt).parent / "meta.json"))
        self.buckets = K.Buckets.from_state(self.cfg.buckets, meta["buckets"]); self.pos_buckets = K.Buckets.from_state(self.cfg.buckets, meta["pos_buckets"])
        self.codebook = Codebook.load(str(Path(data_dir) / "codebook.npz"))
        self.sharp = self.codebook.sharp()

    @torch.no_grad()
    def predict(self, board: chess.Board, g=None, memo=None, cp: int | None = None):
        """`cp`: the engine's evaluation now (side to move, centipawns) when the caller has it."""
        g = g if g is not None else build(board, memo if memo is not None else Memo())
        s, _ = net_sample(g, board, self.buckets, self.pos_buckets)
        b = K.collate([s], self.device)
        o = self.model(b)
        fl = self.model.future_head(o["phi"]).view(2, self.cfg.n_codes)
        pr = F.softmax(fl, dim=-1).cpu().numpy()
        cp_now = torch.tensor([math.tanh(cp / 400.0)], device=self.device) if cp is not None else None
        va = self.model.value_ahead(o["phi"], cp_now)[0].cpu().numpy()
        self._sharp_window = float(torch.sigmoid(self.model.sharp_head(o["phi"])[0, 0])) if hasattr(self.model, "sharp_head") else None
        code_now = int(self.codebook.assign(g.position_vector()[None])[0])
        return g, pr, va, code_now

    def plan(self, board: chess.Board, memo=None, top: int = 3, cp: int | None = None) -> Plan:
        g, pr, va, code_now = self.predict(board, memo=memo, cp=cp)
        cb = self.codebook
        def futures(p):
            merged: dict[str, Future] = {}                         # codes with the same words are one future to the reader
            for c in np.argsort(-p):
                t = cb.describe(int(c))
                if t in merged:
                    f = merged[t]; f.p += float(p[c]); f.value = (f.value + float(cb.values[c])) / 2
                else:
                    merged[t] = Future(int(c), float(p[c]), float(cb.values[c]), t, bool(self.sharp[c]))
            return sorted(merged.values(), key=lambda f: -f.p)[:top]
        f10, f20 = futures(pr[0]), futures(pr[1])
        exp10, exp20 = float(pr[0] @ cb.values), float(pr[1] @ cb.values)
        sharp10, sharp20 = float(pr[0][self.sharp].sum()), float(pr[1][self.sharp].sum())
        ent = float(-(pr[0] * np.log(pr[0] + 1e-9)).sum())
        plan = Plan(code_now, cb.describe(code_now), f10, f20, float(cb.values[code_now]), float(va[0]), float(va[1]), exp10, exp20, sharp10, sharp20, ent)
        plan.steer = self.steering(board, g, pr[1], exp20, code_now)
        plan.sharp_window = self._sharp_window if self._sharp_window is not None else sharp10
        plan.budget = budget_for(plan.sharp_window, window=self._sharp_window is not None)
        plan.lines = narrate(board, plan)
        return plan

    def steering(self, board: chess.Board, g, p20: np.ndarray, expected: float, code_now: int, top: int = 2) -> list[Steer]:
        """The futures worth steering to: reachable (some predicted mass), better than the expectation, and turned
        into needs by the differences of their centroid from the present."""
        cb = self.codebook
        now = cb.raw(code_now)
        runners = any(n.piece.color == board.turn and n.piece.piece_type == chess.PAWN and n.answers.get("passed") for n in g.nodes)
        def reachable(tgt):
            """A future we can PURSUE, not merely wish for: material within a piece of the present, no queen
            appearing without a runner, the phase not going backwards."""
            if abs(tgt["material_diff"] - now["material_diff"]) > 3.5:
                return False
            if tgt["queens_on"] > now["queens_on"] + 0.5 and not runners:
                return False
            if tgt["phase"] > now["phase"] + 6:
                return False
            return True
        cands = [(int(c), float(p20[c])) for c in np.argsort(-p20)[:16] if p20[c] >= 0.02 and cb.values[c] > expected + 0.03 and reachable(cb.raw(int(c)))]
        cands.sort(key=lambda cp: -(cb.values[cp[0]] - expected) * (0.3 + cp[1]))
        out = []; seen = set()
        for c, p in cands:
            tgt = cb.raw(c)
            needs = needs_for(board, g, now, tgt)
            key = tuple(t for _, _, t in needs)
            if not needs or key in seen:
                continue
            seen.add(key)
            out.append(Steer(c, cb.describe(c), float(cb.values[c]), float(cb.values[c] - expected), p, needs))
            if len(out) >= top:
                break
        return out


def needs_for(board: chess.Board, g, now: dict, tgt: dict) -> list[tuple[str, int | None, str]]:
    """The named differences between the target regime and the present, as needs the tactical layer understands.
    The mapping is a vocabulary, not a valuation: which INDICATOR is which NEED KIND; what each is worth comes
    from the codes' values."""
    from .facts import value as pval
    from . import rules as R
    us = board.turn
    out = []
    def name(sq):
        return f"the {chess.piece_name(board.piece_at(sq).piece_type)} on {chess.square_name(sq)}"
    if tgt["queens_on"] < now["queens_on"] - 0.5:
        theirs = [sq for sq in board.pieces(chess.QUEEN, not us)]
        if theirs:
            out.append(("remove", theirs[0], f"trade or remove {name(theirs[0])} (queens off)"))
    if tgt["queens_on"] > now["queens_on"] + 0.5:
        out.append(("keep", None, "keep the queens on"))
    if tgt["passed_ours"] > now["passed_ours"] + 0.5:
        out.append(("lever", None, "create a passed pawn (a lever or an exchange that leaves one)"))
    if tgt["passed_ours"] >= 0.5 and now["passed_ours"] >= 0.5 and tgt["runner_ours"] < now["runner_ours"] - 0.5:
        runners = [n.square for n in g.nodes if n.piece.color == us and n.piece.piece_type == chess.PAWN and n.answers.get("passed")]
        if runners:
            r = min(runners, key=lambda sq: g.nodes[g.index()[sq]].answers["to_go"])
            out.append(("race", r, f"push {name(r)} (the promotion race)"))
    if tgt["passed_theirs"] < now["passed_theirs"] - 0.5:
        theirs = [n.square for n in g.nodes if n.piece.color != us and n.piece.piece_type == chess.PAWN and n.answers.get("passed")]
        if theirs:
            out.append(("win", theirs[0], f"win or blockade {name(theirs[0])} (their passed pawn)"))
    if tgt["material_diff"] > now["material_diff"] + 0.75:
        targets = sorted((sq for sq, p in board.piece_map().items() if p.color != us and not R.is_royal(board, sq)), key=lambda sq: -pval(board, sq))
        if targets:
            out.append(("win", targets[0], f"win material (the dearest target: {name(targets[0])})"))
    if tgt["king_attackers"] > now["king_attackers"] + 1.0 or tgt["mate"] >= 0.5:
        out.append(("mate", None, "attack the king (the mate goal)"))
    if tgt["threat_max"] < BIG <= now["threat_max"]:
        out.append(("defend", None, "meet the standing threats first (the target is a calmer position)"))
    if tgt["phase"] < now["phase"] - 12:
        out.append(("trade", None, "trade pieces (towards the ending)"))
    if tgt["rooks_connected_ours"] > now["rooks_connected_ours"] + 0.5:
        out.append(("connect", None, "connect the rooks"))
    if tgt["open_files"] > now["open_files"] + 0.75:
        out.append(("open", None, "open a file"))
    return out


def budget_for(sharp: float, window: bool = False) -> dict:
    """The search budget from the expected sharpness: the ranking's combination cap, proof budget and mate search
    scale with how likely tactics are within ten plies. Piecewise so the decision reads as three regimes. The
    thresholds sit around the label's base rate (0.29 for a sharp code at +10; 0.84 for a big threat anywhere in
    the window, so that signal is read by how far it falls short)."""
    hi, lo = (0.9, 0.7) if window else (0.45, 0.2)
    if sharp >= hi:
        return dict(level="deep", combo_cap=1500, budget=120, mate_budget=1500)
    if sharp >= lo:
        return dict(level="normal", combo_cap=600, budget=60, mate_budget=800)
    return dict(level="shallow", combo_cap=200, budget=30, mate_budget=300)


def narrate(board: chess.Board, plan: Plan) -> list[str]:
    who = "White" if board.turn else "Black"
    out = [f"Now: {plan.now_text} (positions like this score {plan.value_now:.2f} for {who}).",
           f"In ten moves' time, most likely: " + "; ".join(f"{f.text} ({f.p:.0%}{', sharp' if f.sharp else ''})" for f in plan.futures10[:3]) + "."]
    out.append(f"Expected: value {plan.expected10:.2f} at +10 and {plan.expected20:.2f} at +20 (the engine's evaluation there, predicted: {plan.value10:+.2f} / {plan.value20:+.2f} in tanh units); "
               f"tactics likely within ten plies: {plan.sharp_window:.0%} (search budget: {plan.budget['level']}); the future is {'open' if plan.entropy10 > 2.5 else 'fairly set' if plan.entropy10 > 1.5 else 'set'} (entropy {plan.entropy10:.1f}).")
    for st in plan.steer:
        out.append(f"Worth steering to: {st.text} -- value {st.value:.2f} ({st.gain:+.2f} over the expectation), already {st.p:.0%} likely; the needs: " + "; ".join(t for _, _, t in st.needs) + ".")
    if not plan.steer:
        out.append("No reachable future beats the expectation by much: play the position as it is.")
    return out


def plan_lines(board: chess.Board, planner: "Planner | None" = None, memo=None) -> list[str]:
    try:
        pl = planner if planner is not None else Planner()
        return pl.plan(board, memo=memo).lines
    except Exception as ex:
        return [f"(no plan: {ex})"]


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--fen", required=True); ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    a = ap.parse_args(argv)
    pl = Planner(a.ckpt)
    for l in pl.plan(chess.Board(a.fen)).lines:
        print(l)


if __name__ == "__main__":
    main()
