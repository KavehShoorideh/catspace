"""WHAT MATTERS, BY ABLATION (Kaveh 2026-09-28: "can the difference in the frame be a guidance to me that there's something
here? this piece is important ... if the removal of this piece makes a big difference"; "how do I attack this? how can
it be stopped? how can I stop it from being stopped? ... questions of this sort all need to be asked, [with] a
mechanistic and generic way to ask them").

The ABLATION MAP: every piece removed in turn, the frames re-resolved (cheap: the store and the lemmas), the change in
each side's best frame read off -- how much that piece holds for its side (a defender, a shield, an attacker, a
blocker). A removal that leaves its own king in check with the other side to move marks a SHIELD: the piece blocks a
line to its king. Large changes are where the questions go:

  support(side)    the pieces a side's best threat cannot do without (removing any one drops it) -- where it can be
                   stopped (Kf5 in 6k1/6p1/..: the c5 rook that sacrifices, the c2 rook that skewers, the queen behind)
  attack(side)     the enemy pieces whose removal the side would gain most from, shields first, each with the ways to
                   remove it now -- a capture (a sacrifice when the gain pays for it) -- and the net, by the frames
                   after the capture ("e5 shields the king: Rxe5+ gives a rook, Rc5+ then skewers king and queen, +5")

    python -m catspace.incubator.analyst.ablation --fen "<fen>"
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field

import chess

from .fates import captured_value, joint_threats, passed, square_frame


def best_frame(b: chess.Board, side: chess.Color):
    """The side's best frame on b: (value, line) with the side to move, or with the other side passing; None if it
    cannot be read (the side not to move is in check)."""
    t = b if b.turn == side else passed(b)
    if t is None:
        return None
    v, _, ln = joint_threats(t, square_frame)
    return v, ln, t


@dataclass
class Holding:
    square: int
    piece: str
    shield: bool                    # removing it leaves its own king in check with the other side to move
    delta: dict = field(default_factory=dict)   # side -> change of that side's best frame when the piece is removed (None: shield)


def ablation_map(b: chess.Board) -> list:
    """Every non-king piece removed in turn: the change in each side's best frame; shields flagged. Largest first."""
    base = {c: best_frame(b, c) for c in (chess.WHITE, chess.BLACK)}
    out = []
    for s, pc in b.piece_map().items():
        if pc.piece_type == chess.KING:
            continue
        e = b.copy(stack=False); e.remove_piece_at(s)
        k = e.king(pc.color)
        shield = k is not None and e.is_attacked_by(not pc.color, k) and e.turn != pc.color
        d = {}
        for c in (chess.WHITE, chess.BLACK):
            if shield:
                d[c] = None; continue
            v = best_frame(e, c)
            d[c] = None if v is None or base[c] is None else v[0] - base[c][0]
        out.append(Holding(s, pc.symbol(), shield, d))
    out.sort(key=lambda h: (not h.shield, -max(abs(v) for v in h.delta.values() if v is not None) if any(v is not None for v in h.delta.values()) else 0))
    return out


def support(b: chess.Board, side: chess.Color, amap: list | None = None) -> list:
    """The pieces the side's best threat cannot do without: removing any one of them drops the side's best frame."""
    got = best_frame(b, side)
    if got is None or got[0] <= 0:
        return []
    amap = amap if amap is not None else ablation_map(b)
    return [h for h in amap if not h.shield and h.delta.get(side) is not None and h.delta[side] <= -0.5]


@dataclass
class Question:
    target: int
    kind: str                       # "remove the shield" | "remove"
    gain_if_gone: float | None      # the side's gain if the piece were simply gone (None: a shield -- its king would be in check)
    ways: list = field(default_factory=list)   # [(move san, net by the frames, line)]


def attack(b: chess.Board, side: chess.Color, amap: list | None = None, top: int = 3) -> list:
    """How to attack what matters: the enemy pieces whose removal the side gains most from (shields first); for each,
    the side's captures of it now -- sacrifices included -- and the net after the opponent's best reply by the
    frames: what the capture takes, less what they take back, plus what the side's frames win after that."""
    amap = amap if amap is not None else ablation_map(b)
    t = b if b.turn == side else passed(b)
    if t is None:
        return []
    out = []
    for h in amap:
        pc = b.piece_at(h.square)
        if pc is None or pc.color == side:
            continue
        gain = h.delta.get(side)
        if not h.shield and (gain is None or gain < 0.5):
            continue
        q = Question(h.square, "remove the shield" if h.shield else "remove", None if h.shield else gain)
        for m in t.generate_legal_captures(to_mask=chess.BB_SQUARES[h.square]):
            nb = t.copy(stack=False); nb.push(m)
            v, _, ln = joint_threats(nb, square_frame)          # the opponent's best reply by the frames (their gain)
            after = nb.copy(stack=False)
            for u in ln:
                mm = chess.Move.from_uci(u)
                if mm not in after.legal_moves:
                    break
                after.push(mm)
            ours = best_frame(after, side)
            net = captured_value(t, m) - v + (ours[0] if ours is not None and after.turn == side else 0.0)
            line = [t.san(m)] + _san(nb, ln) + (_san(after, ours[1]) if ours is not None and after.turn == side else [])
            q.ways.append((t.san(m), net, line))
        q.ways.sort(key=lambda w: -w[1])
        out.append(q)
        if len(out) >= top:
            break
    return out


def _san(b: chess.Board, line: list) -> list:
    bb = b.copy(stack=False); out = []
    for u in line:
        m = chess.Move.from_uci(u)
        if m not in bb.legal_moves:
            break
        out.append(bb.san(m)); bb.push(m)
    return out


def describe(b: chess.Board) -> list:
    """The ablation map, each side's support set, and the attack questions, in words."""
    amap = ablation_map(b)
    name = lambda s: f"{b.piece_at(s).symbol()}{chess.square_name(s)}"
    lines = ["what matters (each piece removed in turn; change in each side's best frame):"]
    for h in amap:
        if h.shield:
            lines.append(f"   {name(h.square)}: a SHIELD -- without it its own king stands in check")
        elif any(v is not None and abs(v) >= 0.5 for v in h.delta.values()):
            lines.append(f"   {name(h.square)}: White {h.delta[chess.WHITE]:+g}, Black {h.delta[chess.BLACK]:+g}")
    for side in (b.turn, not b.turn):
        nm = "White" if side else "Black"
        got = best_frame(b, side)
        sup = support(b, side, amap)
        if got is not None and got[0] > 0:
            lines.append(f"{nm}'s best threat {got[0]:+g} ({' '.join(_san(got[2], got[1]))}) depends on: {', '.join(name(h.square) for h in sup) or 'no single piece'}")
        for q in attack(b, side, amap):
            what = "its king would be in check" if q.gain_if_gone is None else f"{nm} would gain {q.gain_if_gone:+g}"
            ways = "; ".join(f"{w[0]} nets {w[1]:+g} ({' '.join(w[2])})" for w in q.ways[:3]) or "no capture now"
            lines.append(f"{nm}: how to {q.kind} {name(q.target)} ({what})? {ways}")
    return lines


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--fen", required=True)
    a = ap.parse_args(argv)
    for l in describe(chess.Board(a.fen)):
        print(l)


if __name__ == "__main__":
    main()
