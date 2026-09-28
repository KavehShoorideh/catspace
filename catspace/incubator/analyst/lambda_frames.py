"""ORDER-n THREATS OVER FRAMES -- lambda-search (Thomsen 2000, "Lambda-search in game trees -- with application to Go")
with the frames as the order-0 oracle; one of the tactical plan layers (`plan_layers`). A move is an order-n threat
when, if the defender passes, the attacker wins by an order n-1 search; order 0 is what the frames win now. The
attacker's candidates are the frame-interface operators (a move that adds an attacker to an enemy piece or takes a
counted defender off it, or a check); the defender's replies are those touching the threat's relevance zone
(Cazenave 2000/2002: the squares the proof depends on), its captures and checks, and an irrelevant move (the pass).
Values, not booleans: every leaf is the frames' resolution.
"""
from __future__ import annotations

import chess

from .fates import captured_value, joint_threats_book, passed, square_frame


def _pinned_off(b: chess.Board, d: int, s: int) -> bool:
    pc = b.piece_at(d)
    return pc is not None and b.is_pinned(pc.color, d) and not (b.pin(pc.color, d) & chess.BB_SQUARES[s])


def interface(b: chess.Board, side: chess.Color) -> dict:
    """{enemy square: (attackers of `side`, counted defenders)}."""
    out = {}
    for s, pc in b.piece_map().items():
        if pc.color == side or pc.piece_type == chess.KING:
            continue
        out[s] = (len(b.attackers(side, s)), sum(1 for d in b.attackers(not side, s) if not _pinned_off(b, d, s)))
    return out


def operators(b: chess.Board) -> list:
    side = b.turn; before = interface(b, side); out = []
    for m in b.legal_moves:
        if b.is_capture(m):
            continue
        nb = b.copy(stack=False); nb.push(m)
        if nb.is_check():
            out.append(m); continue
        for s, (a1, d1) in interface(nb, side).items():
            a0, d0 = before.get(s, (0, d1))
            if (a1 > a0 and a1 > 0) or d1 < d0:
                out.append(m); break
    return out


def _lines(members) -> set:
    z = set(); ms = list(members)
    for i, p in enumerate(ms):
        for q in ms[i + 1:]:
            z |= set(chess.SquareSet(chess.between(p, q)))
    return z


def frames_now(b: chess.Board) -> tuple:
    """Order 0: what the frames win for the side to move now -> (value, line, zone)."""
    (v, _, ln), book = joint_threats_book(b, square_frame)
    if v <= 0:
        return 0.0, [], frozenset()
    zone = set()
    for fr, rel, fv, fln in book.values():
        if fv >= v:
            zone |= set(fr[0]) | set(fr[2])
    return v, list(ln), frozenset(zone | _lines(zone))


class Search:
    """Order-n threat search; a node budget keeps it bounded."""

    def __init__(self, budget: int = 4000):
        self.budget = budget; self.nodes = 0; self.tt = {}

    def attack(self, b: chess.Board, n: int, depth: int) -> tuple:
        key = (b._transposition_key(), n, depth)
        if key in self.tt:
            return self.tt[key]
        self.nodes += 1
        best = frames_now(b)
        if n > 0 and depth > 0 and self.nodes < self.budget and not b.is_check():
            for m in operators(b):
                nb = b.copy(stack=False); nb.push(m)
                if nb.is_checkmate():
                    best = (1000.0, [m.uci()], frozenset({m.from_square, m.to_square})); break
                pb = passed(nb)
                if pb is not None:
                    t, tl, tz = self.attack(pb, n - 1, depth - 1)
                    if captured_value(b, m) + t <= best[0]:
                        continue
                    zone = tz | {m.from_square, m.to_square}
                else:
                    t, tl, zone = 0.0, [], frozenset({m.from_square, m.to_square})
                v, vl, vz = self.defend(nb, n, depth - 1, zone, (t, tl))
                v += captured_value(b, m)
                if v > best[0]:
                    best = (v, [m.uci()] + vl, frozenset(zone | vz))
        self.tt[key] = best
        return best

    def defend(self, b: chess.Board, n: int, depth: int, zone: frozenset, if_ignored: tuple) -> tuple:
        self.nodes += 1
        if b.is_checkmate():
            return 1000.0, [], zone
        worst = (if_ignored[0], ["--"] + list(if_ignored[1]), zone) if not b.is_check() else (1e9, [], zone)
        for d in b.legal_moves:
            if not (b.is_check() or d.from_square in zone or d.to_square in zone or b.is_capture(d) or b.gives_check(d)):
                continue
            nb = b.copy(stack=False); nb.push(d)
            if depth > 0 and self.nodes < self.budget:
                v, vl, vz = self.attack(nb, n, depth - 1)
            else:
                v, vl, vz = frames_now(nb)
            v -= captured_value(b, d)
            if v < worst[0]:
                worst = (v, [d.uci()] + vl, frozenset(zone | vz))
        return worst if worst[0] < 1e9 else (0.0, [], zone)


def threat_after(b: chess.Board, m: chess.Move, order: int = 1, budget: int = 400) -> float:
    """What the move's order-`order` threats force after the opponent's best relevant reply (0 when none)."""
    nb = b.copy(stack=False); nb.push(m)
    if nb.is_checkmate():
        return 1000.0
    pb = passed(nb)
    if pb is None:
        return 0.0                                                # a check: the frames resolve it
    base, _, _ = frames_now(passed(b)) if passed(b) is not None else (0.0, [], frozenset())
    s = Search(budget)
    t, tl, tz = s.attack(pb, order - 1, 2 * order - 1)
    if t <= 0:
        return 0.0
    v, _, _ = s.defend(nb, order, 2 * order - 1, tz | {m.from_square, m.to_square}, (t, tl))
    return max(0.0, v)


def line_san(b: chess.Board, line: list) -> str:
    out = []; bb = b.copy()
    for u in line:
        if u == "--":
            out.append("(pass)"); bb = passed(bb) or bb; continue
        m = chess.Move.from_uci(u)
        if m not in bb.legal_moves:
            out.append(f"?{u}"); break
        out.append(bb.san(m)); bb.push(m)
    return " ".join(out)
