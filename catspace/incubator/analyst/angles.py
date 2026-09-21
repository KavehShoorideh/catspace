"""ANGLES OF ATTACK ON THE KING (Kaveh 2026-09-20: "keep track of angles of attack to our king, so we can refute
whole lines of attack, e.g. if I'm sitting behind 3 pawns").

An ANGLE is a way into the king: one of the eight rays from its square (files, ranks, diagonals), or the knight's
hops. Along a ray the SHIELD is the first piece from the king outward; the angle is CLOSED while the shield is our
own piece that stands (attacked no more than defended, by the ledger), CONTESTED when our shield can be taken or
must move, OPEN when the ray is empty to the edge or the first piece is an enemy slider that moves along it. The
knight angle is closed when no enemy knight's region reaches a square that attacks the king.
Every angle names the enemy pieces it concerns: the sliders whose regions reach the ray's open segment. A closed
angle is a LEMMA over them: every attack that would arrive along it is refuted by the shield, as a family, without
searching any of them; and across plies an angle that opens is the event the tracker reports. Rules only: lines,
the first piece on them, attack counts, regions."""
from __future__ import annotations

from dataclasses import dataclass, field

import chess

from .defense import defense
from .regions import region

RAYS = {"north": (0, 1), "south": (0, -1), "east": (1, 0), "west": (-1, 0), "north-east": (1, 1), "north-west": (-1, 1), "south-east": (1, -1), "south-west": (-1, -1)}
RAY_KIND = {"north": "file", "south": "file", "east": "rank", "west": "rank", "north-east": "diagonal", "north-west": "diagonal", "south-east": "diagonal", "south-west": "diagonal"}


@dataclass
class Angle:
    name: str                              # the ray, or "knight"
    king: int
    state: str                             # closed | contested | open | edge (the ray ends at the board's edge at once)
    shield: int | None = None              # the first piece's square
    shield_ours: bool = True
    segment: list = field(default_factory=list)      # the empty squares between the king and the shield / the edge
    concerns: list = field(default_factory=list)     # enemy pieces (squares) that could attack along this angle
    why: str = ""

    def text(self, board: chess.Board) -> str:
        who = lambda s: f"{chess.piece_name(board.piece_at(s).piece_type)} {chess.square_name(s)}" if board.piece_at(s) else chess.square_name(s)
        if self.name == "knight":
            return f"knight angle: {self.state}" + (f" ({', '.join(who(s) for s in self.concerns)} could reach)" if self.concerns else "")
        return f"{self.name} {RAY_KIND[self.name]}: {self.state}" + (f", shield {who(self.shield)}" if self.shield is not None and self.shield_ours else "") + (f" ({self.why})" if self.why else "")


def _ray_squares(sq: int, d: tuple) -> list[int]:
    f, r = chess.square_file(sq) + d[0], chess.square_rank(sq) + d[1]; out = []
    while 0 <= f < 8 and 0 <= r < 8:
        out.append(chess.square(f, r)); f += d[0]; r += d[1]
    return out


def _slides_along(piece: chess.Piece, kind: str) -> bool:
    return piece.piece_type == chess.QUEEN or (piece.piece_type == chess.ROOK and kind in ("file", "rank")) or (piece.piece_type == chess.BISHOP and kind == "diagonal")


def angles(board: chess.Board, us: chess.Color, memo=None, regs: dict | None = None) -> list[Angle]:
    k = board.king(us)
    if k is None:
        return []
    them = not us
    regs = regs if regs is not None else {sq: region(board, sq) for sq, p in board.piece_map().items() if p.color == them and p.piece_type in (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT)}
    out = []
    for name, d in RAYS.items():
        sqs = _ray_squares(k, d); kind = RAY_KIND[name]
        a = Angle(name, k, "open")
        if not sqs:
            a.state = "edge"; out.append(a); continue
        shield = next((s for s in sqs if board.piece_at(s) is not None), None)
        a.shield = shield; a.segment = sqs[:sqs.index(shield)] if shield is not None else sqs
        seg_bb = 0
        for s in a.segment:
            seg_bb |= chess.BB_SQUARES[s]
        # the enemy sliders that could use this ray: their region reaches a square of the open segment (or the shield's square)
        reach_bb = seg_bb | (chess.BB_SQUARES[shield] if shield is not None else 0)
        for sq, r in regs.items():
            p = board.piece_at(sq)
            if p is not None and p.color == them and _slides_along(p, kind) and ((r.safe | r.contested_bb) & reach_bb):
                a.concerns.append(sq)
        if shield is None:
            a.state = "open"; a.why = "the ray runs to the edge"
        else:
            sp = board.piece_at(shield); a.shield_ours = sp.color == us
            if sp.color == them:
                a.state = "open" if _slides_along(sp, kind) else "closed"
                a.why = f"an enemy {chess.piece_name(sp.piece_type)} stands on it" if a.state == "open" else f"an enemy {chess.piece_name(sp.piece_type)} blocks it (and cannot slide along)"
                a.shield_ours = False
            else:
                d_ = defense(board, shield, memo=memo)
                attacked = len(board.attackers(them, shield)); defended = len(board.attackers(us, shield))
                if d_ is not None and d_.falls:
                    a.state = "contested"; a.why = f"the shield can be taken ({attacked} attackers, {defended} defenders)"
                elif attacked > 0 and not a.concerns:
                    a.state = "closed"; a.why = f"the shield is attacked {attacked}x but nothing could use the line"
                elif attacked > 0:
                    a.state = "contested"; a.why = f"the shield is attacked {attacked}x, defended {defended}x"
                else:
                    a.state = "closed"; a.why = "the shield stands unattacked" if a.concerns else "nothing could use the line"
        out.append(a)
    # the knight angle
    kn = Angle("knight", k, "closed")
    hop_bb = int(chess.BB_KNIGHT_ATTACKS[k])
    for sq, r in regs.items():
        p = board.piece_at(sq)
        if p is not None and p.color == them and p.piece_type == chess.KNIGHT and ((r.safe | r.contested_bb) & hop_bb):
            kn.concerns.append(sq)
    kn.state = "open" if kn.concerns else "closed"
    out.append(kn)
    return out


def summary(board: chess.Board, us: chess.Color, memo=None) -> str:
    """'Your king sits behind three pawns: five angles closed, one contested (the north-east diagonal at f2), two open.'"""
    ang = angles(board, us, memo)
    closed = [a for a in ang if a.state == "closed" and a.name != "knight"]; cont = [a for a in ang if a.state == "contested"]; opn = [a for a in ang if a.state == "open" and a.name != "knight"]
    shields = [a.shield for a in closed + cont if a.shield is not None and a.shield_ours and board.piece_at(a.shield).piece_type == chess.PAWN]
    parts = [f"{'White' if us else 'Black'}'s king on {chess.square_name(board.king(us))}: {len(closed)} angles closed, {len(cont)} contested, {len(opn)} open"]
    if shields:
        parts.append(f"behind {len(set(shields))} pawn{'s' if len(set(shields)) != 1 else ''}")
    for a in cont:
        parts.append(f"contested: {a.name} at {chess.square_name(a.shield)}" if a.shield is not None else f"contested: {a.name}")
    for a in opn:
        who = ", ".join(f"{chess.piece_name(board.piece_at(s).piece_type)} {chess.square_name(s)}" for s in a.concerns)
        parts.append(f"open: {a.name}" + (f" ({who} could use it)" if who else " (nothing to use it)"))
    kn = next(a for a in ang if a.name == "knight")
    if kn.state == "open":
        parts.append("a knight could reach a checking square")
    return "; ".join(parts)


def closed_families(board: chess.Board, us: chess.Color, memo=None) -> list[tuple[str, list[int]]]:
    """The LEMMA: for each closed angle, the enemy pieces whose attacks along it are refuted by the shield --
    a family that needs no search while the shield stands."""
    return [(a.text(board), a.concerns) for a in angles(board, us, memo) if a.state == "closed" and a.concerns]


def events(before: chess.Board, after: chess.Board, us: chess.Color) -> list[str]:
    """Across a ply: angles that opened or closed (the tracker's line)."""
    b0 = {a.name: a.state for a in angles(before, us)}; b1 = {a.name: a.state for a in angles(after, us)}
    out = []
    for n in b0:
        if b0[n] != b1.get(n):
            out.append(f"the {n} angle on {'White' if us else 'Black'}'s king: {b0[n]} -> {b1.get(n)}")
    return out


if __name__ == "__main__":
    import sys
    b = chess.Board(sys.argv[1] if len(sys.argv) > 1 else "r3kb1r/ppp1qpp1/3pbn1p/4P3/2BQ1B2/2N5/PPP2PPP/R3K2R w KQkq - 0 1")
    for us in (chess.WHITE, chess.BLACK):
        print(summary(b, us))
        for a in angles(b, us):
            print("   ", a.text(b))
