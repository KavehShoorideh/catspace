"""THE PIN AS AN ARMED LEMMA (Kaveh 2026-10-03: "let's go on to the next lemma", after the discovered check; earlier,
2026-10-01: "a high-value lemma would be a pin ... putting pressure on a pinned piece"). The same four stages as
discovery.py:
- PIN: a slider of ours (bishop, rook, queen) whose line meets an enemy piece (the PINNED piece) and, behind it, a more
  valuable enemy piece or the king (the SCREENED piece). Absolute when the king is behind (the pinned piece may not
  move off the line at all); relative otherwise (moving it loses the bigger piece).
- ARMED: such a pin stands on the board. What it is worth is known in advance: the pinned piece cannot run, so it falls
  when attacked more often than it is defended -- or at once when attacked by something cheaper.
- STRIKE (pressure on the pinned piece): a move of ours that adds an attacker to the pinned piece and wins material,
  verified by the zone search on that one move. Played at once.
- LURE (the goal before the pin exists): a move of ours that CREATES a pin on a piece we can then win -- the slider
  steps onto the line (Bg5 pinning the knight to the queen); verified the same way.

    python -m catspace.incubator.analyst.pin --puzzles 5        # does it fire on Lichess pin puzzles (a smoke, no measurement)
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass

import chess

from .discovered_check import DIRS, VAL, _ray, _verify


@dataclass
class Pin:
    slider: int                     # ours
    pinned: int                     # theirs, in front
    screened: int                   # theirs, behind it: the king or a piece worth more
    absolute: bool

    def text(self, b: chess.Board) -> str:
        n = lambda s: f"{b.piece_at(s).symbol()}{chess.square_name(s)}"
        return f"{n(self.slider)} pins {n(self.pinned)} to {n(self.screened)} ({'absolute' if self.absolute else 'relative'})"


def pins(b: chess.Board, side: chess.Color | None = None) -> list:
    """Every pin `side`'s sliders hold (default: the side to move)."""
    side = b.turn if side is None else side
    out = []
    for s in chess.SquareSet(b.occupied_co[side]):
        pt = b.piece_type_at(s)
        if pt not in DIRS:
            continue
        for df, dr in DIRS[pt]:
            first = None
            for q in _ray(s, df, dr):
                p = b.piece_at(q)
                if p is None:
                    continue
                if first is None:
                    if p.color == side or p.piece_type == chess.KING:
                        break
                    first = q
                    continue
                if p.color != side and (p.piece_type == chess.KING or VAL[p.piece_type] > VAL[b.piece_type_at(first)]):
                    out.append(Pin(s, first, q, p.piece_type == chess.KING))
                break
    return out


def _attackers_after(b: chess.Board, m: chess.Move, sq: int) -> bool:
    nb = b.copy(stack=False); nb.push(m)
    return sq in chess.SquareSet(nb.attacks_mask(m.to_square))


def _fake_guard(b: chess.Board, m: chess.Move, p: Pin) -> bool:
    """An absolute pin's guard is fake: the move lands on a square the pinned piece attacks but cannot take on (it would
    leave the line), and no other enemy piece guards it."""
    if not p.absolute or m.to_square not in chess.SquareSet(b.attacks_mask(p.pinned)):
        return False
    line = set(chess.SquareSet(chess.between(p.slider, p.screened))) | {p.slider}
    return m.to_square not in line and b.attackers(not b.turn, m.to_square) == chess.SquareSet(chess.BB_SQUARES[p.pinned])


def strikes(b: chess.Board) -> list:
    """Moves that exploit a pin and win material (zone search): pressure on the pinned piece (a new attacker, or
    taking it), or a move onto a square only the pinned piece guards (an absolute pin's guard is fake). [(move, gain, pin)]."""
    out = []; seen = set()
    for p in pins(b):
        pv = VAL[b.piece_type_at(p.pinned)]
        for m in b.legal_moves:
            if m in seen:
                continue
            takes = m.to_square == p.pinned
            if _fake_guard(b, m, p) or takes:
                g = _verify(b, m)
                if g >= 1:
                    out.append((m, g, p)); seen.add(m)
                continue
            if m.from_square == p.slider or not _attackers_after(b, m, p.pinned):
                continue
            if VAL[b.piece_type_at(m.from_square)] >= pv and len(b.attackers(b.turn, p.pinned)) < len(b.attackers(not b.turn, p.pinned)):
                continue                                              # neither cheaper nor outnumbering: no pressure worth a search
            g = _verify(b, m)
            if g >= 1:
                out.append((m, g, p)); seen.add(m)
    return sorted(out, key=lambda x: -x[1])


def lures(b: chess.Board) -> list:
    """Moves that create a new pin on a piece worth winning: [(move, gain by the zone search, the pin it makes)]."""
    before = {(p.slider, p.pinned) for p in pins(b)}
    out = []
    for m in b.legal_moves:
        if b.piece_type_at(m.from_square) not in DIRS:
            continue
        nb = b.copy(stack=False); nb.push(m)
        new = [p for p in pins(nb, b.turn) if p.slider == m.to_square and (p.slider, p.pinned) not in before]
        if new:
            g = _verify(b, m)
            if g >= 1:
                out.append((m, g, new[0]))
    return sorted(out, key=lambda x: -x[1])


def puzzles(n: int, offset: int = 400, log=print) -> None:
    """A smoke on Lichess pin puzzles: the lemma's pins, strikes and lures at the solution's first move."""
    from .frame_puzzles import fresh_rows
    shown = 0
    for row in fresh_rows(40 * n, 997, offset):
        if "pin" not in row["Themes"].split():
            continue
        b = chess.Board(row["FEN"]); mv = row["Moves"].split(); b.push_uci(mv[0]); sol = chess.Move.from_uci(mv[1])
        st = strikes(b); lu = lures(b); ps = pins(b)
        log(f"{row['PuzzleId']} solution {b.san(sol)} | pins now: {'; '.join(p.text(b) for p in ps) or 'none'}"
            f" | strikes: {', '.join(f'{b.san(m)} {g:+g}' for m, g, _ in st[:3]) or 'none'}"
            f" | lures: {', '.join(f'{b.san(m)} {g:+g}' for m, g, _ in lu[:3]) or 'none'}"
            f" | the solution is {'a STRIKE' if any(m == sol for m, _, _ in st) else 'a LURE' if any(m == sol for m, _, _ in lu) else 'not named'}")
        shown += 1
        if shown >= n:
            break


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--puzzles", type=int, default=0); ap.add_argument("--offset", type=int, default=400)
    a = ap.parse_args(argv)
    if a.puzzles:
        puzzles(a.puzzles, a.offset)


if __name__ == "__main__":
    main()
