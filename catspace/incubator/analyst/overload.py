"""THE OVERLOADED DEFENDER AS AN ARMED LEMMA (Kaveh 2026-10-03: the lemmas one by one -- after the discovered check, the
pin, the back rank and the fork; his earlier words, 2026-09-29: "a piece's coming may cost its other job"). The same
four stages:
- CHARGE: an enemy piece we attack that would fall without a given defender -- that defender is what holds it.
- ARMED (overload): one enemy piece holds two or more charges at once: it cannot answer both.
- STRIKE: take one charge (when the defender takes back, the other falls) or hit the defender itself -- verified by the
  zone search. Played at once.
- LURE (the goal): add an attacker to a second piece the defender guards -- creating the overload -- verified a little
  deeper.

    python -m catspace.incubator.analyst.overload --puzzles 8      # does it fire on deflection / capturing-defender puzzles (a smoke)
"""
from __future__ import annotations

import argparse

import chess

from .discovery import _verify


def _holds(b: chess.Board, d: int, t: int) -> bool:
    """Defender d holds the attacked piece t: we attack t, d defends it, and without d we attack it at least as often as
    the rest defend it."""
    me = b.turn
    if not b.attackers(me, t) or d not in b.attackers(not me, t):
        return False
    rest = len(b.attackers(not me, t)) - 1
    return len(b.attackers(me, t)) > rest


def overloads(b: chess.Board) -> dict:
    """{defender square: [charge squares]} for every enemy piece holding two or more charges (attacked pieces it alone
    holds). Narrow: it named 1 of 8 deflection puzzles; adding the job ledger's guards and mate jobs (jobs.ledger) was
    tried and is too slow for play (over 10 minutes for 8 puzzles, 2026-10-03) -- kept for later."""
    me = b.turn; out = {}
    for d in chess.SquareSet(b.occupied_co[not me]):
        charges = [t for t in chess.SquareSet(b.attacks_mask(d)) if b.piece_at(t) is not None and b.piece_at(t).color != me
                   and _holds(b, d, t)]
        if len(charges) >= 2:
            out[d] = charges
    return out


def armed(b: chess.Board) -> dict:
    return overloads(b)


def strikes(b: chess.Board) -> list:
    """Take a charge of an overloaded defender, or hit the defender: [(move, gain, defender)] by the zone search."""
    out = []; seen = set()
    for d, charges in overloads(b).items():
        for m in b.legal_moves:
            if m in seen:
                continue
            nb = b.copy(stack=False); nb.push(m)
            if m.to_square in charges or m.to_square == d or d in chess.SquareSet(nb.attacks_mask(m.to_square)):
                g = _verify(b, m)
                if g >= 1:
                    out.append((m, g, d)); seen.add(m)
    return sorted(out, key=lambda x: -x[1])


def lures(b: chess.Board) -> list:
    """Moves that give a defender a second charge (an attack on another piece it guards), winning by the zone search at
    6 plies: [(move, gain, defender)]."""
    me = b.turn; before = overloads(b); out = []
    for m in b.legal_moves:
        nb = b.copy(stack=False); nb.push(m)
        tb = nb.copy(stack=False); tb.turn = me; tb.ep_square = None    # the position as it stands for us next move
        if not tb.is_valid():
            continue
        new = {d: c for d, c in overloads(tb).items() if d not in before}
        if new:
            g = _verify(b, m, depth=6)
            if g >= 1:
                out.append((m, g, next(iter(new))))
    return sorted(out, key=lambda x: -x[1])


def puzzles(n: int, offset: int = 400, log=print) -> None:
    """A smoke on deflection and capturing-defender puzzles: overloads, strikes and lures at the solution's first move."""
    from .frame_puzzles import fresh_rows
    shown = 0
    for row in fresh_rows(30 * n, 997, offset):
        if not ({"deflection", "capturingDefender", "overloading"} & set(row["Themes"].split())):
            continue
        b = chess.Board(row["FEN"]); mv = row["Moves"].split(); b.push_uci(mv[0]); sol = chess.Move.from_uci(mv[1])
        st = strikes(b); lu = lures(b); ov = overloads(b)
        log(f"{row['PuzzleId']} solution {b.san(sol)} | overloaded: "
            + ("; ".join(f"{b.piece_at(d).symbol()}{chess.square_name(d)} holds {', '.join(chess.square_name(t) for t in c)}" for d, c in ov.items()) or "none")
            + f" | strikes: {', '.join(f'{b.san(m)} {g:+g}' for m, g, _ in st[:3]) or 'none'}"
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
