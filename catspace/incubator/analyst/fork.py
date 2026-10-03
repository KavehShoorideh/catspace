"""THE FORK AS AN ARMED LEMMA (Kaveh 2026-10-03: the next lemma after the discovered check, the pin and the back rank).
The same four stages:
- FORK SQUARE: a square where one of our pieces, landing, attacks two or more enemy targets worth taking -- the king and
  a piece, or pieces each worth more than the forker or undefended.
- ARMED: such a square is reachable now (a forking move exists, safe or not).
- STRIKE: the forking move that wins material, verified by the zone search. Played at once.
- LURE (the goal): a check or a capture whose forced replies -- every reply to a check, the natural recapture to a
  capture -- put the enemy pieces where a fork wins (Bxf7+ Kxf7 then the knight fork), verified a little deeper.

    python -m catspace.incubator.analyst.fork --puzzles 8          # does it fire on Lichess fork puzzles (a smoke)
"""
from __future__ import annotations

import argparse

import chess

from .discovery import VAL, _verify


def _targets(b: chess.Board, m: chess.Move) -> list:
    """The enemy targets the moved piece attacks after m that are worth taking."""
    me = b.turn
    nb = b.copy(stack=False); nb.push(m)
    fv = VAL[b.piece_type_at(m.from_square)]
    out = []
    for t in chess.SquareSet(nb.attacks_mask(m.to_square)):
        p = nb.piece_at(t)
        if p is None or p.color == me:
            continue
        if p.piece_type == chess.KING or VAL[p.piece_type] > fv or not nb.attackers(not me, t):
            out.append(t)
    return out


def forks(b: chess.Board) -> list:
    """The side to move's forking moves (two or more targets worth taking): [(move, targets)] -- armed when non-empty."""
    out = []
    for m in b.legal_moves:
        ts = _targets(b, m)
        if len(ts) >= 2:
            out.append((m, ts))
    return out


def armed(b: chess.Board) -> list:
    return forks(b)


def strikes(b: chess.Board) -> list:
    """Forking moves that win material (zone search): [(move, gain, targets)]."""
    out = []
    for m, ts in forks(b):
        g = _verify(b, m)
        if g >= 1:
            out.append((m, g, ts))
    return sorted(out, key=lambda x: -x[1])


def lures(b: chess.Board) -> list:
    """Checks and captures whose forced replies leave a fork that wins (every reply to a check; the natural recapture
    to a capture), verified by the zone search at 6 plies: [(move, gain, None)]."""
    out = []
    for m in b.legal_moves:
        check, cap = b.gives_check(m), b.is_capture(m)
        if not (check or cap):
            continue
        nb = b.copy(stack=False); nb.push(m)
        replies = list(nb.legal_moves) if check else [r for r in nb.legal_moves if r.to_square == m.to_square and nb.is_capture(r)]
        if not replies or len(replies) > 6:
            continue
        ok = True
        for r in replies:
            rb = nb.copy(stack=False); rb.push(r)
            if not forks(rb):
                ok = False
                break
        if ok:
            g = _verify(b, m, depth=6)
            if g >= 1:
                out.append((m, g, None))
    return sorted(out, key=lambda x: -x[1])


def puzzles(n: int, offset: int = 400, log=print) -> None:
    """A smoke on Lichess fork puzzles: the strikes and lures at the solution's first move."""
    from .frame_puzzles import fresh_rows
    shown = 0
    for row in fresh_rows(30 * n, 997, offset):
        if "fork" not in row["Themes"].split():
            continue
        b = chess.Board(row["FEN"]); mv = row["Moves"].split(); b.push_uci(mv[0]); sol = chess.Move.from_uci(mv[1])
        st = strikes(b); lu = lures(b)
        log(f"{row['PuzzleId']} solution {b.san(sol)} | strikes: {', '.join(f'{b.san(m)} {g:+g}' for m, g, _ in st[:3]) or 'none'}"
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
