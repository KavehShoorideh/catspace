"""THE BACK-RANK MATE AS AN ARMED LEMMA (Kaveh 2026-10-03, after the discovered check and the pin: "back-rank mate";
2026-10-01: "my rook is not on the back rank, and they have a back rank situation ... if I have two pieces, they have
one, I can do a back rank mate ... a blocked king side, three pieces where the king is behind and cannot move ... a
frame where a back rank is threatened"). The same four stages as discovery.py and pin.py:
- BOXED: the enemy king on its home rank, every square in front of it taken by its own pieces or covered by ours --
  it can only slide along the rank.
- ARMED: boxed, and a rook or queen of ours can reach that rank with check. The count that decides it: our heavy pieces
  that can arrive against their pieces that guard the rank (two against one mates, as Kaveh put it).
- STRIKE: the check on the back rank that mates or wins -- also through a blocker (Rd8+ Rf8 Rxf8#) -- verified by the
  zone search. Played at once.
- LURE (the goal while armed): take, attack or drag away the last defender of the back rank -- verified the same way,
  a little deeper (a decoy sacrifice needs a few more plies).

    python -m catspace.incubator.analyst.backrank --puzzles 8      # does it fire on Lichess back-rank puzzles (a smoke)
"""
from __future__ import annotations

import argparse

import chess

from .discovery import _verify

HEAVY = (chess.ROOK, chess.QUEEN)


def home_rank(colour: chess.Color) -> int:
    return 0 if colour == chess.WHITE else 7


def boxed(b: chess.Board, victim: chess.Color) -> bool:
    """The victim's king is on its home rank and every square in front of it is its own piece or covered by us."""
    k = b.king(victim)
    if k is None or chess.square_rank(k) != home_rank(victim):
        return False
    fwd = 1 if victim == chess.WHITE else -1
    f = chess.square_file(k); r = chess.square_rank(k) + fwd
    for ff in (f - 1, f, f + 1):
        if 0 <= ff < 8:
            s = chess.square(ff, r)
            p = b.piece_at(s)
            if not (p is not None and p.color == victim) and not b.is_attacked_by(not victim, s):
                return False
    return True


def _rank_checks(b: chess.Board) -> list:
    """Our rook and queen moves onto the enemy's home rank that give check."""
    victim = not b.turn; hr = home_rank(victim)
    return [m for m in b.legal_moves if b.piece_type_at(m.from_square) in HEAVY and chess.square_rank(m.to_square) == hr and b.gives_check(m)]


def defenders(b: chess.Board) -> set:
    """The enemy pieces (not the king) that guard a square of its home rank where one of our heavy pieces could land."""
    victim = not b.turn; hr = home_rank(victim)
    out = set()
    for m in b.legal_moves:
        if b.piece_type_at(m.from_square) in HEAVY and chess.square_rank(m.to_square) == hr:
            out |= {s for s in chess.SquareSet(b.attackers(victim, m.to_square)) if b.piece_type_at(s) != chess.KING}
    return out


def armed(b: chess.Board) -> dict | None:
    """Boxed and reachable: the count of our heavy pieces that can reach the rank against their defenders of it."""
    if not boxed(b, not b.turn):
        return None
    hr = home_rank(not b.turn)
    ours = {m.from_square for m in b.legal_moves if b.piece_type_at(m.from_square) in HEAVY and chess.square_rank(m.to_square) == hr}
    if not ours:
        return None
    return dict(attackers=sorted(chess.square_name(s) for s in ours), defenders=sorted(chess.square_name(s) for s in defenders(b)))


def strikes(b: chess.Board) -> list:
    """Back-rank checks that mate or win (zone search): [(move, gain, None)]."""
    if not boxed(b, not b.turn):
        return []
    out = []
    for m in _rank_checks(b):
        g = _verify(b, m)
        if g >= 1:
            out.append((m, g, None))
    return sorted(out, key=lambda x: -x[1])


def lures(b: chess.Board) -> list:
    """While armed: moves that take, attack or drag away a defender of the back rank and win (zone search, 6 plies):
    [(move, gain, the defender's square)]."""
    if armed(b) is None:
        return []
    ds = defenders(b)
    out = []
    for m in b.legal_moves:
        nb = b.copy(stack=False); nb.push(m)
        hit = (m.to_square in ds) or bool(ds & set(chess.SquareSet(nb.attacks_mask(m.to_square))))
        if not hit:
            continue
        g = _verify(b, m, depth=6)
        if g >= 1:
            d = m.to_square if m.to_square in ds else next(iter(ds & set(chess.SquareSet(nb.attacks_mask(m.to_square)))))
            out.append((m, g, chess.square_name(d)))
    return sorted(out, key=lambda x: -x[1])


def puzzles(n: int, offset: int = 400, log=print) -> None:
    """A smoke on Lichess back-rank puzzles: armed?, strikes and lures at the solution's first move."""
    from .frame_puzzles import fresh_rows
    shown = 0
    for row in fresh_rows(60 * n, 997, offset):
        if "backRankMate" not in row["Themes"].split():
            continue
        b = chess.Board(row["FEN"]); mv = row["Moves"].split(); b.push_uci(mv[0]); sol = chess.Move.from_uci(mv[1])
        st = strikes(b); lu = lures(b); arm = armed(b)
        log(f"{row['PuzzleId']} solution {b.san(sol)} | armed: {arm} | strikes: {', '.join(f'{b.san(m)} {g:+g}' for m, g, _ in st[:3]) or 'none'}"
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
