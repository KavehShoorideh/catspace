"""BRIDGES: a Lean theorem is about chess-lean's rules; the player runs on python-chess. A bridge checks the
detector's claims by exhaustive search in python-chess -- on held-out puzzle positions and on random instances -- and
reports how many claims it checked and how many were wrong. Any wrong claim makes the idea unsound (it may not close
search nodes). Each returns dict(checked, wrong, what, seconds, ...)."""
from __future__ import annotations

import random
import time

import chess

from ..lean_ideas import LeanIdeas


def backrank_heldout(li: LeanIdeas, n: int = 300, offset: int = 17) -> dict:
    """Every position along n held-out back-rank puzzle lines: the kernel's legal moves against python-chess, and every
    MATE claim checked as checkmate."""
    from ..lean_backrank_puzzles import bridge, puzzle_rows
    t0 = time.perf_counter()
    rows = puzzle_rows(n, 97, offset, None)
    r = bridge(rows, li, log=lambda *a: None)
    return dict(checked=r["mate_claims"] + r["positions"], wrong=(r["mate_claims"] - r["mate_true"]) + (r["positions"] - r["agree"]),   # a legal-move
                what=f"MATE claims on {r['positions']} positions of {n} held-out back-rank puzzle lines (rows i % 97 == {offset}), "     # disagreement is
                     f"each checked as checkmate, and the kernel's legal-move set against python-chess's at each: {r['agree']}/{r['positions']} agree",  # wrong too
                legal_agree=r["agree"], positions=r["positions"], seconds=time.perf_counter() - t0)


def random_backrank_position(rng: random.Random) -> chess.Board | None:
    """A random position around a boxed king: Black's king on its home rank, pawns before it (most of the time), up to
    two defenders, one or two white rooks or queens, White to move."""
    b = chess.Board(None)
    kf = rng.randrange(8)
    b.set_piece_at(chess.square(kf, 7), chess.Piece(chess.KING, chess.BLACK))
    for f in (kf - 1, kf, kf + 1):
        if 0 <= f < 8 and rng.random() < 0.75:
            b.set_piece_at(chess.square(f, 6), chess.Piece(chess.PAWN, chess.BLACK))
    for _ in range(rng.randint(0, 2)):
        sq = rng.choice([s for s in range(32, 64) if b.piece_at(s) is None])
        b.set_piece_at(sq, chess.Piece(rng.choice([chess.ROOK, chess.BISHOP, chess.KNIGHT, chess.QUEEN]), chess.BLACK))
    for _ in range(rng.randint(1, 2)):
        sq = rng.choice([s for s in range(0, 48) if b.piece_at(s) is None])
        b.set_piece_at(sq, chess.Piece(rng.choice([chess.ROOK, chess.ROOK, chess.QUEEN]), chess.WHITE))
    b.set_piece_at(rng.choice([s for s in range(0, 24) if b.piece_at(s) is None]), chess.Piece(chess.KING, chess.WHITE))
    for _ in range(rng.randint(0, 3)):
        sq = rng.choice([s for s in range(8, 48) if b.piece_at(s) is None])
        b.set_piece_at(sq, chess.Piece(chess.PAWN, chess.WHITE))
    b.turn = chess.WHITE
    return b if b.is_valid() else None


def backrank_random(li: LeanIdeas, n: int = 1000, seed: int = 1) -> dict:
    t0 = time.perf_counter()
    rng = random.Random(seed)
    positions = checked = wrong = 0
    examples = []
    while positions < n:
        b = random_backrank_position(rng)
        if b is None:
            continue
        positions += 1
        for inst in li.backrank(b):
            if inst.status != "MATE":
                continue
            checked += 1
            nb = b.copy(stack=False)
            nb.push(inst.move)
            if not nb.is_checkmate():
                wrong += 1
                examples.append((b.fen(), inst.move.uci()))
    return dict(checked=checked, wrong=wrong, what=f"MATE claims on {n} random boxed-king positions (seed {seed}), each checked as checkmate",
                positions=positions, examples=examples[:5], seconds=time.perf_counter() - t0)


def passer_random(li: LeanIdeas, form: int, n: int, seed: int, their_pawns: int = 0) -> dict:
    from ..lean_passer_check import check
    r = check(n=n, seed=seed, knights=0, form=form, their_pawns=their_pawns, li=li)
    return dict(checked=r["proved"], wrong=r["wrong"], what=f"PROVED claims (form {form}) on {n} random pawn endings (seed {seed}, enemy pawns <= {their_pawns}), each checked by the exhaustive race search",
                positions=r["positions"], plies=r["plies"], examples=r["examples"][:5], seconds=r["seconds"])


def passer_heldout(li: LeanIdeas, n: int = 500, offset: int = 17, max_plies: int = 11) -> dict:
    """Every position along n held-out pawn-ending puzzle lines, both sides to move: each PROVED claim of the passer2
    detector (either form) checked by the exhaustive race search (claims longer than max_plies are counted, not
    searched)."""
    from ..lean_backrank_puzzles import puzzle_rows
    from ..lean_passer_check import promotes_within
    t0 = time.perf_counter()
    positions = checked = wrong = skipped = 0
    examples = []
    for row in puzzle_rows(n, 97, offset, None, theme="pawnEndgame"):
        b = chess.Board(row["FEN"])
        mv = row["Moves"].split()
        b.push_uci(mv[0])
        for i in range(1, len(mv)):
            positions += 1
            for inst in li.passer2(b):
                if inst.status != "PROVED":
                    continue
                if inst.plies > max_plies:
                    skipped += 1
                    continue
                checked += 1
                if not promotes_within(b.copy(), chess.square_file(inst.square), inst.plies):
                    wrong += 1
                    examples.append((b.fen(), chess.square_name(inst.square)))
            b.push_uci(mv[i])
    return dict(checked=checked, wrong=wrong, what=f"PROVED claims on {positions} positions of {n} held-out pawn-ending puzzle lines (rows i % 97 == {offset}), each checked by the exhaustive race search ({skipped} longer than {max_plies} plies not searched)",
                positions=positions, skipped=skipped, examples=examples[:5], seconds=time.perf_counter() - t0)


BRIDGES = {
    "backrank_heldout": lambda li: backrank_heldout(li),
    "backrank_random": lambda li: backrank_random(li, n=1000, seed=1),
    "passer1_random": lambda li: passer_random(li, form=1, n=1000, seed=1),
    "passer2_random": lambda li: passer_random(li, form=2, n=2000, seed=7, their_pawns=3),
    "passer2_heldout": lambda li: passer_heldout(li),
}
