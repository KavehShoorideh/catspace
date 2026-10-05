"""THE PASSER LEMMA AGAINST AN EXHAUSTIVE RACE SEARCH (the bridge for lemma 2).

The Lean theorem `unstoppablePasser_sound` says: when its hypotheses hold, our pawn stands on the promotion square as a
queen within 2n-1 plies whatever the opponent does. This script asks the detector on many pawn-ending positions and,
for every PROVED claim, checks the same statement with python-chess by a full AND/OR search over the race (we push,
they play anything). A single disagreement is a bug in the bridge or the kernel; none is expected.

    python -m catspace.incubator.analyst.lean_passer_check --n 2000 --seed 1
"""
from __future__ import annotations

import argparse
import random
import time

import chess

from .lean_ideas import LeanIdeas, encode


def random_pawn_ending(rng: random.Random, our_pieces: int = 0, their_pawns: int = 0) -> chess.Board | None:
    """Our king, one to three pawns (one of them the candidate), optional knights, vs a king with `their_pawns`
    pawns at most; white to move."""
    b = chess.Board(None)
    nt = rng.randint(0, their_pawns)
    squares = rng.sample(range(64), 3 + rng.randint(0, 2) + our_pieces + nt)
    k, ek, *rest = squares
    if chess.square_distance(k, ek) <= 1:
        return None
    b.set_piece_at(k, chess.Piece(chess.KING, chess.WHITE))
    b.set_piece_at(ek, chess.Piece(chess.KING, chess.BLACK))
    ours = rest[:len(rest) - nt]
    kinds = [chess.PAWN] * (len(ours) - our_pieces) + [chess.KNIGHT] * our_pieces
    for sq, kind in zip(ours, kinds):
        if kind == chess.PAWN and chess.square_rank(sq) in (0, 7):
            return None
        b.set_piece_at(sq, chess.Piece(kind, chess.WHITE))
    for sq in rest[len(ours):]:
        if chess.square_rank(sq) in (0, 7):
            return None
        b.set_piece_at(sq, chess.Piece(chess.PAWN, chess.BLACK))
    b.turn = chess.WHITE
    if not b.is_valid() or b.is_check() or b.attackers_mask(chess.WHITE, ek):
        return None
    return b


def promotes_within(b: chess.Board, file: int, plies: int) -> bool:
    """We (side to move at the root) get a queen on the promotion square of `file` within `plies` plies, whatever the
    opponent does; a stalemated or mated opponent does not count as reaching it (the Lean `forcedWithinB`)."""
    us = b.turn
    promo_sq = chess.square(file, 7 if us == chess.WHITE else 0)

    def goal(bd):
        p = bd.piece_at(promo_sq)
        return p is not None and p.color == us and p.piece_type == chess.QUEEN

    def rec(bd, n):
        if goal(bd):
            return True
        if n == 0:
            return False
        moves = list(bd.legal_moves)
        if bd.turn == us:
            # the race first: pushes of the candidate pawn (promotions ahead), then everything else
            moves.sort(key=lambda m: (0 if (chess.square_file(m.from_square) == file and bd.piece_type_at(m.from_square) == chess.PAWN
                                            and chess.square_file(m.to_square) == file) else 1, 0 if m.promotion else 1))
            for m in moves:
                bd.push(m)
                ok = rec(bd, n - 1)
                bd.pop()
                if ok:
                    return True
            return False
        if not moves:
            return False
        for m in moves:
            bd.push(m)
            ok = rec(bd, n - 1)
            bd.pop()
            if not ok:
                return False
        return True

    return rec(b, plies)


def parse(line: str):
    out = []
    for cand in (x for x in line.split(" | ") if x):
        tok = cand.split()
        sq = chess.parse_square(tok[1])
        status = tok[2]
        plies = int(tok[3].split("=")[1]) if status == "PROVED" else None
        hyps = {k: v == "1" for k, v in (t.split("=") for t in tok[4 if status != "ONEAWAY" else 5:] if "=" in t and not t.startswith("targets"))}
        out.append((sq, status, plies, cand))
    return out


def check(n: int = 2000, seed: int = 1, knights: int = 0, form: int = 1, their_pawns: int = 0, li: LeanIdeas | None = None) -> dict:
    """The bridge as a function: n random pawn endings, every PROVED claim of the detector checked by the exhaustive
    race search. -> dict(positions, proved, confirmed, wrong, oneaway, plies, examples, seconds)."""
    rng = random.Random(seed)
    t0 = time.perf_counter()
    positions = proved = oneaway = confirmed = wrong = 0
    plies_hist = {}
    examples = []
    own = li is None
    li = li or LeanIdeas()
    try:
        while positions < n:
            b = random_pawn_ending(rng, knights, their_pawns if form == 2 else 0)
            if b is None:
                continue
            positions += 1
            (line,) = li._ask(("passer2 " if form == 2 else "passer ") + encode(b), until_end=False)
            if line.startswith("error"):
                raise RuntimeError(line)
            for sq, status, plies, cand in parse(line):
                if status == "ONEAWAY":
                    oneaway += 1
                if status != "PROVED":
                    continue
                proved += 1
                plies_hist[plies] = plies_hist.get(plies, 0) + 1
                ok = promotes_within(b.copy(), chess.square_file(sq), plies)
                if ok:
                    confirmed += 1
                else:
                    wrong += 1
                    if len(examples) < 5:
                        examples.append((b.fen(), cand))
    finally:
        if own:
            li.close()
    return dict(positions=positions, proved=proved, confirmed=confirmed, wrong=wrong, oneaway=oneaway,
                plies=dict(sorted(plies_hist.items())), examples=examples, seconds=time.perf_counter() - t0)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--knights", type=int, default=0, help="our knights (allowed by the lemma: no sliders)")
    ap.add_argument("--form", type=int, default=1, choices=[1, 2], help="1: bare enemy king; 2: enemy king and pawns")
    ap.add_argument("--their-pawns", type=int, default=0, help="form 2: up to this many enemy pawns")
    a = ap.parse_args(argv)
    r = check(a.n, a.seed, a.knights, a.form, a.their_pawns)
    print(f"OBS passer bridge (form {a.form}): {r['positions']} random pawn endings (seed {a.seed}, knights {a.knights}, "
          f"their pawns <= {a.their_pawns}), "
          f"PROVED claims {r['proved']} (by plies: {r['plies']}), confirmed by exhaustive race search "
          f"{r['confirmed']}/{r['proved']}, WRONG {r['wrong']}; one-away instances {r['oneaway']}; {r['seconds']:.0f}s")
    for fen, what in r["examples"]:
        print(f"  {what}: {fen}")


if __name__ == "__main__":
    main()
