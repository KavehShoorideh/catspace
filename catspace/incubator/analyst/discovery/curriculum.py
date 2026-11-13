"""PLAN 78.22 THE MATERIAL CURRICULUM (stage 2 of the initialization phase): the engine explores the game from the smallest material upward -- the initial position with most
of its units removed (what the game cannot do without, plus a few more), the pieces standing on their starting squares -- then four units, five, ... Self-play starts from
these (wake --starts) and every later stage (cards, surprises, lemmas) carries the start.

GENERIC: a position is an opaque string the kernel accepts (`CE.state(variant, start, moves)`). The only game knowledge is a `PositionCodec` behind the registry `CODECS`
(by game name): `units(position)` lists the position's units as opaque handles in a stable order; `keep(position, units)` is the same position with only those units left,
everything that depended on a removed one made consistent. `ChessCodec` (python-chess FENs) is the one place chess appears. WHAT A GAME REQUIRES is found by test, not by
name (`required`): a unit is required when the initial position without it alone is rejected by the kernel, already over, or without legal moves.

  python -m catspace.incubator.analyst.discovery.curriculum chess --size 3 [--limit N] [--seed 1] [--out <file>]
"""
from __future__ import annotations

import argparse
import itertools
import math
import random
import sys
from typing import Protocol

import cs_engine as CE

from catspace.io import paths

MAX_ENUM = 200_000          # without a limit, a game with more candidate combinations than this is refused (sample with `limit`)


class PositionCodec(Protocol):
    """The adapter at the boundary."""

    def units(self, position: str) -> list:
        """The position's units: opaque handles, hashable, in a stable order."""

    def keep(self, position: str, units: list) -> str:
        """The same position with only `units` left, everything that depended on a removed unit made consistent."""

    def describe(self, position: str) -> list:
        """(side, kind) of each unit of `units(position)`, in the same order: the side named as the kernel names it (a mover), the kind as the description's own constant."""

    def unit_term(self, side: str, kind: str, tag: str) -> tuple:
        """(term text, [place variables]): the word-term literal "a unit of `side` (a variable or a constant) and `kind` stands at a place", its place variables named with `tag`."""

    def opposed_term(self, a: str, b: str) -> str:
        """The word-term literal "side `a` and side `b` are opposed"."""

    # optional: `kinds(position) -> list`, the kind of each unit (hashable), so a change of the kinds' multiset (a capture, a promotion) can be told from a move




class ChessCodec:
    """The chess family over python-chess FENs: a unit is a square holding a piece (handle: the square's name). `keep` clears the other squares, keeps the side to move, drops
    the en passant square, resets the clocks, and keeps a castling right only where the king and that rook still stand on their squares."""

    def units(self, position: str) -> list:
        import chess
        return [chess.square_name(sq) for sq in sorted(chess.Board(position).piece_map())]

    SIDES = {True: "white", False: "black"}
    KINDS = {"k": "king", "q": "queen", "r": "rook", "b": "bishop", "n": "knight", "p": "pawn"}

    def describe(self, position: str) -> list:
        import chess
        b = chess.Board(position)
        return [(self.SIDES[b.piece_map()[sq].color], self.KINDS[b.piece_map()[sq].symbol().lower()]) for sq in sorted(b.piece_map())]

    def unit_term(self, side: str, kind: str, tag: str) -> tuple:
        f, r = f"?f{tag}", f"?r{tag}"
        return f"(cell {f} {r} {side} {kind})", [f, r]

    def opposed_term(self, a: str, b: str) -> str:
        return f"(opponent {a} {b})"

    def kinds(self, position: str) -> list:
        import chess
        return sorted(p.symbol() for p in chess.Board(position).piece_map().values())

    def owned(self, position: str) -> list:
        """(side, kind as the kernel and a valuation record name it) of each unit: what a material balance is summed over (plan 78.37)."""
        import chess
        return [(self.SIDES[p.color], p.symbol().lower()) for _, p in sorted(chess.Board(position).piece_map().items())]

    def keep(self, position: str, units: list) -> str:
        import chess
        board = chess.Board(position)
        keep = {chess.parse_square(u) for u in units}
        for sq in list(board.piece_map()):
            if sq not in keep:
                board.remove_piece_at(sq)
        board.castling_rights = board.clean_castling_rights()
        board.ep_square = None
        board.halfmove_clock, board.fullmove_number = 0, 1
        return board.fen(en_passant="fen")


CODECS: dict = {"chess": ChessCodec()}          # by game name: the one registry (add a game by adding its codec here)


def codec_of(game: str) -> PositionCodec:
    if game not in CODECS:
        raise ValueError(f"no position codec for {game!r}; known: {sorted(CODECS)}")
    return CODECS[game]


def check(variant: str, position: str) -> "str | None":
    """None when the kernel accepts the position, it is not over and has legal moves; else why not: `rejected` | `over` | `no_moves`."""
    try:
        _fen, _mover, legal, outcome, _key = CE.state(variant, position, [])
    except Exception:
        return "rejected"
    if outcome is not None:
        return "over"
    return None if legal else "no_moves"


def initial(variant: str) -> str:
    return CE.state(variant, None, [])[0]


def required(game: str, variant: "str | None" = None) -> list:
    """The units the game cannot do without, FOUND BY TEST: removing one alone from the initial position makes the kernel reject the position, report it over, or leave no
    legal move."""
    variant, codec = variant or game, codec_of(game)
    start = initial(variant)
    units = codec.units(start)
    return [u for u in units if check(variant, codec.keep(start, [x for x in units if x != u])) is not None]


def unrank(n: int, k: int, idx: int) -> tuple:
    """The `idx`-th (from 0) k-combination of range(n) in lexicographic order (itertools.combinations' order)."""
    out, x = [], 0
    for left in range(k, 0, -1):
        while True:
            c = math.comb(n - x - 1, left - 1)
            if idx < c:
                break
            idx -= c
            x += 1
        out.append(x)
        x += 1
    return tuple(out)


def survey(game: str, size: int, *, variant: "str | None" = None, limit: "int | None" = None, seed: int = 1) -> dict:
    """-> {starts, required, size, candidates, kept, rejected {why: n}, total}: see `starts`. `candidates` is how many sub-positions were examined, `total` how many exist."""
    variant, codec = variant or game, codec_of(game)
    start = initial(variant)
    units = codec.units(start)
    req = required(game, variant)
    others = [u for u in units if u not in req]
    k = size - len(req)
    if k < 0 or k > len(others):
        raise ValueError(f"size {size} is outside what {game} allows: {len(req)} required units, {len(units)} in all")
    total = math.comb(len(others), k)
    if limit is None and total > MAX_ENUM:
        raise ValueError(f"{total} sub-positions of size {size}: pass a limit to sample them")
    if limit is None or limit >= total:
        order = ((i, c) for i, c in enumerate(itertools.combinations(range(len(others)), k)))
    else:
        rng = random.Random(seed)
        if total <= MAX_ENUM:
            draws = iter(rng.sample(range(total), total))
        else:
            def lazy():
                seen = set()
                for _ in range(50 * limit):
                    i = rng.randrange(total)
                    if i not in seen:
                        seen.add(i)
                        yield i
            draws = lazy()
        order = ((i, unrank(len(others), k, i)) for i in draws)
    kept, rejected, candidates = [], {}, 0
    for i, comb in order:
        if limit is not None and len(kept) >= limit:
            break
        candidates += 1
        pos = codec.keep(start, req + [others[j] for j in comb])
        why = check(variant, pos)
        if why is None:
            kept.append((i, pos))
        else:
            rejected[why] = rejected.get(why, 0) + 1
    kept.sort()                                                  # the deterministic order: by combination index
    return {"starts": [p for _, p in kept], "required": req, "size": size, "candidates": candidates, "kept": len(kept), "rejected": rejected, "total": total}


def starts(game: str, size: int, *, variant: "str | None" = None, limit: "int | None" = None, seed: int = 1) -> list:
    """The sub-positions of the game's initial position that keep exactly `size` units (the required ones plus `size - len(required)` others), each accepted by the kernel,
    not over, with legal moves, in a deterministic order (the combinations' lexicographic order over the units' stable order). More than `limit`: a seeded uniform sample
    of them (drawn lazily: a huge count is never enumerated). Without a limit, more than MAX_ENUM combinations is refused."""
    return survey(game, size, variant=variant, limit=limit, seed=seed)["starts"]


def main(argv: list) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("game")
    ap.add_argument("--variant", default=None)
    ap.add_argument("--size", type=int, required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default=None, help="the starts file wake --starts reads (default paths.derived(curriculum_<game>_<size>.txt))")
    a = ap.parse_args(argv)
    r = survey(a.game, a.size, variant=a.variant, limit=a.limit, seed=a.seed)
    out = a.out or paths.derived(f"curriculum_{a.game}_{a.size}.txt")
    with open(out, "w") as f:
        f.write(f"# curriculum {a.game} size {a.size} seed {a.seed} limit {a.limit}: one start position per line\n")
        f.writelines(p + "\n" for p in r["starts"])
    print(f"OBS curriculum {a.game}: size {a.size}, required {len(r['required'])} {r['required']}, combinations {r['total']}, candidates {r['candidates']}, kept {r['kept']}, "
          f"rejected {sum(r['rejected'].values())} {r['rejected']} -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
