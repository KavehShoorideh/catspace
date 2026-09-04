"""L0 -- GEOMETRY. The board as a graph: squares, direction families, rays, distances,
symmetries. Rules-agnostic. Board size/shape are PARAMETERS here, not a layer above. A thin,
named interface over python-chess's square algebra so that higher layers never handle raw
integers with chess-specific meaning."""
from __future__ import annotations

import chess

FILES = 8
RANKS = 8
SQUARES = tuple(chess.SQUARES)

# direction families as (dfile, drank) unit steps
ORTHOGONAL = ((0, 1), (0, -1), (1, 0), (-1, 0))
DIAGONAL = ((1, 1), (1, -1), (-1, 1), (-1, -1))
KNIGHT_JUMPS = ((1, 2), (2, 1), (-1, 2), (-2, 1), (1, -2), (2, -1), (-1, -2), (-2, -1))


def step(sq: int, dfile: int, drank: int) -> int | None:
    f, r = chess.square_file(sq) + dfile, chess.square_rank(sq) + drank
    return chess.square(f, r) if 0 <= f < FILES and 0 <= r < RANKS else None


def ray(sq: int, dfile: int, drank: int) -> list[int]:
    """Squares from sq along one unit step until the edge (sq excluded)."""
    out, cur = [], step(sq, dfile, drank)
    while cur is not None:
        out.append(cur)
        cur = step(cur, dfile, drank)
    return out


def between(a: int, b: int) -> chess.SquareSet:
    return chess.SquareSet.between(a, b)


def aligned(a: int, b: int) -> str | None:
    df, dr = abs(chess.square_file(a) - chess.square_file(b)), abs(chess.square_rank(a) - chess.square_rank(b))
    if a == b:
        return None
    if df == 0 or dr == 0:
        return "orthogonal"
    if df == dr:
        return "diagonal"
    return None


def distance(a: int, b: int) -> int:
    return chess.square_distance(a, b)


def manhattan(a: int, b: int) -> int:
    return chess.square_manhattan_distance(a, b)


def mirror_rank(sq: int) -> int:
    return chess.square_mirror(sq)


def square_color(sq: int) -> str:
    return "light" if (chess.square_file(sq) + chess.square_rank(sq)) % 2 else "dark"


def name(sq: int) -> str:
    return chess.square_name(sq)
