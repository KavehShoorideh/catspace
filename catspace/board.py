"""
Shared board geometry and move utilities for 5x5 toy chess domains.
"""
from __future__ import annotations

N = 5
NSQ = N * N


def sq(r: int, c: int) -> int:
    return r * N + c


def rc(s: int) -> tuple[int, int]:
    return divmod(s, N)


def chebyshev(a: int, b: int) -> int:
    ra, ca = rc(a)
    rb, cb = rc(b)
    return max(abs(ra - rb), abs(ca - cb))


def is_adjacent(a: int, b: int) -> bool:
    return chebyshev(a, b) <= 1


def distinct_positions(*positions: int) -> bool:
    return len(set(positions)) == len(positions)


def king_moves(s: int) -> list[int]:
    r, c = rc(s)
    out: list[int] = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            rr, cc = r + dr, c + dc
            if 0 <= rr < N and 0 <= cc < N:
                out.append(sq(rr, cc))
    return out

KING_MOVES = [king_moves(s) for s in range(NSQ)]


def rook_slides(rook: int, blockers: set[int]) -> list[int]:
    r, c = rc(rook)
    out: list[int] = []
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        rr, cc = r + dr, c + dc
        while 0 <= rr < N and 0 <= cc < N:
            t = sq(rr, cc)
            if t in blockers:
                break
            out.append(t)
            rr += dr
            cc += dc
    return out


def rook_moves(rook: int, wk: int) -> list[int]:
    """Rook slides; blocked by own king (cannot pass through or land on it)."""
    return rook_slides(rook, {wk})


def rook_attacks(rook: int, target: int, blockers: set[int]) -> bool:
    r, c = rc(rook)
    tr, tc = rc(target)
    if r != tr and c != tc:
        return False
    if r == tr:
        lo, hi = sorted((c, tc))
        return not any(sq(r, x) in blockers for x in range(lo + 1, hi))
    lo, hi = sorted((r, tr))
    return not any(sq(x, c) in blockers for x in range(lo + 1, hi))


def knight_moves(s: int) -> list[int]:
    r, c = rc(s)
    out: list[int] = []
    for dr, dc in ((1, 2), (2, 1), (-1, 2), (-2, 1), (1, -2), (2, -1), (-1, -2), (-2, -1)):
        rr, cc = r + dr, c + dc
        if 0 <= rr < N and 0 <= cc < N:
            out.append(sq(rr, cc))
    return out

KNIGHT = [knight_moves(s) for s in range(NSQ)]
KN_ATT = [set(m) for m in KNIGHT]
