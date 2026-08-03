"""catspace/reachgame.py -- the REACHABILITY GAME: the correct definition of "forceable"
(Kaveh 2026-07-23: "we need to better define the concepts of forceable").

A region G is FORCEABLE from s at horizon h iff White has a strategy reaching G within h
plies against a defender who avoids G BY ANY LEGAL MEANS -- not a defender playing optimal
chess (the earlier canonical-defense DFS let Black defend on autopilot, which OVERSTATES
forceability: dodging the region may require "bad" chess moves, and a real vetoing opponent
will happily play them).

  adv_reach(board, hit, h)     -> minimal forcing horizon d_adv (<= h) or None (not forceable
                                  within h). AND-OR search: White nodes EXISTS a move, Black
                                  nodes FOR-ALL moves; hit() tested at every node; memoized on
                                  (position, plies_left). Pure rules -- no tablebase, no eval.

Concept ladder (INQUIRY_TACTICS / the veto):
  d_coop  cooperative distance (both collude)        -- random-walk estimates
  d_adv   adversarial reach distance (this module)   -- forceable@h  <=>  d_adv <= h
  denied  d_adv = infinity while d_coop small        -- his cooperation is required
  veto    d_adv - d_coop                             -- what the learned channel gap estimates
"""
from __future__ import annotations

import chess


def adv_reach(board: chess.Board, hit, h: int, _memo=None) -> int | None:
    """Minimal plies within which White can FORCE a hit()-position, against a defender
    avoiding it by any legal means. None = not forceable within h. board's side to move
    may be either color (the mover at each node plays its role: White = reacher,
    Black = avoider)."""
    if _memo is None:
        _memo = {}
    if hit(board):
        return 0
    if h == 0 or board.is_game_over(claim_draw=True):
        return None
    key = (board._transposition_key(), h)
    if key in _memo:
        return _memo[key]
    _memo[key] = None                      # cycle guard: revisiting within budget = fail branch
    best = None
    if board.turn == chess.WHITE:          # reacher: EXISTS a move that keeps the guarantee
        for m in board.legal_moves:
            c = board.copy(stack=False); c.push(m)
            r = adv_reach(c, hit, h - 1, _memo)
            if r is not None and (best is None or r + 1 < best):
                best = r + 1
                if best == 1:
                    break
    else:                                  # avoider: FOR ALL moves the reacher still succeeds
        worst = -1
        for m in board.legal_moves:
            c = board.copy(stack=False); c.push(m)
            r = adv_reach(c, hit, h - 1, _memo)
            if r is None:
                worst = None; break        # the avoider has an escape -> not forceable here
            worst = max(worst, r + 1)
        best = worst if worst != -1 else None
    _memo[key] = best
    return best


def forceable_avoid(board: chess.Board, hit, h: int) -> bool:
    """True iff hit-region is forceable within h against an AVOIDING defender."""
    return adv_reach(board, hit, h) is not None


def coop_reach(board: chess.Board, hit, h: int, _memo=None) -> int | None:
    """Cooperative reach distance: min plies to a hit() position when BOTH sides collude
    (EXISTS at every node). The companion lower level of the ladder: coop-reachable but
    adv-denied = the true veto squares."""
    if _memo is None:
        _memo = {}
    if hit(board):
        return 0
    if h == 0 or board.is_game_over(claim_draw=True):
        return None
    key = (board._transposition_key(), h)
    if key in _memo:
        return _memo[key]
    _memo[key] = None
    best = None
    for m in board.legal_moves:
        c = board.copy(stack=False); c.push(m)
        r = coop_reach(c, hit, h - 1, _memo)
        if r is not None and (best is None or r + 1 < best):
            best = r + 1
            if best == 1:
                break
    _memo[key] = best
    return best


def reach_square_sets(board: chess.Board, tb, h: int, white_cap: int = 10):
    """ALL confinement squares in ONE pass (the 40-searches-in-one fix): returns
    (adv_mask, coop_mask) 64-bit ints. Bit s set in adv_mask = White can FORCE a
    still-won position with the black king within 1 of square s, inside h plies
    (union at White nodes, INTERSECTION at Black nodes); coop_mask = same under
    cooperation (union everywhere). One game-tree traversal, memoized; tb probed
    once per node (probe cache absorbs repeats)."""
    memo = {}

    def hitmask(b):
        bk = b.king(chess.BLACK)
        if bk is None:
            return 0
        w, _d = tb.wdl_dtz(b)
        w = w if b.turn == chess.WHITE else (-w if w is not None else None)
        if w != 2:
            return 0
        m = 1 << bk
        for s in chess.SquareSet(chess.BB_KING_ATTACKS[bk]):
            m |= 1 << s
        return m

    def rec(b, d):
        hm = hitmask(b)
        if d == 0 or b.is_game_over(claim_draw=True):
            return hm, hm
        key = (b._transposition_key(), d)
        if key in memo:
            a, c = memo[key]
            return a | hm, c | hm
        memo[key] = (0, 0)                       # cycle guard
        if b.turn == chess.WHITE:
            # SOUND cap: a union over FEWER White moves is still a guarantee (green never
            # lies; worst case we miss a forceable square). Tactical-first ordering keeps
            # coverage. Black's side below stays EXHAUSTIVE (intersection soundness).
            moves = list(b.legal_moves)
            if len(moves) > white_cap:
                moves.sort(key=lambda m: not (b.is_capture(m) or b.gives_check(m)))
                moves = moves[:white_cap]
            A = C = 0
            for m in moves:
                nb = b.copy(stack=False); nb.push(m)
                a, c = rec(nb, d - 1)
                A |= a; C |= c
        else:
            A = None; C = 0
            for m in b.legal_moves:
                nb = b.copy(stack=False); nb.push(m)
                a, c = rec(nb, d - 1)
                A = a if A is None else (A & a)
                C |= c
            A = A if A is not None else 0
        A |= hm; C |= hm
        memo[key] = (A, C)
        return A, C

    return rec(board, h)


def king_zone_won(square: int, tb, zone_tol: int = 1):
    """Per-square confinement predicate: black king within zone_tol of `square` AND the
    position still tablebase-won for the attacker. Cheap zone check first; tb probed only
    on zone hits (probe cache absorbs repeats)."""
    def hit(b: chess.Board) -> bool:
        bk = b.king(chess.BLACK)
        if bk is None or chess.square_distance(bk, square) > zone_tol:
            return False
        w, _d = tb.wdl_dtz(b)
        if w is None:
            return False
        return (w if b.turn == chess.WHITE else -w) == 2
    return hit


def region_neighborhood(g: chess.Board, bk_tol: int = 1, wk_tol: int = 2):
    """The goal-as-region predicate (canonical home; measure_adversarial_veto has the
    original): same material signature + black king within bk_tol + white king within wk_tol."""
    mat = "".join(sorted(p.symbol() for p in g.piece_map().values()))
    gbk, gwk = g.king(chess.BLACK), g.king(chess.WHITE)

    def hit(b: chess.Board) -> bool:
        if "".join(sorted(p.symbol() for p in b.piece_map().values())) != mat:
            return False
        bk, wk = b.king(chess.BLACK), b.king(chess.WHITE)
        return (bk is not None and wk is not None
                and chess.square_distance(bk, gbk) <= bk_tol
                and chess.square_distance(wk, gwk) <= wk_tol)
    return hit


def exact_position(g: chess.Board):
    """Exact-target predicate (transposition identity) -- the 'aim at this precise
    diagram' plan, for the exact-vs-region contrast."""
    key = g._transposition_key()
    return lambda b: b._transposition_key() == key
