"""The CHESS ADAPTER for lemma_gen: the only file that knows what a piece is. It supplies the rules and nothing
else -- entity types, relations as programs over a board, actions and the transition, a scalar (material), and
placements (every board holding exactly the frame's pieces plus both kings, canonicalized by the board's symmetry).
No pawns in this first version: the eightfold symmetry then holds and the frame is orientation-free."""
from __future__ import annotations

import itertools
import random

import chess

from catspace.io import paths as _paths
import chess.syzygy

from .lemma_gen import Domain, Relation

VAL = {chess.PAWN: 1.0, chess.KNIGHT: 3.0, chess.BISHOP: 3.0, chess.ROOK: 5.0, chess.QUEEN: 9.0, chess.KING: 100.0}
KIND = {"K": chess.KING, "Q": chess.QUEEN, "R": chess.ROOK, "B": chess.BISHOP, "N": chess.KNIGHT, "P": chess.PAWN}
TYPES = ["wK", "wQ", "wR", "wB", "wN", "wP", "bK", "bQ", "bR", "bB", "bN", "bP"]   # pawns are entities too (2026-09-23: most real-board duties are about pawns)
CANON = [chess.A1, chess.B1, chess.C1, chess.D1, chess.B2, chess.C2, chess.D2, chess.C3, chess.D3, chess.D4]


class State:
    __slots__ = ("board", "ids")

    def __init__(self, board: chess.Board, ids: list):
        self.board = board; self.ids = ids                                # ids = the slot entities' squares, in frame order

    def fen(self):
        return self.board.fen()


def _piece(t: str) -> chess.Piece:
    return chess.Piece(KIND[t[1]], t[0] == "w")


_SHAPES = {"on": True}


def shape_key(squares: tuple) -> tuple:
    """A configuration up to translation and left-right mirror (the two spatial symmetries the rules allow: pawns fix
    the vertical direction), TOGETHER with the edges it touches: a king on the back rank is not the same closed world
    as a king in the middle (Kaveh 2026-09-24: back-rank mates exist because the edge takes the escape squares).
    Kaveh's "infinite chessboard" holds away from the edges; at the edge the board itself is a piece of the frame."""
    fs = [chess.square_file(s) for s in squares]; rs = [chess.square_rank(s) for s in squares]
    f0, r0 = min(fs), min(rs); w = max(fs) - f0
    edges = (min(fs) == 0 or max(fs) == 7, min(rs) == 0, max(rs) == 7)      # a side edge (mirror-invariant), the bottom rank, the top rank
    a = tuple((f - f0, r - r0) for f, r in zip(fs, rs))
    b = tuple((w - (f - f0), r - r0) for f, r in zip(fs, rs))
    return (min(a, b), edges)


def centred(squares: tuple) -> tuple:
    """The same configuration moved so its bounding box sits as centrally as the board allows: the edges become
    'the rest of the board' as far as the shape permits."""
    fs = [chess.square_file(s) for s in squares]; rs = [chess.square_rank(s) for s in squares]
    f0, r0 = min(fs), min(rs); w = max(fs) - f0; h = max(rs) - r0
    df = (7 - w) // 2 - f0; dr = (7 - h) // 2 - r0
    return tuple(chess.square(f + df, r + dr) for f, r in zip(fs, rs))


def placements(types: tuple, cap: int, seed: int):
    """Boards holding exactly the frame's entities plus both kings (added if the frame does not name them). The
    first slot's square is fixed to the ten canonical squares (symmetry). Exhaustive when the raw count fits the
    cap, else a uniform sample with the flag False. White to move; actions flip the turn to the actor's side."""
    abstract = [t for t in types if t not in TYPES]
    if abstract:                                                      # TOP-DOWN types (Kaveh 2026-09-24): an abstract slot is drawn as one of its leaf types per placement
        rng0 = random.Random(seed + 7)
        pools = [leaves_of(t) for t in types]
        if all(len(pl) == 1 for pl in pools):
            pools = None
        if pools is not None:
            yielded = 0; attempts = 0
            while yielded < cap and attempts < cap * 50:
                attempts += 1
                concrete = tuple(rng0.choice(pl) for pl in pools)
                got = 0
                for st, _ in placements(concrete, 1, rng0.randrange(1 << 30)):
                    yield st, False; yielded += 1; got += 1
                    break
            return
    pieces = [_piece(t) for t in types]
    if sum(1 for p in pieces if p.piece_type == chess.KING and p.color) > 1 or sum(1 for p in pieces if p.piece_type == chess.KING and not p.color) > 1:
        return                                                        # a domain rule: one king a side; such a frame has no placements
    extra = []
    if not any(p.piece_type == chess.KING and p.color for p in pieces):
        extra.append(chess.Piece(chess.KING, True))
    if not any(p.piece_type == chess.KING and not p.color for p in pieces):
        extra.append(chess.Piece(chess.KING, False))
    allp = pieces + extra
    n = len(allp)
    raw = len(CANON) * 1
    for i in range(1, n):
        raw *= (64 - i)
    exhaustive = raw <= cap * 2
    rng = random.Random(seed)
    yielded = 0; attempts = 0
    if exhaustive:
        it = ((c,) + rest for c in CANON for rest in itertools.permutations([s for s in range(64)], n - 1) if c not in rest)
    else:
        def sampler():
            while True:
                c = rng.choice(CANON)
                rest = rng.sample([s for s in range(64) if s != c], n - 1)
                yield (c,) + tuple(rest)
        it = sampler()
    seen_shapes = set()
    for squares in it:
        attempts += 1
        if attempts > cap * 200:
            return                                                    # bounded, never a loop: nothing valid in many draws, or every shape already seen
        if not exhaustive and yielded >= cap:
            return
        if exhaustive and yielded >= cap * 2:
            return
        if _SHAPES["on"]:
            key = shape_key(tuple(squares))
            if key in seen_shapes:
                continue                                              # one placement per shape: translation and mirror images are the same closed world
            seen_shapes.add(key)
            if not any(key[1]):
                squares = centred(tuple(squares))                     # away from every edge the placement is free to sit centrally; touching an edge, it stays where it is
        b = chess.Board(None)
        for sq, p in zip(squares, allp):
            b.set_piece_at(sq, p)
        b.turn = chess.WHITE
        if not b.is_valid():
            continue
        if b.is_check():
            continue
        # the black king must not be attacked either (a frame is a quiet configuration for both sides)
        b2 = b.copy(stack=False); b2.turn = chess.BLACK
        if not b2.is_valid() or b2.is_check():
            continue
        yielded += 1
        yield State(b, list(squares[:len(types)])), exhaustive


def placement_count(types: tuple) -> int:
    """How many placements `placements` would enumerate exhaustively (before legality filtering): the raw count."""
    if any(t not in TYPES for t in types):
        return 10 ** 12                                                       # abstract frames are never enumerated exhaustively
    pieces = [_piece(t) for t in types]
    n = len(pieces) + (0 if any(p.piece_type == chess.KING and p.color for p in pieces) else 1) + (0 if any(p.piece_type == chess.KING and not p.color for p in pieces) else 1)
    raw = len(CANON)
    for i in range(1, n):
        raw *= (64 - i)
    return raw


def slots_of(state: State):
    return state.ids


def actions(state: State, e: int):
    b = state.board.copy(stack=False); b.turn = state.board.piece_at(e).color
    if not b.is_valid():
        return []
    return [(b, m) for m in b.legal_moves if m.from_square == e]


def transition(state: State, action):
    b, m = action
    nb = b.copy(stack=False); nb.push(m)
    moved = {m.from_square: m.to_square}
    if b.is_castling(m):                                                      # the rook moves too: its id follows it (a rule of the game, not a lemma)
        rank = chess.square_rank(m.from_square)
        if chess.square_file(m.to_square) > chess.square_file(m.from_square):
            moved[chess.square(7, rank)] = chess.square(5, rank)
        else:
            moved[chess.square(0, rank)] = chess.square(3, rank)
    captured = {m.to_square} if b.is_capture(m) else set()
    if b.is_en_passant(m):
        captured = {chess.square(chess.square_file(m.to_square), chess.square_rank(m.from_square))}
    ids = [moved.get(s, None if s in captured else s) for s in state.ids]
    return State(nb, ids)


def _material(board: chess.Board, side: chess.Color) -> float:
    return sum(VAL[p.piece_type] for p in board.piece_map().values() if p.color == side and p.piece_type != chess.KING) - \
           sum(VAL[p.piece_type] for p in board.piece_map().values() if p.color != side and p.piece_type != chess.KING)


def scalar(state: State, e: int) -> float:
    p = state.board.piece_at(e)
    return _material(state.board, p.color)


def reply_scalar(after: State, e: int) -> float:
    """The entity's side's material after the opponent's best immediate reply (the reply that minimizes it)."""
    b = after.board
    side = not b.turn                                                 # the actor's side: it just moved
    if b.is_checkmate():                                              # NOT is_game_over(): insufficient material would end K+B vs K before the king captures (2026-09-22)
        return _material(b, side) + (-1000.0 if b.turn == side else 1000.0)
    if b.is_stalemate():
        return _material(b, side)
    worst = None
    for r in b.legal_moves:
        if not (b.is_capture(r) or b.gives_check(r)):
            continue                                                  # only a capture or a check can lower the actor's material within one reply
        nb = b.copy(stack=False); nb.push(r)
        v = _material(nb, side) + (-1000.0 if nb.is_checkmate() else 0.0)
        worst = v if worst is None else min(worst, v)
    return worst if worst is not None else _material(b, side)


# ---------------------------------------------------------------- relations: programs over the board, nothing named
def _alive(state, *ids):
    return all(i is not None and state.board.piece_at(i) is not None for i in ids)


def same_side(state, i, j):
    return state.board.piece_at(i).color == state.board.piece_at(j).color


def attacks(state, i, j):
    return j in state.board.attacks(i)


def defends(state, i, j):
    return same_side(state, i, j) and j in state.board.attacks(i)


def aligned(state, i, j):
    return bool(chess.SquareSet.ray(i, j)) and not any(state.board.piece_at(s) for s in chess.SquareSet.between(i, j))


def between(state, i, j, k):
    """j on the segment i..k, nothing else between."""
    if not chess.SquareSet.ray(i, k) or j not in chess.SquareSet.between(i, k):
        return False
    return not any(state.board.piece_at(s) for s in chess.SquareSet.between(i, k) if s != j)


def worth_ge(state, i, j):
    return VAL[state.board.piece_at(i).piece_type] >= VAL[state.board.piece_at(j).piece_type]


def tb_worth(board: chess.Board, sq: int) -> float:
    """A piece's worth from the tables, no value table (Kaveh 2026-09-23: lemmas within five men use tablebase values,
    not canonical piece values): the expected score its own side loses when the piece is removed, with that side to
    move, in expected-score units (WDL -2..2 -> 0..1). Kings are worth 1."""
    p = board.piece_at(sq)
    if p is None:
        return 0.0
    if p.piece_type == chess.KING:
        return 1.0
    def score(b):
        try:
            return (tablebase().probe_wdl(b) + 2) / 4.0
        except (KeyError, chess.syzygy.MissingTableError):
            return 0.5
    b0 = board.copy(stack=False); b0.turn = p.color; b0.castling_rights = 0; b0.ep_square = None
    if not b0.is_valid():
        return 0.0
    b1 = b0.copy(stack=False); b1.remove_piece_at(sq)
    if not b1.is_valid():
        return 0.0
    return score(b0) - score(b1)


_BLEND = dict(tau=4.0, samples=6, cache={})


def blend_worth(board: chess.Board, sq: int) -> float:
    """A piece's worth anywhere on the board (Kaveh 2026-09-23: "we want both ... a tunable variable going from
    canonical values to tablebase values depending on how many more pieces are left to reach the tablebase edge").
    TABLES side: the mean, over sampled sub-frames of at most five men that contain the piece and both kings (the
    position itself at the edge), of the expected score the piece's side loses when the piece is removed. CANONICAL
    side: the value table, normalized to expected-score units (a queen = 1). Blend weight on the tables
    lam = exp(-(men - 5) / tau); lam = 1 at the edge. Cached per (position, square)."""
    import math, random as _r
    p = board.piece_at(sq)
    if p is None:
        return 0.0
    if p.piece_type == chess.KING:
        return 1.0
    canon = VAL[p.piece_type] / VAL[chess.QUEEN]
    men = len(board.piece_map())
    extra = men - 5
    lam = math.exp(-max(0, extra) / _BLEND["tau"]) if extra > 0 else 1.0
    if lam < 1e-3:
        return canon
    key = (board.board_fen(), board.turn, sq)
    cache = _BLEND["cache"]
    if key in cache:
        tbv = cache[key]
    else:
        others = [s for s, q in board.piece_map().items() if q.piece_type != chess.KING and s != sq and q.piece_type != chess.PAWN]
        kings = [s for s, q in board.piece_map().items() if q.piece_type == chess.KING]
        rng = _r.Random(hash(key) & 0xFFFF)
        vals = []
        for _ in range(_BLEND["samples"] if extra > 0 else 1):
            keep = set(kings) | {sq} | set(rng.sample(others, min(len(others), 5 - len(kings) - 1))) if extra > 0 else set(board.piece_map())
            b = chess.Board(None)
            for s_ in keep:
                b.set_piece_at(s_, board.piece_at(s_))
            b.turn = board.turn; b.castling_rights = 0; b.ep_square = None
            if any(q.piece_type == chess.PAWN for q in b.piece_map().values()):
                continue
            v = tb_worth(b, sq)
            vals.append(v)
        tbv = sum(vals) / len(vals) if vals else canon
        if len(cache) > 200000:
            cache.clear()
        cache[key] = tbv
    return lam * tbv + (1 - lam) * canon


def worth_ge_blend(state, i, j):
    return blend_worth(state.board, i) >= blend_worth(state.board, j)


def captures_worth_ge_blend(state, action, j):
    b, m = action
    if _captured(b, m) is None:
        return False
    sq = chess.square(chess.square_file(m.to_square), chess.square_rank(m.from_square)) if b.is_en_passant(m) else m.to_square
    return blend_worth(b, sq) >= blend_worth(b, j)


def tb_worth_ge(state, i, j):
    return tb_worth(state.board, i) >= tb_worth(state.board, j)


def tb_captures_worth_ge(state, action, j):
    b, m = action
    if not b.is_capture(m):
        return False
    return tb_worth(b, m.to_square) >= tb_worth(b, j)


def attacked(state, i):
    p = state.board.piece_at(i)
    return bool(state.board.attackers(not p.color, i))


def defended(state, i):
    p = state.board.piece_at(i)
    return bool(state.board.attackers(p.color, i))


def slides(state, i):
    return state.board.piece_at(i).piece_type in (chess.QUEEN, chess.ROOK, chess.BISHOP)


# action-level: fn(state, action, *slots); the action is (board, move)
def gives_check(state, action):
    b, m = action; nb = b.copy(stack=False); nb.push(m); return nb.is_check()


def captures(state, action):
    b, m = action; return b.is_capture(m)


def _captured(b: chess.Board, m: chess.Move):
    """The piece a move captures (en passant takes the pawn beside the destination), or None."""
    if not b.is_capture(m):
        return None
    if b.is_en_passant(m):
        return b.piece_at(chess.square(chess.square_file(m.to_square), chess.square_rank(m.from_square)))
    return b.piece_at(m.to_square)


def captures_worth_ge(state, action, j):
    b, m = action
    p = _captured(b, m)
    if p is None:
        return False
    return VAL[p.piece_type] >= VAL[b.piece_at(j).piece_type]


def stays_aligned(state, action, j):
    """After the action the mover is still on a line with slot j (nothing between)."""
    b, m = action
    if m.to_square == j:
        return False
    nb = b.copy(stack=False); nb.push(m)
    return bool(chess.SquareSet.ray(m.to_square, j)) and not any(nb.piece_at(s) for s in chess.SquareSet.between(m.to_square, j))


def attacked_by_lower(state, i):
    """SQUARE CONDITIONS (Kaveh 2026-09-27: "whether or not a square is attacked by a piece of higher value or lower
    value ... generic enough that can be used as an interface to the outside world"): an enemy piece worth less than
    this one attacks it (whoever else is around, the first capture already wins)."""
    p = state.board.piece_at(i)
    return any(VAL[state.board.piece_at(a).piece_type] < VAL[p.piece_type] for a in state.board.attackers(not p.color, i))


def outnumbered(state, i):
    """More enemy pieces attack it than its own pieces defend it."""
    p = state.board.piece_at(i)
    return len(state.board.attackers(not p.color, i)) > len(state.board.attackers(p.color, i))


def last_attacker(state, i, j):
    """i attacks j and no other piece of i's side does: nothing comes after i (Kaveh: "is the bishop the last piece
    attacking c5? No ... there's another attacker coming afterwards")."""
    b = state.board; pi = b.piece_at(i); pj = b.piece_at(j)
    if pi is None or pj is None or pi.color == pj.color:
        return False
    att = b.attackers(pi.color, j)
    return i in att and len(att) == 1


def lands_attacked(state, action):
    b, m = action; nb = b.copy(stack=False); nb.push(m)
    return bool(nb.attackers(nb.turn, m.to_square))


def _quiesce(b: chess.Board, side: chess.Color, alpha: float, beta: float, depth: int) -> float:
    """The RESOLVED EXCHANGE: material from `side` after captures (and promotions) settle, both sides choosing, standing
    pat allowed except in check. The opponent's reply in the generator's outcome becomes this instead of one capture
    (2026-09-23: the surprise gauge's 121 contradictions were trades onto defended squares; three methods pointed at
    "what happens to the piece after it lands"). Rules only: captures, checks, material."""
    stand = _material(b, side) + ((-1000.0 if b.turn == side else 1000.0) if b.is_checkmate() else 0.0)
    if b.is_checkmate() or b.is_stalemate() or depth == 0:
        return stand
    us = b.turn == side
    if us:
        if stand >= beta:
            return stand
        alpha = max(alpha, stand)
    else:
        if stand <= alpha:
            return stand
        beta = min(beta, stand)
    moves = [m for m in b.legal_moves if b.is_capture(m) or m.promotion] if not b.is_check() else list(b.legal_moves)
    moves.sort(key=lambda m: -(VAL[b.piece_at(m.to_square).piece_type] if b.piece_at(m.to_square) else 0.0))
    best = stand
    for m in moves:
        b.push(m); v = _quiesce(b, side, alpha, beta, depth - 1); b.pop()
        if us:
            best = max(best, v); alpha = max(alpha, best)
        else:
            best = min(best, v); beta = min(beta, best)
        if alpha >= beta:
            break
    return best


def resolved_reply_scalar(after: State, e: int) -> float:
    """The actor's side's material after the opponent's best reply, the exchanges RESOLVED (captures both ways, up to
    six plies). A capture onto a defended square reads as a trade, a hanging piece as a loss, a safe check as a check."""
    b = after.board
    side = not b.turn
    if b.is_checkmate():
        return _material(b, side) + (-1000.0 if b.turn == side else 1000.0)
    if b.is_stalemate():
        return _material(b, side)
    worst = None
    for r in b.legal_moves:
        if not (b.is_capture(r) or b.gives_check(r)):
            continue
        nb = b.copy(stack=False); nb.push(r)
        v = _quiesce(nb, side, -1e9, 1e9, 6)
        worst = v if worst is None else min(worst, v)
    return worst if worst is not None else _material(b, side)


def resolved_null_reply_scalar(state: State, e: int) -> float:
    b = state.board.copy(stack=False)
    side = b.piece_at(e).color
    b = passed(b, side)
    if b is None:
        return None
    return resolved_reply_scalar(State(b, list(state.ids)), e)


def _frame_material(state: State, side: chess.Color) -> float:
    """Material counted over the FRAME'S pieces only (state.ids), from `side`: the rest of the board is present but
    uncounted (Kaveh 2026-09-23: "a lemma should assert a per-frame outcome ... the interaction with the rest of the
    board is how the frame grows")."""
    b = state.board; tot = 0.0
    for sq in set(state.ids):
        if sq is None:
            continue                                                          # a captured frame piece
        p = b.piece_at(sq)
        if p is None or p.piece_type == chess.KING:
            continue
        tot += VAL[p.piece_type] if p.color == side else -VAL[p.piece_type]
    return tot


def frame_scalar(state: State, e: int) -> float:
    return _frame_material(state, state.board.piece_at(e).color)


def frame_reply_scalar(after: State, e: int, depth: int | None = None) -> float:
    """The frame's material from the actor's side after the opponent's best immediate reply (the reply that minimizes
    it): a capture of a frame piece counts, a capture elsewhere does not; a checkmate counts as +-1000. With an
    episode depth above 1, the k-ply frame episode instead (depth 0: the frame's size; None: the module default)."""
    b = after.board
    side = not b.turn
    depth = _EPISODE["depth"] if depth is None else depth
    depth = depth if depth > 0 else len(after.ids)                            # 0: each frame piece gets its move (Kaveh 2026-09-24)
    if depth > 1:
        return frame_episode_scalar(after, side, depth)
    if b.is_checkmate():
        return _frame_material(after, side) + (-1000.0 if b.turn == side else 1000.0)
    if b.is_stalemate():
        return _frame_material(after, side)
    worst = None
    for r in b.legal_moves:
        if not (b.is_capture(r) or b.gives_check(r)):
            continue
        nb = b.copy(stack=False); nb.push(r)
        ids = [r.to_square if s == r.from_square else (None if s == r.to_square else s) for s in after.ids]   # a frame piece the reply captures is gone
        v = _frame_material(State(nb, ids), side) + (-1000.0 if nb.is_checkmate() else 0.0)
        worst = v if worst is None else min(worst, v)
    return worst if worst is not None else _frame_material(after, side)


_EPISODE = {"depth": 1}                                                         # plies after the action the frame outcome searches (1 = one reply)


def _ids_after(ids: list, b: chess.Board, m: chess.Move) -> list:
    moved = {m.from_square: m.to_square}
    if b.is_castling(m):
        rank = chess.square_rank(m.from_square)
        moved[chess.square(7, rank) if chess.square_file(m.to_square) > chess.square_file(m.from_square) else chess.square(0, rank)] = chess.square(5, rank) if chess.square_file(m.to_square) > chess.square_file(m.from_square) else chess.square(3, rank)
    captured = {m.to_square} if b.is_capture(m) else set()
    if b.is_en_passant(m):
        captured = {chess.square(chess.square_file(m.to_square), chess.square_rank(m.from_square))}
    return [moved.get(s, None if s in captured else s) for s in ids]


def passed(board: chess.Board, side: chess.Color | None = None) -> chess.Board | None:
    """THE PASS (the null move) by `side` (default: the side to move): the turn goes to the opponent and the
    en-passant right lapses, as after any real move and as the compiled core's swap_turn does. None when `side` is in
    check (standing still is illegal). A bare turn flip kept the en-passant square, which python-chess calls invalid,
    so every pass after a double pawn push read as illegal (review 2026-09-24)."""
    b = board.copy(stack=False)
    if side is not None:
        b.turn = side
    if b.is_check():
        return None
    b.turn = not b.turn; b.ep_square = None
    return b if b.is_valid() else None


def _episode(st: State, side: chess.Color, depth: int, alpha: float, beta: float, want_pv: bool) -> tuple:
    """THE FRAME EPISODE (Kaveh 2026-09-24: "k pieces in a frame imply as many plies -- each piece makes its move in
    its right time"): after the action, up to `depth` plies; the side to move may PASS (the null move: the line goes
    on, it does not end -- standing pat that ends the line let a side freeze a favourable count, 22:20), capture or
    check with any piece, or make a quiet move with a FRAME piece; the frame's material from `side` at the leaves,
    mate +-1000. Returns (value, line of states) -- the line only when want_pv."""
    b = st.board
    mate = (-1000.0 if b.turn == side else 1000.0) if b.is_checkmate() else 0.0
    stand = _frame_material(st, side) + mate
    if depth == 0 or b.is_checkmate() or b.is_stalemate():
        return stand, []
    us = b.turn == side
    frame_sq = {x for x in st.ids if x is not None}
    children = []
    if b.is_check():
        moves = list(b.legal_moves)
    else:
        moves = [m for m in b.legal_moves if b.is_capture(m) or m.promotion or b.gives_check(m) or (depth > 1 and m.from_square in frame_sq)]
        nb = passed(b)
        if nb is not None:
            children.append(State(nb, list(st.ids)))                         # the pass
    moves.sort(key=lambda m: -(VAL[b.piece_at(m.to_square).piece_type] if b.piece_at(m.to_square) else 0.0))
    for m in moves:
        nb = b.copy(stack=False); nb.push(m)
        children.append(State(nb, _ids_after(st.ids, b, m)))
    best = None; best_pv = []
    for child in children:
        v, pv = _episode(child, side, depth - 1, alpha, beta, want_pv)
        if best is None or (v > best if us else v < best):
            best = v; best_pv = ([child] + pv) if want_pv else []
        if us:
            alpha = max(alpha, best)
        else:
            beta = min(beta, best)
        if alpha >= beta:
            break
    return (stand, []) if best is None else (best, best_pv)


def frame_episode_scalar(st: State, side: chess.Color, depth: int) -> float:
    return _episode(st, side, depth, -1e9, 1e9, False)[0]


def frame_episode_pv(st: State, side: chess.Color, depth: int) -> tuple:
    """(value, [states after each ply of the episode's line]); same search as `frame_episode_scalar`."""
    return _episode(st, side, depth, -1e9, 1e9, True)


def frame_episode_states(after: State, depth: int | None = None) -> list:
    """The domain's EPISODE hook: from the state after our action, the states along the episode's line, one per ply
    (the opponent's reply first). Empty when the episode has no ply beyond the last (depth < 3), so a lifted
    relation can never restate the outcome."""
    depth = _EPISODE["depth"] if depth is None else depth
    depth = depth if depth > 0 else len(after.ids)
    if depth < 3:
        return []
    side = not after.board.turn
    _, pv = frame_episode_pv(after, side, depth)
    return pv[:-1]                                                            # every state before the last ply


def frame_threat_state(state: State, e: int) -> State | None:
    """The threat executed: e's side passes, the opponent plays the best reply of the frame episode (one ply). None
    when the pass is illegal (in check). The opponent's pass leaves the board as it is: no threat."""
    b = state.board.copy(stack=False)
    side = b.piece_at(e).color if b.piece_at(e) else None
    if side is None:
        return None
    b = passed(b, side)
    if b is None:
        return None
    after_pass = State(b, list(state.ids))
    _, pv = frame_episode_pv(after_pass, side, 1)
    return pv[0] if pv else after_pass


def frame_null_reply_scalar(state: State, e: int, depth: int | None = None) -> float:
    b = state.board.copy(stack=False)
    side = b.piece_at(e).color
    b = passed(b, side)
    if b is None:
        return None
    return frame_reply_scalar(State(b, list(state.ids)), e, depth)


def null_reply_scalar(state: State, e: int) -> float:
    """The actor's side's material after the opponent's best immediate reply if the actor PASSES (the null move): what
    the standing threats already take. On a full board the opponent can nearly always capture something, so an
    outcome must be read relative to this or almost every move "loses" (gauge_lemmas, 2026-09-23). Undefined (None)
    when the actor is in check: there is no null move."""
    b = state.board.copy(stack=False)
    side = b.piece_at(e).color
    b = passed(b, side)
    if b is None:
        return None
    return reply_scalar(State(b, list(state.ids)), e)


KIND_NAME = {v: k for k, v in KIND.items()}


def contests(board: chess.Board, max_pieces: int = 4) -> list:
    """THE HUMAN ATOM (Kaveh 2026-09-24, from the reasoning exercise: "what attacks this piece, what defends it, is
    that enough"): every attacked piece with its attackers and defenders, nearest by worth first, capped. Returns
    [(priority, types at the KIND level, ids, literals)] with literals attacks(i,0) / defends(j,0). Priority: the
    king in check first, then undefended attacked pieces, then contests where attackers and defenders balance,
    then the rest -- the order of the questions, computable from the rules."""
    out = []
    for sq, p in board.piece_map().items():
        att = sorted(board.attackers(not p.color, sq), key=lambda a: VAL[board.piece_at(a).piece_type])
        if not att:
            continue
        dfd = sorted(board.attackers(p.color, sq), key=lambda a: VAL[board.piece_at(a).piece_type])
        ids = [sq]; lits = []
        for a in att:
            if len(ids) >= max_pieces: break
            lits.append(("attacks", (len(ids), 0), True)); ids.append(a)
        for d in dfd:
            if len(ids) >= max_pieces: break
            lits.append(("defends", (len(ids), 0), True)); ids.append(d)
        if p.piece_type == chess.KING:
            pri = 0
        elif not dfd:
            pri = 1
        elif len(att) >= len(dfd):
            pri = 2
        else:
            pri = 3
        types = tuple(KIND_NAME[board.piece_at(i).piece_type] for i in ids)
        out.append((pri, types, ids, tuple(lits)))
    return out


def game_placements(fens: list, per_position: int = 2, try_factor: int = 50):
    """A placements source drawn from REAL positions (games): for a frame's types, every assignment of distinct pieces
    of those types on each position, in a seeded order, up to the cap; never exhaustive. The rest of the board stays,
    so a lemma learned here is learned where it is applied.

    With `accept` (the frame's literals) the cap counts ACCEPTED placements -- the frame's evidence -- at most
    `per_position` from one position, looking at no more than cap * try_factor assignments, and a board is copied
    only when it is accepted. Without it the cap counts assignments (relation discovery draws plain placements).
    Review 2026-09-24: with the cap counting tries and one shuffled order for every frame, all frames read the same
    few boards (full2's size-3 lemmas: a median of 22 positions) and a rare frame got no rows at any corpus size."""
    import random as _r

    boards: dict = {}                                                   # parsed once per process (fork copies what the parent parsed)

    def placements(types: tuple, cap: int, seed: int, accept=None):
        rng = _r.Random(seed)
        order = list(range(len(fens))); rng.shuffle(order)
        scan = getattr(placements, "scan_limit", None)                   # the frequency count scans a bounded prefix of the positions
        if scan:
            order = order[:scan]
        need = [[_piece(x) for x in leaves_of(t)] for t in types]
        yielded = 0; looked = 0; budget = cap * try_factor
        for k in order:
            b = boards.get(k)
            if b is None:
                b = boards[k] = chess.Board(fens[k])
            pools = [[sq for p in ps for sq in b.pieces(p.piece_type, p.color)] for ps in need]
            if any(not pl for pl in pools):
                continue
            here = 0
            prng = _r.Random(seed * 1000003 + k)                        # a position's assignments in a seeded random order: in piece order the first two
            total = 1                                                   # matches of an abstract slot were always the kings (the root frame's 750 rows, 2026-09-27)
            for pl in pools:
                total *= len(pl)
            if total <= 512:
                combos = list(itertools.product(*pools)); prng.shuffle(combos)
            else:
                seen_c = set(); combos = []
                for _ in range(512):
                    t = tuple(prng.choice(pl) for pl in pools)
                    if t not in seen_c:
                        seen_c.add(t); combos.append(t)
            for ids in combos:
                if len(set(ids)) < len(ids):
                    continue
                if accept is not None:
                    looked += 1
                    if looked > budget:
                        return
                    if not accept(State(b, list(ids))):                   # the literals read the shared board; only an accepted placement is copied
                        continue
                yield State(b.copy(stack=False), list(ids)), False
                yielded += 1; here += 1
                if yielded >= cap:
                    return
                if accept is not None and per_position and here >= per_position:
                    break
    placements.accepts = True
    placements.sampling = f"matches, {per_position} a position, {try_factor}x tries"
    return placements


_TB = None


def tablebase():
    """The Syzygy tables (3-5 men, WDL + DTZ) under assets/tablebases/syzygy, opened once per process."""
    global _TB
    if _TB is None:
        _TB = chess.syzygy.open_tablebase("assets/tablebases/syzygy")
    return _TB


def tb_scalar(state: State, e: int) -> float:
    """The game-theoretic value of the position from the entity's side: WDL in {-2..2} (loss .. win), the tablebase's
    truth for every position with at most five men -- exactly the sampled-frame semantics (only the frame's pieces
    plus the kings are on the board). Kaveh 2026-09-23: "use the structure of the tablebase: per-piece-count truth"."""
    b = state.board
    side = b.piece_at(e).color
    bb = b if b.turn == side else _flip_turn(b)
    if bb is None:
        return 0.0
    try:
        w = tablebase().probe_wdl(bb)
    except (KeyError, chess.syzygy.MissingTableError):
        return 0.0
    return float(w)


def _flip_turn(b: chess.Board):
    nb = b.copy(stack=False); nb.turn = not nb.turn
    return nb if nb.is_valid() else None


def tb_reply_scalar(after: State, e: int) -> float:
    """After our move the opponent is to move: the value from our side is minus the tablebase's value for them.
    Checkmate and stalemate are in the tables' semantics already (probe_wdl handles terminal positions)."""
    b = after.board
    side = not b.turn
    if b.is_checkmate():
        return 2.0 if b.turn != side else -2.0
    if b.is_stalemate():
        return 0.0
    try:
        w = tablebase().probe_wdl(b)
    except (KeyError, chess.syzygy.MissingTableError):
        return 0.0
    return float(-w)


COLOUR_FLIP = {t: ("b" if t[0] == "w" else "w") + t[1:] for t in TYPES}     # the one symmetry of the rules: colours swap (relations and outcomes are colour-blind)
TYPE_TREE = {"*": ["K", "Q", "R", "B", "N", "P"], **{k: ["w" + k, "b" + k] for k in "KQRBNP"}}   # any piece -> a kind -> a colour: the rules' own hierarchy (kinds move differently; colours are the symmetry)


def leaves_of(t: str) -> list:
    if t in TYPES:
        return [t]
    return [x for c in TYPE_TREE[t] for x in leaves_of(c)]


try:
    import analyst_core as _core
except Exception:                                                             # the compiled core is optional: `maturin develop -m rust/analyst_core/Cargo.toml`
    _core = None


def core_outcomes(depth: int):
    """(state, e, relative) -> [((board, move), delta)] through the compiled core, for the frame oracle."""
    def fn(state: State, e: int, relative: bool):
        b = state.board
        acts = {m.uci(): (bb, m) for bb, m in actions(state, e)}
        if not acts:
            return []
        bb = next(iter(acts.values()))[0]
        out = []
        for uci, delta in _core.outcomes(bb.fen(), [(-1 if i is None else i) for i in state.ids], e, depth, relative):
            if uci in acts:
                out.append((acts[uci], delta))
        return out
    return fn


CORE_RELATIONS = frozenset({"same_side", "attacks", "defends", "aligned", "between", "worth_ge", "attacked", "defended", "slides",
                            "gives_check", "captures", "captures_worth_ge", "lands_attacked", "stays_aligned"})   # what analyst_core's literal_masks implements


def core_literals(rows, lits):
    """Bitmasks for the literals the compiled core implements; None for any other relation (the core answers False
    for a name it does not know -- a silent all-false column; the square conditions of 2026-09-27 would have been
    one), so the caller computes those in Python."""
    packed = []
    for state, ids, a, _ in rows:
        bb, m = a
        packed.append((bb.fen(), [(-1 if i is None else i) for i in ids], m.uci()))
    known = [(n, list(sl)) for n, sl in lits if n in CORE_RELATIONS]
    cols = iter(_core.literal_masks(packed, known)) if known else iter(())
    out = []
    for n, sl in lits:
        if n not in CORE_RELATIONS:
            out.append(None); continue
        m = 0
        for i, v in enumerate(next(cols)):
            if v:
                m |= 1 << i
        out.append(m)
    return out


def exchange_frames_of(path, max_pieces: int = 8):
    """FRAMES FROM EXCHANGES (Kaveh 2026-09-27: "when we see a six-piece frame and we label it, let's learn it ... that
    six-piece frame can be deconstructed to a five, four, three, two, one-piece frame"). -> () -> [(priority, types,
    ids, literals)] over the labeled exchanges of the TRAINING games (game id % 5 != 0) and every sub-frame that keeps
    the contested piece: stars around it, kind-level types, an attacker linked by attacks(i,0) & not same_side(i,0), a
    defender by defends(i,0), an x-ray by between(k,i,0) through its blocker k (so it enters a sub-frame only with
    its blocker); a piece with no link at the start (it arrived later) is left out."""
    import itertools, json as _json

    def fn():
        out = []
        for line in open(path):
            e = _json.loads(line)
            if e["game"] % 5 == 0:
                continue
            b = chess.Board(e["fen"]); c = e["squares"][0]; occ = b.piece_at(c)
            members = {chess.parse_square(n) for n in e["pieces"]} - {c}
            direct = []; xray = []
            for q in sorted(members):
                pc = b.piece_at(q)
                if pc is None:
                    continue
                if q in b.attackers(pc.color, c):
                    direct.append((q, (("defends", None, True),) if pc.color == occ.color else (("attacks", None, True), ("same_side", None, False))))
                elif pc.piece_type in (chess.BISHOP, chess.ROOK, chess.QUEEN) and chess.SquareSet.ray(q, c):
                    blockers = [x for x in chess.SquareSet.between(q, c) if b.piece_at(x) is not None]
                    if len(blockers) == 1 and blockers[0] in members:
                        xray.append((q, blockers[0]))
            links = (direct + [(q, ("xray", k)) for q, k in xray])[: max_pieces - 1]
            for n in range(0, len(links) + 1):
                for sub in itertools.combinations(links, n):
                    chosen = {q for q, _ in sub}
                    if any(isinstance(l, tuple) and l and l[0] == "xray" and l[1] not in chosen for _, l in sub):
                        continue                                          # an x-ray without its blocker is no link
                    ids = [c] + [q for q, _ in sub]
                    pos = {q: j for j, q in enumerate(ids)}
                    lits = []
                    for q, l in sub:
                        if l[0] == "xray":
                            lits.append(("between", (pos[l[1]], pos[q], 0), True))
                        else:
                            lits.extend((r, (pos[q], 0), pol) for r, _, pol in l)
                    out.append((0, tuple(KIND_NAME[b.piece_at(i).piece_type] for i in ids), ids, tuple(lits)))
        return out
    return fn


def exchange_outcomes(state: State, e: int, relative: bool = True):
    """THE EXCHANGE ORACLE: every action of e valued by the exchange frames on the real board -- what it wins, the
    exchange the opponent can start on its destination, against the one standing still allows on its origin
    (fates.exchange_move_value, the direct frames). The lemma's frame groups situations; the outcome is the board's."""
    from .fates import direct_frame, exchange_move_value
    b = state.board.copy(stack=False); b.turn = state.board.piece_at(e).color
    if not b.is_valid():
        return []
    return [((b, m), exchange_move_value(b, m, direct_frame)[0]) for m in b.legal_moves if m.from_square == e]


def domain(fens: list | None = None, oracle: str = "material", worth: str = "canonical", worth_tau: float = 4.0, depth: int = 1, shapes: bool = True, per_position: int = 2, try_factor: int = 50, frames: str = "extend") -> Domain:
    """depth: plies of the frame episode after the action (frame oracle only); 1 = the single best reply. shapes: the
    sampled placements are one centred placement per configuration up to translation and mirror."""
    _EPISODE["depth"] = depth; _SHAPES["on"] = shapes
    """worth: 'canonical' (the value table), 'blend' (tables inside five men, blended toward canonical by distance to
    the edge; tau tunable), used by the worth relations of the material and games domains."""
    _BLEND["tau"] = worth_tau
    rels = [Relation("same_side", 2, fn=same_side), Relation("attacks", 2, fn=attacks), Relation("defends", 2, fn=defends),
            Relation("aligned", 2, fn=aligned), Relation("between", 3, fn=between), Relation("worth_ge", 2, fn=worth_ge),
            Relation("attacked", 1, fn=attacked), Relation("defended", 1, fn=defended), Relation("slides", 1, fn=slides),
            Relation("gives_check", 0, True, gives_check), Relation("captures", 0, True, captures),
            Relation("captures_worth_ge", 1, True, captures_worth_ge), Relation("stays_aligned", 1, True, stays_aligned),
            Relation("lands_attacked", 0, True, lands_attacked),
            Relation("attacked_by_lower", 1, fn=attacked_by_lower), Relation("outnumbered", 1, fn=outnumbered), Relation("last_attacker", 2, fn=last_attacker)]
    if frames == "exchanges":
        contests_of = exchange_frames_of(_paths.derived("exchanges.jsonl"))
    else:
        contests_of = (lambda: [(pri, t, ids, lits) for f in fens[: 5000] for pri, t, ids, lits in contests(chess.Board(f))]) if fens is not None else None
    if oracle == "exchange":
        from functools import partial
        return Domain("chess_exchange" if fens is None else "chess_games_exchange", TYPES, rels, placements if fens is None else game_placements(fens, per_position, try_factor), slots_of, actions, transition, frame_scalar, partial(frame_reply_scalar, depth=1), contests=contests_of,
                      entities=entities, state_of=lambda board, ids: State(board.copy(stack=False), list(ids)), actor_ok=lambda board, e: board.piece_at(e).color == board.turn,
                      placement_count=(placement_count if fens is None else None), symmetries=[COLOUR_FLIP], type_tree=TYPE_TREE, root="*",
                      null_reply_scalar=lambda st, e: None if passed(st.board, st.board.piece_at(e).color) is None else 0.0,   # the oracle's values are already against standing still
                      episode=partial(frame_episode_states, depth=1), threat=frame_threat_state, batch_outcomes=exchange_outcomes, batch_literals=(core_literals if _core is not None else None))
    if oracle == "frame":
        from functools import partial
        return Domain("chess_frame" if fens is None else "chess_games_frame", TYPES, rels, placements if fens is None else game_placements(fens, per_position, try_factor), slots_of, actions, transition, frame_scalar, partial(frame_reply_scalar, depth=depth), contests=contests_of,
                      entities=entities, state_of=lambda board, ids: State(board.copy(stack=False), list(ids)), actor_ok=lambda board, e: board.piece_at(e).color == board.turn,
                      placement_count=(placement_count if fens is None else None), symmetries=[COLOUR_FLIP], type_tree=TYPE_TREE, root="*", null_reply_scalar=partial(frame_null_reply_scalar, depth=depth), episode=partial(frame_episode_states, depth=depth), threat=frame_threat_state,
                      batch_outcomes=(core_outcomes(depth) if _core is not None and worth == "canonical" else None), batch_literals=(core_literals if _core is not None else None))
    if oracle == "resolved":
        return Domain("chess_resolved" if fens is None else "chess_games_resolved", TYPES, rels, placements if fens is None else game_placements(fens, per_position, try_factor), slots_of, actions, transition, scalar, resolved_reply_scalar,
                      entities=entities, state_of=lambda board, ids: State(board.copy(stack=False), list(ids)), actor_ok=lambda board, e: board.piece_at(e).color == board.turn,
                      placement_count=(placement_count if fens is None else None), symmetries=[COLOUR_FLIP], type_tree=TYPE_TREE, root="*", null_reply_scalar=resolved_null_reply_scalar)
    if oracle == "tablebase":
        assert fens is None, "the tablebase oracle is for sampled frames (at most five men)"
        rels = [Relation("tb_worth_ge", 2, fn=tb_worth_ge) if r.name == "worth_ge" else Relation("captures_tb_worth_ge", 1, True, tb_captures_worth_ge) if r.name == "captures_worth_ge" else r for r in rels]
        return Domain("chess_tb", TYPES, rels, placements, slots_of, actions, transition, tb_scalar, tb_reply_scalar,
                      entities=entities, state_of=lambda board, ids: State(board.copy(stack=False), list(ids)), actor_ok=lambda board, e: board.piece_at(e).color == board.turn)
    if worth == "blend":
        rels = [Relation("worth_ge", 2, fn=worth_ge_blend) if r.name == "worth_ge" else Relation("captures_worth_ge", 1, True, captures_worth_ge_blend) if r.name == "captures_worth_ge" else r for r in rels]
    return Domain("chess" if fens is None else "chess_games", TYPES, rels, placements if fens is None else game_placements(fens, per_position, try_factor), slots_of, actions, transition, scalar, reply_scalar,
                  entities=entities, state_of=lambda board, ids: State(board.copy(stack=False), list(ids)), actor_ok=lambda board, e: board.piece_at(e).color == board.turn,
                  placement_count=(placement_count if fens is None else None), symmetries=[COLOUR_FLIP], type_tree=TYPE_TREE, root="*", null_reply_scalar=null_reply_scalar)


def entities(board: chess.Board) -> list:
    """Every piece of a frame type on a real board: (square, type). Pawns are not frame entities (yet)."""
    out = []
    for sq, p in board.piece_map().items():
        t = ("w" if p.color else "b") + {v: k for k, v in KIND.items()}.get(p.piece_type, "")
        if t in TYPES:
            out.append((sq, t))
    return out
