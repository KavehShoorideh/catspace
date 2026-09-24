"""The CHESS ADAPTER for lemma_gen: the only file that knows what a piece is. It supplies the rules and nothing
else -- entity types, relations as programs over a board, actions and the transition, a scalar (material), and
placements (every board holding exactly the frame's pieces plus both kings, canonicalized by the board's symmetry).
No pawns in this first version: the eightfold symmetry then holds and the frame is orientation-free."""
from __future__ import annotations

import itertools
import random

import chess
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


def placements(types: tuple, cap: int, seed: int):
    """Boards holding exactly the frame's entities plus both kings (added if the frame does not name them). The
    first slot's square is fixed to the ten canonical squares (symmetry). Exhaustive when the raw count fits the
    cap, else a uniform sample with the flag False. White to move; actions flip the turn to the actor's side."""
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
    for squares in it:
        attempts += 1
        if attempts > cap * 200 and yielded == 0:
            return                                                    # nothing valid in many draws: the frame is not placeable (bounded, never a loop)
        if not exhaustive and yielded >= cap:
            return
        if exhaustive and yielded >= cap * 2:
            return
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
    if not b.is_capture(m):
        return False
    return blend_worth(b, m.to_square) >= blend_worth(b, j)


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


def captures_worth_ge(state, action, j):
    b, m = action
    if not b.is_capture(m):
        return False
    return VAL[b.piece_at(m.to_square).piece_type] >= VAL[b.piece_at(j).piece_type]


def stays_aligned(state, action, j):
    """After the action the mover is still on a line with slot j (nothing between)."""
    b, m = action
    if m.to_square == j:
        return False
    nb = b.copy(stack=False); nb.push(m)
    return bool(chess.SquareSet.ray(m.to_square, j)) and not any(nb.piece_at(s) for s in chess.SquareSet.between(m.to_square, j))


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
    b.turn = side
    if b.is_check():
        return None
    b.turn = not side
    if not b.is_valid():
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


def frame_reply_scalar(after: State, e: int) -> float:
    """The frame's material from the actor's side after the opponent's best immediate reply (the reply that minimizes
    it): a capture of a frame piece counts, a capture elsewhere does not; a checkmate counts as +-1000."""
    b = after.board
    side = not b.turn
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


def frame_null_reply_scalar(state: State, e: int) -> float:
    b = state.board.copy(stack=False)
    side = b.piece_at(e).color
    b.turn = side
    if b.is_check():
        return None
    b.turn = not side
    if not b.is_valid():
        return None
    return frame_reply_scalar(State(b, list(state.ids)), e)


def null_reply_scalar(state: State, e: int) -> float:
    """The actor's side's material after the opponent's best immediate reply if the actor PASSES (the null move): what
    the standing threats already take. On a full board the opponent can nearly always capture something, so an
    outcome must be read relative to this or almost every move "loses" (gauge_lemmas, 2026-09-23). Undefined (None)
    when the actor is in check: there is no null move."""
    b = state.board.copy(stack=False)
    side = b.piece_at(e).color
    b.turn = side
    if b.is_check():
        return None
    b.turn = not side
    if not b.is_valid():
        return None
    return reply_scalar(State(b, list(state.ids)), e)


def game_placements(fens: list):
    """A placements source drawn from REAL positions (games): for a frame's types, every assignment of distinct pieces
    of those types on each position, in a seeded order, up to the cap; never exhaustive. The rest of the board stays,
    so a lemma learned here is learned where it is applied."""
    import random as _r

    boards: dict = {}                                                   # parsed once per process (fork copies what the parent parsed)

    def placements(types: tuple, cap: int, seed: int):
        rng = _r.Random(seed)
        order = list(range(len(fens))); rng.shuffle(order)
        scan = getattr(placements, "scan_limit", None)                   # the frequency count scans a bounded prefix of the positions
        if scan:
            order = order[:scan]
        need = [_piece(t) for t in types]
        yielded = 0
        for k in order:
            b = boards.get(k)
            if b is None:
                b = boards[k] = chess.Board(fens[k])
            pools = [list(b.pieces(p.piece_type, p.color)) for p in need]
            if any(not pl for pl in pools):
                continue
            for ids in itertools.product(*pools):
                if len(set(ids)) < len(ids):
                    continue
                yield State(b.copy(stack=False), list(ids)), False
                yielded += 1
                if yielded >= cap:
                    return
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


def domain(fens: list | None = None, oracle: str = "material", worth: str = "canonical", worth_tau: float = 4.0) -> Domain:
    """worth: 'canonical' (the value table), 'blend' (tables inside five men, blended toward canonical by distance to
    the edge; tau tunable), used by the worth relations of the material and games domains."""
    _BLEND["tau"] = worth_tau
    rels = [Relation("same_side", 2, fn=same_side), Relation("attacks", 2, fn=attacks), Relation("defends", 2, fn=defends),
            Relation("aligned", 2, fn=aligned), Relation("between", 3, fn=between), Relation("worth_ge", 2, fn=worth_ge),
            Relation("attacked", 1, fn=attacked), Relation("defended", 1, fn=defended), Relation("slides", 1, fn=slides),
            Relation("gives_check", 0, True, gives_check), Relation("captures", 0, True, captures),
            Relation("captures_worth_ge", 1, True, captures_worth_ge), Relation("stays_aligned", 1, True, stays_aligned),
            Relation("lands_attacked", 0, True, lands_attacked)]
    if oracle == "frame":
        return Domain("chess_frame" if fens is None else "chess_games_frame", TYPES, rels, placements if fens is None else game_placements(fens), slots_of, actions, transition, frame_scalar, frame_reply_scalar,
                      entities=entities, state_of=lambda board, ids: State(board.copy(stack=False), list(ids)), actor_ok=lambda board, e: board.piece_at(e).color == board.turn,
                      placement_count=(placement_count if fens is None else None), null_reply_scalar=frame_null_reply_scalar)
    if oracle == "resolved":
        return Domain("chess_resolved" if fens is None else "chess_games_resolved", TYPES, rels, placements if fens is None else game_placements(fens), slots_of, actions, transition, scalar, resolved_reply_scalar,
                      entities=entities, state_of=lambda board, ids: State(board.copy(stack=False), list(ids)), actor_ok=lambda board, e: board.piece_at(e).color == board.turn,
                      placement_count=(placement_count if fens is None else None), null_reply_scalar=resolved_null_reply_scalar)
    if oracle == "tablebase":
        assert fens is None, "the tablebase oracle is for sampled frames (at most five men)"
        rels = [Relation("tb_worth_ge", 2, fn=tb_worth_ge) if r.name == "worth_ge" else Relation("captures_tb_worth_ge", 1, True, tb_captures_worth_ge) if r.name == "captures_worth_ge" else r for r in rels]
        return Domain("chess_tb", TYPES, rels, placements, slots_of, actions, transition, tb_scalar, tb_reply_scalar,
                      entities=entities, state_of=lambda board, ids: State(board.copy(stack=False), list(ids)), actor_ok=lambda board, e: board.piece_at(e).color == board.turn)
    if worth == "blend":
        rels = [Relation("worth_ge", 2, fn=worth_ge_blend) if r.name == "worth_ge" else Relation("captures_worth_ge", 1, True, captures_worth_ge_blend) if r.name == "captures_worth_ge" else r for r in rels]
    return Domain("chess" if fens is None else "chess_games", TYPES, rels, placements if fens is None else game_placements(fens), slots_of, actions, transition, scalar, reply_scalar,
                  entities=entities, state_of=lambda board, ids: State(board.copy(stack=False), list(ids)), actor_ok=lambda board, e: board.piece_at(e).color == board.turn,
                  placement_count=(placement_count if fens is None else None), null_reply_scalar=null_reply_scalar)


def entities(board: chess.Board) -> list:
    """Every piece of a frame type on a real board: (square, type). Pawns are not frame entities (yet)."""
    out = []
    for sq, p in board.piece_map().items():
        t = ("w" if p.color else "b") + {v: k for k, v in KIND.items()}.get(p.piece_type, "")
        if t in TYPES:
            out.append((sq, t))
    return out
