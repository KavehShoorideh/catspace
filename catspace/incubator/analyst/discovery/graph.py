"""LAYER 1 OF THE CORE (ANALYST_PLAN 52.1): the board as a graph that answers attack / reach / COUNTERFACTUAL questions. Pure functions over python-chess
boards (the rules) and, where the vocabulary has the word, the Rust `analyst_core` reading (`see`). A counterfactual question is a board with a piece placed
or removed, re-read; it is never a claim: the guarantee of anything built on it comes from a proved schema's binding (binder.py).

Colours are python-chess colours (True = white). Squares are python-chess ints.
KNOWN TRAP: the Rust word `see` reads 0 for the side NOT to move (see reasoned.py's docstring): `see` here always reads with the mover = the side to move
at the board, so for the exchange WE would win on their piece, put us on move first (`with_turn`).
"""
from __future__ import annotations

import chess

import analyst_core as _A
from catspace.incubator.analyst.discovery import base_vocabulary as BV
from catspace.incubator.analyst.discovery import selfmine as SM

WORTH = SM.PIECE_WORTH                       # p 1, n 3, b 3, r 5, q 9, k 0
SLIDING = {chess.BISHOP: (-9, -7, 7, 9), chess.ROOK: (-8, -1, 1, 8), chess.QUEEN: (-9, -8, -7, -1, 1, 7, 8, 9)}


def worth(piece: chess.Piece | None) -> int:
    return 0 if piece is None else WORTH[piece.symbol().lower()]


def balance(board: chess.Board, owner: bool) -> int:
    """The vocabulary word `balance` (owner's material less the other's, plus 64) in python-chess, for the chase, which reads it on every reply of every route
    (the Rust call costs a FEN each; tested equal to `word(board, "balance", owner)` in test_regress)."""
    own = sum(WORTH[chess.piece_symbol(pt)] * chess.popcount(board.pieces_mask(pt, owner)) for pt in chess.PIECE_TYPES if pt != chess.KING)
    oth = sum(WORTH[chess.piece_symbol(pt)] * chess.popcount(board.pieces_mask(pt, not owner)) for pt in chess.PIECE_TYPES if pt != chess.KING)
    return own - oth + 64


def name(sq: int) -> str:
    return chess.square_name(sq)


def colour_of(us: str | bool) -> bool:
    return us if isinstance(us, bool) else us == "w"


def with_turn(board: chess.Board, colour: bool) -> chess.Board:
    """A copy with `colour` to move (en passant square and the move stack dropped: a counterfactual board has no history). Always a copy: callers push on it."""
    c = board.copy(stack=False)
    c.turn = colour
    c.ep_square = None
    return c


CLEAR_HOOKS: list = []     # 54 (B4): per-root caches elsewhere (binder's verify_tried) register their `clear` here; graph.clear_caches calls them

_FENS: dict = {}           # 52.20 (1): transposition key (bitboards, turn, castling, e.p.) -> FEN; a board's FEN is built once per root question


def fen_of(board: chess.Board) -> str:
    """The FEN of `board` for the Rust evaluator, from the per-board cache (149k FEN strings were built for 19k distinct boards on 0KCeb)."""
    k = board._transposition_key()
    f = _FENS.get(k)
    if f is None:
        f = _FENS[k] = board.fen()
    return f


_XFENS: dict = {}          # 54 (B5.2): the exact key (every field the FEN prints) -> FEN, per root question


def exact_fen(board: chess.Board) -> str:
    """`board.fen()` from a cache keyed by every field the FEN prints (the bitboards, the side to move, the castling field, the e.p. square, BOTH clocks), so it is the
    same string `board.fen()` gives for any board that shares them -- unlike `fen_of`, whose key leaves the clocks out. A board changed by a push, a pop or a set_piece_at
    changes the key; nothing is stored on the board, so no mutation can leave a stale entry."""
    k = (board.pawns, board.knights, board.bishops, board.rooks, board.queens, board.kings, board.occupied_co[True], board.occupied_co[False],
         board.turn, board.castling_rights, board.ep_square, board.halfmove_clock, board.fullmove_number)
    f = _XFENS.get(k)
    if f is None:
        f = _XFENS[k] = board.fen()
    return f


def clear_caches() -> None:
    """Called at every fresh root question: the root is the unit of attention; knowledge across moves lives in the fact store, not here."""
    _FENS.clear()
    _XFENS.clear()
    SM.clear_word_cache()
    _A.clear_caches()                          # 54 (B3): the core's parsed positions
    for hook in CLEAR_HOOKS:
        hook()


# ---------------------------------------------------------------------------------------------------------------- attack questions

def attackers_of(board: chess.Board, sq: int, colour: bool) -> list:
    """Squares of `colour`'s pieces that attack `sq` now (x-rays are `xrays_to`)."""
    return sorted(board.attackers(colour, sq))


def defenders_of(board: chess.Board, sq: int) -> list:
    """Squares of the pieces that guard the piece on `sq` (its own colour's attackers of the square)."""
    pc = board.piece_at(sq)
    return [] if pc is None else attackers_of(board, sq, pc.color)


def flights(board: chess.Board, king_sq: int) -> list:
    """Squares the king on `king_sq` can step to that are not attacked by the other side (the king itself lifted off the board, so a slider's shadow behind
    it still attacks), not occupied by its own side, and not the square of a piece giving check (taking the checker is the defect `checker takeable`, not a
    flight)."""
    king = board.piece_at(king_sq)
    colour = king.color
    lifted = board.copy(stack=False)
    lifted.remove_piece_at(king_sq)
    checkers = set(board.attackers(not colour, king_sq))
    out = []
    for to in board.attacks(king_sq):
        pc = board.piece_at(to)
        if (pc is not None and pc.color == colour) or to in checkers:
            continue
        if lifted.is_attacked_by(not colour, to):
            continue
        out.append(to)
    return sorted(out)


def xrays_to(board: chess.Board, sq: int, colour: bool, max_blockers: int = 2) -> list:
    """The sliders of `colour` that would attack `sq` if the pieces between them and it were gone: [(piece_sq, [blocker squares])] with 1..max_blockers
    blockers on the line (any colour), the nearest slider per ray only counts (a rook behind a rook is a battery, found through its blocker)."""
    out = []
    for ps in chess.SQUARES:
        pc = board.piece_at(ps)
        if pc is None or pc.color != colour or pc.piece_type not in SLIDING:
            continue
        between = chess.SquareSet(chess.between(ps, sq))
        # the piece must move along that line: rook-like on a file / rank, bishop-like on a diagonal
        same_file_rank = chess.square_file(ps) == chess.square_file(sq) or chess.square_rank(ps) == chess.square_rank(sq)
        diag = abs(chess.square_file(ps) - chess.square_file(sq)) == abs(chess.square_rank(ps) - chess.square_rank(sq))
        if ps == sq or not (same_file_rank or diag):
            continue
        if pc.piece_type == chess.ROOK and not same_file_rank:
            continue
        if pc.piece_type == chess.BISHOP and not diag:
            continue
        blockers = [b for b in between if board.piece_at(b) is not None]
        if 1 <= len(blockers) <= max_blockers:
            blockers.sort(key=lambda b: chess.square_distance(b, ps))
            out.append((ps, blockers))
    return sorted(out)


# ---------------------------------------------------------------------------------------------------------------- counterfactuals

def without(board: chess.Board, sq: int) -> chess.Board:
    """The board with the piece on `sq` removed."""
    c = board.copy(stack=False)
    c.remove_piece_at(sq)
    return c


def with_piece(board: chess.Board, piece: chess.Piece, at: int, removed_from: int | None = None) -> chess.Board:
    """The counterfactual placement: `piece` on `at` (whatever stood there is gone), lifted from `removed_from` when given."""
    c = board.copy(stack=False)
    if removed_from is not None:
        c.remove_piece_at(removed_from)
    c.set_piece_at(at, piece)
    return c


def mobility(board: chess.Board, colour: bool) -> int:
    """Legal moves `colour` would have on move."""
    return with_turn(board, colour).legal_moves.count()


def mobility_of(board: chess.Board, sq: int) -> int:
    """The Rust word `(count (mobility sq))`: the squares the piece on `sq` can legally go to and be safe on by the exchange. Read with the piece's OWNER as the
    mover (`with_turn`: the evaluator reads from the side to move, and a piece of the side not to move reads wrong -- the trap `see` has too)."""
    return word(board, ("count", ("mobility", name(sq))), board.piece_at(sq).color)


def see(board: chess.Board, sq: int) -> int:
    """The Rust word `see` on `sq`, read with the mover = the side to move at `board`: what the SIDE TO MOVE wins by the exchange started by a capture on `sq`
    (it reads 0 for the side not to move, and nothing for a piece of the mover's own side). `see_for` puts the capturer on move first."""
    BV.install_terms()
    return SM.val_at(fen_of(board), ("see", name(sq)), "w" if board.turn else "b") or 0


def see_for(board: chess.Board, sq: int, capturer: bool) -> int:
    """What `capturer` wins by the exchange on the piece on `sq` (a piece of the other side): `see` read with `capturer` on move."""
    return see(with_turn(board, capturer), sq)


# ---------------------------------------------------------------------------------------------------------------- the Rust words (promotion, 52.6)

def word(board: chess.Board, term, owner: bool) -> int:
    """A word of the Rust vocabulary read on `board` with `owner` as the mover. The evaluator reads from the mover's side, and reads 0 / mirrored squares when
    the mover is not the side to move, so the board is first given to `owner` (`with_turn`); a pawn word (`pushes`, `ahead`, `promo`) is read with its owner."""
    BV.install_terms()
    b = board if (board.turn == owner and board.ep_square is None) else with_turn(board, owner)      # 52.20 (1): no copy when the board already reads so
    return SM.val_at(fen_of(b), term, "w" if owner else "b")


def piece_moves_to(board: chess.Board, frm: int, targets: set, limit: int) -> int | None:
    """Fewest moves the non-pawn piece on `frm` needs to stand on one of `targets` (static board: the other pieces do not move, own-colour squares are not
    entered, a piece of the other colour may be taken; no pins, no checks), or None beyond `limit`. The `reach` of a slider / knight."""
    pc = board.piece_at(frm)
    seen, frontier = {frm}, [frm]
    for k in range(1, limit + 1):
        nxt = []
        for s in frontier:
            c = board.copy(stack=False)
            c.remove_piece_at(frm)
            c.set_piece_at(s, pc)
            for t in c.attacks(s):
                o = board.piece_at(t)
                if t in seen or (o is not None and o.color == pc.color and t != frm):
                    continue
                if t in targets:
                    return k
                seen.add(t)
                nxt.append(t)
        frontier = nxt
    return None


# ---------------------------------------------------------------------------------------------------------------- the battery and the discovery (52.18 (2); Lean Ideas.Battery)

def _aligned(piece: chess.Piece, s: int, t: int) -> bool:
    """A slider of this kind can move along the line s-t (queen: any line; rook: a file or rank; bishop: a diagonal)."""
    if piece.piece_type == chess.QUEEN:
        return chess.BB_RAYS[s][t] != 0
    if piece.piece_type == chess.ROOK:
        return chess.square_file(s) == chess.square_file(t) or chess.square_rank(s) == chess.square_rank(t)
    if piece.piece_type == chess.BISHOP:
        return chess.BB_RAYS[s][t] != 0 and chess.square_file(s) != chess.square_file(t) and chess.square_rank(s) != chess.square_rank(t)
    return False


def batteries_to(board: chess.Board, colour: bool, t: int) -> list:
    """Lean `batteryB`: [(s2, [s1 ...])] -- a slider of `colour` on s2 aims along a line at t (geometry, blockers aside) and the squares strictly between hold only pieces of `colour`
    (at least one, listed from the rear slider outward) or nothing."""
    out = []
    for s2 in chess.SQUARES:
        pc = board.piece_at(s2)
        if s2 == t or pc is None or pc.color != colour or pc.piece_type not in SLIDING or not _aligned(pc, s2, t):
            continue
        between = sorted(chess.SquareSet(chess.between(s2, t)), key=lambda u: chess.square_distance(u, s2))
        held = [u for u in between if board.piece_at(u) is not None]
        if held and all(board.piece_at(u).color == colour for u in held):
            out.append((s2, held))
    return out


def almost_battery(board: chess.Board, colour: bool, s1: int, s2: int, t: int) -> bool:
    """Lean `almostBatteryB`: a battery whose only piece between is the one on s1."""
    return any(r == s2 and held == [s1] for r, held in batteries_to(board, colour, t))


def discoverers(board: chess.Board, colour: bool, t: int) -> list:
    """The DISCOVERY candidates aimed at t: [(s1, s2, rest, almost)] -- the piece on s1 (the only blocker that cannot ride the line: a knight, a king, a pawn, a slider of the wrong
    kind) stands in front of the rear slider on s2; when it leaves, the rear attacks t directly (almost = it was the only piece between: Lean `discovered_attack`) or joins the
    exchange behind the aligned sliders `rest` that remain (the x-ray of the exchange law, Lean `battery_persists`)."""
    out = []
    for s2, held in batteries_to(board, colour, t):
        bad = [u for u in held if not _aligned(board.piece_at(u), u, t)]
        if len(bad) == 1:
            out.append((bad[0], s2, [u for u in held if u != bad[0]], len(held) == 1))
    return out


def discovery_text(board: chess.Board, s1: int, s2: int, rest: list, t: int) -> str:
    """'the N leaves e4, the B on g2 joins the attack on a8 (battery with the Q on b7)'."""
    nm = lambda s: board.piece_at(s).symbol().upper()
    with_ = ("battery with " + ", ".join(f"the {nm(u)} on {name(u)}" for u in rest)) if rest else "almost a battery: nothing else between"
    return f"the {nm(s1)} leaves {name(s1)}, the {nm(s2)} on {name(s2)} joins the attack on {name(t)} ({with_})"


def skewer_after(board: chess.Board, colour: bool, t: int) -> "int | None":
    """52.23 (4): THE SKEWER, read on the board AFTER our move (their turn): a slider of `colour` that gives CHECK and whose line runs on through their king to `t`
    (a piece of theirs) with every other square between empty -> the slider's square, else None. Lean `Ideas.Skewer.skewerB`; `skewer_attack`: when the king steps off
    the line the slider attacks `t`; `skewer_blocked`: before that it does not (the king is in the way)."""
    k = board.king(not colour)
    pt = board.piece_at(t)
    if k is None or pt is None or pt.color == colour:
        return None
    for s2 in board.checkers():
        pc = board.piece_at(s2)
        if pc is None or pc.color != colour or pc.piece_type not in (chess.ROOK, chess.BISHOP, chess.QUEEN):
            continue
        line = chess.SquareSet(chess.between(s2, t))
        if k not in line:
            continue
        straight = chess.square_file(s2) == chess.square_file(t) or chess.square_rank(s2) == chess.square_rank(t)
        if (pc.piece_type == chess.ROOK and not straight) or (pc.piece_type == chess.BISHOP and straight):
            continue
        if all(board.piece_at(u) is None for u in line if u != k):
            return s2
    return None
