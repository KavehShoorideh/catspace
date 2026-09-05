"""structure.py -- L3.5: positional STRUCTURE facts (Kaveh 2026-09-05: the ~30% of errors with no tactic in the
reply, and the even trades whose cost is 'what the trade does to the plans'). Same shape as L3: each fact is a
Relation (name, squares, owner, tags, piece symbols) so L4 can diff it per move and L6 can say what changed.
These are OBSERVATIONS about the position, not priced causes -- the plan layer (ANALYST_PLAN §14) explains
why a structure matters; here we only say that it appeared or disappeared.

Pawn facts use only what L1 gives: pawn colour, the pawn's forward direction, its attack squares.
  isolated_pawn    (sq)              no own pawn on an adjacent file
  doubled_pawns    (sq, sq, ...)     two or more own pawns on one file (one fact per file)
  passed_pawn      (sq)              no enemy pawn ahead on its file or the adjacent files; tag = ranks to go
  backward_pawn    (sq)              cannot be supported by a pawn (its neighbours are ahead of it) and its
                                     stop square is controlled by an enemy pawn
  pawn_islands     ()                per side, tag = count (a state fact; changes when islands split or join)
File facts:
  rook_on_open_file (rook)           its file has no pawns;  rook_on_half_open_file: no own pawn, enemy pawns
Piece facts:
  outpost          (piece, pawn)     a minor piece in the enemy half, held by a pawn, no enemy pawn can ever attack it
  bishop_pair      (b1, b2)          both bishops
  bad_bishop       (bishop)          two or more own central pawns on its colour that are blocked (tags = them)
King facts:
  king_open_file   (king)            an own-pawn-free file on or next to the king (tag = file letter)"""
from __future__ import annotations

import chess

from .relations import Relation, _rel
from . import rules as R

STRUCTURE_NAMES = ("isolated_pawn", "doubled_pawns", "passed_pawn", "backward_pawn", "pawn_islands",
                   "rook_on_open_file", "rook_on_half_open_file", "outpost", "bishop_pair", "bad_bishop",
                   "king_open_file", "scope")
MINORS = (chess.KNIGHT, chess.BISHOP)


def _forward(color: chess.Color) -> int:
    return 8 if color == chess.WHITE else -8


def _ahead(sq: int, color: chess.Color) -> list[int]:
    """Squares in front of sq on its file, in the pawn's direction of travel."""
    out, q = [], sq + _forward(color)
    while 0 <= q < 64:
        out.append(q); q += _forward(color)
    return out


def _pawns(board: chess.Board, color: chess.Color) -> chess.SquareSet:
    return board.pieces(chess.PAWN, color)


def pawn_facts(board: chess.Board) -> list[Relation]:
    out = []
    for c in (chess.WHITE, chess.BLACK):
        own, enemy = _pawns(board, c), _pawns(board, not c)
        own_files = {chess.square_file(q) for q in own}
        for f in range(8):
            on_file = sorted(q for q in own if chess.square_file(q) == f)
            if len(on_file) >= 2:                                  # identity = (side, file): the pawns may shuffle along it
                out.append(Relation("doubled_pawns", len(on_file), (), c, (chess.FILE_NAMES[f],),
                                    tuple(board.piece_at(q).symbol() for q in on_file)))
        for sq in own:
            f, r = chess.square_file(sq), chess.square_rank(sq)
            adj = {f - 1, f + 1} & set(range(8))
            if not (adj & own_files):
                out.append(_rel(board, "isolated_pawn", 1, (sq,), c))
            ahead_ranks = {chess.square_rank(a) for a in _ahead(sq, c)}
            blockers = [q for q in enemy if abs(chess.square_file(q) - f) <= 1 and chess.square_rank(q) in ahead_ranks]
            if not blockers:
                to_go = (7 - r) if c == chess.WHITE else r
                out.append(_rel(board, "passed_pawn", 1, (sq,), c, (f"{to_go}_to_go",)))
            # backward: every neighbouring own pawn is AHEAD of it (none level or behind, so it is the laggard of its
            # group and no pawn can come up beside it), and its stop square is controlled by an enemy pawn
            behind_or_level = [q for q in own if chess.square_file(q) in adj and
                               ((chess.square_rank(q) <= r) if c == chess.WHITE else (chess.square_rank(q) >= r))]
            stop = sq + _forward(c)
            if not behind_or_level and 0 <= stop < 64 and any(
                    board.piece_at(a) and board.piece_at(a).piece_type == chess.PAWN and board.piece_at(a).color != c
                    for a in board.attackers(not c, stop)):
                out.append(_rel(board, "backward_pawn", 1, (sq,), c))
        # islands
        files = sorted(own_files)
        islands = sum(1 for i, f in enumerate(files) if i == 0 or files[i - 1] != f - 1)
        if own:
            out.append(Relation("pawn_islands", 0, (), c, (str(islands),), ()))
    return out


def file_facts(board: chess.Board) -> list[Relation]:
    out = []
    for c in (chess.WHITE, chess.BLACK):
        own, enemy = _pawns(board, c), _pawns(board, not c)
        for rook in board.pieces(chess.ROOK, c):
            f = chess.square_file(rook)
            own_on, enemy_on = any(chess.square_file(q) == f for q in own), any(chess.square_file(q) == f for q in enemy)
            if not own_on and not enemy_on:
                out.append(_rel(board, "rook_on_open_file", 1, (rook,), c, (chess.FILE_NAMES[f],)))
            elif not own_on:
                out.append(_rel(board, "rook_on_half_open_file", 1, (rook,), c, (chess.FILE_NAMES[f],)))
    return out


def piece_facts(board: chess.Board) -> list[Relation]:
    out = []
    for c in (chess.WHITE, chess.BLACK):
        own, enemy = _pawns(board, c), _pawns(board, not c)
        bishops = sorted(board.pieces(chess.BISHOP, c))
        if len(bishops) >= 2 and len({(chess.square_file(b) + chess.square_rank(b)) % 2 for b in bishops}) == 2:
            out.append(Relation("bishop_pair", 0, (), c, (), ()))      # identity = the side: bishops move, the pair stays
        for b in bishops:
            colour = (chess.square_file(b) + chess.square_rank(b)) % 2
            # the human concept: own CENTRAL pawns (c-f) on the bishop's colour that are BLOCKED (a piece on their stop square)
            blocked = [q for q in own if (chess.square_file(q) + chess.square_rank(q)) % 2 == colour
                       and 2 <= chess.square_file(q) <= 5 and 0 <= q + _forward(c) < 64 and board.piece_at(q + _forward(c)) is not None]
            if len(blocked) >= 2:
                out.append(_rel(board, "bad_bishop", 1, (b,), c, tuple(sorted(chess.square_name(q) for q in blocked))))
        for pt in MINORS:
            for sq in board.pieces(pt, c):
                r = chess.square_rank(sq)
                in_enemy_half = r >= 4 if c == chess.WHITE else r <= 3
                if not in_enemy_half:
                    continue
                holders = [a for a in board.attackers(c, sq) if board.piece_at(a).piece_type == chess.PAWN]
                if not holders:
                    continue
                f = chess.square_file(sq)
                # an enemy pawn on an adjacent file that is still 'behind' the square (from its own side) could advance to attack it
                can_be_driven = any(abs(chess.square_file(q) - f) == 1 and
                                    ((chess.square_rank(q) > r) if c == chess.WHITE else (chess.square_rank(q) < r))
                                    for q in enemy)
                if not can_be_driven:
                    out.append(_rel(board, "outpost", 2, (sq, holders[0]), c))
    return out


SCOPE_PIECES = (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)
SCOPE_STEP = 3          # a line opened or closed for a piece that did not move: at least this many squares


def scope_facts(board: chess.Board) -> list[Relation]:
    """How many squares each piece reaches (tag). Only its CHANGE for a piece that stayed put is ever phrased:
    'opens a line for the bishop on c8' -- the vocabulary item behind 'missed e5' with no tactic in it."""
    out = []
    for c in (chess.WHITE, chess.BLACK):
        for pt in SCOPE_PIECES:
            for sq in board.pieces(pt, c):
                out.append(_rel(board, "scope", 1, (sq,), c, (str(len(board.attacks(sq))),)))
    return out


def king_facts(board: chess.Board) -> list[Relation]:
    out = []
    for c in (chess.WHITE, chess.BLACK):
        k = R.royal_square(board, c)
        if k is None or board.has_castling_rights(c):          # before castling the king's files are the opening, not a shelter
            continue
        own = _pawns(board, c)
        kf = chess.square_file(k)
        for f in range(max(0, kf - 1), min(7, kf + 1) + 1):
            if not any(chess.square_file(q) == f for q in own):
                out.append(Relation("king_open_file", 0, (), c, (chess.FILE_NAMES[f],), (board.piece_at(k).symbol(),)))
    return out


def structure(board: chess.Board) -> list[Relation]:
    return pawn_facts(board) + file_facts(board) + piece_facts(board) + king_facts(board) + scope_facts(board)


def structure_diff(board: chess.Board, moves) -> tuple[list[Relation], list[Relation]]:
    """(created, ended) structure between `board` and the position after `moves` (uci or Move), so a trade
    is judged after the recapture, not after the first capture."""
    t = board.copy(stack=False)
    before = {r.key: r for r in structure(t)}
    for m in moves:
        t.push(chess.Move.from_uci(m) if isinstance(m, str) else m)
    after = {r.key: r for r in structure(t)}
    order = lambda r: (r.name, r.squares, -1 if r.owner is None else int(r.owner), r.tags)
    created = sorted((after[k] for k in after.keys() - before.keys()), key=order)
    ended = sorted((before[k] for k in before.keys() - after.keys()), key=order)
    return created, ended


# ---- L6 side: what a structural change says ------------------------------------------------------
def _side(c: bool) -> str:
    return "White" if c else "Black"


def _pc(sym: str, sq: int) -> str:
    from .verbalize import pc
    return pc(sym, sq)


def structure_phrase(r: Relation, created: bool, board: chess.Board | None = None) -> str:
    """One clause for a structure fact that appeared (created=True) or disappeared."""
    pc = _pc
    n = r.name
    if n == "isolated_pawn":
        return f"isolates {pc(r.pieces[0], r.squares[0])}" if created else f"{pc(r.pieces[0], r.squares[0])} is no longer isolated"
    if n == "doubled_pawns":
        return f"{'doubles' if created else 'undoubles'} {_side(r.owner)}'s pawns on the {r.tags[0]}-file"
    if n == "passed_pawn":
        return (f"{_side(r.owner)} gets a passed pawn on {chess.square_name(r.squares[0])}" if created
                else f"{_side(r.owner)}'s passed pawn on {chess.square_name(r.squares[0])} is gone")
    if n == "backward_pawn":
        return f"{pc(r.pieces[0], r.squares[0])} {'becomes' if created else 'is no longer'} backward"
    if n == "pawn_islands":
        return f"{_side(r.owner)} now has {r.tags[0]} pawn island{'s' if r.tags[0] != '1' else ''}"
    if n == "rook_on_open_file":
        return (f"{pc(r.pieces[0], r.squares[0])} {'takes' if created else 'leaves'} the open {r.tags[0]}-file")
    if n == "rook_on_half_open_file":
        return (f"{pc(r.pieces[0], r.squares[0])} {'takes' if created else 'leaves'} the half-open {r.tags[0]}-file")
    if n == "outpost":
        return (f"{pc(r.pieces[0], r.squares[0])} {'settles on' if created else 'leaves'} an outpost (no enemy pawn can ever attack it)")
    if n == "bishop_pair":
        return f"{_side(r.owner)} {'gets' if created else 'gives up'} the bishop pair"
    if n == "bad_bishop":
        return (f"{pc(r.pieces[0], r.squares[0])} is {'hemmed in by' if created else 'freed from'} its own blocked pawns ({', '.join(r.tags)})")
    if n == "king_open_file":
        return f"the {r.tags[0]}-file opens next to the {_side(r.owner).lower()} king"
    return n


def structure_changes(created: list[Relation], ended: list[Relation]) -> list[str]:
    """Phrases for one move's structural diff. A passed pawn that advanced shows once as a push; a change in
    the island count shows once with the old count."""
    out, used = [], set()
    for r in created:
        if r.name == "passed_pawn":
            prev = next((e for e in ended if e.name == "passed_pawn" and e.owner == r.owner
                         and chess.square_file(e.squares[0]) == chess.square_file(r.squares[0])), None)
            if prev is not None:
                used.add(prev.key)
                out.append(f"pushes the passed pawn to {chess.square_name(r.squares[0])}")
                continue
        if r.name == "scope":
            prev = next((e for e in ended if e.name == "scope" and e.squares == r.squares and e.pieces == r.pieces), None)
            if prev is not None:
                used.add(prev.key)
                d = int(r.tags[0]) - int(prev.tags[0])
                if d >= SCOPE_STEP:
                    out.append(f"opens a line for {_pc(r.pieces[0], r.squares[0])} (+{d} squares)")
                elif d <= -SCOPE_STEP:
                    out.append(f"shuts in {_pc(r.pieces[0], r.squares[0])} ({d} squares)")
            continue                                     # a piece that moved: its own scope change is not a fact
        if r.name in ("isolated_pawn", "backward_pawn"):
            prev = next((e for e in ended if e.name == r.name and e.owner == r.owner
                         and chess.square_file(e.squares[0]) == chess.square_file(r.squares[0])), None)
            if prev is not None:                         # the same pawn moved along its file: still isolated / backward
                used.add(prev.key)
                continue
        if r.name == "pawn_islands":
            prev = next((e for e in ended if e.name == "pawn_islands" and e.owner == r.owner), None)
            if prev is not None:
                used.add(prev.key)
                out.append(f"{_side(r.owner)}'s pawns go from {prev.tags[0]} island{'s' if prev.tags[0] != '1' else ''} to {r.tags[0]}")
                continue
            continue                                     # islands only speak when they change
        if r.name in ("rook_on_open_file", "rook_on_half_open_file", "king_open_file", "bad_bishop"):
            same = (lambda e: e.squares == r.squares) if r.name == "bad_bishop" else (lambda e: e.tags == r.tags)   # same bishop / same file
            prev = next((e for e in ended if e.name == r.name and e.owner == r.owner and same(e)), None)
            if prev is not None:                         # same side, same file: the piece moved along it, nothing changed
                used.add(prev.key)
                continue
        out.append(structure_phrase(r, True))
    for e in ended:
        if e.key in used or e.name in ("pawn_islands", "scope", "king_open_file"):
            continue
        out.append(structure_phrase(e, False))
    return out


PRIORITY = ("passed_pawn", "bishop_pair", "outpost", "scope", "isolated_pawn", "backward_pawn", "doubled_pawns",
            "rook_on_open_file", "rook_on_half_open_file", "king_open_file", "bad_bishop", "pawn_islands")


def _scope_delta(r: Relation, ended: list[Relation]) -> int:
    prev = next((e for e in ended if e.name == "scope" and e.squares == r.squares and e.pieces == r.pieces), None)
    return abs(int(r.tags[0]) - int(prev.tags[0])) if prev is not None else 0


def top_changes(created: list[Relation], ended: list[Relation], k: int = 2) -> list[str]:
    """The k most telling structural changes of a move (by PRIORITY, then the size of a scope change), phrased."""
    order = lambda r: (PRIORITY.index(r.name) if r.name in PRIORITY else 99, -_scope_delta(r, ended) if r.name == "scope" else 0)
    c = sorted(created, key=order); e = sorted(ended, key=order)
    phrases = structure_changes(c, e)
    return phrases[:k]


def structure_tags(created: list[Relation], ended: list[Relation], us: chess.Color) -> list[str]:
    """Owner-relative labels for a move's structural changes, for statistics: 'own:bishop_pair-' (we lost it),
    'their:isolated_pawn+' (they got one), 'own:scope-' (a piece of ours was shut in)."""
    tags = set()
    for r, sign in [(r, "+") for r in created] + [(r, "-") for r in ended]:
        who = "own" if r.owner == us else "their"
        if r.name == "scope":
            prev = next((e for e in ended if e.name == "scope" and e.squares == r.squares and e.pieces == r.pieces), None)
            if sign == "+" and prev is not None:
                d = int(r.tags[0]) - int(prev.tags[0])
                if d >= SCOPE_STEP: tags.add(f"{who}:scope+")
                elif d <= -SCOPE_STEP: tags.add(f"{who}:scope-")
            continue
        if r.name == "pawn_islands" or (r.name == "king_open_file" and sign == "-"):   # a king walking away 'closes' nothing
            continue
        tags.add(f"{who}:{r.name}{sign}")
    return sorted(tags)
