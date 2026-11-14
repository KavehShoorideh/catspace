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

    def cells(self, position: str) -> list:
        """(side as the kernel names it, kind as the description names it, place as the tuple of its handle's characters) of each unit, in the order of `units`: the units the fact view's `cell`
        relation holds, read without the fact view (plan 78.42: the light reading)."""
        import chess
        b = chess.Board(position)
        return [(self.SIDES[p.color], self.KINDS[p.symbol().lower()], tuple(chess.square_name(sq))) for sq, p in sorted(b.piece_map().items())]

    def bearing(self, position: str) -> dict:
        """{(side, kind, place): [places the unit bears on]} of every unit, places as tuples of the handle's characters: what the fact view's `strike` relation holds for the units that stand (a
        place held by the unit's own side counts: the unit guards it)."""
        import chess
        b = chess.Board(position)
        return {(self.SIDES[p.color], self.KINDS[p.symbol().lower()], tuple(chess.square_name(sq))): [tuple(chess.square_name(t)) for t in b.attacks(sq)] for sq, p in b.piece_map().items()}

    # ---- plan 78.42, the whole sheet on the light reading: the move tables of a unit standing on ANY place (the fact view's `strike` and `go` relations), without the fact view

    def _board(self, position: str):
        """The python-chess board of `position`, kept (a few hundred at most) for the readers that only look: never mutate what this returns."""
        import chess
        memo = self.__dict__.setdefault("_board_memo", {})
        got = memo.get(position)
        if got is None:
            if len(memo) >= 512:
                memo.clear()
            got = memo[position] = chess.Board(position)
        return got

    _KIND_TYPE = {"king": 6, "queen": 5, "rook": 4, "bishop": 3, "knight": 2, "pawn": 1}

    def opposite(self, side: str) -> str:
        """The side that is not `side` (as the kernel names the sides)."""
        return self.SIDES[not {v: k for k, v in self.SIDES.items()}[side]]

    def bearing_from(self, position: str, side: str, kind: str, at: str) -> list:
        """The places a unit of `side` and `kind` STANDING ON handle `at` would bear on, the board as it stands (a line stops at the first unit; a place held by its own side counts): the fact
        view's `strike` relation for any placement, present unit or not."""
        import chess
        b = self._board(position)
        color = {v: k for k, v in self.SIDES.items()}[side]
        sq, t = chess.parse_square(at), self._KIND_TYPE[kind]
        if t == chess.PAWN:
            mask = chess.BB_PAWN_ATTACKS[color][sq]
        elif t == chess.KNIGHT:
            mask = chess.BB_KNIGHT_ATTACKS[sq]
        elif t == chess.KING:
            mask = chess.BB_KING_ATTACKS[sq]
        else:
            occ, mask = b.occupied, 0
            if t in (chess.BISHOP, chess.QUEEN):
                mask |= chess.BB_DIAG_ATTACKS[sq][chess.BB_DIAG_MASKS[sq] & occ]
            if t in (chess.ROOK, chess.QUEEN):
                mask |= chess.BB_RANK_ATTACKS[sq][chess.BB_RANK_MASKS[sq] & occ] | chess.BB_FILE_ATTACKS[sq][chess.BB_FILE_MASKS[sq] & occ]
        return [tuple(chess.square_name(s)) for s in chess.scan_forward(mask)]

    def reach_from(self, position: str, side: str, kind: str, at: str) -> list:
        """The places a unit of `side` and `kind` standing on handle `at` may move to in one move, the board as it stands and legality of the side's own goal unit not read (the fact view's `go`
        relation for any placement): an empty place or a place held by the other side but not its goal unit; a pawn by its single forward step onto an empty place or its diagonal onto the other side."""
        import chess
        b = self._board(position)
        color = {v: k for k, v in self.SIDES.items()}[side]
        sq, t = chess.parse_square(at), self._KIND_TYPE[kind]
        foes = b.occupied_co[not color] & ~b.kings
        empty = ~b.occupied & chess.BB_ALL
        strike = {chess.parse_square("".join(p)) for p in self.bearing_from(position, side, kind, at)}
        bits = 0
        for s in strike:
            bits |= chess.BB_SQUARES[s]
        if t == chess.PAWN:
            reach = bits & foes
            ahead = sq + (8 if color else -8)
            if 0 <= ahead < 64 and (chess.BB_SQUARES[ahead] & empty):
                reach |= chess.BB_SQUARES[ahead]
        else:
            reach = bits & (empty | foes)
        return [tuple(chess.square_name(s)) for s in chess.scan_forward(reach)]

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

    def find_move(self, position: str, origin: str, destination: str) -> "str | None":
        """The legal move (as the kernel names it) of the unit on handle `origin` to handle `destination`, None when there is none (plan 78.41: a way's first move)."""
        import chess
        board = self._board(position)                                                   # read only: the position's board is kept (plan 78.42: the sheet asks of one position again and again)
        found = [m for m in board.legal_moves if chess.square_name(m.from_square) == origin and chess.square_name(m.to_square) == destination]
        found.sort(key=lambda m: m.promotion != chess.QUEEN)
        return found[0].uci() if found else None

    def kind_at(self, position: str, place: str) -> "str | None":
        """The kind (the description's word) of the unit standing on handle `place`, None when none does (plan 78.41 frames)."""
        import chess
        p = self._board(position).piece_at(chess.parse_square(place))
        return self.KINDS[p.symbol().lower()] if p else None

    def takings_on(self, position: str, place: str, legal: list) -> list:
        """[(move, kind of the unit that plays it)] of the legal moves that land on handle `place` while a unit stands there (the takings of that unit's place); `legal` is the kernel's list."""
        import chess
        b = self._board(position)
        sq = chess.parse_square(place)
        if b.piece_at(sq) is None:
            return []
        return [(u, self.KINDS[b.piece_at(chess.Move.from_uci(u).from_square).symbol().lower()]) for u in legal if chess.Move.from_uci(u).to_square == sq]

    def moves_of(self, place: str, legal: list) -> list:
        """The moves of `legal` played by the unit standing on handle `place`."""
        return [u for u in legal if u[:2] == place]

    def move_ends(self, move: str) -> tuple:
        """(origin handle, destination handle) of a move."""
        return move[:2], move[2:4]

    def taken_within(self, start: "str | None", moves: list, ply: int, place: str, k: int) -> "bool | None":
        """Is the unit standing on handle `place` after `moves[:ply]` (of either side) taken by a move of the other side within the next `k` plies of the same game? The unit is followed
        when its own side moves it. None when the game has no move left after `ply` or no unit stands there."""
        import chess
        board = chess.Board(start) if start else chess.Board()
        for u in moves[:ply]:
            board.push_uci(u)
        sq = chess.parse_square(place)
        owner = board.color_at(sq)
        rest = moves[ply:ply + k]
        if not rest or owner is None:
            return None
        for u in rest:
            m = chess.Move.from_uci(u)
            if board.turn == owner:
                if m.from_square == sq:
                    sq = m.to_square
            elif m.to_square == sq:
                return True
            board.push(m)
        return False

    # ---- plan 78.41 routes: what the reasoner that follows a whole route asks of the game (generic names, chess underneath)

    def why_not(self, position: str, origin: str, destination: str) -> tuple:
        """(reason, [places involved]) for why the unit on handle `origin` may not move to handle `destination` in `position` (its side to move): `unit_gone` (no unit of the mover there),
        `place_taken` (the destination holds a unit it may not take), `line_shut` (units stand between), `may_not_move` (anything else: the move would leave its own goal unit struck, or
        the goal unit is struck and the move does not answer it)."""
        import chess
        b = chess.Board(position)
        f, t = chess.parse_square(origin), chess.parse_square(destination)
        p, occ = b.piece_at(f), b.piece_at(t)
        if p is None or p.color != b.turn:
            return "unit_gone", []
        if occ is not None and occ.color == p.color:
            return "place_taken", [destination]
        if p.piece_type == chess.PAWN and occ is not None and chess.square_file(f) == chess.square_file(t):
            return "place_taken", [destination]
        if p.piece_type in (chess.BISHOP, chess.ROOK, chess.QUEEN):
            between = [chess.square_name(s) for s in chess.SquareSet(chess.between(f, t)) if b.piece_at(s) is not None]
            if between:
                return "line_shut", between
        return "may_not_move", [origin]

    def move_info(self, position: str, move: str) -> dict:
        """{kind, origin, destination, captured}: the kind (the description's word) of the unit that plays `move` in `position`, its handles, and the kind it takes (None: nothing)."""
        import chess
        b = chess.Board(position)
        m = chess.Move.from_uci(move)
        p, occ = b.piece_at(m.from_square), b.piece_at(m.to_square)
        if occ is None and b.is_en_passant(m):
            occ = chess.Piece(chess.PAWN, not b.turn)
        return {"kind": self.KINDS[p.symbol().lower()], "origin": move[:2], "destination": move[2:4], "captured": self.KINDS[occ.symbol().lower()] if occ else None}

    def rank_answers(self, position: str, moves: list, unit_place: str, route_places: list, target_place: str) -> list:
        """`moves` (the legal moves of the side to move in `position`) ordered so that the likely stoppers of a route come first: the move that takes the travelling unit on handle
        `unit_place`, then one that lands on a place of the route, then one after which that side strikes the unit's place or a place of the route, then one that moves the target
        standing on `target_place` or newly guards it, then the rest (each group in the move names' order)."""
        import chess
        b = chess.Board(position)
        me = b.turn
        unit_sq = chess.parse_square(unit_place)
        route_sq = [chess.parse_square(x) for x in route_places]
        tgt = chess.parse_square(target_place)
        guarded_before = b.is_attacked_by(me, tgt)
        scored = []
        for u in sorted(moves):
            m = chess.Move.from_uci(u)
            if m.to_square == unit_sq:
                score = 0
            elif m.to_square in route_sq:
                score = 1
            else:
                b.push(m)
                try:
                    if any(b.is_attacked_by(me, s) for s in [unit_sq] + route_sq):
                        score = 2
                    elif m.from_square == tgt or (not guarded_before and b.is_attacked_by(me, tgt)):
                        score = 3
                    else:
                        score = 4
                finally:
                    b.pop()
            scored.append((score, u))
        scored.sort()
        return [u for _s, u in scored]

    def without(self, position: str, place: str) -> str:
        """HYPOTHETICAL: the position with the unit on handle `place` removed (its side to move kept). Never a position of the game; the kernel may refuse it (a goal unit removed)."""
        import chess
        b = self._board(position).copy(stack=False)
        b.remove_piece_at(chess.parse_square(place))
        b.castling_rights = b.clean_castling_rights()
        b.ep_square = None
        return b.fen(en_passant="fen")

    def relocate(self, position: str, origin: str, destination: str) -> "str | None":
        """HYPOTHETICAL: the position with the unit on `origin` standing on `destination` instead (a unit there is removed), the side to move kept; None when nothing stands on `origin`."""
        import chess
        b = self._board(position).copy(stack=False)
        p = b.remove_piece_at(chess.parse_square(origin))
        if p is None:
            return None
        b.set_piece_at(chess.parse_square(destination), p)
        b.castling_rights = b.clean_castling_rights()
        b.ep_square = None
        return b.fen(en_passant="fen")

    def cell_reader(self, position: str):
        """A function place -> (place, side, kind or None, units of the first side bearing on it, units of the second side bearing on it) for one position, memoised: what a fact's read set is
        compared by (mirrors facts.board_cell: the unit on a place and the number of units of each side bearing on it)."""
        import chess
        b = chess.Board(position)
        memo: dict = {}

        def cell(place):
            c = memo.get(place)
            if c is None:
                sq = chess.parse_square(place)
                p = b.piece_at(sq)
                c = memo[place] = (place, self.SIDES[p.color] if p else None, self.KINDS[p.symbol().lower()] if p else None, len(b.attackers(chess.WHITE, sq)), len(b.attackers(chess.BLACK, sq)))
            return c
        return cell

    def changed_places(self, before: str, after: str) -> set:
        """The handles whose content or whose count of units bearing on them (either side) differs between two positions: what a move changed (mirrors regress.diff_squares)."""
        import chess
        a, b = chess.Board(before), chess.Board(after)
        out = set()
        for sq in chess.SQUARES:
            if (a.piece_at(sq) != b.piece_at(sq) or len(a.attackers(chess.WHITE, sq)) != len(b.attackers(chess.WHITE, sq))
                    or len(a.attackers(chess.BLACK, sq)) != len(b.attackers(chess.BLACK, sq))):
                out.add(chess.square_name(sq))
        return out

    def contents(self, position: str, places: list) -> tuple:
        """What stands on each handle of `places` (in the order given): (side, kind) or None."""
        import chess
        b = chess.Board(position)
        out = []
        for x in places:
            p = b.piece_at(chess.parse_square(x))
            out.append((self.SIDES[p.color], self.KINDS[p.symbol().lower()]) if p else None)
        return tuple(out)

    def tracks(self, start: "str | None", moves: list) -> tuple:
        """(by_ply, taken): `by_ply[i]` maps handle -> (unit id, side, kind) in the position after `moves[:i]` (an id follows its unit through moves, a castling, a promotion);
        `taken[i]` = (unit id, side, kind) of the unit the move `moves[i]` takes, absent when it takes none."""
        import chess
        board = chess.Board(start) if start else chess.Board()
        ids = {sq: n for n, sq in enumerate(sorted(board.piece_map()))}

        def snap():
            return {chess.square_name(sq): (ids[sq], self.SIDES[p.color], self.KINDS[p.symbol().lower()]) for sq, p in board.piece_map().items()}
        by_ply, taken = [snap()], {}
        for i, u in enumerate(moves):
            m = chess.Move.from_uci(u)
            gone = None
            if board.is_en_passant(m):
                gone = chess.square(chess.square_file(m.to_square), chess.square_rank(m.from_square))
            elif board.piece_at(m.to_square) is not None:
                gone = m.to_square
            if gone is not None:
                p = board.piece_at(gone)
                taken[i] = (ids.pop(gone), self.SIDES[p.color], self.KINDS[p.symbol().lower()])
            ids[m.to_square] = ids.pop(m.from_square)
            if board.is_castling(m):
                rook_from = chess.square(7 if chess.square_file(m.to_square) > chess.square_file(m.from_square) else 0, chess.square_rank(m.from_square))
                rook_to = chess.square(5 if chess.square_file(m.to_square) > chess.square_file(m.from_square) else 3, chess.square_rank(m.from_square))
                ids[rook_to] = ids.pop(rook_from)
            board.push(m)
            by_ply.append(snap())
        return by_ply, taken

    def events(self, start: "str | None", moves: list) -> list:
        """One dict per move of the game: {mover: (unit id, side, kind) of the unit that moves, origin, dest: its handles, taken: (unit id, side, kind) of the unit the move takes or None,
        taken_at: the handle that unit stood on or None, arrives: the kind the moving unit has once the move is made (another kind than `mover`'s when the move changes what the unit is)}. The ids are those of `tracks` (an id follows its unit through moves, a castling, a promotion)."""
        import chess
        by_ply, taken = self.tracks(start, moves)
        board = chess.Board(start) if start else chess.Board()
        out = []
        for i, u in enumerate(moves):
            m = chess.Move.from_uci(u)
            origin, dest = chess.square_name(m.from_square), chess.square_name(m.to_square)
            at = None
            if i in taken:
                at = chess.square_name(chess.square(chess.square_file(m.to_square), chess.square_rank(m.from_square))) if board.is_en_passant(m) else dest
            out.append({"mover": by_ply[i][origin], "origin": origin, "dest": dest, "taken": taken.get(i), "taken_at": at, "arrives": by_ply[i + 1][dest][2]})
            board.push(m)
        return out

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
