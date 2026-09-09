"""memo.py -- the concept store shared across positions and across runs (Kaveh 2026-09-05: "if the knight is still
three hops away in the new position, don't recompute it; but be intelligent about what changed").

The transposition table of a searcher keys on the whole position. A concept keys on its FOOTPRINT: the squares its
computation actually read. Every operator that runs through the store records, as a bitboard, the squares whose
occupancy could change its answer -- the target, its attackers and defenders, the full rays and jumps into every
square it asked about, the path of a piece, the kings it asked legality about. An entry stores the result together
with that mask and a hash of the pieces standing on it. A lookup from ANY position, transposed or not, is a hit when
the pieces on the mask are the same: everything outside the footprint is free to differ.

Sound by construction as long as each operator declares what it reads (the property test in tests/ compares
memo-served answers with fresh ones over many positions and moves). Persisted to sqlite so runs share it."""
from __future__ import annotations

import json
import pickle
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import chess
import chess.polyglot

_KEYS = chess.polyglot.POLYGLOT_RANDOM_ARRAY


# ---------------------------------------------------------------- footprints
def full_rays(sq: int) -> int:
    """Every square a slider could reach sq from, blockers or not: the sound footprint of attackers(sq) with x-rays."""
    return chess.BB_DIAG_ATTACKS[sq][0] | chess.BB_FILE_ATTACKS[sq][0] | chess.BB_RANK_ATTACKS[sq][0]


_DIRS = (1, -1, 8, -8, 9, -9, 7, -7)


def _ray_squares(sq: int, d: int):
    """Squares from sq stepping by d until the edge (d in _DIRS)."""
    f, r = chess.square_file(sq), chess.square_rank(sq)
    df = {1: 1, -1: -1, 8: 0, -8: 0, 9: 1, -9: -1, 7: -1, -7: 1}[d]
    dr = {1: 0, -1: 0, 8: 1, -8: -1, 9: 1, -9: -1, 7: 1, -7: -1}[d]
    f += df; r += dr
    while 0 <= f < 8 and 0 <= r < 8:
        yield chess.square(f, r)
        f += df; r += dr


def _can_join(piece: chess.Piece, d: int, dist: int) -> bool:
    """Could this piece capture along direction d (from `dist` squares away) -- i.e. take part in an exchange on the
    origin square through this ray?"""
    t = piece.piece_type
    straight = d in (1, -1, 8, -8)
    if t == chess.QUEEN:
        return True
    if t == chess.ROOK:
        return straight
    if t == chess.BISHOP:
        return not straight
    if t == chess.KING:
        return dist == 1
    if t == chess.PAWN:
        return dist == 1 and not straight
    return False


def ray_reads(board: chess.Board, sq: int) -> int:
    """Along each ray from sq: every square up to and including the first piece that could NOT take part in an
    exchange on sq through that ray. Beyond such a piece nothing can ever matter to sq; before it, any arrival can."""
    m = 0
    for d in _DIRS:
        for i, q in enumerate(_ray_squares(sq, d), 1):
            m |= chess.BB_SQUARES[q]
            p = board.piece_at(q)
            if p is not None and not _can_join(p, d, i):
                break
    return m


def attack_reads(board: chess.Board, sq: int) -> int:
    """Squares whose occupancy decides who attacks sq, through any number of exchange participants."""
    return (chess.BB_SQUARES[sq] | chess.BB_KNIGHT_ATTACKS[sq] | chess.BB_KING_ATTACKS[sq]
            | chess.BB_PAWN_ATTACKS[chess.WHITE][sq] | chess.BB_PAWN_ATTACKS[chess.BLACK][sq] | ray_reads(board, sq))


def pin_line(board: chess.Board, color: chess.Color, sq: int) -> int:
    """The squares that decide whether the piece on sq is pinned to its king (or a line through sq is opened onto
    the king): the whole line from the king through sq to the edge. Nothing when they are not aligned."""
    k = board.king(color)
    if k is None or k == sq:
        return 0
    ray = chess.ray(k, sq)
    if not ray:
        return 0
    # the far half of the line (beyond the king) cannot pin sq; keep from the king outward through sq
    line = 0
    for q in chess.SquareSet(ray):
        if q == k:
            continue
        if chess.square_distance(q, sq) < chess.square_distance(q, k) or q in chess.SquareSet.between(k, sq) or q == sq:
            line |= chess.BB_SQUARES[q]
    return line | chess.BB_SQUARES[k]


def move_reads(board: chess.Board, sq: int) -> int:
    """Squares whose occupancy decides the moves of the piece on sq (its rays or jumps). Legality is NOT here: the
    caller records the pin line and the checkers predicate (Reads.legality), which are exact and small."""
    p = board.piece_at(sq)
    if p is None:
        return chess.BB_SQUARES[sq]
    m = chess.BB_SQUARES[sq]
    if p.piece_type == chess.KNIGHT:
        m |= chess.BB_KNIGHT_ATTACKS[sq]
    elif p.piece_type == chess.KING:
        m |= chess.BB_KING_ATTACKS[sq]
        for k in chess.SquareSet(chess.BB_KING_ATTACKS[sq]):    # each destination's safety
            m |= attack_reads(board, k)
    elif p.piece_type == chess.PAWN:
        m |= chess.BB_PAWN_ATTACKS[p.color][sq]
        step = 8 if p.color == chess.WHITE else -8
        for k in (sq + step, sq + 2 * step):
            if 0 <= k < 64:
                m |= chess.BB_SQUARES[k]
    else:                                                       # a slider: its rays up to and including the first blocker
        for d in _DIRS:
            straight = d in (1, -1, 8, -8)
            if (p.piece_type == chess.ROOK and not straight) or (p.piece_type == chess.BISHOP and straight):
                continue
            for q in _ray_squares(sq, d):
                m |= chess.BB_SQUARES[q]
                if board.piece_at(q) is not None:
                    break
    return m


class Reads:
    """An accumulator the operators write their footprint into: an occupancy mask plus cheap PREDICATES that are
    recomputed at lookup (who checks a king), so the king's whole neighbourhood need not be in the mask."""
    __slots__ = ("mask", "preds")

    def __init__(self):
        self.mask = 0
        self.preds = []

    def add(self, mask: int):
        self.mask |= mask

    def square(self, sq: int):
        self.mask |= chess.BB_SQUARES[sq]

    def attacks_of(self, board: chess.Board, sq: int):
        self.mask |= attack_reads(board, sq)

    def moves_of(self, board: chess.Board, sq: int):
        self.mask |= move_reads(board, sq)
        p = board.piece_at(sq)
        if p is not None and p.piece_type != chess.KING:
            self.legality(board, p.color, sq)

    def legality(self, board: chess.Board, color: chess.Color, sq: int):
        """What decides whether the piece on sq may move: its pin line and whether its king is in check."""
        self.mask |= pin_line(board, color, sq)
        self.checkers(board, color)

    def checkers(self, board: chess.Board, color: chess.Color):
        k = board.king(color)
        if k is not None:
            self.preds.append(("checkers", (color,), int(board.attackers(not color, k))))
            self.mask |= chess.BB_SQUARES[k]


def eval_pred(board: chess.Board, kind: str, args: tuple):
    if kind == "checkers":
        k = board.king(args[0])
        return int(board.attackers(not args[0], k)) if k is not None else None
    if kind == "castling":
        return int(board.castling_rights)
    if kind == "ep":
        return board.ep_square
    return None


def subhash(board: chess.Board, mask: int) -> int:
    """Zobrist hash of the pieces standing on the masked squares (identity and colour), nothing else."""
    h = 0
    for sq in chess.scan_forward(mask & board.occupied):
        p = board.piece_at(sq)
        h ^= _KEYS[64 * ((p.piece_type - 1) * 2 + int(p.color)) + sq]
    return h


# ---------------------------------------------------------------- the store
@dataclass
class Entry:
    mask: int
    sub: int
    value: object
    hits: int = 0
    preds: tuple = ()               # ((kind, args, value), ...) recomputed at lookup


@dataclass
class Memo:
    path: str | None = None
    table: dict = field(default_factory=dict)      # (kind, args) -> list[Entry]  (several footprints can coexist)
    stats: dict = field(default_factory=lambda: {"hit": 0, "miss": 0, "kinds": {}})
    _rec: list = field(default_factory=list)       # recording accumulators: the union of every footprint read (see recording)

    @contextmanager
    def recording(self):
        """Record the union of the footprints of everything served or computed inside the block: the squares a
        whole computation (a target's ledger and reach, a goal search) depended on. `with memo.recording() as acc:`
        ... `acc[0]` is the mask. Nested recordings each see everything read inside them."""
        acc = [0]
        self._rec.append(acc)
        try:
            yield acc
        finally:
            self._rec = [a for a in self._rec if a is not acc]      # by identity: nested accumulators can hold equal masks

    def _record(self, mask: int):
        for acc in self._rec:
            acc[0] |= mask

    def get(self, kind: str, args: tuple, board: chess.Board, compute):
        """compute(reads) -> value; the operator fills `reads` with its footprint. Served from the store when a stored
        footprint carries the same pieces in this position."""
        key = (kind, args)
        entries = self.table.get(key)
        if entries:
            for e in entries:
                if subhash(board, e.mask) == e.sub and all(eval_pred(board, pk, pa) == pv for pk, pa, pv in e.preds):
                    e.hits += 1; self.stats["hit"] += 1
                    k = self.stats["kinds"].setdefault(kind, {"hit": 0, "miss": 0}); k["hit"] += 1
                    if self._rec:
                        self._record(e.mask)
                    return e.value
        reads = Reads()
        value = compute(reads)
        if self._rec:
            self._record(reads.mask)
        preds = {}
        for pk, pa, pv in reads.preds:                          # one value per predicate (the last recorded is the current board's)
            preds[(pk, pa)] = pv
        e = Entry(reads.mask, subhash(board, reads.mask), value, 0, tuple((pk, pa, pv) for (pk, pa), pv in preds.items()))
        self.table.setdefault(key, []).append(e)
        if len(self.table[key]) > 8:                           # keep the most recent footprints for this key
            self.table[key] = self.table[key][-8:]
        self.stats["miss"] += 1
        k = self.stats["kinds"].setdefault(kind, {"hit": 0, "miss": 0}); k["miss"] += 1
        return value

    def lookup(self, kind: str, args: tuple, board: chess.Board):
        """Serve a stored value whose footprint carries the same pieces here, or None (no computation)."""
        for e in self.table.get((kind, args), ()):
            if subhash(board, e.mask) == e.sub and all(eval_pred(board, pk, pa) == pv for pk, pa, pv in e.preds):
                e.hits += 1; self.stats["hit"] += 1
                k = self.stats["kinds"].setdefault(kind, {"hit": 0, "miss": 0}); k["hit"] += 1
                if self._rec:
                    self._record(e.mask)
                return e.value
        self.stats["miss"] += 1
        k = self.stats["kinds"].setdefault(kind, {"hit": 0, "miss": 0}); k["miss"] += 1
        return None

    def put(self, kind: str, args: tuple, board: chess.Board, mask: int, value, preds=()):
        """Store a value computed by a caller that knows its own footprint (a proof, a sketch)."""
        key = (kind, args)
        e = Entry(mask, subhash(board, mask), value, 0, tuple((pk, pa, eval_pred(board, pk, pa)) for pk, pa in preds))
        self.table.setdefault(key, []).append(e)
        if len(self.table[key]) > 8:
            self.table[key] = self.table[key][-8:]

    def hit_rate(self) -> float:
        n = self.stats["hit"] + self.stats["miss"]
        return self.stats["hit"] / n if n else 0.0

    def report(self) -> str:
        return f"memo: {self.stats['hit']} hits / {self.stats['miss']} misses ({self.hit_rate():.0%}), " + ", ".join(
            f"{k} {v['hit']}/{v['hit']+v['miss']}" for k, v in sorted(self.stats["kinds"].items()))

    # ---- persistence: the store outlives the run
    def save(self, path: str | None = None):
        path = path or self.path
        if not path:
            return
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        con = sqlite3.connect(path)
        con.execute("PRAGMA journal_mode=DELETE")
        con.execute("CREATE TABLE IF NOT EXISTS memo (kind TEXT, args TEXT, mask TEXT, sub TEXT, value BLOB, hits INTEGER, preds BLOB, PRIMARY KEY (kind, args, sub))")
        rows = [(k, json.dumps(a), f"{e.mask:016x}", str(e.sub), pickle.dumps(e.value), e.hits, pickle.dumps(e.preds)) for (k, a), es in self.table.items() for e in es]
        con.executemany("INSERT OR REPLACE INTO memo VALUES (?,?,?,?,?,?,?)", rows)
        con.commit(); con.close()

    @classmethod
    def load(cls, path: str) -> "Memo":
        m = cls(path)
        if not Path(path).exists():
            return m
        con = sqlite3.connect(path)
        try:
            for kind, args, mask, sub, value, hits, preds in con.execute("SELECT kind, args, mask, sub, value, hits, preds FROM memo"):
                key = (kind, tuple(json.loads(args)))
                m.table.setdefault(key, []).append(Entry(int(mask, 16), int(sub), pickle.loads(value), hits, pickle.loads(preds)))
        except sqlite3.OperationalError:
            pass
        con.close()
        return m

    def __len__(self):
        return sum(len(v) for v in self.table.values())
