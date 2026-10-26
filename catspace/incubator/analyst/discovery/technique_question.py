"""PLAN 73.5b (4) THE TECHNIQUE QUESTION: the agenda asks a TECHNIQUE entry of the lemma store as a GOAL, generically -- any entry not REFUTED whose measure declares a
class (a material signature) that is the position's, read from the side to move, and whose quantity is a term of the miner's library (technique_miner.measure_of).
No unit kind is named anywhere: the class is the signature, the measure the entry's term, the floor the engine's mate search, the routes an AND-OR search.

THE ROUTES at a position of the class (the winning side to move):
  AT THE FLOOR the entry's n-move mate search decides (technique_sample.floor_move: the mate words at n = 2, mating.rs's net at n = 3): the route is its first move;
  ELSE the FIRST MOVES OF A STRATEGY of at most d moves (the entry's depth) that lowers the measure below the TARGET against every reply -- the AND-OR search at the
       root, the never-rules on every node: no stalemate, no unit of ours the exchange law loses (technique_miner.unit_lost), no claimable draw (the fifty-move clock,
       and a position the game has seen -- `ctx.history`, the engine's CYCLE reading of a repetition); the fewest moves first (the smallest j with a route), then the
       lowest worst measure after the replies, then the move;
  NONE -> the technique is silent: a GAP "no measure-dropping move", never a kill, and the agenda goes on as before.
THE TARGET is the measure here, unless this position continues the strategy the last question chose (`carried`: its target and the positions after its route and
every reply) and the measure has not yet dropped below that target: then the target carries. Why: with a fresh target at every move a tempo move (j > 1) could
let the measure rise from step to step and cycle; with the target carried and the fewest moves taken, the drop comes within d moves, and the step starts strictly
decrease -- so no position (us to move) of the game repeats.
"""
from __future__ import annotations

import chess

from catspace.incubator.analyst.discovery import lemma_store as LS
from catspace.incubator.analyst.discovery import technique_corpus as TC
from catspace.incubator.analyst.discovery import technique_measures as TM
from catspace.incubator.analyst.discovery import technique_miner as MN
from catspace.incubator.analyst.discovery import technique_sample as TS


def entries_for(board: chess.Board, side: bool) -> list:
    """The TECHNIQUE entries the agenda can ask on `board` for `side` (see the docstring), by name."""
    try:
        store = LS.load()
    except LS.LemmaStoreError:
        return []
    sig = TC.signature(board, side)
    out = []
    for e in sorted(store.values(), key=lambda e: e["name"]):
        mm = e["measure"] if isinstance(e.get("measure"), dict) else {}
        if e["kind"] == "TECHNIQUE" and e["status"] != "REFUTED" and mm.get("class") == sig and MN.measure_of(mm.get("quantity", "")) is not None:
            out.append(e)
    return out


def _key(fen: str) -> str:
    return " ".join(fen.split()[:4])


class Search:
    """The AND-OR search of ONE root under the never-rules and the game's draws: can we force, within j of our moves, a mate or a position (us to move) whose
    measure is below `target`? Memoised by (position, j, ply) for the one target."""

    def __init__(self, m: "MN.Measure", side: bool, target: int, history, clock: int):
        self.m, self.side, self.target, self.history, self.clock = m, side, target, history or frozenset(), clock
        self.memo: dict = {}
        self.vals: dict = {}
        self.nodes = 0

    def value(self, b: chess.Board) -> int:
        k = b._transposition_key()
        v = self.vals.get(k)
        if v is None:
            v = self.vals[k] = self.m.value(TM.atom_values(b, self.side))
        return v

    def drawn(self, b: chess.Board, ply: int) -> bool:
        """A claimable draw at this node: the fifty-move clock, or (us to move) a position the game has seen."""
        return self.clock + ply >= 100 or (b.turn == self.side and _key(b.fen()) in self.history)

    def first(self, b: chess.Board, m: chess.Move, j: int, ply: int) -> "tuple | None":
        """Our move `m` at `b` -> ('mate', None) / ('drop', the worst measure after the replies) when it forces the drop within j moves, else None."""
        b.push(m)
        try:
            self.nodes += 1
            if b.is_checkmate():
                return ("mate", None)
            if b.is_stalemate() or MN.unit_lost(b, self.side) or self.drawn(b, ply + 1):
                return None
            worst = None
            for r in list(b.legal_moves):
                b.push(r)
                try:
                    self.nodes += 1
                    if self.drawn(b, ply + 2):
                        return None
                    v = self.value(b)
                    if not (v < self.target or (j > 1 and self.force(b, j - 1, ply + 2))):
                        return None
                    worst = v if worst is None else max(worst, v)
                finally:
                    b.pop()
            return ("drop", worst)
        finally:
            b.pop()

    def force(self, b: chess.Board, j: int, ply: int) -> bool:
        k = (b._transposition_key(), j, ply)
        if k not in self.memo:
            self.memo[k] = any(self.first(b, m, j, ply) is not None for m in list(b.legal_moves))
        return self.memo[k]


def technique_routes(board: chess.Board, entry: dict, history=frozenset(), carried: "dict | None" = None) -> dict:
    """THE TECHNIQUE QUESTION of `entry` at `board` (the winning side to move; see the docstring) -> {kind: 'floor' | 'drop' | 'silent', routes [(uci, san, j,
    worst)], target, value, tuple (the measure's ranks here), j, word (the floor's), carried (the target carried), nodes (the search's), next (the carry for the
    next question: the target and the positions after the first route and every reply)}."""
    side = board.turn
    mm = entry["measure"]
    m = MN.measure_of(mm["quantity"])
    d, n = int(mm.get("depth", 1)), int(mm.get("n", 2))
    vals = TM.atom_values(board, side)
    now = m.value(vals)
    out = dict(kind="silent", routes=[], target=now, value=now, tuple=m.tuple_of(vals), j=None, word=None, carried=False, nodes=0, next=None)
    fm = TS.floor_move(board, n)
    if fm is not None:
        u, w = fm
        out.update(kind="floor", routes=[(u, board.san(chess.Move.from_uci(u)), 0, None)], word=w)
        return out
    target = now
    if carried is not None and _key(board.fen()) in carried.get("expect", ()) and now >= carried["target"]:
        target, out["carried"] = carried["target"], True
    for attempt in ((target, now) if target != now else (now,)):
        S = Search(m, side, attempt, history, board.halfmove_clock)
        for j in range(1, d + 1):
            found = []
            for mv in list(board.legal_moves):
                got = S.first(board, mv, j, 0)
                if got is not None:
                    found.append((0 if got[0] == "mate" else 1, got[1] if got[1] is not None else -1, mv.uci(), board.san(mv)))
            if found:
                found.sort()
                out.update(kind="drop", routes=[(u, san, j, w) for _k, w, u, san in found], target=attempt, j=j)
                break
        out["nodes"] += S.nodes
        if out["routes"]:
            break
        out["carried"] = False                                 # (the carried target found no route under the game's draws: a fresh target here)
    if out["routes"]:
        b = board.copy(stack=False)
        b.push(chess.Move.from_uci(out["routes"][0][0]))
        exp = set()
        for r in list(b.legal_moves):
            b.push(r)
            exp.add(_key(b.fen()))
            b.pop()
        out["next"] = dict(target=out["target"], expect=frozenset(exp))
    return out


def show(m: "MN.Measure", value: int) -> str:
    """A measure's value as printed: the tuple of its ranks (a lexicographic measure), else the number."""
    if m.op != "lex":
        return str(value)
    xs = []
    for _ in m.parts:
        xs.append(value % MN.RADIX)
        value //= MN.RADIX
    return "(" + ", ".join(str(x) for x in reversed(xs)) + ")"
