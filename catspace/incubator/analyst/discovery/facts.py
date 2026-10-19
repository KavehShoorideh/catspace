"""52.19 THE FACT STORE WITH JUSTIFICATIONS (ANALYST_PLAN 52.19; Kaveh 2026-10-19: "anytime a decision is rendered we need a listener that listens for changes in the premises it used").

Related work: Doyle 1979 (a truth maintenance system, justification-based: a belief records the beliefs it was derived from, and retracting a premise retracts what stands on it) and de Kleer
1986 (the assumption-based TMS: which environments a belief holds in). Ours is justification-based, with the READ SET as the leaf premise: a fact reads squares (and the pieces and attacker
counts on them), and it is valid in ANY position where those cells are as they were (the signature), not only in the FEN it was derived on.

A FACT: id, kind (claim | kill | gap), the question key, the value, the read set and its signature, the JUSTIFICATIONS (ids of the facts it was derived from), the stake of the question that
asked it, the CHAIN STAKE (the largest stake of any fact that stands on it: "the highest stake of the entire chain is the stake of that chain") and a STALE flag.

THE LISTENER (`invalidate`): when a move changes squares, every fact whose read set holds one of them is flagged stale through the inverted read index (square -> facts), and the flag
propagates through the justifications to every fact derived from it. Nothing is recomputed there: a cheap flag flip. In the chase the flags are a FRAME (restored on the way back up the
tree); `advance` (between the moves of a game) flags for good. A stale fact is recomputed LAZILY: in the foreground when a live question needs it (and only then), in the background by
`FactStore.queue` order (chain stake) when the clock is the opponent's.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import chess


@dataclass
class Fact:
    id: int
    kind: str                                 # 'claim' | 'kill' | 'gap'
    key: object                               # the question key (position independent) or a positional key ("inner", pk, ...)
    turn: bool                                # the side the question was asked for
    text: str
    value: int = 0
    reads: frozenset = frozenset()
    sig: tuple = ()
    just: tuple = ()                          # ids of the facts this one was derived from
    stake: int = 0
    chain_stake: int = 0
    depth: int = 0
    under: str = ""                           # "Q3 at depth 2": the question that asked it
    pk: str = ""                              # the position it was derived on (without move counters)
    portable: bool = True                     # valid on any board whose read cells match (False: only on its own position, the FEN memo's case)
    power: int = 2                            # kill only: how hard the question was chased when it was killed: 0 = not chased to its end (as strong as "no direct route"), 1 = an inner agenda's chase (checks, captures, mate threats), 2 = the root's (every forcing move); a kill serves only an asker whose own chase is no stronger
    payload: dict = field(default_factory=dict)
    dependants: list = field(default_factory=list)
    stale: bool = False
    superseded: bool = False
    resolved: bool = False                    # gap only: the route was later followed to its end
    resume: dict | None = None                # gap only: where to resume (position, route, reply, the time mark)
    served: int = 0

    def short(self) -> str:
        return f"#{self.id} {self.kind} {self.text}"


def signature(cell, reads) -> tuple:
    """What a read set read: for every square (sorted) its piece and the number of attackers of each colour (exactly what `regress.diff_squares` compares). `cell(sq)` -> the tuple for one square."""
    return tuple(cell(s) for s in sorted(reads))


def board_cell(board: chess.Board):
    """A memoising `cell` for one board."""
    memo: dict = {}

    def cell(s: int):
        c = memo.get(s)
        if c is None:
            p = board.piece_at(s)
            c = memo[s] = (s, p.symbol() if p is not None else "", len(board.attackers(chess.WHITE, s)), len(board.attackers(chess.BLACK, s)))
        return c
    return cell


class FactStore:
    def __init__(self):
        self.facts: dict = {}                                  # id -> Fact (kept for the justification graphs, superseded ones too)
        self.next_id = 1
        self.groups: dict = defaultdict(dict)                  # (key, turn) -> {reads: {sig: Fact}}   (live, portable facts)
        self.index: dict = defaultdict(set)                    # THE INVERTED READ INDEX: square -> ids of live facts that read it
        self.stale: set = set()
        self.board: chess.Board | None = None                  # the last position `advance` was told about (the game's line)
        self.derived = 0
        self.served = 0
        self.revalidated = 0                                   # served although flagged stale (the cells matched: the flag was conservative)
        self.recomputed = 0                                    # foreground: asked afresh because the stale fact it would have been served from no longer matched
        self.flagged = 0                                       # flag flips by the listener

    # ------------------------------------------------------------------------------------------------------------ add / lookup
    def add(self, kind: str, key, turn: bool, text: str, *, value: int = 0, reads=None, sig: tuple = (), just=(), stake: int = 0, depth: int = 0, under: str = "", pk: str = "",
            portable: bool = True, power: int = 2, payload: dict | None = None, resume: dict | None = None) -> Fact:
        just = tuple(dict.fromkeys(j for j in just if j in self.facts))
        f = Fact(self.next_id, kind, key, turn, text, value, frozenset(reads) if reads is not None else frozenset(), sig, just, stake, stake, depth, under, pk,
                 portable and reads is not None, power, payload or {}, resume=resume)
        self.next_id += 1
        self.facts[f.id] = f
        self.derived += 1
        for j in just:                                         # a dependant: the premise learns who stands on it, and its chain stake rises
            p = self.facts[j]
            p.dependants.append(f.id)
            self._raise(p, f.stake)
        for s in f.reads:                                      # the inverted read index holds every fact that read a square (a positional one too: the listener flags it, and what stands on it)
            self.index[s].add(f.id)
        if f.portable:
            g = self.groups[(key, turn)].setdefault(f.reads, {})
            old = g.get(sig)
            if old is not None:
                self._retire(old)                              # the same premises, derived again: the newer replaces the older
            g[sig] = f
            for rs, sg in list(self.groups[(key, turn)].items()):       # a recomputation replaces the timeline-stale facts of the same question (they no longer match this board)
                for sgn, o in list(sg.items()):
                    if o is not f and o.stale and o.id in self.stale and o.payload.get("perm"):
                        self._retire(o)
        return f

    def _raise(self, f: Fact, stake: int) -> None:
        """Chain stake: the max stake over every fact that depends on this one, propagated up the justifications."""
        todo = [f]
        while todo:
            x = todo.pop()
            if stake > x.chain_stake:
                x.chain_stake = stake
                todo += [self.facts[j] for j in x.just]

    def _retire(self, f: Fact) -> None:
        f.superseded = True
        self.stale.discard(f.id)
        g = self.groups.get((f.key, f.turn), {}).get(f.reads)
        if g is not None and g.get(f.sig) is f:
            del g[f.sig]
            if not g:
                del self.groups[(f.key, f.turn)][f.reads]
        for s in f.reads:
            self.index[s].discard(f.id)

    def lookup(self, key, turn: bool, cell, power: int = 0) -> tuple:
        """-> (fact or None, candidates, stale_miss). A fact is SERVED when its read cells on this board equal its signature (position independent); a kill whose chase was not run to its end is not
        served to an asker whose chase is stronger (`power`). `candidates`: facts of this question existed (the asker reopens it); `stale_miss`: one of them was flagged stale and no longer matches (a foreground recomputation)."""
        gs = self.groups.get((key, turn))
        if not gs:
            return None, False, False
        best, stale_miss = None, False
        for reads, sg in gs.items():
            sig = signature(cell, reads)
            f = sg.get(sig)
            if f is None:
                stale_miss = stale_miss or any(o.id in self.stale for o in sg.values())
                continue
            if f.kind == "kill" and f.power < power:
                continue
            if f.kind == "gap":
                continue
            if best is None or f.id > best.id:
                best = f
        if best is not None:
            best.served += 1
            self.served += 1
            if best.id in self.stale:
                self.revalidated += 1
                self.stale.discard(best.id)
        return best, True, stale_miss and best is None

    # ------------------------------------------------------------------------------------------------------------ the listener
    def invalidate(self, diff, permanent: bool = False) -> list:
        """The squares `diff` changed: every live fact that read one is flagged stale, and so is every fact derived from it. -> the ids newly flagged (a frame, to restore)."""
        new, todo = [], []
        for s in diff:
            for fid in self.index.get(s, ()):
                if fid not in self.stale:
                    todo.append(fid)
                    self.stale.add(fid)
                    new.append(fid)
        while todo:
            f = self.facts[todo.pop()]
            for d in f.dependants:
                g = self.facts[d]
                if d not in self.stale and not g.superseded:
                    self.stale.add(d)
                    new.append(d)
                    todo.append(d)
        self.flagged += len(new)
        if permanent:
            for fid in new:
                self.facts[fid].payload["perm"] = True
        return new

    def restore(self, ids) -> None:
        for fid in ids:
            self.stale.discard(fid)

    def advance(self, board: chess.Board) -> list:
        """Between the moves of a game (or the puzzles of a run): the diff between the last position and this one flags what its moves changed, for good. -> the ids flagged."""
        from catspace.incubator.analyst.discovery import regress as R
        new = []
        if self.board is not None:
            new = self.invalidate(R.diff_squares(self.board, board), permanent=True)
        self.board = board.copy(stack=False)
        return new

    def touched_stake(self, diff) -> int:
        """The largest chain stake of any live fact that reads a square of `diff`: how much a move that changes those squares disturbs (the order of the ponder queue)."""
        best = 0
        for s in diff:
            for fid in self.index.get(s, ()):
                best = max(best, self.facts[fid].chain_stake)
        return best

    # ------------------------------------------------------------------------------------------------------------ the queue
    def queue(self) -> list:
        """The BACKGROUND queue: stale live facts and the open gaps (routes the fuse or the clock cut), by chain stake, highest first."""
        out = [f for f in self.facts.values() if not f.superseded and ((f.id in self.stale and f.kind != "gap") or (f.kind == "gap" and not f.resolved and f.resume is not None))]
        out.sort(key=lambda f: (-f.chain_stake, f.id))
        return out

    def snapshot(self) -> dict:
        return dict(derived=self.derived, served=self.served, revalidated=self.revalidated, recomputed=self.recomputed, flagged=self.flagged)

    # ------------------------------------------------------------------------------------------------------------ the graph
    def graph_lines(self, fid: int, limit: int = 60) -> list:
        """The justification graph of one fact as indented lines: each fact once with its read set and stake, a repeat as '(see #n)'. `limit` bounds the printed facts (the rest is counted)."""
        out, seen = [], set()
        total = self._count(fid)

        def walk(i: int, pad: str):
            f = self.facts[i]
            if i in seen:
                out.append(f"{pad}(see #{i})")
                return
            if len(seen) >= limit:
                return
            seen.add(i)
            names = ", ".join(chess.square_name(s) for s in sorted(f.reads))
            out.append(f"{pad}#{i} {f.kind} {f.text} [value {f.value}, stake {f.stake}, chain stake {f.chain_stake}, {f.under}" + (", stale" if i in self.stale else "") + f"] reads {{{names}}}")
            for j in f.just:
                walk(j, pad + "  ")
        walk(fid, "")
        if total > len(seen):
            out.append(f"... and {total - len(seen)} more facts below")
        return out

    def _count(self, fid: int) -> int:
        seen, todo = set(), [fid]
        while todo:
            i = todo.pop()
            if i in seen:
                continue
            seen.add(i)
            todo += list(self.facts[i].just)
        return len(seen)
