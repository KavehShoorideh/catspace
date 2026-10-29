"""PLAN 73.5b (2) THE AMENDED MINER: a technique is MINED by the plan's own Lean obligation, not by the table's moves (73.5's miner and diagnostic stay as they were:
technique_measures.py, the record). A measure M (a composition of the base atoms of 73.5 (3)) is a TECHNIQUE for a material class when every position of THE
SAMPLE (technique_sample.py: the class's won positions, the winning side to move) is either AT THE FLOOR -- the engine's own mate search finds a mate within n
moves (n = 2: the mate words; n = 3: mating.rs's net) -- or admits a DROP: a strategy of ours of at most d moves (an AND-OR search: some move of ours, every legal
reply, ...) after which M is strictly lower (or they are mated), never through a stalemate, a unit of ours the exchange law loses, or a claimable draw. M is a
well-founded order (natural numbers; a lexicographic tuple as its mixed-radix number), so the drops end and the floor mates: Lean `Ideas.Forced.forced_by_measure`
(the method, PROVED) turns the obligation into a forced mate; the OBLIGATION itself is checked on the sample, not proved (a CONJECTURE).

  python -m catspace.incubator.analyst.discovery.technique_miner KRvK [--corpus <file>] [--uniform 3000] [--seed 2]

THE ATOMS: technique_measures.ATOMS less the atoms that read TEMPO (the fifty-move clock): "TEMPO enters as d" (73.5b) -- the strategy's length is the tempo the
technique spends; a clock that rises on every ply cannot fall, and it is no function of the position (two lines to one position carry two clocks). THE
COMPOSITIONS (generic, never named after an ending): rank 1 -- every atom, the sum and the product of every two (73.5 (3)); ranks 2 and 3 -- the LEXICOGRAPHIC
TUPLE of two or three distinct atoms, `(lex A B ...)`: compare A; on a tie, B; ...

THE 73.6 LIBRARY (`--library 73.6`, plan 73.6 (2)): the same atoms and the PER-AXIS ATOMS of technique_measures.AXIS_ATOMS (every unit's rank / file distance to
an edge, every two units' rank / file distance, whose move), the class's constant atoms left out (`constant_atoms`: a candidate with one equals one without it);
the search, the criterion and the fuse unchanged. `--record --from _class_736` writes the run into the class's mined entry's evidence (status and measure
unchanged: the whole-class check is evidence, never the proof).

THE SEARCH, simplest first (73.5b): d = 1, 2, 3; for each d, ranks 1, 2, 3; for each (d, ranks), the floor n = 2, then n = 3. The FIRST (d, ranks, n) where some
measure has ZERO failures is the answer; among those the fewest base reads (then the fewest atoms, the shorter term). THE FUSE is the hand-made K+Q technique's
own shape (mating.rs's doc header: three ranks, a tempo move within three, a three-move net): (d, ranks, n) = (3, 3, 3) -- beyond it the generator stops and
reports the best measure and its failing positions (FEN): a result, never a relaxation, and nothing is written. CHEAPEST FIRST: every candidate at d = 1 on
every sample position; only its failures at d = 2; only those at d = 3. (At d >= n the floor adds nothing: a mate within d moves is a success of the search.)

THE AND-OR SEARCH (the `Graph`): the positions of the class with the winning side to move are NODES, keyed by the position (python-chess's transposition key),
each EXPANDED once into our legal moves under the never-rules (a move that stalemates or leaves a unit of ours the exchange law loses -- graph.see_for -- is
refused; a mate is a success) and, for each, the nodes after every legal reply. For a measure M the search is the min-max of the path minimum:
G_1(p) = min over our moves of max over replies of M(child) (a mate: minus infinity; no move: plus infinity), V_k(q) = min(M(q), G_(k-1)(q)),
G_k(p) = min over our moves of max over replies of V_k(child); a drop within d exists at p if and only if G_d(p) < M(p). G does not depend on the root, so it is
MEMOISED BY (position, d, measure) -- one array per (d, measure) over the graph's nodes (numpy), every candidate read at once. The claimable draws: no position
repeats three times within the six plies of d = 3 from a sample position (its FEN carries no history), and the fifty-move clock cannot run out (the sample's
largest clock is checked below 100 - 6, technique_sample.py) -- so the miner's search needs neither; the TECHNIQUE QUESTION (the agenda's) reads both.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import time
from array import array
from collections import Counter
from dataclasses import dataclass

import chess
import numpy as np

from catspace.incubator.analyst.discovery import graph as G
from catspace.incubator.analyst.discovery import lemma_store as LS
from catspace.incubator.analyst.discovery import technique_corpus as TC
from catspace.incubator.analyst.discovery import technique_measures as TM
from catspace.incubator.analyst.discovery import technique_sample as TS
from catspace.io import paths

DEPTHS = (1, 2, 3)          # d: our moves before the measure must be lower
RANKS = (1, 2, 3)           # the lexicographic ranks
NETS = (2, 3)               # n: the floor's mate search (2: the mate words; 3: mating.rs's net)
FUSE = (3, 3, 3)            # (d, ranks, n) of the hand-made K+Q technique (mating.rs's doc header): beyond it the generator stops
RADIX = 128                 # a lexicographic tuple is the mixed-radix number of its atoms (every atom of a class is below it: asserted on the sample)
TOP = 5                     # the best candidates printed per level
INF = 1 << 50

ATOMS = tuple(a for a in TM.ATOMS if "tempo" not in LS.term_reads(a.sexpr))      # "TEMPO enters as d"
# PLAN 73.6 (2): the library with the PER-AXIS ATOMS (technique_measures.AXIS_ATOMS: every unit's rank / file distance to an edge, every two units' rank / file
# distance, whose move). The clock stays out (73.5b's reason: it rises on every ply and is no function of the position); WHOSE MOVE is a function of the position
# and enters. 73.5b's library (ATOMS) is the record and the default.
ATOMS_736 = ATOMS + TM.AXIS_ATOMS
LIBRARIES = {"73.5b": ATOMS, "73.6": ATOMS_736}
KEYS = tuple(a.key for a in ATOMS_736)                  # the graph's atom row: every atom of every library
COL = {k: i for i, k in enumerate(KEYS)}


# ---------------------------------------------------------------------------------------------------------------- the measures

@dataclass(frozen=True)
class Measure:
    sexpr: str
    op: str             # 'atom' | 'sum' | 'product' | 'lex'
    parts: tuple        # atom keys, in rank order for 'lex'
    gloss: str

    @property
    def ranks(self) -> int:
        return len(self.parts) if self.op == "lex" else 1

    @property
    def reads(self) -> frozenset:
        """THE BASE READS: lemma_store.term_reads of the term (the vocabulary's own table)."""
        return frozenset(LS.term_reads(self.sexpr))

    def simplicity(self) -> tuple:
        """The fewest base reads first, then the fewest atoms, then the shorter term (then the term: a fixed last resort)."""
        return (len(self.reads), len(self.parts), len(self.sexpr), self.sexpr)

    def value(self, vals: dict) -> int:
        """M on one position (`vals`: technique_measures.atom_values); a tuple as its mixed-radix number (lexicographic order = numeric order)."""
        xs = [vals[k] for k in self.parts]
        if self.op == "atom":
            return xs[0]
        if self.op == "sum":
            return xs[0] + xs[1]
        if self.op == "product":
            return xs[0] * xs[1]
        v = 0
        for x in xs:
            v = v * RADIX + x
        return v

    def tuple_of(self, vals: dict) -> tuple:
        """M as printed: the tuple of its ranks (one number for rank 1)."""
        return tuple(vals[k] for k in self.parts) if self.op == "lex" else (self.value(vals),)

    def column(self, A: np.ndarray) -> np.ndarray:
        """M on every node of the graph at once (`A`: the atom matrix, one row per node)."""
        xs = [A[:, COL[k]].astype(np.int64) for k in self.parts]
        if self.op == "atom":
            return xs[0]
        if self.op == "sum":
            return xs[0] + xs[1]
        if self.op == "product":
            return xs[0] * xs[1]
        v = np.zeros(len(A), dtype=np.int64)
        for x in xs:
            v = v * RADIX + x
        return v


def candidates(ranks: int, atoms: tuple = ATOMS) -> list:
    """THE CANDIDATES of a rank over `atoms` (a library): rank 1 -- the atoms, the sum and the product of every two (73.5 (3)); rank k > 1 -- the lexicographic
    tuple of k distinct atoms."""
    if ranks == 1:
        out = [Measure(a.sexpr, "atom", (a.key,), a.gloss) for a in atoms]
        for a, b in itertools.combinations(atoms, 2):
            out.append(Measure(f"(sum {a.sexpr} {b.sexpr})", "sum", (a.key, b.key), f"{a.gloss} plus {b.gloss}"))
        for a, b in itertools.combinations(atoms, 2):
            out.append(Measure(f"(product {a.sexpr} {b.sexpr})", "product", (a.key, b.key), f"{a.gloss} times {b.gloss}"))
        return out
    out = []
    for t in itertools.permutations(atoms, ranks):
        out.append(Measure("(lex " + " ".join(a.sexpr for a in t) + ")", "lex", tuple(a.key for a in t),
                           "lexicographic: " + "; on a tie, ".join(a.gloss for a in t)))
    return out


_BY_SEXPR: dict = {}


def measure_of(sexpr: str) -> "Measure | None":
    """The measure printed as `sexpr` (an entry's quantity) in any library (73.6's holds 73.5b's), or None when it is no term of a library (a hand-made entry's
    prose)."""
    if not _BY_SEXPR:
        for r in RANKS:
            _BY_SEXPR.update({m.sexpr: m for m in candidates(r, ATOMS_736)})
    return _BY_SEXPR.get(sexpr)


def constant_atoms(graph: "Graph", atoms: tuple) -> list:
    """The atoms of `atoms` that read one value on every node of the graph (the class's constants: material in a class where nothing is captured, whose move at
    nodes where the winning side always moves) -- a candidate with one is EQUIVALENT to one without it (a sum shifts, a product scales or reads 0, a lexicographic
    rank on it ties everywhere), so the 73.6 library leaves them out (printed); 73.5b's run kept them (the record)."""
    A = np.array(graph.rows, dtype=np.int64)
    return [a for a in atoms if A[:, COL[a.key]].min() == A[:, COL[a.key]].max()]


def atom_row(board: chess.Board, side: bool) -> tuple:
    v = TM.atom_values(board, side)
    return tuple(v[k] for k in KEYS)


# ---------------------------------------------------------------------------------------------------------------- the never-rules and the floor

def unit_lost(board: chess.Board, side: bool) -> bool:
    """After our move (they to move): a unit of ours (not the king) that the exchange law loses -- what they win by the exchange on it is positive (graph.see_for)."""
    return any(G.see_for(board, s, not side) > 0 for s in chess.scan_forward(board.occupied_co[side] & ~board.kings))



# ---------------------------------------------------------------------------------------------------------------- the class graph and the AND-OR search

class Graph:
    """THE CLASS GRAPH (see the docstring): nodes = positions of the class with the winning side `side` to move, each expanded once under the never-rules."""

    LO, HI = -1, -2                     # the child of a mating move (minus infinity) / of a node with no move (plus infinity)

    def __init__(self, side: bool = chess.WHITE):
        self.side = side
        self.ids: dict = {}
        self.fens: list = []
        self.rows: list = []            # the atom row of each node
        self.done = bytearray()         # per node: expanded
        self.order = array("q")         # the expanded nodes, in the order expanded
        self.where: dict = {}           # ... node -> its index in `order`
        self.mstart = array("q")        # ... the index of each one's first move (moves are stored flat, node after node)
        self.rstart = array("q")        # per move: the index of its first child
        self.child = array("q")         # the children of every move (node ids; LO for a mate, HI for no move)
        self.expanded = 0
        self.edges = 0                  # (our move, reply) pairs stored
        self.refused: Counter = Counter()
        self._arr = None

    def add(self, board: chess.Board) -> int:
        k = board._transposition_key()
        i = self.ids.get(k)
        if i is None:
            i = self.ids[k] = len(self.fens)
            self.fens.append(board.fen())
            self.rows.append(atom_row(board, self.side))
            self.done.append(0)
        return i

    def expand(self, i: int) -> list:
        """Expand node i once (the never-rules on our moves) -> the ids of its children (empty when expanded before)."""
        if self.done[i]:
            return []
        b = chess.Board(self.fens[i])
        self.where[i] = len(self.order)
        self.order.append(i)
        self.mstart.append(len(self.rstart))
        kids_all = []
        moves = 0
        for m in list(b.legal_moves):
            b.push(m)
            try:
                if b.is_checkmate():
                    self.rstart.append(len(self.child))
                    self.child.append(self.LO)
                    moves += 1
                    continue
                if b.is_stalemate():
                    self.refused["stalemate"] += 1
                    continue
                if unit_lost(b, self.side):
                    self.refused["a unit the exchange law loses"] += 1
                    continue
                self.rstart.append(len(self.child))
                for r in list(b.legal_moves):
                    b.push(r)
                    c = self.add(b)
                    b.pop()
                    self.child.append(c)
                    kids_all.append(c)
                    self.edges += 1
                moves += 1
            finally:
                b.pop()
        if not moves:
            self.rstart.append(len(self.child))
            self.child.append(self.HI)
        self.done[i] = 1
        self.expanded += 1
        self._arr = None
        if self.expanded % 2000 == 0:
            G.clear_caches()                                    # (graph.see_for's handles: one per board read)
        return kids_all

    def grow(self, roots, d: int) -> None:
        """Expand every node within d - 1 of our moves of `roots` (breadth first): G_d is then exact at the roots."""
        frontier = list(dict.fromkeys(roots))
        seen = set(frontier)
        for j in range(d):
            nxt = []
            for i in frontier:
                if self.done[i]:
                    kids = self._kids(i) if j < d - 1 else ()
                else:
                    kids = self.expand(i)
                for c in kids:
                    if c not in seen:
                        seen.add(c)
                        nxt.append(c)
            frontier = nxt if j < d - 1 else []
        G.clear_caches()

    def _kids(self, i: int) -> list:
        """The children of an expanded node i (read back from the flat arrays)."""
        k = self.where[i]
        m0 = self.mstart[k]
        m1 = self.mstart[k + 1] if k + 1 < len(self.mstart) else len(self.rstart)
        c0 = self.rstart[m0]
        c1 = self.rstart[m1] if m1 < len(self.rstart) else len(self.child)
        return [c for c in self.child[c0:c1] if c >= 0]

    def arrays(self) -> tuple:
        """(atom matrix, child ids (LO -> n, HI -> n + 1), the start of each move's children, the start of each expanded node's moves, the expanded nodes)."""
        if self._arr is not None:
            return self._arr
        n = len(self.fens)
        c = np.frombuffer(self.child, dtype=np.int64).copy()
        c[c == self.LO] = n
        c[c == self.HI] = n + 1
        self._arr = (np.array(self.rows, dtype=np.int64), c, np.frombuffer(self.rstart, dtype=np.int64).copy(),
                     np.frombuffer(self.mstart, dtype=np.int64).copy(), np.frombuffer(self.order, dtype=np.int64).copy())
        return self._arr

    def solve(self, Mv: np.ndarray, d: int) -> np.ndarray:
        """G_d of every node (see the docstring; plus infinity where not expanded): exact at every node whose d - 1 neighbourhood is expanded."""
        _A, child, rstart, mstart, order = self.arrays()
        V = Mv
        g = None
        tail = np.array([-INF, INF], dtype=np.int64)
        for _k in range(d):
            Vx = np.concatenate([V, tail])
            worst = np.maximum.reduceat(Vx[child], rstart)
            g = np.full(len(Mv), INF, dtype=np.int64)
            g[order] = np.minimum.reduceat(worst, mstart)
            V = np.minimum(Mv, g)
        return g


def drops(graph: Graph, m: Measure, roots: list, d: int) -> list:
    """The roots (node ids) with a drop of `m` within d -> [bool per root]."""
    A = graph.arrays()[0]
    Mv = m.column(A)
    g = graph.solve(Mv, d)
    return [bool(g[i] < Mv[i]) for i in roots]


# ---------------------------------------------------------------------------------------------------------------- the miner

def with_surprises(sig: str, rows: list, log=print) -> list:
    """plan 76.3 (d) THE MINER READS THE SURPRISES: the class's REFUTED MEASURES' cases (discovery/surprises.py, lemmas/surprises/) join the sample as CONSTRAINTS --
    every candidate must drop (or be at the floor) on each of them too, so a measure the play refuted is never chosen again (counterexamples are permanent). A case
    already in the sample is not added twice; one with the losing side to move is no position of the miner's (the winning side moves first). -> the rows with the
    new cases appended, their floors read."""
    from catspace.incubator.analyst.discovery import surprises as SU
    have = {TS._key(r["fen"]) for r in rows}
    cases = SU.refuted_for(sig)
    new = [dict(fen=c["fen"], source=f"surprise {c['id']}", dtz=None) for c in cases
           if TS._key(c["fen"]) not in have and chess.Board(c["fen"]).turn == chess.WHITE]
    if new:
        TS.read_floors(new, log=lambda *_a: None, sig=sig)
    log(f"OBS miner {sig} surprises: the class's refuted cases {len(cases)} ({len(new)} new to the sample) join it as constraints"
        + (": " + "; ".join(r["fen"] for r in new) if new else ""))
    return rows + new


def mine(sig: str, rows: list, log=print, top: int = TOP, library: str = "73.5b") -> dict:
    """THE AMENDED MINER over a sample -> {sig, levels, choice (a dict or None), best (when beyond the fuse), floor}; prints the OBS lines. `library`: 73.5b's
    atoms (the record) or 73.6's (with the per-axis atoms; the class's constant atoms left out, `constant_atoms`). plan 76.3 (d): the class's refuted cases join the
    sample first (`with_surprises`)."""
    TC.parse_signature(sig)
    t_all = time.monotonic()
    rows = with_surprises(sig, rows, log)
    graph = Graph(chess.WHITE)
    roots = [graph.add(chess.Board(r["fen"])) for r in rows]
    fl = {n: {j for j, r in enumerate(rows) if r[f"floor{n}"] is not None} for n in NETS}      # the sample's floors (technique_sample.py)
    big = int(np.array(graph.rows).max())
    assert big < RADIX, f"an atom reads {big} >= RADIX"
    atoms = LIBRARIES[library]
    dropped = []
    if library != "73.5b":
        dropped = constant_atoms(graph, atoms)
        atoms = tuple(a for a in atoms if a not in dropped)
        log(f"OBS miner {sig} library {library}: the constants of the class left out (a candidate with one equals one without it): "
            + (", ".join(f"{a.sexpr} = {graph.rows[0][COL[a.key]]}" for a in dropped) or "none"))
    lib = {r: candidates(r, atoms) for r in RANKS}
    for r in RANKS:
        for m in lib[r]:
            assert m.reads and m.reads <= set(LS.BASE), m.sexpr
    log(f"OBS miner {sig} library{'' if library == '73.5b' else ' ' + library}: atoms {len(atoms)} ({', '.join(a.sexpr for a in atoms)}; the tempo atom is d), "
        f"candidates rank 1 {len(lib[1])} (atoms, sums, products), rank 2 {len(lib[2])}, rank 3 {len(lib[3])} (lexicographic tuples of distinct atoms); the fuse "
        f"(d, ranks, n) = {FUSE}")
    todo = {(m.sexpr, 0): set(range(len(rows))) - fl[2] for r in RANKS for m in lib[r]}       # no drop "at d = 0": every position off the n = 2 floor
    levels, choice = [], None
    best_seen = None
    for d in DEPTHS:
        for rk in RANKS:
            t0 = time.monotonic()
            e0, x0 = graph.expanded, graph.edges
            need = set()
            for m in lib[rk]:
                need |= todo[(m.sexpr, d - 1)]
            graph.grow([roots[j] for j in sorted(need)], d)
            t_grow = time.monotonic() - t0
            res = []
            for m in lib[rk]:
                prev = sorted(todo[(m.sexpr, d - 1)])
                ok = drops(graph, m, [roots[j] for j in prev], d) if prev else []
                nd = {j for j, o in zip(prev, ok) if not o}
                todo[(m.sexpr, d)] = nd
                res.append((m, nd))
            secs = time.monotonic() - t0
            child = graph.arrays()[1]
            lv = dict(d=d, ranks=rk, candidates=len(lib[rk]), checked=len(need), secs=round(secs, 1), grow_secs=round(t_grow, 1), nodes=len(graph.fens),
                      expanded=graph.expanded, new_expanded=graph.expanded - e0, edges=graph.edges, new_edges=graph.edges - x0, dp_reads=int(len(child)) * d * len(lib[rk]))
            for n in NETS:
                fails = sorted(((m, sorted(nd - fl[n])) for m, nd in res), key=lambda t: (len(t[1]), t[0].simplicity()))
                lv[f"n{n}"] = [(m.sexpr, len(f)) for m, f in fails[:top]]
                log(f"OBS miner {sig} level d={d} ranks={rk} n={n}: candidates {len(lib[rk])}, positions checked {len(need)} (the failures of the level before), "
                    f"best: " + "; ".join(f"{m.sexpr} {len(f)}" for m, f in fails[:top]) + f"; cost {secs:.1f} s (graph growth {t_grow:.1f} s), nodes {len(graph.fens)}, "
                    f"expanded {graph.expanded} (+{graph.expanded - e0}), edges {graph.edges} (+{graph.edges - x0}), AND-OR reads {len(child) * d * len(lib[rk])}")
                if best_seen is None or (len(fails[0][1]), d, rk, n) < (len(best_seen[1]), best_seen[2], best_seen[3], best_seen[4]):
                    best_seen = (fails[0][0], fails[0][1], d, rk, n)
                zero = [m for m, f in fails if not f]
                if zero and choice is None:
                    pick = min(zero, key=lambda m: m.simplicity())
                    choice = dict(measure=pick, d=d, ranks=rk, n=n, tied=len(zero), zero=[m.sexpr for m in sorted(zero, key=lambda m: m.simplicity())])
            levels.append(lv)
            if choice is not None:
                break
        if choice is not None:
            break
    out = dict(sig=sig, levels=levels, choice=choice, floor={2: len(fl[2]), 3: len(fl[3])}, positions=len(rows), refused=dict(graph.refused),
               secs=round(time.monotonic() - t_all, 1), library=library, dropped=[a.sexpr for a in dropped], atoms=[a.sexpr for a in atoms])
    if choice is not None:
        m = choice["measure"]
        log(f"OBS miner {sig} CHOICE: {m.sexpr} -- {m.gloss}; d {choice['d']}, ranks {choice['ranks']}, floor n {choice['n']}; zero failures on {len(rows)} positions; "
            f"{choice['tied']} measure(s) with zero failures at this level ({', '.join(choice['zero'][:6])}{', ...' if choice['tied'] > 6 else ''}); reads {', '.join(sorted(m.reads))}; "
            f"never-rule refusals {dict(graph.refused)}; {out['secs']} s")
        if library != "73.5b":
            log(f"OBS miner {sig} library {library} every zero-failure measure of the chosen level (d={choice['d']}, ranks={choice['ranks']}, n={choice['n']}; "
                f"{choice['tied']}, simplest first): " + "; ".join(choice["zero"]))
    else:
        m, f, d, rk, n = best_seen
        out["best"] = dict(measure=m.sexpr, failures=[rows[j]["fen"] for j in f], d=d, ranks=rk, n=n)
        log(f"OBS miner {sig}: BEYOND THE FUSE {FUSE} -- no measure with zero failures; the best {m.sexpr} fails {len(f)} at d={d} ranks={rk} n={n}: "
            + "; ".join(rows[j]["fen"] for j in f[:12]) + ("; ..." if len(f) > 12 else "") + "; nothing is written")
    out["graph"] = graph
    out["roots"] = roots
    return out


# ---------------------------------------------------------------------------------------------------------------- the entry (73.5b (3), as 73.5 (4))

ENTRY_NAME = "mined technique {sig}"          # the store's word for a class's mined technique (lemma_store.slug: mined-technique-<sig>.json)


def obligation_text(sig: str, m: Measure, d: int, n: int, checked: str) -> str:
    """THE LEAN OBLIGATION of a mined technique, as a statement (what would make the entry PROVED); `checked`: on what it was checked."""
    k = max(d, n)
    return (f"Let S be the positions of the class {sig} with the winning side to move, ours the moves of that side that neither stalemate, nor leave a unit of ours the "
            f"exchange law loses, nor allow a claimable draw, mu p the mixed-radix number of {m.sexpr} at p. OBLIGATION: for every p in S, "
            f"Ideas.Forced.Reach ours theirs mate (fun r => S r and mu r < mu p) {k} p -- a mate within {n} moves (the floor), or a strategy of at most {d} moves after "
            f"which, against every reply, mu is strictly lower or they are mated. Then Ideas.Forced.forced_by_measure (proved) gives the mate from every p in S within "
            f"{k} * (mu p + 1) moves. The obligation is checked {checked}, NOT proved: no theorem of docs/LEAN_INDEX.md states it.")


def entry_for(res: dict, journal: str, note: str = "") -> dict:
    """THE ENTRY of a class's mined technique from the miner's saved result (`save`): kind TECHNIQUE, origin MINED, status CONJECTURE; `measure` = the chosen term,
    its floor and terminal (the floor's mate word), its class, its tempo depth d, its floor's n and the Lean obligation; `evidence` = the run; `note` = what more
    the note says (an earlier entry it replaces)."""
    sig, c = res["sig"], res["choice"]
    m, d, n = measure_of(c["measure"]), c["d"], c["n"]
    lv = res["levels"][-1]
    words = "the mate words mate_in_one / mate_in_two" if n == 2 else "mating.rs's net (net_mate_of, n = 3)"
    sample = res.get("sample") or f"technique_sample: the 73.5 corpus positions + {TS.UNIFORM} uniform won positions, seed {TS.SAMPLE_SEED}"
    checked = "on every position of the class (by the miner's min-max over the class graph)" if res.get("sample") else "on the sample"
    return dict(
        name=ENTRY_NAME.format(sig=sig), kind="TECHNIQUE", reads=sorted(set(m.reads) | {"tempo", "material", "movement", "attack"}), terms=[m.sexpr], status="CONJECTURE", lean=[],
        origin="MINED", plan=["73.5b"], journal=journal,
        measure={"quantity": m.sexpr, "floor": f"a mate within {n} moves found by the engine's mate search ({words})", "terminal": "mate_in_two" if n == 2 else "net_mate",
                 "class": sig, "depth": d, "n": n, "obligation": obligation_text(sig, m, d, n, checked)},
        evidence=dict(uses={}, confirmed=[
            f"technique_miner {sig} ({journal}): zero failures at d={d} ranks={c['ranks']} n={n} on {res['positions']} positions ({sample}); tied at that level: "
            f"{', '.join(c['zero'])}; the graph {lv['nodes']} nodes, {lv['expanded']} expanded, {lv['edges']} (move, reply) pairs; {res['secs']} s"], refuted=[]),
        implementation=["catspace/incubator/analyst/discovery/technique_miner.py:mine", "catspace/incubator/analyst/discovery/technique_miner.py:entry_for",
                        "catspace/incubator/analyst/discovery/technique_question.py:technique_routes", "catspace/incubator/analyst/discovery/agenda.py:_technique_roots"],
        note=(f"Mined by plan 73.5b from the base atoms, not written: in {sig}, a mate at the floor, else a strategy of at most {d} moves that lowers {m.sexpr} against "
              f"every reply; a CONJECTURE -- the obligation is checked {checked}, not proved in Lean." + (f" {note}" if note else "")))


def write_entry(e: dict) -> str:
    """Write the entry into the lemma store (lemmas/<slug>.json) after the schema check; the store reloads."""
    bad = LS.check_entry(e)
    assert not bad, bad
    p = LS.DIR / f"{LS.slug(e['name'])}.json"
    p.write_text(json.dumps(e, indent=1) + "\n")
    LS.load(force=True)
    return str(p)


def explain(sig: str, log=print, tag: str = "", others: bool = True) -> list:
    """THE NEGATIVE, BUG-CHECKED (Kaveh's standing rule): for a run beyond the fuse (the saved result), each failing position of the best measure read again --
    the table's DTZ, the shortest mate the net finds within 4 moves, the measure's ranks there, and the drop at d = 1, 2, 3 by the per-root AND-OR search
    (technique_question.Search, a second implementation of the graph's min-max) -- and how many measures of the library do drop there within 3 (the position
    fails the best measure, not every measure; `others` False skips that count). Prints `OBS miner <sig> failing position ...` and a summary line."""
    from collections import Counter as _C
    from catspace.incubator.analyst.discovery import technique_question as TQ
    got = json.load(open(paths.experiment("technique_miner", f"{sig}{tag}.json")))
    best = got.get("best")
    if not best:
        log(f"OBS miner {sig}: no failing positions (a measure was chosen)")
        return []
    m = measure_of(best["measure"])
    lib = [c for r in RANKS for c in candidates(r)]
    out, summary = [], []
    for f in best["failures"]:
        b = chess.Board(f)
        v = m.value(TM.atom_values(b, b.turn))
        net = TS.net(b, 4)
        drop = [TQ.Search(m, b.turn, v, frozenset(), b.halfmove_clock).force(b, d, 0) for d in DEPTHS]
        k = sum(TQ.Search(c, b.turn, c.value(TM.atom_values(b, b.turn)), frozenset(), b.halfmove_clock).force(b, max(DEPTHS), 0) for c in lib) if others else None
        G.clear_caches()
        line = (f"OBS miner {sig} failing position {f}: DTZ {TC.tablebase().probe_dtz(b)}, the net's shortest mate " + (f"{net[1]} moves ({net[0]})" if net else "none within 4")
                + f"; {best['measure']} = {TQ.show(m, v)}; a drop within d = 1, 2, 3 by the per-root search: {drop}"
                + (f"; measures of the library that drop here within 3: {k} of {len(lib)}" if others else ""))
        summary.append((TC.tablebase().probe_dtz(b), net[1] if net else None, TQ.show(m, v), any(drop)))
        log(line)
        out.append(line)
    log(f"OBS miner {sig} failing positions of {best['measure']}: {len(summary)}; by (DTZ, the net's mate within 4, the measure): "
        + ", ".join(f"{k} {v}" for k, v in sorted(_C((a, b, c) for a, b, c, _d in summary).items(), key=str)) + f"; a drop found by the per-root search at any d: "
        f"{sum(x[3] for x in summary)}")
    return out


def record_evidence(sig: str, tag: str, journal: str, log=print) -> "str | None":
    """PLAN 73.6 (2): the re-mine's result (the saved `<sig><tag>.json`) into the evidence of the class's mined entry (its status and its measure unchanged: the
    whole-class check is EVIDENCE, never the proof) -> the entry's path, or None when the class has no mined entry (the result stays in the journal)."""
    got = json.load(open(paths.experiment("technique_miner", f"{sig}{tag}.json")))
    e = LS.load(force=True).get(ENTRY_NAME.format(sig=sig))
    c = got.get("choice")
    if c is not None:
        line = (f"technique_miner {sig} --whole-class --library {got.get('library')} ({journal}, plan 73.6 (2): the per-axis atoms): the first zero-failure level "
                f"d={c['d']} ranks={c['ranks']} n={c['n']} on {got['positions']} positions; every zero-failure measure there ({c['tied']}): {', '.join(c['zero'])}; "
                f"the constants left out: {', '.join(got.get('dropped') or []) or 'none'}; the entry's measure {'is' if e and e['measure']['quantity'] in c['zero'] else 'is not'} "
                f"among them; {got['secs']} s")
    else:
        b = got.get("best") or {}
        line = (f"technique_miner {sig} --whole-class --library {got.get('library')} ({journal}, plan 73.6 (2): the per-axis atoms): beyond the fuse {FUSE}; the best "
                f"{b.get('measure')} fails {len(b.get('failures', []))} at d={b.get('d')} ranks={b.get('ranks')} n={b.get('n')}; {got['secs']} s")
    if e is None:
        log(f"OBS miner {sig}: no mined entry -- the re-mine's result stays in the journal: {line}")
        return None
    if line not in e["evidence"]["confirmed"]:
        e["evidence"]["confirmed"].append(line)
    p = write_entry(e)
    log(f"OBS miner {sig} evidence recorded in {os.path.relpath(p, str(paths.REPO_ROOT))} (status {e['status']}, measure {e['measure']['quantity']}, unchanged): {line}")
    return p


def save(got: dict, tag: str = "") -> str:
    """The miner's result as JSON under artifacts/experiments/technique_miner/<sig>.json (the levels, the choice or the best and its failing positions)."""
    res = {k: v for k, v in got.items() if k not in ("graph", "roots", "choice")}
    if got["choice"] is not None:
        c = got["choice"]
        res["choice"] = dict(measure=c["measure"].sexpr, d=c["d"], ranks=c["ranks"], n=c["n"], tied=c["tied"], zero=c["zero"])
    out = paths.experiment("technique_miner", f"{got['sig']}{tag}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(res, f, indent=1, default=str)
    return out


# ---------------------------------------------------------------------------------------------------------------- the CLI

def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("signature")
    ap.add_argument("--corpus", default=None, help="the 73.5 corpus file (default: artifacts/experiments/technique_corpus/<sig>_seed1.jsonl)")
    ap.add_argument("--uniform", type=int, default=TS.UNIFORM)
    ap.add_argument("--seed", type=int, default=TS.SAMPLE_SEED)
    ap.add_argument("--explain", action="store_true", help="read the saved result's failing positions again (beyond the fuse), nothing mined")
    ap.add_argument("--write", action="store_true", help="write the saved result's chosen measure into the lemma store as the class's mined technique (nothing mined)")
    ap.add_argument("--whole-class", action="store_true", help="the sample is EVERY position of the class (technique_sample.whole_class); saved as <sig>_class.json")
    ap.add_argument("--from", dest="src", default="", help="with --write / --explain: the saved result's tag ('' the sample, '_class' the whole class)")
    ap.add_argument("--quick", action="store_true", help="with --explain: skip the count of the library's measures that drop at each failing position")
    ap.add_argument("--note", default="", help="with --write: a sentence more for the entry's note (the earlier entry it replaces and why)")
    ap.add_argument("--library", default="73.5b", choices=sorted(LIBRARIES), help="the atoms: 73.5b's (the record) or 73.6's (the per-axis atoms)")
    ap.add_argument("--record", action="store_true", help="with --from: the saved result into the class's mined entry's evidence (nothing mined; plan 73.6 (2))")
    a = ap.parse_args(argv)
    if a.record:
        return record_evidence(a.signature, a.src, time.strftime("%Y-%m-%d"))
    if a.explain:
        return explain(a.signature, tag=a.src, others=not a.quick)
    if a.write:
        res = json.load(open(paths.experiment("technique_miner", f"{a.signature}{a.src}.json")))
        if res.get("choice") is None:
            print(f"OBS miner {a.signature}: no measure was chosen (beyond the fuse): no entry is written")
            return None
        e = entry_for(res, time.strftime("%Y-%m-%d"), a.note)
        print(f"OBS miner {a.signature} entry written: {os.path.relpath(write_entry(e), str(paths.REPO_ROOT))} -- {e['name']}: {e['measure']['quantity']}, d {e['measure']['depth']}, "
              f"n {e['measure']['n']}, {e['status']}, {e['origin']}")
        return e
    lt = "" if a.library == "73.5b" else "_" + a.library.replace(".", "")
    if a.whole_class:
        rows = TS.whole_class(a.signature)
        got = mine(a.signature, rows, library=a.library)
        got["sample"] = "technique_sample.whole_class: every position of the class with the winning side to move"
        print(f"(the levels: {save(got, tag='_class' + lt)})")
        return got
    corpus = a.corpus or paths.experiment("technique_corpus", f"{a.signature}_seed1.jsonl")
    rows = TS.sample(a.signature, corpus, a.uniform, a.seed)
    got = mine(a.signature, rows, library=a.library)
    print(f"(the levels: {save(got, tag=lt)})")
    return got


if __name__ == "__main__":
    main()
