"""PLAN 78.53: THE GRAPH, LAYER 2 -- concepts mined bottom-up as patterns over the layer-1 graph (SYSTEM_SPEC sections 4, 6, 11). A measurement and a store in a sandbox: nothing is wired into any player and
nothing is written to the real store. Game-free: this module names no unit, place, line family or tactic; a kind appears only as the raw word the graph's own `kind:<word>` node gives, read at run time.

THE LANGUAGE (the matching and counting are in Rust: rust/engine/tactics/src/sheet/patterns.rs; this is its readable mirror).
  A pattern is a small connected set of RELATION NODES of layer 1 (its SIZE), over variables: u0, u1 ... for units, p0 ... for EMPTY places, l0 ... for lines. The atoms:
      bear(u, t, l)       unit u bears on t (a unit variable, or an empty place variable), along line l where the bearing follows one (`-` where it follows none)
      move(u, t, l)       unit u may move to t
      through(u, t, l, b) (78.54) unit u bears on unit b along l and is of a kind that continues along l; t is the next unit beyond b on l (layer 1's `through` node, BOTH sides)
      shut(u, t, l, b)    unit u runs along l and the one unit b between shuts it before t (first-blocker form; the SIDE TO MOVE ONLY: the rules' `why_not` answers for it alone, and the
                          hypothetical board that would answer for the other side is refused by the game where it leaves the side not to move exposed to a line -- measured by `first_blocker_check`)
  Two atoms that name one object share its variable. The LEVEL is what is said of the units, relative to the side the pattern is counted for:
      0  mine / theirs                                                      (kind-free)
      1  + the goal unit of its side or not                                 (kind-free)
      2  level 1 + on an edge of some direction or not (nearest edge distance 0)  (kind-free; the one place absoluteness enters)
      3  level 1 + the KIND of each unit, the raw symbol of the rules (the kind-bound children of the level 1 pattern of the same atoms)
  No worth enters a pattern. A pattern and its mirror (mine and theirs swapped) are one concept: the value at a position is the count for the side to move less the count for the other.

IDENTITY. The canonical text of a definition (variables by first appearance, atoms in the least order, the least of the pattern and its mirror), its sha256 hash is the concept's identity, `dcN` its display
name by order of first mining; the concepts file (tab separated: dcN, hash, size, level, definition in plain words, our name, EMPTY) is written under the experiment directory; a mapping file
(hash, our name) is READ when it exists and never written by us.

THE ENUMERATION. Size 1: every pattern found; size k grown from size k-1: a pattern is a candidate only when one of its connected parts of one relation node less is kept. A named fuse keeps the
`FUSE` most supported patterns of each size and level (support = the fit rows it occurs in); the rest are counted as dropped by the fuse. A kept pattern of size 2 or 3 is REDUCED to its parts when
(THE REDUCTION TEST, stated before it is run): on the fit games, split by game into two halves A and B, the vector of the counts of its kept parts (for a kind-bound pattern also its kind-free parent, the
level 1 pattern of the same atoms) determines its count on A (no vector has two counts), and on B every vector seen was seen on A with the same count -- counts taken per side per position. Reduced
patterns are recorded with their parts and are not concepts. No threshold is tuned.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

FUSE = 100                       # the named fuse: patterns kept per size and per level, the most supported (a budget of the fit's columns, not tuned to a result)
MAX_SIZE = 3
LEVELS = (0, 1, 2, 3)
TYPES = ("bear", "move", "shut", "through")
NONE = 255


# ------------------------------------------------------------------------------------------------------------------------------ the definition

@dataclass
class Pattern:
    level: int
    atoms: list                                 # (type, actor var, target sort 'u'/'p', target var, line var or None, between var or None)
    sides: list = field(default_factory=list)   # per unit variable: True = mine
    goal: list = field(default_factory=list)
    edge: list = field(default_factory=list)
    kind: list = field(default_factory=list)    # per unit variable: the kind's index (level 3)

    @property
    def size(self) -> int:
        return len(self.atoms)

    def n_units(self) -> int:
        return len(self.sides)


def decode(raw: bytes) -> Pattern:
    """The pattern an encoding of the engine stands for."""
    level, k = raw[0], raw[1]
    atoms, units, i = [], 0, 2
    for _ in range(k):
        ty, a, tsort, t, line, between = raw[i:i + 6]
        i += 6
        atoms.append((TYPES[ty], a, "u" if tsort == 0 else "p", t, None if line == NONE else line, None if between == NONE else between))
        units = max(units, a + 1, t + 1 if tsort == 0 else 0, between + 1 if between != NONE else 0)
    prof = list(raw[i:])
    p = Pattern(level, atoms, [not x for x in prof[:units]])        # the engine writes the bit 1 for theirs
    off = units
    if level >= 1:
        p.goal = [bool(x) for x in prof[off:off + units]]
        off += units
    if level == 2:
        p.edge = [bool(x) for x in prof[off:off + units]]
    if level == 3:
        p.kind = list(prof[off:off + units])
    return p


def canonical_text(p: Pattern, kinds: list) -> str:
    """The canonical definition: the atoms in the engine's order, then the attributes of the units."""
    def atom(x: tuple) -> str:
        ty, a, ts, t, ln, b = x
        return f"{ty}(u{a},{ts}{t},{'-' if ln is None else 'l' + str(ln)}" + ("" if b is None else f",u{b}") + ")"
    atoms = ";".join(atom(x) for x in p.atoms)
    attrs = []
    for u in range(p.n_units()):
        a = [f"u{u}", "mine" if p.sides[u] else "theirs"]
        if p.level >= 1:
            a.append("goal" if p.goal[u] else "plain")
        if p.level == 2:
            a.append("edge" if p.edge[u] else "inner")
        if p.level == 3:
            a.append(kinds[p.kind[u]])
        attrs.append(":".join(a))
    return f"L{p.level}|{atoms}|{','.join(attrs)}"


def hash_of(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def plain_words(p: Pattern, kinds: list) -> str:
    """The definition in plain words (kinds as the graph's words)."""
    def unit(u: int) -> str:
        bits = ["mine" if p.sides[u] else "theirs"]
        if p.level >= 1 and p.goal[u]:
            bits.append("the goal unit of its side")
        if p.level == 2:
            bits.append("on an edge" if p.edge[u] else "not on an edge")
        if p.level == 3:
            bits.append(kinds[p.kind[u]])
        return f"u{u} ({', '.join(bits)})"
    seen: set = set()

    def u_(u: int) -> str:
        if u in seen:
            return f"u{u}"
        seen.add(u)
        return unit(u)
    out = []
    for ty, a, ts, t, ln, b in p.atoms:
        tgt = u_(t) if ts == "u" else f"empty place p{t}"
        line = "" if ln is None else f" along line l{ln}"
        if ty == "bear":
            out.append(f"{u_(a)} bears on {tgt}{line}")
        elif ty == "move":
            out.append(f"{u_(a)} may move to {tgt}{line}")
        elif ty == "through":
            out.append(f"{u_(a)} bears on {u_(b)}{line} and, going on past it, next meets {tgt}")
        else:
            out.append(f"{u_(a)} would bear on {tgt}{line} but {u_(b)} stands between")
    return "; ".join(out)


# ------------------------------------------------------------------------------------------------------------------------------ the store (sandbox)

HEADER = ("dc", "hash", "size", "level", "definition", "our name")


def write_concepts(path: str, rows: list) -> None:
    """rows: [(dcN, hash, size, level, definition)]; our name is written empty."""
    lines = ["\t".join(HEADER)] + ["\t".join([str(x) for x in r] + [""]) for r in rows]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n")


def read_mapping(path: str) -> dict:
    """hash -> our name, from a mapping file (two tab separated columns, hash then name; `#` lines skipped); {} when the file is not there. Written by no code of ours."""
    p = Path(path)
    if not p.is_file():
        return {}
    out = {}
    for line in p.read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            h, _, name = line.partition("\t")
            if name.strip():
                out[h.strip()] = name.strip()
    return out


# ------------------------------------------------------------------------------------------------------------------------------ the enumeration

def unpack(raw: bytes) -> np.ndarray:
    """The sparse counts the engine returns: rows of (row, column, count for the side to move, count for the other)."""
    a = np.frombuffer(raw, dtype=np.uint16).reshape(-1, 6) if raw else np.zeros((0, 6), dtype=np.uint16)
    row = a[:, 0].astype(np.int64) | (a[:, 1].astype(np.int64) << 16)
    col = a[:, 2].astype(np.int64) | (a[:, 3].astype(np.int64) << 16)
    return np.stack([row, col, a[:, 4].astype(np.int64), a[:, 5].astype(np.int64)], axis=1)


def counts_matrix(raw: bytes, n_rows: int, n_cols: int):
    """(us, them): scipy sparse matrices rows x columns of the counts for the side to move and for the other."""
    from scipy import sparse
    t = unpack(raw)
    us = sparse.csc_matrix((t[:, 2].astype(np.float32), (t[:, 0], t[:, 1])), shape=(n_rows, n_cols))
    them = sparse.csc_matrix((t[:, 3].astype(np.float32), (t[:, 0], t[:, 1])), shape=(n_rows, n_cols))
    return us, them


def determined(vectors: np.ndarray, values: np.ndarray, half_a: np.ndarray) -> bool:
    """THE REDUCTION TEST. `vectors` (samples x parts) the counts of the parts, `values` the count of the pattern, `half_a` a boolean mask of the samples in half A. True when the vector
    determines the count on A (no vector has two counts) and every vector seen on B was seen on A with that count."""
    if vectors.shape[1] == 0:
        return False
    _, code = np.unique(vectors, axis=0, return_inverse=True)
    code = code.reshape(-1)
    ca, va = code[half_a], values[half_a]
    cb, vb = code[~half_a], values[~half_a]
    if len(ca) == 0 or len(cb) == 0:
        return False
    table: dict = {}
    for c, v in zip(ca.tolist(), va.tolist()):
        if table.setdefault(c, v) != v:
            return False
    return all(table.get(c) == v for c, v in zip(cb.tolist(), vb.tolist()))


def enumerate_patterns(model, games: list, fit_rows: list, goal_kind, threads: int, fuse: int = FUSE, max_size: int = MAX_SIZE, log=print) -> dict:
    """Mine the patterns on the fit rows only. `games` the engine's (start, moves) wires, `fit_rows` per game the plies. -> {size: {hash int: record}} of the patterns kept by the fuse, with the
    found counts. A record: level, support, total, bytes, parts (hashes of the kept parts), free_parent (hash of the kind-free parent, 0 for kind-free)."""
    kept: dict = {}
    found: dict = {}
    allowed = None
    for size in range(1, max_size + 1):
        f, rec = model.pattern_freq(games, fit_rows, size, list(LEVELS), allowed, goal_kind, threads, fuse)
        found[size] = f
        kept[size] = {h: dict(level=lv, support=sup, total=tot, bytes=bytes(b), parts=parts, free_parent=fp) for h, lv, sup, tot, b, parts, fp in rec}
        allowed = list(kept[size])
        log(f"mined size {size}: found per level {f}, kept {[sum(1 for r in kept[size].values() if r['level'] == lv) for lv in LEVELS]}")
    return {"kept": kept, "found": found}


def evaluate(model, fens: list, keys: list, goal_kind, max_size: int = MAX_SIZE, threads: int = 1) -> np.ndarray:
    """Positions (by text) x keys: the value of each concept, the count for the side to move less the count for the other; a position that is over is a row of zeros."""
    raw = model.pattern_count([(f, []) for f in fens], [[0] for _ in fens], keys, max_size, list(LEVELS), goal_kind, threads)
    us, them = counts_matrix(raw, len(fens), len(keys))
    return (us - them).toarray()
