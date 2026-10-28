"""PLAN 76.1 (a) THE ONTOLOGY SHELF (Kaveh 2026-10-28: "one physical store, three logical shelves" -- RULE / LEMMA / FACT -- with THE ONTOLOGY below the rules:
"whatever we define as an object will therefore get a node in the graph"; "one name, three bindings"; "I don't want to invent a fifth vocabulary").

Every OBJECT kind (a square, a unit, a side, a move, the board, the clock, ...) and every RELATION kind (attack, slide, capture, check, ...) the engine speaks of is
declared ONCE, as one JSON file of `lemmas/ontology/` (the one physical store of the lemma shelf, plan 73: a sibling directory the lemma store does not read),
with ONE NAME and its BINDINGS:

  gdl     the relation (or the fluent, the keyword, an argument slot, a constant) of rules/chess.gdl it is -- the RULE shelf's spelling;
  rust    `<file>:<identifier>` of rust/analyst_core/src -- the Rust relation (rules.rs's Datalog) or the `Pos` method that computes it;
  lean    the Lean definition by its full name: the chess-lean kernel's, or lean/ideas's (present in docs/LEAN_INDEX.md);
  python  the python-chess / Game predicate the engine calls (the names table's `kernel`, plan 75.2 (c));

each null only with its reason (`why`); and (76.1 (b)) `bind`, the translator's binding (scripts/gdl_to_lean.py reads its table from here: an object's Lean type
and its constants' terms or a generated enumeration's name, a fluent's or an input's Lean template), and `sort` (an object's sort of the spec; a relation's
signature by object kinds, checked against the spec's declared one). THE NAMES ARE NOT NEW: an entry is named by the names table's word (names.py: the words we supply, now read from here)
or, for every other relation of the spec, by the spec's own relation name -- no name is invented. `origin`: SUPPLIED (a word of the names table, an object kind,
or a base of the spec: a fluent of its state, an input, a table of facts) or DERIVED (a relation the spec defines by its clauses, or the engine derives). `reads`:
the SUPPLIED base entries a relation reads in the spec (its dependency closure, computed from chess.gdl and checked). `node` (an object) / `edge` (a relation): the
kind's place in the graph network's input (graph_policy.py, 76.1 (d)). `vocabulary` (76.1 (c)): the base vocabulary's primitives (base_vocabulary.PRIMITIVES)
that read the relation -- the term language's atoms ARE the shelf's relations.

THE VALIDATOR (`validate`): the schema; every relation of chess.gdl is an entry's binding or is listed as UNNAMED (printed; the goal is none); every binding
resolves (the GDL relation exists with that slot / constant; the Rust identifier is in the crate's source; the Lean name is in the kernel or the index; the
Python name imports); the stored `reads` equal the spec's; the names table's words are all here; and THE BINDINGS AGREE on a sample of positions where they can
be evaluated -- the GDL interpreter (gdl.py) against python-chess and the Rust core (analyst_core.Pos), and, with `--lean`, the chess-lean kernel's decidable
definitions by `#eval`. A disagreement is a SURPRISE: filed under rules/surprises/ (as plan 76.0's cross-play files them) unless it is a case filed already.

  python -m catspace.incubator.analyst.discovery.ontology report
  python -m catspace.incubator.analyst.discovery.ontology validate [--lean]
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

DIR = Path(__file__).resolve().parent / "lemmas" / "ontology"
ROOT = Path(__file__).resolve().parents[4]
RUST_SRC = ROOT / "rust" / "analyst_core" / "src"
KERNEL_SRC = ROOT / "lean" / "ideas" / ".lake" / "packages" / "ChessLean" / "Chess"
LEAN_INDEX = ROOT / "docs" / "LEAN_INDEX.md"
SPEC = ROOT / "rules" / "chess.gdl"

FIELDS = ("name", "kind", "origin", "sort", "reads", "gdl", "rust", "lean", "python", "bind", "node", "edge", "vocabulary", "names", "means", "why", "plan", "journal")
KINDS = ("OBJECT", "RELATION")
ORIGINS = ("SUPPLIED", "DERIVED")
BINDINGS = ("gdl", "rust", "lean", "python")
GDL_KEYS = {"relation", "fluent", "keyword", "slot", "constant", "args"}
KEYWORDS = ("true", "init", "next")          # the GDL keywords an entry may bind (`true`: the state, the board; `init.*` / `next.*`: the initial and the next state)

_STORE: list = [None]


class OntologyError(ValueError):
    pass


def slug(name: str) -> str:
    """The file name of an entry (the lemma store's rule: lower case, every run of characters other than letters, digits and '_' one '-')."""
    return re.sub(r"[^a-z0-9_]+", "-", name.lower()).strip("-")[:80]


# ---------------------------------------------------------------------------------------------------------------- the entries

def check_entry(e: dict, where: str = "") -> list:
    """The schema of one entry -> what is wrong (empty: a good entry). The bindings are resolved by `validate`."""
    at = where or e.get("name", "?")
    if set(e) != set(FIELDS):
        return [f"{at}: fields {sorted(set(e) ^ set(FIELDS))} (the schema is exactly {', '.join(FIELDS)})"]
    bad = []
    if not isinstance(e["name"], str) or not e["name"]:
        bad.append(f"{at}: name must be a non-empty string")
    if e["kind"] not in KINDS:
        bad.append(f"{at}: kind {e['kind']!r} not in {KINDS}")
    if e["origin"] not in ORIGINS:
        bad.append(f"{at}: origin {e['origin']!r} not in {ORIGINS}")
    if e["kind"] == "OBJECT" and not (e["sort"] is None or isinstance(e["sort"], str) or (isinstance(e["sort"], list) and all(isinstance(s, str) for s in e["sort"]))):
        bad.append(f"{at}: an OBJECT's sort is the spec's sort name (a string), a product of sorts (a list) or null")
    b = e["bind"]
    if b is not None:
        ok = isinstance(b, dict) and (
            (e["kind"] == "OBJECT" and (set(b) in ({"type"}, {"type", "constants"}, {"type", "terms"}, {"enum"})))
            or (e["kind"] == "RELATION" and set(b) == {"template", "params"} and isinstance(b["template"], str) and isinstance(b["params"], list)))
        if not ok:
            bad.append(f"{at}: bind is null, an OBJECT's Lean type ({{type [, constants | terms]}} or {{enum}}) or a RELATION's template ({{template, params}})")
    if e["kind"] == "RELATION" and not (e["sort"] is None or (isinstance(e["sort"], list) and all(isinstance(s, str) for s in e["sort"]))):
        bad.append(f"{at}: a RELATION's sort is its signature (a list of object kinds) or null")
    if not isinstance(e["reads"], list) or any(not isinstance(r, str) for r in e["reads"]):
        bad.append(f"{at}: reads must be a list of entry names")
    g = e["gdl"]
    if g is not None:
        if not isinstance(g, dict) or not g or set(g) - GDL_KEYS or sum(k in g for k in ("relation", "fluent", "keyword")) != 1:
            bad.append(f"{at}: gdl is null or {{relation | fluent | keyword, [slot], [constant], [args]}}")
    for b in ("rust", "lean", "python"):
        if not (e[b] is None or (isinstance(e[b], str) and e[b])):
            bad.append(f"{at}: {b} must be a name or null")
    if e["rust"] is not None and ":" not in e["rust"]:
        bad.append(f"{at}: rust must be '<file>:<identifier>' of rust/analyst_core/src")
    why = e["why"]
    if not isinstance(why, dict) or set(why) - set(BINDINGS) or any(not isinstance(v, str) or not v for v in why.values()):
        bad.append(f"{at}: why must map a null binding to its reason")
    else:
        for b in BINDINGS:
            if (e[b] is None) != (b in why):
                bad.append(f"{at}: the binding {b} is {'null without a reason' if e[b] is None else 'set and has a reason'} (why names exactly the null bindings)")
    n = e["node"]
    if n is not None and (e["kind"] != "OBJECT" or not isinstance(n, dict) or set(n) != {"kind", "order"} or not isinstance(n["kind"], str) or not isinstance(n["order"], int)):
        bad.append(f"{at}: node is null or, on an OBJECT, {{kind, order}} (the graph network's node type)")
    if e["edge"] is not None and (e["kind"] != "RELATION" or not isinstance(e["edge"], str)):
        bad.append(f"{at}: edge is null or, on a RELATION, the graph network's edge name")
    if not isinstance(e["vocabulary"], list) or any(not isinstance(v, str) for v in e["vocabulary"]):
        bad.append(f"{at}: vocabulary must be a list of base vocabulary primitives")
    if not isinstance(e["names"], bool):
        bad.append(f"{at}: names must be true (a word of the names table) or false")
    if not isinstance(e["means"], str) or not e["means"]:
        bad.append(f"{at}: means must be one plain sentence")
    if not isinstance(e["plan"], list) or not all(isinstance(p, str) for p in e["plan"]):
        bad.append(f"{at}: plan must be a list of plan numbers")
    if not isinstance(e["journal"], str):
        bad.append(f"{at}: journal must be the date")
    return bad


def _read_dir(d: Path) -> tuple:
    out, bad = {}, []
    for p in sorted(d.glob("*.json")):
        try:
            e = json.loads(p.read_text())
        except ValueError as x:
            bad.append(f"{p.name}: not JSON ({x})")
            continue
        errs = check_entry(e, p.name) if isinstance(e, dict) else [f"{p.name}: not an object"]
        bad += errs
        if errs:
            continue
        if p.stem != slug(e["name"]):
            bad.append(f"{p.name}: the file is named {slug(e['name'])}.json by the entry's name")
        if e["name"] in out:
            bad.append(f"{p.name}: duplicate name {e['name']!r}")
            continue
        out[e["name"]] = e
    return out, bad


def load(force: bool = False) -> dict:
    """Every entry {name: entry}, cached; a bad shelf raises OntologyError."""
    if force or _STORE[0] is None:
        out, bad = _read_dir(DIR)
        if bad:
            raise OntologyError("; ".join(bad))
        _STORE[0] = out
    return _STORE[0]


def get(name: str) -> dict:
    e = load().get(name)
    if e is None:
        raise OntologyError(f"no shelf entry {name!r}")
    return e


def objects() -> list:
    return [e for e in load().values() if e["kind"] == "OBJECT"]


def relations() -> list:
    return [e for e in load().values() if e["kind"] == "RELATION"]


def names_table() -> list:
    """The entries that are words of the names table (plan 75.2 (c); names.py reads them)."""
    return [e for e in load().values() if e["names"]]


def node_kinds() -> list:
    """76.1 (d): the graph network's node types, by the OBJECT kinds that declare one, in their declared order."""
    return [e["node"]["kind"] for e in sorted((e for e in objects() if e["node"]), key=lambda e: e["node"]["order"])]


def node_of(name: str) -> str:
    """The node type of the OBJECT kind `name` (OntologyError if it declares none)."""
    n = get(name)["node"]
    if n is None:
        raise OntologyError(f"the object {name!r} is no node kind")
    return n["kind"]


def edge_of(name: str) -> str:
    """The edge name of the RELATION kind `name` in the graph network (OntologyError if it declares none)."""
    e = get(name)["edge"]
    if e is None:
        raise OntologyError(f"the relation {name!r} is no edge kind")
    return e


def vocabulary_map() -> dict:
    """76.1 (c): base vocabulary primitive -> the shelf relations it reads (sorted names)."""
    out: dict = {}
    for e in load().values():
        for v in e["vocabulary"]:
            out.setdefault(v, []).append(e["name"])
    return {k: sorted(v) for k, v in out.items()}


# 76.1 (c) THE TERMS ARE THE RELATIONS: the base vocabulary's primitives (base_vocabulary.PRIMITIVES, the Rust compile_term's symbols) each read shelf relations (the
# entries' `vocabulary`); these combine numbers and sets and read none; these are the technique library's own readings (technique_measures.py), bound like primitives
COMBINATORS = ("and", "not", "count", "sum", "or", "min", "max", "diff", "eq", "if", "area", "product")
READINGS = ("turn",)


def _symbols(sexpr: str) -> set:
    from catspace.incubator.analyst.discovery import base_vocabulary as BV
    out: set = set()

    def walk(u):
        if isinstance(u, str):
            out.add(u)
        else:
            for v in u:
                walk(v)
    walk(BV.expand(BV.parse(sexpr)))
    return out


def term_relations(sexpr: str) -> list:
    """76.1 (c): the shelf relations an s-expression of the vocabulary reads (its chess terms unfolded; combinators and numbers read none), sorted. An unbound symbol
    raises OntologyError (the term would read something the shelf does not name)."""
    vm = vocabulary_map()
    out: set = set()
    for s in _symbols(sexpr):
        if s in COMBINATORS or s.lstrip("-").isdigit():
            continue
        if s not in vm:
            raise OntologyError(f"the term {sexpr!r} reads {s!r}, which no shelf entry binds")
        out |= set(vm[s])
    return sorted(out)


def vocabulary_problems() -> list:
    """Every primitive of the vocabulary that reads the position or the board's geometry is bound to a shelf relation; every bound symbol is a primitive (or a reading
    of the technique library)."""
    from catspace.incubator.analyst.discovery import base_vocabulary as BV
    vm = vocabulary_map()
    bad = [f"vocabulary: the primitive {s} is bound to no shelf entry" for s in BV.PRIMITIVES
           if s not in vm and s not in COMBINATORS and s not in BV.CHESS_TERMS and not s.lstrip("-").isdigit()]
    bad += [f"vocabulary: {s} (bound by {vm[s]}) is no primitive of the vocabulary" for s in vm if s not in BV.PRIMITIVES and s not in READINGS]
    return bad


def _spec_rels(g: dict) -> list:
    """The GDL relations (program nodes) a gdl binding names: a relation, the `true` fluent, or a keyword's flattened relations."""
    if "relation" in g:
        return [g["relation"]]
    if "fluent" in g:
        return ["true." + g["fluent"]]
    return []


def covers(e: dict, nodes) -> set:
    """The relations of the spec (program nodes) an entry covers: its relation (unspecialised), its fluent, or every relation of its keyword."""
    g = e["gdl"]
    if g is None or "args" in g or "constant" in g or "slot" in g:
        return set()
    if "keyword" in g:
        return {n for n in nodes if n.startswith(g["keyword"] + ".")} if g["keyword"] != "true" else set()
    return set(_spec_rels(g)) & set(nodes)


# ---------------------------------------------------------------------------------------------------------------- the spec

def program():
    from catspace.incubator.analyst.discovery import gdl_chess as GC
    return GC.program()


BASE_KINDS = ("fluent", "input", "table")


def base_kind(prog, rel: str) -> "str | None":
    """A relation of the spec is a BASE when it is a fluent of the state (`true.*`), an input (`does`, and any other of gdl.INPUTS), or a table (only facts)."""
    from catspace.incubator.analyst.discovery import gdl as GDL
    if rel.startswith("true."):
        return "fluent"
    if rel in GDL.INPUTS:
        return "input"
    cs = prog.rules_of.get(rel)
    if cs and all(not c.body for c in cs):
        return "table"
    return None


def spec_reads(prog, store: dict) -> dict:
    """entry name -> the SUPPLIED base entries its spec relations read (their dependency closure, the entry itself excluded), sorted."""
    owner: dict = {}
    for e in store.values():
        for r in covers(e, prog.nodes):
            owner[r] = e["name"]
    out = {}
    for e in store.values():
        g = e["gdl"]
        if e["kind"] != "RELATION" or g is None:
            out[e["name"]] = []
            continue
        rels = [n for n in prog.nodes if n.startswith(g["keyword"] + ".")] if "keyword" in g else [r for r in _spec_rels(g) if r in prog.nodes]
        seen = set()
        for r in rels:
            seen |= prog.needed((r,))
        names = {owner[r] for r in seen if r in owner and base_kind(prog, r) and store[owner[r]]["origin"] == "SUPPLIED"} - {e["name"]}
        out[e["name"]] = sorted(names)
    return out


def _slot_constants(prog, rel: str, slot: int) -> set:
    """The constants that occur at argument `slot` of `rel` in the spec: in a fact or a rule's head or body (a fluent's true / next / init forms together)."""
    out = set()
    names = {rel, "next." + rel[5:], "init." + rel[5:]} if rel.startswith("true.") else {rel}
    for c in prog.clauses:
        atoms = [c.head] + [(l[1], l[2]) for l in c.body if l[0] in ("pos", "neg")]
        for r, args in atoms:
            if r in names and slot < len(args) and isinstance(args[slot], str) and not args[slot].startswith("?"):
                out.add(args[slot])
    for comp, _rules in prog.strata:                     # the static relations' facts (a rule-derived table such as `square`)
        for r in comp:
            if r in names and r in prog.static:
                out |= {t[slot] for t in prog.static[r].facts if slot < len(t) and isinstance(t[slot], str)}
    return out


def _arity(prog, rel: str) -> "int | None":
    if rel in prog.rules_of:
        return len(prog.rules_of[rel][0].head[1])
    for c in prog.clauses:
        for l in c.body:
            if l[0] in ("pos", "neg") and l[1] == rel:
                return len(l[2])
    return None


def gdl_problems(e: dict, prog) -> list:
    g = e["gdl"]
    if g is None:
        return []
    at = e["name"]
    if "keyword" in g:
        if g["keyword"] not in KEYWORDS:
            return [f"{at}: gdl keyword {g['keyword']!r} not in {KEYWORDS}"]
        if g["keyword"] != "true" and not any(n.startswith(g["keyword"] + ".") for n in prog.nodes):
            return [f"{at}: the spec has no {g['keyword']} relation"]
        return []
    rel = _spec_rels(g)[0]
    if rel not in prog.nodes:
        return [f"{at}: gdl {rel} is no relation of the spec"]
    n = _arity(prog, rel)
    bad = []
    slots = g.get("slot")
    slots = [] if slots is None else (slots if isinstance(slots, list) else [slots])
    if any(not isinstance(s, int) or n is None or not 0 <= s < n for s in slots):
        bad.append(f"{at}: gdl slot {g.get('slot')} outside {rel}'s {n} arguments")
    if "constant" in g:
        if len(slots) != 1 or g["constant"] not in _slot_constants(prog, rel, slots[0]):
            bad.append(f"{at}: gdl constant {g['constant']!r} does not occur at {rel}'s slot {g.get('slot')}")
    for k, v in (g.get("args") or {}).items():
        if not k.isdigit() or n is None or int(k) >= n or v not in _slot_constants(prog, rel, int(k)):
            bad.append(f"{at}: gdl args {k}={v!r} does not occur at {rel}'s slot {k}")
    return bad


# ---------------------------------------------------------------------------------------------------------------- the other bindings

def rust_problems(e: dict) -> list:
    if e["rust"] is None:
        return []
    f, ident = e["rust"].rsplit(":", 1)
    p = RUST_SRC / f
    if not p.is_file():
        return [f"{e['name']}: rust file {f} is not in rust/analyst_core/src"]
    if not re.search(r"\b" + re.escape(ident) + r"\b", p.read_text()):
        return [f"{e['name']}: rust {f} has no identifier {ident}"]
    return []


_KERNEL: list = [None]
_DECL = re.compile(r"^\s*(?:@\[[^\]]*\]\s*)?(?:private\s+|protected\s+|noncomputable\s+)*(theorem|lemma|def|abbrev|structure|inductive|class)\s+([A-Za-z_][\w.'!?]*)")


def kernel_names() -> set:
    """The chess-lean kernel's declarations by their full names (namespaces followed), with every structure's fields and every inductive's constructors."""
    if _KERNEL[0] is None:
        out = set()
        for p in sorted(KERNEL_SRC.glob("*.lean")):
            ns: list = []
            cur = None
            for line in p.read_text().splitlines():
                m = re.match(r"^namespace\s+(\S+)", line)
                if m:
                    ns.append(m.group(1))
                    continue
                m = re.match(r"^end\s+(\S+)", line)
                if m and ns and ns[-1] == m.group(1):
                    ns.pop()
                    continue
                m = _DECL.match(line)
                if m:
                    full = ".".join(ns + [m.group(2)])
                    out.add(full)
                    cur = full if m.group(1) in ("structure", "inductive") else None
                    continue
                if cur is not None:
                    f = re.match(r"^\s+\|\s*([A-Za-z_][\w']*)", line) or re.match(r"^\s+([a-z][\w']*)\s*:", line)
                    if f:
                        out.add(cur + "." + f.group(1))
                    elif line.strip() and not line.startswith(" "):
                        cur = None
        _KERNEL[0] = out
    return _KERNEL[0]


def lean_index() -> dict:
    out = {}
    for m in re.finditer(r"^- \*\*(OPEN )?(theorem|lemma|def|abbrev|structure|inductive) `([^`]+)`", LEAN_INDEX.read_text(), re.M):
        out[m.group(3)] = (m.group(1) or "") + m.group(2)
    return out


def lean_problems(e: dict, idx: dict) -> list:
    n = e["lean"]
    if n is None or n in idx or n in kernel_names():
        return []
    head = n.rsplit(".", 1)[0]                           # a structure field or a constructor of an indexed type
    if head in idx and idx[head] in ("structure", "inductive"):
        return []
    return [f"{e['name']}: lean {n} is neither in the kernel nor in docs/LEAN_INDEX.md"]


def resolve_python(dotted: str):
    """The object a dotted name of the python binding names: the longest importable module, then its attributes (an attribute a class only sets per
    instance -- python-chess's `Board.ep_square` -- read on an instance built with no arguments)."""
    import importlib
    parts = dotted.split(".")
    obj, k = None, 0
    for i in range(len(parts), 0, -1):
        try:
            obj, k = importlib.import_module(".".join(parts[:i])), i
            break
        except ImportError:
            continue
    if obj is None:
        raise ImportError(f"no module in {dotted}")
    for part in parts[k:]:
        try:
            obj = getattr(obj, part)
        except AttributeError:
            if isinstance(obj, type):
                obj = getattr(obj(), part)
            else:
                raise
    return obj


def python_problems(e: dict) -> list:
    if e["python"] is None:
        return []
    try:
        resolve_python(e["python"])
    except Exception as x:
        return [f"{e['name']}: python {e['python']} does not resolve ({x!r})"]
    return []


# ---------------------------------------------------------------------------------------------------------------- the agreement of the bindings

# THE SAMPLE (named): the rule corners where every relation's evaluation differs from the plain case -- castling both ways and through an attacked square, rights in
# part, e.p. legal and pinned, promotions with and without capture, mate, stalemate, the clock at and below its limit, insufficient material of each kind and its
# near misses -- and a middlegame (pos2 of the guard). Each is a legal position (python-chess `is_valid`).
SAMPLE = (
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
    "r3k2r/8/8/8/8/8/5r2/R3K2R w KQkq - 0 1",
    "r3k2r/8/8/8/8/8/8/R3K2R w Kq - 0 1",
    "4k3/8/8/r2pP2K/8/8/8/8 w - d6 0 2",
    "4k3/8/8/4pP2/8/8/8/4K3 w - e6 0 2",
    "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2",
    "1n2k3/P7/8/8/8/8/8/4K3 w - - 0 1",
    "4k3/1P6/8/8/8/8/6p1/4K2R b K - 0 1",
    "r1bqnr1k/1p1n2p1/p2p2P1/2pPp3/4Pp1b/2N1BP1N/PPP4Q/2KR1B2 w - - 0 1",
    "R5k1/5ppp/8/8/8/8/8/6K1 b - - 100 80",
    "7k/5Q2/6K1/8/8/8/8/8 b - - 100 80",
    "8/8/8/4k3/8/8/8/KR6 w - - 99 80",
    "8/8/8/4k3/8/8/8/KR6 w - - 100 80",
    "8/8/8/4k3/8/8/8/KN6 w - - 0 1",
    "8/8/8/4k3/8/3b4/8/KB6 w - - 0 1",
    "8/8/8/4k3/8/2b5/8/KB6 w - - 0 1",
    "8/8/8/4k3/8/8/3n4/KN6 w - - 0 1",
    "8/8/8/4k3/8/8/2B5/KB6 w - - 0 1",
    "6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1",
)


def _sq(f: str, r: str) -> int:
    return "abcdefgh".index(f) + 8 * (int(r) - 1)


def _name(s: int) -> tuple:
    return ("abcdefgh"[s % 8], str(s // 8 + 1))


COLOUR = {True: "white", False: "black"}


def readings_gdl(fen: str, prog=None) -> dict:
    """The spec's readings at a position: relation name -> a set of tuples (strings), by the GDL interpreter on gdl_chess's binding of the board."""
    import chess
    from catspace.incubator.analyst.discovery import gdl_chess as GC
    prog = prog or program()
    b = chess.Board(fen)
    st = GC.state_of(b)
    db = prog.run(st, inputs=GC.inputs_of())
    mover = COLOUR[b.turn]
    legal = sorted(GC.uci_of(m) for role, m in db.facts("legal") if role == mover and m != "noop")
    two = lambda t: t[0] + t[1] + t[2] + t[3]
    out = {
        "cell": {tuple(a) for a in db.facts("true.cell")},
        "tempo": {tuple(a) for a in db.facts("true.control")},
        "castling_right": {tuple(a) for a in db.facts("true.castling_right")},
        "ep_square": {tuple(a) for a in db.facts("true.ep_square")},
        "halfmove": {tuple(a) for a in db.facts("true.halfmove")},
        "attack": {tuple(a) for a in db.facts("attack")},
        "attacked": {tuple(a) for a in db.facts("attacked")},
        "check": {tuple(a) for a in db.facts("check")},
        "legal": set(legal),
        "capture": {u for u in legal if any(two(a[2:]) == u[:4] for a in db.facts("capture"))},
        "promotion": {u for u in legal if len(u) == 5},
        "mate": {("mate",)} if prog.holds(db, "mate") else set(),
        "stalemate": {("stalemate",)} if prog.holds(db, "stalemate") else set(),
        "draw": {("draw",)} if prog.holds(db, "draw") else set(),
        "draw by the fifty-move rule": {("draw_by_fifty_moves",)} if prog.holds(db, "draw_by_fifty_moves") else set(),
    }
    nxt = {}
    for u in legal:
        ns = GC.next_state(st, u, mover, prog)
        nxt[u] = frozenset(ns)
    out["next"] = {(u, s) for u, s in nxt.items()}
    return out


def readings_python(fen: str) -> dict:
    """python-chess's readings of the same relations (the kernel `Game.terminal` reads)."""
    import chess
    from catspace.incubator.analyst.discovery import gdl_chess as GC
    b = chess.Board(fen)
    pm = b.piece_map()
    out = {
        "cell": {_name(s) + (COLOUR[p.color], chess.piece_name(p.piece_type)) for s, p in pm.items()},
        "tempo": {(COLOUR[b.turn],)},
        "castling_right": {GC.RIGHT[s] for s in chess.SquareSet(b.clean_castling_rights())},
        "ep_square": {_name(b.ep_square)} if b.ep_square is not None else set(),
        "halfmove": {(str(min(b.halfmove_clock, GC.CLOCK_LIMIT)),)},
        "attack": {(COLOUR[p.color],) + _name(s) + _name(t) for s, p in pm.items() for t in b.attacks(s)},
        "attacked": {(COLOUR[c],) + _name(t) for c in (True, False) for t in chess.SQUARES if b.is_attacked_by(c, t)},
        "check": {(COLOUR[b.turn],)} if b.is_check() else set(),
        "legal": {m.uci() for m in b.legal_moves},
        "capture": {m.uci() for m in b.legal_moves if b.is_capture(m)},
        "promotion": {m.uci() for m in b.legal_moves if m.promotion},
        "mate": {("mate",)} if b.is_checkmate() else set(),
        "stalemate": {("stalemate",)} if b.is_stalemate() else set(),
        "draw": {("draw",)} if b.is_insufficient_material() else set(),
        "draw by the fifty-move rule": {("draw_by_fifty_moves",)} if b.is_fifty_moves() else set(),
    }
    nxt = set()
    for m in b.legal_moves:
        c = b.copy(stack=False)
        c.push(m)
        nxt.add((m.uci(), frozenset(GC.state_of(c))))
    out["next"] = nxt
    return out


def readings_rust(fen: str) -> dict:
    """The Rust core's readings (analyst_core.Pos: the methods the shelf binds), and its draws at the successors (game.rs `drawn` through `game_draw_check`)."""
    import chess
    import analyst_core as AC
    from catspace.incubator.analyst.discovery import gdl_chess as GC
    p = AC.Pos(fen)
    cells = set()
    attack = set()
    for s in range(64):
        x = p.piece_at(s)
        if x is None:
            continue
        kind, white = x
        cells.add(_name(s) + (COLOUR[white], chess.piece_name(kind)))
        bb = p.attacks(s)
        attack |= {(COLOUR[white],) + _name(s) + _name(t) for t in range(64) if bb >> t & 1}
    parts = p.tkey()
    ep = p.ep_square()
    legal = list(p.legal_ucis())
    out = {
        "cell": cells,
        "tempo": {(COLOUR[p.turn_white()],)},
        "castling_right": {GC.RIGHT[s] for s in range(64) if parts[9] >> s & 1},
        "ep_square": {_name(ep)} if ep is not None else set(),
        "halfmove": {(str(min(int(p.fen().split()[4]), GC.CLOCK_LIMIT)),)},
        "attack": attack,
        "attacked": {(COLOUR[c],) + _name(t) for c in (True, False) for t in range(64) if p.attackers(t, c)},
        "check": {(COLOUR[p.turn_white()],)} if p.is_check() else set(),
        "legal": set(legal),
        "capture": {u for u in legal if p.is_capture(u)},
        "promotion": {u for u in legal if len(u) == 5},
        "mate": {("mate",)} if p.is_checkmate() else set(),
        "stalemate": {("stalemate",)} if p.is_stalemate() else set(),
    }
    nxt = set()
    for u in legal:
        q = p.push(u)
        nxt.add((u, frozenset(GC.state_of(chess.Board(q.fen_exact()), ep=q.ep_square()))))
    out["next"] = nxt
    out["draw at the successors"] = {(u, w) for u, w in AC.game_draw_check(fen, [])}
    return out


def _rust_draws_expected(fen: str, prog=None) -> set:
    """What game.rs `drawn` should read at each successor, by the SPEC's facts in the Rust core's own order (gdl_chess.expected_rust_draw)."""
    import chess
    from catspace.incubator.analyst.discovery import gdl_chess as GC
    prog = prog or program()
    b = chess.Board(fen)
    st = GC.state_of(b)
    out = set()
    for m in b.legal_moves:
        ns = GC.next_state(st, m.uci(), COLOUR[b.turn], prog)
        w = GC.expected_rust_draw(ns, prog)
        if w is not None:
            out.add((m.uci(), w))
    return out


LEAN_RELATIONS = ("cell", "tempo", "castling_right", "ep_square", "halfmove", "attack", "attacked", "check", "legal", "capture", "promotion", "mate",
                  "stalemate", "draw by the fifty-move rule", "next")
_LEAN_HELPERS = r'''
import Chess
open Chess

def fileName (f : Fin 8) : String := String.singleton (fileChar f)
def rankName (r : Fin 8) : String := String.singleton (rankChar r)
def colourName : Color → String | .white => "white" | .black => "black"
def kindName : PieceKind → String
  | .king => "king" | .queen => "queen" | .rook => "rook" | .bishop => "bishop" | .knight => "knight" | .pawn => "pawn"
def sqName (s : Square) : String := fileName s.1 ++ rankName s.2
/-- THE MOVE'S TEXT (the spec's move term: UCI; castling the king's two-file move). -/
def uci : Move → String
  | .normal s t => sqName s ++ sqName t
  | .enPassant s t => sqName s ++ sqName t
  | .promotion s t k => sqName s ++ sqName t ++ (match k with | .queen => "q" | .rook => "r" | .bishop => "b" | .knight => "n")
  | .whiteKingCastle => "e1g1" | .whiteQueenCastle => "e1c1" | .blackKingCastle => "e8g8" | .blackQueenCastle => "e8c8"
def rightName : CastleRight → String
  | .whiteKingSide => "white kingside" | .whiteQueenSide => "white queenside" | .blackKingSide => "black kingside" | .blackQueenSide => "black queenside"
def rows (p : Position) : List String :=
  (allSquares.filterMap fun s => (p.board s).map fun q => s!"cell {fileName s.1} {rankName s.2} {colourName q.color} {kindName q.kind}") ++
  [s!"tempo {colourName p.turn}", s!"halfmove {min p.halfmove 100}"] ++
  ([CastleRight.whiteKingSide, .whiteQueenSide, .blackKingSide, .blackQueenSide].filter (fun r => decide (r ∈ p.castling))).map (fun r => s!"castling_right {rightName r}") ++
  (match p.enPassant with | some e => [s!"ep_square {fileName e.1} {rankName e.2}"] | none => []) ++
  ([Color.white, .black].flatMap fun c => allSquares.flatMap fun s => allSquares.filterMap fun t =>
    if pieceAttacksB p c s t then some s!"attack {colourName c} {fileName s.1} {rankName s.2} {fileName t.1} {rankName t.2}" else none) ++
  ([Color.white, .black].flatMap fun c => allSquares.filterMap fun t =>
    if isAttackedB p c t then some s!"attacked {colourName c} {fileName t.1} {rankName t.2}" else none) ++
  (if inCheckB p p.turn then [s!"check {colourName p.turn}"] else []) ++
  ((legalMoves p).flatMap fun m =>
    [s!"legal {uci m}", s!"next {uci m} {printFen (applyMove p m)}"] ++ (if moveIsCapture p m then [s!"capture {uci m}"] else []) ++
      (if m.promotionKind.isSome then [s!"promotion {uci m}"] else [])) ++
  (if checkmateB p then ["mate mate"] else []) ++ (if stalemateB p then ["stalemate stalemate"] else []) ++
  (if 100 ≤ p.halfmove && !(legalMoves p).isEmpty then ["draw by the fifty-move rule draw_by_fifty_moves"] else [])
'''


def readings_lean(fens) -> "dict | str":
    """The chess-lean kernel's readings of LEAN_RELATIONS at each position, by `#eval` of its decidable definitions (`Chess.parseFen`, `pieceAttacksB`, `isAttackedB`,
    `inCheckB`, `legalMoves`, `applyMove` / `printFen`, `moveIsCapture`, `checkmateB`, `stalemateB`) in one `lake env lean` run: fen -> {relation: set}; a string on a
    failure (no lake, a Lean error). The move's text is the spec's (UCI)."""
    import shutil
    import subprocess
    import tempfile
    import chess
    from catspace.incubator.analyst.discovery import gdl_chess as GC
    lake = shutil.which("lake") or str(Path.home() / ".elan" / "bin" / "lake")
    if not Path(lake).exists():
        return "no lake"
    src = _LEAN_HELPERS + "\n" + "\n".join(
        f'#eval (match parseFen "{f}" with | some p => rows p | none => ["unparsed"]).forM (fun r => IO.println ("{i}|" ++ r))' for i, f in enumerate(fens))
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "OntologyEval.lean"
        path.write_text(src)
        r = subprocess.run([lake, "env", "lean", str(path)], cwd=ROOT / "lean" / "ideas", capture_output=True, text=True)
    if r.returncode != 0:
        return (r.stdout + r.stderr)[-2000:]
    out = {f: {k: set() for k in LEAN_RELATIONS} for f in fens}
    for line in r.stdout.splitlines():
        if "|" not in line:
            continue
        i, row = line.split("|", 1)
        f = fens[int(i)]
        rel = next((k for k in sorted(LEAN_RELATIONS, key=len, reverse=True) if row.startswith(k + " ")), None)
        if rel is None:
            return f"unread row {row!r}"
        rest = row[len(rel) + 1:]
        if rel == "next":
            u, fen2 = rest.split(" ", 1)
            out[f]["next"].add((u, frozenset(GC.state_of(chess.Board(fen2)))))
        elif rel in ("legal", "capture", "promotion"):
            out[f][rel].add(rest)
        elif rel == "castling_right":
            out[f][rel].add(tuple(rest.split()))
        else:
            out[f][rel].add(tuple(rest.split()))
    return out


def _signature(rel: str, a: set, b: set) -> str:
    """A disagreement's class signature: which side has extra tuples, by the relation's first field (a short, stable key)."""
    def head(s):
        return ",".join(sorted({str(x[0] if isinstance(x, tuple) else x)[:12] for x in s}))[:60]
    return f"spec+{head(a - b) or '-'} kernel+{head(b - a) or '-'}"


def agreement(fens=SAMPLE, lean: bool = False, prog=None) -> dict:
    """THE BINDINGS AGREE: at each position, every relation the shelf binds in two systems or more read by each -- the spec (GDL) against python-chess and the Rust core
    (and the kernel, `lean`) -- -> {'compared': {relation: {kernel: positions}}, 'disagree': [(kernel, relation, signature, fen, detail)]}. The Rust core's draws are read
    at the successors (game.rs `drawn`), the spec's by its facts in the Rust core's order. `prog`: another spec (a negative control)."""
    prog = prog or program()
    lean_r = readings_lean(list(fens)) if lean else None
    if isinstance(lean_r, str):
        raise OntologyError(f"the Lean readings failed: {lean_r}")
    compared: dict = {}
    out = []
    for f in fens:
        g = readings_gdl(f, prog)
        kern = {"python": readings_python(f), "rust": readings_rust(f)}
        if lean_r is not None:
            kern["lean"] = lean_r[f]
        for k, rd in kern.items():
            for rel, val in rd.items():
                if rel == "draw at the successors":
                    exp = _rust_draws_expected(f, prog)
                    got = {x for x in val if x[1] != "stalemate"}
                    compared.setdefault("draw (at the successors)", {}).setdefault(k, 0)
                    compared["draw (at the successors)"][k] += 1
                    if exp != got:
                        out.append((k, "draw", _draw_signature(f, exp, got, prog), f, f"spec {sorted(exp)}, Rust game_draw_check {sorted(got)}"))
                    continue
                compared.setdefault(rel, {}).setdefault(k, 0)
                compared[rel][k] += 1
                if g[rel] != val:
                    out.append((k, rel, _signature(rel, g[rel], val), f, f"spec only {sorted(map(str, g[rel] - val))[:4]}, kernel only {sorted(map(str, val - g[rel]))[:4]}"))
    return {"compared": compared, "disagree": out}


def _draw_signature(fen: str, exp: set, got: set, prog) -> str:
    """The cross-play's signature of a Rust draw disagreement (scripts/crossplay.py `_compare_rust`): the first differing successor's readings and its material."""
    import chess
    from catspace.incubator.analyst.discovery import gdl_chess as GC
    b = chess.Board(fen)
    e, g = dict(exp), dict(got)
    for u in sorted(set(e) | set(g)):
        if e.get(u) != g.get(u):
            ns = GC.next_state(GC.state_of(b), u, COLOUR[b.turn], prog)
            return f"spec {e.get(u)} kernel {g.get(u)}: {GC.material_signature(ns)}"
    return "?"


def surprise_keys() -> set:
    from catspace.incubator.analyst.discovery import gdl_chess as GC
    return {(c["kernel"], c["class"], c["signature"]) for c in GC.load_surprises()}


# ---------------------------------------------------------------------------------------------------------------- the validator, the report

def object_of_sort(store: dict) -> dict:
    """76.1 (b) (1): the spec's sort -> the OBJECT kind that is it (a kind's own sort; a value of a sort, the unit kind `pawn`, is not the sort's object)."""
    out = {}
    for e in store.values():
        if e["kind"] == "OBJECT" and isinstance(e["sort"], str) and not (e["gdl"] or {}).get("constant"):
            out.setdefault(e["sort"], []).append(e["name"])
    return out


def sort_problems(store: dict, prog) -> list:
    """76.1 (b) (1), (2): the spec type-checks; every declared sort is one OBJECT kind with a Lean binding, every object's sort is declared; every relation's
    signature (`sort`, by object kinds) is the spec's; every fluent and input of the spec has a binding template."""
    bad = [f"spec type check: {x}" for x in prog.typecheck()]
    obj = object_of_sort(store)
    bad += [f"sort {s}: more than one object kind {ns}" for s, ns in sorted(obj.items()) if len(ns) > 1]
    for s in prog.sorts:
        if s not in obj:
            bad.append(f"sort {s} of the spec is no object kind of the shelf")
        elif store[obj[s][0]]["bind"] is None:
            bad.append(f"sort {s}: its object {obj[s][0]} has no Lean binding")
    for e in store.values():
        if e["kind"] == "OBJECT" and e["sort"] is not None:
            for s in ([e["sort"]] if isinstance(e["sort"], str) else e["sort"]):
                if s not in prog.sorts:
                    bad.append(f"{e['name']}: sort {s} is not declared in the spec")
        if e["kind"] != "RELATION" or e["gdl"] is None or "keyword" in e["gdl"]:
            continue
        rel = _spec_rels(e["gdl"])[0]
        sig = prog.signature(rel)
        want = [obj[s][0] if s in obj else f"?{s}" for s in sig] if sig is not None else None
        if e["sort"] != want:
            bad.append(f"{e['name']}: signature {e['sort']} but the spec's is {want}")
    for r in sorted(prog.nodes):
        if r.startswith("true.") or (r in prog.nodes and base_kind(prog, r) == "input"):
            if not any(covers(e, prog.nodes) & {r} and e["bind"] for e in store.values()):
                bad.append(f"{r}: a fluent or input of the spec with no binding template")
    return bad


def coverage(store: dict, prog) -> tuple:
    """(covered relations of the spec, the UNNAMED ones: no entry binds them), each sorted."""
    covered = set()
    for e in store.values():
        covered |= covers(e, prog.nodes)
    return sorted(covered & prog.nodes), sorted(prog.nodes - covered)


def validate(lean: bool = False, log=print, file: bool = True) -> int:
    """76.1 (a) THE VALIDATOR -> 0 when the shelf is good (see the module's docstring), else 1. `lean`: the kernel's readings by `#eval` too; `file`: a new
    disagreement is filed under rules/surprises/."""
    bad = []
    try:
        store = load(force=True)
    except OntologyError as x:
        log(f"OBS ontology validate: FAILED to load: {x}")
        return 1
    prog = program()
    idx = lean_index()
    for e in store.values():
        bad += gdl_problems(e, prog) + rust_problems(e) + lean_problems(e, idx) + python_problems(e)
    # one entry per relation of the spec (a specialised binding -- a constant, a slot, args -- may share it)
    owners: dict = {}
    for e in store.values():
        for r in covers(e, prog.nodes):
            owners.setdefault(r, []).append(e["name"])
    bad += [f"two entries bind {r}: {ns}" for r, ns in sorted(owners.items()) if len(ns) > 1]
    bad += sort_problems(store, prog)
    bad += vocabulary_problems()
    covered, unnamed = coverage(store, prog)
    reads = spec_reads(prog, store)
    bad += [f"{n}: reads {store[n]['reads']} but the spec's reads are {r}" for n, r in sorted(reads.items()) if store[n]["reads"] != r]
    bad += [f"{n}: reads {r}, no entry" for n, e in sorted(store.items()) for r in e["reads"] if r not in store]
    from catspace.incubator.analyst.discovery import names as NM
    bad += [f"names: {x}" for x in NM.problems()]
    nk = [e["node"]["order"] for e in store.values() if e["node"]]
    if len(set(nk)) != len(nk):
        bad.append(f"node orders repeat: {sorted(nk)}")
    ag = agreement(lean=lean)
    known = surprise_keys()
    new = []
    for k, rel, sig, f, d in ag["disagree"]:
        if (k, rel, sig) in known:
            continue
        new.append((k, rel, sig, f, d))
    if new and file:
        from catspace.incubator.analyst.discovery import gdl_chess as GC
        for k, rel, sig, f, d in {(k, rel, sig): (k, rel, sig, f, d) for k, rel, sig, f, d in new}.values():
            p = GC.file_surprise((k, rel, sig), f, f, d)
            log(f"OBS ontology surprise FILED {p.name}: {k} {rel} [{sig}] at {f}")
    bad += [f"bindings disagree: {k} {rel} [{sig}] at {f}: {d}" for k, rel, sig, f, d in new]
    objs = [e for e in store.values() if e["kind"] == "OBJECT"]
    rels = [e for e in store.values() if e["kind"] == "RELATION"]
    log(f"OBS ontology shelf: entries {len(store)} -- object kinds {len(objs)} (supplied {sum(e['origin'] == 'SUPPLIED' for e in objs)}, derived "
        f"{sum(e['origin'] == 'DERIVED' for e in objs)}), relation kinds {len(rels)} (supplied {sum(e['origin'] == 'SUPPLIED' for e in rels)}, derived "
        f"{sum(e['origin'] == 'DERIVED' for e in rels)}); words of the names table {sum(e['names'] for e in store.values())}; node kinds {node_kinds()}")
    log(f"OBS ontology bindings: gdl {sum(e['gdl'] is not None for e in store.values())}, rust {sum(e['rust'] is not None for e in store.values())}, lean "
        f"{sum(e['lean'] is not None for e in store.values())}, python {sum(e['python'] is not None for e in store.values())} of {len(store)}")
    log(f"OBS ontology spec coverage: relations of rules/chess.gdl {len(prog.nodes)}, bound by an entry {len(covered)}, UNNAMED {len(unnamed)}"
        + (f": {', '.join(unnamed)}" if unnamed else ""))
    vm = vocabulary_map()
    log(f"OBS ontology vocabulary: symbols bound {len(vm)} (by {sum(bool(e['vocabulary']) for e in store.values())} entries); unbound primitives "
        f"{sum(1 for x in bad if 'is bound to no shelf entry' in x)}")
    log(f"OBS ontology agreement ({len(SAMPLE)} positions; spec vs python-chess, the Rust core{' and the Lean kernel' if lean else ''}): readings compared "
        + ", ".join(f"{r} ({'/'.join(f'{k} {n}' for k, n in sorted(ks.items()))})" for r, ks in sorted(ag['compared'].items()))
        + f"; disagreements {len(ag['disagree'])} (filed already {len(ag['disagree']) - len(new)}, new {len(new)})")
    log(f"OBS ontology validate: problems {len(bad)}")
    for x in bad[:60]:
        log("  FAIL " + x)
    return 1 if bad else 0


def report(log=print) -> None:
    store = load()
    for kind in KINDS:
        for e in sorted((e for e in store.values() if e["kind"] == kind), key=lambda e: (e["origin"], e["name"])):
            g = e["gdl"]
            gtxt = "-" if g is None else " ".join(f"{k}={v}" for k, v in g.items())
            log(f"OBS ontology {kind.lower()} {e['name']} ({e['origin'].lower()}): gdl {gtxt}; rust {e['rust'] or '-'}; lean {e['lean'] or '-'}; python {e['python'] or '-'}"
                + (f"; node {e['node']['kind']}" if e["node"] else "") + (f"; edge {e['edge']}" if e["edge"] else ""))


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a or a[0] == "report":
        report()
    elif a[0] == "validate":
        sys.exit(validate(lean="--lean" in a))
    else:
        sys.exit("usage: python -m catspace.incubator.analyst.discovery.ontology report | validate [--lean]")
