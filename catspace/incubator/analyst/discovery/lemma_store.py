"""PLAN 73.2 THE LEMMA STORE (Kaveh 2026-10-25: "the concept of a tempo and the concept of material are the only base concepts ... everything else should be
derived ... I would expect these things to be in a lemma store, not a fact store"). The store holds the DERIVED WORDS: what the agenda may say beyond the base
(movement and attack, tempo, material), where each came from, what it rests on and whether it is proved. It changes no decision: `name(s)` returns `s`.

  catspace/incubator/analyst/discovery/lemmas/<slug>.json   one entry per word (the schema: lemmas/README.md; plan 73.1)

  python -m catspace.incubator.analyst.discovery.lemma_store report               # OBS lines: totals by kind / status / origin, no citation, no reads
  python -m catspace.incubator.analyst.discovery.lemma_store validate [<dir> ...]  # exit 1 on a bad entry or an uncatalogued word; <dir>: guard snapshots
  python -m catspace.incubator.analyst.discovery.lemma_store uses <trace | qlog.jsonl>   # OBS lemma uses: name count, ...

THE ADAPTER (73.4): every site of the agenda that sets a claim's lemma, a kill reason or a GAP reason passes the string through `name(s)`. A string is CATALOGUED
when it is an entry's name, or it is built by one of the PATTERNS below from entry names (a composition: the chase's "chase: step_after+step_least over X, Y"; a
prefix that is not a word: "after Qxd5: ", "inherited: ", "Q12 <question>: "; a template whose slots are moves, squares and numbers: "our best claim Qxc7 wins 5,
the reply took 6"); or it begins with an entry's name followed by its detail ("tempo arithmetic: the chain has spent ..."). An uncatalogued string is COUNTED
(`UNCATALOGUED`; the root agenda reads it into `ctx.uncatalogued` and the trace gains "GAP: uncatalogued word <s>") and returned unchanged: never an exception in
a run. `validate` fails on it.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

DIR = Path(__file__).resolve().parent / "lemmas"
ROOT = Path(__file__).resolve().parents[4]
LEAN_INDEX = ROOT / "docs" / "LEAN_INDEX.md"

FIELDS = ("name", "kind", "reads", "terms", "status", "lean", "origin", "plan", "journal", "measure", "evidence", "implementation", "note")
OPTIONAL = ("params",)              # plan 75.2: the values the agenda READS from an entry at run time (a GOAL's stake and roles, a VALUATION's worths, a STOP's shape)
KINDS = ("CLOSING", "DEFECT_FIX", "FORCING", "TECHNIQUE", "STOP", "TABLE", "GOAL", "VALUATION", "RULE")      # plan 76.2 (c): RULE -- a rule of the game (given)
from catspace.incubator.analyst.discovery import names as NM  # noqa: E402  (plan 75.2 (c): the store's base reads are names of the names table)
BASE = tuple(NM.READS)                                         # movement, attack, tempo, material: each bound to a kernel predicate and a Lean twin (names.py)
STATUSES = ("PROVED", "CONJECTURE", "REFUTED", "HEURISTIC")
ORIGINS = ("BASE", "HAND", "MINED")
MEASURE_KEYS = ("quantity", "floor", "terminal")
MEASURE_OPTIONAL = ("class", "depth", "n", "obligation")     # plan 73.5b: a technique the agenda ASKS says its class (a material signature), its tempo depth d, its floor's n, its Lean obligation

USES: Counter = Counter()           # entry name -> how many printed strings of this process resolved to it (a claim made, a kill, a GAP: the word was used)
UNCATALOGUED: Counter = Counter()   # printed string -> how many times it resolved to no entry (the root agenda reads the change into ctx.uncatalogued)

_STORE: list = [None]               # the loaded entries {name: entry}, cached
_ERROR: list = [None]               # the load error, if the store did not load (name() then catalogues nothing; validate reports it)
_MEMO: dict = {}                    # printed string -> the entry names it resolves to (a tuple), or None


class LemmaStoreError(ValueError):
    pass


def slug(name: str) -> str:
    """The file name of an entry (without .json): its name in lower case, every run of characters other than letters, digits and '_' one '-', at most 80
    characters ('mate_in_one' the schema and 'mate in one' the rule are two files)."""
    return re.sub(r"[^a-z0-9_]+", "-", name.lower()).strip("-")[:80]


# ---------------------------------------------------------------------------------------------------------------- the entry (73.1)

def check_entry(e: dict, where: str = "") -> list:
    """The schema of one entry (73.1) -> the list of what is wrong (empty: a good entry). The Lean citations are checked by `validate` against the index."""
    bad = []
    at = where or e.get("name", "?")
    if set(e) - set(OPTIONAL) != set(FIELDS):
        bad.append(f"{at}: fields {sorted((set(e) - set(OPTIONAL)) ^ set(FIELDS))} (the schema is exactly {', '.join(FIELDS)}, and may say {', '.join(OPTIONAL)})")
        return bad
    if not isinstance(e["name"], str) or not e["name"]:
        bad.append(f"{at}: name must be a non-empty string")
    if e["kind"] not in KINDS:
        bad.append(f"{at}: kind {e['kind']!r} not in {KINDS}")
    if not isinstance(e["reads"], list) or not e["reads"]:
        bad.append(f"{at}: reads must be a non-empty list of base words {BASE}")
    elif any(r not in BASE for r in e["reads"]):
        bad.append(f"{at}: reads {e['reads']} must be a subset of {BASE}")
    if not isinstance(e["terms"], list) or any(not isinstance(t, str) for t in e["terms"]):
        bad.append(f"{at}: terms must be a list of s-expressions (strings)")
    if e["status"] not in STATUSES:
        bad.append(f"{at}: status {e['status']!r} not in {STATUSES}")
    if not isinstance(e["lean"], list) or any(not isinstance(t, str) for t in e["lean"]):
        bad.append(f"{at}: lean must be a list of theorem names")
    elif (e["status"] == "PROVED") != bool(e["lean"]):
        bad.append(f"{at}: lean must be non-empty if and only if the status is PROVED (status {e['status']}, lean {e['lean']})")
    if e["origin"] not in ORIGINS:
        bad.append(f"{at}: origin {e['origin']!r} not in {ORIGINS}")
    if not isinstance(e["plan"], list) or not all(isinstance(p, str) for p in e["plan"]):
        bad.append(f"{at}: plan must be a list of plan numbers (strings)")
    if not isinstance(e["journal"], str):
        bad.append(f"{at}: journal must be the date (a string)")
    if e["kind"] == "TECHNIQUE":
        m = e["measure"]
        if not isinstance(m, dict) or not set(MEASURE_KEYS) <= set(m) <= set(MEASURE_KEYS) | set(MEASURE_OPTIONAL):
            bad.append(f"{at}: a TECHNIQUE has a measure {{quantity, floor, terminal}} (and may say {', '.join(MEASURE_OPTIONAL)})")
        else:
            bad += _measure_problems(m, at)
    elif e["measure"] is not None:
        bad.append(f"{at}: only a TECHNIQUE has a measure")
    ev = e["evidence"]
    if not isinstance(ev, dict) or set(ev) != {"uses", "confirmed", "refuted"} or not isinstance(ev["uses"], dict) or not isinstance(ev["confirmed"], list) or not isinstance(ev["refuted"], list):
        bad.append(f"{at}: evidence must be {{uses: {{}}, confirmed: [], refuted: []}}")
    elif any(not isinstance(u, dict) or set(u) != set(USE_FIELDS) or not isinstance(u["count"], int) or not isinstance(u["date"], str) for u in ev["uses"].values()):
        bad.append(f"{at}: evidence.uses maps a batch to {{count, date}} (plan 76.3 (e), written by `lemma_store uses --write`)")
    if not isinstance(e["implementation"], list) or not e["implementation"] or any(not isinstance(x, str) or ":" not in x for x in e["implementation"]):
        bad.append(f"{at}: implementation must list 'file:function' strings")
    if not isinstance(e["note"], str) or not e["note"]:
        bad.append(f"{at}: note must be one plain sentence")
    bad += _params_problems(e, at)
    return bad


# ---------------------------------------------------------------------------------------------------------------- plan 75.2 (a): the goal kinds and the valuation
#
# THE GOAL is the outcome (plan 75: don't lose, then win -- the status ladder of plan 59, kept in the worklist's keep arithmetic). The goal KINDS the agenda pursues
# toward it are GOAL entries, each with what the agenda reads of it: the word its questions carry, its basis (a RULE of the game bound to a kernel predicate, or
# a count of a VALUATION entry), its stake (a number, the valuation's worth of the unit a promotion places less the unit it replaces, or the question's own: the
# worth of its target), whether it ends the game, whether it is a keep (don't lose), the question's pattern key naming its location, whether it is chased, and
# whether a kill records the question's reads, and how near (in the goal's own count: a pawn's pushes) a unit must be for a root to be asked (null: every one).
# The VALUATION is one entry: the worth of each unit kind (by its symbol), the unit of a gain, the royal unit.
GOAL_PARAMS = ("question", "basis", "rule", "valuation", "stake", "ends_game", "keep", "about", "chase", "reads_on_kill", "near")
VALUATION_PARAMS = ("worth", "unit", "royal")
# plan 76.2 (c) THE RULE SHELF: a RULE entry is a rule of the game the engine reads -- GIVEN, never derived; in Lean a DEFINITION (generated from rules/chess.gdl), never
# an axiom. Its params say where it lives: `spec` (the generated Lean definition, Ideas.Spec.Chess.*), `gdl` (the spec's relation), `surface` (the Game methods that
# implement it, game.SPEC); its `lean` (when PROVED) the theorems that prove the spec's definition equal to the rule kernel's (Ideas.Spec.LegalEquiv, NextEquiv);
# CONJECTURE with no such theorem, its obstacle in the note. The names table's words (the ontology shelf's entries marked `names`) are the shelf's other half.
RULE_PARAMS = ("spec", "gdl", "surface")
BASES = ("RULE", "VALUATION")


def _params_problems(e: dict, at: str) -> list:
    """plan 75.2: a GOAL or a VALUATION entry has `params` of its schema; a STOP may carry its shape there (a dict); no other kind has params."""
    p = e.get("params")
    k = e["kind"]
    if k == "GOAL":
        if not isinstance(p, dict) or set(p) != set(GOAL_PARAMS):
            return [f"{at}: a GOAL has params exactly {', '.join(GOAL_PARAMS)}"]
        bad = []
        if not isinstance(p["question"], str) or not p["question"]:
            bad.append(f"{at}: params question must be the goal word its questions carry")
        if p["basis"] not in BASES:
            bad.append(f"{at}: params basis {p['basis']!r} not in {BASES}")
        elif (p["basis"] == "RULE") != isinstance(p["rule"], str):
            bad.append(f"{at}: params rule names the kernel predicate if and only if the basis is RULE")
        if not isinstance(p["valuation"], str):
            bad.append(f"{at}: params valuation must name the VALUATION entry its stake counts in")
        s = p["stake"]
        if not (s is None or (isinstance(s, int) and not isinstance(s, bool)) or (isinstance(s, dict) and set(s) == {"unit", "less"} and all(isinstance(v, str) for v in s.values()))):
            bad.append(f"{at}: params stake must be a number, {{unit, less}} (two unit kinds of the valuation) or null (the question's own)")
        for b in ("ends_game", "keep", "chase", "reads_on_kill"):
            if not isinstance(p[b], bool):
                bad.append(f"{at}: params {b} must be true or false")
        if not isinstance(p["about"], str):
            bad.append(f"{at}: params about must be the question's pattern key")
        if not (p["near"] is None or (isinstance(p["near"], int) and not isinstance(p["near"], bool) and p["near"] >= 1)):
            bad.append(f"{at}: params near must be null or a whole number >= 1")
        return bad
    if k == "VALUATION":
        if not isinstance(p, dict) or set(p) != set(VALUATION_PARAMS):
            return [f"{at}: a VALUATION has params exactly {', '.join(VALUATION_PARAMS)}"]
        bad = []
        if not isinstance(p["worth"], dict) or not p["worth"] or any(not isinstance(v, int) or isinstance(v, bool) for v in p["worth"].values()):
            bad.append(f"{at}: params worth must map each unit kind to a whole number")
        if not isinstance(p["unit"], int) or p["unit"] < 1:
            bad.append(f"{at}: params unit (the least gain) must be a whole number >= 1")
        if p.get("royal") not in (p["worth"] if isinstance(p["worth"], dict) else {}):
            bad.append(f"{at}: params royal must be a unit kind of the worth table")
        return bad
    if k == "RULE":
        if not isinstance(p, dict) or set(p) != set(RULE_PARAMS):
            return [f"{at}: a RULE has params exactly {', '.join(RULE_PARAMS)}"]
        bad = []
        if not isinstance(p["spec"], str) or not p["spec"].startswith("Ideas.Spec."):
            bad.append(f"{at}: params spec must name the generated Lean definition (Ideas.Spec.*)")
        if not isinstance(p["gdl"], str) or not p["gdl"]:
            bad.append(f"{at}: params gdl must name the spec's relation")
        if not isinstance(p["surface"], list) or not p["surface"] or not all(isinstance(x, str) for x in p["surface"]):
            bad.append(f"{at}: params surface must list the Game methods implementing it")
        return bad
    if k == "STOP" and p is not None:
        if not isinstance(p, dict) or p.get("context") not in STOP_CONTEXTS:
            return [f"{at}: a STOP's params say the context it applies in ({', '.join(STOP_CONTEXTS)}) and its shape"]
        return []
    if k in WORD_PARAMS and p is not None:
        allowed = WORD_PARAMS[k]
        if not isinstance(p, dict) or not p or any(f not in allowed or not isinstance(v, allowed[f]) or isinstance(v, bool) != (allowed[f] is bool) for f, v in p.items()):
            return [f"{at}: a {k}'s params are among {', '.join(f'{f} ({t.__name__})' for f, t in allowed.items())}"]
        if "units" in p and (not p["units"] or not all(isinstance(u, str) and len(u) == 1 for u in p["units"])):
            return [f"{at}: params units must be unit kinds (one letter each)"]
        if "order" in p and (not p["order"] or not all(isinstance(c, list) and len(c) == 2 and isinstance(c[0], str) and c[1] in (1, -1) for c in p["order"])):
            return [f"{at}: params order must be [component, sign] pairs, sign 1 (fewer first) or -1 (more first)"]
        return []
    if p is not None:
        return [f"{at}: only a GOAL, a VALUATION, a RULE, a STOP, a FORCING, a CLOSING or a DEFECT_FIX entry has params"]
    return []


# ---------------------------------------------------------------------------------------------------------------- plan 75.2 (c): what a printed word's entry says
# The agenda used to read a word's properties off its printed text (the forced words by their names, the schema families by their prefixes). Plan 75.2 (c) makes them FIELDS
# of the word's entry: a FORCING word is `forced` (its replies are forced: the chase follows it below the root, the judge may let it decide), ranks the chase's routes (`rank`
# when the route's word IS the entry's name, `rank_any` when it begins with it), is `walked` (a plain check: the walk of checks tests it); a CLOSING word `reads_every_reply` (its
# claim is no fact of another position), was made by the `chase`, is followed to its `leaf_read`, is a gain within the `window`, or is a `draw`.
# Plan 75.2 (d) adds: a FORCING word `at_king` (its threat is on their royal unit: the king's zone joins its route's frame); a CLOSING word `refutes` (a route killed by it is REFUTED: the kill proves a loss), the `units` a measure holds for (the unit kinds of ours beside the
# royal unit and the pawns), the `order` of a tie-break key (its components, each [name, sign]: +1 fewer first, -1 more first); a DEFECT_FIX its arithmetic's `ahead` (the
# moves the side to move has in hand over the count it is compared with).
WORD_PARAMS = {"FORCING": {"forced": bool, "rank": int, "rank_any": int, "walked": bool, "at_king": bool},
               "CLOSING": {"reads_every_reply": bool, "chase": bool, "leaf_read": bool, "window": bool, "draw": bool, "refutes": bool, "units": list, "order": list, "terminal": bool},
               "DEFECT_FIX": {"ahead": int}}
_WORDS: list = [None, None]          # (the store object, the derived tables)


def _word_tables() -> dict:
    store = load()
    if _WORDS[0] is not store:
        forcing = [(n, e.get("params") or {}) for n, e in store.items() if e["kind"] == "FORCING"]
        ranked = [(n, p) for n, p in forcing if "rank" in p or "rank_any" in p]
        _WORDS[0], _WORDS[1] = store, dict(
            forced=tuple(n for n, p in forcing if p.get("forced")),
            exact={n: p["rank"] for n, p in ranked if "rank" in p},
            prefix=sorted(((n, p["rank_any"]) for n, p in ranked if "rank_any" in p), key=lambda t: -len(t[0])),
            rest=1 + max([p.get("rank", p.get("rank_any", 0)) for _n, p in ranked] or [-1]),
            walked=frozenset(n for n, p in forcing if p.get("walked")),
            at_king=frozenset(n for n, p in forcing if p.get("at_king")))
    return _WORDS[1]


def forced_words() -> tuple:
    """The FORCING words whose replies are forced (their entries' `forced`): a printed route word that begins with one of them is forced."""
    return _word_tables()["forced"]


def forcing_rank(why: str) -> int:
    """The chase's order of a route by its word: the entry the word IS (`rank`), else the longest entry it begins with (`rank_any`), else after every ranked word."""
    t = _word_tables()
    if why in t["exact"]:
        return t["exact"][why]
    return next((r for n, r in t["prefix"] if why.startswith(n)), t["rest"])


def walked(why: str) -> bool:
    """Is the route's word the one the walk of checks tests (a FORCING entry the word IS, `walked`)?"""
    return why in _word_tables()["walked"]


def at_king(why: str) -> bool:
    """Is the route's word a FORCING entry whose threat is on their royal unit (`at_king`: the frame of its route takes the king's zone -- its square and the squares next to it)?"""
    return why in _word_tables()["at_king"]


def says(word: str, field: str) -> bool:
    """Does an entry the printed word resolves to (the word itself, its composition's head, its slots) say `field`? (False when the word is not catalogued.)"""
    store = load()
    return any((store[n].get("params") or {}).get(field) for n in (resolve(word) or ()) if n in store)


def params_of(name: str) -> dict:
    """The params of the entry named `name` (the agenda reads a word's shape there: a measure's units, a key's order, an arithmetic's `ahead`). KeyError when the entry or its params are missing."""
    p = load()[name].get("params")
    if p is None:
        raise KeyError(f"the store entry {name!r} has no params")
    return p


def head_says(word: str, field: str) -> bool:
    """Does the entry the printed word is headed by (the word itself, or its composition's head) say `field`?"""
    r = resolve(word)
    if not r:
        return False
    e = load().get(r[0])
    return bool(e is not None and (e.get("params") or {}).get(field))


# ---------------------------------------------------------------------------------------------------------------- plan 75.2 (b): the stops
# A STOP's params hold its SHAPE (the counts it stops at, the goals it judges) and the CONTEXT it applies in: 'root' (the root agenda's question budget), 'chase' (the chase's
# stops: the clock, tempo arithmetic, the second check, the fuse, the walk of checks, the follow list), 'judge' (the cost judge's cut). The agenda asks `stops_for(context)` and
# reads each stop's shape at run time; a stop's GAP text begins with the entry's name.
STOP_CONTEXTS = ("root", "chase", "judge")
_STOPS: list = [None, None]          # (the store object, {context: {name: params}})


def stops_for(context: str) -> dict:
    """The STOP entries that apply in `context`: {entry name: its params} (built once per loaded store)."""
    store = load()
    if _STOPS[0] is not store:
        by: dict = {c: {} for c in STOP_CONTEXTS}
        for n, e in store.items():
            if e["kind"] == "STOP" and e.get("params") is not None:
                by[e["params"]["context"]][n] = e["params"]
        _STOPS[0], _STOPS[1] = store, by
    return _STOPS[1][context]


class Valuation:
    """A VALUATION entry as the agenda reads it: the worth of a unit (by its symbol, either case), the unit of a gain, the royal unit (worth nothing: its loss is terminal)."""

    def __init__(self, e: dict):
        p = e["params"]
        self.name = e["name"]
        self.worth = dict(p["worth"])
        self.unit = p["unit"]
        self.royal = p["royal"]

    def of(self, symbol: str) -> int:
        return self.worth[symbol.lower()]

    def worth_of(self, unit) -> int:
        """The worth of a unit (an object with `.symbol()`, the domain's piece), 0 for none."""
        return 0 if unit is None else self.worth[unit.symbol().lower()]


class Goal:
    """A GOAL entry as the agenda reads it (plan 75.2 (a)). `stake` is a number, or None: the question's own (the worth of its target)."""

    def __init__(self, e: dict, valuation: Valuation):
        p = e["params"]
        self.name, self.entry = e["name"], e
        self.question, self.basis, self.rule = p["question"], p["basis"], p["rule"]
        self.valuation = p["valuation"]
        self.ends_game, self.keep, self.about = p["ends_game"], p["keep"], p["about"]
        self.chase, self.reads_on_kill, self.near = p["chase"], p["reads_on_kill"], p["near"]
        s = p["stake"]
        self.places = s["unit"] if isinstance(s, dict) else None          # (a promotion: the unit kind it places)
        self.stake = valuation.of(s["unit"]) - valuation.of(s["less"]) if isinstance(s, dict) else s

    @property
    def promotes(self) -> bool:
        """The goal is the promotion rule (it places a unit)."""
        return self.places is not None

    @property
    def gains(self) -> bool:
        """The goal raises the valuation (a count, not a keep)."""
        return self.basis == "VALUATION" and not self.keep

    def __repr__(self) -> str:
        return f"Goal({self.name!r}, {self.question!r}, stake {self.stake})"


class _NoGoal:
    """What a question kind no GOAL entry names reads as (a fork, a technique, a draw, a fix question): no role, no stake."""
    name = question = basis = rule = valuation = about = stake = places = entry = near = None
    ends_game = keep = chase = reads_on_kill = promotes = gains = False


NO_GOAL = _NoGoal()


class GoalKinds:
    """THE GOAL KINDS OF THE STORE (plan 75.2 (a)): every GOAL entry by the word its questions carry, the four roles the agenda reads -- the TERMINAL (the goal that
    ends the game: its stake is above every count), the PROMOTION (the rule that places a unit), the GAIN (raise the valuation), the KEEP (don't lose it) -- and the
    VALUATION they count in. Built from the loaded store; `goal_kinds()` rebuilds it when the store object changes (a test's patched store)."""

    def __init__(self, store: dict):
        vals = {n: Valuation(e) for n, e in store.items() if e["kind"] == "VALUATION"}
        goals = [e for e in store.values() if e["kind"] == "GOAL"]
        self.by_question = {}
        for e in goals:
            v = vals.get(e["params"]["valuation"])
            if v is None:
                raise LemmaStoreError(f"{e['name']}: its valuation {e['params']['valuation']!r} is no VALUATION entry")
            g = Goal(e, v)
            if g.question in self.by_question:
                raise LemmaStoreError(f"{e['name']}: two GOAL entries carry the question word {g.question!r}")
            self.by_question[g.question] = g
        gs = list(self.by_question.values())

        def one(what, pred):
            hit = [g for g in gs if pred(g)]
            if len(hit) != 1:
                raise LemmaStoreError(f"the store must have exactly one {what} GOAL entry (it has {len(hit)})")
            return hit[0]
        self.terminal = one("game-ending", lambda g: g.ends_game)
        self.promotion = one("promotion", lambda g: g.promotes)
        self.gain = one("gain", lambda g: g.gains)
        self.keep = one("keep", lambda g: g.keep)
        names = {g.valuation for g in gs}
        if len(names) != 1:
            raise LemmaStoreError(f"the GOAL entries count in {len(names)} valuations: one is read")
        self.valuation = vals[names.pop()]

    def of(self, word: str) -> "Goal | _NoGoal":
        """The GOAL entry whose questions carry `word` (a question's or a claim's goal); NO_GOAL for a kind no GOAL entry names."""
        return self.by_question.get(word, NO_GOAL)


_GOALS: list = [None, None]          # (the store object it was built from, the GoalKinds)


def goal_kinds() -> GoalKinds:
    """THE GOAL KINDS of the loaded store, built once per store object (plan 75.2 (a): the agenda reads them at run time)."""
    store = load()
    if _GOALS[0] is not store:
        _GOALS[0], _GOALS[1] = store, GoalKinds(store)
    return _GOALS[1]


def _measure_problems(m: dict, at: str) -> list:
    """plan 73.5b: the optional parts of a technique's measure -- `class` a material signature (technique_corpus.parse_signature), `depth` and `n` whole numbers
    of at least 1, `obligation` a statement (a string)."""
    bad = []
    if "class" in m:
        from catspace.incubator.analyst.discovery import technique_corpus as TC
        try:
            TC.parse_signature(m["class"])
        except (ValueError, AttributeError):
            bad.append(f"{at}: measure class {m['class']!r} is no material signature")
    for k in ("depth", "n"):
        if k in m and not (isinstance(m[k], int) and m[k] >= 1):
            bad.append(f"{at}: measure {k} must be a whole number >= 1")
    if "obligation" in m and not (isinstance(m["obligation"], str) and m["obligation"]):
        bad.append(f"{at}: measure obligation must be a statement")
    return bad


def _read_dir(d: Path) -> tuple:
    """-> ({name: entry}, [what is wrong]) for every *.json of `d`."""
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
    """Every entry {name: entry}, cached; validated on load (the schema, the file names, no duplicate): a bad store raises LemmaStoreError (the failure is cached
    too: a run with a broken store does not re-read the directory per word -- every word is then uncatalogued, and `validate` says why)."""
    if force or (_STORE[0] is None and _ERROR[0] is None):
        out, bad = _read_dir(DIR)
        _MEMO.clear()
        _STORE[0], _ERROR[0] = (None, "; ".join(bad)) if bad else (out, None)
    if _ERROR[0] is not None:
        raise LemmaStoreError(_ERROR[0])
    return _STORE[0]


def get(name: str) -> "dict | None":
    return load().get(name)


# ---------------------------------------------------------------------------------------------------------------- the adapter (73.4)

SAN = r"(?:[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?|O-O-O|O-O)[+#]?|\?"
SQ = r"[a-h][1-8]"
INT = r"[+-]?\d+"
PIECE = r"[KQRBNP]"

# THE PATTERNS: how the engine prints a word with what is not a word around it. Each is (kind, regex, the entries it names, the groups resolved again).
# kind: 'prefix' (what is before the word is a position in the trace -- a reply, a question, a tier -- not a word), 'suffix' (likewise after it), 'composition'
# (a word built from words: every slot is resolved again and must be catalogued; a slot named `list` is a ", "-joined sequence of words), 'template' (a word
# whose text carries moves, squares and numbers in slots; the entry's name writes each slot as "..."). A pattern's entries are words of the store.
PATTERNS: list = [
    # prefixes: not words
    ("prefix", r"inherited: (?P<s>.*)", (), ("s",)),                                              # a kill served by the inheritance or the fact store (52.12 (3), 52.19 (1))
    ("prefix", r"after (?:" + SAN + r"): (?P<s>.*)", (), ("s",)),                                  # a route's reply (the chase, 52.12)
    ("prefix", r"after (?:" + SAN + r") (?P<s>they mate us|stalemate)", (), ("s",)),              # ... the rule closes it with no agenda (we have no legal move)
    ("prefix", r"(?:" + SAN + r"): (?P<s>.*)", (), ("s",)),                                        # a route (the leaf read's downgrade, 52.10 / 64.0)
    ("prefix", r"Q\d+ (?P<question>.*)", (), ("question",)),                                     # a question and its text: the reason follows the first ": " that resolves
    ("prefix", r"(?P<q>.+) \(stake -?\d+\): (?P<s>.*)", (), ("s",)),                               # a question not asked (the clock, 52.21 (3))
    ("prefix", r"\[(?:tier |pool |tier pool )\d+\] (?P<s>.*)", (), ("s",)),                        # a wider tier's or a pass's line (59, 69.1)
    ("prefix", r"carried from tier \d+: (?P<s>.*)", (), ("s",)),                                   # a proof carried across the tiers (66.1)
    # suffixes: not words
    ("suffix", r"(?P<s>.*) \(standing kill: nothing the inner agenda read after (?:" + SAN + r") (?:" + SAN + r") has changed\)", ("standing kill",), ("s",)),
    ("suffix", r"(?P<s>.*) \(seen before with no more fuse: not re-run\)", (), ("s",)),
    # compositions: words built from words
    ("composition", r"chase: step_after\+step_least over (?P<list>.+)", ("chase: step_after+step_least",), ("list",)),
    ("composition", r"chain stop: (?:" + SAN + r") not followed past (?:" + SAN + r") \((?P<s>.*)\)", (), ("s",)),
    ("composition", r"the (?P<k>[a-z ]+?) defect stands \(.*\) and no fix is new there", ("the ... defect stands (...) and no fix is new there",), ("k",)),
    ("composition", r"[+-]\d+ stands only as a balance, their .+? wins -?\d+ \(lemma (?P<L>.+)\), no parry",
     ("... stands only as a balance, their ... wins ... (lemma ...), no parry",), ("L",)),
    ("composition", r"[+-]\d+ stands only as a balance and their .+? \(wins -?\d+, lemma (?P<L>.+?)\) is not parried YET: .*",
     ("... stands only as a balance and their ... (wins ..., lemma ...) is not parried YET: ...",), ("L",)),
    ("composition", r"refuted at the leaf: their .+? wins -?\d+ \(lemma (?P<L>.+?)\) after .*", ("refuted at the leaf",), ("L",)),
    ("composition", r"the threat \(.+?\) still binds \(\+\d+\); (?P<c>.*)", ("the threat ... still binds",), ("c",)),
    ("composition", r"no fix for '(?P<k>.+?)' in this slice", ("no fix for '...' in this slice",), ("k",)),
    # templates: one word, its text carries moves, squares and numbers
    ("template", r"chase fuse: (?:" + SAN + r") not followed past (?:" + SAN + r")(?: \(the inner agenda was cut by the fuse\))?", ("chase fuse",), ()),
    ("template", r"out of time: (?:" + SAN + r") cut by the judge(?: past (?:" + SAN + r")| \(.*\))", ("cut by the judge",), ()),
    ("template", r"budget: .+ \(stake -?\d+\) not asked", ("budget",), ()),
    ("template", r"our best claim .+? wins -?\d+, the reply took -?\d+ \([+-]\d+\)", ("our best claim ... wins ..., the reply took ...",), ()),
    ("template", r"[+-]\d+ stands only as a balance and we are in check \(no pass to read their threat, no claim of ours\): not parried YET",
     ("... stands only as a balance and we are in check (no pass to read their threat, no claim of ours): not parried YET",), ()),
    ("template", r"[+-]\d+ stands only as a balance and their question '.*' \(stake -?\d+\) is OPEN with the move passed \(no lemma fires yet\): not read YET",
     ("their open question",), ()),
    ("template", r"open: no lemma fires yet \(72\.2 d\)", ("their open question",), ()),
    ("template", r"stuck_reach \(72\.1\): the priced reach, no reply read", ("stuck_reach",), ()),      # (its claim lemma: a chase lists it among its words)
    ("template", r"[+-]\d+ stands only as a balance; our .+ cannot run and their .+ reaches it in .+ \(lemma stuck_reach, 72\.1\): not parried YET", ("stuck_reach",), ()),
    ("template", r"after (?:" + SAN + r") \([^)]*\) we win -?\d+(?:, .*)?", ("one move answers both threats",), ()),      # (the fact store keeps 80 characters of a kill's reason: the word is read from its head)
    ("template", r"exchange \+\d+ \(.*\), nothing of theirs", ("nothing of theirs",), ()),
]

_COMPILED: list = [None]


def _compiled() -> list:
    if _COMPILED[0] is None:
        _COMPILED[0] = [(k, re.compile(rx, re.S), tuple(ents), tuple(slots)) for k, rx, ents, slots in PATTERNS]
    return _COMPILED[0]


_HEAD_SEP = (": ", " (", ", ", "; ", " -- ")


def _resolve(s: str, store: dict, depth: int = 0, strict: bool = False) -> "tuple | None":
    """The entry names `s` is made of, or None when it is not catalogued. (1) an entry's name; (2) a pattern: its entries and every slot resolved again (a
    `list` slot word by word, a `question` slot after the first ": " that resolves); (3) unless `strict` (an item of a list: a detail would swallow the items after
    it), an entry's name followed by its detail (": ", " (", ", ", "; ", " -- ")."""
    if s in store:
        return (s,)
    if depth > 24:
        return None
    for _k, rx, ents, slots in _compiled():
        m = rx.fullmatch(s)
        if m is None:
            continue
        words = list(ents)
        ok = True
        for g in slots:
            v = m.group(g)
            if v is None:
                continue
            sub = _resolve_list(v, store, depth + 1) if g == "list" else _resolve_question(v, store, depth + 1) if g == "question" else _resolve(v, store, depth + 1)
            if sub is None:
                ok = False
                break
            words += sub
        if ok:
            return tuple(dict.fromkeys(words))
    if strict:
        return None
    best = None
    for n in store:
        if len(n) < len(s) and s.startswith(n) and s[len(n):].startswith(_HEAD_SEP) and (best is None or len(n) > len(best)):
            best = n
    return (best,) if best is not None else None


def _resolve_list(text: str, store: dict, depth: int) -> "tuple | None":
    """A ", "-joined sequence of words (the chase's "over X, Y"; an item may itself be a chase, whose own list runs on in the same text): every item catalogued."""
    cuts = [k for k in _find_all(text, ", ", 0)] + [len(text)]
    seen: dict = {}

    def go(i: int) -> "tuple | None":
        if i in seen:
            return seen[i]
        seen[i] = None
        for j in cuts:
            if j <= i:
                continue
            head = _resolve(text[i:j], store, depth + 1, strict=True)
            if head is None:
                continue
            rest = () if j == len(text) else go(j + 2)
            if rest is not None:
                seen[i] = head + rest
                return seen[i]
        return None
    return go(0)


def _resolve_question(text: str, store: dict, depth: int) -> "tuple | None":
    """A question's text followed by ": " and its reason: the reason after the first ": " that resolves (a question's text may hold ": " itself)."""
    for k in _find_all(text, ": ", 0):
        r = _resolve(text[k + 2:], store, depth + 1)
        if r is not None:
            return r
    return None


def resolve(s: str) -> "tuple | None":
    """`_resolve` against the loaded store, memoised; None when the store did not load."""
    if s in _MEMO:
        return _MEMO[s]
    try:
        store = load()
    except LemmaStoreError:
        return None
    r = _resolve(s, store)
    _MEMO[s] = r
    return r


def name(s):
    """THE ADAPTER: returns `s` unchanged (the same object). A catalogued string counts a use of every word it is made of; an uncatalogued one is counted in
    UNCATALOGUED. Never raises."""
    try:
        if isinstance(s, str):
            r = resolve(s)
            if r is None:
                UNCATALOGUED[s] += 1
            else:
                USES.update(r)
    except Exception:                                   # the adapter must not stop a run (73.4): the word is counted as uncatalogued
        try:
            UNCATALOGUED[str(s)] += 1
        except Exception:
            pass
    return s


def mark() -> tuple:
    """The counters now (a root agenda reads the change of its run): (a copy of UNCATALOGUED, a copy of USES)."""
    return Counter(UNCATALOGUED), Counter(USES)


def since(m: tuple) -> tuple:
    """-> (how many uncatalogued strings were counted since the mark, those strings (distinct, sorted), the uses since the mark)."""
    u0, w0 = m
    new = UNCATALOGUED - u0
    return sum(new.values()), sorted(new), USES - w0


def uses_line(uses: Counter, ident: str = "") -> str:
    """`OBS lemma uses <id>: name count, ...` (the most used first, then by name)."""
    items = sorted(uses.items(), key=lambda t: (-t[1], t[0]))
    return f"OBS lemma uses{' ' + ident if ident else ''}: " + (", ".join(f"{n} {c}" for n, c in items) if items else "none")


# ---------------------------------------------------------------------------------------------------------------- the extractor (73.2: the words a trace prints)

_LEMMA_SLOT = re.compile(r"(?:\(|, |\), )lemma ")
_ENDS = (")", ";", ",", " --", " keeps", " after", " is ")
_EOL_SLOTS = [re.compile(rx) for rx in (
    r"-> KILLED Q\d+: (?P<s>.*)$",                                                    # the root's kill (`_kill`)
    r"(?:" + SAN + r"): killed -- (?P<s>.*)$",                                        # a chased route's kill (`_chase_routes`)
    r": GAP, (?P<s>.*?)(?: \(seen before with no more fuse: not re-run\))?$",        # a chased route's GAP, a fix question's GAP (`_trace_fix`)
    r"-> GAP Q\d+: (?P<s>.*)$",                                                       # a kill relabelled a GAP (62.1c)
    r"-> GAP: (?P<s>.*)$",                                                            # the passer lemma that does not fire (the next lemma to prove)
    r"chase: not started -- (?P<s>.*) \(a GAP, never a kill\)$",                      # a chain stop (`_chain_gap`)
    r"^GAP: uncatalogued word (?P<s>.*)$",                                            # the adapter's own note (73.4): a word no entry names
)]
_JUDGE = re.compile(r"\bjudge: .* cut\b.*-> GAP")
_DECISION = re.compile(r"^DECISION (?P<what>.+?) (?P<value>-?\d+|None) (?P<s>.*) proved=(?:True|False|None)$")


def _lemma_at(line: str, i: int) -> str:
    """The lemma printed at position i of the line: the longest catalogued text that ends at a terminator (or the line's end); else the text to the first terminator."""
    best, first = None, None
    cuts = sorted({j for e in _ENDS for j in _find_all(line, e, i)} | {len(line)})
    for j in cuts:
        cand = line[i:j]
        if not cand:
            continue
        if first is None:
            first = cand
        if resolve(cand) is not None:
            best = cand
    return best if best is not None else (first or line[i:])


def _find_all(s: str, sub: str, start: int) -> list:
    out, k = [], s.find(sub, start)
    while k >= 0:
        out.append(k)
        k = s.find(sub, k + 1)
    return out


def extract_trace(text: str) -> list:
    """The words a trace prints, as (slot, string): every claim's lemma ('lemma'), the DECISION line's lemma, every kill reason ('kill'), every GAP reason ('gap'),
    every cut of the judge ('judge'). The strings are as printed; `resolve` says whether each is catalogued."""
    out = []
    for line in text.splitlines():
        for m in _LEMMA_SLOT.finditer(line):
            out.append(("lemma", _lemma_at(line, m.end())))
        for rx in _EOL_SLOTS:
            m = rx.search(line)
            if m is not None:
                out.append(("gap" if "GAP" in rx.pattern else "kill", m.group("s")))
                break
        if _JUDGE.search(line):
            out.append(("judge", "cut by the judge"))
        m = _DECISION.match(line)
        if m is not None and m.group("s") != "None":
            out.append(("lemma", m.group("s")))
    return out


def harvest(ag) -> list:
    """The words a run SET, read from the agenda object (73.2: the fact store's lemma fields): (slot, string) for every claim and estimated claim of the root,
    every kill and GAP of the root, the decision, and every fact of the run's store (a claim's lemma, a kill's reason)."""
    out = []
    st = ag.state
    out += [("lemma", c.lemma) for c in st.claims + st.estimated]
    out += [("kill", r) for _q, r in ag.kills]
    out += [("gap", g) for g in st.gaps]
    d = ag.decision()
    if d is not None:
        out.append(("lemma", d.lemma))
    facts = ag.ctx.facts.facts if ag.ctx is not None else {}
    for f in sorted(facts.values(), key=lambda f: f.id):
        p = f.payload if isinstance(f.payload, dict) else {}
        if p.get("claim") is not None:
            out.append(("fact", p["claim"].lemma))
        if f.kind == "kill" and p.get("why"):
            out.append(("fact-why", p["why"]))
    return out


def unresolved(pairs: list) -> list:
    """The (slot, string) pairs that are not catalogued (a fact's why is the kill reason cut at 80 characters: it is catalogued when its head is)."""
    bad = []
    for slot, s in pairs:
        if resolve(s) is not None:
            continue
        if slot in ("fact-why", "inherited") and _head_only(s):
            continue
        bad.append((slot, s))
    return bad


def _head_only(s: str) -> bool:
    """A reason cut short (the fact store keeps 80 characters): catalogued when it resolves after its prefix, or begins with an entry's name."""
    try:
        store = load()
    except LemmaStoreError:
        return False
    for _k, rx, _e, _s in _compiled():
        if _k == "prefix":
            m = rx.fullmatch(s)
            if m is not None and _head_only(m.group(_s[0])):
                return True
    return any(s.startswith(n) for n in store)


# ---------------------------------------------------------------------------------------------------------------- the base reads of a term

# THE BASE WORDS each primitive of the vocabulary (base_vocabulary.PRIMITIVES) reads of the position. A primitive absent here names squares or pieces (W, L, a1, edge,
# ours, theirpawns ...) or combines numbers and sets (and, count, min, eq ...): it reads no base word by itself. Chess terms (base_vocabulary.CHESS_TERMS) are
# unfolded first. MOVEMENT: how units move (legal moves, king steps, knight hops, the lines a slider rides, pawn pushes); ATTACK: the squares units bear on (check
# is an attack on the king); TEMPO: whose move and how many moves (`after` / `least` / `most` are a move and the replies to it); MATERIAL: a unit's worth, the exchange.
TERM_READS = {
    "movement": {"moves", "mobility", "reach", "dist", "box", "knight1", "knight2", "lines", "diagonals", "segment", "between", "beyond", "next", "line", "offset",
                 "front", "promo", "pushes", "ahead", "passer", "nearer", "movecount", "after", "least", "most", "attackable", "declined"}
                | {f"forcing_{s}_{v}" for s in ("us", "them") for v in ("1", "3", "5", "9", "mate")},
    "attack": {"attacks", "attacked", "free", "hit", "unhit", "xray", "attackers", "defenders", "check", "attackable", "mobility", "see", "declined"}
              | {f"forcing_{s}_{v}" for s in ("us", "them") for v in ("1", "3", "5", "9", "mate")},
    "tempo": {"pushes", "halfmoves", "ago", "seen", "movecount", "after", "least", "most", "delta", "gained", "lost",
              "turn"}                                                   # (plan 73.6: whose move, a reading of the technique miner's library)
             | {f"forcing_{s}_{v}" for s in ("us", "them") for v in ("1", "3", "5", "9", "mate")},
    "material": {"see", "worth", "maxworth", "balance", "material_us", "material_them", "declined", "without", "mobility", "attackable"}
                | {f"forcing_{s}_{v}" for s in ("us", "them") for v in ("1", "3", "5", "9", "mate")},
}


def term_reads(sexpr: str) -> set:
    """The base words an s-expression reads (its symbols, its chess terms unfolded)."""
    from catspace.incubator.analyst.discovery import base_vocabulary as BV
    t = BV.expand(BV.parse(sexpr))
    syms: set = set()

    def walk(u):
        if isinstance(u, str):
            syms.add(u)
        else:
            for v in u:
                walk(v)
    walk(t)
    return {b for b, ps in TERM_READS.items() if syms & ps}


# ---------------------------------------------------------------------------------------------------------------- the index, the report, the validator

def lean_index(path: Path = LEAN_INDEX) -> dict:
    """{full name: 'theorem' | 'OPEN theorem' | 'def' ...} of docs/LEAN_INDEX.md."""
    out = {}
    for m in re.finditer(r"^- \*\*(OPEN )?(theorem|lemma|def|abbrev|structure|inductive) `([^`]+)`", path.read_text(), re.M):
        out[m.group(3)] = (m.group(1) or "") + m.group(2)
    return out


def citation_problems(store: dict, idx: "dict | None" = None) -> list:
    """A PROVED entry citing a name that is not a proved theorem of the index (absent, a definition, or OPEN)."""
    idx = lean_index() if idx is None else idx
    bad = []
    for n, e in store.items():
        if e["status"] != "PROVED":
            continue
        for t in e["lean"]:
            k = idx.get(t)
            if k is None:
                bad.append(f"{n}: cites {t}, absent from docs/LEAN_INDEX.md")
            elif k not in ("theorem", "lemma"):
                bad.append(f"{n}: cites {t}, a {k} of the index, not a proved theorem")
    return bad


def implementation_problems(store: dict) -> list:
    """An implementation 'file:function' whose file is missing under the repo, or does not define the name (def / fn / class / an assignment; a Lean theorem or
    lemma, plan 73.6: a mined sub-lemma is implemented by its Lean file)."""
    bad, texts = [], {}
    for n, e in store.items():
        for imp in e["implementation"]:
            f, fn = imp.rsplit(":", 1)
            p = ROOT / f
            if f not in texts:
                texts[f] = p.read_text() if p.is_file() else None
            t = texts[f]
            if t is None:
                bad.append(f"implementation: {n}: {f} does not exist")
            elif not re.search(r"(?:\bdef |\bfn |\bclass |\bstatic |\bconst |\btheorem |\blemma |^)" + re.escape(fn) + r"\b\s*[(:=<\[]", t, re.M):
                bad.append(f"implementation: {n}: {f} defines no {fn}")
    return bad


def report(log=print, store: "dict | None" = None) -> list:
    """The OBS lines of the store (73.3)."""
    store = load() if store is None else store
    es = list(store.values())
    idx = lean_index()
    lines = [f"OBS lemma store: entries {len(es)}",
             "OBS lemma store by kind: " + ", ".join(f"{k} {sum(e['kind'] == k for e in es)}" for k in KINDS),
             "OBS lemma store by status: " + ", ".join(f"{k} {sum(e['status'] == k for e in es)}" for k in STATUSES),
             "OBS lemma store by origin: " + ", ".join(f"{k} {sum(e['origin'] == k for e in es)}" for k in ORIGINS),
             "OBS lemma store by base read: " + ", ".join(f"{b} {sum(b in e['reads'] for e in es)}" for b in BASE)]
    cp = citation_problems(store, idx)
    lines.append(f"OBS lemma store PROVED entries whose theorem is missing from LEAN_INDEX ({len(cp)}): " + ("; ".join(cp) if cp else "none"))
    nc = sorted(n for n, e in store.items() if e["status"] == "CONJECTURE")
    lines.append(f"OBS lemma store no citation (CONJECTURE: no theorem of the index proves it as stated -- its implementation cites none, or a definition, or a part) ({len(nc)}): "
                 + (" | ".join(nc) if nc else "none"))
    hs = sorted(n for n, e in store.items() if e["status"] == "HEURISTIC")
    lines.append(f"OBS lemma store HEURISTIC (a stop or a tie-break, no proof obligation) ({len(hs)}): " + (" | ".join(hs) if hs else "none"))
    nr = sorted(n for n, e in store.items() if not e["reads"])
    lines.append(f"OBS lemma store entries with no reads ({len(nr)}): " + (" | ".join(nr) if nr else "none"))
    nt = [e for e in es if e["terms"]]
    terms: dict = {}
    for e in nt:
        for t in e["terms"]:
            terms.setdefault(t, []).append(e["name"])
    lines.append(f"OBS lemma store terms: {sum(len(e['terms']) for e in nt)} s-expressions in {len(nt)} entries ({len(terms)} distinct)")
    for t in sorted(terms, key=lambda x: (len(x), x)):                 # each compiled term with the base words it reads (73.3: the generic part we have)
        lines.append(f"OBS lemma store term {t}: reads {', '.join(sorted(term_reads(t))) or 'none'}; in {' | '.join(terms[t][:4])}{' ...' if len(terms[t]) > 4 else ''}")
    none = sorted((t for t in terms if not term_reads(t)), key=lambda x: (len(x), x))
    lines.append(f"OBS lemma store terms reading no base word ({len(none)}): " + (" | ".join(none) if none else "none"))
    lines.append(f"OBS lemma store patterns: {len(PATTERNS)} (" + ", ".join(f"{k} {sum(p[0] == k for p in PATTERNS)}" for k in ("prefix", "suffix", "composition", "template")) + ")")
    lines += rule_shelf_lines(store, idx)
    lines += unused_lines(store)
    for x in lines:
        log(x)
    return lines


def rule_shelf_lines(store: dict, idx: "dict | None" = None) -> list:
    """plan 76.2 (c) THE RULE SHELF, printed: the RULE entries of the store (each with its generated Spec definition, its spec relation, the Game methods implementing it
    and the theorems proving the definition the kernel's, or why not) and the names table's words (the ontology shelf's `names` entries, each bound to the kernel and a
    Lean twin)."""
    idx = lean_index() if idx is None else idx
    rules = sorted((e for e in store.values() if e["kind"] == "RULE"), key=lambda e: e["name"])
    words = list(NM.NAMES.values())
    out = [f"OBS lemma store RULE shelf: {len(rules) + len(words)} rules -- the RULE entries {len(rules)} (PROVED {sum(e['status'] == 'PROVED' for e in rules)}, "
           f"CONJECTURE {sum(e['status'] == 'CONJECTURE' for e in rules)}) and the names table's words {len(words)} (with a Lean twin in the index "
           f"{sum(1 for n in words if n.lean in idx)})"]
    for e in rules:
        p = e["params"]
        eq = ", ".join(e["lean"]) if e["lean"] else "none (a CONJECTURE: the obstacle is the entry's note)"
        out.append(f"OBS rule {e['name']}: {e['status']}; spec {p['spec']} (GDL {p['gdl']}); surface {', '.join(p['surface'])}; equivalence {eq}")
    for n in words:
        out.append(f"OBS rule (name) {n.word}: kernel {n.kernel}; Lean {n.lean or 'none'}")
    return out


def rule_problems(store: dict, idx: "dict | None" = None) -> list:
    """plan 76.2 (c): a RULE entry whose generated Spec definition is absent from docs/LEAN_INDEX.md, or whose surface names a method the Game protocol does not bind
    to a rule (game.SPEC)."""
    from catspace.incubator.analyst.discovery import game as GM
    idx = lean_index() if idx is None else idx
    bad = []
    for n, e in store.items():
        if e["kind"] != "RULE":
            continue
        if e["params"]["spec"] not in idx:
            bad.append(f"{n}: spec {e['params']['spec']} absent from docs/LEAN_INDEX.md")
        bad += [f"{n}: surface {m} is no rule method of the Game surface (game.SPEC)" for m in e["params"]["surface"] if m not in GM.SPEC]
    return bad


SNAPSHOT_WORDS = ".words.txt"       # the trace guard's file of a run's words beside its trace (scripts/trace_identity.py): "slot<TAB>string" per line


def _snapshot_texts(d: Path) -> list:
    return [p for p in sorted(Path(d).glob("*.txt")) if not p.name.endswith((".reads.txt", ".leaf.txt", ".why.txt", SNAPSHOT_WORDS))]      # (plan 74 (a): the provenance list is no trace)


def _snapshot_words(d: Path) -> list:
    """The (slot, string) pairs of every word list of a snapshot directory."""
    out = []
    for p in sorted(Path(d).glob("*" + SNAPSHOT_WORDS)):
        for ln in p.read_text().splitlines():
            if "\t" in ln:
                out.append(tuple(ln.split("\t", 1)))
    return out


def fixed_run(tier: int = 0) -> list:
    """The two fixed positions of the trace guard (the discovery position, pos2), run at the count budget: -> [(id, the trace, the harvest, the uncatalogued count)]."""
    from catspace.incubator.analyst.discovery.agenda import Agenda
    from catspace.incubator.analyst.discovery.state import State
    fens = {"discovery": ("r3q2k/pQ1nb1pp/6n1/5bP1/3PN3/2N1P3/PPP3B1/R1B3K1 w - - 0 1", "w"),
            "pos2": ("r1bqnr1k/1p1n2p1/p2p2P1/2pPp3/4Pp1b/2N1BP1N/PPP4Q/2KR1B2 w - - 0 1", "w")}
    out = []
    for pid, (fen, us) in fens.items():
        ag = Agenda(State(fen=fen, us=us, pieces=[]), tier=tier)
        ag.ask()
        d = ag.decision()
        trace = ag.trace() + f"\nDECISION {d.what if d else None} {d.value if d else None} {d.lemma if d else None} proved={d.proved if d else None}\n"
        out.append((pid, trace, harvest(ag), ag.ctx.uncatalogued))
    return out


def validate(dirs: "list | None" = None, log=print, run: bool = True) -> int:
    """73.2 THE VALIDATOR -> 0 when the store is good, else 1. Fails on: an entry that breaks the schema (73.1); a PROVED entry citing a theorem absent from
    docs/LEAN_INDEX.md (or a definition, or an OPEN one); an entry with no reads; two entries of one name; an implementation the repo does not define; a term
    reading a base word its entry does not list; a pattern naming no entry; a word printed in a trace guard snapshot (`dirs`: the traces' lemmas, kills, GAPs and
    the judge's cuts, and the word lists: the root's claims, kills, GAPs and every fact's lemma) or set by a run of the two fixed positions (`run`: the same, and
    the run's own uncatalogued counter) that is not catalogued."""
    bad = []
    try:
        store = load(force=True)
    except LemmaStoreError as x:
        log(f"OBS lemma store validate: FAILED to load: {x}")
        return 1
    bad += [f"citation: {x}" for x in citation_problems(store)]
    bad += [f"no reads: {n}" for n, e in store.items() if not e["reads"]]
    try:                                                 # plan 75.2 (a): the goal kinds the agenda reads (one terminal, one promotion, one gain, one keep, one valuation)
        GoalKinds(store)
    except LemmaStoreError as x:
        bad.append(f"goal kinds: {x}")
    bad += [f"names: {x}" for x in NM.problems()]       # plan 75.2 (c): every name bound to a kernel predicate and a Lean twin present in the index
    bad += [f"rule: {x}" for x in rule_problems(store)]  # plan 76.2 (c): every RULE entry's Spec definition in the index, its surface the Game's
    bad += implementation_problems(store)
    for n, e in store.items():
        for t in e["terms"]:
            try:
                miss = term_reads(t) - set(e["reads"])
            except Exception as x:                       # a term the vocabulary does not parse
                bad.append(f"term: {n}: {t!r} does not parse ({x})")
                continue
            if miss:
                bad.append(f"reads: {n} reads {e['reads']} but its term {t[:60]!r} reads {sorted(miss)} too")
    for k, rx, ents, _slots in PATTERNS:
        bad += [f"pattern {rx!r} names {e!r}, not an entry" for e in ents if e not in store]
    seen: Counter = Counter()
    miss: Counter = Counter()
    for d in dirs or []:
        for p in _snapshot_texts(Path(d)):
            pairs = extract_trace(p.read_text())
            seen.update(slot for slot, _ in pairs)
            for slot, s in unresolved(pairs):
                miss[(slot, s)] += 1
        pairs = _snapshot_words(Path(d))                  # the words the run set, the fact store's lemma fields among them (a snapshot written since plan 73.2)
        seen.update(slot for slot, _ in pairs)
        for slot, s in unresolved(pairs):
            miss[(slot, s)] += 1
    if run:
        for pid, trace, got, unc in fixed_run():
            pairs = extract_trace(trace) + got
            seen.update(slot for slot, _ in pairs)
            for slot, s in unresolved(pairs):
                miss[(slot, s)] += 1
            if unc:
                bad.append(f"run {pid}: ctx.uncatalogued {unc}")
    for (slot, s), c in sorted(miss.items(), key=lambda t: (-t[1], t[0])):
        bad.append(f"uncatalogued {slot} ({c}x): {s[:160]}")
    log(f"OBS lemma store validate: entries {len(store)}, patterns {len(PATTERNS)}; words read {sum(seen.values())} ({', '.join(f'{k} {v}' for k, v in sorted(seen.items()))}) "
        f"from {len(dirs or [])} snapshot dir(s)" + (" and the two fixed positions" if run else "") + f"; uncatalogued {sum(miss.values())} ({len(miss)} distinct); problems {len(bad)}")
    for x in bad[:60]:
        log("  FAIL " + x)
    return 1 if bad else 0


def _pairs_of(p: Path) -> list:
    """The (slot, printed string) pairs a run left: a question log (jsonl: every row's `word`), a trace file, or a guard SNAPSHOT DIRECTORY (plan 76.3 (e): every
    case's trace -- its printed lemmas, kills and GAPs, each as often as printed -- and every case's word list, each word once per case: what the run's fact store set)."""
    if p.is_dir():
        out = []
        for t in _snapshot_texts(p):
            out += extract_trace(t.read_text())
        return out + _snapshot_words(p)
    if p.suffix == ".jsonl":
        out = []
        for ln in p.read_text().splitlines():
            try:
                row = json.loads(ln)
            except ValueError:
                continue
            if row.get("word"):
                out.append(("word", row["word"]))
        return out
    return extract_trace(p.read_text())


def tally(path: str) -> tuple:
    """THE RUN TALLY of a run's output -> (uses: entry name -> how many printed strings resolved to it, the strings that resolved to none)."""
    p = Path(path)
    c: Counter = Counter()
    unres: Counter = Counter()
    store = load()
    for slot, s in _pairs_of(p):
        r = resolve(s)
        if r is None and slot == "word":                  # the question log keeps a kill's reason up to its first ':' -- the entry it begins, when only one does
            heads = [n for n in store if n.startswith(s + ":")]
            r = (heads[0],) if len(heads) == 1 else None
        if r is None:
            unres[s] += 1
        else:
            c.update(r)
    return c, unres


USE_FIELDS = ("count", "date")       # plan 76.3 (e): evidence.uses = {source: {count, date}} -- one record per tallied batch, written AFTER the batch (never during a run)


def write_uses(c: Counter, source: str, date: str, d: Path = DIR) -> int:
    """plan 76.3 (e): THE USE COUNTS WRITTEN: every entry's `evidence.uses[source]` = {count: its uses in the batch (0 when none), date: the batch's date}. Each file keeps
    its formatting. -> the entries written."""
    n = 0
    for f in sorted(d.glob("*.json")):
        text = f.read_text()
        e = json.loads(text)
        ascii_ = json.dumps(e, indent=1, ensure_ascii=True) + "\n" == text
        e["evidence"]["uses"][source] = {"count": int(c.get(e["name"], 0)), "date": date}
        f.write_text(json.dumps(e, indent=1, ensure_ascii=ascii_) + "\n")
        n += 1
    load(force=True)
    return n


def uses(path: str, log=print, write: bool = False, source: "str | None" = None, date: "str | None" = None) -> Counter:
    """`OBS lemma uses: name count, ...` of a trace file (a guard snapshot or the analyser's output), a guard snapshot directory, or a question log (jsonl: every row's
    `word`). plan 76.3 (e) `write`: the tally into every entry's evidence.uses under `source` (default: the path's name) with `date` (default: today) -- the step the
    parent runs AFTER a batch (the guard sets, the loop batches), never during a run."""
    import time
    p = Path(path)
    c, unres = tally(path)
    log(uses_line(c))
    if unres:
        log(f"OBS lemma uses not catalogued: {sum(unres.values())} ({len(unres)} distinct): " + " | ".join(s[:80] for s, _ in unres.most_common(10))
            + (" (a question log's word is a kill's reason cut at its first ':')" if p.suffix == ".jsonl" else ""))
    if write:
        src = source or p.name
        n = write_uses(c, src, date or time.strftime("%Y-%m-%d"))
        log(f"OBS lemma uses written: source {src}, entries {n}, used {sum(1 for v in c.values() if v)} ({sum(c.values())} uses)")
    return c


# plan 76.3 (e) THE PRUNING RULE (Kaveh 2026-10-28: "unused entries are pruned by a stated rule, never by a number"): an entry is a PRUNING CANDIDATE when it was never
# used -- no printed claim, kill or GAP resolved to it -- in any batch tallied since it was written (its `journal` date), at least one such batch tallied. The kinds the
# agenda reads through their fields rather than prints (RULE, GOAL, VALUATION) are no candidates. NOTHING IS DELETED by the machinery: the list is for the owner.
NOT_PRINTED = ("RULE", "GOAL", "VALUATION")


def unused(store: dict) -> list:
    """-> [(name, written, the batches tallied since, their sources)] of the pruning candidates (the rule above), by name."""
    out = []
    for n, e in sorted(store.items()):
        if e["kind"] in NOT_PRINTED:
            continue
        since = {src: u for src, u in e["evidence"]["uses"].items() if isinstance(u, dict) and u.get("date", "") >= e["journal"]}
        if since and not any(u.get("count") for u in since.values()):
            out.append((n, e["journal"], len(since), sorted(since)))
    return out


def unused_lines(store: dict) -> list:
    cands = unused(store)
    tallied = sorted({src for e in store.values() for src in e["evidence"]["uses"]})
    out = [f"OBS lemma store unused since written (the pruning rule: never used in the guard sets and loop batches tallied since it was written; RULE, GOAL and "
           f"VALUATION entries, read through their fields, are no candidates; nothing is deleted -- the list is for the owner): {len(cands)} of {len(store)} entries; "
           f"batches tallied {len(tallied)}" + (f" ({', '.join(tallied)})" if tallied else "")]
    for n, written, k, _srcs in cands:
        out.append(f"OBS lemma store unused since {written}: {n} (in {k} batch(es))")
    return out


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "report"
    if cmd == "report":
        report()
    elif cmd == "validate":
        sys.exit(validate(sys.argv[2:]))
    elif cmd == "uses":
        a = sys.argv[2:]
        uses(a[-1] if "--write" not in a else a[a.index("--write") + 1], write="--write" in a,
             source=a[a.index("--source") + 1] if "--source" in a else None, date=a[a.index("--date") + 1] if "--date" in a else None)
    else:
        sys.exit("usage: python -m catspace.incubator.analyst.discovery.lemma_store report | validate [<snapshot dir> ...] | "
                 "uses [--write] <trace, snapshot dir or qlog.jsonl> [--source <name>] [--date YYYY-MM-DD]")
