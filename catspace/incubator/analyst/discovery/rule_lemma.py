"""PLAN 79.44 PART A: THE LEMMA FORM AND ITS READER (SYSTEM_SPEC 11 "One lemma machinery, used twice"; pieces: rules, and the long-term memory's statements; ledger rows 49, 44).

THIS TEXT IS THE INTERFACE, written before any body (docs/AGENT_BRIEF.md section 0). The module is one piece of the machinery that holds, certifies and states a lemma. It
names no game and no relation of any description: a game, its relations and its constants occur only in the lemma documents themselves, in their fixtures, and in the
descriptions under rules/. Its neighbours (the certificate: `rule_lemma_certify.py`; the proof obligation: `rule_lemma_lean.py`; the statement: `statement_of` here) read
a lemma only through the functions below.

WHAT A LEMMA IS. One form, for a lemma about the rules (a cheaper condition that gives the same rows as a relation of the description) and for a lemma about play (a
relation that is contained in another). Two forms of claim:
  SAME     "under the condition, relation L has the same rows as relation M"
  IMPLIES  "under the condition, every row of L is a row of M"
L and M are named by relation and number of places; each exists in the description composed with the lemma's delta, with that many places (its declared signature).
The CONDITION is a list of literals in the form the learned lemmas already use ({"relation", "args", "neg"}, rust/engine/rep/src/lemma.rs; an argument that starts with
`?` is a variable): a conjunction read on the state; no literal means always. A relation named in a literal that the description declares as a state fluent is read as the fluent.
The DELTA is a list of clauses in the description's own language (a clause or a signature declaration each), ADDITIVE ONLY: it defines new relations (and their helpers),
retracts nothing, redefines no relation the description defines, may read any relation of the description, and with the description is still stratified, safe and well sorted
by the reader's own test (discovery/gdl.py). It defines at least one of L and M. The ORIGIN is HAND (knowledge given, printed wherever the lemma is) or MINED (a later row).
The STANDING a document CLAIMS is CONJECTURE, CERTIFIED (with the certificate's key) or PROVED (with the name of a Lean theorem); the standing a lemma HAS is computed
(`standing_of`): a CERTIFIED claim holds only while the certificate beside it is fresh, a PROVED claim only while the theorem is in docs/LEAN_INDEX.md and not OPEN; otherwise it has
CONJECTURE.

WHERE LEMMAS LIVE (the folder): `rules/lemmas/<game>/`, beside the descriptions (rules/<description>.gdl, rules/variants/, rules/concepts/); not rules/learned, not data/derived.
  rules/lemmas/<game>/<name>.json              the lemma (one document)
  rules/lemmas/<game>/<name>.certificate.json  its certificate, written by `rule_lemma_certify.run`
  rules/lemmas/<game>/states.txt               the fixtures: one state a line, as the fluents of the state without `true` -- `(f a b) (g c) ...`; a line that starts with `;` is a note
  lean/ideas/Ideas/Mined/<...>.lean            the Lean obligation, written by `rule_lemma_lean.write` (an OPEN statement unless PROVED)

THE DOCUMENT (JSON, keys in this order; `delta` one clause a string):
  {"name": "<slug>", "game": "<game>", "variant": "<variant of the game's vocabulary>", "form": "SAME" | "IMPLIES", "says": "<the lemma in plain words>",
   "left": {"relation": "<L>", "places": n}, "right": {"relation": "<M>", "places": n}, "condition": [{"relation": "r", "args": ["?x", "c"], "neg": false}, ...],
   "delta": ["(signature ...)", "(<= (head ...) ...)", ...], "origin": "HAND", "standing": {"is": "CONJECTURE"} | {"is": "CERTIFIED", "key": "<16 hex>"} | {"is": "PROVED", "theorem": "<Lean name>"},
   "lean": {"module": "<Lean module of the generated description>", "namespace": "<its namespace>", "position": "<the Lean type of a state>"}}

WHAT THE MODULE TAKES AND RETURNS (who calls it: the certificate module and the Lean writer here; later the translator, which reads a SAME lemma as a rewrite of L by M, and the
layers above, which read the statements):
  `lemma_dir(game) -> Path`                 the folder above (game-free: a game is a name)
  `load(game, name) -> Lemma`               one document, read and checked; raises `LemmaError` listing every problem
  `load_all(game) -> [Lemma]`               every document of the folder, by name
  `problems(doc: dict, base_text: str) -> [str]`   the problems of a document read against a description text (none: well formed); the tests it makes are named in the body
  `description_text(lemma) -> str`          the description the lemma is about (the game's composition of its base, variant and concept deltas: game_store.spec_texts)
  `composed_text(lemma) -> str`             the description with the delta composed on (cs_gdl compose, which the variants use)
  `program(lemma) -> gdl.Program`           the composed text read by the reader; its evaluator is the oracle of the certificate
  `condition_rows(lemma, db) -> [binding]`  the bindings of the condition on an evaluated state (one binding with no variable when the condition has no literal)
  `standing_of(lemma, fresh: bool) -> (word, detail)`  the standing the lemma HAS, given whether its certificate is fresh (the certificate module tells)
  `statement_of(lemma, fresh: bool) -> dict`  PART D: the lemma as the JSON of a `Statement` (rust/engine/rep/src/statement.rs): id the name, pattern the condition, says Holds, holder
                                            ours, `also` the keys the statement stands on (its form, its two relations, the key of the text it was certified against), standing
                                            Proved(theorem) for PROVED and Conjecture(<the sample or the certificate>) for the others. No reader of it exists in the wake yet.
Failure modes: a document with a problem is refused whole (`LemmaError`), never half read; a standing that its evidence does not hold is `CONJECTURE`, printed with the cause.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import cs_engine as CE

from catspace.incubator.analyst.discovery import game_store as GS
from catspace.incubator.analyst.discovery import gdl as GDL
from catspace.io import paths

ROOT = paths.REPO_ROOT
LEMMAS = ROOT / "rules" / "lemmas"                      # the folder: beside the descriptions
INDEX = ROOT / "docs" / "LEAN_INDEX.md"                 # what is proved (AGENTS.md rule 11)
FORMS = ("SAME", "IMPLIES")
ORIGINS = ("HAND", "MINED")
STANDINGS = ("CONJECTURE", "CERTIFIED", "PROVED")
KEYS = ("name", "game", "variant", "form", "says", "left", "right", "condition", "delta", "origin", "standing", "lean")
RESERVED = {"role", "init", "true", "does", "legal", "next", "terminal", "goal", "record"}      # the language's own words: a delta defines none of them
DIFFER = "differ"                                       # the literal form's one literal that is not a row (rust/engine/base/src/logic/mod.rs)


class LemmaError(ValueError):
    pass


@dataclass
class Lemma:
    name: str
    game: str
    variant: str
    form: str
    says: str
    left: dict
    right: dict
    condition: list
    delta: list
    origin: str
    standing: dict
    lean: dict
    path: "Path | None" = None

    def doc(self) -> dict:
        """The document as written (the keys of KEYS, in order)."""
        return {k: getattr(self, k) for k in KEYS}

    def delta_text(self) -> str:
        return "\n".join(self.delta)


def lemma_dir(game: str) -> Path:
    return LEMMAS / game


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "-", name.lower()).strip("-")


def description_text(lemma: Lemma) -> str:
    """The description the lemma is about: the game's composition of its base, its variant and its concept deltas (game_store.spec_texts)."""
    texts = GS.spec_texts(lemma.game)
    if lemma.variant not in texts:
        raise LemmaError(f"{lemma.name}: the game {lemma.game} has no variant {lemma.variant} (it has {sorted(texts)})")
    return texts[lemma.variant]


def compose(base_text: str, delta: list) -> str:
    return CE.compose(base_text, "\n".join(delta))


def composed_text(lemma: Lemma) -> str:
    return compose(description_text(lemma), lemma.delta)


def program(lemma: Lemma) -> GDL.Program:
    return GDL.Program(composed_text(lemma), f"{lemma.game}/{lemma.name}")


# ------------------------------------------------------------------------------------------------------------------------------------------ the tests of a document

def _heads(sexprs: list) -> set:
    """The relation names the clauses of these forms define (a fluent's form by its flattened name)."""
    out = set()
    for x in sexprs:
        if isinstance(x, list) and x and x[0] in GDL.DECLARATIONS:
            continue
        head = x[1] if isinstance(x, list) and x and x[0] == "<=" else x
        out.add(GDL.to_atom(head)[0])
    return out


def _literal_problems(at: str, lits: list, prog: "GDL.Program | None") -> list:
    """The form of a condition: each literal {relation, args, neg}, one positive row at least when there is any, every variable of a negation or a `differ` bound by a positive row; with
    the description at hand, each relation exists and takes that many arguments, and each constant is a member of its slot's sort."""
    bad = []
    for i, l in enumerate(lits):
        if not isinstance(l, dict) or not isinstance(l.get("relation"), str) or not isinstance(l.get("args"), list) or not all(isinstance(a, str) for a in l["args"]) \
                or not isinstance(l.get("neg", False), bool) or set(l) - {"relation", "args", "neg"}:
            return [f"{at}: literal {i} is not {{relation, args, neg}}"]
    rows = [l for l in lits if not l.get("neg") and l["relation"] != DIFFER]
    if lits and not rows:
        bad.append(f"{at}: a condition needs a positive literal")
    bound = {a for l in rows for a in l["args"] if a.startswith("?")}
    for l in lits:
        if l.get("neg") or l["relation"] == DIFFER:
            if l["relation"] == DIFFER and (len(l["args"]) != 2 or l.get("neg")):
                bad.append(f"{at}: differ takes two arguments and is positive")
            for a in l["args"]:
                if a.startswith("?") and a not in bound:
                    bad.append(f"{at}: the literal {l['relation']} uses {a}, which no positive literal binds")
    if prog is not None:
        env: dict = {}
        for l in lits:
            if l["relation"] == DIFFER:
                continue
            rel = fluent_or_relation(prog, l["relation"])
            if rel not in prog.nodes:
                bad.append(f"{at}: the relation {l['relation']} is not in the description")
                continue
            sig = prog.signature(rel)
            if sig is None or len(sig) != len(l["args"]):
                bad.append(f"{at}: {l['relation']} takes {0 if sig is None else len(sig)} places, the literal gives {len(l['args'])}")
                continue
            for a, s in zip(l["args"], sig):
                prog.member(a, s, env, f"{at}: {l['relation']}", bad)
    return bad


def fluent_or_relation(prog: GDL.Program, rel: str) -> str:
    """A relation named in a literal: the description's relation, or the state fluent of that name (`true.<name>`)."""
    if rel in prog.nodes:
        return rel
    return "true." + rel if ("true." + rel) in prog.nodes else rel


def problems(doc: dict, base_text: str) -> list:
    """The problems of a document read against a description text (none: well formed). The tests: the keys and their forms; the delta is additive (no retract, no relation the
    description defines, no word of the language); with the description it is stratified, safe and well sorted by the reader's own test; each of L and M exists with that many
    places and the delta defines one of them; the condition is a well formed body over the description."""
    at = str(doc.get("name", "?"))
    bad = []
    if list(doc) != list(KEYS):
        return [f"{at}: the keys are {list(KEYS)} in that order, the document has {list(doc)}"]
    if not isinstance(doc["name"], str) or slug(doc["name"]) != doc["name"] or not doc["name"]:
        bad.append(f"{at}: a name is a slug")
    if doc["form"] not in FORMS:
        bad.append(f"{at}: the form is one of {FORMS}")
    if doc["origin"] not in ORIGINS:
        bad.append(f"{at}: the origin is one of {ORIGINS}")
    st = doc["standing"]
    if not isinstance(st, dict) or st.get("is") not in STANDINGS:
        bad.append(f"{at}: the standing is one of {STANDINGS}")
    elif (st["is"] == "CERTIFIED" and not isinstance(st.get("key"), str)) or (st["is"] == "PROVED" and not isinstance(st.get("theorem"), str)):
        bad.append(f"{at}: a {st['kind']} standing names its {'key' if st['kind'] == 'CERTIFIED' else 'theorem'}")
    for which in ("left", "right"):
        r = doc[which]
        if not isinstance(r, dict) or not isinstance(r.get("relation"), str) or not isinstance(r.get("places"), int):
            bad.append(f"{at}: {which} is {{relation, places}}")
    if not isinstance(doc["delta"], list) or not doc["delta"] or not all(isinstance(c, str) for c in doc["delta"]):
        bad.append(f"{at}: the delta is a list of clauses")
    if not isinstance(doc["condition"], list):
        bad.append(f"{at}: the condition is a list of literals")
    for k in ("game", "variant", "says"):
        if not isinstance(doc[k], str) or not doc[k]:
            bad.append(f"{at}: {k} is text")
    if not isinstance(doc["lean"], dict) or set(doc["lean"]) != {"module", "namespace", "position"}:
        bad.append(f"{at}: lean is {{module, namespace, position}}")
    if bad:
        return bad
    try:
        base = GDL.parse(base_text)
        forms = [GDL.parse(c) for c in doc["delta"]]
    except GDL.GDLError as x:
        return [f"{at}: the text does not read: {x}"]
    if any(len(f) != 1 for f in forms):
        return [f"{at}: each entry of the delta is one clause"]
    forms = [f[0] for f in forms]
    for f in forms:
        if isinstance(f, list) and f and f[0] == "retract":
            return [f"{at}: the delta is additive only: it retracts {GDL._unparse(f)[:80]}"]
    base_heads = _heads([x for x in base if not (isinstance(x, list) and x and x[0] == "retract")])
    try:
        mine = _heads(forms)
    except (GDL.GDLError, IndexError, TypeError) as x:
        return [f"{at}: a clause does not read: {x}"]
    clash = sorted(mine & base_heads)
    if clash:
        bad.append(f"{at}: the delta defines relations the description defines: {clash}")
    kw = sorted(h for h in mine if h.split(".")[0] in RESERVED)
    if kw:
        bad.append(f"{at}: the delta defines words of the language: {kw}")
    if bad:
        return bad
    try:
        prog = GDL.Program(compose(base_text, doc["delta"]), at)
    except (GDL.GDLError, ValueError) as x:
        return [f"{at}: with the description the delta does not read as a program: {x}"]
    for p in prog.typecheck():
        bad.append(f"{at}: the sort test: {p}")
    sides = []
    for which in ("left", "right"):
        r = doc[which]
        sig = prog.signature(r["relation"])
        if r["relation"] not in prog.nodes or sig is None:
            bad.append(f"{at}: {which} {r['relation']} is not a relation of the description with the delta")
        elif len(sig) != r["places"]:
            bad.append(f"{at}: {which} {r['relation']} has {len(sig)} places, the document says {r['places']}")
        sides.append(r["relation"])
    if doc["left"]["places"] != doc["right"]["places"]:
        bad.append(f"{at}: the two ends have different places")
    if not (set(sides) & mine):
        bad.append(f"{at}: the delta defines neither end")
    bad += _literal_problems(at, doc["condition"], prog)
    return bad


# ------------------------------------------------------------------------------------------------------------------------------------------ loading

def parse(doc: dict, path: "Path | None" = None, base_text: "str | None" = None) -> Lemma:
    """A document as a lemma, once `problems` finds none (against `base_text`, or the game's own composition)."""
    if base_text is None:
        texts = GS.spec_texts(doc.get("game", ""))
        if doc.get("variant") not in texts:
            raise LemmaError(f"{doc.get('name')}: the game {doc.get('game')} has no variant {doc.get('variant')}")
        base_text = texts[doc["variant"]]
    bad = problems(doc, base_text)
    if bad:
        raise LemmaError("\n".join(bad))
    return Lemma(path=path, **{k: doc[k] for k in KEYS})


def load(game: str, name: str) -> Lemma:
    p = lemma_dir(game) / f"{name}.json"
    if not p.is_file():
        raise LemmaError(f"{p}: no lemma {name} of the game {game}")
    lem = parse(json.loads(p.read_text()), p)
    if lem.game != game or p.stem != lem.name:
        raise LemmaError(f"{p.name}: the document is named {lem.name} and says the game {lem.game}")
    return lem


def load_all(game: str) -> list:
    d = lemma_dir(game)
    out = []
    for p in sorted(d.glob("*.json")) if d.is_dir() else []:
        if p.name.endswith(".certificate.json"):
            continue
        out.append(load(game, p.stem))
    return out


def write(lemma: Lemma) -> Path:
    """The document, written as `load` reads it (one clause a line)."""
    p = lemma_dir(lemma.game) / f"{lemma.name}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    d = lemma.doc()
    lines = ["{"]
    for i, k in enumerate(KEYS):
        v = d[k]
        comma = "," if i < len(KEYS) - 1 else ""
        if k in ("delta", "condition"):
            items = [json.dumps(x, ensure_ascii=True) for x in v]
            lines.append(f' "{k}": [' + (("\n  " + ",\n  ".join(items) + "\n ]") if items else "]") + comma)
        else:
            lines.append(f' "{k}": ' + json.dumps(v, ensure_ascii=True) + comma)
    lines.append("}")
    p.write_text("\n".join(lines) + "\n")
    return p


# ------------------------------------------------------------------------------------------------------------------------------------------ the condition on a state

def condition_rows(lemma: Lemma, db: GDL.DB, prog: "GDL.Program | None" = None) -> list:
    """The bindings of the condition on an evaluated state: a list of {variable: value}; one binding with no variable when the condition has no literal, none when it does not hold."""
    prog = prog or program(lemma)
    pos = [l for l in lemma.condition if not l.get("neg") and l["relation"] != DIFFER]
    rest = [l for l in lemma.condition if l.get("neg") or l["relation"] == DIFFER]
    rows = [{}]
    for l in pos:
        facts = db.facts(fluent_or_relation(prog, l["relation"]))
        nxt = []
        for b in rows:
            for t in facts:
                nb = dict(b)
                ok = True
                for a, v in zip(l["args"], t):
                    if a.startswith("?"):
                        if nb.setdefault(a, v) != v:
                            ok = False
                            break
                    elif a != v:
                        ok = False
                        break
                if ok:
                    nxt.append(nb)
        rows = nxt
    out = []
    for b in rows:
        good = True
        for l in rest:
            if l["relation"] == DIFFER:
                x, y = (b.get(a, a) for a in l["args"])
                good = good and x != y
            else:
                row = tuple(b.get(a, a) for a in l["args"])
                good = good and (row not in db.facts(fluent_or_relation(prog, l["relation"])))
        if good:
            out.append(b)
    return out


def condition_relations(lemma: Lemma, prog: GDL.Program) -> tuple:
    return tuple(sorted({fluent_or_relation(prog, l["relation"]) for l in lemma.condition if l["relation"] != DIFFER}))


# ------------------------------------------------------------------------------------------------------------------------------------------ standing and statement

def index_text() -> str:
    return INDEX.read_text() if INDEX.is_file() else ""


def proved_in_index(theorem: str) -> bool:
    """The theorem is in the index and not OPEN (a proof that still holds `sorry` is listed as OPEN)."""
    text = index_text()
    return f"- **theorem `{theorem}`**" in text or f"- **lemma `{theorem}`**" in text


def standing_of(lemma: Lemma, fresh: bool, key: "str | None" = None) -> tuple:
    """The standing the lemma HAS -> (word, detail). PROVED only while its theorem is in the index and not OPEN; CERTIFIED only while the certificate beside it is fresh and holds the
    key the document names (`fresh`, `key`: what the certificate module says); otherwise CONJECTURE, with the cause."""
    st = lemma.standing
    if st["is"] == "PROVED":
        if proved_in_index(st["theorem"]):
            return "PROVED", st["theorem"]
        return "CONJECTURE", f"the theorem {st['theorem']} is not in the index as proved"
    if st["is"] == "CERTIFIED":
        if not fresh:
            return "CONJECTURE", "the certificate is not fresh"
        if key is not None and key != st["key"]:
            return "CONJECTURE", f"the document names the key {st['key']}, the certificate holds {key}"
        return "CERTIFIED", st["key"]
    return "CONJECTURE", "no certificate named"


def statement_of(lemma: Lemma, fresh: bool = False, key: "str | None" = None) -> dict:
    """PART D: the lemma as the JSON of rep's `Statement`: id the name, pattern the condition, says Holds, holder ours, `also` the keys it stands on, standing Proved(theorem) or
    Conjecture(the sample). Built with what rep has: a statement of the form `L is M` would be a saying of its own, which rep does not have (reported, not added)."""
    word, detail = standing_of(lemma, fresh, key)
    standing = {"kind": "proved", "theorem": detail} if word == "PROVED" else \
        {"kind": "conjecture", "sample": (f"certificate {detail}" if word == "CERTIFIED" else f"no certificate: {detail}") + f"; origin {lemma.origin}"}
    pattern = [{"relation": l["relation"], "args": list(l["args"]), "neg": bool(l.get("neg"))} for l in lemma.condition]
    also = [f"lemma:{lemma.form}:{lemma.left['relation']}/{lemma.left['places']}:{lemma.right['relation']}/{lemma.right['places']}", f"game:{lemma.game}:{lemma.variant}"]
    if word == "CERTIFIED":
        also.append(f"certificate:{detail}")
    return {"id": lemma.name, "pattern": pattern, "says": {"kind": "holds"}, "standing": standing, "holder": "ours", "also": also}
