"""PLAN 79.59 PART C (SYSTEM_SPEC 11, block "The lemma miner of layer 2"; LAYER 2, piece: rules; ledger rows 59, 49): THE THIRD FORM OF THE ONE LEMMA DOCUMENT, FILTER, AND THE STORE OF THE MINED LEMMAS.

THIS TEXT IS THE INTERFACE, written before any body. It names no game and no relation of any description. It reads a document with the reader of `rule_lemma.py` (the keys of the form, the delta, the composition, the tests of the delta) and adds what the form FILTER needs. It is a module of its own, not
an edit of `rule_lemma.py`, because the bytes of `rule_lemma.py` are part of the key of every certificate of the hand-given lemmas (rule_lemma_certify.SOURCES): an edit there would make every certificate stale. The document is the same document, the same folder layout and the same loader contract.

THE FORM. FILTER   "for every key x of the test L (left) at a state: if x is not a row of the hits relation then x is a row of L exactly when it is a row of the answer relation (right)". The document is the keys of `rule_lemma.KEYS` and two more, in this order:
  `filter`    {"hits": <relation>, "tested": <relation, optional>, "classes": n, "reads": [{"relation", "literal", "clause", "trail"}, ...]}   the relation that holds the keys whose differing rows meet a read of the null test's proof; the copy of the test made for those keys only (the mask); the places of the class of a key;
              each read, with the hit relation made for it, the literal and the clause it comes from and the chain of calls it was unfolded through.
  `evidence`  {} or what the judge printed (the field it was tested on, the gate, the narrowness, the timing, the use: paying or not used for speed, the segments it failed on).
  The delta is additive, as ever: it defines the answer relation, the hits relation, the tested copy and what they read. The condition is void. The origin is MINED.
THE STANDING (MINED_STANDINGS): PROPOSED (made by a proposer, not judged); TESTED (the gate held at every state of the segments it was tested on, which `evidence` names); REJECTED (the gate failed: the reason and the segments it failed on are kept, conditional on the field it was tested on, never read as a verdict beyond it);
  PROVED (TESTED, and the Lean theorem of its type compiles with the usual axioms: `lean.uses` names it; the theorem is general, the instance party conditions are the construction of lemma_miner.py, held by the gate).
WHERE THEY LIVE: rules/lemmas/<game>/mined/<name>.json (`mined_dir`), never beside the hand-given ones, never in the learned store.

WHAT THE MODULE TAKES AND RETURNS (who calls it: lemma_miner.py, scripts/lemma_miner_run.py):
  `filter_problems(doc, base_text) -> [str]`     the problems of a document of the form FILTER read against a description text (none: well formed)
  `parse(doc, path=None, base_text=None) -> FilterLemma`   the document as a lemma, once it has no problem (`rule_lemma.LemmaError` otherwise)
  `mined_dir(game) -> Path`, `load_mined(game, base_text=None) -> [FilterLemma]`, `write_mined(lemma) -> Path`
  `filter_standing(lemma) -> (word, detail)`     the standing the document HAS: PROVED only while its theorem compiles and the evidence says the gate held, otherwise TESTED; any other word as claimed
  `lean_theorem_compiles(source, theorem) -> (bool, what was observed)`
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from catspace.incubator.analyst.discovery import gdl as GDL
from catspace.incubator.analyst.discovery import rule_lemma as RL

ROOT = RL.ROOT
KEYS_FILTER = RL.KEYS + ("filter", "evidence")
MINED_STANDINGS = ("PROPOSED", "TESTED", "REJECTED", "PROVED")
THEOREM_FILE = ROOT / "lean" / "ideas" / "Ideas" / "NullFilter.lean"


@dataclass
class FilterLemma(RL.Lemma):
    """A document of the form FILTER: the fields of `rule_lemma.Lemma` and the two keys that follow them."""
    filter: "dict | None" = None
    evidence: "dict | None" = None

    def doc(self) -> dict:
        return {k: getattr(self, k) for k in KEYS_FILTER}


def mined_dir(game: str) -> Path:
    """Where the MINED lemmas of a game are written, beside the hand-given ones (a folder of their own, which `rule_lemma.load_all` does not read)."""
    return RL.lemma_dir(game) / "mined"


def filter_problems(doc: dict, base_text: str) -> list:
    """The problems of a document of the form FILTER read against a description text: the keys (KEYS_FILTER, in order); the origin MINED; a standing of MINED_STANDINGS; the delta additive, defining the answer relation (right), the hits relation and the tested copy; with the description the delta is
    stratified, safe and well sorted; the test (left), the answer (right), the hits and the tested copy have the places the document says; the condition is void."""
    at = str(doc.get("name", "?"))
    if list(doc) != list(KEYS_FILTER):
        return [f"{at}: the keys of a FILTER are {list(KEYS_FILTER)} in that order, the document has {list(doc)}"]
    bad = []
    if doc["form"] != "FILTER":
        bad.append(f"{at}: the form is FILTER")
    if doc["origin"] != "MINED":
        bad.append(f"{at}: a FILTER is mined: the origin is MINED")
    st = doc["standing"]
    if not isinstance(st, dict) or st.get("is") not in MINED_STANDINGS:
        bad.append(f"{at}: the standing is one of {MINED_STANDINGS}")
    if not isinstance(doc["name"], str) or RL.slug(doc["name"]) != doc["name"] or not doc["name"]:
        bad.append(f"{at}: a name is a slug")
    for which in ("left", "right"):
        r = doc[which]
        if not isinstance(r, dict) or not isinstance(r.get("relation"), str) or not isinstance(r.get("places"), int):
            bad.append(f"{at}: {which} is {{relation, places}}")
    if not isinstance(doc["delta"], list) or not doc["delta"] or not all(isinstance(c, str) for c in doc["delta"]):
        bad.append(f"{at}: the delta is a list of clauses")
    if doc["condition"] != []:
        bad.append(f"{at}: a FILTER has no condition")
    f = doc["filter"]
    if not isinstance(f, dict) or not isinstance(f.get("hits"), str) or not isinstance(f.get("classes"), int) or not isinstance(f.get("reads"), list) or not isinstance(f.get("tested", ""), str):
        bad.append(f"{at}: filter is {{hits, tested (optional), classes, reads}}")
    if not isinstance(doc["evidence"], dict):
        bad.append(f"{at}: evidence is an object")
    for k in ("game", "variant", "says"):
        if not isinstance(doc[k], str) or not doc[k]:
            bad.append(f"{at}: {k} is text")
    ln = doc["lean"]
    if not isinstance(ln, dict) or not {"module", "namespace", "position"} <= set(ln):
        bad.append(f"{at}: lean is {{module, namespace, position}}")
    if bad:
        return bad
    try:
        base = GDL.parse(base_text)
        forms = [GDL.parse(c) for c in doc["delta"]]
    except GDL.GDLError as x:
        return [f"{at}: the text does not read: {x}"]
    if any(len(x) != 1 for x in forms):
        return [f"{at}: each entry of the delta is one clause"]
    forms = [x[0] for x in forms]
    if any(isinstance(x, list) and x and x[0] == "retract" for x in forms):
        return [f"{at}: the delta is additive only"]
    base_heads = RL._heads([x for x in base if not (isinstance(x, list) and x and x[0] == "retract")])
    try:
        mine = RL._heads(forms)
    except (GDL.GDLError, IndexError, TypeError) as x:
        return [f"{at}: a clause does not read: {x}"]
    clash = sorted(mine & base_heads)
    if clash:
        bad.append(f"{at}: the delta defines relations the description defines: {clash}")
    kw = sorted(h for h in mine if h.split(".")[0] in RL.RESERVED)
    if kw:
        bad.append(f"{at}: the delta defines words of the language: {kw}")
    for rel in (doc["right"]["relation"], f["hits"]) + ((f["tested"],) if f.get("tested") else ()):
        if rel not in mine:
            bad.append(f"{at}: the delta does not define {rel}")
    if bad:
        return bad
    try:
        prog = GDL.Program(RL.compose(base_text, doc["delta"]), at)
    except (GDL.GDLError, ValueError) as x:
        return [f"{at}: with the description the delta does not read as a program: {x}"]
    for q in prog.typecheck():
        bad.append(f"{at}: the sort test: {q}")
    for rel, places in ((doc["left"]["relation"], doc["left"]["places"]), (doc["right"]["relation"], doc["right"]["places"]), (f["hits"], doc["left"]["places"])) + (((f["tested"], doc["left"]["places"]),) if f.get("tested") else ()):
        sig = prog.signature(rel)
        if rel not in prog.nodes or sig is None:
            bad.append(f"{at}: {rel} is not a relation of the description with the delta")
        elif len(sig) != places:
            bad.append(f"{at}: {rel} has {len(sig)} places, the document says {places}")
    return bad


def parse(doc: dict, path: "Path | None" = None, base_text: "str | None" = None) -> FilterLemma:
    """A document as a lemma, once `filter_problems` finds none (against `base_text`, or the game's own composition)."""
    if base_text is None:
        from catspace.incubator.analyst.discovery import game_store as GS
        texts = GS.spec_texts(doc.get("game", ""))
        if doc.get("variant") not in texts:
            raise RL.LemmaError(f"{doc.get('name')}: the game {doc.get('game')} has no variant {doc.get('variant')}")
        base_text = texts[doc["variant"]]
    bad = filter_problems(doc, base_text)
    if bad:
        raise RL.LemmaError("\n".join(bad))
    return FilterLemma(path=path, **{k: doc[k] for k in KEYS_FILTER})


def load_mined(game: str, base_text: "str | None" = None) -> list:
    """Every MINED document of a game (rules/lemmas/<game>/mined/), read and checked."""
    d = mined_dir(game)
    return [parse(json.loads(p.read_text()), p, base_text) for p in (sorted(d.glob("*.json")) if d.is_dir() else [])]


def write_mined(lemma: FilterLemma) -> Path:
    """The document, written to the folder of the mined lemmas of its game (one clause a line); never to the hand-given folder."""
    p = mined_dir(lemma.game) / f"{lemma.name}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    d = lemma.doc()
    lines = ["{"]
    for i, k in enumerate(KEYS_FILTER):
        v = d[k]
        comma = "," if i < len(KEYS_FILTER) - 1 else ""
        if k in ("delta", "condition"):
            items = [json.dumps(x, ensure_ascii=True) for x in v]
            lines.append(f' "{k}": [' + (("\n  " + ",\n  ".join(items) + "\n ]") if items else "]") + comma)
        else:
            lines.append(f' "{k}": ' + json.dumps(v, ensure_ascii=True, sort_keys=(k == "evidence")) + comma)
    lines.append("}")
    p.write_text("\n".join(lines) + "\n")
    return p


def lean_theorem_compiles(source: Path, theorem: str) -> "tuple[bool, str]":
    """Whether a Lean source compiles with no `sorry` and the theorem depends on the usual axioms only (the `#print axioms` of the source is what is read). -> (ok, what was observed)."""
    import subprocess
    if not source.is_file() or "sorry" in re.sub(r"/-.*?-/", "", source.read_text(), flags=re.S):
        return False, f"{source.name} is missing or holds sorry"
    r = subprocess.run(["lake", "env", "lean", str(source.relative_to(ROOT / "lean" / "ideas"))], cwd=ROOT / "lean" / "ideas", capture_output=True, text=True)
    out = (r.stdout + r.stderr).strip()
    line = next((x for x in out.splitlines() if f"'{theorem}' depends on axioms" in x), "")
    ok = r.returncode == 0 and bool(line) and all(a.strip() in ("propext", "Quot." + "so" + "und", "Classical.choice") for a in line.split("[", 1)[1].rstrip("]").split(",") if a.strip())
    return ok, line or out[-300:]


def filter_standing(lemma: FilterLemma, theorem_file: "Path | None" = None) -> tuple:
    """The standing a FILTER HAS -> (word, detail). PROVED only while the Lean theorem the document names (`lean.uses`) compiles in its source with the usual axioms and the evidence says the gate held; lowered to TESTED when it claims PROVED and either fails; any other word as claimed."""
    st = lemma.standing
    word = st["is"]
    if word != "PROVED":
        return word, st.get("detail", "")
    uses = lemma.lean.get("uses", [])
    if not uses:
        return "TESTED", "the document names no theorem"
    ok, observed = lean_theorem_compiles(theorem_file or THEOREM_FILE, uses[0])
    if not ok:
        return "TESTED", f"the theorem {uses[0]} does not compile: {observed}"
    if (lemma.evidence or {}).get("gate") != "held":
        return "TESTED", "the evidence does not say the gate held"
    return "PROVED", f"{uses[0]} ({observed})"
