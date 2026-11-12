"""PLAN 78.35 (3) THE REGISTRY OF A STORE'S WORDS, AND GENERAL vs GAME KNOWLEDGE TOLD APART BY A PROGRAM. Generic: no object of any game is named here.

THE REGISTRY  `game_store.learned_dir(game) / "words.json"` (so a sandbox store has its own; format "words-1"): {"format": "words-1", "words": [entry ..]} in definition order, an entry
  {name, term (text), signature [sort ..], rules (the compiled text, word_terms.compile), size (word_terms.size), origin ("hand" | "generated" | "mined"), layer ("general" | "game"),
   expansion (the term with every earlier word replaced by its own term: the expansion in base words), note, gloss (the meaning in plain words: a hand word's `note`, else word_terms.gloss of
   its term with the vocabulary record's `"glosses": {relation: template}`; an old entry with no `gloss` reads as its `note`: gloss_of(entry))}.
  load(game) / save(game, reg)
  define_new(game, term_text, *, origin, name=None, note="", vocabulary=None) -> (entry, created)       define(...) -> entry
        compiles the term on the vocabulary's composed description (earlier words included: a word may use them) and records it. NAMES: origin "hand" REQUIRES an explicit `name` (a descriptive
        name given by its author) and a non-empty `note` (its meaning in plain words, kept as the gloss); for the other origins `name=None` is word_names.composed_name (built from the relations
        the term is made of). An explicit name is word_names.check_name'd: legal, and neither a word nor a relation of the description. A term equal to an existing word's -- after expansion and a canonical renaming of the variables -- is NOT defined again:
        the existing entry is returned (created False): one definition, one name. A term that is a single literal of one relation (distinct variables) is refused: it is that relation already.
  gloss_of(entry) -> the entry's gloss (its note when it has none)
  delta(game) -> the rules text of all words in definition order
  WIRING (game_store): a vocabulary record with `"words": true` composes delta(game) after the concept deltas and before the structure delta, and appends the registry's names to the
        vocabulary's `derived`. Absent (the default) or false: nothing changes, byte for byte. The vocabulary record's `"word_delta": [names]` lists words appended to the vocabulary's `delta`
        (the relations a move's effect is read in); words are left out of it unless listed.

GENERAL OR GAME, BY A PROGRAM (the rule, precisely)
  A relation's definition is READ FROM THE RULES TEXT, not declared. Its CLAUSES are the clauses of the composed description that are not clauses of the BASE description (the vocabulary's `spec`
  text: the game's own rules, `base_text`). A relation with no clause of its own is the game's RULES or its state read by variables: GENERAL to read (a word that only reads them is general).
  A relation is GAME when
    (a) one of its own clauses (head and body, a ground fact too) holds a CONSTANT: any leaf symbol that is not a variable, other than a member of the DISTANCE sort (`distance_sort`, default
        "distance": the step counts d0, d1 ..). A constant of the side (role) sort is a game constant like any other (a definition naming one side names the game's own), unless
        `roles_general=True`; the small integers a term uses as a step count are unrolled into rules and never appear as constants; or
    (b) one of its own clauses READS a relation that is GAME (negatively too), followed down transitively (recursion: a relation in a cycle is game when any member's own clauses name a
        constant or read a game relation outside the cycle).
  Otherwise GENERAL. The relations of the vocabulary's generated structure (`generated`: board_structure.relation_names) are GENERAL by construction: a game-free program writes their
  facts from the description, so their own ground facts (which do hold coordinate constants) are not counted. A relation of the base description that a delta ADDS clauses to is judged by the
  added clauses alone.
  layer_of(relation, spec_text, *, role_sorts=None, distance_sort="distance", base_text=None, generated=(), roles_general=False) -> "general" | "game"   (role_sorts: the sort(s) of the sides;
        default the slot sorts of the description's `role` relation; only used to say WHICH constant it is, and to let `roles_general` free them)
  analyse(spec_text, ...) -> {relation: {"layer", "why"}}   every relation with a clause, `why` the constant that decided it, or the chain of game relations read down to one ("" for general)
  layers(game, vocabulary_name) -> {relation: layer} for every relation of the vocabulary's `derived` and every registry word
  rule_layer(lemma, layers) -> "general" when every relation the lemma's body and head name is general (a relation not in `layers` is the description's own: general), else "game"
  origin_of(lemma) -> "hand" when meta.origin == "hand", else "machine"

  .venv/bin/python -m catspace.incubator.analyst.discovery.word_registry <game> --vocabulary NAME [--store-root DIR] [--expand]
  prints `OBS word <name> (<origin>, <layer>, size N): <gloss>` and, with --expand, `    = <term>` on the next line
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cs_engine as CE

from catspace.incubator.analyst.discovery import board_structure as BS
from catspace.incubator.analyst.discovery import gdl as GDL
from catspace.incubator.analyst.discovery import word_names as WN
from catspace.incubator.analyst.discovery import word_terms as WT

FORMAT = "words-1"
ORIGINS = ("hand", "generated", "mined")


def _gs():
    from catspace.incubator.analyst.discovery import game_store as GS          # (game_store reads this module's delta: imported late to break the cycle)
    return GS


def path_of(game: str) -> Path:
    return _gs().learned_dir(game) / "words.json"


def load(game: str) -> dict:
    p = path_of(game)
    if not p.is_file():
        return {"format": FORMAT, "words": []}
    d = json.loads(p.read_text())
    if d.get("format") != FORMAT:
        raise ValueError(f"{p}: not a {FORMAT} document")
    return d


def save(game: str, reg: dict) -> Path:
    p = path_of(game)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"format": FORMAT, "words": reg["words"]}, indent=1) + "\n")
    tmp.replace(p)
    return p


def gloss_of(entry: dict) -> str:
    return entry.get("gloss") or entry.get("note", "")


def names_of(game: str) -> list:
    return [e["name"] for e in load(game)["words"]]


def delta(game: str) -> str:
    """The rules text of every word, in definition order ("" with none)."""
    words = load(game)["words"]
    if not words:
        return ""
    return ";;; the words of the store (word_registry.py, plan 78.35): terms compiled to rules; ADDITIVE ONLY.\n" + "".join(e["rules"] for e in words)


# ------------------------------------------------------------------------------------------------------------------------------------------ general or game

_CACHE: dict = {}


def _leaves(t, acc: list) -> list:
    if isinstance(t, str):
        if not t.startswith("?"):
            acc.append(t)
    else:
        for y in t:
            _leaves(y, acc)
    return acc


def role_sorts_of(text: str) -> tuple:
    return tuple(GDL.split_declarations(GDL.parse(text))[1].get("role", ()))


def analyse(spec_text: str, *, role_sorts: "tuple | None" = None, distance_sort: str = "distance", base_text: "str | None" = None, generated=(), roles_general: bool = False) -> dict:
    key = (hash(spec_text), len(spec_text), hash(base_text), distance_sort, tuple(role_sorts or ()), tuple(generated), roles_general)
    if key in _CACHE:
        return _CACHE[key]
    forms = GDL.parse(spec_text)
    sorts, sigs, _rest = GDL.split_declarations(forms)
    clauses = GDL.clauses(forms)
    base = {c.text for c in GDL.clauses(GDL.parse(base_text))} if base_text else set()
    roles = tuple(role_sorts) if role_sorts is not None else tuple(sigs.get("role", ()))
    allowed = {m for m in sorts.get(distance_sort, []) if isinstance(m, str)}
    role_members = {m for s in roles for m in sorts.get(s, []) if isinstance(m, str)}
    if roles_general:
        allowed |= role_members
    gen = set(generated)
    why: dict = {}
    reads: dict = {}
    for c in clauses:
        h = c.head[0]
        if h in gen or c.text in base:
            continue
        leaves = _leaves(c.head[1], [])
        for l in c.body:
            if l[0] in ("pos", "neg"):
                leaves += _leaves(l[2], [])
                reads.setdefault(h, set()).add(l[1])
            else:
                leaves += _leaves(l[1:], [])
        reads.setdefault(h, set())
        const = next((x for x in leaves if x not in allowed), None)
        if const is not None and h not in why:
            kind = "side constant " if const in role_members else "constant "
            why[h] = kind + const
    game = set(why)
    changed = True
    while changed:
        changed = False
        for h, rs in reads.items():
            if h in game or h in gen:
                continue
            bad = sorted(r for r in rs if r in game and r not in gen)
            if bad:
                game.add(h)
                why[h] = "reads " + bad[0]
                changed = True
    def chain(h: str, seen: tuple = ()) -> str:
        w = why.get(h, "")
        if w.startswith("reads ") and w[6:] not in seen:
            return f"{w} -> {chain(w[6:], seen + (h,))}"
        return w
    out = {h: {"layer": "game" if h in game else "general", "why": chain(h)} for h in reads}
    for h in gen:
        out[h] = {"layer": "general", "why": ""}
    _CACHE[key] = out
    return out


def layer_of(relation: str, spec_text: str, *, role_sorts: "tuple | None" = None, distance_sort: str = "distance", base_text: "str | None" = None, generated=(), roles_general: bool = False) -> str:
    """"general" or "game" for `relation` (see the module's rule); a relation with no clause of its own (the game's rules and state, read by variables) is general."""
    return analyse(spec_text, role_sorts=role_sorts, distance_sort=distance_sort, base_text=base_text, generated=generated, roles_general=roles_general).get(relation, {"layer": "general"})["layer"]


def _context(game: str, vocabulary: "str | None") -> tuple:
    """(composed text of the first variant, base text of it, vocabulary record) -- the words of the store composed in whether or not the record says `words`."""
    GS = _gs()
    vf = GS.vocabulary_file(game, vocabulary)
    texts = GS.spec_texts(game, vocabulary)
    variant = next(iter(vf["variants"]))
    text = texts[variant]
    if not vf.get("words"):
        d = delta(game)
        if d:
            text = CE.compose(text, d)
    base = (GS.RULES / vf["spec"]).read_text()
    if vf["variants"][variant] is not None:
        base = CE.compose(base, (GS.RULES / vf["variants"][variant]).read_text())
    return text, base, vf


def layers(game: str, vocabulary: "str | None" = None, *, why: bool = False) -> dict:
    """{relation: "general" | "game"} for the vocabulary's `derived` relations and the registry's words (`why=True`: {relation: (layer, why)})."""
    text, base, vf = _context(game, vocabulary)
    a = analyse(text, distance_sort=vf.get("distance_sort", "distance"), base_text=base, generated=BS.relation_names(vf))
    rels = list(vf.get("derived", [])) + BS.relation_names(vf) + names_of(game)
    out = {}
    for r in dict.fromkeys(rels):
        e = a.get(r, {"layer": "general", "why": ""})
        out[r] = (e["layer"], e["why"]) if why else e["layer"]
    return out


def rule_layer(lemma: dict, layers: dict) -> str:
    rels = [l["relation"] for l in lemma.get("body", [])]
    resp = lemma.get("response") or {}
    for k in ("removed", "added"):
        rels += [l["relation"] for l in resp.get(k, [])]
    return "general" if all(layers.get(r, "general") == "general" for r in rels) else "game"


def origin_of(lemma: dict) -> str:
    return "hand" if (lemma.get("meta") or {}).get("origin") == "hand" else "machine"


# ------------------------------------------------------------------------------------------------------------------------------------------ define

def _relations_of(text: str) -> set:
    forms = GDL.parse(text)
    out = set(GDL.split_declarations(forms)[1])
    for c in GDL.clauses(forms):
        out.add(c.head[0])
    return out


def define_new(game: str, term_text: str, *, origin: str, name: "str | None" = None, note: str = "", vocabulary: "str | None" = None) -> tuple:
    if origin not in ORIGINS:
        raise ValueError(f"origin is one of {ORIGINS}, not {origin!r}")
    if origin == "hand":
        if not name:
            raise ValueError("a hand-written word is named by its author with a descriptive name: pass name=")
        if not (note or "").strip():
            raise ValueError("a hand-written word needs a note: its meaning in plain words (it is kept as the gloss)")
    term = WT.parse(term_text)
    reg = load(game)
    text, base, vf = _context(game, vocabulary)
    desc = WT.read_description(text)
    registry = {e["name"]: e["term"] for e in reg["words"]}
    exp = WT.expand(term, registry)
    key = WT.show(WT.canonical(exp))
    for e in reg["words"]:
        if WT.show(WT.canonical(WT.parse(e["expansion"]))) == key:
            return e, False
    if term[0] == "lit" and len(set(term[2])) == len(term[2]) and all(WT.is_var(a) for a in term[2]):
        raise ValueError(f"{WT.show(term)} is the relation {term[1]} itself: nothing to define")
    used = set(registry) | _relations_of(text)
    if name is None:
        name = WN.composed_name(term, used)
    else:
        WN.check_name(name, used)
    rules = WT.compile(name, term, desc.signatures, fluents=desc.fluents, sorts=desc.sorts)
    full = CE.compose(text, rules)
    layer = layer_of(name, full, distance_sort=vf.get("distance_sort", "distance"), base_text=base, generated=BS.relation_names(vf))
    gloss = note.strip() if origin == "hand" else WT.gloss(term, vf.get("glosses") or {}, {e["name"]: gloss_of(e) for e in reg["words"]})
    entry = {"name": name, "term": WT.show(term), "signature": list(WT.sort_of(term, desc.signatures, desc.sorts)), "rules": rules, "size": WT.size(term), "origin": origin,
             "layer": layer, "expansion": WT.show(exp), "note": note, "gloss": gloss}
    reg["words"].append(entry)
    save(game, reg)
    return entry, True


def define(game: str, term_text: str, *, origin: str, name: "str | None" = None, note: str = "", vocabulary: "str | None" = None) -> dict:
    return define_new(game, term_text, origin=origin, name=name, note=note, vocabulary=vocabulary)[0]


# ------------------------------------------------------------------------------------------------------------------------------------------ the CLI

def say_words(game: str, vocabulary: "str | None", say=print, *, expand: bool = False) -> dict:
    lay = layers(game, vocabulary, why=True)
    gen = sorted(r for r, (l, _w) in lay.items() if l == "general")
    gm = sorted(r for r, (l, _w) in lay.items() if l == "game")
    say(f"OBS words layer {game} {vocabulary}: general {gen} game {gm}")
    for r in gm:
        say(f"OBS words game because {r}: {lay[r][1]}")
    for e in load(game)["words"]:
        say(f"OBS word {e['name']} ({e['origin']}, {lay.get(e['name'], (e['layer'],))[0]}, size {e['size']}): {gloss_of(e)}")
        if expand:
            say(f"    = {e['term']}")
    return lay


def main(argv: list) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("game")
    ap.add_argument("--vocabulary", default=None)
    ap.add_argument("--store-root", default=None, help="a sandbox store root (its learned/ is used for the vocabulary records and words.json)")
    ap.add_argument("--expand", action="store_true", help="also print each word's term on the next line")
    a = ap.parse_args(argv)
    GS = _gs()
    if a.store_root:
        GS.LEARNED_ROOT = Path(a.store_root) / "learned"
    say_words(a.game, a.vocabulary, expand=a.expand)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
