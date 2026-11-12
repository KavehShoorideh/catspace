"""PLAN 78.32 INDUCTION FROM EXACT LABELS. A PROBLEM is a position where the wake's own decision plays a move after which the exact class for the mover is lower (playout_sleep). The corpus
(corpus.py) holds tens of thousands of positions with every legal move labelled keeps / loses by an exact labeller. INDUCTION is local and deterministic, no sampled bar:

  HEAD   the response of the rule, from the exact Reading of the problem (targeted.heads_for: a pattern over a move's effect that keeping moves have and no losing move has at the problem), the
         heads that single out the fewest moves first.
  BODY   a beam search over LIFTED literals true at the problem (targeted.body_candidates: every constant of the position one variable everywhere, the head's own variables shared with the
         head), scored on the corpus's DEVELOPMENT part by the full drill (corpus.drill_many: right = every named move keeps the class, wrong = a named move loses it, trivial = every move keeps):
         zero wrong first, then the most non-trivial right firings, then the fewest literals. It stops at the first conjunction size that has a body with zero wrong and at least `min_right`
         non-trivial right firings. At most `max_literals` literals; `beam` conjunctions kept per size (the single literals are first read on every `stride`-th development position, to save time, and the `beam`
         kept are read again on the whole part; every figure printed and every admission is on the whole part). A child's firings are a subset of its parent's, so a child is read only where the parent fired.
         NO LITERAL OVER A MOVE COUNTER: a fact of one numeric argument (relation(N): a half-move clock or move number) differs between positions by the game's age alone, so it is never a candidate
         (targeted.is_counter, generic: no relation is named).
  ADMISSION  (a) zero wrong and at least `min_right` non-trivial right firings on the development part, (b) zero wrong and at least `min_held` non-trivial right firings on the HELD-OUT part (groups the
         search never read), (c) the problem's own check: the wake's decision with a trial snapshot that holds the rule keeps the class, (d) no problem solved before is broken. The EXAM is never
         used here. The rule is written by the store's own writer (game_store.write_lemma + publish) with provenance `origin induced, mined_from problem`; no verdict of the labeller and no count of
         a drill is in the lemma or the snapshot.

  TIERS (plan 78.33)  the body literals come in three tiers (tiered_candidates): RELATIVE (every coordinate a variable), STRUCTURE (plus edge_gap / goal_gap literals on coordinate variables already
         in the rule, board_structure.py), NAMED (a coordinate constant kept). For each head the search runs on the narrowest tier first and widens only when no candidate passes the whole admission;
         the lemma records `meta.tier`. The `tiers` setting can stop the run early. A vocabulary with no `axes` has one "untiered" pass, as before.

  HEAD TIERS (plan 78.34)  with a structure (a vocabulary with `axes`) the heads come in two tiers (tiered_heads): RELATIVE (every coordinate argument of the head a variable) tried first, NAMED (a
         coordinate constant in the head, e.g. rank 5) only after every relative head failed; a named head also gives its relative form (coordinates lifted) unless that is a head already there. The
         lemma records `meta.head_tier`. The `head_tiers` setting orders / stops them. `max_problems` caps the problems attempted per call (the oldest open first; the rest are printed as left).

  NAMING (plan 78.35)  the setting `name_bodies` (default 0 = off): with N > 0 an ADMITTED rule whose body has N literals or more has its body defined as one word (word_registry.define, origin
         "mined"; body_term / name_body) and the lemma's meta gains `word` and `word_expansion`; what the rule matches is unchanged. `OBS word named <name>: <term> (rule <rule name>, size N)`.

Game-free: the variant string, the store's vocabulary, objects passed in (corpus, labeller); no game object is named here.

  induce(problem, corpus, *, reader, ...) -> {status "found" | "open", lemma, head, dev, held, why, stage, heads_tried}      (the search and the held-out part of the admission)
  solve_problems(game, problems, corpus, *, labeller, ...) -> {total, attempted, admitted, solved, still_open, reasons, rows, snapshot}      (the loop, the problem's check, the writer)
"""
from __future__ import annotations

import json
import tempfile
import time
from collections import Counter

from catspace.incubator.analyst.discovery import abstraction as AB
from catspace.incubator.analyst.discovery import apply_audit as AU
from catspace.incubator.analyst.discovery import board_structure as BS
from catspace.incubator.analyst.discovery import corpus as CP
from catspace.incubator.analyst.discovery import game_store as GS
from catspace.incubator.analyst.discovery import playout_sleep as PS
from catspace.incubator.analyst.discovery import targeted as TG
from catspace.incubator.analyst.discovery import word_registry as WR
from catspace.incubator.analyst.discovery import word_terms as WT

SETTINGS = {"max_literals": 4, "beam": 8, "min_right": 10, "min_held": 5, "max_heads": 4, "finalists": 3, "max_bindings": 256, "stride": 3, "tiers": ["relative", "structure", "named"],
            "max_problems": None, "head_tiers": ["relative", "named"], "name_bodies": 0}
TIERS = ("relative", "structure", "named")       # plan 78.33: the order of the body literals' tiers (the narrowest first)
HEAD_TIERS = ("relative", "named")               # plan 78.34: the order of the heads' tiers (coordinates as variables first, a named coordinate last)
STAGES = ("position", "head", "dev", "held", "check", "broken")          # how far a candidate got: the reason of the furthest one is the problem's


def _nconst(body: list) -> int:
    return sum(1 for l in body for a in l["args"] if not a.startswith("?"))


def _key(t: dict, body: list) -> tuple:
    return (t["wrong"], -t["right"], len(body), _nconst(body), json.dumps(body, sort_keys=True))


def tiered_candidates(facts: list, names: dict, structure: "BS.Structure") -> dict:
    """PLAN 78.33: the body literals of a problem position in three TIERS (cumulative: a wider tier's search may use the narrower tiers' literals too).
    Every constant of the position is one variable everywhere in a lifted literal (the head's own constants are the head's variables), as targeted.body_candidates. An argument is a
    COORDINATE when its declared sort (the `signature` of the relation) is a coordinate sort of `structure` (the vocabulary's axes); the others (sides, kinds, distances) are not.
      relative   no coordinate constant: each fact once with every constant lifted, once with the coordinates lifted and the other constants kept (a distance, a side token, a piece kind).
      named      a coordinate constant stays (the constants-kept form of targeted.body_candidates); a fact whose arguments have no declared sort counts here when it has any constant.
      structure  per coordinate of the position, the structure facts of the card (edge_gap_<sort>(?v, dK), goal_gap_<sort>(side, ?v, dK): the engine reads them at the position in the mover's
                 perspective, so the side token is us / them) with the distance constant kept and the side token kept or lifted. Carried as (lit, description, needs) with `needs` = the variables the literal is attached to: the search uses it
                 only in a body where those variables already occur in the head or in a non-structure literal.
    -> {"relative": [(lit, desc)], "structure": [(lit, desc, {needs})], "named": [(lit, desc)]}"""
    allmap = dict(names)
    srel = set(structure.relations())
    sfacts = [f for f in facts if f[0] in srel]
    facts = [f for f in facts if not TG.is_counter(f) and f[0] not in srel]
    for f in facts:
        for a in f[1:]:
            if TG.is_const(a) and a not in allmap:
                allmap[a] = f"?k{len(allmap)}"
    coord = set(structure.coordinate_sorts)
    out: dict = {"relative": [], "structure": [], "named": []}
    seen: set = set()
    coords: dict = {}                                                # coordinate constant -> axis sort, in order of first appearance
    for f in facts:
        rel, args = f[0], list(f[1:])
        sig = structure.signatures.get(rel)
        sorts = list(sig) if sig and len(sig) == len(args) else [None] * len(args)
        for a, srt in zip(args, sorts):
            if srt in coord and TG.is_const(a):
                coords.setdefault(a, srt)
        for kind in ("lift", "mixed", "keep"):
            new = []
            for a, srt in zip(args, sorts):
                if not TG.is_const(a):
                    new.append(a)
                elif kind == "lift" or a in names or (kind == "mixed" and srt in coord):
                    new.append(allmap[a])
                else:
                    new.append(a)
            lit = AB.lit(rel, new)
            key = json.dumps(lit, sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            named = any(TG.is_const(x) and (srt in coord or srt is None) for x, srt in zip(new, sorts) if not x.startswith("?"))
            out["named" if named else "relative"].append((lit, f"{rel}({', '.join(lit['args'])})"))
    for c in names:                                                  # a coordinate the head names but no fact shows
        for srt, seq in structure.order.items():
            if c in seq:
                coords.setdefault(c, srt)
    for f in sfacts:                                                 # the structure facts the engine reads at the position (perspective us / them), of the coordinates the rule can speak of
        rel, args = f[0], list(f[1:])
        c, d = args[-2], args[-1]
        if c not in coords:
            continue
        prefix = args[:-2]
        forms = [prefix] + ([[allmap[x] for x in prefix]] if prefix and all(x in allmap for x in prefix) else [])
        for fm in forms:
            lit = AB.lit(rel, fm + [allmap[c], d])
            key = json.dumps(lit, sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            out["structure"].append((lit, f"{rel}({', '.join(lit['args'])})", {allmap[c]}))
    return out


def tier_passes(facts: list, names: dict, structure: "BS.Structure | None", tiers=TIERS) -> list:
    """The searches of one head, narrowest first: [(tier, cands [(lit, desc)], needs {literal json: {variables}})]. A tier that adds no literal to the one before is left out. With no
    `structure` (or none with axes) there is ONE pass, tier "untiered": targeted.body_candidates as before."""
    if structure is None or not structure.axes:
        return [("untiered", TG.body_candidates(facts, names, set()), {})]
    c = tiered_candidates(facts, names, structure)
    passes, cands, needs, prev = [], [], {}, 0
    for tier in TIERS:
        if tier not in tiers:
            continue
        if tier == "relative":
            cands = cands + c["relative"]
        elif tier == "structure":
            cands = cands + [(l, d) for l, d, _n in c["structure"]]
            needs = {json.dumps(l, sort_keys=True): n for l, _d, n in c["structure"]}
        else:
            cands = cands + c["named"]
        if len(cands) > prev:
            passes.append((tier, list(cands), dict(needs)))
            prev = len(cands)
    return passes


def _alpha_key(head: dict) -> str:
    """The head's text with its variables renamed by first occurrence: two heads that differ only in their variable names are one."""
    ren: dict = {}
    f = lambda a: ren.setdefault(a, f"?v{len(ren)}") if a.startswith("?") else a  # noqa: E731
    return json.dumps({"removed": [[rel, [f(a) for a in args]] for rel, args in head["removed"]], "added": [[rel, [f(a) for a in args]] for rel, args in head["added"]]}, sort_keys=True)


def head_coordinates(head: dict, structure: "BS.Structure") -> list:
    """The coordinate CONSTANTS of a head ({removed, added}: [(relation, args)]): the arguments that are not variables and whose declared sort (the relation's `signature`) is a coordinate sort of
    `structure`; a relation with no declared signature has none. -> [constant, ...] in order of appearance, with repeats."""
    coord = set(structure.coordinate_sorts)
    out = []
    for lits in (head["removed"], head["added"]):
        for rel, args in lits:
            sig = structure.signatures.get(rel)
            if not sig or len(sig) != len(args):
                continue
            out += [a for a, srt in zip(args, sig) if srt in coord and TG.is_const(a)]
    return out


def relative_form(h: dict, card: dict, keep: dict, structure: "BS.Structure") -> "dict | None":
    """The head `h` (an entry of targeted.heads_for) with every coordinate constant lifted to a variable (the same constant, the same variable; the other constants as they were), or None
    when it has no coordinate constant or the lifted head names a losing move at this position (a head must name keeping moves only) or no move."""
    consts = head_coordinates(h["head"], structure)
    if not consts:
        return None
    names = dict(h["names"])
    taken = {a for lits in (h["head"]["removed"], h["head"]["added"]) for _r, args in lits for a in args if a.startswith("?")}
    for c in consts:
        if c not in names:
            k = 0
            while f"?hc{k}" in taken:
                k += 1
            names[c] = f"?hc{k}"
            taken.add(names[c])
    lift = lambda a: names[a] if a in consts else a  # noqa: E731
    head = {"removed": [(rel, [lift(a) for a in args]) for rel, args in h["head"]["removed"]], "added": [(rel, [lift(a) for a in args]) for rel, args in h["head"]["added"]]}
    moves = {m["name"]: m for m in card["moves"]}
    matched = sorted(x for x, mv in moves.items() if TG.head_matches(head, mv["removed"], mv["added"]))
    if not matched or any(not keep.get(x, True) for x in matched):
        return None
    nconst = len({a for lits in (head["removed"], head["added"]) for _r, args in lits for a in args if TG.is_const(a)})
    return {"head": head, "matched": matched, "lits": h["lits"], "consts": nconst, "names": names}


def tiered_heads(card: dict, keep: dict, structure: "BS.Structure | None", *, max_heads: int, head_tiers=HEAD_TIERS) -> list:
    """PLAN 78.34: the heads to try, RELATIVE heads (no coordinate constant: every coordinate a variable) before NAMED heads (a coordinate named), each entry gaining `head_tier`. With no `structure`
    (no axes in the vocabulary) the heads of targeted.heads_for as before, no `head_tier`. Otherwise targeted.heads_for is asked for a wider list; its heads with no coordinate constant are the
    relative ones, and a named head also gives its RELATIVE FORM (coordinates lifted, kept only if it still names keeping moves alone) unless another relative head is the same head up to the
    names of its variables (no duplicate). Each tier is ordered as heads_for orders (fewest moves matched, fewest literals, fewest constants) and cut to `max_heads`; `head_tiers` says which tiers
    are tried and in what order (default relative, named)."""
    if structure is None or not structure.axes:
        return TG.heads_for(card, keep, limit=max_heads)
    pool = TG.heads_for(card, keep, limit=max(4 * max_heads, 16))
    rel, named, seen = [], [], set()
    for h in pool:
        if not head_coordinates(h["head"], structure):
            if _alpha_key(h["head"]) not in seen:
                seen.add(_alpha_key(h["head"]))
                rel.append(h)
    for h in pool:
        if head_coordinates(h["head"], structure):
            named.append(h)
            r = relative_form(h, card, keep, structure)
            if r is not None and _alpha_key(r["head"]) not in seen:
                seen.add(_alpha_key(r["head"]))
                rel.append(r)
    order = lambda c: (len(c["matched"]), c["lits"], c["consts"], json.dumps(c["head"], sort_keys=True))  # noqa: E731
    tiers = {"relative": sorted(rel, key=order)[:max_heads], "named": sorted(named, key=order)[:max_heads]}
    out = []
    for t in head_tiers:
        if t not in tiers:
            raise ValueError(f"unknown head tier {t!r}; the tiers are {HEAD_TIERS}")
        out += [{**h, "head_tier": t} for h in tiers[t]]
    return out


def search_body(corpus: "CP.Corpus", facts: list, names: dict, head: dict, *, max_literals: int, beam: int, min_right: int, finalists: int, max_bindings: int, part: str = "dev", stride: int = 1,
                cands: "list | None" = None, needs: "dict | None" = None) -> dict:
    """The beam search for one head -> {found: [(body, tally)] (zero wrong and >= min_right right, the best first; [] if none), best: (body, tally) with the fewest wrong then the most right
    seen at any size, levels: [{literals, evaluated, kept}]}. `cands` [(lit, desc)] replaces targeted.body_candidates (a tier's literals); `needs` {literal json: {variables}} marks the
    structure literals, which join a body only where each of their variables is a variable of the head or occurs in a non-structure literal of the body."""
    if cands is None:
        cands = TG.body_candidates(facts, names, set())
    needs = needs or {}
    head_vars = set(names.values())
    levels, best = [], None
    frontier = [([], None)]                                         # (body, tally of the parent: None = read on the whole part)
    seen: set = set()
    for size in range(1, max_literals + 1):
        children = []
        for body, ptally in frontier:
            have = {json.dumps(l, sort_keys=True) for l in body}
            scope = head_vars | {a for l in body if json.dumps(l, sort_keys=True) not in needs for a in l["args"] if a.startswith("?")}
            todo = []
            for lit, _desc in cands:
                lk = json.dumps(lit, sort_keys=True)
                if lk in have or (lk in needs and not needs[lk] <= scope):
                    continue
                nb = body + [lit]
                ck = frozenset(json.dumps(l, sort_keys=True) for l in nb)
                if ck in seen:
                    continue
                seen.add(ck)
                todo.append(nb)
            if not todo:
                continue
            idx = corpus.idx(part)[::max(1, stride)] if ptally is None else sorted(ptally["right_at"] + ptally["wrong_at"])
            if idx is not None and not idx:
                continue
            ts = CP.drill_many([TG.lemma_of(nb, head) for nb in todo], corpus, part=part, indices=idx, max_bindings=max_bindings)
            for nb, t in zip(todo, ts):
                if t["truncated"] or t["right"] == 0:
                    continue
                children.append((nb, t))
        levels.append({"literals": size, "evaluated": len(children)})
        if not children:
            break
        children.sort(key=lambda c: _key(c[1], c[0]))
        if size == 1 and stride > 1:                                # the first size was read on every `stride`-th position: the conjunctions kept are read again on the whole part
            top = children[:beam]
            full = CP.drill_many([TG.lemma_of(nb, head) for nb, _t in top], corpus, part=part, max_bindings=max_bindings)
            children = [(nb, t) for (nb, _o), t in zip(top, full) if not t["truncated"] and t["right"] > 0]
            if not children:
                break
            children.sort(key=lambda c: _key(c[1], c[0]))
        if best is None or _key(children[0][1], children[0][0])[:2] < _key(best[1], best[0])[:2]:
            best = children[0]
        good = [c for c in children if c[1]["wrong"] == 0 and c[1]["right"] >= min_right]
        levels[-1]["kept"] = min(beam, len(children))
        if good:
            good.sort(key=lambda c: (-c[1]["right"], len(c[0]), _nconst(c[0]), json.dumps(c[0], sort_keys=True)))
            return {"found": good[:finalists], "best": best, "levels": levels}
        frontier = children[:beam]
    return {"found": [], "best": best, "levels": levels}


def _further(cur: dict, new: dict) -> dict:
    """The candidate that got further (STAGES), at the same stage on the development part the one with fewer wrong, then more right firings."""
    a, b = STAGES.index(cur["stage"]), STAGES.index(new["stage"])
    if b != a:
        return new if b > a else cur
    if new["stage"] == "dev" and new["dev"] is not None and (cur["dev"] is None or (new["dev"]["wrong"], -new["dev"]["right"]) < (cur["dev"]["wrong"], -cur["dev"]["right"])):
        return new
    return cur


def rule_text(lemma: "dict | None") -> str:
    return AU.render(lemma).splitlines()[0] if lemma else "none"


def _counts(t: "dict | None") -> str:
    return "n/a" if t is None else f"{t['binds']}/{t['right']}/{t['wrong']}/{t['trivial']}"


def induce(problem: dict, corpus: "CP.Corpus", *, reader: "TG.Reader", max_literals: int = SETTINGS["max_literals"], beam: int = SETTINGS["beam"], min_right: int = SETTINGS["min_right"],
           min_held: int = SETTINGS["min_held"], max_heads: int = SETTINGS["max_heads"], finalists: int = SETTINGS["finalists"], max_bindings: int = SETTINGS["max_bindings"], stride: int = SETTINGS["stride"], accept=None, say=print, tag: str = "",
           structure: "BS.Structure | None" = None, tiers=SETTINGS["tiers"], head_tiers=SETTINGS["head_tiers"]) -> dict:
    """A rule for `problem` ({start, moves}) from the corpus's exact labels: the heads of its exact Reading, a body by beam search on the development part, the held-out admission on the held-out
    part, then `accept(candidate) -> (ok, why, stage)` when given (solve_problems': the problem's own check and the no-regression check on a trial snapshot; a rule it refuses
    is not found, and the search goes on with the next candidate and head). -> {status: "found" | "open", lemma, head, dev, held (drill counts), why, stage (how far the furthest candidate
    got: STAGES), heads_tried, candidates ([the accepted rule]), tier (the tier of the rule found), tiers {tier: {stage, near, why}}}.
    PLAN 78.33: with a `structure` (board_structure.Structure of the game's axes) each head is searched tier by tier (`tiers`, from TIERS: relative, structure, named; each wider tier
    only when the narrower ones gave no rule that passes both parts of the drill and the problem's check); without one, a single "untiered" pass as before."""
    p = {"start": problem.get("start"), "moves": list(problem["moves"])}
    try:
        rd = reader.read(p, "problem", "problem")
    except LookupError as e:
        return {"status": "open", "why": f"the labeller does not cover the position ({e})", "stage": "position", "lemma": None, "head": None, "dev": None, "held": None, "heads_tried": 0, "candidates": []}
    if rd is None:
        return {"status": "open", "why": "the position is over", "stage": "position", "lemma": None, "head": None, "dev": None, "held": None, "heads_tried": 0, "candidates": []}
    if not rd.good or not rd.bad:
        why = "no move keeps the class" if not rd.good else "every move keeps the class"
        return {"status": "open", "why": why, "stage": "position", "lemma": None, "head": None, "dev": None, "held": None, "heads_tried": 0, "candidates": []}
    card = json.loads(reader.cards([(p["start"], p["moves"], "p", "problem")]).card(0))
    heads = tiered_heads(card, rd.keep, structure, max_heads=max_heads, head_tiers=head_tiers)
    if not heads:
        return {"status": "open", "why": "no pattern over the moves' effects separates the keeping moves from the losing ones", "stage": "head", "lemma": None, "head": None, "dev": None, "held": None,
                "heads_tried": 0, "candidates": []}
    say(f"OBS induce {tag}: position {len(p['moves'])} plies, class {rd.before}, {len(rd.keep)} legal moves, keep {sorted(rd.good)}, lose {sorted(rd.bad)}; {len(heads)} head(s): "
        + "; ".join(f"{TG.head_text(h['head'])} names {h['matched']}" + (f" [{h['head_tier']} head]" if h.get("head_tier") else "") for h in heads))
    furthest = {"stage": "head", "why": "", "lemma": None, "head": None, "dev": None, "held": None}
    progress: dict = {}                                             # tier -> {stage, near (a rule passed the development part and failed the held-out part), why}: how far each tier got
    for hi, h in enumerate(heads, 1):
        if h.get("head_tier"):
            say(f"OBS induce {tag} head {hi}: {h['head_tier']} head {TG.head_text(h['head'])}")
        for tier, cands, needs in tier_passes(card["facts"], h["names"], structure, tiers):
            pg = progress.setdefault(tier, {"stage": "head", "near": False, "why": ""})
            r = search_body(corpus, card["facts"], h["names"], h["head"], max_literals=max_literals, beam=beam, min_right=min_right, finalists=finalists, max_bindings=max_bindings, stride=stride,
                            cands=cands, needs=needs)
            tt = f" {tier}" if tier != "untiered" else ""
            if not r["found"]:
                b = r["best"]
                if b is None:
                    why = f"head {hi}: no literal true at the problem leaves a non-trivial right firing on the development part"
                    lem, dev = None, None
                else:
                    lem = TG.lemma_of(b[0], h["head"])
                    dev = {k: b[1][k] for k in ("binds", "names_a_move", "right", "wrong", "trivial")}
                    why = (f"head {hi}: no conjunction of at most {max_literals} literals has zero wrong and at least {min_right} right on the development part (the best has {b[1]['wrong']} wrong, "
                           f"{b[1]['right']} right with {len(b[0])} literal(s))")
                furthest = _further(furthest, {"stage": "dev", "why": why, "lemma": lem, "head": h["head"], "dev": dev, "held": None})
                if STAGES.index(pg["stage"]) < STAGES.index("dev"):
                    pg.update(stage="dev", why=why)
                say(f"OBS induce {tag} head {hi}{tt}: {TG.head_text(h['head'])}: {why}")
                continue
            passed = []
            for body, t in r["found"]:
                lem = TG.lemma_of(body, h["head"])
                lem["name"] = AB.name_of(lem)
                dev = CP.drill(lem, corpus, part="dev", max_bindings=max_bindings)
                held = CP.drill(lem, corpus, part="held", max_bindings=max_bindings)
                ok = held["wrong"] == 0 and held["right"] >= min_held and not held["truncated"] and dev["wrong"] == 0
                say(f"OBS induce {tag} head {hi}{tt} candidate: rule {rule_text(lem)}; dev binds/right/wrong/trivial {_counts(dev)}, held {_counts(held)}; "
                    + ("passes the held-out part" if ok else f"REJECTED on the held-out part: " + (f"{held['wrong']} wrong firing(s)" if held["wrong"] else f"{held['right']} right firing(s), need {min_held}")))
                if ok:
                    cand = {"lemma": lem, "head": h["head"], "dev": dev, "held": held, "tier": tier, "head_tier": h.get("head_tier")}
                    good, why2, stage2 = accept(cand) if accept else (True, "", "held")
                    if good:
                        passed.append(cand)
                    else:
                        furthest = _further(furthest, {"stage": stage2, "why": f"head {hi}: {why2}", "lemma": lem, "head": h["head"], "dev": dev, "held": held})
                        if STAGES.index(pg["stage"]) < STAGES.index(stage2):
                            pg.update(stage=stage2, why=f"head {hi}: {why2}")
                else:
                    if dev["wrong"] == 0:
                        pg["near"] = True                           # a near miss: passed the development part, failed the held-out part
                    why3 = (f"head {hi}: the rule passes the development part but has {held['wrong']} wrong firing(s) on the held-out part" if held["wrong"]
                            else f"head {hi}: the rule passes the development part but only {held['right']} right firing(s) on the held-out part (need {min_held})")
                    furthest = _further(furthest, {"stage": "held", "why": why3, "lemma": lem, "head": h["head"], "dev": dev, "held": held})
                    if STAGES.index(pg["stage"]) < STAGES.index("held"):
                        pg.update(stage="held", why=why3)
            if passed:
                return {"status": "found", "lemma": passed[0]["lemma"], "head": passed[0]["head"], "dev": passed[0]["dev"], "held": passed[0]["held"], "why": "", "stage": "held", "heads_tried": hi,
                        "candidates": passed, "tier": tier, "tiers": progress, "head_tier": passed[0]["head_tier"]}
    return {"status": "open", "lemma": furthest["lemma"], "head": furthest["head"], "dev": furthest["dev"], "held": furthest["held"], "why": furthest["why"] or "no head gave a rule", "stage": furthest["stage"],
            "heads_tried": len(heads), "candidates": [], "tier": None, "tiers": progress, "head_tier": None}


# ------------------------------------------------------------------------------------------------------------------------------------ the loop over the problems


def drill_admit(lemmas: list, corpus, spec: dict, vocab_json: str, *, max_bindings: int = 256, say=print) -> tuple:
    """The full drill as the admission test of rules that came from ANOTHER miner (the lemma sleep's): each rule is run over both parts of the exact corpus and kept only with zero
    wrong firings on each. -> (kept, dropped) lists of lemma dicts; one OBS line per rule. Nothing of the drill is written into a lemma."""
    corpus.cards(spec, vocab_json)
    kept, dropped = [], []
    for lem in lemmas:
        dev = CP.drill(lem, corpus, part="dev", max_bindings=max_bindings)
        held = CP.drill(lem, corpus, part="held", max_bindings=max_bindings)
        ok = dev["wrong"] == 0 and held["wrong"] == 0
        (kept if ok else dropped).append(lem)
        say(f"OBS drill admit {lem['name']}: dev binds/right/wrong/trivial {dev['binds']}/{dev['right']}/{dev['wrong']}/{dev['trivial']}, held {held['binds']}/{held['right']}/{held['wrong']}/{held['trivial']}; "
            + ("KEPT" if ok else "DROPPED: wrong firings on the exact corpus"))
    say(f"OBS drill admit total: rules {len(lemmas)}, kept {len(kept)}, dropped {len(dropped)}")
    return kept, dropped


def trial_snapshot(game: str, lemma: dict, directory, vocabulary: "str | None" = None, judge: "str | None" = None) -> str:
    """The snapshot the store would publish with `lemma` added (the same content, so the same id as the published one), written under `directory`: `judge` is the NAME of the valuation record the wake
    plays by (game_store.judge_of; None: the store's record in use), so the trial reads the position as the wake will. (targeted.trial_snapshot leaves the record out.)"""
    import cs_engine as CE
    from pathlib import Path
    data = json.loads(GS.build(game, parent=GS.head(game), teacher_used=True, judge=GS.judge_of(game, judge) if judge else "store", vocabulary=vocabulary))
    data["lemmas"] = sorted([l for l in data["lemmas"] if l["name"] != lemma["name"]] + [lemma], key=lambda l: GS.slug(l["name"]))
    data["teacher_used"] = True
    raw = json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
    p = Path(directory) / f"{CE.content_id(raw)}.json"
    p.write_bytes(raw)
    CE.snapshot_info(str(p))
    return str(p)


SIDE_TOKENS = ("us", "them")          # the lemma language's side tokens: in a word they are ordinary variables (a word holds for either side)


def body_term(lemma: dict) -> str:
    """PLAN 78.35: the lemma's BODY as a term text (word_terms): the `and` of its literals (a negated literal as `(not ..)`), the side tokens `us` / `them` as the variables ?us / ?them (and a
    `(distinct ?us ?them)` when both occur), the variables the response uses free and every other variable projected away with `exists`. Raises ValueError when there is nothing to name."""
    def slot(a: str) -> str:
        return "?" + a if a in SIDE_TOKENS else a
    items, seen = [], []
    for l in lemma["body"]:
        args = [slot(a) for a in l["args"]]
        seen += [a for a in args if a.startswith("?")]
        lit = "(" + " ".join([l["relation"]] + args) + ")"
        items.append(f"(not {lit})" if l.get("neg") else lit)
    if len(items) < 1:
        raise ValueError("an empty body names nothing")
    if "?us" in seen and "?them" in seen:
        items.append("(distinct ?us ?them)")
    resp = lemma.get("response") or {}
    head = {slot(a) for k in ("removed", "added") for l in resp.get(k, []) for a in l["args"]}
    hidden = [v for v in dict.fromkeys(seen) if v not in head]
    body = "(and " + " ".join(items) + ")"
    return f"(exists ({' '.join(hidden)}) {body})" if hidden else body


def name_body(game: str, lemma: dict, vocabulary: "str | None", *, say=print) -> "dict | None":
    """PLAN 78.35 (4): define the body of an admitted rule as ONE word (word_registry.define, origin "mined") and record `word` and `word_expansion` in the lemma's meta; what the rule matches is
    NOT changed (the body stays as it is). -> the registry entry, or None when the body cannot be written as a word (said). The word is composed into the description by game_store when the
    vocabulary file says `"words": true`, from the next spec_texts on, so a later rule can use it."""
    try:
        entry, created = WR.define_new(game, body_term(lemma), origin="mined", note=f"the body of rule {lemma['name']}", vocabulary=vocabulary)
    except (ValueError, KeyError, WT.TermError) as e:
        say(f"OBS word not named (rule {lemma['name']}): {e}")
        return None
    lemma.setdefault("meta", {}).update({"word": entry["name"], "word_expansion": entry["expansion"]})
    say(f"OBS word {'named' if created else 'reused'} {entry['name']}: {entry['term']} (rule {lemma['name']}, size {entry['size']})")
    return entry


def provenance(problem: dict, variant: str, labeller, vocabulary: "str | None", tier: "str | None" = None, head_tier: "str | None" = None) -> dict:
    """The lemma's meta: where it came from. NO number and no verdict of the labeller."""
    return {"origin": "induced", "plan": "78.32", "game": variant, "mined_from": "problem", "problem": {"start": problem.get("start"), "moves": len(problem["moves"]), "played": problem.get("played")},
            "teacher": True, "source": getattr(labeller, "source", "exact"), **({"vocabulary": vocabulary} if vocabulary else {}),
            **({"tier": tier} if tier else {}), **({"head_tier": head_tier} if head_tier else {})}


def solve_problems(game: str, problems, corpus: "CP.Corpus", *, variant: "str | None" = None, labeller, vocabulary: "str | None" = None, spec: "dict | None" = None, seconds: float = 0.2,
                   wake_kw: "dict | None" = None, judge: "str | None" = None, settings: "dict | None" = None, say=print) -> dict:
    """Induce a rule for each OPEN problem of `problems` (rows {start, moves, played?}: the cycle's rows, or a problem file's path) and admit it when the held-out part and the problem's own check say
    so. `judge`: the name of the valuation record the wake plays by (trial snapshots and the published snapshot carry it). The store paths must already point at the sandbox (initialize.run_size does that, or targeted.sandbox). The rules are written with game_store.write_lemma and a snapshot is published, as the
    lemma sleep does; after each admission every problem is checked again with the head snapshot (a rule may solve several). `settings`: SETTINGS.
    -> {total, attempted, admitted [{name, text, problem, solved_others, snapshot, dev, held}], solved (open at the start, kept at the end), still_open, reasons {why: problems}, rows, snapshot}.
    Prints `OBS induce problem i/N: ...` per problem and `OBS induce total: ...`."""
    t0 = time.perf_counter()
    s = {**SETTINGS, **(settings or {})}
    variant = variant or game
    rows = list(problems) if isinstance(problems, (list, tuple)) else PS.load_problems(problems)
    spec = spec if spec is not None else GS.spec_texts(game, vocabulary)
    vocab_json = json.dumps(GS.vocabulary(game, vocabulary))
    corpus.cards(spec, vocab_json)
    vfile = GS.vocabulary_file(game, vocabulary)
    structure = BS.from_vocabulary(spec[variant], vfile) if spec.get(variant) else None          # (no axes in the vocabulary file: one untiered pass, as before 78.33)
    if structure is not None and not structure.axes:
        structure = None
    verdicts = PS.Verdicts(labeller, variant)
    reader = TG.Reader(variant, spec, vocab_json, verdicts)
    wk = wake_kw or {}

    def check_rows(snapshot_path: "str | None", which: list) -> list:
        return PS.check([{"start": r.get("start"), "moves": r["moves"]} for r in which], variant=variant, store=snapshot_path, spec=spec, judge=labeller, seconds=seconds, wake_kw=wk, verdicts=verdicts)

    def head_path() -> "str | None":
        h = GS.head(game)
        return str(GS.path_of(game, h)) if h else None

    res = check_rows(head_path(), rows)
    status = ["solved" if c["keeps"] else "open" for c in res]
    start_open = [i for i, st in enumerate(status) if st == "open"]
    info = {i: {"index": i, "status": status[i], "why": ""} for i in range(len(rows))}
    admitted, attempted, reasons, done = [], 0, Counter(), set()
    near: Counter = Counter()                                       # per tier: the problems where a rule passed the development part and failed the held-out part
    with tempfile.TemporaryDirectory() as tmp:
        order = range(len(rows)) if s["max_problems"] is None else sorted(range(len(rows)), key=lambda j: (rows[j].get("first_seen", 0), j))          # plan 78.34: oldest first when capped
        for i in order:
            if status[i] != "open" or i in done:
                continue
            if s["max_problems"] is not None and attempted >= int(s["max_problems"]):
                break
            done.add(i)
            attempted += 1
            tag = f"problem {i + 1}/{len(rows)}"
            solved_rows = [rows[j] for j in range(len(rows)) if status[j] == "solved"]
            seen_check: dict = {}

            def accept(cand, i=i, solved_rows=solved_rows, seen_check=seen_check):
                lem = cand["lemma"]
                lem["meta"] = provenance(rows[i], variant, labeller, vocabulary, cand.get("tier") if cand.get("tier") != "untiered" else None, cand.get("head_tier"))
                lem["stats"] = {"status": "INDUCED"}
                GS.check_lemma(lem)
                snap = trial_snapshot(game, lem, tmp, vocabulary, judge)
                chk = check_rows(snap, [rows[i]] + solved_rows)
                own, broken = chk[0], [r for r, c in zip(solved_rows, chk[1:]) if not c["keeps"]]
                seen_check["keeps"] = own["keeps"]
                if not own["keeps"]:
                    return False, f"the problem's own check fails with the rule: the decision still chooses {own['move']}", "check"
                if broken:
                    return False, f"the rule breaks {len(broken)} problem(s) solved before", "broken"
                return True, "", "held"
            out = induce(rows[i], corpus, reader=reader, max_literals=s["max_literals"], beam=s["beam"], min_right=s["min_right"], min_held=s["min_held"], max_heads=s["max_heads"],
                         finalists=s["finalists"], max_bindings=s["max_bindings"], stride=s["stride"], accept=accept, say=say, tag=tag, structure=structure, tiers=s["tiers"], head_tiers=s["head_tiers"])
            why, stage, own_keeps = out["why"], out["stage"], seen_check.get("keeps")
            final = out["lemma"] if out["status"] == "found" else None
            line = (f"OBS induce {tag}: head {TG.head_text(out['head']) if out['head'] else 'none'}, rule {rule_text(out['lemma'])}, dev binds/right/wrong/trivial {_counts(out['dev'])}, "
                    f"held binds/right/wrong/trivial {_counts(out['held'])}, problem check kept {own_keeps if own_keeps is not None else 'not run'}; ")
            tier_prog = out.get("tiers") or {}
            for tn, pg in tier_prog.items():
                if pg["near"]:
                    near[tn] += 1
            if final is None:
                info[i]["why"] = why
                reasons[_reason_key(stage, why)] += 1
                say(line + f"open because {why}" + (("; tiers: " + "; ".join(f"{tn} got as far as {pg['stage']}" + (" (near miss: passed the development part, failed the held-out part)" if pg["near"] else "")
                                                    for tn, pg in tier_prog.items())) if tier_prog and "untiered" not in tier_prog else ""))
                continue
            if int(s["name_bodies"] or 0) > 0 and len(final["body"]) >= int(s["name_bodies"]):
                name_body(game, final, vocabulary, say=say)                  # (plan 78.35: the lemma's body is named; its meaning is unchanged)
            GS.write_lemma(game, final)
            sid = GS.publish(game, teacher_used=True, vocabulary=vocabulary, **({"judge": GS.judge_of(game, judge)} if judge else {}))
            now_res = check_rows(head_path(), rows)
            now = ["solved" if c["keeps"] else "open" for c in now_res]
            newly = [j for j in range(len(rows)) if status[j] == "open" and now[j] == "solved"]
            lost = [j for j in range(len(rows)) if status[j] == "solved" and now[j] == "open"]
            status = now
            for j in newly:
                info[j]["status"] = "solved"
                info[j]["why"] = f"by rule {final['name']}"
            admitted.append({"name": final["name"], "text": rule_text(final), "problem": i, "solved_others": [j for j in newly if j != i], "broke": lost, "snapshot": sid, "dev": out["dev"], "held": out["held"], "tier": out.get("tier"), "head_tier": out.get("head_tier")})
            say(line + f"ADMITTED {final['name']} (snapshot {sid}, tier {out.get('tier')}" + (f", head tier {out.get('head_tier')}" if out.get("head_tier") else "") + f"); solved {len(newly)} problem(s) in all, the others {[j + 1 for j in newly if j != i]}; broke {len(lost)}")
    solved = sum(1 for i in start_open if status[i] == "solved")
    left = sum(1 for i in start_open if status[i] == "open" and i not in done)
    if s["max_problems"] is not None:
        say(f"OBS induce deferred: max_problems {int(s['max_problems'])}, open at the start {len(start_open)}, attempted {attempted}, left for the next cycle {left}")
    open_n = sum(1 for st in status if st == "open")
    say(f"OBS induce total: problems {len(rows)}, attempted {attempted}, admitted {len(admitted)}, solved {solved}, still open {open_n}, reasons {dict(reasons)}, "
        f"admitted per tier {dict(Counter(a['tier'] for a in admitted))}, near misses per tier {dict(near)}")
    for a in admitted:
        say(f"OBS induce rule {a['name']}: {a['text']}; dev binds/right/wrong/trivial {_counts(a['dev'])}, held {_counts(a['held'])}; mined from problem {a['problem'] + 1}, also solved {[j + 1 for j in a['solved_others']]}")
    say(f"OBS induce time: {time.perf_counter() - t0:.1f} s, labeller calls {verdicts.calls}")
    return {"total": len(rows), "attempted": attempted, "admitted": admitted, "solved": solved, "still_open": open_n, "reasons": dict(reasons), "left": left, "rows": [info[i] for i in range(len(rows))],
            "snapshot": GS.head(game)}


def _reason_key(stage: str, why: str) -> str:
    """A short stable name for the reason a problem stayed open (the counts of `OBS induce total`)."""
    if stage == "dev":
        return "no conjunction with zero wrong and enough right on the development part"
    if stage == "held":
        return "held-out part: wrong firings" if "wrong firing" in why else "held-out part: too few right firings"
    if stage == "check":
        return "the problem's own check fails with the rule"
    if stage == "broken":
        return "the rule breaks a problem solved before"
    return why.split(" (")[0]
