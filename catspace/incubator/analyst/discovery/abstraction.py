"""PLAN 78.14 ABSTRACTION: from one example of a saving move to a lemma that generalizes, by EXACT checks (the architecture document's step 5, "the hardest step and the
main research risk"). Symbolic generalization pruned by search, as the document's open question names it:

  SEED        the example as a GROUND lemma: the response is the saving move's effect on the facts; the body is the facts about what the move touches -- the facts that
              share a constant with the response, ignoring constants that appear in a large share of the position's facts (a side, a common kind), plus the facts
              made only of the response's own constants.
  GENERALIZE  hill-climb over single steps: LIFT a constant to a variable everywhere it occurs, or DROP a body literal. A step is taken only if the lemma it gives is
              still ADMISSIBLE on the development cards -- precision at the gate, an excess over chance, a response that does not match most of the legal moves -- and
              it fires on at least as many cards. The first rank among admissible steps is the one that fires most, then the fewer constants (lifting generalizes: it comes before
              dropping, so the body keeps the literals that exclude the cases the claim is false on), then the fewer literals.
              The cards are read by the engine (CardSet.eval): the checks are the engine's, not Python's.
  The perspective constants `us` and `them` (and `r1`, `r2` ... of a game of more roles) are never lifted: they are the side to move and the other side.

Game-free: it reads facts as relation-and-arguments text and the engine's evaluation of a lemma on the cards; it names no game.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field

PROTECTED = re.compile(r"^(us|them|r\d+)$")


def is_var(a: str) -> bool:
    return a.startswith("?")


def lit(relation: str, args: list, neg: bool = False) -> dict:
    return {"relation": relation, "args": list(args), "neg": neg}


def lower_bound(holds: int, judged: int, confidence: float = 0.95) -> float:
    """The lower end of the Wilson score interval for the precision `holds / judged` at `confidence` (one-sided): what the precision is at least, given how few cards
    were read. 3 of 3 at 95 %: 0.53; 10 of 10: 0.79; 14 of 14: 0.84; 29 of 29: 0.91. Zero when nothing was judged."""
    if judged <= 0:
        return 0.0
    z = _z(confidence)
    p = holds / judged
    denom = 1 + z * z / judged
    centre = p + z * z / (2 * judged)
    margin = z * math.sqrt(p * (1 - p) / judged + z * z / (4 * judged * judged))
    return max(0.0, (centre - margin) / denom)


def _z(confidence: float) -> float:
    """The one-sided standard normal quantile of `confidence` (the inverse of the normal CDF, by bisection on erf: no dependency)."""
    if not 0.5 <= confidence < 1.0:
        raise ValueError(f"confidence {confidence}: between 0.5 and 1")
    lo, hi = 0.0, 10.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if 0.5 * (1 + math.erf(mid / math.sqrt(2))) < confidence:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


@dataclass
class Gates:
    """The admission gates. A lemma is ADMISSIBLE when its precision on the cards is at least `min_precision` (1.0: no refutation among the cards it was read on --
    a refutation is an exact counterexample), it beats the chance of an arbitrary choice of as many moves by at least `min_excess`, and its response matches at most
    `max_selectivity` of the legal moves on average. It is SUPPORTED when it fires on enough cards of enough independent groups AND the lower confidence bound of its
    precision (`lower_bound` at `confidence`) is at least `min_lower`: a point estimate of 1.0 on three cards says little (its bound is 0.53); the bound says how much
    the cards have shown. The 96-game sleep of 2026-11-06 passed eight lemmas on 3-11 clean development firings and lost all eight on held-out games; this bound would
    have stopped six of them before that stage."""
    min_precision: float = 1.0
    min_excess: float = 0.2
    max_selectivity: float = 0.5
    min_fires: int = 3
    min_groups: int = 2
    min_lower: float = 0.7
    confidence: float = 0.95
    max_bindings: int = 64
    strict_frac: float = 0.10
    max_body: int = 24

    def admissible(self, ev: dict) -> bool:
        return (ev["fires"] > 0 and ev["precision"] >= self.min_precision and ev["excess"] >= self.min_excess
                and ev["selectivity"] <= self.max_selectivity and not ev["truncated"])

    def lower(self, ev: dict) -> float:
        return lower_bound(ev["holds"], ev["holds"] + ev["refuted"], self.confidence)

    def supported(self, ev: dict) -> bool:
        return ev["fires"] >= self.min_fires and ev["groups"] >= self.min_groups and self.lower(ev) >= self.min_lower

    def as_dict(self) -> dict:
        return dict(self.__dict__)


# ---------------------------------------------------------------------------------------------------------------- the lemma as a dict

def constants(lemma: dict) -> list:
    """The constants of the lemma that may be lifted, in order of first appearance."""
    seen = []
    for l in lemma["body"] + lemma["response"]["removed"] + lemma["response"]["added"]:
        for a in l["args"]:
            if not is_var(a) and not PROTECTED.match(a) and a not in seen:
                seen.append(a)
    return seen


def variables(lemma: dict) -> set:
    return {a for l in lemma["body"] + lemma["response"]["removed"] + lemma["response"]["added"] for a in l["args"] if is_var(a)}


def fresh(lemma: dict) -> str:
    used = variables(lemma)
    i = 0
    while f"?v{i}" in used:
        i += 1
    return f"?v{i}"


def _map_args(l: dict, f) -> dict:
    return {"relation": l["relation"], "args": [f(a) for a in l["args"]], "neg": l["neg"]}


def lift(lemma: dict, const: str) -> dict:
    """Every occurrence of the constant becomes one new variable."""
    v = fresh(lemma)
    sub = lambda a: v if a == const else a  # noqa: E731
    return {**lemma, "body": [_map_args(l, sub) for l in lemma["body"]],
            "response": {"removed": [_map_args(l, sub) for l in lemma["response"]["removed"]], "added": [_map_args(l, sub) for l in lemma["response"]["added"]]}}


def drop(lemma: dict, i: int) -> dict:
    return {**lemma, "body": lemma["body"][:i] + lemma["body"][i + 1:]}


def canonical(lemma: dict) -> str:
    """The lemma with its variables numbered by first appearance in a sorted body: two lemmas that differ in variable names only have the same text."""
    def key(l):
        return (l["relation"], l["neg"], [("?" if is_var(a) else a) for a in l["args"]])
    body = sorted(lemma["body"], key=lambda l: json.dumps(key(l)))
    names: dict = {}
    def norm(a):
        if not is_var(a):
            return a
        return names.setdefault(a, f"?{len(names)}")
    out = {"body": [_map_args(l, norm) for l in body],
           "removed": sorted((json.dumps(_map_args(l, norm)) for l in lemma["response"]["removed"])),
           "added": sorted((json.dumps(_map_args(l, norm)) for l in lemma["response"]["added"]))}
    return json.dumps(out, sort_keys=True)


def name_of(lemma: dict) -> str:
    resp = lemma["response"]
    rel = (resp["removed"] or resp["added"])[0]["relation"]
    return f"{rel}-response-{hashlib.sha1(canonical(lemma).encode()).hexdigest()[:8]}"


def size(lemma: dict) -> tuple:
    return (len(lemma["body"]), len(constants(lemma)))


# ---------------------------------------------------------------------------------------------------------------- the seed

def seed_from(card: dict, move: dict, gates: Gates, played: "dict | None" = None) -> dict:
    """The ground lemma of one saving move of a card (both as the engine writes them): its response the move's effect on the facts, its body the facts about what the
    move touches and, when the move that was played is given, about where that move's loss happens (the effect of the capture of our unit along its best line): the
    threat that makes the saving move necessary."""
    facts = [tuple(f) for f in card["facts"]]
    removed = [tuple(f) for f in move["removed"]]
    added = [tuple(f) for f in move["added"]]
    loss = [tuple(f) for f in (played["loss_removed"] + played["loss_added"])] if played else []
    cs = {a for f in removed + added + loss for a in f[1:]}
    total = max(1, len(facts))
    freq = {c: sum(1 for f in facts if c in f[1:]) / total for c in cs}
    strict = {c for c in cs if freq[c] <= gates.strict_frac and not PROTECTED.match(c)}          # (a fact whose arguments are ALL in cs is taken whatever their frequency)
    body = []
    for f in facts:
        args = f[1:]
        if any(a in strict for a in args) or (args and all(a in cs for a in args)):
            body.append(f)
    # the facts that mention a strict constant first, then the others, up to the cap
    body.sort(key=lambda f: (not any(a in strict for a in f[1:]), f))
    body = body[:gates.max_body]
    return {"name": "seed", "body": [lit(f[0], f[1:]) for f in sorted(body)],
            "response": {"removed": [lit(f[0], f[1:]) for f in sorted(removed)], "added": [lit(f[0], f[1:]) for f in sorted(added)]},
            "stats": {}, "meta": {}}


# ---------------------------------------------------------------------------------------------------------------- the climb

@dataclass
class Climb:
    lemma: dict
    ev: dict
    steps: list = field(default_factory=list)
    evals: int = 0


def allowed_refutations(gates: Gates) -> "int | None":
    """How many refuting cards the gate tolerates before a lemma is certainly inadmissible (0 when the gate wants precision 1; otherwise unknown in advance)."""
    return 0 if gates.min_precision >= 1.0 else None


def evaluate(cards, lemma: dict, idx, gates: Gates, early: bool = False) -> dict:
    """The engine's reading of the lemma on the cards at `idx`. With `early`, it stops at the first cut matching or past the refutations the gate allows (the verdict
    'not admissible' is certain; the counts are partial) -- for choosing between candidates, never for the figures a lemma is stored with."""
    return cards.eval(json.dumps(lemma), idx, gates.max_bindings, allowed_refutations(gates) if early else None)


def generalize(seed: dict, cards, idx, gates: Gates, max_steps: int = 400) -> "Climb | None":
    """Hill-climb from the ground seed by single lifts and drops, each step exact-checked on the cards at `idx`. None if the seed itself is not admissible (it is
    the example's own card and must hold there)."""
    cur = seed
    ev = evaluate(cards, cur, idx, gates)
    c = Climb(cur, ev, [], 1)          # (the seed and the chosen steps are read in full: their figures are what the lemma is stored with)
    if not gates.admissible(ev):
        return None
    hard: dict = {}                      # card index -> how many candidates it refuted: the cards that refute most are read first, so a refuted candidate stops early
    for _ in range(max_steps):
        order = sorted(idx, key=lambda i: -hard.get(i, 0)) if hard else list(idx)          # (the order of reading changes the cost of a verdict, never the verdict)
        cands = [("lift " + k, lift(c.lemma, k)) for k in constants(c.lemma)]
        cands += [(f"drop {l['relation']}({', '.join(l['args'])})", drop(c.lemma, i)) for i, l in enumerate(c.lemma["body"]) if len(c.lemma["body"]) > 1 and not l["neg"]]
        best = None
        evs = cards.eval_many([json.dumps(cand) for _, cand in cands], order, gates.max_bindings, allowed_refutations(gates)) if cands else []       # every step read side by side, on the cores
        c.evals += len(cands)
        for (what, cand), e in zip(cands, evs):
            for ci, held, _k in e["cases"]:
                if not held:
                    hard[ci] = hard.get(ci, 0) + 1
            if not gates.admissible(e) or e["fires"] < c.ev["fires"]:
                continue
            gain = (e["fires"] > c.ev["fires"]) or (size(cand) < size(c.lemma))
            if not gain:
                continue
            rank = (e["fires"], -size(cand)[1], -size(cand)[0])          # fires first, then FEWER CONSTANTS (a lift generalizes), then fewer literals
            if best is None or rank > best[0]:
                best = (rank, what, cand, e)
        if best is None:
            break
        _, what, c.lemma, c.ev = best
        c.steps.append(what)
    return c
