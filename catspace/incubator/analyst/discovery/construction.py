"""PLAN 78.38 stage 1 LEARNING BY CONSTRUCTION (the ladder): start from the units a game cannot do without, observe what results exist at that unit set, add one unit, observe again.

GENERIC: a position is an opaque string the kernel accepts; kinds and sides are read through the `PositionCodec` of curriculum.py (`describe`, `unit_term`, `opposed_term`), the labels come
from an exact labeller (exact.py, training kit only), the words from word_registry (a SANDBOX store). No unit kind, side or relation of any game is named here.

A UNIT SET is canonical and hashable: (holder, other), each a sorted tuple of (kind, count); the two sides are put in a fixed order (the side with more units first, then by the tuples), so a set
and the same set with the sides swapped are one set. A side to move is `holder` (it holds the first part), `other`, or `even` (both parts equal).

  signature(game, position) -> unit set               sides_of(game, position) -> (holder side, other side)        size_of(unit_set), show(unit_set)
  ladder(game, max_size)    -> [unit set ..] by size, from the required units alone, each larger set one unit more (of the kinds and sides the initial position holds)
  sample(game, unit_set, n, seed) -> positions of that set, legal and not over (random walks from sub-positions of the initial position that keep the set; a walk stops where the set changes)
  observe(game, unit_set, positions, labeller) -> summary {counts, shares of win / draw / loss by the mover's role, decisive_exists, covered}
  falls_to(game, unit_set, positions) -> {other unit set: number of legal moves that reach it}, by playing the moves through the kernel
  set_word_term(game, unit_set) / define_set_word(game, unit_set, vocabulary) -> the word that says "the units on the board are exactly this set", defined in the (sandbox) store
  set_claims(game, summary, bar, vocabulary=..) -> claim rules (78.36 format) for a role where one class holds in at least `bar` of the positions
  drill_set(game, unit_set, positions, claims, labeller, vocabulary) -> per claim {right, wrong, exceptions}; open_order(ladder, falls) -> the order sets are opened in

  python -m catspace.incubator.analyst.discovery.construction <game> --max-size 3 [--n 2000] [--seed 1] [--bar 0.98] [--store init_store_c38_x] [--vocabulary keys]
"""
from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
from collections import Counter
from pathlib import Path

import cs_engine as CE

from catspace.incubator.analyst.discovery import corpus as CP
from catspace.incubator.analyst.discovery import curriculum as CU
from catspace.incubator.analyst.discovery import exact as EX
from catspace.incubator.analyst.discovery import game_store as GS
from catspace.incubator.analyst.discovery import word_registry as WR
from catspace.incubator.analyst.discovery import word_terms as WT
from catspace.io import paths

CLASSES = ("win", "draw", "loss")
ROLES = ("holder", "other", "even")
MAX_COUNT = 3                  # the most units of one kind and side a word counts (the pairwise-distinct expansion grows as 2^pairs)
STORE_PREFIX = "init_store_c38"


# ------------------------------------------------------------------------------------------------------------------------------------ unit sets

def _parts(raw: dict) -> dict:
    """{side: ((kind, count) ..)} from {side: {kind: count}}."""
    return {s: tuple(sorted((k, n) for k, n in kinds.items() if n > 0)) for s, kinds in raw.items()}


def _raw(game: str, position: str) -> dict:
    out: dict = {}
    for side, kind in CU.codec_of(game).describe(position):
        out.setdefault(side, Counter())[kind] += 1
    return out


def _canonical(parts: dict) -> tuple:
    """(holder part, other part) of {side: part}: the side with more units first, then the larger tuple."""
    a, b = list(parts.values()) + [()] * (2 - len(parts))
    key = lambda p: (sum(n for _k, n in p), p)  # noqa: E731
    return (a, b) if key(a) >= key(b) else (b, a)


def signature(game: str, position: str) -> tuple:
    """The unit set of a position: (holder part, other part), each a sorted tuple of (kind, count)."""
    return _canonical(_parts(_raw(game, position)))


def sides_of(game: str, position: str) -> tuple:
    """(holder side, other side) of the position (an even set: the sides in sorted order)."""
    parts = _parts(_raw(game, position))
    sides = sorted(parts) + [None] * (2 - len(parts))
    hold = _canonical(parts)[0]
    first = next((s for s in sides if s is not None and parts[s] == hold), sides[0])
    return first, next(s for s in sides if s != first)


def size_of(unit_set: tuple) -> int:
    return sum(n for part in unit_set for _k, n in part)


def show(unit_set: tuple) -> str:
    return " v ".join("+".join("+".join([k] * n) for k, n in part) or "nothing" for part in unit_set)


def role_of(game: str, position: str, mover: str) -> str:
    """The mover's role at the position: holder, other, or even (both sides hold the same)."""
    unit_set = signature(game, position)
    if unit_set[0] == unit_set[1]:
        return "even"
    return "holder" if sides_of(game, position)[0] == mover else "other"


def ladder(game: str, max_size: int, *, variant: "str | None" = None) -> list:
    """The unit sets in order of size from the required units alone; each larger set is a smaller one plus one unit of some kind and side that the initial position holds; sets that are
    the same with the sides swapped are one."""
    variant = variant or game
    codec = CU.codec_of(game)
    start = CU.initial(variant)
    units, desc = codec.units(start), codec.describe(start)
    cap = Counter(desc)
    req = set(CU.required(game, variant))
    base = Counter(d for u, d in zip(units, desc) if u in req)
    level = {_canonical(_parts(_regroup(base)))}
    seen, out = set(level), [*level]
    frontier = [base]
    for _ in range(max_size - sum(base.values())):
        nxt = []
        for have in frontier:
            for sk in sorted(cap):
                if have[sk] < cap[sk]:
                    bigger = Counter(have)
                    bigger[sk] += 1
                    c = _canonical(_parts(_regroup(bigger)))
                    if c not in seen:
                        seen.add(c)
                        out.append(c)
                        nxt.append(bigger)
        frontier = nxt
    return sorted(out, key=lambda u: (size_of(u), u))


def _regroup(by_side_kind: Counter) -> dict:
    raw: dict = {}
    for (side, kind), n in by_side_kind.items():
        raw.setdefault(side, Counter())[kind] += n
    return raw


# ------------------------------------------------------------------------------------------------------------------------------------ sample

def sample(game: str, unit_set: tuple, n: int, seed: int, *, variant: "str | None" = None, walk_plies: int = 24, patience: int = 200) -> list:
    """Up to `n` distinct positions (by the kernel's key) of the unit set, each legal and not over: sub-positions of the initial position that keep exactly the set (the units drawn at random,
    the sides in either order), then random legal play from them, every position along the walk (the walk stops where the set changes)."""
    variant = variant or game
    codec = CU.codec_of(game)
    start = CU.initial(variant)
    units, desc = codec.units(start), codec.describe(start)
    sides = sorted({s for s, _k in desc})
    orders = [tuple(sides[:2])] if unit_set[0] == unit_set[1] else [tuple(sides[:2]), tuple(reversed(sides[:2]))]
    by_sk: dict = {}
    for u, d in zip(units, desc):
        by_sk.setdefault(d, []).append(u)
    rng = random.Random(seed)
    out, keys, stale = [], set(), 0

    def add(fen: str) -> bool:
        _f, mover, legal, outcome, key = CE.state(variant, fen, [])
        if mover is None or outcome is not None or not legal or key in keys:
            return False
        keys.add(key)
        out.append(fen)
        return True

    while len(out) < n and stale < patience:
        before = len(out)
        order = rng.choice(orders)
        chosen = []
        for side, part in zip(order, unit_set):
            for kind, count in part:
                pool = by_sk.get((side, kind), [])
                if len(pool) < count:
                    chosen = None
                    break
                chosen += rng.sample(pool, count)
            if chosen is None:
                break
        if chosen is not None:
            fen = codec.keep(start, chosen)
            if CU.check(variant, fen) is None:
                add(fen)
                for _ in range(rng.randint(1, walk_plies)):
                    _f, mover, legal, outcome, _k = CE.state(variant, fen, [])
                    if mover is None or outcome is not None or not legal:
                        break
                    fen = CE.state(variant, fen, [rng.choice(sorted(legal))])[0]
                    if signature(game, fen) != unit_set:
                        break
                    if len(out) >= n:
                        break
                    add(fen)
        stale = stale + 1 if len(out) == before else 0
    return out[:n]


# ------------------------------------------------------------------------------------------------------------------------------------ observe, falls

def observe(game: str, unit_set: tuple, positions: list, labeller, *, variant: "str | None" = None) -> dict:
    """The summary of a unit set: {set, size, positions, labelled, skipped, covered (the labeller labelled at least one), by_mover {role: {n, counts, shares}}, decisive_exists}.
    The label is the exact class for the side to move; a position the labeller does not cover is skipped."""
    variant = variant or game
    tally = {r: Counter() for r in ROLES}
    skipped = 0
    for pos in positions:
        mover = CE.state(variant, pos, [])[1]
        try:
            cls = labeller.result(variant, pos, [])
        except RecursionError:                       # the labeller's search through a state it cannot close (exact.py): the position counts as not covered
            cls = None
        if cls is None:
            skipped += 1
            continue
        tally[role_of(game, pos, mover)][cls] += 1
    by = {}
    for r, c in tally.items():
        n = sum(c.values())
        if n:
            by[r] = {"n": n, "counts": {k: c[k] for k in CLASSES}, "shares": {k: c[k] / n for k in CLASSES}}
    labelled = sum(v["n"] for v in by.values())
    return {"set": unit_set, "size": size_of(unit_set), "positions": len(positions), "labelled": labelled, "skipped": skipped, "covered": labelled > 0, "by_mover": by,
            "decisive_exists": any(v["counts"][k] for v in by.values() for k in CLASSES if k != "draw")}


def falls_to(game: str, unit_set: tuple, positions: list, *, variant: "str | None" = None) -> dict:
    """{other unit set: number of legal moves, over the positions, that reach it}: each legal move played through the kernel, the signature of the position it gives."""
    variant = variant or game
    out: Counter = Counter()
    for pos in positions:
        for m in CE.state(variant, pos, [])[2]:
            after = CE.state(variant, pos, [m])[0]
            s = signature(game, after)
            if s != unit_set:
                out[s] += 1
    return dict(out)


def open_order(lad: list, falls: dict) -> list:
    """The sets of `lad` in the order they are opened: a set after every set of the ladder it falls to (`falls` {set: {target: n}}); ties by size then the set; a cycle is broken at the smallest."""
    inside = set(lad)
    need = {u: {t for t in falls.get(u, {}) if t in inside and t != u} for u in lad}
    done, order = set(), []
    rest = sorted(lad, key=lambda u: (size_of(u), u))
    while rest:
        ready = [u for u in rest if need[u] <= done]
        pick = ready[0] if ready else rest[0]
        order.append(pick)
        done.add(pick)
        rest.remove(pick)
    return order


# ------------------------------------------------------------------------------------------------------------------------------------ the word and the claims

def _at_least(codec, side: str, kind: str, n: int, tag: str) -> tuple:
    """(term text, all place variables): n distinct places holding a unit of the side and kind (pairwise distinct: some place slot differs, as an `or` of `and`s)."""
    lits = [codec.unit_term(side, kind, f"{tag}x{i}") for i in range(n)]
    places = [v for _t, vs in lits for v in vs]
    if n == 1:
        return lits[0][0], places
    pairs = list(itertools.combinations(range(n), 2))
    m = len(lits[0][1])
    branches = []
    for pick in itertools.product(range(m), repeat=len(pairs)):
        ds = [f"(distinct {lits[i][1][k]} {lits[j][1][k]})" for (i, j), k in zip(pairs, pick)]
        branches.append("(and " + " ".join([t for t, _v in lits] + ds) + ")")
    return "(or " + " ".join(branches) + ")", places


def _side_block(codec, side: str, part: tuple, kinds: list, tag: str) -> list:
    held = dict(part)
    items = []
    for ki, kind in enumerate(kinds):
        c = held.get(kind, 0)
        if c > MAX_COUNT:
            raise ValueError(f"a count of {c} {kind} is past what a word counts ({MAX_COUNT})")
        if c:
            have, places = _at_least(codec, side, kind, c, f"{tag}{ki}p")
            items.append(f"(exists ({' '.join(places)}) {have})")
        over, places = _at_least(codec, side, kind, c + 1, f"{tag}{ki}q")
        items.append(f"(not (exists ({' '.join(places)}) {over}))")
    return items


def set_word_term(game: str, unit_set: tuple, *, variant: "str | None" = None) -> str:
    """The term (word_terms language) of "side ?s holds exactly the first part of the set and its opponent exactly the second": for every kind the initial position holds, at least the count and
    not one more (`not exists`), the counts built from the unit literal, `distinct` and `or`; the other side is an `exists`. Uses only the term language's constructors."""
    variant = variant or game
    codec = CU.codec_of(game)
    kinds = sorted({k for _s, k in codec.describe(CU.initial(variant))})
    items = [codec.opposed_term("?s", "?o")] + _side_block(codec, "?s", unit_set[0], kinds, "a") + _side_block(codec, "?o", unit_set[1], kinds, "b")
    return "(exists (?o) (and " + " ".join(items) + "))"


def set_word_name(unit_set: tuple) -> str:
    return "exactly_" + "_against_".join("_".join("_".join([k] * n) for k, n in part) or "nothing" for part in unit_set)


def define_set_word(game: str, unit_set: tuple, vocabulary: str, *, variant: "str | None" = None) -> dict:
    """Defines the word of the set in the store GS.LEARNED_ROOT points at (a SANDBOX store: the caller's duty) and returns the registry entry."""
    name = set_word_name(unit_set)
    term = set_word_term(game, unit_set, variant=variant)
    for e in WR.load(game)["words"]:
        if e["name"] == name and e["term"] == WT.show(WT.parse(term)):          # defined already (the same term): the same word
            return e
    note = f"side {{0}} holds exactly {show(unit_set).split(' v ')[0]} and its opponent exactly {show(unit_set).split(' v ')[1]}, and no other unit stands on the board"
    return WR.define(game, term, origin="hand", name=name, note=note.replace("{0}", "?s"), vocabulary=vocabulary)


def set_claims(game: str, summary: dict, bar: float, *, vocabulary: str, min_n: int = 20, variant: "str | None" = None) -> list:
    """For each role of the summary's side to move where one class has a share of at least `bar` (over at least `min_n` positions), a CLAIM rule (78.36 format) whose body is the set's word
    applied to the side that holds the first part (us when the mover holds it or the sets are even, them when the mover is the other side)."""
    claims = []
    for role, v in sorted(summary["by_mover"].items()):
        if v["n"] < min_n:
            continue
        for cls in CLASSES:
            if v["shares"][cls] >= bar:
                word = define_set_word(game, summary["set"], vocabulary, variant=variant)["name"]
                claims.append({"name": f"set-{word.replace('_', '-')}-{role}-{cls}", "body": [{"relation": word, "args": ["them" if role == "other" else "us"], "neg": False}],
                               "claim": {"class": cls}, "stats": {}, "meta": {"vocabulary": vocabulary, "origin": "construction", "set": show(summary["set"]), "mover": role, "bar": bar}})
    return claims


def drill_set(game: str, unit_set: tuple, positions: list, claims: list, labeller, vocabulary: str, *, variant: "str | None" = None) -> list:
    """Each claim drilled (corpus.drill_cases) on the set's labelled positions: [{name, claim, binds, right, wrong, exceptions: [{position, exact_class}]}]; the exceptions are the next problems."""
    variant = variant or game
    rows = []
    for i, pos in enumerate(positions):
        try:
            cls = labeller.result(variant, pos, [])
        except RecursionError:
            cls = None
        if cls is None:
            continue
        legal = CE.state(variant, pos, [])[2]
        rows.append({"start": pos, "moves": [], "group": "g0", "before": cls, "keep": {m: True for m in legal}})
    if not rows or not claims:
        return [{"name": c["name"], "claim": c["claim"]["class"], "binds": 0, "right": 0, "wrong": 0, "exceptions": []} for c in claims]
    corpus = CP.Corpus(rows, {"g0": "dev"}, game=game, variant=variant, size=size_of(unit_set))
    corpus.cards(GS.spec_texts(game, vocabulary), json.dumps(GS.vocabulary(game, vocabulary)))
    out = []
    for c in claims:
        t = CP.drill_cases(c, corpus, part="all")
        ex = [{"position": corpus.rows[corpus.row_of[ci]]["start"], "exact_class": corpus.rows[corpus.row_of[ci]]["before"]} for ci in t["wrong_at"]]
        out.append({"name": c["name"], "claim": c["claim"]["class"], "binds": t["binds"], "right": t["right"], "wrong": t["wrong"], "exceptions": ex})
    return out


# ------------------------------------------------------------------------------------------------------------------------------------ the sandbox store and the CLI

def sandbox_store(game: str, vocabulary: str, name: str) -> Path:
    """A sandbox store root `paths.derived(name)` (the name starts with init_store_c38) holding a copy of the vocabulary record with `words` on; GS.LEARNED_ROOT points at it."""
    if not name.startswith(STORE_PREFIX):
        raise ValueError(f"a sandbox store's name starts with {STORE_PREFIX}")
    root = Path(paths.derived(name))
    d = root / "learned" / game
    d.mkdir(parents=True, exist_ok=True)
    rec = json.loads((GS.RULES / "learned" / game / f"vocabulary.{vocabulary}.json").read_text())
    rec["words"] = True
    (d / f"vocabulary.{vocabulary}.json").write_text(json.dumps(rec))
    GS.LEARNED_ROOT = root / "learned"
    GS.SNAPSHOT_ROOT = root / "snapshots"
    return root


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def main(argv: list) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("game")
    ap.add_argument("--max-size", type=int, required=True)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--bar", type=float, default=0.98)
    ap.add_argument("--store", default=None, help=f"sandbox store name under data/derived, starting with {STORE_PREFIX} (default {STORE_PREFIX}_<game>_<seed>)")
    ap.add_argument("--vocabulary", default="keys")
    a = ap.parse_args(argv)
    root = sandbox_store(a.game, a.vocabulary, a.store or f"{STORE_PREFIX}_{a.game}_{a.seed}")
    lab = EX.labeller_of(a.game)
    lad = ladder(a.game, a.max_size)
    summaries, falls, samples = {}, {}, {}
    try:
        for u in lad:
            samples[u] = sample(a.game, u, a.n, a.seed)
            summaries[u] = observe(a.game, u, samples[u], lab)
            falls[u] = falls_to(a.game, u, samples[u])
        print(f"OBS ladder {a.game}: {len(lad)} sets up to size {a.max_size}, store {root}")
        for u in open_order(lad, falls):
            s = summaries[u]
            line = f"OBS set [{show(u)}] size {s['size']}: positions {s['positions']}, labelled {s['labelled']}, skipped {s['skipped']}"
            if not s["positions"]:
                print(line + ", no live position (every sub-position of the set is over or refused by the kernel)")
                print(f"OBS   [{show(u)}] falls to {_falls_text(falls[u])}")
                continue
            if not s["covered"]:
                print(line + ", NOT COVERED by the labeller: skipped")
                print(f"OBS   [{show(u)}] falls to {_falls_text(falls[u])}")
                continue
            by = "; ".join(f"mover {r} n {v['n']} " + " ".join(f"{k} {pct(v['shares'][k])}" for k in CLASSES) for r, v in sorted(s["by_mover"].items()))
            claims = set_claims(a.game, s, a.bar, vocabulary=a.vocabulary)
            drills = drill_set(a.game, u, samples[u], claims, lab, a.vocabulary)
            print(f"{line}; {by}; decisive result exists: {s['decisive_exists']}")
            print(f"OBS   [{show(u)}] falls to {_falls_text(falls[u])}")
            for c, d in zip(claims, drills):
                print(f"OBS   [{show(u)}] claim {c['name']} ({d['claim']}, mover {c['meta']['mover']}): binds {d['binds']}, right {d['right']}, wrong {d['wrong']} (exceptions kept: {len(d['exceptions'])})")
            if not claims:
                print(f"OBS   [{show(u)}] no claim at bar {a.bar}")
    finally:
        lab.close()
    return 0


def _falls_text(f: dict) -> str:
    return ", ".join(f"[{show(t)}] x{n}" for t, n in sorted(f.items(), key=lambda kv: (size_of(kv[0]), kv[0]))) or "nothing"


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
