"""PROPOSERS -> FILTER -> VERIFIER (Kaveh 2026-10-12: "we want a proposer or multiple proposers, a filter, and then ... the
verifier for everything that we filtered"; "how do you give feedback back to the proposer?").

  PROPOSERS  each ranks the hole's candidates (terms over the base x schemas): ENUMERATIVE (smaller first -- no learning),
             and LEARNED (models.py: graph network, transformer, context-only) trained on the experience store
  FILTER     merges the proposers' rankings into a budget -- their tops in turn, no repeats, and nothing the store has
             already verified for this hole spends budget (the memo)
  VERIFIER   verifier.py: both stages on every candidate the filter kept; each verdict, good or bad, goes to the store --
             the FEEDBACK the learned proposers retrain on

  python -m catspace.incubator.analyst.discovery.pipeline --evaluate          leave-one-hole-out: verifier calls to a near-best
  python -m catspace.incubator.analyst.discovery.pipeline --run kr_derived --budget 30
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from catspace.io import paths

from . import base_vocabulary as V
from . import holes as H
from . import models as M


def labels(name: str) -> dict:
    return json.load(open(paths.derived(f"hole_{name}_labels.json")))


def candidates(lab: dict) -> list:
    """(term, schema) of the hole, in the enumeration order (smaller first, then the schema order)."""
    order = {s: i for i, s in enumerate(H.SCHEMAS)}
    return sorted([(r["term"], r["schema"]) for r in lab["rows"]], key=lambda x: (V.size(V.parse(x[0])), x[0], order[x[1]]))


class Enumerative:
    name = "enumerative"

    def rank(self, hole, lab, cache):
        return candidates(lab)


class Learned:
    def __init__(self, kind: str, model):
        self.kind, self.model, self.name = kind, model, kind

    def rank(self, hole, lab, cache):
        cands = candidates(lab)
        ctx = np.stack([M.context_features(hole, t, s, lab["boards"], cache) for t, s in cands])
        sc = M.score(self.model, self.kind, [t for t, _ in cands], ctx)
        return [cands[i] for i in np.argsort(-sc, kind="stable")]


def filter_(rankings: list, budget: int, skip: set = frozenset()) -> list:
    """The proposers' tops in turn, no repeats, the memo's verdicts not spending budget -- `budget` candidates."""
    out, seen = [], set(skip)
    ptr = [0] * len(rankings)
    while len(out) < budget and any(p < len(r) for p, r in zip(ptr, rankings)):
        for i, r in enumerate(rankings):
            while ptr[i] < len(r) and r[ptr[i]] in seen:
                ptr[i] += 1
            if ptr[i] < len(r) and len(out) < budget:
                seen.add(r[ptr[i]]); out.append(r[ptr[i]]); ptr[i] += 1
    return out


def train_on(holes: list, kind: str, cache: dict, epochs: int = 40):
    rows, ctx = [], []
    for name in holes:
        h, lab = H.load(name), labels(name)
        for r in lab["rows"]:
            rows.append(dict(r, hole=name))
            ctx.append(M.context_features(h, r["term"], r["schema"], lab["boards"], cache))
    return M.train(kind, rows, np.stack(ctx), epochs=epochs)


def evaluate(kinds=("context", "gnn", "transformer"), epochs: int = 40) -> None:
    """LEAVE ONE HOLE OUT: trained on the other holes' verdicts, how many verifier calls until a candidate within 5% of the
    hole's best (and is the best found within 10 / 30 / 100 calls)?"""
    names = H.available()
    cache: dict = {}
    t0 = time.time()
    table = {k: [] for k in ("enumerative", *kinds, "filter(gnn+transformer)")}
    for held in names:
        h, lab = H.load(held), labels(held)
        truth = {(r["term"], r["schema"]): r["score"] for r in lab["rows"]}
        best = max(truth.values())
        good = {c for c, s in truth.items() if s >= 0.95 * best}
        ranks = {"enumerative": Enumerative().rank(h, lab, cache)}
        for k in kinds:
            m = train_on([n for n in names if n != held], k, cache, epochs)
            ranks[k] = Learned(k, m).rank(h, lab, cache)
        if "gnn" in ranks and "transformer" in ranks:
            ranks["filter(gnn+transformer)"] = filter_([ranks["gnn"], ranks["transformer"]], len(truth))
        for k, r in ranks.items():
            first = next(i for i, c in enumerate(r) if c in good) + 1
            table[k].append((held, first, len(r)))
        print(f"OBS   held out {held} (best {best:.3f}, {len(good)} near-best of {len(truth)}): " +
              ", ".join(f"{k} {next(x[1] for x in table[k] if x[0] == held)}" for k in table if any(x[0] == held for x in table[k])) +
              f" ({time.time() - t0:.0f} s)")
    print(f"OBS leave-one-hole-out over {len(names)} holes -- verifier calls until a near-best candidate (median / mean / worst):")
    for k, v in table.items():
        if v:
            c = [x[1] for x in v]
            print(f"OBS   {k:24s} {int(np.median(c)):5d} / {np.mean(c):7.1f} / {max(c):5d}   within 10: {sum(x <= 10 for x in c)}, 30: {sum(x <= 30 for x in c)}, 100: {sum(x <= 100 for x in c)} of {len(c)}")


def fresh_candidates(hole: dict, boards: list) -> dict:
    """A hole with no labels: its candidates are the enumeration, merged by observational equivalence on its boards."""
    import analyst_core as A
    seen, rows = set(), []
    schemas = H.SCHEMAS if hole["base"] else ("append",)
    for t in V.enumerate_numbers():
        v = tuple(A.vocab_values(boards, V.sexpr(t)))
        if len(set(v)) > 1 and v not in seen:
            seen.add(v)
            rows += [dict(term=V.sexpr(t), schema=s) for s in schemas]
    return dict(hole=hole["name"], boards=boards, n=len(boards), rows=rows)


def run(hole_name: str, budget: int, kinds=("gnn", "transformer"), epochs: int = 40, hole: dict | None = None) -> list:
    """The pipeline on one hole: the learned proposers trained on the LABELED holes' verdicts (the store's feedback; the
    hole itself excluded), the enumerative one beside them, the filter, the verifier (each verdict to the store)."""
    import os
    from .verifier import Verifier
    trained = [n for n in H.available() if n != hole_name and os.path.exists(paths.derived(f"hole_{n}_labels.json"))]
    cache: dict = {}
    h = hole or H.load(hole_name)
    boards = h["stuck"][:100]
    lab = labels(hole_name) if os.path.exists(paths.derived(f"hole_{hole_name}_labels.json")) else fresh_candidates(h, boards)
    rankings = [Enumerative().rank(h, lab, cache)]
    for k in kinds:
        rankings.append(Learned(k, train_on(trained, k, cache, epochs)).rank(h, lab, cache))
    kept = filter_(rankings, budget)
    ver = Verifier()
    base = ver.verify(h, None, "none", boards)
    res = sorted((ver.verify(h, t, s, boards) for t, s in kept), key=lambda r: -r["score"])
    print(f"OBS pipeline on {hole_name} ({len(boards)} boards; {len(lab['rows'])} candidates): {len(rankings)} proposers, the filter "
          f"kept {len(kept)}, all verified; the hole's own measure scores {base['score']:.3f} (mated {base['mated']}, "
          f"certified {base['certified']}/{base['played']}); the best:")
    for r in res[:5]:
        print(f"OBS   {r['score']:.3f} (mated {r['mated']}/{r['n']}, certified {r['certified']}/{r['played']}) [{r['schema']}] {r['term']} = {V.gloss(V.parse(r['term']))}")
    return [base] + res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--evaluate", action="store_true")
    ap.add_argument("--run", default=None)
    ap.add_argument("--budget", type=int, default=30)
    ap.add_argument("--epochs", type=int, default=40)
    a = ap.parse_args(argv)
    if a.evaluate:
        evaluate(epochs=a.epochs)
    if a.run:
        run(a.run, a.budget, epochs=a.epochs)


if __name__ == "__main__":
    main()
