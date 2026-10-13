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
    order = {s: i for i, s in enumerate((*H.SCHEMAS, "prepend"))}
    return sorted([(r["term"], r["schema"]) for r in lab["rows"]], key=lambda x: (V.size(V.parse(x[0])), x[0], order[x[1]]))


class Enumerative:
    name = "enumerative"

    def rank(self, hole, lab, cache):
        return candidates(lab)


class Contrast:
    """THE GAMES' PROPOSER (no learning): the candidates in the order of how well they separate the round's positions
    before a win from its draws (models.contrast) -- the terms the drawn games point to are verified, and their verdicts
    teach the learned proposers the contrast's weight."""
    name = "contrast"

    def rank(self, hole, lab, cache):
        cands = candidates(lab)
        key = [M.contrast(hole, t, cache) for t, _ in cands]
        order = sorted(range(len(cands)), key=lambda i: (-key[i][1], -key[i][0], i))
        return [cands[i] for i in order]


class Learned:
    def __init__(self, kind: str, model):
        self.kind, self.model, self.name = kind, model, kind

    def rank(self, hole, lab, cache):
        cands = candidates(lab)
        ctx = np.stack([M.context_features(hole, t, s, lab["boards"], cache) for t, s in cands])
        sc = M.score(self.model, self.kind, [t for t, _ in cands], ctx)
        return [cands[i] for i in np.argsort(-sc, kind="stable")]


FILTERED_BY: dict = {}                            # (term, schema) -> the proposer whose turn put it through the filter


def filter_(rankings: list, budget: int, skip: set = frozenset(), names: list | None = None) -> list:
    """The proposers' tops in turn, no repeats, the memo's verdicts not spending budget -- `budget` candidates. (Which
    proposer put each one through is kept in FILTERED_BY: is each proposer doing its job?)"""
    out, seen = [], set(skip)
    ptr = [0] * len(rankings)
    while len(out) < budget and any(p < len(r) for p, r in zip(ptr, rankings)):
        for i, r in enumerate(rankings):
            while ptr[i] < len(r) and r[ptr[i]] in seen:
                ptr[i] += 1
            if ptr[i] < len(r) and len(out) < budget:
                seen.add(r[ptr[i]]); out.append(r[ptr[i]])
                if names:
                    FILTERED_BY[r[ptr[i]]] = names[i]
                ptr[i] += 1
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
    schemas = tuple(hole.get("schemas") or (H.SCHEMAS if hole["base"] else ("append",)))
    for t in V.enumerate_numbers():
        v = tuple(A.vocab_values(boards, V.sexpr(t)))
        if len(set(v)) > 1 and v not in seen:
            seen.add(v)
            rows += [dict(term=V.sexpr(t), schema=s) for s in schemas]
    return dict(hole=hole["name"], boards=boards, n=len(boards), rows=rows)


def run(hole_name: str, budget: int, kinds=("gnn", "transformer"), epochs: int = 40, hole: dict | None = None,
        defender: str = "self", n_boards: int = 30, batches: int = 1) -> list:
    """The pipeline on one hole: the learned proposers trained on the LABELED holes' verdicts (the store's feedback; the
    hole itself excluded), the enumerative one beside them, the filter, the verifier (each verdict to the store)."""
    import os
    from .verifier import Verifier
    trained = [n for n in H.available() if n != hole_name and os.path.exists(paths.derived(f"hole_{n}_labels.json"))]
    cache: dict = {}
    h = hole or H.load(hole_name)
    boards = h["stuck"][:100]
    lab = labels(hole_name) if os.path.exists(paths.derived(f"hole_{hole_name}_labels.json")) else fresh_candidates(h, boards)
    boards = lab["boards"]                       # a labeled hole's verdicts were made on its labels' boards
    if defender != "self":
        boards = boards[:n_boards]               # a real defender plays whole games with the planner
    from . import proposers as P
    ver = Verifier(defender=defender)
    base = ver.verify(h, None, "none", boards)
    # KEEP PROPOSING UNTIL IT IS BROKEN (Kaveh: "we should keep proposing until we don't fall into that loop again"): in
    # batches of `budget`; after each, the proposers fine-tune on its verdicts and rank again (the failures steer the next
    # proposals); a natural end -- two batches running with no better candidate (a guard of ten batches, said so)
    res, done, stale, batch_no = [], set(), 0, 0
    best = base
    while True:
        batch_no += 1
        rankings, names = [Enumerative().rank(h, lab, cache)], ["enumerative"]
        for k in kinds:
            rankings.append(Learned(k, P.get(k)).rank(h, lab, cache)); names.append(k)   # the newest checkpoint
        if h.get("contrast"):
            rankings.append(Contrast().rank(h, lab, cache)); names.append("contrast")
        kept = filter_(rankings, budget, skip=done, names=names)
        if not kept:
            break
        got = [ver.verify(h, t, s, boards) for t, s in kept]
        done |= set(kept)
        res += got
        top = max(got, key=lambda r: r["score"])
        if top["score"] > best["score"]:
            best, stale = top, 0
        else:
            stale += 1
        if defender != "self":
            P.update()                                  # this batch's verdicts, before the next batch is ranked
        print(f"OBS   batch {batch_no}: {len(kept)} verified, the batch's best {top['score']:.4f}, the hole's best {best['score']:.4f} (the frame's own {base['score']:.4f})")
        # (Kaveh: not "keep proposing until it's broken" -- one batch unless asked; the loop is the rounds)
        if batch_no >= batches or stale >= 2:
            break
    res = sorted(res, key=lambda r: -r["score"])
    def show(r):
        if "ranks" in r:
            why = ", ".join(f"{k} {v}" for k, v in sorted(r.get("why", {}).items(), key=lambda kv: -kv[1]))
            return (f"held {r['n'] - r['lost']}/{r['n']}, won {r['mated']}, lemma-states {r.get('states', 0) / max(r['n'], 1):.1f} a board, "
                    f"claims kept {r['certified']}/{r['played']}{f'; ground lost: {why}' if why else ''}")
        return f"mated {r['mated']}/{r['n']}, certified {r['certified']}/{r['played']}"
    print(f"OBS pipeline on {hole_name} ({len(boards)} boards; {len(lab['rows'])} candidates; defender {defender}): {len(rankings)} proposers, the filter "
          f"kept {len(res)} over {batch_no} batches, all verified; the hole's own measure scores {base['score']:.4f} ({show(base)}); the best:")
    for r in res[:5]:
        print(f"OBS   {r['score']:.4f} ({show(r)}) [{r['schema']}] {r['term']} -- from {FILTERED_BY.get((r['term'], r['schema']), '?')} = {V.gloss(V.parse(r['term']))}")
    from collections import Counter
    top = [FILTERED_BY.get((r["term"], r["schema"]), "?") for r in res[:10]]
    print(f"OBS   the verifier's top 10 by proposer: {dict(Counter(top))}")
    if defender == "self":
        P.update_in_background()                # this hole's verdicts fine-tune the proposers beside the next work
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
