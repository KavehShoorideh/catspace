"""THE SYMBOLIC PROPOSER (Kaveh 2026-10-12: "the important thing to give it is the vocabulary ... the base has to be
deterministic and interpretable"; build it). It fills an operator's HOLE -- here the MONOTONE hole: a number that, added
to a frame's measure, certifies the positions the frame is stuck on -- with TERMS over the base vocabulary (Lean
Ideas.Vocabulary, evaluated by rust/analyst_core/src/vocab.rs), never with a finished concept:

  1 ENUMERATE typed terms, smallest first (squares W, L, P; sets: edge, corner, attacked, free, box, rank, file, line,
    between, nearer, reach; and / not; numbers: count, dist) -- QuickSpec's way (Claessen, Smallbone, Hughes)
  2 MERGE terms that agree on every sampled board (keep the smaller) -- observational equivalence; drop constants
  3 SCORE each by how many stuck positions the measure extended by it certifies (three schemas: a new last component;
    added to the frame's last component; in its place), against every reply
  4 READ the winners out in the base's own words -- the term IS the explanation

  python -m catspace.incubator.analyst.discovery.enumerative --stuck kr_stuck_derived.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import random
import time

from catspace.io import paths

SQ = {"W": "our king", "L": "their king", "P": "our rook"}
ATOMS = {"edge": "the edge squares", "corner": "the corner squares", "attacked": "the squares we attack",
         "free": "the squares we do not attack"}


def gloss(t) -> str:
    """The term read out in the base's words (deterministic)."""
    if isinstance(t, str):
        return SQ.get(t) or ATOMS.get(t) or t
    h, *a = t
    g = [gloss(x) for x in a]
    return {
        "box": lambda: f"the box around {g[0]}", "rank": lambda: f"the rank of {g[0]}", "file": lambda: f"the file of {g[0]}",
        "line": lambda: f"the line through {g[0]} and {g[1]}", "between": lambda: f"the squares between {g[0]} and {g[1]}",
        "nearer": lambda: f"the squares nearer {g[0]} than {g[1]}", "and": lambda: f"{g[0]} that are also {g[1]}",
        "not": lambda: f"the squares outside {g[0]}", "reach": lambda: f"the squares {g[0]} can walk to over {g[1]}",
        "count": lambda: f"how many of {g[0]}", "dist": lambda: f"the king-step distance from {g[0]} to {g[1]}",
    }[h]()


def sexpr(t) -> str:
    return t if isinstance(t, str) else "(" + " ".join(sexpr(x) for x in t) + ")"


def size(t) -> int:
    return 1 if isinstance(t, str) else 1 + sum(size(x) for x in t[1:])


def enumerate_terms():
    """Number terms, smallest first: count of a set built from at most two base sets, or a distance."""
    sqs = list(SQ)
    base = [a for a in ATOMS]
    base += [(h, s) for h in ("box", "rank", "file") for s in sqs]
    base += [(h, a, b) for h in ("line", "between", "nearer") for a in sqs for b in sqs if a != b]
    base += [("reach", s, a) for s in sqs for a in ("free",)]
    sets = list(base) + [("not", a) for a in base]
    sets += [("and", a, b) for a, b in itertools.combinations(base, 2)]
    sets += [("and", a, ("not", b)) for a in base for b in base if a != b]
    nums = [("count", s) for s in sets] + [("dist", a, b) for a, b in itertools.combinations(sqs, 2)]
    nums.sort(key=size)
    return nums


def propose(stuck: str, sample: int, k: int, top: int) -> None:
    import analyst_core as A
    t0 = time.time()
    fens = json.load(open(paths.derived(stuck)))
    rng = random.Random(0)
    probe = rng.sample(fens, min(sample, len(fens)))
    terms = enumerate_terms()
    # 2 OBSERVATIONAL EQUIVALENCE on the probe boards: one representative (the smallest) per value vector; no constants
    seen, reps = set(), []
    for t in terms:
        v = tuple(A.vocab_values(probe, sexpr(t)))
        if len(set(v)) <= 1 or v in seen:
            continue
        seen.add(v)
        reps.append(t)
    base_score = A.vocab_certify(probe, "(count edge)", "none", k)
    print(f"OBS proposer: hole MONOTONE on {stuck} ({len(fens)} stuck positions; probing {len(probe)}); {len(terms)} terms "
          f"enumerated, {len(reps)} observationally distinct and not constant ({time.time() - t0:.0f} s); the frame alone "
          f"certifies {base_score} of {len(probe)}")
    # 3 SCORE on the probe, every schema, k moves against every reply
    scored = []
    for t in reps:
        for sch in ("append", "sum", "replace"):
            scored.append((A.vocab_certify(probe, sexpr(t), sch, k), -size(t), sch, t))
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    print(f"OBS   scored {len(scored)} (term, schema) pairs in {time.time() - t0:.0f} s; the best, re-scored on all {len(fens)}:")
    shown = 0
    for n, neg, sch, t in scored:
        if shown >= top:
            break
        full = A.vocab_certify(fens, sexpr(t), sch, k)
        shown += 1
        print(f"OBS   {full:4d} / {len(fens)} ({100 * full / len(fens):.1f}%)  [{sch}] size {-neg}  {sexpr(t)}")
        print(f"OBS        = {gloss(t)}")
    print(f"OBS   time {time.time() - t0:.0f} s")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stuck", default="kr_stuck_derived.json")
    ap.add_argument("--sample", type=int, default=120)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--top", type=int, default=8)
    a = ap.parse_args(argv)
    propose(a.stuck, a.sample, a.k, a.top)


if __name__ == "__main__":
    main()
