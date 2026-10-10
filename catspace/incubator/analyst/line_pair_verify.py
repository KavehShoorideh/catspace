"""THE LINE-PAIR LEMMA, verified on the census before it is trusted (Kaveh 2026-10-09: search as the fallback, extract a
lemma from what the search found, keep it only if it holds wherever its pattern matches).

The lemma (rust/analyst_core/src/plan.rs line_pair_lemma): under their slider, with our front and back pieces unable to
cover each other after moving (the colour lemmas), the cheapest way to lose is VALUE ARITHMETIC over the two pieces'
moves -- what the move captures minus the most they can then take. For every census position and every line pair of the
side to move, the lemma's net is compared with alpha-beta in the frame (frame_fallback: the value search over the
frame's first moves, every reply of theirs searched). Accept only on full agreement; every disagreement is listed.
"""
from __future__ import annotations

import os
from collections import Counter


def _one(fen):
    os.environ.setdefault("RAYON_NUM_THREADS", "1")
    import analyst_core
    # where the PLANNER uses the lemma: its defence phase found no defence and the frame is a line pair
    dec = analyst_core.plan_trace(fen)[1]
    used = bool(dec and dec[0] == "cheapest" and "line-pair lemma" in dec[3])
    return fen, analyst_core.line_pair_check(fen), used


def main():
    import time
    from multiprocessing import Pool
    from .tactic_census_measure import load
    fens = [r["fen"] for r in load("c300")]
    t0 = time.time()
    with Pool(10) as p:
        res = list(p.imap_unordered(_one, fens, chunksize=64))
    inst = [(fen, x) for fen, xs, used in res for x in xs]
    used = [(fen, x) for fen, xs, u in res if u for x in xs]
    print(f"OBS positions where the planner decided by the lemma (no defence, a line pair): {sum(1 for _, _, u in res if u)}")
    for label, group in (("all pattern matches (front and back winnable, no cover)", inst), ("where the planner used it (also: no defence)", used)):
        report(label, group)
    # THE NARROWER CLAIM: the lemma prices the pair's own moves (front and back) -- against the search over those moves
    v = Counter(); ex = []
    for fen, (sl, f, r, lemma, fb) in inst:
        pair = [x for x in fb if x[0].startswith("pair:")]
        if not pair or lemma is None:
            v["no comparison"] += 1; continue
        pv = pair[0][1]
        k = "agree" if lemma[1] == pv else ("search found more among the pair's moves" if pv > lemma[1] else "lemma promised more")
        v[k] += 1
        if k != "agree" and len(ex) < 6:
            ex.append((k, lemma, pair[0], fen))
    print(f"OBS THE PAIR'S OWN MOVES, lemma against the frame search over the same moves: {dict(v)}")
    for k, lemma, pb, fen in ex:
        print(f"OBS   {k}: lemma {lemma[0]} {lemma[1]:+d} ({lemma[2]}); search {pb}  {fen}")


def report(label, inst):
    verdict = Counter(); ex = []
    for fen, (sl, f, r, lemma, fb) in inst:
        if not fb or lemma is None:
            verdict["no comparison (no frame move valued)"] += 1; continue
        fb = [x for x in fb if not x[0].startswith("pair:")]
        if not fb:
            verdict["no comparison (no frame move valued)"] += 1; continue
        best_v = fb[0][1]
        if lemma[1] == best_v:
            k = "lemma value = frame search" + (" (same move)" if lemma[0] == fb[0][0] else " (another move of equal value)")
        elif lemma[1] < best_v:
            k = "frame search found MORE than the lemma (the lemma missed a resource)"
        else:
            k = "lemma promised MORE than the search certifies"
        verdict[k] += 1
        if "MORE" in k and len(ex) < 10:
            ex.append((k, sl, f, r, lemma, fb, fen))
    print(f"OBS {label}: {len(inst)} instances")
    for k, n in verdict.most_common():
        print(f"OBS   {k}: {n}")
    for k, sl, f, r, lemma, fb, fen in ex:
        print(f"OBS   {k}: {sl}->{f}->{r} lemma {lemma[0]} {lemma[1]:+d} ({lemma[2]}); search {fb}  {fen}")


if __name__ == "__main__":
    main()
