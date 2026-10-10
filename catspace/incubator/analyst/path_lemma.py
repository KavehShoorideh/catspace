"""THE PATH LEMMA, audited (Kaveh 2026-10-09: "if the line from the king is blocked, there's no piece that can attack
from there -- a base lemma -- so we can see we have accounted for all the threats").

rust/analyst_core/src/paths.rs reads, without search, every way a side can make a NEW attack on a square with its next
move: a piece ARRIVING on a square bearing on it (a line, a jump, a pawn's diagonal, the king's step), or a DISCOVERY
(our piece in front of our slider on a line to it moves off the line). `path_audit` compares that set with brute
force (every legal move played, the attacks recomputed) for both sides, against the other side's king zone and each of
its pieces. A MISS (a move brute force finds and the lemma does not) is a soundness failure -- a surprise to explain
with a refined lemma; an EXTRA (claimed, not real) costs only precision.
"""
from __future__ import annotations

import argparse
import os
import time
from collections import Counter


def _one(fen):
    import analyst_core
    return fen, analyst_core.path_audit(fen), analyst_core.path_audit_counts(fen)


def audit(tag: str, limit: int, workers: int) -> None:
    from multiprocessing import Pool
    from .tactic_census_measure import load
    fens = [r["fen"] for r in load(tag)][:limit]
    t0 = time.time()
    missed = Counter(); extra = Counter(); ex_m = []; ex_e = []; zones = brute = lem = 0
    with Pool(workers) as p:
        for fen, rows, (z, bm, lm) in p.imap_unordered(_one, fens, chunksize=64):
            zones += z; brute += bm; lem += lm
            for side, zone, mi, xt in rows:
                kind = zone.split()[0]
                missed[kind] += len(mi); extra[kind] += len(xt)
                if mi and len(ex_m) < 8:
                    ex_m.append((fen, side, zone, mi))
                if xt and len(ex_e) < 4:
                    ex_e.append((fen, side, zone, xt))
    dt = time.time() - t0
    print(f"OBS path lemma audit: {len(fens)} positions, both sides, the king zone and every piece ({zones} zones); {dt:.0f} s on {workers} processes")
    print(f"OBS   new-attack moves: brute force {brute}, the lemma {lem}")
    for kind in sorted(set(missed) | set(extra)):
        print(f"OBS   {kind}: moves the lemma MISSED {missed[kind]}, moves it over-claimed {extra[kind]}")
    for fen, side, zone, mi in ex_m:
        print(f"OBS   missed  {side} -> {zone}: {mi[:6]}  {fen}")
    for fen, side, zone, xt in ex_e:
        print(f"OBS   extra   {side} -> {zone}: {xt[:6]}  {fen}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="c300")
    ap.add_argument("--limit", type=int, default=100000)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args(argv)
    audit(a.tag, a.limit, a.workers)


if __name__ == "__main__":
    main()
