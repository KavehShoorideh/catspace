#!/usr/bin/env python
"""gko_chain_values.py -- sidecar: per chain-graph row, the SF state value (max over
legal moves' win-probs from the mover's POV) pulled from the oracle store. Rebuilt
periodically; the trainer rescans it like shards and masks unlabeled rows, so
value-anchored training starts at partial coverage and tightens as labels land
(Kaveh 2026-08-21: predict future SF evals from ROLLED concepts vs ground truth).

    .venv/bin/python -m catspace.research.tools.training_infra.gko_chain_values [--loop]
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import time

import numpy as np


def build_once(out="data/derived/gko1_chain_values.npz"):
    t0 = time.time()
    cf = np.load("data/derived/gko1_chain_fens.npz", allow_pickle=False)
    fens = [str(x) for x in cf["fen"]]
    c = sqlite3.connect("file:data/derived/gko1_oracle.sqlite?mode=ro", uri=True,
                        timeout=90)
    c.execute("PRAGMA busy_timeout=60000")
    fset = set(fens)
    vals = {}
    for fen, payload in c.execute("SELECT fen, payload FROM labels WHERE oracle='sf'"):
        if fen in fset:
            wp = json.loads(payload).get("wp", {})
            if wp:
                vals[fen] = max(wp.values())
    c.close()
    v = np.full(len(fens), np.nan, np.float32)
    for i, f in enumerate(fens):
        if f in vals:
            v[i] = vals[f]
    ok = int(np.isfinite(v).sum())
    np.savez_compressed(out, v=v)
    print(f"[cv] {ok:,}/{len(fens):,} chain rows valued ({100*ok/len(fens):.1f}%) "
          f"[{time.time()-t0:.0f}s]", flush=True)
    return ok / len(fens)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--every", type=int, default=600)
    a = ap.parse_args()
    while True:
        build_once()
        if not a.loop:
            break
        time.sleep(a.every)


if __name__ == "__main__":
    main()
