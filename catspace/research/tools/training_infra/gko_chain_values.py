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


MV_CAP = 48


def _order_cache(fens, path="data/derived/gko1_chain_move_order.npz"):
    """One-time: uci move order per chain row (python-chess legal_moves order --
    the same order the graph cache's mv slots use)."""
    import os
    if os.path.exists(path):
        return np.load(path, allow_pickle=True)["ucis"]
    import chess
    t0 = time.time()
    ucis = []
    for i, f in enumerate(fens):
        b = chess.Board(f)
        ucis.append(" ".join(m.uci() for m in list(b.legal_moves)[:MV_CAP]))
        if i and i % 100000 == 0:
            print(f"  [order] {i:,}/{len(fens):,} [{time.time()-t0:.0f}s]", flush=True)
    ucis = np.array(ucis)
    np.savez_compressed(path, ucis=ucis)
    print(f"  [order] cached {len(ucis):,} rows [{time.time()-t0:.0f}s]", flush=True)
    return ucis


def build_once(out="data/derived/gko1_chain_values.npz"):
    t0 = time.time()
    cf = np.load("data/derived/gko1_chain_fens.npz", allow_pickle=False)
    fens = [str(x) for x in cf["fen"]]
    order = _order_cache(fens)
    c = sqlite3.connect("file:data/derived/gko1_oracle.sqlite?mode=ro", uri=True,
                        timeout=90)
    c.execute("PRAGMA busy_timeout=60000")
    fset = {f: i for i, f in enumerate(fens)}
    v = np.full(len(fens), np.nan, np.float32)
    wpv = np.full((len(fens), MV_CAP), np.nan, np.float16)
    wdl3 = np.full((len(fens), 3), np.nan, np.float16)
    for fen, payload in c.execute("SELECT fen, payload FROM labels WHERE oracle='sf'"):
        i = fset.get(fen)
        if i is None:
            continue
        pl = json.loads(payload)
        wp = pl.get("wp", {})
        if not wp:
            continue
        v[i] = max(wp.values())
        best_u = max(wp, key=wp.get)
        wsrc = pl.get("wdl", {})
        if best_u in wsrc:
            wdl3[i] = np.asarray(wsrc[best_u], np.float16) / 1000.0
        for j, u in enumerate(order[i].split()):
            if u in wp:
                wpv[i, j] = wp[u]
    c.close()
    ok = int(np.isfinite(v).sum())
    np.savez_compressed(out, v=v, wp=wpv, wdl=wdl3)
    print(f"[cv] {ok:,}/{len(fens):,} chain rows valued ({100*ok/len(fens):.1f}%), "
          f"full action-value vectors included [{time.time()-t0:.0f}s]", flush=True)
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
