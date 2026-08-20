#!/usr/bin/env python
"""threat_exec_join.py -- scatter the threat/execution census onto graph-sample rows.

Kaveh 2026-08-19 (the engineered tactical feature): "anytime a piece is threatened to be
taken -- a threat. But how often is it actually taken? That is the execution... insight into
when is a good time to execute the capture." The census (threat_exec_k8) holds one event per
(corpus row, square): a piece stood attacked, and either the capture landed within 8 plies or
it did not. This join turns that into a per-square supervised target aligned with a graph
file's p_row, for a threat->execution head:

    target[i, sq] = -1  no threat event at this square (EXCLUDED from the loss)
                     0  threatened, not executed within 8 plies
                     1  threatened, executed within 8 plies

The base rates that make this a real prediction problem and not a constant (2026-08-19
census): P(exec|threat)=0.149 overall; SEE>0 0.187 vs SEE=0 0.369 (the inversion: winning-
capture victims RUN; even trades sit) vs SEE<0 0.074.

    .venv/bin/python -m ...threat_exec_join   # smoke: join onto the 300k graph file
"""
from __future__ import annotations

import numpy as np


def join_threat_exec(p_row, te_path):
    """-> (N,64) int8 in {-1,0,1}; duplicate (row,sq) events collapse to max(executed)."""
    z = np.load(te_path)
    row, sq, ex = z["row"], z["sq"], z["executed"].astype(np.int8)
    order = np.argsort(p_row, kind="stable")
    sorted_rows = p_row[order]
    pos = np.searchsorted(sorted_rows, row)
    ok = (pos < len(sorted_rows))
    ok[ok] &= sorted_rows[pos[ok]] == row[ok]
    tgt = np.full((len(p_row), 64), -1, np.int8)
    gi = order[pos[ok]]; gsq = sq[ok]; gex = ex[ok]
    # max(executed) per (graph row, square): scatter 0s first, then 1s win
    tgt[gi[gex == 0], gsq[gex == 0]] = 0
    tgt[gi[gex == 1], gsq[gex == 1]] = 1
    return tgt


if __name__ == "__main__":
    g = np.load("data/derived/move_graphs_v2_300000_6545146.npz")
    t = join_threat_exec(g["p_row"], "data/derived/threat_exec_k8_2173011_6545146.npz")
    n_ev = int((t >= 0).sum()); n_ex = int((t == 1).sum())
    rows_with = float((t >= 0).any(1).mean())
    print(f"300k graph rows: {n_ev:,} threat events ({n_ev/len(t):.2f}/row), "
          f"{100*rows_with:.1f} percent of rows have >=1")
    print(f"P(exec|threat) on the joined subset: {n_ex/max(1,n_ev):.3f} "
          f"(census overall: 0.149)")
