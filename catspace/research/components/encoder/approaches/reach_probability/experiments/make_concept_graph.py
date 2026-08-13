#!/usr/bin/env python
"""make_concept_graph.py -- the concept-relation map (Kaveh 2026-08-12: "how do we see the
model's concepts relating to one another?").

Relations come from the GEOMETRY, not co-occurrence counts: every code's anchor lives in the
quasimetric, so concept->concept is a DIRECTED distance. Exported per code pair:
    d_ab, d_ba  (log1p dA between anchors, both directions)
Three relation classes are printed and drawn:
    neighbors   d_ab + d_ba both small       -> the collinearity clusters (dedupe candidates)
    gateways    d_ab small, d_ba large       -> a leads INTO b, no way back (irreversibility
                                                in concept space; macro-step candidates)
    strangers   both large                   -> independent vocabularies
Layout: classical MDS on the symmetrized distances. Node color = leverage (who the concept
serves), size = base rate. Output: <ckpt>_concept_graph.json + printed verdicts.

    .venv/bin/python -m ...make_concept_graph --ckpt <field.pt>
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--min-baserate", type=float, default=0.002,
                    help="drop codes that essentially never activate")
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    import re, os
    base = args.ckpt[:-3] if args.ckpt.endswith(".pt") else args.ckpt
    stem = re.sub(r"_(latest|step\d+)$", "", base)
    from catspace.research.components.planner.approaches.quasimetric_nav.kittychess import KittyChess
    from catspace.research.components.encoder.approaches.reach_probability.experiments.jqt import (
        JQTModule)
    eng = KittyChess(args.ckpt, args.device)
    pj = torch.load(stem + "_jqt.pt", map_location=args.device, weights_only=False)
    jm = JQTModule(d_model=pj["d_in"], heads=pj["heads"], codes=pj["codes"], d=pj["d"],
                   square_codes=pj.get("square_codes", 0),
                   piece_codes=pj.get("piece_codes", 0)).to(args.device)
    jm.load_state_dict(pj["state_dict"], strict=False); jm.eval()
    H, C = pj["heads"], pj["codes"]

    br = np.load(base + "_code_baserates.npy") if os.path.exists(base + "_code_baserates.npy") \
        else np.full((H, C), 0.01, np.float32)
    lev = np.zeros((H, C), np.float32)
    lp = base + "_concept_leverage.npz"
    if os.path.exists(lp):
        z = np.load(lp)
        for sw, hh, cc in zip(z["swing"], z["head"], z["code"]):
            lev[int(hh), int(cc)] = float(sw)

    keep = [(h, c) for h in range(H) for c in range(C) if br[h, c] >= args.min_baserate]
    hc = torch.tensor(keep, dtype=torch.long, device=args.device)
    with torch.no_grad():
        A = jm.anchors_for(hc).float()
        T = len(hc)
        ii = torch.arange(T, device=args.device).repeat_interleave(T)
        jj = torch.arange(T, device=args.device).repeat(T)
        D = torch.log1p(eng.net.dA(A[ii], A[jj]).clamp(min=0)).view(T, T).cpu().numpy()
    print(f"[graph] {T} active codes (baserate >= {args.min_baserate}); "
          f"directed distance matrix {T}x{T}")

    S = 0.5 * (D + D.T)
    np.fill_diagonal(S, 0.0)
    # classical MDS
    n = len(S)
    J = np.eye(n) - np.ones((n, n)) / n
    Bm = -0.5 * J @ (S ** 2) @ J
    w, v = np.linalg.eigh(Bm)
    idx = np.argsort(w)[::-1][:2]
    xy = v[:, idx] * np.sqrt(np.maximum(w[idx], 1e-9))
    xy = (xy - xy.min(0)) / (xy.max(0) - xy.min(0) + 1e-9)

    off = ~np.eye(n, dtype=bool)
    qn = np.quantile(S[off], 0.05)
    asym = D - D.T
    edges = []
    for a in range(n):
        for b in range(a + 1, n):
            near = S[a, b] <= qn
            gate = abs(asym[a, b]) > np.quantile(np.abs(asym[off]), 0.98) \
                and min(D[a, b], D[b, a]) <= np.quantile(D[off], 0.15)
            if near or gate:
                edges.append({"a": a, "b": b, "d": float(S[a, b]),
                              "kind": "gateway" if gate and not near else "neighbor",
                              "dir": int(np.sign(asym[b, a]))})   # +1: a -> b easier
    n_nb = sum(1 for e in edges if e["kind"] == "neighbor")
    print(f"[graph] edges: {n_nb} neighbor (collinearity clusters), "
          f"{len(edges)-n_nb} gateway (directed lead-ins)")
    strong_gates = sorted([e for e in edges if e["kind"] == "gateway"],
                          key=lambda e: e["d"])[:5]
    for e in strong_gates:
        a, b = (e["a"], e["b"]) if e["dir"] > 0 else (e["b"], e["a"])
        print(f"  gateway h{keep[a][0]}/c{keep[a][1]} -> h{keep[b][0]}/c{keep[b][1]} "
              f"(one-way lead-in)")

    out = {"nodes": [{"h": int(h), "c": int(c), "x": float(xy[i, 0]), "y": float(xy[i, 1]),
                      "br": float(br[h, c]), "lev": float(lev[h, c])}
                     for i, (h, c) in enumerate(keep)],
           "edges": edges}
    with open(stem + "_concept_graph.json", "w") as f:
        json.dump(out, f)
    print(f"[graph] -> {stem}_concept_graph.json")


if __name__ == "__main__":
    main()
