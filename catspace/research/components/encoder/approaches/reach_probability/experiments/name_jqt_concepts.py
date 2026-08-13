#!/usr/bin/env python
"""name_jqt_concepts.py -- predicate naming for a JQT codebook (Kaveh 2026-08-12: "if there's
a known correlation with python-chess concepts, I want it known"). Same battery and map format
as concept_vq's naming pass (predicates are NAMING ONLY, never training), applied to the
jointly-trained quantizer in the _jqt sidecar.

    .venv/bin/python -m ...name_jqt_concepts --ckpt <field.pt> [--rows 120000]
writes <stem>_conceptmap.json
"""
from __future__ import annotations

import argparse
import json
import re

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--rows", type=int, default=120_000)
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    from catspace.research.components.encoder.approaches.reach_probability.experiments.concept_vq import (
        predicates_from_tok)
    from catspace.research.components.encoder.approaches.reach_probability.experiments.interpret_reach import (
        load_net)
    from catspace.research.components.encoder.approaches.reach_probability.experiments.jqt import (
        JQTModule)
    from catspace.research.components.encoder.approaches.reach_probability.src import trajectories as T

    base = args.ckpt[:-3] if args.ckpt.endswith(".pt") else args.ckpt
    stem = re.sub(r"_(latest|step\d+)$", "", base)
    net, pay = load_net(args.ckpt, args.device)
    c = pay["cfg"]
    pj = torch.load(stem + "_jqt.pt", map_location=args.device, weights_only=False)
    jm = JQTModule(d_model=pj["d_in"], heads=pj["heads"], codes=pj["codes"], d=pj["d"],
                   square_codes=pj.get("square_codes", 0),
                   piece_codes=pj.get("piece_codes", 0)).to(args.device)
    jm.load_state_dict(pj["state_dict"], strict=False); jm.eval()
    tr = T.build(n_human=0, n_sf=c["games"], seed=c["traj_seed"], max_plies=c["max_plies"],
                 n_piecedown=c.get("n_piecedown", 0), verbose=False)
    rng = np.random.default_rng(0)
    rows = rng.choice(tr.n_positions, min(args.rows, tr.n_positions), replace=False)
    ids_all = np.empty((len(rows), pj["heads"]), np.int64)
    with torch.no_grad():
        for a in range(0, len(rows), 4096):
            rr = rows[a:a + 4096]
            phi = net.backbone(
                torch.from_numpy(tr.tok[rr].astype(np.int64)).to(args.device),
                torch.from_numpy(tr.glob[rr].astype(np.float32)).to(args.device))
            _, ids = jm.target_codes(phi)
            ids_all[a:a + 4096] = ids.cpu().numpy()
    preds = predicates_from_tok(tr.tok[rows])
    cmap = {}
    for k, pv in preds.items():
        b0 = float(np.mean(pv))
        if b0 < 0.01 or b0 > 0.99:
            continue
        n_pred = float(pv.sum())
        best = None
        for h in range(pj["heads"]):
            for code in range(pj["codes"]):
                m = ids_all[:, h] == code
                if m.sum() < 200:
                    continue
                prec = float(np.mean(pv[m]))                  # P(predicate | code)
                rec = float(pv[m].sum() / max(n_pred, 1))     # P(code | predicate)
                f1 = 2 * prec * rec / max(prec + rec, 1e-9)
                if best is None or f1 > best[4]:
                    best = (h, code, prec, rec, f1)
        if best:
            h, code, prec, rec, f1 = best
            # CONTAINMENT VERDICT (Kaveh 2026-08-12: "we need two numbers... broader or
            # narrower"): precision->code implies predicate; recall->predicate implies code
            if prec >= 0.75 and rec < 0.5:
                rel = "narrower"      # the code is a SPECIALIZATION of the named concept
            elif rec >= 0.75 and prec < 0.5:
                rel = "broader"       # the code covers the concept AND more
            elif prec >= 0.6 and rec >= 0.6:
                rel = "equivalent"
            else:
                rel = "overlapping"
            cmap[k] = {"head": h, "code": int(code),
                       "p_given_code": round(prec, 3), "recall": round(rec, 3),
                       "f1": round(f1, 3), "relation": rel,
                       "base": round(b0, 3), "lift": round(prec - b0, 3),
                       "anti": prec < b0}
            # CODE-FAMILY COVER: human concepts are broader than any single code (recall
            # caps at code-rate/predicate-rate), so the real mapping is a SET: greedily add
            # precision->=0.6 codes until the family covers >=80% of the predicate.
            cands = []
            for h2 in range(pj["heads"]):
                for c2 in range(pj["codes"]):
                    m2 = ids_all[:, h2] == c2
                    if m2.sum() < 200:
                        continue
                    pr2 = float(np.mean(pv[m2]))
                    if pr2 >= 0.6:
                        cands.append((pr2, h2, c2, m2))
            cands.sort(key=lambda x: -x[0])
            cover = np.zeros(len(pv), bool)
            fam = []
            for pr2, h2, c2, m2 in cands:
                gain = float(pv[m2 & ~cover].sum() / max(n_pred, 1))
                if gain < 0.005:
                    continue
                cover |= m2
                fam.append([int(h2), int(c2)])
                if float(pv[cover].sum() / max(n_pred, 1)) >= 0.8:
                    break
            fam_rec = float(pv[cover].sum() / max(n_pred, 1))
            fam_prec = float(np.mean(pv[cover])) if cover.any() else 0.0
            cmap[k]["family"] = fam
            cmap[k]["family_recall"] = round(fam_rec, 3)
            cmap[k]["family_precision"] = round(fam_prec, 3)
            print(f"  {k:22s} -> h{h}/c{code:<3} prec {prec:.0%} rec {rec:.0%} [{rel}] | "
                  f"family: {len(fam)} codes -> prec {fam_prec:.0%} rec {fam_rec:.0%}")
    json.dump(cmap, open(stem + "_conceptmap.json", "w"), indent=1)
    print(f"[name] {len(cmap)} named -> {stem}_conceptmap.json")


if __name__ == "__main__":
    main()
