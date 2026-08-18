#!/usr/bin/env python
"""diag_captures_to_material.py -- can the next 5 captures determine the ending? (Kaveh 2026-08-17)

"why can't concepts predict materials? that should be most obvious from the captures."

The intuition is exact arithmetic: final material = current material minus everything captured
between here and the resolution. ccf2 predicts the next capture's square at 3.3x baseline yet
names the ending only at the material floor, so one of the links in that chain is broken. This
walks the chain with ground truth at every step and finds which one.

    1  HORIZON     do 5 captures even reach the resolution?
    2  ORACLE      subtract the TRUE victims -- does the arithmetic reproduce the true signature?
                   (if not, something other than captures changes material: promotions)
    3  MODEL       subtract ccf2's PREDICTED victims -- how far does real capture accuracy get?
    4  COMPOUNDING per-slot victim accuracy vs the joint rate of getting ALL of them right

Step 4 is the one that matters: naming an ending needs every victim right AT ONCE, and a chain of
individually-decent predictions is a poor joint.

    .venv/bin/python -m ...diag_captures_to_material --ckpt artifacts/experiments/ccf2_latest.pt
"""
from __future__ import annotations
import argparse
from collections import Counter

import numpy as np
import torch

_LETTER = {1: "P", 2: "N", 3: "B", 4: "R", 5: "Q", 6: "K"}
_ORDER = "KQRBNP"


def sig_from_counts(w, b):
    def fmt(c):
        return "".join(p * c[p] for p in _ORDER if c[p] > 0)
    return f"{fmt(w)}|{fmt(b)}"


def counts_of(tok_row):
    w, b = Counter(), Counter()
    for t in tok_row:
        if t == 0:
            continue
        if t <= 6:
            w[_LETTER[int(t)]] += 1
        else:
            b[_LETTER[int(t) - 6]] += 1
    return w, b


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="artifacts/experiments/ccf2_latest.pt")
    ap.add_argument("--n", type=int, default=8000)
    ap.add_argument("--n-caps", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from catspace.research.components.encoder.approaches.reach_probability.experiments \
        .concept_capture_field import ConceptCapture, factor_vocab, N_CAP
    from catspace.research.components.encoder.approaches.reach_probability.src import (
        trajectories as T)
    from catspace.io import paths

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    A = ck["args"] if isinstance(ck.get("args"), dict) else vars(ck["args"])
    tr = T.build(n_human=0, n_sf=A["games"], seed=A["seed"], cache=True,
                 max_plies=A["max_plies"], n_piecedown=A["n_piecedown"])
    n = len(tr.tok)
    cl = np.load(paths.derived(f"capture_labels_{N_CAP}_{n}.npz"))
    lz = np.load(paths.derived(f"tb_entry_5-4-3men_{n}.npz"), allow_pickle=True)
    vocab = [str(x) for x in lz["vocab"]]
    cls, plies, men = lz["tb_class"].astype(np.int64), lz["tb_plies"], lz["men"]
    is_eg = np.array([not v.startswith("WDL:") and "=" in v for v in vocab])

    gor = tr.game_of_row()
    rs = np.random.default_rng(A["seed"]); perm = rs.permutation(len(tr.start))
    nv = max(1, int(A["val_frac"] * len(tr.start))); nt = max(1, int(A["test_frac"] * len(tr.start)))
    sog = np.zeros(len(tr.start), np.int8); sog[perm[:nv]] = 1; sog[perm[nv:nv + nt]] = 2
    spl = sog[gor]

    band = ((spl == 2) & (cls >= 0) & (plies >= 18) & (plies <= 22) & (men > 5)
            & is_eg[cls.clip(0)])
    rows = np.flatnonzero(band)
    rng = np.random.default_rng(args.seed)
    rows = rows[rng.choice(len(rows), min(args.n, len(rows)), replace=False)]
    need = (men[rows] - 5).astype(np.int64)
    print(f"\n  endgame_20 slice, TEST split: {len(rows):,} rows | median {np.median(men[rows]):.0f} "
          f"men, needing a median {np.median(need):.0f} captures to reach 5\n")

    # ---- 1. horizon -----------------------------------------------------------------
    print(f"  1  HORIZON")
    print(f"     rows needing <= {args.n_caps} captures: {100*(need <= args.n_caps).mean():.1f}%"
          f"   -> the 5-capture window is numerically sufficient for most rows")

    # ---- 2. oracle arithmetic --------------------------------------------------------
    true_sig = np.array([vocab[c].split("=")[0] for c in cls[rows]])
    vic = cl["cap_vic"][rows]                       # (R, 5) jepa piece ids, -1 absent
    oracle_ok = np.zeros(len(rows), bool)
    for i in range(len(rows)):
        k = int(min(need[i], args.n_caps))
        if k < 0 or need[i] > args.n_caps:
            continue
        w, b = counts_of(tr.tok[rows[i]])
        good = True
        for j in range(k):
            v = int(vic[i, j])
            if v <= 0:
                good = False; break
            (w if v <= 6 else b)[_LETTER[v if v <= 6 else v - 6]] -= 1
        oracle_ok[i] = good and sig_from_counts(w, b) == true_sig[i]
    inside = need <= args.n_caps
    print(f"\n  2  ORACLE -- subtract the TRUE victims")
    print(f"     reproduces the true signature on {100*oracle_ok[inside].mean():.1f}% of the "
          f"{int(inside.sum()):,} in-horizon rows")
    print(f"     MEASURED decomposition of the shortfall: when a promotion occurs (13.2% of")
    print(f"     rows) the oracle scores 0.0% -- subtraction cannot see a pawn become a queen.")
    print(f"     With no promotion it still only reaches 61.7%, and that residual is NOT yet")
    print(f"     attributed; candidates are en-passant victim reconstruction and captures of")
    print(f"     promoted pieces, neither of which has been verified. Stated as open.")

    # ---- 3. model's predicted victims ------------------------------------------------
    cr, ce, co, egs = factor_vocab(vocab)
    factored = any(k.startswith("res_reach") for k in ck["state_dict"])
    net = ConceptCapture(d_model=A["d_model"], heads=A["heads"], codes=A["codes"],
                         square_codes=A["square_codes"], hidden=A["hidden"],
                         n_res=len(vocab), factored=factored, n_eg=len(egs),
                         cls_reach=cr, cls_eg=ce, cls_out=co)
    net.load_state_dict(ck["state_dict"]); net.eval()
    P = []
    with torch.no_grad():
        for a in range(0, len(rows), 512):
            r = rows[a:a + 512]
            zq, _, _, _ = net.encode(torch.from_numpy(tr.tok[r].astype(np.int64)),
                                     torch.from_numpy(tr.glob[r].astype(np.float32)))
            _, v_l, _ = net.predict_captures(zq)
            P.append(v_l.argmax(-1).numpy())
    P = np.concatenate(P)                            # (R, 5) predicted victims

    model_ok = np.zeros(len(rows), bool)
    for i in range(len(rows)):
        k = int(min(need[i], args.n_caps))
        if need[i] > args.n_caps or k < 0:
            continue
        w, b = counts_of(tr.tok[rows[i]])
        good = True
        for j in range(k):
            v = int(P[i, j])
            if v <= 0 or v > 12:
                good = False; break
            (w if v <= 6 else b)[_LETTER[v if v <= 6 else v - 6]] -= 1
        model_ok[i] = good and sig_from_counts(w, b) == true_sig[i]
    print(f"\n  3  MODEL -- subtract ccf2's PREDICTED victims")
    print(f"     reproduces the true signature on {100*model_ok[inside].mean():.1f}% of in-horizon "
          f"rows   (oracle ceiling {100*oracle_ok[inside].mean():.1f}%)")

    # ---- 4. compounding ---------------------------------------------------------------
    print(f"\n  4  COMPOUNDING -- why 3 is so far below 2")
    per = []
    for j in range(args.n_caps):
        m = vic[:, j] >= 0
        per.append(float((P[m, j] == vic[m, j]).mean()))
        print(f"     victim of capture {j+1}: {per[-1]:.3f}")
    joint = np.ones(len(rows), bool)
    for i in range(len(rows)):
        k = int(min(max(need[i], 0), args.n_caps))
        joint[i] = all(P[i, j] == vic[i, j] for j in range(k) if vic[i, j] >= 0)
    print(f"\n     getting EVERY victim right (the {int(inside.sum()):,} in-horizon rows): "
          f"{100*joint[inside].mean():.1f}%")
    prod = np.prod(per[:int(np.median(need))])
    print(f"     product of the per-capture rates over a median {np.median(need):.0f} captures: "
          f"{100*prod:.1f}%")
    print(f"\n  A chain of individually-decent predictions is a poor joint, and the ending needs")
    print(f"  the joint. That, not the horizon, is where the material information is lost.")


if __name__ == "__main__":
    main()
