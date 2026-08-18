#!/usr/bin/env python
"""compare_event_vs_capture.py -- ccf2 (captures) vs cef1 (events + material). (Kaveh 2026-08-17)

"which seems better?"

The two runs share an encoder, both codebooks, the persistence term and the reach x ending x
outcome resolution head -- cef1's model class SUBCLASSES ccf2's -- so the resolution rows are the
SAME task and are directly comparable. The prediction heads are not: ccf2 predicts captures, cef1
predicts captures-or-promotions plus the material left behind, over different labels. Those are
reported separately and against their own baselines rather than pretended to be one number.

THE MATERIAL READOUT THAT MATTERS. `ALL12` in the training log scores all twelve counts at event
5, which is not the question. The question is whether the model names the material AT THE EVENT
THAT REACHES THE TABLEBASE -- an event index that differs per row. That is the direct analogue of
naming the ending, and it is what this whole objective change was built for.

    .venv/bin/python -m ...compare_event_vs_capture
"""
from __future__ import annotations

import argparse
import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--capture-ckpt", default="artifacts/experiments/ccf2_latest.pt")
    ap.add_argument("--event-ckpt", default="artifacts/experiments/cef1_latest.pt")
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=512)
    args = ap.parse_args()

    from catspace.research.components.encoder.approaches.reach_probability.experiments \
        .concept_capture_field import ConceptCapture, factor_vocab
    from catspace.research.components.encoder.approaches.reach_probability.experiments \
        .concept_event_field import ConceptEvent, N_EV
    from catspace.research.components.encoder.approaches.reach_probability.src import (
        trajectories as T)
    from catspace.io import paths

    ck0 = torch.load(args.capture_ckpt, map_location="cpu", weights_only=False)
    A = ck0["args"] if isinstance(ck0.get("args"), dict) else vars(ck0["args"])
    tr = T.build(n_human=0, n_sf=A["games"], seed=A["seed"], cache=True,
                 max_plies=A["max_plies"], n_piecedown=A["n_piecedown"])
    n = len(tr.tok)
    lz = np.load(paths.derived(f"tb_entry_5-4-3men_{n}.npz"), allow_pickle=True)
    vocab = [str(x) for x in lz["vocab"]]
    cls, plies, men = lz["tb_class"].astype(np.int64), lz["tb_plies"], lz["men"]
    ev = np.load(paths.derived(f"event_labels_{N_EV}_{n}.npz"), allow_pickle=True)
    cr, ce, co, egs = factor_vocab(vocab)

    gor = tr.game_of_row()
    rs = np.random.default_rng(A["seed"]); perm = rs.permutation(len(tr.start))
    nv = max(1, int(A["val_frac"] * len(tr.start))); nt = max(1, int(A["test_frac"] * len(tr.start)))
    sog = np.zeros(len(tr.start), np.int8); sog[perm[:nv]] = 1; sog[perm[nv:nv + nt]] = 2
    spl = sog[gor]

    def load(path, cls_):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        m = cls_(d_model=A["d_model"], heads=A["heads"], codes=A["codes"],
                 square_codes=A["square_codes"], hidden=A["hidden"], n_res=len(vocab),
                 factored=True, n_eg=len(egs), cls_reach=cr, cls_eg=ce, cls_out=co)
        m.load_state_dict(ck["state_dict"]); m.eval()
        return m, ck.get("step")

    net_c, st_c = load(args.capture_ckpt, ConceptCapture)
    net_e, st_e = load(args.event_ckpt, ConceptEvent)
    print(f"\n  ccf2 (captures)          step {st_c}")
    print(f"  cef1 (events + material) step {st_e}")

    def encode(net, rows):
        out = []
        with torch.no_grad():
            for a in range(0, len(rows), args.batch):
                r = rows[a:a + args.batch]
                zq, _, _, _ = net.encode(torch.from_numpy(tr.tok[r].astype(np.int64)),
                                         torch.from_numpy(tr.glob[r].astype(np.float32)))
                out.append(zq)
        return torch.cat(out)

    # ---- 1. the SHARED task: resolution at 20 plies ----------------------------------
    rng = np.random.default_rng(0)
    base = (spl == 2) & (cls >= 0)
    band = base & (plies >= 18) & (plies <= 22) & (men > 5)
    r20 = np.flatnonzero(band)
    r20 = r20[rng.choice(len(r20), min(args.n, len(r20)), replace=False)]
    is_eg = np.array([not vocab[c].startswith("WDL:") for c in cls[r20]])
    print(f"\n  ================ SHARED TASK: resolution @20 plies "
          f"(n={len(r20):,}, {100*is_eg.mean():.1f}% true endgames) ================")
    print(f"  {'':26} {'ccf2':>9} {'cef1':>9}   baseline")
    res = {}
    for tag, net in (("ccf2", net_c), ("cef1", net_e)):
        zq = encode(net, r20)
        with torch.no_grad():
            P = net.resolution_logits(zq).argmax(-1).numpy()
            _, e_l, _ = net.predict_resolution(zq)
            eg_ids = np.flatnonzero(np.array([not v.startswith("WDL:") for v in vocab]))
            pe = eg_ids[e_l.argmax(-1).numpy()]
        sig_of = np.array([v.split("=")[0] for v in vocab])
        out_of = np.array([v.split("=")[-1] for v in vocab])
        res[tag] = {
            "resolution_20": float((P == cls[r20]).mean()),
            "endgame_20": float((P[is_eg] == cls[r20][is_eg]).mean()),
            "pred_eg_rate": float(np.array([not vocab[c].startswith("WDL:") for c in P]).mean()),
            "distinct": len(set(P.tolist())),
            "cond_ending": float((pe[is_eg] == cls[r20][is_eg]).mean()),
            "cond_sig": float((sig_of[pe[is_eg]] == sig_of[cls[r20][is_eg]]).mean()),
            "cond_out": float((out_of[pe[is_eg]] == out_of[cls[r20][is_eg]]).mean()),
        }
    for k, bl in (("resolution_20", "0.606"), ("endgame_20", "0.213 <- HEADLINE"),
                  ("cond_ending", "0.211"), ("cond_sig", ""), ("cond_out", ""),
                  ("pred_eg_rate", f"true {is_eg.mean():.3f}"), ("distinct", "of 401")):
        a, b = res["ccf2"][k], res["cef1"][k]
        fmt = (lambda x: f"{x:>9.3f}") if isinstance(a, float) else (lambda x: f"{x:>9}")
        mark = "  <-- cef1 better" if isinstance(a, float) and b > a + 1e-9 else (
            "  <-- ccf2 better" if isinstance(a, float) and a > b + 1e-9 else "")
        print(f"  {k:<26}{fmt(a)}{fmt(b)}   {bl}{mark}")

    # ---- 2. cef1 ONLY: does it name the tablebase material? ---------------------------
    eg_rows = np.flatnonzero(band)
    keep = np.array([not vocab[c].startswith("WDL:") and "=" in vocab[c] for c in cls[eg_rows]])
    eg_rows = eg_rows[keep]
    eg_rows = eg_rows[rng.choice(len(eg_rows), min(args.n, len(eg_rows)), replace=False)]
    emen = ev["ev_men"][eg_rows]                      # (R, N_EV)
    ecnt = ev["ev_counts"][eg_rows].astype(np.int64)  # (R, N_EV, 12)
    hit_k = np.full(len(eg_rows), -1)
    for k in range(N_EV):
        m = (emen[:, k] == 5) & (hit_k < 0)
        hit_k[m] = k
    reach = hit_k >= 0
    zq = encode(net_e, eg_rows)
    with torch.no_grad():
        pc = net_e.predict_material(zq).argmax(-1).numpy()      # (R, N_EV, 12)
    idx = np.clip(hit_k, 0, N_EV - 1)
    got = pc[np.arange(len(eg_rows)), idx]
    want = ecnt[np.arange(len(eg_rows)), idx]
    all12 = (got == want).all(-1)
    per = (got == want).mean()
    print(f"\n  ================ cef1 ONLY: naming the TABLEBASE MATERIAL ================")
    print(f"  endgame rows in the 20-ply band: {len(eg_rows):,}")
    print(f"  an event reaches 5 men within {N_EV}: {100*reach.mean():.1f}%")
    print(f"  at THAT event (not event 5):")
    print(f"     per-count accuracy            {per:.3f}")
    print(f"     ALL 12 counts correct         {all12[reach].mean():.3f}   <- names the material")
    print(f"     ccf2's route (chained victims)  0.081   (measured, diag_captures_to_material)")
    print(f"     oracle ceiling for that route   0.548")
    # men alone -- an easier readout that still carries most of the endgame question
    men_ok = (got.sum(-1) == want.sum(-1))
    print(f"     total man-count correct       {men_ok[reach].mean():.3f}")


if __name__ == "__main__":
    main()
