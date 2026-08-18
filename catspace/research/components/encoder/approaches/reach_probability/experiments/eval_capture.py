#!/usr/bin/env python
"""eval_capture.py -- THE ACCEPTANCE INSTRUMENT for capture-trained concepts.

Written BEFORE the first training run, per TESTING.md rule 13 ("the smoke must smoke the CLAIM
with a pre-registered miniature of the acceptance instrument"). Every threshold below is a
MEASURED baseline from the corpus, not chance, and every one was recorded before any model saw
a capture label:

    next capture: square      > 5.1%    (e5 is the modal capture square; chance is 1.6%)
    next capture: attacker    > 13.8%
    next capture: victim      > 29.1%   (49% of captures are pawns -- a strong constant)
    5th capture: square       > 4.2%
    endgame class @20 plies   > 0.213   <- THE HEADLINE. cdf5 scored 0.138 here, i.e. WORSE
                                           than a material lookup. Beating 0.213 is what
                                           "the concepts encode material trajectory" means.
    resolution @20 plies      > 0.606   (material lookup on the full 401-class label)

A row that fails is a row that fails; the instrument does not get renegotiated after seeing the
numbers.

    .venv/bin/python -m ...eval_capture --ckpt artifacts/experiments/ccf1_latest.pt
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

# pre-registered, measured on the corpus before training
BASELINE = {"next_sq": 0.051, "next_att": 0.138, "next_vic": 0.291, "fifth_sq": 0.042,
            "endgame_20": 0.213, "resolution_20": 0.606}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--split", type=int, default=2, help="0 train / 1 val / 2 test")
    ap.add_argument("--batch", type=int, default=512)
    args = ap.parse_args()

    from catspace.research.components.encoder.approaches.reach_probability.experiments \
        .concept_capture_field import ConceptCapture, N_CAP
    from catspace.research.components.encoder.approaches.reach_probability.src import (
        trajectories as T)
    from catspace.io import paths

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    A = ck["args"] if isinstance(ck.get("args"), dict) else vars(ck["args"])
    tr = T.build(n_human=0, n_sf=A["games"], seed=A["seed"], cache=True,
                 max_plies=A["max_plies"], n_piecedown=A["n_piecedown"])
    n = len(tr.tok)
    is_event = any(k.startswith("ev_kind") for k in ck["state_dict"])
    # capture labels exist only for corpora built for the capture objective; an event model does
    # not use them, so requiring the file would make the instrument unrunnable for no reason.
    cl = None if is_event else np.load(paths.derived(f"capture_labels_{N_CAP}_{n}.npz"))
    lz = np.load(paths.derived(f"tb_entry_5-4-3men_{n}.npz"), allow_pickle=True)
    vocab = [str(x) for x in lz["vocab"]]
    res_cls, plies, men = lz["tb_class"].astype(np.int64), lz["tb_plies"], lz["men"]
    gor = tr.game_of_row()
    rs = np.random.default_rng(A["seed"]); perm = rs.permutation(len(tr.start))
    nv = max(1, int(A["val_frac"] * len(tr.start))); nt = max(1, int(A["test_frac"] * len(tr.start)))
    sog = np.zeros(len(tr.start), np.int8); sog[perm[:nv]] = 1; sog[perm[nv:nv + nt]] = 2
    spl = sog[gor]

    # Score factored and flat checkpoints with the SAME instrument: which head a run used is
    # read off its state_dict rather than assumed from the current default, so ccf1's flat
    # numbers stay directly comparable to ccf2's factored ones.
    from catspace.research.components.encoder.approaches.reach_probability.experiments \
        .concept_capture_field import factor_vocab
    factored = any(k.startswith("res_reach") for k in ck["state_dict"])
    cr, ce, co, egs = factor_vocab(vocab)
    if is_event:
        from catspace.research.components.encoder.approaches.reach_probability.experiments \
            .concept_event_field import ConceptEvent as _Cls
    else:
        _Cls = ConceptCapture
    net = _Cls(d_model=A["d_model"], heads=A["heads"], codes=A["codes"],
                         square_codes=A["square_codes"], hidden=A["hidden"],
                         n_res=len(vocab), factored=factored, n_eg=len(egs),
                         cls_reach=cr, cls_eg=ce, cls_out=co)
    net.load_state_dict(ck["state_dict"]); net.eval()
    print(f"  [head: {'SPLIT reach x ending' if factored else 'FLAT 401-way'}"
          f"{' | EVENT model -- capture rows N/A' if is_event else ''}]")

    def run(rows):
        sq, vi, at, rz = [], [], [], []
        with torch.no_grad():
            for a in range(0, len(rows), args.batch):
                r = rows[a:a + args.batch]
                zq, _, _, _ = net.encode(
                    torch.from_numpy(tr.tok[r].astype(np.int64)),
                    torch.from_numpy(tr.glob[r].astype(np.float32)))
                if is_event:
                    z0 = np.zeros((len(r), N_CAP), np.int64) - 1
                    sq.append(z0); vi.append(z0); at.append(z0)
                else:
                    s, v, t_ = net.predict_captures(zq)
                    sq.append(s.argmax(-1).numpy()); vi.append(v.argmax(-1).numpy())
                    at.append(t_.argmax(-1).numpy())
                rz.append(net.resolution_logits(zq).argmax(-1).numpy())
        return (np.concatenate(sq), np.concatenate(vi), np.concatenate(at),
                np.concatenate(rz))

    rng = np.random.default_rng(0)
    base = (spl == args.split) & (res_cls >= 0)
    base_tr = (spl == 0) & (res_cls >= 0)      # material floor is FITTED ON TRAIN
    got = {}
    if is_event:
        rows = np.flatnonzero(base)
        rows = rows[rng.choice(len(rows), min(args.n, len(rows)), replace=False)]
    else:
        rows = np.flatnonzero(base & (cl["cap_sq"][:, 0] >= 0))
        rows = rows[rng.choice(len(rows), min(args.n, len(rows)), replace=False)]
        P_sq, P_vi, P_at, _ = run(rows)
        got["next_sq"] = float((P_sq[:, 0] == cl["cap_sq"][rows, 0]).mean())
        got["next_vic"] = float((P_vi[:, 0] == cl["cap_vic"][rows, 0]).mean())
        got["next_att"] = float((P_at[:, 0] == cl["cap_att"][rows, 0]).mean())
        m5 = cl["cap_sq"][rows, N_CAP - 1] >= 0
        got["fifth_sq"] = float((P_sq[m5, N_CAP-1] == cl["cap_sq"][rows[m5], N_CAP-1]).mean())

    # resolution at 20 plies, whole label and endgame-only
    band = base & (plies >= 18) & (plies <= 22) & (men > 5)
    r20 = np.flatnonzero(band)
    r20 = r20[rng.choice(len(r20), min(args.n, len(r20)), replace=False)]
    _, _, _, P_res = run(r20)
    got["resolution_20"] = float((P_res == res_cls[r20]).mean())
    is_eg = np.array([not vocab[c].startswith("WDL:") for c in res_cls[r20]])
    got["endgame_20"] = float((P_res[is_eg] == res_cls[r20][is_eg]).mean()) if is_eg.any() \
        else float("nan")

    print(f"\n  ACCEPTANCE INSTRUMENT — {args.ckpt.split('/')[-1]} step {ck.get('step')}, "
          f"split {args.split}, n={len(rows):,}\n")
    print(f"  {'claim':<26} {'result':>8} {'baseline':>9} {'delta':>8}   verdict")
    order = (["endgame_20", "resolution_20"] if is_event else
             ["next_sq", "next_att", "next_vic", "fifth_sq", "endgame_20", "resolution_20"])
    npass = 0
    for k in order:
        d = got[k] - BASELINE[k]
        ok = d > 0
        npass += ok
        star = " <- HEADLINE" if k == "endgame_20" else ""
        print(f"  {k:<26} {got[k]:>8.3f} {BASELINE[k]:>9.3f} {d:>+8.3f}   "
              f"{'PASS' if ok else 'FAIL'}{star}")
    print(f"\n  {npass}/{len(order)} rows beat their pre-registered baseline")
    print(f"  reference: cdf5 scored 0.138 on endgame_20 (worse than the 0.213 material lookup)")

    # THE COLLAPSE DIAGNOSTIC, printed always. A flat resolution head can pass `resolution_20`
    # while never naming an endgame at all -- ccf1 predicted 95.5% WDL against a 77.9% true
    # rate. An aggregate accuracy hides that; this does not.
    true_eg = is_eg
    pred_eg = np.array([not vocab[c].startswith("WDL:") for c in P_res])
    print(f"\n  collapse check @20 plies")
    print(f"    true  endgame {100*true_eg.mean():>5.1f}%   predicted endgame "
          f"{100*pred_eg.mean():>5.1f}%   (a head that never names one reads 0.0%)")
    if true_eg.any():
        print(f"    of true-endgame rows, {100*(~pred_eg[true_eg]).mean():>5.1f}% are given a "
              f"WDL class instead")
    print(f"    distinct classes ever predicted: {len(set(P_res.tolist()))} of {len(vocab)}")

    # ---- POST-HOC, NOT PRE-REGISTERED ------------------------------------------------------
    # `endgame_20` above scores the UNCONDITIONAL argmax over all 401 classes on endgame rows,
    # which is the decision rule cdf5 and ccf1 were scored under and therefore stays exactly as
    # written. But it structurally punishes a calibrated model: P(ends first) concentrates on 3
    # WDL classes while P(reaches) spreads over ~400 endings, so naming an ending requires
    # unusual confidence. The question actually asked -- "of the games that DO make it to a
    # tablebase ending, how do we perform 20 plies before?" -- is CONDITIONAL, and is reported
    # here separately and labelled, rather than by redefining the row above.
    if factored and true_eg.any():
        eg_ids = np.flatnonzero(np.array([not v.startswith("WDL:") for v in vocab]))
        cond = []
        with torch.no_grad():
            for a in range(0, len(r20), args.batch):
                r = r20[a:a + args.batch]
                zq, _, _, _ = net.encode(torch.from_numpy(tr.tok[r].astype(np.int64)),
                                         torch.from_numpy(tr.glob[r].astype(np.float32)))
                _, e_l, _ = net.predict_resolution(zq)
                cond.append(e_l.argmax(-1).numpy())
        pred_eg_id = eg_ids[np.concatenate(cond)]
        acc_cond = float((pred_eg_id[true_eg] == res_cls[r20][true_eg]).mean())
        sig_of = np.array([vocab[c].split("=")[0] for c in range(len(vocab))])
        acc_sig = float((sig_of[pred_eg_id[true_eg]] == sig_of[res_cls[r20][true_eg]]).mean())
        out_of = np.array([vocab[c].split("=")[-1] for c in range(len(vocab))])
        acc_out = float((out_of[pred_eg_id[true_eg]] == out_of[res_cls[r20][true_eg]]).mean())
        # conditional material floor, fitted on TRAIN, restricted to endgame classes
        from collections import Counter, defaultdict
        trn = np.flatnonzero(base_tr & (plies >= 18) & (plies <= 22) & (men > 5))
        trn = trn[np.array([not vocab[c].startswith("WDL:") for c in res_cls[trn]])]
        trn = trn[rng.choice(len(trn), min(60000, len(trn)), replace=False)]
        def msig(rows):
            t = tr.tok[rows]
            return [tuple(int((t[i] == v).sum()) for v in range(1, 13)) for i in range(len(rows))]
        tab = defaultdict(Counter)
        for k, y in zip(msig(trn), res_cls[trn]):
            tab[k][y] += 1
        bestc = {k: c.most_common(1)[0][0] for k, c in tab.items()}
        mj = Counter(res_cls[trn].tolist()).most_common(1)[0][0]
        te = r20[true_eg]
        base_cond = float(np.mean([bestc.get(k, mj) == y
                                   for k, y in zip(msig(te), res_cls[te].tolist())]))
        print(f"\n  POST-HOC (not pre-registered) — given the game DOES reach a tablebase:")
        print(f"    name the exact ending      {acc_cond:>8.3f}   material floor {base_cond:>6.3f}"
              f"   {'beats it' if acc_cond > base_cond else 'below it'}")
        print(f"    name the material signature{acc_sig:>8.3f}")
        print(f"    name the outcome (W/D/L)   {acc_out:>8.3f}")
        print(f"    n = {int(true_eg.sum()):,} endgame rows in the 20-ply band")


if __name__ == "__main__":
    main()
