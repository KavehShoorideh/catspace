#!/usr/bin/env python
"""eval_horizon_distribution.py -- how far out is the ending detectable, as a DISTRIBUTION?

"let's see how we do with ten plies out and five plies out and fifteen plies out... a sensitivity
analysis to see what is the threshold that is detectable... put statistical bounds around it and
say this is where the distribution is likely to be... ask for a distribution, not just a single
type of endgame, or just a single combination of pieces." (Kaveh 2026-08-17)

WHY DISTRIBUTIONAL SCORING IS THE RIGHT MOVE HERE. `endgame_20` scores an argmax over 401 classes,
and that was already measured to be a decision-rule artifact: the same model answers 0.211 when
asked "which ending, given it reaches one" and 0.058 when asked to pick a single class outright,
because the three WDL classes hold most of the probability mass. Scoring the predicted
distribution asks what the model actually knows instead of how it breaks a tie.

METRICS, each with a 95% bootstrap CI over rows:
    top1/3/5    is the truth in the k most likely endings?
    log-loss    the proper scoring rule -- rewards putting mass on the truth, punishes
                confident nonsense. Lower is better.
    Brier       bounded alternative, less dominated by single bad rows
    ECE         calibration: when it says 30%, does it happen 30% of the time?

BASELINES, both DISTRIBUTIONS fitted on TRAIN, so the comparison is like-for-like:
    marginal    the overall frequency of each ending, ignoring the position entirely
    material    the empirical P(ending | material signature) -- "counting pieces", but as a
                distribution rather than its argmax. This is the honest floor.

    .venv/bin/python -m ...eval_horizon_distribution --horizons 5,10,15,20
"""
from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np
import torch


def ece(p_true, correct, bins=10):
    """Expected calibration error over the model's confidence in its top pick."""
    e, n = 0.0, len(p_true)
    for lo in np.linspace(0, 1, bins + 1)[:-1]:
        m = (p_true >= lo) & (p_true < lo + 1.0 / bins)
        if m.sum():
            e += m.sum() / n * abs(correct[m].mean() - p_true[m].mean())
    return float(e)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="artifacts/experiments/cef1_latest.pt")
    ap.add_argument("--horizons", default="5,10,15,20")
    ap.add_argument("--band", type=int, default=2)
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--boot", type=int, default=400)
    ap.add_argument("--batch", type=int, default=512)
    args = ap.parse_args()

    from catspace.research.components.encoder.approaches.reach_probability.experiments \
        .concept_capture_field import ConceptCapture, factor_vocab
    from catspace.research.components.encoder.approaches.reach_probability.experiments \
        .concept_event_field import ConceptEvent
    from catspace.research.components.encoder.approaches.reach_probability.src import (
        trajectories as T)
    from catspace.io import paths

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    A = ck["args"] if isinstance(ck.get("args"), dict) else vars(ck["args"])
    tr = T.build(n_human=0, n_sf=A["games"], seed=A["seed"], cache=True,
                 max_plies=A["max_plies"], n_piecedown=A["n_piecedown"])
    n = len(tr.tok)
    lz = np.load(paths.derived(f"tb_entry_5-4-3men_{n}.npz"), allow_pickle=True)
    vocab = [str(x) for x in lz["vocab"]]
    cls, plies, men = lz["tb_class"].astype(np.int64), lz["tb_plies"], lz["men"]
    cr, ce, co, egs = factor_vocab(vocab)
    eg_ids = np.flatnonzero(np.array([not v.startswith("WDL:") and "=" in v for v in vocab]))
    eg_pos = {int(c): i for i, c in enumerate(eg_ids)}          # class id -> ending index

    is_event = any(k.startswith("ev_kind") for k in ck["state_dict"])
    Cls = ConceptEvent if is_event else ConceptCapture
    net = Cls(d_model=A["d_model"], heads=A["heads"], codes=A["codes"],
              square_codes=A["square_codes"], hidden=A["hidden"], n_res=len(vocab),
              factored=True, n_eg=len(egs), cls_reach=cr, cls_eg=ce, cls_out=co)
    net.load_state_dict(ck["state_dict"]); net.eval()

    gor = tr.game_of_row()
    rs = np.random.default_rng(A["seed"]); perm = rs.permutation(len(tr.start))
    nv = max(1, int(A["val_frac"] * len(tr.start))); nt = max(1, int(A["test_frac"] * len(tr.start)))
    sog = np.zeros(len(tr.start), np.int8); sog[perm[:nv]] = 1; sog[perm[nv:nv + nt]] = 2
    spl = sog[gor]

    def sig_of_row(rows):
        t = tr.tok[rows]
        return [tuple(int((t[i] == v).sum()) for v in range(1, 13)) for i in range(len(rows))]

    print(f"\n  {args.ckpt.split('/')[-1]} step {ck.get('step')} "
          f"({'event' if is_event else 'capture'} model)")
    print(f"  Distribution over {len(eg_ids)} ENDINGS, on rows whose game truly reaches a "
          f"tablebase.\n  95% bootstrap CIs over rows ({args.boot} resamples).\n")
    hdr = (f"  {'h':>3} {'n':>7} | {'top1':>15} {'top3':>15} {'top5':>15} | "
           f"{'logloss':>15} {'ECE':>7}")
    print(hdr)

    rows_by_h = {}
    for H in [int(x) for x in args.horizons.split(",")]:
        band = ((spl == 2) & (cls >= 0) & (plies >= H - args.band) & (plies <= H + args.band)
                & (men > 5))
        rr = np.flatnonzero(band)
        keep = np.array([c in eg_pos for c in cls[rr]])
        rr = rr[keep]
        if len(rr) < 200:
            print(f"  {H:>3} {len(rr):>7} | too few endgame rows at this horizon")
            continue
        rng = np.random.default_rng(0)
        rr = rr[rng.choice(len(rr), min(args.n, len(rr)), replace=False)]
        rows_by_h[H] = rr
        y = np.array([eg_pos[int(c)] for c in cls[rr]])

        P = []
        with torch.no_grad():
            for a in range(0, len(rr), args.batch):
                r = rr[a:a + args.batch]
                zq, _, _, _ = net.encode(torch.from_numpy(tr.tok[r].astype(np.int64)),
                                         torch.from_numpy(tr.glob[r].astype(np.float32)))
                _, e_l, _ = net.predict_resolution(zq)
                P.append(torch.softmax(e_l, -1).numpy())
        P = np.concatenate(P)                                     # (R, n_endings)

        order = np.argsort(-P, axis=1)
        rank = np.array([np.where(order[i] == y[i])[0][0] for i in range(len(y))])
        pt = P[np.arange(len(y)), y]
        ptop = P[np.arange(len(y)), order[:, 0]]
        corr = (rank == 0).astype(float)

        def ci(v):
            rg = np.random.default_rng(1)
            b = [v[rg.integers(0, len(v), len(v))].mean() for _ in range(args.boot)]
            return f"{v.mean():.3f}[{np.percentile(b,2.5):.3f},{np.percentile(b,97.5):.3f}]"

        ll = -np.log(np.clip(pt, 1e-12, None))
        print(f"  {H:>3} {len(rr):>7} | {ci(corr):>15} {ci((rank<3).astype(float)):>15} "
              f"{ci((rank<5).astype(float)):>15} | {ci(ll):>15} "
              f"{ece(ptop, corr):>7.3f}")

    # ---- baselines, as DISTRIBUTIONS, fitted on train --------------------------------
    print(f"\n  BASELINE DISTRIBUTIONS (fitted on TRAIN, scored the same way)")
    print(f"  {'h':>3} | {'marginal top1':>15} {'material top1':>15} | "
          f"{'marginal LL':>13} {'material LL':>13}")
    for H, rr in rows_by_h.items():
        trn = np.flatnonzero((spl == 0) & (cls >= 0) & (plies >= H - args.band)
                             & (plies <= H + args.band) & (men > 5))
        trn = trn[np.array([c in eg_pos for c in cls[trn]])]
        if len(trn) < 500:
            continue
        rng = np.random.default_rng(0)
        trn = trn[rng.choice(len(trn), min(80000, len(trn)), replace=False)]
        ytr = np.array([eg_pos[int(c)] for c in cls[trn]])
        marg = np.bincount(ytr, minlength=len(eg_ids)).astype(float) + 0.5
        marg /= marg.sum()
        tab = defaultdict(lambda: np.zeros(len(eg_ids)))
        for k, yy in zip(sig_of_row(trn), ytr):
            tab[k][yy] += 1
        y = np.array([eg_pos[int(c)] for c in cls[rr]])
        kte = sig_of_row(rr)
        Pm = np.tile(marg, (len(rr), 1))
        Pk = np.stack([(tab[k] + 0.5 * marg) / (tab[k].sum() + 0.5) if k in tab else marg
                       for k in kte])
        t1m = (Pm.argmax(1) == y).mean(); t1k = (Pk.argmax(1) == y).mean()
        llm = -np.log(np.clip(Pm[np.arange(len(y)), y], 1e-12, None)).mean()
        llk = -np.log(np.clip(Pk[np.arange(len(y)), y], 1e-12, None)).mean()
        print(f"  {H:>3} | {t1m:>15.3f} {t1k:>15.3f} | {llm:>13.3f} {llk:>13.3f}")


if __name__ == "__main__":
    main()
