#!/usr/bin/env python
"""experiments/train_jepa_t1.py -- T1 of the anchored JEPA (Kaveh's draft §3.3, §3.7):
train encoder + ALL THREE loss terms jointly on the human corpus. No A/B staging
(Kaveh 2026-07-30 rule): this is the best-shot configuration, complete.

  L = a_dyn * || P_dyn(phi(s), a) - sg[phi_EMA(s')] ||^2            (world model)
    + a_haz * masked-censored BCE of the ANY-event hazard            (JEPA energy,
              aggregate kappa_0 at T1; per-atom keys arrive at T2-T4)
    + a_dest * [ CE(d(s_t), sg d(s_{t+1}))  +  CE(d(s_b), tb one-hot) ]   (anchored
              destination: along-game bootstrap + EXACT Syzygy clamp at the boundary)

Anti-collapse guards (paper Fig 3b): EMA target, stop-gradient, weaker predictor,
labelled terms, exact terminal clamp. Collapse GATE (standing rule): effective rank
+ eigenspectrum of phi printed at every eval; a slide toward rank ~1 aborts.

VERDICT lines on held-out rows: per-term losses, dest clamp accuracy, any-event
hazard NLL vs train-marginal baseline, eff_rank.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from catspace.diagnostics import eff_rank                               # noqa: E402
from catspace.encoder.jepa import JepaT1                                # noqa: E402
from catspace.train.scaffold import resolve_device                      # noqa: E402

GAPS = np.array([1, 2, 3, 5, 8, 13, 21, 34])


def bucket_of(dec):
    return np.searchsorted(GAPS, dec, side="left").clip(0, len(GAPS) - 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/derived/checkpoints/jepa_t1_corpus.npz")
    ap.add_argument("--out", default="artifacts/experiments/jepa_t1")
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--batch", type=int, default=192)
    ap.add_argument("--bd-batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--a-dyn", type=float, default=1.0)
    ap.add_argument("--a-haz", type=float, default=1.0)
    ap.add_argument("--a-dest", type=float, default=1.0)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--ckpt-every", type=int, default=5000)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time(); rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    dev = resolve_device("auto")

    d = dict(np.load(args.data, allow_pickle=True))
    NC = int(d["bd_class"].max()) + 1 if len(d["bd_class"]) else 151
    T = {k: torch.as_tensor(v) for k, v in d.items()
         if k.startswith(("tr_", "bd_", "cx_")) and v.dtype != object}
    n_tr, n_bd, n_cx = len(d["tr_tok"]), len(d["bd_tok"]), len(d["cx_tok"])
    ho_tr = rng.random(n_tr) < 0.05
    ho_bd = rng.random(n_bd) < 0.10
    ho_cx = d["cx_gid"] % 10 == 0
    j_star = np.where(d["cx_gap_dec"] > 0, bucket_of(np.maximum(d["cx_gap_dec"], 1)), -1)
    e_cens = bucket_of(np.maximum(d["cx_end_dec"], 1))
    occurred = d["cx_gap_dec"] > 0
    cx_ctx = np.stack([(d["cx_elo_victim"] - 1500) / 400,
                       (d["cx_elo_opp"] - 1500) / 400], -1).astype(np.float32)
    print(f"rows: trans {n_tr:,} (ho {ho_tr.sum():,}) | boundary {n_bd:,} | "
          f"contexts {n_cx:,} (occurred {occurred.mean():.1%}) | classes {NC}")

    model = JepaT1(d=args.d, layers=args.layers, n_class=NC).to(dev)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)

    def enc(tok, glob, rows, target=False):
        tk = tok[rows].to(dev); gl = glob[rows].to(dev)
        return (model.tgt if target else model.enc)(tk, gl)

    tr_idx = np.flatnonzero(~ho_tr); bd_idx = np.flatnonzero(~ho_bd)
    cx_idx = np.flatnonzero(~ho_cx)
    H = len(GAPS)
    hh = torch.arange(H, device=dev)[None, :]

    def losses(rt, rb, rc):
        rt_t = torch.as_tensor(rt); rb_t = torch.as_tensor(rb); rc_t = torch.as_tensor(rc)
        # L_dyn + destination bootstrap on transitions
        phi_s = enc(T["tr_tok"], T["tr_glob"], rt_t)
        with torch.no_grad():
            phi_s1_t = enc(T["tr_tok1"], T["tr_glob1"], rt_t, target=True)
            dlog_s1 = model.dest(model.enc(T["tr_tok1"][rt_t].to(dev),
                                           T["tr_glob1"][rt_t].to(dev))).flatten(1)
        pred = model.dyn(phi_s, T["tr_act"][rt_t].to(dev))
        l_dyn = F.mse_loss(pred, phi_s1_t)
        l_boot = F.kl_div(F.log_softmax(model.dest(phi_s).flatten(1), -1),
                          F.softmax(dlog_s1, -1), reduction="batchmean")
        # destination clamp on boundary rows
        phi_b = enc(T["bd_tok"], T["bd_glob"], rb_t)
        tgt = (T["bd_class"][rb_t] * 3 + T["bd_wdl"][rb_t]).long().to(dev)
        l_clamp = F.cross_entropy(model.dest(phi_b).flatten(1), tgt)
        # any-event hazard on contexts
        phi_c = enc(T["cx_tok"], T["cx_glob"], rc_t)
        lg = model.haz(phi_c, torch.as_tensor(cx_ctx[rc], device=dev))
        occ = torch.as_tensor(occurred[rc], device=dev)
        js = torch.as_tensor(j_star[rc], device=dev)
        es = torch.as_tensor(e_cens[rc], device=dev)
        m = torch.where(occ[:, None], hh <= js[:, None], hh <= es[:, None])
        y = (occ[:, None] & (hh == js[:, None])).float()
        l_haz = (F.binary_cross_entropy_with_logits(lg, y, reduction="none") * m
                 ).sum() / m.sum().clamp(min=1)
        return l_dyn, l_haz, l_boot + l_clamp

    model.train()
    for step in range(1, args.steps + 1):
        rt = rng.choice(tr_idx, args.batch)
        rb = rng.choice(bd_idx, min(args.bd_batch, len(bd_idx)))
        rc = rng.choice(cx_idx, args.batch)
        l_dyn, l_haz, l_dest = losses(rt, rb, rc)
        L = args.a_dyn * l_dyn + args.a_haz * l_haz + args.a_dest * l_dest
        opt.zero_grad(); L.backward(); opt.step(); sched.step()
        model.ema_update()
        if step % args.eval_every == 0:
            model.eval()
            with torch.no_grad():
                re = torch.as_tensor(rng.choice(np.flatnonzero(ho_cx), 1024))
                phi_e = enc(T["cx_tok"], T["cx_glob"], re)
                er = eff_rank(phi_e.cpu().numpy())
                # dest clamp acc on held-out boundary
                rbh = torch.as_tensor(np.flatnonzero(ho_bd)[:2048])
                acc = float((model.dest(enc(T["bd_tok"], T["bd_glob"], rbh)).flatten(1)
                             .argmax(1).cpu() ==
                             (T["bd_class"][rbh] * 3 + T["bd_wdl"][rbh]).long()).float()
                            .mean()) if len(rbh) else float("nan")
            print(f"  step {step} | dyn {float(l_dyn):.4f} haz {float(l_haz):.4f} "
                  f"dest {float(l_dest):.4f} | eff_rank(phi) {er:.1f}/{args.d} | "
                  f"clamp-acc {acc:.3f} [{time.time()-t0:.0f}s]", flush=True)
            assert er > args.d * 0.02, f"COLLAPSE GATE: eff_rank {er:.1f}"
            model.train()
        if step % args.ckpt_every == 0 or step == args.steps:
            torch.save({"state_dict": model.state_dict(),
                        "cfg": dict(d=args.d, layers=args.layers, n_class=NC,
                                    H=H, gaps=GAPS.tolist()),
                        "meta": dict(data=args.data, step=step, seed=args.seed)},
                       f"{args.out}_step{step}.pt")
            torch.save(torch.load(f"{args.out}_step{step}.pt", weights_only=False),
                       f"{args.out}_latest.pt")

    # ---- final held-out verdicts ------------------------------------------------------
    model.eval()
    with torch.no_grad():
        hoc = np.flatnonzero(ho_cx & occurred)
        marg = np.bincount(j_star[cx_idx][occurred[cx_idx]], minlength=H) + 1.0
        marg = marg / marg.sum()
        nll = []
        for i0 in range(0, len(hoc), 2048):
            r = hoc[i0:i0 + 2048]
            phi_c = enc(T["cx_tok"], T["cx_glob"], torch.as_tensor(r))
            lam = torch.sigmoid(model.haz(phi_c, torch.as_tensor(cx_ctx[r], device=dev)))
            S = torch.cumprod(1 - lam, 1)
            f_ = lam * torch.cat([torch.ones(len(r), 1, device=dev), S[:, :-1]], 1)
            js = torch.as_tensor(j_star[r], device=dev)
            nll.append(-torch.log(f_.gather(1, js[:, None]).clamp(1e-9)).squeeze(1)
                       .cpu().numpy())
        nll = np.concatenate(nll)
        base = float(-np.log(marg[j_star[hoc]]).mean())
        print(f"VERDICT T1 any-event timing: NLL {nll.mean():.4f} vs marginal "
              f"{base:.4f} | lift {base - nll.mean():+.4f} nats "
              f"({'PASS' if base - nll.mean() > 0 else 'not better'})")
    print(f"DONE train_jepa_t1 [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
