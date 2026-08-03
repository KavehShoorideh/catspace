#!/usr/bin/env python
"""catspace/research/components/encoder/approaches/jepa_tokenizer/experiments/train_hazard_head.py -- Stage 2 of the anchored-JEPA plan (Kaveh's draft
§3.3-3.4): the CENSORED DISCRETE-TIME HAZARD head over the checkpoint corpus.

    lambda_g(h | s, omega) = sigma( <q_h(phi(s), omega), kappa_g> / sqrt(d_r) + b_g )

v0 scoping (each deferral is the paper's own fallback): frozen-Leela-trunk phi (no
JEPA encoder yet -- the encoder question stays a separable A/B); atoms = the 1024 v4
region centroids (the sparse atom layer enters only if it beats this rung of the
representation ladder); omega = (Elo_victim, Elo_opp), z later.

Loss = masked BCE over (atom, bucket) cells (paper eq. 4): the true atom contributes
y=1 at its occurrence bucket and 0 before; sampled negative atoms contribute 0 up to
the row's censoring bucket and SILENCE after (censored pairs are never negatives).
An aggregate any-event key kappa_0 gets the same likelihood (dense gradient).

Identities recovered at eval (paper eq. 6-7): S_g, f_g, R_g, gamma_g.

VERDICTS (held-out games, gid%10==0):
  timing  : -log f_g*(j*) on occurred pairs vs the train-split marginal-timing baseline
  ranking : AUC of R_g(H) true-atom vs 63 sampled negatives per held-out context
  calib   : any-event R(h) vs realized occurrence frequency per bucket (max |gap|)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from catspace.research.tools.training_infra.train.scaffold import resolve_device                      # noqa: E402
from catspace.io import paths

GAPS = np.array([1, 2, 3, 5, 8, 13, 21, 34])                            # bucket edges = H=8


class HazardHead(nn.Module):
    def __init__(self, d_phi=64, d_ctx=2, d_r=64, H=8, G=1024):
        super().__init__()
        self.H, self.d_r = H, d_r
        self.trunk = nn.Sequential(nn.Linear(d_phi + d_ctx, 256), nn.ReLU(),
                                   nn.Linear(256, 256), nn.ReLU())
        self.query = nn.Linear(256, H * d_r)
        self.wk = nn.Linear(d_phi, d_r, bias=False)          # atom keys from centroids
        self.b_g = nn.Parameter(torch.zeros(G))
        self.k_any = nn.Parameter(torch.randn(d_r) * 0.02)   # aggregate "any event" key
        self.b_any = nn.Parameter(torch.zeros(1))

    def queries(self, phi, ctx):
        q = self.query(self.trunk(torch.cat([phi, ctx], -1)))
        return q.view(-1, self.H, self.d_r)                  # (B, H, d_r)

    def lam_logits(self, q, keys, b):
        return torch.einsum("bhd,gd->bhg", q, keys) / self.d_r**0.5 + b

    def lam_any(self, q):
        return (q @ self.k_any) / self.d_r**0.5 + self.b_any # (B, H)


def bucket_of(dec):
    """victim-decision count -> bucket idx (largest GAPS value <= dec covers it)."""
    return np.searchsorted(GAPS, dec, side="left").clip(0, len(GAPS) - 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=paths.derived("checkpoints/checkpoints_v1_emb.npz"))
    ap.add_argument("--out", default=paths.experiment("hazard_v0"))
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--neg", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time(); rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    dev = resolve_device("auto")

    d = dict(np.load(args.data, allow_pickle=True))
    bank = torch.as_tensor(d["bank"], dtype=torch.float32, device=dev)
    G = len(bank); H = len(GAPS)
    phi = torch.as_tensor(d["cx_phi"], dtype=torch.float32, device=dev)
    ctx = torch.stack([(torch.as_tensor(d["cx_elo_victim"], dtype=torch.float32, device=dev) - 1500) / 400,
                       (torch.as_tensor(d["cx_elo_opp"], dtype=torch.float32, device=dev) - 1500) / 400], -1)
    gap = d["cx_gap_dec"]; ckrow = d["cx_ckpt_row"]
    occurred = gap > 0
    j_star = np.where(occurred, bucket_of(np.maximum(gap, 1)), -1)      # occurrence bucket
    e_cens = bucket_of(np.maximum(d["cx_end_dec"], 1))                  # exposure bucket
    g_star = np.where(ckrow >= 0, d["ck_region"][np.maximum(ckrow, 0)], -1)
    gid = d["cx_gid"]
    is_ev = gid % 10 == 0
    tr = np.flatnonzero(~is_ev); ev = np.flatnonzero(is_ev)
    print(f"rows: train {len(tr):,} eval {len(ev):,} | occurred {occurred.mean():.1%} | "
          f"goals {G} buckets {H}")

    model = HazardHead(G=G).to(dev)
    base = max(occurred[~is_ev].mean() / (G * H) * 4, 1e-5)             # rough cell base rate
    with torch.no_grad():
        model.b_g.fill_(float(np.log(base / (1 - base))))
        model.b_any.fill_(float(np.log(0.02 / 0.98)))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    def loss_of(rows):
        r = torch.as_tensor(rows, device=dev)
        q = model.queries(phi[r], ctx[r])
        keys = model.wk(bank)                                            # (G, d_r)
        B = len(rows)
        gs = torch.as_tensor(np.where(g_star[rows] >= 0, g_star[rows], 0), device=dev)
        js = torch.as_tensor(j_star[rows], device=dev)
        es = torch.as_tensor(e_cens[rows], device=dev)
        occ = torch.as_tensor(occurred[rows], device=dev)
        hh = torch.arange(H, device=dev)[None, :]                        # (1, H)
        # true atom cells: y=1 at j*, 0 before, masked after (only occurred rows)
        lg_true = (torch.einsum("bhd,bd->bh", q, keys[gs]) / model.d_r**0.5
                   + model.b_g[gs, None])
        m_true = (hh <= js[:, None]) & occ[:, None]
        y_true = (hh == js[:, None]).float()
        l_true = nn.functional.binary_cross_entropy_with_logits(
            lg_true, y_true, reduction="none")[m_true].sum()
        # negatives: sampled atoms, y=0 up to censoring bucket (never after)
        neg = torch.as_tensor(rng.integers(0, G, (B, args.neg)), device=dev)
        kneg = keys[neg]                                                 # (B, neg, d_r)
        lg_neg = torch.einsum("bhd,bnd->bhn", q, kneg) / model.d_r**0.5 \
            + model.b_g[neg].unsqueeze(1)
        m_neg = (hh[..., None] <= es[:, None, None])
        l_neg = nn.functional.binary_cross_entropy_with_logits(
            lg_neg, torch.zeros_like(lg_neg), reduction="none")[m_neg.expand_as(lg_neg)].sum()
        # aggregate any-event key
        lg_any = model.lam_any(q)
        m_any = torch.where(occ[:, None], hh <= js[:, None], hh <= es[:, None])
        y_any = (occ[:, None] & (hh == js[:, None])).float()
        l_any = nn.functional.binary_cross_entropy_with_logits(
            lg_any, y_any, reduction="none")[m_any].sum()
        return (l_true + l_neg / args.neg + l_any) / B

    model.train()
    for step in range(1, args.steps + 1):
        rows = rng.choice(tr, args.batch)
        opt.zero_grad(); L = loss_of(rows); L.backward(); opt.step()
        if step % 200 == 0:
            print(f"  step {step} loss {float(L):.4f} [{time.time()-t0:.0f}s]", flush=True)

    # ---- VERDICTS on held-out games --------------------------------------------------
    model.eval()
    with torch.no_grad():
        keys = model.wk(bank)
        evo = ev[occurred[ev]]
        # timing NLL of f_g*(j*) vs train marginal baseline
        nll, auc_hits = [], []
        marg = np.bincount(j_star[tr][occurred[tr]], minlength=H) + 1.0
        marg = marg / marg.sum()
        for i0 in range(0, len(evo), 2048):
            r = evo[i0:i0 + 2048]
            rt = torch.as_tensor(r, device=dev)
            q = model.queries(phi[rt], ctx[rt])
            gs = torch.as_tensor(g_star[r], device=dev)
            lam = torch.sigmoid(torch.einsum("bhd,bd->bh", q, keys[gs])
                                / model.d_r**0.5 + model.b_g[gs, None])
            S = torch.cumprod(1 - lam, 1)
            Sprev = torch.cat([torch.ones(len(r), 1, device=dev), S[:, :-1]], 1)
            f = lam * Sprev
            js = torch.as_tensor(j_star[r], device=dev)
            nll.append(-torch.log(f.gather(1, js[:, None]).clamp(1e-9)).squeeze(1).cpu().numpy())
            # ranking: R_g(H) of true atom vs 63 negatives
            neg = torch.as_tensor(rng.integers(0, G, (len(r), 63)), device=dev)
            def R_of(ks, bs):
                lamn = torch.sigmoid(torch.einsum("bhd,bnd->bhn", q, ks)
                                     / model.d_r**0.5 + bs.unsqueeze(1))
                return 1 - torch.prod(1 - lamn, 1)
            Rn = R_of(keys[neg], model.b_g[neg])
            Rt = 1 - torch.prod(1 - lam, 1)
            auc_hits.append((Rt[:, None] > Rn.squeeze(-1) if Rn.dim() > 2 else Rt[:, None] > Rn
                             ).float().mean(1).cpu().numpy())
        nll = np.concatenate(nll); auc = float(np.concatenate(auc_hits).mean())
        base_nll = float(-np.log(marg[j_star[evo]]).mean())
        print(f"VERDICT hazard timing: NLL {nll.mean():.4f} vs marginal baseline "
              f"{base_nll:.4f} | lift {base_nll - nll.mean():+.4f} nats "
              f"({'PASS' if base_nll - nll.mean() > 0 else 'not better'})")
        print(f"VERDICT hazard ranking: true-atom vs 63 negatives AUC {auc:.3f} "
              f"(0.5 = chance)")
        # any-event calibration per bucket
        re = torch.as_tensor(ev, device=dev)
        gaps_line = []
        for i0 in range(0, len(ev), 4096):
            r = re[i0:i0 + 4096]
            q = model.queries(phi[r], ctx[r])
            lam = torch.sigmoid(model.lam_any(q))
            R = (1 - torch.cumprod(1 - lam, 1)).cpu().numpy()
            gaps_line.append(R)
        R = np.concatenate(gaps_line)
        occ_ev = occurred[ev]; j_ev = j_star[ev]; e_ev = e_cens[ev]
        line = []
        for h in range(H):
            seen = occ_ev & (j_ev <= h)
            atrisk = (e_ev >= h) | occ_ev
            line.append((float(R[atrisk, h].mean()), float(seen[atrisk].mean())))
        mx = max(abs(a - b) for a, b in line)
        print("VERDICT any-event calibration (pred vs realized by bucket): "
              + " ".join(f"{a:.2f}/{b:.2f}" for a, b in line) + f" | max|gap| {mx:.3f}")
    out = Path(args.out)
    torch.save({"state_dict": model.state_dict(),
                "cfg": dict(d_phi=64, d_ctx=2, d_r=64, H=H, G=G, gaps=GAPS.tolist()),
                "meta": dict(data=args.data, steps=args.steps, seed=args.seed)},
               f"{out}_latest.pt")
    print(f"saved {out}_latest.pt [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
