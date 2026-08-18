#!/usr/bin/env python
"""concept_length_field.py -- ONE ruler, all of it length. The B head is gone.

"get rid of the B head is what I'm saying... we already have a committor head prediction, so we
don't even need that. Devote the entire thing to the length head." (Kaveh 2026-08-18)

VERIFIED BEFORE BUILDING, because the whole change rests on it. On the identical held-out probe
and the same 414 resolution classes:

    cef3's EXISTING resolution head   accuracy 0.603   top5 0.816
    csg1's B block, trained for it    accuracy 0.616   top5 0.790

The B block was spending half the geometry's capacity rebuilding a committor the concept model
already carries, and losing top-5 in the process. So it goes, the committor is read from
`ConceptEvent.resolution_logits`, and every dimension here measures LENGTH.

STILL A QUASIMETRIC. The "probability space wants a symmetric metric" argument applies to the
committor, which is exactly the part being removed. Length is not symmetric: you cannot
un-capture a piece, and cqm1 measured d(t+24 -> t) at 8.65x d(t -> t+24) on 98.5% of pairs. That
asymmetry is real structure and the IQE keeps it.

TWO SIGNALS, because each fixes what the other gets wrong -- both measured here:
    QRL      cqm1: rank correlation with k +0.740, asymmetry 8.65x, pole approach -0.980, but
             SUBLINEAR -- 16 plies read 4.80 units, so the ordering was right and the unit
             meaningless.
    ply MSE  csg1's A block: 3.2/4.3/5.9/8.5/11.9 for k=1/2/4/8/16 -- also uncalibrated, and it
             has no reason to spread unrelated states apart at all.
QRL supplies the ordering and the asymmetry; the MSE ties one unit to one ply. The calibration
readout is the fitted slope, which should approach 1.0.

    .venv/bin/python -m ...concept_length_field --codes <npz> --steps 20000 --device mps
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from catspace.research.tools.training_infra.losses import (
    qrl_constraint_violation, qrl_maximise)


class ConceptLengthField(nn.Module):
    """frozen concept state -> projection -> ONE IQE, all of it length."""

    def __init__(self, d_in, d_iqe=512, components=16, hidden=1024, leak_beta=0.0):
        super().__init__()
        from catspace.research.components.encoder.approaches.jepa_tokenizer.src.iqe import IQE
        assert d_iqe % components == 0
        self.proj = nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(),
                                  nn.Linear(hidden, hidden), nn.GELU(),
                                  nn.Linear(hidden, d_iqe))
        self.iqe = IQE(d_iqe, components=components, leak_beta=leak_beta)

    def embed(self, z):
        return self.proj(z)

    def dist(self, za, zb):
        return self.iqe(self.embed(za), self.embed(zb))


def build_step(args, dev):
    from catspace.research.components.encoder.approaches.reach_probability.src import (
        trajectories as T)

    t0 = time.time()
    z = np.load(args.codes, allow_pickle=True)
    g, s = z["g"].astype(np.int64), z["s"].astype(np.int64)
    g_book = torch.from_numpy(z["g_book"]).to(dev)
    s_book = torch.from_numpy(z["s_book"]).to(dev)
    n, H = g.shape
    d_in = H * g_book.shape[-1] + 64 * s_book.shape[-1]
    G = torch.from_numpy(g).to(dev); S = torch.from_numpy(s).to(dev)
    ar_h = torch.arange(H, device=dev)

    def state(rows):
        zq = torch.cat([g_book[ar_h[None, :], G[rows]].reshape(len(rows), -1),
                        s_book[S[rows]].reshape(len(rows), -1)], -1)
        lo = zq.min(-1, keepdim=True).values; hi = zq.max(-1, keepdim=True).values
        return (zq - lo) / (hi - lo).clamp(min=1e-5)

    tr = T.build(n_human=0, n_sf=args.games, seed=args.seed, cache=True,
                 max_plies=args.max_plies, n_piecedown=args.n_piecedown)
    assert len(tr.tok) == n
    gor = tr.game_of_row()
    rs = np.random.default_rng(args.seed); perm = rs.permutation(len(tr.start))
    nt = max(1, int(0.1 * len(tr.start)))
    test_g = np.zeros(len(tr.start), bool); test_g[perm[:nt]] = True
    is_test = test_g[gor]
    ok = np.zeros(n, bool); ok[:-1] = gor[1:] == gor[:-1]
    adj = np.flatnonzero(ok & (~is_test))
    free = np.flatnonzero(~is_test)
    print(f"[clf] {n:,} cached states | d_in {d_in} -> IQE {args.d_iqe} "
          f"({args.components} components) | adjacent {len(adj):,} | free {len(free):,} "
          f"[{time.time()-t0:.0f}s]", flush=True)

    net = ConceptLengthField(d_in, d_iqe=args.d_iqe, components=args.components,
                             hidden=args.hidden, leak_beta=args.leak_beta).to(dev)
    print(f"[clf] {sum(p.numel() for p in net.parameters()):,} params", flush=True)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    lam = torch.tensor(float(args.lam0), device=dev)
    rng = np.random.default_rng(args.seed)

    te = np.flatnonzero(is_test)
    probe = te[te + args.k_max < n]
    probe = probe[gor[probe + args.k_max] == gor[probe]]
    probe = probe[rng.choice(len(probe), min(3000, len(probe)), replace=False)]
    PB = torch.from_numpy(probe).to(dev)

    def step_fn(model, step):
        nonlocal lam
        model.train()
        # --- QRL: one ply costs one unit, and unrelated states are pushed apart -----------
        a = torch.from_numpy(adj[rng.integers(0, len(adj), args.batch)]).to(dev)
        d_adj = model.dist(state(a), state(a + 1))
        viol = qrl_constraint_violation(d_adj, ceiling=args.step_cost, eps=args.eps)
        i = torch.from_numpy(free[rng.integers(0, len(free), args.batch)]).to(dev)
        j = torch.from_numpy(free[rng.integers(0, len(free), args.batch)]).to(dev)
        d_far = model.dist(state(i), state(j))
        obj = qrl_maximise(d_far, knee=args.knee, beta=args.phi_beta)

        # --- calibration: one unit should BE one ply over multi-ply gaps ------------------
        p0 = adj[rng.integers(0, len(adj), args.batch)]
        k = rng.integers(1, args.k_max + 1, len(p0))
        p1 = np.clip(p0 + k, 0, n - 1)
        keep = gor[p1] == gor[p0]
        if keep.sum() > 8:
            A_ = torch.from_numpy(p0[keep]).to(dev); B_ = torch.from_numpy(p1[keep]).to(dev)
            l_cal = F.mse_loss(model.dist(state(A_), state(B_)),
                               torch.from_numpy(k[keep].astype(np.float32)).to(dev))
        else:
            l_cal = torch.zeros((), device=dev)

        loss = obj + lam.detach() * viol + args.w_cal * l_cal
        opt.zero_grad(); loss.backward(); opt.step()
        lam = (lam + args.lr_lam * viol.detach()).clamp(min=0.0, max=args.lam_max)

        m = {}
        if step % args.eval_every == 0 or step == 1:
            model.eval()
            with torch.no_grad():
                zp = model.embed(state(PB))
                ks = [1, 2, 4, 8, 16, args.k_max]
                dk, dd, kk = [], [], []
                for kv in ks:
                    dv = model.iqe(zp, model.embed(state(PB + kv))).cpu().numpy()
                    dk.append(float(dv.mean())); dd.append(dv); kk.append(np.full(len(dv), kv))
                from scipy.stats import spearmanr
                rho = float(spearmanr(np.concatenate(kk), np.concatenate(dd)).statistic)
                slope = float(np.polyfit(np.concatenate(kk), np.concatenate(dd), 1)[0])
                rev = float(model.iqe(model.embed(state(PB + args.k_max)), zp).mean())
                self_d = float(model.iqe(zp, zp).abs().max())
            el = time.time() - t0
            print(f"[clf] step {step:>6}/{args.steps} {el/60:>5.1f}m "
                  f"({el/max(1,step):.3f}s/step) | d_adj {float(d_adj.mean()):.3f} "
                  f"viol {float(viol):+.4f} lam {float(lam):.1f} cal {float(l_cal):.2f} | "
                  f"d@k " + "/".join(f"{x:.1f}" for x in dk) +
                  f" | slope {slope:.3f} rho {rho:+.3f} | rev/fwd "
                  f"{rev/max(dk[-1],1e-9):.1f}x self {self_d:.1e}", flush=True)
            m = {"d_adj": float(d_adj.mean()), "viol": float(viol), "lam": float(lam),
                 "slope": slope, "rho_k": rho, "rev_ratio": rev / max(dk[-1], 1e-9),
                 "cal": float(l_cal), **{f"d_k{kv}": v for kv, v in zip(ks, dk)}}
            model.train()
        return {"loss": float(loss.detach()), "obj": float(obj.detach()),
                "violation": float(viol.detach()), **m}

    return net, step_fn


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--codes", required=True)
    ap.add_argument("--games", type=int, default=20000)
    ap.add_argument("--n-piecedown", type=int, default=45906)
    ap.add_argument("--max-plies", type=int, default=400)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--d-iqe", type=int, default=512)
    ap.add_argument("--components", type=int, default=16)
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--leak-beta", type=float, default=0.0)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--step-cost", type=float, default=1.0)
    ap.add_argument("--eps", type=float, default=0.25)
    ap.add_argument("--knee", type=float, default=60.0)
    ap.add_argument("--phi-beta", type=float, default=0.03)
    ap.add_argument("--w-cal", type=float, default=0.05,
                    help="ply calibration. QRL fixes the ORDER, this fixes the UNIT.")
    ap.add_argument("--lam0", type=float, default=1.0)
    ap.add_argument("--lr-lam", type=float, default=0.01)
    ap.add_argument("--lam-max", type=float, default=1e4)
    ap.add_argument("--k-max", type=int, default=24)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--ckpt-every", type=int, default=1000)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="artifacts/experiments/clf1")
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    from catspace.research.tools.training_infra.train.scaffold import (
        TrainConfig, standard_train)
    net, step_fn = build_step(args, args.device)
    cfg = TrainConfig(out=args.out, steps=args.steps, ckpt_every=args.ckpt_every,
                      eval_every=args.eval_every, experiment="catspace",
                      run_name=args.out.split("/")[-1], device=args.device)
    standard_train(step_fn, net, cfg, args)


if __name__ == "__main__":
    main()
