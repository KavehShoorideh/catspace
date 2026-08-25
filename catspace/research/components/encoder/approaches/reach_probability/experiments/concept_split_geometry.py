#!/usr/bin/env python
"""concept_split_geometry.py -- the split A/B IQE, rebuilt on the FROZEN CONCEPT SET.

"the previous IQE... would take two sets of blocks, a and b, and a would be length, and b would
be a committor. What we want is now, instead of the trunk, use the concept set. And off of the
concept set, train the IQE that has two blocks, and then reuse everything that was used there."
(Kaveh 2026-08-18)

WHAT IS REUSED, verbatim in structure from `concept_state_field.py`:
    A block   iqe_a on the FIRST half of the state -- the LENGTH ruler, graded by MSE against
              the observed ply gap. Deliberately weak, per the original instruction to "nail
              down the ply count poorly but at least get the committor right".
    B block   iqe_b on the SECOND half -- the OUTCOME ruler. One learned pole per resolution
              class; the committor is softmax(-dB(z, pole_k)/temp), so the GEOMETRY IS the
              committor and there is no separate WDL head to keep in sync.
    poles + learnable temperature, with the measured init discipline below.

WHAT CHANGES: the input. Previously board -> trunk -> VQ -> state, all trained together. Here
the concepts are FROZEN (cef3) and cached as codes, and the state is a codebook gather. Stage 1
decided what a concept is; this stage only learns the geometry over it, and cannot reach back and
reshape the vocabulary to make its own job easier.

POLE INIT AND TEMPERATURE ARE ONE CONSTRAINT, NOT TWO KNOBS. Carried over because it was paid
for: csf7 sat at chance for 2,000 steps at (init 0.5, T 5.0), and the diagnosis was that placing
z EXACTLY on its target pole -- the most favourable case possible -- would also have scored 0.6%
against a 0.4% chance rate. Hundreds of random points in a high-dimensional space are all nearly
equidistant, so the softmax stays flat and flat softmaxes give the poles no gradient. Measured:
(0.5, 5.0) -> 0.006, (0.5, 0.25) -> 0.983, (2.0, 1.0) -> 0.986. Defaults are (2.0, 1.0).

WHY THE COMMITTOR IS THE POINT. The pure-QRL geometry (cqm1) learned "how far" almost perfectly
-- pole approach -0.980, odometer +0.740 -- and barely learned "toward what": AUC 0.655 for
which pole is ours, because nothing in its objective ever distinguished a white-win terminal from
a black-win one. The B block exists to fix exactly that, and this run is scored against those
numbers.

    .venv/bin/python -m ...concept_split_geometry --codes <npz> --steps 20000 --device mps
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConceptSplitGeometry(nn.Module):
    """frozen concept state -> projection -> two IQE blocks (length | outcome) + poles."""

    def __init__(self, d_in, d_state=256, components=16, n_poles=414, proj_h=512,
                 pole_init=2.0, temp_init=1.0):
        super().__init__()
        from catspace.research.components.encoder.approaches.jepa_tokenizer.src.iqe import IQE
        h = d_state // 2
        assert d_state % 2 == 0 and h % components == 0, (
            f"d_state={d_state} splits into two {h}-d rulers; each must divide into "
            f"{components} components")
        self.proj = nn.Sequential(nn.Linear(d_in, proj_h), nn.GELU(),
                                  nn.Linear(proj_h, proj_h), nn.GELU(),
                                  nn.Linear(proj_h, d_state))
        self.iqe_a = IQE(h, components=components)      # LENGTH ruler (plies)
        self.iqe_b = IQE(h, components=components)      # OUTCOME ruler -> committor
        self.n_poles = int(n_poles)
        self.poles = nn.Parameter(torch.randn(self.n_poles, d_state) * pole_init)
        self.log_temp = nn.Parameter(torch.tensor(float(np.log(temp_init))))

    def embed(self, z):
        return self.proj(z)

    def dA(self, u, v):
        k = u.shape[-1] // 2
        return self.iqe_a(u[..., :k], v[..., :k])

    def dB(self, u, v):
        k = u.shape[-1] // 2
        return self.iqe_b(u[..., k:], v[..., k:])

    def resolution(self, z, temp=None):
        """-> (N, n_poles) P(this game resolves at pole k), from the OUTCOME ruler.
        One batched call, not a python loop over poles."""
        n, P = len(z), self.n_poles
        zz = z.repeat_interleave(P, 0)
        pp = self.poles.unsqueeze(0).expand(n, -1, -1).reshape(n * P, -1)
        d = self.dB(zz, pp).view(n, P)
        t = self.log_temp.exp() if temp is None else torch.as_tensor(temp)
        return torch.softmax(-d / t.clamp(min=1e-3), 1)


def build_step(args, dev):
    from catspace.io import paths
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
    lz = np.load(paths.derived(f"tb_entry_5-4-3men_{n}.npz"), allow_pickle=True)
    vocab = [str(x) for x in lz["vocab"]]
    cls = lz["tb_class"].astype(np.int64)
    gor = tr.game_of_row()
    rs = np.random.default_rng(args.seed); perm = rs.permutation(len(tr.start))
    nt = max(1, int(0.1 * len(tr.start)))
    test_g = np.zeros(len(tr.start), bool); test_g[perm[:nt]] = True
    is_test = test_g[gor]

    lab = np.flatnonzero((~is_test) & (cls >= 0))
    ok = np.zeros(n, bool); ok[:-1] = gor[1:] == gor[:-1]
    pair_rows = np.flatnonzero(ok & (~is_test))
    print(f"[csg] {n:,} cached states | d_in {d_in} | {len(vocab)} poles | "
          f"labelled train rows {len(lab):,} | ply-pair rows {len(pair_rows):,} "
          f"[{time.time()-t0:.0f}s]", flush=True)

    net = ConceptSplitGeometry(d_in, d_state=args.d_state, components=args.components,
                               n_poles=len(vocab), proj_h=args.proj_h,
                               pole_init=args.pole_init, temp_init=args.temp_init).to(dev)
    print(f"[csg] {sum(p.numel() for p in net.parameters()):,} params | A/B rulers "
          f"{args.d_state//2}-d x {args.components} components | pole_init {args.pole_init} "
          f"temp_init {args.temp_init}", flush=True)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    rng = np.random.default_rng(args.seed)
    CLS = torch.from_numpy(cls).to(dev)

    te = np.flatnonzero(is_test & (cls >= 0))
    probe = te[rng.choice(len(te), min(4000, len(te)), replace=False)]
    PB = torch.from_numpy(probe).to(dev)
    maj = np.bincount(cls[lab], minlength=len(vocab)).argmax()
    base = float((cls[probe] == maj).mean())

    def step_fn(model, step):
        model.train()
        # --- B block: the committor, graded on the true resolution class -----------------
        r = lab[rng.integers(0, len(lab), args.batch)]
        R = torch.from_numpy(r).to(dev)
        zb = model.embed(state(R))
        l_out = F.nll_loss((model.resolution(zb) + 1e-8).log(), CLS[R])

        # --- A block: the LENGTH ruler, MSE against the observed ply gap ------------------
        a = pair_rows[rng.integers(0, len(pair_rows), args.batch)]
        k = rng.integers(1, args.ply_gap + 1, len(a))
        b = np.clip(a + k, 0, n - 1)
        keep = gor[b] == gor[a]
        if keep.sum() > 8:
            A_ = torch.from_numpy(a[keep]).to(dev); B_ = torch.from_numpy(b[keep]).to(dev)
            za = model.embed(state(A_)); zb2 = model.embed(state(B_))
            l_ply = F.mse_loss(model.dA(za, zb2),
                               torch.from_numpy(k[keep].astype(np.float32)).to(dev))
        else:
            l_ply = torch.zeros((), device=dev)

        loss = args.w_out * l_out + args.w_ply * l_ply
        opt.zero_grad(); loss.backward(); opt.step()

        m = {}
        if step % args.eval_every == 0 or step == 1:
            model.eval()
            with torch.no_grad():
                zp = model.embed(state(PB))
                p = model.resolution(zp)
                acc = float((p.argmax(1) == CLS[PB]).float().mean())
                top5 = float((p.topk(5, 1).indices == CLS[PB][:, None]).any(1).float().mean())
                conf = float(p.max(1).values.mean())
                # is the A ruler calibrated? d should track k
                kk = [1, 2, 4, 8, 16]
                dk = []
                for kv in kk:
                    q = probe[probe + kv < n]
                    q = q[gor[q + kv] == gor[q]][:1500]
                    if len(q) > 50:
                        dk.append(float(model.dA(model.embed(state(torch.from_numpy(q).to(dev))),
                                                 model.embed(state(torch.from_numpy(q+kv).to(dev)))
                                                 ).mean()))
                    else:
                        dk.append(float("nan"))
                pol = model.poles.detach()
                sep = float(torch.cdist(pol, pol).add(
                    torch.eye(len(pol), device=dev) * 1e9).min())
            el = time.time() - t0
            print(f"[csg] step {step:>6}/{args.steps} {el/60:>5.1f}m "
                  f"({el/max(1,step):.3f}s/step) | out {float(l_out.detach()):.3f} "
                  f"ply {float(l_ply.detach()):.3f} | committor acc {acc:.3f} "
                  f"(base {base:.3f}) top5 {top5:.3f} conf {conf:.3f} | dA@k " +
                  "/".join(f"{x:.1f}" for x in dk) +
                  f" | T {float(model.log_temp.exp()):.2f} pole_sep {sep:.2f}", flush=True)
            m = {"committor_acc": acc, "committor_top5": top5, "conf": conf,
                 "temp": float(model.log_temp.exp()), "pole_sep": sep,
                 **{f"dA_k{kv}": v for kv, v in zip(kk, dk)}}
            model.train()
        return {"loss": float(loss.detach()), "out": float(l_out.detach()),
                "ply": float(l_ply.detach()), **m}

    step_fn.opt = opt   # standard_train checkpoints optimizer state from here (resumability)
    return net, step_fn


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--codes", required=True)
    ap.add_argument("--games", type=int, default=20000)
    ap.add_argument("--n-piecedown", type=int, default=45906)
    ap.add_argument("--max-plies", type=int, default=400)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--d-state", type=int, default=256)
    ap.add_argument("--components", type=int, default=16)
    ap.add_argument("--proj-h", type=int, default=512)
    ap.add_argument("--pole-init", type=float, default=2.0,
                    help="JOINT constraint with temp: (0.5,5.0) measured at chance, "
                         "(2.0,1.0) at 0.986")
    ap.add_argument("--temp-init", type=float, default=1.0)
    ap.add_argument("--ply-gap", type=int, default=16)
    ap.add_argument("--w-out", type=float, default=1.0)
    ap.add_argument("--w-ply", type=float, default=0.1,
                    help="deliberately weak -- 'nail the ply count poorly, get the committor "
                         "right'")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--ckpt-every", type=int, default=1000)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="artifacts/experiments/csg1")
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
