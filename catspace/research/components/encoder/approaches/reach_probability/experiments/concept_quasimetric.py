#!/usr/bin/env python
"""concept_quasimetric.py -- a quasimetric over CONCEPT states, trained with QRL.

"train a quasimetric on top of the concept states, using something like the QRL loss that we had
for the quasimetric, for adjacent concept states = 1 ply." (Kaveh 2026-08-18)

Stage 2 of concepts-first. The encoder is FROZEN: stage 1 decided what a concept is, and this
stage only learns the geometry over those concepts. Nothing here can reach back and reshape the
vocabulary to make the geometry easier, which is the point of doing it in two stages.

THE OBJECTIVE is Wang et al.'s QRL (arXiv 2304.01203), whose two halves are already tested in
`losses.py` and are imported rather than re-derived:

    constraint   d(z_t -> z_{t+1}) <= 1     one ply costs one unit
    objective    push d(z_i -> z_j) UP for INDEPENDENTLY sampled i, j

Maximising subject to a local constraint is what makes the learned distance the *optimal*
number of plies rather than the on-policy one -- pairs for the objective are drawn from separate
marginals precisely because same-trajectory sampling is what makes contrastive methods recover
on-policy values instead. The constraint is enforced with a dual multiplier updated by ascent on
the violation, so lambda finds its own level rather than being a hyperparameter.

SPEED. The frozen encoder is not in the loop at all: `gen_concept_codes.py` cached the codes, and
the concept state is rebuilt by a codebook gather (microseconds) instead of a transformer forward
(2.35 ms/position). That is what makes it affordable to sample fresh independent pairs every step.

READOUTS, all of which must hold for the geometry to mean anything:
    adjacent d      should sit at ~1 (times the 1+eps relaxation QRL's Theorem 3 predicts)
    k-ply odometer  d(z_t -> z_{t+k}) should grow with k, and its RANK correlation with k is the
                    honest summary -- this is the ruler working
    asymmetry       d(t -> t+k) should exceed d(t+k -> t): chess is not reversible, and a
                    symmetric metric would be evidence the head collapsed to one
    self-distance   d(z, z) == 0 by construction; verified, not assumed

    .venv/bin/python -m ...concept_quasimetric --codes <npz> --steps 20000 --device mps
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import torch.nn as nn

from catspace.research.tools.training_infra.losses import (
    qrl_constraint_violation, qrl_maximise)


class ConceptQuasimetric(nn.Module):
    """Frozen concept state -> IQE latent -> directed distance."""

    def __init__(self, d_state, d_iqe=512, components=16, hidden=1024, leak_beta=0.0):
        super().__init__()
        from catspace.research.components.encoder.approaches.jepa_tokenizer.src.iqe import IQE
        self.proj = nn.Sequential(nn.Linear(d_state, hidden), nn.GELU(),
                                  nn.Linear(hidden, d_iqe))
        self.iqe = IQE(d_iqe, components=components, leak_beta=leak_beta)

    def embed(self, z):
        return self.proj(z)

    def dist(self, za, zb):
        return self.iqe(self.embed(za), self.embed(zb))


def build_step(args, dev):
    from catspace.io import paths

    t0 = time.time()
    z = np.load(args.codes, allow_pickle=True)
    g, s = z["g"].astype(np.int64), z["s"].astype(np.int64)
    g_book = torch.from_numpy(z["g_book"]).to(dev)        # (H, C, d_code)
    s_book = torch.from_numpy(z["s_book"]).to(dev)        # (SC, d_fine)
    n, H = g.shape
    d_state = H * g_book.shape[-1] + 64 * s_book.shape[-1]
    print(f"[qm] {n:,} cached concept states | d_state {d_state} "
          f"({H} global x {g_book.shape[-1]} + 64 x {s_book.shape[-1]}) "
          f"| from {z['ckpt']} step {z['step']} [{time.time()-t0:.0f}s]", flush=True)

    G = torch.from_numpy(g).to(dev)
    S = torch.from_numpy(s).to(dev)
    ar_h = torch.arange(H, device=dev)

    def state(rows):
        """codes -> concept state, by gather. The frozen trunk never runs."""
        gi = G[rows]                                        # (B, H)
        si = S[rows]                                        # (B, 64)
        zg = g_book[ar_h[None, :], gi].reshape(len(rows), -1)
        zs = s_book[si].reshape(len(rows), -1)
        zq = torch.cat([zg, zs], -1)
        lo = zq.min(dim=-1, keepdim=True).values
        hi = zq.max(dim=-1, keepdim=True).values
        return (zq - lo) / (hi - lo).clamp(min=1e-5)        # same range-norm as the encoder

    # adjacency: r and r+1 are one ply apart iff they are in the same game
    from catspace.research.components.encoder.approaches.reach_probability.src import (
        trajectories as T)
    ck_args = None
    tr = T.build(n_human=0, n_sf=args.games, seed=args.seed, cache=True,
                 max_plies=args.max_plies, n_piecedown=args.n_piecedown)
    assert len(tr.tok) == n, f"codes are for {n} rows, corpus has {len(tr.tok)}"
    gor = tr.game_of_row()
    ok = np.zeros(n, bool)
    ok[:-1] = gor[1:] == gor[:-1]
    adj = np.flatnonzero(ok)
    # split by GAME so the readouts are on games the geometry never trained on
    rs = np.random.default_rng(args.seed); perm = rs.permutation(len(tr.start))
    nt = max(1, int(0.1 * len(tr.start)))
    test_g = np.zeros(len(tr.start), bool); test_g[perm[:nt]] = True
    is_test = test_g[gor]
    adj_tr = adj[~is_test[adj]]
    all_tr = np.flatnonzero(~is_test)
    print(f"[qm] adjacent pairs: {len(adj_tr):,} train | free rows {len(all_tr):,} | "
          f"held-out games {nt:,}", flush=True)

    net = ConceptQuasimetric(d_state, d_iqe=args.d_iqe, components=args.components,
                             hidden=args.hidden, leak_beta=args.leak_beta).to(dev)
    print(f"[qm] {sum(p.numel() for p in net.parameters()):,} params | "
          f"IQE {args.components} components x {args.d_iqe//args.components}", flush=True)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    lam = torch.tensor(float(args.lam0), device=dev)
    rng = np.random.default_rng(args.seed)

    # a fixed held-out probe set for the odometer readout, built once
    te_rows = np.flatnonzero(is_test)
    probe = te_rows[rng.choice(len(te_rows), min(4000, len(te_rows)), replace=False)]
    probe = probe[np.array([gor[min(r + args.k_max, n - 1)] == gor[r] for r in probe])]
    P0 = torch.from_numpy(probe).to(dev)

    def step_fn(model, step):
        nonlocal lam
        model.train()
        a = torch.from_numpy(adj_tr[rng.integers(0, len(adj_tr), args.batch)]).to(dev)
        za, zb = state(a), state(a + 1)
        d_adj = model.dist(za, zb)
        viol = qrl_constraint_violation(d_adj, ceiling=args.step_cost, eps=args.eps)

        # INDEPENDENT marginals -- deliberately not same-trajectory
        i = torch.from_numpy(all_tr[rng.integers(0, len(all_tr), args.batch)]).to(dev)
        j = torch.from_numpy(all_tr[rng.integers(0, len(all_tr), args.batch)]).to(dev)
        d_far = model.dist(state(i), state(j))
        obj = qrl_maximise(d_far, knee=args.knee, beta=args.phi_beta)

        loss = obj + lam.detach() * viol
        opt.zero_grad(); loss.backward(); opt.step()
        # dual ascent: lambda rises while the constraint is violated, relaxes when it is not
        lam = (lam + args.lr_lam * viol.detach()).clamp(min=0.0, max=args.lam_max)

        m = {}
        if step % args.eval_every == 0 or step == 1:
            model.eval()
            with torch.no_grad():
                zp = state(P0)
                ks = [1, 2, 4, 8, args.k_max]
                dk = [float(model.dist(zp, state(P0 + k)).mean()) for k in ks]
                dback = float(model.dist(state(P0 + args.k_max), zp).mean())
                dself = float(model.dist(zp, zp).abs().mean())
                # rank correlation of d with k, the honest odometer summary
                kk = np.repeat(ks, len(P0))
                dd = np.concatenate([model.dist(zp, state(P0 + k)).cpu().numpy() for k in ks])
                from scipy.stats import spearmanr
                rho = float(spearmanr(kk, dd).statistic)
            el = time.time() - t0
            print(f"[qm] step {step:>6}/{args.steps} {el/60:>6.1f}m "
                  f"({el/max(1,step):.3f}s/step) | d_adj {float(d_adj.mean()):.3f} "
                  f"viol {float(viol):+.4f} lam {float(lam):.2f} | d_far {float(d_far.mean()):.2f} "
                  f"| d@k " + "/".join(f"{x:.1f}" for x in dk) +
                  f" rho {rho:+.3f} | rev {dback:.1f} self {dself:.2e}", flush=True)
            m = {"d_adj": float(d_adj.mean()), "viol": float(viol), "lam": float(lam),
                 "d_far": float(d_far.mean()), "rho_k": rho, "d_rev": dback, "d_self": dself,
                 **{f"d_k{k}": v for k, v in zip(ks, dk)}}
            model.train()
        return {"loss": float(loss.detach()), "obj": float(obj.detach()),
                "violation": float(viol.detach()), **m}

    step_fn.opt = opt   # standard_train checkpoints optimizer state from here (resumability)
    return net, step_fn


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--codes", required=True, help="npz from gen_concept_codes.py")
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
    ap.add_argument("--step-cost", type=float, default=1.0, help="one ply costs one unit")
    ap.add_argument("--eps", type=float, default=0.25,
                    help="QRL's relaxation; distances read ~(1+eps)x true (their Theorem 3)")
    ap.add_argument("--knee", type=float, default=60.0,
                    help="phi's knee -- OUR horizon. Chess games run 40-150 plies.")
    ap.add_argument("--phi-beta", type=float, default=0.03)
    ap.add_argument("--lam0", type=float, default=1.0)
    ap.add_argument("--lr-lam", type=float, default=0.01)
    ap.add_argument("--lam-max", type=float, default=1e4)
    ap.add_argument("--k-max", type=int, default=16)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--ckpt-every", type=int, default=1000)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="artifacts/experiments/cqm1")
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
