#!/usr/bin/env python
"""concept_reach_field.py -- P(reach concept goal g within H plies | state s), from hindsight.

THE OBJECT (Kaveh 2026-08-19): "is there a way for us to find concept states and trajectories
between the concept states, like probability? From this state that I'm in, how do I reach this
concept state where my opponent is not castled..." -- and the M3 lock said it first (2026-07-28):
the learned field is P(reach g | s), not a distance. The quasimetric detour measured why: a
distance learned from played lines is value-blind (DTZ corr -0.15, winning-track at chance). A
probability of reaching a named predicate is the decision-relevant object, and it transfers:
this file uses trajectories + hindsight only, which is exactly what a robot's logged experience
provides. No engine anywhere.

LABELS, free from the corpus. For every (row, goal): did the goal's pattern become true within H
plies of this row, in this game, and when. Hindsight first-hit with CENSORING -- rows where the
game ends before H without a hit are explicit negatives. That censoring is load-bearing: the
8/14-era retraction (d97f22a) found strata unlearnable from POSITIVES-ONLY reachability, so both
loss terms here are the already-tested censored forms from losses.py, imported not re-derived.

GOALS come from `mine_concept_goals.py`: atomic patterns that measured PERSISTENT (mean run >=
min-run plies), REACHABLE (P(hit<=H) above floor) and RECURRENT (>= min games). Random
conjunctions were measured 46% dead, so atoms first; compositions are a later stage built from
atoms that individually pass.

TWO HEADS per goal, both readouts of the same trunk:
    hit    P(reach g within H | s)            first_hit_bce
    time   log1p(plies to first hit)          censored_plies_loss (censored rows contribute 0)

    .venv/bin/python -m ...concept_reach_field --codes <npz> --goals <npz> --steps 4000
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import torch.nn as nn

from catspace.research.tools.training_infra.losses import (
    censored_plies_loss, first_hit_bce)


class ConceptReachField(nn.Module):
    """frozen concept state -> trunk -> per-goal (hit logit, log-time)."""

    def __init__(self, d_in, n_goals, hidden=1024, depth=3):
        super().__init__()
        layers = [nn.Linear(d_in, hidden), nn.GELU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.GELU()]
        self.trunk = nn.Sequential(*layers)
        self.hit = nn.Linear(hidden, n_goals)
        self.log_time = nn.Linear(hidden, n_goals)

    def forward(self, z):
        h = self.trunk(z)
        return self.hit(h), self.log_time(h)


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
    g_end = np.zeros(n, np.int64)
    for gi in range(len(tr.start)):
        s0, ln = int(tr.start[gi]), int(tr.length[gi])
        g_end[s0:s0 + ln] = s0 + ln - 1

    gz = np.load(args.goals, allow_pickle=True)
    atoms, passing = gz["atoms"], gz["passing"]
    HZN = int(gz["horizon"])
    sel = np.flatnonzero(passing)
    # rank passing goals by persistence x reach and keep the top n_goals
    score = atoms["persist"][sel] * atoms["p_hit"][sel]
    sel = sel[np.argsort(-score)][:args.n_goals]
    idx_of = []
    for a in atoms[sel]:
        col = g[:, a["slot"]] if a["kind"] == "g" else s[:, a["slot"]]
        idx_of.append(np.flatnonzero(col == a["code"]))
    NG = len(sel)
    print(f"[crf] {n:,} states | {NG} goals (of {int(passing.sum()):,} passing) | horizon {HZN} "
          f"| d_in {d_in} [{time.time()-t0:.0f}s]", flush=True)

    rs = np.random.default_rng(args.seed); perm = rs.permutation(len(tr.start))
    nt = max(1, int(0.1 * len(tr.start)))
    test_g = np.zeros(len(tr.start), bool); test_g[perm[:nt]] = True
    is_test = test_g[gor]
    train_rows = np.flatnonzero(~is_test)
    rng = np.random.default_rng(args.seed)

    def labels(rows):
        """-> hit (B,NG) float, plies (B,NG) float (-1 censored), active_now (B,NG) bool."""
        B = len(rows)
        hit = np.zeros((B, NG), np.float32)
        pls = np.full((B, NG), -1.0, np.float32)
        for j, idx in enumerate(idx_of):
            pos = np.searchsorted(idx, rows)
            nxt = np.where(pos < len(idx), idx[np.minimum(pos, len(idx) - 1)], n + 10**9)
            ok = (nxt - rows <= HZN) & (nxt <= g_end[rows])
            hit[:, j] = ok
            pls[ok, j] = (nxt - rows)[ok]
        return hit, pls

    net = ConceptReachField(d_in, NG, hidden=args.hidden, depth=args.depth).to(dev)
    print(f"[crf] {sum(p.numel() for p in net.parameters()):,} params", flush=True)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)

    te = np.flatnonzero(is_test)
    probe = np.sort(te[rng.choice(len(te), min(6000, len(te)), replace=False)])
    p_hit, p_pls = labels(probe)
    PB = torch.from_numpy(probe).to(dev)
    PH = torch.from_numpy(p_hit)
    base = p_hit.mean(0)                                # per-goal held-out base rate

    def auc_mean(scores, y):
        """mean per-goal AUC; goals with one class present are skipped."""
        out = []
        for j in range(y.shape[1]):
            yj = y[:, j]
            if yj.min() == yj.max():
                continue
            o = np.argsort(scores[:, j]); r = np.empty(len(o)); r[o] = np.arange(len(o))
            pos = yj > 0.5
            out.append((r[pos].mean() - (pos.sum() - 1) / 2) / max((~pos).sum(), 1))
        return float(np.mean(out)) if out else float("nan")

    def step_fn(model, step):
        model.train()
        r = np.sort(train_rows[rng.integers(0, len(train_rows), args.batch)])
        hit, pls = labels(r)
        R = torch.from_numpy(r).to(dev)
        lg, lt = model(state(R))
        hit_t = torch.from_numpy(hit).to(dev)
        l_hit = first_hit_bce(lg, hit_t)
        l_time = censored_plies_loss(lt, torch.from_numpy(pls).to(dev), hit_t)
        loss = l_hit + args.w_time * l_time
        opt.zero_grad(); loss.backward(); opt.step()

        m = {}
        if step % args.eval_every == 0 or step == 1:
            model.eval()
            with torch.no_grad():
                sc = []
                for a in range(0, len(probe), 2048):
                    lg2, _ = model(state(PB[a:a + 2048]))
                    sc.append(torch.sigmoid(lg2).cpu().numpy())
                sc = np.concatenate(sc)
            auc = auc_mean(sc, p_hit)
            # calibration: mean |predicted - empirical| over 10 confidence bins, pooled
            b = np.clip((sc * 10).astype(int), 0, 9)
            ece = 0.0
            for k in range(10):
                mk = b == k
                if mk.sum():
                    ece += mk.mean() * abs(p_hit[mk].mean() - sc[mk].mean())
            el = time.time() - t0
            print(f"[crf] step {step:>6}/{args.steps} {el/60:>5.1f}m "
                  f"({el/max(1,step):.3f}s/step) | hit {float(l_hit):.4f} "
                  f"time {float(l_time):.4f} | AUC {auc:.4f} ECE {ece:.4f} "
                  f"| base med {np.median(base):.3f}", flush=True)
            m = {"auc": auc, "ece": ece}
            model.train()
        return {"loss": float(loss.detach()), "hit": float(l_hit.detach()),
                "time": float(l_time.detach()), **m}

    return net, step_fn


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--codes", required=True)
    ap.add_argument("--goals", required=True)
    ap.add_argument("--games", type=int, default=20000)
    ap.add_argument("--n-piecedown", type=int, default=45906)
    ap.add_argument("--max-plies", type=int, default=400)
    ap.add_argument("--n-goals", type=int, default=256)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--w-time", type=float, default=0.25)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--ckpt-every", type=int, default=1000)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="artifacts/experiments/crf1")
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
