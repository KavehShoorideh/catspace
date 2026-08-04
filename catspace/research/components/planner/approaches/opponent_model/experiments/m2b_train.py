#!/usr/bin/env python
"""catspace/research/components/planner/approaches/opponent_model/experiments/m2b_train.py -- M2b step 3: jointly fit the style-axis head U, the per-train-player
Delta table, and the prior mu(Elo). TRAIN individual players contribute NLL + shape their own Delta;
PROVISIONAL players contribute NLL with Delta tied to 0 -> their moves estimate mu(Elo) (the prior).
Held-out players are excluded (recovered post-hoc at eval). Loss unit-tested before the run (memory
rule: test losses / inspect targets first). MLflow-tracked via catspace/train/scaffold.
"""
from __future__ import annotations

import argparse, sys, time
from pathlib import Path

import numpy as np
import torch

from catspace.research.components.planner.approaches.opponent_model.src.style_model import StyleResidual, VOCAB, elo_norm
from catspace.research.components.planner.approaches.opponent_model.src.style_dataio import load_cache as load_cache_arrays
from catspace.research.tools.training_infra.train.scaffold import standard_train, TrainConfig, resolve_device
from catspace.io import paths


def load_cache(path, dev):
    z = load_cache_arrays(path)                            # dict; shard-dir or legacy .npz
    K = z["cand_idx"].shape[1]
    t = {
        "phi": torch.from_numpy(z["phi"].astype(np.float32)),
        "cand_idx": torch.from_numpy(z["cand_idx"].astype(np.int64)),
        "cand_logp": torch.from_numpy(z["cand_logp"].astype(np.float32)),
        "played_slot": torch.from_numpy(z["played_slot"].astype(np.int64)),
        "pidx": torch.from_numpy(z["pidx"].astype(np.int64)),
        "elo_self": torch.from_numpy(z["elo_self"].astype(np.float32)),
    }
    t["cand_mask"] = t["cand_idx"] != VOCAB
    t["rank"] = (torch.arange(K).float() / (K - 1)).unsqueeze(0).expand(t["phi"].shape[0], -1).contiguous()
    valid = z["valid"] if "valid" in z else np.ones(len(z["split"]), bool)
    meta = {"split": z["split"], "player_id": z["player_id"], "prov": z["prov"], "valid": valid}
    return t, meta


def feats_at(t, idx, dev):
    return {k: t[k][idx].to(dev) for k in ("phi", "cand_idx", "cand_logp", "cand_mask", "rank",
                                           "played_slot", "pidx", "elo_self")}


def selftest(dev):
    """unit-test the loss BEFORE training (memory rule)."""
    torch.manual_seed(0); ok = True
    m = StyleResidual(n_individual=5, lam_prior=1.0).to(dev)
    B, K = 12, 17
    phi = torch.randn(B, 64, device=dev)
    cand_idx = torch.randint(0, VOCAB, (B, K), device=dev)
    cand_logp = torch.randn(B, K, device=dev)
    cand_mask = torch.ones(B, K, dtype=torch.bool, device=dev)
    rank = (torch.arange(K).float() / (K - 1)).unsqueeze(0).expand(B, -1).to(dev)
    played = torch.randint(0, K, (B,), device=dev)
    pidx = torch.tensor([0, 1, 2, -1, -1, 0, 1, -1, 3, 4, -1, 2], device=dev)
    elo = torch.full((B,), 1500.0, device=dev)
    # (1) identity: style forced to 0 (delta=0, mu=0) -> equals base_nll over the candidate set
    with torch.no_grad():
        U = m.U_of(phi, cand_idx, cand_logp, rank)
        z0 = torch.zeros(B, m.d_z, device=dev)
        nll_style0 = m.nll(m.logits(z0, U, cand_logp, cand_mask), played)
        nll_base = m.base_nll(cand_logp, cand_mask, played)
    id_ok = torch.allclose(nll_style0, nll_base, atol=1e-5)
    print(f"  {'OK ' if id_ok else 'FAIL'} identity: z=0 reproduces base NLL (max|d|={float((nll_style0-nll_base).abs().max()):.2e})")
    ok &= id_ok
    # (2) prior penalises only train (pidx>=0) deltas; provisional rows add 0
    m2 = StyleResidual(n_individual=5, lam_prior=1.0).to(dev)
    with torch.no_grad(): m2.delta.weight.normal_(0, 1.0)
    _, prior = m2(phi, cand_idx, cand_logp, cand_mask, rank, played, pidx, elo)
    p_ok = prior.item() > 0
    _, prior_allprov = m2(phi, cand_idx, cand_logp, cand_mask, rank, played,
                          torch.full_like(pidx, -1), elo)
    pp_ok = prior_allprov.item() == 0.0
    print(f"  {'OK ' if p_ok and pp_ok else 'FAIL'} prior: train-delta>0 ({prior.item():.3f}), all-prov=0 ({prior_allprov.item():.1f})")
    ok &= p_ok and pp_ok
    # (3) a few Adam steps reduce NLL on a fixed batch (loss is trainable)
    opt = torch.optim.Adam(m2.parameters(), lr=1e-2)
    nll0 = None
    for s in range(50):
        nll, prior = m2(phi, cand_idx, cand_logp, cand_mask, rank, played, pidx, elo)
        loss = nll.mean() + 1.0 * prior
        if s == 0: nll0 = nll.mean().item()
        opt.zero_grad(); loss.backward(); opt.step()
    dec = nll.mean().item() < nll0 - 0.05
    print(f"  {'OK ' if dec else 'FAIL'} trainable: NLL {nll0:.3f} -> {nll.mean().item():.3f}")
    ok &= dec
    print("SELFTEST PASSED" if ok else "SELFTEST FAILED")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", default=paths.derived("m2b/cache.npz"))
    ap.add_argument("--out", default=paths.experiment("m2b_style.pt"))
    ap.add_argument("--steps", type=int, default=6000); ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=2e-3); ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--learn-mu", action="store_true", help="learn a prior mean mu(Elo); DEFAULT off "
                    "(mu=0: raw Maia is the universal rating prior). If set, provisional players are "
                    "included to shape mu and mu is regularized by --lam-mu.")
    ap.add_argument("--lam-mu", type=float, default=1.0, help="regularize a learned mu(Elo) toward 0 "
                    "(only used with --learn-mu; breaks the mu/Delta gauge)")
    ap.add_argument("--d-z", type=int, default=16); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--selftest-only", action="store_true")
    args = ap.parse_args()
    t0 = time.time(); dev = resolve_device("auto"); torch.manual_seed(args.seed)

    if not selftest(dev):
        sys.exit("loss selftest failed; aborting before training")
    if args.selftest_only:
        return

    t, meta = load_cache(args.cache, dev)
    split = meta["split"]
    base_mask = (split == "train") | ((split == "prov") & args.learn_mu)   # prov ONLY shapes a learned mu
    train_idx = np.flatnonzero(base_mask & meta["valid"])
    n_individual = int(t["pidx"].max().item()) + 1
    rng = np.random.default_rng(args.seed)
    print(f"[m2b-train] train+prov positions {len(train_idx):,} | train-players {n_individual:,} | "
          f"device {dev} [{time.time()-t0:.0f}s]", flush=True)

    model = StyleResidual(n_individual=n_individual, d_z=args.d_z, lam_prior=args.lam,
                          learn_mu=args.learn_mu).to(dev)
    T = {k: v.to(dev) for k, v in t.items()}
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)

    def step(_net, s):
        b = train_idx[rng.integers(0, len(train_idx), args.batch)]
        nll, prior = model(T["phi"][b], T["cand_idx"][b], T["cand_logp"][b], T["cand_mask"][b],
                           T["rank"][b], T["played_slot"][b], T["pidx"][b], T["elo_self"][b])
        loss = nll.mean() + args.lam * prior
        mu_pen = torch.zeros((), device=dev)
        if model.learn_mu:
            mu_b = model.mu_of(T["elo_self"][b])
            mu_pen = (mu_b ** 2).sum(-1).mean()            # keep a learned prior mean near 0 (gauge)
            loss = loss + args.lam_mu * mu_pen
        opt.zero_grad(); loss.backward(); opt.step()
        return {"loss": float(loss), "nll": float(nll.mean()), "prior": float(prior), "mu_pen": float(mu_pen)}

    cfg = TrainConfig(out=args.out.replace(".pt", ""), steps=args.steps, ckpt_every=args.steps,
                      eval_every=max(1, args.steps // 10), experiment="catspace_m2b_z", run_name="style_z")
    standard_train(step, model, cfg, args=None)

    # sanity: mean NLL and base NLL on a held-in train sample (delta should help in-sample)
    model.eval()
    with torch.no_grad():
        b = train_idx[rng.integers(0, len(train_idx), min(20000, len(train_idx)))]
        nll, _ = model(T["phi"][b], T["cand_idx"][b], T["cand_logp"][b], T["cand_mask"][b],
                       T["rank"][b], T["played_slot"][b], T["pidx"][b], T["elo_self"][b])
        base = model.base_nll(T["cand_logp"][b], T["cand_mask"][b], T["played_slot"][b])
    print(f"[m2b-train] in-sample NLL: base {base.mean():.4f} -> style {nll.mean():.4f} "
          f"(lift {float(base.mean()-nll.mean()):+.4f} nats) [{time.time()-t0:.0f}s]")
    torch.save({"state_dict": model.state_dict(), "n_individual": n_individual, "d_z": args.d_z,
                "lam": args.lam, "learn_mu": args.learn_mu}, args.out)
    print(f"[m2b-train] saved {args.out}")
    print("DONE m2b_train", flush=True)


if __name__ == "__main__":
    main()
