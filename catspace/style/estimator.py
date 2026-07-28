"""catspace/style/estimator.py -- M2c: the ONLINE OPPONENT-STATE ESTIMATOR (Kaveh 2026-07-27).

A single stateful belief about ONE opponent -- (Elo, z) -- updated from that opponent's OWN moves,
whether they come from HISTORY (offline games) or LIVE play as it arrives. Uniform for every opponent:
a <20-game player is estimated from their few games; a well-known player keeps updating live; nobody's
estimate is frozen. The rating prior is only the true cold-start (zero history).

  z   : recovered from the observed moves by the CONVEX weighted MAP (recent moves weighted more --
        style drifts), then INFER-THEN-CONDITION -- retrieve the k-NN nearest CLEAN training-player
        styles (Elo-banded), predict with their blend (NOT the overfit additive point-estimate; see
        memory infer_then_condition_z).
  Elo : KNOWN -> a tight prior (clamped). UNKNOWN -> a broad population prior that TIGHTENS from the
        moves: Maia-2's 11 rating buckets are a rating estimator, and the bucket whose move-distribution
        best explains the observed moves is the Elo. The retrieval band widens with Elo uncertainty, so
        "unknown Elo" is just the widest band -- graceful, never a special case.

This packages the validated method (experiments/m2b_condition.py, m2c_ingame.py, m2c_elo_id.py) into
the deployable object the exploiter (M6) consumes. It operates on FEATURES (phi + Maia candidate
log-probs) so the same object works off the cache (eval) or off live Maia+field calls (play).
"""
from __future__ import annotations

import numpy as np
import torch

from catspace.style.model import VOCAB
from catspace.style.recover import recover_delta, score_nll

# Maia-2 rating buckets: representative Elo per bucket (0=<1100 ... 10=>=2000)
ELO_REPS = np.array([1050] + [1100 + (k - 1) * 100 + 50 for k in range(1, 10)] + [2050], dtype=np.float32)


class OpponentEstimator:
    def __init__(self, model, z_train, train_elo, *, k=50, elo_band=100, recency_halflife=None,
                 lam=1.0, elo_known=None, elo_prior=None, device="cpu"):
        self.model = model.eval(); self.device = device
        self.z_train = z_train.to(device)                          # (n_train, d_z) clean styles
        self.train_elo = torch.as_tensor(train_elo, dtype=torch.float32, device=device)
        self.k = k; self.elo_band = elo_band; self.halflife = recency_halflife; self.lam = lam
        self.elo_known = elo_known
        p = np.ones(11) / 11 if elo_prior is None else np.asarray(elo_prior, float) / np.sum(elo_prior)
        self._logpost = np.log(p)                                  # accumulating logprior + Σ loglik
        self._feats = []                                           # observed feature dicts, arrival order

    # ---- observe the opponent's moves (history or live) ----
    def observe(self, feats, bucket_logliks=None):
        """feats: dict of tensors (phi, cand_idx, cand_logp, cand_mask, rank, played_slot, elo) for one
        or more observed moves, in arrival order. bucket_logliks (11,): Σ log p_maia(played|bucket) over
        these moves (only needed when Elo is unknown -- drives the Elo posterior)."""
        self._feats.append({k: v.to(self.device) for k, v in feats.items()})
        if self.elo_known is None and bucket_logliks is not None:
            self._logpost = self._logpost + np.asarray(bucket_logliks, float)

    @property
    def n_observed(self):
        return sum(len(f["played_slot"]) for f in self._feats)

    # ---- Elo belief ----
    def elo_belief(self):
        """(mean_elo, posterior|None, std). Known -> the clamped value."""
        if self.elo_known is not None:
            return float(self.elo_known), None, 0.0
        post = np.exp(self._logpost - self._logpost.max()); post /= post.sum()
        mean = float((post * ELO_REPS).sum())
        std = float(np.sqrt((post * (ELO_REPS - mean) ** 2).sum()))
        return mean, post, std

    def _recency_weights(self, n):
        if self.halflife is None or n <= 1:
            return None
        age = np.arange(n)[::-1].astype(np.float32)               # most recent (last) -> age 0
        return np.power(0.5, age / float(self.halflife))

    def _cat(self):
        keys = self._feats[0].keys()
        return {k: torch.cat([f[k] for f in self._feats], 0) for k in keys}

    # ---- style belief: weighted MAP recover -> infer-then-condition ----
    def z(self):
        if not self._feats:                                       # cold start = prior mean (raw Maia)
            return torch.zeros(self.model.d_z, device=self.device)
        feats = self._cat()
        w = self._recency_weights(len(feats["played_slot"]))
        delta, _ = recover_delta(self.model, feats, lam=self.lam, steps=60, device=self.device, weights=w)
        mean_elo, _, std = self.elo_belief()
        band = self.elo_band + (0.0 if self.elo_known is not None else std)   # widen under Elo uncertainty
        d2 = ((self.z_train - delta.unsqueeze(0)) ** 2).sum(-1)
        if band > 0 and self.elo_band > 0:
            d2 = d2 + ((self.train_elo - mean_elo).abs() > band).float() * 1e9
        idx = torch.argsort(d2)[: self.k]
        return self.z_train[idx].mean(0)                          # conditioned style

    # ---- prediction ----
    @torch.no_grad()
    def predict_nll(self, feats):
        """per-position NLL of the played move under the current opponent belief (feats' cand_logp are
        the Maia base at the opponent's Elo -- MAP bucket when unknown)."""
        return score_nll(self.model, {k: v.to(self.device) for k, v in feats.items()}, self.z(),
                         device=self.device).numpy()


def _selftest():
    from catspace.style.model import StyleResidual
    torch.manual_seed(0); dev = "cpu"; ok = True

    def check(name, cond):
        nonlocal ok; ok &= bool(cond); print(f"  {'OK ' if cond else 'FAIL'} {name}")

    n_train, d_z, K = 40, 16, 17
    model = StyleResidual(n_individual=n_train, d_z=d_z).to(dev)
    with torch.no_grad():
        model.delta.weight.normal_(0, 0.5)
    z_train = model.delta.weight.detach()
    train_elo = np.linspace(1100, 1900, n_train)

    def mkfeats(n):
        ci = torch.randint(0, VOCAB, (n, K))
        return {"phi": torch.randn(n, 64), "cand_idx": ci, "cand_logp": torch.randn(n, K),
                "cand_mask": torch.ones(n, K, dtype=torch.bool),
                "rank": (torch.arange(K).float() / (K - 1)).expand(n, -1).contiguous(),
                "played_slot": torch.randint(0, K, (n,)), "elo": torch.full((n,), 1500.0)}

    est = OpponentEstimator(model, z_train, train_elo, k=8, elo_band=100, elo_known=1500, device=dev)
    check("cold start z == 0 (prior)", torch.allclose(est.z(), torch.zeros(d_z)))
    est.observe(mkfeats(20))
    z1 = est.z()
    check("after observing, z is a conditioned style (nonzero, finite)", z1.abs().sum() > 0 and torch.isfinite(z1).all())
    check("predict_nll runs on query", np.isfinite(est.predict_nll(mkfeats(10))).all())

    # Elo estimation: unknown Elo, feed bucket logliks peaked at bucket 7 (1750) -> posterior mean near it
    est2 = OpponentEstimator(model, z_train, train_elo, k=8, elo_known=None, device=dev)
    ll = np.full(11, -3.0); ll[7] = -0.1                         # bucket 7 explains the moves best
    est2.observe(mkfeats(5), bucket_logliks=ll)
    mean_elo, post, std = est2.elo_belief()
    check("unknown-Elo posterior peaks at the explaining bucket", int(np.argmax(post)) == 7 and abs(mean_elo - ELO_REPS[7]) < 120)
    check("Elo std finite and > 0 under uncertainty", 0 < std < 500)

    # recency weighting changes the estimate (recent moves weighted more)
    est3 = OpponentEstimator(model, z_train, train_elo, k=8, elo_known=1500, recency_halflife=3, device=dev)
    est3.observe(mkfeats(30)); zr = est3.z()
    check("recency-weighted z finite", torch.isfinite(zr).all())

    print("ESTIMATOR SELFTEST PASSED" if ok else "ESTIMATOR SELFTEST FAILED")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if _selftest() else 1)
