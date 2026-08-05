"""probability_less_than -- the shipped predicate over position pairs (reach_probability).

    probability_less_than(a, b, eps)  ->  REACHABLE | IMPOSSIBLE | UNKNOWN

WHAT EACH VERDICT IS WORTH, precisely -- the two are NOT symmetric and must not be read as if they
were:

  REACHABLE   b was observed after a in real play. A witness exists. This one is certain.

  IMPOSSIBLE  the conformal test REJECTS reachability at level eps. Formally: under the null
              hypothesis "this pair is reachable, and is exchangeable with the calibration
              positives", a nonconformity score this extreme occurs with probability <= eps. So the
              guarantee is a bound on the FALSE-"impossible" RATE (a Type-I error rate), which is
              exactly the property asked for -- "if there is a legal way, it should say false for
              sure" holds up to a rate you choose.

  UNKNOWN     neither established.

NAMING CAVEAT, stated rather than buried. `probability_less_than(a,b,eps)` reads as the posterior
claim P(b reachable from a) < eps. What is actually delivered is the frequentist one: P(we say
IMPOSSIBLE | the pair really is reachable) <= eps. Turning that into a posterior would need a prior
over how often queried pairs are reachable at all, which nothing in the data supplies -- and
inventing one would be exactly the sort of unearned number this repo retracts. The two coincide in
ranking and differ in interpretation; `p_value` is named honestly on the dataclass for that reason.

WHY CONFORMAL AND NOT A TRAINED CLASSIFIER. This approach has no negative class (Kaveh 2026-08-05:
"I don't want negatives"), so there is nothing to fit a decision threshold against. Split conformal
needs only held-out POSITIVES: rank the query's score against calibration scores from pairs known to
be reachable, and the coverage guarantee follows from exchangeability alone -- distribution-free,
finite-sample, no negatives, no distributional assumption on the score.

THE LIMIT OF THE GUARANTEE. Validity is marginal over the calibration distribution. A search that
queries successors one ply off the data manifold is outside it, and the bound does not transfer
there -- the same off-support failure expectimax_reachability.py exists to flag. Calibrate per
material bucket and ply band (Mondrian) and report the WORST bucket, not the pooled number: a pooled
95% can hide a bucket at 60%.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

REACHABLE, IMPOSSIBLE, UNKNOWN = "REACHABLE", "IMPOSSIBLE", "UNKNOWN"


@dataclass
class ReachVerdict:
    verdict: str                 # REACHABLE | IMPOSSIBLE | UNKNOWN
    p_value: float               # conformal p-value under the null "this pair is reachable"
    eps: float                   # the level the test was run at
    score: float                 # raw nonconformity score (higher = looks more reachable)
    witness: tuple | None = None  # (game, ply_a, ply_b) when REACHABLE was established by observation


class ReachPredicate:
    """A trained ReachJEPA plus a calibration set of scores from held-out REACHABLE pairs.

    `cal_scores` must come from pairs that are (a) genuinely reachable and (b) disjoint by game from
    both training and from whatever set is later used to verify coverage. Anything else silently
    invalidates the guarantee this class exists to provide.
    """

    def __init__(self, net, cal_scores: np.ndarray, device="cpu", witness_fn=None):
        self.net = net.eval()
        self.cal = np.sort(np.asarray(cal_scores, dtype=np.float64))
        self.device = device
        self.witness_fn = witness_fn

    @torch.no_grad()
    def score(self, feats_a, feats_b) -> np.ndarray:
        """(B,C,8,8) x2 -> (B,) nonconformity score; higher = b fits a's predicted reachable region."""
        fa = torch.as_tensor(feats_a, dtype=torch.float32, device=self.device)
        fb = torch.as_tensor(feats_b, dtype=torch.float32, device=self.device)
        z_a = self.net.encode(fa)
        z_b = self.net.encode_target(fb)
        return self.net.score(z_a, z_b).float().cpu().numpy()

    def p_value(self, scores) -> np.ndarray:
        """Conformal p-value: (1 + #{calibration scores <= s}) / (n + 1).

        Valid under exchangeability with the calibration positives. The +1s are not cosmetic --
        without them the p-value is anti-conservative at small n and the coverage claim is simply
        wrong (Vovk's finite-sample correction).
        """
        s = np.atleast_1d(np.asarray(scores, dtype=np.float64))
        rank = np.searchsorted(self.cal, s, side="right")
        return (1.0 + rank) / (len(self.cal) + 1.0)

    def tau(self, eps: float) -> float:
        """The score threshold below which the test rejects reachability at level `eps`."""
        k = int(np.floor(eps * (len(self.cal) + 1))) - 1
        return float(self.cal[max(k, 0)]) if k >= 0 else -np.inf

    def __call__(self, feats_a, feats_b, eps: float = 0.01, witness=None):
        """-> list[ReachVerdict], one per row."""
        s = self.score(feats_a, feats_b)
        p = self.p_value(s)
        out = []
        for k in range(len(s)):
            w = witness[k] if witness is not None else (
                self.witness_fn(k) if self.witness_fn is not None else None)
            if w is not None:
                out.append(ReachVerdict(REACHABLE, float(p[k]), eps, float(s[k]), w))
            elif p[k] <= eps:
                out.append(ReachVerdict(IMPOSSIBLE, float(p[k]), eps, float(s[k])))
            else:
                out.append(ReachVerdict(UNKNOWN, float(p[k]), eps, float(s[k])))
        return out


def probability_less_than(predicate: ReachPredicate, feats_a, feats_b, eps: float = 0.01):
    """Module-level convenience wrapper. See ReachPredicate.__call__ and the caveat in the module
    docstring about what `eps` does and does not bound."""
    return predicate(feats_a, feats_b, eps=eps)
