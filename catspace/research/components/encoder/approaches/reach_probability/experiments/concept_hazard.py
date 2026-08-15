#!/usr/bin/env python
"""concept_hazard.py -- WHEN does a concept fire? (Kaveh 2026-08-15: "we still have a one ply
distance between concepts... anytime a concept flips, we know it flips in three plies or two
plies or one ply... I would still wanna know I am four steps away from castling, and there's
a threat coming in three steps, so I'm not gonna be in time.")

THE GEOMETRY SITS ON TOP OF THE CONCEPTS, not underneath them. Rather than learning a
continuous quasimetric and reading concept timing out of it, predict the timing DIRECTLY as a
discrete time-to-event distribution per concept. That reframing removes every failure this
project hit in the metric:

  * no maximisation objective, so nothing competes with the codebooks for the representation
    (measured: the QRL push and the VQ vocabularies could not both win at any weighting)
  * no triangle inequality to satisfy, so short horizons are not sacrificed to global
    consistency (the old dA read 8.9 plies for goals 1-3 plies away -- a floor that made
    "castling in 2" and "castling in 20" indistinguishable)
  * censoring is native: "never happens" is simply one more class, not a term that had to be
    bolted on with a hinge

OUTPUT per (position, concept): a probability over
    ply 1, 2, ..., PLY_EXACT,  then coarse buckets,  then NEVER
so one-ply resolution exists exactly where plans are made and coarsens where it stops
mattering. Everything a planner needs falls out of that distribution:

    P(activates within k)   = cumulative sum
    expected plies          = weighted mean over the finite classes
    RACE: P(mine lands before theirs) = sum_k P(mine at k) * P(theirs strictly later)

That last one is the point -- "four steps to castle against a threat in three" is a
comparison the model can now answer directly, because both are distributions over the SAME
ply axis rather than distances in a learned space with no calibrated units.

    .venv/bin/python -m ...concept_hazard          # self-tests
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# one class per ply out to PLY_EXACT, then widening buckets, then "never".
PLY_EXACT = 12
COARSE = ((13, 20), (21, 40), (41, 10_000))
N_CLASS = PLY_EXACT + len(COARSE) + 1          # +1 = NEVER
NEVER = N_CLASS - 1


def plies_to_class(plies, hit):
    """(plies, hit) -> class index. hit=0 (censored / never activated) -> NEVER."""
    p = np.asarray(plies)
    h = np.asarray(hit) > 0.5
    out = np.full(len(p), NEVER, np.int64)
    ex = h & (p >= 1) & (p <= PLY_EXACT)
    out[ex] = (p[ex] - 1).astype(np.int64)
    for i, (lo, hi) in enumerate(COARSE):
        m = h & (p >= lo) & (p <= hi)
        out[m] = PLY_EXACT + i
    return out


def class_midpoints():
    """representative ply value per class, for expectations. NEVER -> inf."""
    mid = [float(k + 1) for k in range(PLY_EXACT)]
    mid += [0.5 * (lo + min(hi, 80)) for lo, hi in COARSE]
    mid += [float("inf")]
    return np.array(mid, np.float64)


class ConceptHazard(nn.Module):
    """phi -> per-(head, code) distribution over time-to-first-activation.

    Deliberately reads the POSITION EMBEDDING, not a distance to a concept anchor. Anchors
    forced every concept through one point in a shared metric space, which is what coupled
    the vocabularies to the geometry; here each concept owns its own output head and they
    only have to agree on the ply axis."""

    def __init__(self, d_in=192, heads=8, codes=64, hidden=512):
        super().__init__()
        self.heads, self.codes = heads, codes
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(),
                                 nn.Linear(hidden, heads * codes * N_CLASS))

    def forward(self, phi):
        """-> (B, heads, codes, N_CLASS) logits."""
        return self.net(phi).view(len(phi), self.heads, self.codes, N_CLASS)

    def loss(self, logits, rows, head, code, cls):
        """cross-entropy on the sampled (row, head, code) triples only -- the index gives us
        a balanced sample of hits and censored, so we grade exactly those."""
        sel = logits[rows, head, code]                      # (N, N_CLASS)
        return F.cross_entropy(sel, cls)

    # ---- readouts the planner actually consumes -------------------------------------
    @staticmethod
    def p_within(prob, k):
        """P(activates within k plies). prob: (..., N_CLASS)."""
        cls = plies_to_class(np.full(1, k), np.ones(1))[0]
        upto = cls + 1 if k <= PLY_EXACT else cls + 1
        return prob[..., :upto].sum(-1)

    @staticmethod
    def expected_plies(prob, cap=80.0):
        """expected plies to activation, censored mass folded in at `cap`."""
        mid = torch.as_tensor(class_midpoints(), dtype=prob.dtype, device=prob.device)
        mid = torch.where(torch.isinf(mid), torch.full_like(mid, cap), mid)
        return (prob * mid).sum(-1)

    @staticmethod
    def wins_race(p_mine, p_theirs):
        """P(my concept fires STRICTLY before theirs) -- the castling-vs-threat question.
        Both are distributions over the same ply axis, which is the whole reason this is
        answerable at all. NEVER is excluded from 'mine' (it cannot win) but counts as
        'theirs never arrives', so a plan that lands at all beats a threat that never comes."""
        fin = N_CLASS - 1
        pm, pt = p_mine[..., :fin], p_theirs[..., :fin]
        later = torch.flip(torch.cumsum(torch.flip(pt, [-1]), -1), [-1])   # P(theirs >= k)
        strictly_later = torch.cat([later[..., 1:], torch.zeros_like(later[..., :1])], -1)
        never_theirs = p_theirs[..., fin:].sum(-1, keepdim=True)
        return (pm * (strictly_later + never_theirs)).sum(-1)


def _tests():
    ok = True
    # class mapping: exact plies, buckets, censored
    cls = plies_to_class(np.array([1, 2, 12, 13, 25, 90, -1]), np.array([1, 1, 1, 1, 1, 1, 0]))
    ok &= list(cls[:3]) == [0, 1, 11]                      # ply 1,2,12 -> own classes
    ok &= cls[3] == PLY_EXACT and cls[4] == PLY_EXACT + 1   # 13 and 25 -> first two buckets
    ok &= cls[5] == PLY_EXACT + 2 and cls[6] == NEVER
    print(f"[hazard] class map: one-ply resolution to {PLY_EXACT}, buckets, NEVER  "
          f"{'OK' if ok else 'FAIL'}")

    torch.manual_seed(0)
    m = ConceptHazard(d_in=32, heads=2, codes=4, hidden=64)
    phi = torch.randn(8, 32)
    lg = m(phi)
    ok &= lg.shape == (8, 2, 4, N_CLASS)

    # it must LEARN timing: concept (0,0) fires at ply 3 whenever feature 0 is positive,
    # and never otherwise. A model that cannot separate 3 from never is useless for racing.
    X = torch.randn(512, 32)
    fires = X[:, 0] > 0
    tgt = torch.where(fires, torch.tensor(2), torch.tensor(NEVER))     # class 2 == ply 3
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    r = torch.zeros(512, dtype=torch.long)
    for _ in range(300):
        lg = m(X)
        loss = m.loss(lg, torch.arange(512), r, r, tgt)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        pr = torch.softmax(m(X)[:, 0, 0], -1)
    got_3 = float(pr[fires].argmax(-1).eq(2).float().mean())
    got_never = float(pr[~fires].argmax(-1).eq(NEVER).float().mean())
    ok &= got_3 > 0.9 and got_never > 0.9
    print(f"[hazard] learned 'fires at ply 3' {got_3:.0%} / 'never fires' {got_never:.0%}  "
          f"{'OK' if ok else 'FAIL'}")

    # THE RACE READOUT: mine at ply 4, theirs at ply 3 -> I lose. Reverse -> I win.
    def at(k):
        p = torch.zeros(N_CLASS); p[plies_to_class(np.array([k]), np.ones(1))[0]] = 1.0
        return p
    lose = float(ConceptHazard.wins_race(at(4), at(3)))
    win = float(ConceptHazard.wins_race(at(3), at(4)))
    tie = float(ConceptHazard.wins_race(at(3), at(3)))
    never_them = torch.zeros(N_CLASS); never_them[NEVER] = 1.0
    beats_never = float(ConceptHazard.wins_race(at(9), never_them))
    ok &= lose < 0.01 and win > 0.99 and tie < 0.01 and beats_never > 0.99
    print(f"[hazard] race: mine@4 vs theirs@3 -> {lose:.2f} (lose) | mine@3 vs theirs@4 -> "
          f"{win:.2f} (win) | vs never -> {beats_never:.2f}  {'OK' if ok else 'FAIL'}")

    # expected plies is calibrated in PLIES, which is what makes concepts comparable
    e3, e12 = float(ConceptHazard.expected_plies(at(3))), float(
        ConceptHazard.expected_plies(at(12)))
    ok &= abs(e3 - 3.0) < 1e-4 and abs(e12 - 12.0) < 1e-4
    print(f"[hazard] expected plies exact at short range: {e3:.2f}, {e12:.2f}  "
          f"{'OK' if ok else 'FAIL'}")
    print("ALL CONCEPT-HAZARD TESTS PASSED" if ok else "TESTS FAILED")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    _tests()
