"""PLAN 76.4c (c) MATERIAL AS THE FIRST MINED VALUATION (docs/ANALYST_PLAN.md 76.4; Kaveh: "if the generator cannot rediscover material, nothing above it is
derived"). GENERIC: this module names no game object. Material is a COUNT TERM of the one term language (terms.py) over the spec's unit kinds with WEIGHTS --

    (sub (wcount 1 V (cell ? us ?)) (wcount 1 V (cell ? them ?)))        V: the worth of each unit kind (the placement fluent's unit sort)

-- and the weights are fit ONLY TO OUTCOMES: a corpus of positions, each with the result the game reached from the side to move's view (a win, a draw, a loss).
THE ATOMS: per unit kind k the count term (sub (count (cell ? us k)) (count (cell ? them k))), read through the reader; a kind whose count difference is 0 at every
corpus position is CONSTANT (the royal unit: one each) -- it carries no outcome and takes no weight (it is the VALUATION's `royal`).
THE MODEL: the outcome is ordered (loss < draw < win); with m the term's value, P(win) = sigma(s m - c), P(loss) = sigma(-s m - c), P(draw) the rest (s the scale,
c >= 0 the draw band: a symmetric ordinal logistic, fit by maximum likelihood). THE SCORE: the mean LOG LOSS (a proper scoring rule, nats per position) and the
mean BRIER score, both printed; positions are grouped by their atom vector (the counts are all the model reads), so a fit costs a pass over the distinct vectors.
THE SEARCH, PRICED, INTEGERS FIRST: (1) the continuous maximum-likelihood weights (the best any weights do); (2) the integer weight vectors by PRICE (the sum of
the weights: the size of the term's weights), cheapest first, each fit for its own s and c -- at each price THE VECTORS NEAREST THE BEST WEIGHTS (each weight the
floor or the ceiling of its share of the price: the rounding of the relaxation; up to CHECK_PRICE every vector of the price is read too and the two bests compared,
printed); THE FIRST PRICE at which an integer vector predicts the outcomes AS WELL AS THE BEST WEIGHTS -- its mean log loss above the continuous fit's by no more
than ONE STANDARD ERROR OF THAT DIFFERENCE, paired position by position with the GAMES as clusters (the positions of one game are not independent): the corpus
cannot tell the two apart -- is the answer, the lowest loss at that price (the one-standard-error rule of Breiman, Friedman, Olshen and Stone 1984, read on the
paired difference; the looser reading, the error of the best fit's own mean loss, is printed beside it). A vector with a common divisor is the same valuation as
its quotient (the scale s absorbs it) and is skipped. THE FUSE: the price of the location count (a weight above the places a unit could stand is never reached).
"""
from __future__ import annotations

import itertools
import math
import time
from collections import defaultdict

import numpy as np

from catspace.incubator.analyst.discovery import terms as TL


def count_terms(dom: TL.Domain) -> dict:
    """The atoms: unit kind -> (sub (count (cell ? us k)) (count (cell ? them k))) over the spec's placement fluent."""
    plc = dom.placement
    out = {}
    for k in dom.domain(TL.UNIT):
        us = TL.mk("count", TL.lit(dom, plc, None, TL.US, k))
        them = TL.mk("count", TL.lit(dom, plc, None, TL.THEM, k))
        out[k] = TL.mk("sub", us, them)
    return out


def material_term(dom: TL.Domain, name: str, worth: dict) -> TL.T:
    """THE VALUATION AS A TERM: (sub (wcount 1 V (cell ? us ?)) (wcount 1 V (cell ? them ?))) with V = `worth` (unit kind -> weight)."""
    plc = dom.placement
    us = TL.mk("wcount", 1, name, worth, TL.lit(dom, plc, None, TL.US, None))
    them = TL.mk("wcount", 1, name, worth, TL.lit(dom, plc, None, TL.THEM, None))
    return TL.mk("sub", us, them)


class Corpus:
    """Positions grouped by their atom vector: `vec` (G x K counts), `n` (G x 3: losses, draws, wins), `game_of` (per position: its game, its group), the kinds."""

    def __init__(self, kinds: list, rows: list):
        """`rows`: [(game id, atom vector (tuple over `kinds`), outcome 0 | 0.5 | 1)]."""
        self.kinds_all = list(kinds)
        idx: dict = {}
        cnt = []
        pos_group, pos_game, pos_out = [], [], []
        for g, v, o in rows:
            j = idx.get(v)
            if j is None:
                j = idx[v] = len(cnt)
                cnt.append([0, 0, 0])
            k = 0 if o == 0 else 1 if o == 0.5 else 2
            cnt[j][k] += 1
            pos_group.append(j)
            pos_game.append(g)
            pos_out.append(k)
        V = np.array(sorted(idx, key=idx.get), dtype=np.float64).reshape(len(idx), len(kinds))
        live = [i for i in range(len(kinds)) if V[:, i].min() != V[:, i].max() or V[:, i].min() != 0]
        self.constant = [kinds[i] for i in range(len(kinds)) if i not in live]
        self.kinds = [kinds[i] for i in live]
        self.vec = V[:, live]
        self.n = np.array(cnt, dtype=np.float64)
        self.pos_group = np.array(pos_group)
        games = sorted(set(pos_game))
        gi = {g: i for i, g in enumerate(games)}
        self.pos_game = np.array([gi[g] for g in pos_game])
        self.pos_out = np.array(pos_out)
        self.games = len(games)
        self.positions = len(rows)


def _probs(m: np.ndarray, s: float, c: float) -> tuple:
    with np.errstate(over="ignore"):
        pw = 1 / (1 + np.exp(-(s * m - c)))
        pl = 1 / (1 + np.exp(-(-s * m - c)))
    pd = np.clip(1 - pw - pl, 1e-12, 1)
    return np.clip(pl, 1e-12, 1), pd, np.clip(pw, 1e-12, 1)


def _nll(m: np.ndarray, n: np.ndarray, s: float, c: float) -> float:
    pl, pd, pw = _probs(m, s, c)
    return float(-(n[:, 0] * np.log(pl) + n[:, 1] * np.log(pd) + n[:, 2] * np.log(pw)).sum())


def _sig(x):
    with np.errstate(over="ignore"):
        return 1 / (1 + np.exp(-x))


def fit_scale(m: np.ndarray, n: np.ndarray, start: "tuple | None" = None, iters: int = 50) -> tuple:
    """The maximum-likelihood (s, c) of the ordinal model for one term's values `m` per group -> (s, c, total negative log-likelihood): Newton's method on the
    log-likelihood (concave in (s, c): a cumulative logit; its gradient and Hessian in closed form), each step halved until the likelihood rises, c kept above
    0; started at the term's spread scaled to one logit (or at `start` when that is better); stopped when a step changes the likelihood by less than 1e-9 of it."""
    nl, nd, nw = n[:, 0], n[:, 1], n[:, 2]
    w = n.sum(axis=1)
    spread = float(np.sqrt((w * m * m).sum() / max(w.sum(), 1.0)))
    s, c = (0.5 / spread if spread > 0 else 0.5), 0.3            # the start: the term's spread scaled to one logit (the sigmoid unsaturated: Newton converges)
    if start is not None:
        s2, c2 = float(start[0]), max(float(start[1]), 1e-3)
        if _nll(m, n, s2, c2) < _nll(m, n, s, c):
            s, c = s2, c2
    cur = _nll(m, n, s, c)
    for _ in range(iters):
        a, b = s * m - c, -s * m - c
        pw, pl = _sig(a), _sig(b)
        pd = np.clip(1 - pw - pl, 1e-12, 1)
        da, db = pw * (1 - pw), pl * (1 - pl)
        dda, ddb = da * (1 - 2 * pw), db * (1 - 2 * pl)
        la = nw * (1 - pw) - nd * da / pd
        lb = nl * (1 - pl) - nd * db / pd
        laa = -nw * da - nd * (dda / pd + da * da / pd ** 2)
        lbb = -nl * db - nd * (ddb / pd + db * db / pd ** 2)
        lab = -nd * da * db / pd ** 2
        g = np.array([(m * (la - lb)).sum(), -(la + lb).sum()])
        H = np.array([[(m * m * (laa - 2 * lab + lbb)).sum(), (m * (lbb - laa)).sum()], [(m * (lbb - laa)).sum(), (laa + 2 * lab + lbb).sum()]])
        try:
            delta = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            break
        t = 1.0
        while t > 1e-6:
            s2, c2 = s - t * delta[0], max(c - t * delta[1], 1e-6)
            new = _nll(m, n, s2, c2)
            if new <= cur:
                break
            t /= 2
        if t <= 1e-6:
            break
        done = cur - new <= 1e-9 * max(1.0, cur)
        s, c, cur = s2, c2, new
        if done:
            break
    return s, c, cur


def fit_continuous(C: Corpus) -> tuple:
    """THE BEST ANY WEIGHTS DO: the maximum-likelihood linear predictor beta (one per live kind, the scale in it) and c -> (beta, c, total negative log-likelihood)."""
    from scipy.optimize import minimize

    def f(p):
        beta, c = p[:-1], abs(p[-1])
        return _nll(C.vec @ beta, C.n, 1.0, c)
    p0 = np.concatenate([np.full(len(C.kinds), 0.3), [1.0]])
    r = minimize(f, p0, method="L-BFGS-B")
    r = minimize(f, r.x, method="Nelder-Mead", options=dict(xatol=1e-7, fatol=1e-7, maxiter=4000))
    return r.x[:-1], abs(float(r.x[-1])), float(r.fun)


def per_position_loss(C: Corpus, m: np.ndarray, s: float, c: float) -> np.ndarray:
    """The log loss of every corpus position (nats) under a fitted term."""
    pl, pd, pw = _probs(m, s, c)
    P = np.stack([pl, pd, pw], axis=1)
    return -np.log(P[C.pos_group, C.pos_out])


def brier(C: Corpus, m: np.ndarray, s: float, c: float) -> float:
    pl, pd, pw = _probs(m, s, c)
    P = np.stack([pl, pd, pw], axis=1)
    tot = 0.0
    for k in range(3):
        y = np.zeros(3)
        y[k] = 1
        tot += float((C.n[:, k][:, None] * (P - y) ** 2).sum())
    return tot / C.positions


def clustered_se(C: Corpus, diff: np.ndarray) -> float:
    """The standard error of the mean of a per-position quantity with the games as clusters."""
    g = np.bincount(C.pos_game, weights=diff, minlength=C.games)
    ng = np.bincount(C.pos_game, minlength=C.games)
    mean = diff.mean()
    resid = g - ng * mean
    return float(math.sqrt((resid ** 2).sum()) / C.positions)


def vectors(k: int, price: int):
    """The non-negative integer vectors of `k` weights summing to `price`, no common divisor above one."""
    for cut in itertools.combinations(range(price + k - 1), k - 1):
        w = []
        prev = -1
        for x in cut + (price + k - 1,):
            w.append(x - prev - 1)
            prev = x
        if math.gcd(*w) == 1:
            yield tuple(w)


CHECK_PRICE = 12               # the prices at which the rounding's best is checked against every vector of the price (the count grows as the price to the fourth)


def nearest(beta: np.ndarray, price: int) -> list:
    """The integer vectors of `price` nearest the best weights: each weight the floor or the ceiling of its share of the price (non-negative), the sum `price`, no
    common divisor above one."""
    pos = np.clip(beta, 0, None)
    share = pos / pos.sum() * price if pos.sum() > 0 else np.full(len(beta), price / len(beta))
    lo = np.floor(share).astype(int)
    out = set()
    for up in itertools.product((0, 1), repeat=len(beta)):
        w = tuple(int(x) for x in lo + np.array(up))
        if sum(w) == price and min(w) >= 0 and math.gcd(*w) == 1:
            out.add(w)
    return sorted(out)


def search(C: Corpus, log=print, fuse: int = 64) -> dict:
    """THE PRICED SEARCH (see the docstring) -> {beta, c, nll (continuous), choice: {worth, price, s, c, logloss, brier, se, tried}, levels}. `fuse`: the price
    the search gives up at (named: the board's locations -- a weight larger than the count of places a unit could stand is never reached)."""
    t0 = time.monotonic()
    beta, cc, nll = fit_continuous(C)
    ref = per_position_loss(C, C.vec @ beta, 1.0, cc)
    se_best = clustered_se(C, ref)                                # the best fit's own mean log loss: its standard error, games as clusters
    log(f"OBS valuation continuous fit: kinds {C.kinds} (constant, no weight: {C.constant}); beta " + ", ".join(f"{k} {b:.4f}" for k, b in zip(C.kinds, beta))
        + f" (as worths relative to the smallest positive: " + ", ".join(f"{k} {b / min(x for x in beta if x > 0):.3f}" for k, b in zip(C.kinds, beta))
        + f"); c {cc:.4f}; log loss {nll / C.positions:.5f} nats per position; Brier {brier(C, C.vec @ beta, 1.0, cc):.5f}; positions {C.positions}, games {C.games}, "
        f"distinct atom vectors {len(C.n)}; one standard error of its mean log loss {se_best:.5f}")
    best_seen = None
    start = None
    tried = 0
    checked = agree = 0

    def fit_all(ws):
        out = []
        for w in ws:
            m = C.vec @ np.array(w, dtype=np.float64)
            s, c, L = fit_scale(m, C.n, start)
            out.append((L, w, s, c))
        return sorted(out)
    for price in range(1, fuse + 1):
        cands = fit_all(nearest(beta, price))
        tried += len(cands)
        if price <= CHECK_PRICE:
            every = fit_all(vectors(len(C.kinds), price))
            checked += 1
            agree += bool(cands) and every[0][1] == cands[0][1]
            if not cands or every[0][0] < cands[0][0]:
                cands = every[:1] + cands
        if not cands:
            continue
        L, w, s, c = cands[0]
        m = C.vec @ np.array(w, dtype=np.float64)
        diff = per_position_loss(C, m, s, c) - ref
        se = clustered_se(C, diff)
        within = diff.mean() <= se
        if best_seen is None or L < best_seen[0]:
            best_seen = (L, w, s, c, price)
        log(f"OBS valuation price {price}: vectors {len(cands)}, best {dict(zip(C.kinds, w))} log loss {L / C.positions:.5f} (the continuous fit's + {diff.mean():.5f}; "
            f"the paired error {se:.5f}{', WITHIN' if within else ''}; the best fit's own error {se_best:.5f})")
        if within:
            log(f"OBS valuation the rounding checked against every vector of the price at prices 1..{min(price, CHECK_PRICE)}: the same best at {agree} of {checked}")
            return dict(beta=dict(zip(C.kinds, map(float, beta))), c=cc, nll=nll, se_best=se_best, choice=dict(worth=dict(zip(C.kinds, w)), price=price, s=s, c=c,
                        logloss=L / C.positions, brier=brier(C, m, s, c), excess=float(diff.mean()), se=se, own=se_best, tried=tried, checked=checked, agree=agree),
                        secs=round(time.monotonic() - t0, 1), constant=C.constant, ref=ref)
    return dict(beta=dict(zip(C.kinds, map(float, beta))), c=cc, nll=nll, se_best=se_best, choice=None, best=best_seen, secs=round(time.monotonic() - t0, 1),
                constant=C.constant, ref=ref)


def score_of(C: Corpus, worth: dict, ref: "np.ndarray | None" = None) -> dict:
    """A given valuation (e.g. the hand's) fit on the same corpus: its own s and c, its log loss and Brier score; with `ref` (the continuous fit's per-position
    losses) its excess over the continuous fit and the paired error of that excess."""
    w = np.array([worth.get(k, 0) for k in C.kinds], dtype=np.float64)
    m = C.vec @ w
    s, c, L = fit_scale(m, C.n)
    out = dict(s=s, c=c, logloss=L / C.positions, brier=brier(C, m, s, c))
    if ref is not None:
        diff = per_position_loss(C, m, s, c) - ref
        out.update(excess=float(diff.mean()), paired=clustered_se(C, diff))
    return out
