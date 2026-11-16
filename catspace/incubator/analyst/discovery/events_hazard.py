"""PLAN 78.64 B, C, F: A HAZARD HEAD PER (EVENT, SIDE), WITH NO FIXED HORIZON. A measurement in a sandbox; nothing is wired into any player, nothing is written to rules/learned or data/derived.
Game-free: the events, sides and kinds are the words events_rules.py read from the rules, the tracker and the engine; none is named here. No engine evaluation, no labeller: what happened on the board.

THE TARGET. From a settled (quiet) row of 78.59 at position p, for one (event class, side): the TIME to the next event of that class by that side, in plies, or "none before the record ended" (the record
ended first: CUT, not a zero). A taking with no settled position after it is cut too (the derived reading "keeps"). THERE IS NO HORIZON: every step from the row to the event (or to the cut) is a record of the
head's data. A step belongs to the head's RISK SET unless the side cannot act on it: a flag (the step is the side's own move, or not) is in the risk set when at least one percent of the class's events in the FIT
games fall on it (the tracker records a taking on the taker's own move, so the other flag has none).

THE HEAD. A discrete hazard: the chance of the event at a step of distance d from the row is a logistic function  b[bin(d), flag] + w . x  of the row's inputs x (standardised on the fit rows), where the
baseline b[bin, flag] is one number per bin of distance (pairs of plies 1, 2, 3, 4, 5-6, 7-8, 9-12, 13-20, 21-40, 41 and beyond: a parametrisation of the curve of the time since the row, not a window: the
last bin has no end) and flag. The fit maximises the likelihood of the whole history of every row (the sum over its steps of log(1 - h) up to the event and log h at it); the L2 penalty on w is chosen on a
fifth of the fit games held out inside the fit games (grid below). BASE HAZARD: the same with w = 0 (closed form, per bin and flag), fitted on the same fit games.

WHAT IS READ on the TEST games (the 78.59 split, never fitted on): the log loss per step (the head's negative log likelihood over the steps in the head's risk set, divided by their number) against the
base hazard's, the gain with a 90 pct band over games (draws), the gain by distance bin (the horizon the data gives), a learning curve at every set of inputs (an eighth, a quarter, a half, all of the fit
games), and the WHO FIRST reading: from the two sides' hazards of one class (the sides alternate: a step is one side's or the other's), the chance that the side to move at the row has the first event, against
what happened, on the test rows where the class occurred at all before the record ended; accuracy, log loss against the base hazard's own chance, calibration.

INPUT SETS: R (the two ratings' mean and difference, ply), K (the kinds ours less theirs), K+S (the 26 statements, ours less theirs), K+S+E (edge distances of the units by kind, ours less theirs), K+S+E+R,
then on the rows of the 78.62 state read the present tactical state (per side: units read, forced safe, safe by the estimate, threatened, trapped, immobile, resting on a bound, lost by the estimate), and on
the thinned rows (every 8th quiet row of a game) the 64 concepts of 78.54.

  .venv/bin/python -m catspace.incubator.analyst.discovery.events_hazard --sample-dir D --out-dir O
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np
import torch

from catspace.incubator.analyst.discovery import events_rules as ER
from catspace.incubator.analyst.discovery import human_heads_data as HD
from catspace.io import paths

BIN_UPPER = np.array([1, 2, 3, 4, 6, 8, 12, 20, 40, 10 ** 9])            # the pair-of-plies count j = ceil(d / 2), the upper end of each bin; the last has no end
NB = len(BIN_UPPER)
LAMBDAS = (1e-4, 1e-3, 1e-2, 1e-1)
FRACTIONS = (0.125, 0.25, 0.5, 1.0)
DRAWS = 200
SEED = 6464
FLAG_SHARE = 0.01
MAXD = 260
INF = 10 ** 9
torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))


def first_at_or_after(arr: np.ndarray, p: np.ndarray) -> np.ndarray:
    """For each p, the first element of the sorted array at or after it, INF where none."""
    if len(arr) == 0:
        return np.full(len(p), INF, np.int64)
    i = np.searchsorted(arr, p)
    return np.where(i < len(arr), arr[np.minimum(i, len(arr) - 1)], INF)


def bin_of_d(d: np.ndarray) -> np.ndarray:
    return np.searchsorted(BIN_UPPER, (np.asarray(d) + 1) // 2)


def step_table() -> np.ndarray:
    """TAB[D, bin, parity]: how many of the steps d = 1..D fall in each bin, split by the parity of d (0 odd, 1 even)."""
    tab = np.zeros((MAXD + 1, NB, 2), np.float32)
    for d in range(1, MAXD + 1):
        tab[d] = tab[d - 1]
        tab[d, bin_of_d(d), 1 - (d % 2)] += 1
    return tab


TAB = step_table()


# ------------------------------------------------------------------------------------------------------------------------------ the rows

def load_sample(sample_dir: Path) -> dict:
    z = pickle.loads((sample_dir / "games.pkl").read_bytes())
    return z


def end_events(res: list, ids: list, gmeta: dict, roles: list) -> None:
    """Add the end of each game as an event of the class (end, side) at the last step: the side at the highest goal value (the record's result); no side: the class (end, '')."""
    for r, gid in zip(res, ids):
        y = gmeta[gid]["y"]
        win = r["first"] if y == 0 else ("" if y == 1 else [x for x in roles if x != r["first"]][0])
        r["events"].setdefault(("end", win), []).append(r["n"] - 1)


def build_rows(sample: dict, heads_dir: "Path | None", state_dir: "Path | None", rows_src: "dict | None" = None, index: "dict | None" = None) -> dict:
    """The quiet rows of the sample's games with the inputs; `g` the index in the sample, `T` the steps of the game (the last step is T - 1). The rows are those of the 78.59 file unless `rows_src`
    (the same layout) and `index` (game id to the game number in it) are given."""
    if rows_src is None:
        allg = [json.loads(x) for x in (heads_dir / "games.jsonl").read_text().splitlines() if x.strip()]
        index = {g["id"]: i for i, g in enumerate(allg)}
        rows = HD.load_rows(heads_dir / "rows.npz")
    else:
        rows = rows_src
    mine = {index[i]: k for k, i in enumerate(sample["ids"])}
    sel = np.flatnonzero(np.isin(rows["game"], list(mine)))
    gl = np.array([mine[int(x)] for x in rows["game"][sel]], np.int32)
    order = np.lexsort((rows["ply"][sel], gl))
    sel, gl = sel[order], gl[order]
    res = sample["res"]
    ply = rows["ply"][sel].astype(int)
    mover = np.zeros(len(sel), np.int8)
    edge = np.zeros((len(sel), 2, len(res[0]["kinds"])), np.float32)
    off = {}
    for k in np.unique(gl):
        s = np.flatnonzero(gl == k)
        r = res[int(k)]
        assert len(s) == len(r["mover_q"]), (k, len(s), len(r["mover_q"]))
        mover[s] = r["mover_q"]
        edge[s] = r["edge"]
        off[int(k)] = s
    e_diff = np.zeros((len(sel), edge.shape[2]), np.float32)
    for i in range(len(sel)):
        e_diff[i] = edge[i, mover[i]] - edge[i, 1 - mover[i]]
    meanr = (rows["rm"][sel] + rows["ro"][sel]) / 2.0
    out = {"g": gl, "ply": ply, "mover": mover, "T": np.array([res[int(k)]["n"] for k in gl]), "src": sel,
           "K": rows["kd"][sel].astype(np.float32), "S": rows["sd"][sel].astype(np.float32), "E": e_diff,
           "R": np.stack([meanr, (rows["rm"][sel] - rows["ro"][sel]).astype(float), ply.astype(float)], axis=1).astype(np.float32),
           "fit": rows["fit"][sel].astype(bool), "band": rows["band"][sel].astype(int), "y": rows["y"][sel].astype(int), "cmask": rows["cmask"][sel].astype(bool)}
    cx_full = np.zeros((len(rows["game"]), rows["cx"].shape[1]), np.float32)
    cx_full[np.flatnonzero(rows["cmask"])] = rows["cx"]
    out["C"] = cx_full[sel]
    out["meta"] = {"kinds": rows["meta"]["kinds"], "statements": rows["meta"]["statements"], "concepts": rows["meta"]["concepts"], "edge_kinds": res[0]["kinds"]}
    out["state"] = None
    if state_dir is not None and (state_dir / "state.npz").exists():
        out["state"] = present_state(out, sample, state_dir)
    return out


def present_state(rows: dict, sample: dict, state_dir: Path) -> np.ndarray:
    """The present tactical state of the 78.62 read on each row that has one (NaN where the game is not among the state games or the row was not read)."""
    from catspace.incubator.analyst.discovery import tactical_next as TN
    from catspace.incubator.analyst.discovery import tactical_tables as TT
    d = TT.load_all(state_dir)
    st, tb, games = d["st"], d["tb"], d["games"]
    ids, first, last = TT.game_slices(st)
    state_slot = {games[int(g)]["id"]: k for k, g in enumerate(ids)}
    lo = int(st["ply"].min())
    out = np.full((len(rows["g"]), 2 * len(TN.FIELDS)), np.nan, np.float32)
    for j, (g, p) in enumerate(zip(rows["g"], rows["ply"])):
        k = state_slot.get(sample["ids"][int(g)])
        if k is None:
            continue
        i = int(first[k]) + int(p) - lo
        if i < first[k] or i >= last[k] or int(st["ply"][i]) != int(p):
            continue
        c = int(st["mover"][i])
        out[j] = np.array([tb[f][i, cc] for f in TN.FIELDS for cc in (c, 1 - c)], np.float32)
    return out


def sets_of(rows: dict) -> dict:
    s = {"R": rows["R"], "K": rows["K"], "K+S": np.hstack([rows["K"], rows["S"]]), "K+S+E": np.hstack([rows["K"], rows["S"], rows["E"]])}
    s["K+S+E+R"] = np.hstack([s["K+S+E"], rows["R"]])
    return s


# ------------------------------------------------------------------------------------------------------------------------------ the targets of one head

def targets(core: tuple, rel: str, roles: list, rows: dict, res: list) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    """Per row, for the class `core` by the side `rel` (us: the side to move at the row, them: the other): D the steps from the row that are before the event or the cut (all of them to the end of
    the record when neither happens: the record ended first), has (the event happened), cut (a taking with no settled position after it came first). The event's own step is d = D + 1."""
    n = len(rows["g"])
    D = np.zeros(n, np.int64)
    has = np.zeros(n, bool)
    cutp = np.zeros(n, bool)
    g, ply, mover = rows["g"], rows["ply"], rows["mover"]
    bounds = np.flatnonzero(np.diff(g)) + 1
    starts = np.concatenate([[0], bounds])
    ends = np.concatenate([bounds, [n]])
    for s, e in zip(starts, ends):
        r = res[int(g[s])]
        T = r["n"]
        p = ply[s:e]
        m = mover[s:e]
        for side in (0, 1):
            pick = (m == side) if rel == "us" else (m != side)
            if not pick.any():
                continue
            key = (core[0], roles[side]) + core[1:]
            evs = np.array(sorted(r["events"].get(key, []) + r["kept"].get(key, [])), np.int64)
            cut = np.array(sorted(r["cut"].get(key, [])), np.int64)
            pp = p[pick]
            fe = first_at_or_after(evs, pp)
            fc = first_at_or_after(cut, pp)
            is_ev = (fe < fc) & (fe <= T - 1)
            is_cut = (~is_ev) & (fc <= T - 1)
            idx = np.arange(s, e)[pick]
            D[idx] = np.where(is_ev, fe - pp, np.where(is_cut, fc - pp, T - pp))
            has[idx] = is_ev
            cutp[idx] = is_cut
    return D, has, cutp


class Head:
    """One head's data on all rows: the counts of steps by bin and flag, the event's bin and flag, and the active flags."""

    def __init__(self, core: tuple, rel: str, D: np.ndarray, has: np.ndarray, cutp: np.ndarray):
        self.core, self.rel = core, rel
        self.D, self.has, self.cut = D, has, cutp
        self.de = np.where(has, D + 1, 0)
        assert D.max() <= MAXD
        self.own_par = 0 if rel == "us" else 1                      # the parity of d at which the side is the one to move
        # events' flag: 0 own, 1 not own
        own = ((self.de % 2) == 1) == (rel == "us")                    # the event's step is the side's own move
        self.evflag = np.where(has & ~own, 1, 0)
        self.evbin = np.where(has, bin_of_d(np.maximum(self.de, 1)), 0)
        self.active = np.array([True, True])
        self.Nf = None

    def activate(self, fit: np.ndarray) -> None:
        ev = fit & self.has
        n = max(1, int(ev.sum()))
        share = np.array([(ev & (self.evflag == f)).sum() / n for f in (0, 1)])
        self.active = share >= FLAG_SHARE
        t = TAB[self.D]                                              # (n, NB, 2 parity)
        nf = np.zeros((len(self.D), NB, 2), np.float32)
        for f in (0, 1):
            par = f if self.own_par == 0 else 1 - f                  # the parity index (0 odd d, 1 even d) whose steps are flag f
            if self.active[f]:
                nf[:, :, f] = t[:, :, par]
        self.Nf = nf

    def steps(self) -> np.ndarray:
        """Steps of each row in the risk set: the non-event steps in the active flags, plus the event's step."""
        return self.Nf.sum(axis=(1, 2)) + self.has

    def base_logit(self, sel: np.ndarray) -> np.ndarray:
        """The base hazard per (bin, flag) as a logit: events over risk steps on the rows `sel`."""
        risk = self.Nf[sel].sum(axis=0).astype(np.float64)
        ev = np.zeros((NB, 2))
        w = sel[self.has[sel]]
        np.add.at(ev, (self.evbin[w], self.evflag[w]), 1.0)
        risk = risk + ev
        h = np.where(risk > 0, ev / np.maximum(risk, 1e-9), 0.0)
        h = np.clip(h, 1e-6, 1 - 1e-6)
        return np.log(h / (1 - h)).astype(np.float32)


# ------------------------------------------------------------------------------------------------------------------------------ the fit

def nll_terms(eta0: torch.Tensor, b: torch.Tensor, Nf: torch.Tensor, evbin: torch.Tensor, evflag: torch.Tensor, has: torch.Tensor) -> "tuple[torch.Tensor, torch.Tensor]":
    """Per row: the negative log likelihood, and per row and bin its parts. eta0 (n,), b (NB,2)."""
    eta = eta0[:, None, None] + b[None]
    nonev = Nf * torch.nn.functional.softplus(eta)                              # (n, NB, 2)
    ev_eta = eta0 + b[evbin, evflag]
    ev = has * torch.nn.functional.softplus(-ev_eta)
    perbin = nonev.sum(dim=2)
    perbin = perbin.clone()
    perbin[torch.arange(len(evbin)), evbin] += ev
    return perbin.sum(dim=1), perbin


def fit_hazard(h: Head, X: np.ndarray, sel: np.ndarray, lam: float, mean: np.ndarray, scale: np.ndarray, iters: int = 60) -> dict:
    Z = torch.from_numpy(((X[sel] - mean) / scale).astype(np.float32))
    Nf = torch.from_numpy(h.Nf[sel])
    evbin, evflag = torch.from_numpy(h.evbin[sel]), torch.from_numpy(h.evflag[sel])
    has = torch.from_numpy(h.has[sel].astype(np.float32))
    b0 = h.base_logit(sel)
    b = torch.tensor(b0, requires_grad=True)
    w = torch.zeros(Z.shape[1], requires_grad=True)
    n = float(len(sel))
    opt = torch.optim.LBFGS([w, b], lr=1.0, max_iter=iters, history_size=10, line_search_fn="strong_wolfe", tolerance_grad=1e-7, tolerance_change=1e-10)

    def closure():
        opt.zero_grad()
        tot, _ = nll_terms(Z @ w, b, Nf, evbin, evflag, has)
        loss = tot.sum() / n + lam * (w * w).sum()
        loss.backward()
        return loss
    opt.step(closure)
    return {"w": w.detach().numpy().copy(), "b": b.detach().numpy().copy(), "mean": mean, "scale": scale, "lam": lam}


def eval_head(h: Head, X: "np.ndarray | None", m: "dict | None", sel: np.ndarray, base_b: "np.ndarray | None" = None) -> "tuple[np.ndarray, np.ndarray]":
    """(per row nll, per row and bin nll) of the model `m` (None: the base hazard `base_b`) on the rows sel."""
    Nf = torch.from_numpy(h.Nf[sel])
    evbin, evflag = torch.from_numpy(h.evbin[sel]), torch.from_numpy(h.evflag[sel])
    has = torch.from_numpy(h.has[sel].astype(np.float32))
    if m is None:
        eta0, b = torch.zeros(len(sel)), torch.from_numpy(base_b)
    else:
        eta0 = torch.from_numpy((((X[sel] - m["mean"]) / m["scale"]) @ m["w"]).astype(np.float32))
        b = torch.from_numpy(m["b"])
    with torch.no_grad():
        tot, per = nll_terms(eta0, b, Nf, evbin, evflag, has)
    return tot.numpy().astype(np.float64), per.numpy().astype(np.float64)


def standardise(X: np.ndarray, sel: np.ndarray) -> "tuple[np.ndarray, np.ndarray]":
    mean = X[sel].mean(axis=0)
    sd = X[sel].std(axis=0)
    return mean, np.where(sd > 0, sd, 1.0)


def fit_best(h: Head, X: np.ndarray, rows: dict, fit_sel: np.ndarray, lam: "float | None") -> dict:
    """Fit on fit_sel; the penalty by a fifth of the fit games held out inside it (grid LAMBDAS) unless `lam` is given."""
    mean, scale = standardise(X, fit_sel)
    if lam is None:
        gs = np.unique(rows["g"][fit_sel])
        rng = np.random.default_rng(SEED)
        hold = set(rng.choice(gs, max(1, len(gs) // 5), replace=False).tolist())
        is_hold = np.array([x in hold for x in rows["g"][fit_sel]])
        tr, ho = fit_sel[~is_hold], fit_sel[is_hold]
        best, lam = None, LAMBDAS[0]
        for L in LAMBDAS:
            m = fit_hazard(h, X, tr, L, mean, scale)
            tot, _ = eval_head(h, X, m, ho)
            v = tot.sum() / max(1.0, h.steps()[ho].sum())
            if best is None or v < best:
                best, lam = v, L
    m = fit_hazard(h, X, fit_sel, lam, mean, scale)
    return m


def boot(game: np.ndarray, num: np.ndarray, den: np.ndarray, draws: int, seed: int) -> "tuple[float, float, float]":
    """Ratio of sums (num over den) over the rows with a 90 pct band over resamples of the games."""
    ids, inv = np.unique(game, return_inverse=True)
    a = np.zeros(len(ids)); b = np.zeros(len(ids))
    np.add.at(a, inv, num); np.add.at(b, inv, den)
    rng = np.random.default_rng(seed)
    v = []
    for _ in range(draws):
        k = rng.integers(0, len(ids), len(ids))
        v.append(a[k].sum() / max(b[k].sum(), 1e-12))
    return float(a.sum() / max(b.sum(), 1e-12)), float(np.percentile(v, 5)), float(np.percentile(v, 95))


def boot_bins(game: np.ndarray, num: np.ndarray, den: np.ndarray, draws: int, seed: int) -> list:
    """The same per bin (num, den of shape (rows, NB))."""
    ids, inv = np.unique(game, return_inverse=True)
    a = np.zeros((len(ids), num.shape[1])); b = np.zeros((len(ids), num.shape[1]))
    np.add.at(a, inv, num); np.add.at(b, inv, den)
    rng = np.random.default_rng(seed)
    v = []
    for _ in range(draws):
        k = rng.integers(0, len(ids), len(ids))
        v.append(a[k].sum(axis=0) / np.maximum(b[k].sum(axis=0), 1e-12))
    v = np.array(v)
    tot = a.sum(axis=0) / np.maximum(b.sum(axis=0), 1e-12)
    return [(float(tot[j]), float(np.percentile(v[:, j], 5)), float(np.percentile(v[:, j], 95))) for j in range(num.shape[1])]


def bin_steps(h: Head, sel: np.ndarray) -> np.ndarray:
    st = h.Nf[sel].sum(axis=2).astype(np.float64)
    w = h.has[sel]
    st[np.flatnonzero(w), h.evbin[sel][w]] += 1
    return st


# ------------------------------------------------------------------------------------------------------------------------------ who first

def race_probabilities(hu: Head, ht: Head, eu: np.ndarray, et: np.ndarray, bu: np.ndarray, bt: np.ndarray, remaining: np.ndarray) -> "tuple[np.ndarray, np.ndarray]":
    """(P(mover first), P(other first)) summed over the steps d = 1..remaining of each row; the odd steps are the mover's hazard (flag 0 of the head `us`), the even steps the other's (flag 0 of `them`)."""
    n = len(eu)
    surv = np.ones(n)
    pu = np.zeros(n); pt = np.zeros(n)
    top = int(remaining.max())
    for d in range(1, top + 1):
        live = remaining >= d
        j = int(bin_of_d(d))
        if d % 2 == 1:
            if not hu.active[0]:
                continue
            h = 1 / (1 + np.exp(-(eu + bu[j, 0])))
            pu += np.where(live, surv * h, 0.0)
        else:
            if not ht.active[0]:
                continue
            h = 1 / (1 + np.exp(-(et + bt[j, 0])))
            pt += np.where(live, surv * h, 0.0)
        surv = np.where(live, surv * (1 - h), surv)
    return pu, pt


def first_side_truth(hu: Head, ht: Head) -> "tuple[np.ndarray, np.ndarray]":
    """(an event of the class occurred before the record ended for either side, the mover had the first one) from the two heads' targets (the time to the event of each side)."""
    tu = np.where(hu.has, hu.de, INF)
    tt = np.where(ht.has, ht.de, INF)
    occurred = (tu < INF) | (tt < INF)
    return occurred, tu < tt


# ------------------------------------------------------------------------------------------------------------------------------ the run

def curve(h: Head, name: str, sname: str, X: np.ndarray, rows: dict, fit_sel: np.ndarray, test_sel: np.ndarray, perm: np.ndarray, a, say, curves: bool):
    """The learning curve of one head on one input set: the fit on an eighth, a quarter, a half and all of the fit games (taken in a fixed random order), each read on the test rows as the gain in
    log loss per step over the base hazard fitted on the same games, with a 90 pct band over test games. The penalty is chosen on all the fit games and kept for the smaller fits. Returns the full fit."""
    den = h.steps()[test_sel].astype(np.float64)
    full, lam, said = None, None, []
    for fr in (reversed(FRACTIONS) if curves else (1.0,)):
        k = max(1, int(round(len(perm) * fr)))
        chosen = set(perm[:k].tolist())
        sub = fit_sel[np.array([x in chosen for x in rows["g"][fit_sel]])]
        m = fit_best(h, X, rows, sub, lam if fr < 1.0 else None)
        if fr == 1.0:
            lam = m["lam"]
        tot_h, per_h = eval_head(h, X, m, test_sel)
        tot_bs, per_bs = eval_head(h, None, None, test_sel, h.base_logit(sub))
        g, glo, ghi = boot(rows["g"][test_sel], tot_bs - tot_h, den, a.draws, SEED)
        said.append(f"HEAD {name} set {sname}: fit games {k} (fit rows {len(sub)}): test log loss gain over the base hazard (same fit games) per step {g:+.5f} [{glo:+.5f}, {ghi:+.5f}]; penalty {m['lam']}")
        if fr == 1.0:
            full = (m, tot_h, per_h, tot_bs, per_bs)
    for line in reversed(said):
        say(line)
    return full


OLD_HORIZON = 8


def within(h: Head, name: str, X: np.ndarray, m: dict, rows: dict, fit_sel: np.ndarray, test_sel: np.ndarray, a, say) -> None:
    """FOR COMPARISON ONLY, with the old fixed-window targets of 78.62: the chance of the class within OLD_HORIZON plies of the row read off the hazard (one minus the product of the steps' survivals),
    against whether it happened, on the rows whose record runs that far and that are not cut; its log loss against the plain rate of the fit rows and against the base hazard's own chance."""
    ok = ((rows["T"] - rows["ply"]) >= OLD_HORIZON) & ~h.cut
    hit = (h.has & (h.de <= OLD_HORIZON)).astype(float)
    sel_fit, sel_test = fit_sel[ok[fit_sel]], test_sel[ok[test_sel]]
    if len(sel_test) < 100:
        return
    def chance(model_eta0, b):
        surv = np.ones(len(model_eta0))
        for d in range(1, OLD_HORIZON + 1):
            f = (0 if (d % 2 == 1) == (h.rel == "us") else 1)
            if h.active[f]:
                surv *= 1 - 1 / (1 + np.exp(-(model_eta0 + b[int(bin_of_d(d)), f])))
        return 1 - surv
    eta = (((X[sel_test] - m["mean"]) / m["scale"]) @ m["w"])
    ph = chance(eta, m["b"])
    pb = chance(np.zeros(len(sel_test)), h.base_logit(fit_sel))
    rate = hit[sel_fit].mean()
    y = hit[sel_test]
    def ll(p):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return -(y * np.log(p) + (1 - y) * np.log(1 - p))
    one = np.ones(len(y))
    g = rows["g"][sel_test]
    r1 = boot(g, ll(np.full(len(y), rate)) - ll(ph), one, a.draws, SEED)
    r2 = boot(g, ll(pb) - ll(ph), one, a.draws, SEED)
    say(f"HEAD {name}: DERIVED CHANCE OF THE CLASS WITHIN {OLD_HORIZON} PLIES (the old window, for comparison only; {len(sel_test)} test rows, observed rate {y.mean():.4f}, fit rate {rate:.4f}): log loss gain over the plain rate {r1[0]:+.5f} [{r1[1]:+.5f}, {r1[2]:+.5f}]; over the base hazard's own chance {r2[0]:+.5f} [{r2[1]:+.5f}, {r2[2]:+.5f}]")


def cores_of(res: list, roles: list, fit_flags: np.ndarray) -> "dict":
    """{core: fit events} of the classes with at most one word after (type, side), pooled over the sides."""
    out: dict = {}
    for r, f in zip(res, fit_flags):
        if not f:
            continue
        for key, steps in list(r["events"].items()) + list(r["kept"].items()):
            if len(key) >= 2 and key[1] in roles and len(key) <= 3:
                core = (key[0],) + key[2:]
                out[core] = out.get(core, 0) + len(steps)
    return out


def main(argv: "list | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sample-dir", default=paths.experiment("events_78_64", "sample"))
    ap.add_argument("--heads-dir", default=paths.experiment(*ER.HEADS_DIR))
    ap.add_argument("--state-dir", default=paths.experiment("tactical_78_62", "state"))
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--only", default="", help="comma list of class words (a head is run if its core contains any); empty: all")
    ap.add_argument("--no-curves", action="store_true")
    ap.add_argument("--games", type=int, default=0, help="the first N games of the sample (0: all)")
    ap.add_argument("--draws", type=int, default=DRAWS)
    a = ap.parse_args(argv)
    t0 = time.perf_counter()
    out = Path(a.out_dir or paths.experiment("events_78_64", "hazard"))
    out.mkdir(parents=True, exist_ok=True)
    lines: list = []

    def say(msg: str) -> None:
        lines.append(f"OBS hazard: {msg}")
        print(lines[-1], flush=True)

    sample = load_sample(Path(a.sample_dir))
    if a.games:
        sample = {"ids": sample["ids"][:a.games], "res": sample["res"][:a.games]}
    heads_dir = Path(a.heads_dir)
    allg = {json.loads(x)["id"]: json.loads(x) for x in (heads_dir / "games.jsonl").read_text().splitlines() if x.strip()}
    spec, _v = ER.spec_and_vocab()
    clauses = ER.rule_clauses(spec)
    roles = ER.role_words(clauses)
    res, ids = sample["res"], sample["ids"]
    end_events(res, ids, allg, roles)
    rows = build_rows(sample, heads_dir, Path(a.state_dir) if a.state_dir else None)
    if rows["state"] is not None and np.isnan(rows["state"]).any():
        rows["state"] = None                                         # the state read exists for the 800 games of 78.62 only: the set is run only where every row has it
    fit_g = np.array([allg[i]["fit"] for i in ids])
    fit_rows = rows["fit"]
    say(f"rows: {len(rows['g'])} settled rows of {len(res)} games; fit rows {int(fit_rows.sum())} ({len(np.unique(rows['g'][fit_rows]))} games), test rows {int((~fit_rows).sum())} ({len(np.unique(rows['g'][~fit_rows]))} games); thinned rows with concepts {int(rows['cmask'].sum())}; rows with the present tactical state {0 if rows['state'] is None else int((~np.isnan(rows['state'][:, 0])).sum())}")
    cores = cores_of(res, roles, fit_g)
    say("heads: " + ", ".join(f"{' | '.join(c)} ({n} fit events pooled over the sides)" for c, n in sorted(cores.items(), key=lambda kv: -kv[1])))
    sets = sets_of(rows)
    heads: dict = {}
    for core, n in sorted(cores.items(), key=lambda kv: -kv[1]):
        if a.only and not any(w in core for w in a.only.split(",")):
            continue
        for rel in ("us", "them"):
            h = Head(core, rel, *targets(core, rel, roles, rows, res))
            h.activate(fit_rows)
            nev = int((h.has & fit_rows).sum())
            if nev < ER.MIN_EVENTS:
                say(f"head {' | '.join(core)} [{rel}]: {nev} events in the fit rows' first events, under {ER.MIN_EVENTS}: too rare, left out")
                continue
            heads[(core, rel)] = h
    say(f"heads fitted: {len(heads)}")
    results: dict = {}
    fit_all = np.flatnonzero(fit_rows)
    test = np.flatnonzero(~fit_rows)
    gfit = np.unique(rows["g"][fit_all])
    perm = np.random.default_rng(SEED + 1).permutation(gfit)
    thin = np.flatnonzero(rows["cmask"])
    fit_thin, test_thin = np.intersect1d(thin, fit_all), np.intersect1d(thin, test)
    has_state = rows["state"] is not None
    for (core, rel), h in heads.items():
        name = f"{' | '.join(core)} [{rel}]"
        nev_fit = int((h.has & fit_rows).sum())
        nev_test = int((h.has & ~fit_rows).sum())
        st_fit = h.steps()[fit_all].sum()
        say(f"HEAD {name}: first events in rows: fit {nev_fit}, test {nev_test}; cut rows {int(h.cut.sum())}; risk steps in the fit rows {int(st_fit)}; flags in the risk set {[('own' if f == 0 else 'other', bool(x)) for f, x in enumerate(h.active)]}; base hazard of the head over the fit rows {nev_fit / st_fit:.5f} per step")
        tot_b, _per_b = eval_head(h, None, None, test, h.base_logit(fit_all))
        v, lo, hi = boot(rows["g"][test], tot_b, h.steps()[test].astype(np.float64), a.draws, SEED)
        say(f"HEAD {name}: base hazard, test log loss per step {v:.5f} [{lo:.5f}, {hi:.5f}]")
        todo = list(sets.items()) + ([("K+S+E+R+state", np.hstack([sets["K+S+E+R"], rows["state"]]))] if has_state else [])
        for sname, X in todo:
            results[(core, rel, sname)] = curve(h, name, sname, X, rows, fit_all, test, perm, a, say, not a.no_curves)
        if len(fit_thin) > 200 and len(test_thin) > 100:
            pt = np.array([x in set(np.unique(rows["g"][fit_thin]).tolist()) for x in perm])
            pt = perm[pt]
            for sname, X in (("K+S+E+R", sets["K+S+E+R"]), ("K+S+E+R+C", np.hstack([sets["K+S+E+R"], rows["C"]]))):
                curve(h, name + " [thinned rows]", sname, X, rows, fit_thin, test_thin, pt, a, say, not a.no_curves)
        m, tot_h, per_h, tot_bs, per_bs = results[(core, rel, "K+S+E+R")]
        bs = bin_steps(h, test)
        bands = boot_bins(rows["g"][test], per_bs - per_h, bs, a.draws, SEED)
        say(f"HEAD {name} set K+S+E+R: gain by distance bin (pairs of plies 1, 2, 3, 4, 5-6, 7-8, 9-12, 13-20, 21-40, 41+), per step in the bin: " + "; ".join(f"{g:+.5f} [{lo:+.5f}, {hi:+.5f}] ({int(bs[:, j].sum())} steps)" for j, (g, lo, hi) in enumerate(bands)))
        within(h, name, sets["K+S+E+R"], m, rows, fit_all, test, a, say)
    # WHO FIRST, for every class that has both sides
    for core in sorted({c for (c, _r) in heads}):
        if (core, "us") not in heads or (core, "them") not in heads:
            continue
        hu, ht = heads[(core, "us")], heads[(core, "them")]
        occurred, truth = first_side_truth(hu, ht)
        sel = test[occurred[test]]
        if len(sel) < 100:
            continue
        remaining = (rows["T"] - rows["ply"])[sel]
        mu, mt = results[(core, "us", "K+S+E+R")][0], results[(core, "them", "K+S+E+R")][0]
        X = sets["K+S+E+R"]
        eu = (((X[sel] - mu["mean"]) / mu["scale"]) @ mu["w"])
        et = (((X[sel] - mt["mean"]) / mt["scale"]) @ mt["w"])
        pu, pt = race_probabilities(hu, ht, eu, et, mu["b"], mt["b"], remaining)
        pk = pu / np.maximum(pu + pt, 1e-12)
        pu0, pt0 = race_probabilities(hu, ht, np.zeros(len(sel)), np.zeros(len(sel)), hu.base_logit(fit_all), ht.base_logit(fit_all), remaining)
        base_p = pu0 / np.maximum(pu0 + pt0, 1e-12)
        y = truth[sel].astype(float)

        def ll(p):
            p = np.clip(p, 1e-6, 1 - 1e-6)
            return -(y * np.log(p) + (1 - y) * np.log(1 - p))
        one = np.ones(len(sel))
        acc_h, alo, ahi = boot(rows["g"][sel], ((pk > 0.5) == (y > 0.5)).astype(float), one, a.draws, SEED)
        acc_b, blo, bhi = boot(rows["g"][sel], ((base_p > 0.5) == (y > 0.5)).astype(float), one, a.draws, SEED)
        g, glo, ghi = boot(rows["g"][sel], ll(base_p) - ll(pk), one, a.draws, SEED)
        say(f"WHO FIRST {' | '.join(core)}: {len(sel)} test rows where the class occurred before the record ended (the mover first in {y.mean():.4f}); accuracy head {acc_h:.4f} [{alo:.4f}, {ahi:.4f}], base hazard {acc_b:.4f} [{blo:.4f}, {bhi:.4f}]; log loss gain of the head over the base hazard's chance {g:+.5f} [{glo:+.5f}, {ghi:+.5f}]")
        order = np.argsort(pk, kind="stable")
        say(f"WHO FIRST {' | '.join(core)}: calibration (predicted>observed, ten equal-count bins) " + " ".join(f"{pk[b].mean():.3f}>{y[b].mean():.3f}" for b in np.array_split(order, 10) if len(b)))
    say(f"wall {time.perf_counter() - t0:.1f} s")
    (out / "hazard_obs.txt").write_text("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
