"""PLAN 78.69: THE TWO READINGS OF A POSITION AND WHAT THEIR DIFFERENCE TELLS. SYSTEM_SPEC section 6, "When it may be read" (quiet or sharp told by two readings agreeing; open tactics checked before the
estimate is trusted), and section 4, "open tactics are carried forward"; the piece is the ESTIMATOR (the readings and the rule that says when one may be trusted). A measurement in a sandbox on rows that
exist (the settled rows of 9000 human games of 78.65 and their frame-read rows); no new games, no player changed, nothing written to rules/learned or data/derived.

INTERFACE (what this module takes, what it returns, who calls it).
  fit(out_dir, say)            takes: nothing but the rows of 78.65 (frames_heads.build) and the saved name of the output directory. Returns: the path of a pickle with, for each class (the result,
                               and "wins and keeps a unit") and each side (`us` the side to move, `them`), TWO fitted linear hazard heads (events_hazard.fit_best, the 78.64 head): `kinds` (inputs: the kinds of
                               units, ours less theirs, alone: the BASELINE) and `all` (the inputs of 78.65's last step: kinds, statements, edge distances, rating, ply, the frame blocks: the baseline with its
                               CORRECTION), and on the thinned rows a third, `concepts` (all and the concepts). Fitted on the fit games of the 78.59 split only. Called by: `measure`, `matches`.
  Readings.of(d, heads)        takes the rows and the saved heads; returns per class on the held-out rows: the chance each side has the event within the record (the chance, side_chance of 78.65), the
                               reading r = chance of `us` less chance of `them` on kinds alone and on all parts, and the difference delta = r_all - r_kinds. Called by: the stage functions.
  measure(a)                   stages A to C and E, held-out rows only (the games the 78.59 split did not fit), every band of uncertainty by bootstrap over GAMES. Returns lines: `OBS readings: ...`.
  matches(a)                   stage D: the same two readings read on the positions of the 78.67 match games where a player decided, by the play-time reader of tiebreak_number (the fit's inputs rebuilt
                               from the moves). Returns lines. Called from the command line only; nothing calls it.
  Nothing here is called by a player; the rule it measures (a difference as a signal of sharpness, an open claim with a stake above the spread) enters the frame read only in a later plan, if the
  measurement holds. It reads other pieces only through their public functions (frames_heads, events_hazard, events_rules, tactical_state, tiebreak_number); it copies none of them.

THE STAKE of an open claim (stage E) is put in the unit of the readings: the change of the kinds head's reading if the claimed unit were lost (its kind taken out of the side's kinds, or added to ours if it is
theirs), in the chance of the class; never a worth table and never a unit count as a value. A unit of the goal kind is not among the head's inputs, so its claims have no stake in this unit and are counted apart.

THE EVENTS of the rules (78.64) are those of the sample's own record (`events`), read as words; the end of the game is the class this module calls the result, the taking class and the derived reading of a taking
that is kept are those events_fill names. No unit is named here. No engine evaluation, no labeller, no external model: what happened on the board and the results of the games. No assumed worth.

  .venv/bin/python -m catspace.incubator.analyst.discovery.two_readings fit     [--out-dir D]
  .venv/bin/python -m catspace.incubator.analyst.discovery.two_readings measure [--out-dir D] [--games N]
  .venv/bin/python -m catspace.incubator.analyst.discovery.two_readings matches [--out-dir D] [--games N] [--match NAME] [--workers K]
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")                       # before numpy and torch load (workers of the match reading)

import numpy as np

from catspace.incubator.analyst.discovery import events_fill as EF
from catspace.incubator.analyst.discovery import events_hazard as EH
from catspace.incubator.analyst.discovery import frames_heads as FH
from catspace.incubator.analyst.discovery import tactical_state as TS
from catspace.io import paths

OUT = ("readings_78_69",)
END = "end"                                                          # the class of the end of the game (events_hazard.end_events)
CLASSES = {"result": (END,), "keeps": (EF.KEPT,)}                    # the result where a head exists; "wins and keeps a unit"
BANDS = 5                                                            # quintile bands of the size of the difference
DRAWS = 200
SEED = 6969
OPEN_CLASSES = ("lost", "both", "contested")                          # the frame read's classes of an open claim (units standing lost or contested)
DEFINE_AGREE = 0.5                                                   # "the readings agree": the size of the difference at or below this quantile of the held-out rows


def out_dir(p: "str | None" = None) -> Path:
    d = Path(p) if p else Path(paths.experiment(*OUT))
    d.mkdir(parents=True, exist_ok=True)
    return d


def heads_file(p: "str | None" = None) -> Path:
    return out_dir(p) / "heads.pkl"


# ------------------------------------------------------------------------------------------------------------------------------ the rows

def load(say=print) -> dict:
    """The rows of 78.65 (hazard rows with the frame blocks) and, per row, the frame read's unit records aligned to it."""
    d = FH.build(Path(paths.experiment(*FH.STATE)), say)
    st, state_ids, kinds = FH.load_state(Path(paths.experiment(*FH.STATE)))
    ix = FH.align_state(d["rows"], d["sample"]["ids"], st, state_ids)
    d["st"], d["ix"], d["state_kinds"] = st, ix, kinds
    steps = FH.steps_of(d)
    d["X_all"] = steps[-1][1]
    d["X_kinds"] = d["rows"]["K"]
    d["cols_all"] = steps[-1][2]
    return d


def heads_of(d: dict) -> dict:
    """{(class, rel): the Head with its data, active flags set on the fit rows}."""
    rows, res, roles = d["rows"], d["sample"]["res"], d["roles"]
    out = {}
    for cname, core in CLASSES.items():
        for rel in ("us", "them"):
            h = EH.Head(core, rel, *EH.targets(core, rel, roles, rows, res))
            h.activate(rows["fit"])
            out[(cname, rel)] = h
    return out


# ------------------------------------------------------------------------------------------------------------------------------ the fit

def fit(out: "str | None" = None, say=print) -> Path:
    t0 = time.perf_counter()
    d = load(say)
    rows = d["rows"]
    heads = heads_of(d)
    fit_all = np.flatnonzero(rows["fit"])
    thin = np.flatnonzero(rows["cmask"])
    fit_thin = np.intersect1d(thin, fit_all)
    Xc = np.hstack([d["X_all"], d["C"][0]])
    saved = {"models": {}, "cols": {"all": d["cols_all"]}, "n_concepts": d["C"][0].shape[1], "concept_cols": d["C"][1], "active": {k: [bool(x) for x in h.active] for k, h in heads.items()}}
    for (cname, rel), h in heads.items():
        t1 = time.perf_counter()
        nfit = int((h.has & rows["fit"]).sum())
        saved["models"][(cname, rel, "kinds")] = EH.fit_best(h, d["X_kinds"], rows, fit_all, None)
        saved["models"][(cname, rel, "all")] = EH.fit_best(h, d["X_all"], rows, fit_all, None)
        saved["models"][(cname, rel, "all thinned")] = EH.fit_best(h, d["X_all"], rows, fit_thin, None)
        saved["models"][(cname, rel, "concepts thinned")] = EH.fit_best(h, Xc, rows, fit_thin, None)
        say(f"FIT {cname} [{rel}]: first events in the fit rows {nfit}; fit rows {len(fit_all)} ({len(np.unique(rows['g'][fit_all]))} games), thinned fit rows {len(fit_thin)}; active flags {[bool(x) for x in h.active]}; wall {time.perf_counter() - t1:.0f} s")
    p = heads_file(out)
    with open(p, "wb") as f:
        pickle.dump(saved, f)
    say(f"FIT saved {p}; wall {time.perf_counter() - t0:.0f} s")
    return p


# ------------------------------------------------------------------------------------------------------------------------------ the readings on held-out rows

class Readings:
    """The two readings on the held-out rows `te` (row numbers), per class."""

    def __init__(self, d: dict, saved: dict, te: np.ndarray):
        rows = d["rows"]
        self.te, self.d = te, d
        self.remaining = (rows["T"] - rows["ply"])[te]
        self.heads = heads_of(d)
        self.chance: dict = {}                                       # (class, which): (p_us, p_them)
        self.r: dict = {}                                            # (class, which): p_us - p_them
        self.delta: dict = {}                                        # class: r_all - r_kinds
        self.saved = saved
        for cname in CLASSES:
            for which, X in (("kinds", d["X_kinds"]), ("all", d["X_all"])):
                pu, pt = (self.read(cname, rel, which, X[te]) for rel in ("us", "them"))
                self.chance[(cname, which)] = (pu, pt)
                self.r[(cname, which)] = pu - pt
            self.delta[cname] = self.r[(cname, "all")] - self.r[(cname, "kinds")]

    def read(self, cname: str, rel: str, which: str, Xte: np.ndarray, remaining: "np.ndarray | None" = None) -> np.ndarray:
        """The chance that the side has the event of the class within the record, from the head `which`, on the rows whose inputs are Xte (the held-out rows in order)."""
        h = self.heads[(cname, rel)]
        m = self.saved["models"][(cname, rel, which)]
        rem = self.remaining if remaining is None else remaining
        return FH.side_chance(h, FH.table_linear(m, Xte, np.arange(len(Xte))), rem)


# ------------------------------------------------------------------------------------------------------------------------------ bootstrap over games, bands

class GameBoot:
    """Ratios of sums per group with a 90 pct band over resamples of GAMES (never rows)."""

    def __init__(self, game: np.ndarray, draws: int = DRAWS, seed: int = SEED):
        self.ids, self.inv = np.unique(game, return_inverse=True)
        G = len(self.ids)
        rng = np.random.default_rng(seed)
        self.W = np.stack([np.bincount(rng.integers(0, G, G), minlength=G) for _ in range(draws)]).astype(np.float64)

    def sums(self, num: np.ndarray, den: np.ndarray, group: np.ndarray, ng: int) -> "tuple[np.ndarray, np.ndarray]":
        a = np.zeros((len(self.ids), ng))
        b = np.zeros((len(self.ids), ng))
        np.add.at(a, (self.inv, group), num)
        np.add.at(b, (self.inv, group), den)
        return a, b

    def ratio(self, num: np.ndarray, den: np.ndarray, group: np.ndarray, ng: int) -> "tuple[np.ndarray, np.ndarray]":
        """(point per group, draws per group of shape (D, ng))."""
        a, b = self.sums(num, den, group, ng)
        return a.sum(0) / np.maximum(b.sum(0), 1e-12), (self.W @ a) / np.maximum(self.W @ b, 1e-12)


def band(p: np.ndarray, d: np.ndarray, level: float = 0.90) -> "tuple[float, float]":
    return float(np.percentile(d, 50 * (1 - level))), float(np.percentile(d, 100 - 50 * (1 - level)))


def quantile_bands(v: np.ndarray, nb: int = BANDS) -> np.ndarray:
    """Band number by rank (equal counts)."""
    order = np.argsort(v, kind="stable")
    out = np.zeros(len(v), np.int64)
    out[order] = (np.arange(len(v)) * nb) // len(v)
    return out


def strata_bands(v: np.ndarray, stratum: np.ndarray, nb: int = BANDS) -> np.ndarray:
    """Bands of v by rank inside each stratum (the control that holds the stratum's own effect fixed)."""
    out = np.zeros(len(v), np.int64)
    for s in np.unique(stratum):
        i = np.flatnonzero(stratum == s)
        out[i] = quantile_bands(v[i], nb)
    return out


# ------------------------------------------------------------------------------------------------------------------------------ the events after a row

def event_times(d: dict, te: np.ndarray, kind: "str | None") -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    """For each held-out row: (steps to the next event of the rules by either side counting the event's own step, or the steps to the end of the record when none came: the exposure; whether the event
    came; the step numbers of the next event, INF when none). `kind`: the event's type word (None: every event of the sample's record except the end of the game)."""
    rows, res = d["rows"], d["sample"]["res"]
    g, ply = rows["g"][te], rows["ply"][te]
    expo = np.zeros(len(te))
    obs = np.zeros(len(te), bool)
    nxt = np.full(len(te), EH.INF, np.int64)
    bounds = np.concatenate([[0], np.flatnonzero(np.diff(g)) + 1, [len(g)]])
    for s, e in zip(bounds[:-1], bounds[1:]):
        r = res[int(g[s])]
        steps = sorted({int(x) for key, v in r["events"].items() if key[0] != END and (kind is None or key[0] == kind) for x in v})
        pp = ply[s:e]
        fe = EH.first_at_or_after(np.array(steps, np.int64), pp)
        got = fe <= r["n"] - 1
        expo[s:e] = np.where(got, fe - pp + 1, r["n"] - pp)
        obs[s:e] = got
        nxt[s:e] = np.where(got, fe, EH.INF)
    return expo, obs, nxt


def units_left(d: dict, te: np.ndarray) -> np.ndarray:
    rows, res = d["rows"], d["sample"]["res"]
    return np.array([res[int(g)]["census"][p].sum() for g, p in zip(rows["g"][te], rows["ply"][te])])


def claim_arrays(d: dict, te: np.ndarray, classes: tuple = OPEN_CLASSES) -> dict:
    """From the frame read's unit records of the held-out rows: counts of units standing in an open class per row, by owner (1: the side to move, 0: the other) and by kind of the state's order, and per
    class the share of rows with at least one."""
    st, ix = d["st"], d["ix"][te]
    uoff = st["uoff"]
    nk = len(d["state_kinds"])
    n = len(te)
    cls = [TS.CLASS_WORDS.index(w) for w in classes]
    by_owner_kind = np.zeros((n, 2, nk), np.int16)
    by_class_owner = np.zeros((n, len(classes), 2), np.int16)
    for j, i in enumerate(ix):
        u = st["units"][uoff[i]:uoff[i + 1]]
        for ci, c in enumerate(cls):
            m = u[:, 2] == c
            if m.any():
                np.add.at(by_owner_kind[j], (u[m, 0].astype(int), u[m, 1].astype(int)), 1)
                np.add.at(by_class_owner[j, ci], u[m, 0].astype(int), 1)
    return {"owner_kind": by_owner_kind, "class_owner": by_class_owner}


# ------------------------------------------------------------------------------------------------------------------------------ the stages

def test_rows(d: dict, games: "int | None") -> np.ndarray:
    rows = d["rows"]
    te = np.flatnonzero(~rows["fit"])
    if games:
        keep = np.unique(rows["g"][te])[:games]
        te = te[np.isin(rows["g"][te], keep)]
    return te


def stage_a(d: dict, R: Readings, say) -> None:
    rows, te = d["rows"], R.te
    ng = len(np.unique(rows["g"][te]))
    say(f"STAGE A: HELD-OUT ROWS ONLY: the games the 78.59 split did not fit (split by game, as 78.65): {len(te)} settled rows of {ng} games; the heads were fitted on the fit games ({int(rows['fit'].sum())} rows, {len(np.unique(rows['g'][rows['fit']]))} games) and never on these")
    men = units_left(d, te)
    edges = [(0, 5), (6, 12), (13, 20), (21, 99)]
    say("ENDGAME COVERAGE of the held-out rows (rows by number of units left on the board by the census; human games end early): " + "; ".join(
        f"{a}-{b if b < 99 else 'more'}: {int(((men >= a) & (men <= b)).sum())} rows ({100.0 * ((men >= a) & (men <= b)).mean():.2f} pct)" for a, b in edges))
    games_end = [(int(r["census"][-1].sum() if len(r["census"]) else 0)) for r in d["sample"]["res"]]
    say(f"ENDGAME COVERAGE: of the {len(games_end)} games (all), the last position of the census has at most 12 units in {100.0 * np.mean(np.array(games_end) <= 12):.2f} pct and at most 5 in {100.0 * np.mean(np.array(games_end) <= 5):.2f} pct")
    boot = GameBoot(rows["g"][te])
    for cname in CLASSES:
        # the heads' own loss on the held-out rows, per step, for the two readings and the base hazard
        for rel in ("us", "them"):
            h = R.heads[(cname, rel)]
            den = h.steps()[te].astype(np.float64)
            fit_all = np.flatnonzero(rows["fit"])
            base, _ = EH.eval_head(h, None, None, te, h.base_logit(fit_all))
            parts = [f"base hazard {base.sum() / den.sum():.5f}"]
            for which, X in (("kinds", d["X_kinds"]), ("all", d["X_all"])):
                tot, _ = EH.eval_head(h, X, R.saved["models"][(cname, rel, which)], te)
                g, lo, hi = EH.boot(rows["g"][te], base - tot, den, DRAWS, SEED)
                parts.append(f"{which} {tot.sum() / den.sum():.5f} (gain over base {g:+.5f} [{lo:+.5f}, {hi:+.5f}])")
            say(f"HEAD {cname} [{rel}] first events in the held-out rows {int(h.has[te].sum())}: test log loss per step: " + "; ".join(parts))
        for which in ("kinds", "all"):
            pu, pt = R.chance[(cname, which)]
            say(f"READING {cname} on {which}: chance of the side to move, mean {pu.mean():.4f}, of the other side {pt.mean():.4f}; r (the first less the second): mean {R.r[(cname, which)].mean():+.4f}, sd {R.r[(cname, which)].std():.4f}")
        dl = R.delta[cname]
        a = np.abs(dl)
        q = np.percentile(a, [10, 25, 50, 75, 90, 99])
        corr = float(np.corrcoef(R.r[(cname, "kinds")], R.r[(cname, "all")])[0, 1])
        say(f"DIFFERENCE {cname}: delta = r(all parts) less r(kinds alone); mean {dl.mean():+.4f}, |delta| quantiles 10/25/50/75/90/99: " + " / ".join(f"{x:.4f}" for x in q) + f"; correlation of the two readings {corr:.4f}; sign of the two readings agrees in {100.0 * np.mean(np.sign(R.r[(cname, 'kinds')]) == np.sign(R.r[(cname, 'all')])):.2f} pct of rows; correlation of |delta| with ply {float(np.corrcoef(a, rows['ply'][te])[0, 1]):+.4f}")
    # the concepts, on the thinned held-out rows
    thin = np.flatnonzero(rows["cmask"] & ~rows["fit"])
    Xc = np.hstack([d["X_all"], d["C"][0]])
    for cname in CLASSES:
        parts = []
        for rel in ("us", "them"):
            h = R.heads[(cname, rel)]
            den = h.steps()[thin].astype(np.float64)
            fit_all = np.flatnonzero(rows["fit"])
            base, _ = EH.eval_head(h, None, None, thin, h.base_logit(fit_all))
            ta, _ = EH.eval_head(h, d["X_all"], R.saved["models"][(cname, rel, "all thinned")], thin)
            tc, _ = EH.eval_head(h, Xc, R.saved["models"][(cname, rel, "concepts thinned")], thin)
            g, lo, hi = EH.boot(rows["g"][thin], ta - tc, den, DRAWS, SEED)
            parts.append(f"[{rel}] all parts {ta.sum() / den.sum():.5f}, with the concepts {tc.sum() / den.sum():.5f}, gain of the concepts {g:+.5f} [{lo:+.5f}, {hi:+.5f}]")
        say(f"CONCEPTS {cname} (thinned held-out rows {len(thin)}, heads refitted on the thinned fit rows): " + "; ".join(parts))


def band_lines(say, title: str, nb: int, point: np.ndarray, dr: np.ndarray, extra: "list | None" = None, fmt: str = "{:.4f}", top_vs_bottom: bool = True, scale: float = 1.0) -> None:
    parts = []
    for b in range(nb):
        lo, hi = band(point[b], dr[:, b] * scale)
        parts.append(f"band {b + 1}: {fmt.format(point[b] * scale)} [{fmt.format(lo)}, {fmt.format(hi)}]" + (f" ({extra[b]})" if extra else ""))
    line = f"{title}: " + "; ".join(parts)
    if top_vs_bottom:
        rr = dr[:, nb - 1] / np.maximum(dr[:, 0], 1e-12)
        lo, hi = band(0, rr)
        line += f"; top band over bottom band {point[nb - 1] / max(point[0], 1e-12):.3f} [{lo:.3f}, {hi:.3f}]"
    say(line)


def stage_b(d: dict, R: Readings, say) -> None:
    rows, te = d["rows"], R.te
    boot = GameBoot(rows["g"][te])
    rng = np.random.default_rng(SEED + 1)
    ply = rows["ply"][te]
    ply_band = quantile_bands(ply.astype(np.float64))
    ca = claim_arrays(d, te)
    say(f"STAGE B: the events counted are those of the rules in the sample's record (any event by either side except the end of the game, and the takings alone); the rate is events per ply of exposure (rows' steps to the next event, or to the end of the record when none came; no horizon); bands of the size of the difference |delta| by equal counts of held-out rows; the uncertainty is the 90 pct band over resamples of games")
    for kind_label, kind in (("any event", None), ("takings", EF.TAKING)):
        expo, obs, _nx = event_times(d, te, kind)
        say(f"B {kind_label}: {int(obs.sum())} of {len(te)} rows see one before the record ends; mean exposure {expo.mean():.2f} plies; overall rate per 100 plies {100.0 * obs.sum() / expo.sum():.3f}")
        by_ply, dr = boot.ratio(obs.astype(float), expo, ply_band, BANDS)
        band_lines(say, f"B {kind_label}: CONTROL ply alone, rate per 100 plies by quintile of ply", BANDS, by_ply, dr, scale=100.0, fmt="{:.3f}", extra=[f"mean ply {ply[ply_band == b].mean():.1f}" for b in range(BANDS)])
        for cname in CLASSES:
            a = np.abs(R.delta[cname])
            bd = quantile_bands(a)
            sh = quantile_bands(rng.permutation(a))
            pm = strata_bands(a, ply_band)
            pt, dr = boot.ratio(obs.astype(float), expo, bd, BANDS)
            band_lines(say, f"B {kind_label} [{cname}]: rate per 100 plies by quintile of |delta|", BANDS, pt, dr, scale=100.0, fmt="{:.3f}", extra=[f"|delta| up to {a[bd == b].max():.4f}, mean ply {ply[bd == b].mean():.1f}" for b in range(BANDS)])
            pt, dr = boot.ratio(obs.astype(float), expo, sh, BANDS)
            band_lines(say, f"B {kind_label} [{cname}]: CONTROL the same bands made from a SHUFFLED |delta|", BANDS, pt, dr, scale=100.0, fmt="{:.3f}")
            pt, dr = boot.ratio(obs.astype(float), expo, pm, BANDS)
            band_lines(say, f"B {kind_label} [{cname}]: CONTROL |delta| bands made INSIDE each quintile of ply (ply held fixed)", BANDS, pt, dr, scale=100.0, fmt="{:.3f}")
    # the frame read's own classes where the difference is large
    for cname in CLASSES:
        a = np.abs(R.delta[cname])
        bd = quantile_bands(a)
        pm = strata_bands(a, ply_band)
        for ci, w in enumerate(OPEN_CLASSES):
            for owner, label in ((1, "the side to move"), (0, "the other side")):
                has = (ca["class_owner"][:, ci, owner] > 0).astype(float)
                if has.sum() < 50:
                    say(f"B classes [{cname}]: units of class '{w}' of {label}: only {int(has.sum())} rows, not tabled")
                    continue
                pt, dr = boot.ratio(has, np.ones(len(te)), bd, BANDS)
                band_lines(say, f"B classes [{cname}]: share of rows with a unit of class '{w}' of {label} ({int(has.sum())} rows) by quintile of |delta|", BANDS, pt, dr, scale=100.0, fmt="{:.2f}")
                pt, dr = boot.ratio(has, np.ones(len(te)), pm, BANDS)
                band_lines(say, f"B classes [{cname}]: the same with the bands made inside each quintile of ply", BANDS, pt, dr, scale=100.0, fmt="{:.2f}")


def next_rows(d: dict, te: np.ndarray, lag_rows: int = 1) -> np.ndarray:
    """Index (into te) of the row lag_rows later in the same game, or -1."""
    g = d["rows"]["g"][te]
    j = np.arange(len(te)) + lag_rows
    ok = j < len(te)
    jj = np.where(ok, j, 0)
    return np.where(ok & (g[jj] == g), jj, -1)


def stage_c(d: dict, R: Readings, say) -> None:
    rows, te = d["rows"], R.te
    boot = GameBoot(rows["g"][te])
    rng = np.random.default_rng(SEED + 2)
    ply = rows["ply"][te]
    ply_band = quantile_bands(ply.astype(np.float64))
    _e, _o, nxt = event_times(d, te, None)
    nr = next_rows(d, te)
    has_next = nr >= 0
    gap = np.where(has_next, ply[np.where(has_next, nr, 0)] - ply, 0)
    say(f"STAGE C: next settled row = the next row of the same game in the held-out rows; {int(has_next.sum())} of {len(te)} rows have one; the gap in plies to it: mean {gap[has_next].mean():.2f}, share at 1 ply {100.0 * np.mean(gap[has_next] == 1):.1f} pct, at 2 plies {100.0 * np.mean(gap[has_next] == 2):.1f} pct, over 4 plies {100.0 * np.mean(gap[has_next] > 4):.1f} pct")
    for cname in CLASSES:
        a = np.abs(R.delta[cname])
        sgn = np.sign(R.delta[cname]) * (1 - 2 * rows["mover"][te])      # the side of the baseline, seen from the same side whoever moves
        for qlabel, q in (("top fifth", 0.80), ("top tenth", 0.90)):
            thr = float(np.quantile(a, q))
            large = a > thr
            sel = large & has_next
            nx = nr[sel]
            stays = a[nx] > thr
            same_sign = sgn[nx] == sgn[sel]
            ev_between = nxt[sel] < ply[nx]                         # an event of the rules at a step before the next row's ply
            base = float(np.mean(a[nr[has_next]] > thr))
            g = rows["g"][te][sel]
            ones = np.ones(int(sel.sum()))
            sb = GameBoot(g)
            p1, d1 = sb.ratio(stays.astype(float), ones, np.zeros(len(ones), np.int64), 1)
            lo1, hi1 = band(p1[0], d1[:, 0])
            p2, d2 = sb.ratio((stays & same_sign).astype(float), ones, np.zeros(len(ones), np.int64), 1)
            lo2, hi2 = band(p2[0], d2[:, 0])
            # control: the partner is a random row of the same ply quintile
            rnd = np.empty(len(nx), np.int64)
            for s in np.unique(ply_band):
                pool = np.flatnonzero(ply_band == s)
                m = ply_band[np.flatnonzero(sel)] == s
                rnd[m] = rng.choice(pool, int(m.sum()))
            ctrl = a[rnd] > thr
            p3, d3 = sb.ratio(ctrl.astype(float), ones, np.zeros(len(ones), np.int64), 1)
            lo3, hi3 = band(p3[0], d3[:, 0])
            n3 = next_rows(d, te, 3)[sel]
            stays3 = (n3 >= 0)
            allthree = stays3 & (a[np.where(stays3, n3, 0)] > thr)
            ev_pct = 100.0 * ev_between.mean()
            res_ev = float(np.mean(ev_between[~stays])) if (~stays).any() else float("nan")
            stay_ev = float(np.mean(ev_between[stays])) if stays.any() else float("nan")
            say(f"C [{cname}] large = |delta| above the {qlabel} of the held-out rows (threshold {thr:.4f}): {int(sel.sum())} large rows with a next row; RESOLVED (next row's |delta| at or under the threshold) {100.0 * (1 - p1[0]):.2f} pct [{100.0 * (1 - hi1):.2f}, {100.0 * (1 - lo1):.2f}]; STAYS {100.0 * p1[0]:.2f} pct [{100.0 * lo1:.2f}, {100.0 * hi1:.2f}]; stays on the same side of the baseline {100.0 * p2[0]:.2f} pct [{100.0 * lo2:.2f}, {100.0 * hi2:.2f}]; CONTROL base rate that a row is large (all rows with a next row) {100.0 * base:.2f} pct; CONTROL a random row of the same quintile of ply is large {100.0 * p3[0]:.2f} pct [{100.0 * lo3:.2f}, {100.0 * hi3:.2f}]")
            say(f"C [{cname}] {qlabel}: an event of the rules lay between the row and its next row in {ev_pct:.2f} pct of large rows; among the RESOLVED {100.0 * res_ev:.2f} pct, among those that STAY {100.0 * stay_ev:.2f} pct; the next three settled rows all stay large (of those with three more rows, {int(stays3.sum())}): {100.0 * (allthree.sum() / max(1, stays3.sum())):.2f} pct")


DEFINITIONS = (("lost or contested (the plan's words)", OPEN_CLASSES), ("lost only", ("lost", "both")))


def stratified_ratio(gb: GameBoot, obs: np.ndarray, expo: np.ndarray, grp: np.ndarray, strat: np.ndarray, ns: int, shuffle_rng=None) -> "tuple[float, np.ndarray]":
    """Rate of group 2 over rate of group 0 with the strata mix of group 2's rows held fixed for both (the strata's own effect taken out); with `shuffle_rng` the three labels are shuffled inside each
    stratum first (the control: a ratio of one is what no signal gives)."""
    g = grp.copy()
    if shuffle_rng is not None:
        for s in range(ns):
            i = np.flatnonzero(strat == s)
            g[i] = shuffle_rng.permutation(g[i])
    A, B = gb.sums(obs.astype(float), expo, g + 3 * strat, 3 * ns)
    w = np.array([(grp == 2)[strat == s].sum() for s in range(ns)], np.float64)

    def f(W):
        r = (W @ A) / np.maximum(W @ B, 1e-12)
        num = sum(w[s] * r[..., 2 + 3 * s] for s in range(ns))
        den = sum(w[s] * r[..., 0 + 3 * s] for s in range(ns))
        return num / np.maximum(den, 1e-12)
    return float(f(np.ones((1, A.shape[0])))[0]), f(gb.W)


def stage_e(d: dict, R: Readings, say) -> None:
    rows, te = d["rows"], R.te
    boot = GameBoot(rows["g"][te])
    rng = np.random.default_rng(SEED + 3)
    ply = rows["ply"][te]
    ply_band = quantile_bands(ply.astype(np.float64))
    men_band = quantile_bands(units_left(d, te).astype(np.float64))
    names = d["state_kinds"]
    kcols = [names.index(k) if k in names else -1 for k in rows["meta"]["kinds"]]      # the head's column -> the state's kind number
    in_head = np.zeros(len(names), bool)
    for kc in kcols:
        if kc >= 0:
            in_head[kc] = True
    exp_any = event_times(d, te, None)
    exp_tak = event_times(d, te, EF.TAKING)
    Kte = d["X_kinds"][te]
    # the stake of a claim on a unit of a kind, for each class: the change of the kinds head's reading if the unit is lost (the kind leaves the side's kinds; for a unit of the other side it joins ours)
    stake = {}
    for cname in CLASSES:
        st = np.zeros((len(te), 2, len(names)))
        for owner in (1, 0):
            for col, kc in enumerate(kcols):
                if kc < 0:
                    continue
                Kv = Kte.copy()
                Kv[:, col] += -1.0 if owner == 1 else 1.0
                pu, pt = (R.read(cname, rel, "kinds", Kv) for rel in ("us", "them"))
                st[:, owner, kc] = np.abs((pu - pt) - R.r[(cname, "kinds")])
        stake[cname] = st
    for dlabel, classes in DEFINITIONS:
        ca = claim_arrays(d, te, classes)
        ok_ = ca["owner_kind"]                                       # (n, owner, state kind)
        has = ok_ > 0
        n_claims = ok_.sum(axis=(1, 2))
        n_goal = ok_[:, :, ~in_head].sum(axis=(1, 2))
        anyc = (n_claims - n_goal) > 0
        say(f"STAGE E, open claims = units of the frame read in class {', '.join(repr(w) for w in classes)} [{dlabel}] (by owner: the side to move or the other): {int((n_claims > 0).sum())} of {len(te)} held-out rows have at least one ({100.0 * (n_claims > 0).mean():.2f} pct); claims on units outside the kinds head's inputs (the goal kind): {int(n_goal.sum())} in {int((n_goal > 0).sum())} rows, left out of the stake")
        for cname in CLASSES:
            delta = np.abs(R.delta[cname])
            st = stake[cname]
            mx = np.where(has, st, 0.0).max(axis=(1, 2))
            mx_own = np.where(has[:, 1], st[:, 1], 0.0).max(axis=1)
            mx_their = np.where(has[:, 0], st[:, 0], 0.0).max(axis=1)
            fired = anyc & (mx > delta)
            fired_own = has[:, 1][:, in_head].any(axis=1) & (mx_own > delta)
            fired_their = has[:, 0][:, in_head].any(axis=1) & (mx_their > delta)
            q = np.percentile(mx[anyc], [10, 50, 90]) if anyc.any() else [0, 0, 0]
            say(f"E1 [{dlabel}] [{cname}]: rows with an open claim that has a stake {int(anyc.sum())} ({100.0 * anyc.mean():.2f} pct); stake of the largest claim, quantiles 10/50/90: {q[0]:.4f} / {q[1]:.4f} / {q[2]:.4f}; spread |delta| of those rows, median {np.median(delta[anyc]) if anyc.any() else 0:.4f}")
            zero = np.zeros(len(te), np.int64)
            a1, d1 = boot.ratio(fired.astype(float), np.ones(len(te)), zero, 1)
            lo, hi = band(a1[0], d1[:, 0])
            a2, d2 = boot.ratio(fired.astype(float), anyc.astype(float), zero, 1)
            lo2, hi2 = band(a2[0], d2[:, 0])
            say(f"E1 [{dlabel}] [{cname}]: rows with an open claim whose stake is above the spread: {int(fired.sum())} = {100.0 * a1[0]:.2f} pct of all held-out rows [{100.0 * lo:.2f}, {100.0 * hi:.2f}], {100.0 * a2[0]:.2f} pct of the rows with an open claim [{100.0 * lo2:.2f}, {100.0 * hi2:.2f}]; the claim is on the mover's unit in {int(fired_own.sum())} rows, on the other side's in {int(fired_their.sum())}")
            thr = float(np.quantile(delta, DEFINE_AGREE))
            agree = delta <= thr
            grp = np.where(~(n_claims > 0), 0, np.where(fired, 2, 1))          # 0: no open claim; 1: open claim, stake not above the spread; 2: above
            m = np.flatnonzero(agree)
            gb = GameBoot(rows["g"][te][m])
            for kind_label, (expo, obs, _nx) in (("any event", exp_any), ("takings", exp_tak)):
                pt, dr = gb.ratio(obs[m].astype(float), expo[m], grp[m], 3)
                lab = ["no open claim", "open claim, stake not above the spread", "open claim, stake above the spread"]
                parts = []
                for b in range(3):
                    lo, hi = band(pt[b], dr[:, b] * 100)
                    parts.append(f"{lab[b]} ({int((grp[m] == b).sum())} rows): {100 * pt[b]:.3f} [{lo:.3f}, {hi:.3f}]")
                rr = dr[:, 2] / np.maximum(dr[:, 0], 1e-12)
                lo, hi = band(0, rr)
                say(f"E2 [{dlabel}] [{cname}] {kind_label}: AMONG ROWS WHERE THE READINGS AGREE (|delta| at or under the {DEFINE_AGREE:.2f} quantile = {thr:.4f}; {len(m)} rows) rate per 100 plies: " + "; ".join(parts) + f"; above-spread over no-claim {pt[2] / max(pt[0], 1e-12):.3f} [{lo:.3f}, {hi:.3f}]")
                ctl = []
                for slabel, strat in (("ply", ply_band), ("units left", men_band)):
                    pm, dm = stratified_ratio(gb, obs[m], expo[m], grp[m], strat[m], BANDS)
                    ps, ds = stratified_ratio(gb, obs[m], expo[m], grp[m], strat[m], BANDS, rng)
                    lo, hi = band(pm, dm)
                    lo2, hi2 = band(ps, ds)
                    ctl.append(f"{slabel} held fixed {pm:.3f} [{lo:.3f}, {hi:.3f}], and with the labels shuffled inside each quintile of {slabel} {ps:.3f} [{lo2:.3f}, {hi2:.3f}]")
                say(f"E2 [{dlabel}] [{cname}] {kind_label}: CONTROLS, above-spread over no-claim with " + "; ".join(ctl))
        # E3: carrying; the claim's identity is (owner, kind): the stored rows keep no place and no read set
        key = {(int(g), int(p)): j for j, (g, p) in enumerate(zip(rows["g"][te], rows["ply"][te]))}
        pair = np.array([key.get((int(g), int(p) + 2), -1) for g, p in zip(rows["g"][te], rows["ply"][te])], np.int64)
        ok = pair >= 0
        ok &= np.where(ok, rows["mover"][te] == rows["mover"][te][np.where(ok, pair, 0)], False)
        n0, n2 = ok_[ok], ok_[pair[ok]]
        quiet2 = ~(exp_any[2][ok] <= rows["ply"][te][ok] + 1)           # no event of the rules at the two steps between the rows
        unread0 = d["st"]["unread"][d["ix"][te]][ok] > 0
        held = np.minimum(n0, n2).sum(axis=(1, 2)).astype(float)
        tot0 = n0.sum(axis=(1, 2)).astype(float)
        new = np.maximum(n2 - n0, 0).sum(axis=(1, 2)).astype(float)
        tot2 = n2.sum(axis=(1, 2)).astype(float)
        gb = GameBoot(rows["g"][te][ok])
        z = np.zeros(int(ok.sum()), np.int64)
        say(f"E3 CARRYING [{dlabel}] (the stored rows keep no place and no read set: a claim is an (owner, kind) count, and 'nothing changed' is read as no event of the rules at the two steps between; the true read-set test is NOT made): {int(ok.sum())} of {len(te)} held-out rows have a row two plies later in the same game with the same side to move; {int((tot0 > 0).sum())} of those hold an open claim")
        for label, mk in (("all such pairs", np.ones(len(z), bool)), ("pairs with no event of the rules between", quiet2), ("pairs with an event of the rules between", ~quiet2)):
            pt, dr = gb.ratio(held * mk, tot0 * mk, z, 1)
            lo, hi = band(pt[0], dr[:, 0])
            pn, dn = gb.ratio(new * mk, tot2 * mk, z, 1)
            lo2, hi2 = band(pn[0], dn[:, 0])
            say(f"E3 [{dlabel}] {label}: claims at ply n held again at n+2: {100.0 * pt[0]:.2f} pct [{100.0 * lo:.2f}, {100.0 * hi:.2f}] of {int((tot0 * mk).sum())} claims; claims at n+2 not held at n (FIRST SEEN at n+2): {100.0 * pn[0]:.2f} pct [{100.0 * lo2:.2f}, {100.0 * hi2:.2f}] of {int((tot2 * mk).sum())}")
        nm = new * quiet2
        pt, dr = gb.ratio(nm * unread0, nm, z, 1)
        lo, hi = band(pt[0], dr[:, 0])
        say(f"E3 [{dlabel}] of the claims first seen at n+2 in pairs with no event between ({int(nm.sum())}), the share whose row at n had lines the read left unread (a miss of the read's budget that carrying would meet at the next move): {100.0 * pt[0]:.2f} pct [{100.0 * lo:.2f}, {100.0 * hi:.2f}]; the others could still have been true at n (an upper bound: the truth at n is not stored)")


# ------------------------------------------------------------------------------------------------------------------------------ the commands

def make_say(out: Path, name: str):
    lines: list = []

    def say(m: str) -> None:
        lines.append(f"OBS readings: {m}")
        print(lines[-1], flush=True)
        (out / name).write_text("\n".join(lines) + "\n")
    return say


# ------------------------------------------------------------------------------------------------------------------------------ stage D: the match games

_READER = None


def _read_games(job: tuple) -> list:
    """Worker: for the games of one chunk, the inputs of the heads at every position where a player decided (from the 78.67 match record): [(game, ply, player, equal moves, changed, quiet, row of inputs)].
    The inputs are those of the fit, rebuilt from the moves by the play-time reader of tiebreak_number (verified there against the fit's own matrix)."""
    global _READER
    from catspace.incubator.analyst.discovery import tiebreak_number as TN
    if _READER is None:
        _READER = TN.Reader(threads=1)
    r = _READER
    games, lo_ply = job
    out = []
    for gi, rec in games:
        moves = rec["moves"]
        sts = r.sheet.subgoal_games([(None, moves)], 1)[0]
        want = [(p, pi, int(ties), int(ch)) for pi, st in enumerate(rec["stats"]) for p, ties, ch in st["log"] if p >= lo_ply and p < len(moves)]
        if not want:
            continue
        pos = [(None, list(moves[:p])) for p, _pi, _t, _c in want]
        sd = np.array([np.array(sts[p][:r.n_statements], np.float32) - np.array(sts[p][r.n_statements:], np.float32) for p, _pi, _t, _c in want], np.float32)
        X, keep = r.matrix(pos, sd)
        _kd, quiet = r.kinds_part([pos[int(i)] for i in keep])
        for k, i in enumerate(keep):
            p, pi, ties, ch = want[int(i)]
            out.append((gi, p, pi, ties, ch, bool(quiet[k]), X[k]))
    return out


def matches(a) -> int:
    """Stage D: on the 78.67 match games, at the positions where a player decided (the root of its search: the leaves where a search ended are not kept in the match files), how often the two readings agree,
    by whether the search held one move best or several equal, and by the player; the thresholds are those the held-out human rows gave in `measure`; no player is changed."""
    import multiprocessing
    t0 = time.perf_counter()
    out = out_dir(a.out_dir)
    say = make_say(out, "matches_obs.txt")
    f = Path(paths.experiment("tiebreak_78_67", "full", "matches", a.match + ".jsonl"))
    if not f.exists() or not (out / "thresholds.json").exists():
        say(f"STAGE D CUT: {'the match file ' + str(f) if not f.exists() else 'the thresholds of `measure` (run it on all held-out games first)'} is missing")
        return 1
    recs = [json.loads(x) for x in f.read_text().splitlines() if x.strip()][:a.games]
    thr = json.loads((out / "thresholds.json").read_text())
    saved = pickle.loads(heads_file(a.out_dir).read_bytes())
    LO = 10                                                          # the fit's rows start at this ply
    say(f"STAGE D: match record {f.name}: {len(recs)} games, the positions where a player decided from ply {LO} on, read at the root of its search (the match files keep no leaf); the readings are those of the heads fitted on the fit games; thresholds from the held-out human rows: agree = |delta| at or under {DEFINE_AGREE:.2f} quantile, large = above the 0.80 quantile; the ply cap of the matches stands for the record's end")
    chunks = [([(i, recs[i]) for i in range(c, min(c + 4, len(recs)))], LO) for c in range(0, len(recs), 4)]
    rows = []
    with multiprocessing.get_context("spawn").Pool(a.workers) as pool:
        for k, part in enumerate(pool.imap_unordered(_read_games, chunks)):
            rows += part
            if k % 5 == 0:
                print(f"  chunk {k + 1}/{len(chunks)}, positions {len(rows)}, {time.perf_counter() - t0:.0f} s", flush=True)
    if not rows:
        say("STAGE D CUT: no position could be read")
        return 1
    game = np.array([r[0] for r in rows])
    ply = np.array([r[1] for r in rows])
    player = np.array([r[2] for r in rows])
    ties = np.array([r[3] for r in rows])
    changed = np.array([r[4] for r in rows])
    quiet = np.array([r[5] for r in rows])
    X = np.stack([r[6] for r in rows])
    say(f"STAGE D: positions read {len(rows)} of {len(game)} (quiet {int(quiet.sum())} = {100.0 * quiet.mean():.1f} pct); the reading cost {time.perf_counter() - t0:.0f} s on {a.workers} workers")
    Xk = X[:, :len(saved["models"][(next(iter(CLASSES)), "us", "kinds")]["mean"])]
    cap = 120
    rem = np.maximum(1, cap - ply)
    d_heads = {}
    for cname in CLASSES:
        for rel in ("us", "them"):
            h = EH.Head.__new__(EH.Head)
            h.rel = rel
            h.active = np.array(saved["active"][(cname, rel)])
            d_heads[(cname, rel)] = h
    res = {}
    for cname in CLASSES:
        rr = {}
        for which, Xw in (("kinds", Xk), ("all", X)):
            ch = []
            for rel in ("us", "them"):
                m = saved["models"][(cname, rel, which)]
                ch.append(FH.side_chance(d_heads[(cname, rel)], FH.table_linear(m, Xw, np.arange(len(Xw))), rem))
            rr[which] = ch[0] - ch[1]
        res[cname] = np.abs(rr["all"] - rr["kinds"])
    for cname in CLASSES:
        v = res[cname]
        ta, tl = thr[cname]["agree"], thr[cname]["large"]
        for label, sel in (("all positions", np.ones(len(v), bool)), ("search held ONE move best", ties == 1), ("search held SEVERAL moves equal", ties >= 2), ("the choice differed from the engine's own move", changed == 1)):
            for who, ps in (("both players", np.ones(len(v), bool)), ("the reading player (a)", player == 0), ("the control or random (b)", player == 1)):
                m = np.flatnonzero(quiet & sel & ps)
                if len(m) < 30:
                    continue
                gb = GameBoot(game[m])
                z = np.zeros(len(m), np.int64)
                p1, d1 = gb.ratio((v[m] <= ta).astype(float), np.ones(len(m)), z, 1)
                p2, d2 = gb.ratio((v[m] > tl).astype(float), np.ones(len(m)), z, 1)
                l1, h1 = band(p1[0], d1[:, 0])
                l2, h2 = band(p2[0], d2[:, 0])
                say(f"D [{cname}] {label}, {who}: {len(m)} quiet positions: the readings AGREE in {100.0 * p1[0]:.2f} pct [{100.0 * l1:.2f}, {100.0 * h1:.2f}], differ LARGELY in {100.0 * p2[0]:.2f} pct [{100.0 * l2:.2f}, {100.0 * h2:.2f}]; mean |delta| {v[m].mean():.4f}")
        ref = thr[cname]["by_ply"]
        mm = np.flatnonzero(quiet)
        say(f"D [{cname}] REFERENCE: on the held-out human rows (by definition {100.0 * DEFINE_AGREE:.0f} pct agree) the share that agrees by ply: " + "; ".join(f"ply {k}-: {100.0 * v_:.1f} pct" for k, v_ in ref.items()) + "; in the match positions by ply: " + "; ".join(
            f"ply {lo}-{hi}: {100.0 * np.mean(v[mm][(ply[mm] >= lo) & (ply[mm] < hi)] <= ta):.1f} pct ({int(((ply[mm] >= lo) & (ply[mm] < hi)).sum())} positions)" for lo, hi in ((10, 20), (20, 40), (40, 80), (80, 400))))
    say(f"wall {time.perf_counter() - t0:.0f} s")
    return 0


def measure(a) -> int:
    t0 = time.perf_counter()
    out = out_dir(a.out_dir)
    say = make_say(out, "measure_obs.txt")
    d = load(say)
    saved = pickle.loads(heads_file(a.out_dir).read_bytes())
    te = test_rows(d, a.games)
    R = Readings(d, saved, te)
    thr = {c: {"agree": float(np.quantile(np.abs(R.delta[c]), DEFINE_AGREE)), "large": float(np.quantile(np.abs(R.delta[c]), 0.80)), "rows": int(len(te)),
               "by_ply": {str(lo): float(np.mean(np.abs(R.delta[c])[(d["rows"]["ply"][te] >= lo) & (d["rows"]["ply"][te] < hi)] <= np.quantile(np.abs(R.delta[c]), DEFINE_AGREE)))
                          for lo, hi in ((10, 20), (20, 40), (40, 80), (80, 400))}} for c in CLASSES}
    if not a.games:                                                   # the thresholds of stage D are those of the whole held-out set
        (out / "thresholds.json").write_text(json.dumps(thr))
    for name, fn in (("A", stage_a), ("B", stage_b), ("C", stage_c), ("E", stage_e)):
        t1 = time.perf_counter()
        fn(d, R, say)
        say(f"stage {name} wall {time.perf_counter() - t1:.0f} s")
    say(f"wall {time.perf_counter() - t0:.0f} s")
    return 0


def main(argv: "list | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--out-dir", default=None)
    m = sub.add_parser("measure")
    m.add_argument("--out-dir", default=None)
    m.add_argument("--games", type=int, default=0)
    x = sub.add_parser("matches")
    x.add_argument("--out-dir", default=None)
    x.add_argument("--games", type=int, default=60)
    x.add_argument("--match", default="trees_v_control_A")
    x.add_argument("--workers", type=int, default=9)
    a = ap.parse_args(argv)
    if a.cmd == "fit":
        fit(a.out_dir, make_say(out_dir(a.out_dir), "fit_obs.txt"))
        return 0
    if a.cmd == "measure":
        return measure(a)
    return matches(a)


if __name__ == "__main__":
    raise SystemExit(main())
