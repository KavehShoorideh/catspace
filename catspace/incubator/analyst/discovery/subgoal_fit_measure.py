"""PLAN 78.49 stages B, C, D: which statements, read in a QUIET position, foretell a CONFIRMED threat move arriving (SYSTEM_SPEC section 5: go toward the tactics)?  A measurement; nothing is wired into
play, nothing here is read by any player, and no subgoal enters any value (AGENTS rule 7).  Game-free: the statements are the engine's (rust/engine/tactics/src/sheet/subgoal.rs, `SheetModel.subgoal_games`).

B. THE STATEMENTS.  `SheetModel.subgoal_names()` lists them (name, one line of plain text); each yields one number per side at a position, the side to move and the other.  The forcing terms are the
   counts of threat moves by verdict from the reads kept by 78.48 (fuse 5000), which are not rerun.
C. THE SPARSE FIT.  Quiet positions as stage A defines them (forcing_arrival_measure.quiet_indices), the target as stage A defines it (arrival_targets: the mover's CONFIRMED threat move arrives within h plies
   less the other side's).  Features: ours less theirs of each statement, standardised on the fit half.  Fit half: the games with an even id; an L1-penalised linear model (scikit-learn LassoCV), the penalty
   chosen by 5-fold cross-validation GROUPED BY GAME on the fit half only.  Test half: the odd games only.  The rank correlation of the fit with the target (arrival, then material won less lost, the 78.44
   outcome) with a 90 pct band by bootstrap over games, beside the count difference of stage A and the fit shuffled within the game.  Two feature sets, labelled: P (the statements of the position alone) and
   P+F (those with the counts of threat moves by verdict).  Every kept statement is printed with its weight and its own single-statement rank correlation on the test half.
D. ONE STEP OF CONTROL.  Only for a horizon where C's P fit is CLEAR of stage A's line (the lower end of the fit's band above the upper end of the count difference's band, arrival target, test half).  For each
   quiet test position: the fitted number (P fit) on the position after the move the game's player chose, against the mean over all legal moves whose position after is not over.  SIGN CONVENTION: the number
   of a child is read from the side that moved, i.e. the fit's weights applied to (the statements of the side that moved less the other side's) on the position after the move; the engine gives a child's
   statements with the side to move of the child first, so the number is the weights on (the child's second list less its first).  A higher number is better for the side that moved.

  .venv/bin/python -m catspace.incubator.analyst.discovery.subgoal_fit_measure --games-dir D --reads R --opp-table T [--out-dir O] [--draws N] [--stage-d]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np

from catspace.incubator.analyst.discovery import forcing_arrival_measure as FA
from catspace.incubator.analyst.discovery import forcing_opportunity_measure as FO
from catspace.incubator.analyst.discovery import opportunity_measure as OM
from catspace.io import paths

HORIZONS = FA.HORIZONS
VERDICT_FEATURES = ("NOT SETTLED", "NOT BORNE OUT", "NO CLAIM")          # CONFIRMED is zero by the definition of a quiet position
FOLDS = 5
SHUFFLE_SEED = 7149
BOOT_SEED = 4449


# --------------------------------------------------------------------------------------------------------------------------- the statements and the reads (B)

def statements_of(variant: str, records: list, threads: int, ckpt: str) -> "tuple[list, list]":
    """(names, per game an array (positions, 2 x statements) with NaN where the position is over): from the engine in one call across threads, cached in `ckpt`."""
    from catspace.incubator.analyst.discovery import question_sheet as QS
    from catspace.incubator.analyst.discovery import sheet_rust as SR
    model = SR.model(QS.Env(variant), QS.store_worth(variant))
    names = model.subgoal_names()
    p = Path(ckpt)
    if p.exists():
        z = np.load(p)
        if int(z["n_games"]) == len(records) and int(z["n_stat"]) == len(names):
            return names, [z[f"g{i}"] for i in range(len(records))]
    t0 = time.perf_counter()
    res = model.subgoal_games([(r.get("start"), list(r["moves"])) for r in records], threads)
    n = len(names)
    arrs = [np.array([x if x is not None else [np.nan] * (2 * n) for x in g], dtype=float).reshape(len(g), 2 * n) for g in res]
    print(f"OBS subgoal statements of {len(records)} games in {time.perf_counter() - t0:.1f} s wall, threads {threads}", flush=True)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(p), n_games=len(records), n_stat=n, **{f"g{i}": a for i, a in enumerate(arrs)})
    return names, arrs


def verdict_counts(read: "dict | None") -> list:
    """The number of threat moves of one read at each of VERDICT_FEATURES (an unread move counts as not settled); an absent read counts nothing."""
    if read is None:
        return [0] * len(VERDICT_FEATURES)
    words = [FO.verdict_of(t[0]) for t in read["threats"]]
    return [sum(1 for w in words if w == v) for v in VERDICT_FEATURES]


def placed_diff(stat: np.ndarray) -> np.ndarray:
    """ours less theirs of every statement, per position: the first half of the columns less the second."""
    n = stat.shape[1] // 2
    return stat[:, :n] - stat[:, n:]


def forcing_diff(plies: list) -> np.ndarray:
    """ours less theirs of the counts of threat moves by verdict, per position (the other side's read absent: zero, as stage A)."""
    return np.array([[a - b for a, b in zip(verdict_counts(p["ours"]), verdict_counts(p["theirs"]))] if p else [0] * len(VERDICT_FEATURES) for p in plies], dtype=float).reshape(len(plies), len(VERDICT_FEATURES))


def table_of(records: list, reads_by_id: dict, stats: list, a_opp: dict) -> dict:
    """One row per quiet position of every game (stage A's definition): game (index into `records`), game id, ply, the statements' differences P, the forcing differences F, the count difference of stage
    A, the arrival target and the material outcome and worth won/lost at each horizon (NaN where the horizon runs past the game or has no outcome)."""
    idx = FO.outcome_arrays(a_opp)
    cols: dict = {"game": [], "id": [], "ply": [], "P": [], "F": [], "count": []}
    for h in HORIZONS:
        for k in ("arr", "out", "won", "lost"):
            cols[f"{k}{h}"] = []
    for g, r in enumerate(records):
        plies = reads_by_id[r["game"]]
        own, oth = FA.counts_of_game(plies)
        quiet = FA.quiet_indices(plies, own, oth)
        if not quiet:
            continue
        pd, fd = placed_diff(stats[g]), forcing_diff(plies)
        for t in quiet:
            cols["game"].append(g)
            cols["id"].append(r["game"])
            cols["ply"].append(t)
            cols["P"].append(pd[t])
            cols["F"].append(fd[t])
            cols["count"].append(own[t][0] - oth[t][0])
            for h, tg in zip(HORIZONS, FA.arrival_targets(own, len(plies), t)):
                cols[f"arr{h}"].append(np.nan if tg is None else tg[0] - tg[1])
                row = idx.get((g, t))
                for k in ("out", "won", "lost"):
                    cols[f"{k}{h}"].append(np.nan if row is None else float(a_opp[f"{k}{h}"][row]))
    out = {k: np.asarray(v, dtype=float) for k, v in cols.items() if k not in ("P", "F")}
    out["P"] = np.asarray(cols["P"], dtype=float).reshape(len(cols["game"]), -1)
    out["F"] = np.asarray(cols["F"], dtype=float).reshape(len(cols["game"]), -1)
    return out


def split_rows(t: dict) -> "tuple[np.ndarray, np.ndarray]":
    """(fit rows, test rows): the games with an even id fit, the odd ones are the test (the split of 78.48 and of stage A)."""
    even = t["id"] % 2 == 0
    return np.flatnonzero(even), np.flatnonzero(~even)


# --------------------------------------------------------------------------------------------------------------------------- the sparse fit (C)

def standardise(x: np.ndarray) -> "tuple[np.ndarray, np.ndarray]":
    """(mean, scale) of each column; a column that does not vary gets scale 1 (it stays zero)."""
    sd = x.std(axis=0)
    return x.mean(axis=0), np.where(sd > 0, sd, 1.0)


def fit_sparse(x: np.ndarray, y: np.ndarray, groups: np.ndarray, folds: int = FOLDS, jobs: int = -1) -> dict:
    """The L1-penalised linear fit of y on the standardised x; the penalty chosen by cross-validation grouped by `groups` (the games), on these rows only.
    -> {alpha, mean, scale, coef (on the standardised columns), intercept, alphas, cv_mse}"""
    from sklearn.linear_model import LassoCV
    from sklearn.model_selection import GroupKFold
    mean, scale = standardise(x)
    z = (x - mean) / scale
    cv = list(GroupKFold(n_splits=folds).split(z, y, groups))
    m = LassoCV(cv=cv, alphas=60, max_iter=20000, n_jobs=jobs).fit(z, y)
    return {"alpha": float(m.alpha_), "mean": mean, "scale": scale, "coef": m.coef_.copy(), "intercept": float(m.intercept_), "alphas": m.alphas_.copy(), "cv_mse": m.mse_path_.mean(axis=1)}


def predict(model: dict, x: np.ndarray) -> np.ndarray:
    return ((x - model["mean"]) / model["scale"]) @ model["coef"] + model["intercept"]


def raw_weights(model: dict) -> np.ndarray:
    """The weights on the unstandardised differences (the intercept is constant over the moves of a position and left out)."""
    return model["coef"] / model["scale"]


def shuffled_within_game(values: np.ndarray, games: np.ndarray, seed: int) -> np.ndarray:
    """`values` permuted among the rows of each game."""
    out = values.copy()
    rng = random.Random(seed)
    for g in np.unique(games):
        ix = np.flatnonzero(games == g).tolist()
        v = values[ix].tolist()
        rng.shuffle(v)
        out[ix] = v
    return out


def features(t: dict, which: str) -> np.ndarray:
    return t["P"] if which == "P" else np.hstack([t["P"], t["F"]])


def fit_all(t: dict, fit: np.ndarray, which: str, target: str, h: int, jobs: int = -1) -> "dict | None":
    """The fit of `target` ('arr' or 'out') at horizon h on the fit rows that have the target; None when there are too few."""
    y = t[f"{target}{h}"][fit]
    ok = ~np.isnan(y)
    if ok.sum() < 3 * FOLDS:
        return None
    rows = fit[ok]
    return fit_sparse(features(t, which)[rows], t[f"{target}{h}"][rows], t["game"][rows], jobs=jobs)


# --------------------------------------------------------------------------------------------------------------------------- the test (C)

def corr_fn(x: np.ndarray, y_name: str):
    """A function of a (sub)table: rank correlation of the column `x_name` with `y_name` over the rows where both are present."""
    return None


def spec_for(pred_col: str, y_col: str):
    def f(d: dict) -> float:
        ok = ~np.isnan(d[pred_col]) & ~np.isnan(d[y_col])
        return OM.spearman(d[pred_col][ok], d[y_col][ok])
    return f


def test_table(t: dict, test: np.ndarray, models: dict, which_sets: tuple) -> dict:
    """The arrays of the test rows with every reading's column beside: pred_<which>_<target><h>, shuf_<which>_<target><h>, count, plus the single statements s<i>."""
    a = {k: t[k][test] for k in t if k not in ("P", "F")}
    for (which, target, h), m in models.items():
        p = predict(m, features(t, which)[test])
        a[f"pred_{which}_{target}{h}"] = p
        a[f"shuf_{which}_{target}{h}"] = shuffled_within_game(p, a["game"], SHUFFLE_SEED + h)
    return a


def band_lines(label: str, a: dict, draws: int, readings: list) -> "tuple[list, dict]":
    """readings: [(what, pred column, target column)]: the value and the 90 pct band over games of each."""
    fns = [spec_for(c, y) for _w, c, y in readings]
    cis = OM.bootstrap(a, fns, draws, BOOT_SEED)
    lines, vals = [], {}
    for (what, c, y), f, (lo, hi) in zip(readings, fns, cis):
        v = f(a)
        vals[(c, y)] = (v, lo, hi)
        lines.append(f"OBS subgoal {label}: {what}: rank correlation {v:+.4f} [{lo:+.4f}, {hi:+.4f}] ({draws} draws)")
    return lines, vals


def clear_of_count(vals: dict, which: str, h: int) -> bool:
    """Stage D's condition at horizon h: the lower end of the P fit's band (arrival target) above the upper end of the count difference's band."""
    fit = vals[(f"pred_{which}_arr{h}", f"arr{h}")]
    cnt = vals[("count", f"arr{h}")]
    return bool(fit[1] > cnt[2])


# --------------------------------------------------------------------------------------------------------------------------- stage D

def control_stats(results: list, seed: int) -> dict:
    """results: per quiet test position (played number, numbers of all the moves listed). -> counts and means of: played less mean over moves; the same for one random listed move (a control); the
    played move's percentile among the moves (the share of listed moves it beats, ties half)."""
    rng = random.Random(seed)
    diffs, ctrl, pct = [], [], []
    for played, vals in results:
        if played is None or len(vals) < 2:
            continue
        mean = sum(vals) / len(vals)
        diffs.append(played - mean)
        ctrl.append(vals[rng.randrange(len(vals))] - mean)
        pct.append((sum(1 for v in vals if v < played) + 0.5 * sum(1 for v in vals if v == played)) / len(vals))
    n = len(diffs)
    return {"n": n, "mean_diff": float(np.mean(diffs)) if n else float("nan"), "above": sum(1 for d in diffs if d > 1e-12), "below": sum(1 for d in diffs if d < -1e-12),
            "ctrl_mean_diff": float(np.mean(ctrl)) if n else float("nan"), "ctrl_above": sum(1 for d in ctrl if d > 1e-12), "pct": float(np.mean(pct)) if n else float("nan")}


def stage_d(variant: str, records: list, t: dict, test: np.ndarray, model: dict, h: int, threads: int, label: str, draws: int) -> "tuple[list, dict]":
    """For every quiet test position the fitted number after the move played against the mean over the legal moves (see the module docstring for the sign)."""
    from catspace.incubator.analyst.discovery import question_sheet as QS
    from catspace.incubator.analyst.discovery import sheet_rust as SR
    sheet = SR.model(QS.Env(variant), QS.store_worth(variant))
    games = sorted(set(int(g) for g in t["game"][test]))
    wanted = [sorted(int(p) for p in t["ply"][test][t["game"][test] == g]) for g in games]
    w = [float(x) for x in raw_weights(model)]
    t0 = time.perf_counter()
    res = sheet.subgoal_children([(records[g].get("start"), list(records[g]["moves"])) for g in games], threads, wanted, w)
    lines = [f"OBS subgoal {label}: stage D h={h}: children read for {len(games)} test games in {time.perf_counter() - t0:.1f} s wall"]
    per_game = []
    for g, rows in zip(games, res):
        per_game.append([(dict(vals).get(played), [v for _i, v in vals]) for _ply, played, vals in rows if played is not None])
    st = control_stats([r for g in per_game for r in g], SHUFFLE_SEED)
    rng = np.random.default_rng(BOOT_SEED)
    boots = []
    for _ in range(draws):
        pick = rng.integers(0, len(per_game), len(per_game))
        s = control_stats([r for i in pick for r in per_game[i]], SHUFFLE_SEED)
        boots.append((s["mean_diff"], s["above"] / max(1, s["n"]), s["pct"]))
    bs = np.array(boots)
    lo = np.nanpercentile(bs, 5, axis=0)
    hi = np.nanpercentile(bs, 95, axis=0)
    lines.append(f"OBS subgoal {label}: stage D h={h}: quiet test positions with the played move's position read and two or more moves listed {st['n']}; the number after the played move less the mean over the legal moves: mean {st['mean_diff']:+.4f} [{lo[0]:+.4f}, {hi[0]:+.4f}], above the mean in {st['above']} ({100 * st['above'] / max(1, st['n']):.1f} pct) [{100 * lo[1]:.1f}, {100 * hi[1]:.1f}], below in {st['below']}")
    lines.append(f"OBS subgoal {label}: stage D h={h}: the played move beats {100 * st['pct']:.1f} pct of the listed moves on average [{100 * lo[2]:.1f}, {100 * hi[2]:.1f}] (50 is chance)")
    lines.append(f"OBS subgoal {label}: stage D h={h}: control, one random listed move in place of the played one: mean {st['ctrl_mean_diff']:+.4f}, above the mean in {st['ctrl_above']} ({100 * st['ctrl_above'] / max(1, st['n']):.1f} pct)")
    return lines, st


# --------------------------------------------------------------------------------------------------------------------------- the run

def run(variant: str, records: list, reads_by_id: dict, a_opp: dict, stats: list, names: list, threads: int, draws: int, label: str, do_d: bool, jobs: int = -1) -> "tuple[list, dict]":
    stat_names = [n for n, _text in names]
    feature_names = {"P": stat_names, "PF": stat_names + [f"threat moves {v.lower()}" for v in VERDICT_FEATURES]}
    t = table_of(records, reads_by_id, stats, a_opp)
    fit, test = split_rows(t)
    lines = [f"OBS subgoal {label}: statements {len(names)} per side; the position's quiet rows {len(t['ply'])} (fit half, even game ids: {len(fit)} rows, {len(np.unique(t['game'][fit]))} games; test half, odd: {len(test)} rows, {len(np.unique(t['game'][test]))} games)"]
    for n, text in names:
        lines.append(f"OBS subgoal {label}: statement {n}: {text}")
    models = {}
    for which in ("P", "PF"):
        for target in ("arr", "out"):
            for h in HORIZONS:
                m = fit_all(t, fit, which, target, h, jobs)
                if m is not None:
                    models[(which, target, h)] = m
                    lines.append(f"OBS subgoal {label}: fit {which} {'arrival' if target == 'arr' else 'material'} h={h}: penalty {m['alpha']:.5f} by {FOLDS}-fold cross-validation grouped by game on the even games, statements kept {int((m['coef'] != 0).sum())} of {len(m['coef'])}")
    a = test_table(t, test, models, ("P", "PF"))
    for i, nm in enumerate(feature_names["PF"]):
        a[f"s{i}"] = (t["P"] if i < len(stat_names) else t["F"])[test][:, i if i < len(stat_names) else i - len(stat_names)]
    data = {"label": label, "vals": {}}
    allvals = {}
    for target, tname in (("arr", "the arrival target (ours arrives less theirs)"), ("out", "material won less lost (the 78.44 outcome)")):
        for h in HORIZONS:
            y = f"{target}{h}"
            ok = ~np.isnan(a[y])
            lines.append(f"OBS subgoal {label}: test half h={h} {tname}: rows with the target {int(ok.sum())}, mean {np.nanmean(a[y]):+.4f}")
            readings = [(f"h={h} {tname}: fit P", f"pred_P_{y}", y), (f"h={h} {tname}: fit P+F", f"pred_PF_{y}", y), (f"h={h} {tname}: count difference of stage A", "count", y),
                        (f"h={h} {tname}: fit P shuffled within the game", f"shuf_P_{y}", y), (f"h={h} {tname}: fit P+F shuffled within the game", f"shuf_PF_{y}", y)]
            readings = [r for r in readings if r[1] in a]
            more, vals = band_lines(label, a, draws, readings)
            lines += more
            allvals.update(vals)
    # the kept statements with weights and their own rank correlation on the test half
    for (which, target, h), m in sorted(models.items(), key=lambda kv: (kv[0][1] != "arr", kv[0][2], kv[0][0])):
        y = f"{target}{h}"
        kept = [i for i, c in enumerate(m["coef"]) if c != 0]
        for i in sorted(kept, key=lambda j: -abs(m["coef"][j])):
            a[f"s{i}"] if False else None
            f = spec_for(f"s{i}", y)
            lines.append(f"OBS subgoal {label}: kept {which} {'arrival' if target == 'arr' else 'material'} h={h}: {feature_names['PF'][i]} (ours less theirs): weight {m['coef'][i]:+.5f} per standard deviation, single-statement rank correlation on the test half {f(a):+.4f}")
        if not kept:
            lines.append(f"OBS subgoal {label}: kept {which} {'arrival' if target == 'arr' else 'material'} h={h}: none (the penalty chosen leaves every weight zero)")
    data["vals"] = {f"{k[0]}|{k[1]}": v for k, v in allvals.items()}
    if do_d:
        for h in HORIZONS:
            m = models.get(("P", "arr", h))
            if m is None or not (m["coef"] != 0).any():
                lines.append(f"OBS subgoal {label}: stage D h={h}: skipped, the P fit keeps no statement")
                continue
            if not clear_of_count(allvals, "P", h):
                lines.append(f"OBS subgoal {label}: stage D h={h}: skipped, the P fit is not clear of the count difference of stage A (lower end of its band not above the upper end of the count's)")
                continue
            more, st = stage_d(variant, records, t, test, m, h, threads, label, draws)
            lines += more
    return lines, data


def main(argv: "list | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--games-dir", required=True)
    ap.add_argument("--reads", required=True, help="the reads kept by 78.48 (reads_fuse*.jsonl)")
    ap.add_argument("--opp-table", required=True, help="the 78.44 table of the games (opp_table_n*.npz of 78.48): the material outcome")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--games", type=int, default=0, help="a subset: the first N games by id")
    ap.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--draws", type=int, default=300)
    ap.add_argument("--stage-d", action="store_true")
    a = ap.parse_args(argv)
    records = FO.read_games(a.games_dir)
    variant = records[0]["variant"]
    label = Path(a.games_dir).name
    out = Path(a.out_dir or paths.experiment("subgoal_fit_78_49", label))
    reads_by_id = {}
    with open(a.reads) as f:
        for line in f:
            g = json.loads(line)
            reads_by_id[g["game"]] = g["plies"]
    records = sorted(records, key=lambda r: r["game"])
    z = np.load(a.opp_table)
    a_opp = {k: z[k] for k in z.files if k != "n_games"}
    if a.games:
        records = records[:a.games]
    t0 = time.perf_counter()
    names, stats = statements_of(variant, records, a.threads, str(out / f"statements_n{len(records)}.npz"))
    lines, data = run(variant, records, reads_by_id, a_opp, stats, names, a.threads, a.draws, label, a.stage_d)
    lines.append(f"OBS subgoal {label}: total wall {time.perf_counter() - t0:.1f} s")
    for line in lines:
        print(line, flush=True)
    out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").write_text(json.dumps({"lines": lines, "data": data}, indent=1, default=float))
    (out / "obs.txt").write_text("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
