"""PLAN 78.59 C AND D: THE RESULT HEAD FROM HUMAN GAMES, ITS KIND TERMS, AND THE CORRECTION BY CONCEPTS. Reads the rows of human_heads_data.py (no engine call here). A measurement in a sandbox; nothing is wired
into any player; no engine evaluation enters; game-free (kinds, statements and concepts are raw symbols read from the rows' metadata).

C. THE RESULT: the probability of win, draw, loss of the side to move, regularised multinomial logistic (L2, the penalty by 5-fold cross-validation grouped by game on the fit games). Input sets, nested:
   R = mean rating, rating difference (mover less other), ply;  K = kinds ours less theirs;  R+K;  R+K+S (the 26 statements);  R+K+S+C (the kept concepts, on the thinned rows, the others refitted on the same rows).
   Kind columns are raw counts (a kind's term is per extra unit, as in 78.58); the other columns are centred and scaled on the fit games. Reported on the TEST games: log loss of the base rate and of each set with a 90 pct
   band by bootstrap over games, the gain of each set over the one before, calibration in ten equal-count bins, and the log loss by band of rating.
   THE ORDER OF KINDS: the terms of K fitted on the fit games (log-odds of a win against a loss per extra unit; draw the reference), with a 90 pct band over refits on resampled fit games, beside the order of the
   self-play record (78.58); again within each band of rating, and in R+K.
D. THE CORRECTION: for the TOP_CONCEPTS concepts with the largest terms in R+K+S+C: on the test concept rows where the concept is held more by the side to move than by the other (and the reverse), the change in the
   expected score (win plus half draw) when that one column is set to zero, everything else (kinds above all) held as it is; a band over games; by band of rating; and the same read without the model: the observed score
   of rows that hold the concept less the rows that do not, within rows of equal kinds (matched). And how each kind's term moves when the statements, then the concepts, are added (refits on the concept rows, a band over games).

  .venv/bin/python -m catspace.incubator.analyst.discovery.human_heads_result --rows R --out-dir O [--draws D] [--procs K]
"""
from __future__ import annotations

import argparse
import datetime
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np

from catspace.incubator.analyst.discovery import estimator_fit as EF
from catspace.incubator.analyst.discovery import estimator_heads as EH
from catspace.incubator.analyst.discovery import human_heads_data as HD
from catspace.io import paths

C_GRID = (0.001, 0.01, 0.1, 1.0, 10.0)
FOLDS = 5
TOP_CONCEPTS = 20
SEED = 5959
_G: dict = {}                                                      # shared with forked workers


# ------------------------------------------------------------------------------------------------------------------------------ the design

def raw_columns(rows: dict, sets: "list | None" = None) -> "tuple[dict, dict]":
    """{set name: (raw matrix over ALL rows, column names)}: R, K, S as above; the concept columns are over the thinned rows only (see `cmask`)."""
    meta = rows["meta"]
    rm, ro = rows["rm"].astype(float), rows["ro"].astype(float)
    r = np.column_stack([(rm + ro) / 2, rm - ro, rows["ply"].astype(float)])
    cols = {"R": (r, ["mean rating", "rating difference (mover less other)", "ply"]),
            "K": (rows["kd"].astype(float), list(meta["kinds"])),
            "S": (rows["sd"].astype(float), list(meta["statements"]))}
    return cols, meta


def stack(cols: dict, names: str) -> "tuple[np.ndarray, list]":
    """The raw matrix of the sets named in `names` (letters, in order) and the column names."""
    mats, nm = [], []
    for ch in names:
        m, n = cols[ch]
        mats.append(m)
        nm += [f"{ch}:{x}" for x in n]
    return np.hstack(mats), nm


def transform_of(x: np.ndarray, names: list, fit: np.ndarray) -> "tuple[np.ndarray, np.ndarray]":
    """(centre, scale) of each column from the fit rows; kind columns stay raw (centre 0, scale 1) so a kind's term is per extra unit."""
    c = x[fit].mean(axis=0)
    s = x[fit].std(axis=0)
    s = np.where(s > 0, s, 1.0)
    for j, n in enumerate(names):
        if n.startswith("K:"):
            c[j], s[j] = 0.0, 1.0
    return c, s


# ------------------------------------------------------------------------------------------------------------------------------ fitting (forked workers)

class FreshPool:
    """A pool forked at each call, so the workers see the shared table `_G` as it is at that moment."""

    def __init__(self, n: int):
        self.n = n

    def map(self, fn, tasks):
        with mp.get_context("fork").Pool(self.n) as p:
            return p.map(fn, tasks, chunksize=1)

    def close(self):
        pass

    def join(self):
        pass


def _fit_task(args):
    kind, xs, ys, c = args
    return EF.fit_lr(xs, ys, c)


def _cv_task(args):
    c, tr, te = args
    x, y = _G["x"], _G["y"]
    m = EF.fit_lr(x[tr], y[tr], c)
    return c, EF.logloss(EF.probs(m, x[te]), y[te])


def choose_c(pool, x, y, groups) -> "tuple[float, dict]":
    from sklearn.model_selection import GroupKFold
    _G["x"], _G["y"] = x, y
    cv = list(GroupKFold(n_splits=FOLDS).split(x, y, groups))
    res = pool.map(_cv_task, [(c, tr, te) for c in C_GRID for tr, te in cv])
    sc = {c: float(np.mean([v for cc, v in res if cc == c])) for c in C_GRID}
    return min(sc, key=sc.get), sc


def _terms_task(args):
    """Refit on resampled games: the kind terms of the first `nk` columns."""
    idx, c, nk = args
    x, y = _G["xb"], _G["yb"]
    m = EF.fit_lr(x[idx], y[idx], c)
    return EF.kind_terms(list(m.classes_), m.coef_[:, :nk])


def boot_terms(pool, x, y, game, sel, c, nk, draws, seed):
    """(draws x nk) kind terms of refits on resampled games among the rows `sel`."""
    _G["xb"], _G["yb"] = x, y
    games = np.unique(game[sel])
    by = {int(g): sel[game[sel] == g] for g in games}
    rng = np.random.default_rng(seed)
    tasks = []
    for _ in range(draws):
        pick = rng.choice(games, size=len(games), replace=True)
        tasks.append((np.concatenate([by[int(g)] for g in pick]), c, nk))
    return np.array(pool.map(_terms_task, tasks))


def bands_of(draws: np.ndarray) -> "tuple[np.ndarray, np.ndarray]":
    return np.percentile(draws, 5, axis=0), np.percentile(draws, 95, axis=0)


# ------------------------------------------------------------------------------------------------------------------------------ the pieces

def loss_band(ll: np.ndarray, game: np.ndarray, draws: int, seed: int) -> "tuple[float, float, float]":
    tab = EF.per_game_table(game, [ll])
    return EF.game_bootstrap(tab, lambda t: t[:, 1].sum() / t[:, 0].sum(), draws, seed)


def model_head(m, c, centre, scale, names, source_data, what) -> dict:
    return {"type": "probability", "inputs": names, "transform": {"centre": [float(v) for v in centre], "scale": [float(v) for v in scale]}, "model": EF.model_record(m, c),
            "source": "human games", "data": source_data, "date": datetime.date.today().isoformat(), "what": what}


def head_probs(head: dict, x: np.ndarray) -> np.ndarray:
    """A probability head of the record on RAW inputs: the transform of the head, then the record's model (estimator_heads.head_predict)."""
    t = head.get("transform")
    z = (x - np.asarray(t["centre"])) / np.asarray(t["scale"]) if t else x
    return EH.head_predict({"heads": {"h": head}}, "h", z)


def main(argv: "list | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--rows", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("--procs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args(argv)
    t0 = time.perf_counter()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    lines: list = []

    def say(msg: str) -> None:
        lines.append(f"OBS human result: {msg}")
        print(lines[-1], flush=True)

    rows = HD.load_rows(Path(a.rows))
    cols, meta = raw_columns(rows)
    y, game, fit = rows["y"].astype(int), rows["game"], rows["fit"]
    cm = rows["cmask"]
    cpos = np.cumsum(cm) - 1                                    # a row's place among the concept rows
    cx = rows["cx"].astype(float)
    cnames = meta["concepts"]
    fit_i, test_i = np.flatnonzero(fit), np.flatnonzero(~fit)
    say(f"data: rows {len(y)}, fit {len(fit_i)} ({len(np.unique(game[fit_i]))} games), test {len(test_i)} ({len(np.unique(game[test_i]))} games); kinds {meta['kinds']}; statements {len(meta['statements'])}; concepts {len(cnames)} on {int(cm.sum())} rows")
    pool = FreshPool(a.procs)
    heads: dict = {}
    data_note = {"sample": "human games, lichess 2019-01 shards (see data_obs.txt), Normal terminations, 20..200 plies", "rows": int(len(y)), "fit_rows": int(len(fit_i)), "test_rows": int(len(test_i)),
                 "split": "by game, hash 7 in 10 fit", "quiet": "nothing stands to be lost by inclusion for either side (position_table)", "from_ply": HD.OPEN_PLIES}

    # ---------------------------------------------------------------- the sets on all quiet rows (no concepts)
    sets_all = [("R", "R"), ("K", "K"), ("R+K", "RK"), ("R+K+S", "RKS")]
    p_test = {"base rate": EF.base_probs(y[fit_i], len(test_i))}
    fitted = {}
    for label, letters in sets_all:
        x, nm = stack(cols, letters)
        centre, scale = transform_of(x, nm, fit_i)
        z = (x - centre) / scale
        c, sc = choose_c(pool, z[fit_i], y[fit_i], game[fit_i])
        m = EF.fit_lr(z[fit_i], y[fit_i], c)
        p_test[label] = EF.probs(m, z[test_i])
        fitted[label] = (m, c, centre, scale, nm)
        say(f"{label}: penalty C {c} by {FOLDS}-fold cross-validation grouped by game ({', '.join(f'C={k}: {v:.5f}' for k, v in sc.items())}); {z.shape[1]} columns")
        heads[f"result_human_{letters}"] = model_head(m, c, centre, scale, nm, data_note, f"the result of the side to move from {label} (human games)")
    yt, gt = y[test_i], game[test_i]
    per = {}
    for nm_, p in p_test.items():
        ll = -np.log(p[np.arange(len(yt)), yt])
        per[nm_] = ll
        v, lo, hi = loss_band(ll, gt, a.draws, SEED)
        say(f"[all quiet rows] test log loss, {nm_}: {v:.5f} [{lo:.5f}, {hi:.5f}] over {len(yt)} rows of {len(np.unique(gt))} games ({a.draws} draws over games)")
    names_l = list(p_test)
    for lo_name, hi_name in (("base rate", "R"), ("base rate", "K"), ("R", "R+K"), ("K", "R+K"), ("R+K", "R+K+S")):
        d = per[lo_name] - per[hi_name]
        v, lo, hi = loss_band(d, gt, a.draws, SEED + 1)
        say(f"[all quiet rows] test log loss gain of {hi_name} over {lo_name}: {v:+.5f} [{lo:+.5f}, {hi:+.5f}] (positive: {hi_name} is better)")
    for nm_, p in p_test.items():
        say(f"[all quiet rows] calibration {nm_} (predicted score > observed, ten equal-count bins): " + " ".join(f"{mp:.4f}>{ob:.4f}" for _n, mp, ob, _w, _o in EF.calibration(p, yt)))
    for b, bn in enumerate(HD.BAND_NAMES):
        sel = rows["band"][test_i] == b
        s = []
        for nm_ in names_l:
            ll = per[nm_][sel]
            v, lo, hi = loss_band(ll, gt[sel], a.draws, SEED + 2)
            s.append(f"{nm_} {v:.4f} [{lo:.4f}, {hi:.4f}]")
        say(f"[band {bn}] test log loss over {int(sel.sum())} rows of {len(np.unique(gt[sel]))} games: " + "; ".join(s))

    # ---------------------------------------------------------------- the same on the concept rows, with the concepts
    ci_fit = fit_i[cm[fit_i]]
    ci_test = test_i[cm[test_i]]
    pc = {"base rate": EF.base_probs(y[ci_fit], len(ci_test))}
    fitted_c = {}
    for label, letters in (("R+K", "RK"), ("R+K+S", "RKS"), ("R+K+S+C", "RKSC")):
        if letters.endswith("C"):
            xs, nm = stack(cols, "RKS")
            x = np.hstack([xs[cm], cx])
            nm = nm + [f"C:{n}" for n in cnames]
            fitm = cpos[ci_fit]
            tstm = cpos[ci_test]
        else:
            xs, nm = stack(cols, letters)
            x = xs[cm]
            fitm, tstm = cpos[ci_fit], cpos[ci_test]
        centre, scale = transform_of(x, nm, fitm)
        z = (x - centre) / scale
        c, sc = choose_c(pool, z[fitm], y[ci_fit], game[ci_fit])
        m = EF.fit_lr(z[fitm], y[ci_fit], c)
        pc[label] = EF.probs(m, z[tstm])
        fitted_c[label] = (m, c, centre, scale, nm, z, fitm, tstm)
        say(f"{label} on the concept rows: penalty C {c} ({', '.join(f'C={k}: {v:.5f}' for k, v in sc.items())}); {z.shape[1]} columns")
        heads[f"result_human_{letters}_thinned"] = model_head(m, c, centre, scale, nm, {**data_note, "rows": int(cm.sum()), "thinned": f"every {HD.CONCEPT_EVERY}th quiet row of a game"}, f"the result from {label} on the concept rows")
    ytc, gtc = y[ci_test], game[ci_test]
    perc = {}
    for nm_, p in pc.items():
        ll = -np.log(p[np.arange(len(ytc)), ytc])
        perc[nm_] = ll
        v, lo, hi = loss_band(ll, gtc, a.draws, SEED + 3)
        say(f"[concept rows] test log loss, {nm_}: {v:.5f} [{lo:.5f}, {hi:.5f}] over {len(ytc)} rows of {len(np.unique(gtc))} games")
    for lo_name, hi_name in (("base rate", "R+K"), ("R+K", "R+K+S"), ("R+K+S", "R+K+S+C"), ("R+K", "R+K+S+C")):
        d = perc[lo_name] - perc[hi_name]
        v, lo, hi = loss_band(d, gtc, a.draws, SEED + 4)
        say(f"[concept rows] test log loss gain of {hi_name} over {lo_name}: {v:+.5f} [{lo:+.5f}, {hi:+.5f}] (positive: {hi_name} is better)")
    for nm_, p in pc.items():
        say(f"[concept rows] calibration {nm_} (predicted score > observed, ten equal-count bins): " + " ".join(f"{mp:.4f}>{ob:.4f}" for _n, mp, ob, _w, _o in EF.calibration(p, ytc)))
    for b, bn in enumerate(HD.BAND_NAMES):
        sel = rows["band"][ci_test] == b
        s = []
        for nm_ in pc:
            v, lo, hi = loss_band(perc[nm_][sel], gtc[sel], a.draws, SEED + 5)
            s.append(f"{nm_} {v:.4f} [{lo:.4f}, {hi:.4f}]")
        say(f"[concept rows, band {bn}] test log loss over {int(sel.sum())} rows of {len(np.unique(gtc[sel]))} games: " + "; ".join(s))

    # ---------------------------------------------------------------- the order of kinds
    kn = meta["kinds"]
    nk = len(kn)
    mK, cK, _c, _s, _n = fitted["K"]
    termsK = EF.kind_terms(list(mK.classes_), mK.coef_)
    xk = cols["K"][0]
    dr = boot_terms(pool, xk, y, game, fit_i, cK, nk, a.draws, SEED + 6)
    lo, hi = bands_of(dr)
    order = np.argsort(termsK)
    say("ORDER OF KINDS, human games, kinds alone (term = log-odds of a win against a loss per extra unit, 90 pct band over refits on resampled fit games): "
        + "; ".join(f"{kn[k]} {termsK[k]:+.4f} [{lo[k]:+.4f}, {hi[k]:+.4f}]" for k in order))
    say("ORDER OF KINDS, human games, as a multiple of the lowest-term kind's term: " + "; ".join(f"{kn[k]} {termsK[k] / termsK[order[0]]:.2f}" for k in order))
    unordered = []
    for i in range(nk):
        for j in range(i + 1, nk):
            f = float((dr[:, j] > dr[:, i]).mean())
            if 0.05 < f < 0.95:
                unordered.append((kn[i], kn[j], f))
    say(f"kinds the human data cannot order (pairs, share of draws): {unordered if unordered else 'none'}")
    rec58 = json.loads(Path(paths.experiment("estimator_78_58", "full", "estimator.json")).read_text())
    sp = {d["kind"]: d for d in rec58["order_of_kinds"]}
    say("ORDER OF KINDS, self-play record of 78.58 beside: " + "; ".join(f"{k} {sp[k]['term']:+.4f} [{sp[k]['lo']:+.4f}, {sp[k]['hi']:+.4f}]" for k in sorted(sp, key=lambda k: sp[k]["term"])))
    say("ORDER OF KINDS, rank of each kind (1 lowest) human: " + ", ".join(f"{kn[k]} {r + 1}" for r, k in enumerate(order)) + "; self-play: " + ", ".join(f"{k} {r + 1}" for r, k in enumerate(sorted(sp, key=lambda k: sp[k]['term']))))
    heads["kind_baseline_human"] = {"type": "terms", "inputs": "none (a term per kind)", "kinds": kn, "terms": [float(v) for v in termsK], "order_of_kinds": [{"kind": kn[k], "term": float(termsK[k]), "lo": float(lo[k]), "hi": float(hi[k])} for k in order],
                                    "source": "human games", "data": data_note, "date": datetime.date.today().isoformat()}
    # in R+K
    mRK, cRK, _c, _s, nmRK = fitted["R+K"]
    tRK = EF.kind_terms(list(mRK.classes_), mRK.coef_[:, 3:3 + nk])
    xrk, _ = stack(cols, "RK")
    centre, scale = fitted["R+K"][2], fitted["R+K"][3]
    zrk = (xrk - centre) / scale
    dr2 = boot_terms(pool, zrk, y, game, fit_i, cRK, 3 + nk, a.draws, SEED + 7)
    # boot_terms returns the terms of the first 3 + nk columns; the kind columns come after the three rating columns
    lo2, hi2 = bands_of(dr2[:, 3:3 + nk]) if dr2.shape[1] >= 3 + nk else bands_of(dr2)
    say("kind terms with the ratings in the model (R+K), as kinds alone: " + "; ".join(f"{kn[k]} {tRK[k]:+.4f} [{lo2[k]:+.4f}, {hi2[k]:+.4f}] against {termsK[k]:+.4f}" for k in order))
    rcoef = (mRK.coef_[0] - mRK.coef_[2])[:3] if len(mRK.classes_) == 3 else None
    if rcoef is not None:
        say(f"R+K rating terms (win less loss, per standard deviation of the column; mean rating, rating difference, ply): {', '.join(f'{v:+.4f}' for v in rcoef)}")
    # per band of rating
    band_terms = {}
    for b, bn in enumerate(HD.BAND_NAMES):
        sel = fit_i[rows["band"][fit_i] == b]
        m = EF.fit_lr(xk[sel], y[sel], cK)
        t = EF.kind_terms(list(m.classes_), m.coef_)
        d = boot_terms(pool, xk, y, game, sel, cK, nk, max(50, a.draws // 2), SEED + 10 + b)
        l, h = bands_of(d)
        band_terms[bn] = (t, l, h)
        say(f"ORDER OF KINDS, band {bn} ({len(np.unique(game[sel]))} fit games): " + "; ".join(f"{kn[k]} {t[k]:+.4f} [{l[k]:+.4f}, {h[k]:+.4f}]" for k in np.argsort(t)))
    for k in range(nk):
        lo_b, hi_b = band_terms[HD.BAND_NAMES[0]], band_terms[HD.BAND_NAMES[-1]]
        say(f"term of {kn[k]} from the lowest band to the highest: {lo_b[0][k]:+.4f} [{lo_b[1][k]:+.4f}, {lo_b[2][k]:+.4f}] to {hi_b[0][k]:+.4f} [{hi_b[1][k]:+.4f}, {hi_b[2][k]:+.4f}]")
    heads["kind_baseline_human"]["by_band"] = {bn: [float(v) for v in t[0]] for bn, t in band_terms.items()}

    # ---------------------------------------------------------------- D: the concepts and the correction
    m4, cfull, centre4, scale4, nm4, z4, fitm, tstm = fitted_c["R+K+S+C"]
    coef = m4.coef_
    cl = list(m4.classes_)
    full = np.zeros((3, coef.shape[1]))
    full[cl] = coef
    termall = full[0] - full[2]
    c0 = 3 + nk + len(meta["statements"])
    cterm = termall[c0:]
    top = np.argsort(-np.abs(cterm))[:TOP_CONCEPTS]
    cdiff = cx[:, :]                                           # over the concept rows
    p_all = EF.predict_record(EF.model_record(m4, cfull), z4[tstm])
    base_score = p_all[:, 0] + 0.5 * p_all[:, 1]
    tg, tb = game[ci_test], rows["band"][ci_test]
    ty = y[ci_test]
    obs_score = (ty == 0) + 0.5 * (ty == 1)
    kd_test = rows["kd"][ci_test]
    kd_keys = np.array([hash(tuple(r)) for r in kd_test.tolist()])
    dstats = []
    for j in top:
        zz = z4[tstm].copy()
        col = c0 + j
        zero = (0.0 - centre4[col]) / scale4[col]
        zz[:, col] = zero
        p0 = EF.predict_record(EF.model_record(m4, cfull), zz)
        eff = base_score - (p0[:, 0] + 0.5 * p0[:, 1])               # change in the mover's score from the concept being as held
        held = cdiff[tstm, j] > 0
        against = cdiff[tstm, j] < 0
        res = [cnames[j], float(cterm[j])]
        for nm_, msk in (("more for the mover", held), ("more for the other side", against)):
            tab = EF.per_game_table(tg, [eff * msk, msk.astype(float)])
            v, lo_, hi_ = EF.game_bootstrap(tab, lambda t: t[:, 1].sum() / max(t[:, 2].sum(), 1.0), a.draws, SEED + 20)
            res += [nm_, int(msk.sum()), v, lo_, hi_]
        # matched: observed score of rows holding it (>0) less the rows not, within equal kinds, mover's side
        hold = cdiff[tstm, j] > 0
        notholdd = cdiff[tstm, j] == 0
        keys = kd_keys
        num = np.zeros(len(keys))
        # per kinds-vector stratum: (mean of holders - mean of non-holders) weighted by the number of holders
        order_k = np.argsort(keys, kind="stable")
        ks = keys[order_k]
        bounds = np.flatnonzero(np.diff(ks)) + 1
        starts = np.concatenate([[0], bounds])
        ends = np.concatenate([bounds, [len(ks)]])
        m_eff = np.zeros(len(keys))
        m_w = np.zeros(len(keys))
        for s, e in zip(starts, ends):
            idx = order_k[s:e]
            h_, n_ = hold[idx], notholdd[idx]
            if h_.any() and n_.any():
                d = obs_score[idx][n_].mean()
                for r in idx[h_]:
                    m_eff[r] = obs_score[r] - d
                    m_w[r] = 1.0
        tab = EF.per_game_table(tg, [m_eff, m_w])
        mv, mlo, mhi = EF.game_bootstrap(tab, lambda t: t[:, 1].sum() / max(t[:, 2].sum(), 1.0), a.draws, SEED + 21)
        res += ["matched", int(m_w.sum()), mv, mlo, mhi]
        bandres = []
        for b, bn in enumerate(HD.BAND_NAMES):
            sb = (tb == b) & held
            tabb = EF.per_game_table(tg[tb == b], [(eff * held)[tb == b], held[tb == b].astype(float)])
            vb, lb, hb = EF.game_bootstrap(tabb, lambda t: t[:, 1].sum() / max(t[:, 2].sum(), 1.0), max(50, a.draws // 2), SEED + 22 + b)
            bandres.append((bn, int(sb.sum()), vb, lb, hb))
        res.append(bandres)
        dstats.append(res)
    for res in dstats:
        name, term = res[0], res[1]
        s = f"D concept {name}: term in R+K+S+C {term:+.4f} per standard deviation; "
        for k in range(2, 12, 5):
            nm_, n_, v, l, h = res[k:k + 5]
            s += f"{nm_} (rows {n_}): change in the mover's expected score when the concept is as held {v:+.4f} [{l:+.4f}, {h:+.4f}]; "
        s += "by band (concept held more by the mover): " + ", ".join(f"{bn} rows {n_}: {v:+.4f} [{l:+.4f}, {h:+.4f}]" for bn, n_, v, l, h in res[-1])
        say(s)
    # how the kind terms move when statements, then concepts, are added (concept rows)
    base3 = {}
    for label in ("R+K", "R+K+S", "R+K+S+C"):
        m, c, centre, scale, nm, z, fitm_, tstm_ = fitted_c[label]
        base3[label] = EF.kind_terms(list(m.classes_), m.coef_[:, 3:3 + nk])
    dd = {}
    for label in ("R+K+S", "R+K+S+C"):
        m, c, centre, scale, nm, z, fitm_, tstm_ = fitted_c[label]
        _G["xb"], _G["yb"] = z, y[cm]
        gg = game[cm]
        gfit = ci_fit
        games = np.unique(game[ci_fit])
        by = {int(g): cpos[ci_fit[game[ci_fit] == g]] for g in games}
        rng = np.random.default_rng(SEED + 30)
        tasks = [(np.concatenate([by[int(g)] for g in rng.choice(games, size=len(games), replace=True)]), c, 3 + nk) for _ in range(max(50, a.draws // 2))]
        dd[label] = np.array(pool.map(_terms_task, tasks))[:, 3:3 + nk]
    zr = fitted_c["R+K"][5]
    gamesr = np.unique(game[ci_fit])
    byr = {int(g): cpos[ci_fit[game[ci_fit] == g]] for g in gamesr}
    _G["xb"], _G["yb"] = zr, y[cm]
    rng = np.random.default_rng(SEED + 30)
    tasks = [(np.concatenate([byr[int(g)] for g in rng.choice(gamesr, size=len(gamesr), replace=True)]), fitted_c["R+K"][1], 3 + nk) for _ in range(max(50, a.draws // 2))]
    dd["R+K"] = np.array(pool.map(_terms_task, tasks))[:, 3:3 + nk]
    for label in ("R+K+S", "R+K+S+C"):
        shift = dd[label] - dd["R+K"]                          # the same resample (same seed) of games: a paired band of how the term moved
        l, h = bands_of(shift)
        say(f"kind terms moved by adding {'the statements' if label == 'R+K+S' else 'the statements and the concepts'} to R+K (concept rows, paired refits over games): "
            + "; ".join(f"{kn[k]} {base3['R+K'][k]:+.4f} to {base3[label][k]:+.4f} (moved {base3[label][k] - base3['R+K'][k]:+.4f} [{l[k]:+.4f}, {h[k]:+.4f}])" for k in order))
    heads["result_human_RKSC_thinned"]["corrections"] = [{"concept": r[0], "term": r[1]} for r in dstats]
    pool.close()
    pool.join()
    rec = json.loads(Path(paths.experiment("estimator_78_58", "full", "estimator_heads.json")).read_text())
    for k, v in heads.items():
        assert k not in rec["heads"], k
        rec["heads"][k] = v
    (out / "estimator_heads_human.json").write_text(json.dumps(rec, indent=1))
    say(f"record with the heads {list(rec['heads'])} written to {out / 'estimator_heads_human.json'}; the self-play heads are as they were; wall {time.perf_counter() - t0:.1f} s")
    (out / "result_obs.txt").write_text("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
