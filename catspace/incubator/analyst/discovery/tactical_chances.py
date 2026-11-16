"""PLAN 78.62 C: THE CHANCES, OURS AND THEIRS APART. Two heads over the parts of a SETTLED position (a quiet row of 78.59: no unit of either side borne on by the other and not by its own): the chance that within
HORIZON plies a unit of THEIRS stands lost by the estimate with US to move (the plies HORIZON/2 of them, the ones where we move), and the chance that within HORIZON plies a unit of OURS stands lost by the estimate
with THEM to move (the plies where they move). A standing loss is that of tactical_tables.py (stage A's frame read, the order of kinds of the human record of 78.59). Rows: the quiet rows of the 78.59 file that
fall in the games of stage A, with HORIZON rows of the game after them. THE INPUTS (the parts): K the kinds (ours less theirs), S the 26 statements (ours less theirs), E the edge distances of the units (for each kind,
the sum of the distances of its units from the nearest edge, ours less theirs; the distance is read from the place names), R the ratings (mean, difference of the mover's and the other's, ply), and on the thinned rows
(every CONCEPT_EVERY-th quiet row of a game, 78.59) C the 64 kept concepts of 78.54 (ours less theirs). Sets: R, K, K+S, K+S+E, K+S+E+R; on the thinned rows K+S+E+R against K+S+E+R+C, every set refitted on the same
thinned rows. FIT: L1-penalised logistic regression on the standardised columns, the penalty by 5-fold cross-validation grouped by game on the fit games only (the games' own 78.59 split, 7 in 10 fit); READ on the test
games: log loss against the base rate of the fit games, calibration (ten equal-count bins of the predicted chance), rank correlation with the target, each with a 90 pct band over games (draws); by band of rating.
BESIDE IT the arrival head of 78.59 (a sparse linear fit of the arrival difference h=8 on K+S, the same rows), and how the two targets agree. Direction only; no engine evaluation enters.

  .venv/bin/python -m catspace.incubator.analyst.discovery.tactical_chances --state-dir D --out-dir O
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np

from catspace.incubator.analyst.discovery import forcing_arrival_measure as FA
from catspace.incubator.analyst.discovery import human_heads_data as HD
from catspace.incubator.analyst.discovery import opportunity_measure as OM
from catspace.incubator.analyst.discovery import subgoal_fit_measure as SF
from catspace.incubator.analyst.discovery import tactical_state as TS
from catspace.incubator.analyst.discovery import tactical_tables as TT
from catspace.io import paths

warnings.filterwarnings("ignore", module="sklearn")
HORIZON = 8
DRAWS = 300
SEED = 6262
FOLDS = 5
C_GRID = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0)
CALIB_BINS = 10


def target_rows(d: dict, heads_dir: Path) -> dict:
    """The rows of the 78.59 file in the games of stage A, with their state row, the two targets and the inputs."""
    st, tb, games = d["st"], d["tb"], d["games"]
    rows = HD.load_rows(heads_dir / "rows.npz")
    allg = [json.loads(x) for x in (heads_dir / "games.jsonl").read_text().splitlines() if x.strip()]
    index = {g["id"]: i for i, g in enumerate(allg)}
    ids, first, last = TT.game_slices(st)
    slot = {int(g): k for k, g in enumerate(ids)}
    src_to_mine = {index[games[int(g)]["id"]]: int(g) for g in ids}
    lo = int(st["ply"].min())
    sel, state_row, mine = [], [], []
    for r in range(len(rows["game"])):
        g = int(rows["game"][r])
        if g not in src_to_mine:
            continue
        m = src_to_mine[g]
        k = slot[m]
        i = int(first[k]) + int(rows["ply"][r]) - lo
        if i < first[k] or i + HORIZON >= last[k] or int(st["ply"][i]) != int(rows["ply"][r]):
            continue
        sel.append(r); state_row.append(i); mine.append(m)
    sel, state_row, mine = np.array(sel), np.array(state_row), np.array(mine)
    x = tb["lost_est"] > 0
    mv = st["mover"].astype(int)
    c0 = mv[state_row]
    y_us = np.zeros(len(sel), np.int8)
    y_them = np.zeros(len(sel), np.int8)
    for j, (i, c) in enumerate(zip(state_row, c0)):
        y_us[j] = any(x[i + u, 1 - c] for u in range(2, HORIZON + 1, 2))
        y_them[j] = any(x[i + u, c] for u in range(1, HORIZON + 1, 2))
    edge = tb["edge_k"][state_row, c0, :] - tb["edge_k"][state_row, 1 - c0, :]
    mean_r = (rows["rm"][sel] + rows["ro"][sel]) / 2.0
    out = {"game": mine.astype(int), "ply": rows["ply"][sel].astype(int), "state_row": state_row, "y_us": y_us, "y_them": y_them,
           "K": rows["kd"][sel].astype(float), "S": rows["sd"][sel].astype(float), "E": edge.astype(float),
           "R": np.stack([mean_r, (rows["rm"][sel] - rows["ro"][sel]).astype(float), rows["ply"][sel].astype(float)], axis=1),
           "fit": rows["fit"][sel].astype(bool), "band": rows["band"][sel].astype(int), "y": rows["y"][sel].astype(int),
           "cmask": rows["cmask"][sel].astype(bool), "src_row": sel}
    cx_full = np.zeros((len(rows["game"]), rows["cx"].shape[1]), np.float32)
    cx_full[np.flatnonzero(rows["cmask"])] = rows["cx"]
    out["C"] = cx_full[sel].astype(float)
    out["meta"] = {"kinds": rows["meta"]["kinds"], "statements": rows["meta"]["statements"], "concepts": rows["meta"]["concepts"], "edge_kinds": st["meta"]["kinds"]}
    out["rows_total"] = len(rows["game"])
    return out


def arrival_columns(d: dict, t: dict, heads_dir: Path) -> np.ndarray:
    """The arrival difference h=8 of 78.59 (the mover's confirmed threat move arrives less the other's) on each row, NaN where the game ends first or no read is kept."""
    out = np.full(len(t["game"]), np.nan)
    reads = {}
    p = heads_dir.parent / "full_arrival" / "reads_light.jsonl"
    if not p.exists():
        return out
    for line in p.read_text().splitlines():
        if line.strip():
            x = json.loads(line)
            reads[x["game"]] = x["plies"]
    counted = {}
    for j, (g, ply) in enumerate(zip(t["game"], t["ply"])):
        if g not in reads:
            continue
        if g not in counted:
            counted[g] = FA.counts_of_game(reads[g])[0]
        tg = FA.arrival_targets(counted[g], len(reads[g]), int(ply))[1]
        if tg is not None:
            out[j] = tg[0] - tg[1]
    return out


def std(x: np.ndarray, ref: np.ndarray) -> np.ndarray:
    m, s = ref.mean(axis=0), ref.std(axis=0)
    return (x - m) / np.where(s > 0, s, 1.0)


def fit_logit(x: np.ndarray, y: np.ndarray, groups: np.ndarray) -> dict:
    """L1 logistic fit of y on the standardised columns, C by grouped 5-fold cross-validation (log loss)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    cv = list(GroupKFold(n_splits=FOLDS).split(x, y, groups))
    mean, scale = x.mean(axis=0), np.where(x.std(axis=0) > 0, x.std(axis=0), 1.0)
    z = (x - mean) / scale
    best, best_c = None, None
    for c in C_GRID:
        ls = []
        for tr, te in cv:
            m = LogisticRegression(penalty="l1", solver="liblinear", C=c, max_iter=500).fit(z[tr], y[tr])
            ls.append(logloss(m.predict_proba(z[te])[:, 1], y[te]))
        if best is None or np.mean(ls) < best:
            best, best_c = float(np.mean(ls)), c
    m = LogisticRegression(penalty="l1", solver="liblinear", C=best_c, max_iter=500).fit(z, y)
    return {"C": best_c, "mean": mean, "scale": scale, "coef": m.coef_[0].copy(), "intercept": float(m.intercept_[0]), "cv": best}


def predict_logit(m: dict, x: np.ndarray) -> np.ndarray:
    z = ((x - m["mean"]) / m["scale"]) @ m["coef"] + m["intercept"]
    return 1.0 / (1.0 + np.exp(-z))


def logloss(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def row_loss(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def calib(p: np.ndarray, y: np.ndarray) -> str:
    order = np.argsort(p, kind="stable")
    return " ".join(f"{p[b].mean():.3f}>{y[b].mean():.3f}" for b in np.array_split(order, CALIB_BINS) if len(b))


def mean_band(games: np.ndarray, vals: np.ndarray, draws: int, seed: int) -> "tuple[float, float, float]":
    """The mean of per-row values with a 90 pct band over resamples of the games."""
    ids, inv = np.unique(games, return_inverse=True)
    s = np.zeros(len(ids)); n = np.zeros(len(ids))
    np.add.at(s, inv, vals); np.add.at(n, inv, 1.0)
    rng = np.random.default_rng(seed)
    v = []
    for _ in range(draws):
        k = rng.integers(0, len(ids), len(ids))
        v.append(s[k].sum() / n[k].sum())
    return float(s.sum() / n.sum()), float(np.percentile(v, 5)), float(np.percentile(v, 95))


def paired_gain(games: np.ndarray, base_loss: np.ndarray, loss: np.ndarray, draws: int, seed: int) -> "tuple[float, float, float]":
    return mean_band(games, base_loss - loss, draws, seed)


def sets_of(t: dict) -> dict:
    return {"R": t["R"], "K": t["K"], "K+S": np.hstack([t["K"], t["S"]]), "K+S+E": np.hstack([t["K"], t["S"], t["E"]]), "K+S+E+R": np.hstack([t["K"], t["S"], t["E"], t["R"]])}


def names_of(t: dict, which: str) -> list:
    m = t["meta"]
    parts = {"R": ["mean rating", "rating difference", "ply"], "K": [f"kind {k}" for k in m["kinds"]], "S": [f"statement {i}" for i in range(t["S"].shape[1])],
             "E": [f"edge distance of {k}" for k in m["edge_kinds"]], "C": [f"concept {i}" for i in range(t["C"].shape[1])]}
    out = []
    for p in which.split("+"):
        out += parts[p]
    return out


def main(argv: "list | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--state-dir", default=paths.experiment("tactical_78_62", "state"))
    ap.add_argument("--heads-dir", default=paths.experiment(*TS.HEADS_DIR))
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--draws", type=int, default=DRAWS)
    a = ap.parse_args(argv)
    out = Path(a.out_dir or paths.experiment("tactical_78_62", "chances"))
    out.mkdir(parents=True, exist_ok=True)
    lines: list = []

    def say(msg: str) -> None:
        lines.append(f"OBS tactical chances: {msg}")
        print(lines[-1], flush=True)

    d = TT.load_all(Path(a.state_dir))
    t = target_rows(d, Path(a.heads_dir))
    n = len(t["game"])
    fit, test = np.flatnonzero(t["fit"]), np.flatnonzero(~t["fit"])
    say(f"rows: {n} quiet rows of the 78.59 file in the {len(np.unique(t['game']))} games of stage A (of {t['rows_total']} rows in all) with {HORIZON} rows after them; fit {len(fit)} ({len(np.unique(t['game'][fit]))} games), test {len(test)} ({len(np.unique(t['game'][test]))} games); thinned rows with concepts {int(t['cmask'].sum())}")
    for lab, key in (("US (a unit of theirs stands lost by the estimate with us to move within 8 plies)", "y_us"), ("THEM (a unit of ours stands lost by the estimate with them to move within 8 plies)", "y_them")):
        say(f"target {lab}: fit rate {t[key][fit].mean():.4f}, test rate {t[key][test].mean():.4f}")
    sets = sets_of(t)
    arr = arrival_columns(d, t, Path(a.heads_dir))
    have = ~np.isnan(arr)
    say(f"arrival difference h=8 of 78.59 on these rows: present on {int(have.sum())} rows; mean over the test rows {np.nanmean(arr[test]):+.4f}")
    result: dict = {}
    saved: dict = {}
    for tag, key in (("us", "y_us"), ("them", "y_them")):
        y = t[key].astype(float)
        base = y[fit].mean()
        base_loss = row_loss(np.full(len(test), base), y[test])
        say(f"[{tag}] test log loss, base rate: {mean_band(t['game'][test], base_loss, a.draws, SEED)[0]:.5f}")
        preds = {}
        for nm, x in sets.items():
            m = fit_logit(x[fit], y[fit], t["game"][fit])
            p = predict_logit(m, x[test])
            preds[nm] = p
            ll = row_loss(p, y[test])
            v, lo, hi = mean_band(t["game"][test], ll, a.draws, SEED)
            g, glo, ghi = paired_gain(t["game"][test], base_loss, ll, a.draws, SEED)
            kept = int((m["coef"] != 0).sum())
            say(f"[{tag}] test log loss, {nm}: {v:.5f} [{lo:.5f}, {hi:.5f}]; gain over the base rate {g:+.5f} [{glo:+.5f}, {ghi:+.5f}]; penalty C {m['C']}, columns kept {kept} of {len(m['coef'])}")
            result[(tag, nm)] = m
            say(f"[{tag}] calibration {nm} (predicted>observed, ten equal-count bins): {calib(p, y[test])}")
        for what, base_name, better in (("K+S over K", "K", "K+S"), ("K+S+E over K+S", "K+S", "K+S+E"), ("K+S+E+R over K+S+E", "K+S+E", "K+S+E+R"), ("K over R", "R", "K")):
            g, glo, ghi = paired_gain(t["game"][test], row_loss(preds[base_name], y[test]), row_loss(preds[better], y[test]), a.draws, SEED)
            say(f"[{tag}] test log loss gain of {what}: {g:+.5f} [{glo:+.5f}, {ghi:+.5f}] (positive: the second is better)")
        # rank correlation with a band over games, all and by band of rating
        at = {"game": t["game"][test], "band": t["band"][test], "target": y[test], "arrival": arr[test]}
        reads = []
        for nm in ("R", "K", "K+S", "K+S+E", "K+S+E+R"):
            at[f"pred_{nm}"] = preds[nm]
            reads.append((f"[{tag}] head {nm} against its target", f"pred_{nm}", "target"))
        ls, _v = SF.band_lines("tactical chances", at, a.draws, reads)
        for ln in ls:
            say(ln.replace("OBS subgoal tactical chances: ", ""))
        for b in range(4):
            sel = at["band"] == b
            sub = {k: v[sel] for k, v in at.items()}
            if len(np.unique(sub["game"])) < 10:
                continue
            ls, _v = SF.band_lines("tactical chances", sub, max(100, a.draws // 2), [(f"[{tag}] band {HD.BAND_NAMES[b]}, head K+S+E+R", "pred_K+S+E+R", "target"), (f"[{tag}] band {HD.BAND_NAMES[b]}, head K", "pred_K", "target")])
            for ln in ls:
                say(ln.replace("OBS subgoal tactical chances: ", ""))
        # how the target agrees with the arrival target of 78.59, on the test rows that have both
        ok = ~np.isnan(arr[test])
        say(f"[{tag}] rank correlation of the target with the arrival difference h=8 of 78.59 over the {int(ok.sum())} test rows that have both: {OM.spearman(y[test][ok], arr[test][ok]):+.4f}; of the head K+S+E+R: {OM.spearman(preds['K+S+E+R'][ok], arr[test][ok]):+.4f}")
        saved[tag] = {"test": test, "pred": preds["K+S+E+R"]}
        # the thinned rows with the concepts
        thin = np.flatnonzero(t["cmask"])
        tf, tt = np.intersect1d(thin, fit), np.intersect1d(thin, test)
        if len(tf) > 200 and len(tt) > 100:
            yb = y[tt]
            bl = row_loss(np.full(len(tt), y[tf].mean()), yb)
            full_x = sets["K+S+E+R"]
            mc = {}
            for nm, x in (("K+S+E+R", full_x), ("K+S+E+R+C", np.hstack([full_x, t["C"]]))):
                m = fit_logit(x[tf], y[tf], t["game"][tf])
                p = predict_logit(m, x[tt])
                mc[nm] = (m, p)
                v, lo, hi = mean_band(t["game"][tt], row_loss(p, yb), a.draws, SEED)
                say(f"[{tag}] [thinned rows {len(tt)} of {len(np.unique(t['game'][tt]))} games] test log loss, {nm}: {v:.5f} [{lo:.5f}, {hi:.5f}]; columns kept {int((m['coef'] != 0).sum())} of {len(m['coef'])}")
            g, glo, ghi = paired_gain(t["game"][tt], row_loss(mc["K+S+E+R"][1], yb), row_loss(mc["K+S+E+R+C"][1], yb), a.draws, SEED)
            say(f"[{tag}] [thinned rows] test log loss gain of adding the concepts: {g:+.5f} [{glo:+.5f}, {ghi:+.5f}] (positive: with them is better)")
            result[(tag, "K+S+E+R+C thinned")] = mc["K+S+E+R+C"][0]
    # the arrival head of 78.59 beside: the sparse linear fit of the arrival difference on K+S, the same rows
    for nm in ("K+S", "K+S+E"):
        ok_fit, ok_test = fit[have[fit]], test[have[test]]
        if len(ok_fit) > 100 and len(ok_test) > 100:
            m = SF.fit_sparse(sets[nm][ok_fit], arr[ok_fit], t["game"][ok_fit])
            p = SF.predict(m, sets[nm][ok_test])
            ad = {"game": t["game"][ok_test], "pred": p, "arr": arr[ok_test]}
            ls, _v = SF.band_lines("tactical chances", ad, a.draws, [(f"the arrival head h=8 of 78.59 refit on these rows, inputs {nm}", "pred", "arr")])
            for ln in ls:
                say(ln.replace("OBS subgoal tactical chances: ", ""))
    # the inputs that carry weight, in words (standardised columns)
    for (tag, nm), m in sorted(result.items()):
        if nm not in ("K+S+E+R",):
            continue
        names = names_of(t, nm)
        order = np.argsort(-np.abs(m["coef"]))
        top = [f"{names[i]} {m['coef'][i]:+.3f}" for i in order[:12] if m["coef"][i] != 0]
        say(f"[{tag}] the inputs of K+S+E+R with the largest weight (per standard deviation, log-odds): " + "; ".join(top))
    np.savez_compressed(out / "chances.npz", **{f"{k}_{tag}": v for tag, s in saved.items() for k, v in s.items()}, game=t["game"], ply=t["ply"], state_row=t["state_row"], fit=t["fit"], y_us=t["y_us"], y_them=t["y_them"], band=t["band"], y=t["y"])
    models = {f"{tag}|{nm}": {"C": m["C"], "mean": m["mean"].tolist(), "scale": m["scale"].tolist(), "coef": m["coef"].tolist(), "intercept": m["intercept"]} for (tag, nm), m in result.items()}
    (out / "models.json").write_text(json.dumps({"names": {nm: names_of(t, nm.split(" thinned")[0]) for nm in ("R", "K", "K+S", "K+S+E", "K+S+E+R", "K+S+E+R+C")}, "models": models}))
    (out / "chances_obs.txt").write_text("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
