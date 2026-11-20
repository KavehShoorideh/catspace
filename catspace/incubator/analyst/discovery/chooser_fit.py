"""PLAN 79.12 PART D, THE FIT, IN SLEEP (SYSTEM_SPEC 9 sleep, 7 "Calculation and estimate, decided together" points 4 to 6: a learned last layer trained on results only; the inputs never add to the judge's value; piece: the wake's last layer).
Nothing of this runs in the wake: it reads the logs of part C (the decisions in which the move was DRAWN uniformly among two or more moves the search holds equal, mode Explore) and writes a table of weights, one a named input. No oracle number is read; the only target is the game's result from the mover's side.

ROWS. A decision with two or more equal moves gives one row: the inputs of the move played less the MEAN of the inputs of its equal set, so that only what distinguishes the move within its set is weighed (no intercept). Target: the mover's result (a win 1, a draw or an unfinished game 0.5, a loss 0), taken as a fractional label of a logistic fit.
FIT. A logistic fit with an L2 penalty (scikit-learn), the inputs standardised by the std of the rows, the penalty chosen by grouped cross-validation in the fit half. GAMES ARE SPLIT BY GAME into a fit half and a held-out half (a fixed seed). Weights are printed with a 90 pct band by bootstrap over games (5th to 95th percentile of the weight refit on resampled games), on the fit half
and on the held-out half; a weight is CLEAR on held-out where the band of the held-out refit excludes nought and its sign is the fit half's. BESIDE IT the same procedure on SHUFFLED results (the results of whole games permuted among games, kept with their seat), repeated, so that how many inputs look clear with no signal is printed next to how many do here.
STOP RULE (the plan's): no weight clear on held-out: print it, write no weights, and the matches of part E are not played.
THE WEIGHTS WRITTEN: the inputs clear on held-out, refit on ALL games (with the penalty found), returned to the scale of the raw inputs; every other input has weight nought.

INTERFACE (written before the bodies).
  load(tag, runs) -> Data                      rows, targets, game index, seat, the white score of each game, the names.
  fit_weights(X, y, w, c) -> array             the standardised weights of one penalised fit (rows weighted by w, the penalty c).
  analyse(data, ...) -> dict                   the whole of part D: the fit, the bands, the held-out test, the shuffled comparison; the lines to print.
  Called by: the command line and tests/test_plan79_12.py.

  .venv/bin/python -m catspace.incubator.analyst.discovery.chooser_fit TAG [--runs a,b] [--boot N] [--shuffles N]
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from catspace.incubator.analyst.discovery import whole_79_12 as W12
from catspace.incubator.analyst.discovery import whole_79_12_print as P
from catspace.incubator.analyst.discovery import whole_79_13 as W13

SEED = 7912
PENALTIES = (0.001, 0.01, 0.1, 1.0, 10.0)
BAND = (5.0, 95.0)                      # a 90 pct band


@dataclass
class Data:
    X: np.ndarray                       # rows: the inputs of the move played less the mean of its equal set
    seat: np.ndarray                    # the seat of the mover (0 moves first)
    game: np.ndarray                    # the index of the game of each row
    white: np.ndarray                   # the white score of each game (1, 0.5, 0)
    gids: list
    names: list

    @property
    def y(self) -> np.ndarray:
        w = self.white[self.game]
        return np.where(self.seat == 0, w, 1.0 - w)


def load(tag: str, runs: list, root: "Path | None" = None) -> Data:
    X: list = []
    seat: list = []
    game: list = []
    white: list = []
    gids: list = []
    names = None
    for run in runs:
        for decs, res in P.games_of((root or W12.out_root(tag)) / "logs" / run):
            if res is None:
                continue
            w = 0.5 if res.get("outcome") is None else res["outcome"] / 100.0
            gi = None
            for d in decs:
                c = P.varied(d)
                if c is None or c["mode"] != "Explore":
                    continue
                rows = np.array([x["row"] for x in c["candidates"]], dtype=float)
                i = [x["move"] for x in c["candidates"]].index(c["played"])
                names = c["inputs"]
                if gi is None:
                    gi = len(white)
                    white.append(w)
                    gids.append(f"{run}/{d['gid']}")
                X.append(rows[i] - rows.mean(axis=0))
                seat.append(d["seat"])
                game.append(gi)
    return Data(np.array(X), np.array(seat), np.array(game), np.array(white), gids, names or [])


def fit_weights(X: np.ndarray, y: np.ndarray, w: np.ndarray, c: float) -> np.ndarray:
    """One penalised fractional-logistic fit (no intercept) of the standardised inputs: each row is a win with weight w*y and a loss with weight w*(1-y)."""
    n = len(X)
    Z = np.vstack([X, X])
    lab = np.r_[np.ones(n), np.zeros(n)]
    sw = np.r_[w * y, w * (1.0 - y)]
    keep = sw > 0
    m = LogisticRegression(C=c, fit_intercept=False, solver="lbfgs", max_iter=500)
    m.fit(Z[keep], lab[keep], sample_weight=sw[keep])
    return m.coef_[0]


def logloss(X: np.ndarray, y: np.ndarray, beta: np.ndarray) -> float:
    z = X @ beta
    return float(np.mean(-(y * -np.logaddexp(0, -z) + (1 - y) * -np.logaddexp(0, z))))


def choose_penalty(X: np.ndarray, y: np.ndarray, game: np.ndarray, rng: np.random.Generator, folds: int = 5) -> float:
    """The penalty with the least held-out log-loss in a grouped cross-validation over the games of the fit half."""
    ids = np.unique(game)
    rng.shuffle(ids)
    part = {g: k % folds for k, g in enumerate(ids)}
    fold = np.array([part[g] for g in game])
    best, arg = np.inf, PENALTIES[0]
    for c in PENALTIES:
        tot = 0.0
        for k in range(folds):
            tr, te = fold != k, fold == k
            tot += logloss(X[te], y[te], fit_weights(X[tr], y[tr], np.ones(tr.sum()), c)) * te.sum()
        if tot / len(X) < best:
            best, arg = tot / len(X), c
    return arg


def boot_band(X: np.ndarray, y: np.ndarray, game: np.ndarray, c: float, draws: int, rng: np.random.Generator) -> np.ndarray:
    """The band (rows: low, high) of each weight over bootstrap resamples of the GAMES."""
    ids = np.unique(game)
    index = {g: k for k, g in enumerate(ids)}
    gi = np.array([index[g] for g in game])
    out = []
    for _ in range(draws):
        mult = np.bincount(rng.integers(0, len(ids), len(ids)), minlength=len(ids))[gi].astype(float)
        out.append(fit_weights(X, y, mult, c))
    return np.percentile(np.array(out), BAND, axis=0)


def clear(band: np.ndarray, beta_fit: np.ndarray) -> np.ndarray:
    """A weight is clear where its band excludes nought and its sign is the sign of the fit half's weight."""
    return ((band[0] > 0) | (band[1] < 0)) & (np.sign((band[0] + band[1]) / 2) == np.sign(beta_fit))


def split_games(n_games: int, rng: np.random.Generator) -> np.ndarray:
    side = np.zeros(n_games, dtype=bool)
    side[rng.permutation(n_games)[: n_games // 2]] = True               # True: the fit half
    return side


def analyse(d: Data, boot: int = 200, shuffles: int = 10) -> dict:
    rng = np.random.default_rng(SEED)
    sd = d.X.std(axis=0)
    live = sd > 0
    # inputs identical on every row are one input: the later ones are left out (noted)
    dup: dict = {}
    cols = [k for k in range(len(sd)) if live[k]]
    for a in cols:
        for b in cols:
            if b > a and live[b] and np.allclose(d.X[:, a], d.X[:, b]):
                live[b] = False
                dup[d.names[b]] = d.names[a]
    use = np.where(live)[0]
    Xs = d.X[:, use] / sd[use]
    names = [d.names[k] for k in use]
    side = split_games(len(d.white), rng)
    fit_rows, held_rows = side[d.game], ~side[d.game]
    y = d.y
    res: dict = {"names": names, "dropped_nought": [d.names[k] for k in range(len(sd)) if sd[k] == 0], "dropped_duplicate": dup, "games": len(d.white), "rows": len(y),
                 "fit_games": int(side.sum()), "held_games": int((~side).sum()), "fit_rows": int(fit_rows.sum()), "held_rows": int(held_rows.sum())}
    c = choose_penalty(Xs[fit_rows], y[fit_rows], d.game[fit_rows], rng)
    res["penalty"] = c
    beta = fit_weights(Xs[fit_rows], y[fit_rows], np.ones(fit_rows.sum()), c)
    res["beta_fit"] = beta
    res["band_fit"] = boot_band(Xs[fit_rows], y[fit_rows], d.game[fit_rows], c, boot, rng)
    res["beta_held"] = fit_weights(Xs[held_rows], y[held_rows], np.ones(held_rows.sum()), c)
    res["band_held"] = boot_band(Xs[held_rows], y[held_rows], d.game[held_rows], c, boot, rng)
    res["clear"] = clear(res["band_held"], beta)
    res["held_logloss_fit_weights"] = logloss(Xs[held_rows], y[held_rows], beta)
    res["held_logloss_nought"] = float(np.log(2.0))
    # the same procedure on shuffled results: whole games' results permuted among games, kept with their seat
    shuffled = []
    for _ in range(shuffles):
        wp = d.white[rng.permutation(len(d.white))]
        ys = np.where(d.seat == 0, wp[d.game], 1.0 - wp[d.game])
        b_s = fit_weights(Xs[fit_rows], ys[fit_rows], np.ones(fit_rows.sum()), c)
        band_s = boot_band(Xs[held_rows], ys[held_rows], d.game[held_rows], c, max(40, boot // 4), rng)
        shuffled.append({"beta": b_s, "clear": clear(band_s, b_s), "held_logloss": logloss(Xs[held_rows], ys[held_rows], b_s), "max_abs": float(np.abs(b_s).max())})
    res["shuffled"] = shuffled
    # the weights written: the clear inputs, refit on all games, in the scale of the raw inputs
    if res["clear"].any():
        sel = np.where(res["clear"])[0]
        allw = fit_weights(Xs[:, sel], y, np.ones(len(y)), c)
        res["weights"] = {names[k]: float(allw[j] / sd[use[k]]) for j, k in enumerate(sel)}
        res["weights_std"] = {names[k]: float(allw[j]) for j, k in enumerate(sel)}
    else:
        res["weights"] = {}
    return res


def beats_shuffled(r: dict) -> tuple:
    """PLAN 79.13 D (1), the rule in sense: a fitted chooser is played only if the fit beats the same fit on shuffled results on BOTH counts: the held-out improvement over nought weights (ln 2 less the held-out log-loss) is larger than every shuffled fit's, and the count of inputs clear on held-out is larger than every shuffle's.
    Returns (played, the reason in words)."""
    sh = r["shuffled"]
    real_gain = r["held_logloss_nought"] - r["held_logloss_fit_weights"]
    best_sh = max(r["held_logloss_nought"] - x["held_logloss"] for x in sh)
    real_clear = int(r["clear"].sum())
    most_clear = max(int(x["clear"].sum()) for x in sh)
    ok = real_gain > best_sh and real_clear > most_clear
    return ok, f"held-out gain over nought weights real {real_gain:+.5f} against the best shuffled {best_sh:+.5f}; clear inputs real {real_clear} against the most of any shuffle {most_clear}"


def lines_of(tag: str, runs: list, d: Data, r: dict, plan: str = "79_12") -> list:
    t = f"OBS whole_{plan} fit {'+'.join(runs)}"
    out = [f"{t}: games {r['games']}, rows (varied decisions) {r['rows']}; split by game: fit half {r['fit_games']} games {r['fit_rows']} rows, held-out half {r['held_games']} games {r['held_rows']} rows; penalty C {r['penalty']} (grouped cross-validation in the fit half)",
           f"{t}: inputs with no spread on the sample (left out): {r['dropped_nought']}; identical to an earlier input (left out): {r['dropped_duplicate']}",
           f"{t}: mean result of the mover over the rows {d.y.mean():.4f}; share of rows from games that ended on a result {np.mean(np.isin(d.white[d.game], (0.0, 1.0))):.4f}"]
    for k, nm in enumerate(r["names"]):
        out.append(f"{t} weight {nm}: fit half {r['beta_fit'][k]:+.4f} [{r['band_fit'][0][k]:+.4f}, {r['band_fit'][1][k]:+.4f}]; held-out refit {r['beta_held'][k]:+.4f} [{r['band_held'][0][k]:+.4f}, {r['band_held'][1][k]:+.4f}]; clear on held-out: {'YES' if r['clear'][k] else 'no'}   (per standard deviation of the input; 90 pct band by bootstrap over games)")
    n_clear = int(r["clear"].sum())
    sh = [int(x["clear"].sum()) for x in r["shuffled"]]
    out.append(f"{t}: inputs clear on held-out: {n_clear} of {len(r['names'])}")
    out.append(f"{t}: SHUFFLED results ({len(sh)} shuffles of whole games' results, the same procedure): inputs clear on held-out per shuffle {sh}, mean {np.mean(sh):.2f}; largest |weight| of the fit half real {np.abs(r['beta_fit']).max():.4f}, shuffled mean {np.mean([x['max_abs'] for x in r['shuffled']]):.4f} (max {np.max([x['max_abs'] for x in r['shuffled']]):.4f})")
    out.append(f"{t}: held-out mean log-loss of the fit half's weights {r['held_logloss_fit_weights']:.5f} against nought weights {r['held_logloss_nought']:.5f} (smaller is better); shuffled fits {np.mean([x['held_logloss'] for x in r['shuffled']]):.5f}")
    ok, why = beats_shuffled(r)
    out.append(f"{t}: PLAN 79.13 RULE (played only if the fit beats the same fit on shuffled results, held-out log-loss and count of clear inputs): {'PLAYED' if ok else 'NOT PLAYED'}: {why}")
    if n_clear == 0:
        out.append(f"{t}: STOP: no weight's band is clear of nought on held-out games: the inputs do not tell equal moves apart by result at this sample; no weights are written and part E is not played")
    else:
        out.append(f"{t}: weights {'written' if ok else 'NOT written (the fit does not beat the shuffled fits)'} (clear inputs, refit on all games, scale of the raw inputs): " + ", ".join(f"{k} {v:+.6g}" for k, v in r["weights"].items()))
    return out


def main(argv: "list | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("tag")
    ap.add_argument("--runs", default="train_self,train_old")
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--shuffles", type=int, default=10)
    ap.add_argument("--plan", default="79_12", choices=("79_12", "79_13"), help="whose folder holds the logs and receives the fit (PLAN 79.13 reads its own games, which carry the gain inputs)")
    a = ap.parse_args(argv)
    runs = a.runs.split(",")
    root = W12.out_root(a.tag) if a.plan == "79_12" else W13.out_root(a.tag)
    d = load(a.tag, runs, root)
    r = analyse(d, a.boot, a.shuffles)
    lines = lines_of(a.tag, runs, d, r, a.plan)
    (root / "fit").mkdir(parents=True, exist_ok=True)
    (root / "fit" / "fit_obs.txt").write_text("\n".join(lines) + "\n")
    if r["weights"] and beats_shuffled(r)[0]:
        (root / "fit" / "weights.json").write_text(json.dumps({"weights": r["weights"], "weights_standardised": r["weights_std"], "penalty": r["penalty"], "runs": runs}, indent=1))
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
