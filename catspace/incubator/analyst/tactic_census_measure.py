"""The two measurements of the tactic census (tactic_census.py), Kaveh 2026-10-07, before any planner -- with the two
false gates found on the smoke data removed (2026-10-07 12:xx): rates per EPISODE, not per position (consecutive
positions are near-duplicates: double_threat k=1 was 365 positions but 79 episodes), and the latent tokens' contribution
ISOLATED from the immediate ones (`material_now` alone gave 38% vs 13% on the 20-ply gain -- one-ply search, not planning).

(a) DO BARRIERS PREDICT AVAILABILITY? For each family and each k now (the side to move's best token of the family), the
    fraction of EPISODES (the first position of a run of the same k for the same side) in which the family is ON
    (k = 0) for that side within 10 plies -- beside the rate when the family is absent now and the rate at the far
    state (the largest k seen); lift = rate / far-state rate, with a 95% bootstrap interval over games. The same for
    the opponent's latent tactics (read through a null move) becoming available to THEM.
    This is association, not cause: whether REMOVING a barrier makes the tactic come is the planner's claim, testable
    only in play.
(b) DO THE TOKENS PREDICT THE FUTURE BEYOND THE EVALUATION? The side to move's material swing over the next 20 plies
    (gain >= 2 pawns; loss >= 2), logistic regression, 5-fold cross-validation grouped by GAME, AUC, by feature set:
    base (Stockfish's evaluation at depth 10, material, phase); + immediate tokens (k = 0); + latent tokens (k >= 1,
    ours and theirs); + structure. The latent tokens' increment over base + immediate is the planner's number, with a
    95% paired bootstrap interval over games on the out-of-fold predictions.
Pass criteria, fixed BEFORE the numbers (2026-10-07; the reference restated after the review, before the c300
measurement was read): (a) in at least two families k = 1 fires at >= 2x the REFERENCE state's rate (k = 2 where the
family has it: back_rank, passer, discovery; "absent" for pin and double_threat, whose k never exceeds 1) with >= 50
episodes and the interval excluding 1x; (b) latent tokens add >= 0.02 AUC over base + immediate with the interval
excluding 0. Prints OBS lines; reads data/derived/tactic_census/census_<tag>.jsonl."""
from __future__ import annotations

import json
from collections import defaultdict

import numpy as np

FAMILIES = ("back_rank", "passer", "pin", "discovery", "double_threat", "material_now")
HORIZON_FIRE = 10
HORIZON_SWING = 20
BOOT = 1000


def load(tag: str) -> list:
    from .tactic_census import census_path
    with open(census_path(tag)) as fh:
        return [json.loads(line) for line in fh]


def _best_k(toks, family):
    ks = [t[1] for t in (toks or []) if t[0] == family]
    return min(ks) if ks else None


def episodes(rows: list, who: str = "stm") -> dict:
    """(a): {family: [(game, k_now, fired)]} over EPISODES. who='stm': the side to move's tokens; an episode starts at a
    position whose same-side predecessor (two plies back) had a different k for the family; fired = the family on (k = 0)
    for the same side within HORIZON_FIRE plies. who='opp': the opponent's null-move tokens; fired = on for THEM at a
    position where they are to move within the horizon. A window cut by the game's end counts as seen (fired if a firing
    was observed, else not)."""
    by_game = defaultdict(dict)
    for r in rows:
        by_game[r["game"]][r["ply"]] = r
    out = {f: [] for f in FAMILIES}
    for g, plies in by_game.items():
        last = max(plies)
        for t in sorted(plies):
            r = plies[t]
            toks = r["stm"] if who == "stm" else r["opp"]
            if toks is None:
                continue
            prev = plies.get(t - 2)
            prev_toks = None if prev is None else (prev["stm"] if who == "stm" else prev["opp"])
            later = range(t + 2, t + HORIZON_FIRE + 1, 2) if who == "stm" else range(t + 1, t + HORIZON_FIRE + 1, 2)
            for f in FAMILIES:
                k = _best_k(toks, f)
                if prev_toks is not None and _best_k(prev_toks, f) == k:
                    continue                                             # inside a run: not a new episode
                fired = any(u in plies and _best_k(plies[u]["stm"], f) == 0 for u in later)
                out[f].append((g, k, fired))                            # a window cut by the game's end counts as seen:
    return out                                                           # fired if a firing was observed, else not (biased
                                                                         # down; review 2026-10-07: skipping it dropped every
                                                                         # firing back-rank episode, a mate ends the game)


def reference_k(ks: list):
    """The far state a family's k = 1 is compared with: the state with fewer of the tactic's preconditions -- k = 2
    ("two or more fail") where the family has it, else "absent" (None: no pin, no battery, no candidate at all).
    Fixed with the criteria (review 2026-10-07: max(ks) compared pin's and double_threat's k = 1 with itself)."""
    return 2 if 2 in ks else None


def _rate(eps, k):
    sel = [e for e in eps if e[1] == k]
    return (sum(e[2] for e in sel) / len(sel) if sel else float("nan")), len(sel)


def _boot_lift(eps, k, far, rng) -> tuple:
    """A 95% interval over games for rate(k) / rate(far)."""
    games = sorted({e[0] for e in eps})
    by = defaultdict(list)
    for e in eps:
        by[e[0]].append(e)
    vals = []
    for _ in range(BOOT):
        pick = rng.choice(games, size=len(games), replace=True)
        sample = [e for g in pick for e in by[g]]
        a, na = _rate(sample, k)
        b, nb = _rate(sample, far)
        if na and nb and b > 0:
            vals.append(a / b)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if len(vals) >= BOOT // 2 else (float("nan"), float("nan"))


def _features(r: dict, sets: tuple) -> dict:
    x = {}
    if "base" in sets:
        ev = max(-1500, min(1500, r["eval"])) / 100.0
        x.update(eval=ev, material=r["material"], phase=r["phase"] / 10.0)
    for who in ("stm", "opp"):
        for t in (r[who] or []):
            k = min(t[1], 2)
            if (k == 0 and "immediate" in sets) or (k >= 1 and "latent" in sets):
                key = f"{who}:{t[0]}:{k}"
                x[key] = x.get(key, 0) + 1
    if "structure" in sets:
        x.update({f"s:{k}": v for k, v in r["structure"].items()})
    return x


def _oof_predictions(rows: list, sets: tuple, target: str, folds: int = 5):
    """Out-of-fold predictions of a logistic regression from the feature sets to the target, grouped by game.
    -> (pred, y, groups) in row order."""
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    by = {(r["game"], r["ply"]): r for r in rows}
    X, y, groups = [], [], []
    for r in rows:
        later = by.get((r["game"], r["ply"] + HORIZON_SWING))
        if later is None or r["eval"] is None:
            continue
        swing = later["material"] - r["material"]                    # the same side is to move HORIZON_SWING plies later
        X.append(_features(r, sets) or {"_": 0.0})
        y.append(swing >= 2 if target == "gain" else swing <= -2)
        groups.append(r["game"])
    y = np.array(y, dtype=int)
    groups = np.array(groups)
    pred = np.zeros(len(y))
    for fit, test in GroupKFold(n_splits=folds).split(X, y, groups):
        model = make_pipeline(DictVectorizer(sparse=False), StandardScaler(), LogisticRegression(max_iter=3000, C=1.0))
        model.fit([X[i] for i in fit], y[fit])
        pred[test] = model.predict_proba([X[i] for i in test])[:, 1]
    return pred, y, groups


def _auc_ci(pred_a, pred_b, y, groups, rng) -> tuple:
    """AUC of each prediction and a 95% paired bootstrap interval over games for AUC(b) - AUC(a)."""
    from sklearn.metrics import roc_auc_score
    ga = sorted(set(groups))
    idx = {g: np.where(groups == g)[0] for g in ga}
    diffs = []
    for _ in range(BOOT):
        pick = rng.choice(ga, size=len(ga), replace=True)
        sel = np.concatenate([idx[g] for g in pick])
        if len(set(y[sel])) < 2:
            continue
        diffs.append(roc_auc_score(y[sel], pred_b[sel]) - roc_auc_score(y[sel], pred_a[sel]))
    return roc_auc_score(y, pred_a), roc_auc_score(y, pred_b), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def measure(tag: str, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    rows = load(tag)
    games = len({r["game"] for r in rows})
    print(f"OBS census {tag}: {len(rows)} positions from {games} games")
    for who in ("stm", "opp"):
        eps = episodes(rows, who)
        label = "ours (side to move)" if who == "stm" else "theirs (null move)"
        for f in FAMILIES:
            e = eps[f]
            ks = sorted({x[1] for x in e if x[1] is not None})
            if not ks or ks == [0]:
                continue                                                 # material_now has no barrier ladder
            ref = reference_k(ks)
            parts = []
            for k in [None] + ks:
                r, n = _rate(e, k)
                name = "absent" if k is None else f"k={k}"
                tag = " (reference)" if k == ref else ""
                if k is not None and k != ref and _rate(e, ref)[0] == _rate(e, ref)[0] and _rate(e, ref)[0] > 0:
                    lo, hi = _boot_lift(e, k, ref, rng)
                    parts.append(f"{name} {100 * r:.1f}% (n {n}; lift {r / _rate(e, ref)[0]:.1f}x [{lo:.1f}, {hi:.1f}])")
                else:
                    parts.append(f"{name}{tag} {100 * r:.1f}% (n {n})")
            print(f"OBS (a) {f:13s} {label}: on within {HORIZON_FIRE} plies, per episode -- " + "; ".join(parts))
    for target in ("gain", "loss"):
        preds = {}
        for sets in (("base",), ("base", "immediate"), ("base", "immediate", "latent"), ("base", "immediate", "latent", "structure"),
                     ("immediate", "latent")):
            preds[sets] = _oof_predictions(rows, sets, target)
        y, groups = preds[("base",)][1], preds[("base",)][2]
        base = preds[("base",)][0]
        line = [f"base {_auc_ci(base, base, y, groups, rng)[0]:.3f}"]
        for sets, ref in ((("base", "immediate"), ("base",)), (("base", "immediate", "latent"), ("base", "immediate")),
                          (("base", "immediate", "latent", "structure"), ("base", "immediate", "latent"))):
            a, b, lo, hi = _auc_ci(preds[ref][0], preds[sets][0], y, groups, rng)
            line.append(f"+{sets[-1]} {b:.3f} (+{b - a:.3f} [{lo:+.3f}, {hi:+.3f}])")
        a, b, lo, hi = _auc_ci(base, preds[("immediate", "latent")][0], y, groups, rng)
        line.append(f"tokens alone {b:.3f}")
        print(f"OBS (b) material {target} >= 2 within {HORIZON_SWING} plies (n {len(y)}, {games} games, base rate {100 * y.mean():.1f}%; constant 0.500): "
              + "; ".join(line))


# --- tactics in the field by player strength (Kaveh 2026-10-07: "group the tactics by the probability of outs conditioned
# on player strength ... the expectation value of each tactic conditioned on Elo") ---

FIELD_FAMILIES = ("material_now", "double_threat", "pin", "discovery", "back_rank", "passer")
DEFAULT_BANDS = ((1000, 1400), (1400, 1700), (1700, 2000), (2000, 2500))


def elo_curves(tag: str, bands=DEFAULT_BANDS, window: int = 4) -> dict:
    """Per family and rating band of the ACTOR: P(executed) = the holder (a token at k = 0 for the side to move) gains 1+
    pawn within `window` plies, with the mean gain; P(out found) = the side to move, with the opponent's family available
    against it (the null-move tokens), loses no pawn within the window. -> {(family, band, kind): (n, rate, mean gain)};
    prints OBS lines."""
    rows = load(tag)
    by = defaultdict(dict)
    for r in rows:
        by[r["game"]][r["ply"]] = r

    def band_of(e):
        for lo, hi in bands:
            if lo <= e < hi or (e == hi and hi == bands[-1][1]):
                return f"{lo}-{hi}"
        return None

    acc = defaultdict(lambda: [0, 0, 0.0])
    for g, plies in by.items():
        for t, r in plies.items():
            later = plies.get(t + window)
            if later is None:
                continue
            swing = (later["material"] if later["side"] == r["side"] else -later["material"]) - r["material"]
            b = band_of(r["elo"])
            if b is None:
                continue
            for f in FIELD_FAMILIES:
                if any(x[0] == f and x[1] == 0 for x in r["stm"]):
                    c = acc[(f, b, "executed")]; c[0] += 1; c[1] += swing >= 1; c[2] += swing
                if any(x[0] == f and x[1] == 0 for x in (r["opp"] or [])):
                    c = acc[(f, b, "out found")]; c[0] += 1; c[1] += swing > -1; c[2] += swing
    out = {}
    names = [f"{lo}-{hi}" for lo, hi in bands]
    for kind, title in (("executed", f"P(executed: the holder gains 1+ within {window} plies | the holder's band), mean gain in brackets"),
                        ("out found", f"P(the out is found: no loss of 1+ within {window} plies | the victim's band), the victim's mean swing in brackets")):
        print(f"OBS {tag}: {title}")
        for f in FIELD_FAMILIES:
            cells = []
            for b in names:
                n, k, s = acc[(f, b, kind)]
                out[(f, b, kind)] = (n, k / n if n else float("nan"), s / n if n else float("nan"))
                cells.append(f"{b}: {100 * k / n:3.0f}% [{s / n:+.2f}] (n {n})" if n >= 20 else f"{b}: - (n {n})")
            print(f"OBS   {f:13s} " + "  ".join(cells))
    return out
