"""PLAN 79.13 PART B, THE TABLE OF THE CHANCE OF A GOAL, IN SLEEP (SYSTEM_SPEC 7 "Calculation and estimate, decided together" point 2: what is not near is estimated per goal; 6 the event heads; 9 sleep; piece: the estimate per goal).
Counted from play, no oracle: the rows of goal_reach.py (every goal logged at a decision of the wake in the training games of 79.12, with whether it was reached within K plies). Nothing of it runs in the wake; the wake reads the finished table (cs_decision::goal_table).
Game-free: the keys are made of what the log holds for a goal and name no object of the game.

THE KEYS. A cell is a goal's values on a list of FIELDS: source (the class word of the goal: unread, refuted, two_moves and the like), unmet (the facts unmet, a set), in_way (the things in the way), depth (in its chain), rank (the rank of the target's stake among the kinds: goal_reach.rank_of),
fix (a fix that has a move exists), mine (the goal is the side to move's). The table BACKS OFF from the full key to fewer fields (LEVELS, fixed here and not tuned): a goal is estimated by the first level whose cell holds at least FLOOR goals (78.74's floor, 20), else by the rate of all goals; every estimate is smoothed (reached + 1) / (n + 2) as in 78.74.
THE TEST. GAMES ARE SPLIT BY GAME (goal_reach.half_of): the table is counted on the fit half and read on the held-out half. Printed beside the old table (78.74's, as logged at the decision: another event, "reached in four plies by own moves", read here as a predictor of this one) and beside the base rate of the fit half: the mean log-loss, a band
for the difference by bootstrap over held-out GAMES, and the share reached by tenth of predicted chance (is a goal said to be 0.3 reached about three in ten). The OLD table is also shown RECALIBRATED (a logistic fit of the event on the logit of its number, on the fit half) so that its ordering is judged apart from its scale.
STOP RULE (the plan's, meant in sense): if the new table is no better than the old on held-out goals, or no better than the base rate, say so and stop: the facts of a goal do not predict it being reached, no table is written for the wake.

INTERFACE (written before the bodies).
  LEVELS                                       the fields of each level, from the full key to the fewest.
  key_strings(d, fields, rows) -> [str]        the key of each row as the wake builds it (values joined by "|": source, unmet joined by "+", the integers, fix and mine as 0 or 1).
  fit_table(d, j, mask) -> dict                the table counted on the rows of `mask` at the K of column j: {"k", "floor", "levels": [{"fields", "cells": [[key, n, reached]]}] (cells under the floor left out), "total": [n, reached]}.
  predict(table, d, rows, j) -> (chance, level)   the estimate of each row and the index of the level that spoke (len(levels): the whole).
  evaluate(...) -> lines                       the test above.
  Called by: the command line, tests/test_plan79_13.py; the table is read by whole_79_13.py and given to the wake through the binding.

  .venv/bin/python -m catspace.incubator.analyst.discovery.goal_table TAG [--also 8,16]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from catspace.incubator.analyst.discovery import goal_reach as GR

FLOOR = 20
LEVELS = (("source", "unmet", "in_way", "depth", "rank", "fix", "mine"),
          ("source", "unmet", "in_way", "rank", "fix", "mine"),
          ("source", "unmet", "in_way", "rank", "mine"),
          ("unmet", "in_way", "rank", "mine"),
          ("unmet", "rank", "mine"),
          ("rank", "mine"),
          ("mine",))
INT_FIELDS = ("in_way", "depth", "rank", "fix", "mine")


def load(tag: str) -> dict:
    z = np.load(GR.out_root(tag) / "goal_rows.npz", allow_pickle=False)
    return {k: z[k] for k in z.files}


def field_values(d: dict, field: str, rows: np.ndarray) -> list:
    if field in INT_FIELDS:
        return [str(int(x)) for x in d[field][rows]]
    vocab = d[field + "_vocab"]
    return [str(vocab[c]) for c in d[field][rows]]


def key_strings(d: dict, fields: tuple, rows: np.ndarray) -> list:
    cols = [field_values(d, f, rows) for f in fields]
    return ["|".join(v) for v in zip(*cols)]


def _codes(d: dict, fields: tuple) -> np.ndarray:
    """One integer a row for the values of `fields` (a mixed radix over the codes), for counting by key."""
    key = np.zeros(len(d["game"]), np.int64)
    for f in fields:
        c = d[f].astype(np.int64)
        key = key * (int(c.max()) + 2) + (c - int(c.min()))
    return key


def fit_table(d: dict, j: int, mask: np.ndarray) -> dict:
    ok = mask & (d["reached"][:, j] >= 0)
    hit = (d["reached"][:, j] == 1)
    levels = []
    for fields in LEVELS:
        code = _codes(d, fields)
        uniq, inv = np.unique(code[ok], return_inverse=True)
        n = np.bincount(inv)
        h = np.bincount(inv, weights=hit[ok].astype(float)).astype(int)
        rows = np.where(ok)[0]
        first = np.zeros(len(uniq), np.int64)
        first[inv[::-1]] = rows[::-1]                                              # a row of each cell, to name its key
        keep = np.where(n >= FLOOR)[0]
        names = key_strings(d, fields, first[keep]) if len(keep) else []
        levels.append({"fields": list(fields), "cells": [[names[i], int(n[k]), int(h[k])] for i, k in enumerate(keep)]})
    return {"k": GR.KS[j], "floor": FLOOR, "levels": levels, "total": [int(ok.sum()), int(hit[ok].sum())]}


def predict(table: dict, d: dict, rows: np.ndarray, j: int) -> tuple:
    chance = np.full(len(rows), (table["total"][1] + 1.0) / (table["total"][0] + 2.0))
    level = np.full(len(rows), len(table["levels"]))
    done = np.zeros(len(rows), bool)
    for li, lv in enumerate(table["levels"]):
        cells = {c[0]: (c[1], c[2]) for c in lv["cells"]}
        keys = key_strings(d, tuple(lv["fields"]), rows)
        for i, k in enumerate(keys):
            if not done[i] and k in cells:
                n, h = cells[k]
                chance[i] = (h + 1.0) / (n + 2.0)
                level[i] = li
                done[i] = True
    return chance, level


def goal_key(gl: dict, mine: bool, worth: dict, goal_kind: str, fields: tuple, owner: str) -> str:
    """The key of one goal as the wake logs it (a goal of a decision's log; `owner`: the role that holds its target), on `fields`, in the text the wake builds from the goal itself: the two must agree (tests/test_plan79_13.py reads a decision of the wake against this)."""
    v = {"source": gl["source"], "unmet": "+".join(sorted(set(gl["unmet"]))), "in_way": str(int(gl["in_the_way"])), "depth": str(int(gl["depth"])),
         "rank": str(GR.rank_of(worth, GR.kind_of(gl["target"], owner), goal_kind)), "fix": "1" if any(f["moves"] for f in gl["fixes"]) else "0", "mine": "1" if mine else "0"}
    return "|".join(v[f] for f in fields)


def binding_form(table: dict) -> tuple:
    """The table as the wake's binding takes it (its `goal_table` argument): (the plies of the event, the floor, [(the field names, [(key, goals, reached)])], (goals, reached) in all)."""
    return (int(table["k"]), int(table["floor"]), [(list(lv["fields"]), [(c[0], int(c[1]), int(c[2])) for c in lv["cells"]]) for lv in table["levels"]], (int(table["total"][0]), int(table["total"][1])))


def load_table(tag: str) -> dict:
    """The table written by `main` for the wake (all the training games' goals counted)."""
    return json.loads((GR.out_root(tag) / "table.json").read_text())


def logloss_rows(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def by_tenth(p: np.ndarray, y: np.ndarray) -> list:
    out = []
    for b in range(10):
        m = (p >= b / 10) & (p < (b + 1) / 10 if b < 9 else p <= 1.0)
        out.append((b, int(m.sum()), float(p[m].mean()) if m.any() else float("nan"), float(y[m].mean()) if m.any() else float("nan")))
    return out


def boot_diff(loss_a: np.ndarray, loss_b: np.ndarray, game: np.ndarray, draws: int = 400, seed: int = 7913) -> tuple:
    """The mean of loss_a - loss_b and a 90 pct band by bootstrap over GAMES."""
    ids, inv = np.unique(game, return_inverse=True)
    s = np.bincount(inv, weights=loss_a - loss_b)
    c = np.bincount(inv).astype(float)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(draws):
        m = np.bincount(rng.integers(0, len(ids), len(ids)), minlength=len(ids)).astype(float)
        out.append((m * s).sum() / (m * c).sum())
    lo, hi = np.percentile(out, (5, 95))
    return float(s.sum() / c.sum()), float(lo), float(hi)


def recalibrate(old_fit: np.ndarray, y_fit: np.ndarray, old_held: np.ndarray) -> np.ndarray:
    """The old table's number through a logistic fit of the event on its logit (fit half), the scale mended and the order kept."""
    from sklearn.linear_model import LogisticRegression
    z = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))[:, None]
    m = LogisticRegression(C=1e6).fit(z(old_fit), y_fit)
    return m.predict_proba(z(old_held))[:, 1]


def evaluate(d: dict, j: int, say) -> dict:
    K = GR.KS[j]
    fit = GR.fit_mask(d)
    ok = d["reached"][:, j] >= 0
    y = (d["reached"][:, j] == 1).astype(float)
    fi, ho = np.where(fit & ok)[0], np.where(~fit & ok)[0]
    table = fit_table(d, j, fit)
    p_new, lv = predict(table, d, ho, j)
    base = (table["total"][1] + 1.0) / (table["total"][0] + 2.0)
    has_old = d["old_chance"][ho] >= 0
    p_old = np.where(has_old, d["old_chance"][ho], base)
    fi_old = d["old_chance"][fi] >= 0
    p_old_cal = np.where(has_old, recalibrate(d["old_chance"][fi][fi_old], y[fi][fi_old], d["old_chance"][ho]), base)
    t = f"OBS whole_79_13 B K {K}"
    l_new, l_old, l_cal, l_base = (logloss_rows(p, y[ho]) for p in (p_new, p_old, p_old_cal, np.full(len(ho), base)))
    say(f"{t}: fit half {len(fi)} goals ({len(np.unique(d['game'][fi]))} games), held-out half {len(ho)} goals ({len(np.unique(d['game'][ho]))} games); the table: {sum(len(lv_['cells']) for lv_ in table['levels'])} cells of at least {FLOOR} goals, base rate of the fit half {base:.4f}")
    say(f"{t}: the level that spoke on the held-out goals (0 full key ... {len(LEVELS) - 1} the fewest, {len(LEVELS)} the whole): " + ", ".join(f"{i} {100 * np.mean(lv == i):.1f} pct" for i in range(len(LEVELS) + 1)))
    out = {"table": table, "k": K}
    for name, a, b, tag in (("old table as logged", l_old, l_new, "old"), ("old table recalibrated", l_cal, l_new, "cal"), ("base rate", l_base, l_new, "base")):
        m, lo, hi = boot_diff(a, b, d["game"][ho])
        out[tag] = (m, lo, hi)
        say(f"{t}: held-out mean log-loss: new table {l_new.mean():.5f}, {name} {a.mean():.5f}; {name} less new {m:+.5f} [{lo:+.5f}, {hi:+.5f}] (90 pct band by bootstrap over games; positive: the new table is better)")
    for name, p in (("new", p_new), ("old as logged", p_old), ("old recalibrated", p_old_cal)):
        say(f"{t}: reached share by tenth of predicted chance, {name} table (tenth: goals, mean predicted, reached): " + "; ".join(f"{b} {n} {pm:.3f} {ym:.3f}" for b, n, pm, ym in by_tenth(p, y[ho]) if n))
    out["better_than_old"] = out["old"][1] > 0 and out["cal"][1] > 0
    out["better_than_base"] = out["base"][1] > 0
    out["stop"] = not (out["better_than_old"] and out["better_than_base"])
    say(f"{t}: {'STOP: the new table is not clearly better than the old one and the base rate on held-out goals: the facts of a goal in the log do not predict it being reached' if out['stop'] else 'the new table is better than the old one (as logged and recalibrated) and than the base rate on held-out goals, the bands clear of nought: it predicts'}")
    return out


def main(argv: "list | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("tag")
    ap.add_argument("--also", default="8,16", help="other K read for direction only")
    a = ap.parse_args(argv)
    d = load(a.tag)
    root = GR.out_root(a.tag)
    lines: list = []

    def say(x: str) -> None:
        lines.append(x)
        print(x)
    k_star = int(d["k_star"])
    res = evaluate(d, GR.KS.index(k_star), say)
    for k in [int(x) for x in a.also.split(",") if x]:
        evaluate(d, GR.KS.index(k), say)
    (root / "table_obs.txt").write_text("\n".join(lines) + "\n")
    if not res["stop"]:
        j = GR.KS.index(k_star)
        (root / "table_fit_half.json").write_text(json.dumps(res["table"]))
        full = fit_table(d, j, np.ones(len(d["game"]), bool))
        (root / "table.json").write_text(json.dumps(full))
        say(f"OBS whole_79_13 B: table written: {sum(len(l['cells']) for l in full['levels'])} cells from all {full['total'][0]} counted goals of the training games, K {full['k']}")
    else:
        say("OBS whole_79_13 B: no table written for the wake; the plan stops here")
    (root / "table_obs.txt").write_text("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
