"""PLAN 78.58 A AND B: THE ESTIMATOR OF THE RESULT, FITTED FROM SELF-PLAY ONLY, AND THE ORDER OF KINDS IT GIVES. A measurement in a sandbox; nothing is wired into any player, nothing is written to
rules/learned or data/derived; no oracle, no human game, no puzzle, no stored valuation. Game-free: a kind is a word read from the graph at run time; no unit, place or kind of any game is named here.

THE DATA (stage A): every self-play game record on disk whose players read no teacher and no oracle (the SETS table below carries each set's provenance, checked against the scripts that wrote it).
A game is used only if it FINISHED (a game stopped at the move cap has no result and is left out, counted). A position is a row when it is QUIET and the side to move is not the random opening's:
QUIET here = nothing stands to be lost by inclusion for either side (no unit borne on by the other side and not by its own): the engine's `position_table`, the 78.57 Q0 asked of both sides, because the
forcing reads of the quiet rows of 78.49 exist only for 700 of these games and cost 0.05 s a position. The split is BY OPENING (all games that start from the same random opening sit on one side):
fit half and test half. Duplicate games (the same moves) are counted and dropped.

THE ESTIMATOR (stage B): in, the parts of a position as counts: for each side the units by kind (raw symbols read from the graph), the difference ours less theirs of the side to move; the second set of inputs
adds the counts of the distinct kept concepts of 78.54 (the top CONCEPTS by |z| on its half A), also ours less theirs. Out, the probability of win, draw and loss for the side to move: a regularised multinomial
logistic model (L2, the penalty chosen by cross-validation grouped by game on the fit half). Beside it the record of its source, its data and its date (`estimator.json`).
THE ORDER OF KINDS: the term of a kind is the log-odds of a win against a loss, per extra unit of the kind, for the side to move (coefficient on win less coefficient on loss; the draw is the reference).
Its 90 pct band is by bootstrap over the games of the fit half (the model refitted on each draw). Two kinds are ORDERED when one is above the other in at least 95 pct of the draws, otherwise NOT ORDERED.

  .venv/bin/python -m catspace.incubator.analyst.discovery.estimator_fit --out-dir O [--short N] [--draws D] [--threads K]
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

from catspace.incubator.analyst.discovery import frames_measure as OLD
from catspace.incubator.analyst.discovery import graph_layer1 as GL
from catspace.incubator.analyst.discovery import graph_layer2 as LAYER2
from catspace.incubator.analyst.discovery import graph_layer2_measure as G2M
from catspace.io import paths

CONCEPTS = 64
CONCEPT_EVERY = 4                                                    # the concepts are counted on every 4th quiet row of each game (the count costs about 9 ms a row over 10 threads)
PROFILE_GAMES = 50
DRAWS = 200
SEED = 5858
C_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)
FOLDS = 5
CALIB_BINS = 10
VARIANT = "chess"
GAME = "chess"
ARTIFACTS = Path(paths.experiment("estimator_78_58")).parent

# (label, directory under artifacts/experiments, what the players were, what they read). Provenance checked on 2026-11-15 against the scripts that wrote each set:
# opportunity_measure.make_games and forcing_opportunity_measure.make_weak_games (sheet_player `control`: the plain search `cs_engine.solve_batch` of the wake against itself, the judge being the lemma
# store's VALUATION "units (no values)" -- origin HAND, every unit counts one -- no teacher, no oracle, no human game, no tablebase; the opening is four random plies) and subgoal_play_measure matches
# (the same plain search; `random` a seeded random mover; `engine` the same search with the equal-ties reading; `candidate` and `ordered*` the search with a root order from the 78.49 fit, itself fitted on
# the even games of the 500 self-play games above; `random_order_ties` the same with a random order). None of these reads a teacher, an oracle or a human game.
SETS = [
    ("selfplay500", "opportunity_78_44/games_s5016_n500", "plain search v plain search, 500 games, seeds 5016..5515 (78.44)"),
    ("selfplay16", "opportunity_78_44/games_s5000_n16", "plain search v plain search, 16 games, seeds 5000..5015 (78.44 short run)"),
    ("weak200", "forcing_opportunity_78_48/weak_s5516_n200", "plain search v plain search with one colour on 0.02 s a move, 200 games (78.48)"),
    ("m50_control_control", "subgoal_play_78_50/full/match_control_v_control", "plain v plain, 600 games (78.50)"),
    ("m50_random_control", "subgoal_play_78_50/full/match_random_v_control", "seeded random mover v plain search, 600 games (78.50)"),
    ("m50_candidate_control", "subgoal_play_78_50/full/match_candidate_v_control", "search with a root order from a self-play fit v plain search, 600 games (78.50)"),
    ("m50_candidate_engine", "subgoal_play_78_50/full/match_candidate_v_engine", "the same candidate v the search's own equal reading, 600 games (78.50)"),
    ("m50_engine_control", "subgoal_play_78_50/full/match_engine_v_control", "the search's equal reading v plain search, 600 games (78.50)"),
    ("m51_ordered_control", "subgoal_play_78_51/full/match_ordered_v_control", "ordered root v plain search, 600 games (78.51)"),
    ("m51_ordered_ties_control", "subgoal_play_78_51/full/match_ordered_ties_v_control", "ordered root, ties only, v plain search, 600 games (78.51)"),
    ("m51_random_order_ties_control", "subgoal_play_78_51/full/match_random_order_ties_v_control", "random order, ties only, v plain search, 600 games (78.51)"),
    ("m51_rep2_ordered_ties_control", "subgoal_play_78_51/rep2/match_ordered_ties_v_control", "a second run of ordered ties v plain search, 600 games (78.51 rep2)"),
]
SCORE = {(100, 0): 0, (50, 50): 1, (0, 100): 2}                     # the result for the first mover: 0 win, 1 draw, 2 loss


def games_of(short: int) -> "tuple[list, list]":
    """(games, set lines): each game {set, id, moves, opening_len, first_mover_result, key}; unfinished games, openings-only and duplicates are counted per set."""
    games, lines, seen = [], [], set()
    for label, sub, what in SETS:
        p = ARTIFACTS / sub / "games.jsonl"
        n = fin = unf = dup = plies = 0
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            n += 1
            if short and n > short:
                break
            if r.get("unfinished") or not r.get("outcome"):
                unf += 1
                continue
            tup = tuple(r["moves"])
            if tup in seen:
                dup += 1
                continue
            seen.add(tup)
            fin += 1
            plies += len(tup)
            gid = r.get("gid") or str(r.get("game"))
            ol = int(r["opening_len"])
            games.append({"set": label, "id": f"{label}:{gid}", "moves": list(r["moves"]), "opening_len": ol, "result": SCORE[tuple(r["outcome"])], "start": r.get("start"),
                          "half": int(hashlib.md5(" ".join(r["moves"][:ol]).encode()).hexdigest(), 16) % 2})
        lines.append({"set": label, "dir": sub, "what": what, "read": n if not short else min(n, short), "finished": fin, "unfinished": unf, "duplicates": dup, "plies": plies})
    return games, lines


def rows_of(model, games: list, threads: int) -> "tuple[dict, list]":
    """The quiet rows of all games: arrays game, ply, mover result, the unit counts by kind (ours, theirs), and the kind names."""
    wires = [(g["start"], g["moves"]) for g in games]
    names, raw = model.position_table(wires, threads)
    nk = len(names)
    flat = np.frombuffer(raw, dtype=np.int16)
    gi, ply, res, qs, ours, theirs = [], [], [], [], [], []
    off = 0
    w = 2 + 2 * nk
    for i, g in enumerate(games):
        n = len(g["moves"]) + 1
        t = flat[off:off + n * w].reshape(n, w).astype(int)
        off += n * w
        for p in range(g["opening_len"], n):
            if t[p, 0] != 1 or t[p, 1] != 1:
                continue
            mover_first = p % 2 == 0
            r = g["result"] if mover_first else (2 - g["result"])
            gi.append(i); ply.append(p); res.append(r); ours.append(t[p, 2:2 + nk]); theirs.append(t[p, 2 + nk:2 + 2 * nk])
    return {"game": np.array(gi), "ply": np.array(ply), "y": np.array(res), "ours": np.array(ours), "theirs": np.array(theirs)}, names


def concept_columns(model, games: list, rows: dict, goal: "str | None", threads: int, concepts_dir: Path):
    """(names, matrix) of ours less theirs for the top CONCEPTS distinct kept concepts of 78.54, over the rows."""
    picked = OLD.pick_concepts(concepts_dir, CONCEPTS)
    gidx = np.unique(rows["game"])
    wires, rls = [], []
    starts = {}
    for g in gidx:
        sel = np.flatnonzero(rows["game"] == g)
        starts[int(g)] = sel
        wires.append((games[int(g)]["start"], games[int(g)]["moves"]))
        rls.append([int(x) for x in rows["ply"][sel]])
    order = np.concatenate([starts[int(g)] for g in gidx])
    raw = model.pattern_count(wires, rls, [k for _dc, k, _d, _z in picked], 3, list(LAYER2.LEVELS), goal, threads)
    us, them = LAYER2.counts_matrix(raw, len(order), len(picked))
    diff = (us - them).toarray()
    out = np.zeros((len(order), len(picked)))
    out[:] = diff
    mat = np.zeros_like(out)
    mat[order] = out                                                                        # back to the row order of `rows`
    return [f"{dc}: {d}" for dc, _k, d, _z in picked], mat


def fit_lr(x: np.ndarray, y: np.ndarray, c: float):
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(C=c, max_iter=1000).fit(x, y)


def probs(m, x: np.ndarray) -> np.ndarray:
    p = np.zeros((len(x), 3))
    p[:, m.classes_] = m.predict_proba(x)
    return np.clip(p, 1e-9, 1.0)


def model_record(m, c: float) -> dict:
    """The fitted model as plain numbers: the record's `kinds_model`."""
    return {"C": c, "intercept": [float(v) for v in m.intercept_], "coef": [[float(v) for v in row] for row in m.coef_], "classes": [int(k) for k in m.classes_]}


def predict_record(rec: dict, x: np.ndarray) -> np.ndarray:
    """The probabilities (win, draw, loss) of the side to move from a record's model (`kinds_model`) and the counts x (ours less theirs, one column per kind of the record): the record alone is the estimator."""
    z = np.asarray(rec["intercept"])[None, :] + x @ np.asarray(rec["coef"]).T
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    q = e / e.sum(axis=1, keepdims=True)
    p = np.zeros((len(x), 3))
    p[:, rec["classes"]] = q
    return np.clip(p, 1e-9, 1.0)


def kind_terms(classes: list, coef: np.ndarray) -> np.ndarray:
    """The term of each kind: the coefficient on a win less the coefficient on a loss (the draw is the reference), per extra unit of the kind for the side to move."""
    full = np.zeros((3, coef.shape[1]))
    full[list(classes)] = coef
    return full[0] - full[2]


def logloss(p: np.ndarray, y: np.ndarray) -> float:
    return float(-np.mean(np.log(p[np.arange(len(y)), y])))


def choose_c(x: np.ndarray, y: np.ndarray, groups: np.ndarray) -> "tuple[float, dict]":
    from sklearn.model_selection import GroupKFold
    cv = list(GroupKFold(n_splits=FOLDS).split(x, y, groups))
    scores = {}
    for c in C_GRID:
        s = []
        for tr, te in cv:
            m = fit_lr(x[tr], y[tr], c)
            s.append(logloss(probs(m, x[te]), y[te]))
        scores[c] = float(np.mean(s))
    return min(scores, key=scores.get), scores


def base_probs(y_fit: np.ndarray, n: int) -> np.ndarray:
    f = np.bincount(y_fit, minlength=3) / len(y_fit)
    return np.tile(f, (n, 1))


def calibration(p: np.ndarray, y: np.ndarray) -> list:
    """Equal-count bins of the predicted score (win + half a draw): [(rows, mean predicted score, observed score, mean predicted win, observed win)]."""
    sc = p[:, 0] + 0.5 * p[:, 1]
    obs = (y == 0) + 0.5 * (y == 1)
    order = np.argsort(sc, kind="stable")
    out = []
    for b in np.array_split(order, CALIB_BINS):
        if len(b):
            out.append((len(b), float(sc[b].mean()), float(obs[b].mean()), float(p[b, 0].mean()), float((y[b] == 0).mean())))
    return out


def game_bootstrap(per_game: np.ndarray, fn, draws: int, seed: int) -> "tuple[float, float, float]":
    """(value, 5th, 95th percentile) of fn(resampled game table) over resamplings of the games; per_game rows are one game each."""
    rng = np.random.default_rng(seed)
    n = len(per_game)
    vals = np.array([fn(per_game[rng.integers(0, n, n)]) for _ in range(draws)])
    return float(fn(per_game)), float(np.percentile(vals, 5)), float(np.percentile(vals, 95))


def per_game_table(game: np.ndarray, columns: list) -> np.ndarray:
    """Per game: row count, then the sum of each column over its rows (so a mean over a resample of games is the ratio of sums)."""
    ids, inv = np.unique(game, return_inverse=True)
    t = np.zeros((len(ids), 1 + len(columns)))
    np.add.at(t[:, 0], inv, 1.0)
    for j, c in enumerate(columns):
        np.add.at(t[:, 1 + j], inv, c)
    return t


def main(argv: "list | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--short", type=int, default=0, help="the first N records of every set only")
    ap.add_argument("--draws", type=int, default=DRAWS)
    ap.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--no-concepts", action="store_true")
    a = ap.parse_args(argv)
    t0 = time.perf_counter()
    out = Path(a.out_dir or paths.experiment("estimator_78_58", "short" if a.short else "full"))
    out.mkdir(parents=True, exist_ok=True)
    lines: list = []

    def say(msg: str) -> None:
        lines.append(f"OBS estimator: {msg}")
        print(lines[-1], flush=True)

    games, setlines = games_of(a.short)
    for s in setlines:
        say(f"A set {s['set']} ({s['what']}): records read {s['read']}, finished and kept {s['finished']}, unfinished (no result, left out) {s['unfinished']}, duplicates of an earlier game dropped {s['duplicates']}, plies {s['plies']}")
    model = GL.model(GAME, VARIANT)
    goal = G2M.goal_kind_of(VARIANT)
    first = [g for g in games if g["set"] == "selfplay500"][:PROFILE_GAMES] or games[:PROFILE_GAMES]
    model.learn_profile([(g["start"], g["moves"]) for g in first])
    t1 = time.perf_counter()
    rows, kinds = rows_of(model, games, a.threads)
    say(f"A positions: games {len(games)}, quiet rows after the opening {len(rows['y'])} (the position table took {time.perf_counter() - t1:.1f} s); goal kind (read from the graph) {goal}")
    half = np.array([games[i]["half"] for i in rows["game"]])
    fit_sel, test_sel = np.flatnonzero(half == 0), np.flatnonzero(half == 1)
    dec = lambda sel: len({int(g) for g in rows["game"][sel] if games[int(g)]["result"] != 1})
    say(f"A split by opening: fit half {len(np.unique(rows['game'][fit_sel]))} games, {len(fit_sel)} rows, {dec(fit_sel)} decisive games; test half {len(np.unique(rows['game'][test_sel]))} games, {len(test_sel)} rows, {dec(test_sel)} decisive games")
    for h, sel in (("fit", fit_sel), ("test", test_sel)):
        c = np.bincount(rows["y"][sel], minlength=3)
        say(f"A result of the side to move over the {h} rows: win {c[0] / len(sel):.4f}, draw {c[1] / len(sel):.4f}, loss {c[2] / len(sel):.4f}")
    use = [k for k, nm in enumerate(kinds) if nm != goal]
    kind_names = [kinds[k] for k in use]
    xk = (rows["ours"] - rows["theirs"])[:, use].astype(float)
    y = rows["y"]
    groups = rows["game"]
    first_of = np.searchsorted(groups, groups)                                  # rows are in game order: the index of the first row of each row's game
    sub = ((np.arange(len(groups)) - first_of) % CONCEPT_EVERY) == 0               # the rows on which the concepts are counted (and the three-way comparison is made)
    cnames, xc = [], np.zeros((int(sub.sum()), 0))
    if not a.no_concepts:
        t2 = time.perf_counter()
        cdir = Path(paths.experiment("unit_fates_78_54", "games_s5016_n500"))
        rows_s = {k: v[sub] for k, v in rows.items()}
        cnames, xc = concept_columns(model, games, rows_s, goal, a.threads, cdir)
        say(f"B concepts: the top {len(cnames)} distinct kept concepts of 78.54 by |z| on its half A, counted ours less theirs on every {CONCEPT_EVERY}th quiet row of each game, {int(sub.sum())} rows ({time.perf_counter() - t2:.1f} s)")

    # ---------------------------------------------------------------- the estimators
    p_base = base_probs(y[fit_sel], len(test_sel))
    c_k, sc_k = choose_c(xk[fit_sel], y[fit_sel], groups[fit_sel])
    m_k = fit_lr(xk[fit_sel], y[fit_sel], c_k)
    p_k = probs(m_k, xk[test_sel])
    say(f"B kinds (all rows): penalty C {c_k} by {FOLDS}-fold cross-validation grouped by game ({', '.join(f'C={c}: {s:.5f}' for c, s in sc_k.items())})")
    sets = {"all quiet rows": (test_sel, {"base rate": p_base, "kinds": p_k})}
    m_kc = None
    c_kc = None
    sd = None
    fs, ts = fit_sel[sub[fit_sel]], test_sel[sub[test_sel]]                     # the same comparison on the rows that carry concepts
    pos = np.cumsum(sub) - 1                                                      # a row's place among the concept rows
    if xc.shape[1]:
        sd = xc[pos[fs]].std(axis=0)
        sd = np.where(sd > 0, sd, 1.0)
        x2 = np.hstack([xk[sub], xc / sd])
        fi, ti = pos[fs], pos[ts]
        c_ks, sc_ks = choose_c(xk[fs], y[fs], groups[fs])
        m_ks = fit_lr(xk[fs], y[fs], c_ks)
        c_kc, sc_kc = choose_c(x2[fi], y[fs], groups[fs])
        m_kc = fit_lr(x2[fi], y[fs], c_kc)
        sets[f"every {CONCEPT_EVERY}th row (the concept rows)"] = (ts, {"base rate": base_probs(y[fs], len(ts)), "kinds": probs(m_ks, xk[ts]), "kinds + concepts": probs(m_kc, x2[ti])})
        say(f"B kinds on the concept rows: penalty C {c_ks} ({', '.join(f'C={c}: {s:.5f}' for c, s in sc_ks.items())}); kinds + concepts: penalty C {c_kc} ({', '.join(f'C={c}: {s:.5f}' for c, s in sc_kc.items())}); concept columns scaled by their standard deviation")
    for label, (sel, forms) in sets.items():
        yt, gt = y[sel], groups[sel]
        per = {}
        for nm, p in forms.items():
            ll = -np.log(p[np.arange(len(yt)), yt])
            per[nm] = ll
            tab = per_game_table(gt, [ll])
            v, lo, hi = game_bootstrap(tab, lambda t: t[:, 1].sum() / t[:, 0].sum(), a.draws, SEED)
            say(f"B [{label}] test log loss, {nm}: {v:.5f} [{lo:.5f}, {hi:.5f}] over {len(yt)} rows of {len(np.unique(gt))} games ({a.draws} draws over games)")
        names = list(forms)
        for i in range(1, len(names)):
            for j in range(i):
                d = per[names[j]] - per[names[i]]
                tab = per_game_table(gt, [d])
                v, lo, hi = game_bootstrap(tab, lambda t: t[:, 1].sum() / t[:, 0].sum(), a.draws, SEED + 1)
                say(f"B [{label}] test log loss gain of {names[i]} over {names[j]}: {v:+.5f} [{lo:+.5f}, {hi:+.5f}] (positive: the second is better)")
        for nm, p in forms.items():
            for k, (n_, mp, ob, mw, ow) in enumerate(calibration(p, yt)):
                say(f"B [{label}] calibration {nm}, bin {k + 1} of {CALIB_BINS}: rows {n_}, predicted score {mp:.4f}, observed score {ob:.4f}, predicted win {mw:.4f}, observed win {ow:.4f}")

    # ---------------------------------------------------------------- the order of kinds, with bands by bootstrap over the games of the fit half
    def terms(m) -> np.ndarray:
        return kind_terms(list(m.classes_), m.coef_[:, :xk.shape[1]])
    base_terms = terms(m_k)
    fit_games = np.unique(groups[fit_sel])
    by_game = {int(g): np.flatnonzero(groups == g) for g in fit_games}
    rng = np.random.default_rng(SEED + 2)
    t3 = time.perf_counter()
    draws = []
    for _ in range(a.draws):
        pick = rng.choice(fit_games, size=len(fit_games), replace=True)
        sel = np.concatenate([by_game[int(g)] for g in pick])
        draws.append(terms(fit_lr(xk[sel], y[sel], c_k)))
    draws = np.array(draws)
    lo, hi = np.percentile(draws, 5, axis=0), np.percentile(draws, 95, axis=0)
    order = np.argsort(base_terms)
    say(f"B ORDER OF KINDS (term = log-odds of a win against a loss per extra unit, side to move; 90 pct band over {a.draws} refits on resampled fit games; {time.perf_counter() - t3:.1f} s): "
        + "; ".join(f"{kind_names[k]} {base_terms[k]:+.4f} [{lo[k]:+.4f}, {hi[k]:+.4f}]" for k in order))
    pairs, unordered = [], []
    for i in range(len(kind_names)):
        for j in range(i + 1, len(kind_names)):
            f = float((draws[:, j] > draws[:, i]).mean())
            tag = "ordered" if (f >= 0.95 or f <= 0.05) else "NOT ORDERED"
            pairs.append((kind_names[i], kind_names[j], f, tag))
            if tag != "ordered":
                unordered.append((kind_names[i], kind_names[j]))
            say(f"B pair {kind_names[j]} above {kind_names[i]} in {f:.3f} of the draws: {tag}")
    say(f"B kinds the data cannot order (pairs): {unordered if unordered else 'none'}")

    # ---------------------------------------------------------------- the record
    scale = 1000
    vals = {kind_names[k]: int(round(scale * base_terms[k])) for k in range(len(kind_names))}
    if goal is not None:
        vals[goal] = int(sum(max(0, v) for v in vals.values()) + 1)                # a RULE, not a fit: the goal unit outweighs every other unit together (its loss is the game)
    rec = {
        "estimator": "multinomial logistic on counts of kinds (ours less theirs), side to move; classes 0 win 1 draw 2 loss",
        "source": "self-play only: games of the plain search against itself and against random and ordered variants of itself; no teacher, no oracle, no human game, no tablebase, no stored valuation",
        "date": datetime.date.today().isoformat(),
        "data": {"sets": setlines, "games": len(games), "rows": int(len(y)), "fit_rows": int(len(fit_sel)), "test_rows": int(len(test_sel)),
                 "quiet": "nothing stands to be lost by inclusion for either side (engine position_table)", "split": "by the random opening (md5 of the opening moves, parity)"},
        "kinds": kind_names, "goal_kind": goal, "scale": scale,
        "kinds_model": model_record(m_k, c_k),
        "concepts_model": None if m_kc is None else {"names": cnames, "concept_scale": [float(v) for v in (sd if sd is not None else [])], "C": c_kc, "intercept": [float(v) for v in m_kc.intercept_], "coef": [[float(v) for v in row] for row in m_kc.coef_], "classes": [int(c) for c in m_kc.classes_]},
        "order_of_kinds": [{"kind": kind_names[k], "term": float(base_terms[k]), "lo": float(lo[k]), "hi": float(hi[k]), "value": vals[kind_names[k]]} for k in order],
        "unordered_pairs": unordered,
        "kind_values": vals,
        "kind_values_note": f"integer = round({scale} x term); the goal kind's value is a rule (more than every other together), not a fit",
    }
    (out / "estimator.json").write_text(json.dumps(rec, indent=1))
    say(f"B record written: {out / 'estimator.json'}; kind values for the exchange {vals}")
    say(f"done in {time.perf_counter() - t0:.1f} s wall")
    (out / "obs.txt").write_text("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
