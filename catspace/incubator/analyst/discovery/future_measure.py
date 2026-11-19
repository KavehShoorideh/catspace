"""PLAN 78.94 (SYSTEM_SPEC 7 "the future imagination layer", 9 the oracle's labels, 10 the data rules, 11 the judge against the board): THE MEASURE STAGES of the future judge. A sandbox: no player is changed; nothing is written to rules/learned or data/derived; every output goes under
artifacts/experiments/future_78_94/<tag>/. Game-free: the game is named only through the pieces that exist (future_labels, future_goals, future_judge, human_heads_data, graph_layer1, symbols_play's pattern reader); the board and the words of a case are printed by ladder_print only.
Every number is an OBS line (also in obs.txt). One run each: directions, not verdicts. No player id is written (a game is an ordinal).

STAGES (each a command; the labels come first, made by future_labels.py `label`).
  counts   (step two)   the labels read: classes by band of units and by part; coverage per band for every sample; how often goals occur (by band and by ply, for the mover and for the other side), their kinds and stakes, what followed within six plies (units won, part, nothing, lost) and
                        the game's result beyond what the count of units already says; the frames at the changes of class and at goals, by type, and which of those that precede a win are NOT two threats at once; the share of oracle frames our reasoner does not see, by band; the cost.
  fit      (step one)   the judge: its inputs, the effective rank of the encoder's output (bootstrapped), the heads on FIND (held-out loss printed), the pairwise form; saved to judge.pkl.
  score                 on CONFIRM, per head and window: log loss, Brier, area under the ranking curve and calibration, against the base rate per band of units and the two baselines (the count of units; the pattern reasons of 78.81), paired bands over games.
  explicit (the cross-reading)  on a sample of CONFIRM positions: the explicit run of own moves (imagine, veto) against the judge's chance.
  human    (step three) among the moves the search holds equal in the human samples' CONFIRM positions: the move the judge picks against the move played, the search's first move and the pattern pick, paired bands.
  cases                 thirty positions for ladder_print --future94.

  .venv/bin/python -m catspace.incubator.analyst.discovery.future_measure counts|fit|score|explicit|human|cases --tag short|full
Called by: the command line only; ladder_print.py reads cases94.json.
"""
from __future__ import annotations

from catspace.incubator.analyst.discovery import tiebreak_measure as TM          # first: it sets one thread a process before numpy is loaded

import argparse
import json
import os
import pickle
import time
from collections import Counter
from pathlib import Path

import numpy as np

from catspace.incubator.analyst.discovery import future_judge as FJ
from catspace.incubator.analyst.discovery import future_labels as FL
from catspace.incubator.analyst.discovery import ladder_measure as LM
from catspace.incubator.analyst.discovery import symbols_measure as SM
from catspace.io import paths

BANDS = LM.BANDS                                              # phase by the count of units (the ladder's cut: few 0-8, mid 9-16, many 17 up)
FINE = ((0, 4), (5, 8), (9, 12), (13, 16), (17, 24), (25, 99))  # the finer bins of the coverage print
DRAWS = 400
SEED = 9494


def band_index(units: np.ndarray) -> np.ndarray:
    return np.array([next(i for i, (_n, lo, hi) in enumerate(BANDS) if lo <= u <= hi) for u in units])


def load_recs(root: Path) -> list:
    return [json.loads(x) for x in (root / "labels.jsonl").read_text().splitlines() if x.strip()]


def boot(game: np.ndarray, fn, draws: int = DRAWS, seed: int = SEED) -> "tuple[float, float, float]":
    """(value, 5th, 95th percentile) of fn(selected row indices) with the games resampled whole."""
    gs = np.unique(game)
    index = {g: np.flatnonzero(game == g) for g in gs}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(draws):
        pick = rng.choice(gs, len(gs), replace=True)
        v = fn(np.concatenate([index[g] for g in pick]))
        if v == v:
            vals.append(v)
    full = fn(np.arange(len(game)))
    if not vals:
        return full, float("nan"), float("nan")
    return float(full), float(np.percentile(vals, 5)), float(np.percentile(vals, 95))


# ------------------------------------------------------------------------------------------------------------------------------ counts

def frame_type(q: dict) -> str:
    """The word for the oracle's frame at a labelled position, read in our own facts: the first step of its line at which a goal of the side that moves first holds (open or fork), else a forced end, else a threat, else a taking along the line, else nothing read."""
    fr = q.get("frame") or {}
    for j in sorted(fr, key=int):
        if fr[j]["goal"]:
            return f"goal ({fr[j]['kind']})"
    if any(fr[j]["forced"] for j in fr):
        return "a forced end"
    if any(fr[j]["threats"] > 0 for j in fr):
        return "one threat, no goal"
    root_other = q["units"] - q["n_us"]
    if fr and min(fr[j]["n_second"] for j in fr) < root_other:
        return "a unit is taken along the line, no threat read"
    return "nothing our reading names (quiet line)"


def follow(pos: list, i: int, side_is_mover: bool, steps: int = 3) -> "str | None":
    """What followed a goal at pos[i] within `steps` sampled positions (six plies): the units the side won less the units it lost, by the count of units of each side (a count, no worth): 'won', 'nothing', 'lost' or None when the game ended first."""
    q = pos[i]
    if i + steps >= len(pos) or pos[i + steps]["ply"] != q["ply"] + 2 * steps:
        return None
    r = pos[i + steps]
    ns0, no0 = (q["n_us"], q["units"] - q["n_us"]) if side_is_mover else (q["units"] - q["n_us"], q["n_us"])
    ns1, no1 = (r["n_us"], r["units"] - r["n_us"]) if side_is_mover else (r["units"] - r["n_us"], r["n_us"])
    gain = (no0 - no1) - (ns0 - ns1)
    return "won" if gain > 0 else ("lost" if gain < 0 else "nothing")


def cmd_counts(root: Path, say) -> None:
    recs = load_recs(root)
    pos = [(r, q) for r in recs for q in r["positions"]]
    say(f"labels: {len(recs)} games (find {sum(r['part'] == 'find' for r in recs)}, confirm {sum(r['part'] == 'confirm' for r in recs)}), {len(pos)} sampled positions (every {FL.STRIDE}nd ply from {FL.MIN_PLY}); the oracle: depth {FL.FUTURE_DEPTH}, one line, one thread, default hash; "
        f"class winning or losing at {FL.WIN_CP} on its own scale or a forced end, else neither")
    wall = {k: sum(r["wall"][k] for r in recs) for k in ("read", "lines", "oracle", "frames")}
    say(f"COST per position, one thread's seconds: threat reading {wall['read'] / len(pos):.3f}, concrete lines {wall['lines'] / len(pos):.3f}, oracle {wall['oracle'] / len(pos):.3f}, frames (the oracle's line read, only where one is made) {wall['frames'] / len(pos):.3f}")
    # coverage and classes
    for part in ("find", "confirm"):
        sel = [(r, q) for r, q in pos if r["part"] == part]
        u = np.array([q["units"] for _r, q in sel])
        say(f"COVERAGE {part}: {len(sel)} positions in {len({r['ord'] for r, _q in sel})} games; by band of units: " + "; ".join(f"{n} ({lo}-{hi if hi < 99 else 'more'}) {int(((u >= lo) & (u <= hi)).sum())} ({100 * ((u >= lo) & (u <= hi)).mean():.1f} pct)" for n, lo, hi in BANDS)
            + "; finer: " + "; ".join(f"{lo}-{hi if hi < 99 else 'more'}: {int(((u >= lo) & (u <= hi)).sum())}" for lo, hi in FINE))
        for n, lo, hi in BANDS:
            c = Counter(q["cls"] for _r, q in sel if lo <= q["units"] <= hi)
            tot = max(1, sum(c.values()))
            say(f"CLASS {part} {n}: winning {c[1]} ({100 * c[1] / tot:.1f} pct), losing {c[-1]} ({100 * c[-1] / tot:.1f} pct), neither {c[0]} ({100 * c[0] / tot:.1f} pct) for the side to move")
    ends = Counter(r["y"] for r in recs)
    say(f"RESULTS of the games for the first mover: win {ends[0]}, draw {ends[1]}, loss {ends[2]}")
    # goals by band and by ply
    for who, key in (("the side to move", "mover_g"), ("the other side", "other_g")):
        for n, lo, hi in BANDS:
            sel = [q for _r, q in pos if lo <= q["units"] <= hi]
            k = [q[key] for q in sel]
            say(f"GOALS {who} {n}: positions {len(sel)}, a goal holds in {sum(g['goal'] for g in k)} ({100 * sum(g['goal'] for g in k) / max(1, len(k)):.2f} pct); open {sum(g['kind'] == 'open' for g in k)}, fork {sum(g['kind'] == 'fork' for g in k)}; "
                f"a forced end of the game found {sum(g['forced'] for g in k)}; a confirmed threat but no goal {sum(g['threats'] > 0 and not g['goal'] for g in k)}; threats cut or needing a quiet first move {sum(g['unread'] + g['deferred'] > 0 for g in k)}")
        cuts = [(10, 19), (20, 29), (30, 39), (40, 59), (60, 999)]
        say(f"GOALS {who} by ply: " + "; ".join(f"{lo}-{hi if hi < 999 else 'more'}: {sum(q[key]['goal'] for _r, q in pos if lo <= q['ply'] <= hi)} of {sum(1 for _r, q in pos if lo <= q['ply'] <= hi)}" for lo, hi in cuts))
    shapes = Counter()
    stakes: dict = {"open": [], "fork": []}
    for _r, q in pos:
        for key in ("mover_g", "other_g"):
            g = q[key]
            if g["goal"]:
                shapes[(tuple(g["shape"]), g["kind"])] += 1
                stakes[g["kind"]].append(g["stake"])
    say("GOAL KINDS (the two threatened units by the judge's own worth order, and open or fork): " + "; ".join(f"{'+'.join(s)} {k}: {n}" for (s, k), n in shapes.most_common(10)))
    say("GOAL STAKES (the line reading's net in the measured record's own integers): " + "; ".join(f"{k}: median {np.median(v):.0f}, quartiles {np.percentile(v, 25):.0f} to {np.percentile(v, 75):.0f}, n {len(v)}" for k, v in stakes.items() if v))
    # what followed a goal, against the same band's other positions
    for kind in ("open", "fork", None):
        tallies: Counter = Counter()
        base: Counter = Counter()
        for r in recs:
            ps = r["positions"]
            for i, q in enumerate(ps):
                for key, side_is_mover in (("mover_g", True), ("other_g", False)):
                    f = follow(ps, i, side_is_mover)
                    if f is None:
                        continue
                    if q[key]["goal"] and (kind is None or q[key]["kind"] == kind):
                        tallies[f] += 1
                    elif not q[key]["goal"]:
                        base[f] += 1
        tt, bb = max(1, sum(tallies.values())), max(1, sum(base.values()))
        say(f"WHAT FOLLOWED a goal of kind {kind or 'any'} within six plies (units won less lost by count): won {tallies['won']} ({100 * tallies['won'] / tt:.1f} pct), nothing {tallies['nothing']} ({100 * tallies['nothing'] / tt:.1f} pct), lost {tallies['lost']} ({100 * tallies['lost'] / tt:.1f} pct) of {tt}; "
            f"the same for a side with no goal: won {100 * base['won'] / bb:.1f} pct, nothing {100 * base['nothing'] / bb:.1f} pct, lost {100 * base['lost'] / bb:.1f} pct of {bb}")
    real = Counter()
    for _r, q in pos:
        for key, sgn in (("mover_g", 1), ("other_g", -1)):
            if q[key]["goal"]:
                real[sgn * q["cls"]] += 1
    allside = Counter()
    for _r, q in pos:
        allside[q["cls"]] += 1
        allside[-q["cls"]] += 1
    tot_g, tot_a = max(1, sum(real.values())), sum(allside.values())
    say(f"IS THE GOAL REAL (the oracle's class of the side that holds the goal, at every labelled position with one): winning {real[1]} ({100 * real[1] / tot_g:.1f} pct), neither {real[0]} ({100 * real[0] / tot_g:.1f} pct), losing {real[-1]} ({100 * real[-1] / tot_g:.1f} pct) of {tot_g}; "
        f"the same over all positions and both sides: winning {100 * allside[1] / tot_a:.1f} pct, neither {100 * allside[0] / tot_a:.1f} pct, losing {100 * allside[-1] / tot_a:.1f} pct")
    paid: dict = {}
    for r in recs:
        ps = r["positions"]
        for i, q in enumerate(ps):
            for key, side_is_mover in (("mover_g", True), ("other_g", False)):
                if q[key]["goal"]:
                    f = follow(ps, i, side_is_mover)
                    if f is not None:
                        paid.setdefault((tuple(q[key]["shape"]), q[key]["kind"]), Counter())[f] += 1
    say("WHAT FOLLOWED by kind of goal (the chance each kind pays, within six plies): " + "; ".join(f"{'+'.join(s)} {k}: won {c['won']}, nothing {c['nothing']}, lost {c['lost']} of {sum(c.values())}" for (s, k), c in sorted(paid.items(), key=lambda t: -sum(t[1].values()))[:8]))
    # the result of the game beyond the count of units
    rows = FJ.rows_from_labels(recs)
    from sklearn.linear_model import LogisticRegression
    y = (rows["y"] == 0).astype(int)
    now_g = rows["now_G"]
    other_g = np.zeros(len(rows), int)
    # the other side's goal at the same position: the row of the other side at the same (game, ply)
    key = {(int(g), int(p), int(s)): i for i, (g, p, s) in enumerate(zip(rows["game"], rows["ply"], rows["mover_side"]))}
    for i, (g, p, s) in enumerate(zip(rows["game"], rows["ply"], rows["mover_side"])):
        other_g[i] = rows["now_G"][key[(int(g), int(p), 1 - int(s))]]
    X0 = np.stack([rows["n_s"], rows["n_o"], rows["ply"]], axis=1).astype(float)
    X1 = np.hstack([X0, now_g[:, None], other_g[:, None]])
    mu, sd = X1.mean(axis=0), X1.std(axis=0)
    Z1 = (X1 - mu) / sd
    def coef(idx):
        m = LogisticRegression(C=1.0, max_iter=2000).fit(Z1[idx], y[idx])
        return float(m.coef_[0][3] / sd[3]), float(m.coef_[0][4] / sd[4])
    c_a, c_b = coef(np.arange(len(y)))
    lo_a, hi_a = [], []
    gs = np.unique(rows["game"])
    index = {g: np.flatnonzero(rows["game"] == g) for g in gs}
    rng = np.random.default_rng(SEED)
    bs = []
    for _ in range(100):
        pick = rng.choice(gs, len(gs), replace=True)
        bs.append(coef(np.concatenate([index[g] for g in pick])))
    bs = np.array(bs)
    say(f"RESULT beyond the count of units (a logistic of the side winning the game on its units, the other's, the ply, and whether a goal of its own or of the other side holds; every labelled row, both parts, a measure not a fit): "
        f"log-odds per goal of its own {c_a:+.3f} (90 pct {np.percentile(bs[:, 0], 5):+.3f} to {np.percentile(bs[:, 0], 95):+.3f}), per goal of the other side {c_b:+.3f} (90 pct {np.percentile(bs[:, 1], 5):+.3f} to {np.percentile(bs[:, 1], 95):+.3f}); rows with a goal of the side {int(now_g.sum())} of {len(y)}")
    # frames
    fr = [(r, q) for r, q in pos if "frame" in q]
    say(f"FRAMES made (the oracle's best line read by our threat list and line reading, at the first and second move of the side and the third): {len(fr)} positions: where the class of the side to move changes two plies on {sum(q['frame_why'] == 'changed' for _r, q in fr)}, only because a goal holds {sum(q['frame_why'] == 'goal' for _r, q in fr)}")
    for title, sel in (("all", fr), ("at a winning point for the side to move (the frame precedes a win)", [(r, q) for r, q in fr if q["cls"] == 1]), ("at a losing point", [(r, q) for r, q in fr if q["cls"] == -1]), ("neither", [(r, q) for r, q in fr if q["cls"] == 0])):
        c = Counter(frame_type(q) for _r, q in sel)
        say(f"FRAME TYPES {title}: {len(sel)}; " + "; ".join(f"{k} {v} ({100 * v / max(1, len(sel)):.1f} pct)" for k, v in c.most_common()))
    win = [(r, q) for r, q in fr if q["cls"] == 1]
    notg = [(r, q) for r, q in win if not frame_type(q).startswith("goal")]
    say(f"FRAMES THAT PRECEDE A WIN AND ARE NOT TWO THREATS AT ONCE: {len(notg)} of {len(win)}; kinds of unit threatened in the one-threat ones and the lengths of the oracle's line are in the cases")
    # what our reasoner sees
    for n, lo, hi in BANDS + (("all", 0, 99),):
        sel = [q for _r, q in fr if lo <= q["units"] <= hi]
        if not sel:
            continue
        unseen = sum(not q["seen"]["any"] for q in sel)
        say(f"THE REASONER'S GAP {n}: of {len(sel)} oracle frames, our concrete list holds a threat whose first move is the oracle's first move, or a forced end, in {len(sel) - unseen}; it cannot see {unseen} ({100 * unseen / len(sel):.1f} pct); "
            f"by a taking {sum(q['seen']['take'] for q in sel)}, by the quiet first move of a threat in two moves {sum(q['seen']['lead'] for q in sel)}, by a forced end {sum(q['seen']['forced'] for q in sel)}")
    for ft in sorted({frame_type(q) for _r, q in fr}):
        sel = [q for _r, q in fr if frame_type(q) == ft]
        say(f"THE REASONER'S GAP by frame type '{ft}': {len(sel)} frames, not seen {sum(not q['seen']['any'] for q in sel)} ({100 * sum(not q['seen']['any'] for q in sel) / len(sel):.1f} pct)")


# ------------------------------------------------------------------------------------------------------------------------------ the inputs of every labelled position

def games_by_ordinal() -> dict:
    return {g["ord"]: g for g in FL.universe()}


def build_inputs(root: Path, recs: list, say) -> dict:
    """The estimator's inputs and the pattern counts of every labelled position, in the order of the records' positions; cached under <tag>/inputs.npz."""
    f = root / "inputs.npz"
    if f.exists():
        z = np.load(f)
        return {k: z[k] for k in z.files}
    from catspace.incubator.analyst.discovery import graph_layer1 as GL
    from catspace.incubator.analyst.discovery import human_heads_data as HD
    from catspace.incubator.analyst.discovery import symbols_play as SP
    t0 = time.perf_counter()
    games = games_by_ordinal()
    wires = [(None, games[r["ord"]]["moves"]) for r in recs]
    plies = [[q["ply"] for q in r["positions"]] for r in recs]
    inp = FJ.Inputs(GL.model(HD.GAME, HD.VARIANT), HD.statements_chunked)
    out = inp.inputs_of(wires, plies, threads=max(1, (os.cpu_count() or 2) - 1))
    say(f"INPUTS: {len(out['ku'])} positions; units by kind {inp.kind_names}; statements kept (the ones whose description names no worth) {len(inp.keep)}: {inp.stat_names}; {time.perf_counter() - t0:.0f} s")
    t1 = time.perf_counter()
    counter = SP.reader82().counter
    us, them = counter.count(wires, plies)
    out["pu"], out["po"] = np.asarray(us.todense(), np.float32), np.asarray(them.todense(), np.float32)
    say(f"INPUTS: the pattern counts of 78.81's reasons: {out['pu'].shape[1]} patterns; {time.perf_counter() - t1:.0f} s")
    np.savez_compressed(f, **out)
    return out


def side_inputs(inp: dict, rows: FJ.Rows) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    """(X of the estimator's inputs, the unit-count baseline's matrix, the pattern baseline's matrix) for the table's rows: two rows a position, the mover's then the other side's."""
    pos = np.arange(len(rows)) // 2
    mv = rows["mover_side"].astype(bool)
    sub = {k: v[pos] for k, v in inp.items()}
    X = FJ.side_matrix(sub, mv)
    U = FJ.unit_matrix(sub, mv)
    f = mv[:, None]
    P = np.hstack([np.where(f, sub["pu"], sub["po"]), np.where(f, sub["po"], sub["pu"])])
    return X, U, P


def population(rows: FJ.Rows, name: str) -> np.ndarray:
    """The rows a target is asked on: the window inside the game, and the condition not already true at the row (it is a question about arriving)."""
    use = rows[name] >= 0
    w = name.rstrip("0123456789")
    if w == "GW":
        return use & (rows["now_G"] == 0) & (rows["now_W"] == 0)
    return use & (rows[f"now_{w}"] == 0)


def cmd_fit(root: Path, say) -> None:
    recs = load_recs(root)
    rows = FJ.rows_from_labels(recs)
    inp = build_inputs(root, recs, say)
    X, U, P = side_inputs(inp, rows)
    assert len(X) == len(rows)
    find = rows["part"] == 0
    say(f"ROWS: {len(rows)} (a position and a side); FIND {int(find.sum())} rows of {len(np.unique(rows['game'][find]))} games; CONFIRM {int((~find).sum())} rows of {len(np.unique(rows['game'][~find]))} games; windows k = {FJ.KS} plies; inputs {X.shape[1]} columns (the estimator's: kinds and worth-free statements, both sides, and the mover flag)")
    rank = FJ.effective_rank(X[find])
    gs = np.unique(rows["game"][find])
    rng = np.random.default_rng(SEED)
    idx = {g: np.flatnonzero(rows["game"] == g) for g in gs}
    rb = [FJ.effective_rank(X[np.concatenate([idx[g] for g in rng.choice(gs, len(gs), replace=True)])]) for _ in range(20)]
    say(f"COLLAPSE TEST: effective rank of the encoder's standardised output on FIND {rank:.2f} of {X.shape[1]} columns (the games resampled, 20 draws: {np.percentile(rb, 5):.2f} to {np.percentile(rb, 95):.2f}); the pattern baseline's {FJ.effective_rank(P[find]):.2f} of {P.shape[1]}")
    for k in FJ.KS:
        for name in (f"G{k}", f"W{k}", f"L{k}", f"GW{k}"):
            use = population(rows, name) & find
            say(f"TARGET {name}: rows on FIND {int(use.sum())}, positives {int(rows[name][use].sum())} ({100 * float(rows[name][use].mean()) if use.any() else 0:.2f} pct)")
    j = FJ.FutureJudge()
    t0 = time.perf_counter()
    j.fit(rows.select(find), X[find], say=lambda m: say("FIT " + m))
    say(f"FIT done in {time.perf_counter() - t0:.0f} s; heads fitted: {sorted(n for n, i in j.info.items() if i['fitted'])}; not fitted (too few positives or negatives): {sorted(n for n, i in j.info.items() if not i['fitted'])}")
    base = {}
    for name in [n for n, i in j.info.items() if i["fitted"]]:
        use = population(rows, name) & find
        y, g, b = rows[name][use], rows["game"][use], band_index(rows["units"][use])
        base[name] = {"units": FJ.fit_linear(U[use], y, g), "patterns": FJ.fit_linear(P[use], y, g), "band_rate": [float(y[b == i].mean()) if (b == i).any() else float(y.mean()) for i in range(len(BANDS))]}
    vocab = FJ.target_vocabulary([r for r in recs if r["part"] == "find"])
    say(f"PAIRWISE: targets are descriptions of a goal in facts (the kinds of its two threatened units, larger worth first, and open or fork); {len(vocab)} occur in the FIND games: " + "; ".join("+".join(d[0]) + " " + d[1] for d in vocab[:8]))
    pairs = {}
    for k in (8, 16):
        pr = FJ.pair_rows(recs, rows, vocab, k, seed=SEED)
        fit_pairs = {kk: v[rows["part"][pr["row"]] == 0] for kk, v in pr.items()}
        say(f"PAIRWISE k {k}: pairs on FIND {len(fit_pairs['label'])}, positives {int(fit_pairs['label'].sum())} (negatives: the same game's past goals and descriptions drawn from the vocabulary, labelled by whether the description does come true in the window)")
        if len(fit_pairs["label"]) and int(fit_pairs["label"].sum()) >= 25:
            j.fit_pair(rows, X, fit_pairs, vocab, k, say=lambda m: say("FIT " + m))
        pairs[k] = pr
    pickle.dump({"judge": j, "base": base, "vocab": vocab}, open(root / "judge.pkl", "wb"))
    say(f"saved {root / 'judge.pkl'}")


def cmd_score(root: Path, say) -> None:
    recs = load_recs(root)
    rows = FJ.rows_from_labels(recs)
    inp = build_inputs(root, recs, say)
    X, U, P = side_inputs(inp, rows)
    saved = pickle.load(open(root / "judge.pkl", "rb"))
    j, base = saved["judge"], saved["base"]
    conf = rows["part"] == 1
    band = band_index(rows["units"])
    say(f"SCORE on CONFIRM: {int(conf.sum())} rows of {len(np.unique(rows['game'][conf]))} games; the judge's head per target is the one with the lower held-out loss on FIND; baselines: the count of units; 78.81's pattern counts; the base rate of FIND by band of units. "
        f"Log-loss gain = the band's base rate's loss less the model's (positive is better than the base rate); AUC = the ranking; paired band = games resampled, judge less baseline")
    table = []
    for name in sorted(j.info):
        if not j.info[name]["fitted"]:
            say(f"SCORE {name}: not fitted ({j.info[name]['positives']} positives on FIND)")
            continue
        use = population(rows, name) & conf
        if int(use.sum()) < 20 or len(np.unique(rows[name][use])) < 2:
            say(f"SCORE {name}: too few CONFIRM rows or one class ({int(use.sum())} rows)")
            continue
        y, g, b = rows[name][use], rows["game"][use], band[use]
        pj = FJ.predict(j.head(name), X[use])
        pu = FJ.predict(base[name]["units"], U[use])
        pp = FJ.predict(base[name]["patterns"], P[use])
        pb = np.array(base[name]["band_rate"])[b]
        ll = {n: FJ.logloss(p, y) for n, p in (("judge", pj), ("units", pu), ("patterns", pp), ("base rate", pb))}
        say(f"SCORE {name} (the head used: {j.info[name]['use']}; CONFIRM rows {len(y)}, positives {int(y.sum())}, {100 * y.mean():.2f} pct): log loss judge {ll['judge']:.4f}, units {ll['units']:.4f}, patterns {ll['patterns']:.4f}, base rate by band {ll['base rate']:.4f}; "
            f"Brier judge {FJ.brier(pj, y):.4f} base rate {FJ.brier(pb, y):.4f}; AUC judge {FJ.auc(pj, y):.3f}, units {FJ.auc(pu, y):.3f}, patterns {FJ.auc(pp, y):.3f}")
        say(f"CALIBRATION {name}: (mean predicted, observed share, rows) " + "; ".join(f"({a:.3f}, {o:.3f}, {n})" for a, o, n in FJ.calibration(pj, y, 5)))
        for bi, (bn, _lo, _hi) in enumerate(BANDS):
            m = b == bi
            if int(m.sum()) < 30 or len(np.unique(y[m])) < 2:
                say(f"SCORE {name} band {bn}: {int(m.sum())} rows, positives {int(y[m].sum())}: too thin")
                continue
            gm, ym = g[m], y[m]
            res = {}
            for nm, other in (("units", pu), ("patterns", pp)):
                pj_m, po_m = pj[m], other[m]

                def fn(ix, pj_m=pj_m, po_m=po_m):
                    yy = ym[ix]
                    if len(np.unique(yy)) < 2:
                        return float("nan")
                    return FJ.auc(pj_m[ix], yy) - FJ.auc(po_m[ix], yy)
                res[nm] = boot(gm, fn, 200)
            say(f"SCORE {name} band {bn}: rows {int(m.sum())}, positives {int(ym.sum())}; AUC judge {FJ.auc(pj[m], ym):.3f}; log-loss gain over the band's base rate: judge {FJ.logloss(pb[m], ym) - FJ.logloss(pj[m], ym):+.4f}, units {FJ.logloss(pb[m], ym) - FJ.logloss(pu[m], ym):+.4f}, "
                f"patterns {FJ.logloss(pb[m], ym) - FJ.logloss(pp[m], ym):+.4f}; paired AUC band, judge less units {res['units'][0]:+.3f} [{res['units'][1]:+.3f}, {res['units'][2]:+.3f}], judge less patterns {res['patterns'][0]:+.3f} [{res['patterns'][1]:+.3f}, {res['patterns'][2]:+.3f}]")
        table.append((name, FJ.auc(pj, y), FJ.auc(pu, y), FJ.auc(pp, y)))
    say("SUMMARY AUC on CONFIRM (judge / units / patterns): " + "; ".join(f"{n} {a:.3f}/{b:.3f}/{c:.3f}" for n, a, b, c in table))
    vocab = saved["vocab"]
    for k, p in sorted(j.pair.items()):
        pr = FJ.pair_rows(recs, rows, vocab, k, seed=SEED)
        tr = rows["part"][pr["row"]] == 0
        te = ~tr
        y = pr["label"][te]
        if len(np.unique(y)) < 2:
            say(f"PAIRWISE k {k}: CONFIRM pairs {int(te.sum())}, one class only")
            continue
        F = FJ.FutureJudge.pair_features(X[pr["row"][te]], pr["target"][te], len(vocab))
        pm = p["tree"]["m"].predict_proba(F)[:, 1]
        freq = np.array([pr["label"][tr][pr["target"][tr] == t].mean() if (pr["target"][tr] == t).any() else pr["label"][tr].mean() for t in range(len(vocab))])[pr["target"][te]]
        say(f"PAIRWISE k {k} on CONFIRM: pairs {int(te.sum())}, positives {int(y.sum())}; AUC of the pairwise judge {FJ.auc(pm, y):.3f}, of the target's own frequency alone {FJ.auc(freq, y):.3f}; log loss {FJ.logloss(pm, y):.4f} against the frequency's {FJ.logloss(np.clip(freq, 0.001, 0.999), y):.4f}; "
            f"calibration (mean predicted, observed, rows) " + "; ".join(f"({a:.3f}, {o:.3f}, {n})" for a, o, n in FJ.calibration(pm, y, 4)))


# ------------------------------------------------------------------------------------------------------------------------------ step three: the test on human moves

_W = None


def _init_human() -> None:
    global _W
    from catspace.incubator.analyst.discovery import future_imagine as FI
    from catspace.incubator.analyst.discovery import symbols_play as SP
    _W = {"im": FI.Imaginer(), "sp": SP.reader82()}


def _human_job(row: dict) -> dict:
    """For one human tie position: the pattern pick (78.82's reasons), which tied moves take a unit, the first moves of the mover's goals at the position and, when a goal of the other side holds, for each tied move whether it still holds after it."""
    import cs_engine as CE
    from catspace.incubator.analyst.discovery import future_goals as FG
    im, sp = _W["im"], _W["sp"]
    moves, tied = row["moves"], row["tied"]
    pick, _d = sp.pick(moves, tied, "reasons")
    takes, _k = sp.controls_of(moves, tied)
    out = {"id": row["id"], "pattern": pick, "takes": {m: bool(v) for m, v in takes.items()}, "mover_first": [], "other_goal": False, "survive": {}}
    text = CE.state(im.variant, None, list(moves))[0]
    q = im.question(text)
    if q is not None:
        c, names = q
        info = FG.goals_of_list(c, "mover", im.record)
        out["mover_first"] = sorted({w for g in info["goals"] for w in [FI_word(g["first"], names["places"])] if w})
        out["mover_goal"] = bool(info["goals"])
    p = CE.passed(im.variant, text, [])
    if p is not None:
        q2 = im.question(p[0])
    else:
        q2 = None
    if q2 is not None:
        # the other side's goals at this position: the side to move passes, the other side moves (the mover's list there)
        info2 = FG.goals_of_list(q2[0], "mover", im.record)
        out["other_goal"] = bool(info2["goals"])
    if out["other_goal"]:
        for m in tied:
            t1 = CE.state(im.variant, None, list(moves) + [m])[0]
            r = im.question(t1)
            if r is None:
                out["survive"][m] = None
                continue
            i3 = FG.goals_of_list(r[0], "mover", im.record)
            out["survive"][m] = bool(i3["goals"])
    return out


def FI_word(pair, places: list) -> "str | None":
    from catspace.incubator.analyst.discovery import future_imagine as FI
    return FI.word_of(pair, places)


LIMIT = 0                                                    # a dry run: the human rows and the explicit run's sample are cut to about this many (0: all)


def human_rows() -> list:
    out = []
    for tag, d, t in (("A", "symbols_78_80", "full"), ("B", "symbols_78_89", "full")):
        for r in SM.jsonl(Path(paths.experiment(d, t, "sample.jsonl"))):
            if len(r["tied"]) >= 2 and r["played"] in r["tied"]:
                r = dict(r, sample=tag, id=f"{tag}{r['id']}")
                out.append(r)
    if LIMIT:
        out = out[::max(1, len(out) // LIMIT)]
    return out


def setting_list(judge: FJ.FutureJudge) -> list:
    return [(w, k) for w in ("goal", "either", "win") for k in FJ.KS if judge.head({"goal": "G", "either": "GW", "win": "W"}[w] + str(k)) is not None]


def candidate_chances(judge: FJ.FutureJudge, rows: list, settings: list, inputs: FJ.Inputs, say) -> dict:
    """For every tied move of every row: the chance for the side that just moved (ours) and for the side to move (theirs), per setting: {setting: (ours, theirs)} over the candidates in row order, then tied order."""
    wires = [(None, list(r["moves"]) + [m]) for r in rows for m in r["tied"]]
    ours = {s: [] for s in settings}
    theirs = {s: [] for s in settings}
    t0 = time.perf_counter()
    threads = max(1, (os.cpu_count() or 2) - 1)
    for a in range(0, len(wires), 2000):
        part = wires[a:a + 2000]
        inp = inputs.inputs_of(part, [[len(m)] for _s, m in part], threads)
        xs, xt = FJ.side_matrix(inp, False), FJ.side_matrix(inp, True)
        for s in settings:
            ours[s].append(judge.chance(xs, s[1], s[0]))
            theirs[s].append(judge.chance(xt, s[1], s[0]))
        if (a // 2000) % 5 == 0:
            say(f"human: candidates {min(a + 2000, len(wires))} of {len(wires)}; {time.perf_counter() - t0:.0f} s")
    return {s: (np.concatenate(ours[s]), np.concatenate(theirs[s])) for s in settings}


def judge_picks(rows: list, ours: np.ndarray, theirs: np.ndarray) -> list:
    out, i = [], 0
    for r in rows:
        n = len(r["tied"])
        val = np.round(ours[i:i + n] - theirs[i:i + n], 6)
        best = np.flatnonzero(val == val.max())
        out.append((r["tied"][0] if (0 in best or len(best) == n) else r["tied"][int(best[0])], len(best) == n))
        i += n
    return out


def cmd_human(root: Path, say) -> None:
    import multiprocessing
    from catspace.incubator.analyst.discovery import graph_layer1 as GL
    from catspace.incubator.analyst.discovery import human_heads_data as HD
    saved = pickle.load(open(root / "judge.pkl", "rb"))
    judge = saved["judge"]
    rows = human_rows()
    settings = setting_list(judge)
    inputs = FJ.Inputs(GL.model(HD.GAME, HD.VARIANT), HD.statements_chunked)
    f = root / "human_chances.pkl"
    if f.exists():
        ch = pickle.load(open(f, "rb"))
    else:
        ch = candidate_chances(judge, rows, settings, inputs, say)
        pickle.dump(ch, open(f, "wb"))
    find_ix = [i for i, r in enumerate(rows) if r["part"] == "find"]
    conf_ix = [i for i, r in enumerate(rows) if r["part"] == "confirm"]
    offs = np.cumsum([0] + [len(r["tied"]) for r in rows])

    def sub(ix: list, s: tuple) -> tuple:
        cand = np.concatenate([np.arange(offs[i], offs[i + 1]) for i in ix])
        return ch[s][0][cand], ch[s][1][cand]
    say(f"HUMAN MOVES: tie positions of both human samples with the move played among the equal moves: {len(rows)} (FIND {len(find_ix)}, CONFIRM {len(conf_ix)}); settings tried on FIND: {settings}")
    best, best_hit = None, -1.0
    for s in settings:
        o, t = sub(find_ix, s)
        pk = judge_picks([rows[i] for i in find_ix], o, t)
        hit = float(np.mean([p == rows[i]["played"] for (p, _sil), i in zip(pk, find_ix)]))
        first = float(np.mean([rows[i]["tied"][0] == rows[i]["played"] for i in find_ix]))
        say(f"HUMAN FIND setting {s}: the judge's pick is the move played in {100 * hit:.2f} pct; the search's first move {100 * first:.2f} pct; silent (all equal) {100 * float(np.mean([sil for _p, sil in pk])):.1f} pct")
        if hit > best_hit:
            best, best_hit = s, hit
    say(f"HUMAN: the setting chosen on FIND (the highest agreement with the move played): {best} ({100 * best_hit:.2f} pct); CONFIRM is read once with it")
    o, t = sub(conf_ix, best)
    crow = [rows[i] for i in conf_ix]
    pk = judge_picks(crow, o, t)
    jobs_f = root / "human_jobs.json"
    if jobs_f.exists():
        jobs = json.loads(jobs_f.read_text())
    else:
        t0 = time.perf_counter()
        with multiprocessing.get_context("spawn").Pool(max(1, (os.cpu_count() or 2) - 1), initializer=_init_human) as pool:
            res = pool.map(_human_job, crow, chunksize=20)
        jobs = {r["id"]: r for r in res}
        jobs_f.write_text(json.dumps(jobs))
        say(f"HUMAN: the pattern pick, the takings and the goals read at {len(crow)} CONFIRM positions in {time.perf_counter() - t0:.0f} s")
    played = np.array([r["played"] for r in crow])
    judge_hit = np.array([p == r["played"] for (p, _s), r in zip(pk, crow)], float)
    first_hit = np.array([r["tied"][0] == r["played"] for r in crow], float)
    pat_hit = np.array([jobs[r["id"]]["pattern"] == r["played"] for r in crow], float)
    rnd = np.array([1.0 / len(r["tied"]) for r in crow])
    game = np.array([r["game"] for r in crow])
    units = np.array([r["units"] for r in crow])
    quiet = np.array([not any(jobs[r["id"]]["takes"].values()) for r in crow])
    silent = np.array([s for _p, s in pk])
    bi = band_index(units)
    say(f"HUMAN CONFIRM (setting {best}): agreement with the move played: judge {100 * judge_hit.mean():.2f} pct, search's first move {100 * first_hit.mean():.2f} pct, pattern pick (78.81's reasons) {100 * pat_hit.mean():.2f} pct, a random pick among the equal moves {100 * rnd.mean():.2f} pct; the judge silent (all equal) in {100 * silent.mean():.1f} pct")
    groups = [("all", np.ones(len(crow), bool))] + [(n, bi == i) for i, (n, _l, _h) in enumerate(BANDS)] + [("quiet (no equal move takes a unit)", quiet), ("not quiet", ~quiet)] + [(f"{n} and quiet", (bi == i) & quiet) for i, (n, _l, _h) in enumerate(BANDS)]
    for name, m in groups:
        if int(m.sum()) < 30:
            say(f"HUMAN paired {name}: {int(m.sum())} positions: too few")
            continue
        a = SM.paired(judge_hit[m], first_hit[m], game[m], SEED)
        b = SM.paired(judge_hit[m], pat_hit[m], game[m], SEED + 1)
        c = SM.paired(judge_hit[m], rnd[m], game[m], SEED + 2)
        say(f"HUMAN paired {name}: positions {int(m.sum())}; agreement judge {100 * judge_hit[m].mean():.2f}, first {100 * first_hit[m].mean():.2f}, pattern {100 * pat_hit[m].mean():.2f}; judge less first {100 * a[0]:+.2f} pct points [{100 * a[1]:+.2f}, {100 * a[2]:+.2f}], judge less pattern {100 * b[0]:+.2f} [{100 * b[1]:+.2f}, {100 * b[2]:+.2f}], judge less a random pick {100 * c[0]:+.2f} [{100 * c[1]:+.2f}, {100 * c[2]:+.2f}] (90 pct bands, games resampled)")
    # where a goal of the mover holds and its first move is among the equal moves
    sel = [(r, jobs[r["id"]]) for r in crow if jobs[r["id"]].get("mover_first") and set(jobs[r["id"]]["mover_first"]) & set(r["tied"])]
    say(f"HUMAN GOAL OF THE MOVER holds and its first move is among the equal moves: {len(sel)} positions; the move played is that first move in {sum(r['played'] in j['mover_first'] for r, j in sel)}; the judge's pick is in {sum(p in j['mover_first'] for (p, _s), (r, j) in ((pk[crow.index(r)], (r, j)) for r, j in sel))}; "
        f"the search's first move is in {sum(r['tied'][0] in j['mover_first'] for r, j in sel)}; the pattern pick is in {sum(j['pattern'] in j['mover_first'] for r, j in sel)}; a random equal move would be in {sum(len(set(j['mover_first']) & set(r['tied'])) / len(r['tied']) for r, j in sel):.1f} of them")
    rep = [(r, jobs[r["id"]]) for r in crow if jobs[r["id"]]["other_goal"] and any(v is False for v in jobs[r["id"]]["survive"].values())]
    allg = [(r, jobs[r["id"]]) for r in crow if jobs[r["id"]]["other_goal"]]
    say(f"HUMAN REPLY TO THEIR GOAL: positions where a goal of the other side holds {len(allg)}, of them with an equal move after which it does not survive {len(rep)}; the move played is such a breaking move in {sum(j['survive'].get(r['played']) is False for r, j in rep)}; the judge's pick in {sum(j['survive'].get(pk[crow.index(r)][0]) is False for r, j in rep)}; the first move in {sum(j['survive'].get(r['tied'][0]) is False for r, j in rep)}; the pattern pick in {sum(j['survive'].get(j['pattern']) is False for r, j in rep)}")


# ------------------------------------------------------------------------------------------------------------------------------ the explicit run of own moves of own moves against the judge

BUDGETS = (8, 32)
PER_BAND = 40


def _init_explicit() -> None:
    global _W
    from catspace.incubator.analyst.discovery import future_imagine as FI
    _W = {"im": FI.Imaginer()}


def _explicit_job(job: dict) -> dict:
    im = _W["im"]
    out = {"key": job["key"], "side": job["side"], "runs": {}}
    for n in BUDGETS:
        r = im.imagine(job["moves"], job["side"], n)
        found = [g for g in r["goals"] if g["run"]]
        d = {"questions": r["questions"], "wall": r["wall"], "refused": r["refused"], "found": len(found), "root_goal": any(not g["run"] for g in r["goals"])}
        if found:
            g = found[0]
            d["goal"] = {"run": g["run"], "kind": g["kind"], "stake": g["stake"], "shape": g["shape"]}
            v = im.veto(job["moves"], job["side"], g)
            d["veto"] = {"holds": v["holds"], "broken_by": v["broken_by"], "line": v["line"]}
        out["runs"][str(n)] = d
    return out


def quiet_sample(recs: list, per_band: int) -> list:
    """Quiet positions of the CONFIRM games (no confirmed threat and no forced end for either side by the threat list): a fixed number per band of units, in the order of a hash of (game, ply)."""
    import hashlib
    out = []
    for bi, (_n, lo, hi) in enumerate(BANDS):
        cand = [(r, q) for r in recs if r["part"] == "confirm" for q in r["positions"]
                if lo <= q["units"] <= hi and q["mover_g"]["threats"] == 0 and q["other_g"]["threats"] == 0 and not q["mover_g"]["forced"] and not q["other_g"]["forced"]]
        cand.sort(key=lambda t: hashlib.md5(f"explicit94:{t[0]['ord']}:{t[1]['ply']}".encode()).hexdigest())
        out += [(r, q) for r, q in cand[:per_band]]
    return out


def cmd_explicit(root: Path, say) -> None:
    import multiprocessing
    recs = load_recs(root)
    games = games_by_ordinal()
    sample = quiet_sample(recs, LIMIT or PER_BAND)
    jobs = []
    for r, q in sample:
        for side in ("mover", "other"):
            jobs.append({"key": f"{r['ord']}:{q['ply']}", "side": side, "moves": games[r["ord"]]["moves"][:q["ply"]]})
    f = root / "explicit.json"
    if f.exists():
        res = json.loads(f.read_text())
    else:
        t0 = time.perf_counter()
        with multiprocessing.get_context("spawn").Pool(max(1, (os.cpu_count() or 2) - 1), initializer=_init_explicit) as pool:
            res = pool.map(_explicit_job, jobs, chunksize=2)
        f.write_text(json.dumps(res))
        say(f"EXPLICIT: the explicit run of own moves on {len(sample)} quiet positions of CONFIRM games ({PER_BAND} a band; no confirmed threat and no forced end for either side), both sides, budgets {BUDGETS} questions, in {time.perf_counter() - t0:.0f} s wall")
    by = {(x["key"], x["side"]): x for x in res}
    rows = FJ.rows_from_labels(recs)
    inp = build_inputs(root, recs, say)
    X, _U, _P = side_inputs(inp, rows)
    saved = pickle.load(open(root / "judge.pkl", "rb"))
    j = saved["judge"]
    ridx = {(int(o), int(p), int(s)): i for i, (o, p, s) in enumerate(zip(rows["ord"], rows["ply"], rows["mover_side"]))}
    for n in BUDGETS:
        for side in ("mover", "other"):
            found = [by[(f"{r['ord']}:{q['ply']}", side)]["runs"][str(n)] for r, q in sample]
            ks = [d["found"] > 0 for d in found]
            q_per = np.mean([d["questions"] for d in found])
            ms = 1000 * np.mean([d["wall"] for d in found])
            lens = [len(d["goal"]["run"]) for d in found if "goal" in d]
            vt = [d["veto"]["holds"] for d in found if "veto" in d]
            say(f"EXPLICIT imagine for {side}, budget {n}: positions {len(found)}; a goal found after at least one own move in {sum(ks)} ({100 * sum(ks) / len(ks):.1f} pct); mean questions {q_per:.1f}; {ms:.0f} ms a position (one thread); run lengths {dict(Counter(lens))}; "
                f"of the goals found the veto line (the other side replies by the engine's search) leaves a goal in {sum(vt)} of {len(vt)}; kinds {dict(Counter(d['goal']['kind'] for d in found if 'goal' in d))}; "
                f"by band: " + "; ".join(f"{nm} {sum(k for (r, q), k in zip(sample, ks) if lo <= q['units'] <= hi)} of {sum(1 for r, q in sample if lo <= q['units'] <= hi)}" for nm, lo, hi in BANDS))
    # the judge's chance against the explicit run, and both against what happened
    for what_k in (("goal", 8), ("goal", 16), ("either", 8)):
        w, k = what_k
        name = {"goal": "G", "either": "GW"}[w] + str(k)
        if j.head(name) is None:
            say(f"EXPLICIT the judge's {name} head is not fitted")
            continue
        for n in BUDGETS:
            for side in ("mover", "other"):
                p, imag, real = [], [], []
                for r, q in sample:
                    i = ridx[(r["ord"], q["ply"], 1 if side == "mover" else 0)]
                    if rows[name][i] < 0:
                        continue
                    p.append(float(j.chance(X[i:i + 1], k, w)[0]))
                    imag.append(int(by[(f"{r['ord']}:{q['ply']}", side)]["runs"][str(n)]["found"] > 0))
                    real.append(int(rows[f"G{k}"][i] == 1 if w == "goal" else rows[name][i] == 1))
                p, imag, real = np.array(p), np.array(imag), np.array(real)
                say(f"EXPLICIT {side}, budget {n}, the judge's {name} against the explicit run: positions {len(p)}; mean chance where the run found a goal {p[imag == 1].mean() if imag.any() else float('nan'):.4f}, where it did not {p[imag == 0].mean():.4f}; "
                    f"AUC of the judge's chance for 'the run finds a goal' {FJ.auc(p, imag):.3f}; what happened within {k} plies ({'a goal' if w == 'goal' else 'a goal or a winning point'}): in {int(real.sum())} of {len(real)}; "
                    f"AUC for it: the judge {FJ.auc(p, real):.3f}, the explicit run {FJ.auc(imag.astype(float), real):.3f}; the share that happened where the run found one {real[imag == 1].mean() if imag.any() else float('nan'):.3f}, where not {real[imag == 0].mean():.3f}")


# ------------------------------------------------------------------------------------------------------------------------------ the thirty positions

CASE_RULE = ("THE THIRTY POSITIONS (plan 78.94), all of CONFIRM games, none fitted on: ten where a goal of one side holds by our threat list (in the order of a hash, at most four of one band), ten where the oracle calls the side to move winning and the frame of its best line is not "
             "two threats at once, ten where the judge's chance of a goal or a winning point of the side to move within eight plies is among the highest of the positions where none holds yet. For each: the oracle's class and best line, our reading of its frame, the goals, the judge's chances for both "
             "sides, and what happened in the game.")


def cmd_cases(root: Path, say) -> None:
    import hashlib
    recs = load_recs(root)
    games = games_by_ordinal()
    rows = FJ.rows_from_labels(recs)
    inp = build_inputs(root, recs, say)
    X, _U, _P = side_inputs(inp, rows)
    j = pickle.load(open(root / "judge.pkl", "rb"))["judge"]
    ridx = {(int(o), int(p), int(s)): i for i, (o, p, s) in enumerate(zip(rows["ord"], rows["ply"], rows["mover_side"]))}
    conf = [(r, q) for r in recs if r["part"] == "confirm" for q in r["positions"]]
    key = lambda t, salt: hashlib.md5(f"{salt}:{t[0]['ord']}:{t[1]['ply']}".encode()).hexdigest()
    goals = sorted((t for t in conf if t[1]["mover_g"]["goal"] or t[1]["other_g"]["goal"]), key=lambda t: key(t, "g94"))
    taken, per = [], Counter()
    for t in goals:
        b = band_index(np.array([t[1]["units"]]))[0]
        if per[b] < 4 and len(taken) < 10:
            taken.append(t)
            per[b] += 1
    for t in goals:
        if len(taken) < 10 and t not in taken:
            taken.append(t)
    chosen = [("a goal holds", t) for t in taken]
    win = sorted((t for t in conf if t[1]["cls"] == 1 and "frame" in t[1] and not frame_type(t[1]).startswith("goal")), key=lambda t: key(t, "w94"))
    chosen += [("the oracle calls the side to move winning; its frame is not two threats", t) for t in win[:10]]
    name = "GW8"
    top = []
    if j.head(name) is not None:
        free = [t for t in conf if not t[1]["mover_g"]["goal"] and t[1]["cls"] != 1]
        pm = []
        for r, q in free:
            i = ridx[(r["ord"], q["ply"], 1)]
            pm.append(float(j.chance(X[i:i + 1], 8, "either")[0]) if rows[name][i] >= 0 else -1.0)
        order = np.argsort(-np.array(pm))
        pool = [free[int(o)] for o in order[: max(40, len(free) // 10)]]
        pool.sort(key=lambda t: key(t, "t94"))
        top = pool[:10]
    chosen += [("the judge's chance is among the highest where nothing holds yet", t) for t in top]
    cases = []
    heads = [(w, k) for w, k in (("goal", 8), ("goal", 16), ("either", 8), ("win", 8), ("lose", 8)) if j.head({"goal": "G", "either": "GW", "win": "W", "lose": "L"}[w] + str(k)) is not None]
    for why, (r, q) in chosen:
        im, io = ridx[(r["ord"], q["ply"], 1)], ridx[(r["ord"], q["ply"], 0)]
        ch = {f"{w}{k}": [float(j.chance(X[i:i + 1], k, w)[0]) for i in (im, io)] for w, k in heads}
        mv = games[r["ord"]]["moves"]
        cases.append({"id": f"c{r['ord']}:{q['ply']}", "why": why, "ord": r["ord"], "ply": q["ply"], "moves": mv[:q["ply"]], "next_moves": mv[q["ply"]:q["ply"] + 6], "units": q["units"], "band": BANDS[band_index(np.array([q["units"]]))[0]][0],
                      "cls": q["cls"], "cp": q["cp"], "forced_in": q["forced_in"], "pv": q["pv"], "mover_g": q["mover_g"], "other_g": q["other_g"], "frame": q.get("frame"), "frame_type": frame_type(q) if "frame" in q else None, "seen": q.get("seen"),
                      "chances": ch, "later": {f"{w}{k}": [int(rows[f"{w}{k}"][i]) for i in (im, io)] for w in ("G", "W", "GW") for k in (8, 16)}, "now": {"G": [int(rows["now_G"][im]), int(rows["now_G"][io])]},
                      "result_mover": int(rows["y"][im]), "result_other": int(rows["y"][io])})
    (root / "cases94.json").write_text(json.dumps({"rule": CASE_RULE, "cases": cases}))
    say(f"CASES: {len(cases)} positions written to cases94.json ({Counter(c['why'] for c in cases)})")


def main(argv: "list | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["counts", "fit", "score", "explicit", "human", "cases"])
    ap.add_argument("--tag", default="short")
    ap.add_argument("--limit", type=int, default=0, help="a dry run on a small part of the human rows and of the explicit run's sample")
    a = ap.parse_args(argv)
    global LIMIT
    LIMIT = a.limit
    root = FL.out_root(a.tag)
    t0 = time.perf_counter()
    say = FL.Say(root, a.cmd)
    {"counts": cmd_counts, "fit": cmd_fit, "score": cmd_score, "explicit": cmd_explicit, "human": cmd_human, "cases": cmd_cases}[a.cmd](root, say)
    say(f"wall {time.perf_counter() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
