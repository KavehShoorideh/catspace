"""PLAN 78.59 C (ARRIVAL): THE ARRIVAL HEAD FROM HUMAN GAMES. The target is that of 78.49 (forcing_arrival_measure): at a position where neither side holds a CONFIRMED threat move (the forcing read of the engine,
`SheetModel.forcing_games`, fuse 5000), a CONFIRMED threat move of the side to move arrives within h plies, less the other side's. A measurement in a sandbox; nothing is wired into any player; no engine
evaluation of positions enters (the forcing read is the rules' own read, as in 78.48); game-free (raw symbols only).

THE SAMPLE: the arrival games of human_heads_data (the first ARRIVAL_PER_SHARD of each shard's hash order); the reads of every position of each game are made in chunks and kept (counts of the verdicts only) in a
resumable file; the cost per position is printed. The split is the games' own (7 in 10 fit, by hash). ROWS: the quiet positions (by the forcing read) from ply OPEN_PLIES on whose target exists at the horizon.
INPUTS: K = kinds ours less theirs; S = the 26 statements ours less theirs; C = the kept concepts on every CONCEPT_EVERY-th such row; R = mean rating, rating difference, ply (the ratings, with or without).
FIT: the sparse linear fit of 78.49 (subgoal_fit_measure.fit_sparse: L1, the penalty by 5-fold cross-validation grouped by game, on the fit games). TEST: rank correlation with the target on the test games with a 90 pct
band over games (subgoal_fit_measure.band_lines), beside the count difference of 78.49 and the fit shuffled within the game; the same by band of rating. BESIDE IT: the same fits on the self-play games of 78.49 (even game
ids fit, odd test), and each fit applied to the other source's test rows.

  .venv/bin/python -m catspace.incubator.analyst.discovery.human_heads_arrival --data D --out-dir O [--budget-seconds S] [--threads K]
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import time
from pathlib import Path

import numpy as np

from catspace.incubator.analyst.discovery import forcing_arrival_measure as FA
from catspace.incubator.analyst.discovery import forcing_opportunity_measure as FO
from catspace.incubator.analyst.discovery import graph_layer1 as GL
from catspace.incubator.analyst.discovery import human_heads_data as HD
from catspace.incubator.analyst.discovery import subgoal_fit_measure as SF
from catspace.io import paths

FUSE = 5000
CHUNK = 100
HORIZONS = FA.HORIZONS
H = 8
SEED = 5959


def light(read):
    return None if read is None else {"threats": [[t[0]] for t in read["threats"]]}


def read_all(games: list, threads: int, ckpt: Path, budget: float, say) -> "dict | None":
    """{game index: plies (light reads)}; None when the time budget ended first (the file holds what was read; run again)."""
    done: dict = {}
    if ckpt.exists():
        for line in ckpt.read_text().splitlines():
            if line.strip():
                x = json.loads(line)
                done[x["game"]] = x["plies"]
    todo = [i for i in range(len(games)) if i not in done]
    t0 = time.perf_counter()
    npos = 0
    for s in range(0, len(todo), CHUNK):
        part = todo[s:s + CHUNK]
        res = FO.read_chunk(HD.VARIANT, [{"start": None, "moves": games[i]["moves"]} for i in part], threads, FUSE)
        with open(ckpt, "a") as f:
            for i, plies in zip(part, res):
                lp = [None if p is None else {"ours": light(p["ours"]), "theirs": light(p["theirs"])} for p in plies]
                done[i] = lp
                npos += len(plies)
                f.write(json.dumps({"game": i, "plies": lp}) + "\n")
        el = time.perf_counter() - t0
        say(f"progress forcing reads: {len(done)} of {len(games)} games, {npos} positions in {el:.0f} s since start of this run ({el / npos:.4f} s per position over {threads} threads)")
        if budget and el > budget and len(done) < len(games):
            return None
    return done


def tables_of(games: list, threads: int) -> "tuple[list, list, list]":
    """(kind names, per game kinds diff (positions, kinds), per game statements diff (positions, 26)), all positions, for the quiet-by-forcing rows."""
    model = GL.model(HD.GAME, HD.VARIANT)
    goal = None
    from catspace.incubator.analyst.discovery import graph_layer2_measure as G2M
    goal = G2M.goal_kind_of(HD.VARIANT)
    names, raw = model.position_table([(None, g["moves"]) for g in games], threads)
    nk = len(names)
    flat = np.frombuffer(raw, dtype=np.int16).astype(np.int32)
    w = 2 + 2 * nk
    use = [k for k, nm in enumerate(names) if nm != goal]
    kd, off = [], 0
    for g in games:
        n = len(g["moves"]) + 1
        t = flat[off:off + n * w].reshape(n, w)
        off += n * w
        kd.append((t[:, 2:2 + nk] - t[:, 2 + nk:2 + 2 * nk])[:, use].astype(np.float32))
    _snames, stats = HD.statements_chunked(games, threads)
    return [names[k] for k in use], kd, [SF.placed_diff(s) for s in stats]


def human_table(games: list, reads: dict, kd: list, sd: list) -> dict:
    cols: dict = {k: [] for k in ("game", "ply", "K", "S", "R", "count", "fit", "band")}
    for h in HORIZONS:
        cols[f"arr{h}"] = []
    for i, g in enumerate(games):
        plies = reads[i]
        own, oth = FA.counts_of_game(plies)
        for t in FA.quiet_indices(plies, own, oth):
            if t < HD.OPEN_PLIES or t >= len(sd[i]) or np.isnan(sd[i][t]).any():
                continue
            first = t % 2 == 0
            rm, ro = (g["welo"], g["belo"]) if first else (g["belo"], g["welo"])
            tgs = FA.arrival_targets(own, len(plies), t)
            cols["game"].append(i); cols["ply"].append(t); cols["K"].append(kd[i][t]); cols["S"].append(sd[i][t])
            cols["R"].append([(rm + ro) / 2, rm - ro, t]); cols["count"].append(own[t][0] - oth[t][0]); cols["fit"].append(g["fit"]); cols["band"].append(g["band"])
            for h, tg in zip(HORIZONS, tgs):
                cols[f"arr{h}"].append(np.nan if tg is None else tg[0] - tg[1])
    out = {k: np.asarray(v, dtype=float) for k, v in cols.items() if k not in ("K", "S", "R")}
    for k in ("K", "S", "R"):
        out[k] = np.asarray(cols[k], dtype=float).reshape(len(cols["game"]), -1)
    out["fit"] = out["fit"].astype(bool)
    return out


def selfplay_table(threads: int) -> dict:
    """The rows of the 78.49 test: the quiet rows (by the forcing read) of the 500 self-play games with the kinds added; even ids fit."""
    records = sorted(FO.read_games(str(Path(paths.experiment("opportunity_78_44", "games_s5016_n500")))), key=lambda r: r["game"])
    reads = {}
    with open(paths.experiment("forcing_opportunity_78_48", "games_s5016_n500", "reads_fuse5000.jsonl")) as f:
        for line in f:
            g = json.loads(line)
            reads[g["game"]] = g["plies"]
    z = np.load(paths.experiment("forcing_opportunity_78_48", "games_s5016_n500", "opp_table_n500.npz"))
    a_opp = {k: z[k] for k in z.files if k != "n_games"}
    zs = np.load(paths.experiment("subgoal_fit_78_49", "full", "statements_n500.npz"))
    stats = [zs[f"g{i}"] for i in range(len(records))]
    t = SF.table_of(records, reads, stats, a_opp)
    model = GL.model(HD.GAME, HD.VARIANT)
    from catspace.incubator.analyst.discovery import graph_layer2_measure as G2M
    goal = G2M.goal_kind_of(HD.VARIANT)
    names, raw = model.position_table([(r.get("start"), list(r["moves"])) for r in records], threads)
    nk = len(names)
    flat = np.frombuffer(raw, dtype=np.int16).astype(np.int32)
    w = 2 + 2 * nk
    use = [k for k, nm in enumerate(names) if nm != goal]
    off, kd = 0, []
    for r in records:
        n = len(r["moves"]) + 1
        tt = flat[off:off + n * w].reshape(n, w)
        off += n * w
        kd.append((tt[:, 2:2 + nk] - tt[:, 2 + nk:2 + 2 * nk])[:, use].astype(np.float32))
    t["K"] = np.array([kd[int(g)][int(p)] for g, p in zip(t["game"], t["ply"])], dtype=float).reshape(len(t["game"]), -1)
    t["kind_names"] = [names[k] for k in use]
    t["fit"] = t["id"] % 2 == 0
    return t


def fit_pred(tab: dict, feats: np.ndarray, h: int, sel_fit: np.ndarray, jobs: int):
    y = tab[f"arr{h}"]
    ok = sel_fit & ~np.isnan(y)
    rows = np.flatnonzero(ok)
    return SF.fit_sparse(feats[rows], y[rows], tab["game"][rows], jobs=jobs)


def corr_lines(label: str, a: dict, draws: int, readings: list, say) -> dict:
    lines, vals = SF.band_lines(label, a, draws, readings)
    for ln in lines:
        say(ln.replace("OBS subgoal ", ""))
    return vals


def main(argv: "list | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", required=True, help="the directory of human_heads_data (games.jsonl, rows.npz)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--budget-seconds", type=float, default=0.0, help="stop the forcing reads after this long (rerun resumes)")
    ap.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--draws", type=int, default=300)
    ap.add_argument("--concept-every", type=int, default=3)
    ap.add_argument("--only-games", type=int, default=0, help="the first N arrival games only")
    a = ap.parse_args(argv)
    t0 = time.perf_counter()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    lines: list = []

    def say(msg: str) -> None:
        lines.append(f"OBS human arrival: {msg}")
        print(lines[-1], flush=True)

    allg = [json.loads(x) for x in (Path(a.data) / "games.jsonl").read_text().splitlines() if x.strip()]
    games = [g for g in allg if g["arrival"]]
    if a.only_games:
        games = games[:a.only_games]
    say(f"arrival sample: {len(games)} games, {sum(len(g['moves']) for g in games)} plies; fit {sum(g['fit'] for g in games)}, test {sum(not g['fit'] for g in games)}")
    reads = read_all(games, a.threads, out / "reads_light.jsonl", a.budget_seconds, say)
    if reads is None:
        say("the time budget ended first; run again to resume")
        (out / "arrival_obs.txt").write_text("\n".join(lines) + "\n")
        return 2
    say(f"forcing reads of {len(games)} games done; wall so far {time.perf_counter() - t0:.1f} s")
    kinds, kd, sd = tables_of(games, a.threads)
    tab = human_table(games, reads, kd, sd)
    fit = tab["fit"]
    say(f"quiet rows by the forcing read from ply {HD.OPEN_PLIES} on: {len(tab['game'])}; fit {int(fit.sum())} rows, test {int((~fit).sum())} rows ({len(np.unique(tab['game'][~fit]))} test games)")
    for h in HORIZONS:
        yv = tab[f"arr{h}"]
        say(f"target h={h}: rows with the target {int((~np.isnan(yv)).sum())}; mean over the test rows {np.nanmean(yv[~fit]):+.4f}")
    # the concepts on every nth row
    from catspace.incubator.analyst.discovery import frames_measure as OLD
    from catspace.incubator.analyst.discovery import graph_layer2 as LAYER2
    from catspace.incubator.analyst.discovery import graph_layer2_measure as G2M
    model = GL.model(HD.GAME, HD.VARIANT)
    goal = G2M.goal_kind_of(HD.VARIANT)
    first_of = np.searchsorted(tab["game"], tab["game"])
    cmask = ((np.arange(len(tab["game"])) - first_of) % a.concept_every) == 0
    sel = np.flatnonzero(cmask)
    t1 = time.perf_counter()
    picked = OLD.pick_concepts(Path(paths.experiment("unit_fates_78_54", "games_s5016_n500")), HD.CONCEPTS)
    gidx = np.unique(tab["game"][sel])
    wires, rls, order = [], [], []
    for gx in gidx:
        s = sel[tab["game"][sel] == gx]
        order.append(s)
        wires.append((None, games[int(gx)]["moves"]))
        rls.append([int(v) for v in tab["ply"][s]])
    order = np.concatenate(order)
    raw_c = model.pattern_count(wires, rls, [k for _dc, k, _d, _z in picked], 3, list(LAYER2.LEVELS), goal, a.threads)
    us, them = LAYER2.counts_matrix(raw_c, len(order), len(picked))
    diff = (us - them).toarray()
    cx = np.zeros_like(diff)
    cx[np.searchsorted(sel, order)] = diff
    say(f"concepts: {len(picked)} distinct kept concepts counted on every {a.concept_every}th quiet row of a game, {len(sel)} rows ({time.perf_counter() - t1:.1f} s)")

    # ---------------------------------------------------------------- the fits
    sets = {"K": (tab["K"], False), "S": (tab["S"], False), "K+S": (np.hstack([tab["K"], tab["S"]]), False)}
    thinned = {"S": tab["S"][sel], "S+C": np.hstack([tab["S"][sel], cx]), "K+S+C": np.hstack([tab["K"][sel], tab["S"][sel], cx])}
    tab_c = {k: (v[sel] if isinstance(v, np.ndarray) and len(v) == len(tab["game"]) else v) for k, v in tab.items()}
    jobs = max(1, a.threads)
    result = {}
    t2 = time.perf_counter()

    def run_family(tbl: dict, family: dict, tag: str, rows_label: str) -> None:
        fit_ = tbl["fit"]
        test = np.flatnonzero(~fit_)
        for h in HORIZONS:
            yv = tbl[f"arr{h}"]
            av = {k: tbl[k][test] for k in ("game", "band", f"arr{h}", "count")}
            readings = []
            for nm, x in family.items():
                for rated in (False, True):
                    feats = np.hstack([x, tbl["R"]]) if rated else x
                    m = fit_pred(tbl, feats, h, fit_, jobs)
                    p = SF.predict(m, feats[test])
                    col = f"pred_{nm}{'+R' if rated else ''}"
                    av[col] = p
                    av["shuf_" + col] = SF.shuffled_within_game(p, av["game"], SEED + h)
                    result[(tag, nm, rated, h)] = {"model": m, "kept": int((m["coef"] != 0).sum()), "of": len(m["coef"])}
                    readings.append((f"{rows_label} h={h} arrival, fit {nm}{' with the ratings' if rated else ''}", col, f"arr{h}"))
                    say(f"{rows_label} h={h} fit {nm}{' + ratings' if rated else ''}: penalty {m['alpha']:.5f}, columns kept {int((m['coef'] != 0).sum())} of {len(m['coef'])}")
            readings.append((f"{rows_label} h={h} arrival, count difference of 78.49", "count", f"arr{h}"))
            best = [r for r in readings if "with the ratings" in r[0]][-1]
            readings.append((best[0] + " shuffled within the game", "shuf_" + best[1], f"arr{h}"))
            vals = corr_lines("", av, a.draws, readings, say)
            result[(tag, "vals", h)] = {f"{k[0]}|{k[1]}": v for k, v in vals.items()}
            if h == H:
                for b, bn in enumerate(HD.BAND_NAMES):
                    sel_b = av["band"] == b
                    sub = {k: (v[sel_b] if isinstance(v, np.ndarray) else v) for k, v in av.items()}
                    rd = [(f"{rows_label} h={h} band {bn}, {r[0].split('fit ')[-1]}", r[1], r[2]) for r in readings if r[1].startswith("pred_") or r[1] == "count"]
                    corr_lines("", sub, max(100, a.draws // 2), rd, say)

    run_family(tab, {k: v[0] for k, v in sets.items()}, "all", "[all quiet rows]")
    run_family(tab_c, thinned, "thin", "[concept rows]")
    say(f"fits done in {time.perf_counter() - t2:.1f} s")

    # ---------------------------------------------------------------- the self-play fit beside, and each fit on the other source
    sp = selfplay_table(a.threads)
    spfit = sp["fit"]
    sptest = np.flatnonzero(~spfit)
    for h in (H,):
        for nm, xs, xh in (("K", sp["K"], tab["K"]), ("S", sp["P"], tab["S"]), ("K+S", np.hstack([sp["K"], sp["P"]]), np.hstack([tab["K"], tab["S"]]))):
            m_sp = fit_pred(sp, xs, h, spfit, jobs)
            m_h = fit_pred(tab, xh, h, fit, jobs)
            for what, m, src_x, src_t, dst_x, dst_t in (("self-play fit on self-play test rows (78.49 form)", m_sp, xs, sp, xs, sp), ("human fit on human test rows", m_h, xh, tab, xh, tab),
                                                          ("self-play fit on human test rows", m_sp, xs, sp, xh, tab), ("human fit on self-play test rows", m_h, xh, tab, xs, sp)):
                dst_fit = dst_t["fit"]
                te = np.flatnonzero(~dst_fit)
                av = {"game": dst_t["game"][te], f"arr{h}": dst_t[f"arr{h}"][te], "count": dst_t["count"][te], "pred": SF.predict(m, dst_x[te])}
                readings = [(f"{nm}: {what}, h={h}", "pred", f"arr{h}")]
                corr_lines("", av, a.draws, readings, say)
    rec_path = Path(paths.experiment("estimator_78_59", "full", "estimator_heads_human.json"))
    if rec_path.exists():
        rec = json.loads(rec_path.read_text())
        for (tag, nm, rated, h), r in result.items():
            if tag == "vals" or nm == "vals" or h != H:
                continue
            m = r["model"]
            key = f"arrival_human_{'thin_' if tag == 'thin' else ''}{nm.replace('+', '')}{'_R' if rated else ''}"
            inputs = {"K": "kinds", "S": "statements", "K+S": "kinds and statements", "S+C": "statements and concepts", "K+S+C": "kinds, statements and concepts"}[nm] + (" and ratings (mean, difference, ply) last" if rated else "")
            rec["heads"][key] = {"type": "expected_value", "definition": "plan 78.49 target at h=8 (a CONFIRMED threat move of the side to move arrives within 8 plies, less the other side's)", "inputs": inputs,
                                 "model": {"mean": m["mean"].tolist(), "scale": m["scale"].tolist(), "coef": m["coef"].tolist(), "intercept": m["intercept"], "alpha": m["alpha"]},
                                 "source": "human games", "data": {"games": len(games), "rows": int(len(tab["game"])), "split": "by game hash, 7 in 10 fit"}, "date": datetime.date.today().isoformat()}
        rec_path.write_text(json.dumps(rec, indent=1))
        say(f"arrival heads added to {rec_path}")
    say(f"done in {time.perf_counter() - t0:.1f} s wall")
    (out / "arrival_obs.txt").write_text("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
