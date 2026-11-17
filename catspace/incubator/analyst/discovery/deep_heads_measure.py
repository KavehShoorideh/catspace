"""PLAN 78.68 STAGE G (parts 1 to 3) AND STAGE H AS THE SCORE: THE RUN. SYSTEM_SPEC 9 and 6. A sandbox: no player is changed, nothing is written to rules/learned or data/derived; every output goes under
artifacts/experiments/deep_heads_78_68/. Game-free: the game is named only through the pieces that exist (odds_measure, the engine); the oracle's answers are read only in the `score` stage, to measure.

INTERFACE (what this module takes, what it returns, who calls it).
  sample    the positions the pairs are made from: our own games (the 78.67 matches) and held-out human games (the 78.59 split's games no fit read), by band of the count of units, disjoint (by game) from every game of
            the ladder's 400 positions; the thin band of the human sample is filled from our own games, never from an engine's continuation. Prints the coverage by count of units. Writes positions.jsonl.
  pairs     deep_heads.pairs on every position on a Pool; writes rows.jsonl (resumable). Reads no answer.
  fit       the heads on the rows: the learning curves (a quarter, a half and all of the fit rows, held-out loss on rows of each kind) and the final heads (every row): heads_deep.pkl and heads_both.pkl, beside the
            heads of 78.67 (human games). Reads no answer.
  score     stage H: the candidates scored on the scored part of the ladder's 400, against two controls; the measure against the accepted moves (not fitted); case_rows.json for ladder_print.
Called by: the command line only.

  .venv/bin/python -m catspace.incubator.analyst.discovery.deep_heads_measure sample|pairs|fit|score  [--n N] [--workers W] [--tag T] [--limit L]
"""
from __future__ import annotations

from catspace.incubator.analyst.discovery import tiebreak_measure as TM          # first: it sets one thread a process before numpy is loaded

import argparse
import hashlib
import json
import multiprocessing
import os
import pickle
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from catspace.incubator.analyst.discovery import deep_heads as DH
from catspace.incubator.analyst.discovery import events_hazard as EH
from catspace.incubator.analyst.discovery import frames_heads as FH
from catspace.incubator.analyst.discovery import ladder_measure as LM
from catspace.incubator.analyst.discovery import tiebreak_number as TN
from catspace.io import paths

SEED = 6869
LADDER_TAG = "full"
TEST_EVERY = 5                                          # one game in this many (by a stable hash) is held out of the fits and read for the learning curves
FRACTIONS = (0.25, 0.5, 1.0)
DRAWS = 200
PER_BAND = 2                                            # positions a game offers to one band (a stated choice: the thin bands need them; the pairs are not labelled, so a few more from one game cost no answer)
PER_GAME = 4                                            # positions of one game at most


def out_root(tag: str = "full") -> Path:
    return Path(paths.experiment(*DH.OUT, tag))


def ladder_root() -> Path:
    return LM.out_root(LADDER_TAG)


def say_to(root: Path, stage: str):
    def say(m: str) -> None:
        line = f"OBS deep heads {stage}: {m}"
        print(line, flush=True)
        with open(root / "obs.txt", "a") as f:
            f.write(line + "\n")
    return say


def jsonl(p: Path) -> list:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


# ------------------------------------------------------------------------------------------------------------------------------ the positions

def sample(n: int, say) -> list:
    from catspace.incubator.analyst.discovery import graph_layer1 as GL
    from catspace.incubator.analyst.discovery import human_heads_data as HD
    layer1 = GL.model(HD.GAME, HD.VARIANT)
    used = {DH.game_key(r["id"]) for r in jsonl(ladder_root() / "positions.jsonl")}
    rng = random.Random(SEED)
    pools: dict = {}
    dropped = 0
    for source, games in (("own", LM.own_games()), ("human", LM.human_games())):
        games = [g for g in games if f"{source}:{g[0]}" not in used]
        dropped += 0
        rng.shuffle(games)
        for gid, moves in games:
            cnt = LM.unit_counts(layer1, moves)
            by = defaultdict(list)
            for k in range(LM.MIN_PLY, len(moves)):
                by[LM.band_of(cnt[k])].append(k)
            for b, ks in by.items():
                for k in rng.sample(ks, min(PER_BAND, len(ks))):
                    pools.setdefault((source, b), []).append((gid, k, None, moves, cnt))
        say(f"candidates ({source}, games not among the ladder's 400): " + ", ".join(f"{b} {len(pools.get((source, b), []))}" for b, _l, _h in LM.BANDS))
    half = n // 2
    want = {(s, b): int(round(half * LM.QUOTA[b])) for s in ("own", "human") for b, _l, _h in LM.BANDS}
    taken_games: Counter = Counter()
    take: dict = {}
    for s in ("human", "own"):
        for b, _l, _h in LM.BANDS:
            pool = pools.get((s, b), [])
            rng.shuffle(pool)
            got = []
            for gid, k, _u, moves, cnt in pool:
                if len(got) >= want[(s, b)]:
                    break
                if taken_games[(s, gid)] >= PER_GAME or not LM.playable(moves[:k]):
                    continue
                taken_games[(s, gid)] += 1
                got.append((gid, k, cnt[k], moves))
            take[(s, b)] = got
            if s == "human" and len(got) < want[(s, b)]:
                want[("own", b)] += want[(s, b)] - len(got)                  # the thin band of the human sample is filled from our own games
    rows = []
    for (s, b), got in take.items():
        for gid, k, u, moves in got:
            rows.append({"id": f"{s}:{gid}:{k}", "source": s, "units": u, "band": b, "moves": moves[:k]})
    say(f"the sample: {len(rows)} positions; " + "; ".join(f"{s} {b}: {len(take[(s, b)])} (asked {want[(s, b)]})" for s in ("own", "human") for b, _l, _h in LM.BANDS))
    for s in ("human", "own"):
        us = np.array([r["units"] for r in rows if r["source"] == s])
        if len(us):
            say(f"ENDGAME COVERAGE {s}: positions by count of units: " + ", ".join(f"{lo}-{hi}: {int(((us >= lo) & (us <= hi)).sum())}" for lo, hi in ((0, 4), (5, 8), (9, 12), (13, 16), (17, 24), (25, 99))) + f"; fewest {int(us.min())}, median {int(np.median(us))}")
    return rows


# ------------------------------------------------------------------------------------------------------------------------------ the pairs

_TOOLS = None


def _pairs_job(chunk: list) -> dict:
    global _TOOLS
    if _TOOLS is None:
        _TOOLS = DH.DeepTools()
    t0 = time.perf_counter()
    before = Counter(_TOOLS.skipped)
    try:
        rows = DH.pairs(chunk, DH.WAKE_CLOCK, DH.DEEP_CLOCK, _TOOLS)
        err = None
    except Exception as e:                                         # a position the readers cannot take is reported, never dropped quietly
        import traceback
        rows, err = [], repr(e)[:200] + " | " + " <- ".join(f"{f.name}:{f.lineno}" for f in reversed(traceback.extract_tb(e.__traceback__)))[:300]
    return {"ids": [p["id"] for p in chunk], "rows": rows, "error": err, "wall": time.perf_counter() - t0, "skipped": dict(_TOOLS.skipped - before)}


def run_pairs(positions: list, workers: int, root: Path, say, limit: "int | None", budget: "float | None" = None) -> list:
    out_file = root / "rows.jsonl"
    done_ids: set = set()
    if out_file.exists():
        for ln in out_file.read_text().splitlines():
            if ln.strip():
                done_ids.add(json.loads(ln)["pid"])
    skip_file = root / "skipped.jsonl"
    seen = {json.loads(x)["id"] for x in skip_file.read_text().splitlines() if x.strip()} if skip_file.exists() else set()
    todo = [p for p in positions if p["id"] not in done_ids and p["id"] not in seen]
    random.Random(SEED).shuffle(todo)                                  # a run cut by its time budget leaves a balanced part, not the first bands of the file
    if limit:
        todo = todo[:limit]
    chunks = [todo[i:i + 4] for i in range(0, len(todo), 4)]
    t0 = time.perf_counter()
    sk: Counter = Counter()
    errs, n_rows, n_done = [], 0, 0
    with open(out_file, "a") as f, open(skip_file, "a") as fs, multiprocessing.get_context("spawn").Pool(workers) as pool:
        for d in pool.imap_unordered(_pairs_job, chunks, chunksize=1):
            n_done += len(d["ids"])
            for r in d["rows"]:
                f.write(json.dumps(r) + "\n")
            f.flush()
            for pid in d["ids"]:
                if not any(r["pid"] == pid for r in d["rows"]):
                    fs.write(json.dumps({"id": pid}) + "\n")
            fs.flush()
            n_rows += len(d["rows"])
            sk.update(d["skipped"])
            if d["error"]:
                errs.append(d["error"])
            if budget and time.perf_counter() - t0 > budget:
                say(f"the time budget of {budget:.0f} s is spent: stopping with {n_done} of {len(todo)} positions of this run read (the rest are left unread, not dropped by a rule)")
                pool.terminate()
                break
            if (n_done // 4) % 25 == 0:
                el = time.perf_counter() - t0
                say(f"progress: {n_done} of {len(todo)} positions, {n_rows} rows, {el:.0f} s, about {el / max(1, n_done) * (len(todo) - n_done):.0f} s left")
    say(f"pairs: {n_done} positions in {time.perf_counter() - t0:.0f} s, {n_rows} rows; skipped {dict(sk)}; errors {len(errs)}" + (f" ({Counter(e[:80] for e in errs).most_common(2)})" if errs else ""))
    return jsonl(out_file)


def rows_report(rows: list, say) -> None:
    by = Counter(r["made_by"] for r in rows)
    say(f"ROWS: {len(rows)} from {len({r['pid'] for r in rows})} positions; made by a settled deep reading {by['deep']}, by playouts {by['playout']}; anchors (the record reached an end of the game): "
        f"{sum(r['anchor'] for r in rows)} (deep {sum(r['anchor'] for r in rows if r['made_by'] == 'deep')}, playout {sum(r['anchor'] for r in rows if r['made_by'] == 'playout')}); leaves not settled by the layer-1 reading {sum(not r['quiet'] for r in rows)}")
    for src in ("own", "human"):
        sub = [r for r in rows if r["source"] == src]
        if sub:
            say(f"ROWS {src}: {len(sub)} rows; by band " + ", ".join(f"{b} {sum(1 for r in sub if r['band'] == b)}" for b, _l, _h in LM.BANDS))
    for kind in ("deep", "playout"):
        sub = [r for r in rows if r["made_by"] == kind]
        if sub:
            for rel in ("us", "them"):
                ev = sum(r["target"][rel][1] for r in sub)
                cut = sum(r["target"][rel][2] for r in sub)
                say(f"ROWS target {kind} [{rel}]: rows {len(sub)}, event (a unit won and kept) {ev} ({100 * ev / len(sub):.1f} pct), record cut by an unsettled taking {cut}; mean record {np.mean([r['rec'] for r in sub]):.1f} plies")


# ------------------------------------------------------------------------------------------------------------------------------ the fit and the learning curves

def is_test(pid: str) -> bool:
    return int(hashlib.md5(DH.game_key(pid).encode()).hexdigest(), 16) % TEST_EVERY == 0


def take_rows(rows: list, ids) -> list:
    return [rows[i] for i in ids]


def human_rows(say) -> dict:
    """The fit rows of the human games of 78.65 (the heads' own fit set), as arrays, grouped by game."""
    d = FH.build(Path(paths.experiment(*FH.STATE)), lambda m: None)
    rows, sample_, roles = d["rows"], d["sample"], d["roles"]
    X = FH.steps_of(d)[-1][1]
    assert X.shape[1] == len(pickle.loads(TN.heads_file().read_bytes())["columns"]), "the human rows and the heads read different columns"
    fit_rows = np.flatnonzero(rows["fit"])
    out = {"x": X[fit_rows].astype(np.float32), "g": rows["g"][fit_rows].astype(np.int32)}
    for rel in ("us", "them"):
        D, has, cut = EH.targets(DH.CORE, rel, roles, rows, sample_["res"])
        out[rel] = (D[fit_rows], has[fit_rows], cut[fit_rows])
    say(f"human fit rows loaded: {len(fit_rows)} rows from {len(np.unique(out['g']))} games")
    return out


def human_subset(h: dict, n: int, rng) -> tuple:
    """About `n` rows of the human fit rows, whole games drawn at random: (x, {rel: (D, has, cut)}, g)."""
    games = np.unique(h["g"])
    rng.shuffle(games)
    cnt = Counter(h["g"].tolist())
    pick, tot = [], 0
    for g in games:
        pick.append(g)
        tot += cnt[int(g)]
        if tot >= n:
            break
    sel = np.flatnonzero(np.isin(h["g"], pick))
    return h["x"][sel], {rel: tuple(v[sel] for v in h[rel]) for rel in ("us", "them")}, h["g"][sel]


def stage_fit(rows: list, root: Path, say) -> None:
    template = pickle.loads(TN.heads_file().read_bytes())
    pids = list(dict.fromkeys(r["pid"] for r in rows))
    test_p = {p for p in pids if is_test(p)}
    fit_p = sorted((p for p in pids if p not in test_p), key=lambda p: hashlib.md5(p.encode()).hexdigest())
    test_ix = [i for i, r in enumerate(rows) if r["pid"] in test_p]
    say(f"split by game (one game in {TEST_EVERY} held out by a stable hash): {len(fit_p)} fit positions, {len(test_p)} held-out positions; held-out rows {len(test_ix)}")
    at = DH.arrays_of([rows[i] for i in test_ix])
    trows = [rows[i] for i in test_ix]
    kinds = {"deep rows": np.array([r["made_by"] == "deep" for r in trows]), "playout rows": np.array([r["made_by"] == "playout" for r in trows]), "anchor rows": np.array([r["anchor"] for r in trows])}
    hum = human_rows(say)
    rng = np.random.default_rng(SEED)
    heads_h = template
    curve: dict = {}

    def read(name: str, heads: dict, frac: float) -> None:
        lr = DH.loss_rows(heads, at, np.arange(len(trows)))
        for kn, m in kinds.items():
            if not m.any():
                continue
            for form in ("linear", "trees"):
                v, lo, hi = EH.boot(at["g"][m], lr[form][m], lr["steps"][m], DRAWS, SEED)
                curve[(name, kn, form, frac)] = (v, lo, hi, int(m.sum()))
    read("human games", heads_h, 0.0)
    for frac in FRACTIONS:
        k = max(1, int(round(frac * len(fit_p))))
        keep_p = set(fit_p[:k])
        fr = [r for r in rows if r["pid"] in keep_p]
        a = DH.arrays_of(fr)
        t0 = time.perf_counter()
        hd = DH.fit(None, template, None, "deep", a, say)
        read("our deeper thinking", hd, frac)
        xe, re_, ge = human_subset(hum, len(fr), rng)
        hb = DH.fit(None, template, (xe, re_, ge), "both", a, say)
        read("both", hb, frac)
        say(f"fraction {frac}: {len(fr)} fit rows from {k} positions; wall {time.perf_counter() - t0:.0f} s")
    for kn in kinds:
        n_rows = next((v[3] for key, v in curve.items() if key[1] == kn), 0)
        for form in ("linear", "trees"):
            base = None
            for name in ("human games", "our deeper thinking", "both"):
                for frac in ((0.0,) if name == "human games" else FRACTIONS):
                    v = curve.get((name, kn, form, frac))
                    if v:
                        say(f"CURVE held-out {kn} ({n_rows} rows) {form}: head fitted on {name}, fit fraction {frac if frac else 'n/a'}: log loss per step {v[0]:.5f} [{v[1]:.5f}, {v[2]:.5f}] (90 pct band over games)")
    # the final heads: every row (the scored part of the ladder is disjoint from every row, so nothing is held out here)
    a = DH.arrays_of(rows)
    hd = DH.fit(None, template, None, "deep", a, say)
    xe, re_, ge = human_subset(hum, len(rows), rng)
    hb = DH.fit(None, template, (xe, re_, ge), "both", a, say)
    for nm, h in (("heads_deep.pkl", hd), ("heads_both.pkl", hb)):
        (root / nm).write_bytes(pickle.dumps(h))
    say(f"final heads fitted on all {len(rows)} rows (deep) and on them beside {len(xe)} human rows (both): saved")


# ------------------------------------------------------------------------------------------------------------------------------ stage H

def _prep_job(pos: dict) -> tuple:
    global _TOOLS
    if _TOOLS is None:
        _TOOLS = DH.DeepTools()
    t0 = time.perf_counter()
    try:
        p = DH.prepare(pos, _TOOLS)
        return pos["id"], p, None
    except Exception as e:
        import traceback
        return pos["id"], None, pos["id"] + ": " + repr(e)[:200] + " | " + " <- ".join(f"{f.name}:{f.lineno}" for f in reversed(traceback.extract_tb(e.__traceback__)))[:300]


def prepare_all(positions: list, workers: int, root: Path, say) -> dict:
    f = root / "preps.pkl"
    preps = pickle.loads(f.read_bytes()) if f.exists() else {}
    todo = [p for p in positions if p["id"] not in preps]
    t0 = time.perf_counter()
    errs = []
    with multiprocessing.get_context("spawn").Pool(workers) as pool:
        for pid, p, err in pool.imap_unordered(_prep_job, todo, chunksize=1):
            if p is not None:
                preps[pid] = p
            else:
                errs.append(err)
    f.write_bytes(pickle.dumps(preps))
    say(f"prepared {len(todo)} positions of the scored part in {time.perf_counter() - t0:.0f} s (search at the wake's clock, forcing read, inputs of every tied move at the root and at the leaf); errors {len(errs)}" + (f" ({errs[:3]})" if errs else ""))
    return preps


def candidates(root: Path, preps: dict) -> list:
    hum = pickle.loads(TN.heads_file().read_bytes())
    hum["horizon"] = None
    deep = pickle.loads((root / "heads_deep.pkl").read_bytes())
    both = pickle.loads((root / "heads_both.pkl").read_bytes())
    pos_moves: dict = {}
    spent: dict = {}

    def mk(name: str, heads: dict, kind: str, where: str) -> dict:
        memo: dict = {}

        def pick(pid: str, prep: dict) -> str:
            if pid not in memo:
                t0 = time.perf_counter()
                tied = list(prep["eq"])
                memo[pid] = (DH.root_pick if where == "root" else DH.leaf_pick)({"moves": pos_moves[pid]}, tied, heads, prep, kind)
                spent.setdefault(name, {})[pid] = time.perf_counter() - t0
            return memo[pid]

        def cost(prep: dict) -> float:
            pid = next((k for k, v in preps.items() if v is prep), None)
            return prep["t_root" if where == "root" else "t_leaf"] + spent.get(name, {}).get(pid, 0.0)
        return {"name": name, "pick": pick, "cost": cost, "random": False, "where": where, "kind": kind}
    out = [{"name": "control: the search's own first move", "pick": lambda pid, p: p["move"], "cost": lambda p: 0.0, "random": False, "where": None},
           {"name": "control: a random pick (expected)", "pick": None, "cost": lambda p: 0.0, "random": True, "where": None}]
    for src, heads in (("human games", hum), ("deeper thinking", deep), ("both", both)):
        for kind in ("trees", "linear"):
            for where in ("root", "leaf"):
                out.append(mk(f"{src} / {kind} / {where}", heads, kind, where))
    return out, pos_moves


def stage_score(workers: int, root: Path, say) -> None:
    lr = ladder_root()
    positions = {p["id"]: p for p in jsonl(lr / "positions.jsonl")}
    results = {d["id"]: d for d in jsonl(lr / "results.jsonl")}
    sl = jsonl(lr / "surprise_list.jsonl")
    open_ids = [r["id"] for r in sl]
    resolved_ids = [i for i, d in results.items() if d["result"] and d["result"]["rung"] in (1, 2)]
    scored = [positions[i] for i in positions if DH.split_part(i) == "scored" and results.get(i, {}).get("result")]
    say(f"THE SPLIT of the ladder's 400 positions: by game, a stable hash of the game's key (even: scored, odd: the part a candidate may be made from); scored {len(scored)} positions in {len({DH.game_key(p['id']) for p in scored})} games; "
        f"open surprises there {sum(1 for i in open_ids if DH.split_part(i) == 'scored')} of {len(open_ids)}; resolved at rungs 1 or 2 there {sum(1 for i in resolved_ids if DH.split_part(i) == 'scored')} of {len(resolved_ids)}")
    say(f"load average {os.getloadavg()[0]:.2f}; {workers} workers")
    preps = prepare_all(scored, workers, root, say)
    cands, pos_moves = candidates(root, preps)
    pos_moves.update({i: positions[i]["moves"] for i in preps})
    ans = {d["id"]: d["accepted"] for d in jsonl(lr / "answers.jsonl")}          # the answers file: read here, handed to score() and hit_rate() only
    op_scored = [i for i in open_ids if DH.split_part(i) == "scored" and i in preps]
    reach = [i for i in op_scored if DH.tie_eligible(preps[i], set(ans[i]))]
    say(f"CEILING: of {len(op_scored)} open surprises in the scored part, {len(reach)} have an accepted move among the moves the search holds equal at the wake's clock (the most any tie-break can resolve); "
        f"the ladder's own head at rung 2 (the arrival number) resolves none of the open ones and creates none of the resolved ones, by construction (they were open because it missed)")
    lv = [v for p in preps.values() for v in p["leaf"].values()]
    pl = np.array([v["plies"] for v in lv if v["end"] in (0, 1, 2)])
    say(f"LEAVES: {len(lv)} leaves of tied moves in the scored part (positions with two or more moves held equal: {sum(1 for p in preps.values() if len(p['eq']) >= 2)}); the line's settled end is the other side's reply to the move (two plies from the root, the least a line shows) in {int((pl == 2).sum())} "
        f"({100 * (pl == 2).mean():.1f} pct), mean plies {pl.mean():.2f}; the line ends the game in {sum(1 for v in lv if v['end'] in (1, 2, -1))}; no inputs at the leaf in {sum(1 for v in lv if v['end'] == 0 and v['x'] is None)}; "
        f"moves held equal per position: mean {np.mean([len(p['eq']) for p in preps.values() if len(p['eq']) >= 2]):.1f}, most {max(len(p['eq']) for p in preps.values())}")
    table, hits = [], []
    for c in cands:
        s = DH.score(c, preps, open_ids, resolved_ids, ans, "scored")
        h = DH.hit_rate(c, preps, [p["id"] for p in scored], ans, "scored")
        table.append(s)
        hits.append(h)
    for s, h in zip(table, hits):
        lo_r, hi_r = DH.wilson(s["resolved"], s["n_open"])
        lo_c, hi_c = DH.wilson(s["created"], s["n_resolved_sample"])
        say(f"STAGE H {s['name']}: resolved {s['resolved']:.1f} of {s['n_open']} open surprises (share {s['resolved'] / max(1, s['n_open']):.3f} [{lo_r:.3f}, {hi_r:.3f}]); created {s['created']:.1f} of {s['n_resolved_sample']} resolved positions (share "
            f"{s['created'] / max(1, s['n_resolved_sample']):.3f} [{lo_c:.3f}, {hi_c:.3f}]); net {s['resolved'] - s['created']:+.1f}; cost {s['cost']:.3f} s a position; created over all {s['n_resolved_all']} resolved positions of the part: {s['created_all']:.1f}")
    ctrl = table[0]
    rng = np.random.default_rng(SEED)
    ia = rng.integers(0, len(ctrl["res_vec"]), (1000, len(ctrl["res_vec"])))
    ib = rng.integers(0, max(1, len(ctrl["keep_vec"])), (1000, len(ctrl["keep_vec"])))
    for s in table:
        d_open = np.array(s["res_vec"]) - np.array(ctrl["res_vec"])
        d_keep = np.array(s["keep_vec"]) - np.array(ctrl["keep_vec"])                 # created less: more kept is fewer created
        diff = d_open[ia].sum(axis=1) + d_keep[ib].sum(axis=1)                         # net of the candidate less net of the search's own first move
        lo, hi = np.percentile(diff, [5, 95])
        say(f"STAGE H net against the search's own first move, {s['name']}: {d_open.sum() + d_keep.sum():+.1f} positions [{lo:+.1f}, {hi:+.1f}] (90 pct bootstrap over positions, paired)")
    for h in hits:
        say(f"MEASURE against the accepted moves (not fitted) {h['name']}: picks an accepted move in {h['hits']:.1f} of {h['n']} ties ({100 * h['rate']:.1f} pct [{100 * h['lo']:.1f}, {100 * h['hi']:.1f}], 90 pct Wilson)")
    best = max((s for s, c in zip(table, cands) if c["where"] == "leaf"), key=lambda s: s["resolved"] - s["created"])
    say(f"BEST CANDIDATE by net among those that read at the leaf (the build's question; chosen by this rule before the cases were read): {best['name']}")
    c = next(c for c in cands if c["name"] == best["name"])
    rows = {"best": best["name"], "resolved": [], "created": []}
    for i in [i for i in open_ids if DH.split_part(i) == "scored" and i in preps]:
        p = preps[i]
        if len(p["eq"]) >= 2 and set(ans[i]) & set(p["eq"]) and not set(p["eq"]) <= set(ans[i]):
            mv = c["pick"](i, p)
            if mv in set(ans[i]):
                rows["resolved"].append(_case(i, p, mv, positions[i]))
    samp_ids = [i for i in resolved_ids if DH.split_part(i) == "scored" and i in preps]
    for i in samp_ids:
        p = preps[i]
        if DH.tie_eligible(p, set(ans[i])) and results[i]["result"]["rung"] == 2:
            mv = c["pick"](i, p)
            if mv not in set(ans[i]):
                rows["created"].append(_case(i, p, mv, positions[i]))
    for k in ("resolved", "created"):
        by_end = Counter(bool(r["leaf_end"]) for r in rows[k])
        say(f"CASES {k}: {len(rows[k])}; the pick's line ends the game (a concrete end decided it, no head) in {by_end[True]}, ends settled in {by_end[False]}")
        random.Random(SEED).shuffle(rows[k])                                    # the ten printed are a seeded draw, not the first of a list
    (root / "case_rows.json").write_text(json.dumps(rows))
    say(f"case rows: {len(rows['resolved'])} open surprises the best candidate now picks an accepted move in, {len(rows['created'])} positions it creates a surprise in (ids only with the candidate's own readings; no oracle field)")


def _case(pid: str, prep: dict, mv: str, pos: dict) -> dict:
    lf = prep["leaf"].get(mv) or {}
    return {"id": pid, "moves": pos["moves"], "tied": prep["eq"], "search_move": prep["move"], "pick": mv, "line": lf.get("line"), "leaf_end": lf.get("end"),
            "lines": {m: (v.get("line"), v.get("end")) for m, v in prep["leaf"].items()}}


def main(argv: "list | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stage", choices=("sample", "pairs", "fit", "score"))
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--workers", type=int, default=9)
    ap.add_argument("--tag", default="full")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--budget", type=float, default=None, help="seconds of wall time the pairs stage may spend")
    a = ap.parse_args(argv)
    root = out_root(a.tag)
    root.mkdir(parents=True, exist_ok=True)
    say = say_to(root, a.stage)
    pos_file = root / "positions.jsonl"
    if a.stage == "sample":
        rows = sample(a.n, say)
        pos_file.write_text("".join(json.dumps(r) + "\n" for r in rows))
    elif a.stage == "pairs":
        say(f"load average {os.getloadavg()[0]:.2f}; wake clock {DH.WAKE_CLOCK}, deep clock {DH.DEEP_CLOCK}, record {DH.RECORD_PLIES} plies, playouts {DH.PLAYOUTS} at {DH.PLAYOUT_SECONDS} s a move; {a.workers} workers")
        rows = run_pairs(jsonl(pos_file), a.workers, root, say, a.limit, a.budget)
        rows_report(rows, say)
    elif a.stage == "fit":
        rows = jsonl(root / "rows.jsonl")
        rows_report(rows, say)
        stage_fit(rows, root, say)
    else:
        stage_score(a.workers, root, say)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
