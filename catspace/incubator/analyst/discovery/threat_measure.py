"""PLAN 78.70 D: THE CHECKS ON THE THREAT LIST (a measure script; the list itself is threat_list.py, the print is threat_print.py). SYSTEM_SPEC 8 (the threat list, its four bands) and 4 (open tactics are carried
forward). A measurement in a sandbox on rows that exist: the HELD-OUT games of the 9000 human games of 78.65 (the games the 78.59 split did not fit; the kinds reading was fitted on the others), one run each,
directions only; no player changed, nothing written to rules/learned or data/derived. Game-free: a kind, a role or a place is a word read at run time. No engine evaluation, no labeller, no external model, no
assumed worth (the stake is read from the kinds head of 78.69, whose rows were built with an empty worth). Human games are read only through the existing row builders; no player id is written.

INTERFACE (what this module takes, what it returns, who calls it).
  read(a)           takes the number of held-out games (chosen in the md5 order of their ids, the seed word printed) and returns the path of a pickle with the raw rows of the frame read with answers
                    (threat_list.ThreatReader.read) at every settled row of those games, and the arrays that go with them (game, ply, the spread of the two readings, the stake table). Resumable by chunk.
  measure(a)        stage D on that pickle: rows per position in each band; per band how often a unit of that side is then taken (rate per ply of exposure, time to the event; no horizon; bootstrap over
                    GAMES; the shuffled-label control and ply held fixed, as 78.69); how often the named response, or any answer with the same reason, was the move played (second band); endgame coverage.
  carry(a)          THE CARRYING TEST (amendment, point 2) on every ply of some held-out games, with each defect of the read printed.
  puzzles(a)        (if time allows) on the 78.45 puzzles as a MEASURE only: how often the solution's first move is by a unit that threatens in the top band of the other side's list. No puzzle enters a fit.
  Called by: the command line only; nothing calls it.

BANDS, two ways (the amendment, point 3): by the SPREAD of the two readings at the position (a row is high when its stake is above the spread |delta| of two_readings, class `result`), and by the stake RANKED
within the position (threat_list.rank_high_of). Order of the four bands: 1 high with no response, 2 high with a response, 3 low with no response, 4 low with a response. Neither way is called the winner on one run.

  .venv/bin/python -m catspace.incubator.analyst.discovery.threat_measure read    [--games N] [--out-dir D] [--threads K]
  .venv/bin/python -m catspace.incubator.analyst.discovery.threat_measure measure [--out-dir D]
  .venv/bin/python -m catspace.incubator.analyst.discovery.threat_measure carry   [--games N] [--out-dir D] [--threads K]
"""
from __future__ import annotations

import argparse
import hashlib
import os
import pickle
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")                       # before numpy and torch load

import numpy as np

from catspace.incubator.analyst.discovery import events_hazard as EH
from catspace.incubator.analyst.discovery import threat_list as TL
from catspace.incubator.analyst.discovery import two_readings as TR
from catspace.io import paths

OUT = ("threats_78_70",)
SEED_WORD = "78.70:held-out games"
CHUNK = 50
SEED = 7070
CLASS_NAME = "result"                                               # the class of the kinds reading the stake is read in: the result of the game
N_BANDS = 4


def out_dir(p: "str | None" = None) -> Path:
    d = Path(p) if p else Path(paths.experiment(*OUT))
    d.mkdir(parents=True, exist_ok=True)
    return d


def make_say(out: Path, name: str):
    lines: list = []

    def say(m: str) -> None:
        lines.append(f"OBS threats: {m}")
        print(lines[-1], flush=True)
        (out / name).write_text("\n".join(lines) + "\n")
    return say


def h(text: str) -> int:
    return int(hashlib.md5((SEED_WORD + text).encode()).hexdigest(), 16)


# ------------------------------------------------------------------------------------------------------------------------------ the rows and the readings

def load_world(say):
    """The 78.69 rows (78.65's hazard rows with the frame blocks), its saved heads and the readings on the held-out rows."""
    d = TR.load(say)
    saved = pickle.loads(TR.heads_file(None).read_bytes())
    te = TR.test_rows(d, None)
    R = TR.Readings(d, saved, te)
    return d, R, te


def pick_games(d: dict, te: np.ndarray, n: int) -> list:
    """The first n held-out games in the md5 order of their ids (seed word SEED_WORD): game numbers into the sample."""
    ids = d["sample"]["ids"]
    gs = np.unique(d["rows"]["g"][te])
    return sorted((int(g) for g in gs), key=lambda g: h(ids[g]))[:n]


def stake_table(d: dict, R, te: np.ndarray) -> np.ndarray:
    """The stake table of the held-out rows `te` (see `stake_of_kinds`)."""
    return stake_of_kinds(R, d["X_kinds"][te])


def stake_of_kinds(R, K: np.ndarray, remaining: "np.ndarray | None" = None) -> np.ndarray:
    """(rows, 2, the head's kinds): the change in the owner's chance of the result (the kinds reading of 78.69) if a unit of that kind of that owner is lost, owner index 1 the side to move, 0 the other.
    Signed: a loss that raises the owner's chance by the kinds reading is negative. `K`: the kinds ours less theirs, one row per position; `remaining`: the steps left in the record (the rows' own when None)."""
    pu0, pt0 = (R.read(CLASS_NAME, rel, "kinds", K, remaining) for rel in ("us", "them"))
    r0 = pu0 - pt0
    out = np.zeros((len(K), 2, K.shape[1]))
    for owner in (1, 0):
        for col in range(K.shape[1]):
            Kv = K.copy()
            Kv[:, col] += -1.0 if owner == 1 else 1.0
            pu, pt = (R.read(CLASS_NAME, rel, "kinds", Kv, remaining) for rel in ("us", "them"))
            r1 = pu - pt
            out[:, owner, col] = (r0 - r1) if owner == 1 else (r1 - r0)
    return out


# ------------------------------------------------------------------------------------------------------------------------------ read: the frame read with answers on the held-out rows

def cmd_read(a) -> int:
    t0 = time.perf_counter()
    out = out_dir(a.out_dir)
    say = make_say(out, "read_obs.txt")
    d, R, te = load_world(say)
    rows = d["rows"]
    games = pick_games(d, te, a.games)
    say(f"sample: {len(games)} held-out games (the games the 78.59 split did not fit), the first in the md5 order of their ids with the seed word '{SEED_WORD}'; held-out games in all {len(np.unique(rows['g'][te]))}, rows {len(te)}")
    pos = {int(g): [] for g in games}
    for j, (g, p) in enumerate(zip(rows["g"][te], rows["ply"][te])):
        if int(g) in pos:
            pos[int(g)].append((int(p), j))
    stake = stake_table(d, R, te)
    spread = np.abs(R.delta[CLASS_NAME])
    allg = d["allg"]
    ids = d["sample"]["ids"]
    reader = TL.ThreatReader(a.threads)
    raws: list = []
    names = None
    t1 = time.perf_counter()
    nchunk = (len(games) + CHUNK - 1) // CHUNK
    for c in range(nchunk):
        pf = out / f"raw_part_{c:04d}.pkl"
        ch = games[c * CHUNK:(c + 1) * CHUNK]
        if pf.exists():
            part = pickle.loads(pf.read_bytes())
        else:
            t2 = time.perf_counter()
            res = reader.read([(None, allg[ids[g]]["moves"]) for g in ch], [[p for p, _j in pos[g]] for g in ch])
            rs = res["rows"]
            for r in rs:
                g = ch[r["game"]]
                r["g"] = g
                r["te_ix"] = dict(pos[g])[r["ply"]]
                del r["game"]
            part = {"rows": rs, "names": {k: res[k] for k in ("kinds", "roles", "places")}}
            pf.write_bytes(pickle.dumps(part))
            dt = time.perf_counter() - t2
            say(f"progress: chunk {c + 1} of {nchunk}: {len(rs)} rows, {dt:.1f} s wall over {a.threads} threads ({dt / max(1, len(rs)):.4f} s per row); run so far {time.perf_counter() - t1:.0f} s")
        raws += part["rows"]
        names = part["names"]
    sel = np.array([r["te_ix"] for r in raws], np.int64)
    for r in raws:
        assert r["mover"] == int(rows["mover"][te][r["te_ix"]]), "the mover of a threat row and of a hazard row differ"
    pk = out / "raw.pkl"
    pk.write_bytes(pickle.dumps({"rows": raws, "names": names, "games": games, "te_ix": sel, "stake": stake[sel], "spread": spread[sel], "g": rows["g"][te][sel], "ply": rows["ply"][te][sel],
                                 "mover": rows["mover"][te][sel], "kind_cols": list(rows["meta"]["kinds"]), "sample_ids": [ids[g] for g in games]}))
    say(f"done: {len(raws)} rows of {len(games)} games, {sum(len(f['claims']) for r in raws for f in r['frames'])} claims; saved {pk}; wall {time.perf_counter() - t0:.0f} s")
    return 0


# ------------------------------------------------------------------------------------------------------------------------------ the lists of the read rows

def build_lists(z: dict) -> list:
    """For every raw row of the pickle: the threat list (both sides), each row with its two bands, by the spread of the position and by the stake ranked within it."""
    names, cols, out = z["names"], z["kind_cols"], []
    for j, raw in enumerate(z["rows"]):
        st = z["stake"][j]

        def stake_of(kind, own, st=st):
            return float(st[1 if own else 0, cols.index(kind)]) if kind in cols else None
        lst = TL.threat_rows(raw, names, stake_of)
        both = lst["mover"] + lst["other"]
        rk = TL.rank_high_of(both)
        for r in both:
            r["b_spread"] = TL.band_of(r, float(z["spread"][j]), None, "spread")
            r["b_rank"] = TL.band_of(r, None, rk.get(id(r), False), "rank")
        out.append(lst)
    return out


def pct(x: float) -> str:
    return f"{100.0 * x:.2f}"


def band_txt(point: float, dr: np.ndarray, scale: float = 1.0, fmt: str = "{:.3f}") -> str:
    lo, hi = TR.band(point, dr * scale)
    return f"{fmt.format(point * scale)} [{fmt.format(lo)}, {fmt.format(hi)}]"


def standardised(gb: "TR.GameBoot", obs: np.ndarray, expo: np.ndarray, grp: np.ndarray, ng: int, strat: np.ndarray, ns: int) -> "tuple[np.ndarray, np.ndarray]":
    """The rate of each group with the mix of the strata (here quintiles of ply) held at the cells' own overall mix: the stratum's own effect taken out. (point per group, draws (D, ng))."""
    A, B = gb.sums(obs, expo, grp * ns + strat, ng * ns)
    w = np.bincount(strat, minlength=ns) / len(strat)

    def f(Wm):
        a, b = (Wm @ A).reshape(-1, ng, ns), (Wm @ B).reshape(-1, ng, ns)
        r = a / np.maximum(b, 1e-12)
        ww = (b > 0) * w
        return (r * ww).sum(2) / np.maximum(ww.sum(2), 1e-12)
    return f(np.ones((1, A.shape[0])))[0], f(gb.W)


def rate_lines(say, title: str, gb: "TR.GameBoot", obs: np.ndarray, expo: np.ndarray, steps: np.ndarray, grp: np.ndarray, labels: list, strat: np.ndarray, ns: int, rng) -> None:
    ng = len(labels)
    pt, dr = gb.ratio(obs.astype(float), expo, grp, ng)
    parts = []
    for b in range(ng):
        m = grp == b
        ev = steps[m & (obs > 0)]
        parts.append(f"{labels[b]} ({int(m.sum())} cells, {int((obs[m] > 0).sum())} with the event): {band_txt(pt[b], dr[:, b], 100.0)}; median plies to the event {np.median(ev) if len(ev) else float('nan'):.1f}")
    say(f"{title}: events per 100 plies of exposure, 90 pct band over games: " + "; ".join(parts))
    sg = grp.copy()
    for s in range(ns):
        i = np.flatnonzero(strat == s)
        sg[i] = rng.permutation(sg[i])
    pc, dc = gb.ratio(obs.astype(float), expo, sg, ng)
    say(f"{title}: CONTROL the same labels SHUFFLED inside each quintile of ply: " + "; ".join(f"{labels[b]} {band_txt(pc[b], dc[:, b], 100.0)}" for b in range(ng)))
    ps, ds = standardised(gb, obs.astype(float), expo, grp, ng, strat, ns)
    say(f"{title}: PLY HELD FIXED (the mix of the five quintiles of ply as in all cells): " + "; ".join(f"{labels[b]} {band_txt(ps[b], ds[:, b], 100.0)}" for b in range(ng)))
    if ng > 2:
        rr = dr[:, 1] / np.maximum(dr[:, 0], 1e-12)
        r2 = ds[:, 1] / np.maximum(ds[:, 0], 1e-12)
        say(f"{title}: first row group over no row: {band_txt(pt[1] / max(pt[0], 1e-12), rr)}; with ply held fixed {band_txt(ps[1] / max(ps[0], 1e-12), r2)}")


# ------------------------------------------------------------------------------------------------------------------------------ measure: stage D

def taken_after(res: dict, key: tuple, ply: int) -> "tuple[float, bool, int]":
    """(exposure in plies, whether the event came, plies to it) from the row at `ply`: the first event of the key at or after it, as events_hazard counts; no horizon."""
    import bisect
    steps = res["_steps"].get(key)
    if steps is None:
        steps = res["_steps"][key] = sorted(res["events"].get(key, []))
    T = res["n"]
    k = bisect.bisect_left(steps, ply)
    if k < len(steps) and steps[k] <= T - 1:
        return float(steps[k] - ply + 1), True, steps[k] - ply + 1
    return float(T - ply), False, 0


def cmd_measure(a) -> int:
    import collections
    t0 = time.perf_counter()
    out = out_dir(a.out_dir)
    say = make_say(out, "measure_obs.txt")
    z = pickle.loads((out / "raw.pkl").read_bytes())
    d = TR.load(lambda m: None)
    res_all, roles = d["sample"]["res"], d["roles"]
    for r in res_all:
        r["_steps"] = {}
    lists = build_lists(z)
    n = len(lists)
    g, ply, mover = z["g"], z["ply"], z["mover"]
    say(f"stage D: {n} settled held-out rows of {len(np.unique(g))} games (the games the 78.59 split did not fit; the kinds reading fitted on the others); one run, directions only; the stake is the change of the owner's chance of the result by the kinds reading of 78.69 if the unit is lost; the spread is |delta| of the two readings of 78.69 at the row; bands: 1 {TL.BAND_WORDS[0]}, 2 {TL.BAND_WORDS[1]}, 3 {TL.BAND_WORDS[2]}, 4 {TL.BAND_WORDS[3]}")
    units = np.array([len(r["board"]) for r in z["rows"]])
    edges = [(0, 5), (6, 12), (13, 20), (21, 99)]
    say("ENDGAME COVERAGE of the sample (rows by number of units on the board; human games end early): " + "; ".join(f"{lo}-{hi if hi < 99 else 'more'}: {int(((units >= lo) & (units <= hi)).sum())} rows ({pct(((units >= lo) & (units <= hi)).mean())} pct)" for lo, hi in edges))
    q = np.percentile(z["spread"], [10, 50, 90])
    say(f"the spread |delta| of the rows: quantiles 10/50/90 {q[0]:.4f} / {q[1]:.4f} / {q[2]:.4f}")
    allrows = [r for lst in lists for side in ("mover", "other") for r in lst[side]]
    st = np.array([r["stake"] for r in allrows if r["stake"] is not None])
    q = np.percentile(st, [10, 50, 90])
    say(f"THE ROWS: {len(allrows)} threat rows in {sum(1 for lst in lists if lst['mover'] or lst['other'])} of {n} positions; {sum(1 for r in allrows if r['stake'] is None)} rows have no stake (a unit of the goal kind or a kind outside the head's inputs); stake quantiles 10/50/90 {q[0]:+.4f} / {q[1]:+.4f} / {q[2]:+.4f}; stake above 0 in {pct((st > 0).mean())} pct")
    cc = collections.Counter((r["class"], bool(r["response"])) for r in allrows)
    say("THE ROWS by class and response: " + "; ".join(f"{k[0]} {'with' if k[1] else 'no'} response {v}" for k, v in sorted(cc.items())))
    say(f"THE WIDENING: rows answered in a wider frame than the one that wins: {sum(1 for r in allrows if r['widened'])}, of which lost {sum(1 for r in allrows if r['widened'] and r['class'] == 'lost')}; rows of class lost with no response: {sum(1 for r in allrows if r['class'] == 'lost' and not r['response'])}")
    ec = collections.Counter((r["ended"], r["basis"]) for r in allrows)
    say("HOW THE LINES ENDED (the four reasons of spec 4; the read produces concrete, quiet and out of budget; lost beyond doubt is a verdict of the readings): " + "; ".join(f"{k[0]} (basis {k[1]}) {v}" for k, v in sorted(ec.items())))
    ply_band = TR.quantile_bands(ply.astype(np.float64))
    for way, key in (("SPREAD", "b_spread"), ("RANK", "b_rank")):
        cnt = np.zeros((n, 2, 4))
        for j, lst in enumerate(lists):
            for si, side in enumerate(("mover", "other")):
                for r in lst[side]:
                    if r[key] is not None:
                        cnt[j, si, r[key] - 1] += 1
        tot = cnt.sum(1)
        say(f"BANDS BY {way}: rows per position (mean over all {n} positions; share of positions with at least one): " + "; ".join(f"band {b + 1} {tot[:, b].mean():.3f} ({pct((tot[:, b] > 0).mean())} pct)" for b in range(4)) + "; against the mover's units " + "/".join(f"{cnt[:, 0, b].mean():.3f}" for b in range(4)) + ", against the other side's " + "/".join(f"{cnt[:, 1, b].mean():.3f}" for b in range(4)))
    cells = []
    for j in range(n):
        for si, side in enumerate(("mover", "other")):
            side_abs = int(mover[j]) if side == "mover" else 1 - int(mover[j])
            res = res_all[int(g[j])]
            ex, got, stp = taken_after(res, ("taken", roles[1 - side_abs]), int(ply[j]))
            cells.append((j, si, ex, got, stp))
    cj = np.array([c[0] for c in cells])
    cs = np.array([c[1] for c in cells])
    expo = np.array([c[2] for c in cells])
    obs = np.array([c[3] for c in cells], float)
    steps = np.array([c[4] for c in cells])
    cg = g[cj]
    gb = TR.GameBoot(cg)
    rng = np.random.default_rng(SEED)
    strat = ply_band[cj]
    say(f"THE EVENT: a unit of the side is taken (a taking by the other side, the events of the rules in the sample's record, any kind taken); rate per 100 plies of exposure (to the first such taking at or after the row, or to the end of the record); no horizon; cells (a position and a side) {len(cells)}; with the event {int(obs.sum())}; mean exposure {expo.mean():.2f} plies; overall rate per 100 plies {100 * obs.sum() / expo.sum():.3f}")
    for way, key in (("SPREAD", "b_spread"), ("RANK", "b_rank")):
        top = np.zeros(len(cells), np.int64)
        for k, (j, si, *_r) in enumerate(cells):
            bs = [r[key] for r in lists[j][("mover", "other")[si]] if r[key] is not None]
            top[k] = min(bs) if bs else 0
        labels = ["no row"] + [f"band {b}" for b in range(1, 5)]
        rate_lines(say, f"D BANDS BY {way}, the cell's top band, both sides", gb, obs, expo, steps, top, labels, strat, 5, rng)
        for si, side in enumerate(("the mover's units", "the other side's units")):
            m = cs == si
            gb2 = TR.GameBoot(cg[m])
            rate_lines(say, f"D BANDS BY {way}, units of {side}", gb2, obs[m], expo[m], steps[m], top[m], labels, strat[m], 5, np.random.default_rng(SEED + si))
    cls_top = np.zeros(len(cells), np.int64)
    order = ["lost", "both", "threat", "conditional", "contested"]
    for k, (j, si, *_r) in enumerate(cells):
        cl = [order.index(r["class"]) + 1 for r in lists[j][("mover", "other")[si]]]
        cls_top[k] = min(cl) if cl else 0
    rate_lines(say, "D CLASS of the worst row of the cell (no row, lost, both, threat, conditional, contested), both sides", gb, obs, expo, steps, cls_top, ["no row"] + order, strat, 5, np.random.default_rng(SEED + 9))
    groups = (("all classes", None), ("the classes the read forces or threatens (lost, both, threat, conditional)", {"lost", "both", "threat", "conditional"}), ("contested", {"contested"}))
    for way, key in (("SPREAD", "b_spread"), ("RANK", "b_rank")):
        for b in (2, 4):
            for glabel, gset in groups:
                hit_any, hit_first, rnd, by_reason, gl = [], [], [], collections.Counter(), []
                for j, lst in enumerate(lists):
                    pl = z["rows"][j].get("played")
                    if pl is None:
                        continue
                    for r in lst["mover"]:
                        if r[key] != b or not r["response"] or (gset is not None and r["class"] not in gset):
                            continue
                        ends = [(x[0], x[1]) for x in r["response_ix"]]
                        hit = tuple(pl) in ends
                        hit_any.append(hit)
                        hit_first.append(tuple(pl) == ends[0])
                        rnd.append(len(ends) / max(1, r["resp_tried"]))
                        gl.append(int(g[j]))
                        if hit:
                            by_reason[TL.HOW[[x[2] for x in r["response_ix"] if (x[0], x[1]) == tuple(pl)][0]]] += 1
                if not hit_any:
                    say(f"D RESPONSE PLAYED, bands by {way}, band {b}, {glabel}: no rows with a response against the mover")
                    continue
                ones = np.ones(len(hit_any))
                gbr = TR.GameBoot(np.array(gl))
                z0 = np.zeros(len(hit_any), np.int64)
                p1, dra = gbr.ratio(np.array(hit_any, float), ones, z0, 1)
                p2, drb = gbr.ratio(np.array(hit_first, float), ones, z0, 1)
                p3, drc = gbr.ratio(np.array(rnd), ones, z0, 1)
                say(f"D RESPONSE PLAYED, bands by {way}, band {b} ({TL.BAND_WORDS[b - 1]}), {glabel}: {len(hit_any)} rows against the mover's units that name a response, at positions where the game's move is known: the move played was one of the named responses in {band_txt(p1[0], dra[:, 0], 100.0, '{:.2f}')} pct, the first named in {band_txt(p2[0], drb[:, 0], 100.0, '{:.2f}')} pct; CONTROL a candidate of the read drawn at random would be a response in {band_txt(p3[0], drc[:, 0], 100.0, '{:.2f}')} pct; by the reason of the response played: " + (", ".join(f"{k} {v}" for k, v in by_reason.most_common()) or "none"))
    say(f"stage D wall {time.perf_counter() - t0:.0f} s")
    return 0


# ------------------------------------------------------------------------------------------------------------------------------ carry: the true test of carrying (amendment, point 2)

DEFINITIONS = (("lost or contested (78.69's open claims)", (0, 2, 6)), ("lost only", (0, 2)))


def cmd_carry(a) -> int:
    """Every ply from ply 10 of some held-out games is read (the frame read with its read sets); a claim held at ply n is looked for again at ply n+2 (the same side to move). THE READ SET of a claim is its
    frame's members (absolute side, kind, place) and places; it is UNCHANGED between n and n+2 when every place of it holds the same unit (or none) at both, and, if a frame at n+2 holds the unit, that frame's
    members are the same as at n. A claim whose read set did not change must be held again; a miss is a DEFECT OF THE READ and each kind is printed. The other half: a claim first seen at n+2 whose read set was
    already as it is at n (a miss at n that carrying would have met at the next move)."""
    import collections
    t0 = time.perf_counter()
    out = out_dir(a.out_dir)
    say = make_say(out, "carry_obs.txt")
    d = TR.load(lambda m: None)
    te = TR.test_rows(d, None)
    games = pick_games(d, te, a.games)
    allg, ids = d["allg"], d["sample"]["ids"]
    reader = TL.ThreatReader(a.threads)
    plies = [list(range(10, len(allg[ids[g]]["moves"]) + 1)) for g in games]
    say(f"carrying: {len(games)} held-out games (the first in the md5 order of the ids), every ply from 10 to the end: {sum(len(p) for p in plies)} rows")
    pf = out / "carry_raw.pkl"
    if pf.exists():
        part = pickle.loads(pf.read_bytes())
    else:
        rows, t1 = [], time.perf_counter()
        for c in range(0, len(games), CHUNK):
            ch = list(range(c, min(c + CHUNK, len(games))))
            res = reader.read([(None, allg[ids[games[i]]]["moves"]) for i in ch], [plies[i] for i in ch])
            for r in res["rows"]:
                r["g"] = games[ch[r["game"]]]
                del r["game"]
                r.pop("board_names", None)
            rows += res["rows"]
            say(f"progress: {len(rows)} rows read, {time.perf_counter() - t1:.0f} s")
        part = {"rows": rows, "names": {k: res[k] for k in ("kinds", "roles", "places")}}
        pf.write_bytes(pickle.dumps(part))
    rows = part["rows"]
    by_pos = {(r["g"], r["ply"]): r for r in rows}
    stats = {name: collections.Counter() for name, _c in DEFINITIONS}
    per_game = {name: collections.defaultdict(collections.Counter) for name, _c in DEFINITIONS}

    def members_of(r: dict) -> dict:
        out: dict = {}
        for f in r["frames"]:
            rs = TL.read_set_of(f, r["mover"])
            for m in rs[1]:
                out.setdefault(m, []).append(rs[1])
        return out

    def occupant(r: dict) -> dict:
        return {b[2]: (b[0], b[1]) for b in r["board"]}

    for r in rows:
        r2 = by_pos.get((r["g"], r["ply"] + 2))
        if r2 is None or r2["mover"] != r["mover"]:
            continue
        o1, o2 = occupant(r), occupant(r2)
        cl_n, cl_n2 = TL.claims_of(r), TL.claims_of(r2)
        for name, classes in DEFINITIONS:
            st, pg = stats[name], per_game[name][r["g"]]
            held1 = {(c["owner"], c["kind"], c["place"]): c for c in cl_n if c["class"] in classes}
            held2 = {(c["owner"], c["kind"], c["place"]): c for c in cl_n2 if c["class"] in classes}
            any2 = {(c["owner"], c["kind"], c["place"]): True for c in cl_n2}
            any1 = {(c["owner"], c["kind"], c["place"]): True for c in cl_n}
            mem2, mem1 = members_of(r2), members_of(r)

            def places_of(c):
                return set(c["read_set"][2]) | {pl for _o, _k, pl in c["read_set"][1]}

            def unchanged(c_a, other_frames):
                if any(o1.get(pl) != o2.get(pl) for pl in places_of(c_a)):
                    return False
                if other_frames and not any(f == c_a["members"] for f in other_frames):
                    return False
                return True
            for key, c in held1.items():
                same = unchanged(c, mem2.get(key))
                held = key in held2
                tag = ("unchanged" if same else "changed") + (" held again" if held else " not held again")
                st[tag] += 1
                pg[tag] += 1
                if same and not held:
                    kind = "the unit has a claim of another class at n+2" if key in any2 else ("the unit is a member of a frame at n+2 with no claim on it" if key in mem2 else "no frame at n+2 holds the unit")
                    st["DEFECT: " + kind] += 1
                    pg["DEFECT: " + kind] += 1
            for key, c in held2.items():
                if key in held1:
                    continue
                same = unchanged(c, mem1.get(key))
                tag = "first seen at n+2, read set already as it is at n" if same else "first seen at n+2, read set changed"
                st[tag] += 1
                pg[tag] += 1
                if same:
                    kind = "the unit had a claim of another class at n" if key in any1 else ("the unit was a member of a frame at n with no claim on it" if key in mem1 else "no frame at n held the unit")
                    st["MISS AT n: " + kind] += 1
                    pg["MISS AT n: " + kind] += 1
    for name, _c in DEFINITIONS:
        st, pg = stats[name], per_game[name]
        gl = sorted(pg)
        gb = TR.GameBoot(np.array(gl))
        z1 = np.zeros(len(gl), np.int64)

        def share(num_keys, den_keys):
            nu = np.array([sum(pg[x][k] for k in num_keys) for x in gl], float)
            de = np.array([sum(pg[x][k] for k in den_keys) for x in gl], float)
            p, dr = gb.ratio(nu, de, z1, 1)
            return band_txt(p[0], dr[:, 0], 100.0, "{:.2f}")
        U, C = ("unchanged held again", "unchanged not held again"), ("changed held again", "changed not held again")
        say(f"E(3) CARRYING [{name}]: claims at ply n with a row two plies later (same side to move): {sum(st[k] for k in U + C)}; read set UNCHANGED {sum(st[k] for k in U)}, of which held again {st['unchanged held again']}; read set CHANGED {sum(st[k] for k in C)}, of which held again {st['changed held again']}")
        say(f"E(3) [{name}]: share held again when the read set is UNCHANGED {share(('unchanged held again',), U)} pct; when it CHANGED {share(('changed held again',), C)} pct")
        dk = [k for k in st if k.startswith("DEFECT")]
        say(f"E(3) [{name}] DEFECTS OF THE READ (read set unchanged, claim not held again): " + ("; ".join(f"{k[8:]}: {st[k]}" for k in sorted(dk)) or "none") + f"; share of all unchanged claims {share(('unchanged not held again',), U)} pct")
        F, N = ("first seen at n+2, read set already as it is at n",), ("first seen at n+2, read set changed",)
        say(f"E(3) [{name}] FIRST SEEN AT n+2: {sum(st[k] for k in F + N)}; read set already as it is at n {st[F[0]]} ({share(F, F + N)} pct of them: a miss at n); read set changed {st[N[0]]}")
        mk = [k for k in st if k.startswith("MISS AT n")]
        say(f"E(3) [{name}] MISSES AT n (first seen at n+2 though its read set was already as it is): " + ("; ".join(f"{k[11:]}: {st[k]}" for k in sorted(mk)) or "none"))
    say(f"carrying wall {time.perf_counter() - t0:.0f} s")
    return 0


# ------------------------------------------------------------------------------------------------------------------------------ puzzles: a measure only

PLY_CAP = 120                                                       # the steps left that stand in at a puzzle position: the game's own ply cap of 78.67, a stated choice


def cmd_puzzles(a) -> int:
    """THE 78.45 PUZZLES AS A MEASURE ONLY: nothing of a puzzle enters a fit or a head. The pairs (start text, the solution's first move) come from threat_puzzle_pairs.py. At each position (the side to move is
    the solver) the list is read; the question: is the solution's first move OURS against them in the top band of the other side's list? Two readings of it: the move is made by a unit that bears on a unit of
    theirs in a band-1 row (`from a bearer`), and it takes that unit (`takes it`). The band is by the stake ranked within the position only: the spread of the two readings needs the inputs of the heads at a
    position outside the 9000 games, which are not rebuilt here (said in the note). CONTROL: a unit of ours drawn at random is a bearer of a band-1 row with the share of ours that are."""
    t0 = time.perf_counter()
    out = out_dir(a.out_dir)
    say = make_say(out, "puzzles_obs.txt")
    import json
    pairs = json.loads((out / "puzzle_pairs.json").read_text())
    d, R, _te = load_world(lambda m: None)
    cols = list(d["rows"]["meta"]["kinds"])
    reader = TL.ThreatReader(a.threads)
    raws, names = [], None
    t1 = time.perf_counter()
    for c in range(0, len(pairs), CHUNK):
        ch = pairs[c:c + CHUNK]
        res = reader.read([(p["start"], [p["first"]]) for p in ch], [[0] for _p in ch])
        for r in res["rows"]:
            r["pid"] = ch[r["game"]]["id"]
        raws += res["rows"]
        names = {k: res[k] for k in ("kinds", "roles", "places")}
    kinds = names["kinds"]
    K = np.zeros((len(raws), len(cols)), np.float32)
    for j, r in enumerate(raws):
        for o, k, _p in r["board"]:
            if kinds[k] in cols:
                K[j, cols.index(kinds[k])] += 1.0 if o == r["mover"] else -1.0
    stake = stake_of_kinds(R, K, np.full(len(raws), PLY_CAP, np.int64))
    say(f"puzzles: {len(pairs)} positions read, {len(raws)} rows; solution move known to the engine in {sum(1 for r in raws if r['played'] is not None)}; wall {time.perf_counter() - t1:.0f} s")
    n = top = any_row = from_bearer = takes = 0
    ctrl, ctrl_take = [], []
    for j, r in enumerate(raws):
        if r["played"] is None:
            continue
        n += 1

        def stake_of(kind, own, j=j):
            return float(stake[j, 1 if own else 0, cols.index(kind)]) if kind in cols else None
        lst = TL.threat_rows(r, names, stake_of)
        rk = TL.rank_high_of(lst["mover"] + lst["other"])
        ours = [x for x in lst["other"]]
        for x in ours:
            x["band"] = TL.band_of(x, None, rk.get(id(x), False), "rank")
        any_row += bool(ours)
        tb = [x for x in ours if x["band"] == 1]
        top += bool(tb)
        bearer_places = {p for x in tb for _k, p in x["by"]}
        frm = names["places"][r["played"][0]]
        to = names["places"][r["played"][1]]
        from_bearer += frm in bearer_places
        takes += any(frm in [p for _k, p in x["by"]] and to == x["place"] for x in tb)
        mine = sum(1 for o, _k, _pl in r["board"] if o == r["mover"])
        ctrl.append(len(bearer_places) / max(1, mine))
        ctrl_take.append(sum(1 for x in tb) / max(1, mine))
    say(f"PUZZLES (a measure only, one run): {n} puzzles with a known first move; with at least one row against the other side's units {any_row} ({pct(any_row / n)} pct); with a row in the top band (band 1 by the rank of the stake: high, no response) {top} ({pct(top / n)} pct)")
    say(f"PUZZLES: the solution's first move is made by a unit that bears on a unit of theirs in a band-1 row: {from_bearer} of {n} ({pct(from_bearer / n)} pct); it takes that unit: {takes} of {n} ({pct(takes / n)} pct); CONTROL a unit of ours drawn at random is such a bearer in {pct(float(np.mean(ctrl)))} pct (mean over the puzzles)")
    say(f"puzzles wall {time.perf_counter() - t0:.0f} s")
    return 0


# ------------------------------------------------------------------------------------------------------------------------------ the commands

def main(argv: "list | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("read", "measure", "carry", "puzzles"):
        s = sub.add_parser(name)
        s.add_argument("--out-dir", default=None)
        s.add_argument("--games", type=int, default=1000 if name == "read" else 150)
        s.add_argument("--threads", type=int, default=10)
    a = ap.parse_args(argv)
    return {"read": cmd_read, "measure": cmd_measure, "carry": cmd_carry, "puzzles": cmd_puzzles}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
