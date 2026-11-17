"""PLAN 78.72 E and F: THE CHECKS ON THE CONCRETE THREAT LIST (a measure script; the list is threat_concrete.py, the print is threat_print.py). SYSTEM_SPEC 8 (the threat list, its four bands) and 4 (open tactics are
carried forward). A measurement in a sandbox on rows that exist: the SAME held-out games and the same frame-read rows as plan 78.70 (artifacts/experiments/threats_78_70/raw.pkl; the kinds reading was fitted on the other
games), one run each, directions only; no player changed, nothing written to rules/learned or data/derived. Game-free: a kind, a role or a place is a word read at run time. No engine evaluation, no labeller, no
external model, no assumed worth (the judge of the lines is the measured record of the kinds). Human games are read only through the existing row builders; no player id is written.

INTERFACE (what this module takes, what it returns, who calls it).
  read(a)       takes the number of held-out games (the first of the 78.70 sample, in its order; `--games`) and returns the path of a pickle with the concrete list (threat_concrete.concrete_lists) of every settled row
                of those games, reduced to what the checks read, with the cost of reading it. Resumable by chunk.
  measure(a)    F on that pickle: the verdicts and the drops counted; per cell (a position and a side) the rate of a unit then being taken for the cell's top verdict (CONFIRMED, UNREAD, EXCHANGE, no row) and for the bands
                (1 high no response ... 4 low with a response, by the spread of the two readings and by the stake ranked within the position), ply held fixed, the shuffled control, bootstrap over games; how often a kept
                response was the move played against a control; endgame coverage.
  carry(a)      E: the list carried from move to move on every ply from ply 10 of some held-out games (the frame-read rows of plan 78.70's carry run): a claim whose read set is unchanged at the position two plies on
                (same side to move) is kept without being asked again; measured: how often the carried verdict is the verdict a fresh read gives, the flicker of the 78.70 E(3) again, and the share of the reading cost saved.
  puzzles(a)    the 78.45 puzzles as a MEASURE only: how often the solution's first move is the first move of a CONFIRMED row of ours, or of a forced end of the game of ours. No puzzle enters a fit.
  Called by: the command line only; nothing calls it.

  .venv/bin/python -m catspace.incubator.analyst.discovery.threat_concrete_measure read    [--games N] [--out-dir D] [--threads K]
  .venv/bin/python -m catspace.incubator.analyst.discovery.threat_concrete_measure measure [--out-dir D]
  .venv/bin/python -m catspace.incubator.analyst.discovery.threat_concrete_measure carry   [--games N] [--out-dir D] [--threads K]
  .venv/bin/python -m catspace.incubator.analyst.discovery.threat_concrete_measure puzzles [--out-dir D] [--threads K]
"""
from __future__ import annotations

import argparse
import collections
import os
import pickle
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")                       # before numpy loads

import numpy as np

from catspace.incubator.analyst.discovery import threat_concrete as TC
from catspace.incubator.analyst.discovery import threat_list as TL
from catspace.incubator.analyst.discovery import threat_measure as TM
from catspace.incubator.analyst.discovery import two_readings as TR
from catspace.io import paths

OUT = ("threats_78_73",)
OLD = ("threats_78_72",)                                            # the outputs of plan 78.72 (read only: the old verdicts and the old print)
SRC = ("threats_78_70",)                                             # the frame-read rows of plan 78.70 (the same held-out games)
CHUNK = 400
SEED = 7272
VERDICT_ORDER = ("CONFIRMED", "UNREAD", "EXCHANGE")                  # the cell's top verdict, the strongest first


def out_dir(p: "str | None" = None) -> Path:
    d = Path(p) if p else Path(paths.experiment(*OUT))
    d.mkdir(parents=True, exist_ok=True)
    return d


def make_say(out: Path, name: str):
    lines: list = []

    def say(m: str) -> None:
        lines.append(f"OBS threats73: {m}")
        print(lines[-1], flush=True)
        (out / name).write_text("\n".join(lines) + "\n")
    return say


def src_dir() -> Path:
    return Path(paths.experiment(*SRC))


# ------------------------------------------------------------------------------------------------------------------------------ reduction

KEEP = ("side", "owner", "kind", "place", "place_ix", "class", "stake", "verdict", "net", "resolution", "line", "outcome", "lead", "two_moves", "deferred", "goal", "ended", "reason", "read_set", "resp_read",
        "resp_unread", "nodes", "pv_disagrees", "depth", "first", "by", "response_ix", "resp_unread_moves", "resp_states")


def reduce_row(r: dict) -> dict:
    out = {k: r.get(k) for k in KEEP}
    out["response"] = [{"first": x["first"], "names": x["names"][:2], "net": x["net"], "how": x["how"], "source": x["source"], "state": x["state"], "loses_nothing": x.get("loses_nothing"), "taker_lost": x.get("taker_lost"),
                        "taker_won": x.get("taker_won"), "resp_loses": x.get("resp_loses")} for x in r.get("response", [])]
    return out


def reduce_pos(c: dict) -> dict:
    return {"mover": [reduce_row(r) for r in c["mover"]], "other": [reduce_row(r) for r in c["other"]], "exchange": [reduce_row(r) for r in c["exchange"]], "dropped": c["dropped"], "resp_read": c["resp_read"],
            "resp_dropped": c["resp_dropped"], "resp_sweeps": c["resp_sweeps"], "unread_rows": c["unread_rows"]}


def band_rows(c: dict, spread: "float | None") -> None:
    """The two bands of every threat row (CONFIRMED or UNREAD, both sides): by the spread, and by the stake ranked within the position."""
    both = c["mover"] + c["other"]
    rk = TL.rank_high_of(both)
    for r in both:
        r["b_spread"] = TL.band_of(r, spread, None, "spread") if spread is not None else None
        r["b_rank"] = TL.band_of(r, None, rk.get(id(r), False), "rank")


# ------------------------------------------------------------------------------------------------------------------------------ read

def cmd_read(a) -> int:
    t0 = time.perf_counter()
    out = out_dir(a.out_dir)
    say = make_say(out, "read_obs.txt")
    z = pickle.loads((src_dir() / "raw.pkl").read_bytes())
    lists = TM.build_lists(z)
    d = TR.load(lambda m: None)
    allg = d["allg"]
    keep_games = set(z["games"][:a.games])
    idx = [j for j in range(len(lists)) if int(z["g"][j]) in keep_games]
    rec = TC.Record.load()
    lines = TC.Lines(rec, threads=a.threads)
    say(f"read: the {len(keep_games)} first held-out games of the 78.70 sample, {len(idx)} settled rows; the judge: the measured record {rec.source}; line depth {lines.depth} plies, node fuse {lines.fuse} per line; sweep of every legal move at {lines.sweep_depth} plies, fuse {lines.sweep_fuse} (a move it cuts is read again with fuse {lines.fuse}); the end of the game within {lines.end_plies} plies, fuse {lines.end_fuse}; {a.threads} threads")
    nchunk = (len(idx) + CHUNK - 1) // CHUNK
    parts = []
    t1 = time.perf_counter()
    for c in range(nchunk):
        pf = out / f"conc_part_{c:04d}.pkl"
        ch = idx[c * CHUNK:(c + 1) * CHUNK]
        if pf.exists():
            part = pickle.loads(pf.read_bytes())
        else:
            items = []
            for j in ch:
                gid = z["sample_ids"][z["games"].index(int(z["g"][j]))]
                items.append({"start": None, "moves": allg[gid]["moves"][:int(z["ply"][j])], "raw": z["rows"][j], "names": z["names"], "list": lists[j]})
            w0 = lines.cost["wall"]
            t2 = time.perf_counter()
            res = TC.concrete_lists(items, lines)
            dt = time.perf_counter() - t2
            reds = []
            for j, cc in zip(ch, res):
                band_rows(cc, float(z["spread"][j]))
                r = reduce_pos(cc)
                for side in ("mover", "other"):
                    for x, y in zip(cc[side], r[side]):
                        y["b_spread"], y["b_rank"] = x.get("b_spread"), x.get("b_rank")
                reds.append(r)
            part = {"j": ch, "pos": reds, "wall": dt, "rust_wall": lines.cost["wall"] - w0}
            pf.write_bytes(pickle.dumps(part))
            say(f"progress: chunk {c + 1} of {nchunk}: {len(ch)} positions, {dt:.1f} s wall over {a.threads} threads (Rust reading {part['rust_wall']:.1f} s; {dt * a.threads / max(1, len(ch)):.3f} s of one thread per position); run so far {time.perf_counter() - t1:.0f} s")
        parts.append(part)
    j_all = [j for p in parts for j in p["j"]]
    pos_all = [x for p in parts for x in p["pos"]]
    pk = out / "conc.pkl"
    pk.write_bytes(pickle.dumps({"j": j_all, "pos": pos_all, "wall": sum(p["wall"] for p in parts), "threads": a.threads}))
    n = len(pos_all)
    vc = collections.Counter(r["verdict"] for p in pos_all for side in ("mover", "other") for r in p[side])
    ec = len([r for p in pos_all for r in p["exchange"]])
    dc: collections.Counter = collections.Counter()
    for p in pos_all:
        dc.update(p["dropped"])
    goal = collections.Counter((r["verdict"], r["side"]) for p in pos_all for side in ("mover", "other") for r in p[side] if r.get("goal"))
    two = sum(1 for p in pos_all for side in ("mover", "other") for r in p[side] if r.get("two_moves"))
    old = sum(len(lists[j]["mover"]) + len(lists[j]["other"]) for j in j_all)
    say(f"done: {n} positions; wall {sum(p['wall'] for p in parts):.0f} s over {a.threads} threads = {sum(p['wall'] for p in parts) * a.threads / n:.3f} s of one thread per position; saved {pk}")
    say(f"THE ROWS: the 78.70 list had {old} rows at these positions; the concrete list has threat rows " + "; ".join(f"{k} {v}" for k, v in sorted(vc.items())) + f"; EXCHANGE rows apart {ec}; of the threat rows, threats in two moves {two}")
    say("DROPPED (counted): " + "; ".join(f"{k}: {v}" for k, v in sorted(dc.items())))
    say(f"THE GOAL (a forced end of the game within the plies asked, for either side): rows " + ("; ".join(f"{k[0]} against the {k[1]}'s units {v}" for k, v in sorted(goal.items())) or "none"))
    say(f"THE RESPONSES: read {sum(p['resp_read'] for p in pos_all)}, dropped {sum(p['resp_dropped'] for p in pos_all)}; sweeps of every legal move {sum(p['resp_sweeps'] for p in pos_all)}; rows UNREAD {sum(p['unread_rows'] for p in pos_all)}")
    cr = [r for p in pos_all for side in ("mover", "other") for r in p[side] if r["verdict"] == "CONFIRMED" and not r.get("goal") and not r.get("deferred")]
    say(f"THE RESPONSES by row: of {len(cr)} CONFIRMED rows with a response owed, with at least one kept response {sum(1 for r in cr if r['response'])}, none {sum(1 for r in cr if not r['response'])}")
    pvd = sum(1 for p in pos_all for side in ("mover", "other") for r in p[side] if r.get("pv_disagrees"))
    say(f"THE LINE AGAINST ITS VALUE: rows whose line's ledger in the record's worths differs from the search's value: {pvd}")
    units = np.array([len(z["rows"][j]["board"]) for j in j_all])
    edges = [(0, 5), (6, 12), (13, 20), (21, 99)]
    say("ENDGAME COVERAGE of the sample (rows by number of units on the board; human games end early): " + "; ".join(f"{lo}-{hi if hi < 99 else 'more'}: {int(((units >= lo) & (units <= hi)).sum())} rows ({100 * ((units >= lo) & (units <= hi)).mean():.2f} pct)" for lo, hi in edges))
    say(f"wall {time.perf_counter() - t0:.0f} s")
    return 0


# ------------------------------------------------------------------------------------------------------------------------------ measure: F

def cmd_measure(a) -> int:
    t0 = time.perf_counter()
    out = out_dir(a.out_dir)
    say = make_say(out, "measure_obs.txt")
    z = pickle.loads((src_dir() / "raw.pkl").read_bytes())
    cz = pickle.loads((out / "conc.pkl").read_bytes())
    j_all, pos = cz["j"], cz["pos"]
    d = TR.load(lambda m: None)
    res_all, roles = d["sample"]["res"], d["roles"]
    for r in res_all:
        r["_steps"] = {}
    g = np.array([int(z["g"][j]) for j in j_all])
    ply = np.array([int(z["ply"][j]) for j in j_all])
    mover = np.array([int(z["mover"][j]) for j in j_all])
    n = len(pos)
    say(f"stage F: {n} settled held-out rows of {len(np.unique(g))} games (the games the 78.59 split did not fit); one run, directions only; the event: a unit of the side is taken (as 78.70); bands: 1 {TL.BAND_WORDS[0]}, 2 {TL.BAND_WORDS[1]}, 3 {TL.BAND_WORDS[2]}, 4 {TL.BAND_WORDS[3]}; a threat row is CONFIRMED or UNREAD, EXCHANGE rows are apart")
    ply_band = TR.quantile_bands(ply.astype(np.float64))
    cells = []
    for k in range(n):
        for si, side in enumerate(("mover", "other")):
            side_abs = int(mover[k]) if side == "mover" else 1 - int(mover[k])
            res = res_all[int(g[k])]
            ex, got, stp = TM.taken_after(res, ("taken", roles[1 - side_abs]), int(ply[k]))
            cells.append((k, si, ex, got, stp))
    cj = np.array([c[0] for c in cells])
    cs = np.array([c[1] for c in cells])
    expo = np.array([c[2] for c in cells])
    obs = np.array([c[3] for c in cells], float)
    steps = np.array([c[4] for c in cells])
    cg = g[cj]
    gb = TR.GameBoot(cg)
    strat = ply_band[cj]
    say(f"THE EVENT: cells (a position and a side) {len(cells)}; with the event {int(obs.sum())}; mean exposure {expo.mean():.2f} plies; overall rate per 100 plies {100 * obs.sum() / expo.sum():.3f}")
    # ---- by the cell's top verdict
    top = np.zeros(len(cells), np.int64)                          # 0 no row of any verdict; 1 EXCHANGE only; 2 UNREAD; 3 CONFIRMED
    for c, (k, si, *_r) in enumerate(cells):
        side = ("mover", "other")[si]
        vs = {r["verdict"] for r in pos[k][side]}
        owner_abs = int(mover[k]) if side == "mover" else 1 - int(mover[k])
        ex = any(r["owner"] == owner_abs for r in pos[k]["exchange"])
        top[c] = 3 if "CONFIRMED" in vs else (2 if "UNREAD" in vs else (1 if ex else 0))
    labels = ["no row", "EXCHANGE only", "UNREAD", "CONFIRMED"]
    TM.rate_lines(say, "F BY VERDICT, the cell's top verdict, both sides", gb, obs, expo, steps, top, labels, strat, 5, np.random.default_rng(SEED))
    for si, side in enumerate(("the mover's units", "the other side's units")):
        m = cs == si
        TM.rate_lines(say, f"F BY VERDICT, units of {side}", TR.GameBoot(cg[m]), obs[m], expo[m], steps[m], top[m], labels, strat[m], 5, np.random.default_rng(SEED + 1 + si))
    # ---- by the goal: a forced end of the game against the side
    gl = np.zeros(len(cells), np.int64)
    for c, (k, si, *_r) in enumerate(cells):
        gl[c] = 1 if any(r.get("goal") for r in pos[k][("mover", "other")[si]]) else 0
    if gl.sum() > 0:
        TM.rate_lines(say, "F A FORCED END OF THE GAME against the side (the goal rows), both sides", gb, obs, expo, steps, gl, ["no goal row", "a goal row"], strat, 5, np.random.default_rng(SEED + 5))
    else:
        say("F A FORCED END OF THE GAME against the side: no goal row in the sample")
    # ---- by band
    for way, key in (("SPREAD", "b_spread"), ("RANK", "b_rank")):
        cnt = np.zeros((n, 2, 4))
        for k in range(n):
            for si, side in enumerate(("mover", "other")):
                for r in pos[k][side]:
                    if r[key] is not None:
                        cnt[k, si, r[key] - 1] += 1
        tot = cnt.sum(1)
        say(f"BANDS BY {way}: rows per position (mean over all {n} positions; share of positions with at least one): " + "; ".join(f"band {b + 1} {tot[:, b].mean():.3f} ({100 * (tot[:, b] > 0).mean():.2f} pct)" for b in range(4)))
        for variant, skip_goal in (("", False), (" WITHOUT the goal rows (the game itself ending is no taking of a unit)", True)):
            topb = np.zeros(len(cells), np.int64)
            for c, (k, si, *_r) in enumerate(cells):
                bs = [r[key] for r in pos[k][("mover", "other")[si]] if r[key] is not None and not (skip_goal and r.get("goal"))]
                topb[c] = min(bs) if bs else 0
            lab = ["no row"] + [f"band {b}" for b in range(1, 5)]
            TM.rate_lines(say, f"F BANDS BY {way}{variant}, the cell's top band, both sides", gb, obs, expo, steps, topb, lab, strat, 5, np.random.default_rng(SEED + 10))
            pt, _dr = gb.ratio(obs.astype(float), expo, topb, 5)
            ps, _ds = TM.standardised(gb, obs.astype(float), expo, topb, 5, strat, 5)
            say(f"F BANDS BY {way}{variant}: order of the bands by rate (highest first, point estimates): " + " > ".join(lab[i] for i in np.argsort(-pt)) + f"; band 1 is {'the highest' if np.argmax(pt[1:]) == 0 else 'not the highest'} of the four bands; with ply held fixed: " + " > ".join(lab[i] for i in np.argsort(-ps)) + f"; band 1 is {'the highest' if np.argmax(ps[1:]) == 0 else 'not the highest'}")
    # ---- the response played
    for way, key in (("SPREAD", "b_spread"), ("RANK", "b_rank")):
        for glabel, cond in (("all CONFIRMED rows with a kept response", lambda r: r["verdict"] == "CONFIRMED" and r["response"] and not r.get("goal")),
                             ("CONFIRMED rows of the first two bands (high)", lambda r: r["verdict"] == "CONFIRMED" and r["response"] and not r.get("goal") and r[key] in (1, 2))):
            hit, first, rnd, gl2 = [], [], [], []
            for k in range(n):
                pl = z["rows"][j_all[k]].get("played")
                if pl is None:
                    continue
                for r in pos[k]["mover"]:
                    if not cond(r):
                        continue
                    ends = [tuple(x["first"]) for x in r["response"]]
                    hit.append(tuple(pl) in ends)
                    first.append(tuple(pl) == ends[0])
                    rnd.append(len(ends) / max(1, r["resp_read"]))
                    gl2.append(int(g[k]))
            if not hit:
                say(f"F RESPONSE PLAYED, bands by {way}, {glabel}: no rows")
                continue
            ones = np.ones(len(hit))
            gbr = TR.GameBoot(np.array(gl2))
            z0 = np.zeros(len(hit), np.int64)
            pa, da = gbr.ratio(np.array(hit, float), ones, z0, 1)
            pb, db = gbr.ratio(np.array(first, float), ones, z0, 1)
            pc, dc = gbr.ratio(np.array(rnd), ones, z0, 1)
            say(f"F RESPONSE PLAYED, bands by {way}, {glabel}: {len(hit)} rows against the mover's units; the move played was one of the kept responses in {TM.band_txt(pa[0], da[:, 0], 100.0, '{:.2f}')} pct, the first kept in {TM.band_txt(pb[0], db[:, 0], 100.0, '{:.2f}')} pct; CONTROL a candidate read, drawn at random, is a kept response in {TM.band_txt(pc[0], dc[:, 0], 100.0, '{:.2f}')} pct")
    say(f"stage F wall {time.perf_counter() - t0:.0f} s")
    return 0


# ------------------------------------------------------------------------------------------------------------------------------ carry: E

def cmd_carry(a) -> int:
    """Every ply from ply 10 of some held-out games is read (the frame-read rows of 78.70's carry run, the same games and plies) and made concrete. A claim read at ply n is looked for again at ply n+2 (the same side
    to move). Its READ SET is unchanged when every place of it holds the same unit (or none) at both plies and, if a frame at n+2 holds the unit, that frame's members are the same as at n (78.70 E(3)). The list
    CARRIED from n to n+2 keeps, for every claim whose read set is unchanged, the verdict read at n without asking again, and asks again only the claims whose read set changed."""
    t0 = time.perf_counter()
    out = out_dir(a.out_dir)
    say = make_say(out, "carry_obs.txt")
    part = pickle.loads((src_dir() / "carry_raw.pkl").read_bytes())
    d = TR.load(lambda m: None)
    allg, ids = d["allg"], d["sample"]["ids"]
    rows, names = part["rows"], part["names"]
    games = sorted({r["g"] for r in rows})[:a.games]
    rows = [r for r in rows if r["g"] in set(games)]
    say(f"carrying: {len(games)} held-out games (the same as 78.70's carry run, every ply from 10 to the end): {len(rows)} rows")
    rec = TC.Record.load()
    lines = TC.Lines(rec, threads=a.threads)
    pf = out / "carry_conc.pkl"
    if pf.exists():
        res = pickle.loads(pf.read_bytes())
    else:
        res = []
        t1 = time.perf_counter()
        for c in range(0, len(rows), CHUNK):
            items = []
            for r in rows[c:c + CHUNK]:
                lst = TL.threat_rows(r, names, None)
                items.append({"start": None, "moves": allg[ids[r["g"]]]["moves"][:r["ply"]], "raw": r, "names": names, "list": lst})
            cc = TC.concrete_lists(items, lines)
            res += [reduce_pos(x) for x in cc]
            say(f"progress: {len(res)} rows made concrete, {time.perf_counter() - t1:.0f} s")
        pf.write_bytes(pickle.dumps(res))
    by_pos = {(r["g"], r["ply"]): (r, c) for r, c in zip(rows, res)}
    pos_n = len(rows)
    tot_nodes = sum(r["nodes"] or 0 for c in res for side in ("mover", "other") for r in c[side]) + sum(r["nodes"] or 0 for c in res for r in c["exchange"])
    say(f"carrying: the whole read of the {pos_n} rows cost {tot_nodes} line-search nodes in the rows' own readings and {lines.cost['wall'] if lines.cost['calls'] else 0.0:.0f} s of Rust wall this run (zero when the pickle was kept)")

    def occupant(r: dict) -> dict:
        return {b[2]: (b[0], b[1]) for b in r["board"]}

    def places_of(c: dict) -> set:
        return set(c["read_set"][2]) | {pl for _o, _k, pl in c["read_set"][1]}

    def members_of(r: dict) -> dict:
        out_: dict = {}
        for f in r["frames"]:
            rs = TL.read_set_of(f, r["mover"])
            for m in rs[1]:
                out_.setdefault(m, []).append(rs[1])
        return out_

    def verdict_map(c: dict) -> dict:
        m: dict = {}
        for side in ("mover", "other"):
            for r in c[side]:
                if not r.get("goal"):
                    m[(r["owner"], r["kind"], r["place_ix"])] = r
        for r in c["exchange"]:
            m[(r["owner"], r["kind"], r["place_ix"])] = r
        return m
    st = collections.Counter()
    diff_cat: collections.Counter = collections.Counter()
    diff_pair: collections.Counter = collections.Counter()
    per_game: dict = collections.defaultdict(collections.Counter)
    saved_nodes = 0
    fresh_nodes = 0
    for (g, p), (r, c) in by_pos.items():
        nxt = by_pos.get((g, p + 2))
        if nxt is None or nxt[0]["mover"] != r["mover"]:
            continue
        rn, cn = nxt
        o1, o2 = occupant(r), occupant(rn)
        mem2 = members_of(rn)
        v1, v2 = verdict_map(c), verdict_map(cn)

        def unchanged(claim: dict, other_frames) -> bool:
            if any(o1.get(pl) != o2.get(pl) for pl in places_of(claim)):
                return False
            if other_frames and not any(f == claim["members"] for f in other_frames):
                return False
            return True
        for key, row in v1.items():
            ck = key
            claim = {"read_set": row["read_set"], "members": row["read_set"][1]}
            same = unchanged(claim, mem2.get(ck))
            fresh = v2.get(key)
            agree = fresh is not None and fresh["verdict"] == row["verdict"]
            tag = ("unchanged" if same else "changed")
            st[tag] += 1
            per_game[g][tag] += 1
            st[tag + " verdict agrees"] += int(agree)
            per_game[g][tag + " verdict agrees"] += int(agree)
            held = fresh is not None
            st[tag + " held again"] += int(held)
            per_game[g][tag + " held again"] += int(held)
            if same:
                saved_nodes += int(fresh["nodes"] or 0) if fresh is not None else 0
                if fresh is not None and fresh["verdict"] != row["verdict"]:                      # PLAN 78.73 D: the two lines of a row whose read set is unchanged and whose verdicts differ
                    l1, l2 = list(row["line"] or []), list(fresh["line"] or [])
                    k = 0
                    while k < min(len(l1), len(l2)) and l1[k] == l2[k]:
                        k += 1
                    if l1 == l2:
                        cat = "same line, another verdict"
                    elif k == min(len(l1), len(l2)):
                        cat = "one line is the start of the other (they differ only beyond where the shorter one ends)"
                    elif k == 0:
                        cat = "the first move differs"
                    else:
                        cat = "the lines part inside both"
                    diff_cat[cat] += 1
                    diff_pair[(row["verdict"], fresh["verdict"])] += 1
        fresh_nodes += sum(int(x["nodes"] or 0) for x in v2.values())
        carried = [k for k in v2 if k in v1 and unchanged({"read_set": v1[k]["read_set"], "members": v1[k]["read_set"][1]}, mem2.get(k))]
        st["rows at n+2"] += len(v2)
        st["rows at n+2 that could be carried (read set unchanged, present at n)"] += len(carried)
        per_game[g]["rows at n+2"] += len(v2)
        per_game[g]["carried"] += len(carried)
    gl = sorted(per_game)
    gb = TR.GameBoot(np.array(gl))
    z1 = np.zeros(len(gl), np.int64)

    def share(num_keys, den_keys):
        nu = np.array([sum(per_game[x][k] for k in num_keys) for x in gl], float)
        de = np.array([sum(per_game[x][k] for k in den_keys) for x in gl], float)
        p, dr = gb.ratio(nu, de, z1, 1)
        return TM.band_txt(p[0], dr[:, 0], 100.0, "{:.2f}")
    nd = sum(diff_cat.values())
    say(f"D STABILITY: rows with an unchanged read set and a row again at n+2 whose fresh verdict differs from the carried one: {nd}; by verdict pair " + "; ".join(f"{a_} -> {b_}: {v}" for (a_, b_), v in sorted(diff_pair.items())))
    say("D STABILITY: how the two lines (the one read at n, the fresh one at n+2) differ: " + "; ".join(f"{k_}: {v} ({100 * v / max(1, nd):.1f} pct)" for k_, v in sorted(diff_cat.items(), key=lambda kv: -kv[1])))
    say(f"D STABILITY: the share that differs only beyond the depth cut (one line the start of the other, or the same line): {100 * (diff_cat['one line is the start of the other (they differ only beyond where the shorter one ends)'] + diff_cat['same line, another verdict']) / max(1, nd):.1f} pct; the share where a move inside the lines' reach, with the read set unchanged, changed the line (the lines part, or the first move differs): {100 * (diff_cat['the lines part inside both'] + diff_cat['the first move differs']) / max(1, nd):.1f} pct")
    say(f"E CARRYING of the concrete list: claims read at ply n (a row of any verdict) with the same side to move two plies on: read set UNCHANGED {st['unchanged']}, CHANGED {st['changed']}")
    say(f"E: of the claims with the read set UNCHANGED, the unit has a row again at n+2 in {share(('unchanged held again',), ('unchanged',))} pct, and the CARRIED verdict (the one read at n) is the verdict a fresh read gives at n+2 in {share(('unchanged verdict agrees',), ('unchanged',))} pct")
    say(f"E: of the claims with the read set CHANGED (asked again), the verdict read at n is the fresh verdict at n+2 in {share(('changed verdict agrees',), ('changed',))} pct; the unit has a row again in {share(('changed held again',), ('changed',))} pct")
    say(f"E TIME SAVED: of the rows at n+2, {st['rows at n+2 that could be carried (read set unchanged, present at n)']} of {st['rows at n+2']} are carried without being asked again ({share(('carried',), ('rows at n+2',))} pct of the rows); the reading cost (line-search nodes) the carried rows would have cost is {saved_nodes} of {fresh_nodes} ({100 * saved_nodes / max(1, fresh_nodes):.2f} pct)")
    say(f"carrying wall {time.perf_counter() - t0:.0f} s")
    return 0


# ------------------------------------------------------------------------------------------------------------------------------ puzzles: a measure only

def cmd_puzzles(a) -> int:
    """THE 78.45 PUZZLES AS A MEASURE ONLY: nothing of a puzzle enters a fit or a head. The pairs (start text, the solution's first move) come from threat_puzzle_pairs.py (the file of 78.70). At each position (the side to
    move is the solver) the concrete list is made. The questions: is the solution's first move the first move of a CONFIRMED row of ours (against their units), and is it the first move of a forced end of the game of ours.
    CONTROL: the share of our legal first moves of such rows among our units that can move, drawn at random (the share of our units that are such first movers)."""
    import json
    t0 = time.perf_counter()
    out = out_dir(a.out_dir)
    say = make_say(out, "puzzles_obs.txt")
    pairs = json.loads((src_dir() / "puzzle_pairs.json").read_text())
    from catspace.incubator.analyst.discovery import threat_measure as TMm
    reader = TL.ThreatReader(a.threads)
    rec = TC.Record.load()
    lines = TC.Lines(rec, threads=a.threads)
    items, raws, names = [], [], None
    t1 = time.perf_counter()
    for c in range(0, len(pairs), 50):
        ch = pairs[c:c + 50]
        res = reader.read([(p["start"], [p["first"]]) for p in ch], [[0] for _p in ch])
        names = {k: res[k] for k in ("kinds", "roles", "places")}
        for r in res["rows"]:
            p = ch[r["game"]]
            raws.append((p, r))
    for p, r in raws:
        lst = TL.threat_rows(r, names, None)
        items.append({"start": p["start"], "moves": [], "raw": r, "names": names, "list": lst})
    cc = TC.concrete_lists(items, lines)
    say(f"puzzles: {len(pairs)} positions read and made concrete, {len(items)} rows; wall {time.perf_counter() - t1:.0f} s")
    n = any_row = conf = from_first = move_first = goal_any = goal_first = 0
    ctrl = []
    for (p, r), c in zip(raws, cc):
        if r["played"] is None:
            continue
        n += 1
        ours = [x for x in c["other"] if not x.get("goal")]
        cf = [x for x in ours if x["verdict"] == "CONFIRMED"]
        any_row += bool(ours)
        conf += bool(cf)
        firsts = [tuple(x["first"]) for x in cf]
        move_first += tuple(r["played"]) in firsts
        from_first += r["played"][0] in {f[0] for f in firsts}
        gs = [x for x in c["other"] if x.get("goal")]
        goal_any += bool(gs)
        goal_first += any(tuple(x["first"]) == tuple(r["played"]) for x in gs)
        mine = sum(1 for o, _k, _pl in r["board"] if o == r["mover"])
        ctrl.append(len({f[0] for f in firsts}) / max(1, mine))
    say(f"PUZZLES (a measure only, one run): {n} puzzles with a known first move; with at least one row against the other side's units {any_row} ({100 * any_row / n:.2f} pct); with a CONFIRMED row of ours {conf} ({100 * conf / n:.2f} pct)")
    say(f"PUZZLES: the solution's first move is the first move of a CONFIRMED row of ours: {move_first} of {n} ({100 * move_first / n:.2f} pct); it is made by a unit that is the first mover of one: {from_first} of {n} ({100 * from_first / n:.2f} pct); CONTROL a unit of ours drawn at random is such a first mover in {100 * float(np.mean(ctrl)):.2f} pct (mean over the puzzles)")
    say(f"PUZZLES: a forced end of the game of ours is found in {goal_any} of {n} ({100 * goal_any / n:.2f} pct) and its first move is the solution's in {goal_first} of {n} ({100 * goal_first / n:.2f} pct)")
    say(f"puzzles wall {time.perf_counter() - t0:.0f} s")
    return 0



# ------------------------------------------------------------------------------------------------------------------------------ plan 78.73 A: the verdicts against those of 78.72

def cmd_verdicts(a) -> int:
    """A: the rows of the held-out games read now (conc.pkl of this plan) against the same rows of 78.72 (the old pickle; a row is the position, the owner, the kind and the place): how many EXCHANGE rows of 78.72 change verdict
    once like for like cancels, and every other change of verdict. A row absent from the lists now was REFUTED (counted as dropped)."""
    out = out_dir(a.out_dir)
    say = make_say(out, "verdicts_obs.txt")
    old = pickle.loads((Path(paths.experiment(*OLD)) / "conc.pkl").read_bytes())
    new = pickle.loads((out / "conc.pkl").read_bytes())
    assert old["j"] == new["j"], "the two reads are not of the same positions"

    def vmap(p: dict) -> dict:
        m = {}
        for side in ("mover", "other"):
            for r in p[side]:
                if not r.get("goal"):
                    m[(r["owner"], r["kind"], r["place_ix"], bool(r.get("two_moves")))] = r["verdict"]
        for r in p["exchange"]:
            m[(r["owner"], r["kind"], r["place_ix"], bool(r.get("two_moves")))] = "EXCHANGE"
        return m
    tr = collections.Counter()
    for po, pn in zip(old["pos"], new["pos"]):
        vo, vn = vmap(po), vmap(pn)
        for k, v in vo.items():
            tr[(v, vn.get(k, "REFUTED (dropped)"))] += 1
        for k, v in vn.items():
            if k not in vo:
                tr[("REFUTED (dropped)", v)] += 1
    n_old_ex = sum(v for (o, _n), v in tr.items() if o == "EXCHANGE")
    ch = sum(v for (o, nw), v in tr.items() if o == "EXCHANGE" and nw != "EXCHANGE")
    say(f"A: the {len(old['pos'])} positions of the 1000 held-out games; EXCHANGE rows of 78.72: {n_old_ex}; of them the verdict CHANGES for {ch} ({100 * ch / max(1, n_old_ex):.2f} pct)")
    say("A: every transition of verdict, 78.72 -> 78.73: " + "; ".join(f"{o} -> {nw}: {v}" for (o, nw), v in sorted(tr.items())))
    pvd = sum(1 for p in new["pos"] for side in ("mover", "other") for r in p[side] if r.get("pv_disagrees")) + sum(1 for p in new["pos"] for r in p["exchange"] if r.get("pv_disagrees"))
    say(f"A: rows whose ledger (after like for like) differs from the search's value: {pvd}")
    return 0


# ------------------------------------------------------------------------------------------------------------------------------ plan 78.73 C: the sweep's replies against a read at the rows' depth

def cmd_replies(a) -> int:
    """C: the twenty printed positions. Every response of a CONFIRMED row that the sweep found (the frame read's answers are read at the rows' depth already) is read again at the rows' depth (6 plies after the first move,
    the sweep read 4): how many first replies change (the second move of the line), for the printed ones (the first three kept of each row) and for all kept. Then the whole list is made with the sweep at the rows' depth and
    its fuse, and the time of the two is printed."""
    from catspace.incubator.analyst.discovery import threat_print as TP
    out = out_dir(a.out_dir)
    say = make_say(out, "replies_obs.txt")
    z, lists, allg, picked, _n = TP.load_twenty()
    rec = TC.Record.load()
    l4 = TC.Lines(rec, threads=1)
    l6 = TC.Lines(rec, threads=1, sweep_depth=TC.DEPTH, sweep_fuse=TC.FUSE)
    st = collections.Counter()
    t4, t6 = [], []
    for k, (_ph, j) in enumerate(picked, 1):
        gid = z["sample_ids"][z["games"].index(int(z["g"][j]))]
        moves = allg[gid]["moves"][:int(z["ply"][j])]
        item = {"start": None, "moves": moves, "raw": z["rows"][j], "names": z["names"], "list": lists[j]}
        t0 = time.perf_counter()
        res_short = TC.concrete_lists([item], l4)[0]
        t4.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        res_deep = TC.concrete_lists([item], l6)[0]
        t6.append(time.perf_counter() - t0)
        games = [(None, moves)]
        for side in ("mover", "other"):
            for row in res_short[side]:
                if row.get("goal") or row["verdict"] != "CONFIRMED" or row.get("deferred"):
                    continue
                for rank, x in enumerate(row["response"]):
                    if x["source"] != "the sweep" or len(x["names"]) < 2:
                        continue
                    ro = row["_ro"]
                    deep = l4.ask(games, [(0, 6, ro["owner_p"], 0, x["first"][0], x["first"][1], row["place_ix"])])[0]
                    mine = [af for af in (deep["afters"] if deep["ok"] else []) if af["whole"]["names"][0] == x["names"][0]]
                    st["responses all"] += 1
                    st["responses printed"] += int(rank < 3)
                    if not mine:
                        st["not read again"] += 1
                        continue
                    w = mine[0]["whole"]
                    if w["cut"] or len(w["names"]) < 2:
                        st["read again but cut or one move"] += 1
                        continue
                    diff = w["names"][1] != x["names"][1]
                    st["first reply changed all"] += int(diff)
                    if rank < 3:
                        st["first reply changed printed"] += int(diff)
        sets = []
        for c in (res_short, res_deep):
            sets.append({(r["owner"], r["place_ix"], tuple(x["first"])) for side in ("mover", "other") for r in c[side] if not r.get("goal") for x in r.get("response", [])})
        st["kept responses (row, move) with the sweep at 4 plies"] += len(sets[0])
        st["kept responses (row, move) with the sweep at 6 plies"] += len(sets[1])
        st["of them in both"] += len(sets[0] & sets[1])
    say(f"C: {len(picked)} positions; sweep responses of CONFIRMED rows read again at {l4.depth} plies after the first move (the sweep: {l4.sweep_depth}): printed (first three kept of a row) {st['responses printed']}, all kept {st['responses all']}")
    say(f"C: FIRST REPLIES CHANGED: printed {st['first reply changed printed']} of {st['responses printed']}; all kept {st['first reply changed all']} of {st['responses all']}; not read again {st['not read again']}; read again but cut or a one-move line {st['read again but cut or one move']}")
    say(f"C: the lists with the sweep at {l6.sweep_depth} plies: kept (row, move) pairs {st['kept responses (row, move) with the sweep at 6 plies']} against {st['kept responses (row, move) with the sweep at 4 plies']} at 4, in both {st['of them in both']}")
    f = lambda t: f"median {sorted(t)[len(t) // 2]:.2f} s, mean {sum(t) / len(t):.2f} s, slowest {max(t):.2f} s, total {sum(t):.1f} s"
    say(f"C: TIME of the concrete read per position (one thread): sweep at 4 plies (fuse {l4.sweep_fuse}, then {l4.fuse} for a cut move): {f(t4)}; sweep at {l6.sweep_depth} plies (fuse {l6.sweep_fuse}): {f(t6)}")
    return 0


# ------------------------------------------------------------------------------------------------------------------------------ the commands

def main(argv: "list | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("read", "measure", "carry", "puzzles", "verdicts", "replies"):
        s = sub.add_parser(name)
        s.add_argument("--out-dir", default=None)
        s.add_argument("--games", type=int, default=1000 if name == "read" else 150)
        s.add_argument("--threads", type=int, default=10)
    a = ap.parse_args(argv)
    return {"read": cmd_read, "measure": cmd_measure, "carry": cmd_carry, "puzzles": cmd_puzzles, "verdicts": cmd_verdicts, "replies": cmd_replies}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
