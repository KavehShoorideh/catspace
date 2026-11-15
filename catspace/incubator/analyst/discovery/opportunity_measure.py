"""PLAN 78.44, THE OPPORTUNITY NUMBER ALONG A GAME: is it a target worth steering by?  A measurement; nothing is fitted and no player is changed.

THE NUMBER, at a position, for the side to move ("mine" = its units, "theirs" = the other side's), from the Rust question sheet (`SheetModel.games`, clock present):
  A claim is OPEN against a unit when the top verdict of the unit's question is one of OPEN_KINDS (live at level 0, live at level 1, lost where it stands, confirmed, waiting, not settled).
  Closed claims (closed1..3, refuted), idle ones, and a question not yet read (unread) are not open.
  The chance of an open claim is the fitted table's chance for the question's record key (owner, verdict kind, moves, route), read with the table's own prefix fallback, at one horizon
  (CHANCE_HORIZON, the sheet's own default). Worth is the unit's worth in the game's worth table.
  gain = sum over THEIR units of chance x worth; risk = the same over MY units; net = gain - risk.
THE BASELINES from the same sheet: net0 = the same sum over the claims level 0 read as live (what stands lost as the board stands; the chance is the table's cell for (owner, "live0"));
  count = open claims against theirs minus open claims against mine (no chances, no worths); shuffled = the net values permuted among the plies of the same game (fixed seed).
A SECOND NUMBER, labelled: net_d = net over the verdict kinds that say danger only (live0, lost, confirmed: the sheet's own DANGER set), the ones that are close to the board.
THE OUTCOME, per ply t and horizon h: the mover's balance of worth (worth of its units minus the other side's, over the units the sheet read) at ply t+h minus at ply t, from the mover-at-t's
  point of view; split into worth won (the other side's worth at t minus at t+h) and worth lost (the mover's worth at t minus at t+h). A ply with no t+h reading is left out for that h.
  The outcome is the material of the games themselves; no outside engine, labeller or judge enters.

  .venv/bin/python -m catspace.incubator.analyst.discovery.opportunity_measure --make N --seed0 S --out-dir D --variant V [--jobs J]
  .venv/bin/python -m catspace.incubator.analyst.discovery.opportunity_measure --read D --variant V [--out FILE] [--threads T] [--draws 1000]
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import random
import time
from pathlib import Path

import numpy as np

from catspace.io import paths

HORIZONS = (4, 8, 16)
OPEN_KINDS = ("live0", "live1", "lost", "confirmed", "waiting", "unsettled")
DANGER_KINDS = ("live0", "lost", "confirmed")
CHANCE_HORIZON = 4
BANDS = 5
DRAWS = 1000
BOOT_SEED = 4417
SHUFFLE_SEED = 7101
COST_MODEL = (0.002, 0.01, 0.03, 0.06, 0.1)        # the stated seconds per level of the sheet's schedule (as the 78.43 measures), so a number does not depend on the load of the machine
OWN, OTHER = "us", "them"                           # the owner words of a target in the brief: the side to move, the other side


# --------------------------------------------------------------------------------------------------------------------------- the number, from a brief

def chance_of(table, key, horizon: int = CHANCE_HORIZON) -> float:
    return table.chance(tuple(key), horizon)


def number_of(brief: dict, table, horizon: int = CHANCE_HORIZON) -> dict:
    """The number and its baselines at one position, from a brief of the sheet (`SheetModel.game`): {gain, risk, net, net0, count, net_d, mine, theirs, open_any}, where `mine` and
    `theirs` are the worth of the units the sheet read (for the balance), and `open_any` the number of open claims on either side."""
    gain = risk = gain0 = risk0 = gain_d = risk_d = 0.0
    n_mine = n_theirs = 0
    mine_w = theirs_w = 0.0
    for t in brief["targets"]:
        owner, worth, verdict, key, live0 = t[0], float(t[3]), t[4], t[8], bool(t[9])
        mine_side = owner == OWN
        if mine_side:
            mine_w += worth
        else:
            theirs_w += worth
        if verdict in OPEN_KINDS:
            x = chance_of(table, key, horizon) * worth
            if mine_side:
                risk += x
                n_mine += 1
                if verdict in DANGER_KINDS:
                    risk_d += x
            else:
                gain += x
                n_theirs += 1
                if verdict in DANGER_KINDS:
                    gain_d += x
        if live0:
            x = chance_of(table, (key[0], "live0"), horizon) * worth
            if mine_side:
                risk0 += x
            else:
                gain0 += x
    return {"gain": gain, "risk": risk, "net": gain - risk, "net0": gain0 - risk0, "count": float(n_theirs - n_mine), "net_d": gain_d - risk_d,
            "mine": mine_w, "theirs": theirs_w, "open_any": n_mine + n_theirs}


def outcome_at(plies: list, t: int, h: int) -> "tuple | None":
    """(outcome, won, lost) of ply `t` at horizon `h` from a list of {mover, mine, theirs} (None for a position that is over), or None when there is no reading at t or t+h."""
    if t + h >= len(plies) or plies[t] is None or plies[t + h] is None:
        return None
    a, b = plies[t], plies[t + h]
    mover = a["mover"]
    mine_b, theirs_b = (b["mine"], b["theirs"]) if b["mover"] == mover else (b["theirs"], b["mine"])
    won = a["theirs"] - theirs_b
    lost = a["mine"] - mine_b
    return won - lost, won, lost


def shuffled_net(nets: list, rng: random.Random) -> list:
    """The net values permuted among the plies of one game (a ply with no reading keeps None)."""
    idx = [i for i, x in enumerate(nets) if x is not None]
    vals = [nets[i] for i in idx]
    rng.shuffle(vals)
    out = list(nets)
    for i, v in zip(idx, vals):
        out[i] = v
    return out


def table_of(games_plies: list, table, horizon: int = CHANCE_HORIZON) -> dict:
    """The flat arrays of the reading: one row per ply with a reading, the games in order. {game, ply, phase_pos, <features>, out_h, won_h, lost_h}; NaN where there is no outcome."""
    rows: dict = {k: [] for k in ("game", "ply", "gain", "risk", "net", "net0", "count", "net_d", "shuffled", "open_any")}
    for h in HORIZONS:
        for k in ("out", "won", "lost"):
            rows[f"{k}{h}"] = []
    for g, briefs in enumerate(games_plies):
        feats = [None if b is None else number_of(b, table, horizon) for b in briefs]
        plies = [None if f is None else {"mover": b["mover"], "mine": f["mine"], "theirs": f["theirs"]} for f, b in zip(feats, briefs)]
        shuf = shuffled_net([None if f is None else f["net"] for f in feats], random.Random(SHUFFLE_SEED + g))
        for t, f in enumerate(feats):
            if f is None:
                continue
            rows["game"].append(g)
            rows["ply"].append(t)
            for k in ("gain", "risk", "net", "net0", "count", "net_d", "open_any"):
                rows[k].append(f[k])
            rows["shuffled"].append(shuf[t])
            for h in HORIZONS:
                o = outcome_at(plies, t, h)
                for k, v in zip(("out", "won", "lost"), o if o else (np.nan,) * 3):
                    rows[f"{k}{h}"].append(v)
    return {k: np.asarray(v, dtype=float) for k, v in rows.items()}


# --------------------------------------------------------------------------------------------------------------------------- the reading

def rank(x: np.ndarray) -> np.ndarray:
    from scipy.stats import rankdata
    return rankdata(x)


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3:
        return float("nan")
    rx, ry = rank(x), rank(y)
    sx, sy = rx.std(), ry.std()
    if sx == 0 or sy == 0:
        return float("nan")
    return float(((rx - rx.mean()) * (ry - ry.mean())).mean() / (sx * sy))


def partial_spearman(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    """The rank correlation of x and y with the rank of z held out."""
    rxy, rxz, ryz = spearman(x, y), spearman(x, z), spearman(y, z)
    d = (1 - rxz ** 2) * (1 - ryz ** 2)
    return float("nan") if not d > 0 else (rxy - rxz * ryz) / d ** 0.5


def specs(a: dict) -> list:
    """[(label, function of the (sub)table -> correlation)]: every reading that gets an interval."""
    out = []
    third = max(1.0, a["ply"].max() / 3.0) if len(a["ply"]) else 1.0
    for h in HORIZONS:
        o, w, l = f"out{h}", f"won{h}", f"lost{h}"
        ok = lambda d, *names: np.all([~np.isnan(d[n]) for n in names], axis=0)
        for name in ("net", "net0", "count", "shuffled", "net_d", "gain", "risk"):
            out.append((f"h{h} {name} vs outcome", lambda d, n=name, o=o: spearman(d[n][ok(d, o, n)], d[o][ok(d, o, n)])))
        out.append((f"h{h} gain vs worth won", lambda d, w=w: spearman(d["gain"][ok(d, w)], d[w][ok(d, w)])))
        out.append((f"h{h} risk vs worth lost", lambda d, l=l: spearman(d["risk"][ok(d, l)], d[l][ok(d, l)])))
        out.append((f"h{h} net - net0 vs outcome", lambda d, o=o: spearman((d["net"] - d["net0"])[ok(d, o)], d[o][ok(d, o)])))
        out.append((f"h{h} net vs outcome, plies with net0 = 0", lambda d, o=o: spearman(d["net"][ok(d, o) & (d["net0"] == 0)], d[o][ok(d, o) & (d["net0"] == 0)])))
        out.append((f"h{h} net vs outcome, net0 held out (partial rank)", lambda d, o=o: partial_spearman(d["net"][ok(d, o)], d[o][ok(d, o)], d["net0"][ok(d, o)])))
        for p in range(3):
            lo, hi = p * third, (p + 1) * third
            out.append((f"h{h} net vs outcome, phase {p + 1} of 3", lambda d, o=o, lo=lo, hi=hi, last=(p == 2): spearman(
                d["net"][ok(d, o) & (d["ply"] >= lo) & ((d["ply"] < hi) | last)], d[o][ok(d, o) & (d["ply"] >= lo) & ((d["ply"] < hi) | last)])))
    return out


def subset(a: dict, idx: np.ndarray) -> dict:
    return {k: v[idx] for k, v in a.items()}


def bootstrap(a: dict, fns: list, draws: int, seed: int) -> list:
    """For each function the 5th and 95th percentile of its value over `draws` resamplings of the GAMES (with replacement)."""
    rng = np.random.default_rng(seed)
    games = np.unique(a["game"]).astype(int)
    first = {int(g): np.flatnonzero(a["game"] == g) for g in games}
    vals = np.full((draws, len(fns)), np.nan)
    for i in range(draws):
        pick = rng.choice(games, size=len(games), replace=True)
        d = subset(a, np.concatenate([first[int(g)] for g in pick]))
        for j, f in enumerate(fns):
            vals[i, j] = f(d)
    return [(float(np.nanpercentile(vals[:, j], 5)), float(np.nanpercentile(vals[:, j], 95))) if np.isfinite(vals[:, j]).any() else (float("nan"), float("nan")) for j in range(len(fns))]


def bands(a: dict, h: int) -> list:
    """[(band, plies, mean net, mean outcome, mean won, mean lost)] over five equal-count bands of net (ties share a band), plies with an outcome at h only."""
    m = ~np.isnan(a[f"out{h}"])
    net = a["net"][m]
    if len(net) == 0:
        return []
    r = rank(net)
    band = np.minimum(BANDS - 1, ((r - 1) * BANDS / len(net)).astype(int))
    rows = []
    for b in range(BANDS):
        s = band == b
        if s.any():
            rows.append((b + 1, int(s.sum()), float(net[s].mean()), float(a[f"out{h}"][m][s].mean()), float(a[f"won{h}"][m][s].mean()), float(a[f"lost{h}"][m][s].mean())))
    return rows


def report(a: dict, games: int, draws: int, label: str) -> "tuple[list, dict]":
    """The OBS lines of the reading, and the same as data."""
    lines, data = [], {"label": label}
    n = len(a["ply"])
    lines.append(f"OBS opportunity {label}: games {games}, plies read {n}, plies with any open claim {int((a['open_any'] > 0).sum())}, plies with net0 = 0 {int((a['net0'] == 0).sum())}, plies with net = 0 {int((a['net'] == 0).sum())}")
    for h in HORIZONS:
        o = a[f"out{h}"]
        lines.append(f"OBS opportunity {label}: horizon {h}: plies with an outcome {int((~np.isnan(o)).sum())}, mean outcome {np.nanmean(o):.4f}, mean worth won {np.nanmean(a[f'won{h}']):.4f}, mean worth lost {np.nanmean(a[f'lost{h}']):.4f}")
    sp = specs(a)
    t0 = time.perf_counter()
    cis = bootstrap(a, [f for _l, f in sp], draws, BOOT_SEED)
    data["correlations"] = []
    for (name, f), (lo, hi) in zip(sp, cis):
        v = f(a)
        data["correlations"].append({"name": name, "spearman": v, "ci90": [lo, hi]})
        lines.append(f"OBS opportunity {label}: spearman {name}: {v:+.4f}  90% interval over games [{lo:+.4f}, {hi:+.4f}] ({draws} draws)")
    lines.append(f"OBS opportunity {label}: bootstrap seconds {time.perf_counter() - t0:.1f}")
    data["bands"] = {}
    for h in HORIZONS:
        data["bands"][h] = bands(a, h)
        for b, c, mn, mo, mw, ml in data["bands"][h]:
            lines.append(f"OBS opportunity {label}: horizon {h} band {b} of {BANDS} of net: plies {c}, mean net {mn:+.4f}, mean outcome {mo:+.4f}, mean worth won {mw:.4f}, mean worth lost {ml:.4f}")
    return lines, data


# --------------------------------------------------------------------------------------------------------------------------- the games

def random_opening(variant: str, seed: int, plies: int = 4) -> list:
    import cs_engine as CE
    rng = random.Random(seed)
    while True:
        moves = []
        for _ in range(plies):
            legal = CE.state(variant, None, moves)[2]
            moves.append(rng.choice(sorted(legal)))
        if CE.state(variant, None, moves)[3] is None:
            return moves


def make_games(variant: str, n: int, seed0: int, out_dir: str, jobs: int) -> None:
    """N self-play games of sheet_player's `control` config, seeds seed0 .. seed0+n-1, as games.jsonl (the record format of the 78.43 measures)."""
    from catspace.incubator.analyst.discovery import game_store as GS
    from catspace.incubator.analyst.discovery import sheet_player as SP
    out = Path(out_dir)
    store = str(GS.path_of(variant, GS.head(variant))) if GS.head(variant) else None
    specs_ = [{"game": variant, "store": store, "scratch": str(out / "scratch"), "gid": f"{s}", "opening_id": s, "guided": 0, "opening": random_opening(variant, s), "config": SP.CONFIGS["control"],
               "config_name": "control", "plain_seconds": 0.2, "max_plies": 120} for s in range(seed0, seed0 + n)]
    out.mkdir(parents=True, exist_ok=True)
    t0, done = time.perf_counter(), []
    with multiprocessing.get_context("spawn").Pool(jobs) as pool:
        for g in pool.imap_unordered(SP.play_game, specs_):
            done.append(g)
            print(f"OBS progress game {g['gid']} plies {len(g['moves'])} outcome {g['outcome']} ({len(done)} of {n}) {time.perf_counter() - t0:.0f}s", flush=True)
    done.sort(key=lambda g: g["opening_id"])
    with open(out / "games.jsonl", "w") as f:
        for g in done:
            f.write(json.dumps({"variant": variant, "game": g["opening_id"], "start": None, "moves": g["moves"], "outcome": g["outcome"], "unfinished": g["unfinished"], "opening_len": g["opening_len"]}) + "\n")
    print(f"OBS made {len(done)} games in {time.perf_counter() - t0:.0f}s, wall, jobs {jobs}")


def read_games(games_dir: str) -> list:
    return [json.loads(line) for line in (Path(games_dir) / "games.jsonl").read_text().splitlines() if line.strip()]


def briefs_of(variant: str, records: list, threads: int) -> list:
    """The briefs of every ply of every game, in one Rust call across threads (clock present, the stated cost model)."""
    from catspace.incubator.analyst.discovery import question_sheet as QS
    from catspace.incubator.analyst.discovery import sheet_rust as SR
    env = QS.Env(variant)
    worth = QS.store_worth(variant)
    model = SR.model(env, worth)
    params = QS.Params(compare_old=False, cost_model=COST_MODEL, clock="present")
    return SR.games(model, [(r.get("start"), r["moves"]) for r in records], threads, params)


def read_run(games_dir: str, variant: str, threads: int, draws: int, out: "str | None", chances: "str | None") -> list:
    from catspace.incubator.analyst.discovery import question_sheet as QS
    t0 = time.perf_counter()
    records = read_games(games_dir)
    table = QS.load_chances(chances or paths.derived("question_chances", "fit1112.json"))
    briefs = briefs_of(variant, records, threads)
    t_sheet = time.perf_counter() - t0
    a = table_of(briefs, table)
    label = Path(games_dir).name
    lines = [f"OBS opportunity {label}: sheet of {len(records)} games in {t_sheet:.1f} s wall, threads {threads}; unread positions {sum(b is None for g in briefs for b in g)}",
             f"OBS opportunity {label}: open kinds {','.join(OPEN_KINDS)}; danger kinds {','.join(DANGER_KINDS)}; chance horizon {CHANCE_HORIZON}; cost model {COST_MODEL}; clock present"]
    more, data = report(a, len(records), draws, label)
    lines += more
    lines.append(f"OBS opportunity {label}: total wall {time.perf_counter() - t0:.1f} s")
    for line in lines:
        print(line, flush=True)
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps({"lines": lines, "data": data}, indent=1, default=float))
        np.savez_compressed(str(Path(out).with_suffix(".npz")), **a)
    return lines


def main(argv: "list | None" = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--variant", required=True, help="the game's name as the engine knows it")
    ap.add_argument("--make", type=int, default=0, help="make N self-play games")
    ap.add_argument("--seed0", type=int, default=5000)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--read", default=None, help="a games directory to read")
    ap.add_argument("--out", default=None)
    ap.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--draws", type=int, default=DRAWS)
    ap.add_argument("--chances", default=None)
    args = ap.parse_args(argv)
    if args.make:
        if args.seed0 < 5000:
            raise SystemExit("seeds below 5000 are used by earlier sets")
        make_games(args.variant, args.make, args.seed0, args.out_dir or paths.experiment("opportunity_78_44", f"games_s{args.seed0}_n{args.make}"), args.jobs)
    if args.read:
        read_run(args.read, args.variant, args.threads, args.draws, args.out, args.chances)


if __name__ == "__main__":
    main()
