"""PLAN 78.48, THE OPPORTUNITY NUMBER WITH THE FORCING READ: do its claims foretell what is won, beyond the next move?  A measurement; nothing is wired into play, nothing here is read by any player.

A. THE TERMS.  For every position of a game the Rust forcing read (rust/engine/tactics/src/sheet/forcing.rs, `SheetModel.forcing_games`) is made for the side to move and, where the game gives a pass,
   for the other side on the position after the pass. Where the game gives none (the mover is under the goal's sign) the other side's terms are EMPTY for that position and the count of such
   positions is printed. Per threat move: the verdict (CONFIRMED, NOT BORNE OUT, NOT SETTLED, NO CLAIM; a move the fuse left unread is counted NOT SETTLED), the level it was reached at, the kind of its
   claim (the first present of ends, goal, exchange, takes, sign), and the claimed worth: for a CONFIRMED move its value (the floor over the replies; the goal at the goal's stake), for the others the best
   claim it makes (the goal at the goal's stake). The fuse is a parameter (`--fuse`); `--probe` measures the cost per position at two fuses first.
B. THE CHANCES.  Cell = (verdict, kind of claim, level). For each cell and each horizon h of 4, 8, 16 plies: how often the side whose threat it was GAINED AT LEAST the claimed worth (and something)
   within h plies: the mover's worth won (78.44's outcome code) for the mover's threats, its worth lost for the other side's. Fitted on the games with an even number, the test is the odd ones. The chance of
   a cell is the sheet's own `Chances` (the table of 78.41: counts, smoothed by the chance of the coarser key); the key is a prefix chain (verdict, kind, level) -> (verdict, kind) -> (verdict) -> all, so a
   cell with few cases falls back to the coarser one by the same smoothing, no count limit is tuned. The base rate (all cases) is printed beside.
C. THE NUMBER.  Per position, at the horizon h: forcing = max over the mover's threat moves of chance x stake, minus the same for the other side (the best single threat: the mover makes one move; the sum
   double-counts threats on the same unit). The stake is the claimed worth. forcing_sum, the sum over threat moves instead of the max, is printed as a labelled second form. Both forms were fixed before the
   long run.
D. THE TEST on the odd games, bootstrap bands by game as in 78.44: rank correlation with the outcome (worth won less lost) 4, 8, 16 plies later for (1) the 78.44 number (`net`: the same sheet, same
   chance table fit1112), (2) the forcing number, (3) both (their sum: the same unit, chance x worth), and the same on the positions where the 78.44 first level (`net0`) is zero and the mover has no
   CONFIRMED threat. Controls: the count alone (threat moves of the mover less the other side's, no chances), the forcing number shuffled within a game.
E. OPPONENT (`--weak-dir`): the same table fitted on games where the answering side is weaker, printed beside the first (cells compared).

  .venv/bin/python -m catspace.incubator.analyst.discovery.forcing_opportunity_measure --games-dir D --out-dir O --probe 10 --fuses 20000,5000
  .venv/bin/python -m catspace.incubator.analyst.discovery.forcing_opportunity_measure --games-dir D --out-dir O --fuse F [--games N] [--weak-dir W]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np

from catspace.incubator.analyst.discovery import opportunity_measure as OM
from catspace.io import paths

HORIZONS = OM.HORIZONS
GOAL = 1e9                                               # the engine's number for the goal (forcing_soundness_measure.GOAL); a value above half of it is the goal
VERDICTS = ("CONFIRMED", "NOT BORNE OUT", "NOT SETTLED", "NO CLAIM")
KIND_ORDER = ("ends", "goal", "exchange", "takes", "sign")
SHUFFLE_SEED = 7148
BOOT_SEED = 4448


# --------------------------------------------------------------------------------------------------------------------------- the terms (A)

def verdict_of(word: str) -> str:
    """The four verdicts of the plan; a move the fuse left unread is not settled."""
    return "NOT SETTLED" if word == "UNREAD" else word


def kind_of(claims: list) -> str:
    kinds = {c[0] for c in claims}
    return next((k for k in KIND_ORDER if k in kinds), "none")


def claimed_worth(threat: tuple, goal_stake: float) -> float:
    """The worth a threat move claims. threat = (verdict, value, level, [(claim kind, worth)], took_worth, plies)."""
    verdict, value, _level, claims, _took, _plies = threat
    if verdict == "CONFIRMED":
        return goal_stake if value >= GOAL / 2 else float(value)
    best = 0.0
    for kind, w in claims:
        best = max(best, goal_stake if (kind in ("ends", "goal") or w >= GOAL / 2) else float(w))
    return best


def case_of(threat: tuple, goal_stake: float) -> tuple:
    """(key, worth): the cell of a threat move and the worth it claims."""
    return (verdict_of(threat[0]), kind_of(threat[3]), str(int(threat[2]))), claimed_worth(threat, goal_stake)


def cases_of_ply(ply: "dict | None") -> dict:
    """{ours: [(key, worth)], theirs: [(key, worth)], has_pass: bool, confirmed: int} for one position of a read game (None: nothing)."""
    if ply is None:
        return {}
    ours = [case_of(t, ply["ours"]["goal_stake"]) for t in ply["ours"]["threats"]]
    theirs = [] if ply["theirs"] is None else [case_of(t, ply["theirs"]["goal_stake"]) for t in ply["theirs"]["threats"]]
    return {"ours": ours, "theirs": theirs, "has_pass": ply["theirs"] is not None, "confirmed": sum(1 for k, _w in ours if k[0] == "CONFIRMED")}


# --------------------------------------------------------------------------------------------------------------------------- the reads

def read_chunk(variant: str, records: list, threads: int, fuse: "int | None") -> list:
    """The forcing reads of the games, one list per game of None | {mover, ours, theirs}, in one Rust call across threads."""
    from catspace.incubator.analyst.discovery import question_sheet as QS
    from catspace.incubator.analyst.discovery import sheet_rust as SR
    env = QS.Env(variant)
    model = SR.model(env, QS.store_worth(variant))
    return model.forcing_games([(r.get("start"), list(r["moves"])) for r in records], threads, fuse)


def collect(variant: str, records: list, threads: int, fuse: "int | None", ckpt: str, chunk: int) -> dict:
    """{game id: list of plies}: read in chunks of games, each chunk appended to `ckpt` (one json line per game) so a stop loses at most one chunk; a rerun skips the games already there."""
    done: dict = {}
    p = Path(ckpt)
    if p.exists():
        for line in p.read_text().splitlines():
            if line.strip():
                x = json.loads(line)
                if x["fuse"] == fuse:
                    done[x["game"]] = x["plies"]
    todo = [r for r in records if r["game"] not in done]
    t0 = time.perf_counter()
    p.parent.mkdir(parents=True, exist_ok=True)
    for i in range(0, len(todo), chunk):
        part = todo[i:i + chunk]
        res = read_chunk(variant, part, threads, fuse)
        with open(p, "a") as f:
            for r, plies in zip(part, res):
                done[r["game"]] = plies
                f.write(json.dumps({"game": r["game"], "fuse": fuse, "plies": plies}) + "\n")
        el = time.perf_counter() - t0
        print(f"OBS progress forcing reads: {len(done)} of {len(records)} games, {el:.0f} s since start of this run, estimated {el / (i + len(part)) * (len(todo) - i - len(part)):.0f} s left", flush=True)
    return {r["game"]: done[r["game"]] for r in records}


# --------------------------------------------------------------------------------------------------------------------------- the fit (B)

def outcome_arrays(a: dict) -> dict:
    """{(game index, ply): row} of the 78.44 table."""
    return {(int(g), int(t)): i for i, (g, t) in enumerate(zip(a["game"], a["ply"]))}


def gained(a: dict, row: int, side: str, h: int) -> float:
    """What the side of a threat gained within h plies: the mover's worth won (its threats), its worth lost (the other side's). NaN where there is no reading at t + h."""
    return float(a[f"won{h}"][row] if side == "ours" else a[f"lost{h}"][row])


def hit(worth: float, got: float) -> bool:
    """Gained at least the claimed worth, and something."""
    return got > 0 and got >= worth - 1e-9


def fit_cells(reads: list, a: dict, games: list, keep=None) -> dict:
    """The counts {key: {h: [taken, total]}} over the games in `games` (indexes into `reads`): one case per threat move of either side, a case with no outcome at h left out for that h.
    `keep(game index, ply, side)`: when given, only the cases it accepts count (stage E: by who answers)."""
    idx = outcome_arrays(a)
    cells: dict = {}
    for g in games:
        for t, ply in enumerate(reads[g]):
            row = idx.get((g, t))
            if ply is None or row is None:
                continue
            c = cases_of_ply(ply)
            for side in ("ours", "theirs"):
                if keep is not None and not keep(g, t, side):
                    continue
                for key, worth in c[side]:
                    slot = cells.setdefault(key, {})
                    for h in HORIZONS:
                        got = gained(a, row, side, h)
                        if np.isnan(got):
                            continue
                        s = slot.setdefault(h, [0, 0])
                        s[0] += int(hit(worth, got))
                        s[1] += 1
    return cells


def table_of_cells(cells: dict):
    from catspace.incubator.analyst.discovery import question_sheet as QS
    return QS.Chances({k: {h: tuple(v) for h, v in c.items()} for k, c in cells.items()})


def base_rate(cells: dict, h: int) -> "tuple[int, int]":
    return sum(c.get(h, [0, 0])[0] for c in cells.values()), sum(c.get(h, [0, 0])[1] for c in cells.values())


def cell_lines(cells: dict, table, label: str) -> list:
    lines = []
    for h in HORIZONS:
        k, n = base_rate(cells, h)
        lines.append(f"OBS forcing_opportunity {label}: horizon {h}: cases {n}, base rate (gained at least the claimed worth) {k / max(1, n):.4f} ({k} of {n}), cells {sum(1 for c in cells.values() if c.get(h, [0, 0])[1] > 0)}")
        for key in sorted(cells, key=lambda x: -cells[x].get(h, [0, 0])[1]):
            tk, tot = cells[key].get(h, [0, 0])
            if tot:
                lines.append(f"OBS forcing_opportunity {label}: horizon {h} cell {'/'.join(key)}: cases {tot}, gained {tk}, rate {tk / tot:.4f}, fitted chance {table.chance(key, h):.4f}")
    return lines


# --------------------------------------------------------------------------------------------------------------------------- the number (C)

def number_at(c: dict, table, h: int) -> dict:
    """The forcing number of one position at horizon h: max and sum over the mover's threat moves of chance x stake, less the other side's; the count (no chances)."""
    def terms(cases):
        return [table.chance(k, h) * max(0.0, w) for k, w in cases]
    o, t = terms(c["ours"]), terms(c["theirs"])
    return {"f": max(o, default=0.0) - max(t, default=0.0), "fsum": sum(o) - sum(t), "count": float(len(o) - len(t))}


def build(reads: list, a_opp: dict, table, games: list) -> dict:
    """The arrays of the test (rows of the 78.44 table for the games in `games`) with the forcing number per horizon beside: f{h}, fsum{h}, both{h}, fshuf{h}, fcount, nconf, has_pass."""
    idx = outcome_arrays(a_opp)
    rows = [idx[(g, t)] for g in games for t in range(len(reads[g])) if (g, t) in idx and reads[g][t] is not None]
    a = {k: v[rows] for k, v in a_opp.items()}
    n = len(rows)
    out = {k: np.zeros(n) for k in ("fcount", "nconf", "has_pass") + tuple(f"{p}{h}" for p in ("f", "fsum", "both", "fshuf") for h in HORIZONS)}
    pos = 0
    for g in games:
        first = pos
        for t, ply in enumerate(reads[g]):
            if (g, t) not in idx or ply is None:
                continue
            c = cases_of_ply(ply)
            out["nconf"][pos] = c["confirmed"]
            out["has_pass"][pos] = c["has_pass"]
            nums = {h: number_at(c, table, h) for h in HORIZONS}
            out["fcount"][pos] = nums[HORIZONS[0]]["count"]
            for h in HORIZONS:
                out[f"f{h}"][pos] = nums[h]["f"]
                out[f"fsum{h}"][pos] = nums[h]["fsum"]
                out[f"both{h}"][pos] = a["net"][pos] + nums[h]["f"]
            pos += 1
        for h in HORIZONS:
            vals = out[f"f{h}"][first:pos].tolist()
            out[f"fshuf{h}"][first:pos] = OM.shuffled_net(vals, random.Random(SHUFFLE_SEED + g * 10 + h))
    a.update(out)
    return a


# --------------------------------------------------------------------------------------------------------------------------- the test (D)

def specs(a: dict) -> list:
    """[(label, function of the (sub)table -> correlation)] for every reading that gets an interval."""
    ok = lambda d, *names: np.all([~np.isnan(d[n]) for n in names], axis=0)
    out = []
    for h in HORIZONS:
        o, w = f"out{h}", f"won{h}"
        for sub, keep in (("all test positions", lambda d: np.ones(len(d["ply"]), bool)), ("net0 = 0 and no CONFIRMED threat", lambda d: (d["net0"] == 0) & (d["nconf"] == 0))):
            for name, col in (("78.44 number", "net"), ("forcing number", f"f{h}"), ("both", f"both{h}"), ("forcing number, sum form", f"fsum{h}"), ("control: count of threat moves", "fcount"),
                              ("control: forcing number shuffled in game", f"fshuf{h}")):
                out.append((f"h{h} {name} vs outcome, {sub}", lambda d, c=col, o=o, keep=keep: OM.spearman(d[c][ok(d, o, c) & keep(d)], d[o][ok(d, o, c) & keep(d)])))
            out.append((f"h{h} forcing number vs worth won, {sub}", lambda d, c=f"f{h}", w=w, keep=keep: OM.spearman(d[c][ok(d, w, c) & keep(d)], d[w][ok(d, w, c) & keep(d)])))
            out.append((f"h{h} forcing number vs outcome, 78.44 number held out (partial rank), {sub}", lambda d, c=f"f{h}", o=o, keep=keep: OM.partial_spearman(
                d[c][ok(d, o) & keep(d)], d[o][ok(d, o) & keep(d)], d["net"][ok(d, o) & keep(d)])))
    return out


def report(a: dict, games: int, draws: int, label: str) -> "tuple[list, dict]":
    n = len(a["ply"])
    sub = (a["net0"] == 0) & (a["nconf"] == 0)
    lines = [f"OBS forcing_opportunity {label}: test games {games}, test positions {n}, positions where the other side has no pass (its terms empty) {int(n - a['has_pass'].sum())}, "
             f"positions with net0 = 0 and no CONFIRMED threat {int(sub.sum())}, positions with a CONFIRMED threat {int((a['nconf'] > 0).sum())}"]
    for h in HORIZONS:
        lines.append(f"OBS forcing_opportunity {label}: horizon {h}: test positions with an outcome {int((~np.isnan(a[f'out{h}'])).sum())}, of them in the subset {int(((~np.isnan(a[f'out{h}'])) & sub).sum())}")
    sp = specs(a)
    t0 = time.perf_counter()
    cis = OM.bootstrap(a, [f for _l, f in sp], draws, BOOT_SEED)
    data = {"label": label, "correlations": [], "bands": {}}
    for (name, f), (lo, hi) in zip(sp, cis):
        v = f(a)
        data["correlations"].append({"name": name, "spearman": v, "ci90": [lo, hi]})
        lines.append(f"OBS forcing_opportunity {label}: spearman {name}: {v:+.4f}  90% interval over games [{lo:+.4f}, {hi:+.4f}] ({draws} draws)")
    lines.append(f"OBS forcing_opportunity {label}: bootstrap seconds {time.perf_counter() - t0:.1f}")
    for h in HORIZONS:
        b = dict(a)
        b["net"] = a[f"f{h}"]
        data["bands"][h] = OM.bands(b, h)
        for band, c, mn, mo, mw, ml in data["bands"][h]:
            lines.append(f"OBS forcing_opportunity {label}: horizon {h} band {band} of {OM.BANDS} of the forcing number: plies {c}, mean number {mn:+.4f}, mean outcome {mo:+.4f}, mean worth won {mw:.4f}, mean worth lost {ml:.4f}")
    return lines, data


# --------------------------------------------------------------------------------------------------------------------------- the 78.44 side

def opp_table(variant: str, records: list, threads: int, chances: "str | None", cache: str) -> dict:
    """The 78.44 table (number, baselines and outcome per ply) of the games, from the same Rust sheet and chance table; cached in `cache`."""
    from catspace.incubator.analyst.discovery import question_sheet as QS
    p = Path(cache)
    if p.exists():
        z = np.load(p)
        if len(z["game"]) and int(z["n_games"]) == len(records):
            return {k: z[k] for k in z.files if k != "n_games"}
    table = QS.load_chances(chances or paths.derived("question_chances", "fit1112.json"))
    t0 = time.perf_counter()
    a = OM.table_of(OM.briefs_of(variant, records, threads), table)
    print(f"OBS 78.44 table of {len(records)} games in {time.perf_counter() - t0:.1f} s", flush=True)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(p), n_games=len(records), **a)
    return a


# --------------------------------------------------------------------------------------------------------------------------- the fuse probe

def probe(variant: str, records: list, threads: int, fuses: list, label: str) -> list:
    """Seconds per position at each fuse on the first games: the read's own time (ours and the other side's, summed over threads), the wall of the batch, how often the fuse stopped the read, and how
    often the verdicts of the moves differ from the first fuse's."""
    lines, kept = [], []
    for fuse in fuses:
        t0 = time.perf_counter()
        res = read_chunk(variant, records, threads, fuse)
        wall = time.perf_counter() - t0
        pos = [p for g in res for p in g if p is not None]
        reads = [r for p in pos for r in (p["ours"], p["theirs"]) if r is not None]
        sec = sum(r["seconds"] for r in reads)
        cut = sum(1 for r in reads if r["cut"])
        thr = sum(len(r["threats"]) for r in reads)
        conf = sum(1 for r in reads for t in r["threats"] if t[0] == "CONFIRMED")
        lines.append(f"OBS forcing_opportunity {label}: fuse {fuse}: {len(records)} games, {len(pos)} positions, wall {wall:.1f} s on {threads} threads, read seconds per position {sec / max(1, len(pos)):.4f} (both sides, summed), "
                     f"wall seconds per position {wall / max(1, len(pos)):.4f}, reads stopped by the fuse {cut} of {len(reads)}, threat moves per read {thr / max(1, len(reads)):.2f}, CONFIRMED per read {conf / max(1, len(reads)):.3f}, "
                     f"positions without the other side's read {sum(1 for p in pos if p['theirs'] is None)}")
        kept.append(res)
    base = kept[0]
    for fuse, res in zip(fuses[1:], kept[1:]):
        diff = tot = 0
        for first, second in zip(base, res):
            for p0, p1 in zip(first, second):
                if p0 is None:
                    continue
                for side in ("ours", "theirs"):
                    if p0[side] is None:
                        continue
                    tot += 1
                    diff += sorted((t[0], t[1], t[2]) for t in p0[side]["threats"]) != sorted((t[0], t[1], t[2]) for t in p1[side]["threats"])
        lines.append(f"OBS forcing_opportunity {label}: reads whose threat moves (verdict, value, level) differ between fuse {fuses[0]} and fuse {fuse}: {diff} of {tot}")
    return lines


# --------------------------------------------------------------------------------------------------------------------------- stage E: the opponent

def make_weak_games(variant: str, n: int, seed0: int, out_dir: str, jobs: int, seconds: float) -> None:
    """N games of the plain search (the control player) in which ONE colour (alternating by seed pairs) searches `seconds` a move instead of the control clock: the weaker answerer. Each record
    names the weaker colour (`weak`: 0 moves first). Seeds are the engine's openings as in 78.44's `make_games`."""
    import multiprocessing
    from catspace.incubator.analyst.discovery import game_store as GS
    from catspace.incubator.analyst.discovery import sheet_player as SP
    out = Path(out_dir)
    store = str(GS.path_of(variant, GS.head(variant))) if GS.head(variant) else None
    cfg = {**SP.CONFIGS["control"], "seconds": seconds}
    specs_ = [{"game": variant, "store": store, "scratch": str(out / "scratch"), "gid": f"{s}", "opening_id": s, "guided": (s // 2) % 2, "opening": OM.random_opening(variant, s), "config": cfg,
               "config_name": "control_weak", "plain_seconds": SP.PlayerConfig().seconds, "max_plies": 120} for s in range(seed0, seed0 + n)]
    out.mkdir(parents=True, exist_ok=True)
    t0, done = time.perf_counter(), []
    with multiprocessing.get_context("spawn").Pool(jobs) as pool:
        for g in pool.imap_unordered(SP.play_game, specs_):
            done.append(g)
            print(f"OBS progress weak game {g['gid']} plies {len(g['moves'])} outcome {g['outcome']} ({len(done)} of {n}) {time.perf_counter() - t0:.0f}s", flush=True)
    done.sort(key=lambda g: g["opening_id"])
    with open(out / "games.jsonl", "w") as f:
        for g in done:
            f.write(json.dumps({"variant": variant, "game": g["opening_id"], "start": None, "moves": g["moves"], "outcome": g["outcome"], "unfinished": g["unfinished"], "opening_len": g["opening_len"],
                                "weak": g["guided"], "weak_seconds": seconds}) + "\n")
    print(f"OBS made {len(done)} weak-answerer games in {time.perf_counter() - t0:.0f}s, wall, jobs {jobs}")


def answerer_weak(record: dict, ply: int, side: str) -> bool:
    """Is the side that ANSWERS the threat of this case the weaker colour? The mover at `ply` is the colour ply % 2 (0 moves first); its threats are answered by the other colour, the other side's by the mover."""
    mover = ply % 2
    answering = 1 - mover if side == "ours" else mover
    return answering == record["weak"]


def compare_lines(cells_a: dict, cells_b: dict, label_a: str, label_b: str, label: str) -> list:
    """Each cell with cases in both fits, side by side: cases, rate, fitted chance, at each horizon."""
    ta, tb = table_of_cells(cells_a), table_of_cells(cells_b)
    lines = []
    for h in HORIZONS:
        for key in sorted(set(cells_a) & set(cells_b), key=lambda k: -min(cells_a[k].get(h, [0, 0])[1], cells_b[k].get(h, [0, 0])[1])):
            (ka, na), (kb, nb) = cells_a[key].get(h, [0, 0]), cells_b[key].get(h, [0, 0])
            if na and nb:
                lines.append(f"OBS forcing_opportunity {label}: horizon {h} cell {'/'.join(key)}: {label_a} cases {na} rate {ka / na:.4f} chance {ta.chance(key, h):.4f}; {label_b} cases {nb} rate {kb / nb:.4f} chance {tb.chance(key, h):.4f}")
        ka, na = base_rate(cells_a, h)
        kb, nb = base_rate(cells_b, h)
        lines.append(f"OBS forcing_opportunity {label}: horizon {h} base rate: {label_a} {ka / max(1, na):.4f} ({na} cases); {label_b} {kb / max(1, nb):.4f} ({nb} cases)")
    return lines


def stage_e(records: list, reads: list, a_opp: dict, label: str) -> list:
    """The cells fitted on the even games of the weak-answerer set, cases split by who answers: the weaker colour, or the control colour in the same games; printed side by side."""
    fit, _test = split_games(records)
    weak = fit_cells(reads, a_opp, fit, lambda g, t, side: answerer_weak(records[g], t, side))
    strong = fit_cells(reads, a_opp, fit, lambda g, t, side: not answerer_weak(records[g], t, side))
    return compare_lines(strong, weak, "answerer at the control clock", f"answerer weaker (clock {records[0].get('weak_seconds')})", label)


# --------------------------------------------------------------------------------------------------------------------------- the run

def read_games(games_dir: str) -> list:
    recs = OM.read_games(games_dir)
    return sorted(recs, key=lambda r: r["game"])


def split_games(records: list) -> "tuple[list, list]":
    """(fit, test): indexes into `records`; the games with an even id fit the chances, the odd ones are the test."""
    return [i for i, r in enumerate(records) if r["game"] % 2 == 0], [i for i, r in enumerate(records) if r["game"] % 2 == 1]


def run(records: list, variant: str, a_opp: dict, reads: list, draws: int, label: str) -> "tuple[list, dict]":
    fit, test = split_games(records)
    cells = fit_cells(reads, a_opp, fit)
    table = table_of_cells(cells)
    lines = [f"OBS forcing_opportunity {label}: games {len(records)} (fit {len(fit)} even-numbered, test {len(test)} odd-numbered); horizons {HORIZONS}; verdicts {VERDICTS}; kinds {KIND_ORDER}; fallback by smoothing to (verdict, kind), (verdict), all"]
    lines += cell_lines(cells, table, label)
    rd = [r for g in reads for p in g if p is not None for r in (p["ours"], p["theirs"]) if r is not None]
    npos = sum(1 for g in reads for p in g if p is not None)
    lines.append(f"OBS forcing_opportunity {label}: positions {npos}, reads {len(rd)}, read seconds per position (both sides, summed over threads) {sum(r['seconds'] for r in rd) / max(1, npos):.4f}, reads stopped by the fuse {sum(1 for r in rd if r['cut'])}")
    verdicts = Counter(verdict_of(t[0]) for r in rd for t in r["threats"])
    lines.append(f"OBS forcing_opportunity {label}: threat moves by verdict {dict(verdicts)}")
    a = build(reads, a_opp, table, test)
    more, data = report(a, len(test), draws, label)
    data["cells"] = {"/".join(k): v for k, v in cells.items()}
    return lines + more, data


def main(argv: "list | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--games-dir", default=None)
    ap.add_argument("--variant", default=None, help="only for --make-weak")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--games", type=int, default=0, help="a subset: the first N games by id")
    ap.add_argument("--fuse", type=int, default=None, help="positions the forcing read may read (default: the engine's)")
    ap.add_argument("--probe", type=int, default=0, help="measure seconds per position on the first N games at each of --fuses, then stop")
    ap.add_argument("--fuses", default="20000,5000")
    ap.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--chunk", type=int, default=0, help="games per call of the engine (a checkpoint after each); default 4 x threads")
    ap.add_argument("--draws", type=int, default=OM.DRAWS)
    ap.add_argument("--chances", default=None)
    ap.add_argument("--weak-dir", default=None, help="stage E: a games directory of games with a weaker answerer (see --make-weak)")
    ap.add_argument("--make-weak", type=int, default=0, help="stage E: make N weaker-answerer games into --weak-dir, then stop")
    ap.add_argument("--weak-seed0", type=int, default=5516)
    ap.add_argument("--weak-seconds", type=float, default=0.02, help="the weaker colour's clock a move")
    a = ap.parse_args(argv)
    if a.make_weak:
        make_weak_games(a.variant, a.make_weak, a.weak_seed0, a.weak_dir, max(1, (os.cpu_count() or 2) - 1), a.weak_seconds)
        return 0
    records = read_games(a.games_dir)
    variant = records[0]["variant"]
    label = Path(a.games_dir).name
    out = Path(a.out_dir or paths.experiment("forcing_opportunity_78_48", label))
    if a.probe:
        for line in probe(variant, records[:a.probe], a.threads, [int(x) for x in a.fuses.split(",")], label):
            print(line, flush=True)
        return 0
    if a.games:
        records = records[:a.games]
    t0 = time.perf_counter()
    chunk = a.chunk or 4 * a.threads
    plies = collect(variant, records, a.threads, a.fuse, str(out / f"reads_fuse{a.fuse}.jsonl"), chunk)
    reads = [plies[r["game"]] for r in records]
    a_opp = opp_table(variant, records, a.threads, a.chances, str(out / f"opp_table_n{len(records)}.npz"))
    lines, data = run(records, variant, a_opp, reads, a.draws, label)
    lines.append(f"OBS forcing_opportunity {label}: fuse {a.fuse if a.fuse is not None else 'the engine default'}, total wall {time.perf_counter() - t0:.1f} s")
    if a.weak_dir:
        wrec = read_games(a.weak_dir)
        wlabel = Path(a.weak_dir).name
        wplies = collect(variant, wrec, a.threads, a.fuse, str(out / f"weak_reads_fuse{a.fuse}.jsonl"), chunk)
        wreads = [wplies[r["game"]] for r in wrec]
        w_opp = opp_table(variant, wrec, a.threads, a.chances, str(out / f"weak_opp_table_n{len(wrec)}.npz"))
        lines += stage_e(wrec, wreads, w_opp, wlabel)
        base_cells = {tuple(k.split("/")): {int(h): v for h, v in c.items()} for k, c in data["cells"].items()}
        weak_cells = fit_cells(wreads, w_opp, split_games(wrec)[0], lambda g, t, side: answerer_weak(wrec[g], t, side))
        lines += compare_lines(base_cells, weak_cells, "the 500-game set (control against control)", "the weaker-answerer games, answerer weaker", wlabel + " against base")
    for line in lines:
        print(line, flush=True)
    (out / "result.json").write_text(json.dumps({"lines": lines, "data": data}, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
