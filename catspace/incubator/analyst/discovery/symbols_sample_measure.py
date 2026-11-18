"""PLAN 78.89 (SYSTEM_SPEC 7 "subgoals as symbols that go with winning" -- the opportunity policy's hint among equal moves --, 9 the score, 10 the endings risk and the fill): OPENING OR MIDDLE GAME, AND A SAMPLE THAT COVERS THE LATER GAME.
A sandbox: no player is changed; nothing is written to rules/learned or data/derived; every output goes under artifacts/experiments/symbols_78_89/<tag>/ (OUT below, set before any run). Game-free: the game is named only through
the pieces that exist (ladder_measure's bands and unit counts, human_heads_data, graph_layer1/2, symbols, symbols_measure, symbols_reasons_measure, symbols_groups_measure, symbols_play); the board and the words of a case are
printed by ladder_print only. The outputs of 78.80, 78.81, 78.82, 78.83 and 78.88 are READ, never rebuilt, never written. No player id is written (a game is an ordinal of this run).

INTERFACE (what each stage takes and writes; who calls it). Each stage is a command and prints OBS lines (also to obs.txt).
  PART ONE (prints and two matches, no new method)
  first    (a)  78.81's agreement with the move played on the 78.80 CONFIRM positions of the many band (17 units or more), by PLY (four bins of equal count) and by UNIT COUNT (the band cut in three by cuts taken from the unit
                counts of the whole 78.80 sample BEFORE any agreement is read, printed first), each for quiet (no tied move takes a unit) and not, paired against the search's first move (symbols_measure.paired).
  cut      (b)  the point that cuts the many band in two for the window players: the median unit count of the many band's tie decisions of the reasons' side in 78.82's reasons v control games (sets A and B, uncharged). Writes
                cut.json; prints the cut and the earlier matches (78.82's full player, 78.83's four windows) from their reports.
  match    (b)  one match, window player v control: `--a manyhi` (the many band above the cut only) or `--a manylo` (the many band from 17 units to the cut), `--set A|B`, `--pairs`, the load wait of 78.67 first; the games and
                a meta file go to <tag>/matches/. report: every match found, per set and pooled, with the decisions.
  cases    (c)  thirty decisions: fifteen of the LOWER many part and fifteen of the mid band, from 78.82's stored games, by the rule in cases_rule(); writes cases89.json for `ladder_print --play89`.
  PART TWO (the sample)
  sample        a second human sample drawn evenly over bins of unit count (symbols_measure.plies_stratified; the bins 0-4, 5-8, 9-12, 13-16 and 17 to the cut) from the held-out games of 78.80 (78.59's file) less the ladder's games and
                less every game that has a kept position in the first sample, then, where those run out, from further games of the same shards by the same filter (HD.sample_shard: ranks past the 78.59 file's own 1500 a shard);
                the search at the wake's clock keeps a position when two or more moves are equal and the move played is among them; the part (FIND or CONFIRM) is symbols_measure.part_of (78.80's hash). Writes sample.jsonl, tries.jsonl.
  fill          where a bin of FIND holds fewer than TARGET human tie positions: positions of our own self-play games (78.58's plain-search sets, the result known; never a human game continued, never an engine line), marked
                source "generated", FIND only; "the move played" is the mover's own move in that game. Writes sample_gen.jsonl.
  coverage      per bin: positions tried and kept by part and source, games, the generated share of each FIND bin, how far the few band falls short.
  events        for the merged sample (human then generated): the events of the 136 reasons of 78.81 (symbols.deltas_matrix on the reasons' keys, laid out as 78.80's event matrix so symbols_groups_measure.reason_columns reads it),
                the cells of the split (symbols_groups_measure._split_job), the controls (symbols_reasons_measure._controls_job). Writes events.npz, cells.npz, cells_keys.json, controls.npz.
  fit           the reasons as they are (variant ii of 78.88: the groups of 78.88 and the split): the one joint choice fit refitted on the new FIND (generated rows read: Choice.fill_ok), and one fit for each band of units; writes fits.pkl.
  score         on the new CONFIRM (human only): the two new fits and the old weights unchanged, by band, quiet and not, each paired against the search's first move and against the old weights; the old weights on the old CONFIRM
                beside; the condition for play.
Called by: the command line only; ladder_print.py reads cases89.json (`--play89`).

  .venv/bin/python -m catspace.incubator.analyst.discovery.symbols_sample_measure first|cut|cases|sample|fill|coverage|events|fit|score|match|report --tag short|full
"""
from __future__ import annotations

from catspace.incubator.analyst.discovery import tiebreak_measure as TM          # first: it sets one thread a process before numpy is loaded

import argparse
import json
import math
import multiprocessing
import os
import pickle
import re
import time
from collections import Counter as Tally
from pathlib import Path

import numpy as np
from scipy import sparse

from catspace.incubator.analyst.discovery import estimator_fit as EF
from catspace.incubator.analyst.discovery import graph_layer1 as GL
from catspace.incubator.analyst.discovery import graph_layer2_measure as G2M
from catspace.incubator.analyst.discovery import human_heads_data as HD
from catspace.incubator.analyst.discovery import ladder_measure as LM
from catspace.incubator.analyst.discovery import symbols as SY
from catspace.incubator.analyst.discovery import symbols_groups_measure as SGM
from catspace.incubator.analyst.discovery import symbols_measure as SM
from catspace.incubator.analyst.discovery import symbols_play as SP
from catspace.incubator.analyst.discovery import symbols_reasons_measure as SR
from catspace.io import paths

OUT = ("symbols_78_89",)                                          # this run's outputs; set before any run (the 78.80 to 78.88 runs are read only)
SRC80 = ("symbols_78_80", "full")
SRC88 = ("symbols_78_88", "full")
OUT82 = ("symbols_play_78_82", "full")
OUT83 = ("symbols_play_78_83", "full")
SEED = 8989
MANY_LOW = 17                                                     # the many band starts at this many units (ladder_measure.BANDS)
TARGET = 1000                                                     # tie positions a bin of unit count should hold, in each part (the plan's aim)
PER_GAME_BIN = 3                                                  # positions of one game in one bin at most
BLOCK = 400                                                       # games taken from the pool at a time
EXTRA_PER_SHARD = 1500                                            # further games of a shard: ranks 1500 to 3000 of its hash order (the 78.59 file holds ranks 0 to 1499)
EXTRA_BASE = 1_000_000                                            # the ordinal of a further game: this plus its rank times eight plus the shard's place (never a game number)
GEN_BASE = 5_000_000                                              # the ordinal of a generated game
GEN_SETS = ("selfplay500", "selfplay16", "m50_control_control")   # 78.58's plain search v plain search sets (estimator_fit.SETS), the result known
FIND_YIELD = 1.15                                                 # tries planned over the shortfall divided by the observed yield
CASES_PER_PART = 15
CASE_NTH = 3                                                      # the third such decision of a game in the part (never the first)
DRAWS = SM.DRAWS_BOOT


def out_root(tag: str) -> Path:
    p = Path(paths.experiment(*OUT, tag))
    p.mkdir(parents=True, exist_ok=True)
    return p


class Say:
    def __init__(self, root: Path, stage: str):
        self.root, self.stage = root, stage

    def __call__(self, msg: str) -> None:
        line = f"OBS sample89 {self.stage}: {msg}"
        print(line, flush=True)
        with open(self.root / "obs.txt", "a") as f:
            f.write(line + "\n")


def src80() -> Path:
    return Path(paths.experiment(*SRC80))


def src88() -> Path:
    return Path(paths.experiment(*SRC88))


# ------------------------------------------------------------------------------------------------------------------------------ PART ONE (a): the finer scale

def _some_take(rows: SY.Choice, P: int) -> np.ndarray:
    sizes = np.bincount(rows.pos)
    start = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    return np.array([bool(rows.C[start[p]:start[p] + sizes[p], 0].any()) for p in range(P)])


def scale_cuts(sample: list) -> dict:
    """The cuts of the many band, from counts only (no agreement is read): `ply_edges` three plies at the quarter points of the plies of the CONFIRM positions of the band (equal-count bins), `unit_edges` two unit counts at the
    third points of the unit counts of the band's positions in the whole 78.80 sample (FIND and CONFIRM; the lower value, an integer count)."""
    conf = [r for r in sample if r["part"] == "confirm" and r["units"] >= MANY_LOW]
    plies = np.sort(np.array([r["ply"] for r in conf]))
    n = len(plies)
    ply_edges = [int(plies[n * i // 4]) for i in (1, 2, 3)]
    units = np.array([r["units"] for r in sample if r["units"] >= MANY_LOW])
    unit_edges = [int(x) for x in np.quantile(units, [1 / 3, 2 / 3], method="lower")]
    return {"ply_edges": ply_edges, "unit_edges": unit_edges, "n_conf": n, "n_all": len(units)}


def _bin_index(values: np.ndarray, edges: list) -> np.ndarray:
    """0 for values below the first edge, 1 from the first edge up to below the second ..., len(edges) for the rest."""
    return np.array([sum(v >= e for e in edges) for v in values])


def stage_first(root: Path, say, limit: int = 0) -> None:
    inp = SR.load_inputs(limit)
    ctl = SR.load_controls(SGM.src81(), limit)
    res = SR.score_confirm_reasons(SGM.src81(), inp, ctl, lambda m: None)
    rows, hits, game = res["rows"], res["hits"], res["game"]
    sample_c = [r for r in inp["sample"] if r["part"] == "confirm"]
    P = rows.n_positions()
    units = np.array([r["units"] for r in sample_c])
    ply = np.array([r["ply"] for r in sample_c])
    many = units >= MANY_LOW
    cuts = scale_cuts(inp["sample"])
    say(f"CUTS fixed from counts alone, before any agreement was read: the ply cuts (four bins of equal count among the {cuts['n_conf']} CONFIRM positions of the many band) at plies {cuts['ply_edges']}; "
        f"the unit-count cuts (three parts of the {cuts['n_all']} positions of the whole 78.80 sample in the many band) at {cuts['unit_edges'][0]} and {cuts['unit_edges'][1]} units")
    pb = _bin_index(ply, cuts["ply_edges"])
    ue = cuts["unit_edges"]
    ub = _bin_index(units, [ue[0] + 1, ue[1] + 1])
    pnames = [f"ply below {cuts['ply_edges'][0]}", f"ply {cuts['ply_edges'][0]} to {cuts['ply_edges'][1] - 1}", f"ply {cuts['ply_edges'][1]} to {cuts['ply_edges'][2] - 1}", f"ply {cuts['ply_edges'][2]} and over"]
    unames = [f"{MANY_LOW} to {ue[0]} units", f"{ue[0] + 1} to {ue[1]} units", f"{ue[1] + 1} units and over"]
    some = _some_take(rows, P)
    say(f"FIRST: 78.81's agreement with the move played on the {int(many.sum())} CONFIRM positions of the many band ({MANY_LOW} units or more), the search's first move beside; quiet = no tied move takes a unit; "
        f"paired = points, 90 pct bootstrap over games ({DRAWS} draws); 'new' = the reasons' pick without the controls, 'ctl' = with the controls in the pick")
    layout = [("many band, all", many)]
    layout += [(f"by ply, {pnames[i]}", many & (pb == i)) for i in range(4)]
    layout += [(f"by units, {unames[i]}", many & (ub == i)) for i in range(3)]
    for label, sel0 in layout:
        for tag, sel in ((label, sel0), (label + ", quiet", sel0 & ~some), (label + ", some tied move takes a unit", sel0 & some)):
            m = int(sel.sum())
            if m < 5:
                say(f"FIRST {tag}: {m} positions (too few)")
                continue
            parts = [f"new {SM.wilson_line(hits['new'][sel].sum(), m)}", f"ctl {SM.wilson_line(hits['new with the controls in the pick'][sel].sum(), m)}", f"search first {SM.wilson_line(hits['search first'][sel].sum(), m)}"]
            dnew = SM.paired(hits["new"][sel], hits["search first"][sel], game[sel], SEED + 1)
            dctl = SM.paired(hits["new with the controls in the pick"][sel], hits["search first"][sel], game[sel], SEED + 2)
            say(f"FIRST {tag} ({m} positions, {len(set(game[sel].tolist()))} games): " + "; ".join(parts) + f"; paired new less first {100 * dnew[0]:+.2f} [{100 * dnew[1]:+.2f}, {100 * dnew[2]:+.2f}]; ctl less first {100 * dctl[0]:+.2f} [{100 * dctl[1]:+.2f}, {100 * dctl[2]:+.2f}]")


# ------------------------------------------------------------------------------------------------------------------------------ PART ONE (b): the cut and the matches

KIND89 = dict(SP.KIND, manyhi="sy_reasons", manylo="sy_reasons")      # the two window players are the full reasons player of 78.82 with a window in the spec


def _games82(set_name: str) -> list:
    return TM.read_games(Path(paths.experiment(*OUT82, "matches", f"reasons_v_control_{set_name}.jsonl")))


def stage_cut(root: Path, say) -> None:
    """THE CUT, fixed before any game of this plan: the median of the unit counts at the tie decisions of the reasons' side in the many band, in 78.82's reasons v control games of both sets (uncharged). The lower part is
    MANY_LOW to the median (inclusive), the upper part is the median plus one and up."""
    layer1 = GL.model(HD.GAME, HD.VARIANT)
    counts_at = []
    n_dec = 0
    for s in ("A", "B"):
        for g in _games82(s):
            st = g["stats"][TM.seat_of(g, 0)]
            c = LM.unit_counts(layer1, g["moves"])
            for ply, _n, _ch, _ms in st["dlog"]:
                n_dec += 1
                if c[ply] >= MANY_LOW:
                    counts_at.append(c[ply])
    arr = np.array(counts_at)
    med = float(np.median(arr))
    cut = int(math.floor(med))
    spread = np.quantile(arr, [0, 0.25, 0.5, 0.75, 1.0])
    res = {"median": med, "cut": cut, "lower": [MANY_LOW, cut], "upper": [cut + 1, 99], "n_many_ties": int(len(arr)), "n_ties": n_dec}
    (root / "cut.json").write_text(json.dumps(res))
    say(f"CUT: the median unit count of the many band's tie decisions of the reasons' side in 78.82's reasons v control games (sets A and B, uncharged): {med} over {len(arr)} decisions (of {n_dec} tie decisions in all); "
        f"quartiles {spread.tolist()}; the lower part is {MANY_LOW} to {cut} units ({int((arr <= cut).sum())} decisions), the upper part is {cut + 1} units and more ({int((arr > cut).sum())} decisions)")
    earlier(say)


def earlier(say) -> None:
    """78.82's full player and 78.83's four windows, as their reports printed them (uncharged, v control)."""
    pat = re.compile(r"^OBS symbols_play (match uncharged|pooled) (reasons|many|mid|few|midfew) v control")
    for exp, name in ((OUT82, "78.82"), (OUT83, "78.83")):
        for ln in (Path(paths.experiment(*exp)) / "report.txt").read_text().splitlines():
            if pat.match(ln) and " W " in ln:
                head, _sep, rest = ln.partition(": ")
                say(f"EARLIER {name}: {head.replace('OBS symbols_play ', '')}: {rest}")


def load_cut(root: Path) -> dict:
    return json.loads((root / "cut.json").read_text())


def window_of(a: str, cut: dict) -> list:
    return cut["upper"] if a == "manyhi" else cut["lower"]


def match_specs89(a: str, set_name: str, pairs: int, seconds: float, root: Path, cut: dict) -> list:
    specs = TM.match_specs(a, "control", set_name, pairs, seconds, root / "scratch", False, KIND89)
    for s in specs:
        for pl in s["players"]:
            if pl["kind"] == "sy_reasons":
                pl["window"] = list(window_of(a, cut))
    return specs


def games_file(root: Path, a: str, set_name: str) -> Path:
    return root / "matches" / f"{a}_v_control_{set_name}.jsonl"


def stage_match(root: Path, a: str, set_name: str, pairs: int, workers: int, seconds: float, say) -> None:
    cut = load_cut(root)
    load, busy = TM.wait_quiet(lambda s: TM.emit([s], root / "obs.txt"))
    t0 = time.perf_counter()
    gf = games_file(root, a, set_name)
    games = TM.run_games(match_specs89(a, set_name, pairs, seconds, root, cut), workers, gf, lambda s: TM.emit([s], root / "obs.txt"), play=SP.play82)
    wall = time.perf_counter() - t0
    gf.with_suffix(".meta.json").write_text(json.dumps({"load": load, "busy": busy, "wall": wall, "workers": workers, "seconds": seconds, "window": window_of(a, cut)}))
    TM.emit(SP.lines_of(a, "control", set_name, games, load, busy, wall, seconds, workers, False), root / "obs.txt")
    TM.emit(SP.decision_lines(a, "control", set_name, games, GL.model(HD.GAME, HD.VARIANT)), root / "obs.txt")


def stage_report(root: Path, seconds: float, workers: int) -> None:
    layer1 = GL.model(HD.GAME, HD.VARIANT)
    found: dict = {}
    for p in sorted((root / "matches").glob("*.jsonl")):
        x, rest = p.stem.split("_v_control_")
        found.setdefault(x, {})[rest] = TM.read_games(p)
    for x, sets in sorted(found.items()):
        for s, games in sorted(sets.items()):
            meta = json.loads(games_file(root, x, s).with_suffix(".meta.json").read_text()) if games_file(root, x, s).with_suffix(".meta.json").exists() else {}
            TM.emit(SP.lines_of(x, "control", s, games, meta.get("load", float("nan")), meta.get("busy", float("nan")), meta.get("wall", float("nan")), seconds, workers, False), None)
            TM.emit(SP.decision_lines(x, "control", s, games, layer1), None)
        if len(sets) > 1:
            TM.emit(SP.pooled_lines(x, "control", sets, False), None)


# ------------------------------------------------------------------------------------------------------------------------------ PART ONE (c): the thirty decisions

def cases_rule() -> str:
    return (f"THE RULE (fixed before any decision was read): in 78.82's stored games of reasons v control (set A in the order of the game ids, then set B if fewer than {CASES_PER_PART} games qualify), for each game the decisions of the reasons' "
            f"side where the pick differed from the search's first move and the position's units lie in the part (lower many: {MANY_LOW} to the cut; mid: 9 to 16); the {CASE_NTH}rd such decision of the game in that part (never the first), "
            f"one a game; the first {CASES_PER_PART} games by id for each part")


def stage_cases(root: Path, say) -> None:
    cut = load_cut(root)
    layer1 = GL.model(HD.GAME, HD.VARIANT)
    parts = (("lower many", cut["lower"][0], cut["lower"][1]), ("mid", 9, 16))
    out = []
    say("CASES " + cases_rule())
    for name, lo, hi in parts:
        got: list = []
        avail = 0
        for s in ("A", "B"):
            if len(got) >= CASES_PER_PART:
                break
            for g in sorted(_games82(s), key=lambda x: x["gid"]):
                cs = g["stats"][TM.seat_of(g, 0)]["cases"]
                c = LM.unit_counts(layer1, g["moves"]) if cs else []
                inpart = [x for x in cs if lo <= c[x["ply"]] <= hi]
                if len(inpart) >= CASE_NTH:
                    avail += 1
                    if len(got) < CASES_PER_PART:
                        got.append(dict(inpart[CASE_NTH - 1], gid=g["gid"], band=name, units=c[inpart[CASE_NTH - 1]["ply"]], decisions_in_part=len(inpart)))
        out += got
        say(f"CASES {name} ({lo} to {hi} units): {len(got)} decisions chosen from {avail} games with at least {CASE_NTH} such decisions; the pick equals the first move in 0 by construction")
    (root / "cases89.json").write_text(json.dumps(out))
    say(f"CASES written: {len(out)} decisions to {root / 'cases89.json'} (print with ladder_print --play89)")


# ------------------------------------------------------------------------------------------------------------------------------ PART TWO: the bins and the games

def bins_of(cut: dict) -> tuple:
    return ((0, 4), (5, 8), (9, 12), (13, 16), (MANY_LOW, cut["cut"]))


def label_of(b: tuple) -> str:
    return f"{b[0]}-{b[1]}"


def sample_file(root: Path, gen: bool = False) -> Path:
    return root / ("sample_gen.jsonl" if gen else "sample.jsonl")


def load_sample(root: Path) -> list:
    """The merged sample: the human positions, then the generated ones (the order the matrices of the events stage follow)."""
    out = SM.jsonl(sample_file(root))
    if sample_file(root, True).exists():
        out += SM.jsonl(sample_file(root, True))
    return out


def held_left() -> list:
    """The held-out games of 78.80 (78.59's file) less the ladder's games and less every game that has a kept position in the first sample."""
    games = SM.all_games()
    held = SM.held_out(games, SM.ladder_ordinals())
    used = {r["game"] for r in SM.jsonl(src80() / "sample.jsonl")}
    return [dict(g, origin="held") for g in held if g["ord"] not in used]


def extra_pool(root: Path) -> list:
    """Further games of the 78.59 shards, the same filter (HD.sample_shard), ranks past the file's own: ordinal = EXTRA_BASE + 8 x rank + shard place, interleaved so the pool mixes the shards. Cached (moves and the result only;
    the game number of the source is dropped here)."""
    f = root / "pool_extra.jsonl"
    if f.exists():
        return SM.jsonl(f)
    per = []
    for i in HD.SHARDS:
        games, _c = HD.sample_shard(i, HD.PER_SHARD + EXTRA_PER_SHARD, 0)
        per.append(games[HD.PER_SHARD:])
    out = []
    for rank in range(EXTRA_PER_SHARD):
        for si, games in enumerate(per):
            if rank < len(games):
                g = games[rank]
                out.append({"ord": EXTRA_BASE + 8 * rank + si, "moves": g["moves"], "y": g["y"], "origin": "extra"})
    f.write_text("\n".join(json.dumps(g) for g in out) + "\n")
    return out


def first_yield() -> dict:
    """The share of tried positions kept as a tie by bin of units in the first sample (78.80's tries): the starting guess of how many tries a bin needs."""
    tr = [r for r in SM.jsonl(src80() / "tries.jsonl") if r.get("ok")]
    out = {}
    for lo, hi in ((0, 4), (5, 8), (9, 12), (13, 16), (17, 99)):
        s = [r for r in tr if lo <= r["units"] <= hi]
        out[(lo, hi)] = max(0.2, sum(1 for r in s if r["tie"]) / max(1, len(s)))
    return out


def yield_of(b: tuple, table: dict) -> float:
    for (lo, hi), y in table.items():
        if lo <= b[0] <= hi:
            return y
    return 0.4


def _search_jobs(jobs: list, workers: int, say, label: str) -> list:
    t0 = time.perf_counter()
    rows = []
    with multiprocessing.get_context("spawn").Pool(workers) as pool:
        for i, r in enumerate(pool.imap(SM._try, jobs, chunksize=8), 1):
            rows.append(r)
            if i % 4000 == 0:
                say(f"progress {label}: {i} of {len(jobs)} positions tried, {time.perf_counter() - t0:.0f} s, load average {os.getloadavg()[0]:.2f}")
    return rows


def _keep(rows: list, meta: dict, games: dict, source: str, part_fixed: "str | None") -> "tuple[list, list]":
    """The kept positions (a tie, the move played among the equal moves) and the tries of one search round."""
    kept, tries = [], []
    for r in rows:
        mt = meta[(r["game"], r["ply"])]
        part = part_fixed or SM.part_of(r["game"])
        t = {"game": r["game"], "ply": r["ply"], "ok": r["ok"], "n_eq": r["n_eq"], "tie": r["tie"], "units": mt["units"], "bin": mt["bin"], "origin": mt["origin"], "part": part, "source": source}
        tries.append(t)
        if r["tie"]:
            kept.append({"id": f"s{r['game']}:{r['ply']}", "game": r["game"], "ply": r["ply"], "moves": games[r["game"]]["moves"][:r["ply"]], "tied": r["tied"], "move": r["move"], "played": r["played"],
                         "units": mt["units"], "part": part, "source": source, "origin": mt["origin"], "bin": mt["bin"]})
    return kept, tries


def _draw(pool_slice: list, layer1, bins: tuple, short: dict, need: dict, planned: Tally, part_fixed: "str | None", games: dict, meta: dict) -> list:
    """The jobs (ordinal, ply, moves, played) of the games in `pool_slice`: the stratified plies of each game in the bins that are still short, until the planned tries of a (bin, part) reach the need."""
    jobs = []
    for g in pool_slice:
        part = part_fixed or SM.part_of(g["ord"])
        c = LM.unit_counts(layer1, g["moves"])
        games[g["ord"]] = g
        for bi, plies in SM.plies_stratified(g["ord"], c, bins, PER_GAME_BIN, len(g["moves"]) - 1).items():
            key = (bi, part)
            if key not in short or planned[key] >= need[key]:
                continue
            for k in plies:
                jobs.append((g["ord"], k, g["moves"][:k], g["moves"][k]))
                meta[(g["ord"], k)] = {"units": c[k], "bin": label_of(bins[bi]), "origin": g["origin"]}
                planned[key] += 1
    return jobs


def stage_sample(root: Path, workers: int, target: int, say, max_pool: int = 0) -> None:
    cut = load_cut(root)
    bins = bins_of(cut)
    layer1 = GL.model(HD.GAME, HD.VARIANT)
    left = held_left()
    extra = extra_pool(root)
    pool = left + extra
    if max_pool:
        pool = pool[:max_pool]
    say(f"POOL: held-out games of 78.80 less the ladder's games and less every game with a kept position in the first sample: {len(left)}; further games of the same shards (same filter, ranks past the 78.59 file's own): {len(extra)}; "
        f"bins of units {[label_of(b) for b in bins]} (the lower many part is {MANY_LOW} to {cut['cut']}); target {target} tie positions a bin in each part; at most {PER_GAME_BIN} positions of a game in a bin; load average {os.getloadavg()[0]:.2f}; {workers} workers")
    yld = first_yield()
    index_of = {label_of(b): i for i, b in enumerate(bins)}
    games: dict = {}
    meta: dict = {}
    kept_all, tries_all = [], []
    kept = Tally()
    tried = Tally()
    ptr, rnd = 0, 0
    t0 = time.perf_counter()
    while True:
        short = {(bi, p): target - kept[(bi, p)] for bi in range(len(bins)) for p in ("find", "confirm") if target - kept[(bi, p)] > 0}
        if not short or ptr >= len(pool):
            break
        rnd += 1
        need = {}
        for (bi, p), v in short.items():
            y = max(0.15, kept_rate(bi, tries_all, yld, bins))
            need[(bi, p)] = int(math.ceil(v / y * FIND_YIELD))
        planned: Tally = Tally()
        jobs: list = []
        while ptr < len(pool) and any(planned[k] < need[k] for k in short):
            jobs += _draw(pool[ptr:ptr + BLOCK], layer1, bins, short, need, planned, None, games, meta)
            ptr += BLOCK
        rows = _search_jobs(jobs, workers, say, f"round {rnd}")
        k_new, t_new = _keep(rows, meta, games, "human", None)
        kept_all += k_new
        tries_all += t_new
        for r in k_new:
            kept[(index_of[r["bin"]], r["part"])] += 1
        say(f"ROUND {rnd}: shortfalls {sorted((label_of(bins[b]), p, v) for (b, p), v in short.items())}; planned tries {len(jobs)} from games up to pool place {ptr} of {len(pool)}; kept {len(k_new)}; "
            f"{time.perf_counter() - t0:.0f} s; load average {os.getloadavg()[0]:.2f}")
    sample_file(root).write_text("\n".join(json.dumps(r) for r in kept_all) + ("\n" if kept_all else ""))
    (root / "tries.jsonl").write_text("\n".join(json.dumps(r) for r in tries_all) + ("\n" if tries_all else ""))
    say(f"SAMPLE done: {len(tries_all)} positions tried, {len(kept_all)} kept as ties with the move played among the equal moves; games used {len({r['game'] for r in tries_all})} of the pool's {ptr}; rounds {rnd}; {time.perf_counter() - t0:.0f} s")


def kept_rate(bi: int, tries: list, table: dict, bins: tuple) -> float:
    lab = label_of(bins[bi])
    s = [t for t in tries if t["bin"] == lab and t["ok"]]
    if len(s) < 200:
        return yield_of(bins[bi], table)
    return sum(1 for t in s if t["tie"]) / len(s)


def generated_pool() -> list:
    """78.58's plain search v plain search games (estimator_fit.SETS, the sets in GEN_SETS), the result known (a game with no result is left out by estimator_fit.games_of); the game ordered by a hash of its moves."""
    import hashlib
    saved = EF.SETS
    EF.SETS = [s for s in saved if s[0] in GEN_SETS]
    try:
        games, _lines = EF.games_of(0)
    finally:
        EF.SETS = saved
    games = sorted(games, key=lambda g: hashlib.md5(" ".join(g["moves"]).encode()).hexdigest())
    return [{"ord": GEN_BASE + i, "moves": g["moves"], "y": g["result"], "origin": "generated"} for i, g in enumerate(games)]


def stage_fill(root: Path, workers: int, target: int, say) -> None:
    cut = load_cut(root)
    bins = bins_of(cut)
    layer1 = GL.model(HD.GAME, HD.VARIANT)
    human = SM.jsonl(sample_file(root))
    tries = SM.jsonl(root / "tries.jsonl")
    index_of = {label_of(b): i for i, b in enumerate(bins)}
    kept = Tally((index_of[r["bin"]], r["part"]) for r in human)
    short = {(bi, "find"): target - kept[(bi, "find")] for bi in range(len(bins)) if target - kept[(bi, "find")] > 0}
    pool = generated_pool()
    say(f"FILL: FIND bins short of {target} human tie positions: {sorted((label_of(bins[b]), v) for (b, _p), v in short.items())}; generated pool: {len(pool)} games of 78.58's plain search v plain search sets {list(GEN_SETS)}, the result known; "
        "used for FIND only; the move played is the mover's own move in the game")
    if not short:
        sample_file(root, True).write_text("")
        return
    yld = first_yield()
    need = {k: int(math.ceil(v / max(0.15, kept_rate(k[0], tries, yld, bins)) * FIND_YIELD)) for k, v in short.items()}
    games: dict = {}
    meta: dict = {}
    planned: Tally = Tally()
    jobs: list = []
    ptr = 0
    while ptr < len(pool) and any(planned[k] < need[k] for k in short):
        jobs += _draw(pool[ptr:ptr + BLOCK], layer1, bins, short, need, planned, "find", games, meta)
        ptr += BLOCK
    t0 = time.perf_counter()
    rows = _search_jobs(jobs, workers, say, "fill")
    kept_new, tries_new = _keep(rows, meta, games, "generated", "find")
    sample_file(root, True).write_text("\n".join(json.dumps(r) for r in kept_new) + ("\n" if kept_new else ""))
    (root / "tries_gen.jsonl").write_text("\n".join(json.dumps(r) for r in tries_new) + ("\n" if tries_new else ""))
    say(f"FILL done: {len(jobs)} generated positions tried from {min(ptr, len(pool))} of the {len(pool)} games, {len(kept_new)} kept as ties with the mover's own move among the equal moves; {time.perf_counter() - t0:.0f} s")


def stage_coverage(root: Path, target: int, say) -> None:
    cut = load_cut(root)
    bins = bins_of(cut)
    tries = SM.jsonl(root / "tries.jsonl")
    tg = SM.jsonl(root / "tries_gen.jsonl") if (root / "tries_gen.jsonl").exists() else []
    human = SM.jsonl(sample_file(root))
    gen = SM.jsonl(sample_file(root, True)) if sample_file(root, True).exists() else []
    first = SM.jsonl(src80() / "sample.jsonl")
    say(f"COVERAGE of the second sample by bin of units on the board; the aim is {target} tie positions a bin in each part; human positions tried / kept as ties, then generated:")
    for b in bins:
        lab = label_of(b)
        tr = [t for t in tries if t["bin"] == lab and t["ok"]]
        hk = [r for r in human if r["bin"] == lab]
        gk = [r for r in gen if r["bin"] == lab]
        gt = [t for t in tg if t["bin"] == lab and t["ok"]]
        hf, hc = [r for r in hk if r["part"] == "find"], [r for r in hk if r["part"] == "confirm"]
        fa = len(hf) + len(gk)
        say(f"COVERAGE bin {lab}: human tried {len(tr)}, kept {len(hk)} (FIND {len(hf)} in {len({r['game'] for r in hf})} games, CONFIRM {len(hc)} in {len({r['game'] for r in hc})} games; "
            f"held-out games of the first sample's file {sum(1 for r in hk if r['origin'] == 'held')}, further games {sum(1 for r in hk if r['origin'] == 'extra')}); short of {target}: FIND {max(0, target - len(hf))}, CONFIRM {max(0, target - len(hc))}; "
            f"generated: tried {len(gt)}, kept {len(gk)} in {len({r['game'] for r in gk})} games; FIND of the bin {fa} of which generated {len(gk)} ({100 * len(gk) / max(1, fa):.1f} pct); the first sample held "
            f"{sum(1 for r in first if b[0] <= r['units'] <= b[1])} in this bin")
    few = [r for r in human if r["units"] <= 8]
    say(f"COVERAGE the few band (8 units or fewer): human kept {len(few)} (FIND {sum(1 for r in few if r['part'] == 'find')}, CONFIRM {sum(1 for r in few if r['part'] == 'confirm')}) against the aim of {2 * target} in each of the two bins 0-4 and 5-8 "
        f"(the first sample held {sum(1 for r in first if r['units'] <= 8)}); generated kept {sum(1 for r in gen if r['units'] <= 8)}")
    say(f"COVERAGE all: human {len(human)} positions in {len({r['game'] for r in human})} games ({sum(1 for r in human if r['part'] == 'find')} FIND, {sum(1 for r in human if r['part'] == 'confirm')} CONFIRM); "
        f"generated {len(gen)} positions in {len({r['game'] for r in gen})} games (FIND only)")


# ------------------------------------------------------------------------------------------------------------------------------ PART TWO: the events, the cells, the controls

def _sizes(sample: list) -> np.ndarray:
    return np.array([len(r["tied"]) for r in sample])


def stage_events(root: Path, workers: int, say) -> None:
    sample = load_sample(root)
    reasons = json.loads((SGM.src81() / "reasons.json").read_text())
    keys80 = json.loads((src80() / "keys.json").read_text())
    n_full = len(keys80)
    ukeys = sorted({int(r["key"]) for r in reasons})
    col = {k: j for j, k in enumerate(ukeys)}
    model = GL.model(HD.GAME, HD.VARIANT)
    counter = SY.Counter(model, ukeys, G2M.goal_kind_of(HD.VARIANT), 2, (0, 1, 2, 3), workers)
    t0 = time.perf_counter()
    parts = []
    for i in range(0, len(sample), 300):
        chunk = sample[i:i + 300]
        parts.append(SY.deltas_matrix([{"moves": r["moves"]} for r in chunk], [r["tied"] for r in chunk], counter))
    M = sparse.vstack(parts, format="csc")
    ev_cols, ev_rows, ev_new = [], [], []
    for r in reasons:
        j = SY.event_index(col[int(r["key"])], r["side"], r["dir"], len(ukeys))
        c = M[:, j]
        ev_rows.append(c.indices)
        ev_new.append(np.full(len(c.indices), int(r["event"])))
    ri = np.concatenate(ev_rows) if ev_rows else np.zeros(0, int)
    ci = np.concatenate(ev_new) if ev_new else np.zeros(0, int)
    full = sparse.csr_matrix((np.ones(len(ri), np.float32), (ri, ci)), shape=(M.shape[0], 4 * n_full))
    sparse.save_npz(root / "events.npz", full)
    say(f"EVENTS: {len(sample)} positions, {M.shape[0]} tied moves; the {len(reasons)} reasons of 78.81 over {len(ukeys)} patterns, laid out as 78.80's event matrix; {time.perf_counter() - t0:.1f} s on {workers} threads")
    keys, raws = SGM.split_patterns(reasons)
    chunks = [[(r["moves"], r["tied"]) for r in sample[i:i + 25]] for i in range(0, len(sample), 25)]
    t1 = time.perf_counter()
    res = []
    with multiprocessing.get_context("spawn").Pool(workers) as pool:
        for part in pool.imap(SGM._split_job, [(c, raws) for c in chunks], chunksize=1):
            res += part
    r_i = [i for i, cols in enumerate(res) for _c in cols]
    c_i = [c for cols in res for c in cols]
    C = sparse.csr_matrix((np.ones(len(r_i), np.float32), (r_i, c_i)), shape=(len(res), 8 * len(keys)))
    sparse.save_npz(root / "cells.npz", C)
    (root / "cells_keys.json").write_text(json.dumps({"keys": keys}))
    say(f"CELLS: {len(keys)} patterns of the form 'a unit of the mover's bears on a unit of theirs', {len(res)} tied moves; {time.perf_counter() - t1:.1f} s")
    chunks = [[(r["moves"], r["tied"]) for r in sample[i:i + 100]] for i in range(0, len(sample), 100)]
    with multiprocessing.get_context("spawn").Pool(workers) as pool:
        flat = [x for p in pool.map(SR._controls_job, chunks, chunksize=1) for x in p]
    old_kinds = json.loads((SGM.src81() / "controls_kinds.json").read_text())
    new_kinds = sorted({k for _t, k in flat})
    if not set(new_kinds) <= set(old_kinds):
        raise ValueError(f"kinds of the new sample not in 78.81's controls: {sorted(set(new_kinds) - set(old_kinds))}")
    np.savez_compressed(root / "controls.npz", takes=np.array([t for t, _k in flat], np.uint8), kind=np.array([old_kinds.index(k) for _t, k in flat], np.int16))
    say(f"CONTROLS: {len(flat)} tied moves; the move takes a unit in {100 * np.mean([t for t, _k in flat]):.1f} pct of them; total {time.perf_counter() - t0:.0f} s")


def load_data(root: Path) -> dict:
    """A `data` dict in the shape symbols_groups_measure's functions read (inp: sample, M, n; ctl; reasons), for the second sample."""
    sample = load_sample(root)
    z = np.load(root / "controls.npz", allow_pickle=False)
    ctl = {"takes": z["takes"].astype(float), "kind": z["kind"].astype(int), "kinds": json.loads((SGM.src81() / "controls_kinds.json").read_text())}
    reasons = json.loads((SGM.src81() / "reasons.json").read_text())
    M = sparse.load_npz(root / "events.npz").tocsr()
    return {"inp": {"sample": sample, "M": M, "n": M.shape[1] // 4}, "ctl": ctl, "reasons": reasons, "n": M.shape[1] // 4}


def rows89(data: dict, E, part: str) -> SY.Choice:
    """The choice rows of one part of the second sample (symbols_groups_measure.rows_of), the source of each row the sample's own (human or generated); fill_ok says the fit may read the generated rows."""
    rows = SGM.rows_of(data, E, part)
    src = np.array([r["source"] for r in data["inp"]["sample"] if r["part"] == part for _m in r["tied"]])
    rows.source = src
    rows.fill_ok = part == "find"
    return rows


# ------------------------------------------------------------------------------------------------------------------------------ PART TWO: the fits

def stage_fit(root: Path, say) -> None:
    data = load_data(root)
    cells = SGM.load_cells(root)
    old = pickle.load(open(src88() / "fits.pkl", "rb"))
    groups = old["variants"]["ii"]["groups"]
    _R, E = SGM.reason_columns(data, True, cells)
    Eg = SY.group_matrix(E, groups)
    rows = rows89(data, Eg, "find")
    names = tuple(rows.control_names)
    npos = rows.n_positions()
    gen = int((np.asarray(rows.source) == "generated").sum())
    say(f"FIT inputs: the reasons as they are (variant ii of 78.88: {len(_R)} reasons in {len(groups)} groups, with the split); the new FIND holds {npos} positions, {len(rows.pos)} tied moves, of which generated rows {gen}; seed {SEED}")
    out: dict = {"groups": groups, "controls": list(names), "fits": {}}
    t0 = time.perf_counter()
    f = SY.fit_choice(rows, controls=names, events=None, seed=SEED)
    out["fits"]["all"] = f
    cv = f["cv"][f["pick"]]
    say(f"FIT all positions: {len(f['events'])} of {f['n_events']} groups with a weight; penalty {f['lambda']:.5f}; held out inside FIND {cv[1]:.4f} (standard error {cv[2]:.4f}); {time.perf_counter() - t0:.0f} s")
    for b, lo, hi in SGM.BANDS:
        t1 = time.perf_counter()
        fb = SY.fit_choice(rows, controls=names, events=None, seed=SEED, band=(lo, hi))
        fb["band"] = [lo, hi]
        out["fits"][b] = fb
        cvb = fb["cv"][fb["pick"]] if fb.get("cv") else None
        say(f"FIT band {b} ({lo}-{hi} units): {fb['positions']} positions; {len(fb['events'])} of {fb['n_events']} groups with a weight; penalty {fb['lambda']:.5f}; "
            + (f"held out inside FIND {cvb[1]:.4f} (standard error {cvb[2]:.4f}); " if cvb else "no path (too few positions); ") + f"{time.perf_counter() - t1:.0f} s")
    po = old["variants"]["ii"]["fits"]["all"]
    say(f"FIT old weights for reference: {len(po['events'])} of {po['n_events']} groups with a weight, penalty {po['lambda']:.5f} (fitted on 78.80's FIND, {po['positions']} positions)")
    pickle.dump(out, open(root / "fits.pkl", "wb"))


# ------------------------------------------------------------------------------------------------------------------------------ PART TWO: the scores

def stage_score(root: Path, say) -> dict:
    data = load_data(root)
    cells = SGM.load_cells(root)
    new = pickle.load(open(root / "fits.pkl", "rb"))
    old = pickle.load(open(src88() / "fits.pkl", "rb"))
    groups = old["variants"]["ii"]["groups"]
    _R, E = SGM.reason_columns(data, True, cells)
    rows = rows89(data, SY.group_matrix(E, groups), "confirm")
    P = rows.n_positions()
    sample_c = [r for r in data["inp"]["sample"] if r["part"] == "confirm"]
    assert all(r["source"] == "human" for r in sample_c)
    game = np.array([r["game"] for r in sample_c])
    units = np.array([r["units"] for r in sample_c])
    band = np.array([LM.band_of(int(u)) for u in units])
    some = _some_take(rows, P)
    first_hit = np.array([r["tied"][0] == r["played"] for r in sample_c], float)
    variants = {"old weights": ("old", old["variants"]["ii"]["fits"]), "new, one fit": ("new", {"all": new["fits"]["all"]}), "new, by band": ("newband", {b: new["fits"][b] for b, _l, _h in SGM.BANDS})}
    hits: dict = {"search first": first_hit}
    for name, (tag, fits) in variants.items():
        for ctl_in in (False, True):
            hits[(name, ctl_in)] = SGM.hit_vector(rows, SGM.picks_of(rows, fits, ctl_in))
    cells_def = [("all", np.ones(P, bool))]
    for b, _l, _h in SGM.BANDS:
        cells_def += [(b, band == b), (f"{b}, quiet", (band == b) & ~some), (f"{b}, some tied move takes a unit", (band == b) & some)]
    cells_def += [("all, quiet", ~some), ("all, some tied move takes a unit", some)]
    say(f"SCORE on the new CONFIRM (human positions only): {P} positions in {len(set(game.tolist()))} games; by band " + ", ".join(f"{b} {int((band == b).sum())}" for b, _l, _h in SGM.BANDS)
        + "; the FIND positions that fitted the new weights are of other games (the part of a game is a hash of its ordinal)")
    better: list = []
    for ctl_in, kind in ((False, "pick without the controls"), (True, "pick with the controls")):
        for label, sel in cells_def:
            m = int(sel.sum())
            if m < 5:
                say(f"SCORE {label} ({kind}): {m} positions (too few)")
                continue
            parts = [f"search first {SM.wilson_line(first_hit[sel].sum(), m)}"] + [f"{n} {SM.wilson_line(hits[(n, ctl_in)][sel].sum(), m)}" for n in variants]
            pd = []
            for n in variants:
                d = SM.paired(hits[(n, ctl_in)][sel], first_hit[sel], game[sel], SEED + 3)
                pd.append(f"{n} less first {100 * d[0]:+.2f} [{100 * d[1]:+.2f}, {100 * d[2]:+.2f}]")
            for n in ("new, one fit", "new, by band"):
                d = SM.paired(hits[(n, ctl_in)][sel], hits[("old weights", ctl_in)][sel], game[sel], SEED + 4)
                pd.append(f"{n} less old {100 * d[0]:+.2f} [{100 * d[1]:+.2f}, {100 * d[2]:+.2f}]")
                if not ctl_in and label in ("few", "mid", "many") and d[1] > 0:
                    better.append((label, n, d))
            say(f"SCORE {label} ({kind}, {m} positions): " + "; ".join(parts) + "; paired (points, 90 pct bootstrap over games): " + "; ".join(pd))
    below = [x for x in better]
    say("PLAY CONDITION (agreement in a band below the upper many part rises over the old weights, the pick without the controls, paired band clear of zero): "
        + (f"MET in {[(b, n, f'{100 * d[0]:+.2f} [{100 * d[1]:+.2f}, {100 * d[2]:+.2f}]') for b, n, d in below]}" if below else "NOT MET in any band (few, mid, lower many) by either new fit"))
    # the old weights on the old CONFIRM, beside their score on the new one
    res = SGM.stage_score(src88(), lambda m: None)
    hb = res["hits"][("ii", False)]
    fb = res["hits_base"]["search first"]
    og, ob, osome = res["game"], res["band"], res["some_take"]
    say("OLD WEIGHTS on the old CONFIRM (78.80's, positions drawn early in the game) beside the new CONFIRM, pick without the controls: agreement with the move played, and the search's first move on the same positions")
    for label, so, sn in [("all", np.ones(len(ob), bool), np.ones(P, bool))] + [(b, ob == b, band == b) for b, _l, _h in SGM.BANDS] + [(f"{b}, quiet", (ob == b) & ~osome, (band == b) & ~some) for b, _l, _h in SGM.BANDS]:
        mo, mn = int(so.sum()), int(sn.sum())
        if mo < 5 or mn < 5:
            say(f"OLD v NEW {label}: old CONFIRM {mo} positions, new CONFIRM {mn} positions (too few)")
            continue
        say(f"OLD v NEW {label}: old weights on the old CONFIRM {SM.wilson_line(hb[so].sum(), mo)}, search first there {SM.wilson_line(fb[so].sum(), mo)}; old weights on the new CONFIRM {SM.wilson_line(hits[('old weights', False)][sn].sum(), mn)}, "
            f"search first there {SM.wilson_line(first_hit[sn].sum(), mn)}")
    return {"better": better}


# ------------------------------------------------------------------------------------------------------------------------------ main

def main(argv: "list | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stage", choices=("first", "cut", "cases", "sample", "fill", "coverage", "events", "fit", "score", "match", "report"))
    ap.add_argument("--tag", default="short")
    ap.add_argument("--a", default="manyhi", choices=("manyhi", "manylo"))
    ap.add_argument("--set", default="A", choices=("A", "B"))
    ap.add_argument("--pairs", type=int, default=100)
    ap.add_argument("--seconds", type=float, default=TM.SECONDS)
    ap.add_argument("--workers", type=int, default=9)
    ap.add_argument("--target", type=int, default=TARGET)
    ap.add_argument("--max-pool", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(argv)
    root = out_root(a.tag)
    say = Say(root, a.stage)
    t0 = time.perf_counter()
    say(f"stage start; tag {a.tag}; load average {os.getloadavg()[0]:.2f}")
    if a.stage == "first":
        stage_first(root, say, a.limit)
    elif a.stage == "cut":
        stage_cut(root, say)
    elif a.stage == "cases":
        stage_cases(root, say)
    elif a.stage == "sample":
        stage_sample(root, a.workers, a.target, say, a.max_pool)
    elif a.stage == "fill":
        stage_fill(root, a.workers, a.target, say)
    elif a.stage == "coverage":
        stage_coverage(root, a.target, say)
    elif a.stage == "events":
        stage_events(root, a.workers, say)
    elif a.stage == "fit":
        stage_fit(root, say)
    elif a.stage == "score":
        stage_score(root, say)
    elif a.stage == "match":
        stage_match(root, a.a, a.set, a.pairs, a.workers, a.seconds, say)
    elif a.stage == "report":
        stage_report(root, a.seconds, a.workers)
    say(f"stage done in {time.perf_counter() - t0:.0f} s; load average {os.getloadavg()[0]:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
