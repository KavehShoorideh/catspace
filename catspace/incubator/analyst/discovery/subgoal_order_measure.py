"""PLAN 78.51: THE SAME CHOICE AT NO COST.  The fitted number of plan 78.49 (the weights file of plan 78.50, unchanged) as the ORDER in which the search reads the root moves.  A measurement; the wake and every
existing player are unchanged, the order is passed by this script only.  Game-free: the game is named only by the caller (`--variant`).

The search keeps the first root move of the best value (`v > best`, rust/engine/tactics/src/ledger.rs).  Hand it the root moves in the order of the number (highest first) and its own pick is the
highest-numbered move among the moves it holds equal, with no second reading (78.50 paid more than half the clock for that reading).  The Python entries `cs_engine.solve_batch` and
`cs_engine.solve_equals_batch` take `root_orders` (per position: None, or a permutation of the position's legal-move indices, first read first) and `root_order_mode`:

  "strict"  the order REPLACES the root's reading order: the previous deepening's best is not read first, the lemma / claim orders and the root's late-move reduction are not used at the root.
            Then the move kept is exactly the first move IN THE GIVEN ORDER among those whose value at the finished depth equals the best (tested against `solve_equals_batch`, tests/test_plan78_51_root_order.py).
  "ties"    the stable variant: the root's own order stays and the given order only breaks ties inside the groups it treats alike (same rank, same gain, same killer or history standing).  The move kept is the
            previous deepening's best while it still holds the best value (the search reads it first), else the first of the best value in that order: sticky, not "the highest number among the equals".

Stage B (`--stage b`): on about 1000 positions of the kept games, the depth the ledger reaches and its nodes at the clock, today's order against the number's order and against a random order, each in both modes,
and "today" read twice (the noise of the clock).  The variant that goes to the matches is decided from these lines alone and written into the docs before any match.
Stage C (`--stage match`): the matches of plan 78.50 (300 opening pairs, both colours, the same clock both sides, the same move cap, bands by bootstrap over opening pairs), seeds from 9000.
Stage D (`--stage arrival`, `--stage differs`): the arrival of confirmed claims in 60 games of a match (as 78.50), and how often the ordered player's move differs from what the control's default call plays on the same position.

  .venv/bin/python -m catspace.incubator.analyst.discovery.subgoal_order_measure --stage b|match|arrival|differs|report --variant V ...
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

import cs_engine as CE

from catspace.incubator.analyst.discovery import forcing_opportunity_measure as FO
from catspace.incubator.analyst.discovery import sheet_player as SP
from catspace.incubator.analyst.discovery import subgoal_play_measure as PM
from catspace.io import paths

FIRST_SEED = 9000
MODES = ("strict", "ties")
MATCH_PLAYERS = tuple(f"{k}_{m}" for k in ("ordered", "random_order") for m in MODES)
MATCHES = {f"{k}_v_control" if m == "strict" else f"{k}_{m}_v_control": (f"{k}_{m}", "control", FIRST_SEED) for k in ("ordered", "random_order") for m in MODES}
B_SAMPLE = 1000
B_SEED = 5151
PLUGIN = "catspace.incubator.analyst.discovery.subgoal_order_measure"
DEFAULT_WEIGHTS = "subgoal_play_78_50/full/weights_horizon8.json"          # the 78.50 weights, unchanged (under artifacts/experiments)

PM.MATCHES.update(MATCHES)
PM.WEIGHTED.update(k for k in MATCH_PLAYERS if k.startswith("ordered"))
PM.PLUGIN.update({name: PLUGIN for name in MATCHES})


def out_root() -> Path:
    return Path(paths.experiment("subgoal_play_78_51"))


# --------------------------------------------------------------------------------------------------------------------------- the order

def order_of(numbers: dict, n: int) -> list:
    """The root moves (indices 0..n-1) best number first, a tie at the same number in the engine's own index order; a move with no number (the position after it is over) is read after the numbered ones."""
    numbered = sorted(numbers, key=lambda i: (-numbers[i], i))
    rest = [i for i in range(n) if i not in numbers]
    return numbered + rest


def random_order(n: int, rng: random.Random) -> list:
    o = list(range(n))
    rng.shuffle(o)
    return o


def numbers_at(sheet, weights: list, moves: list, probe: str) -> dict:
    """legal-move index -> the fitted number of the position after that move, for the position after `moves` (`probe`: any legal move there: it only makes the position a played one for the reader)."""
    res = sheet.subgoal_children([(None, list(moves) + [probe])], 1, [[len(moves)]], weights)
    rows = res[0]
    return dict(rows[0][2]) if rows else {}


# --------------------------------------------------------------------------------------------------------------------------- the search with an order

class OrderedSearch(SP.PlainSearch):
    """The wake's plain call (`PlainSearch._solve`: the same arguments, the same trace) with the root order of the position passed on; `orders` None is the plain call."""

    orders: "list | None" = None
    mode: str = "strict"

    def _solve(self, positions: list, seconds: float, threads: int) -> tuple:
        if os.path.exists(self.trace_path):
            os.remove(self.trace_path)
        labels = [f"s:{i}:0:0" for i in range(len(positions))]
        extra = {} if self.orders is None else {"root_orders": self.orders, "root_order_mode": self.mode}
        out = CE.solve_batch(positions, seconds, snapshot="s", trace_path=self.trace_path, threads=threads, prior=False, ties=False, reason=False, wdl=False, wdl_table=None,
                             store=self.store, specs=self.specs, labels=labels, **extra)
        return out, self._rows()


def ledger_step(row: dict) -> dict:
    return next((s for s in row.get("steps", []) if s.get("operator") == "ledger"), {})


class OrderedPlayer:
    """The plain search at the full clock with the root moves in an order: the fitted number's ('ordered_*') or a random one ('random_order_*'); the suffix is the mode.  The number's reading time is
    taken off the clock by the running estimate of the game (as the 78.50 candidate); the random order costs nothing."""

    def __init__(self, w: dict, seconds: float, kind: str, seed: str):
        self.w, self.seconds, self.kind, self.seed = w, seconds, kind, seed
        self.mode = kind.rsplit("_", 1)[1]
        self.by_number = kind.startswith("ordered")
        self.costs: list = []
        key = ("ordered_search", self.w["variant"])
        if key not in w:
            base = w["search"]
            w[key] = OrderedSearch(base.variant, store=base.store, specs=base.specs, trace_path=base.trace_path + ".ordered")
        self.search = w[key]

    def clock(self) -> float:
        if not self.by_number:
            return self.seconds
        est = sum(self.costs) / len(self.costs) if self.costs else 0.005
        return max(0.5 * self.seconds, self.seconds - est)

    def choose(self, moves: list) -> tuple:
        w = self.w
        legal = CE.state(w["variant"], None, list(moves))[2]
        order = None
        cost = 0.0
        if len(legal) >= 2:
            t0 = time.perf_counter()
            if self.by_number:
                order = order_of(numbers_at(w["sheet"], w["weights"], moves, legal[0]), len(legal))
            else:
                order = random_order(len(legal), random.Random(f"{self.seed}:{len(moves)}"))
            cost = time.perf_counter() - t0
            if self.by_number:
                self.costs.append(cost)
        self.search.orders = None if order is None else [order]
        self.search.mode = self.mode
        out, rows = self.search._solve([(w["variant"], None, list(moves))], self.clock(), 1)
        led = ledger_step(rows.get(0, {}))
        return out[0][0], {"depth": led.get("depth"), "legal": len(legal), "ordered": int(order is not None), "cost": cost}


for _kind in MATCH_PLAYERS:
    PM.EXTRA_PLAYERS[_kind] = OrderedPlayer


# --------------------------------------------------------------------------------------------------------------------------- stage B: the cost of the order

def sample_positions(records: list, n: int, seed: int) -> list:
    """`n` (game id, ply) pairs drawn without replacement from every position before a move of the kept games, seeded."""
    allp = [(r["game"], t) for r in records for t in range(len(r["moves"]))]
    rng = random.Random(seed)
    return sorted(rng.sample(allp, min(n, len(allp))))


def numbers_for_sample(variant: str, records: list, sample: list, weights: list, threads: int) -> dict:
    """(game, ply) -> {legal index: number}, read through the games (outside the search's clock)."""
    from catspace.incubator.analyst.discovery import question_sheet as QS
    from catspace.incubator.analyst.discovery import sheet_rust as SR
    sheet = SR.model(QS.Env(variant), QS.store_worth(variant))
    by_game: dict = {}
    for g, t in sample:
        by_game.setdefault(g, []).append(t)
    recs = {r["game"]: r for r in records}
    ids = sorted(by_game)
    res = sheet.subgoal_children([(recs[g].get("start"), list(recs[g]["moves"])) for g in ids], threads, [sorted(by_game[g]) for g in ids], weights)
    out = {}
    for g, rows in zip(ids, res):
        for ply, _played, vals in rows:
            out[(g, ply)] = dict(vals)
    return out


def b_variants(sample: list, positions: list, numbers: dict, legal: list, seed: int) -> dict:
    """variant name -> the per-position orders (None: today's order), for the batch of `positions`."""
    rng = random.Random(seed)
    num = [order_of(numbers.get(k, {}), len(lg)) if len(lg) >= 2 else None for k, lg in zip(sample, legal)]
    rnd = [random_order(len(lg), rng) if len(lg) >= 2 else None for lg in legal]
    return {"today": (None, None), "today_again": (None, None), "number_strict": (num, "strict"), "random_strict": (rnd, "strict"), "number_ties": (num, "ties"), "random_ties": (rnd, "ties")}


def read_batch(search: OrderedSearch, positions: list, seconds: float, threads: int, orders, mode: str) -> list:
    """-> per position: (the move, ledger depth, ledger nodes, all nodes)."""
    search.orders, search.mode = orders, mode or "strict"
    out, rows = search._solve(positions, seconds, threads)
    res = []
    for i, r in enumerate(out):
        led = ledger_step(rows.get(i, {}))
        res.append((r[0], led.get("depth"), led.get("nodes"), r[2]))
    return res


def stage_b(variant: str, records: list, weights: list, seconds: float, threads: int, scratch: Path, n: int = B_SAMPLE) -> "tuple[list, dict]":
    from catspace.incubator.analyst.discovery import game_store as GS
    store = str(GS.path_of(variant, GS.head(variant))) if GS.head(variant) else None
    specs = GS.spec_texts(variant) if store else {}
    scratch.mkdir(parents=True, exist_ok=True)
    search = OrderedSearch(variant, store=store, specs=specs, trace_path=str(scratch / "trace_b.jsonl"))
    sample = sample_positions(records, n, B_SEED)
    recs = {r["game"]: r for r in records}
    positions = [(variant, None, list(recs[g]["moves"][:t])) for g, t in sample]
    legal = [CE.state(variant, None, p[2])[2] for p in positions]
    t0 = time.perf_counter()
    numbers = numbers_for_sample(variant, records, sample, weights, threads)
    t_num = time.perf_counter() - t0
    variants = b_variants(sample, positions, numbers, legal, B_SEED)
    reads = {}
    for name, (orders, mode) in variants.items():
        t1 = time.perf_counter()
        reads[name] = read_batch(search, positions, seconds, threads, orders, mode)
        print(f"OBS progress stage B: {name} read in {time.perf_counter() - t1:.0f} s", flush=True)
    base = reads["today"]
    lines = [f"OBS subgoal_order stage B: {len(positions)} positions drawn (seed {B_SEED}) from {len(records)} kept games, {sum(1 for lg in legal if len(lg) >= 2)} with two or more legal moves; the search at {seconds} s a move on {threads} threads, no equal reading; "
             f"the number of every legal move of the {len(numbers)} positions read in {t_num:.1f} s in all ({1000 * t_num / max(1, len(numbers)):.1f} ms a position, outside the clock here, inside the candidate's in a match)"]
    data = {}
    for name, rd in reads.items():
        d = np.array([x[1] if x[1] is not None else 0 for x in rd], dtype=float)
        ln = np.array([x[2] if x[2] is not None else 0 for x in rd], dtype=float)
        an = np.array([x[3] for x in rd], dtype=float)
        bd = np.array([x[1] if x[1] is not None else 0 for x in base], dtype=float)
        bn = np.array([x[2] if x[2] is not None else 0 for x in base], dtype=float)
        same = sum(1 for x, y in zip(rd, base) if x[0] == y[0])
        data[name] = {"depth": float(d.mean()), "ledger_nodes": float(ln.mean()), "nodes": float(an.mean())}
        lines.append(f"OBS subgoal_order stage B: order {name}: mean ledger depth {d.mean():.3f} (median {np.median(d):.1f}), mean ledger nodes {ln.mean():.0f}, mean all nodes {an.mean():.0f}; against today's order on the same positions: "
                     f"deeper in {int((d > bd).sum())}, shallower in {int((d < bd).sum())}, same in {int((d == bd).sum())}; mean depth difference {float((d - bd).mean()):+.3f}; same move as today's order in {same} of {len(rd)}")
    return lines, data


# --------------------------------------------------------------------------------------------------------------------------- stage D: differs from the control's call

def stage_d_differs(variant: str, games: list, who: str, seconds: float, threads: int, scratch: Path, chunk: int = 200) -> list:
    """For every decision of player `who` in `games`: its move in the game, the control's default call on the same position (a fresh call at the game's clock, outside the game's clock), and the control's
    default call a second time (the noise of the clock).  -> OBS lines."""
    from catspace.incubator.analyst.discovery import game_store as GS
    store = str(GS.path_of(variant, GS.head(variant))) if GS.head(variant) else None
    specs = GS.spec_texts(variant) if store else {}
    scratch.mkdir(parents=True, exist_ok=True)
    search = OrderedSearch(variant, store=store, specs=specs, trace_path=str(scratch / "trace_d.jsonl"))
    items = []
    for g in games:
        side = "log_a" if g["a"] == who else "log_b"
        for ply, _dt, log in g[side]:
            if log.get("legal", 2) >= 2:
                items.append((list(g["moves"][:ply]), g["moves"][ply]))
    n_diff = n_diff_noise = 0
    n_diff_ordered = n_ordered = 0
    for i in range(0, len(items), chunk):
        part = items[i:i + chunk]
        pos = [(variant, None, m) for m, _x in part]
        a = read_batch(search, pos, seconds, threads, None, None)
        b = read_batch(search, pos, seconds, threads, None, None)
        for (m, played), x, y in zip(part, a, b):
            n_diff += played != x[0]
            n_diff_noise += x[0] != y[0]
    n = max(1, len(items))
    return [f"OBS subgoal_order differs: {who} in {len(games)} games, {len(items)} decisions with two or more legal moves; its move differed from the control's default call on the same position (0.2 s, outside the game clock) in {n_diff} ({100 * n_diff / n:.1f} pct); "
            f"two default calls of the control on the same positions differed from each other in {n_diff_noise} ({100 * n_diff_noise / n:.1f} pct)"]


# --------------------------------------------------------------------------------------------------------------------------- the run

def emit(lines: list, out_file: "Path | None") -> None:
    PM.emit(lines, out_file)


def main(argv: "list | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stage", required=True, choices=("b", "match", "arrival", "differs", "report"))
    ap.add_argument("--variant", default=None)
    ap.add_argument("--name", default=None, choices=tuple(MATCHES))
    ap.add_argument("--pairs", type=int, default=300)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 3))
    ap.add_argument("--seconds", type=float, default=PM.CLOCK)
    ap.add_argument("--games-dir", default=None, help="stage b: the kept games (an absolute path)")
    ap.add_argument("--weights", default=None, help="the 78.50 weights file (default: under artifacts/experiments)")
    ap.add_argument("--tag", default="full")
    ap.add_argument("--sample", type=int, default=30, help="arrival and differs: opening pairs (two games each) of the sample")
    ap.add_argument("--first", type=int, default=None)
    ap.add_argument("--n", type=int, default=B_SAMPLE, help="stage b: positions")
    a = ap.parse_args(argv)
    root = out_root() / a.tag
    threads = max(1, (os.cpu_count() or 2) - 3)
    weights = a.weights or str(Path(paths.experiment(DEFAULT_WEIGHTS)))
    obs = root / "obs.txt"
    if a.stage == "b":
        records = FO.read_games(a.games_dir)
        w = PM.load_weights(weights)["weights"]
        lines, _data = stage_b(records[0]["variant"], records, w, a.seconds, threads, root / "scratch_b", a.n)
        emit(lines, obs)
    elif a.stage == "match":
        t0 = time.perf_counter()
        games = PM.run_match(a.name, a.variant, a.pairs, a.workers, a.seconds, weights, root / f"match_{a.name}", a.first)
        lines = PM.match_lines(a.name, MATCHES[a.name][0], MATCHES[a.name][1], games)
        lines.append(f"OBS subgoal_order match {a.name}: workers {a.workers} (cores {os.cpu_count()}), clock {a.seconds} s a move for both sides, wall {time.perf_counter() - t0:.0f} s in this run")
        emit(lines, obs)
    elif a.stage in ("arrival", "differs"):
        games = PM.read_games_file(root / f"match_{a.name}" / "games.jsonl")
        seeds = sorted({g["seed"] for g in games})[:a.sample]
        sample = [g for g in games if g["seed"] in seeds]
        if a.stage == "arrival":
            per_game = PM.arrival_rows(a.variant, sample, threads, PM.FUSE)
            emit(PM.arrival_lines(a.name, MATCHES[a.name][0], MATCHES[a.name][1], per_game), obs)
        else:
            emit(stage_d_differs(a.variant, sample, MATCHES[a.name][0], a.seconds, threads, root / "scratch_d"), obs)
    else:
        lines = []
        for name, (pa, pb, _s) in MATCHES.items():
            f = root / f"match_{name}" / "games.jsonl"
            if f.exists():
                lines += PM.match_lines(name, pa, pb, PM.read_games_file(f))
        emit(lines, None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
