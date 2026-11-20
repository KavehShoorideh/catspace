"""PLAN 79.13 PART A, THE EVENT: WAS A LOGGED GOAL REACHED IN PLAY (SYSTEM_SPEC 7 "Calculation and estimate, decided together" point 2, the estimate per goal; 6 the event heads: an event of the rules and the time to it; 9 sleep; piece: the estimate per goal).
A MEASURE MODULE OVER THE LOGS (sleep). Nothing of it runs in the wake, no engine change, no oracle: the logged games of 79.12 hold, at every decision of the wake, every goal of both sides and the game that followed; whether a goal was then reached is an event of the rules, counted from the moves.
Game-free: a goal names its target in the game's words ("<role> <kind> on <place>"), which this module only matches against the tracker's own list of the position (the same words); a kind is a string it never reads.

THE EVENT. For a goal logged at the decision after ply t of a game that has the moves m_1..m_n:
  REACHED within K plies when, within the next K plies (t+1..t+K),
    (target not the goal kind) the target unit, followed from ply t by the tracker (sheet/fates.rs), was TAKEN, and the side of the goal was not BEHIND by the count of worths over the same plies: the balance of worths (own units less the other side's, by the record of
    worths given, a kind it does not hold is worth nought) at ply min(t+K, n) is not under the balance at ply t;
    (target the goal kind) the game ENDED for that side within the K plies (it won, the game being decided; a game that reached the ply cap is not decided).
  A window that runs past the end of a game that reached the ply cap is CENSORED for that K (the game was cut, not decided): -1, never counted. Past the end of a decided game it is fully known: not reached.
  Times: `KS` is the ladder of K read; 3 values are printed (`PRINT_KS`) and ONE is chosen by the plan's rule: the smallest K of the ladder at which nine in ten of the goals ever reached (reached within `EVER`, the longest K of the ladder) are reached; the rule is read on the FIT half of the games.

INTERFACE (written before the bodies).
  rank_of(worth, kind, goal_kind) -> int          the rank of a target's stake among the kinds: the number of distinct worths of the record strictly under the target's stake (the goal kind's stake is above every worth).
  game_rows(model, decs, res, worth, goal_kind) -> dict of columns     one row a goal logged in the game: game-free columns (below) and `reached`, an int8 row over `KS` (1 reached, 0 not, -1 censored).
  load_runs(tag, runs, workers) -> dict of arrays   every game of the runs (a Pool over a real script file, one thread a worker), the rows concatenated; strings kept as codes with their vocabularies.
  half_of(gid) -> bool                            True: the fit half (a fixed function of the game id; both seats of one game are one game).
  choose_k(reached, mask) -> (K, table)           the plan's rule on the rows of `mask`.
  Columns: game (int), ply, role (0: the side that moves first), mine (the goal is the mover's), source, unmet (sorted, distinct, joined), in_way, depth, rank, fix (a fix with a move exists), goal_unit, old_chance, old_n, old_src, reached[len(KS)].
  Called by: goal_table.py (part B), the tests (tests/test_plan79_13.py).

  .venv/bin/python -m catspace.incubator.analyst.discovery.goal_reach TAG [--runs a,b] [--workers W] [--limit N]
"""
from __future__ import annotations

from catspace.incubator.analyst.discovery import tiebreak_measure as TM          # first: it sets one thread a process before numpy is loaded

import argparse
import gzip
import json
import multiprocessing as mp
import zlib
from pathlib import Path

import numpy as np

from catspace.incubator.analyst.discovery import graph_layer1 as GL
from catspace.incubator.analyst.discovery import graph_layer2_measure as G2M
from catspace.incubator.analyst.discovery import human_heads_data as HD
from catspace.incubator.analyst.discovery import record_play as RP
from catspace.io import paths

OUT = ("whole_79_13",)
SRC = ("whole_79_12",)                                    # the logged games of 79.12 (training sets C, D and T; read only)
RUNS = ("train_self", "train_old")
KS = (1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 40, 60)
PRINT_KS = (4, 8, 16)
EVER = KS[-1]
EVER_SHARE = 0.9
_MODEL = None
_WORTH = None
_GOAL = None


def out_root(tag: str) -> Path:
    return Path(paths.experiment(*OUT, tag))


def half_of(gid: str) -> bool:
    """True: the fit half of the games (a fixed function of the game id)."""
    return zlib.crc32(gid.encode()) % 2 == 0


def rank_of(worth: dict, kind: str, goal_kind: str) -> int:
    """The number of distinct worths of the record strictly under the stake of a target of this kind; the goal kind's stake is above every worth."""
    vals = sorted(set(int(v) for v in worth.values()))
    stake = float("inf") if kind == goal_kind else float(worth.get(kind, 0))
    return sum(1 for v in vals if v < stake)


def kind_of(target: str, role: str) -> str:
    """The kind inside "<role> <kind> on <place>" (the words the engine gave)."""
    rest = target[len(role) + 1:] if target.startswith(role + " ") else target
    return rest.rsplit(" on ", 1)[0]


def game_rows(model, decs: list, res: dict, worth: dict, goal_kind: str) -> dict:
    """The rows of one game (see the module doc)."""
    moves = res["moves"]
    n = len(moves)
    unfinished = bool(res.get("unfinished"))
    outcome = res.get("outcome")
    horizons = list(range(1, EVER + 1))
    _kinds, per_ply = model.unit_fates((None, list(moves)), horizons)
    first_role = per_ply[0][0]
    # the balance of worths of each role at each ply
    roles = sorted({u[0] for u in per_ply[0][1]})
    mat = {r: np.zeros(n + 1) for r in roles}
    for t in range(n + 1):
        for (o, k, _p, _c) in per_ply[t][1]:
            mat[o][t] += worth.get(k, 0)
    out = {k: [] for k in ("ply", "role", "mine", "source", "unmet", "in_way", "depth", "rank", "fix", "goal_unit", "old_chance", "old_n", "old_src", "reached")}
    for d in decs:
        t = d["ply"]
        if t > n:
            continue
        mover, units = per_ply[t]
        index = {f"{o} {k} on {p}": i for i, (o, k, p, _c) in enumerate(units)}
        for gl in d["wake"]["log"]["goals"]:
            side, target = gl["side"], gl["target"]
            if side not in mat:
                continue
            other = next(r for r in roles if r != side)
            bal = mat[side] - mat[other]
            reached = []
            if gl["goal_unit"]:
                side_won = outcome is not None and ((outcome == 100) == (side == first_role)) and outcome != 50
                for K in KS:
                    if unfinished and t + K > n:
                        reached.append(-1)
                    else:
                        reached.append(1 if (side_won and n - t <= K) else 0)
            else:
                i = index.get(target)
                if i is None:
                    continue
                codes = units[i][3]
                tt = next((h for h in horizons if h <= n - t and 3 <= codes[h - 1] < 255), None)
                for K in KS:
                    h = min(K, n - t)
                    taken = tt is not None and tt <= h
                    if unfinished and t + K > n:
                        reached.append(-1)
                    else:
                        reached.append(1 if (taken and bal[min(t + K, n)] - bal[t] >= 0) else 0)
            ch = gl.get("chance") or {}
            out["ply"].append(t)
            out["role"].append(0 if side == first_role else 1)
            out["mine"].append(1 if side == mover else 0)
            out["source"].append(gl["source"])
            out["unmet"].append("+".join(sorted(set(gl["unmet"]))))
            out["in_way"].append(int(gl["in_the_way"]))
            out["depth"].append(int(gl["depth"]))
            out["rank"].append(rank_of(worth, kind_of(target, side), goal_kind))
            out["fix"].append(1 if any(f["moves"] for f in gl["fixes"]) else 0)
            out["goal_unit"].append(1 if gl["goal_unit"] else 0)
            out["old_chance"].append(float(ch.get("chance", -1.0)))
            out["old_n"].append(int(ch.get("n", 0)))
            out["old_src"].append(ch.get("source", ""))
            out["reached"].append(reached)
    return out


def read_game(p: Path) -> tuple:
    decs, res = [], None
    with gzip.open(p, "rt") as f:
        for line in f:
            r = json.loads(line)
            if r["kind"] == "decision":
                decs.append(r)
            else:
                res = r
    return decs, res


def _init() -> None:
    global _MODEL, _WORTH, _GOAL
    _MODEL = GL.model(HD.GAME, HD.VARIANT)
    _WORTH = dict(RP.worth_of("human"))
    _GOAL = G2M.goal_kind_of(HD.VARIANT)


def _one(job: tuple) -> "dict | None":
    run, path = job
    decs, res = read_game(Path(path))
    if res is None or not decs:
        return None
    rows = game_rows(_MODEL, decs, res, _WORTH, _GOAL)
    rows["gid"] = res["gid"]
    rows["run"] = run
    return rows


STR_COLS = ("source", "unmet", "old_src")
INT_COLS = ("ply", "role", "mine", "in_way", "depth", "rank", "fix", "goal_unit", "old_n")


def load_runs(tag_src: str, runs: tuple, workers: int, limit: int = 0) -> dict:
    jobs = []
    for run in runs:
        files = sorted((Path(paths.experiment(*SRC, tag_src)) / "logs" / run).glob("*.jsonl.gz"))
        jobs += [(run, str(f)) for f in (files[:limit] if limit else files)]
    with mp.get_context("spawn").Pool(workers, initializer=_init) as pool:
        games = [g for g in pool.imap_unordered(_one, jobs, chunksize=4) if g is not None]
    games.sort(key=lambda g: (g["run"], g["gid"]))
    cols: dict = {k: [] for k in INT_COLS + STR_COLS + ("old_chance", "reached", "game")}
    gids, run_of = [], []
    for gi, g in enumerate(games):
        m = len(g["ply"])
        gids.append(g["gid"])
        run_of.append(g["run"])
        cols["game"] += [gi] * m
        for k in INT_COLS + STR_COLS + ("old_chance", "reached"):
            cols[k] += g[k]
    out: dict = {"game": np.array(cols["game"], np.int32), "reached": np.array(cols["reached"], np.int8).reshape(-1, len(KS)), "old_chance": np.array(cols["old_chance"], np.float32),
                 "gids": np.array(gids), "run_of": np.array(run_of)}
    for k in INT_COLS:
        out[k] = np.array(cols[k], np.int32)
    for k in STR_COLS:
        vocab, codes = np.unique(np.array(cols[k]), return_inverse=True)
        out[k] = codes.astype(np.int32)
        out[k + "_vocab"] = vocab
    return out


def fit_mask(d: dict) -> np.ndarray:
    side = np.array([half_of(str(g)) for g in d["gids"]])
    return side[d["game"]]


def choose_k(reached: np.ndarray, mask: np.ndarray) -> tuple:
    """THE PLAN'S RULE: the smallest K of the ladder at which at least nine in ten of the goals ever reached (reached within EVER) are reached, on the rows of `mask`; the rows censored at a K are left out of that K.
    Returns (K, [(K, goals counted, reached, share of the reached at EVER)])."""
    r = reached[mask]
    ever = r[:, -1] == 1
    table = []
    pick = None
    for j, K in enumerate(KS):
        ok = r[:, j] >= 0
        hit = int(((r[:, j] == 1) & ok).sum())
        tot_ever = int((ever & ok).sum())
        share = hit / max(1, tot_ever)
        table.append((K, int(ok.sum()), hit, share))
        if pick is None and share >= EVER_SHARE:
            pick = K
    return (pick if pick is not None else EVER), table


def lines_of(d: dict, k_star: int, table: list) -> list:
    n = len(d["game"])
    reached = d["reached"]
    mask = fit_mask(d)
    t = "OBS whole_79_13 A the event"
    j_star = KS.index(k_star)
    out = [f"{t}: games {len(d['gids'])} ({int(mask[np.unique(d['game'], return_index=True)[1]].sum())} rows' games in the fit half by the id), goals {n} at decisions of the wake; goals of the mover {int(d['mine'].sum())}, of the other side {n - int(d['mine'].sum())}; goal-kind goals {int(d['goal_unit'].sum())}"]
    for K in PRINT_KS + (k_star,):
        j = KS.index(K)
        ok = reached[:, j] >= 0
        hit = (reached[:, j] == 1) & ok
        out.append(f"{t}: K {K}: goals counted {int(ok.sum())} (censored {int((~ok).sum())}), reached {int(hit.sum())}, share {hit.sum() / max(1, ok.sum()):.4f}; "
                   f"mover's goals {hit[ok & (d['mine'] == 1)].sum() / max(1, (ok & (d['mine'] == 1)).sum()):.4f}, other side's {hit[ok & (d['mine'] == 0)].sum() / max(1, (ok & (d['mine'] == 0)).sum()):.4f}; "
                   f"side that moves first {hit[ok & (d['role'] == 0)].sum() / max(1, (ok & (d['role'] == 0)).sum()):.4f}, the other {hit[ok & (d['role'] == 1)].sum() / max(1, (ok & (d['role'] == 1)).sum()):.4f}; goal-kind goals {hit[ok & (d['goal_unit'] == 1)].sum() / max(1, (ok & (d['goal_unit'] == 1)).sum()):.4f}")
    out.append(f"{t}: THE RULE (fit half): reached within {EVER} plies is the ever; share of those reached by K: " + ", ".join(f"K {K} {s:.3f}" for K, _n, _h, s in table))
    out.append(f"{t}: K CHOSEN {k_star} (the smallest K at which {EVER_SHARE:.1f} of the goals ever reached are reached)")
    return out


def main(argv: "list | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("tag")
    ap.add_argument("--src", default="full", help="the tag of 79.12's logs")
    ap.add_argument("--runs", default=",".join(RUNS))
    ap.add_argument("--workers", type=int, default=9)
    ap.add_argument("--limit", type=int, default=0, help="games a run (0: all)")
    a = ap.parse_args(argv)
    root = out_root(a.tag)
    root.mkdir(parents=True, exist_ok=True)
    d = load_runs(a.src, tuple(a.runs.split(",")), a.workers, a.limit)
    k_star, table = choose_k(d["reached"], fit_mask(d))
    lines = lines_of(d, k_star, table)
    np.savez_compressed(root / "goal_rows.npz", **d, k_star=np.array(k_star), ks=np.array(KS))
    (root / "event_obs.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
