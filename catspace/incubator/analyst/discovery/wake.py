"""PLAN 78.14 WAKE: the engine plays games and everything it did is logged (the architecture document's wake phase). Self-play in lockstep: every ply the games still
running are decided in ONE `solve_batch` call, so the cores are shared by the games and each decision is a row of the trace log (its steps, goals, evidence, measured
cost in parts, the snapshot it read). The seeded random opening plies are not decisions and are not logged as such. Games end by the game's own outcome, by a
position met three times (the rules' draw, as the record), or by the ply cap (unfinished).

  .venv/bin/python -m catspace.incubator.analyst.discovery.wake chess --games 8 --seconds 0.2 [--store <snapshot.json>] [--log <path>]

The log (JSON Lines, append-only) holds two kinds of row: `position` rows (solve_batch's, labelled `run:game:ply:role`) and `game` rows (the game's moves and outcome).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time

import cs_engine as CE

from catspace.incubator.analyst.discovery import game_store as GS
from catspace.io import paths

ROLES = {"white": 0, "black": 1}        # the chess family's seats (the kernels' roles)
DRAW = [50, 50]


def play_self(variant: str, games: int, *, seconds: float = 0.2, random_plies: int = 4, max_plies: int = 120, seed: int = 1, store: "str | None" = None,
              specs: "dict | None" = None, threads: int = 0, log: "str | None" = None, run: str = "r0", prior: bool = False, ties: bool = False, reason: bool = False, wdl: bool = False, wdl_table: "str | None" = None,
              starts: "list | None" = None) -> list:
    """Play `games` self-play games of `variant`; -> [{run, game, variant, start, moves, outcome, unfinished, snapshot}]. Positions are decided `seconds` each. `starts`
    (plan 78.22): position strings the kernel accepts (a FEN for the chess family); game g starts from `starts[g % len(starts)]`, None / empty = the game's own start. The
    random opening plies are played after the start."""
    starts = [x for x in (starts or [])] or [None]
    rng = random.Random(seed)
    state = [{"run": run, "game": g, "variant": variant, "moves": [], "outcome": None, "unfinished": False, "seen": {}, "over": False,
              "start": starts[g % len(starts)]} for g in range(games)]
    snapshot = ""
    ply = 0
    while True:
        active = [s for s in state if not s["over"]]
        if not active:
            break
        engine, labels, positions = [], [], []
        for s in active:
            fen, mover, legal, outcome, key = CE.state(variant, s["start"], s["moves"])
            if outcome is not None:
                s["over"], s["outcome"] = True, list(outcome)
                continue
            if not legal:
                s["over"], s["unfinished"] = True, True
                continue
            s["seen"][key] = s["seen"].get(key, 0) + 1
            if s["seen"][key] >= 3:
                s["over"], s["outcome"] = True, list(DRAW)
                continue
            if len(s["moves"]) >= max_plies:
                s["over"], s["unfinished"] = True, True
                continue
            if len(s["moves"]) < random_plies:
                s["moves"].append(sorted(legal)[rng.randrange(len(legal))])
                continue
            engine.append(s)
            positions.append((variant, s["start"], list(s["moves"])))
            labels.append(f"{run}:{s['game']}:{len(s['moves'])}:{ROLES[mover]}")
        if positions:
            rows = CE.solve_batch(positions, seconds, snapshot=run, trace_path=log, threads=threads, prior=prior, ties=ties, reason=reason, wdl=wdl, wdl_table=wdl_table, store=store, specs=specs or {}, labels=labels)
            for s, (mv, _proved, _nodes, _secs, err) in zip(engine, rows):
                if mv is None:
                    s["over"], s["unfinished"] = True, True
                else:
                    s["moves"].append(mv)
        ply += 1
    snap = ""
    if store:
        snap = CE.snapshot_info(store)["id"]
    out = [{"run": s["run"], "game": s["game"], "variant": variant, "start": s["start"], "moves": s["moves"], "outcome": s["outcome"], "unfinished": s["unfinished"], "snapshot": snap or snapshot} for s in state]
    if log:
        with open(log, "a") as f:
            for g in out:
                f.write(json.dumps({"kind": "game", **g}) + "\n")
    return out


def read_starts(path: str) -> list:
    """The positions of a starts file: one per line, blank lines and `#` comments ignored."""
    out = []
    for line in open(path):
        line = line.strip()
        if line and not line.startswith("#"):
            out.append(line)
    return out


def main(argv: list) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("game")
    ap.add_argument("--variant", default=None)
    ap.add_argument("--games", type=int, default=8)
    ap.add_argument("--seconds", type=float, default=0.2)
    ap.add_argument("--random-plies", type=int, default=4)
    ap.add_argument("--max-plies", type=int, default=120)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--store", default=None, help="a snapshot file (default: the game's HEAD, if any; 'none' for no store)")
    ap.add_argument("--log", default=None)
    ap.add_argument("--starts", default=None, help="a file of start positions, one per line (blank lines and # comments ignored); game g starts from line g mod n")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--prior", action="store_true", help="read the ledger's TIES (the controller reads them only when the prior is on; it then also runs the density prior over them) so the log's `candidates` counts them")
    ap.add_argument("--reason", action="store_true", help="plan 78.25: run the reasoning chain (threats by stake, the frame, parry or counter) before the ledger: its candidates are read first, its lines join the trace; the ledger's value still decides")
    ap.add_argument("--wdl", action="store_true", help="plan 78.26: estimate win / draw / loss first (playouts, pooled with --wdl-table); sharp or unknown keeps reasoning, quiet lets the probabilities pick among the moves the ledger holds equal; the mode line joins the trace")
    ap.add_argument("--wdl-table", default=None, help="plan 78.26: a results table file (wdl_table.py) the estimate pools with its playouts")
    ap.add_argument("--ties", action="store_true", help="read the ledger's ties without the prior (the ledger keeps its share of the clock): the tie rate becomes measurable")
    a = ap.parse_args(argv)
    variant = a.variant or a.game
    tag = a.tag or time.strftime("%Y%m%d-%H%M%S")
    log = a.log or paths.experiment(f"wake_{a.game}_{tag}.jsonl")
    store = a.store
    if store is None and GS.head(a.game):
        store = str(GS.path_of(a.game, GS.head(a.game)))
    if store == "none":
        store = None
    specs = GS.spec_texts(a.game) if store else None
    t0 = time.perf_counter()
    games = play_self(variant, a.games, seconds=a.seconds, random_plies=a.random_plies, max_plies=a.max_plies, seed=a.seed, store=store, specs=specs, threads=a.threads, log=log, run=tag, prior=a.prior, ties=a.ties, reason=a.reason, wdl=a.wdl, wdl_table=a.wdl_table,
                       starts=read_starts(a.starts) if a.starts else None)
    plies = sum(len(g["moves"]) for g in games)
    print(f"OBS wake {a.game}: {len(games)} games, {plies} plies, {sum(1 for g in games if g['unfinished'])} unfinished, snapshot {games[0]['snapshot'] if store else 'none'}, "
          f"{time.perf_counter() - t0:.1f} s -> {log}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
