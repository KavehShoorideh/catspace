"""PLAN 78.55 G: THE MEASURE OF FRAMES, on the quiet positions of the test half of the 500 games. A measurement in a sandbox; nothing is wired into any player, nothing is written to rules/learned or
data/derived/snapshots; no oracle, no human game, no puzzle. Game-free: no unit, place or kind of any game is named here; a kind is a word read from the graph at run time. No worth table.

STATED BEFORE THE RUN (and in docs/ANALYST_PLAN.md): WHICH CONCEPTS SEED THE FRAMES AND HOW MANY. The kept concepts of 78.54 (artifacts/experiments/unit_fates_78_54/<games>/concepts.tsv), those with no `same_units_as`
(the distinct ones: a concept that binds the same units as another is that other), ordered by |z| of the selected cell on HALF A (the fit half, ids mod 4 = 0: the half 78.54 selected on, so the test half is not
used to choose seeds), the top SEEDS = 12. A frame is made from each match of such a concept on a quiet position of the test half T (the units and empty places the match binds); frames in contact are joined.
The rows are the quiet positions of T (78.49's, `forcing_arrival_measure.quiet_indices`).

WHAT IS COUNTED (per frame member, at a horizon of 4 and of 8 plies): what became of the unit in the game that was in fact played (the stage 0 tracker: taken, taken for nothing = taken having taken
nothing, taken after taking = an exchange), by the CLASS the frame gave the unit at the position:
  LOST         the frame says the other side can force the unit taken, none of its own lost (a forced claim, the sound form: the reading side spends no pass, the other may pass at every ply or leave)
  SAFE         the frame says its own side can force it not taken within the frame's horizon
  THREAT       the claim holds only if the other side passes once (or its side has a spare move); CONDITIONAL: only after one arrival of a unit of the side to move on or onto the frame
  CONTESTED    no claim holds and every line was read; UNREAD: no claim holds and the fuse ended some line (never counted as a verdict); BOTH: both LOST and SAFE (counted, read)
beside the BASE: all units on the same rows by their kind and whether their owner is the side to move (the base of stage 0), and the same base re-weighted to each class's own mix of kinds and owners.

THE FUSES, named (they are costs, not tuned to a result): FUSE = 5000 positions per reading (a short run showed frames cost milliseconds and a 20000 fuse spent minutes on a few big frames); ROW_FUSE = 100000 positions over all frames of one row; the depth of a line is one ply per
member plus one (counted as BOUNDED); a frame of more than 12 members is refused and counted. The composition test (F) is run on every COMPOSE_EVERY-th row that has two frames apart, for a pair of at most 9 members together, with a fuse of 5000 (both in frames_py.rs).

  .venv/bin/python -m catspace.incubator.analyst.discovery.frames_measure --game NAME --games-dir D --reads R [--concepts-dir C] [--games N] [--threads K] [--out-dir O]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

import numpy as np

from catspace.incubator.analyst.discovery import forcing_arrival_measure as FA
from catspace.incubator.analyst.discovery import forcing_opportunity_measure as FO
from catspace.incubator.analyst.discovery import graph_layer1 as GL
from catspace.incubator.analyst.discovery import graph_layer2_measure as G2M
from catspace.incubator.analyst.discovery import subgoal_fit_measure as SF
from catspace.io import paths

HORIZONS = (4, 8)
SEEDS = 12
MAX_SEED_SIZE = 3
FUSE = 5_000
ROW_FUSE = 100_000
COMPOSE_EVERY = 5
PROFILE_GAMES = 50
CLASSES = ["LOST", "SAFE", "BOTH", "THREAT", "CONDITIONAL", "CONTESTED", "UNREAD"]


def pick_concepts(concepts_dir: Path, n: int) -> list:
    """The top `n` distinct kept concepts of 78.54 by |z| on half A: [(dc, key, definition, z_A)]."""
    keys = json.loads((concepts_dir / "concept_keys.json").read_text())
    rows = []
    with open(concepts_dir / "concepts.tsv") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r["same_units_as"]:
                continue
            rows.append((abs(float(r["z_A"])), r["dc"], r["definition"], float(r["z_A"])))
    rows.sort(key=lambda x: -x[0])
    return [(dc, int(keys[dc]["key"]), definition, z) for _a, dc, definition, z in rows[:n]]


def rate(v: list) -> float:
    return v[1] / v[0] if v[0] else float("nan")


def main(argv: "list | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--game", required=True)
    ap.add_argument("--games-dir", required=True)
    ap.add_argument("--reads", required=True)
    ap.add_argument("--concepts-dir", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--games", type=int, default=0, help="a subset: the first N games by id")
    ap.add_argument("--seeds", type=int, default=SEEDS)
    ap.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args(argv)
    t0 = time.perf_counter()
    records = sorted(FO.read_games(a.games_dir), key=lambda r: r["game"])
    variant = records[0]["variant"]
    label = Path(a.games_dir).name + (f"_first{a.games}" if a.games else "")
    out = Path(a.out_dir or paths.experiment("frames_78_55", label))
    out.mkdir(parents=True, exist_ok=True)
    concepts_dir = Path(a.concepts_dir or paths.experiment("unit_fates_78_54", Path(a.games_dir).name))
    reads_by_id = {}
    with open(a.reads) as f:
        for line in f:
            g = json.loads(line)
            reads_by_id[g["game"]] = g["plies"]
    if a.games:
        records = records[:a.games]
    lines: list = []

    def say(msg: str) -> None:
        lines.append(f"OBS frames {label}: {msg}")
        print(lines[-1], flush=True)

    ids, gix, plies = [], [], []
    for g, r in enumerate(records):
        pl = reads_by_id[r["game"]]
        own, oth = FA.counts_of_game(pl)
        for t in FA.quiet_indices(pl, own, oth):
            ids.append(r["game"]); gix.append(g); plies.append(t)
    t = {"id": np.array(ids), "game": np.array(gix), "ply": np.array(plies)}
    _fit, test = SF.split_rows(t)
    say(f"games {len(records)}, quiet rows {len(ids)}, test half T (odd ids) {len(test)} rows")
    picked = pick_concepts(concepts_dir, a.seeds)
    say(f"SEEDS stated before the run: the top {a.seeds} distinct kept concepts of 78.54 by |z| on half A; found {len(picked)}")
    for dc, key, definition, z in picked:
        say(f"seed {dc} (z_A {z:+.2f}): {definition}")
    model = GL.model(a.game, variant)
    kinds = [n[len("kind:"):] for ty, n in model.static_part()["nodes"] if ty == "kind"]
    wires = [(r.get("start"), list(r["moves"])) for r in records]
    model.learn_profile(wires[:PROFILE_GAMES])
    goal = G2M.goal_kind_of(variant)
    gi, rows = G2M.layout(t, test)
    t1 = time.perf_counter()
    res = model.frames_measure([wires[g] for g in gi], rows, [k for _d, k, _x, _z in picked], goal, list(HORIZONS), FUSE, ROW_FUSE, COMPOSE_EVERY, MAX_SEED_SIZE, True, a.threads)
    wall = time.perf_counter() - t1
    c, s, obs = res["count"], res["secs"], res["obs"]
    say(f"run: {wall:.1f} s wall over {a.threads} threads")

    # ---- (1) cost
    n = lambda k: c.get(k, 0)
    rows_all, rows_seed = n("rows"), n("rows_with_seeds")
    say(f"1 cost: rows {rows_all}, rows with a concept match {rows_seed}, matches {n('seeds')}, frames made {n('frames_before_join')} and {n('frames')} after joining those in contact, frames read {n('frames_read')}, "
        f"frames skipped by the row fuse {n('frames_skipped_by_row_fuse')}")
    refused = {k.split('|', 1)[1]: v for k, v in c.items() if k.startswith("refused|")}
    say(f"1 cost: frames refused {refused or 0}; members per frame read {n('members') / max(1, n('frames_read')):.2f}")
    fr = max(1, n("frames_read"))
    say(f"1 cost: positions read per frame {n('nodes') / fr:.1f} (all readings of the frame: the forced form of both sides, the flipped root, the waiting root, each conditional arrival)")
    say(f"1 cost: frames with some line UNREAD (fuse) {n('frames_with_unread_lines')} of {n('frames_read')} ({n('frames_with_unread_lines') / fr:.4f}); lines unread {n('unread_lines')}; lines ended by the depth bound {n('bounded')}; "
        f"stuck lines {n('stuck')}; lines ended by a leave {n('left')}; forced passes {n('forced_pass')}; passes not realisable {n('pass_unrealisable')}; passes the game refused {n('pass_refused')}; units read as OTHER {n('others')}")
    sec = s.get("seconds_seeds", 0) + s.get("seconds_frames_build", 0) + s.get("seconds_read", 0)
    say(f"1 cost: seconds summed over threads: matching {s.get('seconds_seeds', 0):.1f}, making frames {s.get('seconds_frames_build', 0):.1f}, reading frames {s.get('seconds_read', 0):.1f}; "
        f"per row with a match {1e6 * sec / max(1, rows_seed):.0f} microseconds; per frame read {1e6 * s.get('seconds_read', 0) / fr:.0f} microseconds; per row (all rows) {1e6 * sec / max(1, rows_all):.0f} microseconds")
    say(f"1 cost: arrivals tried (conditional claims) {n('arrivals_tried')}")

    # ---- (2) do the reasoned fates foretell the observed ones
    def total(cls: str, h: int) -> list:
        tot = [0, 0, 0, 0]
        for k, v in obs.items():
            p = k.split("|")
            if p[0] == cls and int(p[-1]) == h:
                tot = [x + y for x, y in zip(tot, v)]
        return tot

    def expected(cls: str, h: int) -> "tuple[float, float]":
        """The taken rate and the taken-for-nothing rate the BASE gives the class's own mix of (kind, owner is the side to move)."""
        n_tot, e_taken, e_free = 0, 0.0, 0.0
        for k, v in obs.items():
            p = k.split("|")
            if p[0] != cls or int(p[-1]) != h:
                continue
            b = obs.get(f"ALL|{p[1]}|{p[2]}|{h}")
            if not b or not b[0]:
                continue
            n_tot += v[0]; e_taken += v[0] * b[1] / b[0]; e_free += v[0] * b[2] / b[0]
        return (e_taken / n_tot, e_free / n_tot) if n_tot else (float("nan"), float("nan"))

    table = []
    for h in HORIZONS:
        base = total("ALL", h)
        say(f"2 base h={h}: units {base[0]}, taken {rate(base):.4f}, taken for nothing {base[2] / max(1, base[0]):.4f}, taken after taking {base[3] / max(1, base[0]):.4f}")
        for cls in ["MEMBER"] + CLASSES:
            v = total(cls, h)
            if not v[0]:
                continue
            et, ef = expected(cls, h)
            say(f"2 class {cls:<11} h={h}: units {v[0]:>7}, taken {rate(v):.4f} (the base for the same kinds and owners {et:.4f}), taken for nothing {v[2] / v[0]:.4f} (base {ef:.4f}), taken after taking {v[3] / v[0]:.4f}")
            table.append((cls, h, v, et, ef))
    # ---- (3) conditional claims
    for h in HORIZONS:
        for when in ("now", "next", "no"):
            v = obs.get(f"CONDCLAIM|{when}|{h}", [0, 0, 0, 0])
            if v[0]:
                say(f"3 conditional claims h={h}, the named change played {when}: claims {v[0]}, the unit taken {v[1] / v[0]:.4f}, taken for nothing {v[2] / v[0]:.4f}, taken after taking {v[3] / v[0]:.4f}")
    # ---- (4) composition
    f = {k[2:]: v for k, v in c.items() if k.startswith("F|")}
    say(f"4 composition against the joint read: rows with two frames apart {n('rows_with_two_frames')}; " + "; ".join(f"{k} {v}" for k, v in sorted(f.items())))
    (out / "disagreements.txt").write_text("\n".join(res["dump"]) + "\n")
    say(f"4 composition: {len(res['dump'])} disagreeing cases kept in {out / 'disagreements.txt'}")
    with open(out / "class_table.tsv", "w") as fo:
        fo.write("class\thorizon\tunits\ttaken\ttaken_for_nothing\ttaken_after_taking\tbase_taken\tbase_for_nothing\n")
        for cls, h, v, et, ef in table:
            fo.write(f"{cls}\t{h}\t{v[0]}\t{v[1]}\t{v[2]}\t{v[3]}\t{et:.5f}\t{ef:.5f}\n")
    (out / "raw_counts.json").write_text(json.dumps({"count": c, "secs": s, "obs": obs}, sort_keys=True))
    say(f"done in {time.perf_counter() - t0:.1f} s; kinds {kinds}")
    (out / "obs.txt").write_text("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
