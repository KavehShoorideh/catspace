"""PLAN 78.62 A: THE TACTICAL STATE OF A POSITION, PER SIDE, ALONG SAMPLED HUMAN GAMES. A measurement in a sandbox; nothing is wired into any player, nothing is written to rules/learned or data/derived. Game-free: a
kind, a role or a place is a word read from the engine at run time; no unit, place or kind is named here. No engine evaluation, no labeller, no judge: the frame read of 78.60 (the rules' own reading with the order of
kinds of a FITTED head record as its exchange estimate), the rules' forcing read where it was kept, and what happened on the board.

THE STATE of a row (a position of a game): the member units of every frame read at it (`BaseGraph.questions_state`, Rust, all cores), each with its owner (the side to move or the other), its kind, its CLASS (lost
for its owner, safe, both, unread, threat, conditional, contested), what the class RESTS ON (forced within the frame, pruned by inclusion, by the estimate, bound), its TRAP word, and what became of it within HORIZON
plies (not taken, taken for nothing, taken after taking). The seeds are the 78.55 seeds (the top SEEDS distinct kept concepts of 78.54), the fuses and the depth those of 78.60 (named there); a unit that no frame
holds has no record (it is not read, which is not the same as safe). THE ORDER OF KINDS the estimate reads is the human-game record of 78.59 (`heads.kind_baseline_human` of estimator_heads_human.json: terms and the
band of each, as the record's own; the goal kind's worth is a rule, more than all the others together, as in 78.58). Nothing else is a worth.

THE SAMPLE (stated before the run, see docs/ANALYST_PLAN.md "78.62 SAMPLE"): the arrival games of 78.59 (the first PER_SHARD of each shard's hash order, the games whose forcing reads of 78.59 are kept), every ply
from LO on to the end of the game, in resumable chunks of CHUNK games.

  .venv/bin/python -m catspace.incubator.analyst.discovery.tactical_state --out-dir O [--per-shard N] [--lo P] [--hi P] [--threads K]
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from catspace.incubator.analyst.discovery import frames_measure as OLD
from catspace.incubator.analyst.discovery import graph_layer1 as GL
from catspace.incubator.analyst.discovery import graph_layer2_measure as G2M
from catspace.incubator.analyst.discovery import human_heads_data as HD
from catspace.io import paths

LO = 10
HORIZON = 8
SEEDS = 12
FUSE = 1_500
ROW_FUSE = 30_000
TRAP_FUSE = 400
CHUNK = 20
PER_SHARD = 100
RULES = (True, True, True, True)                                       # settle, progress, cause, band: the 78.60 read
CLASS_WORDS = ["lost", "safe", "both", "unread", "threat", "conditional", "contested"]
BASIS_WORDS = ["forced", "pruned", "estimate", "bound"]
TRAP_WORDS = ["free", "trapped", "immobile", "not_quiet", "ended", "unread"]
FATE_WORDS = {0: "not taken", 1: "taken for nothing", 2: "taken after taking", 255: "unknown"}
HEADS_DIR = ("estimator_78_59", "full")


def human_kind_record() -> dict:
    """The order of kinds of the human-game record of 78.59 as the estimate's input: {kind: worth (whole numbers)}, {kind: (low, high)}, the goal kind's worth a rule."""
    rec = json.loads(Path(paths.experiment(*HEADS_DIR, "estimator_heads_human.json")).read_text())
    h = rec["heads"]["kind_baseline_human"]
    scale = 1000
    vals = {o["kind"]: int(round(scale * o["term"])) for o in h["order_of_kinds"]}
    bands = {o["kind"]: (int(round(scale * o["lo"])), int(round(scale * o["hi"]))) for o in h["order_of_kinds"]}
    goal = G2M.goal_kind_of(HD.VARIANT)
    vals[goal] = int(sum(max(0, v) for v in vals.values()) + 1)                       # a RULE, not a fit
    return {"values": vals, "bands": bands, "goal": goal, "source": "estimator_78_59/full/estimator_heads_human.json heads.kind_baseline_human (human games)"}


def load_sample(data_dir: Path, per_shard: int, only: int = 0) -> list:
    allg = [json.loads(x) for x in (data_dir / "games.jsonl").read_text().splitlines() if x.strip()]
    games = [g for g in allg if g["arrival"]]
    seen: dict = {}
    keep = []
    for g in games:
        sh = g["id"].split(":")[0]
        seen[sh] = seen.get(sh, 0) + 1
        if seen[sh] <= per_shard:
            keep.append(g)
    return keep[:only] if only else keep


def make_model(games: list, rec: dict, root_cap: int):
    model = GL.model(HD.GAME, HD.VARIANT)
    model.learn_profile([(None, g["moves"]) for g in games[:OLD.PROFILE_GAMES]])
    model.set_kind_values([(k, int(v)) for k, v in rec["values"].items()])
    model.set_kind_bands([(k, lo, hi) for k, (lo, hi) in rec["bands"].items()])
    model.set_rules(*RULES, int(root_cap))
    return model


def read_chunk(model, games: list, lo: int, hi: int, keys: list, goal: str, trap_fuse: int, threads: int, base: int = 0) -> dict:
    """The rows of the plies lo..hi (hi 0: to the end) of each game of the chunk, as flat arrays."""
    rows = [list(range(lo, (min(hi, len(g["moves"]) + 1) if hi else len(g["moves"]) + 1))) for g in games]
    res = model.questions_state([(None, g["moves"]) for g in games], rows, keys, goal, FUSE, ROW_FUSE, OLD.MAX_SEED_SIZE, True, trap_fuse, HORIZON, threads)
    rs = res["rows"]
    uoff, boff, units, board = [0], [0], [], []
    for r in rs:
        units += r["units"]
        board += r["board"]
        uoff.append(len(units))
        boff.append(len(board))
    return {"game": np.array([r["game"] + base for r in rs], np.int32), "ply": np.array([r["ply"] for r in rs], np.int16), "mover": np.array([r["mover"] for r in rs], np.int8),
            "questions": np.array([r["questions"] for r in rs], np.int64), "unread": np.array([r["unread"] for r in rs], np.int32), "frames": np.array([r["frames"] for r in rs], np.int16),
            "uoff": np.array(uoff, np.int64), "units": np.array(units, np.uint8).reshape(-1, 6), "boff": np.array(boff, np.int64), "board": np.array(board, np.uint8).reshape(-1, 3),
            "meta": {"kinds": res["kinds"], "roles": res["roles"], "places": res["places"]}}


def merge(parts: list) -> dict:
    out: dict = {}
    for k in ("game", "ply", "mover", "questions", "unread", "frames", "units", "board"):
        out[k] = np.concatenate([p[k] for p in parts])
    for key in ("uoff", "boff"):
        base, offs = 0, [np.zeros(1, np.int64)]
        for p in parts:
            offs.append(p[key][1:] + base)
            base += int(p[key][-1])
        out[key] = np.concatenate(offs)
    out["meta"] = parts[0]["meta"]
    return out


def save(out: dict, p: Path) -> None:
    np.savez_compressed(p, **{k: v for k, v in out.items() if k != "meta"})
    p.with_suffix(".json").write_text(json.dumps(out["meta"]))


def load(p: Path) -> dict:
    z = np.load(p)
    out = {k: z[k] for k in z.files}
    out["meta"] = json.loads(p.with_suffix(".json").read_text())
    return out


def coverage_lines(games: list, st: dict) -> list:
    """THE STANDING ENDGAME RULE: the table of the sample, by units left (share of the read positions and of the games), per small set of units by kind, and how the games ended."""
    meta = st["meta"]
    kinds = meta["kinds"]
    goal = G2M.goal_kind_of(HD.VARIANT)
    gk = kinds.index(goal)
    nrow = len(st["ply"])
    men = np.zeros(nrow, np.int32)
    key: list = []
    for i in range(nrow):
        b = st["board"][st["boff"][i]:st["boff"][i + 1]]
        men[i] = len(b)
        sets = []
        for role in (0, 1):
            ks = sorted(kinds[int(x[1])] for x in b if int(x[0]) == role and int(x[1]) != gk)
            sets.append(ks)
        key.append(tuple(sorted(["+".join(s) or "-" for s in sets])))
    lines = []
    games_of = st["game"]
    ng = len(np.unique(games_of))
    lines.append(f"coverage: games read {ng} of {len(games)}, rows (positions) {nrow}")
    for cut in (24, 16, 12, 8, 6, 4):
        sel = men <= cut
        lines.append(f"coverage: units left <= {cut}: positions {int(sel.sum())} ({sel.mean() * 100:.2f} pct), games reaching it {len(np.unique(games_of[sel]))} ({len(np.unique(games_of[sel])) / max(1, ng) * 100:.1f} pct)")
    small = [i for i in range(nrow) if men[i] <= 8]
    cnt: dict = {}
    gs: dict = {}
    for i in small:
        cnt[key[i]] = cnt.get(key[i], 0) + 1
        gs.setdefault(key[i], set()).add(int(games_of[i]))
    top = sorted(cnt, key=lambda k: -cnt[k])
    lines.append(f"coverage: small sets (8 units or fewer on the board, the goal units not counted, one list of kinds per side): {len(cnt)} distinct, {sum(1 for k in cnt if len(gs[k]) >= 30)} with 30 games or more, {sum(1 for k in cnt if len(gs[k]) < 5)} with fewer than 5 games")
    for k in top[:14]:
        lines.append(f"coverage: set {k[0]} | {k[1]}: positions {cnt[k]}, games {len(gs[k])}")
    return lines


def main(argv: "list | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--data", default=paths.experiment(*HEADS_DIR))
    ap.add_argument("--per-shard", type=int, default=PER_SHARD)
    ap.add_argument("--only-games", type=int, default=0)
    ap.add_argument("--lo", type=int, default=LO)
    ap.add_argument("--hi", type=int, default=0, help="the last ply read (0: the end of the game)")
    ap.add_argument("--root-cap", type=int, default=0)
    ap.add_argument("--trap-fuse", type=int, default=TRAP_FUSE)
    ap.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--budget-seconds", type=float, default=0.0)
    a = ap.parse_args(argv)
    t0 = time.perf_counter()
    out = Path(a.out_dir or paths.experiment("tactical_78_62", "state"))
    out.mkdir(parents=True, exist_ok=True)
    (out / "parts").mkdir(exist_ok=True)
    lines: list = []

    def say(msg: str) -> None:
        lines.append(f"OBS tactical state: {msg}")
        print(lines[-1], flush=True)

    games = load_sample(Path(a.data), a.per_shard, a.only_games)
    rec = human_kind_record()
    say(f"sample: {len(games)} games, {sum(len(g['moves']) for g in games)} plies; rows from ply {a.lo} to {a.hi or 'the end'}; kinds' worth from {rec['source']}: {rec['values']}; bands {rec['bands']}")
    model = make_model(games, rec, a.root_cap)
    picked = OLD.pick_concepts(Path(paths.experiment("unit_fates_78_54", "games_s5016_n500")), SEEDS)
    keys = [k for _dc, k, _d, _z in picked]
    say(f"seeds: the top {len(keys)} distinct kept concepts of 78.54 by |z| on its half A; rules on {RULES}; root cap {a.root_cap or 10}; fuses {FUSE}/{ROW_FUSE}, trap fuse {a.trap_fuse}; horizon {HORIZON}")
    goal = G2M.goal_kind_of(HD.VARIANT)
    npart = (len(games) + CHUNK - 1) // CHUNK
    nrows_done, t1 = 0, time.perf_counter()
    for i in range(npart):
        pf = out / "parts" / f"part_{i:04d}.npz"
        if pf.exists():
            continue
        ch = games[i * CHUNK:(i + 1) * CHUNK]
        t2 = time.perf_counter()
        part = read_chunk(model, ch, a.lo, a.hi, keys, goal, a.trap_fuse, a.threads, i * CHUNK)
        np.savez_compressed(pf, **{k: v for k, v in part.items() if k != "meta"})
        pf.with_suffix(".json").write_text(json.dumps(part["meta"]))
        nrows_done += len(part["ply"])
        say(f"progress: chunk {i + 1} of {npart}: {len(part['ply'])} rows, {time.perf_counter() - t2:.1f} s wall over {a.threads} threads ({(time.perf_counter() - t2) / max(1, len(part['ply'])):.4f} s per row), {int(part['questions'].sum())} questions")
        if a.budget_seconds and time.perf_counter() - t1 > a.budget_seconds and i + 1 < npart:
            say("the time budget ended first; run again to resume")
            (out / "state_obs.txt").write_text("\n".join(lines) + "\n")
            return 2
    parts = []
    for i in range(npart):
        pf = out / "parts" / f"part_{i:04d}.npz"
        z = np.load(pf)
        d = {k: z[k] for k in z.files}
        d["meta"] = json.loads(pf.with_suffix(".json").read_text())
        parts.append(d)
    st = merge(parts)
    save(st, out / "state.npz")
    (out / "games.json").write_text(json.dumps([{k: g[k] for k in ("id", "welo", "belo", "y", "band", "fit")} | {"n": len(g["moves"])} for g in games]))
    nr = len(st["ply"])
    say(f"rows {nr}; questions per row {st['questions'].sum() / max(1, nr):.1f}; rows with a frame {int((st['frames'] > 0).sum())} ({(st['frames'] > 0).mean():.3f}); rows with an unread line {int((st['unread'] > 0).sum())}; unit records {len(st['units'])}")
    u = st["units"]
    say("unit records by class: " + ", ".join(f"{w} {int((u[:, 2] == c).sum())}" for c, w in enumerate(CLASS_WORDS)))
    lost = (u[:, 2] == 0) | (u[:, 2] == 2)
    say("unit records lost or both by what the class rests on: " + ", ".join(f"{w} {int((lost & (u[:, 3] == b)).sum())}" for b, w in enumerate(BASIS_WORDS)))
    for ln in coverage_lines(games, st):
        say(ln)
    say(f"wall {time.perf_counter() - t0:.1f} s")
    (out / "state_obs.txt").write_text("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
