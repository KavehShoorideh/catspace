"""PLAN 78.65 A: THE FRAME READ'S ROW ON MORE GAMES, COMPACT. A measurement in a sandbox; nothing is wired into any player, nothing is written to rules/learned or data/derived. Game-free: a kind, a role or a
place is a word read from the engine at run time. No engine evaluation, no labeller: the frame read of 78.60 as 78.62 made it (the same model, seeds, fuses and order of kinds), read at the SETTLED rows of
the 78.59 file only (the rows the hazard heads of 78.64 use), and stored in the compact form stage B needs: per row the mover, the count of frames, the unread lines, and per unit its owner, kind, class, what the
class rests on and its trap word. NOT stored: what became of the unit afterwards (the fate of 78.62: it is the future of the row and no input may hold it), the board, the places.

THE SAMPLE. The 800 games that already have the 78.62 state (their settled rows are cut from it: the same read, no new time), plus games drawn AT RANDOM (an md5 order of the game ids, seeded) from the rest of
the 12000 of 78.59, up to the count asked for. The count and the expected time are printed before the run. Chunks of CHUNK games, each chunk a file, a line per chunk; a run that is cut resumes from the chunks
on disk (`--budget-seconds` ends a run after the chunk that crosses it).

  .venv/bin/python -m catspace.incubator.analyst.discovery.frames_state_rows human --n-games N [--budget-seconds S] [--out-dir O]
  .venv/bin/python -m catspace.incubator.analyst.discovery.frames_state_rows fill --fill-dir D [--out-dir O]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

from catspace.incubator.analyst.discovery import frames_measure as OLD
from catspace.incubator.analyst.discovery import graph_layer2_measure as G2M
from catspace.incubator.analyst.discovery import human_heads_data as HD
from catspace.incubator.analyst.discovery import tactical_state as TS
from catspace.io import paths

CHUNK = 40
SEED_WORD = "78.65:state rows"
OUT = ("frames_heads_78_65", "state_rows")
OLD_STATE = ("tactical_78_62", "state")
KEEP = [0, 1, 2, 3, 4]                                             # the unit record's columns kept: owner, kind, class, basis, trap (column 5, the fate, is the row's future and is dropped)


def h(text: str) -> int:
    return int(hashlib.md5((SEED_WORD + text).encode()).hexdigest(), 16)


def all_games() -> "tuple[list, dict]":
    heads = Path(paths.experiment(*HD_DIR))
    games = [json.loads(x) for x in (heads / "games.jsonl").read_text().splitlines() if x.strip()]
    return games, {g["id"]: i for i, g in enumerate(games)}


HD_DIR = TS.HEADS_DIR


def settled_plies(rows: dict) -> dict:
    """{game number in the 78.59 file: sorted list of its settled plies}."""
    out: dict = {}
    for g, p in zip(rows["game"].tolist(), rows["ply"].tolist()):
        out.setdefault(g, []).append(p)
    return {g: sorted(v) for g, v in out.items()}


def old_state_ids() -> list:
    return [g["id"] for g in json.loads((Path(paths.experiment(*OLD_STATE)) / "games.json").read_text())]


def choose(n_games: int, have: list, allg: list, rows_games: set) -> list:
    """The 800 games that have a state, then random others (an md5 order) that have settled rows, up to n_games in all."""
    seen = set(have)
    rest = sorted((g["id"] for g in allg if g["id"] not in seen and g["id"] in rows_games), key=h)
    return list(have) + rest[:max(0, n_games - len(have))]


def pack(parts: list) -> dict:
    """The rows of parts (dicts of arrays with `uoff`) as one dict."""
    out: dict = {}
    for k in ("game", "ply", "mover", "frames", "unread", "questions", "units"):
        out[k] = np.concatenate([p[k] for p in parts]) if parts else np.zeros(0)
    base, offs = 0, [np.zeros(1, np.int64)]
    for p in parts:
        offs.append(p["uoff"][1:] + base)
        base += int(p["uoff"][-1])
    out["uoff"] = np.concatenate(offs)
    return out


def read_rows(model, games: list, plies: list, keys: list, goal: str, trap_fuse: int, threads: int, base: int) -> dict:
    """The frame read at the given plies of each game (games: (start, moves) pairs; plies: a list per game)."""
    res = model.questions_state(games, plies, keys, goal, TS.FUSE, TS.ROW_FUSE, OLD.MAX_SEED_SIZE, True, trap_fuse, TS.HORIZON, threads)
    rs = res["rows"]
    uoff, units = [0], []
    for r in rs:
        units += r["units"]
        uoff.append(len(units))
    u = np.array(units, np.uint8).reshape(-1, 6)[:, KEEP]
    return {"game": np.array([r["game"] + base for r in rs], np.int32), "ply": np.array([r["ply"] for r in rs], np.int16), "mover": np.array([r["mover"] for r in rs], np.int8),
            "frames": np.array([r["frames"] for r in rs], np.int16), "unread": np.array([r["unread"] for r in rs], np.int32), "questions": np.array([r["questions"] for r in rs], np.int64),
            "units": u, "uoff": np.array(uoff, np.int64), "meta": {"kinds": res["kinds"], "roles": res["roles"]}}


def from_old(ids: list, plies_by_id: dict, base: int) -> dict:
    """The settled rows of the games that already have the 78.62 state, cut from it (the same read)."""
    d = Path(paths.experiment(*OLD_STATE))
    st = TS.load(d / "state.npz")
    gj = json.loads((d / "games.json").read_text())
    num_of_id = {g["id"]: k for k, g in enumerate(gj)}
    want = {num_of_id[i]: (k, set(plies_by_id[i])) for k, i in enumerate(ids) if i in num_of_id}
    keep = [j for j in range(len(st["ply"])) if int(st["game"][j]) in want and int(st["ply"][j]) in want[int(st["game"][j])][1]]
    uoff, units = [0], []
    for j in keep:
        u = st["units"][st["uoff"][j]:st["uoff"][j + 1]][:, KEEP]
        units.append(u)
        uoff.append(uoff[-1] + len(u))
    return {"game": np.array([want[int(st["game"][j])][0] + base for j in keep], np.int32), "ply": st["ply"][keep], "mover": st["mover"][keep], "frames": st["frames"][keep],
            "unread": st["unread"][keep], "questions": st["questions"][keep], "units": np.concatenate(units) if units else np.zeros((0, 5), np.uint8), "uoff": np.array(uoff, np.int64),
            "meta": st["meta"] | {}}


def save_part(p: Path, d: dict) -> None:
    np.savez_compressed(p, **{k: v for k, v in d.items() if k != "meta"})


def load_part(p: Path) -> dict:
    z = np.load(p)
    return {k: z[k] for k in z.files}


def cmd_human(a) -> int:
    t0 = time.perf_counter()
    out = Path(a.out_dir or paths.experiment(*OUT))
    (out / "human").mkdir(parents=True, exist_ok=True)
    lines: list = []

    def say(m: str) -> None:
        lines.append(f"OBS state rows: {m}")
        print(lines[-1], flush=True)
    heads = Path(paths.experiment(*HD_DIR))
    allg, _idx = all_games()
    rows = HD.load_rows(heads / "rows.npz")
    plies = settled_plies(rows)
    ids_with_rows = {allg[g]["id"] for g in plies}
    have = old_state_ids()
    ids = choose(a.n_games, have, allg, ids_with_rows)
    by_id = {g["id"]: g for g in allg}
    num = {g["id"]: i for i, g in enumerate(allg)}
    pl = {i: plies[num[i]] for i in ids}
    total = sum(len(v) for v in pl.values())
    new_ids = [i for i in ids if i not in set(have)]
    new_rows = sum(len(pl[i]) for i in new_ids)
    est = a.rate * new_rows
    say(f"sample: {len(ids)} games of the 12000, of which {len(have)} cut from the 78.62 state (no new time) and {len(new_ids)} drawn at random (md5 order of the ids, seed word '{SEED_WORD}'); settled rows {total}, of which {new_rows} to read; expected {est / 60:.0f} minutes at {a.rate} s a row (the rate measured in 78.62 was 0.023 over 10 threads), chunks of {CHUNK} games")
    (out / "human" / "ids.json").write_text(json.dumps(ids))
    if a.dry:
        return 0
    rec = TS.human_kind_record()
    sample = TS.load_sample(Path(paths.experiment(*HD_DIR)), TS.PER_SHARD, 0)
    model = TS.make_model(sample, rec, 0)
    picked = OLD.pick_concepts(Path(paths.experiment("unit_fates_78_54", "games_s5016_n500")), TS.SEEDS)
    keys = [k for _dc, k, _d, _z in picked]
    goal = G2M.goal_kind_of(HD.VARIANT)
    nchunk = (len(ids) + CHUNK - 1) // CHUNK
    t1, done, nread = time.perf_counter(), 0, 0
    old = set(have)
    for c in range(nchunk):
        pf = out / "human" / f"part_{c:04d}.npz"
        if pf.exists():
            continue
        ch = ids[c * CHUNK:(c + 1) * CHUNK]
        t2 = time.perf_counter()
        base = c * CHUNK
        oldpart = from_old([i for i in ch], pl, base) if any(i in old for i in ch) else None
        newi = [(k, i) for k, i in enumerate(ch) if i not in old]
        parts = []
        if oldpart is not None:
            parts.append(oldpart)
        if newi:
            gm = [(None, by_id[i]["moves"]) for _k, i in newi]
            part = read_rows(model, gm, [pl[i] for _k, i in newi], keys, goal, TS.TRAP_FUSE, a.threads, 0)
            part["game"] = np.array([newi[int(g)][0] + base for g in part["game"]], np.int32)         # the game's place in the sample
            parts.append(part)
        d = pack(parts)
        order = np.lexsort((d["ply"], d["game"]))                                           # one order: by game, then ply (units re-cut in that order)
        d = reorder(d, order)
        save_part(pf, d)
        done += len(d["ply"])
        dt = time.perf_counter() - t2
        el = time.perf_counter() - t1
        nread += 1
        left = sum(1 for k in range(c + 1, nchunk) if not (out / "human" / f"part_{k:04d}.npz").exists())
        say(f"progress: chunk {c + 1} of {nchunk}: {len(d['ply'])} rows, {dt:.1f} s wall over {a.threads} threads ({dt / max(1, len(d['ply'])):.4f} s per row), {int(d['questions'].sum())} questions; run so far {el:.0f} s; about {left * el / nread:.0f} s left at this run's mean chunk time")
        if a.budget_seconds and el > a.budget_seconds and c + 1 < nchunk:
            say("the time budget ended first; run again to resume")
            (out / "human" / "state_obs.txt").write_text("\n".join(lines) + "\n")
            return 2
    parts = [load_part(out / "human" / f"part_{c:04d}.npz") for c in range(nchunk)]
    d = pack(parts)
    np.savez_compressed(out / "human" / "state_rows.npz", **d)
    (out / "human" / "meta.json").write_text(json.dumps({"kinds": json.loads((Path(paths.experiment(*OLD_STATE)) / "state.json").read_text())["kinds"], "ids": ids}))
    say(f"done: {len(ids)} games, {len(d['ply'])} rows, {len(d['units'])} unit records; wall {time.perf_counter() - t0:.1f} s; disk {sum(f.stat().st_size for f in (out / 'human').glob('*')) / 1e6:.1f} MB")
    (out / "human" / "state_obs.txt").write_text("\n".join(lines) + "\n")
    return 0


def reorder(d: dict, order: np.ndarray) -> dict:
    """The rows of d in the given order, the unit records re-cut with them."""
    out = {k: d[k][order] for k in ("game", "ply", "mover", "frames", "unread", "questions")}
    chunks = [d["units"][d["uoff"][j]:d["uoff"][j + 1]] for j in order]
    out["units"] = np.concatenate(chunks) if chunks else d["units"]
    out["uoff"] = np.concatenate([[0], np.cumsum([len(c) for c in chunks])]).astype(np.int64)
    return out


def cmd_fill(a) -> int:
    from catspace.incubator.analyst.discovery import events_fill as EF
    t0 = time.perf_counter()
    out = Path(a.out_dir or paths.experiment(*OUT))
    (out / "fill").mkdir(parents=True, exist_ok=True)
    lines: list = []

    def say(m: str) -> None:
        lines.append(f"OBS state rows: {m}")
        print(lines[-1], flush=True)
    games = json.loads((Path(a.fill_dir) / "fill_games.json").read_text())
    HD.OPEN_PLIES = EF.OPENING
    rows = HD.build_rows(games, a.threads, lambda m: None, concepts=False)
    plies = settled_plies(rows)
    say(f"fill games {len(games)}; settled rows {sum(len(v) for v in plies.values())}; expected {a.rate * sum(len(v) for v in plies.values()) / 60:.0f} minutes at {a.rate} s a row")
    rec = TS.human_kind_record()
    sample = TS.load_sample(Path(paths.experiment(*HD_DIR)), TS.PER_SHARD, 0)
    model = TS.make_model(sample, rec, 0)
    picked = OLD.pick_concepts(Path(paths.experiment("unit_fates_78_54", "games_s5016_n500")), TS.SEEDS)
    keys = [k for _dc, k, _d, _z in picked]
    goal = G2M.goal_kind_of(HD.VARIANT)
    order = sorted(plies)
    nchunk = (len(order) + CHUNK - 1) // CHUNK
    for c in range(nchunk):
        pf = out / "fill" / f"part_{c:04d}.npz"
        if pf.exists():
            continue
        ch = order[c * CHUNK:(c + 1) * CHUNK]
        t2 = time.perf_counter()
        part = read_rows(model, [(games[g].get("start"), games[g]["moves"]) for g in ch], [plies[g] for g in ch], keys, goal, TS.TRAP_FUSE, a.threads, 0)
        part["game"] = np.array([ch[int(g)] for g in part["game"]], np.int32)               # the number of the game in fill_games.json
        d = reorder(part, np.lexsort((part["ply"], part["game"])))
        save_part(pf, d)
        say(f"progress: chunk {c + 1} of {nchunk}: {len(d['ply'])} rows, {time.perf_counter() - t2:.1f} s")
    parts = [load_part(out / "fill" / f"part_{c:04d}.npz") for c in range(nchunk)]
    d = pack(parts)
    np.savez_compressed(out / "fill" / "state_rows.npz", **d)
    say(f"done: {len(d['ply'])} rows, {len(d['units'])} unit records; wall {time.perf_counter() - t0:.1f} s")
    (out / "fill" / "state_obs.txt").write_text("\n".join(lines) + "\n")
    return 0


def main(argv: "list | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    hm = sub.add_parser("human")
    hm.add_argument("--n-games", type=int, required=True)
    hm.add_argument("--dry", action="store_true")
    hm.add_argument("--rate", type=float, default=0.023)
    hm.add_argument("--budget-seconds", type=float, default=0.0)
    fl = sub.add_parser("fill")
    fl.add_argument("--fill-dir", default=paths.experiment("events_78_64", "fill"))
    fl.add_argument("--rate", type=float, default=0.023)
    for p in (hm, fl):
        p.add_argument("--out-dir", default=None)
        p.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args(argv)
    return {"human": cmd_human, "fill": cmd_fill}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
