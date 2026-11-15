"""PLAN 78.47 B THE SHORTCUT AGAINST THE READ AS BUILT: a reply that touches nothing the read looked at is answered from the stored result of an earlier reply of the same move. This must change NO verdict and NO value
where the fuse does not bind, so every position is read twice at the same (large) fuse, with the shortcut and without it (`forcing(..., shortcut=False)`: every reply read in full), and the answers are compared threat
move by threat move: the verdict, the value, and the list of replies with the value and the open flag of each.

Only positions where NEITHER read was cut by the fuse are compared (a cut read ends where the fuse ends, and the shortcut reads fewer positions to get there, so a cut read is not comparable); the others are counted apart.
Printed: positions read, compared, cut (either), threat moves compared, moves that differ, replies answered from the stored result, and every difference with position, move and the two answers.

  .venv/bin/python -m catspace.incubator.analyst.discovery.forcing_shortcut_measure --variant chess [--games-dir DIR --every N | --sets dev_300] [--fuse N] [--limit N] [--jobs N] [--tag NAME]
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import time
from pathlib import Path

import cs_engine as CE

_ST: dict = {}


def _init(variant: str) -> None:
    from catspace.incubator.analyst.discovery import question_sheet as QS
    from catspace.incubator.analyst.discovery import sheet_rust as SR
    env = QS.Env(variant)
    _ST.update(model=SR.model(env, QS.store_worth(variant)))


def answer_key(t: dict) -> tuple:
    """What a threat move's answer is, for the comparison: verdict, value, and every reply (name, value, open, ends)."""
    return (t["verdict"], t["value"], tuple((r["reply"], r["value"], r["open"], r["ends"]) for r in t["replies"]))


def compare(a: dict, b: dict) -> list:
    """The differences between two reads of one position -> [(move, why, answer a, answer b)] (a: the read as built, b: with the shortcut)."""
    da = {t["move"]: t for t in a["threats"]}
    db = {t["move"]: t for t in b["threats"]}
    out = []
    for m in sorted(set(da) | set(db)):
        if m not in da or m not in db:
            out.append((m, "move found by one read only", None, None))
            continue
        ka, kb = answer_key(da[m]), answer_key(db[m])
        if ka == kb:
            continue
        why = "verdict" if ka[0] != kb[0] else "value" if ka[1] != kb[1] else "replies"
        out.append((m, why, ka, kb))
    return out


def run_one(arg: tuple) -> dict:
    pos, fuse = arg
    model = _ST["model"]
    out = {"pos": pos, "read": 0, "compared": 0, "cut": 0, "moves": 0, "diffs": [], "unchanged": 0, "nodes_as_built": 0, "nodes_shortcut": 0, "sec_as_built": 0.0, "sec_shortcut": 0.0}
    try:
        t0 = time.perf_counter()
        a = model.forcing(pos, fuse, None, False)
        t1 = time.perf_counter()
        b = model.forcing(pos, fuse, None, True)
        t2 = time.perf_counter()
    except ValueError:
        return out
    out["read"] = 1
    out["sec_as_built"], out["sec_shortcut"] = t1 - t0, t2 - t1
    out["nodes_as_built"], out["nodes_shortcut"] = a["nodes"], b["nodes"]
    out["unchanged"] = b["cost"]["unchanged_took"] + b["cost"]["unchanged_quiet"]
    out["replies"] = b["cost"]["replies_took"] + b["cost"]["replies_quiet"]
    out["untouched"] = b["cost"]["untouched"]
    out["why"] = b["cost"]["why"]
    if a["cut"] or b["cut"]:
        out["cut"] = 1
        return out
    out["compared"] = 1
    out["moves"] = len(a["threats"])
    out["diffs"] = compare(a, b)
    return out


def positions_of_games(games_dir: str, every: int) -> list:
    recs = [json.loads(line) for line in (Path(games_dir) / "games.jsonl").read_text().splitlines() if line.strip()]
    out = []
    for r in recs:
        for ply in range(0, len(r["moves"]) + 1, every):
            out.append(CE.state(r["variant"], r.get("start"), r["moves"][:ply])[0])
    return out


def main(argv: "list | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--variant", required=True)
    ap.add_argument("--games-dir", default=None)
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--sets", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--fuse", type=int, default=2_000_000)
    ap.add_argument("--tag", default=None)
    a = ap.parse_args(argv)
    poss: list = []
    if a.games_dir:
        poss += positions_of_games(a.games_dir, a.every)
    if a.sets:
        from catspace.incubator.analyst.discovery import puzzle_apply as PA
        for name in a.sets.split(","):
            poss += [r["fen"] for r in PA.load_set(name)]
    if a.limit:
        poss = poss[:a.limit]
    t0 = time.perf_counter()
    with multiprocessing.get_context("spawn").Pool(a.jobs, initializer=_init, initargs=(a.variant,)) as pool:
        rows = pool.map(run_one, [(p, a.fuse) for p in poss], chunksize=1)
    tag = a.tag or "shortcut"
    read = sum(r["read"] for r in rows)
    comp = [r for r in rows if r["compared"]]
    diffs = [(r["pos"], d) for r in comp for d in r["diffs"]]
    print(f"OBS forcing shortcut {tag}: positions {len(rows)} (read {read}), fuse {a.fuse}, jobs {a.jobs}, wall {time.perf_counter() - t0:.1f} s", flush=True)
    print(f"OBS forcing shortcut {tag}: compared (neither read cut) {len(comp)}, cut by the fuse in either read {sum(r['cut'] for r in rows)}; threat moves compared {sum(r['moves'] for r in comp)}; "
          f"threat moves that differ in verdict, value or replies {len(diffs)}; positions with a difference {len({p for p, _ in diffs})}", flush=True)
    rep = max(1, sum(r.get("replies", 0) for r in comp))
    print(f"OBS forcing shortcut {tag}: on the compared positions, replies read {rep}, answered from the stored result {sum(r['unchanged'] for r in comp)} ({100 * sum(r['unchanged'] for r in comp) / rep:.1f} pct); "
          f"positions asked: as built {sum(r['nodes_as_built'] for r in comp)}, with the shortcut {sum(r['nodes_shortcut'] for r in comp)}; "
          f"seconds: as built {sum(r['sec_as_built'] for r in comp):.1f}, with the shortcut {sum(r['sec_shortcut'] for r in comp):.1f}", flush=True)
    why = [sum(r["why"][i] for r in comp if "why" in r) for i in range(8)]
    print(f"OBS forcing shortcut {tag}: on the compared positions, replies that touch nothing {sum(r.get('untouched', 0) for r in comp)}; replies that touch something by reason "
          f"(1 takes or ends the game, 2 not a plain move of a unit no exchange involves, 3 another unit changes, 4 a line is opened or blocked, 5 lands on or leaves a place of ours, "
          f"6 bears on a hot place, 7 would after one more move of its own): {why[1:8]}", flush=True)
    for pos, (m, why, ka, kb) in diffs:
        print(f"OBS forcing shortcut {tag}: DIFFERENCE ({why}) position {pos} move {m}: as built {ka}; shortcut {kb}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
