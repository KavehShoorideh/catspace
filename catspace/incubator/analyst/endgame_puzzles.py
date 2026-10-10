"""THE PLANNER ON ENDGAME PUZZLES (Kaveh 2026-10-10: "before you do the playable engine, let's try our lemmas on some
endgame puzzles").

Lichess puzzles with the endgame themes (pawn, rook, bishop, knight, queen, queen+rook endgames), sampled from a slice
of the puzzle file no earlier experiment used. A puzzle's FEN is the position BEFORE the opponent's move; the first move
of its line is theirs, the second is the solution. Per puzzle: the planner's decision (rust/analyst_core/src/plan.rs:
take a certified win, defend, the cheapest loss, an armed threat) and the move it would play, against the solution;
proofs, nodes, time. Reference: Maia-1900's most likely move. Misses are kept for digging, one by one.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter, defaultdict

import chess

from catspace.io import paths

THEMES = ("pawnEndgame", "rookEndgame", "bishopEndgame", "knightEndgame", "queenEndgame", "queenRookEndgame")


def collect(per_theme: int, stride: int = 11, offset: int = 5) -> dict:
    import csv, io, zstandard
    out = {t: [] for t in THEMES}
    with open(paths.assets_dir() / "lichess" / "lichess_db_puzzle.csv.zst", "rb") as fh:
        reader = csv.DictReader(io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(fh), encoding="utf-8"))
        for i, row in enumerate(reader):
            if i % stride != offset:
                continue
            for t in row["Themes"].split():
                if t in out and len(out[t]) < per_theme:
                    out[t].append(row)
            if all(len(v) >= per_theme for v in out.values()):
                break
    return out


def played(dec) -> str | None:
    if not dec:
        return None
    kind, move = dec[0], dec[1]
    if kind in ("", "defend", "cheapest"):
        return move
    return kind                                                       # an armed threat: the preparing move is played


def _one(args):
    theme, row = args
    os.environ.setdefault("RAYON_NUM_THREADS", "1")
    import analyst_core
    b = chess.Board(row["FEN"])
    moves = row["Moves"].split()
    b.push_uci(moves[0])
    fen = b.fen()
    t = time.time()
    trace, dec, proofs, nodes, ms, imm = analyst_core.plan_trace(fen)
    mv = played(dec)
    maia = analyst_core.maia_policy(fen, str(paths.assets_dir() / "engines" / "maia_onnx" / "maia-1900.onnx"), 1.0)
    maia_top = max(maia, key=lambda x: x[1])[0] if maia else None
    return dict(theme=theme, id=row["PuzzleId"], rating=int(row["Rating"]), fen=fen, solution=moves[1], line=moves[1:],
                played=mv, kind=(dec[0] if dec else None), summary=(dec[3] if dec else None),
                imm=(imm[1] if imm else None), proofs=proofs, nodes=nodes, ms=ms, maia=maia_top, trace=trace,
                themes=row["Themes"])


def main(argv=None):
    from multiprocessing import Pool
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-theme", type=int, default=60)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args(argv)
    t0 = time.time()
    rows = collect(a.per_theme)
    print(f"TIME collect: {time.time() - t0:.0f} s", flush=True)
    jobs = [(t, r) for t, rs in rows.items() for r in rs]
    t1 = time.time()
    with Pool(a.workers) as p:
        res = list(p.imap_unordered(_one, jobs, chunksize=4))
    print(f"TIME planner on {len(res)} puzzles: {time.time() - t1:.0f} s", flush=True)
    out = paths.experiment(f"endgame_puzzles_{a.per_theme}.jsonl")
    with open(out, "w") as fh:
        for r in res:
            fh.write(json.dumps(r) + "\n")
    by = defaultdict(list)
    for r in res:
        by[r["theme"]].append(r)
    print(f"OBS endgame puzzles: solved = the planner's move is the first solution move; reference: Maia-1900's top move")
    print(f"OBS   {'theme':18s} {'n':>4s} {'planner':>8s} {'decided':>8s} {'Maia-1900':>10s} {'proofs med':>11s} {'ms med':>7s}")
    for t in THEMES + ("ALL",):
        rs = res if t == "ALL" else by[t]
        n = len(rs)
        if not n:
            continue
        ok = sum(r["played"] == r["solution"] for r in rs)
        dec = sum(r["played"] is not None for r in rs)
        mz = sum(r["maia"] == r["solution"] for r in rs)
        pr = sorted(r["proofs"] for r in rs); ms = sorted(r["ms"] for r in rs)
        print(f"OBS   {t:18s} {n:4d} {100 * ok / n:7.0f}% {100 * dec / n:7.0f}% {100 * mz / n:9.0f}% {pr[n // 2]:11d} {ms[n // 2]:7.1f}")
    kinds = Counter()
    for r in res:
        if r["played"] == r["solution"]:
            kinds[f"solved by: {r['kind'] or 'an immediate win'}"] += 1
        elif r["played"] is None:
            kinds["miss: the planner had no decision"] += 1
        else:
            kinds[f"miss: decided {r['kind'] or 'an immediate win'}, another move"] += 1
    print("OBS   outcomes:", dict(kinds))
    print(f"written to {out}")


if __name__ == "__main__":
    main()
