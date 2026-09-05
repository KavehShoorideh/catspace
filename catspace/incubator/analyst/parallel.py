"""parallel.py -- review a player's games on all cores: shard the games, run one review process
per shard (each owns a Stockfish and the Maia ladder), merge the records, write one report.
    python -m catspace.incubator.analyst.parallel data/lichess/x.pgn --user NAME --workers 4 [review args...]"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

import chess.pgn


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("pgn"); ap.add_argument("--user", required=True); ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--since", default=""); ap.add_argument("--perf", default=""); ap.add_argument("--max-games", type=int, default=100000)
    ap.add_argument("--out", default="tmp/review_parallel.txt")
    ap.add_argument("--fresh-cache", action="store_true", help="timing runs: every worker gets an empty cache")
    a, rest = ap.parse_known_args(argv)
    t0 = time.perf_counter()
    games = []
    with open(a.pgn) as f:
        while (g := chess.pgn.read_game(f)) and len(games) < a.max_games:
            if a.since and g.headers.get("Date", "") < a.since: continue
            if a.perf and not any(g.headers.get("TimeControl", "").startswith(x) for x in a.perf.split(",")): continue
            games.append(g)
    tmp = tempfile.mkdtemp(prefix="analyst_shards_")
    shards = [[] for _ in range(a.workers)]
    for i, g in enumerate(games):
        shards[i % a.workers].append(g)
    procs = []
    for w, gs in enumerate(shards):
        if not gs: continue
        path = os.path.join(tmp, f"shard{w}.pgn")
        with open(path, "w") as f:
            for g in gs:
                f.write(str(g) + "\n\n")
        out = os.path.join(tmp, f"shard{w}.txt")
        cmd = [sys.executable, "-m", "catspace.incubator.analyst.review", path, "--user", a.user, "--max-games", str(len(gs)), "--out", out] + rest
        if a.fresh_cache:
            cmd += ["--cache", os.path.join(tmp, f"cache{w}.sqlite")]
        if "--threads" not in rest:
            cmd += ["--threads", "1"]
        procs.append((w, out, subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)))
    for w, out, p in procs:
        p.wait()
    rows, texts = [], []
    offset = 0
    for w, out, p in sorted(procs):
        jl = out.replace(".txt", ".jsonl")
        if not os.path.exists(jl): continue
        rs = [json.loads(l) for l in open(jl) if l.strip()]
        for r in rs:                                   # JSON turned the rating keys into strings
            r["p_played"] = {int(k): v for k, v in r["p_played"].items()}
            r["p_best"] = {int(k): v for k, v in r.get("p_best", {}).items()}
        n = max((r["game"] for r in rs), default=0)
        for r in rs:
            r["game"] += offset
        rows += rs
        body = open(out).read().split("=" * 80, 1)
        if len(body) > 1:
            texts.append(body[1].replace("### game ", f"### shard{w} game "))
        offset += n
    from .review import write_report
    class A: pass
    aa = A(); aa.user = a.user; aa.depth = "?"; aa.ladder_only = False; aa.out = a.out
    ratings = sorted({int(k) for r in rows for k in r["p_played"]})
    write_report(aa, games, rows, ["\n".join(texts)], ratings, t0, final=True)
    print(f"[parallel] {len(games)} games, {len(rows)} moves, {a.workers} workers, {time.perf_counter()-t0:.0f}s -> {a.out}")


if __name__ == "__main__":
    main()
