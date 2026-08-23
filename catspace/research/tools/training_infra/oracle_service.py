#!/usr/bin/env python
"""oracle_service.py -- the gko1 flywheel's producer: pool -> queue -> oracle labels.

Kaveh 2026-08-21: "start the flywheel where you generate positions, ask maia, train on
the results, and keep asking." This is the ASK half, pipelined per the plan
(docs/GKO1_PLAN.md sec.3): a sqlite store (journal DELETE -- the tb-cache scar) holds the
position POOL, an acquisition-ranked QUEUE, and atomic LABELS. Oracles:

    sf           stockfish multipv over ALL legal moves, depth+time capped
                 (ChessBench parity); cp -> win prob via the lichess logistic
    maia:<band>  lc0 + maia-<band>.pb.gz at nodes=1 with verbose-move-stats: the RAW
                 POLICY distribution P(move) -- verified to sum to 1

Commands:
    init            create tables
    seed --pool-*   fill the pool from corpus caches + random walks
    enqueue --n N --oracle sf|maia:1500 [--scores f.npz]   rank by scores (or random)
    produce --oracle ... [--budget N]                      worker loop, atomic claims
    stats           health snapshot (the observability heartbeat)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sqlite3
import time

DB = "data/derived/gko1_oracle.sqlite"
WP_K = 0.00368208          # lichess/lc0 logistic constant: wp = 1/(1+exp(-k*cp))


def db():
    c = sqlite3.connect(DB, timeout=60)
    # WAL, not DELETE (2026-08-21 revision): the DELETE ruling was for the perpetual-
    # reader tb-cache topology (readers pinned the WAL -> 25G growth). THIS store is
    # many-tiny-writers + rare-readers -- DELETE mode gave four reader/writer
    # starvation incidents in one day, ending with a wedged writer's journal blocking
    # the fleet for an hour. WAL + autocheckpoint + size cap fits this topology;
    # journal_mode is set once by the store owner, NEVER forced per-connection
    # (a per-connection mode flip needs an exclusive lock and crashes concurrent
    # clients -- the bug that emptied the fleet at relaunch).
    c.execute("PRAGMA busy_timeout=60000")
    c.execute("PRAGMA synchronous=NORMAL")
    return c


def cmd_init(_a):
    c = db()
    c.executescript("""
    CREATE TABLE IF NOT EXISTS pool(fen TEXT PRIMARY KEY, source TEXT);
    CREATE TABLE IF NOT EXISTS queue(
      fen TEXT, oracle TEXT, score REAL DEFAULT 0, status TEXT DEFAULT 'todo',
      claimed_at REAL, PRIMARY KEY(fen, oracle));
    CREATE TABLE IF NOT EXISTS labels(
      fen TEXT, oracle TEXT, payload TEXT, created REAL, PRIMARY KEY(fen, oracle));
    CREATE INDEX IF NOT EXISTS q_status ON queue(status, score);
    CREATE INDEX IF NOT EXISTS q_oracle ON queue(oracle, status, score);
    """)
    c.commit(); c.close()
    print(f"[oracle] initialized {DB}")


def cmd_seed(a):
    import numpy as np
    import chess
    from catspace.research.components.encoder.approaches.reach_probability.src import (
        trajectories as T)
    from catspace.research.tools.training_infra.concepts_reference import board_from_row
    c = db(); n_in = 0
    rng = np.random.default_rng(a.seed)
    specs = []
    if a.pool_human:
        specs.append(("human", dict(n_human=30000, n_sf=0, n_piecedown=0), a.pool_human))
    if a.pool_sf:
        specs.append(("sf", dict(n_human=0, n_sf=20000, n_piecedown=45906), a.pool_sf))
    for src, kw, n_want in specs:
        tr = T.build(seed=0, cache=True, max_plies=400, **kw)
        rows = rng.choice(len(tr.tok), min(n_want, len(tr.tok)), replace=False)
        for r in rows:
            b = board_from_row(tr.tok[r], tr.glob[r])
            if not b.is_game_over():
                c.execute("INSERT OR IGNORE INTO pool VALUES (?,?)", (b.fen(), src))
                n_in += 1
    for _ in range(a.pool_walks):
        b = chess.Board()
        depth = int(rng.integers(6, 60))
        for _2 in range(depth):
            mv = list(b.legal_moves)
            if not mv or b.is_game_over():
                break
            b.push(mv[rng.integers(0, len(mv))])
        if not b.is_game_over():
            c.execute("INSERT OR IGNORE INTO pool VALUES (?,?)", (b.fen(), "walk"))
            n_in += 1
    c.commit()
    tot = c.execute("SELECT COUNT(*) FROM pool").fetchone()[0]
    c.close()
    print(f"[oracle] seeded {n_in:,} (pool total {tot:,})")


def cmd_enqueue(a):
    import numpy as np
    c = db()
    if a.scores and os.path.exists(a.scores):
        z = np.load(a.scores, allow_pickle=True)
        fens, sc = list(z["fen"]), z["score"]
        order = np.argsort(-sc)[:a.n]
        rows = [(str(fens[i]), a.oracle, float(sc[i])) for i in order]
        mode = "scored"
    else:
        # subsample candidates by random rowid FIRST, then PK-indexed anti-join --
        # the old NOT IN + ORDER BY RANDOM() scanned 900k x 600k under a SHARED lock
        # and starved every writer for minutes (the 2026-08-21 producer collapse)
        rows = [(f, a.oracle, 0.0) for (f,) in c.execute(
            "SELECT fen FROM pool WHERE rowid IN "
            "(SELECT rowid FROM pool ORDER BY RANDOM() LIMIT ?) "
            "AND NOT EXISTS (SELECT 1 FROM labels "
            "WHERE labels.fen = pool.fen AND labels.oracle = ?) LIMIT ?",
            (max(a.n * 4, 20000), a.oracle, a.n))]
        mode = "random(bootstrap)"
    c.executemany("INSERT OR IGNORE INTO queue(fen, oracle, score) VALUES (?,?,?)", rows)
    c.commit(); c.close()
    print(f"[oracle] enqueued {len(rows):,} for {a.oracle} ({mode})")


def _sf_engine(a):
    import chess.engine
    eng = chess.engine.SimpleEngine.popen_uci(a.engine)
    eng.configure({"Threads": 2, "Hash": 64, "UCI_ShowWDL": True})
    return eng


def _maia_engine(band):
    import chess.engine
    return chess.engine.SimpleEngine.popen_uci(
        ["lc0", f"--weights=data/external/maia/maia-{band}.pb.gz",
         "--verbose-move-stats", "--threads=1", "--backend=eigen"])
    # eigen = pure CPU: at nodes=1 the policy read needs no GPU, and lc0-on-Metal
    # wedges the shared command queue against the kitty server's MPS forwards
    # (2026-08-21 "waiting for localhost" lockup)


_P_RE = re.compile(r"^([a-h][1-8][a-h][1-8]\S*).*?P: *([0-9.]+)%")


def query_sf(eng, b, a):
    import chess.engine
    mv = list(b.legal_moves)
    info = eng.analyse(b, chess.engine.Limit(depth=a.sf_depth, time=a.sf_time),
                       multipv=len(mv))
    out = {}
    wdl_out = {}
    for e in info:
        if "pv" not in e or not e["pv"]:
            continue
        m = e["pv"][0]
        # TRUE W/D/L (Kaveh 2026-08-23: scalar wp destroys basin structure --
        # drawn-fortress vs coin-flip are different positions; SF computes wdl
        # internally, record it): per-mille from the mover's POV
        if "wdl" in e:
            w = e["wdl"].pov(b.turn)
            wdl_out[m.uci()] = [w.wins, w.draws, w.losses]
        s = e["score"].pov(b.turn)
        if s.is_mate():
            wp = 1.0 if s.mate() > 0 else 0.0
        else:
            wp = 1.0 / (1.0 + math.exp(-WP_K * s.score()))
        out[m.uci()] = round(wp, 5)
    return {"kind": "sf_av", "wp": out, "wdl": wdl_out, "n_legal": len(mv)}


def query_maia(eng, b):
    import chess.engine
    with eng.analysis(b, chess.engine.Limit(nodes=1),
                      info=chess.engine.INFO_ALL) as an:
        strings = [i["string"] for i in an if "string" in i]
    ps = {}
    for s in strings:
        m = _P_RE.search(s)
        if m and m.group(1) != "node":
            ps[m.group(1)] = float(m.group(2)) / 100.0
    tot = sum(ps.values())
    if tot > 0:
        ps = {k: round(v / tot, 6) for k, v in ps.items()}
    return {"kind": "maia_policy", "p": ps, "n_legal": len(list(b.legal_moves))}


def cmd_produce(a):
    import chess
    oracle = a.oracle
    band = oracle.split(":")[1] if oracle.startswith("maia") else None
    eng = _maia_engine(band) if band else _sf_engine(a)
    c = db()
    done = errs = 0
    t0 = time.time(); t_hb = t0
    while a.budget <= 0 or done < a.budget:
        # Claim among the top-8 by score, chosen at random, retrying lost races:
        # with N siblings and a deterministic LIMIT 1, every cycle all N fetch the
        # SAME row, one wins the UPDATE and the rest read rowcount=0 as "queue
        # empty" and exit (the 2026-08-21 sf-fleet silent walkout -- 72,800 rows
        # todo, four workers gone). Randomized top-K + retry makes losing a race
        # a delay, not a death.
        row = None
        for _ in range(6):
            with c:                                # atomic claim (U10b)
                rs = c.execute("SELECT fen FROM queue WHERE oracle=? AND "
                               "status='todo' ORDER BY score DESC LIMIT 8",
                               (oracle,)).fetchall()
                if not rs:
                    break
                cand = rs[int(time.time() * 1e6) % len(rs)][0]
                got = c.execute(
                    "UPDATE queue SET status='claimed', claimed_at=? "
                    "WHERE fen=? AND oracle=? AND status='todo'",
                    (time.time(), cand, oracle)).rowcount
                if got == 1:
                    row = cand
            if row is not None:
                break
            time.sleep(0.05)
        if row is None:
            print(f"[oracle:{oracle}] queue empty after {done:,} labels; exiting", flush=True)
            break
        try:
            b = chess.Board(row)
            payload = query_maia(eng, b) if band else query_sf(eng, b, a)
            with c:
                c.execute("INSERT OR REPLACE INTO labels(fen, oracle, payload, created)"
                          " VALUES (?,?,?,?)",
                          (row, oracle, json.dumps(payload), time.time()))
                c.execute("UPDATE queue SET status='done' WHERE fen=? AND oracle=?",
                          (row, oracle))
            done += 1
        except Exception as e:
            errs += 1
            with c:
                c.execute("UPDATE queue SET status='error' WHERE fen=? AND oracle=?",
                          (row, oracle))
            if errs % 25 == 1:
                print(f"[oracle:{oracle}] error #{errs}: {str(e)[:90]}", flush=True)
            if errs > 200:
                break
        if time.time() - t_hb > 60:
            r = (time.time() - t0)
            print(f"[oracle:{oracle}] {done:,} labeled ({done/r:.1f}/s) errs {errs} "
                  f"[{r/60:.0f}m]", flush=True)
            t_hb = time.time()
    eng.quit(); c.close()
    print(f"[oracle:{oracle}] DONE {done:,} labels, {errs} errors "
          f"[{(time.time()-t0)/60:.1f}m]", flush=True)


def cmd_stats(_a):
    c = db()
    for t in ("pool", "labels"):
        print(t, c.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0])
    for row in c.execute("SELECT oracle, status, COUNT(*) FROM queue "
                         "GROUP BY oracle, status"):
        print("queue", *row)
    for row in c.execute("SELECT oracle, COUNT(*) FROM labels GROUP BY oracle"):
        print("labels", *row)
    c.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init")
    p = sub.add_parser("seed")
    p.add_argument("--pool-human", type=int, default=300000)
    p.add_argument("--pool-sf", type=int, default=150000)
    p.add_argument("--pool-walks", type=int, default=50000)
    p.add_argument("--seed", type=int, default=0)
    p = sub.add_parser("enqueue")
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--oracle", required=True)
    p.add_argument("--scores", default="")
    p = sub.add_parser("produce")
    p.add_argument("--oracle", required=True)
    p.add_argument("--budget", type=int, default=0)
    p.add_argument("--engine", default="/opt/homebrew/bin/stockfish")
    p.add_argument("--sf-depth", type=int, default=10)
    p.add_argument("--sf-time", type=float, default=0.05)
    p = sub.add_parser("loop", help="produce FOREVER: when the queue drains, self-refill "
                       "with random pool batches (Kaveh: 'if you don't know what to "
                       "generate, generate something')")
    p.add_argument("--oracle", required=True)
    p.add_argument("--refill", type=int, default=5000)
    p.add_argument("--engine", default="/opt/homebrew/bin/stockfish")
    p.add_argument("--sf-depth", type=int, default=10)
    p.add_argument("--sf-time", type=float, default=0.05)
    sub.add_parser("stats")
    a = ap.parse_args()
    def cmd_loop(a):
        while True:
            c = db()
            todo = c.execute("SELECT COUNT(*) FROM queue WHERE oracle=? AND status='todo'",
                             (a.oracle,)).fetchone()[0]
            c.close()
            if todo == 0:
                class E: n = a.refill; oracle = a.oracle; scores = ""
                cmd_enqueue(E)
            a.budget = 0
            cmd_produce(a)
            time.sleep(2)
    {"init": cmd_init, "seed": cmd_seed, "enqueue": cmd_enqueue,
     "produce": cmd_produce, "stats": cmd_stats, "loop": cmd_loop}[a.cmd](a)


if __name__ == "__main__":
    main()
