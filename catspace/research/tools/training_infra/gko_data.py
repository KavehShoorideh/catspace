#!/usr/bin/env python
"""gko_data.py -- the flywheel's converter stage: oracle labels -> training shards.

Drains (fen, oracle, payload) rows from the gko1 store, builds the graph arrays via the
SAME parent_graph/glob_of_board the models train on, and writes tensorized shards
(npz, SHARD_N rows each) that gko_train memory-maps as they appear. Runs forever
(pipelining mandate); tracks progress in the store (labels.converted flag added here).

Per row the shard carries: graph arrays (sq/gl/mv/mut/vac/dsc/att/n_mv/n_att),
oracle kind, elo band (0 for sf), per-move targets aligned to the move-node order:
    sf     wp[MV_CAP] float32 (nan where absent)
    maia   p[MV_CAP]  float32 (nan where absent)

    .venv/bin/python -m ...gko_data loop --out data/derived/gko1_shards
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import time

import numpy as np

DB = "data/derived/gko1_oracle.sqlite"
SHARD_N = 2048
MV_CAP = 48


def db():
    c = sqlite3.connect(DB, timeout=60)
    c.execute("PRAGMA busy_timeout=60000")  # WAL store; never force mode per-connection
    return c


def ensure_flag():
    c = db()
    cols = [r[1] for r in c.execute("PRAGMA table_info(labels)")]
    if "converted" not in cols:
        c.execute("ALTER TABLE labels ADD COLUMN converted INTEGER DEFAULT 0")
        c.commit()
    c.close()


def convert_batch(out_dir, batch=SHARD_N, oracle_like=None, newest=False):
    """oracle_like: optional SQL LIKE pattern ('sf', 'maia:%') restricting which
    oracle's labels this batch drains. The unfiltered scan drains in rowid order,
    which starves whichever oracle was enqueued later -- sf sat at 4.9pct converted
    while maia bands ran to 96pct, so the shard window went sf-free and silently
    switched off l_sf/l_sv and every sf_av gradient (2026-08-23)."""
    import chess
    from catspace.research.tools.training_infra.gen_move_graphs import (
        parent_graph, glob_of_board)
    c = db()
    # newest-first matters: the WDL re-labelling APPENDED rows, so the newest sf
    # labels carry wdl triples (100pct of the newest 5k) and the oldest carry none
    # (0pct). Draining ascending yields wdl-free shards and silently kills l_sv.
    order = "ORDER BY rowid DESC " if newest else ""
    if oracle_like:
        rows = c.execute("SELECT fen, oracle, payload, rowid FROM labels "
                         f"WHERE converted=0 AND oracle LIKE ? {order}LIMIT ?",
                         (oracle_like, batch)).fetchall()
    else:
        rows = c.execute("SELECT fen, oracle, payload, rowid FROM labels "
                         f"WHERE converted=0 {order}LIMIT ?", (batch,)).fetchall()
    if not rows:
        c.close()
        return 0
    KEYS = ("sq", "gl", "mv", "mut", "vac", "dsc", "att", "pslot")
    acc = {k: [] for k in KEYS}
    n_mv_l, n_att_l, kind_l, band_l, tgt_l, fen_l, wdl_l = [], [], [], [], [], [], []
    rep_l = []
    done_ids = []
    for fen, oracle, payload, rid in rows:
        try:
            b = chess.Board(fen)
            g, _ = parent_graph(b)
            pl = json.loads(payload)
            mvs = [m.uci() for m in list(b.legal_moves)[:int(g["n_mv"])]]
            tgt = np.full(MV_CAP, np.nan, np.float32)
            wdl = np.full((MV_CAP, 3), np.nan, np.float16)
            rep = np.full((MV_CAP, 3), 255, np.uint8)   # SF best reply per child:
            if pl["kind"] == "sf_av":                    # [from, to, promo] (PV[1])
                src = pl["wp"]; kind = 0; band = 0
                wsrc = pl.get("wdl", {})
                psrc = pl.get("pv", {})
                for j, u in enumerate(mvs):
                    if u in wsrc:
                        wdl[j] = np.asarray(wsrc[u], np.float16) / 1000.0
                    line = psrc.get(u)
                    if line:
                        rm = chess.Move.from_uci(line[0])
                        rep[j] = (rm.from_square, rm.to_square,
                                  rm.promotion or 0)
            else:
                src = pl["p"]; kind = 1
                band = int(oracle.split(":")[1])
            for j, u in enumerate(mvs):
                if u in src:
                    tgt[j] = src[u]
            for k in KEYS:
                if k == "gl":
                    acc[k].append(glob_of_board(b))
                else:
                    acc[k].append(g[k])
            n_mv_l.append(g["n_mv"]); n_att_l.append(g["n_att"])
            kind_l.append(kind); band_l.append(band); tgt_l.append(tgt)
            wdl_l.append(wdl)
            rep_l.append(rep)
            fen_l.append(fen)
        except Exception:
            pass
        done_ids.append(rid)
    if fen_l:
        os.makedirs(out_dir, exist_ok=True)
        stamp = f"{int(time.time()*1000)%10**10}_{len(fen_l)}"
        np.savez_compressed(
            os.path.join(out_dir, f"shard_{stamp}.npz"),
            **{k: np.stack(acc[k]) for k in KEYS},
            n_mv=np.array(n_mv_l, np.uint8), n_att=np.array(n_att_l, np.uint16),
            kind=np.array(kind_l, np.int8), band=np.array(band_l, np.int16),
            tgt=np.stack(tgt_l), wdl=np.stack(wdl_l), rep=np.stack(rep_l),
            fen=np.array(fen_l))
    with c:
        c.executemany("UPDATE labels SET converted=1 WHERE rowid=?",
                      [(i,) for i in done_ids])
    c.close()
    return len(fen_l)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("loop")
    p.add_argument("--out", default="data/derived/gko1_shards")
    p.add_argument("--batch", type=int, default=SHARD_N)
    p.add_argument("--oracle", default=None, help="SQL LIKE filter, e.g. 'sf'")
    p.add_argument("--shards", type=int, default=0, help="stop after N shards (0=forever)")
    p.add_argument("--newest", action="store_true", help="drain newest rowids first (wdl-bearing)")
    p = sub.add_parser("once")
    p.add_argument("--out", default="data/derived/gko1_shards")
    p.add_argument("--batch", type=int, default=SHARD_N)
    p.add_argument("--oracle", default=None, help="SQL LIKE filter, e.g. 'sf'")
    p.add_argument("--shards", type=int, default=0, help="stop after N shards (0=forever)")
    p.add_argument("--newest", action="store_true", help="drain newest rowids first (wdl-bearing)")
    a = ap.parse_args()
    ensure_flag()
    t0 = time.time(); tot = 0; nsh = 0
    while True:
        n = convert_batch(a.out, a.batch, a.oracle, a.newest)
        tot += n
        if n:
            nsh += 1
            print(f"[gko-data] {tot:,} converted ({tot/(time.time()-t0):.0f}/s)"
                  + (f" [{a.oracle}] shard {nsh}"
                     + (f"/{a.shards}" if a.shards else "") if a.oracle else ""),
                  flush=True)
        if a.shards and nsh >= a.shards:
            print(f"[gko-data] reached --shards {a.shards}; stopping", flush=True)
            break
        if a.cmd == "once" or (n == 0 and a.cmd == "once"):
            break
        if n == 0:
            time.sleep(20)


if __name__ == "__main__":
    main()
