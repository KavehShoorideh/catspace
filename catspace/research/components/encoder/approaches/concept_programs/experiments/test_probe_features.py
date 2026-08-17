#!/usr/bin/env python
"""test_probe_features.py -- the Rust probe basis must equal the python-chess oracle EXACTLY.

"then test it on the corpus for accuracy" (Kaveh 2026-08-17). A port that disagrees with its
specification is a bug, not a variant, so this is a differential test with a target of ZERO
mismatches -- and nothing downstream (no concept program, no distillation round) may be claimed
until it passes.

WHAT IS COMPARED, and why it is the whole position. The occupancy columns (`occ_*`, 12 per square)
together with `stm_white`, the four `castle_*` flags and `ep_file` fully determine the position
that `(tok, glob)` encodes. Elementwise equality on those columns IS the board round-trip: if
shakmaty and python-chess disagreed about what position the corpus row means, those columns would
be the first to diverge. The remaining columns then test the probes themselves on a board both
sides agree on, so a failure localises cleanly to either RECONSTRUCTION or a PROBE.

Rows are stratified across game phases, because a bug in castling rights or en passant only shows
up in the opening and a bug in SEE mostly shows up once the board is sparse.

    .venv/bin/python -m ...test_probe_features --n 200000
"""
from __future__ import annotations

import argparse
import time
from collections import Counter

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=20000, help="corpus rows to compare")
    ap.add_argument("--games", type=int, default=4000)
    ap.add_argument("--n-piecedown", type=int, default=45906)
    ap.add_argument("--max-plies", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=2000)
    ap.add_argument("--bench", type=int, default=20000,
                    help="rows for the speed benchmark (Rust only)")
    args = ap.parse_args()

    import chess
    import catspace_concepts as cc
    from catspace.research.tools.training_infra import concepts_reference as R
    from catspace.research.components.encoder.approaches.reach_probability.src import (
        trajectories as T)

    ok = True

    def chk(name, cond, extra=""):
        nonlocal ok
        cond = bool(cond); ok &= cond
        print(f"  {'OK  ' if cond else 'FAIL'} {name}{('  ' + extra) if extra else ''}")

    # ---- 1. names ---------------------------------------------------------------------
    sq_n, gl_n = cc.feature_names()
    chk("column names identical (rust == oracle), per-square", sq_n == R.SQ_NAMES,
        f"{len(sq_n)} dims")
    chk("column names identical (rust == oracle), global", gl_n == R.GLOB_NAMES,
        f"{len(gl_n)} dims")
    if not ok:
        print("\n  names differ -- comparing numbers would be meaningless. Stopping.")
        return 1

    # ---- 2. stratified corpus sample --------------------------------------------------
    t0 = time.time()
    tr = T.build(n_human=0, n_sf=args.games, seed=args.seed, cache=True,
                 max_plies=args.max_plies, n_piecedown=args.n_piecedown)
    n = len(tr.tok)
    ply = tr.ply_of_row()
    rng = np.random.default_rng(args.seed)
    # stratify: opening / middlegame / endgame / deep endgame, equal shares
    bands = [(0, 20), (20, 60), (60, 120), (120, 10**6)]
    per = max(1, args.n // len(bands))
    rows = []
    for lo, hi in bands:
        cand = np.flatnonzero((ply >= lo) & (ply < hi))
        if len(cand):
            rows.append(cand[rng.choice(len(cand), min(per, len(cand)), replace=False)])
    rows = np.sort(np.concatenate(rows))
    print(f"\n  corpus {n:,} rows | comparing {len(rows):,} stratified rows "
          f"[{time.time()-t0:.0f}s]")
    print(f"  ply bands: " + "  ".join(
        f"{lo}-{hi if hi < 10**6 else '+'}: {int(((ply[rows] >= lo) & (ply[rows] < hi)).sum()):,}"
        for lo, hi in bands))

    # ---- 3. elementwise comparison ----------------------------------------------------
    sq_bad = Counter(); gl_bad = Counter()
    worst = []
    n_failed = 0
    t0 = time.time(); t_rust = 0.0; t_py = 0.0
    for a in range(0, len(rows), args.chunk):
        r = rows[a:a + args.chunk]
        tk = np.ascontiguousarray(tr.tok[r]); gb = np.ascontiguousarray(tr.glob[r])
        t1 = time.time(); rs, rg, failed = cc.extract(tk, gb); t_rust += time.time() - t1
        n_failed += failed
        t1 = time.time(); ps, pg = R.extract(tk, gb); t_py += time.time() - t1
        ds = np.abs(rs - ps); dg = np.abs(rg - pg)
        for i in np.flatnonzero(ds.max(axis=(0, 1)) > 1e-4):
            sq_bad[sq_n[i]] += int((ds[:, :, i] > 1e-4).sum())
            if len(worst) < 8:
                rr, ss = np.unravel_index(np.argmax(ds[:, :, i]), ds[:, :, i].shape)
                b = R.board_from_row(tk[rr], gb[rr])
                worst.append((sq_n[i], chess.SQUARE_NAMES[ss], float(rs[rr, ss, i]),
                              float(ps[rr, ss, i]), b.fen()))
        for i in np.flatnonzero(dg.max(axis=0) > 1e-4):
            gl_bad[gl_n[i]] += int((dg[:, i] > 1e-4).sum())
            if len(worst) < 8:
                rr = int(np.argmax(dg[:, i]))
                b = R.board_from_row(tk[rr], gb[rr])
                worst.append((gl_n[i], "-", float(rg[rr, i]), float(pg[rr, i]), b.fen()))
        done = a + len(r)
        if done % (args.chunk * 5) == 0 or done >= len(rows):
            print(f"    {done:>7,}/{len(rows):,}  mismatched columns so far: "
                  f"{len(sq_bad) + len(gl_bad)}  [{time.time()-t0:.0f}s]", flush=True)

    chk("every row built a legal position", n_failed == 0, f"{n_failed} failed")
    # RECONSTRUCTION vs PROBE: which half failed?
    recon_cols = {c for c in sq_bad if c.startswith("occ_")} | \
                 {c for c in gl_bad if c in ("stm_white", "castle_K", "castle_Q", "castle_k",
                                             "castle_q", "ep_file")}
    chk("board round-trip: occupancy / turn / castling / ep identical",
        not recon_cols, f"differing: {sorted(recon_cols)}" if recon_cols else "")
    chk("all per-square probes identical", not sq_bad,
        f"{len(sq_bad)} columns differ" if sq_bad else f"{len(sq_n)} columns clean")
    chk("all global probes identical", not gl_bad,
        f"{len(gl_bad)} columns differ" if gl_bad else f"{len(gl_n)} columns clean")

    if sq_bad or gl_bad:
        print("\n  MISMATCHED COLUMNS (count of differing cells):")
        for k, v in (sq_bad + gl_bad).most_common(20):
            print(f"    {k:24s} {v:,}")
        print("\n  worst cases (column, square, rust, oracle, fen):")
        for nm, s, rv, pv, fen in worst:
            print(f"    {nm:20s} {s:3s} rust {rv:10.3f}  oracle {pv:10.3f}   {fen}")

    # ---- 4. speed ---------------------------------------------------------------------
    nb = min(args.bench, len(rows))
    tk = np.ascontiguousarray(tr.tok[rows[:nb]]); gb = np.ascontiguousarray(tr.glob[rows[:nb]])
    cc.extract(tk[:64], gb[:64])
    t1 = time.time(); cc.extract(tk, gb); el = time.time() - t1
    print(f"\n  SPEED")
    print(f"    rust   {1e6*el/nb:9.1f} us/position  ({nb:,} rows in {el:.2f}s, all cores)")
    print(f"    oracle {1e6*t_py/len(rows):9.1f} us/position  (python-chess, 1 core)")
    print(f"    ratio  {t_py/len(rows)/(el/nb):9.0f}x")
    print(f"    corpus {n:,} rows: rust {el/nb*n:.1f}s vs oracle {t_py/len(rows)*n/3600:.1f}h")
    print(f"    reference: the learned trunk costs 2347 us/position single-threaded")

    print("\nALL PROBE-BASIS TESTS PASSED" if ok else "\nPROBE-BASIS TESTS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
