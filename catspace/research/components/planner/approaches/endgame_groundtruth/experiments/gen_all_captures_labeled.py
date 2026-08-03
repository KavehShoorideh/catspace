#!/usr/bin/env python
"""catspace/research/components/planner/approaches/endgame_groundtruth/experiments/gen_all_captures_labeled.py -- expand the hanging-piece probe's data
supply. `sf_label_transitions.py` only labeled the ONE capture a human happened to
play per position (18,925 captures across 70k transitions, only 96 "hanging" after
the committor-swing filter -- too few to trust an MLP readout). Here we enumerate
EVERY legal capture at every position already in the corpus and get Stockfish's own
committor swing across each one -- same ground truth as
catspace/research/components/planner/approaches/endgame_groundtruth/experiments/hanging_piece_probe.py (real win-probability swing across a real
capture), just no longer bottlenecked on a human having played that exact move.

Two phases, both parallel SF workers (depth 12, WDL, matching sf_label_transitions.py
convention): (1) committor_before per unique position (one SF call each, shared
across all of that position's captures), (2) committor_after per (position, capture)
pair (one fresh SF call each -- this is the bulk of the work). Progress printed
per-chunk (not just at executor.map's return) per the "check long runs early" lesson
from the opening-pool run (JOURNAL: ProcessPoolExecutor.map silence gap).
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from catspace.io import paths



def _committor(eng, board, depth):
    info = eng.analyse(board, __import__("chess").engine.Limit(depth=depth))
    w = info["wdl"].white(); tot = max(1, w.wins + w.draws + w.losses)
    return w.wins / tot


def before_worker(task):
    fens, depth = task
    import chess, chess.engine
    eng = chess.engine.SimpleEngine.popen_uci(shutil.which("stockfish") or "/opt/homebrew/bin/stockfish")
    try:
        eng.configure({"UCI_ShowWDL": True})
    except Exception:
        pass
    out = []
    for fen in fens:
        try:
            out.append(_committor(eng, chess.Board(fen), depth))
        except Exception:
            out.append(float("nan"))
    eng.quit()
    return out


def after_worker(task):
    pairs, depth, wid = task    # pairs: list of (fen, move_uci)
    import chess, chess.engine
    eng = chess.engine.SimpleEngine.popen_uci(shutil.which("stockfish") or "/opt/homebrew/bin/stockfish")
    try:
        eng.configure({"UCI_ShowWDL": True})
    except Exception:
        pass
    out = []
    for i, (fen, mv) in enumerate(pairs):
        try:
            b = chess.Board(fen); b.push(chess.Move.from_uci(mv))
            out.append(_committor(eng, b, depth))
        except Exception:
            out.append(float("nan"))
    eng.quit()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labeled", default=paths.derived("transition_data_labeled.npz"))
    ap.add_argument("--out", default=paths.derived("all_captures_labeled.npz"))
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--chunk-size", type=int, default=200)
    ap.add_argument("--limit-pos", type=int, default=0, help="0 = all unique positions (smoke testing)")
    args = ap.parse_args()
    t0 = time.time()
    import chess

    d = np.load(args.labeled, allow_pickle=True)
    fen_all = d["fen"]; game_all = d["game"]
    uniq_fens, first_idx = np.unique(fen_all, return_index=True)
    game_of = {f: int(game_all[i]) for f, i in zip(uniq_fens, first_idx)}
    if args.limit_pos:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(uniq_fens), min(args.limit_pos, len(uniq_fens)), replace=False)
        uniq_fens = uniq_fens[idx]
    print(f"[gen-all-captures] {len(uniq_fens)} unique positions", flush=True)

    W = args.workers or max(1, (os.cpu_count() or 4) - 1)

    print("[gen-all-captures] phase 1: committor_before per position ...", flush=True)
    chunks = np.array_split(uniq_fens, W) if len(uniq_fens) >= W else [uniq_fens]
    with ProcessPoolExecutor(max_workers=W) as ex:
        results = list(ex.map(before_worker, [(list(c), args.depth) for c in chunks]))
    cb_map = {}
    for chunk, res in zip(chunks, results):
        for f, v in zip(chunk, res):
            cb_map[f] = v
    n_ok = sum(1 for v in cb_map.values() if not np.isnan(v))
    print(f"[gen-all-captures] phase 1 done: {n_ok}/{len(uniq_fens)} ok [{time.time() - t0:.0f}s]", flush=True)

    pairs = []   # (fen, move_uci, game)
    for f in uniq_fens:
        if np.isnan(cb_map[f]):
            continue
        b = chess.Board(f)
        for mv in b.legal_moves:
            if b.is_capture(mv):
                pairs.append((f, mv.uci(), game_of[f]))
    print(f"[gen-all-captures] {len(pairs)} (position, capture) pairs to evaluate "
          f"({len(pairs) / max(1, n_ok):.2f} captures/position)", flush=True)

    task_chunks = [pairs[i:i + args.chunk_size] for i in range(0, len(pairs), args.chunk_size)]
    tasks = [([(p[0], p[1]) for p in c], args.depth, i) for i, c in enumerate(task_chunks)]
    print(f"[gen-all-captures] phase 2: {len(tasks)} chunks x ~{args.chunk_size} pairs, {W} workers ...", flush=True)
    ca_all = []
    report_every = max(1, len(tasks) // 30)
    with ProcessPoolExecutor(max_workers=W) as ex:
        for i, res in enumerate(ex.map(after_worker, tasks)):
            ca_all.extend(res)
            if (i + 1) % report_every == 0 or (i + 1) == len(tasks):
                done = len(ca_all)
                print(f"[gen-all-captures] chunk {i + 1}/{len(tasks)} | {done}/{len(pairs)} pairs "
                      f"[{time.time() - t0:.0f}s elapsed]", flush=True)

    fens_out = np.array([p[0] for p in pairs])
    moves_out = np.array([p[1] for p in pairs])
    game_out = np.array([p[2] for p in pairs])
    cb_out = np.array([cb_map[p[0]] for p in pairs])
    ca_out = np.array(ca_all)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, fen=fens_out, move=moves_out, game=game_out,
                         committor_before=cb_out, committor_after=ca_out)
    ok = ~np.isnan(ca_out)
    print(f"=== {args.out}: {ok.sum()}/{len(pairs)} labeled [{time.time() - t0:.0f}s] ===", flush=True)

    white_to_move = np.array([chess.Board(f).turn for f in fens_out])
    gain = np.where(white_to_move, ca_out - cb_out, cb_out - ca_out)
    hang = ok & (gain >= 0.25)
    fair = ok & (np.abs(gain) <= 0.05)
    print(f"  hanging (gain>=0.25): {hang.sum()} | defended-fair (|gain|<=0.05): {fair.sum()}")
    print("DONE gen_all_captures_labeled", flush=True)


if __name__ == "__main__":
    main()
