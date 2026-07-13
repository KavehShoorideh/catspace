#!/usr/bin/env python
"""
experiments/viz/build_fullboard_viewer.py — populate
catspace/viz/templates/fullboard_viewer.html: a background cloud of holdout
positions (projected F, colored by reach-to-MATE_DIFF) plus a handful of full
games (holdout human games, balanced win/loss/draw, and optionally arena PGN
games) with per-ply reach + descriptive/normative expected-score curves.
Boards are NOT pre-rendered to SVG here -- only FEN + the two last-move
square names are stored; the template renders whichever ply is on screen
client-side, on demand (a FEN is ~70 bytes vs. ~31KB for a pre-rendered SVG).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import chess
import numpy as np
import torch

from catspace.data.encode import board_from_packed
from catspace.data.shards import sample_shard_rows
from catspace.io.paths import derived_dir, generated_dir, newest_shard_dir
from catspace.nn.eval_head import load_heads
from catspace.nn.fb import load_ckpt, pick_device
from catspace.viz.build_html import build_html
from catspace.viz.realboard import (embed_positions, fit_projection, games_from_pgn,
                                    infer_san, load_games_from_shard)

COLS = ("packed", "meta", "ply", "clock", "eval_cp", "result", "white_elo", "black_elo", "game_id")


def load_rows(shard_dir: Path, picks: list) -> dict:
    by_file: dict = {}
    for name, row in picks:
        by_file.setdefault(name, []).append(row)
    out: dict = {k: [] for k in COLS}
    for name, rows in sorted(by_file.items()):
        npz = np.load(shard_dir / name)
        idx = np.array(sorted(rows))
        for k in COLS:
            out[k].append(npz[k][idx])
    return {k: np.concatenate(v) for k, v in out.items()}


def score_heads(desc, norm, F_t):
    with torch.no_grad():
        e_d = desc.expected_score(F_t).cpu().numpy()
        e_n = norm.expected_score(F_t).cpu().numpy()
    return e_d, e_n


def build_game_plies(packed, meta, ply, clock, white_elo, black_elo, fb, desc, norm,
                     z, proj, device):
    F, _ = embed_positions(fb, packed, meta, white_elo, black_elo, clock, device)
    F_t = torch.from_numpy(F).to(device)
    reach = F @ z
    e_d, e_n = score_heads(desc, norm, F_t)
    xy = proj.transform(F)
    plies = []
    prev_board = None
    for i in range(len(ply)):
        board = board_from_packed(packed[i], meta[i])
        san, last_from, last_to = None, None, None
        if prev_board is not None:
            san, mv = infer_san(prev_board, packed[i], meta[i])
            if mv is not None:
                last_from, last_to = chess.square_name(mv.from_square), chess.square_name(mv.to_square)
        plies.append(dict(ply=int(ply[i]), san=san, xy=[round(float(xy[i, 0]), 2), round(float(xy[i, 1]), 2)],
                          reach=round(float(reach[i]), 4), e_desc=round(float(e_d[i]), 3),
                          e_norm=round(float(e_n[i]), 3), fen=board.fen(),
                          last_from=last_from, last_to=last_to))
        prev_board = board
    return plies


def result_name(res: int) -> str:
    return {1: "1-0", -1: "0-1", 0: "1/2-1/2"}[res]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shards", default=None)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--heads", default=None, help="default: data/derived/eval_heads.pt")
    ap.add_argument("--pgn", default=None, help="optional arena_real.py --save-pgn output")
    ap.add_argument("--n-games", type=int, default=9)
    ap.add_argument("--n-bg", type=int, default=4000)
    ap.add_argument("--projection", choices=("pca", "tsne"), default="pca")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    shard_dir = Path(args.shards) if args.shards else newest_shard_dir()
    ckpt_path = Path(args.ckpt) if args.ckpt else derived_dir() / "lichess_fb.pt"
    heads_path = Path(args.heads) if args.heads else derived_dir() / "eval_heads.pt"
    device = pick_device(args.device)

    t0 = time.time()
    fb, payload = load_ckpt(ckpt_path, device)
    fb.eval()
    desc, norm, heads_meta = load_heads(heads_path, device)
    desc.eval(); norm.eval()
    assert heads_meta.get("repr", "F") == "F", "fullboard viewer expects an F-repr eval_heads.pt"
    step = payload.get("step", "?")
    zdiff = (payload["zgoals"]["MATE_W"] - payload["zgoals"]["MATE_B"]).numpy().astype(np.float32)
    z = zdiff / np.linalg.norm(zdiff)
    print(f"load: {time.time() - t0:.1f}s  ckpt step={step}  shards={shard_dir.name} device={device}")

    # ---------------------------------------------------------- background cloud
    t0 = time.time()
    picks = sample_shard_rows(shard_dir, args.n_bg, seed=args.seed, holdout_only=True)
    bg = load_rows(shard_dir, picks)
    F_bg, _ = embed_positions(fb, bg["packed"], bg["meta"], bg["white_elo"], bg["black_elo"],
                              bg["clock"], device)
    proj = fit_projection(F_bg, kind=args.projection, seed=args.seed)
    bg_xy = proj.fit_points()          # the actual fit embedding, not an out-of-sample approx
    bg_reach = F_bg @ z
    print(f"background {len(F_bg)} rows embedded+projected: {time.time() - t0:.1f}s")

    # ---------------------------------------------------------- shard games
    t0 = time.time()
    raw_games = load_games_from_shard(shard_dir, args.n_games, seed=args.seed, holdout_only=True,
                                      min_plies=20, max_plies=140)
    games = []
    for g in raw_games:
        plies = build_game_plies(g["packed"], g["meta"], g["ply"], g["clock"],
                                 g["white_elo"], g["black_elo"], fb, desc, norm, z, proj, device)
        res = int(g["result"][0])
        games.append(dict(name=f"gid {int(g['game_id'][0])} ({result_name(res)})",
                          result=result_name(res), plies=plies))
    print(f"{len(games)} shard games embedded: {time.time() - t0:.1f}s")

    # ---------------------------------------------------------- optional PGN games
    if args.pgn:
        t0 = time.time()
        from catspace.data.encode import encode_meta, encode_packed
        from catspace.nn.features import omega_ids
        pgn_games = games_from_pgn(args.pgn)
        for gi, g in enumerate(pgn_games):
            n = len(g["plies"])
            packed = np.stack([encode_packed(b) for b, _, _ in g["plies"]])
            meta = np.stack([encode_meta(b) for b, _, _ in g["plies"]])
            om_row = omega_ids(np.array([1800]), np.array([1800]), np.array([300.0]))[0]
            white_elo = np.full(n, 1800); black_elo = np.full(n, 1800); clock = np.full(n, 300.0)
            # san/last_from/last_to come from build_game_plies's own infer_san pass
            # (exact -- compares consecutive encoded positions), NOT from g["plies"]:
            # that list's san at index i is the UPCOMING move from ply i, not the
            # move that led into ply i, so pairing it with plies[i] would be off-by-one.
            plies = build_game_plies(packed, meta, np.arange(n), clock, white_elo, black_elo,
                                     fb, desc, norm, z, proj, device)
            h = g["headers"]
            games.append(dict(name=f"{h.get('White','?')} vs {h.get('Black','?')} #{gi}",
                              result=h.get("Result", "*"), plies=plies))
        print(f"{len(pgn_games)} pgn games embedded: {time.time() - t0:.1f}s")

    data = dict(meta=dict(title=f"catspace — full-board cone viewer  ·  ckpt step {step}"),
               map=dict(bg=[[round(float(x), 2), round(float(y), 2)] for x, y in bg_xy],
                        reach=[round(float(r), 4) for r in bg_reach]),
               games=games)

    out = Path(args.out) if args.out else generated_dir() / "fullboard-viewer.html"
    template = Path(__file__).resolve().parents[2] / "catspace" / "viz" / "templates" / "fullboard_viewer.html"
    build_html(template, data, out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
