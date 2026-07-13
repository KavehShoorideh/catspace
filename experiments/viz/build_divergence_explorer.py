#!/usr/bin/env python
"""
experiments/viz/build_divergence_explorer.py — populate
catspace/viz/templates/divergence_explorer.html: E_desc vs E_norm scatter
over annotated holdout positions (finite eval_cp only), with per-position
divergence (E_desc - E_norm, +: humans overperform the engine) and a
sortable top-divergence table. Boards render client-side from FEN.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from catspace.data.encode import board_from_packed
from catspace.data.shards import sample_shard_rows
from catspace.io.paths import derived_dir, generated_dir, newest_shard_dir
from catspace.nn.eval_head import load_heads
from catspace.nn.features import elo_bin, winprob_cp
from catspace.nn.fb import load_ckpt, pick_device
from catspace.viz.build_html import build_html
from catspace.viz.realboard import embed_positions

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


def sample_annotated(shard_dir: Path, n: int, seed: int, oversample: int = 12) -> dict:
    """eval_cp annotation rate is ~8-10%, so oversample then filter to finite."""
    picks = sample_shard_rows(shard_dir, n * oversample, seed=seed, holdout_only=True)
    data = load_rows(shard_dir, picks)
    fin = np.isfinite(data["eval_cp"])
    data = {k: v[fin] for k, v in data.items()}
    if len(data["ply"]) > n:
        idx = np.random.default_rng(seed).choice(len(data["ply"]), n, replace=False)
        data = {k: v[idx] for k, v in data.items()}
    return data


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shards", default=None)
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--heads", default=None, help="default: data/derived/eval_heads.pt (F-repr)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--top-n", type=int, default=80)
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
    assert heads_meta.get("repr", "F") == "F", "divergence explorer expects an F-repr eval_heads.pt"
    step = payload.get("step", "?")
    print(f"load: {time.time() - t0:.1f}s  ckpt step={step}  device={device}")

    t0 = time.time()
    data = sample_annotated(shard_dir, args.n, args.seed)
    n = len(data["ply"])
    print(f"sampled {n} annotated holdout rows: {time.time() - t0:.1f}s")

    t0 = time.time()
    F, _ = embed_positions(fb, data["packed"], data["meta"], data["white_elo"], data["black_elo"],
                           data["clock"], device)
    F_t = torch.from_numpy(F).to(device)
    with torch.no_grad():
        e_desc = desc.expected_score(F_t).cpu().numpy()
        e_norm = norm.expected_score(F_t).cpu().numpy()
    sf = winprob_cp(data["eval_cp"])
    div = e_desc - e_norm
    print(f"embed+score {n} rows: {time.time() - t0:.1f}s")

    fens = [board_from_packed(data["packed"][i], data["meta"][i]).fen() for i in range(n)]
    welo_bin = elo_bin(data["white_elo"])

    points = dict(fen=fens, ply=[int(p) for p in data["ply"]],
                 white_elo_bin=[int(b) for b in welo_bin],
                 e_desc=[round(float(v), 4) for v in e_desc],
                 e_norm=[round(float(v), 4) for v in e_norm],
                 sf=[round(float(v), 4) for v in sf],
                 div=[round(float(v), 4) for v in div])
    top_indices = [int(i) for i in np.argsort(-np.abs(div))[: args.top_n]]

    data_out = dict(meta=dict(title=f"catspace — divergence explorer  ·  ckpt step {step}  ·  {n} annotated holdout positions"),
                    points=points, top_indices=top_indices)

    out = Path(args.out) if args.out else generated_dir() / "divergence-explorer.html"
    template = Path(__file__).resolve().parents[2] / "catspace" / "viz" / "templates" / "divergence_explorer.html"
    build_html(template, data_out, out)
    print(f"wrote {out}  ({out.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
