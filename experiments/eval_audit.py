#!/usr/bin/env python
"""
experiments/eval_audit.py — the agreed shallow-label-independent audit: on a
held-out sample, compare every eval signal against DEEP Stockfish (default
depth 22; --depth 30 is the full audit, same code, ~minutes more), so heads
trained on shallow labels are never graded by their own teacher.

Columns compared against deep-SF expected score (spearman):
  desc_head    descriptive eval head (game-results probe)
  norm_head    normative eval head (lichess-eval probe)
  lichess_eval winprob(eval_cp) where annotated (the shallow teacher itself)
  reach_mate   F(s) @ zMATE_W (the raw cone signal, no head at all)

lc0 WDL column: pass --lc0-cmd if you have lc0+weights installed; otherwise
it is SKIPPED with a note (not installed on this machine).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import chess
import chess.engine
import numpy as np
import torch
from scipy.stats import spearmanr

from catspace.data.encode import board_from_packed
from catspace.io.paths import derived_dir, newest_shard_dir
from catspace.nn.eval_head import load_heads
from catspace.nn.fb import load_ckpt, pick_device
from catspace.nn.features import feature_planes, omega_ids, winprob_cp


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shards", default=None)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--heads", default=None)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--depth", type=int, default=22, help="deep-SF depth (30 = full audit)")
    ap.add_argument("--engine", default="stockfish")
    ap.add_argument("--lc0-cmd", default=None, help="lc0 command if installed (optional)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    shard_dir = Path(args.shards) if args.shards else newest_shard_dir()
    device = pick_device(args.device)
    fb, payload = load_ckpt(Path(args.ckpt) if args.ckpt else derived_dir() / "lichess_fb.pt", device)
    fb.eval()
    desc, norm, _ = load_heads(Path(args.heads) if args.heads else derived_dir() / "eval_heads.pt", device)
    desc.eval(); norm.eval()
    z_mate = payload["zgoals"]["MATE_W"].to(device)

    from catspace.data.shards import sample_shard_rows
    picks = sample_shard_rows(shard_dir, args.n * 2, args.seed, holdout_only=True)

    boards, planes_rows, om_rows, lichess_wp = [], [], [], []
    by_shard: dict[str, list[int]] = {}
    for s, r in picks:
        by_shard.setdefault(s, []).append(r)
    for shard_name, rows in sorted(by_shard.items()):
        npz = np.load(shard_dir / shard_name)
        data = {k: npz[k] for k in npz.files}
        for row in rows:
            board = board_from_packed(data["packed"][row], data["meta"][row])
            if board.is_game_over():
                continue
            boards.append(board)
            planes_rows.append((data["packed"][row], data["meta"][row]))
            om_rows.append((data["white_elo"][row], data["black_elo"][row], data["clock"][row]))
            lichess_wp.append(winprob_cp(np.array([data["eval_cp"][row]]))[0])
            if len(boards) >= args.n:
                break
        if len(boards) >= args.n:
            break

    packed = np.stack([p for p, _ in planes_rows]); meta = np.stack([m for _, m in planes_rows])
    planes = torch.from_numpy(feature_planes(packed, meta)).to(device)
    om = torch.from_numpy(omega_ids(np.array([o[0] for o in om_rows]),
                                    np.array([o[1] for o in om_rows]),
                                    np.array([o[2] for o in om_rows]))).to(device)
    with torch.no_grad():
        f = fb.embed_F(planes, om)
        e_desc = desc.expected_score(f).cpu().numpy()
        e_norm = norm.expected_score(f).cpu().numpy()
        reach = (f @ z_mate).cpu().numpy()

    print(f"deep-SF labeling {len(boards)} holdout positions at depth {args.depth}...")
    engine = chess.engine.SimpleEngine.popen_uci(args.engine)
    engine.configure({"UCI_ShowWDL": True})
    deep = []
    try:
        for i, board in enumerate(boards):
            info = engine.analyse(board, chess.engine.Limit(depth=args.depth))
            wdl = info.get("wdl")
            if wdl is not None:
                w, d, l = wdl.white()
                deep.append((w + 0.5 * d) / 1000.0)
            else:
                deep.append(winprob_cp(np.array([info["score"].white().score(mate_score=3200)]))[0])
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(boards)}", flush=True)
    finally:
        engine.quit()
    deep = np.array(deep)

    def row(name, vals, mask=None):
        m = np.isfinite(vals) if mask is None else mask
        rho = spearmanr(vals[m], deep[m]).statistic
        print(f"  {name:>14}: spearman {rho:+.3f}  (n={int(m.sum())})")

    print(f"\nVERDICT vs deep SF (depth {args.depth}):")
    row("desc_head", e_desc)
    row("norm_head", e_norm)
    row("lichess_eval", np.array(lichess_wp))
    row("reach_mate", reach)
    if args.lc0_cmd:
        print("  lc0 column requested -- not implemented beyond SF parity here; run with lc0 as --engine instead")
    else:
        print("  lc0: SKIPPED (no --lc0-cmd; lc0+weights not installed)")


if __name__ == "__main__":
    main()
