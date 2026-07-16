#!/usr/bin/env python
"""
experiments/viz/build_mate_attempt_viewer.py — WHERE did a mate attempt go in
embedding space? (Kaveh 2026-07-15: "visualize a couple of mate attempts to
see what happened and where it went in the space")

Plays fixed-start test games with the incumbent (cached MCTS vs tb-optimal
defender) until it has --n-wins conversions and --n-fails failures, then emits
a single-file HTML viewer: per game a board scrubber synced to
  (a) the game's path through a UMAP of the certainty field (background =
      R2 table states colored by rollout P-hat, red 0 -> green 1),
  (b) strip charts of learned d(F(s), zMATE_W) and the win-prob head P(win)
      along the game (the field's own story of the attempt).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import chess
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from catspace.viz.build_html import build_html
from catspace.viz.realboard import board_svg
from experiments.value_fixed_point import TB, tb_best_move


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="data/derived/sep/cert_base_full.pt")
    ap.add_argument("--phead", default="data/derived/sep/cert_base_full_phead.pt")
    ap.add_argument("--table", default="artifacts/experiments/certainty_table_r2_K16.json")
    ap.add_argument("--fixed-set", default="artifacts/experiments/krrkbp_fixed_test_n200.json")
    ap.add_argument("--nodes", type=int, default=800)
    ap.add_argument("--n-wins", type=int, default=2)
    ap.add_argument("--n-fails", type=int, default=2)
    ap.add_argument("--max-plies", type=int, default=120)
    ap.add_argument("--bg-sample", type=int, default=3000)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--whead", default=None,
                    help="committor W-head (*_whead.pt): policy uses the committor "
                         "readout; strips show d_W and calibrated P_W (iso) instead "
                         "of pole distance + P-head")
    ap.add_argument("--out", default="artifacts/generated/mate_attempts.html")
    args = ap.parse_args()

    import torch
    from umap import UMAP
    from catspace.nn.eval_head import EvalHead
    from catspace.nn.fb import load_ckpt, pick_device
    from catspace.nn.mcts import FBMCTSPolicy
    from catspace.data.encode import encode_meta, encode_packed
    from catspace.nn.features import feature_planes, omega_ids

    dev = pick_device(args.device)
    fb, pay = load_ckpt(Path(args.ckpt), dev)
    fb.eval()
    zW = pay["zgoals"]["MATE_W"]
    zW = zW.to(dev).float() if torch.is_tensor(zW) else torch.as_tensor(
        np.asarray(zW), dtype=torch.float32, device=dev)
    whead, iso = None, None
    if args.whead:
        wp = torch.load(args.whead, map_location=dev, weights_only=False)
        whead = torch.nn.Sequential(torch.nn.Linear(wp["d_in"], 128), torch.nn.ReLU(),
                                    torch.nn.Linear(128, 1), torch.nn.Softplus()).to(dev)
        whead.load_state_dict(wp["state"]); whead.eval()
        iso = wp.get("iso")
    else:
        hp = torch.load(args.phead, map_location=dev, weights_only=False)
        phead = EvalHead(d_in=hp["d_in"]).to(dev)
        phead.load_state_dict(hp["state"])
        phead.eval()
    om_row = omega_ids(np.array([1800]), np.array([1800]), np.array([np.nan]))[0]

    def embed(boards):
        packed = np.stack([encode_packed(b) for b in boards])
        meta = np.stack([encode_meta(b) for b in boards])
        with torch.no_grad():
            pl = torch.from_numpy(feature_planes(packed, meta)).to(dev)
            om = torch.from_numpy(np.tile(om_row, (len(boards), 1))).to(dev)
            f = fb.embed_F(pl, om)
            if whead is not None:
                d = whead(f).squeeze(-1).cpu().numpy()
                pw = (np.interp(d, iso["x"], iso["y"]) if iso
                      else np.exp(-np.maximum(d, 1e-4)))
            else:
                d = fb.distance_matrix(f, zW[None, :])[:, 0].cpu().numpy()
                pw = torch.softmax(phead(f), dim=1)[:, 0].cpu().numpy()
        return f.cpu().numpy(), d, pw

    rows = json.loads(Path(args.table).read_text())["rows"]
    rng = np.random.default_rng(0)
    bg_rows = [rows[i] for i in rng.choice(len(rows), size=min(args.bg_sample, len(rows)),
                                           replace=False)]
    F_bg, _, _ = embed([chess.Board(r["fen"]) for r in bg_rows])
    print(f"fitting UMAP on {len(bg_rows)} table states...")
    um = UMAP(n_neighbors=30, min_dist=0.3, random_state=0).fit(F_bg)
    bg_xy = um.embedding_

    tb = TB("data/syzygy")
    pol = FBMCTSPolicy(fb, zW, max_nodes=args.nodes, device=dev, committor_head=whead)
    starts = json.loads(Path(args.fixed_set).read_text())["fens"]
    games, wins, fails = [], 0, 0
    for si, fen in enumerate(starts):
        if wins >= args.n_wins and fails >= args.n_fails:
            break
        b = chess.Board(fen)
        boards, sans, seen = [b.copy(stack=False)], [""], set()
        g_rng = np.random.default_rng([0, si])
        for _ in range(args.max_plies):
            if b.is_game_over(claim_draw=True):
                break
            m = pol.move(b, g_rng) if b.turn == chess.WHITE else tb_best_move(b, tb, seen)
            if b.turn == chess.BLACK:
                seen.add(b.board_fen())
            if m is None:
                break
            sans.append(b.san(m))
            b.push(m)
            boards.append(b.copy(stack=False))
        out = b.outcome(claim_draw=True)
        won = bool(out and out.winner == chess.WHITE)
        if won and wins >= args.n_wins:
            continue
        if not won and fails >= args.n_fails:
            continue
        wins, fails = wins + int(won), fails + int(not won)
        F, d, pw = embed(boards)
        xy = um.transform(F)
        # named-stage timeline (EVAL-ONLY instruments, per the design contract):
        # first ply each concept fires, appended to the game label
        from experiments.stage_checkers import annotate_game
        ann = annotate_game([bd.fen() for bd in boards])
        stages = "  ".join(f"{k.replace('capture_','x').replace('king_','')}@{v}"
                           for k, v in ann.items()
                           if v is not None and k in ("capture_pawn", "capture_bishop",
                                                      "king_corner", "mate_edge",
                                                      "mate_midboard"))
        games.append(dict(
            label=(f"start {si}: "
                   f"{'MATE in ' + str(b.ply()) + ' plies' if won else 'FAILED (' + (out.termination.name if out else 'cutoff') + ')'}"
                   + (f"  [{stages}]" if stages else "")),
            won=won,
            moves=[dict(san=s, svg=board_svg(bd, lastmove=bd.peek() if bd.move_stack else None, size=360),
                        d=float(dd), p=float(pp), x=float(x), y=float(y))
                   for s, bd, dd, pp, (x, y) in zip(sans, boards, d, pw, xy)]))
        print(f"  game {len(games)}: {games[-1]['label']} ({len(boards)} positions)")
    tb.close()

    data = dict(bg=dict(x=bg_xy[:, 0].tolist(), y=bg_xy[:, 1].tolist(),
                        p=[r["p_hat"] for r in bg_rows]),
                games=games,
                meta=dict(ckpt=args.ckpt, nodes=args.nodes,
                          note="background = R2 certainty table (own-play P-hat), "
                               "path = this game's F trajectory (UMAP transform)"))
    build_html(Path("catspace/viz/templates/mate_attempts.html"), data, Path(args.out))
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
