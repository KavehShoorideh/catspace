#!/usr/bin/env python
"""experiments/build_home_book.py -- the HOME BOOK (Kaveh 2026-07-30): per-cell measurement of
where OUR ENGINE specifically is strong, from our own accumulated games, used as navigation
targets ("prep": steer the game onto home turf, especially from the opening).

Sources: every PGN of our engine family vs Maia (side named catspace/field_v3 in headers).
Per OUR move: committor before/after (v3 field, instrument-grade; SF-refereed upgrade recorded),
augmented-cell assignment of the position. Per cell x phase(opening ply<=24 / later):
  strength = shrunken( -mean committor loss of OUR moves )   (how little we bleed there)
  convert  = shrunken( mean game score | we visited )        (how often visits become points)
  home     = zscore(strength) + zscore(convert), squashed to [0,1]
Output: data/derived/reach/home_book_v1.npz (home (NC,2 phases), counts, provenance) --
consumed by SubgoalRanker (rank(..., home) component) and the generator's kappa mix.
"""
from __future__ import annotations

import argparse
import glob
import sys
import time
from pathlib import Path

import chess
import chess.pgn
import io
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

OUR_NAMES = ("catspace", "field_v3")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pgns", nargs="+", default=sorted(
        glob.glob("artifacts/experiments/m4_steering_*.pgn")
        + glob.glob("artifacts/experiments/field_v3_*maia*.pgn")
        + glob.glob("/Users/kav/.claude/jobs/20b9956a/tmp/m4e_*.pgn")))
    ap.add_argument("--ckpt", default="artifacts/experiments/field_fullgame_v3_final.pt")
    ap.add_argument("--field", default="artifacts/experiments/reach_v3_full_latest.pt")
    ap.add_argument("--reach", default="data/derived/reach/reach_v3.npz")
    ap.add_argument("--table", default="data/derived/reach/region_table_v3.npz")
    ap.add_argument("--our-elo", type=float, default=1800.0)
    ap.add_argument("--opp-elo", type=float, default=1100.0)
    ap.add_argument("--opening-ply", type=int, default=24)
    ap.add_argument("--shrink", type=float, default=15.0)
    ap.add_argument("--out", default="data/derived/reach/home_book_v1.npz")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    t0 = time.time()
    from catspace.research.tools.training_infra.train.scaffold import resolve_device
    dev = resolve_device(args.device)
    from experiments.play_vs_maia import CommittorGreedy
    from experiments.m4_play_steering import maia_feats
    from catspace.research.components.encoder.approaches.reachability_field.src.field import ReachabilityField
    from catspace.research.components.planner.approaches.atlas_region_stats.src.ranker import SubgoalRanker
    from lczerolens import LczeroBoard
    from maia2 import model as maia_model, inference as m2_inf
    m2_inf.prepare()
    m2 = maia_model.from_pretrained(type="rapid", device=str(dev))
    val = CommittorGreedy(args.ckpt, dev)
    rf = ReachabilityField(device=str(dev))
    rk = SubgoalRanker(args.field, args.reach, args.table, device=str(dev))

    rows = []                                   # (fen_before, our_white, ply, score, planes...)
    n_games = 0
    for path in args.pgns:
        text = Path(path).read_text()
        for chunk in text.split("\n\n\n"):
            g = chess.pgn.read_game(io.StringIO(chunk))
            if g is None:
                continue
            w, b = g.headers.get("White", ""), g.headers.get("Black", "")
            our_white = any(n in w for n in OUR_NAMES)
            our_black = any(n in b for n in OUR_NAMES)
            if not (our_white ^ our_black):
                continue
            res = g.headers.get("Result", "*")
            score = {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}.get(res)
            if score is None:
                continue
            if not our_white:
                score = 1.0 - score
            board = LczeroBoard(); n_games += 1
            for mv in g.mainline_moves():
                if board.turn == (chess.WHITE if our_white else chess.BLACK):
                    rows.append((board.fen(), board.to_input_tensor().float().numpy(),
                                 our_white, board.ply(), score, mv.uci()))
                try:
                    board.push(mv)
                except Exception:
                    break
    print(f"{n_games} of-our-engine games -> {len(rows):,} our-move observations "
          f"[{time.time()-t0:.0f}s]")

    fens = [r[0] for r in rows]
    planes = [r[1] for r in rows]
    our_white = np.array([r[2] for r in rows]); ply = np.array([r[3] for r in rows])
    score = np.array([r[4] for r in rows], np.float32)
    # committor before + after our move (batched)
    planes_after = []
    for (fen, _, ow, _, _, uci) in rows:
        b = LczeroBoard(fen); b.push(chess.Move.from_uci(uci))
        planes_after.append(b.to_input_tensor().float().numpy())
    cb = val._committor(planes); ca = val._committor(planes_after)
    cb = np.where(our_white, cb, 1 - cb); ca = np.where(our_white, ca, 1 - ca)
    loss = np.maximum(cb - ca, 0.0)             # our committor bleed per move
    # augmented cell of each decision position
    phis = np.concatenate([rf.phi([LczeroBoard(f) for f in fens[i:i+512]]).cpu().numpy()
                           for i in range(0, len(fens), 512)])
    feats = maia_feats(m2, m2_inf, fens, args.our_elo, args.opp_elo)
    reg = rk.assign(phis, feats)
    cell = reg * rk.n_cband + np.digitize(cb, [0.35, 0.65])
    NC = len(rk.flux)

    phase = (ply > args.opening_ply).astype(int)             # 0 = opening/book, 1 = later
    home = np.zeros((NC, 2), np.float32); counts = np.zeros((NC, 2), np.int32)
    for ph in (0, 1):
        m = phase == ph
        cnt = np.bincount(cell[m], minlength=NC)
        counts[:, ph] = cnt
        s_loss = np.bincount(cell[m], weights=loss[m], minlength=NC)
        s_conv = np.bincount(cell[m], weights=score[m], minlength=NC)
        w = cnt / (cnt + args.shrink)
        mloss = np.where(cnt > 0, s_loss / np.maximum(cnt, 1), loss[m].mean())
        mconv = np.where(cnt > 0, s_conv / np.maximum(cnt, 1), score[m].mean())
        strength = w * (-(mloss - loss[m].mean())) + 0.0
        convert = w * (mconv - score[m].mean())
        zs = lambda x: (x - x.mean()) / (x.std() + 1e-9)
        home[:, ph] = 1 / (1 + np.exp(-(zs(strength) + zs(convert))))
    print(f"AUDIT: cells with opening data {int((counts[:,0]>0).sum())}/{NC} | "
          f"our mean loss/move {loss.mean():.4f} | mean score {score.mean():.3f} | "
          f"home[open] spread {home[:,0].min():.2f}-{home[:,0].max():.2f}")
    np.savez_compressed(args.out, home=home, counts=counts,
                        opening_ply=args.opening_ply, sources=np.array(args.pgns),
                        n_games=n_games, meta_note="instrument-grade committor (v3 field); "
                        "SF-refereed upgrade recorded")
    print(f"wrote {args.out} [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
