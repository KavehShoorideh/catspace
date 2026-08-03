#!/usr/bin/env python
"""catspace/research/components/encoder/approaches/concept_quantization/experiments/explain_mate_clusters.py -- Kaveh 2026-07-22: "why are nonsimilar mates
clustered together in the figure?" Decompose WHAT the lichess-B distance between human
mates is actually made of: correlate pairwise B-dist with board factors (piece count,
king squares, material, pattern, elo), and re-test the pattern cohesions CONTROLLING for
piece count -- if ksupport's 0.31 survives among matched-piece-count pairs it is pattern
recognition; if it rises to ~1 it was PHASE (endgame-ness) clustering wearing a pattern
costume.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import chess
import numpy as np
import torch


from scipy.stats import spearmanr

from catspace.research.tools.chess_specific.chessdata.encode import board_from_packed
from catspace.research.components.encoder.approaches.jepa_tokenizer.src.features import feature_planes
from catspace.research.components.encoder.approaches.jepa_tokenizer.src.fb import load_ckpt, pick_device
from catspace.research.components.planner.approaches.endgame_groundtruth.experiments.catalog_mate_directions import harvest
from catspace.io import paths


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--field", default=paths.sep("lichess_gn_iqeqrl_sf.pt"))
    ap.add_argument("--shards", default=paths.shards("lichess_db_standard_rated_2019-01.prefix256mb"))
    ap.add_argument("--n-mates", type=int, default=1500)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time(); dev = pick_device(args.device); rng = np.random.default_rng(args.seed)

    (mpk, mmt), _tails, pats, elos = harvest(args.shards, args.n_mates, 6, rng)
    boards = [board_from_packed(mpk[i], mmt[i]) for i in range(len(mpk))]
    bks = np.array([b.king(chess.BLACK) for b in boards])
    wks = np.array([b.king(chess.WHITE) for b in boards])
    npc = np.array([len(b.piece_map()) for b in boards])
    npawn = np.array([len(b.pieces(chess.PAWN, chess.WHITE)) + len(b.pieces(chess.PAWN, chess.BLACK))
                      for b in boards])
    mats = np.array(["".join(sorted(p.symbol() for p in b.piece_map().values())) for b in boards])
    print(f"[explain] {len(boards)} mates  piece-count: min {npc.min()} med {int(np.median(npc))} "
          f"max {npc.max()}  [{time.time()-t0:.0f}s]", flush=True)

    fb, _ = load_ckpt(Path(args.field), dev); fb.eval()
    B = []
    for s in range(0, len(mpk), 1024):
        pl = feature_planes(mpk[s:s + 1024], mmt[s:s + 1024])
        with torch.no_grad():
            B.append(fb.embed_B(torch.from_numpy(pl).to(dev)).cpu().numpy())
    B = np.concatenate(B)

    # ---- pairwise decomposition: what is B-dist made of?
    ii = rng.integers(0, len(B), 12000); jj = rng.integers(0, len(B), 12000)
    ok = ii != jj; ii, jj = ii[ok], jj[ok]
    bd = np.linalg.norm(B[ii] - B[jj], axis=1)
    factors = {
        "piece-count |diff|": np.abs(npc[ii] - npc[jj]),
        "pawn-count |diff|": np.abs(npawn[ii] - npawn[jj]),
        "black-king sq-dist": np.array([chess.square_distance(a, c) for a, c in zip(bks[ii], bks[jj])]),
        "white-king sq-dist": np.array([chess.square_distance(a, c) for a, c in zip(wks[ii], wks[jj])]),
        "diff-material": (mats[ii] != mats[jj]).astype(float),
        "diff-pattern": (pats[ii] != pats[jj]).astype(float),
        "elo |diff|": np.abs(elos[ii].astype(float) - elos[jj].astype(float)),
    }
    parts = "  ".join(f"{k} {spearmanr(bd, v).correlation:+.2f}" for k, v in factors.items())
    print(f"VERDICT EXPLAIN_B_DIST field={Path(args.field).stem}  spearman(B-dist, factor): {parts}", flush=True)

    # ---- pattern cohesion CONTROLLING piece count (|dNpc|<=2 stratum)
    uniq, cnt = np.unique(pats, return_counts=True)
    big = [u for u, c in zip(uniq, cnt) if c >= 40]

    def cohesion(idx, match_pc):
        ds, rs = [], []
        tries = 0
        while (len(ds) < 3000 or len(rs) < 3000) and tries < 400000:
            tries += 1
            if len(ds) < 3000:
                i, j = idx[rng.integers(len(idx))], idx[rng.integers(len(idx))]
                if i != j and (not match_pc or abs(int(npc[i]) - int(npc[j])) <= 2):
                    ds.append(np.linalg.norm(B[i] - B[j]))
            if len(rs) < 3000:
                a, c = rng.integers(len(B)), rng.integers(len(B))
                if a != c and (not match_pc or abs(int(npc[a]) - int(npc[c])) <= 2):
                    rs.append(np.linalg.norm(B[a] - B[c]))
        return float(np.median(ds) / np.median(rs)) if ds and rs else float("nan")

    rows = []
    for p in big:
        pi = np.flatnonzero(pats == p)
        raw = cohesion(pi, match_pc=False)
        ctl = cohesion(pi, match_pc=True)
        # STRICTEST control: everything inside the pattern's own piece-count stratum
        # (|npc - pattern median| <= 2) -- same-pattern pairs vs random pairs, both in-stratum.
        med = int(np.median(npc[pi]))
        strat = np.flatnonzero(np.abs(npc - med) <= 2)
        pi_s = np.intersect1d(pi, strat)
        if len(pi_s) >= 15 and len(strat) >= 60:
            ds = [np.linalg.norm(B[pi_s[a]] - B[pi_s[b]])
                  for a in range(len(pi_s)) for b in range(a + 1, len(pi_s))]
            ri = strat[rng.integers(0, len(strat), 6000)]; rj = strat[rng.integers(0, len(strat), 6000)]
            m_ = ri != rj
            rs = np.linalg.norm(B[ri[m_]] - B[rj[m_]], axis=1)
            stratum = float(np.median(ds) / np.median(rs))
        else:
            stratum = float("nan")
        rows.append((p, raw, ctl, stratum, med))
    rows.sort(key=lambda r: r[1])
    print("VERDICT EXPLAIN_COHESION  pattern: raw -> diff-matched -> IN-STRATUM  (survives all = real pattern signal)", flush=True)
    for p, raw, ctl, stratum, med in rows:
        print(f"    {p:12s} raw {raw:.2f} -> matched {ctl:.2f} -> stratum {stratum:.2f}   (median pieces {med})", flush=True)
    print(f"[{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
