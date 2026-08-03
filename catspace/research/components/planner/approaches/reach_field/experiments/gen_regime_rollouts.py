#!/usr/bin/env python
"""catspace/research/components/planner/approaches/reach_field/experiments/gen_regime_rollouts.py -- SHARED-ANCHOR regime rollouts (Kaveh 2026-07-23:
"positions sampled from human lichess data, followed by sf-vs-sf rollout or random-vs-sf
rollout"). Both regimes continue from the SAME human anchors, so channel support overlap is
~1.0 BY CONSTRUCTION (TRAINING_STANDARDS #15; fixes the 0.13-overlap confound at the root).

Per anchor (sampled from the human shards):
  regime 2  SF vs SF        purposeful play, both sides (depth-limited for throughput)
  regime 3  RANDOM vs SF    the anchor's side-to-move drifts (uniform random), opponent
                            resists with SF -- the drift-under-resistance channel
Output: lichess-format shards (packed/meta/ply/clock/result/elos/game_id) + regime tags +
anchor_idx linking the two regimes' walks to their shared anchor (sidecar anchors.json).
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import chess
import chess.engine
import numpy as np


from catspace.research.tools.chess_specific.chessdata.encode import board_from_packed, encode_meta, encode_packed
from catspace.io import paths

REGIME_SF_SF = 2
REGIME_RAND_SF = 3


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shards", default=paths.shards("lichess_db_standard_rated_2019-01.prefix1gb"))
    ap.add_argument("--n-anchors", type=int, default=4000)
    ap.add_argument("--j", type=int, default=12)
    ap.add_argument("--sf-depth", type=int, default=8)
    ap.add_argument("--engine", default="stockfish")
    ap.add_argument("--out-dir", default=paths.shards("regime_rollouts_v1"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time(); rng = np.random.default_rng(args.seed)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    eng = chess.engine.SimpleEngine.popen_uci(args.engine); eng.configure({"Threads": 1})

    files = sorted(glob.glob(str(Path(args.shards) / "shard_*.npz")))
    z = np.load(files[0])
    N = len(z["game_id"])
    rows = rng.choice(N, size=args.n_anchors * 2, replace=False)
    P, M = z["packed"], z["meta"]
    WE, BE, CK = z["white_elo"], z["black_elo"], z["clock"]

    cols = dict(pk=[], mt=[], ply=[], clk=[], res=[], we=[], be=[], gid=[], reg=[], aidx=[])
    anchors_meta = []
    gid_next = 0
    made = 0

    def sf_move(b):
        info = eng.analyse(b, chess.engine.Limit(depth=args.sf_depth))
        return info["pv"][0] if info.get("pv") else None

    def record(walk, regime, aidx):
        nonlocal gid_next
        r = rows_meta = anchors_meta[aidx]
        for t, w in enumerate(walk):
            cols["pk"].append(encode_packed(w)); cols["mt"].append(encode_meta(w))
            cols["ply"].append(t); cols["clk"].append(r["clock"]); cols["res"].append(0)
            cols["we"].append(r["white_elo"]); cols["be"].append(r["black_elo"])
            cols["gid"].append(gid_next); cols["reg"].append(regime); cols["aidx"].append(aidx)
        gid_next += 1

    for src_row in rows:
        if made >= args.n_anchors:
            break
        b0 = board_from_packed(P[src_row], M[src_row])
        if b0.is_game_over(claim_draw=True) or len(b0.piece_map()) < 5:
            continue
        aidx = len(anchors_meta)
        anchors_meta.append(dict(source_file=Path(files[0]).name, source_row=int(src_row),
                                 fen=b0.fen(), white_elo=int(WE[src_row]),
                                 black_elo=int(BE[src_row]), clock=float(CK[src_row]),
                                 anchor_turn="w" if b0.turn == chess.WHITE else "b"))
        drift_side = b0.turn                     # regime 3: the anchor's mover drifts
        ok_both = True
        walks = {}
        for regime in (REGIME_SF_SF, REGIME_RAND_SF):
            b = b0.copy(stack=False); walk = [b.copy(stack=False)]
            for _t in range(args.j):
                if b.is_game_over(claim_draw=True):
                    break
                if regime == REGIME_SF_SF or b.turn != drift_side:
                    mv = sf_move(b)
                    if mv is None:
                        ok_both = False; break
                else:
                    lm = list(b.legal_moves)
                    mv = lm[int(rng.integers(len(lm)))]
                b.push(mv); walk.append(b.copy(stack=False))
            if not ok_both or len(walk) < 4:
                ok_both = False; break
            walks[regime] = walk
        if not ok_both:
            anchors_meta.pop()
            continue
        for regime, walk in walks.items():
            record(walk, regime, aidx)
        made += 1
        if made % 250 == 0:
            print(f"  {made}/{args.n_anchors} anchors ({gid_next} walks, {len(cols['pk'])} rows)  "
                  f"[{time.time()-t0:.0f}s]", flush=True)

    eng.quit()
    np.savez_compressed(out / "shard_000.npz",
                        packed=np.stack(cols["pk"]), meta=np.stack(cols["mt"]),
                        ply=np.array(cols["ply"], np.int32), clock=np.array(cols["clk"], np.float32),
                        result=np.array(cols["res"], np.int8),
                        white_elo=np.array(cols["we"], np.uint16), black_elo=np.array(cols["be"], np.uint16),
                        game_id=np.array(cols["gid"], np.uint32),
                        regime=np.array(cols["reg"], np.int8),
                        anchor_idx=np.array(cols["aidx"], np.int32))
    (out / "anchors.json").write_text(json.dumps(anchors_meta))
    print(f"VERDICT REGIME_ROLLOUTS anchors={made} walks={gid_next} rows={len(cols['pk'])} "
          f"j={args.j} sf_depth={args.sf_depth} regimes=[2:sf_sf, 3:rand_vs_sf] "
          f"-> {out}  [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
