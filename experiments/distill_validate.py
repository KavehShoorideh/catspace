#!/usr/bin/env python
"""experiments/distill_validate.py -- validate the DISTILLATION teacher (Kaveh 2026-07-21): MCTS searches a
6-piece position down to <=5 pieces, where the well-trained field is accurate (spearman(d,DTM)~0.5-0.7), and
backs up a distance-to-mate the RAW 6-piece field cannot (0.21). If the search-backed estimate beats the raw
field, the search extends the accurate region outward and its 6-piece values are a good distillation target
(and the loop can go PAST the tablebase to 7+ pieces).

Compares, on 6-piece KRRvKBP positions, spearman(estimate, DTM) for:
  * field-alone   : d(F(s), MATE_W)                      (the raw extrapolation, ~0.21)
  * MCTS+field    : -root.Q after an MCTS whose leaf value is the field's -d(., MATE_W), mate_stop on
                    (search reaches <=5-piece leaves where the field is good, backs up to the 6-piece root)
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from catspace.research.tools.chess_specific.chessdata.encode import board_from_packed, encode_meta, encode_packed
from catspace.research.components.encoder.approaches.jepa_tokenizer.src.features import feature_planes, omega_ids
from catspace.research.components.encoder.approaches.jepa_tokenizer.src.fb import load_ckpt, pick_device
from catspace.research.components.search.approaches.puct_mcts.src.mcts import MCTS


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--field", default="data/derived/sep/iqe_nucleus_gn.pt")
    ap.add_argument("--dtm-npz", default="data/derived/dtm_endgame.npz")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--nodes", type=int, default=800)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time(); dev = pick_device(args.device); rng = np.random.default_rng(args.seed)
    fb, extra = load_ckpt(Path(args.field), dev); fb.eval()
    zg = torch.load(args.field, map_location="cpu", weights_only=False)["zgoals"]
    MATE_W = (zg["MATE_W"].detach().float() if torch.is_tensor(zg["MATE_W"])
              else torch.tensor(np.asarray(zg["MATE_W"], np.float32))).to(dev)[None, :]
    om = omega_ids(np.array([1800]), np.array([1800]), np.array([300.0]))[0]

    def embF(boards):
        pk = np.stack([encode_packed(b) for b in boards]); mt = np.stack([encode_meta(b) for b in boards])
        with torch.no_grad():
            return fb.embed_F(torch.from_numpy(feature_planes(pk, mt)).to(dev),
                              torch.from_numpy(np.tile(om, (len(boards), 1))).to(dev))

    def reach_fn(boards):                                          # leaf value = -distance to mate (higher=closer)
        with torch.no_grad():
            return -fb.distance_matrix(embF(boards), MATE_W)[:, 0].cpu().numpy()

    mcts = MCTS(reach_fn, max_nodes=args.nodes, mate_stop=True, pw_c=1.5, root_min_visits=10)

    dz = np.load(args.dtm_npz); dtm = np.asarray(dz["dtm"]).astype(float)
    P, M = np.asarray(dz["packed"]), np.asarray(dz["meta"])
    six = [i for i in rng.permutation(len(P)) if dtm[i] > 0 and len(board_from_packed(P[i], M[i]).piece_map()) == 6]
    six = six[:args.n]
    field_d, mcts_d, dtms = [], [], []
    for i in six:
        b = board_from_packed(P[i], M[i])
        if b.is_game_over() or not any(b.legal_moves):
            continue
        field_d.append(float(fb.distance_matrix(embF([b]), MATE_W)[0, 0]))   # raw field distance
        root = mcts.run(b)
        mcts_d.append(-float(root.Q))                                        # -Q ~ distance (higher Q=closer to mate)
        dtms.append(dtm[i])
    dtms = np.array(dtms)
    sf = spearmanr(field_d, dtms).correlation
    sm = spearmanr(mcts_d, dtms).correlation
    print(f"VERDICT DISTILL_VALIDATE field={Path(args.field).stem} n={len(dtms)} nodes={args.nodes} (6-piece KRRvKBP)")
    print(f"  field-alone   spearman(d, DTM) = {sf:+.3f}   (the raw extrapolation)")
    print(f"  MCTS+field    spearman(-Q, DTM) = {sm:+.3f}  ({'BETTER teacher -> distill' if sm > sf + 0.05 else 'no gain'})")
    print(f"[done] {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
