#!/usr/bin/env python
"""Engine-cohort move-selection rows from the banked shared-anchor rollouts: the flavored-
energy model's identification data (engines pin the value component; the capability spread
identifies the flavors). Same row format as build_move_selection; cohort = regime-mapped."""
from __future__ import annotations
import glob, sys, time
from pathlib import Path
import chess
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from catspace.research.tools.chess_specific.chessdata.encode import board_from_packed, encode_packed
from catspace.research.components.encoder.approaches.jepa_tokenizer.src.opponent import COHORT_ENGINE

def main():
    t0 = time.time(); rng = np.random.default_rng(0)
    L = 80
    PK, MT, F, T, PC, CT, NM, PL, CO = [], [], [], [], [], [], [], [], []
    made = skipped = 0
    files = sorted(glob.glob("data/shards/regime_rollouts_v1/shard_*.npz")); rng.shuffle(files)
    TARGET = 250_000
    for path in files:
        if made >= TARGET: break
        z = np.load(path)
        gid, reg, pk, mt = z["game_id"], z["regime"], z["packed"], z["meta"]
        for i in rng.permutation(len(gid) - 1):
            if made >= TARGET: break
            if gid[i] != gid[i + 1] or int(reg[i]) not in COHORT_ENGINE: continue
            if int(reg[i]) == 3: continue                    # rand_vs_sf mixes two policies per game; skip v1
            b = board_from_packed(pk[i], mt[i])
            moves = list(b.legal_moves)
            if not 1 <= len(moves) <= L: skipped += 1; continue
            nxt = pk[i + 1]; played = -1
            f_ = np.zeros(L, np.uint8); t_ = np.zeros(L, np.uint8)
            p_ = np.zeros(L, np.uint8); c_ = np.zeros(L, np.uint8)
            for j, m in enumerate(moves):
                f_[j], t_[j] = m.from_square, m.to_square
                p_[j] = b.piece_type_at(m.from_square) or 0
                cap = b.piece_type_at(m.to_square)
                c_[j] = cap or (1 if b.is_en_passant(m) else 0)
                if played < 0:
                    cb = b.copy(stack=False); cb.push(m)
                    if np.array_equal(encode_packed(cb), nxt): played = j
            if played < 0: skipped += 1; continue
            PK.append(pk[i]); MT.append(mt[i]); F.append(f_); T.append(t_); PC.append(p_); CT.append(c_)
            NM.append(len(moves)); PL.append(played); CO.append(COHORT_ENGINE[int(reg[i])])
            made += 1
            if made % 50_000 == 0:
                print(f"  {made:,}/{TARGET:,}  [{time.time()-t0:.0f}s]", flush=True)
    np.savez_compressed("data/derived/move_selection_engines_v1.npz",
        packed=np.stack(PK), meta=np.stack(MT), mv_from=np.stack(F), mv_to=np.stack(T),
        mv_piece=np.stack(PC), mv_capt=np.stack(CT), n_moves=np.array(NM, np.uint8),
        played=np.array(PL, np.uint8), cohort=np.array(CO, np.uint8))
    from collections import Counter
    print(f"VERDICT MOVE_SELECTION_ENGINES rows={made:,} skipped={skipped:,} "
          f"cohorts={dict(Counter(CO))}  [{time.time()-t0:.0f}s]", flush=True)

if __name__ == "__main__":
    main()
