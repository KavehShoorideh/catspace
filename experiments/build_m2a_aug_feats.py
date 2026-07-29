#!/usr/bin/env python
"""Production-recipe human-choice features for the M2a labeled set (table/gates assignment must
match the v3 codebook recipe: top-16 Maia-2 candidates AT THE POSITION'S OWN RATINGS,
renormalized entropy/top-p/gap + win_prob)."""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

def main():
    t0 = time.time()
    d = dict(np.load('data/derived/transition_data_labeled.npz', allow_pickle=True))
    fen, em, eo = d['fen'], d['elo_mover'], d['elo_opp']
    from maia2 import model as maia_model, inference
    from catspace.train.scaffold import resolve_device
    dev = resolve_device('auto')
    inference.prepare()
    maia = maia_model.from_pretrained(type='rapid', device=str(dev))
    feats = np.zeros((len(fen), 4), np.float32)
    B = 2048
    for i in range(0, len(fen), B):
        df = pd.DataFrame({'fen': fen[i:i+B], 'move': ['0000']*len(fen[i:i+B]),
                           'elo_self': em[i:i+B].astype(int), 'elo_oppo': eo[i:i+B].astype(int)})
        df, _ = inference.inference_batch(df, maia, verbose=False, batch_size=1024, num_workers=0)
        for j, (probs, wp) in enumerate(zip(df['move_probs'], df['win_probs'])):
            p = np.array(sorted(probs.values(), reverse=True)[:16], dtype=np.float64)
            p = p / p.sum()
            e = -(p*np.log(p+1e-12)).sum()
            feats[i+j] = [e, p[0], p[0]-(p[1] if len(p) > 1 else 0.0), wp]
        if i % 16384 == 0:
            print(f"  {i:,}/{len(fen):,} [{time.time()-t0:.0f}s]", flush=True)
    np.savez_compressed('data/derived/m2a_aug_feats.npz', feats=feats)
    print(f"wrote data/derived/m2a_aug_feats.npz [{time.time()-t0:.0f}s]")

if __name__ == "__main__":
    main()
