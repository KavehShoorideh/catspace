#!/usr/bin/env python
"""Production-recipe human-choice features for the M2a labeled set (table/gates assignment must
match the v3 codebook recipe: top-16 Maia-2 candidates AT THE POSITION'S OWN RATINGS,
renormalized entropy/top-p/gap + win_prob)."""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

NEUTRAL = np.array([1.5, 0.5, 0.3, 0.5], np.float32)   # ~population means (m4 recipe)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--labeled', default='data/derived/transition_data_labeled.npz')
    ap.add_argument('--out', default='data/derived/m2a_aug_feats.npz')
    args = ap.parse_args()
    t0 = time.time()
    d = dict(np.load(args.labeled, allow_pickle=True))
    fen, em, eo = d['fen'], d['elo_mover'], d['elo_opp']
    from maia2 import model as maia_model, inference
    from catspace.train.scaffold import resolve_device
    dev = resolve_device('auto')
    inference.prepare()
    maia = maia_model.from_pretrained(type='rapid', device=str(dev))
    feats = np.tile(NEUTRAL, (len(fen), 1))
    B = 2048
    poisoned = 0

    def run(fs, es, os_):
        df = pd.DataFrame({'fen': list(fs), 'move': ['0000']*len(fs),
                           'elo_self': np.asarray(es, int), 'elo_oppo': np.asarray(os_, int)})
        df, _ = inference.inference_batch(df, maia, verbose=False, batch_size=1024, num_workers=0)
        r = np.zeros((len(fs), 4), np.float32)
        for j, (probs, wp) in enumerate(zip(df['move_probs'], df['win_probs'])):
            p = np.array(sorted(probs.values(), reverse=True)[:16], dtype=np.float64)
            p = p / p.sum()
            r[j] = [-(p*np.log(p+1e-12)).sum(), p[0],
                    p[0]-(p[1] if len(p) > 1 else 0.0), wp]
        return r

    for i in range(0, len(fen), B):
        sl = slice(i, i+B)
        try:
            feats[sl] = run(fen[sl], em[sl], eo[sl])
        except Exception:
            # POISON GUARD (the m4f killer): maia2 preprocessing IndexErrors on
            # dict-gap positions. Per-row retry; poisoned rows keep NEUTRAL.
            for k in range(i, min(i+B, len(fen))):
                try:
                    feats[k] = run(fen[k:k+1], em[k:k+1], eo[k:k+1])[0]
                except Exception:
                    poisoned += 1
        if i % 16384 == 0:
            print(f"  {i:,}/{len(fen):,} [{time.time()-t0:.0f}s]", flush=True)
    np.savez_compressed(args.out, feats=feats)
    print(f"wrote {args.out} [{time.time()-t0:.0f}s] | poisoned rows kept NEUTRAL: {poisoned}")

if __name__ == "__main__":
    main()
