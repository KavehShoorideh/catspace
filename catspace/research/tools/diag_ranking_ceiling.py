"""What is the BEST top-1 a perfect child-WDL predictor could score on our metric?

qat2's top-1 plateaued at ~0.146 across 8000 steps. Before blaming capacity or the move
concepts, check whether the objective and the metric even agree. The readout is trained to
predict each child's WDL; top-1 is scored against `tgt`, Stockfish's per-move score. If
argmax over child WDL does not coincide with argmax over tgt, then a model that predicted
child WDL PERFECTLY would still not top this metric, and the plateau is a ceiling rather
than a training failure.

Pure label arithmetic -- no model is constructed, so this is safe to run beside a trainer.
"""
import glob, numpy as np
from catspace.research.tools.npz_io import load_npz

W, V, T = [], [], []
for f in sorted(glob.glob("data/derived/gko1_shards/shard_*.npz"))[:60]:
    z = load_npz(f)
    if "wdl" not in z.files:
        continue
    k, wd, tg = z["kind"], z["wdl"].astype(np.float32), z["tgt"]
    fin = np.isfinite(wd).all(-1) & (wd.sum(-1) > 0)
    b = np.where(np.isfinite(tg), tg, -1e9).argmax(1)
    ok = (k == 0) & fin[np.arange(len(k)), b]
    if not ok.any():
        continue
    W.append(wd[ok]); V.append(fin[ok]); T.append(tg[ok])
wd, v, tg = np.concatenate(W), np.concatenate(V), np.concatenate(T)
print(f"  {len(wd):,} positions\n")

rng = np.random.default_rng(0)
tv = np.where(np.isfinite(tg) & v, tg, -np.inf)
uniq = (tv == tv.max(1, keepdims=True)).sum(1) == 1        # the eval scores only these
ok = v.any(1) & uniq
sf_best = tv.argmax(1)

# ORACLE: score children by their TRUE WDL, exactly as the model is trained to do
wl = np.where(v, wd[..., 0] - wd[..., 2], -np.inf)
jit = rng.random(wl.shape) * 1e-9
oracle = np.where(v, wl + jit, -np.inf).argmax(1)
ceiling = float((oracle[ok] == sf_best[ok]).mean())

print(f"  positions scored (unique tgt top): {ok.sum():,} of {len(wd):,} "
      f"({100*ok.mean():.1f}%)\n")
print(f"  CEILING for a PERFECT child-WDL predictor : {ceiling:.3f}")
print(f"  what qat2 actually reached                : ~0.146")
print(f"  'always play slot 0' baseline             : 0.065")
print(f"  chance (1 / 32.6 legal moves)             : 0.031")
print(f"\n  -> qat2 is at {100*0.146/max(ceiling,1e-9):.0f}% of the achievable ceiling.")

# WHY do they disagree? Because child WDL saturates: many children share the top W-L.
top = np.where(ok[:, None], wl, -np.inf).max(1, keepdims=True)
ties = ((wl == top) & v).sum(1)
print(f"\n  WHY the ceiling is not 1.0:")
print(f"    positions where the top child W-L is TIED across >1 move: "
      f"{100*(ties[ok] > 1).mean():.1f}%")
print(f"    median size of that tie: {np.median(ties[ok][ties[ok] > 1]):.0f} moves")
gap = np.sort(np.where(v, wl, -np.inf), 1)[:, ::-1]
d12 = gap[:, 0] - gap[:, 1]
print(f"    best-vs-second-best child W-L gap: median {np.median(d12[ok]):.4f}, "
      f"and is exactly 0 in {100*(d12[ok] == 0).mean():.1f}% of positions")
print(f"\n  CONCLUSION: child WDL is a COARSE ranker. Training on it cannot, even in")
print(f"  principle, reproduce tgt's ordering beyond {ceiling:.3f}.")
