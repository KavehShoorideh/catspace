"""How BAD are our misrankings? Regret in win-probability, not top-1 accuracy.

Kaveh 2026-08-26: "if we're misranking two things that are very close to each other, I
think it's okay... penalize based on centipawn loss compared to the best move. So even if
we pick number two and it is really close to number one, the penalty is small."

That is REGRET, and it is the standard chess measure (average centipawn loss). Top-1
accuracy cannot tell a near-tie from a blunder; regret can. We do not have centipawns in
the shards, but we have Stockfish's win probability per move, so regret is measured in
win-probability lost:

    regret(m) = wp(best move) - wp(chosen move)

Also settles Kaveh's challenge -- "I don't understand how a single scalar can be a higher
resolution signal than WDL" -- by comparing SCALARISATIONS of the same WDL triple rather
than blaming WDL itself.
"""
import glob, numpy as np
from catspace.research.tools.npz_io import load_npz

V, T, W = [], [], []
for f in sorted(glob.glob("data/derived/gko1_shards/shard_*.npz"))[:40]:
    z = load_npz(f)
    if "wdl" not in z.files:
        continue
    k, wd, tg = z["kind"], z["wdl"].astype(np.float32), z["tgt"]
    fin = np.isfinite(wd).all(-1) & (wd.sum(-1) > 0)
    b = np.where(np.isfinite(tg), tg, -1e9).argmax(1)
    ok = (k == 0) & fin[np.arange(len(k)), b]
    if ok.any():
        V.append(fin[ok]); T.append(tg[ok]); W.append(wd[ok])
v, tg, wd = np.concatenate(V), np.concatenate(T), np.concatenate(W)
print(f"  {len(v):,} positions, {v.sum():,} child moves\n")

SCAL = {"W - L          ": wd[..., 0] - wd[..., 2],
        "W + 0.5D (exp) ": wd[..., 0] + 0.5 * wd[..., 1],
        "W only         ": wd[..., 0],
        "tgt (win prob) ": tg}
print("  (1) DISTINCT VALUES PER POSITION -- is W-L the problem, or is WDL?")
for nm, s in SCAL.items():
    a = np.where(v, s, np.nan)
    d = np.array([len(np.unique(np.round(r[~np.isnan(r)], 4))) for r in a[:4000]])
    print(f"      {nm} {d.mean():5.1f} distinct of ~{v[:4000].sum(1).mean():.0f} moves")

print("\n  (2) REGRET: how much win probability does each policy give up per move?")
rng = np.random.default_rng(0)
tvv = np.where(np.isfinite(tg) & v, tg, -np.inf)
best = tvv.max(1)
ok = np.isfinite(best)
def regret(choice):
    got = tg[np.arange(len(tg)), choice]
    r = best - got
    return r[ok & np.isfinite(got)]
n = v.sum(1)
rand = np.array([rng.choice(np.nonzero(v[i])[0]) if n[i] else 0 for i in range(len(v))])
pol = {"random legal move": rand,
       "always slot 0    ": np.zeros(len(v), int),
       "best by W-L      ": np.where(v, wd[..., 0]-wd[..., 2], -np.inf).argmax(1),
       "best by W+0.5D   ": np.where(v, wd[..., 0]+0.5*wd[..., 1], -np.inf).argmax(1)}
for nm, c in pol.items():
    r = regret(c)
    print(f"      {nm}  mean {r.mean():.4f}  median {np.median(r):.4f}  "
          f"p90 {np.percentile(r,90):.4f}")

print("\n  (3) ARE THE TIES REAL? If W-L cannot separate two moves, does tgt think they")
print("      differ, or are they genuinely equivalent?")
wl = np.where(v, wd[..., 0]-wd[..., 2], -np.inf)
top = wl.max(1, keepdims=True)
tied = (wl == top) & v
m = tied.sum(1) > 1
sub = np.where(tied, np.where(np.isfinite(tg), tg, np.nan), np.nan)[m]
spread = np.nanmax(sub, 1) - np.nanmin(sub, 1)
print(f"      positions whose top W-L is tied: {100*m.mean():.1f}%")
print(f"      among those tied moves, win-prob spread: median {np.nanmedian(spread):.4f}, "
      f"p90 {np.nanpercentile(spread,90):.4f}")
print(f"      tied moves that really ARE equivalent (win-prob spread < 0.01): "
      f"{100*(spread < 0.01).mean():.1f}%")
