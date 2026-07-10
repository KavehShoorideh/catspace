"""
exp_search.py — two questions:
 Q1: does a shallow MINIMAX search readout (min over black replies, not mean)
     close the gap to exact-DTM play?  k backups ~ (2k+1)-ply search with
     learned leaves.
 Q2: goal-region ablation: z_G from the oracle region {DTM<=3}∪{MATE} vs the
     oracle-free z = B[MATE] alone. Plus: highlight the region on the map.
Domain: KRkn (the hard case). KRRk run for the tempo question too.
"""
import numpy as np, time, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from krkn import KRKNChain
t0 = time.time()

uc = KRKNChain(verbose=False)
dtm = np.load("dtm_krkn.npy"); won = np.isfinite(dtm[:uc.n2])
F = np.load("krkn_F.npy"); Bm = np.load("krkn_B.npy")
mk = uc.move_kind; mp0 = uc.move_ptr[:-1]; op0 = uc.out_ptr[:-1]
out_counts = np.diff(uc.out_ptr); move_counts = np.diff(uc.move_ptr)
n_moves = len(mk); pos_idx = np.arange(n_moves)

dtm_full = np.full(uc.n, 1e6); dtm_full[:uc.nW] = np.where(np.isfinite(dtm), dtm, 1e6)
vf = dtm_full[uc.out_flat]
sm = np.maximum.reduceat(vf, op0)
B_opt = (np.minimum.reduceat(np.where(vf == np.repeat(sm, out_counts),
        np.arange(len(vf)), len(vf)), op0) - op0).astype(np.int32)

BIG = 1e15
def minimax_backup(V):
    """One white-black ply pair, exact rules, learned leaves:
       move value = MIN over black replies (worst case); state = MAX over moves."""
    M = np.minimum.reduceat(V[uc.out_flat], op0)
    M[mk == 1] = BIG
    M[(mk == 2) | (mk == 3)] = -BIG
    sv = np.maximum.reduceat(M, mp0)
    out = V.copy()
    out[:uc.nW] = np.clip(sv, -BIG, BIG)
    return out

def pol_from(V):
    M = np.minimum.reduceat(V[uc.out_flat], op0)
    M[mk == 1] = BIG
    M[(mk == 2) | (mk == 3)] = -BIG
    smax = np.maximum.reduceat(M, mp0)
    return (np.minimum.reduceat(np.where(M == np.repeat(smax, move_counts),
            pos_idx, n_moves), mp0) - mp0).astype(np.int32)

def evaluate(pol, n_eval=400, cap=70, seed=99):
    r = np.random.default_rng(seed)
    widx = np.where(won)[0]; starts = widx[r.integers(0, len(widx), size=n_eval)]
    mates, exact, rl, ratios = 0, 0, 0, []
    for s0 in starts:
        s = int(s0); opt_mv = np.ceil(dtm[s] / 2)
        for wm in range(cap):
            mid = mp0[s] + int(pol[s]); k = mk[mid]
            if k == 1:
                mates += 1; ratios.append((wm + 1) / max(opt_mv, 1))
                exact += (wm + 1) == opt_mv; break
            if k in (2, 3): break
            nxt = int(uc.out_flat[op0[mid] + B_opt[mid]])
            if nxt == uc.DRAW_S: rl += 1; break
            if nxt >= uc.nW: break
            s = nxt
    return mates/n_eval, exact/n_eval, (float(np.mean(ratios)) if ratios else np.nan), rl/n_eval

def run_sweep(zvec, label):
    scores = np.zeros(uc.n)
    scores[:] = F @ zvec
    print(f"\n[{label}]  depth = 2k+1-ply minimax search on the learned field")
    print("  k | search plies | conversion | EXACT-DTM games | moves/optimal | rook-lost")
    V = scores.copy()
    for k in range(0, 7):
        if k > 0: V = minimax_backup(V)
        m, ex, rt, rl = evaluate(pol_from(V))
        print(f"  {k} |     {2*k+1:2d}      |   {m:.3f}    |      {ex:.3f}      |     {rt:.3f}     |   {rl:.3f}   ({time.time()-t0:.0f}s)")
    return V

region_oracle = np.concatenate([np.where(dtm <= 3)[0], [uc.MATE_S]])
z_oracle = Bm[region_oracle].sum(0)
z_pure = Bm[uc.MATE_S]

run_sweep(z_oracle, "goal = oracle region {DTM<=3} ∪ {MATE}")
run_sweep(z_pure,  "goal = B[MATE] alone (oracle-free)")

# ---- map: where IS the DTM<=3 region?
Fw = F[:uc.nW]; X = Fw - Fw.mean(0)
subs = np.random.default_rng(0).choice(uc.nW, 20000, replace=False)
_, _, Vt = np.linalg.svd(X[subs], full_matrices=False)
P2 = X @ Vt[:2].T
near = (dtm[:uc.n2] <= 3)
fig, ax = plt.subplots(figsize=(8, 6.4), facecolor="#10151C")
ax.set_facecolor("#171E27")
ax.scatter(P2[:uc.n2][won & ~near, 0], P2[:uc.n2][won & ~near, 1], s=2, c="#4EC9B0", alpha=.25, linewidths=0, label="won")
ax.scatter(P2[:uc.n2][~won, 0], P2[:uc.n2][~won, 1], s=2, c="#6B7280", alpha=.25, linewidths=0, label="drawn")
ax.scatter(P2[uc.n2:, 0], P2[uc.n2:, 1], s=2, c="#C97B4E", alpha=.3, linewidths=0, label="KRk stratum")
ax.scatter(P2[:uc.n2][near, 0], P2[:uc.n2][near, 1], s=9, c="#F0A83C", alpha=.9, linewidths=0.3,
           edgecolors="#10151C", label=f"the goal region: DTM≤3 ({near.sum():,} states)")
ax.legend(fontsize=9, facecolor="#171E27", labelcolor="#E8E4D9", edgecolor="#2A3542", loc="lower right")
ax.set_title("The plan target G, made visible — z_G = Σ_{g∈G} B(g)\n(oracle-defined; see ablation for the oracle-free alternative)",
             color="#E8E4D9", fontsize=11)
ax.tick_params(colors="#8B94A3", labelsize=7)
for spn in ax.spines.values(): spn.set_color("#2A3542")
plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/milestone1/goal_region.png", dpi=140, facecolor="#10151C")
print(f"\nwrote goal_region.png ({time.time()-t0:.0f}s)")
