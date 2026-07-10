"""
tsne_cones.py — three things:
  1. ORACLE vs PLANNER: DTM-perfect white (true full-strength play, from the
     tablebase) and the learned engine, same starts, same t-SNE map.
  2. KDE territories: regions as shaded areas instead of dot clouds.
  3. THE CONE, literally: Monte-Carlo futures under opponent uncertainty
     (black eps=0.25-optimal) sprayed on the map at successive plies —
     watch it narrow. Width measured in F-space (metric), not t-SNE coords.
"""
import numpy as np, time, pickle, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from openTSNE import TSNE
from krkn import KRKNChain
t0 = time.time()

uc = KRKNChain(verbose=False)
dtm = np.load("dtm_krkn.npy"); won = np.isfinite(dtm[:uc.n2])
scores = np.load("krkn_scores.npy"); F = np.load("krkn_F.npy")
mk = uc.move_kind; mp0 = uc.move_ptr[:-1]; op0 = uc.out_ptr[:-1]
out_counts = np.diff(uc.out_ptr)

dtm_full = np.full(uc.n, 1e6); dtm_full[:uc.nW] = np.where(np.isfinite(dtm), dtm, 1e6)
vf = dtm_full[uc.out_flat]
sm_ = np.maximum.reduceat(vf, op0)
B_opt = (np.minimum.reduceat(np.where(vf == np.repeat(sm_, out_counts),
        np.arange(len(vf)), len(vf)), op0) - op0).astype(np.int32)

Fw = F[:uc.nW].astype(np.float32)
Fn = (Fw - Fw.mean(0)) / (Fw.std(0) + 1e-9)

# ---- refit t-SNE (same recipe/seed as tsne_maps.py -> same map), then cache
try:
    with open("tsne_cache.pkl", "rb") as f:
        emb, fit_idx = pickle.load(f)
    print("t-SNE loaded from cache")
except FileNotFoundError:
    rng = np.random.default_rng(0)
    near = dtm[:uc.n2] <= 3
    idx_won   = rng.choice(np.where(won & ~near)[0], 9000, replace=False)
    idx_drawn = rng.choice(np.where(~won)[0], 7000, replace=False)
    idx_near  = np.where(near)[0]
    idx_near  = idx_near if len(idx_near) <= 3000 else rng.choice(idx_near, 3000, replace=False)
    idx_krk   = uc.n2 + rng.choice(uc.n1, 3000, replace=False)
    fit_idx = np.concatenate([idx_won, idx_drawn, idx_near, idx_krk])
    emb = TSNE(perplexity=40, initialization="pca", random_state=0, n_jobs=1).fit(Fn[fit_idx])
    with open("tsne_cache.pkl", "wb") as f: pickle.dump((emb, fit_idx), f)
print(f"map ready ({time.time()-t0:.0f}s)")
P = np.asarray(emb)
fit_won   = won[np.clip(fit_idx, 0, uc.n2-1)] & (fit_idx < uc.n2)
fit_drawn = (~won[np.clip(fit_idx, 0, uc.n2-1)]) & (fit_idx < uc.n2)
fit_krk   = fit_idx >= uc.n2

# ---- policies
def planner_move(s):
    a, b = mp0[s], uc.move_ptr[s+1]
    best, bv = a, -np.inf
    for mid in range(a, b):
        k = mk[mid]
        if k == 1: return mid
        if k in (2, 3): continue
        v = float(np.min(scores[uc.outs_of(mid)]))       # minimax readout
        if v > bv: bv, best = v, mid
    return best

def oracle_move(s):
    """DTM-perfect white: immediate mate, else minimize black's best reply DTM."""
    a, b = mp0[s], uc.move_ptr[s+1]
    best, bv = a, np.inf
    for mid in range(a, b):
        k = mk[mid]
        if k == 1: return mid
        if k in (2, 3): continue
        worst = dtm_full[int(uc.out_flat[op0[mid] + B_opt[mid]])]
        if worst < bv: bv, best = worst, mid
    return best

def play(start, mover, black_eps=0.0, cap=60, seed=0):
    r = np.random.default_rng(seed)
    s = int(start); path = [s]; result = "unfinished"
    for _ in range(cap):
        mid = mover(s); k = mk[mid]
        if k == 1: result = "mate"; break
        if k in (2, 3): result = "draw"; break
        outs = uc.outs_of(mid)
        bi = int(B_opt[mid]) if r.random() > black_eps else int(r.integers(0, len(outs)))
        nxt = int(outs[bi])
        if nxt >= uc.nW: result = "draw" if nxt == uc.DRAW_S else "end"; break
        path.append(nxt); s = nxt
    return path, result

# same-start pair: learned must mate
r5 = np.random.default_rng(5)
mid_won = np.where(won & (dtm[:uc.n2] >= 11) & (dtm[:uc.n2] <= 17))[0]
for _ in range(120):
    st = int(mid_won[r5.integers(0, len(mid_won))])
    pl, rl = play(st, planner_move)
    if rl == "mate": break
po, ro = play(st, oracle_move)
print(f"start DTM {dtm[st]:.0f}: planner {rl} in {len(pl)} mv | oracle {ro} in {len(po)} mv")

# ---- cone clouds: MC futures under opponent uncertainty
def cone_cloud(s0, mover, n_roll=150, horizon=24, black_eps=0.25, seed=7):
    r = np.random.default_rng(seed)
    pts = []                                  # (state, ply)
    for _ in range(n_roll):
        s = int(s0)
        for t_ in range(horizon):
            mid = mover(s); k = mk[mid]
            if k == 1 or k in (2, 3): break
            outs = uc.outs_of(mid)
            bi = int(B_opt[mid]) if r.random() > black_eps else int(r.integers(0, len(outs)))
            nxt = int(outs[bi])
            if nxt >= uc.nW: break
            pts.append((nxt, t_ + 1)); s = nxt
    return pts

show_plies = [0, max(1, len(pl)//3), max(2, 2*len(pl)//3), len(pl)-1]
show_plies = sorted(set(min(p, len(pl)-1) for p in show_plies))
cones_pl = {p: cone_cloud(pl[p], planner_move) for p in show_plies}
show_po = [min(p, len(po)-1) for p in show_plies]
cones_or = {p: cone_cloud(po[p], oracle_move) for p in show_po}
print(f"cones rolled ({time.time()-t0:.0f}s)")

# transform everything needed
need = set(pl) | set(po)
for d in list(cones_pl.values()) + list(cones_or.values()):
    need |= {s for s, _ in d}
need = sorted(need)
pos = {s: i for i, s in enumerate(need)}
E = np.asarray(emb.transform(Fn[np.array(need)]))
def pt(s): return E[pos[s]]
print(f"transformed {len(need)} points ({time.time()-t0:.0f}s)")

# cone width in F-SPACE (metric), per ply along each game
def width_curve(path, mover):
    w = []
    for p in range(len(path)):
        cl = cone_cloud(path[p], mover, n_roll=60, horizon=12, seed=11)
        if len(cl) < 5: w.append(0.0); continue
        X = Fn[[s for s, _ in cl]]
        w.append(float(np.linalg.norm(X - X.mean(0), axis=1).mean()))
    return w
w_pl, w_or = width_curve(pl, planner_move), width_curve(po, oracle_move)
print(f"width curves ({time.time()-t0:.0f}s)")

# ---- KDE territory backgrounds
def kde_layer(ax, pts2d, color, levels=5, alpha=0.30, nsub=2500):
    if len(pts2d) > nsub:
        pts2d = pts2d[np.random.default_rng(1).choice(len(pts2d), nsub, replace=False)]
    k = gaussian_kde(pts2d.T, bw_method=0.18)
    x0, x1 = P[:,0].min(), P[:,0].max(); y0, y1 = P[:,1].min(), P[:,1].max()
    gx, gy = np.meshgrid(np.linspace(x0, x1, 160), np.linspace(y0, y1, 160))
    z = k(np.stack([gx.ravel(), gy.ravel()])).reshape(gx.shape)
    ax.contourf(gx, gy, z, levels=np.linspace(z.max()*0.12, z.max(), levels),
                colors=[color]*levels, alpha=alpha)

def territories(ax):
    kde_layer(ax, P[fit_won], "#2E7D6B", alpha=0.28)
    kde_layer(ax, P[fit_drawn], "#5A6472", alpha=0.28)
    kde_layer(ax, P[fit_krk], "#8A5A38", alpha=0.30)

def style(ax, title):
    ax.set_facecolor("#12181F"); ax.set_title(title, color="#E8E4D9", fontsize=9.5)
    ax.set_xticks([]); ax.set_yticks([])
    for spn in ax.spines.values(): spn.set_color("#2A3542")

# ================= FIGURE 1: oracle vs planner =================
fig, ax = plt.subplots(figsize=(9.5, 7.5), facecolor="#10151C")
style(ax, f"Oracle (DTM-perfect) vs learned planner — same start (DTM {dtm[st]:.0f}), vs optimal defense\nterritories: won (green) / drawn (gray) / KRk stratum (rust), KDE-shaded")
territories(ax)
for path, res, col, nm, ls in ((po, ro, "#E8E4D9", f"oracle: {ro} in {len(po)} mv", "-"),
                                (pl, rl, "#7FC97F", f"planner: {rl} in {len(pl)} mv", "-")):
    pts = np.stack([pt(s) for s in path])
    ax.plot(pts[:,0], pts[:,1], ls, color=col, lw=2.2, alpha=.95, label=nm)
    for i in range(len(pts)-1):
        ax.annotate("", xy=pts[i+1], xytext=pts[i],
                    arrowprops=dict(arrowstyle="->", color=col, lw=1.2, alpha=.85))
    ax.scatter(*pts[0], color=col, s=90, marker="o", zorder=6, edgecolors="#10151C")
    ax.scatter(*pts[-1], color=col, s=150, marker="*", zorder=6, edgecolors="#10151C")
ax.legend(fontsize=9, facecolor="#171E27", labelcolor="#E8E4D9", edgecolor="#2A3542", loc="lower right")
plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/milestone1/krkn_oracle_vs_planner.png", dpi=140, facecolor="#10151C")
print("wrote krkn_oracle_vs_planner.png")

# ================= FIGURE 2: the cone, collapsing =================
nc = len(show_plies)
fig2, axes = plt.subplots(2, nc, figsize=(3.6*nc, 7.6), facecolor="#10151C")
cmap = plt.get_cmap("plasma")
for row, (paths, cones, plies, mover_nm) in enumerate(
        ((pl, cones_pl, show_plies, "planner"), (po, cones_or, show_po, "oracle"))):
    for j, p in enumerate(plies):
        ax = axes[row, j]
        style(ax, f"{mover_nm} · move {p+1}\ncone = {len(cones[p])} sampled futures")
        territories(ax)
        cl = cones[p]
        if cl:
            xs = np.stack([pt(s) for s, _ in cl])
            ts = np.array([t_ for _, t_ in cl], float)
            ax.scatter(xs[:,0], xs[:,1], s=6, c=cmap(0.15 + 0.7*ts/max(ts.max(),1)),
                       alpha=.35, linewidths=0)
        here = pt(paths[min(p, len(paths)-1)])
        ax.scatter(*here, s=170, marker="*", color="#F0A83C", zorder=6, edgecolors="#10151C")
fig2.suptitle("THE CONE, literally: Monte-Carlo futures under opponent uncertainty (black ε=0.25), "
              "colored by ply depth — watch it narrow toward mate", color="#E8E4D9", fontsize=11.5, y=0.99)
plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig("/mnt/user-data/outputs/milestone1/krkn_cone_filmstrip.png", dpi=135, facecolor="#10151C")
print("wrote krkn_cone_filmstrip.png")

# ================= FIGURE 3: cone width (metric, F-space) =================
fig3, ax = plt.subplots(figsize=(7.2, 4.4), facecolor="#10151C")
style(ax, "Cone WIDTH per move, measured in F-space (metric — not t-SNE coords)\nmean distance of sampled futures from their centroid")
ax.plot(range(1, len(w_pl)+1), w_pl, "-o", color="#7FC97F", ms=4, label=f"planner ({len(pl)} mv)")
ax.plot(range(1, len(w_or)+1), w_or, "-s", color="#E8E4D9", ms=4, label=f"oracle ({len(po)} mv)")
ax.set_xlabel("white move #", color="#8B94A3", fontsize=9)
ax.set_ylabel("cone width (F-space)", color="#8B94A3", fontsize=9)
ax.tick_params(colors="#8B94A3", labelsize=8)
ax.legend(fontsize=9, facecolor="#171E27", labelcolor="#E8E4D9", edgecolor="#2A3542")
ax.grid(alpha=.15)
plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/milestone1/krkn_cone_width.png", dpi=140, facecolor="#10151C")
print(f"wrote krkn_cone_width.png ({time.time()-t0:.0f}s)")
