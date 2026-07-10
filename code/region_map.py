"""
region_map.py — regions as AREAS on the 2D map, not dot clouds.

Panels:
  A: token TERRITORIES — grid-voted majority token, filled regions with boundaries
  B: the two force fields — mate-reach (amber contours) vs draw-reach (red contours),
     both read from the SAME learned cone: F @ B[MATE] and F @ B[DRAW]
  C: the tug-of-war — one PI-engine game and one random game overlaid on the fields:
     white's moves pull toward the mate attractor, black's replies pull toward
     the draw sink; the random game gets dragged in, the learned one escapes.
"""
import numpy as np, scipy.sparse as sp, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from domain import (compute_dtm, concept_features, white_moves, black_moves,
                    classify_b, MATE, STALEMATE)
from learn import Chain, randomized_svd_sm, fb_from_svd

GAMMA = 0.92
ch = Chain()
dtm_w, _ = compute_dtm(ch.W, ch.B)
feats = concept_features(ch.W, dtm_w)
region = np.array(list(np.where(dtm_w <= 3)[0]) + [ch.MATE_S])

# ---- rebuild final PI cone (same schedule/seeds as before, includes absorbing B rows)
W_move_out, W_move_kind, B_opt_reply = [], [], []
for si in range(ch.nW):
    outs_per_move, kinds, opts = [], [], []
    for bnode in white_moves(*ch.W[si]):
        cls = classify_b(*bnode)
        if cls == MATE: kinds.append(1); outs_per_move.append(np.array([ch.MATE_S])); opts.append(0)
        elif cls == STALEMATE: kinds.append(2); outs_per_move.append(np.array([ch.DRAW_S])); opts.append(0)
        else:
            reps = black_moves(*bnode); nxts, bi_, bv_ = [], 0, -np.inf
            for i, (nxt, cap) in enumerate(reps):
                if cap:
                    nxts.append(ch.DRAW_S)
                    if bv_ < 1e6: bv_, bi_ = 1e6, i
                else:
                    wi = ch.Wi[nxt]; nxts.append(wi)
                    v = dtm_w[wi] if np.isfinite(dtm_w[wi]) else 1e6
                    if v > bv_: bv_, bi_ = v, i
            kinds.append(0); outs_per_move.append(np.array(nxts)); opts.append(bi_)
    W_move_out.append(outs_per_move); W_move_kind.append(kinds); B_opt_reply.append(opts)

def greedy_pol(scores):
    pol = np.zeros(ch.nW, dtype=np.int32)
    for s in range(ch.nW):
        best, bv = 0, -np.inf
        for m, outs in enumerate(W_move_out[s]):
            k = W_move_kind[s][m]
            if k == 1: best = m; bv = np.inf; break
            v = -1e9 if k == 2 else float(np.where(outs == ch.DRAW_S, 0.0,
                    scores[np.minimum(outs, ch.nW-1)]).mean())
            if v > bv: bv, best = v, m
        pol[s] = best
    return pol

def sample_round(pol_w, ew, eb, ng, seed):
    r = np.random.default_rng(seed); rows, cols = [], []
    for s0 in r.integers(0, ch.nW, size=ng):
        s = int(s0)
        for _ in range(120):
            m = int(pol_w[s]) if r.random() > ew else int(r.integers(0, len(W_move_out[s])))
            k = W_move_kind[s][m]; outs = W_move_out[s][m]
            if k == 1: nxt = ch.MATE_S
            elif k == 2: nxt = ch.DRAW_S
            else:
                bi = B_opt_reply[s][m] if r.random() > eb else int(r.integers(0, len(outs)))
                nxt = int(outs[bi])
            rows.append(s); cols.append(nxt)
            if nxt >= ch.nW: break
            s = nxt
    return rows, cols

def estimate(all_rows, all_cols, d=64):
    counts = sp.coo_matrix((np.ones(len(all_rows)), (all_rows, all_cols)), shape=(ch.n, ch.n)).tocsr()
    rowsum = np.asarray(counts.sum(1)).ravel(); seen = rowsum > 0; rowsum[rowsum == 0] = 1
    P = (sp.diags(1/rowsum) @ counts).tolil()
    for i in np.where(~seen)[0]: P[i, i] = 1.0
    for a in (ch.MATE_S, ch.DRAW_S): P[a, :] = 0; P[a, a] = 1.0
    U, S, V = randomized_svd_sm(P.tocsr(), GAMMA, d=d, seed=0)
    return fb_from_svd(U, S, V)

all_rows, all_cols, scores = [], [], np.zeros(ch.nW)
for k, (ew, eb, ng) in enumerate([(1,1,20000),(0.3,0.5,20000),(0.3,0.25,20000),
                                   (0.2,0.1,20000),(0.2,0,20000),(0.2,0,20000)]):
    rows, cols = sample_round(greedy_pol(scores), ew, eb, ng, seed=100+k)
    all_rows += rows; all_cols += cols
    F, Bm = estimate(all_rows, all_cols)
    scores = (F @ Bm[region].sum(0))[:ch.nW]
print("PI cone rebuilt")

Fw = F[:ch.nW]
mate_field = Fw @ Bm[ch.MATE_S]     # cone mass flowing into MATE
draw_field = Fw @ Bm[ch.DRAW_S]     # cone mass flowing into DRAW

# ---- 2D projection + VQ tokens
X = Fw - Fw.mean(0)
U2, S2, Vt2 = np.linalg.svd(X, full_matrices=False)
P2 = X @ Vt2[:2].T
def kmeans(Xk, K, iters=60, seed=5):
    r = np.random.default_rng(seed)
    C = Xk[r.choice(len(Xk), K, replace=False)].copy()
    for _ in range(iters):
        a = ((Xk[:, None, :] - C[None, :, :])**2).sum(-1).argmin(1)
        for k in range(K):
            m = a == k
            if m.any(): C[k] = Xk[m].mean(0)
    return a
Xf = Fw / (np.linalg.norm(Fw, axis=1, keepdims=True) + 1e-9)
assign = kmeans(Xf, 12)

# ---- grid fields: majority token + smoothed mate/draw fields
GRID = 220
x0, x1 = P2[:,0].min(), P2[:,0].max(); y0, y1 = P2[:,1].min(), P2[:,1].max()
pad = 0.04 * max(x1-x0, y1-y0); x0-=pad; x1+=pad; y0-=pad; y1+=pad
gx, gy = np.meshgrid(np.linspace(x0, x1, GRID), np.linspace(y0, y1, GRID))
gpts = np.stack([gx.ravel(), gy.ravel()], 1)

# k-NN vote / weighted average (chunked to fit memory)
def knn_fields(gpts, P2, k=25):
    tok_grid = np.zeros(len(gpts), dtype=int)
    mate_grid = np.zeros(len(gpts)); draw_grid = np.zeros(len(gpts))
    dens_grid = np.zeros(len(gpts))
    for i0 in range(0, len(gpts), 4000):
        g = gpts[i0:i0+4000]
        d2 = ((g[:, None, :] - P2[None, :, :])**2).sum(-1)
        idx = np.argpartition(d2, k, axis=1)[:, :k]
        rows = np.arange(len(g))[:, None]
        w = 1.0 / (d2[rows, idx] + 1e-9)
        w /= w.sum(1, keepdims=True)
        mate_grid[i0:i0+4000] = (w * mate_field[idx]).sum(1)
        draw_grid[i0:i0+4000] = (w * draw_field[idx]).sum(1)
        dens_grid[i0:i0+4000] = np.sort(d2, axis=1)[:, k-1]   # kth-NN distance (inverse density)
        for j in range(len(g)):
            tok_grid[i0+j] = np.bincount(assign[idx[j]], weights=w[j]).argmax()
    return tok_grid.reshape(GRID, GRID), mate_grid.reshape(GRID, GRID), \
           draw_grid.reshape(GRID, GRID), dens_grid.reshape(GRID, GRID)
tok_g, mate_g, draw_g, dens_g = knn_fields(gpts, P2)
occupied = dens_g < np.quantile(dens_g, 0.80)   # mask empty space
print("grids done")

# ---- two story games vs optimal defense
def play_story(start, mode, cap=40, seed=0):
    r = np.random.default_rng(seed)
    s = int(start); path = [s]; result = "unfinished"
    for _ in range(cap):
        if mode == "learned":
            best, bv = 0, -np.inf
            for m, outs in enumerate(W_move_out[s]):
                k = W_move_kind[s][m]
                if k == 1: best = m; bv = np.inf; break
                v = -1e9 if k == 2 else float(np.where(outs == ch.DRAW_S, 0.0,
                        scores[np.minimum(outs, ch.nW-1)]).mean())
                if v > bv: bv, best = v, m
            m = best
        else:
            m = int(r.integers(0, len(W_move_out[s])))
        k = W_move_kind[s][m]; outs = W_move_out[s][m]
        if k == 1: result = "mate"; break
        if k == 2: result = "draw"; break
        nxt = int(outs[B_opt_reply[s][m]])
        if nxt >= ch.nW: result = "draw"; break
        path.append(nxt); s = nxt
    return path, result

rng = np.random.default_rng(11)
hard = np.where(dtm_w >= 15)[0]
p_learn = p_rand = None
for t in range(30):
    start = int(hard[rng.integers(0, len(hard))])
    pl, rl = play_story(start, "learned", seed=1 + t)
    pr, rr = play_story(start, "random",  seed=2 + t)
    if rl == "mate" and rr == "draw" and len(pr) >= 3:
        p_learn, r_learn, p_rand, r_rand = pl, rl, pr, rr
        break
if p_learn is None:   # fall back to any learned mate
    for t in range(30):
        start = int(hard[rng.integers(0, len(hard))])
        pl, rl = play_story(start, "learned", seed=100 + t)
        if rl == "mate":
            p_learn, r_learn = pl, rl
            p_rand, r_rand = play_story(start, "random", seed=3)
            break
print(f"story games from same start (DTM {dtm_w[start]:.0f}): learned -> {r_learn} in {len(p_learn)} moves, random -> {r_rand} in {len(p_rand)} moves")

# ---- figure
fig, axes = plt.subplots(1, 3, figsize=(19, 6.4), facecolor="#10151C")
def style(ax, title):
    ax.set_facecolor("#171E27"); ax.set_title(title, color="#E8E4D9", fontsize=11.5)
    ax.tick_params(colors="#8B94A3", labelsize=7)
    for spn in ax.spines.values(): spn.set_color("#2A3542")
ext = [x0, x1, y0, y1]

# A: token territories
axA = axes[0]; style(axA, "A · plan-token TERRITORIES (regions, not dots)\nmajority token per grid cell, K=12")
cmap = plt.get_cmap("tab20")
tok_img = np.ma.masked_where(~occupied, tok_g)
axA.imshow(tok_img, origin="lower", extent=ext, cmap=cmap, alpha=0.75, interpolation="nearest", aspect="auto")
axA.contour(gx, gy, tok_g, levels=np.arange(12)+0.5, colors="#10151C", linewidths=0.6)
# label each territory at its centroid with mean DTM
for t in range(12):
    m = assign == t
    if m.sum() < 30: continue
    cx, cy = P2[m,0].mean(), P2[m,1].mean()
    axA.text(cx, cy, f"#{t}\nDTM {feats['dtm'][m].mean():.0f}", color="#10151C",
             fontsize=8, ha="center", va="center", fontweight="bold")

# B: the two force fields
axB = axes[1]; style(axB, "B · the two attractors, one cone: mate-flow (amber) vs draw-flow (red)\ncontours of F·B[MATE] and F·B[DRAW]")
mg = np.ma.masked_where(~occupied, mate_g); dg = np.ma.masked_where(~occupied, draw_g)
axB.contourf(gx, gy, mg, levels=8, cmap="YlOrBr", alpha=0.55)
cs2 = axB.contour(gx, gy, dg, levels=6, colors="#FF6B6B", linewidths=1.2)
axB.clabel(cs2, fmt="%.2f", fontsize=6, colors="#FF6B6B")
# mark mate-adjacent and high-draw poles
near = dtm_w <= 2
axB.scatter(P2[near,0], P2[near,1], s=10, c="#F0A83C", marker="*", label="DTM≤2 (mate doorstep)")
hd = draw_field > np.quantile(draw_field, 0.97)
axB.scatter(P2[hd,0], P2[hd,1], s=6, c="#FF6B6B", marker="x", label="highest draw-flow (rook in danger)")
axB.legend(fontsize=7.5, facecolor="#171E27", labelcolor="#E8E4D9", edgecolor="#2A3542", loc="lower right")

# C: tug-of-war
axC = axes[2]; style(axC, f"C · tug-of-war from the SAME start (DTM {dtm_w[start]:.0f}) vs optimal defense\nlearned → {r_learn} · random → {r_rand}")
axC.contourf(gx, gy, mg, levels=8, cmap="YlOrBr", alpha=0.35)
axC.contour(gx, gy, dg, levels=5, colors="#FF6B6B", linewidths=0.8, alpha=0.7)
for path, res, col, nm in ((p_learn, r_learn, "#7FC97F", "learned"), (p_rand, r_rand, "#B39DDB", "random")):
    pts = P2[path]
    axC.plot(pts[:,0], pts[:,1], "-", color=col, lw=2.0, alpha=0.95, label=f"{nm} ({res})")
    for i in range(len(pts)-1):
        axC.annotate("", xy=pts[i+1], xytext=pts[i],
                     arrowprops=dict(arrowstyle="->", color=col, lw=1.2, alpha=0.85))
    axC.scatter(*pts[0], color=col, s=80, marker="o", zorder=5, edgecolors="#10151C")
    axC.scatter(*pts[-1], color=col, s=130, marker="*" if res=="mate" else "X", zorder=5, edgecolors="#10151C")
axC.legend(fontsize=8, facecolor="#171E27", labelcolor="#E8E4D9", edgecolor="#2A3542", loc="lower right")

plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/milestone1/region_map.png", dpi=140, facecolor="#10151C")
print("wrote region_map.png")
