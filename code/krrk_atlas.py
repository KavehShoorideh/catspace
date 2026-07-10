"""
krrk_atlas.py — the stratified region map for KRRK, plus a filmstrip.

Panels:
  A: STRATA as regions — the KRRK cloud and the KRK cloud in one cone geometry,
     with observed capture transitions drawn as chutes between them
  B: token territories (K=24) with mean-DTM labels
  C: concept field — the two-rook box area as filled contours (walls = union of
     both rooks' cuts)
  D: tug-of-war from the same start vs optimal defense: learned stays in the
     KRRK stratum and mates; random hangs a rook, drops the chute, draws
Filmstrip: one learned game as rendered boards with move/DTM/box/token per ply.
"""
import numpy as np, matplotlib, time
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from krrk import UnionChain, rc, rook_attacks, KING_MOVES
t0 = time.time()

uc = UnionChain(verbose=False)
dtm = np.load("dtm_union.npy")
F = np.load("krrk_F.npy"); Bm = np.load("krrk_B.npy"); scores = np.load("krrk_scores.npy")
B_opt = None  # rebuilt below (cheap enough)
B_opt = np.zeros(len(uc.move_kind), dtype=np.int32)
for mid in range(len(uc.move_kind)):
    if uc.move_kind[mid] == 0:
        outs = uc.outs_of(mid)
        vals = np.where(outs >= uc.nW, 1e6, dtm[np.minimum(outs, uc.nW-1)])
        B_opt[mid] = int(np.argmax(vals))
print(f"setup ({time.time()-t0:.0f}s)")

# ---- concepts on KRRK states
def box2(wk, ra, rb, bk):
    walls = {ra, rb}
    for t in range(25):
        if t == wk: continue
        if rook_attacks(ra, t, {wk, rb}) or rook_attacks(rb, t, {wk, ra}):
            walls.add(t)
    if bk in walls: return 25
    seen = {bk}; stack = [bk]
    while stack:
        cur = stack.pop()
        for t in KING_MOVES[cur]:
            if t not in walls and t not in seen:
                seen.add(t); stack.append(t)
    return len(seen)

box_area = np.array([box2(*s) for s in uc.W2], dtype=float)
print(f"box areas ({time.time()-t0:.0f}s)")

# ---- 2D projection over ALL union W states
Fw = F[:uc.nW]
X = Fw - Fw.mean(0)
_, S2, Vt2 = np.linalg.svd(X[np.random.default_rng(0).choice(uc.nW, 20000, replace=False)], full_matrices=False)
P2 = X @ Vt2[:2].T
is_krk = np.zeros(uc.nW, bool); is_krk[uc.n2:] = True

# ---- VQ tokens (K=24) on normalized F
def kmeans(Xk, K, iters=40, seed=5):
    r = np.random.default_rng(seed)
    C = Xk[r.choice(len(Xk), K, replace=False)].copy()
    for _ in range(iters):
        d2 = ((Xk[:, None, :] - C[None, :, :])**2).sum(-1)
        a = d2.argmin(1)
        for k in range(K):
            m = a == k
            if m.any(): C[k] = Xk[m].mean(0)
    return a
Xf = Fw / (np.linalg.norm(Fw, axis=1, keepdims=True) + 1e-9)
assign = kmeans(Xf[:, :20], 24)
print(f"tokens ({time.time()-t0:.0f}s)")

# ---- grid fields (subsampled KNN for memory)
GRID = 200
sub = np.random.default_rng(1).choice(uc.nW, 14000, replace=False)
x0, x1 = P2[:,0].min(), P2[:,0].max(); y0, y1 = P2[:,1].min(), P2[:,1].max()
pad = .04*max(x1-x0, y1-y0); x0-=pad; x1+=pad; y0-=pad; y1+=pad
gx, gy = np.meshgrid(np.linspace(x0,x1,GRID), np.linspace(y0,y1,GRID))
gpts = np.stack([gx.ravel(), gy.ravel()], 1)
tokg = np.zeros(len(gpts), int); boxg = np.zeros(len(gpts)); densg = np.zeros(len(gpts))
box_full = np.concatenate([box_area, np.full(uc.n1, np.nan)])
Psub, k = P2[sub], 25
for i0 in range(0, len(gpts), 3000):
    g = gpts[i0:i0+3000]
    d2 = ((g[:,None,:]-Psub[None,:,:])**2).sum(-1)
    idx = np.argpartition(d2, k, 1)[:, :k]
    rows = np.arange(len(g))[:,None]
    w = 1/(d2[rows, idx]+1e-9); w /= w.sum(1, keepdims=True)
    bx = box_full[sub[idx]]
    bw = np.where(np.isnan(bx), 0, w); bx = np.where(np.isnan(bx), 0, bx)
    boxg[i0:i0+3000] = (bw*bx).sum(1)/np.maximum(bw.sum(1), 1e-9)
    densg[i0:i0+3000] = np.sort(d2,1)[:, k-1]
    for j in range(len(g)):
        tokg[i0+j] = np.bincount(assign[sub[idx[j]]], weights=w[j]).argmax()
tokg, boxg, densg = tokg.reshape(GRID,GRID), boxg.reshape(GRID,GRID), densg.reshape(GRID,GRID)
occ = densg < np.quantile(densg, 0.80)
print(f"grids ({time.time()-t0:.0f}s)")

# ---- chute samples: observed capture edges (KRRK -> KRK)
rng = np.random.default_rng(2)
chutes = []
while len(chutes) < 30:
    s = int(rng.integers(0, uc.n2))
    for mid in uc.moves_of(s):
        if uc.move_kind[mid] != 0: continue
        outs = uc.outs_of(mid)
        kk = outs[(outs >= uc.n2) & (outs < uc.nW)]
        if len(kk):
            chutes.append((s, int(kk[rng.integers(0, len(kk))])))
            break

# ---- tug-of-war games vs optimal defense
def play(start, mode, cap=40, seed=0):
    r = np.random.default_rng(seed)
    s = int(start); path = [s]; result = "unfinished"
    for _ in range(cap):
        a, b = uc.move_ptr[s], uc.move_ptr[s+1]
        if mode == "learned":
            best, bv = a, -np.inf
            for mid in range(a, b):
                kk = uc.move_kind[mid]
                if kk == 1: best, bv = mid, np.inf; break
                if kk == 2: continue
                v = float(scores[uc.outs_of(mid)].mean())
                if v > bv: bv, best = v, mid
            mid = best
        else:
            mid = a + int(r.integers(0, b - a))
        kk = uc.move_kind[mid]
        if kk == 1: result = "mate"; break
        if kk == 2: result = "draw"; break
        nxt = int(uc.outs_of(mid)[B_opt[mid]])
        if nxt == uc.DRAW_S: result = "draw"; break
        if nxt == uc.MATE_S: result = "mate"; break
        path.append(nxt); s = nxt
    return path, result

hardest = np.where(dtm[:uc.n2] >= 8)[0]
pl = pr = None
for t in range(40):
    st = int(hardest[rng.integers(0, len(hardest))])
    pl_, rl_ = play(st, "learned", seed=t)
    pr_, rr_ = play(st, "random", seed=1000+t)
    crossed = any(x >= uc.n2 for x in pr_)
    if rl_ == "mate" and rr_ == "draw" and crossed and len(pr_) >= 3:
        pl, rl, pr, rr, start = pl_, rl_, pr_, rr_, st; break
if pl is None:
    st = int(hardest[0]); pl, rl = play(st, "learned", seed=0); pr, rr = play(st, "random", seed=7); start = st
print(f"story: learned {rl} in {len(pl)}, random {rr} in {len(pr)} (crossed strata: {any(x>=uc.n2 for x in pr)})")

# ---- figure
fig, axes = plt.subplots(2, 2, figsize=(15.5, 12.5), facecolor="#10151C")
def style(ax, title):
    ax.set_facecolor("#171E27"); ax.set_title(title, color="#E8E4D9", fontsize=11)
    ax.tick_params(colors="#8B94A3", labelsize=7)
    for spn in ax.spines.values(): spn.set_color("#2A3542")
ext = [x0, x1, y0, y1]

axA = axes[0,0]; style(axA, "A · STRATA in one cone geometry: KRRK (teal) and KRK (rust) clouds\ngray arrows = observed rook-capture chutes")
axA.scatter(P2[~is_krk,0], P2[~is_krk,1], s=2, c="#4EC9B0", alpha=.4, linewidths=0, label=f"KRRK stratum ({uc.n2:,})")
axA.scatter(P2[is_krk,0], P2[is_krk,1], s=2, c="#C97B4E", alpha=.5, linewidths=0, label=f"KRK stratum ({uc.n1:,})")
for s, kkk in chutes:
    axA.annotate("", xy=P2[kkk], xytext=P2[s],
                 arrowprops=dict(arrowstyle="->", color="#8B94A3", lw=0.9, alpha=0.55))
axA.legend(fontsize=8, facecolor="#171E27", labelcolor="#E8E4D9", edgecolor="#2A3542")

axB = axes[0,1]; style(axB, "B · plan-token TERRITORIES (K=24), labels = mean DTM")
tok_img = np.ma.masked_where(~occ, tokg)
axB.imshow(tok_img, origin="lower", extent=ext, cmap="tab20", alpha=.75, interpolation="nearest", aspect="auto")
axB.contour(gx, gy, tokg, levels=np.arange(24)+.5, colors="#10151C", linewidths=.5)
dtm_cap = np.where(np.isfinite(dtm[:uc.nW]), dtm[:uc.nW], 20)
for tk in range(24):
    m = assign == tk
    if m.sum() < 200: continue
    axB.text(P2[m,0].mean(), P2[m,1].mean(), f"{dtm_cap[m].mean():.0f}",
             color="#10151C", fontsize=9, ha="center", va="center", fontweight="bold")

axC = axes[1,0]; style(axC, "C · learned geometry vs a concept never shown to it:\ntwo-rook BOX AREA (filled contours; KRK stratum masked)")
bg = np.ma.masked_where(~occ, boxg)
im = axC.contourf(gx, gy, bg, levels=10, cmap="viridis_r", alpha=.8)
cb = plt.colorbar(im, ax=axC); cb.ax.tick_params(colors="#8B94A3", labelsize=7)
cb.set_label("black king's box (squares)", color="#8B94A3", fontsize=8)

axD = axes[1,1]; style(axD, f"D · tug-of-war, same start (DTM {dtm[start]:.0f}) vs optimal defense\nlearned → {rl} (stays high) · random → {rr} (falls down the chute)")
axD.scatter(P2[~is_krk,0], P2[~is_krk,1], s=1.5, c="#4EC9B0", alpha=.16, linewidths=0)
axD.scatter(P2[is_krk,0], P2[is_krk,1], s=1.5, c="#C97B4E", alpha=.2, linewidths=0)
for path, res, col, nm in ((pl, rl, "#7FC97F", "learned"), (pr, rr, "#B39DDB", "random")):
    pts = P2[path]
    axD.plot(pts[:,0], pts[:,1], "-", color=col, lw=2.2, alpha=.95, label=f"{nm} ({res})")
    for i in range(len(pts)-1):
        axD.annotate("", xy=pts[i+1], xytext=pts[i],
                     arrowprops=dict(arrowstyle="->", color=col, lw=1.3, alpha=.9))
    axD.scatter(*pts[0], color=col, s=90, marker="o", zorder=5, edgecolors="#10151C")
    axD.scatter(*pts[-1], color=col, s=150, marker="*" if res=="mate" else "X", zorder=5, edgecolors="#10151C")
axD.legend(fontsize=8.5, facecolor="#171E27", labelcolor="#E8E4D9", edgecolor="#2A3542")

fig.suptitle("KRRK — stratified region map · cone trained by opponent-curriculum policy iteration · 97.7% vs optimal defense",
             color="#E8E4D9", fontsize=13, y=0.995)
plt.tight_layout(rect=[0,0,1,.97])
plt.savefig("/mnt/user-data/outputs/milestone1/krrk_region_map.png", dpi=135, facecolor="#10151C")
print(f"wrote krrk_region_map.png ({time.time()-t0:.0f}s)")

# ---- filmstrip of the learned game
n_show = len(pl)
fig2, axs = plt.subplots(1, n_show, figsize=(2.1*n_show, 3.2), facecolor="#10151C")
if n_show == 1: axs = [axs]
for i, (ax, s) in enumerate(zip(axs, pl)):
    wk, ra, rb, bk = uc.W2[s] if s < uc.n2 else (None,)*4
    if s >= uc.n2:
        wk, wr1, bk = uc.W1[s - uc.n2]; ra, rb = wr1, None
    ax.set_facecolor("#171E27")
    for r_ in range(5):
        for c_ in range(5):
            ax.add_patch(plt.Rectangle((c_, r_), 1, 1,
                color="#D8C9A3" if (r_+c_)%2 else "#7A6248", zorder=0))
    def put(sqr, glyph, col):
        if sqr is None: return
        rr, cc = rc(sqr)
        ax.text(cc+.5, rr+.5, glyph, fontsize=17, ha="center", va="center",
                color=col, zorder=3, fontweight="bold")
    put(wk, "♔", "#F7F3E8"); put(ra, "♖", "#F7F3E8"); put(rb, "♖", "#F7F3E8"); put(bk, "♚", "#191919")
    ax.set_xlim(0,5); ax.set_ylim(0,5); ax.set_xticks([]); ax.set_yticks([])
    bx = box2(*uc.W2[s]) if s < uc.n2 else float("nan")
    ax.set_title(f"mv {i+1} · DTM {dtm[s]:.0f}\nbox {bx:.0f} · tok #{assign[s]}",
                 color="#E8E4D9", fontsize=8)
fig2.suptitle(f"learned KRRK game vs optimal defense → {rl}", color="#E8E4D9", fontsize=11)
plt.tight_layout(rect=[0,0,1,.9])
plt.savefig("/mnt/user-data/outputs/milestone1/krrk_filmstrip.png", dpi=135, facecolor="#10151C")
print(f"wrote krrk_filmstrip.png ({time.time()-t0:.0f}s)")
