"""
atlas.py — 2D atlas of the neural embedding space.

Panels (PCA projection of neural F, all 7,040 states):
  A: colored by BOX AREA  (the boxing-the-king concept — the requested story)
  B: colored by DTM       (distance to mate)
  C: colored by VQ token
  D: colored by learned reach score, with HELD-OUT states overplotted as
     triangles (do unseen states embed where they belong?)
  E: three game trajectories overlaid on the box-area map: unboxed start ->
     boxed -> mate, each arrowed step = one white move
  F: the same games as curves: box area shrinking + reach rising per ply
"""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from domain import compute_dtm, concept_features, white_moves, black_moves, classify_b, MATE, STALEMATE
from learn import Chain

ch = Chain()
dtm_w, _ = compute_dtm(ch.W, ch.B)
feats = concept_features(ch.W, dtm_w)
Fn = np.load("Fn_neural.npy")
holdout = np.load("holdout_mask.npy")
reach_n = np.load("reach_neural.npy")

# ---- PCA to 2D
X = Fn - Fn.mean(0)
U, S, Vt = np.linalg.svd(X, full_matrices=False)
P2 = X @ Vt[:2].T
print("PCA explained var (2D):", (S[:2]**2).sum() / (S**2).sum())

# ---- VQ on neural embedding
def kmeans(Xk, K, iters=60, seed=5):
    r = np.random.default_rng(seed)
    C = Xk[r.choice(len(Xk), K, replace=False)].copy()
    for _ in range(iters):
        a = ((Xk[:, None, :] - C[None, :, :])**2).sum(-1).argmin(1)
        for k in range(K):
            m = a == k
            if m.any(): C[k] = Xk[m].mean(0)
    return a
Xf = Fn / (np.linalg.norm(Fn, axis=1, keepdims=True) + 1e-9)
assign = kmeans(Xf, 16)   # 16 tokens: honest for a domain this small

# ---- games for trajectory overlay: neural engine vs RANDOM black (its trained regime)
import json
zG = np.array(json.load(open("zg_neural.json"))["zG"], dtype=np.float32)
scores = reach_n
DRAW_PEN = float(np.quantile(reach_n, 0.001))   # same convention as the evaluated engine
def play_traj(start, cap=30, seed=0):
    r = np.random.default_rng(seed)
    s = int(start); path = [s]; result = "unfinished"
    for _ in range(cap):
        bnodes = white_moves(*ch.W[s])
        best, bv = 0, -np.inf
        for m, bnode in enumerate(bnodes):
            cls = classify_b(*bnode)
            if cls == MATE: best = m; bv = np.inf; break
            if cls == STALEMATE: continue
            reps = black_moves(*bnode)
            vals = []
            for nxt, cap_ in reps:
                vals.append(DRAW_PEN if cap_ else float(scores[ch.Wi[nxt]]))
            v = float(np.mean(vals)) if vals else -1e9
            if v > bv: bv, best = v, m
        bnode = bnodes[best]
        cls = classify_b(*bnode)
        if cls == MATE: result = "mate"; break
        if cls == STALEMATE: result = "draw"; break
        reps = black_moves(*bnode)
        nxt, cap_ = reps[int(r.integers(0, len(reps)))]
        if cap_: result = "draw"; break
        s = ch.Wi[nxt]; path.append(s)
    return path, result

rng = np.random.default_rng(3)
big_box = np.where((feats['box_area'] >= 15) & (dtm_w >= 13))[0]
trajs, attempts = [], 0
for i in range(40):
    start = int(big_box[rng.integers(0, len(big_box))])
    path, res = play_traj(start, cap=40, seed=10 + i)
    attempts += 1
    if res == "mate":
        trajs.append((path, res))
        print(f"traj kept: start box={feats['box_area'][start]:.0f} DTM={dtm_w[start]:.0f} -> {res} in {len(path)} white moves")
    if len(trajs) == 3: break
print(f"hard-start (box>=15, DTM>=13) mate hit-rate in this sample: {len(trajs)}/{attempts}")
if len(trajs) < 3:
    print("WARNING: fewer than 3 mating trajectories found from hard starts")

# ---- figure
fig = plt.figure(figsize=(15, 10), facecolor="#10151C")
def style(ax, title):
    ax.set_facecolor("#171E27")
    ax.set_title(title, color="#E8E4D9", fontsize=11)
    ax.tick_params(colors="#8B94A3", labelsize=7)
    for sp in ax.spines.values(): sp.set_color("#2A3542")

s_kw = dict(s=3, alpha=0.55, linewidths=0)

axA = fig.add_subplot(2, 3, 1); style(axA, "A · colored by BOX AREA (the boxing concept)")
sc = axA.scatter(P2[:, 0], P2[:, 1], c=feats['box_area'], cmap="viridis_r", **s_kw)
plt.colorbar(sc, ax=axA).ax.tick_params(colors="#8B94A3", labelsize=7)

axB = fig.add_subplot(2, 3, 2); style(axB, "B · colored by DTM (plies to mate, ground truth)")
sc = axB.scatter(P2[:, 0], P2[:, 1], c=np.minimum(dtm_w, 19), cmap="magma_r", **s_kw)
plt.colorbar(sc, ax=axB).ax.tick_params(colors="#8B94A3", labelsize=7)

axC = fig.add_subplot(2, 3, 3); style(axC, "C · colored by VQ token (16 codes, k-means on F)")
axC.scatter(P2[:, 0], P2[:, 1], c=assign, cmap="tab20", **s_kw)

axD = fig.add_subplot(2, 3, 4); style(axD, "D · learned reach score · triangles = HELD-OUT states")
sc = axD.scatter(P2[~holdout, 0], P2[~holdout, 1], c=reach_n[~holdout], cmap="cividis", **s_kw)
axD.scatter(P2[holdout, 0], P2[holdout, 1], c=reach_n[holdout], cmap="cividis",
            marker="^", s=8, alpha=0.9, linewidths=0.2, edgecolors="#E8E4D9")
plt.colorbar(sc, ax=axD).ax.tick_params(colors="#8B94A3", labelsize=7)

axE = fig.add_subplot(2, 3, 5); style(axE, "E · plans as trajectories: unboxed → boxed → mate")
axE.scatter(P2[:, 0], P2[:, 1], c=feats['box_area'], cmap="viridis_r", s=2, alpha=0.25, linewidths=0)
tcolors = ["#F0A83C", "#FF6B6B", "#7FC97F"]
for (path, res), col in zip(trajs, tcolors):
    pts = P2[path]
    axE.plot(pts[:, 0], pts[:, 1], "-", color=col, lw=1.8, alpha=0.95)
    axE.scatter(pts[0, 0], pts[0, 1], color=col, s=60, marker="o", zorder=5, edgecolors="#10151C")
    axE.scatter(pts[-1, 0], pts[-1, 1], color=col, s=90, marker="*" if res == "mate" else "X",
                zorder=5, edgecolors="#10151C")
    for i in range(len(pts) - 1):
        axE.annotate("", xy=pts[i+1], xytext=pts[i],
                     arrowprops=dict(arrowstyle="->", color=col, lw=1.1, alpha=0.8))

axF = fig.add_subplot(2, 3, 6); style(axF, "F · the boxing transition, per white move")
for (path, res), col in zip(trajs, tcolors):
    box = feats['box_area'][path]
    rch = reach_n[path]
    axF.plot(box, "-o", color=col, ms=3.5, lw=1.6, label=f"box area ({res})")
    axF.plot(rch / max(reach_n.max(), 1e-9) * 20, "--", color=col, lw=1.1, alpha=0.7)
axF.set_xlabel("white move #", color="#8B94A3", fontsize=9)
axF.set_ylabel("box area (solid) · reach, scaled (dashed)", color="#8B94A3", fontsize=9)
axF.legend(fontsize=7, facecolor="#171E27", labelcolor="#E8E4D9", edgecolor="#2A3542")

fig.suptitle("Atlas of the learned space — neural F embedding (trained on random play, 15% of states never seen)",
             color="#E8E4D9", fontsize=13, y=0.99)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig("/mnt/user-data/outputs/milestone1/atlas.png", dpi=140, facecolor="#10151C")
print("wrote atlas.png")
