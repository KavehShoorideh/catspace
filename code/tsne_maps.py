"""
tsne_maps.py — regenerate the KRkn maps with t-SNE (openTSNE).

Fit on a stratified subsample (won / drawn / near-mate / KRk stratum), then
TRANSFORM out-of-sample points (game trajectories, chute endpoints) into the
same map — the reason for openTSNE over vanilla t-SNE.
Caveat baked into captions: t-SNE preserves neighborhoods, not global
distances; cluster gaps are not metric.
"""
import numpy as np, time, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from openTSNE import TSNE
from krkn import KRKNChain, KN_ATT, black_moves
t0 = time.time()

uc = KRKNChain(verbose=False)
dtm = np.load("dtm_krkn.npy"); won = np.isfinite(dtm[:uc.n2])
scores = np.load("krkn_scores.npy")
F = np.load("krkn_F.npy")
mk = uc.move_kind; mp0 = uc.move_ptr[:-1]; op0 = uc.out_ptr[:-1]
out_counts = np.diff(uc.out_ptr)

dtm_full = np.full(uc.n, 1e6); dtm_full[:uc.nW] = np.where(np.isfinite(dtm), dtm, 1e6)
vf = dtm_full[uc.out_flat]
sm = np.maximum.reduceat(vf, op0)
B_opt = (np.minimum.reduceat(np.where(vf == np.repeat(sm, out_counts),
        np.arange(len(vf)), len(vf)), op0) - op0).astype(np.int32)

Fw = F[:uc.nW].astype(np.float32)
Fn = (Fw - Fw.mean(0)) / (Fw.std(0) + 1e-9)

# ---- stratified fit sample
rng = np.random.default_rng(0)
near = dtm[:uc.n2] <= 3
idx_won   = rng.choice(np.where(won & ~near)[0], 9000, replace=False)
idx_drawn = rng.choice(np.where(~won)[0], 7000, replace=False)
idx_near  = np.where(near)[0]
idx_near  = idx_near if len(idx_near) <= 3000 else rng.choice(idx_near, 3000, replace=False)
idx_krk   = uc.n2 + rng.choice(uc.n1, 3000, replace=False)
fit_idx = np.concatenate([idx_won, idx_drawn, idx_near, idx_krk])
print(f"fit sample: {len(fit_idx)} states")

tsne = TSNE(perplexity=40, initialization="pca", random_state=0,
            n_jobs=1, verbose=False)
emb = tsne.fit(Fn[fit_idx])
print(f"t-SNE fit done ({time.time()-t0:.0f}s)")

P = np.asarray(emb)
fit_won   = won[np.clip(fit_idx, 0, uc.n2-1)] & (fit_idx < uc.n2)
fit_drawn = (~won[np.clip(fit_idx, 0, uc.n2-1)]) & (fit_idx < uc.n2)
fit_near  = (dtm_full[fit_idx] <= 3) & (fit_idx < uc.n2)
fit_krk   = fit_idx >= uc.n2

# fork threat on fit sample (KRkn only)
def fork_threat(s):
    wk, wr, bk, bn = uc.W[s]
    for kind, pay in black_moves(wk, wr, bk, bn):
        if kind != 'm': continue
        t = pay[3]
        if wk in KN_ATT[t] and wr in KN_ATT[t]: return True
    return False
fit_fork = np.array([fork_threat(int(s)) if s < uc.n2 else False for s in fit_idx])
print(f"fork threat on sample ({time.time()-t0:.0f}s)")

# ---- games + chutes, transformed into the map
def play(start, cap=60):
    s = int(start); path = [s]; result = "unfinished"
    for _ in range(cap):
        a, b = mp0[s], uc.move_ptr[s+1]
        best, bv = a, -np.inf
        for mid in range(a, b):
            k = mk[mid]
            if k == 1: best, bv = mid, np.inf; break
            if k in (2, 3): continue
            v = float(np.min(scores[uc.outs_of(mid)]))     # minimax readout (v3.3)
            if v > bv: bv, best = v, mid
        k = mk[best]
        if k == 1: result = "mate"; break
        if k in (2, 3): result = "draw"; break
        nxt = int(uc.out_flat[op0[best] + B_opt[best]])
        if nxt == uc.DRAW_S: result = "draw (rook lost)"; break
        if nxt >= uc.nW: result = "draw"; break
        path.append(nxt); s = nxt
    return path, result

r5 = np.random.default_rng(5)
mid_won = np.where(won & (dtm[:uc.n2] >= 9) & (dtm[:uc.n2] <= 15))[0]
pw = None
for t_ in range(80):
    p_, r_ = play(int(mid_won[r5.integers(0, len(mid_won))]))
    if r_ == "mate": pw, rw = p_, r_; break
drawn_idx = np.where(~won)[0]
pd = None
for t_ in range(60):
    p_, r_ = play(int(drawn_idx[r5.integers(0, len(drawn_idx))]), cap=25)
    if len(p_) >= 4: pd, rd = p_, r_; break
if pd is None: pd, rd = play(int(drawn_idx[0]), cap=25)

chutes = []
r2 = np.random.default_rng(2)
while len(chutes) < 25:
    s = int(r2.integers(0, uc.n2))
    for mid in uc.moves_of(s):
        if mk[mid] != 0: continue
        outs = uc.outs_of(mid)
        kk = outs[(outs >= uc.n2) & (outs < uc.nW)]
        if len(kk): chutes.append((s, int(kk[0]))); break

extra = sorted(set(pw) | set(pd) | {a for a, _ in chutes} | {b for _, b in chutes})
extra_pos = {s: i for i, s in enumerate(extra)}
E = np.asarray(emb.transform(Fn[np.array(extra)]))
print(f"out-of-sample transform of {len(extra)} points ({time.time()-t0:.0f}s)")
def pt(s): return E[extra_pos[s]]

# ---- figure 1: the 4-panel region map, t-SNE edition
fig, axes = plt.subplots(2, 2, figsize=(15.5, 12.8), facecolor="#10151C")
def style(ax, title):
    ax.set_facecolor("#171E27"); ax.set_title(title, color="#E8E4D9", fontsize=10.5)
    ax.set_xticks([]); ax.set_yticks([])
    for spn in ax.spines.values(): spn.set_color("#2A3542")

axA = axes[0, 0]; style(axA, "A · WIN/DRAW truth vs the learned field (t-SNE of F; neighborhoods faithful,\ninter-cluster distances not metric) — color intensity = learned reach")
rch = scores[fit_idx]
rch_n = (rch - rch.min()) / (rch.max() - rch.min() + 1e-9)
axA.scatter(P[fit_drawn, 0], P[fit_drawn, 1], s=3, c="#6B7280", alpha=.4, linewidths=0, label="drawn (truth)")
axA.scatter(P[fit_won, 0], P[fit_won, 1], s=3, c=plt.cm.YlOrBr(0.25 + 0.75*rch_n[fit_won]), alpha=.6, linewidths=0)
axA.scatter([], [], s=15, c="#F0A83C", label="won (truth), brighter = higher learned reach")
axA.legend(fontsize=8, facecolor="#171E27", labelcolor="#E8E4D9", edgecolor="#2A3542", loc="lower right")

axB = axes[0, 1]; style(axB, "B · FORK DANGER territory — does it form its own island?")
axB.scatter(P[~fit_fork, 0], P[~fit_fork, 1], s=2.5, c="#3B4656", alpha=.35, linewidths=0)
axB.scatter(P[fit_fork, 0], P[fit_fork, 1], s=4, c="#FF6B6B", alpha=.7, linewidths=0,
            label=f"knight fork available next move")
axB.legend(fontsize=8, facecolor="#171E27", labelcolor="#E8E4D9", edgecolor="#2A3542", loc="lower right")

axC = axes[1, 0]; style(axC, "C · STRATA: KRkn (teal) vs KRk (rust) — clean separation expected;\narrows = knight-capture chutes (out-of-sample transformed)")
axC.scatter(P[~fit_krk, 0], P[~fit_krk, 1], s=2.5, c="#4EC9B0", alpha=.3, linewidths=0, label="KRkn stratum")
axC.scatter(P[fit_krk, 0], P[fit_krk, 1], s=3.5, c="#C97B4E", alpha=.6, linewidths=0, label="KRk stratum")
for a, b in chutes:
    axC.annotate("", xy=pt(b), xytext=pt(a),
                 arrowprops=dict(arrowstyle="->", color="#8B94A3", lw=0.9, alpha=0.55))
axC.legend(fontsize=8, facecolor="#171E27", labelcolor="#E8E4D9", edgecolor="#2A3542", loc="lower right")

axD = axes[1, 1]; style(axD, f"D · trajectories transformed into the map: won start → {rw} ({len(pw)} mv)\ndrawn start vs optimal → {rd} ({len(pd)} mv)")
axD.scatter(P[fit_won, 0], P[fit_won, 1], s=2, c="#4EC9B0", alpha=.15, linewidths=0)
axD.scatter(P[fit_drawn, 0], P[fit_drawn, 1], s=2, c="#6B7280", alpha=.18, linewidths=0)
axD.scatter(P[fit_krk, 0], P[fit_krk, 1], s=2, c="#C97B4E", alpha=.25, linewidths=0)
axD.scatter(P[fit_near, 0], P[fit_near, 1], s=5, c="#F0A83C", alpha=.7, linewidths=0, label="DTM≤3 (retired goal region)")
for path, res, col in ((pw, rw, "#7FC97F"), (pd, rd, "#B39DDB")):
    pts = np.stack([pt(s) for s in path])
    axD.plot(pts[:, 0], pts[:, 1], "-", color=col, lw=2.0, alpha=.95)
    for i in range(len(pts) - 1):
        axD.annotate("", xy=pts[i+1], xytext=pts[i],
                     arrowprops=dict(arrowstyle="->", color=col, lw=1.2, alpha=.85))
    axD.scatter(*pts[0], color=col, s=90, marker="o", zorder=5, edgecolors="#10151C")
    axD.scatter(*pts[-1], color=col, s=140, marker="*" if res == "mate" else "X", zorder=5, edgecolors="#10151C")
axD.legend(fontsize=8, facecolor="#171E27", labelcolor="#E8E4D9", edgecolor="#2A3542", loc="lower right")

fig.suptitle("KRkn — t-SNE region map (openTSNE, perplexity 40, PCA init; out-of-sample points via transform)",
             color="#E8E4D9", fontsize=12.5, y=0.995)
plt.tight_layout(rect=[0, 0, 1, .97])
plt.savefig("/mnt/user-data/outputs/milestone1/krkn_region_map_tsne.png", dpi=135, facecolor="#10151C")
print(f"wrote krkn_region_map_tsne.png ({time.time()-t0:.0f}s)")

# ---- figure 2: goal region, t-SNE edition
fig2, ax = plt.subplots(figsize=(8.5, 6.8), facecolor="#10151C")
style(ax, "The (retired) goal region G = {DTM≤3} in the t-SNE map — condensation intact,\nboomerang gone with the linear projection (neighborhoods faithful; gaps not metric)")
ax.scatter(P[fit_won & ~fit_near, 0], P[fit_won & ~fit_near, 1], s=2.5, c="#4EC9B0", alpha=.3, linewidths=0, label="won")
ax.scatter(P[fit_drawn, 0], P[fit_drawn, 1], s=2.5, c="#6B7280", alpha=.3, linewidths=0, label="drawn")
ax.scatter(P[fit_krk, 0], P[fit_krk, 1], s=2.5, c="#C97B4E", alpha=.35, linewidths=0, label="KRk stratum")
ax.scatter(P[fit_near, 0], P[fit_near, 1], s=10, c="#F0A83C", alpha=.9, linewidths=.3,
           edgecolors="#10151C", label="DTM≤3")
ax.legend(fontsize=9, facecolor="#171E27", labelcolor="#E8E4D9", edgecolor="#2A3542", loc="lower right")
plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/milestone1/goal_region_tsne.png", dpi=140, facecolor="#10151C")
print(f"wrote goal_region_tsne.png ({time.time()-t0:.0f}s)")
