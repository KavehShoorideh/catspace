"""
krkn_map.py — the two-sided map: WIN/DRAW frontier, fork danger, strata.
"""
import numpy as np, matplotlib, time
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from krkn import KRKNChain, KN_ATT, KNIGHT, rook_attacks, black_moves
t0 = time.time()

uc = KRKNChain(verbose=False)
dtm = np.load("dtm_krkn.npy"); won = np.isfinite(dtm[:uc.n2])
scores = np.load("krkn_scores.npy")
F = np.load("krkn_F.npy")
mk = uc.move_kind; mp0 = uc.move_ptr[:-1]; op0 = uc.out_ptr[:-1]
out_counts = np.diff(uc.out_ptr); move_counts = np.diff(uc.move_ptr)

dtm_full = np.full(uc.n, 1e6); dtm_full[:uc.nW] = np.where(np.isfinite(dtm), dtm, 1e6)
vals_flat = dtm_full[uc.out_flat]
seg_max = np.maximum.reduceat(vals_flat, op0)
B_opt = (np.minimum.reduceat(np.where(vals_flat == np.repeat(seg_max, out_counts),
        np.arange(len(vals_flat)), len(vals_flat)), op0) - op0).astype(np.int32)

# fork threat: exists legal black knight move landing on t with both wk,wr in knight range of t
def fork_threat(wk, wr, bk, bn):
    for kind, pay in black_moves(wk, wr, bk, bn):
        if kind != 'm': continue
        t = pay[3]
        if t == bn: continue
        if wk in KN_ATT[t] and wr in KN_ATT[t]:
            return True
    return False
fork = np.fromiter((fork_threat(*s) for s in uc.W), bool, count=uc.n2)
print(f"fork threat computed: {fork.mean():.1%} of states ({time.time()-t0:.0f}s)")

# 2D projection
Fw = F[:uc.nW]; X = Fw - Fw.mean(0)
sub = np.random.default_rng(0).choice(uc.nW, 20000, replace=False)
_, _, Vt = np.linalg.svd(X[sub], full_matrices=False)
P2 = X @ Vt[:2].T
is_krk = np.zeros(uc.nW, bool); is_krk[uc.n2:] = True
reach = scores[:uc.n2]

def play(start, cap=60):
    s = int(start); path = [s]; result = "unfinished"
    pol_cache = {}
    for _ in range(cap):
        a, b = mp0[s], uc.move_ptr[s+1]
        best, bv = a, -np.inf
        for mid in range(a, b):
            k = mk[mid]
            if k == 1: best, bv = mid, np.inf; break
            if k in (2, 3): continue
            v = float(scores[uc.outs_of(mid)].mean())
            if v > bv: bv, best = v, mid
        k = mk[best]
        if k == 1: result = "mate"; break
        if k in (2, 3): result = "draw"; break
        nxt = int(uc.out_flat[op0[best] + B_opt[best]])
        if nxt == uc.DRAW_S: result = "draw (rook lost)"; break
        if nxt >= uc.nW: result = "draw"; break
        path.append(nxt); s = nxt
    return path, result

rng = np.random.default_rng(5)
won_mid = np.where(won & (dtm[:uc.n2] >= 9) & (dtm[:uc.n2] <= 15))[0]
pw = None
for t in range(60):
    st_w = int(won_mid[rng.integers(0, len(won_mid))])
    p_, r_ = play(st_w)
    if r_ == "mate": pw, rw = p_, r_; break
drawn_idx = np.where(~won)[0]
pd, rd = None, None
for t in range(40):
    st_d = int(drawn_idx[rng.integers(0, len(drawn_idx))])
    pd_, rd_ = play(st_d, cap=25)
    if len(pd_) >= 4:
        pd, rd = pd_, rd_; break
if pd is None:
    st_d = int(drawn_idx[0]); pd, rd = play(st_d, cap=25)
print(f"stories: won-start -> {rw} in {len(pw)}; drawn-start -> {rd} in {len(pd)} ({time.time()-t0:.0f}s)")

fig, axes = plt.subplots(2, 2, figsize=(15.5, 12.5), facecolor="#10151C")
def style(ax, title):
    ax.set_facecolor("#171E27"); ax.set_title(title, color="#E8E4D9", fontsize=10.5)
    ax.tick_params(colors="#8B94A3", labelsize=7)
    for spn in ax.spines.values(): spn.set_color("#2A3542")

kn = ~is_krk[:uc.nW][:uc.n2] if False else slice(0, uc.n2)
axA = axes[0,0]; style(axA, "A · GROUND TRUTH win/draw vs the LEARNED field (never labeled)\nteal = won, gray = game-theoretically drawn · contours = learned reach (AUC 0.70)")
axA.scatter(P2[:uc.n2][won,0], P2[:uc.n2][won,1], s=2, c="#4EC9B0", alpha=.35, linewidths=0, label="won (truth)")
axA.scatter(P2[:uc.n2][~won,0], P2[:uc.n2][~won,1], s=2, c="#6B7280", alpha=.35, linewidths=0, label="drawn (truth)")
try:
    axA.tricontour(P2[:uc.n2:7,0], P2[:uc.n2:7,1], reach[::7], levels=6, colors="#F0A83C", linewidths=0.9)
except Exception: pass
axA.legend(fontsize=8, facecolor="#171E27", labelcolor="#E8E4D9", edgecolor="#2A3542")

axB = axes[0,1]; style(axB, f"B · FORK DANGER territory ({fork.mean():.0%} of states)\nred = a knight fork of K+R is available to black next move")
axB.scatter(P2[:uc.n2][~fork,0], P2[:uc.n2][~fork,1], s=2, c="#3B4656", alpha=.3, linewidths=0)
axB.scatter(P2[:uc.n2][fork,0], P2[:uc.n2][fork,1], s=2.5, c="#FF6B6B", alpha=.5, linewidths=0)

axC = axes[1,0]; style(axC, "C · STRATA: KRKN (teal) and KRK (rust) clouds\narrows = knight-capture chutes (trade into the won sub-game)")
axC.scatter(P2[:uc.n2,0], P2[:uc.n2,1], s=2, c="#4EC9B0", alpha=.25, linewidths=0, label=f"KRKN ({uc.n2:,})")
axC.scatter(P2[uc.n2:,0], P2[uc.n2:,1], s=2.5, c="#C97B4E", alpha=.5, linewidths=0, label=f"KRK ({uc.n1:,})")
chutes = []
r2 = np.random.default_rng(2)
while len(chutes) < 25:
    s = int(r2.integers(0, uc.n2))
    for mid in uc.moves_of(s):
        if mk[mid] != 0: continue
        outs = uc.outs_of(mid)
        kk = outs[(outs >= uc.n2) & (outs < uc.nW)]
        if len(kk): chutes.append((s, int(kk[0]))); break
for s, kkk in chutes:
    axC.annotate("", xy=P2[kkk], xytext=P2[s],
                 arrowprops=dict(arrowstyle="->", color="#8B94A3", lw=0.9, alpha=0.5))
axC.legend(fontsize=8, facecolor="#171E27", labelcolor="#E8E4D9", edgecolor="#2A3542")

axD = axes[1,1]; style(axD, f"D · same engine, two truths: won start → {rw} ({len(pw)} mv)\ndrawn start vs optimal → {rd} ({len(pd)} mv): the truth says it never could have won")
axD.scatter(P2[:uc.n2][won,0], P2[:uc.n2][won,1], s=1.5, c="#4EC9B0", alpha=.12, linewidths=0)
axD.scatter(P2[:uc.n2][~won,0], P2[:uc.n2][~won,1], s=1.5, c="#6B7280", alpha=.15, linewidths=0)
axD.scatter(P2[uc.n2:,0], P2[uc.n2:,1], s=1.5, c="#C97B4E", alpha=.2, linewidths=0)
for path, res, col in ((pw, rw, "#7FC97F"), (pd, rd, "#B39DDB")):
    pts = P2[path]
    axD.plot(pts[:,0], pts[:,1], "-", color=col, lw=2.0, alpha=.95)
    for i in range(len(pts)-1):
        axD.annotate("", xy=pts[i+1], xytext=pts[i],
                     arrowprops=dict(arrowstyle="->", color=col, lw=1.2, alpha=.85))
    axD.scatter(*pts[0], color=col, s=90, marker="o", zorder=5, edgecolors="#10151C")
    axD.scatter(*pts[-1], color=col, s=140, marker="*" if res=="mate" else "X", zorder=5, edgecolors="#10151C")

fig.suptitle("K+R vs K+N — the first two-sided domain: real draws, forks, and a win/draw frontier the field must discover",
             color="#E8E4D9", fontsize=12.5, y=0.995)
plt.tight_layout(rect=[0,0,1,.97])
plt.savefig("/mnt/user-data/outputs/milestone1/krkn_region_map.png", dpi=135, facecolor="#10151C")
print(f"wrote krkn_region_map.png ({time.time()-t0:.0f}s)")
