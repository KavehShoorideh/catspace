"""
experiment.py — Milestone 1: does random-play data yield a legible, useful
cone embedding at toy scale?

Outputs: results printed + figures + filmstrip markdown into ./out/
Pre-registered gate G-M1:
  (a) rank-64 relative REACH error (to the near-mate region) < 5%
  (b) >=3 embedding dims with |spearman rho| > 0.5 to distinct ground-truth concepts
  (c) engine mate-rate >= 5x random baseline within 100 plies (vs random black)
"""
import os, time, numpy as np
import scipy.stats as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from domain import enumerate_states, compute_dtm, concept_features
from learn import (Chain, randomized_svd_sm, fb_from_svd, rank_error,
                   sm_matvec, reach_scores)

GAMMA = 0.92
BIG = 1e6
OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(7)

print("== setup ==")
t0 = time.time()
ch = Chain()
dtm_w, dtm_b = compute_dtm(ch.W, ch.B)
feats = concept_features(ch.W, dtm_w)
P_exact = ch.exact_P_uniform()
print(f"setup {time.time()-t0:.1f}s | W={ch.nW}")

# near-mate region: DTM<=3 W-states plus the MATE absorbing state
region = list(np.where(dtm_w <= 3)[0]) + [ch.MATE_S]
region = np.array(region)
print(f"near-mate region size: {len(region)}")

def exact_reach(P):
    """Exact reach-to-region column: M @ 1_G."""
    e = np.zeros((P.shape[0], 1)); e[region] = 1.0
    return sm_matvec(P, e, GAMMA).ravel()

reach_true = exact_reach(P_exact)

# ---------- (1) RANK PROBE on exact M ----------
print("\n== rank probe (exact dynamics) ==")
rank_results = {}
for d in (8, 16, 32, 64, 128):
    U, S, V = randomized_svd_sm(P_exact, GAMMA, d=d, seed=0)
    F, Bm = fb_from_svd(U, S, V)
    fro = rank_error(P_exact, GAMMA, F, Bm, n_probe=8)
    r_hat = reach_scores(F, Bm, region)
    reach_rel = np.linalg.norm(r_hat - reach_true) / np.linalg.norm(reach_true)
    rho = st.spearmanr(r_hat[:ch.nW], reach_true[:ch.nW]).statistic
    rank_results[d] = (fro, reach_rel, rho)
    print(f"d={d:4d}  frobenius={fro:.3f}  reach_rel_err={reach_rel:.4f}  reach_spearman={rho:.4f}")

plt.figure(figsize=(6,4))
ds = sorted(rank_results)
plt.plot(ds, [rank_results[d][0] for d in ds], "o-", label="full-M Frobenius (rel)")
plt.plot(ds, [rank_results[d][1] for d in ds], "s-", label="reach-to-region (rel)")
plt.xlabel("rank d"); plt.ylabel("relative error"); plt.yscale("log")
plt.title("Rank probe: successor measure, 5x5 KRK"); plt.legend(); plt.grid(alpha=.3)
plt.tight_layout(); plt.savefig(f"{OUT}/rank_probe.png", dpi=120)

# ---------- (2) LEARNING CURVE from random-play games ----------
print("\n== learning from random play ==")
D_LEARN = 64
game_counts = [500, 2000, 8000, 32000]
learned = {}
for ng in game_counts:
    tr = ch.sample_games(ng, seed=11)
    Phat, visited = ch.empirical_P(tr)
    U, S, V = randomized_svd_sm(Phat, GAMMA, d=D_LEARN, seed=0)
    F, Bm = fb_from_svd(U, S, V)
    r_hat = reach_scores(F, Bm, region)
    cov = visited[:ch.nW].mean()
    vis = visited[:ch.nW]
    rho = st.spearmanr(r_hat[:ch.nW][vis], reach_true[:ch.nW][vis]).statistic
    learned[ng] = dict(F=F, B=Bm, visited=visited, coverage=cov, rho=rho,
                       n_tr=len(tr))
    print(f"games={ng:6d}  transitions={len(tr):7d}  state-coverage={cov:.2%}  "
          f"reach_spearman(visited)={rho:.4f}")

# exact-dynamics reference embedding ("infinite data")
U, S, V = randomized_svd_sm(P_exact, GAMMA, d=D_LEARN, seed=0)
F_ex, B_ex = fb_from_svd(U, S, V)

# ---------- (3) EMERGENT CONCEPTS: spectral dims vs ground truth ----------
print("\n== emergent concepts (spectral dims vs ground-truth features) ==")
Fn = F_ex[:ch.nW]  # exact embedding for the concept audit (cleanest claim: structure of the *dynamics*)
concept_names = ["dtm", "kk_dist", "bk_edge", "box_area", "rook_bk_dist"]
audit = np.zeros((16, len(concept_names)))
for j, cn in enumerate(concept_names):
    for dim in range(16):
        audit[dim, j] = st.spearmanr(Fn[:, dim], feats[cn]).statistic
best = {cn: (int(np.nanargmax(np.abs(audit[:, j]))), audit[np.nanargmax(np.abs(audit[:, j])), j])
        for j, cn in enumerate(concept_names)}
n_strong, used_dims = 0, set()
for cn, (dim, rho) in best.items():
    print(f"concept {cn:>12s}: best dim {dim:2d}  rho={rho:+.3f}")
    if abs(rho) > 0.5 and dim not in used_dims:
        n_strong += 1; used_dims.add(dim)
print(f"distinct dims with |rho|>0.5: {n_strong}")

plt.figure(figsize=(7,4.5))
plt.imshow(np.abs(audit), aspect="auto", cmap="viridis")
plt.colorbar(label="|spearman rho|")
plt.yticks(range(16)); plt.xticks(range(len(concept_names)), concept_names, rotation=20)
plt.xlabel("ground-truth concept (evaluation only)"); plt.ylabel("embedding dim (F)")
plt.title("Concept audit: nothing below was used in training")
plt.tight_layout(); plt.savefig(f"{OUT}/concept_audit.png", dpi=120)

# ---------- (4) VQ plan tokens ----------
print("\n== VQ plan tokens (k-means on cone shapes F) ==")
def kmeans(X, K, iters=50, seed=5):
    r = np.random.default_rng(seed)
    C = X[r.choice(len(X), K, replace=False)].copy()
    for _ in range(iters):
        d2 = ((X[:, None, :] - C[None, :, :]) ** 2).sum(-1)
        a = d2.argmin(1)
        for k in range(K):
            m = a == k
            if m.any(): C[k] = X[m].mean(0)
    return C, a

Xf = Fn / (np.linalg.norm(Fn, axis=1, keepdims=True) + 1e-9)
K = 32
C, assign = kmeans(Xf[:, :16], K)   # cluster on leading dims
usage = np.bincount(assign, minlength=K) / len(assign)
perp = np.exp(-(usage[usage > 0] * np.log(usage[usage > 0])).sum())
print(f"K={K}  codes used={np.sum(usage>0)}  usage perplexity={perp:.1f}")
print("code | size |  mean DTM | mean box_area")
order = np.argsort([feats['dtm'][assign == k].mean() if (assign == k).any() else 99 for k in range(K)])
for k in order[:10]:
    m = assign == k
    print(f"{k:4d} | {m.sum():4d} | {feats['dtm'][m].mean():8.2f} | {feats['box_area'][m].mean():8.2f}")

# ---------- (5) ENGINE + evaluation ----------
print("\n== engine evaluation (vs uniform-random black, 100-ply cap) ==")
def play(policy, n_games=1000, cap=100, seed=21):
    r = np.random.default_rng(seed)
    starts = r.integers(0, ch.nW, size=n_games)
    mates = 0; ply_counts = []
    for g in range(n_games):
        s = int(starts[g])
        for ply in range(cap):
            mv = policy(s, r)
            outcomes = ch.moves[s][mv]
            nxt = int(outcomes[r.integers(0, len(outcomes))])
            if nxt == ch.MATE_S:
                mates += 1; ply_counts.append(ply + 1); break
            if nxt == ch.DRAW_S: break
            s = nxt
    return mates / n_games, (np.mean(ply_counts) if ply_counts else float("nan"))

def random_policy(s, r): return int(r.integers(0, len(ch.moves[s])))

def make_engine(F, Bm, visited=None):
    zG = Bm[region].sum(axis=0)
    def policy(s, r):
        best, bestv = 0, -np.inf
        for mi, outcomes in enumerate(ch.moves[s]):
            v = 0.0
            for o in outcomes:
                o = int(o)
                if o == ch.MATE_S: v += BIG
                elif o == ch.DRAW_S: v += 0.0
                else:
                    if visited is not None and not visited[o]: continue
                    v += float(F[o] @ zG)
            v /= len(outcomes)
            if v > bestv: bestv, best = v, mi
        return best
    return policy

def dtm_policy(s, r):
    # optimal ceiling: minimize DTM of resulting node
    best, bestv = 0, np.inf
    from domain import classify_b, MATE
    for mi, b in enumerate(ch.moves[s]):
        pass
    # use precomputed: choose move minimizing worst-case... use dtm_b via chain rebuild is costly;
    # approximate ceiling: pick move that mates now, else minimizes expected next dtm over black replies
    for mi, outcomes in enumerate(ch.moves[s]):
        if len(outcomes) == 1 and int(outcomes[0]) == ch.MATE_S: return mi
        vals = [0.0 if int(o) >= ch.nW else dtm_w[int(o)] for o in outcomes]
        v = np.mean([x if np.isfinite(x) else 60.0 for x in vals])
        if v < bestv: bestv, best = v, mi
    return best

base_rate, base_len = play(random_policy)
print(f"random white       : mate-rate={base_rate:.3f}  mean-plies={base_len:.1f}")
ceil_rate, ceil_len = play(dtm_policy)
print(f"DTM-guided ceiling : mate-rate={ceil_rate:.3f}  mean-plies={ceil_len:.1f}")

rates = {}
for ng in game_counts:
    L = learned[ng]
    pol = make_engine(L["F"], L["B"], L["visited"])
    rate, mlen = play(pol)
    rates[ng] = rate
    print(f"learned ({ng:6d} games, cov {L['coverage']:.0%}): "
          f"mate-rate={rate:.3f}  mean-plies={mlen:.1f}  (x{rate/max(base_rate,1e-9):.1f} vs random)")
ex_rate, ex_len = play(make_engine(F_ex, B_ex))
print(f"exact-dynamics ref : mate-rate={ex_rate:.3f}  mean-plies={ex_len:.1f}")

plt.figure(figsize=(6,4))
plt.axhline(base_rate, color="gray", ls="--", label="random baseline")
plt.axhline(ceil_rate, color="green", ls="--", label="DTM ceiling")
plt.plot(game_counts, [rates[n] for n in game_counts], "o-", label="learned engine")
plt.xscale("log"); plt.xlabel("training games (random play)"); plt.ylabel("mate rate vs random black")
plt.title("Learning curve: cone-steering engine"); plt.legend(); plt.grid(alpha=.3)
plt.tight_layout(); plt.savefig(f"{OUT}/learning_curve.png", dpi=120)

# ---------- (6) FILMSTRIP: watch it think ----------
print("\n== filmstrip ==")
L = learned[game_counts[-1]]
pol = make_engine(L["F"], L["B"], L["visited"])
zG = B_ex[region].sum(axis=0)
r = np.random.default_rng(4)
# pick a start with high DTM so there is a story
cand = np.where(dtm_w >= 13)[0]
s = int(cand[r.integers(0, len(cand))])
lines = ["| ply | move | plan token | DTM | box area | reach score |",
         "|---|---|---|---|---|---|"]
for ply in range(60):
    mv = pol(s, r)
    name = ch.move_names[s][mv]
    tok = int(assign[s])
    score = float(F_ex[s] @ zG)
    lines.append(f"| {ply+1} | {name} | #{tok} | {dtm_w[s]:.0f} | {feats['box_area'][s]:.0f} | {score:.4f} |")
    outcomes = ch.moves[s][mv]
    nxt = int(outcomes[r.integers(0, len(outcomes))])
    if nxt == ch.MATE_S: lines.append(f"| {ply+2} | — | **MATE** | 0 | — | — |"); break
    if nxt == ch.DRAW_S: lines.append(f"| {ply+2} | — | draw | — | — | — |"); break
    s = nxt
open(f"{OUT}/filmstrip.md", "w").write("\n".join(lines))
print("\n".join(lines[:12]))
print(f"... full filmstrip in out/filmstrip.md")

# ---------- GATE ----------
print("\n== GATE G-M1 ==")
ga = rank_results[64][1] < 0.05
gb = n_strong >= 3
gc = rates[game_counts[-1]] >= 5 * base_rate
print(f"(a) rank-64 reach error < 5%   : {'PASS' if ga else 'FAIL'} ({rank_results[64][1]:.4f})")
print(f"(b) >=3 concept dims |rho|>0.5 : {'PASS' if gb else 'FAIL'} ({n_strong})")
print(f"(c) engine >= 5x random        : {'PASS' if gc else 'FAIL'} "
      f"({rates[game_counts[-1]]:.3f} vs {base_rate:.3f})")
print("G-M1:", "PASS" if (ga and gb and gc) else "FAIL")
