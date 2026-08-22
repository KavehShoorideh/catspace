import sys, time
import numpy as np, torch, chess
sys.path.insert(0, "/Users/kav/code/remote/github/catspace")
t0 = time.time()
from catspace.research.components.encoder.approaches.reach_probability.experiments \
    .graph_concept_field import GraphConceptField
from catspace.research.components.encoder.approaches.reach_probability.experiments \
    .graph_move_policy import featurize, MV_CAP
from catspace.research.components.encoder.approaches.reach_probability.src \
    .trajectories import normalise_elo
ckf = torch.load("artifacts/experiments/gcf4h_latest.pt", map_location="cpu", weights_only=False)
A = dict(ckf["args"])
ck = torch.load("artifacts/experiments/gko2_12k.pt", map_location="cpu", weights_only=False)
RA = ck["args"]
for k in ("g_heads", "g_codes", "d_code", "slow_heads", "sq_codes"):
    A[k] = RA[k]
f = GraphConceptField(
    d=A["d_model"], layers=A["layers"], slow_heads=A["slow_heads"], n_res=3, n_eg=1,
    cls_reach=np.zeros(3, np.int64), cls_eg=np.zeros(3, np.int64),
    cls_out=np.arange(3, dtype=np.int64), edge_codes=A["edge_codes"],
    move_codes=A["move_codes"], gps_parallel=bool(A["gps_parallel"]),
    attn_bias=bool(A["attn_bias"]), agg_max=bool(A["agg_max"]), kin=bool(A["kin"]),
    file_nodes=bool(A["file_nodes"]), phase_head=bool(A["phase_head"]),
    threat_head=bool(A["threat_head"]), elo_cond=True, g_heads=A["g_heads"],
    g_codes=A["g_codes"], d_code=A["d_code"], sq_codes=A["sq_codes"], mh_vq=True).eval()
f.load_state_dict({k[6:]: v for k, v in ck["state_dict"].items() if k.startswith("field.")})
CH = np.load("data/derived/gko1_chain_graphs_mm.npz", mmap_mode="r")
fens = np.load("data/derived/gko1_chain_fens.npz")["fen"]
rng = np.random.default_rng(13)
rows = np.sort(rng.choice(len(fens), 512, replace=False))
b = {k: torch.from_numpy(np.ascontiguousarray(CH[k][rows])) for k in
     ("sq", "gl", "mv", "mut", "att", "n_mv", "n_att")}
sqf, mvf, edges, mask = featurize(b, "cpu", n_nodes=f.n_nodes)
off = (torch.arange(len(rows)) * f.n_nodes)[:, None]; jj = torch.arange(MV_CAP)[None, :]
for name in ("vac", "dsc"):
    a2 = torch.from_numpy(np.ascontiguousarray(CH[name][rows])).long()
    val = (a2 != 255) & mask[:, :, None]
    edges[name] = (((jj + 64 + off)[:, :, None].expand_as(a2))[val], (a2 + off[:, :, None])[val])
elo = torch.from_numpy(normalise_elo(np.full((len(rows), 2), 1800.0, np.float32)))
with torch.no_grad():
    _, _, _, _, h_mv, _ = f.encode(sqf, mvf, edges, mask, elo=elo)[:6]
    pre = f.mv_vq_proj(h_mv)                       # (B, 48, 32) pre-VQ
    _, codes, _ = f.mv_vq(pre)
X, C, chk, cap = [], [], [], []
for i, r in enumerate(rows):
    bd = chess.Board(str(fens[r]))
    for j, m in enumerate(list(bd.legal_moves)[:MV_CAP]):
        X.append(pre[i, j].numpy()); C.append(int(codes[i, j]))
        chk.append(bd.gives_check(m)); cap.append(bd.is_capture(m))
X = np.array(X); C = np.array(C); chk = np.array(chk); cap = np.array(cap)
n = len(X); print(f"[cap] {n:,} moves | check {chk.mean():.3f} capture {cap.mean():.3f}")

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
tr = rng.random(n) < 0.7
# 1) probe gap: pre-VQ embedding vs discrete code
p1 = LogisticRegression(max_iter=2000).fit(X[tr], chk[tr])
auc_pre = roc_auc_score(chk[~tr], p1.predict_proba(X[~tr])[:, 1])
oh = np.eye(64)[C]
p2 = LogisticRegression(max_iter=2000).fit(oh[tr], chk[tr])
auc_code = roc_auc_score(chk[~tr], p2.predict_proba(oh[~tr])[:, 1])
print(f"[cap] PROBE GAP (gives_check): pre-VQ AUC {auc_pre:.3f} vs code AUC {auc_code:.3f} "
      f"-> gap {auc_pre-auc_code:+.3f}")

# 2) within-code bimodality inside capture codes: does check split code 21/63?
from sklearn.cluster import KMeans
for c in (21, 63, 50):
    m = C == c
    if m.sum() < 40: continue
    km = KMeans(2, n_init=5, random_state=0).fit(X[m])
    lab = km.labels_
    r0, r1 = chk[m][lab == 0].mean(), chk[m][lab == 1].mean()
    print(f"[cap] code {c} (n={int(m.sum())}, capture code): 2-means check-rates "
          f"{r0:.2f} vs {r1:.2f} {'<- SUBCLUSTER' if abs(r0-r1) > 0.15 else '(no split)'}")

# 3) residual by attribute
with torch.no_grad():
    q, _, _ = f.mv_vq(pre)
res = (pre - q).norm(dim=-1).numpy().reshape(-1)[:n*0+len(X)] if False else None
resid = np.array([(pre[i, j] - q[i, j]).norm().item()
                  for i, r in enumerate(rows)
                  for j, m in enumerate(list(chess.Board(str(fens[r])).legal_moves)[:MV_CAP])])
ck_cap = chk & cap
print(f"[cap] VQ residual: quiet {resid[~chk & ~cap].mean():.3f} | capture {resid[cap & ~chk].mean():.3f} "
      f"| check {resid[chk & ~cap].mean():.3f} | CHECK-CAPTURE {resid[ck_cap].mean():.3f} (n={int(ck_cap.sum())})")

# 5) offline rate-distortion: kmeans at 64/128/256 -> best check-capture cluster purity
for K in (64, 128, 256):
    km = KMeans(K, n_init=3, random_state=0).fit(X[::2])
    lab = km.predict(X)
    best = 0.0
    for k in range(K):
        m = lab == k
        if m.sum() > 15:
            best = max(best, ck_cap[m].mean())
    print(f"[cap] offline k={K}: best check-capture cluster purity {best:.2f}")
print(f"[{time.time()-t0:.0f}s]")
