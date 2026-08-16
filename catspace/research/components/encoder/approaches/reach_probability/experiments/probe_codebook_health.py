"""probe_codebook_health.py -- per-head codebook health of a jqt checkpoint (2026-08-15).

TWO TRAPS this exists to avoid, both of which produced wrong answers on the first attempt:

  1. The jqt weights are in a SIDECAR (reach_<gen>_jqt.pt), not in _latest.pt. Building a
     JQTModule and reading its codebooks without loading that sidecar measures a RANDOM INIT
     and looks deceptively healthy.
  2. With geo_heads>0 the trained path is quantize_split (geo_enc/brd_enc). `jqt.enc` is a
     DEAD module in that configuration -- measuring it also looks healthy.

Reports per head: codebook radius, latent radius, their ratio, codes used, and perplexity.

MEASURED (4096 train positions):
  jqt6 (healthy)   ratio 0.63-1.03   16/16 codes   perplexity 11.1-15.3
  jqt8 (collapsed) ratio 3.15-11.26   1-2/16       perplexity 1.00-1.86

CAUTION -- the ratio is a SYMPTOM, not the cause. probe_magnitude_collapse.py drove the ratio
to 10x deliberately and usage stayed at 100%, because EMA lets the codebook grow with the
cloud as long as every code is still being assigned. Once one code wins everything, only that
code gets updated and the rest are frozen at their init radius; the spread you measure here is
what collapse LEAVES BEHIND.

    .venv/bin/python -m ...probe_codebook_health --gen jqt8
"""
import numpy as np, torch, argparse
from catspace.research.components.encoder.approaches.reach_probability.src.reach_vit_jepa import ReachViT
from catspace.research.components.encoder.approaches.reach_probability.experiments.jqt import JQTModule
from catspace.research.components.encoder.approaches.reach_probability.src import trajectories as T

ap = argparse.ArgumentParser(); ap.add_argument("--gen", default="jqt8"); a0 = ap.parse_args()
ck = torch.load(f"artifacts/experiments/reach_{a0.gen}_latest.pt", map_location="cpu", weights_only=False)
A = ck["args"]; A = vars(A) if not isinstance(A, dict) else A
net = ReachViT(split_head=bool(A["split_head"]), move_head=bool(A["move_head"]),
               proj_mlp=int(A.get("proj_mlp", 0)), d_model=A["d_model"], layers=A["layers"],
               heads=A["heads"], d=A["d"], hidden=A["hidden"], components=A["components"],
               ema_decay=A["ema"], dual=False)
net.attach_poles(torch.tensor(ck["cfg"]["pole_parent"]), n_sources=1,
                 fixed=not ck["cfg"].get("learned_poles"),
                 height=ck["cfg"].get("pole_height", 3.0))
r = net.load_state_dict(ck["state_dict"], strict=False)
assert not r.missing_keys, r.missing_keys[:5]
jq = JQTModule(geo_heads=A["geo_heads"], bank_n=A["bank_n"], d_model=A["d_model"],
               heads=A["jqt_heads"], codes=A["jqt_codes"], d=A["d"], ema=A["ema"],
               square_codes=A["jqt_square_codes"], piece_codes=A["jqt_piece_codes"])
if A.get("move_codes"): jq.add_move_codes(A["move_codes"])
jsd = torch.load(f"artifacts/experiments/reach_{a0.gen}_jqt.pt", map_location="cpu", weights_only=False)
jsd = jsd.get("state_dict", jsd) if isinstance(jsd, dict) and "state_dict" in jsd else jsd
r2 = jq.load_state_dict(jsd, strict=False)
print(f"[load] jqt sidecar: missing {len(r2.missing_keys)} unexpected {len(r2.unexpected_keys)}")
net.eval(); jq.eval()

_ne = A["games"] // 2
_nh, _ns = (0, A["games"]) if A.get("sf_only") else (_ne, _ne)
tr = T.build(n_human=_nh, n_sf=_ns, seed=A["seed"], cache=True,
             max_plies=A["max_plies"], n_piecedown=A["n_piecedown"])
rng = np.random.default_rng(0); idx = rng.choice(len(tr.tok), 4096, replace=False)
tok = torch.from_numpy(tr.tok[idx].astype(np.int64)); glb = torch.from_numpy(tr.glob[idx].astype(np.float32))

def stats(name, X):
    s = X.std(0)
    print(f"  {name:30s} dim={X.shape[1]:4d} per-dim std med {float(s.median()):7.4f} "
          f"max/med {float(s.max())/max(float(s.median()),1e-9):6.1f}x  mean|x| {float(X.norm(dim=-1).mean()):7.3f}")

with torch.no_grad():
    phi = net.backbone(tok, glb); zb = net.proj_b(phi)
    POL = net.poles.poles
    atl = jq.atlas(zb, net.dA, net.dB, POL[:3], A.get("basin_temp", 5.0))
    print(f"\n[{a0.gen}] step {ck.get('step')} --- TRAINED quantizer path ---")
    stats("phi (LayerNorm'd)", phi); stats("z_B -> geo_enc", zb); stats("atlas -> geo_enc", atl)
    hp, zqf, ids, vl, zqg, zqb = jq.quantize_split(phi, zb, atl)
    G = A["geo_heads"]
    stats("h_geo -> VQ", hp[:, :G].reshape(-1, jq.d_code))
    stats("h_brd -> VQ", hp[:, G:].reshape(-1, jq.d_code))
    print(f"\n  {'head':>6} {'|codebook|':>11} {'|latent|':>9} {'ratio':>7} {'used':>7} {'perplexity':>11}")
    for i in range(A["jqt_heads"]):
        cb = jq.vq[i]._codebook.embed.reshape(-1, jq.d_code)
        h_i = hp[:, i]
        u, c = torch.unique(ids[:, i], return_counts=True)
        p = c.float() / c.sum(); perp = float(torch.exp(-(p * p.log()).sum()))
        nc, nl = float(cb.norm(dim=-1).mean()), float(h_i.norm(dim=-1).mean())
        tag = "geo" if i < G else "brd"
        print(f"  {tag}{i:3d} {nc:11.4f} {nl:9.4f} {nl/max(nc,1e-9):7.2f} "
              f"{len(u):3d}/{cb.shape[0]:<3d} {perp:11.3f}")
