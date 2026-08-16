"""probe_magnitude_collapse.py -- is latent/codebook magnitude mismatch SUFFICIENT to cause
codebook collapse? (2026-08-15)

Motivation: probe_codebook_health measured jqt8 (collapsed) at |z|/|c| = 3-11x and jqt6
(healthy) at 0.6-1.0x. This tests whether that ratio CAUSES collapse or merely accompanies it,
by forcing the ratio with a fixed gain on the encoder output.

MEASURED -- the hypothesis is REFUTED:
      gain  1x (jqt6-like)   100.0% usage   perplexity 220.6   ratio 1.01
      gain 10x (jqt8-like)   100.0% usage   perplexity 213.2   ratio 0.92
A 10x gain does not collapse anything: EMA simply grows the codebook to match the cloud, so
the ratio returns to ~1 on its own. Magnitude mismatch is a CONSEQUENCE of collapse (one code
wins, only it gets updated, the rest stay frozen at init radius), not a cause.

AND THE PROPOSED FIX WAS THE ONLY HARMFUL ROW:
      gain 10x + LayerNorm    15.6% usage   perplexity 36.4   ratio 7.21
Do NOT put a LayerNorm between the encoder and the quantizer. Pinning every latent to a fixed
radius removes exactly the freedom EMA uses to track the cloud. kmeans_init and
threshold_ema_dead_code were both neutral here (100% usage), because this harness does not
collapse in the first place -- it cannot discriminate cures it never needs.

    .venv/bin/python -m ...probe_magnitude_collapse
"""
import numpy as np, torch, torch.nn as nn, argparse, time
from vector_quantize_pytorch import VectorQuantize
from catspace.research.components.encoder.approaches.reach_probability.src import trajectories as T
import chess
from lczerolens import LczeroBoard
from catspace.research.components.encoder.approaches.reachability_field.src.field import ReachabilityField

ap = argparse.ArgumentParser()
ap.add_argument("--positions", type=int, default=12000); ap.add_argument("--games", type=int, default=600)
ap.add_argument("--steps", type=int, default=1200); ap.add_argument("--batch", type=int, default=512)
ap.add_argument("--codes", type=int, default=256)
a = ap.parse_args()

field = ReachabilityField(onnx="assets/engines/lc0/t1-256x10.onnx")
games = T.load_piecedown_games(a.games, seed=0); rng = np.random.default_rng(0)
fens, ys = [], []
per = max(1, a.positions // max(len(games), 1))
for g in games:
    res, ucis = g[1], g[2]
    b = chess.Board(g[5]) if len(g) > 5 and g[5] else chess.Board()
    cls = 0 if res == 1 else (2 if res == 0 else 1)
    keep = set(rng.choice(len(ucis), size=min(per, len(ucis)), replace=False).tolist())
    for t, u in enumerate(ucis):
        mv = chess.Move.from_uci(u)
        if mv not in b.legal_moves: break
        if t in keep: fens.append(b.fen()); ys.append(cls)
        b.push(mv)
    if len(fens) >= a.positions: break
fens, ys = fens[:a.positions], ys[:a.positions]
out = []
for i in range(0, len(fens), 256):
    planes = [LczeroBoard(f).to_input_tensor().numpy() for f in fens[i:i+256]]
    out.append(field.trunk_feats(planes).mean(dim=(2,3)).detach().cpu().numpy())
X = torch.from_numpy(np.concatenate(out)).float(); Y = torch.from_numpy(np.array(ys, np.int64))
X = (X - X.mean(0)) / X.std(0).clamp(min=1e-6)
print(f"[mag] {len(X):,} positions", flush=True)

# (label, gain on encoder output, kmeans_init, dead-code revival, LayerNorm before VQ)
CONFIGS = [
    ("gain   1x  (jqt6-like)",   1.0,  False, 0, False),
    ("gain   3x",                3.0,  False, 0, False),
    ("gain  10x  (jqt8-like)",  10.0,  False, 0, False),
    ("gain  10x + kmeans_init", 10.0,  True,  0, False),
    ("gain  10x + dead-revive", 10.0,  False, 2, False),
    ("gain  10x + LayerNorm",   10.0,  False, 0, True),
]
print(f"\n{'config':>26} {'usage%':>8} {'perplex':>9} {'|z|/|c|':>9} {'outcome':>9}")
for label, gain, km, dead, ln in CONFIGS:
    torch.manual_seed(0)
    d_in = X.shape[1]
    enc = nn.Sequential(nn.Linear(d_in,128), nn.GELU(), nn.Linear(128,8))
    post = nn.LayerNorm(8) if ln else nn.Identity()
    dec = nn.Sequential(nn.Linear(8,128), nn.GELU(), nn.Linear(128,d_in)); head = nn.Linear(8,3)
    q = VectorQuantize(dim=8, codebook_size=a.codes, decay=0.9, commitment_weight=0.25,
                       kmeans_init=km, threshold_ema_dead_code=dead)
    mods = nn.ModuleList([enc,dec,head,q,post]); opt = torch.optim.Adam(mods.parameters(), lr=3e-4)
    for _ in range(a.steps):
        i = torch.randint(0,len(X),(a.batch,))
        h = post(enc(X[i]) * gain)
        zq, idx, vql = q(h)
        loss = nn.functional.mse_loss(dec(zq), X[i]) + nn.functional.cross_entropy(head(zq), Y[i]) + vql.mean()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        ids, acc, zn = [], [], []
        for s in range(0, min(len(X),12000), 2048):
            h = post(enc(X[s:s+2048]) * gain); zq, idx, _ = q(h)
            ids.append(idx.reshape(-1)); acc.append(float((head(zq).argmax(-1)==Y[s:s+2048]).float().mean()))
            zn.append(float(h.norm(dim=-1).mean()))
        ids = torch.cat(ids); u,c = torch.unique(ids, return_counts=True); p = c.float()/c.sum()
        cb = q._codebook.embed.reshape(-1,8)
        ratio = np.mean(zn)/max(float(cb.norm(dim=-1).mean()),1e-9)
        print(f"{label:>26} {100*len(u)/a.codes:>7.1f}% {float(torch.exp(-(p*p.log()).sum())):>9.1f} "
              f"{ratio:>9.2f} {np.mean(acc):>8.1%}", flush=True)
