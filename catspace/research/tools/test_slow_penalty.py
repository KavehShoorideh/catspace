"""Does the --w-slow penalty actually REDUCE group churn when you minimise it?

qat5 trained with it and ended with MORE churn (46.4 groups vs qat4's 21.9). Two
explanations: the term is implemented wrong, or it is implemented right and simply lost to
the task losses. This separates them by optimising the penalty ALONE for a few steps and
watching the churn it is supposed to control.

Also compares the ONLINE-vs-ONLINE form against the ONLINE-vs-EMA form actually used in
training, which is the specific defect I suspected but did not verify.
"""
import copy, glob, numpy as np, torch, chess
from catspace.research.tools.npz_io import load_npz
from catspace.research.tools.concept_planner import build, Planner
from catspace.research.tools.training_infra.train.scaffold import resolve_device

dev = resolve_device("auto")
model, field, _, _ = build("artifacts/experiments/qat5_all_latest.pt", "qat", dev, field_only=True)
ck = torch.load("artifacts/experiments/qat5_all_latest.pt", map_location="cpu", weights_only=False)
fsd = {k[len("field."):]: v for k, v in ck["state_dict"].items() if k.startswith("field.")}
field.load_state_dict(fsd, strict=False)
field.train()
P = Planner(model, field, dev)

fens = []
for f in sorted(glob.glob("data/derived/gko1_shards/shard_*.npz"))[:4]:
    z = load_npz(f)
    if "wdl" in z.files and (z["kind"] == 0).any():
        fens += [str(x) for x in np.asarray(z["fen"])[:60]]
    if len(fens) >= 150: break
par, gc = [], []
for fn in fens:
    try: b0 = chess.Board(fn)
    except Exception: continue
    lm = list(b0.legal_moves)
    if not lm: continue
    b1 = b0.copy(); b1.push(lm[0])
    r = list(b1.legal_moves)
    if not r: continue
    b2 = b1.copy(); b2.push(r[0])
    par.append(b0); gc.append(b2)
    if len(par) >= 48: break

B0 = field.g_heads * field.code_width
S0 = B0 + 64 * field.sq_width
P0 = field.d_state - field.edge_codes
PW = max(1, (P0 - S0) // 32)

def penalty(z0, z2):
    n = len(z0)
    sq0 = z0[:, B0:S0].reshape(n, 64, field.sq_width)
    sq1 = z2[:, B0:S0].reshape(n, 64, field.sq_width)
    d_sq = ((sq0 - sq1) ** 2).sum(-1).clamp(min=1e-8).sqrt()
    zp = z0[:, S0:P0].reshape(n, 32, PW); tp = z2[:, S0:P0].reshape(n, 32, PW)
    alive = ((zp.abs().sum(-1) > 0) & (tp.abs().sum(-1) > 0)).float()
    d_pc = ((zp - tp) ** 2).sum(-1).clamp(min=1e-8).sqrt() * alive
    d_hi = 0.5 * (z0[:, P0:] - z2[:, P0:]).abs().sum(-1)
    ngrp = 64 + alive.sum(-1) + 1
    moved = ((d_sq > 1e-3).float().sum(-1) + (d_pc > 1e-3).float().sum(-1) + (d_hi > 1e-3).float())
    return ((d_sq.sum(-1) + d_pc.sum(-1) + d_hi) / ngrp).mean(), moved.mean()

def enc_grad(boards, enc):
    from catspace.research.components.encoder.approaches.reach_probability.experiments \
        .graph_move_policy import featurize, MV_CAP
    from catspace.research.components.encoder.approaches.reach_probability.src \
        .trajectories import normalise_elo
    from catspace.research.tools.training_infra.gen_move_graphs import parent_graph, glob_of_board
    acc = {k: [] for k in ("sq","gl","mv","mut","att","n_mv","n_att","vac","dsc","pslot")}
    for b in boards:
        g, _ = parent_graph(b)
        for k in ("sq","mv","mut","att","vac","dsc","pslot"): acc[k].append(g[k])
        acc["n_mv"].append(np.uint8(g["n_mv"])); acc["n_att"].append(np.uint8(g["n_att"]))
        acc["gl"].append(glob_of_board(b))
    R = {k: torch.from_numpy(np.stack(v)) for k, v in acc.items()}
    n = len(boards)
    bb = {k: R[k].to(dev) for k in ("sq","gl","mv","mut","att","n_mv","n_att")}
    sqf, mvf, edges, mask = featurize(bb, dev, n_nodes=enc.n_nodes)
    off = (torch.arange(n, device=dev) * enc.n_nodes)[:, None]
    jj = torch.arange(MV_CAP, device=dev)[None, :]
    for nm in ("vac","dsc"):
        a2 = R[nm].to(dev).long(); val = (a2 != 255) & mask[:, :, None]
        edges[nm] = (((jj+64+off)[:, :, None].expand_as(a2))[val], (a2+off[:, :, None])[val])
    elo = torch.from_numpy(normalise_elo(np.full(n, 2750., np.float32)))[:, None].repeat(1,2).to(dev)
    return enc.encode(sqf, mvf, edges, mask, elo=elo, pslot=R["pslot"].to(dev))[0]

for mode in ("ONLINE vs ONLINE", "ONLINE vs EMA (what training used)"):
    f2 = copy.deepcopy(field); f2.train()
    ema = copy.deepcopy(f2).eval()
    for p_ in ema.parameters(): p_.requires_grad_(False)
    opt = torch.optim.Adam(f2.parameters(), lr=3e-4)
    print(f"\n  === {mode} ===")
    for step in range(6):
        z0 = enc_grad(par, f2)
        if mode.startswith("ONLINE vs ONLINE"):
            z2 = enc_grad(gc, f2)
        else:
            with torch.no_grad():
                z2 = enc_grad(gc, ema)
        pen, moved = penalty(z0, z2)
        if step in (0, 5):
            print(f"    step {step}: penalty {float(pen):.4f}   groups moved {float(moved):.1f}/97")
        opt.zero_grad(); pen.backward(); opt.step()
        with torch.no_grad():
            for t, s_ in zip(ema.parameters(), f2.parameters()):
                t.mul_(0.999).add_(s_.detach(), alpha=0.001)
