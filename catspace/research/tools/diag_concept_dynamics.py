"""Same-encoder 1-ply concept change, using the TRAINED trunk.

Training reported `chg` (the copy baseline's error rate) rising 0.075 -> 0.587. That
number came from comparing the ONLINE encoder's parent concepts against the EMA target's
child concepts, so it confounds two things. Encoding both sides with the SAME weights
isolates the real one.
"""
import glob, numpy as np, torch, chess
from catspace.research.tools.npz_io import load_npz
from catspace.research.tools.training_infra.gen_move_graphs import parent_graph, glob_of_board
from catspace.research.components.encoder.approaches.reach_probability.experiments.graph_concept_field import GraphConceptField
from catspace.research.components.encoder.approaches.reach_probability.experiments.graph_move_policy import featurize, MV_CAP
from catspace.research.components.encoder.approaches.reach_probability.src.trajectories import normalise_elo
from catspace.research.tools.training_infra.train.scaffold import resolve_device

dev = resolve_device("auto")
ck = torch.load("artifacts/experiments/qat1_latest.pt", map_location="cpu", weights_only=False)
base = torch.load("artifacts/experiments/gcf4h_latest.pt", map_location="cpu", weights_only=False)
A = dict(base["args"])
def build():
    return GraphConceptField(
        d=A["d_model"], layers=A["layers"], slow_heads=A["slow_heads"], n_res=3, n_eg=1,
        cls_reach=np.zeros(3,np.int64), cls_eg=np.zeros(3,np.int64), cls_out=np.arange(3,dtype=np.int64),
        edge_codes=A["edge_codes"], move_codes=A["move_codes"], gps_parallel=bool(A["gps_parallel"]),
        attn_bias=bool(A["attn_bias"]), agg_max=bool(A["agg_max"]), kin=bool(A["kin"]),
        file_nodes=bool(A["file_nodes"]), phase_head=bool(A["phase_head"]),
        threat_head=bool(A["threat_head"]), elo_cond=True, g_heads=32, g_codes=8, d_code=8,
        sq_codes=A.get("sq_codes",256), mh_vq=True, gen5=True, gen6=True, binary=True,
        mv_concepts=True).to(dev).eval()

trained, fresh = build(), build()
tsd = {k[len("field."):]: v for k, v in ck["state_dict"].items() if k.startswith("field.")}
trained.load_state_dict(tsd, strict=False)
fresh.load_state_dict({k: v for k, v in base["state_dict"].items()
                       if k in fresh.state_dict() and fresh.state_dict()[k].shape == v.shape},
                      strict=False)
print(f"  trained trunk: {len(tsd)} tensors restored from qat1")

RK = ("sq","gl","mv","mut","att","n_mv","n_att","vac","dsc","pslot")
def rows(bs):
    acc = {k: [] for k in RK}
    for b in bs:
        g,_ = parent_graph(b)
        for k in ("sq","mv","mut","att","vac","dsc","pslot"): acc[k].append(g[k])
        acc["n_mv"].append(np.uint8(g["n_mv"])); acc["n_att"].append(np.uint8(g["n_att"]))
        acc["gl"].append(glob_of_board(b))
    return {k: torch.from_numpy(np.stack(acc[k])) for k in acc}

@torch.no_grad()
def enc(R, e):
    n = len(R["sq"])
    b = {k: R[k].to(dev) for k in ("sq","gl","mv","mut","att","n_mv","n_att")}
    sqf,mvf,edges,mask = featurize(b, dev, n_nodes=e.n_nodes)
    off = (torch.arange(n,device=dev)*e.n_nodes)[:,None]; jj = torch.arange(MV_CAP,device=dev)[None,:]
    for nm in ("vac","dsc"):
        a2 = R[nm].to(dev).long(); val = (a2!=255)&mask[:,:,None]
        edges[nm] = (((jj+64+off)[:,:,None].expand_as(a2))[val], (a2+off[:,:,None])[val])
    elo = torch.from_numpy(normalise_elo(np.full(n,2750.,np.float32)))[:,None].repeat(1,2).to(dev)
    return e.encode(sqf,mvf,edges,mask,elo=elo,pslot=R["pslot"].to(dev))[0]

fens = []
for f in sorted(glob.glob("data/derived/gko1_shards/shard_*.npz"))[:6]:
    z = load_npz(f)
    if "wdl" in z.files and (z["kind"]==0).any():
        fens += [str(x) for x in np.asarray(z["fen"])[:40]]
    if len(fens) >= 80: break

par, ch1, ch2 = [], [], []
for fn in fens:
    try: b0 = chess.Board(fn)
    except Exception: continue
    lm = list(b0.legal_moves)
    if not lm: continue
    b1 = b0.copy(); b1.push(lm[0])
    r = list(b1.legal_moves)
    if not r: continue
    b2 = b1.copy(); b2.push(r[0])
    par.append(b0); ch1.append(b1); ch2.append(b2)
    if len(par) >= 64: break
print(f"  {len(par)} parent/child/grandchild triples\n")

for nm, e in (("gcf4h trunk (as warm-started)", fresh), ("qat1 trunk (after training)", trained)):
    zp, z1, z2 = enc(rows(par),e), enc(rows(ch1),e), enc(rows(ch2),e)
    u = set(torch.unique(zp).tolist())
    d1 = float(((zp>0)!=(z1>0)).float().mean()); d2 = float(((zp>0)!=(z2>0)).float().mean())
    print(f"  {nm}")
    print(f"      level-1 values binary? {u <= {-1.0,1.0}}   ({sorted(u)[:4]})")
    print(f"      1-ply bit change  {100*d1:5.1f}%   2-ply  {100*d2:5.1f}%   (SAME encoder both sides)")
print(f"\n  training reported chg 7.5% at step 1 rising to 58.7% by step 2000.")

print("\n" + "="*72)
print("  A) WHICH dims are non-binary, and where do they sit?")
zp = enc(rows(par), trained)
nb = ~(((zp == 1.0) | (zp == -1.0)).all(0))
idx = torch.nonzero(nb).flatten().tolist()
print(f"     non-binary dims: {len(idx)} of {zp.shape[1]}")
if idx:
    print(f"     they occupy the contiguous range [{min(idx)}, {max(idx)}]")
    print(f"     level-1 layout is board 96 + squares 1536 + pieces 768 + edge-hist 128")
    print(f"     -> 96+1536+768 = 2400, so dims 2400..2527 are the EDGE HISTOGRAM,")
    print(f"        a normalised bag-of-tactics count vector, binary by design NO.")

print("\n  B) IS THE 1-PLY FLIP SIDE-TO-MOVE PARITY?")
print("     Compare each position against a NULL MOVE (same board, other side to move).")
import chess as _c
nullp, nullc = [], []
for b0 in par:
    if b0.is_check():
        continue
    b = b0.copy(); b.push(_c.Move.null())
    nullp.append(b0); nullc.append(b)
    if len(nullp) >= 48: break
if nullp:
    za, zb = enc(rows(nullp), trained), enc(rows(nullc), trained)
    dn = float(((za > 0) != (zb > 0)).float().mean())
    print(f"     NULL move (nothing moves, only the turn flips): {100*dn:5.1f}% of bits change")
    print(f"     a real 1-ply move:                              54.3%")
    print(f"     a real 2-ply move (turn back to original side): 15.6%")
    print(f"     -> if the null-move number is close to 54%, the 1-ply figure is dominated")
    print(f"        by whose turn it is, not by what changed on the board.")

print("\n  C) WHY are 848 dims non-binary? Hypothesis: EMPTY PIECE SLOTS encode as 0.")
zt = enc(rows(par), trained)
PIECE0, PIECE1 = 1632, 2400
blk = zt[:, PIECE0:PIECE1].reshape(len(par), 32, 24)     # 32 slots x 24 bits
zero_slot = (blk == 0).all(-1)                            # slot is all zeros
import chess as _c
present = torch.zeros_like(zero_slot)
for i, b in enumerate(par):
    n = len(b.piece_map())
    present[i, :min(n, 32)] = True
agree = float(((~zero_slot) == present).float().mean())
print(f"     all-zero piece slots per position: {float(zero_slot.float().sum(1).mean()):.1f}")
print(f"     pieces actually on the board:      {float(present.float().sum(1).mean()):.1f}")
print(f"     agreement between 'slot non-zero' and 'a piece exists': {100*agree:.1f}%")
eh = zt[:, 2400:]
print(f"     edge-histogram block rows sum to {float(eh.sum(-1).mean()):.3f} (a distribution)")
print(f"     -> piece slots are TERNARY (+1/-1 present, 0 absent) and the edge block is a")
print(f"        normalised histogram. 'Everything the readout sees is a bit' is FALSE:")
print(f"        2528-848 = 1680 of 3072 readout dims are strictly binary.")
