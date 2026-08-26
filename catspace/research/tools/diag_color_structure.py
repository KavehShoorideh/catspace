"""Is the concept space COLOR-STRUCTURED, or does a turn flip smear across every dim?

Kaveh 2026-08-26: "is there a way for us to make sure that some of the concepts are for
white and some of them are for black so that we don't flip every turn?"

Whether that is buildable depends on a fact we can measure without training anything. A
null move changes nothing on the board and only hands over the turn, and it flips 54% of
concept bits. Two very different worlds produce that same 54%:

  STRUCTURED  each dim is either color-OWNED (flips on ~100% of null moves) or
              color-NEUTRAL (flips ~0%). The per-dim flip-rate histogram is bimodal at
              0 and 1. Then "these concepts are white's, those are black's" already
              almost holds, and binding it is a small nudge -- or free, by relabelling.

  CHAOTIC     every dim flips on roughly half of null moves. The histogram piles up at
              0.5. Then no assignment of dims to colors exists to be enforced, and the
              only honest route is an architectural change to the trunk.

This script decides which, per concept family, and additionally asks whether the flip is
a fixed PERMUTATION (dim i reliably becomes dim j), because a permutation can be undone
analytically -- no retraining, no loss term, just relabel the axes.
"""
import glob, numpy as np, torch, chess
from catspace.research.tools.npz_io import load_npz
from catspace.research.tools.training_infra.gen_move_graphs import parent_graph, glob_of_board
from catspace.research.components.encoder.approaches.reach_probability.experiments.graph_concept_field import GraphConceptField
from catspace.research.components.encoder.approaches.reach_probability.experiments.graph_move_policy import featurize, MV_CAP
from catspace.research.components.encoder.approaches.reach_probability.src.trajectories import normalise_elo
from catspace.research.tools.training_infra.train.scaffold import resolve_device

dev = resolve_device("auto")
base = torch.load("artifacts/experiments/gcf4h_latest.pt", map_location="cpu", weights_only=False)
A = dict(base["args"])
field = GraphConceptField(
    d=A["d_model"], layers=A["layers"], slow_heads=A["slow_heads"], n_res=3, n_eg=1,
    cls_reach=np.zeros(3,np.int64), cls_eg=np.zeros(3,np.int64), cls_out=np.arange(3,dtype=np.int64),
    edge_codes=A["edge_codes"], move_codes=A["move_codes"], gps_parallel=bool(A["gps_parallel"]),
    attn_bias=bool(A["attn_bias"]), agg_max=bool(A["agg_max"]), kin=bool(A["kin"]),
    file_nodes=bool(A["file_nodes"]), phase_head=bool(A["phase_head"]),
    threat_head=bool(A["threat_head"]), elo_cond=True, g_heads=32, g_codes=8, d_code=8,
    sq_codes=A.get("sq_codes",256), mh_vq=True, gen5=True, gen6=True, binary=True,
    mv_concepts=True).to(dev).eval()
ck = torch.load("artifacts/experiments/qat1_latest.pt", map_location="cpu", weights_only=False)
field.load_state_dict({k[len("field."):]: v for k,v in ck["state_dict"].items()
                       if k.startswith("field.")}, strict=False)

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
def enc(R):
    n = len(R["sq"])
    b = {k: R[k].to(dev) for k in ("sq","gl","mv","mut","att","n_mv","n_att")}
    sqf,mvf,edges,mask = featurize(b, dev, n_nodes=field.n_nodes)
    off = (torch.arange(n,device=dev)*field.n_nodes)[:,None]; jj = torch.arange(MV_CAP,device=dev)[None,:]
    for nm in ("vac","dsc"):
        a2 = R[nm].to(dev).long(); val = (a2!=255)&mask[:,:,None]
        edges[nm] = (((jj+64+off)[:,:,None].expand_as(a2))[val], (a2+off[:,:,None])[val])
    elo = torch.from_numpy(normalise_elo(np.full(n,2750.,np.float32)))[:,None].repeat(1,2).to(dev)
    return field.encode(sqf,mvf,edges,mask,elo=elo,pslot=R["pslot"].to(dev))[0]

fens = []
for f in sorted(glob.glob("data/derived/gko1_shards/shard_*.npz"))[:10]:
    z = load_npz(f)
    if "wdl" in z.files and (z["kind"]==0).any():
        fens += [str(x) for x in np.asarray(z["fen"])[:80]]
    if len(fens) >= 400: break
par, nul = [], []
for fn in fens:
    try: b0 = chess.Board(fn)
    except Exception: continue
    if b0.is_check(): continue          # a null move is illegal in check
    b1 = b0.copy(); b1.push(chess.Move.null())
    par.append(b0); nul.append(b1)
    if len(par) >= 256: break

zp, zn = enc(rows(par)), enc(rows(nul))
flip = ((zp > 0) != (zn > 0)).float().mean(0).cpu().numpy()      # per-dim flip rate
print(f"  {len(par)} positions, null move only (nothing moves, turn changes hands)\n")

FAM = [("board",  0,   96), ("squares", 96, 1632),
       ("pieces", 1632, 2400), ("edge-hist", 2400, 2528)]
print(f"  {'family':12s} {'dims':>6s} {'mean flip':>10s} | {'~never':>7s} {'~always':>8s} {'middling':>9s}")
print(f"  {'':12s} {'':>6s} {'':>10s} | {'<0.05':>7s} {'>0.95':>8s} {'0.2-0.8':>9s}")
for nm, a, b in FAM:
    f = flip[a:b]
    print(f"  {nm:12s} {b-a:>6d} {f.mean():>10.3f} | {(f<0.05).mean():>6.0%} "
          f"{(f>0.95).mean():>7.0%} {((f>0.2)&(f<0.8)).mean():>8.0%}")
f = flip
print(f"\n  ALL level-1  {len(f):>5d} {f.mean():>10.3f} | {(f<0.05).mean():>6.0%} "
      f"{(f>0.95).mean():>7.0%} {((f>0.2)&(f<0.8)).mean():>8.0%}")

print("\n  VERDICT")
bimodal = (flip < 0.05).mean() + (flip > 0.95).mean()
print(f"    {100*bimodal:.0f}% of dims are decisive (always or never flip); "
      f"{100*((flip>0.2)&(flip<0.8)).mean():.0f}% are middling.")
if bimodal > 0.7:
    print("    -> STRUCTURED. Dims already partition into color-owned and color-neutral,")
    print("       so binding them is cheap and a stable planning frame already exists.")
else:
    print("    -> NOT cleanly structured. A turn smears across dims rather than swapping")
    print("       a designated half, so there is no existing partition to enforce.")
print(f"    color-NEUTRAL dims usable as a stable planning frame today: "
      f"{int((flip<0.05).sum())} of {len(flip)}")

# Is the flip a fixed PERMUTATION? If dim i reliably becomes dim j, it can be undone by
# relabelling axes -- no retraining and no loss term.
A_, B_ = (zp > 0).float().cpu().numpy(), (zn > 0).float().cpu().numpy()
act = [i for i in range(A_.shape[1]) if 0.05 < A_[:, i].mean() < 0.95]
if len(act) > 4:
    Aa = A_[:, act]; Bb = B_[:, act]
    Az = (Aa - Aa.mean(0)) / (Aa.std(0) + 1e-8); Bz = (Bb - Bb.mean(0)) / (Bb.std(0) + 1e-8)
    C = np.abs(Az.T @ Bz) / len(Aa)
    best = C.max(1); who = C.argmax(1)
    print(f"\n  PERMUTATION TEST on {len(act)} non-degenerate dims:")
    print(f"    best |correlation| to some post-flip dim: median {np.median(best):.3f}")
    print(f"    dims whose partner is near-deterministic (>0.9): {100*(best>0.9).mean():.0f}%")
    print(f"    is the mapping one-to-one? {len(set(who.tolist()))}/{len(act)} distinct targets")
    if (best > 0.9).mean() > 0.7 and len(set(who.tolist())) > 0.7 * len(act):
        print("    -> the turn flip IS close to a fixed permutation: undo it by RELABELLING,")
        print("       no retraining, no loss term.")
    else:
        print("    -> NOT a fixed permutation. Undoing the flip needs the encoder changed,")
        print("       not the axes relabelled.")
