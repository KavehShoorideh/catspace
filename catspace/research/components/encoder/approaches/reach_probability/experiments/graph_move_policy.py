#!/usr/bin/env python
"""graph_move_policy.py -- local-then-global GNN over square+move nodes; one pass scores all moves.

The GNN alternative from the sister design chat (Kaveh 2026-08-19), reviewed and adapted:

    KEPT     square-nodes + MOVE-NODES (readout scales with legal moves, and one forward pass
             replaces the ~34 per-child trunk encodes that make cef6 cost 0.66 s/step);
             hypothetical MUTATION edges (move -> squares it would newly attack), which build in
             the measured one-ply-consequence mechanism -- a move-node hears "your destination is
             attacked-by-2 defended-by-1" and can represent 'hangs the piece' natively;
             interleaved local/global per layer, L=3.
    CHANGED  PyTorch Geometric dropped -- no py3.14 wheels, and at <=112 nodes per position
             sparse machinery is pointless. Local step = per-relation dense masked aggregation
             (index_add over flattened batch graphs), global step = standard MultiheadAttention.
             Plain torch throughout.
    RESCOPED target = Plackett-Luce top-1 on the played move over ALL legal siblings -- the
             repo-legal objective, so this is directly comparable to cpl1 (11.4%% held-out,
             frozen-vocabulary ceiling) and cef6 (the 3.7 h end-to-end run). Reachability
             targets come later if the architecture earns them.

RELATIONS, each with its own message weights:
    att      square -> square it attacks              (the real, static structure)
    src_of   source square -> its move-node
    dst_of   destination square -> its move-node
    mut      move-node -> squares it would NEWLY attack (the hypothetical edge, reversed:
             the move broadcasts its consequence onto the board)

    .venv/bin/python -m ...graph_move_policy --graphs data/derived/move_graphs_*.npz
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MV_CAP, MUT_CAP, ATT_CAP = 48, 24, 224
N_NODES = 64 + MV_CAP


class Relation(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.w = nn.Linear(d, d, bias=False)

    def forward(self, h_flat, src, dst, n_total):
        """mean-aggregate messages src->dst over the flattened batch graph."""
        m = self.w(h_flat[src])
        agg = torch.zeros(n_total, m.shape[-1], device=m.device, dtype=m.dtype)
        agg.index_add_(0, dst, m)
        cnt = torch.zeros(n_total, device=m.device, dtype=m.dtype)
        cnt.index_add_(0, dst, torch.ones_like(dst, dtype=m.dtype))
        return agg / cnt.clamp(min=1).unsqueeze(-1)


class GraphMovePolicy(nn.Module):
    def __init__(self, d=128, layers=3, heads=4):
        super().__init__()
        self.d = d
        self.sq_in = nn.Linear(13 + 7 + 6 + 2, d)      # piece one-hot, counts/flags, glob, file/rank
        self.mv_in = nn.Linear(6 + 4 + 5 + 4, d)       # piece, from/to coords, promo, flags
        self.rel = nn.ModuleList([nn.ModuleDict({r: Relation(d) for r in
                                                 ("att", "src", "dst", "mut")})
                                  for _ in range(layers)])
        self.ln_l = nn.ModuleList([nn.LayerNorm(d) for _ in range(layers)])
        self.attn = nn.ModuleList([nn.MultiheadAttention(d, heads, batch_first=True)
                                   for _ in range(layers)])
        self.ln_g = nn.ModuleList([nn.LayerNorm(d) for _ in range(layers)])
        self.ffn = nn.ModuleList([nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(),
                                                nn.Linear(2 * d, d)) for _ in range(layers)])
        self.ln_f = nn.ModuleList([nn.LayerNorm(d) for _ in range(layers)])
        self.out = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))

    def forward(self, sqf, mvf, edges, mv_mask):
        """sqf (B,64,Fs), mvf (B,MV,Fm), edges dict of flattened (src,dst) longs, mv_mask (B,MV).
        -> logits (B, MV) with -inf on invalid move slots."""
        B = sqf.shape[0]
        h = torch.cat([self.sq_in(sqf), self.mv_in(mvf)], 1)          # (B, 112, d)
        pad = torch.cat([torch.zeros(B, 64, dtype=torch.bool, device=sqf.device),
                         ~mv_mask], 1)
        for i in range(len(self.rel)):
            hf = h.reshape(B * N_NODES, self.d)
            agg = 0.0
            for r in ("att", "src", "dst", "mut"):
                s, t = edges[r]
                if len(s):
                    agg = agg + self.rel[i][r](hf, s, t, B * N_NODES)
            h = self.ln_l[i](h + agg.reshape(B, N_NODES, self.d))
            a, _ = self.attn[i](h, h, h, key_padding_mask=pad, need_weights=False)
            h = self.ln_g[i](h + a)
            h = self.ln_f[i](h + self.ffn[i](h))
        lg = self.out(h[:, 64:]).squeeze(-1)                          # (B, MV)
        return lg.masked_fill(~mv_mask, float("-inf"))


def featurize(batch, dev, n_nodes=N_NODES):
    """uint8 arrays -> dense node features + flattened edge lists. `n_nodes` sets the
    flattening stride for models with extra nodes (gcf3 file-nodes: 120)."""
    sq, gl, mv, mut, att, n_mv, n_att = (batch[k] for k in
                                         ("sq", "gl", "mv", "mut", "att", "n_mv", "n_att"))
    n_mv = n_mv.long(); n_att = n_att.long()          # uint8/uint16 do not promote with Long
    B = sq.shape[0]
    piece = F.one_hot(sq[:, :, 0].long(), 13).float()
    counts = torch.stack([sq[:, :, 1].float() / 7, sq[:, :, 2].float() / 7,
                          sq[:, :, 3].float(), sq[:, :, 4].float(),
                          sq[:, :, 5].float(), sq[:, :, 6].float(),
                          sq[:, :, 7].float() / 7], -1)
    glob = gl.float()[:, None, :].expand(B, 64, 6)
    fr = torch.arange(64, device=dev)
    coords = torch.stack([(fr % 8).float() / 7, (fr // 8).float() / 7], -1)[None].expand(B, 64, 2)
    sqf = torch.cat([piece, counts, glob / torch.tensor([1, 1, 1, 1, 1, 8.], device=dev),
                     coords], -1)

    pt = F.one_hot((mv[:, :, 0].long() - 1).clamp(min=0), 6).float()
    mcoord = torch.stack([(mv[:, :, 1].float() % 8) / 7, (mv[:, :, 1].float() // 8) / 7,
                          (mv[:, :, 2].float() % 8) / 7, (mv[:, :, 2].float() // 8) / 7], -1)
    promo = F.one_hot(mv[:, :, 5].long(), 5).float()
    flags = torch.stack([mv[:, :, 3].float(), mv[:, :, 4].float() / 9,
                         mv[:, :, 6].float(), mv[:, :, 7].float() / MUT_CAP], -1)
    mvf = torch.cat([pt, mcoord, promo, flags], -1)

    mv_mask = (torch.arange(MV_CAP, device=dev)[None, :] < n_mv[:, None])
    off = (torch.arange(B, device=dev) * n_nodes)[:, None]

    ea = att.long()
    val_a = (ea[:, :, 0] != 255) & (torch.arange(ATT_CAP, device=dev)[None, :] < n_att[:, None])
    att_s = (ea[:, :, 0] + off)[val_a]
    att_d = (ea[:, :, 1] + off)[val_a]

    jj = torch.arange(MV_CAP, device=dev)[None, :]
    src_s = (mv[:, :, 1].long() + off)[mv_mask]
    src_d = (jj + 64 + off)[mv_mask]
    dst_s = (mv[:, :, 2].long() + off)[mv_mask]
    dst_d = src_d

    mu = mut.long()
    val_m = (mu != 255) & mv_mask[:, :, None]
    mut_s = ((jj + 64 + off)[:, :, None].expand_as(mu))[val_m]
    mut_d = (mu + off[:, :, None])[val_m]

    edges = {"att": (att_s, att_d), "src": (src_s, src_d),
             "dst": (dst_s, dst_d), "mut": (mut_s, mut_d)}
    return sqf, mvf, edges, mv_mask


def build_step(args, dev):
    from catspace.research.components.encoder.approaches.reach_probability.src import (
        trajectories as T)

    t0 = time.time()
    z = np.load(args.graphs)
    keys = ("sq", "gl", "mv", "mut", "att", "n_mv", "played", "n_att", "p_row")
    D = {k: torch.from_numpy(np.ascontiguousarray(z[k])) for k in keys}
    N = len(D["p_row"])
    tr = T.build(n_human=0, n_sf=args.games, seed=args.seed, cache=True,
                 max_plies=args.max_plies, n_piecedown=args.n_piecedown)
    gor = tr.game_of_row()
    rs = np.random.default_rng(args.seed); perm = rs.permutation(len(tr.start))
    nt = max(1, int(0.1 * len(tr.start)))
    test_g = np.zeros(len(tr.start), bool); test_g[perm[:nt]] = True
    is_te = test_g[gor[D["p_row"].numpy()]]
    tr_i = np.flatnonzero(~is_te); te_i = np.flatnonzero(is_te)
    print(f"[gmp] {N:,} graphs | train {len(tr_i):,} / held-out {len(te_i):,} | "
          f"mean legal {float(D['n_mv'].float().mean()):.1f} [{time.time()-t0:.0f}s]", flush=True)

    net = GraphMovePolicy(d=args.d_model, layers=args.layers, heads=args.attn_heads).to(dev)
    print(f"[gmp] {sum(p.numel() for p in net.parameters()):,} params | plain-torch hetero "
          f"GNN, {args.layers} local+global layers", flush=True)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    rng = np.random.default_rng(args.seed)
    probe = te_i[rng.choice(len(te_i), min(3000, len(te_i)), replace=False)]

    def get(idx):
        b = {k: D[k][idx].to(dev) for k in ("sq", "gl", "mv", "mut", "att", "n_mv", "n_att")}
        return featurize(b, dev), D["played"][idx].long().to(dev)

    def step_fn(model, step):
        model.train()
        idx = torch.from_numpy(tr_i[rng.integers(0, len(tr_i), args.batch)])
        (sqf, mvf, edges, mask), played = get(idx)
        lg = model(sqf, mvf, edges, mask)
        loss = F.cross_entropy(lg, played)              # pure PL top-1
        opt.zero_grad(); loss.backward(); opt.step()

        m = {}
        if step % args.eval_every == 0 or step == 1:
            model.eval()
            h1 = h3 = tot = 0
            with torch.no_grad():
                for a in range(0, len(probe), 256):
                    ii = torch.from_numpy(probe[a:a + 256])
                    (sqf, mvf, edges, mask), played = get(ii)
                    lg = model(sqf, mvf, edges, mask)
                    rk = (lg > lg.gather(1, played[:, None])).sum(1)
                    h1 += int((rk == 0).sum()); h3 += int((rk < 3).sum()); tot += len(ii)
            el = time.time() - t0
            print(f"[gmp] step {step:>6}/{args.steps} {el/60:>5.1f}m "
                  f"({el/max(1,step):.3f}s/step) | loss {float(loss):.4f} | "
                  f"HELD-OUT top1 {h1/tot:.3f} top3 {h3/tot:.3f} "
                  f"(chance {1/float(D['n_mv'].float().mean()):.3f})", flush=True)
            m = {"top1": h1 / tot, "top3": h3 / tot}
            model.train()
        return {"loss": float(loss.detach()), **m}

    step_fn.opt = opt   # standard_train checkpoints optimizer state from here (resumability)
    return net, step_fn


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--graphs", required=True)
    ap.add_argument("--games", type=int, default=20000)
    ap.add_argument("--n-piecedown", type=int, default=45906)
    ap.add_argument("--max-plies", type=int, default=400)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--attn-heads", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--ckpt-every", type=int, default=1000)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", default="", help="checkpoint to continue from (Adam state "
                    "resets -- the scaffold saves models only)")
    ap.add_argument("--out", default="artifacts/experiments/gmp1")
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    from catspace.research.tools.training_infra.train.scaffold import (
        TrainConfig, standard_train)
    net, step_fn = build_step(args, args.device)
    start = 0
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        net.load_state_dict(ck["state_dict"]); start = int(ck.get("step", 0))
        print(f"[gmp] resumed from {args.resume} at step {start}", flush=True)
    cfg = TrainConfig(out=args.out, steps=args.steps, ckpt_every=args.ckpt_every,
                      start_step=start, eval_every=args.eval_every, experiment="catspace",
                      run_name=args.out.split("/")[-1], device=args.device)
    standard_train(step_fn, net, cfg, args)


if __name__ == "__main__":
    main()
