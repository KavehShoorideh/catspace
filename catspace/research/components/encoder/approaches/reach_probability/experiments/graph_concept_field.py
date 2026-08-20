#!/usr/bin/env python
"""graph_concept_field.py -- the graph encoder UNDER the concepts; concepts predict the move
and their own future. (Kaveh 2026-08-19)

His spec, verbatim: "use the graph neural network as an underlying mechanism to encode the
board, then run some sort of transformer layer on it, and predict a latent state made up of
concepts where some concepts are moving fast, some slow, with a loss term that makes the
concepts code different things. And from the concepts, predict a future set of concepts and a
head that predicts the stockfish move -- that way we get some sibling resolution."

MAPPING, component by component:
    GNN + transformer      the gmp encoder: L blocks of (per-relation message passing over
                           att/src/dst/mut/VAC/DSC -> full MultiheadAttention -> FFN). The
                           deletion relations are new -- v1 carried only the added edges, and
                           "some edges are deleted" is half the move's derivative.
    latent concept state   VQ on top of the graph: 64x256 square codes from square-nodes,
                           16x32 global heads from the pooled board. kmeans_init +
                           threshold_ema_dead_code, the collapse fix, non-negotiable.
    slow vs fast           first 8 global heads SLOW via VAMP-2 at lag 8 (vamp2_score in
                           losses.py, tested: constant features score 0 -- collapse cannot
                           fake slowness, unlike the refuted MSE persistence that nulled cef5).
                           Last 8 heads fast, unconstrained in time.
    code different things  VAMP's whitening decorrelates the slow block structurally;
                           vicreg_covariance (tested; drove the measured 77%% head redundancy
                           to zero in cef4+) covers the fast block.
    future concepts        dynamics head: (zq_t, played move) -> the child position's own codes
                           (targets detached; child encoded from the slim child block, never
                           through any transformer trunk).
    stockfish move         PL top-1 over move logits = MLP(concepts-as-context (+) move-node).
                           The concat is deliberate: a concepts-only head cannot see moves, a
                           move-node-only head bypasses concepts; jointly the gradient forces
                           the CONCEPTS to carry choice-relevant context -- his "that way we
                           get some sibling resolution".

Also kept: the factorized resolution head (the committor every downstream consumer reads) and
a small current-material head (representation sanity; w kept low after the cef2 lesson).

    .venv/bin/python -m ...graph_concept_field --graphs data/derived/move_graphs_v2_*.npz
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from catspace.research.tools.training_infra.losses import (
    effective_number_weights, factorized_resolution_ce, vamp2_score, vicreg_covariance)
from catspace.research.components.encoder.approaches.reach_probability.experiments \
    .concept_capture_field import factor_vocab
from catspace.research.components.encoder.approaches.reach_probability.experiments \
    .graph_move_policy import MV_CAP, N_NODES, Relation

RELS = ("att", "src", "dst", "mut", "vac", "dsc")
N_COUNT, N_CLASS = 9, 12


class GraphConceptField(nn.Module):
    def __init__(self, d=128, layers=3, attn_heads=4, g_heads=16, g_codes=32, d_code=32,
                 sq_codes=256, d_fine=32, slow_heads=8, hidden=512, n_res=414, n_eg=397,
                 cls_reach=None, cls_eg=None, cls_out=None):
        super().__init__()
        from vector_quantize_pytorch import VectorQuantize
        self.d, self.g_heads, self.d_code = d, g_heads, d_code
        self.sq_codes, self.d_fine, self.slow_heads = sq_codes, d_fine, slow_heads
        self.sq_in = nn.Linear(13 + 7 + 6 + 2, d)
        self.mv_in = nn.Linear(6 + 4 + 5 + 4, d)
        self.rel = nn.ModuleList([nn.ModuleDict({r: Relation(d) for r in RELS})
                                  for _ in range(layers)])
        self.ln_l = nn.ModuleList([nn.LayerNorm(d) for _ in range(layers)])
        self.attn = nn.ModuleList([nn.MultiheadAttention(d, attn_heads, batch_first=True)
                                   for _ in range(layers)])
        self.ln_g = nn.ModuleList([nn.LayerNorm(d) for _ in range(layers)])
        self.ffn = nn.ModuleList([nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(),
                                                nn.Linear(2 * d, d)) for _ in range(layers)])
        self.ln_f = nn.ModuleList([nn.LayerNorm(d) for _ in range(layers)])

        vq_kw = dict(decay=0.9, commitment_weight=0.25, kmeans_init=True,
                     threshold_ema_dead_code=2)
        self.g_proj = nn.Linear(d, g_heads * d_code)
        self.vq = nn.ModuleList([VectorQuantize(dim=d_code, codebook_size=g_codes, **vq_kw)
                                 for _ in range(g_heads)])
        self.sq_proj = nn.Linear(d, d_fine)
        self.sq_vq = VectorQuantize(dim=d_fine, codebook_size=sq_codes, **vq_kw)
        d_state = g_heads * d_code + 64 * d_fine
        self.d_state = d_state

        self.move_head = nn.Sequential(nn.Linear(d_state + d, hidden), nn.GELU(),
                                       nn.Linear(hidden, 1))
        self.mv_emb = nn.Sequential(nn.Linear(6 + 4 + 5 + 4, 64), nn.GELU())
        self.dyn = nn.Sequential(nn.Linear(d_state + 64, hidden), nn.GELU(),
                                 nn.Linear(hidden, g_heads * g_codes + 64 * sq_codes))
        self.now_head = nn.Sequential(nn.Linear(d_state, hidden), nn.GELU(),
                                      nn.Linear(hidden, N_CLASS * N_COUNT))
        self.res_body = nn.Sequential(nn.Linear(d_state, hidden), nn.GELU())
        self.res_reach = nn.Linear(hidden, 3)
        self.res_eg = nn.Linear(hidden, n_eg)
        self.res_out = nn.Linear(hidden, 3)
        for nm, v in (("cls_reach", cls_reach), ("cls_eg", cls_eg), ("cls_out", cls_out)):
            self.register_buffer(nm, torch.as_tensor(
                v if v is not None else np.zeros(n_res, np.int64)).long())
        # slow-block projection for VAMP-2: 32-d so a few-hundred-pair batch supports the
        # covariances (rank-deficiency is the caller's bug, per the loss docstring)
        self.vamp_proj = nn.Linear(slow_heads * d_code, 32, bias=False)

    def backbone(self, sqf, mvf, edges, mv_mask):
        B = sqf.shape[0]
        h = torch.cat([self.sq_in(sqf), self.mv_in(mvf)], 1)
        pad = torch.cat([torch.zeros(B, 64, dtype=torch.bool, device=sqf.device), ~mv_mask], 1)
        for i in range(len(self.rel)):
            hf = h.reshape(B * N_NODES, self.d)
            agg = 0.0
            for r in RELS:
                sd = edges.get(r)
                if sd is not None and len(sd[0]):
                    agg = agg + self.rel[i][r](hf, sd[0], sd[1], B * N_NODES)
            h = self.ln_l[i](h + (agg.reshape(B, N_NODES, self.d) if torch.is_tensor(agg) else 0))
            a, _ = self.attn[i](h, h, h, key_padding_mask=pad, need_weights=False)
            h = self.ln_g[i](h + a)
            h = self.ln_f[i](h + self.ffn[i](h))
        return h                                              # (B, 112, d)

    def encode(self, sqf, mvf, edges, mv_mask):
        """-> zq (B,d_state) in [0,1], g_ids (B,H), sq_ids (B,64), vq_loss, h_mv, pre_g"""
        h = self.backbone(sqf, mvf, edges, mv_mask)
        h_sq, h_mv = h[:, :64], h[:, 64:]
        pre_g = self.g_proj(h_sq.mean(1)).view(len(h), self.g_heads, self.d_code)
        qs, ids, vl = [], [], 0.0
        for i in range(self.g_heads):
            q, idx, l = self.vq[i](pre_g[:, i])
            qs.append(q); ids.append(idx); vl = vl + l
        q_sq, sq_ids, vl2 = self.sq_vq(self.sq_proj(h_sq))
        zq = torch.cat(qs + [q_sq.reshape(len(h), -1)], -1)
        lo = zq.min(-1, keepdim=True).values; hi = zq.max(-1, keepdim=True).values
        zq = (zq - lo) / (hi - lo).clamp(min=1e-5)
        return zq, torch.stack(ids, 1), sq_ids, vl + vl2, h_mv, pre_g

    def move_logits(self, zq, h_mv, mv_mask):
        x = torch.cat([zq[:, None, :].expand(-1, h_mv.shape[1], -1), h_mv], -1)
        return self.move_head(x).squeeze(-1).masked_fill(~mv_mask, float("-inf"))

    def predict_resolution(self, zq):
        b = self.res_body(zq)
        return self.res_reach(b), self.res_eg(b), self.res_out(b)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--graphs", required=True)
    ap.add_argument("--games", type=int, default=20000)
    ap.add_argument("--n-piecedown", type=int, default=45906)
    ap.add_argument("--max-plies", type=int, default=400)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--vamp-batch", type=int, default=128)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--slow-heads", type=int, default=8)
    ap.add_argument("--w-move", type=float, default=2.0)
    ap.add_argument("--w-dyn", type=float, default=1.0)
    ap.add_argument("--w-res", type=float, default=1.0)
    ap.add_argument("--w-now", type=float, default=0.25)
    ap.add_argument("--w-slow", type=float, default=1.0, help="on -vamp2/32, so O(1)")
    ap.add_argument("--w-dec", type=float, default=0.5)
    ap.add_argument("--cb-beta", type=float, default=0.999)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--ckpt-every", type=int, default=1000)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", default="", help="checkpoint to continue from (Adam resets)")
    ap.add_argument("--out", default="artifacts/experiments/gcf1")
    args = ap.parse_args()
    torch.set_num_threads(args.threads); torch.manual_seed(args.seed)
    dev = args.device

    from catspace.research.components.encoder.approaches.reach_probability.experiments \
        .graph_move_policy import featurize
    from catspace.research.components.encoder.approaches.reach_probability.src import (
        trajectories as T)
    from catspace.io import paths
    from catspace.research.tools.training_infra.train.scaffold import (
        TrainConfig, standard_train)

    t0 = time.time()
    z = np.load(args.graphs)
    D = {k: torch.from_numpy(np.ascontiguousarray(z[k])) for k in z.files}
    N = len(D["p_row"])
    tr = T.build(n_human=0, n_sf=args.games, seed=args.seed, cache=True,
                 max_plies=args.max_plies, n_piecedown=args.n_piecedown)
    n = len(tr.tok)
    lz = np.load(paths.derived(f"tb_entry_5-4-3men_{n}.npz"), allow_pickle=True)
    vocab = [str(x) for x in lz["vocab"]]
    res_cls = lz["tb_class"].astype(np.int64)
    cr, ce, co, egs = factor_vocab(vocab)
    gor = tr.game_of_row()
    rs = np.random.default_rng(args.seed); perm = rs.permutation(len(tr.start))
    nt = max(1, int(0.1 * len(tr.start)))
    test_g = np.zeros(len(tr.start), bool); test_g[perm[:nt]] = True
    is_te = test_g[gor[D["p_row"].numpy()]]
    tr_i = np.flatnonzero(~is_te); te_i = np.flatnonzero(is_te)
    v8 = D["valid8"].numpy().astype(bool)
    tr8 = np.flatnonzero(~is_te & v8)
    RES = torch.from_numpy(res_cls)
    r_reach = np.where(res_cls >= 0, cr[res_cls.clip(0)], -1).astype(np.int64)
    r_eg = np.where(res_cls >= 0, ce[res_cls.clip(0)], -1).astype(np.int64)
    r_out = np.where(res_cls >= 0, co[res_cls.clip(0)], -1).astype(np.int64)
    p_np = D["p_row"].numpy()
    cnt = np.bincount(r_eg[p_np[tr_i]][r_eg[p_np[tr_i]] >= 0], minlength=len(egs))
    eg_w = effective_number_weights(torch.from_numpy(cnt), beta=args.cb_beta).to(dev)
    print(f"[gcf] {N:,} graphs | train {len(tr_i):,} / held-out {len(te_i):,} | "
          f"lag-8 pairs {len(tr8):,} [{time.time()-t0:.0f}s]", flush=True)

    net = GraphConceptField(d=args.d_model, layers=args.layers, slow_heads=args.slow_heads,
                            n_res=len(vocab), n_eg=len(egs),
                            cls_reach=cr, cls_eg=ce, cls_out=co).to(dev)
    print(f"[gcf] {sum(p.numel() for p in net.parameters()):,} params | "
          f"{args.slow_heads} slow + {16-args.slow_heads} fast global heads x 32 | 64x256 sq",
          flush=True)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    rng = np.random.default_rng(args.seed)
    probe = te_i[rng.choice(len(te_i), min(2500, len(te_i)), replace=False)]

    def parent_batch(idx):
        b = {k: D[k][idx].to(dev) for k in ("sq", "gl", "mv", "mut", "att", "n_mv", "n_att")}
        sqf, mvf, edges, mask = featurize(b, dev)
        # deletion relations, same flattening as featurize's mut
        off = (torch.arange(len(idx), device=dev) * N_NODES)[:, None]
        jj = torch.arange(MV_CAP, device=dev)[None, :]
        for name, arr in (("vac", D["vac"][idx].to(dev)), ("dsc", D["dsc"][idx].to(dev))):
            a = arr.long()
            val = (a != 255) & mask[:, :, None]
            s = ((jj + 64 + off)[:, :, None].expand_as(a))[val]
            t = (a + off[:, :, None])[val]
            edges[name] = (s, t)
        return sqf, mvf, edges, mask

    def slim_batch(idx, pfx):
        """child / t+8 square-only graphs: move block empty."""
        B = len(idx)
        b = {"sq": D[f"{pfx}_sq"][idx].to(dev), "gl": D[f"{pfx}_gl"][idx].to(dev),
             "mv": torch.zeros(B, MV_CAP, 10, dtype=torch.uint8, device=dev),
             "mut": torch.full((B, MV_CAP, 24), 255, dtype=torch.uint8, device=dev),
             "att": D[f"{pfx}_att"][idx].to(dev),
             "n_mv": torch.zeros(B, dtype=torch.uint8, device=dev),
             "n_att": D[f"{pfx}_n_att"][idx].to(dev)}
        return featurize(b, dev)

    def step_fn(model, step):
        model.train()
        idx = torch.from_numpy(tr_i[rng.integers(0, len(tr_i), args.batch)])
        sqf, mvf, edges, mask = parent_batch(idx)
        zq, g_ids, sq_ids, vl, h_mv, pre_g = model.encode(sqf, mvf, edges, mask)

        played = D["played"][idx].long().to(dev)
        lg = model.move_logits(zq, h_mv, mask)
        l_move = F.cross_entropy(lg, played)

        with torch.no_grad():
            csqf, cmvf, cedges, cmask = slim_batch(idx, "c")
            _, cg_ids, csq_ids, _, _, _ = model.encode(csqf, cmvf, cedges, cmask)
        mv_played = mvf[torch.arange(len(idx), device=dev), played]
        dy = model.dyn(torch.cat([zq, model.mv_emb(mv_played)], -1))
        H, C, SC = model.g_heads, 32, model.sq_codes
        dg = dy[:, :H * C].reshape(-1, C)
        ds = dy[:, H * C:].reshape(-1, SC)
        l_dyn = F.cross_entropy(dg, cg_ids.reshape(-1)) + F.cross_entropy(ds, csq_ids.reshape(-1))

        rt = torch.from_numpy(p_np[idx.numpy()])
        r_l, e_l, o_l = model.predict_resolution(zq)
        l_res, _ = factorized_resolution_ce(
            r_l, e_l, o_l, torch.from_numpy(r_reach[rt]).to(dev),
            torch.from_numpy(r_eg[rt]).to(dev), torch.from_numpy(r_out[rt]).to(dev),
            eg_weight=eg_w)

        counts = torch.stack([(D["sq"][idx, :, 0] == t).sum(1) for t in range(1, 13)], 1)
        l_now = F.cross_entropy(model.now_head(zq).view(-1, N_COUNT),
                                counts.clamp(max=8).reshape(-1).long().to(dev))

        i8 = torch.from_numpy(tr8[rng.integers(0, len(tr8), args.vamp_batch)])
        s0f, m0f, e0, k0 = parent_batch(i8)
        _, _, _, _, _, pg0 = model.encode(s0f, m0f, e0, k0)
        s8f, m8f, e8, k8 = slim_batch(i8, "c8")
        _, _, _, _, _, pg8 = model.encode(s8f, m8f, e8, k8)
        sl = args.slow_heads * model.d_code
        v0 = model.vamp_proj(pg0.reshape(len(i8), -1)[:, :sl])
        v8_ = model.vamp_proj(pg8.reshape(len(i8), -1)[:, :sl])
        vamp = vamp2_score(v0, v8_)
        l_slow = -vamp / 32.0
        l_dec = vicreg_covariance(pg0.reshape(len(i8), -1)[:, sl:])

        loss = (args.w_move * l_move + args.w_dyn * l_dyn + args.w_res * l_res
                + args.w_now * l_now + args.w_slow * l_slow + args.w_dec * l_dec + vl)
        opt.zero_grad(); loss.backward(); opt.step()

        m = {}
        if step % args.eval_every == 0 or step == 1:
            model.eval()
            h1 = h3 = tot = 0
            with torch.no_grad():
                for a in range(0, len(probe), 256):
                    ii = torch.from_numpy(probe[a:a + 256])
                    sqf2, mvf2, ed2, mk2 = parent_batch(ii)
                    zq2, _, _, _, hm2, _ = model.encode(sqf2, mvf2, ed2, mk2)
                    lg2 = model.move_logits(zq2, hm2, mk2)
                    pl2 = D["played"][ii].long().to(dev)
                    rk = (lg2 > lg2.gather(1, pl2[:, None])).sum(1)
                    h1 += int((rk == 0).sum()); h3 += int((rk < 3).sum()); tot += len(ii)
                dyn_g = float((dg.argmax(-1) == cg_ids.reshape(-1)).float().mean())
                dyn_s = float((ds.argmax(-1) == csq_ids.reshape(-1)).float().mean())
            el = time.time() - t0
            print(f"[gcf] step {step:>6}/{args.steps} {el/60:>5.1f}m "
                  f"({el/max(1,step):.3f}s/step) | move {float(l_move):.3f} "
                  f"dyn {float(l_dyn):.3f} res {float(l_res):.3f} | "
                  f"HELD-OUT top1 {h1/tot:.3f} top3 {h3/tot:.3f} | dyn_g {dyn_g:.2f} "
                  f"dyn_s {dyn_s:.2f} | vamp {float(vamp):.1f}/32 dec {float(l_dec):.3f}",
                  flush=True)
            m = {"top1": h1 / tot, "top3": h3 / tot, "vamp": float(vamp),
                 "dyn_g": dyn_g, "dyn_s": dyn_s}
            model.train()
        return {"loss": float(loss.detach()), "move": float(l_move.detach()),
                "dyn": float(l_dyn.detach()), "res": float(l_res.detach()),
                "slow": float(l_slow.detach()), **m}

    start = 0
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        net.load_state_dict(ck["state_dict"]); start = int(ck.get("step", 0))
        print(f"[gcf] resumed from {args.resume} at step {start}", flush=True)
    cfg = TrainConfig(out=args.out, steps=args.steps, ckpt_every=args.ckpt_every,
                      start_step=start, eval_every=args.eval_every, experiment="catspace",
                      run_name=args.out.split("/")[-1], device=dev)
    standard_train(step_fn, net, cfg, args)


if __name__ == "__main__":
    main()
