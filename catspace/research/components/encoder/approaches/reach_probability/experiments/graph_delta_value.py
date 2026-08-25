#!/usr/bin/env python
"""graph_delta_value.py -- dV/dmove grounded by counterfactual interventions (gdv).

The sibling problem, stated one more time: played-move imitation carries ~5 bits/position, so no
amount of it can rank unplayed siblings ("to get ranking you need extra info from sf beyond what
it played. you can't have that"). The sanctioned source of the missing bits is GENERATION: the
counterfactual branches actually PLAYED the 2nd/3rd-best (and some random) moves and let SF
finish the game. Each branch gives two rollout-grounded labels at the SAME parent --

    played move -> ctrl   (the original game's result, mover POV)
    alt move    -> treat  (the branch's result, mover POV)

so the paired difference  dV(alt) - dV(played) ~ treat - ctrl  cancels the position value and the
shared game-prefix noise, and supervises exactly the within-parent contrast imitation cannot.
delta units are GAME POINTS (win=1): a calibrated derivative of value with respect to the move.

TWO HEADS on one graph backbone (all six relations -- att/src/dst/mut/vac/dsc -- the complete
one-ply derivative, additions and deletions):

    prior  PL top-1 logits, trained on the 300k-parent graph file (gmp1's objective, so its
           0.288/0.532 held-out is the direct baseline: value semantics must not cost imitation)
    dV     per-move-node scalar, trained on the CF pairs with a Huber pair loss plus a gauge
           term pinning dV(played) ~ 0 (SF plays it, so its advantage over "the position's
           value" is zero in expectation)

The arena policy this feeds is  prior + lambda * dV  -- plausibility from imitation, direction
from grounded derivatives. No GPS/bias/max flags here: those are gcf2's experiment; this run
stays architecture-comparable to gmp1 so the ONLY new ingredient is the pair supervision.

    .venv/bin/python -u -m ...graph_delta_value \
        --graphs data/derived/move_graphs_v2_300000_6545146.npz \
        --cf-graphs data/derived/move_graphs_v2_cf_24696_6545146.npz \
        --cf data/derived/counterfactual_45142_6545146.npz --device cpu
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from catspace.research.components.encoder.approaches.reach_probability.experiments \
    .graph_move_policy import MV_CAP, N_NODES, Relation, featurize

RELS = ("att", "src", "dst", "mut", "vac", "dsc")


class GraphDeltaValue(nn.Module):
    def __init__(self, d=128, layers=3, heads=4):
        super().__init__()
        self.d = d
        self.sq_in = nn.Linear(13 + 7 + 6 + 2, d)
        self.mv_in = nn.Linear(6 + 4 + 5 + 4, d)
        self.rel = nn.ModuleList([nn.ModuleDict({r: Relation(d) for r in RELS})
                                  for _ in range(layers)])
        self.ln_l = nn.ModuleList([nn.LayerNorm(d) for _ in range(layers)])
        self.attn = nn.ModuleList([nn.MultiheadAttention(d, heads, batch_first=True)
                                   for _ in range(layers)])
        self.ln_g = nn.ModuleList([nn.LayerNorm(d) for _ in range(layers)])
        self.ffn = nn.ModuleList([nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(),
                                                nn.Linear(2 * d, d)) for _ in range(layers)])
        self.ln_f = nn.ModuleList([nn.LayerNorm(d) for _ in range(layers)])
        self.prior_head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        self.dv_head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        # interval head (Kaveh 2026-08-20 pruning idea): per-move [dV_lo, dV_hi] as
        # OFFSETS below/above the point estimate (softplus keeps lo <= dV <= hi by
        # construction), trained by pinball loss on the CF pair deltas
        self.iv_head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 2))

    def forward(self, sqf, mvf, edges, mv_mask):
        """-> prior logits (B,MV) masked, dV (B,MV) zeroed on invalid slots."""
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
            h = self.ln_l[i](h + agg.reshape(B, N_NODES, self.d))
            a, _ = self.attn[i](h, h, h, key_padding_mask=pad, need_weights=False)
            h = self.ln_g[i](h + a)
            h = self.ln_f[i](h + self.ffn[i](h))
        h_mv = h[:, 64:]
        prior = self.prior_head(h_mv).squeeze(-1).masked_fill(~mv_mask, float("-inf"))
        dv = self.dv_head(h_mv).squeeze(-1) * mv_mask.float()
        off = F.softplus(self.iv_head(h_mv)) * mv_mask.float()[:, :, None]
        self.last_lo = dv - off[:, :, 0]
        self.last_hi = dv + off[:, :, 1]
        return prior, dv


def load_graphs(path):
    z = np.load(path)
    keys = ("sq", "gl", "mv", "mut", "vac", "dsc", "att", "n_mv", "played", "n_att", "p_row")
    return {k: torch.from_numpy(np.ascontiguousarray(z[k])) for k in keys}


def batch_of(D, idx, dev):
    """featurize + the two deletion relations (same flattening as featurize's mut)."""
    b = {k: D[k][idx].to(dev) for k in ("sq", "gl", "mv", "mut", "att", "n_mv", "n_att")}
    sqf, mvf, edges, mask = featurize(b, dev)
    off = (torch.arange(len(idx), device=dev) * N_NODES)[:, None]
    jj = torch.arange(MV_CAP, device=dev)[None, :]
    for name in ("vac", "dsc"):
        a = D[name][idx].to(dev).long()
        val = (a != 255) & mask[:, :, None]
        s = ((jj + 64 + off)[:, :, None].expand_as(a))[val]
        t = (a + off[:, :, None])[val]
        edges[name] = (s, t)
    return sqf, mvf, edges, mask


def match_cf(cf, G):
    """map CF records -> (graph idx, alt move-node, played move-node, delta); drop + count the
    unmatchable (alt move truncated out of the 48-slot cap, or alt == played)."""
    rowmap = {int(r): i for i, r in enumerate(G["p_row"].numpy())}
    mv = G["mv"].numpy(); n_mv = G["n_mv"].numpy(); played = G["played"].numpy()
    gi_l, ja_l, jp_l, dl_l = [], [], [], []
    n_trunc = n_same = n_promo_multi = 0
    for row, rf, rt, tr_, ct in zip(cf["row"], cf["rnd_from"], cf["rnd_to"],
                                    cf["treat"], cf["ctrl"]):
        gi = rowmap.get(int(row))
        if gi is None:
            n_trunc += 1
            continue
        k = n_mv[gi]
        hit = np.flatnonzero((mv[gi, :k, 1] == rf) & (mv[gi, :k, 2] == rt))
        if len(hit) == 0:
            n_trunc += 1                      # alt beyond the MV_CAP truncation
            continue
        if len(hit) > 1:                      # promotion fan: prefer the queen
            n_promo_multi += 1
            q = hit[mv[gi, hit, 5] == 4]
            hit = q if len(q) else hit[:1]
        ja = int(hit[0])
        if ja == int(played[gi]):
            n_same += 1                       # SF's live move ranked 2nd in reanalysis
            continue
        gi_l.append(gi); ja_l.append(ja); jp_l.append(int(played[gi]))
        dl_l.append(float(int(tr_) - int(ct)))
    print(f"[gdv] CF matched {len(gi_l):,}/{len(cf['row']):,} | dropped: {n_trunc} truncated, "
          f"{n_same} alt==played | promo-fan resolved {n_promo_multi}", flush=True)
    return (np.array(gi_l), np.array(ja_l), np.array(jp_l), np.array(dl_l, np.float32))


def build_step(args, dev):
    from catspace.research.components.encoder.approaches.reach_probability.src import (
        trajectories as T)

    t0 = time.time()
    D = load_graphs(args.graphs)          # PL pool
    C = load_graphs(args.cf_graphs)       # CF parents
    cf = np.load(args.cf)
    tr = T.build(n_human=0, n_sf=args.games, seed=args.seed, cache=True,
                 max_plies=args.max_plies, n_piecedown=args.n_piecedown)
    gor = tr.game_of_row()
    rs = np.random.default_rng(args.seed); perm = rs.permutation(len(tr.start))
    nt = max(1, int(0.1 * len(tr.start)))
    test_g = np.zeros(len(tr.start), bool); test_g[perm[:nt]] = True

    is_te = test_g[gor[D["p_row"].numpy()]]
    tr_i = np.flatnonzero(~is_te); te_i = np.flatnonzero(is_te)

    cgi, cja, cjp, cdl = match_cf(cf, C)
    cf_te = test_g[gor[C["p_row"].numpy()[cgi]]]
    ctr = np.flatnonzero(~cf_te); cte = np.flatnonzero(cf_te)
    nz = cdl[cte] != 0
    maj = max((cdl[cte][nz] < 0).mean(), (cdl[cte][nz] > 0).mean()) if nz.any() else 0.0
    print(f"[gdv] PL {len(tr_i):,}/{len(te_i):,} train/held-out | pairs {len(ctr):,}/{len(cte):,}"
          f" | held-out nonzero-delta {int(nz.sum()):,} (sign majority {maj:.3f}) "
          f"[{time.time()-t0:.0f}s]", flush=True)

    net = GraphDeltaValue(d=args.d_model, layers=args.layers, heads=args.attn_heads).to(dev)
    print(f"[gdv] {sum(p.numel() for p in net.parameters()):,} params | 6 relations | "
          f"prior + dV heads", flush=True)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    rng = np.random.default_rng(args.seed)
    probe = te_i[rng.choice(len(te_i), min(3000, len(te_i)), replace=False)]

    def pinball(err, tau):
        return torch.maximum(tau * err, (tau - 1) * err).mean()

    def pair_terms(idx_records):
        gi = torch.from_numpy(cgi[idx_records])
        sqf, mvf, edges, mask = batch_of(C, gi, dev)
        _, dv = net(sqf, mvf, edges, mask)
        ja = torch.from_numpy(cja[idx_records]).long().to(dev)
        jp = torch.from_numpy(cjp[idx_records]).long().to(dev)
        dl = torch.from_numpy(cdl[idx_records]).to(dev)
        dva = dv.gather(1, ja[:, None]).squeeze(1)
        dvp = dv.gather(1, jp[:, None]).squeeze(1)
        # interval pinball on the SAME pair contrast: predicted [lo,hi] of (alt - played)
        lo_a = net.last_lo.gather(1, ja[:, None]).squeeze(1)
        hi_a = net.last_hi.gather(1, ja[:, None]).squeeze(1)
        lo_p = net.last_lo.gather(1, jp[:, None]).squeeze(1)
        hi_p = net.last_hi.gather(1, jp[:, None]).squeeze(1)
        # pair delta bounds: lo(alt)-hi(played) .. hi(alt)-lo(played)
        d_lo, d_hi = lo_a - hi_p, hi_a - lo_p
        l_iv = pinball(dl - d_lo, args.tau_lo) + pinball(dl - d_hi, args.tau_hi)
        return (F.huber_loss(dva - dvp, dl), (dvp ** 2).mean(), dva - dvp, dl, l_iv)

    def step_fn(model, step):
        model.train()
        idx = torch.from_numpy(tr_i[rng.integers(0, len(tr_i), args.batch)])
        sqf, mvf, edges, mask = batch_of(D, idx, dev)
        prior, _ = model(sqf, mvf, edges, mask)
        played = D["played"][idx].long().to(dev)
        l_pl = F.cross_entropy(prior, played)
        l_pair, l_gauge, _, _, l_iv = pair_terms(rng.integers(0, len(ctr), args.pair_batch))
        loss = l_pl + args.w_pair * l_pair + args.w_gauge * l_gauge + args.w_iv * l_iv
        opt.zero_grad(); loss.backward(); opt.step()

        m = {}
        if step % args.eval_every == 0 or step == 1:
            model.eval()
            h1 = h3 = tot = 0
            se = 0.0; sn = 0; sgn_ok = 0; sgn_n = 0; gau = 0.0
            with torch.no_grad():
                for a in range(0, len(probe), 256):
                    ii = torch.from_numpy(probe[a:a + 256])
                    sqf, mvf, edges, mask = batch_of(D, ii, dev)
                    prior, _ = model(sqf, mvf, edges, mask)
                    played = D["played"][ii].long().to(dev)
                    rk = (prior > prior.gather(1, played[:, None])).sum(1)
                    h1 += int((rk == 0).sum()); h3 += int((rk < 3).sum()); tot += len(ii)
                cov = 0; covn = 0
                for a in range(0, len(cte), 256):
                    _, g2, dhat, dl, _liv = pair_terms(cte[a:a + 256])
                    se += float((dhat - dl).abs().sum()); sn += len(dl)
                    nz = dl != 0
                    sgn_ok += int(((dhat[nz] > 0) == (dl[nz] > 0)).sum())
                    sgn_n += int(nz.sum()); gau += float(g2) * len(dl)
                    ja2 = torch.from_numpy(cja[cte[a:a + 256]]).long().to(dev)
                    jp2 = torch.from_numpy(cjp[cte[a:a + 256]]).long().to(dev)
                    dlo = (net.last_lo.gather(1, ja2[:, None]).squeeze(1)
                           - net.last_hi.gather(1, jp2[:, None]).squeeze(1))
                    dhi = (net.last_hi.gather(1, ja2[:, None]).squeeze(1)
                           - net.last_lo.gather(1, jp2[:, None]).squeeze(1))
                    dl2 = torch.from_numpy(cdl[cte[a:a + 256]]).to(dev)
                    cov += int(((dl2 >= dlo) & (dl2 <= dhi)).sum()); covn += len(dl2)
            el = time.time() - t0
            print(f"[gdv] step {step:>6}/{args.steps} {el/60:>5.1f}m "
                  f"({el/max(1,step):.3f}s/step) | pl {float(l_pl.detach()):.3f} pair "
                  f"{float(l_pair.detach()):.3f} | HELD-OUT top1 {h1/tot:.3f} top3 {h3/tot:.3f} | "
                  f"CF sign {sgn_ok/max(1,sgn_n):.3f} mae {se/max(1,sn):.3f} "
                  f"gauge {gau/max(1,sn):.4f} | IV coverage {cov/max(1,covn):.3f} "
                  f"(target {args.tau_hi - args.tau_lo:.1f})", flush=True)
            m = {"top1": h1 / tot, "top3": h3 / tot, "cf_sign": sgn_ok / max(1, sgn_n),
                 "cf_mae": se / max(1, sn)}
            model.train()
        return {"loss": float(loss.detach()), **m}

    step_fn.opt = opt   # standard_train checkpoints optimizer state from here (resumability)
    return net, step_fn


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--graphs", required=True)
    ap.add_argument("--cf-graphs", required=True)
    ap.add_argument("--cf", required=True)
    ap.add_argument("--games", type=int, default=20000)
    ap.add_argument("--n-piecedown", type=int, default=45906)
    ap.add_argument("--max-plies", type=int, default=400)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--pair-batch", type=int, default=64)
    ap.add_argument("--w-pair", type=float, default=1.0)
    ap.add_argument("--w-gauge", type=float, default=0.2)
    ap.add_argument("--w-iv", type=float, default=1.0,
                    help="pinball losses (tau 0.1/0.9) on the interval head")
    ap.add_argument("--tau-lo", type=float, default=0.1)
    ap.add_argument("--tau-hi", type=float, default=0.9)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--attn-heads", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--ckpt-every", type=int, default=1000)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", default="")
    ap.add_argument("--out", default="artifacts/experiments/gdv1")
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
        missing, unexpected = net.load_state_dict(ck["state_dict"], strict=False)
        assert not unexpected, f"unexpected keys: {unexpected[:4]}"
        assert all(k.startswith("iv_head") for k in missing), f"missing non-iv: {missing[:4]}"
        start = int(ck.get("step", 0))
        print(f"[gdv] resumed from {args.resume} at step {start}", flush=True)
    cfg = TrainConfig(out=args.out, steps=args.steps, ckpt_every=args.ckpt_every,
                      start_step=start, eval_every=args.eval_every, experiment="catspace",
                      run_name=args.out.split("/")[-1], device=args.device)
    standard_train(step_fn, net, cfg, args)


if __name__ == "__main__":
    main()
