#!/usr/bin/env python
"""concept_dynamics_field.py -- concepts as a STATE, dynamics as the model, tablebase as the
anchor. No quasimetric. (Kaveh 2026-08-16)

"detach the quasimetric part from the previous part. So we want an encoder that [emits] a
latent state, and the latent state has this dynamics function that advances to the next latent
state. And the latent state itself is vector quantized, so that we have a series of concepts.
And what we're trying to predict is this next set of concepts with the dynamics function, plus
a win/loss/draw probability of the next point... and predict the win/loss probability of this
step. And eventually we will reach a tablebase five piece ending, and when we do, we will
anchor it to the particular tablebase lookup... so it's gonna be one hundred percent, either
win or lost or drawn."  Plus: "we could actually just go directly from the encoder to the
discrete state, and I agree that we should predict an EMA copy."

    board ──encoder──► VQ ──► c_t          the DISCRETE STATE. No continuous state beside it:
                               │           the concept set IS the state.
                 ┌─────────────┼─────────────┐
          (c_t,a)│             │             │
             dyn ▼             ▼             ▼
             ĉ_{t+1}      value(c_t)    value(ĉ_{t+1})
                │          WDL now       WDL of the state we PREDICT
         CE vs c_{t+1} from an EMA COPY of the encoder

WHY THE EMA TARGET. If the dynamics predicted the LIVE encoder's codes, encoder and dynamics
could collude: any constant code assignment satisfies the objective perfectly and says nothing.
An EMA copy cannot be moved by this step's gradient, so agreement has to be earned.

WHY value(ĉ_{t+1}) AND NOT JUST value(c_t). This is the Value Equivalence Principle, and it is
what makes the dynamics worth having: the predicted next state is graded on whether it carries
the right VALUE, not on reproducing the board. "Reconstruct the board only in so much as it
helps me win."

BASELINE, STATED BEFORE ANY RESULT. At <=5 men a majority-WDL-per-material lookup (269 rows,
fitted on train) scores **92.9%** on test. That is the bar, not the 40.4% majority class. The
model earns its keep only in the 7.1% where material does NOT decide -- KP vs K turning on the
pawn's file and the opposition, KRP vs KR on structure. A model at 90% here has learned LESS
than the lookup table.

THE ANCHOR. At <=5 men Syzygy gives the solved answer -- 287,977 corpus positions, exactly
one-hot, no estimation. Everywhere else the only value signal is the game's final result, which
is ONE noisy playout: measured, it disagrees with the tablebase on 6.2% of anchored positions,
because that is where the played line threw the theoretical value away. So the anchor is
weighted well above the outcome term, and the outcome term exists mainly to keep the 93.8% of
positions that never simplify tethered to something real.

    .venv/bin/python -m ...concept_dynamics_field --steps 300      # smoke
"""
from __future__ import annotations

import argparse
import copy
import json
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConceptDynamics(nn.Module):
    def __init__(self, d_model=192, layers=4, tr_heads=8, heads=16, codes=16, d_code=32,
                 square_codes=32, d_fine=32, hidden=512, ema=0.996,
                 kmeans_init=True, dead_code=2):
        super().__init__()
        from vector_quantize_pytorch import VectorQuantize
        from catspace.research.components.encoder.approaches.jepa_tokenizer.src.jepa import (
            JepaEncoder)
        self.heads, self.codes, self.d_code = heads, codes, d_code
        self.square_codes, self.d_fine, self.ema = square_codes, d_fine, float(ema)

        self.trunk = JepaEncoder(d_model, layers, tr_heads)
        # STRAIGHT to the discrete state: one linear, then quantize. No intermediate continuous
        # "latent state" distinct from the concepts (Kaveh: "go directly from the encoder to
        # the discrete state").
        self.proj = nn.Linear(d_model, heads * d_code)
        vq_kw = dict(decay=0.9, commitment_weight=0.25, kmeans_init=bool(kmeans_init),
                     threshold_ema_dead_code=int(dead_code))
        self.vq = nn.ModuleList([VectorQuantize(dim=d_code, codebook_size=codes, **vq_kw)
                                 for _ in range(heads)])
        self.sq_proj = nn.Linear(d_model, d_fine) if square_codes else None
        self.sq_vq = (VectorQuantize(dim=d_fine, codebook_size=square_codes, **vq_kw)
                      if square_codes else None)

        # EMA TARGET: encoder side only. The codebooks are already EMA-updated internally and
        # are shared, so the target uses the live books with a frozen-ish encoder -- the same
        # arrangement jqt used.
        self.t_trunk = copy.deepcopy(self.trunk)
        self.t_proj = copy.deepcopy(self.proj)
        self.t_sq_proj = copy.deepcopy(self.sq_proj) if square_codes else None
        for m in (self.t_trunk, self.t_proj, self.t_sq_proj):
            if m is not None:
                for p in m.parameters():
                    p.requires_grad_(False)

        d_state = heads * d_code + (64 * d_fine if square_codes else 0)
        self.mv_from = nn.Embedding(64, 32); self.mv_to = nn.Embedding(64, 32)
        self.mv_promo = nn.Embedding(5, 32)
        self.dyn = nn.Sequential(nn.Linear(d_state + 32, hidden), nn.GELU(),
                                 nn.Linear(hidden, heads * codes))
        self.dyn_sq = (nn.Sequential(nn.Linear(d_state + 32, hidden), nn.GELU(),
                                     nn.Linear(hidden, 64 * square_codes))
                       if square_codes else None)
        # VALUE reads the QUANTIZED state, never the continuous features: if value could be
        # read off the pre-quantisation vector the concepts would not be required to carry it.
        self.value = nn.Sequential(nn.Linear(d_state, hidden), nn.GELU(),
                                   nn.Linear(hidden, 3))

    # ---- state ---------------------------------------------------------------------
    def encode(self, tok, glob, target=False):
        """-> (zq (B,d_state) straight-through, global ids (B,H), square ids (B,64), vq loss)"""
        trunk = self.t_trunk if target else self.trunk
        proj = self.t_proj if target else self.proj
        sqp = self.t_sq_proj if target else self.sq_proj
        phi, sq_tok = trunk.forward_tokens(tok, glob)
        h = proj(phi).view(len(phi), self.heads, self.d_code)
        qs, ids, vl = [], [], 0.0
        for i in range(self.heads):
            q, idx, l = self.vq[i](h[:, i])
            qs.append(q); ids.append(idx); vl = vl + l
        zq = torch.cat(qs, -1)
        ids_sq = None
        if self.square_codes:
            q_sq, ids_sq, vl_sq = self.sq_vq(sqp(sq_tok))
            vl = vl + vl_sq
            zq = torch.cat([zq, q_sq.reshape(len(phi), -1)], -1)
        return zq, torch.stack(ids, 1), ids_sq, vl

    @torch.no_grad()
    def update_ema(self):
        for tgt, src in ((self.t_trunk, self.trunk), (self.t_proj, self.proj),
                         (self.t_sq_proj, self.sq_proj)):
            if tgt is None:
                continue
            for pt, ps in zip(tgt.parameters(), src.parameters()):
                pt.mul_(self.ema).add_(ps, alpha=1 - self.ema)

    # ---- dynamics ------------------------------------------------------------------
    def act(self, mid):
        return self.mv_from(mid[:, 0]) + self.mv_to(mid[:, 1]) + self.mv_promo(mid[:, 2])

    def predict_next(self, zq, mid):
        x = torch.cat([zq, self.act(mid)], -1)
        g = self.dyn(x).view(len(zq), self.heads, self.codes)
        s = (self.dyn_sq(x).view(len(zq), 64, self.square_codes)
             if self.square_codes else None)
        return g, s

    def soft_state(self, g_logits, s_logits):
        """the PREDICTED next state as a differentiable vector: expected code under the
        dynamics' own distribution. This is what value(c_hat) is read from, so a dynamics that
        predicts value-irrelevant codes is penalised through the value head."""
        pg = g_logits.softmax(-1)                                  # (B,H,C)
        book = torch.stack([self.vq[i].codebook.squeeze(0) for i in range(self.heads)], 0)
        zg = torch.einsum("bhc,hcd->bhd", pg, book).reshape(len(pg), -1)
        if s_logits is None:
            return zg
        ps = s_logits.softmax(-1)                                  # (B,64,S)
        zs = torch.einsum("bnc,cd->bnd", ps, self.sq_vq.codebook.squeeze(0))
        return torch.cat([zg, zs.reshape(len(pg), -1)], -1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--games", type=int, default=4000)
    ap.add_argument("--n-piecedown", type=int, default=45906)
    ap.add_argument("--max-plies", type=int, default=400)
    ap.add_argument("--batch", type=int, default=192)
    ap.add_argument("--heads", type=int, default=16)
    ap.add_argument("--codes", type=int, default=16)
    ap.add_argument("--square-codes", type=int, default=32)
    ap.add_argument("--d-model", type=int, default=192)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--ema", type=float, default=0.996)
    ap.add_argument("--w-dyn", type=float, default=2.0, help="predict the next concept set")
    ap.add_argument("--w-anchor", type=float, default=5.0,
                    help="Syzygy WDL. The ONLY ground truth we have, so it outweighs the "
                         "outcome term; it disagrees with the final result on 6.2%% of "
                         "anchored positions and is right every time it does.")
    ap.add_argument("--w-outcome", type=float, default=1.0,
                    help="the game's final result: available everywhere, but ONE noisy playout")
    ap.add_argument("--w-vnext", type=float, default=1.0,
                    help="value of the state the DYNAMICS predicts (value equivalence)")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--test-frac", type=float, default=0.1)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--ckpt-every", type=int, default=1000)
    ap.add_argument("--out", default="artifacts/experiments/cdf1")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from catspace.research.components.encoder.approaches.reach_probability.src import (
        trajectories as T)
    from catspace.research.tools.training_infra import gen_game_transitions as GT
    from catspace.io import paths

    torch.manual_seed(args.seed)
    dev, t0 = args.device, time.time()
    tr = T.build(n_human=0, n_sf=args.games, seed=args.seed, cache=True,
                 max_plies=args.max_plies, n_piecedown=args.n_piecedown)
    gor = tr.game_of_row()
    gz = GT.ensure(tr, T.load_sf_games(args.games, args.seed, max_plies=args.max_plies))
    par, mid = gz["par"], gz["mid"]

    # ---- targets: EXACT where the tablebase reaches, the game's result elsewhere ------
    syz = np.load(paths.derived(f"syzygy_wdl_{len(tr.tok)}.npz"))["syz_wdl"].astype(np.int64)
    out_w = tr.outcome_of_row_white()
    res = np.where(out_w == T.WIN, 2, np.where(out_w == T.DRAW, 1, 0)).astype(np.int64)
    print(f"[cdf] {len(tr.tok):,} positions | anchored (Syzygy) {int((syz>=0).sum()):,} "
          f"({100*(syz>=0).mean():.2f}%) | outcome-labelled {len(res):,}", flush=True)

    # ---- split BY GAME ---------------------------------------------------------------
    rs = np.random.default_rng(args.seed)
    perm = rs.permutation(len(tr.start))
    nv = max(1, int(args.val_frac * len(tr.start))); nt = max(1, int(args.test_frac * len(tr.start)))
    sog = np.zeros(len(tr.start), np.int8); sog[perm[:nv]] = 1; sog[perm[nv:nv+nt]] = 2
    spl = sog[gor]
    tr_par = par[sog[gor[par]] == 0]; tr_mid = mid[sog[gor[par]] == 0]
    anc = {k: np.flatnonzero((syz >= 0) & (spl == k)) for k in (0, 1, 2)}
    print(f"[cdf] split by GAME: train {int((spl==0).sum()):,} / val {int((spl==1).sum()):,} / "
          f"test {int((spl==2).sum()):,} positions | anchored train {len(anc[0]):,} "
          f"val {len(anc[1]):,} test {len(anc[2]):,}", flush=True)
    print(f"[cdf] {len(tr_par):,} train transitions [{time.time()-t0:.0f}s]", flush=True)

    net = ConceptDynamics(d_model=args.d_model, heads=args.heads, codes=args.codes,
                          square_codes=args.square_codes, hidden=args.hidden,
                          ema=args.ema).to(dev)
    opt = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=args.lr)
    print(f"[cdf] {sum(p.numel() for p in net.parameters()):,} params", flush=True)

    def batch(rows):
        return (torch.from_numpy(tr.tok[rows].astype(np.int64)).to(dev),
                torch.from_numpy(tr.glob[rows].astype(np.float32)).to(dev))

    rng = np.random.default_rng(args.seed)
    log = open(f"{args.out}_steps.jsonl", "w", buffering=1)
    for step in range(1, args.steps + 1):
        net.train()
        # (1) transitions: dynamics + value-of-next
        ti = rng.integers(0, len(tr_par), args.batch)
        p_rows = tr_par[ti]
        zq, ids, ids_sq, vl = net.encode(*batch(p_rows))
        with torch.no_grad():
            _, t_ids, t_ids_sq, _ = net.encode(*batch(p_rows + 1), target=True)
        mv = torch.from_numpy(tr_mid[ti].astype(np.int64)).to(dev)
        g_lg, s_lg = net.predict_next(zq, mv)
        l_dyn = F.cross_entropy(g_lg.reshape(-1, args.codes), t_ids.reshape(-1))
        if s_lg is not None:
            l_dyn = l_dyn + F.cross_entropy(s_lg.reshape(-1, args.square_codes),
                                            t_ids_sq.reshape(-1))
        # value of the state the dynamics PREDICTS, graded against the child's own target
        y_next = torch.from_numpy(np.where(syz[p_rows + 1] >= 0, syz[p_rows + 1],
                                           res[p_rows + 1])).to(dev)
        l_vnext = F.cross_entropy(net.value(net.soft_state(g_lg, s_lg)), y_next)

        # (2) value now: EXACT anchor, and the noisy outcome everywhere
        a_rows = anc[0][rng.integers(0, len(anc[0]), args.batch // 2)]
        zq_a, _, _, vl_a = net.encode(*batch(a_rows))
        l_anchor = F.cross_entropy(net.value(zq_a),
                                   torch.from_numpy(syz[a_rows]).to(dev))
        l_out = F.cross_entropy(net.value(zq), torch.from_numpy(res[p_rows]).to(dev))

        loss = (args.w_dyn * l_dyn + args.w_vnext * l_vnext
                + args.w_anchor * l_anchor + args.w_outcome * l_out + vl + vl_a)
        opt.zero_grad(); loss.backward(); opt.step()
        net.update_ema()

        if step % 20 == 0 or step == 1:
            with torch.no_grad():
                dyn_acc = float((g_lg.argmax(-1) == t_ids).float().mean())
                anc_acc = float((net.value(zq_a).argmax(-1) ==
                                 torch.from_numpy(syz[a_rows]).to(dev)).float().mean())
                ps = [torch.bincount(ids[:, h], minlength=args.codes).float() for h in
                      range(args.heads)]
                perp = float(np.mean([float(torch.exp(-((p/p.sum()) *
                             ((p/p.sum())+1e-9).log()).sum())) for p in ps]))
            rec = {"s": step, "loss": float(loss.detach()), "dyn": float(l_dyn.detach()),
                   "vnext": float(l_vnext.detach()), "anchor": float(l_anchor.detach()),
                   "out": float(l_out.detach()), "dyn_acc": dyn_acc, "anchor_acc": anc_acc,
                   "perp": perp}
            log.write(json.dumps(rec) + "\n")
            if step % 100 == 0 or step == 1:
                print(f"[cdf] {step:5d} loss {float(loss):7.3f} | ANCHOR acc {anc_acc:.3f} "
                      f"(MATERIAL baseline .929) | next-code {dyn_acc:.3f} | perp {perp:5.2f} | "
                      f"{time.time()-t0:.0f}s", flush=True)

        if step % args.eval_every == 0 or step == args.steps:
            net.eval()
            with torch.no_grad():
                vr = anc[1][rng.integers(0, len(anc[1]), 1024)]
                zq_v, _, _, _ = net.encode(*batch(vr))
                va = float((net.value(zq_v).argmax(-1) ==
                            torch.from_numpy(syz[vr]).to(dev)).float().mean())
            net.train()
            print(f"[cdf] {step:5d} VAL anchor accuracy {va:.3f}  (MATERIAL-lookup baseline 0.929; majority DRAW 0.404)",
                  flush=True)
            log.write(json.dumps({"s": step, "val_anchor": va}) + "\n")
        if args.ckpt_every and step % args.ckpt_every == 0:
            torch.save({"state_dict": net.state_dict(), "step": step, "args": vars(args)},
                       f"{args.out}_step{step}.pt")

    net.eval()
    with torch.no_grad():
        te = anc[2][rng.integers(0, len(anc[2]), 4096)]
        zq_t, _, _, _ = net.encode(*batch(te))
        ta = float((net.value(zq_t).argmax(-1) ==
                    torch.from_numpy(syz[te]).to(dev)).float().mean())
    print(f"\n[cdf] ===== TEST (single touch) =====\n[cdf]   anchor accuracy {ta:.3f}", flush=True)
    print(f"[cdf]   BASELINES: material-signature lookup 0.929 | majority DRAW 0.404", flush=True)
    print(f"[cdf]   The bar is 0.929, NOT 0.404. At <=5 men, material alone decides 92.9% of "
          f"positions;\n[cdf]   the 7.1% where it does not is the only part that requires "
          f"reading the board.", flush=True)
    torch.save({"state_dict": net.state_dict(), "step": args.steps, "args": vars(args)},
               f"{args.out}_latest.pt")
    print(f"[cdf] done -> {args.out}_latest.pt [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
