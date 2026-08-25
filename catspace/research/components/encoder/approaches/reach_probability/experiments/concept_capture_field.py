#!/usr/bin/env python
"""concept_capture_field.py -- concepts trained to say WHERE THE ACTION IS. (Kaveh 2026-08-17)

"have a set of concepts that allow us to predict the next five captures... I don't wanna capture
the entire board, but I wanna capture where the action is, and the action is the capture."
Plus "both piece types, e.g. knight takes pawn on e5", and "also the outcome or tablebase class".

THE MEASURED PROBLEM THIS EXISTS TO FIX. The previous vocabulary (cdf5) encodes VALUE but not
material composition or LOCATION:

    name the 5-man endgame class @20 plies   0.138   vs material lookup 0.213  <- worse than
                                                                                  counting pieces
    predict the next capture's SQUARE        0.044   vs baseline        0.044  <- no signal at all
    predict its attacker                     0.217   vs baseline        0.131

Nothing in that loss stack ever asked the concepts to localise anything. Captures are where
material changes, they are dense (18.6% of plies), and the 5th sits a median 19 plies ahead --
the planning range. So they become the target rather than a probe.

WHAT WAS DELETED FROM cdf5, AND WHY IT IS FASTER. The dynamics predicted the EMA encoder's own
codes, which cost an EMA forward over K*B rows -- 36.2% of every step by the profiler. Capture
labels come from disk instead, so the EMA branch and the K=5 unroll both go. Rows encoded per
step fall ~1,536 -> ~340.

CAPACITY, rebalanced on evidence rather than taste: square codes alone reach 0.697 outcome
accuracy, global heads 0.661, and global adds +0.002 on top of square while dying within a ply
(survival half-life 1 ply vs 4). So 16x16 global -> 8x16, and 64x32 square -> 64x64.

    .venv/bin/python -m ...concept_capture_field --steps 200 --out artifacts/experiments/ccf_smoke
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from catspace.research.tools.training_infra.losses import (
    capture_ce, compose_resolution_logits, effective_number_weights,
    factorized_resolution_ce)

N_CAP = 5          # captures predicted ahead; matches gen_capture_labels default
N_PIECE = 13       # jepa token vocab: 0 empty + 6 white + 6 black
OUTCOMES = ["W", "D", "L"]


def factor_vocab(vocab):
    """Split the flat resolution vocabulary into COARSE (does this game simplify?) and FINE
    (which ending, exactly?).

        'WDL:W'     -> reach 0 (ends first),  no ending, outcome W
        'KRP|KR=W'  -> reach 1 (reaches),     ending #k, outcome carried INSIDE the ending
        'OTHER'     -> reach 2,               no ending, no outcome

    The ending index is joint over signature AND outcome: `KRP|KR=D` and `KRP|KR=W` are the
    Philidor and the Lucena, the same material with opposite results, so a product of marginals
    cannot express them (see `factorized_resolution_ce`). cls_out is therefore -1 on endings --
    their outcome must not be counted twice.

    -> (cls_reach (C,), cls_eg (C,), cls_out (C,), egs list[str]), int64 with -1 for
    "this factor does not apply", which the losses skip via ignore_index.
    """
    egs = [v for v in vocab if not v.startswith("WDL:") and "=" in v]
    ei = {s: i for i, s in enumerate(egs)}
    cr = np.zeros(len(vocab), np.int64)
    ce = np.full(len(vocab), -1, np.int64)
    co = np.full(len(vocab), -1, np.int64)
    for i, v in enumerate(vocab):
        if v.startswith("WDL:"):
            cr[i], co[i] = 0, OUTCOMES.index(v.split(":")[1])
        elif "=" in v:
            cr[i], ce[i] = 1, ei[v]              # outcome lives inside the ending index
        else:                                    # 'OTHER' -- no ending, no outcome
            cr[i] = 2
    return cr, ce, co, egs


def signature_of(egs):
    """-> (sig_of_ending (E,) int64, sigs list[str]). Marginalising the joint endgame head over
    these groups recovers "which material signature", and over outcome recovers W/D/L, so the
    interpretable readouts cost no extra parameters."""
    sigs = sorted({v.split("=")[0] for v in egs})
    si = {s: i for i, s in enumerate(sigs)}
    return np.array([si[v.split("=")[0]] for v in egs], np.int64), sigs


class ConceptCapture(nn.Module):
    def __init__(self, d_model=192, layers=4, tr_heads=8, heads=8, codes=16, d_code=32,
                 square_codes=64, d_fine=32, hidden=512, n_res=401,
                 kmeans_init=True, dead_code=2, factored=True, n_eg=397,
                 cls_reach=None, cls_eg=None, cls_out=None):
        super().__init__()
        from vector_quantize_pytorch import VectorQuantize
        from catspace.research.components.encoder.approaches.jepa_tokenizer.src.jepa import (
            JepaEncoder)
        self.heads, self.codes, self.d_code = heads, codes, d_code
        self.square_codes, self.d_fine = square_codes, d_fine

        self.trunk = JepaEncoder(d_model, layers, tr_heads)
        self.proj = nn.Linear(d_model, heads * d_code)
        # kmeans_init + dead-code replacement are NOT optional: they are what took codebook
        # usage from 4.7 to 16 of 16 alive after a session of ablations found nothing else.
        vq_kw = dict(decay=0.9, commitment_weight=0.25, kmeans_init=bool(kmeans_init),
                     threshold_ema_dead_code=int(dead_code))
        self.vq = nn.ModuleList([VectorQuantize(dim=d_code, codebook_size=codes, **vq_kw)
                                 for _ in range(heads)])
        self.sq_proj = nn.Linear(d_model, d_fine)
        self.sq_vq = VectorQuantize(dim=d_fine, codebook_size=square_codes, **vq_kw)

        d_state = heads * d_code + 64 * d_fine
        self.d_state = d_state
        # CAPTURE HEADS: one bank per upcoming capture, three fields each. Kept as separate
        # linear heads on a shared body so each field can be scored against its own measured
        # baseline (square 5.1%, victim 29.1%, attacker 13.8%) instead of hiding inside a joint
        # 4,608-way class where a strong victim prior would flatter everything.
        self.cap_body = nn.Sequential(nn.Linear(d_state, hidden), nn.GELU())
        self.cap_sq = nn.Linear(hidden, N_CAP * 64)
        self.cap_vic = nn.Linear(hidden, N_CAP * N_PIECE)
        self.cap_att = nn.Linear(hidden, N_CAP * N_PIECE)
        # RESOLUTION: 5-piece class WITH W/D/L, or plain W/D/L when the game ends first.
        # FACTORED, because the flat 401-way version collapsed: on ccf1 at step 5000 it put
        # 95.5% of its mass on the three WDL classes (true rate 77.9%) and named only 4.2% of
        # real endgames -- see `factorized_resolution_ce`. The factors are a property of the
        # label, not a trick: `KRP|KR=W` IS a signature and an outcome.
        self.factored = bool(factored)
        if self.factored:
            self.res_body = nn.Sequential(nn.Linear(d_state, hidden), nn.GELU())
            self.res_reach = nn.Linear(hidden, 3)        # ends-first / reaches / OTHER
            self.res_eg = nn.Linear(hidden, n_eg)        # WHICH ending (signature+outcome)
            self.res_out = nn.Linear(hidden, 3)          # W / D / L, for games ending first
            # lookup tables to recompose the flat classes; buffers so they ride the checkpoint
            # and land on the right device without a call-site remembering to move them.
            for nm, v in (("cls_reach", cls_reach), ("cls_eg", cls_eg), ("cls_out", cls_out)):
                self.register_buffer(nm, torch.as_tensor(
                    v if v is not None else np.zeros(n_res, np.int64)).long(), persistent=True)
        else:
            # kept under its original name so the flat-head ccf1 checkpoints still load and
            # can be scored by the same instrument
            self.resolution = nn.Sequential(nn.Linear(d_state, hidden), nn.GELU(),
                                            nn.Linear(hidden, n_res))

    # ---- state ----------------------------------------------------------------------
    @staticmethod
    def norm_state(x, eps=1e-5):
        """MuZero App. G range-normalisation, applied AT THE SOURCE. Putting it at the call
        sites once left one head training on two different input distributions."""
        lo = x.min(dim=-1, keepdim=True).values
        hi = x.max(dim=-1, keepdim=True).values
        return (x - lo) / (hi - lo).clamp(min=eps)

    def encode(self, tok, glob):
        """-> (state (B,d_state) in [0,1], global ids (B,H), square ids (B,64), vq loss)"""
        phi, sq_tok = self.trunk.forward_tokens(tok, glob)
        h = self.proj(phi).view(len(phi), self.heads, self.d_code)
        qs, ids, vl = [], [], 0.0
        for i in range(self.heads):
            q, idx, l = self.vq[i](h[:, i])
            qs.append(q); ids.append(idx); vl = vl + l
        q_sq, ids_sq, vl_sq = self.sq_vq(self.sq_proj(sq_tok))
        vl = vl + vl_sq
        zq = torch.cat(qs + [q_sq.reshape(len(phi), -1)], -1)
        return self.norm_state(zq), torch.stack(ids, 1), ids_sq, vl

    def predict_resolution(self, zq):
        """-> (reach (B,3), ending (B,n_eg), outcome (B,3)) -- the training-time view.
        Raises on a flat model, which has no factors to return."""
        if not self.factored:
            raise RuntimeError("flat resolution head has no factors; use resolution_logits")
        b = self.res_body(zq)
        return self.res_reach(b), self.res_eg(b), self.res_out(b)

    def resolution_logits(self, zq):
        """-> (B, n_res) over the ORIGINAL flat classes, whichever head this model has, so the
        acceptance instrument scores factored and flat models identically."""
        if not self.factored:
            return self.resolution(zq)
        r, e, o = self.predict_resolution(zq)
        return compose_resolution_logits(r, e, o, self.cls_reach, self.cls_eg, self.cls_out)

    def predict_captures(self, zq):
        """-> (square (B,N,64), victim (B,N,13), attacker (B,N,13))"""
        b = self.cap_body(zq); n = len(zq)
        return (self.cap_sq(b).view(n, N_CAP, 64),
                self.cap_vic(b).view(n, N_CAP, N_PIECE),
                self.cap_att(b).view(n, N_CAP, N_PIECE))


def build_step(args, dev):
    """Returns (net, step_fn, meta). step_fn(model, step) -> metrics dict, for standard_train."""
    from catspace.research.components.encoder.approaches.reach_probability.src import (
        trajectories as T)
    from catspace.io import paths

    t0 = time.time()
    tr = T.build(n_human=0, n_sf=args.games, seed=args.seed, cache=True,
                 max_plies=args.max_plies, n_piecedown=args.n_piecedown)
    n = len(tr.tok)
    cl = np.load(paths.derived(f"capture_labels_{N_CAP}_{n}.npz"))
    lz = np.load(paths.derived(f"tb_entry_5-4-3men_{n}.npz"), allow_pickle=True)
    vocab = [str(x) for x in lz["vocab"]]
    res_cls = lz["tb_class"].astype(np.int64)
    gor = tr.game_of_row()

    rs = np.random.default_rng(args.seed); perm = rs.permutation(len(tr.start))
    nv = max(1, int(args.val_frac * len(tr.start))); nt = max(1, int(args.test_frac * len(tr.start)))
    sog = np.zeros(len(tr.start), np.int8); sog[perm[:nv]] = 1; sog[perm[nv:nv + nt]] = 2
    spl = sog[gor]

    # quiet mask for persistence: no capture on this ply (exact -- a capture removes exactly
    # one man, promotion removes none; verified across all 4.6M transitions)
    men = (tr.tok != 0).sum(1)
    gz = np.load(paths.derived(f"game_transitions_FULL_{n}.npz"))
    par = gz["par"]
    quiet = np.zeros(n, bool); quiet[par] = men[par] == men[par + 1]

    # FACTOR the resolution label. Done here rather than in the model so the class-balance
    # weights can be counted on the TRAIN split only -- weights fitted on val/test would leak.
    cr_np, ce_np, co_np, egs = factor_vocab(vocab)
    sig_of_eg, sigs = signature_of(egs)
    r_reach = np.where(res_cls >= 0, cr_np[res_cls.clip(0)], -1).astype(np.int64)
    r_eg = np.where(res_cls >= 0, ce_np[res_cls.clip(0)], -1).astype(np.int64)
    r_out = np.where(res_cls >= 0, co_np[res_cls.clip(0)], -1).astype(np.int64)

    TOK = torch.from_numpy(tr.tok); GLB = torch.from_numpy(tr.glob)
    CSQ = torch.from_numpy(cl["cap_sq"].astype(np.int64))
    CVI = torch.from_numpy(cl["cap_vic"].astype(np.int64))
    CAT = torch.from_numpy(cl["cap_att"].astype(np.int64))
    RES = torch.from_numpy(res_cls)
    RCH = torch.from_numpy(r_reach); REG = torch.from_numpy(r_eg); ROT = torch.from_numpy(r_out)
    SIG_OF_EG = torch.from_numpy(sig_of_eg)

    train_rows = np.flatnonzero((spl == 0) & (res_cls >= 0))
    # persistence needs t and t+2 in the same game
    pers_rows = train_rows[(train_rows + 2 < n)]
    pers_rows = pers_rows[gor[pers_rows + 2] == gor[pers_rows]]
    print(f"[ccf] {n:,} positions | train {len(train_rows):,} | persistence-eligible "
          f"{len(pers_rows):,} | {len(vocab)} resolution classes [{time.time()-t0:.0f}s]",
          flush=True)

    # class-balance weights for the signature head, counted on TRAIN rows only
    eg_w = None
    if args.factored and args.cb_beta > 0:
        cnt = np.bincount(r_eg[train_rows][r_eg[train_rows] >= 0], minlength=len(egs))
        eg_w = effective_number_weights(torch.from_numpy(cnt), beta=args.cb_beta).to(dev)
        nz = cnt > 0
        print(f"[ccf] {len(egs)} endings ({len(sigs)} signatures) | counts {cnt[nz].max():,}..{cnt[nz].min():,} "
              f"({cnt[nz].max()/max(1,cnt[nz].min()):,.0f}x) -> class-balanced weights "
              f"{float(eg_w[nz].max()):.2f}..{float(eg_w[nz].min()):.2f} "
              f"(beta={args.cb_beta})", flush=True)
        rf = r_reach[train_rows]
        print(f"[ccf] reach split: ends-first {100*(rf==0).mean():.1f}%  "
              f"reaches-tablebase {100*(rf==1).mean():.1f}%  OTHER {100*(rf==2).mean():.1f}%",
              flush=True)

    net = ConceptCapture(d_model=args.d_model, heads=args.heads, codes=args.codes,
                         square_codes=args.square_codes, hidden=args.hidden,
                         n_res=len(vocab), factored=args.factored, n_eg=len(egs),
                         cls_reach=cr_np, cls_eg=ce_np, cls_out=co_np).to(dev)
    print(f"[ccf] {sum(p.numel() for p in net.parameters()):,} params | "
          f"{args.heads}x{args.codes} global + 64x{args.square_codes} square", flush=True)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    rng = np.random.default_rng(args.seed)

    def batch(rows):
        r = torch.as_tensor(rows)
        return TOK[r].long().to(dev), GLB[r].float().to(dev)

    def step_fn(model, step):
        model.train()
        rows = train_rows[rng.integers(0, len(train_rows), args.batch)]
        zq, ids, ids_sq, vl = model.encode(*batch(rows))
        rt = torch.as_tensor(rows)
        sq_l, vi_l, at_l = model.predict_captures(zq)
        l_cap = 0.0
        for k in range(N_CAP):
            l_cap = l_cap + (capture_ce(sq_l[:, k], CSQ[rt, k].to(dev))
                             + capture_ce(vi_l[:, k], CVI[rt, k].to(dev))
                             + capture_ce(at_l[:, k], CAT[rt, k].to(dev))) / N_CAP
        if model.factored:
            r_l, e_l, o_l = model.predict_resolution(zq)
            l_res, res_parts = factorized_resolution_ce(
                r_l, e_l, o_l, RCH[rt].to(dev), REG[rt].to(dev), ROT[rt].to(dev),
                eg_weight=eg_w)
        else:
            l_res = F.cross_entropy(model.resolution(zq), RES[rt].to(dev))
            res_parts = {}

        l_pers = torch.zeros((), device=dev)
        if args.w_pers > 0:
            pr = pers_rows[rng.integers(0, len(pers_rows), max(8, args.batch // 4))]
            qm = torch.from_numpy((quiet[pr] & quiet[pr + 1]).astype(np.float32)).to(dev)
            if float(qm.sum()) > 0:
                za, _, _, _ = model.encode(*batch(pr))
                zb, _, _, _ = model.encode(*batch(pr + 2))     # SAME side to move
                d = (za - zb).pow(2).mean(-1)
                l_pers = (qm * d).sum() / qm.sum().clamp(min=1)

        loss = (args.w_cap * l_cap + args.w_res * l_res + args.w_pers * l_pers + vl)
        opt.zero_grad(); loss.backward(); opt.step()

        with torch.no_grad():
            sq_acc = float((sq_l[:, 0].argmax(-1) == CSQ[rt, 0].to(dev))[
                CSQ[rt, 0].to(dev) >= 0].float().mean())
            at_acc = float((at_l[:, 0].argmax(-1) == CAT[rt, 0].to(dev))[
                CAT[rt, 0].to(dev) >= 0].float().mean())
            ps = [torch.bincount(ids[:, h], minlength=args.codes).float()
                  for h in range(args.heads)]
            perp = float(np.mean([float(torch.exp(-((p / p.sum()) *
                         ((p / p.sum()) + 1e-9).log()).sum())) for p in ps]))
            sqp = torch.bincount(ids_sq.reshape(-1), minlength=args.square_codes).float()
            sq_perp = float(torch.exp(-((sqp / sqp.sum()) * ((sqp / sqp.sum()) + 1e-9).log()).sum()))
            # WATCH THE HEADLINE LIVE. The flat head's collapse was invisible in `res` (which
            # fell happily while the model stopped naming endgames), so the endgame-naming
            # accuracy itself is logged every step rather than discovered at eval time.
            eg_acc = mat_acc = eg_cond = float("nan")
            if model.factored:
                mask = REG[rt].to(dev) >= 0
                if bool(mask.any()):
                    tgt_eg = REG[rt].to(dev)[mask]
                    pe = e_l.argmax(-1)[mask]
                    # CONDITIONAL: given that this game does reach a tablebase, do we name
                    # the right ending? This is the question actually asked of the concepts.
                    eg_cond = float((pe == tgt_eg).float().mean())
                    sg = SIG_OF_EG.to(dev)
                    mat_acc = float((sg[pe] == sg[tgt_eg]).float().mean())
                    # UNCONDITIONAL: the pre-registered instrument's decision rule
                    flat = model.resolution_logits(zq).argmax(-1)
                    eg_acc = float((flat[mask] == RES[rt].to(dev)[mask]).float().mean())
        # standard_train logs to MLflow but prints nothing, which left ccf1.log blank for
        # 5,000 steps with no way to watch the run. One compact line per eval fixes that.
        if step % args.eval_every == 0 or step == 1:
            el = time.time() - t0
            print(f"[ccf] step {step:>6}/{args.steps} {el/60:>6.1f}m "
                  f"({el/max(1,step):.3f}s/step) | loss {float(loss.detach()):.3f} "
                  f"cap {float(l_cap.detach()):.3f} res {float(l_res.detach()):.3f} | "
                  f"cap_sq {sq_acc:.3f} cap_att {at_acc:.3f} | eg_cond {eg_cond:.3f} "
                  f"eg_acc {eg_acc:.3f} | perp {perp:.1f} sq_perp {sq_perp:.1f}", flush=True)
        return {"loss": float(loss.detach()), "cap": float(l_cap.detach()),
                "res": float(l_res.detach()), "pers": float(l_pers.detach()),
                "vq": float(vl.detach()), "cap_sq_acc": sq_acc, "cap_att_acc": at_acc,
                "perp": perp, "sq_perp": sq_perp, "mat_acc": mat_acc, "eg_acc": eg_acc,
                "eg_cond": eg_cond,
                **res_parts}

    step_fn.opt = opt   # standard_train checkpoints optimizer state from here (resumability)
    return net, step_fn


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--games", type=int, default=4000)
    ap.add_argument("--n-piecedown", type=int, default=45906)
    ap.add_argument("--max-plies", type=int, default=400)
    ap.add_argument("--batch", type=int, default=192)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--codes", type=int, default=16)
    ap.add_argument("--square-codes", type=int, default=64)
    ap.add_argument("--d-model", type=int, default=192)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--w-cap", type=float, default=2.0, help="the next-5-captures objective")
    ap.add_argument("--w-res", type=float, default=1.0, help="resolution objective")
    ap.add_argument("--factored", type=int, default=1,
                    help="1 = factor the resolution label into reach x signature x outcome "
                         "(the fix); 0 = the flat 401-way head that collapsed onto the three "
                         "WDL classes on ccf1")
    ap.add_argument("--cb-beta", type=float, default=0.999,
                    help="class-balance beta for the signature head (Cui et al. 2019). "
                         "Effective sample size saturates at 1/(1-beta), so 0.999 boosts only "
                         "signatures with under ~1000 rows. 0 disables weighting.")
    ap.add_argument("--w-pers", type=float, default=0.25,
                    help="quiet-ply persistence over t->t+2. 1.0 OVERSHOT last time: concepts "
                         "stabilised but the model's edge over 'nothing changed' fell from "
                         "+0.53 to +0.135, so start low and sweep.")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--test-frac", type=float, default=0.1)
    ap.add_argument("--ckpt-every", type=int, default=1000)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="artifacts/experiments/ccf1")
    args = ap.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    from catspace.research.tools.training_infra.train.scaffold import (
        TrainConfig, standard_train)

    net, step_fn = build_step(args, args.device)
    cfg = TrainConfig(out=args.out, steps=args.steps, ckpt_every=args.ckpt_every,
                      eval_every=args.eval_every, experiment="catspace",
                      run_name=args.out.split("/")[-1], device=args.device)
    standard_train(step_fn, net, cfg, args)


if __name__ == "__main__":
    main()
