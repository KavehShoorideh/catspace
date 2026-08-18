#!/usr/bin/env python
"""concept_event_field.py -- concepts that predict EVENTS and the MATERIAL they leave.

"predict the next five interesting events, whether they're captures or promotions, and we predict
the piece count or type of pieces that are available at that stage of the game. So we're
predicting multiple things at the same time." (Kaveh 2026-08-17)

WHAT THIS CHANGES FROM ccf2, AND THE MEASUREMENT BEHIND EACH CHANGE.

ccf2 learned to localise -- next capture square 0.168 against a 0.051 baseline, where the
previous vocabulary had literally no signal -- but named the 5-man ending only at the material
floor (0.210 vs 0.211). The diagnostic (`diag_captures_to_material.py`) found two causes, and
both are addressed here at the objective:

  1 PROMOTIONS WERE UNREPRESENTABLE. Deriving material by subtracting captured pieces scores
    0.0% when a promotion occurs -- a pawn becoming a queen changes the material with nothing
    leaving the board. Promotions are now EVENTS (4.08% of all events; 36,422 pure promotions
    plus 945 capture-promotions that a captures-only stream also mislabels).

  2 MATERIAL WAS A CONJUNCTION, NOT A PREDICTION. Reconstructing the ending by chaining victims
    needs every victim right at once: per-capture victim accuracy runs 0.436 / 0.412 / 0.389 /
    0.348 / 0.352 and the joint lands at 10.4%, which matches the 8.1% of endings that route
    actually recovers. So the material at each event is now its OWN target -- 12 piece-count
    heads per event, predicted directly from the position rather than composed from five fragile
    predictions.

The 5-event horizon reaches a 5-man position on 76.2% of the endgame_20 rows (by event 3: 36.2%,
by event 4: 57.7%), so the ending is inside the window the model is being asked about.

The encoder, both codebooks, the persistence term and the reach x ending x outcome resolution
head are imported unchanged from `concept_capture_field`, so this is a change of OBJECTIVE, not
of architecture, and ccf2 remains the control on the same corpus, split and seed.

    .venv/bin/python -m ...concept_event_field --steps 200 --out artifacts/experiments/cef_smoke
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from catspace.research.tools.training_infra.losses import (
    capture_ce, effective_number_weights, factorized_resolution_ce)
from catspace.research.components.encoder.approaches.reach_probability.experiments \
    .concept_capture_field import ConceptCapture, factor_vocab, signature_of

N_EV = 5           # events predicted ahead
N_PIECE = 13       # jepa token vocab
N_KIND = 4         # 0 unused, 1 capture, 2 promotion, 3 capture+promotion
N_COUNT = 9        # piece counts run 0..8 (measured max)
N_CLASS = 12       # piece classes: white PNBRQK then black pnbrqk


class ConceptEvent(ConceptCapture):
    """ccf2's encoder and resolution head, with the capture heads replaced by event heads and a
    MATERIAL head. Subclassed rather than copied so the encoder cannot silently drift from the
    control run."""

    def __init__(self, *a, hidden=512, **kw):
        super().__init__(*a, hidden=hidden, **kw)
        d = self.d_state
        del self.cap_body, self.cap_sq, self.cap_vic, self.cap_att
        self.ev_body = nn.Sequential(nn.Linear(d, hidden), nn.GELU())
        self.ev_kind = nn.Linear(hidden, N_EV * N_KIND)
        self.ev_sq = nn.Linear(hidden, N_EV * 64)
        self.ev_mover = nn.Linear(hidden, N_EV * N_PIECE)
        self.ev_victim = nn.Linear(hidden, N_EV * N_PIECE)
        # THE MATERIAL HEAD: what is left on the board after each event. Twelve small
        # classifications instead of one hopeless softmax -- the intermediate stages span 70,031
        # distinct signatures, while the counts compose into exactly the same thing and each is
        # individually easy.
        self.mat_body = nn.Sequential(nn.Linear(d, hidden), nn.GELU())
        self.ev_counts = nn.Linear(hidden, N_EV * N_CLASS * N_COUNT)
        # CURRENT material, read back out of the concepts. Measured on cef1: a linear probe
        # recovers all twelve counts of the board ALREADY IN FRONT OF IT only 42.5% of the time
        # (global codes alone 3.8%, square codes alone 35.6%) -- so the quantiser discards piece
        # identity even though 416 bits are available for 17.2 bits of material. Capacity gives
        # the codes room to keep it; this term gives them a reason. Labels are free: they are
        # just the board. It is also a ceiling-raiser, since future material cannot be predicted
        # better than present material is represented.
        self.now_body = nn.Sequential(nn.Linear(d, hidden), nn.GELU())
        self.now_counts = nn.Linear(hidden, N_CLASS * N_COUNT)

    def predict_events(self, zq):
        b = self.ev_body(zq); n = len(zq)
        return (self.ev_kind(b).view(n, N_EV, N_KIND),
                self.ev_sq(b).view(n, N_EV, 64),
                self.ev_mover(b).view(n, N_EV, N_PIECE),
                self.ev_victim(b).view(n, N_EV, N_PIECE))

    def predict_now(self, zq):
        """-> (B, 12, N_COUNT) logits over the count of each piece class ON THE BOARD NOW."""
        return self.now_counts(self.now_body(zq)).view(len(zq), N_CLASS, N_COUNT)

    def predict_material(self, zq):
        """-> (B, N_EV, 12, N_COUNT) logits over the count of each piece class after each event."""
        return self.ev_counts(self.mat_body(zq)).view(len(zq), N_EV, N_CLASS, N_COUNT)


def build_step(args, dev):
    from catspace.research.components.encoder.approaches.reach_probability.src import (
        trajectories as T)
    from catspace.io import paths

    t0 = time.time()
    tr = T.build(n_human=0, n_sf=args.games, seed=args.seed, cache=True,
                 max_plies=args.max_plies, n_piecedown=args.n_piecedown)
    n = len(tr.tok)
    ev = np.load(paths.derived(f"event_labels_{N_EV}_{n}.npz"), allow_pickle=True)
    lz = np.load(paths.derived(f"tb_entry_5-4-3men_{n}.npz"), allow_pickle=True)
    vocab = [str(x) for x in lz["vocab"]]
    res_cls = lz["tb_class"].astype(np.int64)
    gor = tr.game_of_row()

    rs = np.random.default_rng(args.seed); perm = rs.permutation(len(tr.start))
    nv = max(1, int(args.val_frac * len(tr.start))); nt = max(1, int(args.test_frac * len(tr.start)))
    sog = np.zeros(len(tr.start), np.int8); sog[perm[:nv]] = 1; sog[perm[nv:nv + nt]] = 2
    spl = sog[gor]

    men = (tr.tok != 0).sum(1)
    gz = np.load(paths.derived(f"game_transitions_FULL_{n}.npz"))
    par = gz["par"]
    quiet = np.zeros(n, bool); quiet[par] = men[par] == men[par + 1]

    cr_np, ce_np, co_np, egs = factor_vocab(vocab)
    sig_of_eg, sigs = signature_of(egs)
    r_reach = np.where(res_cls >= 0, cr_np[res_cls.clip(0)], -1).astype(np.int64)
    r_eg = np.where(res_cls >= 0, ce_np[res_cls.clip(0)], -1).astype(np.int64)
    r_out = np.where(res_cls >= 0, co_np[res_cls.clip(0)], -1).astype(np.int64)

    now_cnt = np.stack([(tr.tok == t).sum(1) for t in range(1, 13)], 1).astype(np.int64)
    NOW = torch.from_numpy(now_cnt)
    TOK = torch.from_numpy(tr.tok); GLB = torch.from_numpy(tr.glob)
    EKI = torch.from_numpy(ev["ev_type"].astype(np.int64))
    ESQ = torch.from_numpy(ev["ev_sq"].astype(np.int64))
    EMV = torch.from_numpy(ev["ev_mover"].astype(np.int64))
    EVI = torch.from_numpy(ev["ev_victim"].astype(np.int64))
    ECN = torch.from_numpy(ev["ev_counts"].astype(np.int64))       # (n, N_EV, 12), -1 absent
    EMN = torch.from_numpy(ev["ev_men"].astype(np.int64))
    RES = torch.from_numpy(res_cls)
    RCH = torch.from_numpy(r_reach); REG = torch.from_numpy(r_eg); ROT = torch.from_numpy(r_out)

    train_rows = np.flatnonzero((spl == 0) & (res_cls >= 0))
    pers_rows = train_rows[(train_rows + 2 < n)]
    pers_rows = pers_rows[gor[pers_rows + 2] == gor[pers_rows]]
    print(f"[cef] {n:,} positions | train {len(train_rows):,} | persistence-eligible "
          f"{len(pers_rows):,} | {len(vocab)} resolution classes [{time.time()-t0:.0f}s]",
          flush=True)
    k = ev["ev_type"][train_rows]
    print(f"[cef] events in train: capture {100*(k==1).mean():.1f}%  promotion "
          f"{100*(k==2).mean():.2f}%  capture+promotion {100*(k==3).mean():.2f}%  "
          f"absent {100*(k<0).mean():.1f}%", flush=True)

    eg_w = None
    if args.cb_beta > 0:
        cnt = np.bincount(r_eg[train_rows][r_eg[train_rows] >= 0], minlength=len(egs))
        eg_w = effective_number_weights(torch.from_numpy(cnt), beta=args.cb_beta).to(dev)

    net = ConceptEvent(d_model=args.d_model, heads=args.heads, codes=args.codes,
                       square_codes=args.square_codes, hidden=args.hidden,
                       n_res=len(vocab), factored=True, n_eg=len(egs),
                       cls_reach=cr_np, cls_eg=ce_np, cls_out=co_np).to(dev)
    print(f"[cef] {sum(p.numel() for p in net.parameters()):,} params | "
          f"{args.heads}x{args.codes} global + 64x{args.square_codes} square", flush=True)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    rng = np.random.default_rng(args.seed)

    def batch(rows):
        r = torch.as_tensor(rows)
        return TOK[r].long().to(dev), GLB[r].float().to(dev)

    def step_fn(model, step):
        model.train()
        rows = train_rows[rng.integers(0, len(train_rows), args.batch)]
        rt = torch.as_tensor(rows)
        zq, ids, ids_sq, vl = model.encode(*batch(rows))

        ki, sq, mv, vi = model.predict_events(zq)
        l_ev = 0.0
        for j in range(N_EV):
            l_ev = l_ev + (capture_ce(ki[:, j], EKI[rt, j].to(dev))
                           + capture_ce(sq[:, j], ESQ[rt, j].to(dev))
                           + capture_ce(mv[:, j], EMV[rt, j].to(dev))
                           + capture_ce(vi[:, j], EVI[rt, j].to(dev))) / N_EV

        # MATERIAL: 12 counts per event, each masked independently. This is the term the whole
        # change exists for -- it asks the concepts what will be LEFT, instead of hoping it falls
        # out of a chain of victim predictions.
        mc = model.predict_material(zq)
        tgt_c = ECN[rt].to(dev)                                  # (B, N_EV, 12)
        l_mat = capture_ce(mc.reshape(-1, N_COUNT), tgt_c.reshape(-1))

        l_now = capture_ce(model.predict_now(zq).reshape(-1, N_COUNT), NOW[rt].to(dev).reshape(-1))

        r_l, e_l, o_l = model.predict_resolution(zq)
        l_res, res_parts = factorized_resolution_ce(
            r_l, e_l, o_l, RCH[rt].to(dev), REG[rt].to(dev), ROT[rt].to(dev), eg_weight=eg_w)

        l_pers = torch.zeros((), device=dev)
        if args.w_pers > 0:
            pr = pers_rows[rng.integers(0, len(pers_rows), max(8, args.batch // 4))]
            qm = torch.from_numpy((quiet[pr] & quiet[pr + 1]).astype(np.float32)).to(dev)
            if float(qm.sum()) > 0:
                za, _, _, _ = model.encode(*batch(pr))
                zb, _, _, _ = model.encode(*batch(pr + 2))
                d = (za - zb).pow(2).mean(-1)
                l_pers = (qm * d).sum() / qm.sum().clamp(min=1)

        loss = (args.w_ev * l_ev + args.w_mat * l_mat + args.w_now * l_now
                + args.w_res * l_res + args.w_pers * l_pers + vl)
        opt.zero_grad(); loss.backward(); opt.step()

        with torch.no_grad():
            m1 = ESQ[rt, 0].to(dev) >= 0
            sq_acc = float((sq[:, 0].argmax(-1)[m1] == ESQ[rt, 0].to(dev)[m1]).float().mean()) \
                if bool(m1.any()) else float("nan")
            mk = tgt_c >= 0
            pc = mc.argmax(-1)
            cnt_acc = float((pc[mk] == tgt_c[mk]).float().mean()) if bool(mk.any()) else float("nan")
            # THE HEADLINE PROXY: all 12 counts right at the LAST event -- the composed material
            # state, which is what naming an ending actually requires.
            last = tgt_c[:, N_EV - 1]; lm = (last >= 0).all(-1)
            mat_all = float((pc[:, N_EV - 1][lm] == last[lm]).all(-1).float().mean()) \
                if bool(lm.any()) else float("nan")
            kacc = float((ki[:, 0].argmax(-1)[m1] == EKI[rt, 0].to(dev)[m1]).float().mean()) \
                if bool(m1.any()) else float("nan")
            pn = model.predict_now(zq).argmax(-1)
            now_all = float((pn == NOW[rt].to(dev)).all(-1).float().mean())
            # CONDITIONAL ENDING: given the game reaches a tablebase, is the ending named?
            # cef2 regressed here (0.211 -> 0.193 against its floor) while NOW12 went to 1.000,
            # and the log showed none of it. Watch the thing that actually matters.
            rm = REG[rt].to(dev) >= 0
            eg_cond = float((e_l.argmax(-1)[rm] == REG[rt].to(dev)[rm]).float().mean()) \
                if bool(rm.any()) else float("nan")
            ps = [torch.bincount(ids[:, h], minlength=args.codes).float()
                  for h in range(args.heads)]
            perp = float(np.mean([float(torch.exp(-((p / p.sum()) *
                         ((p / p.sum()) + 1e-9).log()).sum())) for p in ps]))
            sqp = torch.bincount(ids_sq.reshape(-1), minlength=args.square_codes).float()
            sq_perp = float(torch.exp(-((sqp / sqp.sum()) * ((sqp / sqp.sum()) + 1e-9).log()).sum()))
        if step % args.eval_every == 0 or step == 1:
            el = time.time() - t0
            print(f"[cef] step {step:>6}/{args.steps} {el/60:>6.1f}m "
                  f"({el/max(1,step):.3f}s/step) | loss {float(loss.detach()):.3f} "
                  f"ev {float(l_ev.detach()):.3f} mat {float(l_mat.detach()):.3f} "
                  f"res {float(l_res.detach()):.3f} | ev_sq {sq_acc:.3f} kind {kacc:.3f} "
                  f"| count {cnt_acc:.3f} ALL12 {mat_all:.3f} NOW12 {now_all:.3f} "
                  f"eg_cond {eg_cond:.3f} | perp {perp:.1f} "
                  f"sq_perp {sq_perp:.1f}", flush=True)
        return {"loss": float(loss.detach()), "ev": float(l_ev.detach()),
                "mat": float(l_mat.detach()), "res": float(l_res.detach()),
                "pers": float(l_pers.detach()), "vq": float(vl.detach()),
                "now": float(l_now.detach()), "ev_sq_acc": sq_acc, "kind_acc": kacc,
                "count_acc": cnt_acc, "mat_all12": mat_all, "now_all12": now_all,
                "eg_cond": eg_cond,
                "perp": perp, "sq_perp": sq_perp, **res_parts}

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
    ap.add_argument("--w-ev", type=float, default=2.0, help="the next-5-events objective")
    ap.add_argument("--w-mat", type=float, default=2.0,
                    help="material left after each event -- the term this run exists for")
    ap.add_argument("--w-res", type=float, default=1.0)
    ap.add_argument("--w-now", type=float, default=0.25,
                    help="predict the CURRENT board's material from the concepts. cef1 could "
                         "only recover it 42.5%% of the time, which caps everything downstream.")
    ap.add_argument("--w-pers", type=float, default=0.25,
                    help="quiet-ply persistence. DEFAULT 0: persistence rewards codes that do "
                         "not change, while material changes at every capture, so the two are "
                         "plausibly in direct tension. Kaveh 2026-08-18 -- test by removal.")
    ap.add_argument("--cb-beta", type=float, default=0.999)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--test-frac", type=float, default=0.1)
    ap.add_argument("--ckpt-every", type=int, default=1000)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="artifacts/experiments/cef1")
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
