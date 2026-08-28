"""QGT trainer -- increment 4 of the locked spec (JOURNAL 2026-08-28, section 5 as
amended: carrier/observable split).

    base      GraphBase(NoMixer) -- relations only, trained FROM SCRATCH (no trunk)
    carrier   continuous per-square state, rolled LINEARLY: c' = K c + B m
    concepts  quantized OBSERVABLES of the carrier (RVQ level-1 + stacked QA levels),
              read after rolling, never rolled themselves
    readout   qat9 semantics verbatim: linear over the union + turn bias, W<->L
              permutation for Black, CE sees the move term detached

Losses (the handoff's list, nothing else):
    CE            PLAIN, class-weighted (NOT expected-score -- qat11b negative),
                  on logit_ce -> position path only (gradient routing, U13)
    regret        expectation regret + best-vs-random pairwise on SGN-signed (W-L)
                  -> sole trainer of the value move bits (no spoiler term)
    ROLLED VALUE  replaces EMA bit-matching ENTIRELY (no fut BCE, no EMA encoder):
      chain       roll the root carrier through the played dyn move codes of REAL
                  6-ply chains (the gko chain cache -- the load-bearing ingredient
                  of the 8/24 deep_roll table), read the position value at every
                  horizon 1..6, CE vs that row's known White-POV triple
      fan         roll the corpus parent ONE ply through each of several children's
                  dyn codes, CE vs each child's multiPV triple (free labels). The
                  anti-collapse duty: a constant carrier cannot predict per-child
                  values (amended section 5)
    + aux (LFQ entropy + ST-MoE router losses) + roller spectral penalty
    roller.clamp_() after every optimizer step

Judged on the DUAL SPLIT: rolled-value corr at ply 1/3/6 on held-out REAL chains
(the 8/24 reference curve reads 0.850/0.808/0.777) + even-lag observable deltas on
real AND random-uniform constructed sequences. Heartbeat carries regret/balanced as
qat does, plus vroll 1/3/6 and rho.

Deliberately absent: EMA encoder, fut BCE, expected-score, spoiler pairs, value
equivalence, slowness regularizer (all negative or superseded on the record).
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shards", default="data/derived/gko1_shards")
    ap.add_argument("--chain-graphs", default="data/derived/gko3_chain_graphs_mm.npz")
    ap.add_argument("--chain-starts", default="data/derived/gko1_chain_rows.npz")
    ap.add_argument("--chain-values", default="data/derived/gko1_chain_values.npz")
    ap.add_argument("--chain-fens", default="data/derived/gko1_chain_fens.npz")
    ap.add_argument("--out", default="artifacts/experiments/qgt1")
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--chain-batch", type=int, default=16)
    ap.add_argument("--fan-k", type=int, default=4,
                    help="children per position for the 1-ply rolled-value fan")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--ckpt-every", type=int, default=2000)
    ap.add_argument("--max-shards", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    # architecture (locked spec constants as defaults)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--sq-bits", type=int, default=24)
    ap.add_argument("--rvq-levels", type=int, default=2)
    ap.add_argument("--qa-levels", type=int, default=2)
    ap.add_argument("--n-q", type=int, default=16)
    ap.add_argument("--attn-heads", type=int, default=4)
    ap.add_argument("--top-k", type=int, default=4)
    ap.add_argument("--out-bits", type=int, default=8)
    ap.add_argument("--mv-bits", type=int, default=32)
    # readout / losses
    ap.add_argument("--l1-target", type=int, default=400)
    ap.add_argument("--prune-every", type=int, default=50)
    ap.add_argument("--w-rank", type=float, default=1.0)
    ap.add_argument("--w-roll", type=float, default=1.0,
                    help="weight on the rolled-value anchor (chain + fan)")
    ap.add_argument("--units", action="store_true")
    ap.add_argument("--eval-only", default="", metavar="CKPT")
    ap.add_argument("--resume", default="", metavar="CKPT",
                    help="resume a ladder checkpoint: weights + optimizer + RNG + step "
                         "(scaffold.resume_from). The pruner mask is NOT checkpointed; "
                         "it recomputes to the scheduled keep-count at the next prune "
                         "interval, so a resume costs at most one interval of full-width "
                         "readout (pruned columns are zeros in the weights anyway).")
    args = ap.parse_args()

    from catspace.research.tools.training_infra.train.scaffold import (
        TrainConfig, standard_train, resolve_device, resume_from)
    from catspace.research.tools.training_infra.pruning import (
        ConceptPruner, LossSettled)
    from catspace.research.tools.training_infra.qgt_field import QGTField
    from catspace.research.tools.npz_io import load_npz
    from catspace.research.components.encoder.approaches.reach_probability.experiments \
        .graph_move_policy import featurize, MV_CAP

    dev = resolve_device("auto")
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    N_NODES = 64 + MV_CAP

    # ---- corpus: STOCKFISH ROWS ONLY (qat pipeline verbatim, minus spoiler) ---------
    KEYS = ("sq", "gl", "mv", "mut", "vac", "dsc", "att", "n_mv", "n_att",
            "kind", "band", "tgt", "wdl", "pslot")
    store = {k: [] for k in KEYS}
    kept = 0
    for f in sorted(glob.glob(os.path.join(args.shards, "shard_*.npz")))[:args.max_shards]:
        z = load_npz(f)
        if "wdl" not in z.files:
            continue
        k = z["kind"]
        wd = z["wdl"].astype(np.float32)
        tg = z["tgt"]
        t = np.where(np.isfinite(tg), tg, -1e9)
        best = t.argmax(1)
        w = wd[np.arange(len(k)), best]
        fin_all = np.isfinite(wd).all(-1) & (wd.sum(-1) > 0)
        ok = (k == 0) & np.isfinite(w).all(-1) & (w.sum(-1) > 0)
        if not ok.any():
            continue
        idx = np.nonzero(ok)[0]
        store.setdefault("_cw", []).append(wd[idx])
        store.setdefault("_cv", []).append(fin_all[idx])
        for key in KEYS:
            store[key].append(np.ascontiguousarray(z[key][idx]))
        store.setdefault("_fen", []).append(np.asarray(z["fen"])[idx])
        kept += 1
    D = {k: torch.from_numpy(np.concatenate(v)) for k, v in store.items() if k in KEYS}
    CW_ = np.concatenate(store["_cw"]).astype(np.float32)     # (N, MV_CAP, 3)
    CV_ = np.concatenate(store["_cv"])                        # (N, MV_CAP) bool
    FENS = np.concatenate(store["_fen"])
    STM = np.array([str(f).split()[1] == "w" for f in FENS])
    # WHITE-POV: flip W<->L wherever Black is to move (inputs are absolute-color and
    # labels were the only mover-POV pressure -- verified 2026-08-27). Ranking stays
    # MOVER-relative via SGN (gate U9).
    _raw = CW_.copy()                                         # mover-POV, kept for U9
    CY_mover = CW_.argmax(-1)
    CW_[~STM] = CW_[~STM][..., ::-1]
    CY_ = CW_.argmax(-1)
    SGN = np.where(STM, 1.0, -1.0).astype(np.float32)
    TGT = D["tgt"].numpy()                                    # (N, MV_CAP) SF move score
    N = len(STM)
    ccls = np.bincount(CY_[CV_], minlength=3)
    print(f"[qgt] {N:,} Stockfish rows from {kept} shards | MULTI-PV {CV_.sum():,} child "
          f"labels | child mix W {100*ccls[0]/ccls.sum():.1f}% D "
          f"{100*ccls[1]/ccls.sum():.1f}% L {100*ccls[2]/ccls.sum():.1f}%", flush=True)

    # GROUP HOLDOUT by PAWN STRUCTURE (the leak-free split; same construction as qat).
    def _pk(f):
        bd = str(f).split()[0]; out = []
        for r_i, row in enumerate(bd.split("/")):
            c = 0
            for ch in row:
                if ch.isdigit(): c += int(ch)
                else:
                    if ch in "Pp": out.append(f"{ch}{r_i}{c}")
                    c += 1
        return "|".join(out)
    _keys = np.array([_pk(f) for f in FENS])
    _uniq = np.unique(_keys)
    _order = rng.permutation(len(_uniq))
    _want = int(0.10 * N); _te_keys = set(); _got = 0
    from collections import Counter
    _cnt = Counter(_keys.tolist())
    for _k in _uniq[_order]:
        if _got >= _want: break
        _te_keys.add(_k); _got += _cnt[_k]
    te_idx = np.nonzero(np.array([k in _te_keys for k in _keys]))[0]
    te_mask = np.zeros(N, bool); te_mask[te_idx] = True
    tr_idx = np.nonzero(~te_mask)[0]
    print(f"[qgt] holdout grouped by pawn structure: {len(_uniq):,} groups, "
          f"{len(te_idx):,} positions held out, {len(tr_idx):,} train", flush=True)

    # ---- chains: the REAL 6-ply chain cache (labels at every ply) -------------------
    _chg = np.load(args.chain_graphs, mmap_mode="r")
    CH_KEYS = ("sq", "gl", "mv", "mut", "vac", "dsc", "att", "n_mv", "n_att", "played")
    CHA = {k: _chg[k] for k in CH_KEYS}       # hold once; NpzFile re-reads per access
    CH_prow = np.asarray(_chg["p_row"])
    cs = np.load(args.chain_starts)
    starts, games = cs["start"], cs["game"]
    pos = np.searchsorted(CH_prow, starts[:, None] + np.arange(7)[None, :])
    okc = (pos < len(CH_prow))
    okc &= CH_prow[np.minimum(pos, len(CH_prow) - 1)] == (starts[:, None] + np.arange(7))
    keep = okc.all(1)
    CH_pos = pos[keep]; games = games[keep]
    gset = np.unique(games)
    rngc = np.random.default_rng(args.seed + 7)
    te_g = set(gset[rngc.random(len(gset)) < 0.08].tolist())
    te_m = np.array([g in te_g for g in games])
    ch_tr = np.flatnonzero(~te_m); ch_te = np.flatnonzero(te_m)
    _zv = np.load(args.chain_values)
    _wdl_raw = _zv["wdl"].astype(np.float32)                  # mover-POV best-move triple
    cf = np.load(args.chain_fens)
    CH_FEN = cf["fen"]
    CH_STM = np.array([str(f).split()[1] == "w" for f in CH_FEN])
    CH_WDL = _wdl_raw.copy()
    CH_WDL[~CH_STM] = CH_WDL[~CH_STM][:, ::-1]                # White-POV
    CH_OK = np.isfinite(CH_WDL).all(-1) & (np.nan_to_num(_wdl_raw).sum(-1) > 0)
    print(f"[qgt] chains {len(CH_pos):,} ({len(ch_te):,} held out by GAME) | "
          f"{int(CH_OK.sum()):,}/{len(CH_OK):,} rows carry a WDL triple", flush=True)

    # ---- model (from scratch -- no trunk, per the 8/27 trunk-removal design) --------
    model = QGTField(d=args.d, layers=args.layers, sq_bits=args.sq_bits,
                     rvq_levels=args.rvq_levels, mv_bits=args.mv_bits,
                     qa_levels=args.qa_levels, n_q=args.n_q, qa_heads=args.attn_heads,
                     top_k=args.top_k, out_bits=args.out_bits).to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    settle = LossSettled()
    # DEVIATION, recorded: the pruner sees POSITION columns only. The move columns are
    # the entire per-child ranking capacity (32 bits); qat pruned them alongside and
    # had to warn when they all died. The handoff marks this an open question -- the
    # conservative default is taken.
    with torch.no_grad():                      # a plain storage view, no autograd tail
        _pos_cols = model.readout.weight[:, :model.d_pos]
    pruner = ConceptPruner(_pos_cols, args.l1_target,
                           args.steps, interval=args.prune_every)
    print(f"[qgt] {n_par:,} params | carrier 64x{args.sq_bits} | levels: RVQ "
          f"{args.rvq_levels} + QA {args.qa_levels} | readout {model.d_pos}+{args.mv_bits}"
          f" -> prune position cols to {args.l1_target} (move cols exempt)", flush=True)

    CWt = torch.tensor(ccls.sum() / (3.0 * np.maximum(ccls, 1)),
                       dtype=torch.float32, device=dev)
    S3 = torch.tensor([1.0, 0.5, 0.0], device=dev)

    def feat_of(R):
        """R: dict of CPU tensors -> (sqf, mvf, edges, mask) on dev."""
        n = len(R["sq"])
        b = {k: R[k].to(dev) for k in ("sq", "gl", "mv", "mut", "att", "n_mv", "n_att")}
        sqf, mvf, edges, mask = featurize(b, dev, n_nodes=N_NODES)
        off = (torch.arange(n, device=dev) * N_NODES)[:, None]
        jj = torch.arange(MV_CAP, device=dev)[None, :]
        for name in ("vac", "dsc"):
            a2 = R[name].to(dev).long()
            val = (a2 != 255) & mask[:, :, None]
            edges[name] = (((jj + 64 + off)[:, :, None].expand_as(a2))[val],
                           (a2 + off[:, :, None])[val])
        return sqf, mvf, edges, mask

    def batch_of(idx):
        return feat_of({k: D[k][idx] for k in ("sq", "gl", "mv", "mut", "vac", "dsc",
                                               "att", "n_mv", "n_att")})

    def feat_chain(rows):
        R = {k: torch.from_numpy(np.ascontiguousarray(CHA[k][rows]))
             for k in CH_KEYS if k != "played"}
        sqf, mvf, edges, mask = feat_of(R)
        played = torch.from_numpy(np.ascontiguousarray(CHA["played"][rows])
                                  .astype(np.int64)).to(dev)
        return sqf, mvf, edges, mask, played

    def dyn_codes(h_mv):
        q, _, l = model.dyn_mv_lfq(model.dyn_mv_proj(h_mv))
        return q, (l.sum() if torch.is_tensor(l) else torch.zeros((), device=dev))

    def _ce_rows(pv, tg):
        """Plain CE of position-value logits vs a (n,3) triple, normalized."""
        tg = torch.nan_to_num(tg).clamp(min=0)
        tg = tg / tg.sum(-1, keepdim=True).clamp(min=1e-6)
        return -(tg * F.log_softmax(pv, -1)).sum(-1).mean()

    def chain_anchor(cb):
        """Roll REAL chains 6 plies; CE of read value vs each row's White-POV triple."""
        ci = CH_pos[ch_tr[rng.integers(0, len(ch_tr), cb)]]           # (cb,7)
        sq0, mv0, ed0, mk0, pl0 = feat_chain(ci[:, 0])
        c0, h_mv0, _ = model.carrier(sq0, mv0, ed0, mk0)
        dq0, l_dq = dyn_codes(h_mv0)
        ar = torch.arange(cb, device=dev)
        moves = [dq0[ar, pl0]]
        with torch.no_grad():                  # deeper move codes: conditioning only
            rows_mid = ci[:, 1:6].ravel()
            sqm, mvm, edm, mkm, plm = feat_chain(rows_mid)
            _, h_mvm, _ = model.carrier(sqm, mvm, edm, mkm)
            dqm = dyn_codes(h_mvm)[0]
            mm = dqm[torch.arange(len(rows_mid), device=dev), plm].reshape(cb, 5, -1)
        moves += [mm[:, k] for k in range(5)]
        stm_seq = [torch.from_numpy(CH_STM[ci[:, k]].astype(np.int64)).to(dev)
                   for k in range(1, 7)]
        outs = model.roll_and_read(c0, moves, stm_seq)
        l = torch.zeros((), device=dev)
        n_lab = 0
        for k in range(1, 7):
            fin = torch.from_numpy(CH_OK[ci[:, k]]).to(dev)
            if not bool(fin.any()):
                continue
            l = l + _ce_rows(outs[k - 1][1][fin],
                             torch.from_numpy(CH_WDL[ci[:, k]]).to(dev)[fin])
            n_lab += int(fin.sum())
        return l / 6.0, l_dq, n_lab

    def fan_anchor(c_par, h_mv_par, ii, stm_b):
        """Roll the parent 1 ply through fan_k children each; CE vs each child's triple.
        Anti-collapse: different children carry different values."""
        fk = args.fan_k
        dq, l_dq = dyn_codes(h_mv_par)                                # (B,MV_CAP,32)
        ok = CV_[ii] & np.isfinite(CW_[ii]).all(-1)
        B = len(ii)
        cols = np.zeros((B, fk), np.int64)
        keep = np.zeros(B, bool)
        for r in range(B):
            v = np.flatnonzero(ok[r])
            if len(v):
                cols[r] = rng.choice(v, fk, replace=len(v) < fk)
                keep[r] = True
        if not keep.any():
            return torch.zeros((), device=dev), l_dq
        kt = torch.from_numpy(keep).to(dev)
        colst = torch.from_numpy(cols[keep]).to(dev)                  # (nk,fk)
        nk = int(kt.sum())
        ck = c_par[kt].reshape(nk, 1, -1).expand(-1, fk, -1).reshape(nk * fk, -1)
        mk_ = dq[kt].gather(1, colst[..., None].expand(-1, -1, dq.shape[-1])
                            ).reshape(nk * fk, -1)
        c1 = model.roller.step(ck, mk_)
        z1, _ = model.observe(c1.reshape(-1, 64, model.sq_bits))
        pv = F.linear(z1, model.readout.weight[:, :model.d_pos], model.readout.bias)
        stm_c = (1 - stm_b[kt]).repeat_interleave(fk)         # side flips after a ply
        pv = pv + model.turn[stm_c]
        tg = torch.from_numpy(CW_[ii][keep][np.arange(nk)[:, None], cols[keep]]
                              ).to(dev).reshape(nk * fk, 3)
        return _ce_rows(pv, tg), l_dq

    def step_fn(_m, step):
        model.train()
        idx = torch.from_numpy(rng.choice(tr_idx, args.batch, replace=False))
        ii = idx.numpy()
        stm_b = torch.from_numpy(STM[ii].astype(np.int64)).to(dev)
        sqf, mvf, edges, mask = batch_of(idx)
        logit, logit_ce, z, aux = model(sqf, mvf, edges, mask, stm=stm_b)
        vld = torch.from_numpy(CV_[ii]).to(dev)
        # the target must be finite BEFORE the multiply (nan*0 is nan -- the scar)
        tgt = torch.nan_to_num(torch.from_numpy(CW_[ii]).to(dev)) * vld[..., None]
        tgt = tgt / tgt.sum(-1, keepdim=True).clamp(min=1e-6)
        w = CWt[torch.from_numpy(np.where(CV_[ii], CY_[ii], 0)).to(dev)] * vld.float()
        lp = F.log_softmax(logit_ce, -1)
        ce = -(tgt * lp).sum(-1)
        l_wdl = (ce * w).sum() / w.sum().clamp(min=1e-6)
        assert torch.isfinite(l_wdl), "WDL loss is not finite"
        loss = l_wdl + aux["loss_aux"]

        # EXPECTED REGRET + PAIRWISE ANTI-DEGENERACY (qat9's proven pair, verbatim)
        l_rank = torch.zeros((), device=dev)
        if args.w_rank > 0:
            wp_np = np.where(CV_[ii] & np.isfinite(TGT[ii]), TGT[ii], -np.inf)
            good = torch.from_numpy(np.isfinite(wp_np).any(1)).to(dev)
            if bool(good.any()):
                wp = torch.from_numpy(wp_np).to(dev)[good]
                bestwp = wp.max(1, keepdim=True).values
                reg = torch.where(torch.isfinite(wp), bestwp - wp,
                                  torch.zeros_like(wp)).clamp(min=0.0)
                sg = torch.from_numpy(SGN[ii]).to(dev)[good, None]
                sc = (sg * (logit[..., 0] - logit[..., 2])[good]).masked_fill(
                    ~vld[good], -1e9)
                l_rank = (torch.softmax(sc, -1) * reg).sum(-1).mean()
                bi = wp.argmax(1)
                rnd = torch.randint(0, wp.shape[1], (len(wp),), device=dev)
                v_ok = torch.isfinite(wp[torch.arange(len(wp), device=dev), rnd])
                if bool(v_ok.any()):
                    a_ = torch.arange(len(wp), device=dev)
                    gap = (wp[a_, bi] - wp[a_, rnd]).clamp(min=0)[v_ok]
                    d_ = (sc[a_, bi] - sc[a_, rnd])[v_ok]
                    l_rank = l_rank + (F.softplus(-d_) * gap).mean()
                assert torch.isfinite(l_rank), "regret loss is not finite"
                loss = loss + args.w_rank * l_rank

        # ROLLED-VALUE anchor (chain + fan) -- the sole dynamics supervision
        l_ch, l_dq_c, n_lab = chain_anchor(args.chain_batch)
        l_fan, l_dq_f = fan_anchor(aux["carrier"], aux["h_mv"], ii, stm_b)
        l_roll = l_ch + l_fan
        assert torch.isfinite(l_roll), "rolled-value loss is not finite"
        loss = loss + args.w_roll * l_roll + l_dq_c + l_dq_f \
            + model.roller.spectral_penalty()

        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        model.roller.clamp_()
        pruner.step(step, settled=settle.update(float(loss.detach())))

        out = {}
        if step % args.eval_every == 0 or step == 1 or step == args.steps:
            out = run_eval(step, float(l_wdl), float(l_rank), float(l_ch), float(l_fan))
        return out

    # ---- the dual-split instrument --------------------------------------------------
    @torch.no_grad()
    def roll_real(n_ch=192, lag_sub=48):
        """Held-out REAL chains: rolled-value corr at 1/3/6 + even-lag bit deltas."""
        model.eval()
        pred = {k: [] for k in (1, 3, 6)}
        true = {k: [] for k in (1, 3, 6)}
        dl = {k: [0.0, 0.0] for k in (2, 4, 6)}     # rolled-vs-true bits
        cp = {k: [0.0, 0.0] for k in (2, 4, 6)}     # copy baseline: root vs true bits
        for a in range(0, min(n_ch, len(ch_te)), 32):
            ci = CH_pos[ch_te[a:a + 32]]
            cb = len(ci)
            sq0, mv0, ed0, mk0, pl0 = feat_chain(ci[:, 0])
            c0, h_mv0, _ = model.carrier(sq0, mv0, ed0, mk0)
            dq0 = dyn_codes(h_mv0)[0]
            moves = [dq0[torch.arange(cb, device=dev), pl0]]
            rows_mid = ci[:, 1:6].ravel()
            sqm, mvm, edm, mkm, plm = feat_chain(rows_mid)
            _, h_mvm, _ = model.carrier(sqm, mvm, edm, mkm)
            dqm = dyn_codes(h_mvm)[0]
            mm = dqm[torch.arange(len(rows_mid), device=dev), plm].reshape(cb, 5, -1)
            moves += [mm[:, k] for k in range(5)]
            stm_seq = [torch.from_numpy(CH_STM[ci[:, k]].astype(np.int64)).to(dev)
                       for k in range(1, 7)]
            outs = model.roll_and_read(c0, moves, stm_seq)
            for k in (1, 3, 6):
                fin = CH_OK[ci[:, k]]
                if not fin.any():
                    continue
                p = (torch.softmax(outs[k - 1][1], -1) * S3).sum(-1).cpu().numpy()[fin]
                tw = CH_WDL[ci[:, k]][fin]
                tw = tw / tw.sum(-1, keepdims=True).clip(min=1e-6)
                pred[k].append(p); true[k].append(tw @ np.array([1.0, 0.5, 0.0]))
            if a < lag_sub:                          # even-lag deltas on a subsample
                z0 = model.observe(c0)[0]
                for k in (2, 4, 6):
                    sqk, mvk, edk, mkk, _ = feat_chain(ci[:, k])
                    ck_true, _, _ = model.carrier(sqk, mvk, edk, mkk)
                    zt = model.observe(ck_true)[0]
                    zr = outs[k - 1][0]
                    dl[k][0] += float((zr != zt).float().mean()) * cb; dl[k][1] += cb
                    cp[k][0] += float((z0 != zt).float().mean()) * cb; cp[k][1] += cb
        corr = {}
        for k in (1, 3, 6):
            if pred[k]:
                p = np.concatenate(pred[k]); t = np.concatenate(true[k])
                corr[k] = (float(np.corrcoef(p, t)[0, 1])
                           if p.std() > 1e-9 and t.std() > 1e-9 else 0.0)
            else:
                corr[k] = float("nan")
        delta = {k: dl[k][0] / max(dl[k][1], 1) for k in dl}
        copy = {k: cp[k][0] / max(cp[k][1], 1) for k in cp}
        return corr, delta, copy

    @torch.no_grad()
    def roll_random(n_root=24):
        """RANDOM-UNIFORM constructed sequences: even-lag bit deltas (no labels)."""
        import chess
        from catspace.research.tools.training_infra.gen_move_graphs import (
            parent_graph, glob_of_board)
        model.eval()
        ero = np.random.default_rng(1)
        roots = te_idx[ero.choice(len(te_idx), min(n_root, len(te_idx)), replace=False)]
        BKEYS = ("sq", "gl", "mv", "mut", "vac", "dsc", "att", "n_mv", "n_att")

        def build_rows(boards):
            acc = {k: [] for k in BKEYS}
            for b in boards:
                g, _ = parent_graph(b)
                for k in ("sq", "mv", "mut", "att", "vac", "dsc"):
                    acc[k].append(g[k])
                acc["n_mv"].append(np.uint8(g["n_mv"]))
                acc["n_att"].append(np.uint16(g["n_att"]))
                acc["gl"].append(glob_of_board(b))
            return {k: torch.from_numpy(np.stack(acc[k])) for k in acc}

        seqs, mv_js = [], []
        for g in roots:
            try:
                b = chess.Board(str(FENS[g]))
            except Exception:
                continue
            boards, js = [b], []
            okb = True
            for _ in range(6):
                legal = list(boards[-1].legal_moves)[:MV_CAP]
                if not legal:
                    okb = False; break
                j = int(ero.integers(len(legal)))
                nb = boards[-1].copy(); nb.push(legal[j])
                boards.append(nb); js.append(j)
            if okb:
                seqs.append(boards); mv_js.append(js)
        if not seqs:
            return {}
        n = len(seqs)
        flat = [b for boards in seqs for b in boards]                 # n*7 boards
        sqf, mvf, edges, mask = feat_of(build_rows(flat))
        c_all, h_mv_all, _ = model.carrier(sqf, mvf, edges, mask)
        c_all = c_all.reshape(n, 7, 64, model.sq_bits)
        dq = dyn_codes(h_mv_all)[0].reshape(n, 7, MV_CAP, -1)
        jt = torch.tensor(mv_js, device=dev)                          # (n,6)
        moves = [dq[torch.arange(n, device=dev), k].gather(
            1, jt[:, k][:, None, None].expand(-1, 1, dq.shape[-1]))[:, 0]
            for k in range(6)]
        stm0 = torch.tensor([b0[0].turn for b0 in seqs], dtype=torch.long, device=dev)
        stm_seq = [(stm0 + k + 1) % 2 for k in range(6)]
        outs = model.roll_and_read(c_all[:, 0], moves, stm_seq)
        z_true = {k: model.observe(c_all[:, k])[0] for k in (2, 4, 6)}
        return {k: float((outs[k - 1][0] != z_true[k]).float().mean()) for k in (2, 4, 6)}

    def run_eval(step, lw=float("nan"), lr_=float("nan"), lch=float("nan"),
                 lfan=float("nan")):
        model.eval()
        live = int((model.readout.weight[:, :model.d_pos].abs().sum(0) > 0).sum())
        with torch.no_grad():
            preds, ys, top1, nrank, base0 = [], [], 0, 0, 0
            reg_sum, nreg = 0.0, 0
            ero = np.random.default_rng(0)
            for a in range(0, len(te_idx), 64):
                sl = te_idx[a:a + 64]
                sqf, mvf, edges, mask = batch_of(torch.from_numpy(sl))
                lg = model(sqf, mvf, edges, mask,
                           stm=torch.from_numpy(STM[sl].astype(np.int64)).to(dev))[0]
                v = CV_[sl]
                preds.append(lg.argmax(-1).cpu().numpy()[v])
                ys.append(CY_[sl][v])
                p = torch.softmax(lg, -1).cpu().numpy()
                sc = np.where(v, SGN[sl][:, None] * (p[..., 0] - p[..., 2]), -np.inf)
                tv = np.where(np.isfinite(TGT[sl]) & v, TGT[sl], -np.inf)
                uniq = (tv == tv.max(1, keepdims=True)).sum(1) == 1
                ok = v.any(1) & uniq
                jit = ero.random(sc.shape) * 1e-9
                pick = np.where(v, sc + jit, -np.inf).argmax(1)
                top1 += int((pick[ok] == tv.argmax(1)[ok]).sum())
                base0 += int((tv.argmax(1)[ok] == 0).sum())
                nrank += int(ok.sum())
                any_v = v.any(1)
                bw = np.where(np.isfinite(TGT[sl]) & v, TGT[sl], -np.inf).max(1)
                gw = TGT[sl][np.arange(len(sl)), pick]
                fine = any_v & np.isfinite(bw) & np.isfinite(gw)
                reg_sum += float((bw[fine] - gw[fine]).clip(min=0).sum())
                nreg += int(fine.sum())
            pr, yy = np.concatenate(preds), np.concatenate(ys)
            rec = [float((pr[yy == c] == c).mean()) if (yy == c).any() else 0.0
                   for c in range(3)]
            bal = float(np.mean(rec))
            top1_acc = top1 / max(nrank, 1)
            mregret = reg_sum / max(nreg, 1)
            corr, delta, copy = roll_real()
            rnd = roll_random()
            rho = float(model.roller.rho(iters=20))
        print(f"[qgt] step {step:>5}/{args.steps} | wdl {lw:.3f} rloss {lr_:.4f} "
              f"roll {lch:.3f}+{lfan:.3f} | BALANCED {bal:.3f} recall W/D/L "
              f"{rec[0]:.2f}/{rec[1]:.2f}/{rec[2]:.2f} | REGRET {mregret:.4f} "
              f"(qat9 0.0872) | top1 {top1_acc:.3f} | concepts {live} | "
              f"vroll {corr[1]:+.3f}/{corr[3]:+.3f}/{corr[6]:+.3f} "
              f"(8/24 ref +0.850/+0.808/+0.777) | evenlag real "
              f"{delta[2]:.3f}/{delta[4]:.3f}/{delta[6]:.3f} (copy "
              f"{copy[2]:.3f}/{copy[4]:.3f}/{copy[6]:.3f}) rand "
              f"{rnd.get(2, float('nan')):.3f}/{rnd.get(4, float('nan')):.3f}/"
              f"{rnd.get(6, float('nan')):.3f} | rho {rho:.4f}", flush=True)
        mv_alive = int((model.readout.weight[:, model.d_pos:].abs().sum(0) > 0).sum())
        if mv_alive == 0:
            print("  WARNING: the MOVE half of the readout is dead -- ranking impossible",
                  flush=True)
        return {"balanced": bal, "concepts": live, "wdl": lw, "top1": top1_acc,
                "regret": mregret, "rank": lr_, "roll_chain": lch, "roll_fan": lfan,
                "vroll1": corr[1], "vroll3": corr[3], "vroll6": corr[6],
                "evenlag2_real": delta[2], "evenlag6_real": delta[6],
                "evenlag6_rand": rnd.get(6, float("nan")), "rho": rho,
                "move_concepts_alive": mv_alive}

    if args.eval_only:
        sd = torch.load(args.eval_only, map_location="cpu", weights_only=False)
        missing = model.load_state_dict(sd["state_dict"], strict=False)
        print(f"[qgt] eval-only: {args.eval_only} (step {sd.get('step', '?')}, "
              f"{len(missing.missing_keys)} missing)", flush=True)
        run_eval(int(sd.get("step", 0)))
        return

    # ---- gates ----------------------------------------------------------------------
    if args.units:
        idx = torch.from_numpy(rng.choice(tr_idx, 8, replace=False))
        ii = idx.numpy()
        stm8 = torch.from_numpy(STM[ii].astype(np.int64)).to(dev)
        sqf, mvf, edges, mask = batch_of(idx)
        logit, logit_ce, z, aux = model(sqf, mvf, edges, mask, stm=stm8)

        assert logit.shape == (8, MV_CAP, 3), f"per-child logits: {logit.shape}"
        print(f"  U1 PASS a WDL logit for every child: {tuple(logit.shape)}")

        u = set(torch.unique(z).tolist())
        assert u <= {-1.0, 1.0}, f"observables not binary: {sorted(u)[:6]}"
        print(f"  U2 PASS all {z.shape[-1]} observable dims strictly +-1 "
              f"(RVQ {model.d_l1} + QA {model.d_l2})")

        # U3: factorised readout == the honest concatenated form (stm=None path)
        lg_n, _, z_n, aux_n = model(sqf, mvf, edges, mask, stm=None)
        with torch.no_grad():
            mvb = model.mv_lfq(model.mv_proj(aux_n["h_mv"]))[0]
            ref = model.readout(torch.cat(
                [z_n[:, None, :].expand(-1, MV_CAP, -1), mvb], -1))
        err = float((ref - lg_n).detach().abs().max())
        assert err < 1e-4, f"factorisation disagrees: {err}"
        assert set(torch.unique(mvb).tolist()) <= {-1.0, 1.0}, "move concepts not bits"
        print(f"  U3 PASS position+move factorisation exact (max err {err:.2e})")

        sc8 = torch.softmax(logit, -1).detach()
        spread = float((sc8[..., 0] - sc8[..., 2]).std(dim=1).mean())
        assert spread > 1e-4, f"children scored alike (spread {spread:.2e})"
        print(f"  U4 PASS children scored apart (spread {spread:.4f})")

        # U9: White-POV algebra exact; labels flipped on Black rows (corpus + chains)
        m9 = CV_ & np.isfinite(_raw).all(-1)
        lhs = (SGN[:, None] * (CW_[..., 0] - CW_[..., 2]))[m9]
        rhs = (_raw[..., 0] - _raw[..., 2])[m9]
        err9 = float(np.abs(lhs - rhs).max())
        assert err9 < 1e-6, f"POV sign algebra broken: {err9}"
        flipped = int((CY_[~STM] != CY_mover[~STM]).sum())
        assert flipped > 0, "no Black-row label changed under White-POV"
        ch_fin = np.isfinite(_wdl_raw).all(-1) & ~CH_STM
        assert ch_fin.any() and not np.allclose(
            CH_WDL[ch_fin][:, 0], _wdl_raw[ch_fin][:, 0]), "chain POV flip inert"
        st7 = CH_STM[CH_pos[:64]]
        assert (st7[:, :-1] != st7[:, 1:]).all(), "chain side-to-move does not alternate"
        print(f"  U9 PASS White-POV exact (err {err9:.1e}); {flipped:,} corpus labels + "
              f"chain rows flipped; chain stm alternates")

        # U11: the move term transforms by the color-swap PERMUTATION
        with torch.no_grad():
            m1 = model(sqf, mvf, edges, mask,
                       stm=torch.ones(8, dtype=torch.long, device=dev))[0]
            m0 = model(sqf, mvf, edges, mask,
                       stm=torch.zeros(8, dtype=torch.long, device=dev))[0]
        sD = float((m1[..., 1] - m0[..., 1]).std(dim=1).max())
        sX = float((m1[..., 0] - m0[..., 2]).std(dim=1).max())
        assert sD < 1e-5 and sX < 1e-5, f"permutation broken: D {sD}, X {sX}"
        print(f"  U11 PASS color-swap permutation (D {sD:.1e}, W<->L {sX:.1e})")

        # U12-analogue: the rolled-value anchor trains DYN codes, never value codes
        l_ch, _, n_lab = chain_anchor(8)
        assert n_lab > 0, "chain anchor found no labelled rows -- it would train on nothing"
        g_val = torch.autograd.grad(l_ch, model.mv_proj.weight, retain_graph=True,
                                    allow_unused=True)[0]
        g_dyn = torch.autograd.grad(l_ch, model.dyn_mv_proj.weight, retain_graph=True,
                                    allow_unused=True)[0]
        g_K = torch.autograd.grad(l_ch, model.roller.K, allow_unused=True)[0]
        assert g_val is None or float(g_val.abs().sum()) == 0.0, \
            "rolled-value loss leaks into the VALUE move codes"
        assert g_dyn is not None and float(g_dyn.abs().sum()) > 0, \
            "rolled-value loss does not train the dyn codes"
        assert g_K is not None and float(g_K.abs().sum()) > 0, \
            "rolled-value loss does not reach K through the composition"
        print(f"  U12 PASS anchor isolation: value-code grad "
              f"{'None' if g_val is None else 0.0}; dyn codes + K train "
              f"({n_lab} labelled rows on 8 chains)")

        # U13: CE -> position only; regret is the sole trainer of the move bits
        tgt13 = torch.softmax(torch.randn_like(logit), -1)
        l_ce13 = -(tgt13 * F.log_softmax(logit_ce, -1)).sum(-1).mean()
        g13 = torch.autograd.grad(l_ce13, model.mv_proj.weight, retain_graph=True,
                                  allow_unused=True)[0]
        assert g13 is None or float(g13.abs().sum()) == 0.0, "CE trains the move codes"
        l_rk13 = torch.softmax((logit[..., 0] - logit[..., 2]), -1).mean()
        g13b = torch.autograd.grad(l_rk13, model.mv_proj.weight, retain_graph=True,
                                   allow_unused=True)[0]
        assert g13b is not None and float(g13b.abs().sum()) > 0, \
            "regret no longer trains the move codes"
        print(f"  U13 PASS gradient routing: CE->position, regret->move bits")

        # U14: roller health -- clamp works, rolled observables stay binary
        with torch.no_grad():
            model.roller.K.mul_(1.15)
        model.roller.clamp_()
        rr = float(model.roller.rho(iters=30))
        assert rr <= 1.005, f"clamp failed: {rr}"
        dm = [torch.randn(8, args.mv_bits, device=dev).sign() for _ in range(3)]
        outs3 = model.roll_and_read(aux["carrier"], dm, [stm8, 1 - stm8, stm8])
        zr = outs3[-1][0]
        assert set(torch.unique(zr).tolist()) <= {-1.0, 1.0}, "rolled bits not binary"
        print(f"  U14 PASS roller: rho {rr:.4f}<=1 after deliberate 1.15x; bits binary "
              f"after 3 rolls")

        # U15: the fan is non-degenerate -- the roller RECEIVES distinct inputs per
        # child and the fan loss TRAINS the parts that separate them (B, carrier).
        # Bits flipping at init is not required: the STE carries gradient either way.
        l_fan, _ = fan_anchor(aux["carrier"], aux["h_mv"], ii, stm8)
        assert torch.isfinite(l_fan) and float(l_fan) > 0, "fan anchor produced no loss"
        g_B = torch.autograd.grad(l_fan, model.roller.B.weight, retain_graph=True,
                                  allow_unused=True)[0]
        g_cp = torch.autograd.grad(l_fan, model.carrier_proj.weight,
                                   retain_graph=True, allow_unused=True)[0]
        assert g_B is not None and float(g_B.abs().sum()) > 0, \
            "fan loss does not train the roller's move injection B"
        assert g_cp is not None and float(g_cp.abs().sum()) > 0, \
            "fan loss does not reach the carrier"
        # A position's children must not ALL alias to one dyn code (adjacent moves
        # colliding at init is fine -- the STE separates them under training pressure)
        dq15 = dyn_codes(aux["h_mv"])[0]
        v15 = np.flatnonzero(CV_[ii][0])
        codes = {tuple(dq15[0, j].detach().cpu().numpy().tolist()) for j in v15}
        assert len(codes) > 1, f"ALL {len(v15)} children share one dyn code -- the " \
            "roller cannot tell any move apart"
        print(f"  U15 PASS fan anti-collapse: grads reach B + carrier; "
              f"{len(codes)}/{len(v15)} distinct dyn codes among one position's "
              f"children at init")

        logit.sum().backward()
        for nm, prm in (("readout", model.readout.weight),
                        ("move projection", model.mv_proj.weight),
                        ("carrier projection", model.carrier_proj.weight),
                        ("relation", model.base.rel[0]["att"].w.weight)):
            assert prm.grad is not None and prm.grad.abs().sum() > 0, f"no grad: {nm}"
        gm = model.readout.weight.grad[:, model.d_pos:].abs().sum()
        assert float(gm) > 0, "no gradient into the MOVE half of the readout"
        print(f"  U5 PASS gradients reach readout (both halves), mv_proj, carrier, "
              f"relations")
        print("  ALL QGT GATES PASS")
        return

    start = resume_from(args.resume, model, opt=opt, rng=rng) if args.resume else 0
    cfg = TrainConfig(out=args.out, steps=args.steps, ckpt_every=args.ckpt_every,
                      eval_every=args.eval_every, start_step=start,
                      experiment="catspace", run_name=args.out.split("/")[-1],
                      device=str(dev))
    standard_train(step_fn, model, cfg, args, opt=opt, rng=rng)


if __name__ == "__main__":
    main()
