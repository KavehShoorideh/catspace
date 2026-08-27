"""Quantized-attention concepts -> L1-linear readout -> Stockfish WDL.

Kaveh 2026-08-26: "let's get Maia out of it entirely... we just want to get the WDL of
Stockfish using quantized attention. That should be the fastest iteration."

ONE question: can an L1-sparse LINEAR readout over binary concepts, enriched by
QUANTIZED ATTENTION over those concepts, reproduce Stockfish's WDL?

Deliberately absent, because they are separate questions that made each iteration eight
hours: Maia policy, the future-KIND auxiliary, the Koopman operator, six-ply chains,
VAMP, the sibling ranking loss, the interval head.

Present:
  level 1   binary concepts from the frozen-ish trunk (LFQ, every dim +-1)
  level 2   quantized attention over concept tokens -- discrete question, sparse
            answer, quantized result (see tools/training_infra/quantized_attention.py)
  readout   Linear(concepts -> 3) + softmax, pruned to a concept budget by gradual
            magnitude pruning
  dynamics  1- and 2-ply future concept heads with EMA targets. Nonlinear, on purpose:
            the interpretability constraint is on the READOUT, not the transition.

Arms, selected with --arm, sharing trunk, data, schedule and seed:
    level1 readout on level-1 concepts only            (does attention help at all?)
    qat    readout on level-1 + quantized level-2      (the proposal)
    soft   the same with UNCONSTRAINED attention       (what quantization costs)

Reported on a STRATIFIED held-out set as balanced accuracy and per-class recall. Never
raw accuracy: the corpus is 63/14/23 W/D/L, so "always predict win" scores 0.637 for
free, and that skew has already reversed one conclusion in this project.
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
    ap.add_argument("--arm", choices=("level1", "l1", "qat", "soft"), default="qat",
                    help="level1 = level-1 concepts only, NO attention (l1 is a legacy "
                         "alias; it means level-ONE, not L1 regularisation). "
                         "qat = + quantized attention. soft = + ordinary attention.")
    ap.add_argument("--shards", default="data/derived/gko1_shards")
    ap.add_argument("--resume-field", default="artifacts/experiments/gcf4h_latest.pt")
    ap.add_argument("--out", default="artifacts/experiments/qat1")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--future-batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--ckpt-every", type=int, default=1000)
    ap.add_argument("--max-shards", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    # level 2
    ap.add_argument("--n-q", type=int, default=16, help="distinct questions per head")
    ap.add_argument("--attn-heads", type=int, default=4)
    ap.add_argument("--top-k", type=int, default=4, help="how many tokens may answer")
    ap.add_argument("--out-bits", type=int, default=8, help="level-2 bits per token")
    ap.add_argument("--mv-bits", type=int, default=32, help="binary concepts per MOVE")
    ap.add_argument("--pov", choices=("mover", "white"), default="mover",
                    help="'white': labels converted to White-POV so concepts describe the "
                         "BOARD, not the mover's view (a null move flips 54%% of mover-POV "
                         "bits -- tools/diag_color_structure.py). Ranking stays "
                         "mover-relative via a sign; the readout gets a turn bias.")
    ap.add_argument("--l2-scope", choices=("all", "squares"), default="all",
                    help="which concept families level-2 attention spans. 'all' = board + "
                         "squares + pieces + edge histogram (129 tokens). 'squares' is the "
                         "original 64-token version, kept only to score old checkpoints.")
    # readout
    ap.add_argument("--l1-target", type=int, default=400)
    ap.add_argument("--prune-every", type=int, default=50)
    # dynamics
    ap.add_argument("--w-rank", type=float, default=0.0,
                    help="expected-REGRET loss: penalise a move by the win "
                         "probability it gives up against the best move, so near-ties "
                         "cost almost nothing and blunders cost a lot. See "
                         "tools/diag_regret.py.")
    ap.add_argument("--rank-temp", type=float, default=0.1,
                    help="temperature on tgt when forming the target ranking distribution")
    ap.add_argument("--w-slow", type=float, default=0.0,
                    help="SPARSE TEMPORAL PRIOR on the ENCODER: penalise HOW MANY concepts "
                         "flip per ply. Kaveh 2026-08-26: 'we want the fewest number of "
                         "concept changes per ply possible in order for us to be able to "
                         "correctly predict the evaluation'. Applied at EVEN lag only -- "
                         "at odd lags 54%% of bits flip from side-to-move parity, which is "
                         "a representational artefact, not dynamics.")
    ap.add_argument("--w-future", type=float, default=0.3)
    ap.add_argument("--future-hidden", type=int, default=1024)
    ap.add_argument("--ema-decay", type=float, default=0.999)
    ap.add_argument("--units", action="store_true")
    ap.add_argument("--eval-only", default="", metavar="CKPT",
                    help="load CKPT, run one evaluation, print, exit. For scoring "
                         "finished runs on metrics added after they were trained.")
    args = ap.parse_args()
    if args.arm == "l1":                    # legacy spelling of level1
        args.arm = "level1"

    from catspace.research.tools.training_infra.train.scaffold import (
        TrainConfig, standard_train, resolve_device)
    from catspace.research.tools.training_infra.quantized_attention import (
        QuantizedAttention)
    from catspace.research.tools.training_infra.future_concepts_head import (
        FuturePredictor, evaluate as future_eval)
    from catspace.research.tools.training_infra.pruning import ConceptPruner
    from catspace.research.tools.training_infra.qat_model import QATReadout
    from catspace.research.tools.npz_io import load_npz
    from catspace.research.components.encoder.approaches.reach_probability.experiments \
        .graph_concept_field import GraphConceptField
    from catspace.research.components.encoder.approaches.reach_probability.experiments \
        .graph_move_policy import featurize, MV_CAP
    from catspace.research.components.encoder.approaches.reach_probability.src \
        .trajectories import normalise_elo

    dev = resolve_device("auto")
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    # ---- data: STOCKFISH ROWS ONLY --------------------------------------------------
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
        # a usable row: Stockfish-labelled, and its best move carries a finite WDL
        t = np.where(np.isfinite(tg), tg, -1e9)
        best = t.argmax(1)
        w = wd[np.arange(len(k)), best]
        fin_all = np.isfinite(wd).all(-1) & (wd.sum(-1) > 0)      # per-move validity
        ok = (k == 0) & np.isfinite(w).all(-1) & (w.sum(-1) > 0)
        if not ok.any():
            continue
        idx = np.nonzero(ok)[0]
        # MULTI-PV: Stockfish was asked about EVERY child move, so every legal move with
        # a finite WDL is a label. Keeping only the best move discards ~32 of 33 of them
        # -- and discards precisely the contrast that says which moves are BAD.
        store.setdefault("_cw", []).append(wd[idx])                   # (n, MV_CAP, 3)
        store.setdefault("_cv", []).append(fin_all[idx])              # (n, MV_CAP) bool
        for key in KEYS:
            store[key].append(np.ascontiguousarray(z[key][idx]))
        store.setdefault("_best", []).append(best[idx])
        store.setdefault("_wdl", []).append(w[idx])
        store.setdefault("_row", []).append(idx)          # row index WITHIN its shard
        store.setdefault("_fen", []).append(np.asarray(z["fen"])[idx])
        store.setdefault("_shard", []).append(np.full(len(idx), kept, np.int32))
        kept += 1
    D = {k: torch.from_numpy(np.concatenate(v)) for k, v in store.items()
         if k in KEYS}
    WDL = np.concatenate(store["_wdl"])
    CW_ = np.concatenate(store["_cw"]).astype(np.float32)     # (N, MV_CAP, 3) child WDL
    CV_ = np.concatenate(store["_cv"])                        # (N, MV_CAP) which are real
    CY_ = CW_.argmax(-1)                                      # child class
    STM = np.array([str(f).split()[1] == "w" for f in np.concatenate(store["_fen"])])
    SGN = np.ones(len(STM), np.float32)
    if args.pov == "white":
        # WHITE-POV: flip W<->L wherever Black is to move. Inputs are absolute-color and
        # labels were the only mover-POV pressure (verified 2026-08-27), so this is the
        # entire conversion. Ranking stays MOVER-relative: score = SGN * (W - L), and
        # SGN*(W'-L') on the flipped triple equals the mover's (W-L) EXACTLY (gate U9).
        CY_mover = CY_.copy()
        CW_[~STM] = CW_[~STM][..., ::-1]
        CY_ = CW_.argmax(-1)
        SGN = np.where(STM, 1.0, -1.0).astype(np.float32)
    TGT = D["tgt"].numpy()                                    # (N, MV_CAP) SF move score
    ROW = np.concatenate(store["_row"])
    SHARD = np.concatenate(store["_shard"])
    Y = WDL.argmax(1)
    N = len(Y)
    cls = np.bincount(Y, minlength=3)
    ccls = np.bincount(CY_[CV_], minlength=3)
    print(f"[qat] {N:,} Stockfish rows from {kept} shards | best-move W {cls[0]} "
          f"D {cls[1]} L {cls[2]}", flush=True)
    print(f"[qat] MULTI-PV: {CV_.sum():,} child-move WDL labels ({CV_.sum()/max(N,1):.1f} "
          f"per row) | W {100*ccls[0]/ccls.sum():.1f}% D {100*ccls[1]/ccls.sum():.1f}% "
          f"L {100*ccls[2]/ccls.sum():.1f}%  <- the mix the readout is trained on",
          flush=True)

    # Held out by POSITION. Splitting on (row, move) pairs would put siblings of the
    # same board on both sides, and since they share every position concept that is
    # leakage. Balanced accuracy over the children of held-out positions is already
    # skew-proof -- a majority-guesser scores 1/3 -- so the natural child mix is kept
    # rather than subsampled, which leaves far more draws in the test set.
    te_idx = rng.choice(N, max(1, int(0.10 * N)), replace=False)
    te_mask = np.zeros(N, bool); te_mask[te_idx] = True
    tr_idx = np.nonzero(~te_mask)[0]
    tec = np.bincount(CY_[te_idx][CV_[te_idx]], minlength=3)
    print(f"[qat] holdout {len(te_idx):,} POSITIONS -> {CV_[te_idx].sum():,} child labels "
          f"(W {tec[0]} D {tec[1]} L {tec[2]}); train {len(tr_idx):,} positions", flush=True)

    # ---- model ---------------------------------------------------------------------
    ck = torch.load(args.resume_field, map_location="cpu", weights_only=False)
    A = dict(ck["args"])
    field = GraphConceptField(
        d=A["d_model"], layers=A["layers"], slow_heads=A["slow_heads"], n_res=3, n_eg=1,
        cls_reach=np.zeros(3, np.int64), cls_eg=np.zeros(3, np.int64),
        cls_out=np.arange(3, dtype=np.int64), edge_codes=A["edge_codes"],
        move_codes=A["move_codes"], gps_parallel=bool(A["gps_parallel"]),
        attn_bias=bool(A["attn_bias"]), agg_max=bool(A["agg_max"]), kin=bool(A["kin"]),
        file_nodes=bool(A["file_nodes"]), phase_head=bool(A["phase_head"]),
        threat_head=bool(A["threat_head"]), elo_cond=True,
        g_heads=32, g_codes=8, d_code=8, sq_codes=A.get("sq_codes", 256),
        mh_vq=True, gen5=True, gen6=True, binary=True, mv_concepts=True).to(dev)
    sdf = {k: v for k, v in ck["state_dict"].items()
           if k in field.state_dict() and field.state_dict()[k].shape == v.shape}
    field.load_state_dict(sdf, strict=False)
    print(f"[qat] trunk warm-start: {len(sdf)}/{len(field.state_dict())} tensors", flush=True)

    D_STATE = field.d_state
    BINARY_END = 96 + 64 * field.sq_width      # board + squares: the strictly +-1 dims
    PIECE_END = D_STATE - field.edge_codes     # piece slots end where the histogram starts
    PIECE_W = max(1, (PIECE_END - BINARY_END) // 32)          # bits per piece slot
    N_TOK, D_TOK = 64, field.sq_width          # per-square concept tokens for attention

    # architecture lives in tools/training_infra/qat_model.py so the planner, the
    # arena and any probe can build the same thing; it used to be a closure in here and
    # nothing downstream could construct it.
    model = QATReadout(field, arm=args.arm, n_q=args.n_q, attn_heads=args.attn_heads,
                       top_k=args.top_k, out_bits=args.out_bits, mv_bits=args.mv_bits,
                       n_tok=N_TOK, scope=args.l2_scope).to(dev)
    fut = FuturePredictor(field, D_STATE, args.mv_bits, plies=(1, 2),
                          ema_decay=args.ema_decay, hidden=args.future_hidden).to(dev)
    # The 2-ply target depends on TWO moves; conditioning it on one made part of the
    # target irreducible noise (the 2026-08-26 finding). The 2-ply head now takes both
    # move codes. Built BEFORE the optimizer so its parameters are optimised.
    from catspace.research.tools.training_infra.future_concepts_head import FutureConceptHead
    fut.heads["2"] = FutureConceptHead(D_STATE, 2 * args.mv_bits, 2,
                                       hidden=args.future_hidden).to(dev)
    opt = torch.optim.Adam(list(model.parameters()) + list(fut.heads.parameters()),
                           lr=args.lr)
    pruner = ConceptPruner(model.readout.weight, args.l1_target, args.steps,
                           interval=args.prune_every)
    if args.arm != "level1":
        print(f"[qat] level-2 spans {[t[0] for t in model.spans]} = {model.n_tok} tokens "
              f"x {args.out_bits} bits", flush=True)
    print(f"[qat] arm={args.arm} | readout reads {model.d_read} dims -> pruned to "
          f"{args.l1_target} | level-2 {'OFF (no attention)' if args.arm=='level1' else f'{model.n_tok}x{args.out_bits} bits'}",
          flush=True)

    # class weights: the corpus is skewed, and an unweighted loss lets the head ignore
    # draws for free (measured: draw recall 0.138 vs 0.519 when uncorrected)
    CW = torch.tensor(ccls.sum() / (3.0 * np.maximum(ccls, 1)),
                      dtype=torch.float32, device=dev)

    def featurize_rows(R):
        """R: dict of CPU tensors for a batch of positions, however they were built."""
        n = len(R["sq"])
        b = {k: R[k].to(dev) for k in ("sq", "gl", "mv", "mut", "att", "n_mv", "n_att")}
        sqf, mvf, edges, mask = featurize(b, dev, n_nodes=field.n_nodes)
        off = (torch.arange(n, device=dev) * field.n_nodes)[:, None]
        jj = torch.arange(MV_CAP, device=dev)[None, :]
        for name in ("vac", "dsc"):
            a2 = R[name].to(dev).long()
            val = (a2 != 255) & mask[:, :, None]
            edges[name] = (((jj + 64 + off)[:, :, None].expand_as(a2))[val],
                           (a2 + off[:, :, None])[val])
        band = R["band"].float()
        elo_v = torch.where(band > 0, band, torch.full_like(band, 2750.0))
        elo = torch.from_numpy(normalise_elo(
            elo_v.numpy().astype(np.float32)))[:, None].repeat(1, 2).to(dev)
        return sqf, mvf, edges, mask, elo, R["pslot"].to(dev)

    RKEYS = ("sq", "gl", "mv", "mut", "att", "n_mv", "n_att", "vac", "dsc", "band", "pslot")

    def batch_of(idx):
        return featurize_rows({k: D[k][idx] for k in RKEYS})

    def encode_rows(R, enc=None):
        sqf, mvf, edges, mask, elo, psl = featurize_rows(R)
        e = enc or field
        zq, _, _, _, h_mv, _ = e.encode(sqf, mvf, edges, mask, elo=elo, pslot=psl)[:6]
        return zq, h_mv, mask

    def encode(idx, enc=None):
        sqf, mvf, edges, mask, elo, psl = batch_of(idx)
        e = enc or field
        zq, _, _, _, h_mv, _ = e.encode(sqf, mvf, edges, mask, elo=elo, pslot=psl)[:6]
        return zq, h_mv, mask

    # ---- future targets: BUILD the child, do not look for it ------------------------
    # These shards are shuffled INDEPENDENT positions -- all 2048 FENs in a shard are
    # distinct and their fullmove numbers are non-monotone -- so consecutive rows are
    # not consecutive plies and there is no parent/child adjacency to exploit. Measured:
    # 0 of 8 consecutive row pairs were a real parent/child. An earlier version silently
    # reported `fut 0.000` for a whole run because of this.
    #
    # So the child is CONSTRUCTED: push a move and re-featurise the resulting board with
    # the same parent_graph() the corpus itself was generated from. That is strictly
    # better than adjacency -- we choose the move, so the head's conditioning matches its
    # target exactly, and the 1-ply future is the position after Stockfish's own best
    # move rather than whatever a human happened to play.
    import chess
    from catspace.research.tools.training_infra.gen_move_graphs import (
        parent_graph, glob_of_board)
    FENS = np.concatenate(store["_fen"]) if store.get("_fen") else None

    def build_rows(boards, band):
        """python-chess Boards -> the row dict featurize_rows() consumes."""
        acc = {k: [] for k in RKEYS}
        for b in boards:
            g, _ = parent_graph(b)
            for k in ("sq", "mv", "mut", "att", "vac", "dsc", "pslot"):
                acc[k].append(g[k])
            acc["n_mv"].append(np.uint8(g["n_mv"])); acc["n_att"].append(np.uint8(g["n_att"]))
            acc["gl"].append(glob_of_board(b))
        out = {k: torch.from_numpy(np.stack(acc[k])) for k in acc if k != "band"}
        out["band"] = band
        return out

    def future_batch(n):
        """(parent rows, 1-ply child rows, 2-ply grandchild rows, played-move slot).

        The 1-ply move is Stockfish's best. The 2-ply reply is sampled uniformly from the
        opponent's legal moves, so the 2-ply head learns the concepts EXPECTED after that
        reply distribution rather than one arbitrary line -- deliberate, and the copy
        baseline in the lift metric is what says whether it learned anything.
        """
        pick = rng.choice(tr_idx, n * 2, replace=False)
        par, c1, c2, cols, cols2, bands = [], [], [], [], [], []
        for g in pick:
            try:
                b0 = chess.Board(str(FENS[g]))
            except Exception:
                continue
            legal = list(b0.legal_moves)[:int(D["n_mv"][g])]
            if not legal:
                continue
            tv = np.where(np.isfinite(TGT[g][:len(legal)]), TGT[g][:len(legal)], -np.inf)
            j = int(tv.argmax()) if np.isfinite(tv).any() else 0
            b1 = b0.copy(); b1.push(legal[j])
            rep = list(b1.legal_moves)[:MV_CAP]          # move tokens exist only to the cap
            if not rep:
                continue
            j2 = int(rng.integers(len(rep)))
            b2 = b1.copy(); b2.push(rep[j2])
            par.append(b0); c1.append(b1); c2.append(b2)
            cols.append(j); cols2.append(j2); bands.append(float(D["band"][g]))
            if len(par) >= n:
                break
        if not par:
            return None
        bd = torch.tensor(bands)
        return (build_rows(par, bd), build_rows(c1, bd), build_rows(c2, bd),
                torch.tensor(cols, dtype=torch.long),
                torch.tensor(cols2, dtype=torch.long))

    def step_fn(_m, step):
        model.train(); field.train()
        idx = torch.from_numpy(rng.choice(tr_idx, args.batch, replace=False))
        zq, h_mv, _ = encode(idx)
        stm_b = (torch.from_numpy(STM[idx.numpy()].astype(np.int64)).to(dev)
                 if args.pov == "white" else None)
        logit, _, aux = model(zq, h_mv, stm=stm_b)           # (B, MV_CAP, 3)
        ii = idx.numpy()
        vld = torch.from_numpy(CV_[ii]).to(dev)              # only real, finite children
        # Invalid move slots hold NaN. Masking only the WEIGHT is not enough, because
        # nan * 0 is still nan and one poisoned slot takes the whole sum with it -- which
        # is exactly how this run trained on nothing for 200 steps while every other
        # gauge looked alive. The target itself has to be finite BEFORE the multiply.
        tgt = torch.nan_to_num(torch.from_numpy(CW_[ii]).to(dev)) * vld[..., None]
        tgt = tgt / tgt.sum(-1, keepdim=True).clamp(min=1e-6)
        w = CW[torch.from_numpy(np.where(CV_[ii], CY_[ii], 0)).to(dev)] * vld.float()
        ce = -(tgt * F.log_softmax(aux["logit_ce"], -1)).sum(-1)
        l_wdl = (ce * w).sum() / w.sum().clamp(min=1e-6)
        assert torch.isfinite(l_wdl), "WDL loss is not finite"
        loss = l_wdl + (aux.get("vq_loss", 0.0) if aux else 0.0)

        # EXPECTED REGRET -- average centipawn loss, in win-probability units.
        # Kaveh 2026-08-26: "penalize based on centipawn loss compared to the best move.
        # So even if we pick number two and it is really close to number one, the penalty
        # is small. But if number two is far from number one, then it is large."
        #
        #     L = sum_m p_theta(m) * [ wp(best) - wp(m) ]
        #
        # Top-1 accuracy cannot tell a near-tie from a blunder; this can, and it is the
        # standard chess measure. Measured on labels alone: a PERFECT WDL ranker has mean
        # regret 0.034 and MEDIAN 0.000, against 0.266 for a random legal move -- so the
        # moves WDL "gets wrong" are near-equivalent and top-1 was the wrong yardstick.
        # See tools/diag_regret.py. Off by default.
        l_rank = torch.zeros((), device=dev)
        if args.w_rank > 0:
            wp_np = np.where(CV_[ii] & np.isfinite(TGT[ii]), TGT[ii], -np.inf)
            good = torch.from_numpy(np.isfinite(wp_np).any(1)).to(dev)
            if bool(good.any()):
                wp = torch.from_numpy(wp_np).to(dev)[good]
                bestwp = wp.max(1, keepdim=True).values
                # regret is 0 for the best move and >=0 elsewhere; forced to 0 on invalid
                # slots, where the softmax mass is already ~0, so 0*inf never appears
                reg = torch.where(torch.isfinite(wp), bestwp - wp,
                                  torch.zeros_like(wp)).clamp(min=0.0)
                # mover-relative score under any POV: SGN flips the axis for Black
                sg = torch.from_numpy(SGN[ii]).to(dev)[good, None]
                sc = (sg * (logit[..., 0] - logit[..., 2])[good]).masked_fill(
                    ~vld[good], -1e9)
                l_rank = (torch.softmax(sc, -1) * reg).sum(-1).mean()
                assert torch.isfinite(l_rank), "regret loss is not finite"
                loss = loss + args.w_rank * l_rank

        l_fut = torch.zeros((), device=dev)
        l_slow = float("nan"); l_capt = float("nan")
        fp = future_batch(args.future_batch) if args.w_future > 0 else None
        if fp is not None:
            R0, R1, R2, col, col2 = fp
            z0, h0, _ = encode_rows(R0)
            with torch.no_grad():
                t1, h1m, _ = encode_rows(R1, enc=fut.ema)
                t2, _, _ = encode_rows(R2, enc=fut.ema)
            col, col2 = col.to(dev), col2.to(dev)
            ar = torch.arange(len(col), device=dev)
            # DYN codes, not the readout's value codes -- see qat_model for why
            mvb_d, _, vq_d = model.dyn_mv_bits(model.dyn_mv_proj(h0))
            played = mvb_d[ar, col]
            with torch.no_grad():               # reply code: conditioning input only
                mv2 = model.dyn_mv_bits(model.dyn_mv_proj(h1m))[0][ar, col2]
            mvc = {"1": played, "2": torch.cat([played, mv2], -1)}
            loss = loss + vq_d
            l_fut, fstats = fut.loss(z0, mvc, {"1": t1.detach(), "2": t2.detach()})
            loss = loss + args.w_future * l_fut

            # SPARSE TEMPORAL PRIOR. For +-1 concepts, (1 - z.z')/2 is EXACTLY the fraction
            # of concepts that flipped, and the gradient reaches the encoder through the
            # quantiser's straight-through estimator -- so this penalises the COUNT of
            # changes, not their magnitude. On binary variables L2 and L0 coincide anyway
            # (||z'-z||^2 = 4 * #flips); the reason to write it this way is that it stays
            # a fraction, so the weight means something and the number is readable.
            #
            # Only the strictly binary block (board + squares, dims 0..BINARY_END): piece
            # slots are TERNARY (0 = empty) and the last 128 dims are a histogram, so
            # z.z' is not +-1 there and the identity above would not hold.
            #
            # EVEN LAG ONLY. z0 and t2 are both the mover's view; z0 vs t1 differ by a
            # side-to-move flip that moves 54% of bits for representational reasons, and
            # penalising that would fight the encoding rather than slow the concepts.
            #
            # The trivial solution -- freeze every concept -- is priced out by the WDL and
            # regret losses above, which still have to be predicted from these concepts.
            if args.w_slow > 0:
                # GROUP FUSED LASSO on the 2-ply concept change.
                # Kaveh 2026-08-26: "how do we make the minimum number of concepts
                # participate in any change from one ply to two plies ahead?"
                #
                #     penalty = SUM_g || z_g(s') - z_g(s) ||_2        norm, NOT squared
                #
                # The inner L2 is indifferent to how a change distributes inside a group;
                # the OUTER SUM behaves like an L1 across groups and drives whole groups to
                # exactly zero change -- "each difference is typically either full or
                # identically zero" (group fused lasso). Squaring the norm would give plain
                # L2, under which every group drifts a little and none is ever still: that
                # is precisely the failure Kaveh objected to.
                #
                # Heterogeneity is free, because the outer sum only sees one SCALAR per
                # group, so each block may use the distance its geometry demands:
                #   squares / pieces  L2 over their bits   (pieces gated alive-at-both,
                #                                           captures being exogenous)
                #   edge histogram    TOTAL VARIATION      (a simplex cannot "flip")
                # BOTH STATES FROM THE ONLINE ENCODER. t2 comes from the EMA encoder,
                # which is right for the PREDICTION loss (a lagged target prevents the
                # collapse-to-constant solution) and WRONG here: comparing the current
                # encoder at s against a ~1000-step-stale one at s+2 measures encoder
                # DRIFT, not concept change. Proven, not guessed -- optimising the penalty
                # alone for 6 steps:
                #     online vs online   penalty 2.71 -> 1.12   groups moved 50.1 -> 37.7
                #     online vs EMA      penalty 2.71 -> 3.39   groups moved 50.1 -> 72.5
                # The EMA form pushes churn UP, which is exactly what qat5 did (46.4 groups
                # against qat4's 21.9 with no such term at all).
                n = len(z0)
                t2d = encode_rows(R2)[0]
                sq0 = z0[:, 96:BINARY_END].reshape(n, 64, field.sq_width)
                sq1 = t2d[:, 96:BINARY_END].reshape(n, 64, field.sq_width)
                d_sq = ((sq0 - sq1) ** 2).sum(-1).clamp(min=1e-8).sqrt()      # (n,64)

                zp = z0[:, BINARY_END:PIECE_END].reshape(n, 32, PIECE_W)
                tp = t2d[:, BINARY_END:PIECE_END].reshape(n, 32, PIECE_W)
                alive = ((zp.abs().sum(-1) > 0) & (tp.abs().sum(-1) > 0)).float()
                d_pc = ((zp - tp) ** 2).sum(-1).clamp(min=1e-8).sqrt() * alive  # (n,32)

                h0, h1 = z0[:, PIECE_END:], t2d[:, PIECE_END:]
                d_hi = 0.5 * (h0 - h1).abs().sum(-1)                            # (n,) TV

                ngrp = 64 + alive.sum(-1) + 1
                flip_frac = ((d_sq.sum(-1) + d_pc.sum(-1) + d_hi) / ngrp).mean()
                loss = loss + args.w_slow * flip_frac
                l_slow = float(flip_frac)
                # how many groups moved AT ALL -- the readable number, never optimised
                with torch.no_grad():
                    moved = ((d_sq > 1e-3).float().sum(-1) +
                             (d_pc > 1e-3).float().sum(-1) +
                             (d_hi > 1e-3).float())
                    l_capt = float(moved.mean())

        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(list(model.parameters()) +
                                       list(fut.heads.parameters()), 1.0)
        opt.step()
        live = pruner.step(step)
        fut.update_ema(field)

        out = {}
        if step % args.eval_every == 0 or step == 1 or step == args.steps:
            out = run_eval(step, float(l_wdl), float(l_fut), float(l_rank), l_slow, l_capt,
                           fp, locals().get("z0"), locals().get("mvc"),
                           locals().get("t1"), locals().get("t2"))
        return out

    def run_eval(step, lw, lf, lr, ls=float("nan"), lc=float("nan"), fp=None,
                 z0=None, mvc=None, t1=None, t2=None):
        live = int((model.readout.weight.abs().sum(0) > 0).sum())
        if True:
            model.eval(); field.eval()
            with torch.no_grad():
                preds, ys, top1, nrank, base0 = [], [], 0, 0, 0
                reg_sum, nreg = 0.0, 0
                ero = np.random.default_rng(0)      # fixed, so eval is reproducible
                for a in range(0, len(te_idx), 64):
                    sl = te_idx[a:a + 64]
                    zz, hm, _ = encode(torch.from_numpy(sl))
                    lg = model(zz, hm,
                               stm=(torch.from_numpy(STM[sl].astype(np.int64)).to(dev)
                                    if args.pov == "white" else None))[0]
                    v = CV_[sl]
                    preds.append(lg.argmax(-1).cpu().numpy()[v])
                    ys.append(CY_[sl][v])
                    # PLANNER-RELEVANT: does the readout pick the move Stockfish picked?
                    # Ranked against TGT, the per-move score, NOT the child WDL: WDL
                    # saturates and its within-position top ties in most positions, so
                    # ranking on it rewards a constant. tgt's top is unique in 91% of
                    # positions and those are the only ones scored.
                    p = torch.softmax(lg, -1).cpu().numpy()
                    sc = np.where(v, SGN[sl][:, None] * (p[..., 0] - p[..., 2]), -np.inf)
                    tv = np.where(np.isfinite(TGT[sl]) & v, TGT[sl], -np.inf)
                    uniq = (tv == tv.max(1, keepdims=True)).sum(1) == 1
                    ok = v.any(1) & uniq
                    # random tie-break, so a model that scores every child alike lands on
                    # chance rather than on slot 0 -- which Stockfish's own move ordering
                    # makes best 54% of the time
                    jit = ero.random(sc.shape) * 1e-9
                    pick = np.where(v, sc + jit, -np.inf).argmax(1)
                    top1 += int((pick[ok] == tv.argmax(1)[ok]).sum())
                    base0 += int((tv.argmax(1)[ok] == 0).sum())
                    nrank += int(ok.sum())
                    # REGRET is the honest yardstick: how much win probability the chosen
                    # move gives up against the best one. Top-1 counts a near-tie as a
                    # total miss; a perfect WDL ranker scores only 0.519 top-1 yet gives
                    # up just 0.034 win probability (median 0.000). tools/diag_regret.py.
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
                slot0_acc = base0 / max(nrank, 1)
                mregret = reg_sum / max(nreg, 1)
            fl = ""
            if fp is not None and z0 is not None:
                ev = future_eval(fut, z0, mvc, {"1": t1, "2": t2})
                fl = (f" | fut1 lift {ev['ply1']['lift']:+.3f} (chg "
                      f"{ev['ply1']['bits_changed']:.3f}) fut2 lift "
                      f"{ev['ply2']['lift']:+.3f}")
            rk = f" rloss {lr:.4f}" if args.w_rank > 0 else ""
            sl = (f" grpchg {ls:.3f} moved {lc:.1f}/97" if args.w_slow > 0
                  else "")
            print(f"[qat] step {step:>5}/{args.steps} | wdl {lw:.3f}{rk}{sl} "
                  f"fut {lf:.3f} | BALANCED {bal:.3f} "
                  f"recall W/D/L {rec[0]:.2f}/{rec[1]:.2f}/{rec[2]:.2f} | "
                  f"REGRET {mregret:.4f} (perfect-WDL 0.034, random 0.266) | "
                  f"top1 {top1_acc:.3f} | "
                  f"concepts {live}{fl}", flush=True)
            mv_alive = int((model.readout.weight[:, model.d_pos:].abs().sum(0) > 0).sum())
            out = {"balanced": bal, "concepts": live, "wdl": lw, "top1": top1_acc,
                   "top1_slot0_baseline": slot0_acc, "move_concepts_alive": mv_alive,
                   "regret": mregret, "rank": lr, "group_change": ls, "groups_moved": lc}
            if mv_alive == 0:
                print("  WARNING: pruning removed EVERY move concept; per-child ranking "
                      "is now impossible and top-1 will fall to chance", flush=True)
            if args.arm == "qat":
                h = model.attn.questions_asked()
                out["q_max_share"] = float(h.max())
        return out

    if args.eval_only:
        sd = torch.load(args.eval_only, map_location="cpu", weights_only=False)
        missing = model.load_state_dict(sd["state_dict"], strict=False)
        print(f"[qat] eval-only: loaded {args.eval_only} "
              f"(step {sd.get('step','?')}, {len(sd['state_dict'])} tensors, "
              f"{len(missing.missing_keys)} missing)", flush=True)
        run_eval(int(sd.get("step", 0)), float("nan"), float("nan"), float("nan"))
        return

    if args.units:
        idx = torch.from_numpy(rng.choice(tr_idx, 8, replace=False))
        zq, hmv_u, _ = encode(idx)
        # The level-1 state is NOT uniformly binary, and the gate that claimed to check
        # this read `... or True` for three runs. Measured composition (see
        # tools/diag_readout_survival.py): board+squares are strictly +-1, the 32 piece
        # slots are TERNARY (+-1 present, 0 for an empty slot), and the last 128 dims are
        # a normalised edge histogram. Assert what is actually true, per family.
        BS, PC, EH = 96 + 64 * field.sq_width, 2400, D_STATE
        ub = set(torch.unique(zq[:, :BS]).tolist())
        assert ub <= {-1.0, 1.0}, f"board+square concepts are not binary: {sorted(ub)[:6]}"
        up = set(torch.unique(zq[:, BS:PC]).tolist())
        assert up <= {-1.0, 0.0, 1.0}, f"piece slots are not ternary: {sorted(up)[:6]}"
        eh = zq[:, PC:EH].sum(-1)
        assert torch.allclose(eh, torch.ones_like(eh), atol=1e-3), \
            f"edge block is not a distribution: rows sum to {float(eh.mean()):.3f}"
        print(f"  U0 PASS level-1 composition verified: {BS} binary, {PC-BS} ternary "
              f"(empty piece slots are 0), {EH-PC} histogram summing to 1.0")
        logit, z, aux = model(zq, hmv_u)
        assert logit.shape == (8, MV_CAP, 3), f"expected per-child logits, got {logit.shape}"
        print(f"  U1 PASS a WDL logit for every child: {tuple(logit.shape)}")

        if args.arm == "qat":
            l2 = z[:, D_STATE:]
            assert set(torch.unique(l2).tolist()) <= {-1.0, 1.0}, "level 2 is not binary"
            print(f"  U2 PASS level-2 output is binary, {l2.shape[-1]} bits")

        # U3: the readout is hand-factorised into a position term and a move term for
        # memory reasons. If the slice boundary were off by anything, the model would
        # still run and still train -- it would just be reading the wrong weights. So
        # check it reproduces the honest concatenated form exactly.
        with torch.no_grad():
            mvb = model.mv_bits(model.mv_proj(hmv_u))[0]
            cat = torch.cat([z[:, None, :].expand(-1, MV_CAP, -1), mvb], -1)
            ref = model.readout(cat)
        err = float((ref - logit).abs().max())
        assert err < 1e-4, f"factorised readout disagrees with the concatenated one: {err}"
        assert set(torch.unique(mvb).tolist()) <= {-1.0, 1.0}, "move concepts are not bits"
        print(f"  U3 PASS position+move factorisation is exact (max err {err:.2e}); "
              f"move concepts are {mvb.shape[-1]} bits")

        # U4: siblings must be SCORED APART. A dead move term still yields a running
        # model whose per-position scores are all identical -- top-1 would sit at chance
        # and nothing else would look wrong. This is that silent failure, made loud.
        sc = (torch.softmax(logit, -1)[..., 0] - torch.softmax(logit, -1)[..., 2])
        spread = float(sc.std(dim=1).mean())
        assert spread > 1e-4, \
            f"all children of a position score alike (spread {spread:.2e}) -- the move " \
            f"term is dead and ranking is impossible"
        print(f"  U4 PASS children of one position are scored apart "
              f"(mean within-position spread {spread:.4f})")

        # U6: a future loss of exactly 0.000 means "never fired", not "converged". A whole
        # 200-step run reported that while the head was dead, so it is now a gate: the
        # batch must exist, the built child must really differ from its parent, and the
        # target concepts must actually move.
        if args.w_future > 0:
            fb = future_batch(4)
            assert fb is not None, "future batch is empty -- the head would train on nothing"
            R0, R1, R2, cl, _cl2 = fb
            zp, _, _ = encode_rows(R0)
            zc, _, _ = encode_rows(R1)
            assert not torch.equal(R0["sq"], R1["sq"]), "built child equals its parent"
            chg = float(((zp > 0) != (zc > 0)).float().mean())
            assert chg > 0.0, "1-ply child has identical concepts to the parent"
            print(f"  U6 PASS future children are real: one ply flips {100*chg:.1f}% of "
                  f"concept bits")

        # U7: the whole claim for --w-slow is that SUM OF NORMS prefers change
        # CONCENTRATED in few groups, while SUM OF SQUARED norms is indifferent. If that
        # is false the penalty is plain L2 wearing a costume. Two deltas with IDENTICAL
        # total squared magnitude: one group moved fully, vs every group moved a little.
        G, W_ = 64, 24
        conc = torch.zeros(G, W_); conc[0] = 1.0                     # one group, full
        total = float((conc ** 2).sum())
        spread = torch.full((G, W_), (total / (G * W_)) ** 0.5)      # all groups, a little
        assert abs(float((spread ** 2).sum()) - total) < 1e-4, "test deltas differ in energy"
        sn_c = float(((conc ** 2).sum(-1).clamp(min=1e-8).sqrt()).sum())
        sn_s = float(((spread ** 2).sum(-1).clamp(min=1e-8).sqrt()).sum())
        sq_c = float((conc ** 2).sum()); sq_s = float((spread ** 2).sum())
        assert sn_c < sn_s, f"sum-of-norms does NOT prefer concentrated change: {sn_c} vs {sn_s}"
        assert abs(sq_c - sq_s) < 1e-4, "sum-of-squares should be indifferent, but is not"
        print(f"  U7 PASS group sparsity: same energy, sum-of-NORMS {sn_c:.2f} concentrated "
              f"vs {sn_s:.2f} spread ({sn_s/sn_c:.1f}x penalty for smearing); "
              f"sum-of-SQUARES {sq_c:.2f} vs {sq_s:.2f} -- indifferent, which is why "
              f"plain L2 cannot buy group sparsity")

        # U8: a captured piece is an all-zero block, and attending to a piece that is
        # not on the board is meaningless. The mask is claimed in tokens(); check it.
        if args.arm == "qat":
            _, m8 = model.tokens(zq)
            npiece = [t for t in model.spans if t[0] == "piece"]
            if npiece:
                off = sum(t[3] for t in model.spans[:[t[0] for t in model.spans].index("piece")])
                pm = m8[:, off:off + 32]                    # (B,32) alive per SAMPLE
                dead = int((~pm).sum())
                assert dead > 0, "no dead piece slots in this batch -- test is vacuous"
                a8 = model.attn.last["attn"]                 # (B,H,T,T) after the forward
                # per-sample: sample 0's mask does not describe sample 5's captures
                to_dead = float((a8[:, :, :, off:off + 32] *
                                 (~pm)[:, None, None, :].to(a8.dtype)).sum())
                total = float(a8.sum())
                assert to_dead < 1e-6, (f"attention answers with captured pieces: "
                                        f"{to_dead:.3f} of {total:.1f} total mass")
                print(f"  U8 PASS {dead} captured slots masked; attention mass on a dead "
                      f"piece = {to_dead:.2e} of {total:.0f}")

        # U9: with pov=white the ranking algebra must be EXACT: SGN * (W'-L') on the
        # flipped triple equals the mover's (W-L). If it does not, regret is ranking the
        # wrong player's preferences on every Black-to-move row.
        if args.pov == "white":
            raw = np.concatenate(store["_cw"]).astype(np.float32)     # mover-POV triple
            m9 = CV_ & np.isfinite(raw).all(-1)
            lhs = (SGN[:, None] * (CW_[..., 0] - CW_[..., 2]))[m9]
            rhs = (raw[..., 0] - raw[..., 2])[m9]
            err9 = float(np.abs(lhs - rhs).max())
            assert err9 < 1e-6, f"POV sign algebra broken: max err {err9}"
            assert int((~STM).sum()) > 0, "no Black-to-move rows -- U9 is vacuous"
            flipped = int((CY_[~STM] != CY_mover[~STM]).sum())
            assert flipped > 0, "no Black-to-move label changed under White-POV"
            print(f"  U9 PASS White-POV algebra exact (err {err9:.1e}); "
                  f"{flipped:,} Black-row child labels flipped; turn bias in readout")

        # U10: the 2-ply head must actually condition on BOTH moves.
        w2 = fut.heads["2"].net[0].in_features if hasattr(fut.heads["2"].net, "__getitem__") \
            else fut.heads["2"].net.in_features
        assert w2 == D_STATE + 2 * args.mv_bits, \
            f"2-ply head reads {w2}, expected {D_STATE + 2 * args.mv_bits}"
        fb = future_batch(6)
        assert fb is not None and len(fb) == 5, "future_batch must return the reply column"
        _, _, _, c1_, c2_ = fb
        assert c2_.max() < MV_CAP, "reply column exceeds the move-token cap"
        print(f"  U10 PASS 2-ply head reads {w2} = state + BOTH move codes; "
              f"reply columns in range (max {int(c2_.max())})")

        # U11: the White-POV move term must transform by the COLOR-SWAP PERMUTATION
        # (W<->L, D fixed), not negation. Two exact consequences, both move-independent:
        # the D column of (logits at stm=1) - (logits at stm=0), and the cross term
        # m1[...,W] - m0[...,L].
        if args.pov == "white":
            with torch.no_grad():
                m1 = model(zq, hmv_u, stm=torch.ones(len(zq), dtype=torch.long, device=dev))[0]
                m0 = model(zq, hmv_u, stm=torch.zeros(len(zq), dtype=torch.long, device=dev))[0]
                sD = float((m1[..., 1] - m0[..., 1]).std(dim=1).max())
                sX = float((m1[..., 0] - m0[..., 2]).std(dim=1).max())
            assert sD < 1e-5, f"D column not symmetric under side-to-move: {sD}"
            assert sX < 1e-5, f"W/L columns not swapped under side-to-move: {sX}"
            print(f"  U11 PASS move term transforms by the color-swap PERMUTATION "
                  f"(D-symmetry {sD:.1e}, W<->L swap {sX:.1e})")

        # U12: the future loss must be ISOLATED from the value move codes -- reaching
        # them is the exact capacity war that killed qat7b. Proven by autograd, not
        # assumed (follow_the_code_paths).
        if args.w_future > 0:
            fb12 = future_batch(4)
            if fb12 is not None:
                R0b, R1b, R2b, c1b, c2b = fb12
                z0b, h0b, _ = encode_rows(R0b)
                with torch.no_grad():
                    t1b, h1b, _ = encode_rows(R1b, enc=fut.ema)
                    t2b, _, _ = encode_rows(R2b, enc=fut.ema)
                arb = torch.arange(len(c1b), device=dev)
                mvbd, _, _ = model.dyn_mv_bits(model.dyn_mv_proj(h0b))
                pl = mvbd[arb, c1b.to(dev)]
                with torch.no_grad():
                    m2b = model.dyn_mv_bits(model.dyn_mv_proj(h1b))[0][arb, c2b.to(dev)]
                lf, _ = fut.loss(z0b, {"1": pl, "2": torch.cat([pl, m2b], -1)},
                                 {"1": t1b.detach(), "2": t2b.detach()})
                g_val = torch.autograd.grad(lf, model.mv_proj.weight,
                                            retain_graph=True, allow_unused=True)[0]
                g_dyn = torch.autograd.grad(lf, model.dyn_mv_proj.weight,
                                            allow_unused=True)[0]
                assert g_val is None or float(g_val.abs().sum()) == 0.0, \
                    "future loss leaks into the VALUE move codes -- the qat7b war again"
                assert g_dyn is not None and float(g_dyn.abs().sum()) > 0, \
                    "future loss does not train its own dyn move codes"
                print(f"  U12 PASS future loss isolated: value-code grad "
                      f"{'None' if g_val is None else 0.0}, dyn-code grad flows")

        # U13: objective/parameter separation, proven by autograd. CE must not touch the
        # value move codes; regret must be their ONLY trainer.
        tgt13 = torch.softmax(torch.randn_like(logit), -1)
        l_ce13 = -(tgt13 * F.log_softmax(aux["logit_ce"], -1)).sum(-1).mean()
        g13 = torch.autograd.grad(l_ce13, model.mv_proj.weight, retain_graph=True,
                                  allow_unused=True)[0]
        assert g13 is None or float(g13.abs().sum()) == 0.0, \
            "CE still trains the value move codes"
        l_rk13 = torch.softmax((logit[..., 0] - logit[..., 2]), -1).mean()
        g13b = torch.autograd.grad(l_rk13, model.mv_proj.weight, retain_graph=True,
                                   allow_unused=True)[0]
        assert g13b is not None and float(g13b.abs().sum()) > 0, \
            "regret no longer trains the move codes"
        print(f"  U13 PASS gradient routing: CE->position only "
              f"(mv grad {'None' if g13 is None else 0.0}), regret->move bits")

        logit.sum().backward()
        for nm, prm in (("readout", model.readout.weight),
                        ("move projection", model.mv_proj.weight)):
            assert prm.grad is not None and prm.grad.abs().sum() > 0, f"no gradient into {nm}"
        gm = model.readout.weight.grad[:, model.d_pos:].abs().sum()
        assert float(gm) > 0, "no gradient into the MOVE half of the readout"
        print(f"  U5 PASS gradient reaches the readout, its move half, and mv_proj")
        print("  ALL QAT GATES PASS")
        return

    cfg = TrainConfig(out=args.out, steps=args.steps, ckpt_every=args.ckpt_every,
                      eval_every=args.eval_every, experiment="catspace",
                      run_name=args.out.split("/")[-1], device=str(dev))
    standard_train(step_fn, model, cfg, args, opt=opt, rng=rng)


if __name__ == "__main__":
    main()
