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
      branch      (qgt2, Kaveh 2026-08-28) from every chain row k=0..5, roll ONE
                  further step through sampled multiPV moves and RANK the branch
                  reads against that row's per-move SF values (chain_values `wp`) --
                  regret + pairwise at landing depths 1..6. qgt1 proved value-only
                  anchoring lets K contract to rho~0.75 with chance-level rolled
                  bits; ranking ~30 moves at depth is what charges K for losing state
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
    ap.add_argument("--branch-k", type=int, default=4,
                    help="multiPV branch moves sampled per chain row per horizon "
                         "(always includes the row's best move)")
    ap.add_argument("--w-branch", type=float, default=1.0,
                    help="BRANCH RANKING at every rolled horizon (Kaveh 2026-08-28): "
                         "from each chain row k=0..5, roll ONE further step through "
                         "sampled multiPV moves and rank the branch reads against that "
                         "row's per-move SF values (regret + pairwise, qat9's form). A "
                         "contracted carrier can track one drifting scalar; it cannot "
                         "rank ~30 moves four plies deep -- this is what charges K for "
                         "losing state. 0 restores qgt1's value-only anchoring.")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--ckpt-every", type=int, default=2000)
    ap.add_argument("--max-shards", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--blunder-batch", type=int, default=12,
                    help="positions per step for the BLUNDER-PAIR stream (Kaveh: "
                         "'some moves are objectively bad... we should have bad moves "
                         "from the stockfish multi pv. Let's use them'). Each yields "
                         "its best and worst finite-wp child as constructed, labeled "
                         "positions + lag-2 teacher pairs through those moves.")
    ap.add_argument("--fanreg-k", type=int, default=8)
    ap.add_argument("--w-fanreg", type=float, default=0.0,
                    help="FAN-TO-REGRET (qgt8, Kaveh 'go on both'): expectation-"
                         "regret + pairwise over ROLLED reads of fanreg-k multiPV "
                         "children per corpus position -- the canonical pathway "
                         "gets the same ranking pressure the prior enjoys (which "
                         "was ~30x denser; the arena priced that gap at the random "
                         "floor). Rolled bits fully detached: trains readout + QA "
                         "levels only.")
    ap.add_argument("--child-sample", choices=("spoiler-uniform", "uniform",
                    "extremes"), default="spoiler-uniform",
                    help="which multiPV children the constructed stream samples per "
                         "position. spoiler-uniform (Kaveh): the SPOILER -- the best-"
                         "scoring move that drops the outcome class (qat10's 'most "
                         "tempting way to throw away the result') -- plus one uniform "
                         "draw; uniform: two uniform draws; extremes: best+worst.")
    ap.add_argument("--w-blunder", type=float, default=1.0,
                    help="weight of the blunder-child position CE (encoder+readout). "
                         "The teacher half rides in the dynamics BCE at weight 1.")
    ap.add_argument("--w-edge-warm", type=float, default=0.3,
                    help="graph arm: soft supervision of head-0 attention toward the "
                         "true attack adjacency (the WARM START, Kaveh) -- free "
                         "labels; heads/questions beyond it stay unsupervised.")
    ap.add_argument("--t-warmup", type=int, default=0,
                    help="canon: train the TRANSITION ALONE for N steps (closure "
                         "gradient into the encoder held at 0) so the in-cone "
                         "margin is established before anything competes with it "
                         "-- the 8/31 mediator finding: detection tracks whether "
                         "that margin ever opens, and seeds differ on it")
    ap.add_argument("--w-loc", type=float, default=0.0,
                    help="ENCODER-LOCALITY loss: bits(child) must equal "
                         "perm(bits(parent)) OUTSIDE the move's touch set -- "
                         "teaches the CODE that untouched squares encode "
                         "identically (gateFN stuck 0.78-0.81 = the encoder, not "
                         "the gate, is the delocalized party)")
    ap.add_argument("--direct-t", type=int, default=0,
                    help="canon: DIRECT next-state operator -- no identity+delta "
                         "(Kaveh 8/30: the hard gate owns out-of-cone copying; "
                         "the copy prior was qgt17's inertness attractor)")
    ap.add_argument("--hard-gate", type=int, default=0,
                    help="canon: HARD light-cone gating (Kaveh 8/30: 'do the hard "
                         "gating. that will make it sparse'). Squares outside the "
                         "move's exact touch-set (piece+attacker bitboard diff) "
                         "copy BIT-IDENTICALLY by construction; the teacher loss "
                         "sees gated squares only; true flips outside the gate "
                         "are the measured gateFN [science].")
    ap.add_argument("--w-vhead", type=float, default=0.0,
                    help="DEDICATED value head (roadmap step 3; frozen-bits probe "
                         "8/29: refit on same bits recovered +0.11 balanced and "
                         "0.37 draw recall the shared 5-objective readout loses). "
                         "Own linear on position bits, own pruner to the same "
                         "400 budget; when on, ALL position-value reads use it")
    ap.add_argument("--edge-mode", choices=("question", "pair"),
                    default="question",
                    help="transition edge typing: query-side questions (qgt6) or "
                         "pair-typed FSQ codes (spec 4, post-audit 8/30)")
    ap.add_argument("--canon-dir", default="data/derived/gko1_canon")
    ap.add_argument("--w-sib", type=float, default=0.5,
                    help="sibling ranking on PRE-quantization carrier (canon arm)")
    ap.add_argument("--ema-tau", type=float, default=0.995,
                    help="EMA target encoder for the JEPA closure (canon arm)")
    ap.add_argument("--closure-scale", type=float, default=0.25,
                    help="gradient scale of the teacher BCE into the ENCODER "
                         "(T always gets full gradient); the 8/30 collapse guard")
    ap.add_argument("--w-sparse", type=float, default=0.0,
                    help="canon: weak pull of T's output toward perm-copy -- the "
                         "'fewest concept changes per ply' prior; the teacher "
                         "overrides it wherever it is confident (8/30: closure "
                         "alone chose a 0.98-flip churn code, gauge zero)")
    ap.add_argument("--w-var", type=float, default=1.0,
                    help="anti-collapse bit-balance: mean(bit)^2 across the batch "
                         "(VICReg's variance guard, scale-free for +-1 bits)")
    ap.add_argument("--dynamics", choices=("koopman", "discrete", "graph", "canon"),
                    default="koopman",
                    help="'koopman': linear carrier roll c'=Kc+Bm, quantize-after-roll "
                         "(qgt1/qgt2 -- both contracted). 'discrete' (qgt3d, Kaveh): "
                         "the state that rolls IS the level-1 concepts, z'=sign(T(z,m)),"
                         " trained TEACHER-FORCED on the chain cache's true (bits, move "
                         "-> next bits) pairs; no carrier, no spectral radius to "
                         "contract. Same value/branch grounding on free-run rolls, "
                         "same instruments -- a head-to-head at the architecture level.")
    ap.add_argument("--dyn-lr-mult", type=float, default=10.0,
                    help="discrete arm: LR multiplier for bit_trans's own optimizer "
                         "group. qgt3d diagnosis: after 8000 steps at the shared LR the"
                         " zero-init output layer reached only max|w| 0.0275 vs the "
                         "0.46 flip barrier -- random-walk scale, gradient starvation.")
    ap.add_argument("--flip-weight", type=float, default=1.0,
                    help="teacher-forced BCE weight on bits that CHANGE between "
                         "consecutive true positions (~24-30%% of bits). 1.0 = plain "
                         "BCE. Held in reserve; LR was the binding constraint.")
    ap.add_argument("--w-mat", type=float, default=0.3,
                    help="static material-head loss (0 = off)")
    ap.add_argument("--w-mat-roll", type=float, default=1.0,
                    help="rolled-material consistency on teacher steps (0 = off)")
    ap.add_argument("--dyn-d", type=int, default=64,
                    help="transition width (GraphBitTransition d)")
    ap.add_argument("--dyn-layers", type=int, default=2,
                    help="transition depth = attention hops per move-step (the cone)")
    ap.add_argument("--w-id-init", type=float, default=0.5,
                    help="discrete arm: initial copy-prior barrier (see BitTransition)."
                         " 3.0 measured un-erodable; 0.5 keeps the copy init but lets "
                         "learned deltas flip bits within a few hundred steps.")
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

    DISC = args.dynamics in ("discrete", "graph", "canon")
    GRAPH = args.dynamics in ("graph", "canon")
    CANON = args.dynamics == "canon"
    assert not (args.hard_gate and args.w_sparse > 0), \
        "hard gate and the sparse-change prior are incompatible: the prior would " \
        "pull IN-CONE squares toward copy against the teacher (audit 8/30)"
    # THE CANON ARM (spec merge, Kaveh 8/29 "merge the doc like you said"): mover-
    # canonical features (side to move always White -- parity removed at the ROOT,
    # not worked around), ONE-PLY deterministic teacher pairs, JEPA closure (EMA
    # target encoder trains the ENCODER toward predictable bits -- the severed
    # direction, now with stop-grad + variance watch), sibling ranking on the
    # continuous pre-quantization carrier, relation-only quantization in T
    # (discrete edge TYPE, continuous message CONTENT), material grounding kept.
    # Between plies the frame changes by the exact MIRROR_PERM.
    dev = resolve_device("auto")
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    N_NODES = 64 + MV_CAP

    # ---- corpus: STOCKFISH ROWS ONLY (qat pipeline verbatim, minus spoiler) ---------
    KEYS = ("sq", "gl", "mv", "mut", "vac", "dsc", "att", "n_mv", "n_att",
            "kind", "band", "tgt", "wdl", "pslot")
    store = {k: [] for k in KEYS}
    kept = 0
    CANON_FLIP = []                       # per kept row: was the board mirrored?
    for f in sorted(glob.glob(os.path.join(args.shards, "shard_*.npz")))[:args.max_shards]:
        z = load_npz(f)
        if "wdl" not in z.files:
            continue
        if CANON:
            zc = load_npz(os.path.join(args.canon_dir, os.path.basename(f)))
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
        if CANON:
            # labels are stored in ORIGINAL move order; permute into the canonical
            # legal order via canon_of, drop moves lost past MV_CAP (255)
            cm = np.asarray(zc["canon_of"])[idx]                # (n, MV)
            n_, MV_ = cm.shape
            wd_c = np.full_like(wd[idx], np.nan)
            tg_c = np.full_like(tg[idx], np.nan)
            fin_c = np.zeros_like(fin_all[idx])
            ri, ji = np.nonzero(cm != 255)
            ci_ = cm[ri, ji].astype(np.int64)
            wd_c[ri, ci_] = wd[idx][ri, ji]
            tg_c[ri, ci_] = tg[idx][ri, ji]
            fin_c[ri, ci_] = fin_all[idx][ri, ji]
            store.setdefault("_cw", []).append(wd_c)
            store.setdefault("_cv", []).append(fin_c)
            store.setdefault("tgt", []).append(
                np.where(np.isnan(tg_c), -np.inf, tg_c))
            for key in KEYS:
                if key == "tgt":
                    continue                   # permuted above
                zsrc = zc if key in zc.files else z   # labels/kind: original
                store[key].append(np.ascontiguousarray(zsrc[key][idx]))
            CANON_FLIP.append(np.asarray(zc["flipped"])[idx].astype(bool))
        else:
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

    _PV_VAL = {"P": 1, "N": 3, "B": 3, "R": 5, "Q": 9}

    def _material(fen):
        """White-POV material balance from the fen's board field (exact, free)."""
        bd = str(fen).split()[0]
        return float(sum(v * (bd.count(k) - bd.count(k.lower()))
                         for k, v in _PV_VAL.items()))

    MAT = np.array([_material(f) for f in FENS], np.float32)
    # WHITE-POV: flip W<->L wherever Black is to move (inputs are absolute-color and
    # labels were the only mover-POV pressure -- verified 2026-08-27). Ranking stays
    # MOVER-relative via SGN (gate U9).
    _raw = CW_.copy()                                         # mover-POV, kept for U9
    CY_mover = CW_.argmax(-1)
    if CANON:
        # canonical frame: the mover IS White -- labels stay mover-POV, sign +1,
        # and material flips with the mirror (mirror() swaps colors)
        FLIP_C = np.concatenate(CANON_FLIP)
        STM = np.ones_like(STM, dtype=bool)
        MAT = MAT * np.where(FLIP_C, -1.0, 1.0).astype(np.float32)
    else:
        CW_[~STM] = CW_[~STM][..., ::-1]
    CY_ = CW_.argmax(-1)
    SGN = np.where(STM, 1.0, -1.0).astype(np.float32)
    CY_MOV = np.where(np.isfinite(_raw).all(-1), _raw.argmax(-1), 3)  # mover-POV cls
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
    if CANON:
        _chc = np.load(os.path.join(args.canon_dir, "chains.npz"), mmap_mode="r")
        CHA = {k: _chc[k] for k in CH_KEYS}   # canonical features + remapped played
        CH_CANON_OF = np.asarray(_chc["canon_of"])
        CH_FLIP = np.asarray(_chc["flipped"]).astype(bool)
    else:
        CHA = {k: _chg[k] for k in CH_KEYS}   # hold once; NpzFile re-reads per access
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
    CHW = _zv["wp"].astype(np.float32)                        # (rows, MV_CAP) mover-POV
    cf = np.load(args.chain_fens)                             #   per-move win-probs
    CH_FEN = cf["fen"]
    CH_STM = np.array([str(f).split()[1] == "w" for f in CH_FEN])
    CH_MAT = np.array([_material(f) for f in CH_FEN], np.float32)
    if CANON:
        # mover-canonical: labels stay mover-POV, every row is White-to-move,
        # material follows the mirror; per-move branch labels permute into the
        # canonical legal order (same construction as the corpus block)
        CH_STM = np.ones_like(CH_STM, dtype=bool)
        CH_MAT = CH_MAT * np.where(CH_FLIP, -1.0, 1.0).astype(np.float32)
        cwn = np.full_like(CHW, np.nan)
        ri, ji = np.nonzero(CH_CANON_OF != 255)
        cix = CH_CANON_OF[ri, ji].astype(np.int64)
        ok_ix = cix < CHW.shape[1]
        cwn[ri[ok_ix], cix[ok_ix]] = CHW[ri[ok_ix], ji[ok_ix]]
        CHW = cwn
        CH_WDL = _wdl_raw.copy()
    else:
        CH_WDL = _wdl_raw.copy()
        CH_WDL[~CH_STM] = CH_WDL[~CH_STM][:, ::-1]            # White-POV
    CH_SGN = np.where(CH_STM, 1.0, -1.0).astype(np.float32)
    CH_OK = np.isfinite(CH_WDL).all(-1) & (np.nan_to_num(_wdl_raw).sum(-1) > 0)
    print(f"[qgt] chains {len(CH_pos):,} ({len(ch_te):,} held out by GAME) | "
          f"{int(CH_OK.sum()):,}/{len(CH_OK):,} rows carry a WDL triple | "
          f"{int(np.isfinite(CHW).sum()):,} per-move branch labels", flush=True)

    # ---- model (from scratch -- no trunk, per the 8/27 trunk-removal design) --------
    model = QGTField(d=args.d, layers=args.layers, sq_bits=args.sq_bits,
                     rvq_levels=args.rvq_levels, mv_bits=args.mv_bits,
                     qa_levels=args.qa_levels, n_q=args.n_q, qa_heads=args.attn_heads,
                     top_k=args.top_k, out_bits=args.out_bits).to(dev)
    if GRAPH:
        from catspace.research.tools.training_infra.qgt_field import GraphBitTransition
        model.bit_trans = GraphBitTransition(
            sq_bits=args.sq_bits, mv_bits=args.mv_bits, n_q=args.n_q,
            heads=args.attn_heads, top_k=args.top_k, out_bits=args.out_bits,
            w_id_init=args.w_id_init, d=args.dyn_d,
            layers=args.dyn_layers, qa_continuous=CANON,
            edge_mode=args.edge_mode, direct_out=bool(args.direct_t)).to(dev)
    if CANON:
        # sibling ranking reads the PRE-quantization carrier (spec 2.4: the
        # bottleneck is deliberately fuzzy; move-level discrimination must not
        # depend on rounded bits)
        model.sib_scorer = torch.nn.Linear(64 * args.sq_bits, 1).to(dev)
        from catspace.research.tools.training_infra.qgt_canon import MIRROR_PERM
        PERM_T = torch.from_numpy(MIRROR_PERM).to(dev)
    elif DISC:
        from catspace.research.tools.training_infra.qgt_field import BitTransition
        # registered as a submodule BEFORE the optimizer, so it checkpoints + trains
        model.bit_trans = BitTransition(sq_bits=args.sq_bits, mv_bits=args.mv_bits,
                                        n_q=args.n_q, heads=args.attn_heads,
                                        top_k=args.top_k, out_bits=args.out_bits,
                                        w_id_init=args.w_id_init).to(dev)
    # MATERIAL HEAD (Kaveh 8/29: "a basic value function... so that I don't hang
    # a queen"): linear on the position concepts -> White-POV material/10. A hang
    # is a MATERIAL event; grounding rolled states in exact material is what makes
    # captures exist in the rolled bits for search to see.
    model.mat_head = torch.nn.Linear(model.d_pos, 1).to(dev)
    if args.w_vhead > 0:
        model.value_head = torch.nn.Linear(model.d_pos, 3).to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    if DISC and args.dyn_lr_mult != 1.0:
        _bt = {id(p) for p in model.bit_trans.parameters()}
        opt = torch.optim.Adam(
            [{"params": [p for p in model.parameters() if id(p) not in _bt],
              "lr": args.lr},
             {"params": list(model.bit_trans.parameters()),
              "lr": args.lr * args.dyn_lr_mult}])
        print(f"[qgt] bit_trans optimizer group at {args.dyn_lr_mult}x LR "
              f"({args.lr * args.dyn_lr_mult:g})", flush=True)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    if CANON:
        import copy as _copy
        ema = _copy.deepcopy(model)
        for p_ in ema.parameters():
            p_.requires_grad_(False)
        ema.eval()

        def ema_update():
            with torch.no_grad():
                for pe, po in zip(ema.parameters(), model.parameters()):
                    pe.mul_(args.ema_tau).add_(po.detach(), alpha=1 - args.ema_tau)

    settle = LossSettled()
    # DEVIATION, recorded: the pruner sees POSITION columns only. The move columns are
    # the entire per-child ranking capacity (32 bits); qat pruned them alongside and
    # had to warn when they all died. The handoff marks this an open question -- the
    # conservative default is taken.
    with torch.no_grad():                      # a plain storage view, no autograd tail
        _pos_cols = model.readout.weight[:, :model.d_pos]
    pruner = ConceptPruner(_pos_cols, args.l1_target,
                           args.steps, interval=args.prune_every)
    if args.w_vhead > 0:
        with torch.no_grad():
            _v_cols = model.value_head.weight
        vpruner = ConceptPruner(_v_cols, args.l1_target, args.steps,
                                interval=args.prune_every)

    def _pv_read(z, stm_idx):
        """THE position-value read. With the dedicated head, value stops sharing
        weights with ranking/branch/material/pruning objectives."""
        if args.w_vhead > 0:
            return model.value_head(z) + model.turn[stm_idx]
        W = model.readout.weight
        return F.linear(z, W[:, :model.d_pos], model.readout.bias) \
            + model.turn[stm_idx]
    print(f"[qgt] {n_par:,} params | carrier 64x{args.sq_bits} | levels: RVQ "
          f"{args.rvq_levels} + QA {args.qa_levels} | readout {model.d_pos}+{args.mv_bits}"
          f" -> prune position cols to {args.l1_target} (move cols exempt)", flush=True)

    CWt = torch.tensor(ccls.sum() / (3.0 * np.maximum(ccls, 1)),
                       dtype=torch.float32, device=dev)
    S3 = torch.tensor([1.0, 0.5, 0.0], device=dev)
    MVNP = D["mv"].numpy()                    # raw move rows for from/to lookups

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
        return sqf, mvf, edges, mask, played, R["mv"].numpy()

    def dyn_codes(h_mv):
        q, _, l = model.dyn_mv_lfq(model.dyn_mv_proj(h_mv))
        return q, (l.sum() if torch.is_tensor(l) else torch.zeros((), device=dev))

    def _ft(mv_raw, rows_j):
        """from/to squares of moves rows_j (n,) picked in raw mv arrays (n,MV,8):
        cols 1/2 per the graph format (verified in qgt_local_roll)."""
        n = len(rows_j)
        f_ = mv_raw[np.arange(n), rows_j, 1].astype(np.int64)
        t_ = mv_raw[np.arange(n), rows_j, 2].astype(np.int64)
        return torch.from_numpy(f_).to(dev), torch.from_numpy(t_).to(dev)

    def T_forward(z, code, frm, to, frm2=None, to2=None, need_attn=False):
        """-> (logits, attn0 or None, aux loss). frm2/to2 = the REPLY's squares
        (T2-conditional, Kaveh 8/29: the second input we were missing). Plain
        discrete ignores squares entirely (stays marginal, documented)."""
        if GRAPH:
            return model.bit_trans(z, code, frm, to, frm2=frm2, to2=to2,
                                   need_attn=need_attn)
        lg, l_qa = model.bit_trans(z, code)
        return lg, None, l_qa

    def T_roll(z0, packs):
        """packs: list of (code, frm, to[, frm2, to2]). Plain discrete rolls on
        codes alone."""
        if GRAPH:
            return model.bit_trans.roll_bits(z0, packs)
        return model.bit_trans.roll_bits(z0, [p[0] for p in packs])

    def _mat_read(lg):
        """Material read of a TEACHER-STEP state via STE bits (parens load-bearing:
        the sign() scar). Gradients reach T and mat_head, never the encoder."""
        zb = torch.sign(lg)
        zb = torch.where(zb == 0, torch.ones_like(zb), zb)
        zs_ = zb + (lg - lg.detach())
        return model.mat_head(model.levels_from_bits(zs_)[0]).squeeze(-1)

    MATSTAT = {}

    def _auc(logits, targets):
        """ROC-AUC of flat logits vs 0/1 targets (rank formula); 0.5 = chance."""
        lg = logits.detach().reshape(-1).float()
        tg = targets.detach().reshape(-1) > 0
        n_pos = int(tg.sum())
        n_neg = int((~tg).sum())
        if n_pos == 0 or n_neg == 0:
            return float("nan")
        r = lg.argsort().argsort().float() + 1.0          # ranks, 1-based
        return float((r[tg].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))

    def build_rows(boards):
        """python-chess boards -> the row dict feat_of consumes (shared by the
        random-split eval and the blunder stream)."""
        from catspace.research.tools.training_infra.gen_move_graphs import (
            parent_graph, glob_of_board)
        BK_ = ("sq", "gl", "mv", "mut", "vac", "dsc", "att", "n_mv", "n_att")
        acc = {k: [] for k in BK_}
        for b in boards:
            g_, _ = parent_graph(b)
            for k in ("sq", "mv", "mut", "att", "vac", "dsc"):
                acc[k].append(g_[k])
            acc["n_mv"].append(np.uint8(g_["n_mv"]))
            acc["n_att"].append(np.uint16(g_["n_att"]))
            acc["gl"].append(glob_of_board(b))
        return {k: torch.from_numpy(np.stack(acc[k])) for k in acc}

    ORC = [None, 0, 0]                     # ro connection, pv hits, pv misses

    def _pv_reply(fen, uci):
        """Punishing reply from the LIVE oracle store (PV retention + the
        training/backfill PIPELINE, Kaveh 8/29): SF-best at full labeling depth,
        cached, landing continuously as producers run. None while unbackfilled."""
        import sqlite3, json as _json
        if ORC[0] is None:
            ORC[0] = sqlite3.connect(
                "file:data/derived/gko1_oracle.sqlite?mode=ro", uri=True, timeout=2)
        try:
            r = ORC[0].execute(
                "SELECT payload FROM labels WHERE fen=? AND oracle='sf'",
                (fen,)).fetchone()
        except Exception:
            return None
        line = _json.loads(r[0]).get("pv", {}).get(uci) if r else None
        if line:
            ORC[1] += 1
            return line[0]
        ORC[2] += 1
        return None

    REP_CACHE = {}                         # (fen, uci) -> reply uci or None
    DEEPQ = [None, set()]                  # rw conn, pending child fens to enqueue

    def _fen_key(b1):
        """Canonical child key: counters zeroed so the same position reached from
        different parents shares one label row."""
        return " ".join(b1.fen().split()[:4]) + " 0 1"

    def _deep_reply(b1):
        """THE ASYNC DEEP PIPELINE (Kaveh 8/29: "depth 20 is probably what we
        need", approved async over inline): the reply to a move is the BEST MOVE
        of the child position. Look for a depth-20 child label (oracle 'sf20',
        best-line-only); on miss, queue the child for the background producers and
        let the caller fall back to live depth-8. Quality ramps to d20 across the
        run with zero step-time cost."""
        import sqlite3, json as _json
        key = _fen_key(b1)
        try:
            if ORC[0] is None:
                ORC[0] = sqlite3.connect(
                    "file:data/derived/gko1_oracle.sqlite?mode=ro", uri=True,
                    timeout=2)
            r = ORC[0].execute(
                "SELECT payload FROM labels WHERE fen=? AND oracle='sf20'",
                (key,)).fetchone()
        except Exception:
            return None
        if r:
            wp = _json.loads(r[0]).get("wp") or {}
            if wp:
                return max(wp, key=wp.get)
        DEEPQ[1].add(key)
        if len(DEEPQ[1]) >= 200:
            _deepq_flush()
        return None

    def _deepq_flush():
        """Buffered INSERT OR IGNORE of pending child fens; never blocks training
        (short busy timeout, drop on lock -- misses recur next epoch anyway)."""
        import sqlite3
        if not DEEPQ[1]:
            return
        try:
            if DEEPQ[0] is None:
                DEEPQ[0] = sqlite3.connect("data/derived/gko1_oracle.sqlite",
                                           timeout=0.5)
            with DEEPQ[0]:
                DEEPQ[0].executemany(
                    "INSERT OR IGNORE INTO queue(fen, oracle, score) "
                    "VALUES (?, 'sf20', 2)", [(f,) for f in DEEPQ[1]])  # proven-sampled > seeded
            DEEPQ[1].clear()
        except Exception:
            if len(DEEPQ[1]) > 5000:           # store unreachable; stop hoarding
                DEEPQ[1].clear()

    def _pv_reply_sq(fens, fb, tb):
        """Batch REPLY squares for branch moves. Resolution: oracle PV (top-3
        multiPV moves only -- store coverage measured 8/29: 657k/4.2M positions,
        3 pv moves each) -> inline SF depth-8 (~1.3ms, dict-cached) -> ok=False
        ONLY on real failure (illegal reconstruction/engine death). The 8/29
        instrument bug is the warning: masked misses must stay RARE or the branch
        pool silently shrinks to the store's top-3."""
        import chess as _c
        fbn = fb.cpu().numpy() if torch.is_tensor(fb) else np.asarray(fb)
        tbn = tb.cpu().numpy() if torch.is_tensor(tb) else np.asarray(tb)
        n = len(fbn)
        f2 = np.zeros(n, np.int64)
        t2 = np.zeros(n, np.int64)
        ok = np.zeros(n, bool)
        for i in range(n):
            fen = str(fens[i])
            uci = _c.square_name(int(fbn[i])) + _c.square_name(int(tbn[i]))
            key = (fen, uci)
            if key in REP_CACHE:
                ru = REP_CACHE[key]
            else:
                try:
                    b1 = _c.Board(fen)
                    b1.push(_c.Move.from_uci(uci))
                except Exception:
                    b1 = None                      # promo uci etc -- pv may still hit
                ru = _deep_reply(b1) if b1 is not None else None
                if ru is None:
                    ru = _pv_reply(fen, uci)       # d10 parent pv
                if ru is None and b1 is not None:
                    rm = _sf_reply(b1)             # live d8, last resort
                    ru = rm.uci() if rm is not None else None
                if len(REP_CACHE) < 500_000:
                    REP_CACHE[key] = ru
            if ru:
                f2[i] = _c.parse_square(ru[:2])
                t2[i] = _c.parse_square(ru[2:4])
                ok[i] = True
        return (torch.from_numpy(f2).to(dev), torch.from_numpy(t2).to(dev),
                torch.from_numpy(ok).to(dev))

    SF_ENG = [None]                        # lazy; restarted on failure

    def _sf_teardown():
        """The engine's non-daemon thread otherwise blocks Py_Finalize forever --
        the qgt9/qgt10/gates sleep-at-exit, stack-confirmed 8/29."""
        try:
            if SF_ENG[0] is not None:
                SF_ENG[0].quit()
        except Exception:
            pass
        _deepq_flush()
        for h in (ORC, DEEPQ):
            try:
                if h[0] is not None:
                    h[0].close()
            except Exception:
                pass

    # NOTE: atexit is NOT enough -- CPython joins non-daemon threads (the
    # engine's) BEFORE atexit handlers run, so teardown must happen in normal
    # control flow at every main() exit (stack-confirmed twice, 8/29)

    def _sf_reply(b1):
        """The PUNISHING reply (Kaveh: 'why not just play the best opponent reply,
        the stockfish reply?'): SF-best at depth 8 (~1.3ms). The uniform reply
        taught the transition that hangs go unpunished (hang probe, 8/29)."""
        import chess.engine as _ce
        for _try in range(2):
            try:
                if SF_ENG[0] is None:
                    SF_ENG[0] = _ce.SimpleEngine.popen_uci("stockfish")
                    SF_ENG[0].configure({"Threads": 1})
                return SF_ENG[0].play(b1, _ce.Limit(depth=8)).move
            except Exception:
                try:
                    SF_ENG[0].quit()
                except Exception:
                    pass
                SF_ENG[0] = None
        return None

    def blunder_batch(nb):
        """Kaveh 2026-08-28 (after the temporal-freeze collapse): construct the child
        after the BEST and the WORST finite-wp move of a training position. The child
        board is a first-class labeled position (triple = the parent's multiPV row,
        free) and a lag-2 teacher pair (root bits, move) -> child-after-reply bits.
        Near-identical boards, very different labels: the frozen-time optimum cannot
        fit them, and the transition gets direct 'this move drops the WDL' pairs."""
        import chess
        from catspace.research.tools.training_infra.qgt_canon import canon as _canon
        roots, c1_boards, c2_boards, js, gs, r2 = [], [], [], [], [], []
        pick = rng.choice(tr_idx, nb * 3, replace=False)
        for g in pick:
            try:
                b0 = chess.Board(str(FENS[g]))
                if CANON:
                    b0 = _canon(b0)[0]         # labels already in canonical order
            except Exception:
                continue
            legal = list(b0.legal_moves)[:int(D["n_mv"][g])]
            if not legal:
                continue
            wp = TGT[g][:len(legal)]
            fin = np.isfinite(wp) & CV_[g][:len(legal)]
            if int(fin.sum()) < 2:
                continue
            v = np.flatnonzero(fin)
            j_best = int(v[wp[v].argmax()])
            if args.child_sample == "extremes":
                j_a, j_b = j_best, int(v[wp[v].argmin()])
                if wp[j_a] - wp[j_b] < 0.05 or j_a == j_b:
                    continue
            elif args.child_sample == "uniform":
                j_a, j_b = (int(x) for x in rng.choice(v, 2, replace=False))
            else:                             # spoiler-uniform (Kaveh)
                worse = v[(CY_MOV[g][v] > CY_MOV[g][j_best]) & (CY_MOV[g][v] < 3)]
                j_a = (int(worse[wp[worse].argmax()]) if len(worse)
                       else int(rng.choice(v)))
                rest = v[v != j_a]
                j_b = int(rng.choice(rest))
            okpair = True
            pair = []
            for j in (j_a, j_b):
                b1 = b0.copy(); b1.push(legal[j])
                if CANON:                      # ONE ply; the reply is search's job
                    pair.append((_canon(b1)[0], _canon(b1)[0], j))
                    if args.hard_gate:
                        from catspace.research.tools.training_infra.qgt_canon \
                            import touch_set as _tsb
                        r2.append(("g", _tsb(b0, legal[j])))
                    continue
                rep = list(b1.legal_moves)[:MV_CAP]
                if not rep:
                    okpair = False
                    break
                rmv = None
                ru = _deep_reply(b1)               # d20 child label (async pipeline)
                if ru is None:
                    ru = _pv_reply(str(FENS[g]), legal[j].uci())   # d10 parent pv
                if ru is not None:
                    try:
                        rmv = chess.Move.from_uci(ru)
                    except Exception:
                        rmv = None
                if rmv is None:
                    rmv = _sf_reply(b1)            # depth-8 inline (1.3ms) fallback
                if rmv is None or rmv not in rep:
                    rmv = rep[int(rng.integers(len(rep)))]
                b2 = b1.copy(); b2.push(rmv)
                pair.append((b1, b2, j))
            if not okpair:
                continue
            for b1, b2, j in pair:
                roots.append(int(g)); c1_boards.append(b1)
                c2_boards.append(b2); js.append(j)
                if not (CANON and args.hard_gate):
                    r2.append((0, 0) if CANON else
                              (b2.move_stack[-1].from_square,
                               b2.move_stack[-1].to_square))
            if len(js) >= nb * 2:
                break
        if not js:
            return None
        gs = np.array(roots, np.int64)
        m2b = np.array([_material(b.fen()) for b in c2_boards], np.float32)
        if CANON and args.hard_gate:
            rep_out = np.stack([g_ for tag, g_ in r2])       # (n,64) bool gates
        else:
            rep_out = np.array(r2, np.int64)
        return (gs, np.array(js, np.int64), build_rows(c1_boards),
                build_rows(c2_boards), rep_out, m2b)

    def att_targets(rows):
        """(n,64,64) float attack adjacency from the chain graph arrays -- the
        WARM-START labels (free)."""
        a = CHA["att"][rows]                                  # (n, ATT_CAP, 2)
        na = CHA["n_att"][rows].astype(np.int64)
        n, cap = a.shape[0], a.shape[1]
        val = (np.arange(cap)[None, :] < na[:, None]) & (a[..., 0] != 255)
        tgt = np.zeros((n, 64, 64), np.float32)
        ri, ci_ = np.nonzero(val)
        tgt[ri, a[ri, ci_, 0], a[ri, ci_, 1]] = 1.0
        return torch.from_numpy(tgt).to(dev)

    def _ce_rows(pv, tg):
        """Plain CE of position-value logits vs a (n,3) triple, normalized."""
        tg = torch.nan_to_num(tg).clamp(min=0)
        tg = tg / tg.sum(-1, keepdim=True).clamp(min=1e-6)
        return -(tg * F.log_softmax(pv, -1)).sum(-1).mean()

    def chain_anchor(cb):
        """Roll REAL chains 6 plies. TWO grounds at every horizon (all-ply -- the 8/23
        scar: supervising {1,3,6} only left plies 2/5 sign-inverted):
          value    CE of the read position value vs each row's White-POV triple
          branch   from each row k=0..5, roll ONE further step through sampled multiPV
                   moves and RANK the branch reads against that row's per-move SF
                   values (regret + pairwise -- per-move KL is satisfied by the mean,
                   8/24). Landing depths 1..6. This is what charges K for losing
                   state: a contracted carrier cannot rank moves it can no longer see.
        """
        ci = CH_pos[ch_tr[rng.integers(0, len(ch_tr), cb)]]           # (cb,7)
        sq0, mv0, ed0, mk0, pl0, _ = feat_chain(ci[:, 0])
        c0, h_mv0, _ = model.carrier(sq0, mv0, ed0, mk0)
        dq0, l_dq = dyn_codes(h_mv0)                                  # (cb,MV,32) grad
        ar = torch.arange(cb, device=dev)
        with torch.no_grad():                  # deeper move codes: conditioning only
            rows_mid = ci[:, 1:6].ravel()
            sqm, mvm, edm, mkm, plm, _ = feat_chain(rows_mid)
            _, h_mvm, _ = model.carrier(sqm, mvm, edm, mkm)
            dqm_all = dyn_codes(h_mvm)[0]
            plc = dqm_all[torch.arange(len(rows_mid), device=dev), plm
                          ].reshape(cb, 5, -1)
            dqm = dqm_all.reshape(cb, 5, MV_CAP, -1)
        moves = [dq0[ar, pl0]] + [plc[:, k] for k in range(5)]
        cs = model.roller.roll(c0.reshape(cb, -1), moves)             # 6 carriers, grad
        W = model.readout.weight
        l_val = torch.zeros((), device=dev)
        n_lab = 0
        for k in range(1, 7):
            fin = torch.from_numpy(CH_OK[ci[:, k]]).to(dev)
            if not bool(fin.any()):
                continue
            z = model.observe(cs[k - 1].reshape(cb, 64, model.sq_bits))[0]
            pv = F.linear(z, W[:, :model.d_pos], model.readout.bias) + model.turn[
                torch.from_numpy(CH_STM[ci[:, k]].astype(np.int64)).to(dev)]
            l_val = l_val + _ce_rows(pv[fin],
                                     torch.from_numpy(CH_WDL[ci[:, k]]).to(dev)[fin])
            n_lab += int(fin.sum())
        l_val = l_val / 6.0

        bk = args.branch_k
        l_br = torch.zeros((), device=dev)
        n_br = 0
        for k in range(6):                       # branch FROM row k, landing at k+1
            wpk = CHW[ci[:, k]]                                       # (cb,MV) mover-POV
            cols = np.zeros((cb, bk), np.int64)
            ok_r = np.zeros(cb, bool)
            for r in range(cb):
                v = np.flatnonzero(np.isfinite(wpk[r]))
                if len(v) < 2:
                    continue
                b_ = v[wpk[r][v].argmax()]
                rest = v[v != b_]
                cols[r, 0] = b_                  # slot 0 is ALWAYS the row's best move
                cols[r, 1:] = rng.choice(rest, bk - 1, replace=len(rest) < bk - 1)
                ok_r[r] = True
            if not ok_r.any():
                continue
            okt = torch.from_numpy(ok_r).to(dev)
            nv = int(ok_r.sum())
            colst = torch.from_numpy(cols[ok_r]).to(dev)              # (nv,bk)
            src = dq0 if k == 0 else dqm[:, k - 1]                    # (cb,MV,32)
            codes = src[okt].gather(1, colst[..., None].expand(-1, -1, src.shape[-1])
                                    ).reshape(nv * bk, -1)
            base = (c0.reshape(cb, -1) if k == 0 else cs[k - 1])[okt]
            base = base.reshape(nv, 1, -1).expand(-1, bk, -1).reshape(nv * bk, -1)
            zb = model.observe(model.roller.step(base, codes)
                               .reshape(-1, 64, model.sq_bits))[0]
            pvb = F.linear(zb, W[:, :model.d_pos], model.readout.bias)
            stm_c = torch.from_numpy(
                (1 - CH_STM[ci[:, k]].astype(np.int64))[ok_r]).to(dev)
            pvb = pvb + model.turn[stm_c.repeat_interleave(bk)]
            sg = torch.from_numpy(CH_SGN[ci[:, k]][ok_r]).to(dev)[:, None]
            sc = sg * (pvb[:, 0] - pvb[:, 2]).reshape(nv, bk)
            wsel = torch.from_numpy(
                wpk[np.arange(cb)[ok_r][:, None], cols[ok_r]]).to(dev)
            reg = (wsel[:, :1] - wsel).clamp(min=0)  # slot 0 best => its regret is 0
            if args.units:
                assert float(reg[:, 0].abs().max()) == 0.0, \
                    "branch slot 0 is not the best move -- regret is mis-anchored"
            l_br = l_br + (torch.softmax(sc, -1) * reg).sum(-1).mean()
            j = torch.randint(1, bk, (nv,), device=dev)
            arv = torch.arange(nv, device=dev)
            l_br = l_br + (F.softplus(-(sc[:, 0] - sc[arv, j])) * reg[arv, j]).mean()
            n_br += nv
        l_br = l_br / 6.0
        return l_val, l_br, l_dq, n_lab, n_br

    def chain_anchor_discrete(cb):
        """qgt4 (Kaveh 2026-08-28: "every two plies, basically, every move ... predict
        what concepts would flip properly two plies out, when it's my turn again"):
        the transition's unit is THE MOVE -- lag 2, where side-to-move parity cancels
        by construction. T2-CONDITIONAL (Kaveh 8/29, after the qgt7-10 invariance
        table): T conditions on the mover's code AND the reply's squares (row k+1's
        played move) -- nothing marginalized, the target is deterministic mechanics,
        and the qgt7-10 failure (per-bit marginals + sign() MAP rounding away every
        sub-50% tactical flip) has no room to exist. At play time the reply is
        supplied by SEARCH (min/expectimax over enumerated legal replies), never
        predicted by T. Teacher-forced on 5 lag-2 pairs per chain; value CE + branch
        ranking on FREE-RUN rolls at horizons 2/4/6 (replies: played line for the
        value roll, oracle-PV for branches). The encoder is NOT trained by the
        dynamics (inputs and targets both detached)."""
        ci = CH_pos[ch_tr[rng.integers(0, len(ch_tr), cb)]]           # (cb,7)
        with torch.no_grad():
            rows_all = ci.ravel()
            sqa, mva, eda, mka, pla, mv_raw = feat_chain(rows_all)
            c_all, h_all, _ = model.carrier(sqa, mva, eda, mka)
            q1_all = model.rvq_bits(c_all)[0].reshape(cb, 7, 64, model.sq_bits)
            dq_all = dyn_codes(h_all)[0].reshape(cb, 7, MV_CAP, -1)
        pla_np = pla.cpu().numpy()
        frm_a, to_a = _ft(mv_raw, pla_np)                # (cb*7,) played from/to
        frm7, to7 = frm_a.reshape(cb, 7), to_a.reshape(cb, 7)
        # WORLD-MODEL phase (the Dreamer discipline, qgt5): the move-code encoder is
        # part of the world model, so the teacher BCE trains it -- codes recomputed
        # from DETACHED features (the board encoder stays pinned by external labels)
        dq_g, l_dq = dyn_codes(h_all.detach())
        pl_g = dq_g[torch.arange(cb * 7, device=dev),
                    pla].reshape(cb, 7, -1)
        zin = q1_all[:, :5].reshape(cb * 5, 64, -1)      # rows 0..4
        zt = q1_all[:, 2:].reshape(cb * 5, 64, -1)       # rows 2..6: lag-2 targets
        need_w = GRAPH and args.w_edge_warm > 0
        lg, attn0, l_qa = T_forward(zin, pl_g[:, :5].reshape(cb * 5, -1),
                                    frm7[:, :5].reshape(-1), to7[:, :5].reshape(-1),
                                    frm2=frm7[:, 1:6].reshape(-1),
                                    to2=to7[:, 1:6].reshape(-1),
                                    need_attn=need_w)
        w_tf = torch.where(zin != zt, args.flip_weight, 1.0) \
            if args.flip_weight != 1.0 else None
        l_tf = F.binary_cross_entropy_with_logits(lg, (zt + 1) / 2, weight=w_tf)
        if need_w:
            from catspace.research.tools.training_infra.qgt_field import (
                GraphBitTransition as _GBT)
            tgt_rows = ci[:, :5].ravel()
            l_tf = l_tf + args.w_edge_warm * _GBT.edge_warm_loss(
                attn0, att_targets(tgt_rows))
        if args.w_mat_roll > 0:
            mtc = torch.from_numpy(
                CH_MAT[ci[:, 2:].ravel()] / 10.0).to(dev)
            l_tf = l_tf + args.w_mat_roll * F.huber_loss(_mat_read(lg), mtc)
        with torch.no_grad():
            tf_acc = float(((lg > 0) == (zt > 0)).float().mean())
            tf_copy = float((zin == zt).float().mean())
        # BEHAVIOUR phase: free-run rolls with the WORLD MODEL FIXED (DreamerV3:
        # "the world model is fixed during behavior learning, so the actor and value
        # gradients do not affect its representations" -- qgt4 measured exactly that
        # corruption: tfacc 0.78 -> 0.51 under anchor gradients). The anchors train
        # the READOUT + QA levels on rolled states, never T.
        with torch.no_grad():
            zs = T_roll(q1_all[:, 0],
                        [(pl_g[:, k].detach(), frm7[:, k], to7[:, k],
                          frm7[:, k + 1], to7[:, k + 1])
                         for k in (0, 2, 4)])            # horizons 2, 4, 6
        HORIZ = (2, 4, 6)
        W = model.readout.weight
        l_val = torch.zeros((), device=dev)
        n_lab = 0
        for i, k in enumerate(HORIZ):
            fin = torch.from_numpy(CH_OK[ci[:, k]]).to(dev)
            if not bool(fin.any()):
                continue
            z = model.levels_from_bits(zs[i])[0]
            pv = F.linear(z, W[:, :model.d_pos], model.readout.bias) + model.turn[
                torch.from_numpy(CH_STM[ci[:, k]].astype(np.int64)).to(dev)]
            l_val = l_val + _ce_rows(pv[fin],
                                     torch.from_numpy(CH_WDL[ci[:, k]]).to(dev)[fin])
            n_lab += int(fin.sum())
        l_val = l_val / 3.0
        # branch ranking from rows 0/2/4, landing 2/4/6: one T2 step per branch.
        # After my-move + reply it is MY turn again: the landing turn bias is the
        # SOURCE row's stm, not its flip.
        bk = args.branch_k
        l_br = torch.zeros((), device=dev)
        n_br = 0
        for i, k in enumerate((0, 2, 4)):
            wpk = CHW[ci[:, k]]
            cols = np.zeros((cb, bk), np.int64)
            ok_r = np.zeros(cb, bool)
            for r in range(cb):
                v = np.flatnonzero(np.isfinite(wpk[r]))
                if len(v) < 2:
                    continue
                b_ = v[wpk[r][v].argmax()]
                rest = v[v != b_]
                cols[r, 0] = b_
                cols[r, 1:] = rng.choice(rest, bk - 1, replace=len(rest) < bk - 1)
                ok_r[r] = True
            if not ok_r.any():
                continue
            okt = torch.from_numpy(ok_r).to(dev)
            nv = int(ok_r.sum())
            colst = torch.from_numpy(cols[ok_r]).to(dev)
            codes = dq_all[:, k][okt].gather(
                1, colst[..., None].expand(-1, -1, dq_all.shape[-1])
                ).reshape(nv * bk, -1)
            base = (q1_all[:, 0] if k == 0 else zs[i - 1])[okt]
            base = base.reshape(nv, 1, 64, -1).expand(-1, bk, -1, -1
                                                      ).reshape(nv * bk, 64, -1)
            rr7 = (np.arange(cb) * 7 + k)[ok_r]              # chain row index of row k
            mv_k = mv_raw[np.repeat(rr7, bk)]
            fb, tb = _ft(mv_k, cols[ok_r].ravel())
            # T2-conditional branch pricing: the reply after a BRANCH move is not in
            # the chain -- it comes from the oracle PV store (SF-best, cached)
            f2b, t2b, okb = _pv_reply_sq(
                CH_FEN[np.repeat(ci[ok_r, k], bk)], fb, tb)
            okm = okb.reshape(nv, bk)
            with torch.no_grad():                # world model fixed for the anchors
                zbr = T_roll(base, [(codes, fb, tb, f2b, t2b)])[0]
            zb = model.levels_from_bits(zbr)[0]
            pvb = F.linear(zb, W[:, :model.d_pos], model.readout.bias)
            stm_c = torch.from_numpy(
                CH_STM[ci[:, k]].astype(np.int64)[ok_r]).to(dev)
            pvb = pvb + model.turn[stm_c.repeat_interleave(bk)]
            sg = torch.from_numpy(CH_SGN[ci[:, k]][ok_r]).to(dev)[:, None]
            sc = sg * (pvb[:, 0] - pvb[:, 2]).reshape(nv, bk)
            wsel = torch.from_numpy(
                wpk[np.arange(cb)[ok_r][:, None], cols[ok_r]]).to(dev)
            reg = (wsel[:, :1] - wsel).clamp(min=0)
            if args.units:
                assert float(reg[:, 0].abs().max()) == 0.0, \
                    "branch slot 0 is not the best move -- regret is mis-anchored"
            # PV-miss branches: out of the softmax, out of the regret; a row whose
            # BEST branch missed contributes nothing (reg zeroed row-wide)
            sc = sc + torch.where(okm, 0.0, -30.0)
            reg = reg * okm * okm[:, :1]
            l_br = l_br + (torch.softmax(sc, -1) * reg).sum(-1).mean()
            j = torch.randint(1, bk, (nv,), device=dev)
            arv = torch.arange(nv, device=dev)
            l_br = l_br + (F.softplus(-(sc[:, 0] - sc[arv, j])) * reg[arv, j]).mean()
            n_br += nv
        l_br = l_br / 3.0
        return l_tf + l_qa + l_dq, l_val, l_br, n_lab, n_br, tf_acc, tf_copy

    GATE_CACHE = {}

    def _gate_rows(rows):
        """(n,) global chain row ids -> (n,64) bool PARENT-frame touch sets of the
        played move; cached (row -> gate) since chains resample across steps."""
        import chess as _c
        from catspace.research.tools.training_infra.qgt_canon import (
            canon as _cn, touch_set as _ts)
        rr = np.asarray(rows).ravel()
        out = np.zeros((len(rr), 64), bool)
        for i, r in enumerate(rr):
            r = int(r)
            g = GATE_CACHE.get(r)
            if g is None:
                b = _c.Board(str(CH_FEN[r]))
                cb, _fl = _cn(b)
                mv = list(cb.legal_moves)[int(CHA["played"][r])]
                g = _ts(cb, mv)
                GATE_CACHE[r] = g
            out[i] = g
        return out

    def _gate_branch(rows, cols):
        """touch sets for BRANCH moves (row-k canonical board + sampled col)."""
        import chess as _c
        from catspace.research.tools.training_infra.qgt_canon import (
            canon as _cn, touch_set as _ts)
        rr = np.asarray(rows).ravel()
        out = np.zeros((len(rr), 64), bool)
        for i, (r, c_) in enumerate(zip(rr, np.asarray(cols).ravel())):
            b = _c.Board(str(CH_FEN[int(r)]))
            cb, _fl = _cn(b)
            legal = list(cb.legal_moves)
            if int(c_) < len(legal):
                out[i] = _ts(cb, legal[int(c_)])
            else:
                out[i] = True                      # malformed col: gate open
        return out

    def T1_roll(z0, packs):
        """Canon free-run: one PLY per step; the frame changes by MIRROR_PERM
        after every step (parens: the STE scar)."""
        out, z = [], z0
        for pk in packs:
            m, frm, to = pk[0], pk[1], pk[2]
            gate = pk[3] if len(pk) > 3 else None  # (B,64) bool, parent frame
            lg, _, _ = model.bit_trans(z, m, frm, to)
            zb = torch.sign(lg)
            zb = torch.where(zb == 0, torch.ones_like(zb), zb)
            zs = zb + (lg - lg.detach())
            if gate is not None:                   # HARD gate: outside T(a), copy
                zs = torch.where(gate[..., None], zs, z)
            z = zs[:, PERM_T]
            out.append(z)
        return out

    def chain_anchor_canon(cb):
        """THE CANON UNIT (spec merge 8/29): one PLY, mover-canonical frames.
        Teacher: rows k -> k+1 (all six pairs), target = EMA encoder bits of the
        child, stop-grad -- the JEPA closure: the ONLINE encoder gets gradients
        through the transition INPUT, so bits are pressured toward predictability;
        the value CE on all 7 rows is the anti-collapse counterweight. Copy
        baseline in the child frame is perm(z): a stationary piece sits at the
        mirrored token. Branch pricing is negamax: my score of a branch = MINUS
        the child's mover-POV read."""
        cb = min(cb, 8)                        # encoder grads on cb*7 boards
        ci = CH_pos[ch_tr[rng.integers(0, len(ch_tr), cb)]]           # (cb,7)
        rows_all = ci.ravel()
        sqa, mva, eda, mka, pla, mv_raw = feat_chain(rows_all)
        c_all, h_all, _ = model.carrier(sqa, mva, eda, mka)      # GRADS ON
        q1_flat, l_rvq = model.rvq_bits(c_all)
        q1_all = q1_flat.reshape(cb, 7, 64, model.sq_bits)
        with torch.no_grad():                                    # EMA targets
            qe = ema.rvq_bits(ema.carrier(sqa, mva, eda, mka)[0])[0]
            qe = qe.reshape(cb, 7, 64, model.sq_bits)
        dq_g, l_dq = dyn_codes(h_all.detach())
        pl_g = dq_g[torch.arange(cb * 7, device=dev), pla].reshape(cb, 7, -1)
        pla_np = pla.cpu().numpy()
        frm_a, to_a = _ft(mv_raw, pla_np)
        frm7, to7 = frm_a.reshape(cb, 7), to_a.reshape(cb, 7)
        # WARMUP (8/31): closure gradient into the encoder is OFF until the
        # transition has had t_warmup steps to open its in-cone margin alone
        a_ = 0.0 if STEP[0] <= args.t_warmup else args.closure_scale
        q1_cl = q1_all * a_ + q1_all.detach() * (1 - a_)   # encoder pull damped
        zin = q1_cl[:, :6].reshape(cb * 6, 64, -1)
        zt = qe[:, 1:].reshape(cb * 6, 64, -1)                   # sg by no_grad
        need_w = args.w_edge_warm > 0
        lg, attn0, l_qa = model.bit_trans(zin, pl_g[:, :6].reshape(cb * 6, -1),
                                          frm7[:, :6].reshape(-1),
                                          to7[:, :6].reshape(-1),
                                          need_attn=need_w)
        lgp = lg[:, PERM_T]                                      # child frame
        zin_p = zin[:, PERM_T]
        w_tf = torch.where(zin_p != zt, args.flip_weight, 1.0) \
            if args.flip_weight != 1.0 else None
        if args.hard_gate or args.w_loc > 0:
            gate_p = torch.from_numpy(
                _gate_rows(ci[:, :6])).to(dev)               # parent frame
            gp = gate_p[:, PERM_T][..., None].float()        # child frame
        if args.hard_gate:
            w_tf = gp if w_tf is None else w_tf * gp
            with torch.no_grad():
                tf_ = (zin_p != zt)
                MATSTAT["gfn"] = float((tf_ & ~gp.bool()).sum()) / \
                    max(float(tf_.sum()), 1.0)
                MATSTAT["gsz"] = float(gate_p.float().mean())
        l_tf = F.binary_cross_entropy_with_logits(lgp, (zt + 1) / 2, weight=w_tf)
        if need_w:
            from catspace.research.tools.training_infra.qgt_field import (
                GraphBitTransition as _GBT)
            l_tf = l_tf + args.w_edge_warm * _GBT.edge_warm_loss(
                attn0, att_targets(ci[:, :6].ravel()))
        if args.w_mat_roll > 0:
            mtc = torch.from_numpy(CH_MAT[ci[:, 1:].ravel()] / 10.0).to(dev)
            if args.hard_gate:
                zbm = torch.sign(lgp)
                zbm = torch.where(zbm == 0, torch.ones_like(zbm), zbm)
                zsm = zbm + (lgp - lgp.detach())
                zsm = torch.where(gate_p[:, PERM_T][..., None], zsm, zin_p)
                mr = model.mat_head(
                    model.levels_from_bits(zsm)[0]).squeeze(-1)
                l_tf = l_tf + args.w_mat_roll * F.huber_loss(mr, mtc)
            else:
                l_tf = l_tf + args.w_mat_roll * F.huber_loss(_mat_read(lgp), mtc)
        if args.w_sparse > 0:
            l_tf = l_tf + args.w_sparse * F.binary_cross_entropy_with_logits(
                lgp, (zin_p.detach() + 1) / 2)
        with torch.no_grad():
            if args.hard_gate:                 # IN-CONE only: the all-squares
                gm = gate_p[:, PERM_T]         # print dilutes the signal ~5x
                sel = gm[..., None].expand_as(zt)
                tf_acc = float(((lgp > 0) == (zt > 0))[sel].float().mean())
                tf_copy = float((zin_p == zt)[sel].float().mean())
            else:
                tf_acc = float(((lgp > 0) == (zt > 0)).float().mean())
                tf_copy = float((zin_p == zt).float().mean())
            MATSTAT["dead"] = float(
                (q1_flat.mean((0, 1)).abs() > 0.95).float().mean())
        # ENCODER-LOCALITY (8/31, the complement of the gate): the GATE constrains
        # T; this constrains the CODE. Outside the move's touch set the child's
        # bits must equal the parent's mirrored bits -- both sides ONLINE, so the
        # gradient reshapes the encoder toward locally-supported concepts. The
        # bit-balance term + value CE are the anti-collapse counterweights.
        l_loc = torch.zeros((), device=dev)
        if args.w_loc > 0:
            zpar = q1_all[:, :6].reshape(cb * 6, 64, -1)[:, PERM_T]
            zchi = q1_all[:, 1:].reshape(cb * 6, 64, -1)
            ung = ~gate_p[:, PERM_T]                          # (cb*6, 64)
            agree = 0.5 * (1.0 - zpar * zchi)                 # 0 same, 1 opposite
            l_loc = args.w_loc * agree[ung].mean()
            with torch.no_grad():
                MATSTAT["loc"] = float(agree[ung].mean())
        # anti-collapse: every bit should split the batch (dead bits are the JEPA
        # failure; measured 8/30: evenlag copy 0.167 -> 0.035 by step 500)
        l_var = (q1_flat.mean(0) ** 2).mean() * args.w_var
        # value CE on ALL 7 rows of ONLINE bits (mover-POV; encoder counterweight)
        z7 = model.levels_from_bits(q1_flat)[0]
        W = model.readout.weight
        fin = torch.from_numpy(CH_OK[rows_all]).to(dev)
        pv = F.linear(z7, W[:, :model.d_pos], model.readout.bias) + model.turn[1]
        l_val = _ce_rows(pv[fin], torch.from_numpy(CH_WDL[rows_all]).to(dev)[fin])
        if args.w_vhead > 0:
            # DETACHED bits: the head learns to READ, it never pushes the encoder
            # (8/30 blow-up: doubled value gradients + closure tipped the encoder
            # into 96%-dead collapse by step 1000). The frozen-bits probe is the
            # existence proof that reading alone recovers the value.
            pvh = model.value_head(z7.detach()) + model.turn[1]
            l_val = l_val + args.w_vhead * _ce_rows(
                pvh[fin], torch.from_numpy(CH_WDL[rows_all]).to(dev)[fin])
        n_lab = int(fin.sum())
        # 1-ply branch pricing (negamax read), rows 0/2/4; oracle replies unneeded
        bk = args.branch_k
        l_br = torch.zeros((), device=dev)
        n_br = 0
        dq_all = dq_g.detach().reshape(cb, 7, MV_CAP, -1)
        for k in (0, 2, 4):
            wpk = CHW[ci[:, k]]
            cols = np.zeros((cb, bk), np.int64)
            ok_r = np.zeros(cb, bool)
            for r in range(cb):
                v = np.flatnonzero(np.isfinite(wpk[r]))
                if len(v) < 2:
                    continue
                b_ = v[wpk[r][v].argmax()]
                rest = v[v != b_]
                cols[r, 0] = b_
                cols[r, 1:] = rng.choice(rest, bk - 1, replace=len(rest) < bk - 1)
                ok_r[r] = True
            if not ok_r.any():
                continue
            okt = torch.from_numpy(ok_r).to(dev)
            nv = int(ok_r.sum())
            colst = torch.from_numpy(cols[ok_r]).to(dev)
            codes = dq_all[:, k][okt].gather(
                1, colst[..., None].expand(-1, -1, dq_all.shape[-1])
                ).reshape(nv * bk, -1)
            base = q1_all.detach()[:, k][okt]
            base = base.reshape(nv, 1, 64, -1).expand(-1, bk, -1, -1
                                                      ).reshape(nv * bk, 64, -1)
            rr7 = (np.arange(cb) * 7 + k)[ok_r]
            fb, tb = _ft(mv_raw[np.repeat(rr7, bk)], cols[ok_r].ravel())
            gbr = (torch.from_numpy(_gate_branch(
                np.repeat(ci[ok_r, k], bk), cols[ok_r].ravel())).to(dev)
                if args.hard_gate else None)
            with torch.no_grad():
                zbr = T1_roll(base, [(codes, fb, tb, gbr) if args.hard_gate
                                     else (codes, fb, tb)])[0]
            zb = model.levels_from_bits(zbr)[0]
            pvb = F.linear(zb, W[:, :model.d_pos], model.readout.bias)                 + model.turn[1]
            sc = -(pvb[:, 0] - pvb[:, 2]).reshape(nv, bk)        # negamax
            wsel = torch.from_numpy(
                wpk[np.arange(cb)[ok_r][:, None], cols[ok_r]]
                .astype(np.float32)).to(dev)
            reg = (wsel[:, :1] - wsel).clamp(min=0)
            if args.units:
                assert float(reg[:, 0].abs().max()) == 0.0,                     "canon branch slot 0 is not the best move"
            l_br = l_br + (torch.softmax(sc, -1) * reg).sum(-1).mean()
            j = torch.randint(1, bk, (nv,), device=dev)
            arv = torch.arange(nv, device=dev)
            l_br = l_br + (F.softplus(-(sc[:, 0] - sc[arv, j])) * reg[arv, j]).mean()
            n_br += nv
        l_br = l_br / 3.0
        return (l_tf + l_qa + l_dq + 0.1 * l_rvq + l_var + l_loc, l_val, l_br,
                n_lab, n_br, tf_acc, tf_copy)

    def fan_anchor(c_par, h_mv_par, ii, stm_b):
        """Roll the parent 1 ply through fan_k children each; CE vs each child's triple.
        Anti-collapse: different children carry different values."""
        fk = args.fan_k
        if DISC:       # teacher owns the move encoder (qgt5)
            with torch.no_grad():
                dq = dyn_codes(h_mv_par)[0]
            l_dq = torch.zeros((), device=dev)
        else:
            dq, l_dq = dyn_codes(h_mv_par)                            # (B,MV_CAP,32)
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
        mk_ = dq[kt].gather(1, colst[..., None].expand(-1, -1, dq.shape[-1])
                            ).reshape(nk * fk, -1)
        if DISC:
            mv_f = np.repeat(MVNP[ii][keep], fk, axis=0)     # (nk*fk, MV, 8)
            ff, tf_ = _ft(mv_f, cols[keep].ravel())
            # ENCODER GRADIENTS RESTORED (the qgt5/qgt6 collapse A/B, 8/28: both
            # runs that severed the fan's encoder path collapsed mid-run; both that
            # kept it did not -- and Dreamer's stop-grad protects the DYNAMICS, its
            # value heads DO train the representation). The transition stays fixed
            # to anchors by freezing ITS params for this roll; input gradients pass
            # through the STE to the encoder.
            zb0 = model.rvq_bits(c_par[kt])[0]
            zb0 = zb0.reshape(nk, 1, 64, -1).expand(-1, fk, -1, -1
                                                    ).reshape(nk * fk, 64, -1)
            for p_ in model.bit_trans.parameters():
                p_.requires_grad_(False)
            zb_r = T_roll(zb0, [(mk_, ff, tf_)])[0]
            for p_ in model.bit_trans.parameters():
                p_.requires_grad_(True)
            z1 = model.levels_from_bits(zb_r)[0]
        else:
            ck = c_par[kt].reshape(nk, 1, -1).expand(-1, fk, -1).reshape(nk * fk, -1)
            c1 = model.roller.step(ck, mk_)
            z1, _ = model.observe(c1.reshape(-1, 64, model.sq_bits))
        pv = F.linear(z1, model.readout.weight[:, :model.d_pos], model.readout.bias)
        stm_c = (1 - stm_b[kt]).repeat_interleave(fk)         # side flips after a ply
        pv = pv + model.turn[stm_c]
        tg = torch.from_numpy(CW_[ii][keep][np.arange(nk)[:, None], cols[keep]]
                              ).to(dev).reshape(nk * fk, 3)
        return _ce_rows(pv, tg), l_dq

    STEP = [0]

    def step_fn(_m, step):
        STEP[0] = step
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
        if args.w_mat > 0:
            mt = torch.from_numpy(MAT[ii] / 10.0).to(dev)
            mp = model.mat_head(z).squeeze(-1)     # z is already (B, d_pos) bits
            l_mat_s = F.huber_loss(mp, mt)
            loss = loss + args.w_mat * l_mat_s
            MATSTAT["s"] = float((mp - mt).abs().mean()) * 10.0

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

        # dynamics supervision: teacher-forced BCE (discrete arm only) + rolled-value
        # anchors (chain value + branch ranking + fan) in BOTH arms
        tf_acc = tf_copy = float("nan")
        if DISC:
            l_dyn, l_ch, l_br, n_lab, n_br, tf_acc, tf_copy = \
                (chain_anchor_canon if CANON else
                 chain_anchor_discrete)(args.chain_batch)
            l_dq_c = torch.zeros((), device=dev)
        else:
            l_ch, l_br, l_dq_c, n_lab, n_br = chain_anchor(args.chain_batch)
            l_dyn = torch.zeros((), device=dev)
        # the rolled corpus fan is RETIRED for the discrete/graph arms (Kaveh:
        # "why not do the fan at the encoder level?" -- the constructed-children CE
        # below IS that fan, on true boards, without the near-identity averaging
        # pressure that fed the freeze). Koopman keeps the original.
        if DISC:
            l_fan = torch.zeros((), device=dev)
            l_dq_f = torch.zeros((), device=dev)
        else:
            l_fan, l_dq_f = fan_anchor(aux["carrier"], aux["h_mv"], ii, stm_b)
        # MULTIPV CONSTRUCTED-CHILD stream (Kaveh): children as labeled
        # positions (CE -> encoder+readout; breaks the frozen-time optimum) + lag-2
        # teacher pairs through those moves (-> transition + dyn move encoder)
        l_blun = torch.zeros((), device=dev)
        if DISC and args.w_blunder > 0 and args.blunder_batch > 0:
            bb = blunder_batch(args.blunder_batch)
            if bb is not None:
                gs_b, js_b, R1b, R2b, rep2, mat2b = bb
                nbp = len(gs_b)
                sq1, mv1, ed1, mk1 = feat_of(R1b)
                c1b, _, _ = model.carrier(sq1, mv1, ed1, mk1)      # encoder GRAD
                z1b = model.levels_from_bits(model.rvq_bits(c1b)[0])[0]
                if CANON:                  # child read is child-mover POV:
                    stm_c1 = torch.ones(nbp, dtype=torch.long, device=dev)
                    tg1 = torch.from_numpy(
                        np.ascontiguousarray(CW_[gs_b, js_b][..., ::-1])).to(dev)
                else:
                    stm_c1 = torch.from_numpy(
                        (1 - STM[gs_b].astype(np.int64))).to(dev)
                    tg1 = torch.from_numpy(CW_[gs_b, js_b]).to(dev)
                pv1 = F.linear(z1b, model.readout.weight[:, :model.d_pos],
                               model.readout.bias) + model.turn[stm_c1]
                if args.w_vhead > 0:
                    pv1h = model.value_head(z1b.detach()) + model.turn[stm_c1]
                l_blun = _ce_rows(pv1, tg1)
                if args.w_vhead > 0:
                    l_blun = l_blun + args.w_vhead * _ce_rows(pv1h, tg1)
                loss = loss + args.w_blunder * l_blun
                # teacher half. Canon: (root bits, move) -> CHILD bits, one
                # ply, EMA target, perm frame change; sibling ranking on the
                # children's PRE-quantization carrier. T2: as before.
                with torch.no_grad():
                    sq0b, mv0b, ed0b, mk0b = batch_of(torch.from_numpy(gs_b))
                    c0b, h0b, _ = model.carrier(sq0b, mv0b, ed0b, mk0b)
                    q0b = model.rvq_bits(c0b)[0]
                dqb = dyn_codes(h0b.detach())[0]
                code_b = dqb[torch.arange(nbp, device=dev),
                             torch.from_numpy(js_b).to(dev)]
                fb_, tb_ = _ft(MVNP[gs_b], js_b)
                if CANON:
                    with torch.no_grad():
                        q1c = ema.rvq_bits(ema.carrier(sq1, mv1, ed1, mk1)[0])[0]
                    lg_b, _, _ = model.bit_trans(q0b, code_b, fb_, tb_)
                    lg_b = lg_b[:, PERM_T]
                    q0p = q0b[:, PERM_T]
                    w_b = torch.where(q0p != q1c, args.flip_weight, 1.0) \
                        if args.flip_weight != 1.0 else None
                    if args.hard_gate:
                        gb = torch.from_numpy(rep2).to(dev)[:, PERM_T][..., None]
                        w_b = gb.float() if w_b is None else w_b * gb.float()
                    l_dyn = l_dyn + F.binary_cross_entropy_with_logits(
                        lg_b, (q1c + 1) / 2, weight=w_b)
                    if args.w_sib > 0:
                        fsc = model.sib_scorer(
                            c1b.reshape(nbp, -1)).squeeze(-1)
                        wp_p = torch.from_numpy(
                            TGT[gs_b, js_b].astype(np.float32)).to(dev)
                        fa, fb2 = fsc[0::2], fsc[1::2]
                        wa, wb2 = wp_p[0::2], wp_p[1::2]
                        d_ = wa - wb2
                        l_sib = (F.softplus(-torch.sign(d_) * (fa - fb2))
                                 * d_.abs()).mean()
                        loss = loss + args.w_sib * l_sib
                else:
                    with torch.no_grad():
                        sq2, mv2b, ed2b, mk2b = feat_of(R2b)
                        c2b, _, _ = model.carrier(sq2, mv2b, ed2b, mk2b)
                        q2b = model.rvq_bits(c2b)[0]
                    lg_b, _, _ = T_forward(q0b, code_b, fb_, tb_,
                                           frm2=torch.from_numpy(
                                               rep2[:, 0]).to(dev),
                                           to2=torch.from_numpy(
                                               rep2[:, 1]).to(dev))
                    w_b = torch.where(q0b != q2b, args.flip_weight, 1.0) \
                        if args.flip_weight != 1.0 else None
                    l_dyn = l_dyn + F.binary_cross_entropy_with_logits(
                        lg_b, (q2b + 1) / 2, weight=w_b)
                if args.w_mat_roll > 0:
                    mtb = torch.from_numpy(mat2b / 10.0).to(dev)
                    mrb = _mat_read(lg_b)
                    l_dyn = l_dyn + args.w_mat_roll * F.huber_loss(mrb, mtb)
                    MATSTAT["r"] = float((mrb - mtb).abs().mean()) * 10.0
        # FAN-TO-REGRET (qgt8): rank ROLLED reads of K children with the prior's
        # own loss form; world model and encoder untouched (all rolling no_grad)
        l_freg = torch.zeros((), device=dev)
        if DISC and args.w_fanreg > 0:
            fk2 = args.fanreg_k
            wp2 = np.where(CV_[ii] & np.isfinite(TGT[ii]), TGT[ii], -np.inf)
            nb2 = len(ii)
            cols2 = np.zeros((nb2, fk2), np.int64)
            okr2 = np.zeros(nb2, bool)
            for r in range(nb2):
                v = np.flatnonzero(np.isfinite(wp2[r]))
                if len(v) < 2:
                    continue
                b_ = v[wp2[r][v].argmax()]
                rest = v[v != b_]
                cols2[r, 0] = b_
                cols2[r, 1:] = rng.choice(rest, fk2 - 1, replace=len(rest) < fk2 - 1)
                okr2[r] = True
            if okr2.any():
                okt2 = torch.from_numpy(okr2).to(dev)
                nv2 = int(okr2.sum())
                colst2 = torch.from_numpy(cols2[okr2]).to(dev)
                with torch.no_grad():
                    dqf = dyn_codes(aux["h_mv"])[0]
                    codes2 = dqf[okt2].gather(
                        1, colst2[..., None].expand(-1, -1, dqf.shape[-1])
                        ).reshape(nv2 * fk2, -1)
                    mvf2 = np.repeat(MVNP[ii][okr2], fk2, axis=0)
                    ff2, tf2 = _ft(mvf2, cols2[okr2].ravel())
                    zb0f = model.rvq_bits(aux["carrier"][okt2].detach())[0]
                    zb0f = zb0f.reshape(nv2, 1, 64, -1).expand(-1, fk2, -1, -1
                                                               ).reshape(nv2 * fk2, 64, -1)
                    zrf = T_roll(zb0f, [(codes2, ff2, tf2)])[0]
                zf2 = model.levels_from_bits(zrf)[0]
                pvf = F.linear(zf2, model.readout.weight[:, :model.d_pos],
                               model.readout.bias)
                stm_f = torch.from_numpy(STM[ii].astype(np.int64)[okr2]).to(dev)
                pvf = pvf + model.turn[stm_f.repeat_interleave(fk2)]
                sgf = torch.from_numpy(SGN[ii][okr2]).to(dev)[:, None]
                scf = sgf * (pvf[:, 0] - pvf[:, 2]).reshape(nv2, fk2)
                wself = torch.from_numpy(
                    wp2[okr2][np.arange(nv2)[:, None], cols2[okr2]]
                    .astype(np.float32)).to(dev)
                regf = (wself[:, :1] - wself).clamp(min=0)
                if args.units:
                    assert float(regf[:, 0].abs().max()) == 0.0, \
                        "fanreg slot 0 is not the best move"
                l_freg = (torch.softmax(scf, -1) * regf).sum(-1).mean()
                jf = torch.randint(1, fk2, (nv2,), device=dev)
                arf = torch.arange(nv2, device=dev)
                l_freg = l_freg + (F.softplus(-(scf[:, 0] - scf[arf, jf]))
                                   * regf[arf, jf]).mean()
                loss = loss + args.w_fanreg * l_freg
        if DISC:
            l_fan = l_freg                    # heartbeat: 'roll X+Y' shows fanreg as Y
        l_roll = l_ch + (l_fan if not DISC else torch.zeros((), device=dev))
        assert torch.isfinite(l_roll) and torch.isfinite(l_br) \
            and torch.isfinite(l_dyn) and torch.isfinite(l_freg), \
            "dynamics loss is not finite"
        loss = loss + args.w_roll * l_roll + args.w_branch * l_br + l_dyn \
            + l_dq_c + l_dq_f
        if args.dynamics == "koopman":
            loss = loss + model.roller.spectral_penalty()

        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if CANON:
            ema_update()
        if args.dynamics == "koopman":
            model.roller.clamp_()
        is_settled = settle.update(float(loss.detach()))
        pruner.step(step, settled=is_settled)
        if args.w_vhead > 0:
            vpruner.step(step, settled=is_settled)

        out = {}
        if step % args.eval_every == 0 or step == 1 or step == args.steps:
            out = run_eval(step, float(l_wdl), float(l_rank), float(l_ch),
                           float(l_fan), float(l_br), tf_acc, tf_copy,
                           float(l_dyn))
        return out

    # ---- the dual-split instrument --------------------------------------------------
    @torch.no_grad()
    def roll_real_canon(n_ch=96):
        """Canon instrument: 1-ply free-run over held-out chains. vroll = corr of
        the mover-POV read vs each row's mover-POV label at plies 2/4/6; evenlag =
        rolled-vs-true bit deltas with the PERM^k copy baseline; flip per ply."""
        model.eval()
        VKS = (2, 4, 6)
        pred = {k: [] for k in VKS}
        true = {k: [] for k in VKS}
        dl = {k: [0.0, 0.0] for k in VKS}
        cp = {k: [0.0, 0.0] for k in VKS}
        flips = [0.0, 0.0]
        gdiag = {}
        W = model.readout.weight
        for a in range(0, min(n_ch, len(ch_te)), 32):
            ci = CH_pos[ch_te[a:a + 32]]
            cb = len(ci)
            rows_all = ci.ravel()
            sqa, mva, eda, mka, pla, mv_raw = feat_chain(rows_all)
            c_all, h_all, _ = model.carrier(sqa, mva, eda, mka)
            q1 = model.rvq_bits(c_all)[0].reshape(cb, 7, 64, model.sq_bits)
            dq = dyn_codes(h_all)[0]
            pl_c = dq[torch.arange(cb * 7, device=dev), pla].reshape(cb, 7, -1)
            fa, ta = _ft(mv_raw, pla.cpu().numpy())
            f7, t7 = fa.reshape(cb, 7), ta.reshape(cb, 7)
            if args.hard_gate:
                zs = T1_roll(q1[:, 0],
                             [(pl_c[:, k], f7[:, k], t7[:, k],
                               torch.from_numpy(_gate_rows(ci[:, k])).to(dev))
                              for k in range(6)])
            else:
                zs = T1_roll(q1[:, 0],
                             [(pl_c[:, k], f7[:, k], t7[:, k])
                              for k in range(6)])
            prev = q1[:, 0]
            for zb_ in zs:
                flips[0] += float((zb_ != prev[:, PERM_T]).float().mean()) * cb
                flips[1] += cb
                prev = zb_
            if a == 0:
                _, at0, _ = model.bit_trans(q1[:, 0], pl_c[:, 0], f7[:, 0],
                                            t7[:, 0], need_attn=True)
                gdiag["aucT"] = _auc(at0[:, 0], att_targets(ci[:, 0]))
                tch = model.bit_trans.touched(q1[:, 0], pl_c[:, 0], f7[:, 0],
                                              t7[:, 0])
                flp = (q1[:, 0][:, PERM_T] != q1[:, 1]).any(-1)
                n_f = int(flp.sum())
                gdiag["cov"] = (float((flp & tch[:, PERM_T]).sum()) / n_f
                                if n_f else float("nan"))
                gdiag["aucR"] = float("nan")
            zcopy = q1[:, 0]
            for k in VKS:
                zk = zs[k - 1]
                fin = CH_OK[ci[:, k]]
                zl = model.levels_from_bits(zk)[0]
                pv = _pv_read(zl, 1)
                if fin.any():
                    prob = (torch.softmax(pv, -1) * S3).sum(-1).cpu().numpy()[fin]
                    tw = CH_WDL[ci[:, k]][fin]
                    tw = tw / tw.sum(-1, keepdims=True).clip(min=1e-6)
                    pred[k].append(prob)
                    true[k].append(tw @ np.array([1.0, 0.5, 0.0]))
                dl[k][0] += float((zk != q1[:, k]).float().mean()) * cb
                dl[k][1] += cb
                cp[k][0] += float(
                    (zcopy != q1[:, k]).float().mean()) * cb   # PERM^even = id
                cp[k][1] += cb
        corr = {}
        for k in VKS:
            if pred[k]:
                pz = np.concatenate(pred[k]); tz = np.concatenate(true[k])
                corr[k] = (float(np.corrcoef(pz, tz)[0, 1])
                           if pz.std() > 1e-9 and tz.std() > 1e-9 else 0.0)
            else:
                corr[k] = float("nan")
        delta = {k: dl[k][0] / max(dl[k][1], 1) for k in dl}
        copy = {k: cp[k][0] / max(cp[k][1], 1) for k in cp}
        breg = {k: float("nan") for k in (2, 4, 6)}
        breg["rnd"] = float("nan")
        breg["flip"] = flips[0] / max(flips[1], 1)
        breg.update(gdiag)
        return corr, VKS, delta, copy, breg

    @torch.no_grad()
    def roll_real(n_ch=192, lag_sub=48, br_k=8):
        if CANON:
            return roll_real_canon()
        """Held-out REAL chains: rolled-value corr at 1/3/6 + even-lag bit deltas +
        BRANCH REGRET at landing plies 1/3/6 (how much win-prob the model's pick among
        br_k sampled multiPV moves gives up, read from the ROLLED branch state; brnd =
        the random-pick baseline over the same samples)."""
        model.eval()
        ero = np.random.default_rng(2)
        # the discrete arm's unit is THE MOVE (2 plies): horizons and branch
        # landings sit on even plies; koopman keeps the original 1/3/6
        VKS = (2, 4, 6) if DISC else (1, 3, 6)
        BPAIRS = (((0, 2), (2, 4), (4, 6)) if DISC
                  else ((0, 1), (2, 3), (5, 6)))
        pred = {k: [] for k in VKS}
        true = {k: [] for k in VKS}
        dl = {k: [0.0, 0.0] for k in (2, 4, 6)}     # rolled-vs-true bits
        cp = {k: [0.0, 0.0] for k in (2, 4, 6)}     # copy baseline: root vs true bits
        br = {land: [0.0, 0.0] for _, land in BPAIRS}
        brnd = [0.0, 0.0]
        flips = [0.0, 0.0]                          # discrete arm: bits flipped/step
        gdiag = {}                                  # graph arm: pre-registered H-tests
        W = model.readout.weight
        for a in range(0, min(n_ch, len(ch_te)), 32):
            ci = CH_pos[ch_te[a:a + 32]]
            cb = len(ci)
            sq0, mv0, ed0, mk0, pl0, mvr0 = feat_chain(ci[:, 0])
            c0, h_mv0, _ = model.carrier(sq0, mv0, ed0, mk0)
            dq0 = dyn_codes(h_mv0)[0]
            moves = [dq0[torch.arange(cb, device=dev), pl0]]
            rows_mid = ci[:, 1:6].ravel()
            sqm, mvm, edm, mkm, plm, mvrm = feat_chain(rows_mid)
            _, h_mvm, _ = model.carrier(sqm, mvm, edm, mkm)
            dqm_all = dyn_codes(h_mvm)[0]
            mm = dqm_all[torch.arange(len(rows_mid), device=dev), plm
                         ].reshape(cb, 5, -1)
            dqm = dqm_all.reshape(cb, 5, MV_CAP, -1)
            moves += [mm[:, k] for k in range(5)]
            # played from/to per chain row 0..5 (graph arm packs)
            mvr6 = np.concatenate([mvr0[:, None], mvrm.reshape(cb, 5, MV_CAP, -1)],
                                  1).reshape(cb * 6, MV_CAP, -1)
            pl6 = torch.cat([pl0[:, None], plm.reshape(cb, 5)], 1).cpu().numpy()
            f6, t6 = _ft(mvr6, pl6.ravel())
            f6, t6 = f6.reshape(cb, 6), t6.reshape(cb, 6)
            if DISC:
                q1_root = model.rvq_bits(c0)[0]
                # T2 steps: mover's codes at rows 0, 2, 4 + played REPLY squares at
                # rows 1, 3, 5 (T2-conditional) -> horizons 2, 4, 6
                zs_bits = T_roll(
                    q1_root, [(moves[k], f6[:, k], t6[:, k],
                               f6[:, k + 1], t6[:, k + 1]) for k in (0, 2, 4)])
                zs = {k: model.levels_from_bits(zs_bits[i])[0]
                      for i, k in enumerate((2, 4, 6))}
                prev = q1_root
                for zb_ in zs_bits:
                    flips[0] += float((zb_ != prev).float().mean()) * cb
                    flips[1] += cb
                    prev = zb_
                if GRAPH and a == 0:
                    # pre-registered failure tests (JOURNAL 8/28): H1 supervised-but-
                    # unused edges (aucT), H5 hallucinated rollout connectivity
                    # (aucR), H2/H3 under-reach (cov = true-flip coverage by the
                    # move's influence cone, exact thanks to the frame axiom)
                    _, at0, _ = T_forward(q1_root, moves[0], f6[:, 0], t6[:, 0],
                                          frm2=f6[:, 1], to2=t6[:, 1],
                                          need_attn=True)
                    gdiag["aucT"] = _auc(at0[:, 0], att_targets(ci[:, 0]))
                    _, at4, _ = T_forward(zs_bits[1], moves[4], f6[:, 4],
                                          t6[:, 4], frm2=f6[:, 5], to2=t6[:, 5],
                                          need_attn=True)
                    gdiag["aucR"] = _auc(at4[:, 0], att_targets(ci[:, 4]))
                    tch = model.bit_trans.touched(q1_root, moves[0], f6[:, 0],
                                                  t6[:, 0], f6[:, 1], t6[:, 1])
                    sq2, mv2_, ed2, mk2, _, _ = feat_chain(ci[:, 2])
                    q1_2 = model.rvq_bits(
                        model.carrier(sq2, mv2_, ed2, mk2)[0])[0]
                    flp = (q1_root != q1_2).any(-1)
                    n_f = int(flp.sum())
                    gdiag["cov"] = (float((flp & tch).sum()) / n_f if n_f
                                    else float("nan"))
            else:
                cs = model.roller.roll(c0.reshape(cb, -1), moves)
                zs = {k: model.observe(cs[k - 1].reshape(cb, 64, model.sq_bits))[0]
                      for k in range(1, 7)}
            for k in VKS:
                fin = CH_OK[ci[:, k]]
                if not fin.any():
                    continue
                pv = F.linear(zs[k], W[:, :model.d_pos], model.readout.bias) \
                    + model.turn[torch.from_numpy(
                        CH_STM[ci[:, k]].astype(np.int64)).to(dev)]
                p = (torch.softmax(pv, -1) * S3).sum(-1).cpu().numpy()[fin]
                tw = CH_WDL[ci[:, k]][fin]
                tw = tw / tw.sum(-1, keepdims=True).clip(min=1e-6)
                pred[k].append(p); true[k].append(tw @ np.array([1.0, 0.5, 0.0]))
            if a < lag_sub:                          # even-lag deltas on a subsample
                z0 = model.observe(c0)[0]
                for k in (2, 4, 6):
                    sqk, mvk, edk, mkk, _, _ = feat_chain(ci[:, k])
                    ck_true, _, _ = model.carrier(sqk, mvk, edk, mkk)
                    zt = model.observe(ck_true)[0]
                    dl[k][0] += float((zs[k] != zt).float().mean()) * cb; dl[k][1] += cb
                    cp[k][0] += float((z0 != zt).float().mean()) * cb; cp[k][1] += cb
            if a < 64:                               # branch regret on a subsample
                for kf, land in BPAIRS:
                    wpk = CHW[ci[:, kf]]
                    cols = np.zeros((cb, br_k), np.int64)
                    ok_r = np.zeros(cb, bool)
                    for r in range(cb):
                        v = np.flatnonzero(np.isfinite(wpk[r]))
                        if len(v) < 2:
                            continue
                        b_ = v[wpk[r][v].argmax()]
                        rest = v[v != b_]
                        cols[r, 0] = b_
                        cols[r, 1:] = ero.choice(rest, br_k - 1,
                                                 replace=len(rest) < br_k - 1)
                        ok_r[r] = True
                    if not ok_r.any():
                        continue
                    okt = torch.from_numpy(ok_r).to(dev)
                    nv = int(ok_r.sum())
                    colst = torch.from_numpy(cols[ok_r]).to(dev)
                    src = dq0 if kf == 0 else dqm[:, kf - 1]
                    codes = src[okt].gather(
                        1, colst[..., None].expand(-1, -1, src.shape[-1])
                        ).reshape(nv * br_k, -1)
                    if DISC:
                        baseb = (q1_root if kf == 0 else zs_bits[kf // 2 - 1])[okt]
                        baseb = baseb.reshape(nv, 1, 64, -1).expand(
                            -1, br_k, -1, -1).reshape(nv * br_k, 64, -1)
                        rrb = cols[ok_r].ravel()
                        mv_kf = np.repeat((mvr6.reshape(cb, 6, MV_CAP, -1)[:, kf]
                                           )[ok_r], br_k, axis=0)
                        fbb, tbb = _ft(mv_kf, rrb)
                        f2b, t2b, okpv = _pv_reply_sq(
                            CH_FEN[np.repeat(ci[ok_r, kf], br_k)], fbb, tbb)
                        zb = model.levels_from_bits(
                            T_roll(baseb, [(codes, fbb, tbb, f2b, t2b)])[0])[0]
                        # T2 lands with the SAME side to move as the source row
                        stm_c = torch.from_numpy(
                            CH_STM[ci[:, kf]].astype(np.int64)[ok_r]).to(dev)
                    else:
                        base = (c0.reshape(cb, -1) if kf == 0 else cs[kf - 1])[okt]
                        base = base.reshape(nv, 1, -1).expand(-1, br_k, -1
                                                              ).reshape(nv * br_k, -1)
                        zb = model.observe(model.roller.step(base, codes)
                                           .reshape(-1, 64, model.sq_bits))[0]
                        stm_c = torch.from_numpy(
                            (1 - CH_STM[ci[:, kf]].astype(np.int64))[ok_r]).to(dev)
                    pvb = F.linear(zb, W[:, :model.d_pos], model.readout.bias)
                    pvb = pvb + model.turn[stm_c.repeat_interleave(br_k)]
                    sg = torch.from_numpy(CH_SGN[ci[:, kf]][ok_r]).to(dev)[:, None]
                    sc = (sg * (pvb[:, 0] - pvb[:, 2]).reshape(nv, br_k)).cpu().numpy()
                    if DISC:                 # PV-miss branches can't be picked
                        sc = sc + np.where(okpv.cpu().numpy().reshape(nv, br_k),
                                           0.0, -1e9)
                    wsel = wpk[np.arange(cb)[ok_r][:, None], cols[ok_r]]
                    pick = (sc + ero.random(sc.shape) * 1e-9).argmax(1)
                    reg = (wsel[:, 0] - wsel[np.arange(nv), pick]).clip(min=0)
                    br[land][0] += float(reg.sum()); br[land][1] += nv
                    rr = (wsel[:, :1] - wsel).clip(min=0)
                    brnd[0] += float(rr[:, 1:].mean(1).sum()); brnd[1] += nv
        corr = {}
        for k in VKS:
            if pred[k]:
                p = np.concatenate(pred[k]); t = np.concatenate(true[k])
                corr[k] = (float(np.corrcoef(p, t)[0, 1])
                           if p.std() > 1e-9 and t.std() > 1e-9 else 0.0)
            else:
                corr[k] = float("nan")
        delta = {k: dl[k][0] / max(dl[k][1], 1) for k in dl}
        copy = {k: cp[k][0] / max(cp[k][1], 1) for k in cp}
        breg = {k: br[k][0] / max(br[k][1], 1) for k in br}
        breg["rnd"] = brnd[0] / max(brnd[1], 1)
        if DISC:
            breg["flip"] = flips[0] / max(flips[1], 1)
        breg.update(gdiag)
        return corr, VKS, delta, copy, breg

    @torch.no_grad()
    def roll_random(n_root=24):
        """RANDOM-UNIFORM constructed sequences: even-lag bit deltas (no labels)."""
        import chess
        if CANON:
            return {}
        model.eval()
        ero = np.random.default_rng(1)
        roots = te_idx[ero.choice(len(te_idx), min(n_root, len(te_idx)), replace=False)]

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
        R_rows = build_rows(flat)
        mv_rand = R_rows["mv"].numpy().reshape(n, 7, MV_CAP, -1)
        sqf, mvf, edges, mask = feat_of(R_rows)
        c_all, h_mv_all, _ = model.carrier(sqf, mvf, edges, mask)
        c_all = c_all.reshape(n, 7, 64, model.sq_bits)
        dq = dyn_codes(h_mv_all)[0].reshape(n, 7, MV_CAP, -1)
        jt = torch.tensor(mv_js, device=dev)                          # (n,6)
        moves = [dq[torch.arange(n, device=dev), k].gather(
            1, jt[:, k][:, None, None].expand(-1, 1, dq.shape[-1]))[:, 0]
            for k in range(6)]
        stm0 = torch.tensor([b0[0].turn for b0 in seqs], dtype=torch.long, device=dev)
        stm_seq = [(stm0 + k + 1) % 2 for k in range(6)]
        if DISC:
            js = np.asarray(mv_js)                                    # (n,6)
            packs = []
            for k in (0, 2, 4):
                fk_, tk_ = _ft(mv_rand[:, k], js[:, k])
                f2_, t2_ = _ft(mv_rand[:, k + 1], js[:, k + 1])
                packs.append((moves[k], fk_, tk_, f2_, t2_))
            outs_b = T_roll(model.rvq_bits(c_all[:, 0])[0], packs)
            get = {k: model.levels_from_bits(outs_b[i])[0]
                   for i, k in enumerate((2, 4, 6))}
        else:
            outs = model.roll_and_read(c_all[:, 0], moves, stm_seq)
            get = {k: outs[k - 1][0] for k in (2, 4, 6)}
        z_true = {k: model.observe(c_all[:, k])[0] for k in (2, 4, 6)}
        return {k: float((get[k] != z_true[k]).float().mean()) for k in (2, 4, 6)}

    def run_eval(step, lw=float("nan"), lr_=float("nan"), lch=float("nan"),
                 lfan=float("nan"), lbr=float("nan"), tfa=float("nan"),
                 tfc=float("nan"), ldyn=float("nan")):
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
            corr, vks, delta, copy, breg = roll_real()
            rnd = roll_random()
            if DISC:
                rho = float("nan")
                tail = (f"dyn {ldyn:.3f} tfacc {tfa:.3f} (copy {tfc:.3f}) flip "
                        f"{breg.get('flip', float('nan')):.3f} | matMAE "
                        f"{MATSTAT.get('s', float('nan')):.2f}/"
                        f"{MATSTAT.get('r', float('nan')):.2f} dead "
                        f"{MATSTAT.get('dead', float('nan')):.2f}"
                        + (f" gate {MATSTAT.get('gsz', float('nan')):.2f}/"
                           f"FN {MATSTAT.get('gfn', float('nan')):.2f}"
                           if args.hard_gate else "")
                        + (f" loc {MATSTAT.get('loc', float('nan')):.2f}"
                           if args.w_loc > 0 else ""))
                # MARGIN CRITERION [science, alerts only -- Kaveh's gate doctrine]:
                # detection tracks whether the in-cone margin ever opens; a run
                # flat at 3000 is already decided (qgt17 0.9%, qgt19b 7.3% vs
                # qgt19 24.8%). Restart rather than pay out the full 8000.
                if args.hard_gate and step >= 3000 and (tfa - tfc) < 0.02:
                    tail += "  [ALERT: in-cone margin flat at %d -- restart]" % step
                if GRAPH:
                    tail += (f" | cov {breg.get('cov', float('nan')):.3f} auc "
                             f"{breg.get('aucT', float('nan')):.3f}/"
                             f"{breg.get('aucR', float('nan')):.3f}")
            else:
                rho = float(model.roller.rho(iters=20))
                tail = f"rho {rho:.4f}"
        vk_s = "/".join(str(k) for k in vks)
        print(f"[qgt] step {step:>5}/{args.steps} | wdl {lw:.3f} rloss {lr_:.4f} "
              f"roll {lch:.3f}+{lfan:.3f} br {lbr:.3f} | BALANCED {bal:.3f} recall "
              f"W/D/L {rec[0]:.2f}/{rec[1]:.2f}/{rec[2]:.2f} | REGRET {mregret:.4f} "
              f"(qat9 0.0872) | top1 {top1_acc:.3f} | concepts {live} | "
              f"vroll@{vk_s} " + "/".join(f"{corr[k]:+.3f}" for k in vks) +
              f" (8/24 ref +0.850/+0.808/+0.777 @1/3/6) | brreg@{vk_s} " +
              "/".join(f"{breg[k]:.3f}" for k in vks) +
              f" (rnd {breg['rnd']:.3f}) | "
              f"evenlag real {delta[2]:.3f}/{delta[4]:.3f}/{delta[6]:.3f} (copy "
              f"{copy[2]:.3f}/{copy[4]:.3f}/{copy[6]:.3f}) rand "
              f"{rnd.get(2, float('nan')):.3f}/{rnd.get(4, float('nan')):.3f}/"
              f"{rnd.get(6, float('nan')):.3f} | {tail}", flush=True)
        mv_alive = int((model.readout.weight[:, model.d_pos:].abs().sum(0) > 0).sum())
        if mv_alive == 0:
            print("  WARNING: the MOVE half of the readout is dead -- ranking impossible",
                  flush=True)
        return {"balanced": bal, "concepts": live, "wdl": lw, "top1": top1_acc,
                "regret": mregret, "rank": lr_, "roll_chain": lch, "roll_fan": lfan,
                "roll_branch": lbr,
                **{f"vroll{k}": corr[k] for k in vks},
                **{f"brreg{k}": breg[k] for k in vks},
                "brreg_rnd": breg["rnd"],
                "evenlag2_real": delta[2], "evenlag6_real": delta[6],
                "evenlag6_rand": rnd.get(6, float("nan")), "rho": rho,
                "tf_acc": tfa, "tf_copy": tfc,
                "flip": breg.get("flip", float("nan")),
                "flip_cov": breg.get("cov", float("nan")),
                "att_auc_tf": breg.get("aucT", float("nan")),
                "att_auc_roll": breg.get("aucR", float("nan")),
                "move_concepts_alive": mv_alive}

    if args.eval_only:
        sd = torch.load(args.eval_only, map_location="cpu", weights_only=False)
        missing = model.load_state_dict(sd["state_dict"], strict=False)
        print(f"[qgt] eval-only: {args.eval_only} (step {sd.get('step', '?')}, "
              f"{len(missing.missing_keys)} missing)", flush=True)
        run_eval(int(sd.get("step", 0)))
        return

    # ---- gates ----------------------------------------------------------------------
    if args.units and CANON:
        # C-GATES (the canon arm's battery; the U/D battery assumes White-POV
        # flips and 2-ply semantics). Taxonomy unchanged: implementation kills,
        # science alerts.
        import chess as _ch
        from catspace.research.tools.training_infra.qgt_canon import (
            canon as _cn, from_canon as _fc)
        # C1 [impl]: mover-canonical everywhere
        assert bool(np.all(STM)) and bool(np.all(CH_STM)), "canon STM not all-White"
        # C2 [impl]: chain ply integrity -- canon played move replays the chain
        ci_t = CH_pos[ch_tr[:4]]
        for r in range(4):
            for k in range(6):
                b_ = _ch.Board(str(CH_FEN[ci_t[r, k]]))
                cb_, fl_ = _cn(b_)
                pj = int(CHA["played"][ci_t[r, k]])
                mvc = list(cb_.legal_moves)[pj]
                b_.push(_fc(mvc, fl_))
                want = str(CH_FEN[ci_t[r, k + 1]]).split()
                got = b_.fen().split()
                assert got[:2] == want[:2],                     f"C2 ply integrity broken at chain {r} ply {k}"
        # C3 [impl]: frame axiom exact + PERM involution (identity+delta mode
        # only -- in DIRECT mode copying is the GATE's job, asserted by C7)
        with torch.no_grad():
            sqg, mvg, edg, mkg, plg, mvrg = feat_chain(ci_t[:, 0])
            cg, hg, _ = model.carrier(sqg, mvg, edg, mkg)
            q1g = model.rvq_bits(cg)[0]
            codeg = dyn_codes(hg)[0][torch.arange(4, device=dev), plg]
            fg, tg = _ft(mvrg, plg.cpu().numpy())
            model.train()
            lg_g, _, _ = model.bit_trans(q1g, codeg, fg, tg)
            tchg = model.bit_trans.touched(q1g, codeg, fg, tg)
            if not args.direct_t and bool((~tchg).any()):
                du = float((lg_g - model.bit_trans.w_id * q1g)[~tchg].abs().max())
                assert du == 0.0, f"C3 frame axiom broken: {du}"
            assert bool((lg_g[:, PERM_T][:, PERM_T] == lg_g).all()), "PERM not involution"
        # C4 [impl]: material sign follows the mirror
        fl_rows = np.flatnonzero(CH_FLIP)[:4]
        for r_ in fl_rows:
            assert abs(CH_MAT[r_] + _material(CH_FEN[r_])) < 1e-6,                 "C4 flipped-row material sign wrong"
        # C5 [impl]: EMA moves toward online after a perturbation
        with torch.no_grad():
            before = ema.carrier_proj.weight.clone()
            model.carrier_proj.weight.add_(1.0)
        ema_update()
        moved = float((ema.carrier_proj.weight - before).abs().max())
        with torch.no_grad():
            model.carrier_proj.weight.sub_(1.0)
        assert 0 < moved < 1.0, f"C5 EMA update wrong: moved {moved}"
        # C6 [impl]: the canon anchor runs end to end, losses finite, grads reach
        # the ENCODER through the teacher (the closure direction) + sibling head
        l_dyn_c, l_val_c, l_br_c, nl, nb_, tfa_c, tfc_c = chain_anchor_canon(4)
        assert all(torch.isfinite(x) for x in (l_dyn_c, l_val_c, l_br_c))
        g_enc = torch.autograd.grad(l_dyn_c, model.carrier_proj.weight,
                                    retain_graph=True, allow_unused=True)[0]
        assert g_enc is not None and float(g_enc.abs().sum()) > 0,             "C6 closure broken: teacher loss does not reach the encoder"
        bb_c = blunder_batch(4)
        assert bb_c is not None
        _, js_c, R1c, _, _, mat_c = bb_c
        sq1c, mv1c, ed1c, mk1c = feat_of(R1c)
        c1c, _, _ = model.carrier(sq1c, mv1c, ed1c, mk1c)
        f_c = model.sib_scorer(c1c.reshape(len(js_c), -1)).squeeze(-1)
        g_sib = torch.autograd.grad(f_c.sum(), model.carrier_proj.weight,
                                    allow_unused=True)[0]
        assert g_sib is not None and float(g_sib.abs().sum()) > 0,             "C6 sibling scorer does not reach the carrier"
        if args.hard_gate:
            g7 = torch.from_numpy(_gate_rows(ci_t[:, 0])).to(dev)
            zr7 = T1_roll(q1g, [(codeg, fg, tg, g7)])[0]
            zr7p = zr7[:, PERM_T]                  # back to parent frame
            ung = ~g7
            assert bool((zr7p[ung] == q1g[ung]).all()), \
                "C7 hard gate leaks: ungated squares changed"
            fr_in = g7[torch.arange(4, device=dev), fg]
            to_in = g7[torch.arange(4, device=dev), tg]
            assert bool(fr_in.all()) and bool(to_in.all()), \
                "C7 from/to not in own touch set"
            print(f"  C7 PASS hard gate: ungated squares copy BIT-IDENTICALLY "
                  f"through the roll; from/to gated; mean |T| "
                  f"{float(g7.float().mean()) * 64:.1f}/64 squares")
        print(f"  C1-C6 PASS canon arm: all-White frames; ply integrity replayed; "
              f"frame axiom exact ({int((~tchg).sum())}/256 untouched) + PERM "
              f"involution; material sign mirrored; EMA live; closure gradient "
              f"reaches the encoder; sibling head reaches the carrier "
              f"(anchor tfacc {tfa_c:.3f} / copy {tfc_c:.3f}; dead-bit frac "
              f"{MATSTAT.get('dead', float('nan')):.3f} [science])")
        if args.w_loc > 0:
            # C8 [impl]: the locality loss must reach the ENCODER (it is the whole
            # point -- it reshapes the CODE, not the transition)
            l_dyn_l, _, _, _, _, _, _ = chain_anchor_canon(4)
            g_loc = torch.autograd.grad(l_dyn_l, model.carrier_proj.weight,
                                        allow_unused=True)[0]
            assert g_loc is not None and float(g_loc.abs().sum()) > 0, \
                "C8 locality loss does not reach the encoder"
            print(f"  C8 PASS encoder-locality: gradient reaches the carrier; "
                  f"out-of-cone disagreement at init "
                  f"{MATSTAT.get('loc', float('nan')):.2f} [science]")
        print("  ALL QGT GATES PASS")
        _sf_teardown()
        return

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

        # U12-analogue: the rolled anchors train DYN codes, never value codes
        l_ch, l_br, _, n_lab, n_br = chain_anchor(8)
        assert n_lab > 0, "chain anchor found no labelled rows -- it would train on nothing"
        assert n_br > 0, "branch anchor found no rankable rows -- it would train on nothing"
        l_tot = l_ch + l_br
        g_val = torch.autograd.grad(l_tot, model.mv_proj.weight, retain_graph=True,
                                    allow_unused=True)[0]
        g_dyn = torch.autograd.grad(l_tot, model.dyn_mv_proj.weight, retain_graph=True,
                                    allow_unused=True)[0]
        g_K = torch.autograd.grad(l_tot, model.roller.K, retain_graph=True,
                                  allow_unused=True)[0]
        g_Kb = torch.autograd.grad(l_br, model.roller.K, allow_unused=True)[0]
        assert g_val is None or float(g_val.abs().sum()) == 0.0, \
            "rolled anchors leak into the VALUE move codes"
        assert g_dyn is not None and float(g_dyn.abs().sum()) > 0, \
            "rolled anchors do not train the dyn codes"
        assert g_K is not None and float(g_K.abs().sum()) > 0, \
            "rolled anchors do not reach K through the composition"
        assert g_Kb is not None and float(g_Kb.abs().sum()) > 0, \
            "the BRANCH loss alone does not reach K -- deep grounding is dead"
        print(f"  U12 PASS anchor isolation: value-code grad "
              f"{'None' if g_val is None else 0.0}; dyn codes + K train; branch loss "
              f"alone reaches K ({n_lab} value rows, {n_br} branch rows on 8 chains; "
              f"slot-0-is-best asserted in-loop under --units)")

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
        if DISC:
            print("  U15 (discrete/graph): rolled fan retired -- the encoder-level "
                  "fan is the constructed-children stream, gated as D6")
        elif True:
            g_B = torch.autograd.grad(l_fan, model.roller.B.weight,
                                      retain_graph=True, allow_unused=True)[0]
            g_cp = torch.autograd.grad(l_fan, model.carrier_proj.weight,
                                       retain_graph=True, allow_unused=True)[0]
            assert g_B is not None and float(g_B.abs().sum()) > 0, \
                "fan loss does not train the dynamics' move path"
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

        if DISC:
            # D1-D4: teacher pairs are real; the BCE is isolated from the encoder and
            # the value move codes; free-run value/branch gradients reach T by STE
            l_dyn, l_chd, l_brd, n_lab_d, n_br_d, tfa_, tfc_ = chain_anchor_discrete(8)
            assert n_lab_d > 0 and n_br_d > 0, "discrete anchor found nothing to train"
            assert tfc_ < 0.999, \
                f"teacher pairs identical (copy {tfc_:.4f}) -- no transition to learn"
            g_enc = torch.autograd.grad(l_dyn, model.carrier_proj.weight,
                                        retain_graph=True, allow_unused=True)[0]
            g_mv2 = torch.autograd.grad(l_dyn + l_chd + l_brd, model.mv_proj.weight,
                                        retain_graph=True, allow_unused=True)[0]
            _outp = (model.bit_trans.out.weight if GRAPH
                     else model.bit_trans.head[-1].weight)
            g_T = torch.autograd.grad(l_dyn, _outp,
                                      retain_graph=True, allow_unused=True)[0]
            g_dq = torch.autograd.grad(l_dyn, model.dyn_mv_proj.weight,
                                       retain_graph=True, allow_unused=True)[0]
            g_aT = torch.autograd.grad(l_chd + l_brd, model.bit_trans.w_id,
                                       retain_graph=True, allow_unused=True)[0]
            g_aR = torch.autograd.grad(l_chd + l_brd, model.readout.weight,
                                       allow_unused=True)[0]
            assert g_enc is None or float(g_enc.abs().sum()) == 0.0, \
                "teacher-forced BCE trains the ENCODER -- the moving-target disease"
            assert g_mv2 is None or float(g_mv2.abs().sum()) == 0.0, \
                "discrete dynamics leak into the VALUE move codes"
            assert g_T is not None and float(g_T.abs().sum()) > 0, \
                "teacher-forced BCE does not train the transition"
            assert g_dq is not None and float(g_dq.abs().sum()) > 0, \
                "the teacher no longer trains the move encoder (world-model split broken)"
            assert g_aT is None or float(g_aT.abs().sum()) == 0.0, \
                "anchors reach T -- the qgt4 corruption channel is open"
            assert g_aR is not None and float(g_aR.abs().sum()) > 0, \
                "anchors train nothing"
            print(f"  D1-D5 PASS discrete arm (qgt5 separation): teacher pairs real "
                  f"(copy {tfc_:.3f}, tf-acc {tfa_:.3f} at init); BCE trains T + move "
                  f"encoder, never the board encoder or value codes; anchors train "
                  f"the readout and CANNOT reach T ({n_lab_d} value rows, {n_br_d} "
                  f"branch rows on 8 chains)")

            # D6: the blunder-pair stream (implementation wiring; label contrast is
            # a [science] alert)
            bb6 = blunder_batch(6)
            assert bb6 is not None, "blunder stream produced nothing"
            gs6, js6, R16, R26, rep6, mat6 = bb6
            assert rep6.shape == (len(gs6), 2) and int(rep6.max()) < 64, \
                "reply squares malformed"
            assert np.isfinite(mat6).all() and np.abs(mat6).max() <= 39, \
                "material labels malformed"
            gap6 = float(np.mean(np.abs(
                TGT[gs6[::2], js6[::2]] - TGT[gs6[1::2], js6[1::2]])))
            sq16, mv16, ed16, mk16 = feat_of(R16)
            c16, _, _ = model.carrier(sq16, mv16, ed16, mk16)
            z16 = model.levels_from_bits(model.rvq_bits(c16)[0])[0]
            pv16 = F.linear(z16, model.readout.weight[:, :model.d_pos],
                            model.readout.bias)
            l6 = _ce_rows(pv16, torch.from_numpy(CW_[gs6, js6]).to(dev))
            g6e = torch.autograd.grad(l6, model.carrier_proj.weight,
                                      allow_unused=True)[0]
            assert g6e is not None and float(g6e.abs().sum()) > 0, \
                "blunder CE does not reach the encoder -- its whole point"
            if gap6 < 0.1:
                print(f"  ALERT D6 [science]: sampled-children label variance only {gap6:.3f}",
                      flush=True)
            print(f"  D6 PASS blunder stream: {len(gs6)} constructed children, "
                  f"mean |dwp| between sampled children {gap6:.3f}, CE reaches the encoder")

            # D7: material grounding (implementation wiring). No 39-cap:
            # promotions push balance past standard-start material (found 8/29)
            assert np.isfinite(MAT).all() and np.isfinite(CH_MAT).all()
            assert np.percentile(np.abs(MAT), 95) <= 39, "material labels implausible"
            with torch.no_grad():
                sq0d, mv0d, ed0d, mk0d = batch_of(torch.from_numpy(gs6))
                c0d, h0d, _ = model.carrier(sq0d, mv0d, ed0d, mk0d)
                q0d = model.rvq_bits(c0d)[0]
            dqd = dyn_codes(h0d.detach())[0]
            cded = dqd[torch.arange(len(gs6), device=dev),
                       torch.from_numpy(js6).to(dev)]
            fdd, tdd = _ft(MVNP[gs6], js6)
            lg_d, _, _ = T_forward(q0d, cded, fdd, tdd,
                                   frm2=torch.from_numpy(rep6[:, 0]).to(dev),
                                   to2=torch.from_numpy(rep6[:, 1]).to(dev))
            l_mr = F.huber_loss(_mat_read(lg_d),
                                torch.from_numpy(mat6 / 10.0).to(dev))
            gT = torch.autograd.grad(l_mr, model.bit_trans.out.weight,
                                     retain_graph=True, allow_unused=True)[0]
            gM = torch.autograd.grad(l_mr, model.mat_head.weight,
                                     allow_unused=True)[0]
            assert gT is not None and float(gT.abs().sum()) > 0, \
                "rolled-material loss does not reach the transition"
            assert gM is not None and float(gM.abs().sum()) > 0, \
                "rolled-material loss does not reach mat_head"
            print(f"  D7 PASS material grounding: labels exact "
                  f"(corpus max |{np.abs(MAT).max():.0f}|), rolled-material loss "
                  f"reaches T and mat_head")


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
        if GRAPH:
            # pre-registered failure tests, pre-launch halves (JOURNAL 8/28)
            from catspace.research.tools.training_infra.train.scaffold import (
                frozen)
            from catspace.research.tools.training_infra.qgt_field import (
                GraphBitTransition as _GBT2)
            ci_g = CH_pos[ch_tr[:8]]
            with torch.no_grad():
                sqg, mvg, edg, mkg, plg, mvrg = feat_chain(ci_g[:, 0])
                cg, hg, _ = model.carrier(sqg, mvg, edg, mkg)
                q1g = model.rvq_bits(cg)[0]
                codeg = dyn_codes(hg)[0][torch.arange(8, device=dev), plg]
                _, _, _, _, plg1, mvrg1 = feat_chain(ci_g[:, 1])
            fg, tg = _ft(mvrg, plg.cpu().numpy())
            fg2, tg2 = _ft(mvrg1, plg1.cpu().numpy())     # the played REPLY (row 1)
            tgt_g = att_targets(ci_g[:, 0])
            # GATE TAXONOMY (Kaveh 2026-08-28): asserts KILL only on IMPLEMENTATION
            # bugs (wiring, exact algebra, measurement purity). Scientific outcomes
            # -- can the model learn X, is coverage high enough -- ALERT and report;
            # they are findings, never blockers.
            # G2 (implementation -- may kill): frame axiom exact in TRAIN mode
            model.train()
            lg_g, _, _ = model.bit_trans(q1g, codeg, fg, tg, frm2=fg2, to2=tg2)
            tchg = model.bit_trans.touched(q1g, codeg, fg, tg, fg2, tg2)
            if bool((~tchg).any()):
                du_g = float((lg_g - model.bit_trans.w_id * q1g)[~tchg].abs().max())
                assert du_g == 0.0, f"frame axiom broken in train mode: {du_g}"
            else:
                print("  ALERT G2: influence cone covered every square in this "
                      "sample -- frame-axiom train-mode check skipped", flush=True)
            # G1 (SCIENCE -- alerts only): warm-start learnability mini-fit
            with frozen(model):
                o_g = torch.optim.Adam(model.bit_trans.qa[0].parameters(),
                                       lr=3e-3)
                auc0 = None
                for it in range(60):
                    _, a0, _ = model.bit_trans(q1g, codeg, fg, tg, frm2=fg2,
                                               to2=tg2, need_attn=True)
                    if it == 0:
                        auc0 = _auc(a0[:, 0], tgt_g)
                    lw_g = _GBT2.edge_warm_loss(a0, tgt_g)
                    o_g.zero_grad(); lw_g.backward(); o_g.step()
                with torch.no_grad():
                    _, a1, _ = model.bit_trans(q1g, codeg, fg, tg, frm2=fg2,
                                               to2=tg2, need_attn=True)
                    auc1 = _auc(a1[:, 0], tgt_g)
            if auc1 <= auc0 + 0.05:
                print(f"  ALERT G1 (finding, not a blocker): warm start shows no "
                      f"learnability at init -- AUC {auc0:.3f} -> {auc1:.3f} in 60 "
                      f"steps; the run may still prove otherwise", flush=True)
            # G3 (SCIENCE -- informational): the coverage instrument
            with torch.no_grad():
                sq2g, mv2g, ed2g, mk2g, _, _ = feat_chain(ci_g[:, 2])
                q1g2 = model.rvq_bits(model.carrier(sq2g, mv2g, ed2g, mk2g)[0])[0]
            flpg = (q1g != q1g2).any(-1)
            covg = float((flpg & tchg).sum()) / max(int(flpg.sum()), 1)
            print(f"  G1-G3 graph arm: frame axiom exact in TRAIN mode "
                  f"({int((~tchg).sum())}/512 untouched) [implementation]; "
                  f"warm-start mini-fit AUC {auc0:.3f} -> {auc1:.3f} [science]; "
                  f"flip coverage at init {covg:.2f} [science]")
        print("  ALL QGT GATES PASS")
        _sf_teardown()
        return

    start = resume_from(args.resume, model, opt=opt, rng=rng) if args.resume else 0
    cfg = TrainConfig(out=args.out, steps=args.steps, ckpt_every=args.ckpt_every,
                      eval_every=args.eval_every, start_step=start,
                      experiment="catspace", run_name=args.out.split("/")[-1],
                      device=str(dev))
    try:
        standard_train(step_fn, model, cfg, args, opt=opt, rng=rng)
    finally:
        _sf_teardown()


if __name__ == "__main__":
    main()
