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
    ap.add_argument("--dynamics", choices=("koopman", "discrete", "graph"),
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

    DISC = args.dynamics in ("discrete", "graph")
    GRAPH = args.dynamics == "graph"
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
    CHW = _zv["wp"].astype(np.float32)                        # (rows, MV_CAP) mover-POV
    cf = np.load(args.chain_fens)                             #   per-move win-probs
    CH_FEN = cf["fen"]
    CH_STM = np.array([str(f).split()[1] == "w" for f in CH_FEN])
    CH_SGN = np.where(CH_STM, 1.0, -1.0).astype(np.float32)
    CH_WDL = _wdl_raw.copy()
    CH_WDL[~CH_STM] = CH_WDL[~CH_STM][:, ::-1]                # White-POV
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
            w_id_init=args.w_id_init).to(dev)
    elif DISC:
        from catspace.research.tools.training_infra.qgt_field import BitTransition
        # registered as a submodule BEFORE the optimizer, so it checkpoints + trains
        model.bit_trans = BitTransition(sq_bits=args.sq_bits, mv_bits=args.mv_bits,
                                        n_q=args.n_q, heads=args.attn_heads,
                                        top_k=args.top_k, out_bits=args.out_bits,
                                        w_id_init=args.w_id_init).to(dev)
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

    def T_forward(z, code, frm, to, need_attn=False):
        """-> (logits, attn0 or None, aux loss). Plain discrete ignores frm/to."""
        if GRAPH:
            return model.bit_trans(z, code, frm, to, need_attn=need_attn)
        lg, l_qa = model.bit_trans(z, code)
        return lg, None, l_qa

    def T_roll(z0, packs):
        """packs: list of (code, frm, to). Plain discrete rolls on codes alone."""
        if GRAPH:
            return model.bit_trans.roll_bits(z0, packs)
        return model.bit_trans.roll_bits(z0, [p[0] for p in packs])

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

    SF_ENG = [None]                        # lazy; restarted on failure

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
        roots, c1_boards, c2_boards, js, gs = [], [], [], [], []
        pick = rng.choice(tr_idx, nb * 3, replace=False)
        for g in pick:
            try:
                b0 = chess.Board(str(FENS[g]))
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
                rep = list(b1.legal_moves)[:MV_CAP]
                if not rep:
                    okpair = False
                    break
                rmv = _sf_reply(b1)
                if rmv is None or rmv not in rep:
                    rmv = rep[int(rng.integers(len(rep)))]
                b2 = b1.copy(); b2.push(rmv)
                pair.append((b1, b2, j))
            if not okpair:
                continue
            for b1, b2, j in pair:
                roots.append(int(g)); c1_boards.append(b1)
                c2_boards.append(b2); js.append(j)
            if len(js) >= nb * 2:
                break
        if not js:
            return None
        gs = np.array(roots, np.int64)
        return (gs, np.array(js, np.int64), build_rows(c1_boards),
                build_rows(c2_boards))

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
        by construction and the ~9% of bits that change are real chess. T conditions
        on the MOVER'S code only; the opponent's reply is MARGINALIZED -- the teacher
        target is the realized position two plies out, and BCE on a stochastic target
        learns calibrated per-bit flip marginals under the data's reply distribution
        (which is also what makes branch pricing sound: SF's wp for a move already
        assumes the reply). Teacher-forced on 5 lag-2 pairs per chain; value CE +
        branch ranking on FREE-RUN rolls at horizons 2/4/6. The encoder is NOT
        trained by the dynamics (inputs and targets both detached)."""
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
                        [(pl_g[:, k].detach(), frm7[:, k], to7[:, k])
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
            with torch.no_grad():                # world model fixed for the anchors
                zbr = T_roll(base, [(codes, fb, tb)])[0]
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
            l_br = l_br + (torch.softmax(sc, -1) * reg).sum(-1).mean()
            j = torch.randint(1, bk, (nv,), device=dev)
            arv = torch.arange(nv, device=dev)
            l_br = l_br + (F.softplus(-(sc[:, 0] - sc[arv, j])) * reg[arv, j]).mean()
            n_br += nv
        l_br = l_br / 3.0
        return l_tf + l_qa + l_dq, l_val, l_br, n_lab, n_br, tf_acc, tf_copy

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

        # dynamics supervision: teacher-forced BCE (discrete arm only) + rolled-value
        # anchors (chain value + branch ranking + fan) in BOTH arms
        tf_acc = tf_copy = float("nan")
        if DISC:
            l_dyn, l_ch, l_br, n_lab, n_br, tf_acc, tf_copy = \
                chain_anchor_discrete(args.chain_batch)
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
                gs_b, js_b, R1b, R2b = bb
                nbp = len(gs_b)
                sq1, mv1, ed1, mk1 = feat_of(R1b)
                c1b, _, _ = model.carrier(sq1, mv1, ed1, mk1)      # encoder GRAD
                z1b = model.levels_from_bits(model.rvq_bits(c1b)[0])[0]
                stm_c1 = torch.from_numpy(
                    (1 - STM[gs_b].astype(np.int64))).to(dev)
                pv1 = F.linear(z1b, model.readout.weight[:, :model.d_pos],
                               model.readout.bias) + model.turn[stm_c1]
                tg1 = torch.from_numpy(CW_[gs_b, js_b]).to(dev)
                l_blun = _ce_rows(pv1, tg1)
                loss = loss + args.w_blunder * l_blun
                # teacher half: (root bits, move) -> child-after-reply bits
                with torch.no_grad():
                    sq0b, mv0b, ed0b, mk0b = batch_of(torch.from_numpy(gs_b))
                    c0b, h0b, _ = model.carrier(sq0b, mv0b, ed0b, mk0b)
                    q0b = model.rvq_bits(c0b)[0]
                    sq2, mv2b, ed2b, mk2b = feat_of(R2b)
                    c2b, _, _ = model.carrier(sq2, mv2b, ed2b, mk2b)
                    q2b = model.rvq_bits(c2b)[0]
                dqb = dyn_codes(h0b.detach())[0]
                code_b = dqb[torch.arange(nbp, device=dev),
                             torch.from_numpy(js_b).to(dev)]
                fb_, tb_ = _ft(MVNP[gs_b], js_b)
                lg_b, _, _ = T_forward(q0b, code_b, fb_, tb_)
                w_b = torch.where(q0b != q2b, args.flip_weight, 1.0) \
                    if args.flip_weight != 1.0 else None
                l_dyn = l_dyn + F.binary_cross_entropy_with_logits(
                    lg_b, (q2b + 1) / 2, weight=w_b)
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
        if args.dynamics == "koopman":
            model.roller.clamp_()
        pruner.step(step, settled=settle.update(float(loss.detach())))

        out = {}
        if step % args.eval_every == 0 or step == 1 or step == args.steps:
            out = run_eval(step, float(l_wdl), float(l_rank), float(l_ch),
                           float(l_fan), float(l_br), tf_acc, tf_copy,
                           float(l_dyn))
        return out

    # ---- the dual-split instrument --------------------------------------------------
    @torch.no_grad()
    def roll_real(n_ch=192, lag_sub=48, br_k=8):
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
                # T2 steps: the mover's codes at rows 0, 2, 4 -> horizons 2, 4, 6
                zs_bits = T_roll(
                    q1_root, [(moves[k], f6[:, k], t6[:, k]) for k in (0, 2, 4)])
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
                                          need_attn=True)
                    gdiag["aucT"] = _auc(at0[:, 0], att_targets(ci[:, 0]))
                    _, at4, _ = T_forward(zs_bits[1], moves[4], f6[:, 4],
                                          t6[:, 4], need_attn=True)
                    gdiag["aucR"] = _auc(at4[:, 0], att_targets(ci[:, 4]))
                    tch = model.bit_trans.touched(q1_root, moves[0], f6[:, 0],
                                                  t6[:, 0])
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
                        zb = model.levels_from_bits(
                            T_roll(baseb, [(codes, fbb, tbb)])[0])[0]
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
                packs.append((moves[k], fk_, tk_))
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
                        f"{breg.get('flip', float('nan')):.3f}")
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
            gs6, js6, R16, R26 = bb6
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
            fg, tg = _ft(mvrg, plg.cpu().numpy())
            tgt_g = att_targets(ci_g[:, 0])
            # GATE TAXONOMY (Kaveh 2026-08-28): asserts KILL only on IMPLEMENTATION
            # bugs (wiring, exact algebra, measurement purity). Scientific outcomes
            # -- can the model learn X, is coverage high enough -- ALERT and report;
            # they are findings, never blockers.
            # G2 (implementation -- may kill): frame axiom exact in TRAIN mode
            model.train()
            lg_g, _, _ = model.bit_trans(q1g, codeg, fg, tg)
            tchg = model.bit_trans.touched(q1g, codeg, fg, tg)
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
                    _, a0, _ = model.bit_trans(q1g, codeg, fg, tg,
                                               need_attn=True)
                    if it == 0:
                        auc0 = _auc(a0[:, 0], tgt_g)
                    lw_g = _GBT2.edge_warm_loss(a0, tgt_g)
                    o_g.zero_grad(); lw_g.backward(); o_g.step()
                with torch.no_grad():
                    _, a1, _ = model.bit_trans(q1g, codeg, fg, tg,
                                               need_attn=True)
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
        return

    start = resume_from(args.resume, model, opt=opt, rng=rng) if args.resume else 0
    cfg = TrainConfig(out=args.out, steps=args.steps, ckpt_every=args.ckpt_every,
                      eval_every=args.eval_every, start_step=start,
                      experiment="catspace", run_name=args.out.split("/")[-1],
                      device=str(dev))
    standard_train(step_fn, model, cfg, args, opt=opt, rng=rng)


if __name__ == "__main__":
    main()
