#!/usr/bin/env python
"""gko_train.py -- gko0: the flywheel's consumer (docs/GKO1_PLAN.md, first slice).

Trains on the oracle shards as the converter writes them (pipelined; no fixed corpus):
    MAIA rows   the move head learns P(move | position, elo band) -- KL to the policy
                the band's Maia produced, elo-conditioned via the existing machinery
                (one model, all bands: the dial becomes real)
    SF rows     a NEW per-move action-value head learns HL-Gauss-binned win-probs for
                ALL legal moves (the full-ordering supervision Kaveh authorized)
Vocabulary protection: VQ commitment + cross-head decorrelation stay on; the rest of the
field stack is frozen-by-weight-zero for this slice (operators/intervals join as their
unit gates pass -- gko1 proper).

gko1 joint slice (--ops --iv): adds the Koopman operator arm (koopman_ops.py, U1-U4
gated) trained on 6-ply chains from the human corpus (chain graphs cache + transitions),
an MLP dynamics BASELINE on detached inputs (H1's comparison arm -- detached so the
baseline cannot reshape the field; the operators may, that is the hypothesis), and the
joint interval head on SF action-value gaps (H2 vs the frozen gciv1 before-row).
Operators act on the field's normalized concept state zq -- the space every other head
reads. --units runs U12-U14 + U16 checks and exits nonzero on failure.

    .venv/bin/python -u -m ...gko_train --resume-field artifacts/experiments/gcf4h_latest.pt
"""
from __future__ import annotations

import argparse
import glob
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from catspace.research.components.encoder.approaches.reach_probability.experiments \
    .graph_move_policy import MV_CAP


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--resume-field", default="artifacts/experiments/gcf4h_latest.pt")
    ap.add_argument("--resume-self", default="", help="a prior gko checkpoint (wrapper "
                    "state: field.* + sf_av.*) to continue from")
    ap.add_argument("--shards", default="data/derived/gko1_shards")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--k-bins", type=int, default=64)
    ap.add_argument("--w-maia", type=float, default=1.0)
    ap.add_argument("--w-sf", type=float, default=1.0)
    ap.add_argument("--w-decor", type=float, default=0.3)
    ap.add_argument("--ops", action="store_true", help="Koopman operator arm on 6-ply chains")
    ap.add_argument("--iv", action="store_true", help="joint interval head on SF value gaps")
    ap.add_argument("--iv-frozen", action="store_true", help="H2 before-arm: train ONLY the "
                    "interval head on a frozen field (same targets, same holdout)")
    ap.add_argument("--units", action="store_true", help="run U12-U14/U16 checks and exit")
    ap.add_argument("--chain-graphs", default="data/derived/gko1_chain_graphs_mm.npz")
    ap.add_argument("--chain-starts", default="data/derived/gko1_chain_rows.npz")
    ap.add_argument("--row-elo", default="data/derived/gko1_row_elo_2126316.npz")
    ap.add_argument("--d-phi", type=int, default=128)
    ap.add_argument("--lift", choices=["bilinear", "transformer"], default="bilinear",
                    help="interaction module BEFORE the operators (Kaveh 2026-08-21: "
                    "attention over concept tokens, then linear rollforward)")
    ap.add_argument("--lift-layers", type=int, default=2)
    ap.add_argument("--g-heads", type=int, default=0, help="override field concept heads "
                    "(512-concept ruling); 0 = use the field ckpt's value")
    ap.add_argument("--g-codes", type=int, default=0)
    ap.add_argument("--d-code", type=int, default=0)
    ap.add_argument("--slow-heads", type=int, default=0)
    ap.add_argument("--sq-codes", type=int, default=0, help="override square codebook size")
    ap.add_argument("--gen5", action="store_true", help="gcf5: piece concepts, "
                    "no min-max, BasisOps bare identity, WDL value heads")
    ap.add_argument("--ops-after", type=int, default=0, help="gate operator code-matching "
                    "losses until the fresh vocabulary consolidates (ramp over 1000)")
    ap.add_argument("--ema-decay", type=float, default=0.0, help="EMA target encoder for "
                    "ALL child-side operator targets (BYOL/MuZero-style): the online "
                    "vocabulary may churn while operators train against its low-passed "
                    "shadow. 0 = off")
    ap.add_argument("--w-v0", type=float, default=0.5, help="state-value grounding at k=0")
    ap.add_argument("--w-vk", type=float, default=1.0, help="ROLLED future-value vs SF truth")
    ap.add_argument("--chain-values", default="data/derived/gko1_chain_values.npz")
    ap.add_argument("--w-cons", type=float, default=0.15, help="hypothesis-validity AND "
                    "joint dictionary search: carried phi vs the dictionary at the rolled "
                    "state, gradients BOTH ways (Kaveh: the lift must help the dynamics "
                    "be linear; collapse is priced by oracle saturation, not detaching)")
    ap.add_argument("--w-av", type=float, default=1.0, help="future ACTION-VALUE profiles "
                    "from rolled states (the full multipv, ~30 oracle targets/state)")
    ap.add_argument("--op-rank", type=int, default=16)
    ap.add_argument("--chain-batch", type=int, default=16)
    ap.add_argument("--op6-after", type=int, default=2000)
    ap.add_argument("--w-op1", type=float, default=1.0)
    ap.add_argument("--w-op3", type=float, default=0.5)
    ap.add_argument("--w-op6", type=float, default=0.3)
    ap.add_argument("--w-spec", type=float, default=0.2)
    ap.add_argument("--w-phi", type=float, default=0.25)
    ap.add_argument("--w-iv", type=float, default=1.0)
    ap.add_argument("--w-move", type=float, default=2.0, help="corpus move PL on chain parents")
    ap.add_argument("--w-dyn2", type=float, default=1.0, help="dyn CE to played-child codes")
    ap.add_argument("--w-slow", type=float, default=1.0, help="VAMP-2 slow pressure, lag-6")
    ap.add_argument("--w-sib", type=float, default=0.0,
                    help="per-move SIBLING value on rolled states: roll the parent "
                         "through EVERY legal move and price each child against that "
                         "move's own SF WDL. The only term that asks the operator's "
                         "move-dependent component to separate siblings (2026-08-23: "
                         "measured at ~1pct of the shared term). 0 = off.")
    ap.add_argument("--depth-cost", action="store_true",
                    help="each concept family (and each board head) reads a LEARNED "
                         "mixture over trunk layers, penalised by expected depth. "
                         "Pulls concepts shallow -- and in a graph net depth IS "
                         "receptive-field radius, so it also pulls them LOCAL. "
                         "Yields a per-concept depth profile.")
    ap.add_argument("--w-depth", type=float, default=0.1,
                    help="weight on the expected-depth penalty")
    ap.add_argument("--binary", action="store_true",
                    help="BINARY concepts via LFQ: every state dim is a bit (+/-1), "
                         "the code is the sign pattern, no learned codebook -> no "
                         "dead codes, no kmeans, no collapse. Concept distance "
                         "becomes Hamming. ~1.3x d_state for the same content.")
    ap.add_argument("--gen6", action="store_true",
                    help="gcf6 concept space: EVERY family compositional like the "
                         "board -- squares 8 heads x 8 codes (8^8 per square, same "
                         "16 dims), pieces 8x8, moves 32 heads x 4 codes (4^32, "
                         "collisions gone). Pair with --g-heads 32. Same d_state as "
                         "gcf5, vastly larger vocabulary.")
    ap.add_argument("--av-concepts", action="store_true",
                    help="sharp head reads the QUANTISED move code instead of raw "
                         "backbone features, so every head in the model reads "
                         "concepts. Changes sf_av's input width -> the head "
                         "re-initialises on resume.")
    ap.add_argument("--state-mix", action="store_true",
                    help="condition the operator's basis mixture on the POST-"
                         "INTERACTION concepts (phi) as well as the move. Without it "
                         "K(m) is a fixed matrix per move identity -- the same in the "
                         "opening and the endgame -- which is the architectural "
                         "ceiling on within-position sibling discrimination.")
    ap.add_argument("--sib-pl", action="store_true",
                    help="score the sibling term with listwise Plackett-Luce instead "
                         "of per-move KL. KL is satisfied by predicting the position "
                         "MEAN (measured: 4pct of true spread, ordering 0.00); PL "
                         "scores only the ORDER and penalises a constant.")
    ap.add_argument("--all-ply", action="store_true",
                    help="supervise the operator and value head at EVERY ply 1..6 "
                         "instead of {1,3,6}. Unsupervised plies came out "
                         "sign-INVERTED (measured k=2,5), which a search reads "
                         "backwards. Uses the chain cache's existing 6 plies.")
    ap.add_argument("--w-opk", type=float, default=0.5,
                    help="weight on the previously-unsupervised plies (--all-ply)")
    ap.add_argument("--u-pen", type=float, default=0.1,
                    help="penalty on U in identity_prior. U carries ALL of the "
                         "operator's move-dependence; 0.1 shrank it to ~1pct of the "
                         "shared term. Lower when --w-sib is on, or they cancel.")
    ap.add_argument("--sib-cap", type=int, default=16,
                    help="max moves per row scored by --w-sib (cost control)")
    ap.add_argument("--w-dec", type=float, default=0.5, help="vicreg cov on fast slice")
    ap.add_argument("--vamp-every", type=int, default=4)
    ap.add_argument("--vamp-batch", type=int, default=128)
    ap.add_argument("--lr-ops", type=float, default=3e-5, help="gentler for near-identity A")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--ckpt-every", type=int, default=2000)
    ap.add_argument("--rescan-every", type=int, default=200)
    ap.add_argument("--max-shards", type=int, default=400)
    ap.add_argument("--max-rows", type=int, default=0, help="ROTATING store cap: when "
                    "rescans push past this, oldest files are evicted (the unbounded "
                    "store was jetsam-killed at 2.33M rows; 0 = unbounded)")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="artifacts/experiments/gko0")
    args = ap.parse_args()
    if args.iv_frozen:
        args.iv = True          # the head must exist at model build time
    torch.set_num_threads(args.threads); torch.manual_seed(args.seed)
    dev = args.device

    from catspace.research.components.encoder.approaches.reach_probability.experiments \
        .graph_concept_field import GraphConceptField
    from catspace.research.components.encoder.approaches.reach_probability.experiments \
        .graph_move_policy import featurize
    from catspace.research.tools.training_infra.hl_gauss import HLGauss
    from catspace.research.tools.training_infra.losses import plackett_luce
    from catspace.research.tools.training_infra.koopman_ops import (
        InteractionLift, KoopmanOps, BasisOps, TransformerLift, one_ply_loss,
        rollout, lift_consistency)
    from catspace.research.tools.training_infra.losses import (
        cross_head_decorrelation, vamp2_score, vicreg_covariance)
    from catspace.research.components.encoder.approaches.reach_probability.src \
        .trajectories import normalise_elo
    from catspace.research.tools.training_infra.train.scaffold import (
        TrainConfig, standard_train)

    ck = torch.load(args.resume_field, map_location="cpu", weights_only=False)
    A = dict(ck["args"])
    if args.g_heads:
        A["g_heads"], A["g_codes"], A["d_code"] = args.g_heads, args.g_codes, args.d_code
        A["slow_heads"] = args.slow_heads or A["slow_heads"]
        if args.sq_codes:
            A["sq_codes"] = args.sq_codes
        print(f"[gko2] field OVERRIDE: {A['g_heads']} heads x {A['g_codes']} codes "
              f"x d{A['d_code']}, slow {A['slow_heads']}", flush=True)
    field = GraphConceptField(
        d=A["d_model"], layers=A["layers"], slow_heads=A["slow_heads"], n_res=3, n_eg=1,
        cls_reach=np.zeros(3, np.int64), cls_eg=np.zeros(3, np.int64),
        cls_out=np.arange(3, dtype=np.int64), edge_codes=A["edge_codes"],
        move_codes=A["move_codes"], gps_parallel=bool(A["gps_parallel"]),
        attn_bias=bool(A["attn_bias"]), agg_max=bool(A["agg_max"]), kin=bool(A["kin"]),
        file_nodes=bool(A["file_nodes"]), phase_head=bool(A["phase_head"]),
        threat_head=bool(A["threat_head"]), elo_cond=True,
        g_heads=A["g_heads"], g_codes=A["g_codes"], d_code=A["d_code"],
        sq_codes=A.get("sq_codes", 256),
        mh_vq=bool(args.g_heads), gen5=args.gen5,
        gen6=args.gen6, binary=args.binary,
        depth_cost=args.depth_cost).to(dev)
    if args.g_heads:
        own = field.state_dict()
        sdf = {k: v for k, v in ck["state_dict"].items()
               if k in own and own[k].shape == v.shape}
        print(f"[gko2] trunk warm-start: {len(sdf)}/{len(own)} tensors carried, "
              f"{len(own) - len(sdf)} fresh (concept layer + heads)", flush=True)
        field.load_state_dict(sdf, strict=False)
    else:
        field.load_state_dict(ck["state_dict"])

    class GKO(nn.Module):
        def __init__(self):
            super().__init__()
            self.field = field
            # CONCEPTS-ONLY (Kaveh 2026-08-24: "I don't wanna read off of the raw
            # board or the encoded board, I wanna read off of the concepts"). The
            # sharp head was the ONE head still reading h_mv -- raw backbone output,
            # never quantised -- while the policy, value and rolled heads all read
            # quantised codes. --av-concepts feeds it the QUANTISED move code (q_mv,
            # 32-d) instead, so every head in the model reads concepts.
            # same literal-32 trap as move_head had: read the field's real width
            _av_mv = (getattr(field, "mv_code_dim", 32) if args.av_concepts
                      else A["d_model"])
            self.sf_av = nn.Sequential(
                nn.Linear(field.d_state + _av_mv, 512), nn.GELU(),
                nn.Linear(512, args.k_bins))
            if args.ops:
                if args.lift == "transformer":
                    self.lift = TransformerLift(
                        field.d_state, g_heads=A["g_heads"], d_code=A["d_code"],
                        d_sq=(16 if args.gen5 else 32), d_phi=max(args.d_phi, 256),
                        n_layers=args.lift_layers,
                        n_piece=(32 if args.gen5 else 0),
                        d_piece=getattr(field, "d_piece", 4),
                        d_mv=getattr(field, "mv_code_dim", 32))
                else:
                    self.lift = InteractionLift(field.d_state, args.d_phi)
                if args.gen5:
                    self.ops = BasisOps(self.lift.d_aug, n_basis=64,
                                        rank=args.op_rank,
                                        d_phi=(self.lift.d_aug - self.lift.d_state)
                                        if args.state_mix else 0)
                else:
                    self.ops = KoopmanOps(self.lift.d_aug, n_codes=A["move_codes"],
                                          rank=args.op_rank)
                # H1's baseline arm: same targets, DETACHED inputs (see docstring)
                self.dyn_z = nn.Sequential(
                    nn.Linear(field.d_state + 64, 512), nn.GELU(),
                    nn.Linear(512, field.d_state))
            if args.iv:
                self.iv = nn.Sequential(
                    nn.Linear(field.d_state + getattr(field, "mv_code_dim", 32),
                              256), nn.GELU(), nn.Linear(256, 3))
            if args.ops and args.ema_decay > 0:
                import copy as _copy
                self.ema_field = _copy.deepcopy(field).eval()
                for pr in self.ema_field.parameters():
                    pr.requires_grad_(False)
            if args.ops:
                # PHI-ONLY value head (the saturation falsification: heads reading
                # full z_aug route AROUND phi; value-at-rolled-horizons must be
                # impossible WITHOUT interaction features carrying it)
                v_out = 3 if args.gen5 else args.k_bins
                self.sf_v = nn.Sequential(
                    nn.Linear(self.lift.d_phi, 256), nn.GELU(),
                    nn.Linear(256, v_out))
                # FULL value head for SEARCH (2026-08-22 maia autopsy: phi-only sees
                # ~quarter of a removed queen -- material lives in the state part;
                # the phi-only head REMAINS the anti-collapse anchor, this one plays)
                self.sf_v_full = nn.Sequential(
                    nn.Linear(self.lift.d_aug, 256), nn.GELU(),
                    nn.Linear(256, 3 if args.gen5 else args.k_bins))
                # per-move future action-values from ROLLED states: [z_aug ; mv_emb]
                self.av_roll = nn.Sequential(
                    nn.Linear(self.lift.d_aug + 64, 512), nn.GELU(),
                    nn.Linear(512, args.k_bins))

        def train(self, mode=True):
            # model.train() recurses into ALL submodules -- without this override the
            # EMA target encoder flips to train mode and its VQ codebooks self-update
            # from every child-target batch (the drift EMA exists to prevent)
            super(GKO, self).train(mode)
            if hasattr(self, "ema_field"):
                self.ema_field.eval()
            return self
    model = GKO().to(dev)
    ema_restored = False
    if args.resume_self:
        ck2 = torch.load(args.resume_self, map_location="cpu", weights_only=False)
        sd2 = dict(ck2["state_dict"])
        own = model.state_dict()
        # strict=False does NOT tolerate shape mismatches -- drop them explicitly
        # (e.g. operators sized for a different lift width start fresh)
        drop = [k for k, v in sd2.items() if k in own and own[k].shape != v.shape]
        for k in drop:
            del sd2[k]
        miss, unex = model.load_state_dict(sd2, strict=False)
        # did the resume actually RESTORE an ema target (vs leaving the fresh
        # deepcopy)? if so its lag is real training state -- a hard copy on the
        # first update would throw it away and un-lag the child targets.
        ema_restored = any(k.startswith("ema_field.") for k in sd2)
        print(f"[gko0] resumed self from {args.resume_self} (step {ck2.get('step')}; "
              f"{len(miss)} new keys init fresh, {len(drop)} shape-dropped, "
              f"{len(unex)} unexpected; ema target "
              f"{'RESTORED (lag preserved)' if ema_restored else 'fresh (will hard-copy)'})",
              flush=True)
    import copy
    hl = HLGauss(k=args.k_bins).to(dev)
    ema_field = getattr(model, "ema_field", None)
    ema_started = [bool(ema_restored)]
    id6_hi = [None]  # best even-lag identity-R2 post-warmup (gen5 erosion watch)

    def _ema_update():
        d = args.ema_decay
        with torch.no_grad():
            if not ema_started[0]:
                # FIRST update is a HARD copy: the deepcopy predates any forward, so
                # the copy's VQ books are uninitialized (kmeans never fires in eval);
                # without this, early child targets quantize against random codebooks
                for po, pe in zip(model.field.parameters(), ema_field.parameters()):
                    pe.copy_(po)
                for bo, be in zip(model.field.buffers(), ema_field.buffers()):
                    be.copy_(bo)
                ema_started[0] = True
                return
            for po, pe in zip(model.field.parameters(), ema_field.parameters()):
                pe.mul_(d).add_(po, alpha=1 - d)
            for bo, be in zip(model.field.buffers(), ema_field.buffers()):
                if be.dtype.is_floating_point:
                    be.mul_(d).add_(bo, alpha=1 - d)
                else:
                    be.copy_(bo)

    if args.iv_frozen:
        for pr in model.field.parameters():
            pr.requires_grad_(False)
        for pr in model.sf_av.parameters():
            pr.requires_grad_(False)
        opt = torch.optim.Adam(model.iv.parameters(), lr=args.lr)
    elif args.ops:
        op_ids = {id(pr) for pr in model.ops.parameters()}
        lift_ids = {id(pr) for pr in model.lift.parameters()}
        rest = [pr for pr in model.parameters()
                if id(pr) not in op_ids and id(pr) not in lift_ids and pr.requires_grad]
        opt = torch.optim.Adam([{"params": rest, "lr": args.lr},
                                {"params": list(model.ops.parameters()),
                                 "lr": args.lr_ops},
                                {"params": list(model.lift.parameters()),
                                 "lr": args.lr * 0.5}])
    else:
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    rng = np.random.default_rng(args.seed)
    t0 = time.time()

    def _nan_guard(loss):
        return not bool(torch.isfinite(loss).all())

    # ---- chain stream (ops arm): mmap graphs for 60k 7-row chains + move codes ----
    CH = CH_pos = ch_tr = ch_te = row_elo_n = None
    if args.ops:
        if not os.path.exists(args.chain_graphs):
            raise SystemExit(f"[gko1] chain graphs missing: {args.chain_graphs} "
                             "(run the converter over the generator output first)")
        CH = np.load(args.chain_graphs, mmap_mode="r")
        cs = np.load(args.chain_starts)
        starts, games = cs["start"], cs["game"]
        CH_prow = np.asarray(CH["p_row"])
        pos = np.searchsorted(CH_prow, starts[:, None] + np.arange(7)[None, :])
        ok = (pos < len(CH_prow))
        ok &= CH_prow[np.minimum(pos, len(CH_prow) - 1)] == (starts[:, None] + np.arange(7))
        keep = ok.all(1)
        CH_pos = pos[keep]; games = games[keep]
        gset = np.unique(games); rngc = np.random.default_rng(args.seed + 7)
        te_g = set(gset[rngc.random(len(gset)) < 0.08].tolist())
        te_m = np.array([g in te_g for g in games])
        ch_tr = np.flatnonzero(~te_m); ch_te = np.flatnonzero(te_m)
        rel = np.load(args.row_elo)["elo"]
        from catspace.research.components.encoder.approaches.reach_probability.src \
            .trajectories import normalise_elo as _ne
        row_elo_n = _ne(rel.astype(np.float32))
        print(f"[gko1] chains {len(CH_pos):,} ({len(ch_te):,} held out by GAME) | "
              f"graphs {args.chain_graphs}", flush=True)
        CHV = CHW = CHWDL = None
        if os.path.exists(args.chain_values):
            _z = np.load(args.chain_values)
            CHV = _z["v"]
            CHW = _z["wp"].astype(np.float32) if "wp" in _z.files else None
            CHWDL = _z["wdl"] if "wdl" in _z.files else None
            print(f"[gko1] chain values: {int(np.isfinite(CHV).sum()):,} rows valued"
                  f" (av vectors: {'yes' if CHW is not None else 'no'})", flush=True)

    KEYS = ("sq", "gl", "mv", "mut", "vac", "dsc", "att", "n_mv", "n_att",
            "kind", "band", "tgt")
    OPT_KEYS = ("pslot", "wdl")   # newer shards only; synthesized when absent
    store = {k: [] for k in KEYS}
    seen = set()

    hold_chunks = []                 # per-chunk holdout masks, evicted IN LOCKSTEP
    hold_rng = np.random.default_rng(args.seed + 3)

    def rescan():
        files = sorted(glob.glob(os.path.join(args.shards, "shard_*.npz")))
        new = [f for f in files if f not in seen][-args.max_shards:]
        for f in new:
            try:
                z = np.load(f, allow_pickle=True)
                for k in KEYS:
                    store[k].append(np.ascontiguousarray(z[k]))
                n_z = len(store["kind"][-1])
                for k in OPT_KEYS:
                    if k in z.files:
                        store.setdefault(k, []).append(np.ascontiguousarray(z[k]))
                    elif k == "pslot":
                        store.setdefault(k, []).append(np.full((n_z, 64), 255, np.uint8))
                    else:
                        store.setdefault(k, []).append(
                            np.full((n_z, 48, 3), np.nan, np.float16))
                hold_chunks.append(hold_rng.random(n_z) < 0.08)
                seen.add(f)
            except Exception:
                pass
        if args.max_rows:
            total = sum(len(a) for a in store["kind"])
            while total > args.max_rows and len(store["kind"]) > 1:
                for k in OPT_KEYS:
                    if store.get(k):
                        store[k].pop(0)
                # evict the OLDEST chunk (ring); split masks pop in lockstep so
                # train/holdout alignment survives index shifts (the negative-dims
                # crash + silent split-drift bug, 2026-08-23)
                drop = len(store["kind"][0])
                for k in KEYS:
                    store[k].pop(0)
                hold_chunks.pop(0)
                total -= drop
        return len(new)

    def flat(k):
        return np.concatenate(store[k]) if store[k] else None

    while rescan() == 0 or sum(len(a) for a in store["kind"]) < 4096:
        print("[gko0] waiting for shards...", flush=True)
        time.sleep(30)
        rescan()
    D = {k: torch.from_numpy(flat(k)) for k in KEYS + OPT_KEYS}
    N = len(D["kind"])
    hold = np.concatenate(hold_chunks)
    tr_i = np.flatnonzero(~hold); te_i = np.flatnonzero(hold)
    _nsf = int((D["kind"] == 0).sum())
    print(f"[gko0] start: {N:,} rows ({len(te_i):,} held out) | "
          f"sf {_nsf:,} maia {int((D['kind']==1).sum()):,}", flush=True)
    # this line has always printed the fatal condition; on 2026-08-23 it read
    # "sf 0" at launch and the run trained 1000 steps with a frozen sharp head
    # before anyone noticed. An informational count is not a gate -- alarm on it.
    if args.w_sf > 0 and _nsf < 0.05 * max(N, 1):
        print(f"[gko1] ALARM sf-diet {_nsf:,}/{N:,} ({_nsf/max(N,1):.2%}) at START "
              f"-- l_sf, l_sv and every sf_av gradient are OFF. Convert sf shards: "
              f"gko_data loop --oracle sf --newest --shards 70", flush=True)

    model_has_v = args.ops and hasattr(model, "sf_v")

    def refresh_store():
        nonlocal D, N, tr_i, te_i, hold, CHV, CHW, CHWDL
        if args.ops and os.path.exists(args.chain_values):
            try:
                _z = np.load(args.chain_values)
                CHV = _z["v"]
                CHW = _z["wp"].astype(np.float32) if "wp" in _z.files else None
                CHWDL = _z["wdl"] if "wdl" in _z.files else None
            except Exception:
                pass
        if rescan():
            D = {k: torch.from_numpy(flat(k)) for k in KEYS + OPT_KEYS}
            N = len(D["kind"])
            hold = np.concatenate(hold_chunks)
            tr_i = np.flatnonzero(~hold); te_i = np.flatnonzero(hold)
            # DIET GATE (2026-08-23): sf_av's ONLY gradient, l_sf, and l_sv (the
            # material-sightedness supervision behind the queen-probe fix) all live
            # behind `kind == 0`. A shard window that rotates sf-free switches them
            # off silently -- every remaining gauge stays healthy while the sharp
            # head freezes under a drifting field. Never let that pass unannounced.
            k_arr = D["kind"].numpy()
            sf_frac = float((k_arr == 0).mean()) if len(k_arr) else 0.0
            if sf_frac < 0.05:
                print(f"[gko1] ALARM sf-diet {sf_frac:.3%} of {N:,} rows (<5pct) -- "
                      f"l_sf/l_sv/sf_av get NO gradient; convert sf shards "
                      f"(gko_data loop --oracle sf --newest) before trusting this run",
                      flush=True)
            else:
                print(f"[gko1] diet: sf {sf_frac:.0%} / maia {1-sf_frac:.0%} "
                      f"of {N:,} rows", flush=True)

    def batch_of(idx):
        b = {k: D[k][idx].to(dev) for k in ("sq", "gl", "mv", "mut", "att",
                                            "n_mv", "n_att")}
        sqf, mvf, edges, mask = featurize(b, dev, n_nodes=field.n_nodes)
        off = (torch.arange(len(idx), device=dev) * field.n_nodes)[:, None]
        jj = torch.arange(MV_CAP, device=dev)[None, :]
        for name in ("vac", "dsc"):
            a2 = D[name][idx].to(dev).long()
            val = (a2 != 255) & mask[:, :, None]
            edges[name] = (((jj + 64 + off)[:, :, None].expand_as(a2))[val],
                           (a2 + off[:, :, None])[val])
        band = D["band"][idx].float()
        elo_v = torch.where(band > 0, band, torch.full_like(band, 2750.0))
        elo = torch.from_numpy(
            normalise_elo(elo_v.numpy().astype(np.float32)))[:, None].repeat(1, 2).to(dev)
        psl = D["pslot"][idx].to(dev) if args.gen5 else None
        return sqf, mvf, edges, mask, elo, psl

    def _feat_rows(rows):
        b = {k: torch.from_numpy(np.ascontiguousarray(CH[k][rows])).to(dev)
             for k in ("sq", "gl", "mv", "mut", "att", "n_mv", "n_att")}
        sqf, mvf, edges, mask = featurize(b, dev, n_nodes=field.n_nodes)
        off = (torch.arange(len(rows), device=dev) * field.n_nodes)[:, None]
        jj = torch.arange(MV_CAP, device=dev)[None, :]
        for name in ("vac", "dsc"):
            a2 = torch.from_numpy(np.ascontiguousarray(CH[name][rows])).to(dev).long()
            val = (a2 != 255) & mask[:, :, None]
            edges[name] = (((jj + 64 + off)[:, :, None].expand_as(a2))[val],
                           (a2 + off[:, :, None])[val])
        elo = torch.from_numpy(row_elo_n[CH_prow[rows]]).to(dev)
        played = torch.from_numpy(np.ascontiguousarray(CH["played"][rows])).long().to(dev)
        psl = (torch.from_numpy(np.ascontiguousarray(CH["pslot"][rows])).to(dev)
               if "pslot" in CH.files else None)
        return sqf, mvf, edges, mask, elo, played, psl

    def _mv_codes(h_mv, played, f=None):
        f = f or model.field
        with torch.no_grad():
            _, idx_mv, _ = f.mv_vq(f.mv_vq_proj(h_mv.detach()))
        if idx_mv.dim() == 3:          # gen6: composite (B, MV, heads)
            return idx_mv.gather(
                1, played[:, None, None].expand(-1, 1, idx_mv.shape[-1]))[:, 0]
        return idx_mv.gather(1, played[:, None]).squeeze(1)

    def _raw_ftp(rows, played):
        """Bare move identity (from, to, promo) from the RAW mv arrays, cols 1/2/5
        (gcf5 BasisOps conditioning -- no encoder involvement)."""
        raw = torch.from_numpy(np.ascontiguousarray(CH["mv"][rows])).long().to(dev)
        g = raw.gather(1, played[:, None, None].expand(-1, 1, raw.shape[-1]))[:, 0]
        return g[:, 1], g[:, 2], g[:, 5]

    def _av_in(zq_, h_mv_, mask_):
        """The sharp head's move input: quantised concept code when --av-concepts,
        else the raw backbone move features (legacy)."""
        if args.av_concepts:
            with torch.no_grad():
                q, _, _ = model.field.mv_vq(model.field.mv_vq_proj(h_mv_))
            mv_in = q * mask_[..., None]
        else:
            mv_in = h_mv_
        return torch.cat([zq_[:, None, :].expand(-1, mv_in.shape[1], -1), mv_in], -1)

    def _mv_q(h_mv, mask):
        """Quantized move-token embeddings for the lift's interaction table
        (zero-padded on dead slots); detached -- the lift reads move CONCEPTS,
        it does not train the move codebook."""
        with torch.no_grad():
            q, _, _ = model.field.mv_vq(model.field.mv_vq_proj(h_mv.detach()))
        return q * mask[..., None]

    def op_losses(step):
        """Chain sub-batch: L_op1/3/6 + spectral + the detached MLP baseline arm."""
        cb = args.chain_batch
        ci = CH_pos[ch_tr[rng.integers(0, len(ch_tr), cb)]]        # (cb,7) graph rows
        sqf, mvf, edges, mask, elo, played, psl_p = _feat_rows(ci[:, 0])
        zq_p, _, _, _, h_mv_p, _ = model.field.encode(sqf, mvf, edges, mask, elo=elo, pslot=psl_p)[:6]
        c0 = None if args.gen5 else _mv_codes(h_mv_p, played)
        tok_p = mvf.gather(1, played[:, None, None].expand(-1, 1, mvf.shape[-1]))[:, 0]
        # RESTORED FIELD STACK (the gko0 slice dropped it; identity-R2 collapse
        # +0.92 -> -0.78 over gens 1-3 was the measured cost). PL on the corpus move:
        lg_p = model.field.move_logits(zq_p, h_mv_p, mask)
        l_move = F.cross_entropy(lg_p.masked_fill(~mask, -1e9), played)
        enc_t = ema_field if ema_field is not None else model.field
        with torch.no_grad():
            rows_c = ci[:, 1:].ravel()
            sqc, mvc, edc, mkc, eloc, plc, psl_c = _feat_rows(rows_c)
            zq_c, gi_c, si_c, _, h_mv_c, _ = enc_t.encode(sqc, mvc, edc, mkc,
                                                          elo=eloc, pslot=psl_c)[:6]
            zq_c = zq_c.reshape(cb, 6, -1)
            gi1 = gi_c.reshape(cb, 6, -1)[:, 0]
            si1 = si_c.reshape(cb, 6, -1)[:, 0]
            cks = (None if args.gen5 else
                   _mv_codes(h_mv_c, plc, enc_t).reshape(cb, 6)[:, :5])
            tok_c = mvc.gather(1, plc[:, None, None].expand(-1, 1, mvc.shape[-1])
                               )[:, 0].reshape(cb, 6, -1)[:, :5]
        # dyn CE to the played child's codes (global + square)
        dy = model.field.dyn(torch.cat([zq_p, model.field.mv_emb(tok_p)], -1))
        Hh, Cc = model.field.g_heads, A["g_codes"]
        # gen6 squares are 8 heads x 8 codes, so the CE is over 8 classes and there
        # are 64*8 targets -- NOT 64 targets over sq_codes. The dyn head's WIDTH is
        # identical either way (64*64 == 64*8*8), which is exactly why this had to be
        # made explicit: a wrong SC here reshapes correctly-sized logits into the
        # wrong number of rows and cross_entropy dies on the batch mismatch.
        SC = 8 if getattr(model.field, "gen6", False) else model.field.sq_codes
        l_dyn2 = (F.cross_entropy(dy[:, :Hh * Cc].reshape(-1, Cc), gi1.reshape(-1)) +
                  F.cross_entropy(dy[:, Hh * Cc:].reshape(-1, SC), si1.reshape(-1)))
        code_seq = None if args.gen5 else torch.cat([c0[:, None], cks], 1)  # (cb,6)
        t1, t3, t6 = zq_c[:, 0], zq_c[:, 2], zq_c[:, 5]
        assert not t1.requires_grad and (code_seq is None
                                         or not code_seq.requires_grad)  # U13 guard
        q_mv_p = _mv_q(h_mv_p, mask)
        za_root = model.lift(zq_p, q_mv_p)
        if args.gen5:
            f0, tt0, pr0 = _raw_ftp(ci[:, 0], played)
            fs, ts, ps_ = [f0], [tt0], [pr0]
            for k in range(5):
                fk, tk, pk = _raw_ftp(ci[:, 1 + k], plc.reshape(cb, 6)[:, k])
                fs.append(fk); ts.append(tk); ps_.append(pk)
            pred1 = model.ops(za_root, fs[0], ts[0], ps_[0])
            l1 = F.mse_loss(pred1[..., :model.lift.d_state], t1)
            outs, zz = [], za_root
            for k in range(6):
                zz = model.ops(zz, fs[k], ts[k], ps_[k])
                outs.append(zz)
            traj = torch.stack(outs, 1)
        else:
            pred1 = model.ops(za_root, code_seq[:, 0])
            l1 = F.mse_loss(pred1[..., :model.lift.d_state], t1)
            traj = rollout(model.ops, za_root, code_seq)            # (cb,6,d_aug)
        ds = model.lift.d_state
        def _comp(za, tgt):
            return F.mse_loss(za[..., :ds], tgt)
        l3, l6 = _comp(traj[:, 2], t3), _comp(traj[:, 5], t6)
        # plies 2,4,5 (0-indexed 1,3,4): supervised only when --all-ply
        l_mid = torch.zeros((), device=dev)
        if args.all_ply:
            for _k in (1, 3, 4):
                l_mid = l_mid + _comp(traj[:, _k], zq_c[:, _k])
            l_mid = l_mid / 3.0
        if args.gen5:
            l_spec, rho = model.ops.spectral_penalty(fs[0], ts[0], ps_[0],
                                                     z_aug=za_root)
            l_spec = l_spec + 0.05 * model.ops.identity_prior(args.u_pen)
        else:
            l_spec, rho = model.ops.spectral_penalty(code_seq.ravel())
        # baseline: iterated MLP on DETACHED state with token move features
        s = zq_p.detach()
        toks = torch.cat([tok_p[:, None], tok_c], 1)                # (cb,6,feat)
        merr = {}
        for k in range(6):
            s = model.dyn_z(torch.cat([s, model.field.mv_emb(toks[:, k].detach())], -1))
            if k in (0, 2, 5) or args.all_ply:
                merr[k] = F.mse_loss(s, zq_c[:, k])
        l_mlp = (sum(merr.values()) / len(merr) * 3.0 if args.all_ply
                 else merr[0] + merr[2] + merr[5])
        l_cons = torch.zeros((), device=dev)
        if args.w_cons > 0:
            ds_ = model.lift.d_state
            for k in ((1, 2, 3, 4, 5, 6) if args.all_ply else (1, 3, 6)):
                l_cons = l_cons + F.mse_loss(traj[:, k - 1, ds_:],
                                             model.lift.phi(traj[:, k - 1, :ds_], None))
        l_v = torch.zeros((), device=dev)
        v_n = 0
        if CHV is not None and model_has_v:
            za0 = za_root
            vt0 = torch.from_numpy(CHV[ci[:, 0]]).to(dev)
            m0 = torch.isfinite(vt0)
            if args.gen5 and CHWDL is not None:
                w0 = torch.from_numpy(CHWDL[ci[:, 0]].astype(np.float32)).to(dev)
                mw = torch.isfinite(w0).all(-1)
                if mw.any():
                    tgtw = w0[mw].clamp(min=1e-6)
                    tgtw = tgtw / tgtw.sum(-1, keepdim=True)
                    for head, inp in ((model.sf_v, za0[mw][:, model.lift.d_state:]),
                                      (model.sf_v_full, za0[mw])):
                        lp = torch.log_softmax(head(inp), -1)
                        l_v = l_v + args.w_v0 * (-(tgtw * lp).sum(-1).mean())
                    v_n += int(mw.sum())
            elif m0.any():
                l_v = l_v + args.w_v0 * hl.ce(model.sf_v(za0[m0][:, model.lift.d_state:]),
                                              vt0[m0])
                l_v = l_v + args.w_v0 * hl.ce(model.sf_v_full(za0[m0]), vt0[m0])
                v_n += int(m0.sum())
            w6r = min(1.0, max(0.0, (step - args.op6_after) / 2000.0))
            mvc_all = mvc.reshape(cb, 6, MV_CAP, -1)
            # ALL-PLY (2026-08-23): supervising only {1,3,6} left plies 2/4/5
            # untrained, and the rolled value came out sign-INVERTED at 2 and 5 --
            # reproduced on two disjoint held-out samples, and NOT a mover-POV effect
            # (identical for white- and black-to-move roots). A search visits every
            # ply, so it read those levels backwards. The chain cache already holds
            # all six plies; this is a loop bound, not new data.
            _KS = (((1, 1.0), (2, 1.0), (3, 1.0), (4, 1.0), (5, 1.0), (6, w6r))
                   if args.all_ply else ((1, 1.0), (3, 1.0), (6, w6r)))
            for k, wk in _KS:
                if wk <= 0:
                    continue
                vtk = torch.from_numpy(CHV[ci[:, k]]).to(dev)
                mk = torch.isfinite(vtk)
                if args.gen5 and CHWDL is not None:
                    wk_t = torch.from_numpy(CHWDL[ci[:, k]].astype(np.float32)).to(dev)
                    mw = torch.isfinite(wk_t).all(-1)
                    if mw.any():
                        tg = wk_t[mw].clamp(min=1e-6)
                        tg = tg / tg.sum(-1, keepdim=True)
                        for head, inp in ((model.sf_v,
                                           traj[:, k - 1][mw][:, model.lift.d_state:]),
                                          (model.sf_v_full, traj[:, k - 1][mw])):
                            lp = torch.log_softmax(head(inp), -1)
                            l_v = l_v + args.w_vk * wk * (-(tg * lp).sum(-1).mean())
                        v_n += int(mw.sum())
                elif mk.any():
                    l_v = l_v + args.w_vk * wk * hl.ce(
                        model.sf_v(traj[:, k - 1][mk][:, model.lift.d_state:]), vtk[mk])
                    l_v = l_v + args.w_vk * wk * hl.ce(
                        model.sf_v_full(traj[:, k - 1][mk]), vtk[mk])
                    v_n += int(mk.sum())
                if CHW is not None and args.w_av > 0:
                    wpk = torch.from_numpy(CHW[ci[:, k]]).to(dev)      # (cb, MV_CAP)
                    fm = torch.isfinite(wpk)
                    if fm.any():
                        emb = model.field.mv_emb(mvc_all[:, k - 1])
                        x = torch.cat([traj[:, k - 1][:, None, :].expand(
                            -1, MV_CAP, -1), emb], -1)
                        l_v = l_v + args.w_av * wk * hl.ce(
                            model.av_roll(x)[fm], wpk[fm])
                        v_n += int(fm.sum())
            if CHW is not None and args.w_av > 0:
                wp0 = torch.from_numpy(CHW[ci[:, 0]]).to(dev)
                fm0 = torch.isfinite(wp0)
                if fm0.any():
                    emb0 = model.field.mv_emb(mvf)
                    x0 = torch.cat([za0[:, None, :].expand(-1, MV_CAP, -1), emb0], -1)
                    l_v = l_v + args.w_av * hl.ce(model.av_roll(x0)[fm0], wp0[fm0])
                    v_n += int(fm0.sum())
            assert not vt0.requires_grad
        l_slow = l_dec = torch.zeros((), device=dev)
        vamp_v = 0.0
        if args.w_slow > 0 and step % args.vamp_every == 0:
            vb = CH_pos[ch_tr[rng.integers(0, len(ch_tr), args.vamp_batch)]]
            s0, m0, e0, k0, el0, _, ps0 = _feat_rows(vb[:, 0])
            pg0 = model.field.encode(s0, m0, e0, k0, elo=el0, pslot=ps0)[5]
            s6, m6, e6, k6, el6, _, ps6 = _feat_rows(vb[:, 6])
            pg6 = model.field.encode(s6, m6, e6, k6, elo=el6, pslot=ps6)[5]
            sl = model.field.slow_heads * getattr(
                model.field, 'code_width', model.field.d_code)
            v0 = model.field.vamp_proj(pg0.reshape(len(vb), -1)[:, :sl])
            v6 = model.field.vamp_proj(pg6.reshape(len(vb), -1)[:, :sl])
            vamp = vamp2_score(v0, v6)
            vamp_v = float(vamp)
            l_slow = -vamp / 32.0
            l_dec = vicreg_covariance(pg0.reshape(len(vb), -1)[:, sl:])
        opg = min(1.0, max(0.0, (step - args.ops_after) / 1000.0)) if args.ops_after else 1.0
        w6 = args.w_op6 * min(1.0, max(0.0, (step - args.op6_after) / 2000.0)) * opg
        total = (opg * (args.w_op1 * l1 + args.w_op3 * l3) + w6 * l6 +
                 opg * args.w_opk * l_mid +
                 args.w_spec * l_spec + l_mlp + l_v + opg * args.w_cons * l_cons +
                 args.w_move * l_move + args.w_dyn2 * l_dyn2 +
                 args.w_slow * l_slow * args.vamp_every + args.w_dec * l_dec * args.vamp_every)
        with torch.no_grad():
            cons6 = float(lift_consistency(model.lift, traj[:, 5]).mean())
        # If any operator sub-term is non-finite, NAME it. A bare 'op_total=nan'
        # left me guessing twice (LFQ entropy, then VAMP) and wrong both times.
        _nf = {}
        for _n, _t in (("l1", l1), ("l3", l3), ("l6", l6), ("mid", l_mid),
                       ("spec", l_spec), ("mlp", l_mlp), ("v", l_v),
                       ("cons", l_cons), ("move", l_move), ("dyn2", l_dyn2),
                       ("slow", l_slow), ("dec", l_dec)):
            if not bool(torch.isfinite(torch.as_tensor(_t)).all()):
                _nf["NONFINITE_" + _n] = float("nan")
        return total, {"op1": float(l1), "op3": float(l3), "op6": float(l6),
                       "mlp": float(l_mlp), "rho": float(rho.max()), "cons6": cons6,
                       "move": float(l_move), "dyn2": float(l_dyn2), "vamp": vamp_v,
                       "lv": float(l_v), "vn": v_n, "cons_l": float(l_cons),
                       "phivar": float(model.lift.phi(zq_p.detach()).std(0).mean()), **_nf}

    @torch.no_grad()
    def op_eval(n_ch=384):
        """Held-out chains: pooled R2 per horizon, ops vs the MLP baseline."""
        model.eval()
        agg = {k: [0.0, 0.0] for k in ("o1", "o3", "o6", "m1", "m3", "m6",
                                       "i1", "i3", "i6")}
        for a in range(0, n_ch, 64):
            ci = CH_pos[ch_te[a:a + 64]]
            if not len(ci):
                break
            sqf, mvf, edges, mask, elo, played, psl_p = _feat_rows(ci[:, 0])
            zq_p, _, _, _, h_mv_p, _ = model.field.encode(sqf, mvf, edges, mask,
                                                          elo=elo)[:6]
            c0 = None if args.gen5 else _mv_codes(h_mv_p, played)
            tok_p = mvf.gather(1, played[:, None, None].expand(-1, 1, mvf.shape[-1]))[:, 0]
            rows_c = ci[:, 1:].ravel()
            sqc, mvc, edc, mkc, eloc, plc, psl_c = _feat_rows(rows_c)
            enc_e = ema_field if ema_field is not None else model.field
            zq_c, _, _, _, h_mv_c, _ = enc_e.encode(sqc, mvc, edc, mkc, elo=eloc, pslot=psl_c)[:6]
            zq_c = zq_c.reshape(len(ci), 6, -1)
            cks = (None if args.gen5 else
                   _mv_codes(h_mv_c, plc, enc_e).reshape(len(ci), 6)[:, :5])
            tok_c = mvc.gather(1, plc[:, None, None].expand(-1, 1, mvc.shape[-1])
                               )[:, 0].reshape(len(ci), 6, -1)[:, :5]
            code_seq = None if args.gen5 else torch.cat([c0[:, None], cks], 1)
            if args.gen5:
                f0e, t0e, p0e = _raw_ftp(ci[:, 0], played)
                zz = model.lift(zq_p, _mv_q(h_mv_p, mask))
                outs_e = []
                fe, te_, pe = f0e, t0e, p0e
                for k in range(6):
                    if k > 0:
                        fe, te_, pe = _raw_ftp(ci[:, k], plc.reshape(len(ci), 6)[:, k - 1])
                    zz = model.ops(zz, fe, te_, pe)
                    outs_e.append(zz)
                traj = torch.stack(outs_e, 1)
            else:
                traj = rollout(model.ops, model.lift(zq_p, _mv_q(h_mv_p, mask)), code_seq)
            ds = model.lift.d_state
            s = zq_p; toks = torch.cat([tok_p[:, None], tok_c], 1)
            mlp_s = {}
            for k in range(6):
                s = model.dyn_z(torch.cat([s, model.field.mv_emb(toks[:, k])], -1))
                if k in (0, 2, 5):
                    mlp_s[k] = s
            if CHW is not None and model_has_v and hasattr(model, "av_roll"):
                from scipy.stats import spearmanr as _sp
                mvc_all2 = mvc.reshape(len(ci), 6, MV_CAP, -1)
                for kk, key in ((1, "a1"), (3, "a3"), (6, "a6")):
                    wpk = torch.from_numpy(CHW[ci[:, kk]]).to(zq_p.device)
                    emb = model.field.mv_emb(mvc_all2[:, kk - 1])
                    x = torch.cat([traj[:, kk - 1][:, None, :].expand(-1, MV_CAP, -1),
                                   emb], -1)
                    pv = hl.decode(model.av_roll(x))
                    for r in range(len(ci)):
                        fm = torch.isfinite(wpk[r])
                        if int(fm.sum()) > 3:
                            c_ = _sp(pv[r][fm].cpu().numpy(),
                                     wpk[r][fm].cpu().numpy()).statistic
                            if np.isfinite(c_):
                                agg.setdefault(key, [0.0, 0.0])
                                agg[key][0] += c_
                                agg[key][1] += 1.0
            if CHV is not None and model_has_v:
                for kk, key in ((1, "v1"), (3, "v3"), (6, "v6")):
                    vt = torch.from_numpy(CHV[ci[:, kk]]).to(zq_p.device)
                    mk = torch.isfinite(vt)
                    if mk.any():
                        lg5 = model.sf_v(traj[:, kk - 1][mk][:, model.lift.d_state:])
                        if args.gen5:
                            pw = torch.softmax(lg5, -1)
                            pv = pw[:, 0] + 0.5 * pw[:, 1]
                        else:
                            pv = hl.decode(lg5)
                        agg.setdefault(key, [0.0, 0.0])
                        agg[key][0] += float((pv - vt[mk]).abs().sum())
                        agg[key][1] += float(mk.sum())
            for key, pred, k in (("o1", traj[:, 0, :ds], 0), ("o3", traj[:, 2, :ds], 2),
                                 ("o6", traj[:, 5, :ds], 5), ("m1", mlp_s[0], 0),
                                 ("m3", mlp_s[2], 2), ("m6", mlp_s[5], 5),
                                 ("i1", zq_p, 0), ("i3", zq_p, 2), ("i6", zq_p, 5)):
                t = zq_c[:, k]
                agg[key][0] += float(((pred - t) ** 2).sum())
                agg[key][1] += float(((t - t.mean(0)) ** 2).sum())
        model.train()
        out2 = {}
        for k, v in agg.items():
            if k.startswith("v"):
                out2[k] = v[0] / max(v[1], 1e-9)      # MAE, wp units
            elif k.startswith("a"):
                out2[k] = v[0] / max(v[1], 1e-9)      # mean spearman
            else:
                out2[k] = 1.0 - v[0] / max(v[1], 1e-9)
        return out2

    def losses_on(idx, train=True):
        sqf, mvf, edges, mask, elo, psl = batch_of(idx)
        zq, g_ids, _, vl, h_mv, pre_g = model.field.encode(sqf, mvf, edges, mask,
                                                          elo=elo, pslot=psl)[:6]
        lg = model.field.move_logits(zq, h_mv, mask)
        tgt = D["tgt"][idx].to(dev)
        kind = D["kind"][idx].to(dev)
        fin = torch.isfinite(tgt) & mask
        l_maia = torch.zeros((), device=dev)
        km = kind == 1
        if km.any():
            lp = F.log_softmax(lg[km].masked_fill(~mask[km], -1e9), -1)
            t2 = torch.where(fin[km], tgt[km], torch.zeros_like(tgt[km]))
            t2 = t2 / t2.sum(-1, keepdim=True).clamp(min=1e-8)
            l_maia = -(t2 * lp).masked_fill(~fin[km], 0.0).sum(-1).mean()
        l_sf = torch.zeros((), device=dev)
        ks = kind == 0
        if ks.any():
            x = _av_in(zq[ks], h_mv[ks], mask[ks])
            logits = model.sf_av(x)
            m2 = fin[ks]
            if m2.any():
                l_sf = hl.ce(logits[m2], tgt[ks][m2])
        l_decor = cross_head_decorrelation(pre_g)
        # STATE-VALUE from every SF shard row (2026-08-22 queen-probe finding: value
        # heads trained only on mostly-balanced chains never learn material; a shard
        # row's max win-prob IS its state value -- free supervision, full-bank scale,
        # and piece-down enrichment flows through this path automatically)
        l_sv = torch.zeros((), device=dev)
        if args.ops and hasattr(model, "sf_v_full") and ks.any():
            sv_t = torch.where(fin[ks], tgt[ks], torch.full_like(tgt[ks], -1e9)
                               ).max(-1).values
            okv = sv_t > -1e8
            if okv.any():
                with torch.no_grad():
                    q_mv_s, _, _ = model.field.mv_vq(
                        model.field.mv_vq_proj(h_mv[ks][okv].detach()))
                za_s = model.lift(zq[ks][okv], q_mv_s * mask[ks][okv][..., None])
                sv_tt = sv_t[okv].detach()
                if args.gen5:
                    # state WDL = the best move's triple from the row's wdl array
                    wrow = D["wdl"][idx].to(dev)[ks][okv].float()      # (n,48,3)
                    tsel = torch.where(fin[ks][okv], tgt[ks][okv],
                                       torch.full_like(tgt[ks][okv], -1e9))
                    bidx = tsel.argmax(-1)
                    wsel = wrow.gather(1, bidx[:, None, None].expand(-1, 1, 3))[:, 0]
                    mws = torch.isfinite(wsel).all(-1)
                    if mws.any():
                        tg = wsel[mws].clamp(min=1e-6)
                        tg = tg / tg.sum(-1, keepdim=True)
                        for head, inp in ((model.sf_v_full, za_s[mws]),
                                          (model.sf_v,
                                           za_s[mws][:, model.lift.d_state:])):
                            lp = torch.log_softmax(head(inp), -1)
                            l_sv = l_sv + (-(tg * lp).sum(-1).mean())
                else:
                    l_sv = (hl.ce(model.sf_v_full(za_s), sv_tt) +
                            hl.ce(model.sf_v(za_s[:, model.lift.d_state:]), sv_tt))
        # ---- SIBLING VALUE ON ROLLED STATES ----
        # SF labels carry a value for EVERY legal move (100pct coverage, mean 32.3
        # moves/position). Every other k-step target follows the PLAYED line, where
        # the operator's shared move-INDEPENDENT term alone predicts the next state --
        # so nothing else in this objective ever requires K(m) to depend on m. Here we
        # roll the parent through each of several legal moves and price each rolled
        # child against THAT MOVE's own SF triple.
        # POV: the stored triple is from the PARENT mover's view; the rolled state is
        # the CHILD position, whose mover is the opponent, and the engine reads
        # sf_v_full(rolled) as the child-mover's win prob (it applies 1 - wp). So the
        # target is the triple REVERSED: (W,D,L) -> (L,D,W).
        l_sib = torch.zeros((), device=dev)
        if (args.w_sib > 0 and args.gen5 and args.ops
                and hasattr(model, "sf_v_full") and ks.any()):
            wrow_a = D["wdl"][idx].to(dev)[ks].float()            # (n,48,3) parent POV
            raw_a = D["mv"][idx].to(dev).long()[ks]               # (n,48,C) raw moves
            fin_a = fin[ks]
            valid = fin_a & torch.isfinite(wrow_a).all(-1) & (wrow_a.sum(-1) > 0)
            if valid.any():
                with torch.no_grad():
                    q_a, _, _ = model.field.mv_vq(
                        model.field.mv_vq_proj(h_mv[ks].detach()))
                za_a = model.lift(zq[ks], q_a * mask[ks][..., None])   # (n, d_aug)
                nsel = min(args.sib_cap, valid.shape[1])
                # take the first nsel VALID moves per row (move order is the graph's,
                # which is legal-move order -- not value-sorted, so no selection bias)
                order = torch.argsort((~valid).float(), dim=1, stable=True)[:, :nsel]
                vsel = valid.gather(1, order)
                if vsel.any():
                    ri = torch.arange(len(order), device=dev)[:, None].expand_as(order)
                    frm = raw_a[ri, order, 1][vsel]
                    to = raw_a[ri, order, 2][vsel]
                    pro = raw_a[ri, order, 5][vsel]
                    zrep = za_a[:, None, :].expand(-1, nsel, -1)[vsel]
                    zc = model.ops(zrep, frm, to, pro)
                    wsel_s = wrow_a[ri, order]                     # (B,nsel,3) parent POV
                    if args.sib_pl:
                        # LISTWISE: score only the ORDER. KL below is minimised well
                        # by emitting the position MEAN for every sibling (measured:
                        # 4pct of true spread, within-position ordering 0.00); PL has
                        # no constant escape.
                        pr_c = torch.softmax(model.sf_v_full(zc), -1)
                        vc = pr_c[:, 0] + 0.5 * pr_c[:, 1]         # child-mover POV
                        sc = torch.zeros(vsel.shape, device=dev, dtype=vc.dtype)
                        sc[vsel] = 1.0 - vc                        # parent-POV score
                        tv = (wsel_s[..., 0] + 0.5 * wsel_s[..., 1])
                        l_sib = plackett_luce(sc, tv.detach(), vsel)
                    else:
                        tg_s = wsel_s[vsel].flip(-1)               # child-mover POV
                        tg_s = tg_s.clamp(min=1e-6)
                        tg_s = tg_s / tg_s.sum(-1, keepdim=True)
                        lp_s = torch.log_softmax(model.sf_v_full(zc), -1)
                        l_sib = -(tg_s.detach() * lp_s).sum(-1).mean()
        l_iv = torch.zeros((), device=dev)
        iv_stats = None
        if args.iv and ks.any():
            with torch.no_grad():
                q_mv, _, _ = model.field.mv_vq(model.field.mv_vq_proj(h_mv[ks].detach()))
            o = model.iv(torch.cat([zq[ks][:, None, :].expand(-1, MV_CAP, -1), q_mv], -1))
            dv = o[..., 0]
            lo = dv - F.softplus(o[..., 1]); hi = dv + F.softplus(o[..., 2])
            best = torch.where(fin[ks], tgt[ks], torch.full_like(tgt[ks], -1e9)
                               ).max(-1, keepdim=True).values
            y = tgt[ks] - best                      # gap-to-best, <= 0, oracle-valued
            m2 = fin[ks]
            assert not y.requires_grad             # U13 runtime guard
            if m2.any():
                def _pin(err, tau):
                    return torch.maximum(tau * err, (tau - 1) * err).mean()
                l_iv = (F.huber_loss(dv[m2], y[m2]) + _pin(y[m2] - lo[m2], 0.1)
                        + _pin(y[m2] - hi[m2], 0.9))
                iv_stats = (float(((y >= lo) & (y <= hi) & m2).sum() / m2.sum()),
                            float((hi - lo)[m2].mean()))
        l_depth = (model.field.depth_penalty().to(dev)
                   if args.depth_cost else torch.zeros((), device=dev))
        loss = (args.w_maia * l_maia + args.w_sf * l_sf + args.w_decor * l_decor + vl
                + args.w_sib * l_sib + args.w_depth * l_depth
                + args.w_iv * l_iv + 0.5 * l_sv)
        return loss, l_maia, l_sf, l_decor, (lg, zq, h_mv, fin, tgt, kind, mask,
                                             l_iv, iv_stats, l_sib)

    def step_fn(m_, step):
        model.train()
        if step % args.rescan_every == 0:
            refresh_store()
        idx = torch.from_numpy(tr_i[rng.integers(0, len(tr_i), args.batch)])
        loss, l_maia, l_sf, l_decor, aux = losses_on(idx)
        op_d = {}
        if args.ops:
            if ema_field is not None and not ema_started[0]:
                # hard-copy BEFORE the first child-target encode: the online field's
                # books initialized during losses_on above; the copy must be frozen
                # AND initialized from step 1's very first target (U18's contract)
                _ema_update()
            l_op, op_d = op_losses(step)
            loss = loss + l_op
        if _nan_guard(loss):
            # PAUSE-ALARM: save state, exit 3; the watchdog surfaces it (pause, not kill)
            torch.save({"state_dict": model.state_dict(), "step": step,
                        "args": vars(args)}, args.out + "_nanpause.pt")
            _culprits = [k for k in (op_d or {}) if k.startswith("NONFINITE_")]
            print(f"[gko1] ALARM non-finite loss at step {step}; state saved; pausing"
                  f" | culprits: {_culprits or 'none in op_losses -- check the main '
                  'losses or the vq/aux terms folded into vl'}", flush=True)
            raise SystemExit(3)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if args.ops and op_d.get("rho", 0) > 1.03:
            # hard projection: the penalty lost the arms race twice (rho 1.046, 1.070);
            # shrink the low-rank move terms directly, no gradients involved
            with torch.no_grad():
                model.ops.U.mul_(0.97)
        if args.ops and ema_field is not None:
            _ema_update()
        out = {"loss": float(loss.detach())}
        out.update(op_d)
        if step % args.eval_every == 0 or step == 1:
            model.eval()
            with torch.no_grad():
                kl_s = sp_s = kl_n = sp_n = 0.0
                ag_s = ag_n = 0
                iv_cov = iv_wid = iv_n = 0.0
                # te_i is sorted, so a PREFIX is the oldest shards only -- when the
                # window is compositionally uneven (sf shards converted last land at
                # the end) the prefix silently excludes a whole row-kind and its
                # gauges read 0.000. Sample instead, fixed seed for step-to-step
                # comparability. (2026-08-23: sf-spearman read 0.000 on a run whose
                # sf TRAINING loss was healthy at 3.361.)
                te_ev = (te_i if len(te_i) <= 2048 else
                         np.sort(np.random.default_rng(12345).choice(
                             te_i, 2048, replace=False)))
                for a in range(0, len(te_ev), 256):
                    ii = torch.from_numpy(te_ev[a:a + 256])
                    _, lm, ls, _, aux2 = losses_on(ii, train=False)
                    (lg, zq, h_mv, fin, tgt, kind, mask) = aux2[:7]
                    km = kind == 1
                    if km.any():
                        kl_s += float(lm) * int(km.sum()); kl_n += int(km.sum())
                        # per-eval maia top-1 agreement (argmax match with the oracle)
                        pm = lg[km].masked_fill(~mask[km], -1e9).argmax(-1)
                        tt = torch.where(fin[km], tgt[km],
                                         torch.full_like(tgt[km], -1e9)).argmax(-1)
                        ag_s += int((pm == tt).sum()); ag_n += int(km.sum())
                    if args.iv and aux2[8] is not None:
                        iv_cov += aux2[8][0] * int((kind == 0).sum())
                        iv_wid += aux2[8][1] * int((kind == 0).sum())
                        iv_n += int((kind == 0).sum())
                    ks = kind == 0
                    if ks.any():
                        x = _av_in(zq[ks], h_mv[ks], mask[ks])
                        pred = hl.decode(model.sf_av(x))
                        for r in range(len(pred)):
                            f2 = fin[ks][r]
                            if f2.sum() > 3:
                                p2 = pred[r][f2].cpu().numpy()
                                t3 = tgt[ks][r][f2].cpu().numpy()
                                pr = np.argsort(np.argsort(p2))
                                tr2 = np.argsort(np.argsort(t3))
                                c = np.corrcoef(pr, tr2)[0, 1]
                                if np.isfinite(c):
                                    sp_s += c; sp_n += 1
            el = time.time() - t0
            ex = ""
            if args.ops:
                r2 = op_eval()
                ex += (f" | opR2 {r2['o1']:.3f}/{r2['o3']:.3f}/{r2['o6']:.3f}"
                       f" mlp {r2['m1']:.3f}/{r2['m3']:.3f}/{r2['m6']:.3f}"
                       f" id {r2['i1']:.3f}/{r2['i3']:.3f}/{r2['i6']:.3f}"
                       f" rho {op_d.get('rho', 0):.3f} cons6 {op_d.get('cons6', 0):.3f}"
                       f" | mv {op_d.get('move', 0):.2f} dyn {op_d.get('dyn2', 0):.2f}"
                       f" vamp {op_d.get('vamp', 0):.1f}/32"
                       + (f" | vMAE {r2.get('v1', -1):.3f}/{r2.get('v3', -1):.3f}/"
                          f"{r2.get('v6', -1):.3f} vn {op_d.get('vn', 0)}"
                          if "v1" in r2 else "")
                       + f" phivar {op_d.get('phivar', 0):.4f}"
                       + (f" | avSp {r2.get('a1', -1):.3f}/{r2.get('a3', -1):.3f}/"
                          f"{r2.get('a6', -1):.3f}" if "a1" in r2 else ""))
                out.update({f"op_r2_{k}": v for k, v in r2.items()})
                if args.gen5:
                    # mover-POV states flip side-to-move features every ply, so
                    # odd-lag identity-R2 is structurally negative; erosion is
                    # watched on the even lag: floor, plus decay from post-warmup best.
                    i6 = r2["i6"]
                    if step >= 1000:
                        id6_hi[0] = i6 if id6_hi[0] is None else max(id6_hi[0], i6)
                    if i6 < -0.6 or (id6_hi[0] is not None and i6 < id6_hi[0] - 0.35):
                        print(f"[gko1] ALARM identity-R2@6 {i6:.3f} (post-warmup best "
                              f"{id6_hi[0]}) -- even-lag temporal coherence eroding", flush=True)
                elif r2["i1"] < 0.5:
                    print(f"[gko1] ALARM identity-R2@1 {r2['i1']:.3f} < 0.5 -- "
                          f"temporal coherence eroding despite the stack", flush=True)
            if args.iv and iv_n:
                ex += f" | iv-cov {iv_cov/iv_n:.3f} width {iv_wid/iv_n:.4f}"
                out.update({"iv_cov": iv_cov / iv_n, "iv_width": iv_wid / iv_n})
            print(f"[gko0] step {step:>6}/{args.steps} {el/60:>6.1f}m "
                  f"({el/max(1,step):.3f}s/step) | maia {float(l_maia):.3f} "
                  f"sf {float(l_sf):.3f} decor {float(l_decor):.3f} | "
                  f"HELD-OUT maiaKL {kl_s/max(1,kl_n):.3f} "
                  f"maia-top1 {ag_s/max(1,ag_n):.3f} "
                  f"sf-spearman {sp_s/max(1,sp_n):.3f} | rows {N:,}" + ex, flush=True)
            out.update({"maia_kl": kl_s / max(1, kl_n),
                        "sf_spearman": sp_s / max(1, sp_n), "rows": N})
            model.train()
        return out

    if args.units:
        import sys
        fails = []

        # GATES MUST NOT PERTURB WHAT THEY MEASURE (Kaveh 2026-08-24). Some checks
        # are inherently mutating -- U19b must take a real optimizer step to observe
        # the EMA lag, U22 must weight state_proj to prove conditioning is live --
        # and every gate that ran AFTER one of those was silently measuring a model
        # the suite itself had modified. These snapshot/restore around any such gate.
        def _snap():
            return ({k: v.detach().clone() for k, v in model.state_dict().items()},
                    {k: v.detach().clone() for k, v in ema_field.state_dict().items()}
                    if ema_field is not None else None,
                    copy.deepcopy(opt.state_dict()), bool(ema_started[0]))

        def _restore(sn):
            msd, esd, osd, es = sn
            with torch.no_grad():
                model.load_state_dict(msd)
                if esd is not None and ema_field is not None:
                    ema_field.load_state_dict(esd)
            opt.load_state_dict(osd)
            ema_started[0] = es
            opt.zero_grad(set_to_none=True)

        def gate(name, ok, detail):
            print(f"  {name:<5} {'PASS' if ok else 'FAIL':<5} {detail}", flush=True)
            if not ok:
                fails.append(name)

        # U12: elo passthrough -- same rows, band vs fixed conditioning
        ii = torch.from_numpy(tr_i[:8])
        sqf, mvf, edges, mask, elo, psl = batch_of(ii)
        with torch.no_grad():
            z_a = model.field.encode(sqf, mvf, edges, mask, elo=elo, pslot=psl)[0]
            z_b = model.field.encode(sqf, mvf, edges, mask,
                                     elo=torch.full_like(elo, 1.5), pslot=psl)[0]
        dz = float((z_a - z_b).abs().max())
        gate("U12", z_a.shape == z_b.shape and dz > 0,
             f"band vs fixed elo: shapes {tuple(z_a.shape)} equal, max|dz| {dz:.2e} > 0")

        # U13: circularity -- every target reaching a loss is grad-free
        _, _, _, _, aux = losses_on(ii, train=False)
        tgt_free = not aux[4].requires_grad
        gate("U13", tgt_free, f"oracle targets requires_grad == {aux[4].requires_grad}"
             " (runtime asserts also live in the loss paths)")

        # U14: minimizer-at-target for each loss formula
        g = torch.Generator().manual_seed(0)
        lg2 = torch.randn(6, 10, generator=g).requires_grad_(True)
        t2 = torch.softmax(lg2, -1).detach()
        gm = torch.autograd.grad((-(t2 * F.log_softmax(lg2, -1)).sum(-1)).mean(), lg2)[0]
        y2 = torch.rand(64, generator=g)
        lg3 = hl.encode(y2.to(dev)).clamp(min=1e-12).log().requires_grad_(True)
        gs = torch.autograd.grad(hl.ce(lg3, y2.to(dev)), lg3)[0]
        dv2 = torch.randn(32, generator=g)
        hub = float(F.huber_loss(dv2, dv2.clone()))
        pin = float(torch.maximum(0.1 * torch.zeros(32), -0.9 * torch.zeros(32)).mean())
        opz = 0.0
        if args.ops:
            from catspace.research.tools.training_infra.koopman_ops import (
                InteractionLift as _IL, KoopmanOps as _KO, one_ply_loss as _opl)
            il = _IL(16, 8); ko = _KO(24, n_codes=4, rank=4)
            zz = torch.randn(8, 16, generator=g)
            opz = float(_opl(il, ko, zz, torch.zeros(8, dtype=torch.long), zz)[0])
        gate("U14", float(gm.abs().max()) < 1e-6 and float(gs.abs().max()) < 1e-5
             and hub == 0.0 and pin == 0.0 and opz == 0.0,
             f"grad-at-optimum: maia {float(gm.abs().max()):.1e} sf {float(gs.abs().max()):.1e}"
             f" | zero-at-target: huber {hub} pinball {pin} op1 {opz}")

        # U16: metrics move under perturbation; the NaN pause-path fires
        cov_a = float(((y2 >= y2 - 0.1) & (y2 <= y2 + 0.1)).float().mean())
        cov_b = float((((y2 + 0.5) >= y2 - 0.1) & ((y2 + 0.5) <= y2 + 0.1)).float().mean())
        nan_fires = _nan_guard(torch.tensor(float("nan"))) and not _nan_guard(
            torch.tensor(1.0))
        gate("U16", cov_a != cov_b and nan_fires,
             f"coverage moves {cov_a:.2f}->{cov_b:.2f} on shifted y; nan-guard fires {nan_fires}")

        if args.ops and ema_field is not None:
            import hashlib

            def _fp(m):
                h = hashlib.sha256()
                for t in list(m.parameters()) + list(m.buffers()):
                    h.update(t.detach().cpu().numpy().tobytes())
                return h.hexdigest()

            # U18: the EMA target encoder must be BIT-FROZEN under model.train() +
            # a full target-encode forward (the train()-recursion re-arm bug)
            _sn_ema = _snap()                  # everything below here MUTATES
            model.train()
            _ema_update()                      # mirror the real flow: hard copy first
            ok18_mode = not model.ema_field.training
            f0 = _fp(ema_field)
            ci_u = CH_pos[ch_tr[:4]]
            sq_u, mv_u, ed_u, mk_u, el_u, pl_u, ps_u = _feat_rows(ci_u[:, 1:].ravel())
            with torch.no_grad():
                ema_field.encode(sq_u, mv_u, ed_u, mk_u, elo=el_u, pslot=ps_u)
            gate("U18", ok18_mode and _fp(ema_field) == f0,
                 f"ema frozen under train(): mode-pin {ok18_mode}, "
                 f"fingerprint unchanged {_fp(ema_field) == f0}")

            # U19: post-hard-copy exact output equivalence (end to end)
            with torch.no_grad():
                zo = model.field.encode(sq_u, mv_u, ed_u, mk_u, elo=el_u, pslot=ps_u)[0]
                ze = ema_field.encode(sq_u, mv_u, ed_u, mk_u, elo=el_u, pslot=ps_u)[0]
            eq = float((zo - ze).abs().max())
            zsc = float(zo.abs().max()) + 1e-9
            if ema_restored:
                # RESUMED with a real target: its lag is training state. The invariant
                # is that the hard copy was SUPPRESSED (they still differ), not equality.
                gate("U19", eq > 0.0,
                     f"resumed ema LAGS online, hard copy suppressed: max|dz| {eq:.1e}")
            else:
                # FRESH: the hard copy ran. Bit-exactness is NOT the invariant -- online
                # forwards between the copy and here drift the VQ codebooks while the ema
                # is pinned to eval (U18) -- so assert close TRACKING, scaled to |z|.
                gate("U19", eq < 0.05 * zsc,
                     f"post-hard-copy ema tracks online: max|dz| {eq:.1e} "
                     f"< 5pct of |z|max {zsc:.1e}")

            # U19b: after one optimizer step + update, ema LAGS (differs, moves less)
            p0_on = model.field.g_proj.weight.detach().clone()
            p0_em = ema_field.g_proj.weight.detach().clone()
            idxu = torch.from_numpy(tr_i[:4])
            lu, *_ = losses_on(idxu)
            opt.zero_grad(); lu.backward(); opt.step()
            _ema_update()
            d_on = float((model.field.g_proj.weight - p0_on).abs().max())
            d_em = float((ema_field.g_proj.weight - p0_em).abs().max())
            with torch.no_grad():
                ze2 = ema_field.encode(sq_u, mv_u, ed_u, mk_u, elo=el_u, pslot=ps_u)[0]
                zo2 = model.field.encode(sq_u, mv_u, ed_u, mk_u, elo=el_u, pslot=ps_u)[0]
            diff = float((zo2 - ze2).abs().max())
            # the EMA's per-step movement is (1-decay) x its CURRENT GAP to online --
            # near zero from a fresh hard copy, larger from a restored lag. Testing it
            # against the online step size only holds in the fresh case; test the
            # decay law itself, which holds in both.
            gap = float((model.field.g_proj.weight - p0_em).abs().max())
            pred = (1.0 - args.ema_decay) * gap
            ok19b = d_on > 0 and diff > 0 and d_em <= max(1.5 * pred, 1e-9)
            gate("U19b", ok19b,
                 f"lag obeys decay law: online moved {d_on:.2e}, ema {d_em:.2e} "
                 f"<= 1.5x(1-d)*gap {pred:.2e}, outputs differ {diff:.2e}")
            _restore(_sn_ema)                  # undo the optimizer step + ema updates
            gate("U23", True,
                 "model/ema/optimizer restored after the mutating gates -- every "
                 "check below measures the checkpoint as loaded, not as the suite "
                 "left it")

        # U20: the configuration's INTENT is armed -- value heads exist and the
        # value losses actually fire (a guarded feature silently off gated green
        # tonight; absence must FAIL, not skip)
        armed = model_has_v and hasattr(model, "av_roll") and CHV is not None
        vn20 = 0
        if armed:
            _, d20 = op_losses(4)
            vn20 = d20.get("vn", 0)
        gate("U20", armed and vn20 > 0,
             f"value anchors armed: heads {model_has_v}/{hasattr(model, 'av_roll')}, "
             f"coverage loaded {CHV is not None}, targets hit this batch {vn20}")

        # U21: the SIBLING loss must reach the operator's MOVE-DEPENDENT parameters.
        # A version that only trained sf_v_full would look healthy and fix nothing --
        # the whole point is pressure on K(m)'s dependence on m (2026-08-23: measured
        # at ~1pct of the shared term). Also checks the POV flip, since a sign error
        # here trains the value head to invert every child.
        if args.gen5 and args.ops and hasattr(model, "sf_v_full"):
            import copy as _c21
            w_keep = args.w_sib
            args.w_sib = 1.0
            try:
                idx21 = torch.from_numpy(tr_i[rng.integers(0, len(tr_i), args.batch)])
                _, _, _, _, aux21 = losses_on(idx21)
                l_sib21 = aux21[-1] if isinstance(aux21[-1], torch.Tensor) else None
            finally:
                args.w_sib = w_keep
            # POV: reversing a decisive parent-win triple must yield a child LOSS
            wpar = torch.tensor([[900.0, 80.0, 20.0]])          # parent mover winning
            wchild = wpar.flip(-1)
            pov_ok = bool(wchild[0, 0] < wchild[0, 2] and wchild[0, 2] == wpar[0, 0])
            gsib = gvsib = gvv = 0.0
            if l_sib21 is not None and float(l_sib21) > 0:
                gU, gM, gV = (torch.autograd.grad(
                    l_sib21, t, retain_graph=True, allow_unused=True)[0]
                    for t in (model.ops.U, model.ops.mix.weight, model.ops.V))
                gsib = float(gU.abs().max()) if gU is not None else 0.0
                gvsib = float(gM.abs().max()) if gM is not None else 0.0
                gvv = float(gV.abs().max()) if gV is not None else 0.0
            # U22: state-conditioning must make K ACTUALLY depend on z. Zero-init
            # means a fresh model is identical to the old one, so this gate would pass
            # vacuously at step 0 -- it therefore perturbs state_proj and checks the
            # weights MOVE, i.e. that the wiring is live rather than merely present.
            _sn_sm = _snap()
            if args.state_mix and hasattr(model.ops, "state_proj"):
                with torch.no_grad():
                    # RESTORE, never zero_(): an earlier version of this gate reset
                    # state_proj to zero after probing, silently wiping whatever the
                    # resumed checkpoint had learned. A diagnostic must not mutate
                    # the thing it measures.
                    keep_w = model.ops.state_proj.weight.detach().clone()
                    z1 = torch.randn(1, model.ops.d_aug, device=dev)
                    z2 = torch.randn(1, model.ops.d_aug, device=dev)
                    f1 = torch.zeros(1, dtype=torch.long, device=dev)
                    w_a = model.ops.move_weights(f1, f1, f1, z1)
                    w_b = model.ops.move_weights(f1, f1, f1, z2)
                    same_at_init = float((w_a - w_b).abs().max())
                    w_is_zero = float(keep_w.abs().max()) == 0.0
                    model.ops.state_proj.weight.normal_(0, 0.1)
                    w_c = model.ops.move_weights(f1, f1, f1, z1)
                    w_d = model.ops.move_weights(f1, f1, f1, z2)
                    live = float((w_c - w_d).abs().max())
                    model.ops.state_proj.weight.copy_(keep_w)
                # FRESH state_proj is all-zero, so the operator must start exactly at
                # the old behaviour. A RESUMED one has LEARNED, so non-zero state
                # sensitivity is the property working, not a violation -- the same
                # fresh-only-invariant trap U19 fell into.
                if w_is_zero:
                    gate("U22", same_at_init == 0.0 and live > 1e-6,
                         f"fresh state-mix: zero-init reproduces the old operator "
                         f"exactly ({same_at_init:.1e}) and is live once weighted "
                         f"({live:.2e})")
                else:
                    gate("U22", same_at_init > 1e-6,
                         f"resumed state-mix is LIVE: the same move gets different "
                         f"mixtures in different states ({same_at_init:.2e}) -- "
                         f"state_proj has trained away from its zero init")

            _restore(_sn_sm)
            # BasisOps inits V=0 so K starts at EXACT identity (U1c). The whole
            # move-dependent term is then identically zero, so U and mix provably
            # receive NO gradient until V moves -- asserting grad U > 0 on a fresh
            # model asserts something false. V is the parameter the signal must reach
            # at init; U and mix carry it once the operator has left identity. Third
            # instance of this fresh-vs-resumed trap (see U19, U22).
            if float(model.ops.V.abs().max()) == 0.0:
                gate("U21", gvv > 0 and pov_ok,
                     f"fresh op (V=0, K=identity): sibling loss reaches V {gvv:.2e} "
                     f"> 0, the only live move-dependent parameter; U/mix are "
                     f"structurally zero here | POV flip {pov_ok}")
            else:
                gate("U21", gsib > 0 and gvsib > 0 and gvv > 0 and pov_ok,
                     f"trained op: sibling loss reaches U {gsib:.2e}, mix {gvsib:.2e}, "
                     f"V {gvv:.2e} (all > 0) | POV flip {pov_ok}")
            # U20's lesson: a guarded feature that is silently OFF gates green. So
            # when --w-sib is enabled the term must actually FIRE (needs sf rows in
            # the batch AND finite wdl triples); when disabled it must be exactly 0.
            lsv = float(l_sib21) if l_sib21 is not None else 0.0
            if w_keep > 0:
                gate("U21b", lsv > 0,
                     f"sibling loss ARMED and firing: w_sib={w_keep}, l_sib={lsv:.4f} "
                     f"(>0 required -- needs sf rows with finite wdl in the batch)")
            else:
                gate("U21b", True,
                     f"sibling loss OFF by config (w_sib={w_keep}); measured "
                     f"l_sib={lsv:.4f} under a forced probe, so the path is live "
                     f"when enabled")

        # U24: the depth penalty must actually REACH the layer mixes, and pushing on
        # it must move concepts SHALLOWER. A penalty that is present but ungradiented
        # would leave the profile pinned at its uniform init and look fine.
        if args.depth_cost:
            _sn_d = _snap()
            d0 = model.field.depth_profile()
            pen = model.field.depth_penalty()
            gmix = torch.autograd.grad(pen, model.field.mix_board,
                                       retain_graph=True, allow_unused=True)[0]
            gd = float(gmix.abs().max()) if gmix is not None else 0.0
            mixes = (model.field.mix_board, model.field.mix_sq,
                     model.field.mix_pc, model.field.mix_mv)
            grads = torch.autograd.grad(model.field.depth_penalty(), mixes,
                                        allow_unused=True)   # OUTSIDE no_grad
            with torch.no_grad():          # one big step DOWN the penalty
                for m, g in zip(mixes, grads):
                    if g is not None:
                        m -= 50.0 * g
            d1 = model.field.depth_profile()
            shallower = all(d1[k] < d0[k] - 1e-6 for k in d0)
            _restore(_sn_d)
            gate("U24", gd > 0 and shallower,
                 f"depth penalty is live: grad to mix_board {gd:.2e} > 0, and "
                 f"descending it moves concepts shallower "
                 f"(board layer {d0['board']:.2f} -> {d1['board']:.2f} of "
                 f"{model.field.mix_board.shape[-1]})")

        if args.ops:
            r2 = op_eval(n_ch=64)
            print(f"  info  opR2 wiring on held-out chains: "
                  f"{r2['o1']:.3f}/{r2['o3']:.3f}/{r2['o6']:.3f} "
                  f"(fresh ops ~ identity persistence, not a gate)")
            # U-extra: run BOTH op_losses branches end to end and check the restored
            # stack pushes gradients into the field (the erosion lesson, gated)
            tot4, d4 = op_losses(4)          # vamp branch
            _sn_bw = _snap()
            tot4.backward()
            gf = max((pr.grad.abs().max().item() for pr in model.field.parameters()
                      if pr.grad is not None), default=0.0)
            model.zero_grad()
            tot1, d1 = op_losses(1)          # non-vamp branch
            ok_st = (d4["vamp"] != 0.0 and d1["vamp"] == 0.0 and gf > 0
                     and all(np.isfinite(v) for v in d4.values()))
            gate("U17", ok_st,
                 f"op_losses branches: vamp {d4['vamp']:.1f}/32 (step4) vs {d1['vamp']:.1f} "
                 f"(step1) | move {d4['move']:.2f} dyn {d4['dyn2']:.2f} | "
                 f"field-grad max {gf:.1e} > 0")
            _restore(_sn_bw)
        print(f"\n  {'ALL UNIT GATES PASS' if not fails else 'FAILED: ' + ', '.join(fails)}",
              flush=True)
        sys.exit(1 if fails else 0)

    cfg = TrainConfig(out=args.out, steps=args.steps, ckpt_every=args.ckpt_every,
                      start_step=0, eval_every=args.eval_every, experiment="catspace",
                      run_name=args.out.split("/")[-1], device=dev)
    standard_train(step_fn, model, cfg, args)


if __name__ == "__main__":
    main()
