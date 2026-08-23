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
        mh_vq=bool(args.g_heads), gen5=args.gen5).to(dev)
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
            self.sf_av = nn.Sequential(
                nn.Linear(field.d_state + A["d_model"], 512), nn.GELU(),
                nn.Linear(512, args.k_bins))
            if args.ops:
                if args.lift == "transformer":
                    self.lift = TransformerLift(
                        field.d_state, g_heads=A["g_heads"], d_code=A["d_code"],
                        d_sq=(16 if args.gen5 else 32), d_phi=max(args.d_phi, 256),
                        n_layers=args.lift_layers,
                        n_piece=(32 if args.gen5 else 0))
                else:
                    self.lift = InteractionLift(field.d_state, args.d_phi)
                if args.gen5:
                    self.ops = BasisOps(self.lift.d_aug, n_basis=64,
                                        rank=args.op_rank)
                else:
                    self.ops = KoopmanOps(self.lift.d_aug, n_codes=A["move_codes"],
                                          rank=args.op_rank)
                # H1's baseline arm: same targets, DETACHED inputs (see docstring)
                self.dyn_z = nn.Sequential(
                    nn.Linear(field.d_state + 64, 512), nn.GELU(),
                    nn.Linear(512, field.d_state))
            if args.iv:
                self.iv = nn.Sequential(
                    nn.Linear(field.d_state + 32, 256), nn.GELU(), nn.Linear(256, 3))
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
        print(f"[gko0] resumed self from {args.resume_self} (step {ck2.get('step')}; "
              f"{len(miss)} new keys init fresh, {len(drop)} shape-dropped, "
              f"{len(unex)} unexpected)", flush=True)
    hl = HLGauss(k=args.k_bins).to(dev)
    ema_field = getattr(model, "ema_field", None)
    ema_started = [False]

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
    print(f"[gko0] start: {N:,} rows ({len(te_i):,} held out) | "
          f"sf {int((D['kind']==0).sum()):,} maia {int((D['kind']==1).sum()):,}",
          flush=True)

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
        return idx_mv.gather(1, played[:, None]).squeeze(1)

    def _raw_ftp(rows, played):
        """Bare move identity (from, to, promo) from the RAW mv arrays, cols 1/2/5
        (gcf5 BasisOps conditioning -- no encoder involvement)."""
        raw = torch.from_numpy(np.ascontiguousarray(CH["mv"][rows])).long().to(dev)
        g = raw.gather(1, played[:, None, None].expand(-1, 1, raw.shape[-1]))[:, 0]
        return g[:, 1], g[:, 2], g[:, 5]

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
        c0 = _mv_codes(h_mv_p, played)
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
            cks = _mv_codes(h_mv_c, plc, enc_t).reshape(cb, 6)[:, :5]
            tok_c = mvc.gather(1, plc[:, None, None].expand(-1, 1, mvc.shape[-1])
                               )[:, 0].reshape(cb, 6, -1)[:, :5]
        # dyn CE to the played child's codes (global + square)
        dy = model.field.dyn(torch.cat([zq_p, model.field.mv_emb(tok_p)], -1))
        Hh, Cc, SC = model.field.g_heads, A["g_codes"], model.field.sq_codes
        l_dyn2 = (F.cross_entropy(dy[:, :Hh * Cc].reshape(-1, Cc), gi1.reshape(-1)) +
                  F.cross_entropy(dy[:, Hh * Cc:].reshape(-1, SC), si1.reshape(-1)))
        code_seq = torch.cat([c0[:, None], cks], 1)                 # (cb,6)
        t1, t3, t6 = zq_c[:, 0], zq_c[:, 2], zq_c[:, 5]
        assert not t1.requires_grad and not code_seq.requires_grad  # U13 runtime guard
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
        if args.gen5:
            l_spec, rho = model.ops.spectral_penalty(fs[0], ts[0], ps_[0])
            l_spec = l_spec + 0.05 * model.ops.identity_prior()
        else:
            l_spec, rho = model.ops.spectral_penalty(code_seq.ravel())
        # baseline: iterated MLP on DETACHED state with token move features
        s = zq_p.detach()
        toks = torch.cat([tok_p[:, None], tok_c], 1)                # (cb,6,feat)
        merr = {}
        for k in range(6):
            s = model.dyn_z(torch.cat([s, model.field.mv_emb(toks[:, k].detach())], -1))
            if k in (0, 2, 5):
                merr[k] = F.mse_loss(s, zq_c[:, k])
        l_mlp = merr[0] + merr[2] + merr[5]
        l_cons = torch.zeros((), device=dev)
        if args.w_cons > 0:
            ds_ = model.lift.d_state
            for k in (1, 3, 6):
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
            for k, wk in ((1, 1.0), (3, 1.0), (6, w6r)):
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
            sl = model.field.slow_heads * model.field.d_code
            v0 = model.field.vamp_proj(pg0.reshape(len(vb), -1)[:, :sl])
            v6 = model.field.vamp_proj(pg6.reshape(len(vb), -1)[:, :sl])
            vamp = vamp2_score(v0, v6)
            vamp_v = float(vamp)
            l_slow = -vamp / 32.0
            l_dec = vicreg_covariance(pg0.reshape(len(vb), -1)[:, sl:])
        opg = min(1.0, max(0.0, (step - args.ops_after) / 1000.0)) if args.ops_after else 1.0
        w6 = args.w_op6 * min(1.0, max(0.0, (step - args.op6_after) / 2000.0)) * opg
        total = (opg * (args.w_op1 * l1 + args.w_op3 * l3) + w6 * l6 +
                 args.w_spec * l_spec + l_mlp + l_v + opg * args.w_cons * l_cons +
                 args.w_move * l_move + args.w_dyn2 * l_dyn2 +
                 args.w_slow * l_slow * args.vamp_every + args.w_dec * l_dec * args.vamp_every)
        with torch.no_grad():
            cons6 = float(lift_consistency(model.lift, traj[:, 5]).mean())
        return total, {"op1": float(l1), "op3": float(l3), "op6": float(l6),
                       "mlp": float(l_mlp), "rho": float(rho.max()), "cons6": cons6,
                       "move": float(l_move), "dyn2": float(l_dyn2), "vamp": vamp_v,
                       "lv": float(l_v), "vn": v_n, "cons_l": float(l_cons),
                       "phivar": float(model.lift.phi(zq_p.detach()).std(0).mean())}

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
            c0 = _mv_codes(h_mv_p, played)
            tok_p = mvf.gather(1, played[:, None, None].expand(-1, 1, mvf.shape[-1]))[:, 0]
            rows_c = ci[:, 1:].ravel()
            sqc, mvc, edc, mkc, eloc, plc, psl_c = _feat_rows(rows_c)
            enc_e = ema_field if ema_field is not None else model.field
            zq_c, _, _, _, h_mv_c, _ = enc_e.encode(sqc, mvc, edc, mkc, elo=eloc, pslot=psl_c)[:6]
            zq_c = zq_c.reshape(len(ci), 6, -1)
            cks = _mv_codes(h_mv_c, plc, enc_e).reshape(len(ci), 6)[:, :5]
            tok_c = mvc.gather(1, plc[:, None, None].expand(-1, 1, mvc.shape[-1])
                               )[:, 0].reshape(len(ci), 6, -1)[:, :5]
            code_seq = torch.cat([c0[:, None], cks], 1)
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
            x = torch.cat([zq[ks][:, None, :].expand(-1, MV_CAP, -1), h_mv[ks]], -1)
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
                l_sv = (hl.ce(model.sf_v_full(za_s), sv_tt) +
                        hl.ce(model.sf_v(za_s[:, model.lift.d_state:]), sv_tt))
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
        loss = (args.w_maia * l_maia + args.w_sf * l_sf + args.w_decor * l_decor + vl
                + args.w_iv * l_iv + 0.5 * l_sv)
        return loss, l_maia, l_sf, l_decor, (lg, zq, h_mv, fin, tgt, kind, mask,
                                             l_iv, iv_stats)

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
            print(f"[gko1] ALARM non-finite loss at step {step}; state saved; pausing",
                  flush=True)
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
                for a in range(0, min(len(te_i), 2048), 256):
                    ii = torch.from_numpy(te_i[a:a + 256])
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
                        x = torch.cat([zq[ks][:, None, :].expand(-1, MV_CAP, -1),
                                       h_mv[ks]], -1)
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
                if r2["i1"] < 0.5:
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
            model.train()
            _ema_update()                      # mirror the real flow: hard copy first
            ok18_mode = not model.ema_field.training
            f0 = _fp(ema_field)
            ci_u = CH_pos[ch_tr[:4]]
            sq_u, mv_u, ed_u, mk_u, el_u, pl_u = _feat_rows(ci_u[:, 1:].ravel())
            with torch.no_grad():
                ema_field.encode(sq_u, mv_u, ed_u, mk_u, elo=el_u)
            gate("U18", ok18_mode and _fp(ema_field) == f0,
                 f"ema frozen under train(): mode-pin {ok18_mode}, "
                 f"fingerprint unchanged {_fp(ema_field) == f0}")

            # U19: post-hard-copy exact output equivalence (end to end)
            with torch.no_grad():
                zo = model.field.encode(sq_u, mv_u, ed_u, mk_u, elo=el_u)[0]
                ze = ema_field.encode(sq_u, mv_u, ed_u, mk_u, elo=el_u)[0]
            eq = float((zo - ze).abs().max())
            gate("U19", eq == 0.0, f"post-first-update ema==online: max|dz| {eq:.1e} (exact)")

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
                ze2 = ema_field.encode(sq_u, mv_u, ed_u, mk_u, elo=el_u)[0]
                zo2 = model.field.encode(sq_u, mv_u, ed_u, mk_u, elo=el_u)[0]
            diff = float((zo2 - ze2).abs().max())
            gate("U19b", d_on > 0 and 0 < d_em < 0.01 * d_on and diff > 0,
                 f"lag: online moved {d_on:.2e}, ema {d_em:.2e} (<1pct), outputs differ {diff:.2e}")

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

        if args.ops:
            r2 = op_eval(n_ch=64)
            print(f"  info  opR2 wiring on held-out chains: "
                  f"{r2['o1']:.3f}/{r2['o3']:.3f}/{r2['o6']:.3f} "
                  f"(fresh ops ~ identity persistence, not a gate)")
            # U-extra: run BOTH op_losses branches end to end and check the restored
            # stack pushes gradients into the field (the erosion lesson, gated)
            tot4, d4 = op_losses(4)          # vamp branch
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
        print(f"\n  {'ALL UNIT GATES PASS' if not fails else 'FAILED: ' + ', '.join(fails)}",
              flush=True)
        sys.exit(1 if fails else 0)

    cfg = TrainConfig(out=args.out, steps=args.steps, ckpt_every=args.ckpt_every,
                      start_step=0, eval_every=args.eval_every, experiment="catspace",
                      run_name=args.out.split("/")[-1], device=dev)
    standard_train(step_fn, model, cfg, args)


if __name__ == "__main__":
    main()
