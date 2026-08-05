#!/usr/bin/env python
"""catspace/research/components/encoder/approaches/cone_fb_embedding/experiments/train_iqe_head.py -- M1: the IQE reachability head over FROZEN Leela-trunk features.
Geometry-first (MILESTONES locked decision 1): NO committor/WDL head -- the trainable object is a
thin adapter + IQE quasimetric over precomputed trunk features (precompute_trunk_features.py,
fp16 memmap; the trunk itself is never touched).

Losses (all tested, catspace/research/tools/training_infra/losses.py):
  multi-goal  quasimetric_regression( d(phi_i -> phi_j), log1p(ply_gap) )   same-game pairs
  mate        quasimetric_regression( d(phi -> MATE),   log1p(DTZ) )        tablebase-won anchors
  hinge       wdl_hinge( d_mate, won, log(margin) )                          distance margin (geometry)
  repulsion   relu( margin - log1p(d(phi_s -> phi_perm)) )                  anti-collapse

Gates logged on HELD-OUT val games (same protocol as ClockField v3 for the kill decision):
pair-order Spearman | d_mate-vs-DTZ Spearman | eff_rank(phi_head). Off-distribution d_mate +
opening-sanity run post-train (eval_iqe_field.py). Scaffold-tracked (MLflow + ladders + provenance).
"""
from __future__ import annotations

import argparse, json, sys, time
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from catspace.research.components.encoder.approaches.jepa_tokenizer.src.iqe import IQE
from catspace.research.tools.training_infra.losses import (
    quasimetric_regression, wdl_hinge, reachability_target,
    basin_ce, basin_logp, pole_radial_anchor, terminal_repulsion,
    pole_potential, typical_pair_scale, absorbing_penalty, basin_width,
    start_ply_anchor, start_irreversibility, WIN, DRAW, LOSS, START)
from catspace.research.components.encoder.approaches.reachability_field.experiments.arch_bakeoff import eff_rank
from catspace.research.tools.training_infra.train.scaffold import standard_train, TrainConfig, resolve_device


from catspace.research.components.encoder.approaches.reachability_field.src.iqe_head import IQEHead  # component home (refactor 2026-07-30)
from catspace.io import paths


class DualFeats:
    """Row-index -> trunk features across the TWO source memmaps, without concatenating them.

    The combined dataset is human + SF-vs-SF, whose precomputed trunk features are two ~36GB
    fp16 memmaps. Copying them into one file would cost ~72GB of disk and a long serial write
    for zero modelling benefit, so the combined npz carries (source, local_row) per row and this
    gathers in place. Indices are SORTED per source before the memmap read -- sequential access
    on a memmap this size is dramatically friendlier than a random gather -- then unsorted back."""

    def __init__(self, paths, source, local_row):
        self.paths = list(paths)
        self.source, self.local_row = source, local_row
        # LAZY: only sources actually present in `source` are opened, so a --source-restricted
        # run (e.g. SF-only, while the human trunk features are still being precomputed) does not
        # require a memmap it will never read.
        self.mm = [None] * len(self.paths)
        present = np.unique(source)
        for s in present:
            self._open(int(s))
        ref = self.mm[int(present[0])]
        self.shape = (len(source), *ref.shape[1:])
        for s in present:
            assert self.mm[int(s)].shape[1:] == ref.shape[1:], f"feats[{s}] shape mismatch"

    def _open(self, s):
        if self.mm[s] is None:
            self.mm[s] = np.load(self.paths[s], mmap_mode="r")
        return self.mm[s]

    def __len__(self):
        return self.shape[0]

    def gather(self, idx, dtype=np.float32):
        """(B,) global row indices -> (B,C,8,8) in `dtype`.

        dtype=float16 keeps the on-disk precision and defers the cast to the GPU, which halves the
        bytes written here and halves the host->device transfer."""
        idx = np.asarray(idx)
        out = np.empty((len(idx), *self.shape[1:]), dtype=dtype)
        src = self.source[idx]
        for s in np.unique(src):
            s = int(s)
            m = src == s
            if not m.any():
                continue
            loc = self.local_row[idx[m]]
            order = np.argsort(loc)                          # sequential memmap reads
            rows = np.asarray(self._open(s)[loc[order]], dtype=dtype)
            dest = np.flatnonzero(m)[order]
            out[dest] = rows
        return out


def build_pairs(game, ply, games_set, rng, per_game=10):
    rows_by_game = defaultdict(list)
    for i in range(len(game)):
        g = int(game[i])
        if g in games_set:
            rows_by_game[g].append(i)
    S, G, D = [], [], []
    for rows in rows_by_game.values():
        rows = sorted(rows, key=lambda i: ply[i])
        if len(rows) < 2:
            continue
        for _ in range(min(per_game, len(rows))):
            a, b = sorted(rng.integers(0, len(rows), 2))
            if a == b:
                continue
            si, gj = rows[a], rows[b]; delta = int(ply[gj] - ply[si])
            if delta <= 0:
                continue
            S.append(si); G.append(gj); D.append(np.log1p(delta))
    return np.array(S), np.array(G), np.array(D, np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--feats", default=paths.derived("trunk_feats/maia-1500__field_std_v1.npy"))
    ap.add_argument("--data", default=paths.derived("field_std_v1.npz"))
    ap.add_argument("--d", type=int, default=64); ap.add_argument("--components", type=int, default=16)
    ap.add_argument("--adapter-ch", type=int, default=32)
    ap.add_argument("--w-multi", type=float, default=1.0); ap.add_argument("--w-mate", type=float, default=1.0)
    ap.add_argument("--w-hinge", type=float, default=0.5); ap.add_argument("--w-repel", type=float, default=1.0)
    ap.add_argument("--repel-margin", type=float, default=4.0); ap.add_argument("--margin", type=float, default=400.0)
    ap.add_argument("--steps", type=int, default=6000); ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--eval-every", type=int, default=500); ap.add_argument("--ckpt-every", type=int, default=1000)
    ap.add_argument("--rows", default="", help="rows .npy: train on this game-subset of the data")
    ap.add_argument("--train-frac", type=float, default=1.0,
                    help="fraction of TRAIN games to keep (holdout is untouched). For learning "
                         "curves: the val split is drawn FIRST from a fixed seed, so the holdout "
                         "is byte-identical at every size and the curves are comparable.")
    ap.add_argument("--game-mod", default="",
                    help="'K:R' -> keep only TRAIN games with game %% K == R (holdout untouched). "
                         "Two arms at 2:0 and 2:1 see DISJOINT training games, which is what a "
                         "null pair needs: --train-frac 0.5 twice would overlap ~50%% and make the "
                         "two fields artificially similar, understating the noise floor they exist "
                         "to measure.")
    ap.add_argument("--out", default=""); ap.add_argument("--seed", type=int, default=0)
    # --- W/D/L basin poles (Kaveh 2026-08-03). OFF by default: with every w-* at 0 and no
    # --combined, this script's objective and numerics are bit-for-bit the shipped M1 recipe.
    ap.add_argument("--combined", default="", help="combined metadata npz from "
                    "build_combined_field_data.py -> enables the 3-pole basin terms")
    ap.add_argument("--source", choices=["both", "human", "sf"], default="both",
                    help="restrict the combined data to one source (smoke-testing convenience)")
    ap.add_argument("--w-basin", type=float, default=1.0, help="basin cross-entropy (the main term)")
    ap.add_argument("--w-radial", type=float, default=1.0, help="pole shell anchor, TAIL rows only")
    ap.add_argument("--w-termrepel", type=float, default=1.0, help="anti-collapse ON the shell")
    ap.add_argument("--w-polesep", type=float, default=1.0,
                    help="pole-pole potential: LJ-shaped, steep repulsion inside the crossover, "
                         "weak attraction outside")
    ap.add_argument("--polesep-krep", type=float, default=10.0, help="repulsive stiffness")
    ap.add_argument("--polesep-katt", type=float, default=0.05, help="attractive stiffness")
    ap.add_argument("--resume", default="",
                    help="checkpoint to resume from. Restores model weights AND optimizer state "
                         "(Adam moments) and continues the step counter, so the ladder, the movie "
                         "frames and the per-step log stay contiguous with the original run.")
    ap.add_argument("--prefetch", type=int, default=0,
                    help="overlap the next batch's gather with this step's GPU compute. DEFAULT "
                         "OFF: measured on a quiet machine it is a net LOSS (0.39 -> 0.44 s/step "
                         "alone, and 0.21 vs 0.22 on top of --fp16-transfer, i.e. nothing). The "
                         "worker thread contends for memory bandwidth and the GIL with the main "
                         "thread, and MPS dispatch is already async so there is far less idle GPU "
                         "time to hide behind than the serial timing split suggests. Kept as a "
                         "flag because it should win at larger batches or on a CUDA box.")
    ap.add_argument("--fp16-transfer", type=int, default=1,
                    help="ship fp16 to the device and cast there (1=on). Features are fp16 on "
                         "disk; casting on the CPU first reads 282MB and writes 564MB then sends "
                         "564MB. This sends 282MB and casts on device. MEASURED 1.8x end-to-end "
                         "(0.39 -> 0.22 s/step): the gather drops 2.8x and compute drops too, "
                         "since the host->device copy halves.")
    ap.add_argument("--step-log", default="auto",
                    help="per-STEP train+holdout error -> JSONL ('auto' = <out>_steps.jsonl, "
                         "'' = off). standard_train only keeps metrics on eval steps, so a 30k "
                         "run otherwise retains ~120 points out of 30,000.")
    ap.add_argument("--step-val-batch", type=int, default=512,
                    help="held-out rows scored EVERY step for the holdout arm. ~3%% of the "
                         "~17.5k rows a step already gathers -- cheap enough to always pay.")
    ap.add_argument("--timing-every", type=int, default=50,
                    help="print an I/O-vs-compute breakdown every N steps (0 = off)")
    ap.add_argument("--pole-lr-mult", type=float, default=10.0,
                    help="lr multiplier for the pole vertices + temperature (own param group)")
    ap.add_argument("--w-start", type=float, default=1.0,
                    help="START-pole ply anchor: d(P_start->phi) regressed to log1p(ply+1). "
                         "Gives the field an ABSOLUTE ply coordinate; the multi-goal term only "
                         "ever taught relative ply gaps.")
    ap.add_argument("--w-start-irrev", type=float, default=1.0,
                    help="d(phi->P_start) pushed large: you cannot un-play moves")
    ap.add_argument("--start-irrev-margin", type=float, default=4.0,
                    help="MATCHED to --absorb-margin deliberately. Ablated at 400 steps: margin 6 "
                         "gave pole asymmetry -1.90, acc 0.533, terminal eff_rank 13.7; margin 4 "
                         "gave -0.20/0.688/17.6 and crossed positive (+0.47) by step 1500. The "
                         "start term must not overpower absorb, or it drags the coordinate "
                         "ordering that lets P_outcome < s < P_start hold.")
    ap.add_argument("--pole-init", choices=["data", "random"], default="data",
                    help="data = prototype init at each class's mean terminal phi (recommended)")
    ap.add_argument("--w-absorb", type=float, default=1.0, help="d(pole->s) large: cannot leave")
    ap.add_argument("--w-width", type=float, default=4.0,
                    help="Deep-TDA basin-WIDTH term. Default 4.0 not 1.0: it acts on log1p(d), "
                         "which compresses width ratios (a 6x wider basin scores only 0.023), so "
                         "at weight 1 it would be swamped by CE.")
    ap.add_argument("--width-sigma", type=float, default=-1.0,
                    help="<0 = equalize the three basin widths; >=0 = prescriptive Deep-TDA target")
    ap.add_argument("--termrepel-margin", type=float, default=4.0)
    ap.add_argument("--polesep-margin", type=float, default=4.0)
    ap.add_argument("--absorb-margin", type=float, default=4.0)
    args = ap.parse_args()
    t0 = time.time(); dev = resolve_device("auto"); torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    tag = Path(args.feats).stem.split("__")[0]
    out = args.out or paths.experiment(f"iqe_head_{tag}")

    poles_on = bool(args.combined)
    if poles_on:
        z = np.load(args.combined, allow_pickle=True)
        meta = eval(str(z["_meta"][0]))                      # written by build_combined_field_data.py
        keep = np.ones(len(z["y"]), bool) if args.source == "both" else \
            (z["source"] == (0 if args.source == "human" else 1))
        sel = np.flatnonzero(keep)
        dtz = z["dtz"][sel].astype(np.int32); game = z["game"][sel]; ply = z["ply"][sel]
        y_all = z["y"][sel].astype(np.int64)
        n_to_end = z["n_to_end"][sel].astype(np.int32)
        is_tail = z["is_tail"][sel]; is_term = z["is_terminal"][sel]
        feats = DualFeats(meta["feats"], z["source"][sel], z["local_row"][sel])
        fmap = None
        tag = f"combined-{args.source}"
        out = args.out or f"artifacts/experiments/iqe_poles_{args.source}"
    else:
        feats = np.load(args.feats, mmap_mode="r")           # (N,C,8,8) fp16, NEVER fully in RAM
        z = np.load(args.data)
        dtz = z["dtz"].astype(np.int32); game = z["game"]; ply = z["ply"]
        if args.rows:
            rows = np.load(args.rows)
            dtz, game, ply = dtz[rows], game[rows], ply[rows]
            fmap = rows if len(feats) != len(rows) else None  # full-size memmap -> map; subset -> direct
        else:
            fmap = None
    N, C = len(dtz), feats.shape[1]
    games = np.unique(game)
    # Val split FIRST, from the shared seed -> identical holdout regardless of --train-frac.
    val_games = set(rng.choice(games, size=max(1, int(len(games) * args.val_frac)), replace=False).tolist())
    train_games = set(int(g) for g in games) - val_games
    if args.train_frac < 1.0:
        # Subsample by GAME, never by row: positions within a game are highly correlated, so a
        # row-level subsample would leave near-duplicates of held-out positions in training and
        # the learning curve would flatter itself.
        tg = np.array(sorted(train_games))
        keep = np.random.default_rng(args.seed + 991).choice(
            len(tg), max(1, int(len(tg) * args.train_frac)), replace=False)
        train_games = set(int(g) for g in tg[keep])
        print(f"  [learning-curve] train games {len(train_games):,} of {len(tg):,} "
              f"({100*args.train_frac:.0f}%), holdout {len(val_games):,} games FIXED", flush=True)
    if args.game_mod:
        K, R = (int(v) for v in args.game_mod.split(":"))
        before = len(train_games)
        train_games = {g for g in train_games if g % K == R}
        print(f"  [game-mod] train games {len(train_games):,} of {before:,} (game %% {K} == {R}); "
              f"holdout {len(val_games):,} games UNCHANGED", flush=True)
    MG_s, MG_g, MG_d = build_pairs(game, ply, train_games, rng)
    V_s, V_g, V_d = build_pairs(game, ply, val_games, np.random.default_rng(args.seed + 1))
    is_val = np.array([int(g) in val_games for g in game])
    tb_train = np.flatnonzero((dtz >= 1) & ~is_val); tb_val = np.flatnonzero((dtz >= 1) & is_val)
    not_won_train = np.flatnonzero((dtz < 0) & ~is_val)
    va_idx = np.flatnonzero(is_val)
    print(f"[iqe-head:{tag}] N={N:,} C={C} | train pairs {len(MG_s):,} val pairs {len(V_s):,} | "
          f"tb-won train {len(tb_train):,} val {len(tb_val):,} | device {dev}", flush=True)

    if poles_on:
        # Index sets precomputed ONCE as arrays; every step gathers from them rather than
        # re-filtering 2.3M rows per step.
        all_train = np.flatnonzero(~is_val); all_val = va_idx
        tail_train = np.flatnonzero(is_tail & ~is_val)
        term_train = np.flatnonzero(is_term & ~is_val)
        y_t = torch.from_numpy(y_all).to(dev)
        z_ply = ply.astype(np.float64)
        # Radial target: reachability_target(n_to_end, surprisal=0) -- surprisal is 0 because the
        # anchor is restricted to tail rows, where the outcome is already locked (see
        # build_combined_field_data.py). The call site keeps the surprisal channel wired for when
        # a policy model can supply a real P(path). A terminal (n=0 -> floored to 1) targets
        # log1p(1), which IS Kaveh's "every terminal sits one ply from its pole".
        radial_tgt = torch.from_numpy(
            reachability_target(np.maximum(n_to_end, 1), 0.0).astype(np.float32)).to(dev)
        # ply+1, not ply: the generator pushes the move THEN records, so row ply=0 is the position
        # after White's first move and half-moves played = ply+1. Raw ply would target log1p(0)=0
        # and drag P_start onto the centroid of all ~20 after-first-move positions; ply+1 puts that
        # position ONE ply from the start pole -- the same convention terminals use.
        start_tgt = torch.from_numpy(np.log1p(ply.astype(np.float32) + 1.0)).to(dev)
        yv = y_all[all_val]
        print(f"  [poles] tail-anchor rows {len(tail_train):,} | terminals {len(term_train):,} | "
              f"basin mix train win/draw/loss "
              f"{int((y_all[all_train]==WIN).sum()):,}/{int((y_all[all_train]==DRAW).sum()):,}/"
              f"{int((y_all[all_train]==LOSS).sum()):,} | val {len(all_val):,}", flush=True)
        print(f"  [poles] radial target range log1p: {float(radial_tgt.min()):.3f}.."
              f"{float(radial_tgt.max()):.3f} (terminal shell = {float(np.log(2.0)):.4f})", flush=True)

    net = IQEHead(in_ch=C, d=args.d, components=args.components, adapter_ch=args.adapter_ch).to(dev)
    print(f"  head params: {sum(p.numel() for p in net.parameters()):,}", flush=True)
    if poles_on and args.pole_lr_mult != 1.0:
        # The vertices are 3xd + 1 numbers competing against a ~139k-param adapter at one shared
        # lr, so they drift toward each other far slower than the embedding they must separate
        # within (measured: gap 0.91 vs a crossover of 4.55 after 140 steps, and the crossover
        # GROWS as the embedding expands -- the poles chase a receding target). Their own param
        # group is the standard fix and touches no loss term. weight_decay=0 on the poles: they
        # are locations in embedding space, not weights, and decaying them pulls all three back
        # toward the origin -- i.e. directly against the separation this is meant to fix.
        pole_params = {id(net.poles), id(net.log_T)}
        rest = [p_ for p_ in net.parameters() if id(p_) not in pole_params]
        opt = torch.optim.AdamW(
            [{"params": rest, "lr": args.lr, "weight_decay": 1e-4},
             {"params": [net.poles, net.log_T], "lr": args.lr * args.pole_lr_mult,
              "weight_decay": 0.0}])
        print(f"  [poles] own param group: lr {args.lr * args.pole_lr_mult:.1e} "
              f"({args.pole_lr_mult:g}x), weight_decay 0", flush=True)
    else:
        opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    logM = float(np.log1p(args.margin))
    tgt_mate = torch.from_numpy(np.log1p(np.clip(dtz, 0, None)).astype(np.float32)).to(dev)

    # Per-phase timers. The open question on this box is whether a step is dominated by the
    # random memmap gather (71GB of features against 36GB of RAM -> page faults to disk) or by
    # the actual forward/backward on MPS. Wall-clock alone cannot tell those apart, so fx()
    # accumulates its own time and byte count and the step reports the split.
    TM = {"gather": 0.0, "bytes": 0.0, "rows": 0.0, "step": 0.0, "n": 0}

    _HDT = np.float16 if args.fp16_transfer else np.float32

    def _raw(idx):
        """memmap rows -> host array. fp16 keeps the on-disk precision and defers the cast to the
        GPU: casting on the CPU first reads 282MB and writes 564MB, then ships 564MB. This ships
        282MB and casts on device."""
        if poles_on:
            return feats.gather(idx, dtype=_HDT)
        ridx = fmap[idx] if fmap is not None else idx
        return np.asarray(feats[ridx], dtype=_HDT)

    def _to_dev(arr):
        x = torch.from_numpy(arr).to(dev, non_blocking=True)
        return x.float() if arr.dtype == np.float16 else x

    # PREFETCH: one worker thread gathers the NEXT batch while the GPU computes this one. Without
    # it the step is a strict serial alternation (gather 0.17s -> compute 0.13s, each idle while
    # the other runs) at 0.5 of 11 cores; with it the step is ~max() of the two rather than the sum.
    _pf = {"fut": None, "idx": None}
    _pool = ThreadPoolExecutor(max_workers=1) if args.prefetch else None

    def fx(idx):
        _t = time.perf_counter()
        idx = np.asarray(idx)
        if _pool is not None and _pf["fut"] is not None and _pf["idx"] is not None \
                and len(_pf["idx"]) == len(idx) and np.array_equal(_pf["idx"], idx):
            arr = _pf["fut"].result()                        # already in flight -- just collect
            _pf["fut"], _pf["idx"] = None, None
        else:
            arr = _raw(idx)
        x = _to_dev(arr)
        TM["gather"] += time.perf_counter() - _t
        TM["rows"] += len(idx)
        TM["bytes"] += arr.size * 2                          # fp16 on disk, 2 bytes/element
        return x

    def fx_prefetch(idx):
        """Queue a gather to overlap with the current step's GPU work."""
        if _pool is None:
            return
        idx = np.asarray(idx)
        _pf["idx"] = idx
        _pf["fut"] = _pool.submit(_raw, idx)

    if poles_on and args.pole_init == "data":
        # Prototype init: each pole starts at the MEAN phi of its own class's terminal positions
        # -- literally the Prototypical-Networks prototype (mean of support), which is the method
        # this head's softmax-over-distances is borrowed from.
        # Why it matters: randn*0.01 starts all three vertices stacked on top of each other at a
        # scale ~450x smaller than the crossover the potential wants them at, so early training
        # is spent climbing out of that hole (measured: pole gap moved only 0.09 -> 0.27 in 100
        # steps against a crossover of 4.5). Starting them in the right region skips that climb.
        with torch.no_grad():
            pre_train = np.flatnonzero((n_to_end == 1) & ~is_val)
            for k in (WIN, DRAW, LOSS):
                ck = term_train[y_all[term_train] == k]
                if len(ck) == 0:
                    # ENGINES NEVER RESIGN, so in SF-vs-SF no terminal position has the MOVER
                    # winning -- the mover at a terminal is the side that got mated or drawn, and
                    # class WIN has exactly 0 terminals (measured on field_combined_v2: SF terminal
                    # basins 0 win / 64,867 draw / 15,133 loss). Falling through to random init
                    # would leave the SF field's win pole stacked at the origin while the human
                    # field -- whose opponents DO resign, giving it 4,421 terminal wins -- got a
                    # prototype. h = q_SF - q_human would then partly measure that initialisation
                    # asymmetry on exactly the win side it is trying to compare.
                    # The PRE-terminal row is the honest anchor anyway: build_combined_field_data
                    # calls those rows "the win-pole anchors", because one ply before the end the
                    # mover IS the winner. SF has 15,133 of them.
                    ck = pre_train[y_all[pre_train] == k]
                    print(f"  [poles] class {k} has no terminals (engines never resign) -> "
                          f"prototype from {len(ck):,} PRE-terminal rows instead", flush=True)
                if len(ck) == 0:
                    print(f"  [poles] WARNING no terminal or pre-terminal rows for class {k}; "
                          f"keeping random init")
                    continue
                pick = ck[rng.integers(0, len(ck), min(2048, len(ck)))]
                net.poles[k] = net.phi(fx(pick)).mean(0)
            early = np.flatnonzero(ply <= 2)
            if len(early):
                pick = early[rng.integers(0, len(early), min(2048, len(early)))]
                net.poles[START] = net.phi(fx(pick)).mean(0)
            pg = float(torch.log1p(net.d_poles_pairwise()[~torch.eye(4, dtype=torch.bool,
                                                                    device=dev)]).median())
            print(f"  [poles] prototype init from terminals -> pole gap {pg:.2f}", flush=True)

    def pole_terms(_rng, batch):
        """The five W/D/L basin terms. Every distance-to-pole is ONE batched IQE.pairwise call
        (B,3) -- never three per-pole calls -- and all index sets are precomputed arrays."""
        T = net.temperature
        # (1) basin CE over a random slice of ALL training rows: forms + calibrates the simplex.
        bi = _pf_next[0] if _pf_next[0] is not None else all_train[
            _rng.integers(0, len(all_train), batch)]
        _pf_next[0] = None
        e_b = net.phi(fx(bi))
        d_b = net.d_poles(e_b)
        d_b_last[0], bi_last[0] = d_b, bi          # reused by the per-step log; no extra forward
        # Queue the next step's basin batch now, so its gather overlaps this step's GPU work.
        nxt = all_train[_rng.integers(0, len(all_train), batch)]
        _pf_next[0] = nxt
        fx_prefetch(nxt)
        L_basin = basin_ce(d_b, y_t[bi], T)
        # (5) absorbing: d(pole -> s) pushed UP on those same ordinary rows. Trains the ASYMMETRY.
        L_absorb = absorbing_penalty(net.d_from_poles(e_b).reshape(-1), args.absorb_margin)
        # (2) radial anchor on TAIL rows only, at each row's OWN outcome pole.
        ti = tail_train[_rng.integers(0, len(tail_train), batch // 2)]
        d_tail = net.d_poles(net.phi(fx(ti)))
        d_own = d_tail.gather(1, y_t[ti].unsqueeze(1)).squeeze(1)
        L_radial = pole_radial_anchor(d_own, radial_tgt[ti])
        # (3) terminal repulsion: distinct terminals must not pile onto one point of the shell.
        qi = term_train[_rng.integers(0, len(term_train), batch // 4)]
        e_q = net.phi(fx(qi))
        perm = torch.randperm(len(qi), device=dev)
        L_termrepel = terminal_repulsion(net.d_pair_emb(e_q, e_q[perm]), args.termrepel_margin)
        # (4) pole-pole potential. The crossover is the TYPICAL distance between ordinary
        # positions, measured from this same batch and detached: the poles are repelled hard once
        # they come closer to each other than real positions are, and weakly attracted beyond
        # that so the triangle cannot drift apart without bound. Recomputed per step so the
        # crossover tracks the embedding scale instead of being a fixed guess.
        ref = typical_pair_scale(net.d_pair_emb(e_b, e_b[torch.randperm(len(e_b), device=dev)]))
        L_polesep = pole_potential(net.d_poles_pairwise(), ref,
                                   k_rep=args.polesep_krep, k_att=args.polesep_katt)
        # (6) Deep-TDA basin WIDTH: pins the three basins to comparable spread. Measured on the
        # 8k run, the mates formed a knot (IQR 0.038) while the win-side anchors sprawled (0.178,
        # 4.7x) -- nothing in the objective asked them to be comparable. Computed on the SAME
        # all-rows batch as the CE term so no extra feature gather is needed.
        d_own_b = d_b.gather(1, y_t[bi].unsqueeze(1)).squeeze(1)
        L_width = basin_width(d_own_b, y_t[bi],
                              target_sigma=None if args.width_sigma < 0 else args.width_sigma)
        # (7) START pole. Reuses e_b -- the CE batch we already embedded -- so the ply anchor is
        # free. Applied to ALL rows, unlike the outcome radial anchor: ply is a fact about the
        # position's PAST and is already determined, whereas plies-to-end is contingent on both
        # players' future choices (which is why that one stays tail-only).
        L_start = start_ply_anchor(net.d_from_start(e_b), start_tgt[bi])
        L_start_irrev = start_irreversibility(net.d_to_start(e_b), args.start_irrev_margin)
        return {"basin": L_basin, "radial": L_radial, "termrepel": L_termrepel,
                "polesep": L_polesep, "absorb": L_absorb, "width": L_width,
                "start": L_start, "start_irrev": L_start_irrev, "pole_ref": ref}

    _step_t0 = [0.0]
    d_b_last, bi_last = [None], [None]
    _pf_next = [None]

    def step(_net, s):
        _step_t0[0] = time.perf_counter()
        pi = rng.integers(0, len(MG_s), args.batch)
        es = net.phi(fx(MG_s[pi])); eg = net.phi(fx(MG_g[pi]))
        L_multi = quasimetric_regression(net.d_pair_emb(es, eg), torch.from_numpy(MG_d[pi]).to(dev))
        perm = torch.randperm(len(pi), device=dev)
        L_repel = F.relu(args.repel_margin - torch.log1p(net.d_pair_emb(es, eg[perm]).clamp(min=0))).mean()
        hb = args.batch // 4
        bw = tb_train[rng.integers(0, len(tb_train), hb)]
        bn = not_won_train[rng.integers(0, len(not_won_train), hb)]
        e_all = net.phi(fx(np.concatenate([bw, bn])))
        dm = net.d_mate_emb(e_all)
        won = torch.zeros(2 * hb, device=dev); won[:hb] = 1.0
        L_mate = quasimetric_regression(dm[:hb], tgt_mate[bw])
        L_hinge = wdl_hinge(dm, won, logM)
        loss = args.w_multi * L_multi + args.w_repel * L_repel + args.w_mate * L_mate + args.w_hinge * L_hinge
        parts = {"loss": loss, "multi": L_multi, "repel": L_repel, "mate": L_mate, "hinge": L_hinge}
        if poles_on:
            pt = pole_terms(rng, args.batch)
            loss = loss + (args.w_basin * pt["basin"] + args.w_radial * pt["radial"]
                           + args.w_termrepel * pt["termrepel"] + args.w_polesep * pt["polesep"]
                           + args.w_absorb * pt["absorb"] + args.w_width * pt["width"]
                           + args.w_start * pt["start"] + args.w_start_irrev * pt["start_irrev"])
            parts.update(pt); parts["loss"] = loss
            parts["T"] = net.temperature
        opt.zero_grad(); loss.backward(); opt.step()
        # Refresh the optimizer snapshot only on ladder steps: scaffold does payload.update(extra)
        # at save time and cfg.extra is this same dict, so mutating it here is what lands in the
        # checkpoint. Doing it every step would serialize Adam moments 30,000 times for nothing.
        if s % args.ckpt_every == 0 or s == args.steps:
            _extra["opt_state"] = opt.state_dict()
        out_m = {k: float(v.detach()) for k, v in parts.items()}
        # PER-STEP train + holdout error (Kaveh 2026-08-04). The train arm is free -- it is the
        # batch we just did a forward pass on. The holdout arm costs one extra small batch, which
        # is the only way to get a genuine per-step generalization curve rather than interpolating
        # between eval points.
        if step_log is not None and poles_on:
            with torch.no_grad():
                vsub = va_idx[rng.integers(0, len(va_idx), args.step_val_batch)]
                dv = net.d_poles(net.phi(fx(vsub)))
                yv_s = y_t[vsub]
                rec = {"step": s,
                       "train_ce": float(basin_ce(d_b_last[0], y_t[bi_last[0]], net.temperature)),
                       "train_acc": float((d_b_last[0].argmin(1) == y_t[bi_last[0]]).float().mean()),
                       "val_ce": float(basin_ce(dv, yv_s, net.temperature)),
                       "val_acc": float((dv.argmin(1) == yv_s).float().mean())}
                rec.update({k: out_m[k] for k in ("loss", "basin", "radial", "width") if k in out_m})
                step_log.write(json.dumps(rec) + "\n")
                # Flush EVERY step, not every 200. SIGTERM (what `pkill` sends, and what stopping a
                # run to fix it uses) terminates before `finally: close()` runs, so a buffered tail
                # is simply lost -- measured: 50 rows missing across one stop/resume. One extra
                # syscall per step against a ~0.26s step is not worth a hole in the record.
                step_log.flush()
        # MPS is ASYNC: without a sync the backward would appear free and all the time would be
        # misattributed to whatever ran next. Sync only on reporting steps so the common path is
        # not slowed by the measurement itself.
        if args.timing_every and s % args.timing_every == 0:
            if dev.type == "mps":
                torch.mps.synchronize()
            elif dev.type == "cuda":
                torch.cuda.synchronize()
        TM["step"] += time.perf_counter() - _step_t0[0]
        TM["n"] += 1
        if args.timing_every and s % args.timing_every == 0:
            n = max(TM["n"], 1)
            sp = TM["step"] / n; g = TM["gather"] / n
            mb = TM["bytes"] / n / 1048576
            eta = (args.steps - s) * sp / 3600
            print(f"  [t] step {s}/{args.steps} | {sp:.2f}s/step "
                  f"(gather {g:.2f}s = {100*g/max(sp,1e-9):.0f}%, compute {sp-g:.2f}s) | "
                  f"{TM['rows']/n:,.0f} rows = {mb:.0f}MB/step -> {mb/max(g,1e-9):.0f} MB/s "
                  f"| ETA {eta:.1f}h", flush=True)
            for k in ("gather", "bytes", "rows", "step"):
                TM[k] = 0.0
            TM["n"] = 0
        return out_m

    from scipy.stats import spearmanr

    def gates(_net):
        with torch.no_grad():
            te = rng.integers(0, len(V_s), min(4000, len(V_s)))
            dp = net.d_pair_emb(net.phi(fx(V_s[te])), net.phi(fx(V_g[te]))).cpu().numpy()
            pair_order = float(spearmanr(dp, np.expm1(V_d[te])).correlation)
            er = float(eff_rank(net.phi(fx(va_idx[rng.integers(0, len(va_idx), 3000)])).cpu().numpy()))
            if len(tb_val) >= 50:
                dm = net.d_mate_emb(net.phi(fx(tb_val))).cpu().numpy()
                mate_rho = float(spearmanr(dm, dtz[tb_val]).correlation)
            else:
                mate_rho = float("nan")
            g = {"pair_order": pair_order, "eff_rank": er, "mate_rho": mate_rho}
            # TRAIN-side arm of the learning curve, same estimator on the same-size sample so the
            # two arms are directly comparable (a bigger val sample would look artificially smooth).
            if poles_on:
                tr = np.flatnonzero(~is_val)
                ti2 = tr[rng.integers(0, len(tr), min(4000, len(tr)))]
                dtr = net.d_poles(net.phi(fx(ti2)))
                g["tr_basin_ce"] = float(basin_ce(dtr, y_t[ti2], net.temperature))
                g["tr_basin_acc"] = float((dtr.argmin(1) == y_t[ti2]).float().mean())
                vi2 = va_idx[rng.integers(0, len(va_idx), min(4000, len(va_idx)))]
                dva = net.d_poles(net.phi(fx(vi2)))
                g["va_basin_ce"] = float(basin_ce(dva, y_t[vi2], net.temperature))
                g["va_basin_acc"] = float((dva.argmin(1) == y_t[vi2]).float().mean())
            if poles_on:
                g.update(pole_gates())
        return g

    def pole_gates(n=6000, n_bins=15):
        """Basin gates. CALIBRATION is the primary one: the poles must yield PROBABILITIES, not
        just a separating clustering -- an over-confident field would look like clean basins while
        being wrong, and would empty the undetermined middle the whole design depends on."""
        vi = all_val[rng.integers(0, len(all_val), min(n, len(all_val)))]
        e = net.phi(fx(vi))
        d = net.d_poles(e)
        p = basin_logp(d, net.temperature).exp().cpu().numpy()
        yv_b = y_all[vi]
        conf = p.max(1); pred = p.argmax(1); correct = (pred == yv_b).astype(np.float64)
        # Expected Calibration Error (equal-width bins on confidence), fully vectorized.
        b = np.clip(np.digitize(conf, np.linspace(0, 1, n_bins + 1)[1:-1]), 0, n_bins - 1)
        cnt = np.bincount(b, minlength=n_bins).astype(np.float64)
        nz = cnt > 0
        ece = float((cnt[nz] / cnt.sum() * np.abs(
            np.bincount(b, correct, n_bins)[nz] / cnt[nz]
            - np.bincount(b, conf, n_bins)[nz] / cnt[nz])).sum())
        # Terminal effective rank: the DIRECT test that the many mate structures did not collapse
        # into one point on the shell (Kaveh: "I don't want all mates to be one point").
        qi = term_train[rng.integers(0, len(term_train), min(3000, len(term_train)))]
        term_er = float(eff_rank(net.phi(fx(qi)).cpu().numpy()))
        # Shell radius: do terminals actually land ~1 ply from their own pole?
        d_q = net.d_poles(net.phi(fx(qi)))
        shell = float(d_q.gather(1, y_t[qi].unsqueeze(1)).squeeze(1).median())
        # Asymmetry actually achieved: median log1p(d(P->s)) - log1p(d(s->P)). Must be > 0, else
        # the field has quietly learned a symmetric metric and the basin flow means nothing.
        asym = float((torch.log1p(net.d_from_poles(e)) - torch.log1p(d)).median())
        # Where the vertices actually sit relative to the potential's crossover: pole_gap should
        # settle AT or just above pole_ref (repelled to the crossover, weakly held from beyond).
        # START-pole gates. start_rho is THE test that the ply coordinate is real: does
        # d(P_start -> s) actually track ply on held-out games?
        ds_fwd = net.d_from_start(e).cpu().numpy()
        from scipy.stats import spearmanr as _sp
        start_rho = float(_sp(ds_fwd, z_ply[vi]).correlation)
        start_back = float(torch.log1p(net.d_to_start(e)).median())
        pdist = net.d_poles_pairwise()
        d_start_out = [float(pdist[START, k]) for k in (WIN, DRAW, LOSS)]
        # pole_gap measures the OUTCOME triangle only. The start pole's distances are reported
        # separately (start_to_*); folding a time origin into the basin-separation number would
        # make it mean nothing.
        pd = pdist[:3, :3]
        offd = ~torch.eye(3, dtype=torch.bool, device=pd.device)
        pole_gap = float(torch.log1p(pd[offd]).median())
        pole_ref = float(typical_pair_scale(net.d_pair_emb(e, e[torch.randperm(len(e), device=dev)])))
        u = torch.log1p(d.gather(1, y_t[vi].unsqueeze(1)).squeeze(1))
        w = [float(u[torch.from_numpy((yv_b == k)).to(dev)].std()) if (yv_b == k).sum() > 8
             else float("nan") for k in (WIN, DRAW, LOSS)]
        return {"start_rho": start_rho, "start_back": start_back,
                "start_to_win": d_start_out[0], "start_to_draw": d_start_out[1],
                "start_to_loss": d_start_out[2],
                "width_win": w[0], "width_draw": w[1], "width_loss": w[2],
                "width_spread": float(np.nanmax(w) - np.nanmin(w)),
                "basin_acc": float(correct.mean()), "basin_ece": ece, "basin_conf": float(conf.mean()),
                "ambiguous_frac": float((conf < 0.5).mean()), "term_eff_rank": term_er,
                "shell_median": shell, "pole_asym": asym, "T": float(net.temperature),
                "pole_gap": pole_gap, "pole_ref": pole_ref}

    step_log_path = (f"{out}_steps.jsonl" if args.step_log == "auto" else args.step_log)
    step_log = None
    if step_log_path and poles_on:
        Path(step_log_path).parent.mkdir(parents=True, exist_ok=True)
        step_log = open(step_log_path, "a" if args.resume else "w")
        print(f"  [step-log] per-step train+holdout error -> {step_log_path} "
              f"(val batch {args.step_val_batch})", flush=True)

    _extra = {"cfg": {"in_ch": C, "d": args.d, "components": args.components,
                      "adapter_ch": args.adapter_ch, "trunk": tag}}
    start_step = 0
    if args.resume:
        _ck = torch.load(args.resume, map_location=dev, weights_only=False)
        _miss, _unexp = net.load_compat(_ck["state_dict"])
        start_step = int(_ck.get("step", 0))
        if "opt_state" in _ck:
            opt.load_state_dict(_ck["opt_state"])
            _os = "optimizer state restored"
        else:
            # Without Adam moments the first steps after a resume take a transient hit while the
            # moving averages rebuild. Stated rather than hidden -- it shows up as a brief bump in
            # the per-step loss right at the resume boundary.
            _os = "NO optimizer state in ckpt -> Adam moments restart (brief transient expected)"
        print(f"  [resume] {args.resume} @ step {start_step:,} | {_os}"
              + (f" | missing {_miss}" if _miss else ""), flush=True)

    cfg = TrainConfig(out=out, steps=args.steps, start_step=start_step, ckpt_every=args.ckpt_every, eval_every=args.eval_every,
                      experiment="catspace_m1_iqe_head", run_name=Path(out).name,
                      extra=_extra)
    try:
        last = standard_train(step, net, cfg, args=args, gates_fn=gates)
    finally:
        if step_log is not None:
            step_log.close()
    print(f"VERDICT M1-IQE-HEAD {tag}: pair-order {last.get('pair_order', float('nan')):+.3f} "
          f"(gate >=0.94) | d_mate rho {last.get('mate_rho', float('nan')):+.3f} (gate >=0.81) | "
          f"eff_rank {last.get('eff_rank', float('nan')):.1f} | [{time.time()-t0:.0f}s]", flush=True)
    if poles_on:
        print(f"VERDICT BASIN-POLES {tag}: acc {last.get('basin_acc', float('nan')):.3f} | "
              f"ECE {last.get('basin_ece', float('nan')):.4f} (primary gate, lower=better) | "
              f"mean conf {last.get('basin_conf', float('nan')):.3f} | ambiguous<0.5 "
              f"{last.get('ambiguous_frac', float('nan')):.3f} | terminal eff_rank "
              f"{last.get('term_eff_rank', float('nan')):.1f} (anti-collapse) | shell median "
              f"{last.get('shell_median', float('nan')):.2f} (target ~1 ply) | pole asymmetry "
              f"{last.get('pole_asym', float('nan')):+.2f} (must be >0) | pole gap "
              f"{last.get('pole_gap', float('nan')):.2f} vs crossover "
              f"{last.get('pole_ref', float('nan')):.2f} | T "
              f"{last.get('T', float('nan')):.3f}", flush=True)
        print(f"VERDICT START-POLE {tag}: ply rho {last.get('start_rho', float('nan')):+.3f} "
              f"(the ply coordinate is real iff this is high) | d(s->start) median "
              f"{last.get('start_back', float('nan')):.2f} (irreversibility, want large) | "
              f"d(start->win/draw/loss) {last.get('start_to_win', float('nan')):.0f}/"
              f"{last.get('start_to_draw', float('nan')):.0f}/"
              f"{last.get('start_to_loss', float('nan')):.0f} "
              f"(draw furthest => 'straight through is draw')", flush=True)


if __name__ == "__main__":
    main()
