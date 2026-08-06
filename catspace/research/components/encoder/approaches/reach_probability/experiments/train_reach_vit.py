#!/usr/bin/env python
"""train_reach_vit.py -- the FROM-SCRATCH reachability run: ViT over bitboards, two heads, every ply.

THE QUESTION. Can the irreversible stratification of chess -- total piece count only ever falls --
be learned from data, with nothing chess-specific programmed anywhere? The previous answer was
negative but INCONCLUSIVE, because it sat on the frozen lc0 trunk: a pretrained chess net already
contains the ratchet, so its random-init null (0.555) was not zero and the trained score (0.570)
could not be shown to have added anything. A randomly-initialised ViT over tokenized boards knows no
chess, so here the random-init null IS the real zero.

THE OBJECTIVE, in full. Kaveh 2026-08-05: "pushing unreachable pairs far, while keeping reachable
ones close". Every term is grounded in something the data actually contains -- nothing is spliced,
nothing is manufactured:

  OBSERVED-CLOSE   forward pairs (i->j) within a game. Arm A regresses the region NLL onto them;
                   arm B regresses log1p(d) onto log1p(ply gap).
  OBSERVED-CLOSE   REVERSIBLE pairs, from repetitions: if a position occurs twice in one game, every
                   row between the two occurrences genuinely reaches the earlier one. This is the
                   only backward evidence in the corpus and it is visible ONLY because we now keep
                   every ply -- the old 5-plies-per-game sample could not see a repetition at all.
  UNOBSERVED-FAR   the reversal (j->i) where NO repetition covers [i, j], and cross-game pairs.
                   Both carry a repulsion. Neither is a label: "the game did not walk back" is not
                   "walking back is impossible".

  >>> MEASUREMENT CAVEAT, stated here because it decides how the result may be read. The repulsion
  >>> acts UNIFORMLY on every unobserved reversal. So finding afterwards that "reverses are far"
  >>> proves nothing whatever -- it was trained in. The strata test must be the DIFFERENTIAL:
  >>> capture-crossing reversals against quiet REVERSIBLE ones, ply-matched. A uniform training
  >>> signal cannot manufacture a differential between two groups it treats identically; only the
  >>> data can. interpret_reach.py reads exactly that, plus the paired ratchet against a random-init
  >>> null, and every claim here is void without the random-init column beside it.

ANTI-COLLAPSE IS NOT OPTIONAL, and is applied at the TRUNK as well as the head: a collapse of phi is
a collapse of both arms at once, and with no manufactured negatives it is the silent failure mode
(constant encoder -> every region fits perfectly -> beautiful loss, nothing learned). vicreg
variance + covariance (the covariance term is the one that sees rank collapse, which the variance
term provably cannot), plus bootstrapped eff_rank gated every eval -- entropy-of-singular-values
form, deliberately not probe_rank.py's participation ratio; the repo has three definitions and
mixing them has burned us.

NO PAIR DATASET IS MATERIALISED. Every ply gives ~2,300 ordered pairs per game against 10 before, so
pairs are resampled fresh each step from the trajectory store. Splits are by GAME and enforced in
the sampler itself, so a training step cannot touch a calibration game -- if it could, the conformal
guarantee downstream would be void rather than merely weak.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from catspace.io import paths
from catspace.research.components.encoder.approaches.reach_probability.experiments.build_reach_pairs import (
    split_by_game)
from catspace.research.components.encoder.approaches.reach_probability.src import trajectories as T
from catspace.research.components.encoder.approaches.reach_probability.src.reach_vit_jepa import ReachViT
from catspace.research.components.encoder.approaches.reachability_field.experiments.arch_bakeoff import eff_rank
from catspace.research.tools.training_infra.losses import (
    basin_ce, basin_logp, pole_potential, quasimetric_regression, reach_region_margin,
    reach_region_nll, terminal_repulsion, typical_pair_scale, vicreg_covariance, vicreg_variance)
from catspace.research.tools.training_infra.train.scaffold import (
    TrainConfig, resolve_device, standard_train)


def _sync(dev):
    """Make a timing read MEANINGFUL. MPS and CUDA queue work asynchronously, so a bare
    time.perf_counter() around a kernel launch measures the launch, not the work -- every phase
    would look free and the total would land in whichever call happens to block. Syncing costs
    real time, which is why profiling runs only on eval steps."""
    if dev.type == "mps":
        torch.mps.synchronize()
    elif dev.type == "cuda":
        torch.cuda.synchronize()


class Phase:
    """Accumulates per-phase wall time so the log says WHAT is constraining the run.

    Without this, "0.58 s/step" is a single number with no lever attached: it could be the
    encoder, the target branch, the IQE pairwise work, or numpy pair sampling on the main thread,
    and each has a different fix. Logged as t_* metrics beside the losses."""

    def __init__(self, dev, on):
        self.dev, self.on, self.t = dev, on, {}
        self._name, self._t0 = None, None

    def __call__(self, name):
        self._name = name
        return self

    def __enter__(self):
        if self.on:
            _sync(self.dev)
            self._t0 = time.perf_counter()
        return self

    def __exit__(self, *a):
        if self.on:
            _sync(self.dev)
            self.t[f"t_{self._name}"] = self.t.get(f"t_{self._name}", 0.0) + time.perf_counter() - self._t0
        return False


def row_conditioning(tr, d_cond):
    """(N, d_cond) float32 strength+style vector per POSITION.

    Coordinate 0 is STRENGTH: the SIDE-TO-MOVE's REAL Elo, standardised. Human rows carry the
    actual lichess rating (measured range 1016..2318, median 1596); SF rows carry SF_ELO. Not a
    population flag -- that is what makes this "this 1400 vs this 2200" rather than "human vs
    engine", and it is what lets the same query ask about a specific opponent.

    Side-to-move, not a game average: the conditioning asks how likely THIS player is to err from
    here, and the player on move is the one about to make the mistake.

    The remaining coordinates are reserved for the per-player STYLE z and are zero for now, so the
    conditioning is honest about what it actually knows rather than pretending to a style it has
    not been given."""
    if not d_cond:
        return None
    c = np.zeros((tr.n_positions, d_cond), np.float32)
    c[:, 0] = T.normalise_elo(tr.elo_of_row())               # REAL rating of the side to move
    return c


def make_batcher(tr, dev, cond=None):
    """rows -> one encoder pass over the UNIQUE rows, then index back.

    The ViT is ~all of the step cost, and the triples/cross/reversible draws overlap heavily, so
    encoding unique rows once and gathering is a straight multiple off the wall clock (the
    optimize-before-long-runs rule). Returns (index_fn, encode_fn)."""
    def encode(net, rows_list):
        rows = np.concatenate(rows_list)
        uniq, inv = np.unique(rows, return_inverse=True)
        tok = torch.from_numpy(tr.tok[uniq].astype(np.int64)).to(dev)
        glob = torch.from_numpy(tr.glob[uniq].astype(np.float32)).to(dev)
        phi = net.backbone(tok, glob)
        z_a = net.proj_a(phi)
        if net.dual:
            c = torch.from_numpy(cond[uniq]).to(dev) if cond is not None else None
            z_b, z_r = net.qhead.embed(phi, c)      # (z_base, z_conditioned)
        else:
            z_b, z_r = net.proj_b(phi), None
        z_t = net.encode_target(tok, glob)
        cuts = np.cumsum([len(r) for r in rows_list])[:-1]
        idx = [torch.from_numpy(p.astype(np.int64)).to(dev) for p in np.split(inv, cuts)]
        return phi, z_a, z_b, z_r, z_t, idx
    return encode


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--games", type=int, default=200_000,
                    help="TOTAL games; split 1:1 human / SF-vs-SF (the matched-pool decision)")
    ap.add_argument("--max-plies", type=int, default=400)
    ap.add_argument("--no-cache", action="store_true", help="rebuild the trajectory cache")
    # encoder (reused unchanged from jepa_tokenizer -- 64 square tokens, 13-piece vocab, CLS -> phi)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--d", type=int, default=64, help="head dimension (both arms)")
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--components", type=int, default=16,
                    help="IQE components; must divide --d. 16 is the value the working field head "
                         "uses (train_iqe_head.py), not a guess")
    ap.add_argument("--ema", type=float, default=0.996)
    ap.add_argument("--dual", action="store_true",
                    help="best-play FLOOR + strength/style conditioned residual. Off = the legacy "
                         "single-IQE arm B, kept loadable for checkpoints already on disk")
    ap.add_argument("--d-cond", type=int, default=1,
                    help="conditioning width. 1 = ELO ONLY (Kaveh 2026-08-05: 'lets leave the "
                         "style flavor for later where we have lots of data. elo-conditioning is "
                         "fine'). Widen it when a per-player style z exists to put in coords 1+")
    ap.add_argument("--init-from", default=None,
                    help="STAGE 2: warm-start from the unconditional pooled checkpoint. The legacy "
                         "arm-B weights map exactly onto the conditioned head (proj_b -> proj_base, "
                         "iqe -> iqe), so part 1 is reused rather than redone")
    ap.add_argument("--freeze-base", action="store_true",
                    help="STAGE 2: freeze encoder + proj_base + iqe so ONLY the Elo delta learns. "
                         "This is what makes the fine-tune a clean residual on a fixed field -- "
                         "without it the base drifts and 'deviation from the pooled field' stops "
                         "meaning anything measured against part 1")
    ap.add_argument("--w-res-shrink", type=float, default=0.1,
                    help="shrinkage on ||delta||, the conditioned displacement. Keeps the pooled "
                         "base carrying everything common so a population deviates only where it "
                         "really differs. Expresses NO preference about SIGN: moving closer than "
                         "the pooled field is as cheap as moving further")
    ap.add_argument("--w-basin", type=float, default=1.0,
                    help="THE probabilistic-reachability readout: cross-entropy of "
                         "softmax(-log1p(d(s->pole))) against the game's OBSERVED outcome, over "
                         "three W/D/L poles. Proper scoring rule, so its minimiser is the true "
                         "P(outcome|s) -- a position seen in wins 60%% of the time reads 60%%. "
                         "Verified on a planted committor: MAE 0.013. NEEDS NO NEGATIVES; the "
                         "competition is the softmax denominator. Raw attraction to the observed "
                         "pole instead COLLAPSES (measured: mean pole distance 0.000, readout "
                         "uniform) because a quasimetric permits d(s->W)=d(s->D)=0 at once")
    ap.add_argument("--w-polesep", type=float, default=1.0,
                    help="pole-pole separation. Without it merged poles make every distance equal "
                         "and the CE sits at log 3 forever -- a saddle, not a minimum")
    ap.add_argument("--min-ply", type=int, default=0,
                    help="drop the first plies of every game. MUST be 8 for dynamics-conditioned "
                         "runs: SF games are human ply-8 opening prefixes + SF continuation, so "
                         "plies 0..7 of an 'SF' game are HUMAN moves. 0 is correct for the pooled "
                         "strata question, which never reads the source label")
    # objective weights
    ap.add_argument("--w-region", type=float, default=1.0, help="arm A: region NLL on observed pairs")
    ap.add_argument("--w-iqe", type=float, default=1.0, help="arm B: quasimetric regression")
    ap.add_argument("--w-repel", type=float, default=1.0, help="both arms: unobserved-pair repulsion")
    ap.add_argument("--margin", type=float, default=1.0,
                    help="arm A repulsion margin: nats of NLL an unobserved target must exceed the "
                         "observed one of the SAME source by. Relative on purpose -- a Gaussian "
                         "log-density has no absolute scale to floor")
    ap.add_argument("--repel-margin", type=float, default=4.0,
                    help="arm B repulsion floor on log1p(d) for UNOBSERVED pairs, so d >= e^m-1 "
                         "(4.0 -> ~53.6). ABSOLUTE, not relative: a relative margin is satisfied "
                         "by d_close=0.001/d_far=0.01 with the geometry fully collapsed, which is "
                         "the documented IQE failure mode. 4.0 is train_iqe_head.py's measured "
                         "value, not a guess")
    ap.add_argument("--w-var", type=float, default=1.0, help="VICReg variance (anti-collapse)")
    ap.add_argument("--w-cov", type=float, default=0.04, help="VICReg covariance (anti-rank-collapse)")
    ap.add_argument("--w-l1", type=float, default=0.0,
                    help="L1 added to the loss (subgradient route; monitoring only -- measured NOT "
                         "to sparsify under Adam, which is why --l1-prox exists)")
    ap.add_argument("--l1-prox", type=float, default=0.005,
                    help="proximal soft-threshold (ISTA) on the predictor input layer. SWEEP THIS. "
                         "At 2.0 on the trunk model it zeroed the whole layer (support 0/64), i.e. "
                         "it built the source-blind control instead of a model")
    ap.add_argument("--batch", type=int, default=192, help="triples per step (~4 encoder rows each)")
    ap.add_argument("--rev-frac", type=float, default=0.25, help="reversible pairs, as a fraction of --batch")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--steps", type=int, default=20_000)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--ckpt-every", type=int, default=2500)
    ap.add_argument("--val-frac", type=float, default=0.1, help="held-out slice of TRAIN games; the "
                    "calibration split is left untouched so the conformal guarantee stays honest")
    ap.add_argument("--random-init", action="store_true",
                    help="write the ckpt and STOP -- this is THE null. A from-scratch ViT knows no "
                         "chess, so its ratchet score is the real zero the trained one must beat")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=paths.experiment("reach_vit_v1"))
    args = ap.parse_args()

    t0 = time.time()
    dev = resolve_device(args.device)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    n_each = args.games // 2
    tr = T.build(n_human=n_each, n_sf=n_each, seed=args.seed, cache=not args.no_cache,
                 max_plies=args.max_plies)
    cov, reps = tr.coverage(), tr.repeats()
    print(f"[traj] {len(tr):,} games | {tr.n_positions:,} positions | {len(reps[0]):,} repetitions "
          f"in {len(np.unique(tr.game_of_row()[reps[0]])):,} games [{time.time()-t0:.0f}s]", flush=True)

    # SPLIT BY GAME, enforced inside the sampler. Calibration (1) and test (2) are never constructed
    # here, so no training step can see them -- the conformal guarantee depends on that, and on
    # hyperparameters not being chosen on them either.
    split = split_by_game(np.arange(len(tr)), (0.70, 0.15), args.seed)
    tr_games = np.flatnonzero(split == 0)
    rng.shuffle(tr_games)
    n_val = max(1, int(len(tr_games) * args.val_frac))
    val_games, fit_games = tr_games[:n_val], tr_games[n_val:]
    print(f"[split] fit {len(fit_games):,} | val {len(val_games):,} games "
          f"(cal {(split==1).sum():,} / test {(split==2).sum():,} held back)", flush=True)

    if args.dual and args.min_ply < 8:
        print(f"[warn] --dual with --min-ply {args.min_ply}: plies 0..7 of SF games are HUMAN "
              f"opening moves, so 'best play' would be fitted partly on human data. Use 8.",
              flush=True)
    net = ReachViT(d_model=args.d_model, layers=args.layers, heads=args.heads, d=args.d,
                   hidden=args.hidden, components=args.components, ema_decay=args.ema,
                   dual=args.dual, d_cond=args.d_cond if args.dual else 0).to(dev)
    cfg_dict = {"arch": "vit", "d_model": args.d_model, "layers": args.layers, "heads": args.heads,
                "d": args.d, "hidden": args.hidden, "components": args.components,
                "games": args.games, "max_plies": args.max_plies, "traj_seed": args.seed,
                "dual": bool(args.dual), "d_cond": args.d_cond if args.dual else 0,
                "min_ply": args.min_ply}

    if args.init_from:
        pay = torch.load(args.init_from, map_location=dev, weights_only=False)
        sd = pay["state_dict"]
        # Map the unconditional (legacy) arm-B weights onto the conditioned head. proj_delta is NOT
        # in the source and stays zero-init, so the fine-tune starts EXACTLY at the pooled field.
        remap, dropped = {}, []
        for k, v in sd.items():
            if k.startswith("proj_b."):
                remap["qhead.proj_base." + k.split(".", 1)[1]] = v
            elif k.startswith("iqe."):
                remap["qhead.iqe." + k.split(".", 1)[1]] = v
            else:
                remap[k] = v
        missing, unexpected = net.load_state_dict(remap, strict=False)
        got = [m for m in missing if not m.startswith("qhead.proj_delta")]
        print(f"[stage2] warm-started from {args.init_from} step {pay.get('step')} | "
              f"delta stays zero-init | unmapped-missing {len(got)} | unexpected {len(unexpected)}",
              flush=True)
        if args.freeze_base:
            for mod in (net.enc, net.t_enc, net.proj_a, net.qhead.proj_base, net.qhead.iqe):
                for q in mod.parameters():
                    q.requires_grad_(False)
            n_tr = sum(q.numel() for q in net.parameters() if q.requires_grad)
            print(f"[stage2] base FROZEN -- {n_tr:,} trainable params (the Elo delta only)",
                  flush=True)

    if args.random_init:
        from catspace.research.tools.training_infra.train.scaffold import save_torch_ckpt
        p = save_torch_ckpt(net, args.out + "_randinit", 0, args=args, extra={"cfg": cfg_dict})
        print(f"\nVERDICT REACH-VIT-RANDINIT out={p} -- THE null. Score it with interpret_reach.py "
              f"and read every trained number against it. [{time.time()-t0:.0f}s]")
        return

    fit = T.PairSampler(tr, fit_games, seed=args.seed, cov=cov, repeats=reps, min_ply=args.min_ply)
    val = T.PairSampler(tr, val_games, seed=args.seed + 1, cov=cov, repeats=reps,
                        min_ply=args.min_ply)
    opt = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=args.lr)
    # THREE OUTCOME POLES (Kaveh 2026-08-05: "I want three poles (win draw loss) ... and positions
    # orienting themselves towards it, trying to answer the probabilistic reachability question
    # from seeing pairs of positions in game that follow each other, without any negatives").
    outcome = tr.outcome_of_row()
    net.attach_poles([-1, -1, -1], n_sources=1)
    net.poles = net.poles.to(dev)
    print(f"[poles] 3 W/D/L poles | labels W {int((outcome==T.WIN).sum()):,} "
          f"D {int((outcome==T.DRAW).sum()):,} L {int((outcome==T.LOSS).sum()):,} "
          f"| censored (time forfeit) {int((outcome<0).sum()):,}", flush=True)
    cond = row_conditioning(tr, args.d_cond if args.dual else 0)
    encode = make_batcher(tr, dev, cond)
    n_rev = max(1, int(args.batch * args.rev_frac))

    def terms(model, sampler, n, n_rev, ph=None):
        """One objective evaluation on freshly sampled pairs. Returns (loss, metrics)."""
        ph = ph or Phase(dev, False)
        with ph("sample"):
            i, j, k = sampler.triples(n)
            x = sampler.cross(i)                        # one cross-game partner PER SOURCE i
            ra, rb, rgap = sampler.reversible(n_rev)    # OBSERVED backward, via repetitions
        rows = [i, j, k, x, ra, rb]
        with ph("encode"):
            phi, zA, zB, zR, zT, idx = encode(model, rows)
        gi, gj, gk, gx, gra, grb = idx
        # In dual mode zR is the CONDITIONED embedding (z_base + delta), so every distance is the
        # player-tuned quasimetric; zB stays available as the pooled reference for the readout.
        DQ = (lambda u, v: model.qhead.distance(zR[u], zR[v])) if model.dual \
            else (lambda u, v: model.distance(zB[u], zB[v]))
        gap_ij = torch.from_numpy((j - i).astype(np.float32)).to(dev)
        gap_jk = torch.from_numpy((k - j).astype(np.float32)).to(dev)
        gap_ik = torch.from_numpy((k - i).astype(np.float32)).to(dev)
        gap_r = torch.from_numpy(rgap.astype(np.float32)).to(dev)
        # The reversal (j -> i) is UNOBSERVED only where no repetition covers [i, j]. Where one
        # does, the game itself walked back and the pair is evidence of REVERSIBILITY, so it must
        # be excluded from the repulsion rather than pushed apart -- that exclusion is the only
        # reason the capture-crossing vs quiet-reversible differential can exist at all.
        unc_np = sampler.uncovered(i, j)
        unc = torch.from_numpy(unc_np.astype(np.float32)).to(dev)
        um = torch.from_numpy(np.flatnonzero(unc_np)).to(dev)

        # ---- arm A: the region -----------------------------------------------------------------
        mu_i, ls_i = model.predict(zA[gi])
        mu_j, ls_j = model.predict(zA[gj])
        l_nll = (reach_region_nll(mu_i, ls_i, zT[gj]) + reach_region_nll(mu_j, ls_j, zT[gk])
                 + reach_region_nll(mu_i, ls_i, zT[gk])) / 3.0
        if len(ra):
            mu_r, ls_r = model.predict(zA[gra])
            l_nll = 0.75 * l_nll + 0.25 * reach_region_nll(mu_r, ls_r, zT[grb])
        # PAIRED repulsion: same source, observed target as the reference (see reach_region_margin).
        # source j: its real future k is close, its own past i is the unobserved reversal.
        rev_a = (reach_region_margin(mu_j[um], ls_j[um], zT[gk[um]], zT[gi[um]], args.margin)
                 if len(um) else torch.zeros((), device=dev))
        crs_a = reach_region_margin(mu_i, ls_i, zT[gj], zT[gx], args.margin)
        l_rep_a = 0.5 * (rev_a + crs_a)

        # ---- arm B: the quasimetric ------------------------------------------------------------
        d_ij, d_jk, d_ik = DQ(gi, gj), DQ(gj, gk), DQ(gi, gk)
        l_q = (quasimetric_regression(d_ij, torch.log1p(gap_ij))
               + quasimetric_regression(d_jk, torch.log1p(gap_jk))
               + quasimetric_regression(d_ik, torch.log1p(gap_ik))) / 3.0
        if len(ra):
            l_q = 0.75 * l_q + 0.25 * quasimetric_regression(DQ(gra, grb), torch.log1p(gap_r))
        d_ji = DQ(gj, gi)           # the reversal, on the SAME two positions
        d_x = DQ(gi, gx)            # cross-game, from the SAME source i
        # ABSOLUTE log-margin repulsion -- relu(margin - log1p(d)) -- NOT a relative pairwise
        # margin. This is the single most load-bearing lesson in the repo's IQE history and the
        # first draft of this file got it wrong.
        #
        # A relative margin ("the unobserved pair must be further than this observed one") is
        # satisfied exactly by d_close=0.001, d_far=0.01: the ORDER is right and the whole
        # geometry is still collapsed. That is the documented IQE failure mode -- QRL's own words,
        # "the quasimetric could remain arbitrarily small everywhere" -- and it is why pure-ranking
        # InfoNCE trains MRN fine but leaves IQE flat: MRN's bilinear score does not need large
        # absolute distances, IQE's union-of-interval lengths do. Only an absolute floor sets the
        # SCALE, and with margin 4.0 an unobserved pair must sit at d >= e^4-1 ~ 53.6.
        #
        # This is the form that WORKS here: train_iqe_head.py's L_repel at repel_margin 4.0,
        # w_repel 1.0, which on 2026-08-02 beat the shipped M1 field on every metric (pair-order
        # +0.926, d_mate rho +0.818, eff_rank 18.8). Paired with plain ply-gap regression -- so no
        # QRL rewrite is needed; the repulsion was always the missing piece, not the objective.
        um_b = torch.nonzero(unc, as_tuple=True)[0]
        rep_rev = (terminal_repulsion(d_ji[um_b], args.repel_margin)
                   if len(um_b) else torch.zeros((), device=dev))
        l_rep_b = 0.5 * (rep_rev + terminal_repulsion(d_x, args.repel_margin))

        # ---- the three W/D/L poles: probabilistic reachability, positives only -----------------
        # d(s -> P_k) for k = win/draw/loss, then CE against the outcome the game ACTUALLY reached.
        # Censored (time-forfeit) rows are dropped: a flagged position may be dead winning, so its
        # result says nothing about the board.
        l_basin = torch.zeros((), device=dev)
        l_polesep = torch.zeros((), device=dev)
        if args.w_basin > 0 and model.poles is not None:
            y_np = outcome[np.concatenate([i, j, k])]
            live = np.flatnonzero(y_np >= 0)
            if len(live) > 8:
                zrows = torch.cat([zB[gi], zB[gj], zB[gk]], 0)[
                    torch.from_numpy(live.astype(np.int64)).to(dev)]
                y = torch.from_numpy(y_np[live].astype(np.int64)).to(dev)
                P = model.poles.poles                                   # (3, d)
                dP = torch.stack([model.qhead.d_base(zrows, P[c].expand(len(zrows), -1))
                                  if model.dual else model.iqe(zrows, P[c].expand(len(zrows), -1))
                                  for c in range(3)], 1)                # (B,3)
                l_basin = basin_ce(dP, y)
                # Poles must not merge: identical poles make every distance equal and the CE sits
                # at log 3 forever. LJ-shaped potential, same term the field head uses.
                pd = torch.stack([(model.qhead.d_base(P[a:a+1], P[b:b+1])[0] if model.dual
                                   else model.iqe(P[a:a+1], P[b:b+1])[0])
                                  for a in range(3) for b in range(3)]).view(3, 3)
                l_polesep = pole_potential(pd, typical_pair_scale(d_ij.detach()))

        # ---- anti-collapse, at the TRUNK and at the region head --------------------------------
        # zB is included on measurement, not on principle: the first smoke had eff_rank_zB DROP
        # 7.8 -> 5.0/64 over 400 steps while phi and zA rose. Arm B's only spreading pressure is a
        # two-direction hinge, which is far too thin a repulsion to hold 64 axes apart -- and the
        # cure for rank collapse is repulsion, not width.
        # RESIDUAL SHRINKAGE. With every row (engine included) on base+residual, d_res >= 0 alone
        # is satisfied by d_base = 0 with the residuals carrying everything, so the base would be
        # identified only up to SOME lower bound and would drift toward zero. Penalising residual
        # magnitude makes it the TIGHT lower envelope: explain what you can with the shared base,
        # spend residual only where populations genuinely differ. Without this the whole
        # best-play/mistake decomposition is unidentified.
        # SHRINKAGE ON ||delta||. The pooled base should carry everything common; a population
        # deviates only where it genuinely differs. Crucially this expresses NO PREFERENCE ABOUT
        # SIGN -- a stronger player moving CLOSER than the pooled field is exactly as cheap as a
        # weaker one moving further. The earlier Elo-anchored-to-zero term got this wrong: it
        # assumed the base was best play and pushed the top of the rating scale toward zero
        # residual, when a 3500 should sit BELOW the pooled field.
        l_shrink = (zR - zB).norm(dim=-1).mean() if model.dual else torch.zeros((), device=dev)
        # ELO-ANCHORED shrinkage: re-embed the SAME positions at the TOP of the strength scale and
        # push their residual to zero. Best play needs no correction, by definition -- and since
        # human rows simultaneously require a positive residual, the head must express that
        # difference through the strength coordinate. Uniform shrinkage alone cannot do this: it
        # pushes every population's residual down equally and leaves the conditioning unused.
        l_res_elo = torch.zeros((), device=dev)
        l_var = vicreg_variance(phi) + vicreg_variance(zA) + vicreg_variance(zB)
        l_cov = vicreg_covariance(phi) + vicreg_covariance(zA) + vicreg_covariance(zB)
        l_l1 = model.l1_penalty()
        loss = (args.w_region * (l_nll + args.w_repel * l_rep_a)
                + args.w_iqe * (l_q + args.w_repel * l_rep_b)
                + args.w_var * l_var + args.w_cov * l_cov + args.w_l1 * l_l1
                + args.w_res_shrink * l_shrink
                + args.w_basin * l_basin + args.w_polesep * l_polesep)
        met = {"loss": float(loss.detach()), "nll": float(l_nll.detach()),
               "rep_a": float(l_rep_a.detach()), "quasi": float(l_q.detach()),
               "rep_b": float(l_rep_b.detach()), "var": float(l_var.detach()),
               "cov": float(l_cov.detach()), "l1": float(l_l1.detach()),
               # the residue the cross sampler's redraw loop could not clear -- must be ~0, and is
               # printed rather than assumed (a same-game "cross" pair would be a false negative)
               "uncovered_frac": float(unc.mean()),
               "d_fwd": float(d_ij.detach().mean()), "d_rev": float(d_ji.detach().mean()),
               "d_far": float(d_x.detach().mean()), "res_shrink": float(l_shrink.detach()),
               # THE DEAD-ZONE GATE. The IQE ordering collapse (all F above all B in the interval
               # coordinates => every directed distance exactly 0) is an ABSORBING fixed point:
               # torch.maximum has zero gradient there, so nothing escapes once it arrives. It
               # halted qrl_iqe_unreach at 2k. We now aim deliberately NEAR that surface, because
               # subsumption wants exact zeros, so the distinction between "a few intended zeros"
               # and "everything collapsed" has to be a logged number rather than an assumption.
               "zero_frac": float((d_x.detach() < 1e-6).float().mean()),
               # sigma is the region's own width. Driven to the LOG_SIGMA_MIN clamp it means the
               # model claims certainty about the future, which is false and makes the conformal
               # tail meaningless -- so it is watched, not just clamped.
               "sigma_med": float(torch.exp(ls_i.detach()).median()),
               "cross_same_game": float((sampler.game_of_row[i] == sampler.game_of_row[x]).mean())}
        return loss, met, (phi, zA, zB)

    def step_fn(model, step):
        # Profile only on eval steps: _sync() serialises the device and would otherwise tax every
        # step to measure it (observer effect on the very number we are optimising).
        prof = (step % args.eval_every == 0) or step == 1
        ph = Phase(dev, prof)
        loss, met, _ = terms(model, fit, args.batch, n_rev, ph)
        with ph("backward"):
            opt.zero_grad(set_to_none=True)
            loss.backward()
        with ph("opt"):
            opt.step()
            if args.l1_prox > 0:
                model.prox_l1(args.lr * args.l1_prox)   # ISTA step: what actually zeroes coords
            model.update_target()
        if prof:
            tot = sum(ph.t.values()) or 1.0
            met.update(ph.t)
            met["t_step"] = tot
            # ~4 encoder rows per triple (i, j, k, x); the true figure is the unique-row count,
            # which is slightly lower because the draws overlap.
            met["rows_per_s"] = args.batch * 4 / tot
            # share of the step each phase owns -- the lever, not just the cost
            met.update({f"pct_{k[2:]}": v / tot for k, v in ph.t.items()})
        return met

    def gates_fn(model):
        model.eval()
        with torch.no_grad():
            _, met, (phi, zA, zB) = terms(model, val, min(args.batch, 256), n_rev)
        import resource
        met["rss_gb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (2**30 if
                        __import__('sys').platform == 'darwin' else 2**20)
        g = {f"val_{k}": v for k, v in met.items()}
        # Collapse gates: entropy-of-singular-values eff_rank (arch_bakeoff form), on the trunk AND
        # both heads -- a trunk collapse takes both arms down together and must be visible as a
        # number in the log rather than as a suspiciously good loss.
        for name, z in (("phi", phi), ("zA", zA), ("zB", zB)):
            g[f"eff_rank_{name}"] = eff_rank(z.detach().float().cpu().numpy())
        g["z_std"] = float(zA.std())
        if model.poles is not None:
            # THE COLLAPSE SIGNATURE for the three-pole readout: if positions slide "above" all
            # three poles at once (legal in a quasimetric -- d(s->W)=d(s->D)=0 simultaneously),
            # every distance is 0 and P(W/D/L) is uniform everywhere. Measured on the planted
            # committor, pure attraction gave readout spread 0.000. So spread is gated, not hoped.
            P = model.poles.poles
            zr = zB[:min(512, len(zB))]
            dP = torch.stack([(model.qhead.d_base(zr, P[c].expand(len(zr), -1)) if model.dual
                               else model.iqe(zr, P[c].expand(len(zr), -1))) for c in range(3)], 1)
            p_wdl = basin_logp(dP).exp()
            g["basin_spread"] = float(p_wdl.std(0).mean())     # ~0 => uniform => collapsed
            g["d_pole_mean"] = float(dP.mean())
            g["pole_zero_frac"] = float((dP < 1e-6).float().mean())
        if model.dual:
            # THE CHECK THAT DECIDES WHETHER THE BASE IS BEST PLAY. Evaluate the SAME positions at
            # the SF conditioning point and at the human one. d_res@SF near 0 means the base really
            # is the floor; a large d_res@SF means best play still needs a correction, i.e. the base
            # is NOT best play and every human "mistake" number read against it is inflated by that
            # much. This is a prediction the design can fail, which is the point of conditioning
            # rather than masking.
            n_probe = min(256, len(phi))
            pp = phi[:n_probe]
            zb_p, _ = model.qhead.embed(pp)
            for tag, elo in (("sf", T.SF_ELO), ("2200", 2200), ("1400", 1400)):
                c = torch.zeros(n_probe, model.d_cond, device=pp.device)
                c[:, 0] = float(T.normalise_elo(elo))
                _, zc = model.qhead.embed(pp, c)
                # SIGNED: d(.|elo) - d_pooled. Negative means this player is CLOSER than the
                # pooled field, which is what a 3500 should look like.
                g[f"d_res_{tag}"] = float(model.qhead.residual(
                    zb_p, zb_p.flip(0), zc, zc.flip(0)).mean())
            # THE monotonicity the prior predicts: weaker players should carry a LARGER residual.
            g["res_gap"] = g["d_res_1400"] - g["d_res_sf"]
            g["res_monotone"] = float(g["d_res_1400"] >= g["d_res_2200"] >= g["d_res_sf"])
        # Sparsity as an EXACT count, which only means anything because prox_l1 makes true zeros.
        g["l1_support"] = int(model.input_support().sum())
        # DIRECT ratchet readout, free here: the observed forward distance against its own reversal.
        g["rev_ratio"] = float(met["d_rev"] / max(met["d_fwd"], 1e-6))
        model.train()
        return g

    cfg = TrainConfig(out=args.out, steps=args.steps, ckpt_every=args.ckpt_every,
                      eval_every=args.eval_every, experiment="reach_probability",
                      run_name=f"reach_vit_d{args.d_model}x{args.layers}_g{args.games}",
                      device=str(dev), extra={"cfg": cfg_dict})
    last = standard_train(step_fn, net, cfg, args=args, gates_fn=gates_fn)

    print(f"\nVERDICT REACH-VIT steps={args.steps} games={args.games} "
          f"val_nll={last.get('val_nll', float('nan')):.4f} "
          f"val_quasi={last.get('val_quasi', float('nan')):.4f} "
          f"eff_rank phi={last.get('eff_rank_phi', float('nan')):.1f}/{args.d_model} "
          f"zA={last.get('eff_rank_zA', float('nan')):.1f}/{args.d} "
          f"zB={last.get('eff_rank_zB', float('nan')):.1f}/{args.d} "
          f"l1_support={last.get('l1_support', -1)}/{args.d} "
          f"rev_ratio={last.get('rev_ratio', float('nan')):.2f} "
          f"out={args.out}_latest.pt [{time.time()-t0:.0f}s]")
    print("  rev_ratio is d(j->i)/d(i->j) on TRAINING pairs -- it is trained in and proves nothing "
          "on its own.\n  The verdict is interpret_reach.py's paired ratchet against the "
          "--random-init null, and the capture-crossing vs quiet-reversible DIFFERENTIAL.")


if __name__ == "__main__":
    main()
