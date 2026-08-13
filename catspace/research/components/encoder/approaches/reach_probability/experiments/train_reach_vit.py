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
import json
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
    absorbing_penalty, basin_ce, basin_logp, pole_potential, pole_radial_anchor,
    censored_plies_loss, code_persistence, confine_radius, confining_regression,
    fene_confinement, fene_r_max, first_hit_bce, jepa_code_prediction, lj_confinement,
    log_gas_repulsion, screened_repulsion,
    adjacent_anchor,
    start_irreversibility, start_ply_anchor, quasimetric_regression, reach_region_margin,
    pair_walls, terminal_contrast,
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
    def encode(net, rows_list, mirror=False):
        rows = np.concatenate(rows_list)
        uniq, inv = np.unique(rows, return_inverse=True)
        tk, gb = tr.tok[uniq], tr.glob[uniq]
        if mirror:
            from catspace.research.components.encoder.approaches.jepa_tokenizer.src.jepa import (
                mirror_arrays)
            tk, gb = mirror_arrays(tk, gb)
        tok = torch.from_numpy(tk.astype(np.int64)).to(dev)
        glob = torch.from_numpy(gb.astype(np.float32)).to(dev)
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
    ap.add_argument("--w-basin", type=float, default=10.0,
                    help="THE probabilistic-reachability readout: cross-entropy of "
                         "softmax(-log1p(d(s->pole))) against the game's OBSERVED outcome, over "
                         "three W/D/L poles. Proper scoring rule, so its minimiser is the true "
                         "P(outcome|s) -- a position seen in wins 60%% of the time reads 60%%. "
                         "Verified on a planted committor: MAE 0.013. NEEDS NO NEGATIVES; the "
                         "competition is the softmax denominator. Raw attraction to the observed "
                         "pole instead COLLAPSES (measured: mean pole distance 0.000, readout "
                         "uniform) because a quasimetric permits d(s->W)=d(s->D)=0 at once")
    ap.add_argument("--w-hinge-a", type=float, default=0.0,
                    help="decisive hinge on the length ruler: winner's exit must be closer "
                         "than loser's by hinge-eps, per-position credibility-weighted by the "
                         "outcome ruler (detached). Draw pole untouched.")
    ap.add_argument("--hinge-eps", type=float, default=0.5,
                    help="ordering margin in plies; < one tempo so real threat proximity "
                         "(mate-in-1 against the winner) is never denied")
    ap.add_argument("--w-pole-gas-a", type=float, default=0.0,
                    help="screened repulsion between batch rows and the 3 outcome poles on dA "
                         "-- the anchor's counter-force. Without it the anchor is attraction "
                         "without repulsion: poles slide into the cloud and dA(anything->pole) "
                         "collapses to ~2 (measured on reach_abs). With it, the triangle "
                         "ceiling d(s->own pole) <= remaining+1 binds and foreign poles float "
                         "out: the length-committor structure.")
    ap.add_argument("--w-anchor-a", type=float, default=0.0,
                    help="A-side ABSORBING pole anchor (Kaveh 2026-08-08: 'the pole should be "
                         "dominated by the mate before it, fixed one ply after'): "
                         "(dA(terminal -> its white-POV outcome pole) - 1)^2. Extends the walls "
                         "staircase one step past the terminals into the pole; A-ruler only, "
                         "committor calibration (dB) untouched. Requires white-POV basin.")
    ap.add_argument("--w-anchor", type=float, default=0.0,
                    help="terminal instances -> their ending pole, at that ending's certainty "
                         "radius (0 for rules and all draws, 1 ply for resignations)")
    ap.add_argument("--w-absorb", type=float, default=0.0,
                    help="d(pole -> instance) large: a finished game cannot be left. ABSORBING, "
                         "which is the half a symmetric metric could not express")
    ap.add_argument("--w-termrep", type=float, default=0.0,
                    help="distinct ENDING poles apart -- different arrival points of one surface")
    ap.add_argument("--w-subsume", type=float, default=0.0,
                    help="d(ending -> outcome) -> 0: every ending subsumes into its outcome, which "
                         "pins its distance while leaving its DIRECTION free")
    ap.add_argument("--absorb-margin", type=float, default=4.0)
    ap.add_argument("--termrep-margin", type=float, default=4.0)
    ap.add_argument("--n-term", type=int, default=64, help="terminal instances per step")
    ap.add_argument("--w-start", type=float, default=0.0,
                    help="d(start -> s) ~ ply: the START pole as a TIME ORIGIN, which is what pulls "
                         "the 20 first-move positions together instead of leaving them scattered")
    ap.add_argument("--w-start-irr", type=float, default=0.0,
                    help="d(s -> start) LARGE: you can never return to the opening. The sharpest "
                         "statement of irreversibility in the whole objective")
    ap.add_argument("--start-margin", type=float, default=4.0)
    ap.add_argument("--learned-poles", action="store_true",
                    help="let the three W/D/L poles move. DEFAULT IS FIXED: fixed poles pin the "
                         "gauge (a learned embedding has a global scale freedom, so no distance is "
                         "comparable across checkpoints), and they cannot merge, which removes the "
                         "uniform-committor saddle for free")
    ap.add_argument("--poles", default="contrastive", choices=["fixed", "contrastive"],
                    help="fixed = simplex gauge + anchors + trained basin CE. contrastive "
                         "(Kaveh 2026-08-07) = NO fixed poles/anchors/START/basin-CE; terminals "
                         "structure themselves: same-outcome attraction to a 1-ply shell, "
                         "opposite-outcome both-direction floor; committor is a READOUT vs "
                         "terminal exemplars, never a training target")
    ap.add_argument("--adjacent-frac", type=float, default=0.5,
                    help="fraction of triples that are STRICT adjacents (i,i+1,i+2): the local "
                         "signal blunders cannot fake; the rest keep random gaps <=40 for "
                         "mid-range composition")
    ap.add_argument("--term-margin", type=float, default=6.0,
                    help="contrastive: log-space floor between opposite-outcome terminals")
    ap.add_argument("--w-termcon", type=float, default=1.0,
                    help="contrastive: weight on the terminal attraction+repulsion pair")
    ap.add_argument("--move-head", type=int, default=0,
                    help="1 = move-aware delta head g(phi, e(m)) -> R^3, trained by three-pole "
                         "WDL-graded listwise (wdl_labels shards) + consistency vs actually-"
                         "encoded children. THE approach as of 2026-08-08 (Kaveh).")
    ap.add_argument("--w-mh", type=float, default=5.0, help="move-head WDL listwise weight")
    ap.add_argument("--w-mh-cons", type=float, default=1.0, help="move-head consistency weight")
    ap.add_argument("--mh-pos", type=int, default=8,
                    help="WDL-labeled positions fed to the move head per step (feeding knob)")
    ap.add_argument("--w-mh-bell", type=float, default=0.0,
                    help="Bellman on the MOVE-HEAD: min over moves of (d_par + delta)[mover win] "
                         "= d_par - 1, field detached. No bootstrap (nothing chases its own "
                         "estimate); also pins the head's absolute scale, which the listwise "
                         "loss leaves gauge-free (softmax shift-invariance). The corrected "
                         "descent mechanism after field-level --w-bellman failed its smoke "
                         "(semi-gradient calibrates parent levels but cannot reorder children).")
    ap.add_argument("--w-bellman", type=float, default=0.0,
                    help="Bellman residual on the mover's-win distance: |d(s->P_mover) - 1 - "
                         "min_m d(s.m->P_mover)|^2 over full legal child sets (wdl_labels "
                         "shards). Manufactures DESCENT along chosen lines -- the property "
                         "selfplay_probe measured at chance level. Requires --basin-pov white.")
    ap.add_argument("--bellman-pos", type=int, default=8,
                    help="positions per step for the Bellman residual")
    ap.add_argument("--mh-deadband", type=float, default=2.0,
                    help="consistency tolerance (plies): head may deviate from the field by this "
                         "much before the leash engages")
    ap.add_argument("--w-grad", type=float, default=0.0,
                    help="gradient supervision of the field: engine best-move vs alt-move ranking "
                         "on the B-ruler (needs bestmove_labels.npz); softplus ranking, scale-free")
    ap.add_argument("--split-head", type=int, default=0,
                    help="1 = two independent IQE blocks over one embedding: A = length "
                         "(walls/gas), B = outcome (basin CE readouts). One point, two rulers.")
    ap.add_argument("--sf-only", action="store_true",
                    help="train on Stockfish data only (balanced + piece-down); no human games")
    ap.add_argument("--n-piecedown", type=int, default=0,
                    help="piece-down SF-vs-SF games (gen_piecedown_sfsf.py) merged into the SF "
                         "pool: decisive optimal play reaching real endings -- the walls the "
                         "Option-B skeleton needs in the TB region")
    ap.add_argument("--mirror", type=int, default=0,
                    help="color-mirror involution: coin-flip per step, whole batch mirrored "
                         "consistently, W/L labels swapped (lc0/NNUE symmetry, our frame)")
    ap.add_argument("--w-qdistill", type=float, default=0.0,
                    help="TACTICAL DISTILLATION (2026-08-11 external-ladder verdict: 94 pct "
                         "vs random defense, ~4 pct vs any real tactics): CE of the "
                         "probability head toward FROZEN search-resolved committor labels "
                         "(gen_search_labels.py), weighted by the static/resolved gap so "
                         "training concentrates where the static field lies to itself. "
                         "Labels precomputed -- no moving-target bootstrap.")
    ap.add_argument("--qdistill-npz", default=None)
    # JOINT QUANTIZED TRAINING (Kaveh 2026-08-12: "jointly train all the way down to the concepts
    # and see if we can predict future concept activations", + "distance to concept activation by
    # the distance head, so two prediction targets"). See jqt.py for the module.
    ap.add_argument("--jqt", type=int, default=0, help="1 = joint quantized training on")
    ap.add_argument("--jqt-heads", type=int, default=8)
    ap.add_argument("--jqt-codes", type=int, default=64)
    ap.add_argument("--w-vqrec", type=float, default=3.0,
                    help="VQ reconstruction of the field's own six outputs (+ commit loss): "
                         "the faithfulness anchor, unchanged in role from concept_vq")
    ap.add_argument("--w-pers", type=float, default=1.0,
                    help="persistence prior: consecutive-ply pre-quant latents pulled together "
                         "where |dE| < pers-tau -- manufactures the metastability that makes "
                         "future-code prediction well-posed (and codes legible)")
    ap.add_argument("--pers-tau", type=float, default=0.05,
                    help="expected-points scale above which a move counts as an eval jump and "
                         "the persistence prior goes silent")
    ap.add_argument("--bf16", type=int, default=0,
                    help="jqt5: autocast bf16 forward+loss (the profiled lever); "
                         "smoke-validate the curve before any long run")
    ap.add_argument("--w-flip", type=float, default=0.0,
                    help="jqt5 hurdle dynamics: per-head flip BCE (replaces JEPA MSE; "
                         "'at every move there's a concept change -- predict which')")
    ap.add_argument("--w-dest", type=float, default=0.0,
                    help="jqt5 destination CE on FLIPPED heads only (mean-collapse-proof)")
    ap.add_argument("--dest-warmup", type=int, default=2500,
                    help="steps before the destination CE reaches full weight (index-churn "
                         "guard: code identities drift until the first refreshes settle)")
    ap.add_argument("--two-pole-cdb", type=int, default=0,
                    help="jqt5: CDB logit = contrast vs the anti-pole (negative poles)")
    ap.add_argument("--w-negpole", type=float, default=0.0,
                    help="jqt5: pull DEAD states (type fully captured) onto the anti-pole")
    ap.add_argument("--frac-dead", type=float, default=0.0,
                    help="fraction of goal samples drawn as dead-state negatives")
    ap.add_argument("--w-jepa", type=float, default=3.0,
                    help="future-code prediction in EMBEDDING space vs the EMA target branch "
                         "(never indices -- the codebook-churn escape)")
    ap.add_argument("--w-cda", type=float, default=2.0,
                    help="concept-goal LENGTH ruler: censored plies-to-first-activation on "
                         "dA(s -> codebook anchor)")
    ap.add_argument("--w-cdb", type=float, default=2.0,
                    help="concept-goal PROBABILITY ruler: first-hit BCE on a learned link of "
                         "dB(s -> codebook anchor) -> P(activate before game end)")
    ap.add_argument("--jqt-pairs", type=int, default=192,
                    help="corpus transitions (parent, move, child) per step")
    ap.add_argument("--jqt-goals", type=int, default=192,
                    help="activation-label rows per step (balanced pos/censored)")
    ap.add_argument("--jqt-refresh", type=int, default=2000,
                    help="steps between ActivationIndex refreshes (recode a game sample with "
                         "the EMA branch; labels then stand until the next refresh)")
    ap.add_argument("--jqt-refresh-games", type=int, default=1200,
                    help="games recoded per refresh")
    # JQT v2 (Kaveh 2026-08-12): square + piece concept vocabularies, residual on global
    ap.add_argument("--jqt-square-codes", type=int, default=0,
                    help=">0 = per-square shared VQ over the trunk's 64 square tokens")
    ap.add_argument("--jqt-piece-codes", type=int, default=0,
                    help=">0 = per-piece-slot shared VQ (identity tracked through the game)")
    ap.add_argument("--w-sqrec", type=float, default=2.0,
                    help="square-stream residual recon (+commit): explains what global missed")
    ap.add_argument("--w-pcrec", type=float, default=2.0,
                    help="piece-stream residual recon (+commit), after global+square")
    ap.add_argument("--w-pers-sq", type=float, default=1.0,
                    help="persistence on per-square latents (same square, quiet move)")
    ap.add_argument("--w-pers-pc", type=float, default=2.0,
                    help="persistence on per-slot latents: identity-matched across the move, "
                         "THE well-posed persistence space (the metastability bet)")
    ap.add_argument("--w-turnpair", type=float, default=0.0,
                    help="TURN-CONTRAST term (Kaveh 2026-08-12: 'side to move matters for "
                         "distance, and for probability'): null-move minimal pairs, both "
                         "rulers. dB: CE toward each member's search-resolved committor. "
                         "dA: tie-safe pairwise rank per decisive pole -- the member whose "
                         "resolved P(pole) is higher must be closer, one-tempo log-gap, no "
                         "push where search says the turn doesn't matter.")
    ap.add_argument("--turnpair-npz", default=None)
    ap.add_argument("--turnpair-n", type=int, default=48, help="pairs per step")
    ap.add_argument("--balance-npz", default=None,
                    help="stratified anchor-sampling weights (audit_data_balance --save): "
                         "phase x outcome x material inverse-frequency, TRAIN sampler only")
    # STREAMING INGEST (Kaveh 2026-08-12: "generate data on cpu while training also progresses
    # consuming that same data on gpu ... split each batch for training vs holdout"). The
    # filesystem is the queue: the generator flushes complete TSV lines; we poll the line count.
    ap.add_argument("--ingest-tsv", default=None,
                    help="stream new games from this growing TSV (gen_stratified_sfsf format) "
                         "into the live corpus")
    ap.add_argument("--ingest-every", type=int, default=2000,
                    help="steps between ingest polls")
    ap.add_argument("--ingest-min", type=int, default=50,
                    help="skip a poll with fewer than this many new complete games")
    ap.add_argument("--basin-pov", choices=("mover", "white"), default="mover",
                    help="basin CE label POV; 'white' is the gauge-unfrustrated choice "
                         "(2026-08-08 finding), 'mover' reproduces older recipes")
    ap.add_argument("--basin-temp", type=float, default=5.0,
                    help="basin softmax temperature on RAW distances (log-gas mode); ~the scale "
                         "of expected class differences in d")
    ap.add_argument("--anchor-form", default="quadratic",
                    choices=["capped", "quadratic", "quartic"],
                    help="anchor spring shape; force bracketing so far: Huber-1 lost, quartic-800 "
                         "crushed, capped-LJ plateaued mid-range; quadratic ~12 is the middle")
    ap.add_argument("--pole-height", type=float, default=750.0,
                    help="fixed-simplex block height -> pole separation. FIXED at construction, "
                         "never adaptive (Kaveh: 'it needs to just start at a reasonable level "
                         "and stay fixed' -- a widening gauge is a moving target). 300 puts "
                         "pole-pole IQE distance ~200, matching the basin scale: radial anchors "
                         "put positions on shells up to expm1(log1p(200))~200 from their pole, "
                         "so the old default 3.0 (pole-pole distance 2) had basins ~100x wider "
                         "than the gauge separating them. MEASURED (this arch, C=16): pole-pole "
                         "= 0.656*height; disjoint-basin criterion pole-pole >= 2.2*max shell "
                         "(462) gives height ~720; 750 adds margin")
    ap.add_argument("--w-polesep", type=float, default=0.0,
                    help="pole-pole separation. Without it merged poles make every distance equal "
                         "and the CE sits at log 3 forever -- a saddle, not a minimum")
    ap.add_argument("--min-ply", type=int, default=0,
                    help="drop the first plies of every game. MUST be 8 for dynamics-conditioned "
                         "runs: SF games are human ply-8 opening prefixes + SF continuation, so "
                         "plies 0..7 of an 'SF' game are HUMAN moves. 0 is correct for the pooled "
                         "strata question, which never reads the source label")
    # objective weights
    ap.add_argument("--w-region", type=float, default=1.0, help="arm A: region NLL on observed pairs")
    ap.add_argument("--log-gas", type=int, default=1,
                    help="1 = log-gas field (confining spring + unbounded pairwise repulsion + one-body "
                         "confinement); 0 = legacy Huber + relu hinge, for reproducing old runs")
    ap.add_argument("--confine", default="lj", choices=["fene", "lj", "quartic"],
                    help="confining spring shape: fene = one-sided, soft inside the true ply gap "
                         "and divergent at fene_stretch x it; lj = r^12 outside / r^6 inside, "
                         "normalised so the wall is at fene_stretch x the gap; "
                         "quartic = symmetric r^2+q*r^4")
    ap.add_argument("--lj-wall", type=int, default=0,
                    help="0 = raw r^12/r^6 on the log residual (self-scaling, default); "
                         "1 = normalise so the wall sits at fene_stretch x the observed ply gap")
    ap.add_argument("--fene-stretch", type=float, default=2.0,
                    help="the wall sits at this multiple of the OBSERVED ply gap (Kaveh: "
                         "'infinity at twice the observed distance')")
    ap.add_argument("--fene-soft", type=float, default=0.2,
                    help="stiffness on the CLOSER-than-equilibrium side (small by design)")
    ap.add_argument("--confine-quartic", type=float, default=1.0,
                    help="quartic weight in the confining spring r^2 + q*r^4; 0 = plain MSE")
    ap.add_argument("--confine-target", type=float, default=2.0,
                    help="one-body confinement target on log1p(|z|); sets the gas radius")
    ap.add_argument("--w-confine", type=float, default=0.0, help="weight on one-body confinement")
    ap.add_argument("--w-iqe", type=float, default=1.0, help="arm B: quasimetric regression")
    ap.add_argument("--w-repel", type=float, default=40.0, help="both arms: unobserved-pair repulsion")
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
    ap.add_argument("--w-var", type=float, default=2.0, help="VICReg variance (anti-collapse)")
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
    ap.add_argument("--ckpt-every", type=int, default=500,
                    help="500 gives 40 ladder rungs instead of 8 -- the effect curve is the\n"
                         "control that decides this experiment, so resolution on it is worth\n"
                         "the ~1.5 GB of checkpoints, and a crash costs 500 steps not 2500")
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
    rng_np = np.random.default_rng(args.seed + 7)
    torch.manual_seed(args.seed)

    n_each = args.games // 2
    _nh, _ns = (0, args.games) if args.sf_only else (n_each, n_each)
    tr = T.build(n_human=_nh, n_sf=_ns, seed=args.seed, cache=not args.no_cache,
                 max_plies=args.max_plies, n_piecedown=args.n_piecedown)
    cov, reps = tr.coverage(), tr.repeats()
    ply_of_row = tr.ply_of_row()
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
    # basin labels: white-POV kills the gauge frustration (see outcome_of_row_white);
    # mover-POV kept as the reproduction default for pre-2026-08-08 recipes.
    if args.w_bellman > 0 and args.basin_pov != "white":
        raise SystemExit("--w-bellman requires --basin-pov white (colour-fixed mover pole)")
    outcome = (tr.outcome_of_row_white() if args.basin_pov == "white"
               else tr.outcome_of_row())
    # TWO-LEVEL hierarchy: 3 fixed outcome poles + one LEARNED pole per ending type, each
    # subsuming into its outcome. Ending poles are DATA-DERIVED, so they must be built before the
    # model (attach_poles needs the parent vector) and before cfg_dict (which records it so
    # evaluation rebuilds the identical hierarchy).
    t_rows, t_pole, t_names, t_parent = tr.ending_poles()
    # outcome class (0 W / 1 D / 2 L) of each terminal row, via its ending-type's parent outcome
    TPOLE_OUT = np.array([t_parent[k] if t_parent[k] >= 0 else 0 for k in range(len(t_names))])
    TPOLE_OUT = np.where(TPOLE_OUT < 0, 0, TPOLE_OUT)
    t_radius = tr.radial_targets()
    # START POLE (Kaveh 2026-08-05, from the viewer: "the scattering of the starting positions makes
    # me think start positions need to cluster together near a pole"). A TIME ORIGIN, not a basin --
    # appended as its own root so the W/D/L simplex is untouched, and the basin softmax reads
    # poles[:3] only, so a start pole sitting near a position cannot steal outcome probability mass
    # (there is an explicit regression test for that in losses.py).
    #
    # Two terms, and together they are the sharpest statement of the ratchet we have:
    #   d(start -> s) ~ ply    you can reach a ply-N position in about N plies
    #   d(s -> start) LARGE    you can never get back to the opening
    # Material falling is a consequence of irreversibility; "you cannot return to the start" IS
    # irreversibility, and a quasimetric can hold both at once where a metric structurally cannot.
    START_IDX = len(t_names)
    t_names = list(t_names) + ["START"]
    t_parent = np.concatenate([t_parent, [-1]])

    print(f"[split] fit {len(fit_games):,} | val {len(val_games):,} games "
          f"(cal {(split==1).sum():,} / test {(split==2).sum():,} held back)", flush=True)

    if args.dual and args.min_ply < 8:
        print(f"[warn] --dual with --min-ply {args.min_ply}: plies 0..7 of SF games are HUMAN "
              f"opening moves, so 'best play' would be fitted partly on human data. Use 8.",
              flush=True)
    net = ReachViT(split_head=bool(args.split_head), move_head=bool(args.move_head),
                   d_model=args.d_model, layers=args.layers, heads=args.heads, d=args.d,
                   hidden=args.hidden, components=args.components, ema_decay=args.ema,
                   dual=args.dual, d_cond=args.d_cond if args.dual else 0).to(dev)
    cfg_dict = {"arch": "vit", "d_model": args.d_model, "layers": args.layers, "heads": args.heads,
                "d": args.d, "hidden": args.hidden, "components": args.components,
                "games": args.games, "max_plies": args.max_plies, "traj_seed": args.seed,
                "pole_parent": t_parent.tolist(), "pole_names": t_names,
                "pole_height": args.pole_height, "n_piecedown": args.n_piecedown, "split_head": bool(args.split_head), "move_head": bool(args.move_head), "sf_only": bool(args.sf_only), "learned_poles": bool(args.learned_poles),
                "dual": bool(args.dual), "d_cond": args.d_cond if args.dual else 0,
                "min_ply": args.min_ply,
                # FULL args (2026-08-08): couldn't audit mh2's basin weight from its ckpt --
                # instruments must be versioned with the model, loss weights included.
                "train_args": {k: v for k, v in vars(args).items()}}

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

    _bw = None
    if args.balance_npz:
        _bz = np.load(args.balance_npz)
        _bw = _bz["weight"]
        assert len(_bw) == tr.n_positions, \
            (f"balance weights are for a {len(_bw):,}-row corpus, "
             f"this one has {tr.n_positions:,}")
        print(f"[balance] stratified anchor sampling ON: {len(_bz['names'])} strata, "
              f"weight p50 {float(np.percentile(_bw, 50)):.2f} "
              f"p90 {float(np.percentile(_bw, 90)):.2f} (TRAIN sampler only; val untouched "
              f"so metrics stay comparable across runs)", flush=True)
    fit = T.PairSampler(tr, fit_games, seed=args.seed, cov=cov, repeats=reps,
                        min_ply=args.min_ply, row_weight=_bw)
    val = T.PairSampler(tr, val_games, seed=args.seed + 1, cov=cov, repeats=reps,
                        min_ply=args.min_ply)
    # ATTACH POLES BEFORE THE OPTIMIZER. Adam captures net.parameters() at construction, so
    # creating the poles afterwards left ending_delta out of the optimizer entirely: it stayed
    # EXACTLY zero for all 20k steps (verified in the v2 checkpoint, absmax 0.000000), every
    # ending pole sat on top of its outcome pole, and the anchor/absorb/repulsion terms trained
    # the ENCODER against poles that could never move. Ordering bug, silent, and it invalidated
    # the terminal-structure half of that run.
    if args.poles == "contrastive":
        _r0 = np.log1p(600.0)                  # typical init distance scale
        print("[forces] per-pair slopes at init scale (d~600): "
              f"term-attract {2*(_r0-0.7):.1f} x{args.w_termcon} on {args.n_term} pairs | "
              f"term-repel(hinge) {1.0:.1f} x{args.w_termcon} | "
              f"screened gas {args.w_repel/(601.0**2):.1e} | "
              f"spring(LJ) ~12r^11 at r~1: 12 x{args.w_iqe} on 3x{args.batch} legs | "
              f"vicreg hinge ~1 x{args.w_var}", flush=True)
        print("[poles] CONTRASTIVE mode: no fixed poles, no anchors, no trained basin; "
              f"{len(t_rows):,} terminals structure themselves", flush=True)
    if args.poles == "fixed":
        net.attach_poles(t_parent, n_sources=1, fixed=not args.learned_poles,
                         height=args.pole_height)

    # NOTE: the optimizer is created BELOW, after the jqt block -- Adam captures the parameter
    # list at construction (the take-2 scar), and jqt's parameters must be inside it.
    # THREE OUTCOME POLES (Kaveh 2026-08-05: "I want three poles (win draw loss) ... and positions
    # orienting themselves towards it, trying to answer the probabilistic reachability question
    # from seeing pairs of positions in game that follow each other, without any negatives").
    if net.poles is not None:
        net.poles = net.poles.to(dev)
    print(f"[poles] {len(t_names)} poles ({'FIXED simplex' if (net.poles is not None and net.poles.fixed) else ('learned' if net.poles is not None else 'NONE - contrastive')} "
          f"+ {len(t_names)-3} ending types) | {len(t_rows):,} terminal instances | labels W {int((outcome==T.WIN).sum()):,} "
          f"D {int((outcome==T.DRAW).sum()):,} L {int((outcome==T.LOSS).sum()):,} "
          f"| censored (time forfeit) {int((outcome<0).sum()):,}", flush=True)
    cond = row_conditioning(tr, args.d_cond if args.dual else 0)
    encode = make_batcher(tr, dev, cond)
    n_rev = max(1, int(args.batch * args.rev_frac))

    # row hash (shared with the edge-exclusion structure) + per-row game ids, for the wall
    # exemptions and in-batch gas masks
    game_of_row = tr.game_of_row()
    src_of_game = tr.source
    qd = None
    if args.w_qdistill > 0:
        _qz = np.load(args.qdistill_npz)
        _gap = np.abs(_qz["e_static"] - _qz["e_resolved"])
        qd = {"row": _qz["row"], "probs": _qz["probs"],
              "w": (0.25 + _gap / max(1e-6, float(_gap.mean() * 4))).astype(np.float32)}
        print(f"[qdistill] {len(qd['row']):,} search-resolved labels; "
              f"mean gap {float(_gap.mean()):.3f}", flush=True)
    jqt = jqt_gt = jqt_idx = None
    if args.jqt:
        assert args.split_head and not args.dual and args.poles == "fixed", \
            "--jqt needs split_head dA/dB rulers, fixed poles, non-dual (the champion recipe)"
        from catspace.research.components.encoder.approaches.reach_probability.experiments.jqt import (
            ActivationIndex, JQTModule)
        _gtp = paths.derived(f"game_transitions_{len(tr.tok)}.npz")
        _gz = np.load(_gtp)
        # keep TRAIN-split transitions only (child = par+1 shares the parent's game)
        _tm = np.isin(game_of_row[_gz["par"]], np.flatnonzero(split == 0))
        jqt_gt = {"par": _gz["par"][_tm], "mid": _gz["mid"][_tm]}
        jqt = JQTModule(d_model=args.d_model, heads=args.jqt_heads, codes=args.jqt_codes,
                        d=args.d, ema=args.ema, square_codes=args.jqt_square_codes,
                        piece_codes=args.jqt_piece_codes).to(dev)
        st_of_game = None
        if args.jqt_piece_codes:
            assert tr.slots is not None, "piece stream needs the v5 corpus cache (slots)"
            st_of_game = np.zeros((len(tr), 32), np.uint8)
            for _g in range(len(tr)):
                _r0 = tr.start[_g]
                _sl0, _tk0 = tr.slots[_r0], tr.tok[_r0]
                _occ = _sl0 >= 0
                st_of_game[_g, _sl0[_occ]] = _tk0[_occ]
        jqt_idx = ActivationIndex(rng_np, codes=args.jqt_codes,
                                  sq_codes=args.jqt_square_codes,
                                  pc_codes=args.jqt_piece_codes)
        print(f"[jqt] ON: {args.jqt_heads}x{args.jqt_codes} codes | "
              f"{len(jqt_gt['par']):,} train transitions | refresh every "
              f"{args.jqt_refresh} steps ({args.jqt_refresh_games} games)", flush=True)

        @torch.no_grad()
        def jqt_refresh(model):
            """recode a game sample with the EMA branches -> first-activation index."""
            t0r = time.time()
            gsel = rng_np.choice(np.flatnonzero(split == 0), args.jqt_refresh_games,
                                 replace=False)
            gmask = np.isin(game_of_row, gsel)
            rws = np.flatnonzero(gmask)
            ids_all = np.empty((len(rws), args.jqt_heads), np.int16)
            sq_all = (np.empty((len(rws), 64), np.int8) if args.jqt_square_codes else None)
            pc_all = (np.empty((len(rws), 32), np.int8) if args.jqt_piece_codes else None)
            _ys = []
            _P3r = model.poles.poles[:3]
            for a in range(0, len(rws), 4096):
                rr = rws[a:a + 4096]
                tokb = torch.from_numpy(tr.tok[rr].astype(np.int64)).to(dev)
                globb = torch.from_numpy(tr.glob[rr].astype(np.float32)).to(dev)
                phi_t = model.t_enc(tokb, globb)
                _, idsb = jqt.target_codes(phi_t)
                ids_all[a:a + 4096] = idsb.cpu().numpy().astype(np.int16)
                if args.jqt_square_codes or args.jqt_piece_codes:
                    phi_on, sqt_on = model.enc.forward_tokens(tokb, globb)
                    if args.jqt_square_codes:
                        _, _, sqi, _ = jqt.square_stream(sqt_on)
                        sq_all[a:a + 4096] = sqi.cpu().numpy().astype(np.int8)
                    if args.jqt_piece_codes:
                        sl_b = torch.from_numpy(tr.slots[rr].astype(np.int64)).to(dev)
                        st_b = torch.from_numpy(
                            st_of_game[game_of_row[rr]].astype(np.int64)).to(dev)
                        _, _, pci, _, _ = jqt.piece_stream(sqt_on, sl_b, st_b)
                        pc_all[a:a + 4096] = pci.cpu().numpy().astype(np.int8)
                    zbr = model.proj_b(phi_on)
                else:
                    zbr = model.proj_b(model.enc(tokb, globb))
                DAr = torch.stack([model.dA(zbr, _P3r[[k]].expand(len(zbr), -1))
                                   for k in range(3)], 1)
                DBr = torch.stack([model.dB(zbr, _P3r[[k]].expand(len(zbr), -1))
                                   for k in range(3)], 1)
                _ys.append(torch.cat([DAr, torch.softmax(-DBr / args.basin_temp, 1)],
                                     1).detach())
            _Yr = torch.cat(_ys)
            # EXACT recon-normalization stats at EVERY refresh (2026-08-12, the freeze-at-50
            # scar: stale sd 0.003 over-weighted prob recon 60x and left decoded dA
            # untrained; play values through the decode were noise)
            jqt.y_mu.copy_(_Yr.mean(0)); jqt.y_sd.copy_(_Yr.std(0).clamp(min=1e-3))
            jqt.y_n.fill_(999)
            games = []
            gor_s = game_of_row[rws]
            for g in gsel:
                m = gor_s == g
                if m.sum() >= 4:
                    A = None
                    if st_of_game is not None and args.frac_dead > 0:
                        # jqt5 dead-state mining: alive-count per piece TYPE per row. A
                        # type at 0 alive = every piece-concept of that type is DEAD
                        # (anti-pole member; Kaveh's negative-pole design).
                        slg = tr.slots[rws[m]]
                        st_g = st_of_game[g]
                        A = np.zeros((int(m.sum()), 13), np.int8)
                        for k in range(32):
                            pres = (slg == k).any(1)
                            if pres.any():
                                A[pres, int(st_g[k])] += 1
                    games.append((rws[m], ids_all[m].astype(np.int64),
                                  sq_all[m].astype(np.int64) if sq_all is not None else None,
                                  pc_all[m].astype(np.int64) if pc_all is not None else None,
                                  st_of_game[g] if st_of_game is not None else None,
                                  A))
            jqt_idx.refresh(games)
            print(f"[jqt] index refreshed: {len(games)} games, {len(rws):,} rows coded "
                  f"[{time.time()-t0r:.0f}s]", flush=True)

    # optimizer AFTER attach_poles and AFTER jqt construction (both add parameters; Adam
    # captures the list at construction -- the take-2 scar, twice avoided).
    _params = [p for p in net.parameters() if p.requires_grad]
    if args.jqt:
        _params += [p for p in jqt.parameters() if p.requires_grad]
    opt = torch.optim.Adam(_params, lr=args.lr)

    tp = None
    if args.w_turnpair > 0:
        _tz = np.load(args.turnpair_npz)
        tp = {k: _tz[k] for k in ("row", "probs_a", "probs_b", "tok_b", "glob_b")}
        _tg = np.abs((tp["probs_a"][:, 0] + 0.5 * tp["probs_a"][:, 1])
                     - (tp["probs_b"][:, 0] + 0.5 * tp["probs_b"][:, 1]))
        print(f"[turnpair] {len(tp['row']):,} null-move pairs; mean|dE(turn)| {_tg.mean():.3f}, "
              f"decisive (>0.1) {(_tg > 0.1).mean():.0%}", flush=True)

    wdl_pack = None
    if args.move_head or args.w_bellman > 0:
        import os as _os2
        ddir = paths.derived("wdl_labels")
        shards = sorted(_os2.path.join(ddir, f) for f in _os2.listdir(ddir)
                        if f.endswith(".npz")) if _os2.path.isdir(ddir) else []
        if not shards:
            raise SystemExit("[move-head] no wdl_labels shards yet -- run gen_wdl_labels first")
        packs = [np.load(f) for f in shards]
        base = 0; row_l, tok_l, glob_l, wdl_l, mid_l, off_l = [], [], [], [], [], [0]
        for pk in packs:
            row_l.append(pk["row"]); tok_l.append(pk["tok"]); glob_l.append(pk["glob"])
            wdl_l.append(pk["wdl"]); mid_l.append(pk["mid"])
            off_l.extend((pk["off"][1:] + base).tolist()); base += len(pk["tok"])
        wdl_pack = {"row": np.concatenate(row_l), "tok": np.concatenate(tok_l),
                    "glob": np.concatenate(glob_l), "wdl": np.concatenate(wdl_l),
                    "mid": np.concatenate(mid_l), "off": np.array(off_l, np.int64)}
        print(f"[move-head] {len(wdl_pack['row']):,} WDL-labeled positions, "
              f"{len(wdl_pack['tok']):,} children across {len(shards)} shard(s)", flush=True)
    grad_labels = None
    if args.w_grad > 0:
        import os as _os
        _lp = paths.derived("bestmove_listwise.npz")
        if _os.path.exists(_lp):
            _gl = np.load(_lp)
            grad_labels = {"mode": "list", "tok": _gl["tok"], "glob": _gl["glob"],
                           "off": _gl["off"], "best": _gl["best"]}
            print(f"[grad] LISTWISE labels: {len(_gl['best']):,} positions, "
                  f"{len(_gl['tok']):,} children", flush=True)
        else:
            _gl = np.load(paths.derived("bestmove_labels.npz"))
            grad_labels = {"mode": "pair", **{k: _gl[k] for k in
                           ("best_tok", "best_glob", "alt_tok", "alt_glob")}}
            print(f"[grad] pairwise labels: {len(_gl['best_tok']):,}", flush=True)
    ply_of_row = tr.ply_of_row()
    trow_of_game = np.full(len(tr), -1, np.int64)
    trow_of_game[game_of_row[t_rows]] = t_rows          # -1 = censored game, no terminal
    _ = fit.true_edge_mask(np.array([0]), np.array([0]))   # builds tr._row_hash + edge keys once
    tr_row_hash = tr._row_hash

    _ing = {"lines": 0, "games": 0}
    _bw_name_w = None
    if args.balance_npz and _bw is not None:
        _bz2 = np.load(args.balance_npz)
        _bw_name_w = {}
        for _si, _nm in enumerate(_bz2["names"]):
            _m = _bz2["stratum"] == _si
            if _m.any():
                _bw_name_w[str(_nm)] = float(_bz2["weight"][np.flatnonzero(_m)[0]])

    def _classify_rows(rows):
        """phase|outcome|material stratum names for new rows (audit_data_balance's criteria)."""
        VALm = np.array([0, 1, 3, 3, 5, 9, 0, 1, 3, 3, 5, 9, 0])
        tk = tr.tok[rows]
        v = VALm[tk]
        dmat = (np.where((tk >= 1) & (tk <= 6), v, 0).sum(1)
                - np.where(tk >= 7, v, 0).sum(1))
        npc = (tk > 0).sum(1)
        plyr = ply_of_row[rows]
        yw = outcome[rows]
        phase_n = np.where(plyr < 16, "opening", np.where(npc <= 10, "endgame", "middlegame"))
        on = np.where(yw == T.WIN, "white-wins", np.where(yw == T.DRAW, "draw",
                      np.where(yw == T.LOSS, "black-wins", "censored")))
        mc = np.select([dmat > 5, dmat > 3, dmat > 1, dmat >= -1, dmat >= -3, dmat >= -5],
                       ["W-heavy", "W-piece", "W-minor", "even", "B-minor", "B-piece"],
                       default="B-heavy")
        return np.char.add(np.char.add(phase_n.astype(str), "|" + on.astype(str)),
                           "|" + mc.astype(str))

    def ingest():
        """poll the growing TSV, append complete new games, reassign every corpus-derived
        structure. Split is hashed from the START PLACEMENT so both members of a fork pair
        land in the SAME split (a pair straddling train/holdout would be leakage)."""
        nonlocal fit, val, fit_games, val_games, split, game_of_row, src_of_game, \
            ply_of_row, trow_of_game, tr_row_hash, outcome, _bw, cov, reps
        import hashlib as _hl
        t0i = time.time()
        with open(args.ingest_tsv) as fh:
            lines = [l for l in fh if l.endswith("\n")]
        new = lines[_ing["lines"]:]
        if len(new) < args.ingest_min:
            return
        _ing["lines"] = len(lines)
        rows_in = []
        for ln in new:
            p = ln.rstrip("\n").split("\t")
            if len(p) >= 4:
                rows_in.append((20_000_000 + int(p[0]), int(p[1]), p[3].split(), p[2]))
        g0, kept = tr.append_games(rows_in, max_plies=args.max_plies)
        ng = len(kept)
        if ng == 0:
            return
        _ing["games"] += ng
        # split by START PLACEMENT hash -> fork twins co-assigned; 70/15/15 like the boot split
        hs = np.array([int(_hl.blake2b(rows_in[i][3].split()[0].encode(),
                                       digest_size=4).hexdigest(), 16) % 100 for i in kept])
        newsplit = np.where(hs < 70, 0, np.where(hs < 85, 1, 2)).astype(split.dtype)
        split = np.concatenate([split, newsplit])
        gnew = np.arange(g0, g0 + ng)
        # new split-0 games: 10% to val (matching the boot val fraction within the fit pool)
        nv = gnew[newsplit == 0]
        n_valn = max(1, int(len(nv) * args.val_frac)) if len(nv) else 0
        val_games = np.concatenate([val_games, nv[:n_valn]])
        fit_games = np.concatenate([fit_games, nv[n_valn:]])
        game_of_row = tr.game_of_row()
        src_of_game = tr.source
        ply_of_row = tr.ply_of_row()
        outcome = (tr.outcome_of_row_white() if args.basin_pov == "white"
                   else tr.outcome_of_row())
        tg2 = np.full(len(tr), -1, np.int64)
        tg2[:len(trow_of_game)] = trow_of_game
        endsb = (tr.term[gnew] >= 0)                       # board terminals only
        tg2[gnew[endsb]] = (tr.start[gnew] + tr.length[gnew] - 1)[endsb]
        trow_of_game = tg2
        if _bw is not None and _bw_name_w is not None:
            nrows = np.arange(tr.start[g0], tr.n_positions)
            names_n = _classify_rows(nrows)
            wn = np.array([_bw_name_w.get(s, 1.0) for s in names_n], np.float32)
            _bw = np.concatenate([_bw, wn])
        cov, reps = tr.coverage(), tr.repeats()
        fit = T.PairSampler(tr, fit_games, seed=args.seed + _ing["games"], cov=cov,
                            repeats=reps, min_ply=args.min_ply, row_weight=_bw)
        val = T.PairSampler(tr, val_games, seed=args.seed + 1, cov=cov, repeats=reps,
                            min_ply=args.min_ply)
        _ = fit.true_edge_mask(np.array([0]), np.array([0]))
        tr_row_hash = tr._row_hash
        print(f"[ingest] +{ng} games ({(newsplit==0).sum()} train-pool / "
              f"{(newsplit==1).sum()} cal / {(newsplit==2).sum()} test) | corpus now "
              f"{len(tr):,} games / {tr.n_positions:,} rows [{time.time()-t0i:.0f}s]",
              flush=True)

    cur_step = {"s": 0}                     # loop writes; terms reads (dest-CE warmup ramp)

    def terms(model, sampler, n, n_rev, ph=None, audit=False):
        """One objective evaluation on freshly sampled pairs. Returns (loss, metrics)."""
        ph = ph or Phase(dev, False)
        with ph("sample"):
            _na = int(n * args.adjacent_frac)
            if _na:
                _a, _b = sampler.adjacent_triples(_na), sampler.triples(n - _na)
                i, j, k = (np.concatenate([u, v]) for u, v in zip(_a, _b))
            else:
                i, j, k = sampler.triples(n)
            x = sampler.cross(i)                        # one cross-game partner PER SOURCE i
            ra, rb, rgap = sampler.reversible(n_rev)    # OBSERVED backward, via repetitions
            # terminal-contrast rows ride in the SAME fused forward (2026-08-07 speedup: the
            # separate encode_q calls here were two extra encoder launches per step on a
            # dispatch-bound loop)
            if args.poles == "contrastive" and len(t_rows):
                tia = rng_np.integers(0, len(t_rows), args.n_term)
                tib = rng_np.integers(0, len(t_rows), args.n_term)
                rows = [i, j, k, x, ra, rb, t_rows[tia], t_rows[tib]]
            else:
                tia = tib = None
                rows = [i, j, k, x, ra, rb]
            # TERMINAL FOURTH LEG (Kaveh-approved next step, 2026-08-07): every position's path
            # to its OWN game's ending is witnessed, length = remaining plies -- a data-exact
            # bracket d(i -> T_game) in [1, remaining+1] giving the within-basin distance-to-end
            # gradient that DTZ/faithfulness showed was absent. Censored games have no terminal
            # row and are skipped.
            Tg = trow_of_game[game_of_row[i]]
            tvalid = (Tg >= 0) & (src_of_game[game_of_row[i]] == 1)   # SF terminals only (Option B)
            rows.append(np.where(tvalid, Tg, i))            # placeholder self-row where censored
        # COLOR-MIRROR involution (lc0/NNUE adaptation, 2026-08-08): one coin per STEP so every
        # leg of every pair is mirrored consistently. Walls/gas/vicreg are mirror-blind; the two
        # label consumers (basin y, move-head) swap W<->L below when _mirror is set.
        _mirror = bool(args.mirror) and bool(rng_np.integers(0, 2))
        with ph("encode"):
            phi, zA, zB, zR, zT, idx = encode(model, rows, mirror=_mirror)
        if tia is not None:
            gi, gj, gk, gx, gra, grb, gta, gtb, gT = idx[:9] if len(idx) == 9 else (*idx, None)
        else:
            gi, gj, gk, gx, gra, grb, gT = idx
        # In dual mode zR is the CONDITIONED embedding (z_base + delta), so every distance is the
        # player-tuned quasimetric; zB stays available as the pooled reference for the readout.
        if getattr(model, "split_head", False):
            DQ = lambda u, v: model.dA(zB[u], zB[v])
        else:
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
        # CONFINING SPRING (log-gas attraction) rather than Huber. Huber's force is CONSTANT past
        # delta, so against a repulsion that also pushes with a constant force it stalls at an
        # arbitrary balance point instead of hitting the target -- measured as forward median 24.3
        # against a target of <=3.7. r^2 + q*r^4 is gentle near the gap and hauls back hard far
        # from it: "if they're five plies apart, they should shrink back to five."
        # FENE is the default: soft inside the true gap, wall at args.fene_stretch x the gap.
        def _spring(dd, gp):
            tl = torch.log1p(gp)
            if not args.log_gas:
                return quasimetric_regression(dd, tl)
            if args.confine == "lj":
                return lj_confinement(dd, tl, fene_r_max(gp, args.fene_stretch)
                                      if args.lj_wall else None)
            if args.confine == "fene":
                return fene_confinement(dd, tl, fene_r_max(gp, args.fene_stretch),
                                        soft=args.fene_soft)
            return confining_regression(dd, tl, args.confine_quartic)
        # HARD WALLS ONLY (Kaveh 2026-08-07): floor at 1 (nothing is closer than one move),
        # ceiling at each observed gap + one gauge unit (a witnessed path is an exact upper
        # bound). NO spring between the walls -- the interior is the gas's to arrange, and
        # min-over-paths emerges from whichever observation carries the tightest ceiling.
        # Pairs whose boards are IDENTICAL (repetitions) are exempt: their true distance is 0
        # and the floor must not fight it.
        def _same_board(ra, rb):
            return ((tr.tok[ra] == tr.tok[rb]).all(1) & (tr.glob[ra] == tr.glob[rb]).all(1))
        # WALLS CONSTRAIN THE CONDITIONED FIELD AT THE GAME'S OWN ELO (Kaveh 2026-08-07): a
        # witnessed path upper-bounds distance only UNDER THE PLAY THAT PRODUCED IT -- against an
        # optimal opponent the distance may be larger (or infinite). In dual mode the wall legs
        # therefore use the conditioned embeddings (each row at its own side-to-move Elo); the
        # base field is shaped through the shared weights, and higher-Elo readouts stay free to
        # exceed human-witnessed ceilings.
        DQw = (DQ if getattr(model, "split_head", False)
               else ((lambda u, v: model.qhead.iqe(zR[u], zR[v]))
                     if (model.dual and zR is not None) else DQ))
        _n_wall = [0]
        def _leg(rows_a, rows_b, dd, gp):
            distinct = torch.from_numpy(~_same_board(rows_a, rows_b)).to(dev)
            _n_wall[0] += int(distinct.sum())
            if distinct.any() and args.log_gas:
                return pair_walls(dd[distinct], gp[distinct])
            if not args.log_gas:
                return _spring(dd, gp)
            return dd.sum() * 0
        d_ij_w, d_jk_w, d_ik_w = DQw(gi, gj), DQw(gj, gk), DQw(gi, gk)
        # OPTION B (Kaveh 2026-08-07): the QUASIMETRIC is walled by SF games only -- near-optimal
        # play treated as optimal, so ceilings are true minimal-ish path lengths and the metric
        # keeps its minimal semantics. Human games are EXCLUDED from every distance assertion;
        # they train only the probabilistic layers (committor CE, arm-A regions, the Elo residual)
        # -- and the surprisal-priced (-log P) route for human paths is the planned Option A.
        # SF == 1 (HUMAN == 0): this mask was INVERTED from 2026-08-07 until now -- every
        # 'Option-B' run walled HUMAN paths and the sf-only runs had NO walls at all. Mistake #12.
        m_sf = torch.from_numpy((src_of_game[game_of_row[i]] == 1)).to(dev)
        msf_np = m_sf.cpu().numpy()
        def _wleg(ra, rb, dd, gp):
            if not msf_np.any():
                return dd.sum() * 0
            return _leg(ra[msf_np], rb[msf_np], dd[m_sf], gp[m_sf])
        l_q = (_wleg(i, j, d_ij_w, gap_ij) + _wleg(j, k, d_jk_w, gap_jk)
               + _wleg(i, k, d_ik_w, gap_ik)) / 3.0
        if gT is not None and args.log_gas:
            tv = torch.from_numpy(tvalid).to(dev)
            if tv.any():
                Trows = np.where(tvalid, Tg, i)
                d_iT = DQw(gi, gT)[tv]
                gap_T = torch.from_numpy((ply_of_row[Trows] - ply_of_row[i])
                                         .astype(np.float32)).to(dev)[tv].clamp(min=1)
                l_q = 0.75 * l_q + 0.25 * _leg(i[tvalid], Trows[tvalid], d_iT, gap_T)
        if len(ra):
            l_q = 0.75 * l_q + 0.25 * _spring(DQ(gra, grb), gap_r)
        d_ji = DQ(gj, gi)           # the reversal, on the SAME two positions
        d_x = DQ(gi, gx)            # cross-game, from the SAME source i
        # never repel a pair that is ACTUALLY one played move apart (or the same position) --
        # hash-level check catches cross-game transpositions; reverse push stays legal
        _true_edge = sampler.true_edge_mask(i, x)
        if _true_edge.any():
            d_x = d_x[torch.from_numpy(~_true_edge).to(dev)]
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
        # PAIRWISE REPULSION (log-gas): unbounded, never saturating. Acts between position
        # pairs; there is no directed/global force here. The relu hinge below it
        # delivered exactly zero gradient past its margin, which PINNED reverse distances at the
        # margin -- reverse median 55.8 against a floor of e^4-1 = 53.6, so the asymmetry ratio
        # was a readout of repel_margin rather than anything learned. -log(d) keeps pushing
        # forever and stops only when the confining springs on observed pairs tug back.
        # K-FOLD IN-BATCH GAS (Kaveh: 'take ten other games from the same batch and apply ten
        # repulsions -- less force, more repulsions'): a 10x lower-variance Monte Carlo estimate
        # of the full pairwise pressure, from embeddings already in hand.
        K = 10
        n_i = len(i)
        all_rows = np.concatenate([i, j, k]); all_g = np.concatenate([gi.cpu().numpy(),
                                                                      gj.cpu().numpy(), gk.cpu().numpy()])
        pick = rng_np.integers(0, len(all_rows), (n_i, K))
        srcg = game_of_row[i]
        diffgame = game_of_row[all_rows[pick]] != srcg[:, None]
        not_edge = ~np.stack([sampler.true_edge_mask(i, all_rows[pick[:, kk]]) for kk in range(K)], 1)
        okm = (diffgame & not_edge).ravel()
        src_idx = np.repeat(np.arange(n_i), K)[okm]
        tgt_idx = pick.ravel()[okm]
        d_ib = DQ(gi[torch.from_numpy(src_idx).to(dev)],
                  torch.from_numpy(all_g[tgt_idx]).to(dev))
        um_b = torch.nonzero(unc, as_tuple=True)[0]
        _rep = ((lambda d, m: screened_repulsion(d)) if args.log_gas else terminal_repulsion)
        rep_rev = (_rep(d_ji[um_b], args.repel_margin)
                   if len(um_b) else torch.zeros((), device=dev))
        l_rep_b = (rep_rev + _rep(d_x, args.repel_margin)
                   + (_rep(d_ib, args.repel_margin) if len(d_ib) else torch.zeros((), device=dev))) / 3.0
        # One-body confinement makes the log-gas equilibrium EXIST: points that appear only under
        # the pairwise repulsion have nothing pulling them back, and -log(d) alone runs to -inf.
        l_conf = (confine_radius(zB, args.confine_target, args.confine_quartic)
                  if args.log_gas else torch.zeros((), device=dev))

        # ---- the three W/D/L poles: probabilistic reachability, positives only -----------------
        # d(s -> P_k) for k = win/draw/loss, then CE against the outcome the game ACTUALLY reached.
        # Censored (time-forfeit) rows are dropped: a flagged position may be dead winning, so its
        # result says nothing about the board.
        l_basin = torch.zeros((), device=dev)
        l_polesep = torch.zeros((), device=dev)
        if args.w_basin > 0 and model.poles is not None:
            y_np = outcome[np.concatenate([i, j, k])]
            if _mirror and args.basin_pov == "white":
                y_np = np.where(y_np == 0, 2, np.where(y_np == 2, 0, y_np))  # W<->L; mover-POV
                                                        # labels are mirror-invariant, no swap
            live = np.flatnonzero(y_np >= 0)
            if len(live) > 8:
                # CONDITIONED committor (2026-08-07): in dual mode the CE trains P(o | s, elo)
                # through the conditioned embeddings -- each row at its OWN side-to-move Elo.
                # Querying at Elo 3500 then reads the near-minimax committor; at 1500, the
                # exploitation committor. Base embeddings otherwise never receive outcome
                # gradient conditioned on who is playing.
                _zsrc = zR if (model.dual and zR is not None) else zB
                zrows = torch.cat([_zsrc[gi], _zsrc[gj], _zsrc[gk]], 0)[
                    torch.from_numpy(live.astype(np.int64)).to(dev)]
                y = torch.from_numpy(y_np[live].astype(np.int64)).to(dev)
                P = model.poles.poles                                   # (3, d)
                _dpole = (model.dB if getattr(model, "split_head", False)
                          else (model.qhead.d_base if model.dual else model.iqe))
                dP = torch.stack([_dpole(zrows, P[c].expand(len(zrows), -1))
                                  for c in range(3)], 1)                # (B,3)
                l_basin = basin_ce(dP, y, temperature=args.basin_temp, raw=bool(args.log_gas))
                # Poles must not merge: identical poles make every distance equal and the CE sits
                # at log 3 forever. LJ-shaped potential, same term the field head uses.
                # Fixed poles CANNOT merge, so the separation term is unnecessary -- it exists
                # only to stop learned poles collapsing into the log-3 saddle.
                if not model.poles.fixed:
                    pd = torch.stack([(model.qhead.d_base(P[a:a+1], P[b:b+1])[0] if model.dual
                                       else model.iqe(P[a:a+1], P[b:b+1])[0])
                                      for a in range(3) for b in range(3)]).view(3, 3)
                    l_polesep = pole_potential(pd, typical_pair_scale(d_ij.detach()))

        # ---- the ENDING-TYPE layer: anchor, absorb, keep arrival points distinct ---------------
        # Terminal instances sit at their ending's certainty radius (0 for rules and every draw,
        # 1 ply for resignations, where position and result measurably disagree); the ending pole
        # must be ABSORBING (you cannot leave a finished game); distinct endings must not pile onto
        # one point; and each ending must subsume into its outcome pole.
        l_anchor = l_absorb = l_termrep = l_subsume = torch.zeros((), device=dev)
        l_start = l_startirr = torch.zeros((), device=dev)
        # Anchor regression form, shared by the start and terminal anchors (defined here, OUTSIDE
        # the conditionals, so disabling one anchor cannot NameError the other).
        # QUARTIC for the anchors, not LJ-12: anchor residuals START at r ~ log1p(750) - 0.7 =
        # 5.9, where raw r^12 is ~2e9 -- the smoke with LJ anchors nuked every other term
        # (quasi 771, rev_ratio 0.31, rank 3.7). r^2 + r^4 pulls with force ~834 at r=5.9 and ~6
        # near the target: strong enough to beat the repulsion (Huber's constant 1.0 was not),
        # never explosive. Pairs keep LJ-12 because their residuals start small.
        # CAPPED anchors (take 3). The force is now bracketed by two measured failures: Huber's
        # constant 1.0 LOST to the gas (terminal->pole 557->1156 rising), the quartic's ~800 at
        # the initial residual CRUSHED it (all three seeds collapsed to eff rank ~2 with d_far
        # 6->3.4 and quasi 20-40 by step 18000). Instead of bisecting force constants, cap the
        # residual structurally: LJ with the wall at the LARGEST POSSIBLE residual
        # log1p(pole_height), so the pull is steep only near initialization scale and relaxes as
        # terminals approach their shells -- capture without crush, bounded by construction.
        # Anchor spring selectable; smoke-loop 2026-08-07. capped-LJ has a mid-range plateau
        # (force ~0.4 at r~5.8) which became the binding constraint once the gas was screened and
        # the bowl removed (terminals equidistant from all poles at 673/676/673). quadratic = r^2,
        # force 2r everywhere: ~12 at range, ~2 near target, no plateau, no explosion.
        _anchor_rmax = float(np.log1p(args.pole_height))
        _forms = {"capped": lambda dd, tl: lj_confinement(dd, tl, r_max=_anchor_rmax),
                  "quadratic": lambda dd, tl: confining_regression(dd, tl, quartic=0.0),
                  "quartic": lambda dd, tl: confining_regression(dd, tl, quartic=1.0)}
        _anchor_reg = (_forms[args.anchor_form] if args.log_gas else quasimetric_regression)
        if args.w_start > 0 and model.poles is not None:
            Ps = model.poles.poles[START_IDX:START_IDX + 1]
            zs = zB[gi]                                    # the triple's first position
            dq0 = (model.qhead.d_base if model.dual else model.iqe)
            ply_i = torch.from_numpy(ply_of_row[i].astype(np.float32)).to(dev)
            # Under the log-gas the anchors must be the SAME potential family as the pair
            # springs. Huber's constant restoring force lost the tug-of-war against the repulsion:
            # measured on the first gauge fleet, median d(terminal -> own pole) ROSE 557 -> 706
            # over steps 500-1500 (target ~1) and basin_spread sat at 0.001-0.003 through step
            # 3750 -- positions were not climbing toward their poles at all.
            l_start = _anchor_reg(dq0(Ps.expand(len(zs), -1), zs), torch.log1p(ply_i))
            l_startirr = start_irreversibility(dq0(zs, Ps.expand(len(zs), -1)),
                                               args.start_margin)
        l_termcon_att = l_termcon_rep = torch.zeros((), device=dev)
        if tia is not None:
            # CONTRASTIVE TERMINALS: same-outcome attraction, opposite-outcome floor. Embeddings
            # come from the fused batch forward above -- zero extra encoder calls.
            t_out = TPOLE_OUT[t_pole]
            ia, ib = tia, tib
            za_t, zb_t = zB[gta], zB[gtb]
            # EUCLIDEAN distances, deliberately not IQE (Kaveh: separation should show in the
            # UMAP, which sees Euclidean z-geometry; IQE can be large while Euclidean is tiny).
            # eps under the root: sqrt has an INFINITE gradient at exactly 0, and a random pair
            # draws the same terminal twice every ~50 steps (p~1.8%/step at 64 pairs / 3.5k
            # terminals) -- the first collision NaN'd the entire graph at step ~80. Self-pairs
            # additionally excluded (a point attracting itself teaches nothing).
            d_euc = (za_t - zb_t).pow(2).sum(-1).add(1e-8).sqrt()
            notself = torch.from_numpy((ia != ib)).to(dev)
            same = torch.from_numpy((t_out[ia] == t_out[ib]).astype(np.bool_)).to(dev) & notself
            l_termcon_att, l_termcon_rep = terminal_contrast(
                d_euc[same], d_euc[(~same) & notself], args.term_margin)
        if args.w_anchor > 0 and model.poles is not None and len(t_rows):
            tsel = rng_np.integers(0, len(t_rows), min(args.n_term, len(t_rows)))
            trow = t_rows[tsel]
            tk = torch.from_numpy(tr.tok[trow].astype(np.int64)).to(dev)
            tg = torch.from_numpy(tr.glob[trow].astype(np.float32)).to(dev)
            zt_ = model.encode_q(tk, tg)
            Pall = model.poles.poles
            pid = torch.from_numpy(t_pole[tsel].astype(np.int64)).to(dev)
            Pp = Pall[pid]
            dq = (model.qhead.d_base if model.dual else model.iqe)
            l_anchor = _anchor_reg(
                dq(zt_, Pp), torch.from_numpy(t_radius[tsel]).to(dev))
            l_absorb = absorbing_penalty(dq(Pp, zt_), args.absorb_margin)
            # distinct ENDING poles must stay apart -- different arrival points of one surface
            ne = len(t_names) - 3
            if ne > 1:
                a = torch.arange(3, len(t_names), device=dev)
                b = a[torch.randperm(ne, device=dev)]
                keep = a != b
                if keep.any():
                    l_termrep = terminal_repulsion(dq(Pall[a[keep]], Pall[b[keep]]),
                                                   args.termrep_margin)
            # each ending SUBSUMES into its outcome: d(ending -> outcome) driven to 0 (domination)
            par = model.poles.parent[3:]
            ok_p = par >= 0
            if ok_p.any():
                l_subsume = dq(Pall[3:][ok_p], Pall[par[ok_p]]).mean()

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
        l_qdist = torch.zeros((), device=dev)
        if qd is not None and model.poles is not None and args.basin_pov == "white":
            _qs = rng_np.integers(0, len(qd["row"]), 64)
            _qrows = qd["row"][_qs]
            zq = model.encode_q(
                torch.from_numpy(tr.tok[_qrows].astype(np.int64)).to(dev),
                torch.from_numpy(tr.glob[_qrows].astype(np.float32)).to(dev))
            _P3q = model.poles.poles[:3]
            dPq = torch.stack([model.dB(zq, _P3q[[k]].expand(len(zq), -1))
                               for k in range(3)], 1)
            logq = torch.log_softmax(-dPq / args.basin_temp, 1)
            tgtq = torch.from_numpy(qd["probs"][_qs]).to(dev)
            wq = torch.from_numpy(qd["w"][_qs]).to(dev)
            l_qdist = -(wq * (tgtq * logq).sum(1)).mean()
        # A-SIDE ABSORBING ANCHOR: every terminal sits exactly one ply before its outcome
        # pole on the LENGTH ruler -- the pole becomes the class's absorbing state, and the
        # walls staircase (which already ends at the terminals) extends into it. IQE asymmetry
        # makes many->one at cost 1 representable without the reverse being cheap.
        l_anchor_a = torch.zeros((), device=dev)
        if args.w_anchor_a > 0 and model.poles is not None and len(t_rows) \
                and getattr(model, "split_head", False) and args.basin_pov == "white":
            asel = rng_np.integers(0, len(t_rows), min(args.n_term, len(t_rows)))
            arow = t_rows[asel]
            za_ = model.encode_q(
                torch.from_numpy(tr.tok[arow].astype(np.int64)).to(dev),
                torch.from_numpy(tr.glob[arow].astype(np.float32)).to(dev))
            _mo = TPOLE_OUT[t_pole[asel]]                    # mover-POV outcome at terminal
            _wt = tr.glob[arow, 0].astype(bool)
            _cls = np.where(_mo == 1, 1,
                            np.where(((_mo == 0) & _wt) | ((_mo == 2) & ~_wt), 0, 2))
            Ppa = model.poles.poles[torch.from_numpy(_cls.astype(np.int64)).to(dev)]
            l_anchor_a = ((model.dA(za_, Ppa) - 1.0) ** 2).mean()
        l_pole_gas_a = torch.zeros((), device=dev)
        if args.w_pole_gas_a > 0 and model.poles is not None \
                and getattr(model, "split_head", False):
            _P3 = model.poles.poles[:3]
            _dg = torch.stack([model.dA(zB[gi], _P3[[k]].expand(len(gi), -1))
                               for k in range(3)], 1)
            l_pole_gas_a = (1.0 / (1.0 + _dg)).mean()
        # DECISIVE HINGE (Kaveh 2026-08-10): who-wins enters the LENGTH ruler as an ORDERING
        # between the two decisive exits only -- relu(eps + dA(s->winner's pole) - dA(s->
        # loser's pole)). eps=0.5: the win condition is arriving FIRST (tempo = one ply), a
        # 2-ply gap would deny real threat proximity (the mate-in-1 objection). One-sided:
        # never pulls the winner-exit distance down (odometer preserved). Draw pole untouched:
        # d(s->draw) small everywhere is TRUE (absorption asymmetry, measured 6.3x). Per-game
        # outcome labels are per-POSITION weighted by the outcome ruler's own calibrated
        # belief in that label (detached) -- the back-rank-blunder objection: pre-blunder
        # positions of a lost game get ~0 weight. This is dB distilling into dA.
        l_hinge_a = torch.zeros((), device=dev)
        if args.w_hinge_a > 0 and model.poles is not None \
                and getattr(model, "split_head", False) and args.basin_pov == "white":
            y_h = outcome[i].copy()                              # white-POV: 0 W / 1 D / 2 L
            if _mirror:                                          # mirrored board = swapped colours
                y_h = np.where(y_h == 0, 2, np.where(y_h == 2, 0, y_h))
            dec = np.flatnonzero((y_h == 0) | (y_h == 2))
            if len(dec) > 8:
                gd = torch.from_numpy(dec.astype(np.int64)).to(dev)
                own_c = torch.from_numpy(y_h[dec].astype(np.int64)).to(dev)      # 0 or 2
                for_c = 2 - own_c
                _P3h = model.poles.poles[:3]
                zdec = zB[gi][gd]
                # TRULY one-sided (2026-08-10 smoke lesson: relu(eps + d_own - d_for) also
                # PULLS d_own down -- the whole distance structure compressed to ~2.3): the
                # own-exit distance is a detached reference; gradient only pushes the foreign
                # exit away. Odometer calibration stays the anchor/walls' business.
                d_own = model.dA(zdec, _P3h[own_c]).detach()
                d_for = model.dA(zdec, _P3h[for_c])
                with torch.no_grad():                            # credibility of the label
                    dPh = torch.stack([model.dB(zdec, _P3h[[k]].expand(len(zdec), -1))
                                       for k in range(3)], 1)
                    w_cred = torch.softmax(-dPh / args.basin_temp, 1)[
                        torch.arange(len(own_c), device=dev), own_c]
                l_hinge_a = (w_cred * torch.relu(args.hinge_eps + d_own - d_for)).mean()
        # BELLMAN RESIDUAL (Kaveh 2026-08-08 "we want both"): d(s -> mover's win pole) must
        # equal 1 + min over legal moves of the child's distance to the SAME pole. Walls only
        # enforce consistency along observed game paths; this enforces it along the paths the
        # ENGINE will choose -- the descent property selfplay_probe measured at chance level.
        # One-sided by design: only the mover's own goal admits a min (the opponent picks the
        # other pole's successor, a minimax quantity we don't hand-code).
        l_bell = torch.zeros((), device=dev)
        if args.w_bellman > 0 and wdl_pack is not None and model.poles is not None:
            POLES3b = model.poles.poles[:3]
            _dpb = (model.dB if getattr(model, "split_head", False)
                    else (model.qhead.d_base if model.dual else model.iqe))
            bsel = rng_np.integers(0, len(wdl_pack["row"]), args.bellman_pos)
            terms_bl = []
            for pi_ in bsel:
                a, b2 = int(wdl_pack["off"][pi_]), int(wdl_pack["off"][pi_ + 1])
                prow = int(wdl_pack["row"][pi_])
                _ptk, _pgb = tr.tok[prow:prow+1], tr.glob[prow:prow+1]
                _ctk, _cgb = wdl_pack["tok"][a:b2], wdl_pack["glob"][a:b2]
                if _mirror:
                    from catspace.research.components.encoder.approaches.jepa_tokenizer.src.jepa import (
                        mirror_arrays as _ma)
                    _ptk, _pgb = _ma(_ptk, _pgb)
                    _ctk, _cgb = _ma(_ctk, _cgb)
                _wtmb = bool(tr.glob[prow, 0]) != _mirror
                Pm = POLES3b[[0 if _wtmb else 2]]        # mover's win pole, colour-fixed
                z_par = model.encode_q(
                    torch.from_numpy(_ptk.astype(np.int64)).to(dev),
                    torch.from_numpy(_pgb.astype(np.float32)).to(dev))
                # SEMI-GRADIENT (loss-interaction review, Kaveh 2026-08-08): the min over
                # children is a BOOTSTRAPPED target -- full gradient both sides is Q-learning's
                # underestimation spiral (the min picks the most-underestimated child and drags
                # the parent onto it). Children are DETACHED (no_grad through the online
                # encoder); gradient flows into the parent only. (The JEPA EMA target encoder
                # would be the DQN-style upgrade, but encode_target projects to arm A -- the
                # wrong space for dB -- so online-detached it is.)
                with torch.no_grad():
                    z_ch = model.encode_q(
                        torch.from_numpy(_ctk.astype(np.int64)).to(dev),
                        torch.from_numpy(_cgb.astype(np.float32)).to(dev))
                d_par = _dpb(z_par, Pm)
                d_ch = _dpb(z_ch, Pm.expand(len(z_ch), -1))
                terms_bl.append((d_par[0] - 1.0 - d_ch.min().detach()) ** 2)
            l_bell = torch.stack(terms_bl).mean()
        _audit = {}
        if audit:
            # FORCE AUDIT DURING TRAINING (Kaveh 2026-08-07): per-term gradient magnitude on phi,
            # i.e. the ACTUAL force each group exerts on the shared representation this step --
            # the design-time slope arithmetic, but measured live where the terms compete.
            _groups = {"F_walls": args.w_iqe * l_q,
                       "F_gas": args.w_iqe * args.w_repel * l_rep_b,
                       "F_basin": args.w_basin * l_basin,
                       "F_vicreg": args.w_var * l_var + args.w_cov * l_cov,
                       "F_regionA": args.w_region * (l_nll + args.w_repel * l_rep_a)}
            for _nm, _t in _groups.items():
                if isinstance(_t, torch.Tensor) and _t.requires_grad:
                    _g = torch.autograd.grad(_t, phi, retain_graph=True, allow_unused=True)[0]
                    _audit[_nm] = float(_g.norm(dim=-1).mean()) if _g is not None else 0.0
                else:
                    _audit[_nm] = 0.0
        l_mh = l_mh_cons = l_mh_bell = torch.zeros((), device=dev)
        if wdl_pack is not None and model.poles is not None and getattr(model, "move_head", None) is not None:
            _dp2 = (model.dB if getattr(model, "split_head", False)
                    else (model.qhead.d_base if model.dual else model.iqe))
            POLES3 = model.poles.poles[:3]                  # (W, D, L)
            psel = rng_np.integers(0, len(wdl_pack["row"]), args.mh_pos)
            terms_mh, terms_cons, terms_mhb = [], [], []
            for pi_ in psel:
                a, b2 = int(wdl_pack["off"][pi_]), int(wdl_pack["off"][pi_ + 1])
                prow = int(wdl_pack["row"][pi_])
                _ptk, _pgb = tr.tok[prow:prow+1], tr.glob[prow:prow+1]
                _mid_np = wdl_pack["mid"][a:b2].astype(np.int64)
                if _mirror:
                    from catspace.research.components.encoder.approaches.jepa_tokenizer.src.jepa import (
                        mirror_arrays, mirror_squares)
                    _ptk, _pgb = mirror_arrays(_ptk, _pgb)
                    _mid_np = _mid_np.copy()
                    _mid_np[:, 0] = mirror_squares(_mid_np[:, 0])
                    _mid_np[:, 1] = mirror_squares(_mid_np[:, 1])
                # parent context: one encoder forward
                phi_p = model.backbone(
                    torch.from_numpy(_ptk.astype(np.int64)).to(dev),
                    torch.from_numpy(_pgb.astype(np.float32)).to(dev))
                z_p = (model.qhead.embed(phi_p)[0] if model.dual else model.proj_b(phi_p))
                d_par = torch.stack([_dp2(z_p, POLES3[[o]]) for o in range(3)], 1)  # (1,3)
                mids = torch.from_numpy(_mid_np).to(dev)
                delta = model.move_head(phi_p.expand(len(mids), -1), mids)          # (n,3)
                d_hat = d_par.expand(len(mids), -1) + delta                          # (n,3)
                # three-pole graded listwise: model's softmax over moves of -d_hat[:,o]/tau vs
                # the engine's normalized o-probabilities. Child-mover POV wdl: (l,d,w) stored ->
                # our (win, draw, oppwin) target axes map to pole distances (LOSS, DRAW, WIN).
                wdl = torch.from_numpy(wdl_pack["wdl"][a:b2].astype(np.float32)).to(dev)
                tgt = wdl / wdl.sum(0, keepdim=True).clamp(min=1e-6)                 # per-axis
                # axis mapping depends on pole POV. mover-POV poles: our win = child's
                # LOSS-pole distance (child's mover = opponent). white-POV poles: our win =
                # WIN pole iff the PARENT's mover is white -- poles are colour-fixed.
                if args.basin_pov == "white":
                    # MISTAKE #14 (2026-08-08, caught by the checkmate/material convention
                    # probe): stored col0 = the CHILD-mover's win. Parent white -> child mover
                    # is BLACK -> col0 supervises the LOSS pole. The first white-POV map had
                    # this exactly backwards and trained the head's W/L axes inverted.
                    _wtm = bool(tr.glob[prow, 0]) != _mirror     # mirror flips the mover's colour
                    _cmap = ((0, 2), (1, 1), (2, 0)) if _wtm else ((0, 0), (1, 1), (2, 2))
                else:
                    _cmap = ((0, 2), (1, 1), (2, 0))
                for o, col in _cmap:
                    logp = torch.log_softmax(-d_hat[:, col] / args.basin_temp, 0)
                    terms_mh.append(-(tgt[:, o] * logp).sum())
                # BELLMAN ON THE HEAD (2026-08-08, after the field-level term failed): the best
                # move must chart a distance of exactly parent-1 on the mover's-win axis. Field
                # detached -- no bootstrap; also pins the head's absolute scale, which the
                # listwise CE above leaves gauge-free (softmax shift-invariance per axis).
                if args.w_mh_bell > 0:
                    # d_par is one row, so min(d_par + delta) - (d_par - 1) == min(delta) + 1:
                    # the constraint is purely on the head, zero gradient into the field.
                    _mwcol = _cmap[0][1]                 # mover's-win pole column
                    terms_mhb.append((delta[:, _mwcol].min() + 1.0) ** 2)
                # consistency: on 4 sampled children, the chart must match the encoded truth
                cs = rng_np.integers(0, len(mids), 4)
                _ctk, _cgb = wdl_pack["tok"][a:b2][cs], wdl_pack["glob"][a:b2][cs]
                if _mirror:
                    _ctk, _cgb = mirror_arrays(_ctk, _cgb)
                zc = model.encode_q(
                    torch.from_numpy(_ctk.astype(np.int64)).to(dev),
                    torch.from_numpy(_cgb.astype(np.float32)).to(dev))
                d_true = torch.stack([_dp2(zc, POLES3[[o]].expand(len(zc), -1))
                                      for o in range(3)], 1)
                # DEADBAND consistency (2026-08-08): hard MSE pinned the head to the field's
                # sibling-blindness (head went inert: mean |delta| 0.054 on a ~600 scale). Inside
                # tau the head answers only to the engine labels -- the sibling-scale resolution
                # the field cannot carry; beyond tau the chart still cannot hallucinate.
                terms_cons.append((torch.relu((d_hat[cs] - d_true).abs() - args.mh_deadband) ** 2).mean())
            l_mh = torch.stack(terms_mh).mean()
            l_mh_cons = torch.stack(terms_cons).mean()
            if terms_mhb:
                l_mh_bell = torch.stack(terms_mhb).mean()
        l_grad = torch.zeros((), device=dev)
        if grad_labels is not None and model.poles is not None:
            _dp = (model.dB if getattr(model, "split_head", False)
                   else (model.qhead.d_base if model.dual else model.iqe))
            PL = model.poles.poles[2]                       # LOSS pole: children's mover = opponent
            if grad_labels["mode"] == "list":
                # LISTWISE (Kaveh's pick): the engine's move must have the LOWEST d among ALL
                # legal children -- softmax CE over the full move set, ~30x the contrast of
                # best-vs-one at the same oracle budget.
                psel = rng_np.integers(0, len(grad_labels["best"]), 12)
                terms_l = []
                for pi_ in psel:
                    a, b = int(grad_labels["off"][pi_]), int(grad_labels["off"][pi_ + 1])
                    zc = model.encode_q(
                        torch.from_numpy(grad_labels["tok"][a:b].astype(np.int64)).to(dev),
                        torch.from_numpy(grad_labels["glob"][a:b].astype(np.float32)).to(dev))
                    dc = _dp(zc, PL.expand(len(zc), -1))
                    tgt = torch.tensor([int(grad_labels["best"][pi_]) - a], device=dev)
                    terms_l.append(torch.nn.functional.cross_entropy(-dc[None, :], tgt))
                l_grad = torch.stack(terms_l).mean()
            else:
                gsel = rng_np.integers(0, len(grad_labels["best_tok"]), 64)
                zbst = model.encode_q(torch.from_numpy(grad_labels["best_tok"][gsel].astype(np.int64)).to(dev),
                                      torch.from_numpy(grad_labels["best_glob"][gsel].astype(np.float32)).to(dev))
                zalt = model.encode_q(torch.from_numpy(grad_labels["alt_tok"][gsel].astype(np.int64)).to(dev),
                                      torch.from_numpy(grad_labels["alt_glob"][gsel].astype(np.float32)).to(dev))
                d_best = _dp(zbst, PL.expand(len(zbst), -1))
                d_alt = _dp(zalt, PL.expand(len(zalt), -1))
                l_grad = torch.nn.functional.softplus(d_best - d_alt).mean()
        # ---- TURN-CONTRAST PAIRS ---------------------------------------------------------------
        # The corpus has no minimal pairs on the turn flag (multiplicity 0.6%), so the flag's
        # pathway atrophied (sensitivity 0.000, unfixed by distillation). These null-move pairs
        # are manufactured minimal pairs: identical placement, only glob[0] differs, targets
        # differ -> the flag is the ONLY input that can separate them.
        l_turn = torch.zeros((), device=dev)
        if tp is not None and model.poles is not None and args.basin_pov == "white":
            with ph("turnpair"):
                _ts = rng_np.integers(0, len(tp["row"]), args.turnpair_n)
                _ra = tp["row"][_ts]
                tok_p = torch.from_numpy(np.concatenate(
                    [tr.tok[_ra], tp["tok_b"][_ts]]).astype(np.int64)).to(dev)
                glob_p = torch.from_numpy(np.concatenate(
                    [tr.glob[_ra], tp["glob_b"][_ts]]).astype(np.float32)).to(dev)
                z_p = model.encode_q(tok_p, glob_p)
                _P3t = model.poles.poles[:3]
                nT = args.turnpair_n
                dBt = torch.stack([model.dB(z_p, _P3t[[k]].expand(len(z_p), -1))
                                   for k in range(3)], 1)
                tgt_t = torch.from_numpy(np.concatenate(
                    [tp["probs_a"][_ts], tp["probs_b"][_ts]]).astype(np.float32)).to(dev)
                l_turn_b = -(tgt_t * torch.log_softmax(-dBt / args.basin_temp, 1)).sum(1).mean()
                dAt = torch.stack([model.dA(z_p, _P3t[[k]].expand(len(z_p), -1))
                                   for k in range(3)], 1)
                l_turn_a = torch.zeros((), device=dev)
                for _pc in (0, 2):                        # decisive poles only; draw untouched
                    pa, pb = tgt_t[:nT, _pc], tgt_t[nT:, _pc]
                    a_closer = (pa > pb).float()
                    d_close = a_closer * dAt[:nT, _pc] + (1 - a_closer) * dAt[nT:, _pc]
                    d_far = a_closer * dAt[nT:, _pc] + (1 - a_closer) * dAt[:nT, _pc]
                    gap = 0.35 * ((pa - pb).abs() > 0.05).float()   # one tempo; tie-safe
                    l_turn_a = l_turn_a + anchored_pairwise_rank(d_close, d_far, gap)
                l_turn = l_turn_b + 0.5 * l_turn_a

        # ---- JOINT QUANTIZED TRAINING ----------------------------------------------------------
        # Gradients from every jqt term flow into the TRUNK -- that is the joint part; the
        # grounding stack above is what makes that safe (you cannot collapse while still being
        # forced to predict real outcomes). Recon targets and dE are DETACHED field readings.
        l_vqrec = l_pers = l_jepa = l_cda = l_cdb = torch.zeros((), device=dev)
        l_sqrec = l_pcrec = l_pers_sq = l_pers_pc = torch.zeros((), device=dev)
        l_flip = l_dest = l_pcflip = l_pcdest = l_negpole = torch.zeros((), device=dev)
        _flip_gap, _dest_top1, _pcdest_top1 = -1.0, -1.0, -1.0
        _jqt_perp, _jqt_flip, _sq_flip, _pc_flip = -1.0, -1.0, -1.0, -1.0
        if jqt is not None and jqt_idx.ready():
            jqt.train(model.training)          # val passes must not update the EMA codebooks
            with ph("jqt_sample"):
                ts = rng_np.integers(0, len(jqt_gt["par"]), args.jqt_pairs)
                pr_ = jqt_gt["par"][ts]
                mid_ = torch.from_numpy(jqt_gt["mid"][ts]).to(dev)
                if args.jqt_square_codes or args.jqt_piece_codes:
                    grows, ggt, gk1, gk2, gplies, ghit, gdead = jqt_idx.sample_mixed(
                        args.jqt_goals, frac_dead=args.frac_dead)
                else:
                    grows, ghc, gplies, ghit = jqt_idx.sample(args.jqt_goals)
                    ggt = np.zeros(len(grows), np.int64)
                    gk1, gk2 = ghc[:, 0].copy(), ghc[:, 1].copy()
                    gdead = np.zeros(len(grows), np.float32)
                rows_j = np.concatenate([pr_, pr_ + 1, grows])
                tok_j = torch.from_numpy(tr.tok[rows_j].astype(np.int64)).to(dev)
                glob_j = torch.from_numpy(tr.glob[rows_j].astype(np.float32)).to(dev)
            with ph("jqt"):
                nP = args.jqt_pairs
                if args.jqt_square_codes or args.jqt_piece_codes:
                    phi_j, sqt_j = model.enc.forward_tokens(tok_j, glob_j)
                else:
                    phi_j = model.backbone(tok_j, glob_j)
                POL = model.poles.poles
                with torch.no_grad():          # the field's own six outputs, white-POV, frozen
                    zb_pc = model.proj_b(phi_j[:2 * nP])
                    DAr = torch.stack([model.dA(zb_pc, POL[[c]].expand(len(zb_pc), -1))
                                       for c in range(3)], 1)
                    DBr = torch.stack([model.dB(zb_pc, POL[[c]].expand(len(zb_pc), -1))
                                       for c in range(3)], 1)
                    prb = torch.softmax(-DBr / args.basin_temp, 1)
                    y6 = torch.cat([DAr, prb], 1)
                    jqt.update_y_stats(y6)
                    yt = (y6 - jqt.y_mu) / jqt.y_sd
                    E_pc = prb[:, 0] + 0.5 * prb[:, 1]
                    dE = E_pc[nP:2 * nP] - E_pc[:nP]
                # (1) faithfulness: codes must regenerate the field's own evaluations
                h_pc, zq_flat, ids_pc, vloss = jqt.quantize(phi_j[:2 * nP])
                l_vqrec = torch.nn.functional.mse_loss(jqt.dec(zq_flat), yt) + vloss
                # (2) metastability: quiet moves keep their pre-quant latents
                l_pers = code_persistence(h_pc[:nP], h_pc[nP:2 * nP], dE, tau=args.pers_tau)
                # ---- v2 residual streams: square then piece, each explaining the residue
                y_g_det = jqt.dec(zq_flat).detach()
                if args.jqt_square_codes:
                    h_sq, y_sq, ids_sq, vl_sq = jqt.square_stream(sqt_j[:2 * nP])
                    l_sqrec = (torch.nn.functional.mse_loss(y_g_det + y_sq, yt) + vl_sq)
                    l_pers_sq = code_persistence(h_sq[:nP], h_sq[nP:2 * nP], dE,
                                                 tau=args.pers_tau)
                    _sq_flip = float((ids_sq[:nP] != ids_sq[nP:2 * nP]).float().mean())
                if args.jqt_piece_codes:
                    _rows_pc = rows_j[:2 * nP]
                    slots_b = torch.from_numpy(tr.slots[_rows_pc].astype(np.int64)).to(dev)
                    st_b = torch.from_numpy(
                        st_of_game[game_of_row[_rows_pc]].astype(np.int64)).to(dev)
                    h_pcs, y_pc, ids_pcs, vl_pc, alive_pc = jqt.piece_stream(
                        sqt_j[:2 * nP], slots_b, st_b)
                    y_sq_det = y_sq.detach() if args.jqt_square_codes else 0.0
                    l_pcrec = (torch.nn.functional.mse_loss(y_g_det + y_sq_det + y_pc, yt)
                               + vl_pc)
                    both = (alive_pc[:nP] & alive_pc[nP:2 * nP]).float()
                    wq = torch.relu(1.0 - dE.abs() / args.pers_tau)[:, None] * both
                    per = (h_pcs[:nP] - h_pcs[nP:2 * nP]).pow(2).mean(-1)
                    l_pers_pc = (wq * per).sum() / wq.sum().clamp(min=1e-6)
                    _mb = both.bool()
                    _pc_flip = float((ids_pcs[:nP][_mb] != ids_pcs[nP:2 * nP][_mb])
                                     .float().mean()) if _mb.any() else -1.0
                # (3) future codes, embedding space, EMA target (both branches EMA: model.t_enc
                # reads the child board, jqt.t_enc + live-EMA codebooks quantize it)
                with torch.no_grad():
                    phi_t_ch = model.t_enc(tok_j[nP:2 * nP], glob_j[nP:2 * nP])
                    zq_t, ids_ch = jqt.target_codes(phi_t_ch)
                if args.w_jepa > 0:
                    pred_ch = jqt.predict_child(zq_flat[:nP], mid_)
                    l_jepa = jepa_code_prediction(pred_ch, zq_t)
                # jqt5 HURDLE DYNAMICS (Kaveh 2026-08-13): predict WHICH heads change and
                # WHERE they land. Flip BCE per head; destination CE on flipped heads only,
                # warm-up-ramped (index churn: identities drift until the books settle).
                if args.w_flip > 0 or args.w_dest > 0:
                    fl_lg, ds_lg = jqt.predict_flips(zq_flat[:nP], mid_)
                    flip_t = (ids_pc[:nP] != ids_ch).float().detach()
                    l_flip = torch.nn.functional.binary_cross_entropy_with_logits(
                        fl_lg, flip_t)
                    fm = flip_t.bool()
                    ramp = min(1.0, max(0.0, cur_step["s"] / max(args.dest_warmup, 1) - 1.0))
                    if fm.any() and args.w_dest > 0:
                        l_dest = ramp * torch.nn.functional.cross_entropy(
                            ds_lg[fm], ids_ch[fm])
                        _dest_top1 = float((ds_lg[fm].argmax(-1) == ids_ch[fm])
                                           .float().mean())
                    with torch.no_grad():
                        pr_f = torch.sigmoid(fl_lg)
                        _flip_gap = (float(pr_f[fm].mean() - pr_f[~fm].mean())
                                     if fm.any() and (~fm).any() else -1.0)
                    if args.jqt_piece_codes:
                        pf_lg, pd_lg = jqt.pc_predict_flips(h_pcs[:nP], mid_)
                        pflip_t = (ids_pcs[:nP] != ids_pcs[nP:2 * nP]).float().detach()
                        pmask = (alive_pc[:nP] & alive_pc[nP:2 * nP]).float()
                        l_pcflip = (torch.nn.functional.binary_cross_entropy_with_logits(
                            pf_lg, pflip_t, reduction="none") * pmask).sum() \
                            / pmask.sum().clamp(min=1)
                        pfm = pflip_t.bool() & pmask.bool()
                        if pfm.any() and args.w_dest > 0:
                            l_pcdest = ramp * torch.nn.functional.cross_entropy(
                                pd_lg[pfm], ids_pcs[nP:2 * nP][pfm])
                            _pcdest_top1 = float((pd_lg[pfm].argmax(-1)
                                                  == ids_pcs[nP:2 * nP][pfm]).float().mean())
                # (4)+(5) the two concept-goal rulers: "can I activate this concept" asked of
                # LENGTH (censored plies-to-first-activation) and PROBABILITY (first-hit BCE)
                gt_t = torch.from_numpy(ggt).to(dev)
                k1_t = torch.from_numpy(gk1).to(dev)
                k2_t = torch.from_numpy(gk2).to(dev)
                anc = torch.zeros(len(ggt), args.d, device=dev)
                m0 = gt_t == 0
                if m0.any():
                    anc[m0] = jqt.anchors_for(torch.stack([k1_t[m0], k2_t[m0]], 1))
                m1 = gt_t == 1
                if m1.any():
                    anc[m1] = jqt.anchors_for_sq(k1_t[m1], k2_t[m1])
                m2 = gt_t == 2
                if m2.any():
                    anc[m2] = jqt.anchors_for_pc(k1_t[m2], k2_t[m2])
                zb_g = model.proj_b(phi_j[2 * nP:])
                dA_g = model.dA(zb_g, anc)
                dB_g = model.dB(zb_g, anc)
                tpl = torch.from_numpy(gplies).to(dev)
                thit = torch.from_numpy(ghit).to(dev)
                l_cda = censored_plies_loss(torch.log1p(dA_g), tpl, thit)
                if args.two_pole_cdb or args.w_negpole > 0:
                    # jqt5 NEGATIVE POLES: anti-anchor per goal; CDB becomes a two-pole
                    # contrast (committor pattern) and DEAD states subsume onto the anti-pole
                    anc_n = torch.zeros(len(ggt), args.d, device=dev)
                    if m0.any():
                        anc_n[m0] = jqt.anchors_neg_for(
                            torch.stack([k1_t[m0], k2_t[m0]], 1))
                    if m1.any():
                        anc_n[m1] = jqt.anchors_neg_for_sq(k1_t[m1], k2_t[m1])
                    if m2.any():
                        anc_n[m2] = jqt.anchors_neg_for_pc(k1_t[m2], k2_t[m2])
                    dB_gn = model.dB(zb_g, anc_n)
                    if args.two_pole_cdb:
                        l_cdb = first_hit_bce(jqt.activation_logit2(dB_g, dB_gn), thit)
                    else:
                        l_cdb = first_hit_bce(jqt.activation_logit(dB_g), thit)
                    if args.w_negpole > 0:
                        tdead = torch.from_numpy(gdead).to(dev).bool()
                        if tdead.any():
                            l_negpole = torch.log1p(dB_gn[tdead]).mean()
                else:
                    l_cdb = first_hit_bce(jqt.activation_logit(dB_g), thit)
                # per-STEP health timeseries (Kaveh 2026-08-12): perplexity is 8 bincounts and
                # the flip-rate is one comparison on ids already in hand -- cheap enough for
                # every step, and flip-rate IS the metastability curve (baseline 0.80/ply).
                _jqt_perp = jqt.perplexity(ids_pc.detach())
                _jqt_flip = float((ids_pc[:nP] != ids_pc[nP:2 * nP]).float().mean())

        loss = (args.w_mh * l_mh + args.w_mh_cons * l_mh_cons
                + args.w_mh_bell * l_mh_bell
                + args.w_bellman * l_bell
                + args.w_grad * l_grad
                + args.w_region * (l_nll + args.w_repel * l_rep_a)
                + args.w_iqe * (l_q + args.w_repel * l_rep_b) + args.w_confine * l_conf
                + args.w_var * l_var + args.w_cov * l_cov + args.w_l1 * l_l1
                + args.w_res_shrink * l_shrink
                + args.w_basin * l_basin + args.w_polesep * l_polesep
                + args.w_anchor_a * l_anchor_a + args.w_pole_gas_a * l_pole_gas_a
                + args.w_qdistill * l_qdist
                + args.w_hinge_a * l_hinge_a
                + args.w_anchor * l_anchor + args.w_absorb * l_absorb
                + args.w_termrep * l_termrep + args.w_subsume * l_subsume
                + args.w_start * l_start + args.w_start_irr * l_startirr
                + args.w_termcon * (l_termcon_att + l_termcon_rep)
                + args.w_vqrec * l_vqrec + args.w_pers * l_pers + args.w_jepa * l_jepa
                + args.w_flip * (l_flip + l_pcflip) + args.w_dest * (l_dest + l_pcdest)
                + args.w_negpole * l_negpole
                + args.w_cda * l_cda + args.w_cdb * l_cdb
                + args.w_sqrec * l_sqrec + args.w_pcrec * l_pcrec
                + args.w_pers_sq * l_pers_sq + args.w_pers_pc * l_pers_pc
                + args.w_turnpair * l_turn)
        met = {**_audit,
               "vqrec": float(l_vqrec.detach()), "pers": float(l_pers.detach()),
               "jepa": float(l_jepa.detach()), "cda": float(l_cda.detach()),
               "cdb": float(l_cdb.detach()), "jqt_perp": _jqt_perp, "jqt_flip": _jqt_flip,
               "flip": float(l_flip.detach()), "dest": float(l_dest.detach()),
               "pcflipL": float(l_pcflip.detach()), "pcdest": float(l_pcdest.detach()),
               "negpole": float(l_negpole.detach()),
               "flip_gap": _flip_gap, "dest_top1": _dest_top1, "pcdest_top1": _pcdest_top1,
               "sqrec": float(l_sqrec.detach()), "pcrec": float(l_pcrec.detach()),
               "pers_sq": float(l_pers_sq.detach()), "pers_pc": float(l_pers_pc.detach()),
               "sq_flip": _sq_flip, "pc_flip": _pc_flip,
               "turn": float(l_turn.detach()),
               "loss": float(loss.detach()), "nll": float(l_nll.detach()),
               "rep_a": float(l_rep_a.detach()), "quasi": float(l_q.detach()), "n_wall": _n_wall[0],
               "confine": float(l_conf.detach()),
               "tc_att": float(l_termcon_att.detach()), "tc_rep": float(l_termcon_rep.detach()),
               "grad": float(l_grad.detach()), "mh": float(l_mh.detach()), "mh_cons": float(l_mh_cons.detach()),
               "bell": float(l_bell.detach()), "mh_bell": float(l_mh_bell.detach()),
               "anchor_a": float(l_anchor_a.detach()), "pole_gas_a": float(l_pole_gas_a.detach()),
               "qdist": float(l_qdist.detach()),
               "hinge_a": float(l_hinge_a.detach()),
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

    # PER-STEP LOSS LOG, full resolution. standard_train only logs at eval_every, so the losses --
    # which are computed EVERY step anyway -- were being thrown away 249 times out of 250. Writing
    # them to a jsonl costs nothing and gives a real learning curve instead of a 40-point sketch.
    # The expensive gates (an extra forward pass + an eff_rank SVD) stay on --eval-every, because
    # running those every step would roughly double the wall clock to measure the same thing.
    step_log = open(f"{args.out}_steps.jsonl", "a", buffering=1 << 16)

    _last_step = [0]

    def step_fn(model, step):
        _last_step[0] = step
        # Profile only on eval steps: _sync() serialises the device and would otherwise tax every
        # step to measure it (observer effect on the very number we are optimising).
        prof = (step % args.eval_every == 0) or step == 1
        if args.ingest_tsv and step % args.ingest_every == 0:
            ingest()
        if jqt is not None and (step == 1 or step % args.jqt_refresh == 0):
            jqt_refresh(model)
        ph = Phase(dev, prof)
        cur_step["s"] = step
        if args.bf16:
            with torch.autocast(device_type=(dev.type if hasattr(dev, "type") else str(dev).split(":")[0]), dtype=torch.bfloat16):
                loss, met, _ = terms(model, fit, args.batch, n_rev, ph,
                                     audit=(step % args.eval_every == 0) or step == 1)
        else:
            loss, met, _ = terms(model, fit, args.batch, n_rev, ph,
                                 audit=(step % args.eval_every == 0) or step == 1)
        with ph("backward"):
            opt.zero_grad(set_to_none=True)
            loss.backward()
        with ph("opt"):
            opt.step()
            if args.l1_prox > 0:
                model.prox_l1(args.lr * args.l1_prox)   # ISTA step: what actually zeroes coords
            model.update_target()
            if jqt is not None:
                jqt.update_target()
        step_log.write(json.dumps({"s": step, **{k: round(float(v), 5)
                                                  for k, v in met.items()}}) + "\n")
        # HARD GATE against the take-2 failure: the pole terms ran for 20,000 steps against poles
        # that could never move, because attach_poles came after the optimizer was built. Nothing
        # in the loss or the gates noticed. Check directly that the learned poles are moving, and
        # abort early rather than spend another 5.5h discovering it at the end.
        if step == 400 and model.poles is not None and getattr(model.poles, "ending_delta", None) is not None:
            mv = float(model.poles.ending_delta.abs().max())
            if mv < 1e-8 and (args.w_anchor > 0 or args.w_termrep > 0 or args.w_start > 0):
                raise SystemExit(
                    f"ABORT at step 400: ending_delta is still {mv:.2e} with pole terms active -- "
                    f"the learned poles are not in the optimizer. This is the take-2 bug; fix the "
                    f"attach_poles/optimizer ordering before rerunning.")
            print(f"[poles] gate OK at step 400: ending_delta absmax {mv:.4f} (moving)", flush=True)
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
            p_wdl = basin_logp(dP, temperature=args.basin_temp, raw=bool(args.log_gas)).exp()
            g["basin_spread"] = float(p_wdl.std(0).mean())     # ~0 => uniform => collapsed
            g["d_pole_mean"] = float(dP.mean())
            # READ THIS AGAINST ~0.33, NOT 0. The radial anchor's target for rules and every draw
            # is radius 0 -- exact subsumption -- so a position SHOULD sit at distance 0 from its
            # OWN outcome pole and nonzero from the other two, which the mutually-incomparable
            # simplex guarantees is possible. So ~1/3 is HEALTHY, ->1.0 is the collapse (every
            # position dominating all three poles at once, committor uniform). An earlier version
            # of this comment treated any rise as the alarm, which would have flagged the objective
            # working as if it were failing.
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
        if jqt is not None:
            # sidecar checkpoint: the jqt payload rides NEXT TO the field ckpt (never inside it,
            # so every legacy loader keeps working); atomic overwrite, refreshed each eval
            _tmp = f"{args.out}_jqt.pt.tmp"
            torch.save({"state_dict": jqt.state_dict(), "d_in": args.d_model,
                        "heads": args.jqt_heads, "codes": args.jqt_codes, "d": args.d,
                        "square_codes": args.jqt_square_codes,
                        "piece_codes": args.jqt_piece_codes,
                        "y_mu": jqt.y_mu.cpu(), "y_sd": jqt.y_sd.cpu(),
                        "train_args": {k: v for k, v in vars(args).items()
                                       if k.startswith(("jqt", "w_vqrec", "w_pers", "w_jepa",
                                                        "w_cda", "w_cdb", "pers_tau"))}}, _tmp)
            __import__("os").replace(_tmp, f"{args.out}_jqt.pt")
        # Sparsity as an EXACT count, which only means anything because prox_l1 makes true zeros.
        g["l1_support"] = int(model.input_support().sum())
        # EVERY gate metric also lands in a plain file (Kaveh 2026-08-12: "keep track of all
        # metrics for all training steps"). MLflow already gets these, but it degrades to a
        # no-op when unavailable -- and a silent no-op is exactly how metrics get lost.
        with open(f"{args.out}_gates.jsonl", "a") as _gf:
            _gf.write(json.dumps({"s": _last_step[0],
                                  **{k: (round(float(v), 6) if isinstance(v, (int, float))
                                         else v) for k, v in g.items()}}) + "\n")
        # DIRECT ratchet readout, free here: the observed forward distance against its own reversal.
        g["rev_ratio"] = float(met["d_rev"] / max(met["d_fwd"], 1e-6))
        model.train()
        return g

    cfg = TrainConfig(out=args.out, steps=args.steps, ckpt_every=args.ckpt_every,
                      eval_every=args.eval_every, experiment="reach_probability",
                      run_name=f"reach_vit_d{args.d_model}x{args.layers}_g{args.games}",
                      device=str(dev), extra={"cfg": cfg_dict})
    last = standard_train(step_fn, net, cfg, args=args, gates_fn=gates_fn)
    step_log.close()
    print(f"[curve] per-step losses -> {args.out}_steps.jsonl", flush=True)

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
