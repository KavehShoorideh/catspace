#!/usr/bin/env python
"""concept_state_field.py -- the IQE moved ONE LAYER DOWNSTREAM (Kaveh 2026-08-15:
"let's replace the iqe and send it one layer downstream. i.e. let's get concepts from some
board trunk. concepts should be predictive of next concepts. then we define states as the set
of concepts active. then we iqe on those states, like we do now, trying to nail down the ply
count poorly but at least get committor right. I don't wanna evaluate sf on every state yet as
learning labels")

THE INVERSION. Until now the quasimetric was primary and concepts were quantised off z_B:

    board -> phi -> z_B -> IQE geometry            concepts quantised the geometry

Here the concepts are primary and the geometry is a readout of the DISCRETE state:

    board -> phi -> VQ concepts -> STATE = the set of active concepts -> IQE

WHAT THIS BUYS. The state is now genuinely discrete: at most K^H distinct states, and the
geometry is PIECEWISE CONSTANT on concept cells. Two positions with the same concepts are the
same point, by construction -- which is the "subgoal = invariance class" definition made
structural rather than hoped for.

WHAT IT COSTS, STATED UP FRONT. Sibling discrimination now depends entirely on the vocabulary
being fine enough. If a blunder and the best move land in the same concept cell, they receive
literally identical value -- not approximately, exactly. probe_value_proxy measured the current
committor at within-parent spearman +0.096 with sibling spread 22% of truth, so this design
does not inherit a good number; it inherits a bad one and makes the cause explicit and
measurable. `--eval-sf` reports whether the partition separates good moves from bad, using the
Stockfish shards STRICTLY AS A YARDSTICK and never as a training label, per the instruction.

NO STOCKFISH IN THE LOSS. Every training signal here is free from the corpus:
    committor   the game's actual W/D/L result            (outcome, the thing to get right)
    dynamics    the next position's own codes             (concepts predict next concepts)
    ply         observed ply gaps inside a game           (deliberately weak; "poorly")
    commitment  the VQ's own term

    .venv/bin/python -m ...concept_state_field --steps 400 --games 400     # smoke
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConceptStateField(nn.Module):
    """board -> concepts -> state -> quasimetric."""

    def __init__(self, d_model=192, layers=4, tr_heads=8, heads=16, codes=16, d_code=32,
                 d_state=256, components=16, ply_cap=60.0, square_codes=32,
                 d_fine=32, kmeans_init=True, dead_code=2, n_poles=3, proj_h=512,
                 pole_init=2.0, temp_init=1.0):
        super().__init__()
        from vector_quantize_pytorch import VectorQuantize
        from catspace.research.components.encoder.approaches.jepa_tokenizer.src.jepa import (
            JepaEncoder)
        from catspace.research.components.encoder.approaches.jepa_tokenizer.src.iqe import IQE

        # ---- ANTI-COLLAPSE MACHINERY, previously never switched on anywhere in this repo.
        # VectorQuantize's OWN defaults are kmeans_init=False and threshold_ema_dead_code=0
        # (the `2` in the file belongs to the inner Codebook class, which VectorQuantize
        # overrides), so every codebook here has been starting from a random normal cloud with
        # no dead-code recovery.
        #
        # kmeans_init: seed the codebook by running k-means on the FIRST training batch, so the
        #   codes start inside the data instead of wherever randn put them. Measured motivation:
        #   jqt8's codebooks sat at 1/10th the radius of their own latent cloud.
        #
        # threshold_ema_dead_code: `cluster_size` is an EMA (decay 0.9) of how many batch
        #   vectors each code won. expire_codes_() marks every code with cluster_size below the
        #   threshold as DEAD and `replace()` overwrites its embedding with a randomly chosen
        #   REAL latent from the current batch, resetting cluster_size/embed_avg so it competes
        #   again. This breaks the lock-in that has produced every collapse we measured: one
        #   code wins everything, so only that code is EMA-updated and the rest stay frozen at
        #   their init radius forever. FSQ's paper measures VQ usage at 0.78% WITHOUT this
        #   splitting versus 81% with it.
        vq_kw = dict(decay=0.9, commitment_weight=0.25, kmeans_init=bool(kmeans_init),
                     threshold_ema_dead_code=int(dead_code))
        self.heads, self.codes, self.d_code = heads, codes, d_code
        self.trunk = JepaEncoder(d_model, layers, tr_heads)
        self.enc = nn.Sequential(nn.Linear(d_model, 256), nn.GELU(),
                                 nn.Linear(256, heads * d_code))
        self.vq = nn.ModuleList([VectorQuantize(dim=d_code, codebook_size=codes, **vq_kw)
                                 for _ in range(heads)])
        # STATE = a function of the CODES ALONE. zq carries the straight-through gradient, and
        # takes only `codes` distinct values per head, so z_state is piecewise constant on
        # concept cells exactly as intended. Head identity is preserved (concatenation, not a
        # permutation-invariant sum) because head h is a specific slot, not an interchangeable
        # set member.
        # SQUARE CONCEPTS (Kaveh 2026-08-15: "don't we have board square concepts? are you
        # including those in the prediction"). They were missing from the first version and
        # that is almost certainly why the state could not separate siblings: a move changes
        # two squares, and a POOLED global embedding barely registers that, so ~30 legal
        # children collapsed to ~2 distinct states (measured `cells` 1.85-2.72). One code per
        # square from a shared book makes the change explicit at the from/to squares.
        self.square_codes, self.d_fine = square_codes, d_fine
        if square_codes:
            self.sq_proj = nn.Linear(d_model, d_fine)
            self.sq_vq = VectorQuantize(dim=d_fine, codebook_size=square_codes,
                                        **vq_kw)
        # RESIDUAL ACROSS GRANULARITIES (restoring jqt's design, which the first version
        # regressed into a concatenation). jqt does it as:
        #     y_g_det = geo_dec(zq_geo).detach()
        #     l_sqrec = mse(y_g_det + y_sq, yt)        <- square is graded on the RESIDUE
        # i.e. each granularity contributes ADDITIVELY to one shared target, and the coarser
        # stream is DETACHED inside the finer stream's loss. Two consequences, both the point:
        # the global stream alone must explain what it can (so it cannot be made redundant),
        # and the square stream has no incentive to re-encode what global already carries --
        # "collinearity removed by construction".
        # Concatenating instead let the 64x32 square stream explain everything on its own and
        # the 16 global heads died: perplexity 2.02 -> 1.22, ~2 of 16 codes.
        # DEEPER/WIDER PROJECTOR into the IQE (Kaveh 2026-08-16: "increase the parameter size
        # of the iqe layer"). The IQE module itself holds only two scalars (alpha_logit,
        # log_scale) -- its capacity lives in the WIDTH of the space it measures and the
        # projector that builds it. QRL's reference is a 256->1024->1024->2048 projector into a
        # 64-component IQE; ours was a 2-layer MLP into 24-d rulers, ~40x smaller, and now has
        # 267 poles to keep apart.
        self.state_g = nn.Sequential(nn.Linear(heads * d_code, proj_h), nn.GELU(),
                                     nn.Linear(proj_h, proj_h), nn.GELU(),
                                     nn.Linear(proj_h, d_state))
        if square_codes:
            self.state_sq = nn.Sequential(nn.Linear(64 * d_fine, proj_h), nn.GELU(),
                                          nn.Linear(proj_h, proj_h), nn.GELU(),
                                          nn.Linear(proj_h, d_state))
            # zero-init the residual so training STARTS at "the square stream says nothing",
            # the same discipline as every other residual in this repo (PoleBank.ending_delta,
            # qhead.proj_delta): the data has to push it off zero.
            nn.init.zeros_(self.state_sq[-1].weight); nn.init.zeros_(self.state_sq[-1].bias)
        h = d_state // 2
        assert d_state % 2 == 0 and h % components == 0, (
            f"d_state={d_state} splits into two {h}-d rulers; each must divide into "
            f"{components} IQE components (try --d-state {components * 2 * 2})")
        self.iqe_a = IQE(h, components=components)      # LENGTH ruler (plies) -- weak on purpose
        self.iqe_b = IQE(h, components=components)      # OUTCOME ruler -> committor
        # ONE POLE PER RESOLUTION CLASS (Kaveh 2026-08-16: "allow it to learn a pole for all
        # 5 pieces endgame types"). Previously three poles for W/D/L. Now one learnable point
        # per class of the tablebase-entrance vocabulary -- 60 endgame types (KRP|KR, KRPP|K,
        # ...) + OTHER + the three WDL resolution modes for games that end before simplifying.
        # The readout is structurally unchanged: softmax(-dB(z, pole_k)/temp). What changes is
        # that "where does this position sit in the space" now answers WHICH ENDGAME rather
        # than merely how good, and each endgame type gets its own anchor to be measured
        # against -- which is the whole point of a geometry over a classifier.
        self.n_poles = int(n_poles)
        # POLE INIT SCALE AND TEMPERATURE ARE A JOINT CONSTRAINT, not two free knobs.
        # MEASURED (placing z exactly ON its target pole -- the most favourable case there is,
        # so this is an upper bound on what any encoder could achieve):
        #     init 0.5, T 5.0  -> best P(correct) 0.0061   <- chance is 1/267 = 0.0037
        #     init 0.5, T 0.25 -> 0.9825
        #     init 2.0, T 1.0  -> 0.9856
        #     init 8.0, T 5.0  -> 0.9157
        # csf7 ran at (0.5, 5.0) and sat at chance for 2,000 steps because a perfect encoder
        # would also have sat at 0.6%. 267 random points in 1024-d are all nearly equidistant,
        # so the pole cloud's whole spread was 20-25 units while ~33 were needed to be
        # confident; the softmax stayed flat, and a flat softmax gives the poles almost no
        # gradient, which is why they never left their init radius.
        self.poles = nn.Parameter(torch.randn(self.n_poles, d_state) * pole_init)
        # LEARNABLE TEMPERATURE (Kaveh 2026-08-16: "make it learnable"). basin_temp was a
        # hardcoded 5.0 inherited from the 3-pole W/D/L committor, and it was already MEASURED
        # to cap that readout: probe_value_proxy found the committor compressed into
        # [0.247, 0.768] because softmax(-dB/5) cannot saturate unless pole distances differ by
        # many multiples of 5. With 267 poles it is worse -- 90% confidence needs the nearest
        # pole ~39 distance units clear of the rest at T=5, against ~11 at three poles. Learned
        # in log space, initialised at the old value so nothing shifts at step 0.
        self.log_temp = nn.Parameter(torch.tensor(float(np.log(temp_init))))
        # dynamics: (state, move) -> next position's code in every head
        # all three share a width because they are SUMMED, as MoveHead does it (64x32, 64x32,
        # 5x32 -> one 32-d action vector)
        self.mv_from = nn.Embedding(64, 32); self.mv_to = nn.Embedding(64, 32)
        self.mv_promo = nn.Embedding(5, 32)
        self.dyn = nn.Sequential(nn.Linear(d_state + 32, 512), nn.GELU(),
                                 nn.Linear(512, heads * codes))
        # concepts predict NEXT concepts at both granularities; the square head is where a
        # move's actual effect lives (from-square empties, to-square changes)
        if square_codes:
            self.dyn_sq = nn.Sequential(nn.Linear(d_state + 32, 512), nn.GELU(),
                                        nn.Linear(512, 64 * square_codes))
        self.ply_cap = ply_cap

    # ---- concepts -------------------------------------------------------------------
    def quantize(self, phi):
        h = self.enc(phi).view(len(phi), self.heads, self.d_code)
        qs, ids, vl = [], [], 0.0
        for i in range(self.heads):
            q, idx, l = self.vq[i](h[:, i])
            qs.append(q); ids.append(idx); vl = vl + l
        return torch.cat(qs, -1), torch.stack(ids, 1), vl

    def encode(self, tok, glob):
        """-> (z_g, z_sq, global ids (B,H), vq loss, square ids (B,64) or None).

        Returns the two granularities SEPARATELY so the caller can form both the full state
        (z_g + z_sq, for every readout) and the residual state (z_g.detach() + z_sq, which is
        the only thing the square path is graded on). forward_tokens costs the SAME as forward
        -- the per-square outputs are already computed by the trunk and were being discarded."""
        phi, sq_tok = self.trunk.forward_tokens(tok, glob)
        zq, ids, vl = self.quantize(phi)
        z_g = self.state_g(zq)
        z_sq, ids_sq = None, None
        if self.square_codes:
            q_sq, ids_sq, vl_sq = self.sq_vq(self.sq_proj(sq_tok))
            vl = vl + vl_sq
            z_sq = self.state_sq(q_sq.reshape(len(phi), -1))
        return z_g, z_sq, ids, vl, ids_sq

    @staticmethod
    def full(z_g, z_sq):
        """the state every readout uses."""
        return z_g if z_sq is None else z_g + z_sq

    @staticmethod
    def residual(z_g, z_sq):
        """what the SQUARE path is graded on: the coarse contribution is a FIXED offset."""
        return z_g.detach() if z_sq is None else z_g.detach() + z_sq

    def dA(self, u, v):
        k = u.shape[-1] // 2
        return self.iqe_a(u[..., :k], v[..., :k])

    def dB(self, u, v):
        k = u.shape[-1] // 2
        return self.iqe_b(u[..., k:], v[..., k:])

    def resolution(self, z, temp=None):
        """-> (N, n_poles): P(this game resolves at pole k), from OUTCOME-ruler distance.

        Computed in ONE batched call rather than a python loop over poles -- at 3 poles the
        loop was free, at 64 it would be 64 IQE evaluations per forward."""
        n, P = len(z), self.n_poles
        zz = z.repeat_interleave(P, 0)                      # (n*P, d)
        pp = self.poles.unsqueeze(0).expand(n, -1, -1).reshape(n * P, -1)
        d = self.dB(zz, pp).view(n, P)
        t = self.log_temp.exp() if temp is None else temp
        return torch.softmax(-d / t.clamp(min=1e-3) if torch.is_tensor(t) else -d / t, 1)

    # kept so older call sites and checkpoints keep working
    def committor(self, z, temp=None):
        return self.resolution(z, temp)

    def predict_next(self, z, mid):
        """-> (global logits (B,H,codes), square logits (B,64,square_codes) or None)."""
        e = self.mv_from(mid[:, 0]) + self.mv_to(mid[:, 1]) + self.mv_promo(mid[:, 2])
        x = torch.cat([z, e], -1)
        g = self.dyn(x).view(len(z), self.heads, self.codes)
        s = (self.dyn_sq(x).view(len(z), 64, self.square_codes)
             if self.square_codes else None)
        return g, s


def perplexity(ids, codes):
    ps = []
    for h in range(ids.shape[1]):
        c = torch.bincount(ids[:, h], minlength=codes).float()
        p = c / c.sum().clamp(min=1)
        ps.append(float(torch.exp(-(p * (p + 1e-9).log()).sum())))
    return float(np.mean(ps)), float(np.mean([len(torch.unique(ids[:, h]))
                                              for h in range(ids.shape[1])]))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--games", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=192)
    ap.add_argument("--max-plies", type=int, default=400)
    ap.add_argument("--heads", type=int, default=16)
    ap.add_argument("--codes", type=int, default=16)
    ap.add_argument("--square-codes", type=int, default=32,
                    help="per-square concept vocabulary; 0 = pooled-global only "
                         "(which measured cells~2, i.e. cannot rank moves)")
    ap.add_argument("--d-model", type=int, default=192)
    ap.add_argument("--d-state", type=int, default=256,
                    help="state width; splits into two IQE rulers of d_state/2. Was 48 (two "
                         "24-d rulers) -- QRL's reference IQE is 2048-d, so ours was ~40x "
                         "smaller and had to separate 64 poles.")
    ap.add_argument("--pole-init", type=float, default=2.0,
                    help="stddev of the pole initialisation. With T, this decides whether a "
                         "confident call is REACHABLE at all: (0.5, T=5) caps any encoder at "
                         "0.6%% on 267 classes; (2.0, T=1) reaches 98.6%%.")
    ap.add_argument("--proj-h", type=int, default=512,
                    help="hidden width of the 3-layer projector feeding the IQE")
    ap.add_argument("--components", type=int, default=16,
                    help="IQE components per ruler; d_state/2 must divide by this")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--basin-temp", type=float, default=1.0,
                    help="INITIAL softmax temperature only. It is a learned parameter "
                         "(log_temp); passing it at the call sites silently pinned it -- "
                         "verified by log_temp sitting bit-identical to its init after 40 "
                         "steps while iqe_b.log_scale moved.")
    ap.add_argument("--w-outcome", type=float, default=3.0, help="committor: GET THIS RIGHT")
    ap.add_argument("--w-dyn", type=float, default=2.0, help="concepts predict next concepts")
    ap.add_argument("--w-ply", type=float, default=0.2, help="deliberately weak ('poorly')")
    ap.add_argument("--ply-gap", type=int, default=8, help="max gap for the dA pair sample")
    ap.add_argument("--labels", default="tb_entry_5-4-3men_{n}.npz",
                    help="TB-entrance label file, {n} = corpus row count. Colour-EXPLICIT "
                         "white-POV types (KRP|KR is not KR|KRP -- opposite propositions), so "
                         "the vocabulary is ~267, not the 139 colour-folded Syzygy tables.")
    ap.add_argument("--n-piecedown", type=int, default=45906,
                    help="piece-down games folded into the corpus. The cache key is the FULL "
                         "build signature, so this must match an existing cache to reuse it: "
                         "games=4000 + n_piecedown=45906 -> the 49,905-game / 4,652,261-position "
                         "corpus jqt6-jqt8 used, which already has its transitions built. "
                         "Running 0 here silently built a 600-game corpus instead and is why "
                         "csf5 overfit.")
    ap.add_argument("--test-frac", type=float, default=0.1,
                    help="fraction of GAMES held out for the single final test read")
    ap.add_argument("--val-frac", type=float, default=0.1,
                    help="fraction of GAMES held out (never positions -- see note "
                         "at the split)")
    ap.add_argument("--kmeans-init", type=int, default=1,
                    help="seed every codebook with k-means on the first batch "
                         "instead of a random normal cloud")
    ap.add_argument("--dead-code", type=int, default=2,
                    help="threshold_ema_dead_code: any code whose EMA assignment "
                         "count falls below this is RESEEDED from a real latent in "
                         "the current batch. 2 is the library's own Codebook "
                         "default, and now ours. 0 disables (what this repo ran "
                         "everywhere until 2026-08-16).")
    ap.add_argument("--stratify", type=int, default=1,
                    help="sample the committor batch EQUALLY from WIN/DRAW/LOSS instead of at "
                         "the corpus rate (measured 12.4/78.9/8.7). Without this the majority "
                         "class supplies ~79%% of the gradient and 'always draw' is a near "
                         "optimum. NOTE the consequence: the committor is then calibrated to a "
                         "UNIFORM prior, not the corpus prior -- to read a true P(outcome) add "
                         "log(corpus prior) before the softmax. Accuracy is reported against "
                         "the 33.3%% balanced baseline, which is why it looks lower and means "
                         "more.")
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--ckpt-every", type=int, default=1000,
                    help="write <out>_step{N}.pt this often. 0 disables. At 20k steps (~7h) a "
                         "single end-of-run save is a data-loss risk, not a style choice.")
    ap.add_argument("--eval-sf", type=int, default=1,
                    help="report the Stockfish within-parent yardstick. EVALUATION ONLY -- "
                         "never enters the loss (Kaveh: no SF as learning labels yet)")
    ap.add_argument("--sf-parents", type=int, default=400)
    ap.add_argument("--out", default="artifacts/experiments/csf1")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from catspace.research.components.encoder.approaches.reach_probability.src import (
        trajectories as T)
    from catspace.io import paths

    torch.manual_seed(args.seed)
    dev = args.device
    t0 = time.time()
    tr = T.build(n_human=0, n_sf=args.games, seed=args.seed, cache=True,
                 max_plies=args.max_plies, n_piecedown=args.n_piecedown)
    gor = tr.game_of_row(); por = tr.ply_of_row()
    # ---- TARGET: where does this game RESOLVE? -------------------------------------
    # Not the 3-class final result (78.9% draw on the balanced pool, near-constant and
    # near-unlearnable) but the tablebase-entrance vocabulary: which 5-man endgame this game
    # falls into, or -- for the 78% of games that end before simplifying -- its W/D/L result.
    # 64 classes, largest 30.7%. Built by gen_tb_entry_labels.py.
    out_w = tr.outcome_of_row_white()          # still needed for per-pole values
    _lp = paths.derived(args.labels.format(n=len(tr.tok)))
    _lz = np.load(_lp, allow_pickle=True)
    VOCAB = [str(x) for x in _lz["vocab"]]
    cls = _lz["tb_class"].astype(np.int64)     # -1 = already inside the tablebase
    tb_plies = _lz["tb_plies"]
    n_cls = len(VOCAB)
    print(f"[csf] target = TB-entrance resolution, {n_cls} classes; "
          f"{int((cls>=0).sum()):,} labelled ({100*(cls>=0).mean():.1f}%)", flush=True)
    # ---- TRAIN / VAL / TEST, SPLIT BY GAME ------------------------------------------
    # By GAME, never by position: consecutive plies are nearly identical and share one label,
    # so a positional split leaks the answer across itself and "validation" becomes training
    # accuracy under another name.
    # THREE ways, not two: val is read every eval and therefore steers our decisions (when to
    # stop, which config to keep), which makes it an optimisation target and no longer an
    # unbiased estimate. test is touched EXACTLY ONCE, at the end.
    _rs = np.random.default_rng(args.seed)
    _perm = _rs.permutation(len(tr.start))
    _nv = max(1, int(args.val_frac * len(tr.start)))
    _nt = max(1, int(args.test_frac * len(tr.start)))
    split_of_game = np.zeros(len(tr.start), np.int8)      # 0 train / 1 val / 2 test
    split_of_game[_perm[:_nv]] = 1
    split_of_game[_perm[_nv:_nv + _nt]] = 2
    spl = split_of_game[gor]                               # per POSITION
    by_cls  = [np.flatnonzero((cls == k) & (spl == 0)) for k in range(n_cls)]
    by_cls_val  = [np.flatnonzero((cls == k) & (spl == 1)) for k in range(n_cls)]
    by_cls_test = [np.flatnonzero((cls == k) & (spl == 2)) for k in range(n_cls)]
    ok = np.flatnonzero((cls >= 0) & (spl == 0))
    print(f"[csf] split by GAME  train {int((split_of_game==0).sum())} / "
          f"val {_nv} / test {_nt} games  ->  "
          f"{int((spl==0).sum()):,} / {int((spl==1).sum()):,} / {int((spl==2).sum()):,} "
          f"positions", flush=True)
    _trk = np.ascontiguousarray(
        np.concatenate([tr.tok[spl == 0], tr.glob[spl == 0]], 1).astype(np.uint8))
    train_keys = np.unique(_trk.view(np.dtype((np.void, _trk.shape[1]))).ravel())
    print(f"[csf] {len(train_keys):,} distinct TRAIN board keys for yardstick leak-screening",
          flush=True)
    # A class absent from a split cannot be trained or scored there, so it is dropped from
    # sampling rather than asserted away -- with 64 classes and a 10% split, rare endgames can
    # legitimately miss. Reported so the loss is never silently over a different vocabulary.
    live      = [k for k in range(n_cls) if len(by_cls[k])]
    live_val  = [k for k in range(n_cls) if len(by_cls_val[k])]
    live_test = [k for k in range(n_cls) if len(by_cls_test[k])]
    print(f"[csf] classes present: train {len(live)}/{n_cls}  val {len(live_val)}  "
          f"test {len(live_test)}", flush=True)
    # PER-POLE VALUE, needed to turn a 64-way resolution distribution back into one number
    # for move ranking: E[score] = sum_k P(k) * V(k). V(k) is the mean white score of TRAIN
    # games that resolved at k -- free from the corpus, no Stockfish, and for the endgame
    # classes it is close to the true value of entering that table since everything after
    # entry is solved. Train-split only, so the yardstick cannot read val/test outcomes.
    _sc = np.where(out_w == T.WIN, 1.0, np.where(out_w == T.DRAW, 0.5, 0.0))
    pole_val = np.full(n_cls, 0.5, np.float32)
    for k in range(n_cls):
        m = (cls == k) & (spl == 0)
        if m.any():
            pole_val[k] = _sc[m].mean()
    print(f"[csf] pole values from TRAIN: WDL:W {pole_val[VOCAB.index('WDL:W')]:.2f} "
          f"WDL:D {pole_val[VOCAB.index('WDL:D')]:.2f} "
          f"WDL:L {pole_val[VOCAB.index('WDL:L')]:.2f} | endgame range "
          f"{pole_val.min():.2f}-{pole_val.max():.2f}", flush=True)
    _b = np.bincount(cls[(cls >= 0) & (spl == 0)], minlength=n_cls)
    print(f"[csf] train majority class {VOCAB[int(_b.argmax())]} at "
          f"{100*_b.max()/_b.sum():.1f}%  (balanced baseline {100/len(live):.1f}%)", flush=True)

    # the transitions cache is keyed by ROW COUNT, so a new corpus needs a new one; build it
    # rather than die (no script in the repo generated this file before gen_game_transitions)
    from catspace.research.tools.training_infra import gen_game_transitions as GT
    gz = GT.ensure(tr, T.load_sf_games(args.games, args.seed, max_plies=args.max_plies))
    par, mid = gz["par"], gz["mid"]
    # LEAK FIX: the dynamics and ply terms sampled EVERY transition, including held-out games,
    # so the encoder saw val/test boards during training even though their outcome labels were
    # withheld. Restrict to train games.
    _ptr = split_of_game[gor[par]] == 0
    par, mid = par[_ptr], mid[_ptr]
    print(f"[csf] {len(tr):,} games | {len(tr.tok):,} positions | {len(ok):,} outcome-labelled "
          f"| {len(par):,} transitions [{time.time()-t0:.0f}s]", flush=True)

    net = ConceptStateField(d_model=args.d_model, heads=args.heads, codes=args.codes,
                            d_state=args.d_state, components=args.components,
                            square_codes=args.square_codes,
                            kmeans_init=bool(args.kmeans_init),
                            dead_code=args.dead_code, n_poles=n_cls,
                            proj_h=args.proj_h, pole_init=args.pole_init,
                            temp_init=args.basin_temp).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    print(f"[csf] {sum(p.numel() for p in net.parameters()):,} params | state space "
          f"{args.codes}^{args.heads}", flush=True)

    def batch(rows):
        return (torch.from_numpy(tr.tok[rows].astype(np.int64)).to(dev),
                torch.from_numpy(tr.glob[rows].astype(np.float32)).to(dev))

    rng = np.random.default_rng(args.seed)
    # LINE-buffered, not 16KB-buffered: at ~200 B/row and one row per 20 steps a 16 KB buffer
    # does not flush until ~1,600 steps, so the metrics file sits EMPTY for the first half hour
    # of a long run and loses everything since the last flush if the run is killed. Same class
    # of mistake as piping progress through grep. (The .log written by `python -u` is
    # unbuffered and remains the authoritative record either way.)
    log = open(f"{args.out}_steps.jsonl", "w", buffering=1)   # NOT append: one run, one file
    for step in range(1, args.steps + 1):
        net.train()
        # --- (1) committor on outcome-labelled positions (the free label) ----------------
        if args.stratify:
            # pick a CLASS uniformly, then a position within it. With 64 classes and batch 192
            # a fixed per-class quota would give 3 each; sampling this way keeps the batch size
            # independent of the class count and still equalises class exposure in expectation.
            ks = rng.choice(live, args.batch)
            r = np.array([by_cls[k][rng.integers(0, len(by_cls[k]))] for k in ks])
        else:
            r = ok[rng.integers(0, len(ok), args.batch)]
        tk, gb = batch(r)
        z_g, z_sq, ids, vl, ids_sq = net.encode(tk, gb)
        y = torch.from_numpy(cls[r]).to(dev)
        # jqt's residual pattern: the coarse stream is graded ALONE (so it cannot become
        # redundant), the fine stream is graded on coarse.detach() + fine (so it learns only
        # the residue). Same target for both, exactly as l_vqrec / l_sqrec do it.
        p3 = net.resolution(net.full(z_g, z_sq))
        l_out = F.nll_loss((net.resolution(z_g) + 1e-8).log(), y)
        if z_sq is not None:
            l_out = l_out + F.nll_loss(
                (net.resolution(net.residual(z_g, z_sq)) + 1e-8).log(), y)

        # --- (2) concepts predict next concepts ------------------------------------------
        ti = rng.integers(0, len(par), args.batch)
        pr = par[ti]
        tk_p, gb_p = batch(pr)
        tk_c, gb_c = batch(pr + 1)
        zg_p, zsq_p, ids_p, vl_p, _ = net.encode(tk_p, gb_p)
        z_p = net.full(zg_p, zsq_p)
        with torch.no_grad():
            _, _, ids_c, _, sq_c = net.encode(tk_c, gb_c)    # target codes, detached
        mv = torch.from_numpy(mid[ti].astype(np.int64)).to(dev)
        lg, lg_sq = net.predict_next(z_p, mv)
        l_dyn = F.cross_entropy(lg.reshape(-1, args.codes), ids_c.reshape(-1))
        if lg_sq is not None and sq_c is not None:
            l_dyn = l_dyn + F.cross_entropy(lg_sq.reshape(-1, args.square_codes),
                                            sq_c.reshape(-1))

        # --- (3) ply, deliberately weak ---------------------------------------------------
        g = gor[pr]; same = np.flatnonzero(gor[np.clip(pr + args.ply_gap, 0, len(gor) - 1)] == g)
        if len(same) > 8:
            a = pr[same]; k = rng.integers(1, args.ply_gap + 1, len(same))
            b = np.clip(a + k, 0, len(gor) - 1)
            keep = gor[b] == gor[a]
            a, b, k = a[keep], b[keep], k[keep]
            za = net.full(*net.encode(*batch(a))[:2])
            zb_ = net.full(*net.encode(*batch(b))[:2])
            l_ply = F.mse_loss(net.dA(za, zb_),
                               torch.from_numpy(k.astype(np.float32)).to(dev))
        else:
            l_ply = torch.zeros((), device=dev)

        loss = (args.w_outcome * l_out + args.w_dyn * l_dyn + args.w_ply * l_ply
                + vl + vl_p)
        opt.zero_grad(); loss.backward(); opt.step()

        if step % 20 == 0 or step == 1:
            with torch.no_grad():
                perp, used = perplexity(ids.detach(), args.codes)
                acc = float((p3.argmax(1) == y).float().mean())
                dyn_acc = float((lg.argmax(-1) == ids_c).float().mean())
                sq_acc = (float((lg_sq.argmax(-1) == sq_c).float().mean())
                          if lg_sq is not None else -1.0)
                sq_perp = (perplexity(ids_sq.detach(), args.square_codes)[0]
                           if ids_sq is not None else -1.0)
            rec = {"s": step, "loss": float(loss.detach()), "outcome": float(l_out.detach()),
                   "dyn": float(l_dyn.detach()), "ply": float(l_ply.detach()),
                   "vq": float(vl.detach()), "perp": perp, "used": used,
                   "out_acc": acc, "dyn_acc": dyn_acc, "sq_acc": sq_acc,
                   "sq_perp": sq_perp}
            log.write(json.dumps(rec) + "\n")
            if step % 100 == 0 or step == 1:
                print(f"[csf] {step:5d} loss {float(loss):8.3f} | resolution acc {acc:.3f} "
                      f"| next-code acc {dyn_acc:.3f} sq {sq_acc:.3f} | perp {perp:5.2f} "
                      f"sq_perp {sq_perp:5.2f} used {used:4.1f}/{args.codes} | {time.time()-t0:.0f}s", flush=True)

        if step % args.eval_every == 0 or step == args.steps:
            net.eval()
            with torch.no_grad():
                _per = 256 // 3
                _kv = rng.choice(live_val, 256)
                _rv = np.array([by_cls_val[k][rng.integers(0, len(by_cls_val[k]))]
                                for k in _kv])
                _zg, _zs, _, _, _ = net.encode(*batch(_rv))
                _p = net.resolution(net.full(_zg, _zs))
                _yv = torch.from_numpy(cls[_rv]).to(dev)
                val_acc = float((_p.argmax(1) == _yv).float().mean())
            net.train()
            print(f"[csf] {step:5d} VAL resolution {val_acc:.3f} "
                  f"(balanced baseline {1/len(live_val):.3f}) | train {acc:.3f}", flush=True)
            log.write(json.dumps({"s": step, "val_acc": val_acc}) + "\n")

        if args.ckpt_every and step % args.ckpt_every == 0:
            # A 20k-step run is ~7h. Saving only at the end means a crash at hour 6 loses
            # everything, so the ladder is not optional at this length.
            torch.save({"state_dict": net.state_dict(), "step": step, "args": vars(args)},
                       f"{args.out}_step{step}.pt")

        if args.eval_sf and (step % args.eval_every == 0 or step == args.steps):
            sf = sf_yardstick(net, args, dev, train_keys, want=1, pole_val=pole_val)
            if sf:
                print(f"[csf] {step:5d} SF YARDSTICK (eval only, never a label): "
                      f"within-parent spearman {sf['sp']:+.3f} | top-1 {sf['top1']:.1%} "
                      f"| distinct concept cells among siblings {sf['cells']:.2f}", flush=True)
                log.write(json.dumps({"s": step, **{f"sf_{k}": v for k, v in sf.items()}}) + "\n")

    # ---- TEST, TOUCHED EXACTLY ONCE ------------------------------------------------
    # val steered every decision above, so it is no longer an unbiased estimate. This is the
    # only read of test in the whole script.
    net.eval()
    with torch.no_grad():
        _per = 512 // 3
        _kt = rng.choice(live_test, 512)
        _rt = np.array([by_cls_test[k][rng.integers(0, len(by_cls_test[k]))] for k in _kt])
        _zg, _zs, _, _, _ = net.encode(*batch(_rt))
        _p = net.resolution(net.full(_zg, _zs))
        test_acc = float((_p.argmax(1) == torch.from_numpy(cls[_rt]).to(dev)).float().mean())
    sf_te = sf_yardstick(net, args, dev, train_keys, want=2, pole_val=pole_val)
    print(f"\n[csf] ===== TEST (single touch) =====", flush=True)
    print(f"[csf]   resolution {test_acc:.3f}   (balanced baseline "
          f"{1/len(live_test):.3f}, {len(live_test)} classes)", flush=True)
    if sf_te:
        print(f"[csf]   move ranking: within-parent spearman {sf_te['sp']:+.3f} | "
              f"top-1 {sf_te['top1']:.1%} | cells {sf_te['cells']:.2f} | n={sf_te['n']}",
              flush=True)
    log.write(json.dumps({"s": args.steps, "test_acc": test_acc,
                          **{f"test_{k}": v for k, v in (sf_te or {}).items()}}) + "\n")
    torch.save({"state_dict": net.state_dict(), "step": args.steps,
                "args": vars(args)}, f"{args.out}_latest.pt")
    print(f"[csf] done -> {args.out}_latest.pt [{time.time()-t0:.0f}s]", flush=True)


def sf_yardstick(net, args, dev, train_keys=None, want=1, pole_val=None):
    """Stockfish shards as a MEASURING STICK ONLY -- no gradient, never in the loss.

    Reports the question this architecture lives or dies on: among the legal children of one
    position, does the concept state rank the moves? Plus `cells`, the mean number of DISTINCT
    concept states among a parent's children -- if that is ~1 the partition cannot rank moves
    even in principle, which is the structural risk of moving the IQE downstream."""
    import glob as _glob
    from scipy.stats import spearmanr
    fs = sorted(_glob.glob("data/derived/wdl_labels/*.npz"))
    if not fs:
        return None
    toks, globs, wdls, cnt = [], [], [], []
    for f in fs:
        z = np.load(f)
        toks.append(z["tok"]); globs.append(z["glob"]); wdls.append(z["wdl"])
        cnt.append(np.diff(z["off"]))
    toks, globs, wdls = np.concatenate(toks), np.concatenate(globs), np.concatenate(wdls)
    off = np.concatenate([[0], np.cumsum(np.concatenate(cnt))]).astype(np.int64)
    # LEAK CHECK BY POSITION IDENTITY, not row index. The shards' `row` field indexes the
    # corpus they were GENERATED against (the 4000-game, 4.65M-row build); this script trains
    # on a different draw (600 games, 69k rows), so those indices are meaningless here -- using
    # them raised IndexError, which is how this was found. Position content is corpus
    # independent: hash each child board and drop any parent with a child we trained on.
    cand = np.arange(len(off) - 1)
    if train_keys is not None:
        # Both inputs are invariant across the run, so this is computed ONCE and cached on the
        # function. Recomputing an isin of 1.76M children against 3.63M keys at every eval was
        # pure waste (40x over a 20k-step run).
        cache = getattr(sf_yardstick, "_screen", None)
        if cache is None:
            key = np.ascontiguousarray(
                np.concatenate([toks, globs], 1).astype(np.uint8)).view(
                    np.dtype((np.void, toks.shape[1] + globs.shape[1]))).ravel()
            seen = np.isin(key, train_keys)
            keep = np.array([not seen[int(off[i]):int(off[i + 1])].any()
                             for i in cand], bool)
            sf_yardstick._screen = (keep, int((~keep).sum()))
            print(f"[csf] yardstick screen: dropped {int((~keep).sum()):,} of {len(cand):,} "
                  f"shard parents seen in TRAIN", flush=True)
        keep, n_drop = sf_yardstick._screen
        cand = cand[keep]
        # The shards carry no game identity and come from a DIFFERENT corpus draw, so they
        # cannot be split by game the way the corpus is. Screening removes boards we trained
        # on; this then splits the surviving pool deterministically in half so the monitoring
        # yardstick (want=1) and the single final read (want=2) are DISJOINT -- otherwise the
        # "test" move-ranking number is the same sample we watched all run and is not a fresh
        # read at all.
        half = np.random.default_rng(12345).permutation(len(cand))
        cut = len(half) // 2
        cand = cand[half[:cut]] if want == 1 else cand[half[cut:]]
        if len(cand) == 0:
            return None
    rng = np.random.default_rng(0)
    pick = rng.choice(cand, size=min(args.sf_parents, len(cand)), replace=False)
    net.eval()
    sps, tops, cells = [], [], []
    with torch.no_grad():
        for pi in pick:
            a, b = int(off[pi]), int(off[pi + 1])
            if b - a < 3:
                continue
            w = wdls[a:b].astype(np.float32)
            e_par = (w[:, 2] + 0.5 * w[:, 1]) / np.clip(w.sum(1), 1.0, None)
            g = globs[a:b]
            e_white = np.where(g[:, 0] == 0, e_par, 1.0 - e_par)
            zg_, zsq_, ids, _, ids_sq = net.encode(
                torch.from_numpy(toks[a:b].astype(np.int64)).to(dev),
                torch.from_numpy(g.astype(np.float32)).to(dev))
            z = net.full(zg_, zsq_)
            full = (ids if ids_sq is None else
                    torch.cat([ids, ids_sq], -1)).cpu().numpy()
            p3 = net.resolution(z)
            # move-ranking still needs a SCALAR: expected score under the
            # resolution distribution, using each pole's own value.
            e_ours = (p3.cpu().numpy() * pole_val[None, :]).sum(-1)
            cells.append(len(np.unique(full, axis=0)))
            if np.ptp(e_white) < 1e-6:
                continue
            s = spearmanr(e_ours, e_white)[0]
            if np.isfinite(s):
                sps.append(s)
            tops.append(float(np.argmax(e_ours) == np.argmax(e_white)))
    net.train()
    if not sps:
        return None
    return {"sp": float(np.mean(sps)), "top1": float(np.mean(tops)),
            "cells": float(np.mean(cells)), "n": len(sps),
            "dropped_seen_in_train": int(n_drop) if train_keys is not None else 0}


if __name__ == "__main__":
    main()
