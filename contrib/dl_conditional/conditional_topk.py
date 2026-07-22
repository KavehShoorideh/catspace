"""Conditional (covariate-gated) sparse autoencoders for `dictionary_learning`.

A drop-in extension of dictionary_learning's TopK SAE that conditions the dictionary on a per-example
COVARIATE vector. Two classes, both subclassing the library's bases so they plug into `trainSAE`:

  * ``ConditionalAutoEncoderTopK`` -- a ``Dictionary`` whose ``encode(x, cond)`` gates each atom by a
    learned function (FiLM) of ``cond`` before the top-k, so an atom can be active only in part of the
    covariate space. ``cond=None`` -> identical to the base ``AutoEncoderTopK``.
  * ``ConditionalTopKTrainer`` -- the matching ``TopKTrainer``: activations arrive as ``[x | cond]``
    (last ``cond_dim`` columns are the covariates), and the L2 + auxk (dead-atom revival) objective is
    computed on the reconstruction of ``x`` only.

MOTIVATION -- what a standard SAE can't do, and this adds:
    A standard SAE discovers ONE global dictionary: every atom is a feature of the whole input
    distribution. But in many domains a concept's *meaning or relevance is context-dependent* -- it
    matters in one sub-population and is noise in another (in chess, the bishop-pair advantage is
    decisive in OPEN endgames and nearly irrelevant in closed openings; in language, a feature may
    read differently across registers or topics). A global SAE AVERAGES over contexts, so a
    context-specific concept is either washed out (never isolated as its own atom) or smeared across
    atoms, and -- crucially -- you cannot recover each concept's DOMAIN OF APPLICABILITY.

    Conditioning the dictionary on observable covariates fixes both: (1) context-specific concepts get
    their own atoms because the gate lets the SAE "spend" dictionary capacity per context; (2) each
    atom's gate, read against the covariates, IS its domain (which contexts it fires in). It is a
    strict generalization -- with ``cond_dim=0`` or a saturated gate it reduces to the standard TopK
    SAE -- so it is safe to add without changing existing behavior. Empirically (chess value field):
    king-safety atoms localize to castled-king contexts and a bishop atom localizes to open positions,
    concepts a global SAE reported only as low-correlation "novel" atoms.

Upstream note: ``TopKTrainer`` builds its dictionary as ``dict_class(activation_dim, dict_size, k)``
with no way to pass extra constructor args, so the trainer re-instantiates the AE + optimizer once.
Adding a ``dict_class_kwargs: dict = {}`` param to the base trainer would remove that re-init and make
this a pure subclass -- a small, backward-compatible upstream change worth proposing alongside.
"""
from __future__ import annotations

from collections import namedtuple

import torch as t
import torch.nn as nn

from dictionary_learning.trainers.top_k import (AutoEncoderTopK, TopKTrainer, geometric_median,
                                                remove_gradient_parallel_to_decoder_directions,
                                                set_decoder_norm_to_unit_norm)
from dictionary_learning.trainers.trainer import get_lr_schedule


class ConditionalAutoEncoderTopK(AutoEncoderTopK):
    """TopK SAE whose atoms are gated by a FiLM function of a covariate vector ``cond``.

    ``encode(x, cond)`` computes ``relu(W_enc (x - b_dec)) * sigmoid(gate(cond))`` and then keeps the
    top-k, so ``gate(cond) in (0,1)^dict`` decides which atoms are AVAILABLE in a given context.
    ``cond=None`` recovers the base behaviour exactly. ``decode`` is inherited unchanged.
    """

    def __init__(self, activation_dim: int, dict_size: int, k: int, cond_dim: int, gate_hidden: int = 64):
        super().__init__(activation_dim, dict_size, k)
        self.cond_dim = cond_dim
        self.gate = nn.Sequential(nn.Linear(cond_dim, gate_hidden), nn.ReLU(), nn.Linear(gate_hidden, dict_size))
        self.gate[-1].bias.data.fill_(2.0)                        # sigmoid(2)~=0.88: gates start ~open

    def encode(self, x: t.Tensor, cond: t.Tensor | None = None,
               return_topk: bool = False, use_threshold: bool = False):
        post_relu_BF = nn.functional.relu(self.encoder(x - self.b_dec))
        if cond is not None:
            post_relu_BF = post_relu_BF * t.sigmoid(self.gate(cond))   # <-- the only addition to the base path

        if use_threshold:
            acts_BF = post_relu_BF * (post_relu_BF > self.threshold)
            if return_topk:
                pt = post_relu_BF.topk(self.k, sorted=False, dim=-1)
                return acts_BF, pt.values, pt.indices, post_relu_BF
            return acts_BF

        pt = post_relu_BF.topk(self.k, sorted=False, dim=-1)
        acts_BF = t.zeros_like(post_relu_BF).scatter_(-1, pt.indices, pt.values)
        if return_topk:
            return acts_BF, pt.values, pt.indices, post_relu_BF
        return acts_BF


class ConditionalTopKTrainer(TopKTrainer):
    """TopKTrainer that trains a ``ConditionalAutoEncoderTopK``.

    Activations are expected as ``[x | cond]`` (concatenated; last ``cond_dim`` columns are the
    covariates), so the trainer stays compatible with the base ``trainSAE`` loop / ActivationBuffer --
    a buffer that appends the covariates is all that's needed. The objective is the base L2 + auxk,
    computed on the reconstruction of ``x`` only.
    """

    def __init__(self, steps: int, activation_dim: int, dict_size: int, k: int, cond_dim: int,
                 layer: int, lm_name: str, lr: float | None = None, auxk_alpha: float = 1 / 32,
                 warmup_steps: int = 1000, decay_start: int | None = None, threshold_beta: float = 0.999,
                 threshold_start_step: int = 1000, seed: int | None = None, device: str | None = None,
                 wandb_name: str = "ConditionalTopKSAE", submodule_name: str | None = None,
                 gate_hidden: int = 64):
        super().__init__(steps=steps, activation_dim=activation_dim, dict_size=dict_size, k=k, layer=layer,
                         lm_name=lm_name, lr=lr, auxk_alpha=auxk_alpha, warmup_steps=warmup_steps,
                         decay_start=decay_start, threshold_beta=threshold_beta,
                         threshold_start_step=threshold_start_step, seed=seed, device=device,
                         wandb_name=wandb_name, submodule_name=submodule_name)
        self.cond_dim = cond_dim
        # swap the standard dictionary for the conditional one (see upstream note in the module docstring)
        self.ae = ConditionalAutoEncoderTopK(activation_dim, dict_size, k, cond_dim, gate_hidden).to(self.device)
        self.optimizer = t.optim.Adam(self.ae.parameters(), lr=self.lr, betas=(0.9, 0.999))
        self.scheduler = t.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=get_lr_schedule(steps, warmup_steps, decay_start=decay_start))

    def loss(self, x: t.Tensor, step: int | None = None, logging: bool = False):
        d = self.ae.activation_dim
        acts, cond = x[..., :d], x[..., d:]                       # split [x | cond]
        f, top_acts_BK, top_idx_BK, post_relu_BF = self.ae.encode(
            acts, cond=cond, return_topk=True, use_threshold=False)
        if step is not None and step > self.threshold_start_step:
            self.update_threshold(top_acts_BK)
        x_hat = self.ae.decode(f)
        e = acts - x_hat                                          # reconstruct x only, not cond
        self.effective_l0 = top_acts_BK.size(1)
        did_fire = t.zeros_like(self.num_tokens_since_fired, dtype=t.bool)
        did_fire[top_idx_BK.flatten()] = True
        self.num_tokens_since_fired += acts.size(0)
        self.num_tokens_since_fired[did_fire] = 0
        l2_loss = e.pow(2).sum(dim=-1).mean()
        auxk_loss = self.get_auxiliary_loss(e.detach(), post_relu_BF) if self.auxk_alpha > 0 else 0.0
        loss = l2_loss + self.auxk_alpha * auxk_loss
        if not logging:
            return loss
        return namedtuple("LossLog", ["x", "x_hat", "f", "losses"])(
            acts, x_hat, f,
            {"l2_loss": l2_loss.item(), "auxk_loss": float(getattr(auxk_loss, "item", lambda: auxk_loss)()),
             "loss": loss.item()})

    def update(self, step: int, x: t.Tensor):
        """identical to TopKTrainer.update, except the step-0 b_dec init uses the ACTIVATION part of
        ``[x | cond]`` (the base would take the geometric median of the appended covariates too)."""
        x = x.to(self.device)
        if step == 0:
            acts = x[..., :self.ae.activation_dim]
            self.ae.b_dec.data = geometric_median(acts).to(self.ae.b_dec.dtype)
        loss = self.loss(x, step=step)
        loss.backward()
        self.ae.decoder.weight.grad = remove_gradient_parallel_to_decoder_directions(
            self.ae.decoder.weight, self.ae.decoder.weight.grad, self.ae.activation_dim, self.ae.dict_size)
        t.nn.utils.clip_grad_norm_(self.ae.parameters(), 1.0)
        self.optimizer.step(); self.optimizer.zero_grad(); self.scheduler.step()
        self.ae.decoder.weight.data = set_decoder_norm_to_unit_norm(
            self.ae.decoder.weight, self.ae.activation_dim, self.ae.dict_size)
        return loss.item()

    @property
    def config(self):
        cfg = dict(super().config)
        cfg.update({"trainer_class": "ConditionalTopKTrainer", "cond_dim": self.cond_dim})
        return cfg
