"""Predicting the concept space 1 and 2 plies ahead, with an EMA target.

Kaveh 2026-08-26: "we probably want a one and two ply future concept space predictor in
there with EMA, which will give us the foundation for growing it later." And, on the
operator it replaces: "we don't need the composability of Koopman. We just want to
predict what the eval is six plies down given these moves."

So these are DIRECT per-ply heads, not one operator applied repeatedly. Adding a 3-ply
head is adding a dict entry; nothing has to compose, stay linear, or have a spectrum.

## The measurement trap this module exists to avoid

A move changes little. Most concept bits are identical before and after, so a predictor
that simply COPIES its input scores very high raw bit-accuracy and looks excellent. Any
number reported here is therefore quoted against the copy baseline:

    lift = (model accuracy - copy accuracy) / (1 - copy accuracy)

which is 0.0 for a copier no matter how high its raw accuracy, and 1.0 for a perfect
predictor. `evaluate()` returns both, and the test includes a copier as a negative
control precisely so the metric is known to distinguish them.

## Why EMA targets

The target concepts come from a slowly-updated copy of the encoder, with stop-gradient.
Without that, "make the future predictable from the present" has a trivial solution:
make the concepts constant. The EMA target is the standard remedy (BYOL and its
descendants), and this codebase already relies on it for the same reason elsewhere.
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


class FutureConceptHead(nn.Module):
    """(concepts, move) -> the concepts that will hold `ply` plies later.

    Concepts are ±1 bits, so this is per-bit binary classification and the natural
    output is a logit per bit.
    """

    def __init__(self, d_concept: int, d_move: int, ply: int, hidden: int = 0):
        super().__init__()
        self.ply, self.d_concept = ply, d_concept
        if hidden:
            self.net = nn.Sequential(nn.Linear(d_concept + d_move, hidden), nn.GELU(),
                                     nn.Linear(hidden, d_concept))
        else:
            # linear by default: the same discipline as the readout -- if a linear map
            # cannot predict the next concepts, that is worth knowing before adding
            # capacity to hide it
            self.net = nn.Linear(d_concept + d_move, d_concept)

    def forward(self, z: torch.Tensor, mv: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z, mv], -1))


class FuturePredictor(nn.Module):
    """A set of per-ply heads plus the EMA target encoder they are trained against."""

    def __init__(self, encoder: nn.Module, d_concept: int, d_move: int,
                 plies=(1, 2), ema_decay: float = 0.999, hidden: int = 0):
        super().__init__()
        self.heads = nn.ModuleDict(
            {str(p): FutureConceptHead(d_concept, d_move, p, hidden) for p in plies})
        self.ema = copy.deepcopy(encoder).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay = ema_decay
        self.d_concept = d_concept

    @torch.no_grad()
    def update_ema(self, encoder: nn.Module) -> None:
        """One EMA step. Call AFTER the optimizer step, every step.

        Resuming previously hard-copied this target and destroyed its lag silently, so
        the decay is applied as a law rather than re-initialised from the online weights.
        """
        for t, s in zip(self.ema.parameters(), encoder.parameters()):
            t.mul_(self.decay).add_(s.detach(), alpha=1 - self.decay)
        for t, s in zip(self.ema.buffers(), encoder.buffers()):
            t.copy_(s)

    def loss(self, z: torch.Tensor, mv: dict, target: dict) -> tuple:
        """z: (B,D) current concepts. mv[p], target[p]: move code and ±1 target per ply.

        Targets must already come from the EMA encoder under no_grad.
        """
        total, stats = z.new_zeros(()), {}
        for p, head in self.heads.items():
            if p not in mv or p not in target:
                continue
            logit = head(z, mv[p])
            tgt = (target[p] > 0).float()                  # ±1 -> {0,1}
            l = F.binary_cross_entropy_with_logits(logit, tgt)
            total = total + l
            with torch.no_grad():
                pred = (logit > 0).float()
                acc = (pred == tgt).float().mean()
                copy_acc = ((z > 0).float() == tgt).float().mean()
                stats[f"ply{p}_bce"] = float(l)
                stats[f"ply{p}_acc"] = float(acc)
                stats[f"ply{p}_copy"] = float(copy_acc)
                stats[f"ply{p}_lift"] = float((acc - copy_acc) /
                                              (1 - copy_acc).clamp(min=1e-6))
        return total, stats


@torch.no_grad()
def evaluate(pred: FuturePredictor, z, mv: dict, target: dict) -> dict:
    """Accuracy AND lift over copy-the-input, per ply. Never quote accuracy alone."""
    out = {}
    for p, head in pred.heads.items():
        if p not in mv or p not in target:
            continue
        tgt = (target[p] > 0).float()
        acc = ((head(z, mv[p]) > 0).float() == tgt).float().mean()
        cp = (((z > 0).float()) == tgt).float().mean()
        out[f"ply{p}"] = {"acc": float(acc), "copy": float(cp),
                          "lift": float((acc - cp) / (1 - cp).clamp(min=1e-6)),
                          "bits_changed": float(1 - cp)}
    return out


def _test():
    torch.manual_seed(0)
    B, D, M = 64, 128, 16
    enc = nn.Linear(D, D)
    fp = FuturePredictor(enc, D, M, plies=(1, 2), hidden=0)

    z = torch.sign(torch.randn(B, D))
    mv = {"1": torch.randn(B, M), "2": torch.randn(B, M)}
    # a realistic future: most bits identical, a minority flipped by the move
    tgt = {}
    for p, frac in (("1", 0.08), ("2", 0.15)):
        flip = (torch.rand(B, D) < frac).float() * -2 + 1
        tgt[p] = z * flip

    # 1. the metric must not reward a copier
    class Copier(nn.Module):
        def forward(self, z_, mv_): return z_ * 10.0
    fp_copy = FuturePredictor(enc, D, M, plies=(1,))
    fp_copy.heads["1"] = Copier()
    ev = evaluate(fp_copy, z, {"1": mv["1"]}, {"1": tgt["1"]})["ply1"]
    assert ev["acc"] > 0.85, f"copier should score high RAW accuracy, got {ev['acc']}"
    assert abs(ev["lift"]) < 1e-5, f"copier must score ZERO lift, got {ev['lift']}"

    # 2. gradients reach both heads, and the loss is finite
    total, st = fp.loss(z, mv, tgt)
    total.backward()
    for p in ("1", "2"):
        g = fp.heads[p].net.weight.grad
        assert g is not None and g.abs().sum() > 0, f"no gradient into the {p}-ply head"
    assert all(torch.isfinite(torch.tensor(v)) for v in st.values())

    # 3. the EMA target lags, and is never touched by gradient
    before = fp.ema.weight.detach().clone()
    with torch.no_grad():
        enc.weight.add_(torch.randn_like(enc.weight))
    fp.update_ema(enc)
    moved = (fp.ema.weight - before).abs().mean()
    jump = (enc.weight - before).abs().mean()
    assert 0 < moved < 0.05 * jump, \
        f"EMA is not lagging: moved {moved:.4f} of the encoder's {jump:.4f}"
    assert not fp.ema.weight.requires_grad, "EMA target is receiving gradient"

    print(f"  copier: raw accuracy {ev['acc']:.3f} but lift {ev['lift']:+.3f} -- the "
          f"metric refuses to reward predicting stasis")
    print(f"  1-ply target flips {100*(1-evaluate(fp,z,mv,tgt)['ply1']['copy']):.0f}% of "
          f"bits, 2-ply {100*(1-evaluate(fp,z,mv,tgt)['ply2']['copy']):.0f}%")
    print(f"  gradient reaches both per-ply heads; adding a 3-ply head is one dict entry")
    print(f"  EMA moved {float(moved/jump):.4f} of the encoder's jump, and takes no "
          f"gradient")
    print("  FUTURE-CONCEPT HEAD TESTS PASS")


if __name__ == "__main__":
    _test()
