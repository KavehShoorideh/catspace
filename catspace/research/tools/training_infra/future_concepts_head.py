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

## Where the interpretability constraint lives

On the READOUT, not on the dynamics. WDL must be a linear function of concepts; how the
board evolves need not be. Kaveh 2026-08-26: "I'm happy to allow the future prediction
of the concept to be nonlinear." An MLP transition does not make any single concept less
readable -- it only predicts where the concepts go -- and since composability was
dropped, a nonlinear per-ply head still costs one forward pass per ply, so search speed
does not depend on it. `hidden=0` keeps the linear form as a comparison arm: if linear
is as good, that is the stronger result.

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

    def __init__(self, d_concept: int, d_move: int, ply: int, hidden: int = 1024):
        super().__init__()
        self.ply, self.d_concept = ply, d_concept
        # NONLINEAR BY DEFAULT (Kaveh 2026-08-26: "I'm happy to allow the future
        # prediction of the concept to be nonlinear").
        #
        # The interpretability constraint lives on the READOUT -- WDL must be a linear
        # function of concepts -- not on the DYNAMICS. How the board evolves is a
        # transition function; making it an MLP does not make a single concept less
        # readable, it only predicts where the concepts go. And since composability was
        # dropped, a nonlinear per-ply head still costs one forward pass per ply, so
        # nothing about search speed depends on it staying linear.
        #
        # hidden=0 keeps the linear form available as a comparison arm: if linear turns
        # out to be as good, that is a stronger result and worth knowing.
        if hidden:
            self.net = nn.Sequential(nn.Linear(d_concept + d_move, hidden), nn.GELU(),
                                     nn.Linear(hidden, d_concept))
        else:
            self.net = nn.Linear(d_concept + d_move, d_concept)

    def forward(self, z: torch.Tensor, mv: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z, mv], -1))


class FuturePredictor(nn.Module):
    """A set of per-ply heads plus the EMA target encoder they are trained against."""

    def __init__(self, encoder: nn.Module, d_concept: int, d_move: int,
                 plies=(1, 2), ema_decay: float = 0.999, hidden: int = 1024):
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
    """Accuracy, lift over copy, AND delta-space precision/recall.

    Kaveh 2026-08-26: "it's mostly okay that most concepts are copied... for concepts to be
    a useful long-term planning tool they MUST change slowly. The whole hypothesis was that
    the operator should be identity plus some deviation."

    Right -- so a high copy baseline is the DESIGN GOAL, not a failure, and whole-state
    accuracy is the wrong lens: it mixes bits that should not change (copying is correct
    there and deserves no credit) with the few that should (where the question lives).
    Measuring the DEVIATION directly:

        flip_recall     of the bits that genuinely changed, how many did we predict changed
        false_flip      of the bits that did NOT change, how many did we wrongly flip
        flip_precision  of the bits we predicted changed, how many really did

    A copier scores flip_recall 0.0 and false_flip 0.0 -- it is not punished for the bits it
    correctly leaves alone, and gets no credit either. This is the "delta-space accuracy"
    the gko1 journal pre-registered as the discriminating metric.
    """
    out = {}
    for p, head in pred.heads.items():
        if p not in mv or p not in target:
            continue
        z0 = (z > 0)
        tgt_b = (target[p] > 0)
        pred_b = (head(z, mv[p]) > 0)
        tgt = tgt_b.float()
        acc = (pred_b.float() == tgt).float().mean()
        cp = (z0.float() == tgt).float().mean()
        did_flip = (tgt_b != z0)                    # the deviation from identity
        said_flip = (pred_b != z0)
        nf = did_flip.sum().clamp(min=1)
        ns = said_flip.sum().clamp(min=1)
        out[f"ply{p}"] = {
            "acc": float(acc), "copy": float(cp),
            "lift": float((acc - cp) / (1 - cp).clamp(min=1e-6)),
            "bits_changed": float(1 - cp),
            "flip_recall": float((said_flip & did_flip).sum() / nf),
            "flip_precision": float((said_flip & did_flip).sum() / ns),
            "false_flip": float((said_flip & ~did_flip).sum() /
                                (~did_flip).sum().clamp(min=1)),
        }
    return out


def _test():
    torch.manual_seed(0)
    B, D, M = 64, 128, 16
    enc = nn.Linear(D, D)
    fp = FuturePredictor(enc, D, M, plies=(1, 2))          # nonlinear by default

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
    assert ev["flip_recall"] == 0.0, "a copier predicts no flips, so recall must be 0"
    assert ev["false_flip"] == 0.0, "a copier invents no flips, so false-flip must be 0"

    # 2. gradients reach both heads, and the loss is finite
    total, st = fp.loss(z, mv, tgt)
    total.backward()
    for p in ("1", "2"):
        net = fp.heads[p].net
        w = net[0].weight if isinstance(net, nn.Sequential) else net.weight
        g = w.grad
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

    print(f"  copier: raw accuracy {ev['acc']:.3f}, lift {ev['lift']:+.3f}, flip-recall "
          f"{ev['flip_recall']:.3f} -- the metric refuses to reward predicting stasis")
    print(f"  1-ply target flips {100*(1-evaluate(fp,z,mv,tgt)['ply1']['copy']):.0f}% of "
          f"bits, 2-ply {100*(1-evaluate(fp,z,mv,tgt)['ply2']['copy']):.0f}%")
    print(f"  gradient reaches both per-ply heads; adding a 3-ply head is one dict entry")
    print(f"  EMA moved {float(moved/jump):.4f} of the encoder's jump, and takes no "
          f"gradient")
    print("  FUTURE-CONCEPT HEAD TESTS PASS")


if __name__ == "__main__":
    _test()
