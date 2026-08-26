"""Gradual magnitude pruning of the CONCEPT readout, done the way the literature does it.

Kaveh 2026-08-26: "I don't want to ever use a broken foundation to make a judgment about
a method... everything has to rest on correct implementations."

The previous attempt was broken in three ways, each observed in a real run:

  1. NO PERSISTENT MASK. Weights were soft-thresholded to zero and then handed straight
     back to Adam, which resurrected them from stored momentum. Once thresholding
     stopped, the readout went from 328 live concepts back to 2528/2528.
  2. A THRESHOLD INSTEAD OF A SCHEDULE. lambda was tuned hoping for an equilibrium.
     There is none: at lambda=10 every weight died within 60 steps, at 2 it decayed
     steadily, at 0.5 it decayed slower. "Stable" was just "not yet dead".
  3. UNSTRUCTURED. It pruned individual weights while every claim made from it was
     about CONCEPTS. "483 concepts" is only meaningful if whole columns die.

This implements Automated Gradual Pruning (Zhu & Gupta, 2017; the GMP* results of
Kurtic & Alistarh 2022 show a well-tuned version of exactly this beats most fancier
methods), adapted to structured column pruning:

  phase 1  WARM-UP     no pruning; let the concepts form before judging their magnitude
  phase 2  CUBIC RAMP  keep-count falls from all -> target on a cubic schedule, with the
                       mask RECOMPUTED at each interval, so a column wrongly pruned
                       early can return if the remaining weights rearrange
  phase 3  FINE-TUNE   mask frozen, sparse model trains to convergence

The mask is re-applied after every optimizer step, so nothing the optimizer does can
bring a pruned concept back within a phase.
"""
from __future__ import annotations

import torch


class ConceptPruner:
    """Structured gradual pruning over the CONCEPT axis of a readout weight.

    `weight` is (out, n_concepts); a concept is pruned by zeroing its whole column, so
    "N concepts survive" is literally true rather than a thresholding artefact.
    """

    def __init__(self, weight: torch.Tensor, target_keep: int, total_steps: int,
                 warmup_frac: float = 0.2, finetune_frac: float = 0.2,
                 interval: int = 100):
        assert weight.dim() == 2, "expected (out, n_concepts)"
        self.w = weight
        self.n = weight.shape[1]
        self.target = max(1, min(target_keep, self.n))
        self.total = max(total_steps, 1)
        self.t0 = int(warmup_frac * self.total)                 # pruning begins
        self.t1 = int((1.0 - finetune_frac) * self.total)       # mask freezes
        self.interval = max(1, interval)
        self.mask = torch.ones(self.n, dtype=torch.bool, device=weight.device)
        self.frozen = False

    def keep_at(self, step: int) -> int:
        """Cubic schedule: slow at first so the model can settle, then accelerating.

        Zhu & Gupta define sparsity s(t) = s_f + (s_i - s_f)(1 - (t-t0)/(t1-t0))^3.
        Expressed as a KEEP COUNT, which is the quantity we actually report.
        """
        if step <= self.t0:
            return self.n
        if step >= self.t1 or self.frozen:
            return self.target
        frac = (step - self.t0) / max(self.t1 - self.t0, 1)
        prog = 1.0 - (1.0 - frac) ** 3
        return int(round(self.n - (self.n - self.target) * prog))

    @torch.no_grad()
    def step(self, step: int) -> int:
        """Recompute the mask when due, then ALWAYS re-apply it. Returns live count."""
        if step >= self.t1:
            self.frozen = True                    # final phase: sparse fine-tuning
        due = (step > self.t0 and not self.frozen and step % self.interval == 0)
        if due:
            keep = self.keep_at(step)
            if keep < int(self.mask.sum()):
                # rank by column norm of the LIVE weight; recomputed each time, so the
                # ordering can change as survivors rearrange (the recovery AGP allows)
                score = self.w.abs().sum(0)
                idx = torch.argsort(score, descending=True)[:keep]
                self.mask = torch.zeros_like(self.mask)
                self.mask[idx] = True
        # re-applied on EVERY step: this is what the previous version lacked, and why
        # Adam kept resurrecting pruned weights from stored momentum
        self.w.mul_(self.mask.unsqueeze(0).to(self.w.dtype))
        return int(self.mask.sum())

    def state_dict(self) -> dict:
        return {"mask": self.mask, "frozen": self.frozen}

    def load_state_dict(self, sd: dict) -> None:
        """A resumed run must not silently restart the schedule with a full mask."""
        if sd:
            self.mask = sd["mask"].to(self.mask.device)
            self.frozen = bool(sd.get("frozen", False))


def _test():
    import torch.nn as nn
    torch.manual_seed(0)
    D, TOTAL, TARGET = 200, 1000, 40
    lin = nn.Linear(D, 3)
    opt = torch.optim.Adam(lin.parameters(), lr=1e-2)
    pr = ConceptPruner(lin.weight, TARGET, TOTAL, interval=25)

    # 1. schedule: full during warm-up, monotone down, target reached and held
    ks = [pr.keep_at(t) for t in range(TOTAL + 1)]
    assert ks[0] == D and ks[pr.t0] == D, "pruning began during warm-up"
    assert all(a >= b for a, b in zip(ks, ks[1:])), "keep-count is not monotone"
    assert ks[-1] == TARGET, f"never reached target: {ks[-1]}"

    live = []
    for t in range(1, TOTAL + 1):
        x, y = torch.randn(16, D), torch.randn(16, 3)
        opt.zero_grad()
        nn.functional.mse_loss(lin(x), y).backward()
        opt.step()                                  # Adam WILL touch pruned weights
        live.append(pr.step(t))

    # 2. the property the old code failed: the optimizer cannot resurrect a concept
    assert live[-1] == TARGET, f"ended with {live[-1]} live, wanted {TARGET}"
    dead = ~pr.mask
    assert float(lin.weight[:, dead].abs().max()) == 0.0, \
        "pruned concepts are non-zero -- the mask is not being re-applied"

    # 3. STRUCTURED: a surviving concept keeps a full column, a dead one loses all of it
    colnz = (lin.weight.abs() > 0).sum(0)
    assert set(colnz[pr.mask].tolist()) == {3}, "surviving columns are partially zeroed"
    assert set(colnz[dead].tolist()) == {0}, "dead columns are not fully zeroed"

    # 4. NEGATIVE CONTROL: without re-applying the mask, Adam brings them back
    lin2 = nn.Linear(D, 3)
    opt2 = torch.optim.Adam(lin2.parameters(), lr=1e-2)
    with torch.no_grad():
        lin2.weight[:, TARGET:] = 0.0                # prune once, then walk away
    for _ in range(50):
        x, y = torch.randn(16, D), torch.randn(16, 3)
        opt2.zero_grad(); nn.functional.mse_loss(lin2(x), y).backward(); opt2.step()
    back = int((lin2.weight[:, TARGET:].abs() > 0).any(0).sum())
    assert back > 0, "control failed: the test cannot tell a mask from no mask"

    # 5. resume: mask survives a round trip
    pr2 = ConceptPruner(nn.Linear(D, 3).weight, TARGET, TOTAL)
    pr2.load_state_dict(pr.state_dict())
    assert bool((pr2.mask == pr.mask).all()) and pr2.frozen == pr.frozen

    print(f"  schedule {D} -> {TARGET} concepts, cubic, monotone, warm-up respected")
    print(f"  after {TOTAL} Adam steps: {live[-1]} live, pruned columns exactly 0.0")
    print(f"  negative control (no re-apply): {back} of {D-TARGET} concepts resurrected "
          f"by the optimizer -- which is the bug this replaces")
    print(f"  mask survives save/load, so a resumed run does not restart the schedule")
    print("  PRUNING TESTS PASS")


if __name__ == "__main__":
    _test()
