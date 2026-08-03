"""
nn/eval_head.py — eval heads that read ONLY F(s), never the raw board (the
agreed design: the eval lives in the embedding space; if position quality
isn't linearly-ish decodable from F, that's a finding about F, not something
to paper over with board features). F is already omega-conditioned (Elo bins,
clock), so a probe on F is automatically per-Elo descriptive.

Two heads, one comparison:
  descriptive  3-way W/D/L softmax on GAME RESULTS -- what actually happens
               from here among these players
  normative    expected-score sigmoid on winprob(eval_cp) -- what should
               happen under best play (lichess server Stockfish / local SF)
Their divergence marks trap regions: positions where humans at this level
lose games the eval says are fine (or vice versa).

Frozen-probe by default: F is detached; joint fine-tuning is a research knob
(--joint in the driver), deliberately off until the probe results are read.

F-only is the headline; the driver's --repr {F,B,FB} trains the same probes
on B or F++B as CONTROLS (B ~ F on results => outcome info is static board
features; FB > F => F loses value info B keeps), and reports the zero-label
F@(zMATE_W - zMATE_B) readout as the no-training floor.
"""
from __future__ import annotations

import torch
from torch import nn

WDL_CLASSES = {1: 0, 0: 1, -1: 2}     # white-POV result -> class (W, D, L)


class EvalHead(nn.Module):
    def __init__(self, d_in: int, hidden: int = 128, n_out: int = 3, seed: int = 0):
        torch.manual_seed(seed)
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(), nn.Linear(hidden, n_out))
        self.n_out = n_out

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        return self.net(f)

    def expected_score(self, f: torch.Tensor) -> torch.Tensor:
        """White expected score in [0,1]: P(W) + 0.5 P(D) for the 3-way head,
        sigmoid for the scalar head -- the common scale both heads share."""
        out = self.net(f)
        if self.n_out == 1:
            return torch.sigmoid(out[:, 0])
        p = torch.softmax(out, dim=1)
        return p[:, 0] + 0.5 * p[:, 1]


def descriptive_loss(head: EvalHead, f: torch.Tensor, result: torch.Tensor) -> torch.Tensor:
    """result: white-POV {-1,0,1} int tensor."""
    target = torch.empty_like(result)
    for res, cls in WDL_CLASSES.items():
        target[result == res] = cls
    return nn.functional.cross_entropy(head(f), target)


def normative_loss(head: EvalHead, f: torch.Tensor, winprob: torch.Tensor) -> torch.Tensor:
    """winprob: [0,1] soft target from winprob_cp(eval_cp)."""
    return nn.functional.binary_cross_entropy_with_logits(head(f)[:, 0], winprob)


def save_heads(path, desc: EvalHead, norm: EvalHead, d_in: int, meta: dict | None = None):
    torch.save(dict(desc_state=desc.state_dict(), norm_state=norm.state_dict(),
                    d_in=d_in, hidden=desc.net[0].out_features, meta=meta or {}), path)


def load_heads(path, device: str = "cpu") -> tuple[EvalHead, EvalHead, dict]:
    p = torch.load(path, map_location=device, weights_only=False)
    desc = EvalHead(p["d_in"], p["hidden"], n_out=3)
    desc.load_state_dict(p["desc_state"])
    norm = EvalHead(p["d_in"], p["hidden"], n_out=1)
    norm.load_state_dict(p["norm_state"])
    return desc.to(device), norm.to(device), p["meta"]
