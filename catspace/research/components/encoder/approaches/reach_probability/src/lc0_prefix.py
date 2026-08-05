"""Lc0Prefix -- the first few layers of the frozen lc0 transformer trunk, as a fixed feature base.

Kaveh 2026-08-05: "if the training is gonna be too expensive, maybe we should take the first few
layers of the lc0 trunk and freeze those, then train on that base" -- "the transformer one" -- and
then, decisively: "basically our hypothesis does not require training our independent encoder ...
our hypothesis is that strata could emerge if we combine IQE + JEPA in chess".

So the encoder is not the object of study here; the IQE + JEPA combination is. This supplies a
chess-literate representation for free and lets the heads do the learning.

WHY A PREFIX AND NOT THE WHOLE TRUNK. Measured on this machine (MPS, batch 256): the full 10-layer
t1-256x10 runs at 751 pos/s, a 3-layer prefix with early exit at 2508 pos/s -- 3.3x, and the
difference is pure waste otherwise, since the deep layers specialise toward lc0's policy/value
heads rather than toward position structure. The prefix is taken at `module.encoder{K}/ln2`, which
is a clean per-layer output boundary in the ONNX graph (10 such layers, encoder0..encoder9).

WHAT THIS COSTS US, STATED PLAINLY. The random-init null stops being a true zero. A from-scratch
ViT over tokens knows no chess, so any ratchet it shows was learned; a frozen lc0 prefix already
contains chess structure, so the honest null becomes "this same frozen prefix with randomly
initialised heads", and the claim weakens from "strata can be learned from data" to "the IQE+JEPA
objective adds strata beyond what the base already encodes". That is the SAME limitation that made
the previous trunk result inconclusive (paired ratchet 0.570 against a 0.555 null), and it is why
the encoder stays pluggable: `--encoder vit` runs the confound-free arm, `--encoder lc0` runs this
one, and interpret_reach.py scores the matching random-init null either way.

THE HISTORY LEAK IS BACK, AND IS HANDLED BY CONSTRUCTION. lc0's 112 planes carry 8 plies of
history, so for a close pair position `a` sits literally inside `b`'s own input tensor. Planes here
are built per position from a REPLAYED board, so the history they carry is that position's real
history -- which is legitimate (in a real game you do know your own history) but means a pair whose
plies are within 8 of each other can be scored right for reasons that are not reachability. Unlike
the token path, this arch therefore still needs gap-stratified reporting; `interpret_reach.py`
reads the verdict on the out-of-history band when the checkpoint says arch == "lc0".
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from catspace.io import paths

N_PLANES, N_SQ = 112, 64
RULE50_PLANE = 109          # the ONE non-binary plane: the halfmove clock, constant over squares
PACKED_BYTES = (N_PLANES - 1) * N_SQ // 8 + 1


def pack_planes(t) -> np.ndarray:
    """(B,112,8,8) float -> (B, PACKED_BYTES) uint8. 8.06x smaller than raw uint8.

    Verified against the data rather than assumed: of the 112 planes exactly ONE (109, the rule50
    halfmove counter) takes non-binary values, and it is constant across all 64 squares. So 111
    planes bit-pack losslessly and rule50 rides along as a single byte. 7168 B -> 889 B, which is
    what turns a 143 GB every-ply plane cache into an 18 GB one.
    """
    a = np.asarray(t, dtype=np.float32).reshape(-1, N_PLANES, N_SQ)
    keep = np.delete(a, RULE50_PLANE, axis=1) > 0.5
    bits = np.packbits(keep.reshape(len(a), -1), axis=1)
    r50 = np.clip(a[:, RULE50_PLANE, 0], 0, 255).astype(np.uint8)[:, None]
    return np.concatenate([bits, r50], axis=1)


def unpack_planes(p) -> np.ndarray:
    """(B, PACKED_BYTES) uint8 -> (B,112,8,8) float32, exactly inverting pack_planes."""
    p = np.asarray(p, dtype=np.uint8).reshape(-1, PACKED_BYTES)
    bits = np.unpackbits(p[:, :-1], axis=1)[:, :(N_PLANES - 1) * N_SQ]
    out = np.zeros((len(p), N_PLANES, N_SQ), np.float32)
    idx = [i for i in range(N_PLANES) if i != RULE50_PLANE]
    out[:, idx] = bits.reshape(len(p), N_PLANES - 1, N_SQ).astype(np.float32)
    out[:, RULE50_PLANE] = p[:, -1:].astype(np.float32)
    return out.reshape(-1, N_PLANES, 8, 8)


class _EarlyExit(Exception):
    """Raised by the layer hook to abandon the rest of the trunk. Not an error path -- it is the
    mechanism that makes the prefix 3.3x cheaper than running all ten layers and discarding seven."""


class Lc0Prefix(nn.Module):
    """FROZEN lc0 transformer prefix. (B,112,8,8) planes -> (B,64,C) square tokens.

    Never trained: every parameter has requires_grad False and forward runs under no_grad, so this
    is a fixed featuriser and the only learning happens in the adapter and heads above it. That is
    the point -- it is the base, not a component of the hypothesis.
    """

    def __init__(self, onnx=None, layer: int = 2, device="cpu"):
        super().__init__()
        from lczerolens import LczeroModel
        self.layer = int(layer)
        self.onnx = str(onnx or paths.engine("lc0/t1-256x10.onnx"))
        self.model = LczeroModel.from_onnx_path(self.onnx).float().to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        name = f"module.encoder{self.layer}/ln2"
        mods = dict(self.model.named_modules())
        if name not in mods:
            raise KeyError(f"{name} not in the graph; layers are encoder0..encoder9")
        self._grab = {}

        def _hook(_m, _i, o):
            self._grab["t"] = o
            raise _EarlyExit
        mods[name].register_forward_hook(_hook)
        self.out_ch = None

    @torch.no_grad()
    def forward(self, planes):
        """(B,112,8,8) -> (B,64,C). Gradient never flows through here, by construction."""
        try:
            self.model(planes)
        except _EarlyExit:
            pass
        t = self._grab["t"]
        C = t.shape[-1]
        self.out_ch = C
        return t.reshape(-1, N_SQ, C)


class Lc0Adapter(nn.Module):
    """The only TRAINABLE part of the lc0 base path: (B,64,C) tokens -> phi (B,d_model).

    Deliberately shaped like IQEHead's adapter (a 1x1 mix over channels, then a linear over the
    flattened board) so this path and the existing field head consume the trunk the same way and
    any difference between them is the objective rather than the input stage.
    """

    def __init__(self, in_ch: int = 256, d_model: int = 256, mix_ch: int = 32):
        super().__init__()
        self.mix = nn.Linear(in_ch, mix_ch)
        self.out = nn.Sequential(nn.Flatten(), nn.Linear(mix_ch * N_SQ, d_model),
                                 nn.LayerNorm(d_model))

    def forward(self, tokens):
        return self.out(torch.relu(self.mix(tokens)))
