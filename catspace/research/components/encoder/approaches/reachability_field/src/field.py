"""catspace/field.py -- the M1 reachability field at PLAY TIME: a FROZEN community Leela distillate
trunk (adopted by fiat, locked decision 8) + our IQE head. Given chess boards it returns phi and the
quasimetric distances d(s->g) / d(s->conversion) the planner & search navigate on (locked decision 1:
geometry-first; NO WDL/committor navigation value).

Trunk = T1-256x10-distilled (distillate of lc0's strongest teachers; runs on this laptop). Transformer
trunk -> hooked features are (B*64, C) tokens -> reshaped square-major to (B, C, 8, 8) for the head.
The conversion anchor is trained vs tablebase DTZ (irreversible-progress toward the <=7p TB-won
boundary = the handoff; NOT DTM/mate -- mate is the tablebase's job post-handoff).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from catspace.research.components.encoder.approaches.reachability_field.src.iqe_head import IQEHead
from catspace.io import paths


class ReachabilityField:
    def __init__(self, onnx=paths.engine("lc0/t1-256x10.onnx"),
                 head=paths.experiment("field_iqe_t1_final.pt"), device="auto", tokens=True):
        from catspace.research.tools.training_infra.train.scaffold import resolve_device
        from lczerolens import LczeroModel
        self.dev = resolve_device(device) if device == "auto" else torch.device(device)
        self.tokens = tokens
        self.trunk = LczeroModel.from_onnx_path(onnx).float().to(self.dev).eval()
        names = [n for n, _ in self.trunk.named_modules()
                 if n and all(k not in n.lower() for k in ("policy", "value", "wdl", "output", "mlh"))]
        self.hook_name = names[-1]
        self._f = {}
        dict(self.trunk.named_modules())[self.hook_name].register_forward_hook(
            lambda mo, i, o: self._f.__setitem__("t", o))
        p = torch.load(head, map_location=self.dev, weights_only=False)
        cfg = p["cfg"]
        self.head = IQEHead(in_ch=cfg["in_ch"], d=cfg["d"], components=cfg["components"],
                            adapter_ch=cfg["adapter_ch"]).to(self.dev)
        self.head.load_state_dict(p["state_dict"]); self.head.eval()

    @torch.no_grad()
    def phi(self, lcboards):
        """LczeroBoards -> phi (B, d). Batched; the trunk is the dominant cost -> keep batches large."""
        x = torch.stack([b.to_input_tensor() for b in lcboards]).float().to(self.dev)
        return self._phi_x(x)

    @torch.no_grad()
    def phi_from_planes(self, planes):
        """(B,112,8,8) numpy planes -> phi. Lets callers cache to_input_tensor —
        the plane build is scalar-read heavy (profiled: dominant cost in the
        traced engine) and worth memoizing per fen."""
        x = torch.as_tensor(np.stack(planes)).float().to(self.dev)
        return self._phi_x(x)

    @torch.no_grad()
    def _phi_x(self, x):
        self.trunk(x); t = self._f["t"]
        if self.tokens:
            B = x.shape[0]; C = t.shape[-1]
            t = t.reshape(B, 64, C).permute(0, 2, 1).reshape(B, C, 8, 8)
        return self.head.phi(t)

    @torch.no_grad()
    def d(self, s_boards, g_boards):
        """directed reachability distance d(s -> g), (B,). s,g are equal-length board lists."""
        return self.head.d_pair_emb(self.phi(s_boards), self.phi(g_boards)).cpu().numpy()

    @torch.no_grad()
    def d_conversion(self, boards):
        """distance toward the forcing-conversion / TB-won boundary (DTZ anchor), (B,)."""
        return self.head.d_mate_emb(self.phi(boards)).cpu().numpy()
