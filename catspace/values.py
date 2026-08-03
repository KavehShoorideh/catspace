"""catspace/engine/values.py -- ValueModel implementations (leaf evaluation, white-POV,
[-1,1]). The value is always the GLOBAL objective; subgoals live in priors.py."""
from __future__ import annotations

from pathlib import Path

import chess
import numpy as np

VALUE_C = 8.0   # squash center: value = tanh((C - dist)/C); closer-to-mate reads higher


class ConstantValue:
    """No signal (pure search). The 0.12-mate-rate baseline on the ladder."""

    def values(self, boards: list) -> np.ndarray:
        return np.zeros(len(boards), dtype=float)


class TablebaseValue:
    """DIAGNOSTIC ONLY (DECISIONS sec 4): the oracle ceiling for experiments -- never wire
    into a shipped engine. |DTZ| as mate-distance proxy."""

    DIAGNOSTIC_ONLY = True

    def __init__(self, tb):
        self.tb = tb
        self._cache = {}

    def values(self, boards: list) -> np.ndarray:
        out = []
        for b in boards:
            k = b._transposition_key()
            if k not in self._cache:
                w, d = self.tb.wdl_dtz(b)
                self._cache[k] = 1.0 if w is None else float(
                    np.tanh((VALUE_C - abs(d if d is not None else 30)) / VALUE_C))
            out.append(self._cache[k])
        return np.array(out, dtype=float)


class DTMCNNValue:
    """The separate mate-distance head (train_dtm_cnn.py): board -> predicted DTM ->
    squashed value. The 'don't overload d' companion (DECISIONS sec 7)."""

    def __init__(self, ckpt: str | Path = "data/derived/sep/dtm_cnn.pt", device: str = "cpu"):
        import torch
        from catspace.research.components.encoder.approaches.jepa_tokenizer.src.features import feature_planes
        from catspace.research.components.encoder.approaches.jepa_tokenizer.src.fb import pick_device
        from catspace.research.components.planner.approaches.endgame_groundtruth.src.dtm import DTMNet
        self._torch, self._feature_planes = torch, feature_planes
        self.device = pick_device(device)
        st = torch.load(ckpt, map_location="cpu", weights_only=False)
        self.net = DTMNet(c=st["c"]).to(self.device)
        self.net.load_state_dict(st["state"]); self.net.eval()
        self.scale = st.get("scale", 20.0)

    def values(self, boards: list) -> np.ndarray:
        from catspace.research.tools.chess_specific.chessdata.encode import encode_meta, encode_packed
        pk = np.stack([encode_packed(b) for b in boards])
        mt = np.stack([encode_meta(b) for b in boards])
        with self._torch.no_grad():
            pred = self.net(self._torch.from_numpy(self._feature_planes(pk, mt))
                            .to(self.device)).cpu().numpy() * self.scale
        return np.tanh((VALUE_C - pred) / VALUE_C)


class FieldGoalDistanceValue:
    """Field distance to a goal bank (Region.bank or mate exemplars) as the value --
    the coarse navigator readout."""

    def __init__(self, field, bank: np.ndarray, scale: float = 6.0):
        self.field, self.bank, self.scale = field, bank, scale

    def values(self, boards: list) -> np.ndarray:
        d = self.field.d_boards_to_bank(boards, self.bank)
        return np.tanh((self.scale - d) / self.scale)
