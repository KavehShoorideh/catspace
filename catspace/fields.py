"""catspace/engine/fields.py -- FieldModel: ONE convention-aware wrapper around a trained
field checkpoint. Kills the bp()/eF()/eB() copy-paste that every experiment script carried,
and resolves the input-plane convention PER CHECKPOINT from its stored args (TRAINING_
STANDARDS rule 2/3: checkpoints self-describe; the old BOARD_ONLY zeroing is honored for
checkpoints trained under it, full planes otherwise)."""
from __future__ import annotations

from pathlib import Path

import chess
import numpy as np
import torch

from catspace.research.tools.chess_specific.chessdata.encode import encode_meta, encode_packed
from catspace.research.components.encoder.approaches.jepa_tokenizer.src.features import feature_planes, omega_ids
from catspace.research.components.encoder.approaches.jepa_tokenizer.src.fb import load_ckpt, pick_device

BOARD_ONLY = (18, 19)                  # halfmove clock + repetition planes (legacy zeroing)
_DEFAULT_OMEGA = (1800, 1800, 300.0)


class FieldModel:
    """Loaded field + its conventions. All embed/distance calls are chunked (never build
    the full n x n IQE matrix -- the silent-OOM lesson, JOURNAL 2026-07-22)."""

    def __init__(self, ckpt: str | Path, device: str = "cpu",
                 zero_board_only: bool | None = None, chunk: int = 512):
        self.path = Path(ckpt)
        self.device = pick_device(device)
        self.fb, self.payload = load_ckpt(self.path, self.device)
        self.fb.eval()
        self.chunk = chunk
        prov = self.payload.get("provenance") or {}
        stored = (prov.get("args") or {}) if isinstance(prov, dict) else {}
        if zero_board_only is None:
            # geometry-lineage ckpts (train_geometry_l1) trained with planes 18/19 zeroed;
            # their stored args include geometry-specific keys. Default: honor the lineage.
            zero_board_only = "repel_floor_all" in stored or "w_repel" in stored
        self.zero_board_only = bool(zero_board_only)
        om = omega_ids(np.array([_DEFAULT_OMEGA[0]]), np.array([_DEFAULT_OMEGA[1]]),
                       np.array([_DEFAULT_OMEGA[2]]))[0]
        self._om = om

    # ----------------------------------------------------------------- planes
    def _planes(self, pk: np.ndarray, mt: np.ndarray) -> torch.Tensor:
        pl = feature_planes(pk, mt)
        if self.zero_board_only:
            pl[:, list(BOARD_ONLY)] = 0.0
        return torch.from_numpy(pl).to(self.device)

    @staticmethod
    def pack(boards: list) -> tuple[np.ndarray, np.ndarray]:
        return (np.stack([encode_packed(b) for b in boards]),
                np.stack([encode_meta(b) for b in boards]))

    # ------------------------------------------------------------- embeddings
    def embed_F(self, pk: np.ndarray, mt: np.ndarray) -> np.ndarray:
        out = []
        for s in range(0, len(pk), self.chunk):
            with torch.no_grad():
                om = torch.from_numpy(np.tile(self._om, (len(pk[s:s + self.chunk]), 1))).to(self.device)
                out.append(self.fb.embed_F(self._planes(pk[s:s + self.chunk], mt[s:s + self.chunk]), om)
                           .cpu().numpy())
        return np.concatenate(out)

    def embed_B(self, pk: np.ndarray, mt: np.ndarray) -> np.ndarray:
        out = []
        for s in range(0, len(pk), self.chunk):
            with torch.no_grad():
                out.append(self.fb.embed_B(self._planes(pk[s:s + self.chunk], mt[s:s + self.chunk]))
                           .cpu().numpy())
        return np.concatenate(out)

    def embed_F_boards(self, boards: list) -> np.ndarray:
        return self.embed_F(*self.pack(boards))

    def embed_B_boards(self, boards: list) -> np.ndarray:
        return self.embed_B(*self.pack(boards))

    # -------------------------------------------------------------- distances
    def d_pairs(self, F: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Row-aligned d(F_i, B_i), chunked."""
        out = []
        for s in range(0, len(F), self.chunk):
            with torch.no_grad():
                out.append(self.fb.distance_matrix(
                    torch.from_numpy(F[s:s + self.chunk]).to(self.device),
                    torch.from_numpy(B[s:s + self.chunk]).to(self.device)).diagonal().cpu().numpy())
        return np.concatenate(out)

    def d_to_bank(self, F: np.ndarray, bank: np.ndarray) -> np.ndarray:
        """min over bank of d(F_i, bank_j) -- the goal-as-region readout (goal_bank lesson:
        nearest exemplar, never a centroid). Chunked on BOTH sides: IQE pairwise
        materializes (rows, cols, k) intermediates, and an unchunked 3k+ bank thrashes
        MPS memory under worker contention (the 1430s-dbank pathology, 2026-07-25)."""
        out = []
        for s in range(0, len(F), self.chunk):
            ft = torch.from_numpy(F[s:s + self.chunk]).to(self.device)
            row_min = None
            for bs in range(0, len(bank), 512):
                bt = torch.from_numpy(bank[bs:bs + 512]).to(self.device)
                with torch.no_grad():
                    m = self.fb.distance_matrix(ft, bt).min(1).values
                row_min = m if row_min is None else torch.minimum(row_min, m)
            out.append(row_min.cpu().numpy())
        return np.concatenate(out)

    def d_boards_to_bank(self, boards: list, bank: np.ndarray) -> np.ndarray:
        return self.d_to_bank(self.embed_F_boards(boards), bank)

    @property
    def zgoals(self) -> dict:
        return {k: (v.detach().float().numpy() if torch.is_tensor(v) else np.asarray(v, np.float32))
                for k, v in (self.payload.get("zgoals") or {}).items()}
