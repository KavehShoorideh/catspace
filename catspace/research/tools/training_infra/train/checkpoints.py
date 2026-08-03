"""
train/checkpoints.py — bounded, resumable trainer checkpoints.

The original exp_krkn2.py pickled the ENTIRE accumulated raw transition list
every round -- unbounded growth across resumes. Here the checkpoint holds the
sufficient statistic (the accumulated transition-COUNT matrix, bounded by the
number of distinct (state, outcome) pairs actually visited) plus the current
field, via np.savez (no pickle of arbitrary objects).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.sparse as sp


@dataclass
class TrainerState:
    round: int
    counts: sp.csr_matrix
    scores: np.ndarray
    F: np.ndarray
    B: np.ndarray


def save_ckpt(state: TrainerState, path: Path) -> None:
    path = Path(path)
    counts = state.counts.tocsr()
    np.savez(
        path,
        round=state.round,
        scores=state.scores, F=state.F, B=state.B,
        counts_data=counts.data, counts_indices=counts.indices,
        counts_indptr=counts.indptr, counts_shape=np.array(counts.shape),
    )


def load_ckpt(path: Path) -> TrainerState:
    path = Path(path)
    npz_path = path if path.suffix == ".npz" else path.with_suffix(path.suffix + ".npz")
    z = np.load(npz_path)
    counts = sp.csr_matrix(
        (z["counts_data"], z["counts_indices"], z["counts_indptr"]),
        shape=tuple(z["counts_shape"]),
    )
    return TrainerState(round=int(z["round"]), counts=counts,
                         scores=z["scores"], F=z["F"], B=z["B"])


def ckpt_exists(path: Path) -> bool:
    path = Path(path)
    npz_path = path if path.suffix == ".npz" else path.with_suffix(path.suffix + ".npz")
    return npz_path.exists()
