"""
train_lichess_fb.py unit tests (skipped without torch): batch_tensors'
holdout filtering and ply-gap wiring. (The 2026-07-11 --winner-pov-only
filter was removed 2026-07-12 -- losing trajectories carry the "bad
future" signal the ply-gap calibration needs; see JOURNAL.md.)
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

import chess

from catspace.data.encode import encode_meta, encode_packed
from catspace.data.sources import PairBatch
from experiments.train_lichess_fb import batch_tensors

N = 8


def _fake_batch(game_ids) -> PairBatch:
    boards = [chess.Board() for _ in range(N)]
    anchors = np.stack([encode_packed(b) for b in boards])
    goals = np.stack([encode_packed(b) for b in boards])
    board_meta = np.stack([encode_meta(b) for b in boards])
    meta = dict(
        game_id=np.asarray(game_ids, dtype=np.uint32),
        result=np.array([1, -1, 1, -1, 1, -1, 1, 0], dtype=np.int8),
        white_elo=np.full(N, 1500, dtype=np.uint16),
        black_elo=np.full(N, 1500, dtype=np.uint16),
        clock=np.full(N, 60.0, dtype=np.float32),
        board_meta=board_meta,
        board_meta_g=board_meta,
        ply=np.arange(N, dtype=np.int32),
        ply_g=np.arange(N, dtype=np.int32) + 7,
    )
    return PairBatch(anchors=anchors, goals=goals, meta=meta)


def test_batch_tensors_drops_only_holdout_rows():
    # game_ids 50 and 100 are holdout (game_id % 50 == 0); the rest train
    batch = _fake_batch([1, 2, 50, 3, 100, 4, 5, 6])
    tensors = batch_tensors(batch, "cpu")
    assert tensors is not None and len(tensors) == 4
    assert all(t.shape[0] == N - 2 for t in tensors)


def test_batch_tensors_ply_gap_is_goal_minus_anchor():
    batch = _fake_batch([1, 2, 3, 4, 5, 6, 7, 8])   # none held out
    *_, ply_gap = batch_tensors(batch, "cpu")
    assert ply_gap.shape == (N,)
    assert torch.equal(ply_gap, torch.full((N,), 7.0))


def test_batch_tensors_all_holdout_returns_none():
    batch = _fake_batch([50, 100, 150, 200, 250, 300, 350, 400])
    assert batch_tensors(batch, "cpu") is None
