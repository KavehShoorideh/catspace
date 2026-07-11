"""
nn-stack tests (skipped wholesale when torch isn't installed): feature
planes, encoder determinism, TorchFB InfoNCE overfit, omega conditioning,
checkpoint round-trip. All on CPU -- MPS is for training runs, not tests.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

import chess

from catspace.data.encode import encode_meta, encode_packed
from catspace.nn.features import (N_PLANES, clock_bucket, elo_bin, feature_planes,
                                     omega_ids, winprob_cp)
from catspace.nn.fb import TorchFB, load_ckpt, save_ckpt

TINY = dict(d=16, channels=16, blocks=2, enc_out=64, dh=64, omega_dim=4)


def _boards(n=8, seed=0):
    rng = np.random.default_rng(seed)
    boards = []
    for _ in range(n):
        b = chess.Board()
        for _ in range(int(rng.integers(0, 30))):
            legal = list(b.legal_moves)
            if not legal:
                break
            b.push(legal[int(rng.integers(0, len(legal)))])
        boards.append(b)
    return boards


def _packed_meta(boards):
    packed = np.stack([encode_packed(b) for b in boards])
    meta = np.stack([encode_meta(b) for b in boards])
    return packed, meta


def test_feature_planes():
    boards = _boards(8)
    packed, meta = _packed_meta(boards)
    x = feature_planes(packed, meta)
    assert x.shape == (8, N_PLANES, 8, 8) and x.dtype == np.float32

    for i, b in enumerate(boards):
        assert x[i, :12].sum() == len(b.piece_map())             # one bit per piece
        assert np.all(x[i, 12] == (0.0 if b.turn == chess.WHITE else 1.0))
        if b.ep_square is not None:
            r, c = b.ep_square // 8, b.ep_square % 8
            assert x[i, 17, r, c] == 1.0 and x[i, 17].sum() == 1.0
        else:
            assert x[i, 17].sum() == 0.0


def test_omega_ids_and_winprob():
    assert elo_bin(np.array([799, 800, 1999, 2799, 3500, 0])).tolist() == [0, 0, 5, 9, 9, 10]
    assert clock_bucket(np.array([1.0, 20.0, 700.0, np.nan])).tolist() == [0, 1, 6, 7]
    om = omega_ids(np.array([1500]), np.array([2100]), np.array([45.0]))
    assert om.shape == (1, 3) and om.tolist() == [[3, 6, 2]]

    wp = winprob_cp(np.array([-3200.0, 0.0, 3200.0]))
    assert wp[1] == 0.5 and wp[0] < 0.001 and wp[2] > 0.999
    assert np.isnan(winprob_cp(np.array([np.nan])))[0]


def test_torchfb_determinism_and_distinct_encoders():
    a = TorchFB(seed=0, **TINY)
    b = TorchFB(seed=0, **TINY)
    for (ka, va), (kb, vb) in zip(a.state_dict().items(), b.state_dict().items()):
        assert ka == kb and torch.equal(va, vb)
    # F and B encoders must NOT share an init (one seed, sequential draws)
    assert not torch.equal(a.encF.stem[0].weight, a.encB.stem[0].weight)
    c = TorchFB(seed=1, **TINY)
    assert not torch.equal(a.encF.stem[0].weight, c.encF.stem[0].weight)


def test_torchfb_overfits_tiny():
    torch.manual_seed(0)
    boards = _boards(32, seed=3)
    packed, meta = _packed_meta(boards)
    planes = torch.from_numpy(feature_planes(packed, meta))
    omega = torch.from_numpy(omega_ids(np.full(32, 1500), np.full(32, 1500), np.full(32, 60.0)))

    fb = TorchFB(seed=0, **TINY)
    opt = torch.optim.AdamW(fb.parameters(), lr=1e-3)
    fb.train()
    loss0 = None
    for _ in range(150):
        loss, top1 = fb.loss_fn(planes, omega, planes)   # identity pairs
        opt.zero_grad(); loss.backward(); opt.step()
        loss0 = loss0 if loss0 is not None else float(loss)
    fb.eval()
    with torch.no_grad():
        loss, top1 = fb.loss_fn(planes, omega, planes)
    assert float(loss) < 0.5 * loss0
    assert float(top1) > 0.5


def test_omega_changes_F_only():
    boards = _boards(4, seed=5)
    packed, meta = _packed_meta(boards)
    planes = torch.from_numpy(feature_planes(packed, meta))
    fb = TorchFB(seed=0, **TINY)
    fb.eval()
    om1 = torch.from_numpy(omega_ids(np.full(4, 1200), np.full(4, 1200), np.full(4, 60.0)))
    om2 = torch.from_numpy(omega_ids(np.full(4, 2400), np.full(4, 2400), np.full(4, 60.0)))
    with torch.no_grad():
        assert not torch.allclose(fb.embed_F(planes, om1), fb.embed_F(planes, om2))
        assert torch.equal(fb.embed_B(planes), fb.embed_B(planes))


def test_ckpt_roundtrip(tmp_path):
    fb = TorchFB(seed=0, **TINY)
    z = np.ones(TINY["d"], dtype=np.float32)
    save_ckpt(fb, tmp_path / "fb.pt", step=7, zgoals={"MATE_W": z})
    fb2, payload = load_ckpt(tmp_path / "fb.pt")
    for (ka, va), (kb, vb) in zip(fb.state_dict().items(), fb2.state_dict().items()):
        assert ka == kb and torch.equal(va, vb)
    assert payload["step"] == 7
    assert torch.equal(payload["zgoals"]["MATE_W"], torch.ones(TINY["d"]))
