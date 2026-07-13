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


def test_quasimetric_reduces_to_dot_product_when_off():
    fb = TorchFB(seed=0, quasimetric=False, **TINY)
    f = torch.randn(5, TINY["d"])
    b = torch.randn(3, TINY["d"])
    assert torch.equal(fb.score_matrix(f, b), f @ b.T)
    assert torch.equal(fb.score(f, b[0]), f @ b[0])


def test_quasimetric_init_matches_negative_euclidean_distance():
    """metric_scale inits to ones and W inits to zero, so score == -||f-g||
    exactly at construction time (before any training moves either)."""
    fb = TorchFB(seed=0, quasimetric=True, **TINY)
    f = torch.nn.functional.normalize(torch.randn(6, TINY["d"]), dim=1)
    b = torch.nn.functional.normalize(torch.randn(4, TINY["d"]), dim=1)
    expected = -torch.cdist(f, b, p=2)
    assert torch.allclose(fb.score_matrix(f, b), expected, atol=1e-4)
    assert torch.allclose(fb.score(f, b[0]), expected[:, 0], atol=1e-4)


def test_quasimetric_distance_matrix_is_a_real_metric():
    """Non-negativity, symmetry, and the triangle inequality on `d` alone
    (score_matrix's `r` residual is explicitly NOT required to satisfy
    these -- only the distance component is)."""
    torch.manual_seed(0)
    fb = TorchFB(seed=0, quasimetric=True, **TINY)
    # train briefly so metric_scale/W move off their initial values
    boards = _boards(24, seed=9)
    packed, meta = _packed_meta(boards)
    planes = torch.from_numpy(feature_planes(packed, meta))
    omega = torch.from_numpy(omega_ids(np.full(24, 1500), np.full(24, 1500), np.full(24, 60.0)))
    opt = torch.optim.AdamW(fb.parameters(), lr=1e-3)
    fb.train()
    for _ in range(20):
        loss, _ = fb.loss_fn(planes, omega, planes)
        opt.zero_grad(); loss.backward(); opt.step()
    fb.eval()

    with torch.no_grad():
        f = fb.embed_F(planes, omega)
        g = fb.embed_B(planes)
        x, y, z = f[:8], g[8:16], f[16:24]
        dxy = fb.distance_matrix(x, y).diagonal()
        dyz = fb.distance_matrix(y, z).diagonal()
        dxz = fb.distance_matrix(x, z).diagonal()

    assert torch.all(dxy >= -1e-4) and torch.all(dyz >= -1e-4) and torch.all(dxz >= -1e-4)
    # symmetry: the SAME formula applied to swapped args must agree (it only
    # depends on the codomain distance, not which encoder produced x vs y)
    with torch.no_grad():
        dyx = fb.distance_matrix(y, x).diagonal()
    assert torch.allclose(dxy, dyx, atol=1e-4)
    # triangle inequality: d(x,z) <= d(x,y) + d(y,z), for every triple
    assert torch.all(dxz <= dxy + dyz + 1e-4)


def test_quasimetric_ckpt_roundtrip_and_old_ckpt_unaffected(tmp_path):
    """A quasimetric checkpoint round-trips its extra params; a NON-
    quasimetric checkpoint is byte-for-byte unaffected by this feature
    existing at all (config-gated, no new params created when off)."""
    fb_q = TorchFB(seed=0, quasimetric=True, **TINY)
    save_ckpt(fb_q, tmp_path / "q.pt", step=3)
    fb_q2, payload = load_ckpt(tmp_path / "q.pt")
    assert payload["config"]["quasimetric"] is True
    assert torch.equal(fb_q.metric_scale, fb_q2.metric_scale)
    assert torch.equal(fb_q.W, fb_q2.W)

    fb_plain = TorchFB(seed=0, quasimetric=False, **TINY)
    assert not hasattr(fb_plain, "metric_scale") and not hasattr(fb_plain, "W")
    save_ckpt(fb_plain, tmp_path / "p.pt", step=3)
    fb_plain2, payload2 = load_ckpt(tmp_path / "p.pt")
    assert payload2["config"]["quasimetric"] is False
    for (ka, va), (kb, vb) in zip(fb_plain.state_dict().items(), fb_plain2.state_dict().items()):
        assert ka == kb and torch.equal(va, vb)


def test_np_score_matrix_matches_torch_and_dot():
    """np_score_matrix is the decompose.py score_pairs adapter: exactly the
    dot product when quasimetric=False (safe to pass unconditionally), and
    exactly score_matrix when quasimetric=True."""
    F = np.random.default_rng(0).normal(size=(5, TINY["d"])).astype(np.float32)
    B = np.random.default_rng(1).normal(size=(3, TINY["d"])).astype(np.float32)

    fb_plain = TorchFB(seed=0, quasimetric=False, **TINY)
    np.testing.assert_allclose(fb_plain.np_score_matrix(F, B), F @ B.T, atol=1e-5)

    fb_q = TorchFB(seed=0, quasimetric=True, **TINY)
    with torch.no_grad():
        expected = fb_q.score_matrix(torch.from_numpy(F), torch.from_numpy(B)).numpy()
    np.testing.assert_allclose(fb_q.np_score_matrix(F, B), expected, atol=1e-5)


def test_ply_gap_calibration_term():
    """ply_gap adds an MSE(d, ply_gap/scale) penalty in quasimetric mode
    (2026-07-12, calibrates absolute distance to real move-count, see
    JOURNAL.md) and is silently ignored otherwise."""
    boards = _boards(8, seed=1)
    packed, meta = _packed_meta(boards)
    planes = torch.from_numpy(feature_planes(packed, meta))
    omega = torch.from_numpy(omega_ids(np.full(8, 1500), np.full(8, 1500), np.full(8, 60.0)))
    ply_gap = torch.tensor([1., 5., 10., 20., 30., 40., 50., 60.])

    fb = TorchFB(seed=0, quasimetric=True, **TINY)
    loss_with, _ = fb.loss_fn(planes, omega, planes, ply_gap=ply_gap)
    loss_without, _ = fb.loss_fn(planes, omega, planes)
    assert float(loss_with) > float(loss_without)
    loss_with.backward()
    assert fb.metric_scale.grad is not None and fb.W.grad is not None

    fb2 = TorchFB(seed=0, quasimetric=False, **TINY)
    a, _ = fb2.loss_fn(planes, omega, planes, ply_gap=ply_gap)
    b, _ = fb2.loss_fn(planes, omega, planes)
    assert torch.equal(a, b), "non-quasimetric mode must ignore ply_gap entirely"


def test_asymmetry_margin_term():
    """asym hinge adds loss only in quasimetric mode with material_drop rows,
    produces gradients, and is a no-op at asym_weight=0 or with no drops."""
    boards = _boards(8, seed=2)
    packed, meta = _packed_meta(boards)
    planes = torch.from_numpy(feature_planes(packed, meta))
    omega = torch.from_numpy(omega_ids(np.full(8, 1500), np.full(8, 1500), np.full(8, 60.0)))
    drop = torch.tensor([True, False, True, False, True, False, True, False])

    fb = TorchFB(seed=0, quasimetric=True, **TINY)
    base, _ = fb.loss_fn(planes, omega, planes)
    with_asym, _ = fb.loss_fn(planes, omega, planes, material_drop=drop, asym_weight=0.1)
    # identity pairs: d_fwd == d_rev, so hinge = relu(margin) > 0 -- term active
    assert float(with_asym) > float(base)
    with_asym.backward()
    assert fb.metric_scale.grad is not None

    off1, _ = fb.loss_fn(planes, omega, planes, material_drop=drop, asym_weight=0.0)
    off2, _ = fb.loss_fn(planes, omega, planes, material_drop=torch.zeros(8, dtype=torch.bool),
                         asym_weight=0.1)
    assert torch.equal(off1, base) and torch.equal(off2, base)


def test_ckpt_roundtrip(tmp_path):
    fb = TorchFB(seed=0, **TINY)
    z = np.ones(TINY["d"], dtype=np.float32)
    save_ckpt(fb, tmp_path / "fb.pt", step=7, zgoals={"MATE_W": z})
    fb2, payload = load_ckpt(tmp_path / "fb.pt")
    for (ka, va), (kb, vb) in zip(fb.state_dict().items(), fb2.state_dict().items()):
        assert ka == kb and torch.equal(va, vb)
    assert payload["step"] == 7
    assert torch.equal(payload["zgoals"]["MATE_W"], torch.ones(TINY["d"]))
