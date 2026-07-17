"""
IQE axiom tests -- the CORRECTNESS GATE for the interval quasimetric. A
quasimetric-by-construction head is only worth anything if the axioms actually
hold, so these are hard asserts: identity, non-negativity, asymmetry
(expressible), and the triangle inequality over many random triples (both
directions of the asymmetric distance).
"""
import pytest

torch = pytest.importorskip("torch")

from catspace.nn.iqe import IQE, _union_length


def test_union_length_basic():
    # [0,2] u [1,3] u [5,6] = 3 + 1 = 4 ; a point contributes 0
    l = torch.tensor([[0.0, 1.0, 5.0, 2.0]])
    r = torch.tensor([[2.0, 3.0, 6.0, 2.0]])
    assert torch.allclose(_union_length(l, r), torch.tensor([4.0]))


def test_identity_is_exactly_zero():
    torch.manual_seed(0)
    q = IQE(d=64, components=8)
    x = torch.randn(32, 64)
    d = q(x, x)
    assert torch.allclose(d, torch.zeros(32), atol=1e-6)


def test_nonnegativity():
    torch.manual_seed(1)
    q = IQE(d=64, components=8)
    x, y = torch.randn(64, 64), torch.randn(64, 64)
    assert (q(x, y) >= -1e-6).all()


def test_asymmetry_is_expressible():
    torch.manual_seed(2)
    q = IQE(d=32, components=4)
    x, y = torch.randn(100, 32), torch.randn(100, 32)
    fwd, rev = q(x, y), q(y, x)
    # the two directions must genuinely differ somewhere (else it's a metric)
    assert (fwd - rev).abs().max() > 1e-3


def test_triangle_inequality_over_random_triples():
    torch.manual_seed(3)
    q = IQE(d=48, components=6)
    x, y, z = (torch.randn(2000, 48) for _ in range(3))
    dxz, dxy, dyz = q(x, z), q(x, y), q(y, z)
    slack = dxy + dyz - dxz
    # d(x->z) <= d(x->y) + d(y->z) for every triple, up to fp tolerance
    assert slack.min() > -1e-4, f"triangle violated by {float(slack.min()):.2e}"


def test_pairwise_matches_forward_on_diagonal():
    torch.manual_seed(4)
    q = IQE(d=64, components=8)
    x, y = torch.randn(16, 64), torch.randn(16, 64)
    diag = q.pairwise(x, y).diagonal()
    assert torch.allclose(diag, q(x, y), atol=1e-5)


def test_triangle_after_a_few_optimizer_steps():
    # the axioms must survive TRAINING, not just hold at init
    torch.manual_seed(5)
    q = IQE(d=32, components=4)
    opt = torch.optim.Adam(q.parameters(), lr=1e-2)
    x, y = torch.randn(256, 32, requires_grad=True), torch.randn(256, 32)
    for _ in range(20):
        loss = q(x, y).mean()          # arbitrary objective to move alpha
        opt.zero_grad(); loss.backward(); opt.step()
    a, b, c = (torch.randn(1000, 32) for _ in range(3))
    slack = q(a, b) + q(b, c) - q(a, c)
    assert slack.min() > -1e-4
