"""CompetenceMap (Method 2): kNN reliability field over embedding space."""
import numpy as np

from catspace.competence import CompetenceMap


def test_competence_map_recovers_a_planted_field():
    # reliability = 1 in one embedding region, 0 in another; the kNN field
    # must recover it on held-out points near each cluster.
    rng = np.random.default_rng(0)
    a = rng.normal(loc=[3, 0], size=(200, 2))          # region A -> reliability 1
    b = rng.normal(loc=[-3, 0], size=(200, 2))         # region B -> reliability 0
    E = np.vstack([a, b]).astype(np.float32)
    r = np.concatenate([np.ones(200), np.zeros(200)]).astype(np.float32)
    cmap = CompetenceMap(E, r, k=8)

    near_a = cmap.query(np.array([3.0, 0.2], dtype=np.float32))
    near_b = cmap.query(np.array([-3.0, -0.2], dtype=np.float32))
    assert near_a > 0.7 and near_b < 0.3

    # batched query returns one prediction per row
    preds = cmap.query(np.array([[3, 0], [-3, 0]], dtype=np.float32))
    assert preds.shape == (2,) and preds[0] > preds[1]


def test_competence_map_save_load_roundtrip(tmp_path):
    rng = np.random.default_rng(1)
    E = rng.normal(size=(50, 4)).astype(np.float32)
    r = rng.random(50).astype(np.float32)
    cmap = CompetenceMap(E, r, k=5)
    p = tmp_path / "cmap.npz"
    cmap.save(p)
    loaded = CompetenceMap.load(p)
    q = rng.normal(size=4).astype(np.float32)
    assert np.isclose(cmap.query(q), loaded.query(q)) and loaded.k == 5
