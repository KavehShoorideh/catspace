"""
Eval-head tests: probe learnability on synthetic separable embeddings, the
winprob transform, expected-score scale agreement, checkpoint round-trip,
and (if a stockfish binary is present) a 3-position live labeling smoke.
"""
import shutil

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from catspace.nn.eval_head import (EvalHead, descriptive_loss, load_heads,
                                      normative_loss, save_heads)
from catspace.nn.features import winprob_cp
from catspace.util import auc


def _separable(n=600, d=16, seed=0):
    """Embeddings where class is decodable from one direction: W/D/L along f[0]."""
    rng = np.random.default_rng(seed)
    f = rng.standard_normal((n, d)).astype(np.float32)
    result = np.select([f[:, 0] > 0.5, f[:, 0] < -0.5], [1, -1], default=0)
    return torch.from_numpy(f), torch.from_numpy(result.astype(np.int64))


def test_descriptive_probe_learns():
    f, result = _separable()
    head = EvalHead(f.shape[1], hidden=32, n_out=3, seed=0)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-2)
    for _ in range(300):
        loss = descriptive_loss(head, f, result)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        e = head.expected_score(f).numpy()
    r = result.numpy()
    assert auc(e[r == 1], e[r == -1]) > 0.95


def test_normative_probe_learns():
    rng = np.random.default_rng(1)
    f = rng.standard_normal((600, 16)).astype(np.float32)
    wp = 1 / (1 + np.exp(-3 * f[:, 0]))                     # winprob decodable from f[0]
    head = EvalHead(16, hidden=32, n_out=1, seed=0)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-2)
    ft, wt = torch.from_numpy(f), torch.from_numpy(wp.astype(np.float32))
    for _ in range(300):
        loss = normative_loss(head, ft, wt)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        e = head.expected_score(ft).numpy()
    from scipy.stats import spearmanr
    assert spearmanr(e, wp).statistic > 0.95


def test_winprob_properties():
    assert winprob_cp(np.array([0.0]))[0] == 0.5
    cps = np.array([-400.0, -100.0, 0.0, 100.0, 400.0])
    wp = winprob_cp(cps)
    assert np.all(np.diff(wp) > 0)                          # monotone
    assert abs(wp[1] + wp[3] - 1.0) < 1e-12                 # symmetric


def test_expected_score_scales_agree():
    """Both head shapes emit [0,1] on the same scale."""
    f = torch.zeros(4, 8)
    e3 = EvalHead(8, n_out=3, seed=0).expected_score(f)
    e1 = EvalHead(8, n_out=1, seed=0).expected_score(f)
    for e in (e3, e1):
        assert e.shape == (4,) and float(e.min()) >= 0.0 and float(e.max()) <= 1.0


def test_heads_roundtrip(tmp_path):
    desc = EvalHead(16, n_out=3, seed=0)
    norm = EvalHead(16, n_out=1, seed=1)
    save_heads(tmp_path / "h.pt", desc, norm, d_in=16, meta={"k": "v"})
    d2, n2, meta = load_heads(tmp_path / "h.pt")
    assert meta == {"k": "v"}
    f = torch.randn(3, 16)
    assert torch.equal(desc(f), d2(f)) and torch.equal(norm(f), n2(f))


@pytest.mark.skipif(shutil.which("stockfish") is None, reason="no stockfish binary")
def test_stockfish_labels_smoke():
    import chess
    import chess.engine
    engine = chess.engine.SimpleEngine.popen_uci("stockfish")
    try:
        engine.configure({"UCI_ShowWDL": True})
        info = engine.analyse(chess.Board(), chess.engine.Limit(nodes=10_000))
        cp = info["score"].white().score(mate_score=3200)
        assert cp is not None and abs(cp) < 200               # startpos is near-equal
        wdl = info.get("wdl")
        assert wdl is not None and abs(sum(wdl.white()) - 1000) <= 1
    finally:
        engine.quit()
