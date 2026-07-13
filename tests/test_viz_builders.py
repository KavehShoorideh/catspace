"""Fast, model-free tests for the viz builder payload helpers (VIZ_PLAN.md).
No shards/checkpoints required: synthetic npz/pgn fixtures + tiny embeddings."""
import json

import chess
import numpy as np
import pytest

from catspace.data.encode import encode_meta, encode_packed
from catspace.viz.build_html import build_html
from catspace.viz.payload import json_default
from catspace.viz.realboard import (board_svg, embed_positions, fit_projection,
                                    games_from_pgn, infer_san, load_games_from_shard)


def test_infer_san_recovers_move():
    board = chess.Board()
    move = chess.Move.from_uci("e2e4")
    child = board.copy(stack=False)
    child.push(move)
    san, mv = infer_san(board, encode_packed(child), encode_meta(child))
    assert san == "e4"
    assert mv == move


def test_infer_san_no_match_returns_none():
    board = chess.Board()
    bogus_packed = np.zeros(12, dtype=np.uint64)
    bogus_meta = np.zeros(8, dtype=np.uint8)
    san, mv = infer_san(board, bogus_packed, bogus_meta)
    assert san is None and mv is None


def test_board_svg_returns_svg_string():
    svg = board_svg(chess.Board(), size=200)
    assert svg.strip().startswith("<svg")


def test_fit_projection_pca_shapes():
    rng = np.random.default_rng(0)
    F = rng.normal(size=(50, 8)).astype(np.float32)
    F /= np.linalg.norm(F, axis=1, keepdims=True)
    fp = fit_projection(F, kind="pca")
    xy = fp.transform(F[:5])
    assert xy.shape == (5, 2)
    assert fp.fit_points().shape == (50, 2)


def _synthetic_shard(tmp_path, n_games=4, plies_per_game=25):
    board = chess.Board()
    packed_row = encode_packed(board)
    meta_row = encode_meta(board)
    rows = n_games * plies_per_game
    packed = np.tile(packed_row, (rows, 1))
    meta = np.tile(meta_row, (rows, 1))
    ply = np.tile(np.arange(plies_per_game), n_games)
    clock = np.full(rows, 300.0, dtype=np.float32)
    eval_cp = np.full(rows, np.nan, dtype=np.float32)
    game_id = np.repeat(np.arange(n_games), plies_per_game)
    results = [1, -1, 0, 1]
    result = np.repeat(np.array(results[:n_games]), plies_per_game)
    white_elo = np.full(rows, 1800, dtype=np.int64)
    black_elo = np.full(rows, 1800, dtype=np.int64)

    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    np.savez(shard_dir / "shard_00000.npz", packed=packed, meta=meta, ply=ply, clock=clock,
             eval_cp=eval_cp, result=result, white_elo=white_elo, black_elo=black_elo,
             game_id=game_id)
    return shard_dir


def test_load_games_from_shard_balances_results(tmp_path):
    shard_dir = _synthetic_shard(tmp_path, n_games=4, plies_per_game=25)
    games = load_games_from_shard(shard_dir, n_games=3, seed=0, holdout_only=False,
                                  min_plies=20, want_results=(1, -1, 0))
    assert len(games) == 3
    for g in games:
        assert set(g.keys()) >= {"packed", "meta", "ply", "result", "game_id"}
        assert len(g["ply"]) >= 20


def test_load_games_from_shard_holdout_filter(tmp_path):
    shard_dir = _synthetic_shard(tmp_path, n_games=4, plies_per_game=25)
    games = load_games_from_shard(shard_dir, n_games=10, seed=0, holdout_only=True,
                                  min_plies=20, want_results=(1, -1, 0))
    # only game_id 0 (== 0 mod 50) qualifies as holdout among {0,1,2,3}
    assert len(games) == 1
    assert int(games[0]["game_id"][0]) == 0


def _tiny_pgn(tmp_path):
    pgn = (
        '[Event "test"]\n[White "a"]\n[Black "b"]\n[Result "1-0"]\n\n'
        '1. e4 e5 2. Nf3 Nc6 1-0\n\n'
    )
    p = tmp_path / "mini.pgn"
    p.write_text(pgn)
    return p


def test_games_from_pgn_parses_plies(tmp_path):
    p = _tiny_pgn(tmp_path)
    games = games_from_pgn(p)
    assert len(games) == 1
    g = games[0]
    assert g["headers"]["Result"] == "1-0"
    sans = [san for _, san, _ in g["plies"] if san is not None]
    assert sans == ["e4", "e5", "Nf3", "Nc6"]
    assert g["plies"][-1][1] is None  # trailing final-position entry


class _FakeFB:
    """Minimal stand-in for TorchFB: a fixed random linear projection to d dims."""
    def __init__(self, d=8, seed=0):
        import torch
        self.d = d
        g = torch.Generator().manual_seed(seed)
        self._w = None
        self._gen = g

    def _proj(self, flat):
        import torch
        if self._w is None or self._w.shape[0] != flat.shape[1]:
            self._w = torch.randn(flat.shape[1], self.d, generator=self._gen)
        return torch.nn.functional.normalize(flat @ self._w, dim=1)

    def embed_F(self, planes, om):
        n = planes.shape[0]
        return self._proj(planes.reshape(n, -1))

    def embed_B(self, planes):
        return self.embed_F(planes, None)


def test_embed_positions_shapes_and_unit_norm(tmp_path):
    board = chess.Board()
    packed = np.tile(encode_packed(board), (5, 1))
    meta = np.tile(encode_meta(board), (5, 1))
    white_elo = np.full(5, 1800)
    black_elo = np.full(5, 1800)
    clock = np.full(5, 300.0)
    F, B = embed_positions(_FakeFB(), packed, meta, white_elo, black_elo, clock, device="cpu", batch=2)
    assert F.shape == (5, 8) and B.shape == (5, 8)
    assert np.allclose(np.linalg.norm(F, axis=1), 1.0, atol=1e-5)


def test_build_html_roundtrip(tmp_path):
    template = tmp_path / "t.html"
    template.write_text("<html><script>const DATA = /*__DATA__*/;</script></html>")
    out = tmp_path / "out.html"
    payload = {"a": np.float32(1.5), "b": [1, 2, 3]}
    build_html(template, payload, out)
    html = out.read_text()
    assert "const DATA = " in html
    start = html.index("const DATA = ") + len("const DATA = ")
    end = html.index(";</script>")
    data = json.loads(html[start:end])
    assert data["a"] == 1.5 and data["b"] == [1, 2, 3]
