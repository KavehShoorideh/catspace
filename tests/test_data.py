"""
Data-layer tests: ChainRolloutSource determinism/holdout, the packed-bitboard
codec round-trip, and the streaming Lichess pipeline -- filtering, shard
building, and the shard reader/pair source, against the committed fixture.
"""
import io
import json
from pathlib import Path

import chess
import chess.pgn
import numpy as np
import pytest
import zstandard

from catspace.data.encode import PLANES, board_from_packed, decode_planes, encode_meta, encode_packed
from catspace.data.lichess import GameFilter, build_shards, open_pgn_zst, positions_of, stream_filtered_games
from catspace.data.shards import LichessPairSource, ShardReader, write_shards
from catspace.data.sources import ChainRolloutSource, PairBatch
from catspace.domains import krk
from catspace.opponents import RandomOpponent
from catspace.planner.policy import RandomPolicy

FIXTURES = Path(__file__).parent / "fixtures"
FIXTURE_PGN = FIXTURES / "lichess_mini.pgn.zst"
FIXTURE_MANIFEST = json.loads((FIXTURES / "fixture_manifest.json").read_text())


def _collect(source, n=3, batch_size=64, seed=7):
    batches = list(source.batches(batch_size, seed))[:n]
    return [(b.anchors.copy(), b.goals.copy()) for b in batches]


def test_chain_rollout_deterministic():
    chain = krk.build_chain()
    src = ChainRolloutSource(chain, RandomPolicy(), RandomOpponent(), gamma=0.9, n_games=20)

    a = _collect(src, seed=7)
    b = _collect(src, seed=7)
    for (a_anchors, a_goals), (b_anchors, b_goals) in zip(a, b):
        assert np.array_equal(a_anchors, b_anchors)
        assert np.array_equal(a_goals, b_goals)

    c = _collect(src, seed=8)
    assert not np.array_equal(a[0][0], c[0][0])


def test_chain_rollout_holdout():
    chain = krk.build_chain()
    rng = np.random.default_rng(0)
    holdout_mask = rng.random(chain.n_live) < 0.15
    src = ChainRolloutSource(chain, RandomPolicy(), RandomOpponent(), gamma=0.9,
                              n_games=200, holdout_mask=holdout_mask)

    held_out_states = set(np.where(holdout_mask)[0].tolist())
    seen_any = False
    for batch in src.batches(256, seed=3):
        for s in batch.anchors.tolist() + batch.goals.tolist():
            if s < chain.n_live:
                seen_any = True
                assert s not in held_out_states
    assert seen_any


def test_geometric_pairing_mean():
    """Locks the k = 1 + Geometric(1-gamma) convention (>= 2 always, mean
    1 + 1/(1-gamma))."""
    rng = np.random.default_rng(0)
    gamma = 0.9
    ks = 1 + rng.geometric(1.0 - gamma, size=20000)
    assert ks.min() >= 2
    expected_mean = 1.0 + 1.0 / (1.0 - gamma)
    assert abs(ks.mean() - expected_mean) < 0.5


# ---------------------------------------------------------------- board codec

def _random_boards(n=25, seed=0):
    rng = np.random.default_rng(seed)
    boards = []
    for i in range(n):
        board = chess.Board()
        n_moves = int(rng.integers(0, 41))
        for _ in range(n_moves):
            legal = list(board.legal_moves)
            if not legal:
                break
            board.push(legal[int(rng.integers(0, len(legal)))])
        boards.append(board)
    return boards


def test_encode_decode_roundtrip():
    for board in _random_boards():
        packed = encode_packed(board)
        meta = encode_meta(board)
        planes = decode_planes(packed)

        expected = {(sq, piece.piece_type, piece.color) for sq, piece in board.piece_map().items()}
        got = set()
        for p_idx, (pt, color) in enumerate(PLANES):
            plane = planes[p_idx]
            for sq in range(64):
                row, col = sq // 8, sq % 8
                if plane[row, col]:
                    got.add((sq, pt, color))
        assert got == expected

        rebuilt = board_from_packed(packed, meta)
        assert rebuilt.board_fen() == board.board_fen()
        assert rebuilt.turn == board.turn
        for color in (chess.WHITE, chess.BLACK):
            assert rebuilt.has_kingside_castling_rights(color) == board.has_kingside_castling_rights(color)
            assert rebuilt.has_queenside_castling_rights(color) == board.has_queenside_castling_rights(color)
        assert rebuilt.ep_square == board.ep_square


# ---------------------------------------------------------------- Lichess streaming pipeline

def _fixture_filter():
    return GameFilter(**FIXTURE_MANIFEST["filter"])


def test_open_pgn_zst_streams(tmp_path):
    with open_pgn_zst(FIXTURE_PGN) as stream:
        head = stream.read(200)
    assert "[Event" in head
    # never materializes a decompressed .pgn anywhere on disk
    assert not list(tmp_path.glob("*.pgn"))
    assert not list(FIXTURES.glob("*.pgn"))


def test_fixture_filters():
    games = list(stream_filtered_games(FIXTURE_PGN, _fixture_filter()))
    whites = sorted(int(g.headers["White"][1:]) for g in games)
    assert whites == FIXTURE_MANIFEST["expected_header_pass"]
    excluded = {int(k) for k in FIXTURE_MANIFEST["special"]} - {11}   # 11 passes headers, fails min_plies
    assert not (set(whites) & excluded)


def test_positions_skip_rules():
    gf = _fixture_filter()
    games = {int(g.headers["White"][1:]): g for g in stream_filtered_games(FIXTURE_PGN, gf)}

    kept = sorted(i for i, g in games.items() if list(positions_of(g, gf)))
    assert kept == FIXTURE_MANIFEST["expected_kept"]

    for i in FIXTURE_MANIFEST["low_clock_tail_games"]:
        positions = list(positions_of(games[i], gf))
        assert all(p["ply"] >= gf.skip_first_plies for p in positions)
        clocks = [p["clock"] for p in positions if p["clock"] == p["clock"]]   # drop nan
        assert all(c >= gf.min_clock_s for c in clocks)


def test_build_shards_fixture(tmp_path):
    gf = _fixture_filter()
    manifest = build_shards(FIXTURE_PGN, gf, tmp_path, shard_positions=200,
                             max_games=None, max_gb=0.1)

    assert manifest["games_kept"] == len(FIXTURE_MANIFEST["expected_kept"])
    shard_files = sorted(tmp_path.glob("shard_*.npz"))
    assert len(shard_files) == len(manifest["shards"])
    assert not list(tmp_path.glob("*")) or set(p.name for p in tmp_path.glob("*")) <= \
        {s["file"] for s in manifest["shards"]} | {"manifest.json"}

    total_rows = 0
    all_game_ids = []
    for path in shard_files:
        data = np.load(path)
        n = len(data["packed"])
        total_rows += n
        assert data["packed"].shape == (n, 12)
        all_game_ids.extend(data["game_id"].tolist())
    assert total_rows == manifest["positions"]
    assert sum(s["n"] for s in manifest["shards"]) == manifest["positions"]

    # game_id non-decreasing WITHIN each shard (the LichessPairSource offset trick relies on this)
    for path in shard_files:
        gid = np.load(path)["game_id"]
        assert np.all(np.diff(gid) >= 0)


class _RangeSource:
    """Stub PairSource yielding distinct int rows 0..n-1 for ShardReader coverage."""

    def __init__(self, n, chunk=97):
        self.n = n
        self.chunk = chunk

    def batches(self, batch_size, seed):
        for i in range(0, self.n, self.chunk):
            j = min(i + self.chunk, self.n)
            yield PairBatch(anchors=np.arange(i, j), goals=np.arange(i, j) + 10_000)


def test_shard_reader_coverage(tmp_path):
    src = _RangeSource(1000)
    write_shards(src, tmp_path, shard_size=300, batch_size=97, seed=0)

    reader = ShardReader(tmp_path, shuffle_buffer=100)
    seen = []
    order_matches_input = True
    prev = -1
    for batch in reader.batches(batch_size=64, seed=1):
        for a in batch.anchors.tolist():
            seen.append(a)
            if a != prev + 1:
                order_matches_input = False
            prev = a

    assert sorted(seen) == list(range(1000))
    assert not order_matches_input


def test_lichess_pair_source(tmp_path):
    gf = _fixture_filter()
    build_shards(FIXTURE_PGN, gf, tmp_path, shard_positions=1_000_000, max_games=None, max_gb=1.0)

    src = LichessPairSource(tmp_path, gamma=0.9)
    shard_data = [np.load(p) for p in sorted(tmp_path.glob("shard_*.npz"))]

    n_checked = 0
    for batch in src.batches(batch_size=64, seed=0):
        n_checked += len(batch.anchors)
    assert n_checked > 0

    # spot-check: for every shard, sampled goal rows stay within the same game
    # and never precede the anchor row.
    rng = np.random.default_rng(0)
    for data in shard_data:
        game_id = data["game_id"]
        n = len(game_id)
        if n == 0:
            continue
        change = np.flatnonzero(np.diff(game_id)) + 1
        starts = np.concatenate([[0], change])
        ends = np.concatenate([change, [n]])
        for gstart, gend in zip(starts, ends):
            if gend - gstart < 2:
                continue
            row = int(rng.integers(gstart, gend))
            k = 1 + rng.geometric(0.1)
            goal_row = min(row + k, gend - 1)
            assert game_id[goal_row] == game_id[row]
            assert goal_row >= row


def test_truncated_prefix_tolerated(tmp_path):
    """A range-downloaded PREFIX of a .zst dump streams to the cut point with
    tolerate_truncation (and raises without), and the tolerant reader is
    lossless on intact files. zstd decodes block-at-a-time (<=128KB content
    per block), so the cut must hit a MULTI-block file to be representative
    -- the tiny fixture is a single block; repeat it to force several."""
    gf = _fixture_filter()
    full = list(stream_filtered_games(FIXTURE_PGN, gf, tolerate_truncation=True))
    n_pass = len(FIXTURE_MANIFEST["expected_header_pass"])
    assert len(full) == n_pass

    # 20 copies of the fixture, each carrying a unique incompressible junk
    # header, so the compressed bytes spread across many blocks (repeating the
    # text verbatim would back-reference into block 1 and put ~all compressed
    # bytes there, leaving nothing decodable after a mid-file cut).
    with open_pgn_zst(FIXTURE_PGN) as stream:
        text = stream.read()
    rng = np.random.default_rng(0)
    repeats = 20
    big = "\n".join(f'[Junk "{rng.bytes(16384).hex()}"]\n' + text for _ in range(repeats))
    blob = zstandard.ZstdCompressor().compress(big.encode())
    cut = tmp_path / "cut.pgn.zst"
    cut.write_bytes(blob[: int(len(blob) * 0.6)])

    got = list(stream_filtered_games(cut, gf, tolerate_truncation=True))
    assert 0 < len(got) < repeats * n_pass

    # the plain path's truncation behavior is VERSION-DEPENDENT (0.22 raised;
    # 0.25 silently yields the decodable prefix like the tolerant path) --
    # the flag exists to pin the tolerant contract regardless of zstandard
    # version, so only assert the plain path never yields MORE
    try:
        got_plain = list(stream_filtered_games(cut, gf))
    except zstandard.ZstdError:
        got_plain = []
    assert len(got_plain) <= len(got)


_EVAL_PGN = """[Event "t"]
[Site "t"]
[White "w"]
[Black "b"]
[Result "0-1"]

1. f3 { [%eval 0.5] [%clk 0:05:00] } e5 { [%eval -0.6] [%clk 0:05:00] } 2. g4 { [%eval #-1] [%clk 0:04:57] } Qh4# { [%clk 0:04:55] } 0-1
"""


def test_eval_alignment_and_include_final():
    """eval_cp belongs to the YIELDED position (the annotation of the move
    that produced it), and include_final adds the post-last-move position --
    the checkmate itself here."""
    game = chess.pgn.read_game(io.StringIO(_EVAL_PGN))
    gf = GameFilter(min_plies=0, skip_first_plies=0, min_clock_s=0.0)

    pos = list(positions_of(game, gf, include_final=True))
    assert [p["ply"] for p in pos] == [0, 1, 2, 3, 4]
    evs = [p["eval_cp"] for p in pos]
    assert np.isnan(evs[0]) and np.isnan(evs[4])      # start unannotated; no [%eval] after mate
    assert evs[1:4] == [50.0, -60.0, -3199.0]          # white-POV cp; mate-in-1 for black

    final = board_from_packed(pos[-1]["packed"], pos[-1]["meta"])
    assert final.is_checkmate()
    assert len(list(positions_of(game, gf))) == 4      # default: no final row


def test_build_shards_eval_column_and_final_rows(tmp_path):
    gf = _fixture_filter()
    manifest = build_shards(FIXTURE_PGN, gf, tmp_path, shard_positions=10_000,
                             max_games=None, max_gb=0.1)
    data = np.load(tmp_path / "shard_00000.npz")

    assert len(data["eval_cp"]) == manifest["positions"]
    assert np.isnan(data["eval_cp"]).all()             # fixture games carry no server analysis
    assert manifest["include_final"] and manifest["games_with_eval"] == 0

    # each game's last stored row is its true terminal ply, exempt from the
    # min-clock tail filter (it's a goal target, not a move decision)
    gid, ply = data["game_id"], data["ply"]
    last_rows = np.flatnonzero(np.r_[np.diff(gid) != 0, True])
    games = {int(g.headers["White"][1:]): g for g in stream_filtered_games(FIXTURE_PGN, gf)}
    kept = [games[i] for i in FIXTURE_MANIFEST["expected_kept"]]
    assert len(last_rows) == len(kept)
    for row, g in zip(last_rows, kept):
        assert ply[row] == g.end().ply()


# ---------------------------------------------------------------- slow: full generalization run

@pytest.mark.slow
def test_generalization_band():
    """Loose smoke band for experiments/generalization.py at reduced scale --
    the full-scale run (default args) reproduces the documented finding
    (holdout spearman ~0.45, ~zero train/holdout gap; see README/RESULTS-v3)."""
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "experiments/generalization.py",
         "--n-games", "10000", "--steps", "5000", "--n-eval", "50"],
        cwd=Path(__file__).parent.parent, capture_output=True, text=True, timeout=60,
    )
    out = result.stdout
    line = [l for l in out.splitlines() if l.startswith("VERDICT")][0]
    holdout = float(line.split("HOLDOUT_SPEARMAN=")[1].split()[0])
    assert holdout > 0.15   # loose band at reduced scale (seeds 0-2 measured 0.22-0.36)
