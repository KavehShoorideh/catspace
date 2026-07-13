"""
viz/realboard.py — shared payload helpers for the real-board interactive
viewers (VIZ_PLAN.md D1-D7): loading balanced game samples from shards,
parsing arena PGNs, batched F/B embedding under each row's OWN omega, board
SVG rendering, and a thin projection-fit wrapper shared across builders.
"""
from __future__ import annotations

from pathlib import Path

import chess
import chess.svg
import numpy as np

from catspace.data.encode import board_from_packed, encode_meta, encode_packed
from catspace.nn.features import feature_planes, omega_ids
from catspace.viz.projection import Normalizer, PCAProjection, TSNEProjection

_COLS = ("packed", "meta", "ply", "clock", "eval_cp", "result", "white_elo", "black_elo", "game_id")


def load_games_from_shard(shard_dir, n_games: int, seed: int = 0, holdout_only: bool = True,
                          min_plies: int = 20, max_plies: int = 160,
                          want_results=(1, -1, 0), max_start_ply: int = 12) -> list:
    """Scan the first shard file and return up to n_games complete games as
    dicts of per-ply arrays sorted by ply. 'Complete' = the game's first
    stored row has ply <= max_start_ply (data.lichess.skip_first_plies is 10
    by default -- games split across shard boundaries mid-game are skipped).
    Round-robins across want_results for contrast."""
    shard_dir = Path(shard_dir)
    path = sorted(shard_dir.glob("shard_*.npz"))[0]
    npz = np.load(path)
    data = {k: npz[k] for k in _COLS if k in npz.files}
    gid = data["game_id"]
    ply = data["ply"]
    n = len(gid)

    change = np.flatnonzero(np.diff(gid)) + 1
    starts = np.concatenate([[0], change])
    ends = np.concatenate([change, [n]])

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(starts))
    by_result: dict = {r: [] for r in want_results}
    for oi in order:
        s0, s1 = starts[oi], ends[oi]
        if ply[s0] > max_start_ply:
            continue
        n_ply = s1 - s0
        if n_ply < min_plies:
            continue
        g = int(gid[s0])
        if holdout_only and g % 50 != 0:
            continue
        res = int(data["result"][s0])
        if res not in by_result:
            continue
        by_result[res].append((s0, min(s1, s0 + max_plies)))

    games = []
    i = 0
    keys = list(want_results)
    while len(games) < n_games and any(by_result[k] for k in keys):
        r = keys[i % len(keys)]
        i += 1
        if not by_result[r]:
            continue
        s0, s1 = by_result[r].pop()
        games.append({k: data[k][s0:s1] for k in data})
    return games


def games_from_pgn(path) -> list:
    """Parse a PGN file into games: list of dicts with headers White/Black/
    Result and a 'plies' list of (board_before_move, san, uci) triples plus a
    trailing (final_board, None, None) entry for the terminal position."""
    import chess.pgn

    games = []
    with open(path) as fh:
        while True:
            game = chess.pgn.read_game(fh)
            if game is None:
                break
            board = game.board()
            plies = []
            node = game
            for move in game.mainline_moves():
                san = board.san(move)
                plies.append((board.copy(stack=False), san, move.uci()))
                board.push(move)
            plies.append((board.copy(stack=False), None, None))
            games.append({"headers": dict(game.headers), "plies": plies})
    return games


def infer_san(prev_board: chess.Board, packed_next: np.ndarray, meta_next: np.ndarray):
    """Recover the SAN of the move between two consecutive stored positions by
    trying each legal move of prev_board and comparing the encoded child to
    the stored next row. Returns (san, move) or (None, None)."""
    for mv in prev_board.legal_moves:
        child = prev_board.copy(stack=False)
        child.push(mv)
        if np.array_equal(encode_packed(child), packed_next) and \
           np.array_equal(encode_meta(child), meta_next):
            return prev_board.san(mv), mv
    return None, None


def embed_positions(fb, packed: np.ndarray, meta: np.ndarray, white_elo: np.ndarray,
                    black_elo: np.ndarray, clock: np.ndarray, device: str, batch: int = 2048):
    """Batched (F, B) for arbitrary rows under each row's OWN omega. Returns
    (F, B) numpy arrays, unit rows."""
    import torch

    om = omega_ids(white_elo, black_elo, clock)
    Fs, Bs = [], []
    n = len(packed)
    for i in range(0, n, batch):
        sl = slice(i, min(i + batch, n))
        planes = torch.from_numpy(feature_planes(packed[sl], meta[sl])).to(device)
        om_t = torch.from_numpy(om[sl]).to(device)
        with torch.no_grad():
            Fs.append(fb.embed_F(planes, om_t).cpu().numpy())
            Bs.append(fb.embed_B(planes).cpu().numpy())
    return np.concatenate(Fs), np.concatenate(Bs)


def board_svg(board: chess.Board, lastmove=None, arrows=(), size: int = 400) -> str:
    return chess.svg.board(board, size=size, lastmove=lastmove, arrows=arrows)


class _FittedProjection:
    def __init__(self, normalizer: Normalizer, projection):
        self.normalizer = normalizer
        self.projection = projection

    def transform(self, F: np.ndarray) -> np.ndarray:
        return self.projection.transform(self.normalizer.apply(F))

    def fit_points(self) -> np.ndarray:
        return self.projection.fit_points()


def fit_projection(F_bg: np.ndarray, kind: str = "pca", seed: int = 0,
                   perplexity: float = 40.0) -> _FittedProjection:
    """Fit a Normalizer + {pca,tsne} projection on F_bg (any 2D float array).
    Not toy-specific (unlike viz.projection.fit_map, which wants dtm/won)."""
    normalizer = Normalizer.fit(F_bg)
    Fn = normalizer.apply(F_bg)
    if kind == "pca":
        proj = PCAProjection().fit(Fn)
    elif kind == "tsne":
        proj = TSNEProjection(perplexity=perplexity, seed=seed).fit(Fn)
    else:
        raise ValueError(f"unknown projection kind {kind!r}")
    return _FittedProjection(normalizer, proj)


def board_from_row(packed_row: np.ndarray, meta_row: np.ndarray) -> chess.Board:
    return board_from_packed(packed_row, meta_row)
