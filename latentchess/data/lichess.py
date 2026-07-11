"""
data/lichess.py — streaming reader + Elo/time-control/clock filter for the
Lichess open PGN database, and the position-shard builder on top of it.

Never materializes a decompressed .pgn on disk: zstandard's stream_reader is
wrapped directly in a TextIOWrapper (needs max_window_size=2**31 -- Lichess
dumps use a long compression window that the default rejects).

The header-level prefilter uses python-chess's own Visitor.end_headers() ->
SKIP extension point (not a hand-rolled PGN tokenizer): a game whose headers
fail GameFilter never gets its movetext SAN-parsed at all (the expensive
part), while the stream still advances correctly to the next game. This
composes correctly where the seemingly-obvious `read_headers` then
`skip_game` pair does NOT (skip_game re-parses from a fresh game boundary,
so calling it after read_headers already consumed that same game's headers
corrupts the stream position -- verified empirically during implementation).
"""
from __future__ import annotations

import io
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

import chess
import chess.pgn
import numpy as np
import zstandard

from latentchess.data.encode import encode_meta, encode_packed


def open_pgn_zst(path) -> io.TextIOWrapper:
    """A text stream over a .pgn.zst file that never decompresses to disk."""
    dctx = zstandard.ZstdDecompressor(max_window_size=2 ** 31)
    fh = open(path, "rb")
    reader = dctx.stream_reader(fh)
    return io.TextIOWrapper(reader, encoding="utf-8", errors="replace")


@dataclass
class GameFilter:
    min_elo: int = 1000
    max_elo: int = 4000
    min_base_seconds: int = 180          # TimeControl "300+0" -> base 300; excludes bullet
    min_plies: int = 20
    skip_first_plies: int = 10           # drop opening-book plies (Maia's recipe)
    min_clock_s: float = 30.0            # drop moves made with < 30s left on the clock
    exclude_bots: bool = True

    def headers_pass(self, h) -> bool:
        try:
            we = int(h.get("WhiteElo", ""))
            be = int(h.get("BlackElo", ""))
        except ValueError:
            return False
        if not (self.min_elo <= we <= self.max_elo and self.min_elo <= be <= self.max_elo):
            return False
        base = _time_control_base_seconds(h.get("TimeControl", ""))
        if base is None or base < self.min_base_seconds:
            return False
        if self.exclude_bots and (h.get("WhiteTitle") == "BOT" or h.get("BlackTitle") == "BOT"):
            return False
        if h.get("Termination") == "Abandoned":
            return False
        return True


def _time_control_base_seconds(tc: str):
    if not tc or tc == "-":
        return None
    base = tc.split("+")[0]
    if "/" in base:            # e.g. "40/9000" (classical, moves/seconds)
        base = base.split("/")[-1]
    try:
        return int(base)
    except ValueError:
        return None


class _FilteringBuilder(chess.pgn.GameBuilder):
    """A GameBuilder that skips movetext parsing entirely (via chess.pgn.SKIP)
    for games whose headers fail `gf`. `self.skipped` disambiguates a
    filtered-out game from a legitimately empty (0-ply) one."""

    def __init__(self, gf: GameFilter):
        super().__init__()
        self.gf = gf
        self.skipped = False

    def end_headers(self):
        if not self.gf.headers_pass(self.game.headers):
            self.skipped = True
            return chess.pgn.SKIP
        return None


def stream_filtered_games(path, gf: GameFilter, max_games: int | None = None) -> Iterator["chess.pgn.Game"]:
    """Yield fully-parsed games whose headers pass `gf`, streaming the .pgn.zst
    without ever materializing it. Counts only YIELDED (header-passing) games
    against `max_games`."""
    with open_pgn_zst(path) as stream:
        kept = 0
        while max_games is None or kept < max_games:
            builder = _FilteringBuilder(gf)
            game = chess.pgn.read_game(stream, Visitor=lambda b=builder: b)
            if game is None:
                return
            if builder.skipped:
                continue
            yield game
            kept += 1


def positions_of(game: "chess.pgn.Game", gf: GameFilter):
    """Yield one dict per kept position: packed bitboards, meta, ply, clock.
    Skips the whole game (no positions) if it's shorter than gf.min_plies."""
    end = game.end()
    if end.ply() < gf.min_plies:
        return
    board = game.board()
    ply = 0
    for node in game.mainline():
        clock = node.clock()
        if ply >= gf.skip_first_plies and not (clock is not None and clock < gf.min_clock_s):
            yield dict(packed=encode_packed(board), meta=encode_meta(board), ply=ply,
                       clock=clock if clock is not None else float("nan"))
        board.push(node.move)
        ply += 1


_RESULT_MAP = {"1-0": 1, "0-1": -1, "1/2-1/2": 0}


def build_shards(pgn_path, gf: GameFilter, out_dir,
                  shard_positions: int = 1_000_000,
                  max_games: int | None = 50_000,
                  max_gb: float | None = 2.0) -> dict:
    """Stream-filter-encode-shard in one bounded pass. Guardrail: at least one
    of max_games/max_gb must bound the run, or a laptop SSD could fill up on
    a full monthly dump."""
    if max_games is None and max_gb is None:
        raise ValueError("max_games and max_gb may not both be None")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    buf = {k: [] for k in ("packed", "meta", "ply", "clock", "result", "white_elo", "black_elo", "game_id")}
    shards = []
    state = {"shard_idx": 0, "bytes_written": 0}

    def flush():
        if not buf["packed"]:
            return
        path = out_dir / f"shard_{state['shard_idx']:05d}.npz"
        np.savez(
            path,
            packed=np.array(buf["packed"], dtype=np.uint64),
            meta=np.array(buf["meta"], dtype=np.uint8),
            ply=np.array(buf["ply"], dtype=np.int32),
            clock=np.array(buf["clock"], dtype=np.float32),
            result=np.array(buf["result"], dtype=np.int8),
            white_elo=np.array(buf["white_elo"], dtype=np.uint16),
            black_elo=np.array(buf["black_elo"], dtype=np.uint16),
            game_id=np.array(buf["game_id"], dtype=np.uint32),
        )
        state["bytes_written"] += path.stat().st_size
        shards.append({"file": path.name, "n": len(buf["packed"])})
        state["shard_idx"] += 1
        for k in buf:
            buf[k] = []

    games_scanned = 0
    games_kept = 0
    positions = 0

    for game in stream_filtered_games(pgn_path, gf, max_games=None):
        games_scanned += 1
        res = _RESULT_MAP.get(game.headers.get("Result", ""), 0)
        we = int(game.headers.get("WhiteElo", 0) or 0)
        be = int(game.headers.get("BlackElo", 0) or 0)
        any_pos = False
        for pos in positions_of(game, gf):
            any_pos = True
            buf["packed"].append(pos["packed"])
            buf["meta"].append(pos["meta"])
            buf["ply"].append(pos["ply"])
            buf["clock"].append(pos["clock"])
            buf["result"].append(res)
            buf["white_elo"].append(we)
            buf["black_elo"].append(be)
            buf["game_id"].append(games_kept)
            positions += 1
            if len(buf["packed"]) >= shard_positions:
                flush()
        if any_pos:
            games_kept += 1
        if max_games is not None and games_kept >= max_games:
            break
        if max_gb is not None and state["bytes_written"] > max_gb * 2 ** 30:
            break
    flush()

    manifest = dict(source=str(pgn_path), filter=asdict(gf), games_scanned=games_scanned,
                     games_kept=games_kept, positions=positions, shards=shards)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest
