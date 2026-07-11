#!/usr/bin/env python
"""
tests/fixtures/make_fixture.py — generates the committed lichess_mini.pgn.zst
fixture (30 seeded synthetic games with %clk comments) + fixture_manifest.json
recording, for the DEFAULT GameFilter, which games are expected to pass the
header filter and which are expected to contribute >=1 kept position -- so
tests/test_data.py reads expectations from this manifest instead of
hardcoding counts.

Run once and commit both outputs:
    python tests/fixtures/make_fixture.py

Special-cased games (0-indexed) exercise one exclusion reason each:
  3  -> WhiteTitle "BOT"        (header filter: exclude_bots)
  5  -> WhiteElo 800            (header filter: min_elo)
  7  -> TimeControl "60+0"      (header filter: bullet, min_base_seconds)
  9  -> Termination "Abandoned" (header filter)
  11 -> only 12 plies total     (position filter: min_plies -- passes headers,
                                  contributes zero positions)
  13, 17 -> normal games whose last few plies drop below 30s on the clock
            (position filter: min_clock_s drops only those late plies, the
            game still contributes its earlier positions)
All other games are "normal": pass every filter, contribute
(n_plies - skip_first_plies) positions each.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import chess
import chess.pgn
import numpy as np
import zstandard

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from latentchess.data.lichess import GameFilter

HERE = Path(__file__).resolve().parent
N_GAMES = 30
DEFAULT_FILTER = GameFilter()   # min_elo=1000, max_elo=4000, min_base_seconds=180,
                                # min_plies=20, skip_first_plies=10, min_clock_s=30


def _hms(total_seconds: float) -> str:
    total_seconds = max(0, int(round(total_seconds)))
    h, rem = divmod(total_seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}"


def _make_game(i: int, rng: np.random.Generator):
    special = {3: "bot", 5: "low_elo", 7: "bullet", 9: "abandoned", 11: "too_short"}.get(i)
    low_clock_tail = i in (13, 17)

    n_plies = 12 if special == "too_short" else int(rng.integers(24, 61))
    white_elo = 800 if special == "low_elo" else int(rng.integers(1200, 2201))
    black_elo = int(rng.integers(1200, 2201))
    time_control = "60+0" if special == "bullet" else "300+0"

    game = chess.pgn.Game()
    game.headers["Event"] = "Rated Blitz game"
    game.headers["White"] = f"p{i}"
    game.headers["Black"] = f"q{i}"
    game.headers["WhiteElo"] = str(white_elo)
    game.headers["BlackElo"] = str(black_elo)
    game.headers["TimeControl"] = time_control
    game.headers["UTCDate"] = "2024.01.01"
    game.headers["UTCTime"] = "12.00.00"
    game.headers["Termination"] = "Abandoned" if special == "abandoned" else "Normal"
    if special == "bot":
        game.headers["WhiteTitle"] = "BOT"

    node = game
    board = chess.Board()
    clocks = {chess.WHITE: 300.0, chess.BLACK: 300.0}
    tail_start = n_plies - 5
    for ply in range(n_plies):
        legal = list(board.legal_moves)
        if not legal:
            break
        move = legal[int(rng.integers(0, len(legal)))]
        mover = board.turn
        if low_clock_tail and ply >= tail_start:
            clocks[mover] = max(0.0, 40.0 - (ply - tail_start) * 8.0)
        else:
            clocks[mover] = max(0.0, clocks[mover] - float(rng.uniform(2.0, 4.0)))
        node = node.add_variation(move)
        node.comment = f"[%clk {_hms(clocks[mover])}]"
        board.push(move)

    result = board.result(claim_draw=True) if board.is_game_over(claim_draw=True) else \
        ["1-0", "0-1", "1/2-1/2"][i % 3]
    game.headers["Result"] = result
    return game, special


def main():
    rng = np.random.default_rng(42)
    games = []
    specials = {}
    for i in range(N_GAMES):
        game, special = _make_game(i, rng)
        games.append(game)
        if special is not None:
            specials[i] = special

    exporter_text = []
    for game in games:
        exporter = chess.pgn.StringExporter(headers=True, variations=False, comments=True)
        game.accept(exporter)
        exporter_text.append(str(exporter))
    pgn_text = "\n\n".join(exporter_text) + "\n"

    out_pgn = HERE / "lichess_mini.pgn.zst"
    cctx = zstandard.ZstdCompressor()
    out_pgn.write_bytes(cctx.compress(pgn_text.encode("utf-8")))

    header_excluded = {3, 5, 7, 9}
    position_excluded = header_excluded | {11}
    manifest = {
        "n_games": N_GAMES,
        "filter": asdict(DEFAULT_FILTER),
        "special": specials,
        "expected_header_pass": sorted(set(range(N_GAMES)) - header_excluded),
        "expected_kept": sorted(set(range(N_GAMES)) - position_excluded),
        "low_clock_tail_games": [13, 17],
    }
    (HERE / "fixture_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {out_pgn} ({out_pgn.stat().st_size} bytes) and fixture_manifest.json")


if __name__ == "__main__":
    main()
