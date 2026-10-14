"""LEARNING BY WATCHING, THEN BY PLAYING (Kaveh 2026-10-13: "We have to start learning by watching, then by playing").

Before the from-scratch run plays its first game it WATCHES a teacher play the material from the same curriculum starts:

  THE TEACHER   the tablebase's own play for five men or fewer (the winner: the fewest plies to the next capture or pawn
                move -- in these endings the mate; the loser: the longest resistance), Stockfish beyond
  LOOPS         the teacher's games read by the loop-frame proposer (patterns.derive_loops): arrangements seen again with
                their king shifted, checked by our own exact search, refined, held out
  SURPRISES     the positions where the teacher's move is proved good and our planner's own move proved bad
                (explorer.locate_watched) -- explored, separated, repaired, through every guard (held out, a second
                surprise, fresh games)

The teacher only shows where something exists; every lesson is checked by our own search before it is kept. Then the
rounds of play begin from what was learned (surprise_loop --watch N).
"""
from __future__ import annotations

import json
import os
import random
import time

import chess

from catspace.io import paths

TB = ENG = None


def _init():
    global TB, ENG
    import chess.engine
    import chess.syzygy
    TB = chess.syzygy.open_tablebase(str(paths.syzygy_dir()))
    ENG = chess.engine.SimpleEngine.popen_uci("stockfish")
    ENG.configure({"SyzygyPath": str(paths.syzygy_dir()), "Threads": 1})


def _tb_move(b: chess.Board) -> chess.Move:
    best = None
    for m in b.legal_moves:
        x = b.copy(stack=False); x.push(m)
        if x.is_checkmate():
            return m
        d = TB.probe_dtz(x)
        key = (abs(d) if TB.probe_wdl(x) == -2 else 10 ** 6) if b.turn == chess.WHITE else -d
        if best is None or key < best[0]:
            best = (key, m)
    return best[1]


def _teach(args) -> dict:
    """One teacher's game from a curriculum start: rows of the teacher's positions (the loop's format)."""
    import chess.engine
    from catspace.incubator.analyst.material_walk import _material, _start, engine_start
    seed, material, band = args
    men = len(material.replace("v", ""))
    b = _start(random.Random(seed), material, TB, band) if men <= 5 else engine_start(random.Random(seed), material, ENG)
    rows = []
    for _ in range(300):
        if b.is_game_over(claim_draw=True):
            break
        m = _tb_move(b) if men <= 5 and len(b.piece_map()) <= 5 else ENG.play(b, chess.engine.Limit(depth=14)).move
        if b.turn == chess.WHITE:
            rows.append(dict(fen=b.fen(), material=_material(b), san=b.san(m), why="the teacher"))
        b.push(m)
    return dict(seed=seed, out="win" if b.is_checkmate() and b.turn == chess.BLACK else "draw", moves=len(rows), rows=rows)


def teacher_games(material: str, n: int, band, seed0: int = 50000, workers: int | None = None) -> list:
    from multiprocessing import get_context
    with get_context("spawn").Pool(workers or max(1, (os.cpu_count() or 8) - 2), initializer=_init) as p:
        return p.map(_teach, [(seed0 + i, material, band) for i in range(n)])


def watch(material: str, n: int, band, frames: dict, tag: str, surprises: int = 12, utility_n: int = 200, log=print) -> dict:
    """THE WATCHING PHASE -> the frames to start playing with."""
    from . import explorer as E
    from . import patterns as Pt
    from . import scratch
    t0 = time.time()
    games = teacher_games(material, n, band)
    path = paths.experiment(f"watch_{material}_{tag}.json")
    json.dump(games, open(path, "w"))
    won = [g for g in games if g["out"] == "win"]
    log(f"OBS WATCHING: the teacher played {len(games)} games of {material} from the curriculum's starts (band {band}): mated "
        f"{len(won)}, {sum(g['moves'] for g in won) / max(len(won), 1):.1f} moves a mate ({time.time() - t0:.0f} s)")
    mats = {r["material"] for g in games for r in g["rows"]}
    loops = Pt.derive_loops([str(path)], mats, log=log)
    frames = dict(frames); frames.update(loops)
    log(f"OBS WATCHING: {len(loops)} loop frame(s) from the teacher's games")
    E.MODE, E.DEPTH = "win", 12
    root = f"scratch/{scratch()}" if scratch() else "explorer"
    frames, found, kept = E.run_surprises(games, frames, surprises, utility_n, watched=True,
                                          ledger_path=paths.derived(f"{root}/watched_lemmas.json"), log=log)
    log(f"OBS WATCHING: {sum(1 for x in found if x.get('found'))} of {len(found)} teacher games showed a surprise of ours; "
        f"{len(kept)} lemma(s) kept from them ({time.time() - t0:.0f} s in all)")
    return frames
