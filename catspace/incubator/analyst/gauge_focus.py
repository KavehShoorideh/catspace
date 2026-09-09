"""gauge_focus.py -- WHERE IS THE ACTION, measured (Kaveh 2026-09-09: "I don't have a prior on moves, I have a prior on
sub-games; recognise the important one, narrow the search to it, and next move it is probably the same one").

Zones are the sub-games the graph finds (components of the lemma graph); a zone's temperature is what a free move
there is worth to whichever side profits most. Two numbers, with degenerate baselines beside them:

  focus accuracy   does the engine's best move TOUCH the hottest zone? (the mover, the captured piece, or a piece the
                   mover attacks on arrival is in the zone) -- against the largest zone, the zone of the previous best
                   move, a random zone, and 'any zone at all'
  persistence      does the engine's best move stay in the zone of the previous ply's best move (zones matched across
                   the move by piece overlap)? and in the zone of the move actually played?

Positions come in windows of consecutive plies from real games so persistence is measurable.

    python -m catspace.incubator.analyst.gauge_focus data/lichess/huvaxshtra_all.pgn [--games 60 --window 12 --depth 12]
"""
from __future__ import annotations

import argparse
import random
import time
from collections import Counter, defaultdict

import chess
import chess.pgn

from .memo import Memo
from .graph import build
from .rank import move_kind


def touched(board: chess.Board, g, m: chess.Move) -> list[int]:
    """Indices of the zones the move touches, hottest first."""
    pieces = {m.from_square}
    if board.piece_at(m.to_square) is not None:
        pieces.add(m.to_square)
    if board.is_en_passant(m):
        pieces.add(m.to_square + (-8 if board.turn == chess.WHITE else 8))
    nb = board.copy(stack=False); nb.push(m)
    for t in nb.attacks(m.to_square):
        if nb.piece_at(t) is not None and nb.piece_at(t).color != board.turn:
            pieces.add(t)
    return [zi for zi, z in enumerate(g.zones) if z & pieces]


def carry(zone: set[int], move: chess.Move) -> set[int]:
    """The zone's squares after the move (the mover carries its membership)."""
    out = set()
    for s in zone:
        out.add(move.to_square if s == move.from_square else s)
    return out


def same_zone(prev: set[int] | None, cur: set[int]) -> bool:
    if not prev:
        return False
    return len(prev & cur) / len(prev) >= 0.5


def run(pgn_path: str, engine, games: int = 60, window: int = 12, depth: int = 12, seed: int = 3, log=print):
    rng = random.Random(seed)
    rows = []; memo = Memo(); t0 = time.perf_counter()
    with open(pgn_path) as f:
        gs = []
        while len(gs) < games * 3:
            g = chess.pgn.read_game(f)
            if g is None:
                break
            gs.append(g)
    rng.shuffle(gs)
    n = 0
    for game in gs[:games]:
        moves = list(game.mainline_moves())
        if len(moves) < 20 + window:
            continue
        start = rng.randrange(10, len(moves) - window)
        board = game.board()
        for m in moves[:start]:
            board.push(m)
        prev_best_zone = None; prev_played_zone = None
        for k in range(window):
            if board.is_game_over():
                break
            try:
                g = build(board, memo)
                line = engine.best(board, depth=depth)
            except Exception as ex:
                log(f"[focus] error {ex}", flush=True); break
            if line is None:
                break
            best = chess.Move.from_uci(line.move)
            tz = touched(board, g, best)
            row = dict(game=game.headers.get("Site", "?"), ply=start + k, zones=len(g.zones), hottest=g.temperatures[0] if g.temperatures else 0.0,
                       hot_positive=bool(g.temperatures and g.temperatures[0] > 0), kind=move_kind(board, best), n_touched=len(tz),
                       hits_hottest=(0 in tz), hits_largest=(max(range(len(g.zones)), key=lambda i: len(g.zones[i])) in tz) if g.zones else False,
                       hits_random=(rng.randrange(len(g.zones)) in tz) if g.zones else False, hits_any=bool(tz),
                       hits_prev_best=any(same_zone(prev_best_zone, g.zones[i]) for i in tz), had_prev_best=prev_best_zone is not None,
                       hits_prev_played=any(same_zone(prev_played_zone, g.zones[i]) for i in tz), had_prev_played=prev_played_zone is not None,
                       phase=g.position["phase"], seconds=g.seconds, in_check=bool(board.is_check()))
            rows.append(row); n += 1
            played = moves[start + k]
            prev_best_zone = carry(g.zones[tz[0]], played) if tz else None
            pz = touched(board, g, played)
            prev_played_zone = carry(g.zones[pz[0]], played) if pz else None
            board.push(played)
            if n % 20 == 0:
                log(f"[focus] {n} positions, {time.perf_counter() - t0:.0f}s, hottest {sum(r['hits_hottest'] for r in rows)}/{n}", flush=True)
    return rows, time.perf_counter() - t0


def report(rows, dt) -> str:
    n = len(rows)
    def pct(key, rs=None, cond=None):
        rs = rs if rs is not None else rows
        rs = [r for r in rs if cond is None or cond(r)]
        return f"{sum(r[key] for r in rs) / len(rs):.0%} ({len(rs)})" if rs else "-"
    out = [f"[focus] {n} positions in {dt:.0f}s ({dt / max(1, n):.1f}s each; graph {sum(r['seconds'] for r in rows) / max(1, n):.2f}s each)",
           f"zones per position {sum(r['zones'] for r in rows) / max(1, n):.2f}; a zone with positive temperature exists in {pct('hot_positive')}; "
           f"the best move touches some zone {pct('hits_any')}",
           f"FOCUS: best move touches the hottest zone {pct('hits_hottest')} | largest zone {pct('hits_largest')} | random zone {pct('hits_random')} | previous best move's zone {pct('hits_prev_best', cond=lambda r: r['had_prev_best'])}",
           f"  when a positive temperature exists: hottest {pct('hits_hottest', cond=lambda r: r['hot_positive'])} | largest {pct('hits_largest', cond=lambda r: r['hot_positive'])} | random {pct('hits_random', cond=lambda r: r['hot_positive'])}",
           f"  when there are two zones or more (focus is a choice): hottest {pct('hits_hottest', cond=lambda r: r['zones'] >= 2)} | largest {pct('hits_largest', cond=lambda r: r['zones'] >= 2)} | random {pct('hits_random', cond=lambda r: r['zones'] >= 2)} | previous best {pct('hits_prev_best', cond=lambda r: r['zones'] >= 2 and r['had_prev_best'])} | any {pct('hits_any', cond=lambda r: r['zones'] >= 2)}",
           f"PERSISTENCE: best move stays in the previous best move's zone {pct('hits_prev_best', cond=lambda r: r['had_prev_best'])}; in the zone of the move actually played {pct('hits_prev_played', cond=lambda r: r['had_prev_played'])}",
           f"  quiet best moves outside every zone (the trajectory model's share): {sum(1 for r in rows if r['kind'] == 'quiet' and not r['hits_any'])}/{sum(1 for r in rows if r['kind'] == 'quiet')}"]
    for title, key in (("by the best move's kind", "kind"), ("by phase", lambda r: "opening 60+" if r["phase"] >= 60 else "middlegame 40-59" if r["phase"] >= 40 else "late 20-39" if r["phase"] >= 20 else "ending <20")):
        groups = defaultdict(list)
        for r in rows:
            groups[key(r) if callable(key) else r[key]].append(r)
        out.append(f"{title:<22}{'n':>5}{'hottest':>9}{'largest':>9}{'random':>8}{'prev':>7}{'any':>6}")
        for gname, rs in sorted(groups.items(), key=lambda kv: -len(kv[1])):
            if len(rs) < 5:
                continue
            m = len(rs); pb = [r for r in rs if r["had_prev_best"]]
            out.append(f"{str(gname):<22}{m:>5}{sum(r['hits_hottest'] for r in rs) / m:>9.0%}{sum(r['hits_largest'] for r in rs) / m:>9.0%}{sum(r['hits_random'] for r in rs) / m:>8.0%}"
                       f"{(sum(r['hits_prev_best'] for r in pb) / len(pb)) if pb else 0:>7.0%}{sum(r['hits_any'] for r in rs) / m:>6.0%}")
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("pgn"); ap.add_argument("--games", type=int, default=60); ap.add_argument("--window", type=int, default=12)
    ap.add_argument("--depth", type=int, default=12); ap.add_argument("--seed", type=int, default=3); ap.add_argument("--json", default="")
    ap.add_argument("--shard", default=""); ap.add_argument("--cache", default="")
    a = ap.parse_args(argv)
    from .engine import Engine
    seed = a.seed
    if a.shard:
        i, k = (int(x) for x in a.shard.split("/")); seed = a.seed * 100 + i
    with Engine(depth=a.depth, cache_path=a.cache or None) as eng:
        rows, dt = run(a.pgn, eng, a.games, a.window, a.depth, seed)
    print(report(rows, dt), flush=True)
    if a.json:
        import json
        json.dump(rows, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
