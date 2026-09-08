"""mine.py -- learning sub-goals from the need trees (Kaveh 2026-09-07: "connect the rooks, develop, take the centre --
I don't want to program these, I want them learned").

The atoms are the rules'; what is learned is which recurring intermediate NEEDS are worth pursuing, and how much.
Over games: at every position, the need tree for every enemy target; every node becomes a TEMPLATE -- the need
described by roles and relations, never coordinates (kind, the mover's and the target's piece types, the square's
class: central / in the enemy camp / by the enemy king, the blocker's type) -- with its status; the root's outcome
is whether its target is taken within k plies of the engine's line from there. A template's value is the gain of
the roots it appears under, credited by completion; support and lift over executed-versus-not give the library.

The first number the pipeline reports is the honest one: of the engine's QUIET best moves (no capture, no check),
how many serve a sub-goal on the agenda at that moment -- the move is a step of a way in one of the agenda's need
trees, or the sub-goal is met once the move is played. That is the coverage of the quiet moves by the learned
vocabulary, before any learning: the agenda's own reach.

    python -m catspace.incubator.analyst.mine data/lichess/latest.pgn [more.pgn ...] [--every 2 --k 6 --depth 10]
"""
from __future__ import annotations

import argparse
import time
from collections import Counter, defaultdict
from dataclasses import dataclass

import chess
import chess.pgn

from . import rules as R
from .facts import value
from .memo import Memo
from .needs import Planner, Need, Line, agenda, frontier


# ---------------------------------------------------------------- templates: roles and relations, never coordinates
CENTRE = {chess.D4, chess.E4, chess.D5, chess.E5}


def square_class(board: chess.Board, sq: int, side: chess.Color) -> str:
    ek = board.king(not side)
    if ek is not None and chess.square_distance(sq, ek) <= 2:
        return "by-their-king"
    if sq in CENTRE:
        return "centre"
    r = chess.square_rank(sq)
    theirs = r >= 4 if side == chess.WHITE else r <= 3
    return "their-camp" if theirs else "own-camp"


def template(board: chess.Board, need: Need, side: chess.Color) -> tuple:
    pt = lambda sq: chess.piece_name(board.piece_at(sq).piece_type) if sq is not None and board.piece_at(sq) else "-"
    if need.kind in ("win", "defend", "remove"):
        t = need.against if need.kind == "remove" else need.square
        return (need.kind, pt(t), square_class(board, t, side))
    if need.kind == "reach":
        return ("reach", pt(need.piece), square_class(board, need.square, side))
    if need.kind == "safe":
        return ("safe", pt(need.piece), square_class(board, need.square, side))
    if need.kind in ("deflect", "block"):
        return (need.kind, pt(need.against), square_class(board, need.square, side))
    return (need.kind,)


def walk(line: Line):
    yield line
    for w in line.ways:
        for s in w.subs:
            yield from walk(s)


# ---------------------------------------------------------------- the mining
@dataclass
class Row:
    game: str
    ply: int
    root_target: int
    root_gain: float
    executed: bool
    nodes: list            # (template, status)


def executed_in_line(board: chess.Board, side: chess.Color, target: int, pv: tuple, k: int) -> bool:
    """Is the piece on target taken by `side` within k plies of the line? (It may move first; we follow it.)"""
    b = board.copy(stack=False); t = target
    for i, u in enumerate(pv[:k]):
        m = chess.Move.from_uci(u)
        if b.turn == side and m.to_square == t and b.is_capture(m):
            return True
        if m.from_square == t:
            t = m.to_square
        b.push(m)
    return False


def mine(pgn_paths: list[str], every: int = 2, k: int = 6, engine_depth: int = 10, budget: int = 30, max_positions: int = 0, max_games: int = 12, log=print):
    from .engine import Engine
    eng = Engine(depth=engine_depth)
    memo = Memo()
    rows: list[Row] = []
    quiet_total = quiet_served = 0
    served_by: Counter = Counter()
    t0 = time.perf_counter(); n_pos = 0
    games = []
    for path in pgn_paths:
        with open(path) as f:
            while len(games) < max_games:
                g = chess.pgn.read_game(f)
                if g is None:
                    break
                games.append((path, g))
    for gi, (path, game) in enumerate(games):
        board = game.board(); moves = list(game.mainline_moves())
        for ply, mv in enumerate(moves):
            if ply % every == 0 and not board.is_game_over() and not board.is_check():
                side = board.turn
                best = eng.best(board, depth=engine_depth)
                pv = best.pv if best else ()
                # the need trees, one per enemy target
                for t, tp in board.piece_map().items():
                    if tp.color == side or R.is_royal(board, t):
                        continue
                    pl = Planner(board, side, memo, budget=budget)
                    line = pl.plan(Need("win", t))
                    nodes = [(template(board, l.need, side), l.status) for l in walk(line)]
                    rows.append(Row(path.split("/")[-1], ply, t, value(board, t), executed_in_line(board, side, t, pv, k), nodes))
                # the quiet-move coverage: does the engine's best quiet move serve an agenda sub-goal?
                if best is not None:
                    bm = chess.Move.from_uci(best.move)
                    if not board.is_capture(bm) and not board.gives_check(bm):
                        quiet_total += 1
                        served = None
                        ag = agenda(board, side, memo, top=5, budget=budget)
                        for sg in ag:
                            pl = Planner(board, side, memo, budget=budget)
                            line = pl.plan(sg.need)
                            steps = {m for l in walk(line) for w in l.ways for m in w.moves[:1]}
                            if bm in steps:
                                served = ("step", sg); break
                            nb = board.copy(stack=False); nb.push(bm); nb.turn = side
                            if nb.is_valid():
                                pl2 = Planner(nb, side, memo, budget=budget)
                                if pl2.plan(sg.need).best is not None:
                                    served = ("met", sg); break
                        if served:
                            quiet_served += 1
                            served_by[(served[0], template(board, served[1].need, side))] += 1
                n_pos += 1
                if n_pos % 20 == 0:
                    log(f"[mine] {n_pos} positions, {len(rows)} trees, quiet {quiet_served}/{quiet_total}, {time.perf_counter()-t0:.0f}s", flush=True)
            board.push(mv)
            if max_positions and n_pos >= max_positions:
                break
        if max_positions and n_pos >= max_positions:
            break
    eng.close()
    return rows, (quiet_served, quiet_total, served_by), time.perf_counter() - t0, memo


def library(rows: list[Row], min_support: int = 5) -> list[tuple]:
    """Per template: support (trees it appears in), the executed rate of those trees against the base rate, the mean
    gain credited (root gain x executed). Lift = executed rate / base rate."""
    base = sum(1 for r in rows if r.executed) / max(1, len(rows))
    per = defaultdict(lambda: [0, 0, 0.0])                     # template -> [trees, executed trees, credited gain]
    for r in rows:
        seen = {tpl for tpl, st in r.nodes}
        for tpl in seen:
            per[tpl][0] += 1
            if r.executed:
                per[tpl][1] += 1; per[tpl][2] += r.root_gain
    out = []
    for tpl, (n, ex, g) in per.items():
        if n < min_support:
            continue
        rate = ex / n
        out.append((tpl, n, ex, rate, rate / base if base else 0.0, g / n))
    out.sort(key=lambda x: (-x[4], -x[1]))
    return out


def report(rows, quiet, dt, memo, top: int = 25) -> str:
    served, total, served_by = quiet
    out = [f"[mine] {len(rows)} need trees from {len({(r.game, r.ply) for r in rows})} positions in {dt:.0f}s | {memo.report()}"]
    base = sum(1 for r in rows if r.executed) / max(1, len(rows))
    out.append(f"trees whose target is taken within the line: {sum(1 for r in rows if r.executed)}/{len(rows)} ({base:.0%}); "
               f"met at the root: {sum(1 for r in rows if r.nodes and r.nodes[0][1] == 'met')}")
    out.append(f"QUIET best moves served by an agenda sub-goal: {served}/{total} ({served / total if total else 0:.0%})")
    for (how, tpl), n in served_by.most_common(10):
        out.append(f"   {n:3d}  {how:<4} {tpl}")
    out.append(f"{'template':<44}{'trees':>6}{'exec':>6}{'rate':>7}{'lift':>7}{'gain':>7}")
    for tpl, n, ex, rate, lift, g in library(rows)[:top]:
        out.append(f"{str(tpl):<44}{n:>6}{ex:>6}{rate:>7.0%}{lift:>7.2f}{g:>7.2f}")
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("pgns", nargs="+"); ap.add_argument("--every", type=int, default=2); ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--depth", type=int, default=10); ap.add_argument("--budget", type=int, default=30); ap.add_argument("--max-positions", type=int, default=0)
    ap.add_argument("--max-games", type=int, default=12)
    a = ap.parse_args(argv)
    rows, quiet, dt, memo = mine(a.pgns, a.every, a.k, a.depth, a.budget, a.max_positions, a.max_games)
    print(report(rows, quiet, dt, memo))


if __name__ == "__main__":
    main()
