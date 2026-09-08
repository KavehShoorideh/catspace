"""rank.py -- rank the legal moves by the REASONING layers alone, and measure the agreement with Stockfish
(Kaveh 2026-09-07: "take a hundred positions, rank the moves with our analysis, how well does it agree?").

The score of a move is what the reasoning says it does, in pawns: mate in one; the first move of a goal the
search proves (its gain; the engine is the judge inside the proof, as designed); a capture priced by the ledger
with the attackers' test; the first step of an idea or a plan that wins its race (its net); a move that saves a
piece of ours that falls (its value); a move that serves an agenda sub-goal (its worth, scaled down); minus the
opportunity cost of the move (what the moving piece held) and the piece itself when it falls where it lands.
Moves the reasoning says nothing about score 0. Ties keep the generation order: no engine, no Maia, no hidden
tie-break. The measurement reports top-1 and top-3 agreement with the engine's best move, the share of positions
where the engine's move has ANY reason at all, split by the kind of move the engine chose, against Maia.

    python -m catspace.incubator.analyst.rank data/lichess/huvaxshtra_all.pgn --positions 100 [--every 7 --depth 12]
"""
from __future__ import annotations

import argparse
import random
import time
from collections import Counter, defaultdict

import chess
import chess.pgn

from . import rules as R
from .defense import defense, all_defenses
from .facts import value
from .memo import Memo
from .subgames import holdings, cost_of_move, mate_in_one
from .needs import ideas_for, agenda, Planner
from .regime import capture_gain as capture_gain_of
from .plans import plans_for, Goal, derive
from .goal_search import pursue


def score_moves(board: chess.Board, memo: Memo | None = None, engine=None, budget: int = 120, engine_depth: int = 6) -> dict[chess.Move, tuple[float, str]]:
    """move -> (score in pawns, the reason)."""
    memo = memo if memo is not None else Memo()
    side = board.turn
    out: dict[chess.Move, tuple[float, str]] = {}
    def add(m, s, why):
        if m in out:
            if s > out[m][0]:
                out[m] = (s, why)
        else:
            out[m] = (s, why)
    # 1. mate in one
    for m in mate_in_one(board, side):
        add(m, 1000.0, "mate")
    # 2. proven goals (the search, engine-judged inside)
    try:
        for r in pursue(board, side, memo=memo, engine=engine, budget=budget, horizon=6, max_goals=3, engine_depth=engine_depth, refinements=0):
            if r.proven and r.line:
                m = board.parse_san(r.line[0])
                add(m, 10.0 + r_value(board, r), f"proven: {r.goal}")
    except Exception:
        pass
    # 3. captures priced by the ledger (the mover's own exchange, attackers' test included)
    for m in board.legal_moves:
        if board.is_capture(m):
            g = capture_gain_of(board, m, side, memo)
            if g > 0:
                add(m, g, "wins the exchange")
    # 4. ideas and plans that win their race
    try:
        for m, reason, line in ideas_for(board, side, memo, None, top=6):
            if line.best is not None and line.race == "wins" and line.best.gain - line.cost > 0:
                add(m, line.best.gain - line.cost, f"idea: {line.need.text(line.best.after or board)}")
    except Exception:
        pass
    try:
        for pl in plans_for(board, side, memo=memo, top=5):
            if pl.steps and pl.steps[0].move != chess.Move.null() and pl.steps[0].move in board.legal_moves:
                if pl.race == "urgent":
                    add(pl.steps[0].move, value(board, pl.goal.square), f"saves {pl.goal.text(board)}")
                elif pl.race == "wins" and pl.cost < 100:
                    add(pl.steps[0].move, max(0.5, value(board, pl.goal.square) - pl.cost) / max(1, pl.hops), f"plan: {pl.goal.text(board)}")
    except Exception:
        pass
    # 5. the agenda: sub-goals served, scaled down (they are preparation, not gains)
    try:
        for sg in agenda(board, side, memo, top=4, budget=30):
            pl = Planner(board, side, memo, budget=30)
            line = pl.plan(sg.need)
            for l in _walk(line):
                for w in l.ways:
                    if w.moves and w.moves[0] in board.legal_moves:
                        add(w.moves[0], 0.25 * sg.worth, f"toward: {sg.text}")
    except Exception:
        pass
    # 6. costs: what the move gives up
    held = holdings(board, memo)
    for m in list(board.legal_moves):
        s, why = out.get(m, (0.0, ""))
        c = sum(h.loss for h in cost_of_move(board, m, held, memo))
        nb = board.copy(stack=False); nb.push(m)
        p = board.piece_at(m.from_square)
        if not board.is_capture(m) and p is not None and not R.is_royal(board, m.from_square):
            d = defense(nb, m.to_square, memo=memo)
            if d is not None and d.falls:
                c += value(board, m.from_square)
        if c:
            out[m] = (s - c, (why + "; " if why else "") + f"gives up {c:g}")
    return out


def r_value(board: chess.Board, r) -> float:
    try:
        sq = chess.parse_square(r.goal.split()[-1])
        return value(board, sq)
    except Exception:
        return 1.0


def _walk(line):
    yield line
    for w in line.ways:
        for s in w.subs:
            yield from _walk(s)


def ranking(board: chess.Board, memo=None, engine=None, **kw) -> list[tuple[chess.Move, float, str]]:
    sc = score_moves(board, memo, engine, **kw)
    legal = list(board.legal_moves)
    return sorted(((m, *sc.get(m, (0.0, ""))) for m in legal), key=lambda x: -x[1])


# ---------------------------------------------------------------- the measurement
def move_kind(board: chess.Board, m: chess.Move) -> str:
    if board.is_capture(m):
        return "capture"
    if board.gives_check(m):
        return "check"
    return "quiet"


def measure(pgn_paths: list[str], positions: int = 100, every: int = 7, engine_depth: int = 12, judge_depth: int = 6, seed: int = 3, maia_rating: int = 1500, log=print):
    from .engine import Engine
    eng = Engine(depth=engine_depth)
    maia = None
    try:
        from .maia import Maia
        maia = Maia(maia_rating)
    except Exception as ex:
        log(f"[rank] no Maia: {ex}")
    random.seed(seed)
    boards = []
    for path in pgn_paths:
        with open(path) as f:
            while True:
                g = chess.pgn.read_game(f)
                if g is None:
                    break
                b = g.board()
                for i, mv in enumerate(g.mainline_moves()):
                    if i >= 10 and i % every == 0 and not b.is_game_over() and not b.is_check():
                        boards.append(b.copy(stack=False))
                    b.push(mv)
                if len(boards) > 20 * positions:
                    break
    random.shuffle(boards)
    boards = boards[:positions]
    memo = Memo()
    stats = defaultdict(Counter)
    t0 = time.perf_counter()
    for n, b in enumerate(boards, 1):
        best = eng.best(b, depth=engine_depth)
        if best is None:
            continue
        bm = chess.Move.from_uci(best.move)
        kind = move_kind(b, bm)
        rk = ranking(b, memo, eng, budget=120, engine_depth=judge_depth)
        order = [m for m, s, w in rk]
        scored = [m for m, s, w in rk if s > 0]
        top = rk[0][1]
        st = stats[kind]; st["n"] += 1
        st["top1"] += order[0] == bm and top > 0
        st["top3"] += bm in [m for m in order[:3] if True] and top > 0
        st["reasoned"] += any(m == bm and s > 0 for m, s, w in rk)
        st["silent"] += top <= 0                                   # the reasoning had nothing positive to say
        if maia is not None:
            try:
                pol = maia.policy(b)
                mb = max(pol.items(), key=lambda kv: kv[1])[0]
                st["maia1"] += (mb == bm.uci())
            except Exception:
                pass
        if n % 10 == 0:
            log(f"[rank] {n}/{len(boards)} positions, {time.perf_counter()-t0:.0f}s, " + ", ".join(f"{k}: top1 {v['top1']}/{v['n']}" for k, v in stats.items()), flush=True)
    eng.close()
    if maia is not None:
        try:
            maia.close()
        except Exception:
            pass
    return stats, time.perf_counter() - t0


def report(stats, dt) -> str:
    rows = list(stats.items()) + [("all", sum(stats.values(), Counter()))]
    out = [f"[rank] {sum(v['n'] for k, v in stats.items())} positions in {dt:.0f}s"]
    out.append(f"{'engine move':<12}{'n':>5}{'top-1':>8}{'top-3':>8}{'reasoned':>10}{'silent':>8}{'maia top-1':>12}")
    for k, v in rows:
        n = v["n"] or 1
        out.append(f"{k:<12}{v['n']:>5}{v['top1']/n:>8.0%}{v['top3']/n:>8.0%}{v['reasoned']/n:>10.0%}{v['silent']/n:>8.0%}{v['maia1']/n:>12.0%}")
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("pgns", nargs="+"); ap.add_argument("--positions", type=int, default=100); ap.add_argument("--every", type=int, default=7)
    ap.add_argument("--depth", type=int, default=12); ap.add_argument("--judge", type=int, default=6); ap.add_argument("--fen", default="")
    a = ap.parse_args(argv)
    if a.fen:
        from .engine import Engine
        b = chess.Board(a.fen)
        with Engine(depth=a.depth) as eng:
            for m, s, w in ranking(b, Memo(), eng, budget=120, engine_depth=a.judge)[:8]:
                print(f"{b.san(m):<8}{s:>7.2f}  {w}")
            print("engine:", b.san(chess.Move.from_uci(eng.best(b, depth=a.depth).move)))
        return
    stats, dt = measure(a.pgns, a.positions, a.every, a.depth, a.judge)
    print(report(stats, dt), flush=True)


if __name__ == "__main__":
    main()
