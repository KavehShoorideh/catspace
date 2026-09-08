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
from .facts import value, material
from .memo import Memo
from .subgames import holdings, cost_of_move, mate_in_one
from .needs import ideas_for, agenda, Planner
from .regime import capture_gain as capture_gain_of
from .plans import plans_for, Goal, derive
from .goal_search import pursue, GoalSearch


def score_moves(board: chess.Board, memo: Memo | None = None, engine=None, budget: int = 120, engine_depth: int = 6, mate_budget: int = 150) -> dict[chess.Move, tuple[float, str]]:
    """move -> (score in pawns, the reason)."""
    memo = memo if memo is not None else Memo()
    side = board.turn
    out: dict[chess.Move, tuple[float, str]] = {}
    net_sources: set = set()                                    # moves whose score already accounts for the reply (no cost subtracted)
    def add(m, s, why, net=False):
        if m in out:
            if s > out[m][0]:
                out[m] = (s, why)
        else:
            out[m] = (s, why)
        if net:
            net_sources.add(m)
    # 1. the terminal goal: win the game by the rules' outcome (mate, in chess), proven by the search without a judge
    for m in mate_in_one(board, side):
        add(m, 1000.0, "mate", net=True)
    if not any(s >= 1000 for s, _ in out.values()):
        try:
            gs = GoalSearch(board, side, None, memo=memo, horizon=6, budget=mate_budget)
            r = gs.run()
            if r.proven and r.line:
                add(board.parse_san(r.line[0]), 1000.0 - len(r.line), f"forced win: {' '.join(r.line)}", net=True)
        except Exception:
            pass
    # 2. proven goals (the search, engine-judged inside)
    try:
        for r in pursue(board, side, memo=memo, engine=engine, budget=budget, horizon=6, max_goals=3, engine_depth=engine_depth, refinements=0):
            if r.proven and r.line:
                m = board.parse_san(r.line[0])
                if r.mates:
                    add(m, 1000.0 - len(r.line), f"forced win: {' '.join(r.line)}", net=True)
                else:
                    add(m, 10.0 + max(r.net, 0.5), f"proven: {r.goal}" + (f" ({r.net:+g})" if r.net else ""), net=True)
    except Exception:
        pass
    # 3. captures priced by the ledger (the mover's own exchange, attackers' test included)
    for m in board.legal_moves:
        if board.is_capture(m):
            g = capture_gain_of(board, m, side, memo)
            if g > 0:
                add(m, g, "wins the exchange", net=True)
    # 3b. the SURE gain one reply deep: after our move, whatever they reply, what the ledger says we then win less
    # what the reply took. Forks, discovered attacks and checks that carry a threat fall out of this with no pattern
    # named: a check is only a move with few replies.
    for m in attacking_moves(board):
        g, why = sure_gain(board, m, memo)
        if g > 0:
            add(m, g, why, net=True)
    # 3b'. combinations: the same idea two of our forcing moves deep (alpha-beta, the ledger at the leaves)
    for m in forcing_moves(board):
        g, why = forcing_gain(board, m, memo, depth=2)
        if g > 0 and g > out.get(m, (0.0, ""))[0]:
            add(m, g, why, net=True)
    # 3c. promotion: a runner that wins its race, or the move that catches theirs
    for m, g, why in promotion_scores(board, memo):
        add(m, g, why, net=True)
    # 3d. a piece of ours that falls, saved by taking its attacker
    for m, g, why in rescue_captures(board, memo):
        add(m, g, why, net=True)
    # 4. ideas and plans that win their race
    try:
        for m, reason, line in ideas_for(board, side, memo, None, top=6):
            if line.best is not None and line.race == "wins" and line.best.gain - line.cost > 0:
                hops = max(1, len(line.steps()) + 1)               # the idea's own move plus its line: a gain k moves away is worth less now
                add(m, (line.best.gain - line.cost) / hops, f"idea: {line.need.text(line.best.after or board)} in {hops}")
    except Exception:
        pass
    try:
        for pl in plans_for(board, side, memo=memo, top=5):
            if pl.steps and pl.steps[0].move != chess.Move.null() and pl.steps[0].move in board.legal_moves:
                if pl.race == "urgent":
                    continue                                          # saves are the additive term below
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
    # 6. what the move SAVES: pieces of ours that fell before it and no longer fall after it (additive: a move that
    # both saves the rook and checks with a threat beats one that only saves it)
    falling = {sq: value(board, sq) for sq, d in all_defenses(board, memo=memo).items() if d.falls and board.piece_at(sq).color == side}
    if falling:
        for m in list(board.legal_moves):
            nb = board.copy(stack=False); nb.push(m)
            saved = 0.0
            for sq, v in falling.items():
                s2 = m.to_square if m.from_square == sq else sq
                if nb.piece_at(s2) is None or nb.piece_at(s2).color != side:
                    continue
                d2 = defense(nb, s2, memo=memo)
                if d2 is None or not d2.falls:
                    saved += v
            if saved > 0:
                s0, why0 = out.get(m, (0.0, ""))
                out[m] = (s0 + saved, (why0 + "; " if why0 else "") + f"saves {saved:g}")
    # 7. costs: what the move gives up -- for the static sources only; a sure gain, a proof or an exchange already
    # priced the opponent's best reply, charging the holdings again would count the same loss twice
    held = holdings(board, memo)
    for m in list(board.legal_moves):
        if m in net_sources:
            continue
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


def attacking_moves(board: chess.Board) -> list[chess.Move]:
    """Moves that create an attack on an enemy piece, give check, or capture: the only ones a one-reply gain can
    come from (a quiet move that attacks nothing is priced by the plans, not here)."""
    side = board.turn; out = []
    for m in board.legal_moves:
        if board.is_capture(m) or board.gives_check(m):
            out.append(m); continue
        nb = board.copy(stack=False); nb.push(m)
        if any(nb.piece_at(t) and nb.piece_at(t).color != side for t in nb.attacks(m.to_square)):
            out.append(m)
    return out


def sure_gain(board: chess.Board, m: chess.Move, memo=None) -> tuple[float, str]:
    """min over their replies of (our best priced capture after the reply) - (what the reply took): the gain the
    move guarantees one reply deep, by the ledger. The move's own capture counts first."""
    from .regime import _best_capture
    side = board.turn
    b = board.copy(stack=False)
    first = capture_gain_of(b, m, side, memo) if b.is_capture(m) else 0.0
    nb = b.copy(stack=False); nb.push(m)
    if nb.is_game_over():
        oc = nb.outcome()
        return (1000.0, "wins by the rules") if oc and oc.winner == side else (0.0, "")
    worst = None; worst_reply = None
    pieces = set(chess.scan_forward(nb.occupied))
    for r in nb.legal_moves:
        took = capture_gain_of(nb, r, not side, memo) if nb.is_capture(r) else 0.0
        if took < 0:
            took = 0.0                                          # a losing capture: they would not
        after = nb.copy(stack=False); after.push(r)
        ours = 0.0
        if not after.is_game_over():
            ours = _best_capture(after, side, set(chess.scan_forward(after.occupied)), memo)
        elif after.outcome() and after.outcome().winner == side:
            ours = 1000.0
        net = first + ours - took
        if worst is None or net < worst:
            worst, worst_reply = net, r
    if worst is None:
        return 0.0, ""
    why = f"sure gain {worst:g} (their best reply {nb.san(worst_reply)})" if worst > 0 else ""
    return max(0.0, worst), why


def forcing_moves(board: chess.Board, min_attack: float = 3.0) -> list[chess.Move]:
    """Captures, checks, and moves that attack a piece worth min_attack or more: the moves a combination is made of.
    Captures first, then checks (the ordering is what makes the pruning bite)."""
    side = board.turn; caps, checks, attacks = [], [], []
    for m in board.legal_moves:
        if board.is_capture(m):
            caps.append(m)
        elif board.gives_check(m):
            checks.append(m)
        else:
            nb = board.copy(stack=False); nb.push(m)
            if any(nb.piece_at(t) and nb.piece_at(t).color != side and value(nb, t) >= min_attack for t in nb.attacks(m.to_square)):
                attacks.append(m)
    caps.sort(key=lambda m: -value(board, m.to_square))
    return caps + checks + attacks


def forcing_gain(board: chess.Board, m: chess.Move, memo=None, depth: int = 2, node_cap: int = 800) -> tuple[float, str]:
    """A small alpha-beta over COMBINATIONS: our forcing moves, ALL their replies, the ledger's best capture at the
    leaves (quiescence), material tracked along the way. Depth 2 = our move, their reply, our forcing move, their
    reply, the ledger. Generic: no pattern, only the rules' moves and the ledger's prices. Returns the guaranteed
    gain of m and the line that shows it."""
    from .regime import _best_capture
    side = board.turn
    base = material(board, side) - material(board, not side)
    nodes = [0]
    def stand(b):                                               # the leaf: what we are up, plus what the ledger says we take next
        n = (material(b, side) - material(b, not side)) - base
        if b.turn == side and not b.is_check():
            n += _best_capture(b, side, set(chess.scan_forward(b.occupied)), memo)
        return n
    def ours(b, d, alpha, beta):
        nodes[0] += 1
        if b.is_game_over():
            oc = b.outcome(); return (1000.0 if oc and oc.winner == side else 0.0), []
        best, best_line = stand(b), []                          # we may stop here
        if d == 0 or nodes[0] > node_cap or best >= beta:
            return best, best_line
        for f in forcing_moves(b):
            nb = b.copy(stack=False); nb.push(f)
            v, line = theirs(nb, d, max(alpha, best), beta)
            if v > best:
                best, best_line = v, [b.san(f)] + line
                if best >= beta:
                    break
        return best, best_line
    def theirs(b, d, alpha, beta):
        nodes[0] += 1
        if b.is_game_over():
            oc = b.outcome(); return (1000.0 if oc and oc.winner == side else 0.0), []
        worst, worst_line = None, []
        replies = sorted(b.legal_moves, key=lambda r: -(value(b, r.to_square) if b.is_capture(r) else 0))
        for r in replies:
            nb = b.copy(stack=False); nb.push(r)
            v, line = ours(nb, d - 1, alpha, worst if worst is not None else beta)
            if worst is None or v < worst:
                worst, worst_line = v, [b.san(r)] + line
                if worst <= alpha:
                    break
            if nodes[0] > node_cap:
                break
        return (worst if worst is not None else 0.0), worst_line
    nb = board.copy(stack=False); nb.push(m)
    v, line = theirs(nb, depth, -1e9, 1e9)
    return max(0.0, v), (f"combination +{v:g}: {board.san(m)} {' '.join(line)}" if v > 0 else "")


def promotion_scores(board: chess.Board, memo=None) -> list[tuple[chess.Move, float, str]]:
    """Runners: a pawn push after which the race is won; catchers: a move after which their winning race is lost."""
    from .regime import race, pawn_path, PROMOTION_VALUE
    side = board.turn; out = []
    theirs = [sq for sq, p in board.piece_map().items() if p.color != side and p.piece_type == chess.PAWN and pawn_path(board, sq)]
    losing = [sq for sq in theirs if (rr := race(board, sq, memo)) is not None and rr.promotes]
    for m in board.legal_moves:
        p = board.piece_at(m.from_square)
        nb = board.copy(stack=False); nb.push(m)
        if p.piece_type == chess.PAWN and not board.is_capture(m):
            if m.promotion:
                out.append((m, PROMOTION_VALUE, "promotes")); continue
            r = race(nb, m.to_square, memo)
            if r is not None and r.promotes:
                out.append((m, PROMOTION_VALUE / max(1, r.steps), f"the runner wins its race ({r.steps} steps)"))
        if losing:
            still = [sq for sq in losing if nb.piece_at(sq) and (rr := race(nb, sq, memo)) is not None and rr.promotes]
            if len(still) < len(losing):
                out.append((m, PROMOTION_VALUE / 2, "catches the runner"))
    return out


def rescue_captures(board: chess.Board, memo=None) -> list[tuple[chess.Move, float, str]]:
    """A piece of ours that falls, saved by taking one of its attackers that count (the capture must not lose)."""
    side = board.turn; out = []
    falling = {sq: d for sq, d in all_defenses(board, memo=memo).items() if d.falls and board.piece_at(sq).color == side}
    for sq, d in falling.items():
        for a in d.effective_attackers:
            for m in board.legal_moves:
                if m.to_square == a and board.is_capture(m):
                    g = capture_gain_of(board, m, side, memo)
                    if g < 0:
                        continue
                    nb = board.copy(stack=False); nb.push(m); nb.turn = side
                    d2 = defense(nb, sq, memo=memo) if nb.piece_at(sq) else None
                    if d2 is None or not d2.falls:
                        out.append((m, max(0.0, g), f"takes the attacker of the {chess.piece_name(board.piece_at(sq).piece_type)} on {chess.square_name(sq)}"))
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
