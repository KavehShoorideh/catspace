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
import math
import random
import time
from collections import Counter, defaultdict

import chess
import chess.pgn

from . import rules as R
from .defense import defense, all_defenses
from .facts import value, material
from .memo import changed_squares
from .memo import Memo
from .subgames import holdings, cost_of_move, mate_in_one
from .needs import ideas_for, agenda, Planner
from .regime import capture_gain as capture_gain_of
from .plans import plans_for, Goal, derive
from .goal_search import pursue, GoalSearch


TRUST = 0.4                      # an unverified promise (an idea, a plan, an agenda step) counts this fraction of its face value


def score_moves(board: chess.Board, memo: Memo | None = None, engine=None, budget: int = 120, engine_depth: int = 6, mate_budget: int = 1500,
                mate_horizon: int = 10, combo_top: int = 8, combo_cap: int = 1500, zone_first: bool = False,
                threat_gate: bool = True, entropy: bool = True, big: float | None = None, w_width: float = 0.1, w_threat2: float = 0.5, top_k: int = 6) -> dict[chess.Move, tuple[float, str]]:
    """move -> (score in pawns, the reason). One scale for everything: the material net one reply deep (the BASE of
    every legal move: what the ledger says after their best reply, negative when the move hangs something), lifted
    by deeper views of the same quantity (a two-move combination, a proof, a forced win), plus the planner's
    promises (ideas, plans, the agenda) on top, discounted. No engine and no Maia in the ranking."""
    memo = memo if memo is not None else Memo()
    side = board.turn
    legal = list(board.legal_moves)
    base: dict[chess.Move, tuple[float, str]] = {}
    widths: dict[chess.Move, int] = {}
    for m in legal:                                             # the base: every move against their best reply, by the ledger
        v, why = sure_gain(board, m, memo, signed=True)
        base[m] = (v, why); widths[m] = width(LAST_NETS.get(m.uci(), []))
    out: dict[chess.Move, tuple[float, str]] = dict(base)
    # THE THREAT (Kaveh 2026-09-15): the loss the cheap observer sees if we do nothing -- their best free-move gain
    # by the ledger; its LEVEL is that magnitude against the scale's smallest piece. Above it the search is
    # constrained to the RESPONSE SET: the moves after which the loss is gone (the base sees it: a move that
    # ignores a hanging piece scores its loss) or that counter with at least as much. Nothing chess-specific:
    # a null move, a value, a scale.
    level = 0.0; responders = set(legal); threat_text = ""
    if threat_gate:
        from .threats import free_move_threats
        from . import rules as R
        big = big if big is not None else min(v for v in (R.scale_of(board).of_type(pt) for pt in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)) if v > 0)
        theirs = free_move_threats(board, not side, memo, top=3)
        if theirs:
            level = theirs[0][1]; threat_text = theirs[0][2]
        if level >= big:
            # a responder keeps the loss below the threat less a pawn, one reply deep: the base already nets what
            # they take against what we take back, so meeting the threat and countering it are one condition
            responders = {m for m in legal if base[m][0] > -min(level, 1000.0) + 1.0}
            if not responders:
                responders = set(legal)
    def lift(m, v, why):
        if v > out[m][0]:
            out[m] = (v, why)
    def bonus(m, v, why):
        s0, why0 = out[m]
        out[m] = (s0 + v, (why0 + "; " if why0 and not why0.startswith("hangs") else "") + why)
    # deeper views of the count: mate in one, the terminal goal, combinations, proofs
    for m in mate_in_one(board, side):
        lift(m, 1000.0, "mate")
    if not any(v >= 1000 for v, _ in out.values()):
        try:
            gs = GoalSearch(board, side, None, memo=memo, horizon=mate_horizon, budget=mate_budget)   # mate in five, ~1 s at most: cheap, no judge
            r = gs.run()
            if r.proven and r.line:
                lift(board.parse_san(r.line[0]), mate_score(r, entropy), f"forced win: {' '.join(r.line)}" + (f" (proof {r.proof} nodes)" if entropy else ""))
        except Exception:
            pass
    if any(v >= 900 for v, _ in out.values()):
        return out                                              # a forced win by the rules: nothing else to weigh
    # combinations: the forcing moves, the most promising first by their one-reply base, the deep search for the top few
    # the gate never withholds a FORCING move from the deep search: a check or a capture may be the counter the
    # cheap observer cannot price (puzzles, 2026-09-15: six lost keys were sacrificial captures outside the set)
    cands = [m for m in forcing_moves(board, memo) if (base[m][0] >= -3.0 or board.gives_check(m))]   # a quiet sacrifice of more than a piece is not a combination we look for
    cands = sorted(cands, key=lambda m: -base[m][0])[:combo_top]
    tt: dict = {}
    for m in cands:
        g, why = forcing_gain(board, m, memo, depth=3, node_cap=combo_cap, tt=tt)
        if g > 0:
            lift(m, g, why)
    try:
        zones = zones_of(board, memo, base) if zone_first else None
        for r in pursue(board, side, memo=memo, engine=engine, budget=budget, horizon=6, max_goals=3, engine_depth=engine_depth, refinements=0, zones=zones):
            if r.proven and r.line:
                m = board.parse_san(r.line[0])
                if r.mates:
                    lift(m, mate_score(r, entropy), f"forced win: {' '.join(r.line)}")
                else:
                    lift(m, max(r.net, 0.5) + 0.02, f"proven: {r.goal}" + (f" ({r.net:+g})" if r.net else ""))
    except Exception:
        pass
    for m, g, why in promotion_scores(board, memo):
        lift(m, g, why)
    if threat_gate and level >= big:
        for m in legal:
            if m not in responders and out[m][0] < 900 and not (board.is_capture(m) or board.gives_check(m)):
                out[m] = (min(out[m][0], -level + 0.5), (out[m][1] + "; " if out[m][1] else "") + f"ignores the threat ({level:g})")
    # the second-level threat and the opponent's freedom, on the top candidates only (a ledger per child)
    if entropy or threat_gate:
        from .regime import _best_capture
        top = sorted(legal, key=lambda m: -out[m][0])[:top_k]
        for m in top:
            if out[m][0] >= 900 or out[m][0] != base[m][0] or board.gives_check(m):
                continue                                                # a deeper view (a proof, a combination) already priced the replies:
                                                                        # the opponent gets no free move inside a forced line (puzzles, 2026-09-15)
            nb = board.copy(stack=False); nb.push(m)
            if nb.is_game_over():
                continue
            if threat_gate and w_threat2 > 0:
                nb2 = nb.copy(stack=False); nb2.turn = not side
                if nb2.is_valid():
                    resid = _best_capture(nb2, not side, set(chess.scan_forward(nb2.occupied)), memo)   # what they take if we pass again
                    if resid > 0.5:
                        bonus(m, -w_threat2 * min(resid, 9.0), f"walks into a threat ({resid:g})")
            if entropy and w_width > 0 and widths.get(m, 0) > 1:
                bonus(m, -w_width * math.log(widths[m]), f"width {widths[m]}")
    # the planner's promises, on top of the count
    try:
        for m, reason, line in ideas_for(board, side, memo, None, top=6):
            if line.best is not None and line.race == "wins" and line.best.gain - line.cost > 0:
                hops = max(1, len(line.steps()) + 1)
                bonus(m, TRUST * (line.best.gain - line.cost) / hops, f"idea: {line.need.text(line.best.after or board)} in {hops}")
    except Exception:
        pass
    try:
        for pl in plans_for(board, side, memo=memo, top=5):
            if pl.steps and pl.steps[0].move != chess.Move.null() and pl.steps[0].move in board.legal_moves:
                if pl.steps[0].move.to_square == pl.goal.square:
                    continue                                    # the capture itself: the base already priced it against the reply
                if pl.race == "wins" and pl.cost < 100:
                    bonus(pl.steps[0].move, TRUST * max(0.5, value(board, pl.goal.square) - pl.cost) / max(1, pl.hops), f"plan: {pl.goal.text(board)}")
    except Exception:
        pass
    try:
        for sg in agenda(board, side, memo, top=4, budget=30):
            pl = Planner(board, side, memo, budget=30)
            line = pl.plan(sg.need)
            for l in _walk(line):
                for w in l.ways:
                    if w.moves and w.moves[0] in board.legal_moves:
                        bonus(w.moves[0], TRUST * 0.25 * sg.worth, f"toward: {sg.text}")
    except Exception:
        pass
    return out


def zones_of(board: chess.Board, memo, base: dict) -> list[tuple[set, float]]:
    """The sub-games (components of the lemma graph) with a temperature: the best sure gain of ours touching the
    zone (from the base already computed) or of theirs with a free move. Hottest first."""
    from .subgames import static_lemmas, mate_lemmas, components
    from .threats import free_move_threats
    lemmas = static_lemmas(board, memo) + mate_lemmas(board)
    zones = [set().union(*(l.participants for l in c)) for c in components(lemmas)]
    temps = [0.0] * len(zones)
    def touch(m, gain):
        for zi, z in enumerate(zones):
            if m.from_square in z or m.to_square in z:
                temps[zi] = max(temps[zi], gain); break
    for m, (v, _) in base.items():
        if v > 0:
            touch(m, v)
    for m, g, _ in free_move_threats(board, not board.turn, memo, top=8):
        touch(m, g)
    order = sorted(range(len(zones)), key=lambda i: (-temps[i], -len(zones[i])))
    return [(zones[i], temps[i]) for i in order]


def attacking_moves(board: chess.Board) -> list[chess.Move]:
    """Moves that create an attack on an enemy piece, give check, or capture: the only ones a one-reply gain can
    come from (a quiet move that attacks nothing is priced by the plans, not here)."""
    side = board.turn; out = []
    for m in board.legal_moves:
        if board.is_capture(m) or board.gives_check(m) or m.promotion:
            out.append(m); continue
        nb = board.copy(stack=False); nb.push(m)
        if any(nb.piece_at(t) and nb.piece_at(t).color != side for t in nb.attacks(m.to_square)):
            out.append(m)
    return out


def leaf_value(b: chess.Board, side: chess.Color, net: float, memo=None, base=None, changed: int = 0) -> float:
    """Our node at the bottom of a combination: the count so far plus the ledger's best capture, unless we are in
    check -- then the best evasion against their best capture after it (no standing pat in check). `base` is a
    ledger computed before the last move with the footprint of every entry; only the entries whose footprint the
    move touched (`changed`) are recomputed -- exact by the store's own validity rule."""
    from .regime import _best_capture
    if b.is_game_over():
        oc = b.outcome(); return 1000.0 if oc and oc.winner == side else (-1000.0 if oc and oc.winner is not None else 0.0)
    if not b.is_check():
        return net + _best_capture(b, side, set(chess.scan_forward(b.occupied)), memo, base=base, changed=changed)
    best = None
    for ev in b.legal_moves:
        nb = b.copy(stack=False); nb.push(ev)
        v = net + (value(b, ev.to_square) if b.is_capture(ev) else 0.0)
        if nb.is_game_over():
            oc = nb.outcome(); v = 1000.0 if oc and oc.winner == side else (-1000.0 if oc and oc.winner is not None else v)
        else:
            v -= _best_capture(nb, not side, set(chess.scan_forward(nb.occupied)), memo)
        if best is None or v > best:
            best = v
    return best if best is not None else net


LAST_NETS: dict = {}                     # move uci -> the nets over the opponent's replies from the last sure_gain of it (this process)


def mate_score(r, entropy: bool) -> float:
    """A forced win's score: by the LINE's length (the fastest mate), or, with the low-entropy preference, by the
    PROOF's size -- the calculation it costs and the replies it must cover -- with the length as the tie-break."""
    if entropy and r.proof:
        return 1000.0 - 2.0 * math.log2(1 + r.proof) - 0.01 * len(r.line)
    return 1000.0 - len(r.line)


def width(nets: list, margin: float = 1.0) -> int:
    """The opponent's FREEDOM after our move: how many of their replies hold us within `margin` of the minimum.
    One is a forced reply; many is a position they can play as they like. The generic 'dependence on the reply'."""
    if not nets:
        return 0
    lo = min(nets)
    return sum(1 for x in nets if x <= lo + margin)


def base_ledger(board: chess.Board, side: chess.Color) -> dict | None:
    """The defence of every enemy piece with `side` to move and the footprint each read, for incremental reuse
    across the opponent's replies (None when no static ledger exists: the enemy is in check)."""
    from .defense import _defense
    from .memo import Reads
    b = board.copy(stack=False); b.turn = side
    if not b.is_valid() or b.is_check():
        return None
    out = {}
    for s in chess.scan_forward(b.occupied):
        p = b.piece_at(s)
        if p.color == side or R.is_royal(b, s):
            continue
        reads = Reads(); d = _defense(b, s, reads)
        out[s] = (d, reads.mask)
    return out


def sure_gain(board: chess.Board, m: chess.Move, memo=None, signed: bool = False) -> tuple[float, str]:
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
    base = base_ledger(nb, side)
    nets = []
    for r in nb.legal_moves:
        took = capture_gain_of(nb, r, not side, memo) if nb.is_capture(r) else 0.0
        if took < 0:
            took = 0.0                                          # a losing capture: they would not
        after = nb.copy(stack=False); after.push(r)
        net = leaf_value(after, side, first - took, memo, base=base, changed=changed_squares(nb, r))
        nets.append(net)
        if worst is None or net < worst:
            worst, worst_reply = net, r
    LAST_NETS[m.uci()] = nets                                    # the reply spread, read by width() right after the call
    if worst is None:
        return 0.0, ""
    if worst > 0:
        why = f"sure gain {worst:g} (their best reply {nb.san(worst_reply)})"
    elif worst < 0:
        why = f"hangs {-worst:g} ({nb.san(worst_reply)})"
    else:
        why = ""
    return (worst if signed else max(0.0, worst)), why


def forcing_moves(board: chess.Board, memo=None, min_attack: float = 3.0) -> list[chess.Move]:
    """The moves a combination is made of, ORDERED BY THE REASONING: mates first; captures by what the ledger says
    the exchange wins (the attackers' test included); checks by how few replies they leave; then moves that
    attack a piece worth min_attack or more, by the value attacked. The ordering is the guidance: alpha-beta
    prunes in proportion to it, and nothing of ours outside these moves is searched (the opponent's replies stay
    complete)."""
    side = board.turn; caps, checks, attacks = [], [], []
    for m in board.legal_moves:
        if board.is_capture(m):
            caps.append((-capture_gain_of(board, m, side, memo), -value(board, m.to_square), m))
        elif board.gives_check(m):
            nb = board.copy(stack=False); nb.push(m)
            checks.append((nb.legal_moves.count(), m))
        else:
            nb = board.copy(stack=False); nb.push(m)
            hit = max([value(nb, t) for t in nb.attacks(m.to_square) if nb.piece_at(t) and nb.piece_at(t).color != side] + [0.0])
            if hit >= min_attack:
                attacks.append((-hit, m))
    caps.sort(key=lambda x: (x[0], x[1])); checks.sort(key=lambda x: x[0]); attacks.sort(key=lambda x: x[0])
    return [m for _, _, m in caps] + [m for _, m in checks] + [m for _, m in attacks]


def reply_order(board: chess.Board) -> list[chess.Move]:
    """Their replies, all of them, best-looking first: captures by value taken, then checks, then the rest."""
    caps, checks, rest = [], [], []
    for r in board.legal_moves:
        if board.is_capture(r):
            caps.append((-value(board, r.to_square), r))
        elif board.gives_check(r):
            checks.append(r)
        else:
            rest.append(r)
    caps.sort(key=lambda x: x[0])
    return [r for _, r in caps] + checks + rest


def forcing_gain(board: chess.Board, m: chess.Move, memo=None, depth: int = 3, node_cap: int = 2500, tt: dict | None = None) -> tuple[float, str]:
    """A small alpha-beta over COMBINATIONS, guided by the reasoning: our forcing moves in the ledger's order, ALL
    their replies, the ledger's best capture at the leaves as the quiescence (no standing pat in check), material
    tracked, a transposition table, and ITERATIVE DEEPENING one, two, three of our moves deep with the node cap as
    the budget -- the deepest completed depth is the answer. Generic: the rules' moves and the ledger's prices,
    no pattern. Returns the guaranteed gain of m and the line that shows it."""
    side = board.turn
    base = material(board, side) - material(board, not side)
    nodes = [0]
    tt = tt if tt is not None else {}                           # shared across the candidates of one position: the lines transpose
    def stand(b):
        n = (material(b, side) - material(b, not side)) - base
        return leaf_value(b, side, n, memo)
    def ours(b, d, alpha, beta):
        nodes[0] += 1
        if b.is_game_over():
            oc = b.outcome(); return (1000.0 if oc and oc.winner == side else 0.0), []
        key = (b._transposition_key(), d, True)
        if key in tt:
            return tt[key]
        if b.is_check():                                        # no standing pat in check: every evasion is a candidate
            if d == 0 or nodes[0] > node_cap:
                return stand(b), []
            best, best_line = None, []
            for f in b.legal_moves:
                nb = b.copy(stack=False); nb.push(f)
                v, line = theirs(nb, d, alpha if best is None else max(alpha, best), beta)
                if best is None or v > best:
                    best, best_line = v, [b.san(f)] + line
                    if best >= beta:
                        break
            res = ((best if best is not None else stand(b)), best_line)
            tt[key] = res
            return res
        best, best_line = stand(b), []                          # we may stop here
        if d == 0 or nodes[0] > node_cap or best >= beta:
            return best, best_line
        for f in forcing_moves(b, memo):
            nb = b.copy(stack=False); nb.push(f)
            v, line = theirs(nb, d, max(alpha, best), beta)
            if v > best:
                best, best_line = v, [b.san(f)] + line
                if best >= beta:
                    break
            if nodes[0] > node_cap:
                break
        tt[key] = (best, best_line)
        return best, best_line
    def theirs(b, d, alpha, beta):
        nodes[0] += 1
        if b.is_game_over():
            oc = b.outcome(); return (1000.0 if oc and oc.winner == side else 0.0), []
        key = (b._transposition_key(), d, False)
        if key in tt:
            return tt[key]
        worst, worst_line = None, []
        for r in reply_order(b):
            nb = b.copy(stack=False); nb.push(r)
            v, line = ours(nb, d - 1, alpha, worst if worst is not None else beta)
            if worst is None or v < worst:
                worst, worst_line = v, [b.san(r)] + line
                if worst <= alpha:
                    break
            if nodes[0] > node_cap:
                break
        res = ((worst if worst is not None else 0.0), worst_line)
        tt[key] = res
        return res
    nb0 = board.copy(stack=False); nb0.push(m)
    best_v, best_line, completed = 0.0, [], 0
    for d in range(1, depth + 1):                               # iterative deepening: the node cap is the budget
        nodes[0] = 0
        v, line = theirs(nb0, d, -1e9, 1e9)
        if nodes[0] > node_cap and d > 1:
            break                                               # this depth did not complete: keep the last one that did
        best_v, best_line, completed = v, line, d
    why = f"combination +{best_v:g} ({completed} deep): {board.san(m)} {' '.join(best_line)}" if best_v > 0 else ""
    return max(0.0, best_v), why


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
                continue                                       # priced by the sure gain: the new piece is in the count, the reply may take it
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
