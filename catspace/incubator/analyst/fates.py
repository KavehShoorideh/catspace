"""WHAT A FRAME ACHIEVES (Kaveh 2026-09-24): the outcome of a frame's line is not a number but a FATE RECORD -- for
each frame piece, alive or captured, by which slot, at which ply -- plus the TERMINAL SIGNATURE (the end state
described in the frame's own language: the relations among its pieces and each piece's move counts) and the
LIBRARY'S JUDGMENT of that end state (what the lemmas claim about each piece from there). A GOAL is a predicate on
fates ("their bishop captured by my knight"); a frame answers "achieved by this piece / by another / not achieved",
and, if asked, searches the line with the goal first and material second. No feature is invented: material is one
function of the fates, the rim is few moves, the judgment is the lemmas'. Bounded: one episode per asked goal,
signatures cached by content."""
from __future__ import annotations

from dataclasses import dataclass, field

import chess

from .chess_domain import VAL, State, _ids_after, frame_episode_pv, frame_scalar, _frame_material, passed
from .frames import signature


@dataclass
class Fate:
    slot: int
    alive: bool
    captor: int | None = None       # the slot that captured it; -1 for an outside piece
    ply: int | None = None
    square: int | None = None       # where it stands at the end (None when captured)


def fates_of(start: State, line: list) -> list:
    """Fates from a line of states (each after one ply): a slot whose id becomes None at ply k was captured by the
    slot whose id moved onto its square at that ply, or by an outside piece."""
    out = [Fate(i, True, square=sq) for i, sq in enumerate(start.ids)]
    prev = list(start.ids)
    for k, st in enumerate(line, start=1):
        cur = list(st.ids)
        for i, (a, b) in enumerate(zip(prev, cur)):
            if a is not None and b is None:
                captor = next((j for j, (pa, pb) in enumerate(zip(prev, cur)) if j != i and pa != pb and pb == a), None)
                if captor is None:                                            # en passant: no slot lands on the victim's square; one move a ply, so the slot that moved took it
                    captor = next((j for j, (pa, pb) in enumerate(zip(prev, cur)) if j != i and pa is not None and pb is not None and pa != pb), -1)
                out[i] = Fate(i, False, captor, k, None)
        prev = cur
    for i, sq in enumerate(prev):
        if sq is not None:
            out[i].square = sq
    return out


def move_counts(dom, state: State) -> dict:
    """Per frame piece: legal moves, safe moves (not landing attacked): the piece's freedom in rules-level terms."""
    counts = {}
    lands = next(r for r in dom.relations if r.name == "lands_attacked")
    for i, e in enumerate(state.ids):
        if e is None:
            counts[i] = (0, 0); continue
        acts = dom.actions(state, e)
        safe = sum(1 for a in acts if not lands.fn(state, a))
        counts[i] = (len(acts), safe)
    return counts


def judge(app, state: State) -> dict:
    """The library's claims about each frame piece from the end state: the worst predicted loss and the best predicted
    gain among its actions, with the lemma that says so (restricted to the frame's own pieces)."""
    out = {}
    bodies = {e for e in state.ids if e is None or True} - {None}
    for turn in (True, False):
        b = state.board.copy(stack=False); b.turn = turn
        if not b.is_valid():
            continue
        try:
            res = app.apply(b, bodies=bodies)
        except Exception:
            continue
        for (actor, uci), vl in res["verdicts"].items():
            for c, pred, fired, ids in vl:
                v = (c.value or {}).get("exception" if fired else "holds")
                if v is None:
                    continue
                cur = out.setdefault(actor, {"worst": None, "best": None})
                if cur["worst"] is None or v < cur["worst"][0]:
                    cur["worst"] = (v, c.id, uci)
                if cur["best"] is None or v > cur["best"][0]:
                    cur["best"] = (v, c.id, uci)
    return out


@dataclass
class Outcome:
    move: chess.Move
    value: float
    line: list
    fates: list
    signature: dict = field(default_factory=dict)
    counts: dict = field(default_factory=dict)
    judgment: dict = field(default_factory=dict)
    intermezzi: list = field(default_factory=list)   # [(ply, side, value, fates)]: the outcome if a pass is IMPOSED at that ply (Kaveh 2026-09-24: a move spent outside the frame)
    end: float | None = None                         # the move's own episode: the frame's material at the end of the line from the mover's side, mate +-1000
    baseline: float | None = None                    # what `value` is measured from: the pass episode (relative) or the material before

    def achieved_by(self, slot: int):
        """Who captured this slot: the capturing slot, -1 for an outside piece, None if it lives."""
        f = self.fates[slot]
        return None if f.alive else f.captor


def pass_value(board: chess.Board, ids: list, side: chess.Color, depth: int) -> float | None:
    """The frame's episode value if `side` stands still (the null move): the baseline a move is measured against, so
    a quiet move does not get credit for a tactic that was available anyway. None when standing still is illegal."""
    b = passed(board, side)
    if b is None:
        return None
    v, _ = frame_episode_pv(State(b, list(ids)), side, depth)
    return v


def outcome(dom, board: chess.Board, ids: list, move: chess.Move, depth: int = 0, app=None, max_depth: int = 5, relative: bool = True, with_intermezzi: bool = False) -> Outcome | None:
    """The frame's structured outcome for one action: the episode's line, the fates, the terminal signature and move
    counts, and the library's judgment of the end state when an applier is given. Value: the frame's material after
    the line minus the baseline -- the pass episode when `relative` (the generator's own convention), else the
    material before the move."""
    live = [i for i in ids if i is not None]
    if move.from_square not in live or move not in board.legal_moves:
        return None
    st = State(board.copy(stack=False), list(ids))
    side = board.piece_at(move.from_square).color
    before = frame_scalar(st, move.from_square)
    if relative:
        pv0 = pass_value(board, ids, side, min(max_depth, depth if depth > 0 else len(live)))
        if pv0 is not None:
            before = pv0
    nb = board.copy(stack=False); nb.push(move)
    after = State(nb, _ids_after(list(ids), board, move))
    d = min(max_depth, depth if depth > 0 else len(live))
    v, pv = frame_episode_pv(after, side, d)
    line = [after] + pv
    fates = fates_of(st, line)
    end = line[-1]
    sig = signature(dom, end.board, end.ids)
    counts = move_counts(dom, end)
    jud = judge(app, end) if app is not None else {}
    o = Outcome(move, v - before, line, fates, sig, counts, jud, end=v, baseline=before)
    if with_intermezzi:
        o.intermezzi = intermezzi(board, ids, move, depth, max_depth)
    return o


def _episode_forced_pass(st: State, side: chess.Color, depth: int, ply: int, at, alpha: float, beta: float, line: list) -> tuple:
    """The frame episode with a PASS imposed at ply `at` -- an int, or a set of plies for an intermezzo of several
    plies (a check elsewhere and its answer): every other ply is the ordinary episode. Returns (value, line)."""
    from .chess_domain import VAL
    b = st.board
    mate = (-1000.0 if b.turn == side else 1000.0) if b.is_checkmate() else 0.0
    stand = _frame_material(st, side) + mate
    if depth == 0 or b.is_checkmate() or b.is_stalemate():
        return stand, list(line)
    us = b.turn == side
    frame_sq = {x for x in st.ids if x is not None}
    children = []
    nb = passed(b)
    if nb is not None:
        children.append((0.0, State(nb, list(st.ids))))
    forced = (ply in at) if isinstance(at, (set, frozenset, tuple, list)) else (ply == at)
    if not forced:                                                            # at an intermezzo ply only the pass is available
        for m in b.legal_moves:
            if b.is_check() or b.is_capture(m) or m.promotion or b.gives_check(m) or (depth > 1 and m.from_square in frame_sq):
                nb = b.copy(stack=False); nb.push(m)
                children.append((VAL[b.piece_at(m.to_square).piece_type] if b.piece_at(m.to_square) else 0.0, State(nb, _ids_after(st.ids, b, m))))
    elif not children:
        return stand, list(line)                                              # the pass is illegal here (in check): the intermezzo cannot happen
    children.sort(key=lambda c: -c[0])
    best = None; best_line = list(line)
    for _, child in children:
        v, ln = _episode_forced_pass(child, side, depth - 1, ply + 1, at, alpha, beta, line + [child])
        if best is None or (v > best if us else v < best):
            best, best_line = v, ln
        if us: alpha = max(alpha, best)
        else: beta = min(beta, best)
        if alpha >= beta:
            break
    return (stand, list(line)) if best is None else (best, best_line)


def intermezzi(board: chess.Board, ids: list, move: chess.Move, depth: int = 0, max_depth: int = 5) -> list:
    """For each ply after the move, the frame's outcome if that ply is spent outside the frame: [(ply, side to move
    there, value over the material before, fates)]. Ply 2 is the opponent's first reply; ply 3 our next move."""
    live = [i for i in ids if i is not None]
    st = State(board.copy(stack=False), list(ids))
    side = board.piece_at(move.from_square).color
    before = frame_scalar(st, move.from_square)
    nb = board.copy(stack=False); nb.push(move)
    after = State(nb, _ids_after(list(ids), board, move))
    d = min(max_depth, depth if depth > 0 else len(live))
    out = []
    for at in range(2, d + 2):
        v, ln = _episode_forced_pass(after, side, d, 2, at, -1e9, 1e9, [after])
        who = side if (at % 2 == 1) else (not side)
        out.append((at, who, v - before, fates_of(st, ln)))
    return out


def with_passes(board: chess.Board, ids: list, move: chess.Move, passes, depth: int = 0, max_depth: int = 5) -> tuple:
    """The frame's outcome for `move` when the plies in `passes` (after the move; ply 2 is the first reply) are spent
    outside the frame: (value over the material before, fates). The frame's side of an intermezzo composition."""
    live = [i for i in ids if i is not None]
    st = State(board.copy(stack=False), list(ids))
    side = board.piece_at(move.from_square).color
    before = frame_scalar(st, move.from_square)
    nb = board.copy(stack=False); nb.push(move)
    after = State(nb, _ids_after(list(ids), board, move))
    d = min(max_depth, depth if depth > 0 else len(live))
    v, ln = _episode_forced_pass(after, side, d, 2, frozenset(passes), -1e9, 1e9, [after])
    return v - before, fates_of(st, ln)


def compose(a_board: chess.Board, a_ids: list, a_move: chess.Move, k: int, b_value: float, b_plies: int, depth: int = 0) -> dict:
    """INTERMEZZO BY COMPOSITION (Kaveh 2026-09-24: "taking a frame, adding a pass, and attaching it to another frame
    -- it doesn't need to go into one big frame"): frame B's line of `b_plies` plies is inserted into frame A's line
    at ply k. A is valued with passes at k .. k+b_plies-1; B contributes its own value; the pieces must be disjoint
    (the listener says whether they are). Returns the composed value and A's fates under the insertion."""
    a_val, a_fates = with_passes(a_board, a_ids, a_move, range(k, k + b_plies), depth)
    return dict(value=a_val + b_value, a_value=a_val, b_value=b_value, a_fates=a_fates, at=k, plies=b_plies)


def outcomes_table(dom, board: chess.Board, ids: list, actor: int, depth: int = 0, app=None) -> list:
    """Every action of the actor with its structured outcome, best value first: the frame's possible endings, ranked."""
    out = []
    for m in board.legal_moves:
        if m.from_square != actor:
            continue
        o = outcome(dom, board, ids, m, depth, app)
        if o is not None:
            out.append(o)
    out.sort(key=lambda o: -o.value)
    return out


# ---------------------------------------------------------------- goals: predicates on fates
def captured(slot: int):
    return lambda f: not f[slot].alive
def captured_by(slot: int, by: int):
    return lambda f: (not f[slot].alive) and f[slot].captor == by
def survives(slot: int):
    return lambda f: f[slot].alive
def both(g1, g2):
    return lambda f: g1(f) and g2(f)


def describe(o: Outcome, board: chess.Board, names: list) -> str:
    parts = []
    for f in o.fates:
        if not f.alive:
            who = "an outside piece" if f.captor == -1 else names[f.captor]
            parts.append(f"{names[f.slot]} taken by {who} at ply {f.ply}")
    free = ", ".join(f"{names[i]} {c[1]}/{c[0]} safe moves" for i, c in o.counts.items() if o.fates[i].alive)
    return f"{board.san(o.move)}: value {o.value:+.1f}; " + ("; ".join(parts) or "nothing captured") + (f"; at the end: {free}" if free else "")


def _episode_goal(st: State, side: chess.Color, depth: int, goal, start: State, line: list, alpha: float, beta: float) -> tuple:
    """The episode with the GOAL first and material second (lexicographic), our side achieving, theirs preventing.
    Returns (value, line). Same move set and pass rule as the frame episode."""
    from .chess_domain import VAL
    b = st.board
    mate = (-1000.0 if b.turn == side else 1000.0) if b.is_checkmate() else 0.0
    def leaf():
        f = fates_of(start, line)
        return (10000.0 if goal(f) else 0.0) + _frame_material(st, side) + mate
    if depth == 0 or b.is_checkmate() or b.is_stalemate():
        return leaf(), list(line)
    us = b.turn == side
    frame_sq = {x for x in st.ids if x is not None}
    children = []
    nb = passed(b)
    if nb is not None:
        children.append((0.0, State(nb, list(st.ids))))
    for m in b.legal_moves:
        if b.is_check() or b.is_capture(m) or m.promotion or b.gives_check(m) or (depth > 1 and m.from_square in frame_sq):
            nb = b.copy(stack=False); nb.push(m)
            children.append((VAL[b.piece_at(m.to_square).piece_type] if b.piece_at(m.to_square) else 0.0, State(nb, _ids_after(st.ids, b, m))))
    children.sort(key=lambda c: -c[0])
    best = None; best_line = list(line)
    for _, child in children:
        v, ln = _episode_goal(child, side, depth - 1, goal, start, line + [child], alpha, beta)
        if best is None or (v > best if us else v < best):
            best, best_line = v, ln
        if us: alpha = max(alpha, best)
        else: beta = min(beta, best)
        if alpha >= beta:
            break
    return (leaf(), list(line)) if best is None else (best, best_line)


def achievable(board: chess.Board, ids: list, actor: int, goal, depth: int = 0, max_depth: int = 5) -> dict:
    """Can the actor's side achieve the goal in this frame against best resistance? -> {'yes': bool, 'move': first
    move or None, 'fates': the fates of the found line, 'value': material at the end}."""
    st = State(board.copy(stack=False), list(ids))
    side = board.piece_at(actor).color
    live = [i for i in ids if i is not None]
    d = min(max_depth, depth if depth > 0 else len(live))
    best = None
    for m in board.legal_moves:
        if m.from_square != actor:
            continue
        nb = board.copy(stack=False); nb.push(m)
        after = State(nb, _ids_after(list(ids), board, m))
        v, ln = _episode_goal(after, side, d, goal, st, [after], -1e9, 1e9)   # d plies after the move, as outcome() counts them (d - 1 missed a fork that wins on ply 3)
        if best is None or v > best[0]:
            best = (v, m, ln)
    if best is None:
        return dict(yes=False, move=None, fates=None, value=None)
    v, m, ln = best
    f = fates_of(st, ln)
    return dict(yes=goal(f), move=m, fates=f, value=v - (10000.0 if goal(f) else 0.0) - frame_scalar(st, actor))   # value: the frame's material delta along the found line


MATE = 1000.0


def _no_action(b: chess.Board) -> bool:
    return not any(True for _ in b.generate_legal_moves())


def exchange(b: chess.Board, s: int, frame: set, side: chess.Color | None = None, alpha: float = -1e9, beta: float = 1e9) -> tuple:
    """THE EXCHANGE FRAME (Kaveh 2026-09-27: "a frame essentially is a sequence of moves ... we will inject a pass at
    each move ... an alpha beta search"). On square s, the side to move either captures there with one of the frame's
    pieces (legal moves only: a pinned or blocked piece waits until the line is clear) or PASSES, which ends the
    exchange for it. Minimax with alpha-beta; the value is material won from `side`'s point of view (default: the side
    to move at the start). Returns (value, line of captures in uci)."""
    side = b.turn if side is None else side
    us = b.turn == side
    if passed(b) is None and _no_action(b):                      # it can neither act nor pass: it has lost (Kaveh 2026-09-28; Qxg7 in
        return (-MATE if us else MATE), []                       # 2r1r1k1/pp3pp1/.. read '+1, a pawn': the king could not take back, so it 'passed')
    best = 0.0; best_line = []                                   # the pass: the exchange stops here
    if us:
        alpha = max(alpha, best)
    else:
        beta = min(beta, best)
    victim = b.piece_at(s)
    if victim is None or alpha >= beta:
        return best, best_line
    gain = (1 if us else -1) * VAL[victim.piece_type]
    fmask = 0
    for q in frame:
        fmask |= chess.BB_SQUARES[q]
    for m in sorted(b.generate_legal_moves(from_mask=fmask, to_mask=chess.BB_SQUARES[s]), key=lambda m: VAL[b.piece_at(m.from_square).piece_type]):   # only the frame's captures onto s: generating every legal move at every node cost most of the ledger
        nb = b.copy(stack=False); nb.push(m)
        promo = (1 if us else -1) * (VAL[m.promotion] - VAL[chess.PAWN]) if m.promotion else 0.0
        v, ln = exchange(nb, s, (frame - {m.from_square}) | {s}, side, alpha - gain - promo, beta - gain - promo)
        v += gain + promo
        if (us and v > best) or (not us and v < best):
            best, best_line = v, [m.uci()] + ln
        if us:
            alpha = max(alpha, best)
        else:
            beta = min(beta, best)
        if alpha >= beta:
            break
    return best, best_line


def exchange_move_value(b: chess.Board, m: chess.Move, frame_of) -> tuple:
    """A move judged in exchange frames, against standing still: what it wins on its destination (the capture itself
    and the exchange the opponent can start there) minus what standing still would lose on its origin (the exchange
    the opponent can start on the piece where it stands). frame_of(board, square) -> the frame's squares.
    Returns (value, destination line, origin line)."""
    side = b.turn
    cap = b.piece_at(m.to_square) if not b.is_en_passant(m) else chess.Piece(chess.PAWN, not side)
    won = (VAL[cap.piece_type] if cap else 0.0) + ((VAL[m.promotion] - VAL[chess.PAWN]) if m.promotion else 0.0)
    nb = b.copy(stack=False); nb.push(m)
    dest, dline = exchange(nb, m.to_square, set(frame_of(nb, m.to_square)), side) if not nb.is_checkmate() else (1000.0, [])
    stand = passed(b)                                            # the opponent to move with our piece where it stands
    if stand is None:
        orig, oline = 0.0, []
    else:
        orig, oline = exchange(stand, m.from_square, set(frame_of(stand, m.from_square)), side)
    return won + dest - orig, dline, oline


def exchange_variations(b: chess.Board, s: int, frame: set, side: chess.Color | None = None) -> list:
    """THE PASS AT EACH PLY (Kaveh 2026-09-27: "we will inject a pass at each move ... those will each be a separate
    outcome"). Along the exchange's principal line, for every ply k: who is to move, the frame's value if that side
    leaves the frame there (an intermezzo elsewhere: the exchange stops at k), the value if it continues, and its
    STAKE -- what leaving costs it. A frame's outcome holds unless, at some ply, the side to move has a move elsewhere
    worth more than its stake: the condition the frame offers the outside world. Values from `side`'s point of view."""
    side = b.turn if side is None else side
    value, line = exchange(b, s, frame, side)
    out = []; so_far = 0.0; nb = b.copy(stack=False); fr = set(frame)
    for k in range(len(line) + 1):
        cont, _ = exchange(nb, s, fr, side)
        mover = nb.turn
        out.append(dict(ply=k, mover="us" if mover == side else "them", leaves=so_far, continues=so_far + cont,
                        stake=(cont if mover == side else -cont)))
        if k == len(line):
            break
        m = chess.Move.from_uci(line[k]); victim = nb.piece_at(s)
        so_far += (1 if mover == side else -1) * (VAL[victim.piece_type] + ((VAL[m.promotion] - VAL[chess.PAWN]) if m.promotion else 0.0))
        nb.push(m); fr = (fr - {m.from_square}) | {s}
    return out


def captured_value(b: chess.Board, m: chess.Move) -> float:
    victim = b.piece_at(m.to_square) if not b.is_en_passant(m) else chess.Piece(chess.PAWN, not b.turn)
    return (VAL[victim.piece_type] if victim else 0.0) + ((VAL[m.promotion] - VAL[chess.PAWN]) if m.promotion else 0.0)


def forcing_value(b: chess.Board, m: chess.Move, frame_of, forcing: int) -> tuple:
    """A FORCING move (Kaveh 2026-09-28: the moves that take the opponent's pass away join the frame's line): what m
    wins, then -- the opponent unable to pass -- the reply that hurts the mover most, answered by the mover's best
    threat one forcing level deeper. No reply: the opponent has lost (or it is a draw when it could pass). Values from
    the mover's side. -> (value, line)."""
    won = captured_value(b, m)
    nb = b.copy(stack=False); nb.push(m)
    if _no_action(nb):
        return (won + MATE if passed(nb) is None else won), [m.uci()]
    worst = None
    for r in nb.legal_moves:
        lost = captured_value(nb, r)
        nb2 = nb.copy(stack=False); nb2.push(r)
        cont = threats(nb2, frame_of, forcing=forcing - 1)
        v = -lost + cont[0]
        if worst is None or v < worst[0]:
            worst = (v, [r.uci()] + cont[2])
    return won + worst[0], [m.uci()] + worst[1]


def threats(b: chess.Board, frame_of, exclude: set = frozenset(), forcing: int = 2) -> tuple:
    """The side to move's best threat: the exchange in the frame of every enemy piece its pieces attack (except
    squares in `exclude`), and every FORCING move -- one after which the opponent cannot pass (the domain's rule; in
    chess, a check) -- played out `forcing` levels deep. A side that can neither act nor pass has lost. -> (value,
    square, line); 0 when nothing beats standing still."""
    if passed(b) is None and _no_action(b):
        return -MATE, None, []
    best = (0.0, None, [])
    for sq, pc in b.piece_map().items():
        if pc.color == b.turn or sq in exclude or not b.attackers(b.turn, sq) or pc.piece_type == chess.KING:
            continue
        v, ln = exchange(b, sq, set(frame_of(b, sq)), b.turn)
        if v > best[0]:
            best = (v, sq, ln)
    if forcing > 0 and best[0] < MATE:
        for m in b.legal_moves:
            nb = b.copy(stack=False); nb.push(m)
            if passed(nb) is not None:
                continue                                          # the opponent may still pass: not forcing
            v, ln = forcing_value(b, m, frame_of, forcing)
            if v > best[0]:
                best = (v, m.to_square, ln)
            if best[0] >= MATE:
                break
    return best


def ledger_move_value(b: chess.Board, m: chess.Move, frame_of, before=None) -> tuple:
    """A move judged against standing still ACROSS FRAMES: what it wins now, minus the opponent's best exchange
    anywhere after it, plus the opponent's best exchange anywhere if we had passed -- a threat elsewhere that the move
    leaves or creates (a queen en prise, an abandoned defence, a discovered attack) is the higher-value intermezzo the
    single frame could not see. A mate after the move is -1000 / +1000. -> (value, their best after, their best if
    we pass)."""
    side = b.turn
    cap = b.piece_at(m.to_square) if not b.is_en_passant(m) else chess.Piece(chess.PAWN, not side)
    won = (VAL[cap.piece_type] if cap else 0.0) + ((VAL[m.promotion] - VAL[chess.PAWN]) if m.promotion else 0.0)
    nb = b.copy(stack=False); nb.push(m)
    if nb.is_checkmate():
        return 1000.0, (0.0, None, []), (0.0, None, [])
    after = threats(nb, frame_of)
    if before is None:                                           # the pass side: the same for every move of a position, so a caller may pass it in
        stand = passed(b)
        before = threats(stand, frame_of) if stand is not None else (0.0, None, [])
    return won - after[0] + before[0], after, before


def direct_frame(b: chess.Board, s: int) -> set:
    """The square's direct attackers and defenders with its occupant: the cheap frame (on the held-out exchanges,
    exact 0.905 against the prior's 0.957; the same score on the verdict gauge)."""
    return {s} | set(b.attackers(chess.WHITE, s)) | set(b.attackers(chess.BLACK, s))


def square_frame(b: chess.Board, s: int) -> set:
    """THE FRAME OF A SQUARE (Kaveh 2026-09-28, from the frame player's Rxg3 miss): its occupant, every piece that
    attacks it, and every piece that would attack it once the frame's pieces are lifted off the board -- it arrives
    when they leave -- to a fixed point. Only frame pieces are lifted: a piece blocked by something outside the frame
    stays out. (Rxg3 in 6r1/p1k2pr1/..: the direct frame lacked the rook behind the rook, +1 for White; with it,
    Rxg3 Nxg3 Rxg3, −3.)"""
    frame = direct_frame(b, s)
    while True:
        lifted = b.copy(stack=False)
        for q in frame:
            if q != s:
                lifted.remove_piece_at(q)
        grown = frame | set(lifted.attackers(chess.WHITE, s)) | set(lifted.attackers(chess.BLACK, s))
        if grown == frame:
            return frame
        frame = grown


def _victim(b: chess.Board, m: chess.Move) -> int:
    return chess.square(chess.square_file(m.to_square), chess.square_rank(m.from_square)) if b.is_en_passant(m) else m.to_square


def _mask(squares) -> int:
    m = 0
    for q in squares:
        m |= chess.BB_SQUARES[q]
    return m


FORCING = 3                                                       # checks on one line, at most (the king hunt Rf2+ Kg1 Qg3+ Kh1 Rh2# takes three)


_READS: list | None = None                                         # the footprint being recorded ([bitmask]), or None


def _closure(b: chess.Board, sq: int, deep: bool = False) -> int:
    """The squares whose contents decide what bears on sq: the lines out of it up to the first piece (to the second
    when `deep`: a pin, an x-ray onto a king), its knight squares, and itself."""
    occ = b.occupied
    m = (chess.BB_DIAG_ATTACKS[sq][chess.BB_DIAG_MASKS[sq] & occ] | chess.BB_RANK_ATTACKS[sq][chess.BB_RANK_MASKS[sq] & occ]
         | chess.BB_FILE_ATTACKS[sq][chess.BB_FILE_MASKS[sq] & occ] | chess.BB_KNIGHT_ATTACKS[sq] | chess.BB_SQUARES[sq])
    if deep:
        occ2 = occ & ~(m & occ & ~chess.BB_SQUARES[sq])
        m |= (chess.BB_DIAG_ATTACKS[sq][chess.BB_DIAG_MASKS[sq] & occ2] | chess.BB_RANK_ATTACKS[sq][chess.BB_RANK_MASKS[sq] & occ2]
              | chess.BB_FILE_ATTACKS[sq][chess.BB_FILE_MASKS[sq] & occ2])
    return m


def _record(b: chess.Board, squares, members, cand: list) -> None:
    """What this node of a resolution reads: both kings (two pieces deep), the frame's squares and pieces, the moves it
    tries; under check also the king's neighbours, the checkers and the squares between them and the king."""
    m = 0
    for c in (chess.WHITE, chess.BLACK):
        k = b.king(c)
        if k is not None:
            m |= _closure(b, k, True)
    for q in set(squares) | set(members):
        m |= _closure(b, q)
    for _, mv, _ in cand:
        m |= _closure(b, mv.from_square) | _closure(b, mv.to_square)
    if b.is_check():
        k = b.king(b.turn)
        for q in chess.SquareSet(chess.BB_KING_ATTACKS[k] | b.checkers_mask()):
            m |= _closure(b, q)
        for c in chess.SquareSet(b.checkers_mask()):
            for q in chess.SquareSet(chess.between(c, k)):
                m |= _closure(b, q)
    _READS[0] |= m


def local_resolve(b: chess.Board, squares, members: set, kings=frozenset(), alpha: float = -1e9, beta: float = 1e9, forcing: int | None = None) -> tuple:
    """A FRAME RESOLVED (Kaveh 2026-09-28: "considering multiple frames simultaneously and resolving them jointly"). A
    frame is a set of SQUARES; its pieces are whatever bears on them. Alpha-beta over the captures ON the frame's squares
    by any piece -- a reply is on the same square (Kaveh: "replies are on the same square, anything somewhere else would
    be an intermezzo"), so what happens elsewhere belongs to a joint frame, not to this one -- and, when a king's frame
    is joined (`kings`), the checks on that king (`forcing` such at most on a line; a check by any piece, not only the
    frame's); the square a check lands on joins the frame for the rest of the line (the reply to a check is on its
    square); each side stops where it may pass; under check every legal evasion (the rules); a side that can neither
    act nor pass has lost. `members`: the frame's pieces, followed through the line. -> (value for the side to move,
    line). One copy of the board, then moves are pushed and popped in place (2026-09-28, 'make frames faster')."""
    return _resolve(b.copy(stack=False), squares, members, kings, alpha, beta, FORCING if forcing is None else forcing)


def _check_moves(b: chess.Board) -> list:
    """The side to move's legal moves that give check: a geometric screen first -- a move checks only if it lands where
    its kind attacks the king, moves a piece off a line between one of its own sliders and the king, promotes, castles
    or takes en passant -- then the rules confirm (gives_check)."""
    k = b.king(not b.turn)
    if k is None:
        return []
    occ = b.occupied; us = b.occupied_co[b.turn]
    knight = chess.BB_KNIGHT_ATTACKS[k]; pawn = chess.BB_PAWN_ATTACKS[not b.turn][k]
    diag = chess.BB_DIAG_ATTACKS[k][chess.BB_DIAG_MASKS[k] & occ]
    line = chess.BB_RANK_ATTACKS[k][chess.BB_RANK_MASKS[k] & occ] | chess.BB_FILE_ATTACKS[k][chess.BB_FILE_MASKS[k] & occ]
    snipers = ((chess.BB_RANK_ATTACKS[k][0] | chess.BB_FILE_ATTACKS[k][0]) & (b.rooks | b.queens) | chess.BB_DIAG_ATTACKS[k][0] & (b.bishops | b.queens)) & us
    disc = 0
    for s in chess.scan_reversed(snipers):
        btw = chess.between(k, s) & occ
        if btw and btw & (btw - 1) == 0 and btw & us:
            disc |= btw
    out = []
    for m in b.generate_pseudo_legal_moves():                     # the screen first, the rules' legality only for what passes it
        pt = b.piece_type_at(m.from_square); to = chess.BB_SQUARES[m.to_square]
        maybe = (chess.BB_SQUARES[m.from_square] & disc or m.promotion or
                 (pt == chess.KNIGHT and to & knight) or (pt == chess.PAWN and to & pawn) or
                 (pt in (chess.BISHOP, chess.QUEEN) and to & diag) or (pt in (chess.ROOK, chess.QUEEN) and to & line) or
                 (pt == chess.KING and (b.is_castling(m))) or (pt == chess.PAWN and b.is_en_passant(m)))
        if maybe and b.is_legal(m) and b.gives_check(m):
            out.append(m)
    return out


_TT: dict = {}                                                     # (position, squares, pieces, kings, checks left) -> (value, bound, line)


def clear_caches(keep_lemmas: bool = True) -> None:
    """Empty the per-position caches (called at the start of every pool task: ten workers holding 5.5 GB each
    stalled a 300-puzzle run on a 36 GB machine, 2026-09-28); the lemma library is kept -- reuse across positions is
    its point."""
    from . import frame_goals
    _TT.clear(); _JOINT.clear(); _RESOLVED.clear(); _KF.clear(); _FOOT.clear(); frame_goals._FRAME_ON.clear()
    if not keep_lemmas:
        _LEMMAS.clear()
_EXACT, _LOWER, _UPPER = 0, 1, 2


def _resolve(b: chess.Board, squares, members: set, kings, alpha: float, beta: float, forcing: int) -> tuple:
    """A transposition table (bounds, as in alpha-beta) shares every node across move orders, frames and candidate
    moves; off while a footprint is being recorded (a hit would skip the reads)."""
    key = None
    if _READS is None:
        key = (b._transposition_key(), frozenset(squares), frozenset(members), frozenset(kings), forcing)
        e = _TT.get(key)
        if e is not None:
            v, flag, ln = e
            if flag == _EXACT or (flag == _LOWER and v >= beta) or (flag == _UPPER and v <= alpha):
                return v, ln
    alpha0 = alpha
    v, ln = _resolve_node(b, squares, members, kings, alpha, beta, forcing)
    if key is not None:
        if len(_TT) > 300_000:
            _TT.clear()
        _TT[key] = (v, _UPPER if v <= alpha0 else _LOWER if v >= beta else _EXACT, ln)
    return v, ln


def _resolve_node(b: chess.Board, squares, members: set, kings, alpha: float, beta: float, forcing: int) -> tuple:
    if b.is_check():                                              # the pass is illegal: every evasion, none is mate
        cand = [(captured_value(b, m), m, False) for m in b.legal_moves]
        if _READS is not None:
            _record(b, squares, members, cand)
        if not cand:
            return -MATE, []
        best = -1e9
    else:
        best = 0.0
        alpha = max(alpha, best)
        if alpha >= beta:
            return best, []
        sqmask = _mask(squares)
        cand = [(captured_value(b, m), m, False) for m in b.generate_legal_captures(to_mask=sqmask)]
        if b.ep_square is not None:                               # en passant: the victim, not the landing square, is on the frame
            cand += [(captured_value(b, m), m, False) for m in b.generate_legal_ep() if _victim(b, m) in squares and not (chess.BB_SQUARES[m.to_square] & sqmask)]
        near = (chess.BB_RANK_6 | chess.BB_RANK_7) if b.turn else (chess.BB_RANK_3 | chess.BB_RANK_2)
        for m in b.generate_legal_moves(from_mask=_mask(members)):
            if m.promotion and not (b.is_capture(m) and chess.BB_SQUARES[m.to_square] & sqmask):
                cand.append((captured_value(b, m), m, False))
            elif (b.pawns & near & chess.BB_SQUARES[m.from_square]) and not b.is_capture(m):
                cand.append((0.0, m, False))                      # a frame's pawn two steps from the last rank advances: it becomes a
                                                                  # queen when it gets there (Kaveh 2026-09-28: not a promotion frame --
                                                                  # "the pawn should just suddenly promote" inside the frame's search)
        if forcing > 0 and (not b.turn) in kings:                 # a check on a joined king, by ANY piece: deeper in a hunt new pieces get checks
            seen = {m for _, m, _ in cand}                        # as the king moves (0Hb6O: Rxh7+ Kxh7 Ng5+ Kg7 Ne6+ -- the f3 knight checks only
            for m in _check_moves(b):                             # once the king stands on h7)
                if m not in seen:
                    cand.append((captured_value(b, m), m, True))
        if _READS is not None:
            _record(b, squares, members, cand)
    best_line = []
    cand.sort(key=lambda t: -t[0])
    for g, m, is_forcing in cand:
        victim = _victim(b, m)
        mem = set(members) - {victim}
        lands = is_forcing or m.promotion or (b.pawns & chess.BB_SQUARES[m.from_square] and not b.is_capture(m))
        sq = (set(squares) | {m.to_square}) if lands else squares  # the reply to a check, a promotion or a pawn's advance is on
                                                                  # its square (Kaveh): d8=Q is answered by Rxd8 (verify mode 2026-09-29)
        if m.from_square in mem or m.to_square in sq:             # a piece that lands on a frame square is a frame piece
            mem = (mem - {m.from_square}) | {m.to_square}
        b.push(m)
        v, ln = _resolve(b, sq, mem, kings, -beta + g, -alpha + g, forcing - (1 if is_forcing else 0))
        b.pop()
        v = g - v
        if v > best:
            best, best_line = v, [m.uci()] + ln
        alpha = max(alpha, best)
        if alpha >= beta:
            break
    return (best if best > -1e9 else 0.0), best_line


def _local_resolve_reference(b: chess.Board, squares, members: set, kings=frozenset(), alpha: float = -1e9, beta: float = 1e9, forcing: int | None = None) -> tuple:
    """A FRAME RESOLVED (Kaveh 2026-09-28: "considering multiple frames simultaneously and resolving them jointly"). A
    frame is a set of SQUARES; its pieces are whatever bears on them. Alpha-beta over the captures ON the frame's squares
    by any piece -- a reply is on the same square (Kaveh: "replies are on the same square, anything somewhere else would
    be an intermezzo"), so what happens elsewhere belongs to a joint frame, not to this one -- and, when a king's frame
    is joined (`kings`), the frame pieces' checks on that king (`forcing` such at most on a line); the square a check
    lands on joins the frame for the rest of the line (the reply to a check is on its square); each side stops where it
    may pass; under check every legal evasion (the rules); a side that can neither act nor pass has lost. `members`:
    the frame's pieces, followed through the line (they give the checks). -> (value for the side to move, line)."""
    forcing = FORCING if forcing is None else forcing
    if b.is_check():                                              # the pass is illegal: every evasion, none is mate
        cand = [(captured_value(b, m), m, False) for m in b.legal_moves]
        if _READS is not None:
            _record(b, squares, members, cand)
        if not cand:
            return -MATE, []
        best = -1e9
    else:
        best = 0.0
        alpha = max(alpha, best)
        if alpha >= beta:
            return best, []
        cand = [(captured_value(b, m), m, False) for m in b.generate_legal_captures() if m.to_square in squares or _victim(b, m) in squares]
        checks = forcing > 0 and (not b.turn) in kings
        for m in b.generate_legal_moves(from_mask=_mask(members)):
            if b.is_capture(m) and (m.to_square in squares or _victim(b, m) in squares):
                continue                                          # already a candidate
            if m.promotion:
                cand.append((captured_value(b, m), m, False))
        if checks:                                                # a check on a joined king, by ANY piece: deeper in a hunt new pieces get checks
            seen = {m for _, m, _ in cand}                        # as the king moves (0Hb6O: Rxh7+ Kxh7 Ng5+ Kg7 Ne6+ -- the f3 knight checks only
            for m in b.legal_moves:                               # once the king stands on h7)
                if m not in seen and b.gives_check(m):
                    cand.append((captured_value(b, m), m, True))
        if _READS is not None:
            _record(b, squares, members, cand)
    best_line = []
    cand.sort(key=lambda t: -t[0])
    for g, m, is_forcing in cand:
        nb = b.copy(stack=False); nb.push(m)
        mem = set(members) - {_victim(b, m)}
        sq = (set(squares) | {m.to_square}) if is_forcing else squares
        if m.from_square in mem or m.to_square in sq:             # a piece that lands on a frame square is a frame piece
            mem = (mem - {m.from_square}) | {m.to_square}
        v, ln = _local_resolve_reference(nb, sq, mem, kings, -beta + g, -alpha + g, forcing - (1 if is_forcing else 0))
        v = g - v
        if v > best:
            best, best_line = v, [m.uci()] + ln
        alpha = max(alpha, best)
        if alpha >= beta:
            break
    return (best if best > -1e9 else 0.0), best_line


def king_squares(b: chess.Board, color: chess.Color) -> set:
    k = b.king(color)
    return set() if k is None else {k} | set(chess.SquareSet(chess.BB_KING_ATTACKS[k]))


_KF: dict = {}


def king_frame(b: chess.Board, color: chess.Color) -> tuple:
    """(memoized per position and colour; the king-join prior and the join read the same king frame)"""
    key = (b._transposition_key(), color)
    got = _KF.get(key)
    if got is None:
        if len(_KF) > 200000:
            _KF.clear()
        got = _KF[key] = _king_frame(b, color)
    return set(got[0]), set(got[1])


def _king_frame(b: chess.Board, color: chess.Color) -> tuple:
    """THE KING FRAME (Kaveh 2026-09-28: "there obviously should be some frames with kings in them ... only the frames
    involving the king or the pieces defending the king are invalidated"): the king's square and the squares around it,
    the pieces bearing on them (each square's frame), the enemy pieces that check it or can in one move, and -- since
    a check must be answered, whatever else the checking piece attacks is an intermezzo the check brings -- the
    squares of the other pieces it attacks from there, with their bearers (Nc7+ in r3k3/8/8/3N4/..: the a8 rook).
    One step, no recursion. -> (squares, members)."""
    sq = king_squares(b, color)
    if not sq:
        return set(), set()
    members = set().union(*(square_frame(b, z) for z in sq))
    own = lambda bb, x: bb.piece_at(x) is not None and bb.piece_at(x).color == color and bb.piece_at(x).piece_type != chess.KING
    targets = set()
    if b.turn == color and b.is_check():                          # the checkers on the board, their squares and their other targets:
        for q in b.checkers():                                    # the reply to a check is on its square (0BHVV: Nxh6+ Qxh6, then Qxh6)
            members.add(q); targets |= {x for x in b.attacks(q) if own(b, x)} | {q}
    t = b if b.turn != color else passed(b)                       # the enemy to move: its checks in one move
    if t is not None:
        for m in t.generate_legal_moves():
            if t.gives_check(m):
                members.add(m.from_square)
                nb = t.copy(stack=False); nb.push(m)
                targets |= {x for x in nb.attacks(m.to_square) if own(nb, x)} | {m.to_square}   # and the square it lands on: the reply is there
    return sq | targets, members | set().union(set(), *(square_frame(b, x) for x in targets))


def interactors(b: chess.Board, members: set, squares=frozenset()) -> tuple:
    """WHAT A FRAME PULLS IN (Kaveh 2026-09-28: "how do you know when to combine frames?"), one step, no recursion: the
    lines the frame opens once its pieces are lifted off the board. A piece that gains an attack on an enemy piece
    brings that piece's SQUARE into the frame (also the frame's own pieces: after g6, exd6 uncovers Qd4 on h8); a piece
    that gains a check brings in that king's frame (after Bxd5 in 1n3r2/1Q2qp2/..: lifting the e6 pawn gives the b1 rook
    Rb6+). A target already among the frame's `squares` is not new; a frame piece standing elsewhere is (after Rg2 in
    2r2r2/1b1k1p2/..: d4+ opens the b7 bishop onto the g2 rook, a piece of White's king frame). -> (squares that join,
    kings whose frame joins)."""
    occ = b.occupied
    mmask = _mask(members)
    lifted = occ & ~mmask
    out = set(); kings = set()
    sliders = (b.bishops | b.rooks | b.queens) & ~mmask
    for q in chess.scan_reversed(sliders):                          # only a slider gains attacks when pieces are lifted
        pt = b.piece_type_at(q); c = bool(b.occupied_co[chess.WHITE] & chess.BB_SQUARES[q])
        before = _att(pt, q, occ); after = _att(pt, q, lifted)
        new = after & ~before
        out |= set(chess.SquareSet(new & b.occupied_co[not c] & ~b.kings))
        king = b.king(not c)
        if king is None:
            continue
        kb = chess.BB_SQUARES[king]
        for t in chess.scan_reversed(after & ~b.occupied_co[c]):  # a move that attacks the royal piece only once the frame is lifted
            if _att(pt, t, (lifted & ~chess.BB_SQUARES[q]) | chess.BB_SQUARES[t]) & kb and not _att(pt, t, (occ & ~chess.BB_SQUARES[q]) | chess.BB_SQUARES[t]) & kb:
                kings.add(not c); break
    for q in members:
        pt = b.piece_type_at(q)
        if pt not in (chess.BISHOP, chess.ROOK, chess.QUEEN):
            continue                                              # a knight, pawn or king attacks the same squares whatever is lifted
        c = bool(b.occupied_co[chess.WHITE] & chess.BB_SQUARES[q])
        new = _att(pt, q, (occ & ~mmask) | chess.BB_SQUARES[q]) & ~_att(pt, q, occ)
        out |= {t for t in chess.SquareSet(new & b.occupied_co[not c] & ~b.kings) if t not in squares}
    return out - set(squares), kings


def _att(pt: int, sq: int, occ: int) -> int:
    """The squares a slider of kind pt on sq attacks, with occupancy occ (the rules' lookup tables)."""
    m = 0
    if pt in (chess.BISHOP, chess.QUEEN):
        m |= chess.BB_DIAG_ATTACKS[sq][chess.BB_DIAG_MASKS[sq] & occ]
    if pt in (chess.ROOK, chess.QUEEN):
        m |= chess.BB_RANK_ATTACKS[sq][chess.BB_RANK_MASKS[sq] & occ] | chess.BB_FILE_ATTACKS[sq][chess.BB_FILE_MASKS[sq] & occ]
    return m


def _interactors_reference(b: chess.Board, members: set, squares=frozenset()) -> tuple:
    """WHAT A FRAME PULLS IN (Kaveh 2026-09-28: "how do you know when to combine frames?"), one step, no recursion: the
    lines the frame opens once its pieces are lifted off the board. A piece that gains an attack on an enemy piece
    brings that piece's SQUARE into the frame (also the frame's own pieces: after g6, exd6 uncovers Qd4 on h8); a piece
    that gains a check brings in that king's frame (after Bxd5 in 1n3r2/1Q2qp2/..: lifting the e6 pawn gives the b1 rook
    Rb6+). A target already among the frame's `squares` is not new; a frame piece standing elsewhere is (after Rg2 in
    2r2r2/1b1k1p2/..: d4+ opens the b7 bishop onto the g2 rook, a piece of White's king frame). -> (squares that join,
    kings whose frame joins)."""
    lifted = b.copy(stack=False)
    for q in members:
        lifted.remove_piece_at(q)
    out = set(); kings = set()
    enemy_piece = lambda t, c: b.piece_at(t) is not None and b.piece_at(t).color != c and b.piece_at(t).piece_type != chess.KING
    for q, pc in b.piece_map().items():
        if q in members:
            continue
        new = set(lifted.attacks(q)) - set(b.attacks(q))
        out |= {t for t in new if enemy_piece(t, pc.color)}
        king = b.king(not pc.color)
        if king is None:
            continue
        for t in lifted.attacks(q):                               # a move that attacks the royal piece only once the frame is lifted
            if b.piece_at(t) is not None and b.piece_at(t).color == pc.color:
                continue
            l2 = lifted.copy(stack=False); l2.remove_piece_at(q); l2.set_piece_at(t, pc)
            b2 = b.copy(stack=False); b2.remove_piece_at(q); b2.set_piece_at(t, pc)
            if king in l2.attacks(t) and king not in b2.attacks(t):
                kings.add(not pc.color); break
    for q in members:
        pc = b.piece_at(q)
        if pc is None:
            continue
        l = b.copy(stack=False)
        for r in members:
            if r != q:
                l.remove_piece_at(r)
        new = set(l.attacks(q)) - set(b.attacks(q))
        out |= {t for t in new if t not in squares and enemy_piece(t, pc.color)}
    return out - set(squares), kings


def joint_base(b: chess.Board, squares: set, kings: set, frame_of=None) -> tuple:
    """A frame joined with what it pulls in, before the counter-check rule: -> (squares, kings, members)."""
    frame_of = frame_of or square_frame
    members = set().union(set(), *(frame_of(b, t) for t in squares))
    for c in kings:
        ks, km = king_frame(b, c); squares = set(squares) | ks; members |= km
    add, kadd = interactors(b, members, squares)
    squares = set(squares) | add; members |= set().union(set(), *(frame_of(b, t) for t in add))
    for c in kadd - set(kings):
        ks, km = king_frame(b, c); squares |= ks; members |= km
    return set(squares), set(kings) | kadd, members


def counter_checks(b: chess.Board, members, c: chess.Color) -> list:
    """The checks the frame's pieces have in one move on the king of colour c (its side to move, the other passing)."""
    t = b if b.turn != c else passed(b)
    if t is None or b.king(c) is None:
        return []
    return [m for m in t.generate_legal_moves(from_mask=_mask(members)) if t.gives_check(m)]


KING_JOIN_GATE = "auto"                                            # callable(b, squares, kings, members, c, checks) -> bool: the learned king-join prior;
                                                                   # "auto": loaded from king_join_prior.json when it exists (Kaveh's 75% rule), else none


def _king_gate():
    global KING_JOIN_GATE
    if KING_JOIN_GATE == "auto":
        try:
            from .king_join_prior import load
            KING_JOIN_GATE = load()
        except (FileNotFoundError, OSError):
            KING_JOIN_GATE = None
    return KING_JOIN_GATE


def joint_frame(b: chess.Board, squares: set, kings: set, frame_of=None) -> tuple:
    """A frame joined with what it pulls in (one step): -> (squares, kings, members). The counter-check rule: a king
    whose frame a frame piece can check joins -- unless the king-join prior (Kaveh's 75% rule: join when the chance
    that it changes the resolution is 25% or more) says it does not matter."""
    squares, kings, members = joint_base(b, squares, kings, frame_of)
    for c in (chess.WHITE, chess.BLACK):                          # a king whose frame a frame piece can check: its frame joins (counter-checks)
        if c in kings or b.king(c) is None:
            continue
        checks = counter_checks(b, members, c)
        gate = _king_gate() if checks else None
        if checks and (gate is None or gate(b, squares, kings, members, c, checks)):
            ks, km = king_frame(b, c); squares |= ks; members |= km; kings.add(c)
        elif checks:
            _LAST.setdefault("skipped", []).append(c)
    return frozenset(squares), frozenset(kings), frozenset(members)


_JOINT: dict = {}


def changed_squares(b: chess.Board, m: chess.Move) -> frozenset:
    """What a move PUBLISHES: the squares whose contents it changes."""
    out = {m.from_square, m.to_square, _victim(b, m)}
    if b.is_castling(m):
        rank = chess.square_rank(m.from_square); kside = chess.square_file(m.to_square) > chess.square_file(m.from_square)
        out |= {chess.square(7 if kside else 0, rank), chess.square(5 if kside else 3, rank)}
    return frozenset(out)


def relations(b: chess.Board, squares, members) -> tuple:
    """A frame's RELATIONAL STATE -- what it subscribes to (Kaveh 2026-09-28: "whenever that piece is affected or if that
    piece moves then we ask what else was depending on this piece"): whose turn it is, and its squares and pieces as
    they stand, who attacks and defends each, which are pinned. Nothing global: a king move touches only the frames the
    king or its defenders are in."""
    rows = []
    for t in sorted(set(squares) | set(members)):
        pc = b.piece_at(t)
        rows.append((t, None if pc is None else pc.symbol(), int(b.attackers_mask(chess.WHITE, t)), int(b.attackers_mask(chess.BLACK, t)),
                     pc is not None and b.is_pinned(pc.color, t)))
    return (b.turn, tuple(rows))


def _piece_att(pt: int, color: chess.Color, sq: int, occ: int) -> int:
    """The squares a piece of kind pt and colour on sq attacks, with occupancy occ."""
    if pt == chess.PAWN:
        return chess.BB_PAWN_ATTACKS[color][sq]
    if pt == chess.KNIGHT:
        return chess.BB_KNIGHT_ATTACKS[sq]
    if pt == chess.KING:
        return chess.BB_KING_ATTACKS[sq]
    return _att(pt, sq, occ)


def potential_capturers(b: chess.Board, squares, members) -> int:
    """Every piece that could capture onto a frame square during the frame's line (the resolver lets ANY piece capture
    there): the attackers of the frame squares with the frame's squares, its members' squares and the capturers found
    so far vacated, to a fixed point -- a rook behind a member bishop comes in once the bishop moves (verify mode
    2026-09-29: an {h1, e3} frame of 4rk2/pp3q2/.. was served Re7xe3 from a board whose rook stood on e7). -> bitboard."""
    lifted = b.occupied
    for q in set(squares) | set(members):
        lifted &= ~chess.BB_SQUARES[q]
    pot = 0
    while True:
        new = 0
        for s in squares:
            new |= b.attackers_mask(chess.WHITE, s, lifted) | b.attackers_mask(chess.BLACK, s, lifted)
        new &= ~pot
        if not new:
            return pot
        pot |= new
        lifted &= ~new


def can_reach_check(b: chess.Board, squares, members) -> bool:
    """CAN THE FRAME'S LINE GIVE CHECK? -- conservatively, with every frame and member square vacated (the line can open
    any of them): a member landing on a frame square would attack the enemy king (a pawn on the last rank as a queen),
    or a slider would attack a king through the vacated squares (a discovered check; a pin that could open). A frame
    whose line can check plays every legal evasion, which reads the whole board: its store key cannot be local
    (verify mode 2026-09-29: the d5 frame of 8/8/2p5/3P3p/2K1k3/.. and of 8/8/2p5/2pP3p/2K1k3/.. shared a local key;
    cxd5+ is answered by Kc5 in one, by Kxc5 -- a pawn outside the frame -- in the other)."""
    movers = set(members) | set(chess.SquareSet(potential_capturers(b, squares, members)))
    lifted = b.occupied
    for q in set(squares) | movers:
        lifted &= ~chess.BB_SQUARES[q]
    for c in (chess.WHITE, chess.BLACK):
        k = b.king(not c)
        if k is None:
            continue
        for m in movers:
            pc = b.piece_at(m)
            if pc is None or pc.color != c or pc.piece_type == chess.KING:
                continue
            for s in squares:
                pt = chess.QUEEN if pc.piece_type == chess.PAWN and chess.square_rank(s) in (0, 7) else pc.piece_type
                if _piece_att(pt, c, s, lifted | chess.BB_SQUARES[s]) & chess.BB_SQUARES[k]:
                    return True
        for s in chess.SquareSet(b.occupied_co[c] & (b.bishops | b.rooks | b.queens)):
            if _att(b.piece_type_at(s), s, lifted | chess.BB_SQUARES[s]) & chess.BB_SQUARES[k]:
                return True
    return False


def near_promotion(b: chess.Board, members) -> bool:
    """A member pawn within two steps of promotion: its advance lands on squares outside the frame, whose occupants and
    attackers the local key does not record (verify mode 2026-09-29: an e8 frame served +1 from a board where d8 was
    blocked to one where d7-d8=Q was open)."""
    near = chess.BB_RANK_6 | chess.BB_RANK_7
    far = chess.BB_RANK_3 | chess.BB_RANK_2
    for m in members:
        pc = b.piece_at(m)
        if pc is not None and pc.piece_type == chess.PAWN and chess.BB_SQUARES[m] & (near if pc.color == chess.WHITE else far):
            return True
    return False


def frame_key(b: chess.Board, squares, kings, members) -> tuple:
    """WHAT A FRAME'S RESOLUTION READS -- its key in the persistent store (Kaveh 2026-09-28: "persistent frames from move
    to move ... every time something moves we incrementally change the state ... a minimal amount of rules based
    calculation ... kind of like the NNUE"): the side to move, both kings' squares (a line opened onto a king changes
    what is legal), the frame's squares with their occupants and who attacks each, and its pieces with their pins. A
    move elsewhere leaves the key -- and so the resolution -- unchanged, whichever position or search branch the frame
    turns up in. The en passant square counts when a capture there is legal (verify mode found a b5 frame served
    'c5xb6 e.p.' in a position without the c5 pawn). A frame that can reach check (a king frame joined, the side in
    check, or a line that can give check -- can_reach_check) plays every legal evasion, by any piece: its key is the
    whole board."""
    if kings or b.is_check() or can_reach_check(b, squares, members) or near_promotion(b, members) or b.has_legal_en_passant():
        return ("board", b._transposition_key(), frozenset(squares), frozenset(kings), frozenset(members))   # (en passant lands
                                                                  # off the frame and may check: hxg3+ e.p. in 7k/6p1/4P3/.. -- verify mode)
    rows = tuple((s, None if b.piece_at(s) is None else b.piece_at(s).symbol(), int(b.attackers_mask(chess.WHITE, s)), int(b.attackers_mask(chess.BLACK, s)))
                 for s in sorted(squares))
    mem = tuple((m, None if b.piece_at(m) is None else b.piece_at(m).symbol(), b.piece_at(m) is not None and b.is_pinned(b.piece_at(m).color, m))
                for m in sorted(members))
    ep = b.ep_square if b.has_legal_en_passant() else None          # an en passant captor attacks the square behind, not the target
    pot = tuple((q, b.piece_at(q).symbol()) for q in chess.SquareSet(potential_capturers(b, squares, members)))
    return ("local", b.turn, b.king(chess.WHITE), b.king(chess.BLACK), ep, rows, mem, pot)


def _wl(labels: list, edge) -> list:
    """Weisfeiler-Lehman refinement: each slot's label absorbs its relations to the others, three rounds."""
    n = len(labels)
    for _ in range(3):
        labels = [hash((labels[i], tuple(sorted((edge(i, j), edge(j, i), labels[j]) for j in range(n) if j != i)))) for i in range(n)]
    return labels


def lemma_key(b: chess.Board, squares, kings, members, strict: bool = True) -> tuple | None:
    """A FRAME'S LEMMA KEY (Kaveh 2026-09-28: "if we are evaluating a frame and we evaluate how it ends out, that should
    become a lemma, so that we keep using it again"): the frame described in the relations of the September lemma
    vocabulary with the squares abstracted away -- its pieces as slots (kind, side relative to the mover, target or not,
    pinned, a pawn a step from promotion) and, for every ordered pair, whether one attacks the other (for its own side:
    defends), through how many frame pieces (x-ray), whether the capture is legal, whether it gives check, and which
    frame pieces the captor bears on from where it lands. Slots in a
    canonical order (labels refined over the relations), so the same relational situation on other squares has the same
    key. -> (key, slots in that order), or None for a frame that can reach check (a king frame joined, the side in check,
    a capture that gives check): its evasions read the whole board.
    strict=False: the frame's TYPE, a description for features, not a key that serves values -- nothing is refused;
    the joined kings, the side in check, en passant, a pawn near promotion, a capturer from outside and a capture that
    gives check are part of the description (Kaveh 2026-09-29: frames as the concepts; "a checkmate frame like this")."""
    flags = ()
    if strict:
        if kings or b.is_check() or b.has_legal_en_passant() or can_reach_check(b, squares, members) or near_promotion(b, members):
            return None                                           # (en passant: the captor is no attacker of the target's square)
    else:
        flags = (tuple(sorted(c == b.turn for c in kings)), b.is_check(), b.has_legal_en_passant(), near_promotion(b, members),
                 bool(potential_capturers(b, squares, members) & ~_mask({q for q in set(members) | set(squares) if b.piece_at(q) is not None})))
    slots = sorted({q for q in set(members) | set(squares) | set(b.king(c) for c in kings if b.king(c) is not None)
                    if b.piece_at(q) is not None})                # (a king frame's members include empty squares)
    if not slots or len(slots) > (12 if strict else 16):
        return None
    if strict and potential_capturers(b, squares, members) & ~_mask(slots):
        return None                                               # a capturer from outside the frame's pieces: the relations do not see it
    mover = b.turn
    info = {}
    for q in slots:
        pc = b.piece_at(q)
        promo = pc.piece_type == chess.PAWN and chess.square_rank(q) == (6 if pc.color == chess.WHITE else 1)
        info[q] = (pc.color == mover, pc.piece_type, q in squares, b.is_pinned(pc.color, q), promo)
    rel = {}
    occ = set(slots)
    side_board = {}
    for c in (chess.WHITE, chess.BLACK):                          # one board per side to move, captures tried by push/pop
        t = b.copy(stack=True); t.turn = c; t.ep_square = None; side_board[c] = t
    for i in slots:
        for j in slots:
            if i == j:
                continue
            pc = b.piece_at(i)
            att = j in b.attacks(i)
            between = [x for x in chess.SquareSet(chess.between(i, j))]
            xr = -1
            if not att and pc.piece_type in (chess.BISHOP, chess.ROOK, chess.QUEEN) and between and all(x in occ for x in between):
                t = b.copy(stack=False)
                for x in between:
                    t.remove_piece_at(x)
                if j in t.attacks(i):
                    xr = len(between)
            legal = chk = False
            lands = frozenset()
            if att and b.piece_at(j).color != pc.color:
                t = side_board[pc.color]
                mv = chess.Move(i, j, promotion=chess.QUEEN if pc.piece_type == chess.PAWN and chess.square_rank(j) in (0, 7) else None)
                legal = t.is_legal(mv)
                chk = legal and t.gives_check(mv)
                if chk and strict:
                    return None                                   # a capture that gives check: the line leaves the frame
                if legal:                                         # what it bears on from where it lands (the September 'lands_attacked':
                    t.push(mv)                                    # verify mode found c5/d5 frames equal in relations, unequal after dxc5)
                    lands = frozenset(k for k in t.attacks(j) if k in occ and k not in (i, j))
                    t.pop()
            rel[(i, j)] = (att, xr, legal, lands) if strict else (att, xr, legal, lands, chk)
    idx = {q: k for k, q in enumerate(slots)}
    labels = _wl([hash(info[q]) for q in slots], lambda a, c: rel[(slots[a], slots[c])][:3] + (len(rel[(slots[a], slots[c])][3]),))
    order = sorted(range(len(slots)), key=lambda k: (labels[k], k))
    ordered = [slots[k] for k in order]
    pos = {q: k for k, q in enumerate(ordered)}
    key = ("lemma" if strict else "type", tuple(info[q] for q in ordered),
           tuple(rel[(x, y)][:3] + (tuple(sorted(pos[k] for k in rel[(x, y)][3])),) + rel[(x, y)][4:] for x in ordered for y in ordered if x != y)) + (flags if not strict else ())
    return key, ordered


def _line_to_slots(b: chess.Board, line: list, ordered: list) -> tuple | None:
    """A frame's line in slot terms (mover slot, victim slot per move), following the slots' squares; None if a move
    is not a capture of one slot by another (a promotion push)."""
    where = {q: k for k, q in enumerate(ordered)}
    out = []
    for u in line:
        m = chess.Move.from_uci(u)
        if m.from_square not in where or m.to_square not in where:
            return None
        a, v = where[m.from_square], where[m.to_square]
        out.append((a, v, m.promotion))
        del where[m.to_square]; where[m.to_square] = where.pop(m.from_square)
    return tuple(out)


def _slots_to_line(slot_line: tuple, ordered: list) -> list:
    at = list(ordered); out = []
    for a, v, promo in slot_line:
        out.append(chess.Move(at[a], at[v], promotion=promo).uci())
        at[a] = at[v]
    return out


TRACE: list | None = None                                          # when a list: every frame analysed, in order (frame_player --trace)
_LAST: dict = {}                                                   # where the last resolution came from; king joins the prior skipped
_LEMMAS: dict = {}                                                 # lemma_key -> (value, line in slot terms): frames generalized
LEMMAS = __import__("os").environ.get("ANALYST_LEMMAS_FRAMES", "1") != "0"
_RESOLVED: dict = {}                                               # frame_key -> (value, line): the persistent store of resolutions
VERIFY_FRAMES = __import__("os").environ.get("ANALYST_VERIFY_FRAMES") == "1"   # recompute on every hit and assert equality (tests)
RESOLVED_STATS = {"hits": 0, "computed": 0}


def resolve_frame(b: chess.Board, squares, kings, members) -> tuple:
    """A frame's resolution through the persistent store: exact recurrence first, then the frame's LEMMA (the same
    relational situation met on other squares); computed only when neither is known, and then stored as both."""
    k = frame_key(b, squares, kings, members)
    hit = _RESOLVED.get(k)
    if hit is not None and not VERIFY_FRAMES:
        RESOLVED_STATS["hits"] += 1
        _LAST["source"] = "exact store"
        return hit
    lk = lemma_key(b, squares, kings, members) if LEMMAS else None
    lem = _LEMMAS.get(lk[0]) if lk is not None else None
    if lem is not None and hit is None and not VERIFY_FRAMES:
        RESOLVED_STATS["lemma_hits"] = RESOLVED_STATS.get("lemma_hits", 0) + 1
        v = (lem[0], _slots_to_line(lem[1], lk[1]) if lem[1] is not None else [])
        _RESOLVED[k] = v
        _LAST["source"] = "lemma"
        return v
    foot_hit = None
    if FOOTPRINTS and k[0] == "board" and hit is None:            # a frame that can reach check: served when its footprint is unchanged
        ident = (frozenset(squares), frozenset(kings), frozenset(members))
        for foot, contents, pos, fv in _FOOT.get(ident, ()):
            if pos == (b.turn, b.castling_rights, b.ep_square) and contents == _contents(b, foot):
                foot_hit = fv; break
        if foot_hit is not None and not VERIFY_FRAMES:
            RESOLVED_STATS["foot_hits"] = RESOLVED_STATS.get("foot_hits", 0) + 1
            _RESOLVED[k] = foot_hit
            return foot_hit
    global _READS
    outer = _READS
    _READS = [0] if (FOOTPRINTS and k[0] == "board") else None
    try:
        v = local_resolve(b, squares, set(members), kings)
        foot = _READS[0] if _READS is not None else 0
    finally:
        _READS = outer
    if foot_hit is not None:
        if (foot_hit[0], list(foot_hit[1])) != (v[0], list(v[1])):
            RESOLVED_STATS["foot_mismatch"] = RESOLVED_STATS.get("foot_mismatch", 0) + 1
            FOOT_MISMATCHES.append((b.fen(), sorted(squares), foot_hit, v))
        else:
            RESOLVED_STATS["foot_hits"] = RESOLVED_STATS.get("foot_hits", 0) + 1
    if FOOTPRINTS and k[0] == "board" and foot:
        ident = (frozenset(squares), frozenset(kings), frozenset(members))
        lst = _FOOT.setdefault(ident, [])
        lst.append((foot, _contents(b, foot), (b.turn, b.castling_rights, b.ep_square), v))
        if len(lst) > 16:
            del lst[0]
        if len(_FOOT) > 200000:
            _FOOT.clear()
    if hit is not None and (hit[0], list(hit[1])) != (v[0], list(v[1])):
        raise AssertionError(f"frame store: {b.fen()} squares {sorted(squares)} stored {hit} recomputed {v}")
    if lem is not None and lem[0] != v[0]:
        RESOLVED_STATS["lemma_mismatch"] = RESOLVED_STATS.get("lemma_mismatch", 0) + 1
        LEMMA_MISMATCHES.append((b.fen(), sorted(squares), lem[0], v[0]))
    elif lem is not None:
        RESOLVED_STATS["lemma_hits"] = RESOLVED_STATS.get("lemma_hits", 0) + 1
    RESOLVED_STATS["computed"] += hit is None
    RESOLVED_STATS["hits"] += hit is not None
    if len(_RESOLVED) > 100000:
        _RESOLVED.clear()
    _RESOLVED[k] = v
    _LAST["source"] = "computed" + (", new lemma" if lk is not None and lk[0] not in _LEMMAS else "" if lk is not None else ", no lemma (can reach check)")
    if lk is not None and lk[0] not in _LEMMAS:
        if len(_LEMMAS) > 100000:
            _LEMMAS.clear()
        _LEMMAS[lk[0]] = (v[0], _line_to_slots(b, v[1], lk[1]))
    return v


LEMMA_MISMATCHES: list = []
FOOT_MISMATCHES: list = []
_FOOT: dict = {}                                                   # (squares, kings, members) -> [(footprint, its contents, turn/castling/ep, resolution)]
FOOTPRINTS = __import__("os").environ.get("ANALYST_FOOTPRINTS") == "1"   # off by default: king frames read a median 59 of 64 squares, so
                                                                          # footprints served 28 of 3,436 lookups and cost 38% more time (2026-09-28)


def _contents(b: chess.Board, foot: int) -> tuple:
    return tuple(int(b.pieces_mask(pt, c)) & foot for c in (chess.WHITE, chess.BLACK) for pt in range(1, 7))


def frame_book(b: chess.Board, frame_of=None, select=None, parent: dict | None = None, changed: frozenset = frozenset()) -> dict:
    """THE FRAME BOOK of a position (Kaveh 2026-09-28, publish-subscribe): under check only the side's own king frame
    (the evasions); otherwise every contested square's frame and the enemy king's frame, each joined with what it pulls
    in, and its resolution -- served by the persistent store (`resolve_frame`) whenever the frame's key recurs, in this
    position's parent, a sibling, or any other branch. `parent`/`changed` are kept for callers; the store supersedes
    them. `select(b, key, squares, kings, members)` may narrow a joined frame (the prior's choice)."""
    specs = {}; skipped = {}
    if b.is_check():
        specs[("king", b.turn)] = joint_frame(b, set(), {b.turn}, frame_of)
    else:
        for sq, pc in b.piece_map().items():
            if pc.color == b.turn or pc.piece_type == chess.KING or not b.attackers(b.turn, sq):
                continue
            _LAST["skipped"] = []
            specs[("sq", sq)] = joint_frame(b, {sq}, set(), frame_of)
            skipped[("sq", sq)] = _LAST.get("skipped", [])
        if b.king(not b.turn) is not None:
            specs[("king", not b.turn)] = joint_frame(b, set(), {not b.turn}, frame_of)
    book = {}
    for key, fr in specs.items():
        if select is not None:
            fr = select(b, key, *fr)
        v, ln = resolve_frame(b, fr[0], fr[1], fr[2])             # the persistent store: unchanged frames are not recalculated
        book[key] = (fr, None, v, ln)
        if TRACE is not None:
            TRACE.append(dict(event="frame", fen=b.fen(), key=key, squares=sorted(fr[0]), kings=sorted(fr[1]), members=sorted(fr[2]),
                              source=_LAST.get("source"), value=v, line=list(ln), skipped=skipped.get(key, [])))
    return book


def joint_threats(b: chess.Board, frame_of=None, forcing: int = 2, select=None, parent: dict | None = None, changed: frozenset = frozenset()) -> tuple:
    """The side to move's best outcome with FRAMES RESOLVED JOINTLY (Kaveh 2026-09-28): the best entry of the position's
    frame book. -> (value, key square, line); 0 when nothing beats standing still (under check it may be negative: the
    evasion costs)."""
    return joint_threats_book(b, frame_of, forcing, select, parent, changed)[0]


def joint_threats_book(b: chess.Board, frame_of=None, forcing: int = 2, select=None, parent: dict | None = None, changed: frozenset = frozenset()) -> tuple:
    key = (b._transposition_key(), id(frame_of), id(select))
    if key in _JOINT:
        if TRACE is not None:
            TRACE.append(dict(event="book from memory", fen=b.fen()))
        return _JOINT[key]
    if len(_JOINT) > 100000:
        _JOINT.clear()
    book = frame_book(b, frame_of, select, parent, changed)
    if b.is_check():                                              # the evasions: the own king frame alone, its value may be negative
        e = book[("king", b.turn)]; best = (e[2], b.king(b.turn), e[3])
    else:
        best = (0.0, None, [])
        for k, (fr, rel, v, ln) in book.items():
            if v > best[0]:
                best = (v, k[1] if k[0] in ("sq", "promo") else b.king(k[1]), ln)
    _JOINT[key] = out = (best, book)
    return out


def line_san(board: chess.Board, move: chess.Move, line: list) -> str:
    """The episode's line in SAN: each state after one ply; a ply whose board is only the turn flipped is a pass."""
    b = board.copy(stack=False); out = [b.san(move)]; b.push(move)
    for st in line[1:]:
        m = next((x for x in b.legal_moves if (lambda c: (c.push(x), c.board_fen())[1])(b.copy(stack=False)) == st.board.board_fen()), None)
        if m is None:
            out.append("(pass)"); b.push(chess.Move.null())
        else:
            out.append(b.san(m)); b.push(m)
    return " ".join(out)


def main(argv=None):
    """python -m catspace.incubator.analyst.fates --fen F --move uci --ids e1,d3,c3,c5 [--depth 0 --max-depth 8]:
    the frame's outcome for the move -- the value against standing still, the frame material at the end and when
    standing still, and the line."""
    import argparse
    from .chess_domain import domain
    ap = argparse.ArgumentParser()
    ap.add_argument("--fen", required=True); ap.add_argument("--move", required=True); ap.add_argument("--ids", required=True)
    ap.add_argument("--depth", type=int, default=0); ap.add_argument("--max-depth", type=int, default=8)
    a = ap.parse_args(argv)
    b = chess.Board(a.fen); m = chess.Move.from_uci(a.move); ids = [chess.parse_square(s) for s in a.ids.split(",")]
    o = outcome(domain(oracle="frame", depth=a.depth), b, ids, m, depth=a.depth, max_depth=a.max_depth)
    print(f"frame {' '.join(f'{b.piece_at(i).symbol()}{chess.square_name(i)}' for i in ids)}: value {o.value:+.1f} "
          f"(frame material at the end {o.end:+.1f}, standing still {o.baseline:+.1f}); line: {line_san(b, m, o.line)}")


if __name__ == "__main__":
    main()
