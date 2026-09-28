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


def exchange(b: chess.Board, s: int, frame: set, side: chess.Color | None = None, alpha: float = -1e9, beta: float = 1e9) -> tuple:
    """THE EXCHANGE FRAME (Kaveh 2026-09-27: "a frame essentially is a sequence of moves ... we will inject a pass at
    each move ... an alpha beta search"). On square s, the side to move either captures there with one of the frame's
    pieces (legal moves only: a pinned or blocked piece waits until the line is clear) or PASSES, which ends the
    exchange for it. Minimax with alpha-beta; the value is material won from `side`'s point of view (default: the side
    to move at the start). Returns (value, line of captures in uci)."""
    side = b.turn if side is None else side
    us = b.turn == side
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


def threats(b: chess.Board, frame_of, exclude: set = frozenset()) -> tuple:
    """The side to move's best exchange anywhere: over every enemy piece its pieces attack (except squares in
    `exclude`), the exchange in that piece's frame. -> (value, square, line)."""
    best = (0.0, None, [])
    for sq, pc in b.piece_map().items():
        if pc.color == b.turn or sq in exclude or not b.attackers(b.turn, sq) or pc.piece_type == chess.KING:
            continue
        v, ln = exchange(b, sq, set(frame_of(b, sq)), b.turn)
        if v > best[0]:
            best = (v, sq, ln)
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
