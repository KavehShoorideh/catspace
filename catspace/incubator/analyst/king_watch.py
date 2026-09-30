"""THE STANDING KING FRAME -- the opponent's root subgoal: check this king (Kaveh 2026-09-29: "the upper layer should
be looking for mates ... what is my opponent's sub goal? let me avoid it ... every frame with a king in it, we keep
that king frame, and if it's not changing we don't do anything; but if something around the king changes we check
again -- did a diagonal just open up? is my king in line with a rook somewhere? ... a king and the opponent's rook on
the same file with a few pieces between ... for this tactic to work all three pieces have to go away ... how do we
remove piece one, two, three ... we want that piece to not be there anymore -- we capture it, it moves, we deflect it --
or we negate its effect -- we constrain it, we add another attacker -- ... some generic way to write it out").

Why (frame_profile mate_misses, 2026-09-29): the mates the frame profile misses are mostly ARRIVALS -- the checking
piece 2+ of its moves away (71 of 117 missed, 50 of 117 caught), often 3+ (29 vs 10) -- and lines that must open (12 vs
3). A snapshot of what bears on what now cannot see them; the pieces that CAN MAKE IT can.

THE WATCH of a king (colour c, horizon k of the attacker's moves):
- MEMBERS: the enemy pieces whose reach (piece_reach) attacks the king within k moves -- an OPEN member -- or, for a
  slider, that can stand within k moves on a line to the king with 1-3 pieces between -- a LATENT member (the line
  would open if those pieces went). Each with its route and moves, and the king's flight squares it can take away.
- CONDITIONS: what stands in a member's way -- the pieces on a latent line (BLOCKS) and the king's side's pieces
  guarding the square a member would check from (GUARDS).
- OPERATORS, one vocabulary for every condition (a piece P on square s doing a job): make P not be there -- CAPTURE
  it, FORCE it away (attack it with something worth less), DEFLECT it (it has a job elsewhere: the ablation read) --
  or negate its job -- OUTNUMBER a guard (another attacker on the square it guards). The attacker's operators are
  the ways the tactic can be made; the defender's answers are the same moves from the other side: step the king off
  the line or to a free flight, INTERPOSE on a member's line, GUARD its checking square, or attack the member.
- PERSISTENCE: the watch is kept under its footprint -- its members' reach footprints, the king's lines and squares --
  and served unchanged until a square in it changes ("if nothing around the king changes, we don't do anything").

    python -m catspace.incubator.analyst.king_watch --fen "<fen>" [--colour white]
    python -m catspace.incubator.analyst.king_watch misses          # does the watch flag the mates the frames missed?
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field

import chess

from .piece_reach import piece_reach

VAL = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
VERIFY = os.environ.get("ANALYST_VERIFY_FRAMES") == "1"
DIRS = {(0, 1): "orth", (0, -1): "orth", (1, 0): "orth", (-1, 0): "orth", (1, 1): "diag", (1, -1): "diag", (-1, 1): "diag", (-1, -1): "diag"}


@dataclass
class Member:
    square: int
    piece: str
    kind: str                           # "open" | "latent"
    moves: int                          # its moves to check (open) or to stand on its line (latent)
    check_from: int                     # the square it checks from
    route: list
    blockers: list = field(default_factory=list)      # latent: the pieces on its line to the king
    flights: list = field(default_factory=list)       # the king's flight squares it can take away within the horizon


@dataclass
class Condition:
    kind: str                           # "blocks" | "guards"
    square: int                         # the piece doing the job
    piece: str
    member: int                         # the member it stands against
    at: int                             # guards: the square it guards (the member's checking square)
    operators: list = field(default_factory=list)     # [(kind, by square, by piece, moves)]


@dataclass
class Watch:
    colour: chess.Color                 # the watched king's side
    king: int
    k: int | None                       # None: no horizon -- every distance kept; what to pursue is the policy's
    members: list
    conditions: list
    flights: list                       # the king's flight squares (empty or enemy-occupied, not attacked now)
    uncovered: list                     # flight squares no member can take away within the horizon
    answers: list                       # the defender's moves: [(kind, move uci, what for)]
    footprint: frozenset

    @property
    def threatened(self) -> bool:
        """An open member one move from checking (the placeholder policy's reading of 'now')."""
        return any(m.kind == "open" and m.moves <= 1 for m in self.members)

    def first_check(self) -> int | None:
        return min((m.moves for m in self.members if m.kind == "open"), default=None)


def _rays(k: int):
    f0, r0 = chess.square_file(k), chess.square_rank(k)
    for (df, dr), kind in DIRS.items():
        out = []; f, r = f0 + df, r0 + dr
        while 0 <= f < 8 and 0 <= r < 8:
            out.append(chess.square(f, r)); f += df; r += dr
        yield kind, out


def _moves_to_attack(b: chess.Board, sq: int, target: int) -> tuple:
    """(moves, route) for the piece on sq to attack target, the others still; (None, []) if it never can."""
    rc = piece_reach(b, sq, k=None)
    d = rc.attack.get(target)
    return (d, rc.attack_route(target)) if d is not None else (None, [])


_STORE: dict = {}
STATS = {"computed": 0, "served": 0}


def _contents(b: chess.Board, squares) -> tuple:
    return tuple((s, b.piece_at(s).symbol() if b.piece_at(s) else None) for s in sorted(squares)) + (b.turn,)


def watch(b: chess.Board, colour: chess.Color, operators: bool = True) -> Watch:
    """The standing frame of the king of `colour` -- every enemy piece with the moves it needs, no horizon (Kaveh:
    "why limit to 4 moves? deciding whether to continue or not should be the job of a tactical policy") -- served
    while its footprint is unchanged."""
    k = None
    ident = (colour, operators)
    for foot, contents, w in _STORE.get(ident, ()):
        if _contents(b, foot) == contents:
            if VERIFY:
                fresh = _compute(b, colour, operators)
                assert [(m.square, m.kind, m.moves) for m in fresh.members] == [(m.square, m.kind, m.moves) for m in w.members], \
                    f"king_watch footprint leak: {b.fen()}"
            STATS["served"] += 1
            return w
    w = _compute(b, colour, operators)
    STATS["computed"] += 1
    lst = _STORE.setdefault(ident, [])
    lst.append((w.footprint, _contents(b, w.footprint), w))
    if len(lst) > 16:
        del lst[0]
    if len(_STORE) > 100000:
        _STORE.clear()
    return w


def clear_store() -> None:
    _STORE.clear(); STATS.update(computed=0, served=0)


def _compute(b: chess.Board, colour: chess.Color, operators: bool) -> Watch:
    K = b.king(colour)
    enemy = not colour
    foot = {K}
    flights = [s for s in chess.SquareSet(chess.BB_KING_ATTACKS[K])
               if (b.piece_at(s) is None or b.piece_at(s).color == enemy) and not b.is_attacked_by(enemy, s)]
    foot |= set(chess.SquareSet(chess.BB_KING_ATTACKS[K]))
    rays = list(_rays(K))
    for _, sqs in rays:
        foot |= set(sqs)
    members = []
    covered = set()
    for s, p in b.piece_map().items():
        if p.color != enemy:
            continue
        rc = piece_reach(b, s, k=None)
        foot |= rc.footprint
        fl = [f for f in flights if rc.attack.get(f) is not None and rc.attack[f] <= 1]   # flights it can take at once
        covered |= set(fl)
        if p.piece_type == chess.KING:
            continue
        d = rc.attack.get(K)
        if d is not None:
            route = rc.attack_route(K)
            members.append(Member(s, p.symbol(), "open", d, route[-1] if route else s, route, flights=fl))
        if p.piece_type in (chess.BISHOP, chess.ROOK, chess.QUEEN):         # a LATENT line: it can stand on a line to the
            best = None                                                     # king, with 1-3 pieces between -- kept even when
                                                                            # an open route exists (bug 2026-09-29: dropping it
                                                                            # lost Kaveh's three-blocker file whenever the rook
                                                                            # could also check by another way)
            for kind, sqs in rays:
                if (p.piece_type == chess.ROOK and kind != "orth") or (p.piece_type == chess.BISHOP and kind != "diag"):
                    continue
                between = []
                for x in sqs:
                    if x == s:
                        n = 0
                    elif b.piece_at(x) is not None:
                        between.append(x); n = None
                    else:
                        n = rc.stand.get(x)
                    if len(between) > 3:
                        break
                    if n is not None and 1 <= len(between) <= 3 and x not in between:
                        cand = (n, len(between), x, list(between))
                        if best is None or cand[:2] < best[:2]:
                            best = cand
                    if x == s:
                        break
            if best is not None:
                n, _, x, bl = best
                members.append(Member(s, p.symbol(), "latent", n, x, rc.route(x) if x != s else [], blockers=bl, flights=fl))
    uncovered = [f for f in flights if f not in covered]
    conditions = []
    for m in members:
        for x in m.blockers:
            conditions.append(Condition("blocks", x, b.piece_at(x).symbol(), m.square, x))
        guards = [g for g in chess.SquareSet(b.attackers(colour, m.check_from)) if g != K] if m.kind == "open" and m.check_from != m.square else []
        for g in guards:
            conditions.append(Condition("guards", g, b.piece_at(g).symbol(), m.square, m.check_from))
    answers = []
    if operators:
        for c in conditions:
            c.operators = _operators(b, c, enemy)
        answers = _answers(b, colour, K, members, flights)
    members.sort(key=lambda m: (m.kind != "open", m.moves))
    return Watch(colour, K, None, members, conditions, flights, uncovered, answers, frozenset(foot))


def _operators(b: chess.Board, c: Condition, side: chess.Color) -> list:
    """The attacking side's ways to deal with the piece doing the job: CAPTURE it or FORCE it away (a piece that can
    attack its square -- worth less: it must move), DEFLECT it (it has a job elsewhere), OUTNUMBER a guard."""
    out = []
    v = VAL[b.piece_at(c.square).piece_type]
    for s, p in b.piece_map().items():
        if p.color != side or p.piece_type == chess.KING:
            continue
        d, _ = _moves_to_attack(b, s, c.square)
        if d is not None:
            out.append(("capture" if d == 0 else "force away" if VAL[p.piece_type] < v else "attack", s, p.symbol(), d))
        if c.kind == "guards":
            d2, _ = _moves_to_attack(b, s, c.at)
            if d2 is not None and d2 > 0:
                out.append(("outnumber", s, p.symbol(), d2))
    other_jobs = sum(1 for m in chess.SquareSet(b.attacks(c.square)) if b.piece_at(m) is not None and b.piece_at(m).color != side
                     and m != c.square and b.piece_at(m).piece_type != chess.KING)
    if other_jobs:
        out.append(("deflect", c.square, b.piece_at(c.square).symbol(), other_jobs))
    out.sort(key=lambda o: (o[3], o[0]))
    return out


def _answers(b: chess.Board, colour: chess.Color, K: int, members: list, flights: list) -> list:
    """The defender's moves (legal now if it is to move; else as plans): step the king to a free flight or off a
    latent line, INTERPOSE on a member's line, GUARD its checking square, attack the member."""
    t = b if b.turn == colour else b.copy(stack=False)
    if t.turn != colour:
        t.turn = colour; t.ep_square = None
        if not t.is_valid():
            return []
    out = []
    lines = {}                                                        # by member (a piece may be an open and a latent member)
    for j, m in enumerate(members):
        lines[j] = set(chess.SquareSet(chess.between(m.check_from, K))) if m.kind == "open" else set()
    for mv in t.legal_moves:
        pc = t.piece_at(mv.from_square)
        if pc.piece_type == chess.KING:
            if mv.to_square in flights:
                out.append(("king steps", mv.uci(), chess.square_name(mv.to_square)))
            continue
        for j, m in enumerate(members):
            if mv.to_square == m.square:
                out.append(("take the member", mv.uci(), f"{m.piece}{chess.square_name(m.square)}"))
            elif mv.to_square in lines[j]:
                out.append(("interpose", mv.uci(), f"{m.piece}{chess.square_name(m.square)}"))
            elif m.kind == "open" and m.check_from != m.square:
                nb = t.copy(stack=False); nb.push(mv)
                if m.check_from in nb.attacks(mv.to_square):
                    out.append(("guard", mv.uci(), f"{m.piece}{chess.square_name(m.square)} from {chess.square_name(m.check_from)}"))
    seen = set(); uniq = []
    for a in out:
        if a[1] not in seen:
            seen.add(a[1]); uniq.append(a)
    return uniq


def policy(w: Watch, budget: int = 8) -> list:
    """THE TACTICAL POLICY (placeholder, to be learned -- Kaveh: "deciding whether to continue or not should be the job
    of a tactical policy"): which members to pursue, in order, within a budget. Here: open members before latent ones,
    fewest moves first, fewest blockers first."""
    return sorted(w.members, key=lambda m: (m.moves + len(m.blockers), m.kind != "open"))[:budget]


def candidate_moves(b: chess.Board, budget: int = 8) -> list:
    """Moves the watches propose for the side to move: toward the enemy king -- the first step of a pursued member's
    route, an immediate capture of a condition piece -- and, if our own king is threatened, the answers."""
    me = b.turn
    out = []
    theirs = watch(b, not me)                                     # the enemy king's watch: its members are OUR pieces
    for m in policy(theirs, budget):
        if m.route:
            mv = chess.Move(m.square, m.route[0])
            if mv in b.legal_moves:
                out.append((m.moves, mv.uci()))
    for c in theirs.conditions:
        for kind, s, _p, d in c.operators:
            if d == 0 and kind == "capture":
                mv = chess.Move(s, c.square)
                if mv in b.legal_moves:
                    out.append((0, mv.uci()))
    mine = watch(b, me)
    if mine.threatened:
        for kind, u, _ in mine.answers:
            out.append((0, u))
    out.sort()
    seen = []
    for _, u in out:
        if u not in seen:
            seen.append(u)
    return seen[:budget]


def features(b: chess.Board) -> dict:
    """Both kings' watches as features for the side to move (distances binned, not cut): open and latent members by
    moves, the pieces they must get past, flights, and conditions."""
    out = {}
    for tag, c in (("our", b.turn), ("their", not b.turn)):
        if b.king(c) is None:
            continue
        w = watch(b, c, operators=False)
        for m in w.members:
            key = f"{tag}K:{m.kind}:{m.piece.lower()}:m{min(m.moves, 4)}" + (f":b{len(m.blockers)}" if m.kind == "latent" else "")
            out[key] = out.get(key, 0) + 1
        out[f"{tag}K:flights{len(w.flights)}"] = 1
        out[f"{tag}K:uncovered{len(w.uncovered)}"] = 1
        out[f"{tag}K:conditions"] = float(len(w.conditions))
    return out


def gauge_candidates(n: int = 300, log=print) -> None:
    """Does the watch add moves the zone ledger's standard candidates lack? Over the 300 test puzzles' first positions:
    the standard candidate set (frame_player: one-ply top 6 and ties, every capture, check and promotion, the quiet
    threats) and the watch's moves -- how many are new."""
    from catspace.io import paths
    from .gauge_puzzles import sample
    from . import frame_player as FP
    from .fates import clear_caches
    rows = sample(str(paths.assets_dir() / "lichess" / "lichess_db_puzzle.csv.zst"), n, 997)
    std_n = w_n = new_n = sol_new = 0
    for r in rows:
        clear_caches()
        b = chess.Board(r["FEN"]); mv = r["Moves"].split(); b.push_uci(mv[0])
        stand = FP.passed(b)
        before, book = FP.joint_threats_book(stand, FP.square_frame) if stand is not None else ((0.0, None, []), None)
        one = FP._one_ply_values(b, before, book)
        ranked = sorted(one, key=lambda u: -one[u]); top1 = max(one.values())
        std = set(ranked[:FP.CANDIDATES]) | {u for u in one if one[u] == top1} | {m.uci() for m in b.legal_moves if b.is_capture(m) or b.gives_check(m) or m.promotion}
        wm = set(candidate_moves(b))
        std_n += len(std); w_n += len(wm); new_n += len(wm - std); sol_new += mv[1] in (wm - std)
    log(f"OBS {len(rows)} puzzle first positions: standard candidates {std_n / len(rows):.1f} a position; the watch's moves "
        f"{w_n / len(rows):.1f}, of which new {new_n / len(rows):.2f}; the solution is one of the watch's new moves in {sol_new}")


def explain(b: chess.Board, w: Watch, top: int = 6) -> list:
    n = lambda s: f"{b.piece_at(s).symbol() if b.piece_at(s) else ''}{chess.square_name(s)}"
    side = "White" if w.colour else "Black"
    lines = [f"{side}'s king on {chess.square_name(w.king)}: flights {' '.join(chess.square_name(s) for s in w.flights) or 'none'}"
             ]
    if not w.members:
        lines.append("  no enemy piece can ever check it with the others still, open or behind 1-3 pieces")
    pursued = policy(w, top)
    for m in pursued:
        via = " ".join(chess.square_name(s) for s in m.route) or "from where it stands"
        if m.kind == "open":
            lines.append(f"  {m.piece}{chess.square_name(m.square)} checks in {m.moves} (from {chess.square_name(m.check_from)}; via {via})"
                         + (f"; takes away {' '.join(chess.square_name(f) for f in m.flights)}" if m.flights else ""))
        else:
            lines.append(f"  {m.piece}{chess.square_name(m.square)} stands on a line to the king in {m.moves} (at {chess.square_name(m.check_from)}), "
                         f"if {', '.join(n(x) for x in m.blockers)} go")
    if len(w.members) > len(pursued):
        lines.append(f"  (+{len(w.members) - len(pursued)} more members, not pursued by the policy's budget)")
    for c in [c for c in w.conditions if c.member in {m.square for m in pursued}]:
        ops = "; ".join(f"{kind} by {p}{chess.square_name(s)} in {d}" if kind != "deflect" else f"deflect ({d} other job{'s' if d > 1 else ''})"
                        for kind, s, p, d in c.operators[:4]) or "no way found"
        what = f"blocks {n(c.member)}'s line" if c.kind == "blocks" else f"guards {chess.square_name(c.at)} against {n(c.member)}"
        lines.append(f"    condition: {n(c.square)} {what} -- {ops}")
    if w.answers and w.threatened:
        lines.append("  answers: " + "; ".join(f"{kind} {chess.Move.from_uci(u).uci()}" for kind, u, _ in w.answers[:6]))
    return lines


def main(argv=None):
    import sys
    if argv is None and len(sys.argv) > 1 and sys.argv[1] == "candidates":
        return gauge_candidates()
    if argv is None and len(sys.argv) > 1 and sys.argv[1] == "misses":
        from .frame_profile import watch_misses
        from catspace.io import paths
        return watch_misses(str(paths.derived("frame_profiles_1000_0.jsonl.gz")))
    ap = argparse.ArgumentParser(); ap.add_argument("--fen", required=True); ap.add_argument("--colour", default=None)
    a = ap.parse_args(argv)
    b = chess.Board(a.fen)
    cols = [chess.WHITE if a.colour == "white" else chess.BLACK] if a.colour else [chess.WHITE, chess.BLACK]
    for c in cols:
        w = watch(b, c)
        for line in explain(b, w, top=6):
            print(line)
    print("candidates for the side to move:", [b.san(chess.Move.from_uci(u)) for u in candidate_moves(b)])


if __name__ == "__main__":
    main()
