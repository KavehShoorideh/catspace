"""THE ZONE SEARCH (Kaveh 2026-09-28: "I find it hard to believe that going like 6 plies is hard ... humans go 6 plies
(takes takes takes!) without trouble. Either we aren't focusing on the right frames, or counting plies in the most
important frame is too slow"). Alpha-beta several plies deep inside the ZONE of an idea -- the squares of the frames its
move touches, growing as the line touches new squares and as a move's threats name new targets -- with the moves that
can matter there: captures, checks, evasions, promotions, moves of zone pieces or onto zone squares, and moves after
which the moved piece bears on the zone (the frame interface). Standing pat is passing, and the pass is searched (the
opponent plays on; their threats count; promotions too). Leaves read only the zone's frames, each by its own exchange
(no king joined: the search above plays the checks) -- which is why it is cheap: Kaveh's Kf5 refutation (Rxe5+ Kxe5 Rc5+ Ke6 Rxh5) in about 300 nodes.

A transposition table (alpha-beta bounds, the depth searched, the best move tried first), a node budget and a cap on
the zone's size keep it bounded; iterative deepening (4, 6, 8 plies) returns the deepest completed answer when the budget
runs out (the schedule is the spec's). rank() deepens the candidates together -- all at 4 plies, then at 6, ... -- each with the budget for each
iteration, so they are compared at the same depth unless one explodes (it keeps its last completed depth).

NO LIMIT OR GOAL IS FIXED HERE (Kaveh 2026-09-30): the depths, budgets, margins, reductions and extensions are a
TacticalSpec the planner sets, and the planner's goals -- a pawn run (0M4g4: Nac5 looked safe until the h-pawn ran),
a king watch -- enter the root zones as squares (`spec.hints`); see frame_planner.py.

    python -m catspace.incubator.analyst.zone_search            # the test positions
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass

import chess

from . import fates as F

MATE = 1000.0


@dataclass
class TacticalSpec:
    """EVERY LIMIT OF THE TACTICAL SEARCH, as a parameter the PLANNER sets (Kaveh 2026-09-30: "remove all the hard-coded
    limits on how deep to search ... those should become parameters that we can tune or specify from the planner side;
    the planner should be able to say don't spend a lot of effort here, or do ... the tactical layer doesn't understand
    any plan, it understands frames"). Nothing here names a plan: depths, budgets, margins, and the squares the planner
    wants in every root zone (its goals). The defaults are the values the zone search has been measured with."""
    depths: tuple = (4, 6, 8)           # the iterative-deepening schedule (open-ended with a `stop`)
    budget: int = 20000                 # nodes per candidate per depth
    total_cap: int | None = None        # a hard node total for the position
    margin: float = 3.0                 # a candidate this far behind the best is not deepened
    exempt_checks: bool = True          # ... unless it gives check (a checking sacrifice pays late)
    tie: float = 0.5                    # candidates within `tie` of the best get exact values, the rest a bound
    null_r: int = 2                     # the pass is searched this many plies shallower (null-move reduction)
    check_ext: int = 0                  # checks that do not use up a ply, per line
    check_ply_cap: int = 12             # a side in check at the horizon is searched on up to this ply
    max_zone: int = 18                  # a threat's target joins the zone while the zone is smaller than this
    hints: tuple = ()                   # the planner's goals: callables (board after the candidate) -> squares for its root zone
    alloc: dict | None = None           # the planner's effort per candidate: move -> (weight, deepest depth)
    stop: object = None                 # the decider: stop(depth, got, history) -> bool after each completed depth
    frame_ext: int = 0                  # depth counted in FRAMES, not plies (Kaveh's bet, 2026-09-30: "each frame needs only two
                                        # plies"): a REPLY ON THE SAME SQUARE -- taking the piece that just moved, or moving a piece
                                        # it now attacks -- does not use up a ply, up to frame_ext a line; a move elsewhere (an
                                        # intermezzo) does
    leaf_mates: bool = False            # at the horizon, a mate in one for the side to move is seen (its checking moves tried):
                                        # the frames' read is material and missed it (verdict_dig 2026-09-30: lines ending where
                                        # the side to move had Rh8#, Rh1#)
    zugzwang_guard: bool = False        # a side with only king and pawns may not pass inside the tree (no searched pass, no
                                        # stand-pat above the horizon) and all its moves are searched: in pawn endings the pass is
                                        # unsound (zugzwang) -- engines switch null moves off there (verdict_dig: lines ending in a pass)
    pawn_moves: bool = False            # every pawn move is searched, like captures, promotions and checks: the irreversible
                                        # moves -- races, breaks, tempo moves (miss_anatomy 2026-09-30: of 24 tops overrated
                                        # because the refutation left the zone, 17 refutation lines left it by a pawn move)
    proved_mates: int = 0               # a back-rank mate PROVED for the side to move (Lean, lean/ideas: backRankMate_sound) closes the
                                        # node at MATE without expanding it -- a value the theorem guarantees (Kaveh 2026-10-04: "cut down
                                        # on search correctly"). 1: inside the tree only (where the search would find the mate itself);
                                        # 2: also at the horizon (mates the frames' material read misses, as leaf_mates by brute force)
    proved_passer: bool = False         # a passer whose promotion the passer lemma (second form, Python prototype of the Lean
                                        # hypotheses, 816/816 claims confirmed by exhaustive search 2026-10-05) proves counts at the
                                        # horizon as the queen it brings, like a 7th-rank promotion already does in `gains`
    knowledge: tuple = ()               # tactical facts the planner hands down (jobs.Job: a pin, a guard...), valid by footprint
    prune_jobs: int = 0                 # drop a move that abandons a valid job worth more than it takes at depth <= this (0: order only)


class Budget(Exception):
    pass


def _att(pt, color, sq, occ):
    if pt == chess.PAWN:
        return chess.BB_PAWN_ATTACKS[color][sq]
    if pt == chess.KNIGHT:
        return chess.BB_KNIGHT_ATTACKS[sq]
    if pt == chess.KING:
        return chess.BB_KING_ATTACKS[sq]
    m = 0
    if pt in (chess.BISHOP, chess.QUEEN):
        m |= chess.BB_DIAG_ATTACKS[sq][chess.BB_DIAG_MASKS[sq] & occ]
    if pt in (chess.ROOK, chess.QUEEN):
        m |= chess.BB_RANK_ATTACKS[sq][chess.BB_RANK_MASKS[sq] & occ] | chess.BB_FILE_ATTACKS[sq][chess.BB_FILE_MASKS[sq] & occ]
    return m


class ZoneSearch:
    def __init__(self, budget: int | None = None, max_zone: int | None = None, spec: TacticalSpec | None = None):
        self.spec = spec if spec is not None else TacticalSpec()
        if budget is not None:
            self.spec = TacticalSpec(**{**self.spec.__dict__, "budget": budget})
        if max_zone is not None:
            self.spec = TacticalSpec(**{**self.spec.__dict__, "max_zone": max_zone})
        self.budget = self.spec.budget; self.max_zone = self.spec.max_zone
        self.cap_total = self.spec.total_cap
        self.nodes = 0; self.total = 0; self.tt = {}; self.reads = {}; self.history = {}
        self.hh = {}                                                  # the history heuristic's counters (its own store: bug 2026-09-30,
                                                                      # they shared `history` with rank()'s per-candidate records)
        self.kstats = {"nodes": 0, "fired": 0, "moves": 0, "pruned": 0}   # how often the knowledge acted

    def read(self, b: chess.Board, s: int) -> float:
        """A leaf's read of the frame on s: its own exchange (the square frame, no king joined -- the search above
        already plays the checks and evasions; a capture-only read at the horizon, as a quiescence search does)."""
        key = (b._transposition_key(), s)
        v = self.reads.get(key)
        if v is None:
            v = self.reads[key] = F.resolve_frame(b, {s}, set(), F.square_frame(b, s))[0]
        return v

    def gains(self, b: chess.Board, zone: frozenset) -> float:
        """What the side to move can win now in the zone: a frame on an enemy piece there, or a promotion that survives
        (its gain less what the opponent then wins on the promotion square)."""
        best = 0.0
        for s in zone:
            pc = b.piece_at(s)
            if pc is not None and pc.color != b.turn and pc.piece_type != chess.KING and b.attackers_mask(b.turn, s):
                best = max(best, self.read(b, s))                 # (a piece nothing attacks cannot be won: no frame to read)
        for m in b.generate_legal_moves(from_mask=b.pawns & b.occupied_co[b.turn] & (chess.BB_RANK_7 if b.turn else chess.BB_RANK_2)):
            if m.promotion == chess.QUEEN:
                b.push(m); lost = 0.0 if b.is_checkmate() else self.read(b, m.to_square); b.pop()
                best = max(best, F.captured_value(b, m) - max(0.0, lost))
        if self.spec.proved_passer:                           # a promotion further back, proved rather than read
            from .lean_passer_v2_proto import proved_passer_gain
            g = proved_passer_gain(b)
            if g:
                self.kstats["passer"] = self.kstats.get("passer", 0) + 1
                best = max(best, g)
        return best

    def stand(self, b: chess.Board, zone: frozenset) -> float:
        """The static stand-pat (a pass after a pass): minus what the opponent wins in the zone at once if we pass
        (their threats count)."""
        pb = F.passed(b)
        return 0.0 if pb is None else -self.gains(pb, zone)

    def leaf(self, b: chess.Board, zone: frozenset) -> float:
        """The horizon: the side to move stands pat or cashes in its best frame in the zone, whichever is better (as a
        quiescence search chooses between the static value and its captures)."""
        return max(self.stand(b, zone), self.gains(b, zone))

    def moves(self, b: chess.Board, zone: frozenset) -> list:
        if b.is_check():
            return list(b.legal_moves)
        zm = 0
        for s in zone:
            zm |= chess.BB_SQUARES[s]
        out = []
        for m in b.legal_moves:
            pt = b.piece_type_at(m.from_square)
            occ = (b.occupied & ~chess.BB_SQUARES[m.from_square]) | chess.BB_SQUARES[m.to_square]
            if (b.is_capture(m) or m.promotion or chess.BB_SQUARES[m.from_square] & zm or chess.BB_SQUARES[m.to_square] & zm
                    or _att(pt, b.turn, m.to_square, occ) & zm or b.gives_check(m) or (self.spec.pawn_moves and pt == chess.PAWN)):
                out.append(m)
        return out

    def search(self, b: chess.Board, zone: frozenset, depth: int, alpha: float, beta: float, null_ok: bool = True, ply: int = 0, ext: int = 0,
               frame: frozenset = frozenset()) -> tuple:
        """Negamax alpha-beta inside the zone. Standing pat is passing, and passing is a move: the opponent then plays on
        at depth - 1 - NULL_R (a two-move threat -- a pawn two pushes from queening -- is searched, not missed because the
        pass was read one move deep); near the horizon, and after a pass, the stand-pat is read statically. A side
        in check at the horizon is searched on (its evasions) up to ply 12."""
        self.nodes += 1
        if self.nodes > self.budget:
            raise Budget()
        if b.is_checkmate():
            return -MATE, []
        check = b.is_check()
        if self.spec.proved_mates and (depth > 0 or self.spec.proved_mates >= 2):
            from .lean_ideas import proved_back_rank_mates
            pms = proved_back_rank_mates(b)
            if pms:                                               # proved: the side to move mates at once; nothing below is searched
                self.kstats["proved"] = self.kstats.get("proved", 0) + 1
                pm = max(pms, key=lambda x: F.captured_value(b, x))
                return MATE + F.captured_value(b, pm), [pm]       # (the search's own value of a mate: MATE plus what the mating move takes)
        if depth <= 0 and (not check or ply >= self.spec.check_ply_cap):
            if self.spec.leaf_mates and not check:
                for m in b.legal_moves:
                    if b.gives_check(m):
                        b.push(m); mate = b.is_checkmate(); b.pop()
                        if mate:
                            return MATE, [m]
            return (0.0 if check else self.leaf(b, zone)), []
        depth = max(depth, 1)
        key = (b._transposition_key(), zone, null_ok)
        e = self.tt.get(key)
        hint = None
        if e is not None:
            d0, v, flag, ln = e
            if d0 >= depth and (flag == 0 or (flag == 1 and v >= beta) or (flag == 2 and v <= alpha)):
                return v, ln
            hint = ln[0] if ln else None                          # the best move of a shallower search is tried first
        alpha0 = alpha
        best = -1e9; best_line = []
        bare = self.spec.zugzwang_guard and not (b.occupied_co[b.turn] & ~(b.pawns | b.kings))   # king and pawns only: it must move
        if not check and not bare:
            r = self.spec.null_r
            pb = F.passed(b) if null_ok and depth - 1 - r >= 1 else None
            if pb is not None:                                    # the pass, searched (reduced by null_r): the opponent plays on
                v, ln = self.search(pb, zone, depth - 1 - r, -beta, -alpha, False, ply + 1, ext)   # (a pass: no frame to reply in)
                best, best_line = -v, [chess.Move.null()] + ln
                if ln and ln[0] and -v < 0:                       # the pass shows a threat: its squares join the zone, so the
                    zone = zone | {ln[0].from_square, ln[0].to_square}   # moves that stop it are searched (0ROfv: Qxh2# and Rf2)
            else:
                best = self.stand(b, zone)
            alpha = max(alpha, best)
        if alpha < beta:
            ms = list(b.legal_moves) if bare else self.moves(b, zone)
            hist = self.hh                                        # quiet moves that refuted before are tried earlier (history heuristic)
            ab = {}
            if self.spec.knowledge:                               # KNOWLEDGE (jobs): a move that walks away from a valid job worth more
                from .jobs import abandons                        # than it takes goes last -- the pin known, not re-found in every
                holders = {j.holder for j in self.spec.knowledge}  # sibling line; near the horizon it may be dropped
                for m in ms:
                    if m.from_square in holders:
                        ab[m] = abandons(b, m, self.spec.knowledge)
                fired = sum(1 for v in ab.values() if v > 0.5)
                self.kstats["nodes"] += 1; self.kstats["fired"] += bool(fired); self.kstats["moves"] += fired
                if self.spec.prune_jobs and depth <= self.spec.prune_jobs:
                    before_n = len(ms)
                    ms = [m for m in ms if ab.get(m, 0.0) <= 1.0 or b.gives_check(m)]
                    self.kstats["pruned"] += before_n - len(ms)
            ms.sort(key=lambda m: (m != hint, ab.get(m, 0.0) > 0.5, -(F.captured_value(b, m) + (5 if b.gives_check(m) else 0)),
                                   -hist.get((b.piece_type_at(m.from_square), b.turn, m.to_square), 0)))
            for m in ms:
                g = F.captured_value(b, m)
                b.push(m)
                nz = zone | {m.from_square, m.to_square}
                if len(nz) < self.max_zone:                       # a threat's target joins the zone: its replies are searched
                    nz |= {t for t in chess.SquareSet(b.attacks_mask(m.to_square)) if b.piece_at(t) is not None and b.piece_at(t).color == b.turn}
                try:
                    x = 1 if ext < self.spec.check_ext and b.is_check() else 0   # a check does not use up a ply (up to check_ext a line)
                    if not x and self.spec.frame_ext and frame and ext < self.spec.check_ext + self.spec.frame_ext \
                            and (m.from_square in frame or m.to_square in frame):
                        x = 1                                     # a reply in the last move's frame (the counter is shared: ext)
                    nf = _frame(b, m) if self.spec.frame_ext else frozenset()
                    v, ln = self.search(b, frozenset(nz), depth - 1 + x, -beta + g, -alpha + g, True, ply + 1, ext + x, nf)
                finally:
                    b.pop()
                v = g - v
                if v > best:
                    best, best_line = v, [m] + ln
                alpha = max(alpha, best)
                if alpha >= beta:
                    if not b.is_capture(m):
                        k = (b.piece_type_at(m.from_square), b.turn, m.to_square)
                        hist[k] = hist.get(k, 0) + depth * depth
                    break
        best = best if best > -1e9 else 0.0
        self.tt[key] = (depth, best, 2 if best <= alpha0 else 1 if best >= beta else 0, best_line)
        return best, best_line

    def root(self, b: chess.Board, m: chess.Move):
        """The move's root zone: the square frames of its from and to squares, the kings, the pieces its piece then
        attacks, the squares of the opponent's winning frames after it, and the runners with the squares ahead of them.
        -> (gain, board after, zone) or None when the move mates."""
        zone = {m.from_square, m.to_square}
        for s in list(zone):
            zone |= F.square_frame(b, s)
        for c in (chess.WHITE, chess.BLACK):
            if b.king(c) is not None:
                zone.add(b.king(c))
        bb = b.copy(stack=False); bb.push(m)
        if bb.is_checkmate():
            return None
        zone |= {t for t in chess.SquareSet(bb.attacks_mask(m.to_square)) if bb.piece_at(t) is not None and bb.piece_at(t).color == bb.turn}
        for fr, _, fv, _ in F.joint_threats_book(bb, F.square_frame)[1].values():
            if fv > 0:                                            # a glance at the rest (Kaveh): the opponent's winning frames after the move
                zone |= set(fr[0])                                # join the zone, so a threat elsewhere is not ignored
        for hint in self.spec.hints:                              # the planner's goals (a pawn run, a king watch...) -- the tactical layer
            zone |= set(hint(bb))                                 # only takes their squares
        return F.captured_value(b, m), bb, frozenset(zone)

    def value(self, b: chess.Board, m: chess.Move, depths=None) -> tuple:
        """The move's value by the zone search: the deepest completed depth within the budget. -> (value, line, depth)."""
        return self.rank(b, [m], depths)[m]

    def rank(self, b: chess.Board, moves: list, depths=None, margin: float | None = None, alloc: dict | None = None, stop=None) -> dict:
        """The candidates deepened together: all at 4 plies, then at 6, ...; each candidate has the node budget for each
        iteration, and one that runs out keeps its previous depth's value (so an explosive candidate does not hold the
        others at a shallower depth -- 0Hb6O: with 27 candidates sharing one budget, none reached 8 plies, where the
        rook sacrifice Rxh7+ pays). A candidate more than `margin` behind the best is not deepened further, unless it
        gives check. Candidates are searched best first; one that cannot come within TIE of the best so far gets an
        upper bound, not its exact value (it is not a contender) -- the value it reports is at most that.
        `alloc` (the tactical policy's call -- Kaveh: "deciding whether to continue or not should be the job of a
        tactical policy"): move -> (weight, deepest depth); each candidate's budget per depth is its share of the same
        total (the base budget times the number of candidates), and it is deepened no further than its depth.
        -> {move: (value, line, depth)}."""
        depths = self.spec.depths if depths is None else depths
        margin = self.spec.margin if margin is None else margin
        alloc = self.spec.alloc if alloc is None else alloc
        stop = self.spec.stop if stop is None else stop
        roots = {m: self.root(b, m) for m in moves}
        got = {m: ((1e6, [m], 1) if r is None else (r[0], [m], 0)) for m, r in roots.items()}
        self.total = 0
        self.history = {m: [] for m in moves}                         # per candidate: (depth, value or None if out of budget, nodes)
        self.hh = {}                                                  # the heuristic starts afresh with each rank (as it always has)
        alloc = alloc or {}
        base = self.budget
        W = sum(alloc.get(m, (1.0, 0))[0] for m in moves) or 1.0
        share = {m: max(1, int(base * len(moves) * alloc.get(m, (1.0, 0))[0] / W)) for m in moves}
        deepest = {m: alloc.get(m, (1.0, max(depths)))[1] for m in moves}
        depths = tuple(sorted(set(depths) | set(range(max(depths) + 2, max(deepest.values(), default=0) + 1, 2))))
        cap = getattr(self, "cap_total", None)                        # a hard total for the position (equal-compute comparisons):
        spent0 = getattr(self, "spent_before", 0)                     # nodes already spent (phase 1) count against it
        for i, d in enumerate(depths):
            if cap is not None and spent0 + self.total >= cap:
                break
            top = max(v for v, _, _ in got.values())
            live = [m for m in moves if roots[m] is not None and d <= deepest[m]
                    and (i == 0 or ((got[m][0] >= top - margin or (self.spec.exempt_checks and b.gives_check(m))) and got[m][2] == depths[i - 1]))]
            # (one out of budget stays out; a check is never left behind for its value: a checking sacrifice pays late)
            best_now = -1e9
            for m in sorted(live, key=lambda x: -got[x][0]):           # the previous depth's best first
                g, bb, zone = roots[m]
                lo = best_now - self.spec.tie                         # below this a candidate only needs a bound: it is not
                self.nodes = 0                                        # a contender (fail-soft: the bound is its value's ceiling)
                self.budget = share[m] if cap is None else max(1, min(share[m], cap - spent0 - self.total))
                if cap is not None and spent0 + self.total >= cap:
                    self.budget = base
                    break
                try:
                    nf = _frame(bb, m) if self.spec.frame_ext else frozenset()
                    v, ln = self.search(bb, zone, d - 1, -1e9, g - lo if lo > -1e8 else 1e9, frame=nf)
                except Budget:
                    self.history[m].append((d, None, self.nodes))
                    continue                                          # out of budget at this depth: it keeps the last one
                finally:
                    self.total += self.nodes
                    self.budget = base
                got[m] = (g - v, [m] + ln, d)
                self.history[m].append((d, g - v, self.nodes))
                if g - v > lo:
                    best_now = max(best_now, g - v)
            if stop is not None and stop(d, got, self.history):    # the decider (depth_ladder): go on to the next depth or stop
                break
        return got


def _frame(b: chess.Board, m: chess.Move) -> frozenset:
    """The frame a move opens, on the board after it: the square it landed on and the pieces it now attacks -- a reply
    there (taking it, moving a threatened piece) is a reply on the same square."""
    return frozenset({m.to_square} | {t for t in chess.SquareSet(b.attacks_mask(m.to_square)) if b.piece_at(t) is not None and b.piece_at(t).color == b.turn})


def _san(b: chess.Board, line: list) -> str:
    bb = b.copy(); out = []
    for m in line:
        out.append("(pass)" if not m else bb.san(m)); bb.push(m)
    return " ".join(out)


CASES = [("0KCNf", "2R5/6pk/4P1pp/Pn1K4/2p3rP/8/8/8 w - - 4 45", ["e7", "Rxc4"]),
         ("0M4g4", "8/k2P4/N1n1N1p1/4p3/4P1rp/8/8/3K4 w - - 3 46", ["Nb4", "d8=Q", "Nac5"]),
         ("Kaveh", "6k1/6p1/8/2r1P2Q/6K1/3P2P1/2r5/8 w - - 1 38", ["Kf5", "Qe8+", "Kf4"]),
         ("1HnWH", "8/4N1pk/7p/1p3R2/q3n3/4P2P/2r1Q1P1/6K1 w - - 2 41", ["Qg4", "Qxb5"])]


def main(argv=None):
    budget = int((argv or sys.argv[1:] or ["4000"])[0])
    for pid, fen, cands in CASES:
        b = chess.Board(fen); ms = [b.parse_san(s) for s in cands]
        zs = ZoneSearch(budget); t = time.perf_counter()
        got = zs.rank(b, ms)
        for m in ms:
            v, ln, d = got[m]
            print(f"{pid} {b.san(m)}: {v:+g} at {d} plies  {_san(b, ln)}")
        print(f"   ({zs.total} nodes, {time.perf_counter() - t:.1f}s)")


if __name__ == "__main__":
    main()
