"""needs.py -- MEANS-ENDS planning over needs (Kaveh 2026-09-07): "the need to get the knight to that square becomes
its own goal with its own blockers; keep going down until I find an answer or not; each line gets a value; I
compare; it is a directed search over the things I need, not the whole tree."

A NEED is a goal node in an AND/OR tree over goals, not over moves: win a piece, reach a square, make a square
safe, remove a piece, deflect a piece, block a line. Each need has WAYS (OR); each way is a move now, or a set of
sub-needs it requires (AND). Every line carries its gain and its cost -- material given, holdings broken, the
king's shelter weakened (structure), tempo -- and alternatives are compared at the node where they split. Give-up
is a need with no way, a cost past the gain, or the depth bound. The base cases are the existing planner
(plans.derive) and reach; the VERIFIER is the goal search with the line's first move as its prior, and a
refutation comes back as a sub-need (secure X), as it does for single goals. Nothing the planner says is believed
until proven; nothing it does not say is pruned from the proof.

Newell and Simon's GPS (1959) with a prover underneath; PARADISE (1980) had the patterns and no prover.

    python -m catspace.incubator.analyst.needs --fen "<fen>" [--target c6] [--engine]
"""
from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass, field

import chess

from . import rules as R
from .defense import defense
from .facts import value, see
from .memo import Memo
from .reach import reach, flip_conditions, _see_lists
from .plans import derive, Goal, Plan, Step, answer as plan_answer
from .subgames import holdings, cost_of_move, holding_text
from .structure import structure_diff

MAX_DEPTH = 4
SHELTER = 1.0                    # a pawn move in front of a castled king, or a king file opened
WEAKNESS = 0.5                   # an isolated / doubled / backward pawn created


@dataclass(frozen=True)
class Need:
    kind: str                    # win | defend | reach | safe | remove | deflect | block
    square: int                  # the target square, or the destination
    piece: int | None = None     # the piece concerned (reach: the mover; safe: the piece that must land there)
    against: int | None = None   # deflect / remove: the piece to move away; block: the coverer

    def text(self, board: chess.Board) -> str:
        p = board.piece_at(self.square)
        name = lambda sq: f"the {chess.piece_name(board.piece_at(sq).piece_type)} on {chess.square_name(sq)}" if board.piece_at(sq) else chess.square_name(sq)
        if self.kind == "win":
            return f"win {name(self.square)}"
        if self.kind == "defend":
            return f"hold {name(self.square)}"
        if self.kind == "reach":
            return f"bring {name(self.piece)} to {chess.square_name(self.square)}"
        if self.kind == "safe":
            return f"make {chess.square_name(self.square)} safe for {name(self.piece)}"
        if self.kind == "remove":
            return f"remove {name(self.against)}"
        if self.kind == "deflect":
            return f"deflect {name(self.against)} from {chess.square_name(self.square)}"
        if self.kind == "block":
            return f"block {name(self.against)} on the way to {chess.square_name(self.square)}"
        return self.kind


@dataclass
class Way:
    moves: list[chess.Move]                            # our own moves for this way, AFTER its sub-needs' moves
    san: str
    why: str
    subs: list["Line"] = field(default_factory=list)   # sub-needs, all required, played first
    gain: float = 0.0
    cost: float = 0.0
    hops: int = 0
    costs: list[str] = field(default_factory=list)     # the costs in words
    reply: str = ""                                    # the reply the way expects and plays through (a lure taken)
    after: chess.Board | None = None                   # the board once the way (subs, moves, expected reply) has happened
    boards: list = field(default_factory=list)         # the board before each of our own moves (for SAN and replay)

    @property
    def net(self) -> float:
        return self.gain - self.cost - sum(s.cost for s in self.subs)

    @property
    def met(self) -> bool:
        return all(s.best is not None for s in self.subs) and self.after is not None


@dataclass
class Line:
    need: Need
    ways: list[Way] = field(default_factory=list)      # explored, best first
    best: Way | None = None
    status: str = "open"                               # met | given_up | verified | refuted
    reason: str = ""                                   # why given up
    depth: int = 0
    answer: str = ""                                   # the opponent's fastest counter to the line, in words
    race: str = ""                                     # "wins" | "met" | ""
    free_tempi: int = 0                                # the opponent's free moves before our final step
    verification: str = ""

    @property
    def net(self) -> float:
        return self.best.net if self.best else 0.0

    @property
    def cost(self) -> float:
        """What meeting this need costs along its chosen way, sub-needs included."""
        if self.best is None:
            return 0.0
        return self.best.cost + sum(s.cost for s in self.best.subs)

    def steps(self) -> list[chess.Move]:
        """Our moves along the chosen line, sub-needs first (they are the preparation), then the way's own moves."""
        return [m for m, _ in self.step_pairs()]

    def step_pairs(self) -> list[tuple[chess.Move, chess.Board]]:
        if self.best is None:
            return []
        out = []
        for s in self.best.subs:
            out += s.step_pairs()
        return out + list(zip(self.best.moves, self.best.boards))

    def sans(self) -> list[str]:
        out = []
        for m, b in self.step_pairs():
            bb = b.copy(stack=False); bb.turn = b.turn
            out.append(bb.san(m) if m in bb.legal_moves else m.uci())
        return out

    @property
    def after(self) -> chess.Board | None:
        return self.best.after if self.best else None


# ---------------------------------------------------------------- the planner
class Planner:
    def __init__(self, board: chess.Board, side: chess.Color, memo: Memo | None = None, max_depth: int = MAX_DEPTH, budget: int = 80):
        self.board = board.copy(stack=False); self.board.turn = side
        self.side, self.memo = side, memo if memo is not None else Memo()
        self.max_depth, self.budget = max_depth, budget
        self.expanded = 0
        self.held = holdings(self.board, self.memo)
        self.trace: list[str] = []

    # ---- costs
    def move_cost(self, board: chess.Board, m: chess.Move) -> tuple[float, list[str]]:
        """What a move of ours gives up: material (a piece that falls where it lands, a capture that loses), holdings
        broken, the king's shelter, structural weaknesses."""
        b = board.copy(stack=False); b.turn = self.side
        cost, why = 0.0, []
        nb = b.copy(stack=False); nb.push(m)
        p = b.piece_at(m.from_square)
        if b.is_capture(m):
            g = see(b, m.to_square, self.side)
            if g < 0:
                cost += -g; why.append(f"the exchange on {chess.square_name(m.to_square)} loses {-g:g}")
        else:
            d = defense(nb, m.to_square, memo=self.memo)
            if d is not None and d.falls and not R.is_royal(nb, m.to_square):
                cost += value(b, m.from_square); why.append(f"{chess.piece_name(p.piece_type)} falls on {chess.square_name(m.to_square)}")
        if board is self.board or board.fen() == self.board.fen():
            broken = cost_of_move(b, m, self.held, self.memo)
        else:
            broken = cost_of_move(b, m, holdings(b, self.memo), self.memo)
        for h in broken:
            cost += h.loss; why.append(f"the {chess.piece_name(p.piece_type)} {holding_text(b, h)}")
        try:
            created, ended = structure_diff(b, [m])
        except Exception:
            created, ended = [], []
        for r in created:
            if r.owner != self.side:
                continue
            if r.name == "king_open_file":
                cost += SHELTER; why.append("opens a file next to the king")
            elif r.name in ("isolated_pawn", "doubled_pawns", "backward_pawn"):
                cost += WEAKNESS; why.append(r.name.replace("_", " "))
        k = b.king(self.side)
        if p is not None and p.piece_type == chess.PAWN and k is not None and not b.has_castling_rights(self.side) \
                and chess.square_distance(m.from_square, k) <= 1:
            cost += SHELTER; why.append("moves a pawn from the king's shelter")
        return cost, why

    # ---- the tree
    def plan(self, need: Need, board: chess.Board | None = None, depth: int = 0, exclude: frozenset = frozenset()) -> Line:
        board = self.board if board is None else board
        line = Line(need, depth=depth)
        if depth > self.max_depth:
            line.status = "given_up"; line.reason = "too deep"; return line
        if self.expanded >= self.budget:
            line.status = "given_up"; line.reason = "budget"; return line
        if need in exclude:
            line.status = "given_up"; line.reason = "circular"; return line
        self.expanded += 1
        ways = getattr(self, f"ways_{need.kind}")(need, board, depth, exclude | {need})
        ways.sort(key=lambda w: (not w.met, -w.net, w.hops))
        line.ways = ways
        for w in ways:
            if w.met and (w.net > 0 or need.kind != "win"):
                line.best = w; line.status = "met"; break
        if line.best is None:
            line.status = "given_up"
            line.reason = "no way" if not ways else ("no way that pays" if any(w.met for w in ways) else "every way needs something we cannot get")
        self.trace.append("  " * depth + self.line_text(line, short=True))
        return line

    def played(self, board: chess.Board, moves: list[chess.Move], reply: chess.Move | None = None) -> chess.Board:
        """The board after our moves (the opponent passing) and, when given, their expected reply."""
        b = board.copy(stack=False)
        for m in moves:
            b.turn = self.side; b.push(m)
        if reply is not None:
            b.turn = not self.side; b.push(reply)
        b.turn = self.side
        return b

    def boards_of(self, board: chess.Board, moves: list[chess.Move]) -> list[chess.Board]:
        """The board before each of our moves, the opponent passing in between."""
        out = []; b = board.copy(stack=False)
        for m in moves:
            b.turn = self.side; out.append(b.copy(stack=False)); b.push(m)
        return out

    # win T: capture now (the ledger), add an attacker (reach a square that attacks T), remove a defender
    def ways_win(self, need: Need, b: chess.Board, depth: int, ex) -> list[Way]:
        t = need.square; out = []
        tp = b.piece_at(t)
        if tp is None or tp.color == self.side:
            return out
        d = defense(b, t, memo=self.memo)
        if d is not None and d.falls:
            for pl in derive(b, self.side, Goal("win", t), self.memo, depth=0)[:2]:
                if pl.steps:
                    c, why = self.move_cost(b, pl.steps[0].move)
                    out.append(Way([pl.steps[0].move], pl.steps[0].san, "it falls now", gain=max(d.see, 0.5), cost=c, hops=1, costs=why,
                                   after=self.played(b, [pl.steps[0].move]), boards=[b]))
            return out
        for f in flip_conditions(b, t, self.side, memo=self.memo):
            if f.kind == "add_attacker":
                attacking = set(d.attackers) if d else set()
                for psq, pp in sorted(b.piece_map().items(), key=lambda kv: value(b, kv[0])):
                    if pp.color != self.side or R.is_royal(b, psq) or psq in attacking or value(b, psq) > f.max_value:
                        continue
                    for sq in self.attack_squares(b, psq, t):
                        sub = self.plan(Need("reach", sq, piece=psq), b, depth + 1, ex)
                        gain = self.flip_gain(b, t, psq)
                        w = Way([], "", f"add the {chess.piece_name(pp.piece_type)} on {chess.square_name(sq)} as an attacker", [sub], gain=gain,
                                hops=len(sub.steps()) + 1, after=sub.after)
                        if sub.best is not None and sub.after is not None:
                            cap = self.capture_move(sub.after, t)
                            if cap is not None:
                                w.moves = [cap]; w.san = self.san(sub.after, cap); w.after = self.played(sub.after, [cap]); w.boards = [sub.after]
                        out.append(w)
                        if sub.best is not None:
                            break
                    if out and out[-1].met:
                        break
            elif f.kind == "remove_defender" and f.square is not None:
                sub = self.plan(Need("remove", t, against=f.square), b, depth + 1, ex)
                out.append(Way([], "", f"remove the defender on {chess.square_name(f.square)}", [sub], gain=value(b, t), hops=len(sub.steps()) + 1, after=sub.after))
        return out

    def ways_defend(self, need: Need, b: chess.Board, depth: int, ex) -> list[Way]:
        out = []
        for pl in derive(b, self.side, Goal("defend", need.square), self.memo, depth=1)[:3]:
            if pl.steps and pl.steps[0].move != chess.Move.null():
                c, why = self.move_cost(b, pl.steps[0].move)
                out.append(Way([pl.steps[0].move], pl.steps[0].san, pl.chain[-1].strip() if pl.chain else "holds it", gain=value(b, need.square), cost=c,
                               hops=pl.hops, costs=why, after=self.played(b, [pl.steps[0].move]), boards=[b]))
        return out

    # reach: the safe path; if none, the first unsafe square on the geometric path becomes 'make it safe', and the
    # path is re-planned on the board where that has been done (the final hop may be a capture that pays)
    def ways_reach(self, need: Need, b: chess.Board, depth: int, ex) -> list[Way]:
        psq, sq = need.piece, need.square; out = []
        if b.piece_at(psq) is None:
            return out
        if psq == sq:
            return [Way([], "", "already there", hops=0, after=b)]
        p = self.safe_path(b, psq, sq, 3)
        if p is not None:
            moves = self.path_moves(b, psq, p)
            if moves:
                c, why = sum(self.move_cost(b, m)[0] for m in moves[:1]), self.move_cost(b, moves[0])[1]
                out.append(Way(moves, " ".join(self.san_seq(b, moves)), f"{len(p)} move{'s' if len(p) > 1 else ''}", cost=c, hops=len(p), costs=why,
                               after=self.played(b, moves), boards=self.boards_of(b, moves)))
                return out
        geo = self.geometric_path(b, psq, sq, 3)
        if geo is None:
            return out
        unsafe = next((s for s in geo if self.unsafe(b, psq, s)), None)
        if unsafe is None:
            return out
        sub = self.plan(Need("safe", unsafe, piece=psq), b, depth + 1, ex)
        w = Way([], "", f"{chess.square_name(unsafe)} on the way is covered", [sub], hops=len(sub.steps()) + len(geo), after=sub.after)
        if sub.best is not None and sub.after is not None and sub.after.piece_at(psq) is not None:
            p2 = self.safe_path(sub.after, psq, sq, 3, capture_last=True)
            if p2 is not None:
                moves = self.path_moves(sub.after, psq, p2)
                if moves:
                    w.moves = moves; w.san = " ".join(self.san_seq(sub.after, moves)); w.hops = len(sub.steps()) + len(moves)
                    w.after = self.played(sub.after, moves); w.boards = self.boards_of(sub.after, moves)
                    c, why = self.move_cost(sub.after, moves[0]); w.cost = c; w.costs = why
            else:
                w.subs = [sub]; w.why += ", and still no way after that"
                w.after = None
        out.append(w)
        return out

    # safe: the coverers of the square; each must be removed, deflected or blocked
    def ways_safe(self, need: Need, b: chess.Board, depth: int, ex) -> list[Way]:
        sq, psq = need.square, need.piece; out = []
        coverers = self.coverers(b, psq, sq)
        if not coverers:
            return [Way([], "", "it is safe", hops=0, after=b)]
        if len(coverers) > 1:
            return out                                        # two coverers within our depth: give it up honestly
        c = coverers[0]
        for kind in ("deflect", "remove", "block"):
            sub = self.plan(Need(kind, sq, piece=psq, against=c), b, depth + 1, ex)
            out.append(Way([], "", f"{kind} the {chess.piece_name(b.piece_at(c).piece_type)} on {chess.square_name(c)}", [sub], hops=len(sub.steps()), after=sub.after))
            if sub.best is not None:
                break
        return out

    def ways_remove(self, need: Need, b: chess.Board, depth: int, ex) -> list[Way]:
        sub = self.plan(Need("win", need.against), b, depth + 1, ex)
        return [Way([], "", "capture it", [sub], hops=len(sub.steps()), after=sub.after)]

    # deflect: give the coverer something it must take (a lure on a square it attacks) so that taking it uncovers the
    # square; priced by what the lure gives and what the move weakens; the expected reply is played through
    def ways_deflect(self, need: Need, b: chess.Board, depth: int, ex) -> list[Way]:
        c = need.against; sq = need.square; out = []
        cp = b.piece_at(c)
        if cp is None:
            return out
        bb = b.copy(stack=False); bb.turn = self.side
        for m in bb.legal_moves:
            if bb.is_capture(m) or m.to_square not in b.attacks(c) or m.from_square == need.piece or value(b, m.from_square) > 3:
                continue
            nb = bb.copy(stack=False); nb.push(m)
            cap = chess.Move(c, m.to_square)
            if cap not in nb.legal_moves:
                continue
            after = nb.copy(stack=False); after.push(cap); after.turn = self.side
            if self.unsafe(after, need.piece, sq):
                continue
            taking = see(nb, m.to_square, not self.side)             # what they net by taking: the lure less our recapture
            if taking < 0:
                continue                                           # a poisoned lure: they will not take, the square stays covered
            forced = self.lure_bites(nb, m.to_square)
            c_, why = self.move_cost(b, m)
            given = max(0.0, taking)
            cost = c_ + given + (0.0 if forced else 0.5)
            if given > 0:
                why = why + [f"gives the {chess.piece_name(b.piece_at(m.from_square).piece_type)}"]
            out.append(Way([m], self.san(b, m), f"a lure on {chess.square_name(m.to_square)}: {chess.piece_name(cp.piece_type)} takes and {chess.square_name(sq)} is free"
                           + ("" if forced else " (it may decline)"), cost=cost, hops=2, costs=why, reply=nb.san(cap), after=after, boards=[b]))
        out.sort(key=lambda w: (w.cost, w.hops))
        return out[:2]

    def ways_block(self, need: Need, b: chess.Board, depth: int, ex) -> list[Way]:
        c = need.against; sq = need.square; out = []
        cp = b.piece_at(c)
        if cp is None or cp.piece_type not in (chess.BISHOP, chess.ROOK, chess.QUEEN):
            return out
        for bsq in chess.SquareSet.between(c, sq):
            for psq, pp in b.piece_map().items():
                if pp.color != self.side or R.is_royal(b, psq) or psq == need.piece:
                    continue
                sub = self.plan(Need("reach", bsq, piece=psq), b, depth + 1, ex)
                if sub.best is not None:
                    out.append(Way([], "", f"interpose on {chess.square_name(bsq)}", [sub], hops=len(sub.steps()), after=sub.after))
                    return out
        return out

    # ---- helpers
    def san(self, b: chess.Board, m: chess.Move) -> str:
        bb = b.copy(stack=False); bb.turn = self.side
        return bb.san(m)

    def san_seq(self, b: chess.Board, moves: list[chess.Move]) -> list[str]:
        bb = b.copy(stack=False); out = []
        for m in moves:
            bb.turn = self.side; out.append(bb.san(m)); bb.push(m)
        return out

    def capture_move(self, b: chess.Board, t: int) -> chess.Move | None:
        """The cheapest capture of the piece on t that counts, on this board, or None."""
        bb = b.copy(stack=False); bb.turn = self.side
        d = defense(bb, t, memo=self.memo)
        if d is None or not d.falls:
            return None
        for a in sorted(d.effective_attackers, key=lambda a: value(bb, a)):
            for m in (chess.Move(a, t), chess.Move(a, t, promotion=chess.QUEEN)):
                if m in bb.legal_moves:
                    return m
        return None

    def attack_squares(self, b: chess.Board, psq: int, t: int) -> list[int]:
        """Squares from which the piece on psq would attack t, nearest by geometry first."""
        p = b.piece_at(psq); out = []
        for sq in chess.SQUARES:
            if sq == t or (b.piece_at(sq) is not None and sq != psq):
                continue
            bb = b.copy(stack=False); bb.remove_piece_at(psq); bb.set_piece_at(sq, p)
            if t in bb.attacks(sq):
                out.append(sq)
        out.sort(key=lambda s: chess.square_distance(psq, s))
        return out[:6]

    def geometric_path(self, b: chess.Board, psq: int, sq: int, max_hops: int) -> list[int] | None:
        """Fewest quiet moves ignoring safety (occupancy respected)."""
        p = b.piece_at(psq)
        q = deque([(b.copy(stack=False), psq, [])]); seen = {psq}
        while q:
            bb, cur, path = q.popleft()
            if len(path) >= max_hops:
                continue
            bb.turn = p.color
            for m in bb.legal_moves:
                if m.from_square != cur or bb.is_capture(m) or m.to_square in seen:
                    continue
                nb = bb.copy(stack=False); nb.push(m); nb.turn = p.color
                np = path + [m.to_square]
                if m.to_square == sq:
                    return np
                seen.add(m.to_square); q.append((nb, m.to_square, np))
        return None

    def safe_path(self, b: chess.Board, psq: int, sq: int, max_hops: int, capture_last: bool = False) -> list[int] | None:
        """Fewest quiet moves to sq with every landing safe; with capture_last, the final hop may be a capture on sq
        that does not lose."""
        p = b.piece_at(psq)
        q = deque([(b.copy(stack=False), psq, [])]); seen = {psq}
        while q:
            bb, cur, path = q.popleft()
            if len(path) >= max_hops:
                continue
            bb.turn = p.color
            for m in bb.legal_moves:
                if m.from_square != cur or m.to_square in seen:
                    continue
                if bb.is_capture(m):
                    if not (capture_last and m.to_square == sq and see(bb, sq, p.color) >= 0):
                        continue
                nb = bb.copy(stack=False); nb.push(m); nb.turn = p.color
                if self.unsafe(nb, m.to_square, m.to_square):
                    continue                                   # it would be taken there, even in an even trade: it never arrives
                np = path + [m.to_square]
                if m.to_square == sq:
                    return np
                seen.add(m.to_square); q.append((nb, m.to_square, np))
        return None

    def path_moves(self, b: chess.Board, psq: int, path: list[int]) -> list[chess.Move]:
        bb = b.copy(stack=False); bb.turn = self.side; cur = psq; out = []
        for to in path:
            m = next((m for m in bb.legal_moves if m.from_square == cur and m.to_square == to), None)
            if m is None:
                return out
            out.append(m); bb.push(m); bb.turn = self.side; cur = to
        return out

    def unsafe(self, b: chess.Board, psq: int, sq: int) -> bool:
        """Would the piece be taken on sq? For a piece that has to ARRIVE, an even trade is as bad as a loss: the
        square is unsafe unless nothing that counts attacks it, or taking loses for them."""
        p = b.piece_at(psq)
        if p is None:
            return False
        bb = b.copy(stack=False)
        if psq != sq:
            bb.remove_piece_at(psq); bb.set_piece_at(sq, p)
        bb.turn = not self.side
        d = defense(bb, sq, memo=self.memo)
        return d is not None and bool(d.effective_attackers) and d.see >= 0

    def coverers(self, b: chess.Board, psq: int, sq: int) -> list[int]:
        """The enemy pieces that make sq unsafe for the piece: those without which it would be safe (one at a time)."""
        p = b.piece_at(psq)
        bb = b.copy(stack=False); bb.remove_piece_at(psq); bb.set_piece_at(sq, p)
        atts = [a for a in bb.attackers(not self.side, sq)]
        out = []
        for a in sorted(atts, key=lambda a: value(b, a)):
            b2 = bb.copy(stack=False); b2.remove_piece_at(a); b2.turn = not self.side
            k = b2.king(not self.side)
            if k is not None and b2.is_attacked_by(self.side, k):
                continue                                       # it shields its king (a pinned piece): not a coverer, it cannot take anyway
            d = defense(b2, sq, memo=self.memo)
            if d is None or not d.falls:
                out.append(a)
        return out

    def lure_bites(self, nb: chess.Board, lure_sq: int) -> bool:
        """Does the coverer have to take? One level: the lure wins material if left (it falls to them for free), or the
        lure itself attacks something of theirs worth more than it."""
        d = defense(nb, lure_sq, memo=self.memo)
        if d is not None and d.falls:
            return True
        lv = value(nb, lure_sq)
        for t in nb.attacks(lure_sq):
            tp = nb.piece_at(t)
            if tp is not None and tp.color != self.side and value(nb, t) > lv:
                return True
        return False

    def flip_gain(self, b: chess.Board, t: int, psq: int) -> float:
        d = defense(b, t, memo=self.memo)
        tv = value(b, t)
        atts = [value(b, q) for q in (d.effective_attackers if d else [])] + [value(b, psq)]
        defs = [value(b, q) for q in (d.effective if d else b.attackers(not self.side, t))]
        return max(_see_lists(tv, atts, defs), 0.0)

    # ---- the race and the verifier
    def answer(self, line: Line) -> Line:
        """The race: at the moment the threat is real (after our preparation, their expected replies played), what is
        the opponent's fastest way to hold the target, and do they have the tempi for it? They get one free move per
        preparatory step of ours, less the replies our lures forced."""
        pairs = line.step_pairs()
        if not pairs or line.need.kind != "win":
            return line
        k = len(pairs)
        forced = sum(1 for w in self.ways_along(line) if w.reply)
        free = max(0, (k - 1) - forced)
        threat_board = pairs[-1][1].copy(stack=False)          # before our final step
        threat_board.turn = not self.side
        t = line.need.square
        if threat_board.piece_at(t) is None:
            return line
        cands = derive(threat_board, not self.side, Goal("defend", t), self.memo, depth=1)
        cands.sort(key=lambda p: p.hops)
        if cands:
            a = cands[0]
            line.answer = f"{' '.join(s.san for s in a.steps)} in {a.hops} ({a.chain[-1].strip()})"
            line.race = "met" if a.hops <= free else "wins"
        else:
            line.race = "wins"
        line.free_tempi = free
        return line

    def ways_along(self, line: Line) -> list[Way]:
        if line.best is None:
            return []
        out = []
        for s in line.best.subs:
            out += self.ways_along(s)
        return out + [line.best]

    def verify(self, line: Line, engine, budget: int = 300, engine_depth: int = 8) -> Line:
        from .goal_search import GoalSearch, result_text
        steps = line.steps()
        if not steps or line.need.kind != "win" or engine is None:
            return line
        gs = GoalSearch(self.board, self.side, line.need.square, memo=self.memo, engine=engine, horizon=min(10, 2 * len(steps) + 2),
                        budget=budget, engine_depth=engine_depth, prior_moves=steps[:1])
        r = gs.run()
        line.status = "verified" if r.proven else "refuted"
        line.verification = result_text(r)
        return line

    # ---- prose
    def line_text(self, line: Line, short: bool = False) -> str:
        b = self.board
        head = line.need.text(b)
        head = head[0].upper() + head[1:]
        if line.best is None:
            return f"{head}: given up ({line.reason})" + ("" if short or not line.ways else "; tried " + "; ".join(w.why or w.san for w in line.ways[:3]))
        w = line.best
        parts = []
        if w.san:
            parts.append(w.san + (f" (then {w.reply})" if w.reply else ""))
        if w.why:
            parts.append(w.why)
        s = f"{head}: {' -- '.join(parts)}" if parts else head
        if w.costs:
            s += f" (costs: {', '.join(w.costs)})"
        if line.need.kind == "win":
            s += f". Gain {w.gain:g}, cost {line.cost:g}, net {w.gain - line.cost:+g}"
            if line.answer:
                s += f". Their answer: {line.answer}, with {line.free_tempi} free move{'s' if line.free_tempi != 1 else ''}"
            s += (": they are in time" if line.race == "met" else ": the race is ours" if line.race == "wins" else "")
        return s + "."

    def tree_text(self, line: Line, indent: int = 0) -> list[str]:
        out = ["  " * indent + self.line_text(line)]
        if line.best is not None:
            for s in line.best.subs:
                out += self.tree_text(s, indent + 1)
        elif line.ways:
            for w in line.ways[:2]:
                for s in w.subs:
                    out += self.tree_text(s, indent + 1)
        return out


def plan_win(board: chess.Board, side: chess.Color, target: int, memo=None, engine=None, verify: bool = True) -> tuple[Line, Planner]:
    pl = Planner(board, side, memo)
    line = pl.plan(Need("win", target))
    pl.answer(line)
    if verify and engine is not None and line.best is not None:
        pl.verify(line, engine)
    return line, pl


# ---------------------------------------------------------------- the idea loop: reasons to move, destinations, what each creates
def ideas_for(board: chess.Board, side: chess.Color, memo=None, engine=None, top: int = 4) -> list[tuple[chess.Move, str, Line]]:
    """For each own piece with a reason to move (attacked, or undeveloped), each quiet destination, the win needs it
    creates (a target that falls or becomes joinable after the move); planned from the position after the move with
    the move as the first step, ranked by net. The trigger-destination-idea-verdict loop."""
    from .reach import progress
    b = board.copy(stack=False); b.turn = side
    memo = memo if memo is not None else Memo()
    before = {t.square: t.status for t in progress(b, side, top=64, memo=memo)}
    out = []
    back = chess.BB_RANK_1 if side == chess.WHITE else chess.BB_RANK_8
    for psq, pp in b.piece_map().items():
        if pp.color != side or pp.piece_type in (chess.PAWN, chess.KING):
            continue
        d = defense(b, psq, memo=memo)
        attacked = d is not None and d.attackers
        undeveloped = pp.piece_type in (chess.KNIGHT, chess.BISHOP) and chess.BB_SQUARES[psq] & back
        if not attacked and not undeveloped:
            continue
        reason = "it is attacked" if attacked else "it is undeveloped"
        for m in b.legal_moves:
            if m.from_square != psq or b.is_capture(m):
                continue
            nb = b.copy(stack=False); nb.push(m); nb.turn = side
            dd = defense(nb, m.to_square, memo=memo)
            if dd is not None and dd.falls:
                continue
            after = {t.square: t.status for t in progress(nb, side, top=64, memo=memo)}
            targets = {t for t, st in after.items() if st in ("falls", "joinable") and before.get(t) not in ("falls", "joinable")}
            targets |= {t for t in nb.attacks(m.to_square) if nb.piece_at(t) and nb.piece_at(t).color != side and not R.is_royal(nb, t)}
            for t in sorted(targets, key=lambda t: -value(nb, t)):
                line, pl = plan_win(nb, side, t, memo, engine, verify=False)
                if line.best is not None:
                    out.append((m, reason, line))
    out.sort(key=lambda x: -x[2].net)
    return out[:top]


def idea_lines(board: chess.Board, side: chess.Color, memo=None, top: int = 3) -> list[str]:
    """The idea loop as prose: 'Bb5 (it is undeveloped): win the knight on c6 -- ... they are in time.'"""
    try:
        out = []
        b = board.copy(stack=False); b.turn = side
        for m, reason, line in ideas_for(board, side, memo, None, top=top):
            out.append(f"{b.san(m)} ({reason}): {Planner(board, side, memo).line_text(line)}")
        return out
    except Exception:
        return []


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--fen", required=True); ap.add_argument("--target", default=""); ap.add_argument("--engine", action="store_true")
    ap.add_argument("--ideas", action="store_true")
    a = ap.parse_args(argv)
    b = chess.Board(a.fen)
    eng = None
    if a.engine:
        from .engine import Engine
        eng = Engine(depth=12)
    if a.target:
        line, pl = plan_win(b, b.turn, chess.parse_square(a.target), Memo(), eng)
        print("\n".join(pl.tree_text(line)))
        if line.verification:
            print("verify:", line.verification)
        print(f"[needs] {pl.expanded} needs expanded")
    if a.ideas or not a.target:
        for m, reason, line in ideas_for(b, b.turn, Memo(), eng):
            bb = b.copy(); print(f"{bb.san(m)} ({reason}): {Planner(b, b.turn).line_text(line)}")
    if eng:
        eng.close()


if __name__ == "__main__":
    main()
