"""reach.py -- WHERE CAN WE MAKE PROGRESS (Kaveh 2026-09-05): the tactics that are NOT executable yet, and what
would make them executable. Two operators over the atoms and the defense ledger, nothing named by hand:

  flip conditions   for a HELD target (the exchange says it does not fall), what changes the answer: one more
                    attacker worth at most v, or one defender gone. The exchange fold over a modified ledger.
  safe reach        for a piece and a target square, the fewest own quiet moves to a square from which it attacks
                    (or defends) the target, landing only where the defense operator says it survives. Walks the
                    piece's move graph FROM THE RULES on a scratch board, so a light-squared bishop never reaches
                    a dark-squared pawn, a king never steps next to a defended piece, and a variant's movement
                    changes the answer without anyone telling it.

Composed: the PROGRESS MAP of a side -- every enemy target as 'falls now', 'falls after <who> joins in <n> moves',
'falls if <defender> is removed', or 'unreachable with this material', and the mirror for our own pieces. This is
the substrate of the plan layer (ANALYST_PLAN §14): a plan is a reinforcement path whose ledger flips, and what an
even trade 'does to the plans' is which targets became reachable or unreachable.

Reach ignores the opponent's replies; verify_plan() gives one side its moves in a row (null moves for the other) and
then asks whether the ledger actually flipped, and whether the opponent's best reply to the first hop stops it."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import chess

from . import rules as R
from .defense import defense, Defense
from .facts import value, see


# ---------------------------------------------------------------- flip conditions
def _see_lists(target: float, attackers: list[float], defenders: list[float]) -> float:
    """The swap list over explicit value lists (attackers capture first, both sides least-valuable-first, free to stop)."""
    a, d = sorted(attackers), sorted(defenders)
    gains = [target]
    side_att = True
    while True:
        pool = a if side_att else d
        if not pool:
            break
        piece = pool.pop(0)
        gains.append(piece)
        side_att = not side_att
        if len(gains) > 32:
            break
    n = len(gains) - 1
    if n == 0:
        return 0.0
    score = [0.0] * (n + 1)
    for i in range(n - 1, -1, -1):
        score[i] = gains[i] - max(0.0, score[i + 1]) if i + 1 <= n - 1 else gains[i]
    return score[0]


@dataclass(frozen=True)
class Flip:
    kind: str                   # "add_attacker" | "remove_defender"
    max_value: float = 0.0      # add_attacker: one more attacker worth at most this flips the exchange
    square: int | None = None   # remove_defender: which defender


def flip_conditions(board: chess.Board, target: int, by: chess.Color, memo=None) -> list[Flip]:
    """What would make the piece on `target` fall to `by`: the cheapest single change of the ledger."""
    d = defense(board, target, memo=memo)
    if d is None:
        # nothing attacks it yet: one attacker alone wins it iff the target is worth more than what stands to be lost
        piece = board.piece_at(target)
        if piece is None or piece.color == by:
            return []
        defs = [value(board, q) for q in board.attackers(not by, target)]
        out = []
        for v in (1.0, 3.0, 5.0, 9.0):
            if _see_lists(value(board, target), [v], defs) > 0:
                out.append(Flip("add_attacker", v)); break
        return out
    if d.falls:
        return []
    tv = value(board, target)
    atts = [value(board, q) for q in d.effective_attackers]          # an attacker that cannot take does not join
    defs = [value(board, q) for q in d.effective]
    out = []
    best_v = None
    for v in (9.0, 5.0, 3.0, 1.0):                       # the LARGEST value that still flips is the loosest condition
        if _see_lists(tv, atts + [v], defs) > 0:
            best_v = v; break                              # (a cheaper attacker flips whenever a dearer one does)
    if best_v is not None:
        out.append(Flip("add_attacker", best_v))
    for q in d.effective:
        if see(board, target, by, exclude={q} | d.excluded) > 0:
            out.append(Flip("remove_defender", 0.0, q))
    return out


# ---------------------------------------------------------------- safe reach
@dataclass(frozen=True)
class Path:
    piece: int                  # where it starts
    squares: tuple[int, ...]    # the landing squares, in order (len = hops)

    @property
    def hops(self) -> int:
        return len(self.squares)


def _quiet_moves(board: chess.Board, sq: int) -> list[chess.Move]:
    """The piece on sq's legal quiet moves (its side to move), by the rules of this board."""
    b = board.copy(stack=False); b.turn = board.piece_at(sq).color
    try:
        return [m for m in b.legal_moves if m.from_square == sq and not b.is_capture(m) and m.promotion in (None, chess.QUEEN)]
    except Exception:
        return []


def _attacks_from(board: chess.Board, sq: int, target: int) -> bool:
    return target in board.attacks(sq)


def reach(board: chess.Board, piece_sq: int, target: int, max_hops: int = 4, memo=None) -> Path | None:
    """Fewest quiet moves for the piece on piece_sq to a square from which it attacks `target` (or defends it, when
    the target is its own piece), each landing square safe by the defense operator. None when it cannot.
    With `memo`, served from the concept store while the pieces on its footprint (the piece, the target, every
    square it could move through and every square that decides the safety of a landing) are unchanged."""
    if memo is not None:
        return memo.get("reach", (piece_sq, target, max_hops), board, lambda r: _reach(board, piece_sq, target, max_hops, r))
    return _reach(board, piece_sq, target, max_hops, None)


def _reach(board: chess.Board, piece_sq: int, target: int, max_hops: int, reads) -> Path | None:
    piece = board.piece_at(piece_sq)
    if reads is not None:
        reads.square(piece_sq); reads.square(target); reads.moves_of(board, piece_sq)
    if piece is None:
        return None
    c = piece.color
    if _attacks_from(board, piece_sq, target):
        return Path(piece_sq, ())
    start = board.copy(stack=False)
    q = deque([(start, piece_sq, ())])
    seen = {piece_sq}
    while q:
        b, sq, path = q.popleft()
        if len(path) >= max_hops:
            continue
        if reads is not None:
            reads.moves_of(b, sq)
        for m in _quiet_moves(b, sq):
            to = m.to_square
            if to in seen:
                continue
            nb = b.copy(stack=False); nb.turn = c; nb.push(m)
            if reads is not None:
                reads.moves_of(nb, to)                   # what it would attack from there
            d = defense(nb, to, reads=reads)             # would it ARRIVE there? a square where it can be taken at no loss
            if d is not None and d.effective_attackers and d.see >= 0:     # (an even trade included) is no path for a joiner
                continue
            seen.add(to)
            npath = path + (to,)
            if _attacks_from(nb, to, target):
                return Path(piece_sq, npath)
            q.append((nb, to, npath))
    return None


# ---------------------------------------------------------------- the progress map
@dataclass
class Target:
    square: int
    piece: str                      # symbol
    status: str                     # "falls" | "joinable" | "undermine" | "unreachable" | "held"
    ledger: Defense | None
    flips: list[Flip] = field(default_factory=list)
    joiner: Path | None = None      # the cheapest piece that can join (status joinable)
    joiners: list[Path] = field(default_factory=list)
    remove: int | None = None       # status undermine: the defender to remove (it falls itself)
    cannot: list[str] = field(default_factory=list)    # why nothing reaches it (per own piece, when unreachable)
    footprint: int = 0              # the squares this target's ledger and reach read (with a recording memo)


def progress(board: chess.Board, by: chess.Color, max_hops: int = 3, top: int = 6, memo=None) -> list[Target]:
    """Every enemy piece as a target of `by`, most valuable first: what it takes to win it. `memo` serves the
    defense ledgers and the reach paths from the concept store."""
    out = []
    own = [(sq, p) for sq, p in board.piece_map().items() if p.color == by and not R.is_royal(board, sq)]
    for sq, p in sorted(board.piece_map().items(), key=lambda kv: -value(board, kv[0])):
        if p.color == by or R.is_royal(board, sq):
            continue
        if memo is not None:
            with memo.recording() as acc:
                t = _target(board, by, sq, p, own, max_hops, memo)
            t.footprint = acc[0]
        else:
            t = _target(board, by, sq, p, own, max_hops, memo)
        out.append(t)
    return out[:top]


def _target(board: chess.Board, by: chess.Color, sq: int, p: chess.Piece, own, max_hops: int, memo) -> Target:
    if True:
        d = defense(board, sq, memo=memo)
        t = Target(sq, p.symbol(), "held", d)
        if d is not None and d.falls:
            t.status = "falls"; return t
        t.flips = flip_conditions(board, sq, by, memo=memo)
        add = next((f for f in t.flips if f.kind == "add_attacker"), None)
        if add is not None:
            attacking_now = set(d.attackers) if d else set()
            paths = []
            for psq, pp in own:
                if psq in attacking_now or value(board, psq) > add.max_value:
                    continue
                pth = reach(board, psq, sq, max_hops, memo=memo)
                if pth is not None and pth.hops > 0:
                    paths.append(pth)
            paths.sort(key=lambda x: (x.hops, value(board, x.piece)))
            if paths:
                t.status = "joinable"; t.joiner = paths[0]; t.joiners = paths[:3]; return t
        for f in t.flips:
            if f.kind == "remove_defender":
                dd = defense(board, f.square, memo=memo)
                if dd is not None and dd.falls:
                    t.status = "undermine"; t.remove = f.square; break
        if t.status == "undermine":
            return t
        t.status = "unreachable" if not (d and d.attackers) else "held"
        return t


# ---------------------------------------------------------------- prose
def _pc(sym: str, sq: int) -> str:
    from .verbalize import pc
    return pc(sym, sq)


def ledger_phrase(board: chess.Board, d) -> str:
    """'attacked 2, effectively defended 1' plus, when an attacker does not count, why: 'the white pawn on b4 cannot
    take: its departure hangs the white rook on a4'."""
    if d is None:
        return "unattacked"
    s = f"attacked {len(d.attackers)}, effectively defended {len(d.effective)}"
    for x in d.attackers_out:
        p = board.piece_at(x.square)
        if p is None:
            continue
        who = _pc(p.symbol(), x.square)
        if x.reason == "exposes" and x.exposed is not None and board.piece_at(x.exposed):
            s += f"; {who} cannot take: its departure hangs {_pc(board.piece_at(x.exposed).symbol(), x.exposed)}"
        elif x.reason == "illegal":
            s += f"; {who} cannot legally take"
    return s


def counts(d) -> str:
    """The compact '2 v 1': attackers that count against defenders that count."""
    return f"{len(d.effective_attackers)} v {len(d.effective)}"


def target_phrase(board: chess.Board, t: Target) -> str:
    who = _pc(t.piece, t.square)
    d = t.ledger
    ledger = ledger_phrase(board, d)
    if t.status == "falls":
        return f"{who} falls now ({ledger})"
    if t.status == "joinable" and t.joiner:
        j = t.joiner; jp = board.piece_at(j.piece)
        via = " via " + " ".join(chess.square_name(s) for s in j.squares[:-1]) if len(j.squares) > 1 else ""
        return (f"{who} is held ({ledger}); {_pc(jp.symbol(), j.piece)} joins from {chess.square_name(j.squares[-1])} "
                f"in {j.hops} move{'s' if j.hops != 1 else ''}{via} and it falls")
    if t.status == "undermine" and t.remove is not None:
        r = board.piece_at(t.remove)
        return f"{who} is held ({ledger}); its defender {_pc(r.symbol(), t.remove)} falls first, then it does"
    if t.status == "unreachable":
        return f"{who} cannot be reached: nothing can attack it within 3 moves"
    add = next((f for f in t.flips if f.kind == "add_attacker"), None)
    need = f" (one more attacker worth {add.max_value:g} or less would win it, nothing can get there)" if add else " (no single reinforcement changes it)"
    return f"{who} is held ({ledger}){need}"


def progress_lines(board: chess.Board, by: chess.Color, top: int = 3, memo=None) -> list[str]:
    """The side's progress map as prose: what falls, who can join where, what can be undermined; then one line for
    what is out of reach (so 'nothing to do here' is said once, not per piece)."""
    ts = progress(board, by, top=64, memo=memo)
    order = {"falls": 0, "joinable": 1, "undermine": 2, "held": 3, "unreachable": 4}
    ts.sort(key=lambda t: (order[t.status], -value(board, t.square)))
    live = [t for t in ts if t.status in ("falls", "joinable", "undermine")][:top]
    out = [target_phrase(board, t) for t in live]
    rest = [t for t in ts if t.status in ("held", "unreachable")]
    if rest and len(out) < top:
        held = [t for t in rest if t.status == "held"][:3]; far = [t for t in rest if t.status == "unreachable"]
        bits = []
        if held:
            bits.append("held: " + ", ".join(f"{_pc(t.piece, t.square)} ({counts(t.ledger)})" for t in held))
        if far:
            bits.append("out of reach within 3 moves: " + ", ".join(_pc(t.piece, t.square) for t in far[:4]) + (f" and {len(far)-4} more" if len(far) > 4 else ""))
        out.append("; ".join(bits))
    return out


# ---------------------------------------------------------------- verification with the engine
@dataclass
class Verified:
    target: int
    path: Path
    unopposed: bool             # the ledger flips after our moves with the opponent passing
    reply: str | None = None    # the opponent's best reply to the first hop (SAN)
    outcome: str | None = None  # after that reply: "wins" (still winnable within the remaining hops), "chased" (the
                                # target moved away: the plan gained a tempo, not the piece), "met" (they defend it)


def verify_plan(board: chess.Board, engine, t: Target, depth: int | None = None) -> Verified | None:
    """Give our side its hops in a row (null moves for the opponent), check the ledger; then let the engine answer
    the first hop and re-run the progress test from there."""
    if t.status != "joinable" or t.joiner is None:
        return None
    by = board.piece_at(t.joiner.piece).color
    b = board.copy(stack=False); b.turn = by
    cur = t.joiner.piece
    for to in t.joiner.squares:
        mv = next((m for m in b.legal_moves if m.from_square == cur and m.to_square == to), None)
        if mv is None:
            return None
        b.push(mv); cur = to
        if not b.is_game_over():
            b.push(chess.Move.null())
    d = defense(b, t.square)
    v = Verified(t.square, t.joiner, bool(d and d.falls))
    b2 = board.copy(stack=False); b2.turn = by
    first = next((m for m in b2.legal_moves if m.from_square == t.joiner.piece and m.to_square == t.joiner.squares[0]), None)
    if first is None or engine is None:
        return v
    b2.push(first)
    if b2.is_game_over():
        return v
    best = engine.best(b2, depth=depth) if depth else engine.best(b2)
    if best is None:
        return v
    reply = chess.Move.from_uci(best.move); v.reply = b2.san(reply); b2.push(reply)
    tp = b2.piece_at(t.square)
    if tp is None or tp.color == by or tp.symbol() != t.piece:
        v.outcome = "chased"; return v
    after = [x for x in progress(b2, by, max_hops=max(1, t.joiner.hops - 1), top=64) if x.square == t.square]
    v.outcome = "wins" if after and after[0].status in ("falls", "joinable") else "met"
    return v
