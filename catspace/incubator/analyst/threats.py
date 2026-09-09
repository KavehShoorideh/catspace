"""threats.py -- a threat with its REQUIRED DEFENCES (Kaveh 2026-09-07): "when I create a threat I should think from
the opponent's side: what must they do? If they do something else, my analysis should immediately say take the
winning move -- no extra search, I already have the answer."

A threat is a goal that our move has made forced against all but a few replies. The goal search's tree has this
already: at the AND node after our move, the replies the judge settled at once are the ones that lose the target,
and the replies that refuted the proof (or stayed open) are the DEFENCES. Everything else is a mistake, and the
proof stores the move that punishes it (the sketch). So the record of a threat is: what it wins, what they must
play, and for every other reply, what we play. Mate in one is the same with the rules as judge.

    python -m catspace.incubator.analyst.threats --fen "<fen after our move>" [--engine]
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field

import chess

from . import rules as R
from .defense import defense
from .facts import value
from .memo import Memo
from .subgames import mate_in_one


@dataclass
class Threat:
    kind: str                                   # "mate" | "win"
    target: int | None                          # the piece we win (kind win)
    text: str                                   # "mate on f7" / "win the knight on c6"
    defences: dict[str, str] = field(default_factory=dict)      # reply SAN -> what it does ("defends f7", "blocks the diagonal", ...)
    punishes: dict[str, str] = field(default_factory=dict)      # reply SAN -> our move SAN that executes the threat
    value: float = 0.0

    def verdict(self, reply_san: str) -> tuple[str, str | None]:
        """('parried', how) when the reply is a defence; ('executes', our move) when it is not; ('unknown', None)."""
        if reply_san in self.defences:
            return "parried", self.defences[reply_san]
        if reply_san in self.punishes:
            return "executes", self.punishes[reply_san]
        return "unknown", None


def describe_defence(board: chess.Board, reply: chess.Move, target: int | None, attacker_from: int | None, mate_square: int | None = None) -> str:
    """What the reply does against the threat, in the ledger's terms."""
    b = board
    after = b.copy(stack=False); after.push(reply)
    who = lambda bb, sq: f"the {chess.piece_name(bb.piece_at(sq).piece_type)} on {chess.square_name(sq)}" if bb.piece_at(sq) else chess.square_name(sq)
    if b.gives_check(reply):
        return "counters with check"
    if target is not None:
        if reply.from_square == target:
            return f"moves {who(b, target)} away"
        if b.is_capture(reply) and attacker_from is not None and reply.to_square == attacker_from:
            return "takes the attacker"
        before = len(b.attackers(b.turn, target)); now = len(after.attackers(b.turn, target))
        if now > before:
            return f"defends {who(b, target)}"
        if attacker_from is not None and reply.to_square in chess.SquareSet.between(attacker_from, target):
            return "blocks the line"
        if b.is_capture(reply):
            return "captures elsewhere"
    else:
        k = b.king(b.turn)
        if reply.from_square == k:
            return "the king steps away"
        if b.is_capture(reply):
            return "takes the attacker" if attacker_from is not None and reply.to_square == attacker_from else "captures"
        if mate_square is not None:
            for a in b.attackers(not b.turn, mate_square):
                if reply.to_square in chess.SquareSet.between(a, mate_square):
                    return f"blocks the line to {chess.square_name(mate_square)}"
            if len(after.attackers(b.turn, mate_square)) > len(b.attackers(b.turn, mate_square)):
                return f"defends {chess.square_name(mate_square)}"
            return f"gives the king a flight square"
        return "covers the mating square or blocks"
    return "a quiet defence"


# ---------------------------------------------------------------- mate threats
def mate_threat(board: chess.Board, us: chess.Color) -> Threat | None:
    """After our move (them to move): do we threaten mate in one, and what must they do about it?"""
    if board.turn == us:
        return None
    mates = mate_in_one(board, us)
    if not mates:
        return None
    m0 = mates[0]
    t = Threat("mate", None, f"mate ({_san_as(board, us, m0)})", value=100.0)
    for reply in board.legal_moves:
        nb = board.copy(stack=False); nb.push(reply)
        if nb.is_game_over():
            continue
        still = mate_in_one(nb, us)
        san = board.san(reply)
        if still:
            t.punishes[san] = nb.san(still[0])
        else:
            t.defences[san] = describe_defence(board, reply, None, m0.from_square, m0.to_square)
    return t


def _san_as(board: chess.Board, side: chess.Color, m: chess.Move) -> str:
    b = board.copy(stack=False); b.turn = side
    return b.san(m) if m in b.legal_moves else m.uci()


# ---------------------------------------------------------------- material threats from the goal search's tree
def win_threat(board_before: chess.Board, our_move: chess.Move, target: int, memo=None, engine=None, budget: int = 300,
               horizon: int = 6, engine_depth: int = 8) -> Threat | None:
    """Our move as the first move of a goal search: the replies it answers are punished, the replies that hold are
    the defences. Returns None when the move creates no such threat (every reply holds, or the goal needs no reply
    at all -- then it is not a threat but a win, and the search's own result says so)."""
    from .goal_search import GoalSearch, INF
    memo = memo if memo is not None else Memo()
    side = board_before.turn
    gs = GoalSearch(board_before, side, target, memo=memo, engine=engine, horizon=horizon, budget=budget, engine_depth=engine_depth,
                    prior_moves=[our_move])
    r = gs.run()
    root = gs.root
    if root is None or not root.children:
        return None
    node = next((c for c in root.children if c.move == our_move), None)
    if node is None or not node.children:
        return None
    after = node.board
    tp = board_before.piece_at(target)
    t = Threat("win", target, f"win the {chess.piece_name(tp.piece_type)} on {chess.square_name(target)}", value=value(board_before, target))
    for c in node.children:
        san = after.san(c.move)
        if c.pn == 0:                                              # settled or proven: the reply loses the target
            ours = next((k for k in (c.children or []) if k.pn == 0), None)
            if ours is not None:
                t.punishes[san] = c.board.san(ours.move)
            else:
                mv = _cheapest_capture(c.board, c.target if c.target is not None else target, side)
                t.punishes[san] = c.board.san(mv) if mv else "(takes)"
        else:
            t.defences[san] = describe_defence(after, c.move, target, our_move.to_square)
    if not t.punishes or not t.defences:
        return None
    return t


def _cheapest_capture(board: chess.Board, sq: int, by: chess.Color) -> chess.Move | None:
    b = board.copy(stack=False); b.turn = by
    for a in sorted(b.attackers(by, sq), key=lambda a: value(board, a)):
        for m in (chess.Move(a, sq), chess.Move(a, sq, promotion=chess.QUEEN)):
            if m in b.legal_moves:
                return m
    return None


# ---------------------------------------------------------------- what our last move threatens, and the reply judged
def threats_after(board_before: chess.Board, our_move: chess.Move, memo=None, engine=None, budget: int = 200) -> list[Threat]:
    """The threats our move creates: mate in one, and every enemy piece the move newly attacks or leaves falling."""
    after = board_before.copy(stack=False); after.push(our_move)
    us = board_before.turn
    out = []
    if after.is_game_over():
        return out                                             # the game is over: nothing is threatened any more
    mt = mate_threat(after, us)
    if mt is not None:
        out.append(mt)
    memo = memo if memo is not None else Memo()
    targets = []
    for sq, p in after.piece_map().items():
        if p.color == us or R.is_royal(after, sq):
            continue
        d = defense(after, sq, memo=memo)
        if d is not None and d.falls and sq in after.attacks(our_move.to_square) | set().union(*[after.attacks(a) for a in after.attackers(us, sq)] or [set()]):
            targets.append(sq)
    for sq in sorted(set(targets), key=lambda s: -value(after, s))[:2]:
        t = win_threat(board_before, our_move, sq, memo, engine, budget=budget)
        if t is not None:
            out.append(t)
    return out


def judge_reply(threats: list[Threat], board_after_our_move: chess.Board, reply: chess.Move) -> list[str]:
    """For each standing threat: did the reply meet it? 'they had to ...; they played ...; X executes'."""
    san = board_after_our_move.san(reply)
    out = []
    for t in threats:
        v, how = t.verdict(san)
        musts = ", ".join(list(t.defences)[:5]) + (" ..." if len(t.defences) > 5 else "")
        if v == "parried":
            out.append(f"{t.text}: parried by {san} ({how})")
        elif v == "executes":
            out.append(f"{t.text}: they had to play one of {musts}; {san} does not: {how} executes")
    return out


def threat_lines(board_before: chess.Board, our_move: chess.Move, memo=None, engine=None) -> list[str]:
    try:
        ts = threats_after(board_before, our_move, memo, engine)
    except Exception:
        return []
    out = []
    for t in ts:
        musts = ", ".join(list(t.defences)[:5]) + (" ..." if len(t.defences) > 5 else "")
        out.append(f"threatens {t.text}: they must {musts}" + (f" (any other reply: {next(iter(t.punishes.values()))})" if t.punishes else ""))
    return out


# ---------------------------------------------------------------- the two-sided threat ledger (Kaveh: a threat is a free move
# that improves the position; big threats win a piece with no payback; the goal is a threat with no adequate defence)
def free_move_threats(board: chess.Board, side: chess.Color, memo=None, top: int = 6) -> list[tuple[chess.Move, float, str]]:
    """The moves `side` would make with a free tempo whose SURE GAIN one reply deep is positive (rank.sure_gain: the
    minimum over the opponent's replies of what the ledger then gives), biggest first. With the turn flipped this is
    the null-move scan, by the ledger instead of the engine. Empty when the other side is in check (no free move)."""
    from .rank import sure_gain, attacking_moves
    b = board.copy(stack=False); b.turn = side
    if not b.is_valid():
        return []
    memo = memo if memo is not None else Memo()
    out = []
    mates = mate_in_one(b, side)
    for m in mates[:1]:
        out.append((m, 1000.0, "mate"))
    for m in attacking_moves(b):
        if m in mates:
            continue
        if not pass_gain(b, m, memo):
            continue
        g, why = sure_gain(b, m, memo)
        if g > 0:
            out.append((m, g, why))
    out.sort(key=lambda x: -x[1])
    return out[:top]


def pass_gain(board: chess.Board, m: chess.Move, memo=None) -> bool:
    """The cheap gate before the reply-minimum: after the move, with the opponent PASSING, does the ledger give the
    mover anything (the capture's own exchange plus the best capture then available)? A move that wins nothing even
    against a pass is not a threat; the minimum over real replies is only taken for the survivors. A move that
    gives check (no pass exists) or ends the game passes the gate."""
    from .regime import capture_gain, _best_capture
    side = board.turn
    first = capture_gain(board, m, side, memo) if board.is_capture(m) else 0.0
    nb = board.copy(stack=False); nb.push(m)
    if nb.is_game_over() or nb.is_check():
        return True
    nb.turn = side
    if not nb.is_valid():
        return True
    return first + _best_capture(nb, side, set(chess.scan_forward(nb.occupied)), memo) > 0


def threat_ledger(board: chess.Board, memo=None, top: int = 4) -> dict:
    """Both sides' threats at this position in one currency: 'ours' = the side to move's, 'theirs' = what the other
    side would do with a free move. Big = a piece or more."""
    memo = memo if memo is not None else Memo()
    us = board.turn
    def rows(side):
        b = board.copy(stack=False); b.turn = side
        out = []
        for m, g, why in free_move_threats(board, side, memo, top):
            out.append({"san": b.san(m) if m in b.legal_moves else m.uci(), "gain": g, "why": why, "big": g >= 3.0})
        return out
    return {"ours": rows(us), "theirs": rows(not us)}


def ledger_lines(board: chess.Board, memo=None, top: int = 4) -> list[str]:
    try:
        led = threat_ledger(board, memo, top)
    except Exception:
        return []
    who = "White" if board.turn else "Black"; them = "Black" if board.turn else "White"
    out = []
    if led["ours"]:
        out.append(f"{who} threatens: " + ", ".join(f"{r['san']} ({'mate' if r['gain'] >= 1000 else '+' + format(r['gain'], 'g')}{', big' if r['big'] and r['gain'] < 1000 else ''})" for r in led["ours"]))
    if led["theirs"]:
        out.append(f"{them} would, with a free move: " + ", ".join(f"{r['san']} ({'mate' if r['gain'] >= 1000 else '+' + format(r['gain'], 'g')}{', big' if r['big'] and r['gain'] < 1000 else ''})" for r in led["theirs"]))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--fen", required=True, help="the position BEFORE our move"); ap.add_argument("--move", required=True, help="our move, SAN")
    ap.add_argument("--reply", default="", help="their reply, SAN, to judge"); ap.add_argument("--engine", action="store_true")
    a = ap.parse_args(argv)
    b = chess.Board(a.fen); m = b.parse_san(a.move)
    eng = None
    if a.engine:
        from .engine import Engine
        eng = Engine(depth=12)
    ts = threats_after(b, m, Memo(), eng)
    after = b.copy(); after.push(m)
    for t in ts:
        print(f"{t.text}: defences {dict(list(t.defences.items())[:8])}; punished replies {len(t.punishes)} (e.g. {dict(list(t.punishes.items())[:3])})")
    if a.reply:
        for l in judge_reply(ts, after, after.parse_san(a.reply)):
            print(l)
    if eng:
        eng.close()


if __name__ == "__main__":
    main()
