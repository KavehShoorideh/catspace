"""THE THREE WATCHERS and THE ACTOR (Kaveh 2026-09-23: "three heads watching three outcomes -- my winning, my losing,
my drawing -- posing questions: how can I kick this knight to win, how can I protect this king ... a driver that
influences the prior ... and on top an actor that looks at everything and makes a decision").

Each watcher poses QUESTIONS with a STAKE, from the rules alone (attack sets, the resolved exchange, the null move):
  LOSS   what the opponent takes if we pass (the null move's resolved exchange): "save X" with the loss as the stake
  WIN    what we take now (our best resolved exchanges), and the opportunities the duties readout holds
         ("free P so that M wins"): stake = the gain
  DRAW   the balance that decides whether we should be trading: king exposure (attacks on the ring around each
         king, ours minus theirs) and material -- "cover the king" / "keep pieces" / "trade", stake from the imbalance
A question names the moves that ANSWER it (moves of the target, captures of the attacker, moves that add a defender,
moves into the king's ring), so it can raise those moves' importance in the scheduler and the prior.
THE ACTOR is one function over everything computed: the search value, the resolved exchange, the lemma verdicts, the
duties, the questions answered -- and returns the ranked analysis; choose() takes the top.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import chess

from .player import MATE, Search, material


@dataclass
class Question:
    watcher: str
    text: str
    stake: float
    answers: set = field(default_factory=set)      # moves (uci) that address it


def king_ring(board: chess.Board, side: chess.Color) -> set:
    k = board.king(side)
    return set(chess.SquareSet(chess.BB_KING_ATTACKS[k])) if k is not None else set()


def defenders_after(board: chess.Board, move: chess.Move, sq: int, side: chess.Color) -> int:
    b = board.copy(stack=False); b.push(move)
    return len(b.attackers(side, sq))


def pose(board: chess.Board, search: Search | None = None, app=None) -> list[Question]:
    side = board.turn
    s = search or Search(node_cap=10 ** 9)
    qs = []
    legal = list(board.legal_moves)
    # ---- LOSS: the null move's resolved exchange, and which of our pieces it costs
    if not board.is_check():
        nb = board.copy(stack=False); nb.turn = not side
        if nb.is_valid():
            worst = None
            for r in nb.legal_moves:
                if not (nb.is_capture(r) or nb.gives_check(r)):
                    continue
                nb.push(r); v = s.quiesce(nb, side, -MATE * 2, MATE * 2, 6); nb.pop()
                if worst is None or v < worst[0]:
                    worst = (v, r)
            base = material(board, side)
            if worst and base - worst[0] >= 1.0:
                loss = base - worst[0]; r = worst[1]; target = r.to_square
                q = Question("loss", f"save the {chess.piece_name(board.piece_at(target).piece_type)} on {chess.square_name(target)} from {chess.square_name(r.from_square)}", loss)
                for m in legal:
                    if m.from_square == target or m.to_square == r.from_square:
                        q.answers.add(m.uci())
                    elif defenders_after(board, m, target, side) > len(board.attackers(side, target)):
                        q.answers.add(m.uci())
                    elif chess.SquareSet.between(r.from_square, target) and m.to_square in chess.SquareSet.between(r.from_square, target):
                        q.answers.add(m.uci())
                    elif board.gives_check(m):
                        q.answers.add(m.uci())
                qs.append(q)
    else:
        q = Question("loss", "get out of check", 2.0, {m.uci() for m in legal}); qs.append(q)
    # ---- WIN: our resolved gains now
    gains = []
    for m in legal:
        board.push(m); v = s.quiesce(board, side, -MATE * 2, MATE * 2, 6); board.pop()
        gains.append((v - material(board, side), m))
    gains.sort(key=lambda x: -x[0])
    if gains and gains[0][0] >= 1.0:
        g, m = gains[0]
        q = Question("win", f"win material with {board.san(m)}", g, {mm.uci() for gg, mm in gains if gg >= g - 0.5}); qs.append(q)
    if app is not None:
        from .lemma_apply import duties
        try:
            d = duties(app, board)
            for e, rec in d.items():
                for o in rec.get("opportunities", [])[:1]:
                    qs.append(Question("win", f"free the {chess.piece_name(board.piece_at(e).piece_type)} on {chess.square_name(e)} (held by {','.join(o['held_by'])}) for {o['move']}", 1.0,
                                       {m.uci() for m in legal if defenders_after(board, m, e, side) > len(board.attackers(side, e))}))
        except Exception:
            pass
    # ---- DRAW / safety: king exposure and the material balance
    ours = king_ring(board, side); theirs = king_ring(board, not side)
    exp_us = sum(len(board.attackers(not side, sq)) for sq in ours); exp_them = sum(len(board.attackers(side, sq)) for sq in theirs)
    if exp_us >= 2:
        q = Question("draw", f"cover the king ({exp_us} enemy attacks on its ring)", 0.3 * exp_us, {m.uci() for m in legal if m.to_square in ours or m.from_square == board.king(side) or (board.piece_at(m.from_square).piece_type != chess.KING and any(sq in ours for sq in board.attacks(m.to_square)))})
        qs.append(q)
    if exp_them >= 2:
        q = Question("win", f"press the king ({exp_them} of our attacks on its ring)", 0.2 * exp_them, {m.uci() for m in legal if m.to_square in theirs or board.gives_check(m) or any(sq in theirs for sq in board.attacks(m.to_square))})
        qs.append(q)
    bal = material(board, side)
    if abs(bal) >= 2:
        want_trade = bal > 0
        q = Question("draw", "trade pieces (ahead)" if want_trade else "keep pieces (behind)", 0.3, {m.uci() for m in legal if board.is_capture(m) == want_trade})
        qs.append(q)
    return qs


def actor(board: chess.Board, cands: dict, questions: list[Question], weight: float = 0.15) -> list[dict]:
    """cands: move -> dict(value, resolved, lemma=(keeps, loses)). Returns the ranked analysis: value plus the stakes of
    the questions the move answers, with the reasons. Generic over what the searches computed."""
    rows = []
    for m, c in cands.items():
        answered = [q for q in questions if m.uci() in q.answers]
        bonus = weight * sum(q.stake for q in answered)
        k, l = c.get("lemma", (0, 0))
        rows.append(dict(move=m, san=board.san(m), value=c["value"], resolved=c["resolved"], total=c["value"] + bonus, lemma=(k, l),
                         answers=[f"{q.watcher}: {q.text} ({q.stake:+.1f})" for q in answered]))
    rows.sort(key=lambda r: (-r["total"], -r["resolved"]))
    return rows
