"""PLAN 78.68 (first run), SYSTEM_SPEC 9 "the ladder on a position" and "what the oracle may and may not do": THE ORACLE'S ANSWERS, the labelling step. This is the only module of the ladder's files that
names the game (the oracle is a chess engine and speaks in the game's move names); the ladder itself (ladder.py) and the measure script (ladder_measure.py) never do.

INTERFACE (what this module takes, what it returns, who calls it).
  label(moves, budget)   takes the moves played from the start of the game and the oracle's budget (ORACLE_DEPTH, ORACLE_LINES, SEVERAL_MARGIN), and returns {"accepted": [move names]}: the oracle's best move and every other
                         move among its first ORACLE_LINES that its own reading puts within SEVERAL_MARGIN of the best ("several best": a hit on any of them counts). THE WALL: the oracle's numbers are used
                         only inside this function, to decide the set; nothing of them leaves it (no score, no depth, no line). Called by ladder_measure.py's `label` stage, which writes the answers file.
  The engine and the tablebase are those of audit_oracle.py (`_init`: the UCI engine named stockfish, one thread, the Syzygy path of catspace.io.paths), reused as it stands.
"""
from __future__ import annotations

ORACLE_DEPTH = 14                 # the fixed modest budget: search depth in plies (a stated choice of this run)
ORACLE_LINES = 5                  # the lines asked for (the first move of each is a candidate for "several best")
SEVERAL_MARGIN = 30               # hundredths of a unit: lines within this of the best are "several best" (a stated choice of this run; the oracle's own scale, used only inside `label`)
MATE_SCORE = 100000

_READY = False


def _engine():
    global _READY
    from catspace.incubator.analyst import audit_oracle as AO
    if not _READY:
        AO._init()
        _READY = True
    return AO._ENG


def board_of(moves: list):
    import chess
    b = chess.Board()
    for m in moves:
        b.push_uci(m)
    return b


def label(moves: list, depth: int = ORACLE_DEPTH, lines: int = ORACLE_LINES, margin: int = SEVERAL_MARGIN) -> dict:
    import chess.engine
    b = board_of(moves)
    n = b.legal_moves.count()
    infos = _engine().analyse(b, chess.engine.Limit(depth=depth), multipv=min(lines, n))
    if isinstance(infos, dict):
        infos = [infos]
    scored = []
    for inf in infos:
        pv = inf.get("pv") or []
        if not pv:
            continue
        scored.append((inf["score"].pov(b.turn).score(mate_score=MATE_SCORE), pv[0].uci()))
    scored.sort(key=lambda x: -x[0])
    top = scored[0][0]
    return {"accepted": [m for s, m in scored if top - s <= margin]}
