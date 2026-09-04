"""L5 -- IDEAS. Value enters here: Stockfish + the objective (Rules.objective is readable ONLY
from this layer up). For a position:
    IDEA          the best line and what it changes (L4 effects)
    ALTERNATIVES  top-k lines that fail, each with its refutation
    KILLED        candidates that were near-best a move-pair ago and are refuted now, with the
                  refutation -- "what no longer works, and why"
    THREATS       standing threats above the tempo baseline, and zugzwang (L4)
For a game: one record per position plus the played move's verdict by eval drop."""
from __future__ import annotations

import io
from dataclasses import dataclass, field

import chess
import chess.pgn

from . import rules as R
from .facts import StateFacts, state_facts
from .dynamics import Effect, Threat, Zugzwang, line_effects, threats, zugzwang
from .engine import Engine, Line

VERDICTS = ((300, "blunder"), (100, "mistake"), (50, "inaccuracy"))


@dataclass
class Candidate:
    san: str
    uci: str
    cp: int                         # side-to-move POV at the record's position
    mate: int | None
    line_san: str
    effects: list[Effect]
    refutation: "Candidate | None" = None

    @classmethod
    def from_line(cls, board: chess.Board, line: Line, max_plies: int = 6) -> "Candidate":
        return cls(board.san(chess.Move.from_uci(line.move)), line.move, line.cp, line.mate,
                   line.san_line(board, max_plies), line_effects(board, line.pv, max_plies))


@dataclass
class KilledIdea:
    candidate: Candidate            # the move as it stands now (with its refutation)
    was_cp: int
    now_cp: int
    killed_by: str                  # the opponent's last move (SAN)


@dataclass
class IdeaRecord:
    fen: str
    side: str
    ply: int
    facts: StateFacts
    eval_cp: int
    objective: str
    idea: Candidate | None = None
    alternatives: list[Candidate] = field(default_factory=list)
    killed: list[KilledIdea] = field(default_factory=list)
    threats: list[Threat] = field(default_factory=list)
    zugzwang: Zugzwang | None = None
    played: Candidate | None = None
    verdict: str | None = None      # blunder / mistake / inaccuracy / ok / best


def _refute(board: chess.Board, move_uci: str, engine: Engine, max_plies: int) -> Candidate | None:
    t = board.copy(stack=False)
    t.push(chess.Move.from_uci(move_uci))
    b = engine.best(t)
    return Candidate.from_line(t, b, max_plies) if b else None


def analyse_position(board: chess.Board, engine: Engine, *, prev: chess.Board | None = None,
                     killed_by: str | None = None, played: chess.Move | None = None, multipv: int = 5,
                     fail_gap: int = 100, kill_drop: int = 150, near_best: int = 80, max_plies: int = 6) -> IdeaRecord:
    lines = engine.analyse(board, multipv=multipv)
    rec = IdeaRecord(board.fen(), "white" if board.turn else "black", board.ply(), state_facts(board),
                     lines[0].cp if lines else 0, R.rules_of(board).objective)
    if not lines:
        return rec
    rec.idea = Candidate.from_line(board, lines[0], max_plies)
    for l in lines[1:]:
        if lines[0].cp - l.cp >= fail_gap:
            c = Candidate.from_line(board, l, max_plies)
            c.refutation = _refute(board, l.move, engine, 2)
            rec.alternatives.append(c)
    if prev is not None:
        pl = engine.analyse(prev, multipv=multipv)
        for l in pl:
            if l.cp < pl[0].cp - near_best:
                continue
            m = chess.Move.from_uci(l.move)
            if m not in board.legal_moves:
                continue
            now = next((x for x in lines if x.move == l.move), None) or engine.eval_move(board, m)
            if now is None or l.cp - now.cp < kill_drop:
                continue
            c = Candidate.from_line(board, now, max_plies)
            c.refutation = _refute(board, l.move, engine, 2)
            rec.killed.append(KilledIdea(c, l.cp, now.cp, killed_by or "the last move"))
    probe = engine.null_move_probe(board, multipv=3)
    rec.threats = [t for t in threats(board, engine, now=lines[0], probe=probe) if t.real]
    rec.zugzwang = zugzwang(board, engine, now=lines[0], probe=probe)
    if played is not None:
        pline = next((x for x in lines if x.move == played.uci()), None) or engine.eval_move(board, played)
        if pline:
            rec.played = Candidate.from_line(board, pline, max_plies)
            drop = rec.eval_cp - pline.cp
            rec.verdict = next((name for thr, name in VERDICTS if drop >= thr), "best" if drop <= 10 else "ok")
            if drop >= 50:
                rec.played.refutation = _refute(board, played.uci(), engine, 2)
    return rec


def analyse_game(pgn_text: str, engine: Engine, **kw) -> tuple[list[IdeaRecord], chess.pgn.Game]:
    game = chess.pgn.read_game(io.StringIO(pgn_text))
    board = game.board()
    boards, sans, recs = [board.copy()], [], []
    for mv in game.mainline_moves():
        prev = boards[-3] if len(boards) >= 3 else None          # same side to move, one move-pair ago
        recs.append(analyse_position(board, engine, prev=prev, killed_by=sans[-1] if sans else None, played=mv, **kw))
        sans.append(board.san(mv))
        board.push(mv)
        boards.append(board.copy())
    return recs, game
