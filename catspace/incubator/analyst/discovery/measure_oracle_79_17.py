"""PLAN 79.17: THE ORACLE OF THE MEASURE (SYSTEM_SPEC 7 "the small endings as the test of means and ends, and a measure of progress"; 9 and the oracle rule of 2026-11-19: the endgame table is read in sleep and in the measure, never by the wake or by anything it reads at play).
THE ONE MODULE THAT READS THE ENDGAME TABLE (assets/tablebases/syzygy, through python-chess) and the one that names the game's units: the harness of parts A, C and E (whole_79_17, measure_miner_79_17, the print scripts) call it and nothing else does. It is used for exactly three things:
  1. LABEL: a position is WON for the stronger side (the table's word), so that the starts of a class are drawn from won positions and the "bad" games of the miner are the games that kept the win by the label and did not reach the end;
  2. DEFENDER: the side with the lone leader plays the table's reply that holds longest (see below), in the games of part A and E;
  3. COVERAGE: the table's distance at the start of each game, printed by bins.
The attacker (the wake, the old player), the Rust crates and everything they read at play never see it. No term, atom or number of the measure is put in here: the classes are material signatures only.

THE DISTANCE: the table has no distance to the end, only the distance to the next irreversible move (DTZ). "The reply that holds longest" is the reply, among those that leave the position as good for the defender as any reply can (a draw if one exists, else the win for the attacker), after which the DTZ is the largest; ties by the move's text. That is what the defender is: say so in the report.

INTERFACE.
  CLASSES                                          {name: (the stronger side's material, the other side's)}: two long-movers and the leader against a lone leader ("two"); one long-mover and the leader against a lone leader ("one")
  sample_starts(cls, n, seed) -> [dict]            n starting positions drawn at random from the class's won positions, the stronger side to move, both colours alternating: {"text" (the position text), "attacker_white", "dtz" (the table's distance), "wdl"}
  defender_reply(text, moves, legal) -> str        the table's reply that holds longest (the move's own name, one of `legal`) at the position reached from `text` by `moves`
  ending_of(text, moves, outcome, unfinished) -> str   how a game ended: "won" (the stronger side's), "stalemate", "unit_lost" (insufficient material left), "repetition" (a position met three times), "quiet_rule" (the game's rule on quiet plies), "cap" (the ply cap), "other"
  coverage(starts) -> {bin: count}                 the starts by the table's distance, bins of five
Called by: whole_79_17.py, measure_miner_79_17.py, the print scripts, the tests.
"""
from __future__ import annotations

import random
from functools import lru_cache
from pathlib import Path

import chess
import chess.syzygy

from catspace.io import paths

CLASSES = {"two": ("KRR", "K"), "one": ("KR", "K")}
_PIECE = {"K": chess.KING, "R": chess.ROOK, "Q": chess.QUEEN, "B": chess.BISHOP, "N": chess.KNIGHT, "P": chess.PAWN}


def table_dir() -> Path:
    return Path(__file__).resolve().parents[4] / "assets" / "tablebases" / "syzygy"


@lru_cache(maxsize=1)
def tables() -> "chess.syzygy.Tablebase":
    tb = chess.syzygy.open_tablebase(str(table_dir()))
    return tb


def wdl_for(board: chess.Board, white: bool) -> int:
    """The table's word on `board` from the view of the colour `white` (2 win, 1 cursed win, 0 draw, -1 blessed loss, -2 loss)."""
    w = tables().probe_wdl(board)                    # from the side to move
    return w if board.turn == white else -w


def board_of(text: str, moves: list) -> chess.Board:
    b = chess.Board(text)
    for m in moves:
        b.push_uci(m)
    return b


def _random_board(cls: str, rng: random.Random, attacker_white: bool) -> "chess.Board | None":
    strong, weak = CLASSES[cls]
    b = chess.Board(None)
    squares = list(range(64))
    rng.shuffle(squares)
    k = 0
    for sym in strong:
        b.set_piece_at(squares[k], chess.Piece(_PIECE[sym], attacker_white))
        k += 1
    for sym in weak:
        b.set_piece_at(squares[k], chess.Piece(_PIECE[sym], not attacker_white))
        k += 1
    b.turn = attacker_white
    b.halfmove_clock = 0
    b.fullmove_number = 1
    if not b.is_valid() or b.is_game_over():
        return None
    return b


def sample_starts(cls: str, n: int, seed: int) -> list:
    """n won positions of the class, the stronger side to move, colours alternating (see the module doc)."""
    rng = random.Random(f"{cls}:{seed}")
    out: list = []
    tb = tables()
    while len(out) < n:
        white = len(out) % 2 == 0
        b = _random_board(cls, rng, white)
        if b is None:
            continue
        wdl = wdl_for(b, white)
        if wdl != 2:
            continue
        out.append({"text": b.fen(), "attacker_white": white, "dtz": abs(tb.probe_dtz(b)), "wdl": wdl, "class": cls})
    return out


def defender_reply(text: str, moves: list, legal: list) -> str:
    """The reply that holds longest: among the legal moves, those after which the table says the least for the stronger side, then the largest distance, then the move's text."""
    b = board_of(text, moves)
    tb = tables()
    best = None
    for uci in sorted(legal):
        b.push_uci(uci)
        if b.is_checkmate():
            score = (10, 0)
        elif b.is_stalemate() or b.is_insufficient_material():
            score = (-10, 0)                               # a draw: the best a defender can have
        else:
            w = tb.probe_wdl(b)                            # from the stronger side to move: the lower, the better for the defender
            d = abs(tb.probe_dtz(b)) if w != 0 else 0
            score = (w, -d)
        b.pop()
        key = (score[0], score[1])
        if best is None or key < best[0]:
            best = (key, uci)
    return best[1]


def ending_of(text: str, moves: list, outcome: "int | None", unfinished: bool, attacker_white: bool) -> str:
    b = board_of(text, moves)
    if unfinished:
        return "cap"
    if b.is_checkmate():
        return "won" if (b.turn != attacker_white) else "lost"
    if b.is_stalemate():
        return "stalemate"
    if b.is_insufficient_material():
        return "unit_lost"
    if b.halfmove_clock >= 100:
        return "quiet_rule"
    if outcome == 50:
        return "repetition"
    return "other"


def coverage(starts: list) -> dict:
    out: dict = {}
    for s in starts:
        lo = (max(s["dtz"], 1) - 1) // 5 * 5 + 1
        key = f"{lo}-{lo + 4}"
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: int(kv[0].split("-")[0])))


# ------------------------------------------------------------------------------------------------------------------------------ the games of the miner's sets (part C) and the far starts

def side_of(attacker_white: bool) -> int:
    """The role number of the stronger side in the engine (white is role 0)."""
    return 0 if attacker_white else 1


def fens_of(text: str, moves: list) -> list:
    """The position text at every ply of a game, the start first."""
    b = chess.Board(text)
    out = [b.fen()]
    for m in moves:
        b.push_uci(m)
        out.append(b.fen())
    return out


def attacker_won(text: str, moves: list, attacker_white: bool) -> bool:
    b = board_of(text, moves)
    return b.is_checkmate() and b.turn != attacker_white


def kept_the_win(text: str, moves: list, attacker_white: bool) -> bool:
    """The label stays WON at every turn of the stronger side along the game (it never gave the win away)."""
    b = chess.Board(text)
    for k, m in enumerate(moves):
        if k % 2 == 0 and not b.is_game_over() and wdl_for(b, attacker_white) != 2:
            return False
        b.push_uci(m)
    return True


def _won_moves(b: chess.Board) -> list:
    """The legal moves after which the table still calls the position won for the stronger side (a move that ends the game in the win included), with the distance after it."""
    tb = tables()
    out = []
    for mv in sorted(b.legal_moves, key=lambda m: m.uci()):
        b.push(mv)
        if b.is_checkmate():
            out.append((mv.uci(), 0))
        elif not b.is_game_over() and tb.probe_wdl(b) == -2:
            out.append((mv.uci(), abs(tb.probe_dtz(b))))
        b.pop()
    return out


def _play_out(start: dict, rng: "random.Random | None", cap: int = 100) -> list:
    """A game from a start, the defender the table's, the stronger side random among the won moves (rng) or the table's shortest (None); ends at the end of the game, the quiet rule, a position met three times or the cap."""
    b = chess.Board(start["text"])
    seen = {b._transposition_key(): 1}
    moves: list = []
    while not b.is_game_over() and b.halfmove_clock < 100 and len(moves) < cap:
        legal = [m.uci() for m in b.legal_moves]
        if len(moves) % 2 == 0:
            won = _won_moves(b)
            if not won:
                break
            uci = (rng.choice(won)[0]) if rng is not None else min(won, key=lambda x: (x[1], x[0]))[0]
        else:
            uci = defender_reply(start["text"], moves, legal)
        moves.append(uci)
        b.push_uci(uci)
        k = b._transposition_key()
        seen[k] = seen.get(k, 0) + 1
        if seen[k] >= 3:
            break
    return moves


def generated_games(cls: str, n: int, seed: int) -> tuple:
    """n games with a generated attacker that moves at random among the moves the table still calls won (the defender the table's): those that reach the end are GOOD (the stronger side wins), the rest BAD (kept the win by the label, did not reach the end)."""
    rng = random.Random(f"gen:{cls}:{seed}")
    good, bad = [], []
    for st in sample_starts(cls, n, seed):
        mv = _play_out(st, rng)
        rec = {"text": st["text"], "moves": mv, "attacker_white": st["attacker_white"], "source": "generated", "dtz": st["dtz"]}
        (good if attacker_won(st["text"], mv, st["attacker_white"]) else bad).append(rec)
    return good, bad


def table_games(cls: str, n: int, seed: int) -> list:
    """The table's own shortest lines (both sides the table's): used only to test October's lesson, never as the teacher."""
    return [{"text": st["text"], "moves": _play_out(st, None), "attacker_white": st["attacker_white"], "source": "table", "dtz": st["dtz"]} for st in sample_starts(cls, n, seed)]


def far_starts(cls: str, n: int, seed: int, pool: int = 20) -> list:
    """The n starts of the greatest table distance among `pool` times n drawn at random (a second sample, printed apart: the random sample of a class is mostly near the end)."""
    big = sample_starts(cls, n * pool, seed)
    big.sort(key=lambda s: (-s["dtz"], s["text"]))
    return big[:n]
