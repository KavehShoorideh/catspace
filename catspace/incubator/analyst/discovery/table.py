"""PLAN 62.3: THE TABLEBASE AS A LABELLED ORACLE. `assets/tablebases/syzygy` holds the 3-5 man tables (290 files). A position whose WHOLE board has at most five men closes by the table:
win (WDL +2), draw, loss (-2); the claim is exact by exhaustion and says nothing about WHY -- the third label between a proved word and an estimate (`Claim.proved_by = "table"`).
python-chess's `chess.syzygy` reads the files (no new dependency). Cursed wins and blessed losses (WDL +-1: they are draws under the 50-move rule) are draws here.

value: a win is worth the material of their remaining pieces (a bare king left: 100, a mate), a draw is worth minus the balance when we are behind (it brings the result to level), a loss is no claim.
"""
from __future__ import annotations

import os

import chess
import chess.syzygy

from catspace.io import paths
from catspace.incubator.analyst.discovery import lemma_store as LS

TABLE = os.environ.get("CATSPACE_TABLE", "1") != "0"        # plan 62.3 (ON; `analyse --no-table` sets CATSPACE_TABLE=0, inherited by the workers: only a comparison condition)
MEN = 5
_TB: list = [None]
_VALUE = LS.valuation().by_kind(royal=False)                # plan 76.5: the store's valuation (the royal unit aside), not a table of its own
STATS = {"probed": 0, "hit": 0}


def tablebase():
    if _TB[0] is None:
        _TB[0] = chess.syzygy.open_tablebase(str(paths.syzygy_dir()))
    return _TB[0]


def men(board: chess.Board) -> int:
    return chess.popcount(board.occupied)


def code(board: chess.Board) -> str:
    """'KRPvKR': white's pieces, v, black's (queen, rook, bishop, knight, pawn)."""
    def side(c: bool) -> str:
        s = "K"
        for pt, ch in ((chess.QUEEN, "Q"), (chess.ROOK, "R"), (chess.BISHOP, "B"), (chess.KNIGHT, "N"), (chess.PAWN, "P")):
            s += ch * chess.popcount(board.pieces_mask(pt, c))
        return s
    return side(chess.WHITE) + "v" + side(chess.BLACK)


def covered(board: chess.Board) -> bool:
    return TABLE and men(board) <= MEN and board.castling_rights == 0 and board.king(True) is not None and board.king(False) is not None


def _material(board: chess.Board, color: bool) -> int:
    return sum(_VALUE[pt] * chess.popcount(board.pieces_mask(pt, color)) for pt in _VALUE)


def probe(board: chess.Board) -> tuple | None:
    """-> (wdl for the side to move in {-2..2}, dtz) or None when a table is missing."""
    try:
        t = tablebase()
        w = t.probe_wdl(board)
        try:
            z = t.probe_dtz(board)
        except Exception:
            z = None
        return w, z
    except Exception:
        return None


def table_claim(board: chess.Board) -> dict | None:
    """The table's answer for the side to move: {'kind': 'win'|'draw', 'move': UCI, 'value': int, 'wdl', 'dtz', 'code', 'text'} or None (no table, a loss, or a draw with nothing to gain)."""
    if not covered(board):
        return None
    STATS["probed"] += 1
    me = board.turn
    r = probe(board)
    if r is None:
        return None
    w, z = r
    bal = _material(board, me) - _material(board, not me)
    c = code(board)
    if w == 2:
        best = None
        for m in board.legal_moves:
            d = board.copy(stack=False)
            d.push(m)
            ra = probe(d) if not d.is_game_over() else None
            if d.is_checkmate():
                key = (0, 0)
            elif ra is not None and ra[0] == -2:
                key = (1, abs(ra[1]) if ra[1] is not None else 99)
            else:
                continue
            if best is None or key < best[0]:
                best = (key, m)
        if best is None:
            return None
        their = _material(board, not me)
        value = 100 if their == 0 else max(their, 1)
        STATS["hit"] += 1
        return dict(kind="win", move=best[1].uci(), value=value, wdl=w, dtz=z, code=c, text=f"closed by the table ({c}: win, DTZ {abs(z) if z is not None else '?'})")
    if w in (-1, 0, 1) and bal < 0:
        best = None
        for m in board.legal_moves:
            d = board.copy(stack=False)
            d.push(m)
            if d.is_stalemate():
                key = 0
            else:
                ra = probe(d) if not d.is_checkmate() else None
                if ra is None or ra[0] < 0 or ra[0] > 1:
                    continue                                  # their win-ish, or our mate (a win: handled above)
                key = 1
            if best is None or key < best[0]:
                best = (key, m)
        if best is None:
            return None
        STATS["hit"] += 1
        return dict(kind="draw", move=best[1].uci(), value=-bal, wdl=w, dtz=z, code=c, text=f"closed by the table ({c}: draw, balance {bal:+d})")
    return None
