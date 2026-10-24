"""THE REASONED SEARCH (ANALYST_PLAN 51): the ruler's alpha-beta where, at the nodes WE move, only the moves the reasoning recommends are searched, and
at their nodes EVERY legal reply is (soundness: a restricted max is at most the full max, so a line found is a real forced line; the search may MISS an
answer, never claim one the full search refutes). Every restriction is a CUT with its reason (`ruler.solve(..., ours=...)` records them).

`recommend(board, us, memory=True) -> (moves | None, reasons)`: pure functions of the board and the vocabulary (the Rust `analyst_core` words `see`
and `attackers`, read through `selfmine.val_at`, exactly as `puzzle_apply.state_lines` reads them), in the plan's order; each rule that applies is a
reason string "<tag>: <why>"; `reasons[-1]` is the deciding one. None = search everything ("none: no reasoning applies").
  1. mate         `selfmine.bind_mate` / `bind_mate2` fires at the node: just that move.
  2. in check     every legal move is an evasion already: nothing is cut (the reason is recorded with 0 cuts).
  3. threat       a piece of ours (not the king) has `see` > 0 (the level-one state's `hanging`): captures of an attacker of it, moves of it, moves after
                  which the biggest hanging piece no longer wins for them (an added defender or an interposition), checks, captures of value >= its worth.
  4. observables  captures, checks, promotions, and moves after which some piece of theirs has `see` > 0 (read with us to move again: the move then a pass).
  5. memory       at the ROOT only: `remember.propose` puts the recalled mover kind + move kind first. It reorders; it never adds or removes a move
                  (the ruler's own order decides, the memory order breaks its ties).
Deviation from 51.2: `see` after a move is read on the position after the move and a pass (their pieces' `see` is the exchange WE would win, which the word
reads with us to move), not through `after`, which leaves them on move.
"""
from __future__ import annotations

import chess

from catspace.incubator.analyst.discovery import base_vocabulary as BV
from catspace.incubator.analyst.discovery import remember as RM
from catspace.incubator.analyst.discovery import ruler as RU
from catspace.incubator.analyst.discovery import selfmine as SM

_CACHE: dict = {}
THREAT_RUST = True         # plan 54 (B5.3): `_threat_moves` reads the exchange after every quiet candidate in one `analyst_core.see_after` call (no copy, push and FEN per move)


def _key(board: chess.Board) -> str:
    return " ".join(board.fen().split()[:4])


def _see(fen: str, sq: int, us: str | None = None) -> int:
    """The word `see` on a square, read with the mover = the side to move at `fen` (the evaluator reads the position from the mover's side and that must be the
    side on move: with the other letter every exchange reads 0). `us` is accepted for the callers' symmetry and ignored."""
    return SM.val_at(fen, ("see", chess.square_name(sq)), fen.split()[1])


def _worth(board: chess.Board, sq: int) -> int:
    return SM.PIECE_WORTH[board.piece_at(sq).symbol().lower()]


def hanging_of_ours(board: chess.Board, us: str) -> list:
    """Squares of our pieces (not the king) the other side wins by the exchange on (`see` > 0), the level-one state's `hanging` rows."""
    fen = board.fen()
    return [s for s in chess.SQUARES if board.piece_at(s) and board.piece_at(s).color == board.turn
            and board.piece_at(s).piece_type != chess.KING and _see(fen, s, us) > 0]


def _mate_move(board: chess.Board, us: str):
    for fam, bind in (("mate in one", SM.bind_mate), ("mate in two", SM.bind_mate2)):
        got = bind(board.fen(), us)
        if got:
            (x, y), = got[0]["pairs"]
            return chess.Move.from_uci(x + y), fam
    return None, None


def _threat_moves(board: chess.Board, us: str, hang: list) -> list:
    big = max(hang, key=lambda s: _worth(board, s))
    v = _worth(board, big)
    attackers = set()
    for s in hang:
        attackers |= set(board.attackers(not board.turn, s))
    out, rest = [], []                              # rest = (index in out, move) of the candidates the exchange must decide
    for m in board.legal_moves:
        if board.gives_check(m) or m.from_square in hang or (board.is_capture(m) and m.to_square in attackers):
            out.append(m)
            continue
        if board.is_capture(m):
            victim = 1 if board.is_en_passant(m) else _worth(board, m.to_square)
            if victim >= v:
                out.append(m)
                continue
        out.append(None)
        rest.append((len(out) - 1, m))
    if rest:
        seen = None
        if THREAT_RUST:
            try:
                import analyst_core as A
                from catspace.incubator.analyst.discovery import graph as G
                seen = A.see_after_pos(G.handle(board), [m.uci() for _, m in rest], big)          # plan 72.7 step 1: the board's handle, no FEN
            except ValueError:
                seen = None
        for j, (i, m) in enumerate(rest):
            if seen is not None:
                low = seen[j] <= 0
            else:
                c = board.copy()
                c.push(m)
                low = _see(c.fen(), big, us) <= 0
            if low:                                  # an added defender or an interposition: the biggest hanging piece no longer wins for them
                out[i] = m
    return [m for m in out if m is not None]


def _creates_hanging(board: chess.Board, m: chess.Move, us: str) -> bool:
    c = board.copy()
    c.push(m)
    if c.is_check():
        return True
    c.push(chess.Move.null())
    fen = c.fen()
    for s in chess.SQUARES:
        p = c.piece_at(s)
        if p and p.color != c.turn and p.piece_type != chess.KING and c.is_attacked_by(c.turn, s) and _see(fen, s, us) > 0:
            return True
    return False


def _observable_moves(board: chess.Board, us: str) -> list:
    return [m for m in board.legal_moves
            if board.is_capture(m) or board.gives_check(m) or m.promotion or _creates_hanging(board, m, us)]


def _memory_order(board: chess.Board, moves: list, us: str, exclude_id) -> list:
    props = RM.propose(board.fen(), us, 3, exclude_id=exclude_id)
    if not props:
        return moves
    _, mover, _, mk, _, _ = props[0]
    fen = board.fen()

    def key(m):
        x, y = chess.square_name(m.from_square), chess.square_name(m.to_square)
        return (0 if board.piece_at(m.from_square).symbol().upper() == mover else 1, 0 if RM.move_kind(fen, x, y) == mk else 1)
    return sorted(moves, key=key)


def recommend(board: chess.Board, us: str | bool, memory: bool = True, exclude_id: str | None = None):
    """-> (moves | None, reasons). `us` = 'w' / 'b' (or a chess colour); the side to move is expected to be us."""
    if isinstance(us, bool):
        us = "w" if us else "b"
    BV.install_terms()
    ck = (_key(board), us, memory and not board.move_stack, exclude_id)
    if ck in _CACHE:
        return _CACHE[ck]
    reasons: list = []
    moves = None
    mv, fam = _mate_move(board, us)
    if mv is not None:
        reasons.append(f"mate: {fam} at this node: nothing beats it")
        moves = [mv]
    elif board.is_check():
        reasons.append("in check: nothing else is legal")
        moves = list(board.legal_moves)
    else:
        hang = hanging_of_ours(board, us)
        if hang:
            v = max(_worth(board, s) for s in hang)
            got = _threat_moves(board, us, hang)
            if got:
                reasons.append(f"threat: piece worth {v} hangs: parry it, or check, or take as much")
                moves = got
        if moves is None:
            got = _observable_moves(board, us)
            if got:
                reasons.append("observables: quiet moves that threaten nothing are not searched")
                moves = got
    if moves is None:
        reasons.append("none: no reasoning applies")
    elif memory and not board.move_stack:
        moves = _memory_order(board, moves, us, exclude_id)
    _CACHE[ck] = (moves, reasons)
    return moves, reasons


def tally(cuts: list) -> dict:
    """Moves cut per reason tag, from `solve`'s cuts."""
    out: dict = {}
    for _ply, _d, n, reason in cuts:
        tag = reason.split(":")[0]
        out[tag] = out.get(tag, 0) + n
    return out


def solve(fen: str, need: int, max_depth: int = 6, trace: int = 0, memory: bool = True, exclude_id: str | None = None, answer_uci: str | None = None) -> dict:
    """The ruler with `ours` = the reasoning's recommendation. The result is the ruler's plus `cut_moves` (total) and `by_reason` (moves cut per tag)."""
    us = "w" if chess.Board(fen).turn else "b"

    def ours(b):
        moves, reasons = recommend(b, us, memory, exclude_id)
        return None if moves is None else (moves, reasons[-1])
    r = RU.solve(fen, need, max_depth, trace, answer_uci, ours=ours)
    r["cut_moves"] = sum(c[2] for c in r["cuts"])
    r["by_reason"] = tally(r["cuts"])
    return r


def main():
    import argparse
    from catspace.incubator.analyst.discovery.puzzle_apply import load_set
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="dev_20")
    ap.add_argument("--id", required=True)
    ap.add_argument("--max-depth", type=int, default=6)
    ap.add_argument("--no-memory", action="store_true")
    a = ap.parse_args()
    p = next(x for x in load_set(a.set) if x["id"] == a.id)
    full = RU.solve(p["fen"], int(p["value"]), a.max_depth, 0)
    red = solve(p["fen"], int(p["value"]), a.max_depth, 0, memory=not a.no_memory, exclude_id=p["id"])
    ans = chess.Board(p["fen"]).san(chess.Move.from_uci(p["answer"]))
    found = lambda r: r["status"] == "forced" and r["line"] and r["line"][0] == ans
    print(f"OBS trace {p['id']}: ruler {full['status']} depth {full['depth']} nodes {full['nodes']} line {' '.join(full['line'])} | reasoned {red['status']} depth {red['depth']} "
          f"nodes {red['nodes']} line {' '.join(red['line'])} | cuts {red['cut_moves']} moves at {len(red['cuts'])} of our nodes, by reason {red['by_reason']} | "
          f"answer {ans}: ruler found {found(full)}, reasoned found {found(red)}")


if __name__ == "__main__":
    main()
