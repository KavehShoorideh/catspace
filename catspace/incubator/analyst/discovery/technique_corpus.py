"""PLAN 73.5 (2) THE CORPUS: won games of one material class, written by the table -- the generator's data. The Syzygy table is the ORACLE here and only here:
it writes the corpus, it never answers inside the agenda (the agenda's own table claim, plan 62.3, is `table.py`'s and is untouched).

A material SIGNATURE names the class from the winning side: "KRvK" = the winning side has a king and a rook, the other side a bare king (the order of
`table.code`: king, queens, rooks, bishops, knights, pawns). The winning side is White in the corpus (the measures are functions of (board, winning side), so
the colour is a parameter of every reading, never of the class).

  python -m catspace.incubator.analyst.discovery.technique_corpus KRvK [--games 100] [--seed 1] [--out <file>]

THE STARTS: random legal placements (`random_start`: the units on distinct random squares, ours first, then theirs, the winning side to move), kept when the
position is valid, the side to move is not in check, the game is not over, the table says WON (WDL +2: a win under the fifty-move rule) and the distance to the
zeroing move (DTZ) is at least the FLOOR. The draw consumes the generator exactly as the 2026-10-23 driver (scratchpad krk/play_krk.py) did, so a signature,
a seed and its floor of 20 give that experiment's start (seed 1: 4K3/R7/8/8/2k5/8/8/8 w).

THE FLOOR (`dtz_floor`): the MEDIAN DTZ of FLOOR_SAMPLE won starts of the class drawn the same way (seeded, printed) -- the corpus is drawn from the harder half of
the class. Why: (i) a fixed number of plies is not generic (K+Q v K's longest win is 17 plies by this table, K+R v K's 31: a floor of 20 admits no K+Q start, a
floor of 5 admits K+R starts the mate words close at once); (ii) every start then lies far beyond the reach of the agenda's mate words (mate_in_one 1 ply,
mate_in_two 3 plies), so every game holds a DRIVE before a mate word can bind -- the part the generator must explain; (iii) it is measured, not chosen.

THE PLAY: the winning side plays the table's DTZ-best move (a mate first; else the move after which the other side is lost, WDL -2, with the smallest |DTZ|; ties
by python-chess's move order -- `table.table_claim`'s key); the defender plays the table's SLOWEST line (the reply after which our DTZ is largest; a reply that
draws is taken at once; ties by move order -- play_krk.py's `defend`). The fifty-move rule and repetition are the game's (`is_game_over(claim_draw=True)` ends it).

GAMES: 100, the middle of the plan's range (50 to 200): several hundred winning moves or more (printed per corpus: K+R v K 1203, K+Q v K 683 at seed 1), so a
measure that rises on one winning move in 200 shows a violation with probability above 0.95 in either (1 - (199/200)^683), and a corpus is written in seconds.

THE ROWS (JSONL, one per position, the game's order): game, ply, fen, side (to move), winner, move (uci) and san (None at the end), dtz_before (the table's DTZ for
the side to move), dtz_after (after the move, for the other side), result (on the last row of a game: "mate", "stalemate", "draw" ...).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import time

import chess
import chess.syzygy

from catspace.io import paths

GAMES = 100                 # see the docstring: the middle of the plan's 50-200
FLOOR_SAMPLE = 1000         # won starts drawn to measure the class's median DTZ (the floor); the median of 1000 draws moves by about a ply between seeds
FLOOR_SEED = 0              # the floor's own sample is seeded apart from the games
UNITS = {"K": chess.KING, "Q": chess.QUEEN, "R": chess.ROOK, "B": chess.BISHOP, "N": chess.KNIGHT, "P": chess.PAWN}
ORDER = "KQRBNP"            # table.code's order
_TB: list = [None]


def tablebase():
    if _TB[0] is None:
        _TB[0] = chess.syzygy.open_tablebase(str(paths.syzygy_dir()))
    return _TB[0]


def parse_signature(sig: str) -> tuple:
    """'KRvK' -> (['K', 'R'], ['K']): the winning side's units, then the other side's; each side one king, the units in table.code's order."""
    if sig.count("v") != 1:
        raise ValueError(f"a signature is <ours>v<theirs>, e.g. KRvK: {sig!r}")
    ours, theirs = sig.split("v")
    for side in (ours, theirs):
        if side.count("K") != 1 or not side.startswith("K") or any(c not in UNITS for c in side) or list(side) != sorted(side, key=ORDER.index):
            raise ValueError(f"each side is one K then its units in the order {ORDER}: {sig!r}")
    return list(ours), list(theirs)


def signature(board: chess.Board, side: bool) -> str:
    """The material signature of `board` read from `side` (the side whose units come first): 'KRvK' when `side` has a king and a rook against a bare king."""
    def units(c: bool) -> str:
        return "".join(ch * chess.popcount(board.pieces_mask(UNITS[ch], c)) for ch in ORDER)
    return units(side) + "v" + units(not side)


def random_start(rng: random.Random, sig: str, floor: int) -> chess.Board:
    """A random legal start of the class, the winning side (White) to move, not in check, not over, WON by the table with DTZ >= floor (see the docstring)."""
    ours, theirs = parse_signature(sig)
    n = len(ours) + len(theirs)
    tb = tablebase()
    while True:
        sq = rng.sample(range(64), n)
        b = chess.Board(None)
        for s, ch in zip(sq[:len(ours)], ours):
            b.set_piece_at(s, chess.Piece(UNITS[ch], chess.WHITE))
        for s, ch in zip(sq[len(ours):], theirs):
            b.set_piece_at(s, chess.Piece(UNITS[ch], chess.BLACK))
        b.turn = chess.WHITE
        if not b.is_valid() or b.is_check() or b.is_game_over():
            continue
        if tb.probe_wdl(b) == 2 and tb.probe_dtz(b) >= floor:
            return b


def dtz_floor(sig: str, sample: int = FLOOR_SAMPLE, seed: int = FLOOR_SEED) -> tuple:
    """-> (the median DTZ of `sample` won starts of the class, the sample's (min, max)): THE FLOOR of the corpus (see the docstring)."""
    rng = random.Random(seed)
    ds = [tablebase().probe_dtz(random_start(rng, sig, 1)) for _ in range(sample)]
    return int(statistics.median_low(ds)), (min(ds), max(ds))


def winner_move(board: chess.Board) -> chess.Move:
    """The table's DTZ-best move for the winning side to move (table.table_claim's key): a mate first, else the move after which the other side is lost with the
    smallest |DTZ|; ties by python-chess's move order."""
    tb = tablebase()
    best = None
    for m in board.legal_moves:
        board.push(m)
        try:
            if board.is_checkmate():
                key = (0, 0)
            elif board.is_game_over() or tb.probe_wdl(board) != -2:
                continue
            else:
                key = (1, abs(tb.probe_dtz(board)))
        finally:
            board.pop()
        if best is None or key < best[0]:
            best = (key, m)
    if best is None:
        raise ValueError(f"no winning move at {board.fen()}: not a won position")
    return best[1]


def defender_move(board: chess.Board) -> chess.Move:
    """The table's SLOWEST line for the lost side to move (play_krk.py's `defend`): the reply after which the winner's DTZ is largest; a reply that draws first."""
    tb = tablebase()
    best = None
    for m in board.legal_moves:
        board.push(m)
        try:
            d = tb.probe_dtz(board)
        finally:
            board.pop()
        score = 10 ** 6 if d == 0 else d
        if best is None or score > best[0]:
            best = (score, m)
    return best[1]


def _result(board: chess.Board) -> str:
    if board.is_checkmate():
        return "mate"
    if board.is_stalemate():
        return "stalemate"
    if board.can_claim_fifty_moves() or board.is_fifty_moves():
        return "fifty-move rule"
    if board.can_claim_threefold_repetition() or board.is_repetition():
        return "repetition"
    if board.is_insufficient_material():
        return "insufficient material"
    return "over"


def play_game(board: chess.Board, game: int) -> tuple:
    """One game from `board` (the winning side to move; pushed on) -> (the rows of every position, the last one with the result; the result)."""
    tb = tablebase()
    winner = "w" if board.turn else "b"
    start = board.fen()
    rows = []
    while not board.is_game_over(claim_draw=True):
        side = "w" if board.turn else "b"
        fen, before = board.fen(), tb.probe_dtz(board)
        m = winner_move(board) if side == winner else defender_move(board)
        san = board.san(m)
        board.push(m)
        after = tb.probe_dtz(board) if not board.is_game_over() else 0
        rows.append(dict(game=game, ply=len(board.move_stack) - 1, fen=fen, side=side, winner=winner, move=m.uci(), san=san, dtz_before=before, dtz_after=after, result=None))
    res = _result(board)
    rows.append(dict(game=game, ply=len(board.move_stack), fen=board.fen(), side="w" if board.turn else "b", winner=winner, move=None, san=None, dtz_before=None,
                     dtz_after=None, result=res, start=start))
    return rows, res


def write_corpus(sig: str, games: int = GAMES, seed: int = 1, out: "str | None" = None, floor: "int | None" = None, log=print) -> dict:
    """THE CORPUS of a signature -> {path, games, positions, plies, mates, floor, ...}; prints `OBS corpus <sig>: ...`."""
    parse_signature(sig)
    t0 = time.monotonic()
    span = None
    if floor is None:
        floor, span = dtz_floor(sig)
    rng = random.Random(seed)
    out = out or paths.experiment("technique_corpus", f"{sig}_seed{seed}.jsonl")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    rows, lengths, mates, results = [], [], 0, {}
    for g in range(games):
        b = random_start(rng, sig, floor)
        grows, res = play_game(b, g)
        results[res] = results.get(res, 0) + 1
        mates += res == "mate"
        lengths.append(len(b.move_stack))
        rows += grows
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    wins = sum(1 for r in rows if r["move"] is not None and r["side"] == r["winner"])
    info = dict(path=out, signature=sig, games=games, positions=len(rows), winning_moves=wins, mean_plies=round(sum(lengths) / len(lengths), 1), mates=mates,
                results=results, floor=floor, seed=seed)
    log(f"OBS corpus {sig}: games {games}, positions {len(rows)}, mean length {info['mean_plies']} plies, mates {mates}; winning moves {wins}; results "
        + ", ".join(f"{k} {v}" for k, v in sorted(results.items())) + f"; starts with DTZ >= {floor}"
        + (f" (the median DTZ of {FLOOR_SAMPLE} won starts of the class, seed {FLOOR_SEED}; their DTZ {span[0]}..{span[1]})" if span else " (given)")
        + f"; seed {seed}; {time.monotonic() - t0:.1f} s; {out}")
    return info


def read_corpus(path: str) -> list:
    """The rows of a corpus file, grouped by game: [[row, ...] per game] in the file's order."""
    games: dict = {}
    with open(path) as f:
        for ln in f:
            if ln.strip():
                r = json.loads(ln)
                games.setdefault(r["game"], []).append(r)
    return [games[g] for g in sorted(games)]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("signature", help="the material class, e.g. KRvK, KQvK")
    ap.add_argument("--games", type=int, default=GAMES)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--floor", type=int, default=None, help="the DTZ floor of the starts (default: the class's median, see the docstring)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    write_corpus(a.signature, a.games, a.seed, a.out, a.floor)


if __name__ == "__main__":
    main()
