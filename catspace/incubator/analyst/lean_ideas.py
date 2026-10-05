"""THE LEAN DETECTOR FROM PYTHON (Kaveh 2026-10-04: "let's use Lean ... I don't want search from it ... I want to be
able to prune some branches, I want to find the search target from it").

`lean/ideas` is a Lake package over the chess-lean rule kernel (Arthur742Ramos/chess-lean, Lean 4 + Mathlib). It holds
ideas in one schema -- individually evaluable hypotheses, a tier, a horizon -- and, for each, a soundness theorem proved
once, offline: when every hypothesis holds, the conclusion is true in every position, whatever the rest of the board.
The `ideas` binary evaluates those same compiled hypothesis functions; it never searches and never runs Lean's kernel.

The first idea is the back-rank mate (`lean/ideas/Ideas/BackRank.lean`, theorem `backRankMate_sound`, resting on the
general `checkmate_of_unmeetable_check` in `Ideas/Mate.lean`): our rook or queen lands on the defender's home rank with
check, the king is boxed, nothing takes the checker and nothing steps between -- proved mate. When exactly one
hypothesis fails, the detector names the squares that broke it (the defender of the landing square, the piece in the
path, the open flight square): the search target.

    python -m catspace.incubator.analyst.lean_ideas --fen "6k1/5ppp/8/8/8/8/8/3R2K1 w - - 0 1"

Build first:  cd lean/ideas && lake build      (see lean/README.md)
"""
from __future__ import annotations

import argparse
import os
import subprocess
from dataclasses import dataclass, field

import chess

from catspace.io import paths

HYPS = ("heavy", "onBackRank", "legal", "oneKing", "check", "boxed", "rankFlights", "noDefender", "noInterpose")


def default_binary():
    return paths.REPO_ROOT / "lean" / "ideas" / ".lake" / "build" / "bin" / "ideas"


@dataclass
class Instance:
    """One back-rank candidate: our rook or queen move onto the defender's home rank."""
    move: chess.Move
    status: str                      # "MATE" (proved), "ONEAWAY" (exactly one hypothesis fails) or "FAIL"
    failing: str | None              # the failing hypothesis when ONEAWAY
    targets: list[int] = field(default_factory=list)   # squares that broke it (python-chess square indices)
    hyps: dict[str, bool] = field(default_factory=dict)


def encode(board: chess.Board) -> str:
    """The position as python-chess bitboards in hex (2.7 us; a FEN costs 19 us to write and 90 us to parse)."""
    ep = chess.square_name(board.ep_square) if board.ep_square is not None else "-"
    return (f"{'w' if board.turn else 'b'} {board.castling_xfen() or '-'} {ep} {board.pawns:x} {board.knights:x} "
            f"{board.bishops:x} {board.rooks:x} {board.queens:x} {board.kings:x} {board.occupied_co[chess.WHITE]:x}")


def _parse_cand(line: str) -> Instance:
    # cand d1d8 ONEAWAY noDefender targets=e8 heavy=1 ...   |   cand d1d8 MATE targets= heavy=1 ...
    tok = line.split()
    move = chess.Move.from_uci(tok[1])
    status = tok[2]
    i = 3
    failing = None
    if status == "ONEAWAY":
        failing = tok[3]; i = 4
    elif status == "FAIL":
        i = 4
    tg = tok[i].split("=", 1)[1]
    targets = [chess.parse_square(s) for s in tg.split(",") if s]
    hyps = {k: v == "1" for k, v in (t.split("=") for t in tok[i + 1:])}
    return Instance(move, status, failing, targets, hyps)


PASSER_HYPS = ("ownPawn", "pathClear", "bareKing", "noSliders", "notInCheck", "kingOutside", "kingHasMove")


@dataclass
class PasserInstance:
    """One pawn of ours: PROVED (promotes within `plies` whatever they do), ONEAWAY or FAIL."""
    square: int
    status: str
    plies: int | None
    failing: str | None
    targets: list[int] = field(default_factory=list)
    hyps: dict[str, bool] = field(default_factory=dict)


def _parse_passer(line: str) -> PasserInstance:
    # cand e3 PROVED plies=9 targets= ownPawn=1 ...  |  cand e3 ONEAWAY kingOutside targets=a4 ownPawn=1 ...
    tok = line.split()
    sq = chess.parse_square(tok[1])
    status = tok[2]
    plies = failing = None
    i = 3
    if status == "PROVED":
        plies = int(tok[3].split("=")[1]); i = 4
        if tok[4].startswith("form="):
            i = 5
    elif status == "ONEAWAY":
        failing = tok[3]; i = 4
    else:
        i = 4
    tg = tok[i].split("=", 1)[1]
    targets = [chess.parse_square(x) for x in tg.split(",") if x]
    hyps = {k: v == "1" for k, v in (t.split("=") for t in tok[i + 1:])}
    return PasserInstance(sq, status, plies, failing, targets, hyps)


class LeanIdeas:
    """A persistent `ideas` process. One line in, the answer out (microseconds to a millisecond per query)."""

    def __init__(self, binary=None):
        self.binary = str(binary or default_binary())
        self.proc = subprocess.Popen([self.binary], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        self.calls = 0

    def _ask(self, line: str, until_end: bool) -> list[str]:
        self.calls += 1
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()
        out = []
        while True:
            got = self.proc.stdout.readline()
            if not got:
                raise RuntimeError(f"ideas binary exited on: {line}")
            got = got.rstrip("\n")
            if not until_end:
                return [got]
            if got == "end":
                return out
            out.append(got)

    def backrank(self, board: chess.Board, fen: bool = False) -> list[Instance]:
        """Every rook/queen move onto the defender's home rank, with its hypotheses (those evaluated: the detector
        stops at the second failure). `fen`: send the FEN instead of bitboards (the bridge check; one line per
        candidate and an `end`; the bitboard answer is one line)."""
        if fen:
            lines = self._ask("backrank " + board.fen(), until_end=True)
        else:
            (line,) = self._ask("bb " + encode(board), until_end=False)
            lines = [x for x in line.split(" | ") if x]
        if lines and lines[0].startswith("error"):
            raise ValueError(f"{lines[0]}: {board.fen()}")
        return [_parse_cand(l) for l in lines]

    def passer(self, board: chess.Board) -> list["PasserInstance"]:
        """Every pawn of the side to move, with the unstoppable-passer hypotheses (those evaluated)."""
        (line,) = self._ask("passer " + encode(board), until_end=False)
        if line.startswith("error"):
            raise ValueError(f"{line}: {board.fen()}")
        return [_parse_passer(x) for x in line.split(" | ") if x]

    def passer2(self, board: chess.Board) -> list["PasserInstance"]:
        """The second form (enemy king and pawns): every pawn of the side to move with its hypotheses."""
        (line,) = self._ask("passer2 " + encode(board), until_end=False)
        if line.startswith("error"):
            raise ValueError(f"{line}: {board.fen()}")
        return [_parse_passer(x) for x in line.split(" | ") if x]

    def legal(self, board: chess.Board, fen: bool = False) -> set[str]:
        """The kernel's legal moves (UCI), for the bridge check against python-chess (FEN or bitboard input)."""
        (line,) = self._ask(("legal " + board.fen()) if fen else ("bblegal " + encode(board)), until_end=False)
        if line.startswith("error"):
            raise ValueError(f"{line}: {board.fen()}")
        return set(line.split())

    def checkmate(self, board: chess.Board) -> bool:
        (line,) = self._ask("checkmate " + board.fen(), until_end=False)
        return line == "1"

    def close(self):
        if self.proc.poll() is None:
            self.proc.stdin.close()
            self.proc.wait(timeout=10)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


_PER_PROCESS: dict = {}


def detector() -> LeanIdeas:
    """One detector process per Python process (fork-safe: a forked worker starts its own; it exits on EOF when the
    worker dies)."""
    pid = os.getpid()
    li = _PER_PROCESS.get(pid)
    if li is None:
        li = _PER_PROCESS[pid] = LeanIdeas()
    return li


def rank_check_possible(b: chess.Board) -> bool:
    """A necessary condition for a back-rank MATE instance, cheap in python-chess (bitboards only, no move
    generation): the defender's king on its home rank, and a rook or queen of ours that can land on a square of that
    rank from which a rook attacks the king along the rank -- every other square of the rank is on the king's own
    file or the king itself. Where it fails the detector is not asked (it runs at every node of the search)."""
    k = b.king(not b.turn)
    home = chess.BB_RANK_8 if b.turn == chess.WHITE else chess.BB_RANK_1
    if k is None or not chess.BB_SQUARES[k] & home:
        return False
    heavy = (b.rooks | b.queens) & b.occupied_co[b.turn]
    if not heavy:
        return False
    landing = home & ~b.occupied_co[b.turn] & ~chess.BB_SQUARES[k]
    for a in chess.scan_forward(heavy):
        if b.attacks_mask(a) & landing:
            return True
    return False


def proved_back_rank_mates(b: chess.Board) -> list[chess.Move]:
    """The moves that mate by `backRankMate_sound` (every hypothesis holds). Never a false mate: the theorem holds for
    every position; the detector may miss mates (its hypotheses over-approximate the defender)."""
    if not rank_check_possible(b):
        return []
    return [inst.move for inst in detector().backrank(b) if inst.status == "MATE"]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--fen", required=True)
    a = ap.parse_args(argv)
    b = chess.Board(a.fen)
    with LeanIdeas() as li:
        for inst in li.backrank(b):
            tg = ",".join(chess.square_name(s) for s in inst.targets)
            print(f"{b.san(inst.move) if b.is_legal(inst.move) else inst.move.uci():8s} {inst.status:8s} "
                  f"{inst.failing or '':12s} targets={tg}")


if __name__ == "__main__":
    main()
