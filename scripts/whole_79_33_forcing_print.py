"""PLAN 79.33 PART D (the forcing read, rust/engine/tactics/src/sheet/forcing.rs): the whole answer and every count of the forcing read on made positions, written line by line to artifacts/experiments/whole_79_33/<label>.forcing.dump, and the time of the reads.
Usage: PYTHONPATH=<folder holding the cs_engine module under test> python scripts/whole_79_33_forcing_print.py <label> [rounds]
It prints the imported module's path first (the module under test must be the one asked for, never main's), then OBS lines. The time is the sum of the reads' own clocks over `rounds` rounds of the same positions (default 1), the first round's answers are the file's.
A read is cut by its node fuse, not by a clock, so the answers do not depend on the machine."""
from __future__ import annotations

import json
import random
import sys
import time

import chess
import cs_engine

from catspace.incubator.analyst.discovery import question_sheet as Q
from catspace.incubator.analyst.discovery import sheet_rust as SR
from catspace.io import paths

WORTH = {"k": 0, "q": 14, "r": 8, "b": 5, "n": 4, "p": 2}
FIXED = [
    "r3k3/8/8/3N4/8/8/8/4K3 w - - 0 1",
    "4k3/8/8/3n1n2/8/4P3/8/6K1 w - - 0 1",
    "4k3/8/2n5/1B2r3/8/8/4R3/4K3 w - - 0 1",
    "2b1kb1r/8/p4n1p/p4p1R/PqPR1P2/NP6/8/2BQ1K2 w k - 3 26",
    "4k3/pp6/8/3n1n2/8/4P3/8/6K1 w - - 0 1",
    "6k1/ppp3pp/4P3/2b5/8/5R2/P1q3PP/1q3R1K w - - 0 54",
    "1r2k3/1bp2p2/p3p2Q/q5B1/3b4/P7/5PPP/2RR2K1 b - - 4 29",
    "4R3/8/7Q/8/6K1/k7/8/8 w - - 0 1",
    "r3q2k/pQ1nb1pp/6n1/5bP1/3PN3/2N1P3/PPP3B1/R1B3K1 w - - 0 1",
    "8/3k4/3Pp3/8/3K4/8/6P1/8 w - - 0 1",
    "3KR3/1k6/1N6/8/8/8/7n/8 w - - 0 1",
]
STARTS = [chess.STARTING_FEN, "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"]


def strip(x):
    if isinstance(x, dict):
        return {k: strip(v) for k, v in x.items() if k != "seconds"}
    if isinstance(x, (list, tuple)):
        return [strip(v) for v in x]
    return x


def positions() -> list[str]:
    rng = random.Random(7933)
    out = list(FIXED)
    for start in STARTS:
        for _ in range(14):
            b = chess.Board(start)
            for _ in range(rng.randrange(4, 30)):
                ms = list(b.legal_moves)
                if not ms or b.is_game_over():
                    break
                b.push(rng.choice(ms))
            if not b.is_game_over():
                out.append(b.fen())
    # positions where the side to move is under the goal's sign (the read then has no pass to take): played on until one is met
    for start in STARTS:
        for _ in range(8):
            b = chess.Board(start)
            for _ in range(200):
                ms = list(b.legal_moves)
                if not ms or b.is_game_over():
                    break
                b.push(rng.choice(ms))
                if b.is_check() and not b.is_game_over():
                    out.append(b.fen())
                    break
    return out


def main() -> int:
    label = sys.argv[1]
    rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    print(f"module under test: {cs_engine.__file__}")
    model = SR.model(Q.Env("chess"), WORTH)
    fens = positions()
    lines = []
    spent = []
    for r in range(rounds):
        t = 0.0
        for fen in fens:
            read = model.forcing(fen, 12_000, None, True)
            t += read["seconds"]
            if r == 0:
                lines.append(f"{fen}\t{json.dumps(strip(read), sort_keys=True)}")
        spent.append(t)
    out = paths.experiment(f"whole_79_33/{label}.forcing.dump")
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"OBS 79.33 forcing: {len(fens)} positions, the reads took {', '.join(f'{x:.3f}' for x in spent)} s over {rounds} rounds")
    return 0


if __name__ == "__main__":
    sys.exit(main())
