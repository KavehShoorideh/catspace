"""PLAN 78.2 THE END-TO-END SYSTEM AS A UCI ENGINE: any UCI GUI (or the play server) plays it in every perfect-information variant -- `setoption name UCI_Variant
value <variant>` (lichess's names: chess, crazyhouse, atomic, antichess, kingofthehill, 3check, racingkings, horde), `setoption name UCI_Chess960 value true`.
Each `go` is ONE call into the Rust core (cs_engine.best_move); the trace is printed as `info string` lines. The clock: `movetime`, or a share of the remaining time.

  .venv/bin/python scripts/plan78_uci.py
"""
from __future__ import annotations

import sys

import cs_engine as CE

SHARE = 30            # a move's share of the remaining clock (1/SHARE of it, plus the increment)


def worth() -> dict:
    from catspace.incubator.analyst.discovery import lemma_store as LS
    return {k: int(v) for k, v in LS.valuation().worth.items()}


def main() -> int:
    variant, c960, fen, moves, w = "chess", False, None, [], None
    out = lambda s: (sys.stdout.write(s + "\n"), sys.stdout.flush())
    for line in sys.stdin:
        t = line.split()
        if not t:
            continue
        if t[0] == "uci":
            out("id name catspace e2e (plan 78)")
            out("id author catspace")
            out("option name UCI_Variant type combo default chess " + " ".join(f"var {v}" for v in CE.variants() if v != "chess960"))
            out("option name UCI_Chess960 type check default false")
            out("uciok")
        elif t[0] == "isready":
            w = w or worth()
            out("readyok")
        elif t[0] == "setoption" and "name" in t and "value" in t:
            name, value = t[t.index("name") + 1], t[t.index("value") + 1]
            if name == "UCI_Variant":
                variant = value
            elif name == "UCI_Chess960":
                c960 = value == "true"
        elif t[0] == "ucinewgame":
            fen, moves = None, []
        elif t[0] == "position":
            fen = None
            if "fen" in t:
                i = t.index("fen") + 1
                j = t.index("moves") if "moves" in t else len(t)
                fen = " ".join(t[i:j])
            moves = t[t.index("moves") + 1:] if "moves" in t else []
        elif t[0] == "go":
            w = w or worth()
            v = "chess960" if c960 and variant == "chess" else variant
            secs = 1.0
            if "movetime" in t:
                secs = int(t[t.index("movetime") + 1]) / 1000
            else:
                _, mover, _, _, _ = CE.state(v, fen, moves)
                side = "w" if mover == "white" else "b"
                if f"{side}time" in t:
                    left = int(t[t.index(f"{side}time") + 1]) / 1000
                    inc = int(t[t.index(f"{side}inc") + 1]) / 1000 if f"{side}inc" in t else 0.0
                    secs = left / SHARE + inc
            mv, proved, trace, nodes, _f = CE.best_move(v, fen, moves, secs, w)
            for x in trace:
                out("info string " + x)
            out(f"info nodes {nodes}")
            out(f"bestmove {mv or '0000'}")
        elif t[0] == "quit":
            break
    return 0


if __name__ == "__main__":
    sys.exit(main())
