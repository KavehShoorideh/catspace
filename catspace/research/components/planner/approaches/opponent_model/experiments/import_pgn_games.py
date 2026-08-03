#!/usr/bin/env python
"""catspace/research/components/planner/approaches/opponent_model/experiments/import_pgn_games.py -- import engine-vs-engine PGNs (gauntlets) into the
experience store (Kaveh: self-play against the different models = WHOLE-SYSTEM data).
Every gauntlet game becomes training data: positions + result flow into the regime-11
export exactly like improvement-loop games. result='mate' iff White won (the store's
result flag is White-POV; both players being our variants is fine -- each row is an
outcome-labeled position regardless of who produced it). Draws import as res=0 (schema
has no draw channel yet -- noted limitation)."""
from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import chess.pgn


from catspace.research.components.memory.approaches.experience_store.src.experience import ExperienceStore


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pgn", required=True)
    ap.add_argument("--scenario", default="gauntlet")
    args = ap.parse_args()
    store = ExperienceStore()
    n = 0
    with open(args.pgn) as f:
        while True:
            g = chess.pgn.read_game(f)
            if g is None:
                break
            b = g.board()
            ucis, epds = [], [b.epd()]
            for mv in g.mainline_moves():
                ucis.append(mv.uci())
                b.push(mv)
                epds.append(b.epd())
            res = g.headers.get("Result", "*")
            opp = g.headers.get("Black", "unknown")
            store.record_game(args.scenario, g.board().fen(),
                              "mate" if res == "1-0" else "fail",
                              {"1-0": "checkmate", "0-1": "checkmate",
                               "1/2-1/2": "draw"}.get(res, "unknown"),
                              ucis, epds, opponent=opp)
            n += 1
    store.close()
    print(f"VERDICT PGN_IMPORT games={n} from {args.pgn}", flush=True)


if __name__ == "__main__":
    main()
