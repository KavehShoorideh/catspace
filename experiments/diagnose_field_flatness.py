#!/usr/bin/env python
"""experiments/diagnose_field_flatness.py -- WHY does the engine draw in won positions?
(Kaveh 2026-07-25: "diagnose the field in the draw cases to see why it's flat. MCTS should
find some path forward.")

Reads FAIL trajectories from a bootstrap results jsonl (start_epd + ucis, recorded since
71c6944), finds the CYCLE positions (White to move, position occurred >=2 in the game),
and scores EVERY legal move there under the engine's own value (WDL from the run's final
banks). Three mutually exclusive diagnoses per position:
  PLATEAU     -- value spread across moves ~ 0: the bank distance does not discriminate
                 locally (bank density / geometry issue)
  FIELD-WRONG -- spread is healthy but the tb-optimal move (referee only) ranks LOW:
                 the field actively prefers non-progress
  SEARCH-MISS -- spread healthy AND tb move ranked top-3 by the field: the value was
                 fine; the draw was a search/budget failure
VERDICT: counts of each + median spread + median tb-move rank + share of positions where
NO move reduces d_win (field sees no progress at all).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import chess
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from catspace.engine.fields import FieldModel
from catspace.tb import TB


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", required=True)
    ap.add_argument("--bank-file", required=True)
    ap.add_argument("--loss-bank-file", default=None)
    ap.add_argument("--field", default="data/derived/sep/lichess_mc2.pt")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--flat-eps", type=float, default=0.01,
                    help="value spread below this = PLATEAU")
    args = ap.parse_args()
    fm = FieldModel(args.field, device=args.device)
    tb = TB()

    bank = fm.embed_B_boards([chess.Board(e) for e in
                              Path(args.bank_file).read_text().splitlines() if e.strip()])
    print(f"[probe] bank {len(bank)} mates", flush=True)

    fails = [json.loads(ln) for ln in Path(args.results).read_text().splitlines()
             if ln.strip() and '"mate": false' in ln]
    fails = [f for f in fails if "ucis" in f]
    print(f"[probe] {len(fails)} FAIL trajectories  terms={Counter(f['term'] for f in fails)}",
          flush=True)

    diags = []
    for f in fails:
        b = chess.Board(f["start_epd"])
        seen: Counter = Counter()
        poss = []                                    # (epd, board) cycle positions, White to move
        for u in f["ucis"]:
            if b.turn == chess.WHITE:
                k = b.epd()
                seen[k] += 1
                if seen[k] >= 2 and all(p[0] != k for p in poss):
                    poss.append((k, b.copy(stack=False)))
            b.push(chess.Move.from_uci(u))
        for k, pb in poss:
            moves = list(pb.legal_moves)
            kids = []
            for m in moves:
                c = pb.copy(stack=False); c.push(m); kids.append(c)
            d_here = float(fm.d_boards_to_bank([pb], bank)[0])
            d_kids = fm.d_boards_to_bank(kids, bank)
            M = max(float(np.median(d_kids)), 1e-6)
            v_kids = np.exp(-d_kids / M) / (np.exp(-d_kids / M) + np.exp(-1.0))
            spread = float(v_kids.max() - v_kids.min())
            # tb referee: rank of the DTZ-optimal move under the engine value
            best_tb, best_dtz = None, None
            for i, c in enumerate(kids):
                w, d = tb.wdl_dtz(c)
                if w is not None and -w == 2:        # child is a win for White (mover POV flip)
                    dz = abs(d) if d is not None else 999
                    if best_dtz is None or dz < best_dtz:
                        best_tb, best_dtz = i, dz
            rank_tb = int((-v_kids).argsort().tolist().index(best_tb)) + 1 if best_tb is not None else -1
            progress = bool((d_kids < d_here).any())
            kind = ("PLATEAU" if spread < args.flat_eps
                    else ("SEARCH-MISS" if 0 < rank_tb <= 3 else "FIELD-WRONG"))
            diags.append(dict(g=f["g"], term=f["term"], spread=spread, rank_tb=rank_tb,
                              n_moves=len(moves), progress=progress, kind=kind))
            print(f"  g{f['g']:03d} {f['term'][:10]:10s} cycle-pos spread={spread:.4f} "
                  f"tb-move rank {rank_tb}/{len(moves)} field-progress={progress} -> {kind}",
                  flush=True)

    if diags:
        kinds = Counter(d["kind"] for d in diags)
        print(f"VERDICT FIELD_FLATNESS n={len(diags)} cycle positions from {len(fails)} FAILs: "
              f"{dict(kinds)}  med_spread={np.median([d['spread'] for d in diags]):.4f}  "
              f"med_tb_rank={np.median([d['rank_tb'] for d in diags if d['rank_tb'] > 0]):.0f}  "
              f"no-progress-share={np.mean([not d['progress'] for d in diags]):.2f}", flush=True)
    else:
        print("VERDICT FIELD_FLATNESS no cycle positions found (draws were not cycles?)", flush=True)
    tb.close()


if __name__ == "__main__":
    main()
