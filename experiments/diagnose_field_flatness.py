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

from catspace.fields import FieldModel
from catspace.research.components.planner.approaches.endgame_groundtruth.src.tb import TB


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", required=True)
    ap.add_argument("--bank-file", required=True)
    ap.add_argument("--loss-bank-file", default=None)
    ap.add_argument("--field", default="data/derived/sep/lichess_mc2.pt")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--flat-eps", type=float, default=0.01,
                    help="value spread below this = PLATEAU")
    ap.add_argument("--support-shards", default=None,
                    help="rollout shard dir(s), comma-separated: kNN support query (Kaveh: "
                         "'search to see if we have data points with embeddings near this "
                         "region') -- separates PLATEAU into coverage-hole vs geometry-flat")
    ap.add_argument("--support-n", type=int, default=100_000)
    ap.add_argument("--energy-ckpt", default=None,
                    help="also rank the tb move under the energy PRIOR (double-handicap check)")
    args = ap.parse_args()
    fm = FieldModel(args.field, device=args.device)
    tb = TB()
    pfn = None
    if args.energy_ckpt:
        from experiments.mate_ladder_eval import make_energy_prior
        pfn = make_energy_prior(ckpt=args.energy_ckpt)

    sup_B = sup_ref = None
    if args.support_shards:
        import glob
        rng = np.random.default_rng(0)
        P_, M_ = [], []
        for d in args.support_shards.split(","):
            files = sorted(glob.glob(str(Path(d) / "shard_*.npz")))
            rng.shuffle(files)
            for fpath in files:
                z = np.load(fpath)
                P_.append(z["packed"]); M_.append(z["meta"])
                if sum(len(p) for p in P_) >= args.support_n:
                    break
        P = np.concatenate(P_)[:args.support_n]; M = np.concatenate(M_)[:args.support_n]
        sup_B = fm.embed_B(P, M)
        # self-calibration: NN distance distribution WITHIN the data (Euclidean, B-space)
        idx = rng.choice(len(sup_B), 1000, replace=False)
        q = sup_B[idx]
        d2 = ((q[:, None, :] - sup_B[None, ::37, :]) ** 2).sum(-1) ** 0.5   # strided ref pool
        d2.sort(1)
        sup_ref = np.median(d2[:, 1])          # typical NN distance among the data itself
        print(f"[support] {len(sup_B)} states embedded; self-NN median {sup_ref:.3f}", flush=True)

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
            prio_txt = ""
            if pfn is not None and best_tb is not None:
                pri = pfn(pb)
                pv = np.array([pri.get(m, 0.0) for m in moves])
                rank_pri = int((-pv).argsort().tolist().index(best_tb)) + 1
                prio_txt = f" prior-rank {rank_pri}/{len(moves)} (p={pv[best_tb]:.3f})"
            progress = bool((d_kids < d_here).any())
            kind = ("PLATEAU" if spread < args.flat_eps
                    else ("SEARCH-MISS" if 0 < rank_tb <= 3 else "FIELD-WRONG"))
            sup_txt = ""
            if sup_B is not None:
                pb_B = fm.embed_B_boards([pb])[0]
                d_eu = np.sort(((sup_B - pb_B) ** 2).sum(-1) ** 0.5)[:10]
                d_iqe = np.sort(fm.d_to_bank(fm.embed_F_boards([pb]), sup_B))  # scalar: min
                ratio = d_eu[0] / max(sup_ref, 1e-9)
                support = "SUPPORTED" if ratio < 3.0 else "COVERAGE-HOLE"
                sup_txt = (f" | knn: eu-NN {d_eu[0]:.3f} ({ratio:.1f}x self-NN) "
                           f"nn10-med {np.median(d_eu):.3f} iqe-min {float(d_iqe[0]):.2f} -> {support}")
                diags.append(dict(g=f["g"], term=f["term"], spread=spread, rank_tb=rank_tb,
                                  n_moves=len(moves), progress=progress, kind=kind,
                                  support=support))
            else:
                diags.append(dict(g=f["g"], term=f["term"], spread=spread, rank_tb=rank_tb,
                                  n_moves=len(moves), progress=progress, kind=kind))
            print(f"  g{f['g']:03d} {f['term'][:10]:10s} cycle-pos spread={spread:.4f} "
                  f"tb-move rank {rank_tb}/{len(moves)}{prio_txt} field-progress={progress} "
                  f"-> {kind}{sup_txt}", flush=True)

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
