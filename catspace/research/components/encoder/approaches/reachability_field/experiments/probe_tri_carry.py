#!/usr/bin/env python
"""catspace/research/components/encoder/approaches/reachability_field/experiments/probe_tri_carry.py -- how far does the triangle-inequality carry work?
(Kaveh 2026-07-25: 'how far out does it work... is there a quick test... can we increase
that number and make it quicker, trading some precision while keeping mate skill high')

On REAL recorded trajectories: anchor at position t-k, candidates = the anchor's nearest-C
bank mates; estimate dmin at t from the subset. Report, per (k, C):
  - dmin error quantiles (est - exact, one-sided >= 0)
  - CHILD-RANKING fidelity at White positions: top-1 agreement + mean spearman of the
    legal children ordered by subset-dmin vs exact-dmin (the quantity MCTS consumes)
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import chess
import numpy as np


from catspace.fields import FieldModel
from catspace.io import paths


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bank-file", default=paths.experiment("wdlr6_bank_n5000_KRRvK-central.fens"))
    ap.add_argument("--field", default=paths.sep("lichess_mc2.pt"))
    ap.add_argument("--results-glob", default=paths.experiment("wdlr[34]_results_n5000_*.jsonl,artifacts/experiments/drawreplay_results.jsonl"))
    ap.add_argument("--ks", default="1,2,3,5,8,12")
    ap.add_argument("--cs", default="64,128,256")
    ap.add_argument("--max-rank-positions", type=int, default=40)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    from scipy.stats import spearmanr
    fm = FieldModel(args.field, device=args.device)
    bank_boards = [chess.Board(e) for e in
                   Path(args.bank_file).read_text().splitlines() if e.strip()]
    B = fm.embed_B_boards(bank_boards)
    print(f"[carry] bank {len(B)}", flush=True)

    trajs = []
    for pat in args.results_glob.split(","):
        for f in glob.glob(pat):
            for ln in Path(f).read_text().splitlines():
                if not ln.strip():
                    continue
                r = json.loads(ln)
                if "ucis" in r and len(r["ucis"]) >= 14:
                    trajs.append((r["start_epd"], r["ucis"]))
    print(f"[carry] {len(trajs)} trajectories", flush=True)

    ks = [int(x) for x in args.ks.split(",")]
    cs = [int(x) for x in args.cs.split(",")]
    errs = {(k, c): [] for k in ks for c in cs}
    rank_top1 = {(k, c): [] for k in ks for c in cs}
    rank_sp = {(k, c): [] for k in ks for c in cs}
    n_rank = 0

    for start_epd, ucis in trajs:
        b = chess.Board(start_epd)
        boards = [b.copy(stack=False)]
        for u in ucis:
            b.push(chess.Move.from_uci(u))
            boards.append(b.copy(stack=False))
        F = fm.embed_F_boards(boards)
        # exact dmin along the trajectory + full distance rows (for candidate sets)
        import torch
        rows = []
        bt = torch.from_numpy(B).to(fm.device)
        for s in range(0, len(F), 256):
            with torch.no_grad():
                rows.append(fm.fb.distance_matrix(
                    torch.from_numpy(F[s:s + 256]).to(fm.device), bt).cpu().numpy())
        D = np.concatenate(rows)                      # (T, bank)
        exact = D.min(1)
        for t in range(len(boards)):
            for k in ks:
                a = max(0, t - k)
                order = np.argsort(D[a])
                for c in cs:
                    est = D[t][order[:c]].min()
                    errs[(k, c)].append(float(est - exact[t]))
        # child-ranking fidelity at sampled White positions
        for t in range(0, len(boards), 4):
            if n_rank >= args.max_rank_positions or boards[t].turn != chess.WHITE:
                continue
            pb = boards[t]
            kids = []
            for m in pb.legal_moves:
                cb = pb.copy(stack=False); cb.push(m); kids.append(cb)
            if len(kids) < 4:
                continue
            Fk = fm.embed_F_boards(kids)
            rowsk = []
            for s in range(0, len(Fk), 256):
                with torch.no_grad():
                    rowsk.append(fm.fb.distance_matrix(
                        torch.from_numpy(Fk[s:s + 256]).to(fm.device), bt).cpu().numpy())
            Dk = np.concatenate(rowsk)
            ex_k = Dk.min(1); n_rank += 1
            for k in ks:
                a = max(0, t - k)
                order = np.argsort(D[a])
                for c in cs:
                    est_k = Dk[:, order[:c]].min(1)
                    rank_top1[(k, c)].append(int(np.argmin(est_k) == np.argmin(ex_k)))
                    sp = spearmanr(est_k, ex_k).correlation
                    if not np.isnan(sp):
                        rank_sp[(k, c)].append(float(sp))

    print(f"[carry] rank positions: {n_rank}", flush=True)
    print("VERDICT TRI_CARRY  (err plies: p50/p95/max | child-rank: top1 / spearman)", flush=True)
    for k in ks:
        for c in cs:
            e = np.array(errs[(k, c)])
            t1 = np.mean(rank_top1[(k, c)]) if rank_top1[(k, c)] else float("nan")
            sp = np.mean(rank_sp[(k, c)]) if rank_sp[(k, c)] else float("nan")
            print(f"  k={k:2d} C={c:3d}: err {np.median(e):.3f}/{np.percentile(e,95):.3f}/"
                  f"{e.max():.3f} | top1 {t1:.2f} spearman {sp:.3f}  (n={len(e)})", flush=True)


if __name__ == "__main__":
    main()
