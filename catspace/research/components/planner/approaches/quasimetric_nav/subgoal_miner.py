#!/usr/bin/env python
"""subgoal_miner.py -- mine candidate SUBGOALS from game trajectories (Kaveh 2026-08-11).

DEFINITION (Kaveh): a subgoal is an INVARIANCE CLASS -- an advantage whose value does not
depend on precise placement/move order, i.e. where the field's smooth interpolation is valid.

Pipeline:
  1. DETECT candidate moments in games, two generators:
       jump  -- committor transition states: |dE| >= jump-thr across one ply and the change
                PERSISTS (mean E after vs before)
       elbow -- coherence elbows: the ply where the winner's length-ruler descent turns
                monotone for the rest of the game (>= elbow-frac of remaining steps decrease)
  2. CLUSTER the moment-states in the quasimetric embedding (k-means).
  3. GATE by invariance: keep clusters whose members (a) span >= min-games distinct games,
     (b) agree in expected points (std <= e-std) AND in distance-to-own-ending (std <= d-std
     plies) -- the cluster's internal diversity IS the perturbation set.
  4. RANK by size x invariance; emit exemplar FENs per surviving cluster.

    .venv/bin/python -m ...subgoal_miner --ckpt <field.pt> [--games 3000] [--k 64]
writes artifacts/experiments/subgoal_candidates.jsonl
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from catspace.io import paths
from catspace.research.components.encoder.approaches.reach_probability.experiments.eval_dtz_gate import (
    row_to_board)
from catspace.research.components.encoder.approaches.reach_probability.experiments.interpret_reach import (
    load_net)
from catspace.research.components.encoder.approaches.reach_probability.src import trajectories as T


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--games", type=int, default=3000)
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--jump-thr", type=float, default=0.25)
    ap.add_argument("--elbow-frac", type=float, default=0.75)
    ap.add_argument("--min-games", type=int, default=8)
    ap.add_argument("--e-std", type=float, default=0.08)
    ap.add_argument("--d-std", type=float, default=3.0)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    net, pay = load_net(args.ckpt, args.device)
    c = pay["cfg"]
    tr = T.build(n_human=0, n_sf=c["games"], seed=c["traj_seed"], max_plies=c["max_plies"],
                 n_piecedown=c.get("n_piecedown", 0), verbose=False)
    game = tr.game_of_row()
    res = np.repeat(tr.result, tr.length)
    pn = c["pole_names"]
    P = net.poles.poles.detach().float().to(args.device)
    pidx = [pn.index(k) for k in ("WIN", "DRAW", "LOSS")]
    rng = np.random.default_rng(0)
    gids = rng.choice(np.unique(game), args.games, replace=False)

    cand_z, cand_row, cand_E, cand_dOwn, cand_kind, cand_game = [], [], [], [], [], []
    B = 4096
    for gi in gids:
        rows = np.flatnonzero(game == gi)
        if len(rows) < 12:
            continue
        r = int(res[rows[0]])
        with torch.no_grad():
            z = net.encode_q(torch.from_numpy(tr.tok[rows].astype(np.int64)).to(args.device),
                             torch.from_numpy(tr.glob[rows].astype(np.float32)).to(args.device))
            DB = torch.stack([net.dB(z, P[[k]].expand(len(z), -1)) for k in pidx], 1)
            pr = torch.softmax(-DB / 5.0, 1).float().cpu().numpy()
            DA = torch.stack([net.dA(z, P[[k]].expand(len(z), -1)) for k in pidx], 1)
            DA = DA.float().cpu().numpy()
        E = pr[:, 0] + 0.5 * pr[:, 1]
        zc = z.float().cpu().numpy()
        # jumps: |dE| big and persistent
        dE = np.diff(E)
        for t in np.flatnonzero(np.abs(dE) >= args.jump_thr):
            if t + 3 < len(E) and t >= 2:
                before, after = E[max(0, t - 3):t + 1].mean(), E[t + 1:t + 4].mean()
                if abs(after - before) >= args.jump_thr * 0.8:
                    cand_z.append(zc[t + 1]); cand_row.append(int(rows[t + 1]))
                    cand_E.append(float(E[t + 1])); cand_kind.append("jump")
                    own = 0 if r == 1 else (2 if r == -1 else 1)
                    cand_dOwn.append(float(DA[t + 1, own])); cand_game.append(int(gi))
        # elbow: winner's descent turns monotone (decisive games only)
        if r != 0:
            own = 0 if r == 1 else 2
            d = DA[:, own]
            n = len(d)
            for t in range(4, n - 6):
                tail = np.diff(d[t:])
                head = np.diff(d[max(0, t - 6):t + 1])
                if (tail < 0).mean() >= args.elbow_frac and (head < 0).mean() <= 0.5:
                    cand_z.append(zc[t]); cand_row.append(int(rows[t]))
                    cand_E.append(float(E[t])); cand_kind.append("elbow")
                    cand_dOwn.append(float(d[t])); cand_game.append(int(gi))
                    break                              # one elbow per game: the earliest

    print(f"[miner] {len(cand_z):,} candidate moments "
          f"({sum(1 for k in cand_kind if k=='jump'):,} jumps, "
          f"{sum(1 for k in cand_kind if k=='elbow'):,} elbows) from {args.games:,} games")
    if len(cand_z) < args.k * 4:
        raise SystemExit("[miner] too few candidates -- loosen thresholds")
    Z = np.stack(cand_z)
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=args.k, n_init=4, random_state=0).fit(Z)
    lab = km.labels_
    E_ = np.array(cand_E); dOwn = np.array(cand_dOwn); gm = np.array(cand_game)
    out_rows = []
    for k in range(args.k):
        m = lab == k
        ngames = len(set(gm[m].tolist()))
        if m.sum() < args.min_games or ngames < args.min_games:
            continue
        e_std, d_std = float(E_[m].std()), float(dOwn[m].std())
        ok = e_std <= args.e_std and d_std <= args.d_std
        exemplars = []
        idx = np.flatnonzero(m)[:200]
        # pick 3 maximally-different exemplars (farthest-point in Z)
        sel = [int(idx[0])]
        for _ in range(2):
            dd = np.linalg.norm(Z[idx] - Z[sel].mean(0), axis=1)
            sel.append(int(idx[int(np.argmax(dd))]))
        for j in sel:
            b = row_to_board(tr.tok[cand_row[j]], tr.glob[cand_row[j]])
            if b.is_valid():
                exemplars.append(b.fen())
        out_rows.append({"rows": [int(cand_row[j]) for j in np.flatnonzero(m)[:300]],
                         "cluster": int(k), "n": int(m.sum()), "games": ngames,
                         "kinds": {kk: int(sum(1 for j in np.flatnonzero(m)
                                               if cand_kind[j] == kk))
                                   for kk in ("jump", "elbow")},
                         "E_mean": round(float(E_[m].mean()), 3), "E_std": round(e_std, 3),
                         "dOwn_mean": round(float(dOwn[m].mean()), 2),
                         "dOwn_std": round(d_std, 2),
                         "invariant": bool(ok), "fens": exemplars})
    out_rows.sort(key=lambda r: (-int(r["invariant"]), -r["n"] * (1.0 / (1e-3 + r["E_std"]))))
    out_path = args.out or paths.experiment("subgoal_candidates.jsonl")
    with open(out_path, "w") as f:
        for r in out_rows:
            f.write(json.dumps(r) + "\n")
    inv = sum(1 for r in out_rows if r["invariant"])
    print(f"[miner] {len(out_rows)} populated clusters, {inv} pass the INVARIANCE gate "
          f"-> {out_path}")
    for r in out_rows[:8]:
        print(f"  c{r['cluster']:>3} n={r['n']:>4} games={r['games']:>3} {r['kinds']} "
              f"E {r['E_mean']:.2f}±{r['E_std']:.2f}  dOwn {r['dOwn_mean']:.1f}±{r['dOwn_std']:.1f}"
              f"  {'INVARIANT' if r['invariant'] else 'sharp'}")


if __name__ == "__main__":
    main()
