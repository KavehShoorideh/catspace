#!/usr/bin/env python
"""experiments/m2c_elo_id.py -- M2c unknown-Elo path: can we RECOVER a hidden Elo from a player's moves?
(Kaveh 2026-07-27: "handle the edge case when we don't know elo".) Maia-2 scores moves conditioned on
11 rating buckets; the bucket whose move-distribution best explains the observed moves IS the rating
estimate. We hide each held-out player's true Elo, score their first N moves under ALL 11 buckets, and
form the posterior p(Elo | moves) ∝ prior(Elo) · Π_t p_maia(move_t | s_t, bucket). Report how the
recovered Elo (MAP + posterior-mean) approaches the true Elo as N grows -- the "wide prior tightens
from play" claim, quantified.
"""
from __future__ import annotations

import argparse, math, sys, time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

BUCKET_REP = {0: 1050, 10: 2050, **{k: 1100 + (k - 1) * 100 + 50 for k in range(1, 10)}}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--positions", default="data/derived/m2b/positions_3k.parquet")
    ap.add_argument("--n-players", type=int, default=150); ap.add_argument("--per-player", type=int, default=60)
    ap.add_argument("--observed", type=int, nargs="+", default=[5, 10, 20, 40])
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    t0 = time.time(); rng = np.random.default_rng(args.seed); Ns = sorted(args.observed)
    from catspace.train.scaffold import resolve_device
    dev = resolve_device(args.device)
    from maia2 import model as maia_model, inference
    from maia2.inference import map_to_category
    amd, elo_dict, _ = inference.prepare()

    t = pq.read_table(args.positions).to_pydict()
    hp = defaultdict(list)
    for i in range(len(t["fen"])):
        if t["split"][i] == "heldout":
            hp[t["player_id"][i]].append(i)
    players = list(hp); rng.shuffle(players); players = players[:args.n_players]
    # population prior over buckets (from ALL held-out players' true Elo)
    prior = np.ones(11)
    for pl in hp:
        prior[map_to_category(int(np.median([t["elo_self"][i] for i in hp[pl]])), elo_dict)] += 1
    logprior = np.log(prior / prior.sum())

    # sample per-player positions; build (position x 11 buckets) inference rows, deduped
    rows = {}; order = []; per = []
    for pl in players:
        idx = hp[pl][:]; rng.shuffle(idx); idx = idx[:args.per_player]
        if len(idx) < Ns[0]:
            continue
        true_cat = map_to_category(int(np.median([t["elo_self"][i] for i in idx])), elo_dict)
        rec = {"true_cat": true_cat, "moves": []}
        for i in idx:
            oppo = BUCKET_REP[map_to_category(int(t["elo_oppo"][i]), elo_dict)]
            keys = []
            for b in range(11):
                k = (t["fen"][i], b)
                if k not in rows:
                    rows[k] = len(order); order.append((t["fen"][i], t["played"][i], BUCKET_REP[b], oppo))
                keys.append(rows[k])
            rec["moves"].append((t["played"][i], keys))
        per.append(rec)
    df = pd.DataFrame(order, columns=["fen", "move", "elo_self", "elo_oppo"])
    print(f"[elo-id] {len(per)} players | {len(df):,} (position x bucket) Maia rows [{time.time()-t0:.0f}s]", flush=True)
    maia = maia_model.from_pretrained(type="rapid", device=str(dev))
    df, _ = inference.inference_batch(df, maia, verbose=False, batch_size=512, num_workers=0)
    probs = df["move_probs"].tolist()

    # per player, per N: posterior over buckets from first N observed moves
    results = {N: {"err": [], "correct": []} for N in Ns}
    for rec in per:
        ll = np.zeros(11)
        for n, (mv, keys) in enumerate(rec["moves"], 1):
            for b in range(11):
                ll[b] += math.log(max(probs[keys[b]].get(mv, 1e-6), 1e-6))
            if n in results:
                post = np.exp((ll + logprior) - (ll + logprior).max()); post /= post.sum()
                map_cat = int(post.argmax()); mean_elo = float(sum(post[b] * BUCKET_REP[b] for b in range(11)))
                results[n]["err"].append(abs(mean_elo - BUCKET_REP[rec["true_cat"]]))
                results[n]["correct"].append(int(map_cat == rec["true_cat"]))

    print(f"\n===== M2c unknown-Elo recovery | {len(per)} held-out players (true Elo hidden) =====")
    print(f"  {'N moves':>8} {'Elo MAE':>10} {'exact-bucket':>14} {'within-1-bucket':>16}")
    for N in Ns:
        err = np.array(results[N]["err"]); cor = np.array(results[N]["correct"])
        if not len(err):
            continue
        # within-1-bucket: MAE <= 100 as a proxy (bucket width)
        print(f"  {N:>8} {err.mean():>9.0f} {100*cor.mean():>13.0f}% {100*np.mean(err<=100):>15.0f}%")
    base_mae = np.mean([abs(BUCKET_REP[rec["true_cat"]] - 1500) for rec in per])
    print(f"  (no-info baseline: guess 1500 for everyone -> Elo MAE {base_mae:.0f})")
    print(f"VERDICT m2c-elo-id done [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
