"""sharpness_check.py -- can the SHARPNESS of the position ten plies ahead be predicted from the present knowledge,
without predicting the tactic (Kaveh 2026-09-11: "the future prediction has to predict the existence of tactics,
or rather the sharpness of the position; I don't think it can predict details")?

Labels come from the tactical layer's own answers at t+k (already in the dataset: the graph's position indicators
at every ply) and from the engine's evaluations along the game: a big threat in the ledger, a hot zone, a mate
threat, a piece falling, the engine's best move forcing, the evaluation's volatility over the window, a swing.
Predictors: logistic regression from the knowledge vector (211 features), from the 15 regime indicators alone,
and PERSISTENCE (the same indicator now). Everything on held-out games.

    python -m catspace.incubator.analyst.sharpness_check --data data/derived/knowledge
"""
from __future__ import annotations

import argparse
import random
import time

import numpy as np

from .dataset import load
from .graph import POSITION_QUESTIONS
from .trajectory_check import features


def main(argv=None):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    ap = argparse.ArgumentParser(); ap.add_argument("--data", default="data/derived/knowledge"); ap.add_argument("--k", type=int, default=10)
    a = ap.parse_args(argv)
    t0 = time.perf_counter()
    d = load(a.data); meta = d["meta"]; pos = d["position"]
    X = features(d)
    P = {q: i for i, q in enumerate(POSITION_QUESTIONS)}
    index = {(m["game"], m["ply"]): i for i, m in enumerate(meta)}
    games = sorted({m["game"] for m in meta}); rng = random.Random(0); rng.shuffle(games)
    n = len(games); test_games = set(games[: n // 10])
    split_test = np.array([m["game"] in test_games for m in meta]); split_train = ~split_test & np.array([m["game"] not in set(games[n // 10: n // 5]) for m in meta])
    # engine evaluation from White's view along the game
    cp_w = np.array([(m["best_cp"] if m["side"] else -m["best_cp"]) if m["best_cp"] is not None else 0.0 for m in meta], dtype=np.float32)
    cp_w = np.clip(cp_w, -1500, 1500)
    k = a.k
    fut = np.array([index.get((m["game"], m["ply"] + k), -1) for m in meta])
    has = fut >= 0
    win_idx = [[index.get((m["game"], m["ply"] + j), -1) for j in range(1, k + 1)] for m in meta]
    def at(key, idx):
        return pos[idx, P[key]]
    # the labels at t+k, and over the window
    big_threat_now = np.maximum(pos[:, P["threat_ours"]], pos[:, P["threat_theirs"]]) >= 3
    labels = {}
    fi = np.where(has, fut, 0)
    labels["big threat in the ledger at t+k"] = (np.maximum(at("threat_ours", fi), at("threat_theirs", fi)) >= 3, big_threat_now)
    labels["mate threat at t+k"] = ((at("mate_ours", fi) + at("mate_theirs", fi)) > 0, (pos[:, P["mate_ours"]] + pos[:, P["mate_theirs"]]) > 0)
    labels["hot zone at t+k"] = (at("hottest", fi) > 0, pos[:, P["hottest"]] > 0)
    labels["a piece falls at t+k"] = ((at("falls_ours", fi) + at("falls_theirs", fi)) > 0, (pos[:, P["falls_ours"]] + pos[:, P["falls_theirs"]]) > 0)
    def window_any(key_fn, now_fn):
        out = np.zeros(len(meta), bool)
        for i, w in enumerate(win_idx):
            ids = [j for j in w if j >= 0]
            out[i] = any(key_fn(j) for j in ids) if ids else False
        return out
    big_win = window_any(lambda j: max(pos[j, P["threat_ours"]], pos[j, P["threat_theirs"]]) >= 3, None)
    labels[f"big threat anywhere in the next {k} plies"] = (big_win, big_threat_now)
    forcing = np.array([m["best_kind"] != "quiet" for m in meta])
    forcing_share = np.array([np.mean([forcing[j] for j in w if j >= 0]) if any(j >= 0 for j in w) else 0.0 for w in win_idx])
    labels[f"engine's best move forcing in half the next {k} plies"] = (forcing_share >= 0.5, forcing)
    vol = np.array([np.std([cp_w[j] for j in w if j >= 0]) if sum(1 for j in w if j >= 0) >= 3 else 0.0 for w in win_idx])
    labels[f"evaluation volatility >= 100 cp over the next {k} plies"] = (vol >= 100, np.abs(cp_w) >= 100)
    swing = np.abs(cp_w[fi] - cp_w) >= 150
    labels[f"evaluation swing >= 150 cp by t+{k}"] = (swing, np.abs(cp_w) >= 150)
    mu, sd = X[split_train].mean(0), X[split_train].std(0) + 1e-6; Xs = (X - mu) / sd
    R = pos[:, [P[q] for q in ("material_diff", "phase", "pieces", "queens_on", "passed_ours", "passed_theirs", "runner_ours", "runner_theirs", "pawns_ours", "pawns_theirs", "open_files", "castle_ours", "castle_theirs", "in_check", "rooks_connected_ours")]]
    Rs = (R - R[split_train].mean(0)) / (R[split_train].std(0) + 1e-6)
    print(f"[sharp] {len(meta)} positions; train {split_train.sum()} test {split_test.sum()}; k={k}; loaded in {time.perf_counter() - t0:.0f}s", flush=True)
    print(f"  {'label':<58}{'base':>6}{'knowledge':>11}{'regime':>8}{'persist':>9}{'know+pers':>11}")
    for name, (y, now) in labels.items():
        tr = split_train & has; te = split_test & has
        if y[tr].sum() < 30 or y[te].sum() < 10 or (~y[te]).sum() < 10:
            continue
        def auc(feats):
            clf = LogisticRegression(max_iter=1000, C=0.3).fit(feats[tr], y[tr])
            return roc_auc_score(y[te], clf.predict_proba(feats[te])[:, 1])
        a_k = auc(Xs); a_r = auc(Rs); a_p = roc_auc_score(y[te], now[te].astype(float)); a_kp = auc(np.concatenate([Xs, now[:, None].astype(np.float32)], 1))
        print(f"  {name:<58}{y[te].mean():>6.2f}{a_k:>11.3f}{a_r:>8.3f}{a_p:>9.3f}{a_kp:>11.3f}", flush=True)
    print("  (AUC: 0.5 = nothing; base = the label's rate on the test split; persist = the same fact about the present position)")


if __name__ == "__main__":
    main()
