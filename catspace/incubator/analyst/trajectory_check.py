"""trajectory_check.py -- is there a trajectory in the knowledge? The cheap answer before any network (Kaveh
2026-09-09): from the knowledge vector at ply t (the position indicators plus the node answers pooled per side:
mean and max over our pieces and theirs), predict the regime indicators 10 and 20 plies later by ridge regression,
against PERSISTENCE (the indicator's value now) and the MEAN (the training mean). Also the pawn arrow without any
model: how many of the regime indicators move monotonically along a game (the pawns count never rises, the phase
never rises), which is what a quasimetric can read.

    python -m catspace.incubator.analyst.trajectory_check --data data/derived/knowledge
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from .dataset import load, REGIME_KEYS
from .graph import NODE_QUESTIONS, POSITION_QUESTIONS
from . import knowledge_net as K


def features(d) -> np.ndarray:
    nodes, mask, pos = d["nodes"], d["mask"], d["position"]
    ours_col = NODE_QUESTIONS.index("ours")
    ours = (nodes[..., ours_col] > 0.5) & mask; theirs = (nodes[..., ours_col] < 0.5) & mask
    def pool(sel):
        w = sel[..., None].astype(np.float32)
        n = np.maximum(w.sum(1), 1.0)
        mean = (nodes * w).sum(1) / n
        mx = np.where(sel[..., None], nodes, -1e9).max(1); mx = np.where(mx < -1e8, 0.0, mx)
        return np.concatenate([mean, mx], axis=1)
    return np.concatenate([pos, pool(ours), pool(theirs)], axis=1).astype(np.float32)


def main(argv=None):
    from sklearn.linear_model import Ridge, LogisticRegression
    ap = argparse.ArgumentParser(); ap.add_argument("--data", default="data/derived/knowledge"); ap.add_argument("--max", type=int, default=0)
    a = ap.parse_args(argv)
    t0 = time.perf_counter()
    d = load(a.data)
    meta = d["meta"]
    X = features(d)
    data = K.Data.__new__(K.Data); data.meta = meta
    games = sorted({m["game"] for m in meta}); import random; rng = random.Random(0); rng.shuffle(games)
    n = len(games); test_games = set(games[: n // 10]); val_games = set(games[n // 10: n // 5])
    split = np.array(["test" if m["game"] in test_games else "val" if m["game"] in val_games else "train" for m in meta])
    tr, te = split == "train", split == "test"
    from .dataset import regime_of
    import chess
    now = np.array([regime_of(chess.Board(m["fen"]), bool(m["side"])) for m in meta], dtype=np.float32)
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6; Xs = (X - mu) / sd
    print(f"[traj] {len(meta)} positions, {X.shape[1]} features; train {tr.sum()} test {te.sum()}; loaded in {time.perf_counter() - t0:.0f}s", flush=True)
    for h in (10, 20):
        Y = np.array([m["ahead"][str(h)] for m in meta], dtype=np.float32)
        ended = np.array([m["ended_within"][str(h)] for m in meta], dtype=bool)
        keep_te = te & ~ended                                                     # score only where the future exists
        model = Ridge(alpha=10.0).fit(Xs[tr], Y[tr])
        pred = model.predict(Xs[keep_te])
        # persistence + ridge on the residual: does the knowledge add to 'nothing changes'?
        resid = Ridge(alpha=10.0).fit(Xs[tr], (Y - now)[tr]); pred_res = now[keep_te] + resid.predict(Xs[keep_te])
        var = Y[keep_te].var(0); live = var > 1e-6
        rows = []
        for j, k in enumerate(REGIME_KEYS):
            if not live[j]:
                continue
            r2 = lambda p: 1 - ((p - Y[keep_te, j]) ** 2).mean() / var[j]
            rows.append((k, r2(pred[:, j]), r2(now[keep_te, j]), r2(pred_res[:, j]), r2(np.full(keep_te.sum(), Y[tr, j].mean()))))
        print(f"[traj] +{h} plies (test positions with a future {keep_te.sum()}): R2 of ridge from the knowledge | persistence | persistence+ridge residual | train mean")
        print(f"  {'indicator':<24}{'ridge':>8}{'persist':>9}{'pers+res':>10}{'mean':>7}")
        for k, r, p, pr, m_ in rows:
            print(f"  {k:<24}{r:>8.3f}{p:>9.3f}{pr:>10.3f}{m_:>7.3f}")
        print(f"  {'MEAN over indicators':<24}{np.mean([r for _, r, _, _, _ in rows]):>8.3f}{np.mean([p for _, _, p, _, _ in rows]):>9.3f}{np.mean([pr for _, _, _, pr, _ in rows]):>10.3f}{np.mean([m_ for _, _, _, _, m_ in rows]):>7.3f}", flush=True)
        # discrete events: queens off / a passed pawn appears / material swings by 2+ within h plies -- logistic vs base rate
        qi = REGIME_KEYS.index("queens_on"); pi = REGIME_KEYS.index("passed_ours"); mi = REGIME_KEYS.index("material_diff")
        events = {"queens come off": (now[:, qi] > 0) & (Y[:, qi] == 0), "a passed pawn of ours appears": (now[:, pi] == 0) & (Y[:, pi] > 0),
                  "material swings by 2 or more": np.abs(Y[:, mi] - now[:, mi]) >= 2}
        for name, ev in events.items():
            sel_tr = tr & ~ended; sel_te = keep_te
            if ev[sel_tr].sum() < 20 or ev[sel_te].sum() < 5:
                continue
            clf = LogisticRegression(max_iter=500, C=0.5).fit(Xs[sel_tr], ev[sel_tr])
            p = clf.predict_proba(Xs[sel_te])[:, 1]; yv = ev[sel_te]
            base = yv.mean()
            ll = -np.mean(yv * np.log(p + 1e-9) + (1 - yv) * np.log(1 - p + 1e-9)); ll0 = -(base * np.log(base + 1e-9) + (1 - base) * np.log(1 - base + 1e-9))
            from sklearn.metrics import roc_auc_score
            print(f"  event '{name}' within {h} plies: base rate {base:.3f}; AUC {roc_auc_score(yv, p):.3f} (0.5 = nothing); log-loss {ll:.3f} against {ll0:.3f} for the base rate", flush=True)
    # the arrow without a model: monotone indicators along games
    by_game = {}
    for i, m in enumerate(meta):
        by_game.setdefault(m["game"], []).append((m["ply"], i))
    mono = {k: [0, 0] for k in ("pawns_ours", "pawns_theirs", "phase", "pieces")}
    for g, lst in by_game.items():
        lst.sort(); idx = [i for _, i in lst]
        for k in mono:
            j = REGIME_KEYS.index(k); v = now[idx, j]
            # from the side to move's view 'ours' alternates; use white's count: pawns_ours when side==white else pawns_theirs
            if k in ("pawns_ours", "pawns_theirs"):
                sides = np.array([meta[i]["side"] for i in idx], dtype=bool)
                jo, jt = REGIME_KEYS.index("pawns_ours"), REGIME_KEYS.index("pawns_theirs")
                v = np.where(sides, now[idx, jo], now[idx, jt]) if k == "pawns_ours" else np.where(sides, now[idx, jt], now[idx, jo])
            dv = np.diff(v)
            mono[k][0] += int((dv > 0).sum()); mono[k][1] += int(len(dv))
    print("[traj] the arrow without a model: steps where the indicator RISES along a game (a rise is impossible for pawn counts and phase except by promotion):")
    for k, (up, tot) in mono.items():
        print(f"  {k:<14} rises {up}/{tot} ({up / max(1, tot):.2%})")


if __name__ == "__main__":
    main()
