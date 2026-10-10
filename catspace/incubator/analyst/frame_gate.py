"""THE GATES BEFORE A CONDITION SEARCH (Kaveh 2026-10-09: "I like the idea of a cheap gate followed by a more thorough
gate"; the audit: 96% of the nodes went to condition searches that found nothing).

  gate 1  a linear score over features that cost nothing (rust waiting.rs GATE_FEATURES: the family, the goal, the
          target's value, its escapes, how many candidate changes, the seed's kind, the frame's status last move, ...):
          below `threshold` the frame is left UNSEARCHED. The seed proof still runs every move, so a tactic is never
          missed -- only a waiting condition can be.
  gate 2  frames scoring below `deep_threshold` get the 2-ply condition search only.

  collect   replay games with the frame log on: (features, conditions found, found within 2 plies, nodes) per frame
  train     logistic regression (scikit-learn) on the TRAIN games; thresholds chosen on out-of-fold scores: gate 1 keeps
            RECALL of the conditions, gate 2 keeps DEEP_RECALL of the conditions only the deep tier finds
  verify    the HELD-OUT games replayed with the gate off and on: time, nodes, conditions found, tactics found
"""
from __future__ import annotations

import argparse
import json
import os
import time

from catspace.io import paths

RECALL, DEEP_RECALL = 0.90, 0.90


def _collect_one(args):
    gid, fens = args
    os.environ.setdefault("RAYON_NUM_THREADS", "1")
    import analyst_core
    analyst_core.log_frames(True)
    pool = analyst_core.FramePool(True)
    t = time.time()
    for f in fens:
        pool.observe(f)
    return dict(game=gid, secs=time.time() - t, plies=len(fens), log=analyst_core.frame_log())


def collect(tag: str, games: int, workers: int) -> list:
    from multiprocessing import Pool
    from .waiting_frames import _games
    ck = paths.experiment(f"frame_gate_log_{tag}_{games}.jsonl")
    done = {}
    if os.path.exists(ck):
        with open(ck) as fh:
            for line in fh:
                try:
                    r = json.loads(line); done[r["game"]] = r
                except json.JSONDecodeError:
                    continue
    todo = [(g, f) for g, f in _games(tag, games) if g not in done]
    print(f"collect: {len(done)} games checkpointed, {len(todo)} to go", flush=True)
    t0 = time.time()
    if todo:
        with Pool(workers) as p, open(ck, "a") as fh:
            for i, r in enumerate(p.imap_unordered(_collect_one, todo, chunksize=1)):
                fh.write(json.dumps(r) + "\n"); fh.flush(); done[r["game"]] = r
                if (i + 1) % 50 == 0:
                    print(f"  {i + 1}/{len(todo)} games, {time.time() - t0:.0f} s", flush=True)
    print(f"TIME collect: {time.time() - t0:.0f} s", flush=True)
    return [done[g] for g in sorted(done)]


def train(runs: list):
    import numpy as np
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler
    import analyst_core
    names = analyst_core.gate_feature_names()
    X, found, shallow, nodes, groups = [], [], [], [], []
    for r in runs:
        for feats, f, s, n in r["log"]:
            X.append(feats); found.append(f); shallow.append(s); nodes.append(n); groups.append(r["game"])
    X = np.array(X, dtype=float); found = np.array(found); shallow = np.array(shallow); nodes = np.array(nodes, dtype=float)
    groups = np.array(groups); y = (found > 0).astype(int)
    print(f"OBS {len(y)} frames searched for conditions in {len(runs)} train games; {y.sum()} found any ({100 * y.mean():.1f}%), "
          f"{found.sum()} conditions, {(found - shallow).sum()} of them only at the deep tier; {nodes.sum() / 1e9:.2f}G condition nodes")
    oof = np.zeros(len(y)); oof_t = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
        sc = StandardScaler().fit(X[tr]); m = LogisticRegression(max_iter=3000).fit(sc.transform(X[tr]), y[tr])
        oof[te] = m.decision_function(sc.transform(X[te]))
        oof_t[te] = HistGradientBoostingClassifier(max_iter=200).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    print(f"OBS gate model, out of fold: logistic AUC {roc_auc_score(y, oof):.3f}; boosted trees AUC {roc_auc_score(y, oof_t):.3f} "
          f"(the ceiling for these features; the Rust gate is the logistic)")
    order = np.argsort(-oof); cf = np.cumsum(found[order]) / max(found.sum(), 1); cn = np.cumsum(nodes[order]) / max(nodes.sum(), 1)
    for keep in (0.5, 0.75, 0.9, 0.95):
        i = int(np.searchsorted(cf, keep))
        print(f"OBS   keep {int(100 * keep)}% of the conditions: search {100 * (i + 1) / len(y):.0f}% of the frames, {100 * cn[i]:.0f}% of the condition nodes")
    i = int(np.searchsorted(cf, RECALL)); threshold = float(oof[order][i])
    deep_only = found - shallow
    kept = oof >= threshold
    o2 = np.argsort(-oof[kept]); cd = np.cumsum(deep_only[kept][o2]) / max(deep_only[kept].sum(), 1)
    j = int(np.searchsorted(cd, DEEP_RECALL)); deep_threshold = float(oof[kept][o2][min(j, len(o2) - 1)])
    print(f"OBS   gate 1 threshold {threshold:.3f} (keeps {int(100 * RECALL)}% of the conditions); gate 2 deep threshold "
          f"{deep_threshold:.3f}: {100 * (j + 1) / max(kept.sum(), 1):.0f}% of the searched frames go deep, keeping {int(100 * DEEP_RECALL)}% of the deep-only conditions")
    sc = StandardScaler().fit(X); m = LogisticRegression(max_iter=3000).fit(sc.transform(X), y)
    w = m.coef_[0] / sc.scale_; b = float(m.intercept_[0] - (m.coef_[0] * sc.mean_ / sc.scale_).sum())
    print("OBS   weights (standardized):", ", ".join(f"{n} {c:+.2f}" for c, n in sorted(zip(m.coef_[0], names), key=lambda t: -abs(t[0]))[:8]))
    gate = dict(features=names, w=[float(x) for x in w], b=b, threshold=threshold, deep_threshold=deep_threshold,
                recall=RECALL, deep_recall=DEEP_RECALL, train_games=[r["game"] for r in runs])
    out = paths.derived("frame_gate.json")
    with open(out, "w") as fh:
        json.dump(gate, fh, indent=1)
    print(f"gate written to {out}")
    return gate


def _verify_one(args):
    gid, fens, gate = args
    os.environ.setdefault("RAYON_NUM_THREADS", "1")
    import analyst_core
    if gate:
        analyst_core.set_gate(gate["w"], gate["b"], gate["threshold"], gate["deep_threshold"])
    else:
        analyst_core.clear_gate()
    analyst_core.audit()
    pool = analyst_core.FramePool(True)
    t = time.time(); tactics = conds = dormant = 0
    for f in fens:
        pool.observe(f)
        stm = "w" if " w " in f else "b"
        fr = pool.frames()
        tactics += sum(1 for x in fr if x[0] == stm and x[6] == "active")
        dormant += sum(1 for x in fr if x[6] == "dormant"); conds += sum(len(x[7]) for x in fr)
    a = analyst_core.audit()
    nodes = sum(a[2 * k + 1] for k in (0, 1, 2, 3, 4, 9, 10, 11))
    return dict(game=gid, gate=bool(gate), secs=time.time() - t, plies=len(fens), tactics=tactics, dormant=dormant,
                conds=conds, nodes=nodes, gated=a[2 * 17], shallow=a[2 * 18])


def verify(held: list, gate: dict, workers: int) -> None:
    from multiprocessing import Pool
    t0 = time.time()
    jobs = [(g, f, gt) for g, f in held for gt in (None, gate)]
    with Pool(workers) as p:
        res = list(p.imap_unordered(_verify_one, jobs, chunksize=1))
    for on in (False, True):
        rs = [r for r in res if r["gate"] == on]
        pl = sum(r["plies"] for r in rs)
        print(f"OBS held-out {len(rs)} games, gate {'ON ' if on else 'OFF'}: {1000 * sum(r['secs'] for r in rs) / pl:.0f} ms a position on one core, "
              f"{sum(r['nodes'] for r in rs) / pl:,.0f} nodes; frame-move readings: tactics for the side to move {sum(r['tactics'] for r in rs)}, "
              f"dormant {sum(r['dormant'] for r in rs)}, conditions {sum(r['conds'] for r in rs)}; frames unsearched {sum(r['gated'] for r in rs)}, "
              f"searched at 2 plies only {sum(r['shallow'] for r in rs)}")
    print(f"TIME verify: {time.time() - t0:.0f} s", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="c300")
    ap.add_argument("--games", type=int, default=300)
    ap.add_argument("--held", type=int, default=60, help="the last games by id, kept out of training")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--verify-only", action="store_true", help="re-verify the saved gate on the held-out games")
    a = ap.parse_args(argv)
    t0 = time.time()
    if a.verify_only:
        from .waiting_frames import _games
        gate = json.load(open(paths.derived("frame_gate.json")))
        held = [(g, f) for g, f in _games(a.tag, a.games) if g not in set(gate["train_games"])]
        verify(held, gate, a.workers)
        return
    runs = collect(a.tag, a.games, a.workers)
    train_runs, held_ids = runs[: len(runs) - a.held], {r["game"] for r in runs[len(runs) - a.held:]}
    t = time.time(); gate = train(train_runs); print(f"TIME train: {time.time() - t:.0f} s", flush=True)
    from .waiting_frames import _games
    held = [(g, f) for g, f in _games(a.tag, a.games) if g in held_ids]
    verify(held, gate, a.workers)
    print(f"TIME all: {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
