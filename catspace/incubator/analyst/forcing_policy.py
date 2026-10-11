"""THE STOP POLICY of the forcing search (Kaveh 2026-10-11: "a policy on when to stop, preferably RL"; the frame: search
as an RL action, reward = outcome - lambda * effort).

PHASE 1 (this file): imitation of the search's own outcomes.
  collect   the planner over census games with the forcing search's log on and a LARGE budget (the ground truth): every
            line of ours the search created -- 14 generic features (rust/analyst_core/src/threat.rs policy_features), whether
            the line ended in a proof, whether the root did; rows tagged by game
  train     a two-layer net (14 -> 16 -> 1, PyTorch) for p(the line ends proven); games split by identity (held-out games)
  front     on the held-out games, for each threshold tau: the RECALL of proven lines kept and the fraction of lines
            stopped (the effort saved) -- the recall/effort front; tau is chosen on it
  export    weights + standardization + tau as JSON for Rust (set_forcing_policy)
PHASE 2 (later): policy gradient on the deployed loop with the reward above.
"""
from __future__ import annotations

import argparse
import json
import os
import time

from catspace.io import paths


def _collect_game(args):
    gid, fens, budget = args
    os.environ.setdefault("RAYON_NUM_THREADS", "1")
    import analyst_core
    analyst_core.forcing_stats(budget)
    analyst_core.set_forcing_log(True)
    analyst_core.forcing_log()
    rows = []
    for fen in fens:
        analyst_core.plan_trace(fen)
        for f, proven, root in analyst_core.forcing_log():
            rows.append((gid, f, int(proven), int(root)))
    return rows


def collect(tag: str, games: int, budget: int, workers: int) -> None:
    from multiprocessing import Pool
    from .waiting_frames import _games
    t0 = time.time()
    jobs = [(g, f, budget) for g, f in _games(tag, games)]
    with Pool(workers) as p:
        rows = [r for rs in p.imap_unordered(_collect_game, jobs, chunksize=1) for r in rs]
    out = paths.derived("forcing_policy_rows.json")
    with open(out, "w") as fh:
        json.dump(rows, fh)
    pos = sum(r[2] for r in rows)
    print(f"OBS collected {len(rows):,} lines from {games} games at budget {budget} ({time.time() - t0:.0f} s on {workers} processes): "
          f"proven {pos:,} ({100 * pos / max(len(rows), 1):.1f}%), roots proven {sum(r[3] for r in rows):,} -> {out}")


def train(hidden: int, epochs: int, recall_target: float, seed: int) -> None:
    import numpy as np
    import torch
    rows = json.load(open(paths.derived("forcing_policy_rows.json")))
    games = sorted({r[0] for r in rows})
    rng = np.random.default_rng(seed)
    held = set(rng.choice(games, size=max(1, len(games) // 4), replace=False).tolist())
    X = np.array([r[1] for r in rows], dtype=np.float32)
    y = np.array([r[2] for r in rows], dtype=np.float32)
    g = np.array([r[0] for r in rows])
    tr = np.array([gg not in held for gg in g]); te = ~tr
    mean, std = X[tr].mean(0), X[tr].std(0) + 1e-6
    Xn = (X - mean) / std
    torch.manual_seed(seed)
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, 1))
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    xt, yt = torch.tensor(Xn[tr]), torch.tensor(y[tr])
    pos_w = torch.tensor((len(yt) - yt.sum()) / max(yt.sum(), 1.0))          # the proven lines are the minority: weighted
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=pos_w)
    for ep in range(epochs):
        perm = torch.randperm(len(xt))
        for i in range(0, len(xt), 4096):
            idx = perm[i:i + 4096]
            opt.zero_grad(); loss = lossf(net(xt[idx]).squeeze(1), yt[idx]); loss.backward(); opt.step()
    with torch.no_grad():
        p_te = torch.sigmoid(net(torch.tensor(Xn[te])).squeeze(1)).numpy()
        p_tr = torch.sigmoid(net(xt).squeeze(1)).numpy()
    from sklearn.metrics import roc_auc_score
    print(f"OBS train {tr.sum():,} lines ({len(games) - len(held)} games), held-out {te.sum():,} ({len(held)} games); "
          f"AUC train {roc_auc_score(y[tr], p_tr):.3f}, held-out {roc_auc_score(y[te], p_te):.3f}; base rate {y[te].mean():.3f}")
    # THE FRONT: recall of proven lines kept vs lines stopped, by tau
    yte = y[te]
    chosen = None
    print("OBS   tau    recall of proven lines   lines stopped")
    for tau in [0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]:
        keep = p_te >= tau
        recall = (keep & (yte == 1)).sum() / max((yte == 1).sum(), 1)
        stopped = 1 - keep.mean()
        print(f"OBS   {tau:.2f}   {recall:.4f}                  {stopped:.3f}")
        if recall >= recall_target and (chosen is None or stopped > chosen[1]):
            chosen = (tau, stopped, recall)
    if chosen is None:
        chosen = (0.02, 0.0, 1.0)
    tau = chosen[0]
    print(f"OBS   chosen tau {tau} (recall {chosen[2]:.4f} >= {recall_target}, stops {chosen[1]:.3f} of lines)")
    w1, b1 = net[0].weight.detach().numpy(), net[0].bias.detach().numpy()
    w2, b2 = net[2].weight.detach().numpy()[0], float(net[2].bias.detach().numpy()[0])
    out = paths.derived("forcing_policy.json")
    with open(out, "w") as fh:
        json.dump(dict(w1=w1.tolist(), b1=b1.tolist(), w2=w2.tolist(), b2=b2, mean=mean.tolist(), std=std.tolist(), tau=tau,
                       features=X.shape[1], hidden=hidden, trained_on=sorted(set(games) - held), held_out=sorted(held)), fh)
    print(f"OBS   -> {out}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect"); c.add_argument("--tag", default="c300"); c.add_argument("--games", type=int, default=40)
    c.add_argument("--budget", type=int, default=2000); c.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    t = sub.add_parser("train"); t.add_argument("--hidden", type=int, default=16); t.add_argument("--epochs", type=int, default=30)
    t.add_argument("--recall", type=float, default=0.99); t.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    if a.cmd == "collect":
        collect(a.tag, a.games, a.budget, a.workers)
    else:
        train(a.hidden, a.epochs, a.recall, a.seed)


if __name__ == "__main__":
    main()
