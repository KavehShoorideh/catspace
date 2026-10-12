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
    analyst_core.set_forcing_policy_on(False)                     # the full search is the truth
    analyst_core.forcing_stats(budget)
    analyst_core.set_forcing_log(True)
    analyst_core.forcing_log()
    import random
    rng = random.Random(gid)
    rows = []
    # every third position; the proven lines (the minority) all kept, the others sampled to at most 20x the proven count
    # per position, so a game yields thousands of rows, not millions
    for i, fen in enumerate(fens):
        if i % 3: continue
        analyst_core.forcing_log()
        analyst_core.plan_trace(fen)
        log = analyst_core.forcing_log()
        pos = [r for r in log if r[1]]
        neg = [r for r in log if not r[1]]
        keep = pos + (rng.sample(neg, min(len(neg), max(200, 20 * len(pos))) ) if neg else [])
        for f, proven, root, cost in keep:
            rows.append((gid, f, int(proven), int(root), int(cost)))
    return rows


def collect(tag: str, games: int, budget: int, workers: int) -> None:
    from multiprocessing import Pool
    from .waiting_frames import _games
    t0 = time.time()
    jobs = [(g, f, budget) for g, f in _games(tag, games)]
    with Pool(workers) as p:
        rows = [r for rs in p.imap_unordered(_collect_game, jobs, chunksize=1) for r in rs]
    import numpy as np
    out = paths.derived("forcing_policy_rows.npz")
    np.savez_compressed(out, game=np.array([r[0] for r in rows]), X=np.array([r[1] for r in rows], dtype=np.float32),
                        y=np.array([r[2] for r in rows], dtype=np.int8), root=np.array([r[3] for r in rows], dtype=np.int8),
                        cost=np.array([r[4] for r in rows], dtype=np.int32))
    pos = sum(r[2] for r in rows)
    print(f"OBS collected {len(rows):,} lines from {games} games at budget {budget} ({time.time() - t0:.0f} s on {workers} processes): "
          f"proven {pos:,} ({100 * pos / max(len(rows), 1):.1f}%), roots proven {sum(r[3] for r in rows):,} -> {out}")


def train(hidden: int, epochs: int, recall_target: float, seed: int) -> None:
    import numpy as np
    import torch
    d = np.load(paths.derived("forcing_policy_rows.npz"))
    X, y, g = d["X"], d["y"].astype(np.float32), d["game"]
    games = sorted(set(g.tolist()))
    rng = np.random.default_rng(seed)
    held = set(rng.choice(games, size=max(1, len(games) // 4), replace=False).tolist())
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


def _rl_game(args):
    gid, fens, lam = args
    os.environ.setdefault("RAYON_NUM_THREADS", "1")
    import catspace.incubator.analyst  # noqa: F401  (loads the policy)
    import analyst_core
    analyst_core.set_rl_sample(True)
    analyst_core.rl_log()
    dec, eps = [], []
    for i, fen in enumerate(fens):
        if i % 3: continue
        analyst_core.plan_trace(fen)
        d, e = analyst_core.rl_log()
        dec += [(gid, *r) for r in d]; eps += [(gid, *r) for r in e]
    return dec, eps


def rl_collect(tag: str, games: int, skip: int, workers: int) -> None:
    """PHASE 2, the episodes: the stochastic policy's decisions and each search's verdict, truth and expansions."""
    from multiprocessing import Pool
    import numpy as np
    from .waiting_frames import _games
    t0 = time.time()
    jobs = [(g, f, 0.0) for g, f in _games(tag, skip + games)[skip:]]
    dec, eps = [], []
    with Pool(workers) as p:
        for d, e in p.imap_unordered(_rl_game, jobs, chunksize=1):
            dec += d; eps += e
    out = paths.derived("forcing_policy_episodes.npz")
    np.savez_compressed(out, d_game=np.array([r[0] for r in dec]), d_ep=np.array([r[1] for r in dec], dtype=np.int64),
                        d_X=np.array([r[2] for r in dec], dtype=np.float32), d_act=np.array([r[3] for r in dec], dtype=np.int8),
                        d_p=np.array([r[4] for r in dec], dtype=np.float32),
                        e_game=np.array([r[0] for r in eps]), e_ep=np.array([r[1] for r in eps], dtype=np.int64),
                        e_verdict=np.array([r[2] for r in eps], dtype=np.int8), e_truth=np.array([r[3] for r in eps], dtype=np.int8),
                        e_exp=np.array([r[4] for r in eps], dtype=np.int64))
    ev, et = np.array([r[2] for r in eps]), np.array([r[3] for r in eps])
    print(f"OBS rl-collect {games} games: {len(eps):,} episodes, {len(dec):,} decisions ({time.time() - t0:.0f} s on {workers} processes); "
          f"verdict = truth {np.mean(ev == et):.3f}, false 'no gain' {np.mean((ev == 0) & (et == 1)):.4f}, unknown {np.mean(ev == -1):.3f}; "
          f"expansions a search {np.mean([r[4] for r in eps]):.0f} -> {out}")


def train2(hidden: int, epochs: int, lam: float, seed: int) -> None:
    """PHASE 2 as a per-line decision (Kaveh's reward at the line: follow a line iff its chance of proving is worth its
    expected cost): one trunk, two heads -- p(the line proves) and log(expansions of its subtree); the deployed rule is
    p >= lam * cost (threat.rs Policy::follow). Games split by identity; the front printed for a few lambdas."""
    import numpy as np
    import torch
    d = np.load(paths.derived("forcing_policy_rows.npz"))
    X, y, g, cost = d["X"], d["y"].astype(np.float32), d["game"], d["cost"].astype(np.float32)
    games = sorted(set(g.tolist()))
    rng = np.random.default_rng(seed)
    held = set(rng.choice(games, size=max(1, len(games) // 4), replace=False).tolist())
    tr = np.array([gg not in held for gg in g]); te = ~tr
    mean, std = X[tr].mean(0), X[tr].std(0) + 1e-6
    Xn = (X - mean) / std
    torch.manual_seed(seed)
    trunk = torch.nn.Sequential(torch.nn.Linear(X.shape[1], hidden), torch.nn.ReLU())
    head_p, head_c = torch.nn.Linear(hidden, 1), torch.nn.Linear(hidden, 1)
    params = list(trunk.parameters()) + list(head_p.parameters()) + list(head_c.parameters())
    opt = torch.optim.Adam(params, lr=3e-3)
    xt, yt, ct = torch.tensor(Xn[tr]), torch.tensor(y[tr]), torch.tensor(np.log1p(cost[tr]))
    pos_w = torch.tensor((len(yt) - yt.sum()) / max(yt.sum(), 1.0))
    bce = torch.nn.BCEWithLogitsLoss(pos_weight=pos_w)
    for ep in range(epochs):
        perm = torch.randperm(len(xt))
        for i in range(0, len(xt), 4096):
            idx = perm[i:i + 4096]
            h = trunk(xt[idx])
            loss = bce(head_p(h).squeeze(1), yt[idx]) + torch.nn.functional.mse_loss(head_c(h).squeeze(1), ct[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        h = trunk(torch.tensor(Xn[te])); p = torch.sigmoid(head_p(h).squeeze(1)).numpy(); c = torch.exp(head_c(h).squeeze(1)).numpy()
    from sklearn.metrics import roc_auc_score
    yte, cte = y[te], cost[te]
    print(f"OBS train2: {tr.sum():,} lines / held-out {te.sum():,}; AUC {roc_auc_score(yte, p):.3f}; cost head: corr(log) {np.corrcoef(np.log1p(cte), np.log(c))[0, 1]:.3f}, mean cost {cte.mean():.1f}")
    print("OBS   lambda   recall of proven lines   lines stopped   expansions kept")
    for lam_i in [0.0003, 0.001, 0.003, 0.01]:
        keep = p >= lam_i * np.maximum(c, 1.0)
        recall = (keep & (yte == 1)).sum() / max((yte == 1).sum(), 1)
        print(f"OBS   {lam_i:<7}  {recall:.4f}                  {1 - keep.mean():.3f}           {cte[keep].sum() / max(cte.sum(), 1):.3f}")
    out = paths.derived("forcing_policy.json")
    with open(out, "w") as fh:
        json.dump(dict(w1=trunk[0].weight.detach().numpy().tolist(), b1=trunk[0].bias.detach().numpy().tolist(),
                       w2=head_p.weight.detach().numpy()[0].tolist(), b2=float(head_p.bias.detach().numpy()[0]),
                       wc=head_c.weight.detach().numpy()[0].tolist(), bc=float(head_c.bias.detach().numpy()[0]), lam=lam,
                       mean=mean.tolist(), std=std.tolist(), tau=0.1, features=X.shape[1], hidden=hidden, phase=2,
                       trained_on=sorted(set(games) - held), held_out=sorted(held)), fh)
    print(f"OBS   -> {out} (lambda {lam})")


def rl_train(lam: float, epochs: int, lr: float, seed: int) -> None:
    """PHASE 2, REINFORCE with a baseline from the phase-1 weights: reward = correctness - lam * expansions."""
    import numpy as np
    import torch
    d = np.load(paths.derived("forcing_policy_episodes.npz"))
    pol = json.load(open(paths.derived("forcing_policy.json")))
    mean, std = np.array(pol["mean"], dtype=np.float32), np.array(pol["std"], dtype=np.float32)
    # the reward of each episode, keyed (game, episode)
    ev, et, ex = d["e_verdict"], d["e_truth"], d["e_exp"]
    # a proof is always right (a stop cannot certify one); "no gain" is right when the truth agrees, wrong when the truth
    # proved a gain (the costly error), unscored when even the truth ran out; unknown scores 0
    correct = np.where(ev == 1, 1.0, np.where(ev == 0, np.where(et == 0, 1.0, np.where(et == 1, -1.0, 0.0)), 0.0))
    R = correct - lam * ex
    key = {(g, e): i for i, (g, e) in enumerate(zip(d["e_game"].tolist(), d["e_ep"].tolist()))}
    idx = np.array([key.get((g, e), -1) for g, e in zip(d["d_game"].tolist(), d["d_ep"].tolist())])
    ok = idx >= 0
    X = torch.tensor((d["d_X"][ok] - mean) / std); act = torch.tensor(d["d_act"][ok].astype(np.float32))
    adv = torch.tensor((R[idx[ok]] - R.mean()) / (R.std() + 1e-6), dtype=torch.float32)
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], pol["hidden"]), torch.nn.ReLU(), torch.nn.Linear(pol["hidden"], 1))
    with torch.no_grad():
        net[0].weight.copy_(torch.tensor(pol["w1"])); net[0].bias.copy_(torch.tensor(pol["b1"]))
        net[2].weight.copy_(torch.tensor([pol["w2"]])); net[2].bias.copy_(torch.tensor([pol["b2"]]))
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    torch.manual_seed(seed)
    print(f"OBS rl-train: {len(R):,} episodes, mean reward {R.mean():.3f} (correct {correct.mean():.3f}, expansions {ex.mean():.0f}, lambda {lam}); {len(X):,} decisions")
    for ep in range(epochs):
        perm = torch.randperm(len(X)); tot = 0.0
        for i in range(0, len(X), 8192):
            j = perm[i:i + 8192]
            logit = net(X[j]).squeeze(1)
            logp = torch.nn.functional.logsigmoid(logit) * act[j] + torch.nn.functional.logsigmoid(-logit) * (1 - act[j])
            loss = -(logp * adv[j]).mean()
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item() * len(j)
        with torch.no_grad():
            pc = torch.sigmoid(net(X).squeeze(1)).mean().item()
        print(f"OBS   epoch {ep}: loss {tot / len(X):.4f}, mean p(continue) {pc:.3f}")
    out = paths.derived("forcing_policy.json")
    pol.update(w1=net[0].weight.detach().numpy().tolist(), b1=net[0].bias.detach().numpy().tolist(),
               w2=net[2].weight.detach().numpy()[0].tolist(), b2=float(net[2].bias.detach().numpy()[0]), tau=0.5, phase=2, lam=lam)
    with open(out, "w") as fh:
        json.dump(pol, fh)
    print(f"OBS   -> {out} (tau 0.5: the policy's own decision)")


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect"); c.add_argument("--tag", default="c300"); c.add_argument("--games", type=int, default=40)
    c.add_argument("--budget", type=int, default=2000); c.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    t = sub.add_parser("train"); t.add_argument("--hidden", type=int, default=16); t.add_argument("--epochs", type=int, default=30)
    t.add_argument("--recall", type=float, default=0.99); t.add_argument("--seed", type=int, default=0)
    rc = sub.add_parser("rl-collect"); rc.add_argument("--tag", default="c300"); rc.add_argument("--games", type=int, default=24)
    rc.add_argument("--skip", type=int, default=0); rc.add_argument("--workers", type=int, default=6)
    t2 = sub.add_parser("train2"); t2.add_argument("--hidden", type=int, default=16); t2.add_argument("--epochs", type=int, default=8)
    t2.add_argument("--lam", type=float, default=0.001); t2.add_argument("--seed", type=int, default=0)
    rt = sub.add_parser("rl-train"); rt.add_argument("--lam", type=float, default=0.001); rt.add_argument("--epochs", type=int, default=5)
    rt.add_argument("--lr", type=float, default=1e-3); rt.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    if a.cmd == "collect":
        collect(a.tag, a.games, a.budget, a.workers)
    elif a.cmd == "train":
        train(a.hidden, a.epochs, a.recall, a.seed)
    elif a.cmd == "train2":
        train2(a.hidden, a.epochs, a.lam, a.seed)
    elif a.cmd == "rl-collect":
        rl_collect(a.tag, a.games, a.skip, a.workers)
    else:
        rl_train(a.lam, a.epochs, a.lr, a.seed)


if __name__ == "__main__":
    main()
