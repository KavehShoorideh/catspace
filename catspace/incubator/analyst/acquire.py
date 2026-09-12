"""acquire.py -- error-driven data acquisition against a random control (Kaveh 2026-09-11: "I want the data random,
not curated by chess concepts; let the error tell us what data to generate").

Corpus: the game positions (69k, Kaveh's games) plus ChessBench (62.5k positions drawn at random from Lichess
games, every legal move labeled with the teacher's win probability). ChessBench's training rows are split in
two: the BASE (labels used) and the POOL (labels withheld). Rounds:
  round0   train on games + base
  active   score the pool by the model's own uncertainty (the policy's entropy over legal moves), reveal the labels
           of the top n, retrain
  random   reveal the labels of n random pool positions instead, retrain
  full     every label revealed (the product's model)
Each model is measured on the ChessBench test positions (top-1 against the teacher's best; REGRET = the best
move's win probability minus the chosen move's), on the game test positions (quiet-move agreement with the
engine) and, separately, on the Strategic Test Suite. The claim tested: active beats random at equal labels.

    python -m catspace.incubator.analyst.acquire --n 6000 --epochs 8
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import knowledge_net as K

GAMES = "data/derived/knowledge"; CB = "data/derived/chessbench_kn"
OUT = Path("data/derived/checkpoints")


def cb_train_ids(data: K.Data) -> np.ndarray:
    return np.array([i for i in data.ids("train") if data.meta[int(i)].get("site") == "chessbench"])


ARCH = dict(d=128, layers=6, heads=8)


OUT_TAG = ""


def train_round(name: str, hide, epochs: int, log) -> tuple[K.KnowledgeNet, K.Data]:
    name = name + OUT_TAG
    out = OUT / f"kn_{name}"
    args = argparse.Namespace(data=f"{GAMES},{CB}", out=str(out), epochs=epochs, **{k: v for k, v in K.Config().__dict__.items() if k != "epochs"})
    args.variant = "full"; args.w_traj = 0.05
    for k_, v in ARCH.items():
        setattr(args, k_, v)
    # train() builds its own Data; pass the hidden set through a module-level hook
    K.HIDE_LABELS = set(int(i) for i in hide) if hide is not None else set()
    t0 = time.perf_counter()
    model, data = K.train(args)
    log(f"[acquire] {name}: trained in {time.perf_counter() - t0:.0f}s with {len(K.HIDE_LABELS)} labels hidden", flush=True)
    return model, data


@torch.no_grad()
def uncertainty(model: K.KnowledgeNet, data: K.Data, ids, device, batch: int = 256) -> np.ndarray:
    """The policy's entropy over legal moves per position (nats)."""
    model.eval(); out = []
    for s in range(0, len(ids), batch):
        chunk = ids[s:s + batch]; b = data.batch(chunk, device)
        pl = model.policy_logits(model(b)["h"], b).view(len(chunk), -1)
        lp = F.log_softmax(pl, dim=-1); p = lp.exp()
        out.append((-(p * lp).masked_fill(~torch.isfinite(lp), 0.0).sum(-1)).cpu().numpy())
    model.train()
    return np.concatenate(out)


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=6000); ap.add_argument("--epochs", type=int, default=8); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rounds", default="round0,active,random,full"); ap.add_argument("--d", type=int, default=128); ap.add_argument("--layers", type=int, default=6); ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--tag", default="")
    a = ap.parse_args(argv)
    ARCH.update(d=a.d, layers=a.layers, heads=a.heads)
    global OUT_TAG; OUT_TAG = a.tag
    log = lambda *x, **k: print(*x, **k)
    rng = random.Random(a.seed)
    probe = K.Data(f"{GAMES},{CB}", meta_only=True)
    cb_tr = cb_train_ids(probe); cb_tr = np.array(sorted(int(i) for i in cb_tr)); rng.shuffle(cb_tr)
    base, pool = cb_tr[: len(cb_tr) // 2], cb_tr[len(cb_tr) // 2:]
    log(f"[acquire] chessbench training rows {len(cb_tr)}: base {len(base)}, pool {len(pool)}; games rows {len(probe.ids('train')) - len(cb_tr)}", flush=True)
    results = {}
    rounds = a.rounds.split(",")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    picks = {}
    if "round0" in rounds or "active" in rounds:
        m0, d0 = train_round("round0", pool, a.epochs, log)
        results["round0"] = json.load(open(OUT / f"kn_round0{OUT_TAG}" / "final.json"))["test"]
        ent = uncertainty(m0, d0, pool, device)
        order = np.argsort(-ent)
        picks["active"] = pool[order[: a.n]]; picks["random"] = np.array(rng.sample(list(pool), a.n))
        log(f"[acquire] pool entropy: mean {ent.mean():.3f}, the {a.n} most uncertain {ent[order[:a.n]].mean():.3f}, random {ent[[np.where(pool == i)[0][0] for i in picks['random'][:2000]]].mean():.3f}", flush=True)
        json.dump({k: [int(i) for i in v] for k, v in picks.items()}, open(OUT / f"acquire_picks{OUT_TAG}.json", "w"))
        del m0, d0
    for name in ("active", "random"):
        if name in rounds:
            reveal = set(int(i) for i in picks[name]); hide = [i for i in pool if int(i) not in reveal]
            train_round(name, hide, a.epochs, log)
            results[name] = json.load(open(OUT / f"kn_{name}{OUT_TAG}" / "final.json"))["test"]
    if "full" in rounds:
        train_round("full_cb", None, a.epochs, log)
        results["full"] = json.load(open(OUT / f"kn_full_cb{OUT_TAG}" / "final.json"))["test"]
    keys = ["cb_policy_top1", "cb_policy_regret", "cb_qhead_top1", "cb_qhead_regret", "cb_q_mae", "policy_top1", "policy_quiet", "cp_r2", "labeled_n"]
    log("[acquire] RESULTS (ChessBench test = the teacher's labels; games test = the engine's best at depth 12):")
    log("   " + f"{'round':<8}" + "".join(f"{k:>18}" for k in keys))
    for name, ev in results.items():
        log("   " + f"{name:<8}" + "".join(f"{ev.get(k, float('nan')):>18.4f}" if isinstance(ev.get(k), float) else f"{ev.get(k, ''):>18}" for k in keys))
    json.dump(results, open(OUT / f"acquire_results{OUT_TAG}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
