#!/usr/bin/env python
"""
experiments/train_policy_head.py — behavioral-cloning MOVE-PRIOR head on a
FROZEN field (Kaveh 2026-07-19: "policy head it is, but only using the field").

Trains PolicyHead: F(s) -> the move actually played from s (from human games).
Reads ONLY F. Doubles as a field diagnostic -- top-1 move accuracy among legal
moves from F measures how much playable structure the field carries. Saves
<ckpt>_policy.pt for the MCTS to use as its child-prior source (so expansion
costs ~1 eval instead of branching-many).

Usage:
  .venv/bin/python experiments/train_policy_head.py            # incumbent
  .venv/bin/python experiments/train_policy_head.py --ckpt <field>.pt --n 200000
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from catspace.research.tools.chess_specific.chessdata.encode import board_from_packed
from catspace.research.tools.chess_specific.chessdata.shards import LichessPairSource
from catspace.io.paths import newest_shard_dir
from catspace.research.components.encoder.approaches.jepa_tokenizer.src.features import feature_planes, omega_ids
from catspace.research.components.encoder.approaches.jepa_tokenizer.src.fb import load_ckpt
from catspace.research.components.encoder.approaches.jepa_tokenizer.src.policy_head import PolicyHead, move_index, policy_loss

HOLDOUT_MOD = 50


def derive_move_idx(anchor_pk, anchor_mt, succ_pk, succ_mt):
    """The from-to index of the move anchor -> successor (match on board_fen)."""
    ba = board_from_packed(anchor_pk, anchor_mt)
    succ_fen = board_from_packed(succ_pk, succ_mt).board_fen()
    legal_idx = []
    hit = -1
    for m in ba.legal_moves:
        legal_idx.append(move_index(m))
        b2 = ba.copy(stack=False)
        b2.push(m)
        if b2.board_fen() == succ_fen:
            hit = move_index(m)
    return hit, legal_idx


def build(fb, src, device, n_target, holdout, seed, want_legal=False):
    Fs, ys, legal = [], [], []
    got = 0
    for batch in src.batches(1024, seed):
        gid = batch.meta["game_id"]
        keep = (~batch.meta["succ_is_last"]) & ((gid % HOLDOUT_MOD == 0) == holdout)
        idx = np.flatnonzero(keep)
        if len(idx) == 0:
            continue
        anc, amt = batch.anchors, batch.meta["board_meta"]
        spk, smt = batch.meta["packed_succ"], batch.meta["board_meta_succ"]
        rows, mvs, legs = [], [], []
        for i in idx:
            hit, li = derive_move_idx(anc[i], amt[i], spk[i], smt[i])
            if hit >= 0:
                rows.append(i); mvs.append(hit); legs.append(li)
        if not rows:
            continue
        rows = np.array(rows)
        planes = feature_planes(anc[rows], amt[rows])
        om = omega_ids(batch.meta["white_elo"][rows], batch.meta["black_elo"][rows],
                       batch.meta["clock"][rows])
        with torch.no_grad():
            f = fb.embed_F(torch.from_numpy(planes).to(device),
                           torch.from_numpy(om).to(device)).cpu().numpy()
        Fs.append(f); ys.append(np.array(mvs))
        if want_legal:
            legal.extend(legs)
        got += len(rows)
        if got >= n_target:
            break
    F = np.concatenate(Fs)[:n_target]
    y = np.concatenate(ys)[:n_target]
    return (F, y, legal[:n_target]) if want_legal else (F, y)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default="data/derived/sep/cert_base_full.pt")
    ap.add_argument("--n", type=int, default=200_000, help="train positions")
    ap.add_argument("--n-val", type=int, default=5000)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--shards", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = "cpu"
    shard_dir = Path(args.shards) if args.shards else newest_shard_dir()
    fb, pay = load_ckpt(Path(args.ckpt), device)
    fb.eval()
    src = LichessPairSource(shard_dir, gamma=0.98)

    t0 = time.time()
    Ftr, ytr = build(fb, src, device, args.n, holdout=False, seed=args.seed)
    Fva, yva, legal_va = build(fb, src, device, args.n_val, holdout=True,
                               seed=args.seed + 1, want_legal=True)
    print(f"[stage] dataset F={Ftr.shape} val={Fva.shape}: {time.time() - t0:.1f}s")
    n_legal = np.mean([len(l) for l in legal_va])
    print(f"  mean legal moves/pos (val) = {n_legal:.1f}  (random-among-legal top1 ~ {1/n_legal:.3f})")

    head = PolicyHead(d_in=fb.d, hidden=args.hidden, seed=args.seed).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=args.lr)
    Xtr = torch.from_numpy(Ftr).to(device); Ytr = torch.from_numpy(ytr).long().to(device)
    Xva = torch.from_numpy(Fva).to(device)
    rng = np.random.default_rng(args.seed)
    t0 = time.time()
    for ep in range(args.epochs):
        head.train()
        perm = rng.permutation(len(Xtr))
        losses = []
        for lo in range(0, len(perm), args.batch):
            b = perm[lo:lo + args.batch]
            loss = policy_loss(head, Xtr[b], Ytr[b])
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        # val top-1 among LEGAL moves (the meaningful metric)
        head.eval()
        with torch.no_grad():
            logits = head(Xva).cpu().numpy()
        legal_hits = 0
        for i, li in enumerate(legal_va):
            pred = li[int(np.argmax(logits[i, li]))]     # best LEGAL slot
            legal_hits += (pred == int(yva[i]))
        unmasked_top1 = float(np.mean(np.argmax(logits, axis=1) == yva))
        print(f"  epoch {ep+1}/{args.epochs} loss {np.mean(losses):.3f}  "
              f"val top1(legal)={legal_hits/len(yva):.3f}  top1(unmasked)={unmasked_top1:.3f}",
              flush=True)

    out = Path(args.ckpt).with_name(Path(args.ckpt).stem + "_policy.pt")
    torch.save({"state": head.state_dict(), "d_in": fb.d, "hidden": args.hidden}, out)
    print(f"saved {out}")
    print(f"VERDICT POLICY_HEAD top1_legal={legal_hits/len(yva):.3f} "
          f"(chance ~{1/n_legal:.3f}) n_train={len(Xtr)} ckpt={Path(args.ckpt).name}")


if __name__ == "__main__":
    main()
