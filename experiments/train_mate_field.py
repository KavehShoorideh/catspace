#!/usr/bin/env python
"""experiments/train_mate_field.py -- S2 (METASTABILITY_PLAN): a field that actually MATES.

Fixes on the S1 quasimetric field (mate-rate 5%, move-selection 52.7% = coin flip):
  * COLLAPSED mate attractor: a single LEARNABLE MATE goal vector; d(s)=IQE(phi(s), MATE).
    All mates -> one point, so d->0 sharply at mate (Defect 1: scattered mates fixed).
  * WDL basins / INFINITE barriers: won -> regress d to log1p(DTM); draw/loss -> hinge d UP to
    a large margin M (relu(log1p(M)-d)); a bounded repeller, not unbounded (Defect 2 + stalemate).
  * BOTH colors + broad random positions (not just optimal lines) so off-optimal children are
    in-distribution (the move-selection fix).
Single-space IQE (composable). Low dim (d=32) until eff-rank saturates. Tensor-batched.
Gate: MATE-RATE vs tablebase-optimal defense (was 5%) + move-selection accuracy.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import chess
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from catspace.research.tools.chess_specific.chessdata.encode import encode_meta, encode_packed
from catspace.research.components.encoder.approaches.jepa_tokenizer.src.fb import pick_device
from catspace.research.components.encoder.approaches.jepa_tokenizer.src.iqe import IQE
from catspace.research.components.planner.approaches.endgame_groundtruth.src.tb import TB, DEFAULT_SYZYGY, rollout_dtm, tb_best_move
from experiments.arch_bakeoff import CNNBackbone, eff_rank, tokens
from experiments.gen_dtm_data import random_class_start
from experiments.value_fixed_point import white_pov_value


class MateField(nn.Module):
    """phi(s) = head(CNN(s)); d(s) = IQE(phi(s), MATE) with a learnable MATE goal vector."""
    def __init__(self, d=32, d_bb=64, blocks=6, iqe_components=16):
        super().__init__()
        self.enc = CNNBackbone(d_bb, blocks)
        self.head = nn.Sequential(nn.Linear(d_bb, d_bb), nn.GELU(), nn.Linear(d_bb, d))
        self.iqe = IQE(d, components=iqe_components)
        self.mate = nn.Parameter(torch.randn(d) * 0.1)

    def phi(self, ids, stm):
        _, pooled = self.enc(ids, stm)
        return self.head(pooled)

    def d_to_mate(self, ids, stm):
        e = self.phi(ids, stm)
        return self.iqe(e, self.mate.expand_as(e))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/derived/wdl_dtm_v1.npz")
    ap.add_argument("--d", type=int, default=32)
    ap.add_argument("--d-bb", type=int, default=64)
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--iqe-components", type=int, default=16)
    ap.add_argument("--margin", type=float, default=400.0, help="M: hinge draw/loss d up to log1p(M)")
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eval-n", type=int, default=200)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--save", default="artifacts/experiments/mate_field_v1.pt")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time(); dev = pick_device(args.device)
    torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed)

    z = np.load(args.data)
    ids, stm = tokens(z["packed"], z["meta"])
    dtm = z["dtm"].astype(np.float32)                          # >=1 won, 0 mate, -1 INF
    ids_t = torch.from_numpy(ids.astype(np.int64)); stm_t = torch.from_numpy(stm.astype(np.int64))
    won = dtm >= 0                                              # finite target (won or mate)
    inf = dtm < 0                                               # draw/loss -> hinge up
    tgt = np.where(won, np.log1p(np.clip(dtm, 0, None)), 0.0).astype(np.float32)
    tgt_t = torch.from_numpy(tgt).to(dev)
    won_t = torch.from_numpy(won.astype(np.float32)).to(dev)
    logM = float(np.log1p(args.margin))
    idx_won = np.flatnonzero(won); idx_inf = np.flatnonzero(inf)
    print(f"[mate-field] {len(dtm)} rows: won/mate {len(idx_won)} | INF {len(idx_inf)} | "
          f"logM {logM:.2f} d{args.d}c{args.iqe_components}", flush=True)

    net = MateField(args.d, args.d_bb, args.blocks, args.iqe_components).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)

    for s in range(args.steps):
        # balanced batch: half won (regress DTM), half INF (hinge up)
        bw = idx_won[rng.integers(0, len(idx_won), args.batch // 2)]
        bi = idx_inf[rng.integers(0, len(idx_inf), args.batch // 2)] if len(idx_inf) else bw
        b = np.concatenate([bw, bi])
        d = net.d_to_mate(ids_t[b].to(dev), stm_t[b].to(dev))
        dlog = torch.log1p(d.clamp(min=0))
        w = won_t[b]
        reg = (F.huber_loss(dlog, tgt_t[b], reduction="none") * w).sum() / w.sum().clamp(min=1)
        hinge = (F.relu(logM - dlog) * (1 - w)).sum() / (1 - w).sum().clamp(min=1)   # push INF up to M
        loss = reg + hinge
        opt.zero_grad(); loss.backward(); opt.step()
        if s % 2000 == 0:
            print(f"  step {s} loss {float(loss):.4f} (reg {float(reg):.3f} hinge {float(hinge):.3f}) "
                  f"[{time.time()-t0:.0f}s]", flush=True)

    net.eval()
    # --- rank/eff-rank diagnostics on won slice ---
    with torch.no_grad():
        sub = idx_won[rng.integers(0, len(idx_won), min(3000, len(idx_won)))]
        e = net.phi(ids_t[sub].to(dev), stm_t[sub].to(dev)).cpu().numpy()
        er = eff_rank(e)
        dd = net.d_to_mate(ids_t[sub].to(dev), stm_t[sub].to(dev)).cpu().numpy()
    from scipy.stats import spearmanr
    sp = float(spearmanr(dd, dtm[sub]).correlation)
    # INF separation: is draw/loss d clearly above won d?
    with torch.no_grad():
        si = idx_inf[rng.integers(0, len(idx_inf), min(3000, len(idx_inf)))]
        di = net.d_to_mate(ids_t[si].to(dev), stm_t[si].to(dev)).cpu().numpy()
    print(f"  eff_rank(phi) {er:.1f}/{args.d} | d-vs-DTM spearman {sp:+.3f} | "
          f"won-d med {np.median(dd):.1f} vs INF-d med {np.median(di):.1f}", flush=True)

    # --- MATE-RATE vs tablebase-optimal defense ---
    tb = TB(str(DEFAULT_SYZYGY), cache_db=None)

    @torch.no_grad()
    def d_children(boards):
        pk = np.stack([encode_packed(b) for b in boards]); mt = np.stack([encode_meta(b) for b in boards])
        i, s2 = tokens(pk, mt)
        return net.d_to_mate(torch.from_numpy(i.astype(np.int64)).to(dev),
                             torch.from_numpy(s2.astype(np.int64)).to(dev)).cpu().numpy()

    classes = ["KQvK", "KRvK", "KRRvK", "KBNvK", "KBBvK"]
    mated = tot = movesel_ok = movesel_n = 0
    for cls in classes:
        got = 0
        while got < args.eval_n // len(classes):
            b0 = random_class_start(rng, cls)
            if b0 is None or b0.turn != chess.WHITE or white_pov_value(b0, tb) != 1.0:
                continue
            opt_dtm = rollout_dtm(b0, tb)
            if not opt_dtm or opt_dtm < 1:
                continue
            got += 1; tot += 1
            b = b0.copy(stack=False); cap = 3 * opt_dtm + 6; ok = False
            while len(b.move_stack) < cap:
                if b.is_checkmate(): ok = True; break
                if b.is_game_over(claim_draw=True): break
                if b.turn == chess.WHITE:
                    moves = list(b.legal_moves); kids = []
                    for m in moves: b.push(m); kids.append(b.copy(stack=False)); b.pop()
                    dc = d_children(kids)
                    chosen = moves[int(np.argmin(dc))]
                    # cheap move-selection metric: did the field KEEP the win? (avoid the INF blunder)
                    b.push(chosen)
                    kept = b.is_checkmate() or (not b.is_game_over(claim_draw=True)
                                                and white_pov_value(b, tb) == 1.0)
                    movesel_n += 1; movesel_ok += int(kept)
                    # already pushed `chosen`
                else:
                    m = tb_best_move(b, tb, set())
                    if m is None: break
                    b.push(m)
            if ok: mated += 1
    tb.close()
    mr = 100 * mated / max(1, tot)
    ms = 100 * movesel_ok / max(1, movesel_n)
    print(f"VERDICT MATE-FIELD d{args.d}: MATE-RATE {mr:.1f}% ({mated}/{tot}) | "
          f"kept-win {ms:.1f}% (blunder-avoidance) | d-vs-DTM {sp:+.3f} | "
          f"eff_rank {er:.1f} | [{time.time()-t0:.0f}s]", flush=True)

    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": net.state_dict(),
                    "cfg": {"d": args.d, "d_bb": args.d_bb, "blocks": args.blocks,
                            "iqe_components": args.iqe_components},
                    "metrics": {"mate_rate": mr, "move_select": ms, "d_vs_dtm": sp, "eff_rank": er}},
                   args.save)
        print(f"  saved -> {args.save}", flush=True)


if __name__ == "__main__":
    main()
