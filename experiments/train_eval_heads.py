#!/usr/bin/env python
"""
experiments/train_eval_heads.py — train the two eval heads (descriptive WDL
on game results; normative expected-score on lichess server evals) as FROZEN
PROBES over the trained TorchFB's F embedding, then report holdout quality
and the descriptive-vs-normative DIVERGENCE (trap-region candidates).

Verdicts:
  DESC_AUC     AUC of descriptive expected-score separating positions from
               won vs lost games (holdout)
  DESC_ACC3    3-class holdout accuracy (majority baseline printed alongside)
  NORM_SPEAR   spearman(normative head, winprob(lichess eval)) on held-out
               ANNOTATED positions
  BASE_AUC/BASE_SPEAR  the ZERO-LABEL readout F(s)@(zMATE_W - zMATE_B) from
               the checkpoint's stored zgoals, scored on the same holdout --
               if this matches the trained probes, the FB geometry already
               carries the eval and the labels add little
  divergence   top FENs where descriptive and normative disagree most, and
               mean |divergence| per white-Elo bin

--repr ablation (headline is F; B and FB are controls, see nn/eval_head.py):
  F   omega-conditioned forward embedding (the hypothesis under test)
  B   board-only goal embedding -- Elo-blind; if it matches F on DESC_AUC,
      outcome info is static board features, not forward structure
  FB  concat -- if it beats F, F is losing value-relevant info that B keeps
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from catspace.data.encode import board_from_packed
from catspace.io.paths import derived_dir, newest_shard_dir
from catspace.nn.eval_head import (EvalHead, descriptive_loss, normative_loss, save_heads)
from catspace.nn.fb import load_ckpt, pick_device
from catspace.nn.features import elo_bin, feature_planes, omega_ids, winprob_cp

HOLDOUT_MOD = 50


def shard_arrays(shard_dir: Path):
    for path in sorted(shard_dir.glob("shard_*.npz")):
        npz = np.load(path)
        yield {k: npz[k] for k in npz.files}     # bind once


def embed_repr(fb, planes, om, repr_: str) -> torch.Tensor:
    if repr_ == "F":
        return fb.embed_F(planes, om)
    if repr_ == "B":
        return fb.embed_B(planes)
    return torch.cat([fb.embed_F(planes, om), fb.embed_B(planes)], dim=1)


@torch.no_grad()
def embed_all(fb, data, rows, device, repr_="F", batch=2048):
    """Embeddings (per --repr) for the given rows of one shard, batched."""
    outs = []
    for i in range(0, len(rows), batch):
        r = rows[i:i + batch]
        planes = torch.from_numpy(feature_planes(data["packed"][r], data["meta"][r])).to(device)
        om = torch.from_numpy(omega_ids(data["white_elo"][r], data["black_elo"][r],
                                        data["clock"][r])).to(device)
        outs.append(embed_repr(fb, planes, om, repr_).cpu())
    return torch.cat(outs)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shards", default=None)
    ap.add_argument("--ckpt", default=None, help="default: data/derived/lichess_fb.pt")
    ap.add_argument("--out", default=None, help="default: data/derived/eval_heads.pt")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max-rows-per-shard", type=int, default=400_000,
                    help="training-row cap per shard per epoch (probe heads saturate fast)")
    ap.add_argument("--joint", action="store_true",
                    help="ALSO fine-tune F (research knob; default frozen probe)")
    ap.add_argument("--repr", choices=("F", "B", "FB"), default="F",
                    help="which embedding the probes read (B/FB are ablation controls)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    shard_dir = Path(args.shards) if args.shards else newest_shard_dir()
    ckpt_path = Path(args.ckpt) if args.ckpt else derived_dir() / "lichess_fb.pt"
    default_out = "eval_heads.pt" if args.repr == "F" else f"eval_heads_{args.repr}.pt"
    out_path = Path(args.out) if args.out else derived_dir() / default_out
    device = pick_device(args.device)

    fb, payload = load_ckpt(ckpt_path, device)
    fb.eval()
    if args.joint:
        fb.train()
    print(f"shards={shard_dir.name} ckpt={ckpt_path.name} device={device} "
          f"joint={args.joint} repr={args.repr}")

    d_in = fb.d * (2 if args.repr == "FB" else 1)
    desc = EvalHead(d_in, args.hidden, n_out=3, seed=args.seed).to(device)
    norm = EvalHead(d_in, args.hidden, n_out=1, seed=args.seed + 1).to(device)
    params = list(desc.parameters()) + list(norm.parameters())
    if args.joint:
        params += list(fb.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr)

    rng = np.random.default_rng(args.seed)
    for epoch in range(args.epochs):
        for data in shard_arrays(shard_dir):
            train_rows = np.flatnonzero(data["game_id"] % HOLDOUT_MOD != 0)
            rng.shuffle(train_rows)
            train_rows = train_rows[: args.max_rows_per_shard]
            for i in range(0, len(train_rows), args.batch):
                r = train_rows[i:i + args.batch]
                planes = torch.from_numpy(feature_planes(data["packed"][r], data["meta"][r])).to(device)
                om = torch.from_numpy(omega_ids(data["white_elo"][r], data["black_elo"][r],
                                                data["clock"][r])).to(device)
                ctx = torch.enable_grad() if args.joint else torch.no_grad()
                with ctx:
                    f = embed_repr(fb, planes, om, args.repr)
                f = f if args.joint else f.detach()
                loss = descriptive_loss(desc, f, torch.from_numpy(
                    data["result"][r].astype(np.int64)).to(device))
                wp = winprob_cp(data["eval_cp"][r])
                fin = np.isfinite(wp)
                if fin.any():
                    loss = loss + normative_loss(
                        norm, f[np.flatnonzero(fin)],
                        torch.from_numpy(wp[fin].astype(np.float32)).to(device))
                opt.zero_grad(); loss.backward(); opt.step()
        print(f"epoch {epoch} done (last loss {float(loss):.4f})", flush=True)

    # ------------------------------------------------------------- holdout report
    desc.eval(); norm.eval(); fb.eval()
    zdiff = payload.get("zgoals", {}).get("MATE_DIFF")
    if zdiff is not None:
        zdiff = torch.as_tensor(zdiff, dtype=torch.float32).cpu()
    e_desc_all, e_norm_all, res_all, wp_all, welo_all, base_all = [], [], [], [], [], []
    fens, divs = [], []
    for data in shard_arrays(shard_dir):
        held = np.flatnonzero(data["game_id"] % HOLDOUT_MOD == 0)
        if held.size == 0:
            continue
        f = embed_all(fb, data, held, device, args.repr).to(device)
        if zdiff is not None:
            fF = f[:, :fb.d].cpu() if args.repr in ("F", "FB") \
                else embed_all(fb, data, held, device, "F")
            base_all.append((fF @ zdiff).numpy())
        with torch.no_grad():
            e_d = desc.expected_score(f).cpu().numpy()
            e_n = norm.expected_score(f).cpu().numpy()
        e_desc_all.append(e_d); e_norm_all.append(e_n)
        res_all.append(data["result"][held]); welo_all.append(data["white_elo"][held])
        wp = winprob_cp(data["eval_cp"][held]); wp_all.append(wp)
        # collect the most divergent ANNOTATED positions of this shard for the report
        fin = np.flatnonzero(np.isfinite(wp))
        top = fin[np.argsort(-np.abs(e_d[fin] - e_n[fin]))[:10]]
        for j in top:
            fens.append(board_from_packed(data["packed"][held[j]], data["meta"][held[j]]).fen())
            divs.append(float(e_d[j] - e_n[j]))

    e_d = np.concatenate(e_desc_all); e_n = np.concatenate(e_norm_all)
    res = np.concatenate(res_all); wp = np.concatenate(wp_all)
    welo = np.concatenate(welo_all)

    from catspace.util import auc
    from scipy.stats import spearmanr
    a = auc(e_d[res == 1], e_d[res == -1])
    cls_pred = np.select([e_d > 0.55, e_d < 0.45], [1, -1], default=0)
    acc3 = float((cls_pred == res).mean())
    base3 = max((res == v).mean() for v in (-1, 0, 1))
    fin = np.isfinite(wp)
    sp = spearmanr(e_n[fin], wp[fin]).statistic

    print(f"holdout rows: {len(e_d)} ({int(fin.sum())} annotated)")
    print(f"VERDICT DESC_AUC={a:.3f} DESC_ACC3={acc3:.3f} (majority {base3:.3f}) NORM_SPEAR={sp:.3f}")
    if base_all:
        base = np.concatenate(base_all)
        b_auc = auc(base[res == 1], base[res == -1])
        b_sp = spearmanr(base[fin], wp[fin]).statistic
        print(f"VERDICT BASE_AUC={b_auc:.3f} BASE_SPEAR={b_sp:.3f} "
              f"(zero-label F@zMATE_DIFF readout on the same holdout)")

    print("\nmean |E_desc - E_norm| by white-Elo bin (annotated holdout):")
    bins = elo_bin(welo[fin])
    dv = np.abs(e_d[fin] - e_n[fin])
    for b in np.unique(bins):
        lo = 800 + 200 * int(b)
        label = f"{lo}-{lo + 199}" if b < 10 else "unknown"
        print(f"  {label:>9}: {dv[bins == b].mean():.3f}  (n={int((bins == b).sum())})")

    order = np.argsort(-np.abs(np.array(divs)))[:20]
    print("\ntop divergent positions (E_desc - E_norm, +: humans overperform eval):")
    for i in order:
        print(f"  {divs[i]:+.3f}  {fens[i]}")

    save_heads(out_path, desc, norm, d_in,
               meta=dict(ckpt=str(ckpt_path), joint=args.joint, repr=args.repr))
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
