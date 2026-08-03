#!/usr/bin/env python
"""catspace/research/components/encoder/approaches/reachability_field/experiments/distill_field.py -- QUASIMETRIC DISTILLATION (Kaveh 2026-07-24: 'distill
some weights into a student model... don't make the model weaker, but faster').

Distills the DISTANCES, not the embeddings: student is a smaller FB/IQE net trained to
reproduce the teacher's full pairwise distance matrix on sampled position batches
(directed, B x B supervision per step). Distance-matching lets the student pick its own
embedding width. Fidelity metric = held-out spearman(student d, teacher d); the play
referendum is the anytime-valid A/B harness. TRAINING_STANDARDS: MLflow, step-suffixed
ckpt ladder, no overwrites, git commit tagged.
"""
from __future__ import annotations

import argparse
import glob
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as tF


from catspace.fields import FieldModel
from catspace.research.components.encoder.approaches.jepa_tokenizer.src.fb import TorchFB as FB, pick_device
from catspace.research.components.encoder.approaches.jepa_tokenizer.src.features import feature_planes
from catspace.research.tools.stats_eval.tracking import track_run
from catspace.io import paths


def load_positions(shards, n, seed):
    rng = np.random.default_rng(seed)
    files = sorted(glob.glob(str(Path(shards) / "shard_*.npz")))
    rng.shuffle(files)
    pk, mt = [], []
    for f in files:
        z = np.load(f)
        take = min(n - len(pk), len(z["packed"]))
        idx = rng.permutation(len(z["packed"]))[:take]
        pk.append(z["packed"][idx]); mt.append(z["meta"][idx])
        if sum(len(x) for x in pk) >= n:
            break
    return np.concatenate(pk)[:n], np.concatenate(mt)[:n]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--teacher", default=paths.sep("self_field_r0.pt"))
    ap.add_argument("--shards", default=paths.shards("lichess_db_standard_rated_2019-01.full"))
    ap.add_argument("--n-pos", type=int, default=200_000)
    ap.add_argument("--steps", type=int, default=20_000)
    ap.add_argument("--batch", type=int, default=192, help="boards per side per step")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--blocks", type=int, default=5)
    ap.add_argument("--iqe-components", type=int, default=16)
    ap.add_argument("--out", default=paths.sep("field_student_v1.pt"))
    ap.add_argument("--ckpt-every", type=int, default=5000)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    from contextlib import ExitStack
    stack = ExitStack()
    trk = stack.enter_context(track_run("field_distill", args, run_name=Path(args.out).stem))
    t0 = time.time(); dev = pick_device(args.device)
    torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed)

    teacher = FieldModel(args.teacher, device=str(dev))     # frozen, its own chunking
    tcfg = teacher.fb.config
    scfg = dict(tcfg)
    scfg.update(d=args.d, channels=args.channels, blocks=args.blocks,
                iqe_components=args.iqe_components, enc_out=args.d, dh=args.d,
                seed=args.seed)
    student = FB(**{k: v for k, v in scfg.items()
                    if k in FB.__init__.__code__.co_varnames}).to(dev)
    n_t = sum(p.numel() for p in teacher.fb.parameters())
    n_s = sum(p.numel() for p in student.parameters())
    print(f"[distill] teacher {n_t/1e6:.1f}M -> student {n_s/1e6:.1f}M "
          f"({n_t/max(n_s,1):.1f}x smaller)", flush=True)

    pk, mt = load_positions(args.shards, args.n_pos, args.seed)
    n_hold = 2048
    hold_pk, hold_mt = pk[:n_hold], mt[:n_hold]
    pk, mt = pk[n_hold:], mt[n_hold:]
    print(f"[data] {len(pk)} train positions + {n_hold} held-out", flush=True)

    opt = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
    omega = None

    def planes(P, M):
        return torch.from_numpy(feature_planes(P, M)).to(dev)

    def teacher_D(P1, M1, P2, M2):
        with torch.no_grad():
            f = teacher.fb.embed_F(planes(P1, M1), omega)
            b = teacher.fb.embed_B(planes(P2, M2))
            return teacher.fb.distance_matrix(f, b)

    for s in range(args.steps):
        ia = rng.integers(0, len(pk), args.batch)
        ib = rng.integers(0, len(pk), args.batch)
        Dt = teacher_D(pk[ia], mt[ia], pk[ib], mt[ib])
        f_s = student.embed_F(planes(pk[ia], mt[ia]), omega)
        b_s = student.embed_B(planes(pk[ib], mt[ib]))
        Ds = student.distance_matrix(f_s, b_s)
        loss = tF.huber_loss(Ds, Dt, delta=5.0)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if s % 200 == 0:
            print(f"  step {s} loss {float(loss.detach()):.4f} "
                  f"[{time.time()-t0:.0f}s]", flush=True)
            trk.metrics(dict(loss=float(loss.detach())), step=s)
        if args.ckpt_every and s > 0 and s % args.ckpt_every == 0:
            op = Path(args.out)
            torch.save({"state_dict": student.state_dict(), "config": student.config,
                        "args": vars(args), "step": s},
                       op.with_name(f"{op.stem}_step{s}{op.suffix}"))

    # ---- held-out distance fidelity
    student.eval()
    with torch.no_grad():
        k = 512
        Dt = teacher_D(hold_pk[:k], hold_mt[:k], hold_pk[k:2*k], hold_mt[k:2*k])
        f_s = student.embed_F(planes(hold_pk[:k], hold_mt[:k]), omega)
        b_s = student.embed_B(planes(hold_pk[k:2*k], hold_mt[k:2*k]))
        Ds = student.distance_matrix(f_s, b_s)
    from scipy.stats import spearmanr
    sp = spearmanr(Dt.cpu().numpy().ravel(), Ds.cpu().numpy().ravel()).correlation
    mae = float((Ds - Dt).abs().mean())
    print(f"VERDICT DISTILL held-out spearman {sp:+.4f} MAE {mae:.3f} "
          f"(n={k*k} pairs) student {n_s/1e6:.1f}M vs teacher {n_t/1e6:.1f}M "
          f"[{time.time()-t0:.0f}s]", flush=True)
    trk.metrics(dict(holdout_spearman=float(sp), holdout_mae=mae))
    torch.save({"state_dict": student.state_dict(), "config": student.config,
                "args": vars(args), "step": args.steps}, args.out)
    print(f"saved {args.out}", flush=True)
    stack.close()


if __name__ == "__main__":
    main()
