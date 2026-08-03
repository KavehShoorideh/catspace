#!/usr/bin/env python
"""experiments/bench_value_speed.py -- SPEED x FIDELITY bench for field variants (Kaveh:
'faster but not weaker'). For each candidate ckpt (and optional fp16 mode): measures
embed_F + d_to_bank throughput (evals/s) on a fixed position suite against a fixed bank
sample, plus distance-spearman vs the fp32 teacher on the same pairs. Prints one VERDICT
line per variant -- the speed side of the accept rule (>=1.5x AND not weaker in A/B).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from catspace.fields import FieldModel
from experiments.distill_field import load_positions


def bench(fm, pk, mt, bank_embs, reps=3):
    # warm-up
    fm.embed_F(pk[:32], mt[:32])
    best = 0.0
    for _ in range(reps):
        t0 = time.perf_counter()
        F = fm.embed_F(pk, mt)
        _ = fm.d_to_bank(F, bank_embs)
        dt = time.perf_counter() - t0
        best = max(best, len(pk) / dt)
    return best, F


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--teacher", default="data/derived/sep/self_field_r0.pt")
    ap.add_argument("--candidates", default="",
                    help="comma list of extra ckpts (students); teacher fp32+fp16 always run")
    ap.add_argument("--shards", default="data/shards/lichess_db_standard_rated_2019-01.full")
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--bank", type=int, default=4096, help="bank rows sampled for d_to_bank")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    pk, mt = load_positions(args.shards, args.n + args.bank, args.seed + 7)
    qpk, qmt = pk[:args.n], mt[:args.n]
    bpk, bmt = pk[args.n:], mt[args.n:]

    teacher = FieldModel(args.teacher, device=args.device)
    bank_T = teacher.embed_B(bpk, bmt)
    sp_T, F_T = bench(teacher, qpk, qmt, bank_T)
    D_ref = None
    with torch.no_grad():
        ft = torch.from_numpy(F_T[:256]).to(args.device)
        bt = torch.from_numpy(bank_T[:1024]).to(args.device)
        D_ref = teacher.fb.distance_matrix(ft, bt).cpu().numpy().ravel()
    print(f"VERDICT SPEED teacher-fp32: {sp_T:.0f} evals/s (baseline, fidelity 1.0)",
          flush=True)

    from scipy.stats import spearmanr

    def fidelity(fm, bank):
        with torch.no_grad():
            F = fm.embed_F(qpk[:256], qmt[:256])
            f = torch.from_numpy(np.asarray(F)).to(args.device)
            b = torch.from_numpy(np.asarray(bank[:1024])).to(args.device)
            if next(fm.fb.parameters()).dtype == torch.float16:
                f, b = f.half(), b.half()
            D = fm.fb.distance_matrix(f, b).float().cpu().numpy().ravel()
        return spearmanr(D_ref, D).correlation

    # fp16 teacher
    try:
        t16 = FieldModel(args.teacher, device=args.device)
        t16.fb.half()
        old_embed = t16._planes

        def _planes16(pkx, mtx, _old=old_embed):
            return _old(pkx, mtx).half()
        t16._planes = _planes16
        bank16 = t16.embed_B(bpk, bmt)
        sp16, _ = bench(t16, qpk, qmt, bank16)
        fid16 = fidelity(t16, bank16)
        print(f"VERDICT SPEED teacher-fp16: {sp16:.0f} evals/s ({sp16/sp_T:.2f}x) "
              f"fidelity-spearman {fid16:+.4f}", flush=True)
    except Exception as e:                                  # noqa: BLE001
        print(f"VERDICT SPEED teacher-fp16: FAILED ({e})", flush=True)

    for c in [x for x in args.candidates.split(",") if x]:
        try:
            st = FieldModel(c, device=args.device)
            bank_S = st.embed_B(bpk, bmt)
            sp_S, _ = bench(st, qpk, qmt, bank_S)
            fid = fidelity(st, bank_S)
            print(f"VERDICT SPEED {Path(c).stem}: {sp_S:.0f} evals/s ({sp_S/sp_T:.2f}x) "
                  f"fidelity-spearman {fid:+.4f}", flush=True)
        except Exception as e:                              # noqa: BLE001
            print(f"VERDICT SPEED {Path(c).stem}: FAILED ({e})", flush=True)


if __name__ == "__main__":
    main()
