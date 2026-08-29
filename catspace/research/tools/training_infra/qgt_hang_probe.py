"""qgt_hang_probe.py -- does the roll SEE the hang? And what does T2 actually
approximate at one ply? (Kaveh 2026-08-28: quick probes pre-approved; "how is our
one ply behaving? because we train on the two ply.")

Two questions, one instrument, on corpus positions with multiPV labels:

  A. HANG DETECTION: construct each position's best and worst labelled move (label
     gap > 0.3 = a real blunder). Roll each through T2, read the value. Does the
     model's read DROP for the blunder the way the labels say it should? The reply
     (the capture) is INSIDE the T2 step -- this is the marginal-then-binarize
     mechanism on trial.
  B. ONE-PLY ATTRIBUTION: T2's output is supposed to be the position two plies out.
     Compare its bits against (i) the encoded board after MY move only (1 ply) and
     (ii) the encoded board after my move + a sampled reply (2 plies), with the
     root-copy baseline for scale. Tells us whether T2 learned my-move effects,
     reply expectations, both, or neither.

    .venv/bin/python -m catspace.research.tools.training_infra.qgt_hang_probe \
        --ckpt artifacts/experiments/qgt7_step8000.pt
"""
from __future__ import annotations

import argparse
import glob

import chess
import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="artifacts/experiments/qgt7_step8000.pt")
    ap.add_argument("--n", type=int, default=150)
    args = ap.parse_args()
    from catspace.research.tools.training_infra.qgt_search import QGTDynamics
    from catspace.research.tools.npz_io import load_npz
    dyn = QGTDynamics(args.ckpt)
    rng = np.random.default_rng(0)
    es = lambda pv, white: (0.5 * (1.0 + (1.0 if white else -1.0)
                                   * (pv[:, 0] - pv[:, 2])))
    rows = []
    for f in sorted(glob.glob("data/derived/gko1_shards/shard_*.npz"))[:10]:
        z = load_npz(f)
        if "wdl" not in z.files:
            continue
        tg = z["tgt"]
        for i in np.flatnonzero(z["kind"] == 0):
            fen = str(z["fen"][i])
            wp = tg[i]
            fin = np.isfinite(wp)
            if fin.sum() < 2:
                continue
            rows.append((fen, wp))
            if len(rows) >= args.n:
                break
        if len(rows) >= args.n:
            break

    gaps_l, gaps_m, corr_p, corr_t = [], [], [], []
    hang_l, hang_m = [], []
    a1 = a2 = a0_1 = a0_2 = an = 0.0
    with torch.no_grad():
        for fen, wp in rows:
            try:
                b = chess.Board(fen)
            except Exception:
                continue
            legal = list(b.legal_moves)[:48]
            v = np.flatnonzero(np.isfinite(wp[:len(legal)]))
            if len(v) < 2:
                continue
            state, ctx = dyn.encode(b)
            moves, codes, frm, to = dyn.root_moves(b, ctx)
            j_b = int(v[wp[v].argmax()]); j_w = int(v[wp[v].argmin()])
            sel = np.unique(np.concatenate([[j_b, j_w],
                                            rng.choice(v, min(6, len(v)),
                                                       replace=False)]))
            sel = sel[sel < len(moves)]
            st = state.expand(len(sel), -1, -1)
            nxt = dyn.step(st, codes[sel], frm[sel], to[sel])
            pv = dyn.value(nxt, 1 if b.turn else 0).cpu().numpy()
            r_es = es(pv, b.turn)
            lab = wp[sel]
            if len(sel) >= 4 and lab.std() > 1e-6 and r_es.std() > 1e-6:
                corr_p.append(np.corrcoef(r_es, lab)[0, 1])
            gap_l = float(wp[j_b] - wp[j_w])
            ib = int(np.where(sel == j_b)[0][0]); iw = int(np.where(sel == j_w)[0][0])
            gap_m = float(r_es[ib] - r_es[iw])
            gaps_l.append(gap_l); gaps_m.append(gap_m)
            if gap_l > 0.3:                       # a real blunder exists here
                hang_l.append(gap_l); hang_m.append(gap_m)
            # B: one-ply attribution on the WORST move (the interesting one)
            mv = moves[j_w]
            b1 = b.copy(); b1.push(mv)
            rep = list(b1.legal_moves)
            if rep:
                b2 = b1.copy(); b2.push(rep[int(rng.integers(len(rep)))])
                s1, _ = dyn.encode(b1)
                s2, _ = dyn.encode(b2)
                t2 = dyn.step(state, codes[j_w:j_w + 1], frm[j_w:j_w + 1],
                              to[j_w:j_w + 1])
                a1 += float((t2 == s1).float().mean())
                a2 += float((t2 == s2).float().mean())
                a0_1 += float((state == s1).float().mean())
                a0_2 += float((state == s2).float().mean())
                an += 1

    gl, gm = np.array(gaps_l), np.array(gaps_m)
    hl, hm = np.array(hang_l), np.array(hang_m)
    print(f"[hang] {len(gl)} positions | best-vs-worst: label gap mean "
          f"{gl.mean():.3f}, MODEL rolled-read gap mean {gm.mean():.3f}, "
          f"corr {np.corrcoef(gm, gl)[0, 1]:.3f}", flush=True)
    print(f"[hang] within-position corr(read, label) over sampled children: "
          f"mean {np.mean(corr_p):.3f} (n={len(corr_p)})", flush=True)
    if len(hl):
        det = float((hm > 0.1).mean())
        print(f"[hang] BLUNDERS (label gap>0.3, n={len(hl)}): label {hl.mean():.3f} "
              f"vs model {hm.mean():.3f} | detection rate (model gap>0.1): "
              f"{det:.1%}", flush=True)
    print(f"[oneply] T2 output agrees with: 1-ply child {a1/an:.3f} | 2-ply child "
          f"{a2/an:.3f} | (root-copy baselines: {a0_1/an:.3f} / {a0_2/an:.3f}; "
          f"n={int(an)})", flush=True)
    print("[hang] interpretation belongs to the journal.", flush=True)


if __name__ == "__main__":
    main()
