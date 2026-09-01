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
    ap.add_argument("--oracle-dyn", type=int, default=0,
                    help="ceiling arm (canon only): score each (move, reply) by "
                         "ENCODING the true child board instead of rolling -- "
                         "detection under perfect dynamics. Prints beside the "
                         "rolled arm from the same positions/moves/replies.")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    from catspace.research.tools.training_infra.qgt_search import QGTDynamics
    from catspace.research.tools.npz_io import load_npz
    dyn = QGTDynamics(args.ckpt, device=args.device)
    if args.oracle_dyn:
        assert getattr(dyn, "canon", False), "--oracle-dyn is canon-arm only"
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
    gaps_o, corr_o, hang_o = [], [], []          # oracle-dynamics ceiling arm
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
            if getattr(dyn, "canon", False):
                # canon: two 1-ply steps; min over replies (negamax, my POV back)
                r_es = np.empty(len(sel))
                o_es = np.empty(len(sel)) if args.oracle_dyn else None
                g_root = dyn.gates_for(b, moves) if hasattr(dyn, "gates_for") \
                    else None
                for si, j in enumerate(sel):
                    z1 = dyn.step(state, codes[j:j + 1], frm[j:j + 1],
                                  to[j:j + 1],
                                  gate=None if g_root is None
                                  else g_root[j:j + 1])
                    b1 = b.copy(stack=False); b1.push(moves[j])
                    reps1, cd1, f1, t1 = dyn.root_moves(b1, dyn.encode(b1)[1])
                    if not reps1:
                        r_es[si] = 1.0 if b1.is_checkmate() else 0.5
                        if o_es is not None:
                            o_es[si] = r_es[si]
                        continue
                    nr = len(reps1)
                    z2 = dyn.step(z1.expand(nr, -1, -1), cd1, f1, t1,
                                  gate=dyn.gates_for(b1, reps1))
                    pv = dyn.value(z2, 1).cpu().numpy()
                    r_es[si] = float((0.5 * (1.0 + pv[:, 0] - pv[:, 2])).min())
                    if o_es is not None:
                        # ceiling: SAME replies, SAME readout -- but the state is
                        # the ENCODED true child (perfect dynamics stand-in)
                        s2s = []
                        for rm in reps1:
                            b2 = b1.copy(stack=False); b2.push(rm)
                            s2s.append(dyn.encode(b2)[0])
                        pvo = dyn.value(torch.cat(s2s, 0), 1).cpu().numpy()
                        o_es[si] = float(
                            (0.5 * (1.0 + pvo[:, 0] - pvo[:, 2])).min())
            elif dyn.conditional:
                # T2-conditional (Kaveh 8/29): the reply is an INPUT; a move's read
                # is the MIN over its legal replies (the refutation, enumerated) --
                # the exact quantity Greedy1 backs up in the arena
                r_es = np.empty(len(sel))
                for si, j in enumerate(sel):
                    b1 = b.copy(stack=False); b1.push(moves[j])
                    reps = list(b1.legal_moves)
                    if not reps:
                        r_es[si] = 1.0 if b1.is_checkmate() else 0.5
                        continue
                    f2 = torch.tensor([r.from_square for r in reps],
                                      device=frm.device)
                    t2_ = torch.tensor([r.to_square for r in reps],
                                       device=frm.device)
                    nr = len(reps)
                    nxt = dyn.step(state.expand(nr, -1, -1),
                                   codes[j].expand(nr, -1),
                                   frm[j].expand(nr), to[j].expand(nr),
                                   frm2=f2, to2=t2_)
                    pv = dyn.value(nxt, 1 if b.turn else 0).cpu().numpy()
                    r_es[si] = float(es(pv, b.turn).min())
            else:
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
            if args.oracle_dyn:
                if len(sel) >= 4 and lab.std() > 1e-6 and o_es.std() > 1e-6:
                    corr_o.append(np.corrcoef(o_es, lab)[0, 1])
                gaps_o.append(float(o_es[ib] - o_es[iw]))
                if gap_l > 0.3:
                    hang_o.append(float(o_es[ib] - o_es[iw]))
            # B: one-ply attribution on the WORST move (the interesting one)
            mv = moves[j_w]
            b1 = b.copy(); b1.push(mv)
            rep = list(b1.legal_moves)
            if rep:
                b2 = b1.copy(); b2.push(rep[int(rng.integers(len(rep)))])
                s1, _ = dyn.encode(b1)
                s2, _ = dyn.encode(b2)
                rmv = b2.move_stack[-1]
                kw2 = (dict(frm2=torch.tensor([rmv.from_square], device=frm.device),
                            to2=torch.tensor([rmv.to_square], device=frm.device))
                       if dyn.conditional else {})
                t2 = dyn.step(state, codes[j_w:j_w + 1], frm[j_w:j_w + 1],
                              to[j_w:j_w + 1], **kw2)
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
    if args.oracle_dyn:
        go, ho = np.array(gaps_o), np.array(hang_o)
        print(f"[oracle] CEILING (encoded true children, same readout/replies): "
              f"within-position corr mean {np.mean(corr_o):.3f} (n={len(corr_o)}) "
              f"| blunder gap {ho.mean():.3f} | detection rate (gap>0.1): "
              f"{float((ho > 0.1).mean()):.1%} (n={len(ho)})", flush=True)
    print(f"[oneply] T2 output agrees with: 1-ply child {a1/an:.3f} | 2-ply child "
          f"{a2/an:.3f} | (root-copy baselines: {a0_1/an:.3f} / {a0_2/an:.3f}; "
          f"n={int(an)})", flush=True)
    print("[hang] interpretation belongs to the journal.", flush=True)


if __name__ == "__main__":
    main()
