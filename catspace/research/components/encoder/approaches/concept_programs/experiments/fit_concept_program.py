#!/usr/bin/env python
"""fit_concept_program.py -- distil ONE learned concept into a program over named primitives.

Step 2 of the distillation loop. Takes a trained checkpoint, picks a concept (one of the 8 global
VQ heads, or the shared 64-code square codebook), and fits a `SoftAST` over the Rust probe basis
to reproduce THAT CONCEPT'S CODE. Fit and gate are the same quantity -- code agreement -- with the
soft nearest-code relaxation supplying the gradient.

WHAT IS REPORTED, and why all of it. A fidelity number alone is not enough to justify freezing a
concept:

    agreement (soft)    what the annealed program scores while still at temperature
    agreement (hard)    what the DISCRETE program scores -- the only one that can be emitted
    hardening cost      the drop between them; a large drop means the anneal did not finish,
                        which is the central failure mode of this design
    program size        live nodes reachable from the outputs; a program that only matches by
                        being enormous defeats the point of doing this at all
    majority baseline   the code that fires most often; agreement below this is worse than a
                        constant, and a constant is not a concept

    .venv/bin/python -m ...fit_concept_program --ckpt artifacts/experiments/ccf2_latest.pt --head 0
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="artifacts/experiments/ccf2_latest.pt")
    ap.add_argument("--head", type=int, default=0,
                    help="global VQ head 0..heads-1, or -1 for the shared square codebook")
    ap.add_argument("--n", type=int, default=20000, help="rows to fit on")
    ap.add_argument("--n-test", type=int, default=8000, help="HELD-OUT rows for the gate")
    ap.add_argument("--nodes", type=int, default=8)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--tau0", type=float, default=2.0)
    ap.add_argument("--tau1", type=float, default=0.02)
    ap.add_argument("--tau-c", type=float, default=0.5)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    import catspace_concepts as cc
    from catspace.research.components.encoder.approaches.concept_programs.src.ast import (
        SoftAST, code_agreement, describe_program, soft_code_ce)
    from catspace.research.components.encoder.approaches.reach_probability.experiments \
        .concept_capture_field import ConceptCapture, factor_vocab
    from catspace.research.components.encoder.approaches.reach_probability.src import (
        trajectories as T)
    from catspace.io import paths

    torch.manual_seed(args.seed)
    t0 = time.time()
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    A = ck["args"] if isinstance(ck.get("args"), dict) else vars(ck["args"])
    tr = T.build(n_human=0, n_sf=A["games"], seed=A["seed"], cache=True,
                 max_plies=A["max_plies"], n_piecedown=A["n_piecedown"])
    n = len(tr.tok)
    lz = np.load(paths.derived(f"tb_entry_5-4-3men_{n}.npz"), allow_pickle=True)
    vocab = [str(x) for x in lz["vocab"]]
    cr, ce, co, egs = factor_vocab(vocab)
    factored = any(k.startswith("res_reach") for k in ck["state_dict"])
    net = ConceptCapture(d_model=A["d_model"], heads=A["heads"], codes=A["codes"],
                         square_codes=A["square_codes"], hidden=A["hidden"],
                         n_res=len(vocab), factored=factored, n_eg=len(egs),
                         cls_reach=cr, cls_eg=ce, cls_out=co)
    net.load_state_dict(ck["state_dict"]); net.eval()

    # split by GAME, matching the trainer, so the gate is measured on games the concept model
    # never saw either
    gor = tr.game_of_row()
    rs = np.random.default_rng(A["seed"]); perm = rs.permutation(len(tr.start))
    nv = max(1, int(A["val_frac"] * len(tr.start))); nt = max(1, int(A["test_frac"] * len(tr.start)))
    sog = np.zeros(len(tr.start), np.int8); sog[perm[:nv]] = 1; sog[perm[nv:nv + nt]] = 2
    spl = sog[gor]
    rng = np.random.default_rng(args.seed)
    tr_rows = np.flatnonzero(spl == 0); te_rows = np.flatnonzero(spl == 2)
    tr_rows = tr_rows[rng.choice(len(tr_rows), min(args.n, len(tr_rows)), replace=False)]
    te_rows = te_rows[rng.choice(len(te_rows), min(args.n_test, len(te_rows)), replace=False)]

    def features_and_codes(rows):
        sq, gl, failed = cc.extract(np.ascontiguousarray(tr.tok[rows]),
                                    np.ascontiguousarray(tr.glob[rows]))
        assert failed == 0, f"{failed} rows failed to build a position"
        with torch.no_grad():
            ids, ids_sq = [], []
            for a in range(0, len(rows), 1024):
                r = rows[a:a + 1024]
                _, i_g, i_s, _ = net.encode(
                    torch.from_numpy(tr.tok[r].astype(np.int64)),
                    torch.from_numpy(tr.glob[r].astype(np.float32)))
                ids.append(i_g); ids_sq.append(i_s)
        return (torch.from_numpy(sq), torch.from_numpy(gl),
                torch.cat(ids), torch.cat(ids_sq))

    sq_tr, gl_tr, ids_tr, idsq_tr = features_and_codes(tr_rows)
    sq_te, gl_te, ids_te, idsq_te = features_and_codes(te_rows)

    is_square = args.head < 0
    if is_square:
        codebook = net.sq_vq._codebook.embed[0].detach()          # (64, d_fine)
        y_tr, y_te = idsq_tr.reshape(-1), idsq_te.reshape(-1)
        label = "square codebook (shared across 64 squares)"
    else:
        codebook = net.vq[args.head]._codebook.embed[0].detach()  # (16, d_code)
        y_tr, y_te = ids_tr[:, args.head], ids_te[:, args.head]
        label = f"global head {args.head}"
    K, d_out = codebook.shape

    maj = int(torch.bincount(y_tr, minlength=K).argmax())
    base = float((y_te == maj).float().mean())
    used = int((torch.bincount(y_tr, minlength=K) > 0).sum())
    print(f"\n  {args.ckpt.split('/')[-1]} step {ck.get('step')} | {label}")
    print(f"  {K} codes ({used} in use) | fit {len(tr_rows):,} rows, gate {len(te_rows):,} "
          f"HELD-OUT rows | [{time.time()-t0:.0f}s]")
    print(f"  majority-code baseline on held-out: {base:.4f}  (chance {1/K:.4f})")

    sq_names, gl_names = cc.feature_names()
    ast = SoftAST(sq_tr.shape[-1], gl_tr.shape[-1], n_nodes=args.nodes, d_out=d_out,
                  seed=args.seed)
    opt = torch.optim.Adam(ast.parameters(), lr=args.lr)

    def predict(a_sq, a_gl, tau, hard=False):
        o = ast(a_sq, a_gl, tau=tau, hard=hard)          # (B,64,d)
        return o.reshape(-1, d_out) if is_square else o.mean(1)

    print(f"\n  {'step':>6} {'tau':>6} {'loss':>8} {'agree':>7}")
    nrow = len(sq_tr)
    for s in range(1, args.steps + 1):
        # geometric anneal: the selectors must be one-hot by the end or the hardened program is
        # not the program that was trained
        tau = args.tau0 * (args.tau1 / args.tau0) ** (s / args.steps)
        idx = torch.from_numpy(rng.choice(nrow, min(args.batch, nrow), replace=False))
        p = predict(sq_tr[idx], gl_tr[idx], tau)
        # square concepts carry one label per (row, square); global concepts one per row
        t = y_tr.reshape(nrow, 64)[idx].reshape(-1) if is_square else y_tr[idx]
        loss = soft_code_ce(p, codebook, t, tau_c=args.tau_c)
        opt.zero_grad(); loss.backward(); opt.step()
        if s % max(1, args.steps // 10) == 0 or s == 1:
            with torch.no_grad():
                ag = code_agreement(p, codebook, t)
            print(f"  {s:>6} {tau:>6.3f} {float(loss):>8.4f} {ag:>7.4f}", flush=True)

    # ---- the gate: HELD-OUT, and the hardened program, not the soft one -------------
    with torch.no_grad():
        y_te_flat = y_te
        soft_ag = code_agreement(predict(sq_te, gl_te, args.tau1), codebook, y_te_flat)
        hard_ag = code_agreement(predict(sq_te, gl_te, args.tau1, hard=True), codebook, y_te_flat)
    prog = ast.harden()
    size = ast.size()

    print(f"\n  RESULT — {label}")
    print(f"    agreement, soft   {soft_ag:.4f}")
    print(f"    agreement, HARD   {hard_ag:.4f}   <- the only one that can be emitted")
    print(f"    hardening cost    {soft_ag - hard_ag:+.4f}")
    print(f"    majority baseline {base:.4f}   lift {hard_ag - base:+.4f}")
    print(f"    program size      {size} live nodes of {args.nodes}")
    print(f"\n  PROGRAM (only the sub-DAG reachable from the outputs):")
    print(describe_program(prog, sq_names, gl_names))

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"program": prog, "head": args.head, "soft": soft_ag, "hard": hard_ag,
                       "baseline": base, "size": size, "ckpt": args.ckpt}, f, indent=1)
        print(f"\n  wrote {args.out}")
    print(f"\n  [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
