#!/usr/bin/env python
"""probe_resolution_lr.py -- can a LINEAR readout of the concepts name the tablebase entrance
we are heading for, N plies before we get there? (Kaveh 2026-08-17)

"build a classifier on top of the concepts and see whether we can, simple logistic regression,
whether we can get the four hundred and one different outcome types out of the games, from the
concepts... let's try three different ply counts... ten moves ahead, so twenty plies ahead."

WHY LINEAR AND NOT A NETWORK. A linear probe measures what is LINEARLY DECODABLE from the
concept vocabulary -- i.e. what the concepts themselves encode, rather than what a sufficiently
large head could reconstruct from them. That is the interpretability-relevant quantity: a
concept set worth having should make "I am heading for a won rook endgame" readable, not merely
recoverable.

THE TARGET is the full 401-class resolution label: material signature AND solved outcome
(`KRP|KR=W` is a different class from `KRP|KR=D` -- the Philidor and the Lucena), plus the three
WDL classes for games that end before simplifying.

BASELINES, both reported, because the 401 classes are very unevenly used:
    majority   always predict the commonest class at this horizon
    material   the class most often reached from this position's MATERIAL SIGNATURE, fitted on
               train. At 20 plies out material is a weak predictor of the endgame you land in,
               but it is the honest floor, not chance.

    .venv/bin/python -m ...probe_resolution_lr --plies 10,20,40
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn.functional as F


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="artifacts/experiments/cdf5_latest.pt")
    ap.add_argument("--plies", default="10,20,40", help="horizons: plies before resolution")
    ap.add_argument("--band", type=int, default=2, help="+/- tolerance around each horizon")
    ap.add_argument("--n-train", type=int, default=60000)
    ap.add_argument("--n-test", type=int, default=20000)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--endgame-only", action="store_true",
                    help="restrict BOTH train and test to games that actually reach a "
                         "tablebase. Answers 'of the games that get there, can we name the "
                         "ending?' without the 78%% of positions whose game just stops.")
    args = ap.parse_args()

    from catspace.research.components.encoder.approaches.reach_probability.experiments \
        .concept_dynamics_field import ConceptDynamics
    from catspace.research.components.encoder.approaches.reach_probability.src import (
        trajectories as T)
    from catspace.io import paths

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    A = ck["args"]
    tr = T.build(n_human=0, n_sf=A["games"], seed=A["seed"], cache=True,
                 max_plies=A["max_plies"], n_piecedown=A["n_piecedown"])
    n = len(tr.tok)
    z = np.load(paths.derived(f"tb_entry_5-4-3men_{n}.npz"), allow_pickle=True)
    vocab = [str(x) for x in z["vocab"]]
    cls, plies, men = z["tb_class"].astype(np.int64), z["tb_plies"], z["men"]
    gor = tr.game_of_row()
    rs = np.random.default_rng(A["seed"]); perm = rs.permutation(len(tr.start))
    nv = max(1, int(A["val_frac"] * len(tr.start))); nt = max(1, int(A["test_frac"] * len(tr.start)))
    sog = np.zeros(len(tr.start), np.int8); sog[perm[:nv]] = 1; sog[perm[nv:nv + nt]] = 2
    spl = sog[gor]

    net = ConceptDynamics(d_model=A["d_model"], heads=A["heads"], codes=A["codes"],
                          square_codes=A["square_codes"], hidden=A["hidden"], ema=A["ema"])
    net.load_state_dict(ck["state_dict"]); net.eval()
    H, C, SC = A["heads"], A["codes"], A["square_codes"]

    # material signature per row, for the honest floor
    L = {1: "P", 2: "N", 3: "B", 4: "R", 5: "Q", 6: "K"}
    def sig_of(rows):
        t = tr.tok[rows]
        return [tuple((t == v).sum(1)) for v in range(1, 13)]

    def codes_for(rows, bs=1024):
        G = np.zeros((len(rows), H), np.int64); S = np.zeros((len(rows), 64), np.int64)
        with torch.no_grad():
            for a in range(0, len(rows), bs):
                r = rows[a:a + bs]
                _, g, s, _ = net.encode(torch.from_numpy(tr.tok[r].astype(np.int64)),
                                        torch.from_numpy(tr.glob[r].astype(np.float32)))
                G[a:a + len(r)] = g.numpy(); S[a:a + len(r)] = s.numpy()
        return G, S

    def onehot(G, S):
        X = torch.zeros(len(G), H * C + 64 * SC)
        gi = torch.from_numpy(G) + torch.arange(H)[None, :] * C
        X.scatter_(1, gi, 1.0)
        si = torch.from_numpy(S) + torch.arange(64)[None, :] * SC + H * C
        X.scatter_(1, si, 1.0)
        return X

    print(f"[lr] {args.ckpt.split('/')[-1]} | target = {len(vocab)}-class resolution "
          f"(material signature AND solved outcome)\n")
    print(f"  {'horizon':>9} {'n train':>9} {'n test':>8} {'classes':>8} "
          f"{'LINEAR':>8} {'majority':>9} {'material':>9} {'on WDL':>9} "
          f"{'on ENDGAME':>10} {'eg outcome':>10}  {'%eg':>6}")
    for P in [int(x) for x in args.plies.split(",")]:
        band = (plies >= P - args.band) & (plies <= P + args.band) & (cls >= 0) & (men > 5)
        if args.endgame_only:
            is_eg = np.array([not vocab[c].startswith("WDL:") if c >= 0 else False
                              for c in cls])
            band = band & is_eg
        rtr = np.flatnonzero(band & (spl == 0)); rte = np.flatnonzero(band & (spl == 2))
        if len(rtr) < 2000 or len(rte) < 500:
            print(f"  {P:>9} {len(rtr):>9,} {len(rte):>8,}   too few positions at this horizon")
            continue
        rng = np.random.default_rng(0)
        rtr = rtr[rng.choice(len(rtr), min(args.n_train, len(rtr)), replace=False)]
        rte = rte[rng.choice(len(rte), min(args.n_test, len(rte)), replace=False)]
        ytr, yte = cls[rtr], cls[rte]

        Gtr, Str = codes_for(rtr); Gte, Ste = codes_for(rte)
        Xtr, Xte = onehot(Gtr, Str), onehot(Gte, Ste)
        # multinomial logistic regression = ONE linear layer + softmax, fitted with Adam
        lin = torch.nn.Linear(Xtr.shape[1], len(vocab))
        opt = torch.optim.Adam(lin.parameters(), lr=3e-3, weight_decay=1e-5)
        Ytr = torch.from_numpy(ytr)
        for ep in range(args.epochs):
            pm = torch.randperm(len(Xtr))
            for a in range(0, len(Xtr), 4096):
                idx = pm[a:a + 4096]
                loss = F.cross_entropy(lin(Xtr[idx]), Ytr[idx])
                opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            pred = lin(Xte).argmax(1).numpy()
        acc = float((pred == yte).mean())
        # SPLIT THE TASK. 78% of these positions carry a WDL class ("the game just ends"),
        # so an overall number is mostly 3-way outcome prediction wearing a 401-class costume.
        # The endgame-naming question lives in the other 22% and must be scored separately.
        is_wdl = np.array([vocab[c].startswith("WDL:") for c in yte])
        acc_wdl = float((pred[is_wdl] == yte[is_wdl]).mean()) if is_wdl.any() else float("nan")
        acc_eg = float((pred[~is_wdl] == yte[~is_wdl]).mean()) if (~is_wdl).any() else float("nan")
        # and among true endgame positions, does it at least get the OUTCOME letter right?
        out_of = np.array([vocab[c].split("=")[-1] if "=" in vocab[c] else vocab[c][-1]
                           for c in range(len(vocab))])
        acc_eg_outcome = float((out_of[pred[~is_wdl]] == out_of[yte[~is_wdl]]).mean()) \
            if (~is_wdl).any() else float("nan")

        maj = Counter(ytr.tolist()).most_common(1)[0][0]
        acc_maj = float((yte == maj).mean())
        # material floor: commonest resolution per material signature, fitted on train
        str_tr = np.stack(sig_of(rtr), 1); str_te = np.stack(sig_of(rte), 1)
        ktr = [tuple(r) for r in str_tr]; kte = [tuple(r) for r in str_te]
        tab = defaultdict(Counter)
        for k, yy in zip(ktr, ytr.tolist()): tab[k][yy] += 1
        best = {k: c.most_common(1)[0][0] for k, c in tab.items()}
        acc_mat = float(np.mean([best.get(k, maj) == yy for k, yy in zip(kte, yte.tolist())]))
        print(f"  {P:>9} {len(rtr):>9,} {len(rte):>8,} {len(set(ytr.tolist())):>8} "
              f"{acc:>8.3f} {acc_maj:>9.3f} {acc_mat:>9.3f} {acc_wdl:>9.3f} "
              f"{acc_eg:>10.3f} {acc_eg_outcome:>10.3f}  {100*(~is_wdl).mean():>5.1f}%")
    print(f"\n  LINEAR = multinomial logistic regression on one-hot concept codes "
          f"({H}x{C} global + 64x{SC} square).")


if __name__ == "__main__":
    main()
