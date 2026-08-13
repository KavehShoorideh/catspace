#!/usr/bin/env python
"""probe_codes_outcome.py -- how much W/D/L do the CONCEPTS ALONE carry? (Kaveh 2026-08-13:
"how much do the concepts predict the win/loss/draw of the entire dataset ... take that
predictor from the SF corpus and apply it to the human dataset")

Protocol (linear-probe standard):
    1. code positions from the SF-SF corpus with the frozen quantizer (global 8-head ids)
    2. features = one-hot codes (8x64) + side-to-move (NOTHING else -- no board, no eval)
    3. fit multinomial logistic W/D/L on probe-TRAIN games, report on probe-TEST games
       (split by GAME so no position leaks)
    4. apply the SAME fitted probe, unchanged, to lichess human games (time-forfeits
       dropped: board-unrelated endings, the conversion-censoring rule)
Baselines: majority class; a ply-phase-only probe (is it just "endgames are decisive"?).

    .venv/bin/python -m ...probe_codes_outcome --ckpt artifacts/experiments/reach_jqt3_latest.pt
"""
from __future__ import annotations

import argparse
import time

import chess
import numpy as np
import torch


def code_games(model, jqt, games_rows, dev, batch=256):
    """games_rows: list of (tok(N,64), glob(N,g), result, plies_total) -> features+labels"""
    X, y, phase = [], [], []
    toks = np.concatenate([g[0] for g in games_rows])
    globs = np.concatenate([g[1] for g in games_rows])
    ids_all = []
    with torch.no_grad():
        for a in range(0, len(toks), batch):
            phi = model.backbone(
                torch.from_numpy(toks[a:a + batch].astype(np.int64)).to(dev),
                torch.from_numpy(globs[a:a + batch].astype(np.float32)).to(dev))
            ids_all.append(jqt.target_codes(phi)[1].cpu().numpy())
    ids_all = np.concatenate(ids_all)
    ofs = 0
    for tok_g, glob_g, res, _tp in games_rows:
        n = len(tok_g)
        ids = ids_all[ofs:ofs + n]
        ofs += n
        for t in range(n):
            X.append(ids[t])
            y.append(res)
            phase.append(min(t / max(n - 1, 1), 1.0))
    return np.array(X), np.array(y), np.array(phase), ids_all


def onehot(ids, stm, C=64):
    H = ids.shape[1]
    X = np.zeros((len(ids), H * C + 1), np.float32)
    for h in range(H):
        X[np.arange(len(ids)), h * C + ids[:, h]] = 1.0
    X[:, -1] = stm
    return X


def fit_logreg(X, y, l2=1e-3, iters=300, lr=0.5, dev="cpu"):
    """multinomial logistic via torch (no sklearn dependency); full-batch LBFGS-ish Adam."""
    Xt = torch.from_numpy(X).to(dev)
    yt = torch.from_numpy(y).long().to(dev)
    W = torch.zeros(X.shape[1], 3, requires_grad=True, device=dev)
    b = torch.zeros(3, requires_grad=True, device=dev)
    opt = torch.optim.Adam([W, b], lr=lr)
    for _ in range(iters):
        lg = Xt @ W + b
        loss = torch.nn.functional.cross_entropy(lg, yt) + l2 * W.pow(2).sum()
        opt.zero_grad(); loss.backward(); opt.step()
    return W.detach(), b.detach()


def eval_probe(W, b, X, y, dev="cpu"):
    with torch.no_grad():
        lg = torch.from_numpy(X).to(dev) @ W + b
        p = torch.softmax(lg, -1).numpy()
    acc = float((p.argmax(1) == y).mean())
    nll_bits = float(-np.log2(np.clip(p[np.arange(len(y)), y], 1e-9, 1)).mean())
    return acc, nll_bits


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="artifacts/experiments/reach_jqt3_latest.pt")
    ap.add_argument("--sf-games", type=int, default=500)
    ap.add_argument("--human-games", type=int, default=500)
    ap.add_argument("--plies-per-game", type=int, default=24)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    import os, re
    from catspace.research.components.encoder.approaches.reach_probability.src import (
        trajectories as T)
    from catspace.research.components.encoder.approaches.reach_probability.experiments.jqt import (
        JQTModule)
    from catspace.research.components.encoder.approaches.reach_probability.experiments.interpret_reach import (
        load_net)
    from catspace.research.components.encoder.approaches.jepa_tokenizer.src.jepa import (
        tokenize)
    dev = args.device
    base = args.ckpt[:-3] if args.ckpt.endswith(".pt") else args.ckpt
    stem = re.sub(r"_(latest|step\d+)$", "", base)
    model, _ = load_net(args.ckpt, dev)
    model.eval()
    pj = torch.load(next(p for p in (base + "_jqt.pt", stem + "_jqt.pt")
                         if os.path.exists(p)), map_location=dev, weights_only=False)
    jqt = JQTModule(d_model=pj["d_in"], heads=pj["heads"], codes=pj["codes"], d=pj["d"],
                    square_codes=pj.get("square_codes", 0),
                    piece_codes=pj.get("piece_codes", 0)).to(dev)
    jqt.load_state_dict(pj["state_dict"], strict=False)
    jqt.eval()
    C = pj["codes"]
    rng = np.random.default_rng(0)

    def replay_sample(gtuples, n_g, tag):
        """(gid,res,ucis,flag[,elos[,start]]) -> per-game (toks, globs, cls, total)"""
        out, used = [], 0
        for g in gtuples:
            res, ucis, flag = g[1], g[2], g[3]
            if flag:
                continue                       # time forfeits: board-unrelated endings
            b = chess.Board(g[5]) if len(g) > 5 and g[5] else chess.Board()
            keep = sorted(rng.choice(len(ucis), size=min(args.plies_per_game, len(ucis)),
                                     replace=False))
            tks, gls, stms = [], [], []
            ok = True
            ptr = 0
            for t, u in enumerate(ucis):
                try:
                    mv = chess.Move.from_uci(u)
                    if mv not in b.legal_moves:
                        ok = False; break
                    if ptr < len(keep) and t == keep[ptr]:
                        ptr += 1
                        tk, gl = tokenize(b)
                        tks.append(np.asarray(tk)); gls.append(np.asarray(gl))
                        stms.append(1.0 if b.turn else 0.0)
                    b.push(mv)
                except Exception:
                    ok = False; break
            if ok and tks:
                cls = 0 if res == 1 else (2 if res == 0 else 1)   # W/D/L white-POV
                out.append((np.stack(tks), np.stack(gls), cls, len(ucis)))
                out[-1] = out[-1] + (np.array(stms, np.float32),)
                used += 1
            if used >= n_g:
                break
        print(f"[probe] {tag}: {used} games replayed", flush=True)
        return out

    t0 = time.time()
    sf_games = T.load_piecedown_games(args.sf_games * 2, seed=1)
    sf = replay_sample(sf_games, args.sf_games, "sf-corpus")
    hu_games = T.load_human_games(args.human_games * 2, seed=1)
    hu = replay_sample(hu_games, args.human_games, "human")
    print(f"[probe] replay {time.time()-t0:.0f}s", flush=True)

    def featurize(games_list):
        rows = [(g[0], g[1], g[2], g[3]) for g in games_list]
        ids, y, phase, _ = code_games(model, jqt, rows, dev)
        stm = np.concatenate([g[4] for g in games_list])
        return onehot(ids, stm, C), y, phase

    t0 = time.time()
    n_tr = int(len(sf) * 0.7)
    Xtr, ytr, phtr = featurize(sf[:n_tr])
    Xte, yte, phte = featurize(sf[n_tr:])
    Xhu, yhu, phhu = featurize(hu)
    print(f"[probe] coded sf-train {len(Xtr)} sf-test {len(Xte)} human {len(Xhu)} "
          f"[{time.time()-t0:.0f}s]", flush=True)

    maj = np.bincount(ytr, minlength=3).argmax()
    W, b = fit_logreg(Xtr, ytr)
    acc_te, nll_te = eval_probe(W, b, Xte, yte)
    acc_hu, nll_hu = eval_probe(W, b, Xhu, yhu)
    # ---- BY STAGE (Kaveh 2026-08-13: "what stage did you take the concepts from? the
    # last concept might just [give away] the position"): position-level accuracy per
    # game-fraction bucket -- how early do the codes know?
    print("\n[probe] position-level accuracy BY GAME STAGE (SF-test | human-transfer):")
    for lo, hi in ((0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)):
        m1 = (phte >= lo) & (phte < hi)
        m2 = (phhu >= lo) & (phhu < hi)
        a1, n1 = eval_probe(W, b, Xte[m1], yte[m1]) if m1.sum() > 30 else (float("nan"), 0)
        a2, n2 = eval_probe(W, b, Xhu[m2], yhu[m2]) if m2.sum() > 30 else (float("nan"), 0)
        print(f"  {int(lo*100):3d}-{int(hi*100):3d}%   sf {a1:.1%} ({n1:.2f} bits)   "
              f"human {a2:.1%} ({n2:.2f} bits)")
    # ---- WINDOWED CONCEPT SET (one sample per GAME): bag-of-codes seen inside a
    # fractional window -> final outcome. The clean "prediction, not description" number.
    def window_feats(games_list, X, ph, lo, hi):
        Xg, yg, ofs = [], [], 0
        for g in games_list:
            n = len(g[0])
            sl = slice(ofs, ofs + n)
            m = (ph[sl] >= lo) & (ph[sl] < hi)
            if m.sum() >= 2:
                Xg.append(np.clip(X[sl][m][:, :-1].sum(0), 0, 1))   # multi-hot, stm dropped
                yg.append(g[2])
            ofs += n
        return np.stack(Xg).astype(np.float32), np.array(yg)
    print("\n[probe] WINDOWED concept-set -> final outcome (one sample per game):")
    for lo, hi in ((0.1, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9)):
        Xg_tr, yg_tr = window_feats(sf[:n_tr], Xtr, phtr, lo, hi)
        Xg_te, yg_te = window_feats(sf[n_tr:], Xte, phte, lo, hi)
        Xg_hu, yg_hu = window_feats(hu, Xhu, phhu, lo, hi)
        Wg, bg = fit_logreg(Xg_tr, yg_tr, l2=3e-3)
        ag_te, ng_te = eval_probe(Wg, bg, Xg_te, yg_te)
        ag_hu, ng_hu = eval_probe(Wg, bg, Xg_hu, yg_hu)
        mj_te = float((yg_te == np.bincount(yg_tr, minlength=3).argmax()).mean())
        mj_hu = float((yg_hu == np.bincount(yg_tr, minlength=3).argmax()).mean())
        print(f"  plies {int(lo*100):3d}-{int(hi*100):3d}% of game:  "
              f"sf {ag_te:.1%} ({ng_te:.2f} bits, maj {mj_te:.1%})   "
              f"human {ag_hu:.1%} ({ng_hu:.2f} bits, maj {mj_hu:.1%})")
    # phase-only control: is it just "late positions are decisive"?
    Ptr = np.stack([phtr, phtr ** 2, np.ones_like(phtr)], 1).astype(np.float32)
    Pte = np.stack([phte, phte ** 2, np.ones_like(phte)], 1).astype(np.float32)
    Wp, bp = fit_logreg(Ptr, ytr)
    acc_ph, nll_ph = eval_probe(Wp, bp, Pte, yte)
    # human-refit ceiling: same features, probe refitted ON human games (70/30)
    nh = int(len(hu) * 0.7)
    Xh1, yh1, _ = featurize(hu[:nh])
    Xh2, yh2, _ = featurize(hu[nh:])
    Wh, bh = fit_logreg(Xh1, yh1)
    acc_hr, nll_hr = eval_probe(Wh, bh, Xh2, yh2)

    print("\n[probe] VERDICT codes-only W/D/L (8x64 one-hot + stm; game-split):")
    print(f"  sf-corpus test   acc {acc_te:.1%}  nll {nll_te:.2f} bits   "
          f"(majority {float((yte==maj).mean()):.1%}, phase-only {acc_ph:.1%}/{nll_ph:.2f})")
    print(f"  ->human TRANSFER acc {acc_hu:.1%}  nll {nll_hu:.2f} bits   "
          f"(majority {float((yhu==maj).mean()):.1%})")
    print(f"  human-refit      acc {acc_hr:.1%}  nll {nll_hr:.2f} bits   "
          f"(same codes, probe refitted on human games)")


if __name__ == "__main__":
    main()
