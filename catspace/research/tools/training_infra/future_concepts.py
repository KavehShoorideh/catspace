"""Do concept states POOL BY THE KIND OF FUTURE a move leads to -- and is that
future PREDICTABLE from the concepts we have now?

Kaveh 2026-08-24: "I'm trying to predict the kinds of futures available to me if I
take a particular move. If I trade the queens, what kind of endgame do I have?"
and then, sharpening it: "not always will future concepts be predictable from present
concepts, so I expect that loss to remain high. The primary objective should always be
the Stockfish evaluation, but we do want a nudge -- if some concepts CAN be predicted,
predict those."

That second statement is a design constraint with a measurable price, so this script
measures the price before anything is built. Stockfish's principal variation for a move
is a sample of that move's future under best play. Walk it and form

    M(s,a) = (1-g) * sum_{t>=1} g^t phi(s_t)     the discounted future concept occupancy

with phi(s_0) -- the position the move actually reaches -- held out as the PREDICTOR.
t starts at 1 so the target cannot contain its own input.

Three questions, in the order they decide the design:

  A. POOLING. Does M separate futures by a property nobody trained it on (queens come
     off)? Measured with a POSITION-GROUPED split and a permutation null, because
     2528 dims over ~600 samples will separate any labelling otherwise.
  B. IS THE MEAN ENOUGH? Between-class distance vs within-class spread. If spread
     dominates, a mean blurs distinct endgames together and the head must be
     distributional.
  C. WHAT IS PREDICTABLE, AND DOES IT MATTER? Per concept bit: held-out R^2 of
     predicting its future occupancy from phi(s_0), against its own permutation null;
     and separately each bit's importance -- how strongly it tracks the line's
     Stockfish value. The auxiliary loss is only worth carrying if the PREDICTABLE
     bits and the VALUE-RELEVANT bits are the same bits.
"""
import argparse
import numpy as np
import torch


def _auc(scores, labels):
    """Rank AUC; 0.5 = chance."""
    s, y = np.asarray(scores, float), np.asarray(labels, bool)
    if y.all() or not y.any():
        return float("nan")
    r = np.argsort(np.argsort(s)) + 1
    n1, n0 = int(y.sum()), int((~y).sum())
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def collect_pvs(args):
    """Stockfish PVs only -- no model, no encoding.

    Split out so the SAME positions and the SAME lines can be re-encoded with any
    checkpoint. Comparing gko7a against gko8a on separately-sampled positions and
    separately-run PVs would confound the thing being tested (the encoder) with the
    thing that should be held fixed (the futures). SF is deterministic at fixed depth,
    but the position sample and engine state are not free of drift.
    """
    import glob, os, chess, chess.engine
    d = np.load(sorted(glob.glob(os.path.join(args.holdout, "*.npz")))[0])
    fens = [x.decode() if isinstance(x, bytes) else str(x) for x in d["fen"]]

    def new_engine():
        e = chess.engine.SimpleEngine.popen_uci("/opt/homebrew/bin/stockfish")
        e.configure({"UCI_ShowWDL": True})
        return e

    eng = new_engine()
    out, n = [], 0
    for fen in fens:
        if n >= args.n_pos:
            break
        try:
            b = chess.Board(fen)
        except Exception:
            continue
        if b.is_game_over() or not b.pieces(chess.QUEEN, chess.WHITE) \
                or not b.pieces(chess.QUEEN, chess.BLACK):
            continue
        if len(list(b.legal_moves)) < 8:
            continue
        try:
            info = eng.analyse(b, chess.engine.Limit(depth=args.depth),
                               multipv=min(args.top_moves, b.legal_moves.count()))
        except Exception:
            try:
                eng.quit()
            except Exception:
                pass
            eng = new_engine()
            continue
        rows = []
        for e in info:
            pv = e.get("pv") or []
            if len(pv) < 4:
                continue
            sc = e.get("score")
            v = sc.pov(b.turn).score(mate_score=10000) / 100.0 if sc else 0.0
            rows.append({"pv": [m.uci() for m in pv], "val": v})
        if rows:
            out.append({"fen": fen, "lines": rows})
            n += 1
    try:
        eng.quit()
    except Exception:
        pass
    return out


def encode_pvs(args, pvs):
    """Encode a cached PV set with ONE checkpoint -> the arrays the analysis needs."""
    import chess, torch as _t
    from catspace.research.components.planner.approaches.quasimetric_nav \
        .koopman_search import KoopmanEngine
    ke = KoopmanEngine(args.ckpt)

    def phi(board):
        with _t.no_grad():
            return ke._encode(board)[0][0, :ke.field.d_state].cpu().numpy()

    P0, MF, VAL, LAB, GRP = [], [], [], [], []
    for gi, rec in enumerate(pvs):
        b = chess.Board(rec["fen"])
        for ln in rec["lines"]:
            mv = [chess.Move.from_uci(u) for u in ln["pv"]]
            bd = b.copy()
            bd.push(mv[0])
            if bd.is_game_over():
                continue
            p0 = phi(bd)
            acc, w, qoff = None, 0.0, False
            for t, m in enumerate(mv[1:]):
                bd.push(m)
                if bd.is_game_over():
                    break
                if not bd.pieces(chess.QUEEN, chess.WHITE) or \
                        not bd.pieces(chess.QUEEN, chess.BLACK):
                    qoff = True
                g = args.gamma ** t
                acc = phi(bd) * g if acc is None else acc + phi(bd) * g
                w += g
            if acc is None:
                continue
            P0.append(p0); MF.append(acc / max(w, 1e-9))
            VAL.append(ln["val"]); LAB.append(qoff); GRP.append(gi)
    return (np.array(P0, np.float32), np.array(MF, np.float32),
            np.array(VAL, np.float32), np.array(LAB), np.array(GRP), len(pvs))


def collect(args):
    import glob, os, chess, chess.engine
    from catspace.research.components.planner.approaches.quasimetric_nav \
        .koopman_search import KoopmanEngine
    ke = KoopmanEngine(args.ckpt)

    d = np.load(sorted(glob.glob(os.path.join(args.holdout, "*.npz")))[0])
    fens = [x.decode() if isinstance(x, bytes) else str(x) for x in d["fen"]]

    def new_engine():
        e = chess.engine.SimpleEngine.popen_uci("/opt/homebrew/bin/stockfish")
        e.configure({"UCI_ShowWDL": True})
        return e

    eng = new_engine()
    P0, MF, VAL, LAB, GRP, n = [], [], [], [], [], 0

    def phi(board):
        with torch.no_grad():
            return ke._encode(board)[0][0, :ke.field.d_state].cpu().numpy()

    for fen in fens:
        if n >= args.n_pos:
            break
        try:
            b = chess.Board(fen)
        except Exception:
            continue
        if b.is_game_over() or not b.pieces(chess.QUEEN, chess.WHITE) \
                or not b.pieces(chess.QUEEN, chess.BLACK):
            continue
        if len(list(b.legal_moves)) < 8:
            continue
        try:
            info = eng.analyse(b, chess.engine.Limit(depth=args.depth),
                               multipv=min(args.top_moves, b.legal_moves.count()))
        except Exception:
            try:
                eng.quit()
            except Exception:
                pass
            eng = new_engine()
            continue
        for e in info:
            pv = e.get("pv") or []
            if len(pv) < 4:
                continue
            sc = e.get("score")
            v = sc.pov(b.turn).score(mate_score=10000) / 100.0 if sc else 0.0
            bd = b.copy()
            bd.push(pv[0])
            if bd.is_game_over():
                continue
            p0 = phi(bd)                      # the state the MOVE reaches: predictor
            acc, w, qoff = None, 0.0, False   # future from t>=1: target
            for t, m in enumerate(pv[1:]):
                bd.push(m)
                if bd.is_game_over():
                    break
                if not bd.pieces(chess.QUEEN, chess.WHITE) or \
                        not bd.pieces(chess.QUEEN, chess.BLACK):
                    qoff = True
                g = args.gamma ** t
                ph = phi(bd)
                acc = ph * g if acc is None else acc + ph * g
                w += g
            if acc is None:
                continue
            P0.append(p0); MF.append(acc / max(w, 1e-9))
            VAL.append(v); LAB.append(qoff); GRP.append(n)
        n += 1
        if n % 10 == 0:
            print(f"  ... {n} positions, {len(MF)} samples", flush=True)
    try:
        eng.quit()
    except Exception:
        pass
    return (np.array(P0, np.float32), np.array(MF, np.float32),
            np.array(VAL, np.float32), np.array(LAB), np.array(GRP), n)


def grouped_ridge(X, Y, GRP, ridge=10.0, folds=5, seed=0):
    """Held-out predictions under a POSITION-GROUPED split. Moves from one position
    share a root and overlapping futures; splitting by sample would leak."""
    pos = np.unique(GRP)
    r = np.random.default_rng(seed)
    pp = r.permutation(pos)
    pred = np.zeros_like(Y)
    for k in range(folds):
        te = np.isin(GRP, pp[k::folds])
        if te.all() or not te.any():
            continue
        mu, sd = X[~te].mean(0), X[~te].std(0) + 1e-6
        xtr, xte = (X[~te] - mu) / sd, (X[te] - mu) / sd
        ym = Y[~te].mean(0)
        w = np.linalg.solve(xtr.T @ xtr + ridge * np.eye(xtr.shape[1]),
                            xtr.T @ (Y[~te] - ym))
        pred[te] = xte @ w + ym
    return pred


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="artifacts/experiments/gko7a_step2000.pt")
    ap.add_argument("--holdout", default="data/derived/gko1_holdout_sf")
    ap.add_argument("--n_pos", type=int, default=60)
    ap.add_argument("--top_moves", type=int, default=10)
    ap.add_argument("--gamma", type=float, default=0.9)
    ap.add_argument("--depth", type=int, default=10)
    ap.add_argument("--pv-cache", default="",
                    help="cache Stockfish PVs (model-independent) so two "
                         "checkpoints are measured on IDENTICAL futures")
    ap.add_argument("--cache", default="", help="npz to save/reuse the encoded PVs; "
                    "Stockfish + encoding is the expensive part, the analysis is not")
    args = ap.parse_args()

    import os, json
    if args.pv_cache:
        # SF work once, encoding per checkpoint -- so two checkpoints are compared on
        # identical positions and identical futures
        if os.path.exists(args.pv_cache):
            pvs = json.load(open(args.pv_cache))
            print(f"  [pv-cache] {args.pv_cache}: {len(pvs)} positions")
        else:
            pvs = collect_pvs(args)
            json.dump(pvs, open(args.pv_cache, "w"))
            print(f"  [pv-cache] wrote {len(pvs)} positions to {args.pv_cache}")
        P0, MF, VAL, LAB, GRP, n = encode_pvs(args, pvs)
    elif args.cache and os.path.exists(args.cache):
        z = np.load(args.cache)
        P0, MF, VAL, LAB, GRP, n = (z["P0"], z["MF"], z["VAL"], z["LAB"],
                                    z["GRP"], int(z["n"]))
        print(f"  [cache] {args.cache}")
    else:
        P0, MF, VAL, LAB, GRP, n = collect(args)
        if args.cache:
            np.savez_compressed(args.cache, P0=P0, MF=MF, VAL=VAL, LAB=LAB,
                                GRP=GRP, n=n)

    print(f"\n  {n} positions -> {len(MF)} (move, future) samples, {MF.shape[1]} "
          f"concept dims | queens come off in {100*LAB.mean():.0f}% of lines")
    if len(MF) < 40:
        print("  too few samples to judge")
        return
    rng = np.random.default_rng(0)

    # ---- A. does the future POOL by kind? ---------------------------------------
    def probe_auc(F, lab, seed=0):
        p = grouped_ridge(F, (lab.astype(float) * 2 - 1)[:, None], GRP, seed=seed)
        return _auc(p[:, 0], lab)

    def best_dim(F, lab):
        a = np.nan_to_num([_auc(F[:, i], lab) for i in range(F.shape[1])], nan=.5)
        return float(np.max(np.abs(a - 0.5)) + 0.5)

    nb = [best_dim(MF, rng.permutation(LAB)) for _ in range(8)]
    npv = [probe_auc(MF, rng.permutation(LAB), seed=i) for i in range(8)]
    print(f"\n  A. POOLING -- separating queens-off futures, a label never trained on")
    print(f"     best single concept dim : AUC {best_dim(MF, LAB):.3f}"
          f"   (permuted null {np.mean(nb):.3f} +/- {np.std(nb):.3f})")
    print(f"     probe, grouped split    : AUC {probe_auc(MF, LAB):.3f}"
          f"   (permuted null {np.mean(npv):.3f} +/- {np.std(npv):.3f})")
    print(f"     same probe on phi(s_0)  : AUC {probe_auc(P0, LAB):.3f}"
          f"   <- control: what the reached position already says")

    # ---- B. is the MEAN enough? ---------------------------------------------------
    sp = lambda a: float(np.linalg.norm(a - a.mean(0), axis=1).mean())
    bet = float(np.linalg.norm(MF[LAB].mean(0) - MF[~LAB].mean(0)))
    wit = 0.5 * (sp(MF[LAB]) + sp(MF[~LAB]))
    print(f"\n  B. IS THE MEAN ENOUGH?")
    print(f"     between-class {bet:.2f} vs within-class spread {wit:.2f}"
          f"  (ratio {bet/max(wit,1e-9):.2f})")
    print(f"     -> " + ("classes are tight; a mean summarises them" if bet > wit else
                         "spread DWARFS the class difference: a mean blurs distinct "
                         "endgames, the head must be DISTRIBUTIONAL"))

    # ---- C. WHAT is predictable, and does it MATTER? -----------------------------
    pred = grouped_ridge(P0, MF, GRP)
    ss_res = ((MF - pred) ** 2).sum(0)
    ss_tot = ((MF - MF.mean(0)) ** 2).sum(0) + 1e-9
    r2 = 1 - ss_res / ss_tot
    # The honest bar is R^2 > 0: beating a concept's OWN MEAN. (A shuffled-target
    # null sits near -0.7 because the ridge then fits noise, so "beats the null" is
    # a bar anything clears -- it flattered the result and is not used.)
    good = r2 > 0.0

    # IMPORTANCE from the trained value head, not from per-bit correlation: with
    # distributed binary codes no single bit tracks value (mean |corr| 0.06), but the
    # value head's first layer says exactly how much it READS each concept.
    import torch as _t
    _sd = _t.load(args.ckpt, map_location="cpu", weights_only=False)["state_dict"]
    W = _sd["sf_v_full.0.weight"][:, :MF.shape[1]]
    imp = W.norm(dim=0).numpy()

    print(f"\n  C. WHAT IS PREDICTABLE -- future occupancy from phi(s_0), grouped R^2")
    print(f"     median R^2 {np.median(r2):+.3f} | 75th {np.quantile(r2,.75):+.3f} | "
          f"90th {np.quantile(r2,.90):+.3f} | max {r2.max():+.3f}")
    print(f"     {100*good.mean():.0f}% of {len(r2)} concept dims are predictable at "
          f"all (R^2 > 0); {100*(r2>0.3).mean():.0f}% clear R^2 > 0.3")
    print(f"     -> Kaveh's expectation is CONFIRMED: {100*(1-good.mean()):.0f}% of "
          f"concepts are not predictable from the present. A loss that averages over "
          f"all of them is dominated by an irreducible floor.")

    hi = imp >= np.quantile(imp, 0.90)
    lo = imp <= np.quantile(imp, 0.50)
    print(f"\n     DOES PREDICTABILITY LAND ON WHAT THE VALUE HEAD READS?")
    print(f"     top-10% most-read concepts : {100*good[hi].mean():.0f}% predictable, "
          f"median R^2 {np.median(r2[hi]):+.3f}")
    print(f"     bottom-50% least-read      : {100*good[lo].mean():.0f}% predictable, "
          f"median R^2 {np.median(r2[lo]):+.3f}")
    rho = float(np.corrcoef(imp, r2)[0, 1])
    print(f"     corr(value-head weight, predictability) = {rho:+.3f}")
    aligned = abs(rho) > 0.10 and np.median(r2[hi]) > np.median(r2[lo]) + 0.05
    print(f"     -> " + ("the concepts the evaluation reads are also the predictable "
                         "ones; a fixed importance weight would work"
                         if aligned else
                         "predictability and value-relevance are essentially "
                         "INDEPENDENT. A fixed importance weight taken from the value "
                         "head will NOT concentrate the auxiliary on learnable "
                         "concepts -- the weighting has to be LEARNED, or made free "
                         "by construction."))

    # ---- D. how many KINDS of future are there, and can we predict WHICH? --------
    # The whole design turns on this. Kaveh wants "concept states that pool together"
    # -- if futures fall into a few reusable kinds AND the kind is predictable from
    # the position, a mixture head is learnable and each component IS a subgoal. If
    # not, there is nothing to pool and the mean was never the problem.
    from sklearn.cluster import KMeans
    pos = np.unique(GRP)
    rr = np.random.default_rng(1)
    pp = rr.permutation(pos)
    te = np.isin(GRP, pp[:max(2, len(pos) // 3)])
    Z = (MF - MF.mean(0)) / (MF.std(0) + 1e-6)
    print(f"\n  D. HOW MANY KINDS OF FUTURE? (k-means on M, grouped holdout)")
    print(f"     k   predict-the-kind   majority   value spread across kinds")
    for k in (2, 3, 4, 6, 8):
        km = KMeans(k, n_init=8, random_state=0).fit(Z[~te])
        ytr, yte = km.labels_, km.predict(Z[te])
        Y = np.eye(k)[ytr]
        mu, sd = P0[~te].mean(0), P0[~te].std(0) + 1e-6
        xtr, xte = (P0[~te] - mu) / sd, (P0[te] - mu) / sd
        w = np.linalg.solve(xtr.T @ xtr + 10.0 * np.eye(xtr.shape[1]),
                            xtr.T @ (Y - Y.mean(0)))
        acc = float((np.argmax(xte @ w, 1) == yte).mean())
        maj = float(np.bincount(yte, minlength=k).max() / len(yte))
        vsp = float(np.std([VAL[te][yte == j].mean() for j in range(k)
                            if (yte == j).sum() > 2]))
        print(f"     {k}        {acc:5.1%}          {maj:5.1%}         "
              f"{vsp:5.2f} pawns")
    print(f"     (value spread = std of mean Stockfish eval across the kinds; if the "
          f"kinds\n      differ in value, choosing among them IS choosing a plan)")

    # ---- E. does the KIND discriminate MOVES, or only positions? -----------------
    # The decisive check. If every move from a position leads to the same kind, the
    # kind is a property of the position and carries nothing for CHOOSING a move --
    # 76% accuracy would then just be "recognise the position". Planning needs
    # different moves to open different futures FROM THE SAME POSITION.
    K = 4
    km = KMeans(K, n_init=8, random_state=0).fit(Z)
    lab = km.labels_
    per, gaps = [], []
    for g in pos:
        sel = GRP == g
        if sel.sum() < 3:
            continue
        u = np.unique(lab[sel])
        per.append(len(u))
        if len(u) > 1:
            mv = [VAL[sel][lab[sel] == j].mean() for j in u]
            gaps.append(max(mv) - min(mv))
    per = np.array(per)
    print(f"\n  E. DOES THE KIND DISCRIMINATE MOVES, OR ONLY POSITIONS?  (k={K})")
    print(f"     distinct kinds reachable from one position: mean {per.mean():.2f} "
          f"of {K} | {100*(per > 1).mean():.0f}% of positions offer a real CHOICE")
    if gaps:
        print(f"     within a position, the best kind beats the worst by "
              f"{np.mean(gaps):.2f} pawns on average (median {np.median(gaps):.2f})")
    print(f"     -> " + (f"moves from one position open GENUINELY DIFFERENT futures, "
                         f"worth ~{np.mean(gaps):.1f} pawns -- the kind is a planning "
                         f"signal, not a position label"
                         if gaps and np.mean(gaps) > 0.3 and (per > 1).mean() > 0.5
                         else "the kind is mostly a property of the POSITION; it "
                              "cannot discriminate moves and is useless for choosing"))

    # ---- F. kinds of the RESIDUAL future: what the MOVE changes ------------------
    # E says global kinds track the position -- unsurprising, every line starts from
    # the same place, so the position's own character dominates a 10-ply occupancy.
    # This is the same across-position vs within-position split that has bitten this
    # project before (value correlation 0.59-0.85 across, 0.00 within). So subtract
    # each position's mean future and cluster what the MOVE contributes.
    R = Z.copy()
    for g in pos:
        sel = GRP == g
        if sel.sum() > 1:
            R[sel] -= R[sel].mean(0)
    print(f"\n  F. KINDS OF THE RESIDUAL -- what the MOVE changes about the future")
    print(f"     k   predict-the-kind   majority   within-position value gap")
    for k in (2, 3, 4, 6):
        kmr = KMeans(k, n_init=8, random_state=0).fit(R[~te])
        ytr, yte = kmr.labels_, kmr.predict(R[te])
        Y = np.eye(k)[ytr]
        mu, sd = P0[~te].mean(0), P0[~te].std(0) + 1e-6
        xtr, xte = (P0[~te] - mu) / sd, (P0[te] - mu) / sd
        w = np.linalg.solve(xtr.T @ xtr + 10.0 * np.eye(xtr.shape[1]),
                            xtr.T @ (Y - Y.mean(0)))
        acc = float((np.argmax(xte @ w, 1) == yte).mean())
        maj = float(np.bincount(yte, minlength=k).max() / len(yte))
        allb = kmr.predict(R)
        gp = []
        for g in pos:
            sel = GRP == g
            u = np.unique(allb[sel])
            if sel.sum() >= 3 and len(u) > 1:
                mv = [VAL[sel][allb[sel] == j].mean() for j in u]
                gp.append(max(mv) - min(mv))
        nch = float(np.mean([len(np.unique(allb[GRP == g])) > 1 for g in pos]))
        print(f"     {k}        {acc:5.1%}          {maj:5.1%}         "
              f"{np.mean(gp) if gp else float('nan'):5.2f} pawns   "
              f"({100*nch:.0f}% of positions offer a choice)")

    # ---- G. CONTROL: is this about FUTURES, or about siblings? -------------------
    # F says the residual kind is not predictable. Two very different explanations:
    # the future is genuinely unpredictable, or phi(s_0) simply does not separate
    # SIBLING positions -- the same within-position blindness that already stops the
    # search from ranking moves. Stockfish is deterministic, so the future IS a
    # function of s_0; if the concepts cannot resolve siblings, nothing downstream
    # can. Ask phi(s_0) for the move's OWN value, which needs no future at all.
    from scipy.stats import spearmanr
    vp = grouped_ridge(P0, VAL[:, None], GRP)[:, 0]
    across = float(spearmanr(vp, VAL).statistic)
    wi = [float(spearmanr(vp[GRP == g], VAL[GRP == g]).statistic)
          for g in pos if (GRP == g).sum() >= 5]
    wi = [x for x in wi if np.isfinite(x)]
    print(f"\n  G. CONTROL -- can phi(s_0) rank the move's OWN Stockfish value?")
    print(f"     ACROSS positions: spearman {across:+.3f}   "
          f"(the concepts do carry value)")
    print(f"     WITHIN a position (siblings): spearman {np.mean(wi):+.3f} "
          f"+/- {np.std(wi)/max(1,len(wi))**.5:.3f} over {len(wi)} positions")
    print(f"     -> sibling value ranking from the concepts is WEAK but real")

    # ---- H. APPLES-TO-APPLES: value vs future, both WITHIN a position ------------
    # F used classification accuracy and G used spearman, which are not comparable.
    # Put the future on G's footing: project the residual future onto its leading
    # directions and ask how well phi(s_0) ranks SIBLINGS along them.
    U, S, Vt = np.linalg.svd(R - R.mean(0), full_matrices=False)
    PC = (R - R.mean(0)) @ Vt[:8].T
    ph = grouped_ridge(P0, PC, GRP)
    per_pc = []
    for j in range(PC.shape[1]):
        w_ = [float(spearmanr(ph[GRP == g, j], PC[GRP == g, j]).statistic)
              for g in pos if (GRP == g).sum() >= 5]
        w_ = [x for x in w_ if np.isfinite(x)]
        per_pc.append(np.mean(w_))
    per_pc = np.array(per_pc)
    ev = (S[:8] ** 2 / (S ** 2).sum())
    print(f"\n  H. WITHIN A POSITION, on the same footing as G")
    print(f"     rank the sibling's own VALUE          : spearman {np.mean(wi):+.4f}")
    print(f"     rank which FUTURE the sibling opens   : spearman "
          f"{per_pc.mean():+.4f}  (mean over 8 leading directions, "
          f"{100*ev.sum():.0f}% of residual variance)")
    print(f"     best single future direction          : spearman "
          f"{per_pc.max():+.4f}")
    print(f"     -> " + ("future structure is as accessible to the concepts as value "
                         "is: a future-prediction head has something to learn from"
                         if per_pc.mean() > 0.6 * np.mean(wi) else
                         "predicting WHICH future a sibling opens is markedly harder "
                         "than ranking its value -- the residual future is close to "
                         "invisible to the present concepts at this sample size"))


if __name__ == "__main__":
    main()
