#!/usr/bin/env python
"""experiments/defender_circuit_probe.py -- does the JEPA T1 trunk have a legible
"defender circuit"? Hypothesis (Kaveh 2026-07-31): a hanging piece's per-square token
should encode capturability, and adding a second defender should causally shift that
token toward "not capturable" -- specifically through the defender's own token, not
just "a piece was added somewhere."

External label only (TESTING.md 2.9: hand-coded concepts are diagnostics/labels, never
fed into the model): is_hanging(sq) = attacked by opponent AND zero same-color
defenders, via python-chess board.attackers() counts (a count-based SEE-lite, not a
full SEE -- noted as a simplification, not a claim of exchange-value accuracy).

Three parts, in order:
  1. READOUT: train a frozen logistic-regression probe on the trunk's per-square
     tokens (train split) to predict is_hanging; report held-out accuracy/AUC. Per
     Kantamneni et al. 2025 ("Are Sparse Autoencoders Useful?"), a plain supervised
     probe -- not an SAE -- is the right tool for a targeted classification question
     like this one.
  2. CAUSAL EDIT: for held-out hanging pieces, construct a minimal-pair "+1 defender"
     edit (place a same-color piece on a square that newly defends the target,
     attacker count unchanged) and a placebo edit (same piece type, same-magnitude
     board change, but it does NOT defend the target). Read the frozen readout's
     logit shift under each; report paired bootstrap CI. This is the causal half --
     activation-patching best practice (Zhang & Nanda 2023): both the corruption and
     the metric are explicit.
  3. ATTRIBUTION: Integrated Gradients (captum) from the baseline embedding to the
     +1-defender embedding, targeting the readout logit at the target square. Reports
     what fraction of cases have the true defender square in the top-1/top-3 by
     attribution mass among the other 63 squares, vs a random-square null.

Usage:
  experiments/defender_circuit_probe.py --n-readout 4000 --n-causal 200
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import chess
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def is_hanging(board: chess.Board, sq: int):
    """-> (hanging: bool, n_attackers: int, n_defenders: int). Count-based, not full SEE."""
    pc = board.piece_map().get(sq)
    if pc is None or pc.piece_type == chess.KING:
        return None
    na = len(board.attackers(not pc.color, sq))
    nd = len(board.attackers(pc.color, sq))
    return (na >= 1 and nd == 0), na, nd


def encode_squares(enc, tok, glob):
    """Replicates JepaEncoder.forward but also returns the 64 per-square tokens
    (pre-pool), ordered by chess square index -- matches `tok`'s ordering."""
    B = tok.shape[0]
    x = enc.piece_emb(tok.long()) + enc.sq_emb.weight[None, :, :]
    g = enc.glob_proj(glob.float())[:, None, :]
    x = torch.cat([enc.cls.expand(B, -1, -1) + g, x], 1)
    y = enc.out(enc.tr(x))
    return y[:, 0], y[:, 1:], x          # phi, per-square tokens, raw embedded input


def encode_squares_from_embed(enc, x):
    """Same as encode_squares but starting from an already-embedded (B,65,d) input --
    the differentiable path IntegratedGradients attributes through."""
    y = enc.out(enc.tr(x))
    return y[:, 0], y[:, 1:]


def try_add_defender(board: chess.Board, target_sq: int, color: bool, na0: int, nd0: int):
    """Try placing a QUEEN (falls back ROOK/BISHOP/KNIGHT) of `color` on an empty
    square s.t. it newly defends target_sq, board stays legal, and n_attackers is
    unchanged (rules out "blocked the attacker" as the mechanism). -> board2 or None."""
    empties = [s for s in chess.SQUARES if board.piece_at(s) is None]
    for pt in (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT):
        for s in empties:
            b2 = board.copy()
            b2.set_piece_at(s, chess.Piece(pt, color))
            if not b2.is_valid():
                continue
            pc = b2.piece_map().get(target_sq)
            if pc is None:
                continue
            na1 = len(b2.attackers(not color, target_sq))
            nd1 = len(b2.attackers(color, target_sq))
            if na1 == na0 and nd1 == nd0 + 1:
                return b2, s, pt
    return None, None, None


def try_placebo(board: chess.Board, target_sq: int, color: bool, na0: int, nd0: int,
                 piece_type: int, rng: np.random.Generator):
    """Same piece type, an empty square that does NOT change attackers/defenders of
    target_sq -- isolates 'a piece was added' from 'a defender was added'."""
    empties = [s for s in chess.SQUARES if board.piece_at(s) is None]
    rng.shuffle(empties)
    for s in empties:
        b2 = board.copy()
        b2.set_piece_at(s, chess.Piece(piece_type, color))
        if not b2.is_valid():
            continue
        na1 = len(b2.attackers(not color, target_sq))
        nd1 = len(b2.attackers(color, target_sq))
        if na1 == na0 and nd1 == nd0:
            return b2, s
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labeled", default="data/derived/transition_data_labeled.npz")
    ap.add_argument("--ckpt", default="artifacts/experiments/jepa_t1_latest.pt")
    ap.add_argument("--n-readout", type=int, default=4000, help="positions for readout train/eval")
    ap.add_argument("--n-causal", type=int, default=200, help="held-out hanging cases for the causal probe")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="artifacts/experiments/defender_circuit_probe")
    args = ap.parse_args()
    t0 = time.time()
    rng = np.random.default_rng(args.seed)

    from catspace.encoder.jepa import JepaT1, tokenize
    from catspace.train.scaffold import resolve_device
    dev = resolve_device("auto")
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    model = JepaT1(**{k: ck["cfg"][k] for k in ("d", "layers", "n_class")}).to(dev)
    model.load_state_dict(ck["state_dict"]); model.eval()
    enc = model.enc
    for p in enc.parameters():
        p.requires_grad_(False)

    d = np.load(args.labeled, allow_pickle=True)
    fens = np.unique(d["fen"])
    rng.shuffle(fens)
    fens = fens[:args.n_readout + args.n_causal * 6]   # oversample for causal-edit yield

    # ---- gather (phi[CLS] ++ per-square token, label) pairs across all non-king
    # pieces. Kaveh's catch: the "who defends me" computation is attention-pooled
    # across the whole board, so it may live in the CLS token rather than staying
    # local to the target square's own token -- concatenate both rather than
    # probing the square token alone. ----
    feats, labels, groups, fen_of = [], [], [], []
    hang_cases = []   # (fen_idx, board, target_sq, color, na0, nd0)
    print(f"scanning {len(fens)} positions ...")
    for gi, fen in enumerate(fens):
        board = chess.Board(fen)
        tok, glob = tokenize(board)
        with torch.no_grad():
            phi, sq_tok, _ = encode_squares(
                enc, torch.as_tensor(tok[None]).to(dev), torch.as_tensor(glob[None]).to(dev))
        phi = phi[0].cpu().numpy()
        sq_tok = sq_tok[0].cpu().numpy()
        for sq, pc in board.piece_map().items():
            if pc.piece_type == chess.KING:
                continue
            h, na, nd = is_hanging(board, sq)
            feats.append(np.concatenate([phi, sq_tok[sq]])); labels.append(int(h)); groups.append(gi)
            if h and len(hang_cases) < args.n_causal * 3:
                hang_cases.append((gi, board, sq, pc.color, na, nd))
    feats = np.array(feats); labels = np.array(labels); groups = np.array(groups)
    print(f"  {len(feats)} (square, label) pairs from {len(fens)} positions | "
          f"{labels.mean():.3f} positive rate | {len(hang_cases)} hanging-piece candidates")

    # ---- 1. READOUT: frozen logistic probe, position-level held-out split ----
    gs = np.unique(groups)
    te_g = set(rng.choice(gs, max(1, int(len(gs) * 0.25)), replace=False).tolist())
    te = np.array([g in te_g for g in groups]); tr = ~te
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    clf = LogisticRegression(max_iter=3000, class_weight="balanced").fit(feats[tr], labels[tr])
    p_te = clf.predict_proba(feats[te])[:, 1]
    acc = clf.score(feats[te], labels[te])
    auc = roc_auc_score(labels[te], p_te) if len(np.unique(labels[te])) > 1 else float("nan")
    print(f"VERDICT defender-circuit-readout: held-out acc {acc:.3f} | AUC {auc:.3f} | "
          f"n_train {tr.sum()} n_test {te.sum()} | feats = [phi(CLS) ++ sq_token] | "
          f"probe LogisticRegression (not SAE, per Kantamneni et al. 2025)")

    # torch-mirrored linear readout, for IG's differentiable path
    W = torch.tensor(clf.coef_[0], dtype=torch.float32, device=dev)
    b0_ = torch.tensor(clf.intercept_[0], dtype=torch.float32, device=dev)

    def readout_logit(sq_tok_vec):
        return sq_tok_vec @ W + b0_

    # ---- 2/3. CAUSAL EDIT + ATTRIBUTION on held-out (test-group) hanging cases ----
    from captum.attr import IntegratedGradients
    rng2 = np.random.default_rng(args.seed + 1)
    d_defend, d_placebo, ig_rank, ig_top1, ig_top3, ig_n_occ = [], [], [], [], [], []
    n_done = 0
    for gi, board, sq, color, na0, nd0 in hang_cases:
        if gi not in te_g:
            continue    # causal probe only on held-out positions, no train leakage
        b_def, def_sq, pt = try_add_defender(board, sq, color, na0, nd0)
        if b_def is None:
            continue
        b_plc, plc_sq = try_placebo(board, sq, color, na0, nd0, pt, rng2)
        if b_plc is None:
            continue

        def embed(b):
            t, g = tokenize(b)
            _, _, x = encode_squares(enc, torch.as_tensor(t[None]).to(dev),
                                      torch.as_tensor(g[None]).to(dev))
            return x

        def feat(phi_, sqt, sq_):
            return torch.cat([phi_[0], sqt[0, sq_]], -1)

        with torch.no_grad():
            x0, xd, xp = embed(board), embed(b_def), embed(b_plc)
            phi0, sq0 = encode_squares_from_embed(enc, x0)
            phid, sqd = encode_squares_from_embed(enc, xd)
            phip, sqp = encode_squares_from_embed(enc, xp)
            l0 = float(readout_logit(feat(phi0, sq0, sq)))
            ld = float(readout_logit(feat(phid, sqd, sq)))
            lp = float(readout_logit(feat(phip, sqp, sq)))
        d_defend.append(ld - l0); d_placebo.append(lp - l0)

        # Empty-board baseline (not B0->B_defend, which differs at exactly one square
        # and would make IG's completeness axiom trivially dump 100% attribution
        # there regardless of what the model does -- caught in review, not a finding).
        empty_board = chess.Board(None); empty_board.turn = board.turn
        t_e, g_e = tokenize(empty_board)
        _, _, x_empty = encode_squares(enc, torch.as_tensor(t_e[None]).to(dev),
                                        torch.as_tensor(g_e[None]).to(dev))

        def target_logit_fn(x):
            phit, sqt = encode_squares_from_embed(enc, x)
            return readout_logit(torch.cat([phit, sqt[:, sq]], -1))
        ig = IntegratedGradients(target_logit_fn)
        attr = ig.attribute(xd, baselines=x_empty, n_steps=32)[0]  # (65, d)
        mag = attr.norm(dim=-1).detach().cpu().numpy()             # (65,)
        sq_mag = mag[1:]                                            # drop CLS, index = square
        occupied = [s for s in chess.SQUARES if b_def.piece_at(s) is not None and s != sq]
        order = sorted(occupied, key=lambda s: -sq_mag[s])
        rank = order.index(def_sq)                                  # 0 = top attributed square
        n_occ = len(occupied)
        ig_rank.append(rank); ig_top1.append(rank == 0); ig_top3.append(rank < 3)
        ig_n_occ.append(n_occ)

        n_done += 1
        if n_done >= args.n_causal:
            break

    d_defend = np.array(d_defend); d_placebo = np.array(d_placebo)
    ig_rank = np.array(ig_rank); ig_top1 = np.array(ig_top1); ig_top3 = np.array(ig_top3)

    def boot_ci(x, n=args.n_boot):
        if len(x) == 0:
            return float("nan"), float("nan"), float("nan")
        idx = rng.integers(0, len(x), (n, len(x)))
        m = x[idx].mean(1)
        return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))

    md, lo_d, hi_d = boot_ci(d_defend)
    mp, lo_p, hi_p = boot_ci(d_placebo)
    diff = d_defend - d_placebo
    mdiff, lo_diff, hi_diff = boot_ci(diff)
    top1_rate = float(ig_top1.mean()) if len(ig_top1) else float("nan")
    top3_rate = float(ig_top3.mean()) if len(ig_top3) else float("nan")
    ig_n_occ = np.array(ig_n_occ)
    # random-square null, matched per-case to the number of occupied non-target squares
    null_top1 = float(np.mean(1 / ig_n_occ)) if len(ig_n_occ) else float("nan")
    null_top3 = float(np.mean(np.minimum(3, ig_n_occ) / ig_n_occ)) if len(ig_n_occ) else float("nan")

    print(f"VERDICT defender-circuit-causal: n_cases {len(d_defend)} | "
          f"d_logit(+defender) {md:+.3f} [{lo_d:+.3f},{hi_d:+.3f}] | "
          f"d_logit(placebo) {mp:+.3f} [{lo_p:+.3f},{hi_p:+.3f}] | "
          f"diff {mdiff:+.3f} [{lo_diff:+.3f},{hi_diff:+.3f}] | "
          f"{'PASS' if lo_diff > 0 or hi_diff < 0 else 'FAIL'} "
          f"(defend and placebo distinguishable)")
    print(f"VERDICT defender-circuit-attribution: n_cases {len(ig_rank)} | "
          f"defender-square top-1 rate {top1_rate:.3f} (null {null_top1:.3f}) | "
          f"top-3 rate {top3_rate:.3f} (null {null_top3:.3f}) | "
          f"{'PASS' if top3_rate > 3 * null_top3 else 'FAIL'} "
          f"(attribution concentrates on the true defender square)")

    # ---- figure ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax = axes[0]
    ax.bar([0, 1], [md, mp], yerr=[[md - lo_d, mp - lo_p], [hi_d - md, hi_p - mp]],
           color=["#3b6", "#999"], capsize=6)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["+1 defender", "placebo edit"])
    ax.set_ylabel("d(readout logit) vs baseline")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_title(f"causal edit (n={len(d_defend)})")
    ax = axes[1]
    ax.bar([0, 1], [top1_rate, top3_rate], color="#36c")
    ax.bar([0, 1], [null_top1, null_top3], color="none", edgecolor="k",
           linestyle="--", label="random-square null")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["top-1 rate", "top-3 rate"])
    ax.set_ylabel("rate defender sq is highest-attributed")
    ax.legend(fontsize=8)
    ax.set_title(f"IG attribution (n={len(ig_rank)})")
    fig.suptitle("defender-circuit probe: JEPA T1 trunk")
    fig.tight_layout()
    Path("artifacts/experiments").mkdir(exist_ok=True, parents=True)
    Path("docs/figures").mkdir(exist_ok=True, parents=True)
    out_png = Path(f"{args.out}.png")
    fig.savefig(out_png, dpi=130)
    fig.savefig(f"docs/figures/{Path(args.out).name}.png", dpi=130)
    np.savez(f"{args.out}.npz", d_defend=d_defend, d_placebo=d_placebo, ig_rank=ig_rank,
             readout_acc=acc, readout_auc=auc)
    print(f"wrote {out_png} + docs/figures/{Path(args.out).name}.png + {args.out}.npz")
    print(f"done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
