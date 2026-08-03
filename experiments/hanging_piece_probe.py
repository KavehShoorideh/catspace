#!/usr/bin/env python
"""experiments/hanging_piece_probe.py -- v2 of the defender-circuit probe
(supersedes experiments/defender_circuit_probe.py's case/control definition; see
JOURNAL 2026-07-31). Kaveh's fix: the v1 count-based "attackers>0, defenders==0"
hanging label was wrong (a totally undefended piece isn't hanging if a bigger threat
elsewhere means it can't actually be captured), and the v1 synthetic "+1 defender"
board edit was confounded (adding a queen anywhere can create its own threats,
contaminating the control).

v2 ground truth: a real capture, played in a real game, from a roughly-balanced
position (|committor_before - 0.5| < 0.15), where Stockfish's own win-fraction swing
across that exact move tells us what actually happened --
  HANGING   (case):    the capture swung win-probability sharply to the capturer
                        (gain >= --hang-thr) -- the piece really was free.
  DEFENDED  (control):  the capture happened but win-probability barely moved
                        (|gain| <= --fair-thr) -- a fair trade / real recapture
                        existed. This is Kaveh's requested control: "a capture is
                        made but the evaluation does not change."
No synthetic board edits anywhere. `committor_before`/`committor_after` are already
Stockfish-computed (depth 12, WDL win-fraction) by experiments/sf_label_transitions.py
-- reused as-is, no new engine calls.

The model only ever sees the PRE-capture board (before the capture is played) --
this asks "can the trunk tell, just from looking at the position, which pieces are
about to be lost for free vs safely traded."

Two parts:
  1. READOUT: frozen logistic probe on [phi(CLS) ++ target-square token], game-level
     held-out split (no game appears in both train and test).
  2. ATTRIBUTION (defended cases only, where a real defender is identifiable): after
     the capture, whoever could recapture on the target square (from the pre-move
     board's own pieces, since the defender hasn't moved yet) is the REAL defender.
     Integrated Gradients (empty-board baseline) ranks how much each occupied square
     contributes to the readout's target-square logit; report how often the real
     defender's square is top-1/top-3 vs a random-square null.

Usage:
  experiments/hanging_piece_probe.py --hang-thr 0.25 --fair-thr 0.05
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
from experiments.defender_circuit_probe import encode_squares, encode_squares_from_embed  # noqa: E402


def captured_square(board: chess.Board, move: chess.Move, board_after: chess.Board):
    """The square whose enemy piece disappears because of this capture. Not always
    move.to_square (en passant: the captured pawn sits one rank off to_square) --
    the general fact: a capture is defined by PIECE COUNT dropping by exactly one
    (generic -- not "does this move flag say capture", which is one more layer than
    needed) -- then, given a capture happened, the missing piece's square is
    "held an enemy piece before, and after the move either it's empty or the mover's
    own piece sits there instead." True for every capture kind, no branch on move
    type (en passant, promotion-capture, ...) needed."""
    before = board.piece_map()
    after = board_after.piece_map()
    if len(after) != len(before) - 1:
        return None    # piece count unchanged (or moved 2, e.g. castling) -> no capture
    for sq, pc in before.items():
        if sq == move.from_square or pc.color == board.turn:
            continue
        if sq not in after or after[sq].color == board.turn:
            return sq
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labeled", default="data/derived/transition_data_labeled.npz")
    ap.add_argument("--ckpt", default="artifacts/experiments/jepa_t1_latest.pt")
    ap.add_argument("--balanced-band", type=float, default=1.0,
                     help="|committor_before-0.5| < this; default 1.0 = no filter "
                          "(Kaveh 2026-07-31: doesn't matter where it starts, only "
                          "whether it suddenly drops toward the attacker)")
    ap.add_argument("--hang-thr", type=float, default=0.25, help="mover-POV committor gain >= this -> hanging")
    ap.add_argument("--fair-thr", type=float, default=0.05, help="|gain| <= this -> defended/fair")
    ap.add_argument("--max-ctrl", type=int, default=3000, help="subsample the (much larger) control class for compute")
    ap.add_argument("--test-frac", type=float, default=0.25)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="artifacts/experiments/hanging_piece_probe")
    args = ap.parse_args()
    t0 = time.time()
    rng = np.random.default_rng(args.seed)

    from catspace.research.components.encoder.approaches.jepa_tokenizer.src.jepa import JepaT1, tokenize
    from catspace.research.tools.training_infra.train.scaffold import resolve_device
    dev = resolve_device("auto")
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    model = JepaT1(**{k: ck["cfg"][k] for k in ("d", "layers", "n_class")}).to(dev)
    model.load_state_dict(ck["state_dict"]); model.eval()
    enc = model.enc
    for p in enc.parameters():
        p.requires_grad_(False)

    d = np.load(args.labeled, allow_pickle=True)
    fen, mv_uci, cb, ca = d["fen"], d["move"], d["committor_before"], d["committor_after"]
    game = d["game"]
    n = len(fen)

    print(f"scanning {n} real-game transitions for captures with a clean win-prob swing "
          f"(balanced_band={args.balanced_band}) ...")
    rows = []   # (label, fen, target_sq, game, gain, board, move, board_after)
    for i in range(n):
        if abs(cb[i] - 0.5) >= args.balanced_band:
            continue
        board = chess.Board(fen[i])
        mv = chess.Move.from_uci(mv_uci[i])
        board2 = board.copy(); board2.push(mv)
        tgt_sq = captured_square(board, mv, board2)
        if tgt_sq is None:
            continue    # piece count didn't drop by one -> not a capture
        white_to_move = board.turn
        gain = (ca[i] - cb[i]) if white_to_move else (cb[i] - ca[i])   # mover-POV committor swing
        if gain >= args.hang_thr:
            label = 1
        elif abs(gain) <= args.fair_thr:
            label = 0
        else:
            continue
        rows.append((label, fen[i], tgt_sq, int(game[i]), float(gain), board, mv, board2))

    n_case_raw = sum(1 for r in rows if r[0] == 1)
    n_ctrl_raw = sum(1 for r in rows if r[0] == 0)
    if n_ctrl_raw > args.max_ctrl:
        case_rows = [r for r in rows if r[0] == 1]
        ctrl_rows = [r for r in rows if r[0] == 0]
        idx = rng.choice(len(ctrl_rows), args.max_ctrl, replace=False)
        rows = case_rows + [ctrl_rows[i] for i in idx]

    labels = np.array([r[0] for r in rows])
    groups = np.array([r[3] for r in rows])
    print(f"  {len(rows)} captures used (of {n_case_raw} hanging / {n_ctrl_raw} defended-fair raw, "
          f"control subsampled to --max-ctrl {args.max_ctrl}) | {int(labels.sum())} hanging (case) | "
          f"{int((labels == 0).sum())} defended/fair (control)")
    if len(rows) < 20:
        print("VERDICT hanging-piece-probe: ABORT -- too few case/control examples to proceed")
        return

    # ---- encode every selected PRE-capture board ----
    feats = []
    defender_true_sq = []   # for label==0 rows: the real recapturing piece's square in P, or -1
    for label, f, tgt_sq, gi, gain, board, mv, board2 in rows:
        tok, glob = tokenize(board)
        with torch.no_grad():
            phi, sq_tok, _ = encode_squares(
                enc, torch.as_tensor(tok[None]).to(dev), torch.as_tensor(glob[None]).to(dev))
        feats.append(np.concatenate([phi[0].cpu().numpy(), sq_tok[0, tgt_sq].cpu().numpy()]))
        if label == 0:
            victim_color = board.piece_at(tgt_sq).color
            # pieces that could recapture, evaluated post-capture; they hadn't moved,
            # so their square is unchanged and still valid to reference in P.
            defenders = list(board2.attackers(victim_color, tgt_sq))
            defender_true_sq.append(defenders[0] if defenders else -1)
        else:
            defender_true_sq.append(-1)
    feats = np.array(feats)
    defender_true_sq = np.array(defender_true_sq)

    # ---- 1. READOUT: linear probe (for the attribution stage below) + MLP w/
    # Hewitt&Liang control task (experiments/probe_readout.py), game-level held-out split ----
    from experiments.probe_readout import linear_readout, mlp_readout_with_control
    gs = np.unique(groups)
    te_g = set(rng.choice(gs, max(1, int(len(gs) * args.test_frac)), replace=False).tolist())
    te = np.array([g in te_g for g in groups]); tr = ~te

    lin = linear_readout(feats, labels, tr, te, rng, args.n_boot)
    acc, auc, lo_auc, hi_auc, p_te, clf = lin["acc"], lin["auc"], lin["lo"], lin["hi"], lin["p_te"], lin["clf"]
    print(f"VERDICT hanging-piece-readout: held-out acc {acc:.3f} | AUC {auc:.3f} "
          f"[{lo_auc:.3f},{hi_auc:.3f}] | n_train {tr.sum()} n_test {te.sum()} | "
          f"ground truth = real SF committor swing across a played capture (not "
          f"attacker/defender counts, not a synthetic edit) | "
          f"{'PASS' if lo_auc > 0.5 else 'FAIL'} (readout beats chance on held-out games)")

    mlp = mlp_readout_with_control(feats, labels, tr, te, rng, hidden=32, n_boot=args.n_boot, seed=args.seed)
    print(f"VERDICT hanging-piece-readout-mlp: AUC {mlp['auc_real']:.3f} "
          f"[{mlp['lo_real']:.3f},{mlp['hi_real']:.3f}] | control-task AUC {mlp['auc_ctrl']:.3f} "
          f"[{mlp['lo_ctrl']:.3f},{mlp['hi_ctrl']:.3f}] | selectivity {mlp['selectivity']:+.3f} | "
          f"{'PASS' if (mlp['lo_real'] > 0.5 and mlp['selectivity'] > 0.1) else 'FAIL'} "
          f"(MLP beats chance AND beats its own control task -- Hewitt & Liang 2019)")

    W = torch.tensor(clf.coef_[0], dtype=torch.float32, device=dev)
    b0_ = torch.tensor(clf.intercept_[0], dtype=torch.float32, device=dev)

    def readout_logit(vec):
        return vec @ W + b0_

    # ---- 2. ATTRIBUTION on held-out DEFENDED (label==0) cases with a known real defender ----
    from captum.attr import IntegratedGradients
    ig_rank, ig_top1, ig_top3, ig_n_occ = [], [], [], []
    for idx, (label, f, tgt_sq, gi, gain, board, mv, board2) in enumerate(rows):
        if label != 0 or gi not in te_g or defender_true_sq[idx] < 0:
            continue
        tok, glob = tokenize(board)
        with torch.no_grad():
            _, _, x = encode_squares(enc, torch.as_tensor(tok[None]).to(dev),
                                      torch.as_tensor(glob[None]).to(dev))
        empty_board = chess.Board(None); empty_board.turn = board.turn
        t_e, g_e = tokenize(empty_board)
        with torch.no_grad():
            _, _, x_empty = encode_squares(enc, torch.as_tensor(t_e[None]).to(dev),
                                            torch.as_tensor(g_e[None]).to(dev))

        def target_logit_fn(xx, tgt_sq=tgt_sq):
            phit, sqt = encode_squares_from_embed(enc, xx)
            return readout_logit(torch.cat([phit, sqt[:, tgt_sq]], -1))

        ig = IntegratedGradients(target_logit_fn)
        attr = ig.attribute(x, baselines=x_empty, n_steps=32)[0]
        mag = attr.norm(dim=-1).detach().cpu().numpy()
        sq_mag = mag[1:]
        occupied = [s for s in chess.SQUARES if board.piece_at(s) is not None and s != tgt_sq]
        if len(occupied) < 2:
            continue
        order = sorted(occupied, key=lambda s: -sq_mag[s])
        def_sq = int(defender_true_sq[idx])
        if def_sq not in order:
            continue
        rank = order.index(def_sq)
        ig_rank.append(rank); ig_top1.append(rank == 0); ig_top3.append(rank < 3)
        ig_n_occ.append(len(occupied))

    ig_n_occ = np.array(ig_n_occ)
    top1_rate = float(np.mean(ig_top1)) if ig_top1 else float("nan")
    top3_rate = float(np.mean(ig_top3)) if ig_top3 else float("nan")
    null_top1 = float(np.mean(1 / ig_n_occ)) if len(ig_n_occ) else float("nan")
    null_top3 = float(np.mean(np.minimum(3, ig_n_occ) / ig_n_occ)) if len(ig_n_occ) else float("nan")
    print(f"VERDICT hanging-piece-attribution: n_cases {len(ig_rank)} | "
          f"real-defender-square top-1 rate {top1_rate:.3f} (null {null_top1:.3f}) | "
          f"top-3 rate {top3_rate:.3f} (null {null_top3:.3f}) | "
          f"{'PASS' if (not np.isnan(top3_rate) and top3_rate > 3 * null_top3) else 'FAIL'} "
          f"(attribution concentrates on the real recapturing piece's square)")

    # ---- figure ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax = axes[0]
    from sklearn.metrics import roc_curve
    if len(np.unique(labels[te])) > 1:
        fpr, tpr, _ = roc_curve(labels[te], p_te)
        ax.plot(fpr, tpr, color="#36c")
        ax.plot([0, 1], [0, 1], "k--", lw=0.7)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title(f"readout ROC, AUC {auc:.3f} [{lo_auc:.2f},{hi_auc:.2f}] (n_test={te.sum()})")
    ax = axes[1]
    ax.bar([0, 1], [top1_rate, top3_rate], color="#3b6")
    ax.bar([0, 1], [null_top1, null_top3], color="none", edgecolor="k",
           linestyle="--", label="random-square null")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["top-1 rate", "top-3 rate"])
    ax.set_ylabel("rate real defender sq is highest-attributed")
    ax.legend(fontsize=8)
    ax.set_title(f"IG attribution, defended cases (n={len(ig_rank)})")
    fig.suptitle("hanging-piece probe v2: real SF committor-swing ground truth, no synthetic edits")
    fig.tight_layout()
    Path("artifacts/experiments").mkdir(exist_ok=True, parents=True)
    Path("docs/figures").mkdir(exist_ok=True, parents=True)
    out_png = Path(f"{args.out}.png")
    fig.savefig(out_png, dpi=130)
    fig.savefig(f"docs/figures/{Path(args.out).name}.png", dpi=130)
    np.savez(f"{args.out}.npz", labels=labels, groups=groups, acc=acc, auc=auc,
             ig_rank=np.array(ig_rank))
    print(f"wrote {out_png} + docs/figures/{Path(args.out).name}.png + {args.out}.npz")
    print(f"done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
