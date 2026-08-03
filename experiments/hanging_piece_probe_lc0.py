#!/usr/bin/env python
"""experiments/hanging_piece_probe_lc0.py -- Kaveh's question: does the RAW frozen lc0
trunk (the community Leela distillate `catspace/encoder/field.py::ReachabilityField`
wraps, T1-256x10, locked-decision M1 trunk -- much bigger/stronger pretraining than
our own JEPA T1) contain hanging-piece information, using the SAME real-outcome
ground truth as experiments/hanging_piece_probe.py (SF committor swing across an
actual played capture -- no synthetic edits, no attacker/defender-count heuristic).

Only the encoder differs from hanging_piece_probe.py; case/control construction is
copied verbatim (kept duplicated rather than shared, matching this repo's existing
per-script convention for engine/eval boilerplate).

IMPORTANT perspective fix: lc0's input planes are POV-oriented -- for Black to move,
LczeroBoard.to_config_tensor() flips the board vertically (rank axis only, files
unchanged: `config_tensor if us==WHITE else config_tensor.flip(1)`) so that row 0 is
always the mover's own back rank. The trunk's per-square token order follows this same
(row, col) spatial layout (field.py's own `t.reshape(B,64,C)` assumes it). So the
token index for an absolute chess.square depends on whose move it is --
pov_row = rank if white-to-move else (7 - rank), pov_col = file (unchanged),
flat_idx = pov_row*8 + pov_col. Getting this wrong would silently misalign every
Black-to-move example's target-square feature with a DIFFERENT square's token --
this fix is load-bearing, not cosmetic.

READOUT only (no attribution pass): the ONNX-derived trunk's autograd behavior under
captum's Integrated Gradients isn't validated here, and the readout is the direct
answer to "does it contain the information at all."

Usage:
  experiments/hanging_piece_probe_lc0.py --hang-thr 0.25 --fair-thr 0.05
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
from experiments.hanging_piece_probe import captured_square  # noqa: E402


def pov_square_index(sq: int, us: bool) -> int:
    """Absolute chess.square -> flat index into the trunk's (64,) POV-oriented
    per-square token sequence. See module docstring -- files never flip, ranks flip
    only when Black is to move."""
    f = chess.square_file(sq)
    r = chess.square_rank(sq)
    pr = r if us == chess.WHITE else 7 - r
    return pr * 8 + f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labeled", default="data/derived/transition_data_labeled.npz")
    ap.add_argument("--onnx", default="data/engines/lc0/t1-256x10.onnx")
    ap.add_argument("--balanced-band", type=float, default=1.0)
    ap.add_argument("--hang-thr", type=float, default=0.25)
    ap.add_argument("--fair-thr", type=float, default=0.05)
    ap.add_argument("--max-ctrl", type=int, default=3000)
    ap.add_argument("--test-frac", type=float, default=0.25)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="artifacts/experiments/hanging_piece_probe_lc0")
    args = ap.parse_args()
    t0 = time.time()
    rng = np.random.default_rng(args.seed)

    from catspace.research.tools.training_infra.train.scaffold import resolve_device
    from lczerolens import LczeroModel, LczeroBoard
    dev = resolve_device("auto")
    trunk = LczeroModel.from_onnx_path(args.onnx).float().to(dev).eval()
    names = [n for n, _ in trunk.named_modules()
             if n and all(k not in n.lower() for k in ("policy", "value", "wdl", "output", "mlh"))]
    hook_name = names[-1]
    feat_box = {}
    dict(trunk.named_modules())[hook_name].register_forward_hook(
        lambda mo, i, o: feat_box.__setitem__("t", o))
    for p in trunk.parameters():
        p.requires_grad_(False)
    print(f"lc0 trunk loaded: {args.onnx}, hook={hook_name}")

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
            continue
        white_to_move = board.turn
        gain = (ca[i] - cb[i]) if white_to_move else (cb[i] - ca[i])
        if gain >= args.hang_thr:
            label = 1
        elif abs(gain) <= args.fair_thr:
            label = 0
        else:
            continue
        rows.append((label, fen[i], tgt_sq, int(game[i])))

    n_case_raw = sum(1 for r in rows if r[0] == 1)
    n_ctrl_raw = sum(1 for r in rows if r[0] == 0)
    if n_ctrl_raw > args.max_ctrl:
        case_rows = [r for r in rows if r[0] == 1]
        ctrl_rows = [r for r in rows if r[0] == 0]
        idx = rng.choice(len(ctrl_rows), args.max_ctrl, replace=False)
        rows = case_rows + [ctrl_rows[i] for i in idx]
    labels = np.array([r[0] for r in rows])
    groups = np.array([r[3] for r in rows])
    print(f"  {len(rows)} captures used (of {n_case_raw} hanging / {n_ctrl_raw} defended-fair raw) | "
          f"{int(labels.sum())} hanging (case) | {int((labels == 0).sum())} defended/fair (control)")
    if len(rows) < 20:
        print("VERDICT hanging-piece-probe-lc0: ABORT -- too few case/control examples")
        return

    # ---- encode every selected PRE-capture board with the lc0 trunk, batched ----
    feats = []
    for i0 in range(0, len(rows), args.batch):
        chunk = rows[i0:i0 + args.batch]
        boards = [chess.Board(f) for _, f, _, _ in chunk]
        lc_boards = [LczeroBoard(b.fen()) for b in boards]
        x = torch.stack([b.to_input_tensor() for b in lc_boards]).float().to(dev)
        with torch.no_grad():
            trunk(x)
            t = feat_box["t"]                      # (B*64, C)
            C = t.shape[-1]
            t = t.reshape(len(chunk), 64, C)        # (B, 64, C), POV-oriented per board
        t = t.cpu().numpy()
        g = t.mean(axis=1)                          # pooled "global" feature
        for j, (label, f, tgt_sq, gi) in enumerate(chunk):
            us = boards[j].turn
            k = pov_square_index(tgt_sq, us)
            feats.append(np.concatenate([g[j], t[j, k]]))
        if (i0 // args.batch) % 10 == 0:
            print(f"  encoded {min(i0 + args.batch, len(rows))}/{len(rows)}")
    feats = np.array(feats)

    # ---- READOUT: linear probe + MLP w/ Hewitt&Liang control task, game-level held-out split ----
    from experiments.probe_readout import linear_readout, mlp_readout_with_control
    from sklearn.metrics import roc_curve
    gs = np.unique(groups)
    te_g = set(rng.choice(gs, max(1, int(len(gs) * args.test_frac)), replace=False).tolist())
    te = np.array([g in te_g for g in groups]); tr = ~te

    lin = linear_readout(feats, labels, tr, te, rng, args.n_boot)
    acc, auc, lo_auc, hi_auc, p_te = lin["acc"], lin["auc"], lin["lo"], lin["hi"], lin["p_te"]
    print(f"VERDICT hanging-piece-readout-lc0: held-out acc {acc:.3f} | AUC {auc:.3f} "
          f"[{lo_auc:.3f},{hi_auc:.3f}] | n_train {tr.sum()} n_test {te.sum()} | "
          f"trunk = frozen lc0 T1-256x10 distillate (data/engines/lc0/t1-256x10.onnx), "
          f"feats = [mean-pooled-square-tokens ++ POV-corrected target-square token] | "
          f"{'PASS' if lo_auc > 0.5 else 'FAIL'} (readout beats chance on held-out games)")

    mlp = mlp_readout_with_control(feats, labels, tr, te, rng, hidden=32, n_boot=args.n_boot, seed=args.seed)
    print(f"VERDICT hanging-piece-readout-lc0-mlp: AUC {mlp['auc_real']:.3f} "
          f"[{mlp['lo_real']:.3f},{mlp['hi_real']:.3f}] | control-task AUC {mlp['auc_ctrl']:.3f} "
          f"[{mlp['lo_ctrl']:.3f},{mlp['hi_ctrl']:.3f}] | selectivity {mlp['selectivity']:+.3f} | "
          f"{'PASS' if (mlp['lo_real'] > 0.5 and mlp['selectivity'] > 0.1) else 'FAIL'} "
          f"(MLP beats chance AND beats its own control task -- Hewitt & Liang 2019)")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 4.5))
    if len(np.unique(labels[te])) > 1:
        fpr, tpr, _ = roc_curve(labels[te], p_te)
        ax.plot(fpr, tpr, color="#c63")
        ax.plot([0, 1], [0, 1], "k--", lw=0.7)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title(f"lc0 trunk readout ROC, AUC {auc:.3f} [{lo_auc:.2f},{hi_auc:.2f}] (n_test={te.sum()})")
    fig.tight_layout()
    Path("artifacts/experiments").mkdir(exist_ok=True, parents=True)
    Path("docs/figures").mkdir(exist_ok=True, parents=True)
    out_png = Path(f"{args.out}.png")
    fig.savefig(out_png, dpi=130)
    fig.savefig(f"docs/figures/{Path(args.out).name}.png", dpi=130)
    np.savez(f"{args.out}.npz", labels=labels, groups=groups, acc=acc, auc=auc)
    print(f"wrote {out_png} + docs/figures/{Path(args.out).name}.png + {args.out}.npz")
    print(f"done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
