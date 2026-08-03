#!/usr/bin/env python
"""experiments/layer_sweep_probe.py -- per-layer readout sweep + a POSITIVE CONTROL,
following McGrath et al. 2022 (PNAS, AlphaZero concept probing)'s "where" axis: a
concept can be computed mid-network and discarded before the final output/pooling, so
probing only the last layer (as hanging_piece_probe.py / _lc0.py did) can miss it.

POSITIVE CONTROL (Kaveh 2026-07-31): before trusting any null on the real hanging-
piece question, verify the harness (features, POV mapping, split, probe) can detect a
concept we KNOW must be there. Label: is the captured piece a MAJOR piece (rook or
queen) vs minor/pawn -- this is piece IDENTITY, directly present in the input token
(piece_emb lookup for JEPA, a literal one-hot plane for lc0) -- should be linearly
decodable at ~1.0 AUC at every layer, including layer 0. If it isn't, the harness
itself is broken and any null on hanging-ness is uninterpretable. Same rows, same
features, same split as the real task -- only the label differs, so this is a clean
apples-to-apples check of the measurement instrument, not a different experiment.

Trunks (--trunk):
  jepa      : our JEPA T1 (6 encoder layers, 256-dim) -- layers 0(embed)..6(post-LN)
  lc0-small : frozen lc0 T1-256x10 distillate (10 layers, 256-dim) -- what
              hanging_piece_probe_lc0.py used
  lc0-big   : frozen lc0 T1-512x15x8h distillate (15 layers, 512-dim, GPT-2-scale --
              matches Jenner et al. 2024's net size, unlike t1-256x10) -- converted
              from data/engines/lc0/t1-512x15x8h-distilled-swa-3395000.pb.gz via
              `lc0 leela2onnx` (Kaveh 2026-07-31, not previously in onnx form)

Per layer: pooled feature = mean over the 64 square tokens at that layer (trunk- and
depth-agnostic, no CLS dependency so JEPA and lc0 are handled identically past
tokenization). Linear probe only (McGrath's choice, for the same Hewitt&Liang
selectivity reason noted in probe_readout.py) -- MLP-with-control-task already run
separately in hanging_piece_probe{,_lc0}.py and didn't move the final-layer result.

Usage:
  experiments/layer_sweep_probe.py --trunk jepa
  experiments/layer_sweep_probe.py --trunk lc0-small
  experiments/layer_sweep_probe.py --trunk lc0-big
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
from experiments.hanging_piece_probe_lc0 import pov_square_index  # noqa: E402
from experiments.probe_readout import linear_readout, mlp_readout_with_control  # noqa: E402


def build_rows(labeled_path, balanced_band, hang_thr, fair_thr, max_ctrl, rng):
    d = np.load(labeled_path, allow_pickle=True)
    fen, mv_uci, cb, ca, game = d["fen"], d["move"], d["committor_before"], d["committor_after"], d["game"]
    rows = []
    for i in range(len(fen)):
        if abs(cb[i] - 0.5) >= balanced_band:
            continue
        board = chess.Board(fen[i])
        mv = chess.Move.from_uci(mv_uci[i])
        board2 = board.copy(); board2.push(mv)
        tgt_sq = captured_square(board, mv, board2)
        if tgt_sq is None:
            continue
        gain = (ca[i] - cb[i]) if board.turn else (cb[i] - ca[i])
        if gain >= hang_thr:
            label = 1
        elif abs(gain) <= fair_thr:
            label = 0
        else:
            continue
        major = int(board.piece_at(tgt_sq).piece_type in (chess.ROOK, chess.QUEEN))
        rows.append((label, major, fen[i], tgt_sq, int(game[i]), board))
    n_case = sum(1 for r in rows if r[0] == 1)
    ctrl = [r for r in rows if r[0] == 0]
    case = [r for r in rows if r[0] == 1]
    if len(ctrl) > max_ctrl:
        idx = rng.choice(len(ctrl), max_ctrl, replace=False)
        rows = case + [ctrl[i] for i in idx]
    print(f"  {len(rows)} rows ({n_case} hanging / {len(ctrl)} defended-fair raw, "
          f"control capped {max_ctrl}); positive-control label = captured piece is major")
    return rows


# ---------------- trunk-specific per-layer feature extraction ----------------

def jepa_layers(rows, ckpt_path, dev, batch=256):
    from catspace.research.components.encoder.approaches.jepa_tokenizer.src.jepa import JepaT1, tokenize
    ck = torch.load(ckpt_path, map_location=dev, weights_only=False)
    model = JepaT1(**{k: ck["cfg"][k] for k in ("d", "layers", "n_class")}).to(dev)
    model.load_state_dict(ck["state_dict"]); model.eval()
    enc = model.enc
    for p in enc.parameters():
        p.requires_grad_(False)
    n_layers = len(enc.tr.layers)
    layer_names = [f"L{i}" for i in range(n_layers + 2)]   # 0=embed, 1..n=post-block, n+1=post-LN
    feats = {name: [] for name in layer_names}
    with torch.no_grad():
        for i0 in range(0, len(rows), batch):
            chunk = rows[i0:i0 + batch]
            toks, globs = zip(*[tokenize(r[5]) for r in chunk])
            tok = torch.as_tensor(np.stack(toks)).to(dev)
            glob = torch.as_tensor(np.stack(globs)).to(dev)
            B = tok.shape[0]
            x = enc.piece_emb(tok.long()) + enc.sq_emb.weight[None, :, :]
            g = enc.glob_proj(glob.float())[:, None, :]
            x = torch.cat([enc.cls.expand(B, -1, -1) + g, x], 1)
            feats["L0"].append(x[:, 1:].cpu().numpy())
            h = x
            for li, layer in enumerate(enc.tr.layers):
                h = layer(h)
                feats[f"L{li + 1}"].append(h[:, 1:].cpu().numpy())
            h_final = enc.out(h)
            feats[f"L{n_layers + 1}"].append(h_final[:, 1:].cpu().numpy())
    for name in feats:
        feats[name] = np.concatenate(feats[name], 0)   # (N, 64, C)
    tgt_idx = np.array([r[3] for r in rows])            # absolute square == token index (no POV flip for JEPA)
    return feats, tgt_idx, layer_names


def lc0_layers(rows, onnx_path, dev, batch=64):
    from lczerolens import LczeroModel, LczeroBoard
    trunk = LczeroModel.from_onnx_path(onnx_path).float().to(dev).eval()
    for p in trunk.parameters():
        p.requires_grad_(False)
    import re
    names_all = [n for n, _ in trunk.named_modules() if re.fullmatch(r"module\.encoder\d+/ln2", n)]
    names_all = sorted(names_all, key=lambda n: int(n.split("encoder")[1].split("/")[0]))
    n_layers = len(names_all)
    store = {}
    for n in names_all:
        mod = dict(trunk.named_modules())[n]
        mod.register_forward_hook((lambda nm: lambda mo, i, o: store.__setitem__(nm, o))(n))
    print(f"  lc0 trunk: {n_layers} layers hooked ({names_all[0]} .. {names_all[-1]})")

    layer_names = [f"L{i + 1}" for i in range(n_layers)]
    feats = {name: [] for name in layer_names}
    with torch.no_grad():
        for i0 in range(0, len(rows), batch):
            chunk = rows[i0:i0 + batch]
            boards = [r[5] for r in chunk]
            lc_boards = [LczeroBoard(b.fen()) for b in boards]
            x = torch.stack([b.to_input_tensor() for b in lc_boards]).float().to(dev)
            trunk(x)
            B = len(chunk)
            for i, n in enumerate(names_all):
                t = store[n]
                C = t.shape[-1]
                feats[f"L{i + 1}"].append(t.reshape(B, 64, C).cpu().numpy())
            if (i0 // batch) % 10 == 0:
                print(f"    encoded {min(i0 + batch, len(rows))}/{len(rows)}")
    for name in feats:
        feats[name] = np.concatenate(feats[name], 0)
    tgt_idx = np.array([pov_square_index(r[3], r[5].turn) for r in rows])   # POV-corrected
    return feats, tgt_idx, layer_names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trunk", choices=["jepa", "lc0-small", "lc0-big"], required=True)
    ap.add_argument("--labeled", default="data/derived/transition_data_labeled.npz")
    ap.add_argument("--jepa-ckpt", default="artifacts/experiments/jepa_t1_latest.pt")
    ap.add_argument("--lc0-small-onnx", default="data/engines/lc0/t1-256x10.onnx")
    ap.add_argument("--lc0-big-onnx", default="data/engines/lc0/t1-512x15x8h.onnx")
    ap.add_argument("--balanced-band", type=float, default=1.0)
    ap.add_argument("--hang-thr", type=float, default=0.25)
    ap.add_argument("--fair-thr", type=float, default=0.05)
    ap.add_argument("--max-ctrl", type=int, default=3000)
    ap.add_argument("--test-frac", type=float, default=0.25)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=0, help="0 = trunk default")
    ap.add_argument("--mlp", action="store_true",
                     help="also run MLP+control-task per layer (Kaveh 2026-07-31: is "
                          "the concept present-but-nonlinearly-entangled early, only "
                          "becoming linearly readable late -- a la logit-lens -- "
                          "rather than genuinely computed late?)")
    ap.add_argument("--mlp-hidden", type=int, nargs="+", default=[128, 32],
                     help="MLP hidden layer sizes, e.g. --mlp-hidden 128 32")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    t0 = time.time()
    rng = np.random.default_rng(args.seed)
    out = args.out or f"artifacts/experiments/layer_sweep_{args.trunk}"

    from catspace.research.tools.training_infra.train.scaffold import resolve_device
    dev = resolve_device("auto")

    print(f"building rows from {args.labeled} ...")
    rows = build_rows(args.labeled, args.balanced_band, args.hang_thr, args.fair_thr,
                       args.max_ctrl, rng)
    if len(rows) < 20:
        print("VERDICT layer-sweep: ABORT -- too few rows"); return
    labels = np.array([r[0] for r in rows])
    majors = np.array([r[1] for r in rows])
    groups = np.array([r[4] for r in rows])
    # Nonlinear-methodology positive control (Kaveh 2026-07-31): "major-piece" is
    # trivially LINEARLY decodable (AUC 1.000 even with logistic regression), so it
    # validates feature extraction but says nothing about whether the MLP can find
    # genuinely nonlinear structure. XOR of two independently-linearly-readable facts
    # is the classic case a linear probe CANNOT solve but a 1-hidden-layer MLP can --
    # if the MLP fails this, no MLP result anywhere else can be trusted, however big.
    whites = np.array([int(r[5].turn) for r in rows])
    xor_label = (majors ^ whites)

    print(f"encoding with trunk={args.trunk} ...")
    if args.trunk == "jepa":
        feats_by_layer, tgt_idx, layer_names = jepa_layers(
            rows, args.jepa_ckpt, dev, batch=args.batch or 256)
    elif args.trunk == "lc0-small":
        feats_by_layer, tgt_idx, layer_names = lc0_layers(
            rows, args.lc0_small_onnx, dev, batch=args.batch or 128)
    else:
        feats_by_layer, tgt_idx, layer_names = lc0_layers(
            rows, args.lc0_big_onnx, dev, batch=args.batch or 32)

    gs = np.unique(groups)
    te_g = set(rng.choice(gs, max(1, int(len(gs) * args.test_frac)), replace=False).tolist())
    te = np.array([g in te_g for g in groups]); tr = ~te

    results = []   # (layer_name, auc_hang, lo_h, hi_h, auc_ctrl, lo_c, hi_c)
    for name in layer_names:
        tok = feats_by_layer[name]                       # (N, 64, C)
        pooled = tok.mean(axis=1)
        target = tok[np.arange(len(tok)), tgt_idx]
        feat = np.concatenate([pooled, target], axis=1)

        lin_h = linear_readout(feat, labels, tr, te, rng, args.n_boot)
        lin_c = linear_readout(feat, majors, tr, te, rng, args.n_boot)
        row = [name, lin_h["auc"], lin_h["lo"], lin_h["hi"], lin_c["auc"], lin_c["lo"], lin_c["hi"]]
        msg = (f"  {name}: hanging AUC {lin_h['auc']:.3f} [{lin_h['lo']:.3f},{lin_h['hi']:.3f}] | "
               f"positive-control (major-piece) AUC {lin_c['auc']:.3f} "
               f"[{lin_c['lo']:.3f},{lin_c['hi']:.3f}]")
        if args.mlp:
            hidden = tuple(args.mlp_hidden)
            xor_mlp = mlp_readout_with_control(feat, xor_label, tr, te, rng, hidden=hidden,
                                                n_boot=args.n_boot, seed=args.seed)
            xor_ok = xor_mlp["lo_real"] > 0.5 and xor_mlp["selectivity"] > 0.1
            msg2 = (f"    [nonlinear-control XOR(major,white-to-move)] MLP AUC "
                    f"{xor_mlp['auc_real']:.3f} [{xor_mlp['lo_real']:.3f},{xor_mlp['hi_real']:.3f}] "
                    f"ctrl {xor_mlp['auc_ctrl']:.3f} selectivity {xor_mlp['selectivity']:+.3f} "
                    f"({'MLP CAN find nonlinear structure here' if xor_ok else 'MLP FAILED XOR -- do not trust its hanging-task result at this layer'})")
            print(msg2)

            mlp = mlp_readout_with_control(feat, labels, tr, te, rng, hidden=hidden,
                                            n_boot=args.n_boot, seed=args.seed)
            row += [mlp["auc_real"], mlp["lo_real"], mlp["hi_real"], mlp["selectivity"], xor_ok]
            trustworthy = mlp["lo_real"] > 0.5 and mlp["selectivity"] > 0.1 and xor_ok
            msg += (f" | MLP AUC {mlp['auc_real']:.3f} [{mlp['lo_real']:.3f},{mlp['hi_real']:.3f}] "
                    f"ctrl {mlp['auc_ctrl']:.3f} selectivity {mlp['selectivity']:+.3f} "
                    f"({'TRUSTWORTHY+POSITIVE' if trustworthy else 'null-or-untrustworthy'})")
        results.append(row)
        print(msg)

    ctrl_hits = sum(1 for r in results if r[5] > 0.9)
    hang_hits = sum(1 for r in results if r[2] > 0.5)
    print(f"VERDICT layer-sweep-{args.trunk}: {len(layer_names)} layers | "
          f"positive-control >0.9 AUC (lower CI) at {ctrl_hits}/{len(layer_names)} layers "
          f"({'harness OK -- can detect a known-present signal' if ctrl_hits > 0 else 'HARNESS SUSPECT -- cannot detect even the positive control'}) | "
          f"hanging-ness clears chance (lower CI>0.5) at {hang_hits}/{len(layer_names)} layers")

    if args.mlp:
        mlp_hits = [i for i, r in enumerate(results) if r[8] > 0.5 and r[10] > 0.1 and r[11]]
        lin_hits = [i for i, r in enumerate(results) if r[2] > 0.5]
        first_mlp = mlp_hits[0] if mlp_hits else None
        first_lin = lin_hits[0] if lin_hits else None
        if first_mlp is not None and (first_lin is None or first_mlp < first_lin):
            reading = ("MLP (trustworthy, beat its control) clears chance BEFORE linear does -- "
                       "consistent with 'present early but nonlinearly entangled, only becomes "
                       "linearly readable late' (Kaveh's logit-lens-style hypothesis)")
        elif first_mlp is not None and first_mlp == first_lin:
            reading = "MLP and linear clear chance at the same layer -- no evidence of earlier nonlinear presence"
        else:
            reading = ("MLP never clears chance (trustworthily) before linear does -- no evidence the "
                       "concept is present-but-entangled early; consistent with genuine late computation")
        print(f"VERDICT layer-sweep-{args.trunk}-mlp-vs-linear: first trustworthy MLP hit = "
              f"{layer_names[first_mlp] if first_mlp is not None else 'none'} | "
              f"first linear hit = {layer_names[first_lin] if first_lin is not None else 'none'} | {reading}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.5))
    xs = np.arange(len(results))
    h_auc = [r[1] for r in results]; h_lo = [r[2] for r in results]; h_hi = [r[3] for r in results]
    c_auc = [r[4] for r in results]; c_lo = [r[5] for r in results]; c_hi = [r[6] for r in results]
    ax.plot(xs, c_auc, "o-", color="#3b6", label="positive control (major piece, linear)")
    ax.fill_between(xs, c_lo, c_hi, color="#3b6", alpha=0.15)
    ax.plot(xs, h_auc, "o-", color="#c33", label="hanging piece, linear")
    ax.fill_between(xs, h_lo, h_hi, color="#c33", alpha=0.15)
    save_kw = dict(layer_names=layer_names, h_auc=h_auc, c_auc=c_auc)
    if args.mlp:
        m_auc = [r[7] for r in results]; m_lo = [r[8] for r in results]; m_hi = [r[9] for r in results]
        ax.plot(xs, m_auc, "s--", color="#c60", label="hanging piece, MLP")
        ax.fill_between(xs, m_lo, m_hi, color="#c60", alpha=0.12)
        save_kw["m_auc"] = m_auc
    ax.axhline(0.5, color="k", ls="--", lw=0.7)
    ax.set_xticks(xs); ax.set_xticklabels([r[0] for r in results], rotation=45, ha="right")
    ax.set_ylabel("held-out AUC"); ax.set_ylim(0.2, 1.05)
    ax.set_title(f"per-layer probe sweep -- trunk={args.trunk}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    Path("artifacts/experiments").mkdir(exist_ok=True, parents=True)
    Path("docs/figures").mkdir(exist_ok=True, parents=True)
    fig.savefig(f"{out}.png", dpi=130)
    fig.savefig(f"docs/figures/{Path(out).name}.png", dpi=130)
    np.savez(f"{out}.npz", **save_kw)
    print(f"wrote {out}.png + docs/figures/{Path(out).name}.png + {out}.npz")
    print(f"done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
