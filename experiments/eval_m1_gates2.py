#!/usr/bin/env python
"""experiments/eval_m1_gates2.py -- the remaining two M1 gates, v3 vs the frozen-trunk IQE head,
same probes for both:
  OFF-DISTRIBUTION d_mate : fresh synthetic KQvK/KRvK winning positions (no game history),
                            d_mate vs true |DTZ| Spearman. (ClockField measured 0.505 here before.)
  OPENING SANITY          : (a) no NaN/inf on startpos + 300 real opening positions (ply<=8);
                            (b) opening pair-order: same-game pairs BOTH inside ply<=12 -- does the
                            metric order tiny early gaps at all? (v3 never saw openings.)
The head path runs the FULL pipeline on fresh boards: ONNX trunk forward (hooked features) -> head.
"""
from __future__ import annotations

import argparse, sys, time
from pathlib import Path

import chess
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.train_iqe_head import IQEHead, build_pairs
from experiments.train_clock_field import ClockField
from catspace.train.scaffold import resolve_device
from scipy.stats import spearmanr


class TrunkField:
    """full path for fresh boards: lczerolens trunk (frozen, hooked) -> IQE head. M1's play-time object."""

    def __init__(self, onnx, head_ckpt, device):
        from lczerolens import LczeroModel
        self.dev = device
        self.trunk = LczeroModel.from_onnx_path(onnx).float().to(device).eval()
        names = [n for n, _ in self.trunk.named_modules()
                 if n and all(k not in n for k in ("policy", "value", "wdl", "output"))]
        self._f = {}
        dict(self.trunk.named_modules())[names[-1]].register_forward_hook(
            lambda mo, i, o: self._f.__setitem__("t", o))
        p = torch.load(head_ckpt, map_location=device, weights_only=False); cfg = p["cfg"]
        self.head = IQEHead(in_ch=cfg["in_ch"], d=cfg["d"], components=cfg["components"],
                            adapter_ch=cfg["adapter_ch"]).to(device)
        self.head.load_state_dict(p["state_dict"]); self.head.eval()

    def phi_boards(self, lcboards):
        x = torch.stack([b.to_input_tensor() for b in lcboards]).float().to(self.dev)
        with torch.no_grad():
            self.trunk(x)
            return self.head.phi(self._f["t"])

    def d_mate_boards(self, lcboards):
        with torch.no_grad():
            return self.head.d_mate_emb(self.phi_boards(lcboards)).cpu().numpy()

    def d_pair_boards(self, a, b):
        with torch.no_grad():
            return self.head.d_pair_emb(self.phi_boards(a), self.phi_boards(b)).cpu().numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--onnx", default="data/engines/maia/maia-1500.onnx")
    ap.add_argument("--head", default="artifacts/experiments/iqe_head_maia1500_latest.pt")
    ap.add_argument("--v3", default="artifacts/experiments/field_fullgame_v3_final.pt")
    ap.add_argument("--data", default="data/derived/field_std_v1.npz")
    ap.add_argument("--n-endgames", type=int, default=1500); ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time(); dev = resolve_device("auto"); rng = np.random.default_rng(args.seed)
    from lczerolens import LczeroBoard
    from catspace.tb import TB, DEFAULT_SYZYGY
    from experiments.gen_dtm_data import random_class_start

    tf = TrunkField(args.onnx, args.head, dev)
    p3 = torch.load(args.v3, map_location=dev, weights_only=False); cfg3 = p3["cfg"]
    v3 = ClockField(cfg3["d"], ch=cfg3["ch"], blocks=cfg3["blocks"], in_planes=112).to(dev)
    v3.load_state_dict(p3["state_dict"]); v3.eval()

    def v3_dmate(lcbs):
        x = torch.stack([b.to_input_tensor() for b in lcbs]).float().to(dev)
        with torch.no_grad():
            return v3.d_mate(x).cpu().numpy()

    def v3_dpair(a, b):
        xa = torch.stack([x.to_input_tensor() for x in a]).float().to(dev)
        xb = torch.stack([x.to_input_tensor() for x in b]).float().to(dev)
        with torch.no_grad():
            return v3.d_pair(xa, xb).cpu().numpy()

    # ---- GATE A: off-distribution d_mate (fresh synthetic endgames, no history)
    tb = TB(str(DEFAULT_SYZYGY), cache_db=None); syz = tb.tb
    boards, dtzs = [], []
    while len(boards) < args.n_endgames:
        cls = "KQvK" if rng.random() < 0.5 else "KRvK"
        b = random_class_start(rng, cls)
        if b is None or b.is_game_over() or b.turn != chess.WHITE:
            continue
        try:
            d = abs(syz.probe_dtz(b))
        except Exception:
            continue
        if d < 1:
            continue
        boards.append(LczeroBoard(b.fen())); dtzs.append(d)
    tb.close()
    dtzs = np.array(dtzs)
    dm_head = np.concatenate([tf.d_mate_boards(boards[i:i+512]) for i in range(0, len(boards), 512)])
    dm_v3 = np.concatenate([v3_dmate(boards[i:i+512]) for i in range(0, len(boards), 512)])
    offd_head = float(spearmanr(dm_head, dtzs).correlation)
    offd_v3 = float(spearmanr(dm_v3, dtzs).correlation)

    # ---- GATE B: opening sanity
    z = np.load(args.data); game = z["game"]; ply = z["ply"]
    open_rows = np.flatnonzero(ply <= 8)[:300]
    planes = z["planes"]
    x_open = torch.from_numpy(planes[open_rows].astype(np.float32)).to(dev)
    with torch.no_grad():
        tf.trunk(torch.stack([LczeroBoard().to_input_tensor()]).float().to(dev))
        start_dm_head = float(tf.d_mate_boards([LczeroBoard()])[0])
        start_dm_v3 = float(v3_dmate([LczeroBoard()])[0])
        head_open_phi = tf.head.phi(  # trunk features for real opening rows via full forward
            (lambda: (tf.trunk(x_open), tf._f["t"])[1])())
        dm_open_head = tf.head.d_mate_emb(head_open_phi).cpu().numpy()
        dm_open_v3 = v3.d_mate(x_open).cpu().numpy()
    finite_head = bool(np.isfinite(dm_open_head).all()); finite_v3 = bool(np.isfinite(dm_open_v3).all())
    # opening-only pair ordering (both ends ply<=12): does the metric order tiny early gaps?
    open_games = {}
    for i in np.flatnonzero(ply <= 12):
        open_games.setdefault(int(game[i]), []).append(i)
    S, G, D = [], [], []
    for rows in open_games.values():
        rows = sorted(rows, key=lambda i: ply[i])
        for a in range(len(rows)):
            for b in range(a + 1, len(rows)):
                S.append(rows[a]); G.append(rows[b]); D.append(ply[rows[b]] - ply[rows[a]])
    S, G, D = np.array(S[:3000]), np.array(G[:3000]), np.array(D[:3000], float)
    xa = torch.from_numpy(planes[S].astype(np.float32)); xb = torch.from_numpy(planes[G].astype(np.float32))
    with torch.no_grad():
        tf.trunk(xa.to(dev)); fa = tf._f["t"]
        tf.trunk(xb.to(dev)); fb = tf._f["t"]
        dp_head = tf.head.d_pair_emb(tf.head.phi(fa), tf.head.phi(fb)).cpu().numpy()
        dp_v3 = v3.d_pair(xa.to(dev), xb.to(dev)).cpu().numpy()
    op_head = float(spearmanr(dp_head, D).correlation)
    op_v3 = float(spearmanr(dp_v3, D).correlation)

    print("\n===== M1 GATES 2/2 (same probes for both) =====")
    print(f"{'probe':<44} {'v3 incumbent':>14} {'frozen-trunk head':>18}")
    print(f"{'off-dist d_mate rho (fresh endgames)':<44} {offd_v3:>+14.3f} {offd_head:>+18.3f}")
    print(f"{'opening d_mate finite (startpos+300)':<44} {str(finite_v3):>14} {str(finite_head):>18}")
    print(f"{'startpos d_mate value':<44} {start_dm_v3:>14.1f} {start_dm_head:>18.1f}")
    print(f"{'opening-only pair-order (ply<=12 pairs)':<44} {op_v3:>+14.3f} {op_head:>+18.3f}")
    print(f"VERDICT M1-GATES2 done [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
