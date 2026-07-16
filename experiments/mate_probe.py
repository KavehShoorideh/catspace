#!/usr/bin/env python
"""
experiments/mate_probe.py — single-position mate diagnostic (Kaveh 2026-07-16).

Given a FEN with a short forced mate, print:
  1. the FIELD's own ranking of every legal move (committor d_W of each child,
     no search) -- does the mating/boxing move rank first at zero depth?
  2. playouts vs the tablebase-optimal defender at several node budgets --
     does SEARCH find the mate, and in how many plies (vs the true DTM)?

Default position: KRR vs k, wK a3, Ra1, Ra2, bK h8, White mates in 2
(1.Rg1 Kh7 2.Rh2#).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import chess
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from catspace.data.encode import encode_meta, encode_packed
from catspace.nn.features import feature_planes, omega_ids
from experiments.value_fixed_point import TB, tb_best_move

MATE_IN_2 = "7k/8/8/8/8/K7/R7/R7 w - - 0 1"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fen", default=MATE_IN_2)
    ap.add_argument("--ckpt", default="data/derived/sep/rootloop_r12.pt")
    ap.add_argument("--whead", default="data/derived/sep/rootloop_r12_whead.pt")
    ap.add_argument("--budgets", type=int, nargs="+", default=[200, 800, 1600])
    ap.add_argument("--clearance-beta", type=float, default=0.0,
                    help="draw-surface clearance: reach = -d_W + beta*d_D "
                         "(loads the _dhead sibling of --whead)")
    ap.add_argument("--phead", default=None,
                    help="use a full-board-trained outcome head (*_phead.pt, "
                         "EvalHead CE on game results) as the W-committor: "
                         "d_W = -ln softmax(logits)[W]. Zero-training test of "
                         "'train full board, read out committor in the toy'. "
                         "Overrides --whead.")
    ap.add_argument("--max-plies", type=int, default=40)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    import torch
    from catspace.nn.fb import load_ckpt, pick_device
    from catspace.nn.mcts import FBMCTSPolicy

    dev = pick_device(args.device)
    fb, pay = load_ckpt(Path(args.ckpt), dev)
    fb.eval()
    whead = None
    if args.phead:
        from catspace.nn.eval_head import EvalHead
        hp = torch.load(args.phead, map_location=dev, weights_only=False)
        ph = EvalHead(d_in=hp["d_in"]).to(dev)
        ph.load_state_dict(hp["state"]); ph.eval()

        class PheadCommittor(torch.nn.Module):
            def forward(self, f):
                pw = torch.softmax(ph(f), dim=1)[:, 0].clamp_min(1e-6)
                return -torch.log(pw).unsqueeze(-1)
        whead = PheadCommittor()
        print(f"W-committor = full-board outcome head {args.phead}")
        args.whead = None
    elif args.whead:
        hp = torch.load(args.whead, map_location=dev, weights_only=False)
        whead = torch.nn.Sequential(torch.nn.Linear(hp["d_in"], 128), torch.nn.ReLU(),
                                    torch.nn.Linear(128, 1), torch.nn.Softplus()).to(dev)
        whead.load_state_dict(hp["state"]); whead.eval()
    dhead = None
    if args.clearance_beta and args.whead:
        dp = torch.load(args.whead.replace("_whead", "_dhead"),
                        map_location=dev, weights_only=False)
        dhead = torch.nn.Sequential(torch.nn.Linear(dp["d_in"], 128), torch.nn.ReLU(),
                                    torch.nn.Linear(128, 1), torch.nn.Softplus()).to(dev)
        dhead.load_state_dict(dp["state"]); dhead.eval()
        print(f"clearance readout: beta={args.clearance_beta}")
    om_row = omega_ids(np.array([1800]), np.array([1800]), np.array([np.nan]))[0]

    b0 = chess.Board(args.fen)
    tb = TB("data/syzygy")
    _, dtz = tb.wdl_dtz(b0)
    print(f"position: {args.fen}")
    print(f"tablebase: DTZ {dtz} (White {'wins' if dtz else '??'})")

    # 1. field-only ranking of root moves
    moves = list(b0.legal_moves)
    children = []
    for m in moves:
        b = b0.copy(stack=False); b.push(m)
        children.append(b)
    with torch.no_grad():
        packed = np.stack([encode_packed(b) for b in children])
        meta = np.stack([encode_meta(b) for b in children])
        pl = torch.from_numpy(feature_planes(packed, meta)).to(dev)
        om = torch.from_numpy(np.tile(om_row, (len(children), 1))).to(dev)
        f = fb.embed_F(pl, om)
        if whead is not None:
            d = whead(f).squeeze(-1).cpu().numpy()
        else:
            zW = pay["zgoals"]["MATE_W"]
            zW = zW.to(dev).float() if torch.is_tensor(zW) else torch.as_tensor(
                np.asarray(zW), dtype=torch.float32, device=dev)
            d = fb.distance_matrix(f, zW[None, :])[:, 0].cpu().numpy()
    mate_now = [b.is_checkmate() for b in children]
    order = np.argsort(d)
    print("\nfield ranking of root moves (lower d_W = closer to mate surface):")
    for rank, i in enumerate(order, 1):
        tbv = tb.wdl_dtz(children[i])
        tag = " MATE" if mate_now[i] else (f" dtz={tbv[1]}" if tbv[1] is not None else "")
        print(f"  #{rank:>2} {b0.san(moves[i]):>6}  d_W={d[i]:.4f}{tag}")
    spread = float(d.max() - d.min())
    print(f"root move-spread: {spread:.4f} "
          f"({'FLAT' if spread < 0.02 else 'has gradient'})")

    # 2. search playouts
    for nodes in args.budgets:
        pol = FBMCTSPolicy(fb, pay["zgoals"]["MATE_W"], max_nodes=nodes, device=dev,
                           committor_head=whead, committor_dhead=dhead,
                           clearance_beta=args.clearance_beta)
        b = chess.Board(args.fen)
        seen = set()
        rng = np.random.default_rng(0)
        line = []
        for _ in range(args.max_plies):
            if b.is_game_over(claim_draw=True):
                break
            if b.turn == chess.WHITE:
                m = pol.move(b, rng)
            else:
                m = tb_best_move(b, tb, seen)
                seen.add(b.board_fen())
            line.append(b.san(m))
            b.push(m)
        out = b.outcome(claim_draw=True)
        res = ("MATE in " + str(b.ply()) + " plies" if out and out.winner == chess.WHITE
               else (out.termination.name if out else "cutoff"))
        print(f"VERDICT MATE_PROBE nodes={nodes}: {res}  line: {' '.join(line[:12])}")
    tb.close()


if __name__ == "__main__":
    main()
