#!/usr/bin/env python
"""
experiments/mate_bench.py — score a checkpoint on the mate-in-N benchmark sets
(mined from Lichess puzzles; EVAL-ONLY, see data_registry.json).

Two measurements per set:
  FIELD-ONLY (mate-in-1 set): fraction of positions where the field's
    top-ranked move delivers mate immediately -- the purest full-board
    rim-resolution number, no search involved.
  SEARCH: the engine (committor readout) plays the mating side vs a
    full-strength depth-limited Stockfish DEFENDER (defense role only,
    leakage-clean); success = checkmate within 2N + slack plies.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import chess
import chess.engine
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from catspace.data.encode import encode_meta, encode_packed
from catspace.nn.features import feature_planes, omega_ids


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--phead", default=None, help="outcome head used as committor readout")
    ap.add_argument("--whead", default=None, help="alternative: distilled committor W-head")
    ap.add_argument("--sets", nargs="+",
                    default=["artifacts/experiments/mate_in_1_n500.json",
                             "artifacts/experiments/mate_in_2_n500.json",
                             "artifacts/experiments/mate_in_3_n500.json"])
    ap.add_argument("--n", type=int, default=120, help="positions per set")
    ap.add_argument("--nodes", type=int, default=800)
    ap.add_argument("--slack-plies", type=int, default=4)
    ap.add_argument("--sf-depth", type=int, default=12, help="defender depth (full strength)")
    ap.add_argument("--stockfish", default="stockfish")
    ap.add_argument("--label", default="")
    ap.add_argument("--dump-results", default=None,
                    help="write per-position win/loss vectors (json) for overlap "
                         "comparison across checkpoints")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    import torch
    from catspace.nn.fb import load_ckpt, pick_device
    from catspace.nn.mcts import FBMCTSPolicy

    dev = pick_device(args.device)
    fb, pay = load_ckpt(Path(args.ckpt), dev)
    fb.eval()
    if args.phead:
        from catspace.nn.eval_head import EvalHead
        hp = torch.load(args.phead, map_location=dev, weights_only=False)
        ph = EvalHead(d_in=hp["d_in"]).to(dev)
        ph.load_state_dict(hp["state"]); ph.eval()

        class PheadCommittor(torch.nn.Module):
            def forward(self, f):
                pw = torch.softmax(ph(f), dim=1)[:, 0].clamp_min(1e-6)
                return -torch.log(pw).unsqueeze(-1)
        head = PheadCommittor()
    elif args.whead:
        hp = torch.load(args.whead, map_location=dev, weights_only=False)
        head = torch.nn.Sequential(torch.nn.Linear(hp["d_in"], 128), torch.nn.ReLU(),
                                   torch.nn.Linear(128, 1), torch.nn.Softplus()).to(dev)
        head.load_state_dict(hp["state"]); head.eval()
    else:
        raise SystemExit("need --phead or --whead")
    om_row = omega_ids(np.array([1800]), np.array([1800]), np.array([np.nan]))[0]

    def d_of(boards):
        with torch.no_grad():
            packed = np.stack([encode_packed(b) for b in boards])
            meta = np.stack([encode_meta(b) for b in boards])
            pl = torch.from_numpy(feature_planes(packed, meta)).to(dev)
            om = torch.from_numpy(np.tile(om_row, (len(boards), 1))).to(dev)
            return head(fb.embed_F(pl, om)).squeeze(-1).cpu().numpy()

    eng = chess.engine.SimpleEngine.popen_uci(args.stockfish)
    for set_path in args.sets:
        payload = json.loads(Path(set_path).read_text())
        fens = payload["fens"][:args.n]
        n_mate = int(payload["meta"]["theme"].replace("mateIn", ""))

        # field-only top-move-is-mate (meaningful for mate-in-1)
        if n_mate == 1:
            hits = 0
            for fen in fens:
                b0 = chess.Board(fen)
                moves = list(b0.legal_moves)
                kids = []
                for m in moves:
                    b = b0.copy(stack=False); b.push(m)
                    kids.append(b)
                d = d_of(kids)
                if kids[int(np.argmin(d))].is_checkmate():
                    hits += 1
            print(f"VERDICT MATE_BENCH{('_' + args.label) if args.label else ''} "
                  f"mateIn1 FIELD-ONLY top-move-mates {hits}/{len(fens)} "
                  f"= {hits/len(fens):.3f}")

        # search vs SF defender
        wins = 0
        winvec = []
        budget = 2 * n_mate + args.slack_plies
        for i, fen in enumerate(fens):
            pol = FBMCTSPolicy(fb, pay["zgoals"]["MATE_W"], max_nodes=args.nodes,
                               device=dev, committor_head=head)
            b = chess.Board(fen)
            us = b.turn
            rng = np.random.default_rng([args.seed, i])
            for _ in range(budget):
                if b.is_game_over(claim_draw=True):
                    break
                if b.turn == us:
                    b.push(pol.move(b, rng))
                else:
                    r = eng.play(b, chess.engine.Limit(depth=args.sf_depth))
                    b.push(r.move)
            out = b.outcome(claim_draw=True)
            won = bool(out and out.winner == us)
            wins += int(won)
            winvec.append(int(won))
        print(f"VERDICT MATE_BENCH{('_' + args.label) if args.label else ''} "
              f"mateIn{n_mate} SEARCH@{args.nodes}n mates {wins}/{len(fens)} "
              f"= {wins/len(fens):.3f} (within {budget} plies, SF depth-{args.sf_depth} defender)")
        if args.dump_results:
            p = Path(args.dump_results)
            d = json.loads(p.read_text()) if p.exists() else {}
            d[f"{args.label}_mateIn{n_mate}"] = winvec
            p.write_text(json.dumps(d))
    eng.quit()


if __name__ == "__main__":
    main()
