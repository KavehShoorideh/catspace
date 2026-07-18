#!/usr/bin/env python
"""experiments/show_mate.py — play toy starts (White = field+MCTS, Black =
tablebase-optimal) and print the first converted game as SAN so the mate is
visible, not just a rate."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import chess, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.value_fixed_point import TB, tb_best_move
from experiments.playout_ab import mate_vector  # reuse loading? no -- need moves; inline

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="data/derived/sep/cert_base_full.pt")
    ap.add_argument("--phead", default="data/derived/sep/cert_base_full_phead.pt")
    ap.add_argument("--fixed-set", default="artifacts/experiments/krrkbp_test_n200.json")
    ap.add_argument("--nodes", type=int, default=800)
    ap.add_argument("--max-starts", type=int, default=6)
    ap.add_argument("--max-plies", type=int, default=120)
    ap.add_argument("--syzygy-dir", default="data/syzygy")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    import torch
    from catspace.nn.eval_head import EvalHead
    from catspace.nn.fb import load_ckpt, pick_device
    from catspace.nn.policy_fb import make_search_policy
    dev = pick_device(args.device)
    fb, pay = load_ckpt(Path(args.ckpt), dev)
    hp = torch.load(args.phead, map_location=dev, weights_only=False)
    ph = EvalHead(d_in=hp["d_in"]).to(dev); ph.load_state_dict(hp["state"]); ph.eval()
    class Committor(torch.nn.Module):
        def forward(self, f):
            p = torch.softmax(ph(f), dim=1)
            return -torch.log(p[:, 0].clamp_min(1e-6)).unsqueeze(-1)
    pol = make_search_policy("mcts", fb, pay["zgoals"]["MATE_W"], max_nodes=args.nodes,
                             device=dev, committor_head=Committor())
    tb = TB(args.syzygy_dir)
    starts = json.loads(Path(args.fixed_set).read_text())["fens"][:args.max_starts]
    rng = np.random.default_rng(0)
    for i, fen in enumerate(starts):
        b = chess.Board(fen)
        moves, seen = [], set()
        for _ in range(args.max_plies):
            if b.is_game_over(claim_draw=True):
                break
            m = pol.move(b, rng) if b.turn == chess.WHITE else tb_best_move(b, tb, seen)
            if b.turn == chess.BLACK:
                seen.add(b.board_fen())
            if m is None:
                break
            moves.append(b.san(m)); b.push(m)
        out = b.outcome(claim_draw=True)
        won = out and out.winner == chess.WHITE
        print(f"start {i}: {fen}")
        print(f"  result: {'1-0 MATE in ' + str(len(moves)) + ' plies' if won else (out.termination.name if out else 'budget')}")
        if won:
            san = []
            bb = chess.Board(fen)
            n0 = bb.fullmove_number
            for j, mv in enumerate(moves):
                if bb.turn == chess.WHITE:
                    san.append(f"{bb.fullmove_number}.{mv}")
                else:
                    san.append(mv)
                bb.push_san(mv)
            print("  " + " ".join(san))
            break
    tb.close()

if __name__ == "__main__":
    main()
