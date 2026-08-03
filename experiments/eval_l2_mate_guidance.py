#!/usr/bin/env python
"""experiments/eval_l2_mate_guidance.py -- the ACCEPTANCE TEST (Kaveh 2026-07-20):
"if the planner gets within 5 moves of mate, the L2 layer should be accurate enough to
guide it exactly to mate."

From won positions within N plies of mate, White plays L2-GREEDY (the move whose child
minimizes L2's expected distance-to-mate) while Black plays TABLEBASE-OPTIMAL defense.
Success = White delivers mate within a ply cap. Baseline = a random legal White move
under the same optimal defense. Uses the Syzygy tablebase only as the adversary + oracle.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np, torch, chess
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from catspace.research.tools.chess_specific.chessdata.encode import board_from_packed, encode_meta, encode_packed
from catspace.research.components.encoder.approaches.jepa_tokenizer.src.features import feature_planes, omega_ids
from catspace.research.components.encoder.approaches.jepa_tokenizer.src.fb import load_ckpt, pick_device
from experiments.value_fixed_point import TB, tb_best_move

BOARD_ONLY = (18, 19)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--l1", default="data/derived/sep/iqe_geom_min.pt")
    ap.add_argument("--l2", default="data/derived/sep/l2_head.pt")
    ap.add_argument("--data", default="data/derived/lichess_nearmate.npz")
    ap.add_argument("--syzygy", default="data/syzygy")
    ap.add_argument("--max-dtm", type=int, default=10, help="start within this many plies of mate")
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--ply-cap", type=int, default=40)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    dev = pick_device(args.device)
    fb, _ = load_ckpt(Path(args.l1), dev); fb.eval()
    hp = torch.load(args.l2, map_location=dev, weights_only=False)
    head = torch.nn.Sequential(torch.nn.Linear(hp["d_in"], hp["hidden"]), torch.nn.ReLU(),
                               torch.nn.Linear(hp["hidden"], hp["n_class"])).to(dev)
    head.load_state_dict(hp["state"]); head.eval()
    center = torch.from_numpy(np.asarray(hp["bin_center"])).float().to(dev)
    om = omega_ids(np.array([1800]), np.array([1800]), np.array([300.0]))[0]
    tb = TB(args.syzygy)
    nz = np.load(args.data); dtm = nz["dtm"].astype(np.float32)
    allp, allm = np.asarray(nz["packed"]), np.asarray(nz["meta"])

    def exp_dist(boards):
        pk = np.stack([encode_packed(b) for b in boards]); mt = np.stack([encode_meta(b) for b in boards])
        pl = feature_planes(pk, mt); pl[:, BOARD_ONLY] = 0.0
        with torch.no_grad():
            F = fb.embed_F(torch.from_numpy(pl).to(dev), torch.from_numpy(np.tile(om, (len(boards), 1))).to(dev))
            P = torch.softmax(head(F), 1)
            return (P @ center).cpu().numpy()                      # expected distance-to-mate

    def l2_move(board):                                            # White: child minimizing L2 expected dist
        mv = list(board.legal_moves); kids = []
        for m in mv:
            c = board.copy(stack=False); c.push(m); kids.append(c)
        return mv[int(np.argmin(exp_dist(kids)))]

    # candidate starts: won, White-to-move, within max-dtm of mate
    won = np.flatnonzero((dtm > 0) & (dtm <= args.max_dtm))
    rng = np.random.default_rng(0)
    starts = []
    for j in rng.permutation(won):
        b = board_from_packed(allp[j], allm[j])
        if b.turn == chess.WHITE and not b.is_game_over():
            starts.append(b)
        if len(starts) >= args.n:
            break

    def play(board, white_policy):
        b = board.copy(stack=False)
        for _ in range(args.ply_cap):
            if b.is_game_over():
                return b.is_checkmate() and not b.turn        # mate delivered by White (Black to move, mated)
            if b.turn == chess.WHITE:
                b.push(white_policy(b))
            else:
                m = tb_best_move(b, tb)                        # optimal defense
                b.push(m if m is not None else next(iter(b.legal_moves)))
        return False

    l2_win = sum(play(b, l2_move) for b in starts)
    rnd = np.random.default_rng(1)
    rand_win = sum(play(b, lambda bd: list(bd.legal_moves)[int(rnd.integers(len(list(bd.legal_moves))))]) for b in starts)
    n = len(starts)
    print(f"starts: {n} won positions, White-to-move, DTM<= {args.max_dtm}")
    print(f"  L2-GREEDY mate rate (vs optimal defense): {l2_win}/{n} = {l2_win/n:.3f}")
    print(f"  random-move baseline:                     {rand_win}/{n} = {rand_win/n:.3f}")
    print(f"VERDICT L2_GUIDANCE mate_rate={l2_win/n:.3f} (n={n}, dtm<={args.max_dtm}) baseline={rand_win/n:.3f}")


if __name__ == "__main__":
    main()
