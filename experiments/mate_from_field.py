#!/usr/bin/env python
"""experiments/mate_from_field.py -- can the trained quasimetric field actually MATE?
(Kaveh 2026-07-26). Ordering metrics (+0.96) are a proxy; this is the real test: use the
field as a GREEDY planner and see if it forces checkmate.

Policy: at White's turn, pick the legal move minimising d(child, MATE-region), where
d(s, MATE) = min over a mate-landmark bank of IQE(phi(s), phi(mate_i)) -- the region-as-min
readout. Black (defender) plays TABLEBASE-OPTIMAL defense (maximal delay) -- the hardest
possible opponent. We stop at mate / draw / ply cap.

Reports per class and overall: mate RATE (% of starts forced to mate under optimal defense),
median plies-to-mate, and EXCESS plies over the optimal DTM (how much slower than perfect
play). A field that has really learned distance-to-mate should mate at a high rate with low
excess; ordering that looks great but can't close out mate would show up as a low rate.
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

from catspace.data.encode import encode_meta, encode_packed
from catspace.nn.fb import pick_device
from catspace.tb import TB, DEFAULT_SYZYGY, rollout_dtm, tb_best_move
from experiments.arch_bakeoff import tokens
from experiments.gen_dtm_data import random_class_start
from experiments.train_quasimetric import TwoTowerIQE
from experiments.value_fixed_point import white_pov_value


def load_field(ckpt_path, dev):
    ck = torch.load(ckpt_path, map_location=dev, weights_only=False)
    c = ck["cfg"]
    net = TwoTowerIQE(c["d"], c["d_bb"], c["blocks"], c["iqe_components"], c["shared"]).to(dev)
    net.load_state_dict(ck["state_dict"]); net.eval()
    return net, ck


@torch.no_grad()
def embed(net, boards, dev):
    pk = np.stack([encode_packed(b) for b in boards])
    mt = np.stack([encode_meta(b) for b in boards])
    ids, stm = tokens(pk, mt)
    return net.embedF(torch.from_numpy(ids.astype(np.int64)).to(dev),
                      torch.from_numpy(stm.astype(np.int64)).to(dev))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="artifacts/experiments/quasimetric_shared_v1.pt")
    ap.add_argument("--classes", nargs="*", default=["KQvK", "KRvK", "KRRvK", "KBBvK", "KBNvK"])
    ap.add_argument("--n", type=int, default=200, help="test starts per class")
    ap.add_argument("--bank", type=int, default=512, help="mate-landmark bank size")
    ap.add_argument("--cap-mult", type=float, default=3.0, help="ply cap = cap_mult * optimal_dtm")
    ap.add_argument("--mate-in-1", action="store_true", help="take an immediate mate if available")
    ap.add_argument("--diag", action="store_true",
                    help="along-line diagnostic: does d track distance-to-mate locally?")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()
    t0 = time.time(); dev = pick_device(args.device)
    rng = np.random.default_rng(args.seed)
    net, ck = load_field(args.ckpt, dev)
    print(f"[mate-from-field] ckpt {args.ckpt} cfg {ck['cfg']} "
          f"train-metrics {ck.get('metrics')}", flush=True)

    tb = TB(str(DEFAULT_SYZYGY), cache_db=None)

    if args.diag:
        torch.set_grad_enabled(False)
        from catspace.tb import rollout_line
        from scipy.stats import spearmanr
        line_sp, step_ok, pol_ok, pol_n = [], 0, 0, 0
        got = 0
        while got < args.n:
            cls = args.classes[rng.integers(0, len(args.classes))]
            b = random_class_start(rng, cls)
            if b is None or b.turn != chess.WHITE or white_pov_value(b, tb) != 1.0:
                continue
            line = rollout_line(b, tb, cap=200)
            if not line or len(line) < 4 or not line[-1].is_checkmate():
                continue
            got += 1
            mate_e = embed(net, [line[-1]], dev)                       # THIS line's terminal mate
            de = net.iqe.pairwise(embed(net, line, dev), mate_e)[:, 0].cpu().numpy()
            rem = np.arange(len(line) - 1, -1, -1)                     # true plies-to-mate
            line_sp.append(float(spearmanr(de, rem).correlation))
            step_ok += int(np.mean(np.diff(de) < 0) > 0.5)            # mostly-decreasing?
            # policy accuracy: at each White node, does greedy-on-d pick a DTM-reducing move?
            for i in range(0, len(line) - 1, 2):                      # White to move on even i
                s = line[i]
                if s.turn != chess.WHITE: continue
                dtm_s = rollout_dtm(s, tb)
                if dtm_s is None: continue
                kids, mvs = [], list(s.legal_moves)
                for m in mvs:
                    s.push(m); kids.append(s.copy(stack=False)); s.pop()
                dk = net.iqe.pairwise(embed(net, kids, dev), mate_e)[:, 0].cpu().numpy()
                s.push(mvs[int(np.argmin(dk))]); chosen_dtm = rollout_dtm(s, tb); s.pop()
                pol_n += 1
                # after a good White move it's Black-to-move with dtm ~ dtm_s-1; reducing = <=dtm_s
                if chosen_dtm is not None and chosen_dtm <= dtm_s: pol_ok += 1
        print(f"DIAG along-line (goal = THIS line's true mate, no bank):", flush=True)
        print(f"  d-vs-remaining-DTM spearman (per line): mean {np.mean(line_sp):+.3f} "
              f"median {np.median(line_sp):+.3f}", flush=True)
        print(f"  lines mostly-monotone-decreasing: {100*step_ok/max(1,got):.0f}%", flush=True)
        print(f"  greedy picks a DTM-reducing move: {100*pol_ok/max(1,pol_n):.1f}% "
              f"({pol_ok}/{pol_n})   [random~ chance]", flush=True)
        tb.close(); return

    # --- mate-landmark bank: sample real checkmate positions from optimal lines ---
    bank_boards = []
    while len(bank_boards) < args.bank:
        cls = args.classes[rng.integers(0, len(args.classes))]
        b = random_class_start(rng, cls)
        if b is None or b.turn != chess.WHITE or white_pov_value(b, tb) != 1.0:
            continue
        from catspace.tb import rollout_line
        line = rollout_line(b, tb, cap=200)
        if line and line[-1].is_checkmate():
            bank_boards.append(line[-1])
    bank_emb = embed(net, bank_boards, dev)                 # (B,d)
    print(f"  mate bank: {len(bank_boards)} checkmate positions [{time.time()-t0:.0f}s]", flush=True)

    @torch.no_grad()
    def d_to_mate(boards):
        e = embed(net, boards, dev)                         # (N,d)
        return net.iqe.pairwise(e, bank_emb).min(dim=1).values.cpu().numpy()   # (N,)

    overall = {"mate": 0, "n": 0, "excess": [], "plies": []}
    for cls in args.classes:
        res = {"mate": 0, "n": 0, "excess": [], "plies": []}
        got = 0
        while got < args.n:
            b0 = random_class_start(rng, cls)
            if b0 is None or b0.turn != chess.WHITE or white_pov_value(b0, tb) != 1.0:
                continue
            opt = rollout_dtm(b0, tb)
            if opt is None or opt < 1:
                continue
            got += 1
            cap = int(args.cap_mult * opt) + 4
            b = b0.copy(stack=False)
            plies = 0; mated = False
            while plies < cap:
                if b.is_checkmate(): mated = True; break
                if b.is_game_over(claim_draw=True): break
                if b.turn == chess.WHITE:                   # FIELD plays (greedy descent)
                    moves = list(b.legal_moves)
                    if args.mate_in_1:
                        mm = [m for m in moves if _gives_mate(b, m)]
                        if mm: b.push(mm[0]); plies += 1; mated = b.is_checkmate(); continue
                    kids = []
                    for m in moves:
                        b.push(m); kids.append(b.copy(stack=False)); b.pop()
                    d = d_to_mate(kids)
                    b.push(moves[int(np.argmin(d))])
                else:                                       # TABLEBASE-optimal defense
                    m = tb_best_move(b, tb, set())
                    if m is None: break
                    b.push(m)
                plies += 1
            res["n"] += 1
            if mated:
                res["mate"] += 1; res["plies"].append(plies); res["excess"].append(plies - opt)
        mr = 100 * res["mate"] / max(1, res["n"])
        mp = int(np.median(res["plies"])) if res["plies"] else -1
        ex = float(np.median(res["excess"])) if res["excess"] else float("nan")
        print(f"  {cls}: mate-rate {mr:5.1f}%  ({res['mate']}/{res['n']})  "
              f"median plies {mp}  median excess-over-optimal {ex:+.0f} [{time.time()-t0:.0f}s]",
              flush=True)
        for k in ("mate", "n"): overall[k] += res[k]
        overall["excess"] += res["excess"]; overall["plies"] += res["plies"]

    mr = 100 * overall["mate"] / max(1, overall["n"])
    print(f"VERDICT MATE-FROM-FIELD: mate-rate {mr:.1f}% ({overall['mate']}/{overall['n']}) "
          f"| median plies {int(np.median(overall['plies'])) if overall['plies'] else -1} "
          f"| median excess {float(np.median(overall['excess'])) if overall['excess'] else float('nan'):+.0f} "
          f"| [{time.time()-t0:.0f}s]", flush=True)
    tb.close()


def _gives_mate(board, move):
    board.push(move); m = board.is_checkmate(); board.pop(); return m


if __name__ == "__main__":
    main()
