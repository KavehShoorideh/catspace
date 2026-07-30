#!/usr/bin/env python
"""experiments/play_traced.py -- the traced engine, end to end: recognize (memory)
-> verify (honest, refutation-friendly) -> commit (or value fallback), with the
per-move TRACE as the product.

Modes:
  --fen "<fen>"        one decision: pretty-print the trace, optionally --fig a
                       panel (current position + each candidate trap exemplar,
                       outlined by verification verdict)
  --games N            play vs maia-<elo>; traces to --trace-out (JSONL) +
                       VERDICT lines (trap proposals / confirmations / refutation
                       reasons / score)

Bank: build once per encoder with
  python -c "from catspace.memory.checkpoint_bank import build; build('data/derived/checkpoints/checkpoints_v1_full.npz','data/derived/checkpoints/bank_trunk.npz')"
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import chess
import chess.engine
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from catspace.memory.checkpoint_bank import CheckpointBank              # noqa: E402
from catspace.planner.trap_trace import TrapTracePlanner                # noqa: E402
from catspace.train.scaffold import resolve_device                      # noqa: E402
from catspace.predictor.value import CommittorGreedy                    # noqa: E402


def make_planner(args, dev):
    bank = CheckpointBank(args.bank)
    cg = CommittorGreedy(args.ckpt, dev)

    if bank.encoder == "trunk":
        from catspace.encoder import ReachabilityField
        from lczerolens import LczeroBoard
        rf = ReachabilityField()

        def embed(boards):
            return rf.phi([b if isinstance(b, LczeroBoard) else LczeroBoard(b.fen())
                           for b in boards]).cpu().numpy()
    else:
        import torch
        from catspace.encoder.jepa import JepaT1, tokenize
        ck = torch.load(bank_meta_ckpt(args.bank), map_location=dev, weights_only=False)
        model = JepaT1(**{k: ck["cfg"][k] for k in ("d", "layers", "n_class")}).to(dev)
        model.load_state_dict(ck["state_dict"]); model.eval()

        def embed(boards):
            tg = [tokenize(b) for b in boards]
            with torch.no_grad():
                return model.enc(
                    torch.as_tensor(np.stack([t for t, _ in tg])).to(dev),
                    torch.as_tensor(np.stack([g for _, g in tg])).to(dev)).cpu().numpy()

    def committor(boards):
        return cg._committor([b.to_input_tensor().float().numpy()
                              if hasattr(b, "to_input_tensor") else
                              _planes(b) for b in boards])

    from lczerolens import LczeroBoard

    def _planes(b):
        return LczeroBoard(b.fen()).to_input_tensor().float().numpy()

    return TrapTracePlanner(bank, embed, committor, eps=args.eps)


def bank_meta_ckpt(path):
    return str(np.load(path, allow_pickle=True)["meta_ckpt"])


def pretty(trace):
    lines = [f"position: {trace['fen']}",
             f"committor (white POV): {trace['committor_now']:.3f} | "
             f"verification: {trace['verification']}"]
    for i, c in enumerate(trace["candidates"]):
        v = c["verify"]
        lines.append(f"  trap {i+1}: seen {c['support']}x (agreement "
                     f"{c['agreement']:.0%}), ~{c['med_gap']:.0f} decisions out, "
                     f"swing {c['med_delta']:.2f} -> {v['verdict']}: {v['reason']}"
                     + (f" [move {v['best_move']}, concedes {v['concession']}]"
                        if v["verdict"] == "CONFIRMED" else ""))
    d = trace["decision"]
    lines.append(f"  DECISION [{d['source']}]: {d['move']} — {d['rationale']}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", default="data/derived/checkpoints/bank_trunk.npz")
    ap.add_argument("--ckpt", default="artifacts/experiments/field_fullgame_v3_final.pt")
    ap.add_argument("--eps", type=float, default=0.05)
    ap.add_argument("--fen", default="")
    ap.add_argument("--fig", default="")
    ap.add_argument("--games", type=int, default=0)
    ap.add_argument("--maia-elo", type=int, default=1100)
    ap.add_argument("--opening-plies", type=int, default=4)
    ap.add_argument("--max-plies", type=int, default=200)
    ap.add_argument("--trace-out", default="artifacts/experiments/traces.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    dev = resolve_device("auto"); rng = np.random.default_rng(args.seed)
    planner = make_planner(args, dev)

    if args.fen:
        from lczerolens import LczeroBoard
        mv, trace = planner.decide(LczeroBoard(args.fen))
        print(pretty(trace))
        if args.fig:
            from tools import figlib
            from tools.fig_retrieval import draw_board
            n = len(trace["candidates"])
            fig, axes = figlib.new_fig(n + 1, 1, w=1.9, h=2.05)
            axes = np.atleast_1d(axes)
            draw_board(axes[0], trace["fen"], figlib.INK, "current")
            for j, c in enumerate(trace["candidates"]):
                ok = c["verify"]["verdict"] == "CONFIRMED"
                draw_board(axes[j + 1], c["exemplar_fen"],
                           "#2FA089" if ok else "#C0392B",
                           f"{c['verify']['verdict'].lower()} ({c['support']}x)")
            figlib.save(fig, args.fig, f"Trace — {trace['decision']['source']}: "
                                       f"{trace['decision']['move']}")
        return

    maia = chess.engine.SimpleEngine.popen_uci(
        ["lc0", f"--weights=data/engines/maia/maia-{args.maia_elo}.pb.gz",
         "--backend=eigen"])
    from lczerolens import LczeroBoard
    W = D = L = 0; stats = Counter(); t0 = time.time()
    out = open(args.trace_out, "w")
    for g in range(args.games):
        board = LczeroBoard(); our_white = g % 2 == 0
        for _ in range(args.opening_plies):
            ms = list(board.legal_moves)
            board.push(ms[rng.integers(0, len(ms))])
        ply = board.ply()
        while not board.is_game_over(claim_draw=True) and ply < args.max_plies:
            if board.turn == (chess.WHITE if our_white else chess.BLACK):
                mv, trace = planner.decide(board)
                trace["game"] = g; trace["ply"] = ply
                out.write(json.dumps(trace) + "\n")
                stats["decisions"] += 1
                stats[trace["decision"]["source"]] += 1
                for c in trace["candidates"]:
                    stats["proposed"] += 1
                    stats["confirmed" if c["verify"]["verdict"] == "CONFIRMED"
                          else "refuted"] += 1
            else:
                mv = maia.play(board, chess.engine.Limit(nodes=1)).move
            if mv is None:
                break
            board.push(mv); ply += 1
        res = board.result(claim_draw=True)
        s = 0.5 if res == "1/2-1/2" else (1.0 if (res == "1-0") == our_white else 0.0)
        W += s == 1.0; D += s == 0.5; L += s == 0.0
        print(f"  game {g+1}/{args.games} -> {res} (us {s}) [{time.time()-t0:.0f}s]",
              flush=True)
    maia.quit(); out.close()
    n = args.games
    print(f"VERDICT traced score: {(W + 0.5*D)/n:.3f} (W{W} D{D} L{L} of {n}) "
          f"vs maia-{args.maia_elo}")
    print(f"VERDICT traced explanations: {stats['decisions']} decisions | trap-moves "
          f"{stats['trap']} ({stats['trap']/max(stats['decisions'],1):.0%}) | proposals "
          f"{stats['proposed']} -> confirmed {stats['confirmed']} "
          f"({stats['confirmed']/max(stats['proposed'],1):.0%}), refuted "
          f"{stats['refuted']} — honest refutation is the loop working")


if __name__ == "__main__":
    main()
