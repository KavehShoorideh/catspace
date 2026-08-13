#!/usr/bin/env python
"""search_sanity.py -- the OBVIOUS-SOLUTION battery (Kaveh 2026-08-11: 'set up board positions
that should have an obvious solution and see if it is found... make sure everything is built
properly before running a big job'). Every search mode must pass before it earns compute.

    .venv/bin/python -m ...search_sanity --ckpt <field.pt> [--budget 1.5]
"""
from __future__ import annotations

import argparse
import time

import chess

from catspace.research.components.planner.approaches.quasimetric_nav.kittychess import KittyChess

# (name, fen, acceptable move ucis)
CASES = [
    ("mate in 1 (back rank)", "6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1", {"d1d8"}),
    ("mate in 1 (queen)", "7k/6pp/8/8/8/8/6PP/K2Q4 w - - 0 1", {"d1d8"}),
    ("punish the exposed queen", "rnb1kbnr/pppp1ppp/8/4p3/4q3/5N2/PPPPPPPP/RNBQKB1R w KQkq - 0 1", None),
    ("recapture the knight", "r1bqkb1r/pppp1ppp/2n2n2/4N3/4P3/8/PPPP1PPP/RNBQKB1R b KQkq - 0 4", {"c6e5", "f6e4"}),
    ("promote (or TB-equal)", "8/4P3/8/8/8/2k5/8/4K3 w - - 0 1", None),
    ("avoid stalemate, keep winning", "1R6/k7/8/2N5/4PBP1/p7/P1P3P1/2K5 w - - 0 43", None),
    ("block the mate threat", "6k1/5ppp/8/8/8/8/1r4PP/R5K1 w - - 0 1", None),
    ("escape the attacked queen", "rnb1kbnr/pppp1ppp/8/8/3q4/2P5/PP1PPPPP/RNBQKBNR b KQkq - 0 1",
     {"d4d8", "d4d6", "d4e5", "d4c5", "d4b6", "d4a4", "d4e4", "d4f4", "d4g4", "d4h4", "d4d5", "d4f6", "d4c4", "d4d3", "d4e3"}),
    # HOLD THE DRAW while down a queen-for-rook (TB-exact fortress; the E-conditional-margin
    # era showed nothing gated draw-SEEKING when lost -- 2026-08-12)
    ("hold the fortress (down Q for R)", "8/8/8/2K1kq2/8/3R4/8/8 w - - 0 1", None),
    ("mate in 2 (ladder)", "7k/8/8/8/8/8/1R6/K1R5 w - - 0 1", {"b2b7", "c1c7", "b2h2", "c1h1", "b2g2", "c1g1"}),
    ("win the skewered rook", "8/8/8/3k4/8/3r4/3B4/3K3R w - - 0 1", None),
]


def check(name, b, mv, eng, sf=None):
    """None-expected cases get referee-verified: the move must not tank the eval."""
    return mv is not None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--budget", type=float, default=1.5)
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()
    eng = KittyChess(args.ckpt, args.device)
    import chess.engine
    from catspace.io import paths
    sf = chess.engine.SimpleEngine.popen_uci("stockfish")
    sf.configure({"Threads": 1, "SyzygyPath": str(paths.syzygy_dir())})

    def referee_ok(b, mv):
        """for open-ended cases: the move must stay within 0.15 expected points of SF-best."""
        import math
        def ev(bb):
            info = sf.analyse(bb, chess.engine.Limit(nodes=120_000))
            if "wdl" in info:
                w, d, l = tuple(info["wdl"].pov(b.turn)); return (w + 0.5 * d) / 1000.0
            sc = info["score"].pov(b.turn)
            return 1.0 if (sc.is_mate() and sc.mate() > 0) else (0.0 if sc.is_mate()
                    else 1 / (1 + math.exp(-sc.score() / 200)))
        best = ev(b)
        b.push(mv)
        info = sf.analyse(b, chess.engine.Limit(nodes=120_000))
        if "wdl" in info:
            w, d, l = tuple(info["wdl"].pov(not b.turn)); after = (w + 0.5 * d) / 1000.0
        else:
            sc = info["score"].pov(not b.turn)
            import math as _m
            after = 1.0 if (sc.is_mate() and sc.mate() > 0) else (0.0 if sc.is_mate()
                     else 1 / (1 + _m.exp(-sc.score() / 200)))
        b.pop()
        return after >= best - 0.15

    def mode_move(mode, b):
        t0 = time.time()
        if mode == "recursive-d2":
            rows = eng.search(b, depth=2)
        elif mode == "recursive-d3":
            rows = eng.search(b, depth=3)
        elif mode == "wave":
            rows = eng.search_wave(b, budget=args.budget)
        elif mode == "coherent":
            rows = eng.search_coherent(b, budget=args.budget)
        return (rows[0]["mv"] if rows else None), time.time() - t0

    modes = ["recursive-d2", "recursive-d3", "wave", "coherent"]
    results = {m: [] for m in modes}
    for name, fen, accept in CASES:
        for m in modes:
            b = chess.Board(fen)
            eng._mcache.clear()
            try:
                mv, dt = mode_move(m, b)
            except Exception as e:
                results[m].append((name, "ERROR", 0)); continue
            if mv is None:
                ok = False
            elif accept is not None:
                ok = mv.uci() in accept
            else:
                ok = referee_ok(b, mv)
            results[m].append((name, "pass" if ok else f"FAIL({mv.uci() if mv else '-'})", dt))
    sf.quit()
    print(f"{'case':28s}" + "".join(f"{m:>16s}" for m in modes))
    for i, (name, fen, _) in enumerate(CASES):
        row = f"{name:28s}"
        for m in modes:
            row += f"{results[m][i][1]:>16s}"
        print(row)
    for m in modes:
        n_ok = sum(1 for r in results[m] if r[1] == "pass")
        t_avg = sum(r[2] for r in results[m]) / len(results[m])
        print(f"[sanity] {m}: {n_ok}/{len(CASES)} pass  avg {t_avg:.2f}s")


if __name__ == "__main__":
    main()
