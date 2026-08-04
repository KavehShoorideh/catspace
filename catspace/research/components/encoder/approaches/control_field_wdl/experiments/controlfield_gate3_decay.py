#!/usr/bin/env python
"""catspace/research/components/encoder/approaches/control_field_wdl/experiments/controlfield_gate3_decay.py -- Kaveh's reframing of gate 3
(2026-08-02): a gambit's compensation is not a static property of the position
right after the sacrifice, it's a property of whether the WDL/committor holds up
over the FOLLOWING moves. Even a genuinely good sacrifice should see its winning
probability decay if the attacker fails to find the continuation; it should hold
(or grow) if they keep finding forcing/cone-building moves. The original gate 3
(catspace/research/components/encoder/approaches/control_field_wdl/experiments/controlfield_gates.py::gate3_gambits) only compared one static
snapshot -- this is the dynamic replacement.

Design: from each gambit's "accepted" position (sacrificer just completed the
setup), run TWO branches for the sacrificer's next K own-moves:
  PRESSING : sacrificer plays Stockfish's own top move each turn (multipv=1,
             depth 12) -- "finds the continuation."
  DRIFTING : sacrificer plays a deliberately passive/non-developing legal move
             each turn (heuristic: a king-side pawn shuffle move that doesn't
             capture, check, or develop a piece, chosen as the lowest-ranked
             non-immediately-losing move by the same multipv scan) --
             "loses the thread."
In both branches, the OPPONENT plays Stockfish's own top move (best defense) --
only the sacrificer's move-finding quality differs between branches.

At each of the sacrificer's turns, record: SF committor (WDL win-fraction,
sacrificer POV, depth 12) and whether the ACTUAL move played was in the
sacrificer's own ascent cone K(s) (tau=0, king_zone target).

Prediction under Kaveh's reframing: PRESSING commitor trend >= DRIFTING
committor trend, and in-cone occupancy should track which branch is which.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import chess
import chess.engine
import numpy as np

from catspace.research.components.encoder.approaches.control_field_wdl.src.derivative import ascent_cone, ConeConfig   # noqa: E402
from catspace.research.components.encoder.approaches.control_field_wdl.experiments.controlfield_gates import GAMBITS                     # noqa: E402


def committor(eng, board, depth, pov):
    info = eng.analyse(board, chess.engine.Limit(depth=depth))
    w = info["wdl"].pov(pov)
    tot = max(1, w.wins + w.draws + w.losses)
    return w.wins / tot


def best_move(eng, board, depth):
    info = eng.analyse(board, chess.engine.Limit(depth=depth), multipv=1)
    return info[0]["pv"][0]


def worst_reasonable_move(eng, board, depth, sacrificer_color):
    """'Drifting': the lowest-ranked legal move that doesn't immediately hang
    mate-in-1 or drop the game outright -- i.e. a plausible human "lost the
    thread" choice, not a random blunder. Ranks all legal moves via multipv and
    takes the worst one whose eval doesn't collapse below a floor relative to
    the best move (so this is "passive", not "actively suicidal")."""
    legal = list(board.legal_moves)
    info = eng.analyse(board, chess.engine.Limit(depth=depth), multipv=len(legal))
    scored = []
    for pv in info:
        if not pv.get("pv"):
            continue
        wdl = pv.get("wdl")
        if wdl is None:
            continue
        w = wdl.pov(sacrificer_color)
        tot = max(1, w.wins + w.draws + w.losses)
        scored.append((pv["pv"][0], w.wins / tot))
    if not scored:
        return legal[0]
    best_c = scored[0][1]
    floor = best_c - 0.35   # "passive" tolerance -- don't pick outright blunders
    candidates = [mv for mv, c in scored if c >= floor]
    return candidates[-1] if candidates else scored[-1][0]


def run_branch(eng, start_board, sacrificer_color, mode, k_moves, depth):
    board = start_board.copy()
    committors, in_cone_flags = [], []
    for _ in range(k_moves):
        if board.is_game_over():
            break
        if board.turn == sacrificer_color:
            c = committor(eng, board, depth, sacrificer_color)
            committors.append(c)
            out = ascent_cone(board, cone_cfg=ConeConfig(target_mode="king_zone"))
            if mode == "pressing":
                mv = best_move(eng, board, depth)
            else:
                mv = worst_reasonable_move(eng, board, depth, sacrificer_color)
            in_cone = mv in out["moves"] and out["in_cone"][out["moves"].index(mv)]
            in_cone_flags.append(bool(in_cone))
            board.push(mv)
        else:
            mv = best_move(eng, board, depth)
            board.push(mv)
    return committors, in_cone_flags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k-moves", type=int, default=5)
    ap.add_argument("--depth", type=int, default=12)
    args = ap.parse_args()
    t0 = time.time()
    eng = chess.engine.SimpleEngine.popen_uci(shutil.which("stockfish") or "/opt/homebrew/bin/stockfish")
    try:
        eng.configure({"UCI_ShowWDL": True})
    except Exception:
        pass

    rows = []
    for name, acc, dec in GAMBITS:
        board = chess.Board()
        for mv in acc:
            board.push(chess.Move.from_uci(mv))
        sacrificer = chess.WHITE   # all three GAMBITS entries are White gambits
        print(f"=== {name} ===", flush=True)
        c_press, ic_press = run_branch(eng, board, sacrificer, "pressing", args.k_moves, args.depth)
        c_drift, ic_drift = run_branch(eng, board, sacrificer, "drifting", args.k_moves, args.depth)
        print(f"  pressing committor: {[f'{c:.3f}' for c in c_press]} | in-cone: {ic_press}")
        print(f"  drifting committor: {[f'{c:.3f}' for c in c_drift]} | in-cone: {ic_drift}")
        trend_press = c_press[-1] - c_press[0] if len(c_press) > 1 else float("nan")
        trend_drift = c_drift[-1] - c_drift[0] if len(c_drift) > 1 else float("nan")
        rows.append(dict(name=name, c_press=c_press, c_drift=c_drift,
                          ic_press=ic_press, ic_drift=ic_drift,
                          trend_press=trend_press, trend_drift=trend_drift))
        print(f"  trend: pressing {trend_press:+.3f} vs drifting {trend_drift:+.3f} | "
              f"{'CONFIRMS (pressing holds/grows more than drifting)' if trend_press > trend_drift else 'DOES NOT CONFIRM'}",
              flush=True)

    n_confirm = sum(1 for r in rows if r["trend_press"] > r["trend_drift"])
    print(f"\nVERDICT gate3-decay: {n_confirm}/{len(rows)} gambits show pressing-committor-trend "
          f"> drifting-committor-trend (Kaveh's reframed hypothesis)")
    press_in_cone_rate = np.mean([f for r in rows for f in r["ic_press"]])
    drift_in_cone_rate = np.mean([f for r in rows for f in r["ic_drift"]])
    print(f"VERDICT gate3-decay-cone-check: in-cone rate pressing={press_in_cone_rate:.1%} "
          f"drifting={drift_in_cone_rate:.1%} | "
          f"{'cone tracks pressing' if press_in_cone_rate > drift_in_cone_rate else 'cone does NOT track pressing'}")

    eng.quit()
    print(f"DONE controlfield_gate3_decay [{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
