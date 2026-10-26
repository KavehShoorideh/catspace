"""PLAN 73.5b ACCEPTANCE (b): the mined measure against the HAND-MADE one, read only after the generator had chosen (blind). Both measures are put to the SAME
test -- the amended criterion on the same positions (the graph's min-max: at the floor, or a drop within d against every reply) -- and both are PLAYED from the
same starts against the table's slowest defence: the mined measure by the technique question's own routes (technique_question.technique_routes, no agenda), the
hand-made one by its own driver (mating.rs's frame move, `analyst_core.kq_move_of`, and its measure's components read by `analyst_core.component_values`).

  .venv/bin/python scripts/technique_compare.py KQvK --mined "(lex ...)" --hand fn:kq_offedge,fn:kq_approach,fn:kq_region --driver kq_move_of [--whole-class]

Printed: per (d, n) the failures of each measure (OBS), and per seed the play-out of each (result, plies, our moves, the moves where the technique was silent).
"""
from __future__ import annotations

import argparse
import random
import time

import chess
import numpy as np

import analyst_core as A
from catspace.incubator.analyst.discovery import technique_corpus as TC
from catspace.incubator.analyst.discovery import technique_miner as MN
from catspace.incubator.analyst.discovery import technique_question as TQ
from catspace.incubator.analyst.discovery import technique_sample as TS
from catspace.io import paths


def criterion(sig: str, rows: list, mined: str, hand: list, log=print) -> dict:
    """The failures of the mined measure and of the hand-made one (its components as a lexicographic tuple, mixed-radix) per (d, n) on `rows`."""
    t0 = time.monotonic()
    g = MN.Graph(chess.WHITE)
    roots = [g.add(chess.Board(r["fen"])) for r in rows]
    g.grow(roots, max(MN.DEPTHS))
    A_, *_ = g.arrays()
    m = MN.measure_of(mined)
    cols = np.array(A.component_values(g.fens, hand), dtype=np.int64)
    assert cols.max() < MN.RADIX
    hv = np.zeros(len(g.fens), dtype=np.int64)
    for j in range(cols.shape[1]):
        hv = hv * MN.RADIX + cols[:, j]
    out = {}
    for name, Mv in (("mined " + mined, m.column(A_)), ("hand-made (" + ", ".join(hand) + ")", hv)):
        for d in MN.DEPTHS:
            gd = g.solve(Mv, d)
            nd = [j for j, i in enumerate(roots) if not gd[i] < Mv[i]]
            for n in MN.NETS:
                f = [j for j in nd if rows[j][f"floor{n}"] is None]
                out[(name, d, n)] = f
                log(f"OBS compare {sig} {name}: d={d} n={n}: failures {len(f)} of {len(rows)}" + (f" (e.g. {rows[f[0]]['fen']})" if f else ""))
    log(f"OBS compare {sig}: the same criterion on {len(rows)} positions, the graph {len(g.fens)} nodes; {time.monotonic() - t0:.1f} s")
    return out


def play(sig: str, seed: int, mover, label: str, log=print) -> dict:
    """One play-out from play_technique.py's start of `seed` (the class's median DTZ floor): `mover(board, state) -> (uci or None, note)` for our side, the table's
    slowest line for theirs; the game's rules end it."""
    floor = TC.dtz_floor(sig)[0]
    board = TC.random_start(random.Random(seed), sig, floor)
    start, me = board.fen(), board.turn
    state: dict = {}
    ours, silent = 0, []
    while not board.is_game_over(claim_draw=True):
        if board.turn == me:
            u, note = mover(board, state)
            if u is None:
                silent.append((len(board.move_stack), note))
                u = next(iter(board.legal_moves)).uci()                       # (silent: the first legal move, said; the play-out records it)
            board.push(chess.Move.from_uci(u))
            ours += 1
        else:
            board.push(TC.defender_move(board))
    why = TC._result(board)
    line = (f"OBS compare play {sig} {label} seed {seed}: start {start}; result {board.result(claim_draw=True)} ({why}) after {len(board.move_stack)} plies; our moves {ours}; "
            f"silent {len(silent)}" + (f" ({silent[:3]})" if silent else ""))
    log(line)
    return dict(result=why, plies=len(board.move_stack), ours=ours, silent=silent)


def mined_mover(sig: str, mined: str, d: int, n: int):
    entry = dict(name="mined (compare)", kind="TECHNIQUE", measure={"quantity": mined, "floor": "", "terminal": "", "class": sig, "depth": d, "n": n})

    def mover(board: chess.Board, state: dict):
        hist = state.setdefault("history", set())
        got = TQ.technique_routes(board, entry, history=frozenset(hist), carried=state.get("carry"))
        hist.add(" ".join(board.fen().split()[:4]))
        state["carry"] = got["next"]
        return (got["routes"][0][0] if got["routes"] else None), got["kind"]
    return mover


def frame_mover(fn: str):
    f = getattr(A, fn)

    def mover(board: chess.Board, state: dict):
        got = f(board.fen())
        return (got[0] if got else None), fn
    return mover


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("signature")
    ap.add_argument("--mined", required=True, help="the mined measure (a term of technique_miner's library)")
    ap.add_argument("--hand", required=True, help="the hand-made measure's components, comma separated (vocab.rs `comp` specs)")
    ap.add_argument("--driver", required=True, help="the hand-made technique's move (an analyst_core function fen -> (uci, ...))")
    ap.add_argument("--d", type=int, default=3)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--whole-class", action="store_true")
    a = ap.parse_args()
    sig = a.signature
    rows = TS.whole_class(sig) if a.whole_class else TS.sample(sig, paths.experiment("technique_corpus", f"{sig}_seed1.jsonl"))
    criterion(sig, rows, a.mined, a.hand.split(","))
    for s in (int(x) for x in a.seeds.split(",")):
        play(sig, s, mined_mover(sig, a.mined, a.d, a.n), f"mined {a.mined} (d {a.d}, n {a.n})")
        play(sig, s, frame_mover(a.driver), f"hand-made {a.driver}")


if __name__ == "__main__":
    main()
