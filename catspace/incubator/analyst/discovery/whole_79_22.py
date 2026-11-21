"""PLAN 79.22 THE WAKE READS LAYER 1'S TAGS AND SHARES ONE ROOT GRAPH (SYSTEM_SPEC 11 the layers, one graph, the history; piece: representation and its use by the wake). The measures of part A and of the guard's speed; no game object is named here (the positions come from whole_79_19_print).
INTERFACE.  builds(world, texts, kind) -> list of OBS lines: for each position the number of graphs the wake builds in one decision and the places that asked (kind "no_goal" | "all_inputs" | "played": a short game of decisions in one slot).
            speed(world, texts, label, reps) -> list of OBS lines: nodes a second of the bare search and of the wake at a fixed depth, and the wall of a whole decision at 0.2 s and 2.0 s clocks, medians of `reps` runs.
Called by: the builder's runs (`python -m catspace.incubator.analyst.discovery.whole_79_22 builds|speed LABEL OUT`).
"""
from __future__ import annotations

from catspace.incubator.analyst.discovery import tiebreak_measure as TM          # first: it sets one thread a process before numpy is loaded

import json
import statistics
import sys
import time

from catspace.incubator.analyst.discovery import whole_79_16 as W16
from catspace.incubator.analyst.discovery import whole_79_18 as W18
from catspace.incubator.analyst.discovery import whole_79_19 as G
from catspace.incubator.analyst.discovery import whole_play as WP

DEPTH = 6


def _decide(w, text: str, kind: str, history: list, slot, seconds: float = None, depth: int = None) -> dict:
    a = W18.NO_GOAL_ARGS
    if kind == "no_goal":
        return json.loads(w.model.wake_decide(text, history, seconds or a["seconds"], depth or a["depth"], a["worth"], {}, w.goal, a["pieces"], 0, 0, None, None, None, None, carry_slot=slot))
    raw = w.model.wake_decide(text, history, seconds or G.SECONDS, depth or G.DEPTH, dict(w.worth), {}, w.goal, W16.CONFIGS[G.CHAIN], 0, 0, w.keys, WP.fuses(), w.chances, w.reasons, TM.LEDGER_SHARE, carry_slot=slot, **G._extra_all())
    return json.loads(raw)


def _says(got: list) -> tuple:
    """(root graphs built, other graphs built, the words that say where from) of one decision's counts."""
    root = sum(n for r, _, n in got if r)
    other = sum(n for r, _, n in got if not r)
    where = ", ".join(f"{'root ' if r else 'other '}{p} x{n}" for r, p, n in got)
    return root, other, where


def builds(w, texts: list, kind: str) -> list:
    """For each position: how many times one decision builds the graph of its root position (and of other positions) and where from. `played`: the second position played on for six decisions of one seat with the history given, one slot."""
    out = []
    roots: list = []
    m = w.model
    if kind in ("no_goal", "all_inputs"):
        for i, t in enumerate(texts if kind == "no_goal" else G.positions(texts, G.ALL_POSITIONS)):
            m.graph_builds_start(t)
            _decide(w, t, kind, [], None)
            root, other, where = _says(m.graph_builds_take())
            out.append(f"OBS 79.22 A builds {kind} position {i}: the root's graph built {root} times in one decision, graphs of other positions {other}; by place: {where}")
            roots.append(root)
        out.append(f"OBS 79.22 A builds {kind}: {len(roots)} positions, the root's graph built {sum(roots)} times in all, median {statistics.median(roots)} a decision")
    else:
        CE = WP.CE
        moves: list = []
        start = texts[1]
        total = []
        for _ in range(12):
            pos, side, legal = CE.state(WP.OM.VARIANT, start, list(moves))[:3]
            if not legal:
                break
            if side == "white":
                hist = [CE.state(WP.OM.VARIANT, start, list(moves[:k]))[4] for k in range(len(moves))]
                m.graph_builds_start(pos)
                d = _decide(w, pos, "all_inputs", hist, "built22")
                root, other, where = _says(m.graph_builds_take())
                total.append(root)
                out.append(f"OBS 79.22 A builds played ply {len(moves)}: the root's graph built {root} times in one decision, graphs of other positions {other}; by place: {where}")
                mv = d["move"] if d["move"] in legal else legal[0]
            else:
                mv = sorted(legal)[-1]
            moves.append(mv)
        out.append(f"OBS 79.22 A builds played: {len(total)} decisions, the root's graph built {sum(total)} times in all, median {statistics.median(total) if total else 0} a decision")
    return out


def speed(w, texts: list, label: str, reps: int = 3) -> list:
    """Nodes a second at a fixed depth (the bare search and the wake) and the wall of a decision at two clocks. Each line is the median over positions of one run; `reps` runs are printed so that the caller takes the median of the runs."""
    out = []
    a = W18.NO_GOAL_ARGS
    m = w.model
    for r in range(reps):
        bare, wake = [], []
        for t in texts:
            t0 = time.perf_counter()
            d = json.loads(m.wake_decide(t, [], 600.0, DEPTH, a["worth"], {}, w.goal, [], 0, 0, None, None, None, None))
            bare.append(d["nodes"] / (time.perf_counter() - t0))
            t0 = time.perf_counter()
            d = json.loads(m.wake_decide(t, [], 600.0, DEPTH, a["worth"], {}, w.goal, a["pieces"], 0, 0, None, None, None, None))
            wake.append(d["nodes"] / (time.perf_counter() - t0))
        out.append(f"OBS 79.22 speed {label} run {r} bare search at fixed depth {DEPTH}, {len(texts)} positions: nodes a second {statistics.median(bare):.0f}")
        out.append(f"OBS 79.22 speed {label} run {r} wake at fixed depth {DEPTH}, {len(texts)} positions: nodes a second {statistics.median(wake):.0f}")
    for secs in (0.2, 2.0):
        for r in range(reps):
            walls, rates = [], []
            for t in texts:
                t0 = time.perf_counter()
                d = json.loads(m.wake_decide(t, [], secs, 64, a["worth"], {}, w.goal, a["pieces"], 0, 0, None, None, None, None))
                walls.append(time.perf_counter() - t0)
                rates.append(d["nodes"] / walls[-1])
            out.append(f"OBS 79.22 speed {label} run {r} wall of a whole decision at a clock of {secs} s, {len(texts)} positions: median {statistics.median(walls) * 1000:.1f} ms; nodes a second {statistics.median(rates):.0f}")
    # the real path of a game at hand: decisions of one seat in one slot with the history given, at a fixed depth (the clock not reached), so that what the game's memory keeps and plays is in the wall
    CE = WP.CE
    for r in range(reps):
        walls, moves = [], []
        start = texts[1]
        for _ in range(12):
            pos, side, legal = CE.state(WP.OM.VARIANT, start, list(moves))[:3]
            if not legal:
                break
            if side == "white":
                hist = [CE.state(WP.OM.VARIANT, start, list(moves[:k]))[4] for k in range(len(moves))]
                t0 = time.perf_counter()
                d = _decide(w, pos, "all_inputs", hist, f"speed22{label}{r}", depth=4)
                walls.append(time.perf_counter() - t0)
                mv = d["move"] if d["move"] in legal else legal[0]
            else:
                mv = sorted(legal)[-1]
            moves.append(mv)
        out.append(f"OBS 79.22 speed {label} run {r} wall of a whole decision of a game at hand (a slot, every optional input, fixed depth 4), {len(walls)} decisions: median {statistics.median(walls) * 1000:.1f} ms")
    return out


def main(argv: "list | None" = None) -> int:
    from catspace.incubator.analyst.discovery import whole_79_14 as W14
    from catspace.incubator.analyst.discovery import whole_79_19_print as P
    a = list(argv if argv is not None else sys.argv[1:])
    if a[0] == "speed" and a[1].startswith("before"):
        # the binding of the parent commit takes no fact declaration: the measure of the build before this plan hands it none
        from catspace.incubator.analyst.discovery import game_store as GS
        _gv = GS.graph_vocabulary
        GS.graph_vocabulary = lambda game: {k: v for k, v in _gv(game).items() if k != "fact_decl"}
    w = W14.world()
    lines = []
    if a[0] == "builds":
        for kind in ("no_goal", "all_inputs", "played"):
            lines += builds(w, P.POSITIONS, kind)
    elif a[0] == "speed":
        lines += speed(w, P.POSITIONS, a[1], int(a[3]) if len(a) > 3 else 3)
    for l in lines:
        print(l)
    if len(a) > 2:
        with open(a[2], "a") as f:
            f.write("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
