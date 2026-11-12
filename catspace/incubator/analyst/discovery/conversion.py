"""PLAN 78.36 THE CONVERSION EXAM. The one-decision exam (exam.py) asks whether the engine's move at a position keeps the exact result class. Conversion asks whether the engine
can CARRY the class through a game: from fixed positions (taken deterministically from the exam's own positions, half win-class and half draw-class when both exist) the game is played
out, the engine (the wake's own decision: `playout_sleep.check`, the same call the exam grades with, the game's record passed as the wake passes it) moving for the side that is to move
at the start, and the OTHER side played by the exact labeller's defence. The game stops at the game's end, when the engine drops the class (the labeller says the class after its move
is lower than the class it started with: the ply and position are recorded), or after `max_plies` plies (engine and opponent moves together).

  THE DEFENCE (training kit: it lives here and in the labeller, never in a decision-path crate, and nothing of it reaches the wake): among the opponent's moves that keep ITS class
  (the labeller's `moves_keep`) the first in sorted order, preferring a move that does not end the game at once when another exists. The exact labeller gives no distance to the end (the
  tables Syzygy reads here are win / draw / loss only), so for a lost side every move "keeps" its class and "most resistant" cannot be told from the labeller; this is the documented
  fallback, and a labeller that offers distances would replace `_defence`.

  convert(exam, *, variant, store, spec, seconds, wake_kw, labeller, n=60, max_plies=80, decide=None) -> dict
  say_conversion(label, result, say=print)     prints `OBS conversion <label>: ...` and up to five example lines of drops
     kept      games in which the engine never dropped the class (a game cut by `max_plies` or by a position the labeller does not cover is kept)
     finished  win-class games that reached the game's own win within `max_plies`
  `decide(problems) -> [{move, ..}]` is the engine's decision on a batch of positions ({start, moves}); the default is playout_sleep.check with the given store, clock and switches.
Game-free: the game is the variant string; the labeller object is given.
"""
from __future__ import annotations

import statistics

from catspace.incubator.analyst.discovery import exact as EX
from catspace.incubator.analyst.discovery import playout_sleep as PS

DEFAULT_N = 60
DEFAULT_MAX_PLIES = 80
EXAMPLES = 5


def pick(exam: dict, n: int = DEFAULT_N) -> list:
    """Indices into `exam['positions']`: half win-class and half draw-class (when both exist; a short class leaves its share to the other), in the exam's own order, deterministic."""
    pos = exam["positions"]
    by = {c: [i for i, p in enumerate(pos) if p["before"] == c] for c in ("win", "draw")}
    classes = [c for c in ("win", "draw") if by[c]]
    if not classes:
        return []
    share = {c: n // len(classes) for c in classes}
    for c in classes[: n - sum(share.values())]:
        share[c] += 1
    chosen = {c: by[c][: share[c]] for c in classes}
    left = n - sum(len(v) for v in chosen.values())
    for c in classes:
        if left <= 0:
            break
        extra = by[c][len(chosen[c]): len(chosen[c]) + left]
        chosen[c] += extra
        left -= len(extra)
    return sorted(i for v in chosen.values() for i in v)


def _defence(labeller, variant: str, start, moves: list, v: "PS.Verdicts") -> "str | None":
    """The opponent's move: the first, in sorted order, of those that keep its class and do not end the game at once (else of those that keep its class); None when the labeller does not cover the position."""
    keep = labeller.moves_keep(variant, start, moves)
    if keep is None:
        return None
    keeping = sorted(m for m, k in keep.items() if k) or sorted(keep)
    for m in keeping:
        if v.seat(start, list(moves) + [m])[1] is None:
            return m
    return keeping[0] if keeping else None


def convert(exam: dict, *, variant: "str | None" = None, store: "str | None" = None, spec=None, seconds: float = 0.2, wake_kw: "dict | None" = None, labeller, n: int = DEFAULT_N,
            max_plies: int = DEFAULT_MAX_PLIES, decide=None, threads: int = 0) -> dict:
    """Play the games out (see the module doc). All games advance together, one ply at a time, so the engine's decisions of a ply are ONE batch.
    -> {games, win_games, draw_games, kept, kept_win, kept_draw, finished, mean_plies, median_drop_ply (None without a drop), drops: [{i, start, moves, ply, move, before, after}],
        max_plies, rows: [{i, before, plies, status, ...}]}. `status`: finished (the game ended), dropped, max_plies, uncovered (the labeller does not cover a position), error (the
        engine gave no move: counted as a drop)."""
    variant = variant or exam["variant"]
    v = PS.Verdicts(labeller, variant)
    if decide is None:
        def decide(problems):
            return PS.check(problems, variant=variant, store=store, spec=spec, judge=labeller, seconds=seconds, wake_kw=wake_kw, threads=threads, verdicts=v)
    chosen = pick(exam, n)
    games = []
    for i in chosen:
        p = exam["positions"][i]
        role = v.seat(p["start"], p["moves"])[0]
        games.append({"i": i, "start": p["start"], "base": len(p["moves"]), "moves": list(p["moves"]), "role": role, "before": p["before"], "status": None, "outcome": None, "drop": None})
    for _ply in range(max_plies):
        live = [g for g in games if g["status"] is None]
        if not live:
            break
        engine, opp = [], []
        for g in live:
            role, outcome = v.seat(g["start"], g["moves"])
            if outcome is not None:
                g["status"], g["outcome"] = "finished", PS.outcome_class(outcome, g["role"])
            elif role == g["role"]:
                engine.append(g)
            else:
                opp.append(g)
        if engine:
            rows = decide([{"start": g["start"], "moves": g["moves"]} for g in engine])
            for g, r in zip(engine, rows):
                mv = r.get("move")
                if mv is None:
                    g["status"], g["drop"] = "error", {"move": None, "after": None}          # (no move from the engine: counted as a drop)
                    continue
                try:
                    after = v.after(g["start"], g["moves"] + [mv], g["role"])
                except EX.NotCovered:
                    g["status"] = "uncovered"
                    continue
                g["moves"].append(mv)
                if PS.rank_of(after) < PS.rank_of(g["before"]):
                    g["status"], g["drop"] = "dropped", {"move": mv, "after": after}
        for g in opp:
            mv = _defence(labeller, variant, g["start"], g["moves"], v)
            if mv is None:
                g["status"] = "uncovered"
                continue
            g["moves"].append(mv)
    for g in games:
        if g["status"] is None:
            role, outcome = v.seat(g["start"], g["moves"])
            if outcome is not None:
                g["status"], g["outcome"] = "finished", PS.outcome_class(outcome, g["role"])
            else:
                g["status"] = "max_plies"
    rows, drops = [], []
    for g in games:
        plies = len(g["moves"]) - g["base"]
        row = {"i": g["i"], "before": g["before"], "plies": plies, "status": g["status"], "outcome": g["outcome"]}
        if g["drop"] is not None:
            d = g["drop"]
            ply = plies if d["move"] is not None else plies + 1          # (the ply of the engine's move: it is the last one played, or the one it could not make)
            row["drop"] = {"ply": ply, **d}
            drops.append({"i": g["i"], "start": g["start"], "moves": g["moves"][: g["base"] + ply - 1] if d["move"] is not None else list(g["moves"]), "ply": ply, "move": d["move"], "before": g["before"], "after": d["after"]})
        rows.append(row)
    dropped = {r["i"] for r in rows if "drop" in r}
    win = [r for r in rows if r["before"] == "win"]
    draw = [r for r in rows if r["before"] == "draw"]
    kept = lambda rs: sum(1 for r in rs if r["i"] not in dropped)  # noqa: E731
    return {"games": len(rows), "win_games": len(win), "draw_games": len(draw), "kept": kept(rows), "kept_win": kept(win), "kept_draw": kept(draw),
            "finished": sum(1 for r in win if r["status"] == "finished" and r["outcome"] == "win"), "mean_plies": (sum(r["plies"] for r in rows) / len(rows)) if rows else 0.0,
            "median_drop_ply": statistics.median(d["ply"] for d in drops) if drops else None, "drops": drops, "max_plies": max_plies, "rows": rows}


def say_conversion(label: str, r: dict, say=print) -> None:
    """`OBS conversion <label>: games G (win-class W, draw-class D); never dropped the class K (win-class KW of W, draw-class KD of D); wins finished F of W within P plies; first drop at ply
    (median) M`, then up to five lines, one per drop (the position and the move that dropped the class)."""
    med = "none" if r["median_drop_ply"] is None else f"{r['median_drop_ply']:g}"
    say(f"OBS conversion {label}: games {r['games']} (win-class {r['win_games']}, draw-class {r['draw_games']}); never dropped the class {r['kept']} (win-class {r['kept_win']} of {r['win_games']}, "
        f"draw-class {r['kept_draw']} of {r['draw_games']}); wins finished {r['finished']} of {r['win_games']} within {r['max_plies']} plies; first drop at ply (median) {med}")
    for d in r["drops"][:EXAMPLES]:
        say(f"OBS conversion example: start {d['start']} after {' '.join(d['moves']) or '-'}: ply {d['ply']}, played {d['move']}, class {d['before']} -> {d['after']}")
