"""PLAN 79.28 PART E: LINES FOR A READER (SYSTEM_SPEC 11 "Reached means the goal outlasts the reply", "Layer 4 takes a goal as an input"; ledger rows 36, 14; piece: tactical reasoner).
A PRINTING SCRIPT (it may name the games' units because it only writes text for a person). It reads the answers layer 4 gave at the top budget (one JSON line each, written by rust/engine/tactics/tests/layer4_goals.rs when LAYER4_ANSWERS_OUT is set) and writes, for
TWELVE REACHED and SIX TOUCHED answers chosen by a FIXED RULE across the kinds of goal, text a person can read against a board: the position, the goal in plain words, the line, the position at its end (for a touched goal also the reply that undoes it and the position after it).
THE RULE (nothing is chosen by hand and nothing by the answer's look): the kinds in alphabetical order; within a kind, the answers of the status asked with at least one ply (a goal reached in nought plies has no line to read), the longest lines first, ties by the position's id; one answer of each kind in turn, round and round,
until the count is reached or no kind has one left. The judge of the lines is the reader (the main agent), who reads them against the board and says what a strong player would not agree with; nothing here judges.

  .venv/bin/python -m catspace.incubator.analyst.discovery.layer4_lines_print ANSWERS_JSONL [OUT_FILE]

OUT_FILE defaults to artifacts/experiments/whole_79_28/lines.txt (catspace.io.paths).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import chess
import chess.variant

from catspace.io import paths

BOARDS = {"chess": chess.Board, "kingofthehill": chess.variant.KingOfTheHillBoard}
WANT = {"reached": 12, "touched": 6}


def board_of(game: str, text: str) -> "chess.Board":
    return BOARDS[game](text)


def pick(rows: list, status: str, count: int) -> list:
    """The fixed rule: kinds alphabetical, longest lines first (ties by position id), one of each kind in turn."""
    by_kind: dict = {}
    for r in rows:
        if r["status"] == status and r.get("plies", 0) >= 1:
            by_kind.setdefault(r["kind"], []).append(r)
    for k in by_kind:
        by_kind[k].sort(key=lambda r: (-r["plies"], r["position"], r["goal"]))
    chosen = []
    while len(chosen) < count and any(by_kind.values()):
        for k in sorted(by_kind):
            if by_kind[k] and len(chosen) < count:
                chosen.append(by_kind[k].pop(0))
    return chosen


def words(r: dict, b: "chess.Board") -> str:
    """The goal in plain words, from the kind and the pattern's constants; the placeholders are named by what the line found."""
    me = "white" if b.turn else "black"
    them = "black" if b.turn else "white"
    pat = r["pattern"]
    consts = lambda rel: [l["args"] for l in pat if l["relation"] == rel and not l["neg"]]
    if r["kind"] == "bears":
        _, kind, sq = consts("unit")[0]
        return f"a unit of {me} bears on the {them} {kind} on {sq}"
    if r["kind"] == "two_bear":
        _, kind, sq = consts("unit")[0]
        return f"two different units of {me} bear on the {them} {kind} on {sq}"
    if r["kind"] == "on_line":
        line = consts("on_line")[0][1]
        head, origin = line.split("@place:")
        axis = head.split(":")[1]
        what = "the file" if axis == "north" else "the rank"
        return f"a unit of {me} stands on {what} through {origin} (the graph's {line})"
    if r["kind"] == "on_edge":
        kind = consts("unit")[0][1]
        return f"a {me} {kind} stands on the {consts('on_edge')[0][1]} edge of the board"
    if r["kind"] == "may_take":
        target = [a for a in consts("may_move")][0][3]
        neg = [l["args"] for l in pat if l["neg"]][0]
        return f"a unit of {me} may move to {target} while the {neg[0]} {neg[1]} on {neg[2]} does not bear on it"
    return json.dumps(pat)


def san_line(b: "chess.Board", line: list) -> tuple:
    """The line as numbered moves, and the board after it."""
    c = b.copy()
    out = []
    for u in line:
        mv = chess.Move.from_uci(u)
        out.append(f"{c.fullmove_number}{'.' if c.turn else '...'} {c.san(mv)}")
        c.push(mv)
    return " ".join(out), c


def show(b: "chess.Board") -> str:
    return "\n".join("    " + l for l in str(b).splitlines()) + f"\n    {b.fen()}  ({'white' if b.turn else 'black'} to move)"


def found(r: dict) -> str:
    bs = r.get("bindings") or []
    if not bs:
        return ""
    return "where it binds: " + "; ".join(", ".join(f"{k}={v}" for k, v in sorted(b.items())) for b in bs[:2])


def entry(n: int, r: dict) -> str:
    b = board_of(r["game"], r["text"])
    text, end = san_line(b, r["line"])
    out = [f"--- {n}. {r['status'].upper()} ({r['kind']}), position {r['position']}, {r['plies']} plies; the reference says {r['reference']}",
           "THE POSITION", show(b), f"THE GOAL: {words(r, b)}", f"THE LINE: {text}"]
    if r["status"] == "reached":
        out += ["THE POSITION AT ITS END (the goal holds here, and the side to move is the side that asked, or the game is over)", show(end), found(r)]
    else:
        out += ["THE POSITION AT THE LINE'S END (the goal binds here, after a move of the side that asked)", show(end), found(r)]
        mv = chess.Move.from_uci(r["reply"])
        reply = f"{end.fullmove_number}{'.' if end.turn else '...'} {end.san(mv)}"
        after = end.copy()
        after.push(mv)
        out += [f"THE REPLY THAT UNDOES IT: {reply}", "THE POSITION AFTER THE REPLY (the goal no longer binds)", show(after)]
    return "\n".join(x for x in out if x)


def main(argv: "list | None" = None) -> int:
    args = argv or sys.argv[1:]
    rows = [json.loads(l) for l in Path(args[0]).read_text().splitlines() if l.strip()]
    out = Path(args[1]) if len(args) > 1 else Path(paths.experiment("whole_79_28", "lines.txt"))
    chosen = {s: pick(rows, s, n) for s, n in WANT.items()}
    head = [f"PLAN 79.28 PART E: layer 4's answers at the top budget ({rows[0]['budget']} nodes, depth 5 plies) for a reader; the rule that chose them is in layer4_lines_print.py",
            f"answers: {len(rows)}; reached {sum(r['status'] == 'reached' for r in rows)}, touched {sum(r['status'] == 'touched' for r in rows)}; shown: {len(chosen['reached'])} reached, {len(chosen['touched'])} touched", ""]
    body = []
    n = 0
    for s in ("reached", "touched"):
        for r in chosen[s]:
            n += 1
            body.append(entry(n, r))
            body.append("")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(head + body))
    print(f"OBS 79.28 lines: {len(chosen['reached'])} reached and {len(chosen['touched'])} touched answers written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
