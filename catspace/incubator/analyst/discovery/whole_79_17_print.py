"""PLAN 79.17 PRINTS (SYSTEM_SPEC 7; piece: the wake): the tables of part A and part E side by side, and twenty decisions of the games of part E with the held goal read as words (the position, the measure, the line asked for, the move, why). A printing script: it may use the game's renderer; it decides nothing and writes nothing but its own text.
  .venv/bin/python -m catspace.incubator.analyst.discovery.whole_79_17_print tables [--a TAG --e TAG]      part A beside part E: class, attacker, defender, clock, games, wins, mean plies, ended by repetition, by the quiet-ply rule
  .venv/bin/python -m catspace.incubator.analyst.discovery.whole_79_17_print twenty --tag T [--cell NAME]  twenty decisions from twenty games of a cell with the goal held, the board drawn
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import chess

from catspace.incubator.analyst.discovery import whole_79_17 as W17
from catspace.io import paths


def root_of(tag: str) -> Path:
    return Path(paths.experiment(*W17.OUT, tag))


def cells(tag: str, phase: str) -> dict:
    out = {}
    for f in sorted(root_of(tag).glob("cell_*.json")):
        name = f.stem[len("cell_"):]
        m = re.match(r"^(\w)_(\w+?)_(wake_goal|wake|old)_v_(\w+)_([\d.]+)$", name)
        if not m:
            continue
        ph, cls, att, dfn, sec = m.groups()
        if ph != phase:
            continue
        out[(cls, att, dfn, float(sec))] = json.loads(f.read_text())
    return out


def row(key: tuple, s: "dict | None") -> str:
    if s is None:
        return "-"
    e = s["ended"]
    return f"{s['games']:>3} games {s['wins']:>3} wins mean plies {s['mean_plies']:>5.1f} repetition {e.get('repetition', 0):>3} quiet-ply rule {e.get('quiet_rule', 0):>3} (ply cap {e.get('cap', 0)}, stalemate {e.get('stalemate', 0)}, unit lost {e.get('unit_lost', 0)})"


def cmd_tables(a) -> None:
    ca, ce = cells(a.a, "A"), {(k[0], "wake", k[2], k[3]): v for k, v in cells(a.e, "E").items() if k[1] == "wake_goal"}          # part E's attacker is the wake with the goal held: it stands beside the wake of part A
    keys = sorted(set(ca) | set(ce), key=lambda k: (k[0], k[3], k[2], k[1]))
    print(f"{'class':<7}{'attacker':<11}{'defender':<9}{'clock':<6}| part A {a.a}  |  part E {a.e}")
    for k in keys:
        print(f"{k[0]:<7}{k[1]:<11}{k[2]:<9}{k[3]:<6}| {row(k, ca.get(k))}  |  {row(k, ce.get(k))}")


def board_text(text: str, moves: list) -> str:
    b = chess.Board(text)
    for m in moves:
        b.push_uci(m)
    return str(b)


def cmd_twenty(a) -> None:
    root = root_of(a.tag)
    dirs = sorted((root / "logs").glob(a.cell or "*wake_goal*"))
    picked = 0
    for d in dirs:
        for name, decs, res in W17.decisions_of(d):
            if picked >= 20:
                return
            rows = [x for x in decs if x["wake"]["log"].get("measure")]
            if len(rows) < 3:
                continue
            x = rows[len(rows) // 2]
            m = x["wake"]["log"]["measure"]
            print(f"--- {d.name} {name} ply {x['ply']}")
            print(x["position"])
            print(chess.Board(x["position"]))
            print(f"measure {m['term']} value now {m['value_now']} lowest held {m['held_after']} subgoals reached so far {m['reached_total']} plies since lowered {m['since']}")
            print(f"asked: a line of at most {m['d']} of our moves lowers it against every reply: first move {m['first']} in {m['moves']} moves (game end in it: {m['game_end']}, cut by the clock: {m['cut']}, {m['nodes']} nodes, {m['seconds']:.3f} s)")
            print(f"played {x['played']} (toward the goal: {m['played_toward']}); the search's own first move {x['search_first']}; depth {x['wake']['depth']}; why: {x['wake']['log']['why']}")
            print(f"said: {m['said']}")
            picked += 1


def main(argv: "list | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stage", choices=("tables", "twenty"))
    ap.add_argument("--a", default="full")
    ap.add_argument("--e", default="full")
    ap.add_argument("--tag", default="full")
    ap.add_argument("--cell", default=None)
    a = ap.parse_args(argv)
    {"tables": cmd_tables, "twenty": cmd_twenty}[a.stage](a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
