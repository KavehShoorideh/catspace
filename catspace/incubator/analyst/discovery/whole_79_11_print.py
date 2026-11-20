"""PLAN 79.11 PART E: THE JUDGE, TWENTY DECISIONS OF THE GAMES READ AGAINST THE BOARD (SYSTEM_SPEC 8 the whole is judged, 1 and 11: the judge stays outside the engine; piece: the wake).
A PRINTING SCRIPT: the only place of this plan where the game's own text appears. It reads the decision logs the matches wrote (`whole_play`: one gzip file a game), takes the decisions in which the wake's move DIFFERED from the search's first move, and prints twenty of them with the position (the board), the goals,
the names given to the search, the fact about their goals after the move played and after the search's own move, the search's line and the reason the log gives. The twenty are chosen by a rule written here and nothing else: the differing decisions are grouped by the reason given and taken round
robin over the groups, in each group at even steps through the order of (game, ply). It judges nothing: the builder and the main agent read them against the board. No number is written here that the engine did not print.

  python -m catspace.incubator.analyst.discovery.whole_79_11_print [--tag full] [--n 20] [--logs m1_A m1_B m2_A m2_B]
Output: artifacts/experiments/whole_79_11/<tag>/twenty_print.txt (through catspace.io.paths).
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import chess

from catspace.incubator.analyst.discovery import whole_79_08_print as W8
from catspace.incubator.analyst.discovery import whole_play as WP


def differing(log_dirs: list) -> list:
    """Every decision in the logs whose played move differs from the search's first move: [(reason, gid, ply, record)]."""
    out = []
    for d in log_dirs:
        for p in sorted(Path(d).glob("*.jsonl.gz")):
            with gzip.open(p, "rt") as f:
                for line in f:
                    r = json.loads(line)
                    if r["kind"] == "decision" and r["differs"]:
                        out.append((r["reason"], r["gid"], r["ply"], r))
    out.sort(key=lambda x: (x[0], x[1], x[2]))
    return out


def choose_twenty(rows: list, n: int) -> list:
    """The rule: group by reason, round robin over the reasons (in the order of their size, largest first), in a group at even steps through its order."""
    groups: dict = {}
    for r in rows:
        groups.setdefault(r[0], []).append(r)
    order = sorted(groups, key=lambda k: (-len(groups[k]), k))
    left = min(n, len(rows))
    quota = {k: 0 for k in order}
    while left:
        for k in order:
            if left and quota[k] < len(groups[k]):
                quota[k] += 1
                left -= 1
    picked: list = []
    for k in order:
        g, q = groups[k], quota[k]
        picked += [g[(j * len(g)) // q] for j in range(q)]
    return picked


def say_after(rec: dict, mv: str) -> str:
    m = next((x for x in rec["wake"]["log"]["moves"] if x["move"] == mv), None)
    if not m:
        return "(not a candidate in the log)"
    a = m.get("their_after")
    return f"value {m['value']}, held equal {m['equal']}, named for the root {m['named']}, serves {m['serves'][:3] or '-'}, answers {m['answers'][:3] or '-'}, reasons' score {m['reasons_score']}; their goals after it: {a['said'] if a else '(not read)'}; line {' '.join(m['line'][:8])}"


def report(i: int, reason: str, rec: dict, lines: list) -> None:
    w, log = rec["wake"], rec["wake"]["log"]
    board = chess.Board(rec["position"])
    lines.append(f"=== {i}. game {rec['gid']}, seat {rec['seat']}, ply {rec['ply']}, reason given: {reason}; wall {rec['wall']:.3f} s")
    lines.append(f"  position: {rec['position']}")
    lines += ["    " + x for x in str(board).splitlines()]
    lines.append(f"  THE WAKE PLAYED {rec['played']}; THE SEARCH'S FIRST MOVE was {rec['search_first']}; held equal by the search: {' '.join(w['equal'])}")
    lines.append(f"  WHY (the log): {log['why']}")
    lines.append(f"  THEIR LARGEST STANDING THREAT: {log['their_threat'] or 'none standing'}")
    lines.append("  THREAT LIST: " + ("(no row)" if not log["rows"] else ""))
    lines += [W8.say_row(r) for r in log["rows"][:8]]
    lines.append("  GOALS (* the goal a named move serves):")
    lines += [W8.say_goal(x) for x in log["goals"][:12]]
    if log["far_goal"]:
        lines.append(f"  the far goal the chosen subgoal serves: {log['far_goal']}")
    lines.append(f"  NAMED, OURS (to the search, in order of stake): {', '.join(w['named_mine']) or '-'}; NAMED, THEIRS at the ply after ours: {', '.join(w['named_theirs']) or '-'}")
    lines.append(f"  THE MOVE PLAYED, as the log has it: {say_after(rec, rec['played'])}")
    lines.append(f"  THE SEARCH'S FIRST MOVE, as the log has it: {say_after(rec, rec['search_first'])}")
    lines.append(f"  SEARCH: depth {w['depth']}, value {w['value']}, nodes {w['nodes']}; line {' '.join(w['chain'][0]['line'][:10]) if w['chain'] else '-'}")
    for c in w["chain"][1:]:
        lines.append(f"    claim [{c['by']}]: {' '.join(c['line'][:10])} -- ended: {c['ended']}")
    lines.append("  NOTES: " + " | ".join(log["notes"]))
    lines.append("")


def main(argv: "list | None" = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="full")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--logs", nargs="+", default=["m1_A", "m1_B", "m2_A", "m2_B"])
    a = ap.parse_args(argv)
    root = WP.out_root(a.tag)
    dirs = [root / "logs" / x for x in a.logs if (root / "logs" / x).exists()]
    rows = differing(dirs)
    by = {}
    for r in rows:
        by[r[0]] = by.get(r[0], 0) + 1
    lines = [f"WHOLE 79.11 PART E: {len(rows)} decisions in which the wake's move differed from the search's first move, in {len(dirs)} folders of logs; by reason {by}; {min(a.n, len(rows))} printed (round robin over the reasons); one run, directions only", ""]
    for i, (reason, _g, _p, rec) in enumerate(choose_twenty(rows, a.n), 1):
        report(i, reason, rec, lines)
    text = "\n".join(lines)
    (root / "twenty_print.txt").write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
