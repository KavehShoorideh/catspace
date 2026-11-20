"""PLAN 79.10 PART E: THE POSITION TEST OF THE WHOLE WAKE AGAIN, with their goals read after each move of ours, the next goal of a chain, and the names given to the search directly (SYSTEM_SPEC 8 the rule for choosing a move, 7 one way of asking for every goal and the stake, 4; piece: the wake).
A PRINTING SCRIPT: the only place the game's own text appears. The same sixteen positions and the same inputs as whole_79_08_print.py / whole_79_09_print.py (imported; nothing about a position is changed here), at the play clock (0.2 s) and ten times it (2.0 s). The wake is run with the pieces of 79.08
and `after_move` and `direct_names`. The old player is run TWICE for each: at the same clock given, and at EQUAL WALL TIME (`equal_wall.at_wall`: the clock at which the old player's wall is near the wake's; the old player does not spend the clock it is given). It judges nothing: no number is written here
that the engine did not print; nothing is tuned to a position.
Order of work (the owner's): run only after the Rust unit tests and the integration tests passed. Outputs artifacts/experiments/whole_79_10/ (through catspace.io.paths).

  python -m catspace.incubator.analyst.discovery.whole_79_10_print missed|puzzles|all [--seconds 0.2 2.0]
(`job_equal_wall`, below, is the arm the puzzle measure adds; the 300 puzzles are run by the main agent, not here)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cs_engine as CE
from catspace.incubator.analyst.discovery import equal_wall as EW
from catspace.incubator.analyst.discovery import odds_measure as OM
from catspace.incubator.analyst.discovery import subgoal_play_measure as SPM
from catspace.incubator.analyst.discovery import whole_79_08_print as W8
from catspace.incubator.analyst.discovery import tiebreak_measure as TM
from catspace.io import paths

OUT = ("whole_79_10",)
PIECES = W8.PIECES + ["after_move", "direct_names"]


def out_dir() -> Path:
    p = Path(paths.experiment(*OUT))
    p.mkdir(parents=True, exist_ok=True)
    return p


class World(W8.World):
    """The world of 79.08 (read, never refitted) with the wake of 79.10 and the old player at equal wall time."""

    def wake10(self, text: str, seconds: float) -> dict:
        return self.wake(text, seconds, PIECES)

    def old_equal(self, text: str, target: float) -> dict:
        r = EW.at_wall(lambda c: self.old(text, c), target)
        d = dict(r["row"])
        d.update({"clock": r["clock"], "tried": r["tried"], "reached": r["reached"]})
        return d

    def child_value(self, text_moves: tuple, seconds: float) -> dict:
        """The old player's reading of the position after a move: (state text of the position, the moves played): its value (from the side to move there), depth and line-free move."""
        pos = CE.state(OM.VARIANT, text_moves[0], list(text_moves[1]))[0]
        row = CE.solve_equals_batch([(OM.VARIANT, pos, [])], seconds, worth=self.worth, ours_first=False, valuation="odds record", threads=1, ledger_share=TM.LEDGER_SHARE)[0]
        return SPM.equal_set(row)


def job_equal_wall(world: "World", text: str, answer: str, seconds: float) -> dict:
    """THE ARM THE PUZZLE MEASURE ADDS (PLAN 79.10 D): for one puzzle at one clock, the wake, the old player at the same clock given and the old player at the clock whose wall matches the wake's. Returns the moves and the walls; the caller (the main agent's measure) counts."""
    w = world.wake10(text, seconds)
    o = world.old(text, seconds)
    e = world.old_equal(text, w["wall"])
    return {"answer": answer, "clock": seconds, "wake": w["move"], "wake_wall": w["wall"], "old": o["move"], "old_wall": o["wall"], "old_eq": e["move"], "old_eq_clock": e["clock"], "old_eq_wall": e["wall"], "old_eq_reached": e["reached"]}


def say_names_under(log: dict) -> list:
    out = []
    for m in log["moves"]:
        a = m.get("their_after")
        if a and a["names"]:
            out.append(f"      {m['move']}: names for them {' '.join(a['names'])} | {a['said']}")
    return out


def report10(title: str, text: str, w: dict, old: dict, old_eq: dict, answer: "str | None", seconds: float, lines: list) -> dict:
    log = w["log"]
    lines.append(f"=== {title} (clock {seconds} s)")
    lines.append(f"  position: {text}")
    lines.append("  THREAT LIST: " + ("(no row)" if not log["rows"] else ""))
    lines += [W8.say_row(r) for r in log["rows"]]
    lines.append("  GOALS (* the goal a named move serves; the order is the proposer's):")
    lines += [W8.say_goal(x) for x in log["goals"][:24]]
    if log["far_goal"]:
        lines.append(f"  the far goal the chosen subgoal serves: {log['far_goal']}")
    lines.append(f"  THEIR LARGEST STANDING THREAT: {log['their_threat'] or 'none standing'}")
    lines.append(f"  NAMED, OURS (given to the search directly, in order of stake, the answers first): {', '.join(w['named_mine']) or '-'}")
    lines.append(f"  NAMED, THEIRS at the ply after ours (answers to our threats): {', '.join(w['named_theirs']) or '-'}")
    under = say_names_under(log)
    lines.append(f"  NAMED UNDER EACH OF OUR MOVES (their fixes at the position after it; the fact about the move) -- {len(under)} of {len(log['moves'])} moves got names; not reached by the clock: {', '.join(log['after_not_asked']) or 'none'}")
    lines += under
    lines.append("  NAMED MOVES' OWN LINES (read AFTER the search, where the clock allowed): " + ", ".join(f"{k['move']} [{k['verdict']}, net {k['net']}]" for k in log["named_mine"]) + f"; REFUTED: " + ", ".join(f"{r['move']}" for r in log["refuted"]) + "; NOT NAMED OR NOT READ: " + ", ".join(m["move"] for m in log["unlisted"]))
    lines.append(f"  SEARCH: depth {w['depth']}, value {w['value']}, nodes {w['nodes']}; line {' '.join(w['chain'][0]['line'][:10])}; held equal: {' '.join(w['equal'])}")
    lines.append(f"  THE WAKE PLAYS {w['move']}: {log['why']}")
    for c in w["chain"][1:]:
        lines.append(f"    claim [{c['by']}]: {' '.join(c['line'][:10])} -- ended: {c['ended']}")
    top = [m for m in log["moves"] if m["equal"]][:8]
    lines.append("  CANDIDATES HELD EQUAL: " + " | ".join(f"{m['move']} v={m['value']} serves {m['serves'][:2] or '-'} answers {m['answers'][:2] or '-'} reasons {m['reasons_score']}" for m in top))
    lines.append("  TIME BY STEP: " + "; ".join(f"step {s['step']} {s['piece']} {s['seconds']:.3f} s ({s['note']})" for s in w["steps"]))
    lines.append("  NOTES: " + " | ".join(log["notes"]))
    lines.append(f"  OLD PLAYER at the same clock given plays {old['move']} (depth {old['depth']}), wall {old['wall']:.3f} s")
    lines.append(f"  OLD PLAYER at equal wall plays {old_eq['move']} (depth {old_eq['depth']}), clock given {old_eq['clock']:.3f} s, wall {old_eq['wall']:.3f} s against the wake's {w['wall']:.3f} s; matched within the tolerance: {old_eq['reached']}; tried (clock, wall): {[(round(c, 3), round(x, 3)) for c, x in old_eq['tried']]}")
    if answer:
        named = w["move"] in w["named_mine"]
        lines.append(f"  THE ANSWER is {answer}: the wake {'HITS' if w['move'] == answer else 'misses'} it, the old player at the same clock {'HITS' if old['move'] == answer else 'misses'} it, the old player at equal wall {'HITS' if old_eq['move'] == answer else 'misses'} it; the answer is "
                     f"{'held equal by the search' if answer in w['equal'] else 'not among the moves the search holds equal'}{' and named for the root' if answer in w['named_mine'] else ' and not named for the root'}" + (f"; it is a fix under: {[m['serves'] for m in log['moves'] if m['move'] == answer][0]}" if any(m['move'] == answer for m in log['moves']) else ""))
    lines.append("")
    return {"title": title, "text": text, "seconds": seconds, "wake": w, "old": old, "old_equal": old_eq, "answer": answer}


def table_row(d: dict) -> str:
    w, o, e = d["wake"], d["old"], d["old_equal"]
    ans = d["answer"] or "-"
    return f"{d['title'][:48]:48s} | {d['seconds']:>4} s | wake {w['move']} ({w['wall']:.2f} s) | old same clock {o['move']} ({o['wall']:.2f} s) | old equal wall {e['move']} (clock {e['clock']:.2f} s, wall {e['wall']:.2f} s, matched {e['reached']}) | answer {ans}"


def missed_extra(world: "World", d: dict, lines: list) -> None:
    """For the missed puzzle: the fact logged about the taking move and about the answer, what the wake's search read for each, and the old player's value for the position after each (the child where their fix was named), at the clock of the decision."""
    w, log = d["wake"], d["wake"]["log"]
    text = d["text"]
    lines.append(f"  MISSED PUZZLE, WHAT THE SEARCH READ (clock {d['seconds']} s):")
    for mv in (W8.MISSED_OLD, W8.MISSED_ANSWER):
        m = next((x for x in log["moves"] if x["move"] == mv), None)
        if not m:
            continue
        a = m.get("their_after")
        lines.append(f"    {mv}: named for the root {m['named']}, held equal {m['equal']}, the search's null-window answer {m['value']}, its line {' '.join(m['line'][:10])}; serves {m['serves'][:3] or '-'}; fact about the move: {a['said'] if a else '(not read)'}; names for them at the child: {a['names'] if a else '-'}")
        try:
            cv = world.child_value((W8.MISSED, [mv]), d["seconds"])
            lines.append(f"      the position after {mv} read by the old player at the same clock: value {cv['value']} (for the side to move there), depth {cv['depth']}, its move {cv['move']}")
        except Exception as e:   # a position with no legal move has no reading: said
            lines.append(f"      the position after {mv}: no reading ({e})")


def main(argv: "list | None" = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=["missed", "puzzles", "all"])
    ap.add_argument("--seconds", type=float, nargs="+", default=[0.2, 2.0])
    a = ap.parse_args(argv)
    world = World()
    done: list = []
    table: list = []
    for which, pos in (("missed", W8.positions_missed), ("puzzles", W8.positions_puzzles)):
        if a.which not in (which, "all"):
            continue
        lines: list = [f"WHOLE WAKE 79.10, {which}: wake with the pieces {', '.join(PIECES)}; the same inputs as 79.08; setup {world.setup_seconds:.1f} s; one run, directions only; nothing tuned", ""]
        for title, text, answer in pos():
            for sec in a.seconds:
                w = world.wake10(text, sec)
                o = world.old(text, sec)
                e = world.old_equal(text, w["wall"])
                d = report10(title, text, w, o, e, answer, sec, lines)
                if which == "missed" and answer:
                    missed_extra(world, d, lines)
                    lines.append("")
                done.append(d)
                table.append(table_row(d))
        (out_dir() / f"{which}_print.txt").write_text("\n".join(lines))
        print("\n".join(lines))
    (out_dir() / f"decisions_{a.which}.json").write_text(json.dumps(done))
    (out_dir() / f"table_{a.which}.txt").write_text("\n".join(table))
    print("\n".join(table))
    return 0


if __name__ == "__main__":
    sys.exit(main())
