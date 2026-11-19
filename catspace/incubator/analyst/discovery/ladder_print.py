"""PLAN 78.68 (first run), SYSTEM_SPEC 9: THE SURPRISES AS READABLE CASES. The only place of the ladder's files where the game's renderer and names appear (for the reader of the report, never read by the engine).

INTERFACE. cases(tag, n) takes the run's results, answers and groups under artifacts/experiments/ladder_78_68/<tag>/ and writes cases.txt: n surprises, spread over the groups largest first, each with the board, the
move our tools chose, the moves the oracle accepts (the answers file, read here for the human reader only), what our deepest reading found, and the group. Called by the command line only.

  .venv/bin/python -m catspace.incubator.analyst.discovery.ladder_print [--tag full] [--n 20]

PLAN 78.76 `--records NAME NAME ...` (`records_cases`): the same positions side by side under each measured record of the kinds. THE SET IS FIXED BY A RULE BEFORE ANY RESULT IS READ: the first six open surprises of the 78.68 list
(surprise_list.jsonl of the full run, in file order) and the first six positions, in the order of positions.jsonl, whose rung under some record differs from the first record's. Under each record: the moves the search holds
equal, its first move, the best line and where it ends (each read at the clock that record's ladder stopped at: the wake's for rungs 1 and 2, the clock it was found at for rung 3, the last for an open one), and the accepted
moves (the answers file, for the human reader only). Writes ladder_78_76/cases.txt.

PLAN 78.78 `--pick TAG` (`pick_cases`): THE CASES OF THE SUBGOAL RULE scored on the open surprises (subgoal_pick_measure.py). THE TWELVE ARE FIXED BY RULE BEFORE THE CASES WERE READ (the measure writes them): the first six open
surprises of the scored part whose accepted move is among the moves the search holds equal (set A) and the first six resolved positions the best candidate breaks (set B). Each: the board, the moves held equal, the first five goals
in words, the count of each of them before and after each tied move (own = layer one's count of our own moves, None where it is not defined; w = the stoppers still in the way), the pick of every candidate and the accepted moves
(the answers file, for the human reader only). Reads artifacts/experiments/subgoal_pick_78_78/<tag>/ (preps.pkl, case_rows.json); writes cases.txt beside them.

PLAN 78.86 `--reread` (`reread_cases`): the twenty positions whose rung changed between 78.77's run and the re-read (rule in ladder_reread.pick_changed), each from the stored results: board, units under attack, the earlier and
the new reading. Writes ladder_78_86/cases.txt.
PLAN 78.80 `--symbols TAG` (`symbols_cases`): THE SYMBOLS AS READABLE CASES. Part A: the twelve positions of 78.78 with, per tied move, the passing events (patterns made true or false, for each side) it carries, in words; part B: the ten
strongest passing patterns (a rule fixed in the function's doc) each drawn on three positions where the move played carries the event, the units of one instance in brackets. Reads artifacts/experiments/symbols_78_80/<tag>/ and writes
cases.txt there. The only place of the symbols' files where the game's renderer and names appear.
PLAN 78.81 `--reasons TAG` (`reasons_cases`): THE REASONS AS READABLE CASES (symbols_reasons_measure.py; reasons.json and ahead.json under symbols_78_81/<tag>/, the 78.80 sample, events and keys). Part A: THE STRONGEST TWENTY REASONS by
|weight| in the joint fit with the controls (a rule fixed here before the cases were read), each in words and drawn on the first three CONFIRM positions, in the file's order, where the move played carries the event, the units of one
instance in brackets, with whether the move takes a unit and what kind of unit moves said beside it. Part B: the twelve positions of 78.78 (subgoal_pick_78_78 case_rows.json) with, per tied move, the reasons it carries in words, whether it
takes a unit, the kind of unit that moves and the sum the pick reads; the accepted moves are the answers file, for the human reader only. Writes cases.txt beside the reasons.
PLAN 78.84 `--cut84 TAG` (`cut_cases`): THE TWENTY FLAGGED DECISIONS of part one (tie_cut_measure.py `report`): a tie set with a move whose line gives up a unit for nothing and a move whose line does not, each with the board, the units under attack,
what the search reported, both moves with their lines and numbers, and the cause the measure named. Writes cases.txt under symbols_play_78_84/<tag>/.
PLAN 78.90 `--own90 TAG` (`own_cases`): THE THIRTY DECISIONS where the fit with the other side's cells picks otherwise than the old weights (symbols_own_measure.py `cases`: cases90.json, groups_A.json under symbols_78_90/<tag>/), ten
a band, each with the board, what each side has under attack, the moves that leave a unit of the mover's borne on and unguarded, and for the old pick, the new pick and the first move what it leaves under attack and its groups in words.
PLAN 78.91 `--order91 TAG` (`order_cases`): THE THIRTY POSITIONS of the hint in the search's order (symbols_order_measure.py `cases`: cases91.json, order_78_91/<tag>/; the groups are 78.90's fit A, symbols_78_90/full/groups_A.json): twenty where the
reasons' order puts the reference move (the search at ten times the play clock, root moves in a random order) first among the quiet moves and the static order does not, ten of the opposite. Each with the board, what each side has under attack,
the moves that take, the reference move and its place in both orders, the first five quiet moves of each order, and the strongest groups of the reference, of the static first and of the reasons' first in words. Writes cases.txt.
PLAN 78.92 `--choice92 TAG` (`choice_cases`): THE TWENTY POSITIONS where the search's played move is under the reference's best (choice_measure.py `cases`: cases92.json under reasoner_choice_78_92/<tag>/; the reference is the same search with
the late-move reduction and the null move off): the board, what each side has under attack, the played move, the reference's best and both values at the search's depth and two plies deeper. Writes cases.txt.
PLAN 78.98 `--equal98 TAG` (`equal98_cases`): THE TRACE OF THE EQUAL MOVES (equal_trace.py, trace.jsonl under equal_78_98/<tag>/): the five named positions (trace.txt) or ten more stuck moves of kind all present (cases.txt): the board, the stuck move and a class-keeping move with what the root holds for each, the rule of equality, the lines the values come from and the units each side has at their end.
PLAN 78.99 `--record99 TAG` (`record99_cases`): THE JUDGED POSITIONS (record_measure.py `cases`, cases99.json under record_78_99/<tag>/): the five positions of 78.98 and ten room positions where the first move changes between the odds record and the human-result record, with the move the search plays under each measured record. Writes cases.txt.
PLAN 79.00 `--stack79 TAG` (`stack79_cases`): THE TWENTY JUDGED POSITIONS of the new room (stack_measure.py `cases`, cases79.json under stack_79_00/<tag>/): room positions where the first move of the search under the stack (human-result record, verified choice) loses the oracle's class: the board, what each side has under attack, the first move, the pick, the oracle's line after the first move, the class after it and the equal moves that keep the class. Writes cases.txt (the reader's judgement is appended by hand).
PLAN 79.01 `--leaf01 TAG` (`leaf01_cases`): THE FIFTEEN JUDGED POSITIONS of the firm room (leaf_measure.py `cases`, cases_leaf.json under leaf_79_01/<tag>/): positions where the equal move with the highest estimator chance at the end of the line is not the search's first move: the board, what each side has under attack, every equal move with its firm label, the oracle's two askings, the units each side has at the end of its line, the chances, and the line in words. Writes cases.txt (the reader's judgement is appended by hand).
PLAN 78.82 `--play82 TAG` (`play_cases`): THE THIRTY DECISIONS of the reasons' pick in play where it differed from the search's first move (symbols_play.py `cases`), each with the board and the reasons of both moves in words. Writes cases.txt under symbols_play_78_82/<tag>/.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import chess

from catspace.incubator.analyst.discovery import ladder as LD
from catspace.incubator.analyst.discovery import ladder_measure as LM
from catspace.io import paths


def san_line(b: chess.Board, names: list, cap: int = 8) -> str:
    out, x = [], b.copy()
    for m in names[:cap]:
        try:
            mv = chess.Move.from_uci(m)
            out.append(x.san(mv))
            x.push(mv)
        except Exception:
            out.append(m + "?")
            break
    return " ".join(out)


def our_move(res: dict) -> "str | None":
    ms = [v["move"] for k, v in res["chose"].items() if k.startswith("3 search")]
    return ms[-1] if ms else res["chose"].get("1 search", {}).get("move")


def pick(rows: list, groups: list, n: int) -> list:
    """Round robin over the groups, the largest first, one surprise a turn."""
    queues = [list(g["ids"]) for g in groups]
    out = []
    while len(out) < n and any(queues):
        for q in queues:
            if q and len(out) < n:
                out.append(q.pop(0))
    return out


def deep_cases(tag: str = "full", n: int = 10) -> Path:
    """PLAN 78.68 stage G and H: THE CASES OF THE BEST CANDIDATE as readable text (the board, the tied moves, the move the candidate picks and the line it read after it to its settled end, the moves the oracle accepts, for
    the reader only): `n` open surprises it now picks an accepted move in, and `n` positions where it creates a surprise. Reads deep_heads_78_68/<tag>/case_rows.json (which holds no oracle field) and the ladder's answers
    file here, for the human reader. Writes cases.txt beside it."""
    root = Path(paths.experiment("deep_heads_78_68", tag))
    cr = json.loads((root / "case_rows.json").read_text())
    ans = {d["id"]: d["accepted"] for d in (json.loads(x) for x in (LM.out_root("full") / "answers.jsonl").read_text().splitlines() if x.strip())}
    lines = [f"THE BEST CANDIDATE: {cr['best']}", ""]
    for title, key in (("OPEN SURPRISES THE CANDIDATE NOW PICKS AN ACCEPTED MOVE IN", "resolved"), ("POSITIONS WHERE THE CANDIDATE CREATES A SURPRISE", "created")):
        lines += [f"##### {title} ({min(n, len(cr[key]))} of {len(cr[key])})", ""]
        for k, c in enumerate(cr[key][:n], 1):
            b = chess.Board()
            for m in c["moves"]:
                b.push_uci(m)

            def nm(m: str) -> str:
                return b.san(chess.Move.from_uci(m))
            lines += [f"=== {key.upper()} {k}: {c['id']}  ({'white' if b.turn else 'black'} to move)", str(b), "",
                      f"the search held equal: {', '.join(nm(m) for m in c['tied'])}   (its own first move: {nm(c['search_move'])})",
                      f"the candidate picks: {nm(c['pick'])}; its best line after it, to the settled end: {san_line(b, c['line'] or [], 10)}; the line's end: {c['leaf_end']}",
                      f"the oracle accepts: {', '.join(nm(m) for m in ans[c['id']])}"]
            for m, (ln, end) in c["lines"].items():
                if m != c["pick"]:
                    lines.append(f"   the line after {nm(m)}: {san_line(b, ln or [], 10)} (end {end})")
            lines.append("")
    (root / "cases.txt").write_text("\n".join(lines))
    print(f"OBS ladder print: deep cases written to {root / 'cases.txt'}")
    return root / "cases.txt"


def pick_cases(tag: str = "full", shown: int = 5) -> Path:
    root = Path(paths.experiment("subgoal_pick_78_78", tag))
    preps = pickle.loads((root / "preps.pkl").read_bytes())
    cr = json.loads((root / "case_rows.json").read_text())
    ans = {d["id"]: d["accepted"] for d in (json.loads(x) for x in (LM.out_root("full") / "answers.jsonl").read_text().splitlines() if x.strip())}
    lines = [f"THE BEST CANDIDATE (by net, chosen before the cases were read): {cr['best']}", ""]
    for title, key in (("SET A: OPEN SURPRISES WHOSE ACCEPTED MOVE IS AMONG THE MOVES HELD EQUAL", "A"), ("SET B: RESOLVED POSITIONS THE BEST CANDIDATE BREAKS", "B")):
        lines += [f"##### {title}", ""]
        for k, pid in enumerate(cr[key], 1):
            p = preps[pid]
            b = chess.Board()
            for m in cr["moves"][pid]:
                b.push_uci(m)

            def nm(m: str) -> str:
                return b.san(chess.Move.from_uci(m))
            lines += [f"=== SET {key} CASE {k}: {pid}  ({'white' if b.turn else 'black'} to move)", str(b), "",
                      f"held equal ({len(p['tied'])}), the search's first move first: {', '.join(nm(m) for m in p['tied'])}",
                      f"the accepted moves: {', '.join(nm(m) for m in ans[pid])}   (accepted among the tied: {', '.join(nm(m) for m in p['tied'] if m in ans[pid]) or 'none'})",
                      f"goals read: {len(p['goals'])}; threats against us (rule step 1): {p.get('threats')}; the rule fired branch {p['branch_rule']}"]
            for i, g in enumerate(p["goals"][:shown], 1):
                st = "; ".join(f"{x['step']}: {x['unit'][1] if x['unit'] else '?'} on {x['route'][0] if x['route'] else '?'} by {x['moves']} move(s)" for x in g["steps"]) or "no step"
                lines.append(f"   goal {i}: {g['words']}; source {g['source']}; own count {g['own_moves']}; steps: {st}; stoppers {[f'{u[0]} {u[1]} {u[2]}' for u in g['stoppers']]}")
            t = p["table"]
            lines.append("   the counts after each tied move, own/w per goal (before: " + " ".join(f"{x['own']}/{x['w']}" for x in t["before"][:shown]) + ")")
            for m in p["tied"]:
                mark = "*" if m in ans[pid] else " "
                near = sum(x["w"] for x in t["before"]) - sum(x["w"] for x in t["after"][m])
                lines.append(f"    {mark} {nm(m):8s}" + " ".join(f"{x['own']}/{x['w']}" for x in t["after"][m][:shown]) + f"   (the stoppers of all goals summed: {near:+d} nearer)")
            for name, mv in cr["picks"][pid].items():
                lines.append(f"   pick of {name}: {nm(mv)}" + (" (accepted)" if mv in ans[pid] else ""))
            lines += [f"   the rule's reason: {p['reason_rule']}", ""]
    (root / "cases.txt").write_text("\n".join(lines))
    print(f"OBS ladder print: subgoal pick cases written to {root / 'cases.txt'}")
    return root / "cases.txt"


def pick_cases_79(tag: str = "full", shown: int = 5) -> Path:
    """PLAN 78.79: the twelve positions of 78.78 again (case_rows.json of subgoal_pick_78_79/<tag> holds the same two sets), each with the board, the moves held equal, our first five goals in words with the long count of the first three
    before and after each tied move, the OTHER side's first five goals in words, what each tied move costs them (+1 harder for them, -1 easier, the state of the goal), whether the line after the move keeps the unit that moved, the pick of
    every candidate and the accepted moves (the answers file, for the human reader only). Writes cases.txt beside the preps."""
    root = Path(paths.experiment("subgoal_pick_78_79", tag))
    preps = pickle.loads((root / "preps.pkl").read_bytes())
    cr = json.loads((root / "case_rows.json").read_text())
    ans = {d["id"]: d["accepted"] for d in (json.loads(x) for x in (LM.out_root("full") / "answers.jsonl").read_text().splitlines() if x.strip())}
    lines = ["THE SAME TWELVE POSITIONS AS PLAN 78.78 (set A: the first six open surprises of the scored part with an accepted move among the moves held equal; set B: the first six resolved positions the best candidate of 78.78 broke)", ""]
    for title, key in (("SET A", "A"), ("SET B", "B")):
        lines += [f"##### {title}", ""]
        for k, pid in enumerate(cr[key], 1):
            p = preps[pid]
            b = chess.Board()
            for m in cr["moves"][pid]:
                b.push_uci(m)

            def nm(m: str) -> str:
                return b.san(chess.Move.from_uci(m))
            lines += [f"=== SET {key} CASE {k}: {pid}  ({'white' if b.turn else 'black'} to move)", str(b), "",
                      f"held equal ({len(p['tied'])}), the search's first move first: {', '.join(nm(m) for m in p['tied'])}",
                      f"the accepted moves: {', '.join(nm(m) for m in ans[pid])}   (accepted among the tied: {', '.join(nm(m) for m in p['tied'] if m in ans[pid]) or 'none'})",
                      f"OUR goals read: {len(p['goals'])}"]
            lt = p.get("long")
            for i, g in enumerate(p["goals"][:shown], 1):
                lines.append(f"   our goal {i}: {g['words']}; layer one's count {g['own_moves']}; the long count {lt['before'][i - 1] if lt else '-'}")
            if lt:
                lines.append(f"   the long count of our first three goals after each tied move (before: {lt['before'][:3]}):")
                for m in p["tied"]:
                    lines.append(f"      {'*' if m in ans[pid] else ' '} {nm(m):8s}{lt['after'][m][:3]}")
            th = p.get("their")
            lines.append(f"THEIR goals (the same proposer, the sides exchanged): {p.get('their_n', 'not read')}")
            for i, w in enumerate(p.get("their_words", [])[:shown], 1):
                lines.append(f"   their goal {i}: {w}")
            if th:
                lines.append("   what each tied move costs their first five goals (+1 harder for them, -1 easier; g=gone, w=works, k=kept), the net over all their goals, and whether the line after the move keeps the unit that moved (reply read):")
                for m in p["tied"]:
                    row = th["after"][m]
                    if row is None:
                        lines.append(f"      {'*' if m in ans[pid] else ' '} {nm(m):8s}(the position after it was not read)")
                        continue
                    cs = " ".join(f"{r['cost']:+d}{r['state'][0]}" for r in row[:shown])
                    rep = p.get("replied_raw", {}).get(m)
                    lines.append(f"      {'*' if m in ans[pid] else ' '} {nm(m):8s}{cs}   net {sum(r['cost'] for r in row):+d}   unit kept by its line: {'yes' if rep is True else ('NO' if rep is False else 'not read')}")
            for name, mv in cr["picks"][pid].items():
                lines.append(f"   pick of {name}: {nm(mv)}" + (" (accepted)" if mv in ans[pid] else ""))
            lines.append("")
    (root / "cases.txt").write_text("\n".join(lines))
    print(f"OBS ladder print: subgoal pick 78.79 cases written to {root / 'cases.txt'}")
    return root / "cases.txt"


# ------------------------------------------------------------------------------------------------------------------------------ PLAN 78.80: the symbols as readable cases

def _words(words: str, side: str) -> str:
    """The pattern's words with `mine` and `theirs` said for the event's side: the event of the mover counts the instances whose `mine` units are the mover's; the event of the other side, the other side's."""
    a, b = ("the mover's", "the other side's") if side == "mover" else ("the other side's", "the mover's")
    return words.replace("(mine", "(@A").replace("(theirs", "(@B").replace("@A", a).replace("@B", b)


def _board_marked(b: chess.Board, marks: set) -> str:
    rows = []
    for r in range(7, -1, -1):
        cells = []
        for f in range(8):
            sq = chess.square(f, r)
            p = b.piece_at(sq)
            ch = p.symbol() if p else "."
            cells.append(f"[{ch}]" if sq in marks else f" {ch} ")
        rows.append(f"{r + 1} " + "".join(cells))
    rows.append("   " + "".join(f" {c} " for c in "abcdefgh"))
    return "\n".join(rows)


def _instances(model, static, goal, pat, kinds, b: chess.Board) -> "tuple[list, list]":
    """The instances (sets of unit ids, by the unit variables) of a pattern on a board: (read for the side to move, read for the other side)."""
    from catspace.incubator.analyst.discovery import graph_layer1 as GL
    from catspace.incubator.analyst.discovery import graph_layer2_match as GM
    items, units = GM.graph_items(GL.graph(model, b.fen(), static), goal)
    s0, s1 = GM.match_binds(items, units, pat, kinds)
    return [t for found in s0 for t in found], [t for found in s1 for t in found]


def _sq(unit_id: str) -> int:
    return chess.parse_square(unit_id.rsplit(":", 1)[1])


def _draw_event(model, static, goal, kinds, pat, moves: list, played: str, side: str, dr: str, pid: str) -> list:
    """One drawing of an event on a position where the move played carries it: for an event made true the board after the move with the units of one new instance in brackets, for an event made false the board before
    with the units of one lost instance in brackets. Also the two things the choice fit controls for, said in words: whether the move takes a unit and what kind of unit moves. -> the lines."""
    b0 = chess.Board()
    for m in moves:
        b0.push_uci(m)
    mv = chess.Move.from_uci(played)
    san = b0.san(mv)
    mover_piece = chess.piece_name(b0.piece_at(mv.from_square).piece_type)
    takes = b0.is_capture(mv)
    b1 = b0.copy()
    b1.push(mv)
    before0, before1 = _instances(model, static, goal, pat, kinds, b0)
    after0, after1 = _instances(model, static, goal, pat, kinds, b1)
    own_before = before0 if side == "mover" else before1                   # before the move the mover is the side to move (side 0), after it the mover is the other side (side 1)
    own_after = after1 if side == "mover" else after0
    frm, to = chess.square_name(mv.from_square), chess.square_name(mv.to_square)

    def remap(t: tuple, frm=frm, to=to) -> tuple:
        return tuple(u.rsplit(":", 1)[0] + ":" + to if u.endswith(":" + frm) else u for u in t)
    old = {remap(t) for t in own_before}
    if dr == "up":
        new = [t for t in own_after if t not in old]
        inst, board, label = (new[0] if new else (own_after[0] if own_after else ())), b1, "after the move"
    else:
        now = set(own_after)
        lost = [t for t in own_before if remap(t) not in now]
        inst, board, label = (lost[0] if lost else (own_before[0] if own_before else ())), b0, "before the move"
    marks = {_sq(u) for u in inst}
    return [f"  --- position {pid} ({'white' if b0.turn else 'black'} plays {san}: a {mover_piece} moves{', taking a unit' if takes else ''}); counts of the pattern for {side}: before {len(own_before)}, after {len(own_after)}; the board {label}, the units of one instance in brackets",
            _board_marked(board, marks), ""]


def symbols_cases(tag: str = "full", shown: int = 8) -> Path:
    """PLAN 78.80: THE SYMBOLS AS READABLE CASES (symbols_measure.py; passing.json, keys.json, sample.jsonl, events.npz under symbols_78_80/<tag>/). Part A: the twelve positions of plan 78.78 (the first six open surprises of the
    scored part with an accepted move among the moves held equal and the first six resolved positions its best candidate broke; subgoal_pick_78_78 case_rows.json), each with the board, the moves held equal and, per tied move, the passing
    events it carries in words (the strongest `shown` by weight) and the sum the pick reads; the accepted moves are the answers file, for the human reader only. Part B: THE TEN STRONGEST PASSING PATTERNS, fixed by this rule before the
    cases were read: the largest |z| of the confirm log ratio (the confirm interval's half width over 1.645), a pattern whose confirm counts equal those of a stronger one is its twin and is skipped (the twins are counted);
    each drawn on the first three positions, in the confirm part's file order, where the move played carries the event: for an event made true the board after the move with the units of one new instance in brackets, for an
    event made false the board before with the units of one lost instance in brackets."""
    import numpy as np
    from scipy import sparse
    from catspace.incubator.analyst.discovery import graph_layer1 as GL
    from catspace.incubator.analyst.discovery import graph_layer2 as L2
    from catspace.incubator.analyst.discovery import graph_layer2_measure as G2M
    from catspace.incubator.analyst.discovery import human_heads_data as HD
    from catspace.incubator.analyst.discovery import symbols as SY
    from catspace.incubator.analyst.discovery import symbols_measure as SM
    root = Path(paths.experiment("symbols_78_80", tag))
    keys = json.loads((root / "keys.json").read_text())
    passing = json.loads((root / "passing.json").read_text())
    sample = [json.loads(x) for x in (root / "sample.jsonl").read_text().splitlines() if x.strip()]
    model = GL.model(HD.GAME, HD.VARIANT)
    static = model.static_part()
    kinds = [n[len("kind:"):] for ty, n in static["nodes"] if ty == "kind"]
    goal = G2M.goal_kind_of(HD.VARIANT)
    counter = SY.Counter(model, [int(k["key"]) for k in keys], goal, 2, (0, 1, 2, 3), 1)
    n_cols = len(keys)
    pass_by = {(d["column"], d["side"], d["dir"]): d for d in passing}
    lines = [f"THE SYMBOLS (plan 78.80): {len(passing)} passing events of {len({d['column'] for d in passing})} patterns; the weights are the find part's log ratios", ""]
    # ---- Part A
    prepsr = Path(paths.experiment("subgoal_pick_78_79", "full"))
    preps = pickle.loads((prepsr / "preps.pkl").read_bytes())
    cr = json.loads((prepsr / "case_rows.json").read_text())
    ans = {d["id"]: d["accepted"] for d in (json.loads(x) for x in (LM.out_root("full") / "answers.jsonl").read_text().splitlines() if x.strip())}
    lines += ["##### PART A: THE TWELVE POSITIONS OF PLAN 78.78", ""]
    for key in ("A", "B"):
        for k, pid in enumerate(cr[key], 1):
            moves = cr["moves"][pid]
            p = preps[pid]
            b = chess.Board()
            for m in moves:
                b.push_uci(m)

            def nm(m: str, b=b) -> str:
                return b.san(chess.Move.from_uci(m))
            tied = p["tied"]
            ev = SY.deltas_of({"moves": moves}, tied, counter)
            lines += [f"=== SET {key} CASE {k}: {pid}  ({'white' if b.turn else 'black'} to move)", str(b), "",
                      f"held equal ({len(tied)}), the search's first move first: {', '.join(nm(m) for m in tied)}",
                      f"the accepted moves: {', '.join(nm(m) for m in ans[pid])}   (accepted among the tied: {', '.join(nm(m) for m in tied if m in ans[pid]) or 'none'})"]
            w = {e: pass_by[e]["weight"] for e in pass_by}
            for m in tied:
                carried = sorted((e for e in ev[m] if e in pass_by), key=lambda e: -abs(w[e]))
                score = sum(w[e] for e in carried)
                lines.append(f"  {'*' if m in ans[pid] else ' '} {nm(m)}: {len(carried)} passing events, sum of weights {score:+.2f}")
                for e in carried[:shown]:
                    col, side, d = e
                    lines.append(f"        {'made TRUE' if d == 'up' else 'made FALSE'} for {side} (weight {w[e]:+.2f}, pattern {keys[col]['id'][:8]}): {_words(keys[col]['words'], side)}")
            pick = SY.symbol_pick({"moves": moves}, tied, [{"column": c, "side": s, "dir": d, "weight": w[(c, s, d)]} for c, s, d in w], None, ev)
            lines += [f"  the symbol pick: {nm(pick)}" + (" (accepted)" if pick in ans[pid] else ""), ""]
    # ---- Part B
    lines += ["##### PART B: THE TEN STRONGEST PASSING PATTERNS, EACH ON THREE POSITIONS WHERE THE MOVE PLAYED MAKES IT TRUE OR FALSE", ""]
    chosen, twins = SM.strongest(passing)
    M = sparse.load_npz(root / "events.npz").tocsc()
    sizes = np.array([len(r["tied"]) for r in sample])
    start = np.concatenate([[0], np.cumsum(sizes)])
    for rank, d in enumerate(chosen, 1):
        col, side, dr = d["column"], d["side"], d["dir"]
        pat = L2.decode(bytes.fromhex(keys[col]["bytes"]))
        idx = SY.event_index(col, side, dr, n_cols)
        hit = set(M.indices[M.indptr[idx]:M.indptr[idx + 1]].tolist())
        lines += [f"=== PATTERN {rank}: {keys[col]['id'][:8]} (size {keys[col]['size']}, level {keys[col]['level']}), event: {'made TRUE' if dr == 'up' else 'made FALSE'} for {side}; twins skipped: {twins.get(rank - 1, 0)}",
                  f"    {_words(keys[col]['words'], side)}",
                  f"    CONFIRM: played moves {d['c_kp']:.0f} (share {d['c_sp']:.3f}) against other tied moves {d['c_ko']:.0f} (share {d['c_so']:.3f}); log ratio {d['c_lr']:+.2f} [{d['c_lo']:+.2f}, {d['c_hi']:+.2f}]; "
                  f"test one term {d['c_term']:+.3f} [{d['c_tlo']:+.3f}, {d['c_thi']:+.3f}]", ""]
        shown_n = 0
        for i, r in enumerate(sample):
            if r["part"] != "confirm" or shown_n >= 3:
                continue
            j = r["tied"].index(r["played"])
            if start[i] + j not in hit:
                continue
            shown_n += 1
            lines += _draw_event(model, static, goal, kinds, pat, r["moves"], r["played"], side, dr, r["id"])
    (root / "cases.txt").write_text("\n".join(lines))
    print(f"OBS ladder print: symbols cases written to {root / 'cases.txt'}")
    return root / "cases.txt"


def reasons_cases(tag: str = "full", shown: int = 8, top: int = 20) -> Path:
    """PLAN 78.81: see the module doc (`--reasons`)."""
    import numpy as np
    from scipy import sparse
    from catspace.incubator.analyst.discovery import graph_layer1 as GL
    from catspace.incubator.analyst.discovery import graph_layer2 as L2
    from catspace.incubator.analyst.discovery import graph_layer2_measure as G2M
    from catspace.incubator.analyst.discovery import human_heads_data as HD
    from catspace.incubator.analyst.discovery import symbols as SY
    root = Path(paths.experiment("symbols_78_81", tag))
    src = Path(paths.experiment("symbols_78_80", "full"))
    keys = json.loads((src / "keys.json").read_text())
    reasons = sorted(json.loads((root / "reasons.json").read_text()), key=lambda d: -abs(d["weight"]))
    sample = [json.loads(x) for x in (src / "sample.jsonl").read_text().splitlines() if x.strip()]
    model = GL.model(HD.GAME, HD.VARIANT)
    static = model.static_part()
    kinds = [n[len("kind:"):] for ty, n in static["nodes"] if ty == "kind"]
    goal = G2M.goal_kind_of(HD.VARIANT)
    n_cols = len(keys)
    cols = sorted({d["column"] for d in reasons})
    ci = {c: j for j, c in enumerate(cols)}
    counter = SY.Counter(model, [int(keys[c]["key"]) for c in cols], goal, 2, (0, 1, 2, 3), 1)
    w = {(ci[d["column"]], d["side"], d["dir"]): d["weight"] for d in reasons}
    by_event = {(ci[d["column"]], d["side"], d["dir"]): d for d in reasons}
    rank = {e: k for k, e in enumerate(sorted(by_event, key=lambda e: -abs(w[e])), 1)}
    ctl_dir = root
    takes = np.load(ctl_dir / "controls.npz")["takes"] if (ctl_dir / "controls.npz").exists() else None
    lines = [f"THE REASONS (plan 78.81): {len(reasons)} reasons with a weight in the joint fit with the controls; the weights are the fit's (log-odds of being the move played among the moves held equal, the controls held fixed)", ""]
    # ---- Part A: the twenty strongest, each drawn three times
    lines += [f"##### PART A: THE STRONGEST {min(top, len(reasons))} REASONS, EACH ON THREE CONFIRM POSITIONS WHERE THE MOVE PLAYED CARRIES IT", ""]
    M = sparse.load_npz(src / "events.npz").tocsc()
    sizes = np.array([len(r["tied"]) for r in sample])
    start = np.concatenate([[0], np.cumsum(sizes)])
    for k, d in enumerate(reasons[:top], 1):
        pat = L2.decode(bytes.fromhex(keys[d["column"]]["bytes"]))
        idx = SY.event_index(d["column"], d["side"], d["dir"], n_cols)
        hit = set(M.indices[M.indptr[idx]:M.indptr[idx + 1]].tolist())
        share = float(takes[sorted(hit)].mean()) if takes is not None and hit and len(takes) > max(hit) else float("nan")
        lines += [f"=== REASON {k}: {d['id'][:8]} (size {d['size']}, level {d['level']}), the event: {'made TRUE' if d['dir'] == 'up' else 'made FALSE'} for {d['side']}; weight {d['weight']:+.3f}"
                  + (f" (events alone {d['weight_events_alone']:+.3f})" if d.get("weight_events_alone") is not None else " (no weight without the controls)"),
                  f"    {_words(keys[d['column']]['words'], d['side'])}",
                  f"    carried by {len(hit)} tied moves of the 78.80 sample, {100 * share:.1f} pct of them take a unit", ""]
        shown_n = 0
        for i, r in enumerate(sample):
            if r["part"] != "confirm" or shown_n >= 3:
                continue
            j = r["tied"].index(r["played"])
            if start[i] + j not in hit:
                continue
            shown_n += 1
            lines += _draw_event(model, static, goal, kinds, pat, r["moves"], r["played"], d["side"], d["dir"], r["id"])
    # ---- Part B: the twelve positions
    prepsr = Path(paths.experiment("subgoal_pick_78_79", "full"))
    preps = pickle.loads((prepsr / "preps.pkl").read_bytes())
    cr = json.loads((prepsr / "case_rows.json").read_text())
    ans = {d["id"]: d["accepted"] for d in (json.loads(x) for x in (LM.out_root("full") / "answers.jsonl").read_text().splitlines() if x.strip())}
    lines += ["##### PART B: THE TWELVE POSITIONS OF PLAN 78.78", ""]
    pats = [{"column": c, "side": sd, "dir": dr, "weight": wt} for (c, sd, dr), wt in w.items()]
    for key in ("A", "B"):
        for k, pid in enumerate(cr[key], 1):
            moves = cr["moves"][pid]
            tied = preps[pid]["tied"]
            b = chess.Board()
            for m in moves:
                b.push_uci(m)

            def nm(m: str, b=b) -> str:
                return b.san(chess.Move.from_uci(m))
            ev = SY.deltas_of({"moves": moves}, tied, counter)
            lines += [f"=== SET {key} CASE {k}: {pid}  ({'white' if b.turn else 'black'} to move)", str(b), "",
                      f"held equal ({len(tied)}), the search's first move first: {', '.join(nm(m) for m in tied)}",
                      f"the accepted moves: {', '.join(nm(m) for m in ans[pid])}   (accepted among the tied: {', '.join(nm(m) for m in tied if m in ans[pid]) or 'none'})"]
            for m in tied:
                mv = chess.Move.from_uci(m)
                carried = sorted((e for e in ev[m] if e in w), key=lambda e: -abs(w[e]))
                lines.append(f"  {'*' if m in ans[pid] else ' '} {nm(m)} ({chess.piece_name(b.piece_at(mv.from_square).piece_type)}{', takes a unit' if b.is_capture(mv) else ''}): {len(carried)} reasons, sum of weights {sum(w[e] for e in carried):+.2f}")
                for e in carried[:shown]:
                    lines.append(f"        {'made TRUE' if e[2] == 'up' else 'made FALSE'} for {e[1]} (weight {w[e]:+.3f}, reason {rank[e]}, pattern {keys[cols[e[0]]]['id'][:8]}): {_words(keys[cols[e[0]]]['words'], e[1])}")
            pk = SY.symbol_pick({"moves": moves}, tied, pats, None, ev)
            lines += [f"  the pick of the reasons: {nm(pk)}" + (" (accepted)" if pk in ans[pid] else ""), ""]
    (root / "cases.txt").write_text("\n".join(lines))
    print(f"OBS ladder print: reasons cases written to {root / 'cases.txt'}")
    return root / "cases.txt"


def play_cases(tag: str = "full", shown: int = 6, exp: str = "symbols_play_78_82", cases_file: str = "cases82.json", title: "str | None" = None, attacked: bool = False) -> Path:
    """PLAN 78.82: THE THIRTY DECISIONS WHERE THE REASONS' PICK DIFFERED FROM THE SEARCH'S FIRST MOVE (symbols_play.py `cases`: cases82.json under symbols_play_78_82/<tag>/, chosen by the rule in that module's doc before any game was
    read; the reasons are reasons.json of symbols_78_81/full). Each: the board, the moves held equal in words, the search's first move and the pick, each with the reasons it carries in words (the strongest `shown` by |weight|), the
    sum of the weights, whether it takes a unit and the kind of unit that moves. For the human reader's judgement only. Writes cases.txt beside the decisions."""
    root = Path(paths.experiment(exp, tag))
    reasons = json.loads((Path(paths.experiment("symbols_78_81", "full")) / "reasons.json").read_text())
    cases = json.loads((root / cases_file).read_text())
    lines = [title or f"THE THIRTY DECISIONS (plan 78.82): the first decision per game of reasons v control, set A, where the pick differed from the search's first move; {len(cases)} read. The reasons are 78.81's, weights are log-odds of being the move played among equals.", ""]
    for k, c in enumerate(cases, 1):
        b = chess.Board()
        for m in c["moves"]:
            b.push_uci(m)

        def nm(m: str, b=b) -> str:
            return b.san(chess.Move.from_uci(m))

        def side(m: str, key: str, b=b) -> list:
            mv = chess.Move.from_uci(m)
            out = [f"    {nm(m)} ({chess.piece_name(b.piece_at(mv.from_square).piece_type)}{', takes a unit' if b.is_capture(mv) else ''}): {len(c[key])} reasons, sum of weights {sum(w for _i, w in c[key]):+.2f}"]
            for i, w in sorted(c[key], key=lambda x: -abs(x[1]))[:shown]:
                r = reasons[i]
                out.append(f"        {'made TRUE' if r['dir'] == 'up' else 'made FALSE'} for {r['side']} (weight {w:+.3f}, reason {i}): {_words(r['words'], r['side'])}")
            if attacked:                                                    # PLAN 78.89: what each move leaves loose, for the reader's judgement
                x = b.copy()
                x.push(chess.Move.from_uci(m))
                out.append(f"        after it, the mover's units under attack: {_attacked(x, b.turn)}; the other side's: {_attacked(x, not b.turn)}")
            return out
        lines += [f"=== DECISION {k}: {c['gid']} ply {c['ply']}  ({'white' if b.turn else 'black'} to move)" + (f"  [band {c['band']}, {c['units']} units]" if "band" in c else ""), str(b), "",
                  *([f"the mover's units under attack before the move: {_attacked(b, b.turn)}", f"the other side's units under attack before the move: {_attacked(b, not b.turn)}"] if attacked else []),
                  f"held equal ({len(c['tied'])}), the search's first move first: {', '.join(nm(m) for m in c['tied'])}",
                  f"the search's first move: {nm(c['first'])}; the pick: {nm(c['pick'])}", "  the search's first move:"] + side(c["first"], "carried_first") + ["  the pick:"] + side(c["pick"], "carried_pick") + [""]
    (root / "cases.txt").write_text("\n".join(lines))
    print(f"OBS ladder print: {len(cases)} play decisions written to {root / 'cases.txt'}")
    return root / "cases.txt"


def _attacked(b: chess.Board, side: bool) -> str:
    """The units of `side` that the other side attacks on this board, each with how many attack it and how many guard it (for the reader's judgement only)."""
    out = []
    for sq, p in b.piece_map().items():
        if p.color == side:
            a = len(b.attackers(not side, sq))
            if a:
                out.append(f"{chess.piece_name(p.piece_type)} {chess.square_name(sq)} attacked {a}, guarded {len(b.attackers(side, sq))}")
    return "; ".join(out) or "none"


def cut_cases(tag: str = "full", exp: str = "symbols_play_78_84") -> Path:
    """PLAN 78.84 PART ONE: THE TWENTY FLAGGED DECISIONS (tie_cut_measure.py `report`: cases84.json under symbols_play_78_84/<tag>/, chosen by the rule in that module's doc): a tie set that holds a move whose line gives up a unit for nothing
    and a move whose line does not. Each: the board, the units under attack, the moves held equal, what the search reported (the depth it finished, its value, whether the equal reading was cut or the position decided, the bound the equality
    is), then for the bad move and the good one the line in words, the ledger of the line, the line search's own value, and the numbers of the moves read as the search reads them, and the cause the measure named. For the human reader's judgement only.
    Writes cases.txt beside the decisions."""
    root = Path(paths.experiment(exp, tag))
    cases = json.loads((root / "cases84.json").read_text())
    lines = [f"THE TWENTY FLAGGED DECISIONS (plan 78.84 part one): {len(cases)} read. A move FLAGGED BAD has a line (read as threat_concrete reads lines: the move, then the best replies of both sides to a settled end, 8 plies) that ends with the mover "
             f"having lost units and taken none; the GOOD move's line does not. The search holds both EQUAL: its test is a bound (at least the best at depth D), it reports no value per move.", ""]
    for k, c in enumerate(cases, 1):
        b = chess.Board()
        for m in c["moves"]:
            b.push_uci(m)
        me = b.turn

        def nm(m: str, b=b) -> str:
            return b.san(chess.Move.from_uci(m))

        s = c["search"]
        pair = c["pair"]
        lines += [f"=== CASE {k}: {c['gid']} ply {c['ply']}  band {c['band']}, {c['units']} units  ({'white' if me else 'black'} to move)", str(b), "",
                  f"the mover's units under attack before the move: {_attacked(b, me)}",
                  f"the other side's units under attack before the move: {_attacked(b, not me)}",
                  f"held equal ({len(c['tied'])}), the search's first move first: {', '.join(nm(m) for m in c['tied'])}",
                  f"the search (0.2 s again): played {nm(s['move'])}; finished depth {s['depth']}, value {s['value']}, proved {s['proved']}, equal reading cut by the clock {s['cut']}, {s['nodes']} nodes, {s['seconds']:.3f} s, of which {s['ties_seconds']:.3f} s for the ties; {s['legal']} legal moves",
                  f"all the tied moves whose line gives up a unit for nothing: {', '.join(nm(m) for m in c['bad'])}; none of the others does ({len(c['good'])})", f"CAUSE named by the measure: {c['cause']}", ""]
        for role, m in (("BAD", pair["bad"]), ("GOOD", pair["good"])):
            ln = c["lines"][m]
            x = b.copy()
            x.push(chess.Move.from_uci(m))
            lines += [f"  {role}: {nm(m)}   units of the mover under attack after it: {_attacked(x, me)}",
                      f"    the line: {san_line(b, ln['names'], 9)}   (verdict {ln['verdict']}, the mover lost {ln['ev_lost']} and took {ln['ev_won']}, net {ln['net']} against a resolution of {ln['res']:.0f}; "
                      f"the line search's own value {ln['net_search']}; depth {ln['depth']}; events disagree with value: {ln['pv_dis']})"]
            nb = c["numbers"]["bad" if role == "BAD" else "good"]
            lines += [f"    the move as the search reads it at depth {s['depth']}: gain {nb.get('gain')}, the position after it searched to depth {(s['depth'] or 1) - 1}: finished {nb.get('child_depth')}, value for the other side {nb.get('child_value')}, "
                      f"so the move's value {nb.get('value')} against the search's value {s['value']}"]
        d = c["numbers"]["bad_deeper"]
        lines += [f"  BAD at two plies deeper: finished {d.get('child_depth')}, value {d.get('value')}", ""]
    (root / "cases.txt").write_text("\n".join(lines))
    print(f"OBS ladder print: {len(cases)} flagged decisions written to {root / 'cases.txt'}")
    return root / "cases.txt"


def groups_cases(tag: str = "full", exp: str = "symbols_78_88", top: int = 20) -> Path:
    """PLAN 78.88 `--groups88 TAG` (symbols_groups_measure.py): PART A: the twenty largest groups of the 136 reasons (by members, then by the tied moves that carry them; groups_i.json), every member in words with side,
    direction and weight, for the reader to judge whether a group is one idea. PART B: the thirty decisions of `cases` (cases88.json: ten a band, the rule in the measure's doc), each with the board, the units under attack
    (for the reader's judgement only), the moves held equal, the move the human played, the search's first move and the pick, and for each of the two the three strongest groups they carry in words. Writes cases.txt under
    symbols_78_88/<tag>/."""
    root = Path(paths.experiment(exp, tag))
    gi = json.loads((root / "groups_i.json").read_text())
    lines = [f"PART A. THE TWENTY LARGEST OF THE {len(gi)} GROUPS of the 136 reasons (rule in symbols.py: average linkage on the Jaccard index of the carrier sets on the FIND rows, same side, direction and sign of weight, at or above 0.5). "
             f"Groups with two or more members first, then the singletons with the most carrying tied moves.", ""]
    for k, g in enumerate(sorted(gi, key=lambda g: (-len(g["members"]), -g["carriers"]))[:top], 1):
        lines.append(f"=== GROUP {k} (number {g['group']}): {len(g['members'])} members, {g['side']} {g['dir']}, carried by {g['carriers']} FIND tied moves")
        for m in sorted(g["members"], key=lambda m: -abs(m["weight"])):
            lines.append(f"      weight {m['weight']:+.3f} (size {m['size']}, level {m['level']}): {_words(m['words'], m['side'])}")
        lines.append("")
    c88 = json.loads((root / "cases88.json").read_text())
    groups = json.loads((root / c88["groups_file"]).read_text())
    lines += [f"PART B. THE THIRTY DECISIONS: variant {c88['variant']}; {len(c88['cases'])} read. The pick is the highest sum of group weights among the moves the search holds equal; the groups are those of the variant.", ""]

    def side(b: chess.Board, m: str, tops: list) -> list:
        mv = chess.Move.from_uci(m)
        out = [f"    {b.san(mv)} ({chess.piece_name(b.piece_at(mv.from_square).piece_type)}{', takes a unit' if b.is_capture(mv) else ''}): the three strongest groups it carries"]
        for gidx, w in tops:
            g = groups[gidx]
            mem = sorted(g["members"], key=lambda x: -abs(x["weight"]))
            out.append(f"        weight {w:+.3f}, group {gidx} ({len(mem)} members, {g['side']} {g['dir']}): {_words(mem[0]['words'], mem[0]['side'])}" + (f"  [and {len(mem) - 1} like it]" if len(mem) > 1 else ""))
        return out or ["        none"]
    for k, c in enumerate(c88["cases"], 1):
        b = chess.Board()
        for m in c["moves"]:
            b.push_uci(m)
        me = b.turn
        nm = lambda m, b=b: b.san(chess.Move.from_uci(m))
        lines += [f"=== DECISION {k}: {c['id']} ply {c['ply']}  band {c['band']}, {c['units']} units  ({'white' if me else 'black'} to move)", str(b), "",
                  f"the mover's units under attack: {_attacked(b, me)}", f"the other side's units under attack: {_attacked(b, not me)}",
                  f"held equal ({len(c['tied'])}), the search's first move first: {', '.join(nm(m) for m in c['tied'])}",
                  f"the human played: {nm(c['played'])}; the search's first move: {nm(c['first'])}; the pick: {nm(c['pick'])}", "  the search's first move:"] + side(b, c["first"], c["first_groups"]) + ["  the pick:"] + side(b, c["pick"], c["pick_groups"]) + [""]
    (root / "cases.txt").write_text("\n".join(lines))
    print(f"OBS ladder print: groups and {len(c88['cases'])} decisions written to {root / 'cases.txt'}")
    return root / "cases.txt"


def own_cases(tag: str = "full", exp: str = "symbols_78_90") -> Path:
    """PLAN 78.90 `--own90 TAG` (symbols_own_measure.py `cases`): THE THIRTY DECISIONS where the pick of the fit with the other side's cells (A) differs from the pick of the old weights, ten a band (five from each CONFIRM part), by the rule
    in that module's doc. Each: the board, the units of each side under attack before the move, the moves held equal with a mark on those that leave a unit of the mover's borne on and unguarded that was not so before, the move the
    human played, the search's first move, the old pick and the new pick, each with what it leaves under attack and the three strongest groups it carries in words (the groups of A, groups_A.json). For the reader's judgement only.
    Writes cases.txt under symbols_78_90/<tag>/."""
    root = Path(paths.experiment(exp, tag))
    c90 = json.loads((root / "cases90.json").read_text())
    groups = json.loads((root / c90["groups_file"]).read_text())
    lines = [f"THE THIRTY DECISIONS (plan 78.90): {len(c90['cases'])} read; ten a band, five from each CONFIRM part (sample 1 the early game, sample 2 the later game), where the pick of A (the fit with the cells of the other side's "
             "bearing) differs from the pick of the old weights (78.88's). Groups are those of A; weights are log-odds of being the move played among equals.", ""]

    def side(b: chess.Board, c: dict, m: str, tops: list) -> list:
        mv = chess.Move.from_uci(m)
        loose = c["left"][c["tied"].index(m)]
        out = [f"    {b.san(mv)} ({chess.piece_name(b.piece_at(mv.from_square).piece_type)}{', takes a unit' if b.is_capture(mv) else ''}{'; LEAVES a unit of the mover borne on and unguarded that was not so before' if loose else ''})"]
        x = b.copy()
        x.push(mv)
        out.append(f"        after it, the mover's units under attack: {_attacked(x, b.turn)}; the other side's: {_attacked(x, not b.turn)}")
        for gidx, w in tops:
            g = groups[gidx]
            mem = sorted(g["members"], key=lambda y: -abs(y["weight"]))
            out.append(f"        weight {w:+.3f}, group {gidx} ({len(mem)} members, {g['side']} {g['dir']}): {_words(mem[0]['words'], mem[0]['side'])}" + (f"  [and {len(mem) - 1} like it]" if len(mem) > 1 else ""))
        return out
    for k, c in enumerate(c90["cases"], 1):
        b = chess.Board()
        for m in c["moves"]:
            b.push_uci(m)
        me = b.turn
        nm = lambda m, b=b: b.san(chess.Move.from_uci(m))
        mark = lambda m, c=c, nm=nm: nm(m) + ("*" if c["left"][c["tied"].index(m)] else "")
        lines += [f"=== DECISION {k}: {c['id']} ply {c['ply']}  band {c['band']}, {c['units']} units, sample {c['sample']}  ({'white' if me else 'black'} to move)", str(b), "",
                  f"the mover's units under attack: {_attacked(b, me)}", f"the other side's units under attack: {_attacked(b, not me)}",
                  f"held equal ({len(c['tied'])}), the search's first move first; * leaves a unit of the mover's borne on and unguarded that was not so before: {', '.join(mark(m) for m in c['tied'])}",
                  f"the human played: {nm(c['played'])}; the search's first move: {nm(c['first'])}; the old pick: {nm(c['old'])}; the new pick: {nm(c['new'])}",
                  "  the old pick (the three strongest groups of A it carries):"] + side(b, c, c["old"], c["old_groups"]) + ["  the new pick:"] + side(b, c, c["new"], c["new_groups"]) + ["  the search's first move:"] + side(b, c, c["first"], c["first_groups"]) + [""]
    (root / "cases.txt").write_text("\n".join(lines))
    print(f"OBS ladder print: {len(c90['cases'])} decisions written to {root / 'cases.txt'}")
    return root / "cases.txt"


def order_cases(tag: str = "full", exp: str = "order_78_91") -> Path:
    """PLAN 78.91 `--order91 TAG`; see the module doc. For the reader's judgement only."""
    root = Path(paths.experiment(exp, tag))
    c91 = json.loads((root / "cases91.json").read_text())
    groups = json.loads((Path(paths.experiment(c91["groups_exp"], "full")) / c91["groups_file"]).read_text())
    lines = [f"THE THIRTY POSITIONS (plan 78.91): {len(c91['cases'])} read. 'reasons first' = the reasons' order (78.90's fit A) puts the reference move first among the quiet moves and the static order does not; 'static first' = the opposite. "
             "The static order is the search's own before any reading (the moves that gain, then the rest as generated); the reasons' order reorders the quiet moves only. Weights are log-odds of being the move played among equals.", ""]
    for k, c in enumerate(c91["cases"], 1):
        b = chess.Board()
        for m in c["moves"]:
            b.push_uci(m)
        me = b.turn
        san = lambda i, b=b, c=c: b.san(chess.Move.from_uci(c["legal"][i]))
        takes = [san(i) for i, g in enumerate(c["gains"]) if g != 0]

        def groups_of(i: int, c=c) -> list:
            out = []
            for gidx, w in c["groups"].get(str(i), []):
                g = groups[gidx]
                mem = sorted(g["members"], key=lambda y: -abs(y["weight"]))
                out.append(f"        weight {w:+.3f}, group {gidx} ({len(mem)} members, {g['side']} {g['dir']}): {_words(mem[0]['words'], mem[0]['side'])}" + (f"  [and {len(mem) - 1} like it]" if len(mem) > 1 else ""))
            return out or ["        no group carried"]
        ref, st1, h1 = c["ref"], c["static_quiet"][0], c["hint_quiet"][0]
        lines += [f"=== POSITION {k}: {c['id']} ply {c['ply']}  band {c['band']}, {c['units']} units, sample {c['sample']}, {c['kind'].replace('_', ' ')}  ({'white' if me else 'black'} to move)", str(b), "",
                  f"the mover's units under attack: {_attacked(b, me)}", f"the other side's units under attack: {_attacked(b, not me)}",
                  f"moves that take or change a unit: {', '.join(takes) or 'none'}; {c['n_quiet']} quiet moves",
                  f"the reference move: {san(ref)} (place among the quiet moves: static {c['place_static']}, reasons' own fit {c['place_hint']}, 78.81's weights {c['place_reasons']}, short reading {c['place_short']})",
                  f"first five quiet moves, static order: {', '.join(san(i) for i in c['static_quiet'])}", f"first five quiet moves, the reasons' order: {', '.join(san(i) for i in c['hint_quiet'])}",
                  f"  the reference move {san(ref)} (strongest groups it carries):"] + groups_of(ref) + [f"  the static first {san(st1)}:"] + groups_of(st1) + [f"  the reasons' first {san(h1)}:"] + groups_of(h1) + [""]
    (root / "cases.txt").write_text("\n".join(lines))
    print(f"OBS ladder print: {len(c91['cases'])} positions written to {root / 'cases.txt'}")
    return root / "cases.txt"


def choice_cases(tag: str = "full", exp: str = "reasoner_choice_78_92") -> Path:
    """PLAN 78.92 `--choice92 TAG`; see the module doc. For the reader's judgement only."""
    root = Path(paths.experiment(exp, tag))
    c92 = json.loads((root / "cases92.json").read_text())
    lines = [f"THE TWENTY POSITIONS (plan 78.92): {len(c92['cases'])} read. {c92['rule']} Values are in the judge's own units (the record's table), for the side the search plays for (the mover): the value of the whole line from here, "
             "the played move's own gain included. 'plain' is the search with both prunings off (the reference), read to the depth the search finished and, where it finished, two plies deeper.", ""]
    for k, c in enumerate(c92["cases"], 1):
        b = chess.Board()
        for m in c["moves"]:
            b.push_uci(m)
        me = b.turn
        san = lambda u, b=b: b.san(chess.Move.from_uci(u)) if u else None
        lines += [f"=== POSITION {k}: {c['id']}  band {c['band']}, {c['units']} units  ({'white' if me else 'black'} to move)", str(b), "",
                  f"the mover's units under attack: {_attacked(b, me)}", f"the other side's units under attack: {_attacked(b, not me)}",
                  f"the search (both prunings on) finished depth {c['depth']} with value {c['value']} and plays {san(c['played'])}",
                  f"plain at depth {c['depth']}: best {san(c['best_move0'])} = {c['best0']}; the played move {san(c['played'])} = {c['played0']}; shortfall {c['best0'] - c['played0']}",
                  (f"plain at depth {c['depth'] + 2}: best {san(c['best_move2'])} = {c['best2']}; the played move = {c['played2']}; shortfall {c['best2'] - c['played2']}"
                   if c["best2"] is not None and c["played2"] is not None else f"plain at depth {c['depth'] + 2}: not finished within the fuse"),
                  f"null move off alone, depth {c['depth']}: best {c['no_null0'][0]}, played move {c['no_null0'][1]}; late reduction off alone: best {c['no_late0'][0]}, played move {c['no_late0'][1]}",
                  f"the played move asked again without the two prunings (null window at {c['value']}): probe {c['verify_probe']}", ""]
    (root / "cases.txt").write_text(chr(10).join(lines))
    print(f"OBS ladder print: {len(c92['cases'])} positions written to {root / 'cases.txt'}")
    return root / "cases.txt"


def _goal_words(g: dict, who: str) -> str:
    if not g["goal"]:
        return f"{who}: no goal ({g['threats']} confirmed threat(s); {g['unread']} cut, {g['deferred']} needing a quiet first move)" + ("; a forced end of the game is found" if g["forced"] else "")
    return (f"{who}: a goal ({g['kind']}): two threats on {' and '.join(g['shape'])}, stake {g['stake']} in the record's integers, {g['n_pairs']} such pair(s)" + ("; a forced end of the game is found" if g["forced"] else ""))


def future_cases(tag: str = "full") -> Path:
    """PLAN 78.94 `--future94 TAG`: the thirty positions of future_78_94/<tag>/cases94.json, for the reader's judgement only."""
    root = Path(paths.experiment("future_78_94", tag))
    data = json.loads((root / "cases94.json").read_text())
    lines = [data["rule"], ""]
    cls_word = {1: "winning", 0: "neither", -1: "losing"}
    for k, c in enumerate(data["cases"], 1):
        b = chess.Board()
        for m in c["moves"]:
            b.push_uci(m)
        me = b.turn
        lines += [f"=== POSITION {k}: {c['id']}  [{c['why']}]  {c['band']} band, {c['units']} units, ply {c['ply']}  ({'white' if me else 'black'} to move)", str(b), "",
                  f"the oracle (depth 16): the side to move is {cls_word[c['cls']]}" + (f" (score {c['cp']})" if c["cp"] is not None else f" (forced end in {c['forced_in']})") + f"; its best line: {san_line(b, c['pv'])}",
                  f"the mover's units under attack: {_attacked(b, me)}; the other side's: {_attacked(b, not me)}",
                  "our reading: " + _goal_words(c["mover_g"], "the side to move") + " | " + _goal_words(c["other_g"], "the other side"),
                  f"our reading of the oracle's line (frame): {c['frame_type']}" + (" -- " + "; ".join(f"after {j} plies {v['threats']} threat(s), goal {v['goal']}, units {v['n_first']}/{v['n_second']}" for j, v in sorted(c["frame"].items(), key=lambda t: int(t[0]))) if c["frame"] else "")
                  + (f"; our concrete list {'holds' if c['seen']['any'] else 'does not hold'} the oracle's first move" if c["seen"] else ""),
                  "the judge's chances (the side to move, the other side): " + "; ".join(f"{n} {v[0]:.3f}, {v[1]:.3f}" for n, v in c["chances"].items()),
                  f"what happened (the side to move, the other): a goal within 8 plies {c['later']['G8']}, a winning point within 8 {c['later']['W8']}, either within 16 {c['later']['GW16']} (-1 where the game ended first); the game's result for the side to move: {['won', 'drawn', 'lost'][c['result_mover']]}",
                  f"the next moves in the game: {san_line(b, c['next_moves'])}", ""]
    (root / "cases.txt").write_text(chr(10).join(lines))
    print(f"OBS ladder print: {len(data['cases'])} positions written to {root / 'cases.txt'}")
    return root / "cases.txt"


def _kind_name(k: str) -> str:
    try:
        return chess.piece_name(chess.PIECE_SYMBOLS.index(k.lower()))
    except ValueError:
        return str(k)


def _threat_words(th: dict, places: list) -> str:
    return f"{_kind_name(th['kind'])} on {places[th['place']]} (the line reading's net {th['net']})"


def future95_cases(tag: str = "full") -> Path:
    """PLAN 78.95 `--future95 TAG`: the thirty positions of future_78_95/<tag>/cases95.json, three groups of ten, for the reader's judgement only: goals of 78.94 that fall when every legal reply is tried (the reply that meets both threats), frames of one threat with the chance
    that kind pays as played, picks of the weighted chooser that differ from both the search's first move and the pattern pick."""
    root = Path(paths.experiment("future_78_95", tag))
    data = json.loads((root / "cases95.json").read_text())
    lines = [data["rule"], ""]
    for k, c in enumerate(data["cases"], 1):
        b = chess.Board()
        for m in c["moves"]:
            b.push_uci(m)
        me = b.turn
        pl = c["places"]
        head = f"=== POSITION {k}: {c['id']}  [{c['group']}]  {c['units']} units, ply {len(c['moves'])}  ({'white' if me else 'black'} to move)"
        if c["group"] == "a goal that falls":
            owner_board = b.copy()
            if c["side"] == "mover":
                owner_board.push(chess.Move.null())                       # the owner of the threatened units moves after the side to move's pass
            kt = c["kept"]
            lines += [head + f"  [{c['part']} game]", str(b), "",
                      f"the mover's units under attack: {_attacked(b, me)}; the other side's: {_attacked(b, not me)}",
                      f"the goal of {'the side to move' if c['side'] == 'mover' else 'the other side'} by the kept replies: two threats ({kt['kind']}): " + " and ".join(_threat_words(t, pl) for t in kt["threats"]) + f"; stake {kt['stake']}",
                      f"legal replies of the owner tried: {c['replies']}"]
            for fl in c["fallen"]:
                com = fl["common"]
                words = []
                for m in com[:6]:
                    try:
                        words.append(owner_board.san(chess.Move.from_uci(m)))
                    except Exception:
                        words.append(m + "?")
                lines.append("  a pair that falls: " + " and ".join(_threat_words(t, pl) for t in fl["threats"]) + f"; {fl['n_common']} replies meet both after which the taking is no longer confirmed, for instance {', '.join(words)}")
            lines += [f"the oracle (depth 16, side to move class {c['cls']}): its best line {san_line(b, c['pv'])}", f"the next moves in the game: {san_line(b, c['next_moves'])}", ""]
        elif c["group"] == "a frame of one threat":
            f = c["frame"]
            ch = c["chance_band"] or {}
            lines += [head, str(b), "",
                      f"the mover's units under attack: {_attacked(b, me)}; the other side's: {_attacked(b, not me)}",
                      f"the oracle: the side to move is winning; its best line: {san_line(b, c['pv'])}",
                      f"our reading of that line: {f['kind']} (one threat); steps read " + "; ".join(f"after {j} plies {s['threats']} threat(s), units of the victim {s['n_v']}" for j, s in sorted(f["steps"].items(), key=lambda t: int(t[0]))) + f"; the material of the line in the record's worths {f['net']}",
                      f"the three questions: the search at the play clock {'plays' if f['q1']['plays'] else ('holds equal' if f['q1']['equal'] else 'does not play')} the oracle's first move; the line reading of it says {f['q2']['verdict']}; the threat list {'names' if f['q3']['any'] else 'does not name'} it",
                      f"the chance this kind pays as played ({c['band']} band, FIND): " + (f"the first move of the line is played in {100 * ch['first']:.1f} pct of {ch['n']} points, the winning side's class holds at the next position in {100 * ch['pay_first']:.1f} pct when it was played and {100 * ch['pay_not']:.1f} pct when not" if ch and ch.get("pay_first") is not None and ch.get("pay_not") is not None else "too few points"),
                      f"what happened here: the class {'held' if c['stable'] else 'did not hold'} at the next position; the next moves in the game: {san_line(b, c['next_moves'])}", ""]
        else:
            nm = lambda m: b.san(chess.Move.from_uci(m))
            p = c["picks"]
            lines += [head, str(b), "",
                      f"the mover's units under attack: {_attacked(b, me)}; the other side's: {_attacked(b, not me)}",
                      f"held equal ({len(c['tied'])}), the search's first move first: {', '.join(nm(m) for m in c['tied'])}",
                      f"picks: the weighted chooser {nm(p['chooser'])}; the search's first move {nm(p['first'])}; the pattern pick {nm(p['pattern'])}; the judge of 78.94 {nm(p['judge94'])}; the joined pick {nm(p['joined'])}; the human played {nm(p['played'])}",
                      f"what the chooser reads (window {c['setting']} plies; weight x (chance for the side that moves less the chance for the other side), by kind of frame): " + " | ".join(f"{nm(m)}: " + ", ".join(f"{kd} {v:+.1f}" for kd, v in sorted(pk.items())) for m, pk in c["per_kind"].items()), ""]
    (root / "cases.txt").write_text(chr(10).join(lines))
    print(f"OBS ladder print: {len(data['cases'])} positions written to {root / 'cases.txt'}")
    return root / "cases.txt"


def prior96_cases(tag: str = "full") -> Path:
    """PLAN 78.96 `--prior96 TAG`: the twenty positions of prior_78_96/<tag>/cases96.json (prior_measure.py `cases`), ten where the fitted prior's pick keeps the oracle's class and the pattern pick does not, ten the other way, all CONFIRM positions where the moves the search holds
    equal differ in keeping the class. Each: the board, what each side has under attack, the moves held equal with the oracle's class after each (for the reader's judgement only; no oracle number is read by the engine), the picks and what the prior read (its chance per move)."""
    root = Path(paths.experiment("prior_78_96", tag))
    cases = json.loads((root / "cases96.json").read_text())
    cname = {1: "winning", 0: "neither", -1: "losing"}
    lines = ["THE TWENTY POSITIONS (plan 78.96): CONFIRM positions where the equal moves differ in keeping the oracle's class; the class is the oracle's at depth 16 for the side that moves, after each move; for the reader's judgement only.", ""]
    for k, c in enumerate(cases, 1):
        b = chess.Board()
        for m in c["moves"]:
            b.push_uci(m)
        me = b.turn
        nm = lambda m, b=b: b.san(chess.Move.from_uci(m))
        lines += [f"=== POSITION {k}: {c['id']}  [{c['group']}]  {c['units']} units ({c['band']}), ply {len(c['moves'])}  ({'white' if me else 'black'} to move); the position's class: {cname[c['cls']]}", str(b), "",
                  f"the mover's units under attack: {_attacked(b, me)}; the other side's: {_attacked(b, not me)}",
                  f"held equal ({len(c['tied'])}), the search's first move first; class after each, the prior's chance that it keeps the class, whether it takes a unit:"]
        for m, cm, sc, tk in zip(c["tied"], c["cm"], c["scores"], c["takes"]):
            tag_ = [n for n, x in (("search's first", c["first"]), ("pattern pick", c["pattern"]), ("judge of 78.94", c["judge"]), ("joined", c["joined"]), ("PRIOR", c["prior"]), ("oracle's move", c["best"])) if x == m]
            lines.append(f"    {nm(m)}: {cname[cm]}; chance {sc:.2f}{'; takes a unit' if tk else ''}" + (f"   <- {', '.join(tag_)}" if tag_ else ""))
        if c["best"] and c["best"] not in c["tied"]:
            lines.append(f"    the oracle's own move {nm(c['best'])} is not among them (class after it: {cname[c['best_cls']]})")
        lines.append("")
    (root / "cases.txt").write_text(chr(10).join(lines))
    print(f"OBS ladder print: {len(cases)} positions written to {root / 'cases.txt'}")
    return root / "cases.txt"


def room97_cases(tag: str = "full") -> Path:
    """PLAN 78.97 `--room97 TAG`: the twenty class-losing equal moves that stay equal at the longest clock, ten tactical and ten slow (room_measure.py `cases`: cases97.json under room_78_97/<tag>/). Each: the board, what each side has under attack, the move, the class after it
    and the oracle's own move, the oracle's line after the move in words (a tactical one: what it gives up), what our reading says of the other side's first reply (the threat list, the line reading, the search at the play clock), and how many moves each setting holds equal. For the reader's judgement only;
    no oracle number is read by the engine."""
    root = Path(paths.experiment("room_78_97", tag))
    cases = json.loads((root / "cases97.json").read_text())
    cname = {1: "winning", 0: "neither", -1: "losing"}
    lines = ["THE TWENTY POSITIONS (plan 78.97): class-losing equal moves (the oracle's class after the move is lower than before it) that stay in the equal set at 5 times the clock, at 25 times it and with the prunings off; CONFIRM positions, one move a position; "
             "the class is the oracle's at depth 16 for the side that moves; for the reader's judgement only.", ""]
    for k, c in enumerate(cases, 1):
        b = chess.Board()
        for m in c["moves"]:
            b.push_uci(m)
        me = b.turn
        san = b.san(chess.Move.from_uci(c["move"]))
        a = b.copy()
        a.push_uci(c["move"])
        words, t = [], a.copy()
        for m in c["pv"]:
            words.append(t.san(chess.Move.from_uci(m)))
            t.push_uci(m)
        lines += [f"=== POSITION {k}: {c['id']}  [{c['group']}]  {c['units']} units ({c['band']}), ply {len(c['moves'])}  ({'white' if me else 'black'} to move); the position's class: {cname[c['cls']]}", str(b), "",
                  f"the mover's units under attack: {_attacked(b, me)}; the other side's: {_attacked(b, not me)}",
                  f"the stuck move: {san}; class after it: {cname[c['cls_after']]} (asked again: {cname[c['cls_again']]}); the oracle's own move: {b.san(chess.Move.from_uci(c['best'])) if c['best'] else 'none'} (class after it: {cname[c['best_cls']] if c['best_cls'] is not None else 'none'})",
                  f"the oracle's line after it: {' '.join(words) or 'none'}; net units lost by the mover in the measured worths after the even plies: {c['net']}" + (f"; forced end in {c['forced_in']}" if c["forced_in"] is not None else ""),
                  f"the search's first move: {b.san(chess.Move.from_uci(c['first']))}; moves held equal by the 78.96 run {len(c['tied'])}, at 5 times {len(c['sets']['x5'])}, at 25 times {len(c['sets']['x25'])}, prunings off {len(c['sets']['off'])}; ledger depth by setting "
                  + ", ".join(f"{n} {d}" for n, d in c["depths"].items())]
        if c["reply"] and c["group"] == "tactical":
            lines.append(f"our reading of the reply {a.san(chess.Move.from_uci(c['reply']))} at the position after the move: the threat list names it: {c['seen']}; the line reading says {c['verdict']}; the search at the play clock holds it: {c['held']}; the statement missing: {c['kind']}")
        keepers = [b.san(chess.Move.from_uci(m)) for m, cm in zip(c["tied"], c["cm"]) if cm >= c["cls"]]
        lines += [f"the equal moves that keep the class ({len(keepers)} of {len(c['tied'])}): {', '.join(keepers[:12])}", ""]
    (root / "cases.txt").write_text(chr(10).join(lines))
    print(f"OBS ladder print: {len(cases)} positions written to {root / 'cases.txt'}")
    return root / "cases.txt"


def equal98_cases(tag: str = "full", which: str = "named") -> Path:
    """PLAN 78.98 `--equal98 TAG [--equal98-set named|rest]`: THE TRACE OF THE EQUAL MOVES (equal_trace.py: trace.jsonl under equal_78_98/<tag>/). `named`: the five positions of the plan, written to trace.txt; `rest`: ten more stuck moves of kind all present (the first ten
    in the order of the trace, one per position), written to cases.txt. Each: the board, the stuck move and the class-keeping move in words, what the root holds for each (the value of the position, the gain, the answer of the rule, the
    answer with the prunings off, the exact value), the rule by which they are held equal, the line each value comes from in words with the units given and taken and the units each side has at its end, and the cause. For the reader's judgement only; the reading of the judge is appended by hand."""
    root = Path(paths.experiment("equal_78_98", tag))
    rows = [json.loads(x) for x in (root / "trace.jsonl").read_text().splitlines() if x.strip()]
    games = {}
    from catspace.incubator.analyst.discovery import future_labels as FL
    for g in FL.universe():
        games[g["ord"]] = g
    named = [r for r in rows if str(r.get("tag", "")).startswith("named")]
    rest, seen = [], set()
    for r in rows:
        if r.get("tag") == "all present" and "error" not in r and r["id"] not in seen:
            seen.add(r["id"])
            rest.append(r)
    shown = named if which == "named" else rest[:10]
    out = ["THE TRACE OF THE EQUAL MOVES (plan 78.98). The rule of the code: a root move is held equal iff its null-window answer at the finished depth D is at least v, the value of the position at D (warm table, late-move reduction and null move on; margin zero). The judge's worths are the measured record "
           "of the odds games; a unit given costs its worth, a unit taken adds it. For the reader's judgement only.", ""]
    for k, r in enumerate(shown, 1):
        if "error" in r:
            continue
        ord_, ply = r["id"].split("_")
        moves = games[int(ord_)]["moves"][:int(ply)]
        b = chess.Board()
        for m in moves:
            b.push_uci(m)

        def nm(m, b=b):
            return b.san(chess.Move.from_uci(m)) if m else "none"

        def words(rd, b=b, m0=None):
            if not rd or not rd.get("ok"):
                return "unread"
            return san_line(b, rd["names"], cap=12)
        s, kp = r["stuck_row"], r.get("keeper_row") or {}
        out += [f"=== CASE {k}: {r['id']}  [{r.get('tag')}]  band {r['band']}, ply {ply} ({'white' if b.turn else 'black'} to move); D {r['depth']}, value of the position at D {r['value']}; equal set {r['equal_set']} moves (verified without the prunings {r['verified_set']}); the answer was {'cut' if r['tie_cut'] else 'whole'}", str(b), "",
                f"the stuck move {nm(r['stuck'])}: gain {s['gain']}; answer of the rule {s['warm']}; answer with the prunings off {s['plain']}; exact value {s['exact']}; held equal: {s['held']}",
                f"  the line its value comes from: {words(r['stuck_line'])}; value of the line {r['stuck_line'].get('value')}; given {r['stuck_line'].get('exchange', {}).get('gave')} taken {r['stuck_line'].get('exchange', {}).get('took')} kind changes {r['stuck_line'].get('exchange', {}).get('changed')}; "
                f"units at its settled end (side to move first, then the other side): {r['stuck_line'].get('inputs')}",
                f"the class-keeping equal move {nm(r['keeper'])}: gain {kp.get('gain')}; answer of the rule {kp.get('warm')}; answer with the prunings off {kp.get('plain')}; exact value {kp.get('exact')}; held equal: {kp.get('held')}",
                f"  the line its value comes from: {words(r.get('keeper_line'))}; value of the line {(r.get('keeper_line') or {}).get('value')}; given {(r.get('keeper_line') or {}).get('exchange', {}).get('gave')} taken {(r.get('keeper_line') or {}).get('exchange', {}).get('took')}; "
                f"units at its settled end: {(r.get('keeper_line') or {}).get('inputs')}",
                f"the prunings-off reading of the stuck move at D and at D+{2}: its value {(r.get('deep') or {}).get('other_d')} against the best {(r.get('deep') or {}).get('best_d')}, then {(r.get('deep') or {}).get('other')} against {(r.get('deep') or {}).get('best')}; the cause: {r['cause']}", ""]
    name = "trace.txt" if which == "named" else "cases.txt"
    (root / name).write_text(chr(10).join(out))
    print(f"OBS ladder print: {len(shown)} positions written to {root / name}")
    return root / name


def record99_cases(tag: str = "full") -> Path:
    """PLAN 78.99 `--record99 TAG`: THE JUDGED POSITIONS (record_measure.py `cases`: cases99.json under record_78_99/<tag>/): the five positions of 78.98 and ten room positions where the search's first move changes between the odds record and the human-result record. Each: the board, what each
    side has under attack, and for each measured record of the kinds the move the search plays at the play clock, how many moves it holds equal, its depth, and the class after that move where the 78.96 labels read it; the oracle's own move as the labels have it. Written to cases.txt
    (the reader's judgement is appended by hand). For the reader's judgement only; no oracle number is read by the engine."""
    root = Path(paths.experiment("record_78_99", tag))
    cases = json.loads((root / "cases99.json").read_text())
    cname = {1: "winning", 0: "neither", -1: "losing"}
    lines = ["THE JUDGED POSITIONS (plan 78.99): the search at the play clock under each measured record of the kinds (gen2 = the odds games', the judge in play; selfplay_78_58; human_78_59). The class is the oracle's (78.96 labels) for the side that moves; for the reader's judgement only.", ""]
    for k, c in enumerate(cases, 1):
        b = chess.Board()
        for m in c["moves"]:
            b.push_uci(m)
        me = b.turn

        def nm(m, b=b):
            return b.san(chess.Move.from_uci(m)) if m else "none"
        lines += [f"=== POSITION {k}: {c['id']}  [{c['tag']}]  band {c['band']}, ply {len(c['moves'])}  ({'white' if me else 'black'} to move); the position's class: {cname[c['cls']]}", str(b), "",
                  f"the mover's units under attack: {_attacked(b, me)}; the other side's: {_attacked(b, not me)}",
                  f"the oracle's own move as the labels have it: {nm(c['best'])} (class after it: {cname[c['best_cls']] if c['best_cls'] is not None else 'none'}); the labels' equal set held {len(c['tied'])} moves, the first {nm(c['label_first'])}"]
        for n, row in c["rows"].items():
            cl = c["class_of_first"][n]
            lines.append(f"  under {n}: plays {nm(row['first'])}; holds {len(row['equals'])} equal; depth {row['depth']}; class after the move: {cname[cl] if cl is not None else 'no class (not read by the labels)'}")
        lines.append("")
    (root / "cases.txt").write_text(chr(10).join(lines))
    print(f"OBS ladder print: {len(cases)} positions written to {root / 'cases.txt'}")
    return root / "cases.txt"


def stack79_cases(tag: str = "full") -> Path:
    """PLAN 79.00 `--stack79 TAG`: THE TWENTY JUDGED POSITIONS (stack_measure.py `cases`: cases79.json under stack_79_00/<tag>/): the room of the stack (the search under the human-result record with the verified choice), the first move loses the oracle's class. Each: the board, what each side has under attack,
    the first move and the pick, the oracle's line after the first move in words, the class after it (and asked again), the oracle's own move and the equal moves that keep the class. For the reader's judgement only (appended by hand); no oracle number is read by the engine."""
    root = Path(paths.experiment("stack_79_00", tag))
    cases = json.loads((root / "cases79.json").read_text())
    cname = {1: "winning", 0: "neither", -1: "losing"}
    lines = ["THE TWENTY JUDGED POSITIONS (plan 79.00): positions of the room under the stack (human-result record, verified choice, 78.82's pick) where the search's first move loses the oracle's class (its class after the move is lower than the position's). The class is the oracle's at depth 16 for the side "
             "that moves; for the reader's judgement only.", ""]
    for k, c in enumerate(cases, 1):
        b = chess.Board()
        for m in c["moves"]:
            b.push_uci(m)
        me = b.turn

        def nm(m, b=b):
            return b.san(chess.Move.from_uci(m)) if m else "none"
        a = b.copy()
        a.push_uci(c["first"])
        words, t = [], a.copy()
        for m in c["pv_first"]:
            words.append(t.san(chess.Move.from_uci(m)))
            t.push_uci(m)
        keepers = [nm(m) for m, cm in zip(c["tied"], c["cm"]) if cm >= c["cls"]]
        pick_cls = c["cm"][c["tied"].index(c["pick"])] if c["pick"] in c["tied"] else None
        lines += [f"=== POSITION {k}: {c['id']}  band {c['band']} ({c['units']} units), ply {len(c['moves'])}  ({'white' if me else 'black'} to move); the position's class: {cname[c['cls']]}; the verification: {['off', 'held', 'replaced', 'failed', 'cut'][c['state']]}", str(b), "",
                  f"the mover's units under attack: {_attacked(b, me)}; the other side's: {_attacked(b, not me)}",
                  f"the search's first move: {nm(c['first'])}; class after it: {cname[c['cm'][c['tied'].index(c['first'])]]} (asked again: {cname[c['cls_again']]}); the equal set holds {len(c['tied'])} moves, depth {c['depth']}; the pick: {nm(c['pick'])} (class after it: {cname[pick_cls] if pick_cls is not None else 'none'})",
                  f"the oracle's line after the first move: {' '.join(words) or 'none'}" + (f"; forced end in {c['forced_in']}" if c["forced_in"] is not None else ""),
                  f"the oracle's own move: {nm(c['best'])} (class after it: {cname[c['best_cls']] if c['best_cls'] is not None else 'none'}); the equal moves that keep the class ({len(keepers)} of {len(c['tied'])}): {', '.join(keepers[:12])}", ""]
    (root / "cases.txt").write_text(chr(10).join(lines))
    print(f"OBS ladder print: {len(cases)} positions written to {root / 'cases.txt'}")
    return root / "cases.txt"


def leaf01_cases(tag: str = "full") -> Path:
    """PLAN 79.01 `--leaf01 TAG`: THE FIFTEEN JUDGED POSITIONS (leaf_measure.py `cases`: cases_leaf.json under leaf_79_01/<tag>/). Each: the board, what each side has under attack, the first move and the pick, and for every equal move its firm label (keeps / loses / neither the class), the oracle's class and number asked twice
    (the second asking in brackets), the units each side has by kind at the end of the line the search's value comes from, the estimator's two chances for the mover at the root (end of the line, one move after), and the line in words. For the reader's judgement only (appended by hand); no oracle number
    is read by the engine."""
    root = Path(paths.experiment("leaf_79_01", tag))
    cases = json.loads((root / "cases_leaf.json").read_text())
    cname = {1: "winning", 0: "neither", -1: "losing"}
    lines = ["THE FIFTEEN JUDGED POSITIONS (plan 79.01): positions of the firm room (some equal move firmly keeps the oracle's class and some firmly loses it) where the equal move with the highest estimator chance at the END OF THE LINE is not the search's first move. The class is the oracle's at depth 16 "
             "for the side that plays the move; the chance is the human-results estimator's chance for the mover at the root; for the reader's judgement only.", ""]
    for k, c in enumerate(cases, 1):
        b = chess.Board()
        for m in c["moves"]:
            b.push_uci(m)
        me = b.turn

        def nm(m, b=b):
            return b.san(chess.Move.from_uci(m)) if m else "none"
        lines += [f"=== POSITION {k}: {c['id']}  part {c['part']}, band {c['band']} ({c['units']} units), ply {len(c['moves'])}  ({'white' if me else 'black'} to move); the position's class: {cname[c['cls']]}; depth of the search {c['depth']}", str(b), "",
                  f"the mover's units under attack: {_attacked(b, me)}; the other side's: {_attacked(b, not me)}",
                  f"the search's first move: {nm(c['first'])}; the pick: {nm(c['pick'])}; the equal set holds {len(c['tied'])} moves; the highest end-of-line chance: {', '.join(nm(c['tied'][j]) for j in c['end_argmax'])}"]
        for j, m in enumerate(c["tied"]):
            ln = c["lines"][j]
            tag_m = ("FIRST " if j == 0 else "") + ("PICK " if m == c["pick"] else "") + ("HIGHEST " if j in c["end_argmax"] else "")
            cp = lambda x: "forced" if x is None else f"{x:+d}"
            lines.append(f"  {nm(m)} {tag_m}| {c['vec'][j] or 'neither'} (class {cname[c['cm'][j]]} [{cname[c['cls2'][j]]}], number {cp(c['cp'][j])} [{cp(c['cp2'][j])}]) | chance at the end of the line {c['v_end'][j]:.3f}, one move after {c['v_one'][j]:.3f} | line ({ln['settled']} plies, value {ln['value']}"
                         + (", cut" if ln["cut"] else "") + f"): {san_line(b, ln['names'], 14)}")
            side = "mover" if ln["end_is_root"] else "other side"
            lines.append("      units at the end of the line (side to move there = the " + side + ", then the other): " + "; ".join(f"{kk} {v[0]}-{v[1]}" for kk, v in sorted(ln["end"].items())))
        lines.append("")
    (root / "cases.txt").write_text(chr(10).join(lines))
    print(f"OBS ladder print: {len(cases)} positions written to {root / 'cases.txt'}")
    return root / "cases.txt"


def cut02_cases(tag: str = "full") -> Path:
    """PLAN 79.02 `--cut02 TAG`: THE JUDGED POSITIONS OF THE CUT AND THE MEND (cut_mend.py `cases`: cases_cut.json under cut_79_02/<tag>/): 79.01's positions 2, 3 and 5 and twelve where the mend puts out a firmly losing move. Each: the board, the first move and the pick without the mend (A) and with it (M), and for every equal move of A's set its firm label, the oracle's
    class and number asked twice, the audit's answers at the search's depth (the pruned null-window answer the equal reading held it by, with the late-move reduction off, the null move off, both off, and the exact value, against the value of the position), by what kind of answer it was held, whether M put it out, and the line in words. For the reader's judgement only
    (appended by hand); no oracle number is read by the engine."""
    root = Path(paths.experiment("cut_79_02", tag))
    cases = json.loads((root / "cases_cut.json").read_text())
    cname = {1: "winning", 0: "neither", -1: "losing"}
    lines = ["THE JUDGED POSITIONS (plan 79.02): positions 2, 3 and 5 of 79.01's fifteen, and positions of the firm room where the mend puts out a firmly losing move. A = the equal reading as it was, M = the equal reading asking a held move again without the two prunings in the deepening's order. The class is the oracle's at depth 16 for the side that plays the move; "
             "the answers are the audit's at the search's depth D (the value v of the position; a move is held when its answer is v or more); for the reader's judgement only.", ""]
    for k, c in enumerate(cases, 1):
        b = chess.Board()
        for m in c["moves"]:
            b.push_uci(m)
        me = b.turn

        def nm(m, b=b):
            return b.san(chess.Move.from_uci(m)) if m else "none"
        rd = c.get("reads") or {}
        lines += [f"=== POSITION {k}: {c['id']}{'  [named by the plan: 79.01]' if c['named'] else ''}  part {c['part']}, band {c['band']} ({c['units']} units), ply {len(c['moves'])}  ({'white' if me else 'black'} to move); the position's class: {cname[c['cls']]}; depth of the stored run {c['depth']}; the audit's value v {c['value']}", str(b), "",
                  f"the mover's units under attack: {_attacked(b, me)}; the other side's: {_attacked(b, not me)}"]
        if rd:
            lines.append(f"A: plays {nm(rd['a']['first'])}, set {[nm(x) for x in sorted(set(rd['a']['equals']) | {rd['a']['first']})]}, pick {nm(rd['a'].get('pick'))}, depth {rd['a']['depth']}, cut {rd['a']['cut']}; M: plays {nm(rd['m']['first'])}, set {[nm(x) for x in sorted(set(rd['m']['equals']) | {rd['m']['first']})]}, pick {nm(rd['m'].get('pick'))}, depth {rd['m']['depth']}, cut {rd['m']['cut']}, "
                         f"second asking [asked, held, asked again, put out, not reached, first reading cut] {rd['m']['reask']}")
        inm = set(rd["m"]["equals"]) | {rd["m"]["first"]} if rd else set()
        for j, m in enumerate(c["tied"]):
            ln = (c.get("lines") or [None] * len(c["tied"]))[j]
            r = c["rows"][j]
            cp = lambda x: "forced" if x is None else f"{x:+d}"
            ans = (f"warm {r['warm']}, no-late {r['no_late']}, no-null {r['no_null']}, plain {r['plain']}, exact {r['exact']}" if r else "no audit")
            reduced = c["hold"][j] in ("late", "null", "both", "joint")
            if rd and m not in inm and m not in (set(rd["a"]["equals"]) | {rd["a"]["first"]}):
                gone_note = "in neither A's nor M's set at this reading (79.00's run held it: the clock)"
            else:
                gone_note = None
            gone = (gone_note if gone_note else ("NOT IN M's set; the audit's unpruned answer is under v, so the unpruned question is the reason the count names (M's own count of moves it put out is in the header: when it is 0 the move was absent from M's first reading instead)" if reduced else "NOT IN M's set although the audit's plain answer holds it too (the cut or the order of M's first reading, not the unpruned question)") if rd and m not in inm else ("kept by M" if rd else ""))
            tag_m = ("FIRST " if m == c["first"] else "") + ("PICK(stored) " if m == c["pick"] else "")
            lines.append(f"  {nm(m)} {tag_m}| {c['vec'][j] or 'neither'} (class {cname[c['cm'][j]]} [{cname[c['cls2'][j]]}], number {cp(c['cp'][j])} [{cp(c['cp2'][j])}]) | held by: {c['hold'][j]} ({ans}) | {gone}"
                         + (f" | line ({ln['settled']} plies, value {ln['value']}" + (", cut" if ln["cut"] else "") + f"): {san_line(b, ln['names'], 14)}" if ln else ""))
        lines.append("")
    (root / "cases.txt").write_text(chr(10).join(lines))
    print(f"OBS ladder print: {len(cases)} positions written to {root / 'cases.txt'}")
    return root / "cases.txt"


def best03_cases(tag: str = "full") -> Path:
    """PLAN 79.03 `--best03 TAG`: THE FIFTEEN JUDGED POSITIONS OF THE SEARCH'S VALUE UNDER THE BEST IT COULD READ AT ITS OWN DEPTH (best_cases.py: cases_best.json under best_79_03/<tag>/): position 2 of 79.01's fifteen and fourteen where the best root move's own value at the depth is a unit of the weakest kind or more above the value the deepening reports. Each: the board, the
    depth, the move the deepening chooses and its value, the move that has the higher value at that depth and its value, with the switch off and on, the line that starts with each (one line read at the depth, under the human-result record), the unpruned reading's best where it was read, 79.00's stored class of the moves in its equal set, and the clocked reads under A and M where they exist.
    For the reader's judgement only (appended by hand); no oracle number is read by the engine."""
    root = Path(paths.experiment("best_79_03", tag))
    cases = json.loads((root / "cases_best.json").read_text())
    cname = {1: "winning", 0: "neither", -1: "losing"}
    lines = ["THE JUDGED POSITIONS (plan 79.03): position 2 of 79.01's fifteen and fourteen positions where the best root move's own full-window value at the depth D the stored run finished is a unit of the weakest kind or more above the value v the deepening reports. `off` is the search as it is, `on` the same with no "
             "late-move reduction at the root (the switch root_full). A line is the engine's line read from the move at depth D under the human-result record; a value is in the judge's integers (the worths of the human-result record, printed with best_mend's first line). The class is the oracle's at depth 16 for the side that plays the move (79.00's stored label); for the reader's judgement only.", ""]
    for k, c in enumerate(cases, 1):
        b = chess.Board()
        for m in c["moves"]:
            b.push_uci(m)
        me = b.turn

        def nm(m, b=b):
            return b.san(chess.Move.from_uci(m)) if m else "none"
        off, on = c["off"], c["on"]
        lines += [f"=== POSITION {k}: {c['id']}  band {c['band']} ({c['units']} units), ply {len(c['moves'])}  ({'white' if me else 'black'} to move); the position's class: {cname[c['cls']]}; depth {c['depth']}; kind of gap {c['kind']}, gap {c['gap']}", str(b), "",
                  f"the mover's units under attack: {_attacked(b, me)}; the other side's: {_attacked(b, not me)}",
                  f"OFF: chooses {nm(off['first'])}, value v {off['value']}, nodes {off['run_nodes']}; the best root move's own value at D: " + ", ".join(f"{nm(n)} {e}" for e, n in off["exact"][:4]),
                  f"ON : chooses {nm(on['first'])}, value v {on['value']}, nodes {on['run_nodes']}"]
        for tag_, lab in (("played", "the line from the move OFF chooses"), ("best", "the line from the best root move")):
            ln = c["lines"].get(tag_)
            if ln and ln.get("ok"):
                m0 = off["first"] if tag_ == "played" else c["best"]
                lines.append(f"  {lab} ({nm(m0)}; {ln['settled']} plies to the settled end, value {ln['value']}" + (", cut" if ln["cut"] else "") + f", exchange {ln['exchange']}): {san_line(b, ln['names'], 14)}")
            else:
                lines.append(f"  {lab}: not read")
        if c.get("which"):
            w = c["which"]
            lines.append(f"  the unpruned deepening at D: best {w['plain_best']}, the best move's value under it {w['plain_other']}; with the reduction off alone best {w['no_late_best']}, with the null move off alone best {w['no_null_best']}; pruning the best move's reading passed through: {w['cause']}")
        if c.get("clock"):
            ck = c["clock"]
            lines.append(f"  at the play clock: A (off) plays {nm(ck['a']['first'])} (pick {nm(ck['a'].get('pick'))}, depth {ck['a']['depth']}); M (on) plays {nm(ck['m']['first'])} (pick {nm(ck['m'].get('pick'))}, depth {ck['m']['depth']}); N (off again) plays {nm(ck['n']['first'])}")
        lines.append("  79.00's stored class after each move of its equal set: " + ", ".join(f"{nm(m)}{' [played]' if m == c['first'] else ''}{' [pick]' if m == c['pick'] else ''} {cname[x]}" for m, x in zip(c["tied"], c["cm"])))
        lines.append("")
    (root / "cases.txt").write_text(chr(10).join(lines))
    print(f"OBS ladder print: {len(cases)} positions written to {root / 'cases.txt'}")
    return root / "cases.txt"


def _reading(b: chess.Board, r: dict, ans: set) -> list:
    """The stored reading of one run of one position, in words: the rung, the move the search chose at each clock, what it held equal, the deepest line with its events and its verdict. No reader is run."""
    def nm(m: str) -> str:
        return b.san(chess.Move.from_uci(m))
    d = r.get("deep") or {}
    out = [f"    rung {r['rung']}" + (f" (found at {r['found_at']:g} s)" if r["found_at"] else "") + f"; the wake's search held {r['equals']} moves equal; the first move at the wake's clock is "
           + ("accepted" if r.get("first_hit") else "not accepted")]
    for k, v in r["chose"].items():
        if k.startswith("3 search"):
            out.append(f"    {k}: chose {nm(v['move'])} ({'accepted' if v['move'] in ans else 'not accepted'}), held {v['equals']} equal, depth {v['depth']}")
    if d:
        out.append(f"    the deepest line: {san_line(b, d.get('names', []), 14)}; events {d.get('events')}; kind changes {d.get('changes')}; ended {d.get('end')} after {d.get('depth')} plies, cut {d.get('cut')}")
        if d.get("verdict"):
            out.append(f"    the verdict: {d['verdict']}")
    if r.get("features"):
        out.append(f"    features: {r['features']}")
    return out


def reread_cases(exp: str = "ladder_78_86") -> Path:
    """PLAN 78.86: THE TWENTY CHANGED POSITIONS (ladder_reread.pick_changed: the rule is in its doc; chosen_ids.json beside the run) printed from the STORED results of the earlier run (ladder_78_77/selfplay_78_58_on)
    and the new one: the board, the units of both sides under attack with how many guard them, the accepted moves (the answers file, for the human reader only), and each run's reading. Runs no reader. Writes cases.txt in
    ladder_78_86/."""
    from catspace.incubator.analyst.discovery import ladder_reread as LR
    root = Path(paths.experiment(exp))
    chosen = json.loads((root / "chosen_ids.json").read_text())
    old = LR.load(Path(paths.experiment(*LR.EARLIER)))
    new = LR.load(root / "selfplay_78_58_on")
    src = LM.out_root("full")
    ans = {d["id"]: d["accepted"] for d in (json.loads(x) for x in (src / "answers.jsonl").read_text().splitlines() if x.strip())}
    pos = {d["id"]: d for d in (json.loads(x) for x in (src / "positions.jsonl").read_text().splitlines() if x.strip())}
    lines = [f"THE TWENTY CHANGED POSITIONS (plan 78.86): {len(chosen)} positions, by the rule of ladder_reread.pick_changed. Earlier run: ladder_78_77/selfplay_78_58_on. New run: this folder.", ""]
    for k, pid in enumerate(chosen, 1):
        b = chess.Board()
        for m in pos[pid]["moves"]:
            b.push_uci(m)
        me = b.turn
        lines += [f"=== CASE {k}: {pid}  (rung {old[pid]['result']['rung']} -> {new[pid]['result']['rung']}; {'white' if me else 'black'} to move)", str(b), "",
                  f"the mover's units under attack: {_attacked(b, me)}", f"the other side's units under attack: {_attacked(b, not me)}",
                  f"the accepted moves: {', '.join(b.san(chess.Move.from_uci(m)) for m in ans[pid])}", "  EARLIER:"]
        lines += _reading(b, old[pid]["result"], set(ans[pid]))
        lines.append("  NOW:")
        lines += _reading(b, new[pid]["result"], set(ans[pid]))
        lines.append("")
    out = root / "cases.txt"
    out.write_text("\n".join(lines))
    print(f"OBS ladder print: {len(chosen)} changed positions written to {out}")
    return out


TAIL_WORDS = {0: "no", 1: "won by the mover", 2: "lost by the mover"}


def verdict_cases(exp: str = "verdict_78_87") -> Path:
    """PLAN 78.87: THE TEN LINES WHOSE VERDICT WORD CHANGED (verdict_measure.pick: the rule is in its doc; picked.json and the before/after reads in verdict_78_87/): the board before the line's first move, the units
    under attack, and the line with its word, its net, the line search's own number and how its tail ended, before (main's extension) and after (the new tail and the word QUIET). Runs no reader. Writes cases.txt."""
    root = Path(paths.experiment(exp))
    picked = json.loads((root / "picked.json").read_text())
    moves = {}
    for name in ("a", "b"):
        for x in (json.loads(t) for t in (root / f"before_{name}.jsonl").read_text().splitlines() if t.strip()):
            moves[(name.upper(), x.get("key", x.get("id")))] = list(x["moves"])
    lines = [f"THE TEN LINES WHOSE VERDICT WORD CHANGED (plan 78.87): {len(picked)} lines, by the rule of verdict_measure.pick. A = the bad move of a flagged decision of 78.84, B = a position of 78.86 at rung 3 or open.", ""]
    for k, r in enumerate(picked, 1):
        b = chess.Board()
        for m in moves[(r["set"], r["id"])]:
            b.push_uci(m)
        me = b.turn
        lines += [f"=== CASE {k}: {r['set']} {r['id']}  ({'white' if me else 'black'} to move; the line starts with the mover's {'move played' if r['set'] == 'A' else 'move the search chose'})", str(b), "",
                  f"the mover's units under attack: {_attacked(b, me)}", f"the other side's units under attack: {_attacked(b, not me)}"]
        for label, v in (("BEFORE", r["before"]), ("AFTER", r["after"])):
            lines += [f"  {label}: {v['verdict']}; the line: {san_line(b, v['full_names'], 20)}",
                      f"    the mover lost {v['taker_lost']} and took {v['taker_won']}; net {v['net']}, the line search's own number {v['net_search']}, kind change in the net {v['kind_net']}; "
                      f"the tail ended the game: {TAIL_WORDS[v['tail_end']]}; unsettled {v['unsettled']}"]
        lines.append("")
    out = root / "cases.txt"
    out.write_text("\n".join(lines))
    print(f"OBS ladder print: {len(picked)} changed lines written to {out}")
    return out


_CASE_TOOLS: dict = {}


def _case_job(job: tuple) -> dict:
    """One (record, position): the readers of our own reasoning at the clock the record's ladder stopped at."""
    name, pid, moves, clock = job
    if name not in _CASE_TOOLS:
        rec, kc = LM.split_run(name)
        _CASE_TOOLS[name] = LM.Tools(rec, kind_change=kc)
    t = _CASE_TOOLS[name]
    s = t.search({"moves": moves}, clock)
    ln = t.line({"moves": moves}, clock)
    return {"name": name, "id": pid, "clock": clock, "search": s, "line": ln}


def stop_clock(res: dict) -> float:
    r = res["result"]
    if r["found_at"]:
        return r["found_at"]
    if r["rung"] in (1, 2):
        return LD.LIMITS["wake_seconds"]
    return LD.LIMITS["ladder"][-1]


def fixed_cases(names: list, per: int = 6) -> "tuple[list, list]":
    """The two sets of the rule: the first `per` open surprises of the 78.68 list in file order; the first `per` positions of positions.jsonl whose rung under some record differs from the first record's."""
    from catspace.incubator.analyst.discovery import ladder_compare as LC
    src = LM.out_root("full")
    lst = [json.loads(x)["id"] for x in (src / "surprise_list.jsonl").read_text().splitlines() if x.strip()]
    runs = {n: LC.load_results(n) for n in names}
    order = [json.loads(x)["id"] for x in (src / "positions.jsonl").read_text().splitlines() if x.strip()]
    changed = [i for i in order if all(i in runs[n] for n in names) and len({runs[n][i]["result"]["rung"] for n in names}) > 1]
    return lst[:per], changed[:per]


def ids_of_cases_file(path: "str | Path") -> list:
    """The ids of the positions of an earlier cases file (the line `=== SET A CASE 3: <id>  (...)`), in file order, each once."""
    out = []
    for ln in Path(path).read_text().splitlines():
        if ln.startswith("=== SET "):
            pid = ln.split(": ", 1)[1].split("  (")[0].strip()
            if pid not in out:
                out.append(pid)
    return out


def records_cases(names: list, per: int = 6, workers: int = 10, fixed_from: "str | None" = None) -> Path:
    """`fixed_from` (PLAN 78.77): a cases file of an earlier plan whose positions are printed again (as set A), then the first `per` positions whose rung differs between the runs `names` (set B)."""
    import multiprocessing
    from catspace.incubator.analyst.discovery import ladder_compare as LC
    first, changed = fixed_cases(names, 400 if fixed_from else per)
    if fixed_from:
        old = ids_of_cases_file(fixed_from)
        first = old
        changed = [i for i in changed if i not in old][:per]               # (the first `per` positions whose rung changes that the earlier file did not already print)
    runs = {n: LC.load_results(n) for n in names}
    ans = {d["id"]: d["accepted"] for d in (json.loads(x) for x in (LM.out_root("full") / "answers.jsonl").read_text().splitlines() if x.strip())}
    pos = {d["id"]: d for d in (json.loads(x) for x in (LM.out_root("full") / "positions.jsonl").read_text().splitlines() if x.strip())}
    ids = list(dict.fromkeys(first + changed))
    jobs = [(n, i, pos[i]["moves"], stop_clock(runs[n][i])) for i in ids for n in names if i in runs[n]]
    jobs.sort(key=lambda j: -j[3])
    with multiprocessing.get_context("spawn").Pool(workers) as pool:
        got = {(r["name"], r["id"]): r for r in pool.imap_unordered(_case_job, jobs, chunksize=1)}
    lines = [f"RECORDS {names}; set A: the first {per} open surprises of the 78.68 list in file order; set B: the first {per} positions whose rung differs between the records", ""]
    for title, sel in (("A", first), ("B", changed)):
        for k, pid in enumerate(sel, 1):
            b = chess.Board()
            for m in pos[pid]["moves"]:
                b.push_uci(m)

            def nm(m: str) -> str:
                return b.san(chess.Move.from_uci(m))
            lines += [f"=== SET {title} CASE {k}: {pid}  ({'white' if b.turn else 'black'} to move)", str(b), "", f"the accepted moves: {', '.join(nm(m) for m in ans[pid])}"]
            for n in names:
                if (n, pid) not in got:
                    continue
                g, r = got[(n, pid)], runs[n][pid]["result"]
                s, ln = g["search"], g["line"]
                lines += [f"  --- record {n}: rung {r['rung']}" + (f" (found at {r['found_at']:g} s)" if r["found_at"] else "") + f"; read at {g['clock']:g} s",
                          f"      held equal ({len(s['equals'])}): {', '.join(nm(m) for m in sorted(s['equals']))}",
                          f"      the search's first move: {nm(s['move'])} ({'accepted' if s['move'] in ans[pid] else 'not accepted'})",
                          f"      its best line (after the search's own move): {san_line(b, ln['names'], 10)}; events {ln['events']}; kind changes {ln.get('changes')}; ended {ln['end']} after {ln['depth']} plies, cut {ln['cut']}"]
            lines.append("")
    out = Path(paths.experiment("ladder_78_77" if fixed_from else "ladder_78_76", "cases.txt"))
    out.write_text("\n".join(lines))
    print(f"OBS ladder print: {len(first)} + {len(changed)} cases under records {names} written to {out}")
    return out


def main(argv: "list | None" = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", nargs="+", default=None, help="PLAN 78.76: the fixed cases under each named record of the kinds (the first is the control)")
    ap.add_argument("--fixed-from", default=None, help="PLAN 78.77: print the positions of this earlier cases file again, then the first six whose rung differs between the named runs (RECORD@on, RECORD@off)")
    ap.add_argument("--pick", default=None, help="PLAN 78.78: the cases of the subgoal rule scored on the open surprises (subgoal_pick_78_78/<tag>)")
    ap.add_argument("--pick79", default=None, help="PLAN 78.79: the twelve cases of 78.78 again, with the other side's goals and the long count (subgoal_pick_78_79/<tag>)")
    ap.add_argument("--symbols", default=None, help="PLAN 78.80: the symbols as readable cases (symbols_78_80/<tag>)")
    ap.add_argument("--reasons", default=None, help="PLAN 78.81: the reasons as readable cases (symbols_78_81/<tag>)")
    ap.add_argument("--play82", default=None, help="PLAN 78.82: the thirty decisions where the reasons' pick differed from the search's first move (symbols_play_78_82/<tag>)")
    ap.add_argument("--play83", default=None, help="PLAN 78.83: thirty decisions, ten from each band of units on the board (symbols_play_78_83/<tag>)")
    ap.add_argument("--play89", default=None, help="PLAN 78.89: thirty decisions, fifteen of the lower many part and fifteen of the mid band (symbols_78_89/<tag>)")
    ap.add_argument("--groups88", default=None, help="PLAN 78.88: the twenty largest groups in words and the thirty decisions (symbols_78_88/<tag>)")
    ap.add_argument("--own90", default=None, help="PLAN 78.90: thirty decisions where the fit with the other side's cells picks otherwise than the old weights (symbols_78_90/<tag>)")
    ap.add_argument("--order91", default=None, help="PLAN 78.91: thirty positions where the reasons' order and the static order differ on the reference move (order_78_91/<tag>)")
    ap.add_argument("--choice92", default=None, help="PLAN 78.92: twenty positions where the played move is under the reference's best (reasoner_choice_78_92/<tag>)")
    ap.add_argument("--future95", default=None, help="PLAN 78.95: thirty positions: goals that fall, frames of one threat, picks of the weighted chooser (future_78_95/<tag>)")
    ap.add_argument("--prior96", default=None, help="PLAN 78.96: twenty positions where the equal moves differ in keeping the oracle's class, the fitted prior against the pattern pick (prior_78_96/<tag>)")
    ap.add_argument("--room97", default=None, help="PLAN 78.97: twenty class-losing equal moves that stay equal at the longest clock, ten tactical and ten slow (room_78_97/<tag>)")
    ap.add_argument("--equal98", default=None, help="PLAN 78.98: the trace of the equal moves (equal_78_98/<tag>): the five named positions (trace.txt) or ten more stuck moves (--equal98-set rest, cases.txt)")
    ap.add_argument("--equal98-set", default="named", choices=["named", "rest"])
    ap.add_argument("--record99", default=None, help="PLAN 78.99: the judged positions, the search under each measured record (record_78_99/<tag>/cases99.json, written to cases.txt)")
    ap.add_argument("--stack79", default=None, help="PLAN 79.00: twenty room positions where the first move under the stack loses the class (stack_79_00/<tag>/cases79.json, written to cases.txt)")
    ap.add_argument("--leaf01", default=None, help="PLAN 79.01: fifteen firm-room positions where the highest end-of-line chance is not the first move (leaf_79_01/<tag>/cases_leaf.json, written to cases.txt)")
    ap.add_argument("--cut02", default=None, help="PLAN 79.02: the judged positions of the cut and the mend (cut_79_02/<tag>/cases_cut.json, written to cases.txt)")
    ap.add_argument("--best03", default=None, help="PLAN 79.03: the fifteen positions where the best root move's own value at the depth is above the value the search reports (best_79_03/<tag>/cases_best.json, written to cases.txt)")
    ap.add_argument("--future94", default=None, help="PLAN 78.94: thirty positions with the oracle's class and frame, our goals and the judge's chances (future_78_94/<tag>)")
    ap.add_argument("--cut84", default=None, help="PLAN 78.84: the twenty flagged decisions of part one (symbols_play_78_84/<tag>)")
    ap.add_argument("--reread", action="store_true", help="PLAN 78.86: the twenty changed positions of ladder_78_86 beside the earlier run (ladder_78_77/selfplay_78_58_on)")
    ap.add_argument("--verdict87", action="store_true", help="PLAN 78.87: the ten lines whose verdict word changed (verdict_78_87/picked.json)")
    ap.add_argument("--tag", default="full")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--deep", action="store_true", help="the cases of plan 78.68 stages G and H (deep_heads_78_68/<tag>/case_rows.json)")
    a = ap.parse_args(argv)
    if a.verdict87:
        verdict_cases()
        return 0
    if a.reread:
        reread_cases()
        return 0
    if a.pick:
        pick_cases(a.pick)
        return 0
    if a.pick79:
        pick_cases_79(a.pick79)
        return 0
    if a.symbols:
        symbols_cases(a.symbols)
        return 0
    if a.reasons:
        reasons_cases(a.reasons)
        return 0
    if a.play82:
        play_cases(a.play82)
        return 0
    if a.play83:
        play_cases(a.play83, exp="symbols_play_78_83", cases_file="cases83.json",
                   title="THE THIRTY DECISIONS (plan 78.83): from the games of reasons v control, set B, of plan 78.82: in each band of units on the board the first decision per game where the pick differed from the search's first move, the first "
                         "ten games by id that have one. The reasons are 78.81's, weights are log-odds of being the move played among equals.")
        return 0
    if a.play89:
        play_cases(a.play89, shown=3, exp="symbols_78_89", cases_file="cases89.json", attacked=True,
                   title="THE THIRTY DECISIONS (plan 78.89): from the games of reasons v control of plan 78.82 (sets A then B): fifteen of the lower many part and fifteen of the mid band, the third decision of a game in the part where the pick differed "
                         "from the search's first move, one a game, the first fifteen games by id. The reasons are 78.81's, weights are log-odds of being the move played among equals.")
        return 0
    if a.cut84:
        cut_cases(a.cut84)
        return 0
    if a.groups88:
        groups_cases(a.groups88)
        return 0
    if a.own90:
        own_cases(a.own90)
        return 0
    if a.order91:
        order_cases(a.order91)
        return 0
    if a.choice92:
        choice_cases(a.choice92)
        return 0
    if a.future94:
        future_cases(a.future94)
        return 0
    if a.future95:
        future95_cases(a.future95)
        return 0
    if a.prior96:
        prior96_cases(a.prior96)
        return 0
    if a.room97:
        room97_cases(a.room97)
        return 0
    if a.equal98:
        equal98_cases(a.equal98, a.equal98_set)
        return 0
    if a.record99:
        record99_cases(a.record99)
        return 0
    if a.stack79:
        stack79_cases(a.stack79)
        return 0
    if a.leaf01:
        leaf01_cases(a.leaf01)
        return 0
    if a.cut02:
        cut02_cases(a.cut02)
        return 0
    if a.best03:
        best03_cases(a.best03)
        return 0
    if a.records:
        records_cases(a.records, fixed_from=a.fixed_from)
        return 0
    if a.deep:
        deep_cases(a.tag, a.n)
        return 0
    root = LM.out_root(a.tag)
    res = {d["id"]: d for d in (json.loads(x) for x in (root / "results.jsonl").read_text().splitlines() if x.strip())}
    ans = {d["id"]: d["accepted"] for d in (json.loads(x) for x in (root / "answers.jsonl").read_text().splitlines() if x.strip())}
    groups = json.loads((root / "groups.json").read_text())
    af = json.loads((root / "answer_features.json").read_text()) if (root / "answer_features.json").exists() else {}
    where = {i: g for g in groups for i in g["ids"]}
    lines = []
    for k, pid in enumerate(pick(list(res), groups, a.n), 1):
        d = res[pid]
        r = d["result"]
        b = chess.Board()
        for m in d["moves"]:
            b.push_uci(m)
        mine = our_move(r)
        g = where[pid]
        lines += [f"=== CASE {k}: {pid}  ({d['source']}, {d['units']} units, band {d['band']}, {'white' if b.turn else 'black'} to move)",
                  str(b), "",
                  f"our move at the last clock: {b.san(chess.Move.from_uci(mine)) if mine else None}   rung {r['rung']}" + (f", found at {r['found_at']:g} s" if r["found_at"] else ", not found"),
                  f"the oracle accepts: {', '.join(b.san(chess.Move.from_uci(m)) for m in ans[pid])}" + (f"   (types: {af[pid]})" if pid in af else ""),
                  f"the wake's search held {r['equals']} moves equal; the heads: {r['chose'].get('2 heads')}",
                  f"our deepest reading: {san_line(b, (r['deep'] or {}).get('names', []))}; events {(r['deep'] or {}).get('events')}; ended {(r['deep'] or {}).get('end')} after {(r['deep'] or {}).get('depth')} plies, cut {(r['deep'] or {}).get('cut')}",
                  f"features: {r['features']}",
                  f"group: size {g['size']}, key {g['key']}", ""]
    (root / "cases.txt").write_text("\n".join(lines))
    print(f"OBS ladder print: {min(a.n, len(res))} cases written to {root / 'cases.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
