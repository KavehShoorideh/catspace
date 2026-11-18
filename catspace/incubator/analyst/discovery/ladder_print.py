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

PLAN 78.80 `--symbols TAG` (`symbols_cases`): THE SYMBOLS AS READABLE CASES. Part A: the twelve positions of 78.78 with, per tied move, the passing events (patterns made true or false, for each side) it carries, in words; part B: the ten
strongest passing patterns (a rule fixed in the function's doc) each drawn on three positions where the move played carries the event, the units of one instance in brackets. Reads artifacts/experiments/symbols_78_80/<tag>/ and writes
cases.txt there. The only place of the symbols' files where the game's renderer and names appear.
PLAN 78.81 `--reasons TAG` (`reasons_cases`): THE REASONS AS READABLE CASES (symbols_reasons_measure.py; reasons.json and ahead.json under symbols_78_81/<tag>/, the 78.80 sample, events and keys). Part A: THE STRONGEST TWENTY REASONS by
|weight| in the joint fit with the controls (a rule fixed here before the cases were read), each in words and drawn on the first three CONFIRM positions, in the file's order, where the move played carries the event, the units of one
instance in brackets, with whether the move takes a unit and what kind of unit moves said beside it. Part B: the twelve positions of 78.78 (subgoal_pick_78_78 case_rows.json) with, per tied move, the reasons it carries in words, whether it
takes a unit, the kind of unit that moves and the sum the pick reads; the accepted moves are the answers file, for the human reader only. Writes cases.txt beside the reasons.
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
    ap.add_argument("--tag", default="full")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--deep", action="store_true", help="the cases of plan 78.68 stages G and H (deep_heads_78_68/<tag>/case_rows.json)")
    a = ap.parse_args(argv)
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
