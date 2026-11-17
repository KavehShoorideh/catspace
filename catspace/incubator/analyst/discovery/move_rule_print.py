"""PLAN 78.75 B: THE TRACE OF THE RULE FOR CHOOSING A MOVE, printed for the owner to read. A printing script, NOT a layer, sheet, binding, measure file or the rule's own module: it is the one place that renders a position and
spells moves in the game's own words (the board library the rule kernel's own spelling comes from, as subgoal_print.py does), so that a reader can set the position up and judge the line of thought against the board.

For the SAME twenty positions as plan 78.74 (threat_print.load_twenty): the threats against the side to move (confirmed, with their kept answers), the moves the search holds equal (`solve_equals_batch`, the 78.67 form), the first
goals by stoppers, and the decision of move_rule.choose as a line of thought: this threat, that answer, so this move; or no threat, this goal, this fact, so this move. Also, one line each, the pick of the rule with step 2 off
(threats only) and with step 1 and 3 off (goals only). The held subgoal starts empty at every position (a position is a single decision here); the trace of a carried goal is in the match logs.

  .venv/bin/python -m catspace.incubator.analyst.discovery.move_rule_print [--out-dir D] [--seconds S]
"""
from __future__ import annotations

from catspace.incubator.analyst.discovery import tiebreak_measure as TM          # first: one thread a process

import argparse
import re
import time
from pathlib import Path

import chess
import cs_engine as CE

from catspace.incubator.analyst.discovery import human_heads_data as HD
from catspace.incubator.analyst.discovery import move_rule as MR
from catspace.incubator.analyst.discovery import odds_measure as OM
from catspace.incubator.analyst.discovery import subgoal_goals as SG
from catspace.incubator.analyst.discovery import subgoal_play_measure as PM
from catspace.incubator.analyst.discovery import subgoal_print as SP
from catspace.incubator.analyst.discovery import subgoal_units as SU
from catspace.incubator.analyst.discovery import threat_concrete as TC
from catspace.incubator.analyst.discovery import threat_list as TL
from catspace.incubator.analyst.discovery import threat_print as TP
from catspace.io import paths

OUT = ("move_rule_78_75",)
RUN = re.compile(r"(?:\b[a-h][1-8][a-h][1-8][qrbn]?\b ?)+")


def spell(text: str, board: chess.Board) -> str:
    """Move names in a text as the game's notation: a run of names is a line, played from the position (a line of the side not to move starts after a pass)."""
    def one(m: "re.Match") -> str:
        names = m.group(0).split()
        for first in (False, True):
            b = board.copy()
            try:
                if first:
                    b.push(chess.Move.null())
                out = []
                for n in names:
                    mv = chess.Move.from_uci(n)
                    out.append(b.san(mv))
                    b.push(mv)
                return " ".join(out) + (" " if m.group(0).endswith(" ") else "")
            except Exception:
                continue
        return m.group(0)
    return RUN.sub(one, text)


def threat_text(r: dict, board: chess.Board) -> str:
    st = "goal row" if r.get("goal") else ("no stake" if r["stake"] is None else f"stake {r['stake']:+.3f}")
    ans = MR.answers_of(r)
    head = f"    - {r.get('kind') or 'the game'} {r.get('place') or ''} ({st}; line {spell(' '.join(r['line']), board) if r.get('line') else '-'}; {r.get('ended', '')})"
    return head + ("; kept answers: " + ", ".join(spell(m, board) + (" [counter-threat]" if "counter" in x.get("how", "") else "") for m, x in ans.items()) if ans else "; kept answers: none")


def main(argv: "list | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--seconds", type=float, default=TM.SECONDS)
    a = ap.parse_args(argv)
    out = Path(a.out_dir) if a.out_dir else Path(paths.experiment(*OUT))
    out.mkdir(parents=True, exist_ok=True)
    z, lists, allg, picked, _n_first = TP.load_twenty()
    rec = TC.Record.load()
    lines1 = TC.Lines(rec, threads=1)
    facts = SG.Facts(HD.VARIANT)
    ed = SP.GameEditor(HD.VARIANT)
    base = OM.players_of(TM.WORTH_GEN, OM.out_root() / "full")
    roles = z["names"]["roles"]
    text = ["THE TRACE OF PLAN 78.75 B: the rule for choosing a move (SYSTEM_SPEC 8) on the twenty positions of 78.74. For each: the threats against the side to move, the moves the search holds equal "
            f"({a.seconds} s, the 78.67 form), the first goals by stoppers, and the decision as a line of thought. Judge the line against the board."]
    tally = {"branch": {}, "n": 0, "ties": 0, "changed": 0, "wall": []}
    for k, (phase, j) in enumerate(picked, 1):
        gid = z["sample_ids"][z["games"].index(int(z["g"][j]))]
        moves_all = allg[gid]["moves"]
        ply = int(z["ply"][j])
        moves = list(moves_all[:ply])
        item = {"start": None, "moves": moves, "raw": z["rows"][j], "names": z["names"], "list": lists[j]}
        mover = z["rows"][j]["mover"]
        t0 = time.perf_counter()
        q0 = lines1.cost["queries"]
        c = TC.concrete_lists([item], lines1)[0]
        SU.every_unit([item], [[mover]], [c], facts, lines1, [SU.stake_closure(z["stake"][j], z["kind_cols"])])
        goals = SG.goals(item, mover, c, facts, ed)
        t_read = time.perf_counter() - t0
        n_q = lines1.cost["queries"] - q0
        n_nodes = sum(r.get("nodes", 0) or 0 for pool in (c["mover"], c["other"], c["exchange"], c["dropped_rows"]) for r in pool)
        rk = TL.rank_high_of([r for r in c["mover"] + c["other"] if not r.get("goal")])
        for r in c["mover"] + c["other"]:
            r["big"] = bool(rk.get(id(r), False))
        row = PM.equal_set(CE.solve_equals_batch([(OM.VARIANT, None, moves)], a.seconds, worth=dict(base["worth"]), ours_first=False, valuation="odds record", threads=1, ledger_share=TM.LEDGER_SHARE)[0])
        tied = [row["move"]] + sorted(m for m in row["equals"] if m != row["move"]) if row["move"] else []
        board = chess.Board(CE.state(HD.VARIANT, None, moves)[0])
        text += ["", f"=== POSITION {k} of {len(picked)} ({phase} play) ===", TP.render(z, j, gid, moves_all)]
        threats = MR.threat_rows(c)
        text.append(f"THREATS AGAINST {roles[mover].upper()} (the side to move), confirmed: {len(threats)}")
        for r in threats:
            text.append(threat_text(r, board) + (f"  [big: {r.get('big')}]" if not r.get("goal") else ""))
        text.append(f"TIED MOVES (held equal by the search): {len(tied)}: " + ", ".join(board.san(chess.Move.from_uci(m)) for m in tied))
        ranked = MR.rank_goals(goals)
        text.append(f"GOALS of {roles[mover]} by stoppers (the first three of {len(goals)}):")
        for g in ranked[:3]:
            text.append("    - " + spell(MR.goal_words(g), board) + (f"; line {spell(' '.join(g['line']), board)}" if g.get("line") else ""))
        if len(tied) < 2:
            text.append("DECISION: one move only; the rule has no choice to make.")
            tally["n"] += 1
            continue
        pos = MR.Position(facts, HD.VARIANT, None, moves, roles, mover)
        pick, reason, held = MR.choose(pos, tied, c, goals, None)
        tally["n"] += 1
        tally["ties"] += 1
        tally["changed"] += pick != tied[0]
        tally["branch"][reason.branch] = tally["branch"].get(reason.branch, 0) + 1
        tally["wall"].append(t_read)
        text.append(f"COST OF THIS DECISION: {n_q} line queries (the threat list and the line read of every unit of theirs), {n_nodes} nodes read; {t_read:.2f} s")
        text.append(f"THE RULE'S LINE OF THOUGHT [step {reason.branch}]: " + spell(str(reason), board))
        text.append(f"  the move: {board.san(chess.Move.from_uci(pick))} ({'the search\'s own move' if pick == tied[0] else 'not the search\'s own move ' + board.san(chess.Move.from_uci(tied[0]))}); subgoal held after it: "
                    + ("none" if held is None else (f"refute the threat on our {held['target'][1]} {held['target'][2]}" if held["origin"] == "refute" else f"win the {held['target'][0]} {held['target'][1]} on {held['target'][2]}")))
        for nm, st in (("threats only (step 2 off)", (1, 3)), ("goals only (steps 1, 3 off)", (2,))):
            p2, r2, _h = MR.choose(MR.Position(facts, HD.VARIANT, None, moves, roles, mover), tied, c, goals, None, st)
            text.append(f"  {nm}: {board.san(chess.Move.from_uci(p2))} [step {r2.branch}]")
    ws = sorted(tally["wall"])
    text.append("")
    text.append(f"OBS trace78_75: positions {tally['n']}, with two or more equal moves {tally['ties']}; the rule's pick differs from the search's own move in {tally['changed']}; steps that fired: " + "; ".join(f"{b} {n}" for b, n in sorted(tally["branch"].items())))
    if ws:
        text.append(f"OBS trace78_75: reading time per position with a tie (one thread, the concrete read, the line read of every unit of theirs and the goals): median {ws[len(ws) // 2]:.2f} s, mean {sum(ws) / len(ws):.2f} s, slowest {ws[-1]:.2f} s")
    (out / "trace.txt").write_text("\n".join(text) + "\n")
    print("\n".join(text))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
