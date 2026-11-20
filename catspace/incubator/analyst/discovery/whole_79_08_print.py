"""PLAN 79.08 PART E: THE POSITION TEST OF THE WHOLE WAKE (SYSTEM_SPEC 8 all seven steps, "the whole is judged, not the piece"; piece: the wake). A PRINTING SCRIPT: the only place the game's own text appears. It prints, for each position, what the Rust wake says -- the threat list of both sides, the goals with what
stops each and the subgoal, the moves named and their lines, the search's line, the choice and why, the time by step -- beside the old player's move. It runs the wake (one binding, `BaseGraph.wake_decide`) and the old player (`solve_equals_batch`) and judges nothing: no number is written here that the engine did not print; no tuning.
Order of work (the owner's): run only after the Rust integration tests passed. Outputs artifacts/experiments/whole_79_08/.

  python -m catspace.incubator.analyst.discovery.whole_79_08_print missed|puzzles|all [--seconds 0.2 2.0]
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import chess

import cs_engine as CE
from catspace.incubator.analyst.discovery import frames_measure as OLD
from catspace.incubator.analyst.discovery import graph_layer2_measure as G2M
from catspace.incubator.analyst.discovery import human_heads_data as HD
from catspace.incubator.analyst.discovery import ledger_puzzles as LP
from catspace.incubator.analyst.discovery import odds_measure as OM
from catspace.incubator.analyst.discovery import subgoal_measure as SM
from catspace.incubator.analyst.discovery import subgoal_play_measure as SPM
from catspace.incubator.analyst.discovery import symbols_play as SP
from catspace.incubator.analyst.discovery import tactical_state as TS
from catspace.incubator.analyst.discovery import tiebreak_measure as TM
from catspace.io import paths

MISSED = "1k1rr3/2p1qpb1/p1R3pp/3n4/7P/4PP1B/PP1Q1P1B/1K5R b - - 0 21"
MISSED_ANSWER, MISSED_OLD = "d5c3", "e7h4"
PIECES = ["carried", "frames", "threats", "goals", "two_threat_goals", "reasons_pick"]
OUT = ("whole_79_08",)


def out_dir() -> Path:
    p = Path(paths.experiment(*OUT))
    p.mkdir(parents=True, exist_ok=True)
    return p


class World:
    """Everything the wake is given and none of it is made here: the model (learned profile, kind values, bands, rules) as the threat reader built it, the learned seeds, the record of worths, the reasons and their weights (78.81, read, never refitted), the table of the chance of a goal (78.74, read)."""

    def __init__(self) -> None:
        t0 = time.perf_counter()
        rec = TS.human_kind_record()
        sample = TS.load_sample(Path(paths.experiment(*TS.HEADS_DIR)), TS.PER_SHARD, 0)
        self.model = TS.make_model(sample, rec, 0)
        picked = OLD.pick_concepts(Path(paths.experiment("unit_fates_78_54", "games_s5016_n500")), TS.SEEDS)
        self.keys = [int(k) for _dc, k, _d, _z in picked]
        self.goal = G2M.goal_kind_of(HD.VARIANT)
        values = json.loads(Path(paths.experiment("ladder_78_76", "human_78_59", "record.json")).read_text())["heads"]["odds_effect"]["kind_values"]
        names = OM.kernel_names()
        self.worth = {names[k]: int(v) for k, v in values.items()}
        root = SP.src_root()
        reasons = json.loads((root / "reasons.json").read_text())
        keys = sorted({int(r["key"]) for r in reasons})
        col = {k: j for j, k in enumerate(keys)}
        self.reasons = (keys, [(col[int(r["key"])], r["side"], r["dir"], float(r["weight"])) for r in reasons], 2, [0, 1, 2, 3], SP.TIE_DIGITS)
        z = pickle.loads((Path(paths.experiment("subgoals_78_74", "full600")) / "goals.pkl").read_bytes())
        goals = [(gi, pos, g) for gi, r in enumerate(z["res"]) for pos in r["positions"] for g in pos["goals"]]
        bykey, byfacts, tot = SM.table_of(goals, 2)                        # fold 2: no game is left out (the table of the whole run; read, not fitted again)
        self.chances = ([(k, int(v[0]), int(v[1])) for k, v in bykey.items()], [(k, int(v[0]), int(v[1])) for k, v in byfacts.items()], (int(tot[0]), int(tot[1])), 20)
        self.setup_seconds = time.perf_counter() - t0

    def wake(self, text: str, seconds: float, pieces: list = PIECES) -> dict:
        t0 = time.perf_counter()
        out = json.loads(self.model.wake_decide(text, [], seconds, 64, self.worth, {}, self.goal, pieces, 0, 0, self.keys, (OLD.MAX_SEED_SIZE, TS.FUSE, TS.ROW_FUSE, TS.FUSE), self.chances, self.reasons, TM.LEDGER_SHARE))
        out["wall"] = time.perf_counter() - t0
        return out

    def old(self, text: str, seconds: float) -> dict:
        t0 = time.perf_counter()
        row = CE.solve_equals_batch([(OM.VARIANT, text, [])], seconds, worth=self.worth, ours_first=False, valuation="odds record", threads=1, ledger_share=TM.LEDGER_SHARE)[0]
        d = SPM.equal_set(row)
        d["wall"] = time.perf_counter() - t0
        return d


def say_row(r: dict) -> str:
    resp = "; ".join(f"{x['move']} {x['state']} ({x['net']})" for x in r["responses"][:6]) or "none"
    return f"    [{r['list']}] {r['unit']}: {r['class']}, verdict {r['verdict']}, net {r['net']}, ended: {r['ended']}{', CARRIED' if r['carried'] else ''}; line {' '.join(r['line'][:8])}; responses: {resp}"


def say_goal(x: dict) -> str:
    fixes = "; ".join(f"{f['kind']}: {' '.join(f['moves'][:8]) or '-'}" for f in x["fixes"])
    ch = f", chance {x['chance']['chance']:.3f} (n {x['chance']['n']}, {x['chance']['source']})" if x["chance"] else ""
    sub = f" SUBGOAL of {x['parent']} via {x['via']} (depth {x['depth']})" if x["parent"] else ""
    return f"    {'*' if x['chosen'] else ' '} {x['side']} on {x['target']} [{x['source']}]{sub}: unmet {', '.join(x['unmet']) or '-'}; stops it: {', '.join(x['stoppers']) or '-'}; things in the way {x['in_the_way']}{ch}; fixes {fixes or '-'}"


def report(title: str, text: str, w: dict, old: dict, answer: "str | None", seconds: float, lines: list) -> dict:
    log = w["log"]
    lines.append(f"=== {title} (clock {seconds} s)")
    lines.append(f"  position: {text}")
    lines.append(f"  THREAT LIST: " + ("(no row)" if not log["rows"] else ""))
    lines += [say_row(r) for r in log["rows"]]
    lines.append("  GOALS (* the chosen goal, the subgoal of the moment; goals are ordered by the proposer, then chosen by the fewest things in the way):")
    lines += [say_goal(x) for x in log["goals"][:24]]
    if log["far_goal"]:
        lines.append(f"  the far goal the chosen subgoal serves: {log['far_goal']}")
    lines.append(f"  THEIR LARGEST STANDING THREAT: {log['their_threat'] or 'none standing'}")
    lines.append(f"  NAMED, OURS (their line read, not refuted): " + ", ".join(f"{k['move']} [{k['verdict']}, net {k['net']}]" for k in log["named_mine"]) + "")
    lines.append(f"  NAMED, THEIRS (at the ply after ours): " + ", ".join(f"{k['move']} [{k['verdict']}, net {k['net']}]" for k in log["named_theirs"]))
    lines.append(f"  REFUTED, NOT NAMED: " + ", ".join(f"{r['move']}{' (theirs)' if r['theirs'] else ''}: {r['ended']}" for r in log["refuted"]) + f"; NOT READ (over the budget): " + ", ".join(m['move'] for m in log["unlisted"]))
    lines.append(f"  SEARCH: depth {w['depth']}, value {w['value']}, nodes {w['nodes']}; line {' '.join(w['chain'][0]['line'][:10])}; held equal: {' '.join(w['equal'])}")
    lines.append(f"  THE WAKE PLAYS {w['move']}: {log['why']}")
    for c in w["chain"][1:]:
        lines.append(f"    claim [{c['by']}]: {' '.join(c['line'][:10])} -- ended: {c['ended']}")
    top = [m for m in log["moves"] if m["equal"]][:8]
    lines.append("  CANDIDATES HELD EQUAL (value as the null window reads it; what each serves or answers; the chance of its goal; the reasons' score): " + " | ".join(
        f"{m['move']} v={m['value']} serves {m['serves'][:2] or '-'} answers {m['answers'][:2] or '-'} chance {m['chance']} reasons {m['reasons_score']}" for m in top))
    lines.append("  TIME BY STEP: " + "; ".join(f"step {s['step']} {s['piece']} {s['seconds']:.3f} s ({s['note']})" for s in w["steps"]))
    lines.append("  NOTES: " + " | ".join(log["notes"]))
    lines.append(f"  OLD PLAYER plays {old['move']} (depth {old['depth']}, equal set {' '.join(old['equals'])}), {old['wall']:.3f} s; wake wall {w['wall']:.3f} s")
    if answer:
        lines.append(f"  THE ANSWER is {answer}: the wake {'HITS' if w['move'] == answer else 'misses'} it, the old player {'HITS' if old['move'] == answer else 'misses'} it; the answer is {'held equal by the search' if answer in w['equal'] else 'not among the moves the search holds equal'}"
                     f"{' and named by the wake' if any(k['move'] == answer for k in log['named_mine']) else ' and not named'}" + (f"; it is a fix under: {[m['serves'] for m in log['moves'] if m['move'] == answer][0]}" if any(m['move'] == answer for m in log['moves']) else ""))
    lines.append("")
    return {"title": title, "text": text, "seconds": seconds, "wake": w, "old": old, "answer": answer}


def positions_missed() -> list:
    t = CE.state(OM.VARIANT, MISSED, [])[0]
    t2 = CE.state(OM.VARIANT, MISSED, [MISSED_OLD])[0]
    return [("the missed puzzle, the position itself", t, MISSED_ANSWER), (f"the missed puzzle after the old player's {MISSED_OLD}", t2, None)]


def positions_puzzles() -> list:
    out = []
    for i, row in enumerate(LP.dev_rows(14, offset=11)):
        b = chess.Board(row["FEN"])
        mv = row["Moves"].split()
        b.push_uci(mv[0])
        out.append((f"puzzle {i + 1} (id {row['PuzzleId']}, themes {row['Themes']}, rating {row['Rating']})", CE.state(OM.VARIANT, b.fen(), [])[0], mv[1]))
    return out


def main(argv: "list | None" = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=["missed", "puzzles", "all"])
    ap.add_argument("--seconds", type=float, nargs="+", default=[0.2, 2.0])
    a = ap.parse_args(argv)
    world = World()
    done: list = []
    for which, pos in (("missed", positions_missed), ("puzzles", positions_puzzles)):
        if a.which not in (which, "all"):
            continue
        lines: list = [f"WHOLE WAKE 79.08, {which}: wake with the pieces {', '.join(PIECES)}; setup {world.setup_seconds:.1f} s; one run, directions only; nothing tuned", ""]
        for title, text, answer in pos():
            for sec in a.seconds:
                w, o = world.wake(text, sec), world.old(text, sec)
                done.append(report(title, text, w, o, answer, sec, lines))
        (out_dir() / f"{which}_print.txt").write_text("\n".join(lines))
        print("\n".join(lines))
    (out_dir() / "decisions.json").write_text(json.dumps(done))
    return 0


if __name__ == "__main__":
    sys.exit(main())
