"""PLAN 78.71 A: THE PRINT OF THE GOALS for the owner to read, and THE GAME'S OWN EDIT PATH. A printing script, NOT a layer, sheet, binding or measure file: it is the one place that renders a position and spells an edit in the
game's own words (the python-chess board, the library the rule kernel's own spelling comes from: names.py) so that a reader can set the position up and the what-if edits of plan 78.71 B can be made on the text the engine takes.
Everything else of the plan (subgoal_goals.py, subgoal_measure.py) is game-free and reaches the game only through the objects built here.

INTERFACE (what this module takes, what it returns, who calls it).
  GameEditor                  the EDIT PATH (the smallest that can edit the position the Rust read sees: the Rust read takes a position as text, so the edit is made on the text and the engine then accepts or refuses it).
                               `apply(text, edits)` -> (the new text, None) or (None, the reason it was rejected): an edit is ("remove", place) or ("relocate", place, place) as subgoal_goals gives them; the result must be a
                               legal position of the game (the board library's validity status, no right of capture in passing left, castling rights cleaned, and the engine's own acceptance of the text); otherwise the edit
                               is rejected and the reason returned (the callers count them). `relocate(text, a, b)` -> text or None, the helper subgoal_goals.bring_fix calls.
  render_goal(goal)            the text of one goal: its facts met or unmet, the cause of each unmet fact, the fix, in the words of the game's own renderer.
  main                         the print of the twenty positions of 78.70 to 78.73 (threat_print.load_twenty): the goals of both sides of each position and what they cost.

  .venv/bin/python -m catspace.incubator.analyst.discovery.subgoal_print [--out-dir D]
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import chess
import cs_engine as CE

from catspace.incubator.analyst.discovery import human_heads_data as HD
from catspace.incubator.analyst.discovery import subgoal_goals as SG
from catspace.incubator.analyst.discovery import threat_concrete as TC
from catspace.incubator.analyst.discovery import threat_list as TL
from catspace.incubator.analyst.discovery import threat_print as TP
from catspace.io import paths

OUT = ("subgoals_78_71",)


class GameEditor:
    """The edit path: edits on the game's text, accepted only if a legal position results (see the module doc)."""

    def __init__(self, variant: str):
        self.variant = variant
        self.count = {"applied": 0, "rejected": 0}
        self.reasons: dict = {}

    def build(self, text: str, edits: list) -> tuple:
        try:
            b = chess.Board(text)
            for e in edits:
                if e[0] == "remove":
                    sq = chess.parse_square(e[1])
                    if b.piece_at(sq) is None:
                        return None, "nothing stands on the place"
                    b.remove_piece_at(sq)
                elif e[0] == "relocate":
                    a, c = chess.parse_square(e[1]), chess.parse_square(e[2])
                    pc = b.piece_at(a)
                    if pc is None or b.piece_at(c) is not None:
                        return None, "the place to leave is empty or the place to go is not"
                    b.remove_piece_at(a)
                    b.set_piece_at(c, pc)
                else:
                    return None, "not an edit of the text"
            b.ep_square = None
            b.clean_castling_rights()
            st = b.status()
            if st != chess.STATUS_VALID:
                return None, "not a legal position: " + ", ".join(x.name.lower() for x in chess.Status if st & x)
            new = b.fen()
            CE.state(self.variant, new, [])                          # the engine's own acceptance of the text
            return new, None
        except Exception as ex:                                       # the engine or the board library refused the text
            return None, "refused: " + str(ex)[:60]

    def apply(self, text: str, edits: list) -> tuple:
        """A fix applied: counted (the legal position it gives, or the reason it was rejected)."""
        new, why = self.build(text, edits)
        if new is None:
            self.count["rejected"] += 1
            self.reasons[why] = self.reasons.get(why, 0) + 1
        else:
            self.count["applied"] += 1
        return new, why

    def relocate(self, text: str, a: str, b: str) -> "str | None":
        """A probe of where a unit could stand: not counted as an applied or a rejected fix."""
        return self.build(text, [("relocate", a, b)])[0]


def unit_text(u: "tuple | None") -> str:
    return "?" if u is None else f"{u[0]} {u[1]} {u[2]}"


def edit_text(e: tuple) -> str:
    if e[0] == "remove":
        return f"take out the unit on {e[1]}"
    if e[0] == "relocate":
        return f"bring the unit on {e[1]} to {e[2]}"
    return f"play {e[1]} first"


def render_goal(g: dict, k: int) -> str:
    row = g["row"]
    st = "no stake (goal kind)" if g["stake"] is None else f"stake {g['stake']:+.3f}"
    head = f"  GOAL {k} [{g['source']}]: win the {unit_text(g['target'])} ({st}); the row: {row.get('verdict')} line {' '.join(g['line']) if g['line'] else '(none)'}"
    if row.get("outcome"):
        o = row["outcome"]
        head += f"; the taker takes {','.join(o['taker_won']) or 'nothing'} and loses {','.join(o['taker_lost']) or 'nothing'} (kinds by the line search's letters; net {g['net']}, resolution {row.get('resolution', 0):.0f})"
    out = [head]
    for f in g["facts"]:
        vs = ", ".join(f"{n}={unit_text(u)}" for n, u in f["vars"].items() if u is not None)
        out.append(f"      {f['name']}({', '.join(f['vars'])}) {f['status'].upper()}" + (f"   [{vs}]" if vs else ""))
        if f["cause"] and f["status"] in ("unmet", "unread"):
            c = f["cause"]
            sp = ("; stopped by " + ", ".join(unit_text(u) for u in c["stoppers"])) if c["stoppers"] else ""
            vic = ("; it would uncover/leave loose " + unit_text(c["victim"])) if c["victim"] and c["type"] in ("only_guard", "uncovered_line", "loose", "tied") else ""
            out.append(f"          cause: {c['type']}: {c['words']}{sp}{vic}")
            if f["fix"]:
                out.append(f"          fix: {f['fix']['words']}: " + "; ".join(edit_text(e) for e in f["fix"]["edits"]))
            else:
                out.append("          fix: none proposed")
    return "\n".join(out)


def positions_c(items: list, lines: TC.Lines) -> list:
    return TC.concrete_lists(items, lines)


def main(argv: "list | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args(argv)
    out = Path(a.out_dir) if a.out_dir else Path(paths.experiment(*OUT))
    out.mkdir(parents=True, exist_ok=True)
    z, lists, allg, picked, n_first = TP.load_twenty()
    rec = TC.Record.load()
    lines1 = TC.Lines(rec, threads=1)
    nm = TP.letter_names()
    facts = SG.Facts(HD.VARIANT)
    ed = GameEditor(HD.VARIANT)
    roles = z["names"]["roles"]
    text = [f"THE GOALS OF PLAN 78.71 A (the same twenty positions as plan 78.73 E): for each position the goals of the side to move and of the other side (its threats after the side to move passes: hypothetical). A GOAL is a threat of ours that does not work yet: "
            "the facts that would make it work (units as variables), met or unmet; for each unmet fact what stops it (from the line that was read and the graph) and the proposed fix. Threats that already work are listed apart (not goals). "
            f"The judge of the lines: the measured record {rec.source}; line depth {lines1.depth} plies, node fuse {lines1.fuse}."]
    text.append("THE FOUR CLAIMS each goal makes (spec 7), and the referee of each (our own reasoning and estimate; no oracle): ")
    for cl in SG.subgoal_claims({"edits": [], "key": ""}):
        text.append(f"  - {cl['name']}: {cl['statement']}. Referee: {cl['referee']}.")
    tally = {"positions": 0, "goals": 0, "by_source": {}, "unmet": {}, "with_fix": 0, "no_fix": 0, "works": 0, "wall": []}
    for k, (phase, j) in enumerate(picked, 1):
        gid = z["sample_ids"][z["games"].index(int(z["g"][j]))]
        moves = allg[gid]["moves"]
        ply = int(z["ply"][j])
        item = {"start": None, "moves": moves[:ply], "raw": z["rows"][j], "names": z["names"], "list": lists[j]}
        mover = z["rows"][j]["mover"]
        t0 = time.perf_counter()
        c = positions_c([item], lines1)[0]
        t_conc = time.perf_counter() - t0
        both = c["mover"] + c["other"]
        spread = float(z["spread"][j])
        rk = TL.rank_high_of(both)
        for r in both:
            r["b_spread"] = TL.band_of(r, spread, None, "spread")
            r["b_rank"] = TL.band_of(r, None, rk.get(id(r), False), "rank")
        text.append("")
        text.append(f"=== POSITION {k} of {len(picked)} ({phase} play) ===")
        text.append(TP.render(z, j, gid, moves))
        for side, label in ((mover, "the side to move"), (1 - mover, "the other side (after a pass)")):
            t1 = time.perf_counter()
            gl = SG.goals(item, side, c, facts, ed)
            wk = SG.works(item, side, c)
            dt = time.perf_counter() - t1
            tally["wall"].append(dt)
            tally["positions"] += 1
            tally["goals"] += len(gl)
            tally["works"] += len(wk)
            text.append(f"--- GOALS of {roles[side]} ({label}): {len(gl)} goals, {len(wk)} threats already work; concrete read {t_conc:.2f} s, goals {dt:.2f} s (one thread) ---")
            for n, g in enumerate(gl, 1):
                tally["by_source"][g["source"]] = tally["by_source"].get(g["source"], 0) + 1
                for f in g["facts"]:
                    if f["status"] in ("unmet", "unread"):
                        key = f"{f['name']}:{f['cause']['type'] if f['cause'] else '-'}"
                        tally["unmet"][key] = tally["unmet"].get(key, 0) + 1
                if g["fixes"]:
                    tally["with_fix"] += 1
                else:
                    tally["no_fix"] += 1
                text.append(render_goal(g, n))
            for w in wk:
                text.append(f"  already works: {w['kind']} {w['place']} ({'line ' + ' '.join(w['line']) if w.get('line') else 'goal row'}), not a goal")
        c = None
    ws = sorted(tally["wall"])
    text.append("")
    text.append(f"OBS goals78_71: positions {len(picked)}, goal lists {tally['positions']} (both sides), goals {tally['goals']}; threats already working (not goals) {tally['works']}")
    text.append(f"OBS goals78_71: goals by source: " + "; ".join(f"{k} {v}" for k, v in sorted(tally["by_source"].items())))
    text.append(f"OBS goals78_71: unmet facts by fact and cause: " + "; ".join(f"{k} {v}" for k, v in sorted(tally["unmet"].items())))
    text.append(f"OBS goals78_71: goals with a fix proposed {tally['with_fix']}, with none {tally['no_fix']}")
    text.append(f"OBS goals78_71: time of the goals per position and side (one thread, the concrete read apart): median {ws[len(ws) // 2]:.2f} s, mean {sum(ws) / len(ws):.2f} s, slowest {ws[-1]:.2f} s")
    text.append(f"OBS goals78_71: the edit path: probes and edits applied {ed.count['applied']}, rejected {ed.count['rejected']}")
    (out / "print.txt").write_text("\n".join(text) + "\n")
    print("\n".join(text))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
