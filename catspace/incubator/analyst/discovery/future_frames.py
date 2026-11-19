"""PLAN 78.95 PART ONE (b) AND (c), THE DATA OF PART TWO (SYSTEM_SPEC 7 "the future imagination layer" and "the prior learns the tactical situations ... probabilistic outcomes of frames", 9 "the oracle labels positions and frames in sleep, for the prior", 5 the line reading,
8 the threat list). A SLEEP-SIDE READING of the labels of 78.94: it reads the oracle's stored best lines (`pv` in the label records) and calls the oracle NEVER (no import of ladder_oracle by this file; future_labels, which it uses for the reader, the lines and the class, does). A sandbox:
nothing is written to rules/learned or data/derived; every output goes under artifacts/experiments/future_78_95/<tag>/frames/. Game-free: a kind, a place or a move is a word or a number read from the engine at run time; the game is named only through the pieces that exist.

WHAT IT MAKES. For every labelled position whose class for the side to move X is winning or losing (class 1 or -1; the other side is Y):
  the FRAME of the side that the oracle says is winning there, read in our own facts from the oracle's best line (as 78.94 read it, at the positions after the winning side's own moves in the line: after 1 and 3 plies for X, after 2 and 4 plies for Y, the line cut at the first
  move the engine's rules refuse). Its KIND (decided here, before any count; the first that applies):
    two      at some step the winning side holds a goal (two threats that no one reply meets; the COMPLETE reply test of 78.95 part one (a): future_goals.goals_of(complete=True));
    end      at some step the forcing read finds a forced end of the game for the winning side;
    one      at some step a confirmed threat of the winning side stands (no goal);
    take     the other side (the victim) has fewer units after some step than at the root: a unit is left to be taken or is taken along the line, no threat read;
    none     nothing our reading names.
  Its STAKE (decided here): the units the victim loses along the read line less the units the winning side loses, each in the measured record's worth of its kind (the judge's own worths; no worth is assumed), read between the root and the last step read.
For the positions where X is winning, also the three questions of part one (b), kept apart (a measure on the oracle's first move; the oracle is not called, its stored line is read):
  q1  does our search at the play clock (0.2 s, ledger share 0.4, the search of the matches) play the oracle's first move, or hold it equal to its best? (solve_equals_batch, the equal set)
  q2  given the oracle's first move, does our line reading confirm the gain? (the line that starts with that move, read to a settled end by the line reading, verdict CONFIRMED for the side that plays it)
  q3  does our threat list name it? (a taking, or the quiet first move of a threat in two moves, or a forced end: future_labels.Labeller.sees)
and whether the point is STABLE (future_labels.stable_points: the class holds at the next sampled position of the same side).

INTERFACE (written before the bodies).
  KINDS, KIND_WORDS, STEPS            the five kinds, their words, and the line steps read for the winning side.
  frame_kind(steps, n_victim_root) -> kind    the rule above on the step summaries {j: {"goal", "forced", "threats", "n_v"}}.
  material_net(root_board, last_board, owner, victim, worth_of_kind) -> int   the stake: the units the victim lost less the units `owner` lost between two boards, in the record's worths (kinds by their engine symbol).
  FrameReader()                       per process: the label reader, the complete reply reader, the line reader and the search's worth.
    .read_game(moves, rec) -> [ {"ply", "cls", "win", "lose", "stable_win", "stable_lose", "owner": "x" | "y", "kind", "net", "steps", "pv_n", "q1", "q2", "q3"} ]   per position with class 1 or -1 (q1 to q3 for class 1 only)
  run(root, tag, workers, say)        the Pool over all cores; one file a game (a game with a file is not redone), then frames.jsonl.
  load(root) -> {(ord, ply): record}  for the measure and the judge's data.
  Called by: future_measure.py (stages fair, kinds, fitk, scorek, human95, cases95), the command line (`frames`).

  .venv/bin/python -m catspace.incubator.analyst.discovery.future_frames --src future_78_94/full --tag short --per-part 6 | --tag full [--workers K]
"""
from __future__ import annotations

from catspace.incubator.analyst.discovery import tiebreak_measure as TM          # first: it sets one thread a process before numpy is loaded

import argparse
import json
import multiprocessing
import os
import time
from pathlib import Path

import cs_engine as CE

from catspace.incubator.analyst.discovery import future_goals as FG
from catspace.incubator.analyst.discovery import future_labels as FL
from catspace.incubator.analyst.discovery import human_heads_data as HD
from catspace.incubator.analyst.discovery import odds_measure as OM
from catspace.incubator.analyst.discovery import subgoal_play_measure as PM
from catspace.incubator.analyst.discovery import threat_list as TL
from catspace.io import paths

OUT = ("future_78_95",)
SRC = ("future_78_94", "full")
KINDS = ("take", "one", "two", "end", "none")
KIND_WORDS = {"take": "a unit is left to be taken", "one": "one threat", "two": "two threats", "end": "a forced end", "none": "none named"}
STEPS = {"x": (1, 3), "y": (2, 4)}


def out_root(tag: str) -> Path:
    p = Path(paths.experiment(*OUT, tag))
    p.mkdir(parents=True, exist_ok=True)
    return p


def frame_kind(steps: dict, n_victim_root: int) -> str:
    """The kind of a frame by the rule in the module doc."""
    if any(s["goal"] for s in steps.values()):
        return "two"
    if any(s["forced"] for s in steps.values()):
        return "end"
    if any(s["threats"] > 0 for s in steps.values()):
        return "one"
    if steps and min(s["n_v"] for s in steps.values()) < n_victim_root:
        return "take"
    return "none"


def _count(board: list, owner: int) -> dict:
    out: dict = {}
    for o, k, _p in board:
        if o == owner:
            out[k] = out.get(k, 0) + 1
    return out


def material_net(root_board: list, last_board: list, owner: int, victim: int, worth_of_kind) -> int:
    """The stake: what the victim lost between the two boards less what `owner` lost, each unit in its kind's worth (`worth_of_kind(kind index) -> int`)."""
    def lost(side: int) -> int:
        a, b = _count(root_board, side), _count(last_board, side)
        return sum(worth_of_kind(k) * max(0, n - b.get(k, 0)) for k, n in a.items())
    return lost(victim) - lost(owner)


class FrameReader:
    """See the module doc."""

    def __init__(self):
        self.lab = FL.Labeller()
        self.ctx = FG.Complete(HD.VARIANT, self.lab.reader, self.lab.lines, self.lab.record)
        from catspace.incubator.analyst.discovery import symbols_play as SP
        self.cut = SP.LineCut(self.lab.record)
        self.sym = OM.kernel_names()
        self.search_worth = dict(OM.players_of(TM.WORTH_GEN, OM.out_root() / "full")["worth"])
        self.names = None

    def worth_of(self, idx: int) -> int:
        return int(self.lab.record.worth.get(self.sym.get(self.names["kinds"][idx], ""), 0))

    def q1(self, moves: list, first: str) -> dict:
        raw = CE.solve_equals_batch([(OM.VARIANT, None, list(moves))], TM.SECONDS, worth=dict(self.search_worth), ours_first=False, valuation="odds record", threads=1, ledger_share=TM.LEDGER_SHARE)[0]
        row = PM.equal_set(raw)
        return {"plays": row["move"] == first, "equal": first in row["equals"] or row["move"] == first, "n_equal": len(row["equals"]), "depth": row["depth"]}

    def q2(self, moves: list, first: str) -> dict:
        v = self.cut.reads(list(moves), [first]).get(first)
        return {"verdict": v["verdict"] if v else None, "net": v["net"] if v else None}

    def read_game(self, moves: list, rec: dict) -> list:
        pos = rec["positions"]
        flags = FL.stable_points(rec)
        want = [i for i, q in enumerate(pos) if q["cls"] != 0]
        if not want:
            return []
        res = self.lab.reader.read([(None, moves)], [[pos[i]["ply"] for i in want]])
        self.names = {k: res[k] for k in ("kinds", "roles", "places")}
        names = self.names
        root = {r["ply"]: r for r in res["rows"]}
        items = {p: {"start": None, "moves": moves[:p], "raw": r, "names": names, "list": TL.threat_rows(r, names, None)} for p, r in root.items()}
        win_plies = [pos[i]["ply"] for i in want if pos[i]["cls"] == 1 and pos[i]["ply"] in items]
        lists = {}
        if win_plies:
            for p, g in zip(win_plies, FG.goals_of([items[p] for p in win_plies], self.lab.lines, self.lab.record)):
                lists[p] = g["list"]
        wires, wplies, tags, pvs = [], [], [], {}
        for i in want:
            q = pos[i]
            p = q["ply"]
            if p not in root:
                continue
            pv = self.lab.legal_prefix(moves[:p], q["pv"]) if q["pv"] else []
            pvs[p] = pv
            side = "x" if q["cls"] == 1 else "y"
            for j in STEPS[side]:
                if len(pv) >= j:
                    wires.append(moves[:p] + pv[:j])
                    wplies.append(p + j)
                    tags.append((p, j))
        fitems = self.lab._items(wires, wplies) if wires else []
        ok = [(t, it) for t, it in zip(tags, fitems) if it is not None]
        gl = FG.goals_of([it for _t, it in ok], self.lab.lines, self.lab.record, complete=True, ctx=self.ctx) if ok else []
        steps_of: dict = {}
        last_board: dict = {}
        for ((p, j), it), g in zip(ok, gl):
            raw = it["raw"]
            mv = raw["mover"]                                                        # the side to move after j plies is the victim of the side that has just moved
            s = FL.summary(g["other"])
            steps_of.setdefault(p, {})[j] = {"goal": s["goal"], "kind": s["kind"], "stake": s["stake"], "forced": s["forced"], "threats": s["threats"], "n_v": sum(1 for o, _k, _pl in raw["board"] if o == mv),
                                              "n_s": sum(1 for o, _k, _pl in raw["board"] if o != mv), "fell": g["other"].get("fell", 0), "kept_pairs": g["other"].get("kept_pairs", 0)}
            last_board[p] = raw["board"]
        out = []
        for i in want:
            q = pos[i]
            p = q["ply"]
            if p not in root:
                continue
            r0 = root[p]
            x_is = r0["mover"]
            side = "x" if q["cls"] == 1 else "y"
            owner = x_is if side == "x" else 1 - x_is
            victim = 1 - owner
            n_v0 = sum(1 for o, _k, _pl in r0["board"] if o == victim)
            st = steps_of.get(p, {})
            rec_out = {"ply": p, "cls": q["cls"], "win": flags[i]["win"], "lose": flags[i]["lose"], "stable_win": flags[i]["stable_win"], "stable_lose": flags[i]["stable_lose"], "owner": side, "kind": frame_kind(st, n_v0),
                       "net": material_net(r0["board"], last_board[p], owner, victim, self.worth_of) if p in last_board else 0, "steps": {str(j): s for j, s in st.items()}, "pv_n": len(pvs.get(p, [])), "units": q["units"]}
            if q["cls"] == 1 and pvs.get(p):
                first = pvs[p][0]
                rec_out["q1"] = self.q1(moves[:p], first)
                rec_out["q2"] = self.q2(moves[:p], first)
                rec_out["q3"] = FL.Labeller.sees(lists[p], first, names["places"])
            out.append(rec_out)
        return out


_FR = None


def _init_worker() -> None:
    global _FR
    _FR = FrameReader()


def _job(args: tuple) -> tuple:
    game, rec, root = args
    f = Path(root) / "frames" / f"g{rec['ord']}.json"
    if f.exists():
        return rec["ord"], 0.0
    t0 = time.perf_counter()
    try:
        out = _FR.read_game(game["moves"], rec)
    except Exception as e:                                                          # a game the reading cannot take is left out and counted: the file names the error
        (Path(root) / "frames" / f"g{rec['ord']}.error").write_text(f"{type(e).__name__}: {str(e)[:300]}")
        return rec["ord"], -1.0
    f.write_text(json.dumps({"ord": rec["ord"], "part": rec["part"], "positions": out}))
    return rec["ord"], time.perf_counter() - t0


def load(root: Path) -> dict:
    out = {}
    for x in (root / "frames.jsonl").read_text().splitlines():
        if x.strip():
            d = json.loads(x)
            for q in d["positions"]:
                out[(d["ord"], q["ply"])] = dict(q, part=d["part"])
    return out


def run(root: Path, labels: list, workers: int, say) -> None:
    (root / "frames").mkdir(exist_ok=True)
    games = {g["ord"]: g for g in FL.universe()}
    t0 = time.perf_counter()
    done = 0
    todo = [(games[r["ord"]], r, str(root)) for r in labels]
    say(f"FRAMES: {len(todo)} games; every position whose class for the side to move is winning or losing is read (the oracle's stored best line, no oracle call); {workers} workers; load average {os.getloadavg()[0]:.2f}")
    with multiprocessing.get_context("spawn").Pool(workers, initializer=_init_worker) as pool:
        for _o, _dt in pool.imap_unordered(_job, todo, chunksize=1):
            done += 1
            if done % 20 == 0 or done == len(todo):
                say(f"FRAMES progress: {done} of {len(todo)} games; {time.perf_counter() - t0:.0f} s; load average {os.getloadavg()[0]:.2f}")
    have = [r for r in labels if (root / "frames" / f"g{r['ord']}.json").exists()]
    if len(have) < len(labels):
        say(f"FRAMES games the reading could not take (left out, each with an error file): {len(labels) - len(have)}")
    (root / "frames.jsonl").write_text("\n".join((root / "frames" / f"g{r['ord']}.json").read_text() for r in have) + "\n")
    say(f"FRAMES done: {len(have)} games in {time.perf_counter() - t0:.0f} s wall")


def main(argv: "list | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tag", default="short")
    ap.add_argument("--per-part", type=int, default=0, help="a dry run on the first games of each part (0: every game of the labels)")
    ap.add_argument("--workers", type=int, default=9)
    a = ap.parse_args(argv)
    root = out_root(a.tag)
    src = Path(paths.experiment(*SRC))
    labels = [json.loads(x) for x in (src / "labels.jsonl").read_text().splitlines() if x.strip()]
    if a.per_part:
        labels = [r for part in ("find", "confirm") for r in [r for r in labels if r["part"] == part][:a.per_part]]
    run(root, labels, a.workers, FL.Say(root, "frames"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
