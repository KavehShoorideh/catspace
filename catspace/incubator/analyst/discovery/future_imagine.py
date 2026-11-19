"""PLAN 78.94 THE EXPLICIT RUN (SYSTEM_SPEC 7 "the future imagination layer": layer one of reachable, a run of own moves with the other side standing still; layer two, the veto; the budget is questions; 8 the threat list). Game-free: a move is a word, a place a number,
both read from the engine at run time; the pass is the game's own (CE.passed: a state where passing is refused gives no goals). No oracle, no worth assumed: a stake is the line reading's net in the measured record's own integers.

INTERFACE (written before the bodies; the plan's: imagine, veto).
  Imaginer(variant=None, record=None)     per process: the threat reader, the line reader and the record.
  .imagine(moves, side, budget, fuse=FRAME_FUSE) -> {"goals": [Goal], "questions": int, "wall": seconds, "refused": bool}
        `moves` the game so far (from the start), `side` "mover" (the side to move; it moves first) or "other" (the other side; the side to move passes first). A run of the side's own moves with the other side standing still, searched best-first
        (the quiet first moves of threats in two moves and the takings of the confirmed threats of the node first, then every other legal move; shallower first within a class), each node costing ONE QUESTION: "does the concrete threat list of this position hold two
        threats of the side that no one reply meets?" (future_goals.goals_of_list). Past `budget` questions the search goes on only through nodes of the first class (a frame it is reading that still makes progress: a threat made or taken), at most `fuse` further questions,
        and stops when none is left. A Goal is future_goals' dict plus {"run": the own moves as words, "first": the first move as a word, "questions": the questions asked when it was found}.
  .veto(moves, side, goal) -> {"holds": bool, "broken_by": a move word or None, "questions": int, "line": [words]}
        layer two: along the goal's run the OTHER side moves in the gaps, each time by the engine's own search (CE.best_move at VETO_SECONDS), the run's own moves played as they stand; at the end, with the other side's reply made, does the side still hold a goal
        (future_goals.goals_of_list on the real position)? `broken_by` is the reply after which an own move of the run is illegal or no goal holds at the end. One line, the search's: a lower bound on what a veto can do.
  Called by: future_measure.py (stage explicit), the match player of stage five.
"""
from __future__ import annotations

import heapq
import time

import cs_engine as CE

from catspace.incubator.analyst.discovery import future_goals as FG
from catspace.incubator.analyst.discovery import human_heads_data as HD
from catspace.incubator.analyst.discovery import threat_concrete as TC
from catspace.incubator.analyst.discovery import threat_list as TL

FRAME_FUSE = 8                                                # past the budget: at most this many further questions, inside a frame that still makes progress (a named fuse of this build; the spec leaves N and the frame's end open)
VETO_SECONDS = 0.2                                            # the clock of the other side's reply in the veto (the play clock of the matches)


def word_of(pair, places: list) -> "str | None":
    if pair is None or len(pair) < 2 or not isinstance(pair[0], int):
        return None
    return f"{places[pair[0]]}{places[pair[1]]}"


class Imaginer:
    """See the module doc."""

    def __init__(self, variant: "str | None" = None, record: "TC.Record | None" = None):
        self.variant = variant or HD.VARIANT
        self.reader = TL.ThreatReader(1)
        self.record = record or TC.Record.load()
        self.lines = TC.Lines(self.record, threads=1)

    # one question: the concrete list of one position, as the side to move's
    def question(self, text: str) -> "tuple[dict, dict] | None":
        res = self.reader.read([(text, [])], [[0]])
        if not res["rows"]:
            return None
        names = {k: res[k] for k in ("kinds", "roles", "places")}
        r = res["rows"][0]
        item = {"start": text, "moves": [], "raw": r, "names": names, "list": TL.threat_rows(r, names, None)}
        c = TC.concrete_lists([item], self.lines, self.record)[0]
        return c, names

    def root_text(self, moves: list, side: str) -> "str | None":
        text = CE.state(self.variant, None, list(moves))[0]
        if side == "other":
            p = CE.passed(self.variant, text, [])
            return None if p is None else p[0]
        return text

    def children(self, text: str, c: dict, names: dict) -> list:
        """The legal moves of the side to move at `text`, ordered: the quiet first moves of its threats in two moves, then the first moves of its confirmed threats, then the rest in the engine's order; (class, move word)."""
        legal = list(CE.state(self.variant, text, [])[2])
        places = names["places"]
        lead, take = [], []
        for r in c["other"]:
            if r.get("goal"):
                continue
            ld = word_of(r.get("lead"), places) if r.get("two_moves") else None
            tk = word_of(r.get("first"), places) if r["verdict"] == "CONFIRMED" and not r.get("two_moves") else None
            for w, bucket in ((ld, lead), (tk, take)):
                if w:
                    hit = [m for m in legal if m.startswith(w)]
                    bucket += [m for m in hit if m not in bucket]
        first = [m for m in lead + take if m in legal]
        seen = set(first)
        return [(0, m) for m in first] + [(1, m) for m in legal if m not in seen]

    def imagine(self, moves: list, side: str, budget: int, fuse: int = FRAME_FUSE) -> dict:
        t0 = time.perf_counter()
        text = self.root_text(moves, side)
        out = {"goals": [], "questions": 0, "wall": 0.0, "refused": text is None}
        if text is None:
            return out
        q = self.question(text)
        out["questions"] = 1
        if q is None:
            return out
        c, names = q
        info = FG.goals_of_list(c, "mover", self.record)
        for g in info["goals"][:1]:
            out["goals"].append(dict(g, run=[], first=None, questions=1))
        heap: list = []
        tie = 0
        for rank, (cls, m) in enumerate(self.children(text, c, names)):
            heapq.heappush(heap, (cls, 1, rank, tie, text, [m]))
            tie += 1
        spent_frame = 0
        while heap:
            cls, depth, rank, _t, base, run = heapq.heappop(heap)
            over = out["questions"] >= budget
            if over and (cls != 0 or spent_frame >= fuse):
                continue
            if over:
                spent_frame += 1
            t1 = CE.state(self.variant, base, [run[-1]])[0]
            ps = CE.passed(self.variant, t1, [])
            if ps is None:
                continue
            res = self.question(ps[0])
            out["questions"] += 1
            if res is None:
                continue
            c_next, nm2 = res
            info = FG.goals_of_list(c_next, "mover", self.record)
            if info["goals"]:
                g = info["goals"][0]
                out["goals"].append(dict(g, run=list(run), first=run[0], questions=out["questions"]))
                continue
            if c_next["other"] and any(r["verdict"] == "CONFIRMED" or r.get("two_moves") for r in c_next["other"] if not r.get("goal")) or depth < 2:
                for rk, (cl2, m) in enumerate(self.children(ps[0], c_next, nm2)):
                    heapq.heappush(heap, (cl2, depth + 1, rk, tie, ps[0], run + [m]))
                    tie += 1
        out["goals"].sort(key=lambda g: (-g["stake"], len(g["run"])))
        out["wall"] = time.perf_counter() - t0
        return out

    def veto(self, moves: list, side: str, goal: dict) -> dict:
        """See the module doc."""
        text = CE.state(self.variant, None, list(moves))[0]
        worth = {k: int(v) for k, v in self.record.worth.items()}
        line, asked = [], 0
        if side == "other":                                      # the side to move acts first in the real line: its reply is the search's
            mv = CE.best_move(self.variant, text, [], VETO_SECONDS, worth)[0]
            if mv is None:
                return {"holds": False, "broken_by": None, "questions": 0, "line": []}
            line.append(mv)
            text = CE.state(self.variant, text, [mv])[0]
        for m in goal["run"]:
            legal = CE.state(self.variant, text, [])[2]
            if m not in legal:
                return {"holds": False, "broken_by": line[-1] if line else None, "questions": asked, "line": line}
            text = CE.state(self.variant, text, [m])[0]
            line.append(m)
            reply = CE.best_move(self.variant, text, [], VETO_SECONDS, worth)[0]
            if reply is None:
                return {"holds": True, "broken_by": None, "questions": asked, "line": line}
            text = CE.state(self.variant, text, [reply])[0]
            line.append(reply)
        res = self.question(text)
        asked += 1
        if res is None:
            return {"holds": False, "broken_by": line[-1] if line else None, "questions": asked, "line": line}
        info = FG.goals_of_list(res[0], "mover", self.record)
        return {"holds": bool(info["goals"]), "broken_by": None if info["goals"] else (line[-1] if line else None), "questions": asked, "line": line}
