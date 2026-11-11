"""PLAN 78.32 THE EXAM. October's way of knowing whether anything improved: the SAME positions re-run after every change, the changed decisions listed. An exam is a fixed set of NON-TRIVIAL positions
(some legal move keeps the exact result class and some loses it) of a game and a size, picked once from the exact corpus (corpus.py), stratified by result class and by group, and taken from the corpus's
HELD-OUT groups only, so no development position is in it; `corpus.excluding(exam_keys(exam))` then takes them out of the corpus the induction works on, so induction and admission never see them. The file
lives in the training data area (paths.derived); the wake never reads it. NOTHING admits a rule by the exam: it is only graded.

  build_exam(game, size, *, n=300, seed, corpus) -> exam dict (written to `path` when given)        load_exam(path) / exam_keys(exam)
  grade(exam, *, variant, store, spec, seconds, wake_kw, labeller) -> {kept, total, by_class, decisions}
        the wake's own decision on every position (playout_sleep.check: wake.py's solve_batch call, same clock and switches, with the snapshot `store`), then the exact labeller says whether the chosen
        move keeps the result class
  compare(prev, cur) -> {gained, lost, changed, same}          say_exam(label, g, cmp) -> prints `OBS exam <label>: kept N of M (...), gained N, lost N`
  grade_twice(...) -> prints `OBS exam spread: A vs B, decisions differing N` (the clock's noise: the same snapshot graded twice)
"""
from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path

from catspace.incubator.analyst.discovery import playout_sleep as PS
from catspace.io import paths

FORMAT = "exam-1"


def default_path(game: str, size: int, n: int, seed: int) -> str:
    return paths.derived(f"exam_{game}_{size}_{n}_{seed}.json")


def build_exam(game: str, size: int, *, n: int = 300, seed: int = 1, corpus, path: "str | None" = None, say=print) -> dict:
    """`n` non-trivial positions from the held-out part of `corpus`, half and half by result class where both exist (the rest from whichever class has more), spread over the groups round-robin.
    -> {format, game, variant, size, seed, positions: [{start, moves, group, before}]}, written to `path` when given."""
    rng = random.Random(seed)
    pool: dict = {}
    for i in corpus.part("held"):
        if corpus.non_trivial(i):
            r = corpus.rows[i]
            pool.setdefault(r["before"], {}).setdefault(r["group"], []).append(i)
    for by_group in pool.values():
        for lst in by_group.values():
            rng.shuffle(lst)

    def take(cls: str, quota: int) -> list:
        by = pool.get(cls, {})
        out, queues = [], [by[g] for g in sorted(by)]
        while queues and len(out) < quota:
            nxt = []
            for q in queues:
                if len(out) >= quota:
                    break
                out.append(q.pop())
                if q:
                    nxt.append(q)
            queues = nxt
        return out
    classes = sorted(pool)
    chosen, share = [], {c: n // max(1, len(classes)) for c in classes}
    for c in classes:
        chosen += take(c, share[c])
    for c in sorted(classes, key=lambda c: -sum(len(v) for v in pool[c].values())):          # (a short class leaves its quota to the others)
        if len(chosen) >= n:
            break
        chosen += take(c, n - len(chosen))
    chosen = chosen[:n]
    chosen.sort(key=lambda i: (corpus.rows[i]["group"], corpus.rows[i]["moves"]))
    positions = [{"start": corpus.rows[i]["start"], "moves": list(corpus.rows[i]["moves"]), "group": corpus.rows[i]["group"], "before": corpus.rows[i]["before"]} for i in chosen]
    exam = {"format": FORMAT, "game": game, "variant": corpus.variant, "size": size, "seed": seed, "positions": positions}
    by_class = {c: sum(1 for p in positions if p["before"] == c) for c in sorted({p["before"] for p in positions})}
    say(f"OBS exam built: {len(positions)} positions, by result class {by_class}, groups {len({p['group'] for p in positions})}, from the held-out part of a corpus of {len(corpus)}")
    if path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(json.dumps(exam))
        os.replace(tmp, p)
    return exam


def load_exam(path) -> dict:
    d = json.loads(Path(path).read_text())
    assert d.get("format") == FORMAT, f"{path}: not an exam file"
    return d


def exam_keys(exam: dict) -> set:
    return {(p["start"], tuple(p["moves"])) for p in exam["positions"]}


def ensure_exam(game: str, size: int, *, n: int = 300, seed: int = 1, corpus, path: "str | None" = None, say=print) -> dict:
    """The exam file at `path` (default: default_path) when it exists, else built from `corpus` and written there."""
    path = path or default_path(game, size, n, seed)
    if Path(path).exists():
        e = load_exam(path)
        say(f"OBS exam built: {len(e['positions'])} positions read from {path}")
        return e
    return build_exam(game, size, n=n, seed=seed, corpus=corpus, path=path, say=say)


def grade(exam: dict, *, variant: "str | None" = None, store: "str | None", spec, seconds: float, wake_kw: "dict | None" = None, labeller, threads: int = 0) -> dict:
    """The wake's own decision on every exam position with the snapshot `store`, and whether the chosen move keeps the position's exact class (`labeller.moves_keep`).
    -> {kept, total, by_class {class: {kept, total}}, decisions: [{i, move, keeps, before}]} (a decision that failed keeps nothing)."""
    variant = variant or exam["variant"]
    pos = exam["positions"]
    res = PS.check([{"start": p["start"], "moves": p["moves"]} for p in pos], variant=variant, store=store, spec=spec, judge=labeller, seconds=seconds, wake_kw=wake_kw, threads=threads)
    decisions, by_class = [], {}
    for i, (p, c) in enumerate(zip(pos, res)):
        keep = labeller.moves_keep(variant, p["start"], p["moves"])
        keeps = bool(c["move"] is not None and keep is not None and keep.get(c["move"], False))
        decisions.append({"i": i, "move": c["move"], "keeps": keeps, "before": p["before"]})
        b = by_class.setdefault(p["before"], {"kept": 0, "total": 0})
        b["total"] += 1
        b["kept"] += keeps
    return {"kept": sum(d["keeps"] for d in decisions), "total": len(decisions), "by_class": by_class, "decisions": decisions}


def compare(prev: "dict | None", cur: dict) -> dict:
    """The decisions that changed between two gradings of the same exam: gained (keeps now, did not), lost (kept, does not now), changed (a different move), same."""
    if prev is None:
        return {"gained": [], "lost": [], "changed": [], "same": 0}
    gained, lost, changed, same = [], [], [], 0
    for a, b in zip(prev["decisions"], cur["decisions"]):
        if a["move"] != b["move"]:
            changed.append(b["i"])
        else:
            same += 1
        if b["keeps"] and not a["keeps"]:
            gained.append(b["i"])
        elif a["keeps"] and not b["keeps"]:
            lost.append(b["i"])
    return {"gained": gained, "lost": lost, "changed": changed, "same": same}


def say_exam(label: str, g: dict, cmp: "dict | None" = None, say=print) -> None:
    bc = g["by_class"]
    part = lambda c: f"{bc.get(c, {}).get('kept', 0)} of {bc.get(c, {}).get('total', 0)}"  # noqa: E731
    cmp = cmp or {"gained": [], "lost": []}
    say(f"OBS exam {label}: kept {g['kept']} of {g['total']} (win-keeping {part('win')}, draw-keeping {part('draw')}), gained {len(cmp['gained'])}, lost {len(cmp['lost'])}")


def grade_twice(exam: dict, *, say=print, **kw) -> tuple:
    """The same snapshot graded twice: the decision depends on a clock, so the two can differ. Prints `OBS exam spread: A vs B, decisions differing N` and returns (first grading, second, how many
    decisions (moves) differ)."""
    a, b = grade(exam, **kw), grade(exam, **kw)
    diff = sum(1 for x, y in zip(a["decisions"], b["decisions"]) if x["move"] != y["move"])
    say(f"OBS exam spread: {a['kept']} vs {b['kept']}, decisions differing {diff}")
    return a, b, diff
