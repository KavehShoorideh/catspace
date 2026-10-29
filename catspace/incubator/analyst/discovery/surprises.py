"""PLAN 76.3 (d) THE SURPRISE SHELF: COUNTEREXAMPLES ARE PERMANENT (the surprise loop's first borrowing, docs/ANALYST_PLAN.md 76). Every disagreement the machinery
meets is filed as a CASE and read forever after -- by the guard (`scripts/trace_identity.py --sets surprises`: the decisions printed, never required identical: a
surprise is a known miss until fixed) and by the miner (a refuted measure's case joins its class's sample: a constraint every candidate must meet). Two directories,
one shelf:

  rules/surprises/                                   THE RULES' CASES (plan 76.0): the spec against a kernel (the cross-play, crossplay.py) or against the Game
                                                     surface (the oracle, gdl_chess.game_oracle) -- schema in gdl_chess.file_surprise
  catspace/incubator/analyst/discovery/lemmas/surprises/   THE ENGINE'S CASES (plan 76.3): a puzzle the loop got wrong ('loop miss'), a measure the play refuted
                                                     ('refuted measure') -- the schema below

AN ENGINE CASE: {id, class, fen, us (the side the engine played), found (where and when it was met), engine {answer, uci, read} (what the engine said and the trace's
reading of why), right {answer, uci, source, loss_cp} (the right answer and its SOURCE: the table, Stockfish in a separate process, or the rules), shrunk (the
smallest position that keeps the surprise, or null) and shrink (how, or why no shrinker applies: a puzzle miss is its position; a measure's class fixes its
material), status ('open' -- a known miss -- or 'fixed', with fixed_by), measure (a refuted measure only: {class, quantity, depth, n}), plan, journal, note}.

  python -m catspace.incubator.analyst.discovery.surprises report      # OBS lines: the shelf's cases by directory, class and status; problems
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
ENGINE_DIR = Path(__file__).resolve().parent / "lemmas" / "surprises"
RULES_DIR = ROOT / "rules" / "surprises"
CLASSES = ("loop miss", "refuted measure")
STATUSES = ("open", "fixed")
FIELDS = ("id", "class", "fen", "us", "found", "engine", "right", "shrunk", "shrink", "status", "fixed_by", "measure", "plan", "journal", "note")


def check_case(c: dict, where: str = "") -> list:
    """The schema of one engine case -> what is wrong (empty: a good case)."""
    import chess
    at = where or c.get("id", "?")
    if set(c) != set(FIELDS):
        return [f"{at}: fields {sorted(set(c) ^ set(FIELDS))} (the schema is exactly {', '.join(FIELDS)})"]
    bad = []
    if c["class"] not in CLASSES:
        bad.append(f"{at}: class {c['class']!r} not in {CLASSES}")
    if c["status"] not in STATUSES:
        bad.append(f"{at}: status {c['status']!r} not in {STATUSES}")
    elif (c["status"] == "fixed") != bool(c["fixed_by"]):
        bad.append(f"{at}: fixed_by is given if and only if the status is fixed")
    try:
        b = chess.Board(c["fen"])
        if not b.is_valid():
            bad.append(f"{at}: the position is not valid")
        if c["us"] != ("w" if b.turn else "b"):
            bad.append(f"{at}: us must be the side to move")
    except ValueError:
        bad.append(f"{at}: the fen does not parse")
    for k in ("engine", "right"):
        if not isinstance(c[k], dict) or not {"answer", "uci"} <= set(c[k]):
            bad.append(f"{at}: {k} must say the answer and its uci")
    if not isinstance(c["right"], dict) or not c["right"].get("source"):
        bad.append(f"{at}: the right answer must name its source (the table, Stockfish in a separate process, the rules)")
    if (c["class"] == "refuted measure") != isinstance(c["measure"], dict):
        bad.append(f"{at}: a refuted measure (and only it) says its measure")
    if not isinstance(c["shrink"], str) or not c["shrink"]:
        bad.append(f"{at}: shrink must say how the case was shrunk, or why no shrinker applies")
    return bad


def load_engine(d: Path = ENGINE_DIR) -> list:
    """Every engine case (with its file name), schema-checked: a bad case raises."""
    out, bad = [], []
    for p in sorted(Path(d).glob("*.json")):
        c = json.loads(p.read_text())
        bad += check_case(c, p.name)
        out.append(dict(c, file=p.name))
    if bad:
        raise ValueError("; ".join(bad))
    return out


def load_rules() -> list:
    """Every rules' case of rules/surprises/ (the cross-play's and the oracle's)."""
    return [dict(json.loads(p.read_text()), file=p.name) for p in sorted(RULES_DIR.glob("*.json"))]


def guard_cases() -> list:
    """THE GUARD SET of the shelf (`trace_identity.py --sets surprises`): (id, fen, us) of every case of both directories -- an engine case by its id and position (the
    shrunk one when it has one), a rules' case by its file's stem, the side to move playing."""
    import chess
    out = []
    for c in load_engine():
        fen = c["shrunk"] or c["fen"]
        out.append((c["id"], fen, c["us"]))
    for c in load_rules():
        b = chess.Board(c["fen"])
        if b.is_valid() and not b.is_game_over():
            out.append((c["file"].split("_")[0], c["fen"], "w" if b.turn else "b"))
    return out


def refuted_for(sig: str) -> list:
    """The REFUTED MEASURES' cases of a material class (the miner's constraints): every engine case of class 'refuted measure' whose measure is of the class `sig`."""
    return [c for c in load_engine() if c["class"] == "refuted measure" and c["measure"]["class"] == sig]


def report(log=print) -> list:
    eng, rul = load_engine(), load_rules()
    lines = [f"OBS surprise shelf: cases {len(eng) + len(rul)} -- the engine's {len(eng)} (lemmas/surprises/), the rules' {len(rul)} (rules/surprises/)",
             "OBS surprise shelf engine by class: " + ", ".join(f"{k} {v}" for k, v in sorted(Counter(c["class"] for c in eng).items())),
             "OBS surprise shelf engine by status: " + ", ".join(f"{k} {v}" for k, v in sorted(Counter(c["status"] for c in eng).items())),
             "OBS surprise shelf rules by kernel and class: " + ", ".join(f"{k} {v}" for k, v in sorted(Counter(f"{c['kernel']} {c['class']}" for c in rul).items()))]
    for c in eng:
        lines.append(f"OBS surprise {c['id']} ({c['class']}, {c['status']}): {c['fen']} -- the engine {c['engine']['answer']}, right {c['right']['answer']} "
                     f"({c['right']['source'].split(' (')[0]})")
    for x in lines:
        log(x)
    return lines


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] == "report":
        report()
    else:
        sys.exit("usage: python -m catspace.incubator.analyst.discovery.surprises report")
