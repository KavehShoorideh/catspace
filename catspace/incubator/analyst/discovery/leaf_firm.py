"""PLAN 79.01 PART ONE, THE FIRM ROOM (SYSTEM_SPEC 4 the equal reading, 9 "the oracle labels positions and frames in sleep"; piece: the sleep's labels, a measure). PURE: this module makes no engine call and no oracle call; it reads the labels of plan 79.00 (stack_room's equals.jsonl) and the second asking
of leaf_ask.py (ask.jsonl), and decides which labels are firm. Game-free: a move is a word, a class is -1, 0 or 1.

WHY. The room of 79.00 (positions whose equal moves differ in keeping the oracle's class) rests on a class that is the oracle's score cut at WIN_CP, and a score near the cut flips when the oracle is asked again. A label that is firm is one that does not.

THE FIRM LABEL (decided here, before any number of the new run was looked at).
  DEPTH INSIDE A CLASS: how far the oracle's number lies inside the class it was cut into: class 1, the number less WIN_CP; class -1, minus the number less WIN_CP; class 0, the smaller of the distances to the two cuts; a forced end (no number) is infinitely far. Computed for the first asking (79.00's stored number) and for the second.
  A move FIRMLY KEEPS when both askings give the same class, that class is not lower than the class before the move, and both depths inside it are at least the MARGIN; it FIRMLY LOSES likewise with the class lower than the class before it. Any other move is NEITHER.
  THE MARGIN is chosen on FIND ONLY (`choose_margin`), by a rule fixed here: of the grid GRID, the smallest value at which, among the equal moves of FIND's room positions whose FIRST number is at least that deep inside its class, the second asking gives the same class in at least AGREE of them (and at least
  MIN_MOVES such moves exist; if no grid value does, the largest). The choice is written to margin.json BEFORE CONFIRM is read: every function that reads CONFIRM takes the margin from that file (`load_margin`) and refuses to run without it.
  FIND AND CONFIRM: a SPLIT BY GAME, `part_of(ord)` (a hash of the game's ordinal; 79.00's sample is CONFIRM positions of 78.94's labels and has no split of its own, so this one is made here).
  THE FIRM ROOM: the positions of the room (79.00's room: some equal move keeps the class and some does not, by the first asking) with at least one firmly keeping and one firmly losing equal move.

INTERFACE (written before the bodies).
  part_of(ord) -> "find" | "confirm"                          the split by game (pure; tested).
  depth_inside(cls, cp) -> float                              see above (pure; tested).
  firm_class(cls_a, cp_a, cls_b, cp_b, margin) -> int | None          the class both askings agree on and are inside by the margin, else None (pure; tested).
  firm_vec(rec, ask, margin) -> [ "keep" | "lose" | None ]    per equal move of the label record (pure; tested).
  in_room(rec) / is_firm(vec)                                  (pure; tested).
  choose_margin(pairs, grid, agree, min_moves) -> (margin, [(m, n, agreeing)])    pairs = [(first depth, class asked first, class asked again)] over FIND's room moves (pure; tested).
  save_margin / load_margin(tag)                              margin.json beside the labels.
  weights(values) -> [float] | None                            the uniform weights over the moves with the highest value (ties within EPS), None when a value is missing (pure; tested).
  rule_scores(vec, first_i, pick_i, v_end, v_one, v_sym) -> {rule: (keep, lose)}   the share of weight on firmly keeping and on firmly losing moves for the rules first, pick, random, end, one, and end_sym (the end-of-line chance with the side to move's own bonus taken out: leaf_line.Chance.sym; a reading added after the judged positions showed that the highest chance often differed only by whose turn it is at the end of the line) (pure; tested).
  Called by: leaf_ask.py (which moves to ask again), leaf_line.py (the firm room's positions), leaf_measure.py (the tables), ladder_print --leaf01.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from catspace.incubator.analyst.discovery import future_labels as FL
from catspace.incubator.analyst.discovery import prior_equal as PE

OUT = ("leaf_79_01",)
GRID = (0, 25, 50, 75, 100, 150)
AGREE = 0.99                                                    # fixed after the short run showed 96 pct agreement over all moves: a firm label should repeat in 99 of 100 askings
MIN_MOVES = 200
EPS = 1e-9
SPLIT_SEED = "leaf7901"
BANDS = ("few", "mid", "many")
RULES = ("first", "pick", "random", "end", "one", "end_sym")


def out_root(tag: str) -> Path:
    from catspace.io import paths
    p = Path(paths.experiment(*OUT, tag))
    p.mkdir(parents=True, exist_ok=True)
    return p


def part_of(ordinal: int) -> str:
    return "find" if int(hashlib.md5(f"{SPLIT_SEED}:{ordinal}".encode()).hexdigest(), 16) % 2 == 0 else "confirm"


def depth_inside(cls: int, cp: "float | None") -> float:
    w = FL.WIN_CP
    if cp is None:
        return float("inf")
    if cls == 1:
        return cp - w
    if cls == -1:
        return -w - cp
    return min(w - cp, cp + w)


def firm_class(cls_a: int, cp_a: "float | None", cls_b: int, cp_b: "float | None", margin: float) -> "int | None":
    if cls_a != cls_b:
        return None
    if depth_inside(cls_a, cp_a) < margin or depth_inside(cls_b, cp_b) < margin:
        return None
    return cls_a


def firm_vec(rec: dict, ask: dict, margin: float) -> list:
    out = []
    for i, m in enumerate(rec["tied"]):
        fc = firm_class(rec["cm"][i], rec["cp"][i], ask["cls2"][i], ask["cp2"][i], margin)
        out.append(None if fc is None else ("keep" if PE.keeps(rec["cls"], fc) else "lose"))
    return out


def in_room(rec: dict) -> bool:
    if rec.get("depth") is None or len(rec["tied"]) < 2:
        return False
    k = [PE.keeps(rec["cls"], c) for c in rec["cm"]]
    return any(k) and not all(k)


def is_firm(vec: list) -> bool:
    return "keep" in vec and "lose" in vec


def choose_margin(pairs: list, grid: tuple = GRID, agree: float = AGREE, min_moves: int = MIN_MOVES) -> tuple:
    table = []
    for m in grid:
        sel = [(d, a, b) for d, a, b in pairs if d >= m]
        table.append((m, len(sel), sum(1 for _d, a, b in sel if a == b)))
    for m, n, k in table:
        if n >= min_moves and k >= agree * n:
            return m, table
    return grid[-1], table


def save_margin(tag: str, margin: float, table: list, find_rows: int) -> None:
    (out_root(tag) / "margin.json").write_text(json.dumps({"margin": margin, "table": table, "grid": list(GRID), "agree": AGREE, "min_moves": MIN_MOVES, "chosen_on": "find", "find_room_positions": find_rows}))


def load_margin(tag: str) -> float:
    p = out_root(tag) / "margin.json"
    if not p.exists():
        raise RuntimeError("margin.json is missing: the margin is chosen on FIND first (leaf_measure margin) and written before CONFIRM is read")
    return float(json.loads(p.read_text())["margin"])


def weights(values: list) -> "list | None":
    if any(v is None for v in values):
        return None
    top = max(values)
    on = [abs(v - top) <= EPS for v in values]
    n = sum(on)
    return [1.0 / n if o else 0.0 for o in on]


def rule_scores(vec: list, first_i: int, pick_i: int, v_end: list, v_one: list, v_sym: "list | None" = None) -> dict:
    n = len(vec)

    def of(w: "list | None") -> "tuple | None":
        if w is None:
            return None
        return (sum(x for x, v in zip(w, vec) if v == "keep"), sum(x for x, v in zip(w, vec) if v == "lose"))
    one_hot = lambda i: [1.0 if j == i else 0.0 for j in range(n)]
    return {"first": of(one_hot(first_i)), "pick": of(one_hot(pick_i)), "random": of([1.0 / n] * n), "end": of(weights(v_end)), "one": of(weights(v_one)), "end_sym": of(weights(v_sym)) if v_sym is not None else None}
