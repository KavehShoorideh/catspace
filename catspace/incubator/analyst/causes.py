"""causes.py -- WHY an error was an error (Kaveh 2026-09-04): did the player IGNORE a standing
threat, WALK INTO one (the move itself created the vulnerability the refutation exploits), or
MISS something better (the move was safe; the omission cost)? And what KIND of threat: simple
(a plain attack on a hanging or bigger piece) or compound (pin, skewer, fork, discovered
attack, overload, mate). Computed from the played move's own effects (L4) and the refutation's
mechanism -- engine-free, so it can be recomputed offline from stored records."""
from __future__ import annotations

import chess

from .dynamics import move_effects

FAMILY = {"pin": ("pin", "disabled_attack", "disabled_defense", "pinned_and_attacked", "hanging_through_pin"),
          "fork": ("fork",), "discovered": ("discovered_attack", "battery"), "skewer": ("skewer",),
          "hanging": ("hanging",), "attacked_by_lower": ("attacked_by_lower_value",), "overload": ("overloaded",)}
VICTIM_SIDE = ("hanging", "loose", "pinned_and_attacked", "hanging_through_pin", "disabled_attack", "disabled_defense")
SIMPLE = {"hanging", "attacked_by_lower", "capture"}
COMPOUND = {"pin", "fork", "discovered", "skewer", "overload", "mate"}


def families(names) -> list[str]:
    return sorted(f for f, ns in FAMILY.items() if any(n in ns for n in names))


def walked_into(board: chess.Board, played: chess.Move) -> list[str]:
    """Families of vulnerabilities the played move CREATED against its own side."""
    us = board.turn
    e = move_effects(board, played)
    names = []
    for r in e.created:
        ours = (r.owner == us) if r.name in VICTIM_SIDE else (r.owner != us)   # victim-side relations are owned by the victim
        if ours:
            names.append(r.name)
    return families(names)


def classify(board: chess.Board, played: chess.Move, mechanisms: list[str], ignored_threat: bool,
             available_before: bool | None = None) -> dict:
    """available_before: was the refutation already playable AND strong had we passed instead of moving
    (engine-checked by the caller)? True -> we ignored it; False -> our move enabled it."""
    w = walked_into(board, played)
    hit = sorted(set(w) & set(mechanisms))
    if hit:
        cause = "walked_into"                  # the move created the static vulnerability that was exploited
    elif ignored_threat or available_before:
        cause = "ignored_threat"               # it was already there
    elif "mate" in mechanisms and not (set(mechanisms) - {"mate", "capture"}):
        cause = "missed_better"
    elif mechanisms:
        cause = "enabled_tactic"               # the opponent's tactic only became available because of the move
    else:
        cause = "missed_better"
    kinds = set(mechanisms)
    kind = "compound" if kinds & COMPOUND else "simple" if kinds & SIMPLE else "positional"
    return {"cause": cause, "kind": kind, "walked_into": w, "walked_into_hit": hit}
