"""causes.py -- WHY an error was an error (Kaveh 2026-09-04): did the player IGNORE a standing
threat, WALK INTO one (the move itself created the vulnerability the refutation exploits), or
MISS something better (the move was safe; the omission cost)? And what KIND of threat: simple
(a plain attack on a hanging or bigger piece) or compound (pin, skewer, fork, discovered
attack, overload, mate). Computed from the played move's own effects (L4) and the refutation's
mechanism -- engine-free, so it can be recomputed offline from stored records."""
from __future__ import annotations

import chess

from .dynamics import move_effects
from .relations import against

FAMILY = {"pin": ("pin", "disabled_attack", "disabled_defense", "pinned_and_attacked", "hanging_through_pin"),
          "fork": ("fork",), "discovered": ("discovered_attack", "battery"), "skewer": ("skewer",),
          "hanging": ("hanging", "underdefended"), "attacked_by_lower": ("attacked_by_lower_value",), "overload": ("overloaded",)}
SIMPLE = {"hanging", "attacked_by_lower", "capture"}
COMPOUND = {"pin", "fork", "discovered", "skewer", "overload", "mate", "deflection", "attraction", "interference", "clearance", "removal_of_guard", "double_attack", "xray_defense"}


def families(names) -> list[str]:
    return sorted(f for f, ns in FAMILY.items() if any(n in ns for n in names))


def victims(r, victim: chess.Color) -> set:
    """Squares of VICTIM's pieces inside the relation (piece symbols carry the colour)."""
    return {sq for sq, sym in zip(r.squares, r.pieces or ()) if sym.isalpha() and sym.isupper() == (victim == chess.WHITE)}


def walked_into(board: chess.Board, played: chess.Move) -> list:
    """Relations the played move CREATED against its own side (liabilities it walked into)."""
    us = board.turn
    e = move_effects(board, played)
    return [r for r in e.created if against(r, us)]


def exploited(walked: list, e0, victim: chess.Color) -> list[str]:
    """Families of walked-into relations the refutation's first ply actually exploits: the refuting move
    creates/uses/resolves a liability against the same victim piece, or captures it. Name coincidences do
    not count (2026-09-05: Bf4 created a skewer AGAINST the opponent, Bc5+ created another 'skewer' -- the
    real story was the free pawn Bxb6 missed)."""
    if e0 is None:
        return []
    hits = []
    ref = [r for r in (e0.created + e0.uses + e0.resolved) if against(r, victim)]
    to_sq = chess.Move.from_uci(e0.uci).to_square if e0.uci else None
    for r in walked:
        v = victims(r, victim)
        if e0.captured:                        # a capturing reply exploits exactly the liability it captures
            if to_sq in v:
                hits.append(r)
        elif any(v & victims(r2, victim) for r2 in ref):
            hits.append(r)
    return hits


TEETH_CP = 40      # a refutation with less than this over the opponent's next-best reply is not a tactic


def classify(board: chess.Board, played: chess.Move, mechanisms: list[str], ignored_threat: bool,
             available_before: bool | None = None, teeth: int | None = None, even_trade: bool = False,
             refutation_effect=None) -> dict:
    """available_before: was the refutation already playable AND strong had we passed instead of moving
    (engine-checked by the caller)? True -> we ignored it; False -> our move enabled it.
    teeth: how much better the refutation is than the opponent's second-best reply (cp). When the
    reply has no teeth, the loss is the OPPORTUNITY we missed, not a tactic we suffered (Kaveh 2026-09-05:
    Bd2 'ignored' a toothless pin while the real story was the free pawn on c7)."""
    walked = walked_into(board, played)
    w = families([r.name for r in walked])
    hit_rels = exploited(walked, refutation_effect, board.turn) if refutation_effect is not None else []
    hit = families([r.name for r in hit_rels]) if refutation_effect is not None else sorted(set(w) & set(mechanisms))
    if even_trade and "mate" not in mechanisms:    # the trade IS the story; incidental geometry is not (teeth cannot judge this:
        cause = "positional"                       # a recapture always dwarfs the alternative of not recapturing). Kaveh 2026-09-05:
                                                   # a knight-for-bishop trade is materially equal at this layer; the eval change is
                                                   # about plans and belongs one layer up -- say so, don't invent a tactic
    elif teeth is not None and teeth < TEETH_CP and not (set(mechanisms) & {"mate"}):
        cause = "missed_better"
    elif hit:
        cause = "walked_into"                  # the move created the static vulnerability that was exploited
    elif (ignored_threat or available_before) and mechanisms:
        cause = "ignored_threat"               # it was already there, and the reply does something nameable
    elif "mate" in mechanisms and not (set(mechanisms) - {"mate", "capture"}):
        cause = "missed_better"
    elif mechanisms:
        cause = "enabled_tactic"               # the opponent's tactic only became available because of the move
    else:
        cause = "missed_better"
    kinds = set(mechanisms)
    kind = "positional" if cause == "positional" else "compound" if kinds & COMPOUND else "simple" if kinds & SIMPLE else "positional"
    return {"cause": cause, "kind": kind, "walked_into": w, "walked_into_hit": hit,
            "walked_into_rel": {"name": hit_rels[0].name, "squares": list(hit_rels[0].squares), "pieces": list(hit_rels[0].pieces or ()),
                                "owner": hit_rels[0].owner, "tags": list(hit_rels[0].tags or ())} if hit_rels else None}
