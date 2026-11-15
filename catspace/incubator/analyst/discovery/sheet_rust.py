"""PLAN 78.43 STAGE A: the Rust form of the question sheet's light reading and its levels 0 and 1 by geometry (rust/engine/tactics/src/sheet), called from Python.

The Python sheet (question_sheet.py) is the readable mirror and the ORACLE; this module only hands the Rust engine what is the game's and not the position's -- the medium of movement
(read once from the game's description by `Env.medium()`), the worth of each kind, the goal's kind, the number of move relations the vocabulary names -- and returns the questions of
a position with their level 0 and level 1 as plain data. It names no object of any game.

    model = sheet_rust.model(env, worth)
    out = sheet_rust.read(model, env, position)        # {mover, units, order, level0, level1}
"""
from __future__ import annotations

import cs_engine as CE

from catspace.incubator.analyst.discovery import question_sheet as QS


def model(env: "QS.Env", worth: dict, goal_kind: "str | None" = None):
    """The `cs_engine.SheetModel` of a game under a worth table. `goal_kind` is read from a light reading of the initial position when not given."""
    med = env.medium()
    if med is None:
        raise ValueError("the game's vocabulary names no medium of movement")
    if goal_kind is None:
        goal_kind = QS.Reading(env, CE.state(env.variant, None, [])[0], worth, light=True).goal_kind()
    hops = [(k, s, f, t, bool(take)) for (k, s, f), row in med.hop.items() for t, take in row.items()]
    overs = [(k, s, f, t, o) for (k, s, f, t), places in med.over_of.items() for o in places]
    worth_of = env.worth_fn(worth)
    by_kind = {kind: worth_of(kind) for kind in env.kind_keys_all()}
    ledger = {k: int(v) for k, v in worth.items()}                       # the table the tactics search is judged by (by the kernel's own key of a kind)
    return CE.SheetModel(env.variant, hops, overs, sorted(med.takes), by_kind, goal_kind, len(env.relations["can_go"]), ledger, dict(env.kind_keys_all()))


def read(sheet_model, env: "QS.Env", position: str, params: "QS.Params | None" = None) -> dict:
    """The questions of `position` in the first order with level 0 and level 1 (ValueError when the position is over)."""
    p = params or QS.Params()
    return sheet_model.read(position, p.reach_k, p.max_open_exits)


def move_sheet(sheet_model, env: "QS.Env", position: str, params: "QS.Params | None" = None, max_moves: "int | None" = None) -> dict:
    """STAGE C1: the question of each legal move of `position` in the order asked ({order, unread, results, seconds}; see `SheetModel.move_sheet`; ValueError when the position is over)."""
    p = params or QS.Params()
    return sheet_model.move_sheet(position, p.reach_k, p.max_open_exits, max_moves)


def move_question(sheet_model, env: "QS.Env", position: str, move: str, params: "QS.Params | None" = None) -> dict:
    """STAGE C1: the question of one move (see `SheetModel.move_question`)."""
    p = params or QS.Params()
    return sheet_model.move_question(position, move, p.reach_k, p.max_open_exits)


def _event(e: dict) -> tuple:
    return (e["mover"][0], e["mover"][1], e["mover"][2], e["dest"], None if e["taken"] is None else tuple(e["taken"]), e["taken_at"], e.get("arrives"))


def event_wire(events: list) -> list:
    """The events of a game (`codec.events`) in the form the Rust `unit_loss` takes."""
    return [_event(e) for e in events]


def unit_loss(sheet_model, wire_events: list, unit_id: int, start: int, horizon: int) -> "dict | None":
    """STAGE C1: `QS.unit_loss` in Rust, over `event_wire(events)`: None, or {status, balance, place, ply, run, arrival, charged (the ply a lost unit is charged to)}."""
    return sheet_model.unit_loss(wire_events, unit_id, start, horizon)


def settle(sheet_model, env: "QS.Env", position: str, steps: list, params: "QS.Params | None" = None) -> dict:
    """STAGE B: the deeper levels (level 2 and on) of the questions named by `steps`, a list of (side, kind, place, level) taken in that order, as plain data (see `SheetModel.settle`)."""
    p = params or QS.Params()
    return sheet_model.settle(position, steps, p.reach_k, p.max_open_exits)
