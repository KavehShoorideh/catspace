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


def chances_wire(path: "str | None") -> "list | None":
    """The chance table the Python sheet reads from `path` (`QS.load_chances`), as the Rust `sheet` takes it: [(key, [(horizon, (taken, total))])]."""
    if not path:
        return None
    if path not in _WIRE:
        _WIRE[path] = [(list(k), [(int(h), (int(v[0]), int(v[1]))) for h, v in c.items()]) for k, c in QS.load_chances(path).cells.items()]
    return _WIRE[path]


_WIRE: dict = {}


def sheet(sheet_model, env: "QS.Env", position: str, params: "QS.Params | None" = None, max_steps: "int | None" = None, budget_cost: "float | None" = None, only: "list | None" = None,
          finish: bool = True, moves: bool = False) -> dict:
    """STAGE C: the whole sheet of `position` in Rust: the schedule under `params.clock` (present, chance, plain), `params.cost_model` (None: the measured time of each step), `params.chances`,
    `params.risk`; with `finish` (the default) then `finish_levels(1)` and `settle_flagged`, which is what `QS.read_position` does. `only`: [(side, kind, place)] to ask about. ValueError when
    the position is over. See `SheetModel.sheet` for the result."""
    p = params or QS.Params()
    cm = list(p.cost_model) if p.cost_model else None
    return sheet_model.sheet(position, p.reach_k, p.max_open_exits, p.clock, max_steps, budget_cost, cm, chances_wire(p.chances), p.chance_horizon, p.risk, only, finish, moves)


def game(sheet_model, start: "str | None", moves: list, params: "QS.Params | None" = None, max_steps: "int | None" = None, budget_cost: "float | None" = None) -> list:
    """STAGE C, step three: the whole sheet of every position of a game (`start` None: the game's start; `moves` as the kernel names them) made in one call in the engine; a brief per ply
    (None for a position that is over), see `SheetModel.game`."""
    p = params or QS.Params()
    cm = list(p.cost_model) if p.cost_model else None
    return sheet_model.game(start, list(moves), p.reach_k, p.max_open_exits, p.clock, max_steps, budget_cost, cm, chances_wire(p.chances), p.chance_horizon, p.risk)


def games(sheet_model, games: list, threads: int, params: "QS.Params | None" = None, max_steps: "int | None" = None, budget_cost: "float | None" = None) -> list:
    """The many-game form: [(start, moves)] across `threads` std threads (the GIL released), one list of briefs per game, in the order of the games."""
    p = params or QS.Params()
    cm = list(p.cost_model) if p.cost_model else None
    return sheet_model.games([(s, list(m)) for s, m in games], threads, p.reach_k, p.max_open_exits, p.clock, max_steps, budget_cost, cm, chances_wire(p.chances), p.chance_horizon, p.risk)


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


def forcing(sheet_model, position: str, fuse: "int | None" = None, goal_nodes: "int | None" = None) -> dict:
    """PLAN 78.46 THE FORCING READ of the side to move, in Rust only (rust/engine/tactics/src/sheet/forcing.rs): every threat move, every reply to it, the claims that stand after each reply; a verdict per
    move (CONFIRMED | NOT SETTLED | NO CLAIM | UNREAD), its value, the nodes read, whether the fuse (positions read) stopped it. See `SheetModel.forcing`. ValueError when the position is over.
    THE PYTHON MIRROR OF THIS READ IS OWED (SYSTEM_SPEC section 8: validated in Rust first, then mirrored for readability); until it is written this call has no oracle."""
    return sheet_model.forcing(position, fuse, goal_nodes)


def forcing_check(sheet_model, position: str, move: str, depth: int, nodes: int) -> dict:
    """The ledger search (the sheet's worth table, no pruning) from the position after `move`, deepened to `depth` plies under `nodes`: {value, depth, cut, line, nodes, win} (what a confirmed claim
    is checked against)."""
    return sheet_model.forcing_check(position, move, depth, nodes)
