"""PLAN 78.42 USE: THE SHEET PLAYER (training-side experiment; names no object of any one game: the game is reached only through the injected env, codec and kernel).

A player that wraps the wake's plain search. The search is asked for its move (the same call, the same settings as the wake: `cs_engine.solve_batch`, the game's HEAD snapshot, no
prior, no reasoning chain). The position is read ONCE by the question sheet. Then:

  STANDING   which of my weighty units are lost where they stand (`exchange_stands`, the other side to move through the kernel's pass).
  THE MOVE   `move_question` on the search's first choice. ACCEPT it when the verdict is SAFE and no unit that stood lost before is still lost after it with a balance worse than what
             the move took (the exchange frame on the position after the move, the reading `move_question` made reused). Otherwise the search's next choice, up to `max_tries`, and
             while the time spent on the tries after the first is under the budget. None accepted: the search's first choice is played. The search's choice is never overridden when it
             reports a forced win or when the side to move has one legal move.

THE ORDER OF THE NEXT CHOICES. `cs_engine.solve_batch` exposes only the best move of a position, and no argument leaves a root move out. So the search's second and later choices are
asked for by searching the position after each other legal move (the other side to move, the same search at `rank_seconds` each, all in one batch): a move is ranked by the search's
value of the reply position, negated (a proved forced win for the other side the worst, a game over for us the best), ties in the kernel's move order. The first choice is always the
search's own best move at the full clock. This is said again in every report.

Nothing of any outside engine, judge or teacher enters a choice. The judge is used after the games, by `measure_match`, to measure only.

  .venv/bin/python -m catspace.incubator.analyst.discovery.sheet_player <game> --config main|fair|stand|question|control --openings-from <surprise meter dir> --pairs 12 --out DIR
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import cs_engine as CE

from catspace.incubator.analyst.discovery import question_sheet as QS
from catspace.incubator.analyst.discovery import wake
from catspace.io import paths

SATURATED = (1 << 40) - 1000           # |ledger value| at or above this is a line that ends the game (the surprise meter's constant)
FAR = float(1 << 41)                   # the rank of a reply position that is a forced win for the other side / a game over for us
DRAW = [50, 50]


# ------------------------------------------------------------------------------------------------------------------------------------ settings

@dataclass
class PlayerConfig:
    seconds: float = 0.2               # the search's own clock a move
    budget: "float | None" = None      # seconds for the tries after the first (None: `seconds`)
    max_tries: int = 5                 # the search's choices asked about, the first included
    use_standing: bool = True          # a loss that stood and is left unanswered rejects a move
    use_question: bool = True          # the question of the move (SAFE) is required
    rank_seconds: float = 0.03         # the search's clock on each reply position when the next choices are ordered
    rank_threads: int = 2
    light: bool = False                # the cheap reading (see `question_sheet.Reading`): the units and what they bear on from the codec, no fact view
    equal: bool = False                # EQUAL TIME: `seconds` is the clock of the WHOLE move; the search is given it less the running estimate of what the sheet costs a move, never under `floor`
    floor: float = 0.05                # the least clock the search is given under `equal`
    prior_cost: float = 0.005          # the estimate of the sheet's cost a move before any move has been timed
    filtered: bool = False             # when the first choice is rejected: ask the move question of the other moves in the sheet's own order (cheap), then let the search order the accepted ones
    max_alternatives: int = 4          # `filtered`: accepted moves the search orders (the first choice not counted)

    def sheet_budget(self) -> float:
        return self.seconds if self.budget is None else self.budget


CONFIGS = {
    "main": dict(),                                                    # the full clock for the search, and the same again for the sheet
    "fair": dict(seconds=0.1, budget=0.1),                             # half the clock for the search, the other half for the sheet (the plain player keeps the whole clock)
    "stand": dict(use_question=False),                                 # ablation: the standing test alone
    "question": dict(use_standing=False),                              # ablation: the question of the move alone
    "control": dict(use_standing=False, use_question=False),           # the guided player with both parts off: the plain search against itself
    # EQUAL TIME (plan 78.42, the cheap reading): the whole move, search and sheet, takes the plain player's clock; the sheet's measured cost is taken off the search's clock move by move
    "equal": dict(equal=True, light=True, filtered=True, use_standing=False, budget=0.04, rank_seconds=0.015),                    # the question of the move only
    "equal_both": dict(equal=True, light=True, filtered=True, budget=0.04, rank_seconds=0.015),                                 # the standing losses as well
    "equal_wide": dict(equal=True, light=True, filtered=True, use_standing=False, max_alternatives=8, max_tries=9, budget=0.06, rank_seconds=0.015),   # the question on up to eight next choices
}


# ------------------------------------------------------------------------------------------------------------------------------------ the search

class PlainSearch:
    """The wake's plain decision (`wake.play_self`'s call into `cs_engine.solve_batch` with one position), and the ordering of the next choices."""

    def __init__(self, variant: str, *, store: "str | None", specs: "dict | None", trace_path: str, roles: "dict | None" = None):
        self.variant, self.store, self.specs, self.trace_path = variant, store, specs or {}, trace_path
        self.roles = roles or wake.ROLES

    def _rows(self) -> dict:
        out = {}
        if os.path.exists(self.trace_path):
            for line in open(self.trace_path):
                row = json.loads(line)
                if row.get("kind") == "position" and row.get("label"):
                    out[int(row["label"].split(":")[1])] = row
        return out

    @staticmethod
    def _value(row: dict) -> "int | None":
        return next((s["value"] for s in row.get("steps", []) if s.get("operator") == "ledger" and s.get("value") is not None), None)

    def _solve(self, positions: list, seconds: float, threads: int) -> tuple:
        if os.path.exists(self.trace_path):
            os.remove(self.trace_path)
        labels = [f"s:{i}:0:0" for i in range(len(positions))]
        out = CE.solve_batch(positions, seconds, snapshot="s", trace_path=self.trace_path, threads=threads, prior=False, ties=False, reason=False, wdl=False, wdl_table=None,
                             store=self.store, specs=self.specs, labels=labels)
        return out, self._rows()

    def decide(self, start, moves: list, seconds: float) -> dict:
        """{move, proved, value, forced, seconds}: the search's best move. `forced`: it plays a proved forced win, or its value is a line that ends the game in our favour."""
        t0 = time.perf_counter()
        out, rows = self._solve([(self.variant, start, list(moves))], seconds, 1)
        mv, proved, _nodes, _secs, err = out[0]
        value = self._value(rows.get(0, {}))
        return {"move": mv, "proved": bool(proved), "value": value, "forced": bool(proved) or (value is not None and value >= SATURATED), "error": err, "seconds": time.perf_counter() - t0}

    def alternatives(self, start, moves: list, legal: list, exclude: list, seconds: float, threads: int, limit: int) -> dict:
        """{moves: the other legal moves best first by the search's value of the reply position negated, seconds}; at most `limit` of them."""
        t0 = time.perf_counter()
        rest = [m for m in legal if m not in exclude]
        scored, todo = [], []
        for i, m in enumerate(rest):
            _pos, mover, _legal, outcome, _key = CE.state(self.variant, start, list(moves) + [m])
            if outcome is not None:
                scored.append((-FAR if outcome != DRAW and self._won_by_us(start, moves, outcome) else 0.0, i, m))
            else:
                todo.append((i, m))
        if todo:
            out, rows = self._solve([(self.variant, start, list(moves) + [m]) for _i, m in todo], seconds, threads)
            for j, (i, m) in enumerate(todo):
                _mv, proved, _n, _s, _e = out[j]
                v = self._value(rows.get(j, {}))
                scored.append((FAR if proved else (float(v) if v is not None else 0.0), i, m))       # their value of the reply position: the lower, the better for us
        scored.sort(key=lambda x: (x[0], x[1]))
        return {"moves": [m for _v, _i, m in scored][:limit], "seconds": time.perf_counter() - t0}

    def _won_by_us(self, start, moves: list, outcome) -> bool:
        mover = CE.state(self.variant, start, list(moves))[1]
        return list(outcome)[self.roles[mover]] == 100


# ------------------------------------------------------------------------------------------------------------------------------------ the sheet

@dataclass
class Assessment:
    """What the sheet says about one move: the verdict of the question of the move, the standing losses left unanswered, and the words."""
    move: str
    verdict: str = QS.SAFE_V
    took: float = 0.0
    unanswered: list = field(default_factory=list)        # [{"unit": words, "after": value, "excess": value - took}]
    stake: float = 0.0                                    # what the move would lose by the sheet: the question's low bound, or the largest unanswered excess
    lines: list = field(default_factory=list)
    seconds: float = 0.0
    error: "str | None" = None


def accepts(a: Assessment, use_question: bool = True, use_standing: bool = True) -> bool:
    """THE ACCEPTANCE RULE. SAFE by the question of the move (when asked), and no standing loss left unanswered (when asked)."""
    if a.error is not None:
        return False
    if use_question and a.verdict != QS.SAFE_V:
        return False
    if use_standing and a.unanswered:
        return False
    return True


def unanswered_losses(after: list, took: float) -> list:
    """`after`: [(words of a unit that stood lost, what the other side's exchange on it is worth after the move)]; a unit is left unanswered when that is worth more than the move took."""
    return [{"unit": w, "after": v, "excess": v - took} for w, v in after if v - took > 0]


class SheetEngine:
    """The question sheet behind the player: one reading of the position, the standing losses, and the assessment of a move."""

    def __init__(self, env: QS.Env, worth: dict, params: "QS.Params | None" = None, light: bool = False):
        self.env, self.worth, self.params, self.light = env, dict(worth), params or QS.Params(), light

    def warm(self) -> None:
        """Read the game's one-time tables (the medium of movement, the goal's kind) and make one question, before any clock runs."""
        r = self.open(CE.state(self.env.variant, None, [])[0])
        self.env.medium()
        r.goal_kind()
        QS.move_question(self.env, r, r.legal[0], self.params)

    def order(self, r: QS.Reading, moves: list) -> list:
        """The moves in the sheet's own order of asking (struck landing places first, the heavier unit first)."""
        return QS.move_order(r, moves, self.params)

    def open(self, position: str) -> QS.Reading:
        r = QS.Reading(self.env, position, self.worth, light=self.light)
        r.params_k, r.passes, r.params = self.params.reach_k, self.params.passes, self.params
        return r

    def standing(self, r: QS.Reading) -> list:
        """[(unit, value)] of my weighty units (the goal unit apart) that are lost where they stand: the other side's exchange on them is worth more than nothing."""
        out, goal = [], r.goal_unit(QS.SELF)
        for u in r.units:
            if u.side != QS.SELF or not r.weighty(u) or u == goal:
                continue
            v = QS.exchange_stands(r, u, QS.OTHER, self.params)["value"]
            if v > 0:
                out.append((u, v))
        return out

    def assess(self, r: QS.Reading, move: str, standing: list, use_question: bool = True, use_standing: bool = True) -> Assessment:
        t0 = time.perf_counter()
        a = Assessment(move)
        try:
            if use_question:
                res = QS.move_question(self.env, r, move, self.params)
                a.verdict, a.took, a.lines = res["verdict"], res["took"] + res["credit"], QS.explain_move(res)
                a.stake = res["low"] if res["verdict"] == QS.LOSES_V else 0.0
            if use_standing and standing:
                st = r.try_state(r.position, move)
                if st is None:
                    a.error = "the kernel does not accept the move"
                elif st[3] is None:
                    child_r = QS.child_reading(r, st[0], self.params)
                    origin, dest = self.env.codec.move_ends(move)
                    if not use_question:
                        a.took = float(sum(r.worth_of(x.kind) for x in QS._taken_by(r, child_r)))
                        arrived = next((x for x in child_r.units if x.side == QS.OTHER and QS.place_text(x.at) == dest), None)
                        mover = next((x for x in r.units if x.side == QS.SELF and QS.place_text(x.at) == origin), None)
                        if arrived is not None and mover is not None and arrived.kind != mover.kind:
                            a.took += r.worth_of(arrived.kind) - r.worth_of(mover.kind)          # a unit that arrives as another kind is the same unit with its new worth
                    after = []
                    for u, _v in standing:
                        moved = QS.place_text(u.at) == origin
                        place = dest if moved else QS.place_text(u.at)
                        vc = next((x for x in child_r.units if x.side == QS.OTHER and (moved or x.kind == u.kind) and QS.place_text(x.at) == place), None)      # the moved unit is followed by where it lands, whatever it has become
                        if vc is not None:
                            after.append((QS.unit_words(u), QS.exchange_stands(child_r, vc, QS.SELF, self.params)["value"]))
                    a.unanswered = unanswered_losses(after, a.took)
                    if a.unanswered:
                        a.stake = max(a.stake, max(x["excess"] for x in a.unanswered))
                        a.lines.append("a loss that stood is left: " + "; ".join(f"{x['unit']} (its exchange is worth {x['after']:g} after the move, the move took {a.took:g})" for x in a.unanswered))
        except Exception as e:                                              # the sheet could not read it: the move is not accepted, and the row says why
            a.error = f"{type(e).__name__}: {e}"
        a.seconds = time.perf_counter() - t0
        return a


# ------------------------------------------------------------------------------------------------------------------------------------ the player

def kernel_state(variant: str):
    def state(start, moves: list) -> tuple:
        pos, mover, legal, outcome, _key = CE.state(variant, start, list(moves))
        return pos, mover, list(legal or []), outcome
    return state


class SheetPlayer:
    """The plain search with the sheet's second look. `choose(start, moves) -> (move, row)`; the row is the log of the decision (see `choose`)."""

    def __init__(self, search, engine, state, config: "PlayerConfig | None" = None, clock=time.perf_counter):
        self.search, self.engine, self.state, self.cfg, self.clock = search, engine, state, config or PlayerConfig(), clock
        self.costs: list = []              # seconds the sheet took (the whole move less the search) at each move of this game, under `equal`

    def sheet_estimate(self) -> float:
        """The running estimate of what the sheet costs a move: the mean of the moves timed so far, the prior before the first."""
        return sum(self.costs) / len(self.costs) if self.costs else self.cfg.prior_cost

    def search_clock(self) -> float:
        """The clock the search is given: `seconds`; under `equal`, `seconds` less the sheet's estimated cost, never under `floor`."""
        cfg = self.cfg
        return max(cfg.floor, cfg.seconds - self.sheet_estimate()) if cfg.equal else cfg.seconds

    def choose(self, start, moves: list) -> tuple:
        """-> (the move to play, the row): {ply, search_move, played, overridden, why (first_accepted / overridden / none_accepted / forced / one_legal / off / unreadable), proved, value, standing
        [words, value], tries [{move, verdict, accepted, took, unanswered, stake, seconds, error}], rejected_stake (what the search's choice would lose by the sheet, 0 when not overridden), reason
        (the words for the search's choice, the verdict lines), seconds {search, read, standing, first, tries, rank, total}}."""
        cfg, t_all = self.cfg, self.clock()
        position, _mover, legal, _outcome = self.state(start, moves)
        clock_s = self.search_clock()
        s = self.search.decide(start, moves, clock_s)
        first = s["move"]
        row = {"ply": len(moves), "search_move": first, "played": first, "overridden": False, "why": "off", "proved": s["proved"], "value": s["value"], "standing": [], "tries": [],
               "rejected_stake": 0.0, "reason": [], "search_clock": clock_s, "seconds": {"search": s["seconds"], "read": 0.0, "standing": 0.0, "first": 0.0, "tries": 0.0, "rank": 0.0, "total": 0.0}}

        def done(move, why):
            row.update(played=move, overridden=(move != first), why=why)
            row["seconds"]["total"] = self.clock() - t_all
            if cfg.equal:
                self.costs.append(max(0.0, row["seconds"]["total"] - s["seconds"]))
            return move, row

        if first is None:
            return done(first, "no_move")
        if len(legal) <= 1:
            return done(first, "one_legal")
        if s["forced"]:
            return done(first, "forced")
        if not (cfg.use_standing or cfg.use_question):
            return done(first, "off")
        t0 = self.clock()
        try:
            r = self.engine.open(position)
        except ValueError:
            return done(first, "unreadable")
        row["seconds"]["read"] = self.clock() - t0
        standing = []
        if cfg.use_standing:
            t0 = self.clock()
            standing = self.engine.standing(r)
            row["seconds"]["standing"] = self.clock() - t0
            row["standing"] = [[QS.unit_words(u), v] for u, v in standing]

        def ask(move: str) -> bool:
            a = self.engine.assess(r, move, standing, cfg.use_question, cfg.use_standing)
            ok = accepts(a, cfg.use_question, cfg.use_standing)
            row["tries"].append({"move": move, "verdict": a.verdict, "accepted": ok, "took": a.took, "unanswered": a.unanswered, "stake": a.stake, "seconds": a.seconds, "error": a.error, "lines": a.lines})
            return ok

        if ask(first):
            row["seconds"]["first"] = row["tries"][0]["seconds"]
            return done(first, "first_accepted")
        row["seconds"]["first"] = row["tries"][0]["seconds"]
        row["rejected_stake"] = row["tries"][0]["stake"]
        row["reason"] = list(row["tries"][0]["lines"])
        if cfg.max_tries <= 1:
            return done(first, "none_accepted")
        if cfg.filtered:
            return self._filtered(start, moves, legal, first, r, ask, row, done)
        alt = self.search.alternatives(start, moves, legal, [first], cfg.rank_seconds, cfg.rank_threads, cfg.max_tries - 1)
        row["seconds"]["rank"] = alt["seconds"]
        spent = 0.0
        for m in alt["moves"]:
            if spent >= cfg.sheet_budget():
                break
            ok = ask(m)
            spent += row["tries"][-1]["seconds"]
            row["seconds"]["tries"] = spent
            if ok:
                row["reason_played"] = list(row["tries"][-1]["lines"])
                return done(m, "overridden")
        return done(first, "none_accepted")

    def _filtered(self, start, moves: list, legal: list, first, r, ask, row: dict, done) -> tuple:
        """THE NEXT CHOICES, CHEAPLY: the move question is asked of the other moves in the sheet's own order (each costs about what a frame costs), up to `max_alternatives` accepted, while the time spent is
        under the budget; the search then orders only the accepted ones (one short search of each reply, in one batch) and the best by it is played. None accepted: the search's first choice."""
        cfg = self.cfg
        ok, spent = [], 0.0
        for m in self.engine.order(r, [x for x in legal if x != first]):
            if len(ok) >= cfg.max_alternatives or spent >= cfg.sheet_budget():
                break
            if ask(m):
                ok.append(m)
            spent += row["tries"][-1]["seconds"]
        row["seconds"]["tries"] = spent
        if not ok:
            return done(first, "none_accepted")
        pick = ok[0]
        if len(ok) > 1:
            alt = self.search.alternatives(start, moves, ok, [first], cfg.rank_seconds, cfg.rank_threads, 1)
            row["seconds"]["rank"] = alt["seconds"]
            pick = alt["moves"][0] if alt["moves"] else ok[0]
        row["reason_played"] = list(next(t for t in row["tries"] if t["move"] == pick)["lines"])
        return done(pick, "overridden")


# ------------------------------------------------------------------------------------------------------------------------------------ games

_WORLD: dict = {}


def _world(game: str, store: "str | None", out_dir: str) -> dict:
    """What a worker process builds once: the env, the worth table, the kernel's state function, the search and the sheet."""
    key = (game, store, out_dir)
    if _WORLD.get("key") != key:
        from catspace.incubator.analyst.discovery import game_store as GS
        env = QS.Env(game)
        specs = GS.spec_texts(game) if store else None
        os.makedirs(out_dir, exist_ok=True)
        search = PlainSearch(game, store=store, specs=specs, trace_path=str(Path(out_dir) / f"trace_{os.getpid()}.jsonl"))
        _WORLD.clear()
        worth = QS.store_worth(game)
        _WORLD.update(key=key, env=env, search=search, state=kernel_state(game), engine=SheetEngine(env, worth), light_engine=SheetEngine(env, worth, light=True))
        _WORLD["engine"].warm()
        _WORLD["light_engine"].warm()
    return _WORLD


def play_game(spec: dict) -> dict:
    """One game of the guided player (colour spec["guided"]: 0 first to move, 1 second) against the plain search, from the opening `spec["opening"]` (moves). The same end rules as the wake's
    `play_self`: the game's own outcome, a position met three times is a draw, a move cap is unfinished."""
    w = _world(spec["game"], spec["store"], spec["scratch"])
    cfg = PlayerConfig(**spec["config"])
    plain_cfg = PlayerConfig(seconds=spec["plain_seconds"], use_standing=False, use_question=False)
    guided = SheetPlayer(w["search"], w["light_engine"] if cfg.light else w["engine"], w["state"], cfg)
    plain = SheetPlayer(w["search"], w["engine"], w["state"], plain_cfg)
    moves, seen, rows, times = list(spec["opening"]), {}, [], {"guided": [], "plain": []}
    outcome, unfinished = None, False
    for k in range(len(moves)):                                          # the opening's positions count in the repetition rule, as in the wake
        key = CE.state(spec["game"], None, moves[:k])[4]
        seen[key] = seen.get(key, 0) + 1
    while True:
        _fen, _mover, legal, out_, key = CE.state(spec["game"], None, moves)
        if out_ is not None:
            outcome = list(out_)
            break
        if not legal:
            unfinished = True
            break
        seen[key] = seen.get(key, 0) + 1
        if seen[key] >= 3:
            outcome = list(DRAW)
            break
        if len(moves) >= spec["max_plies"]:
            unfinished = True
            break
        mine = (len(moves) % 2) == spec["guided"]
        t0 = time.perf_counter()
        mv, row = (guided if mine else plain).choose(None, moves)
        dt = time.perf_counter() - t0
        if mv is None:
            unfinished = True
            break
        times["guided" if mine else "plain"].append([len(moves), dt])
        if mine:
            rows.append(row)
        moves.append(mv)
    return {"gid": spec["gid"], "opening_id": spec["opening_id"], "guided": spec["guided"], "config": spec["config_name"], "moves": moves, "opening_len": len(spec["opening"]),
            "outcome": outcome, "unfinished": unfinished, "rows": rows, "times": times}


def openings_of(directories: list, plies: int = 4) -> list:
    """The first `plies` moves of every game of the surprise meter directories (the seeded random openings of the earlier games), distinct, in order."""
    out, seen = [], set()
    for d in directories:
        for line in (Path(d) / "games.jsonl").read_text().splitlines():
            if line.strip():
                mv = json.loads(line)["moves"][:plies]
                if tuple(mv) not in seen:
                    seen.add(tuple(mv))
                    out.append(mv)
    return out


def play_match(game: str, config_name: str, openings: list, out_dir: str, *, store: "str | None", workers: int = 6, max_plies: int = 120, overrides: "dict | None" = None,
               say=print) -> list:
    """Every opening twice, colours swapped: the guided player of `config_name` against the plain search at the full clock. -> the game dicts, also written to `<out_dir>/games.jsonl`."""
    cfg = {**CONFIGS[config_name], **(overrides or {})}
    plain_seconds = PlayerConfig().seconds
    specs = []
    for i, op in enumerate(openings):
        for colour in (0, 1):
            specs.append({"game": game, "store": store, "scratch": str(Path(out_dir) / "scratch"), "gid": f"{i}:{colour}", "opening_id": i, "guided": colour, "opening": op, "config": cfg,
                          "config_name": config_name, "plain_seconds": plain_seconds, "max_plies": max_plies})
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    t0, games = time.perf_counter(), []
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(workers) as pool:
        for g in pool.imap_unordered(play_game, specs):
            games.append(g)
            say(f"OBS progress {config_name}: game {g['gid']} done, {len(games)} of {len(specs)}, {time.perf_counter() - t0:.0f} s")
    games.sort(key=lambda g: (g["opening_id"], g["guided"]))
    with open(Path(out_dir) / "games.jsonl", "w") as f:
        for g in games:
            f.write(json.dumps(g) + "\n")
    return games


# ------------------------------------------------------------------------------------------------------------------------------------ the measures

def _mean(xs: list) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def game_stats(env: QS.Env, worth: dict, game: dict, goal_kind: "str | None") -> dict:
    """What happened to the units in one game, by the sheet's own outcome (`unit_loss`, each loss charged by `charged_ply`): for each side (the seat that moves first is 0), the weighty units
    LOST (a unit above the least worth, the goal unit apart), the plies they are charged to, and the balance of worth over all takings (worth taken from the other side less worth lost)."""
    events = env.codec.events(None, game["moves"])
    worth_of = env.worth_fn(worth)
    lowest = min([w for w in worth.values() if w > 0] or [0.0])
    first_side = events[0]["mover"][1] if events else None
    seat = lambda side: 0 if side == first_side else 1
    lost = {0: [], 1: []}
    balance = {0: 0.0, 1: 0.0}
    done = set()
    for e in events:
        if e["taken"] is None:
            continue
        taker, victim = seat(e["mover"][1]), seat(e["taken"][1])
        balance[taker] += worth_of(e["taken"][2])
        balance[victim] -= worth_of(e["taken"][2])
        uid, kind = e["taken"][0], e["taken"][2]
        if uid in done or kind == goal_kind or worth_of(kind) <= lowest:
            continue
        done.add(uid)
        res = QS.unit_loss(events, uid, 0, len(events), worth_of)
        if res is not None and res["status"] == QS.LOST:
            lost[victim].append({"kind": kind, "ply": res["ply"], "charged": QS.charged_ply(events, res), "balance": res["balance"]})
    return {"lost": lost, "balance": balance, "events": len(events)}


def judge_reads(judge_reader, game: dict, variant: str) -> dict:
    """The judge's reading of every position of the game that is read for the measure: {ply: {cp, e}} for the side to move; None past the end of a finished game. Measure only."""
    out = {}
    n = len(game["moves"])
    for p in range(game["opening_len"], n + 1):
        try:
            r = judge_reader.read(variant, None, game["moves"][:p])
        except Exception:
            continue
        out[p] = {"cp": r["cp"], "e": r["wdl"][0] + r["wdl"][1] / 2}
    return out


def _cap(x: "float | None", cap: float = 1000.0) -> "float | None":
    return None if x is None else max(-cap, min(cap, float(x)))


def judge_move_rows(game: dict, reads: dict, judge_reader, variant: str) -> list:
    """One record per decision of the game: who moved (guided or plain), the judge's loss of the move played (centipawns capped at 1000, expected result), and for an overridden move the judge's
    value of the search's choice (read from the position after it) against the move played."""
    rows = []
    guided_rows = {r["ply"]: r for r in game["rows"]}
    for p in range(game["opening_len"], len(game["moves"])):
        if p not in reads or p + 1 not in reads:
            continue
        who = "guided" if (p % 2) == game["guided"] else "plain"
        b, a = reads[p], reads[p + 1]
        rec = {"gid": game["gid"], "ply": p, "who": who, "loss_cp": None if b["cp"] is None or a["cp"] is None else min(1000.0, max(0.0, b["cp"] + a["cp"])), "loss_e": max(0.0, b["e"] - (1.0 - a["e"]))}
        gr = guided_rows.get(p)
        if who == "guided" and gr is not None and gr["overridden"]:
            try:
                alt = judge_reader.read(variant, None, game["moves"][:p] + [gr["search_move"]])
                e_alt, e_pl = 1.0 - (alt["wdl"][0] + alt["wdl"][1] / 2), 1.0 - a["e"]
                cp_alt = None if alt["cp"] is None else -_cap(alt["cp"])
                cp_pl = None if a["cp"] is None else -_cap(a["cp"])
                rec.update(overridden=True, e_search=e_alt, e_played=e_pl, d_e=e_pl - e_alt, d_cp=None if cp_alt is None or cp_pl is None else cp_pl - cp_alt, row=gr)
            except Exception:
                pass
        rows.append(rec)
    return rows


def measure_match(env: QS.Env, worth: dict, games: list, judge_reader, variant: str, label: str, goal_kind: "str | None", eps: float = 0.02, show: int = 5, say=print) -> dict:
    """The OBS lines for one match. `eps`: the judge calls an override better / worse when the expected result of the move played differs from the search's choice by more than this."""
    P = f"OBS ({label})"
    n = len(games)
    w = d = l = 0
    for g in games:
        if g["outcome"] is None:
            continue
        got = g["outcome"][g["guided"]]
        w, d, l = w + (got == 100), d + (got == 50), l + (got == 0)
    unfinished = sum(1 for g in games if g["outcome"] is None)
    score = (w + d / 2) / max(1, w + d + l)
    say(f"{P} games {n}, unfinished at the cap {unfinished} (not counted below); the guided player: wins {w}, draws {d}, losses {l}, score {score:.3f} of {w + d + l} finished games")
    # pair scores: each opening twice, colours swapped
    by_op: dict = {}
    for g in games:
        if g["outcome"] is not None:
            by_op.setdefault(g["opening_id"], []).append(g["outcome"][g["guided"]] / 100.0)
    pairs = [(op, sum(v)) for op, v in sorted(by_op.items()) if len(v) == 2]
    say(f"{P} pairs complete {len(pairs)}: guided scores 2.0/1.5/1.0/0.5/0.0 in {sum(1 for _o, s in pairs if s == 2)}/{sum(1 for _o, s in pairs if s == 1.5)}/{sum(1 for _o, s in pairs if s == 1)}/"
        f"{sum(1 for _o, s in pairs if s == 0.5)}/{sum(1 for _o, s in pairs if s == 0)}")
    stats = [game_stats(env, worth, g, goal_kind) for g in games]
    for who, seat_of in (("guided", lambda g: g["guided"]), ("plain", lambda g: 1 - g["guided"])):
        lost = [len(s["lost"][seat_of(g)]) for g, s in zip(games, stats)]
        bal = [s["balance"][seat_of(g)] for g, s in zip(games, stats)]
        charged_own = [sum(1 for x in s["lost"][seat_of(g)] if x["charged"] is not None and x["charged"] % 2 == seat_of(g)) for g, s in zip(games, stats)]
        say(f"{P} {who}: weighty units LOST {sum(lost)} in {n} games ({_mean(lost):.2f} a game), charged to its own moves {sum(charged_own)}; balance of worth over all takings, mean a game {_mean(bal):+.2f} (sum {sum(bal):+.0f})")
    # overrides
    rows = [r for g in games for r in g["rows"]]
    ov = [r for r in rows if r["overridden"]]
    why: dict = {}
    for r in rows:
        why[r["why"]] = why.get(r["why"], 0) + 1
    say(f"{P} guided decisions {len(rows)}; by outcome of the sheet: " + ", ".join(f"{k} {v}" for k, v in sorted(why.items())))
    say(f"{P} overrides {len(ov)} in {n} games ({len(ov) / max(1, n):.2f} a game), share of the guided positions {len(ov) / max(1, len(rows)):.4f}; the search's choice would lose by the sheet, sum of stake {sum(r['rejected_stake'] for r in ov):g} "
        f"(mean {_mean([r['rejected_stake'] for r in ov]):.2f} an override); rejected-and-kept (none accepted) {why.get('none_accepted', 0)}")
    # charged losses at the guided player's own plies
    charged_plies = {}
    for g, s in zip(games, stats):
        mine = g["guided"]
        charged_plies[g["gid"]] = {x["charged"] for x in s["lost"][mine] if x["charged"] is not None and x["charged"] % 2 == mine}
    ov_hit = sum(1 for g in games for r in g["rows"] if r["overridden"] and r["ply"] in charged_plies[g["gid"]])
    ok_hit = sum(1 for g in games for r in g["rows"] if not r["overridden"] and r["ply"] in charged_plies[g["gid"]])
    ok_n = sum(1 for g in games for r in g["rows"] if not r["overridden"])
    say(f"{P} the guided player's own move at the overridden ply was followed by a charged loss (`charged_ply`) in {ov_hit} of {len(ov)} overrides; at its other plies in {ok_hit} of {ok_n}")
    # time
    for who in ("guided", "plain"):
        ts = [t for g in games for _p, t in g["times"][who]]
        say(f"{P} time a move, {who}: mean {1000 * _mean(ts):.0f} ms, max {1000 * max(ts, default=float('nan')):.0f} ms over {len(ts)} moves")
    tg, tp = [t for g in games for _p, t in g["times"]["guided"]], [t for g in games for _p, t in g["times"]["plain"]]
    say(f"{P} time a move, the guided player minus the plain player: mean {1000 * (_mean(tg) - _mean(tp)):+.1f} ms, max {1000 * (max(tg, default=float('nan')) - max(tp, default=float('nan'))):+.0f} ms; "
        f"moves over 250 ms: guided {sum(1 for t in tg if t > 0.25)} of {len(tg)}, plain {sum(1 for t in tp if t > 0.25)} of {len(tp)}")
    say(f"{P} the search's own clock a guided move: mean {1000 * _mean([r['search_clock'] for r in rows if 'search_clock' in r]):.1f} ms, min {1000 * min([r['search_clock'] for r in rows if 'search_clock' in r], default=float('nan')):.0f} ms")
    sec = lambda k: [r["seconds"][k] for r in rows]
    say(f"{P} guided time split, mean ms: search {1000 * _mean(sec('search')):.0f}, read {1000 * _mean(sec('read')):.0f}, standing {1000 * _mean(sec('standing')):.1f}, first question {1000 * _mean(sec('first')):.0f}, "
        f"later tries {1000 * _mean(sec('tries')):.0f}, ordering the next choices {1000 * _mean(sec('rank')):.0f}")
    # the judge, after the games, to measure only
    jrows, over = [], []
    for g in games:
        reads = judge_reads(judge_reader, g, variant)
        jrows += judge_move_rows(g, reads, judge_reader, variant)
    for who in ("guided", "plain"):
        rs = [r for r in jrows if r["who"] == who]
        cps = [r["loss_cp"] for r in rs if r["loss_cp"] is not None]
        say(f"{P} judge (measure only), {who}: moves {len(rs)}, mean loss {_mean(cps):.1f} centipawns (cap 1000 a move), {_mean([r['loss_e'] for r in rs]):.4f} expected result")
    over = [r for r in jrows if r.get("overridden")]
    better = [r for r in over if r["d_e"] > eps]
    worse = [r for r in over if r["d_e"] < -eps]
    equal = [r for r in over if abs(r["d_e"]) <= eps]
    cpd = [r["d_cp"] for r in over if r["d_cp"] is not None]
    say(f"{P} judge on overridden moves {len(over)} (tolerance {eps} expected result): override better {len(better)}, equal {len(equal)}, worse {len(worse)}; mean expected result played minus search's choice {_mean([r['d_e'] for r in over]):+.4f}, "
        f"mean centipawns {_mean(cpd):+.1f}")
    for name, pick in (("better", sorted(better, key=lambda r: -r["d_e"])), ("worse", sorted(worse, key=lambda r: r["d_e"]))):
        for r in pick[:show]:
            say(f"{P} example override the judge calls {name}: game {r['gid']} ply {r['ply']}: search {r['row']['search_move']}, played {r['row']['played']}, expected result {r['e_played']:.3f} against {r['e_search']:.3f}; "
                f"why: {' | '.join(x.strip() for x in r['row']['reason'])}")
    return {"wins": w, "draws": d, "losses": l, "overrides": len(ov), "better": len(better), "worse": len(worse), "equal": len(equal)}


# ------------------------------------------------------------------------------------------------------------------------------------ command line

def main(argv: list) -> int:
    from catspace.incubator.analyst.discovery import game_store as GS
    from catspace.incubator.analyst.discovery import judges as J
    from catspace.incubator.analyst.discovery import surprise_meter as SM
    from catspace.incubator.analyst.discovery import wdl_sleep as WS
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("game")
    ap.add_argument("--config", default="main", choices=sorted(CONFIGS))
    ap.add_argument("--openings-from", action="append", default=[], help="surprise meter directories whose games' first plies are the openings")
    ap.add_argument("--pairs", type=int, default=12)
    ap.add_argument("--seconds", type=float, default=None)
    ap.add_argument("--max-tries", type=int, default=None)
    ap.add_argument("--max-plies", type=int, default=120)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--judge-engine", default=SM.DEFAULT_ENGINE)
    ap.add_argument("--judge-depth", type=int, default=10)
    ap.add_argument("--out", required=True)
    ap.add_argument("--measure-only", action="store_true", help="read <out>/games.jsonl instead of playing")
    a = ap.parse_args(argv)
    store = None
    if GS.head(a.game):
        store = str(GS.path_of(a.game, GS.head(a.game)))
    over = {}
    if a.seconds is not None:
        over["seconds"] = a.seconds
    if a.max_tries is not None:
        over["max_tries"] = a.max_tries
    out = Path(a.out)
    t0 = time.perf_counter()
    if a.measure_only:
        games = [json.loads(x) for x in (out / "games.jsonl").read_text().splitlines() if x.strip()]
    else:
        ops = openings_of(a.openings_from)[:a.pairs]
        print(f"OBS setup: config {a.config} {json.dumps({**CONFIGS[a.config], **over})}, plain clock {PlayerConfig().seconds}, openings {len(ops)}, max plies {a.max_plies}, snapshot {CE.snapshot_info(store)['id'] if store else 'none'}, workers {a.workers}", flush=True)
        games = play_match(a.game, a.config, ops, str(out), store=store, workers=a.workers, max_plies=a.max_plies, overrides=over, say=lambda m: print(m, flush=True))
    print(f"OBS played {len(games)} games in {time.perf_counter() - t0:.0f} s", flush=True)
    env = QS.Env(a.game)
    worth = QS.store_worth(a.game)
    goal_kind = QS.Reading(env, CE.state(a.game, None, [])[0], worth).goal_kind()
    judge = WS.UciWdlJudge(a.judge_engine, depth=a.judge_depth)
    desc = {**J.describe(judge), "depth": a.judge_depth}
    reader = SM.CachedReader(judge, str(out / "judge_cache.json"), tag=json.dumps(desc, sort_keys=True))
    try:
        measure_match(env, worth, games, reader, a.game, a.config, goal_kind, say=lambda m: print(m, flush=True))
    finally:
        reader.close()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
