"""PLAN 78.50: PLAY WITH IT.  The fitted number of plan 78.49 as the choice among the root moves the search holds EQUAL.  A measurement by matches; the wake and every existing player are unchanged,
the players below live in this file and are read by no other code.  Game-free: the game is named only by the caller (`--variant`), and the statements are the engine's.

A. HOW THE CONTROL CHOOSES AMONG EQUALS.  The control is the plain search (`sheet_player.PlainSearch`, the wake's own call).  Its ledger search keeps the FIRST root move of the best value in its root
   order (rust/engine/tactics/src/ledger.rs: `v > best`, the previous iteration's best first, then the lemmas' order when a store has lemmas, the moves that take, the killers, the history): no random
   choice, no use of any number but the value.  `stage_a` prints, on the positions of the kept games, how often several root moves are held equal and how many (`--stage a`).
B. THE CANDIDATE.  `cs_engine.solve_equals_batch` (rust/engine/py/src/lib.rs): the same decision (proof, claims, lemma order, the ledger search, the same move) with the root moves held equal to the
   best handed back by name.  Equality is exact: after the deepening each root move is asked "at least the best?" by a null window at the depth the ledger finished, the table of the run warm
   (`Search::ties`, unchanged).  Under the search's pruning a root move's value is exact only for the best; the null-window question is the sound way to ask "is this move at least the best", and a move
   that holds is as good as the best by the search's own value.  The ledger deepens within LEDGER_SHARE of the clock and the equal reading takes what is left of the same clock, so the whole move
   stays within the clock; where the clock stops the reading (`cut`) the list is only part of the equals (every move in it held the best, the moves read first), the search's own move is played there by every
   player and the share of such positions is printed.  The candidate plays,
   among the equals, the move whose child has the highest fitted number (the fit P of 78.49, arrival target, horizon HORIZON, weights fitted on the even games and kept in a file); the number is
   the weights on (the statements of the side that moved less the other side's) on the position after the move, `SheetModel.subgoal_children` (sign convention of subgoal_fit_measure).  A tie at the
   top is the search's own move when it is among them, else the lowest index.  The number enters only as this choice among equals: the value is untouched (AGENTS rule 7).  Its reading time is taken
   off the candidate's clock (the running estimate of the game, as `sheet_player.SheetPlayer.search_clock`).
C. THE MATCHES.  Openings as `opportunity_measure.random_opening` makes them, every opening played with both colours (a pair); seeds from 7000 (the control against control match: from 8000);
   the same clock for both sides; score = wins + half the draws (a game stopped at the move cap counts as a draw and is printed); a 90 pct band by bootstrap over opening pairs.  Matches: the candidate
   against the control; the control against the control (the noise floor: the engine takes no seed, the matches differ in openings and in the jitter of the clocks); a player that picks among the
   equals at random (a random choice per position, seeded by the game) against the control; and, added to isolate what the clock split alone costs, the search's own move with the same reading
   (`engine_pick`) against the control.
D. Printed beside: how often the pick differed from the search's own move in the same position (the proxy for the control's: the same search under the same clock allotment, since the control at
   its full clock is not asked a second time), the share of positions with more than one equal best move and the mean number, the depth the ledger finished and the wall time a move for each player,
   and the arrival of CONFIRMED threat moves for each side by the forcing read (fuse FUSE) on a sample of the games (`--stage arrival`).

  .venv/bin/python -m catspace.incubator.analyst.discovery.subgoal_play_measure --stage fit|match|stage_a|arrival|report --variant V ...
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import random
import time
from pathlib import Path

import numpy as np

import cs_engine as CE

from catspace.incubator.analyst.discovery import forcing_arrival_measure as FA
from catspace.incubator.analyst.discovery import forcing_opportunity_measure as FO
from catspace.incubator.analyst.discovery import opportunity_measure as OM
from catspace.incubator.analyst.discovery import sheet_player as SP
from catspace.incubator.analyst.discovery import subgoal_fit_measure as SG
from catspace.io import paths

HORIZON = 8                      # the middle horizon of 78.49, fixed before any match
TARGET = "arr"                   # the arrival target
CLOCK = 0.2                      # seconds a move for both sides, as the earlier plans
LEDGER_SHARE = 0.4               # of the clock for the ledger's deepening; the equal reading takes what is left.  Fixed before any match from a probe of 60 positions (no match): at 0.4 the clock cut the
                                 # equal reading in about a tenth of the positions that had several equals, at 0.5 in a third, at 0.8 in nearly all (the reading is a full pass over the root moves)
FUSE = 5000                      # the forcing read's fuse of 78.48
MAX_PLIES = 120
DRAW = [50, 50]
BOOT_DRAWS = 1000
BOOT_SEED = 7050
OPENING_PLIES = 4
FIRST_SEED = 7000
NOISE_SEED = 8000
PLAYERS = ("control", "candidate", "random_equals", "engine_pick")
MATCHES = {"candidate_v_control": ("candidate", "control", FIRST_SEED), "control_v_control": ("control", "control", NOISE_SEED), "random_v_control": ("random_equals", "control", FIRST_SEED), "engine_v_control": ("engine_pick", "control", FIRST_SEED),
           "candidate_v_engine": ("candidate", "engine_pick", FIRST_SEED)}


def out_root() -> Path:
    return Path(paths.experiment("subgoal_play_78_50"))


# --------------------------------------------------------------------------------------------------------------------------- the weights

def write_weights(path: "str | Path", names: list, model: dict, horizon: int, label: str, rows: int, games: int) -> dict:
    """The fitted weights on the unstandardised differences (ours less theirs), with the statements' names and the standardisation they came from."""
    raw = SG.raw_weights(model)
    doc = {"label": label, "target": TARGET, "horizon": horizon, "names": list(names), "weights": [float(x) for x in raw], "coef_standardised": [float(x) for x in model["coef"]], "mean": [float(x) for x in model["mean"]],
           "scale": [float(x) for x in model["scale"]], "intercept": float(model["intercept"]), "alpha": float(model["alpha"]), "fit_rows": int(rows), "fit_games": int(games),
           "note": "weights on (statement of the side that moved less the other side's) of the position after a move; fitted on the even games only"}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(doc, indent=1))
    return doc


def load_weights(path: "str | Path") -> dict:
    return json.loads(Path(path).read_text())


def fit_weights(games_dir: str, reads: str, opp_table: str, out_file: str, threads: int, jobs: int = -1) -> "tuple[list, dict]":
    """Fit P on the arrival target at HORIZON on the even games and keep the weights in `out_file`; -> (OBS lines, the document)."""
    t0 = time.perf_counter()
    records = sorted(FO.read_games(games_dir), key=lambda r: r["game"])
    variant = records[0]["variant"]
    reads_by_id = {}
    with open(reads) as f:
        for line in f:
            g = json.loads(line)
            reads_by_id[g["game"]] = g["plies"]
    z = np.load(opp_table)
    a_opp = {k: z[k] for k in z.files if k != "n_games"}
    names, stats = SG.statements_of(variant, records, threads, str(Path(out_file).with_name(f"statements_n{len(records)}.npz")))
    t = SG.table_of(records, reads_by_id, stats, a_opp)
    fit, _test = SG.split_rows(t)
    m = SG.fit_all(t, fit, "P", TARGET, HORIZON, jobs)
    if m is None:
        raise SystemExit("too few rows to fit")
    stat_names = [n for n, _x in names]
    doc = write_weights(out_file, stat_names, m, HORIZON, Path(games_dir).name, len(fit), len(np.unique(t["game"][fit])))
    lines = [f"OBS subgoal_play fit: the P fit of the arrival target at horizon {HORIZON} on the even games of {Path(games_dir).name}: {len(fit)} quiet rows, {doc['fit_games']} games, penalty {m['alpha']:.5f} by {SG.FOLDS}-fold cross-validation grouped by game, "
             f"statements kept {int((m['coef'] != 0).sum())} of {len(stat_names)}; weights kept in {out_file}"]
    for i in sorted(range(len(stat_names)), key=lambda j: -abs(m["coef"][j])):
        if m["coef"][i] != 0:
            lines.append(f"OBS subgoal_play fit: weight {stat_names[i]}: {doc['weights'][i]:+.6f} per unit of (ours less theirs), {m['coef'][i]:+.5f} per standard deviation")
    lines.append(f"OBS subgoal_play fit: wall {time.perf_counter() - t0:.1f} s")
    return lines, doc


# --------------------------------------------------------------------------------------------------------------------------- the equal moves and the picks

def equal_set(row: tuple) -> dict:
    """The engine's `solve_equals_batch` row as a dict: move, proved, nodes, seconds, error, equals (names), equal_indices, depth, value, ties_seconds, legal, cut."""
    keys = ("move", "proved", "nodes", "seconds", "error", "equals", "equal_indices", "depth", "value", "ties_seconds", "legal", "cut")
    d = dict(zip(keys, row))
    d["equals"], d["equal_indices"] = list(d["equals"]), list(d["equal_indices"])
    return d


def pick_by_number(engine_move: str, equals: list, indices: list, numbers: dict) -> str:
    """Among the equal moves the one whose child has the highest number (`numbers`: legal-move index -> number); a move with no number is not a candidate (a position that is over has none); a tie at the
    top goes to the search's own move when it is among them, else to the lowest index; with no number at all the search's own move."""
    ranked = [(numbers[i], i, m) for m, i in zip(equals, indices) if i in numbers]
    if not ranked:
        return engine_move
    top = max(v for v, _i, _m in ranked)
    best = sorted((i, m) for v, i, m in ranked if v == top)
    for _i, m in best:
        if m == engine_move:
            return m
    return best[0][1]


def pick_random(equals: list, rng: random.Random) -> str:
    return equals[rng.randrange(len(equals))]


# --------------------------------------------------------------------------------------------------------------------------- one game

_WORLD: dict = {}


def world(spec: dict) -> dict:
    """What a worker builds once: the store and its texts, the control's search and, when a candidate plays, the sheet model and the weights."""
    key = (spec["variant"], spec["scratch"], spec.get("weights"))
    if _WORLD.get("key") != key:
        from catspace.incubator.analyst.discovery import game_store as GS
        variant = spec["variant"]
        store = str(GS.path_of(variant, GS.head(variant))) if GS.head(variant) else None
        specs = GS.spec_texts(variant) if store else None
        os.makedirs(spec["scratch"], exist_ok=True)
        _WORLD.clear()
        _WORLD.update(key=key, variant=variant, store=store, specs=specs or {}, search=SP.PlainSearch(variant, store=store, specs=specs, trace_path=str(Path(spec["scratch"]) / f"trace_{os.getpid()}.jsonl")))
        if spec.get("weights"):
            from catspace.incubator.analyst.discovery import question_sheet as QS
            from catspace.incubator.analyst.discovery import sheet_rust as SR
            _WORLD["sheet"] = SR.model(QS.Env(variant), QS.store_worth(variant))
            _WORLD["weights"] = load_weights(spec["weights"])["weights"]
    return _WORLD


class Control:
    """The plain search at the full clock: the wake's own call, as the control of the 78.4x matches."""

    def __init__(self, w: dict, seconds: float):
        self.w, self.seconds = w, seconds

    def choose(self, moves: list) -> tuple:
        out, rows = self.w["search"]._solve([(self.w["variant"], None, list(moves))], self.seconds, 1)
        depth = next((s.get("depth") for s in rows.get(0, {}).get("steps", []) if s.get("operator") == "ledger"), None)
        return out[0][0], {"depth": depth}


class EqualsPlayer:
    """The search's move and the equal moves in one call, then a choice among the equals: `kind` is 'candidate' (the highest fitted number of the child), 'random_equals', or 'engine_pick' (the search's own move: the same reading and clock allotment, no change of the pick)."""

    def __init__(self, w: dict, seconds: float, kind: str, seed: str):
        self.w, self.seconds, self.kind, self.seed = w, seconds, kind, seed
        self.costs: list = []                      # seconds the number took at each move of this game (the candidate only)
        self.prior_cost = 0.005

    def clock(self) -> float:
        """The search's clock: the move's clock less the running estimate of what the number costs (the candidate), never under half of it."""
        if self.kind != "candidate":
            return self.seconds
        est = sum(self.costs) / len(self.costs) if self.costs else self.prior_cost
        return max(0.5 * self.seconds, self.seconds - est)

    def numbers(self, moves: list, engine_move: str) -> dict:
        """legal-move index -> the fitted number after that move, for the positions the reading gives."""
        res = self.w["sheet"].subgoal_children([(None, list(moves) + [engine_move])], 1, [[len(moves)]], self.w["weights"])
        rows = res[0]
        return dict(rows[0][2]) if rows else {}

    def choose(self, moves: list) -> tuple:
        w = self.w
        r = equal_set(CE.solve_equals_batch([(w["variant"], None, list(moves))], self.clock(), store=w["store"], specs=w["specs"], threads=1, ledger_share=LEDGER_SHARE)[0])
        if r["move"] is None:
            return None, {}
        eq = r["equals"] or [r["move"]]
        pick = r["move"]
        if len(eq) > 1 and not r["cut"]:          # a cut reading is only part of the equals (it lists the moves read first): the search's own move is played then, whatever the player
            if self.kind == "engine_pick":
                pass
            elif self.kind == "random_equals":
                pick = pick_random(eq, random.Random(f"{self.seed}:{len(moves)}"))
            else:
                t0 = time.perf_counter()
                pick = pick_by_number(r["move"], eq, r["equal_indices"], self.numbers(moves, r["move"]))
                self.costs.append(time.perf_counter() - t0)
        return pick, {"legal": r["legal"], "equals": len(eq), "differs": int(pick != r["move"]), "cut": int(bool(r["cut"])), "depth": r["depth"], "proved": int(bool(r["proved"]))}


EXTRA_PLAYERS: dict = {}          # plan 78.51: players another script registers by kind (a worker imports the spec's "plugin" module, which fills this)
WEIGHTED = {"candidate"}          # the kinds that read the fitted number: a match with one of them carries the weights file
PLUGIN: dict = {}                 # match name -> the module a worker imports before it builds its players


def make_player(kind: str, w: dict, seconds: float, seed: str):
    if kind in EXTRA_PLAYERS:
        return EXTRA_PLAYERS[kind](w, seconds, kind, seed)
    return Control(w, seconds) if kind == "control" else EqualsPlayer(w, seconds, kind, seed)


def play_game(spec: dict) -> dict:
    """One game: player `a` (colour `colour`: 0 moves first) against player `b`, from the opening `spec['opening']`.  The end rules are the wake's `play_self` as `sheet_player.play_game` has them:
    the game's own outcome, a position met three times is a draw, a move cap is unfinished.  -> the game with, per decision of each player, its log row [ply, wall seconds, log]."""
    if spec.get("plugin"):
        import importlib
        importlib.import_module(spec["plugin"])
    w = world(spec)
    variant = spec["variant"]
    players = {0: make_player(spec["a"] if spec["colour"] == 0 else spec["b"], w, spec["seconds"], f"{spec['gid']}:0"), 1: make_player(spec["a"] if spec["colour"] == 1 else spec["b"], w, spec["seconds"], f"{spec['gid']}:1")}
    moves, seen, logs = list(spec["opening"]), {}, {0: [], 1: []}
    outcome, unfinished = None, False
    for k in range(len(moves)):
        key = CE.state(variant, None, moves[:k])[4]
        seen[key] = seen.get(key, 0) + 1
    while True:
        _pos, _mover, legal, out_, key = CE.state(variant, None, moves)
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
        seat = len(moves) % 2
        t0 = time.perf_counter()
        mv, log = players[seat].choose(moves)
        dt = time.perf_counter() - t0
        if mv is None:
            unfinished = True
            break
        logs[seat].append([len(moves), dt, log])
        moves.append(mv)
    return {"gid": spec["gid"], "seed": spec["seed"], "colour": spec["colour"], "a": spec["a"], "b": spec["b"], "moves": moves, "opening_len": len(spec["opening"]), "outcome": outcome,
            "unfinished": unfinished, "log_a": logs[spec["colour"]], "log_b": logs[1 - spec["colour"]]}


# --------------------------------------------------------------------------------------------------------------------------- scoring

def a_score(game: dict) -> float:
    """The score of player `a` in a game: 1, a half or 0; a game stopped at the cap is a draw."""
    if game["outcome"] is None:
        return 0.5
    return game["outcome"][game["colour"]] / 100.0


def pair_scores(games: list) -> dict:
    """{opening seed: sum of player a's scores in the two games of the opening} for the openings that have both colours."""
    by: dict = {}
    for g in games:
        by.setdefault(g["seed"], {})[g["colour"]] = a_score(g)
    return {s: v[0] + v[1] for s, v in sorted(by.items()) if len(v) == 2}


def tally(games: list) -> "tuple[int, int, int, int]":
    """(wins, draws, losses of player a, games stopped at the cap, counted among the draws)."""
    w = d = l = 0
    for g in games:
        s = a_score(g)
        w, d, l = w + (s == 1), d + (s == 0.5), l + (s == 0)
    return w, d, l, sum(1 for g in games if g["outcome"] is None)


def score_band(pairs: dict, draws: int = BOOT_DRAWS, seed: int = BOOT_SEED) -> "tuple[float, float, float]":
    """(score, 5th, 95th percentile) of player a's score per game, resampling the opening pairs with replacement."""
    v = np.array(list(pairs.values()), dtype=float)
    if len(v) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boots = np.array([v[rng.integers(0, len(v), len(v))].mean() / 2 for _ in range(draws)])
    return float(v.mean() / 2), float(np.percentile(boots, 5)), float(np.percentile(boots, 95))


def summarise_logs(games: list, side: str) -> dict:
    """The logs of one side ('log_a' or 'log_b') over the games: decisions, mean wall seconds, mean ledger depth; for the equal-reading players also the share of decisions with more than one equal move,
    the mean number of equals (over the decisions with two or more legal moves), the share where the pick differed from the search's own move, the same among the decisions with several equals,
    and the share where the clock cut the equal reading."""
    rows = [r for g in games for r in g[side]]
    out = {"decisions": len(rows), "wall": float(np.mean([r[1] for r in rows])) if rows else float("nan")}
    depths = [r[2]["depth"] for r in rows if r[2].get("depth") is not None]
    out["depth"] = float(np.mean(depths)) if depths else float("nan")
    eq = [r[2] for r in rows if "equals" in r[2] and r[2].get("legal", 0) >= 2]
    out["with_equals_log"] = len(eq)
    if eq:
        whole = [x for x in eq if not x["cut"]]                       # the positions where the equal reading was not cut by the clock
        many = [x for x in whole if x["equals"] > 1]
        out["cut"] = len(eq) - len(whole)
        out["whole"] = len(whole)
        out["many"] = len(many)
        out["mean_equals"] = float(np.mean([x["equals"] for x in whole])) if whole else float("nan")
        out["mean_equals_many"] = float(np.mean([x["equals"] for x in many])) if many else float("nan")
        out["differs"] = sum(x["differs"] for x in eq)
        out["differs_many"] = sum(x["differs"] for x in many)
    return out


def match_lines(name: str, a: str, b: str, games: list, draws: int = BOOT_DRAWS) -> list:
    w, d, l, unf = tally(games)
    pairs = pair_scores(games)
    sc, lo, hi = score_band(pairs, draws)
    lines = [f"OBS subgoal_play match {name}: player {a} against player {b}: games {len(games)}, opening pairs {len(pairs)}; {a}: W {w} D {d} L {l} (draws include {unf} games stopped at the move cap), score {sc:.4f} [{lo:.4f}, {hi:.4f}] 90 pct band by bootstrap over opening pairs ({draws} draws)"]
    for who, side in ((a, "log_a"), (b, "log_b")):
        s = summarise_logs(games, side)
        t = f"OBS subgoal_play match {name}: {who} ({side[-1]}): decisions {s['decisions']}, mean wall {1000 * s['wall']:.1f} ms a move, mean ledger depth {s['depth']:.2f}"
        if "many" in s:
            m = max(1, s["with_equals_log"])
            t += (f"; decisions with two or more legal moves {s['with_equals_log']}, the clock cut the equal reading in {s['cut']} ({100 * s['cut'] / m:.1f} pct; the search's own move is played there); of the {s['whole']} read whole: several equal best moves in {s['many']} "
                  f"({100 * s['many'] / max(1, s['whole']):.1f} pct), mean number of equal moves {s['mean_equals']:.2f} (over those with several: {s['mean_equals_many']:.2f}); "
                  f"the pick differed from the search's own move in {s['differs']} decisions ({100 * s['differs'] / m:.1f} pct of all decisions with two or more legal moves, {100 * s['differs_many'] / max(1, s['many']):.1f} pct of those with several equals)")
        lines.append(t)
    return lines


# --------------------------------------------------------------------------------------------------------------------------- the match

def opening_seeds(first: int, pairs: int) -> list:
    return list(range(first, first + pairs))


def specs_of(name: str, variant: str, pairs: int, seconds: float, weights: "str | None", scratch: str, first: "int | None" = None) -> list:
    a, b, seed0 = MATCHES[name]
    specs = []
    for s in opening_seeds(seed0 if first is None else first, pairs):
        opening = OM.random_opening(variant, s, OPENING_PLIES)
        for colour in (0, 1):
            specs.append({"variant": variant, "scratch": scratch, "gid": f"{s}:{colour}", "seed": s, "colour": colour, "a": a, "b": b, "opening": opening, "seconds": seconds, "max_plies": MAX_PLIES,
                          "weights": weights if WEIGHTED & {a, b} else None, "plugin": PLUGIN.get(name)})
    return specs


def run_match(name: str, variant: str, pairs: int, workers: int, seconds: float, weights: "str | None", out_dir: "str | Path", first: "int | None" = None, say=print) -> list:
    """Play the match, appending every finished game to `<out_dir>/games.jsonl` (a rerun skips the games already there); -> all the games of the match."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ck = out / "games.jsonl"
    done = {}
    if ck.exists():
        for line in ck.read_text().splitlines():
            if line.strip():
                g = json.loads(line)
                done[g["gid"]] = g
    todo = [s for s in specs_of(name, variant, pairs, seconds, weights, str(out / "scratch"), first) if s["gid"] not in done]
    t0 = time.perf_counter()
    n_all = len(todo) + len(done)

    def keep(g: dict) -> None:
        done[g["gid"]] = g
        with open(ck, "a") as f:
            f.write(json.dumps(g) + "\n")
        el = time.perf_counter() - t0
        say(f"OBS progress match {name}: {len(done)} of {n_all} games, {el:.0f} s since start of this run" + (f", estimated {el / max(1, len(done) - (n_all - len(todo))) * (n_all - len(done)):.0f} s left" if todo else ""))

    if workers <= 1 or not todo:
        for s in todo:
            keep(play_game(s))
    else:
        with multiprocessing.get_context("spawn").Pool(workers) as pool:
            for g in pool.imap_unordered(play_game, todo, chunksize=1):
                keep(g)
    wanted = {s["gid"] for s in specs_of(name, variant, pairs, seconds, weights, str(out / "scratch"), first)}
    return [done[g] for g in sorted(wanted, key=lambda x: (int(x.split(":")[0]), x))]


# --------------------------------------------------------------------------------------------------------------------------- stage A: the equals on the kept games

def stage_a(variant: str, records: list, seconds: float, threads: int, ckpt: Path, chunk: int, say=print) -> list:
    """For every position of the kept games (before the move, the game not over): the legal moves, the moves the search holds equal, whether the clock cut the reading and whether the search's own move is
    the lowest-index one among the equals.  Read in chunks, each appended to `ckpt` so a stop loses one chunk."""
    from catspace.incubator.analyst.discovery import game_store as GS
    store = str(GS.path_of(variant, GS.head(variant))) if GS.head(variant) else None
    specs = GS.spec_texts(variant) if store else {}
    items = [(r["game"], t) for r in records for t in range(len(r["moves"]))]
    by_id = {r["game"]: r for r in records}
    done = set()
    rows = []
    if ckpt.exists():
        for line in ckpt.read_text().splitlines():
            if line.strip():
                x = json.loads(line)
                done.add((x["game"], x["ply"]))
                rows.append(x)
    todo = [it for it in items if it not in done]
    t0 = time.perf_counter()
    for i in range(0, len(todo), chunk):
        part = todo[i:i + chunk]
        res = CE.solve_equals_batch([(variant, None, list(by_id[g]["moves"][:t])) for g, t in part], seconds, store=store, specs=specs, threads=threads, ledger_share=LEDGER_SHARE)
        with open(ckpt, "a") as f:
            for (g, t), r in zip(part, res):
                d = equal_set(r)
                x = {"game": g, "ply": t, "legal": d["legal"], "equals": len(d["equals"]), "cut": int(bool(d["cut"])), "proved": int(bool(d["proved"])), "depth": d["depth"], "error": d["error"],
                     "first": int(bool(d["equal_indices"]) and d["move"] in d["equals"] and d["equals"].index(d["move"]) == 0)}
                rows.append(x)
                f.write(json.dumps(x) + "\n")
        el = time.perf_counter() - t0
        say(f"OBS progress stage A: {len(done) + i + len(part)} of {len(items)} positions, {el:.0f} s since start of this run, estimated {el / (i + len(part)) * (len(todo) - i - len(part)):.0f} s left")
    return rows


def stage_a_lines(rows: list, games: int, seconds: float, threads: int) -> list:
    ok = [r for r in rows if r["error"] is None and r["legal"] >= 2]
    whole = [r for r in ok if not r["cut"]]
    many = [r for r in whole if r["equals"] > 1]
    lines = [f"OBS subgoal_play stage A: positions of {games} kept games read {len(rows)} (errors {sum(1 for r in rows if r['error'] is not None)}), with two or more legal moves {len(ok)}; the search at {seconds} s a move, ledger share {LEDGER_SHARE}, threads {threads}",
             f"OBS subgoal_play stage A: the clock cut the equal reading in {len(ok) - len(whole)} of the {len(ok)} positions ({100 * (len(ok) - len(whole)) / max(1, len(ok)):.1f} pct; those are left out below); of the {len(whole)} read whole, several root moves held equal to the best in {len(many)} "
             f"({100 * len(many) / max(1, len(whole)):.1f} pct); mean number of equal moves {np.mean([r['equals'] for r in whole]) if whole else float('nan'):.2f} over all, {np.mean([r['equals'] for r in many]) if many else float('nan'):.2f} over those with several; "
             f"the most {max((r['equals'] for r in whole), default=0)}",
             f"OBS subgoal_play stage A: decided by the line (proved, or the game's own outcome read, so no equal reading) {sum(r['proved'] for r in ok)} positions",
             f"OBS subgoal_play stage A: the search's own move is the lowest-index one among the equals in {sum(r['first'] for r in many)} of the {len(many)} positions with several equals ({100 * sum(r['first'] for r in many) / max(1, len(many)):.1f} pct)"]
    hist: dict = {}
    for r in whole:
        k = min(r["equals"], 10)
        hist[k] = hist.get(k, 0) + 1
    lines.append("OBS subgoal_play stage A: number of equal moves (10 stands for 10 or more) -> positions: " + ", ".join(f"{k}: {hist[k]}" for k in sorted(hist)))
    return lines


# --------------------------------------------------------------------------------------------------------------------------- stage D: the arrival of confirmed threat moves in the games

def arrival_rows(variant: str, games: list, threads: int, fuse: int) -> list:
    """For each game, the quiet positions' arrival (`forcing_arrival_measure`) and the confirmed-threat flag at every position, tagged by which player is to move there (a: the first player of the match)."""
    recs = [{"moves": g["moves"], "start": None} for g in games]
    reads = FO.read_chunk(variant, recs, threads, fuse)
    out = []
    for g, plies in zip(games, reads):
        own, oth = FA.counts_of_game(plies)
        quiet = set(FA.quiet_indices(plies, own, oth))
        rows = []
        for t in range(len(plies)):
            if plies[t] is None:
                continue
            a_moves = (t % 2) == g["colour"]
            hold = int(own[t][2] > 0)
            arr = None
            if t in quiet:
                tg = FA.arrival_targets(own, len(plies), t)[FA.HORIZONS.index(HORIZON)]
                arr = None if tg is None else tg[0]
            rows.append((a_moves, hold, arr))
        out.append(rows)
    return out


def arrival_lines(name: str, a: str, b: str, per_game: list, draws: int = BOOT_DRAWS) -> list:
    """The share of positions at which the side to move holds a CONFIRMED threat move, and, of the quiet positions with the horizon inside the game, the share at which its own CONFIRMED threat move arrives
    within HORIZON plies, for each player, with a 90 pct band by bootstrap over games."""
    lines = [f"OBS subgoal_play arrival {name}: games read {len(per_game)}, forcing read at fuse {FUSE}, horizon {HORIZON} plies"]
    rng = np.random.default_rng(BOOT_SEED)
    for who, flag in ((a, True), (b, False)):
        def stat(games: list) -> "tuple[float, float]":
            held = [h for g in games for (m, h, _x) in g if m == flag]
            arr = [x for g in games for (m, _h, x) in g if m == flag and x is not None]
            return (sum(held) / max(1, len(held)), sum(arr) / max(1, len(arr)))
        h0, a0 = stat(per_game)
        bs = np.array([stat([per_game[i] for i in rng.integers(0, len(per_game), len(per_game))]) for _ in range(draws)])
        lo, hi = np.percentile(bs, 5, axis=0), np.percentile(bs, 95, axis=0)
        held_n = sum(h for g in per_game for (m, h, _x) in g if m == flag)
        pos_n = sum(1 for g in per_game for (m, _h, _x) in g if m == flag)
        arr_n = sum(x for g in per_game for (m, _h, x) in g if m == flag and x is not None)
        q_n = sum(1 for g in per_game for (m, _h, x) in g if m == flag and x is not None)
        lines.append(f"OBS subgoal_play arrival {name}: {who} to move: a CONFIRMED threat move held at {held_n} of {pos_n} positions ({100 * h0:.1f} pct [{100 * lo[0]:.1f}, {100 * hi[0]:.1f}]); of {q_n} quiet positions with the horizon inside the game, "
                     f"its CONFIRMED threat move arrives within {HORIZON} plies in {arr_n} ({100 * a0:.1f} pct [{100 * lo[1]:.1f}, {100 * hi[1]:.1f}])")
    return lines


# --------------------------------------------------------------------------------------------------------------------------- the run

def read_games_file(path: "str | Path") -> list:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def emit(lines: list, out_file: "Path | None") -> None:
    for line in lines:
        print(line, flush=True)
    if out_file is not None:
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "a") as f:
            f.write("\n".join(lines) + "\n")


def main(argv: "list | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stage", required=True, choices=("fit", "match", "stage_a", "arrival", "report"))
    ap.add_argument("--variant", default=None, help="the game's name as the engine knows it (fit and stage_a read it from the games)")
    ap.add_argument("--name", default=None, choices=tuple(MATCHES), help="the match")
    ap.add_argument("--pairs", type=int, default=300)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 3))
    ap.add_argument("--seconds", type=float, default=CLOCK)
    ap.add_argument("--games-dir", default=None, help="fit and stage_a: the kept games (an absolute path)")
    ap.add_argument("--reads", default=None)
    ap.add_argument("--opp-table", default=None)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--tag", default="full", help="the directory under the experiment's own for this run")
    ap.add_argument("--sample", type=int, default=30, help="arrival: opening pairs (two games each) of the sample")
    ap.add_argument("--first", type=int, default=None, help="match: the first opening seed (default: the match's own)")
    a = ap.parse_args(argv)
    root = out_root() / a.tag
    threads = max(1, (os.cpu_count() or 2) - 3)
    weights = a.weights or str(root / f"weights_horizon{HORIZON}.json")
    obs = root / "obs.txt"
    if a.stage == "fit":
        lines, _doc = fit_weights(a.games_dir, a.reads, a.opp_table, weights, threads)
        emit(lines, obs)
    elif a.stage == "match":
        name = a.name
        t0 = time.perf_counter()
        games = run_match(name, a.variant, a.pairs, a.workers, a.seconds, weights, root / f"match_{name}", a.first)
        lines = match_lines(name, MATCHES[name][0], MATCHES[name][1], games)
        lines.append(f"OBS subgoal_play match {name}: workers {a.workers} (cores {os.cpu_count()}), clock {a.seconds} s a move for both sides, ledger share {LEDGER_SHARE}, wall {time.perf_counter() - t0:.0f} s in this run")
        emit(lines, obs)
    elif a.stage == "stage_a":
        records = sorted(FO.read_games(a.games_dir), key=lambda r: r["game"])
        t0 = time.perf_counter()
        rows = stage_a(records[0]["variant"], records, a.seconds, threads, root / "stage_a_rows.jsonl", 360)
        emit(stage_a_lines(rows, len(records), a.seconds, threads) + [f"OBS subgoal_play stage A: wall {time.perf_counter() - t0:.0f} s in this run"], obs)
    elif a.stage == "arrival":
        lines = []
        for name in ("candidate_v_control", "control_v_control"):
            games = read_games_file(root / f"match_{name}" / "games.jsonl")
            seeds = sorted({g["seed"] for g in games})[:a.sample]
            sample = [g for g in games if g["seed"] in seeds]
            per_game = arrival_rows(a.variant, sample, threads, FUSE)
            lines += arrival_lines(name, MATCHES[name][0], MATCHES[name][1], per_game)
        emit(lines, obs)
    else:
        lines = []
        for name, (pa, pb, _s) in MATCHES.items():
            f = root / f"match_{name}" / "games.jsonl"
            if f.exists():
                lines += match_lines(name, pa, pb, read_games_file(f))
        emit(lines, None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
