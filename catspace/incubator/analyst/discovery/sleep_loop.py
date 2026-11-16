"""PLAN 78.63: A SLEEP THAT CHOOSES ITS ACTIONS AND DECIDES WHEN TO WAKE (SYSTEM_SPEC 16). A sandbox: everything is written under artifacts/experiments/sleep_78_63/ through the path registry; nothing is
written to the real store, nothing is wired into any player of the wake. Game-free: kinds, sides and places are words read from the graph at run time; no object of any game is named here.
No engine evaluation, no labeller number and no outside call enters: only the results of games (human games, and the games this sleep plays) and what happened on the board.

THE ACTIONS (one interface: name, what it expects to improve, run for at most a number of seconds, resumable because every game is appended to a log as it ends):
  odds_kind(k)     odds games from the ordinary start, one side short of a unit of kind k (every start of that kind), the even start as the control      -> the term of k
  odds_pair(a, b)  the same for the two kinds of a pair the data cannot yet order                                                                         -> the order of that pair
  odds_later(k)    odds games from positions sampled from the fit human games (two random plies first), a unit of kind k taken from each side in turn      -> the term of k where it is in play
  human_fit        the result head (ratings and kinds) fitted on a further slice of the fit human games                                                   -> the held-out log loss of the result head
THE YARDSTICK (fixed before the loop, never fitted on): (1) the log loss of the human result head on the held-out human games (the 78.59 test games); (2) the log loss of the odds head (a multinomial logit
of the first mover's result on the start's unit-count differences) on the held-out half of the previous generation's odds games; (3) the pairs of kinds left unordered (95 pct level, by refits on resampled games,
printed as a count; the scalar uses the soft share of all pairs, 0 to 1). The scalar Y is the mean of the two log losses, each relative to its value at the start of the first sleep, and that share (smaller is better).
THE CHOOSER: for each action a gain on Y per second COMPUTED before it runs (a learning curve of the held-out loss against the games used; the bands of the pair it would separate), corrected by the running
record of observed against computed for its type; a stated share of trials goes to the type tried longest ago. THE STOP: the time budget, the disk limit, no action expected to pay, or no recent gain observed.
THE WAKE TEST: a match of the new store's player against the previous one's; the new store is kept unless it is clearly worse.

  .venv/bin/python -m catspace.incubator.analyst.discovery.sleep_loop sleeps --tag T --sleeps N --sleep-seconds S --slice-seconds L
  .venv/bin/python -m catspace.incubator.analyst.discovery.sleep_loop fixed  --tag T --sleep-seconds S --slice-seconds L
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np

import cs_engine as CE

from catspace.incubator.analyst.discovery import estimator_fit as EF
from catspace.incubator.analyst.discovery import human_heads_data as HD
from catspace.incubator.analyst.discovery import human_heads_result as HR
from catspace.incubator.analyst.discovery import odds_measure as O
from catspace.io import paths

C_HEAD = 1.0                          # the ridge strength of both heads, fixed in advance
HUMAN_START = 500                     # fit human games the human head starts from
HUMAN_STEP = 500                      # fit human games one human_fit action adds
LATER_PLIES = (12, 40)                # the even plies a later start may be sampled at
LATER_GAMES = 6                       # games per start of a sampled position
LATER_OPENING = 2                     # random plies before play from a sampled position
DRAWS = 200                           # refits on resampled games
Z_ORDERED = 1.645                     # one-sided 95 pct
SOFT_LOW, SOFT_HIGH = 0.05, 0.5       # a pair is wholly ordered at a share of draws <= 0.05 (or >= 0.95), wholly unordered at 0.5
EXPLORE = 0.2                         # share of choices given to the type tried longest ago
PATIENCE_SLICES = 6                   # no new best of Y, by THETA a second, over this many slice lengths wakes the sleep
THETA = 5e-7                          # gain on Y per second under which the sleep wakes
RATE_PRIOR = 1.9                      # games a second at 8 workers: 78.61 played 1860 games in 952 s
DISK_GB = 10.0                        # the sleep wakes when less than this is free
SANDBOX_GB = 3.0                      # ... or when the sandbox holds more than this
ACTION_LIMIT = 3600.0                 # seconds: no action runs longer (the sleep's own scale is passed in the config)
MIN_SLICE_SHARE = 0.25                # a slice shorter than this share of the slice length is not started
WAKE_LEVEL_NOTE = "kept unless the upper end of the 90 pct band of the match score is under 0.5 (clearly worse)"
SEED = 7863


def sandbox(tag: str) -> Path:
    return Path(paths.experiment("sleep_78_63", tag))


def md5i(text: str) -> int:
    return int(hashlib.md5(text.encode()).hexdigest(), 16)


def free_gb(where: "Path | None" = None) -> float:
    st = os.statvfs(str(where or paths.REPO_ROOT))
    return st.f_bavail * st.f_frsize / 1e9


def dir_gb(p: Path) -> float:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e9 if p.exists() else 0.0


def append_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(obj) + "\n")
        f.flush()


def read_json_lines(path: Path) -> list:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []


def phi(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def p_ordered(d: float, s: float) -> float:
    """The chance that an estimate of a difference `d` with spread `s` will be called ordered (one-sided 95 pct each way)."""
    s = max(s, 1e-9)
    return phi(abs(d) / s - Z_ORDERED) + phi(-abs(d) / s - Z_ORDERED)


def soft_unordered(f: float) -> float:
    """0 when the pair is ordered at the 95 pct level, 1 when the draws split evenly, linear between."""
    m = min(f, 1.0 - f)
    return float(min(1.0, max(0.0, (m - SOFT_LOW) / (SOFT_HIGH - SOFT_LOW))))


def curve_slope(points: list) -> float:
    """b of loss = a + b / n through the last points (at least two distinct n); 0 when there is no usable slope."""
    pts = [(n, v) for n, v in points[-4:] if n > 0 and np.isfinite(v)]
    if len({n for n, _ in pts}) < 2:
        return 0.0
    a = np.array([[1.0, 1.0 / n] for n, _ in pts])
    b = np.array([v for _, v in pts])
    coef = np.linalg.lstsq(a, b, rcond=None)[0]
    return float(max(coef[1], 0.0))


# ------------------------------------------------------------------------------------------------------------------ the fixed yardstick sets

class Fixed:
    """The held-out sets and the order of the fit games, made once before any loop. Human: the rows of the 78.59 sample; the games its own split calls test are held out (never fitted on), the games it calls fit are
    taken in a fixed order by a hash; the inputs are the ratings (mean, difference, ply) and the kinds ours less theirs, scaled on the fit side only."""

    def __init__(self, rows: "dict | None" = None):
        self.rows = rows if rows is not None else HD.load_rows(Path(paths.experiment("estimator_78_59", "full", "rows.npz")))
        cols, _meta = HR.raw_columns(self.rows)
        x, names = HR.stack(cols, "RK")
        fit_rows = np.flatnonzero(self.rows["fit"])
        centre, scale = HR.transform_of(x, names, fit_rows)
        self.z = (x - centre) / scale
        self.y = self.rows["y"].astype(int)
        game = self.rows["game"]
        self.test_rows = np.flatnonzero(~self.rows["fit"])
        self.test_games = sorted(set(game[self.test_rows].tolist()))
        order = np.argsort(game, kind="stable")
        sg = game[order]
        cuts = np.flatnonzero(np.diff(sg)) + 1
        self.rows_of = {int(sg[a]): order[a:b] for a, b in zip(np.concatenate([[0], cuts]), np.concatenate([cuts, [len(sg)]]))}
        fit_games = sorted(set(game[self.rows["fit"]].tolist()))
        self.fit_order = sorted(fit_games, key=lambda g: md5i(f"{g}:{SEED}:order"))
        self.kinds_human = self.rows["meta"]["kinds"]

    def fit_rows(self, n: int) -> np.ndarray:
        return np.concatenate([self.rows_of[g] for g in self.fit_order[:n]])

    def fit_head(self, n: int):
        r = self.fit_rows(n)
        return EF.fit_lr(self.z[r], self.y[r], C_HEAD)

    def human_loss(self, model) -> float:
        return EF.logloss(EF.probs(model, self.z[self.test_rows]), self.y[self.test_rows])


def held_out_odds(gid: str) -> bool:
    """Half of the previous generation's odds games are held out, by a hash of the game's name."""
    return md5i(f"{gid}:{SEED}:heldout") % 2 == 0


# ------------------------------------------------------------------------------------------------------------------ memory of odds games

class Memory:
    """The odds games and the starts they were played from. `kd` of a start: first mover's units less the other's, by kind (the goal kind, which cannot be removed, is left out)."""

    def __init__(self, kinds: list, first_side: str):
        self.kinds, self.first = kinds, first_side
        self.registry: dict = {}
        self.games: list = []                      # dicts of play_game, each with a `y` class
        self.sources: list = []

    def kd_of(self, text: str) -> list:
        d = [0] * len(self.kinds)
        for side, kind, _p in O.units_of(text):
            if kind in self.kinds:
                d[self.kinds.index(kind)] += 1 if side == self.first else -1
        return d

    def add_start(self, sid: str, s: dict) -> None:
        if sid not in self.registry:
            self.registry[sid] = {"text": s["text"], "kind": s.get("kind"), "side": s.get("side"), "n": s.get("n", 0), "kd": s.get("kd") or self.kd_of(s["text"])}

    def add_game(self, g: dict, skip=None) -> None:
        if skip is not None and skip(g["gid"]):
            return
        sc = O.white_score(g)
        self.games.append({"gid": g["gid"], "start": g["start"], "y": 0 if sc == 1.0 else (1 if sc == 0.5 else 2), "unfinished": bool(g.get("unfinished"))})

    def arrays(self, order_key=None) -> tuple:
        gs = self.games if order_key is None else sorted(self.games, key=order_key)
        x = np.array([self.registry[g["start"]]["kd"] for g in gs], dtype=float).reshape(len(gs), len(self.kinds))
        return x, np.array([g["y"] for g in gs], dtype=int), [g["start"] for g in gs], [g["gid"] for g in gs]


def previous_generation_games(root_61: Path, gen: int, made: dict, memory: Memory) -> "tuple[list, list]":
    """(held-out games, kept games) of an earlier plan's generation `gen` (the valid directory), by the hash split; its starts are added to the registry."""
    for sid, s in made["starts"].items():
        memory.add_start(sid, s)
    held, kept = [], []
    for g in read_json_lines(root_61 / f"gen{gen}" / "games.jsonl"):
        (held if held_out_odds(g["gid"]) else kept).append(g)
    return held, kept


def fit_odds(x: np.ndarray, y: np.ndarray):
    if len(np.unique(y)) < 2:
        return None
    return EF.fit_lr(x, y, C_HEAD)


def odds_probs(m, x: np.ndarray, y_fit: np.ndarray) -> np.ndarray:
    return EF.base_probs(y_fit, len(x)) if m is None else EF.probs(m, x)


def kind_terms_of(m, nk: int) -> np.ndarray:
    if m is None:
        return np.zeros(nk)
    if m.coef_.shape[0] == 1:                      # two classes: one row, the log odds of the second class against the first, so a win less a loss is its negative
        return -m.coef_[0].copy()
    return EF.kind_terms(list(m.classes_), m.coef_)


def bootstrap_terms(x: np.ndarray, y: np.ndarray, starts: list, draws: int, seed: int) -> np.ndarray:
    """(draws, kinds) kind terms of refits on games resampled within each start."""
    groups: dict = {}
    for i, s in enumerate(starts):
        groups.setdefault(s, []).append(i)
    glist = [np.array(v) for v in groups.values()]
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(draws):
        idx = np.concatenate([g[rng.integers(0, len(g), len(g))] for g in glist])
        out.append(kind_terms_of(fit_odds(x[idx], y[idx]), x.shape[1]))
    return np.array(out)


def pair_table(draws: np.ndarray, kinds: list) -> list:
    out = []
    for i in range(len(kinds)):
        for j in range(i + 1, len(kinds)):
            f = float((draws[:, i] > draws[:, j]).mean())
            out.append({"a": kinds[i], "b": kinds[j], "share": f, "unordered": 0.05 < f < 0.95, "soft": soft_unordered(f), "spread": float((draws[:, i] - draws[:, j]).std())})
    return out


# ------------------------------------------------------------------------------------------------------------------ playing

def play_until(pool, next_spec, deadline: float, out_file: Path, workers: int, on_game=None) -> list:
    """Play games from `next_spec()` until `deadline` (a time.time() value): none is started that is expected to end later than the deadline, those in flight are waited for, each game is appended to
    `out_file` as it ends. `pool` None plays in this process one at a time."""
    done, flying, durs = [], [], []

    def keep(g: dict) -> None:
        done.append(g)
        append_json(out_file, g)
        if on_game:
            on_game(g)
    while True:
        est = float(np.mean(durs)) if durs else 0.0
        can = time.time() + est < deadline
        while can and len(flying) < max(1, workers):
            spec = next_spec()
            if spec is None:
                can = False
                break
            if pool is None:
                t0 = time.time()
                keep(O.play_game(spec))
                durs.append(time.time() - t0)
            else:
                flying.append((time.time(), pool.apply_async(O.play_game, (spec,))))
            est = float(np.mean(durs)) if durs else 0.0
            can = time.time() + est < deadline
        for item in [f for f in flying if f[1].ready()]:
            flying.remove(item)
            durs.append(time.time() - item[0])
            keep(item[1].get())
        if not flying and not can:
            break
        if pool is not None:
            time.sleep(0.02)
    return done


def player_from_values(values: dict) -> dict:
    """A Worth player from integer worths by the graph's kind word (mapped to the engine's symbols as 78.61 does)."""
    keys = O.kernel_names()
    missing = [k for k in values if k not in keys]
    if missing:
        raise ValueError(f"kinds the engine does not know: {missing}")
    return {"kind": "worth", "worth": {keys[k]: v for k, v in values.items()}, "record_kinds": dict(values)}


def run_specs(pool, specs: list, out_file: Path) -> list:
    done = {g["gid"]: g for g in read_json_lines(out_file)}
    todo = [s for s in specs if s["gid"] not in done]
    if pool is None:
        for s in todo:
            g = O.play_game(s)
            done[g["gid"]] = g
            append_json(out_file, g)
    else:
        for s, r in [(s, pool.apply_async(O.play_game, (s,))) for s in todo]:
            g = r.get()
            done[g["gid"]] = g
            append_json(out_file, g)
    return [done[s["gid"]] for s in specs]


def match(pool, label: str, a: dict, b: dict, pairs: int, cfg: dict, out_dir: Path, say) -> dict:
    """A match of player `a` against player `b` over `pairs` openings, colours alternated; the line printed is the 78.61 line (load average and cores busy at the instant of the start included)."""
    specs = []
    for k in range(pairs):
        seed = O.MATCH_FIRST_SEED + k
        opening = O.random_opening(None, seed)
        for colour in (0, 1):
            specs.append({"gid": f"{label}:{seed}:{colour}", "start": f"{seed}", "colour": colour, "text": None, "opening": opening, "players": [a, b] if colour == 0 else [b, a],
                          "seconds": cfg["seconds"], "max_plies": cfg["max_plies"], "scratch": str(out_dir / "scratch")})
    load, busy, t0 = os.getloadavg()[0], O.cpu_busy(), time.perf_counter()
    games = run_specs(pool, specs, out_dir / f"{label}.jsonl")
    sc = np.array([O.white_score(g) if g["colour"] == 0 else 1 - O.white_score(g) for g in games])
    w, d, l = int((sc == 1).sum()), int((sc == 0.5).sum()), int((sc == 0).sum())
    pr = O.pair_scores(games)
    s, lo, hi = O.score_band(pr)
    unf = sum(g["unfinished"] for g in games)
    say(f"OBS sleep match {label}: games {len(games)} ({len(pr)} opening pairs, colours alternated): first player W {w} D {d} L {l}; score {s:.4f} [{lo:.4f}, {hi:.4f}] 90 pct band by bootstrap over opening pairs; unfinished {unf} (counted as draws); "
        f"load average (1 min) at start {load:.2f}, cores busy at the instant of start {busy:.2f}; wall {time.perf_counter() - t0:.0f} s; seconds per move {cfg['seconds']}, ply cap {cfg['max_plies']}, workers {cfg['workers']}")
    return {"label": label, "score": s, "lo": lo, "hi": hi, "w": w, "d": d, "l": l}


def wake_decision(result: dict) -> "tuple[bool, str]":
    """The new store is kept unless the match says it is clearly worse: the upper end of the 90 pct band of its score under one half."""
    keep = not result["hi"] < 0.5
    return keep, WAKE_LEVEL_NOTE + f"; upper end {result['hi']:.4f}"


# ------------------------------------------------------------------------------------------------------------------ the actions

class Action:
    type = ""

    def name(self) -> str:
        raise NotImplementedError

    def expects(self) -> str:
        raise NotImplementedError

    def run(self, ctx: "Sleep", seconds: float) -> dict:
        raise NotImplementedError


class OddsAction(Action):
    """Odds games for a cycle of starts; every game is appended to the sleep's log as it ends, the count per start is read back from the log, so a cut or a crash keeps all that was played."""

    def start_cycle(self, ctx: "Sleep") -> list:
        raise NotImplementedError

    def run(self, ctx: "Sleep", seconds: float) -> dict:
        cyc = self.start_cycle(ctx)
        pos = [0]

        def next_spec():
            sid = cyc[pos[0] % len(cyc)]
            pos[0] += 1
            return ctx.spec_for(sid)
        t0 = time.time()
        games = play_until(ctx.pool, next_spec, t0 + seconds, ctx.dir / "games.jsonl", ctx.cfg["workers"], ctx.on_game)
        return {"games": len(games), "wall": time.time() - t0}


class OddsKind(OddsAction):
    type = "odds_kind"

    def __init__(self, kind: str):
        self.kind = kind

    def name(self) -> str:
        return f"odds_kind({self.kind})"

    def expects(self) -> str:
        return f"the term of {self.kind} (narrower band, so its pairs order)"

    def kinds_touched(self) -> list:
        return [self.kind]

    def start_cycle(self, ctx: "Sleep") -> list:
        mine = [sid for sid, s in ctx.base_starts.items() if s["kind"] == self.kind]
        return mine + ["even"]


class OddsPair(OddsKind):
    type = "odds_pair"

    def __init__(self, a: str, b: str):
        self.a, self.b, self.kind = a, b, a

    def name(self) -> str:
        return f"odds_pair({self.a}, {self.b})"

    def expects(self) -> str:
        return f"the order of {self.a} and {self.b}"

    def kinds_touched(self) -> list:
        return [self.a, self.b]

    def start_cycle(self, ctx: "Sleep") -> list:
        mine = [sid for sid, s in ctx.base_starts.items() if s["kind"] in (self.a, self.b)]
        return mine + ["even"]


class OddsLater(OddsAction):
    type = "odds_later"

    def __init__(self, kind: str):
        self.kind = kind

    def name(self) -> str:
        return f"odds_later({self.kind})"

    def expects(self) -> str:
        return f"the term of {self.kind} measured where it is in play (positions from human games)"

    def kinds_touched(self) -> list:
        return [self.kind]

    def start_cycle(self, ctx: "Sleep") -> list:
        raise NotImplementedError

    def run(self, ctx: "Sleep", seconds: float) -> dict:
        queue: list = []

        def next_spec():
            while not queue:
                made = ctx.sample_later(self.kind)
                for sid in made:
                    queue.extend([sid] * LATER_GAMES)
            return ctx.spec_for(queue.pop(0), opening_plies=LATER_OPENING)
        t0 = time.time()
        games = play_until(ctx.pool, next_spec, t0 + seconds, ctx.dir / "games.jsonl", ctx.cfg["workers"], ctx.on_game)
        return {"games": len(games), "wall": time.time() - t0}


class HumanFit(Action):
    type = "human_fit"

    def name(self) -> str:
        return "human_fit"

    def expects(self) -> str:
        return "the held-out log loss of the result head (ratings and kinds) on human games"

    def run(self, ctx: "Sleep", seconds: float) -> dict:
        t0 = time.time()
        per_game = ctx.human_seconds / max(1, ctx.human_n)
        m = int(min(HUMAN_STEP, len(ctx.fixed.fit_order) - ctx.human_n, max(0.0, seconds / max(per_game, 1e-9))))
        if m < 50 and len(ctx.fixed.fit_order) - ctx.human_n >= 50:
            return {"games": 0, "wall": 0.0, "note": "the slice is too short for a fit"}
        if m <= 0:
            return {"games": 0, "wall": 0.0}
        ctx.set_human(ctx.human_n + m)
        return {"games": m, "wall": time.time() - t0}


# ------------------------------------------------------------------------------------------------------------------ the sleep

class Sleep:
    """One sleep: the loaded store, the held-out sets, the actions, the chooser, the stop rule. `pool` None plays in this process (tests)."""

    def __init__(self, tag: str, index: int, cfg: dict, fixed: Fixed, pool, say=print, store: "dict | None" = None):
        self.tag, self.index, self.cfg, self.fixed, self.pool, self.say = tag, index, cfg, fixed, pool, say
        self.root = sandbox(tag)
        self.dir = self.root / f"sleep{index}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.root_61 = Path(paths.experiment("odds_78_61", "full"))
        self.made = O.make_starts(60)
        self.kinds = sorted({s["kind"] for s in self.made["starts"].values() if s["kind"]})
        self.first = self.made["sides"][0]
        self.base_starts = self.made["starts"]
        self.store = store or self.initial_store()
        self.memory = Memory(self.kinds, self.first)
        self.held_x = self.held_y = None
        self.load_memory()
        self.player = self.store["player"]
        self.later_next = 0
        self.counts: dict = {}
        self.new_games = 0
        self.human_n = self.store.get("human_n", HUMAN_START)
        self.human_seconds = 1.0
        self.human_model = None
        self.hl_curve: list = []
        self.ol_curve: list = []
        self.rate = RATE_PRIOR
        self.record = self.load_chooser()
        self.history: list = []
        self.last_tried: dict = {}
        self.measure_cache: "dict | None" = None
        self.resume_state()

    # ---- the store
    def initial_store(self) -> dict:
        gen = self.cfg.get("start_gen", 2)
        rec = json.loads((self.root_61 / f"gen{gen}" / "record.json").read_text())
        player = {"kind": "nothing", "record_kinds": {}} if gen == 0 else player_from_values(rec["heads"]["odds_effect"]["kind_values"])      # generation zero counts nothing: it is the player its record came from
        return {"record": rec, "player": player, "from": f"odds_78_61 generation {gen}", "memory": [], "human_n": HUMAN_START, "sleep": 0}

    def load_memory(self) -> None:
        held, kept = previous_generation_games(self.root_61, self.cfg.get("start_gen", 2), self.made, self.memory)
        for g in kept:
            self.memory.add_game(g)
        hm = Memory(self.kinds, self.first)
        for sid, s in self.made["starts"].items():
            hm.add_start(sid, s)
        for g in held:
            hm.add_game(g)
        self.held_x, self.held_y, _s, self.held_gids = hm.arrays()
        for src in self.store["memory"]:
            reg = json.loads(Path(src["starts"]).read_text())
            for sid, s in reg.items():
                self.memory.add_start(sid, s)
            for g in read_json_lines(Path(src["games"])):
                self.memory.add_game(g)
        self.memory_games_at_start = len(self.memory.games)

    def load_chooser(self) -> dict:
        p = self.root / "store" / "chooser_record.json"
        if p.exists():
            return json.loads(p.read_text())
        return {"types": {}, "rate": RATE_PRIOR}

    def save_chooser(self) -> None:
        d = self.root / "store"
        d.mkdir(parents=True, exist_ok=True)
        self.record["rate"] = self.rate
        (d / "chooser_record.json").write_text(json.dumps(self.record, indent=1))

    def resume_state(self) -> None:
        reg_path = self.dir / "starts.json"
        if reg_path.exists():
            for sid, s in json.loads(reg_path.read_text()).items():
                self.memory.add_start(sid, s)
        for g in read_json_lines(self.dir / "games.jsonl"):
            self.memory.add_game(g)
            sid, k = g["gid"].split(":", 1)[1].rsplit(":", 1)
            self.counts[sid] = max(self.counts.get(sid, 0), int(k) + 1)
            self.new_games += 1
        st = self.dir / "state.json"
        if st.exists():
            s = json.loads(st.read_text())
            self.later_next = s.get("later_next", 0)
            self.human_n = s.get("human_n", self.human_n)
        self.history = read_json_lines(self.dir / "log.jsonl")

    def save_state(self) -> None:
        (self.dir / "state.json").write_text(json.dumps({"later_next": self.later_next, "human_n": self.human_n}))
        (self.dir / "starts.json").write_text(json.dumps({sid: s for sid, s in self.memory.registry.items() if sid.startswith(("later:",))}))

    # ---- the held-out sets, stated
    def write_yardstick(self) -> dict:
        d = {"human_held_out_games": len(self.fixed.test_games), "human_held_out_rows": int(len(self.fixed.test_rows)), "human_fit_games_available": len(self.fixed.fit_order),
             "odds_held_out_games": len(self.held_gids), "odds_fit_games_at_start": self.memory_games_at_start,
             "human_held_out_digest": hashlib.md5(json.dumps(self.fixed.test_games).encode()).hexdigest(), "odds_held_out_digest": hashlib.md5(json.dumps(sorted(self.held_gids)).encode()).hexdigest()}
        (self.root / "yardstick.json").write_text(json.dumps(d, indent=1))
        return d

    # ---- games
    def spec_for(self, sid: str, opening_plies: int = O.OPENING_PLIES) -> dict:
        s = self.memory.registry.get(sid) or self.base_starts[sid]
        k = self.counts.get(sid, 0)
        self.counts[sid] = k + 1
        seed = SEED + 100000 * (self.index + 1) + 1000 * (O.hash_of(sid) % 997) + k
        return {"gid": f"s{self.index}:{sid}:{k}", "start": sid, "text": s["text"], "opening": O.random_opening(s["text"], seed, opening_plies), "players": [self.player, self.player],
                "seconds": self.cfg["seconds"], "max_plies": self.cfg["max_plies"], "scratch": str(self.dir / "scratch")}

    def on_game(self, g: dict) -> None:
        self.memory.add_game(g)
        self.new_games += 1

    def ensure_base_starts(self) -> None:
        for sid, s in self.base_starts.items():
            self.memory.add_start(sid, s)

    def sample_later(self, kind: str) -> list:
        """The next position of the fixed order of fit human games that has a unit of `kind` on each side and is playable; -> the ids of its three starts (the position as it is, one unit taken from each side)."""
        while True:
            gi = self.fixed.fit_order[self.later_next % len(self.fixed.fit_order)]
            self.later_next += 1
            game = self.games_of_human(gi)
            ply = LATER_PLIES[0] + 2 * (md5i(f"{gi}:{SEED}:ply") % ((LATER_PLIES[1] - LATER_PLIES[0]) // 2 + 1))
            if ply >= len(game):
                continue
            text = CE.state(O.VARIANT, None, game[:ply])[0]
            units = O.units_of(text)
            sides = sorted({u[0] for u in units}, key=lambda x: (x != self.first, x))
            ones = {sd: sorted(u[2] for u in units if u[0] == sd and u[1] == kind) for sd in sides}
            if not all(ones.values()):
                continue
            control = O.board_without(text, [])
            if not O.playable(control):
                continue
            ids, ok = [], True
            base = f"later:{gi}:{ply}:{kind}"
            starts = {f"{base}:even": {"text": control, "kind": None, "side": None, "n": 0}}
            for i, sd in enumerate(sides):
                place = ones[sd][md5i(f"{gi}:{ply}:{sd}:{SEED}") % len(ones[sd])]
                t = O.board_without(text, [place])
                if not O.playable(t):
                    ok = False
                    break
                starts[f"{base}:{i}:1"] = {"text": t, "kind": kind, "side": i, "n": 1}
            if not ok:
                continue
            for sid, s in starts.items():
                self.memory.add_start(sid, s)
                ids.append(sid)
            self.save_state()
            return ids

    def games_of_human(self, gi: int) -> list:
        if not hasattr(self, "_human_games"):
            self._human_games = [json.loads(x)["moves"] for x in (Path(paths.experiment("estimator_78_59", "full", "games.jsonl"))).read_text().splitlines() if x.strip()]
        return self._human_games[gi]

    # ---- the human head
    def set_human(self, n: int) -> None:
        t0 = time.time()
        self.human_model = self.fixed.fit_head(n)
        self.human_seconds = time.time() - t0
        self.human_n = n
        self.hl = self.fixed.human_loss(self.human_model)
        self.hl_curve.append((n, self.hl))
        self.measure_cache = None

    # ---- the yardstick
    def measure(self) -> dict:
        if self.measure_cache is not None:
            return self.measure_cache
        x, y, starts, _g = self.memory.arrays()
        m = fit_odds(x, y)
        ol = EF.logloss(odds_probs(m, self.held_x, y), self.held_y)
        draws = bootstrap_terms(x, y, starts, DRAWS, SEED)
        pairs = pair_table(draws, self.kinds)
        npairs = max(1, len(pairs))
        out = {"hl": self.hl, "ol": ol, "unordered": sum(p["unordered"] for p in pairs), "soft": sum(p["soft"] for p in pairs) / npairs, "pairs": pairs, "terms": kind_terms_of(m, len(self.kinds)), "draws": draws,
               "n_odds": len(y), "n_kind": [int((x[:, i] != 0).sum()) for i in range(len(self.kinds))]}
        base = self.baseline()
        out["Y"] = (out["hl"] / base["hl"] + out["ol"] / base["ol"] + out["soft"]) / 3.0
        self.measure_cache = out
        return out

    def baseline(self) -> dict:
        p = self.root / "baseline.json"
        if not p.exists():
            m = self.measure_plain()
            p.write_text(json.dumps({"hl": m["hl"], "ol": m["ol"]}))
        return json.loads(p.read_text())

    def measure_plain(self) -> dict:
        x, y, starts, _g = self.memory.arrays()
        m = fit_odds(x, y)
        ol = EF.logloss(odds_probs(m, self.held_x, y), self.held_y)
        pairs = pair_table(bootstrap_terms(x, y, starts, DRAWS, SEED), self.kinds)
        return {"hl": self.hl, "ol": ol, "soft": sum(p["soft"] for p in pairs) / max(1, len(pairs))}

    def describe(self, m: dict) -> str:
        return f"Y {m['Y']:.4f} (human log loss {m['hl']:.5f} on {self.human_n} fit games; odds log loss {m['ol']:.5f} on {len(self.held_y)} held-out games, {m['n_odds']} in memory; kinds unordered {m['unordered']} of {len(m['pairs'])}, soft {m['soft']:.3f})"

    def prepare(self) -> None:
        """Before the clock: the human head at the start, its learning curve (a head at half the games and at all of them), the odds head's curve the same way."""
        t0 = time.time()
        n = self.human_n
        self.set_human(max(n // 2, 50))
        self.set_human(n)
        self.ensure_base_starts()
        gs = sorted(self.memory.games, key=lambda g: md5i(f"{g['gid']}:{SEED}:curve"))
        keep = self.memory.games
        self.memory.games = gs[:len(gs) // 2]
        ol_half = self.measure_plain_loss()
        self.memory.games = keep
        self.measure_cache = None
        self.ol_curve = [(len(gs) // 2, ol_half)]
        m = self.measure()
        self.ol_curve.append((m["n_odds"], m["ol"]))
        self.say(f"OBS sleep {self.index} prepared in {time.time() - t0:.1f} s: {self.describe(m)}; the curves: human {self.hl_curve}, odds {[(n, round(v, 5)) for n, v in self.ol_curve]}")

    def measure_plain_loss(self) -> float:
        x, y, _s, _g = self.memory.arrays()
        return EF.logloss(odds_probs(fit_odds(x, y), self.held_x, y), self.held_y)

    # ---- the chooser
    def candidates(self, m: dict) -> list:
        out: list = []
        if self.human_n < len(self.fixed.fit_order):
            out.append(HumanFit())
        for k in self.kinds:
            out.append(OddsKind(k))
            out.append(OddsLater(k))
        for p in m["pairs"]:
            if p["unordered"]:
                out.append(OddsPair(p["a"], p["b"]))
        return out

    def computed_gain(self, action: Action, seconds: float, m: dict) -> dict:
        """The expected gain on Y of the action, a number made before it runs: per second and in all."""
        base = self.baseline()
        if isinstance(action, HumanFit):
            n, step = self.human_n, min(HUMAN_STEP, len(self.fixed.fit_order) - self.human_n)
            secs = self.human_seconds * (n + step) / max(n, 1)
            g = curve_slope(self.hl_curve) * (1.0 / n - 1.0 / (n + step)) / base["hl"] / 3.0
            return {"total": g, "seconds": secs, "per_second": g / max(secs, 1e-9), "parts": {"human": g}}
        games = self.rate * seconds
        n = max(m["n_odds"], 1)
        g_ol = curve_slope(self.ol_curve) * (1.0 / n - 1.0 / (n + games)) / base["ol"] / 3.0
        touched = action.kinds_touched()
        share = games / len(touched)
        idx = {k: i for i, k in enumerate(self.kinds)}
        draws = m["draws"]
        dp = 0.0
        for p in m["pairs"]:
            a, b = idx[p["a"]], idx[p["b"]]
            if p["a"] not in touched and p["b"] not in touched:
                continue
            var = float((draws[:, a] - draws[:, b]).var())
            fut = var
            for k in (a, b):
                if self.kinds[k] in touched:
                    vk = float(draws[:, k].var())
                    nk = max(m["n_kind"][k], 1)
                    fut -= vk * (1.0 - nk / (nk + share))
            d = float(m["terms"][a] - m["terms"][b])
            dp += p_ordered(d, math.sqrt(max(fut, 1e-12))) - p_ordered(d, math.sqrt(max(var, 1e-12)))
        g_us = (dp / max(1, len(m["pairs"]))) / 3.0
        tot = g_ol + g_us
        return {"total": tot, "seconds": seconds, "per_second": tot / seconds, "parts": {"odds_loss": g_ol, "order": g_us}}

    def correction(self, typ: str) -> float:
        t = self.record["types"].get(typ)
        if not t or t["tries"] < 2 or t["computed"] <= 0:
            return 1.0
        return float(min(10.0, max(0.1, t["observed"] / t["computed"])))

    def choose(self, seconds: float, step: int) -> "tuple[Action | None, dict]":
        m = self.measure()
        rows = []
        for a in self.candidates(m):
            g = self.computed_gain(a, seconds, m)
            rows.append((a, g, g["per_second"] * self.correction(a.type)))
        info = {"best_corrected": max((r[2] for r in rows), default=0.0), "ranked": sorted([(r[0].name(), r[1]["per_second"], r[2]) for r in rows], key=lambda x: -x[2])[:5]}
        if not rows:
            return None, info
        rng = random.Random(f"{SEED}:{self.index}:{step}")
        explore = rng.random() < EXPLORE
        if explore:
            types = sorted({r[0].type for r in rows}, key=lambda t: (self.last_tried.get(t, -1), t))
            pick = max((r for r in rows if r[0].type == types[0]), key=lambda r: r[2])
        else:
            pick = max(rows, key=lambda r: r[2])
        info["explore"] = explore
        info["computed"], info["corrected"] = pick[1], pick[2]
        return pick[0], info

    def y_track(self) -> list:
        h = [x for x in self.history if "Y" in x]
        return ([(0.0, h[0]["Y_before"])] if h else []) + [(x["elapsed"], x["Y"]) for x in h]

    def stop_reason(self, elapsed: float, info: dict) -> "str | None":
        return stop_rule(elapsed, self.cfg["sleep_seconds"], self.cfg["slice_seconds"], free_gb(), dir_gb(self.root), info["best_corrected"] if info else 0.0, self.y_track())

    # ---- the loop
    def run(self) -> dict:
        say = self.say
        elapsed0 = sum(h["seconds"] for h in self.history)
        t_start = time.time() - elapsed0
        self.prepare()
        step = len(self.history)
        reason = None
        first = self.measure()
        say(f"OBS sleep {self.index} start: {self.describe(first)}; the sleep lasts at most {self.cfg['sleep_seconds']} s in slices of {self.cfg['slice_seconds']} s; at most {ACTION_LIMIT:.0f} s an action; player: {json.dumps(self.player['record_kinds'])}")
        while True:
            elapsed = time.time() - t_start
            seconds = min(self.cfg["slice_seconds"], ACTION_LIMIT, self.cfg["sleep_seconds"] - elapsed)
            action, info = self.choose(self.cfg["slice_seconds"], step)
            reason = self.stop_reason(elapsed, info) if action is not None else "no action to choose"
            if reason:
                break
            before = self.measure()
            t0 = time.time()
            res = action.run(self, seconds)
            self.measure_cache = None
            if isinstance(action, OddsAction) and res["games"]:
                self.rate = 0.5 * self.rate + 0.5 * res["games"] / max(res["wall"], 1e-9) if self.history else res["games"] / max(res["wall"], 1e-9)
            if isinstance(action, OddsAction):
                self.ol_curve.append((len(self.memory.games), self.measure_plain_loss()))
            after = self.measure()
            wall = time.time() - t0
            observed = before["Y"] - after["Y"]
            comp, corr = info["computed"], info["corrected"]
            t = self.record["types"].setdefault(action.type, {"tries": 0, "computed": 0.0, "observed": 0.0, "seconds": 0.0})
            t["tries"] += 1
            t["computed"] += comp["total"]
            t["observed"] += observed
            t["seconds"] += wall
            self.last_tried[action.type] = step
            rec = {"sleep": self.index, "step": step, "action": action.name(), "type": action.type, "expects": action.expects(), "explore": info["explore"], "seconds": wall, "games": res.get("games"),
                   "computed_total": comp["total"], "computed_per_second": comp["per_second"], "corrected_per_second": corr, "observed_total": observed, "observed_per_second": observed / max(wall, 1e-9),
                   "Y_before": before["Y"], "Y": after["Y"], "hl": after["hl"], "ol": after["ol"], "unordered": after["unordered"], "soft": after["soft"], "elapsed": time.time() - t_start}
            append_json(self.dir / "log.jsonl", rec)
            self.history.append(rec)
            self.save_state()
            self.save_chooser()
            say(f"OBS sleep {self.index} action {step}: {action.name()}{' (exploring)' if info['explore'] else ''}, expects {action.expects()}; {res.get('games')} {'games' if isinstance(action, OddsAction) else 'fit games added'} in {wall:.0f} s; "
                f"expected gain on Y per second, computed {comp['per_second']:+.3e} (corrected {corr:+.3e}), observed {observed / max(wall, 1e-9):+.3e} (in all: computed {comp['total']:+.3e}, observed {observed:+.3e}); parts computed {json.dumps({k: round(v, 6) for k, v in comp['parts'].items()})}")
            say(f"OBS sleep {self.index} yardstick at t={time.time() - t_start:.0f} s: {self.describe(after)}")
            step += 1
        say(f"OBS sleep {self.index} wakes after {time.time() - t_start:.0f} s and {step} actions: {reason}")
        return {"reason": reason, "actions": step, "wall": time.time() - t_start, "final": self.measure()}

    # ---- the new store
    def new_record(self) -> dict:
        m = self.measure()
        lo, hi = np.percentile(m["draws"], 5, axis=0), np.percentile(m["draws"], 95, axis=0)
        order = [self.kinds[i] for i in np.argsort(-m["terms"])]
        values = {k: int(round(O.SCALE * m["terms"][i])) for i, k in enumerate(self.kinds)}
        per = {k: {"one": {"term": float(m["terms"][i]), "lo": float(lo[i]), "hi": float(hi[i])}} for i, k in enumerate(self.kinds)}
        return {"estimator": "the sleep of plan 78.63: the odds head fitted on the memory of odds games", "heads": {"odds_effect": {
            "type": "terms", "source": f"sleep {self.index} of {self.tag}", "played_with": f"worth of {self.store['from']}", "inputs": "the start's unit-count differences by kind (first mover less the other)", "kinds": self.kinds, "per_kind": per,
            "order_of_kinds": order, "unordered_pairs": [[p["a"], p["b"]] for p in m["pairs"] if p["unordered"]], "kind_values": values, "kind_values_note": f"integer = round({O.SCALE} x the term of the logit head), as measured (no clamp)",
            "data": {"games_in_memory": m["n_odds"], "human_fit_games": self.human_n}}}}

    def finish(self, result: dict, pairs: int, say=None) -> dict:
        """The wake test and the store: the new record's player against the player this sleep played with; kept unless clearly worse."""
        say = say or self.say
        rec = self.new_record()
        new = player_from_values(rec["heads"]["odds_effect"]["kind_values"])
        r = match(self.pool, f"sleep{self.index}_new_v_previous", new, self.player, pairs, self.cfg, self.root / "matches", say)
        keep, why = wake_decision(r)
        (self.dir / "record.json").write_text(json.dumps(rec, indent=1))
        mem = {"games": str(self.dir / "games.jsonl"), "starts": str(self.dir / "starts.json")}
        out = {"kept": keep, "why": why, "match": r, "record": rec, "new_player": new}
        say(f"OBS sleep {self.index} wake test: {'KEPT' if keep else 'NOT KEPT, the old store stays'} ({why}); the previous player's worth {json.dumps(self.player['record_kinds'])}, the new one's {json.dumps(new['record_kinds'])}")
        m = result["final"]
        first = json.loads((self.root / "baseline.json").read_text())
        say(f"OBS sleep {self.index} yardstick gain against match: Y at the start of this sleep {self.history[0]['Y_before'] if self.history else m['Y']:.4f}, at its end {m['Y']:.4f} (gain {(self.history[0]['Y_before'] if self.history else m['Y']) - m['Y']:+.4f}); match score of the new store against the previous {r['score']:.4f} [{r['lo']:.4f}, {r['hi']:.4f}]; baselines {json.dumps(first)}")
        if keep:
            new_store = {"record": rec, "player": new, "from": f"sleep {self.index} of {self.tag}", "memory": self.store["memory"] + [mem], "human_n": self.human_n, "sleep": self.index}
        else:
            new_store = dict(self.store)
        out["store"] = new_store
        d = self.root / "store"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"after_sleep{self.index}.json").write_text(json.dumps({"memory": new_store["memory"], "human_n": new_store["human_n"], "from": new_store["from"], "sleep": new_store["sleep"], "kept": keep,
                                                                       "record_path": str(self.dir / "record.json"), "worth": new_store["player"]["record_kinds"]}, indent=1))
        return out


def stop_rule(elapsed: float, budget: float, slice_s: float, free: float, sandbox_gb: float, best_corrected: float, recent: list) -> "str | None":
    """The reasons to wake, in order: the time budget (less than a quarter of a slice left), the disk (free space under the limit, or the sandbox over its own), no action expected to pay (the best corrected
    computed gain per second under THETA), no recent gain (no new best of Y by THETA a second over PATIENCE_SLICES slice lengths). `recent` is a list of (time at the end of an action, Y), the start included."""
    if budget - elapsed < MIN_SLICE_SHARE * slice_s:
        return f"the time budget ({budget:.0f} s) is used"
    if free < DISK_GB:
        return f"the disk limit: {free:.1f} GB free, under {DISK_GB} GB"
    if sandbox_gb > SANDBOX_GB:
        return f"the sandbox holds {sandbox_gb:.2f} GB, over {SANDBOX_GB} GB"
    if best_corrected < THETA:
        return f"no action is expected to pay: the best corrected computed gain per second {best_corrected:+.2e} is under {THETA:.0e}"
    window = PATIENCE_SLICES * slice_s
    if recent and elapsed >= window:
        before = [y for t, y in recent if t <= elapsed - window]
        within = [y for t, y in recent if t > elapsed - window]
        if before and within and min(within) > min(before) - THETA * window:
            return f"no recent gain: in the last {window:.0f} s the best Y was {min(within):.4f}, no better than the best before it, {min(before):.4f}, by {THETA * window:.4f} (theta {THETA:.0e} a second)"
    return None


# ------------------------------------------------------------------------------------------------------------------ the fixed schedule (78.61's: every start, the same games each, the control three times)

def fixed_schedule(sleep: Sleep) -> dict:
    """The comparison of stage F: the same total time on the fixed list of 78.61 (every start in turn, the even start three times a round), no choosing, no human fit."""
    say, cfg = sleep.say, sleep.cfg
    sleep.prepare()
    sleep.ensure_base_starts()
    order = [sid for sid in sleep.base_starts for _ in range(O.CONTROL_FACTOR if sid == "even" else 1)]
    pos = [0]

    def next_spec():
        sid = order[pos[0] % len(order)]
        pos[0] += 1
        return sleep.spec_for(sid)
    t_start = time.time()
    say(f"OBS fixed {sleep.index} start: {sleep.describe(sleep.measure())}; the fixed list of {len(order)} games a round, slices of {cfg['slice_seconds']} s, {cfg['sleep_seconds']} s in all")
    step = 0
    while cfg["sleep_seconds"] - (time.time() - t_start) >= MIN_SLICE_SHARE * cfg["slice_seconds"]:
        seconds = min(cfg["slice_seconds"], cfg["sleep_seconds"] - (time.time() - t_start))
        before = sleep.measure()
        t0 = time.time()
        games = play_until(sleep.pool, next_spec, t0 + seconds, sleep.dir / "games.jsonl", cfg["workers"], sleep.on_game)
        sleep.measure_cache = None
        after = sleep.measure()
        rec = {"step": step, "games": len(games), "seconds": time.time() - t0, "Y_before": before["Y"], "Y": after["Y"], "hl": after["hl"], "ol": after["ol"], "unordered": after["unordered"], "elapsed": time.time() - t_start}
        append_json(sleep.dir / "log.jsonl", rec)
        sleep.history.append(rec)
        say(f"OBS fixed {sleep.index} slice {step}: {len(games)} games in {rec['seconds']:.0f} s; yardstick at t={rec['elapsed']:.0f} s: {sleep.describe(after)}")
        step += 1
    say(f"OBS fixed {sleep.index} ends after {time.time() - t_start:.0f} s and {step} slices: the time budget")
    return {"reason": "the time budget", "actions": step, "wall": time.time() - t_start, "final": sleep.measure()}


# ------------------------------------------------------------------------------------------------------------------ endgame coverage (the standing rule)

def coverage_lines(fixed: Fixed, say) -> None:
    """Units left and how the games ended, for the fit games and the held-out games of the human sample: the share of positions and of games by units left, the small sets of units, the endings."""
    from catspace.incubator.analyst.discovery import graph_layer1 as GL
    games = [json.loads(x) for x in Path(paths.experiment("estimator_78_59", "full", "games.jsonl")).read_text().splitlines() if x.strip()]
    model = GL.model(HD.GAME, HD.VARIANT)
    names, raw = model.position_table([(None, g["moves"]) for g in games], 4)
    nk = len(names)
    flat = np.frombuffer(raw, dtype=np.int16).astype(np.int32)
    w = 2 + 2 * nk
    test = set(fixed.test_games)
    stats = {"fit": {"pos": 0, "games": 0, "end": {}, "reach": {}, "reach_g": {}, "sets": {}, "set_games": {}}, "test": {"pos": 0, "games": 0, "end": {}, "reach": {}, "reach_g": {}, "sets": {}, "set_games": {}}}
    cuts = {"men<=12": 12, "men<=7": 7, "men<=5": 5}
    off = 0
    for i, g in enumerate(games):
        n = len(g["moves"]) + 1
        t = flat[off:off + n * w].reshape(n, w)
        off += n * w
        s = stats["test" if i in test else "fit"]
        s["games"] += 1
        men = t[:, 2:2 + nk].sum(axis=1) + t[:, 2 + nk:2 + 2 * nk].sum(axis=1)
        valid = men > 0                                  # the row after the last move carries no units
        s["pos"] += int(valid.sum())
        end = CE.state(O.VARIANT, None, g["moves"])[3]
        key = ("decisive, stopped before the end on the board" if g["y"] != 1 else "drawn, stopped") if end is None else ("the end on the board, decisive" if g["y"] != 1 else "the end on the board, drawn")
        s["end"][key] = s["end"].get(key, 0) + 1
        for c, lim in cuts.items():
            k = int(((men <= lim) & valid).sum())
            s["reach"][c] = s["reach"].get(c, 0) + k
            s["reach_g"][c] = s["reach_g"].get(c, 0) + (k > 0)
        seen = set()
        for p in range(n):
            if valid[p] and men[p] <= 6:
                lab = " ".join(f"{int(t[p, 2 + j])}{names[j]}" for j in range(nk) if t[p, 2 + j]) + " v " + " ".join(f"{int(t[p, 2 + nk + j])}{names[j]}" for j in range(nk) if t[p, 2 + nk + j])
                s["sets"][lab] = s["sets"].get(lab, 0) + 1
                seen.add(lab)
        for lab in seen:
            s["set_games"][lab] = s["set_games"].get(lab, 0) + 1
    for nm, s in stats.items():
        say(f"OBS sleep coverage [{nm} human games]: games {s['games']}, positions {s['pos']}; " + "; ".join(f"{c}: positions {100 * s['reach'][c] / s['pos']:.2f} pct, games reaching it {100 * s['reach_g'][c] / s['games']:.1f} pct" for c in cuts)
            + "; how games ended: " + "; ".join(f"{k} {v} ({100 * v / s['games']:.1f} pct)" for k, v in sorted(s["end"].items(), key=lambda x: -x[1])))
        top = sorted(s["sets"].items(), key=lambda x: -x[1])[:8]
        thin = sum(1 for lab in s["sets"] if s["set_games"][lab] < 20)
        say(f"OBS sleep coverage [{nm} human games]: sets of 6 units or fewer, the commonest by positions: " + "; ".join(f"[{lab}] {c} positions in {s['set_games'][lab]} games" for lab, c in top)
            + f"; sets seen {len(s['sets'])}, of them in fewer than 20 games {thin} (a set that thin is not read by a fit; the human actions fit every position, no set is picked out)")


def later_coverage(sleep: Sleep, say) -> None:
    texts = [s["text"] for sid, s in sleep.memory.registry.items() if sid.startswith("later:") and sid.endswith(":even")]
    if not texts:
        return
    men = [len(O.units_of(t)) for t in texts]
    say(f"OBS sleep coverage [positions sampled for odds_later]: {len(texts)} positions; units left, least {min(men)}, median {int(np.median(men))}, most {max(men)}; positions with 12 or fewer {sum(m <= 12 for m in men)}, 7 or fewer {sum(m <= 7 for m in men)}, 5 or fewer {sum(m <= 5 for m in men)} "
        f"(the sample is taken between plies {LATER_PLIES[0]} and {LATER_PLIES[1]}: it holds no endings, and is thin for the sets coverage lists; a measure of an ending would need games from generated endings, not these starts)")


# ------------------------------------------------------------------------------------------------------------------ main

def open_log(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    f = open(root / "obs.txt", "a")

    def say(line: str) -> None:
        print(line, flush=True)
        f.write(line + "\n")
        f.flush()
    return say


def main(argv: "list | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stage", choices=["sleeps", "fixed"])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--sleeps", type=int, default=1)
    ap.add_argument("--sleep-seconds", type=float, default=600.0)
    ap.add_argument("--slice-seconds", type=float, default=60.0)
    ap.add_argument("--seconds", type=float, default=0.05)
    ap.add_argument("--max-plies", type=int, default=120)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--pairs", type=int, default=50)
    ap.add_argument("--start-gen", type=int, default=2, help="the generation of plan 78.61 the sleep starts from as the previous one (0: the player that counts nothing)")
    ap.add_argument("--no-coverage", action="store_true")
    ap.add_argument("--vs-sleep", default="", help="fixed stage: the tag/sleep index of a store to play the fixed result against, as tag:index")
    a = ap.parse_args(argv)
    cfg = {"sleep_seconds": a.sleep_seconds, "slice_seconds": a.slice_seconds, "seconds": a.seconds, "max_plies": a.max_plies, "workers": a.workers, "pairs": a.pairs, "start_gen": a.start_gen}
    root = sandbox(a.tag)
    say = open_log(root)
    say(f"OBS sleep settings: {json.dumps(cfg)}; theta {THETA}, patience {PATIENCE_SLICES} slices, explore share {EXPLORE}, human start {HUMAN_START} games and step {HUMAN_STEP}, ridge {C_HEAD}, refit draws {DRAWS}, later plies {LATER_PLIES}, "
        f"games per later start {LATER_GAMES}, wake rule: {WAKE_LEVEL_NOTE}; load average {os.getloadavg()[0]:.2f}, disk free {free_gb():.1f} GB")
    fixed = Fixed()
    if not a.no_coverage:
        coverage_lines(fixed, say)
    pool = multiprocessing.get_context("spawn").Pool(a.workers) if a.workers > 1 else None
    try:
        store = None
        if a.stage == "sleeps":
            for k in range(1, a.sleeps + 1):
                sl = Sleep(a.tag, k, cfg, fixed, pool, say, store)
                if k == 1:
                    sl.write_yardstick()
                res = sl.run()
                later_coverage(sl, say)
                out = sl.finish(res, a.pairs)
                store = out["store"]
        else:
            sl = Sleep(a.tag, 0, cfg, fixed, pool, say, None)
            sl.write_yardstick()
            res = fixed_schedule(sl)
            rec = sl.new_record()
            new = player_from_values(rec["heads"]["odds_effect"]["kind_values"])
            (sl.dir / "record.json").write_text(json.dumps(rec, indent=1))
            match(pool, "fixed_v_previous", new, sl.player, a.pairs, cfg, root / "matches", say)
            if a.vs_sleep:
                tag, idx = a.vs_sleep.split(":")
                other = json.loads((sandbox(tag) / f"sleep{idx}" / "record.json").read_text())
                match(pool, f"fixed_v_sleep{idx}_of_{tag}", new, player_from_values(other["heads"]["odds_effect"]["kind_values"]), a.pairs, cfg, root / "matches", say)
            m = res["final"]
            say(f"OBS fixed yardstick at the end: {sl.describe(m)}; match line above")
    finally:
        if pool is not None:
            pool.terminate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
