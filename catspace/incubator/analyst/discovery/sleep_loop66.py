"""PLAN 78.66: THE SLEEP'S YARDSTICK MADE OF WHAT THE PLAYER READS. Extends sleep_loop.py (78.63); a sandbox under artifacts/experiments/sleep_78_66/ through the path registry; nothing is written to the real store,
nothing is wired into any player. Game-free: kinds, sides and places are words read from the graph at run time.
No engine evaluation, no labeller number and no outside call enters: only the results of games (human games, and the games this sleep plays).

THE PLAYER READS ONE THING: the record of worths per kind (integers, the odds record). THE YARDSTICK scores exactly that record turned into a result prediction: the score s of a position is the sum over kinds of
worth x (units of the side to move less the other side's); the result is predicted by a one-input multinomial logit of the result on s (win, draw, loss), its three-class intercepts and slope FROZEN at the start of
the sleep (fitted before the first action, on the sleep's own fit games for the first held-out set and on the fit human games for the second; the held-out games never enter a fit). Two held-out sets, both fixed
before the sleep's first action: (i) games of the sleep's OWN current player, played fresh at the start of the sleep from the start list (every start, the even start three times a round), a stated number of
them, their time taken out of the sleep's budget; (ii) the held-out human games of the 78.59 split, every settled row. Y is the mean of the two held-out log losses, each over its value at the start of the sleep
(smaller is better). Nothing the player does not read is in it: the human result head of 78.63 is gone from Y.
THE ACTIONS that can change the record: odds_kind, odds_pair, odds_later (every one adds odds games, hence refits the worths). human_fit fits the human result head, which no player reads: it cannot change the
record, its computed gain is zero, it is never chosen.
THE COMPUTED GAIN of an action that can change the record: the expected fall of Y from the narrower spread of the worths it touches, one half x the curvature of Y in a worth (a finite difference of the Y
function itself) x the variance of that worth over the refits on resampled games x the share of that variance the action's games remove; the spread between worths is neglected.
THE THREE-SLICE RULE: an action chosen runs a bout of three slices; its observed gain (Y before its first slice less Y after its third) enters the chooser's record of observed against computed only after those
three slices; a bout the stop cut short enters nothing.

  .venv/bin/python -m catspace.incubator.analyst.discovery.sleep_loop66 sleeps --tag T --sleeps N --sleep-seconds S --slice-seconds L --own-games G --pairs 100
  .venv/bin/python -m catspace.incubator.analyst.discovery.sleep_loop66 fixed  --tag T --sleep-seconds S --slice-seconds L --own-games G --vs-sleep TAG:INDEX
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import os
import random
import time
from pathlib import Path

import numpy as np

from catspace.incubator.analyst.discovery import estimator_fit as EF
from catspace.incubator.analyst.discovery import odds_measure as O
from catspace.incubator.analyst.discovery import sleep_loop as SL
from catspace.io import paths

SLICES_PER_BOUT = 3                   # an action's observed gain is judged over at least this many slices
OWN_GAMES = 300                       # held-out games of the sleep's own player
HALF_STEP = 50                        # a worth unit step either way for the curvature of Y
QUIET_LOAD = 4.0                      # a wake match waits (up to five minutes) for the one-minute load average to fall to this
QUIET_WAIT = 300.0


def sandbox(tag: str) -> Path:
    return Path(paths.experiment("sleep_78_66", tag))


def wait_quiet(say, label: str) -> None:
    """Before a match: print the load average and the cores busy; if the load average is above QUIET_LOAD wait up to five minutes for it to fall, and print what was seen."""
    t0 = time.time()
    load, busy = os.getloadavg()[0], O.cpu_busy()
    say(f"OBS quiet machine before {label}: load average (1 min) {load:.2f}, cores busy {busy:.2f}")
    while os.getloadavg()[0] > QUIET_LOAD and time.time() - t0 < QUIET_WAIT:
        time.sleep(10.0)
    if time.time() - t0 > 1.0:
        say(f"OBS quiet machine before {label}: waited {time.time() - t0:.0f} s; load average (1 min) now {os.getloadavg()[0]:.2f}, cores busy {O.cpu_busy():.2f}")


# ------------------------------------------------------------------------------------------------------------------ the yardstick

class Yardstick:
    """The record's result prediction on two held-out sets. `worth` is the integer vector the player reads, one per kind. The prediction is a one-input multinomial logit on the score s = x . worth / SCALE with
    its parameters frozen at construction. Own set: counts x (the start's first mover's units less the other's) and the first mover's result. Human set: counts and the mover's result, per settled row."""

    def __init__(self, own_x: np.ndarray, own_y: np.ndarray, hum_x: np.ndarray, hum_y: np.ndarray, cal_own: tuple, cal_hum: tuple, worth0: np.ndarray):
        self.own_x, self.own_y, self.hum_x, self.hum_y = own_x, own_y, hum_x, hum_y
        self.model_own = self.calibrate(*cal_own, worth0)
        self.model_hum = self.calibrate(*cal_hum, worth0)
        self.base = self.raw(worth0)

    @staticmethod
    def score(x: np.ndarray, worth: np.ndarray) -> np.ndarray:
        return (x @ np.asarray(worth, dtype=float) / O.SCALE).reshape(-1, 1)

    def calibrate(self, x: np.ndarray, y: np.ndarray, worth0: np.ndarray):
        """The frozen map from the score to the result: fitted once, on fit data only."""
        return EF.fit_lr(self.score(x, worth0), y, SL.C_HEAD) if len(np.unique(y)) > 1 else None

    def raw(self, worth: np.ndarray) -> dict:
        po = EF.probs(self.model_own, self.score(self.own_x, worth)) if self.model_own is not None else EF.base_probs(self.own_y, len(self.own_y))
        ph = EF.probs(self.model_hum, self.score(self.hum_x, worth))
        return {"own": EF.logloss(po, self.own_y), "human": EF.logloss(ph, self.hum_y)}

    def y_of(self, worth: np.ndarray) -> dict:
        r = self.raw(worth)
        return {**r, "Y": 0.5 * (r["own"] / self.base["own"] + r["human"] / self.base["human"])}

    def curvature(self, k: int, worth: np.ndarray) -> float:
        """Y per worth unit squared on kind k, by a finite difference of Y itself (the loss is convex in the worths, so it is not negative)."""
        up, dn = np.array(worth, dtype=float), np.array(worth, dtype=float)
        up[k] += HALF_STEP
        dn[k] -= HALF_STEP
        return float(max(0.0, (self.y_of(up)["Y"] + self.y_of(dn)["Y"] - 2.0 * self.y_of(worth)["Y"]) / HALF_STEP ** 2))


def worth_of(terms: np.ndarray) -> np.ndarray:
    return np.array([int(round(O.SCALE * t)) for t in terms])


# ------------------------------------------------------------------------------------------------------------------ the sleep

class Sleep66(SL.Sleep):
    def __init__(self, tag: str, index: int, cfg: dict, fixed: SL.Fixed, pool, say=print, store: "dict | None" = None):
        self.own_games = int(cfg.get("own_games", OWN_GAMES))
        self.yard: "Yardstick | None" = None
        self.own_set: list = []
        self.prep_seconds = 0.0
        self.base_Y: dict = {}
        self.bouts: list = []
        keep = SL.sandbox
        SL.sandbox = sandbox                                       # the parent builds its directories through this name: they go to this plan's sandbox
        try:
            super().__init__(tag, index, cfg, fixed, pool, say, store)
        finally:
            SL.sandbox = keep

    # ---- the memory: every game of the earlier generation is a fit game (the held-out sets are new and fresh)
    def load_memory(self) -> None:
        held, kept = SL.previous_generation_games(self.root_61, self.cfg.get("start_gen", 2), self.made, self.memory)
        for g in kept + held:
            self.memory.add_game(g)
        for src in self.store["memory"]:
            reg = json.loads(Path(src["starts"]).read_text())
            for sid, s in reg.items():
                self.memory.add_start(sid, s)
            for g in SL.read_json_lines(Path(src["games"])):
                self.memory.add_game(g)
        self.held_x = self.held_y = None
        self.held_gids = []
        self.memory_games_at_start = len(self.memory.games)

    def write_yardstick(self) -> dict:
        h = self.fixed.rows["game"][self.fixed.test_rows]
        d = {"human_held_out_games": len(self.fixed.test_games), "human_held_out_rows": int(len(self.fixed.test_rows)), "own_held_out_games": len(self.own_set), "own_games_requested": self.own_games,
             "own_held_out_digest": SL.hashlib.md5(json.dumps(sorted(g["gid"] for g in self.own_set)).encode()).hexdigest(), "human_held_out_digest": SL.hashlib.md5(json.dumps(self.fixed.test_games).encode()).hexdigest(),
             "human_rows_check": int(len(h)), "odds_fit_games_at_start": self.memory_games_at_start}
        (self.dir / "yardstick.json").write_text(json.dumps(d, indent=1))
        return d

    # ---- the own-player held-out games (before the first action; out of the sleep's budget)
    def own_specs(self) -> list:
        order = [sid for sid in self.base_starts for _ in range(O.CONTROL_FACTOR if sid == "even" else 1)]
        specs = []
        for i in range(self.own_games):
            sid = order[i % len(order)]
            k = i                                                   # the game's own number: the even start appears three times a round, each with its own seed
            s = self.base_starts[sid]
            seed = SL.SEED + 7000000 + 100000 * (self.index + 1) + 1000 * (O.hash_of(sid) % 997) + k
            specs.append({"gid": f"h{self.index}:{sid}:{k}", "start": sid, "text": s["text"], "opening": O.random_opening(s["text"], seed), "players": [self.player, self.player],
                          "seconds": self.cfg["seconds"], "max_plies": self.cfg["max_plies"], "scratch": str(self.dir / "scratch_held")})
        return specs

    def human_x(self, rows: np.ndarray) -> np.ndarray:
        """The human rows' kind counts (mover less the other), in this sleep's kind order; a kind the human record does not hold is zero."""
        kd = self.fixed.rows["kd"][rows].astype(float)
        out = np.zeros((len(rows), len(self.kinds)))
        for j, name in enumerate(self.fixed.kinds_human):
            if name in self.kinds:
                out[:, self.kinds.index(name)] = kd[:, j]
        return out

    def build_yardstick(self) -> None:
        t0 = time.time()
        self.ensure_base_starts()
        self.own_set = SL.run_specs(self.pool, self.own_specs(), self.dir / "heldout_own.jsonl")
        gen_s = time.time() - t0
        for g in self.own_set:
            self.memory.add_start(g["start"], self.base_starts[g["start"]])
        hm = SL.Memory(self.kinds, self.first)
        for sid, s in self.base_starts.items():
            hm.add_start(sid, s)
        for g in self.own_set:
            hm.add_game(g)
        own_x, own_y, _s, own_gids = hm.arrays()
        self.held_gids = own_gids
        x, y, _s2, _g2 = self.memory.arrays()
        worth0 = worth_of(SL.kind_terms_of(SL.fit_odds(x, y), len(self.kinds)))
        fit_rows = self.fixed.fit_rows(SL.HUMAN_START)
        test_rows = self.fixed.test_rows
        self.yard = Yardstick(own_x, own_y, self.human_x(test_rows), self.fixed.y[test_rows], (x, y), (self.human_x(fit_rows), self.fixed.y[fit_rows]), worth0)
        self.base_Y = dict(self.yard.base)
        (self.dir / "baseline.json").write_text(json.dumps(self.base_Y))
        self.prep_seconds = time.time() - t0
        (self.dir / "prep.json").write_text(json.dumps({"seconds": self.prep_seconds, "held_out_seconds": gen_s}))
        names = ", ".join(self.kinds)
        shared = sorted(set(self.fixed.kinds_human) & set(self.kinds))
        self.say(f"OBS sleep {self.index} held-out sets fixed before the first action: own player: {len(self.own_set)} fresh games from the start list in {gen_s:.0f} s (taken out of the sleep's budget; {self.cfg['workers']} workers); "
                 f"human: {len(self.fixed.test_games)} held-out games, {len(test_rows)} settled rows; frozen result maps fitted on {len(y)} fit odds games and on {len(fit_rows)} fit human rows ({SL.HUMAN_START} fit human games); "
                 f"kinds in the record: {names}; shared with the human record's kinds: {len(shared)} of {len(self.kinds)}; starting worths {json.dumps(dict(zip(self.kinds, worth0.tolist())))}; "
                 f"yardstick at the start: own log loss {self.base_Y['own']:.5f}, human log loss {self.base_Y['human']:.5f}")

    # ---- the yardstick of the memory as it is
    def measure(self) -> dict:
        if self.measure_cache is not None:
            return self.measure_cache
        x, y, starts, _g = self.memory.arrays()
        m = SL.fit_odds(x, y)
        terms = SL.kind_terms_of(m, len(self.kinds))
        worth = worth_of(terms)
        draws = SL.bootstrap_terms(x, y, starts, SL.DRAWS, SL.SEED)
        pairs = SL.pair_table(draws, self.kinds)
        r = self.yard.y_of(worth)
        out = {"Y": r["Y"], "own": r["own"], "hl": r["human"], "ol": r["own"], "human": r["human"], "worth": worth, "terms": terms, "draws": draws, "pairs": pairs, "n_odds": len(y),
               "n_kind": [int((x[:, i] != 0).sum()) for i in range(len(self.kinds))], "curv": [self.yard.curvature(k, worth) for k in range(len(self.kinds))], "unordered": sum(p["unordered"] for p in pairs), "soft": sum(p["soft"] for p in pairs) / max(1, len(pairs))}
        self.measure_cache = out
        return out

    def describe(self, m: dict) -> str:
        return (f"Y {m['Y']:.5f} (own-player held-out log loss {m['own']:.5f} on {len(self.own_set)} games, human held-out log loss {m['human']:.5f} on {len(self.yard.hum_y)} rows; "
                f"{m['n_odds']} odds games in memory; worths {json.dumps(dict(zip(self.kinds, m['worth'].tolist())))}; kinds unordered {m['unordered']} of {len(m['pairs'])})")

    def prepare(self) -> None:
        t0 = time.time()
        self.build_yardstick()
        m = self.measure()
        self.say(f"OBS sleep {self.index} prepared in {time.time() - t0:.1f} s: {self.describe(m)}")

    # ---- the actions: which can change what the player reads
    def candidates(self, m: dict) -> list:
        out: list = [SL.HumanFit()]
        for k in self.kinds:
            out.append(SL.OddsKind(k))
            out.append(SL.OddsLater(k))
        for p in m["pairs"]:
            if p["unordered"]:
                out.append(SL.OddsPair(p["a"], p["b"]))
        return out

    @staticmethod
    def changes_record(action: SL.Action) -> bool:
        return isinstance(action, SL.OddsAction)

    def computed_gain(self, action: SL.Action, seconds: float, m: dict) -> dict:
        if not self.changes_record(action):
            return {"total": 0.0, "seconds": seconds, "per_second": 0.0, "parts": {"cannot_change_the_record": 0.0}}
        games = self.rate * seconds
        touched = action.kinds_touched()
        share = games / len(touched)
        worth = m["worth"]
        parts = {}
        for kind in touched:
            k = self.kinds.index(kind)
            var = float(m["draws"][:, k].var()) * O.SCALE ** 2
            nk = max(m["n_kind"][k], 1)
            parts[kind] = 0.5 * m["curv"][k] * var * (1.0 - nk / (nk + share))
        tot = float(sum(parts.values()))
        return {"total": tot, "seconds": seconds, "per_second": tot / seconds, "parts": parts}

    def rank(self, seconds: float) -> "tuple[list, dict]":
        m = self.measure()
        rows = []
        for a in self.candidates(m):
            g = self.computed_gain(a, seconds, m)
            rows.append((a, g, g["per_second"] * self.correction(a.type)))
        live = [r for r in rows if self.changes_record(r[0])]
        info = {"best_corrected": max((r[2] for r in live), default=0.0), "ranked": sorted([(r[0].name(), r[1]["per_second"], r[2]) for r in live], key=lambda x: -x[2])[:5]}
        return rows, info

    def choose(self, seconds: float, step: int) -> "tuple[SL.Action | None, dict]":
        rows, info = self.rank(seconds)
        live = [r for r in rows if self.changes_record(r[0])]
        if not live:
            return None, info
        rng = random.Random(f"{SL.SEED}:{self.index}:{step}")
        explore = rng.random() < SL.EXPLORE
        if explore:
            types = sorted({r[0].type for r in live}, key=lambda t: (self.last_tried.get(t, -1), t))
            pick = max((r for r in live if r[0].type == types[0]), key=lambda r: r[2])
        else:
            pick = max(live, key=lambda r: r[2])
        info["explore"] = explore
        info["computed"], info["corrected"] = pick[1], pick[2]
        return pick[0], info

    def enter_bout(self, bout: dict, y_after: float) -> None:
        """The three-slice rule: the bout's observed gain and computed gain enter the chooser's record, here and nowhere else."""
        observed = bout["Y_start"] - y_after
        t = self.record["types"].setdefault(bout["type"], {"tries": 0, "computed": 0.0, "observed": 0.0, "seconds": 0.0})
        t["tries"] += 1
        t["computed"] += bout["computed"]
        t["observed"] += observed
        t["seconds"] += bout["seconds"]
        self.bouts.append({**{k: bout[k] for k in ("action", "type", "slices", "computed", "seconds")}, "observed": observed})
        SL.append_json(self.dir / "bouts.jsonl", self.bouts[-1])
        self.say(f"OBS sleep {self.index} bout of {bout['action']} ({bout['slices']} slices, {bout['seconds']:.0f} s) enters the chooser's record: computed gain {bout['computed']:+.3e}, observed gain {observed:+.3e}, "
                 f"{'same sign' if (bout['computed'] > 0) == (observed > 0) else 'opposite signs'}")

    # ---- the loop
    def run(self) -> dict:
        say = self.say
        t_start = time.time()
        self.prepare()
        t_clock = t_start                                           # the budget counts from here: the held-out games' time is in it
        step = 0
        reason = None
        first = self.measure()
        live_names = sorted({a.type for a in self.candidates(first) if self.changes_record(a)})
        say(f"OBS sleep {self.index} start: {self.describe(first)}; the sleep lasts at most {self.cfg['sleep_seconds']} s (the {self.prep_seconds:.0f} s of preparation inside it) in slices of {self.cfg['slice_seconds']} s, "
            f"{SLICES_PER_BOUT} slices a bout; the actions that can change the record the player reads: {', '.join(live_names)}; human_fit cannot (the human result head is read by no player): computed gain 0, never chosen; "
            f"player: {json.dumps(self.player['record_kinds'])}")
        bout: "dict | None" = None
        while True:
            elapsed = time.time() - t_clock
            seconds = min(self.cfg["slice_seconds"], SL.ACTION_LIMIT, self.cfg["sleep_seconds"] - elapsed)
            if bout is None:
                action, info = self.choose(self.cfg["slice_seconds"], step)
                reason = self.stop_reason(elapsed, info) if action is not None else "no action can change the record"
            else:
                action = bout["obj"]
                _rows, info = self.rank(self.cfg["slice_seconds"])
                comp = self.computed_gain(action, self.cfg["slice_seconds"], self.measure())
                info["computed"], info["corrected"], info["explore"] = comp, comp["per_second"] * self.correction(action.type), False
                reason = self.stop_reason(elapsed, info)
            if reason:
                break
            before = self.measure()
            if bout is None:
                bout = {"obj": action, "action": action.name(), "type": action.type, "slices": 0, "computed": 0.0, "seconds": 0.0, "Y_start": before["Y"]}
            t0 = time.time()
            res = action.run(self, seconds)
            self.measure_cache = None
            if res["games"]:
                self.rate = 0.5 * self.rate + 0.5 * res["games"] / max(res["wall"], 1e-9) if self.history else res["games"] / max(res["wall"], 1e-9)
            after = self.measure()
            wall = time.time() - t0
            comp = info["computed"]
            observed = before["Y"] - after["Y"]
            bout["slices"] += 1
            bout["computed"] += comp["total"]
            bout["seconds"] += wall
            self.last_tried[action.type] = step
            rec = {"sleep": self.index, "step": step, "action": action.name(), "type": action.type, "bout_slice": bout["slices"], "explore": info["explore"], "seconds": wall, "games": res.get("games"),
                   "computed_total": comp["total"], "computed_per_second": comp["per_second"], "corrected_per_second": info["corrected"], "observed_total": observed, "observed_per_second": observed / max(wall, 1e-9),
                   "Y_before": before["Y"], "Y": after["Y"], "own": after["own"], "human": after["human"], "hl": after["human"], "ol": after["own"], "unordered": after["unordered"], "soft": after["soft"],
                   "elapsed": time.time() - t_clock, "worth": after["worth"].tolist()}
            SL.append_json(self.dir / "log.jsonl", rec)
            self.history.append(rec)
            self.save_state()
            say(f"OBS sleep {self.index} slice {step}: {action.name()} slice {bout['slices']} of {SLICES_PER_BOUT}{' (exploring)' if info['explore'] else ''}; {res.get('games')} games in {wall:.0f} s; "
                f"gain on Y per second, computed {comp['per_second']:+.3e}, observed {observed / max(wall, 1e-9):+.3e} (in all: computed {comp['total']:+.3e}, observed {observed:+.3e}); parts computed {json.dumps({k: round(v, 8) for k, v in comp['parts'].items()})}")
            say(f"OBS sleep {self.index} yardstick at t={time.time() - t_clock:.0f} s: {self.describe(after)}")
            step += 1
            if bout["slices"] >= SLICES_PER_BOUT:
                self.enter_bout(bout, after["Y"])
                self.save_chooser()
                bout = None
        if bout is not None:
            say(f"OBS sleep {self.index} bout of {bout['action']} cut after {bout['slices']} slices by the stop: fewer than {SLICES_PER_BOUT}, so nothing of it enters the chooser's record")
        say(f"OBS sleep {self.index} wakes after {time.time() - t_clock:.0f} s and {step} slices ({len(self.bouts)} bouts entered): {reason}")
        return {"reason": reason, "actions": step, "wall": time.time() - t_clock, "final": self.measure()}

    # ---- the new store
    def finish(self, result: dict, pairs: int, say=None) -> dict:
        say = say or self.say
        rec = self.new_record()
        new = SL.player_from_values(rec["heads"]["odds_effect"]["kind_values"])
        assert [new["record_kinds"][k] for k in self.kinds] == result["final"]["worth"].tolist()          # the player matched is the record the yardstick scored
        wait_quiet(say, f"the wake match of sleep {self.index}")
        r = SL.match(self.pool, f"sleep{self.index}_new_v_previous", new, self.player, pairs, self.cfg, self.root / "matches", say)
        keep, why = SL.wake_decision(r)
        (self.dir / "record.json").write_text(json.dumps(rec, indent=1))
        mem = {"games": str(self.dir / "games.jsonl"), "starts": str(self.dir / "starts.json")}
        m = result["final"]
        y0 = self.history[0]["Y_before"] if self.history else m["Y"]
        agree = (y0 - m["Y"] > 0) == (r["score"] > 0.5)
        say(f"OBS sleep {self.index} wake test: {'KEPT' if keep else 'NOT KEPT, the old store stays'} ({why}); the previous player's worth {json.dumps(self.player['record_kinds'])}, the new one's {json.dumps(new['record_kinds'])}")
        say(f"OBS sleep {self.index} yardstick gain against match: Y at the start {y0:.5f}, at the end {m['Y']:.5f} (gain {y0 - m['Y']:+.5f}); own-player log loss at the end {m['own']:.5f} against {self.base_Y['own']:.5f} at the start, "
            f"human {m['human']:.5f} against {self.base_Y['human']:.5f}; match score of the new store against the previous {r['score']:.4f} [{r['lo']:.4f}, {r['hi']:.4f}]; sign agreement (gain over zero, score over one half): {'yes' if agree else 'no'}")
        sgn = [(b["computed"] > 0) == (b["observed"] > 0) for b in self.bouts]
        say(f"OBS sleep {self.index} computed against observed gain over {len(self.bouts)} bouts: same sign in {sum(sgn)}; " + "; ".join(f"{b['action']} computed {b['computed']:+.2e} observed {b['observed']:+.2e}" for b in self.bouts))
        new_store = ({"record": rec, "player": new, "from": f"sleep {self.index} of {self.tag}", "memory": self.store["memory"] + [mem], "human_n": self.human_n, "sleep": self.index} if keep else dict(self.store))
        d = self.root / "store"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"after_sleep{self.index}.json").write_text(json.dumps({"memory": new_store["memory"], "human_n": new_store["human_n"], "from": new_store["from"], "sleep": new_store["sleep"], "kept": keep,
                                                                       "record_path": str(self.dir / "record.json"), "worth": new_store["player"]["record_kinds"]}, indent=1))
        return {"kept": keep, "why": why, "match": r, "record": rec, "new_player": new, "store": new_store, "agree": agree}


def fixed_schedule(sleep: Sleep66) -> dict:
    """78.61's fixed list for the same total time: every start in turn, the even start three times a round, no choosing; the same yardstick (the held-out games are the start player's own)."""
    say, cfg = sleep.say, sleep.cfg
    t_start = time.time()
    sleep.prepare()
    order = [sid for sid in sleep.base_starts for _ in range(O.CONTROL_FACTOR if sid == "even" else 1)]
    pos = [0]

    def next_spec():
        sid = order[pos[0] % len(order)]
        pos[0] += 1
        return sleep.spec_for(sid)
    say(f"OBS fixed {sleep.index} start: {sleep.describe(sleep.measure())}; the fixed list of {len(order)} games a round, slices of {cfg['slice_seconds']} s, {cfg['sleep_seconds']} s in all (preparation inside it)")
    step = 0
    while cfg["sleep_seconds"] - (time.time() - t_start) >= SL.MIN_SLICE_SHARE * cfg["slice_seconds"]:
        seconds = min(cfg["slice_seconds"], cfg["sleep_seconds"] - (time.time() - t_start))
        before = sleep.measure()
        t0 = time.time()
        games = SL.play_until(sleep.pool, next_spec, t0 + seconds, sleep.dir / "games.jsonl", cfg["workers"], sleep.on_game)
        sleep.measure_cache = None
        after = sleep.measure()
        rec = {"step": step, "games": len(games), "seconds": time.time() - t0, "Y_before": before["Y"], "Y": after["Y"], "own": after["own"], "human": after["human"], "elapsed": time.time() - t_start, "worth": after["worth"].tolist()}
        SL.append_json(sleep.dir / "log.jsonl", rec)
        sleep.history.append(rec)
        say(f"OBS fixed {sleep.index} slice {step}: {len(games)} games in {rec['seconds']:.0f} s; yardstick at t={rec['elapsed']:.0f} s: {sleep.describe(after)}")
        step += 1
    say(f"OBS fixed {sleep.index} ends after {time.time() - t_start:.0f} s and {step} slices: the time budget")
    return {"reason": "the time budget", "actions": step, "wall": time.time() - t_start, "final": sleep.measure()}


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
    ap.add_argument("--pairs", type=int, default=100)
    ap.add_argument("--own-games", type=int, default=OWN_GAMES)
    ap.add_argument("--start-gen", type=int, default=2)
    ap.add_argument("--no-coverage", action="store_true")
    ap.add_argument("--no-match", action="store_true", help="skip the wake match (smoke)")
    ap.add_argument("--vs-sleep", default="", help="fixed stage: tag:index of a sleep's record to match the fixed result against directly")
    a = ap.parse_args(argv)
    cfg = {"sleep_seconds": a.sleep_seconds, "slice_seconds": a.slice_seconds, "seconds": a.seconds, "max_plies": a.max_plies, "workers": a.workers, "pairs": a.pairs, "start_gen": a.start_gen, "own_games": a.own_games}
    root = sandbox(a.tag)
    say = SL.open_log(root)
    say(f"OBS sleep settings: {json.dumps(cfg)}; theta {SL.THETA}, patience {SL.PATIENCE_SLICES} slices, explore share {SL.EXPLORE}, slices a bout {SLICES_PER_BOUT}, ridge {SL.C_HEAD}, refit draws {SL.DRAWS}, later plies {SL.LATER_PLIES}, "
        f"games per later start {SL.LATER_GAMES}, finite-difference step {HALF_STEP} worth units, wake rule: {SL.WAKE_LEVEL_NOTE}; load average {os.getloadavg()[0]:.2f}, disk free {SL.free_gb():.1f} GB")
    fixed = SL.Fixed()
    if not a.no_coverage:
        SL.coverage_lines(fixed, say)
    pool = multiprocessing.get_context("spawn").Pool(a.workers) if a.workers > 1 else None
    try:
        store = None
        if a.stage == "sleeps":
            for k in range(1, a.sleeps + 1):
                sl = Sleep66(a.tag, k, cfg, fixed, pool, say, store)
                res = sl.run()
                sl.write_yardstick()
                SL.later_coverage(sl, say)
                if a.no_match:
                    break
                out = sl.finish(res, a.pairs)
                store = out["store"]
        else:
            sl = Sleep66(a.tag, 0, cfg, fixed, pool, say, None)
            res = fixed_schedule(sl)
            sl.write_yardstick()
            rec = sl.new_record()
            new = SL.player_from_values(rec["heads"]["odds_effect"]["kind_values"])
            (sl.dir / "record.json").write_text(json.dumps(rec, indent=1))
            say(f"OBS fixed yardstick at the end: {sl.describe(res['final'])}; start {sl.history[0]['Y_before'] if sl.history else 'n/a'}")
            if not a.no_match:
                wait_quiet(say, "the fixed schedule's match against the previous player")
                SL.match(pool, "fixed_v_previous", new, sl.player, a.pairs, cfg, root / "matches", say)
                if a.vs_sleep:
                    tag, idx = a.vs_sleep.split(":")
                    other = json.loads((sandbox(tag) / f"sleep{idx}" / "record.json").read_text())
                    wait_quiet(say, "the direct match")
                    SL.match(pool, f"fixed_v_sleep{idx}_of_{tag}", new, SL.player_from_values(other["heads"]["odds_effect"]["kind_values"]), a.pairs, cfg, root / "matches", say)
    finally:
        if pool is not None:
            pool.terminate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
