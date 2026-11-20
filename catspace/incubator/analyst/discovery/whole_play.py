"""PLAN 79.11 PARTS B, C, D, THE WHOLE WAKE AS A PLAYER IN THE MATCH HARNESS, WITH THE LOG OF EVERY DECISION (SYSTEM_SPEC 8 the whole is judged, 7 "Calculation and estimate, decided together": the learned last layer needs the logged games, 4, 11; piece: the wake).
A sandbox: nothing is wired into any player of the wake, no default of the engine is changed, nothing is written to rules/learned or data/derived; every output goes under artifacts/experiments/whole_79_11/ (through catspace.io.paths).
Game-free: the game is named only through the pieces that exist (odds_measure, tiebreak_measure, record_play, whole_79_10_print's World, which hands the wake what it reads). THIS MODULE AND WHAT IT IMPORTS NEVER LOAD THE ORACLE WRAPPER and call no oracle (tests/test_plan79_11_play.py).
NO PYTHON WAKE: the mover calls the Rust binding `BaseGraph.wake_decide` (every piece on) and writes the log; nothing is decided here. The binding holds the open tactics of the last decision of a SLOT (the game and the seat), so the carry crosses no language boundary.

THE PLAYERS (both with the human-result record of worths on both sides; every match is played through `tiebreak_measure.run_games(..., stop=tiebreak_measure.settled)`, the harness of 78.67, reused as it is):
  wake   kind "wk_wake": the whole wake with every piece on (`PIECES`), its clock the spec's "seconds" (or the game's).
  old    kind "tb_control" (78.67's control player, unchanged): the ledger search with the count of worths at its end and the equal reading, nothing else; its clock is the spec's "seconds" (or the game's).
EQUAL WALL (part C): the old player does not spend the clock it is given (it ends a reading when the depth is complete or the fuse is reached), the wake spends its whole clock. `scale_of(games)` is the one scale found once on a sanity set: the wake's mean wall a move over the old player's at the same
clock; the old player is then given `scale x clock` ("old@equal"), and both are also played at the same clock given ("old@same"). The mean wall a move of each is printed by `lines_of`.

THE LOG (part B; the learned chooser's training data). One file a game, `<out>/<tag>/logs/<match>/<gid>.jsonl.gz` (gzip members appended, one JSON line each; written by the wake's mover only, `play11` adds the result line at the game's end):
  {"kind": "decision", "gid", "seat" (0 moves first), "ply" (moves played before it), "position" (the position text the engine gave), "legal" (the legal moves, the position's moves), "played", "search_first" (the move the search alone would play: the first of the moves it holds equal; None
   when it read no depth), "differs" (played != search_first), "reason" (what decided the move: forced_end | own_line_wins | answers_threat | fixes_goal | estimate | search_only | none), "wall" (seconds the whole decision took, the history keys included),
   "wake": the decision as the binding gives it (move, value, depth, nodes, equal, named_mine, named_theirs, chain, steps, and "log": candidates "moves" -- for each: the move, the null-window value, held equal, named, its own line read, what it serves or answers, the chance and the reasons' score,
   the fact about their goals after it -- with the goals, the threat rows, the refuted and unlisted names, the notes)}
  {"kind": "result", "gid", "colour", "outcome" (white's score 100, 50 or 0; None when unfinished), "plies", "unfinished", "moves", "wake_seat"}

INTERFACE (written before the bodies).
  player_spec(name, seconds=None) -> dict        "wake" | "old"; the human record; `seconds` its own clock (None: the game's).
  match_specs(a, b, set_name, pairs, seconds, scratch, log_dir, clock_a=None, clock_b=None) -> [spec]     the games of one match (sets A and B of 78.67, colours alternated); `log_dir` where the wake's logs go.
  MoverWhole(spec, seconds, seed, trace)          the mover; kinds above; other kinds are 78.67's.
  play11(spec) -> game                            one game; module level so the spawned workers import it; the per-seat stats ride with the result.
  mean_wall(games, who) -> float                  the mean wall time a move of side `who` (0: a, 1: b).
  scale_of(games) -> float                        the scale of part C from a sanity match of the wake against the old player at one clock.
  differed(games, who) -> dict                    {reason: count} of the decisions in which the wake's move differed from the search's first move, and the decisions.
  reason_of(why) -> str                           the class of the sentence the log gives as its reason (PLAN 79.12: the classes that name what separated the move from the search's first move, and the last layer's `explore` and `chooser`).
  MoverWhole.chooser_arg(ply) -> tuple | None     PLAN 79.12 B: (mode, weights, the seed of this decision) for the binding, from the spec's "chooser" {"mode": fixed|weights|explore, "weights": {input: weight}, "seed"}; None: no last layer, the rule decides.
                                                  The log's decision line then carries, under wake.log, "chooser" (the candidates' rows and weighted sums, the rule's pick, the move played, the propensity), "first_held", "first_lowered_by" and "step_seconds".
  Called by: the command line (`sanity`, `match`), whole_79_11_print.py (reads the logs), the tests.

  .venv/bin/python -m catspace.incubator.analyst.discovery.whole_play sanity | match --name m1|m2 --set A|B  [--tag T] [--workers W] [--pairs K] [--seconds S]
"""
from __future__ import annotations

from catspace.incubator.analyst.discovery import tiebreak_measure as TM          # first: it sets one thread a process before numpy is loaded

import argparse
import faulthandler
import gzip
import json
import os
import time
import zlib
from pathlib import Path

import cs_engine as CE

from catspace.incubator.analyst.discovery import odds_measure as OM
from catspace.incubator.analyst.discovery import record_play as RP
from catspace.incubator.analyst.discovery import whole_79_10_print as W10
from catspace.io import paths

OUT = ("whole_79_11",)
KIND_WAKE = "wk_wake"
KIND_OLD = TM.KIND["control"]
PIECES = list(W10.PIECES)                           # every piece of the whole on (the heads that read counts summed over the whole position are not ported, SYSTEM_SPEC 7)
MIN_PAIRS = RP.MIN_PAIRS
PLAYERS = ("wake", "old")
REASONS = ("forced_end", "own_line_wins", "answers_threat", "fixes_goal", "estimate", "search_only", "none",
           "first_lowered", "only_own_line", "only_answers", "only_fixes", "explore", "chooser")        # PLAN 79.12: the last six name what separated the move played from the search's first move, or the last layer
WATCHDOG = 300.0                                    # seconds a game may take before its worker is ended with its traceback kept (PLAN 79.11: one game of the first match hung in an engine call)
_WORLD = None
CTX: dict = {}                                      # the game being played in this process: {"gid", "log_dir"}


def out_root(tag: str) -> Path:
    return Path(paths.experiment(*OUT, tag))


def world() -> "W10.World":
    """What the wake reads, built once a process (the learned seeds, the reasons, the table of the chance of a goal: read, never refitted)."""
    global _WORLD
    if _WORLD is None:
        _WORLD = W10.World()
    return _WORLD


def fuses() -> tuple:
    """The frame read's fuses, as the world of 79.08 hands them to the wake (largest seed, fuse, row fuse, answer fuse)."""
    w8 = W10.W8
    return (w8.OLD.MAX_SEED_SIZE, w8.TS.FUSE, w8.TS.ROW_FUSE, w8.TS.FUSE)


def reason_of(why: str) -> str:
    """The class of the sentence the decision log gives as its reason (rule.rs `why_of` and `choose`): which of the reasons decided the move."""
    w = (why or "").strip()
    for prefix, name in (("the end of the game is forced", "forced_end"), ("its own line wins", "own_line_wins"), ("answers the threat", "answers_threat"), ("a fix of the", "fixes_goal"),
                         ("no threat of theirs and no goal decides it: the reasons' score", "estimate"), ("no threat of theirs and no goal decides it", "search_only"),
                         ("the equal reading put the search's first move lower", "first_lowered"), ("only its own line wins", "only_own_line"), ("only it answers the threat", "only_answers"), ("only it fixes the", "only_fixes"),
                         ("nothing separates it from the search's first move but the estimates", "estimate"), ("nothing separates it from the search's first move", "search_only"),
                         ("explore:", "explore"), ("the chooser:", "chooser")):
        if w.startswith(prefix):
            return name
    return "none"


class MoverWhole(TM.Mover67):
    """78.67's movers plus the whole wake (kind `wk_wake`); a spec may carry its own clock ("seconds")."""

    def __init__(self, spec: dict, seconds: float, seed: str, trace: str):
        super().__init__(spec, float(spec.get("seconds", seconds)), seed, trace)
        self.gid, _, seat = seed.rpartition(":")
        self.seat = int(seat)
        self.slot = seed                                                          # the game and the seat: the binding keeps this slot's open tactics for the next decision
        self.chooser = spec.get("chooser")                                        # PLAN 79.12 B: None, or {"mode": "fixed" | "weights" | "explore", "weights": {input: weight}, "seed": base seed}
        self.log_file = None
        if self.kind == KIND_WAKE and CTX.get("log_dir"):
            self.log_file = Path(CTX["log_dir"]) / f"{self.gid.replace(':', '_')}.jsonl.gz"
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
        for r in REASONS:
            self.stats["by_" + r] = 0                                             # decisions whose move differed from the search's first move, by the reason given
        self.stats["differs"] = 0

    def chooser_arg(self, ply: int) -> "tuple | None":
        """The last layer's argument for the binding: (mode, weights, the seed of THIS decision), the seed from the base seed, the game, the seat and the ply (a game replayed gives the same draws)."""
        c = self.chooser
        if not c:
            return None
        return (c["mode"], dict(c.get("weights") or {}), zlib.crc32(f"{c.get('seed', 0)}:{self.seed}:{ply}".encode()))

    def chooser_kw(self, ply: int) -> dict:
        """The keyword for the binding: {"chooser": ...} where a last layer is given, else nothing (the call is then the call of 79.11)."""
        a = self.chooser_arg(ply)
        return {} if a is None else {"chooser": a}

    def write(self, rec: dict) -> None:
        if self.log_file is None:
            return
        with gzip.open(self.log_file, "ab") as f:
            f.write((json.dumps(rec) + "\n").encode())

    def choose(self, text: "str | None", moves: list, legal: list) -> tuple:
        if self.kind != KIND_WAKE:
            return super().choose(text, moves, legal)
        w = world()                                                               # (built once a process, before the clock of a decision starts)
        t0 = time.perf_counter()
        pos = CE.state(OM.VARIANT, text, list(moves))[0]
        history = [CE.state(OM.VARIANT, text, list(moves[:k]))[4] for k in range(len(moves))]
        out = json.loads(w.model.wake_decide(pos, history, self.seconds, 64, dict(self.worth), {}, w.goal, PIECES, 0, 0, w.keys, fuses(), w.chances, w.reasons, TM.LEDGER_SHARE, self.slot, **self.chooser_kw(len(moves))))
        wall = time.perf_counter() - t0
        eq = out["equal"]
        first = eq[0] if eq else None                                             # the search's own move leads the moves it holds equal (wake.rs)
        mv = out["move"] if out["move"] in legal else (first if first in legal else legal[0])
        differs = first is not None and mv != first
        reason = reason_of(out["log"]["why"])
        st = self.stats
        st["decisions"] += 1
        st["ties"] += len(eq) >= 2
        st["changed"] += differs
        st["differs"] += differs
        if differs:
            st["by_" + reason] += 1
        st["wall"] += wall
        st["search"] += wall
        st["nodes"] = st.get("nodes", 0) + int(out["nodes"])
        st["depth_sum"] = st.get("depth_sum", 0) + int(out["depth"])
        self.write({"kind": "decision", "gid": self.gid, "seat": self.seat, "ply": len(moves), "position": pos, "legal": list(legal), "played": mv, "search_first": first, "differs": bool(differs), "reason": reason, "wall": wall, "wake": out})
        return mv, out["depth"]


def play11(spec: dict) -> dict:
    """One game as 78.61 plays it, with the movers above; the per-seat stats ride with the result and the result is appended to the wake's log."""
    OM.Mover = MoverWhole
    TM.REG.clear()
    CTX.clear()
    CTX.update(gid=spec["gid"], log_dir=spec.get("log_dir"))
    os.makedirs(spec["scratch"], exist_ok=True)
    if spec.get("log_dir"):
        (Path(spec["log_dir"]) / f"{spec['gid'].replace(':', '_')}.jsonl.gz").unlink(missing_ok=True)       # a game played again starts its log again
    with open(Path(spec["scratch"]) / f"hang_{os.getpid()}.txt", "w") as hang:
        faulthandler.dump_traceback_later(WATCHDOG, exit=True, file=hang)           # a game that takes this long is a hang (a game takes tens of seconds): its traceback is kept and the worker ends
        g = OM.play_game(spec)
        faulthandler.cancel_dump_traceback_later()
    g["stats"] = [dict(m.stats) for m in TM.REG]
    seats = [s for s in (0, 1) if spec["players"][s]["kind"] == KIND_WAKE]
    if spec.get("log_dir") and seats:
        rec = {"kind": "result", "gid": g["gid"], "colour": g["colour"], "outcome": g["outcome"], "plies": g["plies"], "unfinished": g["unfinished"], "moves": g["moves"], "wake_seat": seats}
        with gzip.open(Path(spec["log_dir"]) / f"{g['gid'].replace(':', '_')}.jsonl.gz", "ab") as f:
            f.write((json.dumps(rec) + "\n").encode())
    return g


# ------------------------------------------------------------------------------------------------------------------------------ the players and the matches

def player_spec(name: str, seconds: "float | None" = None, chooser: "dict | None" = None) -> dict:
    spec = {"kind": KIND_WAKE if name == "wake" else KIND_OLD, "worth": RP.worth_of("human"), "charge": False}
    if seconds is not None:
        spec["seconds"] = float(seconds)
    if chooser is not None:
        spec["chooser"] = chooser                                                 # PLAN 79.12 B (only the wake has a last layer)
    return spec


def match_specs(a: str, b: str, set_name: str, pairs: int, seconds: float, scratch: Path, log_dir: Path, clock_a: "float | None" = None, clock_b: "float | None" = None, label: str = "",
                chooser_a: "dict | None" = None, chooser_b: "dict | None" = None, openings: "list | None" = None) -> list:
    """`chooser_a`, `chooser_b` (PLAN 79.12): the last layer of a wake player; `openings` [(seed, moves)]: given in place of the named set (the spec's start is then `set_name` + the seed)."""
    pa, pb = player_spec(a, clock_a, chooser_a), player_spec(b, clock_b, chooser_b)
    specs = []
    for seed, opening in (openings if openings is not None else TM.openings(set_name, pairs)):
        for colour in (0, 1):
            specs.append({"gid": f"{label or a + '_v_' + b}:{set_name}{seed}:{colour}", "start": f"{set_name}{seed}", "colour": colour, "text": None, "opening": opening,
                          "players": [pa, pb] if colour == 0 else [pb, pa], "seconds": seconds, "max_plies": TM.PLY_CAP, "scratch": str(scratch), "log_dir": str(log_dir)})
    return specs


def mean_wall(games: list, who: int) -> float:
    """The mean wall time a move of side `who` (0: a, 1: b): the seconds of its decisions over their count (the whole call of the player, whatever it did inside)."""
    wall = dec = 0
    for g in games:
        st = g["stats"][TM.seat_of(g, who)]
        wall += st["wall"]
        dec += st["decisions"]
    return wall / max(1, dec)


def scale_of(games: list) -> float:
    """PART C: the one scale of the equal wall: the wake's mean wall a move over the old player's, read on a sanity match (a: the wake, b: the old player, at one clock)."""
    return mean_wall(games, 0) / max(1e-9, mean_wall(games, 1))


def differed(games: list, who: int) -> dict:
    """What the wake did differently from the search alone, over the games: {"decisions", "differs", reason: count of the differing decisions, ...}."""
    out = {"decisions": 0, "differs": 0, **{r: 0 for r in REASONS}}
    for g in games:
        st = g["stats"][TM.seat_of(g, who)]
        out["decisions"] += st["decisions"]
        out["differs"] += st.get("differs", 0)
        for r in REASONS:
            out[r] += st.get("by_" + r, 0)
    return out


def games_file(tag: str, name: str, set_name: str) -> Path:
    return out_root(tag) / "matches" / f"{name}_{set_name}.jsonl"


def log_count(log_dir: Path) -> int:
    """The decisions logged in a folder of game logs."""
    n = 0
    for p in sorted(Path(log_dir).glob("*.jsonl.gz")):
        with gzip.open(p, "rt") as f:
            n += sum(1 for x in f if '"kind": "decision"' in x[:20])
    return n


def lines_of(name: str, a: str, b: str, set_name: str, games: list, load0: float, load1: float, busy: float, wall: float, seconds: float, workers: int, clocks: dict, log_dir: Path) -> list:
    out = [x.replace("OBS tiebreak match", f"OBS whole {name}", 1) for x in TM.match_lines(a, b, set_name, games, load0, busy, wall, seconds, workers)]
    tag = f"OBS whole {name} {a} v {b} [set {set_name}]"
    out.append(f"{tag}: mean wall a move: {a} {mean_wall(games, 0):.4f} s, {b} {mean_wall(games, 1):.4f} s; clocks given: {a} {clocks.get(a, seconds):.4f} s, {b} {clocks.get(b, seconds):.4f} s")
    for who, nm in ((0, a), (1, b)):
        if nm == "wake":
            d = differed(games, who)
            out.append(f"{tag}: the wake's move differed from the search's first move in {d['differs']} of {d['decisions']} decisions ({100 * d['differs'] / max(1, d['decisions']):.2f} pct); by the reason given: " + ", ".join(f"{r} {d[r]}" for r in REASONS if d[r]))
    out.append(f"{tag}: decisions logged {log_count(log_dir)} in {log_dir}")
    return out


def cmd_match(name: str, a: str, b: str, set_name: str, pairs: int, seconds: float, workers: int, tag: str, say, clock_b: "float | None" = None, stop: bool = True) -> list:
    root = out_root(tag)
    log_dir = root / "logs" / f"{name}_{set_name}"
    (root / "matches").mkdir(parents=True, exist_ok=True)
    say(f"OBS whole {name} {a} v {b} [set {set_name}]: the early stop is asked with min_pairs={MIN_PAIRS} (99 pct band on whole opening pairs, clear of a half or within 0.03 of it); the human-result record on both sides; clock {seconds} s a move for {a}"
        + (f", {clock_b:.4f} s for {b} (the scale of equal wall)" if clock_b else f" and {b}") + f"; ledger share {TM.LEDGER_SHARE}, ply cap {TM.PLY_CAP}, workers {workers}")
    load0, busy = TM.wait_quiet(say)
    t0 = time.perf_counter()
    specs = match_specs(a, b, set_name, pairs, seconds, root / "scratch", log_dir, None, clock_b, label=f"{name}_{a}_v_{b}")
    games = TM.run_games(specs, workers, games_file(tag, name, set_name), say, play=play11, stop=(lambda g: TM.settled(g, min_pairs=MIN_PAIRS)) if stop else None)
    wall = time.perf_counter() - t0
    for x in lines_of(name, a, b, set_name, games, load0, os.getloadavg()[0], busy, wall, seconds, workers, {b: clock_b} if clock_b else {}, log_dir):
        say(x)
    return games


def scale_path(tag: str) -> Path:
    return out_root(tag) / "scale.json"


def cmd_sanity(pairs: int, seconds: float, workers: int, tag: str, say) -> float:
    """PART C: the wake against the old player at one clock, 20 games (10 opening pairs of set A, no early stop); the scale of equal wall is read from it, printed and kept."""
    games = cmd_match("sanity", "wake", "old", "A", pairs, seconds, workers, tag, say, stop=False)
    s = scale_of(games)
    scale_path(tag).write_text(json.dumps({"scale": s, "seconds": seconds, "wake_wall": mean_wall(games, 0), "old_wall": mean_wall(games, 1), "games": len(games)}))
    say(f"OBS whole sanity: ONE SCALE of equal wall = {s:.4f} (the wake's mean wall a move {mean_wall(games, 0):.4f} s over the old player's {mean_wall(games, 1):.4f} s at the clock {seconds} s, {len(games)} games); the old player is given {s * seconds:.4f} s for equal mean wall")
    return s


def main(argv: "list | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stage", choices=("sanity", "match"))
    ap.add_argument("--name", default="m1", choices=("m1", "m2"))
    ap.add_argument("--set", default="A", choices=("A", "B"))
    ap.add_argument("--pairs", type=int, default=100)
    ap.add_argument("--seconds", type=float, default=TM.SECONDS)
    ap.add_argument("--workers", type=int, default=9)
    ap.add_argument("--tag", default="full")
    a = ap.parse_args(argv)
    root = out_root(a.tag)
    root.mkdir(parents=True, exist_ok=True)

    def say(line: str) -> None:
        TM.emit([line], root / "obs.txt")
    if a.stage == "sanity":
        cmd_sanity(10, a.seconds, a.workers, a.tag, say)
    else:
        clock_b = None
        if a.name == "m1":
            clock_b = json.loads(scale_path(a.tag).read_text())["scale"] * a.seconds
        cmd_match(a.name, "wake", "old", a.set, a.pairs, a.seconds, a.workers, a.tag, say, clock_b=clock_b)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
