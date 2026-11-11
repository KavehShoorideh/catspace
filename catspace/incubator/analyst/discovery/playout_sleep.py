"""PLAN 78.28 PLAYOUTS TO A REAL RESULT, DIAGNOSED AT SLEEP. A Monte Carlo playout ends in a win, a draw or a loss (`cs_engine.playout_game`: its moves kept, its policy the stored
rules first, the careful look where no rule speaks). At SLEEP an outside engine (a judge object: `wdl(variant, moves, start) -> (win, draw, loss)` for the side to move, as
wdl_sleep.UciWdlJudge) says what the playout's start was worth; where the playout's result is worse, the sleep walks the playout to the ply where the verdict class dropped for
the side that moved (the TURNING move) and hands that position to the lemma sleep (sleep.sleep(extra_positions=...)), whose cards, regret read against the rank source, climb and
held-out check are unchanged. The next playouts follow the new snapshot's rules. NOTHING OF THE JUDGE enters the snapshot or anything the wake reads (the problem file below is
training data: the wake never reads it).

Game-free: the game is the `variant` string and the judge object given; no chess object is named here.

  VERDICT CLASS   win / draw / loss for the side to move, taken from a judge's (win, draw, loss) by its LARGEST share; a tie goes to the lower class (loss < draw < win), so the
                  class never rests on an optimistic tie.
  playout_games   playouts from positions with the snapshot's rules (or the careful look)
  turning         the first ply of a playout where the mover's class drops across its own move (None when the result is not worse than the start's verdict)
  match_share     the share of finished playouts whose result equals the class of their start (the cycle's measure)
  sleep_playouts  positions of a wake log -> playouts -> match_share, turning moves -> seeds for the lemma sleep (`extra_positions`)
  PROBLEMS        a position where the engine's own decision (the wake's solve_batch call, same clock and switches, with a given snapshot) picks a move after which the judge's class
                  for the mover is lower than before. `check` runs that decision; the problem file (JSON, in the sandbox next to the lemmas) keeps them with their history
                  across cycles; `problem_counts` reports solved / still open / came back.

  .venv/bin/python -m catspace.incubator.analyst.discovery.playout_sleep chess --log LOG --store SNAPSHOT --judge-engine /path/to/uci --judge-depth 12 [--judge-cache FILE]
         [--positions 20] [--per-position 2] [--rollout both|careful|rules] [--max-plies 120] [--max-turning 20] [--seed 1] [--vocabulary NAME]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import tempfile
from pathlib import Path

import cs_engine as CE

from catspace.incubator.analyst.discovery import sleep as SL
from catspace.incubator.analyst.discovery import wake as WK
from catspace.incubator.analyst.discovery import wdl_sleep as WS

CLASSES = ("loss", "draw", "win")          # lowest first
PLAYOUT_PLIES = 120                        # a guard, said when hit (the playout's `capped`)


# ------------------------------------------------------------------------------------------------------------------------------------ the verdict class

def verdict_class(wdl) -> str:
    """The class of a judge's (win, draw, loss) for the side to move: its largest share; a tie goes to the lower class."""
    w, d, l = (float(x) for x in wdl)
    best = max(w, d, l)
    return "loss" if l == best else "draw" if d == best else "win"


def rank_of(cls: str) -> int:
    return CLASSES.index(cls)


def flip(cls: str) -> str:
    """The same verdict seen by the other side."""
    return {"win": "loss", "loss": "win", "draw": "draw"}[cls]


def outcome_class(outcome, role: int) -> str:
    """The class of a finished game's result (outcome units per role, 0..100) for `role`: above a draw is a win, below a loss."""
    v = outcome[role]
    return "win" if v > 50 else "loss" if v < 50 else "draw"


class Verdicts:
    """The judge's class by position, remembered (one call per position; a judge may also keep its own disk cache). `of(start, moves)` -> the class for the side to move; `calls` counts
    the judge's calls. Terminal positions are not asked of the judge: `after(start, moves)` reads the game's own result there."""

    def __init__(self, judge, variant: str):
        self.judge, self.variant, self.memory, self.calls = judge, variant, {}, 0

    def of(self, start, moves) -> str:
        key = (start, tuple(moves))
        if key not in self.memory:
            self.calls += 1
            self.memory[key] = verdict_class(self.judge.wdl(self.variant, list(moves), start) if start is not None else self.judge.wdl(self.variant, list(moves)))
        return self.memory[key]

    def seat(self, start, moves):
        """-> (the mover's role index or None when the game is over, the outcome per role or None)."""
        _fen, mover, _legal, outcome, _key = CE.state(self.variant, start, list(moves))
        return (None if mover is None else WK.ROLES[mover]), (list(outcome) if outcome is not None else None)

    def after(self, start, moves, role: int) -> str:
        """The class, for `role`, of the position after `moves`: the game's result when it is over, else the judge's class for the side to move there seen by `role`."""
        mover, outcome = self.seat(start, moves)
        if outcome is not None:
            return outcome_class(outcome, role)
        c = self.of(start, moves)
        return c if mover == role else flip(c)


# ------------------------------------------------------------------------------------------------------------------------------------ the playout games

def _spec_text(spec, variant: str):
    return spec.get(variant) if isinstance(spec, dict) else spec


def playout_games(positions: list, *, variant: str, store: "str | None", spec, n_per_position: int = 1, seed: int = 1, rollout: str = "rules", max_plies: int = PLAYOUT_PLIES,
                  worth: "dict | None" = None, valuation: str = "") -> list:
    """One or more playouts from each position ({start, moves}) to a real result -> [{start, prefix, moves (the playout's), outcome, capped, named, rollout, seed}]. `spec`: the
    variant's description text, or the dict {variant: text} (game_store.spec_texts). `rollout` "rules" follows the snapshot at `store`'s lemmas first; "careful" ignores them."""
    text = _spec_text(spec, variant) if rollout == "rules" else None
    out = []
    for i, p in enumerate(positions):
        for j in range(n_per_position):
            sd = seed + 7919 * i + j
            r = CE.playout_game(variant, p.get("start"), list(p["moves"]), seed=sd, max_plies=max_plies, rollout=rollout, store=store if rollout == "rules" else None,
                                spec=text, worth=worth or {}, valuation=valuation)
            out.append({"start": p.get("start"), "prefix": list(p["moves"]), "moves": r["moves"], "outcome": r["outcome"], "capped": r["capped"], "named": r["named"],
                        "rollout": rollout, "seed": sd})
    return out


def result_class(game: dict, role: int) -> "str | None":
    """The class of a playout's result for `role`, or None when it did not finish."""
    return None if game["outcome"] is None else outcome_class(game["outcome"], role)


def match_share(games: list, judge, variant: str, *, verdicts: "Verdicts | None" = None) -> dict:
    """The share of FINISHED playouts whose result (for the side to move at the start) equals the judge's class of their start. -> {games, finished, capped, matched, worse, better,
    share, calls}. `share` is matched / finished (None when none finished); a capped playout has no result and is counted apart."""
    v = verdicts or Verdicts(judge, variant)
    matched = worse = better = capped = 0
    for g in games:
        if g["outcome"] is None:
            capped += 1
            continue
        role, _ = v.seat(g["start"], g["prefix"])
        if role is None:
            continue
        a, b = rank_of(v.of(g["start"], g["prefix"])), rank_of(outcome_class(g["outcome"], role))
        matched += a == b
        worse += b < a
        better += b > a
    fin = matched + worse + better
    return {"games": len(games), "finished": fin, "capped": capped, "matched": matched, "worse": worse, "better": better, "share": round(matched / fin, 4) if fin else None, "calls": v.calls}


def turning(game: dict, judge, variant: str, *, verdicts: "Verdicts | None" = None, max_scan: int = 60) -> "dict | None":
    """THE TURNING MOVE of a playout. None when the playout did not finish or its result is not worse than the start's verdict (for the side to move at the start). Otherwise the
    first ply k (an index into the playout's moves, at most `max_scan` plies are read) where the class for the side that moved at k is lower after its move than before it ->
    {ply, start, moves (the position before the move: prefix + the playout's first k), played, before, after, start_class, result_class, role}; None when the walk finds no such ply
    (a drop only across the other side's moves, or beyond `max_scan`)."""
    v = verdicts or Verdicts(judge, variant)
    if game["outcome"] is None:
        return None
    start, prefix = game["start"], list(game["prefix"])
    role0, _ = v.seat(start, prefix)
    if role0 is None:
        return None
    start_class, result = v.of(start, prefix), outcome_class(game["outcome"], role0)
    if rank_of(result) >= rank_of(start_class):
        return None
    for k in range(min(len(game["moves"]), max_scan)):
        before_moves = prefix + game["moves"][:k]
        role, outcome = v.seat(start, before_moves)
        if role is None or outcome is not None:
            return None
        b = v.of(start, before_moves)
        a = v.after(start, prefix + game["moves"][:k + 1], role)
        if rank_of(a) < rank_of(b):
            return {"ply": k, "start": start, "moves": before_moves, "played": game["moves"][k], "before": b, "after": a, "start_class": start_class, "result_class": result, "role": role}
    return None


def extra_positions_of(turnings: list, tag: str = "playout") -> list:
    """Turning moves as the lemma sleep's `extra_positions`: {id, start, moves, played, group}. One group per distinct position, so the sleep's development / held split sees them
    as it sees the games'."""
    out, seen = [], set()
    for i, t in enumerate(turnings):
        key = (t["start"], tuple(t["moves"]), t["played"])
        if key in seen:
            continue
        seen.add(key)
        out.append({"id": f"{tag}:{i}:{len(t['moves'])}", "start": t["start"], "moves": list(t["moves"]), "played": t["played"], "group": f"{tag}:{i}"})
    return out


# ------------------------------------------------------------------------------------------------------------------------------------ the positions of a wake log

def sample_positions(logs, n: int, *, seed: int = 1) -> list:
    """`n` start positions {start, moves} for the playouts from the wake log(s): the worklist of wdl_sleep.gaps (the positions where the wake lacked data or a lemma) when the log
    carries the estimate's readings, else the logged decisions; deduplicated, a seeded sample."""
    logs = WS._as_list(logs)
    work = WS.gaps(logs)["worklist"]
    pool = [{"start": w["start"], "moves": list(w["moves"])} for w in work]
    if not pool:
        for lg in logs:
            decisions, games = SL.read_log(lg)
            for (run, g, ply) in sorted(decisions):
                row = games.get((run, g))
                if row is not None:
                    pool.append({"start": row.get("start"), "moves": list(row["moves"][:ply])})
    seen, uniq = set(), []
    for p in pool:
        k = (p["start"], tuple(p["moves"]))
        if k not in seen:
            seen.add(k)
            uniq.append(p)
    rng = random.Random(seed)
    return uniq if len(uniq) <= n else sorted(rng.sample(uniq, n), key=lambda p: (str(p["start"]), p["moves"]))


# ------------------------------------------------------------------------------------------------------------------------------------ problems: the engine's own decision against the judge

def check(problems: list, *, variant: str, store: "str | None", spec, judge, seconds: float, wake_kw: "dict | None" = None, threads: int = 0, verdicts: "Verdicts | None" = None) -> list:
    """Run THE WAKE'S OWN DECISION (wake.py's `solve_batch` call: `seconds` a position, `wake_kw` the same switches -- prior, ties, reason, wdl, wdl_table; the wake of the loop uses
    ties=True) on each problem ({start, moves}) with the snapshot `store`, then ask the judge whether the chosen move keeps the verdict: the class for the mover after the move is
    not lower than before. -> one dict per problem: {move, keeps, before, after, proved, steps (a summary: ['operator:status', ...]), error}. A position with no move (the game is
    over) keeps; a decision that fails does not."""
    v = verdicts or Verdicts(judge, variant)
    if not problems:
        return []
    kw = {"ties": True, **(wake_kw or {})}
    specs = spec if isinstance(spec, dict) else {variant: spec}
    with tempfile.TemporaryDirectory() as d:
        trace = os.path.join(d, "trace.jsonl")
        rows = CE.solve_batch([(variant, p.get("start"), list(p["moves"])) for p in problems], seconds, snapshot="check", trace_path=trace, threads=threads, store=store,
                              specs=specs or {}, labels=[f"check:{i}:0:0" for i in range(len(problems))], **kw)
        steps = {}
        for line in (open(trace) if os.path.exists(trace) else []):
            if line.strip():
                r = json.loads(line)
                if r.get("kind") == "position":
                    steps[r["label"]] = [f"{s['operator']}:{s['status']}" for s in r.get("steps", [])]
    out = []
    for i, (p, (mv, proved, _nodes, _secs, err)) in enumerate(zip(problems, rows)):
        row = {"move": mv, "keeps": False, "before": None, "after": None, "proved": bool(proved), "steps": steps.get(f"check:{i}:0:0", []), "error": err}
        if mv is not None:
            role, outcome = v.seat(p.get("start"), p["moves"])
            if role is not None and outcome is None:
                row["before"] = v.of(p.get("start"), p["moves"])
                row["after"] = v.after(p.get("start"), list(p["moves"]) + [mv], role)
                row["keeps"] = rank_of(row["after"]) >= rank_of(row["before"])
        elif err is None:
            row["keeps"] = True
        out.append(row)
    return out


def load_problems(path) -> list:
    """The problem file's rows ([] when there is none): {start, moves, first_seen, played, history: [{cycle, phase, status, move, before, after, steps}]}."""
    p = Path(path)
    if not p.exists():
        return []
    d = json.loads(p.read_text())
    assert d.get("format") == "playout-problems-1", f"{p}: not a problem file"
    return d["rows"]


def save_problems(path, rows: list) -> None:
    """Write the problem file atomically. TRAINING DATA: it holds the judge's verdict classes, and neither the wake nor the snapshot writer reads it."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps({"format": "playout-problems-1", "rows": rows}, indent=1, sort_keys=True))
    os.replace(tmp, p)


def key_of(row: dict) -> tuple:
    return (row.get("start"), tuple(row["moves"]))


def merge_problems(rows: list, found: list, cycle: int) -> tuple:
    """`found` ({start, moves, played?}) added to `rows` where not already there -> (rows, the rows that are new). A new row has first_seen = `cycle` and an empty history."""
    have = {key_of(r) for r in rows}
    new = []
    for f in found:
        if key_of(f) in have:
            continue
        have.add(key_of(f))
        new.append({"start": f.get("start"), "moves": list(f["moves"]), "played": f.get("played"), "first_seen": cycle, "history": []})
    return rows + new, new


def record(rows: list, results: list, cycle: int, phase: str) -> None:
    """Append each result to its row's history: {cycle, phase ('before' / 'after' the lemma sleep), status 'solved' / 'open', move, before, after, steps}. `results` align with `rows`."""
    for r, c in zip(rows, results):
        r["history"].append({"cycle": cycle, "phase": phase, "status": "solved" if c["keeps"] else "open", "move": c["move"], "before": c["before"], "after": c["after"], "steps": c["steps"]})


def problem_counts(rows: list, cycle: int, new: int) -> dict:
    """The cycle's counts from the rows' histories (both phases recorded for this `cycle`; `new` = rows first seen this cycle and open at the before-check):
    total (all rows); new; solved_before (already kept by the current snapshot at the before-check); solved_after (open at the before-check, kept at the after-check: what the
    sleep did); still_open (open at the after-check); came_back (solved in an EARLIER cycle, open at this cycle's before-check)."""
    def at(r, phase):
        h = [x for x in r["history"] if x["cycle"] == cycle and x["phase"] == phase]
        return h[-1]["status"] if h else None
    sb = sa = so = cb = 0
    for r in rows:
        b, a = at(r, "before"), at(r, "after")
        if b == "solved":
            sb += 1
        elif b == "open" and a == "solved":
            sa += 1
        if a == "open":
            so += 1
        if b == "open" and any(x["status"] == "solved" and x["cycle"] < cycle for x in r["history"]):
            cb += 1
    return {"total": len(rows), "new": new, "solved_before": sb, "solved_after": sa, "still_open": so, "came_back": cb}


def say_counts(c: dict, say=print) -> None:
    say(f"OBS playout_sleep problems: total {c['total']}, new {c['new']}, solved before {c['solved_before']}, solved after {c['solved_after']}, still open {c['still_open']}, came back {c['came_back']}")


# ------------------------------------------------------------------------------------------------------------------------------------ the sleep

def sleep_playouts(game: str, logs, *, variant: "str | None" = None, store: str, judge, teacher=None, spec=None, rollout: str = "rules", positions: int = 20, per_position: int = 2,
                   seed: int = 1, max_plies: int = PLAYOUT_PLIES, max_turning: int = 20, max_scan: int = 60, sleep_kw: "dict | None" = None, say=print) -> dict:
    """The playout sleep over wake log(s): sample `positions` start positions (wdl_sleep.gaps' worklist), play `per_position` playouts from each with the snapshot at `store`
    (`rollout` "rules" or "careful"), measure match_share against `judge`, find the turning moves of the worse playouts (at most `max_turning` kept), and print the OBS lines. With
    `sleep_kw` ({settings, write}, and `teacher` the rank source) the turning positions are fed to the lemma sleep (sleep.sleep(extra_positions=...)) and its report is returned
    as `sleep`; without, `extra_positions` is returned for the caller to feed. -> {positions, games, match, turnings, extra_positions, seconds, [sleep]}."""
    variant = variant or game
    spec = spec if spec is not None else {}
    pos = sample_positions(logs, positions, seed=seed)
    games = playout_games(pos, variant=variant, store=store, spec=spec, n_per_position=per_position, seed=seed, rollout=rollout, max_plies=max_plies)
    v = Verdicts(judge, variant)
    m = match_share(games, judge, variant, verdicts=v)
    say(f"OBS playout_sleep match {game} rollout {rollout}: positions {len(pos)}, playouts {m['games']}, finished {m['finished']}, capped {m['capped']}, matched {m['matched']}, "
        f"worse {m['worse']}, better {m['better']}, match share {m['share']}, rules spoke in {sum(1 for g in games if g['named'])} playouts ({sum(g['named'] for g in games)} plies)")
    turns = []
    for g in games:
        if len(turns) >= max_turning:
            break
        t = turning(g, judge, variant, verdicts=v, max_scan=max_scan)
        if t is not None:
            turns.append(t)
    say(f"OBS playout_sleep turning {game} rollout {rollout}: worse playouts {m['worse']}, turning moves found {len(turns)}, judge calls {v.calls}")
    for t in turns[:5]:
        say(f"OBS playout_sleep turning move {game}: ply {t['ply']} of a playout from a prefix of {len(t['moves']) - t['ply']} moves, {t['played']} takes the mover from {t['before']} to {t['after']} "
            f"(start {t['start_class']}, result {t['result_class']})")
    rep = {"positions": pos, "games": games, "match": m, "turnings": turns, "extra_positions": extra_positions_of(turns)}
    if sleep_kw is not None:
        rep["sleep"] = SL.sleep(game, logs[0] if isinstance(logs, (list, tuple)) else logs, variant=variant, settings=sleep_kw.get("settings"), teacher=teacher, write=sleep_kw.get("write", False),
                                say=say, extra_positions=rep["extra_positions"])
    return rep


def main(argv: list) -> int:
    from catspace.incubator.analyst.discovery import game_store as GS
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("game")
    ap.add_argument("--variant", default=None)
    ap.add_argument("--log", action="append", required=True)
    ap.add_argument("--store", required=True, help="the snapshot file whose rules the playouts follow")
    ap.add_argument("--judge-engine", required=True, help="a UCI engine that reports win / draw / loss (training only)")
    ap.add_argument("--judge-depth", type=int, default=12)
    ap.add_argument("--judge-cache", default=None)
    ap.add_argument("--vocabulary", default=None)
    ap.add_argument("--positions", type=int, default=20)
    ap.add_argument("--per-position", type=int, default=2)
    ap.add_argument("--rollout", default="both", choices=["both", "careful", "rules"])
    ap.add_argument("--max-plies", type=int, default=PLAYOUT_PLIES)
    ap.add_argument("--max-turning", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args(argv)
    judge = WS.UciWdlJudge(a.judge_engine, a.judge_depth, a.judge_cache)
    try:
        for r in (["careful", "rules"] if a.rollout == "both" else [a.rollout]):
            sleep_playouts(a.game, a.log, variant=a.variant, store=a.store, judge=judge, spec=GS.spec_texts(a.game, a.vocabulary), rollout=r, positions=a.positions,
                           per_position=a.per_position, seed=a.seed, max_plies=a.max_plies, max_turning=a.max_turning)
    finally:
        judge.close()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
