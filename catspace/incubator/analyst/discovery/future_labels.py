"""PLAN 78.94 STEP TWO, THE LABELS (SYSTEM_SPEC 7 "the future imagination layer, how it is built", 9 "the oracle labels positions and frames in sleep, for the prior", 10 the data rules). A SLEEP STAGE: the only place of the future judge's files
that calls the oracle. A sandbox: nothing is written to rules/learned or data/derived; every output goes under artifacts/experiments/future_78_94/<tag>/. Game-free: the game is named only through the pieces that exist (ladder_oracle for the oracle,
threat_list / threat_concrete for the reading, symbols_measure for the games); a unit, a place or a move is a number or a word read at run time. No player id is written (a game is an ordinal of the 78.59 file or of 78.89's extra pool).

THE GAMES. The games that carry a kept position in 78.80's or 78.89's human sample (both read, never rebuilt), each game with its part FIND or CONFIRM by symbols_measure.part_of (the hash of the ordinal that 78.80 and 78.89 used, so a game is on one side in every
plan). From them a fixed number of games per part is taken in the order of a hash (the same order for every run). SAMPLED POSITIONS: every second ply of a game from ply MIN_PLY on, up to the game's last move (one side to move per game; the other side's class and goals are
read from the same position), the parity drawn by a hash of the game.

PER POSITION (the labels). (1) THE ORACLE'S CLASS for the side to move: winning, losing or neither, from the oracle's own score at a fixed setting (depth FUTURE_DEPTH, one line, one thread, the engine's default hash, the tablebase path of the repo): winning when it gives a forced
end or its score is at least WIN_CP, losing the same the other way, else neither. The raw score is stored beside the class for the record; no fit reads it (tests/test_plan78_94_future.py). (2) THE GOALS of each side: future_goals.goals_of_list on the concrete threat list read by
threat_concrete (two or more threats of one side that no one reply meets). (3) THE FRAME, at the positions where the side to move's class changes at the next sampled position of the same game (two plies on) or a goal of either side holds: the oracle's best line, read by OUR threat list and
line reading: the threats of the side that moves, at the position after its first move and after its second move of the line, as the same goal reading says them (which threats, on what kind of unit, answered or not, the stake). (4) WHAT OUR REASONER SEES at the position itself: whether the
concrete list of the side to move holds a threat (a taking, or the first quiet move of a threat in two moves) whose first move is the oracle's first move, or a forced end, that is the share the reasoner cannot see is the rest, at positions that have a frame.

INTERFACE (written before the bodies).
  universe() -> [game]               {"ord", "moves", "y", "part"} for every game of the two samples.
  choose_games(universe, per_part) -> [game]    the first `per_part` games of each part in the order of the hash `order_key`.
  plies_of(game) -> [ply]            the sampled plies.
  class_of(analysis) -> -1 | 0 | 1   the class from ladder_oracle.analyse's answer.
  Labeller()                         per process: the reader, the lines, the record; `.label(game) -> record` {"ord", "part", "y", "n_moves", "positions": [ {...} ]} (every field a number, a word or a list of them).
  run(tag, per_part, workers, say)   the Pool over all cores, one file a game under <tag>/labels/ (a game with a file is not redone), then labels.jsonl.
  stable_points(rec) -> [flags]      PLAN 78.95: per position of a label record whether its winning (or losing) class holds at the next sampled position of the same side too; from the stored classes only, no oracle call.
  Called by: future_measure.py (counts, the judge's data); the command line (`label`); future_frames.py (the stable points).

  .venv/bin/python -m catspace.incubator.analyst.discovery.future_labels label --tag short --per-part 12 | --tag full --per-part N [--workers K]
"""
from __future__ import annotations

from catspace.incubator.analyst.discovery import tiebreak_measure as TM          # first: it sets one thread a process before numpy is loaded

import argparse
import hashlib
import json
import multiprocessing
import os
import time
from pathlib import Path

from catspace.incubator.analyst.discovery import future_goals as FG
from catspace.incubator.analyst.discovery import human_heads_data as HD
from catspace.incubator.analyst.discovery import ladder_measure as LM
from catspace.incubator.analyst.discovery import ladder_oracle as LO
from catspace.incubator.analyst.discovery import symbols_measure as SM
from catspace.incubator.analyst.discovery import threat_concrete as TC
from catspace.incubator.analyst.discovery import threat_list as TL
from catspace.io import paths

OUT = ("future_78_94",)
SEED = 9494
MIN_PLY = 10
STRIDE = 2
WIN_CP = 200                                                  # the class thresholds on the oracle's own scale (a stated choice of this run)
FUTURE_DEPTH = LO.FUTURE_DEPTH
FRAME_STEPS = (1, 3, 5)                                      # plies of the oracle's line read for the frame: after its first move and after its second move for the same side
SAMPLES = (("symbols_78_80", "full"), ("symbols_78_89", "full"))
EXTRA_FILE = ("symbols_78_89", "full", "pool_extra.jsonl")


def out_root(tag: str) -> Path:
    p = Path(paths.experiment(*OUT, tag))
    p.mkdir(parents=True, exist_ok=True)
    return p


class Say:
    def __init__(self, root: Path, stage: str):
        self.root, self.stage = root, stage

    def __call__(self, msg: str) -> None:
        line = f"OBS future {self.stage}: {msg}"
        print(line, flush=True)
        with open(self.root / "obs.txt", "a") as f:
            f.write(line + "\n")


def order_key(ordinal: int) -> int:
    return int(hashlib.md5(f"future94:{ordinal}".encode()).hexdigest(), 16)


def universe() -> list:
    """Every game that carries a kept position of 78.80's or 78.89's human sample, with its moves, the result of the first mover (0 win, 1 draw, 2 loss) and its part."""
    used = set()
    for d, t in SAMPLES:
        for r in SM.jsonl(Path(paths.experiment(d, t, "sample.jsonl"))):
            used.add(int(r["game"]))
    base = {g["ord"]: g for g in SM.all_games()}
    for g in SM.jsonl(Path(paths.experiment(*EXTRA_FILE))):
        base[int(g["ord"])] = g
    return [{"ord": o, "moves": base[o]["moves"], "y": base[o]["y"], "part": SM.part_of(o)} for o in sorted(used) if o in base]


def choose_games(games: list, per_part: int) -> list:
    out = []
    for part in ("find", "confirm"):
        out += sorted((g for g in games if g["part"] == part), key=lambda g: order_key(g["ord"]))[:per_part]
    return out


def plies_of(game: dict) -> list:
    n = len(game["moves"])
    off = order_key(game["ord"]) % STRIDE
    first = MIN_PLY + ((off - MIN_PLY) % STRIDE)
    return list(range(first, n, STRIDE))


def class_of(a: dict) -> int:
    if a["forced_in"] is not None:
        return 1 if a["forced_in"] > 0 else -1
    if a["cp"] is None:
        return 0
    return 1 if a["cp"] >= WIN_CP else (-1 if a["cp"] <= -WIN_CP else 0)


def stable_points(rec: dict) -> list:
    """PLAN 78.95 PART ONE (c) (SYSTEM_SPEC 9 "the oracle labels positions and frames in sleep", with the class's edge made fair). A label's class is the oracle's score cut at WIN_CP, and a score near the edge flips a ply later (78.94 counted 3411 of 3532
    frame positions flipping), so a WINNING POINT of the side to move is STABLE when the class is still winning at the next sampled position of the same side (two plies on, the same side to move); a LOSING POINT likewise. Read from the stored classes only:
    this function never calls the oracle and reads no score. Returns, per position of the record in order, {"win": bool, "lose": bool, "stable_win": bool, "stable_lose": bool}; the last position of a game (or one whose next sampled position is not two plies on) is never stable."""
    pos = rec["positions"]
    out = []
    for i, q in enumerate(pos):
        nxt = pos[i + 1] if i + 1 < len(pos) and pos[i + 1]["ply"] == q["ply"] + STRIDE else None
        win, lose = q["cls"] == 1, q["cls"] == -1
        out.append({"win": win, "lose": lose, "stable_win": bool(win and nxt is not None and nxt["cls"] == 1), "stable_lose": bool(lose and nxt is not None and nxt["cls"] == -1)})
    return out


def word_of(pair, places: list) -> "str | None":
    if pair is None or len(pair) < 2:
        return None
    return f"{places[pair[0]]}{places[pair[1]]}"


def summary(g: dict) -> dict:
    """One side's goal reading as the label keeps it."""
    top = g["goals"][0] if g["goals"] else None
    return {"goal": bool(top), "kind": top["kind"] if top else None, "stake": top["stake"] if top else None, "shape": top["shape"] if top else None, "n_pairs": len(g["goals"]), "threats": g["threats"], "unread": g["unread"],
            "deferred": g["deferred"], "forced": g["forced"]}


class Labeller:
    """See the module doc."""

    def __init__(self, depth: int = FUTURE_DEPTH):
        self.reader = TL.ThreatReader(1)
        self.record = TC.Record.load()
        self.lines = TC.Lines(self.record, threads=1)
        self.depth = depth
        self.names = None

    def _items(self, wires: list, plies: list) -> list:
        """The concrete-list items of the positions after `plies[i]` moves of `wires[i]`."""
        res = self.reader.read([(None, list(m)) for m in wires], [[p] for p in plies])
        self.names = {k: res[k] for k in ("kinds", "roles", "places")}
        items = {}
        for r in res["rows"]:
            items[r["game"]] = {"start": None, "moves": list(wires[r["game"]][:r["ply"]]), "raw": r, "names": self.names, "list": TL.threat_rows(r, self.names, None)}
        return [items.get(i) for i in range(len(wires))]

    def label(self, game: dict) -> dict:
        moves, plies = game["moves"], plies_of(game)
        t0 = time.perf_counter()
        res = self.reader.read([(None, moves)], [plies])
        self.names = {k: res[k] for k in ("kinds", "roles", "places")}
        items, keep = [], []
        for r in res["rows"]:
            items.append({"start": None, "moves": moves[:r["ply"]], "raw": r, "names": self.names, "list": TL.threat_rows(r, self.names, None)})
            keep.append(r["ply"])
        t1 = time.perf_counter()
        gs = FG.goals_of(items, self.lines, self.record)
        t2 = time.perf_counter()
        pos = []
        for it, g, p in zip(items, gs, keep):
            a = LO.analyse(moves[:p], self.depth)
            pos.append({"ply": p, "units": len(it["raw"]["board"]), "n_us": sum(1 for o, _k, _pl in it["raw"]["board"] if o == it["raw"]["mover"]), "mover": int(it["raw"]["mover"]), "cp": a["cp"], "forced_in": a["forced_in"], "cls": class_of(a), "pv": a["pv"],
                        "mover_g": summary(g["mover"]), "other_g": summary(g["other"]), "_list": g["list"]})
        t3 = time.perf_counter()
        # the frames: where the class of the mover changes at the next sampled position, or a goal of either side holds
        want = []
        for i, q in enumerate(pos):
            nxt = pos[i + 1] if i + 1 < len(pos) and pos[i + 1]["ply"] == q["ply"] + STRIDE else None
            changed = nxt is not None and nxt["cls"] != q["cls"]
            if (changed or q["mover_g"]["goal"] or q["other_g"]["goal"]) and q["pv"]:
                q["frame_why"] = "changed" if changed else "goal"
                want.append(i)
        wires, wplies, tags = [], [], []
        for i in want:
            q = pos[i]
            q["pv"] = self.legal_prefix(moves[:q["ply"]], q["pv"])
            for j in FRAME_STEPS:
                if len(q["pv"]) >= j:
                    wires.append(moves[:q["ply"]] + q["pv"][:j])
                    wplies.append(q["ply"] + j)
                    tags.append((i, j))
        fitems = self._items(wires, wplies) if wires else []
        ok = [(t, it) for t, it in zip(tags, fitems) if it is not None]
        fg = FG.goals_of([it for _t, it in ok], self.lines, self.record) if ok else []
        for ((i, j), it), g in zip(ok, fg):
            side = "other" if (j % 2 == 1) else "mover"            # the side that moved first in the line is the other side of the position after an odd number of plies
            s = summary(g[side])
            mv = it["raw"]["mover"]
            s["n_first"] = sum(1 for o, _k, _pl in it["raw"]["board"] if (o != mv) == (j % 2 == 1))      # the units of the side that moved first in the line, and of the other, after j plies
            s["n_second"] = len(it["raw"]["board"]) - s["n_first"]
            pos[i].setdefault("frame", {})[str(j)] = s
        places = self.names["places"] if self.names else []
        for q in pos:
            c = q.pop("_list")
            if "frame_why" in q:
                q["seen"] = self.sees(c, q["pv"][0], places)
        t4 = time.perf_counter()
        return {"ord": game["ord"], "part": game["part"], "y": game["y"], "n_moves": len(moves), "positions": pos, "wall": {"read": t1 - t0, "lines": t2 - t1, "oracle": t3 - t2, "frames": t4 - t3}}

    @staticmethod
    def legal_prefix(moves: list, line: list) -> list:
        """The oracle's line cut at the first move the engine's own rules refuse (the oracle and the engine's kernel can differ on a rare move); the cut is kept in the record by the shorter line."""
        import cs_engine as CE
        out = []
        for m in line:
            if m not in CE.state(HD.VARIANT, None, list(moves) + out)[2]:
                break
            out.append(m)
        return out

    @staticmethod
    def sees(c: dict, word: str, places: list) -> dict:
        """Whether the concrete list of the side to move holds a threat of its own whose first move (a taking, or the quiet first move of a threat in two moves) is the oracle's first move, or a forced end of the game."""
        hit_take = hit_lead = forced = False
        for r in c["other"]:
            if r.get("goal"):
                forced = True
                continue
            if r["verdict"] not in ("CONFIRMED", "UNREAD"):
                continue
            w = word_of(r.get("first"), places)
            if w and word.startswith(w):
                hit_take = True
            ld = word_of(r.get("lead"), places) if isinstance(r.get("lead"), (list, tuple)) else None
            if ld and word.startswith(ld):
                hit_lead = True
        return {"take": hit_take, "lead": hit_lead, "forced": forced, "any": bool(hit_take or hit_lead or forced)}


_LAB = None


def _init_worker() -> None:
    global _LAB
    _LAB = Labeller()


def _job(args: tuple) -> tuple:
    game, root = args
    f = Path(root) / "labels" / f"g{game['ord']}.json"
    if f.exists():
        return game["ord"], 0.0
    t0 = time.perf_counter()
    try:
        rec = _LAB.label(game)
    except Exception as e:                                          # a game the reading cannot take is left out and counted, never silently: the file names the error
        (Path(root) / "labels" / f"g{game['ord']}.error").write_text(f"{type(e).__name__}: {str(e)[:300]}")
        return game["ord"], -1.0
    f.write_text(json.dumps(rec))
    return game["ord"], time.perf_counter() - t0


def run(tag: str, per_part: int, workers: int, say) -> None:
    root = out_root(tag)
    (root / "labels").mkdir(exist_ok=True)
    allg = universe()
    games = choose_games(allg, per_part)
    say(f"the games of the two human samples: {len(allg)} (find {sum(g['part'] == 'find' for g in allg)}, confirm {sum(g['part'] == 'confirm' for g in allg)}); taken: {len(games)} ({per_part} a part, in the order of a hash); sampled plies: every {STRIDE}nd from {MIN_PLY}; "
        f"the oracle: depth {FUTURE_DEPTH}, one line, one thread, default hash, the repo's tablebase path; class winning/losing at {WIN_CP} on its own scale or a forced end; {workers} workers; load average {os.getloadavg()[0]:.2f}")
    t0 = time.perf_counter()
    done = 0
    with multiprocessing.get_context("spawn").Pool(workers, initializer=_init_worker) as pool:
        for o, dt in pool.imap_unordered(_job, [(g, str(root)) for g in games], chunksize=1):
            done += 1
            if done % 20 == 0 or done == len(games):
                say(f"progress: {done} of {len(games)} games; {time.perf_counter() - t0:.0f} s; load average {os.getloadavg()[0]:.2f}")
    have = [g for g in games if (root / "labels" / f"g{g['ord']}.json").exists()]
    if len(have) < len(games):
        say(f"games the reading could not take (left out, each with an error file): {len(games) - len(have)}")
    recs = [json.loads((root / "labels" / f"g{g['ord']}.json").read_text()) for g in have]
    (root / "labels.jsonl").write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    n = sum(len(r["positions"]) for r in recs)
    w = {k: sum(r["wall"][k] for r in recs) for k in ("read", "lines", "oracle", "frames")}
    say(f"labelled {len(recs)} games, {n} positions in {time.perf_counter() - t0:.0f} s wall; one thread's seconds a position: reading {w['read'] / n:.3f}, concrete lines {w['lines'] / n:.3f}, oracle {w['oracle'] / n:.3f} (depth {FUTURE_DEPTH}), frames {w['frames'] / n:.3f}")


def main(argv: "list | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["label"])
    ap.add_argument("--tag", default="short")
    ap.add_argument("--per-part", type=int, default=12)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args(argv)
    run(a.tag, a.per_part, a.workers, Say(out_root(a.tag), "label"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
