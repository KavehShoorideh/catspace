"""PLAN 78.68 STAGE G (parts 1 to 3) AND STAGE H AS THE SCORE. SYSTEM_SPEC 9 ("so the heads learn from our own deeper thinking", "a rule is scored by how many surprises it resolves") and 6 ("the estimate scores a
leaf only when the reasoning has ended", "its sources, in order of preference": results of play, then our own deeper reasoning). A sandbox: no player is changed, nothing is written to rules/learned, data/derived or
any store. Game-free: a move, a kind, a role is a word that arrives as data; no object of any game is named here. No oracle number, move or label enters a row or a fit; no outside model is called.

INTERFACE (what this module takes, what it returns, who calls it).
  pairs(positions, wake_clock, deep_clock, tools) -> rows
      positions    [{"id", "source", "band", "units", "moves"}]: the moves played from the start. NO answer of any oracle is in a position and none is read here.
      wake_clock   {"depth", "fuse"}: the clock of the wake's own reading (the line read of 78.72, mode 8, the best line of the mover). Its line's SETTLED END is THE LEAF.
      deep_clock   the same keys, many times the wake's: the deep reading, started FROM THE LEAF.
      tools        DeepTools (below), or anything with the same methods: leaf_of, inputs, quiet, mover_of, ended, read_best, target_of, playouts.
      returns      one row per (leaf, target): {"id", "pid", "source", "band", "units", "leaf_ply", "quiet", "x" (the head inputs AT THE LEAF: the columns of the heads of 78.65 and 78.67, read by the same code),
                   "mover" (the leaf's mover), "made_by" ("deep" | "playout"), "anchor" (the record reached an end of the game), "rec" (plies of the record),
                   "target" {"us": [D, has, cut], "them": [D, has, cut]} (the hazard records of the class "keeps" of 78.64: the steps from the leaf to the event of that side, whether it happened, whether a taking with no
                   settled position after it cut the record), "kept" {"us": [kinds], "them": [kinds]} (the units won and kept along the record, by kind)}.
      THE TARGET comes from our own tools only. Where the deep reading from the leaf is not cut by its fuse it ends settled, and the target is what happened along its line. Where it is cut, playouts by the current player
      (the 78.67 control's search, a random one of the moves it holds equal) from the leaf give the target: each playout is a row. A fifth of the settled leaves also get playouts, so that rows of both kinds exist to be
      compared (a choice of this build, printed). An ANCHOR is a row whose record reached an end of the game (the rules' own outcome); no proved lemma is read here yet.
      Every record is cut to RECORD_PLIES plies from the leaf, so that the deep and the playout rows span the same steps.
  fit(rows, template, extra=None, ...) -> heads
      THE SAME HEAD FORMS as 78.65 and 78.67: for each side the discrete-hazard linear head (events_hazard.fit_best) and the boosted-trees head (frames_heads.fit_tree) of the class "keeps", on the same input columns.
      `template` is the pickle of the 78.67 heads fitted on human games (columns, blocks and constants are taken from it). `extra`: (x, D, has, cut, group) rows of another source to fit beside these (the human games). Returns the
      pickle's own layout, plus "horizon" (the plies the records span: the race is read no further than the data goes) and "source".
  leaf_pick(position, tied, heads, tools, kind) -> move
      For each move the search holds equal, read OUR best line after it to its settled end (read_lines mode 8) and score the head AT THAT LEAF: the estimate where the reasoning ends. Where the line ends the game the
      end itself is the score (concrete first). Returns the move with the best score for the mover (a tie goes to the search's own move). root_pick: the head read at the root, as 78.67 did, for comparison.
  prepare(position, tools) -> prep      the cheap rungs of the ladder (search at the wake's clock, forcing read) and the inputs of every tied move at the root and at the leaf. Reads NO answer.
  score(candidate, preps, open_ids, resolved_ids, answers, split) -> {"resolved", "created", "cost", ...}   STAGE H: rungs 1 and 2 of the ladder rerun with the candidate as the head on (a) the open surprises and (b) an equal
      number of positions the ladder already resolved at rungs 1 or 2; counted only on the scored part of the split. THE ANSWERS FILE IS READ ONLY HERE (and in the measure script that hands it in).
  split_part(pid) -> "scored" | "fit"   the split of the ladder's 400 positions, by game: a stable hash of the game's key; the scored part is never used for pairs (nor is any game that appears in the 400).
Called by: deep_heads_measure.py (the run), tests/test_plan78_68_deep_heads.py. Nothing in play calls it.
"""
from __future__ import annotations

from catspace.incubator.analyst.discovery import tiebreak_measure as TM          # first: it sets one thread a process before numpy is loaded

import hashlib
import json
import math
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import cs_engine as CE

from catspace.incubator.analyst.discovery import events_hazard as EH
from catspace.incubator.analyst.discovery import events_rules as ER
from catspace.incubator.analyst.discovery import frames_heads as FH
from catspace.incubator.analyst.discovery import graph_layer1 as GL
from catspace.incubator.analyst.discovery import graph_layer2_measure as G2M
from catspace.incubator.analyst.discovery import human_heads_data as HD
from catspace.incubator.analyst.discovery import ladder_measure as LM
from catspace.incubator.analyst.discovery import odds_measure as OM
from catspace.incubator.analyst.discovery import tactical_tables as TT
from catspace.incubator.analyst.discovery import threat_concrete as TC
from catspace.io import paths

OUT = ("deep_heads_78_68",)
CORE = ("keeps",)
RECORD_PLIES = 12                                                   # the plies from the leaf every record spans (a stated choice: the deep reading's depth)
WAKE_CLOCK = {"depth": TC.DEPTH, "fuse": TC.FUSE}                   # the wake's own reading (78.72's line read)
DEEP_CLOCK = {"depth": 12, "fuse": TC.FUSE * 16}                    # sixteen times the wake's fuse and twice its depth (a stated choice of this run)
PLAYOUTS = 3                                                        # playouts a leaf gets where the deep reading is cut
PLAYOUT_SECONDS = 0.05                                              # the clock of each move of a playout (as the odds games of 78.61)
CONTROL_EVERY = 5                                                   # one settled leaf in this many also gets playouts (so that rows of both kinds exist)
BIG = 1e6                                                           # the score of a line that ends the game (concrete first)
WAKE_SECONDS = LM.LIMITS["wake_seconds"]


def game_key(pid: str) -> str:
    """The game a position id belongs to (the id less its last field, the ply)."""
    return pid.rsplit(":", 1)[0]


def part_of_key(key: str) -> str:
    return "scored" if int(hashlib.md5(key.encode()).hexdigest(), 16) % 2 == 0 else "fit"


def split_part(pid: str) -> str:
    """THE SPLIT of the ladder's 400 positions: by game. A game whose hash is even is the SCORED part (stage H counts there); the others are the part a candidate may have been made from."""
    return part_of_key(game_key(pid))


# ------------------------------------------------------------------------------------------------------------------------------ the tools

class DeepTools(LM.Tools):
    """The readers of OUR OWN reasoning the pairs and the picks use (the ladder's Tools, plus the leaf, the inputs at a position, the record's events and the playouts). One per process."""

    def __init__(self, scratch: "str | Path | None" = None):
        super().__init__()
        self.reader = TM.reader()
        self.scratch = str(scratch or Path(paths.experiment(*OUT)) / "scratch")
        self.skipped: Counter = Counter()
        self._ev = None
        self._base = None
        self._chain = None

    # -- the line readings
    def read_best(self, moves: list, clock: dict) -> "dict | None":
        r = self.lines.ask([(None, list(moves))], [(0, 8, 0, 1, 0, 0, 0)], clock["depth"], clock["fuse"])[0]
        if not r.get("ok") or not r.get("reads"):
            return None
        return r["reads"][0]

    def leaf_of(self, moves: list, clock: dict) -> "dict | None":
        """The settled end of the mover's best line at `clock`: {"leaf": the moves up to it, "end": how the line ended (0 quiet, 1 the mover wins the game, 2 it loses), "cut", "names"}."""
        rd = self.read_best(moves, clock)
        if rd is None:
            return None
        k = int(rd["settled"])
        return {"leaf": list(moves) + list(rd["names"][:k]), "end": int(rd["end"]), "cut": bool(rd["cut"]), "names": list(rd["names"]), "nodes": int(rd["nodes"])}

    # -- the head inputs at a position (the same columns as the fit of 78.65, read by the same code as 78.67)
    def inputs(self, moves: list) -> "np.ndarray | None":
        R = self.reader
        res = R.sheet.subgoal_games([(None, list(moves))], 1)[0]
        if not res:
            return None
        sts = np.array(res[-1], np.float32)
        ns = R.n_statements
        try:
            x, keep = R.matrix([(None, list(moves))], (sts[:ns] - sts[ns:])[None, :])
        except IndexError:                                             # the frame read has no row for this position
            return None
        return x[0] if len(keep) else None

    def quiet(self, moves: list) -> bool:
        return bool(self.reader.kinds_part([(None, list(moves))])[1][0])

    def mover_of(self, moves: list) -> str:
        return CE.state(OM.VARIANT, None, list(moves))[1]

    def ended(self, moves: list) -> bool:
        """The rules call the position after `moves` over."""
        return CE.state(OM.VARIANT, None, list(moves))[3] is not None

    # -- the events of a record
    def _events_init(self) -> None:
        if self._ev is not None:
            return
        spec, vocab = ER.spec_and_vocab()
        clauses = ER.rule_clauses(spec)
        roles = ER.role_words(clauses)
        meta = json.loads(Path(paths.experiment("tactical_78_62", "state", "state.json")).read_text())
        edge = dict(zip(meta["places"], TT.edge_distances(meta["places"]).tolist()))
        ER._init_worker(spec, vocab, ER.persistent_fluents(clauses), ER.load_worth(), roles, edge)
        self._ev = {"roles": roles, "layer1": GL.model(HD.GAME, HD.VARIANT), "goal": G2M.goal_kind_of(HD.VARIANT)}

    def target_of(self, moves: list, leaf_ply: int) -> dict:
        """The hazard records of the class `CORE` for the two sides at the position after `moves[:leaf_ply]`, read with the code of 78.64 on the moves of the whole record: {"us": [D, has, cut], "them": [...], "kept": {...}}."""
        self._events_init()
        ev = self._ev
        r = ER.read_game((0, list(moves), TM.quiet_plies(ev["layer1"], ev["goal"], list(moves)), None))
        mover = self.mover_of(moves[:leaf_ply])
        roles = ev["roles"]
        rows = {"g": np.array([0]), "ply": np.array([leaf_ply]), "mover": np.array([roles.index(mover)], np.int8)}
        res = [{"n": len(moves), "events": {}, "kept": r["kept"], "cut": r["cut"]}]
        out: dict = {}
        for rel in ("us", "them"):
            D, has, cut = EH.targets(CORE, rel, roles, rows, res)
            out[rel] = [int(D[0]), bool(has[0]), bool(cut[0])]
        kept = {"us": [], "them": []}
        for key, steps in r["kept"].items():
            if len(key) == 3 and key[0] == CORE[0]:
                side = "us" if key[1] == mover else "them"
                kept[side] += [key[2]] * sum(1 for t in steps if t >= leaf_ply)
        out["kept"] = kept
        return out

    # -- the playouts of the current player (the 78.67 control's search; a random one of the moves it holds equal, since the control alone would play the same game each time)
    def playouts(self, leaf: list, gid: str, k: int, seconds: float, plies: int) -> list:
        if self._base is None:
            self._base = OM.players_of(TM.WORTH_GEN, OM.out_root() / "full")
        spec_p = TM.player_spec("random", self._base)
        out = []
        for j in range(k):
            spec = {"gid": f"{gid}:p{j}", "start": None, "colour": 0, "text": None, "opening": list(leaf), "players": [spec_p, spec_p], "seconds": seconds, "max_plies": len(leaf) + plies, "scratch": self.scratch}
            g = TM.play67(spec)
            out.append({"moves": g["moves"], "ended": g["outcome"] is not None})
        return out


# ------------------------------------------------------------------------------------------------------------------------------ the pairs

def _row(pid: str, j: int, pos: dict, leaf: list, x, quiet: bool, made_by: str, anchor: bool, rec: int, t: dict, mover: str) -> dict:
    return {"id": f"{pid}#{j}", "pid": pid, "source": pos.get("source"), "band": pos.get("band"), "units": pos.get("units"), "leaf_ply": len(leaf), "quiet": bool(quiet),
            "x": [float(v) for v in x], "mover": mover, "made_by": made_by, "anchor": bool(anchor), "rec": int(rec), "target": {"us": t["us"], "them": t["them"]}, "kept": t["kept"]}


def pairs(positions: list, wake_clock: dict, deep_clock: dict, tools, record: int = RECORD_PLIES, playouts: int = PLAYOUTS, control_every: int = CONTROL_EVERY,
          playout_seconds: float = PLAYOUT_SECONDS) -> list:
    rows: list = []
    for pos in positions:
        moves, pid = list(pos["moves"]), pos["id"]
        w = tools.leaf_of(moves, wake_clock)
        if w is None:
            tools.skipped["no wake reading"] += 1
            continue
        if w["end"] != 0:
            tools.skipped["the wake's line ends the game (no inputs at its end)"] += 1
            continue
        leaf = w["leaf"]
        x = tools.inputs(leaf)
        if x is None:
            tools.skipped["no inputs at the leaf"] += 1
            continue
        mover = tools.mover_of(leaf)
        quiet = tools.quiet(leaf)
        d = tools.read_best(leaf, deep_clock)
        n0 = len(rows)
        if d is not None and not d["cut"]:
            full = leaf + list(d["names"][:record])
            ended = tools.ended(full)
            rows.append(_row(pid, len(rows) - n0, pos, leaf, x, quiet, "deep", ended, len(full) - len(leaf), tools.target_of(full, len(leaf)), mover))
        want_play = d is None or d["cut"] or int(hashlib.md5(pid.encode()).hexdigest(), 16) % control_every == 0
        if want_play:
            for g in tools.playouts(leaf, pid, playouts, playout_seconds, record):
                m = g["moves"]
                rows.append(_row(pid, len(rows) - n0, pos, leaf, x, quiet, "playout", g["ended"], len(m) - len(leaf), tools.target_of(m, len(leaf)), mover))
    return rows


# ------------------------------------------------------------------------------------------------------------------------------ the fit

def arrays_of(rows: list) -> dict:
    """The arrays the head code takes: x, the records D, has, cut per side, the group (a position) of each row."""
    gid = {p: i for i, p in enumerate(dict.fromkeys(r["pid"] for r in rows))}
    out = {"x": np.array([r["x"] for r in rows], np.float32), "g": np.array([gid[r["pid"]] for r in rows], np.int32)}
    for rel in ("us", "them"):
        t = np.array([r["target"][rel] for r in rows], np.int64).reshape(-1, 3)
        out[rel] = (t[:, 0], t[:, 1].astype(bool), t[:, 2].astype(bool))
    return out


def fit(rows: "list | None", template: dict, extra: "tuple | None" = None, source: str = "deep", arrays: "dict | None" = None, say=print) -> dict:
    """The heads of the class `CORE` on these rows (and `extra` = (x, {"us": (D, has, cut), "them": ...}, g) beside them): the linear and the trees head of each side, as 78.67 fitted them."""
    a = arrays or arrays_of(rows)
    x, g = a["x"], a["g"]
    recs = {rel: a[rel] for rel in ("us", "them")}
    if extra is not None:
        xe, re_, ge = extra
        x = np.vstack([x, xe])
        g = np.concatenate([g, ge + (g.max() + 1 if len(g) else 0)])
        recs = {rel: tuple(np.concatenate([recs[rel][i], re_[rel][i]]) for i in range(3)) for rel in ("us", "them")}
    out = {k: v for k, v in template.items() if k not in ("linear", "trees", "active")}
    out.update({"linear": {}, "trees": {}, "active": {}, "horizon": RECORD_PLIES if extra is None else None, "source": source})
    sel = np.arange(len(x))
    for rel in ("us", "them"):
        h = EH.Head(CORE, rel, *recs[rel])
        h.activate(np.ones(len(x), bool))
        out["active"][rel] = [bool(v) for v in h.active]
        t0 = time.perf_counter()
        out["linear"][rel] = EH.fit_best(h, x, {"g": g}, sel, None)
        out["trees"][rel] = FH.fit_tree(h, x, {"g": g}, sel, None)
        say(f"FIT {source} keeps [{rel}]: rows {len(x)}, trees {out['trees'][rel]['n_iter']} rounds, wall {time.perf_counter() - t0:.0f} s")
    return out


def loss_rows(heads: dict, a: dict, sel: np.ndarray) -> dict:
    """The held-out log loss of every head form of `heads` on the rows `sel` of `a`, per row: {"linear": nll per row, "trees": ..., "steps": the steps of each row in the risk set}, the two sides added. Every
    head is read on the same risk set (that of the rows themselves), so the heads are comparable."""
    out = {"linear": np.zeros(len(sel)), "trees": np.zeros(len(sel)), "steps": np.zeros(len(sel))}
    loc = np.arange(len(sel))
    allx = a["x"][sel]
    for rel in ("us", "them"):
        h = EH.Head(CORE, rel, *(v[sel] for v in a[rel]))
        h.activate(np.ones(len(sel), bool))
        for kind in ("linear", "trees"):
            P = FH.table_linear(heads["linear"][rel], allx, loc) if kind == "linear" else FH.table_tree(heads["trees"][rel], allx, loc)
            out[kind] += FH.nll_table(h, P, loc)[0]
        out["steps"] += h.steps().astype(np.float64)
    return out


# ------------------------------------------------------------------------------------------------------------------------------ the number of a move and the picks

def race_number(heads: dict, kind: str, x: np.ndarray, remaining: int) -> float:
    """At a position: the chance that the side to move there wins and keeps a unit first, less the other side's (the race of 78.65 E), by the head form `kind`, read no further than the heads' own horizon."""
    sel = np.arange(len(x))
    hz = heads.get("horizon")
    rem = np.full(len(x), max(1, int(remaining if hz is None else min(remaining, hz))))
    P = {rel: (FH.table_tree(heads["trees"][rel], x, sel) if kind == "trees" else FH.table_linear(heads["linear"][rel], x, sel)) for rel in ("us", "them")}
    pu, pt = FH.race_table(SimpleNamespace(active=heads["active"]["us"]), SimpleNamespace(active=heads["active"]["them"]), P["us"], P["them"], rem)
    return float((pu - pt)[0])


def prepare(position: dict, tools) -> dict:
    """The cheap rungs and the inputs of the tied moves, for one position. Reads no answer. {"eq", "move", "forced", "root": {move: x}, "leaf": {move: {"x", "end", "plies"}}, "t_root", "t_leaf", "t_cheap"}."""
    moves = list(position["moves"])
    t0 = time.perf_counter()
    s = tools.search(position, WAKE_SECONDS)
    eq = list(s["equals"])
    f = tools.forced(position)
    prep = {"eq": eq, "move": s["move"], "forced": f, "root": {}, "leaf": {}, "t_cheap": time.perf_counter() - t0, "t_root": 0.0, "t_leaf": 0.0}
    if len(eq) < 2:
        return prep
    legal = CE.state(OM.VARIANT, None, moves)[2]
    t1 = time.perf_counter()
    stm = tools.reader.statements_children(moves, legal[0])
    names = list(eq)
    positions = [(None, moves + [nm]) for nm in names]
    sts = np.array([stm.get(legal.index(nm), np.zeros(tools.reader.n_statements, np.float32)) for nm in names], np.float32)
    try:
        x, keep = tools.reader.matrix(positions, sts)
    except IndexError:                                                 # no position of the set has a frame read row
        x, keep = np.zeros((0, 0), np.float32), []
    for k, i in enumerate(keep):
        prep["root"][names[int(i)]] = x[k]
    prep["t_root"] = time.perf_counter() - t1
    t2 = time.perf_counter()
    for nm in names:
        after = moves + [nm]
        st = CE.state(OM.VARIANT, None, after)
        if st[3] is not None:                                         # the move ends the game: the rules' outcome is the score
            prep["leaf"][nm] = {"x": None, "end": -1, "outcome": [int(v) for v in st[3]], "plies": 0}
            continue
        w = tools.leaf_of(after, WAKE_CLOCK)
        if w is None:
            continue
        if w["end"] != 0:
            prep["leaf"][nm] = {"x": None, "end": w["end"], "plies": len(w["leaf"]) - len(moves), "line": w["leaf"][len(moves):]}
            continue
        prep["leaf"][nm] = {"x": tools.inputs(w["leaf"]), "end": 0, "plies": len(w["leaf"]) - len(moves), "line": w["leaf"][len(moves):]}
    prep["t_leaf"] = time.perf_counter() - t2
    return prep


def root_pick(position: dict, tied: list, heads: dict, tools_or_prep, kind: str = "trees") -> str:
    """The move whose position after it has the best number, read at the root (78.67). `tools_or_prep`: a prep of `prepare` (or any object with the same keys)."""
    prep = tools_or_prep
    n = len(position["moves"])
    nums = {m: -race_number(heads, kind, prep["root"][m][None, :], TM.PLY_CAP - (n + 1)) for m in tied if m in prep["root"]}
    return TM.pick_best(prep["move"], nums)


def leaf_pick(position: dict, tied: list, heads: dict, tools_or_prep, kind: str = "trees") -> str:
    """The move whose best line after it ends at the leaf the head scores best for the mover (the estimate where the reasoning ends); a line that ends the game scores by the end itself."""
    prep = tools_or_prep
    n = len(position["moves"])
    nums = {}
    for m in tied:
        lf = prep["leaf"].get(m)
        if lf is None:
            continue
        if lf["end"] == -1:
            out = lf["outcome"]
            nums[m] = BIG * (1 if out[0] > out[1] else (-1 if out[0] < out[1] else 0)) * (1 if n % 2 == 0 else -1)
        elif lf["end"] == 1:                                          # the side to move after the move wins by the line: bad for the mover
            nums[m] = -BIG
        elif lf["end"] == 2:
            nums[m] = BIG
        elif lf["x"] is not None:
            at = n + lf["plies"]
            sign = 1.0 if (at - n) % 2 == 0 else -1.0                # the side to move at the leaf, against the mover of the position
            nums[m] = sign * race_number(heads, kind, lf["x"][None, :], TM.PLY_CAP - at)
    return TM.pick_best(prep["move"], nums)


# ------------------------------------------------------------------------------------------------------------------------------ stage H: the score of a candidate

def wilson(k: float, n: int, z: float = 1.645) -> "tuple[float, float]":
    """A 90 pct Wilson interval of a share k of n."""
    if n <= 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (0.0 if k <= 0 else c - h), (1.0 if k >= n else c + h)


def rung_under(prep: dict, acc: set, pick) -> int:
    """The rung (1, 2, or 3 for 'still open at the cheap rungs') the position reaches with `pick(prep) -> move` as the head of rung 2. The ladder's own rules (ladder.py): rung 1 if every move the search holds equal is
    accepted or the forcing read proves an accepted first move; rung 2 if two or more are held equal, an accepted one is among them, and the head picks an accepted one."""
    eq = prep["eq"]
    if eq and set(eq) <= acc:
        return 1
    f = prep["forced"]
    if f.get("kind") == "proved" and f.get("first") in acc:
        return 1
    if len(eq) >= 2 and acc & set(eq) and pick(prep) in acc:
        return 2
    return 3


def tie_eligible(prep: dict, acc: set) -> bool:
    return len(prep["eq"]) >= 2 and bool(acc & set(prep["eq"])) and not set(prep["eq"]) <= acc


def score(candidate: dict, preps: dict, open_ids: list, resolved_ids: list, answers: dict, split: str = "scored", seed: int = 6868) -> dict:
    """STAGE H. `candidate`: {"name", "pick": f(position id, prep) -> move, "cost": f(prep) -> seconds, "random": bool}. `preps`: {id: prep}. `open_ids`: the ids on the list of open surprises; `resolved_ids`: the ids the
    ladder resolved at rungs 1 or 2. Both are cut to the part `split` of the split of the ladder's 400, and the resolved ones are drawn down to as many as the open ones (seeded). Returns the counts and the cost."""
    op = [i for i in open_ids if split_part(i) == split and i in preps]
    rs = [i for i in resolved_ids if split_part(i) == split and i in preps]
    rng = np.random.default_rng(seed)
    sample = [rs[k] for k in sorted(rng.choice(len(rs), min(len(op), len(rs)), replace=False))] if rs else []

    def one(pid: str) -> "tuple[float, float]":
        """(rung at most 2: 1 or 0, expected for a random pick) under the candidate."""
        prep, acc = preps[pid], set(answers[pid])
        if candidate.get("random"):
            r1 = rung_under(prep, acc, lambda p: None) == 1
            if r1:
                return 1.0, 1.0
            ok = len(prep["eq"]) >= 2 and bool(acc & set(prep["eq"]))
            return (len(acc & set(prep["eq"])) / len(prep["eq"]) if ok else 0.0), 0.0
        return float(rung_under(prep, acc, lambda p: candidate["pick"](pid, p)) <= 2), 0.0

    res = [one(i)[0] for i in op]
    keep = [one(i)[0] for i in sample]
    allr = [i for i in rs]
    keep_all = [one(i)[0] for i in allr]
    cost = [candidate["cost"](preps[i]) for i in op + sample]
    return {"name": candidate["name"], "n_open": len(op), "n_resolved_sample": len(sample), "n_resolved_all": len(allr), "resolved": float(sum(res)), "created": float(len(sample) - sum(keep)),
            "created_all": float(len(allr) - sum(keep_all)), "cost": float(np.mean(cost)) if cost else float("nan"),
            "res_vec": res, "keep_vec": keep}                              # per position, in the same order for every candidate (the same seeded draw): for paired intervals


def hit_rate(candidate: dict, preps: dict, ids: list, answers: dict, split: str = "scored") -> dict:
    """MEASURED, NOT FITTED, against the oracle's accepted moves: over the scored part, the positions where two or more moves are held equal, an accepted one among them and not all accepted (the ladder's ties), how
    often the candidate's pick is an accepted move. A random pick's chance is the mean share of accepted moves among the tied."""
    ids = [i for i in ids if split_part(i) == split and i in preps and tie_eligible(preps[i], set(answers[i]))]
    if candidate.get("random"):
        k = float(sum(len(set(answers[i]) & set(preps[i]["eq"])) / len(preps[i]["eq"]) for i in ids))
    else:
        k = float(sum(candidate["pick"](i, preps[i]) in set(answers[i]) for i in ids))
    lo, hi = wilson(k, len(ids))
    return {"name": candidate["name"], "n": len(ids), "hits": k, "rate": k / max(1, len(ids)), "lo": lo, "hi": hi}
