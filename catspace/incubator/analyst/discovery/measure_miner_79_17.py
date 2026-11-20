"""PLAN 79.17 PART C: THE MINER OF THE MEASURE OF PROGRESS, IN SLEEP (SYSTEM_SPEC 7 "the small endings as the test of means and ends, and a measure of progress", 9 sleep; ledger rows 25 and 35; piece: sleep, over the measure of cs_tactics::measure).
October's miner (technique_miner.py, plans 73.5 to 73.6) was bound to the old engine and to the older library of boards; the idea is ported here, none of its code is imported: a LIBRARY of generic atoms read from the base graph (the Rust piece `cs_tactics::measure`, through the binding) composed into terms, two tests on every term cheapest first, and a term PASSES with no failing position.
  TEST 1 (cheap): the term falls along the GOOD games and does not along the BAD ones, printed as the share of the stronger side's moves after which it is lower (at that side's next turn), by set. A term is carried to test 2 only if its share in GOOD is above its share in BAD; the terms are ranked by the difference.
  TEST 2 (October's criterion, `measure_lowers`): on a sample of the class's won positions (the stronger side to move), at every position a line within d of our moves after which the term is lower against every reply exists (the search of toward.rs), or the game's end is found; d from 1 to 3. A term passes with no failing position (a question the clock cut is a gap, counted and printed, never a pass); the simplest passing term is the class's measure.
THE SETS (the sets of games are made from the oracle of the measure, which alone reads the table):
  GOOD    games that reached the end by steady play: the attackers of part A that won, and generated games in which the stronger side wins;
  BAD     games that kept the win by the table's label at every turn of the stronger side and did not reach the end: the attackers of part A that did not win, and a generated attacker that moves at random among the moves the table still calls won;
  TABLE   the table's own shortest lines, used ONLY to test October's lesson (a perfect line is a poor teacher of a measure: it takes shortcuts every measure counts as going backward), never as the teacher.
THE LIBRARY: `measure_library(groups)`: for the groups of units present, the same atoms for every class; the terms: an atom; the sum or the product of two atoms; a tuple of two or three atoms compared in order. The simplest: fewest atoms, then scalar before tuple, then the name. Nothing is put in by hand for a class.
Written to a file under artifacts/experiments/whole_79_17 only, never into the store. Printed: coverage of every sample (positions by the table's distance, both colours), the passing terms, the best failing terms with their failing positions, the TABLE set on test 1.

INTERFACE (written before the bodies).
  corpus(cls, ...) -> {"good": [record], "bad": [record], "table": [record]}      a record is {"text", "moves", "attacker_white"}
  positions_table(model, records, atoms, side_of) -> Table                         the atoms of the stronger side's turns of each record; `transitions` its consecutive pairs
  test1(table, atoms, terms) -> {term: (share_good, share_bad, share_table)}
  terms_of(atoms) -> [(text, spec)]                                                the terms of the library
  test2_job(job) -> dict                                                           one term on the sample (a Pool job: module level)
  .venv/bin/python -m catspace.incubator.analyst.discovery.measure_miner_79_17 mine --class two|one [--tag T] [--games N] [--sample N] [--top K] [--seconds S] [--workers W]
Called by: the command line, whole_79_17_print.py (reads its json), tests/test_plan79_17.py.
"""
from __future__ import annotations

from catspace.incubator.analyst.discovery import tiebreak_measure as TM          # first: it sets one thread a process before numpy is loaded

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np

from catspace.incubator.analyst.discovery import measure_oracle_79_17 as MO
from catspace.incubator.analyst.discovery import whole_79_14 as W14
from catspace.io import paths

OUT = ("whole_79_17",)
_STATE: dict = {}


def out_root(tag: str) -> Path:
    p = Path(paths.experiment(*OUT, tag))
    p.mkdir(parents=True, exist_ok=True)
    return p


def model():
    """The binding's graph of the game, built once a process."""
    if "model" not in _STATE:
        _STATE["model"] = W14.world().model
    return _STATE["model"]


# ------------------------------------------------------------------------------------------------------------------------------ the terms

def terms_of(atoms: list) -> list:
    """The terms of the library over `atoms` (names): (text, atoms read, size, kind). Atom; sum and product of two; tuple of two and of three. Ordered simplest first."""
    out = [(a, (a,), 1, "atom") for a in atoms]
    for a, b in itertools.combinations(atoms, 2):
        out.append((f"[sum {a} {b}]", (a, b), 2, "sum"))
        out.append((f"[product {a} {b}]", (a, b), 2, "product"))
    for k in (2, 3):
        for t in itertools.permutations(atoms, k):
            out.append((f"[tuple {' '.join(t)}]", t, k, "tuple"))
    out.sort(key=lambda x: (x[2], 0 if x[3] != "tuple" else 1, x[0]))
    return out


# ------------------------------------------------------------------------------------------------------------------------------ the table of the stronger side's turns

class Table:
    """The atoms at every turn of the stronger side in a set of records, and the pairs of consecutive turns of one record (the move between them is the stronger side's)."""

    def __init__(self, atoms: list):
        self.atoms = atoms
        self.rows: list = []                  # [n, len(atoms)]
        self.game: list = []                  # the record each row is of
        self.pairs: list = []                 # (row before, row after)
        self.ends: int = 0                    # stronger-side moves after which the game ended (no next turn): not counted

    def add(self, mdl, rec: dict, groups: list) -> None:
        side = MO.side_of(rec["attacker_white"])
        fens = MO.fens_of(rec["text"], rec["moves"])
        turns = []
        for k, f in enumerate(fens):
            if k % 2:
                continue
            if sorted(map(tuple, mdl.measure_groups(f, side))) != groups:
                break                                                              # a class change ends the record's reading there
            turns.append(f)
        vals = np.array(mdl.measure_atoms(turns, side, self.atoms), dtype=np.int64).reshape(len(turns), len(self.atoms)) if turns else np.zeros((0, len(self.atoms)), dtype=np.int64)
        base = len(self.rows)
        self.rows.extend(list(vals))
        self.game.extend([self.ends] * len(turns))
        self.pairs.extend((base + i, base + i + 1) for i in range(len(turns) - 1))
        self.ends += 1

    def ahead(self, k: int) -> tuple:
        """The pairs (row, the row k turns later of the same record): the stronger side's own k-th next turn."""
        g = np.array(self.game, dtype=np.int64)
        i = np.arange(len(g) - k)
        ok = g[i] == g[i + k] if len(g) > k else np.zeros(0, dtype=bool)
        return i[ok], i[ok] + k

    def arrays(self) -> tuple:
        x = np.array(self.rows, dtype=np.int64).reshape(len(self.rows), len(self.atoms))
        p = np.array(self.pairs, dtype=np.int64).reshape(len(self.pairs), 2)
        return x, p


def _scalar(x: np.ndarray, idx: dict, kind: str, names: tuple) -> np.ndarray:
    cols = [x[:, idx[a]] for a in names]
    return cols[0] + cols[1] if kind == "sum" else cols[0] * cols[1]


def test1(tabs: dict, atoms: list, terms: list) -> dict:
    """The share of the stronger side's moves after which the term is lower, by set: {text: {set: share}}. Lower in the order of the tuple."""
    idx = {a: i for i, a in enumerate(atoms)}
    prep = {}
    for name, t in tabs.items():
        x, p = t.arrays()
        if len(p) == 0:
            continue
        lt_eq = {}
        for i, a in enumerate(atoms):
            u, v = x[p[:, 0], i], x[p[:, 1], i]
            lt_eq[a] = (v < u, v == u)
        prep[name] = (x, p, lt_eq)
    out: dict = {}
    for text, names, size, kind in terms:
        row = {}
        for name, (x, p, lt_eq) in prep.items():
            if kind == "atom":
                fall = lt_eq[names[0]][0]
            elif kind in ("sum", "product"):
                s = _scalar(x, idx, kind, names)
                fall = s[p[:, 1]] < s[p[:, 0]]
            else:
                fall = np.zeros(len(p), dtype=bool)
                alive = np.ones(len(p), dtype=bool)
                for a in names:
                    fall |= alive & lt_eq[a][0]
                    alive &= lt_eq[a][1]
            row[name] = float(fall.mean())
        out[text] = row
    return out


def within(tabs: dict, atoms: list, terms: list, k: int = 3) -> dict:
    """TEST 1, the second reading (October's lesson: a measure is not asked to fall on every move, only to ADMIT a drop within a few of our moves): the share of the stronger side's turns at which the term is lower at some one of its next `k` turns of the same game, by set. {text: {set: share}}."""
    idx = {a: i for i, a in enumerate(atoms)}
    prep = {}
    for name, t in tabs.items():
        x, _ = t.arrays()
        ahead = [t.ahead(j) for j in range(1, k + 1)]
        have = np.zeros(len(x), dtype=bool)
        for i, _j in ahead:
            have[i] = True
        if not have.any():
            continue
        lt = {a: [] for a in atoms}
        eq = {a: [] for a in atoms}
        for i, j in ahead:
            full_lt = np.zeros(len(x), dtype=bool)
            full_eq = np.zeros(len(x), dtype=bool)
            for a in atoms:
                lt_a, eq_a = full_lt.copy(), full_eq.copy()
                lt_a[i] = x[j, idx[a]] < x[i, idx[a]]
                eq_a[i] = x[j, idx[a]] == x[i, idx[a]]
                lt[a].append(lt_a)
                eq[a].append(eq_a)
        prep[name] = (x, have, ahead, lt, eq)
    out: dict = {}
    for text, names, size, kind in terms:
        row = {}
        for name, (x, have, ahead, lt, eq) in prep.items():
            any_low = np.zeros(len(x), dtype=bool)
            for n_, (i, j) in enumerate(ahead):
                if kind == "atom":
                    low = lt[names[0]][n_]
                elif kind in ("sum", "product"):
                    s = _scalar(x, idx, kind, names)
                    low = np.zeros(len(x), dtype=bool)
                    low[i] = s[j] < s[i]
                else:
                    low = np.zeros(len(x), dtype=bool)
                    alive = np.ones(len(x), dtype=bool)
                    for a in names:
                        low |= alive & lt[a][n_]
                        alive &= eq[a][n_]
                any_low |= low
            row[name] = float(any_low[have].mean())
        out[text] = row
    return out


# ------------------------------------------------------------------------------------------------------------------------------ test 2 (October's criterion)

def init_job(variant_unused=None) -> None:
    model()


def test2_job(job: dict) -> dict:
    """One term on the sample: for each position the value now and the question `measure_lowers` within d of our moves; -> {"text", "found": n, "failing": [index], "cut": [index], "d": the largest d needed, "seconds"}."""
    mdl = model()
    text, d, secs = job["term"], job["d"], job["seconds"]
    found = 0
    failing, cut = [], []
    need = 0
    t0 = time.perf_counter()
    for i, pos in enumerate(job["positions"]):
        side = MO.side_of(pos["attacker_white"])
        now = mdl.measure_value(pos["text"], side, text)
        first, moves, game_end, was_cut, nodes = mdl.measure_lowers(pos["text"], side, text, now, d, secs)
        if first is not None:
            found += 1
            need = max(need, moves)
        elif was_cut:
            cut.append(i)
        else:
            failing.append(i)
    return {"term": text, "found": found, "failing": failing, "cut": cut, "d": need, "seconds": time.perf_counter() - t0}


# ------------------------------------------------------------------------------------------------------------------------------ the corpus

def corpus(cls: str, roots: list, generated: int, seed: int) -> dict:
    """GOOD, BAD, TABLE as the module doc says: from the games of part A (`roots`: folders of a run, each holding starts_<class>.json and matches/; the attacker is seat 0) and generated games."""
    good, bad = [], []
    for root in roots:
        sf = Path(root) / f"starts_{cls}.json"
        if not sf.exists():
            continue
        starts = json.loads(sf.read_text())
        for f in sorted((Path(root) / "matches").glob(f"*_{cls}_*.jsonl")):
            for line in f.read_text().splitlines():
                if not line.strip():
                    continue
                g = json.loads(line)
                st = starts[int(g["start"][len(cls):])]
                rec = {"text": st["text"], "moves": g["moves"], "attacker_white": st["attacker_white"], "source": f.stem, "dtz": st["dtz"]}
                (good if MO.attacker_won(rec["text"], rec["moves"], rec["attacker_white"]) else bad).append(rec)
    gen_good, gen_bad = MO.generated_games(cls, generated, seed)
    table = MO.table_games(cls, generated, seed + 1)
    return {"good": good + gen_good, "bad": [b for b in bad if MO.kept_the_win(b["text"], b["moves"], b["attacker_white"])] + gen_bad, "table": table}


def cmd_mine(a, say) -> None:
    import multiprocessing as mp
    root = out_root(a.tag)
    cls = a.cls
    mdl = model()
    starts = MO.sample_starts(cls, 12, 3)
    side0 = MO.side_of(starts[0]["attacker_white"])
    groups = sorted(map(tuple, mdl.measure_groups(starts[0]["text"], side0)))
    atoms = mdl.measure_library(groups)
    kinds = mdl.kinds()
    say(f"OBS whole_79_17 miner {cls}: the groups of the class {[(('theirs' if t else 'ours'), kinds[k]) for t, k in groups]}, the library holds {len(atoms)} atoms")
    dirs = [d for d in (root.parent / t for t in a.games_from.split(",")) if d.exists()]
    sets = corpus(cls, dirs, a.games, 5)
    for name, recs in sets.items():
        pos = sum(len(r["moves"]) // 2 + 1 for r in recs)
        say(f"OBS whole_79_17 miner {cls}: set {name}: {len(recs)} games, {pos} turns of the stronger side; coverage of the starts by the table's distance {MO.coverage([{'dtz': r['dtz']} for r in recs])}")
    tabs = {}
    for name, recs in sets.items():
        t = Table(atoms)
        for r in recs:
            t.add(mdl, r, groups)
        tabs[name] = t
    terms = terms_of(atoms)
    say(f"OBS whole_79_17 miner {cls}: {len(terms)} terms in the library (atoms, sums and products of two, tuples of two and three)")
    t0 = time.perf_counter()
    s1 = test1(tabs, atoms, terms)
    w3 = within(tabs, atoms, terms, 3)
    say(f"OBS whole_79_17 miner {cls}: test 1 read in {time.perf_counter() - t0:.1f} s (the share of moves after which a term is lower, and the share of turns at which it is lower at one of the next three turns)")
    more = [t for t in terms if s1[t[0]].get("good", 0) > s1[t[0]].get("bad", 0)]
    say(f"OBS whole_79_17 miner {cls}: {len(more)} of {len(terms)} terms fall on more of the stronger side's moves in the GOOD games than in the BAD ones")
    # the ranking: the share of turns at which the term is lower within three turns in the GOOD games (October's lesson: it must ADMIT a drop within a few moves), then the BAD share (lower is better), then the simplest; equal readings are one term (the simplest)
    order = sorted(more, key=lambda t: (-round(w3[t[0]].get("good", 0), 6), round(w3[t[0]].get("bad", 0), 6), t[2], t[0]))
    seen_sig, carried = set(), []
    for t in order:
        sig = (round(w3[t[0]].get("good", 0), 6), round(w3[t[0]].get("bad", 0), 6), round(s1[t[0]].get("good", 0), 6), round(s1[t[0]].get("bad", 0), 6), round(s1[t[0]].get("table", 0), 6))
        if sig in seen_sig:
            continue
        seen_sig.add(sig)
        carried.append(t)
        if len(carried) >= a.top:
            break
    say(f"OBS whole_79_17 miner {cls}: the first {len(carried)} terms by the share within three turns in the GOOD games go to test 2; the best reads GOOD {w3[carried[0][0]].get('good', 0):.3f} BAD {w3[carried[0][0]].get('bad', 0):.3f}")
    sample = MO.sample_starts(cls, a.sample, 91)
    say(f"OBS whole_79_17 miner {cls}: the sample of test 2: {len(sample)} won positions, the stronger side to move, coverage by the table's distance {MO.coverage(sample)}")
    jobs = [{"term": t[0], "positions": sample, "d": a.d, "seconds": a.seconds} for t in carried]
    with mp.get_context("spawn").Pool(a.workers, initializer=init_job) as pool:
        res = pool.map(test2_job, jobs, chunksize=1)
    rows = []
    for t, r in zip(carried, res):
        rows.append({"term": t[0], "size": t[2], "good": s1[t[0]].get("good"), "bad": s1[t[0]].get("bad"), "table": s1[t[0]].get("table"), "good3": w3[t[0]].get("good"), "bad3": w3[t[0]].get("bad"), "table3": w3[t[0]].get("table"), **r})
    passing = sorted([r for r in rows if not r["failing"] and not r["cut"]], key=lambda r: (r["d"], r["size"], r["term"]))
    passing_size = sorted(passing, key=lambda r: (r["size"], r["d"], r["term"]))
    failing = sorted([r for r in rows if r["failing"] or r["cut"]], key=lambda r: (len(r["failing"]) + len(r["cut"]), r["size"], r["term"]))
    for r in passing_size[:10]:
        say(f"OBS whole_79_17 miner {cls}: PASSES {r['term']}: no failing position of {len(sample)}, the largest d needed {r['d']}; share of moves after which it is lower: GOOD {r['good']:.3f}, BAD {r['bad']:.3f}, TABLE {r['table']:.3f}; lower within three turns: GOOD {r['good3']:.3f}, BAD {r['bad3']:.3f}, TABLE {r['table3']:.3f}")
    for r in failing[:5]:
        say(f"OBS whole_79_17 miner {cls}: FAILS {r['term']}: {len(r['failing'])} failing positions of {len(sample)} (index {r['failing'][:8]}), {len(r['cut'])} cut by the clock; GOOD {r['good']:.3f}, BAD {r['bad']:.3f}, TABLE {r['table']:.3f}; within three turns GOOD {r['good3']:.3f}, BAD {r['bad3']:.3f}, TABLE {r['table3']:.3f}")
    say(f"OBS whole_79_17 miner {cls}: {len(passing)} of the {len(carried)} terms carried to test 2 pass; {len(failing)} fail")
    chosen = passing_size[0] if passing_size else None
    out = {"class": cls, "groups": groups, "kinds": kinds, "sample_positions": sample, "passing": passing_size, "best_failing": failing[:10], "chosen": chosen, "d": a.d, "test1_top": [{"term": t[0], **s1[t[0]], **{k + "3": v for k, v in w3[t[0]].items()}} for t in carried[:30]]}
    (root / f"measure_{cls}.json").write_text(json.dumps(out))
    if chosen:
        mp_file = root / "measures.json"
        ms = json.loads(mp_file.read_text()) if mp_file.exists() else {}
        ms[cls] = {"term": chosen["term"], "d": a.d}
        mp_file.write_text(json.dumps(ms))
        say(f"OBS whole_79_17 miner {cls}: the class's measure (the simplest that passes): {chosen['term']} at d {max(1, chosen['d'])}")
    else:
        say(f"OBS whole_79_17 miner {cls}: NO TERM PASSES: the best failing terms and their failing positions are in measure_{cls}.json; no part D is built on a failed term")


def main(argv: "list | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stage", choices=("mine",))
    ap.add_argument("--class", dest="cls", default="one")
    ap.add_argument("--tag", default="full")
    ap.add_argument("--games-from", default="full")
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--sample", type=int, default=40)
    ap.add_argument("--top", type=int, default=60)
    ap.add_argument("--d", type=int, default=3)
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--workers", type=int, default=9)
    a = ap.parse_args(argv)
    root = out_root(a.tag)

    def say(line: str) -> None:
        TM.emit([line], root / "obs.txt")
    cmd_mine(a, say)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
