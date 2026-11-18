"""PLAN 78.80 (SYSTEM_SPEC 7 "subgoals as symbols that go with winning", 9 the score): THE STAGES that test fact patterns as symbols on human moves. A sandbox: no player is changed; nothing is written to rules/learned or
data/derived; every output goes under artifacts/experiments/symbols_78_80/<tag>/. Game-free: the game is named only through the pieces that exist (ladder_measure, human_heads_data, graph_layer1/2). The ladder's accepted
moves are read in ONE place, `score_ladder`, which hands them to deep_heads.score and hit_rate (the same wall as 78.68: no pick reads them).

INTERFACE (what each stage takes and writes; who calls it). Each stage is a command and prints OBS lines (also to obs.txt) and saves its rows.
  sample     (a)  held-out human games (78.59's split: not fitted, not the arrival sample), less every game of the ladder's 400 positions; up to PER_GAME positions a game from ply MIN_PLY on, one in each of
                  PER_GAME equal stretches of the game; the search at the wake's clock (ladder_measure.Tools.search: solve_equals_batch); a position is kept when it holds two or more moves equal and the move played is
                  among them. Split by a stable hash of the game into FIND and CONFIRM. Writes sample.jsonl, tries.jsonl. Prints the yield, the tie sizes and the endgame coverage.
  deltas     (b)  the head rows (every HEAD_EVERY-th ply from MIN_PLY of every held-out game, the result of the side to move and its inputs), the patterns mined on the FIND part's rows (symbols.mine), the events of every
                  tied move of every position (symbols.deltas), the pattern counts of the head rows. Writes keys.json, events.npz (the event matrix: a row per tied move), head.npz, head_us.npz, head_them.npz. Prints the pattern counts and the cost.
  test1      (c)  symbols.term_table on the FIND head rows against the held head R+K; writes one_find.pkl.
  test2      (d)  symbols.rate_table on the FIND positions; writes two_find.pkl.
  rule       (e)  symbols.find (the rule of symbols.py, stated there) then symbols.confirm, once, on the CONFIRM part for the picked events only; writes picked.json and passing.json.
  report          what the passing events are, and whether they only stand for taking a unit (a confound, measured: the pick on the positions where no tied move takes a unit).
  score      (f)  the pick (symbols.symbol_pick) scored on the CONFIRM part against the move played (against the search's first move and a random pick), by deep_heads.score on the ladder's scored part and by hit_rate.
  all        the stages in order.
Called by: the command line only; ladder_print.py reads passing.json, keys.json and the sample for the cases.

  .venv/bin/python -m catspace.incubator.analyst.discovery.symbols_measure sample|deltas|test1|test2|rule|score|all --tag short --limit 200 | --tag full
"""
from __future__ import annotations

from catspace.incubator.analyst.discovery import tiebreak_measure as TM          # first: it sets one thread a process before numpy is loaded

import argparse
import hashlib
import json
import multiprocessing
import os
import pickle
import random
import time
from collections import Counter as Tally
from pathlib import Path

import numpy as np

from catspace.incubator.analyst.discovery import deep_heads as DH
from catspace.incubator.analyst.discovery import estimator_fit as EF
from catspace.incubator.analyst.discovery import graph_layer1 as GL
from catspace.incubator.analyst.discovery import graph_layer2 as L2
from catspace.incubator.analyst.discovery import graph_layer2_measure as G2M
from catspace.incubator.analyst.discovery import human_heads_data as HD
from catspace.incubator.analyst.discovery import ladder_measure as LM
from catspace.incubator.analyst.discovery import symbols as SY
from catspace.io import paths

OUT = ("symbols_78_80",)
SEED = 8080
PER_GAME = 5                                                      # positions from one game at most: one in each of this many equal stretches of the game
MIN_PLY = 10                                                      # from ply 10 on (as 78.59's rows)
HEAD_EVERY = 4                                                    # the head rows: every this-many-th ply of each game
MINE_FLOOR = 30                                                   # a pattern is mined when it occurs on this many rows of the find part
LEVELS = (0, 1, 2, 3)
SIZES = (1, 2)                                                    # size 3 was costed on 600 rows: 17 s and 665 thousand patterns found; not run (see the plan note)
DRAWS_ONE = 100                                                   # games resampled for test one's band
DRAWS_BOOT = 1000                                                 # games resampled for the paired differences of stage f
GAMES_FILE = ("estimator_78_59", "full", "games.jsonl")
BINS = ((0, 4), (5, 8), (9, 12), (13, 16), (17, 24), (25, 99))     # units left (for the endgame coverage)


def out_root(tag: str) -> Path:
    p = Path(paths.experiment(*OUT, tag))
    p.mkdir(parents=True, exist_ok=True)
    return p


def jsonl(p: Path) -> list:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


class Say:
    def __init__(self, root: Path, stage: str):
        self.root, self.stage = root, stage

    def __call__(self, msg: str) -> None:
        line = f"OBS symbols {self.stage}: {msg}"
        print(line, flush=True)
        with open(self.root / "obs.txt", "a") as f:
            f.write(line + "\n")


# ------------------------------------------------------------------------------------------------------------------------------ the games and the split

def all_games() -> list:
    """Every game of the 78.59 file in file order, the ordinal in the file standing for the game; no id of any player or game number is kept."""
    out = []
    for i, ln in enumerate(Path(paths.experiment(*GAMES_FILE)).read_text().splitlines()):
        if ln.strip():
            g = json.loads(ln)
            out.append({"ord": i, "moves": g["moves"], "y": g["y"], "rm": g["welo"], "ro": g["belo"], "fit": g["fit"], "arrival": g["arrival"]})
    return out


def ladder_ordinals() -> set:
    """The games of the ladder's 400 positions (their ids 'human:h<ordinal>:<ply>'); both parts of its split."""
    out = set()
    for ln in jsonl(LM.out_root("full") / "positions.jsonl"):
        parts = ln["id"].split(":")
        if parts[0] == "human":
            out.add(int(parts[1][1:]))
    return out


def held_out(games: list, excluded: set) -> list:
    """The held-out games: 78.59's test games that the arrival number was not fitted on, less the games of the ladder's positions."""
    return [g for g in games if not g["fit"] and not g["arrival"] and g["ord"] not in excluded]


def part_of(ordinal: int) -> str:
    """THE SPLIT of this plan: by a stable hash of the game's ordinal, FIND or CONFIRM."""
    return "find" if int(hashlib.md5(f"symbols80:{ordinal}".encode()).hexdigest(), 16) % 2 == 0 else "confirm"


def plies_of(ordinal: int, n: int) -> list:
    """The plies of one game the sample tries: the game's stretch from MIN_PLY to its last move (a move is played there) cut in PER_GAME equal parts, one ply drawn in each."""
    lo, hi = MIN_PLY, n - 1
    if hi < lo:
        return []
    k = min(PER_GAME, hi - lo + 1)
    rng = random.Random(SEED * 100003 + ordinal)
    cuts = [lo + (hi - lo + 1) * i // k for i in range(k + 1)]
    return [rng.randrange(cuts[i], cuts[i + 1]) for i in range(k)]


# ------------------------------------------------------------------------------------------------------------------------------ (a) the sample

_TOOLS = None


def _try(job: tuple) -> dict:
    global _TOOLS
    ordinal, k, moves, played = job
    if _TOOLS is None:
        _TOOLS = LM.Tools()
    pid = f"h{ordinal}:{k}"
    row = {"id": pid, "game": ordinal, "ply": k, "played": played, "ok": False, "n_eq": 0, "tie": False}
    if not LM.playable(moves):
        return row
    t0 = time.perf_counter()
    s = _TOOLS.search({"moves": moves}, LM.LIMITS["wake_seconds"])
    eq = list(s["equals"])
    row.update(ok=True, n_eq=len(eq), tie=len(eq) >= 2 and played in eq, search_seconds=time.perf_counter() - t0)
    if row["tie"]:
        row["tied"] = [s["move"]] + sorted(m for m in eq if m != s["move"])
        row["move"] = s["move"]
    return row


def bin_of(units: int) -> str:
    lo, hi = next((a, b) for a, b in BINS if a <= units <= b)
    return f"{lo}-{hi}" if hi < 99 else f"{lo}+"


def stage_sample(root: Path, limit: int, workers: int, say) -> None:
    games = all_games()
    excl = ladder_ordinals()
    held = held_out(games, excl)
    say(f"the 78.59 file holds {len(games)} games; held out (not fitted, not the arrival sample) {sum(1 for g in games if not g['fit'] and not g['arrival'])}; less the games of the ladder's 400 positions "
        f"({len(excl)} games) {len(held)}; load average {os.getloadavg()[0]:.2f}; {workers} workers")
    jobs = []
    for g in held:
        for k in plies_of(g["ord"], len(g["moves"])):
            jobs.append((g["ord"], k, g["moves"][:k], g["moves"][k]))
    if limit:
        jobs = jobs[:limit]
    t0 = time.perf_counter()
    rows = []
    with multiprocessing.get_context("spawn").Pool(workers) as pool:
        for i, r in enumerate(pool.imap(_try, jobs, chunksize=8), 1):
            rows.append(r)
            if i % 2000 == 0:
                say(f"progress: {i} of {len(jobs)} positions tried, {time.perf_counter() - t0:.0f} s, load average {os.getloadavg()[0]:.2f}")
    by_ord = {g["ord"]: g for g in held}
    layer1 = GL.model(HD.GAME, HD.VARIANT)
    units_of: dict = {}
    for r in rows:
        if r["ok"] and r["game"] not in units_of:
            units_of[r["game"]] = LM.unit_counts(layer1, by_ord[r["game"]]["moves"])
    sample = []
    for r in rows:
        r["units"] = units_of[r["game"]][r["ply"]] if r["ok"] else None
    for r in rows:
        if r["tie"]:
            sample.append({"id": r["id"], "game": r["game"], "ply": r["ply"], "moves": by_ord[r["game"]]["moves"][:r["ply"]], "tied": r["tied"], "move": r["move"], "played": r["played"],
                           "units": r["units"], "part": part_of(r["game"])})
    (root / "sample.jsonl").write_text("\n".join(json.dumps(r) for r in sample) + "\n")
    (root / "tries.jsonl").write_text("\n".join(json.dumps({k: v for k, v in r.items() if k not in ("tied",)}) for r in rows) + "\n")
    ok = [r for r in rows if r["ok"]]
    ties_any = [r for r in ok if r["n_eq"] >= 2]
    say(f"YIELD: positions tried {len(rows)} ({len(ok)} playable), two or more moves held equal {len(ties_any)} ({100 * len(ties_any) / max(1, len(ok)):.1f} pct of playable), the move played among them "
        f"{len(sample)} ({100 * len(sample) / max(1, len(ok)):.1f} pct of playable; {100 * len(sample) / max(1, len(ties_any)):.1f} pct of the ties); search {np.mean([r['search_seconds'] for r in ok]) if ok else 0:.2f} s a position; {time.perf_counter() - t0:.0f} s")
    sizes = Tally(len(r["tied"]) for r in sample)
    say("TIE SIZES (moves held equal, positions kept): " + ", ".join(f"{k}: {sizes[k]}" for k in sorted(sizes)) + f"; mean {np.mean([len(r['tied']) for r in sample]) if sample else 0:.2f}")
    parts = Tally(r["part"] for r in sample)
    gparts = {p: len({r["game"] for r in sample if r["part"] == p}) for p in ("find", "confirm")}
    say(f"SPLIT by a stable hash of the game: FIND {parts['find']} positions in {gparts['find']} games, CONFIRM {parts['confirm']} positions in {gparts['confirm']} games; games with a kept position: "
        f"{len({r['game'] for r in sample})}; positions per game at most {max(Tally(r['game'] for r in sample).values()) if sample else 0}")
    bt = Tally(bin_of(r["units"]) for r in ok)
    bs = Tally(bin_of(r["units"]) for r in sample)
    say("ENDGAME COVERAGE by units left on the board (tried / kept as ties): " + ", ".join(f"{b}: {bt[b]} / {bs[b]}" for b in (bin_of(a) for a, _h in BINS)))
    allg = [min(LM.unit_counts(layer1, g["moves"])[MIN_PLY:] or [99]) for g in held[:2000]]
    say(f"ENDGAME COVERAGE of the games: of the first {len(allg)} held-out games, the fewest units left from ply {MIN_PLY} on is at most 8 in {sum(1 for x in allg if x <= 8)} ({100 * sum(1 for x in allg if x <= 8) / max(1, len(allg)):.1f} pct), "
        f"at most 4 in {sum(1 for x in allg if x <= 4)} ({100 * sum(1 for x in allg if x <= 4) / max(1, len(allg)):.1f} pct); the ladder's bands (few <= 8, mid 9-16, many 17+) in the kept sample: "
        + ", ".join(f"{b}: {sum(1 for r in sample if LM.band_of(r['units']) == b)}" for b, _l, _h in LM.BANDS))


# ------------------------------------------------------------------------------------------------------------------------------ (b) the head rows, the patterns, the events

def head_rows(games: list, model) -> dict:
    """The head rows of the held-out games: every HEAD_EVERY-th ply from MIN_PLY on where the position is open: game ordinal, ply, the result of the side to move, the kinds ours less theirs, the two ratings (mover first), the part."""
    names, raw = model.position_table([(None, g["moves"]) for g in games], max(1, (os.cpu_count() or 2) - 1))
    nk = len(names)
    goal = G2M.goal_kind_of(HD.VARIANT)
    use = [k for k, nm in enumerate(names) if nm != goal]
    flat = np.frombuffer(raw, dtype=np.int16)
    w = 2 + 2 * nk
    off = 0
    gi, ply, y, kd, rm, ro = [], [], [], [], [], []
    for g in games:
        n = len(g["moves"]) + 1
        t = flat[off:off + n * w].reshape(n, w).astype(np.int32)
        off += n * w
        for p in range(MIN_PLY, n, HEAD_EVERY):
            if t[p, 0] != 1 or t[p, 1] != 1:
                continue
            first = p % 2 == 0
            gi.append(g["ord"])
            ply.append(p)
            y.append(g["y"] if first else 2 - g["y"])
            kd.append((t[p, 2:2 + nk] - t[p, 2 + nk:2 + 2 * nk])[use])
            rm.append(g["rm"] if first else g["ro"])
            ro.append(g["ro"] if first else g["rm"])
    return {"game": np.array(gi, np.int32), "ply": np.array(ply, np.int16), "y": np.array(y, np.int8), "kd": np.array(kd, np.int16), "rm": np.array(rm, np.int16), "ro": np.array(ro, np.int16),
            "find": np.array([part_of(int(o)) == "find" for o in gi], bool), "kinds": np.array([names[k] for k in use])}


def stage_deltas(root: Path, workers: int, say) -> None:
    sample = jsonl(root / "sample.jsonl")
    games = all_games()
    held = held_out(games, ladder_ordinals())
    by_ord = {g["ord"]: g for g in held}
    model = GL.model(HD.GAME, HD.VARIANT)
    goal = G2M.goal_kind_of(HD.VARIANT)
    t0 = time.perf_counter()
    head = head_rows(held, model)
    say(f"HEAD ROWS: every {HEAD_EVERY}th ply from {MIN_PLY} on of the {len(held)} held-out games: {len(head['game'])} rows, FIND {int(head['find'].sum())}, CONFIRM {int((~head['find']).sum())} ({time.perf_counter() - t0:.1f} s)")
    # the rows the patterns are mined on: the FIND part only (positions, the positions after each tied move, the head rows)
    wires, rows = [], []
    for r in sample:
        if r["part"] != "find":
            continue
        n = len(r["moves"])
        wires.append((None, r["moves"]))
        rows.append([n])
        for m in r["tied"]:
            wires.append((None, r["moves"] + [m]))
            rows.append([n + 1])
    gfind = sorted({int(o) for o, f in zip(head["game"], head["find"]) if f})
    for o in gfind:
        sel = head["ply"][(head["game"] == o)]
        wires.append((None, by_ord[o]["moves"]))
        rows.append([int(p) for p in sel])
    nrows = sum(len(r) for r in rows)
    t1 = time.perf_counter()
    keys = SY.mine(model, wires, rows, goal, MINE_FLOOR, SIZES, LEVELS, workers)
    kinds = [n[len("kind:"):] for ty, n in model.static_part()["nodes"] if ty == "kind"]
    say(f"PATTERNS mined on the FIND part's {nrows} rows (positions, positions after each tied move, head rows), sizes {list(SIZES)}, levels {list(LEVELS)}, seen on at least {MINE_FLOOR} rows, patterns with a shut atom left out: "
        f"{len(keys)} patterns; by size and level " + ", ".join(f"({s},{lv}): {sum(1 for k in keys if k['size'] == s and k['level'] == lv)}" for s in SIZES for lv in LEVELS) + f"; {time.perf_counter() - t1:.1f} s")
    (root / "keys.json").write_text(json.dumps([{"key": str(k["key"]), "id": k["id"], "size": k["size"], "level": k["level"], "support": k["support"], "total": k["total"], "bytes": k["bytes"].hex(),
                                                   "words": L2.plain_words(L2.decode(k["bytes"]), kinds), "text": L2.canonical_text(L2.decode(k["bytes"]), kinds)} for k in keys]))
    counter = SY.Counter(model, [k["key"] for k in keys], goal, max(SIZES), LEVELS, workers)
    t2 = time.perf_counter()
    from scipy import sparse
    parts_m = []
    for i in range(0, len(sample), 300):
        chunk = sample[i:i + 300]
        parts_m.append(SY.deltas_matrix([{"moves": r["moves"]} for r in chunk], [r["tied"] for r in chunk], counter))
    M = sparse.vstack(parts_m, format="csr") if parts_m else sparse.csr_matrix((0, 4 * len(keys)), dtype=np.float32)
    secs = time.perf_counter() - t2
    nmoves = sum(len(r["tied"]) for r in sample)
    per_pat = np.asarray(M.sum(axis=0)).ravel().reshape(4, len(keys)).sum(axis=0)
    seen = int((per_pat > 0).sum())
    seen_ev = int((np.asarray(M.sum(axis=0)).ravel() > 0).sum())
    cpu = secs * workers
    say(f"DELTAS: {len(sample)} positions, {nmoves} tied moves; distinct patterns that change on some tied move {seen} of {len(keys)}; distinct events (pattern, side, direction) {seen_ev}; events per tied move: "
        f"mean {M.nnz / max(1, nmoves):.1f}; cost {secs:.1f} s wall on {workers} threads = {secs / max(1, len(sample)) * 1000:.1f} ms wall and about {cpu / max(1, len(sample)) * 1000:.0f} ms of thread time a position; "
        f"load average {os.getloadavg()[0]:.2f}")
    sparse.save_npz(root / "events.npz", M)
    t3 = time.perf_counter()
    wires = [(None, by_ord[int(o)]["moves"]) for o in sorted({int(x) for x in head["game"]})]
    rows = [[int(p) for p in head["ply"][head["game"] == int(o)]] for o in sorted({int(x) for x in head["game"]})]
    # the head rows in the order of head.npz: games in ordinal order, plies in order (the arrays are built in file order, which is ordinal order)
    us, them = counter.count(wires, rows)
    from scipy import sparse
    sparse.save_npz(root / "head_us.npz", us.tocsr())
    sparse.save_npz(root / "head_them.npz", them.tocsr())
    np.savez_compressed(root / "head.npz", **head)
    say(f"HEAD COUNTS: {len(head['game'])} rows x {len(keys)} patterns counted in {time.perf_counter() - t3:.1f} s; total wall of the stage {time.perf_counter() - t0:.1f} s")


# ------------------------------------------------------------------------------------------------------------------------------ the head (test one's base)

def head_inputs(head: dict, fit: np.ndarray) -> "tuple[np.ndarray, dict]":
    """The head's inputs R (mean rating, rating difference, ply) and K (kinds ours less theirs) as 78.59 builds them: R centred and scaled on the fit rows, kinds raw."""
    rm, ro = head["rm"].astype(float), head["ro"].astype(float)
    r = np.column_stack([(rm + ro) / 2, rm - ro, head["ply"].astype(float)])
    c, s = r[fit].mean(axis=0), r[fit].std(axis=0)
    s = np.where(s > 0, s, 1.0)
    x = np.hstack([(r - c) / s, head["kd"].astype(float)])
    return x, {"centre": c, "scale": s}


def head_fit(head: dict, find: np.ndarray, say) -> "tuple[object, dict]":
    """The held head R+K fitted on the FIND head rows (the penalty by 5-fold cross-validation grouped by game, 78.59's grid): (the model, the inputs' centre and scale). PLAN 78.81 reads it for rows the head was not built on."""
    x, tf = head_inputs(head, find)
    c, sc = EF.choose_c(x[find], head["y"][find].astype(int), head["game"][find])
    m = EF.fit_lr(x[find], head["y"][find].astype(int), c)
    say(f"HEAD R+K fitted on {int(find.sum())} FIND head rows: penalty C {c}")
    return m, tf


def head_scores(m, x: np.ndarray) -> np.ndarray:
    """The head's scores (win, draw, loss) for the input rows `x` (a class the fit never saw gets -30)."""
    z = np.asarray(m.intercept_)[None, :] + x @ np.asarray(m.coef_).T
    full = np.full((len(x), 3), -30.0)
    for j, k in enumerate(list(m.classes_)):
        full[:, k] = z[:, j]
    return full


def head_offsets(head: dict, find: np.ndarray, say) -> np.ndarray:
    """Fit the held head R+K on the FIND head rows (the penalty by 5-fold cross-validation grouped by game, 78.59's grid) and return its scores (win, draw, loss) for every head row."""
    x, _tf = head_inputs(head, find)
    c, sc = EF.choose_c(x[find], head["y"][find].astype(int), head["game"][find])
    m = EF.fit_lr(x[find], head["y"][find].astype(int), c)
    full = head_scores(m, x)
    pf = np.exp(full - full.max(axis=1, keepdims=True))
    pf /= pf.sum(axis=1, keepdims=True)
    y = head["y"].astype(int)
    ll = lambda sel: float(-np.mean(np.log(np.clip(pf[sel][np.arange(int(sel.sum())), y[sel]], 1e-9, 1))))      # noqa: E731
    base = np.bincount(y[find], minlength=3) / find.sum()
    llb = float(-np.mean(np.log(base[y[~find]])))
    say(f"HEAD R+K fitted on {int(find.sum())} FIND head rows: penalty C {c}; log loss on the FIND rows {ll(find):.4f}, on the CONFIRM rows {ll(~find):.4f} (the base rate of the FIND rows on the CONFIRM rows: {llb:.4f})")
    return full


# ------------------------------------------------------------------------------------------------------------------------------ (c) test one

def _one_block(args: tuple) -> dict:
    off, y, game, us, them, cols, draws, seed = args
    return SY.term_table(off, y, game, us, them, list(range(len(cols))), draws, seed)


def run_term_table(off, y, game, us, them, columns, draws, seed, workers) -> dict:
    """symbols.term_table for the pattern columns `columns`, the columns cut in blocks across a spawned pool (a forked pool hung after the head's fit had used the maths library's threads); each block carries its own
    columns of the two count matrices."""
    us, them = us.tocsc(), them.tocsc()
    blocks = [columns[i::workers * 4] for i in range(workers * 4)]
    blocks = [b for b in blocks if b]
    tasks = [(off, y, game, us[:, b], them[:, b], b, draws, seed) for b in blocks]
    with multiprocessing.get_context("spawn").Pool(workers) as pool:
        parts = pool.map(_one_block, tasks, chunksize=1)
    out: dict = {}
    for b, part in zip(blocks, parts):
        for j, c in enumerate(b):
            out[c] = part[j]
    return out


def load_head(root: Path) -> "tuple[dict, object, object]":
    from scipy import sparse
    z = np.load(root / "head.npz")
    head = {k: z[k] for k in z.files}
    return head, sparse.load_npz(root / "head_us.npz").tocsr(), sparse.load_npz(root / "head_them.npz").tocsr()


def stage_test1(root: Path, workers: int, say) -> None:
    head, us, them = load_head(root)
    keys = json.loads((root / "keys.json").read_text())
    find = head["find"]
    off = head_offsets(head, find, say)
    np.save(root / "head_offsets.npy", off)
    fidx = np.flatnonzero(find)
    usf, themf = us[fidx], them[fidx]
    present = np.asarray(((usf > 0) + (themf > 0)).sum(axis=0)).ravel()
    cols = [c for c in range(len(keys)) if present[c] >= SY.FLOOR_ROWS]
    say(f"TEST ONE on the FIND head rows ({len(fidx)}): the pattern's count for each side added to the held head, {len(cols)} of {len(keys)} patterns occur on at least {SY.FLOOR_ROWS} rows (the floor); {DRAWS_ONE} resamples of games; {workers} workers")
    t0 = time.perf_counter()
    one = run_term_table(off[fidx], head["y"][fidx].astype(int), head["game"][fidx], usf, themf, cols, DRAWS_ONE, SEED, workers)
    pickle.dump(one, open(root / "one_find.pkl", "wb"))
    ex = [sum(1 for c in cols if one[c]["lo"][s] > 0 or one[c]["hi"][s] < 0) for s in (0, 1)]
    say(f"TEST ONE done in {time.perf_counter() - t0:.0f} s: terms whose 90 pct band over games excludes zero: for the mover's count {ex[0]} of {len(cols)} ({100 * ex[0] / max(1, len(cols)):.1f} pct), for the other side's count {ex[1]} ({100 * ex[1] / max(1, len(cols)):.1f} pct) "
        f"(about 10 pct would be seen if no pattern went with the result)")


# ------------------------------------------------------------------------------------------------------------------------------ (d) test two

def event_rows(sample: list, M, part: str) -> "tuple[object, np.ndarray, np.ndarray, list]":
    """For the positions of a part: the rows of the event matrix (one per tied move, in the order of the sample), the game of each row, whether it is the played move, and the positions."""
    sizes = np.array([len(r["tied"]) for r in sample])
    start = np.concatenate([[0], np.cumsum(sizes)])
    keep = [i for i, r in enumerate(sample) if r["part"] == part]
    rows = np.concatenate([np.arange(start[i], start[i + 1]) for i in keep]) if keep else np.zeros(0, int)
    recs = [sample[i] for i in keep]
    game = np.array([r["game"] for r in recs for _m in r["tied"]])
    played = np.array([m == r["played"] for r in recs for m in r["tied"]])
    return M[rows], game, played, recs


def stage_test2(root: Path, say) -> None:
    sample = jsonl(root / "sample.jsonl")
    from scipy import sparse
    M = sparse.load_npz(root / "events.npz").tocsr()
    keys = json.loads((root / "keys.json").read_text())
    n = len(keys)
    E, game, played, rows = event_rows(sample, M, "find")
    two = SY.rate_table(E, game, played)
    pickle.dump(two, open(root / "two_find.pkl", "wb"))
    ev = np.flatnonzero((two["kp"] + two["ko"]) >= SY.FLOOR_MOVES)
    ev = [i for i in ev if two["kp"][i] >= SY.FLOOR_EACH and two["ko"][i] >= SY.FLOOR_EACH]
    ex = sum(1 for i in ev if two["lo"][i] > 0 or two["hi"][i] < 0)
    say(f"TEST TWO on the FIND part: {len(rows)} positions, {len(game)} tied moves, {int(played.sum())} played; events (pattern, side, direction) {4 * n}; with at least {SY.FLOOR_MOVES} tied moves, {SY.FLOOR_EACH} played and {SY.FLOOR_EACH} other "
        f"(the floor on moves) {len(ev)}; of those the 90 pct interval of the log ratio excludes equality for {ex} ({100 * ex / max(1, len(ev)):.1f} pct; about 10 pct if the played move were no different from the others)")
    top = sorted(ev, key=lambda i: -abs(two["lr"][i] / two["se"][i]))[:5]
    for i in top:
        col, side, d = SY.event_of(i, n)
        say(f"TEST TWO strongest by |z| (FIND): pattern {keys[col]['id']} ({keys[col]['size']},{keys[col]['level']}) {side} {d}: played {two['kp'][i]:.0f} of {two['np']:.0f} (share {two['sp'][i]:.3f}), other {two['ko'][i]:.0f} of {two['no']:.0f} (share {two['so'][i]:.3f}), log ratio {two['lr'][i]:+.2f} [{two['lo'][i]:+.2f}, {two['hi'][i]:+.2f}]")


# ------------------------------------------------------------------------------------------------------------------------------ (e) the rule and the confirm

def stage_rule(root: Path, workers: int, say) -> None:
    from scipy import sparse
    sample = jsonl(root / "sample.jsonl")
    M = sparse.load_npz(root / "events.npz").tocsr()
    keys = json.loads((root / "keys.json").read_text())
    n = len(keys)
    two_f = pickle.load(open(root / "two_find.pkl", "rb"))
    one_f = pickle.load(open(root / "one_find.pkl", "rb"))
    say(f"THE RULE (fixed before the run; symbols.py): an event is picked on FIND when it occurs on at least {SY.FLOOR_MOVES} tied moves ({SY.FLOOR_EACH} played, {SY.FLOOR_EACH} other) and its pattern on at least {SY.FLOOR_ROWS} head rows, "
        f"the 90 pct interval of test two excludes equality, test one's 90 pct band for the side excludes zero, and sign(term) x direction = sign(log ratio); it passes when both intervals hold again on CONFIRM with those signs")
    picked, info = SY.find(two_f, one_f, n)
    say(f"PICKED on FIND: {len(picked)} events of {info['tried']} tried (events over the floors; {4 * n} possible); by chance about {info['tried'] * 0.10 * 0.10 * 0.5:.1f} would be picked if the two tests were unrelated to the pattern "
        f"(0.10 x 0.10 x 0.5 an event, the events of one pattern not independent); distinct patterns picked {len({p['column'] for p in picked})}")
    pickle.dump(picked, open(root / "picked_find.pkl", "wb"))
    # CONFIRM, once, for the picked events only
    head, us, them = load_head(root)
    off = np.load(root / "head_offsets.npy")
    cidx = np.flatnonzero(~head["find"])
    E, game, played, rows = event_rows(sample, M, "confirm")
    E = E.tocsc()
    idx = [SY.event_index(p["column"], p["side"], p["dir"], n) for p in picked]
    two_c = SY.rate_table(E[:, idx], game, played) if picked else None
    cols = sorted({p["column"] for p in picked})
    one_c = run_term_table(off[cidx], head["y"][cidx].astype(int), head["game"][cidx], us[cidx], them[cidx], cols, DRAWS_ONE, SEED + 1, workers) if picked else {}
    done = SY.confirm(picked, two_c, one_c) if picked else []
    passing = [d for d in done if d["passes"]]
    say(f"CONFIRM (read once, picked events only; {len(rows)} positions, {len(game)} tied moves): {len(picked)} picked, {len(passing)} pass (test two's interval excludes equality in the find part's direction and test one's band excludes zero in its sign); "
        f"test two alone holds for {sum(1 for d in done if (d['c_lo'] > 0 if d['sign_two'] > 0 else d['c_hi'] < 0))}, test one alone for {sum(1 for d in done if (d['c_tlo'] > 0 if d['sign_one'] > 0 else d['c_thi'] < 0))}; "
        f"by chance about {len(picked) * 0.05 * 0.05:.2f} of the picked would pass (0.05 x 0.05 an event, one direction fixed by the find part), distinct patterns passing {len({d['column'] for d in passing})}")
    out = []
    for d in done:
        k = keys[d["column"]]
        out.append(dict(d, id=k["id"], size=k["size"], level=k["level"], words=k["words"], text=k["text"]))
    (root / "picked.json").write_text(json.dumps(out))
    (root / "passing.json").write_text(json.dumps([d for d in out if d["passes"]]))
    for d in sorted([d for d in out if d["passes"]], key=lambda d: -abs(d["weight"]))[:10]:
        say(f"PASSING {d['id']} ({d['size']},{d['level']}) {d['side']} {d['dir']}: FIND weight {d['weight']:+.2f}; CONFIRM test two: played {d['c_kp']:.0f} (share {d['c_sp']:.3f}) other {d['c_ko']:.0f} (share {d['c_so']:.3f}) log ratio {d['c_lr']:+.2f} [{d['c_lo']:+.2f}, {d['c_hi']:+.2f}]; "
            f"test one term {d['c_term']:+.3f} [{d['c_tlo']:+.3f}, {d['c_thi']:+.3f}] on {d['c_rows']} head rows")


# ------------------------------------------------------------------------------------------------------------------------------ (f) the pick, scored three ways

def wilson_line(k: float, n: int) -> str:
    lo, hi = DH.wilson(k, n)
    return f"{100 * k / max(1, n):.1f} pct [{100 * lo:.1f}, {100 * hi:.1f}] ({k:.0f} of {n})"


def paired(a: np.ndarray, b: np.ndarray, game: np.ndarray, seed: int) -> "tuple[float, float, float]":
    """The mean of a less b (per position) with a 90 pct bootstrap band over games."""
    d = np.asarray(a, float) - np.asarray(b, float)
    _u, g = np.unique(game, return_inverse=True)
    G = len(_u)
    sums = np.bincount(g, weights=d, minlength=G)
    cnt = np.bincount(g, minlength=G).astype(float)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, G, (DRAWS_BOOT, G))
    est = sums[draws].sum(axis=1) / cnt[draws].sum(axis=1)
    lo, hi = np.percentile(est, [5, 95])
    return float(d.mean()), float(lo), float(hi)


def patterns_of(passing: list) -> list:
    return [{"column": d["column"], "side": d["side"], "dir": d["dir"], "weight": d["weight"]} for d in passing]


def confirm_picks(root: Path) -> dict:
    """The pick on every position of the CONFIRM part, from the events of the passing patterns only: {"rows", "pick" (1 where the pick is the move played), "first" (1 where the search's first move is), "rnd" (the chance of a
    random pick), "silent", "E" (the confirm rows' events), "pidx" (the passing events' columns of E), "pats"}. Reads no answer."""
    from scipy import sparse
    sample = jsonl(root / "sample.jsonl")
    M = sparse.load_npz(root / "events.npz").tocsr()
    n_cols = len(json.loads((root / "keys.json").read_text()))
    pats = patterns_of(jsonl_or_list(root / "passing.json"))
    E, _game, _played, rows = event_rows(sample, M, "confirm")
    pidx = [SY.event_index(p["column"], p["side"], p["dir"], n_cols) for p in pats]
    Ep = E.tocsc()[:, pidx].tocsr() if pats else E[:, :0]
    pick, first, rnd, silent, r0 = [], [], [], [], 0
    w = {(q["column"], q["side"], q["dir"]): q["weight"] for q in pats}
    for r in rows:
        tab = {}
        for m in r["tied"]:
            tab[m] = frozenset(SY.event_of(pidx[int(c)], n_cols) for c in Ep.indices[Ep.indptr[r0]:Ep.indptr[r0 + 1]]) if pats else frozenset()
            r0 += 1
        p = SY.symbol_pick({"moves": r["moves"]}, r["tied"], pats, None, tab)
        assert p in r["tied"]
        pick.append(p == r["played"])
        first.append(r["tied"][0] == r["played"])
        rnd.append(1.0 / len(r["tied"]))
        silent.append(len({sum(w.get(e, 0.0) for e in tab[m]) for m in r["tied"]}) == 1)
    return {"rows": rows, "pick": np.array(pick, float), "first": np.array(first, float), "rnd": np.array(rnd), "silent": np.array(silent), "E": E, "pidx": pidx, "pats": pats}


def score_confirm(root: Path, say) -> None:
    c = confirm_picks(root)
    rows, pick, first, rnd, pats = c["rows"], c["pick"], c["first"], c["rnd"], c["pats"]
    game = np.array([r["game"] for r in rows])
    n = len(rows)
    say(f"SCORE 1, agreement with the move played on the CONFIRM part ({n} positions, {len(pats)} passing events): the symbol pick {wilson_line(pick.sum(), n)}; the search's first move {wilson_line(first.sum(), n)}; a random pick (expected) "
        f"{100 * rnd.mean():.1f} pct (90 pct Wilson for the first two)")
    for nm, other, sd in (("the search's first move", first, 1), ("a random pick", rnd, 2)):
        m, lo, hi = paired(pick, other, game, SEED + sd)
        say(f"SCORE 1 paired difference, symbol pick less {nm}: {100 * m:+.2f} points [{100 * lo:+.2f}, {100 * hi:+.2f}] (90 pct bootstrap over games)")
    say(f"SILENCE on the CONFIRM part: no passing event separates the tied moves (every tied move carries the same sum) in {int(c['silent'].sum())} of {n} positions ({100 * c['silent'].sum() / max(1, n):.1f} pct)")


def strongest(passing: list, n: int = 10) -> "tuple[list, dict]":
    """THE TEN STRONGEST PASSING PATTERNS, by a rule fixed before the cases were read: the largest |z| of the confirm log ratio (its distance from zero over the interval's standard error); an event whose confirm counts equal those of
    a stronger one (same side, direction and counts of played and other moves carrying it) is its twin and is skipped, the twins counted. -> (the events, {index in the list: twins skipped})."""
    se = lambda d: max((d["c_hi"] - d["c_lo"]) / (2 * SY.Z90), 1e-9)                      # noqa: E731
    chosen, seen, twins = [], {}, {}
    for d in sorted(passing, key=lambda d: -abs(d["c_lr"]) / se(d)):
        sig = (d["side"], d["dir"], d["c_kp"], d["c_ko"])
        if sig in seen:
            twins[seen[sig]] = twins.get(seen[sig], 0) + 1
        elif len(chosen) < n:
            seen[sig] = len(chosen)
            chosen.append(d)
    return chosen, twins


def removes_a_unit(rows: list, workers: int) -> list:
    """For each position of `rows` and each tied move, whether the count of units on the board falls across it (a unit taken): [[bool]] in the order of the tied moves. The layer-1 table's counts; no names."""
    model = GL.model(HD.GAME, HD.VARIANT)
    out = []
    for i in range(0, len(rows), 400):
        chunk = rows[i:i + 400]
        wires = [(None, r["moves"] + [m]) for r in chunk for m in r["tied"]]
        names, raw = model.position_table(wires, workers)
        w = 2 + 2 * len(names)
        flat = np.frombuffer(raw, dtype=np.int16)
        off, k = 0, 0
        for r in chunk:
            row = []
            for _m in r["tied"]:
                n = len(wires[k][1]) + 1
                last = flat[off + (n - 1) * w: off + n * w].astype(np.int32)
                off += n * w
                row.append(int(last[2:].sum()) < r["units"])
                k += 1
            out.append(row)
    return out


def stage_report(root: Path, workers: int, say) -> None:
    """What the passing events are (counts by side, direction, size, level) and the confound that a symbol may only stand for taking a unit: the share of played and of other tied moves that take a unit, the share of each passing
    event's carriers that take one, and the pick's agreement on the positions where no tied move takes a unit (where the hint cannot be the taking). Reads no answer."""
    passing = jsonl_or_list(root / "passing.json")
    picked = jsonl_or_list(root / "picked.json")
    say("PASSING events by side and direction (picked on FIND / pass on CONFIRM): " + ", ".join(f"{s} {d}: {sum(1 for p in picked if p['side'] == s and p['dir'] == d)} / {sum(1 for p in passing if p['side'] == s and p['dir'] == d)}" for s in SY.SIDES for d in SY.DIRS))
    say("PASSING events by size and level (picked / pass): " + ", ".join(f"({a},{b}): {sum(1 for p in picked if p['size'] == a and p['level'] == b)} / {sum(1 for p in passing if p['size'] == a and p['level'] == b)}" for a in SIZES for b in LEVELS))
    say(f"PASSING events: {len(passing)} of {len({p['column'] for p in passing})} distinct patterns; the weights (log ratios of the find part) from {min(p['weight'] for p in passing):+.2f} to {max(p['weight'] for p in passing):+.2f}; "
        f"positive {sum(1 for p in passing if p['weight'] > 0)}, negative {sum(1 for p in passing if p['weight'] < 0)}")
    c = confirm_picks(root)
    rows = c["rows"]
    takes = removes_a_unit(rows, workers)
    pl = [t[r["tied"].index(r["played"])] for r, t in zip(rows, takes)]
    oth = [x for r, t in zip(rows, takes) for m, x in zip(r["tied"], t) if m != r["played"]]
    say(f"TAKING on the CONFIRM part ({len(rows)} positions): the move played takes a unit in {wilson_line(sum(pl), len(pl))}; of the other tied moves {wilson_line(sum(oth), len(oth))}; positions where some tied move takes a unit "
        f"{sum(1 for t in takes if any(t))}, where the move played is the only one that does {sum(1 for r, t in zip(rows, takes) if t[r['tied'].index(r['played'])] and sum(t) == 1)}")
    flat = np.array([x for t in takes for x in t], bool)
    E = c["E"].tocsc()
    share = []
    for i in c["pidx"]:
        carriers = E[:, i].indices
        share.append(float(flat[carriers].mean()) if len(carriers) else float("nan"))
    sh = np.array(share)
    ok = ~np.isnan(sh)
    say(f"TAKING and the passing events: of {int(ok.sum())} passing events with carriers on CONFIRM, {int((sh[ok] >= 0.9).sum())} have at least 90 pct of their carrying tied moves taking a unit, {int((sh[ok] >= 0.5).sum())} at least 50 pct, "
        f"{int((sh[ok] < 0.1).sum())} under 10 pct; the share of all tied moves that take a unit is {100 * flat.mean():.1f} pct")
    top, twins = strongest(passing)
    col_of = {(p["column"], p["side"], p["dir"]): j for j, p in enumerate(c["pats"])}
    for k, d in enumerate(top):
        j = col_of[(d["column"], d["side"], d["dir"])]
        say(f"TAKING and the ten strongest, number {k + 1} (pattern {d['id'][:8]}, {d['side']} {d['dir']}): {100 * sh[j]:.1f} pct of its carrying tied moves on CONFIRM take a unit (the share over all tied moves is {100 * flat.mean():.1f} pct); twins {twins.get(k, 0)}")
    quiet = np.array([not any(t) for t in takes])
    game = np.array([r["game"] for r in rows])
    for label, sel in (("no tied move takes a unit", quiet), ("some tied move takes a unit", ~quiet)):
        n = int(sel.sum())
        if n == 0:
            continue
        pk, fi, rn = c["pick"][sel], c["first"][sel], c["rnd"][sel]
        m1, lo1, hi1 = paired(pk, fi, game[sel], SEED + 3)
        m2, lo2, hi2 = paired(pk, rn, game[sel], SEED + 4)
        say(f"SCORE 1 by taking, {label}: {n} positions; symbol pick {wilson_line(pk.sum(), n)}; the search's first move {wilson_line(fi.sum(), n)}; random (expected) {100 * rn.mean():.1f} pct; paired pick less first "
            f"{100 * m1:+.2f} points [{100 * lo1:+.2f}, {100 * hi1:+.2f}], pick less random {100 * m2:+.2f} [{100 * lo2:+.2f}, {100 * hi2:+.2f}]; silent in {int(c['silent'][sel].sum())}")


def jsonl_or_list(p: Path) -> list:
    return json.loads(p.read_text())


def ladder_tables(counter, positions: dict, preps: dict) -> dict:
    """The events of every tied move of the ladder's positions that have a tie (the preps of 78.79 hold the tied moves): {id: {move: events}}. Reads no answer."""
    ids = [i for i in preps if len(preps[i]["eq"]) >= 2 and preps[i].get("tied") and i in positions]
    out = {}
    for i in range(0, len(ids), 100):
        chunk = ids[i:i + 100]
        for pid, d in zip(chunk, SY.deltas([{"moves": positions[p]["moves"]} for p in chunk], [preps[p]["tied"] for p in chunk], counter)):
            out[pid] = d
    return out


def score_ladder(root: Path, workers: int, say, results_root: "Path | None" = None) -> None:
    """SCORES 2 AND 3: deep_heads.score on the ladder's scored part (the control picks beside it) and hit_rate against the accepted moves there. This is the one function that reads the ladder's answers; the pick has not."""
    from catspace.incubator.analyst.discovery import subgoal_pick_measure as SM
    lr = LM.out_root("full")
    positions = {p["id"]: p for p in jsonl(lr / "positions.jsonl")}
    rr = results_root or lr                                                    # PLAN 78.86: the results and the list of open surprises of another run of the same 400 positions (default: the 78.68 run)
    results = {d["id"]: d for d in jsonl(rr / "results.jsonl")}
    open_ids = [r["id"] for r in jsonl(rr / "surprise_list.jsonl")]
    resolved_ids = [i for i, d in results.items() if d["result"] and d["result"]["rung"] in (1, 2)]
    preps = pickle.loads((Path(paths.experiment("subgoal_pick_78_79", "full")) / "preps.pkl").read_bytes())
    keys = json.loads((root / "keys.json").read_text())
    model = GL.model(HD.GAME, HD.VARIANT)
    counter = SY.Counter(model, [int(k["key"]) for k in keys], G2M.goal_kind_of(HD.VARIANT), max(SIZES), LEVELS, workers)
    pats = patterns_of(jsonl_or_list(root / "passing.json"))
    tables = ladder_tables(counter, positions, preps)

    def pick(pid: str, prep: dict) -> str:
        if len(prep["eq"]) < 2 or prep.get("tied") is None or pid not in tables:
            return prep["move"]
        return SY.symbol_pick({"moves": positions[pid]["moves"]}, prep["tied"], pats, None, tables[pid])
    cand = {"name": "symbols: the sum of the weights of the passing events a tied move carries", "pick": pick, "cost": lambda p: 0.0, "random": False}
    cands = SM.candidates()[:2] + [cand]
    ans = {d["id"]: d["accepted"] for d in jsonl(lr / "answers.jsonl")}          # the answers file: read here, handed to score() and hit_rate() only
    scored = [i for i in positions if DH.split_part(i) == "scored" and i in preps]
    table = [DH.score(c, preps, open_ids, resolved_ids, ans, "scored") for c in cands]
    hits = [DH.hit_rate(c, preps, scored, ans, "scored") for c in cands]
    for s, h in zip(table, hits):
        lo_r, hi_r = DH.wilson(s["resolved"], s["n_open"])
        lo_c, hi_c = DH.wilson(s["created"], s["n_resolved_sample"])
        say(f"SCORE 2 (STAGE H) {s['name']}: resolved {s['resolved']:.1f} of {s['n_open']} open surprises (share {s['resolved'] / max(1, s['n_open']):.3f} [{lo_r:.3f}, {hi_r:.3f}]); created {s['created']:.1f} of {s['n_resolved_sample']} resolved positions "
            f"(share {s['created'] / max(1, s['n_resolved_sample']):.3f} [{lo_c:.3f}, {hi_c:.3f}]); net {s['resolved'] - s['created']:+.1f}")
        say(f"SCORE 3 (hit against the accepted moves) {s['name']}: {100 * h['rate']:.1f} pct [{100 * h['lo']:.1f}, {100 * h['hi']:.1f}] ({h['hits']:.1f} of {h['n']} ties, 90 pct Wilson)")
    ctrl = table[0]
    rng = np.random.default_rng(SEED)
    ia = rng.integers(0, len(ctrl["res_vec"]), (DRAWS_BOOT, len(ctrl["res_vec"])))
    ib = rng.integers(0, max(1, len(ctrl["keep_vec"])), (DRAWS_BOOT, len(ctrl["keep_vec"])))
    for s in table[1:]:
        d_open = np.array(s["res_vec"]) - np.array(ctrl["res_vec"])
        d_keep = np.array(s["keep_vec"]) - np.array(ctrl["keep_vec"])
        diff = d_open[ia].sum(axis=1) + d_keep[ib].sum(axis=1)
        lo, hi = np.percentile(diff, [5, 95])
        say(f"SCORE 2 net against the search's own first move, {s['name']}: {d_open.sum() + d_keep.sum():+.1f} positions [{lo:+.1f}, {hi:+.1f}] (90 pct bootstrap over positions, paired)")
    tied_ids = [i for i in scored if len(preps[i]["eq"]) >= 2 and preps[i].get("tied") and i in tables]
    sil = 0
    w = {(q["column"], q["side"], q["dir"]): q["weight"] for q in pats}
    for i in tied_ids:
        sc = [sum(w.get(e, 0.0) for e in tables[i][m]) for m in preps[i]["tied"]]
        sil += len(set(sc)) == 1
    say(f"SILENCE on the ladder's scored part: of {len(tied_ids)} tie positions no passing event separates the tied moves in {sil} ({100 * sil / max(1, len(tied_ids)):.1f} pct)")


def stage_score(root: Path, workers: int, say) -> None:
    score_confirm(root, say)
    score_ladder(root, workers, say)


# ------------------------------------------------------------------------------------------------------------------------------ main

def main(argv: "list | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stage", choices=("sample", "deltas", "test1", "test2", "rule", "score", "report", "all"))
    ap.add_argument("--tag", default="short")
    ap.add_argument("--limit", type=int, default=0, help="a short run: only this many positions are tried")
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args(argv)
    root = out_root(a.tag)
    stages = ["sample", "deltas", "test1", "test2", "rule", "score", "report"] if a.stage == "all" else [a.stage]
    for st in stages:
        say = Say(root, st)
        t0 = time.perf_counter()
        say(f"stage start; tag {a.tag}; limit {a.limit}; load average {os.getloadavg()[0]:.2f}")
        if st == "sample":
            stage_sample(root, a.limit, a.workers, say)
        elif st == "deltas":
            stage_deltas(root, a.workers, say)
        elif st == "test1":
            stage_test1(root, a.workers, say)
        elif st == "test2":
            stage_test2(root, say)
        elif st == "rule":
            stage_rule(root, a.workers, say)
        elif st == "report":
            stage_report(root, a.workers, say)
        else:
            stage_score(root, a.workers, say)
        say(f"stage done in {time.perf_counter() - t0:.0f} s; load average {os.getloadavg()[0]:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
