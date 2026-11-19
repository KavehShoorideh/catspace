"""PLAN 78.67 B, C, D: THE PLAYERS THAT BREAK A TIE BY THE NUMBER, their matches and what the number did in the games. A sandbox: nothing is wired into any player of the wake, nothing is written to
rules/learned or data/derived; every output goes under artifacts/experiments/tiebreak_78_67/. Game-free: the game is named only through odds_measure (78.61) and the engine.

THE PLAYERS all search with the Worth judge fed from the generation-2 record of 78.61 (its integer worths, as measured; no worth is assumed) and the same search call (`solve_equals_batch`: the
ledger deepens within a share of the clock and the equal reading takes the rest). The call hands back the moves the search holds EQUAL to its best. What differs between the players is only the
choice among those moves:
  control          the engine's own move (the root order as it is);
  random           a random one of the equal moves (seeded);
  trees / linear   the equal move whose position after it has the highest NUMBER (tiebreak_number: the heads of 78.65 for "wins a unit and keeps it" read through the frame read);
  arrival          the equal move with the highest number of the 78.49 fit (the earlier result to beat).
Every player gets the same search, so the value and the set of moves held best are the search's, whatever the choice (AGENTS rule 7: nothing is added to the value; the choice among equals is all
that changes). The mechanism of 78.51 (`root_orders`, strict) would hand the order BEFORE the search, which needs the number of every legal move; one frame read costs about a tenth of a second
(stage A), so the number is read only for the equal moves, after the search.

  .venv/bin/python -m catspace.incubator.analyst.discovery.tiebreak_measure stage-a | match | stage-d | report   [--set A|B] [--a P] [--b P] [--pairs K] [--seconds S] [--workers W] [--tag T]
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")                                     # one core a worker: the trees' prediction must not spread over every core of a machine that runs the other workers

import argparse
import json
import multiprocessing
import random
import time
from pathlib import Path

import numpy as np

import cs_engine as CE

from catspace.incubator.analyst.discovery import odds_measure as OM
from catspace.incubator.analyst.discovery import subgoal_play_measure as PM
from catspace.io import paths

SECONDS = 0.2                       # a move for every player (78.61 used 0.05; the equal reading needs room, as in 78.50 and 78.51)
LEDGER_SHARE = PM.LEDGER_SHARE      # of the clock for the ledger's deepening; the equal reading takes the rest (as 78.50)
PLY_CAP = 120                       # the ply cap of 78.61
WORTH_GEN = 3                       # `OM.players_of(3)` plays the Worth judge fed from gen2/record.json, the generation-2 record
FIRST_SEEDS = {"A": OM.MATCH_FIRST_SEED, "B": 41000, "C": 73000, "D": 105000}      # (plan 78.93: C and D, the rule of B with other seeds)
PLAYERS = ("control", "random", "trees", "linear", "arrival")
KIND = {p: f"tb_{p}" for p in PLAYERS}
ARRIVAL_WEIGHTS = "subgoal_play_78_50/full/weights_horizon8.json"
VC_KEYS = ("vc_asked", "vc_held", "vc_replaced", "vc_failed", "vc_cut", "vc_nodes", "vc_seconds")
REG: list = []                      # the movers made in this process for the game being played
_READER = None
_SHEET = None
_ARRIVAL = None


def out_root(tag: str = "full") -> Path:
    return Path(paths.experiment("tiebreak_78_67", tag))


# ------------------------------------------------------------------------------------------------------------------------------ the two sets of openings

def openings(set_name: str, pairs: int) -> list:
    """[(seed, opening moves)] of one set. Set A is the seeds of the 78.61 matches. Set B starts far from them and skips any seed whose opening is one of set A's (or already in B): the sets share no opening."""
    if set_name in ("C", "D"):
        return later_set(set_name, pairs)
    first = FIRST_SEEDS["A"]
    a = [(first + k, tuple(OM.random_opening(None, first + k))) for k in range(pairs)]
    if set_name == "A":
        return [(s, list(o)) for s, o in a]
    taken = {o for _s, o in a}
    out, seed = [], FIRST_SEEDS["B"]
    while len(out) < pairs:
        o = tuple(OM.random_opening(None, seed))
        if o not in taken:
            taken.add(o)
            out.append((seed, list(o)))
        seed += 1
    return out


SET_PAIRS = 100                     # the size of a full set: the sets C and D are made against the full A and B whatever number of pairs a run plays


def later_set(set_name: str, pairs: int) -> list:
    """PLAN 78.93: the sets C and D, made by the rule of set B with other seeds (the first seed in FIRST_SEEDS, one seed after the other, a seed skipped when its opening is one the full sets before it, or the set itself so far,
    already hold). C skips the openings of the full A and B (100 pairs each); D those of A, B and the full C. The first `pairs` of the set are returned, so a run of 40 pairs plays the first 40 of the 100."""
    taken = {tuple(o) for name in ("A", "B") for _s, o in openings(name, SET_PAIRS)}
    if set_name == "D":
        taken |= {tuple(o) for _s, o in later_set("C", SET_PAIRS)}
    out, seed = [], FIRST_SEEDS[set_name]
    while len(out) < SET_PAIRS:
        o = tuple(OM.random_opening(None, seed))
        if o not in taken:
            taken.add(o)
            out.append((seed, list(o)))
        seed += 1
    return out[:pairs]


def overlap_lines(pairs: int = SET_PAIRS) -> list:
    """The overlap of every two sets, in openings the two share (must be none) and in seeds."""
    sets = {n: openings(n, pairs) for n in ("A", "B", "C", "D")}
    out = []
    names = sorted(sets)
    for i, x in enumerate(names):
        for y in names[i + 1:]:
            ox, oy = {tuple(o) for _s, o in sets[x]}, {tuple(o) for _s, o in sets[y]}
            sx, sy = {s for s, _o in sets[x]}, {s for s, _o in sets[y]}
            out.append(f"OBS opening sets overlap {x} and {y}: openings in both {len(ox & oy)} (of {len(ox)} and {len(oy)} distinct), seeds in both {len(sx & sy)}")
    out += [f"OBS opening set {n}: {len(v)} pairs, {len({tuple(o) for _s, o in v})} distinct openings, first seed {v[0][0]}, last seed {v[-1][0]}" for n, v in sorted(sets.items())]
    return out


# ------------------------------------------------------------------------------------------------------------------------------ the numbers

def reader():
    global _READER
    if _READER is None:
        from catspace.incubator.analyst.discovery import tiebreak_number as TN
        _READER = TN.Reader()
    return _READER


def arrival_numbers(moves: list, legal: list) -> dict:
    """{legal index: the number of the 78.49 fit} (the 78.51 reading), for the position after `moves`."""
    global _SHEET, _ARRIVAL
    if _SHEET is None:
        from catspace.incubator.analyst.discovery import question_sheet as QS
        from catspace.incubator.analyst.discovery import sheet_rust as SR
        _SHEET = SR.model(QS.Env(OM.VARIANT), {})   # an empty worth: no assumed worth
        _ARRIVAL = json.loads(Path(paths.experiment(ARRIVAL_WEIGHTS)).read_text())["weights"]
    rows = _SHEET.subgoal_children([(None, list(moves) + [legal[0]])], 1, [[len(moves)]], _ARRIVAL)[0]
    return dict(rows[0][2]) if rows else {}


def pick_best(engine_move: str, numbers: dict) -> str:
    """The move with the highest number; a tie at the top goes to the search's own move when it is among them, else to the first name; no number at all: the search's own move."""
    if not numbers:
        return engine_move
    top = max(numbers.values())
    best = sorted(m for m, v in numbers.items() if v == top)
    return engine_move if engine_move in best else best[0]


def reading_clock(seconds: float, costs: list, floor: float = 0.5) -> float:
    """The clock the search gets when the number's reading is paid out of the mover's own clock: the clock less the running mean of the readings so far, never under `floor` of it."""
    if not costs:
        return seconds
    return max(floor * seconds, seconds - sum(costs) / len(costs))


class Mover67(OM.Mover):
    """The movers of 78.61 plus the players above (kinds `tb_*`). `charge` in the spec: the reading is taken out of the clock of the search (`reading_clock`)."""

    def __init__(self, spec: dict, seconds: float, seed: str, trace: str):
        super().__init__(spec, seconds, seed, trace)
        self.charge = bool(spec.get("charge", False))
        self.costs: list = []
        self.verify_choice = bool(spec.get("verify_choice", False))                 # plan 78.92 / 78.93: the verification of the choice, off unless the spec asks
        self.choice_mode = str(spec.get("choice_mode", "slot"))
        self.reask = bool(spec.get("reask", False))                                 # plan 79.02: the equal reading asks a held move again without the prunings, off unless the spec asks
        self.stats = {"decisions": 0, "ties": 0, "changed": 0, "wall": 0.0, "read": 0.0, "search": 0.0, "quiet": 0, "children": 0, "log": []}
        REG.append(self)

    def choice_kw(self) -> dict:
        """The keywords of `solve_equals_batch` for the verification of the choice (empty when it is off: the call is exactly what it was)."""
        kw = {"verify_choice": True, "choice_mode": self.choice_mode} if self.verify_choice else {}
        if self.reask:
            kw["reask"] = True
        return kw

    def note_choice(self, raw: list) -> None:
        """Count what the verification of the choice did in one decision (row 12 the state: 1 held, 2 replaced, 3 failed and kept, 4 cut; 13 its nodes; 14 its seconds)."""
        st = self.stats
        st["nodes"] = st.get("nodes", 0) + int(raw[2] or 0)                         # (the nodes the decision spent, for the mean a decision)
        if self.reask and len(raw) > 15 and raw[15]:                                # plan 79.02: the second asking of the equal reading, summed over the decisions (asked, held, reasked, out, unreached, first reading cut)
            rk = st.setdefault("rk", [0, 0, 0, 0, 0, 0, 0])
            for i, x in enumerate(raw[15][:6]):
                rk[i] += int(x)
            rk[6] += 1
        if not self.verify_choice or not raw[12]:
            return
        for k in VC_KEYS:
            st.setdefault(k, 0)
        st["vc_asked"] += 1
        st[("vc_held", "vc_replaced", "vc_failed", "vc_cut")[raw[12] - 1]] += 1
        st["vc_nodes"] += int(raw[13])
        st["vc_seconds"] += float(raw[14])

    def search_clock(self) -> float:
        return reading_clock(self.seconds, self.costs) if self.charge and self.kind in (KIND["trees"], KIND["linear"], KIND["arrival"]) else self.seconds

    def choose(self, text: "str | None", moves: list, legal: list) -> tuple:
        if not self.kind.startswith("tb_"):
            return super().choose(text, moves, legal)
        t0 = time.perf_counter()
        raw = CE.solve_equals_batch([(OM.VARIANT, text, list(moves))], self.search_clock(), worth=dict(self.worth), ours_first=False, valuation="odds record", threads=1,
                                    ledger_share=LEDGER_SHARE, **self.choice_kw())[0]
        self.note_choice(raw)
        row = PM.equal_set(raw)
        t1 = time.perf_counter()
        mv, eq = row["move"], row["equals"]
        pick, st = mv, self.stats
        st["decisions"] += 1
        if mv is not None and len(eq) >= 2:
            st["ties"] += 1
            if self.kind == KIND["random"]:
                pick = random.Random(f"{self.seed}:{len(moves)}").choice(sorted(eq))
            elif self.kind in (KIND["trees"], KIND["linear"]):
                got = reader().numbers(list(moves), list(legal), list(eq), PLY_CAP)
                key = "trees" if self.kind == KIND["trees"] else "linear"
                st["children"] += len(got)
                st["quiet"] += sum(1 for v in got.values() if v["quiet"])
                pick = pick_best(mv, {m: v[key] for m, v in got.items()})
            elif self.kind == KIND["arrival"]:
                nums = arrival_numbers(list(moves), list(legal))
                pick = pick_best(mv, {m: nums[legal.index(m)] for m in eq if legal.index(m) in nums})
            st["changed"] += pick != mv
            if self.kind != KIND["random"]:
                self.costs.append(time.perf_counter() - t1)
        st["log"].append((len(moves), len(eq), int(pick != mv)))
        t2 = time.perf_counter()
        st["search"] += t1 - t0
        st["read"] += t2 - t1
        st["wall"] += t2 - t0
        return pick, row["depth"]


def play67(spec: dict) -> dict:
    """One game as 78.61 plays it, with the movers of this module; the per-seat stats ride with the result."""
    OM.Mover = Mover67
    REG.clear()
    g = OM.play_game(spec)
    g["stats"] = [dict(m.stats) for m in REG]
    return g


def settled(games: list, level: float = 99.0, min_pairs: int = 30, margin: float = 0.03) -> "str | None":
    """Whether a match may stop early, read on whole opening pairs only: "clear" when the band at `level` per cent (bootstrap over opening pairs) lies wholly on one side of a half, "level" when it lies
    wholly within `margin` of a half, else None. The band is wider than the 90 per cent one that is reported, because it is looked at many times while the games come in."""
    pairs = OM.pair_scores(games)
    if len(pairs) < min_pairs:
        return None
    v = np.array(list(pairs.values()), dtype=float)
    rng = np.random.default_rng(OM.BOOT_SEED)
    boots = np.array([v[rng.integers(0, len(v), len(v))].mean() / 2 for _ in range(OM.DRAWS)])
    lo, hi = float(np.percentile(boots, (100 - level) / 2)), float(np.percentile(boots, 100 - (100 - level) / 2))
    if lo > 0.5 or hi < 0.5:
        return "clear"
    if lo > 0.5 - margin and hi < 0.5 + margin:
        return "level"
    return None


def run_games(specs: list, workers: int, out_file: Path, say=print, play=None, stop=None) -> list:
    """Play the specs on a spawn Pool (a rerun skips the games already in `out_file`); -> the games in the order of the specs. `play` is the function that plays one spec (default `play67`; a later plan hands in its own, a module-level
    function so the spawned workers can import it).
    `stop` (default None: every game is played) is asked after every ten further opening pairs that are whole IN THE ORDER OF THE SPECS (a function of that leading run of games -> a word or None, such
    as `settled`); on a word the pool is ended and only that leading run is returned. The leading run is used, not the games that happen to be done, because short games finish first and are not a fair
    sample of the match."""
    play = play or play67
    done = {}
    if out_file.exists():
        for line in out_file.read_text().splitlines():
            if line.strip():
                g = json.loads(line)
                done[g["gid"]] = g
    todo = [s for s in specs if s["gid"] not in done]
    t0 = time.perf_counter()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    n_all, n0 = len(specs), len(done)
    with open(out_file, "a") as f:
        def keep(g: dict) -> None:
            done[g["gid"]] = g
            f.write(json.dumps(g) + "\n")
            f.flush()
            if len(done) % 20 == 0 or len(done) == n_all:
                el = time.perf_counter() - t0
                say(f"OBS progress {out_file.stem}: {len(done)} of {n_all} games, {el:.0f} s, about {el / max(1, len(done) - n0) * (n_all - len(done)):.0f} s left")
        asked = [0]

        def lead() -> list:
            run = []
            for s in specs:
                if s["gid"] not in done:
                    break
                run.append(done[s["gid"]])
            return run

        def stopped() -> "list | None":
            if stop is None:
                return None
            run = lead()
            if len(run) - asked[0] < 20 or len(run) == n_all:
                return None
            asked[0] = len(run) - len(run) % 20
            word = stop(run[:asked[0]])
            if word:
                say(f"OBS early stop {out_file.stem}: {word} after the first {asked[0]} of {n_all} games in the order of the openings; the rest is not played")
                return run[:asked[0]]
            return None

        cut = stopped()
        if cut is not None:
            return cut
        if workers <= 1 or not todo:
            for s in todo:
                keep(play(s))
                cut = stopped()
                if cut is not None:
                    return cut
        else:
            with multiprocessing.get_context("spawn").Pool(workers) as pool:
                for g in pool.imap_unordered(play, todo, chunksize=1):
                    keep(g)
                    cut = stopped()
                    if cut is not None:
                        pool.terminate()
                        return cut
    return [done[s["gid"]] for s in specs]


# ------------------------------------------------------------------------------------------------------------------------------ the matches

def player_spec(name: str, base: dict, charge: bool = False, kinds: "dict | None" = None) -> dict:
    return dict(base, kind=(kinds or KIND)[name], charge=charge)


def match_specs(a: str, b: str, set_name: str, pairs: int, seconds: float, scratch: Path, charge: bool = False, kinds: "dict | None" = None) -> list:
    base = OM.players_of(WORTH_GEN, OM.out_root() / "full")
    pa, pb = player_spec(a, base, charge, kinds), player_spec(b, base, charge, kinds)
    specs = []
    for seed, opening in openings(set_name, pairs):
        for colour in (0, 1):
            specs.append({"gid": f"{a}_v_{b}:{set_name}{seed}:{colour}", "start": f"{set_name}{seed}", "colour": colour, "text": None, "opening": opening, "players": [pa, pb] if colour == 0 else [pb, pa],
                          "seconds": seconds, "max_plies": PLY_CAP, "scratch": str(scratch)})
    return specs


def seat_of(g: dict, who: int) -> int:
    """The seat (0 moves first) of player `who` (0: a, 1: b) in a game."""
    return g["colour"] if who == 0 else 1 - g["colour"]


def match_lines(a: str, b: str, set_name: str, games: list, load: float, busy: float, wall: float, seconds: float, workers: int) -> list:
    sc = np.array([OM.white_score(g) if g["colour"] == 0 else 1 - OM.white_score(g) for g in games])
    w, d, l = int((sc == 1).sum()), int((sc == 0.5).sum()), int((sc == 0).sum())
    pairs = OM.pair_scores(games)
    s, lo, hi = OM.score_band(pairs)
    unf = sum(g["unfinished"] for g in games)
    dep, st = {0: [0, 0], 1: [0, 0]}, {0: {}, 1: {}}
    for g in games:
        for who in (0, 1):
            seat = seat_of(g, who)
            dep[who][0] += g["depth"][seat][0]
            dep[who][1] += g["depth"][seat][1]
            for k, v in g["stats"][seat].items():
                if k != "log":
                    st[who][k] = st[who].get(k, 0) + v
    dd = {k: (v[0] / v[1] if v[1] else float("nan")) for k, v in dep.items()}
    tag = f"{a} v {b} [set {set_name}]"
    lines = [f"OBS tiebreak match {tag}: games {len(games)} ({len(pairs)} opening pairs, colours alternated): {a} W {w} D {d} L {l}; score {s:.4f} [{lo:.4f}, {hi:.4f}] 90 pct band by bootstrap over opening pairs; "
             f"unfinished {unf} ({100 * unf / len(games):.1f} pct, counted as draws); mean ledger depth {a} {dd[0]:.2f}, {b} {dd[1]:.2f}"]
    for who, nm in ((0, a), (1, b)):
        x = st[who]
        dec, ties = max(1, x["decisions"]), x["ties"]
        lines.append(f"OBS tiebreak match {tag}: {nm}: decisions {x['decisions']}, with two or more equal moves {ties} ({100 * ties / dec:.1f} pct), choice differed from the engine's own move in {x['changed']} "
                     f"({100 * x['changed'] / max(1, ties):.1f} pct of the ties, {100 * x['changed'] / dec:.1f} pct of all decisions); time per decision {1000 * x['wall'] / dec:.1f} ms (search {1000 * x['search'] / dec:.1f}, reading {1000 * x['read'] / dec:.1f}); "
                     + (f"children read {x['children']}, quiet {x['quiet']}" if x["children"] else "no frame read"))
    lines.append(f"OBS tiebreak match {tag}: load average (1 min) at start {load:.2f}, cores busy at the instant of start {busy:.2f}; wall {wall:.0f} s; clock {seconds} s a move for both sides, ledger share {LEDGER_SHARE}, "
                 f"ply cap {PLY_CAP}, workers {workers}")
    return lines


def emit(lines: list, out_file: "Path | None") -> None:
    for x in lines:
        print(x, flush=True)
    if out_file is not None:
        with open(out_file, "a") as f:
            f.write("\n".join(lines) + "\n")


def games_file(tag: str, a: str, b: str, set_name: str) -> Path:
    return out_root(tag) / "matches" / f"{a}_v_{b}_{set_name}.jsonl"


def read_games(p: Path) -> list:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def wait_quiet(say, limit: float = 4.0, patience: float = 300.0) -> "tuple[float, float]":
    """Wait up to `patience` seconds for the one-minute load average to fall to `limit`; print what was seen; -> (load, cores busy) at the start."""
    t0 = time.perf_counter()
    load = os.getloadavg()[0]
    first = load
    while load > limit and time.perf_counter() - t0 < patience:
        time.sleep(15)
        load = os.getloadavg()[0]
    busy = OM.cpu_busy()
    say(f"OBS tiebreak quiet wait: load average first {first:.2f}, at the start {load:.2f} after {time.perf_counter() - t0:.0f} s (limit {limit}); cores busy at the instant of start {busy:.2f}")
    return load, busy


# ------------------------------------------------------------------------------------------------------------------------------ stage D: what the number did in the games

def quiet_plies(layer1, goal: str, moves: list) -> list:
    """The plies from ply 10 on at which the position is settled (the rows of the heads' fit)."""
    names, raw = layer1.position_table([(None, moves)], 1)
    w = 2 + 2 * len(names)
    t = np.frombuffer(raw, dtype=np.int16).reshape(len(moves) + 1, w)
    return [p for p in range(10, len(moves) + 1) if t[p, 0] == 1 and t[p, 1] == 1]


def keeps_per_game(games: list) -> list:
    """Per game: {role: sorted steps at which the role won a unit and kept it} (the events of 78.64, read from the moves of the game), and the first mover's role."""
    from catspace.incubator.analyst.discovery import events_rules as ER
    from catspace.incubator.analyst.discovery import graph_layer1 as GL
    from catspace.incubator.analyst.discovery import graph_layer2_measure as G2M
    from catspace.incubator.analyst.discovery import human_heads_data as HD
    from catspace.incubator.analyst.discovery import tactical_tables as TT
    spec, vocab = ER.spec_and_vocab()
    clauses = ER.rule_clauses(spec)
    roles = ER.role_words(clauses)
    meta = json.loads(Path(paths.experiment("tactical_78_62", "state", "state.json")).read_text())
    edge = dict(zip(meta["places"], TT.edge_distances(meta["places"]).tolist()))
    ER._init_worker(spec, vocab, ER.persistent_fluents(clauses), ER.load_worth(), roles, edge)
    layer1 = GL.model(HD.GAME, HD.VARIANT)
    goal = G2M.goal_kind_of(HD.VARIANT)
    out = []
    for g in games:
        r = ER.read_game((0, list(g["moves"]), quiet_plies(layer1, goal, list(g["moves"])), None))
        out.append({"first": r["first"], "keeps": {ro: sorted(r["kept"].get(("keeps", ro), [])) for ro in roles}, "roles": roles})
    return out


def first_after(steps: list, t: int) -> "int | None":
    for s in steps:
        if s >= t:
            return s
    return None


def mover_first_rates(games: list, who: int, evs: list, only_ties: bool) -> "tuple[np.ndarray, np.ndarray]":
    """Per game, for the decisions of player `who`: (decisions after which some side wins and keeps a unit, of them those where the mover is first)."""
    n, k = np.zeros(len(games)), np.zeros(len(games))
    for i, (g, ev) in enumerate(zip(games, evs)):
        seat = seat_of(g, who)
        roles, first = ev["roles"], ev["first"]
        other = [r for r in roles if r != first][0]
        for ply, n_eq, _changed in g["stats"][seat]["log"]:
            if only_ties and n_eq < 2:
                continue
            me = first if ply % 2 == 0 else other
            them = other if me == first else first
            a, b = first_after(ev["keeps"][me], ply), first_after(ev["keeps"][them], ply)
            if a is None and b is None:
                continue
            n[i] += 1
            k[i] += (b is None) or (a is not None and a <= b)
    return n, k


def band_ratio(n: np.ndarray, k: np.ndarray, draws: int = 1000, seed: int = 6767) -> "tuple[float, float, float]":
    rng = np.random.default_rng(seed)
    v = []
    for _ in range(draws):
        ix = rng.integers(0, len(n), len(n))
        v.append(k[ix].sum() / max(1.0, n[ix].sum()))
    return float(k.sum() / max(1.0, n.sum())), float(np.percentile(v, 5)), float(np.percentile(v, 95))


def stage_d_lines(a: str, b: str, set_name: str, games: list) -> list:
    evs = keeps_per_game(games)
    lines = []
    for only_ties in (False, True):
        for who, nm in ((0, a), (1, b)):
            n, k = mover_first_rates(games, who, evs, only_ties)
            r, lo, hi = band_ratio(n, k)
            lines.append(f"OBS tiebreak played {a} v {b} [set {set_name}]: {nm}, {'decisions with two or more equal moves' if only_ties else 'all decisions'}: after {int(n.sum())} decisions some side next won and kept a unit; "
                         f"the mover was first in {int(k.sum())} ({100 * r:.1f} pct [{100 * lo:.1f}, {100 * hi:.1f}] 90 pct band by bootstrap over games)")
    return lines


# ------------------------------------------------------------------------------------------------------------------------------ stage A: the cost

def sample_positions(n: int, seed: int) -> list:
    """`n` positions (move lists, ply 10 on) drawn at random from the human games of the 78.59 file that the heads were fitted on and tested on, seeded: positions of real play, not of this plan's players
    (the match logs of this plan give the cost in their own play)."""
    heads = Path(paths.experiment("estimator_78_59", "full"))
    games = [json.loads(x) for x in (heads / "games.jsonl").read_text().splitlines()[:3000] if x.strip()]
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        g = games[rng.randrange(len(games))]
        if len(g["moves"]) > 12:
            out.append(list(g["moves"][:rng.randrange(10, len(g["moves"]))]))
    return out


def stage_a(n: int, seconds: float, say) -> None:
    base = OM.players_of(WORTH_GEN, OM.out_root() / "full")
    r = reader()
    tot = {"search": 0.0, "frames": 0.0, "statements": 0.0, "heads": 0.0, "kinds": 0.0, "decisions": 0, "ties": 0, "children": 0, "quiet": 0, "agree": 0, "arrival_agree": 0, "rand_same": 0}
    sizes = []
    for moves in sample_positions(n, 6767):
        legal = CE.state(OM.VARIANT, None, moves)[2]
        t0 = time.perf_counter()
        row = PM.equal_set(CE.solve_equals_batch([(OM.VARIANT, None, moves)], seconds, worth=dict(base["worth"]), ours_first=False, valuation="odds record", threads=1, ledger_share=LEDGER_SHARE)[0])
        tot["search"] += time.perf_counter() - t0
        tot["decisions"] += 1
        eq = row["equals"]
        sizes.append(len(eq))
        if len(eq) < 2 or row["move"] is None:
            continue
        tot["ties"] += 1
        c0 = dict(r.cost)
        got = r.numbers(moves, legal, eq, PLY_CAP)
        for k in ("frames", "statements", "heads", "kinds"):
            tot[k] += r.cost[k] - c0[k]
        tot["children"] += len(got)
        tot["quiet"] += sum(v["quiet"] for v in got.values())
        a = pick_best(row["move"], {m: v["trees"] for m, v in got.items()})
        b = pick_best(row["move"], {m: v["linear"] for m, v in got.items()})
        nums = arrival_numbers(moves, legal)
        c = pick_best(row["move"], {m: nums[legal.index(m)] for m in eq if legal.index(m) in nums})
        tot["agree"] += a == b
        tot["arrival_agree"] += a == c
        tot["rand_same"] += a == random.Random(f"{len(moves)}").choice(sorted(eq))
    sz = np.array(sizes)
    ties = max(1, tot["ties"])
    say(f"OBS tiebreak stage A: {tot['decisions']} positions (human games of the 78.59 file, ply 10 on); the search at {seconds} s a move, ledger share {LEDGER_SHARE}, one thread: {1000 * tot['search'] / tot['decisions']:.1f} ms a move; "
        f"moves held equal per position: mean {sz.mean():.2f}, median {np.median(sz):.0f}, with two or more {tot['ties']} ({100 * tot['ties'] / tot['decisions']:.1f} pct)")
    say(f"OBS tiebreak stage A: the number's reading, per position with two or more equal moves ({tot['ties']}): frame read {1000 * tot['frames'] / ties:.0f} ms, statements {1000 * tot['statements'] / ties:.0f} ms, "
        f"kinds {1000 * tot['kinds'] / ties:.0f} ms, heads {1000 * tot['heads'] / ties:.0f} ms; per child {1000 * tot['frames'] / max(1, tot['children']):.0f} ms of frame read; children read {tot['children']} (quiet {tot['quiet']}, "
        f"{100 * tot['quiet'] / max(1, tot['children']):.1f} pct)")
    say(f"OBS tiebreak stage A: the pick of the trees number is the pick of the linear number in {tot['agree']} of {tot['ties']}, of the arrival number in {tot['arrival_agree']}, of a random equal move in {tot['rand_same']}")


# ------------------------------------------------------------------------------------------------------------------------------ the run

def main(argv: "list | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stage", choices=("stage-a", "match", "stage-d", "report"))
    ap.add_argument("--set", default="A", choices=("A", "B"))
    ap.add_argument("--a", default="trees", choices=PLAYERS)
    ap.add_argument("--b", default="control", choices=PLAYERS)
    ap.add_argument("--pairs", type=int, default=100)
    ap.add_argument("--seconds", type=float, default=SECONDS)
    ap.add_argument("--workers", type=int, default=9)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--charge", action="store_true", help="take the number's reading out of the clock of the search")
    ap.add_argument("--tag", default="full")
    a = ap.parse_args(argv)
    root = out_root(a.tag)
    root.mkdir(parents=True, exist_ok=True)
    obs = root / "obs.txt"

    def say(line: str) -> None:
        emit([line], obs)
    if a.stage == "stage-a":
        stage_a(a.n, a.seconds, say)
    elif a.stage == "match":
        load, busy = wait_quiet(say)
        t0 = time.perf_counter()
        games = run_games(match_specs(a.a, a.b, a.set, a.pairs, a.seconds, root / "scratch", a.charge), a.workers, games_file(a.tag, a.a, a.b, a.set), say)
        emit(match_lines(a.a, a.b, a.set, games, load, busy, time.perf_counter() - t0, a.seconds, a.workers), obs)
    elif a.stage == "stage-d":
        emit(stage_d_lines(a.a, a.b, a.set, read_games(games_file(a.tag, a.a, a.b, a.set))), obs)
    else:
        for p in sorted((root / "matches").glob("*.jsonl")):
            x, rest = p.stem.split("_v_")
            y, s = rest.rsplit("_", 1)
            emit(match_lines(x, y, s, read_games(p), float("nan"), float("nan"), float("nan"), a.seconds, a.workers), None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
