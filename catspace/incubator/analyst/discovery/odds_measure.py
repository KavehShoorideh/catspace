"""PLAN 78.61: FROM A CLEAN STORE WITH NO ASSUMED WORTH -- generations and odds games. A sandbox: nothing is read from the real store (no valuation, no learned rules of a store, no data/derived),
no oracle, no human game, no puzzle, no fit of an earlier plan; nothing is wired into any player of the wake; every output goes under artifacts/experiments/odds_78_61/.
What is read: the rules' own description through the layer-1 graph of the ordinary start (the kinds, the sides and their places) and the game interface (a position text given to the game).
Game-free: no unit, place or kind of any game is named here; one function (`board_without`) knows the grid of the position text and says so.

GENERATION ZERO knows the rules and the end of the game and nothing else: the ledger search with the judge that counts nothing (`judge="nothing"` of `solve_batch`) looks for a forced end within
its clock; the root moves are handed to it in a RANDOM ORDER (strict mode of 78.51), so the move it keeps is a uniformly random one among those whose value at the finished depth is the best:
when no forced end is in sight every move that does not lose by force ties, and one of them is played at random; when a forced end is in sight, a move of it is played.
UNFINISHED GAMES (the ply cap, or no legal move left to the mover without an outcome) count as DRAWS in the table and are counted and printed apart.
ODDS: starts in which one side lacks one unit of a kind (or two), each side in turn, and the even start as the control. The effect of a unit of a kind = the log odds (win against loss, the draw is
the reference, a half added to each count) of the control less those of the short side, from the short side's view, per unit lacking, averaged over the two sides. A band by bootstrap over games,
resampled within each start (the control within its cell). Two kinds are ORDERED when one is above the other in 95 pct of the draws.
GENERATION g+1 plays with the Worth judge fed from the record of generation g (integer worths = round(1000 x effect of one unit short), as measured, no clamp).

  .venv/bin/python -m catspace.incubator.analyst.discovery.odds_measure starts | gen N | matches | compare   [--games-per-start G] [--seconds S] [--max-plies P] [--workers W] [--pairs K] [--tag T]
"""
from __future__ import annotations

import argparse
import itertools
import json
import multiprocessing
import os
import random
import time
from pathlib import Path

import numpy as np

import cs_engine as CE

from catspace.io import paths

VARIANT = "chess"
SEED = 6100
OPENING_PLIES = 4
DRAW = 50
CONTROL_FACTOR = 3                  # the even start gets this many times the games of one odds start
UNITS_ONE = 2                       # at most this many distinct units of a (side, kind) sampled for the one-short starts
PAIRS_TWO = 2                       # at most this many pairs sampled for the two-short starts
DRAWS = 1000
BOOT_SEED = 6161
SCALE = 1000
LEVEL = 0.95                        # two kinds are ordered when one is above the other in this share of the draws
MATCH_FIRST_SEED = 9100


def out_root() -> Path:
    return Path(paths.experiment("odds_78_61"))


# ------------------------------------------------------------------------------------------------------------------ starts

def ordinary_text() -> str:
    return CE.state(VARIANT, None, [])[0]


def units_of(text: "str | None") -> list:
    """The units of a position as the layer-1 graph gives them: [(side name, kind, place)] sorted. A position text is given to the game by the same text."""
    from catspace.incubator.analyst.discovery import graph_layer1 as GL
    model = GL.model(VARIANT)
    pos = model.position(text if text is not None else ordinary_text())
    out = []
    for typ, nid in pos["nodes"]:
        if typ == "unit":
            _tag, side, kind, place = nid.split(":")
            out.append((side, kind, place))
    return sorted(out)


def board_without(text: str, places: list) -> str:
    """The text of the position without the units on `places`, and with the two rights fields neutral (the same in every start, the even control included, so a rook taken away leaves no
    right to a unit that is gone). The ONE place that knows the grid of the position text: the board field is rows from the far side, a digit counts empty cells, a place name is its column letter
    then its row number."""
    fields = text.split(" ")
    rows = [[c for ch in row for c in (["."] * int(ch) if ch.isdigit() else [ch])] for row in fields[0].split("/")]
    for p in places:
        col, row = ord(p[0]) - ord("a"), int(p[1:])
        rows[len(rows) - row][col] = "."
    packed = []
    for r in rows:
        s, run = "", 0
        for c in r:
            if c == ".":
                run += 1
            else:
                s += (str(run) if run else "") + c
                run = 0
        packed.append(s + (str(run) if run else ""))
    fields[0] = "/".join(packed)
    if len(fields) >= 4:
        fields[2], fields[3] = "-", "-"
    return " ".join(fields)


def playable(text: str) -> bool:
    """The game accepts the text and it is not over."""
    try:
        st = CE.state(VARIANT, text, [])
    except Exception:
        return False
    return st[3] is None and len(st[2]) > 0


def make_starts(games: int, seed: int = SEED) -> dict:
    """{start id: {id, side (index of the short side or None), kind, n, places, text, games}} read from the graph of the ordinary start. Kinds the rules do not let be removed (the game refuses the text or is
    over) are listed apart."""
    base = ordinary_text()
    units = units_of(None)
    mover0 = CE.state(VARIANT, None, [])[1]
    names = sorted({u[0] for u in units}, key=lambda s: (s != mover0, s))        # role 0 first
    side_ix = {n: i for i, n in enumerate(names)}
    starts = {"even": {"id": "even", "side": None, "kind": None, "n": 0, "places": [], "text": board_without(base, []), "games": games * CONTROL_FACTOR}}
    refused = {}
    for kind in sorted({u[1] for u in units}):
        probe = [u for u in units if u[1] == kind][0]
        if not playable(board_without(base, [probe[2]])):
            refused[kind] = "the game refuses the start without one, or it is over"
            continue
        for side in names:
            ps = sorted(u[2] for u in units if u[1] == kind and u[0] == side)
            rng = random.Random(f"{seed}:{side}:{kind}")
            ones = rng.sample(ps, min(UNITS_ONE, len(ps)))
            for p in sorted(ones):
                sid = f"{kind}:{side_ix[side]}:1:{p}"
                starts[sid] = {"id": sid, "side": side_ix[side], "kind": kind, "n": 1, "places": [p], "text": board_without(base, [p]), "games": games}
            if len(ps) >= 2:
                combos = list(itertools.combinations(ps, 2))
                twos = rng.sample(combos, min(PAIRS_TWO, len(combos)))
                for pair in sorted(twos):
                    ok = playable(board_without(base, list(pair)))
                    if ok:
                        sid = f"{kind}:{side_ix[side]}:2:{pair[0]}+{pair[1]}"
                        starts[sid] = {"id": sid, "side": side_ix[side], "kind": kind, "n": 2, "places": list(pair), "text": board_without(base, list(pair)), "games": games}
    return {"starts": starts, "refused": refused, "sides": names}


# ------------------------------------------------------------------------------------------------------------------ one game

def random_opening(text: "str | None", seed: int, plies: int = OPENING_PLIES) -> list:
    rng = random.Random(seed)
    while True:
        moves = []
        for _ in range(plies):
            legal = CE.state(VARIANT, text, moves)[2]
            if not legal:
                break
            moves.append(rng.choice(sorted(legal)))
        if CE.state(VARIANT, text, moves)[3] is None:
            return moves


class Mover:
    """One side's way of choosing: kind 'nothing' (generation zero: the search with the judge that counts nothing, root moves in a random order), 'worth' (the search with a Worth judge fed from a record),
    'count' (the search with every unit counted as one: the player of today without a store) or 'random' (a uniformly random legal move)."""

    def __init__(self, spec: dict, seconds: float, seed: str, trace: str):
        self.kind, self.worth, self.seconds, self.seed, self.trace = spec["kind"], spec.get("worth") or {}, seconds, seed, trace

    def choose(self, text: "str | None", moves: list, legal: list) -> tuple:
        if self.kind == "random":
            return random.Random(f"{self.seed}:{len(moves)}").choice(sorted(legal)), None
        if os.path.exists(self.trace):
            os.remove(self.trace)
        kw: dict = {}
        if self.kind == "nothing":
            order = list(range(len(legal)))
            random.Random(f"{self.seed}:{len(moves)}").shuffle(order)
            kw = {"judge": "nothing", "root_orders": [order], "root_order_mode": "strict"}
        elif self.kind == "worth":
            kw = {"worth": dict(self.worth), "ours_first": False, "valuation": "odds record"}
        out = CE.solve_batch([(VARIANT, text, list(moves))], self.seconds, trace_path=self.trace, threads=1, prior=False, labels=["s:0"], **kw)
        depth = None
        if os.path.exists(self.trace):
            lines = [json.loads(x) for x in open(self.trace).read().splitlines() if x.strip()]
            if lines:
                depth = next((s.get("depth") for s in lines[-1].get("steps", []) if s.get("operator") == "ledger"), None)
        return out[0][0], depth


def play_game(spec: dict) -> dict:
    """One game from `spec['text']` (None: the ordinary start) after `spec['opening']`; seat 0 moves first. The end as the earlier matches have it: the game's own outcome, a position met three times
    is a draw, the ply cap is UNFINISHED (a draw in the table, counted apart). -> {gid, outcome (white's score 100, 50 or 0, None when unfinished), plies, depth sums per seat}."""
    text = spec["text"]
    trace = str(Path(spec["scratch"]) / f"trace_{os.getpid()}.jsonl")
    os.makedirs(spec["scratch"], exist_ok=True)
    movers = [Mover(spec["players"][s], spec["seconds"], f"{spec['gid']}:{s}", trace) for s in (0, 1)]
    moves, seen, outcome, unfinished = list(spec["opening"]), {}, None, False
    depth = [[0, 0], [0, 0]]
    for k in range(len(moves) + 1):
        key = CE.state(VARIANT, text, moves[:k])[4]
        seen[key] = seen.get(key, 0) + 1
    while True:
        _pos, _mover, legal, out_, key = CE.state(VARIANT, text, moves)[:5]
        if out_ is not None:
            outcome = out_[0]
            break
        if not legal:
            unfinished = True
            break
        if len(moves) >= spec["max_plies"]:
            unfinished = True
            break
        seat = len(moves) % 2
        mv, d = movers[seat].choose(text, moves, legal)
        if mv is None:
            unfinished = True
            break
        if d is not None:
            depth[seat][0] += d
            depth[seat][1] += 1
        moves.append(mv)
        key = CE.state(VARIANT, text, moves)[4]
        seen[key] = seen.get(key, 0) + 1
        if seen[key] >= 3:
            outcome = DRAW
            break
    return {"gid": spec["gid"], "start": spec.get("start"), "colour": spec.get("colour"), "outcome": outcome, "plies": len(moves), "unfinished": unfinished, "depth": depth}


def run_games(specs: list, workers: int, out_file: Path, say=print) -> list:
    """Play the specs on a Pool (a rerun skips the games already in `out_file`); -> the games in the order of the specs."""
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
            if len(done) % 50 == 0 or len(done) == n_all:
                el = time.perf_counter() - t0
                say(f"OBS progress {out_file.parent.name}: {len(done)} of {n_all} games, {el:.0f} s, about {el / max(1, len(done) - n0) * (n_all - len(done)):.0f} s left")
        if workers <= 1 or not todo:
            for s in todo:
                keep(play_game(s))
        else:
            with multiprocessing.get_context("spawn").Pool(workers) as pool:
                for g in pool.imap_unordered(play_game, todo, chunksize=1):
                    keep(g)
    return [done[s["gid"]] for s in specs]


# ------------------------------------------------------------------------------------------------------------------ the effect of a unit

def white_score(g: dict) -> float:
    """White's score in a game: 1, a half or 0; unfinished is a draw."""
    return DRAW / 100.0 if g["outcome"] is None else g["outcome"] / 100.0


def counts_of(scores: np.ndarray) -> np.ndarray:
    """(white wins, draws, white losses) of an array of scores."""
    return np.array([(scores == 1.0).sum(), (scores == 0.5).sum(), (scores == 0.0).sum()], dtype=float)


def log_odds(c: np.ndarray) -> float:
    """log((wins + 1/2) / (losses + 1/2)): the draw is the reference."""
    return float(np.log((c[0] + 0.5) / (c[2] + 0.5)))


def effects(cells: dict, starts: dict) -> dict:
    """cells: {start id: scores array (white's score per game)}; -> {(kind, n): per-unit effect}, the log odds the control has and the short side has not, from the short side's view, per unit lacking, the
    mean over the two sides."""
    ctrl = counts_of(cells["even"])
    out = {}
    groups: dict = {}
    for sid, s in starts.items():
        if s["kind"] is None:
            continue
        groups.setdefault((s["kind"], s["n"], s["side"]), []).append(sid)
    kinds = sorted({(k, n) for k, n, _s in groups})
    for k, n in kinds:
        per_side = []
        for side in (0, 1):
            sids = groups.get((k, n, side), [])
            if not sids:
                continue
            c = sum(counts_of(cells[sid]) for sid in sids)
            if side == 1:
                c, cc = c[::-1], ctrl[::-1]
            else:
                cc = ctrl
            per_side.append((log_odds(cc) - log_odds(c)) / n)
        if per_side:
            out[(k, n)] = float(np.mean(per_side))
    return out


def bootstrap(cells: dict, starts: dict, draws: int = DRAWS, seed: int = BOOT_SEED) -> "tuple[dict, list]":
    """(effects on the games as played, the effects of `draws` resamples within each start)."""
    base = effects(cells, starts)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(draws):
        res = {sid: v[rng.integers(0, len(v), len(v))] for sid, v in cells.items()}
        boots.append(effects(res, starts))
    return base, boots


def band(boots: list, key) -> "tuple[float, float]":
    v = [b[key] for b in boots if key in b]
    return float(np.percentile(v, 5)), float(np.percentile(v, 95))


def summarise(cells: dict, starts: dict, draws: int = DRAWS) -> dict:
    """The generation's result: per kind the effect of one and of two units short (per unit) with bands, the order by the one-short effect, the pairs not ordered, and the two-against-one difference."""
    base, boots = bootstrap(cells, starts, draws)
    kinds = sorted({k for k, n in base if n == 1})
    per = {}
    for k in kinds:
        row = {"one": {"term": base[(k, 1)], "lo": band(boots, (k, 1))[0], "hi": band(boots, (k, 1))[1]}}
        if (k, 2) in base:
            row["two"] = {"term": base[(k, 2)], "lo": band(boots, (k, 2))[0], "hi": band(boots, (k, 2))[1]}
            diffs = [b[(k, 2)] - b[(k, 1)] for b in boots if (k, 2) in b and (k, 1) in b]
            row["two_minus_one"] = {"term": base[(k, 2)] - base[(k, 1)], "lo": float(np.percentile(diffs, 5)), "hi": float(np.percentile(diffs, 95))}
        per[k] = row
    order = sorted(kinds, key=lambda k: -per[k]["one"]["term"])
    pairs, unordered = [], []
    for i, a in enumerate(order):
        for b in order[i + 1:]:
            f = float(np.mean([bt[(a, 1)] > bt[(b, 1)] for bt in boots if (a, 1) in bt and (b, 1) in bt]))
            tag = "ordered" if f >= LEVEL or f <= 1 - LEVEL else "NOT ordered"
            pairs.append({"above": a, "below": b, "share": f, "tag": tag})
            if tag != "ordered":
                unordered.append([a, b])
    return {"per_kind": per, "order": order, "pairs": pairs, "unordered_pairs": unordered}


def table_of(starts: dict, games: list) -> list:
    """The table of starts and results (the memory the head was distilled from): per start, white's wins, draws (unfinished included), losses, and the unfinished games apart."""
    rows = []
    for sid, s in starts.items():
        gs = [g for g in games if g["start"] == sid]
        sc = np.array([white_score(g) for g in gs])
        c = counts_of(sc)
        rows.append({"start": sid, "side": s["side"], "kind": s["kind"], "n": s["n"], "places": s["places"], "games": len(gs), "white_wins": int(c[0]), "draws": int(c[1]), "white_losses": int(c[2]),
                     "unfinished": int(sum(g["unfinished"] for g in gs)), "plies": int(sum(g["plies"] for g in gs))})
    return rows


def record_of_generation(gen: int, summary: dict, cfg: dict, table: list, judge_from: "int | None") -> dict:
    """The head of the estimator record: the effect per kind and the integer worths a next generation's judge is fed from."""
    values = {k: int(round(SCALE * summary["per_kind"][k]["one"]["term"])) for k in summary["order"]}
    return {"estimator": "odds games: the effect of a unit of a kind on the result, by taking it away",
            "heads": {"odds_effect": {"type": "terms", "source": f"odds games, generation {gen}", "generation": gen, "played_with": ("nothing (counts nothing)" if judge_from is None else f"worth of the record of generation {judge_from}"),
                                       "inputs": "none (a term per kind: log odds of a win against a loss, the draw the reference, per unit lacking)", "kinds": summary["order"],
                                       "per_kind": summary["per_kind"], "order_of_kinds": summary["order"], "unordered_pairs": summary["unordered_pairs"], "pairs": summary["pairs"],
                                       "kind_values": values, "kind_values_note": f"integer = round({SCALE} x the one-short effect), as measured (no clamp, no goal rule)",
                                       "data": {"config": cfg, "table": table}}}}


def players_of(gen: int, root: Path) -> dict:
    """The player of generation `gen`: zero counts nothing; g plays the Worth judge fed from the record of generation g-1 (the file is the only source of its numbers)."""
    if gen == 0:
        return {"kind": "nothing"}
    rec = json.loads((root / f"gen{gen - 1}" / "record.json").read_text())
    return {"kind": "worth", "worth": rec["heads"]["odds_effect"]["kind_values"]}


def gen_specs(gen: int, starts: dict, cfg: dict, root: Path) -> list:
    p = players_of(gen, root)
    specs = []
    for sid, s in starts.items():
        for k in range(s["games"]):
            specs.append({"gid": f"g{gen}:{sid}:{k}", "start": sid, "text": s["text"], "opening": random_opening(s["text"], SEED + 1000 * (hash_of(sid) % 997) + k), "players": [p, p], "seconds": cfg["seconds"],
                          "max_plies": cfg["max_plies"], "scratch": str(root / f"gen{gen}" / "scratch")})
    return specs


def hash_of(text: str) -> int:
    return sum((i + 1) * ord(c) for i, c in enumerate(text))


def fmt(x: dict) -> str:
    return f"{x['term']:+.3f} [{x['lo']:+.3f}, {x['hi']:+.3f}]"


def stage_gen(gen: int, cfg: dict, say=print) -> dict:
    root = out_root() / cfg["tag"]
    made = make_starts(cfg["games_per_start"])
    starts = made["starts"]
    (root / f"gen{gen}").mkdir(parents=True, exist_ok=True)
    say(f"OBS odds gen {gen}: load {os.getloadavg()[0]:.2f} at start; starts {len(starts)} (even control {starts['even']['games']} games, others {cfg['games_per_start']} each; kinds not removable: {made['refused'] or 'none'}); seconds per move {cfg['seconds']}, ply cap {cfg['max_plies']}, workers {cfg['workers']}")
    specs = gen_specs(gen, starts, cfg, root)
    say(f"OBS odds gen {gen}: players: {json.dumps(players_of(gen, root))}; {len(specs)} games")
    t0 = time.perf_counter()
    games = run_games(specs, cfg["workers"], root / f"gen{gen}" / "games.jsonl", say)
    wall = time.perf_counter() - t0
    cells = {sid: np.array([white_score(g) for g in games if g["start"] == sid]) for sid in starts}
    table = table_of(starts, games)
    unf = sum(g["unfinished"] for g in games)
    say(f"OBS odds gen {gen}: {len(games)} games in {wall:.0f} s wall; unfinished {unf} ({100 * unf / len(games):.1f} pct, counted as draws); mean plies {np.mean([g['plies'] for g in games]):.1f}")
    ev = next(r for r in table if r["start"] == "even")
    say(f"OBS odds gen {gen}: even start (white's view): W {ev['white_wins']} D {ev['draws']} L {ev['white_losses']} of {ev['games']}, unfinished {ev['unfinished']}")
    for r in table:
        if r["kind"] is not None:
            say(f"OBS odds gen {gen}: start {r['start']}: white W {r['white_wins']} D {r['draws']} L {r['white_losses']} of {r['games']}, unfinished {r['unfinished']}")
    summary = summarise(cells, starts)
    for k in summary["order"]:
        row = summary["per_kind"][k]
        t = f"OBS odds gen {gen}: kind {k}: one short {fmt(row['one'])}"
        if "two" in row:
            t += f"; two short (per unit) {fmt(row['two'])}; two less one {fmt(row['two_minus_one'])}"
        say(t)
    say(f"OBS odds gen {gen}: order (highest effect first) {summary['order']}; pairs not ordered at {int(100 * LEVEL)} pct: {summary['unordered_pairs'] if summary['unordered_pairs'] else 'none'}")
    rec = record_of_generation(gen, summary, cfg, table, None if gen == 0 else gen - 1)
    (root / f"gen{gen}" / "record.json").write_text(json.dumps(rec, indent=1))
    say(f"OBS odds gen {gen}: record written {root / f'gen{gen}' / 'record.json'}; kind values {rec['heads']['odds_effect']['kind_values']}")
    return rec


def stage_movement(cfg: dict, say=print) -> None:
    root = out_root() / cfg["tag"]
    recs = {}
    for g in range(0, 9):
        p = root / f"gen{g}" / "record.json"
        if p.exists():
            recs[g] = json.loads(p.read_text())["heads"]["odds_effect"]
    kinds = sorted({k for r in recs.values() for k in r["per_kind"]})
    for k in kinds:
        say(f"OBS odds movement: kind {k}: " + "; ".join(f"gen {g} {recs[g]['per_kind'][k]['one']['term']:+.3f}" for g in recs if k in recs[g]["per_kind"]))
    for g, r in recs.items():
        say(f"OBS odds movement: gen {g} order {r['order_of_kinds']}")
    gens = sorted(recs)
    for a, b in zip(gens, gens[1:]):
        say(f"OBS odds movement: order of gen {a} the same as gen {b}: {recs[a]['order_of_kinds'] == recs[b]['order_of_kinds']}")


# ------------------------------------------------------------------------------------------------------------------ matches

def pair_scores(games: list) -> dict:
    by: dict = {}
    for g in games:
        sc = white_score(g)
        by.setdefault(g["start"], {})[g["colour"]] = sc if g["colour"] == 0 else 1 - sc
    return {s: v[0] + v[1] for s, v in sorted(by.items()) if len(v) == 2}


def score_band(pairs: dict, draws: int = DRAWS, seed: int = BOOT_SEED) -> tuple:
    v = np.array(list(pairs.values()), dtype=float)
    rng = np.random.default_rng(seed)
    boots = np.array([v[rng.integers(0, len(v), len(v))].mean() / 2 for _ in range(draws)])
    return float(v.mean() / 2), float(np.percentile(boots, 5)), float(np.percentile(boots, 95))


def cpu_busy() -> float:
    """Cores busy at this instant: the sum of the processes' cpu shares from `ps`, over 100 (the instant, unlike the 1-minute load average)."""
    import subprocess
    out = subprocess.run(["ps", "-A", "-o", "%cpu="], capture_output=True, text=True).stdout.split()
    return sum(float(x) for x in out) / 100.0


def mover_of(name: str, root: Path) -> dict:
    if name == "random":
        return {"kind": "random"}
    if name == "count":
        return {"kind": "count"}
    return players_of(int(name[1:]), root)


MATCHES = [("g1", "g0"), ("g2", "g1"), ("g2", "g0"), ("g0", "random"), ("g1", "random"), ("g2", "random"), ("g0", "count"), ("g1", "count"), ("g2", "count")]


def match_specs(a: str, b: str, pairs: int, cfg: dict, root: Path) -> list:
    pa, pb = mover_of(a, root), mover_of(b, root)
    specs = []
    for k in range(pairs):
        seed = MATCH_FIRST_SEED + k
        opening = random_opening(None, seed)
        for colour in (0, 1):
            specs.append({"gid": f"{a}_v_{b}:{seed}:{colour}", "start": f"{seed}", "colour": colour, "text": None, "opening": opening, "players": [pa, pb] if colour == 0 else [pb, pa],
                          "seconds": cfg["seconds"], "max_plies": cfg["max_plies"], "scratch": str(root / "matches" / "scratch")})
    return specs


def stage_matches(cfg: dict, say=print) -> None:
    root = out_root() / cfg["tag"]
    (root / "matches").mkdir(parents=True, exist_ok=True)
    for a, b in MATCHES:
        if not all(x in ("random", "count") or (root / f"gen{x[1:]}" / "record.json").exists() or x == "g0" for x in (a, b)):
            say(f"OBS odds match {a} v {b}: SKIPPED, the record the player needs is absent")
            continue
        load = os.getloadavg()[0]
        busy = cpu_busy()
        t0 = time.perf_counter()
        games = run_games(match_specs(a, b, cfg["pairs"], cfg, root), cfg["workers"], root / "matches" / f"{a}_v_{b}.jsonl", say)
        sc = np.array([white_score(g) if g["colour"] == 0 else 1 - white_score(g) for g in games])
        w, d, l = int((sc == 1).sum()), int((sc == 0.5).sum()), int((sc == 0).sum())
        pairs = pair_scores(games)
        s, lo, hi = score_band(pairs)
        unf = sum(g["unfinished"] for g in games)
        dep = {0: [0, 0], 1: [0, 0]}                  # player a, player b: depth sum and count
        for g in games:
            for seat in (0, 1):
                who = 0 if seat == g["colour"] else 1
                dep[who][0] += g["depth"][seat][0]
                dep[who][1] += g["depth"][seat][1]
        dd = {k: (v[0] / v[1] if v[1] else float("nan")) for k, v in dep.items()}
        say(f"OBS odds match {a} v {b}: games {len(games)} ({len(pairs)} opening pairs, colours alternated): {a} W {w} D {d} L {l}; score {s:.4f} [{lo:.4f}, {hi:.4f}] 90 pct band by bootstrap over opening pairs; "
            f"unfinished {unf} ({100 * unf / len(games):.1f} pct, counted as draws); mean ledger depth {a} {dd[0]:.2f}, {b} {dd[1]:.2f}; load average (1 min, includes this script's own earlier work) at start {load:.2f}, cores busy at the instant of start {busy:.2f}; wall {time.perf_counter() - t0:.0f} s; "
            f"seconds per move {cfg['seconds']}, ply cap {cfg['max_plies']}, workers {cfg['workers']}")


# ------------------------------------------------------------------------------------------------------------------ for reading only

def stage_compare(cfg: dict, say=print) -> None:
    """PRINTS the orders of kinds of the earlier plans beside this plan's; nothing read here enters a player or a fit."""
    root = out_root() / cfg["tag"]
    base = Path(paths.experiment("odds_78_61")).parent
    mine = {}
    for g in range(0, 9):
        p = root / f"gen{g}" / "record.json"
        if p.exists():
            r = json.loads(p.read_text())["heads"]["odds_effect"]
            mine[g] = [(k, r["per_kind"][k]["one"]["term"]) for k in r["order_of_kinds"]]
    for g, o in mine.items():
        say("OBS odds compare: this plan, generation %d: %s" % (g, "; ".join(f"{k} {t:+.3f}" for k, t in o)))
    for label, rel in (("78.58 self-play fit (players counting one a unit)", "estimator_78_58/full/estimator.json"), ("78.59 human games", "estimator_78_59/full/estimator_heads_human.json")):
        p = base / rel
        if not p.exists():
            say(f"OBS odds compare: {label}: the file {rel} is absent, nothing printed")
            continue
        r = json.loads(p.read_text())
        o = r.get("order_of_kinds")
        if o is None:
            o = r.get("heads", {}).get("kind_baseline", {}).get("order_of_kinds")
        if o is None:
            say(f"OBS odds compare: {label}: no order of kinds in {rel}")
            continue
        say(f"OBS odds compare: {label}: " + "; ".join(f"{x['kind']} {x['term']:+.3f}" for x in o))


# ------------------------------------------------------------------------------------------------------------------ main

def main(argv: "list | None" = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["starts", "gen", "matches", "compare", "movement"])
    ap.add_argument("gen", nargs="?", type=int, default=0)
    ap.add_argument("--games-per-start", type=int, default=40)
    ap.add_argument("--seconds", type=float, default=0.05)
    ap.add_argument("--max-plies", type=int, default=120)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--pairs", type=int, default=50)
    ap.add_argument("--tag", default="full")
    a = ap.parse_args(argv)
    cfg = {"games_per_start": a.games_per_start, "seconds": a.seconds, "max_plies": a.max_plies, "workers": a.workers, "pairs": a.pairs, "tag": a.tag, "opening_plies": OPENING_PLIES, "seed": SEED, "control_factor": CONTROL_FACTOR}
    root = out_root() / a.tag
    root.mkdir(parents=True, exist_ok=True)
    log = open(root / "obs.txt", "a")

    def say(line: str) -> None:
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()
    if a.stage == "starts":
        made = make_starts(a.games_per_start)
        for sid, s in made["starts"].items():
            say(f"OBS odds starts: {sid}: games {s['games']}; text {s['text']}")
        say(f"OBS odds starts: {len(made['starts'])} starts; sides {made['sides']}; not removable: {made['refused'] or 'none'}")
    elif a.stage == "gen":
        stage_gen(a.gen, cfg, say)
        stage_movement(cfg, say)
    elif a.stage == "matches":
        stage_matches(cfg, say)
    elif a.stage == "movement":
        stage_movement(cfg, say)
    else:
        stage_compare(cfg, say)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
