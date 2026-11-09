"""PLAN 78.18 THE RANK BENCH: how good is a RANK SOURCE (something of the shape `rank(variant, moves) -> [(move, score)]`, best first)? Two readings, both generic
over the engine's games (positions come from a wake log; legality and outcomes from `cs_engine.state`):

  1. AGREEMENT with a reference ranking on logged positions (the reference is a teacher: training only, this is a measurement of the tool, never a decision):
     top-1 agreement, the reference's best move's rank in the source's order (mean, median, share within the top three), and the source's cost a position.
  2. A MATCH of two rank sources playing each other (the first move of each source's ranking), colours alternating, seeded random opening plies, the game's own outcome
     or a position met three times (the record's draw) or the ply cap -> score of A with its standard error.

  .venv/bin/python -m catspace.incubator.analyst.discovery.rank_bench chess --log <wake.jsonl> --positions 40 --teacher /opt/homebrew/bin/stockfish --source uniform --source greedy+balance
  .venv/bin/python -m catspace.incubator.analyst.discovery.rank_bench chess --match uniform greedy+balance --games 20 --seconds 0.2

A SOURCE NAME is sleep's (`mcts`, or `mcts:<rollout>/<leaf>` with the options of `cs_engine.estimate`: rollout uniform | greedy | gain | lemma, leaf draw | balance) or the
short `<rollout>[+<leaf>]`; `uniform` alone is today's plain Monte Carlo tree search. The store (snapshot) named gives the judge the balance leaf and the lemma prior read.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
import time

import cs_engine as CE

from catspace.incubator.analyst.discovery import game_store as GS
from catspace.incubator.analyst.discovery import sleep as SL
from catspace.io import paths

DRAW = [50, 50]


def source_of(name: str, seconds: float, seed: int = 1, store: "str | None" = None) -> "SL.EstimateSource":
    """A rank source from its name: sleep's `source_of` names (`mcts`, `mcts:<rollout>/<leaf>`); a bare `<rollout>[+<leaf>]` is read as `mcts:<rollout>/<leaf>`."""
    if name == "human" or name.startswith("human:"):
        return SL.source_of(name, seconds, seed, store)
    if not name.startswith("mcts"):
        rollout, _, leaf = name.partition("+")
        name = f"mcts:{rollout}/{leaf}" if leaf else f"mcts:{rollout}/draw" if rollout != "uniform" else "mcts"
    return SL.source_of(name, seconds, seed, store)


def sample_positions(log: str, n: int, seed: int = 1, min_plies: int = 4) -> list:
    """`n` decided positions of a wake log, spread over its games (a round-robin over the games, each game's decisions shuffled by `seed`) -> [{id, variant, moves_before}]."""
    decisions, games = SL.read_log(log)
    by_game: dict = {}
    for (run, g, ply), d in sorted(decisions.items()):
        if ply < min_plies or (run, g) not in games:
            continue
        by_game.setdefault((run, g), []).append((ply, d))
    rng = random.Random(seed)
    queues = []
    for key in sorted(by_game):
        lst = by_game[key]
        rng.shuffle(lst)
        queues.append((key, lst))
    out = []
    while len(out) < n and any(q for _, q in queues):
        for key, q in queues:
            if q and len(out) < n:
                ply, d = q.pop()
                game = games[key]
                out.append({"id": f"{key[0]}:{key[1]}:{ply}", "variant": game["variant"], "moves_before": list(game["moves"][:ply]), "role": d["role"]})
    return out


def agreement(positions: list, reference, source) -> dict:
    """Rank every position by `reference` and `source` -> {positions, top1, mean_rank, median_rank, top3, seconds, ranks}. A rank is 1-based; a reference best the
    source never listed counts as the number of legal moves plus one."""
    ranks, agree, secs = [], 0, 0.0
    for p in positions:
        ref = reference.rank(p["variant"], p["moves_before"])
        if not ref:
            continue
        best = ref[0][0]
        t0 = time.perf_counter()
        ours = source.rank(p["variant"], p["moves_before"])
        secs += time.perf_counter() - t0
        order = [m for m, _ in ours]
        r = order.index(best) + 1 if best in order else len(ref) + 1
        ranks.append(r)
        agree += int(order[:1] == [best])
    n = len(ranks)
    return {"positions": n, "top1": agree / n if n else None, "mean_rank": statistics.fmean(ranks) if n else None, "median_rank": statistics.median(ranks) if n else None,
            "top3": sum(1 for r in ranks if r <= 3) / n if n else None, "seconds": secs / n if n else None, "ranks": ranks}


def play_match(variant: str, a, b, games: int, *, random_plies: int = 4, max_plies: int = 160, seed: int = 1) -> dict:
    """`games` games of `a` against `b` (A is white in the even games), one position at a time -> {score, se, wins, draws, losses, plies, unfinished, seconds}.
    A source's move is the first of its ranking; the outcome is the game's own (role 0's share of 100) or the record's draw or unfinished at the cap (a draw)."""
    rng = random.Random(seed)
    results, plies_all, unfinished = [], [], 0
    t0 = time.perf_counter()
    for gi in range(games):
        a_white = gi % 2 == 0
        moves, seen, outcome = [], {}, None
        while True:
            _fen, mover, legal, out, key = CE.state(variant, None, moves)
            if out is not None:
                outcome = list(out)
                break
            if not legal or len(moves) >= max_plies:
                unfinished += 1
                outcome = list(DRAW)
                break
            seen[key] = seen.get(key, 0) + 1
            if seen[key] >= 3:
                outcome = list(DRAW)
                break
            if len(moves) < random_plies:
                moves.append(sorted(legal)[rng.randrange(len(legal))])
                continue
            white_to_move = len(moves) % 2 == 0
            src = a if (white_to_move == a_white) else b
            ranked = src.rank(variant, moves)
            moves.append(ranked[0][0] if ranked else sorted(legal)[0])
        a_share = outcome[0] if a_white else outcome[1]
        results.append(a_share / 100.0)
        plies_all.append(len(moves))
    n = len(results)
    score = statistics.fmean(results) if n else None
    se = (statistics.pstdev(results) / math.sqrt(n)) if n > 1 else None
    return {"games": n, "score": score, "se": se, "wins": sum(1 for r in results if r > 0.5), "draws": sum(1 for r in results if r == 0.5), "losses": sum(1 for r in results if r < 0.5),
            "plies": statistics.fmean(plies_all) if n else None, "unfinished": unfinished, "seconds": time.perf_counter() - t0}


def main(argv: list) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("game")
    ap.add_argument("--variant", default=None)
    ap.add_argument("--log", default=None, help="a wake log to draw positions from (agreement)")
    ap.add_argument("--positions", type=int, default=40)
    ap.add_argument("--teacher", default=None, help="the reference UCI engine (training only: a measurement)")
    ap.add_argument("--teacher-ms", type=int, default=200)
    ap.add_argument("--source", action="append", default=[], help="a rank source name <rollout>[+<leaf>] (repeatable)")
    ap.add_argument("--seconds", type=float, default=0.2)
    ap.add_argument("--store", default=None, help="a snapshot file (default: the game's HEAD, if any; 'none' for no store)")
    ap.add_argument("--match", nargs=2, default=None, metavar=("A", "B"))
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--random-plies", type=int, default=4)
    ap.add_argument("--max-plies", type=int, default=160)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--tag", default=None)
    a = ap.parse_args(argv)
    variant = a.variant or a.game
    store = a.store
    if store is None and GS.head(a.game):
        store = str(GS.path_of(a.game, GS.head(a.game)))
    if store == "none":
        store = None
    tag = a.tag or time.strftime("%Y%m%d-%H%M%S")
    rep = {"game": a.game, "variant": variant, "store": store, "seconds": a.seconds, "agreement": {}, "match": None}
    if a.log and a.teacher:
        positions = sample_positions(a.log, a.positions, seed=a.seed)
        ref = SL.UciTeacher(a.teacher, movetime_ms=a.teacher_ms)
        try:
            for name in a.source or ["uniform"]:
                src = source_of(name, a.seconds, seed=a.seed, store=store)
                r = agreement(positions, ref, src)
                rep["agreement"][name] = r
                print(f"OBS rank agreement {a.game} {name}: {r['positions']} positions, top1 {r['top1']:.3f}, top3 {r['top3']:.3f}, mean rank {r['mean_rank']:.2f} (median {r['median_rank']:.0f}), "
                      f"{r['seconds']:.3f} s a position (reference {a.teacher_ms} ms)")
        finally:
            ref.close()
    if a.match:
        na, nb = a.match
        src_a, src_b = source_of(na, a.seconds, seed=a.seed, store=store), source_of(nb, a.seconds, seed=a.seed + 1, store=store)
        m = play_match(variant, src_a, src_b, a.games, random_plies=a.random_plies, max_plies=a.max_plies, seed=a.seed)
        rep["match"] = {"a": na, "b": nb, **m}
        print(f"OBS rank match {a.game} {na} vs {nb}: {m['games']} games, score {m['score']:.3f} (se {m['se']:.3f}), W{m['wins']} D{m['draws']} L{m['losses']}, "
              f"{m['plies']:.1f} plies a game, {m['unfinished']} unfinished, {m['seconds']:.0f} s")
    out = paths.experiment(f"rank_bench_{a.game}_{tag}.json")
    with open(out, "w") as f:
        json.dump(rep, f, indent=1)
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
