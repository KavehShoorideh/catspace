"""THE SCHEDULER: the spine of the architecture Kaveh laid out on 2026-09-23 -- "every time a move is made ... we
should have everything calculated ... recalculate every tactic and lemma that applies ... asynchronous calculation
that shifts every time the board state moves ... old stuff deprioritized if no longer relevant".

One priority queue of ANALYSIS TASKS per position. A task has a kind, a CONTENT KEY (what the answer depends on:
the pieces and squares it reads, the move), a FOOTPRINT (the squares whose change would invalidate it), an
importance, and a thunk that computes it. Results live in a content-keyed store, so an answer computed once is
reused wherever the same content recurs -- later in the game, in another line of the search, in another game.
On a new position: tasks whose footprint the move crossed are dropped (their answers stay in the store under the
old content key, harmless), the rest are kept with their importance decayed; producers add the new position's tasks
with an importance from the prior. `run(budget)` pops by importance until the budget is spent. Generic: kinds,
keys and footprints come from the producers; nothing here knows chess.

    python -m catspace.incubator.analyst.scheduler gauge --games 2      # reuse per move over a Maia game
"""
from __future__ import annotations

import argparse
import heapq
import itertools
import json
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

OUT = Path("data/derived/player")


@dataclass(order=True)
class Task:
    priority: float
    seq: int
    kind: str = field(compare=False)
    key: str = field(compare=False)
    footprint: frozenset = field(compare=False)
    fn: Callable[[], Any] = field(compare=False)
    ply: int = field(compare=False, default=0)


class Scheduler:
    def __init__(self, decay: float = 0.7, store_path: Path | None = None):
        self.decay = decay
        self.queue: list[Task] = []; self.pending: dict = {}
        self.store: dict = {}
        self.store_path = store_path
        if store_path and store_path.exists():
            try:
                self.store = json.load(open(store_path))
            except Exception:
                self.store = {}
        self._seq = itertools.count()
        self.stats = Counter()
        self.ply = 0

    # ---------------------------------------------------------------- producers call these
    def want(self, kind: str, key: str, footprint, importance: float, fn: Callable[[], Any]):
        """Ask for an answer. Already in the store: a hit, nothing queued. Already pending: importance raised."""
        if key in self.store:
            self.stats["hit"] += 1; return True
        if key in self.pending:
            t = self.pending[key]
            if -t.priority < importance:
                t.priority = -importance; heapq.heapify(self.queue)
            self.stats["dup"] += 1; return False
        t = Task(-importance, next(self._seq), kind, key, frozenset(footprint), fn, self.ply)
        self.pending[key] = t; heapq.heappush(self.queue, t); self.stats["queued"] += 1
        return False

    def get(self, key: str):
        return self.store.get(key)

    # ---------------------------------------------------------------- the board moved
    def on_position(self, ply: int, changed) -> dict:
        """changed: the squares the last transition touched (the producer's notion: moved from/to, captured, and
        whatever attack sets shifted). Drops pending tasks whose footprint the change crossed; decays the rest."""
        self.ply = ply
        changed = set(changed)
        kept = []; dropped = 0
        for t in self.queue:
            if t.footprint & changed:
                dropped += 1; self.pending.pop(t.key, None)
            else:
                t.priority *= self.decay; kept.append(t)
        self.queue = kept; heapq.heapify(self.queue)
        self.stats["dropped"] += dropped; self.stats["carried"] += len(kept)
        return dict(dropped=dropped, carried=len(kept))

    # ---------------------------------------------------------------- work
    def run(self, budget: int, cost: Callable[[Any], int] = lambda r: 1) -> int:
        """Pop by importance until the budget (in the producer's cost units) is spent. Returns work done."""
        spent = 0
        while self.queue and spent < budget:
            t = heapq.heappop(self.queue)
            self.pending.pop(t.key, None)
            r = t.fn()
            self.store[t.key] = r
            spent += cost(r); self.stats["done"] += 1; self.stats[f"done:{t.kind}"] += 1
        return spent

    def save(self):
        if self.store_path:
            self.store_path.parent.mkdir(parents=True, exist_ok=True)
            json.dump(self.store, open(self.store_path, "w"))


# ---------------------------------------------------------------- the chess producers (thin: content keys and footprints from the rules)
def changed_squares(before, move) -> set:
    """Squares a move touches, plus every square whose attackers changed: the footprint a lemma answer can depend on."""
    import chess
    after = before.copy(stack=False); after.push(move)
    ch = {move.from_square, move.to_square}
    if before.is_en_passant(move):
        ch.add(chess.square(chess.square_file(move.to_square), chess.square_rank(move.from_square)))
    if before.is_castling(move):
        ch |= {chess.A1, chess.D1, chess.H1, chess.F1} if before.turn else {chess.A8, chess.D8, chess.H8, chess.F8}
    for sq in chess.SQUARES:
        for col in (True, False):
            if before.attackers(col, sq) != after.attackers(col, sq):
                ch.add(sq); break
    return ch


def frame_task(app, board, move, max_bodies: int, ranker=None):
    """A body-search task for one root move. The content key is the pieces within two squares of the move's start and
    landing squares (the bodies a search up to four deep draws from, almost always), so a hit is a position whose
    local content is identical; the answer records the bodies it actually used and the scheduler's footprint is
    those squares. (Exact-by-used-bodies reuse, as in the player's graph cache, is the stricter alternative.)"""
    import chess
    from .body_search import body_search
    key = json.dumps(["frame", board.turn, move.uci(), sorted((chess.square_name(s), p.symbol()) for s, p in board.piece_map().items() if chess.square_distance(s, move.from_square) <= 2 or chess.square_distance(s, move.to_square) <= 2)])
    def fn():
        r = body_search(app, board, move.from_square, move, max_bodies=max_bodies, ranker=ranker)
        last = next((l for l in reversed(r["trace"]) if l["verdict"] is not None), None)
        used = sorted({s for l in r["trace"] for s in l["bodies"]})
        return dict(keeps=last["keeps"] if last else 0, loses=last["loses"] if last else 0, bodies=used, levels=len(r["trace"]))
    footprint = {move.from_square, move.to_square} | {s for s in board.piece_map() if chess.square_distance(s, move.from_square) <= 2 or chess.square_distance(s, move.to_square) <= 2}
    return key, footprint, fn


def gauge(games: int, maia_rating: int, budget: int, max_bodies: int, library: Path, proved: Path, log=print, seed: int = 0):
    """Reuse per move over real games: how much of each position's lemma work the store already holds."""
    import random
    import chess
    from .chess_domain import domain
    from .lemma_apply import Applier, load_library
    from .maia import Maia
    from .player import Player, material
    dom = domain(); app = Applier(dom, load_library(library, proved))
    sched = Scheduler(store_path=OUT / "scheduler_store.json")
    opp = Maia(rating=maia_rating); rng = random.Random(seed)
    pl = Player(library, proved, depth=3, node_cap=4000, use_lemmas=False)
    per_move = []
    for g in range(games):
        b = chess.Board(); our = chess.WHITE if g % 2 == 0 else chess.BLACK; ply = 0
        while not b.is_game_over() and ply < 160:
            if b.turn == our:
                legal = list(b.legal_moves)
                hits = 0
                for m in legal:
                    key, fp, fn = frame_task(app, b, m, max_bodies)
                    if sched.want("frame", key, fp, importance=1.0 + (0.5 if b.is_capture(m) else 0.0), fn=fn):
                        hits += 1
                t = time.perf_counter(); spent = sched.run(budget); dt = time.perf_counter() - t
                answered = sum(1 for m in legal if sched.get(frame_task(app, b, m, max_bodies)[0]) is not None)
                per_move.append(dict(ply=ply, legal=len(legal), hits=hits, answered=answered, spent=spent, secs=dt))
                m, _ = pl.choose(b)
            else:
                m = opp.sample(b, temperature=1.0, rng=rng)
            ch = changed_squares(b, m)
            b.push(m); ply += 1
            sched.on_position(ply, ch)
        log(f"[scheduler] game {g + 1}: {len(per_move)} of our moves so far; store {len(sched.store)} answers; stats {dict(sched.stats)}")
        sched.save()
    n = max(1, len(per_move))
    log(f"[scheduler] {games} games vs Maia-{maia_rating}, budget {budget} frame tasks/move: mean legal {sum(p['legal'] for p in per_move) / n:.1f}, "
        f"already in the store {sum(p['hits'] for p in per_move) / n:.1f} ({100 * sum(p['hits'] for p in per_move) / max(1, sum(p['legal'] for p in per_move)):.1f}%), "
        f"answered after the budget {sum(p['answered'] for p in per_move) / n:.1f}, tasks run {sum(p['spent'] for p in per_move) / n:.1f}, {sum(p['secs'] for p in per_move) / n:.2f} s/move; "
        f"dropped by footprint {sched.stats['dropped']}, carried {sched.stats['carried']}")
    opp.close()
    json.dump(dict(per_move=per_move, stats=dict(sched.stats)), open(OUT / "scheduler_gauge.json", "w"), indent=1)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["gauge"]); ap.add_argument("--games", type=int, default=2); ap.add_argument("--maia", type=int, default=1500); ap.add_argument("--budget", type=int, default=12); ap.add_argument("--max-bodies", type=int, default=4)
    ap.add_argument("--library", default="data/derived/lemmas/library_v2_size3.json"); ap.add_argument("--proved", default="data/derived/lemmas/proved_v2_size3.json")
    a = ap.parse_args(argv)
    gauge(a.games, a.maia, a.budget, a.max_bodies, Path(a.library), Path(a.proved))


if __name__ == "__main__":
    main()
