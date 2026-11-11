"""PLAN 78.32 THE EXACT LABELLER: where an endgame tablebase (or any exact solver) covers a position, the result class for the side to move is KNOWN, not estimated, and every legal move is
labelled keeps / loses the class at the cost of one probe. TRAINING ONLY: the wake and the snapshot writer never read this module (tests/test_plan78_32_exam_drill_induce.py).

THE INTERFACE (game-free; the game is the variant string):
  labeller.result(variant, start, moves)       -> "win" / "draw" / "loss" for the side to move after `moves` from `start`, or None when the position is not covered (or the game is over)
  labeller.moves_keep(variant, start, moves)   -> {legal move: keeps} where keeps = the class for the mover after the move is not lower than before it (a move into a position the game
                                                  itself ends counts by the game's outcome); None when the position or one of its successors is not covered
  labeller.wdl(variant, moves, start=None)     -> (w, d, l) one-hot: the call shape of the judge objects `playout_sleep.Verdicts` accepts, so the labeller IS a judge (a position it does not cover
                                                  raises NotCovered)
  labeller.training_only = True, .source, .save(), .close()

`ExactLabeller` holds the generic parts (the memory, the judge call shape, the generic `moves_keep` over the game's own legal moves and terminal results); an instance writes `_class`. The ONE PLACE A
CHESS LIBRARY APPEARS is `SyzygyChess` below (python-chess `chess.syzygy`): the CHESS INSTANCE. `LABELLERS` is the registry by game name (as curriculum.CODECS).

  SyzygyChess  CURSED WINS and BLESSED LOSSES (a win or loss that the fifty-move rule turns into a draw, wdl +-1 in the tables) are counted as DRAWS: the class is what the game's own rules give
               when the position is played out with the clock as it stands. Positions with castling rights are not in the tables: their class is the best of the successors after each legal move
               (the move that castles leaves the rights), so it is still exact. Repetition and the clock history are not known to a position, so they are not applied. Insufficient material is a draw.
               Variants other than "chess" are not covered.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import cs_engine as CE

from catspace.incubator.analyst.discovery import playout_sleep as PS
from catspace.io import paths

CLASSES = PS.CLASSES


class NotCovered(LookupError):
    """The labeller does not cover this position (a judge call on it fails with this)."""


class ExactLabeller:
    """The generic part. A subclass defines `_class(variant, start, moves) -> class for the mover of a NON-TERMINAL position, or None`."""
    training_only = True
    source = "exact"

    def __init__(self):
        self.memory: dict = {}
        self.asked = self.hits = 0

    # -- what a subclass writes
    def _class(self, variant: str, start, moves: list) -> "str | None":
        raise NotImplementedError

    # -- the interface
    def result(self, variant: str, start, moves: list) -> "str | None":
        key = (variant, start, tuple(moves))
        self.asked += 1
        if key in self.memory:
            self.hits += 1
            return self.memory[key]
        _fen, mover, legal, outcome, _k = CE.state(variant, start, list(moves))
        c = None if (mover is None or outcome is not None or not legal) else self._class(variant, start, list(moves))
        self.memory[key] = c
        return c

    def moves_keep(self, variant: str, start, moves: list) -> "dict | None":
        v = PS.Verdicts(self, variant)
        _fen, mover, legal, outcome, _k = CE.state(variant, start, list(moves))
        if mover is None or outcome is not None or not legal:
            return None
        role = PS.WK.ROLES[mover]
        try:
            before = v.of(start, moves)
            return {m: PS.rank_of(v.after(start, list(moves) + [m], role)) >= PS.rank_of(before) for m in legal}
        except NotCovered:
            return None

    def wdl(self, variant: str, moves: list, start=None) -> tuple:
        c = self.result(variant, start, list(moves))
        if c is None:
            raise NotCovered(f"{self.source}: position not covered ({len(moves)} moves from {start})")
        return tuple(1.0 if c == k else 0.0 for k in ("win", "draw", "loss"))

    def save(self) -> None:
        pass

    def close(self) -> None:
        self.save()


# ------------------------------------------------------------------------------------------------------------------------------------ the chess instance

class SyzygyChess(ExactLabeller):
    """THE CHESS INSTANCE: Syzygy tablebases through python-chess (`chess.syzygy`). `directory`: the tables (default `paths.syzygy_dir()`); `cache`: a JSON file remembering the class of every
    position probed (key: the position's EPD, without the move counters), so re-runs and the corpus repeat; None keeps the memory only."""
    source = "syzygy"
    SYMBOL = {"win": "w", "draw": "d", "loss": "l"}

    def __init__(self, directory: "str | None" = None, cache: "str | None" = None):
        super().__init__()
        self.directory = str(directory or paths.syzygy_dir())
        self.cache_path = Path(cache) if cache else None
        self.disk = json.loads(self.cache_path.read_text()) if self.cache_path and self.cache_path.exists() else {}
        self.unsaved = 0
        self._tb = None
        self._keep_memory: dict = {}

    # python-chess lives only below
    def tablebase(self):
        if self._tb is None:
            import chess.syzygy
            self._tb = chess.syzygy.open_tablebase(self.directory)
        return self._tb

    @staticmethod
    def board(variant: str, start, moves: list):
        import chess
        if variant != "chess":
            return None
        b = chess.Board(start) if start else chess.Board()
        for m in moves:
            b.push_uci(m)
        return b

    def _terminal_or_probe(self, b) -> "str | None":
        """The class for the side to move on board `b`: the game's end, else the table (castling rights: the best successor)."""
        import chess
        if b.is_checkmate():
            return "loss"
        if b.is_stalemate() or b.is_insufficient_material():
            return "draw"
        key = b.epd()
        sym = self.disk.get(key)
        if sym is not None:
            return {v: k for k, v in self.SYMBOL.items()}[sym]
        if b.castling_rights:
            best, ok = -1, True
            for mv in list(b.legal_moves):
                b.push(mv)
                c = self._terminal_or_probe(b)
                b.pop()
                if c is None:
                    ok = False
                    break
                best = max(best, PS.rank_of(PS.flip(c)))          # the class for the mover after the move, seen by the mover
            out = CLASSES[best] if ok and best >= 0 else None
        else:
            try:
                w = self.tablebase().probe_wdl(b)
            except (KeyError, chess.syzygy.MissingTableError):
                return None
            out = "win" if w == 2 else "loss" if w == -2 else "draw"          # (+-1: cursed win / blessed loss: a draw)
        if out is not None:
            self.disk[key] = self.SYMBOL[out]
            self.unsaved += 1
        return out

    def _class(self, variant: str, start, moves: list) -> "str | None":
        b = self.board(variant, start, moves)
        return None if b is None else self._terminal_or_probe(b)

    def moves_keep(self, variant: str, start, moves: list) -> "dict | None":
        """One board, one probe per legal move (the generic `moves_keep` gives the same answers, slower: the test compares them)."""
        _fen, mover, legal, outcome, _k = CE.state(variant, start, list(moves))
        if mover is None or outcome is not None or not legal:
            return None
        b = self.board(variant, start, moves)
        if b is None:
            return None
        before = self._terminal_or_probe(b)
        if before is None:
            return None
        out = {}
        for m in legal:
            try:
                b.push_uci(m)
            except ValueError:
                return None
            after = self._terminal_or_probe(b)
            b.pop()
            if after is None:
                return None
            flipped = PS.flip(after)
            out[m] = PS.rank_of(flipped) >= PS.rank_of(before)
        return out

    def save(self) -> None:
        if self.cache_path is None or not self.unsaved:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.cache_path.with_suffix(self.cache_path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.disk))
        os.replace(tmp, self.cache_path)
        self.unsaved = 0

    def close(self) -> None:
        try:
            self.save()
        finally:
            tb, self._tb = self._tb, None
            if tb is not None:
                tb.close()


LABELLERS: dict = {"chess": SyzygyChess}          # by game name: the one registry (add a game by adding its labeller here)


def labeller_of(game: str, **kw) -> ExactLabeller:
    """The exact labeller of `game` (chess: Syzygy, the cache file under paths.derived)."""
    if game not in LABELLERS:
        raise ValueError(f"no exact labeller for {game!r}; known: {sorted(LABELLERS)}")
    if game == "chess" and "cache" not in kw:
        kw["cache"] = paths.derived("exact_cache_syzygy_chess.json")
    return LABELLERS[game](**kw)


# ------------------------------------------------------------------------------------------------------------------------------------ the check against an outside engine

def agree_with_engine(labeller: ExactLabeller, game: str, size: int, *, variant: "str | None" = None, engine: str, depth: int = 20, options: "dict | None" = None, n: int = 200,
                      seed: int = 1, say=print) -> dict:
    """The labeller's class against an outside engine's WDL (UciWdlJudge, with its own tablebase option) on `n` random positions of `size` units: random legal play from the curriculum's starts.
    -> {agree, total, disagree: [...]}; prints `OBS exact agree: N of M`."""
    import random

    from catspace.incubator.analyst.discovery import curriculum as CU
    from catspace.incubator.analyst.discovery import wdl_sleep as WS
    variant = variant or game
    starts = CU.starts(game, size, variant=variant, seed=seed)
    rng = random.Random(seed)
    judge = WS.UciWdlJudge(engine, depth, None, options or {})
    agree, total, bad = 0, 0, []
    try:
        guard = 0
        while total < n and guard < 50 * n:
            guard += 1
            start = rng.choice(starts)
            moves = []
            for _ in range(rng.randint(0, 12)):
                _f, mover, legal, outcome, _k = CE.state(variant, start, moves)
                if mover is None or outcome is not None or not legal:
                    break
                moves.append(rng.choice(sorted(legal)))
            _f, mover, legal, outcome, _k = CE.state(variant, start, moves)
            if mover is None or outcome is not None or not legal:
                continue
            c = labeller.result(variant, start, moves)
            if c is None:
                continue
            total += 1
            e = PS.verdict_class(judge.wdl(variant, moves, start))
            if e == c:
                agree += 1
            else:
                bad.append({"start": start, "moves": list(moves), "exact": c, "engine": e})
    finally:
        judge.close()
    say(f"OBS exact agree: {agree} of {total}")
    return {"agree": agree, "total": total, "disagree": bad}
