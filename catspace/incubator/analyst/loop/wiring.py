"""THE PLAYER'S WIRING FOR A ROUND: active.json (written by the use stage) turned into switches.
- closures of proved ideas -> the zone search's existing switches (TacticalSpec.proved_mates, proved_passer and the
  passer's candidate push), exactly as the Lean lemmas are wired by hand;
- order uses of mined (conjecture-tier) ideas -> frame_player.LOOP_ORDER: among the moves the tactics tie, the moves
  the lemmas' composed verdicts rate highest stay on top (lemma_apply's Applier and compose). A tie-break only: it
  never changes a value the search computed.
frame_puzzles --loop-active <active.json> calls activate()."""
from __future__ import annotations

from pathlib import Path

import chess


class MinedOrder:
    """The order use of mined ideas: a callable (board, tied ucis) -> {uci: score}, and a fire scan."""

    def __init__(self, entries: list[dict]):
        from ..chess_domain import domain
        from ..lemma_apply import Applier, load_library
        libs: dict = {}
        compiled = []
        for e in entries:
            lib = e.get("library")
            if not lib:
                continue
            if lib not in libs:
                libs[lib] = {c.idx: c for c in load_library(Path(lib), None, proved_only=False)}
            c = libs[lib].get(e.get("index"))
            if c is not None:
                compiled.append(c)
        self.ids = sorted({"mined:" + c.id for c in compiled})
        self.app = Applier(domain(), compiled) if compiled else None
        self.fired = 0

    def __call__(self, b: chess.Board, ucis: list[str]) -> dict:
        from ..lemma_apply import compose
        if self.app is None:
            return {}
        res = self.app.apply(b)
        out = {}
        spoke = False
        for u in ucis:
            m = chess.Move.from_uci(u)
            vs = res["verdicts"].get((m.from_square, u), [])
            if not vs:
                out[u] = 0.0
                continue
            spoke = True
            v = compose(vs)
            out[u] = v if v is not None else float(sum(p == "keeps" for _, p, _, _ in vs) - sum(p == "loses" for _, p, _, _ in vs))
        self.fired += spoke
        return out if spoke else {}

    def fire(self, fens: list[str]) -> dict:
        """Per idea: positions where its frame embeds (instances) and where it gives a verdict on a move (speaks)."""
        out = {i: dict(positions=len(fens), instances=0, speaks=0) for i in self.ids}
        if self.app is None:
            return out
        for f in fens:
            res = self.app.apply(chess.Board(f))
            inst = {"mined:" + c.id for c, _ in res["embeddings"]}
            speaks = {"mined:" + c.id for vs in res["verdicts"].values() for c, _, _, _ in vs}
            for i in inst:
                if i in out:
                    out[i]["instances"] += 1
            for i in speaks:
                if i in out:
                    out[i]["speaks"] += 1
        return out


def activate(active: dict, FP, spec_kw: dict) -> str:
    """Set the player's switches from active.json. -> a tag suffix for the dump's name."""
    closures = set(active.get("closures", []))
    if "lean:backRankMate" in closures:
        spec_kw["proved_mates"] = 1
    if closures & {"lean:unstoppablePasser", "lean:unstoppablePasser2"}:
        spec_kw["proved_passer"] = True
        FP.PASSER = True
    if active.get("order"):
        FP.LOOP_ORDER = MinedOrder(active["order"])
    return f"_loopactive{int(active.get('round', 0)):03d}"
