"""THE VERIFIER and THE EXPERIENCE STORE (Kaveh 2026-10-12: the verifier checks everything the filter kept; "how do you
give feedback back to the proposer? if they tried a lemma that didn't work, do you provide negative feedback?").

  STAGE 1  certification on the hole's stuck boards (every reply, within two moves) -- cheap, but GAMEABLE: a measure
           we can lower ourselves ("how many squares we attack") certifies every stuck board
  STAGE 2  the closed loop: the candidate measure plays from the hole's boards against the DEFENDER (a switch,
           defenders.py: self -- the measure's adversary; tablebase; stockfish:D); scored by the games won and the moves
           certified along them -- a gameable measure wins almost none
  SCORE    certified fraction of the closed-loop moves x mated fraction (both must hold)

Every verdict -- success or failure -- is appended to the EXPERIENCE STORE (data/derived/discovery_experience.jsonl): the
filter's memo (an exact (hole, term, schema) is never verified twice) and the learned proposers' training data (a failure
lowers the terms that look like it, not only itself).
"""
from __future__ import annotations

import json
import os
import time

from catspace.io import paths

STORE = "discovery_experience.jsonl"
LIMIT = 60                                          # moves per closed-loop game


def _key(hole: str, term: str, schema: str, boards: str = "", spec: str = "") -> str:
    return f"{hole}|{schema}|{term}|{boards}|{spec}"


def spec_of(hole: dict, defender: str = "self") -> str:
    """What a verdict was made against besides the term and the boards: the hole's base measure, the safety, the horizon,
    the closed loop's defender -- a hole of the same name and boards with another base (the next run's) must not reuse it
    (BUGS 2026-10-12), nor a verdict against another opponent."""
    # a stage (a situational frame) is judged under its own key, with or without a region to leave
    stage = hole.get("leave") or ":if:" in (hole.get("frame_key") or "")
    return json.dumps([hole["base"], hole.get("piece") or hole["material"][1], hole["k"]] + ([] if defender == "self" else [defender])
                      + ([hole["frame_key"], hole.get("leave")] if stage else []))


def board_hash(boards: list) -> str:
    """The verdict depends on the boards: a hole of the same name with other boards (a new round of surprises) must not
    reuse it (BUGS 2026-10-12)."""
    import hashlib
    return hashlib.sha1("\n".join(boards).encode()).hexdigest()[:12]


class Verifier:
    def __init__(self, store: str = STORE, defender: str = "self"):
        from .defenders import parse
        parse(defender)
        self.defender = defender                     # stage 2's opponent (defenders.py): self, tablebase, stockfish:D
        self.path = paths.derived(store)
        self.memo = {}
        if os.path.exists(self.path):
            for line in open(self.path):
                r = json.loads(line)
                self.memo[_key(r["hole"], r["term"], r["schema"], r.get("boards", ""), r.get("spec", ""))] = r

    def verify_many(self, hole: dict, cands: list, boards: list) -> list:
        """verify for several candidates [(term, schema)] on the same boards: those the store has not seen played in ONE
        pool (a real defender only) -> a record a candidate, in order."""
        from . import defenders as D
        if D.parse(self.defender)[0] == "self":
            return [self.verify(hole, t, s, boards) for t, s in cands]
        bh, sp = board_hash(boards), spec_of(hole, self.defender)
        key = lambda t, s: _key(hole["name"], t or "", s, bh, sp)
        todo = [(t, s) for t, s in cands if not (key(t, s) in self.memo and "per_board" in self.memo[key(t, s)])]
        if todo:
            import analyst_core as A
            piece = hole.get("piece") or hole["material"][1]
            outs = D.closed_loop_many(boards, hole["base"], todo, piece, LIMIT, self.defender, player="planner",
                                      material=hole.get("frame_key") or hole["material"], frames=hole.get("frames"),
                                      leave=tuple(hole["leave"]) if hole.get("leave") else None)
            with open(self.path, "a") as fh:
                for (t, s), o in zip(todo, outs):
                    r = dict(hole=hole["name"], term=t or "", schema=s, boards=bh, spec=sp, defender=self.defender, n=len(boards),
                             stage1=A.verify_count(boards, hole["base"], t, s, hole["k"], piece),
                             mated=o["won"], lost=o["lost"], played=o["claims"], certified=o["kept"], ranks=list(D.ranked(o)),
                             states=o["states"], why=o["why"], examples=o["examples"], per_board=o["per_board"], score=D.score(o), t=time.time())
                    self.memo[key(t, s)] = r
                    fh.write(json.dumps(r) + "\n")
        return [self.memo[key(t, s)] for t, s in cands]

    def verify(self, hole: dict, term: str | None, schema: str, boards: list | None = None) -> dict:
        """Both stages on the hole's boards; remembered and stored."""
        import analyst_core as A
        boards = boards if boards is not None else hole["stuck"][:100]
        bh = board_hash(boards)
        sp = spec_of(hole, self.defender)
        k = _key(hole["name"], term or "", schema, bh, sp)
        if k in self.memo and ("per_board" in self.memo[k] or self.defender == "self"):   # (older verdicts lack the boards' outcomes)
            return self.memo[k]
        piece = hole.get("piece") or hole["material"][1]
        s1 = A.verify_count(boards, hole["base"], term, schema, hole["k"], piece)
        from . import defenders as D
        n = len(boards)
        if D.parse(self.defender)[0] == "self":
            mated, moves, played, cert = D.closed_loop(boards, hole["base"], term, schema, piece, LIMIT, self.defender)
            r = dict(hole=hole["name"], term=term or "", schema=schema, boards=bh, spec=sp, defender=self.defender, n=n, stage1=s1,
                     mated=mated, played=played, certified=cert, score=(cert / max(played, 1)) * (mated / max(n, 1)), t=time.time())
        else:
            # a real defender: the whole player with the candidate as its material's frame, scored by Kaveh's ranks --
            # held, then won, then the frame's claims kept
            o = D.closed_loop(boards, hole["base"], term, schema, piece, LIMIT, self.defender, player="planner",
                              material=hole.get("frame_key") or hole["material"], frames=hole.get("frames"),
                              leave=tuple(hole["leave"]) if hole.get("leave") else None)
            r = dict(hole=hole["name"], term=term or "", schema=schema, boards=bh, spec=sp, defender=self.defender, n=n, stage1=s1,
                     mated=o["won"], lost=o["lost"], played=o["claims"], certified=o["kept"], ranks=list(D.ranked(o)),
                     states=o["states"], why=o["why"], examples=o["examples"], per_board=o["per_board"], score=D.score(o), t=time.time())
        self.memo[k] = r
        with open(self.path, "a") as fh:
            fh.write(json.dumps(r) + "\n")
        return r


def split(boards: list, n_boards: int, games: list, seed: str = "") -> tuple:
    """A hole's boards -> (the boards the pipeline CHOOSES on, the boards the choice is CONFIRMED on, their games): never
    the same GAME on both sides (Kaveh: "split by identity, not by list" -- 2026-10-12: split by list, 27-36 of the 50
    confirmation boards came from games also choosing; the best of ~60 on the boards that chose it is biased upward, and
    the game's other boards share that bias). Whole games, in a fixed shuffled order: the choosing side up to half the
    boards (n_boards at most), the rest confirming (n_boards at most)."""
    import random
    from collections import defaultdict
    by = defaultdict(list)
    for b, g in zip(boards, games):
        by[g].append(b)
    order = sorted(by, key=lambda g: random.Random(f"{seed}|{g}").random())
    half = min(n_boards, len(boards) // 2)
    choose, check, check_games = [], [], []
    for g in order:                                   # (a game joins a side only if it fits: no side overshoots)
        if len(choose) + len(by[g]) <= half or not choose:
            choose += by[g]
        elif len(check) + len(by[g]) <= n_boards:
            check += by[g]; check_games += [g] * len(by[g])
    return choose, check, check_games


def spread(boards: list, games: list, cap: int) -> tuple:
    """Up to `cap` boards taken one game at a time, round robin (games in their first appearance's order) -> (boards,
    games): as many games as the boards allow -- the evidence is one vote a game (2026-10-12: a stage's first 100
    refuted boards came from a few long games, its confirmation from one)."""
    from collections import defaultdict
    by, order = defaultdict(list), []
    for b, g in zip(boards, games):
        if g not in by:
            order.append(g)
        by[g].append(b)
    out, og = [], []
    while len(out) < cap and any(by[g] for g in order):
        for g in order:
            if by[g] and len(out) < cap:
                out.append(by[g].pop(0)); og.append(g)
    return out, og


def confirm(hole: dict, best: dict | None, boards: list, defender: str, games: list | None = None) -> dict:
    """THE PROMOTION TEST: the candidate the pipeline chose and the frame before it (no candidate), both played on the
    confirmation boards, compared board by board and voted ONE GAME ONE VOTE (defenders.paired_e)."""
    from . import defenders as D
    if not best or not boards:
        return dict(promoted=False, w=0, l=0, e=1.0, n=len(boards), games=len(set(games or [])), cand=None, base=None)
    ver = Verifier(defender=defender)
    cand = ver.verify(hole, best["term"], best["schema"], boards)
    base = ver.verify(hole, None, "none", boards)
    w, l, e = D.paired_e(cand["per_board"], base["per_board"], games)
    return dict(promoted=e >= D.E_PROMOTE, w=w, l=l, e=e, n=len(boards), games=len(set(games or [])), cand=cand, base=base)
