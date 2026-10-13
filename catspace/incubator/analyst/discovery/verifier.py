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
    return json.dumps([hole["base"], hole.get("piece") or hole["material"][1], hole["k"]] + ([] if defender == "self" else [defender])
                      + ([hole["frame_key"], hole.get("leave")] if hole.get("leave") else []))


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

    def verify(self, hole: dict, term: str | None, schema: str, boards: list | None = None) -> dict:
        """Both stages on the hole's boards; remembered and stored."""
        import analyst_core as A
        boards = boards if boards is not None else hole["stuck"][:100]
        bh = board_hash(boards)
        sp = spec_of(hole, self.defender)
        k = _key(hole["name"], term or "", schema, bh, sp)
        if k in self.memo:
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
                     states=o["states"], why=o["why"], examples=o["examples"], score=D.score(o), t=time.time())
        self.memo[k] = r
        with open(self.path, "a") as fh:
            fh.write(json.dumps(r) + "\n")
        return r
