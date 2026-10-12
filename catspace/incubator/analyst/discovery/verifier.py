"""THE VERIFIER and THE EXPERIENCE STORE (Kaveh 2026-10-12: the verifier checks everything the filter kept; "how do you
give feedback back to the proposer? if they tried a lemma that didn't work, do you provide negative feedback?").

  STAGE 1  certification on the hole's stuck boards (every reply, within two moves) -- cheap, but GAMEABLE: a measure
           we can lower ourselves ("how many squares we attack") certifies every stuck board
  STAGE 2  the closed loop: the candidate measure plays from the hole's boards against the measure's adversary; scored
           by the games mated and the moves certified along them -- a gameable measure mates almost none
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


def _key(hole: str, term: str, schema: str) -> str:
    return f"{hole}|{schema}|{term}"


class Verifier:
    def __init__(self, store: str = STORE):
        self.path = paths.derived(store)
        self.memo = {}
        if os.path.exists(self.path):
            for line in open(self.path):
                r = json.loads(line)
                self.memo[_key(r["hole"], r["term"], r["schema"])] = r

    def verify(self, hole: dict, term: str | None, schema: str, boards: list | None = None) -> dict:
        """Both stages on the hole's boards; remembered and stored."""
        import analyst_core as A
        k = _key(hole["name"], term or "", schema)
        if k in self.memo:
            return self.memo[k]
        boards = boards if boards is not None else hole["stuck"][:100]
        piece = hole["material"][1]
        s1 = A.verify_count(boards, hole["base"], term, schema, hole["k"], piece)
        mated, moves, played, cert = A.verify_play(boards, hole["base"], term, schema, piece, LIMIT)
        n = len(boards)
        r = dict(hole=hole["name"], term=term or "", schema=schema, n=n, stage1=s1, mated=mated, played=played, certified=cert,
                 score=(cert / max(played, 1)) * (mated / max(n, 1)), t=time.time())
        self.memo[k] = r
        with open(self.path, "a") as fh:
            fh.write(json.dumps(r) + "\n")
        return r
