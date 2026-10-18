"""THE ANALYSER (ANALYST_PLAN 52.2): the state, the agenda trace, the claims, the gaps, and one OBS line per puzzle.

  .venv/bin/python -m catspace.incubator.analyst.discovery.analyse --set dev_20 --id 01IFU
  .venv/bin/python -m catspace.incubator.analyst.discovery.analyse --set dev_20            # the batch: one OBS line per puzzle + a tally (--quiet: OBS only)
"""
from __future__ import annotations

import argparse
import time
from collections import Counter

import chess

from catspace.incubator.analyst.discovery import base_vocabulary as BV
from catspace.incubator.analyst.discovery import puzzle_apply as PA
from catspace.incubator.analyst.discovery.agenda import Agenda
from catspace.incubator.analyst.discovery.state import state_of


def analyse(p: dict, quiet: bool = False, log=print) -> dict:
    BV.install_terms()
    t0 = time.time()
    fen = p["fen"]
    b = chess.Board(fen)
    us = "w" if b.turn else "b"
    st = state_of(fen, us)
    if not quiet:
        log(f"position {p['id']}: {fen}  {'white' if b.turn else 'black'} to move; the puzzle's window: gain {p['value']}")
        for line in st.lines():
            log(line)
    ag = Agenda(st)
    ag.ask()
    ans = b.san(chess.Move.from_uci(p["answer"]))
    d = ag.decision()
    secs = round(time.time() - t0, 1)
    if not quiet:
        log("AGENDA (top to bottom, by stake)")
        for n in ag.notes:
            log("  note: " + n)
        log(ag.trace())
        log("CLAIMS")
        for c in st.claims:
            log(f"  {c.kind} {c.what} value {c.value} lemma {c.lemma}: {c.line}")
        if not st.claims:
            log("  none")
        log("GAPS")
        for g in st.gaps or ["none"]:
            log("  " + g)
    decided = d.what if d else "none"
    verdict = "MATCH" if d and d.what == ans else "MISS"
    log(f"OBS agenda {p['id']}: questions asked {ag.asked}, killed by tempo {ag.killed_tempo}, claims {len(st.claims)}, decision {decided} by "
        f"{d.lemma if d else '-'}, answer {ans} {verdict}, seconds {secs}")
    return dict(id=p["id"], asked=ag.asked, tempo=ag.killed_tempo, claims=len(st.claims), decision=decided, match=verdict == "MATCH",
                decided=d is not None, kills=[r.split(":")[0] for _, r in ag.kills], gaps=len(st.gaps), secs=secs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="dev_20")
    ap.add_argument("--id")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    ps = PA.load_set(a.set)
    if a.id:
        analyse(next(x for x in ps if x["id"] == a.id))
        return
    res = [analyse(p, quiet=True) for p in ps]
    kills = Counter(k for r in res for k in r["kills"])
    dec = [r["id"] for r in res if r["decided"]]
    gap = [r["id"] for r in res if not r["decided"]]
    print(f"OBS agenda {a.set}: {len(res)} puzzles, decided {len(dec)} ({' '.join(dec)}), MATCH {sum(r['match'] for r in res)}, "
          f"not decided {len(gap)} ({' '.join(gap)}), questions asked {sum(r['asked'] for r in res)}, killed by tempo {sum(r['tempo'] for r in res)}, "
          f"killing reasons {dict(kills.most_common())}, seconds {round(sum(r['secs'] for r in res), 1)}")


if __name__ == "__main__":
    main()
