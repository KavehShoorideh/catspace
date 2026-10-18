"""THE ANALYSER (ANALYST_PLAN 52.2): the state, the agenda trace, the claims, the gaps, and one OBS line per puzzle.

  .venv/bin/python -m catspace.incubator.analyst.discovery.analyse --set dev_20 --id 01IFU
  .venv/bin/python -m catspace.incubator.analyst.discovery.analyse --set dev_20            # the batch: one OBS line per puzzle + a tally (--quiet: OBS only)
"""
from __future__ import annotations

import argparse
import time
from collections import Counter

import chess

from catspace.io import paths
from catspace.incubator.analyst.discovery import base_vocabulary as BV
from catspace.incubator.analyst.discovery import puzzle_apply as PA
from catspace.incubator.analyst.discovery import remember as RM
from catspace.incubator.analyst.discovery.agenda import Agenda
from catspace.incubator.analyst.discovery.state import state_of


AGENDA_STORE = "agenda_states.jsonl"       # 52.9: the agenda's own store (plan 50's reasoned_states.jsonl keeps its record of the exhaustive ladder)


def analyse(p: dict, quiet: bool = False, log=print, memory: bool = True, store: bool = False, set_name: str = "", k: int = 3) -> dict:
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
    recalled = RM.recall(st.tokens(), k, exclude_id=p["id"], path=paths.derived(AGENDA_STORE)) if memory else []
    if not quiet:
        for sim, r in recalled:
            log(f"recall: seen {r['id']} (sim {sim}): {r.get('goal')} by {r.get('lemma')}, fixes {r.get('fixes')}")
    ag = Agenda(st, memory=recalled)
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
    if store and d is not None:
        rec = RM.make_record(p, set_name, fen, us, d.move, d.lemma, d.value, d.move[:4] == p["answer"][:4], RM.threat_token(fen, us, check=b.is_check()),
                             goal=d.goal, fixes=d.fixes, lemma=d.lemma)
        RM.store(rec, paths.derived(AGENDA_STORE))
    decided = d.what if d else "none"
    # a depth-two decision reads "Nxf6+ then Bxh7": it MATCHES the puzzle when its FIRST move is the answer (the second move is the plan's, not the puzzle's)
    verdict = "MATCH" if d and d.move[:4] == p["answer"][:4] and (len(d.move) < 5 or len(p["answer"]) < 5 or d.move[4] == p["answer"][4]) else "MISS"
    log(f"OBS agenda {p['id']}: questions asked {ag.asked}, killed by tempo {ag.killed_tempo}, claims {len(st.claims)}, decision {decided} by "
        f"{d.lemma if d else '-'}, answer {ans} {verdict}, their agendas {ag.their_runs}, seconds {secs}")
    return dict(id=p["id"], asked=ag.asked, tempo=ag.killed_tempo, claims=len(st.claims), decision=decided, match=verdict == "MATCH",
                decided=d is not None, kills=[r.split(":")[0] for _, r in ag.kills], gaps=len(st.gaps), secs=secs, theirs=ag.their_runs, until=ag.asked_at_claim, move=d.move if d else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="dev_20")
    ap.add_argument("--id")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--store", action="store_true", help="write a record per decided puzzle (per-item checkpoint: stored ids are skipped)")
    ap.add_argument("--no-memory", action="store_true", help="do not recall: the stake-only order")
    a = ap.parse_args()
    ps = PA.load_set(a.set)
    if a.store:
        done = RM.stored_ids(paths.derived(AGENDA_STORE))
        ps = [p for p in ps if p["id"] not in done] if not a.id else ps
    if a.id:
        analyse(next(x for x in ps if x["id"] == a.id), memory=not a.no_memory, store=a.store, set_name=a.set)
        return
    res = [analyse(p, quiet=True, memory=not a.no_memory, store=a.store, set_name=a.set) for p in ps]
    if a.store:
        print(f"OBS agenda store: {len(RM.load(paths.derived(AGENDA_STORE)))} records")
    for r in res:
        print(f"OBS questions asked until the deciding claim {r['id']}: {r['until']} (of {r['asked']} asked), decision {r['decision']} {r['move']}")
    print(f"OBS questions asked until the deciding claim, total: {sum(r['until'] for r in res)} (of {sum(r['asked'] for r in res)} asked)")
    kills = Counter(k for r in res for k in r["kills"])
    dec = [r["id"] for r in res if r["decided"]]
    gap = [r["id"] for r in res if not r["decided"]]
    print(f"OBS agenda {a.set}: {len(res)} puzzles, decided {len(dec)} ({' '.join(dec)}), MATCH {sum(r['match'] for r in res)}, "
          f"not decided {len(gap)} ({' '.join(gap)}), questions asked {sum(r['asked'] for r in res)}, killed by tempo {sum(r['tempo'] for r in res)}, "
          f"killing reasons {dict(kills.most_common())}, their agendas {sum(r['theirs'] for r in res)}, seconds {round(sum(r['secs'] for r in res), 1)}")


if __name__ == "__main__":
    main()
