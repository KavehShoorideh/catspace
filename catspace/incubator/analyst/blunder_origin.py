"""WHY DOES THE PLANNER BLUNDER? (Kaveh 2026-10-10: "with a logic-based play there should be no blunders ... whatever the
blunder guard is doing should be part of the main planner's job -- so why are we not getting that?"). For every
decision the after-check warns about (avoid off): which phase made it (an immediate win, a defence, an armed threat,
the cheapest loss) and whether their winning reply after it was ALREADY their threat before our move (their best
immediate win on the null move -- the planner saw it and did not stop it) or NEW (our move created it)."""
from __future__ import annotations

import os
import re
from collections import Counter


def _one(fen):
    os.environ["RAYON_NUM_THREADS"] = "1"
    import analyst_core
    analyst_core.set_after_check(True)
    analyst_core.set_avoid(False)
    tr, dec, *_ = analyst_core.plan_trace(fen)
    warn = next((l for l in tr if l.startswith("after:")), None)
    if not warn or not dec:
        return None
    kind = {"": "immediate win", "defend": "defence", "cheapest": "cheapest loss"}.get(dec[0], "armed (preparing move)")
    m = re.search(r"lets them (\S+) wins", warn)
    reply = m.group(1) if m else "?"
    before = next((l for l in tr if l.startswith("their best immediate win if we pass:")), "")
    already = reply != "?" and before.split(": ", 1)[-1].startswith(reply + " ")
    sub = ""
    if kind == "immediate win":
        sub = "answered by the rule 'a certified win answers a material threat'" if any("also stops their threat" in l for l in tr) else (
              "taken as worth at least their threat" if any(": take it" in l for l in tr) else "no threat of theirs before")
    elif kind == "defence":
        sub = "a defence of ONE threat, their win is another" if not already else "the defence does not stop it"
    return kind, ("already their threat" if already else "created by our move"), sub, fen, dec[3], warn


def main():
    from multiprocessing import Pool
    from .waiting_frames import _games
    fens = [f for _, fs in _games("c300", 100) for f in fs]
    with Pool(os.cpu_count()) as p:
        rs = [r for r in p.map(_one, fens, chunksize=8) if r]
    print(f"OBS warned decisions {len(rs)} of {len(fens)} positions")
    print("OBS by phase and origin:", dict(Counter((k, o) for k, o, *_ in rs).most_common()))
    print("OBS immediate wins / defences by detail:", dict(Counter((k, s) for k, o, s, *_ in rs if s).most_common()))
    shown = Counter()
    for k, o, s, fen, d, w in rs:
        if shown[(k, o)] < 2:
            shown[(k, o)] += 1
            print(f"  [{k} / {o}] {fen}  {d}  |  {w[:110]}")


if __name__ == "__main__":
    main()
