"""miner.py -- a player's TENDENCIES from analyst records (the prep product's core loop).
For every opponent move we have: verdict, the MECHANISM of the refutation (which relation
family the punishing move used), whether a standing threat was ignored, and which families
were merely PRESENT in the position. Tendency = rate of a mechanism among mistakes vs its
presence rate across all of that player's positions (the degenerate baseline). OBS lines only:
one player, one sample, directional.
    python -m catspace.incubator.analyst.miner tmp/maia_records.jsonl"""
from __future__ import annotations

import json
import sys
from collections import Counter


def main(path: str):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    games = {r["game"] for r in rows}
    bad = [r for r in rows if r["verdict"] in ("blunder", "mistake", "inaccuracy")]
    print(f"[miner] {len(games)} games, {len(rows)} opponent moves, {len(bad)} errors "
          f"({Counter(r['verdict'] for r in rows).most_common()})")
    fams = ("pin", "fork", "discovered", "skewer", "hanging", "loose", "attacked_by_lower", "overload", "capture", "mate")
    print(f"{'mechanism':16s} {'in refutations of errors':>26s}   {'present in all positions':>26s}   lift")
    for f in fams:
        m = sum(f in r["mechanisms"] for r in bad); p = sum(f in r["present"] for r in rows)
        rm, rp = m / max(len(bad), 1), p / max(len(rows), 1)
        base = f"{rp:.2f} (n={len(rows)})" if f not in ("capture", "mate") else "--"
        lift = f"x{rm/rp:.1f}" if f not in ("capture", "mate") and rp > 0 else ""
        print(f"OBS {f:16s} {rm:.2f} ({m}/{len(bad)})".ljust(45) + f"{base:>26s}   {lift}")
    ign = sum(r["ignored_threat"] for r in bad); had = sum(r["threat_existed"] for r in rows)
    print(f"OBS ignored a standing threat: {ign}/{len(bad)} errors ({ign/max(len(bad),1):.2f}); a real threat existed in {had}/{len(rows)} positions ({had/max(len(rows),1):.2f})")
    for name, sel in (("opening (ply<=20)", lambda r: r["ply"] <= 20), ("middlegame", lambda r: r["ply"] > 20 and r["phase_nonpawn"] > 12), ("endgame (nonpawn<=12)", lambda r: r["ply"] > 20 and r["phase_nonpawn"] <= 12)):
        rs = [r for r in rows if sel(r)]; bs = [r for r in rs if r["verdict"] in ("blunder", "mistake", "inaccuracy")]
        print(f"OBS {name:22s} error rate {len(bs)}/{len(rs)} = {len(bs)/max(len(rs),1):.2f}; mean drop of errors {sum(b['drop'] for b in bs)/max(len(bs),1):.0f} cp")


if __name__ == "__main__":
    main(sys.argv[1])
