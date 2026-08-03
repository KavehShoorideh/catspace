#!/usr/bin/env python
"""tests/build_integration_suite.py -- REGRESSION SUITE HARVESTER (Kaveh 2026-07-25:
'take each of the positions in past runs that failed and later succeeded as test cases.
Future works have to succeed on these specific positions').

Scans every recorded results jsonl, groups by (scenario, game index), and keeps positions
that FAILED in any run (FAIL rows carry the exact start FEN). Kinds:
  must-win        -- failed somewhere, succeeded somewhere (or tb-certified won): the
                     engine must convert these, forever
  clock-pressure  -- synthesized from must-win FENs with halfmove clock set to 88:
                     convert under deadline (kappa + zeroing behavior)
Writes tests/integration_positions.json (name, fen, kind, source history)."""
from __future__ import annotations

import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import chess

from catspace.research.components.planner.approaches.endgame_groundtruth.src.tb import TB
from catspace.io import paths


def main():
    hist = defaultdict(list)          # (scenario, g) -> [(file, mate, start_epd?)]
    for f in sorted(glob.glob(paths.experiment("*results*.jsonl"))):
        scen = None
        for tag in ("KRRvK-central", "KRRvKBP", "KRRvKB", "KRRvKP", "KRvK-technique",
                    "KRRvKBNP-7p"):
            if tag in f:
                scen = tag; break
        if scen is None and "selfloop" in f:
            scen = "KRRvK-central"
        if scen is None:
            continue
        for ln in Path(f).read_text().splitlines():
            if not ln.strip():
                continue
            try:
                r = json.loads(ln)
            except Exception:
                continue
            hist[(scen, r["g"])].append((Path(f).name, bool(r["mate"]), r.get("start_epd")))

    tb = TB()
    cases = []
    for (scen, g), rows in sorted(hist.items()):
        fails = [r for r in rows if not r[1]]
        if not fails:
            continue
        fen = next((r[2] for r in fails if r[2]), None)
        if fen is None:
            continue
        b = chess.Board(fen)
        w, _ = tb.wdl_dtz(b)
        wpov = w if b.turn == chess.WHITE else (-w if w is not None else None)
        if wpov != 2 and len(b.piece_map()) <= 6:
            continue                                    # only tb-certified wins are tests
        succeeded = any(r[1] for r in rows)
        cases.append(dict(
            name=f"{scen}_g{g:03d}",
            fen=b.fen(),
            kind="must-win",
            history=f"failed x{len(fails)}, later-succeeded={succeeded}",
        ))
    tb.close()

    # clock-pressure synthetics from the first few must-wins
    for c in list(cases)[:6]:
        b = chess.Board(c["fen"])
        b.halfmove_clock = 88
        cases.append(dict(name=c["name"] + "_clock88", fen=b.fen(),
                          kind="clock-pressure", history="synth from " + c["name"]))

    out = Path("tests/integration_positions.json")
    out.write_text(json.dumps(cases, indent=1))
    print(f"harvested {len(cases)} cases -> {out}")
    from collections import Counter
    print(Counter(c["kind"] for c in cases))


if __name__ == "__main__":
    main()
