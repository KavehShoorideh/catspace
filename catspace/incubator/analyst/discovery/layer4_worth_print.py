"""PLAN 79.26 PART A: THE MAKER OF LAYER 4'S WORTH FIXTURE (SYSTEM_SPEC 11 "the layers", 8 the wake; ledger row 36; piece: tactical reasoner).
A PRINTING SCRIPT (it may name the game's kinds because it only writes a data file). It writes rust/engine/tactics/tests/fixtures/worth.fixture: the worths of each kind from the record the wake plays with (the human-result record of
plan 78.59, `record_play.worth_of("human")`: worths read from results, none assumed), so that layer 4's test can judge with the same table the wake's judge holds (`ledger::Worth`, `ours_first` off). It fits nothing and calls no outside program.

  .venv/bin/python -m catspace.incubator.analyst.discovery.layer4_worth_print OUT_FILE
"""
from __future__ import annotations

import sys
from pathlib import Path

from catspace.incubator.analyst.discovery import record_play as RP


def main(argv: "list | None" = None) -> int:
    out = Path((argv or sys.argv[1:])[0])
    w = RP.worth_of("human")
    lines = ["note\tthe worths the wake's judge plays with (the human-result record of plan 78.59, worth = round(1000 x term)); a kind the record does not name is worth nothing in the judge, as in the wake"]
    lines += [f"worth\t{k}\t{int(v)}" for k, v in sorted(w.items())]
    out.write_text("\n".join(lines) + "\n")
    print(f"OBS 79.26 worth fixture: {len(w)} kinds written to {out}: {sorted(w.items())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
