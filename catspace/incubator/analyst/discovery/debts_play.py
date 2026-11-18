"""PLAN 78.85 PART THREE: the matches of 78.82 and 78.83 again on the equal reading as it is now (the second reading of `ties`, rust/engine/tactics/src/ledger.rs). SYSTEM_SPEC 5, 7.

INTERFACE. None new: this file sets the output folder of the harness of symbols_play (`OUT`) to this plan's before any run and hands the command line to it. The players, the opening sets, the clock, the load wait and the
reports are symbols_play's and tiebreak_measure's, as they were; the extension that runs is the one on the path. A real script file, so that the spawned workers of the Pool import the harness as it is.

  .venv/bin/python -m catspace.incubator.analyst.discovery.debts_play match --a reasons --b control --set A --pairs 100 --tag full | report --tag full
"""
from __future__ import annotations

from catspace.incubator.analyst.discovery import symbols_play as SP

SP.OUT = ("reasoner_debts_78_85",)                                # set before any run: this plan's outputs (the 78.82, 78.83 and 78.84 runs are read only)


if __name__ == "__main__":
    raise SystemExit(SP.main())
