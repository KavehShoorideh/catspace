"""PLAN 79.57 PART C (SYSTEM_SPEC 11, blocks "A divergence found and named" and "The support filter"; LAYER 2, piece: rules; ledger rows 58, 57): HOW MANY FACTS A MOVE DISTURBS, and the facts of one state made from the description. Light: no clock, a few tens of games of a few tens of plies, cargo with at most 4 jobs.
  python scripts/facts_moved_print.py measure [<games> [<plies>]]   writes the moves of the first <games> (default 30) games of the records folder lichess_2019-01 to a file (no name of a player is read or written: scripts/ply_profile_print.py games) and runs the cargo
                                                                    measure rust/engine/game/tests/translated_reads.rs `facts_moved`; prints its OBS lines: the relation rows on the graph a state; under the hand rule and under the support rule the rows kept and the rows derived again a move; the rows that are new after a move.
  python scripts/facts_moved_print.py facts [<ply>]                 the graph of the state after <ply> plies (default 12) of the first game of that file, made from the description: each unit with what it bears on and may move to (and the position text of the state).
Needs the records under LCP_DATA (the folder of the repository's data). The cargo side is the test file named above; this script only calls it and prints."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CARGO = ["cargo", "test", "--release", "--offline", "-j", "4", "--manifest-path", str(ROOT / "rust" / "engine" / "Cargo.toml"), "-p", "cs_game", "--test", "translated_reads", "--", "--ignored", "--nocapture"]


def games_file(n: int) -> Path:
    from catspace.io import paths
    out = paths.experiment("facts_moved_games.txt")
    subprocess.run([sys.executable, str(ROOT / "scripts" / "ply_profile_print.py"), "games", "lichess_2019-01", str(out), str(n)], check=True)
    return Path(out)


def run(test: str, env: dict) -> int:
    r = subprocess.run(CARGO + [test], env={**os.environ, **env}, capture_output=True, text=True)
    for ln in r.stdout.splitlines():
        if ln.startswith(("OBS 79.57", "STATE", "FACTS", "tempo", "support of")) or " bears on [" in ln or "bear and may_move rows" in ln:
            print(ln)
    if r.returncode:
        print(r.stdout[-3000:], r.stderr[-3000:])
    return r.returncode


def main(argv) -> int:
    if not argv:
        print(__doc__)
        return 2
    if argv[0] == "measure":
        n = int(argv[1]) if len(argv) > 1 else 30
        f = games_file(n)
        return run("facts_moved", {"CS_FACTS_GAMES": str(f), "CS_FACTS_N": str(n), "CS_FACTS_MAX": argv[2] if len(argv) > 2 else "40"})
    if argv[0] == "facts":
        f = games_file(1)
        return run("facts", {"CS_FACTS_GAMES": str(f), "CS_FACTS_PLY": argv[1] if len(argv) > 1 else "12"})
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
