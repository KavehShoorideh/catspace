"""PLAN 79.59 PART A (SYSTEM_SPEC 11, block "The lemma miner of layer 2"; LAYER 2, piece: rules; ledger rows 59, 58, 57): THE EXPLANATIONS, printed. A printing script: it may name the games and the relations; it only calls the cargo test and prints what the program says.
  python scripts/lemma_explain_print.py explain [<games> [<plies>]]   for each of the seven games, at every state of the games of a file (the first game) or of seeded walks (the others), every unit of the side to act is asked for its allowed moves, EXPLAINED: every read of a fluent
        the answer made is tagged with the clause and the literal of the description that made it. Prints, for each game, the kinds of read grouped by (fluent, which places are bound, clause, literal) with how many reads a question and in how many questions, and the text of the clause and the
        literal, all from the module's own tables. The support of plan 79.58 (the line, a read of every unit, a read of every unit of the goal kind, two whole fluents) is thereby traced to the clauses that make it.
Needs the records under LCP_DATA. The cargo side is rust/engine/game/tests/translated_mined.rs."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CARGO = ["cargo", "test", "--release", "--offline", "-j", "4", "--manifest-path", str(ROOT / "rust" / "engine" / "Cargo.toml"), "-p", "cs_game", "--test", "translated_mined", "--", "--ignored", "--nocapture"]


def games_file(n: int) -> Path:
    from catspace.io import paths
    out = paths.experiment("facts_moved_games.txt")
    subprocess.run([sys.executable, str(ROOT / "scripts" / "ply_profile_print.py"), "games", "lichess_2019-01", str(out), str(n)], check=True)
    return Path(out)


def run(test: str, env: dict, keep=("OBS 79.59", "EXPL", "STATE", "MINED", "MASK")) -> int:
    r = subprocess.run(CARGO + [test], env={**os.environ, **env}, capture_output=True, text=True)
    for ln in r.stdout.splitlines():
        if ln.startswith(keep):
            print(ln)
    if r.returncode:
        print(r.stdout[-3000:], r.stderr[-3000:])
    return r.returncode


def main(argv) -> int:
    if not argv:
        print(__doc__)
        return 2
    if argv[0] == "explain":
        n = int(argv[1]) if len(argv) > 1 else 30
        f = games_file(n)
        return run("explain", {"CS_FACTS_GAMES": str(f), "CS_FACTS_N": str(n), "CS_FACTS_MAX": argv[2] if len(argv) > 2 else "40"})
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
