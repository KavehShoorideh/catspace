"""PLAN 79.59 (THE LEMMA MINER OF LAYER 2; SYSTEM_SPEC 11, block "The lemma miner of layer 2"; LAYER 2, piece: rules; ledger rows 59, 58, 57): THE PRINTING SCRIPT. It may name the games; it only calls the programs and prints what they say.
  python scripts/lemma_miner_print.py explain [<games> [<plies>]]   PART A and the analysis of B: for each of the seven games, the reads that make a support wider than a line of places (every place of the board, a whole fluent) with the clause and the literal that make them (the program's explanations,
                                                                    scripts/lemma_explain_print.py), and for each the findings that follow mechanically from the clauses and the order they are joined in (`lemma_miner.findings_at`): a read of one row, a read for existence only, a literal whose free variable
                                                                    another literal binds and could come first, a scan.
  python scripts/lemma_miner_print.py statements                    PART B: for each game, the proposals of the null change with the reads of the proof, in the description's own words (no judge: scripts/lemma_miner_run.py judge prints the statements with the counts).
  python scripts/lemma_miner_print.py count [<games> [<plies>]]     PART F, A COUNT ONLY: what it would take for the graph to hold EVERY relation a description defines: the relations of each description by shape (a unit and a place, two places, a unit and two places, one place, the rest), the rows a state of each shape, and what the rest are.
Needs the records under LCP_DATA (the games file of the per-ply measure)."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

from catspace.incubator.analyst.discovery import gdl_to_rust as T
from catspace.incubator.analyst.discovery import lemma_miner as M
from catspace.io import paths


def games_file(n: int) -> Path:
    out = paths.experiment("facts_moved_games.txt")
    subprocess.run([sys.executable, str(ROOT / "scripts" / "ply_profile_print.py"), "games", "lichess_2019-01", str(out), str(n)], check=True)
    return Path(out)


def cargo(test: str, n: int, plies: int) -> str:
    f = games_file(n)
    cmd = ["cargo", "test", "--release", "--offline", "-j", "8", "--manifest-path", str(ROOT / "rust" / "engine" / "Cargo.toml"), "-p", "cs_game", "--test", "translated_mined", "--", "--ignored", "--nocapture", test]
    r = subprocess.run(cmd, env={**os.environ, "CS_FACTS_GAMES": str(f), "CS_FACTS_N": str(n), "CS_FACTS_MAX": str(plies)}, capture_output=True, text=True)
    if r.returncode:
        print(r.stdout[-3000:], r.stderr[-3000:])
        raise SystemExit(r.returncode)
    return r.stdout


def explain(n: int, plies: int) -> None:
    out = cargo("explain", n, plies)
    game = None
    block: list = []

    def flush():
        if not block:
            return
        # the lines of one WIDE read: its OBS line, the CALL frames (outermost first), the READ frame
        head = block[0]
        print(head)
        frames = []
        for ln in block[1:]:
            if ln.startswith("EXPLF"):
                _a, g, kind, rest = ln.split(" ", 3)
                clause, mask, order, pos, work = [x.strip() for x in rest.split("|||")]
                frames.append((kind, clause, mask, [int(x) for x in order.split(",") if x != ""], int(pos), work == "true"))
        for kind, clause, mask, order, pos, work in frames:
            lit = M.parse_clause_text(clause).body[order[pos]]
            where = f"{kind} by clause {clause}, joined in the order {order} with the head places {mask} bound, at its position {pos}: {M.lit_text(lit[0], lit[1], lit[2]) if lit[0] != 'distinct' else lit}"
            print(f"MINED FRAME {game}: {where}")
            if work:
                continue
            for f in M.findings_at(clause, mask, order, pos):
                print(f"MINED FINDING {game}: {kind.lower()} {M.lit_text(lit[0], lit[1], lit[2]) if lit[0] != 'distinct' else lit}: {f}")
        block.clear()

    for ln in out.splitlines():
        if ln.startswith("OBS 79.59 A") and ": WIDE " in ln:
            flush()
            game = ln.split()[3].rstrip(":")
            block.append(ln)
        elif ln.startswith("OBS 79.59 A"):
            flush()
            if "READS THAT MAKE" not in ln:
                print(ln)
        elif ln.startswith("EXPLF") and block:
            block.append(ln)
    flush()


def statements() -> None:
    from catspace.incubator.analyst.discovery import rule_lemma as RL
    import importlib.util
    spec = importlib.util.spec_from_file_location("lemma_miner_run", ROOT / "scripts" / "lemma_miner_run.py")
    run = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run)
    for game, text, pr, why in run.proposals():
        if pr is None:
            print(f"MINED NOTHING {game}: {why[:300]}")
            continue
        for ln in M.statement(pr):
            print(f"MINED STATEMENT {game}: {ln}")


def shape_of(sig: tuple, place: tuple, unit: tuple) -> tuple:
    """-> (number of places, the other sorts): the signature with every adjacent pair of the PLACE product taken as one place."""
    toks, i = [], 0
    while i < len(sig):
        if i + 1 < len(sig) and (sig[i], sig[i + 1]) == place:
            toks.append("P")
            i += 2
        else:
            toks.append(sig[i])
            i += 1
    return toks.count("P"), sorted(x for x in toks if x != "P")


def count(n: int, plies: int) -> None:
    from catspace.incubator.analyst.discovery import rule_lemma  # noqa: F401  (the reader of the descriptions' composition is the translator's)
    import importlib.util
    spec = importlib.util.spec_from_file_location("gdl_to_rust_make", ROOT / "scripts" / "gdl_to_rust_make.py")
    mk = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mk)
    out = cargo("relations_count", n, plies)
    rows: dict = {}
    for ln in out.splitlines():
        if ln.startswith("COUNT "):
            _c, g, r, mean, mx, st = ln.split()
            rows.setdefault(g, {})[r] = (float(mean), int(mx), int(st))
    for game, srcs in mk.GAMES.items():
        text = T.compose_text(srcs) if len(srcs) > 1 else srcs[0].read_text()
        d = T.Description(text)
        plan = T.Plan(d, ("bits",))
        place = max(plan.units, key=lambda k: plan.units[k])            # the product of two sorts the clauses use together most: the description's own PLACE
        fluents = {r: d.prog.signature(r) for r in d.arity if r.startswith("true.") and d.prog.signature(r)}
        # the UNIT: the sorts the fluent that holds a place carries besides it (the fluent with a place and the most other sorts)
        holder = max((r for r, s in fluents.items() if shape_of(s, place, ())[0] == 1), key=lambda r: len(fluents[r]), default=None)
        unit = tuple(shape_of(fluents[holder], place, ())[1]) if holder else ()
        classes: dict = {}
        total_rows = 0.0
        derived = [r for r in sorted(d.arity) if r in rows.get(game, {}) and not r.startswith(("true.", "init.", "next.")) and d.prog.signature(r) is not None]
        for r in derived:
            sig = tuple(d.prog.signature(r))
            np_, other = shape_of(sig, place, unit)
            if np_ == 1 and tuple(other) == unit:
                cls = "a unit and a place"
            elif np_ == 2 and tuple(other) == unit:
                cls = "a unit and two places"
            elif np_ == 2 and len(other) <= 1:
                cls = "two places"
            elif np_ == 1 and len(other) <= 1:
                cls = "one place"
            elif np_ == 0:
                cls = "no place"
            else:
                cls = f"{np_} places and {len(other)} other sorts"
            mean = rows[game][r][0]
            e = classes.setdefault(cls, {"relations": [], "rows": 0.0, "static": 0})
            e["relations"].append(r)
            e["rows"] += mean
            total_rows += mean
        print(f"OBS 79.59 F {game}: {len(derived)} relations defined (not the fluents, their initial and next relations), the place is the product {place} (weight {plan.units[place]}), the unit is {unit} ({holder}); rows a state over all of them {total_rows:.1f}")
        for cls, e in sorted(classes.items(), key=lambda kv: -len(kv[1]["relations"])):
            print(f"OBS 79.59 F {game}: {cls}: {len(e['relations'])} relations, {e['rows']:.1f} rows a state; " + ", ".join(e["relations"][:14]) + (" ..." if len(e["relations"]) > 14 else ""))


def main(argv) -> int:
    if not argv:
        print(__doc__)
        return 2
    n = int(argv[1]) if len(argv) > 1 else 30
    plies = int(argv[2]) if len(argv) > 2 else 40
    if argv[0] == "explain":
        explain(n, plies)
        return 0
    if argv[0] == "statements":
        statements()
        return 0
    if argv[0] == "count":
        count(n, plies)
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
