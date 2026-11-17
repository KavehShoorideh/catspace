"""PLAN 78.68 (first run), SYSTEM_SPEC 9: THE SURPRISES AS READABLE CASES. The only place of the ladder's files where the game's renderer and names appear (for the reader of the report, never read by the engine).

INTERFACE. cases(tag, n) takes the run's results, answers and groups under artifacts/experiments/ladder_78_68/<tag>/ and writes cases.txt: n surprises, spread over the groups largest first, each with the board, the
move our tools chose, the moves the oracle accepts (the answers file, read here for the human reader only), what our deepest reading found, and the group. Called by the command line only.

  .venv/bin/python -m catspace.incubator.analyst.discovery.ladder_print [--tag full] [--n 20]

PLAN 78.76 `--records NAME NAME ...` (`records_cases`): the same positions side by side under each measured record of the kinds. THE SET IS FIXED BY A RULE BEFORE ANY RESULT IS READ: the first six open surprises of the 78.68 list
(surprise_list.jsonl of the full run, in file order) and the first six positions, in the order of positions.jsonl, whose rung under some record differs from the first record's. Under each record: the moves the search holds
equal, its first move, the best line and where it ends (each read at the clock that record's ladder stopped at: the wake's for rungs 1 and 2, the clock it was found at for rung 3, the last for an open one), and the accepted
moves (the answers file, for the human reader only). Writes ladder_78_76/cases.txt.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import chess

from catspace.incubator.analyst.discovery import ladder as LD
from catspace.incubator.analyst.discovery import ladder_measure as LM
from catspace.io import paths


def san_line(b: chess.Board, names: list, cap: int = 8) -> str:
    out, x = [], b.copy()
    for m in names[:cap]:
        try:
            mv = chess.Move.from_uci(m)
            out.append(x.san(mv))
            x.push(mv)
        except Exception:
            out.append(m + "?")
            break
    return " ".join(out)


def our_move(res: dict) -> "str | None":
    ms = [v["move"] for k, v in res["chose"].items() if k.startswith("3 search")]
    return ms[-1] if ms else res["chose"].get("1 search", {}).get("move")


def pick(rows: list, groups: list, n: int) -> list:
    """Round robin over the groups, the largest first, one surprise a turn."""
    queues = [list(g["ids"]) for g in groups]
    out = []
    while len(out) < n and any(queues):
        for q in queues:
            if q and len(out) < n:
                out.append(q.pop(0))
    return out


def deep_cases(tag: str = "full", n: int = 10) -> Path:
    """PLAN 78.68 stage G and H: THE CASES OF THE BEST CANDIDATE as readable text (the board, the tied moves, the move the candidate picks and the line it read after it to its settled end, the moves the oracle accepts, for
    the reader only): `n` open surprises it now picks an accepted move in, and `n` positions where it creates a surprise. Reads deep_heads_78_68/<tag>/case_rows.json (which holds no oracle field) and the ladder's answers
    file here, for the human reader. Writes cases.txt beside it."""
    root = Path(paths.experiment("deep_heads_78_68", tag))
    cr = json.loads((root / "case_rows.json").read_text())
    ans = {d["id"]: d["accepted"] for d in (json.loads(x) for x in (LM.out_root("full") / "answers.jsonl").read_text().splitlines() if x.strip())}
    lines = [f"THE BEST CANDIDATE: {cr['best']}", ""]
    for title, key in (("OPEN SURPRISES THE CANDIDATE NOW PICKS AN ACCEPTED MOVE IN", "resolved"), ("POSITIONS WHERE THE CANDIDATE CREATES A SURPRISE", "created")):
        lines += [f"##### {title} ({min(n, len(cr[key]))} of {len(cr[key])})", ""]
        for k, c in enumerate(cr[key][:n], 1):
            b = chess.Board()
            for m in c["moves"]:
                b.push_uci(m)

            def nm(m: str) -> str:
                return b.san(chess.Move.from_uci(m))
            lines += [f"=== {key.upper()} {k}: {c['id']}  ({'white' if b.turn else 'black'} to move)", str(b), "",
                      f"the search held equal: {', '.join(nm(m) for m in c['tied'])}   (its own first move: {nm(c['search_move'])})",
                      f"the candidate picks: {nm(c['pick'])}; its best line after it, to the settled end: {san_line(b, c['line'] or [], 10)}; the line's end: {c['leaf_end']}",
                      f"the oracle accepts: {', '.join(nm(m) for m in ans[c['id']])}"]
            for m, (ln, end) in c["lines"].items():
                if m != c["pick"]:
                    lines.append(f"   the line after {nm(m)}: {san_line(b, ln or [], 10)} (end {end})")
            lines.append("")
    (root / "cases.txt").write_text("\n".join(lines))
    print(f"OBS ladder print: deep cases written to {root / 'cases.txt'}")
    return root / "cases.txt"


_CASE_TOOLS: dict = {}


def _case_job(job: tuple) -> dict:
    """One (record, position): the readers of our own reasoning at the clock the record's ladder stopped at."""
    name, pid, moves, clock = job
    if name not in _CASE_TOOLS:
        rec, kc = LM.split_run(name)
        _CASE_TOOLS[name] = LM.Tools(rec, kind_change=kc)
    t = _CASE_TOOLS[name]
    s = t.search({"moves": moves}, clock)
    ln = t.line({"moves": moves}, clock)
    return {"name": name, "id": pid, "clock": clock, "search": s, "line": ln}


def stop_clock(res: dict) -> float:
    r = res["result"]
    if r["found_at"]:
        return r["found_at"]
    if r["rung"] in (1, 2):
        return LD.LIMITS["wake_seconds"]
    return LD.LIMITS["ladder"][-1]


def fixed_cases(names: list, per: int = 6) -> "tuple[list, list]":
    """The two sets of the rule: the first `per` open surprises of the 78.68 list in file order; the first `per` positions of positions.jsonl whose rung under some record differs from the first record's."""
    from catspace.incubator.analyst.discovery import ladder_compare as LC
    src = LM.out_root("full")
    lst = [json.loads(x)["id"] for x in (src / "surprise_list.jsonl").read_text().splitlines() if x.strip()]
    runs = {n: LC.load_results(n) for n in names}
    order = [json.loads(x)["id"] for x in (src / "positions.jsonl").read_text().splitlines() if x.strip()]
    changed = [i for i in order if all(i in runs[n] for n in names) and len({runs[n][i]["result"]["rung"] for n in names}) > 1]
    return lst[:per], changed[:per]


def ids_of_cases_file(path: "str | Path") -> list:
    """The ids of the positions of an earlier cases file (the line `=== SET A CASE 3: <id>  (...)`), in file order, each once."""
    out = []
    for ln in Path(path).read_text().splitlines():
        if ln.startswith("=== SET "):
            pid = ln.split(": ", 1)[1].split("  (")[0].strip()
            if pid not in out:
                out.append(pid)
    return out


def records_cases(names: list, per: int = 6, workers: int = 10, fixed_from: "str | None" = None) -> Path:
    """`fixed_from` (PLAN 78.77): a cases file of an earlier plan whose positions are printed again (as set A), then the first `per` positions whose rung differs between the runs `names` (set B)."""
    import multiprocessing
    from catspace.incubator.analyst.discovery import ladder_compare as LC
    first, changed = fixed_cases(names, 400 if fixed_from else per)
    if fixed_from:
        old = ids_of_cases_file(fixed_from)
        first = old
        changed = [i for i in changed if i not in old][:per]               # (the first `per` positions whose rung changes that the earlier file did not already print)
    runs = {n: LC.load_results(n) for n in names}
    ans = {d["id"]: d["accepted"] for d in (json.loads(x) for x in (LM.out_root("full") / "answers.jsonl").read_text().splitlines() if x.strip())}
    pos = {d["id"]: d for d in (json.loads(x) for x in (LM.out_root("full") / "positions.jsonl").read_text().splitlines() if x.strip())}
    ids = list(dict.fromkeys(first + changed))
    jobs = [(n, i, pos[i]["moves"], stop_clock(runs[n][i])) for i in ids for n in names if i in runs[n]]
    jobs.sort(key=lambda j: -j[3])
    with multiprocessing.get_context("spawn").Pool(workers) as pool:
        got = {(r["name"], r["id"]): r for r in pool.imap_unordered(_case_job, jobs, chunksize=1)}
    lines = [f"RECORDS {names}; set A: the first {per} open surprises of the 78.68 list in file order; set B: the first {per} positions whose rung differs between the records", ""]
    for title, sel in (("A", first), ("B", changed)):
        for k, pid in enumerate(sel, 1):
            b = chess.Board()
            for m in pos[pid]["moves"]:
                b.push_uci(m)

            def nm(m: str) -> str:
                return b.san(chess.Move.from_uci(m))
            lines += [f"=== SET {title} CASE {k}: {pid}  ({'white' if b.turn else 'black'} to move)", str(b), "", f"the accepted moves: {', '.join(nm(m) for m in ans[pid])}"]
            for n in names:
                if (n, pid) not in got:
                    continue
                g, r = got[(n, pid)], runs[n][pid]["result"]
                s, ln = g["search"], g["line"]
                lines += [f"  --- record {n}: rung {r['rung']}" + (f" (found at {r['found_at']:g} s)" if r["found_at"] else "") + f"; read at {g['clock']:g} s",
                          f"      held equal ({len(s['equals'])}): {', '.join(nm(m) for m in sorted(s['equals']))}",
                          f"      the search's first move: {nm(s['move'])} ({'accepted' if s['move'] in ans[pid] else 'not accepted'})",
                          f"      its best line (after the search's own move): {san_line(b, ln['names'], 10)}; events {ln['events']}; kind changes {ln.get('changes')}; ended {ln['end']} after {ln['depth']} plies, cut {ln['cut']}"]
            lines.append("")
    out = Path(paths.experiment("ladder_78_77" if fixed_from else "ladder_78_76", "cases.txt"))
    out.write_text("\n".join(lines))
    print(f"OBS ladder print: {len(first)} + {len(changed)} cases under records {names} written to {out}")
    return out


def main(argv: "list | None" = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", nargs="+", default=None, help="PLAN 78.76: the fixed cases under each named record of the kinds (the first is the control)")
    ap.add_argument("--fixed-from", default=None, help="PLAN 78.77: print the positions of this earlier cases file again, then the first six whose rung differs between the named runs (RECORD@on, RECORD@off)")
    ap.add_argument("--tag", default="full")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--deep", action="store_true", help="the cases of plan 78.68 stages G and H (deep_heads_78_68/<tag>/case_rows.json)")
    a = ap.parse_args(argv)
    if a.records:
        records_cases(a.records, fixed_from=a.fixed_from)
        return 0
    if a.deep:
        deep_cases(a.tag, a.n)
        return 0
    root = LM.out_root(a.tag)
    res = {d["id"]: d for d in (json.loads(x) for x in (root / "results.jsonl").read_text().splitlines() if x.strip())}
    ans = {d["id"]: d["accepted"] for d in (json.loads(x) for x in (root / "answers.jsonl").read_text().splitlines() if x.strip())}
    groups = json.loads((root / "groups.json").read_text())
    af = json.loads((root / "answer_features.json").read_text()) if (root / "answer_features.json").exists() else {}
    where = {i: g for g in groups for i in g["ids"]}
    lines = []
    for k, pid in enumerate(pick(list(res), groups, a.n), 1):
        d = res[pid]
        r = d["result"]
        b = chess.Board()
        for m in d["moves"]:
            b.push_uci(m)
        mine = our_move(r)
        g = where[pid]
        lines += [f"=== CASE {k}: {pid}  ({d['source']}, {d['units']} units, band {d['band']}, {'white' if b.turn else 'black'} to move)",
                  str(b), "",
                  f"our move at the last clock: {b.san(chess.Move.from_uci(mine)) if mine else None}   rung {r['rung']}" + (f", found at {r['found_at']:g} s" if r["found_at"] else ", not found"),
                  f"the oracle accepts: {', '.join(b.san(chess.Move.from_uci(m)) for m in ans[pid])}" + (f"   (types: {af[pid]})" if pid in af else ""),
                  f"the wake's search held {r['equals']} moves equal; the heads: {r['chose'].get('2 heads')}",
                  f"our deepest reading: {san_line(b, (r['deep'] or {}).get('names', []))}; events {(r['deep'] or {}).get('events')}; ended {(r['deep'] or {}).get('end')} after {(r['deep'] or {}).get('depth')} plies, cut {(r['deep'] or {}).get('cut')}",
                  f"features: {r['features']}",
                  f"group: size {g['size']}, key {g['key']}", ""]
    (root / "cases.txt").write_text("\n".join(lines))
    print(f"OBS ladder print: {min(a.n, len(res))} cases written to {root / 'cases.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
