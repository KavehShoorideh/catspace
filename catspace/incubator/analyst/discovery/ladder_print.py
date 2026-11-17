"""PLAN 78.68 (first run), SYSTEM_SPEC 9: THE SURPRISES AS READABLE CASES. The only place of the ladder's files where the game's renderer and names appear (for the reader of the report, never read by the engine).

INTERFACE. cases(tag, n) takes the run's results, answers and groups under artifacts/experiments/ladder_78_68/<tag>/ and writes cases.txt: n surprises, spread over the groups largest first, each with the board, the
move our tools chose, the moves the oracle accepts (the answers file, read here for the human reader only), what our deepest reading found, and the group. Called by the command line only.

  .venv/bin/python -m catspace.incubator.analyst.discovery.ladder_print [--tag full] [--n 20]
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


def main(argv: "list | None" = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="full")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--deep", action="store_true", help="the cases of plan 78.68 stages G and H (deep_heads_78_68/<tag>/case_rows.json)")
    a = ap.parse_args(argv)
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
