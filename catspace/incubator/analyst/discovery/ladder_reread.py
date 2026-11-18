"""PLAN 78.86, SYSTEM_SPEC 9 (the ladder as the diagnosis) and 5 (the line verdict): THE LADDER READ AGAIN WITH THE REPAIRED VERDICT. A re-measurement with existing code; game-free (it names no object of any game and
prints no oracle number); the answers file is read only by the ladder's own scoring (ladder.ladder, deep_heads.score) and is never handed to anything that reasons.

INTERFACE (what each part takes, what it returns, who calls it).
  run(tag, limit, workers)        the ladder of 78.68 on the same 400 positions, the same record (the self-play record of 78.58) and the same settings as the last run of 78.77 (the event of a unit changing kind IN the
                                  judge, 9 workers, one thread a worker, the equal reading's second step at its default, off), through ladder_measure.main; outputs to ladder_78_86/selfplay_78_58_on/[tag/].
                                  Since 78.85 the line read settles its tail, and the deep line of a row also carries the repaired verdict (ladder_measure.Tools.verdict, ladder.DEEP_FIELDS).
  rung_table(old, new)            {(from, to): n} of the positions whose rung differs between two runs (ladder_compare.rung_changes).
  division(results)               (rung 1, rung 2, rung 3, open) counts of a run.
  verdict_counts(results)         how many deep lines carry the verdict UNSETTLED, and how many positions have a verdict whose net changed because a unit changing kind is counted (kind_net not 0), and how many of
                                  those change the verdict word.
  groups_of(path)                 the groups of a run's open list as (size, key) in the file's order.
  pick_changed(order, old, new, k) the twenty positions of the printed cases, by a rule fixed before any was looked at (see its doc).
  pick_line_changed(order, old, new, taken, k)  the rest of the twenty: positions that kept their rung and whose deepest line differs, in file order.
  compare(tag)                  the OBS lines beside the earlier run (ladder_78_77/selfplay_78_58_on): rungs, open, groups, the from-to table, the verdict counts.
  score(workers)                  the hint's score on the scored part once more, picks as stored (78.80: symbols_78_80/full; 78.81: symbols_78_81/full; nothing refitted): resolved less created with its band and the
                                  hit rate against the search's first move, on the open list of the earlier run of this self-play record (78.77) and on the new one, beside the lines of those plans' OBS files.
                                  symbols_measure.score_ladder and symbols_reasons_measure.score_ladder_reasons take the run's folder (`results_root`); they are the functions that read the answers.
  Called by the command line only:
    python -m catspace.incubator.analyst.discovery.ladder_reread run [--tag full] [--limit N] [--workers 9]
    python -m catspace.incubator.analyst.discovery.ladder_reread compare [--tag full]
    python -m catspace.incubator.analyst.discovery.ladder_reread score [--workers 9]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from catspace.incubator.analyst.discovery import ladder_compare as LC
from catspace.incubator.analyst.discovery import ladder_measure as LM
from catspace.io import paths

OUT = ("ladder_78_86",)
LM.OUT_EVENT = OUT                                                        # the output folder is set before any run: ladder_78_86/<record>_on/ (a short run: <tag>/ below it)
RUN = "selfplay_78_58@on"                                                 # the last run of 78.77: the self-play record of 78.58, the event on
EARLIER = ("ladder_78_77", "selfplay_78_58_on")                           # that run's folder (read only)
RUNGS = (1, 2, 3, 0)


def run(tag: str = "full", limit: int = 0, workers: int = 9) -> int:
    argv = ["run", "--record", RUN, "--tag", tag, "--workers", str(workers)] + (["--limit", str(limit)] if limit else [])
    return LM.main(argv)


def load(root: Path) -> dict:
    return {d["id"]: d for d in (json.loads(x) for x in (root / "results.jsonl").read_text().splitlines() if x.strip()) if d["result"]}


def division(results: dict) -> tuple:
    c = Counter(d["result"]["rung"] for d in results.values())
    return tuple(c[r] for r in RUNGS)


def rung_table(old: dict, new: dict) -> dict:
    return LC.rung_changes(old, new)


def verdict_counts(results: dict) -> dict:
    vs = [(d["result"].get("deep") or {}).get("verdict") for d in results.values()]
    have = [v for v in vs if v]
    return {"lines": len(have), "unsettled": sum(1 for v in have if v["unsettled"] or v["verdict"] == "UNSETTLED"), "kind_net": sum(1 for v in have if v["kind_net"] != 0),
            "kind_net_changes_word": sum(1 for v in have if v["kind_net"] != 0 and v["verdict"] != v["without_kind"]),
            "words": dict(Counter(v["verdict"] for v in have))}


def groups_of(path: Path) -> list:
    g = json.loads(Path(path).read_text())
    rows = g if isinstance(g, list) else g.get("groups", g)
    out = []
    for r in rows:
        if isinstance(r, dict):
            out.append((r.get("size", r.get("n")), tuple(r.get("key", ()))))
        else:
            out.append((r[0], tuple(r[1])))
    return out


def pick_changed(order: list, old: dict, new: dict, k: int = 20) -> list:
    """THE RULE FOR THE PRINTED CASES (fixed before any position was looked at): the positions whose rung differs between the runs are grouped by the direction of the change (from rung, to rung); the directions are taken
    in order of how many positions they hold, the larger first (ties: the smaller from, then to, in the order 1, 2, 3, open); in each direction the positions are in the order of the positions file (`order`); one position is
    taken from each direction in turn, round and round, until `k` are taken or none is left. Returns the ids in the order taken."""
    rank = {r: j for j, r in enumerate(RUNGS)}
    by = {}
    for i in order:
        if i in old and i in new and old[i]["result"]["rung"] != new[i]["result"]["rung"]:
            by.setdefault((old[i]["result"]["rung"], new[i]["result"]["rung"]), []).append(i)
    dirs = sorted(by, key=lambda d: (-len(by[d]), rank[d[0]], rank[d[1]]))
    out = []
    while len(out) < k and any(by[d] for d in dirs):
        for d in dirs:
            if by[d] and len(out) < k:
                out.append(by[d].pop(0))
    return out


def pick_line_changed(order: list, old: dict, new: dict, taken: list, k: int) -> list:
    """THE SECOND PART OF THE RULE (fixed before any position was looked at, used because fewer than twenty positions change rung): after the positions that changed rung, the positions that kept their rung and whose deepest
    line (its moves) differs between the runs, in the order of the positions file, until `k` positions are chosen in all."""
    out = list(taken)
    for i in order:
        if len(out) >= k:
            break
        if i in old and i in new and i not in out and (old[i]["result"].get("deep") or {}).get("names") != (new[i]["result"].get("deep") or {}).get("names"):
            out.append(i)
    return out


def compare(tag: str = "full") -> int:
    old_root = Path(paths.experiment(*EARLIER))
    new_root = Path(paths.experiment(*OUT, "selfplay_78_58_on")) / ("" if tag == "full" else tag)
    out = Path(paths.experiment(*OUT)) / ("" if tag == "full" else tag)
    out.mkdir(parents=True, exist_ok=True)

    def say(m: str) -> None:
        line = f"OBS ladder reread: {m}"
        print(line, flush=True)
        with open(out / "compare_obs.txt", "a") as f:
            f.write(line + "\n")
    old, new = load(old_root), load(new_root)
    ids = [i for i in new if i in old]
    o, n = {i: old[i] for i in ids}, {i: new[i] for i in ids}
    say(f"positions read in both runs: {len(ids)} (earlier run {old_root}, this run {new_root})")
    for name, r in (("earlier", o), ("now", n)):
        d = division(r)
        say(f"DIVISION {name}: rung 1 {d[0]}, rung 2 {d[1]}, rung 3 {d[2]}, open {d[3]}")
    ch = rung_table(o, n)
    say(f"RUNG CHANGES earlier -> now: {sum(ch.values())} of {len(ids)} positions changed rung")
    say("RUNG TABLE (from, to): " + (", ".join(f"({f}->{t}) {c}" for (f, t), c in sorted(ch.items())) or "none"))
    for f in RUNGS:
        row = [sum(1 for i in ids if o[i]["result"]["rung"] == f and n[i]["result"]["rung"] == t) for t in RUNGS]
        say(f"RUNG MATRIX from rung {f}: to 1 {row[0]}, to 2 {row[1]}, to 3 {row[2]}, to open {row[3]}")
    for name, root in (("earlier", old_root), ("now", new_root)):
        gp = root / "groups.json"
        if gp.exists():
            gs = groups_of(gp)
            say(f"OPEN LIST {name}: {sum(s for s, _ in gs)} rows in {len(gs)} groups")
            for s, k in gs:
                say(f"GROUP {name} size {s}: key {list(k)}")
    vc = verdict_counts(n)
    say(f"VERDICT lines read with a verdict {vc['lines']} (positions at rung 3 or open); UNSETTLED {vc['unsettled']}; positions whose verdict net changed because a unit changing kind is counted {vc['kind_net']}, "
        f"of them where the verdict word changes {vc['kind_net_changes_word']}; words {vc['words']}")
    say("the earlier run had no verdict in its rows, no UNSETTLED and no kind change in the net by construction (78.77 left it out)")
    order = [json.loads(x)["id"] for x in (LM.out_root("full") / "positions.jsonl").read_text().splitlines() if x.strip()]
    ids_ch = [i for i in order if i in o and i in n and o[i]["result"]["rung"] != n[i]["result"]["rung"]]
    chosen = pick_line_changed(order, o, n, pick_changed(order, o, n, 20), 20)
    n_line = sum(1 for i in order if i in o and i in n and (o[i]["result"].get("deep") or {}).get("names") != (n[i]["result"].get("deep") or {}).get("names"))
    say(f"LINES the deepest line (its moves) differs between the runs in {n_line} of the {sum(1 for d in n.values() if d['result'].get('deep'))} positions that have one")
    (out / "changed_ids.json").write_text(json.dumps(ids_ch))
    (out / "chosen_ids.json").write_text(json.dumps(chosen))
    say(f"CASES the twenty positions chosen by the rule of pick_changed (directions by size, round and round, file order): {len(chosen)} of {len(ids_ch)} changed; directions " + ", ".join(f"({o[i]['result']['rung']}->{n[i]['result']['rung']})" for i in chosen))
    return 0


def score(workers: int = 9) -> int:
    from catspace.incubator.analyst.discovery import symbols_measure as SM
    from catspace.incubator.analyst.discovery import symbols_reasons_measure as SR
    out = Path(paths.experiment(*OUT))
    out.mkdir(parents=True, exist_ok=True)

    def say(m: str) -> None:
        line = f"OBS ladder reread score: {m}"
        print(line, flush=True)
        with open(out / "score_obs.txt", "a") as f:
            f.write(line + "\n")
    r80, r81 = SM.out_root("full"), SR.out_root("full")
    for plan, root in (("78.80", r80), ("78.81", r81)):
        for ln in (root / "obs.txt").read_text().splitlines():
            if "SCORE 2" in ln or "SCORE 3" in ln:
                say(f"EARLIER, the line of {plan}'s obs.txt (the open list of the 78.68 run, generation-2 record): {ln}")
    for name, rr in (("the earlier run of the self-play record (78.77)", Path(paths.experiment(*EARLIER))), ("the new run (78.86)", Path(paths.experiment(*OUT, "selfplay_78_58_on")))):
        say(f"=== open list: {name}")
        SM.score_ladder(r80, workers, lambda m, n=name: say(f"[{n[:12]}] 78.80 picks: {m}"), results_root=rr)
        SR.score_ladder_reasons(r81, workers, lambda m, n=name: say(f"[{n[:12]}] 78.81 picks: {m}"), results_root=rr)
    return 0


def main(argv: "list | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stage", choices=("run", "compare", "score"))
    ap.add_argument("--tag", default="full")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=9)
    a = ap.parse_args(sys.argv[1:] if argv is None else argv)
    return {"run": lambda: run(a.tag, a.limit, a.workers), "compare": lambda: compare(a.tag), "score": lambda: score(a.workers)}[a.stage]()


if __name__ == "__main__":
    raise SystemExit(main())
