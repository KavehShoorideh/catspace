"""PLAN 78.87 THE VERDICT'S TWO FAULTS. SYSTEM_SPEC 5 (the tactical reasoner: its line verdict) and 9 (the ladder as the diagnosis). A re-measurement with existing readers; game-free (it names no object of any game and prints no
oracle number). The Rust change it measures: lines.rs `settle_tail` is a reading by the judge (the end of the game is counted as the search counts it; the side to move stands when every taking leaves it worse) and
`LineRead.tail_end`; threat_concrete.verdict_of gives the word QUIET to a line with nothing taken, no kind change and net zero.

INTERFACE (what each stage takes, what it returns, who calls it; written before the bodies).
  read --label before|after    reads, with the extension on the path, the deepest line of (A) the bad and the good move of each of 78.84's flagged decisions (symbols_play.LineCut: the settings of the cut; the stored
                               decisions of symbols_play_78_84/full, read only) and (B) the 164 positions of 78.86 that stood at rung 3 or open (ladder_78_86/selfplay_78_58_on, read only: the line that follows the
                               move the search chose at the last clock, at that clock's depth and fuse, as ladder_measure.Tools.line reads it) with the verdict as verdict_of gives it. Writes
                               verdict_87/<label>_a.jsonl and <label>_b.jsonl (a verdict is data: word, net, net_search, end, tail_end, unsettled, kind_net, the units left, the line).
  compare                      takes both labels and prints OBS lines: the from-to table of the verdict word, how often net and the search's own value differ in sign, how many tails reach the end of the game,
                               how many lines are UNSETTLED; writes picked.json (the ten positions of the fixed rule below).
  pick(rows)                   THE RULE (fixed before any position was looked at): the lines whose word changed, in file order (set A in the order of the key, then set B in the order of the positions file); the first three
                               whose new word is QUIET, then the first three whose line differs in its moves and whose new word is not QUIET (the tail kind), then the first of the rest, until ten; shown in file order.
  Called by the command line only (a spawn Pool from this file):
    PYTHONHASHSEED=0 python -m catspace.incubator.analyst.discovery.verdict_measure read --label before    (main's installed extension)
    PYTHONHASHSEED=0 PYTHONPATH=<scratch extension>:<worktree> python -m ...verdict_measure read --label after
    python -m catspace.incubator.analyst.discovery.verdict_measure compare
"""
from __future__ import annotations

from catspace.incubator.analyst.discovery import tiebreak_measure as TM          # first: it sets one thread a process before numpy is loaded

import argparse
import json
import multiprocessing as mp
import time
from collections import Counter
from pathlib import Path

from catspace.incubator.analyst.discovery import threat_concrete as TC
from catspace.io import paths

OUT = ("verdict_78_87",)                                                   # the output constant, set before any run
SOURCE_A = ("symbols_play_78_84", "full")                                 # read only
SOURCE_B = ("ladder_78_86", "selfplay_78_58_on")                          # read only
RECORD_B = "selfplay_78_58"                                               # the record of the ladder's run
KEEP = ("verdict", "net", "net_search", "end", "tail_end", "unsettled", "kind_net", "left_lost", "left_won", "taker_lost", "taker_won", "pv_disagrees", "depth", "resolution", "full_names")


def out_root() -> Path:
    return Path(paths.experiment(*OUT))


def compact(v: "dict | None") -> "dict | None":
    return None if v is None else {k: v[k] for k in KEEP}


# ------------------------------------------------------------------------------------------------------------------------------ the reads

def flagged_rows() -> list:
    from catspace.incubator.analyst.discovery import debts_measure as DM
    return DM.flagged_rows()


def chunk_a(chunk: list) -> list:
    from catspace.incubator.analyst.discovery import symbols_play as SP
    cut = SP.cutter()
    out = []
    for d in chunk:
        tied = [d["bad"], d["good"]]
        v = cut.reads(d["moves"], tied)
        out.append({"key": d["key"], "band": d["band"], "moves": d["moves"], "bad": d["bad"], "good": d["good"], "lines": {m: compact(v[m]) for m in tied}})
    return out


def rows_b() -> list:
    from catspace.incubator.analyst.discovery import ladder_reread as LR
    from catspace.incubator.analyst.discovery import ladder as LD
    root = Path(paths.experiment(*SOURCE_B))
    res = LR.load(root)
    out = []
    for pid, d in res.items():
        r = d["result"]
        if r["rung"] not in (3, 0) or not r.get("deep"):
            continue
        last = [k for k in r["chose"] if k.startswith("3 search")][-1]
        clock = float(last.split()[2])
        i = list(LD.LIMITS["ladder"]).index(clock) if clock in LD.LIMITS["ladder"] else 0
        out.append({"id": pid, "moves": d["moves"], "last": r["chose"][last]["move"], "scale": i})
    return out


def chunk_b(chunk: list) -> list:
    rec = TC.Record.load(RECORD_B)
    lines = TC.Lines(rec, threads=1)
    out = []
    for d in chunk:
        i = d["scale"]
        r = lines.ask([(None, list(d["moves"]) + [d["last"]])], [(0, 9, 0, 1, 0, 0, 0)], TC.DEPTH + 2 * i + 1, TC.FUSE * 4 ** i)[0]
        rd = (r.get("reads") or [None])[0] if r.get("ok") else None
        out.append({"id": d["id"], "moves": d["moves"], "last": d["last"], "scale": i, "line": compact(TC.verdict_of(dict(rd, kind_change=True), rec)) if rd else None})
    return out


def run_reads(label: str, workers: int, say) -> None:
    root = out_root()
    root.mkdir(parents=True, exist_ok=True)
    ctx = mp.get_context("spawn")
    for name, rows, fn, size in (("a", flagged_rows(), chunk_a, 8), ("b", rows_b(), chunk_b, 4)):
        chunks = [rows[i:i + size] for i in range(0, len(rows), size)]
        t0 = time.perf_counter()
        got = []
        with ctx.Pool(workers) as pool:
            for res in pool.imap_unordered(fn, chunks):
                got.extend(res)
        key = "key" if name == "a" else "id"
        got.sort(key=lambda r: r[key])
        (root / f"{label}_{name}.jsonl").write_text("\n".join(json.dumps(r) for r in got) + "\n")
        say(f"READ {label} set {name.upper()}: {len(got)} {'flagged decisions' if name == 'a' else 'positions'} in {time.perf_counter() - t0:.0f} s, {workers} workers")


# ------------------------------------------------------------------------------------------------------------------------------ the comparison

def load(label: str, name: str) -> list:
    return [json.loads(x) for x in (out_root() / f"{label}_{name}.jsonl").read_text().splitlines() if x.strip()]


def sign(x: int) -> int:
    return (x > 0) - (x < 0)


def lines_of(rows: list, which: str) -> list:
    """(id, line) pairs of a set: set A the bad move's line (or the good move's, `which` "good"), set B the position's line; None lines are left out."""
    out = []
    for r in rows:
        if "lines" in r:
            ln = r["lines"][r[which]]
        else:
            ln = r["line"]
        if ln is not None:
            out.append((r.get("key", r.get("id")), ln))
    return out


def stats(b: list, a: list) -> dict:
    """The counts of a set from the lines before and after (same order, same ids)."""
    tab = Counter((x["verdict"], y["verdict"]) for (_i, x), (_j, y) in zip(b, a))
    def opp(ls: list) -> int:                                                  # the verdict's net and the search's own value have opposite signs (both not zero)
        return sum(1 for _i, x in ls if sign(x["net"]) * sign(x["net_search"]) < 0)
    def diff(ls: list) -> int:                                                 # the signs differ (zero against not zero counts)
        return sum(1 for _i, x in ls if sign(x["net"]) != sign(x["net_search"]))
    return {"n": len(a), "table": tab, "changed": sum(v for (f, t), v in tab.items() if f != t),
            "opp": (opp(b), opp(a)), "diff": (diff(b), diff(a)),
            "tail_end": (sum(1 for _i, x in b if x["tail_end"]), sum(1 for _i, x in a if x["tail_end"])),
            "tail_end_split": (Counter(x["tail_end"] for _i, x in b if x["tail_end"]), Counter(x["tail_end"] for _i, x in a if x["tail_end"])),
            "unsettled": (sum(1 for _i, x in b if x["verdict"] == "UNSETTLED"), sum(1 for _i, x in a if x["verdict"] == "UNSETTLED")),
            "words": (Counter(x["verdict"] for _i, x in b), Counter(x["verdict"] for _i, x in a)),
            "moves_differ": sum(1 for (_i, x), (_j, y) in zip(b, a) if x["full_names"] != y["full_names"])}


def pick(rows: list) -> list:
    """THE RULE (see the module doc). `rows`: dicts {"set", "id", "before", "after"} in file order, only those whose word changed. Returns the ten (fewer if fewer exist) in file order."""
    chosen: list = []
    def take(pred, k: int) -> None:
        n = 0
        for r in rows:
            if n >= k:
                break
            if r not in chosen and pred(r):
                chosen.append(r)
                n += 1
    take(lambda r: r["after"]["verdict"] == "QUIET", 3)
    take(lambda r: r["after"]["verdict"] != "QUIET" and r["before"]["full_names"] != r["after"]["full_names"], 3)
    take(lambda r: True, 10 - len(chosen))
    return [r for r in rows if r in chosen]


def compare(say) -> None:
    changed = []
    for name, which in (("a", "bad"), ("a", "good"), ("b", "bad")):
        rb, ra = load("before", name), load("after", name)
        for r in rb:                                                            # (the "before" reads are main's extension through this file's verdict_of, so the word QUIET is already in them: before 78.87 that line was REFUTED, by construction of the one branch it is cut from)
            for ln in ([r["line"]] if "line" in r else list(r["lines"].values())):
                if ln is not None and ln["verdict"] == "QUIET":
                    ln["verdict"] = "REFUTED"
        assert [r.get("key", r.get("id")) for r in rb] == [r.get("key", r.get("id")) for r in ra]
        b, a = lines_of(rb, which), lines_of(ra, which)
        ids_b, ids_a = [i for i, _ in b], [i for i, _ in a]
        keep = [i for i in ids_b if i in set(ids_a)]
        b, a = [(i, x) for i, x in b if i in set(keep)], [(i, x) for i, x in a if i in set(keep)]
        s = stats(b, a)
        label = {"a": f"set A ({which} move of 78.84's flagged decisions)", "b": "set B (the 78.86 positions at rung 3 or open)"}[name]
        say(f"{label}: {s['n']} lines read")
        say(f"  WORD CHANGED {s['changed']} of {s['n']}; FROM-TO " + ", ".join(f"{f}->{t} {v}" for (f, t), v in sorted(s["table"].items()) if f != t))
        say(f"  WORDS before {dict(sorted(s['words'][0].items()))}; after {dict(sorted(s['words'][1].items()))}")
        say(f"  NET AND THE SEARCH'S OWN VALUE of opposite signs: before {s['opp'][0]}, after {s['opp'][1]}; the signs differ at all (zero against not zero too): before {s['diff'][0]}, after {s['diff'][1]}")
        say(f"  TAILS THAT REACH THE END OF THE GAME (won 1, lost 2): before {s['tail_end'][0]} {dict(s['tail_end_split'][0])}, after {s['tail_end'][1]} {dict(s['tail_end_split'][1])}; UNSETTLED before {s['unsettled'][0]}, after {s['unsettled'][1]}; lines whose moves differ {s['moves_differ']}")
        if which == "bad":
            for (i, x), (_j, y) in zip(b, a):
                if x["verdict"] != y["verdict"]:
                    changed.append({"set": name.upper(), "id": i, "before": x, "after": y})
    picked = pick(changed)
    say(f"PICKED {len(picked)} of {len(changed)} lines whose word changed (rule in pick): " + ", ".join(f"{r['set']}:{r['id']} {r['before']['verdict']}->{r['after']['verdict']}" for r in picked))
    (out_root() / "picked.json").write_text(json.dumps(picked))


def main(argv: "list | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stage", choices=("read", "compare"))
    ap.add_argument("--label", default="after")
    ap.add_argument("--workers", type=int, default=9)
    a = ap.parse_args(argv)
    root = out_root()
    root.mkdir(parents=True, exist_ok=True)

    def say(line: str) -> None:
        TM.emit([f"OBS verdict {line}"], root / "obs.txt")
    if a.stage == "read":
        import cs_engine as CE
        load_avg, busy = TM.wait_quiet(lambda s: TM.emit([s], root / "obs.txt"))
        say(f"READ {a.label} start: load average {load_avg:.2f}, cores busy {busy:.2f}; extension {CE.__file__}")
        run_reads(a.label, a.workers, say)
    else:
        compare(say)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
