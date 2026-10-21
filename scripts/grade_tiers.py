"""Plan 59 acceptance: TIERS off vs on, under the count fuse and under --seconds N, on fresh samples, each decision graded by Stockfish in THIS process (never one running an agenda).

  .venv/bin/python scripts/grade_tiers.py run --out DIR [--sets rand1_10 ...] [--seconds 10]     # the analyser (pool, all cores) per set and configuration; logs in DIR
  .venv/bin/python scripts/grade_tiers.py grade --out DIR [--seconds 10]                           # RIGHT / WRONG / none counts, the puzzles whose decision changed, with both verdicts
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

SETS = ["rand1_10", "rand2_10", "rand3_10", "rand22_5"] + [f"rand{i}_10" for i in range(4, 9)]
DEC = re.compile(r"^OBS questions asked until the deciding claim (\S+): \d+ \(of \d+ asked\), decision (.*) (\S+)$")
TIE = re.compile(r"^OBS tiers (\S+): (.*)$")


def configs(seconds):
    cfg = [("fuse", "off", []), ("fuse", "on", ["--tiers"])]
    if seconds:
        cfg += [(f"s{seconds:g}", "off", ["--seconds", str(seconds)]), (f"s{seconds:g}", "on", ["--tiers", "--seconds", str(seconds)])]
    return cfg


def run(a) -> None:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for s in a.sets:
        for regime, mode, extra in configs(a.seconds):
            f = out / f"{s}.{regime}.{mode}.log"
            if f.exists() and not a.force:
                continue
            r = subprocess.run([sys.executable, "-m", "catspace.incubator.analyst.discovery.analyse", "--set", s, "--quiet", *extra], capture_output=True, text=True)
            f.write_text(r.stdout + ("\nSTDERR\n" + r.stderr[-3000:] if r.returncode else ""))
            print(f"OBS ran {s} {regime} {mode}: " + next((l for l in r.stdout.splitlines() if l.startswith("OBS set wall time")), "no wall line"), flush=True)


def parse(f: Path) -> dict:
    dec, tie = {}, {}
    for l in f.read_text().splitlines():
        m = DEC.match(l)
        if m:
            dec[m.group(1)] = (m.group(2), None if m.group(3) == "None" else m.group(3))
        m = TIE.match(l)
        if m:
            tie[m.group(1)] = m.group(2)
    return dict(dec=dec, tie=tie)


def grade(a) -> None:
    from catspace.incubator.analyst.discovery import puzzle_apply as PA
    from catspace.incubator.analyst.discovery import verdict as VD
    out = Path(a.out)
    cache: dict = {}

    def vd(fen, uci):
        if uci is None:
            return "none"
        if (fen, uci) not in cache:
            v = VD.verdict(fen, uci)
            cache[(fen, uci)] = f"{v[0]}" + (f" (loss {v[1]} cp)" if v[0] == "WRONG" else "")
        return cache[(fen, uci)]
    fens = {p["id"]: p["fen"] for s in SETS for p in PA.load_set(s)}
    for regime, _, _ in configs(a.seconds)[::2]:
        tot = {"off": {}, "on": {}}
        changed = []
        for s in SETS:
            offf, onf = out / f"{s}.{regime}.off.log", out / f"{s}.{regime}.on.log"
            if not (offf.exists() and onf.exists()):
                continue
            off, on = parse(offf), parse(onf)
            for pid in off["dec"]:
                vo, vn = vd(fens[pid], off["dec"][pid][1]), vd(fens[pid], on["dec"].get(pid, ("", None))[1])
                tot["off"][vo.split()[0]] = tot["off"].get(vo.split()[0], 0) + 1
                tot["on"][vn.split()[0]] = tot["on"].get(vn.split()[0], 0) + 1
                print(f"OBS puzzle {regime} {pid} ({s}): off {off['dec'][pid][0]} {vo}; on {on['dec'].get(pid, ('none',))[0]} {vn}; {on['tie'].get(pid, '')}")
                if off["dec"][pid][1] != on["dec"].get(pid, ("", None))[1]:
                    changed.append((pid, s, off["dec"][pid][0], vo, on["dec"].get(pid, ("none",))[0], vn))
        print(f"OBS graded {regime}: TIERS off {dict(sorted(tot['off'].items()))}, TIERS on {dict(sorted(tot['on'].items()))}")
        for c in changed:
            print(f"OBS changed {regime} {c[0]} ({c[1]}): off {c[2]} [{c[3]}] -> on {c[4]} [{c[5]}]")
    VD.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "grade"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--sets", nargs="*", default=SETS)
    ap.add_argument("--seconds", type=float, default=None)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    run(a) if a.cmd == "run" else grade(a)


if __name__ == "__main__":
    main()
