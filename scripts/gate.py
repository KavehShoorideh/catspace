#!/usr/bin/env python
"""gate.py -- the two test gates: `engine` (every change) and `full` (before a commit that
touches shared code). docs/ANALYST_PLAN.md 72.6 THE GATE SPLIT.

Why: timed engine tests are flaky under CPU contention; on 2026-10-22 three stacked pytest
runs made a gate take two hours. So: the engine gate is a short fixed list of files, run
ONE FILE AT A TIME (each its own unbuffered pytest subprocess, durations logged, output
appended to a log under artifacts/experiments/), under a LOCK so a second gate refuses to
start while one runs. Other running pytest processes are listed as a warning; nothing is
ever killed.

    .venv/bin/python scripts/gate.py engine
    .venv/bin/python scripts/gate.py full
    .venv/bin/python scripts/gate.py engine --dry-run          # files + lock state, runs nothing
    .venv/bin/python scripts/gate.py engine --only tests/test_game.py

Ends with one line: `OBS gate <cmd>: files N, passed P, failed F, xfailed X, errors E, wall S s`
and the FAILED ids. Exit 1 if anything failed or errored.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

from catspace.io import paths

REPO_ROOT = Path(__file__).resolve().parents[1]
TESTS = REPO_ROOT / "tests"
LOCK = Path(tempfile.gettempdir()) / "catspace_gate.lock"

ENGINE_NAMES = (
    "test_plan62 test_plan65 test_plan66 test_plan67 test_plan68 test_plan69 test_plan70 "
    "test_plan72 test_regress test_agenda test_tiers test_binder_rust test_fact_store test_game "
    "test_graph_policy test_question_prior test_offbuilt64 test_cost61 test_duty63 test_estimate61 "
    "test_forcing61 test_discovery_race test_discovery_vocabulary test_discovery_evidence "
    "test_discovery_known test_least_loss_keeps_material test_floor_at_decision test_anytime "
    "test_plan72_3 test_plan72_5 test_plan72_8 test_plan72_9 test_plan72_10 "          # 2026-10-25: the plan 72 / 73 files each agent gated alone, never on main's gate
    "test_plan72_7 test_plan72_7_step2 test_plan72_7_step3 test_plan73 test_plan73_5 test_plan73_5b test_plan73_6 "
    "test_plan74 "                                                                      # 2026-10-27: plan 74 the reasoning graph (the graph store, the worklist, the pending protocol, observations)
    "test_plan75 "                                                                      # 2026-10-27: plan 75.2 the move (goal kinds and the valuation from the store, the stops, the names table, the generic test)
    "test_rules_draws "                                                                 # 2026-10-28: plan 75.4 the kernel's rule gaps (the draws by rule, the cycle claim, the clock's events, the identity, promotion)
    "test_rules_gdl "                                                                   # 2026-10-28: plan 76.0 the rule spec in GDL (the interpreter, the surprises replayed, the generated rook, --jobs)
    "test_plan76 "                                                                      # 2026-10-28: plan 76.1 the ontology shelf (the bindings agree; the extensions; the terms, the GNN's kinds, the Rust fix)
    "test_rules_shelf "                                                                 # 2026-10-28: plan 76.2 the rule shelf (legal / terminal / goal proved; the GDL oracle against Game; the RULE entries)
    "test_plan76_4a "                                                                   # 2026-10-28: plan 76.4a defects and fixes from the spec (the regression; the hand tables as oracles)
    "test_plan76_4b "                                                                   # 2026-10-29: plan 76.4b the regression in Rust (the spec loaded from its text; the Python reading the oracle)
    "test_plan76_4c "                                                                   # 2026-10-29: plan 76.4c mining objects and material (the term language; mined relations; the mined valuation)
    "test_plan76_5 "                                                                     # 2026-10-29: plan 76.5 the valuation as one definition (the core's table from the store; the scan; the readers agree)
    "test_plan77_1 test_plan77_1d test_plan78 "                                                     # 2026-11-04: plan 77.1 (a) the one definition's port; (d) the certify stage (fresh certificates, online demotion)
    "test_plan78_12 test_plan78_14 "                                                                 # 2026-11-06: plan 78.12 the interface and the batch solver; 78.13-78.15 the fact layer, snapshots, wake and sleep
    "test_plan78_17 test_plan78_18 test_plan78_19 test_plan78_19e test_plan78_20 test_plan78_21 test_plan78_21_concepts test_plan78_22 test_plan78_23 test_plan78_24 test_plan78_24_concepts test_plan78_24_loop test_plan78_25_reason test_plan78_26_wdl test_plan78_27_wdl_sleep test_plan78_28_playout_sleep test_plan78_29_targeted test_plan78_31_result_standard test_plan78_32_exam_drill_induce test_plan78_32_keys test_plan78_33_tiers test_plan78_34_corpus_problems test_plan78_35_words test_plan78_36_claims test_plan78_37_surprise_meter test_plan78_38_construction test_plan78_39_scenario test_plan78_40_fact_forward test_plan78_41_question_sheet test_plan78_42_sheet_player test_plan78_42_light_sheet test_plan78_42_sheet_events test_plan78_43_rust_sheet test_plan78_44_opportunity test_plan78_45_puzzle_sheet test_plan78_46_forcing_read test_plan78_47_forcing_spent test_plan78_48_forcing_opportunity test_plan78_49_subgoal_fit test_plan78_50_subgoal_play test_plan78_51_root_order test_plan78_52_graph_layer1 test_plan78_53_graph_layer2 test_plan78_54_unit_fates test_plan78_55_frames test_plan78_57_frame_questions test_plan78_58_estimator_exchange test_plan78_60_settled test_plan78_59_human_heads test_plan78_61_odds test_plan78_62_state test_plan78_63_sleep test_plan78_64_events test_plan78_64_hazard test_plan78_64_fill test_plan78_65_frames test_plan78_66_sleep test_plan78_67_tiebreak test_plan78_69_readings test_plan78_70_threats test_plan78_72_concrete test_plan78_73_verdict test_plan78_71_goals test_plan78_74_every_unit test_plan78_75_move_rule test_plan78_68_ladder test_plan78_68_deep_heads test_plan78_76_records test_plan78_77_events test_plan78_78_goal_pick test_plan78_79_defence test_plan78_80_symbols test_plan78_81_reasons test_plan78_82_play test_plan78_83_window test_plan78_84_cut test_plan78_85_debts test_plan78_86_reread test_plan78_87_verdict test_plan78_88_groups"                                                               # 2026-11-08: plan 78.17 the two kits: diagnosis, rank sources (teacher, estimate), the sleep's tool choice
).split()

PYTEST_FLAGS = ["-q", "-p", "no:cacheprovider", "-W", "ignore", "--durations=3"]
COUNT = re.compile(r"(\d+) (passed|failed|xfailed|errors?|xpassed|skipped|deselected|warnings?)")


def engine_files() -> tuple[list[Path], list[str]]:
    """(existing engine-gate files, names that are missing)."""
    have, missing = [], []
    for n in ENGINE_NAMES:
        p = TESTS / f"{n}.py"
        if p.exists():
            have.append(p)
        else:
            missing.append(n)
    return have, missing


def full_files() -> list[Path]:
    return sorted(TESTS.glob("test_*.py"))


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def read_lock() -> dict | None:
    """The live holder of the lock, or None. A stale lock (dead pid, unreadable) is removed."""
    try:
        info = json.loads(LOCK.read_text())
    except FileNotFoundError:
        return None
    except (ValueError, OSError):
        info = {}
    if info.get("pid") and pid_alive(int(info["pid"])):
        return info
    try:
        LOCK.unlink()
    except FileNotFoundError:
        pass
    return None


def acquire(cmd: str) -> dict | None:
    """Take the lock; return None on success, else the holder's info (refuse)."""
    for _ in range(3):
        holder = read_lock()
        if holder:
            return holder
        try:
            fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            continue
        with os.fdopen(fd, "w") as f:
            json.dump({"pid": os.getpid(), "cmd": cmd, "start": datetime.now().isoformat(timespec="seconds")}, f)
        return None
    return read_lock() or {"pid": "?", "start": "?", "cmd": "?"}


def release() -> None:
    try:
        if json.loads(LOCK.read_text()).get("pid") == os.getpid():
            LOCK.unlink()
    except (OSError, ValueError):
        pass


def other_pytests() -> list[str]:
    """`ps` lines of other running pytest processes (never killed, only listed)."""
    try:
        out = subprocess.run(["ps", "-axo", "pid,etime,command"], capture_output=True, text=True).stdout
    except OSError:
        return []
    me = {os.getpid(), os.getppid()}
    found = []
    for line in out.splitlines()[1:]:
        parts = line.split(None, 2)
        if len(parts) < 3 or not parts[0].isdigit() or int(parts[0]) in me:
            continue
        if re.match(r"(\S*python\S*\s+(-\S+\s+)*-m\s+pytest\b|\S*/(py\.)?pytest\b|(py\.)?pytest\b)", parts[2]) \
                and "gate.py" not in parts[2]:
            found.append(line.strip())
    return found


def parse_summary(text: str) -> dict[str, int] | None:
    """Counts from the last pytest summary line of one file's output, or None if absent."""
    for line in reversed(text.splitlines()):
        if re.search(r" in \d+(\.\d+)?s", line) and COUNT.search(line):
            c = {"passed": 0, "failed": 0, "xfailed": 0, "errors": 0}
            for n, k in COUNT.findall(line):
                k = "errors" if k.startswith("error") else k
                if k in c:
                    c[k] += int(n)
            return c
    return None


def run_file(f: Path, log) -> tuple[dict[str, int], list[str]]:
    rel = f.relative_to(REPO_ROOT)
    log.write(f"\n===== {rel} =====\n")
    log.flush()
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    r = subprocess.run([sys.executable, "-m", "pytest", str(rel), *PYTEST_FLAGS], cwd=REPO_ROOT,
                       env=env, capture_output=True, text=True)
    out = r.stdout + r.stderr
    log.write(out)
    log.flush()
    c = parse_summary(out)
    if c is None:                       # crash / collection error with no summary line
        c = {"passed": 0, "failed": 0, "xfailed": 0, "errors": 0 if r.returncode in (0, 5) else 1}
    failed = [l.split(" - ")[0].strip() for l in out.splitlines() if l.startswith(("FAILED ", "ERROR "))]
    return c, failed


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["engine", "full"])
    ap.add_argument("--dry-run", action="store_true", help="print the files and the lock state; run nothing")
    ap.add_argument("--only", action="append", default=[], help="run just this file (repeatable)")
    ap.add_argument("--log", default=None, help="log path (default artifacts/experiments/gate_<cmd>_<ts>.log)")
    a = ap.parse_args(argv)

    if a.cmd == "engine":
        files, missing = engine_files()
    else:
        files, missing = full_files(), []
    if a.only:
        files = [Path(o) if Path(o).is_absolute() else REPO_ROOT / o for o in a.only]
        missing = [str(f) for f in files if not f.exists()]
        files = [f for f in files if f.exists()]
    for m in missing:
        print(f"MISSING (skipped): {m}")
    print(f"gate {a.cmd}: {len(files)} files")
    for f in files:
        print(f"  {f.relative_to(REPO_ROOT)}")

    if a.dry_run:
        holder = read_lock()
        print(f"lock {LOCK}: " + (f"HELD by pid {holder['pid']} since {holder['start']} ({holder.get('cmd')})" if holder else "free"))
        for l in other_pytests():
            print(f"  warning: pytest running: {l}")
        return 0

    holder = acquire(a.cmd)
    if holder:
        print(f"REFUSED: gate '{holder.get('cmd')}' is already running (pid {holder['pid']}, started {holder['start']}). "
              f"Timed tests are flaky under contention; wait for it to finish. Lock: {LOCK}")
        return 2
    try:
        others = other_pytests()
        if others:
            print("WARNING: other pytest processes are running (not killed; timed tests may flake):")
            for l in others:
                print(f"  {l}")
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = a.log or paths.experiment(f"gate_{a.cmd}_{stamp}.log")
        print(f"log: {log_path}")
        tot = {"passed": 0, "failed": 0, "xfailed": 0, "errors": 0}
        failed: list[str] = []
        t0 = time.time()
        with open(log_path, "a") as log:
            for f in files:
                t1 = time.time()
                c, fl = run_file(f, log)
                for k in tot:
                    tot[k] += c[k]
                failed += fl
                print(f"  {f.name}: passed {c['passed']}, failed {c['failed']}, xfailed {c['xfailed']}, "
                      f"errors {c['errors']} ({time.time() - t1:.1f} s)", flush=True)
        wall = time.time() - t0
        print(f"OBS gate {a.cmd}: files {len(files)}, passed {tot['passed']}, failed {tot['failed']}, "
              f"xfailed {tot['xfailed']}, errors {tot['errors']}, wall {wall:.1f} s")
        for i in failed:
            print(f"  {i}")
        return 1 if (tot["failed"] or tot["errors"]) else 0
    finally:
        release()


if __name__ == "__main__":
    sys.exit(main())
