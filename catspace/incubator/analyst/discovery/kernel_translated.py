"""PLAN 79.46 PART B (SYSTEM_SPEC 11 "The description is the source of the rules; fast code is only a certified copy"; piece: rules; ledger rows 44, 45): THE RECORD OF THE KERNEL HELD TO THE TRANSLATED MODULE.

THIS TEXT IS THE INTERFACE, written before the body. Game-free: the game comes as an argument; the module names no game.

WHAT IT KEEPS. For a game whose description has a kept translated module (rust/engine/game/tests/generated/<game>_fast.rs, every pass), the Rust test rust/engine/game/tests/translated_kernel.rs co-plays the
kernel and the module on seeded random games and compares, at every state, the five answers (whose turn, the allowed moves, the next state after every allowed move, whether ended, the result). This module
runs it, reads the counts it writes and keeps a RECORD of them with the hashes of what the answers depend on. The record is of its own kind: it lives in `discovery/lemmas/copy_records/<game>.json`, apart
from the twelve certificates of discovery/certify.py (`certify status` keeps printing twelve and the tests that count them are not touched) and from the lemma certificates (rules/lemmas).
THE RECORD (JSON, one object per game):
  binding, ingests, oracle      what is held to what (the kernel to the translated module, itself held to the evaluator of the description)
  game, module                  the game's name and the path of the module file
  small                         the run on the existing certificate's corpus: {games, plies, states, moves, nexts, ended, results, left_out{class: n}, causes{cause: n}, examples[...], seconds}
                                and `certificate_positions` (the positions of the kernel certificate of the game) and `equals_certificate` (states == certificate_positions)
  large                         the run on the fixed seed list: the same counts and {seeds, seed0, games_a_seed, cores, seconds, states_a_second}
  chain                         the numbers of the chain: the states the module was held to the evaluator on (translated_agree.rs) and the comparisons it was held to the plain module by (translated_pieces.rs),
                                beside the states the kernel is now held to the module on
  disagreements                 the total of both runs (an integer); `examples` keeps the first few of each cause WHOLE: {cause, seed, game, line, state, kernel, translated, evaluator}
  hashes                        sha256 (16 hex) of: the description text (the file, its delta and its list of starts), the translator (gdl_to_rust.py), the module file, kernel.rs, outside.rs (the bridge `state_facts`), the
                                test file and its common part, Cargo.lock; the header lines the translator wrote in the module file are kept as they are
  key                           16 hex digits over the hashes
  date, seconds                 the day the record was made and the wall of the remake
STATUS (`status(game)`): 'missing' (no record), 'failing' (it found a disagreement), 'stale' (its key is not the current files'), 'short' (the small run's states differ from the current certificate's, or a
count is zero), else 'fresh'. The gate (tests/test_plan79_46.py) fails on anything but fresh.
INTERFACE (who calls it: the tests; `python -m catspace.incubator.analyst.discovery.kernel_translated run <game> [--small-only] [--no-chain]`, `status [<game>]`, `chain <game>`):
  `games() -> list`                             the games with a kept module (the files `<game>_fast.rs`)
  `hashes_of(game) -> dict`                     the hashes above, from the files as they are now
  `key_of(game) -> str`                         the 16 hex digits
  `path_of(game) -> Path`, `load(game) -> dict | None`, `write(game, record)`
  `status(game, record=None) -> str`            as above
  `small_counts(game) -> dict`                  run the small test of the game and return the counts file it wrote (the gate's own small run)
  `chain(game) -> dict`                         run the tests that hold the module to the evaluator and to the plain module and read their OBS lines
  `run(game, small_only=False, with_chain=True) -> dict`   make the record: the small test, the large (the ignored test, all cores, a few minutes), the chain, the hashes, the date; written; the OBS lines printed
Nothing here is on the engine's path: the Rust test only reads kernel.rs and the module.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from catspace.incubator.analyst.discovery import certify as CF
from catspace.incubator.analyst.discovery import lemma_store as LS

ROOT = CF.ROOT
ENGINE = CF.ENGINE
TESTS = ENGINE / "game" / "tests"
GEN = TESTS / "generated"
DIR = LS.DIR / "copy_records"
TRANSLATOR = CF.DISC / "gdl_to_rust.py"
SHARED = [ENGINE / "game" / "src" / "kernel.rs", ENGINE / "game" / "src" / "outside.rs", TESTS / "translated_kernel.rs", TESTS / "translated_common" / "mod.rs", ENGINE / "Cargo.lock"]
EXAMPLES = 5


def games() -> list:
    return sorted(p.name[: -len("_fast.rs")] for p in GEN.glob("*_fast.rs"))


def module_path(game: str) -> Path:
    return GEN / f"{game}_fast.rs"


def binding_of(game: str) -> str:
    return f"kernel {game} (cs_game)"


def description_files(game: str) -> list:
    """The description of the game as files: the base text, its delta, its list of starts (certify.py's own table of them)."""
    out = [CF.GC.SPEC]
    if CF.VARIANT_TEXTS.get(game) is not None:
        out.append(CF.VARIANT_TEXTS[game])
    if CF.VARIANT_STARTS.get(game) is not None:
        out.append(CF.VARIANT_STARTS[game])
    return out


def _h(*paths) -> str:
    h = hashlib.sha256()
    for p in paths:
        h.update(str(Path(p).relative_to(ROOT)).encode())
        h.update(Path(p).read_bytes() if Path(p).is_file() else b"")
    return h.hexdigest()[:16]


def header_of(game: str) -> dict:
    """The lines the translator wrote at the head of the module file (the hashes it made the module from)."""
    out = {}
    for ln in module_path(game).read_text().splitlines()[:6]:
        m = re.match(r"// ([a-z0-9-]+): (.*)", ln)
        if m:
            out[m.group(1)] = m.group(2)
    return out


def hashes_of(game: str) -> dict:
    return {"description": _h(*description_files(game)), "translator": _h(TRANSLATOR), "module": _h(module_path(game)), "kernel": _h(ENGINE / "game" / "src" / "kernel.rs"),
            "bridge": _h(ENGINE / "game" / "src" / "outside.rs"), "test": _h(TESTS / "translated_kernel.rs", TESTS / "translated_common" / "mod.rs"), "lock": _h(ENGINE / "Cargo.lock")}


def key_of(game: str) -> str:
    h = hashlib.sha256()
    for k, v in sorted(hashes_of(game).items()):
        h.update(f"{k}={v};".encode())
    return h.hexdigest()[:16]


def path_of(game: str) -> Path:
    return DIR / f"{game}.json"


def load(game: str) -> "dict | None":
    p = path_of(game)
    return json.loads(p.read_text()) if p.exists() else None


def write(game: str, record: dict) -> None:
    DIR.mkdir(exist_ok=True)
    path_of(game).write_text(json.dumps(record, indent=1, ensure_ascii=True) + "\n")


def certificate_positions(game: str) -> "int | None":
    c = CF.load(binding_of(game))
    return None if c is None else c["positions"]


def status(game: str, record: "dict | None" = None) -> str:
    r = record if record is not None else load(game)
    if r is None:
        return "missing"
    if r["disagreements"]:
        return "failing"
    if r.get("key") != key_of(game):
        return "stale"
    zero = any(not r.get(k, {}).get(f) for k in ("small", "large") for f in ("games", "states", "moves", "nexts", "ended", "results"))
    if zero or r["small"]["states"] != certificate_positions(game):
        return "short"
    return "fresh"


# ------------------------------------------------------------------------------------------------------------------------------------------ running the Rust tests

def cargo(args: list, env_extra: dict, stream: bool = False) -> tuple:
    """`cargo test --release --offline -p cs_game <args>` in rust/engine (a target dir of the environment's, CARGO_TARGET_DIR, is used when set). -> (return code, output text)."""
    cmd = ["cargo", "test", "--release", "--offline", "-p", "cs_game", *args]
    env = dict(os.environ, **env_extra)
    p = subprocess.Popen(cmd, cwd=ENGINE, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    out = []
    for ln in p.stdout:
        out.append(ln)
        if stream and (ln.startswith("OBS 79.46") or ln.startswith("FINDING")):
            print(ln.rstrip()[:600], flush=True)
    p.wait()
    return p.returncode, "".join(out)


def _counts_file(game: str, kind: str, tests: list, extra: list, env_extra: dict, stream: bool) -> dict:
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / f"{kind}.json"
        rc, text = cargo(["--test", "translated_kernel", *tests, "--", *extra, "--nocapture"], dict(env_extra, CS_KT_OUT=str(out)), stream)
        if not out.exists():
            raise RuntimeError(f"the {kind} run of {game} wrote no counts (cargo returned {rc}):\n" + text[-3000:])
        c = json.loads(out.read_text())
        c["test_returned"] = rc
        return c


def small_counts(game: str) -> dict:
    return _counts_file(game, "small", [f"{game}_the_kernel"], [], {}, False)


def large_counts(game: str, env_extra: "dict | None" = None) -> dict:
    return _counts_file(game, "large", [f"{game}_large"], ["--ignored"], env_extra or {}, True)


CHAIN_PATTERNS = {
    "coplay": r"OBS 79\.37 D coplay {g}: (\d+) seeded games \(seed \d+\), (\d+) plies, (\d+) states, (\d+) allowed moves, (\d+) next states compared",
    "fixture": r"OBS 79\.37 D fixture {g}: (\d+) positions \((\d+) ended\), (\d+) allowed moves, (\d+) moves made, (\d+) nexts",
    "counts": r"OBS 79\.37 D counts {g}: (\d+) positions counted",
    "relations": r"OBS 79\.37 D coplay {g}: (\d+) relations that read the state, (\d+) relation-by-state comparisons",
}


def chain(game: str) -> dict:
    """The numbers of the chain below the kernel: what the module (every pass) was held to the evaluator on (translated_agree.rs, the tests named `with_the_passes`) and to the plain module (translated_pieces.rs)."""
    rc, text = cargo(["--test", "translated_agree", f"{game}_with_the_passes", "--", "--nocapture"], {})
    if rc:
        raise RuntimeError("translated_agree failed:\n" + text[-2000:])
    g = re.escape(game)
    m = {k: re.search(p.format(g=g), text) for k, p in CHAIN_PATTERNS.items()}
    if not all(m.values()):
        raise RuntimeError("an OBS line of translated_agree was not found: " + str([k for k, v in m.items() if not v]))
    out = {"held_to_the_evaluator": {
        "coplay_states": int(m["coplay"].group(3)), "coplay_allowed_moves": int(m["coplay"].group(4)), "coplay_next_states": int(m["coplay"].group(5)),
        "coplay_relation_by_state": int(m["relations"].group(2)),
        "fixture_positions": int(m["fixture"].group(1)), "fixture_allowed_moves": int(m["fixture"].group(3)), "fixture_nexts": int(m["fixture"].group(5)),
        "counts_positions": int(m["counts"].group(1))}}
    rc, text = cargo(["--test", "translated_pieces", f"{game}_fast_", "--", "--nocapture"], {})
    if rc:
        raise RuntimeError("translated_pieces failed:\n" + text[-2000:])
    tot = [int(x) for x in re.findall(rf"OBS 79\.38 A pieces {g} fast [^:]*: (\d+) piece-by-piece comparisons", text)]
    out["held_to_the_plain_module"] = {"tests": len(tot), "piece_by_piece_comparisons": sum(tot)}
    return out


def run(game: str, small_only: bool = False, with_chain: bool = True) -> dict:
    t0 = time.time()
    if game not in games():
        raise ValueError(f"{game} has no kept translated module: {games()}")
    small = small_counts(game)
    cp = certificate_positions(game)
    small["certificate_positions"], small["equals_certificate"] = cp, small["states"] == cp
    print(f"OBS 79.46 B small {game}: {small['states']} states, the certificate's {cp}, equal {small['equals_certificate']}, disagreements {sum(small['causes'].values())}", flush=True)
    large = {} if small_only else large_counts(game)
    if large:
        large["states_a_second"] = round(large["states"] / max(large["seconds"], 1e-9))
    ch = chain(game) if (with_chain and not small_only) else {}
    for r in (small, large):
        r["examples"] = r.get("examples", [])[:EXAMPLES * 4]
    total = sum(small["causes"].values()) + sum(large.get("causes", {}).values())
    record = {"binding": f"the kernel of {game} held to its translated module",
              "ingests": "the rules of the game as the kernel answers them: the mover, the allowed moves, the next state after every allowed move, whether ended, the result (plan 79.46)",
              "oracle": "the translated module of the description (every pass), itself held to the evaluator of the description (translated_agree.rs) and to the plain translation piece by piece (translated_pieces.rs)",
              "game": game, "module": str(module_path(game).relative_to(ROOT)), "module_header": header_of(game),
              "small": small, "large": large, "chain": ch, "disagreements": total,
              "examples": small.get("examples", []) + large.get("examples", []),
              "hashes": hashes_of(game), "key": key_of(game), "date": datetime.date.today().isoformat(), "seconds": round(time.time() - t0, 1)}
    chain_states = ch.get("held_to_the_evaluator", {}).get("coplay_states")
    print(f"OBS 79.46 B record {game}: small {small['states']} states, large {large.get('states', 0)} states ({large.get('states_a_second', 0)} a second on {large.get('cores', 0)} cores), disagreements {total}, "
          f"the module held to the evaluator on {chain_states} co-play states, {record['seconds']} s", flush=True)
    write(game, record)
    return record


def main(argv: list) -> int:
    cmd = argv[1] if len(argv) > 1 else "status"
    names = [a for a in argv[2:] if not a.startswith("--")]
    if cmd == "status":
        bad = 0
        for g in names or games():
            s = status(g)
            r = load(g)
            print(f"OBS 79.46 status {g}: {s}" + (f", small {r['small']['states']} states, large {r['large'].get('states', 0)} states, disagreements {r['disagreements']}, {r['date']}" if r else ""))
            bad += s != "fresh"
        return 1 if bad else 0
    if cmd == "run":
        for g in names:
            run(g, "--small-only" in argv, "--no-chain" not in argv)
        return 0
    if cmd == "chain":
        for g in names:
            print(json.dumps(chain(g), indent=1))
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
