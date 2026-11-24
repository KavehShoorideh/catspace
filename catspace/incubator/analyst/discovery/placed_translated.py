"""PLAN 79.48 PART D (SYSTEM_SPEC 11 "The description is the source of the rules; fast code is only a certified copy"; piece: rules; ledger rows 44, 42): THE RECORD OF THE KERNEL'S PLACED ANSWERS HELD TO THE TRANSLATED MODULE OF THEIR DEFINITIONS.

THIS TEXT IS THE INTERFACE, written before the body. Game-free: the game comes as an argument; the module names no object of any game.

WHAT IT KEEPS. For a game whose description composed with the additive texts of rules/placed/ has a kept translated module (rust/engine/game/tests/generated_placed/<game>.rs, every pass, made by
`scripts/gdl_to_rust_make.py --placed`), the Rust test rust/engine/game/tests/translated_placed.rs holds the kernel's placed answers (placed_questions.py lists them) to the module's rows of the definitions at every state of seeded games:
the small corpus is the certificate's (its count of states is the certificate's `positions`), the large run a fixed list of seeds on all cores. On the small corpus the module's rows of each definition are also held to the
evaluator of the composed description (the lower link). This module runs the tests, reads the counts they write and keeps a RECORD of them with the hashes of what the answers depend on.
THE RECORD (JSON, one object a game, in discovery/lemmas/copy_records/placed/<game>.json, apart from the records of 79.46 and from the twelve certificates):
  binding, ingests, oracle     what is held to what
  game, module                 the game's name and the path of the module file
  small, large                 {games, plies, states, probes, ended, compared{question: [comparisons, rows]}, lower{relation: [comparisons, rows]}, lower_states, left_out{class: n}, causes{cause: n}, examples[...], seconds, ...}
                               and for the small one `certificate_positions` and `equals_certificate`
  known                        the causes this game's record may hold, each with its explanation (KNOWN below); a cause not named there makes the status `failing`
  disagreements                the total of both runs; `examples` keeps the first few of each cause WHOLE
  hashes, key                  sha256 (16 hex) of: the description files of the game, the documents of rules/placed/ that apply, the translator, the module file, kernel.rs, outside.rs, the test and its common part, Cargo.lock
STATUS (`status(game)`): 'missing', 'failing' (a cause that KNOWN does not name), 'stale' (its key is not the current files'), 'short' (the small run's states differ from the certificate's, or a question was not compared,
or a count is zero), else 'fresh'. A record with a named cause is fresh; the gate's test holds that the causes are exactly the named ones.
INTERFACE (who calls it: the tests; `python -m catspace.incubator.analyst.discovery.placed_translated run <game> [--small-only] [--games-a-seed=N]`, `status [<game>]`):
  `games() -> list`, `hashes_of(game) -> dict`, `key_of(game) -> str`, `path_of(game)`, `load(game)`, `write(game, record)`, `status(game, record=None) -> str`,
  `small_counts(game) -> dict`, `large_counts(game, env_extra=None) -> dict`, `run(game, small_only=False, games_a_seed=None, probes=None) -> dict`
Nothing here is on the engine's path: the Rust test only reads kernel.rs, outside.rs and the module.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from catspace.incubator.analyst.discovery import certify as CF
from catspace.incubator.analyst.discovery import kernel_translated as KT
from catspace.incubator.analyst.discovery import lemma_store as LS
from catspace.incubator.analyst.discovery import placed_questions as PQ

ROOT = CF.ROOT
ENGINE = CF.ENGINE
TESTS = ENGINE / "game" / "tests"
GEN = TESTS / "generated_placed"
DIR = LS.DIR / "copy_records" / "placed"
TRANSLATOR = CF.DISC / "gdl_to_rust.py"
EXAMPLES = 5
QUESTIONS = ["cells", "kind_at", "bearing_from", "bearing_from (given)", "reach_from", "reach_from (given)", "strikes_place", "move_ends", "find_step", "move_info (kinds played)", "move_info (kinds taken)",
             "place_names", "kind_names"]
LOWER = ["placed_cell", "placed_bears", "placed_strikes", "placed_mover", "placed_takes", "placed_reaches"]

# THE CAUSES A RECORD MAY HOLD, by game: {cause: its explanation}. Filled from the runs (each is a finding with its whole state in the record's examples); a cause that is not here makes the record `failing`.
KNOWN: dict = json.loads((ROOT / "rules" / "placed" / "known.json").read_text())["games"]


def games() -> list:
    return sorted(p.stem for p in GEN.glob("*.rs"))


def module_path(game: str) -> Path:
    return GEN / f"{game}.rs"


def _h(*paths) -> str:
    h = hashlib.sha256()
    for p in paths:
        h.update(str(Path(p).relative_to(ROOT)).encode())
        h.update(Path(p).read_bytes() if Path(p).is_file() else b"")
    return h.hexdigest()[:16]


def hashes_of(game: str) -> dict:
    return {"description": KT._h(*KT.description_files(game)), "definitions": _h(*PQ.documents(game)), "translator": _h(TRANSLATOR), "module": _h(module_path(game)),
            "kernel": _h(ENGINE / "game" / "src" / "kernel.rs"), "bridge": _h(ENGINE / "game" / "src" / "outside.rs"),
            "test": _h(TESTS / "translated_placed.rs", TESTS / "translated_common" / "mod.rs"), "lock": _h(ENGINE / "Cargo.lock")}


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
    DIR.mkdir(parents=True, exist_ok=True)
    path_of(game).write_text(json.dumps(record, indent=1, ensure_ascii=True) + "\n")


def unexplained(game: str, record: dict) -> dict:
    """The causes of a record that KNOWN does not name for the game."""
    named = KNOWN.get(game, {})
    out = {}
    for run in ("small", "large"):
        for c, n in record.get(run, {}).get("causes", {}).items():
            if c not in named:
                out[c] = out.get(c, 0) + n
    return out


def status(game: str, record: "dict | None" = None) -> str:
    r = record if record is not None else load(game)
    if r is None:
        return "missing"
    if unexplained(game, r):
        return "failing"
    if r.get("key") != key_of(game):
        return "stale"
    zero = any(not r.get(k, {}).get(f) for k in ("small", "large") for f in ("games", "states", "probes"))
    missing = any(q not in r.get(k, {}).get("compared", {}) for k in ("small", "large") for q in QUESTIONS) or any(q not in r.get("small", {}).get("lower", {}) for q in LOWER)
    cp = KT.certificate_positions(game)
    if zero or missing or r["small"]["states"] != cp:
        return "short"
    return "fresh"


# ------------------------------------------------------------------------------------------------------------------------------------------ running the Rust tests

def cargo(args: list, env_extra: dict, stream: bool = False) -> tuple:
    cmd = ["cargo", "test", "--release", "--offline", "-p", "cs_game", *args]
    env = dict(os.environ, **env_extra)
    p = subprocess.Popen(cmd, cwd=ENGINE, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    out = []
    for ln in p.stdout:
        out.append(ln)
        if stream and (ln.startswith("OBS 79.48") or ln.startswith("FINDING")):
            print(ln.rstrip()[:600], flush=True)
    p.wait()
    return p.returncode, "".join(out)


def _counts_file(game: str, kind: str, tests: list, extra: list, env_extra: dict, stream: bool) -> dict:
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / f"{kind}.json"
        rc, text = cargo(["--test", "translated_placed", *tests, "--", *extra, "--nocapture"], dict(env_extra, CS_PL_OUT=str(out)), stream)
        if not out.exists():
            raise RuntimeError(f"the {kind} run of {game} wrote no counts (cargo returned {rc}):\n" + text[-3000:])
        c = json.loads(out.read_text())
        c["test_returned"] = rc
        return c


def small_counts(game: str) -> dict:
    return _counts_file(game, "small", [f"{KT.rust_name(game)}_the_kernel_and_the_placed_module_on_the_certificates_games"], ["--exact"], {}, False)


def large_counts(game: str, env_extra: "dict | None" = None) -> dict:
    return _counts_file(game, "large", [f"{KT.rust_name(game)}_large_the_kernel_and_the_placed_module_on_the_fixed_seed_list"], ["--exact", "--ignored"], env_extra or {}, True)


def run(game: str, small_only: bool = False, games_a_seed: "int | None" = None, probes: "int | None" = None) -> dict:
    t0 = time.time()
    if game not in games():
        raise ValueError(f"{game} has no kept placed module: {games()}")
    small = small_counts(game)
    cp = KT.certificate_positions(game)
    small["certificate_positions"], small["equals_certificate"] = cp, small["states"] == cp
    print(f"OBS 79.48 D small {game}: {small['states']} states, the certificate's {cp}, equal {small['equals_certificate']}, {small['probes']} probes, the lower link at {small['lower_states']} states, disagreements {sum(small['causes'].values())} {small['causes']}", flush=True)
    large = {}
    if not small_only:
        env = {}
        if games_a_seed is not None:
            env["CS_PL_GAMES"] = str(games_a_seed)
        if probes is not None:
            env["CS_PL_PROBES"] = str(probes)
        large = large_counts(game, env)
        large["states_a_second"] = round(large["states"] / max(large["seconds"], 1e-9))
    for r in (small, large):
        r["examples"] = r.get("examples", [])[:EXAMPLES * 6]
    total = sum(small["causes"].values()) + sum(large.get("causes", {}).values())
    record = {"binding": f"the placed answers of the kernel of {game} held to the translated module of their definitions",
              "ingests": "the placed questions of layer 2 (rust/engine/game/src/placed.rs) as the kernel answers them: what stands where, what a unit bears on and may move to, what a side strikes, the ends of the allowed moves and what they play and take, the names (plan 79.48)",
              "oracle": "the translated module (every pass) of the description composed with rules/placed/, its rows held to the evaluator of the composed description on the small corpus (the lower link)",
              "game": game, "module": str(module_path(game).relative_to(ROOT)), "small": small, "large": large, "known": KNOWN.get(game, {}), "disagreements": total,
              "examples": small.get("examples", []) + large.get("examples", []), "hashes": hashes_of(game), "key": key_of(game), "date": datetime.date.today().isoformat(), "seconds": round(time.time() - t0, 1)}
    print(f"OBS 79.48 D record {game}: small {small['states']} states, large {large.get('states', 0)} states ({large.get('states_a_second', 0)} a second on {large.get('cores', 0)} cores), "
          f"{large.get('probes', 0)} probes, disagreements {total}, {record['seconds']} s", flush=True)
    write(game, record)
    return record


def refresh(game: str) -> dict:
    """Write the record again with the KNOWN of the game (the explanations of the causes it holds): nothing is run, no count and no hash changes."""
    r = load(game)
    r["known"] = KNOWN.get(game, {})
    write(game, r)
    return r


def main(argv: list) -> int:
    cmd = argv[1] if len(argv) > 1 else "status"
    names = [a for a in argv[2:] if not a.startswith("--")]
    if cmd == "status":
        bad = 0
        for g in names or games():
            s = status(g)
            r = load(g)
            print(f"OBS 79.48 status {g}: {s}" + (f", small {r['small']['states']} states, large {r['large'].get('states', 0)} states, disagreements {r['disagreements']}, {r['date']}" if r else ""))
            bad += s != "fresh"
        return 1 if bad else 0
    if cmd == "run":
        gs = next((int(a.split("=", 1)[1]) for a in argv if a.startswith("--games-a-seed=")), None)
        pr = next((int(a.split("=", 1)[1]) for a in argv if a.startswith("--probes=")), None)
        for g in names:
            run(g, "--small-only" in argv, gs, pr)
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
