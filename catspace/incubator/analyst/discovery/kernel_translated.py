"""PLAN 79.46 PART B (SYSTEM_SPEC 11 "The description is the source of the rules; fast code is only a certified copy"; piece: rules; ledger rows 44, 45): THE RECORD OF THE KERNEL HELD TO THE TRANSLATED MODULE.

THIS TEXT IS THE INTERFACE, written before the body. Game-free: the game comes as an argument; the module names no game.

WHAT IT KEEPS. For a game whose description has a kept translated module (rust/engine/game/tests/generated/<game>_fast.rs, every pass), the Rust test rust/engine/game/tests/translated_kernel.rs co-plays the
kernel and the module on seeded random games and compares, at every state, the five answers (whose turn, the allowed moves, the next state after every allowed move, whether ended, the result). This module
runs it, reads the counts it writes and keeps a RECORD of them with the hashes of what the answers depend on. The record is of its own kind: it lives in `discovery/lemmas/copy_records/<game>.json`, apart
from the twelve certificates of discovery/certify.py (`certify status` keeps printing twelve and the tests that count them are not touched) and from the lemma certificates (rules/lemmas).
THE RECORD (JSON, one object per game):
  binding, ingests, oracle      what is held to what (the kernel to the translated module, itself held to the evaluator of the description)
  game, module                  the game's name and the path of the module file
  small                         the run on the existing certificate's corpus: {games, seed, plies, states, moves, nexts, ended, results, left_out{class: n}, causes{cause: n}, examples[...], seconds}
                                and `certificate_positions` (the positions of the kernel certificate of the game) and `equals_certificate` (states == certificate_positions)
  large                         the run on the fixed seed list: the same counts and {seeds, seed0, games_a_seed, cores, seconds, states_a_second}
  disagreements                 the total of both runs (an integer); `examples` keeps the first few of each cause WHOLE: {cause, seed, game, line, state, kernel, translated, evaluator}
  hashes                        sha256 (16 hex) of: the description text (the file and its delta), the translator (gdl_to_rust.py), the module file, kernel.rs, outside.rs (the bridge `state_facts`), the test file, Cargo.lock
  key                           16 hex digits over the hashes
  date, seconds                 the day the record was made and the wall of the remake
STATUS (`status(game)`): 'missing' (no record), 'failing' (it found a disagreement), 'stale' (its key is not the current files'), 'short' (the small run's states differ from the certificate's, or a count is
zero), else 'fresh'. The gate (tests/test_plan79_46.py) fails on anything but fresh.
INTERFACE (who calls it: the tests; `python -m catspace.incubator.analyst.discovery.kernel_translated run <game> [--small-only]`, `status [<game>]`):
  `hashes_of(game) -> dict`                     the hashes above, from the files as they are now
  `key_of(game) -> str`                         the 16 hex digits
  `path_of(game) -> Path`, `load(game) -> dict | None`, `write(game, record)`
  `status(game) -> str`                         as above
  `small_counts(game) -> dict`                  run the small test of the game and return its counts file (what the gate's own small run reads)
  `run(game, small_only=False) -> dict`         make the record: the small test, then the large (the ignored test, all cores, a few minutes), the hashes, the date; written; the OBS lines printed
  `GAMES`                                       the games with a kept translated module: {name: {module, delta, test names}} (a table of facts about files, no rule)
"""
from __future__ import annotations


def hashes_of(game: str) -> dict:
    raise NotImplementedError("plan 79.46 part B")


def key_of(game: str) -> str:
    raise NotImplementedError("plan 79.46 part B")


def status(game: str) -> str:
    raise NotImplementedError("plan 79.46 part B")


def run(game: str, small_only: bool = False) -> dict:
    raise NotImplementedError("plan 79.46 part B")
