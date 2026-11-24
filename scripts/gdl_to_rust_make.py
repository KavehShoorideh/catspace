"""PLAN 79.37, 79.38, 79.40: make the generated Rust files again (rust/engine/game/tests/generated/<game>.rs and <game>_<variant>.rs) from the descriptions with the translator (catspace/incubator/analyst/discovery/gdl_to_rust.py).
This script names the games; the translator does not. Run it after the translator or a description changed: tests/test_plan79_37.py and tests/test_plan79_40.py fail on a file whose hashes differ.
Only the plain file of a game (no pass) and the file with every pass (`fast`) are kept in the repository (plan 79.40 part F). With --scratch the files of the single passes and of the four passes of 79.38 together (`prev`) are made under
generated/scratch (not kept: that folder is ignored); the tests that hold each pass to the plain module, and the clocks, are compiled with `--cfg singles` and read them there.
  .venv/bin/python scripts/gdl_to_rust_make.py [--scratch]
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
from catspace.incubator.analyst.discovery import gdl_to_rust as T

GAMES = {"chess": [ROOT / "rules" / "chess.gdl"], "atomic": [ROOT / "rules" / "chess.gdl", ROOT / "rules" / "variants" / "atomic.gdl"]}
KEPT = {"": [], "_fast": list(T.PASSES)}
COMBOS = {"bits_keep": ["bits", "keep"], "bits_order": ["bits", "order"], "bits_order_keep": ["bits", "order", "keep"], "bits_demand": ["bits", "demand"], "bits_dense": ["bits", "dense"], "bits_copy": ["bits", "copy"], "bits_delta": ["bits", "delta"]}
# plan 79.41: each new pass alone on top of bits (the clocks), and `prev`, the module of every pass of 79.40 (the one the new passes are clocked against); `prev38` is the four passes of 79.38
PASSES_7940 = ["dense", "demand", "copy", "delta", "bits", "order", "keep"]
NEW = [p for p in T.PASSES if p not in PASSES_7940]
COMBOS.update({f"bits_{p}": ["bits", p] for p in NEW})
SCRATCH = {"_prev": PASSES_7940, "_prev38": ["dense", "demand", "copy", "delta"], **{f"_{p}": [p] for p in T.PASSES}, **{f"_{k}": v for k, v in COMBOS.items()}}
OUT = ROOT / "rust" / "engine" / "game" / "tests" / "generated"
OUT_SCRATCH = OUT / "scratch"


def make(out: Path, variants: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for game, sources in GAMES.items():
        text = T.compose_text(sources) if len(sources) > 1 else sources[0].read_text()
        for variant, passes in variants.items():
            module = game + variant
            src = T.translate(text, module, passes)
            (out / f"{module}.rs").write_text(src)
            print(f"OBS 79.40 translate {module}: description {len(text)} bytes, passes {','.join(passes) or '-'}, Rust {len(src)} bytes {src.count(chr(10))} lines")


def main(argv=None) -> int:
    args = sys.argv[1:] if argv is None else argv
    make(OUT, KEPT)
    if "--scratch" in args:
        make(OUT_SCRATCH, SCRATCH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
