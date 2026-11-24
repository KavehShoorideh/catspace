"""PLAN 79.37, 79.38: make the generated Rust files again (rust/engine/game/tests/generated/<game>.rs and <game>_<variant>.rs) from the descriptions with the translator (catspace/incubator/analyst/discovery/gdl_to_rust.py).
This script names the games; the translator does not. Run it after the translator or a description changed: tests/test_plan79_37.py and tests/test_plan79_38.py fail on a file whose hashes differ.
The plain file of a game has no pass; the others have the passes named in VARIANTS (each alone, then all three together as `fast`).
  .venv/bin/python scripts/gdl_to_rust_make.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
from catspace.incubator.analyst.discovery import gdl_to_rust as T

GAMES = {"chess": [ROOT / "rules" / "chess.gdl"], "atomic": [ROOT / "rules" / "chess.gdl", ROOT / "rules" / "variants" / "atomic.gdl"]}
VARIANTS = {"dense": ["dense"], "demand": ["demand"], "copy": ["copy"], "delta": ["delta"], "fast": ["dense", "demand", "copy", "delta"]}
OUT = ROOT / "rust" / "engine" / "game" / "tests" / "generated"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for game, sources in GAMES.items():
        text = T.compose_text(sources) if len(sources) > 1 else sources[0].read_text()
        for variant, passes in [("", [])] + [(f"_{k}", v) for k, v in VARIANTS.items()]:
            module = game + variant
            src = T.translate(text, module, passes)
            (OUT / f"{module}.rs").write_text(src)
            print(f"OBS 79.38 translate {module}: description {len(text)} bytes, passes {','.join(passes) or '-'}, Rust {len(src)} bytes {src.count(chr(10))} lines")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
