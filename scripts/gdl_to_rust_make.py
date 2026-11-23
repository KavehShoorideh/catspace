"""PLAN 79.37: make the generated Rust files again (rust/engine/game/tests/generated/<game>.rs) from the descriptions with the translator (catspace/incubator/analyst/discovery/gdl_to_rust.py).
This script names the games; the translator does not. Run it after the translator or a description changed: tests/test_plan79_37.py fails on a file whose hashes differ.
  .venv/bin/python scripts/gdl_to_rust_make.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
from catspace.incubator.analyst.discovery import gdl_to_rust as T

GAMES = {"chess": [ROOT / "rules" / "chess.gdl"], "atomic": [ROOT / "rules" / "chess.gdl", ROOT / "rules" / "variants" / "atomic.gdl"]}
OUT = ROOT / "rust" / "engine" / "game" / "tests" / "generated"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for module, sources in GAMES.items():
        text = T.compose_text(sources) if len(sources) > 1 else sources[0].read_text()
        src = T.translate(text, module)
        (OUT / f"{module}.rs").write_text(src)
        print(f"OBS 79.37 translate {module}: description {len(text)} bytes, Rust {len(src)} bytes {src.count(chr(10))} lines")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
