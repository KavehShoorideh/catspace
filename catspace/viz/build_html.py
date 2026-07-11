#!/usr/bin/env python
"""
viz/build_html.py — the missing build step: inject a JSON payload into a
viewer template's `const DATA = /*__DATA__*/;` placeholder.

No committed script previously did this -- the generators stopped at
json.dump() and the final HTML artifacts were produced by a manual/external
step. This closes that gap.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from catspace.viz.payload import json_default

PLACEHOLDER = "const DATA = /*__DATA__*/;"


def build_html(template: Path, data: dict, out: Path) -> None:
    template = Path(template)
    out = Path(out)
    html = template.read_text()
    if PLACEHOLDER not in html:
        raise ValueError(f"{template} has no {PLACEHOLDER!r} placeholder")
    payload = json.dumps(data, default=json_default)
    html = html.replace(PLACEHOLDER, f"const DATA = {payload};")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", required=True, type=Path)
    ap.add_argument("--data", required=True, type=Path, help="path to a JSON file")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    data = json.loads(args.data.read_text())
    build_html(args.template, data, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
