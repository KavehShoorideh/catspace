"""Create __init__.py for the new package dirs, and only those.

experiments/, logs/, artifacts/ and data/ deliberately stay non-packages so they
are excluded from the wheel and from setuptools discovery.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "catspace"
NON_PKG = {"experiments", "logs", "artifacts", "data", "__pycache__", "vendor",
           "docs", "weekly_report", "assets", "web", "docker", "templates"}

created = []
for d in sorted(p for p in PKG.rglob("*") if p.is_dir()):
    rel = d.relative_to(ROOT)
    if any(part in NON_PKG for part in rel.parts):
        continue
    init = d / "__init__.py"
    if not init.exists():
        init.write_text("")
        created.append(str(rel))

for c in created:
    print(f"+ {c}/__init__.py")
print(f"\ncreated {len(created)} __init__.py")
