"""Replace CWD-relative path literals with path-registry calls.

"data/derived/foo.npz" only resolved when a script happened to be launched from the repo
root, which is why cross-module artifact handoff kept breaking. Every such literal becomes
a catspace.io.paths accessor: the directory is declared once in the registry, the filename
stays at the call site.

This walks the AST and replaces by source position. A line-based regex is not safe enough
here -- two things it gets wrong:

  * implicit concatenation. `default="data/shards/a,"\\n"data/shards/b"` is ONE string
    expression spanning two lines; rewriting the first line alone produces
    `paths.shards("a,")\\n"data/shards/b"`, which is a syntax error. Working from AST
    nodes, an implicitly-concatenated literal is a single node whose source segment does
    not fullmatch a lone quoted string, so it is skipped rather than mangled.
  * the registry itself. catspace/io/paths.py contains the literals it exists to define;
    rewriting them there yields `artifacts_dir()` calling `paths.artifacts_dir()`.

Other guards: docstrings are skipped, and only strings that are ENTIRELY a path (no
whitespace) are touched, so prose like "writes its output to data/derived/" stays prose.
f-strings keep their prefix: f"docs/figures/{name}.png" -> paths.figure(f"{name}.png").

Throwaway tooling for the 2026-08-03 restructure; delete once the migration lands.

    python tools/_migration/rewrite_path_literals.py            # report
    python tools/_migration/rewrite_path_literals.py --apply
"""
from __future__ import annotations

import ast
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SKIP_DIRS = {".venv", ".git", ".claude", "node_modules", "__pycache__", ".dvc", "mlruns", "_migration"}
# The registry defines these paths; it cannot also consume them.
SKIP_FILES = {ROOT / "catspace" / "io" / "paths.py"}

# Longest prefix first -- data/derived/sep must beat data/derived.
PREFIXES: list[tuple[str, str, str]] = [
    ("data/derived/sep/",      "paths.sep",        "str(paths.sep_dir())"),
    ("data/derived/reach/",    "paths.reach",      "str(paths.reach_dir())"),
    ("data/derived/",          "paths.derived",    "str(paths.derived_dir())"),
    ("data/shards/",           "paths.shards",     "str(paths.shards_dir())"),
    ("data/records/",          "paths.records",    "str(paths.records_dir())"),
    ("data/selfplay/",         "paths.selfplay",   "str(paths.selfplay_dir())"),
    ("data/lichess/",          "paths.lichess",    "str(paths.lichess_dir())"),
    ("data/syzygy/",           "paths.syzygy",     "str(paths.syzygy_dir())"),
    ("data/engines/",          "paths.engine",     "str(paths.engines_dir())"),
    ("data/raw/",              "paths.raw",        "str(paths.raw_dir())"),
    ("data/eco/",              "paths.eco",        "str(paths.eco_dir())"),
    ("artifacts/experiments/", "paths.experiment", "str(paths.experiments_dir())"),
    ("artifacts/generated/",   "paths.generated",  "str(paths.generated_dir())"),
    ("artifacts/",             "paths.artifact",   "str(paths.artifacts_dir())"),
    ("docs/figures/",          "paths.figure",     "str(paths.figures_dir())"),
]
BARE = {p.rstrip("/"): bare for p, _, bare in PREFIXES}
BARE["data/lichess_db_puzzle.csv.zst"] = "str(paths.puzzle_db())"

# A source segment that is exactly one single-line quoted literal, nothing else.
LONE_STRING = re.compile(r"""(?P<pre>[fF]?)(?P<q>['"])(?P<body>[^'"\n]*)(?P=q)""")
IMPORT_LINE = "from catspace.io import paths\n"


def replacement(pre: str, q: str, body: str) -> str | None:
    if not body or re.search(r"\s", body):      # prose, or empty -- not a path
        return None
    body = body[2:] if body.startswith("./") else body
    if body in BARE:
        return BARE[body]
    for prefix, acc, _ in PREFIXES:
        if body.startswith(prefix):
            rest = body[len(prefix):]
            return f"{acc}({pre}{q}{rest}{q})" if rest else BARE[prefix.rstrip("/")]
    return None


def docstring_nodes(tree: ast.AST) -> set[int]:
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            d = node.body[0] if node.body else None
            if isinstance(d, ast.Expr) and isinstance(d.value, ast.Constant) \
                    and isinstance(d.value.value, str):
                out.add(id(d.value))
    return out


def import_insert_line(tree: ast.Module, lines: list[str]) -> int:
    last = 0
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            last = max(last, node.end_lineno or node.lineno)
        elif last:
            break
    if last:
        return last
    floor = 1 if lines and lines[0].startswith("#!") else 0
    doc = tree.body[0] if tree.body else None
    if isinstance(doc, ast.Expr) and isinstance(doc.value, ast.Constant) \
            and isinstance(doc.value.value, str):
        floor = max(floor, doc.value.end_lineno or doc.value.lineno)
    return floor


def process(path: Path) -> tuple[str, int] | None:
    text = path.read_text()
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return None
    skip = docstring_nodes(tree)
    lines = text.splitlines(keepends=True)
    # byte offset of the start of each line, for position -> index
    starts, acc = [], 0
    for ln in lines:
        starts.append(acc)
        acc += len(ln)

    edits: list[tuple[int, int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Constant, ast.JoinedStr)):
            continue
        if isinstance(node, ast.Constant) and not isinstance(node.value, str):
            continue
        if id(node) in skip:
            continue
        if node.lineno != node.end_lineno:          # multi-line: implicit concat or triple
            continue
        seg = ast.get_source_segment(text, node)
        if seg is None:
            continue
        m = LONE_STRING.fullmatch(seg)              # rejects implicit concat on one line
        if not m:
            continue
        rep = replacement(m.group("pre"), m.group("q"), m.group("body"))
        if rep is None:
            continue
        a = starts[node.lineno - 1] + node.col_offset
        b = starts[node.end_lineno - 1] + node.end_col_offset
        edits.append((a, b, rep))

    if not edits:
        return None
    edits.sort(reverse=True)
    out = text
    for a, b, rep in edits:
        out = out[:a] + rep + out[b:]

    if not re.search(r"^from catspace\.io import paths$", out, re.M):
        olines = out.splitlines(keepends=True)
        olines.insert(import_insert_line(tree, lines), IMPORT_LINE)
        out = "".join(olines)
    return out, len(edits)


def main() -> int:
    apply = "--apply" in sys.argv
    per_file: Counter[str] = Counter()
    results: dict[Path, str] = {}
    for p in sorted(ROOT.rglob("*.py")):
        rel = p.relative_to(ROOT)
        if any(part in SKIP_DIRS for part in rel.parts) or p in SKIP_FILES:
            continue
        if p.is_symlink() or not p.is_file():
            continue
        got = process(p)
        if got:
            results[p], per_file[str(rel)] = got[0], got[1]

    print(f"{len(results)} files, {sum(per_file.values())} literals")
    for f, n in per_file.most_common(8):
        print(f"  {n:4d}  {f}")
    if apply:
        for p, txt in results.items():
            p.write_text(txt)
        print("applied")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
