"""THE LEAN INDEX: what we have proved, searchable (Kaveh 2026-10-10: "every frame or the general form of a frame or a
lemma proven in Lean and kept there as the source of truth ... every time you try to create something new, check if it
has been proven ... that should prevent duplicates"; the lesson: a Rust static exchange written beside the verified
exchange law it duplicated).

Parses every .lean file of the project (lean/ideas, not .lake): each theorem / lemma -- and each documented def, the
kernel's decision procedures -- with its file, line, docstring and statement; a proof still holding `sorry` is marked
OPEN. USED BY: the Rust and Python files that name it (code cites Lean results by name, e.g. Ideas.PathLemma.path_lemma).

    python -m catspace.incubator.analyst.lean_index build          # docs/LEAN_INDEX.md + derived lean_index.json
    python -m catspace.incubator.analyst.lean_index search exchange xray
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

from catspace.io import paths

ROOT = Path(__file__).resolve().parents[3]
LEAN = ROOT / "lean" / "ideas"
CODE_DIRS = [ROOT / "rust" / "analyst_core" / "src", ROOT / "catspace"]
SHELF_DIR = ROOT / "catspace" / "incubator" / "analyst" / "discovery" / "lemmas" / "ontology"      # plan 76.1: the ontology shelf (JSON)

DECL = re.compile(r"^\s*(?:@\[[^\]]*\]\s*)?(?:private\s+|protected\s+|noncomputable\s+)*(theorem|lemma|def|abbrev|structure|inductive)\s+([A-Za-z_][\w.'!?]*)")


def _decls(path: Path) -> list:
    text = path.read_text()
    lines = text.splitlines()
    ns = []
    out = []
    i = 0
    while i < len(lines):
        ln = lines[i]
        m_ns = re.match(r"^\s*namespace\s+(\S+)", ln)
        if m_ns:
            ns.append(m_ns.group(1))
        if re.match(r"^\s*end\s+(\S+)", ln) and ns and ln.split()[1] == ns[-1]:
            ns.pop()
        m = DECL.match(ln)
        if m:
            kind, name = m.group(1), m.group(2)
            # the docstring just above (/-- ... -/)
            doc = ""
            j = i - 1
            while j >= 0 and lines[j].strip() == "":
                j -= 1
            if j >= 0 and lines[j].rstrip().endswith("-/"):
                k = j
                while k >= 0 and "/--" not in lines[k]:
                    k -= 1
                if k >= 0:
                    doc = " ".join(l.strip() for l in lines[k:j + 1]).replace("/--", "").replace("-/", "").strip()
            # the statement: up to ':=' (or 'where' / '|' for defs), and the body up to the next declaration
            stmt = []
            k = i
            while k < len(lines):
                stmt.append(lines[k].strip())
                if ":=" in lines[k] or re.search(r"\bwhere\b", lines[k]):
                    break
                k += 1
            statement = " ".join(stmt).split(":=")[0].strip()
            body = []
            k += 1
            while k < len(lines) and not DECL.match(lines[k]) and not re.match(r"^\s*(end|namespace|section)\b", lines[k]):
                body.append(lines[k]); k += 1
            open_ = "sorry" in " ".join(stmt + body)
            if kind in ("theorem", "lemma") or doc:
                out.append(dict(kind=kind, name=name, full=".".join(ns + [name]), file=str(path.relative_to(ROOT)), line=i + 1,
                                doc=doc, statement=statement[:400], open=open_))
        i += 1
    return out


def _used_by(names: list) -> dict:
    """{name: [files naming it]} over the Rust and Python code (one grep per name, word-bounded)."""
    used = {}
    for n in names:
        parts = n.split(".")
        short = parts[-1]
        # a distinctive name (an underscore or a capital past the first letter, 10+ letters) is grepped alone; any other
        # only in its qualified form (Namespace.name) -- `targets` alone matched every file that says "targets"
        distinctive = len(short) >= 10 and ("_" in short or any(c.isupper() for c in short[1:]))
        pattern = short if distinctive else ".".join(parts[-2:])
        if len(pattern) < 6:
            continue
        try:
            r = subprocess.run(["grep", "-rlwF", "--include=*.rs", "--include=*.py", pattern] + [str(d) for d in CODE_DIRS],
                               capture_output=True, text=True, timeout=30)
            # plan 76.1 (a): the ontology shelf's entries bind Lean names (the names table's twins moved there from names.py): its JSON is read too
            r2 = subprocess.run(["grep", "-rlwF", "--include=*.json", pattern, str(SHELF_DIR)], capture_output=True, text=True, timeout=30)
            files = sorted({str(Path(f).relative_to(ROOT)) for f in (r.stdout + "\n" + r2.stdout).split() if "lean_index" not in f})
            if files:
                used[n] = files
        except Exception:
            pass
    return used


def build() -> list:
    files = sorted(p for p in LEAN.rglob("*.lean") if ".lake" not in p.parts)
    decls = [d for f in files for d in _decls(f)]
    used = _used_by([d["full"] for d in decls])
    for d in decls:
        d["used_by"] = used.get(d["full"], [])
    json.dump(decls, open(paths.derived("lean_index.json"), "w"), indent=1)
    thm = [d for d in decls if d["kind"] in ("theorem", "lemma")]
    lines = ["# The Lean index", "",
             "What we have proved -- the source of truth (Kaveh 2026-10-10: check here before building anything; prove before",
             "relying). Generated by `python -m catspace.incubator.analyst.lean_index build`; search with `... lean_index search <words>`.",
             "Do not edit by hand.", "",
             f"{len(thm)} theorems/lemmas ({sum(d['open'] for d in thm)} OPEN -- still `sorry`) and {len(decls) - len(thm)} documented "
             f"definitions (the kernel's decision procedures) in {len(files)} files.", ""]
    for f in files:
        ds = [d for d in decls if d["file"] == str(f.relative_to(ROOT))]
        if not ds:
            continue
        lines += [f"## {f.relative_to(ROOT)}", ""]
        for d in ds:
            tag = "OPEN " if d["open"] else ""
            first = d["doc"].split(". ")[0][:220] if d["doc"] else ""
            lines.append(f"- **{tag}{d['kind']} `{d['full']}`** (line {d['line']}){': ' + first if first else ''}")
            if d["used_by"]:
                lines.append(f"  - used by: {', '.join(d['used_by'][:6])}{' ...' if len(d['used_by']) > 6 else ''}")
        lines.append("")
    out = ROOT / "docs" / "LEAN_INDEX.md"
    out.write_text("\n".join(lines))
    print(f"OBS indexed {len(thm)} theorems/lemmas ({sum(d['open'] for d in thm)} open) and {len(decls) - len(thm)} documented defs "
          f"in {len(files)} files; {sum(1 for d in decls if d['used_by'])} cited by code -> {out.relative_to(ROOT)}")
    return decls


def search(words: list) -> list:
    """Declarations whose name, docstring or statement contain every word (case-insensitive), theorems first."""
    p = Path(paths.derived("lean_index.json"))
    decls = json.load(open(p)) if p.exists() else build()
    ws = [w.lower() for w in words]
    hits = [d for d in decls if all(w in (d["full"] + " " + d["doc"] + " " + d["statement"]).lower() for w in ws)]
    hits.sort(key=lambda d: (d["kind"] not in ("theorem", "lemma"), d["file"], d["line"]))
    return hits


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "search":
        for d in search(sys.argv[2:]):
            print(f"{'OPEN ' if d['open'] else ''}{d['kind']} {d['full']}  ({d['file']}:{d['line']})\n    {d['doc'][:200]}")
    else:
        build()
