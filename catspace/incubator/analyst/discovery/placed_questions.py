"""PLAN 79.48 PARTS A, B, F (SYSTEM_SPEC 11 "The description is the source of the rules; fast code is only a certified copy"; piece: rules; ledger rows 44, 42): THE PLACED QUESTIONS OF LAYER 2, THEIR DEFINITIONS OVER THE
DESCRIPTION, AND THE LIST OF EVERY FAST ANSWER OF LAYER 2 AGAINST WHAT HOLDS IT.

THIS TEXT IS THE INTERFACE, written before the body. The module names no object of any game: a game is a name, its description is read from where the other modules of discovery/ read it; the relations of the definitions are
in the documents under rules/placed/ and nowhere in this source.

WHAT IT KEEPS.
  A. THE LIST. `questions()`: every question of the trait `Placed` of rust/engine/game/src/placed.rs, read from that source: its name, its signature, what it says in words (the doc comment), whether the trait gives a body
     (a provided method), and WHO ASKS IT ABOVE LAYER 2 (a count of call sites by crate, in each crate's src; a method of another type with the same name is counted too: the scan reads names, not types). `fast_answers()`:
     every answer of the kernel that layer 2 gives, read from the sources that hold them (kernel.rs: the kernel's `Game` answers; outside.rs: the placed reading, the text door, the state as facts, the notation of a move).
  B. THE DEFINITIONS. The placed questions are said over the description in ADDITIVE TEXTS, one document each under rules/placed/ (the form of the lemma documents: a JSON object whose `delta` is a list of clauses;
     `rule_lemma.compose` composes it on a description): `base.json` for every game and `<game>.json` where a game's description says a relation another way (a delta of the game that retracts a clause of the base one).
     `composed_text(game)` is the description of the game with the documents composed on it: the text the translator is given as it is.
  F. THE GATE'S LIST. `held_by(item)` says what holds a fast answer: a CERTIFICATE (certify.py), a RECORD of the kernel against the translated module (kernel_translated.py, the five answers and the state as facts), a RECORD
     of the placed questions (placed_translated.py), or an EXCEPTION with a named cause (`EXCEPTIONS`, short, each a cause in words). `gate_rows()` lists every fast answer with its holder; the test fails on an answer whose holder
     is none, or whose record or certificate is not fresh.

INTERFACE (who calls it: tests/test_plan79_48.py, scripts/gdl_to_rust_make.py, discovery/placed_translated.py, `python -m catspace.incubator.analyst.discovery.placed_questions list`):
  `documents(game) -> list[Path]`       the documents of rules/placed/ that apply to the game, base first
  `delta(game) -> list[str]`            the clauses of those documents, in order
  `composed_text(game) -> str`          the description of the game composed with them
  `question_documents() -> list[dict]`  the documents as read (their `questions` say which relation answers which method of the trait)
  `questions() -> list[dict]`           part A
  `fast_answers() -> list[dict]`        every answer of layer 2 in kernel.rs and outside.rs
  `held_by(item) -> dict`               what holds it: {"kind": "certificate" | "record" | "placed" | "exception", "what": ..., "cause": ...}
  `gate_rows() -> list[dict]`           every fast answer with its holder
Nothing here is on the engine's path.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from catspace.incubator.analyst.discovery import certify as CF
from catspace.incubator.analyst.discovery import gdl_to_rust as T
from catspace.incubator.analyst.discovery import rule_lemma as RL

ROOT = CF.ROOT
ENGINE = CF.ENGINE
DOCS = ROOT / "rules" / "placed"
PLACED_RS = ENGINE / "game" / "src" / "placed.rs"
OUTSIDE_RS = ENGINE / "game" / "src" / "outside.rs"
KERNEL_RS = ENGINE / "game" / "src" / "kernel.rs"
BASE_DESCRIPTION = CF.GC.SPEC
VARIANTS = ROOT / "rules" / "variants"


# ------------------------------------------------------------------------------------------------------------------------------------------ B: the definitions

def description_paths(game: str) -> list:
    """The description of a game as the files the translator composes: the base text, then the game's delta (the first game has none)."""
    delta = VARIANTS / f"{game}.gdl"
    return [BASE_DESCRIPTION] + ([delta] if delta.exists() else [])


def documents(game: str) -> list:
    out = [DOCS / "base.json"]
    own = DOCS / f"{game}.json"
    if own.exists():
        out.append(own)
    return out


def question_documents() -> list:
    return [json.loads(p.read_text()) for p in sorted(DOCS.glob("*.json")) if p.name != "held.json"]


def delta(game: str) -> list:
    out = []
    for p in documents(game):
        out.extend(json.loads(p.read_text())["delta"])
    return out


def composed_text(game: str) -> str:
    """The description of the game with the additive texts of the placed questions composed on it (one document at a time, in order: a game's own document may retract a clause of the base one)."""
    paths = description_paths(game)
    text = T.compose_text(paths) if len(paths) > 1 else paths[0].read_text()
    for p in documents(game):
        text = RL.compose(text, json.loads(p.read_text())["delta"])
    return text


def relations_of_questions() -> dict:
    """The relation of the definitions that answers each method of the trait: {method: [relation, ...]} (from the `asks` of the documents)."""
    out: dict = {}
    for doc in question_documents():
        for q in doc.get("questions", []):
            for m in [x.strip() for x in q["asks"].split(",")]:
                out.setdefault(m, [])
                if q["relation"] not in out[m]:
                    out[m].append(q["relation"])
    return out


# ------------------------------------------------------------------------------------------------------------------------------------------ reading the Rust sources

def _strip_comments(text: str) -> str:
    return "\n".join(ln.split("//")[0] for ln in text.splitlines())


def _block(text: str, start: int) -> str:
    """The text of the braces that open at or after `start` (the first `{`), to its match."""
    i = text.index("{", start)
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1:j]
    raise ValueError("unbalanced braces")


def _fn_at_depth0(body: str) -> list:
    """The `fn` items at the top level of a block of an impl or a trait: (name, text of the signature, has a body)."""
    out, depth, i = [], 0, 0
    while i < len(body):
        c = body[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
        elif depth == 0:
            m = re.match(r"(?:pub\s+)?fn\s+([a-z_0-9]+)", body[i:])
            if m and (i == 0 or not (body[i - 1].isalnum() or body[i - 1] == "_")):
                end = i
                d2 = 0
                while end < len(body):
                    if body[end] == "(":
                        d2 += 1
                    elif body[end] == ")":
                        d2 -= 1
                    elif d2 == 0 and body[end] in "{;":
                        break
                    end += 1
                out.append((m.group(1), " ".join(body[i:end].split()), end < len(body) and body[end] == "{"))
                i = end
                continue
        i += 1
    return out


def _docs_before(raw: str, name: str) -> str:
    """The doc comment (the `///` lines) right above `fn <name>` in the raw source."""
    lines = raw.splitlines()
    for k, ln in enumerate(lines):
        if re.match(rf"\s*(pub\s+)?fn\s+{name}\b", ln):
            doc = []
            j = k - 1
            while j >= 0 and lines[j].strip().startswith(("///", "#[")):
                if lines[j].strip().startswith("///"):
                    doc.insert(0, lines[j].strip()[3:].strip())
                j -= 1
            return " ".join(doc)
    return ""


def callers(name: str) -> dict:
    """The call sites of a method by name in every crate's src: {crate: count}, the definitions (`fn name`) and the file that defines it left out."""
    pat = re.compile(rf"(?:\.|::){name}\s*\(")
    out: dict = {}
    for p in sorted(ENGINE.glob("*/src/**/*.rs")):
        if p in (PLACED_RS,):
            continue
        t = _strip_comments(p.read_text())
        n = len(pat.findall(t))
        if n:
            crate = p.relative_to(ENGINE).parts[0]
            out[crate] = out.get(crate, 0) + n
    return out


def test_callers(name: str) -> int:
    pat = re.compile(rf"(?:\.|::){name}\s*\(")
    return sum(len(pat.findall(_strip_comments(p.read_text()))) for p in sorted(ENGINE.glob("*/tests/**/*.rs")) if "generated" not in p.parts)


def questions() -> list:
    """Part A: every method of the trait `Placed`, from its source."""
    raw = PLACED_RS.read_text()
    code = _strip_comments(raw)
    m = re.search(r"pub trait Placed\b", code)
    body = _block(code, m.start())
    out = []
    for name, sig, provided in _fn_at_depth0(body):
        out.append({"name": name, "signature": sig.replace("pub fn", "fn"), "says": _docs_before(raw, name), "provided": provided, "callers": callers(name), "test_callers": test_callers(name)})
    return out


def _impl_items(path: Path) -> list:
    """The fast answers a source gives for the kernel: the fns of every `impl ... KernelGame` block and the free public fns, as {file, owner, name, public}."""
    raw = path.read_text()
    code = _strip_comments(raw)
    out = []
    for m in re.finditer(r"^impl\b([^{]*)\{", code, flags=re.M):
        head = " ".join(m.group(1).split())
        if "KernelGame" not in head:
            continue
        body = _block(code, m.start())
        owner = head.replace("for KernelGame", "").replace("KernelGame", "").strip() or "KernelGame"
        owner = re.sub(r"^<[^>]*>\s*", "", owner)
        for name, sig, _prov in _fn_at_depth0(body):
            out.append({"file": path.name, "owner": owner if "for KernelGame" in head else "KernelGame", "trait": owner if "for KernelGame" in head else "", "name": name, "public": sig.startswith("pub ")})
    for m in re.finditer(r"^pub fn ([a-z_0-9]+)", code, flags=re.M):
        out.append({"file": path.name, "owner": "", "trait": "", "name": m.group(1), "public": True})
    return out


def fast_answers() -> list:
    """Every answer layer 2's kernel gives, from the two sources that hold them (kernel.rs: the rules' answers; outside.rs: their placed reading, notation and bridges)."""
    return _impl_items(KERNEL_RS) + _impl_items(OUTSIDE_RS)


# ------------------------------------------------------------------------------------------------------------------------------------------ F: what holds a fast answer

RECORD = "record"          # the record of the kernel against the translated module (kernel_translated.py): the five answers and the state as facts
CERT = "certificate"       # the twelve certificates (certify.py)
PLACED = "placed"          # the record of the placed questions (placed_translated.py)

def _held() -> dict:
    return json.loads((DOCS / "held.json").read_text())


# The placed methods of the trait `Placed` that the definitions of rules/placed/ answer: held by the record of the placed questions.
HELD_BY_PLACED = {"place_names", "kind_names", "cells", "bearing_from", "reach_from", "move_ends", "kind_at", "find_step", "move_info", "strikes_place"}


def _key(item: dict) -> str:
    return f"{item['trait'] or item['owner']}::{item['name']}"


def held_by(item: dict) -> dict:
    key, held = _key(item), _held()
    if item["trait"] == "Placed" and item["name"] in HELD_BY_PLACED:
        return {"kind": PLACED, "what": "the placed record: " + ", ".join(relations_of_questions().get(item["name"], [item["name"]]))}
    if key in held["record"]:
        return {"kind": RECORD, "what": held["record"][key]}
    if key in held["exceptions"]:
        return {"kind": "exception", "cause": held["exceptions"][key]}
    return {"kind": "none"}


def gate_rows() -> list:
    return [dict(item, held=held_by(item)) for item in fast_answers()]


# ------------------------------------------------------------------------------------------------------------------------------------------ the printing

def obs_lines() -> list:
    out = []
    qs = questions()
    for q in qs:
        c = q["callers"]
        above = {k: v for k, v in c.items() if k != "game"}
        out.append(f"OBS 79.48 A {q['name']}: {q['signature']} | {q['says'][:160]} | {'provided' if q['provided'] else 'required'} | above layer 2: {sum(above.values())} call sites {above or '-'}; in layer 2 (game): {c.get('game', 0)}; tests: {q['test_callers']}")
    rows = gate_rows()
    for r in rows:
        h = r["held"]
        out.append(f"OBS 79.48 A fast {r['file']}: {r['trait'] or r['owner'] or 'fn'}::{r['name']} -> {h['kind']}: {h.get('what') or h.get('cause', '')}"[:420])
    kinds: dict = {}
    for r in rows:
        kinds[r["held"]["kind"]] = kinds.get(r["held"]["kind"], 0) + 1
    out.append(f"OBS 79.48 A totals: {len(qs)} methods of the trait, {len(rows)} fast answers in kernel.rs and outside.rs, held by {kinds}")
    return out


def main(argv: list) -> int:
    cmd = argv[1] if len(argv) > 1 else "list"
    if cmd == "list":
        print("\n".join(obs_lines()))
        return 0
    if cmd == "text":
        print(composed_text(argv[2]))
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
