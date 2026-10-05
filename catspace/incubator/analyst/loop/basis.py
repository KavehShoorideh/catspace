"""THE BASIS: the ideas the player may use, append-only. Each line is a JSON record -- op "promote" (an idea with its
tier and evidence) or "retire" (an earlier id withdrawn, by a rollback or a later round) -- chained by hash: a line's
`hash` covers the previous line's hash and its own payload, so rewriting any line breaks every hash after it and
`verify` refuses the file. A round records the chain head it started from; a rollback appends retire lines."""
from __future__ import annotations

import json
from pathlib import Path

from .store import canonical, digest, record_root

GENESIS = "0" * 64


class BasisError(Exception):
    pass


def path() -> Path:
    return record_root() / "basis.jsonl"


def read() -> list[dict]:
    p = path()
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def verify(lines: list[dict] | None = None) -> str:
    """Check the chain; -> the head hash. Raises BasisError on any rewrite."""
    lines = read() if lines is None else lines
    prev = GENESIS
    for i, e in enumerate(lines):
        payload = {k: v for k, v in e.items() if k not in ("hash", "prev")}
        if e.get("seq") != i or e.get("prev") != prev or e.get("hash") != digest({"prev": prev, "payload": payload}):
            raise BasisError(f"basis line {i} does not chain (rewritten or reordered): {canonical(e)[:120]}")
        prev = e["hash"]
    return prev


def head() -> str:
    return verify()


def _append(payloads: list[dict]) -> str:
    lines = read()
    prev = verify(lines)
    p = path()
    with open(p, "a") as fh:
        for k, payload in enumerate(payloads):
            payload = dict(payload, seq=len(lines) + k)
            h = digest({"prev": prev, "payload": payload})
            fh.write(canonical(dict(payload, prev=prev, hash=h)) + "\n")
            prev = h
    return prev


def promote(round_: int, records: list[dict], uses: dict | None = None) -> str:
    """Append promoted ideas (Promoted records as dicts) with their accepted uses (id -> [UseRecord dicts]).
    -> the new head."""
    uses = uses or {}
    return _append([dict(op="promote", round=round_, id=r["candidate"]["id"], tier=r["tier"], record=r,
                         uses=uses.get(r["candidate"]["id"], [])) for r in records])


def retire(round_: int, ids: list[str], reason: str) -> str:
    return _append([dict(op="retire", round=round_, id=i, reason=reason) for i in ids])


def active(at: str | None = None) -> dict:
    """id -> the latest promote record of every idea not retired after it. With `at` (a head hash), the basis as it
    stood at that head: a round reads the basis it started from, never today's (BUGS 2026-10-05)."""
    lines = read()
    if at is not None:
        if at == GENESIS:
            lines = []
        else:
            idx = next((i for i, e in enumerate(lines) if e.get("hash") == at), None)
            if idx is None:
                raise BasisError(f"head {at[:12]} is not in the basis")
            lines = lines[: idx + 1]
    out: dict = {}
    for e in lines:
        if e["op"] == "promote":
            out[e["id"]] = e
        elif e["op"] == "retire":
            out.pop(e["id"], None)
    return out
