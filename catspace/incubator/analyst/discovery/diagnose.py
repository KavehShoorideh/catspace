"""PLAN 78.17 DIAGNOSE: the engine's game-free self-diagnosis over a wake log (JSON Lines: `position` rows with `steps`, `game` rows). It names what the engine lacks, in plain
quantities, and reads nothing of the game:

  tie rate         the share of decisions whose ledger step read ties (`candidates` > 1), and the mean count. CAUTION: the controller reads the ledger's ties ONLY when the
                   wake ran with the prior on (`wake --prior`); otherwise `candidates` is 0 everywhere and `ties_read` says so (0 decisions asked), not "no ties".
  surprises        by kind: an unforeseen loss (the second side sees units lost that the first did not foresee), a proved line (the second side proves a forced win after
                   the first's move), a teacher disagreement (only when the log's rows carry one: `kind: "surprise"` rows with `surprise: "teacher"`; the sleep reports it).
  cost             seconds and nodes per operator and per part (the parts an operator names its time under, grouped by the first word of the part name: `read g3g4` and `read d2c2` are the part `read`).
  depth            the distribution of the ledger step's depth.
  gap share        the share of decisions whose ledger step ended `Gap` (the budget cut the reading: unknown, never a refutation).

  .venv/bin/python -m catspace.incubator.analyst.discovery.diagnose <wake.jsonl> [--min-units 1]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter

from catspace.incubator.analyst.discovery import sleep as SL


def _rows(path: str) -> list:
    return [json.loads(x) for x in open(path) if x.strip()]


def diagnose(path: str, min_units: int = 1, say=print) -> dict:
    """-> the report dict of the wake log at `path`; prints OBS lines."""
    rows = _rows(path)
    pos = [r for r in rows if r.get("kind") == "position" and not r.get("error")]
    ledgers = [next((s for s in r["steps"] if s["operator"] == "ledger"), None) for r in pos]
    read = [l for l in ledgers if l is not None]
    n = len(pos)
    counts = [l.get("candidates", 0) or 0 for l in read]
    ties = {"decisions": len(read), "ties_read": sum(1 for c in counts if c > 0), "with_ties": sum(1 for c in counts if c > 1),
            "rate": (sum(1 for c in counts if c > 1) / len(counts)) if counts else None,
            "mean_count": (sum(c for c in counts if c > 0) / max(1, sum(1 for c in counts if c > 0))) if counts else None}
    ties["asked"] = ties["ties_read"] > 0
    cost_op: dict = {}
    cost_part: dict = {}
    for r in pos:
        for s in r["steps"]:
            o = cost_op.setdefault(s["operator"], {"steps": 0, "seconds": 0.0, "nodes": 0})
            o["steps"] += 1
            o["seconds"] += s.get("seconds") or 0.0
            o["nodes"] += s.get("nodes") or 0
            for p in s.get("parts", []):
                q = cost_part.setdefault(f"{s['operator']}.{p['name'].split()[0]}", {"seconds": 0.0, "nodes": 0})
                q["seconds"] += p.get("seconds") or 0.0
                q["nodes"] += p.get("nodes") or 0
    depth = dict(sorted(Counter(l["depth"] for l in ledgers if l is not None and l.get("depth") is not None).items()))
    gaps = sum(1 for l in ledgers if l is not None and l["status"] == "Gap")
    decisions, games = SL.read_log(path)
    sur = SL.surprises(decisions, games, min_units)
    by_kind = Counter("proved line" if decisions[(s["run"], s["game"], s["ply"] + 1)]["proved"] else "unforeseen loss" for s in sur)
    teacher = sum(1 for r in rows if r.get("kind") == "surprise" and r.get("surprise") == "teacher")
    kinds = {"unforeseen loss": by_kind.get("unforeseen loss", 0), "proved line": by_kind.get("proved line", 0), "teacher disagreement": teacher}
    rep = {"log": path, "games": len(games), "decisions": n, "ledger_steps": len(read), "ties": ties, "surprises": kinds, "cost_by_operator": cost_op, "cost_by_part": cost_part,
           "ledger_depth": depth, "gap": {"ledger_steps": gaps, "share": (gaps / len(read)) if read else None}}
    say(f"OBS diagnose {path}: {len(games)} games, {n} decisions, {len(read)} with a ledger step")
    say(f"OBS diagnose ties: ties read on {ties['ties_read']} of {ties['decisions']} decisions" + ("" if ties["asked"] else " (none: the wake did not ask; run it with --prior)")
        + f"; more than one candidate on {ties['with_ties']} (rate {ties['rate']}); mean candidates where read {ties['mean_count']}")
    say("OBS diagnose surprises: " + ", ".join(f"{k} {v}" for k, v in kinds.items()))
    for k, v in sorted(cost_op.items(), key=lambda kv: -kv[1]["seconds"]):
        say(f"OBS diagnose cost operator {k}: {v['steps']} steps, {v['seconds']:.3f} s, {v['nodes']} nodes")
    for k, v in sorted(cost_part.items(), key=lambda kv: -kv[1]["seconds"]):
        say(f"OBS diagnose cost part {k}: {v['seconds']:.3f} s, {v['nodes']} nodes")
    say("OBS diagnose ledger depth: " + ", ".join(f"{d}: {c}" for d, c in depth.items()))
    say(f"OBS diagnose gap: {gaps} of {len(read)} ledger steps (share {rep['gap']['share']})")
    return rep


def main(argv: list) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("log")
    ap.add_argument("--min-units", type=int, default=1)
    a = ap.parse_args(argv)
    diagnose(a.log, a.min_units)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
