"""Append ONE dated entry to JOURNAL.md (append only; never rewrite)."""
import sys

J = sys.argv[1]
STAMP = sys.argv[2]
obs = [l.strip() for l in open("/tmp/audit751/gen.log") if l.startswith("OBS ")]
top = [l.strip() for l in open("/tmp/audit751/count.log") if l.startswith("OBS audit fact ")][:10]
entry = f"""

## {STAMP} -- plan 75.1 THE AGENDA AUDIT (read-only): docs/AGENDA_AUDIT.md
Every branch, constant and comparison of discovery/agenda.py (the agenda, the chase, the tiers driver `_ask_tiers`, the pool driver `_ask_pool`, the judge -- all in agenda.py)
and ponder.py, classified LOGIC / RULE / NAME / CHESS FACT / HEURISTIC by an AST extractor and a rule table (scratch scripts, not committed; the rules are stated in the doc's
section 1 and 3). One row per line, class and fact family; each CHESS FACT row names its store entry (existing, or proposed: name, kind, reads) and how many guard cases pass
through it. No engine file, test or other doc changed; no gate, no snapshot, no timed run (the prover agent was running beside).
{chr(10).join(obs)}
Decisions depending, read from the 73.5b guard snapshots (guard_after/, ten + wide, tiers 0 and 2, count budget; agenda.py unchanged since 635990e9): the top ten families by cases
passing through (directional: 'passes through' is an upper bound of 'depends on'; the count budget never runs the status driver, the pool or the judge):
{chr(10).join(top)}
Reading (directional): the goal kinds and the stakes (mate 100, unit worth, the gain unit 1) are the most-travelled chess facts in agenda.py, so 75.2 (a) moves the most;
the stops (b) are few rows but the fuse's shape, tempo arithmetic and the second check pass through most cases. Findings for 75.2 (the doc's section 5): castling's rook,
the repetition key and the exchange's swap list are rules re-implemented in Python; check and capture are read off the SAN for the judge; lemma names and the forcing words are
tested as strings; material is supplied by the Game and BaseWords interfaces although plan 75 calls it derived; the stake of mate is defined three times; ponder.resume_gap files
every resumed route under a 'win' question.
"""
with open(J, "a") as fh:
    fh.write(entry)
print(entry)
