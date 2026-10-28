# agenda_audit -- plan 75.1's classifier (read-only tooling)

The scripts that produced docs/AGENDA_AUDIT.md (2026-10-27): `extract.py` walks agenda.py / ponder.py into atoms (constants, branches, comparisons, orderings,
literals, piece-kind uses); `classify.py` assigns each row one of LOGIC / RULE / NAME / CHESS FACT / HEURISTIC and a store entry (existing or proposed);
`facts.py` groups the CHESS FACT rows into families; `count.py` prints the OBS lines; `snap.py` / `sigtest.py` count the guard cases whose trace passes
through a family; `gen.py` writes the document. Kept so plan 75.2 can RECOUNT the CHESS FACT rows before and after each stage (the number to drive to zero).
Run from the repo root with the project venv; outputs are written under artifacts/experiments/ (untracked).
