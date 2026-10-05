# WORKLOG — the agent's journal (terse: what ran, what's next, open bugs)

Kaveh's two-journals rule (2026-09-16): this file keeps the agent on track; JOURNAL.md is Kaveh's, written for review.
Newest entry first. Times from `date` (PDT).

## 2026-10-04 — Lean idea basis, lemma 1 (back-rank mate) end to end

- Plan doc: ~/Downloads/"Lean Chess — Idea Basis & Live Engine Handoff Plan.md". Kaveh: Lean in, for targets + pruning, never search.
- Toolchain: elan + leanprover/lean4:v4.32.0 (default). Kernel = Arthur742Ramos/chess-lean @3a59b2ce (Apache-2.0), dep of lean/ideas.
  First build ~4.5 min (20:35-20:40), incremental seconds. .lake is 7.8G (Mathlib cache) — git-ignored.
- Proved (std axioms only): Ideas.checkmate_of_unmeetable_check, Ideas.backRankMate_sound, Ideas.memoBoard_eq.
- Binary: lean/ideas/.lake/build/bin/ideas. PERF TRAP: the compiler folded the array copy back into every lookup
  (~100 ms a query); binding the array in readPosition fixed it (0.2-0.3 ms). Keep the array bound at the call site.
- Python: lean_ideas.py (client, per-process detector, rank_check_possible gate), lean_backrank_puzzles.py
  (bridge + exact mate search base/close/target). zone_search TacticalSpec.proved_mates (0/1/2, off by default),
  frame_puzzles --proved-mates --theme --tag. Tests: tests/test_analyst_lean_ideas.py (11).
- Dev rows: i % 97 == 5. Test rows: i % 97 == 17 (backRankMate). Do not tune on 17.
- Negative checked (not a bug): one-away targets point at the defender a move early; first-move coverage by "touch"
  alone 9/33 (mateIn3 dev) -> adding the one-away move itself 22/33. Ordering stays mixed: helps mateIn2, hurts mateIn3.
- Committed 9452b1e2 (no AI trailers). Held-out runs: tmp/heldout.sh, started 21:00.
- Held-out OBS so far: bridge 1316/1316, mates 306/306. Mate search CLOSE/BASE nodes 0.003 / 0.724 / 0.975 (mate in
  1/2/3), TARGET 0.003 / 0.630 / 1.032, audit failures 0, same answers. Wall time up (detector ~0.3 ms a query = ~8 nodes
  of the python-chess search: base mateIn3 0.80 s / 21003 nodes = 38 us a node). Player plain on 300 back-rank: 300/300 tied, 289/300 unique, 1616 nodes a step.
- Player (frame_puzzles, held-out back-rank, paired 658 steps): proved1 0.786, proved2 0.794, leaf-mates 0.983;
  same 300/300 & 289/300; wall 133/136/145/136 s. General control (stride 997 off 300): plain 30490 n/step 279/227
  762 s (= the pause entry exactly), proved1 0.987 paired, same answers, 846 s (+11%, pipe cost).
- Journal entry 21:37 written (newest-first block; 17:35 entry moved above the Oct 2 entries). BUGS.md 2026-10-04.
- NEXT: generic bounded certificate (any goal) proved adequate; forcing-tier back-rank mate in 2 (one-away + forced
  deflection); passer KPK; in-process detector (FFI) to remove the pipe cost.
