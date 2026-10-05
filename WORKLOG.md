# WORKLOG — the agent's journal (terse: what ran, what's next, open bugs)

Kaveh's two-journals rule (2026-09-16): this file keeps the agent on track; JOURNAL.md is Kaveh's, written for review.
Newest entry first. Times from `date` (PDT).

## 2026-10-04 (late) — detector economics (Kaveh: in-process detector first, then the passer)

- MEASURED (ping/parse commands): pipe round trip 10 us; FEN parse 90 us; hypotheses 70-230 us. FFI is NOT the fix.
- Bitboard input (python-chess bitboards in hex, 2.7 us to encode vs 19 us for a FEN) + @[noinline] on the array
  builders (the let-floating trap again) + evalUntil (stop at the 2nd failure, cheap hyps first):
  mate 186->94 us, two 325->218, midgame 320->68. Bridge: FEN and bitboard inputs agree 3157/3157 verdicts.
- Kernel `decide` CAN prove rayBetween = squaresBetween over 4096 pairs: 80 s (list equality, needs the walk from
  the lexicographically smaller end so the order matches allSquares) / 144 s (membership form). A `decide` that
  returns in 2 s with "false" stopped at the first counterexample -- not evidence of speed.
- Ideas/Fast.lean + FastMate.lean: fast primitives proved equal to the kernel's; BackRank hyps on them. Gain small
  (94->87 us): the loops were not the cost. In-binary `bench` found it: each hyp recomputed `after p ps`
  (applyMove, ~0.8 us) INSIDE its per-square lambda -- boxed applied the move 56x = 51 us. Fix: @[noinline] wrappers
  taking the after-position. Now mate 41 us, two 88, midgame 55, one-away 46 (was 186/325/320). Per fully evaluated
  candidate ~24 us (oneKing 6, noInterpose 7, legal 5 = three 64-ish scans through the Board closure, ~50-90 ns a
  lookup = the floor of this design). Pipe 9 us, Python encode 3, UInt64 bitboard parse 4. STOPPED tuning here.
- Python pre-gate rank_check_possible: bitboards only, 0.25-1.4 us (was a gives_check loop at every node).
- Reruns launched 00:16 (tmp/rerun.sh). Back-rank proved1: 136 s again (plain 133); paired nodes 836385 = 836385
  (verdicts unchanged). The detector was never the wall-time driver there (22.5k closures x 0.25 ms saved = 0.6 s
  wall on 10 workers). General proved1: 809 s (was 846; plain 762): +6% wall for -1.3% nodes; nodes identical to 2026-10-04 (verdicts unchanged).
  ECONOMICS CLOSED: the detector costs ~5% where it never fires; benefit is bounded by fire rate, not query cost.
- PASSER PROVED (01:0x PDT): Ideas/Passer.lean compiles with no sorry; unstoppablePasser_sound' and race depend
  only on propext/Classical.choice/Quot.sound. ~800 lines; the proof is an induction on remaining pushes with the
  Chebyshev invariant kdist(ek, promo) > r; the stalemate exclusion is the step-back argument (needs noSliders so a
  push never uncovers a check). Lean traps hit: `subst h` eliminates the RHS variable (lost y2); `cases` on
  `some ⟨opp c,king⟩ = some ⟨c,pawn⟩` tries to unify colours first (use simp at h); `rfl` cannot reduce `c == c` for a
  variable c (go through Bool.eq_iff_iff + simp); docstrings cannot precede set_option; `first | A | B` picks A if it
  merely succeeds (a `cases` on a trivially true equation).
- Passer v1 design (Ideas/Passer.lean): enemy = bare king; OUR side without sliders (a push could uncover a check and
  break the step-back argument); hyps ownPawn, pathClear, bareKing, noSliders, notInCheck, kingOutside, kingHasMove;
  goal: our queen on the promotion square within 2n-1 plies. Proof plan: induction on remaining pushes r with the
  invariant kdist(ek, promo) > r; E's legal moves = king steps (bare_king_move); E never in check (no sliders, pawn
  attack squares too close to promo); E's step-back square is free; the path stays clear (E too far).

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
