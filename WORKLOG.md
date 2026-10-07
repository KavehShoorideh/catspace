# WORKLOG — the agent's journal (terse: what ran, what's next, open bugs)

Kaveh's two-journals rule (2026-09-16): this file keeps the agent on track; JOURNAL.md is Kaveh's, written for review.
Newest entry first. Times from `date` (PDT).

## 2026-10-06 18:02 PDT — PoC round 2: composition reach, the refutation veto, four rules

- Composition (solution, any reply, double threat >= 1): 1/16 quiet misses. Mate in 2/4: 0/30. Lines read: pins that
  win later, king attacks, races, deflections; several tops were ledger false positives (+5/+6 captures refuted).
- refute rule (veto, reads_ledger): opponent's certified double threat after each tied top; charges k - after0.
  Dev: refutation exists in 8/30 tops, ledger had charged 6 (checks are forcing moves in the frames), veto fired 2.
  Three wrong constructed test positions (diagonal check, Qd8+, Qg8+): tests now assert the position first.
- Held-out: four rules general 176.4 (32/6), pawns 23.7 (31/7). SF-2000 games running (no rules -> all four).
- NEXT (Kaveh): where depth comes from (composition has nothing to compose yet vs a fast proved generator); the
  games' verdict; a pin family with a provable invariant?

## 2026-10-06 17:09 PDT — PoC round 1: the double threat in Lean, certified in play

- Ideas/Fork.lean: localCheck (3 plies after m, captures as our answers, kernel rules) -> forcedAtLeaves; axioms std.
  First 2-ply statement certified Ng7+ for +3 (the recapture!): caught by the smoke, reply ply added. Binary: fork cmd.
- frame_player fork rule = Python proposes, Lean certifies (bisection), tier proved; fork_hint = uncertified proposals
  as a conjecture. OBS 200 dev positions: 105 proposals, 43 certified, 62 refused. Checker bridge vs python 4-ply
  enumeration: 105/105 agree (test). Held-out: general 164.1 -> 168.7 (certified) / 172.1 (hint); pawns 22.0 -> 22.8 /
  24.7. Both tiers: running (poc3). Games zero search: Maia-1500 95%; 1900 + random running.
- Both tiers (poc3) 174.5; + pin family (poc4) 174.6 (null); review fixes (answers = captures ∪ attacked ∪ king moves;
  hint max(claim, cert); cache; errors) -> poc5: general 176.2 (25/6), pawns 22.5 (15/7), dev 98.3. Cert 57/119.
  Maia-1900 92%, random 100%. Depth-5 kernel check: minutes/claim (killed). BUGS: 5 entries (2026-10-06).
- NEXT (Kaveh): composition vs faster generator vs wider proposers; Stockfish UCI_Elo 2000+ as the games bar.

## 2026-10-05 20:30 PDT — the PoC (Kaveh's scope; three standing rules)

- Scope confirmed: zero-search frame player + rolled-forward state + agent-written rules; off the shelf; reference +
  review per commit; one scope. Watcher/cases/viewer parked (committed, unused).
- Baselines (one_ply, verify clean): general 236/149/164.1 (zone 279/227/238.3); pawns 226/6/22.0 (249/48/76.8);
  backrank 300/289 both. Maia-1100 at zero search: 98% (20 games). Random-mover and Maia-1500/1900 runs killed (CPU).
- player --engine frame (d05f29b5, review -> 864557a5); frame_player.RULES hook (7aa6abff, review -> ec822612);
  passer rule (2 up/0 down); fork rule tactic_rules.py (bridge 105/0 wrong; AS IF PROVED: dev misses fixed 4/30, 0/40
  hurt; as registered -- a conjecture -- it was inert until the review: fixed 56acd6f6, see BUGS).
- Scars: zsh word-split repeat (usage not grepped); a test position with the king in check (my error, caught at once).
- NEXT (Kaveh): the Lean statement for the double threat with the exhaustive-reply hypothesis; heavy runs when CPU free.

## 2026-10-05 13:28 — watcher batch 2, health line

- Words: mate1_tomove, promo_tomove, promo_mover (tested). Rules F9 pinned, F10 in check (under F2), R11 mate in one,
  E12 promotes now, E13 quiet. One-move what-if left out (182 ms/position, weak signal).
- Health (bits saved/position vs base rates, same positions): frames +0.948 (b1), +1.005 (b2); elsewhere and result
  about even. `watcher health`, `watcher credit`.
- Scar x2 today: journal headers pre-written with a guessed time (13:22, 13:34) instead of `date` -> fixed commits.
- NEXT: the next review (batch 2's surprises); off-frame tactics need words (forks by quiet moves, discovered attacks).

## 2026-10-05 13:21 — the surprise watcher (Kaveh's design; decisions asked one by one)

- Pass 2 of the bug loop (reviewer #2 on my fixes): 10 more, fixed in 202c41f8 (18 loop tests). Old idea loop: round 1
  ROLLED BACK by Kaveh ("move on"); frame ties ON; compose per partner kept.
- Maia cache key now (weights, FEN, last 8 moves). A Maia transposition test deadlocked (lc0 + python at 0% CPU, 6 min):
  killed; not chased (no Maia in the watcher now: Kaveh "use existing games, anybody vs anybody").
- watch/: frames.py (live frames, attributes, see(), observe), rules.py (language, Ripple-Down tree, counts, ref,
  POLICIES slot), watcher.py (batch/show/add/replay/credit). 6 tests. Lichess shard 0, game_id % 97 == 7.
- Batches 0-1 + 8 rules (F1-F5, R6-R8). BUG: result counted per position -> per game; replayed.
- NEXT: elsewhere vocabulary (promotion threat, pins, discovered attacks, overload); Kaveh's call on frame counting
  per ply; batch 2 review.

## 2026-10-05 12:34 — bug loop pass 1 (Kaveh: fix bugs, ASK on design choices)

- Reviewer agent (read-only, 236k tokens, 17 min): 13 findings, fixed in 6b5575d6 with tests (16 loop tests). Mine
  too: Applier keeps one copy of a lemma found twice (exch4 54/5280; loop library none); dumps store the whole tie.
- Round 0's general arm replayed with its real wiring (loop000full); rounds 0 and 1 re-rendered (--render-only).
  Corrected: backrank 300/300/300 (the +2.5 was the mate-tie artefact); pawns 76.8/82.0/92.9; general
  238.3/239.8/241.0. JOURNAL 12:34 entry corrects the 11:55 one (banner added, old text untouched).
- MY ERRORS today (BUGS.md): "accuracy" was guard precision; "72/92 distinct", "a fifth/1% of votes" from reading
  Q as white (Q = either colour); the dig's step choice; claims before counting. Count, then narrate.
- Design switches left OFF pending Kaveh: frame ties (cfg frame.ties), per-game cap (frame.per_game). Kept, flagged:
  unspoken moves never demoted; a gate decision is final. Open: compose votes per partner vs per lemma.
- NEXT: pass-2 review of the 6b5575d6 diff; Kaveh's answers (round 1's fate, design switches, next build).

## 2026-10-05 (day) — the idea loop's framework (Opus, executing Fable's plan)

- Built catspace/incubator/analyst/loop: contracts, hash-chained append-only basis, six stages, ledger (4 controls),
  gate (continue/rollback/swap), driver (stage cache by input hash; refuses N+1 without gate N; swap needs a new
  module). Commit 4844bd3f. docs/LOOP.md. Tests: tests/test_analyst_loop.py (7, incl. the false idea -> RED).
- DEVIATION from the plan: data/derived is git-ignored, so the reviewable record AND the basis live in artifacts/loop
  (committed); bulky outputs stay in data/derived/loop. Rollback = retire lines (append-only), not a git revert:
  rounds commit only records, the wiring is data (active.json from the basis).
- lemma_gen --source fens --fens-file (the framed batch). 200 dev positions: run 58 s -> 91 lemmas; prove re-test
  (lichess test split, 6000 positions, cap 400) 12 s -> 39 survived. lemma_gen "PROVED" = held-out evidence, tier
  conjecture in the loop. Applier: 1 ms/position (fire scans and tie-breaks are cheap).
- frame_player.LOOP_ORDER: tie-break among still-tied top moves only; per-step `order` flag in dumps.
- Leakage check: first 150 dev rows (i%97==5, up to row 14458) share no row with the held-out sets.
- Round 0 (11:01): axioms clean; bridges backrank held-out 306/306, random 318/318; passer1 972/972; passer2
  random 1239/1239, held-out 112/112. Baseline re-runs identical to the frozen dumps (138 s, 16 s, 748 s). Gated
  continue by me (baseline only), 1a4ce48a; Kaveh to confirm.
- Round 1 (11:17-11:45, 28.1 min): 6/150 dev misses + 150 game positions (32 games) -> 199 lemmas -> 100 survive the
  re-test -> 103 promoted (3 Lean + 100 conjecture), 206 uses. Expected score (random pick among tied top moves):
  backrank 294.5 -> 297.0, pawns 82.0 -> 92.9, general 237.3 -> 241.0 vs round 0's arm. Ledger 04097b86.
- MY BUG + RETRACTION: first ledger flagged RESULTS FALL from tied-set membership (pawns "lost 82"); a tie-break can
  only shrink ties (BUGS class 10 mirror). I told Kaveh "249 -> 169" and "22 from a flaw in my code" (the flaw is
  lemma_apply.compose, the design). Push correction sent ~11:47. Fix f6fd8312 (expected_score, vs_previous, enrich),
  a7f51e14 (proposals), re-render from cache.
- loop/dig.py: classes the moved puzzles at the step that moved them. Trap: dumps keep best[:6]; use n_top for tie
  sizes (27 pawn / 16 general classified puzzles read a truncated list).
- Acceptance: round 2 REFUSED without gate 1 (exit 1). NEXT: Kaveh's gate on round 1; Fable's analysis; research
  slots R1-R6. Open: frame yield (6 misses/10.5 min), game clustering (cap per game), pawn step time 3.02x.

## 2026-10-05 — passer second form PROVED (2026-10-05 01:29 PDT)

- Ideas/Passer2.lean proved (no sorry; std axioms after replacing chess-lean's native_decide allSquares_nodup).
- Proof found: Python kingRoom skipped forbidden squares (unsound: capturable pieces); back-rank pawn gap (double
  step after one push). Lean vs Python agree 11734/11736. Bridge 832/832, 285/285, 109/109.
- Player on the Lean detector: 251/300, 53 single, = prototype run exactly. Pawn-ending baseline 249/300, 48 single.
- Lean traps added: docstring before set_option (3rd time!); `subst` eliminates the RHS var; `cases` on colour-
  mismatched piece equations; `rfl` cannot reduce `c == c` for a variable; `first | A | B` picks a trivially
  succeeding A; `simp_all` loses move equations -> extract coordinate equations first; a `cases` on a term does not
  rewrite inside hypotheses (revert first); omega needs the bounds of every Fin in scope (isLt).
- NEXT: explainer sentence for proved/one-away passers; middlegame lemma (enemy pieces, over-approximated reach).

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
- FIRE RATE v1 (bare king): 0 PROVED in 4,714 real positions (pawnEndgame/promotion/advancedPawn puzzles); the
  one-aways fail bareKing (the enemy has pawns). v2 PROTOTYPE in Python (lean_passer_v2_proto.py): enemy king+pawns
  with a pawn-distance bound to bad squares (path, our pawn, check squares, their promotion rank) and kingRoom (every
  square the enemy king can reach has more free unattacked neighbours than the enemy has pawns): pawnEndgame 2672
  positions -> PROVED 116, ONEAWAY 333 (kingOutside 110, pawnsFar 165, kingRoom 39, pathClear 10); race search
  confirmed 116/116; random pawn endings 400/400. Hypotheses empirically sound -> worth proving, IF it pays in play.
- Player on 300 held-out pawnEndgame puzzles (offset 17): plain 249/300 tied, 48/300 single, 2219 nodes/step, 16 s.
  --proved-passer (prototype: queen at the horizon + the push as a candidate): running.
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

## 2026-10-06 19:22 PDT — the cone form proved; setsUpPasser2 measured and wired (passer_setup)
- Passer2.lean: captureSquaresB, pawnFarB (cone form), Inv2.pawnsFar as a disjunction, enemy_move2 carries the captured
  piece, captureSquares_mono, five proof sites; build 10.7 s first try; axioms standard (4 theorems checked).
- Mirror (lean_passer_v2_proto.passer_v2) on the same pdist bound; 0 mismatches on 2,701 instances.
- OBS: held-out pawn endings PROVED 57 -> 62 (bridge 62/62), random 354 -> 360 (360/360); blocked: kingRoom 25,
  kingOutside 16, pawnsFar 10 (held-out).
- Setup.lean + Main.lean passersetup + LeanIdeas.passer_setup; measurement bug: one-shot arguments to a stdin-loop
  binary (0/185) -> through the wrapper 185/185 certified, 0 wrong; 17 held-out positions reached beyond the direct passer.
- Rule passer_setup (frame_player.RULE_BOOK); tests in test_analyst_lean_ideas / test_analyst_frame_rules.
- OBS held-out pawn endings expected: passer 22.2 (either kernel) -> +setup 23.1; bundle 23.7 -> +setup 24.6.
- test_data fixture collision fixed (pre-existing). Full suite green. SF-2000 phase 1: 20 games: 19 losses, 1 draw at the 200-ply cap (game 19 as white); score 2%; s/move 6.11 (the player's summary line).
- NEXT: kingRoom / kingOutside invariants (the blocked list); phase 2 games; Kaveh's view on a tempo tie-break
  between a queen now and a proved queen in three plies (equal credit today).

## 2026-10-06 19:33 PDT — kingRoom tightened (nearPawnsB): 62 -> 72 held-out proved pawns; play unchanged
- Prototype 10/25 rescued, bridge 10/10; Lean: nearPawnsB, Inv2.room, exists_not_pawn over G's pawns, filter_length_mono,
  filter_length_le_swap (Finset), both room sites; build 6.9 s; axioms standard; mirror 0 mismatches.
- OBS held-out PROVED 62 -> 72 (72/72), random 360 -> 366 (366/366); play arms 22.2/23.1/23.7/24.6 unchanged (0 up, 0 down).
- Test row g6 -> PROVED 3 plies (bridge in the test); new kingRoom ONEAWAY row 7k/4N1pp/P7/8/8/8/8/1K6 w a6.
- NEXT: dig which family the held-out misses belong to before another invariant; SF-2000 phase 2 result.
- 2026-10-06 19:37 PDT dig: pawn-ending misses are quiet pawn pushes 52/80; 5 certified set-ups refused by the race credit (right);
  only 9/52 one-away passers; any-pawn composition reaches 1 position. General misses: piece play, gap >= 2 in 29/64.
  NEXT family for pawn endings: king play (opposition/tempo/breakthrough), not passers; for the general set: the double threat.

## 2026-10-06 20:38 PDT — the measured answer written (JOURNAL); SF-2000 phase 2
- OBS SF-2000 with fork + fork_hint: 2 wins by mate (as White), 18 losses by mate; 10%; rules applied in 15/1,066 choices.
- Replay of the two wins (tmp/attribute_wins.py): rules spoke in 7/82 of our moves (6 tie-narrowings, 1 certified Be4
  skewer +5); replay reproduced every played move. Attribution of the wins open (Stockfish limited mode is random).
- Derived (tmp/derived.py): rules recover 17% of the general gap, 5% of pawn endings'.
- NEXT: Kaveh picks A (search only where rules are silent; recommended) / B (prior tie-break) / C (subgoals) / D (families).

## 2026-10-06 21:59 PDT — mate veto measured (no reach), the opening slide, opening subgoals started
- tmp/mate_reach.py, tmp/mate_sf_judge.py: 18/37 avoidable first allowances, all 18 lost already (SF depth 16). Not built.
- tmp/decisive_moves.py, tmp/eval_slide.py: decisive single move in 4/37; slide from move ~4; 32/36 crossings had a tied
  alternative >= -150; opening ties median 19.
- Bigger run partial (21:52): norules 2W/50L of 52; rules 0W/1D/66L of 67 -> the 10% was noise.
- Kaveh: opening subgoals as tie-breaks. opening_subgoals.py (castle > develop > central > centre pawn > steady),
  RULE_BOOK "opening" (conjecture), 8 tests pass; review agent running; short run tmp/openprobe_* (40 games x 20 plies).
- 2026-10-06 22:21 PDT opening rule reviewed (3 bugs fixed + docstring + RULE_ORDER), 13 tests; short run -71/-180 vs base -176/-356;
  WAC 60 20.6 -> 20.4. Launched sf2000x_open_s1..5 (full bundle + opening, 100 games). Stopped norules s1, s4.

## 2026-10-07 00:14 PDT — games at 100 a side done
- OBS: no rules 3.2% (78), fork+hint 2.0% (100), full bundle + opening 5.0% (100; Fisher p 0.37 wins). Opening medians
  -160/-364 -> -50/-164 (moves 5/10). Candidate losses: first fall below -150 at ply 16, tied alternative fine in 59/73.
- NEXT (Kaveh's pick): what breaks the middlegame ties -- search only among the ledger's ties / middlegame subgoals /
  a Maia prior.

## 2026-10-07 12:30 PDT — the tactic census built, reviewed, measured
- tactic_census.py (tokens from the existing detectors, null-move opponent tokens, structure, Stockfish d10), 300 games
  = 22,861 positions in 18 min; tactic_census_measure.py (episodes + bootstrap; ablations base/immediate/latent/structure).
- Smoke caught two false gates; the review caught three bugs (censoring, reference, check gate) -> fixed, rows re-tokenized.
- OBS gate: (a) PASS (back_rank 5.7x, discovery 4.3x, pin 2.0x ours; all five theirs); (b) FAIL as registered (latent
  +0.004); post hoc: no-eval baseline at 10 plies latent +0.013/+0.019 (CI > 0). One of two -> inspect; Kaveh's call.
