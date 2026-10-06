# BUGS.md -- the agent's own bug log

Kaveh 2026-09-29: "Always check your negative results for bugs. Focus on places with high probability of errors based
on your past knowledge of yourself. Store bugs in a bugs file for your own reference."

Every bug found in agent-written code goes here: what it was, how it showed (or did not), how it was found, the fix,
and the CLASS it belongs to. The classes are the checklist: before reporting a negative result (a null, a "does not
help", a plateau), check the result's code against every class below that applies, most frequent first. The companion
checks for plausible-but-wrong positives are in docs/PROMPTS.md.

## Where I make errors (the checklist, most frequent first)

1. **Cache and key soundness** -- a key that does not record something the cached computation read. Symptom: results
   depend on the order positions are met; nothing crashes. Check: run with the verify switch
   (ANALYST_VERIFY_FRAMES=1) until clean; replay one failing case alone vs in a batch.
2. **Silent skips** -- `except Exception: continue`, a filter that drops rows, a `None` treated as "no signal". Check:
   count what was skipped and print it beside the result.
3. **Side / perspective / parity** -- side to move, owner vs mover, colour-relative features, odd vs even plies.
   Check: a mirrored position gives the mirrored answer; lag 1 and lag 2 both shown.
4. **Off-by-one in time and horizon** -- "within k plies", "t - 1 moves when they move second", search depth vs plies.
   Check: hand-trace one example at the boundary.
5. **Horizon and search artefacts misread as facts** -- a value at depth d taken as the truth; a budget that stopped
   one candidate and not another. Check: the same case at a larger budget or depth.
6. **A claim not tested** -- a mechanism credited with a fix it did not make. Check: remove the mechanism and rerun
   the example before writing the explanation.
7. **Process and file hygiene** -- a detached `&` job, two writers to one file, a stale dump overwritten or compared
   to the wrong run. Check: every long job through the tracked background runner; `pgrep` before relaunching;
   baselines copied under their own name before a rerun.
8. **Wrong baseline quoted** -- a number from an older run than the one compared. Check: every baseline number in a
   write-up traced to the printed line it came from.
9. **Test assumptions** -- a test expecting what I believed the code did. Check: when a new test fails, decide first
   whether the code or the expectation is wrong, with the actual values printed.
10. **A hit that ties count** -- "the answer is among the top" when the top is a tie of many: a metric a flat
   evaluation passes. Check: print the tie sizes beside the hit rate, and count the answer ALONE on top. The mirror
   form: a TIE-BREAK judged by "among the tied top" can only lose (it only removes moves from the tie) and by "alone
   on top" can only win; judge it by the expected score of a random pick among the tied top moves.

## Log

### 2026-09-28/29 (session 72b40b8f)
- **Frame store keys leaked four ways** (class 1). `fates.frame_key` keyed a frame locally while its resolution read
  squares outside: a capture that gives check (every evasion follows -- Kxc5 took a pawn off the frame), a promotion
  push (d8=Q stood unanswered), en passant landing off the frame, a capturer behind a member (a rook behind a bishop).
  Silent; headline counts unchanged, but results order-dependent. Found by verify mode over the 150 quiet puzzles, one
  per run. Fixed: `can_reach_check`, `near_promotion`, e.p., `potential_capturers` in the key. Also a real resolver
  bug found on the way: a promotion's landing square did not join the frame (the reply is on its square).
- **Zone search stand-pat read one move deep** (class 5): a pass was scored by what the opponent wins at once, so a
  two-move threat (a pawn two pushes from queening) never showed. Found by Stockfish disagreeing on 0M4g4 (Nac5 0 vs
  -7.2). Fixed: the pass is searched (null-move reduction).
- **Zone search leaf in check had the wrong sign** (class 3): returned minus what the side in check could win. Found
  reading the code while fixing the above. Fixed: check extension.
- **Zone search false mate** (class 1-like, a boundary leak): Kh8 read +1000 in 0ROfv because the defence Rf2 bore on
  h2, outside the zone. Found by the Stockfish dig of quiet misses. Fixed: the pass's threat joins the zone.
- **Shared budget held all candidates at 6 plies** (class 5): 27 candidates, one budget; the sacrifice needing 8 plies
  read -1. Found on 0Hb6O by searching it alone. Fixed: per-candidate budgets.
- **Check exemption credited with a fix it did not make** (class 6): I explained the 0K3Ll miss as the margin pruning
  a checking sacrifice and added an exemption; removing it changed nothing. Retracted in the journal.
- **Wrong baseline in the journal** (class 8): wrote 2000+ puzzles "was 34/58"; the compared run had 38/58. Found
  rereading the journal before commit.
- **A goal frame's job counted only our gain** (class 3): a member's cost ignored its own side's lost threat (the
  rook's threat on b5 read as "no job"). Fixed: our best frame's rise plus theirs's fall.
- **Two writers to one gzip file** (class 7): a shell `&` launch I thought had died was running; the tracked relaunch
  wrote the same file; the gauge read a corrupt stream. Found by the gauge crashing. Fixed: killed the stray, reran.
- **Test assumed a footprint** (class 9): expected the rook's reach at k=1 not to read a7; it reads the 7th rank from
  d7 (attacks after one move). The code was right.
- **Canonical order was not canonical** (class 1): `lemma_key` broke ties among equally-labelled slots by square
  number, so the same frame on mirrored squares got different keys and types (inflating the frame-type vocabulary
  behind a negative result; costing the lemma store reuse, not soundness). Found by the colour-mirror check
  (`frame_profile check`, 16 of 388 positions). Fixed: the smallest key over the orders of each tied class (up to 720;
  beyond, square order and the key marked "nc").
- **King joins depended on colour order** (class 3): `joint_frame` judged Black's king on a frame already enlarged by
  White's king frame -- order-dependent, and a cascade against the one-step rule. Found by the same mirror check
  (all 13 remaining differences). Fixed: both kings judged on the same base frame.
- **One fixed C for feature sets on different scales** (class 5-like, a modelling artefact): the frame-profile gauge
  used C=0.5 everywhere; adding atoms to board+material LOWERED the mate AUC 0.854 -> 0.814, the signature of a
  mis-set model, not of chess. Found checking the negative result per Kaveh's rule. Fixed: MaxAbsScaler and C chosen
  per feature set on held-in games.
- **"Arrived" was true by definition** (class 4/6, a definition error): classifying how a mate's first check came, I
  called the checking piece "arrived" when its start square differed from its checking square -- true for every
  check but a discovered one (104 of 117 "arrived"). Found because the number looked too clean. Fixed: arrived = it
  needed 2+ of its own moves (the check itself is one).
- **Capture landings read as unreachable** (class 2-like, a silent None): the arrival distance used the reach map's
  `stand`, which holds empty squares only; most mating checks are captures, so 65 of 117 read "not within 4 moves"
  for mates within 4 plies. Found by the same check. Fixed: an occupied square costs its attack distance plus one.
- **A latent line dropped when an open route existed** (class 6, a definition gap): with no horizon every rook has
  some open route to check, and the watch took it and skipped the rook's latent line -- losing Kaveh's
  three-blocker file. Found by the unit test written from his example. Fixed: a piece can be an open and a latent
  member; the answers keyed by member, not square.
- **Not a bug, a noise source** (class 5): the zone search's transposition table is shared by all candidates of a
  position, so a candidate's depth and value depend on which other candidates were searched (2WXJR: Qf3 alone 0.0 at 6
  plies, among all 27 candidates 1.0 at 8). Benchmark churn of a few puzzles between two settings that change zones
  (king ring, watch zone) is this and budget/horizon effects -- check the lost puzzles' depths before reading a
  difference.
- **A policy gate that was always on** (class 2, a degenerate gate): the placeholder budget policy marked a candidate
  "king danger" if it touched a king watch's squares; with no horizon those cover most of the board -- 27 of 27, 16
  of 16, 22 of 23 candidates -- so it reallocated nothing and only raised every cap (twice the time, no gain). Found
  checking a mixed result (print the gate's firing rate beside any gated result). The second criterion (danger = the
  move CHANGES a king watch) fires on 12-28 of 12-31 candidates too: with no horizon every enemy piece is a member and
  nearly every move changes some member's distance. Not fixable by a threshold rule -- the budget policy has to be
  learned; flag stays off.
- **A two-phase policy that spent its whole budget in phase 1** (class 5, a protocol flaw): the learned budget
  policy's phase 1 searched every candidate to 8 plies -- about the fixed search's whole cost -- so at equal total
  nodes phase 2 had nothing (0Knjg: phase 1 spent 6,534 of a 6,533 cap) and it scored 3/14 against 8/14, with
  nonsense values on easy cases (a mate read 0.0). Found checking the negative result. Fixed: phase 1 to 6 plies,
  the model trained with the 8-ply features masked.
- **A multi-hour chain launched as a tracked background task** (class 7): the runner stops tasks after 30 minutes, so
  the overnight collect-train-benchmark chain died a third of the way into collection (95 of 300 puzzles). Fixed:
  `search_policy collect --resume` keeps whole puzzles and appends; anything over ~25 minutes runs under `nohup` with
  its PID recorded and a re-armed Monitor on its output.
- **Overwrote a tracked file with a new module** (class 7; the SECOND time -- plans.py on 2026-09-28): wrote the new
  planner module to `planner.py`, which already held Kaveh's 2026-09-11 planning layer; the write tool said "updated",
  not "created". Found at once from that word; restored with `git checkout`, the new module moved to
  `frame_planner.py`. Check: `git ls-files --error-unmatch <path>` before writing any "new" file.
- **Mates dropped out of the ladder's decisions** (class 2, a silent empty): the zone search scores a candidate that
  mates at once and never searches it, so its per-depth history is empty; `depth_ladder.decisions` read only
  histories, so in 7 of 23 pilot positions the mating solution vanished and "the solution is never on top, even at
  12 plies" read 8/23 instead of 1/23. Found checking the result. Fixed: an empty history is a mate.
- **A harness with its own candidate set** (class 9-like, a mismatch): `search_policy._candidates` (and the ladder
  through it) lacked the player's quiet threats (00Ygn's Nc4 was never a candidate). Fixed: one function,
  `frame_player.candidates`, for the player and every harness.

- **Ties counted as hits in the depth ladder** (class 10): "the solution among the top" FELL with depth on the 209
  missed puzzles (45, 34, 27 at 4, 6, 8 plies) -- read at first as deeper search losing the solution; it was shallow
  searches leaving 2+ moves level (3+ in 85 positions at 4 plies) and ties breaking as depth grew. Found because a
  quantity that should rise with depth fell. Fixed: `depth_ladder report` counts the solution alone on top (10, 12,
  16, 43, 53) and prints the tie sizes.
- **A negative that was half unmeasured** (class 5): "112 never alone on top within 12 plies" -- 56 of them had the
  solution's own search stop short of 12 by budget. Fixed: the report splits "never" into stopped-short / tied / below.
- **A comparison set misnamed** (class 8, caught in the draft): the journal draft called the Stockfish depth study
  "the same misses"; it ran on the 9 misses of the 300 test puzzles, not these 209. Fixed before commit.

- **One dict, two meanings** (class 1-like, a store keyed two ways): `ZoneSearch.history` held rank()'s per-candidate
  records (Move -> list) AND the history heuristic's counters ((piece, colour, square) -> int), because rank() reset
  it and search() wrote into it. Every reader indexed by candidate move, so no number was wrong; the depth decider
  iterated it and crashed on an int. Fixed: the heuristic has its own store `hh`, reset at the same point; the
  search's values and nodes on the module's cases unchanged (the same digest before and after).

- **A negative made by the label** (class 10, inverted -- a label blind to ties): the depth decider trained on "the
  deep top shares no move with this rung's" matched fixed depth (price 1e-6: 277 vs 280 alone on top on ordinary
  positions, 19 vs 16 on the misses; within-misses AUC 0.583). That label calls a tie that resolves "no change", yet
  a tied top IS an undecided search and the solution-alone gains (280 -> 288 from 8 to 10 plies) are exactly ties
  resolving. With "the top set differs at all": held-out AUC 0.907 (misses 0.795), and the priced decider beats
  fixed depth in the replay. Found by bug-checking the negative (my own label choice first).
- **OpenMP threads spinning beside busy workers** (class 7): a tiny HistGradientBoosting fit sat minutes at 52% CPU
  in libomp barriers while 10 search workers held the cores. Fixed: OMP_NUM_THREADS=1 for the decider (2 s).

- **A monitor that watched itself** (class 7): the heartbeat found the job with `pgrep -f "frame_puzzles --n 300"`,
  which matched the monitor's own shell (its command line holds that text): "busy 0/10" while 10 workers ran, and a
  stall check that could never fire. Fixed: a bracket pattern (`analyst.frame_[p]uzzles`) and the process that has
  children.

- **A diagnostic with its own spec** (class 9-like, the harness mismatch again): miss_anatomy's zone check built the
  default TacticalSpec (no pawn moves) while the run it judged searched every pawn move -- "T overrated, refutation
  outside its zone" read 20 where the run's own rules give 6. Fixed: the check takes the run's rules (`--pawn-moves`,
  on by itself for a dump tagged _pawnmoves). Rule: a diagnostic replays the RUN's configuration, read from the run.

- **Mates dropped from a probe, again** (class 2, the ladder's bug repeated): verdict_dig's probe read a move's history
  to see whether a deeper search finds it; a move that mates at once is never searched (empty history), so Rh8# and
  Rh1# read "not found by 12". Fixed: an immediate mate counts as found. The same pattern twice in a day: any reader of
  ZoneSearch.history must treat an empty history as a mate.
- **A test that expected the leaf not to stand pat** (class 9): the zugzwang-guard test asserted no pass at all; the
  horizon's static read keeps its stand-pat by design (quiescence). The test now stubs the leaf and checks the tree.

### 2026-10-01 (imagined frames)
- **Lemma world-means read as an action's gain** (class 5-like, a stored statistic misread): the first imagined-lemma
  measurement composed lemma VALUES (means over each frame's world) as if they were a move's stake -- a free queen read
  +1.0, a free rook 0, a lone rook's quiet moves six "targets". Caught by the zero share of interactions (0.000 of all
  moves): a degenerate number. Fixed: stakes from the exact frames (what-if edits); lemmas as the naming layer.
- **A claim from a sorted, truncated print** (class 6): "the forked queen has no three-piece lemma" -- my debug print
  sorted frames by a stake column that was 0 for "keeps, unless" lemmas and cut at six rows; the queen's 3-piece frames
  call all 23 of its actions losing. Retracted to Kaveh. Rule: a "none exists" claim needs an unsorted, untruncated
  count.
- **A sentinel inside the value range** (class 2-like): moves that mate were marked by adding 1,000 to every reading,
  while frame gains can be mate-sized themselves (1,000) -- the counts with a "< 1e3" filter dropped real frames. Fixed:
  a separate mate flag, ranked first.

### 2026-10-02 (lemmas in play, the explainer)
- **A rebuild wiped hand labels** (class 7, a file with two writers): `lemma_detect.families()` rewrites
  tactic_families.json; my merge script read the old names AFTER calling it and crashed on a key-type mismatch (int
  keys in memory, string keys in JSON), so the LLM-layer names were gone and the explainer raised KeyError in 24 of 200
  puzzles. Fixed: families() itself carries the names over; the names restored from the labelling script.
- **A macOS fork-safety abort** (class 7): a multiprocessing pool forked while the parent's Stockfish engine thread was
  alive; workers that then loaded numpy/sklearn aborted ("objc ... fork() ... Crashing instead") and the pool hung.
  Fixed for the run with OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES; rule: no engine thread alive in a parent at fork.
- **A calibration read as a decision** (class 10-like): the explainer called a move a tactic when the detectors gave it
  P(any family) >= 0.5 -- balanced training inflates those probabilities (quiet moves at 0.95), so luft (f3) was named
  "starts back-rank mate". Fixed: a tactic is decided by Stockfish's line (material or mate within 8 plies); the
  detectors only name the family.

### 2026-10-04 (the Lean detector)
- **A cache rebuilt on every lookup** (class 1-like; speed only, every value right): `memoBoard` built the board's
  64-entry array and returned a closure over it in one definition; the Lean compiler eta-expanded the definition, so
  every square lookup rebuilt the array through the FEN parser's 64 nested updates -- 100-300 ms a detector query, 2 s
  for one legal-move list. Found by timing each command alone. Fixed: the array is bound at the call site
  (`readPosition`) and the closure captures it -- 0.2-0.3 ms. Rule: in Lean, build a cache in the caller, never inside
  a definition whose result is a function.
- **An unsound hypothesis in a first draft** (class 6): the passer lemma's `noEnemyReach` read only the enemy's attacks
  on today's board -- enemy pieces move -- and nothing excluded stalemate, so its (`sorry`) theorem was false as
  stated. Caught on review before any build; the draft was deleted. Rule: for every hypothesis, ask what it says about
  the opponent's NEXT moves, not about the current board.
- **A filter hid two crashes** (class 2): a smoke loop passed `$flags` unquoted in zsh, which does not split words, so
  argparse rejected "--proved-mates 1" as one argument and the runs exited at once; my `grep OBS` showed nothing and
  the 0.07 s runtime was the only sign. Fixed: `${=flags}`. Rule: every grep over a run's output also matches
  `error|usage|Traceback`.

### 2026-10-05 (the passer lemma, second form)
- **An unsound over-approximation in the prototype** (class 6, a claim not tested against the right adversary): the
  Python kingRoom skipped every forbidden square when demanding room for the enemy king, including squares holding our
  capturable pieces -- the king may take the piece and stand there. 816 exhaustive race-search checks had not hit a
  case. Found by the Lean proof (the invariant could not be stated). Fix: skip only squares next to our king or within
  one of the promotion square. Rule: an over-approximation is sound only if the excluded cases are IMPOSSIBLE, not
  merely unlikely; the proof is the test.
- **A hypothesis missing for illegal-but-representable records** (class 6): a pawn on its own back rank gains the
  double step after one push, so the distance bound could drop by two; the record allows it. Fix: a hypothesis.
- **An axiom through the kernel** (class 7-like): chess-lean's `allSquares_nodup` is `native_decide`; using it put
  `native_decide.ax` into our theorem's axioms. `#print axioms` after every theorem; reprove with `decide`.
- **Docstring before `set_option`** (three times tonight): a `/-- -/` must precede a declaration; use `--`.

### 2026-10-05 (the idea loop, round 1)
- **The ledger judged a tie-break by tied-set membership** (class 10, mirror form). RESULTS FALL fired when a set
  "lost" more puzzles than it "gained", lost meaning the solution left the tied top moves. A tie-break only removes
  moves from the tie, so that count can only fall when one acts, right or wrong. Round 1 read "pawn endings 249 -> 169,
  lost 82" while the expected score of a random pick among the tied top moves ROSE (round 0's arm 82.0 -> 92.9). I
  reported the fall to Kaveh as a finding before checking it against class 10, and retracted it within the hour. Found
  by asking what a fair measure of an order use is. Fix: `measure.expected_score`; RESULTS FALL reads it against the
  plain baseline and the previous kept round (`vs_previous`); tied/single stay as context; two tests.
- **"A flaw in my code" that was the design** (class 6, a claim not tested): I told Kaveh that 22 of the down puzzles
  came from a flaw, the tie-break ranking unanimous "keeps" moves by each rule's average gain. That ranking is
  lemma_apply.compose, Kaveh's "prefer this by this much" (2026-09-23), and on net the tie-break beat chance. Retracted.
  Rule: before calling a mechanism a bug, check whether it is the design, and measure its net effect.

- **"Mirror duplicates" that were not** (class 3, colour, plus class 6, a claim not tested). Counting the mined
  lemmas "distinct up to colour", I treated the library's "Q" and "P" as white pieces. In chess_domain's type tree a
  bare letter is a KIND, a piece of either colour ("Q" = any queen); only "wQ"/"bQ" carry a colour. The scratch count
  merged different lemmas, and I told Kaveh "72 distinct", then "92 distinct", then "a fifth of the votes repeat",
  then "1% of votes are mirrored copies" -- all wrong; with the domain's own COLOUR_FLIP the loop's library has 199
  distinct of 199. Sentences also omit a frame's own literals ("defends(0,1)"), so two sentences can read alike for
  different lemmas. Rule: identify "the same lemma" by `lemma_apply.canonical_form` (the domain's symmetries, the frame
  literals, the guard), never by text; and count before narrating.
- **Exact duplicates in older libraries** (real, small): the same lemma found twice (up to the colour flip and slot
  order) voted twice through the Applier; exch4 holds 54 of 5,280, exch2 50 of 5,232, none in the loop's. Fix:
  `Applier` keeps one per `canonical_form` (`app.duplicates` lists the dropped); test
  `test_applier_drops_a_lemma_found_twice` (fails with the dedupe disabled).
- **"Accuracy 80% to 99.7%"** (class 6): lemma_gen's purity is the precision of a lemma's EXCEPTION GUARD (of the
  rows where the guard fires, the share that are real exceptions), not the rule's accuracy. Told Kaveh; retracted.
- **The dig's step choice** (class 4-like): the first step where the two arms' top sets differ is not the step that
  moved the puzzle; a tie-break only shrinks ties, so DOWN = the step that demoted the solution. And the dumps keep
  only best[:6]; tie sizes come from n_top. Both fixed in loop/dig.py before its numbers were quoted.
- **Not a bug, checked**: lifted "after_*" relations raise TypeError when the move captures a frame piece (316 of
  411,278 evaluations on the held-out positions); both lemma_gen's `_lit` and the Applier read that as False
  (deliberate since 2026-09-24), so mining and play agree.
- **Independent review of the loop (a reviewer agent, read-only, demos on temp roots; 13 findings, most confirmed by
  running).** Fixed, each with a test where it is testable:
  - an idea found unsound on re-check stayed wired as a closure, and no gate could retire it (a registered idea was
    promoted in round 0): the wiring drops closures refused this round; a red round's gate retires its unsound ideas
    whatever round promoted them (class 6);
  - stage keys chained configs, not outputs: `--stage frame --force` left mine..measure cached on the old batch; keys
    now hold the previous stage's OUTPUT and the stage module's source; `--stage X` reruns X and what follows it (1);
  - stages read today's basis, not the one the round started from; a re-run with a new config re-froze the inputs
    from today's basis (round 0 would have counted round 1's ideas as its own): `basis.active(at=head)`, inputs
    written once (1);
  - game-loss framing read material exactly 4 plies on, right after the opponent's move: 95 of 150 positions were
    back above the threshold after our recapture; 28 more were losses the move's own forecast saw. Now the loss must
    stand after the reply, and the move's own forecast must not have seen it (4, 5);
  - expected_score read a tie of mating moves as a gamble; frame_puzzles accepts any mate. The back-rank "rise"
    (294.5 -> 297.0) was this artefact (10);
  - MinedOrder trusted (library, index) without checking the id; the mine stage took the first glob match and
    tolerated a missing re-test file; now an id mismatch raises, the round keeps its own copies of the library and
    the re-test, and the record keeps the promoted lemmas' definitions (lemmas.json) (2, 7);
  - one passer switch served two ideas: form 1's acceptance alone switched on the passer2 detector (6);
  - re-gating overwrote GATE.json and left the basis as the first decision made it: a second decision is refused (7);
  - gate._commit committed the whole index: path-scoped now (7);
  - off-by-ones in game_losses (the last horizon, the games counter) (4);
  - the back-rank bridge did not count legal-move disagreements as wrong (6);
  - dig classed puzzles the tie-break never touched (order = 0, values changed) as tie-break effects (6).
  Checked fine by the reviewer: frames_value equals the player's one-ply value (150/150), library/proved/active
  alignment, n_top on every step, dev/held-out disjointness (latent: dev rows meet the general set past 260 puzzles;
  the frame stage now refuses that).
- **The dumps kept best[:6]** while n_top was exact: analysis of ties larger than 6 read a truncated list. frame_puzzles
  now stores the whole tie.
- **A move no lemma speaks about scored 0** in MinedOrder and lost every tie to a "keeps" move (whose composed value is
  a mean outcome above 0): no evidence read as evidence. Now it is never demoted (round 1 unaffected: every move was
  spoken about). Flagged to Kaveh as possibly a design question.

- **Pass 2: an independent review of my pass-1 fixes (commit 6b5575d6) found 10 more, most confirmed by running.**
  Fixed with tests (18 loop tests): a gated round could be re-run and promote ideas no gate reviews (now refused;
  --render-only only); the output chain held only the previous stage's output while prove reads mine directly (now
  every earlier output); the back-rank bridge's `checked` counted positions, so a bridge with zero MATE claims passed
  "at least one claim checked" (now MATE claims; legal-move disagreements still count as wrong); the source digest
  covered only the stage's own file (now all analyst .py and lean/ideas .lean); --render-only rewrote the stored
  config and defeated a swap gate (now it renders under the stored config); `identical` compared whole ties against
  frozen dumps that kept best[:6]; expected_score credited a mating move where the solution does not mate, and my test
  had encoded that belief (class 9: a test of what I believed); the mine stage's copies were overwritten on a re-run
  (now named by content); the fire scan credited passer form 1's closure, which the player no longer switches on;
  stale text in dig.py. Ledger numbers unchanged (no affected steps in these dumps).
- **Code drift found by the reviewer**: today's plain player differs from the frozen pawn-ending baseline in nodes at
  56 steps, so a fresh baseline re-run would fail the reproducibility check. Not chased: the old miner's rounds are
  retired (Kaveh, "roll back, move on"); any future loop must re-freeze its baselines under the code it measures.

### 2026-10-05 (the surprise watcher)
- **Results counted once per position** (class 1-like, the evidence's unit): a game's result is ONE observation; one
  long game outweighed every other. Fixed (per-game weights), replayed.
- **The backfill's progress line said 0 for every result rule** while the counts were right (a report that did not
  follow the code path: class 2). Checked directly (R14 = the 52 games), fixed.
- **Sacrifice success read at a material PEAK mid-exchange** (class 4, the third time today: framing, settled
  material, and now this): "the material comes back" counted the instant the queen took a knight, one reply before it
  was taken. Caught by a constructed test before the index was used; the success must stand after the next reply; the
  half-built index was stopped and rebuilt. Rule: any "material at time t" in this repo is read where it STANDS.
- **A test with a wrong expectation** (class 9): my "success" line had a knight hanging to h6xg5; the rule was right.

### 2026-10-05 (evening, the PoC baselines)
- **zsh did not word-split `$args`, and my grep missed `usage`** (class 7, a REPEAT of the 2026-10-04 entry whose rule
  reads "every grep over a run's output also matches error|usage|Traceback"): three baseline runs exited 0 in a second
  having printed argparse's usage; my filter showed only my own headers, and I went to read the results. Caught by the
  missing dumps. Fix: `${=args}`; the filter now matches `usage`. Rule restated: a run that ends in seconds has not run.

- **Review of the frame-engine harness (d05f29b5), 8 findings, none changing a move** (verify mode passed; three
  seed-0 runs identical): the tie pick drew from the OPPONENT's random stream, so two engines were not opponent-matched
  (now its own stream, seed + 1); the memory share counted exact-store hits only and missed frames served by a lemma
  (`lemma_hits`) or a footprint; `nodes` read the zone search's stale counter (now 0 by construction, with the frames
  resolved per move reported instead); the final log and the JSON described the rules player's parameters (now the
  engine is named and the frame stats stored); `--frame-budget` crashed with the frame engine (refused); `move` ignored
  `--engine`. Noted by the reviewer, not fixed: VERIFY asserts only exact-store hits -- lemma and footprint
  disagreements are counted (printed now), and the per-position book cache `_JOINT` is outside VERIFY's reach.

- **A test position with the mover in check** (class 9): my knight-fork test put a black rook on the white king's
  file; the fork move was illegal and the rule "claimed nothing". Caught by debugging the position, not the rule.
  Rule: print `board.legal_moves` for a constructed test position before asserting what a rule does on it.

### Earlier scars (one line each; details in memory)
- Side-to-move flip gave an extra tempo (+3.2 vs ~+2) -- class 3 (no_side_to_move_flip).
- Split by list leaked 54/67 expert games -- class 2-like (split_by_identity_not_by_list).
- Mis-sliced lift, unmasked LFQ, ELMo missing gamma, empty fast-head slice -- silent (silent_failures_need_prompts).
- `assert ... or True`; a 0.000 loss meaning gated off -- class 2 (dead_asserts_and_silent_gates).
- Parity ate a 1-ply future lift -- class 3 (health_gate_thresholds_dont_transfer).
- Resume copied the EMA lag away -- class 1-like (resume_semantics_are_state).
