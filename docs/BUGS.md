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
   evaluation passes. Check: print the tie sizes beside the hit rate, and count the answer ALONE on top.

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

### Earlier scars (one line each; details in memory)
- Side-to-move flip gave an extra tempo (+3.2 vs ~+2) -- class 3 (no_side_to_move_flip).
- Split by list leaked 54/67 expert games -- class 2-like (split_by_identity_not_by_list).
- Mis-sliced lift, unmasked LFQ, ELMo missing gamma, empty fast-head slice -- silent (silent_failures_need_prompts).
- `assert ... or True`; a 0.000 loss meaning gated off -- class 2 (dead_asserts_and_silent_gates).
- Parity ate a 1-ply future lift -- class 3 (health_gate_thresholds_dont_transfer).
- Resume copied the EMA lag away -- class 1-like (resume_semantics_are_state).
