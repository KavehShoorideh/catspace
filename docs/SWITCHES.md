# SWITCHES.md -- the inventory of the analyst engine's switches and plan-numbered mechanisms

Written 2026-10-23 (read-only inventory for plan 71.5; no engine code changed, no engine and no test run). Kaveh's words that started it: "I feel like there's a lot of
vestigial things in the code that you have forgotten why you created." Plan 71.5 (`ANALYST_PLAN.md`, "71.5 THE CONSOLIDATIONS") says: "every comparison-condition switch
that no test or doc compares is deleted (list them), the rest documented in one table". This is that table, with evidence, **before anyone deletes anything**.

Scope: `catspace/incubator/analyst/discovery/` -- `agenda.py`, `regress.py`, `binder.py`, `race.py`, `plan67.py`, `ponder.py`, `facts.py`, `graph_policy.py`,
`question_prior.py`, `table.py`, `landing.py`, `words.py`, `game.py` -- plus the `analyse.py` command-line flags that set them.

How it was made: `grep -n` for module-level UPPERCASE assignments and `os.environ.get("CATSPACE_...")`; `grep -w` for every name in `catspace/` (archive excluded) and in
`tests/` (a "flip" means a test assigns or monkeypatches the name; "assert" means a test only reads it); the plan section heading in `ANALYST_PLAN.md` and the JOURNAL
heading for the date. Everything below is from reading code, not from running it. File:line is as of this commit; `agenda.py` lines drift fast.

## Verdict legend

* **LIVE** -- on the default path, and the purpose still stands. (The tests column says whether the *switch* is ever flipped; a LIVE mechanism may still have a deletable switch -- see section 71.5 list.)
* **FALLBACK** -- off by default, or reached only when something else is missing; kept as a comparison or as a path a doc names.
* **SUPERSEDED** -- a later mechanism does its job; candidate for retirement after the owner's word.
* **VESTIGIAL** -- nothing on the default path depends on it and no test or doc compares it.
* **UNKNOWN** -- I could not find why it exists (none found in this pass, see the closing section).

Abbreviations: ANALYST_PLAN section = "AP"; JOURNAL heading date = "J". `pool` = the plan 69.1 question pool; `tiers` = the plan 59 widening driver.
Test file names are given without `test_`/`.py`.

## How the default path runs (needed to read the table)

`Agenda.ask` (`agenda.py:851`): with a **time budget** (`ctx.seconds` set) the root goes through `_ask_pool` if a graph policy or question prior file loads, else `_ask_tiers`
(`TIERS_UNDER_TIME` is always True). With **no time budget** (the count fuse: unit tests) the root is `_ask_once` unless `TIERS` is True. `graph_policy.pt` and
`question_prior.json` exist in `data/derived/` of the main checkout. So the pool is the default driver; the tiers driver and everything that only the tiers driver reads
(COMPETE, STOP_AFFORD, RAISE_TO_WIN, DEFER, the RESUME pass, `held`) are the **fallback path**. In the pool `ctx.tier = TIER_COUNT - 1` (`agenda.py:922`), so every tier gate
(`ctx.tier >= 1/2/3`) is open.

## The table

| # | Name | File:line | Plan (date) | Built to solve | Default | Read by / depends on | Tests | Superseded by | Verdict |
|---|------|-----------|-------------|----------------|---------|----------------------|-------|---------------|---------|
| **A. `agenda.py` module flags and constants** |
| 1 | GAME | agenda.py:58 | AP 71.2(c), J 2026-10-22 23:41 | the agenda names squares and reads base words through the Game interface, not python-chess directly | `GM.CHESS` | every `GAME.` call in agenda (the board calls are still python-chess) | all engine tests, indirectly | -- | LIVE |
| 2 | W | agenda.py:59 | AP 71.3(b), J 2026-10-22 23:36 | the agenda asks the domain's Words (forcing_why, threat_of, defects, closes) through one interface | `CHESS_WORDS` | ~50 `W.` sites; words.py delegates to regress/race/plan67/table/binder | all engine tests | -- | LIVE |
| 3 | DEPTH_FUSE | agenda.py:62 | AP 52.13 (2b), 2026-10-18 | cut a chase 30 inner levels deep: python's stack (~9 frames a level) hit the recursion limit with the count fuse alone | 30 | `_inner` (2360) | none | -- | LIVE |
| 4 | RULE2_FROM | agenda.py:63 | AP 52.17 (2), 2026-10-19 | from which chase depth a check must address a defect of its goal (root keeps every forcing move) | mate 2, win 1, promote 1 | `_chase` line 3358 | none | -- | LIVE |
| 5 | FUSE | agenda.py:64 | AP 52.13 (2b), 2026-10-18 | the chase budget is a fuse at 10x the old question count | 10 | `_ask_pool` count budget, `ponder.py` import, regress | tiers, regress | the clock under a time budget (`NO_FUSE`) | LIVE |
| 6 | NEED_AWARE_CLOSE | agenda.py:65 | AP 52.23 (2), 2026-10-20; ON by AP 64.2, J 2026-10-21 23:20 | a rule closes a reply only at a value that can still rescue the route the agenda serves | True | `_t(...,3)` gate line 2978; `analyse --no-chain` | flip: offbuilt64, tiers, regress (isolation fixture) | -- | LIVE |
| 7 | EVASION_CHASE | agenda.py:66 | AP 52.23 (3), 2026-10-20; ON by 64.2 | the save-king question of an agenda in check is chased (its routes are the evasions) | True | lines 3537, 3937 | same three | -- | LIVE |
| 8 | NEED_PROPAGATION | agenda.py:67 | AP 52.23 (7), 2026-10-20; ON by 64.2 | the need propagates to inner agendas; a chain of two meets the need (zBouR) | True | `_t(...,3)` x4 | same three | -- | LIVE |
| 9 | GOAL_JUDGE | agenda.py:68 | AP 52.23 (6), 2026-10-20; ON by 64.2 | tempo arithmetic ends a mate/win chain ("mating the king is like mating the bishop") | True | line 3558 | same three | -- | LIVE |
| 10 | DEEP_WIN_CHASE | agenda.py:82 | AP 52.23 (5), 2026-10-20; ON by 64.2 | win chases below depth two (the skewer, the drive onto its line) | True | lines 3538, 3590 | same three | -- | LIVE |
| 11 | `_t(flag, k)` tier gate | agenda.py:412 | AP 59, J 2026-10-20 20:24; flags forced ON by 64.2 | a 52.23 chain item is ON when its flag forces it or the root's tier >= k | returns True always (all five flags True; the pool is tier 3) | 8 call sites above | via the fixtures that set the flags False | plan 64.2 ("off is not built") made it constant; with the five flags False it is still True at tier 3 | SUPERSEDED |
| 12 | CHASE_CLOCK / `_clock` | agenda.py:80, 416 | AP 52.23 (8), 2026-10-20 ("four or five ... tune it later") | stop a chain with no capture that is not bad for us after 4 of our moves (fifty-move-rule echo) | 4 (99 = off) | `_clock` (`min(CHASE_CLOCK,4)` from tier 3), analyse `_FLAGS` | flip: offbuilt64, regress, tiers; assert: plan62, plan66 | -- | LIVE |
| 13 | CLOCK_PAWN | agenda.py:69 | AP 62.1c / 66.5, J 2026-10-22 | our pawn move resets the chase clock (an irreversible move) | True | clock reset code; analyse `_FLAGS` | assert: plan62 | merged in spirit into CLOCK_IRREVERSIBLE (row 14) | LIVE |
| 14 | CLOCK_IRREVERSIBLE | agenda.py:79 | AP 66.5 (Kaveh 2026-10-22) | the clock resets on a capture not bad for us or a pawn move; a capturing check is no perpetual | True | analyse `--no-irreversible` | flip: plan66 | -- | LIVE |
| 15 | EVADED_RESET | agenda.py:78 | AP 66.5(b), separated 2026-10-22 (eSJqH trace) | a capturing check does not count as "a second check in a row" | **False** | `_chase` second-check test (3544) | flip: plan66 | the principled ends are CYCLE and the clock; the comment says "open" | FALLBACK |
| 16 | SECOND_CHECK_STOP | agenda.py:77 | AP 52.23 (3), "a switch for the audit of 2026-10-22" | "a second check of theirs in a row is not followed" (a hand-coded count) | True | line 3544 only | none | meant to be replaced by CYCLE + clock; not yet | VESTIGIAL (switch) |
| 17 | CHASE_SAVES | agenda.py:76 | AP 70.1, J 2026-10-22 20:03 (2TUzC) | a SAVE question is chased with KEEP semantics (the one-ply keep family saw a lost bishop one ply before the queen won) | True | `_chase` line 3538 | flip: plan70; regress | -- | LIVE |
| 18 | CLOSE_AFTER_CAPTURE | agenda.py:85 | 2026-10-20 (kcQil), J 2026-10-20 22:37 | a reply closes by the exchange only if their agenda after our capture finds nothing either | True | lines 1860, 2992; analyse `_FLAGS` | offbuilt64 (via `_their_after_capture`, no flip) | -- | LIVE (switch untested) |
| 19 | SURVEY | agenda.py:70 | AP 72.5, J 2026-10-23 13:30 | once per root print "what each unit can reach" and give the policy features; adds lines only | True | line 401; `regress.survey_lines`; graph_policy | flip: plan72_5 | -- | LIVE |
| 20 | SKELETON_GOALS (env `CATSPACE_SKELETON`) | agenda.py:71 | AP 72.8, J 2026-10-23 14:48 | imagine our fast units arranged against the fixed board, price by the roll, add first steps as routes/roots | True (env "0" = off) | lines 404, 763; `regress.skeleton_goals` | flip: plan72_8 | -- | LIVE |
| 21 | PASSER_ROOTS | agenda.py:72 | AP 72.2(a), J 2026-10-23 10:30 | a passed pawn is a promote root at any distance | True | line 666 | assert: plan72 | -- | LIVE (switch untested) |
| 22 | PARRY_PASS | agenda.py:73 | AP 72.2(c) | at the static leaf a one-ply parry is read once more with the move passed ("not parried YET", I7k2H) | True | line 2692 | assert: plan72 | -- | LIVE (switch untested) |
| 23 | THEIR_GAPS | agenda.py:74 | AP 72.2(d) | an open question of theirs at the static leaf makes our leaf a GAP, never a proof | True | lines 2086, 2703, 2735 | assert: plan72 | -- | LIVE (switch untested) |
| 24 | STUCK_LEAF | agenda.py:75 | AP 72.2(b) | a unit of ours that cannot run, reached by theirs below its worth, is a threat not parried YET | True | lines 2093, 2744 | assert: plan72 | -- | LIVE (switch untested) |
| 25 | PARRIES | agenda.py:87 | AP 61 A2, J 2026-10-21 11:47; ON by 64.4 | name a route's threat and label a closed reply that does not parry it | True | analyse `--parries/--no-parries`, regress.parries | flip: forcing61; offbuilt64 assert | -- | LIVE |
| 26 | DELAYED_THREAT | agenda.py:88 | AP 61 A3 / 63.3 | a quiet move after which every reply allows a forcing move is a route (tier 1+, root chase) | True | line 3576 | flip: forcing61; assert: offbuilt64 | -- | LIVE |
| 27 | NO_SAFE_MOVE | agenda.py:89 | AP 61 A4 / 64.4 | a quiet move after which every reply loses the stake by rule (zugzwang) is a route | True | line 3576 | flip: forcing61 | -- | LIVE |
| 28 | ESTIMATE | agenda.py:90 | AP 61 B2, J 2026-10-21 12:43; ON by 64.5 | a GAP route whose open replies are quiet closes as an ESTIMATED claim (status minus their pass threat) | True ("inert on the 90" per the comment: the estimator never fired there) | `_estimate_route` 3777; question_prior calibration | flip: estimate61 | -- | LIVE (never observed to change a decision) |
| 29 | COST_JUDGE | agenda.py:91 | AP 61 C2, J 2026-10-21 14:47; ON by 64.3 | under a time budget measured cost x seconds-per-node decides what is read on and replaces the count fuse; a cut is a GAP | True | `_judge_node` 3153, `_judge_stake` 3014 | flip: cost61 | the cost-only form is being replaced by AP 72.9 (rate and share); not built yet | LIVE |
| 30 | CYCLE | agenda.py:92 | AP 61 C4, J 2026-10-21 14:47 | a position repeating one of the line (or the game history) closes as a PROVED 'draw' claim when we are behind | True | `_cycle_claim` 3173; play_server history | flip: cost61 | -- | LIVE |
| 31 | JUDGE_PRICE_CAP | agenda.py:93 | AP 64.3 | stop pricing a route at 120 cost units while no rate is known (pricing 50-parry routes took 7.2 of 15 s on 07Z0P) | 120 | lines 1163, 3087 | none | AP 72.9 targets the judge's rate; may change | LIVE |
| 32 | JUDGE_MIN_AGENDAS | agenda.py:94 | AP 61 C2 | the judge reads on until 8 cost units were timed (no rate, no cut) | 8 | `_judge_rate` 3012 | none | AP 72.9 | LIVE |
| 33 | TIERS_UNDER_TIME | agenda.py:95 | AP 59 default, J 2026-10-20 20:24 (85 random puzzles) | a time budget routes the root through the widening driver (the tiers share the seconds) | True | `Agenda.ask` 853; analyse `_FLAGS` | none | the pool is chosen inside this gate (row 36) | LIVE (constant True) |
| 34 | TIERS | agenda.py:97 | AP 59, J 2026-10-20 | status-driven widening at the root; under the count fuse it is OFF unless set | **False** (CLI `--tiers`) | `Agenda.ask`, `_ask_tiers` | flip: tiers, plan65/66/68, regress, offbuilt64; `scripts/compare_tiers.py` | the pool is the default driver (AP 69.1, J 2026-10-22 20:03) | FALLBACK |
| 35 | TIER_COUNT | agenda.py:98 | AP 59 | tiers 0..3 | 4 | `_ask_pool` (sets tier 3), tiers driver, `_competitor` | assert: plan65 | -- | LIVE |
| 36 | POOL | agenda.py:104 | AP 69.1 (Kaveh 2026-10-22 "no tier label; a question pool"), J 2026-10-22 20:03 | one pool, every word available, no stages, no count fuse; the graph policy orders | True (needs a loadable policy, else tiers) | `Agenda.ask`, `_ask_pool` 911, many `ctx.pool` tests | flip False: offbuilt64, tiers, plan68, plan69 | supersedes the tiers (rows 34, 38-42) | LIVE |
| 37 | NO_FUSE | agenda.py:107 | AP 52.23 (1) | the fuse when the budget is time: never reached | 2**40 | `_ask_pool`, `_next_agenda` | plan69 | -- | LIVE |
| 38 | COMPETE | agenda.py:99 | AP 65.1, J 2026-10-22 02:28 | the next tier's cheapest route competes with this tier's root routes (Lean `judgeNext`) | True | `_competitor` returns None when `ctx.held is None`, and the pool sets `held=None` (941) -> inert in the pool | none (judge_next/_competitor: plan65) | the pool (no tiers, nothing to compete with) | FALLBACK |
| 39 | STOP_AFFORD | agenda.py:102 | AP 65.2 | the widening ends when no open route of any tier is priced within the time left | True | tiers driver line 1011 only | none | the pool | FALLBACK |
| 40 | RAISE_TO_WIN | agenda.py:86 | AP 59 (4), J 2026-10-20 21:16 | under a time budget a reached 'not losing' that is not 'winning' raises the line while time remains | True | tiers driver 989, 1010, 1016 only | none | the pool has no equivalent step (its passes continue only while the status is not kept) | FALLBACK |
| 41 | DEFER | agenda.py:105 | AP 68, J 2026-10-22 17:35 (eSJqH) | root questions whose stake is below the status's need are deferred behind the next tier's relevant ones | True | tiers driver 975/1045/1098; ask-loop 1479 is `not ctx.pool` | flip: plan68 | the pool orders by the policy (need is a feature) | FALLBACK |
| 42 | RESUME | agenda.py:103 | AP 66.4, J 2026-10-22 17:21 | with time left, chain-stopped root GAP routes are resumed in price order only while the line is still forcing | True | `_resume_chain_stops` 1136 (tiers only); 3698 gates recording `chain_gaps` | flip/assert: plan66 | the pool's passes use `extend=p` (a different path to the same end) | FALLBACK |
| 43 | CARRY | agenda.py:106 | AP 66.1 / 66.2, J 2026-10-22 17:21 | a proof carries across tiers/passes (Lean `claim_mono`); a wider read re-reads only kills, GAPs, and what could beat a floor | True | `_carry_question` 1521, 2404, 2420 (both drivers) | flip: plan66 | AP 71.5 will fold it into the fact store (not done: `carry_r` is still on `ChaseCtx`, line 245) | LIVE |
| 44 | UNPROVED | agenda.py:100 | AP 65.3 | no claim at all -> the best UNFINISHED root route (a GAP) decides, marked unproved | True | `decision()` 3858 (both drivers) | flip: plan65 | -- | LIVE |
| 45 | DEEP_PASS | agenda.py:101 | AP 65.4 | the status reads their pass with the deep family when d1 finds no claim and a piece of ours is attacked | True | `status()` 868 (both drivers) | flip: plan65 | -- | LIVE |
| 46 | PRIOR | agenda.py:83 | AP 53 (2), J 2026-10-20 13:21 | the learned linear prior orders the root questions (never decides) | **False** (CLI `--prior`) | ask-loop 1456/1465, route order 3605 -- all also open when `ctx.pool` | none | the graph policy through the pool (`or ctx.pool` in the same conditions) | SUPERSEDED |
| 47 | `ctx.tier >= 1/2` gates | agenda.py:3201, 3482, 3575-3576, 3616 | AP 59 / 61 A3 A4 | tier 1 opens quiet/delayed threats, tier 2 mate-in-two threats | open at every tier in the pool (tier 3) | forcing_why args | tiers tests | the pool (every word available) | FALLBACK |
| **B. `regress.py`** |
| 48 | DUTY (env `CATSPACE_DUTY`) | regress.py:630 | AP 63.2, J 2026-10-21 15:36 | a safe square reached by a move that abandons a duty >= the piece's worth is not safe ("pinned to a threat") | True | trap_fix, run test; analyse `--no-duty`; inside the skeleton cache key (2476) | flip: duty63; assert plan72_8 | -- | LIVE |
| 49 | STAKE_DUTY_MATE | regress.py:631 | AP 63.2 | stake of a duty on a mate threat | 100 | line 702 | none | -- | LIVE |
| 50 | STAKE_MATE / STAKE_PROMOTE / PROMOTE_MAX_PUSHES | regress.py:35,37,43 | AP 52.6 | stakes of a mate, of a promotion (queen less the pawn), and how many pushes a promote root counts | 100 / 8 / 3 | roots, fixes | regress | PROMOTE_MAX_PUSHES: the 3-push count stays only for non-passers (row 59) | LIVE |
| 51 | PASS_CLAIMS | regress.py:36 | AP 52.18 (3) | OBS counter: forcing_why answers that were a claim bound after a pass | [0] | analyse OBS line | regress | -- | LIVE (counter) |
| 52 | DEFECTS_RUST | regress.py:39 | AP 54 B1, J 2026-10-20 | mate_defects/material_defects read by `analyst_core` (Python path stays) | True | line ~110, 203 | flip: regress_rust | the Rust path; Python kept as the parity oracle | FALLBACK |
| 53 | ROUTES_RUST | regress.py:40 | AP 54 B2 | `routes_of` `_still` filter and `forcing_why` batched in Rust | True | -- | flip: regress_rust | same | FALLBACK |
| 54 | TRAP_RUST | regress.py:41 | AP 54 B4 | trap_fix's three reads in one Rust call | True | 2 sites | none | the Rust path | VESTIGIAL (switch) |
| 55 | SKEWER_RUST | regress.py:42 | AP 54 B4 | skewer_routes / drive_onto_routes through `analyst_core` | True | 3 sites | flip: regress_rust | same | FALLBACK |
| 56 | REGRESS_LOG | regress.py:43 | AP 54 (B) | a list -> every fixes_of/routes_of/forcing_why call appends what it answered | None | `scripts/compare_regress.py` | none (script) | -- | FALLBACK (harness hook) |
| 57 | MOVES_CACHE | regress.py:997 | AP 54 B5.3 | `_moves` answers once per root question | True | graph.clear_caches | flip: spread_fastpaths | -- | FALLBACK (parity switch) |
| 58 | SKEWER_FIX | regress.py:1226 | AP 52.23 (4), "switch (bisect)" | the skewer fix route in `fixes_of` | True | line 1301 only | none | -- | VESTIGIAL (switch) |
| 59 | PRICED_STOPPER | regress.py:1813 | AP 72.2, J 2026-10-23 | promote_defects' piece stoppers must reach the path free by the exchange (priced reach), not on attack count alone | True | lines 1671, 2476 | none | -- | LIVE (switch untested) |
| 60 | REACH_K | regress.py:1814 | AP 72.1 | the fuse on the roll-forward depth (a count; 72.3 intended to replace it by the race's tempo horizon) | 3 | 12 sites (reach_map, reach_price, stuck_threats, survey, skeleton) | none | AP 72.3's tempo horizon (not done) | LIVE |
| 61 | JOIN_FIXES, JOIN_KINDS, JOIN_KING | regress.py:1995-1997 | AP 72.3, J 2026-10-23 13:12, corrected 14:05 | a defect needing a piece of ours at L also gets "bring U to bear on L" for each unit, priced by the reach; kept out of fixes (quiet plans) | True | `joins_of` 2143 (root only since 16:20) | flip: plan72_3 | -- | LIVE |
| 62 | SURVEY_LINES | regress.py:2162 | AP 72.5 | print cap of the survey (a count named as one) | 8 | survey_lines | plan72_5 | -- | LIVE |
| 63 | SKELETON_MAX_JOBS / _UNITS / SKELETON_LINES | regress.py:2312-2314 | AP 72.8 (5)(12)(10), "NAMED FUSES" | drop a goal whose skeleton names more than 4 jobs; at most 7 fast units; print cap 6 | 4 / 7 / 6 | skeleton_goals 2488 | none (plan72_8 uses the function) | -- | LIVE |
| 64 | COST_CANDIDATES | regress.py:929 | AP 61 C1 | forcing moves (fewest replies first) whose parries `node_cost` counts | 3 | node_cost 933 | cost61 (via node_cost) | -- | LIVE |
| **C. `binder.py`, `facts.py`, `race.py`, `plan67.py`, `table.py`, `graph_policy.py`, `question_prior.py`** |
| 65 | BINDER_RUST | binder.py:25 | AP 54 A, J 2026-10-20 13:54 | schema reads go to `analyst_core.verify_route`; False = the Python path | True | `_read`, `_read2` | flip: binder_rust | the Rust path | FALLBACK |
| 66 | SECOND_RUST | binder.py:26 | AP 54 B4 | `second_moves` in Rust | True | -- | flip: binder_rust | same | FALLBACK |
| 67 | VERIFY_CACHE | binder.py:27 | AP 54 B4 | verify_tried answered once per root agenda | True | graph.clear_caches | flip: binder_rust | -- | FALLBACK (parity switch) |
| 68 | FEN_CACHE | binder.py:28 | AP 54 B5.2 | FEN of a board from `graph.exact_fen` | True | -- | flip: binder_rust | -- | FALLBACK (parity switch) |
| 69 | TRIED_LOG | binder.py:30 | AP 54 | harness list: every verify_tried appends | None | `scripts/compare_binder.py` | none (script) | -- | FALLBACK (harness hook) |
| 70 | PIECE_MEMO | facts.py:66 | AP 54 B5.3 | `board_piece` reads each square once | True | -- | flip: spread_fastpaths | -- | FALLBACK (parity switch) |
| 71 | RACE (env `CATSPACE_RACE`) | race.py:35 | AP 62.1, J 2026-10-22 17:21 | the passer race word (a king that must stand on h1 is not a stopper once a second passer deflects it); Lean `Race.lean` | True | agenda 4 uses, words | flip: plan62, plan67, regress | -- | LIVE |
| 72 | DENY (env) | race.py:36 | AP 62.1b | "deny the stopper's path" routes | True | regress | flip: regress | AP 71.5: merge with 62.1c support and 67.4 `support_fixes` into one weak-moment generator (not done) | LIVE |
| 73 | LIFT (env) | race.py:37 | AP 62.1c | with pieces beyond king and pawns the stopper verdict judges only their KING | True | regress | flip: plan62 | AP 71.5 (same merge) | LIVE |
| 74 | FRAME (env) | race.py:38 | AP 62.2 | the frame's condition (b) gate (tempo-immunity recheck) | True | recheck | flip: plan62 | -- | LIVE |
| 75 | STATS (race) | race.py:39 | AP 62 | OBS counters | dict | analyse | plan62 | -- | LIVE (counter) |
| 76 | PLAN (env `CATSPACE_PLAN`) | plan67.py:30 | AP 67, J 2026-10-22 17:21 | roll our plan forward without the opponent, make weak moments their goals, close by arithmetic (Lean `Ideas.Plan`) | True | words; `support_fixes` | flip: regress (isolation) | -- | LIVE |
| 77 | TABLE (env `CATSPACE_TABLE`) | table.py:16 | AP 62.3 | the tablebase (<= 5 men) as a labelled oracle (`proved_by = "table"`) | True | agenda via TB | flip: plan62, plan67, regress, estimate61 | -- | LIVE |
| 78 | MEN, STATS (table) | table.py:17, 20 | AP 62.3 | 5-man limit; probe counters | 5 / dict | table | plan62 | -- | LIVE |
| 79 | TRAIN_SURVEY | graph_policy.py:44 | AP 72.5 | `train` saves a feature-set version 1 (with the survey columns) only when on | False | graph_policy 160, 165 | none | version-2 flag (row 80) | FALLBACK |
| 80 | TRAIN_ARRANGEMENT | graph_policy.py:45 | AP 72.8 | `train` saves feature-set version 2 (arrangement column) only when on | False | graph_policy 160, 165 | none | -- | FALLBACK |
| 81 | MODEL / GOALS / STATUSES / WORDS / RWORDS / HIDDEN | graph_policy.py:32-37 | AP 60 / 69.1 | model file name and the vocabularies of the graph's node features | constants | policy | graph_policy | -- | LIVE |
| 82 | DEFAULT_CONFIDENCE, CALIBRATION, EST_LOG | question_prior.py:206-208 | AP 61 B2 | confidence of an estimated leaf's class until `calibrate` measured it; calibration and log file names | 0.5 | `estimate` path | estimate61 | -- | LIVE |
| 83 | LOG / MODEL / METRICS | question_prior.py:33,34,348 | AP 53 | file names of the question log, the prior, the RL loop's metrics | constants | `--qlog`, `train` | question_prior | graph_policy.pt for the policy; this prior is the fallback loader (`QP.load()`) in `_ask_pool` | FALLBACK |
| **D. Command-line flags in `analyse.py` that set the above** |
| 84 | `--no-chain`, `--chain` | analyse.py:194-195 | AP 64.2 | `--no-chain` sets the five chain flags False and CHASE_CLOCK 99; `--chain` is "the default since 64.2; kept for the harness" | none | -- | offbuilt64 (sets the same flags directly) | `--chain` is a no-op; in the pool `--no-chain` cannot switch the chain off (tier 3 forces it: row 11) | VESTIGIAL (`--chain`) |
| 85 | `--cycle`, `--duty`, `--cost-judge`, `--parries`, `--delayed`, `--nosafe`, `--forcing61`, `--estimate` | analyse.py:197-204,225 | AP 61 / 63 | turn ON what is now ON by default (kept so old batch scripts still run) | no-ops except `--estimate`, `--tiers`, `--prior` | -- | none | plan 64 made them defaults | VESTIGIAL |
| 86 | `--no-memory` | analyse.py:189 | AP 50 / 52.9 | "(kept for the batch scripts) the stake-only order, the default" | no-op | -- | none | -- | VESTIGIAL |
| 87 | `--memory` | analyse.py:185 | AP 50 / 52.9, J 2026-10-17 | recall stored states to reorder equal stakes (`_recall_score` 437); with 11 records it asked MORE questions on 5 of 60 puzzles | **False** | `Agenda.memory` 335; also keeps the fact store between puzzles (`carry`) | none | -- | FALLBACK |
| 88 | `--leaf-all` / `leaf_all` | analyse.py:186; agenda.py (5 uses) | AP 52.10 | read the leaf of every reply of theirs, not only the minimising ones (slower) | False | `_leaf_read` | none | their-pass leaf (52.13/14) | FALLBACK |
| 89 | `--sf` | analyse.py:191 | AP 53 (1) | retired: raises SystemExit pointing at `question_prior grade` | retired | -- | none | `question_prior grade` | VESTIGIAL |
| 90 | `--pool` / `--no-pool` help text | analyse.py:213-214 | AP 69.1 | help says the pool is "OFF by default until the agent exists" and the tiers are "the default"; code default is POOL = True | doc drift | -- | none | -- | VESTIGIAL (stale help text; fix, do not delete) |
| 91 | `_FLAGS` worker hand-off | analyse.py:135-141, 309-310 | AP 59 | module globals are not inherited by spawned workers, so the list copies the switches | 25 names | `_init_worker` | none | POOL, DEFER, SURVEY, PASSER_ROOTS, PARRY_PASS, THEIR_GAPS, STUCK_LEAF, CHASE_SAVES, EVADED_RESET, JOIN_FIXES are **not** in it (from reading: `--no-pool` / `--no-defer` / `--no-irreversible`... are applied in the parent only and lost in workers; env-backed switches survive) | LIVE (incomplete -- see actions) |
| 92 | `CATSPACE_QLOG`, `CATSPACE_ITEMLOG` | analyse.py:173, 176 | AP 53 (1), 69.3 | redirect the question/item log so a run beside another does not write the default | unset | analyse | none | -- | LIVE |
| **E. Mechanisms (plan-numbered) that are not a single switch** |
| 93 | `_ask_pool` | agenda.py:911 | AP 69.1, J 2026-10-22 20:03 | one pool ordered by the graph policy; passes with every proof carried until the status is kept or nothing changes | default driver under a time budget | `Agenda.ask`; GP/QP loaders | plan69, offbuilt64 (False branch), plan68 | supersedes `_ask_tiers` | LIVE |
| 94 | `_ask_tiers` (driver) | agenda.py:965 | AP 59 (3), J 2026-10-20 20:24 | run tier 0, widen only while no claim keeps the status | used when no policy loads, or `TIERS` with no clock | `Agenda.ask` | tiers, plan65/66/68 (POOL False) | pool; docstring of `_ask_pool`: "The tiers remain the fallback when no policy is loaded" | FALLBACK |
| 95 | status rule: `status` / `keeps` / `status_line` | agenda.py:868, 890, 898 | AP 59 (1)(2), J 2026-10-20 | stop when a claim keeps the status (balance - their pass threat); line 'reach level' vs 'stay ahead' | both drivers | pool, tiers | plan65, tiers | -- | LIVE |
| 96 | the judge: `judge_next`, `_judge_stake`, `_judge_node`, `_competitor` | agenda.py:110, 3014, 3153, 3037 | AP 61 C2 (J 2026-10-21 14:47), 64.3, 65.1 | cost x measured rate decides what a route may read; the cut is a GAP | COST_JUDGE True | `_inner` | cost61, plan65 | AP 72.9 (rate and share) is written, not built; `_competitor` is tiers-only (row 38) | LIVE |
| 97 | the cycle claim `_cycle_claim` | agenda.py:3173 | AP 61 C4 | repetition closes as a proved draw | CYCLE True | `_inner` | cost61 | -- | LIVE |
| 98 | the estimate `_estimate_route` + calibration | agenda.py:3777; question_prior.py:206-208 | AP 61 B2 | quiet leaves close as estimated claims | ESTIMATE True | `decision()` consults it only when no proved claim keeps the status | estimate61 | -- | LIVE |
| 99 | carries `carry_r` / `proofs` / `carry_q` / `chain_gaps` | agenda.py:245-254, 1115-1147, 1521 | AP 66.1-66.4, J 2026-10-22 17:21 | proofs and chain-stopped GAPs outlive a tier/pass | CARRY, RESUME True | `_next_agenda`, `_fold_tier`, `_carry_question` | plan66 | AP 71.5 asked to fold them into the fact store -- not done | LIVE |
| 100 | the fact store `FactStore` + `Fact` | facts.py:101, 26 | AP 52.19, J 2026-10-19 | justification-based store: a fact is valid wherever its read-set cells match; stale flags; a stake-ordered gap queue | on | agenda (3 uses), ponder, play_server | fact_store | its queue superseded plan 56's scheduler copy (retired 2026-10-22, `discovery/archive/scheduler_plan56.py`) | LIVE |
| 101 | fact-store serving (`roots_served`, `chase_served`, `finished`) | agenda.py:222-224 | AP 52.21 | serve root lists and chase claims across positions; the deadline finishes the route in hand | on | `ctx.facts` | fact_store | -- | LIVE |
| 102 | the landing table `LandingTable` | landing.py:121 (ctx.landings agenda.py:206) | AP 52.18 (J 2026-10-19) | memo of landing answers (targets/threats) with an inverted read index | on | agenda mate roots; `knight_fork_kills`, `fork_candidates` | regress (LandingTable) | -- | LIVE |
| 103 | the refuter fix `_refuter_fixes` / `_refuter_fix` | agenda.py:2794, 2832 | AP 52.22 (3), 2026-10-19 (07Z0P) | a killed route is saved by exchanging the refuting piece with tempo and asking again | on (no switch) | `_inner`; carry skips `:refuter` keys | none by name | -- | LIVE |
| 104 | `_refuted(why)` regex | agenda.py:1196 | AP 52.22 | classify a kill by its text | on | decision, plan65 | plan65 | AP 71.5: typed kill reason (not done) | LIVE (flagged for 71.5) |
| 106 | the reach roots `_scan_reach` | agenda.py:617 | AP 52.12 | roots from what a piece can reach (the reach read set) | on | root build | regress (no direct name hit) | AP 72 priced reach is its newer caller | LIVE |
| 107 | the leaf read `_leaf_read` | agenda.py:2104 | AP 52.10 | the trapped piece, and the leaf read by their agenda | on | `_inner` | none by name | extended by 52.13/14 (their pass on every static leaf), 70.1, 72.2 | LIVE |
| 108 | their pass `_their_pass`, `_their_agenda` | agenda.py:3321 | AP 52.13 (3), 52.14 | their agenda on every static leaf; their threat is a save root of the inner agenda | on | status, leaf, `_inner` | regress | -- | LIVE |
| 109 | `_decide_by_smallest_loss` | agenda.py:2213 | AP 52.8 | when nothing gains, the parry by the smallest loss | on | decision | none by name | -- | LIVE |
| 110 | the static leaf rules (70.1 in-check leaf is a GAP; 72.2 a-d) | agenda.py:2086-2093, 2692-2744 | AP 70.1, 72.2 | an unread threat is "not parried YET", never a proof or a kill | on (rows 22-24) | `_inner` | plan70, plan72 | -- | LIVE |
| 114 | the ponderer and delayed queue | ponder.py:96 `Ponderer`, :72 `drain`, :34 `resume_gap`; play_server.py:207 | AP 52.19 (3), 71.1 (2026-10-22) | after our move, drain the gap queue then pre-ask the human's replies in order of chain stake | used by the play server and `analyse --ponder` | FactStore.queue | fact_store | plan 56's scheduler (retired); this is "the ONE delayed queue" | LIVE |
| 115 | `one_route` rule | agenda.py:259, 921, 3122; ponder.py `_scoped` | J 2026-10-22 (no AP section number; AP 69.1 context) | when the status is LOST (or a gap is resumed) one route takes the whole share of the time; otherwise breadth, half | True/derived | pool, ponder | none by name | -- | LIVE |
| 116 | memory recall (`Agenda.memory`, `_recall_score`) | agenda.py:335, 437, 1477 | AP 50 / 52.9, J 2026-10-17 | stored solves reorder equal stakes | `memory=None` | `--memory` (row 87) | none for the recall itself | -- | FALLBACK |
| 117 | the Game and Words interfaces | game.py:23-186; words.py:211 | AP 71.2 / 71.3 / 71.6 | machinery speaks to a domain through a protocol; CHESS960 and KING_OF_THE_HILL prove extensibility | `CHESS`, `CHESS_WORDS` | agenda (rows 1-2) | test_game | -- | LIVE |

The survey, the arrangements, the joins and the chase clock were asked for as mechanisms: they are rows 19, 20, 61 and 12-14 (not repeated).

## Counts (from this table, counted by script over the rows above)

OBS switches table rows: 113 -- LIVE 77, FALLBACK 26, SUPERSEDED 2, VESTIGIAL 8, UNKNOWN 0 (a LIVE row may carry a note such as "switch untested"; counted as LIVE).

## Comparison-condition switches (the list plan 71.5 meant)

Rule 71.5: "every comparison-condition switch that no test or doc compares is deleted (list them in the commit), the rest documented in one table". "Compares" = a
test flips it to compare both forms, or a doc/script records a result of the A/B. I list every switch whose help text or comment says it is a comparison condition, with
the date of the comparison the docs record and whether a test still flips it. A switch a test flips only to *isolate* another mechanism is marked "isolation".

| Switch | CLI flag / env | Comparison (date, where) | A test still flips it? | Candidate under 71.5 |
|--------|----------------|--------------------------|------------------------|----------------------|
| TIERS (off) | `--tiers` | AP 59 vs pre-59, J 2026-10-20 20:24; `scripts/compare_tiers.py` | yes (tiers, plan65/66/68) | keep: it is the fallback driver |
| COST_JUDGE | `--no-cost-judge` | J 2026-10-21 14:47 ("OFF: 2 RIGHT LOST, diagnosed") | yes (cost61) | keep |
| CYCLE | `--no-cycle` | J 2026-10-21 14:47 | yes (cost61) | keep |
| ESTIMATE | `--estimate` | J 2026-10-21 12:43 (never fired on the 90) | yes (estimate61) | keep |
| PARRIES / DELAYED_THREAT / NO_SAFE_MOVE | `--no-parries/--no-delayed/--no-nosafe` | AP 61 A, J 2026-10-21 11:47; 63.3; 64.4 | yes (forcing61) | keep |
| NEED_AWARE_CLOSE .. DEEP_WIN_CHASE, CHASE_CLOCK | `--no-chain` | AP 64.2, J 2026-10-21 23:20 ("off is not built") | yes (offbuilt64, tiers, regress: isolation) | keep as isolation; `--chain` is a no-op |
| COMPETE, STOP_AFFORD | `--no-compete`, `--no-stop-afford` | AP 65, J 2026-10-22 02:28 | **no** | **candidate: delete the switch** (mechanism is tiers-only) |
| UNPROVED, DEEP_PASS | `--no-unproved`, `--no-deep-pass` | AP 65 | yes (plan65) | keep |
| DEFER | `--no-defer` | AP 68, J 2026-10-22 17:35 | yes (plan68) | keep (and note: lost in workers) |
| RESUME, CARRY | `--no-resume`, `--no-carry` | AP 66, J 2026-10-22 17:21 | yes (plan66) | keep |
| CLOCK_IRREVERSIBLE | `--no-irreversible` | AP 66.5, J 2026-10-22 | yes (plan66) | keep |
| EVADED_RESET | none (module only) | AP 66.5(b), 2026-10-22 | yes (plan66) | keep (its default is False) |
| RACE, FRAME, LIFT, TABLE | `--no-race/--no-frame/--no-lift/--no-table` | AP 62, J 2026-10-22 17:21 | yes (plan62, plan67, regress: isolation) | keep as isolation |
| DENY | `--no-deny` | AP 62.1b | yes (regress: isolation) | keep until the 71.5 support-generator merge |
| PLAN | `--no-plan` | AP 67 | yes (regress: isolation) | keep |
| DUTY | `--no-duty` / `CATSPACE_DUTY` | AP 63.2, J 2026-10-21 15:36 | yes (duty63) | keep |
| POOL | `--pool/--no-pool` | AP 69.1, J 2026-10-22 20:03 ("parity with the tiers on the 90, 66/1 both") | yes (offbuilt64, tiers, plan68, plan69) | keep |
| SKELETON_GOALS | `CATSPACE_SKELETON=0` | AP 72.8 | yes (plan72_8) | keep |
| SURVEY | none | AP 72.5 | yes (plan72_5) | keep |
| JOIN_FIXES | none | AP 72.3, J 2026-10-23 16:20 (flag attribution: JOIN_FIXES off 10.4 s proved vs 15.3 s unproved) | yes (plan72_3) | keep |
| CHASE_SAVES | none | AP 70.1 | yes (plan70) | keep |
| SECOND_CHECK_STOP | none | "for the audit of 2026-10-22" | **no** | **candidate: delete the switch** |
| SKEWER_FIX | none | AP 52.23 (4) "switch (bisect)" | **no** | **candidate: delete the switch** |
| TRAP_RUST | none | AP 54 B4 | **no** | **candidate: delete the switch** (its Python path is then dead too) |
| PASSER_ROOTS, PARRY_PASS, THEIR_GAPS, STUCK_LEAF | none | AP 72.2 a-d, J 2026-10-23 10:30 | **no** (tests only assert True) | **candidate: delete the four switches** (mechanisms stay) |
| CLOSE_AFTER_CAPTURE, RAISE_TO_WIN | none | 2026-10-20 (kcQil); AP 59 (4) | **no** | **candidate: delete CLOSE_AFTER_CAPTURE's switch; RAISE_TO_WIN goes with the tiers if they retire** |
| PRICED_STOPPER | none | AP 72.2 | **no** | candidate: delete the switch |
| CLOCK_PAWN | none (analyse `_FLAGS` only) | AP 62.1c / 66.5 | only asserted True (plan62) | candidate: fold into CLOCK_IRREVERSIBLE |
| the Rust parity switches DEFECTS_RUST, ROUTES_RUST, SKEWER_RUST, BINDER_RUST, SECOND_RUST, VERIFY_CACHE, FEN_CACHE, MOVES_CACHE, PIECE_MEMO | none | AP 54, J 2026-10-20 (identical results) | yes (regress_rust, binder_rust, spread_fastpaths) | keep while the Python path is the parity oracle; owner's call |

## Recommended actions, for the owner's decision (nothing below has been done)

**Retire (switch only; the mechanism stays on):** SECOND_CHECK_STOP, SKEWER_FIX, TRAP_RUST, PRICED_STOPPER, CLOSE_AFTER_CAPTURE, PASSER_ROOTS / PARRY_PASS / THEIR_GAPS /
STUCK_LEAF, COMPETE and STOP_AFFORD as flags (their code only runs in the tiers driver). Also the no-op CLI flags `--chain`, `--cycle`, `--duty`, `--cost-judge`,
`--parries`, `--delayed`, `--nosafe`, `--forcing61`, `--no-memory`, and the retired `--sf` (rows 84-86, 89).

**Retire only after your word on the tiers:** the tiers driver `_ask_tiers`, TIERS, RAISE_TO_WIN, DEFER, RESUME's pass `_resume_chain_stops`, `_competitor`, the tier
gates (row 47) and PRIOR (superseded by the pool). They are the stated fallback "when no policy is loaded" and several test files (tiers, plan65, plan66, plan68, regress, offbuilt64) flip them, so retiring them means
moving those tests to the pool. The `_t(flag, k)` gate (row 11) can go as soon as the five chain flags are constants.

**Keep, and document where it already is:** every LIVE row; the Rust parity switches (they are the oracle for the Rust ports); the comparison switches with a flipping test.

**Fix, not delete:**
1. `analyse.py`'s `_FLAGS` (row 91) omits POOL, DEFER, SURVEY, PASSER_ROOTS, PARRY_PASS, THEIR_GAPS, STUCK_LEAF, CHASE_SAVES, EVADED_RESET, JOIN_FIXES. From reading the code, a
   parallel run with `--no-pool`, `--no-defer` or `--no-irreversible`... applies the switch in the parent only. Worth a look before the next comparison run (I did not run anything).
2. The `--pool` / `--no-pool` help text says the pool is OFF by default (row 90); `POOL = True`.
3. In the pool path the five 52.23 chain flags cannot be turned off (tier 3 forces them, row 11), so `--no-chain` compares nothing there.

**Pending plan items found while counting (so they are not mistaken for dead code):** the 71.5 consolidations are not done (`carry_r`/`proofs`/`chain_gaps` are still
on `ChaseCtx`; `_refuted` is still text; the three support generators are still `race.DENY`, `race.LIFT`, `plan67.support_fixes`); AP 72.9 (the judge's rate and share)
and AP 72.10 (the local inner agenda: `grep` finds no `_local_inner`) are written but not in the code; REACH_K waits for 72.3's tempo horizon.

**Not found / unknown:** no mechanism in scope was left without a stated reason, except 72.10 (not built). `landing.py`, `words.py` and `game.py` have no flags of their own.

## Words: the lemma store (plan 73, 2026-10-25)

This inventory is of switches and mechanisms. The engine's WORDS -- every claim lemma, kill reason and GAP reason it prints, its defects, fixes and forcing words, its
techniques and its stops -- are catalogued one per file in `catspace/incubator/analyst/discovery/lemmas/` (the schema: its README), each with its kind, the base words it
reads (movement, attack, tempo, material), its Lean citations or their absence, its origin and where it is implemented. `python -m catspace.incubator.analyst.discovery.lemma_store
report` prints the inventory; `validate` fails on a word the engine prints that the store does not name. Rows above that name a word: FUSE / DEPTH_FUSE (5, 3) are `chase fuse`;
GOAL_JUDGE (9) is `tempo arithmetic`; CHASE_CLOCK (12) is `the chase clock`; SECOND_CHECK_STOP (16) is `a second check of theirs in a row`; SKELETON_GOALS (20) is
`arrangement`; THEIR_GAPS / STUCK_LEAF (23, 24) are `their open question` / `stuck_reach`; PARRIES (25) is `the threat ... still binds`; DELAYED_THREAT / NO_SAFE_MOVE
(26, 27) are `delayed threat` / `no safe move`; ESTIMATE (28) is `chase: quiet-status estimate`; COST_JUDGE (29) is `cut by the judge`; CYCLE (30) is `repetition`;
UNPROVED (44) is `unfinished line`; DUTY (48) is `the duty`; PRICED_STOPPER (59) is `stopper`; JOIN_FIXES (61) is `join`; RACE (71) is `Ideas.Race.race_sound` and
`lifted`; DENY (72) is `deny the stopper's path`; PLAN (76) is `plan_sound`; TABLE (77) is `syzygy`; LOCAL_INNER (agenda.py, plan 72.10) is `closed by the frame`.
