# SYSTEM_SPEC: what we are building

Written 2026-11-14 from Kaveh's statements of that day. This is the high-level spec of the target system: the pieces
it must end up with, in his words where possible. It is a draft he asked to have "reworded and reconsidered" as it
grows, so later changes are appended as dated sections, not edited in. It holds no measured numbers; those are in
docs/ANALYST_PLAN.md beside the runs that printed them. The older documents of record (docs/THESIS.md,
docs/ARCHITECTURE.md) describe earlier systems and stand as written.

Kaveh does not claim these pieces as novelties. They are the complexities he wants built in. Most of the parts
should already exist in this repo. Nothing here is to be built anew before the history has been searched for it.

## 1. In one paragraph

A machine that learns a game from its rules, for chess first and any perfect-information game after. It holds one
graph of what it knows about a position: the units, the frames they form, the facts, and the claims reasoned from
them. It lives in two cycles. Awake, it plays: it reasons about tactics where there are any, and where there are
none it steers toward positions richer in tactical opportunity, saying which subgoal it is pursuing. Asleep, an
agent with tools studies what surprised the wake, explains it, and hands the explanation back as lemmas, memory and
a better policy. Every decision can be read as a chain of claims.

## 2. The graph

- **One place for everything.** Units are graph nodes in the same band as the facts and the claims, together with
  the observations. Some facts are unobservable; they sit in the same graph.
- **Frames are nodes too.** A frame is a small group of units described by their RELATIVE placing. Absoluteness is
  added only by asking how far the edge is. Questions can be about frames.
- **Every conclusion knows what it depends on.** A claim records what it read. A change wakes only the claims that
  read it. This is the event memory of plan 78.42.
- **Why a graph.** It makes each new kind of knowledge easy to join to the rest, and a graph network can sit on top
  of it for a policy. Kaveh is not married to the form.

## 3. Tactical reasoning

- **No game written in.** The reasoner reads the rules as data and a goal. Targets, routes, blockers and exchanges
  are derived. We supply names bound to the rules, never knowledge.
- **Questions are the unit of work, not moves.** Each asks about a vulnerability of ours or theirs, is ordered by
  stake, and ends in a verdict with its reasons. Obstacles are found by removing them and looking again.
- **Chances are the currency.** Each kind of verdict has a chance of coming true, fitted from the engine's own
  games. Chance times stake is what time is spent by, and it is the number the policy of section 5 is trained on.
- **Frames focus attention.** When the action is in one frame, a frame in a far corner with no bearing on it is not
  looked at.

## 4. Frames from the bottom up

In the first sleep cycle, positions are analysed bottom-up.

- Start with the relative placing of a few units, then three, then four, then five.
- Try to reduce a larger frame to smaller ones: a five-unit frame to a four-unit frame, or to a two-unit frame and
  a one-unit frame.
- The result is a rich store of lemmas at different frame sizes.
- A lemma is proved in Lean, by reasoning. A check over a whole class of positions is not a proof.

Open point: when two frames stop being separate. A long-range unit can join a quiet corner to the action in one
move. A candidate test: two frames are separate while no claim in one reads a place of the other.

## 5. The opportunity policy

In some positions no tactic is available now.

- **Target.** The abundance of high-value tactical opportunity. The policy steers the game toward where it is high.
- **Where the maxims come from.** Rules such as connecting the rooks or taking the centre should come OUT of this,
  from understanding the tactics. They are not supplied.
- **Training.** A separate part of the sleep cycle. The tactical state is tracked through each game and the policy
  is trained on those sequences. Kaveh floated a quasi-metric model and left the form open.
- **It talks in subgoals.** It does not say "go this way". It says "connect the rooks" or "open a file", in a
  vocabulary. Goal-conditioned reasoning then works toward the subgoal, with the tactical search serving it.
- **Two halves at the wake.** Go toward the tactics, then recognise when they are there and use them.

Working suggestions, not yet decisions:

- Opportunity as a number: the sum of chance times stake over our open claims, minus the same for theirs, credited
  only where it later turned into material.
- Subgoals as game-free statements over the graph, for example "two of my units bear on each other along an open
  line".
- A sparse readable fit over such statements before any network, so each subgoal weight is a separate item with its
  own evidence.

Standing rule that applies: subgoals enter through the prior, never the value (AGENTS.md rule 7).

## 6. Sleep

Kaveh: "the most complex piece".

- **What it is.** An agent that does tool use with the goal of improving our game. It decides what to do to
  increase its knowledge. It is not bound to the linear order of a game; it can go back and forth.
- **Its tools.**

| Tool | What it gives |
|---|---|
| The wake's own games | Where play went wrong |
| Human game databases | How people handled such positions; it can follow a game along |
| The oracle, an engine such as Stockfish | The evaluation of any position, move, line or game |
| The tactical reasoner and local simulation | Why a line works or fails, in claims |
| Lean | A proof of a stated lemma, or the failure to find one |
| Knowledge of the game | The rules and the store so far |

- **What it produces.** Lemmas, memory and other shortcuts for the wake; the vocabulary of the reasoning; the
  policy.
- **Both halves improve each cycle.** The tactics and the policy are both improved in each sleep, and sleep looks
  at which is weaker and strengthens that. Tactics come first, since steering toward opportunity is worth nothing
  without tactics that can recognise and use it.
- **The goal of a step.** A surprise generates a question. Being able to explain that surprise away is the goal.
  The wake's results are too far from one step to learn from, so each step is judged by this nearer signal, and the
  wake's results judge the whole sleep.
- **Start simple.** Before a full learning agent, a chooser that tracks which tool paid off on which kind of
  weakness.
- **Do not break the rest, and do not retest everything.** New learning must leave the other frames working, yet
  rechecking every earlier learning each time is too slow. Open, with this candidate: proved lemmas cannot be
  broken by later ones; fitted things are rechecked only where the dependency memory says the new item reaches;
  each explained surprise is kept as a standing test case.

Open points:

- What crosses from sleep into the wake. The oracle is a sleep tool. Whether anything trained on its numbers may be
  carried by the wake is Kaveh's call; until he says, no oracle number is read at wake time.
- Who states the lemmas. No outside language model is called inside the learning loop, so statements come from the
  miner and the proofs from the prover.

## 7. Wake

- Split the position into frames and drop those with no bearing on the action.
- Ask the questions by stake inside the frames that matter, using the lemmas and memory sleep provided.
- Where a tactic is confirmed, use it. Where none is, follow the policy toward opportunity and name the subgoal.
- Give the reasons for the move as a chain of claims.

## 8. Engineering rules

- **Rust in the hot paths**, for sleep and for wake, above all in position analysis, which is the bottleneck when
  many positions are read. A method is validated in Rust first and then mirrored in Python for readability.
- **Layers are game-free.** A layer names no object of any one game.
- **Lean is the source of truth** for every frame and lemma.
- **Reuse.** Most parts exist. Search the history and the indexes before building.

## 9. What exists

An inventory of the repo against these pieces was started on 2026-11-14. Its result is appended below when done.

### 9.1 Inventory of 2026-11-14

Made by reading code, docs and the git log only; no test was run for it, so "has tests" means a test file exists.
LIVE = reached by the plan-78 system (wake, sleep, the Rust engine). STANDING = in the tree with tests, not wired
in. The headline: most parts exist, and most of the October analysis parts are chess-specific Python or sit in the
older Rust crate, outside the game-free engine.

| Piece | Exists and LIVE | Exists, STANDING | Missing |
|---|---|---|---|
| Sleep and wake (section 6) | wake.py, sleep.py (surprises, cards, climb, held-out gate, publish, cycles), detectors.py, diagnose.py with `plan_tools` (a one-rule choice of tool from the wake's log), game_store.py, cs_memory snapshots and lemmas, initialize.py (the loop on a sandbox store), curriculum.py; behind flags: win/draw/loss table, playouts, targeted miner, exact labeller, corpus, exam, induction, human game index | advice.rs (advisor, teacher operator), construction.py, the October surprise loops, the idea loop of docs/LOOP.md | A sleep agent that chooses among tools from a record of what paid; a curriculum chosen from the diagnosis. The real lemma store is still empty by Kaveh's call. |
| Tactical reasoning (section 3) | Rust cs_tactics: forcing, ledger; behind flags: census, reason (threats by stake, parry or counter, irrelevant units), toward; the fact view | question_sheet.py and sheet_events.py (Python only); frame_search.py (game-free, used by the sheet); scenario.rs; October: exchange law, waiting frames, chase, trap, regions, landing table, priced reach, agenda.py, agenda.rs and the event loop, the fact store facts.py | A game-free Rust form of the question sheet, frame_search, the fact store, the landing table, regions and priced reach. The engine's own fact event stream (78.19 stages B and C) was planned and not built. |
| One graph (section 2) | fact_graph.py and decision/graph.rs: term, fact and move nodes; the claim type is declared and empty | graph_policy.py (October; squares, units, questions and routes as nodes, a graph network ordering questions; chess-specific), reasoning.py (question, line, answer, observation nodes), facts.py with read sets | One graph in the engine with units, facts, claims and frames as nodes and supports / refutes / reads edges. Frames are nodes nowhere. |
| Oracle and the wall (section 6) | The wall is in the types: an inference drive refuses a training operator, and a test reads the manifests so no decision crate depends on the learning crate. Teacher, judges and labellers are marked training-only; provenance is written on lemmas and snapshots. | October graders and the tablebase claim inside the old agenda | Nothing structural. No test yet states "no outside number reaches the wake" beyond the flags and the manifest test. |
| Miner (sections 4 and 6) | sleep.py with abstraction.py; words as terms with names built from their relations (no language model); valuation miner in Rust; behind flags: targeted, induce, corpus withdrawal | lemma_gen.py, the October proposer / verifier pipeline, technique miner, Lean obligation emitter and prover hookup (plan 73.6; the tools found no proof at the kernel's shape), lean_index.py | Question discovery (plan 57) and the loop proposing its own words (77.3) are plan text only. |
| Opportunity policy and subgoals (section 5) | `keep` heuristic; density prior behind a flag | tactic_census.py and its family (tactic tokens measured on games, "a measurement before any planner"), plan67.py (a plan's own weaknesses), question_prior.py, the summer subgoal and quasimetric stack | A policy trained toward tactical opportunity; a subgoal vocabulary in the engine. Neither was ever built past measurement. |
| Bottom-up frames (section 4) | board_structure.py with the induce tiers: relative first, then edge gap and goal gap, behind a flag | lemma_gen.py (frames of typed slots by size, a larger frame born only from a failure at the smaller), tower.py (material ladder, arrangements as offsets with edge as the only absolute), patterns.py, race.py with the proved "a far piece's move counts as a pass", Lean Frame.lean, FrameLemma.lean, ForcedByMeasure.lean, Region.lean | Reduction of a larger frame to smaller ones (plan 38.6 was written, not built); a lemma keyed by a relative frame in cs_memory; any of it in the Rust engine. |
| Rust hot paths (section 8) | cs_game, cs_tactics, cs_memory, cs_planning, cs_decision, cs_learn behind `cs_engine` | The October crate analyst_core (chess-specific): waiting, agenda, reach, trap, chase, regression | The whole question sheet and its memory are Python only. |

Also found: the engine gate's file list had two names joined by a missing space, so tests/test_plan78_14.py and
tests/test_plan78_17.py were not run by it. Fixed the same day; the two files pass.

## 10. Opponent conditioning (added 2026-11-14)

Kaveh: another piece is opponent conditioning, "telling me which tactics are likely to go the way I want depending
on opponent".

- The chance of a claim coming true is not one number. It depends on who is answering. Against one opponent a
  prepared attack is met; against another it is missed.
- So the chance table of section 3 takes the opponent as a condition, and the opportunity number of section 5 is
  the opportunity AGAINST THIS OPPONENT.
- The wake then prefers the tactics most likely to go its way against the opponent in front of it, and can say so
  in its reasons.

Working suggestions, not decisions: the opponent enters as a key of the chance table (a class of opponent first,
for example by strength or by a measured rate of missing each kind of claim), fitted in sleep from games against
that class; with no knowledge of the opponent the table falls back to the unconditioned chance, as it already falls
back by prefix. Immediate use case named the same day: learn to play and analyse chess more or less like a human
chess master.

## 11. The graph, in three layers (added 2026-11-15)

Kaveh, on hearing that the facts on the ground are not graph nodes today:

- **Layer 1, from the rules.** The engine consults the rules (GDL) and builds a graph of the basic facts and core
  concepts it finds there, one node per concept: unit, place, line, edge and the like.
- **Layer 2, from the first sleep.** Bottom-up analysis adds the concepts it mines: pins, discovered attacks and other
  base patterns. The engine does not name them. It calls them derived concept 1, 2, 3 (dc1, dc2, dc3), and we rename
  them later through a mapping file.
- **Layer 3, inside one game.** Nodes for the claims found in this game, including CONDITIONAL claims: "if we can
  attack their goal unit from here, we win the unit behind it". Nodes and edges together stand for the attack along
  the line, the unit to be won, and the condition "if I can get my unit onto that line on that side". The condition is
  a subgoal the graph policy can take up: it either finds nearer subgoals step by step, or picks one and the tactical
  reasoner gets us there.

Working suggestions, not decisions:

- A derived concept's identity is its definition (a hash of it), with dcN as the display name, so the mapping file
  survives a re-mining that finds the concepts in another order.
- A line is a maximal chain of places under one step relation of the rules; an edge place is one where some step
  relation has no next place. Both are derived, not supplied.
- A conditional claim is made by the existing counterfactual read (place a unit elsewhere, read again) and is the
  same thing section 5 calls a subgoal. The night's measurements point the same way: what foretold anything was a
  confirmed claim ARRIVING, and a conditional claim says what would make one arrive.
- The mapping file holds names only, never knowledge (the standing rule on supplied names).

Kaveh, the same day: "this means the edges will also be as graph nodes". Read as: a relation is itself a node. "This
unit bears on that unit along this line" is a node joined to the two units and the line, so a claim, a condition or a
derived concept can point at the relation itself. The graph is then two-sided: concept nodes and relation nodes, with
plain links between them. (If he meant the board's edge: that is a concept node of layer 1 either way.)

Correction, the same day: Kaveh meant the BOARD's edge. It is a concept, so it is a node of layer 1, like place and
line. The reading above, that every relation is itself a node, is not his statement. It stays only as a suggestion of
the parent's, undecided.

Decided by Kaveh the same day ("let's do it your way"): relations that a claim, a condition or a derived concept must
point at are nodes of their own; the basic layer 1 facts (this place is on this line, this unit stands on this place)
are plain links.

## 12. Frames, outcomes and value (added 2026-11-15)

Kaveh, on the mined concepts of layer 2:

- **A concept involves a frame.** A frame is a few units and a limited set of places, occupied and empty. The whole
  board is not involved.
- **Inside a frame there are sequences**, and different sequences have different outcomes: this unit is lost or won,
  that unit ends up there.
- **An intermezzo links frames.** One move here, one move in another frame elsewhere, then back here.
- **Units are variables.** "A unit that attacks here."
- **No value is assumed.** He considered conditioning on relative value and took it back. Patterns are kept in the raw
  symbols of the kinds ("when this kind attacks that kind"), and we learn what happens when each kind is left behind.
- **Value comes from statistics.** Today's value is a statistic over whole games: that is the BASELINE. The value we
  want is the same statistic asked under the condition that a given tactic, or a given fortress, is present: a
  correction to the baseline. It is not a static number.
- **Bring the earlier work over.** Frames and intermezzos were written up around early October. They are to be
  brought over, not rediscovered.

Consequences drawn by the parent, not yet his decisions: the outcome of a frame is a record of which kinds each side
is left with, not a number; a concept's worth is then how the game's result statistics differ where the concept holds;
and the worth table the sheet and the search use today is the baseline only.

### 12.1 What exists for frames and intermezzos (search of 2026-11-15)

Made by a read-only search of the journal, the plan file, the Lean files, the code and the notes for 2026-09-20 to
2026-10-31. The parent spot-checked that the files and the named definitions exist; nothing was run.

| Piece | Where | State |
|---|---|---|
| A frame as a pattern of relations over typed slots, units as variables | catspace/incubator/analyst/lemma_gen.py (`Domain`, `Frame`) | Built, game-free |
| A frame as a set of places, its units whatever bears on them | catspace/incubator/analyst/fates.py (`local_resolve`, `square_frame`, `joint_frame`) | Built, chess-bound |
| A frame as a small game with exits | catspace/incubator/analyst/frame_search.py | Built, game-free |
| Outcome as a FATE RECORD: per unit alive or taken, by which slot, at which ply, ending where; every sequence with its outcome; goals as predicates over fates | fates.py (`Fate`, `Outcome`, `outcomes_table`, `achievable`) | Built, chess-bound |
| Intermezzo: a pass imposed at a ply; one frame's line inserted into another's at ply k, units disjoint | fates.py (`intermezzi`, `with_passes`, `compose`) | Built with tests; one review finding left open (the passes use up the frame's own plies; baselines differ) |
| A move elsewhere counts as a pass | lean/ideas/Ideas/Race.lean (`withPass`, `frame_sound_pass`, `far_stays_off`), Frame.lean (`neglect`, `frame_of_diff`) | Proved in Lean |
| The exchange law with named hypotheses, the intermezzo exchange among them | exchange_law.py, ExchangeLaw.lean (`intermezzoB`, `composedValue`) | Built; the law itself is stated and checked per position, not proved |
| Waiting frames: active, dormant, far; the one board change missing; breakers | rust/analyst_core/src/waiting.rs | Built, chess-bound |
| Worth from results | catspace/incubator/analyst/discovery/valuation_miner.py; earlier piece_values.py, frame_placement.py | Built; the hand table was kept in use all the same |
| Splitting a large frame into smaller overlapping or sequential ones | plan 38.6 | Plan text only |
| A Lean theorem for composing two frames through an intermezzo | | Not found |

Two things the earlier work already learned, to be kept: nesting one frame's threat scan inside another fails (the
innermost in-between move is always left unanswered), so frames in contact are resolved jointly; and a tempo is worth
what the other side proves with the move passed to it, never a constant (plan 72.3).



### 12.2 The October Lean library against the game-free graph (inventory, 2026-11-15)

Kaveh, 2026-11-15: "October has a bunch of chess specific stuff that i would love to consider for generalization, like
this races. Lean". A read-only inventory of lean/ideas/Ideas was made by a search agent; the imports and the locations
of the twelve theorems named below were spot-checked by the main agent; no proof was read in full and nothing was
built, so whether the project builds today is not known. The rust/engine crates cite no Lean theorem yet; every citation
is in catspace/incubator/analyst or rust/analyst_core.

Three classes. ALREADY GENERIC: stated over any type and relation. LIGHTLY BOUND: the statement names chess types but
the idea needs only what layer 1 gives. CHESS ONLY: no game-free counterpart.

Five hypotheses carry nearly all of the lightly bound results. Each is a property the graph can state for any game:

| Name | Says |
|---|---|
| touch | a move changes the board only at the places it touches |
| line | whether a unit bears on a place depends only on the unit and on the places between being empty; emptying a place can only add bearings |
| footprint | a unit that bears on a place stands in that place's footprint (the place, the lines through it, the leap sources) |
| goal unit | a legal move leaves the mover's goal unit on a place no opposing unit bears on |
| distance | a distance that changes by at most one along each step of the static step relation |

| Result | File | Class | What a game-free form needs |
|---|---|---|---|
| forced_by_measure, Reach.bind, hold_blocks, toward_known | ForcedByMeasure.lean (no imports) | already generic | nothing; this file is the natural core |
| race_sound, race_first | Race.lean, Game.lean | already generic | moved out of a file that imports the chess kernel |
| frame_sound, frame_sound_pass, withPass | Race.lean | already generic | the same move |
| neglect, neglect_step, frame_of_diff | Frame.lean | already generic | the same move |
| forced_cycle_threefold, Record.step_sound | Draws.lean | already generic | drop the chess import; the move count as a parameter |
| claim_mono, judge and route lemmas | Forcing.lean | already generic | the same move |
| far_stays_off, far_not_bearing, deny_sound, plan_sound | Race.lean, Plan.lean | bound by the square type only | places for squares; one new lemma: shortest path in a kind's step relation is a distance. It replaces six chess distance lemmas |
| overload_sound, overloaded_exclusive | Race.lean | bound by the square type only | a complete list of places |
| path_lemma, attack_unchanged, lost_path_lemma | PathLemma.lean | lightly bound | touch and line |
| frame_unchanged, chess_frame, chess_neglect | FrameLemma.lean, Frame.lean | lightly bound | touch, line and footprint |
| discovered_attack, skewer_attack, battery_persists | Battery.lean, Skewer.lean | lightly bound | the `through` relation and line |
| checkmate_of_unmeetable_check | Mate.lean | lightly bound | goal unit and line, over the board without the goal unit |
| closed_binds, quiet_breaks, free_move_breaks, delayed_forcing | Forcing.lean | lightly bound | the Game structure of Game.lean in place of Position |
| exchange value reads only the footprint (seeXB_local) | ExchangeLocal.lean | lightly bound | line, footprint and chains of `through`; it compares by a worth table, which section 12 no longer assumes |
| forcedWithin_mirror | Mirror.lean | lightly bound | any map of states and moves that commutes with the rules |
| unstoppable passer (race, race2), pawn pushes, promotion, castling, en passant, the 8 by 8 speed-ups | Passer.lean, Passer2.lean, Fast.lean, Spec | chess only | the passer race is written in the shape of race_sound but does not call it; a game-free form needs a relation the graph does not have yet: a unit that advances toward a place where it changes kind |
| the exchange law | ExchangeLaw.lean | not a theorem | stated as definitions, never proved |

Order suggested: (1) a core of files that import no chess, holding the already generic results; (2) the five hypotheses
as a structure over places, units and the step relation, with chess shown to satisfy it from the existing chess lemmas;
(3) the lightly bound results restated over that structure, the frame and path lemmas first because the frames of
layer 3 rest on them; (4) the tests that tie the Rust graph to the Lean statements. Whether a game's rules satisfy the
five hypotheses is a property to be checked per game, by the engine on sampled positions and in Lean for the games we
have a kernel for.


## 13. The value at the end of a reasoning step (Kaveh, 2026-11-15)

His words: "things like damage structure and like material imbalances, all of that can be statistically answered using the query question from human games ... We can even condition on ELO. So during the sleep cycle, these should be available. And then that should be then distilled into some sort of a probabilistic value head ... so that at the end of a reasoning step, we can basically do what like Stockfish does ... We will also estimate the value, but using some sort of a stats thing that we have learned." And: "sometimes we will never have seen this exact position. But this position is made up of smaller frames, and those frames we have seen ... Frames with certain properties. So like connected rooks, for example."

Read as a design statement (main agent):
- Sleep queries games for the result given what is present in a position: the kinds each side has, and the concepts and frames of layers 2 and 3 that hold. Ratings of both players are conditions of the query.
- The answers are distilled into one estimator of the result as a probability. Its inputs are counts of kinds and of concepts and frame properties, never the whole position, so a position never seen is valued from parts that were. This is the baseline and the correction of section 12 made into one model.
- The wake reads it at the end of a reasoning step, where the frame reads say nothing more is forced. It replaces the hand worth table and settles the unequal trade that plan 78.57 leaves open.
- Open, his to decide: (1) the standing limit that human game files are not read until he asks is not lifted by this statement; a run needs his word. (2) What crosses from sleep to wake: this makes a fitted estimator from human games cross, where so far only self-play fits did; no engine number crosses. (3) AGENTS rule 7 says subgoals enter through the prior, never the value: a concept as an input of the value estimate is a feature, not a subgoal bonus, and the rule is read as still standing; to be confirmed.
- Cautions carried into any plan: corrections of frames are not additive in general (frames in contact); what players did is not what was forced, so the estimate is read only where the reasoning is quiet; rating conditions only part of the confound between a concept being present and the stronger side having made it.

Clarified by Kaveh the same day: "the estimator can be based on anything. Human games, logic, self-play, Monte Carlo playout, it can be anything. But eventually there is an estimator ... the agent in the sleep cycle will use all the tools that are available to it to train up this estimator." So the estimator is one object with many possible sources, and choosing the source is the sleep agent's work. Its interface is fixed first (parts of a position in, a probability of the result out, the source of each fit recorded beside it); the first source is self-play, which needs no decision of his. Still his to say before a run uses them: the human game files, and whether a fit that used the oracle may be read by the wake.

Decided by Kaveh, 2026-11-15 evening: "Yes, human games can be read." On the oracle, after thought: "no, stockfish cannot be read in. It has to be statistical. So either Monte Carlo playouts or human games." So the sources of the estimator are results of games: human games, self-play, playouts. No engine evaluation enters it. The oracle stays what it was: a tool of sleep for diagnosis and a measure, with no number of it in anything the wake reads.
