# SYSTEM_SPEC: what we are building

The high-level spec of the target system, from Kaveh's statements of 2026-11-14 to 2026-11-16, in his words where
possible. This is the clean version of 2026-11-16, made at his request ("do a once-over over the spec to clean it
up and remove any inconsistencies and make sure that we don't miss anything").

- **The record it was made from** is docs/SYSTEM_SPEC_HISTORY.md: every statement in the order it was made, with
  the full quotes, the corrections, and the three inventories of existing code. Nothing was dropped from it.
- **How this file is kept.** A new decision is written into the section it belongs to, with its date, and gets a
  row in the index of section 16. Where it replaces an earlier one, the earlier one is moved to section 14.
- **No measured number is here.** Numbers are in docs/ANALYST_PLAN.md beside the runs that printed them.
- **Marks.** DECIDED: Kaveh said it. PROPOSED: the main agent's design, told to him, not confirmed. OPEN: undecided.

Kaveh does not claim these pieces as novelties. They are the complexities he wants built in. Most parts should
already exist in this repo, and nothing is built anew before the history has been searched for it. The older
documents of record (docs/THESIS.md, docs/ARCHITECTURE.md) describe earlier systems and stand as written.

## 1. In one paragraph

A machine that learns a game from its rules, for chess first and any perfect-information game after. The immediate
use: "learn to play and analyze chess more or less like a human chess master". It holds one graph of what it knows
about a position: the units, the frames they form, the facts, and the claims reasoned from them. It lives in two
cycles. Awake, it plays: it reasons about tactics where there are any, reads an estimate of each side's chances
where the forcing ends, and steers toward positions richer in tactical opportunity, saying which subgoal it is
pursuing. Asleep, an agent with tools studies what surprised the wake, explains it concretely, and hands the
explanation back as lemmas, statistics and a better policy. Nothing about the game is assumed, not even what a unit
is worth. Every decision can be read as a chain of claims.

## 2. The graph (DECIDED 2026-11-14, 2026-11-15)

- **One place for everything.** Units are graph nodes in the same band as the facts, the claims and the
  observations. Some facts are unobservable; they sit in the same graph.
- **Three layers.**
  1. *From the rules.* The engine consults the rules (GDL) and builds the basic facts and core concepts it finds
     there, one node per concept: unit, place, line, the board's edge and the like. A line and an edge are derived
     from the step relations, not supplied.
  2. *From sleep.* Mined concepts: pins, discovered attacks and other patterns. The engine does not name them. It
     calls them dc1, dc2, dc3, and we rename them later through a mapping file that holds names only, never
     knowledge. A concept's identity is its definition, so the mapping survives a re-mining.
  3. *Inside one game.* Claims found in this game, including CONDITIONAL claims: "if my unit gets onto that line, I
     win the unit behind their goal unit". The condition is a subgoal the policy can take up.
- **Which relations are nodes** ("let's do it your way"): a relation that a claim, a condition or a concept must
  point at is a node of its own; the basic layer 1 facts (this unit stands on this place) are plain links.
- **Frames are nodes too**, and questions can be about frames.
- **Every conclusion knows what it depends on.** A claim records what it read, and a change wakes only the claims
  that read it.
- **Why a graph.** Each new kind of knowledge joins the rest easily, and a graph network can sit on top for the
  policy. Kaveh is not married to the form.

## 3. Frames (DECIDED 2026-11-15)

- **What a frame is.** A few units and a limited set of places, occupied and empty; not the whole board. Units are
  variables ("a unit that attacks here"). Placing is relative; absoluteness enters only as distance to the edge.
- **Bound to the units, not the places.** "The frame should not be bound to the square. The frame should be bound
  to the piece."
- **Sequences and outcomes.** Inside a frame there are sequences, and each has an outcome: a record of which units
  each side is left with and where, not a number.
- **Read with a few questions,** as a person does, not by listing every line. A branch that loses material is cut
  at once, with the reason kept.
- **The sacrifice goes through another frame.** A cut losing move is reopened only when it joins another frame by
  an INTERMEZZO (a move here, a move elsewhere, back here) with the right tempo: "inside of the frame, we're losing,
  but outside of the frame, we're winning."
- **Where concepts and frames come from:** from games, human games first. "We can skip the bottoms up one": the
  pass that enumerated small material bottom-up is dropped (see section 14).
- **The smallest frame that explains wins.** "If a smaller frame can explain it, the smaller frame is great. Like
  the minimum amount of information." A larger frame is kept only for what it adds to the immediate outcome.
- **Frames focus attention.** A frame in a far corner with no bearing on the action is not looked at.
- **Reuse.** The September and October work on frames, the intermezzo, the chase, the trap and cornering is brought
  over, "otherwise we will be rehashing those exact same problems, specifically around the chase."
- OPEN: when two frames stop being separate. A long-range unit can join a quiet corner to the action in one move.
  Candidate: two frames are separate while no claim in one reads a place of the other. Known from October: frames in
  contact are resolved jointly, and a tempo is worth what the other side proves with the move passed to it.

## 4. Tactical reasoning (DECIDED 2026-11-14, 2026-11-15)

- **No game written in.** The reasoner reads the rules as data and a goal. We supply names bound to the rules,
  never knowledge.
- **Questions are the unit of work, not moves.** Each asks about a vulnerability of ours or theirs, is ordered by
  stake, and ends in a verdict with its reasons. Obstacles are found by removing them and looking again.
- **Chances are the currency.** Each kind of verdict has a chance of coming true; chance times stake is what time
  is spent by.
- **A line ends for a reason.** The immediate tactical outcome is drawn "at the end of the forcing sequence": the
  settled end, where nothing stands to be lost for either side. A line cut by a limit is marked as cut, and no
  forced claim is made from it.
- **Chess is concrete** (2026-11-16). Where a concrete answer can be reached it is reached; statistics are for what
  is not sharp. When concreteness costs too much calculation, the finding is compressed (section 9).
- **Damage to the structure is a subgoal in itself.**

## 5. Value: nothing assumed (DECIDED 2026-11-15)

- **No worth of a unit is assumed, not even one per unit.** "Why even assume a piece value?" What is wanted is "the
  net effect of losing this piece on my probability of winning", and it should "fall out through our interactions
  with the game", kept in "some sort of a long-term memory ... a database search".
- **Two ways he named.** Learn from results ("every time I lose my queen, I lose"), and play at odds: "I
  incrementally add the pieces and see what comes out of it ... some sort of ablation".
- **The unknown stays in the analysis.** "An unknown placeholder for the value of each piece, and then do an
  analysis given that variable unknown."
- PROPOSED, built in part: generation zero knows the rules and the end of the game and otherwise moves at random;
  odds games measure each kind's effect on the result; that record is the next generation's judge; repeat. A
  comparison that needs worth is returned as a CONDITION on the unknowns ("good if this kind is worth more than
  that kind"), kept with the claim. The condition in the frame read is not built yet.

## 6. The estimator read where the forcing ends (DECIDED 2026-11-15, 2026-11-16)

- **What it is.** "At the end of a reasoning step" the wake reads a probabilistic estimate, as Stockfish reads its
  evaluation, "but using some sort of a stats thing that we have learned."
- **Its inputs are parts, not the whole position.** "Sometimes we will never have seen this exact position. But
  this position is made up of smaller frames, and those frames we have seen ... like connected rooks."
- **Its sources are results, never an engine's number.** "The estimator can be based on anything. Human games,
  logic, self-play, Monte Carlo playout." "Yes, human games can be read." "Stockfish cannot be read in. It has to be
  statistical." Queries can condition on rating ("we can even condition on ELO").
- **Several heads.** A baseline for the kinds and a correction to it, and the one he prefers: "the prediction of
  future tactical opportunities. So these could be multiple heads." The reasoner compares routes by them: "this is
  the better route because this has a higher probability of success."
- **What is read at the leaf is chances, for each side.** What a forcing line leaves standing ("your queen is going
  to be at the corner", doubled pawns) is input to "the potential future tactical probabilities of you and your
  opponent". A loss now can be right: "I let the queen take my rook because ... I have immense tactical chances."
- **Events come from the rules, and no horizon is put in.** "Event list would be based on gdl": a unit taken, a
  change of kind, the end and its result, a fact the rules never restore; no event name in the code. "Why do we have
  8 plies hard-coded?": the target is an event and the time to it, and how far ahead the inputs tell anything is
  printed by the run.
- **The tactical state is a sequence.** "Get a sense of the tactical states ... it is a sequence ... we should be
  able to model that sequence and predict where it will be."
- **The chance depends on the opponent** (2026-11-14): "telling me which tactics are likely to go the way I want
  depending on opponent." The opponent is a condition of the chance; with none known, the unconditioned chance.
- Cautions carried into every plan: what players did is not what was forced; a concept present may mark the
  stronger side and not cause the result; corrections of frames in contact do not add.

## 7. The opportunity policy and subgoals (DECIDED 2026-11-14)

- **Target.** Where no tactic is available now, the policy steers toward the abundance of high-value tactical
  opportunity. Its target is the chances of section 6.
- **Maxims come out, not in.** Connecting the rooks or taking the centre should come OUT of understanding the
  tactics. They are not supplied.
- **It talks in subgoals:** "connect the rooks", "open a file". Goal-conditioned reasoning works toward the
  subgoal; a conditional claim of the graph (section 2) is such a subgoal.
- **Two halves at the wake.** Go toward the tactics, then recognise when they are there and use them.
- **Standing rule:** subgoals enter through the prior, never the value (AGENTS.md rule 7).
- OPEN: the form of the model (he floated a quasi-metric and left it open). PROPOSED: a sparse readable fit over
  game-free statements before any network.

## 8. Wake

- Split the position into frames and drop those with no bearing on the action.
- Ask the questions inside the frames that matter, with the lemmas and statistics sleep provided.
- Where a tactic is confirmed, use it. Where the forcing ends, read the chances for each side. Where nothing is
  forced, follow the policy toward opportunity and name the subgoal.
- Give the reasons for the move as a chain of claims.
- **The wall.** No number of the oracle is read by the wake or by anything the wake reads.

## 9. Sleep (DECIDED 2026-11-14 to 2026-11-16)

Kaveh: "the most complex piece".

- **What it is.** An agent doing tool use with the goal of improving our game. It is not bound to the linear order
  of a game. "Each generation needs to have some sort of an agent picking multiple different actions to improve
  itself until it's satisfied and wakes up."
- **The menu holds different kinds of action** (2026-11-16: "Why do we only have one item on the menu?"):

| Action | What it gives |
|---|---|
| Playouts from a position (Monte Carlo) | "To see where it goes": statistics of results and events from there |
| A query of the human games | Conditional statistics for a frame, a concept or a set of units, by rating |
| Odds games, also from playout positions | The effect of a kind on the result |
| The oracle: Stockfish, which reads the tablebase itself | The best move of a position, to be explained |
| The tactical reasoner and the frame read | Why a line works or fails, in claims |
| Mining a concept; a Lean statement to prove | New vocabulary; a proved lemma |
| A fit of the heads | The estimator of section 6 |

- **The ladder on a position** (2026-11-16). "During the sleep, we are allowed to consult the oracle in order to ask
  about a position and figure out what the best move is." Then:
  1. Sharp: does our reasoning support the best move? If so, nothing to learn.
  2. Not sharp: "with our statistics kind of explain the best move."
  3. A surprise remains: "mine some sort of a lemma or reasoning, keep going deeper ... get to the point where you
     can explain away the surprise, because chess is concrete."
  4. "If it takes too much, then that means too much calculation. Then ... reduce it to some sort of a lemma or
     ... turn it into some sort of a statistical thing, add it into your learning algorithm."
- **The goal of a step** is a surprise explained away. A surprise generates the question.
- **What the oracle may and may not do.** It supplies the best move, and so the question. PROPOSED boundary, which
  keeps "stockfish cannot be read in": what enters the store is what our own tools made and checked (a proved
  lemma, statistics from games and playouts); no head is fitted to the oracle's evaluation or to its move as a
  label. He has not confirmed this reading.
- **When it wakes.** "No more than 5 hours", and "it could be more than 5 actions". One hour at most an action. He
  left the length to be scaled down ("maybe you need like a 5-minute sleep"); PROPOSED: a test at ten minutes, then
  an hour, the length then read from the printed curve of gain against time.
- **How it chooses with few, late rewards** (agreed 2026-11-15: "Let's go with your plan"): the expected gain of an
  action is computed where it can be, not learned from reward; each slice is scored by a near signal (a held-out
  yardstick, surprises explained); actions are small so a sleep holds many choices and the record carries across
  sleeps; the wake's match answers one question only, whether gains on the yardstick turn into better play, and
  decides whether the new store is kept. Later, untested: a learned chooser trained on small games.
- **Both halves improve each cycle.** Sleep looks at which of tactics and policy is weaker and strengthens that.
  Tactics first, since steering toward opportunity is worth nothing without tactics to use it. The counts per rung
  of the ladder are that diagnosis.
- **What it produces.** Lemmas, statistics (the heads), vocabulary, the policy.
- **Do not break the rest, and do not retest everything.** OPEN, with this candidate: proved lemmas cannot be
  broken by later ones; fitted things are rechecked only where the dependency memory says the new item reaches;
  each explained surprise is kept as a standing test case.
- **No outside language model inside the learning loop.** Statements come from the miner, proofs from the prover.

## 10. Data rules (DECIDED 2026-11-15)

- **Human games may be read.** No player id is written anywhere.
- **Endings are a known risk.** "Human games often resign before the end of the game." Completing them with a
  tablebase or Stockfish's line: "That's a bad idea, actually." "Make sure ... check for it. Remember that this is a
  risk. And if we turn out to not have enough endgames, then let's balance it by creating some." The standing rule:
  every plan that samples human games prints its endgame coverage and fills thin sets with generated games, never
  with an engine's continuation or evaluation.
- **Puzzles are a measure only.**

## 11. Engineering rules (DECIDED 2026-11-14)

- **Rust in the hot paths**, for sleep and for wake, above all in position analysis. A method is validated in Rust
  first and then mirrored in Python for readability.
- **Layers are game-free.** A layer names no object of any one game.
- **Lean is the source of truth** for every frame and lemma. A lemma is proved by reasoning; a check over a whole
  class of positions is not a proof.
- **Reuse.** Search the history and the indexes before building.
- **The real lemma store is written only on his word.**

## 12. What exists

The three inventories (the repo against these pieces; the earlier frame and intermezzo work; the October Lean
library against the game-free graph) are in docs/SYSTEM_SPEC_HISTORY.md, sections 9.1, 12.1 and 12.2, as made on
2026-11-14 and 2026-11-15. One correction to the third: Reach.lean is chess-bound, not generic.

State on 2026-11-16, in words only (plans 78.52 to 78.67 in docs/ANALYST_PLAN.md hold the runs):

| Piece | State |
|---|---|
| Graph layer 1 from the rules; layer 2 concepts selected by what becomes of their units | Built |
| Frames bound to units, the question read, the settled end, the exchange mirrored from Lean | Built; not yet a few questions a frame |
| The Lean core with chess as an instance | Built; three results not reached |
| No assumed worth: generations, odds games | Built; only the top kind is settled |
| The heads: result, kinds, arrival, events with no horizon, frames as inputs | Built as measurements; weak; not yet a leaf reading to lean on |
| The player reading the heads | Being built (plan 78.67) |
| The sleep loop with a chooser | Built; lost twice to a fixed schedule on a menu of one kind; parked until the menu of section 9 exists |
| The menu and the ladder | Written (plan 78.68), not built |
| The opportunity policy and subgoal vocabulary | Not built |
| The unknown as a condition in the frame read; the sequence model; the endgame fill from human play | Not built |

## 13. Open questions for Kaveh

1. AGENTS rule 7. The chances read at the leaf are treated as an estimate of the position's worth, and subgoal
   bonuses still stay out of the value. Is that reading right?
2. The oracle's boundary in section 9: may a head ever be fitted to the oracle's move or evaluation, or only to
   what our tools derived? Written as "only ours".
3. Does the ceiling of one hour an action stand now that the cap on the count of actions is gone?
4. The design of section 5 (generation zero, odds games as the judge of the next generation) was told and run, not
   confirmed in words.
5. When two frames stop being separate (section 3).
6. How new learning is kept from breaking the rest without retesting everything (section 9).
7. The form of the policy's model (section 7).

## 14. What was replaced, so that no earlier statement misleads

| Earlier | Replaced by | When |
|---|---|---|
| The first sleep analyses positions bottom-up by size, three units, then four, then five | Concepts and frames emerge from games; the bottom-up pass is dropped. Reducing a larger frame to a smaller one survives as "the smallest frame that explains" | 2026-11-15 night |
| A static worth of the kinds as the baseline, with corrections | No worth assumed at all; the baseline is itself measured | 2026-11-15 night |
| He floated conditioning concepts on relative value | Taken back the same day: raw kinds, "learn what happens when each piece is left behind" | 2026-11-15 |
| Chances fitted from the engine's own games only | Any source of results: human games, self-play, playouts | 2026-11-15 |
| Human game files not read until he asks | "Yes, human games can be read" | 2026-11-15 |
| The oracle gives "the evaluation of any position" to the sleep, with what crosses to the wake left open | No oracle number in any fit or in the wake; in sleep it gives the best move, to be explained by the ladder | 2026-11-15, 2026-11-16 |
| Opportunity as chance times stake, credited where it later became material | The chances at the leaf, per side, over events read from the rules | 2026-11-15 night, 2026-11-16 |
| A target within a fixed count of plies | An event and the time to it; no horizon | 2026-11-16 |
| A sleep wakes after five actions | No cap on actions; five hours at most | 2026-11-15 night |
| "Until it improves by a certain percent" as the stop | PROPOSED: the stop is gain per second or the time limit; the improvement is tested by the wake's match | 2026-11-15 night |
| "The edges will also be graph nodes" read as every relation being a node | He meant the board's edge. Decided separately: only relations something must point at are nodes | 2026-11-15 |
| A menu of odds games and fits only | The menu of section 9 | 2026-11-16 |

## 15. Standing limits that apply to every plan

No number of the oracle in a fit or in the wake. No outside language model in the learning loop. No assumed worth.
No game object named in a layer. No player id written. Lean before a lemma is relied on. The real store only on his
word. Endgame coverage printed for every human sample. Numbers only where a program printed them.

## 16. Index: where each decision is written

| Decision or direction | Date | Section |
|---|---|---|
| The pieces of the system; the immediate use | 2026-11-14 | 1 |
| One graph; three layers; dc names and the mapping file; which relations are nodes | 2026-11-14, 15 | 2 |
| Frames: units as variables, bound to units, few questions, the cut and the sacrifice through another frame, the intermezzo, the smallest frame, from human games, the chase work reused | 2026-11-15 | 3 |
| Reasoning by questions; the outcome ends at the settled end; chess is concrete; damaged structure as a subgoal | 2026-11-14, 15, 16 | 4 |
| No assumed worth; odds games; the unknown as a condition | 2026-11-15 | 5 |
| The estimator: parts as inputs, sources, rating, several heads, chances at the leaf, events from the rules, no horizon, the sequence, the opponent | 2026-11-14, 15, 16 | 6 |
| The opportunity policy; maxims come out; subgoals; rule 7 | 2026-11-14 | 7 |
| The wake and the wall | 2026-11-14 | 8 |
| Sleep: the agent, the menu, the oracle and the ladder, the limits, how it chooses, the diagnosis, do not break the rest | 2026-11-14, 15, 16 | 9 |
| Human games may be read; the endings risk and the fill | 2026-11-15 | 10 |
| Rust first; game-free layers; Lean; reuse; the real store | 2026-11-14 | 11 |
| What exists and the state of the build | 2026-11-16 | 12 |
| Open questions | 2026-11-16 | 13 |
| What was replaced | 2026-11-16 | 14 |
