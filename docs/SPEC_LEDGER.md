# SPEC LEDGER: what Kaveh decided, and whether it is in the engine that plays

Made 2026-11-20 at Kaveh's word ("we need a system where these things are not forgotten ... we're kind of going in
circles and looping, and we don't have a kind of structure to our build"). The design is `docs/SYSTEM_SPEC.md`; this
file is only the list of its decisions with their state, one row each, short enough to read whole.

HOW IT IS USED (main agent; PROPOSED until Kaveh confirms):
1. It is read first: at the start of a session, after any summary of the context, before any plan is written.
2. A plan names the rows it moves and says why those before the other rows not built. A plan that moves no row says so.
   The next plan is chosen from this list, never from the last run's numbers alone.
3. A READ ends by changing the state of the rows it moved, here, in the same commit.
4. A report to Kaveh says which rows moved.
5. A new decision of his gets a row the same turn it is written into the spec.
6. A row is IN only when the playing engine does it by default and a named test would fail if it stopped.

STATES. IN: the wake does it by default, tested. PART: some of it is in the wake; what is missing is said.
OFF: built, behind a switch that is off. MEASURED: built once as a measurement outside the wake. NOT: not built.

State on main after plan 79.28 (first made at c4275c44, after 79.16). First version: rows 31 to 34 are from the status table of section 12, the
sections 2, 3, 9 and 10 were not reread for it and are owed a pass.

## The wake, step by step (SYSTEM_SPEC 8)

| Row | Decided | Spec | State | What is missing, or the proof |
|---|---|---|---|---|
| 1 | Open tactics are carried from move to move and updated only where what they read changed | 4, 8 step 1 | PART | carry.rs keeps open tactics a game; the carry by what the graph says was touched is OFF (79.14: not equal to asking afresh) |
| 2 | The position is split into frames; those with no bearing dropped | 8 step 2 | PART | Built; about 0.6 frames a decision at 0.2 s and 1.2 at 2.0 s (79.16), the read does not finish |
| 3 | Questions inside the frames, ordered by hints, stake and chance, with the lemmas and statistics of sleep; new facts reorder them | 8 step 3, 4 | PART | Ordered by stake; nothing from sleep is read but the worths and the goal table; new facts do not reorder |
| 4 | A confirmed tactic is used | 8 step 4 | IN | Named moves read first and in full (79.10); tests of cs_decision |
| 5 | Where a line ends, the chances of each side are read (the estimate scores a leaf), after every open tactic above the spread is checked | 6, 8 step 5 | NOT | A line's end is the count of worths and nothing else. The heads were left out of the wake (79.08) because they read whole-position counts |
| 6 | Where nothing is forced, subgoals order what is tried, the chances decide among what was reached, the subgoal is named | 8 step 6 | PART | Goals order the names; every goal made is the taking of a unit one or two moves deep (79.16 read) |
| 7 | The reasons as a chain of claims, each line with why it ended | 8 step 7 | IN | Logged every decision; about half the reasons are not a player's (79.11 to 79.16 reads) |
| 8 | The rule for choosing a move: their threats first; answer and advance toward our subgoal; a counter-threat is an answer | 8 | IN | rule.rs, tests |
| 9 | The threat list of both sides with stake and response, four bands | 8 | PART | Built in Rust; does not finish at the play clock; takings only |
| 10 | The wall: no oracle number in the wake; never a move the calculation holds worse | 8 | IN | wall.rs, tests wall.rs (79.15) |
| 11 | The whole is judged, every piece folded in, none turned off | 8 | PART | Off today: carry by graph, chance order, gain, the chooser's learned modes; the heads are out; rows 5, 13, 19 are not in the whole at all |
| 12 | Everything in the hot path in Rust and very fast | 11 | PART | In Rust; the steps before the search need about 0.3 s read to their end; the game's calls generate the moves three times a node (79.16) |

## Subgoals and the policy (SYSTEM_SPEC 1, 7)

| Row | Decided | Spec | State | What is missing, or the proof |
|---|---|---|---|---|
| 13 | A subgoal is held across moves, said, and dropped only for a stated cause | 1, 7 | PART | Goals are proposed afresh every move; "a goal was held about one decision" (78.75), nothing since. 79.17: a measure given as a goal is held, carried, said, dropped for a logged cause, in two small classes only. 79.18: a goal sleeps while the search holds the end and wakes after; dropped only when the class changes. 79.19: what is held lives in the memory of the game at hand |
| 14 | A goal is any set of facts of the graph with units as variables; no name needed | 7 | PART | Five fixed relations; one target shape (a unit to be taken). 79.19: placeholders are the one language of every statement. 79.27: layer 4 is asked any pattern as a goal and answers it; the wake's goals are still the five relations. 79.28: two placeholders can be said to differ; a goal still names a unit by the place it stood on |
| 15 | A problem, a proposed fix, and the fix becomes the subgoal; one way of asking for every goal | 7 | IN | goals.rs (79.07) |
| 16 | Reachable has two layers: my moves only, then the other side's veto | 7 | PART | Layer one crude and undefined for most goals (78.78), not in the wake; layer two is the chance table of 79.13 |
| 17 | Subgoals are graded: worth having, the right fix, reachable, the best use of the moves | 7 | NOT | Only the chance of reaching (79.13) |
| 18 | Maxims come out, not in: symbols that go with winning, found by two tests | 7 | MEASURED | 78.80 to 78.91; not in the wake; what passed was not what a player names |
| 19 | The future imagination layer: imagined futures rich in threats, both sides; a goal is two threats no one reply meets; the budget is questions; a goal too far is broken into a closer one | 7 | MEASURED | 78.94, 78.95; not in the wake; the breaking down NOT built. 79.27: a first step in layer 4: a line that reaches two goals, and whether one reply meets both where read in full. 79.28: under the new meaning a pair reached by force is never met by one reply; the double goal (two touched, no one reply undoes both) is not built |
| 20 | A future judge learned from human trajectories; winning frames labelled in sleep; predict whether they can be reached | 7 | MEASURED | 78.94; not in the wake |
| 21 | Among equal moves, the one that brings our goals nearer | 7 | IN | rule.rs; with row 14 as it is, it has takings only to work with |
| 22 | Calculate what is near; estimate per frame and per goal, several estimates; a learned last layer trained on results | 7 | PART | Per goal: the chance table (79.13). Per frame: NOT. Last layer built, its fit found nothing (79.12), the fixed rule plays |
| 23 | The chain of goals must end inside the search's horizon: a far goal is broken down until the nearest subgoal is one the search can reach, each a step to the next | 7 (2026-11-20) | PART | 79.17: the end of the game beyond the horizon, the next fall of the measure inside it, for a class with a mined measure; nothing breaks down a goal in a whole game |
| 24 | A subgoal reached is a training signal, to make the reward denser than the result of the game | 7, 9 (2026-11-20) | NOT | goal_reach.py counts the event (79.13); nothing is trained on it but the table. 79.17 logs each fall of the measure as a subgoal reached; still nothing trained |
| 25 | Subgoals are mined: which tactics, by existing, lead to a gain; which facts lead to those tactics; material is the subgoal on the way to the end of the game | 7, 9 (2026-11-20) | NOT | 78.80 and 78.94 are the nearest work. 79.17 mined a measure for one class from good and bad games; not tactics that lead to a gain, not whole games |
| 26 | Human data is conditioned on rating: high rating for accurate work; low rating only as a model of a low-rated opponent | 6, 10 (2026-11-20) | NOT | Samples so far were not split by rating for this purpose; to be checked |

## The estimator, value, sleep, memory (SYSTEM_SPEC 4, 5, 6, 9)

| Row | Decided | Spec | State | What is missing, or the proof |
|---|---|---|---|---|
| 27 | The estimator reads frames, not the whole position; several heads; the one preferred is the prediction of future tactical opportunities for each side | 6 | MEASURED | Heads exist over whole-position counts, weak; none per frame; none in the wake |
| 28 | Events come from the rules and no horizon is put in | 6 | PART | The heads: yes. The goal table of 79.13 uses fixed windows of plies (a departure to mend or to have decided) |
| 29 | The chance depends on the opponent | 6 | NOT | |
| 30 | No worth of a unit is assumed | 5 | IN | The human-result record; which record is the default is section 13 item 18 |
| 31 | Asleep, an agent studies what surprised the wake and hands back lemmas, statistics and a better policy | 1, 9 | PART | The loop was built and parked; the wake reads nothing it made |
| 32 | The menu of sleep, compression, the rolling measure across sleeps | 9 | NOT | |
| 33 | Lean is the source of truth for frames and lemmas | 3, AGENTS 11 | PART | Core built; three results not reached; no lemma is read by the wake |
| 34 | An event-driven memory in which a claim stands until something it read changes | 4 (and the October design) | PART | The October event loop is Python and not in the wake; a claim's read set held about half the time (78.72); the line's own read set not built. 79.19: the statement carries what it read and falls by change (tested); the wake's open tactics are one slot, not statements that fall one by one |
| 35 | A measure of progress toward a stated goal: did we get closer or are we going in circles; a forced win is a measure that falls against every reply | 7 (2026-11-20; October: plans 52.23, 73.5b, Lean forced_by_measure) | PART | Built and proved in the older system only (agenda.py, technique_miner.py, Lean); nothing of it in rust/engine. Plan 79.17 (go 2026-11-20) built it (cs_tactics::measure, the miner, the wake's held goal; tests measure.rs, wake_measure.rs). 79.18: the fall is counted against the lowest held. Missing: passed on a sample, not proved; one class; the question is cut at a short clock in a few games. 79.24: the measure's terms live in layer 3; along the table's quickest win it rises about a quarter of the time (a different road, not the mined claim); value and measure do not agree in the sample |
| 36 | Layers from the base up (representation, rules, basic facts and frames, means-ends, goals and subgoals with memory and techniques, the opponent, strategic), each with its work, its interface and a test that says it is done, so that one part is worked on at a time | 11 (2026-11-20, UNDER DISCUSSION) | PART | The crates are near these layers but not the same; no layer has a test of being done; Kaveh is deciding the list. Second pass the same day: measure of progress is value and is layer 3 under the tactical layer; memory and claims with placeholders are representation; a prediction of the future under the goals; all in Rust. Representation must be able to say facts, conjectures, probabilities and beliefs, each with its standing, and for the opponent as well (our belief about their belief state). Two memories: long-term of the game as such (lemmas, shortcuts, trained models behind their own interfaces) and the memory of the game at hand. Layer 1 has his go (plan 79.19: one statement form, two memories, a way to change them). 79.19: LAYER 1 built and tested on two games (statement, long-term memory by game, game's memory, four changes; tests memory_part_a.rs, part_b.rs, test_plan79_19.py). Missing in layer 1: old lemmas and the game's record not moved in; the game does not report its rules' address. Layers 2 and up: no test of being done yet. 79.20: layer 1's tests read fixtures of facts and call nothing of the rules (layer1.rs, test_plan79_20.py). Found: the base graph is built from the rules, not from facts; a fact's type lives in the rules' crate. Kaveh 2026-11-20: layer 1 is one graph (board, lines, units, tempo), rules next layer; not built. 79.21: BUILT, layer 1 is cs_rep below the rules with the board, the graph from facts kept step by step, and the history; the rules write relations and tags through its door (rep/tests, graph_layer2.rs, tags_layer2.rs). Missing: the wake reads no tags, builds its root graph several times; words of a game in two layer 1 files. Kaveh 2026-11-20: after 79.22, layer 2 the same way, then layer 3 and so on. 79.22: one root graph a decision, the position kept in the game's memory, repeats read from layer 1, no word of a game in layer 1 (keep_layer2.rs, test_plan79_22.py). Open: graphs after a move still built one by one; layer 2 next. 79.23: LAYER 2 has its interface (cs_game::rules) and its test against an outside reference in three games, no disagreement (game/tests/layer2.rs, test_plan79_23.py). Open: callers reach past the interface. Layer 3 waits on item 39. Kaveh 2026-11-21: layer 3 holds two implementations, value and measure of progress, their agreement measured (plan 79.24). 79.24: LAYER 3 is cs_value (value by kinds, plain count, measure behind one question; value/tests/layer3.rs, test_plan79_24.py). Open: the wake reads no value from it; the question does not take the graph. Layer 4 next (plan 79.25). 79.25: LAYER 4 has one entry (cs_tactics::meansends) and its test on node budgets (tactics/tests/layer4.rs, test_plan79_25.py), no false proof. Open: callers reach past the entry; two faults (plan 79.26). Layer 5 and up: the list waits on Kaveh. 79.26: layer 4's test read with the wake's judge too; the deepening stops on a proved end; the whole decides as before. Open: the entry takes no goal (Kaveh 2026-11-21, section 13 item 46); 9 of 61 quiet positions claimed as a gain by the wake's judge, not read. DECIDED 2026-11-21: layer 4 takes a goal as an input; plan 79.27 builds it. 79.27: built and tested (meansends::ask, layer4_goals.rs); no caller but the test; reached counts a pattern that holds for one ply (item 47). 79.28: reached means held after every reply, touched said apart (tested, lines read on the board). Open: touching doubles the nodes; the line is the shortest whatever it gives up; the test's goals are partly meaningless (item 48) |

## What this list says (main agent, 2026-11-20)

Plans 79.11 to 79.16 moved rows 4, 7, 8, 10 and 21 and the cost of rows 2, 9 and 12. They did not touch rows 5, 13, 14,
17, 19, 23, 24, 25, which are the half of the goal that says "have subgoals and navigate toward them". The main agent
also asked Kaveh (section 13 item 31) a question that row 5 had answered on 2026-11-15.
