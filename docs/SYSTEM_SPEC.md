# SYSTEM_SPEC: what we are building

The high-level spec of the target system, from Kaveh's statements of 2026-11-14 to 2026-11-16, in his words where
possible. This is the clean version of 2026-11-16, made at his request ("do a once-over over the spec to clean it
up and remove any inconsistencies and make sure that we don't miss anything").

- **The record it was made from** is docs/SYSTEM_SPEC_HISTORY.md: every statement in the order it was made, with
  the full quotes, the corrections, and the three inventories of existing code. Nothing was dropped from it.
- **How this file is kept.** A new decision is written into the section it belongs to, with its date, and gets a
  row in the index of section 16. Where it replaces an earlier one, the earlier one is moved to section 14.
- **After every change the whole file is reread** (Kaveh 2026-11-16: "every time you modify the spec, you need to
  do a once-over to check that the spec ... still basically makes sense. There's no kind of conflicting or missing
  pieces"). Conflicts are fixed in the same change; a gap that needs his word goes to section 13.
- **No measured number is here.** Numbers are in docs/ANALYST_PLAN.md beside the runs that printed them.
- **Marks.** DECIDED: Kaveh said it. PROPOSED: the main agent's design, told to him, not confirmed. OPEN: undecided.

Kaveh does not claim these pieces as novelties. They are the complexities he wants built in. Most parts should
already exist in this repo, and nothing is built anew before the history has been searched for it. The older
documents of record (docs/THESIS.md, docs/ARCHITECTURE.md) describe earlier systems and stand as written.

## 1. In one paragraph

A machine that learns a game from its rules, for chess first and any perfect-information game after. The immediate
use: "learn to play and analyze chess more or less like a human chess master". It holds one graph of what it knows
about a position: the units, the frames they form, the facts, and the claims reasoned from them. It lives in two
cycles. Awake, it plays: it reasons concretely while it has budget, carries the open tactics from move to move,
reads an estimate of each side's chances where the reasoning ends, and steers toward positions richer in tactical
opportunity, saying which subgoal it is pursuing. Asleep, an agent with tools studies what surprised the wake, explains it concretely, and hands the
explanation back as lemmas, statistics and a better policy. Nothing about the game is assumed, not even what a unit
is worth. Every decision can be read as a chain of claims.

**The overall goal (Kaveh 2026-11-16).** "The overall goal has to be able to reason like a human being ... Think
through the concrete tactical steps and then evaluate the position at the end of the tactical steps." This is the
test every piece is held to: the tactical reasoning of section 4 is the concrete steps, the estimator of section 6
is the evaluation where those steps end, and the trace of a decision should read like a master's line of thought:
this threat, that answer, so this, and the position that results is about so good.

Same day, the second half of the goal: "and then be able to have sub goals and then navigate towards those sub
goals." So the goal has two halves. Where something is forced, think the concrete steps through and evaluate at
their end. Where nothing is forced, hold a subgoal, say which one, and choose moves that bring it nearer, over
several moves and not only the next one. Section 7 is the piece that serves this half and says what "nearer"
means; section 8 has the rule for the move. It is not built (section 12). PROPOSED: the subgoal is held across
moves the same way open tactics are carried (section 4).

He added: "you already know how to reason like a human being because you've read a lot of chess books." What he
means by it (2026-11-16, late, correcting the main agent's first reading): "I mean like you, the LLM, can judge
whether or not the engine is actually doing its job ... because you, the LLM, knows how humans reason." So the
language model is the JUDGE of the engine's reasoning, at build time and outside the engine. For every piece that
reasons, the building agent reads what the engine printed against the position and says whether a strong human
would reason so: is this a threat a player would name, is this the answer, is this a sensible aim. That reading
is part of the check of a build (section 11). The judge stays outside: it puts nothing into the engine, no worth,
rule of thumb or pattern (sections 5 and 15), and nothing it says is read by the wake or the sleep. It tells the
builders what is wrong; the mending comes from the graph, the reasoning and the sleep.

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
     win the unit behind their goal unit". The condition is a subgoal the policy can take up. Open claims are carried
     from move to move (section 4).
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
  at once, with the reason kept. "Loses material" is judged by the measured reading on kinds (section 5), never by
  a table.
- **The sacrifice goes through another frame.** A cut losing move is reopened when it joins another frame by an
  INTERMEZZO (a move here, a move elsewhere, back here) with the right tempo: "inside of the frame, we're losing,
  but outside of the frame, we're winning."
- **A sacrifice needs a reason, in this order of preference** (DECIDED 2026-11-16). "I feel like there's gotta be a
  concrete reason."
  1. It stops the opponent from making progress, "because you can cite the reason why".
  2. "You can recover what you sacrificed and you can state the reason why."
  3. Neither: "somehow you have to have an estimate ... You don't know why exactly you're sacrificing, you just
     know it's better. That third one is the least desirable, but if it exists, then we should do it. Or ... better
     yet, we should try to figure out why is that better."
  The third case is itself a surprise: "your own reasoning disagreed with your own statistics ... So it should
  become a source of surprise ... we should less and less depend on it ... the surprises will pop up during sleep,
  and we will find the rules to explain them." It is marked as unexplained in the reasons for the move and goes on
  the sleep's list (section 9).
- **Where concepts and frames come from:** from games, human games first. "We can skip the bottoms up one": the
  pass that enumerated small material bottom-up is dropped (see section 14).
- **The smallest frame that explains wins.** "If a smaller frame can explain it, the smaller frame is great. Like
  the minimum amount of information." A larger frame is kept only for what it adds to the immediate outcome.
- **Frames focus attention.** A frame in a far corner with no bearing on the action is not looked at.
- **Reuse.** The September and October work on frames, the intermezzo, the chase, the trap and cornering is brought
  over, "otherwise we will be rehashing those exact same problems, specifically around the chase."
- **When frames are separate, and how they join** (DECIDED 2026-11-16). Frames are separate while the linkage
  between them is not executed. "Every frame needs to have some sort of an interface that allows us to connect to
  intermezzos, passes, other places, threats in other places, deflections."
  - *The far end is a variable.* "It doesn't matter where the piece is deflected to. What matters is it's deflected
    from here." It is enough to know the unit is drawn to something of higher stake, "so this one is unguarded".
    "By maintaining these variables, we're able to kind of limit the size of the frame."
  - *Exact where the stake is smaller.* "You want to be exact on the frame with a smaller stake. The frame with a
    bigger stake, that's the one that's not going to execute because it's going to be vetoed": the defender saves
    the bigger thing, so the smaller one is what actually happens.
  - *Two frames at once.* "I make one move here and then one move there ... there's nothing stopping me from
    maintaining two frames in my memory." Instances of frames are graph nodes, "added, deleted as the stake
    progresses", with the graph network over them.
  - Caution added by the main agent: the bigger frame must still be read far enough to know its threat is real and
    that it can be met; if it cannot be met, it is the one that executes (a double threat). This is the same check
    as in section 6: for an open tactic of high stake, what must be known is that every threat has a defence, not
    the exact line.
  - Known from October and kept: frames in contact are resolved jointly; a tempo is worth what the other side
    proves with the move passed to it. Already built toward this: the named causes of plan 78.60 (the only guard
    left, a line uncovered) and the reopening through another frame.

## 4. Tactical reasoning (DECIDED 2026-11-14 to 2026-11-16)

- **No game written in.** The reasoner reads the rules as data and a goal. We supply names bound to the rules,
  never knowledge.
- **Questions are the unit of work, not moves.** Each asks about a vulnerability of ours or theirs and ends in a
  verdict with its reasons. Obstacles are found by removing them and looking again.
- **Stake, and what time is spent by.** The stake of a question is what is at risk in it, in one unit everywhere:
  the change in the chance of the result (from the measured readings of sections 5 and 6, never a table). Each
  kind of verdict has a chance of coming true. Chance times stake is what time is spent by.
- **The loop while there is budget** (2026-11-16). "Any sort of estimate or hint is basically telling me what order
  to check the questions in. And those orders will kind of create new facts, and the new facts will get me to change
  the order of my questions." Hints and estimates ORDER; they do not conclude.
- **What asks the questions: no model of its own** (DECIDED 2026-11-16, late; the main agent's recommendation, to
  which he said "let's go with your rec"). The order of questions comes from reasoning already in the system: the
  threat list says which questions matter first (section 8); means-ends says what the next one is, since a goal
  that fails names what stops it and "can that be mended" is the next question (section 7); the four ways a line
  ends say when to stop. Asking questions inside one move and holding a subgoal across moves are the same
  mechanism at two scales, so the asking belongs to the opportunity policy and is not a third chooser. What is
  MEASURED instead: how many questions a decision takes. A learned judge of "is one more question worth its
  time" is added only if that count stays far above a human's, and then only to divide a fixed budget among
  questions the reasoning ranks equal. A budget cut must never read as a refusal (plan 78.73).
- **Chess is concrete** (2026-11-16). Where a concrete answer can be reached it is reached. When concreteness
  costs too much calculation, the finding is compressed in sleep (section 9).
- **Open tactics are carried forward** (2026-11-16): "a memory of the tactics we have previously found", updated
  from move to move only where something they read has changed. "I would have to miss it on every move, not on one
  single move." Carrying guards against a miss of attention or budget, not against what the reasoner cannot see at
  all; that is for the sleep to repair.
- **How a line ends.** Every line ends for a stated reason, one of these, in this order of trust:
  1. CONCRETE: the end of the game, or a proved lemma, decides it.
  2. QUIET: nothing stands to be lost for either side ("the end of the forcing sequence", the settled end); no open
     tactic carried in the memory has a stake above the spread between the two readings of section 6; and the two
     readings agree. The estimate is then read. Confirmed 2026-11-16: a line may end so while budget remains "if
     none of the tactics lead anywhere ... we have to check all of our tactics that are open." The budget then goes
     to the lines that still have something sharp.
  3. LOST BEYOND DOUBT: "a big material disadvantage" by the readings. The line "need not be followed to its end".
  4. OUT OF BUDGET: the estimate is read because nothing more can be asked, and the line is marked as cut. No forced
     claim is made from a cut line.
- **Damage to the structure is a subgoal in itself.**

## 5. Value: nothing assumed (DECIDED 2026-11-15)

- **No worth of a unit is assumed, not even one per unit.** "Why even assume a piece value?" What is wanted is "the
  net effect of losing this piece on my probability of winning", and it should "fall out through our interactions
  with the game", kept in "some sort of a long-term memory ... a database search".
- **Two ways he named.** Learn from results ("every time I lose my queen, I lose"), and play at odds: "I
  incrementally add the pieces and see what comes out of it ... some sort of ablation".
- **The unknown stays in the analysis.** "An unknown placeholder for the value of each piece, and then do an
  analysis given that variable unknown."
- **Confirmed intent** (2026-11-16): "I wanted to learn the value ... and I don't want to program it. So it has to
  come somehow naturally from experience."
- MEASURED 2026-11-16 (plan 78.76, one run each, a direction only): three measured records, from odds games, from
  self-play results and from human results, order the kinds differently, and the ladder's division is the same under
  all three within noise. Which record judges is so far not what limits play.
  CORRECTED 2026-11-19 (plans 78.98, 78.99, one run each): that was read from the ladder alone. In PLAY the search under the self-play record
  or the human-result record is clearly ahead of the search under the odds record, on two opening sets each, at equal depth; the odds
  record orders three of the kinds inside its own half widths otherwise than the other two. Which record is the default is Kaveh's
  decision (section 13 item 18); PROPOSED by the main agent: the human-result record. The default is unchanged until he says.
- PROPOSED, built in part: generation zero knows the rules and the end of the game and otherwise moves at random;
  odds games measure each kind's effect on the result; that record is the next generation's judge; repeat. A
  comparison that needs worth is returned as a CONDITION on the unknowns ("good if this kind is worth more than
  that kind"), kept with the claim. The condition in the frame read is not built yet.

- PROPOSED (main agent, 2026-11-18, plan 78.85; not Kaveh's word): the equal reading has two forms. The fast form asks each move "at least the best?" with the search's own pruning and is what play uses. The verified form asks each candidate again on a fresh table without the late reduction and the null move; it is the reference for measurements and for slow readings in sleep, and is off in play until a cheaper sound question exists. A set cut by the clock is smaller, never unverified. MEASURED (one run): on against off the engine played level at the short clock.

- PROPOSED (main agent, 2026-11-19, plan 78.92; not Kaveh's word): the chosen move has a verified form as the equal reading has. The search's move is asked again without the late reduction and the null move; if it fails, the first move of the equal reading that holds takes its place. Off in play until a second run repeats 78.92's result and Kaveh says so. MEASURED (one run): the verifying player scored above the one without it at the short clock.

## 6. The estimator: a belief read where the reasoning ends (DECIDED 2026-11-15, 2026-11-16)

What it is and what it is made of:

- **What it is.** "At the end of a reasoning step" the wake reads a probabilistic estimate, as Stockfish reads its
  evaluation, "but using some sort of a stats thing that we have learned." When the budget is gone, "I have to stop
  and ask a different question ... from here, what are my chances to win? And that chance is essentially an
  estimate and can take into account all of the facts on the board and all of the claims we've gotten, and whether
  or not the rooks are connected is part of those facts. So yes, it can be used as a belief."
- **Its inputs are parts, not the whole position.** "Sometimes we will never have seen this exact position. But
  this position is made up of smaller frames, and those frames we have seen ... like connected rooks."
- **Its sources, in order of preference.** (1) Results of play: "Human games, logic, self-play, Monte Carlo
  playout"; "Yes, human games can be read"; queries can condition on rating. (2) Our own deeper reasoning: what a
  much deeper reading concludes from the leaf where the wake stopped (section 9). (3) The oracle, only as a last
  resort and on his word (section 9); until then "stockfish cannot be read in. It has to be statistical."
- **Several heads.** A reading on the kinds alone (the baseline) and a reading on all the parts (the baseline with
  its correction), and the one he prefers: "the prediction of future tactical opportunities. So these could be
  multiple heads." The reasoner compares routes by them: "this is the better route because this has a higher
  probability of success."
- **What is read is chances, for each side.** What a forcing line leaves standing ("your queen is going to be at
  the corner", doubled pawns) is input to "the potential future tactical probabilities of you and your opponent".
  A loss now can be right: "I let the queen take my rook because ... I have immense tactical chances."
- **Events come from the rules, and no horizon is put in.** "Event list would be based on gdl": a unit taken, a
  change of kind, the end and its result, a fact the rules never restore; no event name in the code. "Why do we have
  8 plies hard-coded?": the target is an event and the time to it, and how far ahead the inputs tell anything is
  printed by the run.
- **The tactical state is a sequence.** "Get a sense of the tactical states ... it is a sequence ... we should be
  able to model that sequence and predict where it will be."
- **The chance depends on the opponent** (2026-11-14): "telling me which tactics are likely to go the way I want
  depending on opponent." The opponent is a condition of the chance; with none known, the unconditioned chance.

When it may be read (section 4 gives the list of how a line ends; these are the rules behind it):

- **Concrete first** (his answer on AGENTS rule 7). "At the leaf, the estimate can be used when we don't have any
  more budget to search reasoning, follow reasonings concretely. But while we do have budget ... there's no reason
  to fall back to an estimate." So the estimate scores a leaf, and never replaces a concrete conclusion the budget
  could still reach. Before that it only orders what is looked at. Subgoal bonuses stay out of the value.
- **Quiet or sharp, told by two readings agreeing.** "When the statistical end aligns with the material reading,
  that means the position is quiet and we can read the estimate. When the statistical reading is not the same as
  the material reading, that means there's something sharp going on ... and then we have to follow the reasoning."
  PROPOSED form: the material reading is the head on kinds alone, the statistical reading the head on all the
  parts; the size of their difference is the signal. Cautions: both can be blind to the same tactic, so the test
  that nothing stands to be lost comes first; a difference can be lasting compensation that no reasoning resolves,
  so one that persists across settled positions is accepted as the estimate; and whether the difference foretells
  events is measured before it is relied on (plan 78.69).
  MEASURED 2026-11-16 (plan 78.69, one run, a direction): with the present heads the size of the difference did
  NOT mark positions where something happens soon. Most large differences lasted to the next settled position.
  What did go with events soon was the frame read's own open claims (a unit standing lost). So for now the frame
  read's open claims are the working signal of sharpness, and the rule of two readings stays his design, NOT YET
  SHOWN. The test was weak in a way that matters: the head on kinds alone is barely better than no head, so the
  "material reading" hardly exists yet and the difference is mostly the other reading by itself. It is measured
  again when the kinds reading is stronger.
- **Open tactics are checked before the estimate is trusted.** "If there are open tactics with a high stake,
  basically larger than the variation between the two readings ... one is saying +1, the other is saying +2, and
  then we have like a rook under attack ... before I trust the +1 or the +2, I should check my rook ... if I can
  find that for every threat to my rook, I have a good defense, then ... I can trust it."

Cautions carried into every plan: what players did is not what was forced; a concept present may mark the stronger
side and not cause the result; corrections of frames in contact do not add; a head that predicts our own thinking
can inherit its errors, so it is anchored by ends of games and proved lemmas and measured against the oracle.

## 7. The opportunity policy and subgoals (DECIDED 2026-11-14, 2026-11-16)

- **Target.** Where no tactic is available now, the policy steers toward the abundance of high-value tactical
  opportunity. Its target is the chances of section 6.
- **Two learned choosers in the system, and they are different** (2026-11-16, late). The sleep's chooser picks
  which tool to run and is paid in surprises resolved per second (section 9). The opportunity policy picks a
  subgoal and the moves toward it in play and is paid in results. The heads predict and choose nothing
  (section 6). The asking of questions is part of this policy, not a separate model (section 4).
- **How it acts, two ways that must not be mixed.** As a BELIEF: the chances are what a leaf is scored by when the
  reasoning has ended (section 6). As a HINT: a subgoal orders which questions and moves are looked at first. A
  subgoal never adds to the value (AGENTS.md rule 7); the chances are not a subgoal bonus, they are the estimate of
  what the position is worth.
- **Maxims come out, not in.** Connecting the rooks or taking the centre should come OUT of understanding the
  tactics. They are not supplied. Connected rooks may then be a fact the belief reads, and "connect the rooks" a
  subgoal the hint proposes.
- **It talks in subgoals:** "connect the rooks", "open a file". Goal-conditioned reasoning works toward the
  subgoal; a conditional claim of the graph (section 2) is such a subgoal.
- **Two halves at the wake.** Go toward the tactics, then recognise when they are there and use them.
- **Holding a subgoal and navigating toward it** (DECIDED 2026-11-16, section 1). PROPOSED form, so that it agrees
  with the rest: a subgoal is a condition on the graph, either the condition of a conditional claim (section 2) or
  an event of the rules (section 6). "Nearer" follows the two layers of reachable (next item): fewer moves of
  mine still needed, and a higher chance that the other side cannot stop it. Moves that
  raise that chance are tried first; the chances of section 6 decide among what was reached, so the subgoal still
  never adds to the value. Where subgoals come from is the next item.
- **Choosing a subgoal, and getting there** (DECIDED 2026-11-16; said in another session and brought over by him).
  - *Many goals, one to choose.* "My goal is to mate the opponent. There might be multiple mates. My goal is to get
    a material advantage. Multiple material advantages. Which one? In which direction? ... this is the material
    that's the most weak. So this is the highest probability of success ... And then I have to search for, okay,
    now what is the move that gets me to that outcome?"
  - *Something points at the weakness, with a reason.* "Something has to look at the position and basically point
    at the weakness for a particular reason." It works from the graph: "something needs to be able to use the
    graph to tell me what sub goal I should be pursuing and how to kind of do the means end analysis to get there."
  - *An imagined future, and whether it can be reached.* "We need to basically imagine a future where a threat is
    happening ... 3, 4, 5 moves in future is this situation ... which one of the ... situations is reachable from
    here?"
  - *No name is needed.* "Even if the mating pattern has no concept, the concept of attacking the king from this
    direction has ... The base concept exists and we can describe it and then we can build upon it ... This is
    pinned, that's under attack. This would deflect here and then we win that ... we don't need to have a name for
    it." So a goal is a set of facts of the graph with units as variables (section 3), not a named pattern.
  - *A problem, a proposed fix, and the fix becomes the subgoal.* "There's a problem and there's a proposer for a
    solution ... Becomes a sub goal itself, and we find a move that gets that sub goal."
  - *Subgoals must be graded.* "We could spend a long time pursuing bad sub goals. Like, how do we know this is a
    good sub goal? It's reachable, and how do we know we got the right mitigation for the problem? ... in
    evaluating sub goals, I don't think we ever got there."
- PROPOSED form of the above (the handoff's suggestion and the main agent's, not decided):
  - *Means-ends over facts.* A goal is a list of facts that must hold. Those that hold are ticked. Each one that
    does not hold is a problem; the proposer names what stops it and a fix (remove, pin, block or draw away the
    unit that stops it, the far end a variable); the fix is the next subgoal, and so on down to a move. Goals are
    compared by how few facts are unmet and by the chance of each.
  - *Reachable has two layers* (DECIDED 2026-11-16, late; asked whether reachable is a chance and not a distance,
    he said: "Reachable has two layers: me playing only, and including opponent veto").
    LAYER ONE, me playing only: can my own moves make the facts true if the other side does nothing against it,
    and in how many of my moves. LAYER TWO, with the opponent's veto: can the other side stop it.
    READING (mine) of how the two fit the rest: layer one is a count of my own moves over the graph, so here a
    distance is the right thing, and it is cheap; a goal that fails layer one is dropped without asking layer two.
    Layer two is first a matter of reasoning: a veto found concretely ends the goal or names the next obstacle
    (section 4). Where no veto is found and none is ruled out, what is left is a chance, which is where
    docs/THESIS.md's "reachability against a player who answers is a probability" applies. His "quasi-metric
    type model" fits layer one; his "quasi-metric is not the right way" (below) was said of the policy's model.
  - *Each subgoal makes four claims, and each is graded:* it is worth having (if the facts held we would be better
    off); the fix is the right one (with the fix alone the goal works, and without it the goal does not); it can be
    reached; it is the best use of the moves. And a pursuit is dropped when the facts still unmet stop falling,
    when its chance falls, when our own estimate falls, or when another subgoal is better by a margin.
  - *Who referees.* In play, only our own reasoning and our own estimate (the wall of section 8). As a MEASURE of
    the proposer, outside play, the oracle may referee (section 9: a measure is not a fit).
  - *The pointing is done from the graph only* (DECIDED 2026-11-16, late). Asked whether a language model may
    point at the weakness, he said: "Graph only. Language is the last step to convert to human friendly if
    needed." So the proposer, the diagnosis and the choice of subgoal use the graph, the frame read and the heads.
    A language model may only turn a finished result into words for a reader; nothing it writes is read back by
    the wake or the sleep.
- MEASURED (plan 78.71, one run, directions only), and where the first build departs from the form above:
  - *The diagnosis holds.* With the named stopper alone taken out, the goal worked in about half the cases; with
    another unit taken out instead, in few. So "is this the right fix" can be refereed by our own reasoning.
  - *One fix is often too small.* Where several units stop a goal, taking out all of them repairs most failures.
    The fix must be allowed to be a set, worked on one at a time.
  - *The proposer points at too little.* It builds goals only from rows of the threat list. Of the units then won
    in the games, few were among its goals, fewer than among units of the other side drawn at random. Its lists
    are short, so the order within a list could not be told from chance. Plan 78.74 then built a goal for every
    unit of the other side and the goal unit: its first three goals hold the unit then won more often than
    chance, and ranking by stake alone is worse than chance.
  - *Departures of the build from this section, to be mended or decided:* the facts come from a fixed list of five
    relations, not any set of facts of the graph; one edit, taking the unit out, stands for remove, pin, block and
    draw away; reach is a table of how often the facts came true, by cause, because the event heads are too weak
    to give it, and it does not yet separate the two layers of reachable.
- MEASURED (plan 78.78, one run, directions only): the goals as a hint among the moves the search holds equal,
  scored on the open surprises of the ladder. Layer one's count is undefined for most goals and the hint is silent
  in most tie positions. On the count of stoppers an accepted move advances a goal no more often than a move that
  is not accepted. Read against the board, the accepted quiet moves mostly keep our own units safe, stop a goal
  of the other side, or walk a unit a long way toward a listed goal.
  PROPOSED (mine, not decided): the proposer is also run for the other side, and a move of ours that adds a
  stopper to one of its goals is a defence, the opponent's veto of the two layers read from the other chair; and
  layer one counts own moves along the graph with no cut at two. Plan 78.79 measures both.
  MEASURED (plan 78.79, one run, directions only): the proposer run for the other side lists takings only, and a
  move of ours costs those goals as often when it is not accepted as when it is; the same for the stoppers with a
  reply read. The long count is undecided (a gap on one part of the split, none on the other). So the proposal
  above is not supported as built. The purposes read off the cases are further kinds each time, and listing them
  by hand is not what this section asks for; the open question is in section 13.
- **Subgoals as symbols that go with winning, and whether one is ahead of us** (2026-11-17). His idea: "if we
  quantize the GNN outputs from the statistical head, then we'll have some symbols correlate with winning
  positions and some with losing, then we can make the achievement of that symbol the subgoal; but we'll have to
  know if its in the future or not, so some sort of quantized quasimetric reachability?" The main agent's
  recommendation, which he accepted ("sure your rec is good"), DECIDED as the order of work:
  - *Fact patterns first.* The graph is already discrete, so the first symbols are small patterns of graph facts
    with units as variables (the layer-2 patterns of plan 78.53), local to a unit or a few units. They are
    readable as they come. A graph network with quantized codes per unit is built only if no pattern passes the
    tests below; its codes would then be described in graph facts.
  - *A symbol must pass two tests, on games it was not found on.* It goes with the result beyond what the count
    of units already says. And the move played in human games makes it true more often than the other moves the
    search holds equal do. The second test keeps out symbols that only mark a side already ahead.
  - *Not the output.* Symbols are never bins of the head's one number; reaching a better bin would be the value
    under another name (AGENTS.md rule 7). The symbol is a hint that orders moves.
  - MEASURED (plan 78.80, one run, a direction): patterns of sizes one and two pass both tests in the hundreds
    where a handful would by chance, and the pick made from them is the first hint to separate from the search's
    own first move. What passes is not yet what a player would name as a goal; see section 12.
  - MEASURED (plan 78.82, one run each, a direction): as the hint among equal moves in play, the reasons of plan
    78.81 scored above half against the control and against a random pick, the first hint to do so. What was read
    was opening play only; see section 12.
  - *Whether it is in the future follows the two layers.* Layer one, me playing only, is a count of my own moves
    until the pattern holds, where a quasi-metric fits and the earlier work on one may be reused. Layer two, with
    the opponent's veto, is a chance: the pattern coming true is one more event class of the heads of section 6.
- **The future imagination layer** (direction given by Kaveh 2026-11-19; a build is not yet agreed). "We do need some sort of a future imagination layer."
  It is for positions with no concrete tactic for either side: it "should help us to evaluate futures that are not directly a response to some
  sort of a threat ... or a tactic that's directly executable." The goals are "future states that are good, future states that I can predict,
  that include some high chance of threat. The higher the concentration of threat, the better." The steps, in his words: "a future imagining
  scenario, and then the evaluation of the threat landscape in this future scenario"; the landscape "could evolve from the current scenario,
  so it doesn't have to start from scratch"; "what is changing from here to get to that future scenario? What am I changing?"; then "what can
  they change on their side? Do they have enough time to change it?"
  READING (main agent, 2026-11-19, PROPOSED, not his word) of how it sits on what exists:
  - The scenario is layer one of "reachable" (above): a short run of my own moves with the other side standing still, kept as the list of facts it
    makes true and false, so the graph and the threat list are updated from the present ones and not rebuilt.
  - The landscape is read with the threat list of section 8 on the imagined position. A threat counts as the line reading counts it, after its
    best answer, and two threats that one answer cannot both meet count for more than two that one answer meets. No worth is assumed.
  - Their side is layer two: the count of their moves that undo a needed fact or meet the threats, set against the count of my moves the scenario
    takes; a scenario they can undo in fewer moves than it takes is dropped or names the next obstacle.
  - It enters as the order of quiet moves (the prior), never as the value (AGENTS.md rule 7): the move that starts the best scenario is read first
    or chosen among equal moves. A learned distance over the graph (the quasi-metric of layer one) is a later shortcut for the run of own moves,
    built only after the plain run is shown to mean something.
  - DECIDED by Kaveh 2026-11-19 (later the same day, answering section 13 item 10):
    WHAT A GOAL IS. "I think of a tactical situation as where two threats can be created at once where none can be answered at the same time.
    The path that every threat is responded is the draw situation ... So we're looking for threats that occur at the same time." Concentration
    is therefore counted by answers: threats that no one reply meets.
    THE STAKE ORDERS GOALS. "The higher the stake of the threat, the better; i.e. pinning queen to king with bishop is best, then is pinning rook
    to king, then is pinning rook to queen." READING (main agent): the stake is what the line reading says is won after the best reply, in the
    judge's own learned worths; no worth is assumed for it (section 5).
    NOT THE LENGTH. "The beginning of the threat sequence is important, not the count of the sequence items or the distance from here." There is
    no fixed length of a run of own moves.
    THE BUDGET IS QUESTIONS. "If we cant find a concrete way to get to an advantage after evaluating N questions, then we move on. If we're in
    the middle of evaluating a frame (i.e. some exchange on a square maybe joined with intermezzo to another frame) and we can still make
    progress, we continue past N questions until we rule out the frame or we succeed and execute. After ruling out we stop and declare the
    subgoal currently too far, and try to break it down into a closer subgoal." N is not fixed by him; it is measured (plan 78.94).
    THEIR SIDE TOO. "Opponents scenarios should be imagined as well. Once we find out our goal(s) and their goal(s) then during tactical
    question/answer and analysis we evaluate both goals to see if we can afford to pursue our goal or do we need to respond to their threats
    first."
    AMONG EQUAL MOVES. "Among equal moves in a tactical sense (i.e. two moves that both defend or both attack) we optimize for the prediction
    from #1, whichever gets us closer to our goal(s)." This is the chooser among the moves the search holds equal. It answers section 13
    items 6, 8 and 9: the choice is made by progress toward the goals, not by a control for takings and not by one-position patterns in the
    reading order. PROPOSED (main agent): the pattern reasons of plans 78.80 to 78.91 stay as the fallback where no goal of either side is found.
  - **The prior learns the tactical situations of the opening** (direction given by Kaveh 2026-11-19). "For openings, our prior should just learn
    to predict the tactical situations present in the openings guided by the oracle or querying human moves and learning probabilistic outcomes
    of frames as opposed to the best-play outcomes. This could probably be applied to the rest of the game too." READING (main agent): what is
    learned is the chance of each frame's outcomes as they happen in play, which is what the event heads of section 6 are for, counted per frame.
    SETTLED 2026-11-19 (section 9): in sleep the oracle may label the positions and frames the prior is trained on. As first written:
    CONFLICT TO SETTLE, not resolved here: "guided by the oracle" meets the standing rule of section 9 that no oracle number enters a fit except
    as the last resort with his word. Until he says this is that word, the prior is fitted on human moves and results only. See section 13.
  - HOW IT IS BUILT, three steps (direction given by Kaveh 2026-11-19, later still): "first we need to train some sort of legal future judge off of
    human game trajectories, that will give us a way to see if we can get to point b from point a legally with high accuracy but not perfect.
    Then we need to evaluate frames that lead to winning situations per the oracle from human games, then we need to predict whether we can get
    from point a to one of these future points where the winning frames exist or something."
    READING (main agent, PROPOSED, not his word):
    (1) The future judge is a learned model over the fact graph: given the present position and a target, the chance that the target comes
        true later in the game. Human trajectories hold the other side's replies, so what it learns is "reached in real play", which is more than
        legal and is the thing wanted. The target is a description in facts (a frame, a goal as defined above), never a whole position: a whole
        position of one game is almost never reached in another.
    (2) The points worth reaching are found in human games: positions where a goal as defined above holds (two threats no one reply meets) and
        what followed, as chances, not best play.
    (3) From the present position, the judge of (1) is asked for the goals of (2); the answer orders the equal moves.
    SETTLED 2026-11-19 (section 9): step two's winning and losing points are labelled with the oracle in sleep. As first written:
    CONFLICT TO SETTLE: "per the oracle" in step two. Under section 9 the oracle may give the best move in sleep and may be a measure; its number
    may not enter a fit except as the last resort with his word. Until he says so in those words, step two labels a winning point by the
    reasoner's own confirmed line reading and the game's result, and the oracle only measures how right those labels are. See section 13 item 12.
    This replaces the order of work of plan 78.94 as first written (an explicit run of own moves first): the labels come first, then the learned
    judge; the explicit run is kept as the check of the judge on a sample.
  - STILL OPEN after his answer: the size of N; how a goal that is "too far" is broken into a closer one beyond the proposer of section 7 above.
- **The form of the model** (direction given 2026-11-16, not settled). Modular: "a combination, mixture of experts
  kind of thing ... we need to be able to slot in different things to work side by side, maybe even in tandem."
  Not a quasi-metric: "it's a distance. What I want is like a sequence, something to model a sequence ... it's a
  kind of a Markov process ... we have a transition to the next state. So what's good for a Markov process? I'm
  actually not sure."
- PROPOSED answer: if the tactical state is Markov, the model is a TRANSITION model (the chance of each event of
  the rules at the next step given the present state, which the event heads already are) and a VALUE over it,
  learned from our own play and our own deeper reasoning (the same sources as section 6). The experts are the
  heads, one per event and side, gated by which frames are present. Whether the state is Markov is measured, not
  assumed: does adding earlier states improve the prediction of the next one. If it does, a short memory of earlier
  states is added as input.

**One way of asking for every goal** (DECIDED 2026-11-19). His words: "the idea that there should be like a special vocabulary for the line
to the king, opening up the line to the king. No, it should just generally be like, how do I get to the goal? What are the things that
stand in my way to getting to the goal? And then can I get rid of those things? So in the case of an attacker, it's either capture the
attacker, pin the attacker, block the attacker, defend the piece sometimes, not always. So like these are the things that you can do to
stop the attack, what's stopping your attack? What's stopping you from mating? What's stopping you from reaching your goal? Whatever
that sub goal is." So the goal unit's goal has no fact and no word that another goal lacks. A goal has facts; an unmet fact names what
stops it; each stopper that is a unit of the other side is itself a target with a goal of the same form, and that is the subgoal; the
chain is goal, what stops it, the subgoal that removes it. The defence is the same read from the other side: our answers to a goal of
theirs are the moves that make one of its facts unmet (take the bearer, tie it, put a unit between, add a holder, move the target),
named by the fact and by nothing else; whether one works is the line read's verdict. A move that is a fix under two goals at once is
how two threats are found. Plan 79.07 as amended.

## 8. Wake

1. Bring forward the open tactics carried from the last move, and update only those whose facts changed.
2. Split the position into frames and drop those with no bearing on the action.
3. Ask the questions inside the frames that matter, ordered by hints, by stake and by chance, with the lemmas and
   statistics sleep provided. New facts reorder the questions.
4. Where a tactic is confirmed, use it.
5. Where a line ends (section 4), read the chances for each side, after checking every open tactic whose stake is
   above the spread of the readings.
6. Where nothing is forced, the subgoals order what is tried and the chances decide among what was reached; name
   the subgoal.
7. Give the reasons for the move as a chain of claims, each line with the reason it ended.

**The rule for choosing a move** (DECIDED 2026-11-16, as a guide). It says how steps 3 to 6 fit together. His
words, in the order he gave them:
- "Evaluate the threats of the opponent. Whenever there is no threat, we advance towards our sub goal. Whenever
  there is a threat, we defend the threat and advance towards our sub goal."
- "Sub-goals might become refuting opponent's threats when the threat is big. Otherwise, if it's small, then we
  might consider sacrifices, things like that, to go crazy and attack."
- "I'm not saying like this is how it should really go, but because at every point something is threatened. It
  might not be immediately apparent now what it is, and so that's when we use statistics. When it is immediately
  apparent what the threat is, then we think through the response."

Put together:
- **The threats are read first**, theirs and ours, as the threat list below. A confirmed tactic of ours is used
  (step 4; section 7, "recognise when they are there and use them").
- **Answering a threat is itself a subgoal**, and it competes with the others by stake. A big threat of theirs:
  refuting it is the subgoal of the moment, and among the answers those that also serve our own subgoal are tried
  first. A small one: it may be left standing and our own subgoal goes on, an attack by sacrifice included. A
  sacrifice still needs one of the three reasons of section 3. A threat that already has a response needs no new
  move; the response is carried.
- **No threat apparent does not mean no threat.** An apparent threat is met by concrete reasoning (section 4). One
  not yet apparent is met by statistics (section 6). READING (mine): the event heads are that reading, the chance
  that an event such as a unit being taken happens to a side, with no line shown. While budget remains they point
  the questions at where the unseen threat most likely is, as a hint; when the reasoning has ended they are part
  of the belief the leaf is scored by. This keeps "concrete first" of section 6.
- **"Advance toward the subgoal" means what step 6 says**: the subgoal orders what is tried, the chances decide
  among what was reached (section 7). The subgoal does not score the position.
- **It is a guide to the shape of the thinking, not a fixed procedure.**

**A counter-threat is an allowed answer** (DECIDED 2026-11-16, late; asked "is a counter-threat an allowed answer
to a threat", he said "Yes"). An answer need not be a defence: a threat of ours with a larger stake than theirs
is also an answer. PROPOSED (mine, the condition): it is allowed only when the reasoning shows it concretely,
never on the estimate alone.

**The threat list: what any position must be able to answer** (DECIDED 2026-11-16). "In any given position, I want
us to be able to answer what are the threats and what is the stake there." A threat that wins in one frame and is
answered when the frame is widened is "a threat with a response". "The analysis should list out threats with their
stake and the response", ordered so:

| Stake | Response | Where it goes |
|---|---|---|
| High | None found | Top. "Those are the ones that are going to kill us immediately": time is spent here first |
| High | Found | Shown: "if the opponent does this, this is what we're going to do" |
| Low | None found | Worked on after the first row; "likely to be found throughout the game, and we will likely find answers" |
| Low | Found | Bottom: "we probably don't want to see them" |

PROPOSED: the line between high and low is not a number put in. A stake is high when it is above the spread of the
two readings of section 6 at that position, the same line that decides whether an open tactic must be checked
before the estimate is trusted. "Big" and "small" in the rule for choosing a move are these bands.
MEASURED (plan 78.69): comparing the stake with the spread added little beyond the claim being present, with the
present heads. So plan 78.70 prints the bands two ways, by this line and by the stake ranked within the position,
and the one that separates what is then lost is kept.
MEASURED (plans 78.70, 78.72): by neither line is the top band the most dangerous. The line between high and low
stays not shown.
MEASURED (plan 78.75): as a tie-break among moves the search holds equal the threat list almost never decides,
because a real threat leaves the search one best move. Its use is where the time goes and what is carried; that
test, cost at equal strength, is owed. The list reads takings only; a unit about to change kind is not a row.

Each row names the frame it was read in, the stake in the one unit of section 4, the response with its reason, and
how the line ended. The list is made for both sides (their threats against us, ours against them). It is the same
object the reasoning works from: the first and third rows are where the budget goes, the second row is what is
carried forward as open tactics, and the list is the readable form of the chain of claims. Our threats against
them that do not work yet, each with the fact that stops it, are the first source of subgoals (section 7).

**The wall.** No number of the oracle is read by the wake or by anything the wake reads. The one exception is the
last resort of section 9, which needs his word and has not been taken.
Since 2026-11-19 (section 9): a prior trained in sleep on frames the oracle helped to label is read by the wake; the oracle's own numbers
still are not.

**The whole is judged, not the piece** (DECIDED 2026-11-19). His words: "all of those things that I told you to build that you did not build,
that you shut off, those all need to be folded in. And then we have to evaluate the thing with all the pieces working together. If you
evaluate every single piece individually, it's not going to work. This is a system that requires all the pieces." So the wake is one
decision in which every step above is called: the threat list of both sides, the goals with their facts and fixes (the goal unit's
first), the question read, the search, the choice among equal moves, the chain of claims. The measure of the engine is that whole,
on the board, on puzzles as a measure, and in play. How a piece enters is unchanged: threats and goals order what is read and say
what must be read in full; they never add to the value (rule 7); the wall stands. READING of the main agent, kept until he says
otherwise: once the whole is measured, a piece is examined by taking it OUT of the whole, never by adding it alone to the bare
search; a piece that costs too much time is made cheaper, not turned off. Plan 79.05.

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
| A deep reading from a leaf of the wake | What longer thought concludes there: a target for the heads |
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
- **Three sources of surprise.** The wake's own games (a claim that failed, a loss not foreseen); the oracle's best
  move that our tools do not pick; and (2026-11-16) our own statistics preferring a move our reasoning cannot
  explain, such as a sacrifice taken on the estimate alone. The share of moves decided by an unexplained estimate
  is printed for each sleep and should fall as the rules that explain them are found.
- **A rule is scored by how many surprises it resolves** (DECIDED 2026-11-16). A blind spot is missed on every move
  for the same reason, so it leaves a LIST of surprises with one cause. "One rule during the sleep phase, which
  covers the blind spot, will then fix all of the surprises at once and will give me a huge positive signal." And:
  "We have to basically look at each rule and how many surprises does it solve, and we kind of learn."
  So the sleep keeps the list of open surprises; a candidate rule, lemma or statistical input is scored by the
  count of surprises it resolves, less the count it creates, on surprises it was not made from; surprises are
  grouped by what they share so that a rule is proposed for a group, not for one position. This count is also the
  near signal the chooser lacked: surprises resolved per second, many times a sleep.
- **What the oracle may and may not do** (DECIDED 2026-11-16). It supplies the best move, and so the question.
  Fitting a head to the oracle is allowed only as a LAST RESORT: "For now, I think it can be fitted to the Oracle,
  but I would rather avoid that if possible. Like, if forced, I will do it." What he would rather: "fit it on the
  traces of our own reasoning ... predict the reasoning trace, and then from a leaf where we actually end the
  reasoning, we should be able to predict the rest of the reasoning trace and see where it goes"; or "do like a
  Monte Carlo and then predict the Monte Carlo output." His ground for it: "Nobody actually knows. So Monte Carlo
  is the best bet, I guess."
- **The oracle labels positions and frames in sleep, for the prior** (DECIDED by Kaveh 2026-11-19). "The oracle is stockfish and in sleep cycles
  it is available for any questions you may want. During sleep, you may ask the oracle to help you identify winning positions and losing
  positions and the frames, then you want to use the frames to train your prior."
  What this changes: in sleep the oracle may be asked anything, and its answers (which positions are winning or losing, which frames are there)
  may make the labels the PRIOR is trained on. This settles section 13 item 12.
  What it does not change, READING (main agent) kept until he says otherwise: the wake never calls the oracle and reads none of its numbers,
  only a prior trained in sleep; the value (the estimate at a leaf) is still fitted on results and not on the oracle (his decision of
  2026-11-15, and the last resort above stands for the heads of the value); the prior only orders and chooses among equal moves, it never
  scores (AGENTS.md rule 7); what is learned is the frame in graph facts, so that every label can be read and checked against the board.
- **So the heads learn from our own deeper thinking** (PROPOSED form of the above). In sleep there is time to read
  far deeper than the wake can. From a leaf where the wake's reading stops, the target is what the deep reading
  concludes where it reaches a settled, concrete end, and the statistics of playouts by our own player where it
  does not. The head then predicts "what I would conclude if I thought longer". It stays tied to the truth by the
  ends of games and by proved lemmas, never by an engine's number. The oracle remains the finder of questions and a
  MEASURE of the heads on held-out positions. If the heads stall, the last resort is taken up with his word.
  MEASURED 2026-11-16 (plan 78.68 stages G and H, one run on a busy machine, a direction only): heads of the same
  form were fitted on our own deeper reading and playouts and read at the settled end of each tied move's line.
  On the part of the ladder's list no fit saw, no candidate separated from the search's own first move, at the
  root or at the leaf, whatever it was fitted on. The fit lowered the loss a little on playout rows and did not
  change the picks. READING (mine): the source of the target is not the weak part. The inputs at the leaf, kinds
  and frames, cannot tell two quiet moves apart, the line's end is mostly two to four plies on, and the line
  search judges by coarse worths. This is not yet "the heads stall" in the sense of the last resort, since the
  inputs have not been widened; see section 13 item 3.
- **When it wakes.** "No more than 5 hours", and "it could be more than 5 actions". One hour at most an action; the
  ceiling stays "unless we see some evidence that that's harming". He left the length to be scaled down ("maybe you need like a 5-minute sleep"); PROPOSED: a test at ten minutes, then
  an hour, the length then read from the printed curve of gain against time.
- **How it chooses with few, late rewards** (agreed 2026-11-15: "Let's go with your plan", sharpened 2026-11-16):
  each action is scored by a near signal, which is the count of the rule above: surprises resolved less surprises
  created, per second, on surprises the action's product was not made from. Actions are small, so a sleep holds
  many choices, and the record of what resolved which group of surprises carries across sleeps. Where an action's
  gain can be computed beforehand it is, and the computed number is checked against what was observed (in the two
  runs so far it did not predict it). The wake's match answers one question only, whether surprises resolved turn
  into better play; it decides whether the sleep's additions are kept, and the rolling measure below watches the
  longer run. Later, untested: a learned chooser trained on small games.
- **Both halves improve each cycle.** Sleep looks at which of tactics and policy is weaker and strengthens that.
  Tactics first, since steering toward opportunity is worth nothing without tactics to use it. The counts per rung
  of the ladder are that diagnosis.
- **What it produces.** Lemmas, statistics (the heads), vocabulary, the policy.
- **Watching for regress, and going back** (DECIDED 2026-11-16). "Some sort of a rolling window statistical measure
  of how many centipawns we lose or something, and then we see over time if we're getting worse or we're getting
  better." And: "we need the ability to revert the final set of things that were added, that were learned, so that
  we can go to a past learning state ... if we see some sort of regression, we can go back and say, okay, why did we
  regress here?" So each sleep's additions are one set that can be taken back, the store keeps its past states, and
  a rolling measure of play is kept across sleeps. The oracle may serve as that measure; a measure is not a fit.
- **Do not break the rest, and do not retest everything.** The count of surprises CREATED by a candidate is the
  first guard, and going back is the last. Between them, this candidate: proved lemmas cannot be broken by later
  ones; fitted things are rechecked only where the dependency memory says the new item reaches; each explained
  surprise stays on the list as a standing test case.
- **No outside language model inside the learning loop or the decision.** Language is only the last step, to put a finished result into words for a reader (2026-11-16). Statements come from the miner, proofs from the prover.

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
- **New Rust dependencies are allowed if they are well maintained** (2026-11-16). The crate, its version and the
  reason are stated in the plan note.
- **The work is pushed to GitHub** (2026-11-16), with no AI trailer in any commit message.
- **Every build is checked against this spec** (2026-11-16). "The system is so big that agents keep forgetting what
  the big picture is, and they build in a way that diverges from the overall picture. So we want it to be modular
  with clean interfaces and according to the overall high-level spec." Each piece of this document is a module.
  A plan names the section and piece it serves and the interface it adds: what it takes, what it returns, who calls
  it. Pieces talk only through such interfaces. A build that departs from the spec is not merged quietly: it is
  fixed, or the departure is brought to Kaveh. The rule for builders is `AGENTS.md` rule 12 and
  `docs/AGENT_BRIEF.md` section 0.

- **Every piece that reasons is judged against the board by the building agent** (2026-11-16, section 1). A build
  that prints reasoning (threats, answers, goals, a line of thought) prints it for a fixed set of positions; the
  builder reads some of them against the board and says what a strong player would not agree with, and the main
  agent reads them again before the merge. A piece is not called done on its numbers alone. The judgement is
  written in the plan note as words, never as a number, and never enters the engine.
## 12. What exists

The three inventories (the repo against these pieces; the earlier frame and intermezzo work; the October Lean
library against the game-free graph) are in docs/SYSTEM_SPEC_HISTORY.md, sections 9.1, 12.1 and 12.2, as made on
2026-11-14 and 2026-11-15. One correction to the third: Reach.lean is chess-bound, not generic.

State on 2026-11-16, in words only (plans 78.52 to 78.75 in docs/ANALYST_PLAN.md hold the runs and the plans):

| Piece | State |
|---|---|
| Graph layer 1 from the rules; layer 2 concepts selected by what becomes of their units | Built |
| Frames bound to units, the question read, the settled end, the exchange mirrored from Lean | Built; not yet a few questions a frame |
| The Lean core with chess as an instance | Built; three results not reached |
| No assumed worth: generations, odds games | Built; only the top kind is settled |
| The heads: result, kinds, arrival, events with no horizon, frames as inputs | Built as measurements; weak; not yet a leaf reading to lean on |
| The player reading a head as tie-break among moves the search holds equal | Built as a measurement (plan 78.67); the cheap arrival number is level with the frame heads and far cheaper; not switched on |
| The sleep loop with a chooser | Built; lost twice to a fixed schedule on a menu of one kind; parked until the menu of section 9 exists |
| The ladder (rungs 1 to 3) and the list of open surprises with groups | Built as a measurement (plan 78.68, first run). Of the positions tried, our reasoning alone reaches an accepted move in under two in five, nearly a third stay open at the longest clock, and those are mostly quiet moves in full positions the leaf's judge cannot tell apart. The heads as tie-break add nothing over the search's own first move. The groups do not yet separate causes |
| Heads fitted on our own deeper reading and playouts, read at the leaf; the score of a rule (surprises resolved less created on an unseen part) | Built as a measurement (plan 78.68 stages G and H). One run: no candidate separates from the search's own first move; fitting on our own deeper reading did not change the picks. The weak part is what the leaf's inputs can see, then the coarse worths |
| The ladder under each measured worth record | In PLAY the record matters for the choice among equal moves (plan 78.98, see the row on the future imagination layer; which record judges is section 13 item 18). On the ladder: Measured (plan 78.76): the odds record, the self-play record and the human-result record give the same division within noise; the worths are not the main cause of the open surprises. Open ones are positions where nothing is taken inside the reading. CORRECTED by plan 78.77: a unit changing kind IS an event of the ledger (since plan 78.5) and only the print left it out; the line read and the search are two readings at different depths, and the printed line now follows the search's move |
| The goals as a hint among equal moves, scored on the open surprises | Measured (plan 78.78): silent in most tie positions by layer one; no signal above the base rate by stoppers. Missing: goals that stop the other side's goals, and a count for long walks |
| The other side's goals as ours to stop, the long count, the replied stoppers | Measured (plan 78.79): defence and replied stoppers show no gap between accepted and not accepted moves; the long count is undecided |
| The search's equal reading and the line reading's verdict | DEBT (plan 78.84, one run). Seen again in plans 78.96 and 78.97 as an unequal exchange held equal with quiet moves; plan 78.98 traces the cause. The equal reading sometimes holds a move equal whose exact value at the same depth is under the best, in a small share of decisions, cause not known. The line verdict taken from events disagrees with the line's own value in most of the cases flagged as giving up a unit (the line ends inside an exchange, a unit changing kind is not in the net); the ladder and the threat list read that verdict. Plan 78.85 (one run): the cause of the first is the search's own forward pruning, a late quiet reply read shallower and the null move, not the table. The equal reading can now ask each candidate again without them (a switch, OFF by default: at the short clock it is cut most of the time, leaves one move in most sets and plays level). The verdict now counts a unit changing kind and reads on to a settled end or says UNSETTLED; events and value agree in nearly all the flagged cases. Plan 78.92 (one run each): the move the search plays is under the best of the same search without the prunings, at the same depth, in roughly one position in seven; the cause is the late-move reduction, the null move does nothing; most shortfalls are smaller than the smallest worth (unequal exchanges of the judge's table). The late reduction buys about two plies and a score not separated from a half. A switch verify_choice (OFF by default) asks the chosen move again without the prunings; in play the verifying player scored above the one without, one run; it cannot see the cases where the choice holds and a better move exists. Plan 78.93: repeated on two new opening sets and at five times the clock, the same way every time (the machine was busy, so the sign more than the size); the late reduction level with its absence again at both clocks for two plies more; a variant taking the verification's time from the main clock is worse; best_move now reaches the search with verify_ties. Owed: why about one replaced choice in forty moves the score, and why two more plies buy nothing. Turning it on in play waits for Kaveh (section 13 item 13). Open: every line read carries the same pruning; a cheaper sound equal question; Plan 78.86 (one run): read again with the repaired verdict, the ladder's open list and its groups did not move, so the earlier readings stand; a position's rung depends on the chosen move, the verdict only labels the line. Two faults left in the verdict: the settled tail does not see the end of a game, and a quiet line with net zero is called REFUTED. Plan 78.87: both closed; the tail counts the end of the game as the search does and stands when every taking is worse, the verdict has the word QUIET, and a line that takes a unit and then loses the game is read as lost. Left: QUIET does not tell a repeating line from a wandering one; UNSETTLED is still handled as a threat by two callers |
| Symbols that go with winning as subgoals (fact patterns, two tests, a wider yardstick) | Built as a measurement (plans 78.80, 78.81). One joint weight per reason, fitted on the choice of the move played among equal moves, with controls. On unseen positions the pick agrees with the move played clearly more than the search's first move, all of it where no equal move takes a unit; where one does it is worse. On the ladder's ties the hit is higher, the net band holds zero. About twelve of the strongest twenty reasons read as purposes, in five ideas. Layer one has a first reading, a weak signal. Played (plan 78.82): the pick among equal moves scored above half against the control on both opening sets, bands clear of half, while a random pick scored level with the control; charging the reading time changed little. First hint told from the control in play. The thirty decisions read were all openings, most a developing or central move in place of a wing-pawn step; the stated reasons are many small mobility patterns, not one purpose. The control's order among equals is no better than random. By part of the game (plan 78.83, one run each): the score is earned while many units are on the board; the middle band is a little above half and the few band at or just under half, so in play the reasons act as an opening development rule. The picks judged worse give up a unit to the next reply, and the search had held those moves equal to sound ones. The pick's cost is the pattern count. Measured (plan 78.84): most such flags are artifacts of the line verdict's events; a real fault of the equal reading is there but small; a cut of the tie set by the line reading cost more than the move's clock and did not lift play. On tie sets verified without the pruning the hint scored level with before (plan 78.85). CORRECTION (2026-11-18): the many band is seventeen units or more, the opening and most of the middle game, so "an opening development rule" above is not shown; the decisions read were first decisions of games by rule. Plan 78.88 (one run): by band the pick agrees with the move played about equally often, but the search's first move agrees more as units come off, so the edge is clear in the many band and in quiet mid positions only; grouping twins and splitting the unit borne on by guarded or not moved agreement about a point; weights by part could not be tested on the thin later-game sample. Plan 78.89 (one run each): the hint is not an opening rule; the lower part of the many band alone played above half with the band clear of it, above the upper part. A second human sample covers the later game with no generated fill. On it the search's first move agrees with the humans more than the pick with sixteen units and fewer; the refit lifts agreement there a little and does not lift play; weights by band are no better than one fit. Missing: a reason for a unit of the mover's left borne on and unguarded; the pick where a tied move takes a unit. Plan 78.90 (one run each): cells for the other side's bearing on the mover's own unit took the weights a player would expect; agreement rose with few units on the later-game sample only; one fit on both samples is still under the old weights in the lower many part; in play above half against the control and not above half against 78.82's player. A pattern of one position cannot tell a unit lost from a unit exchanged. Plan 78.91 (one run each): given to the search as a root order the reasons saved no nodes and the search's own short reading orders moves far better; in play the same as the pick, level when its cost is charged. The hint stays a choice among equal moves; the count in Rust for inner nodes is not justified. The line is at rest until section 13 items 6 and 8 are answered. Open: twins, bearing on a guarded unit, no place named, takings, layer two, the pick's cost |
| The menu, compression (rung 4), the chooser | Written, not built |
| Quiet or sharp by two readings; open tactics checked before the estimate; carrying measured | Measured once (plan 78.69): the difference of the readings not shown as a signal with the present heads; open claims of the frame read go with events soon; carrying first measured by a proxy, then truly (plans 78.70, 78.72): a claim whose read set did not change is held again only about half the time |
| The threat list (threats with stake and response, in four bands) | Built concretely (plan 78.72, after a first form in 78.70): each row and each response is a line read to a settled end and judged by the measured worths; exchanges print apart; claims nothing bears on are dropped; the end of the game is a row for either side. Like for like cancels in the verdict and responses carry their own end and are ordered (plan 78.73). Still short of section 8: a clear gain can print as an exchange because the measured worths are coarse, many positions print no row, threats that need a quiet first move are rare, and the top band is not the most dangerous. Carrying by the frame's read set does not hold for a concrete line; the line's own read set is not built. Not wired into play |
| The future imagination layer (section 7, 2026-11-19) | Not built; decided the same day what a goal is (two threats no one reply meets), that stake orders goals, that the budget is questions and that both sides are imagined. MEASURED (plan 78.94, one run): built as a measurement, with labels from the oracle in sleep and a future judge on the estimator's inputs. A goal as defined is rare in human games and pays where it holds; almost none of the frames at a winning point are two threats at once (mostly a unit left to be taken, or one threat); the judge predicts a goal ahead better than the unit count and the patterns, a winning point only a little better; as a chooser it is level with the pattern pick overall and better where a tied move takes a unit; a goal some own moves away was spoiled by the reply every time. Doubts of measurement and play: plan 78.95. Plan 78.95 (one run): with every reply tried about half of those goals fall, though that test overcorrects (a reply that gives up more elsewhere counts as meeting a threat), so the two counts bound the truth; at winning positions that hold, the frame is mostly a unit left to be taken, one threat or nothing named; the search plays the oracle's first move or holds it equal in about three cases in four, the line reading confirms the gain in two in five, the threat list names it in one in five; heads by kind of frame predict ahead better than the unit count; a chooser weighted by kind is no gain; the plain joining of judge and patterns agrees more with the human's move but in play is level with 78.82's player. Three choosers among equal moves now score alike. Next: plan 78.96, the oracle's class of each equal move, and a prior fitted on which one keeps it. Plan 78.96 (one run): the equal moves differ in the oracle's class at about one sampled position in eight and some equal move nearly always keeps it; every chooser takes the keeping move more often than random and none nearly as often as it is there; a prior fitted on those labels is a little ahead of the pattern pick on unseen games and level with 78.82's player in play. Four choosers among equal moves are level in play. Many of the losing moves in the equal sets are not equal to a player, so the next question is the reasoner's equal reading: plan 78.97. Plan 78.97 (one run): a much longer clock or the prunings off take only a part of the class-losing moves out of the equal set, at a price in good moves; most stay at every setting. The verified choice replaces few choices and most of those go from losing the class to keeping it. On the board the clearest gap is a move that gives up a stronger unit for a weaker one, with every reply seen by our own reading, still held equal and played first. Cause and mend: plan 78.98. Plan 78.98 (one run): the cause in about five cases in six is that the judge's value of the two lines is the same; the judge in the equal reading is the record of worths summed over the takings, and the record in use (the odds record) orders three of the kinds otherwise than a player does, inside its own half widths. Under the self-play or the human-result record about a third of those moves leave the equal sets. Measured in play: plan 78.99. The default record is unchanged. Plan 78.99 (one run each): both other records clearly ahead of the odds record in play on both opening sets, level with each other, and the pick of 78.82 keeps its gain on top. What the record, the verified choice and the pick are worth together, and what room is left: plan 79.00. Plan 79.00 (one run each): the three do not add; the verified choice is a little ahead without a pick and level on top of it; the whole is far ahead of the plain search under the odds record, also when that search has five times the clock. Under the stack the equal moves still differ in the oracle's class at about one sampled position in ten; most of what is left is moves the unit count cannot tell apart (no taking in the line, or a like exchange), a quarter is the debt of plan 78.84, and many labels sit on the class edge. A firm label and the result-fitted value read at the end of the line: plan 79.01. Plan 79.01 (one run): a firm label (the class repeats on a second asking and is a margin from the edge) keeps about three quarters of the room; in it the first move and the pick keep the class a little over half the time and a random equal move a little under. The result-fitted head on the kinds read at the end of the line is no better than the first move, since on equal units it adds only a draw term and a bonus for the side to move; no second key is built. Moves whose own line at the same depth is worth far less still sit in the equal set, and the equal reading is often cut by the clock: plan 79.02. Plan 79.02 (one run): a cut drops moves from the equal set and never adds one; the played move is the only one not asked. Moves held only by a reduced reading are about one firmly losing equal move in nine; a switch (reask, off by default) that asks held moves again unreduced inside the slot puts some out and does not move the choice, so it stays off. Most losing equal moves are held plainly: the value's inputs. In some positions a root move's full value at the finished depth is above the search's reported value: plan 79.03. Plan 79.03 (one run): a reduced root move above the window is already read again in full; the value is under the best root move's own full value in about one sampled position in twenty, mostly through the late-move reduction, a price of depth. A switch (root_full, off by default) lowers that by about a quarter at about a ply of depth and does not move the class kept, so it stays off. The reasoner at this clock is sound enough; what is left is the value's inputs. The existing result-fitted heads with statements as inputs, in the firm room: plan 79.04. Plan 79.04 (one run): read from both sides' view, the heads with statements separate the equal moves in nearly every position and pick about as well as the pick of 78.82, not clearly better; the heads on kinds pick below the first move. Six statements about contact are zero on every quiet row the heads were fitted on and so could not be learned, though they differ between keeping and losing moves. How the estimator is fitted next is section 13 item 23; the line waits there. Pieces it would stand on exist: the threat list, the line reading, the count of own moves along the graph (plans 78.78, 78.79, undefined for most goals as built) |
| The whole wake, every piece of section 8 called in one decision and judged as one (2026-11-19) | Not built. Until now the player was the search with a count of worths at its end; every other piece was a measurement scored alone as a tie-break and left off. Plan 79.05. Plan 79.05 first part (2026-11-19): BEGUN IN RUST. The wake's interface is written with every piece declared; the readings of the graph onto a goal unit (a line one unit shuts, who bears on that unit and who holds it; the moves that bring a bearer), the verdict of a line, and moves named by position read first and in full by the search are in Rust, off by default. Plan 79.06: the concrete threat list, the line read of every unit and the stake's arithmetic are in Rust (cs_decision::threats), agreeing with the Python on small constructed positions; too slow as it stands because every unit is read without a question. Not yet ported: the proposer and the goals of two threats, which also bound that reading by stake and a budget (79.07), the pick, the heads, the carried tactics and the wake's loop (79.08). Order of testing, his: Rust unit tests a piece, then integration of the whole, then positions |
| The frame's interface (ports for the intermezzo, the pass, the deflection) | Not built beyond the named causes of plan 78.60 |
| The rolling measure across sleeps and going back to a past learning state | Not built; snapshots of the store exist |
| The opportunity policy, the subgoal vocabulary, holding a subgoal across moves, the rule for choosing a move | The proposer is built as a measurement (plan 78.71 A to C): goals as facts with the cause and the fix, the fix checked as a what-if edit by our own reasoning, a first grading by hindsight. The diagnosis holds. A goal for every unit of the other side and the goal unit, the fix as a set, reach carried in two layers (plan 78.74): the first three goals hold the unit then won more often than chance; the printed order is not yet a player's (goals through several units between rank first); layer one is crude. The rule for choosing a move is built as a sandbox player (plan 78.75) that picks among moves the search holds equal: in matches it was no better than the search's own choice, a random pick or the arrival number, at half a second a decision. The threat half hardly ever decided, since a real threat leaves no tie; the goal half changed most choices without gain; a goal was held about one decision. A decision took about twenty line queries. The policy as a learned part is not built |
| Earlier work on subgoals to be reused, all on older systems | The what-if edits (catspace/incubator/analyst/whatif*.py); the October vocabulary of why a fix fails; the subgoal former and the waypoint decomposer of the summer (docs/SUBGOALFORMER.md, subgoal_cascade), built on a learned embedding; the October what-if edits were bound to the game and to assumed worths, so only their idea was reused in plan 78.71; the subgoal fit and play of plans 78.49 and 78.50. Subgoals themselves were never graded |
| The unknown as a condition in the frame read; the sequence model | Not built |
| The endgame fill from generated games | Built and exercised once on a throwaway copy (plan 78.89); not used in any fit or score, because the second human sample covered the later game without it |

## 13. Open questions for Kaveh

1. The exact content of a frame's interface (section 3): which ports, and what each carries.
2. Is the tactical state Markov enough (section 7)? To be measured, then put to him.
3. What counts as the heads having stalled, so that the last resort of fitting on the oracle is raised with him
   (section 9)? Proposed: the measure against the oracle's move not rising over three sleeps.
   Added 2026-11-16 after plan 78.68 G and H: the first heads on our own deeper reading did not move the picks.
   PROPOSED (mine): this does not count as stalled until the leaf's inputs have been widened and the worths
   refined, since both are named causes we can still work on without the oracle.
4. Where do subgoals come from, and how is one preferred to another (section 7)? Partly answered 2026-11-16:
   something points at the weakness for a reason, goals are facts with no name needed, the fix of a problem is
   the subgoal. Still open: the proposed form in section 7 (means-ends over facts, the four graded claims, when a
   pursuit is dropped), and the three departures of the first build listed there.
   Added 2026-11-17: may a goal be "stop one of theirs", made by running the proposer for the other side
   (PROPOSED in section 7, being measured in plan 78.79)?
5. Answered 2026-11-17 (section 7, symbols that go with winning): fact patterns first, tested on human moves,
   a graph network only if they fail. Still open: what is done if no pattern passes and the network's codes do
   not either.
6. May the hint among equal moves include that a move takes a unit and the way the moved unit moves (section 7,
   plan 78.81)? They are not reasons, and with them the pick is better where a taking is available.
   In play (plan 78.82, one run) the pick with them scored higher against the control than without. A direction, not a
   verdict.
7. Gaps found by plan 78.89 (2026-11-18), choices the spec is silent on, made by the builder and kept until you say otherwise:
   when the held-out human games run out, further games are taken from the same month file past the ranks already used;
   a generated game used as fill has as its "move played" the mover's own move, which lands among the equal moves far more often
   than a human's, so a filled bin would flatter any pick (no fill was needed or used so far);
   the fit a window plays with was chosen by its agreement on the unseen part, a mild selection.
8. Gap found by plan 78.90 (2026-11-19): the fact layer has no word for "can be taken back". A reason that sees a unit left borne on and unguarded
   reads an even exchange as a loss. My reading (PROPOSED, not your word): that question stays with the line reading of the
   reasoner and is not added to the patterns; the hint is kept to quiet choices and to the order of reading.
   MEASURED after this was written (plan 78.91, one run): in the order of reading the hint bought nothing, so the proposal reduces to "the hint is kept to quiet choices".
9. Gap found by plan 78.91 (2026-11-19): section 7 says subgoals enter through the prior. Measured, the reasons as a root order add nothing over the search's own
   shallow reading. Is "the prior" for these reasons the choice among equal moves and nothing more, or should the order be tried again with subgoals of
   another form (a target several moves away, not a pattern of one position)? My reading (PROPOSED): the second; patterns of one position are what a
   shallow reading already sees.
10. ANSWERED 2026-11-19 (section 7, the future imagination layer; items 6, 8 and 9 above are answered by the same words: among equal moves the
    choice is by progress toward the goals). The questions as they were asked:
    The future imagination layer (section 7, 2026-11-19): how is the concentration of threat counted; how long a run of own moves may a scenario be;
    are the other side's scenarios imagined the same way. And does my reading stand, that it is layer one for the scenario, the threat list for
    the landscape, layer two for their time, and enters as the order of quiet moves only.
11. From plan 78.92 (2026-11-19): may verify_choice be ON in play if plan 78.93 repeats the result on other openings? And three choices the spec is
    silent on, made by the builder and kept until you say otherwise: a chosen move that fails its verification is replaced by the first move
    of the search's own equal reading that holds, else kept; the verification's time comes from the equal reading's slot of the clock;
    the reference for a fault is the same search without the prunings, deepened step by step with a node limit and a fresh table.
12. ANSWERED 2026-11-19 (section 9: yes, in sleep, for the prior). (Raised twice on 2026-11-19: the opening prior, and "winning situations per the oracle" in the three steps of the imagination layer.)
    From his words of 2026-11-19 on the opening prior (section 7): is "guided by the oracle" the word that takes the last resort of section 9,
    so that an oracle's move may enter the fit of the prior? Until answered: human moves and results only.
14. From plan 78.94 (2026-11-19). A FINDING FOR YOU: two threats at once is rare in human games and is not what precedes most wins there; mostly one threat,
    or a unit left to be taken. Should a goal be widened to "a frame that pays as played", with two threats at once as its strong case and the
    chance of paying learned from human games? My reading (PROPOSED): yes, and it is what your words on probabilistic outcomes ask for.
    Choices the builder made where the spec is silent, kept until you say otherwise: the oracle at a fixed depth, a position called winning past
    a fixed edge of its score; every second ply sampled from the tenth; the frame read at the first, third and fifth ply of the oracle's line.
24. From 2026-11-19, his decision that the whole is judged (section 8). Not a question; what follows from it and what is mine. The
    line of item 23 (how the estimator is fitted) waits behind plan 79.05. Readings of the main agent, kept until you say otherwise:
    after the whole is measured a piece is examined by taking it out of the whole; the whole is compared with the old player at the
    same wall clock a position; the clock of the whole is whatever its steps need, and the cost by step is reported as work to do in
    Rust. One thing you floated and took back the same day, recorded so it is not lost: worths from the results of the oracle's own
    playouts in sleep instead of from human games of every level (it would change the sources of section 5; not taken).
26. From 2026-11-19, a question you asked and did not settle: "how do we spend effort chasing a mate that is very far off that we can't
    really reason our way around because there's just too many plies out, or do we chase the smaller stake that is more concrete?"
    What the spec already says (section 7, 2026-11-19): the stake orders goals, not the length; a budget of questions; past it only
    where progress is made; too far means break it down. The proposer's own default orders by the fewest unmet facts first, which is
    the opposite. PROPOSED by the main agent, not decided: two passes. The ASKING is in order of stake and costs almost nothing (the
    graph alone: how many things stand in the way of each goal). The READING, which is what costs, goes to the goal with the fewest
    things in its way, the larger stake breaking ties. A far goal of high stake is never read as a line; it is carried from move to
    move and gives the reading one thing only, its nearest stopper as a subgoal, which is small and concrete. So the far goal chooses
    WHICH small goals are worth reading, and the small goals get the reading.
25. From plan 79.06 (2026-11-19). Choices the builder made where the spec is silent, kept until you say otherwise: until the frame read
    is in Rust the rows of the threat list come from the graph (every unit an opposing unit bears on); the stake is read by a function
    the caller gives. Two behaviours carried over from the Python that look wrong and are to be settled at the integration tests: a
    response whose result only equals the threat's is called kept; a quiet first move that ends level is dropped, not an exchange.
23. From plan 79.04 (2026-11-19). A DECISION FOR YOU: how the estimator is fitted next. The heads were fitted on quiet positions only, so
    the statements about contact (what is borne on, and its worth) are silent on every fit row and carry no weight, while at the
    ends of the search's lines they are among what differs most between a move that keeps the oracle's class and one that loses
    it. PROPOSED by the main agent: fit the result head again on ALL positions of the same human games (results only, as now; no
    oracle number as a target), read from both sides' view, and measure it in the firm room and then in play as the chooser among
    equal moves, entering through the choice and not through the judge's value. What it would not give: races and forced lines,
    which a count of features does not see. Choices the builder made where the spec is silent: both views weigh the same; ratings
    are held at each head's centre at a leaf; a head is CLEARLY AHEAD when its band is wholly above both the first move and the pick.
22. From plan 79.03 (2026-11-19). A switch `root_full` now exists in the reasoner, off by default, not shown to help; I do not propose
    turning it on. Choices the builder made where the spec is silent: a FAULT is a gap of a unit of the weakest kind or more, or a
    value of the game's end; the reference is the full-window value on a fresh table with the prunings on below the root.
21. From plan 79.02 (2026-11-19). A switch `reask` now exists in the reasoner, off by default, not shown to help the choice; I do not
    propose turning it on. What is left of the room is mostly moves the judge's value cannot tell apart, which is the question of
    item 20 (the estimator's inputs) and is yours. Choices the builder made where the spec is silent: with the switch on, a held
    move the second asking does not reach stays in the set; the second asking follows the deepening's own order; the played
    move is not asked again; the binding's row gains a sixteenth element, empty when the switch is off.
20. From plan 79.01 (2026-11-19). The measure PROPOSED in item 19 (the result-fitted value at the end of the line as a second key) is
    NOT supported and is withdrawn. A note for section 6: the head on the kinds gives the side to move a small bonus, so read at the
    end of lines of different length it orders moves by parity; any use at a leaf must take that out. What a player sees in these
    positions (who moves first on which side, the cover of the strongest unit, a unit about to change kind) is not in that head's
    inputs; whether the estimator gets such inputs is yours to say (the open item on its heads). Choices the builder made where the
    spec is silent: FIRM means the class repeats on a second asking and both numbers are a margin inside the class, the margin the
    smallest at which ninety-nine in a hundred repeat on FIND; ties for the highest share the credit.
19. From plan 79.00 (2026-11-19). Sizes for two decisions of yours, not decisions. Item 18 (the record): record, verified choice and pick
    together are far ahead of the plain search under the odds record in play. Item 13 (the verified choice): on top of the pick it is
    level in play and spends more clock, so my recommendation CHANGES: off where the pick is on, on only for a search without a pick.
    PROPOSED by the main agent, a measure only until you say: the value fitted on results read at the end of the search's line as a
    second key under the record of worths (plan 79.01 measures it; nothing is built into the search). Choices the builder made where
    the spec is silent: a further match is played only when every set's band is wholly above a half; at most two losing moves a
    room position are traced; two clocks in one match go through a per-player key.
18. From plan 78.98 (2026-11-19). A DECISION FOR YOU: which measured record of worths judges the search. The one in use (from the odds games
    of plan 78.61) puts the second strongest kind below two weaker ones, inside its own stated uncertainty, so the search holds giving it up
    for one of them equal with quiet moves and often plays it. The records from self-play results and from human results order the kinds
    as a player does. All three are fitted on results; none is put in by hand. My reading (PROPOSED): the human-result record becomes the
    default if plan 78.99 shows it ahead in play. PLAN 78.99 SHOWS IT (and the self-play record) CLEARLY AHEAD, six matches of six. Until you say, the default stays.
    Choices the builder made where the spec is silent: the causes and their order (not held, the pruned value too low, a loss two plies
    past the depth, the judge's value the same); the longer reading is two plies.
17. From plan 78.97 (2026-11-19). Choices the builder made where the spec is silent, kept until you say otherwise: an equal move LEAVES when a
    setting no longer holds it equal; it is STUCK when it stays at every setting tried; a stuck loss is TACTICAL when units are lost within six
    plies of the oracle's line and SLOW otherwise (blunt, by the builder's own reading). Evidence for item 13: in the room the verified
    choice's replacements go mostly from a move that loses the class to one that keeps it.
16. From plan 78.96 (2026-11-19). Choices the builder made where the spec is silent, kept until you say otherwise: a move KEEPS the class when the
    oracle's class after it is not lower than before it (so a losing position has nothing to keep); the ROOM of a choice is a position where one
    equal move keeps the class and another does not. A finding that bears on item 13 (the verified choice in play): the sets the search calls
    equal at the play clock often hold a move that loses the class; plan 78.97 measures what the verified choice takes of that.
15. From plan 78.95 (2026-11-19). Choices the builder made where the spec is silent, kept until you say otherwise: a reply MEETS a threat when the line
    reading no longer confirms it afterwards (too weak: it counts a reply that loses more elsewhere; to be tightened to "and loses no more than the
    threat"); a winning point is STABLE when the class holds at the next sampled position of the same side; the KIND of a frame is read from
    the first plies of the oracle's stored line (two threats, forced end, one threat, unit left to be taken, none named); the STAKE is the
    material of that line in the record's worths; the CHANCE as played is how often the line's first move was played.
    A finding that bears on item 14: with the fairer counts the answer to it is still yes on the evidence.
13. Not understood when asked on 2026-11-19 and asked again in plain words: whether the careful form of the equal reading stays off in play
    (section 5, PROPOSED of 2026-11-18), and whether the check of the chosen move may be on in play if plan 78.93 repeats (item 11).

Answered on 2026-11-16 and written into their sections: the estimate scores a leaf only when the reasoning has
ended, and only orders before that (4, 6); the oracle as a last resort (9); the ceiling of an hour an action stays
(9); worth must come from experience (5); when frames are separate (3); regress watched by a rolling measure, with
the ability to go back (9); the policy is modular and models a sequence, not a distance (7); a line may end quiet while budget remains once
every open tactic is checked (4); a sacrifice needs a reason, and one taken on the estimate alone is a surprise for
the sleep (3, 9); the overall goal and its two halves (1); the rule for choosing a move, as a guide (8); the threat
list (8); every build checked against the spec (11); the pointing at a weakness is done from the graph only, language being the last step for a reader (7, 15); a counter-threat is an allowed answer to a threat (8); reachable has two layers, me playing only and with the opponent's veto (7); the language model is the judge of whether the engine reasons like a human, outside the engine (1, 11); the learned choosers are two, the sleep's and the opportunity policy, and the asking of questions has no model of its own (4, 7, 9).

## 14. What was replaced, so that no earlier statement misleads

| Earlier | Replaced by | When |
|---|---|---|
| The first sleep analyses positions bottom-up by size, three units, then four, then five | Concepts and frames emerge from games; the bottom-up pass is dropped. Reducing a larger frame to a smaller one survives as "the smallest frame that explains" | 2026-11-15 night |
| A piece is built as a measurement, scored alone against the search's first move, and stays off until it wins alone (the practice of plans 78.67 to 79.04, never a decision of his) | Every piece is folded into the wake and the whole is judged; a piece is examined by taking it out of the whole (section 8) | 2026-11-19 |
| A static worth of the kinds as the baseline, with corrections | No worth assumed at all; the baseline is itself measured | 2026-11-15 night |
| He floated conditioning concepts on relative value | Taken back the same day: raw kinds, "learn what happens when each piece is left behind" | 2026-11-15 |
| Chances fitted from the engine's own games only | Any source of results: human games, self-play, playouts | 2026-11-15 |
| Human game files not read until he asks | "Yes, human games can be read" | 2026-11-15 |
| The oracle gives "the evaluation of any position" to the sleep, with what crosses to the wake left open | In sleep it gives the best move, to be explained by the ladder. No oracle number in any fit or in the wake, except the last resort on his word (section 9) | 2026-11-15, 2026-11-16 |
| Opportunity as chance times stake, credited where it later became material | The chances at the leaf, per side, over events read from the rules | 2026-11-15 night, 2026-11-16 |
| A target within a fixed count of plies | An event and the time to it; no horizon | 2026-11-16 |
| A sleep wakes after five actions | No cap on actions; five hours at most | 2026-11-15 night |
| "Until it improves by a certain percent" as the stop | PROPOSED: the stop is gain per second or the time limit; the improvement is tested by the wake's match | 2026-11-15 night |
| The sleep's near signal as a held-out yardstick of loss | Surprises resolved less surprises created, per second | 2026-11-16 |
| A line ends only at the settled end or at a limit | Four stated reasons: concrete, quiet, lost beyond doubt, out of budget (section 4) | 2026-11-16 |
| "The edges will also be graph nodes" read as every relation being a node | He meant the board's edge. Decided separately: only relations something must point at are nodes | 2026-11-15 |
| A menu of odds games and fits only | The menu of section 9 | 2026-11-16 |
| The opportunity policy as a quasi-metric (floated) | "Quasi-metric is not the right way. Like it's a distance. What I want is ... something to model a sequence" | 2026-11-16 |
| "You already know how to reason like a human" read as: what the builder has read may shape the procedure (the main agent's reading) | He meant: the language model can judge whether the engine is doing its job, because it knows how humans reason | 2026-11-16 |
| No head is ever fitted to the oracle (the main agent's proposed boundary) | Allowed as a last resort; preferred: predict our own deeper reasoning and our playouts | 2026-11-16 |

- REPLACED 2026-11-19: "the hint among equal moves is the pattern reasons" (plans 78.80 to 78.91) as the intended chooser. The chooser among equal moves is progress toward the goals of the future imagination layer (section 7); the reasons are kept as a fallback (PROPOSED) and as a measured baseline.

- REPLACED 2026-11-19: "in sleep the oracle gives the best move only, and no oracle number enters any fit". In sleep the oracle may be asked anything and may label winning and losing positions and frames, and the prior is trained on those frames (section 9). The wall around the wake and the rule for the value stand.

## 15. Standing limits that apply to every plan

No number of the oracle in the wake, and none in a fit of the value unless the last resort of section 9 is taken on his word; in sleep the oracle may label winning and losing positions and frames for the training of the prior (2026-11-19). No
outside language model in the learning loop or in any decision; language only puts a finished result into words.
No assumed worth. Names are supplied, never knowledge of the game.
A subgoal orders and never adds to the value. In play a subgoal is refereed by our own reasoning and estimate only.
No game object named in a layer. No player id written. Lean before a lemma is relied on. The real store only on his
word. Endgame coverage printed for every human sample. Numbers only where a program printed them. Every line of
reasoning carries the reason it ended. A candidate rule is scored on surprises it was not made from.

## 16. Index: where each decision is written

| Decision or direction | Date | Section |
|---|---|---|
| The pieces of the system; the immediate use | 2026-11-14 | 1 |
| The overall goal: reason like a human, concrete tactical steps then an evaluation at their end; the language model as the judge of the engine's reasoning, outside the engine; the second half: hold subgoals and navigate toward them over several moves | 2026-11-16 | 1, 7 |
| One graph; three layers; dc names and the mapping file; which relations are nodes | 2026-11-14, 15 | 2 |
| Frames: units as variables, bound to units, few questions, the cut and the sacrifice through another frame, the intermezzo, the smallest frame, from human games, the chase work reused; the three reasons for a sacrifice; the frame's interface, the far end as a variable, exact on the smaller stake, two frames at once | 2026-11-15, 16 | 3 |
| Reasoning by questions; stake in one unit; hints order and facts reorder; the asking of questions has no model of its own, its count per decision is measured; open tactics carried forward; the four ways a line ends; chess is concrete; damaged structure as a subgoal | 2026-11-14, 15, 16 | 4 |
| No assumed worth; odds games; the unknown as a condition | 2026-11-15 | 5 |
| The estimator: a belief; parts as inputs; sources in order of preference; rating; several heads; chances for each side; events from the rules; no horizon; the sequence; the opponent; concrete first; quiet or sharp by two readings; open tactics checked first | 2026-11-14, 15, 16 | 6 |
| The opportunity policy; belief against hint (rule 7); maxims come out; subgoals; modular, a sequence model over a Markov state; a subgoal as a condition; reachable in two layers, me playing only and with the opponent's veto; choosing among goals by the weakness pointed at, goals as facts with no name, the fix of a problem as the subgoal, subgoals graded | 2026-11-14, 16 | 7 |
| The wake, step by step; the rule for choosing a move (threats first, answering a big threat is the subgoal, a small one may be left for an attack, the unseen threat read by statistics; a counter-threat is an allowed answer; a guide); the threat list with stake and response, high and low set by the spread; the wall | 2026-11-14, 16 | 8 |
| Sleep: the agent, the menu, the oracle and the ladder, a rule scored by the surprises it resolves, heads fitted on our own deeper reasoning and playouts (the oracle a last resort), the limits, how it chooses, the diagnosis, the rolling measure and going back to a past learning state, do not break the rest | 2026-11-14, 15, 16 | 9 |
| PROPOSED (main agent): the chosen move has a verified form (verify_choice), off in play until a second run and Kaveh's word | 2026-11-19 | 5 |
| A goal is two threats made at once that no one reply meets; the stake orders goals; not the length; the budget is N questions, past N only inside a frame still making progress; ruled out means too far, break it down; their goals imagined too; among equal moves choose by progress toward the goals | 2026-11-19 | 7 |
| In sleep the oracle (Stockfish) may be asked anything and may label winning and losing positions and frames; the prior is trained on those frames | 2026-11-19 | 9 |
| The imagination layer is built in three steps: a learned future judge from human trajectories, the winning points found in human games, then the prediction from here to those points (direction; the oracle's part is the open conflict of section 13 item 12) | 2026-11-19 | 7 |
| The prior learns the tactical situations of the opening and the chances of frame outcomes as played, not best play (direction; the oracle's part is an open conflict with section 9) | 2026-11-19 | 7 |
| A future imagination layer for positions with no concrete tactic: imagine a future, read its threat landscape, ask what I change and whether they have time to change it back (direction; the reading of how it sits on the two layers is PROPOSED) | 2026-11-19 | 7 |
| Subgoals as symbols that go with winning: fact patterns first, two tests on human moves, a graph network with quantized codes only if they fail; in the future or not by the two layers | 2026-11-17 | 7 |
| PROPOSED: the equal reading has a fast form for play and a verified form without the search's pruning for measurements and sleep | 2026-11-18 | 5 |
| Human games may be read; the endings risk and the fill | 2026-11-15 | 10 |
| Rust first; game-free layers; Lean; reuse; the real store | 2026-11-14 | 11 |
| Every build checked against the spec; modules with stated interfaces; no quiet departure | 2026-11-16 | 11 |
| The pointing at a weakness and the choice of subgoal are from the graph only; language is the last step, for a reader | 2026-11-16 | 7, 11, 15 |
| What exists and the state of the build | 2026-11-16 | 12 |
| Open questions | 2026-11-16 | 13 |
| The pieces are folded into one wake and the whole is judged, not each piece alone | 2026-11-19 | 8 |
| One way of asking for every goal: what stands in the way, can it be removed; a stopper is a target and that is the subgoal; the defence is the same read from the other side; no word special to the goal unit | 2026-11-19 | 7 |
| What was replaced | 2026-11-16 | 14 |
