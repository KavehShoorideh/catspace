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

**Calculation and estimate, decided together** (DECIDED 2026-11-19, his words; the six points below confirmed by him the same day: "Yep, you got all of that right."). "I need an intuition and a calculation, right? And if I can
calculate, if it's close enough for me to be able to calculate, I should be able to calculate using the rules. Follow simple rules of
logic. And if I can't calculate it, then I have to estimate. And I estimate based on frames that I've seen before, based on basically
sub-problems or sub-structures in the structure, instead of just taking the whole thing in at once. And that's the novelty that I'm
bringing there. So these are the two pieces that we need, and they need to kind of go together. And we need information from both, and
then using information from both, we make a decision. ... these are different ways of estimating the same thing. But, you might have
multiple different estimates, and using multiple different estimates, we decide. So reinforcement learning agent as the final layer can
make that decision, right? That's kind of the way it should go." The six points, written by the main agent and confirmed by him: (1) what is
near is calculated by the questions and the line read; (2) what is not is estimated PER FRAME AND PER GOAL (the chance that this goal's
unmet facts come true, that this frame ends well, from frames seen before), not by one number for the whole position; the heads of
section 6 as fitted so far read counts summed over the whole position, which is the thing he says not to do; (3) the wake hands the last
layer, for each candidate move, what was calculated and the several estimates; (4) the last layer is a learned chooser trained in sleep
on the results of games, its reward the result only, so the estimates are its inputs and never a value added to the judge (rule 7);
(5) until it is trained the wake decides by the rule for choosing a move of section 8 and logs those inputs a decision, which is the
chooser's training data; (6) its inputs and weights stay readable.

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

**Goals beyond the search's horizon, broken down into it; a reached subgoal as a training signal; where the subgoals come from**
(DECIDED 2026-11-20, his words, after the main agent's read of plans 79.11 to 79.16).
- THE CHAIN ENDS INSIDE THE HORIZON. "We definitely want goals that are outside the search's horizon, and then we want to break it down
  into sub-goals ... and eventually you want to reach a sub-goal that is in the search's horizon ... So that you can search and get to
  that sub-goal. But the sub-goal has to ... be like a step to reaching the next sub-goal and the next one ... we do not want our final
  sub goal to be outside of the search's horizon because we can never get there then."
- A REACHED SUBGOAL IS A TRAINING SIGNAL. "A reached sub goal actually should be a training signal, and we're not doing that at all. We
  should basically use the reaching of a sub goal to make our reward signal more dense rather than just win and loss of games."
- MEMORY. "Nothing is held from move to move. We talked about this, like we want memory and we want things to be invalidated, and we
  have the whole event-driven system." This restates sections 1 and 4; it was decided and not built in the wake.
- WHERE SUBGOALS COME FROM. "We need to mine the sub-goals from our own data, essentially from our own play, and we need to pick out
  which existence of which tactics lead to improvements ... the existence of those tactics should be the sub goals. Like ... put the
  rook ... on the 7th rank, because rook on the 7th rank leads to better tactics because the king is boxed in ... The king being boxed
  in itself is not valuable ... But once the king is boxed in, then the tactic that wins us some material somewhere, that's valuable.
  ... The mate is ultimately the goal, but the material is the sub-goal to get to that mate."
- THE DATA. "We have to get it from master games. And if we don't get it from master games, we have to condition on ELO. And we should
  not use the low ELO trained stuff when solving puzzles, like when doing accurate stuff. We should only use the low ELO stuff when
  we're dealing with low ELO people in a probabilistic way."
- READING (main agent, PROPOSED, not his word). The chain is read backward from what is valued: the end of the game; a gain of
  material; a tactic that exists (a confirmed row of the threat list, a frame that ends well); the facts of the graph whose holding
  makes such tactics come to exist. Each link is an event that can be counted in games, so each is both a subgoal and a training
  target. Nothing in the chain is given a worth by hand: a fact is a subgoal only because tactics follow it, and a tactic only
  because material follows it. Both our own games and human games are read, the human games by rating.
- THE SYSTEM AGAINST FORGETTING: `docs/SPEC_LEDGER.md`, one row a decision with its state in the playing engine.

**The small endings as the test of means and ends, and a measure of progress** (DIRECTION given by Kaveh 2026-11-20; a build is not yet
agreed). "Maybe we need to understand this, mining these sub-goals in the context of like a two-rook mate ... this is a good test of
our means ends reasoning. Can we mate? Can we find a simple mate?" If not, a prior that gets better at it in games against itself, and
beside it "this sub goal learner, which is a separate thing, looking at it and seeing, okay, what moves did I do that won?" Or: "we want
two sets of games ... some perfectly played games and ... some badly played games. And we want to figure out what makes a bad game
bad, what makes a good game good." With one rook: "can we understand the idea of pushing the person back?" And: "we need some sort of
a measure to basically measure our progress towards goal ... back in October, we had this progress measurement ... Did you bring that
over? ... that was critical in getting close to the end."
- THE ANSWER (main agent): it was NOT brought over. October has, in the older system: the proved Lean theorem forced_by_measure
  (lean/ideas/Ideas/Core/ForcedByMeasure.lean: if from every position of a class some play of ours of at most k moves lowers a
  measure against every reply, the goal is forced); a miner of such measures from generic atoms (plan 73.5b, technique_miner.py), which
  found one for king and rook against king with no failing position and mated in its play-outs where the engine without it repeated
  positions; mined measures for two rooks against a rook (Ideas/Generated/Measures.lean); the goal judge and the clock of the chase in
  agenda.py. The engine of plans 78 and 79 (rust/engine) has none of it: its only word of progress is that a taking restarts the count
  of quiet plies in a line. Ledger row 35.
- A LESSON OF OCTOBER TO KEEP (plan 73.5b): perfect play is a poor teacher of a measure. The table's fastest line takes shortcuts every
  measure counts as going backward. What a measure must do is ADMIT a drop within a few of our moves against every reply, not fall on
  every move of a perfect game.
- READING (main agent, PROPOSED): a measure of progress is the missing link of the block above. The far goal (the end of the game) is
  beyond the search; "the measure one step lower" is a subgoal inside it; reaching it is the dense training event; holding the measure
  from move to move is the memory. With one rook the end is far beyond the search, so that ending is the true test; with two rooks the
  end may lie inside the search already, which is to be measured first, not assumed.

**The go on the small endings** (Kaveh 2026-11-20, "ok go", to the order the main agent proposed; plan 79.17). The order: (1) the
present engine is measured on the two classes before anything is built; (2) the measure is brought into the new engine as a
game-free piece built from the base graph; (3) it is mined from two sets of games, good and bad, and checked by October's
criterion (a drop exists against every reply); (4) the wake holds it as a goal, searches for the next drop, logs each drop as a
subgoal reached, drops the goal only for a stated cause; (5) the classes are played again. The measure orders and chooses among
moves the calculation holds equal and never enters a line's value. READING of the main agent, kept until Kaveh says otherwise
(section 13 item 36): his go covers the use of the endgame table in sleep and in the measure, as the label of a position as won
and as the defender, under the oracle rule of 2026-11-19; the wake never reads it.

**Future prediction: one channel, two sub-layers, and the decision to stop** (Kaveh 2026-11-24: "I need a future prediction channel, and
in the future prediction, I basically want two sub-layers. One is the near future, the future that's kind of concrete, and the other is
kind of a more distant future ... where you can't really calculate ... the far future that I hope to make concrete ... I don't yet have it
concrete. I need to gather the evidence for it. It's kind of like I believe on faith, and that's where the human intuition comes in.").
DECIDED. This NAMES and BINDS what was decided on 2026-11-19 ("Calculation and estimate, decided together") and 2026-11-20 ("Goals beyond
the search's horizon"); it replaces neither. Written out by the main agent, with the choices it made to remove ambiguity marked (M):
- ONE CHANNEL. Everything the engine says about what will happen is one channel, FUTURE PREDICTION, with two sub-layers. Every statement
  of the channel says which sub-layer made it.
- THE NEAR SUB-LAYER is the concrete future: what the rules and a read line settle with every reply read. It is the calculation of
  2026-11-19 (the questions, the line read, the search). A statement of it is a claim that stands or falls by the rules.
- THE FAR SUB-LAYER is the future that cannot be calculated from here: a belief that a goal can still come true. It is the estimate per
  frame and per goal of 2026-11-19. Today it is held on faith; it is to be MADE CONCRETE BY EVIDENCE gathered from results of games
  (rows 24, 25), and whatever of it becomes provable moves to the near sub-layer as a lemma (row 49).
- (M) THE BORDER IS NOT A COUNT OF PLIES (row 28). A goal is NEAR when a line is read in which every unmet fact of it is met against every
  reply. Until then it is FAR, and the far sub-layer says only whether it is ALIVE: whether its unmet facts can still be met. A goal
  moves from far to near when the near sub-layer closes it; nothing else moves it.
- THE DECISION TO STOP. "Being able to stop here is very important ... there's no concrete future, and there's also no distant future
  ... that stop decision is on the distant future, saying there's no future here." A target is LEFT only when BOTH sub-layers say
  nothing: the near one has no line that gains, and the far one holds no goal on it that is alive. The stop is said with its cause and
  is a statement like any other: it falls when a fact it read changes (section 4), and (M) a target that was left is taken up again
  only through another goal or frame that reads it, as the frame rule of 2026-11-15 says for a pruned move.
- THE ORDER OF LOOKING. Targets are taken by stake, the unit whose loss ends the game first, then the next in worth (the mined worths,
  nothing assumed). (M) This is an order, not a cut: a lower target is looked at when the higher ones are left or kept.
- A GOAL SOME UNMET FACTS AWAY. His example (position kept as a fixture, 2026-11-24): a long-moving unit of ours stands on a line with
  a high target of theirs and two units between. That is a goal (the unit takes the target) with two unmet facts (each unit between
  must leave the line), in the words that exist (a goal has facts; an unmet fact names what must change). For each unmet fact the
  channel says which of our moves meet it and whether such a move is FORCING, that is, whether it makes a threat the other side must
  answer. The goal is ALIVE when each unmet fact has a move that meets it, and it is a THREAT NOW when all but the first can be met by
  forcing moves, because the other side then has no free move in which to take the target away.
- FRAMES JOIN. "These two frames join together." A forcing move whose own line gains nothing (a goal the near sub-layer closed with no
  gain) is still a TEMPO: it may be the move that meets an unmet fact of another goal. The channel therefore keeps, for a goal that was
  left, the forcing moves it found, and offers them to every other goal as ways to meet an unmet fact. The join is by units and places
  shared between the two goals, never by a name.
- (M) WHAT THE FAR SUB-LAYER MAY DO. It orders, it keeps alive and it stops. It never adds to the value (rule 7 of AGENTS.md): a goal
  kept alive changes what is read next and what is proposed, and the gain is counted only when the near sub-layer closes it.
- (M) EVIDENCE. Each far statement is logged with what became of it (met, by what move, how many moves later, or dropped and why); a
  kind of far statement that does not come true more often than its base rate on games not used to make it is not believed. The
  oracle rule stands: evidence from results in the wake; labels of an outside engine in sleep only.
- NOT DEFINED YET, and said so: what it means, beyond units on a line, that force CAN ARRIVE at a target ("I can't really imagine my
  queen getting there"). The first concrete form is the reach of 2026-11-16 (my moves only, then the other side's veto, row 16). The
  example gives one case; more cases are gathered before a definition is fixed (section 13).

**A way that wins is taken and locked; resources are what is optimised; narrow replies; the game of preparation** (Kaveh 2026-11-24:
"a normal engine ... will find the best move ... for humans, it's not like that. You follow a logical reasoning, and you find some move
... oftentimes, you find the move that's winning, and it's not the fastest winning move, but it gets you there. And more importantly,
it's independent of what your opponent does ... Once you take in your decision, this is the path I'm going to go, it's locked in ...
you don't spend any more energy calculating anything ... efficiency in terms of amount of power used or amount of time used, both of
those are resources. That is what we have to optimize for ... If you have a winning strategy, you're winning. Now use the one with less
energy." The points below were put to him by the main agent and confirmed the same day: "Yep, I agree with all of your aims.").
DECIDED, as long-term aims; the way to build some of them is not known yet and is said so.
- THE ORDER OF THE TWO AIMS. First the result: a win over a draw over a loss. Then, among ways that secure the SAME result, the least
  cost. Cost never buys a worse result. The value stays the global objective (AGENTS.md rule 7); cost decides only among moves the value
  does not tell apart by result. This binds the rule of 2026-10-20 (search on only while no move keeps our status) and plan 65 (time as a
  priced choice) into one statement.
- THE UNIT OF COST. Wall time and nodes now, both printed today. Energy is the real aim and is not measured yet. Every measure of
  strength is reported with its cost beside it; strength for a given cost is the north star, as it has been since October.
- A WAY THAT WINS IS TAKEN. The engine does not go on looking for a better move once it holds a way that secures the result. A way that
  is not the fastest is accepted.
- THE LOCK. A way taken is LOCKED: it is held from move to move, said, and nothing more is asked of either sub-layer of the future
  channel about it (a locked way is a near statement that is closed). Fewer decisions are made, so fewer can be wrong.
- WHAT MAKES A LOCK SAFE. Two things, both required. (1) A MEASURE THAT FALLS against every reply (row 35, the theorem of October):
  without it a way that is not the fastest can go in circles or run into the draw by the count of moves. The measure must therefore reach its end inside what the game's own counting rules allow (the count of quiet moves, repetition): a way whose measure falls but too slowly secures a draw, not a win, and is judged as that (added 2026-11-24 after reading how endgame tables fail here). (2) A CHEAP CHECK each move
  that the reply did not touch what the way read (section 4: a claim stands until something it read changes). The lock is released
  only by that check or by the measure failing to fall, and the release is said with its cause.
- HOW FAR A WAY DEPENDS ON THE OTHER SIDE is a scale, not a yes or no. Three grades: the same moves whatever they play (rare); ONE
  METHOD that gives my move from their reply with no search (the falling measure of 79.17 is the first case); a line with few points
  where their reply changes my answer. Lower is preferred.
- NARROWNESS. "If my opponent has only one good move, then that kind of makes it easy for me to prepare. And I should basically guide
  them towards areas where there's fewer outcomes of their choices." The count that matters is their GOOD replies, and of those the ones
  that change my answer. It is a number of the near sub-layer (a count over read replies; the answers the threat list finds are its
  cheap first form; the exact count is taken only for a line about to be locked, because counting costs the resource it saves).
- STEERING. Among moves that keep my result, the opportunity policy prefers the one that leaves them the fewest good replies. It is a
  reason to order and to choose among equals; it never adds to the value.
- NARROW FOR ME IS NOT HARD FOR THEM. One good reply that is plain to see is narrow and will be found. Narrowness serves my economy.
  The chance that they err is another number and belongs to the belief about the opponent (2026-11-22); the two are kept apart.
- THEIR BAD REPLIES STILL NEED AN ANSWER. A reply is left unprepared in depth only when the near sub-layer has shown it loses.
- PREPARING. Sleep prepares, the wake recalls, the long-term memory holds what was prepared (section 9). A locked way and a narrow line
  are what is worth preparing.
- THE GAME OF PREPARATION, OWED LATER. "It's meta gameplay ... the opponent is trying to predict what you're preparing, you're trying
  to predict what they're preparing, and so there's this game in preparation going on." The game on the board has perfect information;
  this one does not. Recorded as an aim, not designed. What can be said now: a predictable preparation can be prepared against, so the
  choice among equal lines may be mixed (allowed by the decision of 2026-11-22); it needs a model of what a given opponent plays and
  prepares (row 29).
- WORK OF OTHERS TO READ BEFORE ANYTHING HERE IS BUILT (listed from memory by the main agent, NOT yet checked; nothing is reinvented
  that exists): conformant and contingent planning (plans that work without seeing the other side, or with few branch points);
  satisficing and bounded rationality; rational metareasoning (compute only while the expected gain of computing exceeds its cost);
  proof-number search (finds a win, not the shortest); options and macro-actions in reinforcement learning (a committed course with a
  stated end); the metrics of endgame tables (fastest end against safe conversion); opening-book learning; models that predict human
  moves by rating; the game theory of commitment and mixing. A plan under this block begins with what these say. FIRST READING DONE 2026-11-24 (abstracts and summaries, not full texts): to take as they are, the check of a plan by the facts each remaining step depends on (Fritz and McIlraith 2007), strong plans with the layer number as the falling measure (Cimatti, Pistore, Roveri, Traverso 2003; endgame tables as the worked case), proof-number search with the size of the proof as the cost of holding a win (survey of Kishimoto, Winands, Mueller, Saito 2012). No earlier work was found for the count of result-keeping replies that change my answer, for choosing among winning ways by least compute, or for preparation as a game. IN OUR OWN HISTORY, parked in the older system and not in rust/engine: status and keeps-status (plans 59, 68; Lean keepsStatus), the priced judge (plan 65), the cheapest move that keeps the status (74.2b), the theorem forced_by_measure, pondering (52.19), the only-move flags and width of analyst/rank.py and cards.py; in rust/engine and not called by the wake: the switch read_on, meansends::ask, belief and mixed choice of layer 0.

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
**The layers, each with its own work, its interface and its test** (Kaveh 2026-11-20, thinking out loud; UNDER DISCUSSION, nothing
built, "I will tell you what to do there"). His words in short: clean up the interfaces, define the work of each layer, write good
tests for each layer, "so that we can build on top and rely on what's been done" and "so that we only work on one part". His
layers, from the base up: (1) REPRESENTATION: space, time and material; the units, the board, a notion of tempo. Test: a position
is encoded and recreated. (2) RULES: each side makes one move a tempo, alternating; what move is legal and what is not. All in
Rust. (3) BASIC FACTS, PATTERNS AND FRAMES: frames of two, three, maybe four units. Test: correctness of the basic patterns.
(4) MEANS-ENDS, THE TACTICAL LAYER: decides what move to make to get a goal. Test: positions where the solution is not quiet
(puzzle solutions; the oracle as a measure). Quiet moves need one layer deeper. (5) GOALS AND SUBGOALS, with the memory and the
techniques, working hand in hand with the tactical layer. (6) THE OPPONENT: the effect of the opponent. (7) STRATEGIC: what we are
trying to accomplish. His questions: do we have these interfaces between the layers; do we have tests that let us be sure a layer
is done; is anything missing. The main agent's answer of the same day is in the conversation and in section 13 item 38; the
layers of section 2 stand until he decides.

**The layers, second pass** (Kaveh 2026-11-20, still thinking out loud; UNDER DISCUSSION, nothing built). His words in short:
(a) the measure of progress and value are the same thing ("when you have a really high value, that means you've made a lot of
progress"); it is layer 3, on top of the rules; it needs statistical information; the tactical layer is not statistical but
needs the values to work, so the measure is the more fundamental and "the tactics is the tool by which we advance the measure of
progress". (b) The estimate at a line's end is again the measure of progress; there has to be a prediction of the future, a
layer on top of which goals are defined. (c) Memory is part of representation ("in order to represent, you need memory").
(d) Claims belong to representation too: it must be able to speak with wildcards, any unit here, any place there, an edge or
not, placeholders from which frames are built, "like chemistry has functional groups, the notion of R", said in the language of
the graph. (e) The opponent and the strategic layer come last and are not decided. (f) Wake and sleep are two parts of this, and
the whole stack is in Rust and runs very fast. READING of the main agent, to be confirmed: 1 representation (space, time,
material, memory, claims with placeholders); 2 rules; 3 the measure of progress, which is value; 4 the tactical layer; 5 the
prediction of the future; 6 goals and subgoals with the techniques; then the opponent, then the strategic layer. OPEN, raised
with him the same day (section 13 item 39): how (a) stands with the rule that subgoals enter through the order and the choice
and never the value; and where the frames found by learning live.

**What the representation layer must be able to say** (Kaveh 2026-11-20, correcting the main agent's reading of "claims"): "the
representation layer ought to be able to represent facts and conjectures and probabilities and beliefs and things like that." So
layer 1 is not only the position: it is the language in which every other layer writes what it knows, with the standing of each
statement kept apart. READING of the main agent, to be confirmed: one form of statement (a pattern over the graph with
placeholders, and what it says of it), carrying its standing: a FACT (follows from the rules at this position), a PROVED claim
(Lean), a CONJECTURE (held on a sample, not proved), a BELIEF with a probability (counted from results, with the count), and
for each what it read, so that it falls when that changes. Layers above write and read such statements and never invent a second
form. UNDER DISCUSSION, nothing built.

**The representation says all of this for the opponent too** (Kaveh 2026-11-20): "an extension to the representation layer ought
to be able to do all of this for our opponent as well. So basically, what is the belief state of the opponent?" READING of the
main agent, to be confirmed: every statement carries its HOLDER as well as its standing: ours, or ours about the other side
("we believe that they hold X", "we believe their goal is G"). Facts of the position are common to both in a game where all is
seen; what differs by holder is the conjectures, the beliefs, the goals and how far each side reads. The layer only makes this
sayable; what the other side is likely to hold is filled by the opponent layer and by sleep (ledger rows 29 and 26: the chance
depends on the opponent; data of a rating is a model of an opponent of that rating). One level of holder is proposed, no deeper
nesting, until a need is shown. UNDER DISCUSSION, nothing built. His sharpening the same day: "Or rather, what is my belief
state about the belief state of my opponent?" So it is never their state itself, which we cannot read: it is a belief of ours,
with its own probability and count, about what they hold.

**Two memories** (Kaveh 2026-11-20): "we need two types of memory. One is a long-term memory belonging to the game of chess.
This is where we store lemmas and shortcuts, and this is where we store any trained models that we have, which basically become
interfaces. They have to have their own interfaces. But then we have this other memory, which belongs to the game at hand."
READING of the main agent, to be confirmed: THE LONG-TERM MEMORY belongs to the game as such: lemmas, shortcuts, mined measures,
tables of counts, trained models; written only in sleep, read only in the wake; each trained model is reached through an
interface of its own (what it takes, what it returns, its standing as a belief), never by reaching into it. THE MEMORY OF THE
GAME AT HAND belongs to one game: the statements that stand in it with what they read, the goal held, the open tactics, our
belief about this opponent; written by the wake, emptied at the game's end, and what surprised it is handed to sleep. Both hold
statements of the one form of layer 1. UNDER DISCUSSION, nothing built.

**The go on layer 1** (Kaveh 2026-11-20: "Okay, fix all of this then. Unify the long-term versus the short-term memories, and the
scratch memory of one layer should stay in that layer ... If it's something that we do want to save, then we either add it to the
current thing using the representation ... maybe we need some sort of a mutation API on our representation layer"). DECIDED: one
form of statement with its standing, its holder and what it read; the long-term memory and the memory of the game at hand both
hold it; a layer's scratch that is not to be kept stays in that layer; what is to be kept is written through a way of changing
the memories that the representation layer owns (add, replace, retract, fall by change). Plan 79.19 builds it on `cs_memory`,
which already has the language with variables, the matcher and the snapshots. The layers above 1 are still under discussion.
His addition the same day: the long-term memory is for ANY game the system learns, keyed by the game and its specific variant
(the standard game, its variants, other board games), "properly namespaced, partitioned". A reader sees one game's partition;
nothing learned in one game is read in another unless checked again there (the last clause is the main agent's READING).

**A layer's test uses only that layer and what is below it** (Kaveh 2026-11-20, on plan 79.19's finding about the game's text
form: "tests of this layer should not depend on rules, so FEN should not be used"). DECIDED for layer 1: its tests take positions
as sets of facts and statements, and call nothing of the rules: no move, no legal move, no outcome, no text form of the game.
READING of the main agent: the same holds for every layer; a test that needs the layer above belongs to that layer. Plan 79.20.

**One graph is the representation; the rules are the next layer** (Kaveh 2026-11-20: "One graph should be the board, edges, ranks
files diagonals squares, and pieces and tempo. Rules go in next layer, right?"). DECIDED by Kaveh: layer 1 is one graph holding the
board (its places, its lines, which place is next to which), the units on it and the tempo. The rules are layer 2. READING of the
main agent, PROPOSED until he confirms: (a) the board differs by game, so layer 1 reads the board's part from the game's own
description and hard-codes none; (b) how a unit steps is a rule, so reach and attack edges are written onto the graph by layer 2
through layer 1's door of change, and are not part of the bare graph; (c) the state the rules need beyond the units (rights still
held, the counters) sits in the graph as plain facts, whose meaning layer 2 gives. Not built: today the graph is made by the rules
(section 13 item 41).

**Layer 1 holds the history; the rights are layer 2's tags** (Kaveh 2026-11-20, thinking aloud: "the first layer has to include
history as well, like move counts ... it has to include the whole sequence. So the second layer can come and say, 50 move rule or 3
move rule"). Kaveh's direction: layer 1 holds the whole sequence of the game; what the rules read off it (a right still held, a
taking in passing, the counters, a repeat) is said by layer 2. READING of the main agent, PROPOSED: (a) the history is a sequence of
changes, one a tempo, each the facts added and removed, with the key of each position reached; layer 1 knows no legality; (b) layer 2
writes what it reads off the history as tags on the nodes (a unit, a place, the tempo), through layer 1's door of change, each with
what it read so it falls by change; the history is the truth and the tags are kept step by step for speed; (c) a position given
without its history (a test position, a start in the middle) states the tags as plain facts.

**A slower result is diagnosed and mended, not set aside** (Kaveh 2026-11-20, on the graph of layer 1: "If it doesn't turn out to
be faster, we have to figure out why. We don't just dismiss the idea off the bat. We fix it."). DECIDED: when a part built to the
spec measures slower, the cause is found and fixed before the design is questioned. Plan 79.21's amendment.

**The order of work: layer by layer from the base** (Kaveh 2026-11-20: after layer 1's three open things, "work on making layer 2
like you did for layer 1", "then layer 3 and so on"). DECIDED: each layer in turn gets what layer 1 got: its work said, its
interface, its own crate or clear place, and a test that uses only that layer and what is below it. Layer 2 (the rules) is a go.
READING of the main agent: layer 3 (the measure of progress, which Kaveh calls value) cannot be given its interface before item 39
of section 13 is answered (value as the measure of progress against the rule that subgoals never enter the value); the main agent
asks before building it.

**Layer 3 holds two implementations: value and the measure of progress** (Kaveh 2026-11-21, answering item 39 of section 13: "You
can have two different implementations, basically. One is value, one is measure of progress, but I suspect that they will both start
saying the same thing."). DECIDED: layer 3 has one interface and two implementations behind it, the value (fitted on results only)
and the measure of progress. They stay two things; the rule that subgoals enter through the prior and never the value stands.
Kaveh's expectation that the two come to say the same is a thing to MEASURE, not to assume: layer 3's test prints where they agree
and where they do not. Plan 79.24.

**Layer 4 takes a goal as an input** (Kaveh 2026-11-21: "Layer 4 needs to work towards the sub-goal. I think without a
sub-goal, it's not going to make sense"; then "Right, layer 4 has to take a goal as an input"). DECIDED: the question to the
means-ends layer carries the goal it works toward; the layer says how the goal is reached, and does not choose it.
READING of the main agent, kept until Kaveh says otherwise: the goal is a statement in layer 1's form, a pattern over the graph
that a side wants to hold; taking a unit and ending the game are two goals among others; the answer is reached by force, out of
reach within the depth read, or not settled; the value judges the line's end and the goal never changes a value (rule 7); the
layer above chooses which goal to ask, and layer 4's test takes its goals from fixtures. Plan 79.27.

**Layer 4 is asked with a list of possible goals** (Kaveh 2026-11-21: "maybe even multiple potential goals, or maybe we need
to maintain a list of possible goals and then layer 4 can pick out of those possible goals"; on the main agent's proposal, "Yes,
do it, please"). DECIDED: the entry of layer 4 takes an ordered list of goals with one budget and returns one answer for each
goal, and any line that reaches two at once. Layer 4 does not choose among goals by worth: it reports what is reachable and at
what cost, and the layer above chooses; it spends its budget in the order of the list, so the upper layer's opinion enters as an
order, never as a value. A list of possible goals is kept (READING: in the memory of the game at hand, carried across moves, a
goal falling when what it read changes; not built in plan 79.27). Plan 79.27 amendment.

**Reached means the goal outlasts the reply** (proposed by the main agent in the report of plan 79.27; Kaveh 2026-11-22:
"ok"). DECIDED: layer 4 says a goal is reached only if it still holds after the other side's reply, whatever that reply is; a
goal that holds for one ply and is then undone is said apart as touched and never counted as reached. READING of the main agent
for the exact form: reached is the pattern binding at a position where the asking side is to move or the game is over, on every
reply. Plan 79.28.

**Layers 0, 1 and 2 by what they hold** (Kaveh 2026-11-22: "Layer 1 is all of the observables. Layer 2 is the base rules of the
game"; "if there is a layer 0, that is mathematics and logic"; to the cut below: "Yes, I agree with everything you said and the
line that you want to hold"). DECIDED. LAYER 0 is mathematics and logic, the same for every game and for none: the logic of
statements (placeholders, and, not, two differ, matching a pattern against facts), number and order, proof (Lean), and the logic
of a game of two sides in the abstract (for every reply there is an answer). It is given once and is never extracted or learned
for a game. The search and the matcher as running programs are machinery, not a layer of knowledge; layer 0 is what they may
assume. LAYER 1 is what is seen in one state: the objects, the places, and the relations of space that show in one frozen state
(next to, on one line, on the edge, between), and the history as the sequence of states seen. LAYER 2 is what links one state to
the next and what ends the game: what may move where, what bears on what, whose turn it is, what a rule reads off the history,
the end and its result. It is learned by being told or by watching the states change; watching is the evidence for layer 2 and
does not move its content into layer 1. FOR NOW layers 1 and 2 are supplied by hand for the game at hand; machinery that extracts
them from a game is later work, not built. WHAT THIS CHANGES in what is written above: the pattern language is layer 0's, not
layer 1's (the block "The go on layer 1" says layer 1 is the language); the meaning of the tempo is layer 2's (the block of
2026-11-20 put the tempo in layer 1; what stays in layer 1 is that the states come in a sequence). NOT BUILT as a move of code:
cs_rep holds the language and the rows for bearing, moving and tempo today; whether crates move is not decided.

**Layer 0 is wide enough for two unlike games; the layers above need not be** (Kaveh 2026-11-22: "Let's say chess versus soccer.
If we can handle these two games, we're golden"; "I'm not saying all the above layers have to handle these two games ... but I
think layer 0 should be generic enough that should be able to handle both"). DECIDED: the scope is games of opposed sides; layer 0
must be able to say both a game of turns on finite places and a game of many agents acting at once in continuous space and time
with uncertain outcomes; layers 1 and up are built for the game at hand only. PROPOSED by the main agent, not confirmed: layer 0
holds nine things (things and kinds; statements; quantity and order; space in the abstract with the finite graph and the plane as
cases; time and change; agency; the logic of opposed choice with turns, choices at once and chance as cases; chance; proof), each
an interface with more than one case and its laws stated in Lean, in one crate at the bottom that names no game. PROPOSED test:
no game is named, and each notion is used by fixtures of facts written from both games; the second game is said, not played.
Nothing built.

**Layer 0 holds six things, and is built** (Kaveh 2026-11-22, to the six below: "Great. Okay, now build this layer 0 with unit
tests so that it's self-consistent and I hopefully never need to change it again"). DECIDED; this replaces the nine PROPOSED in
the block above. The test of belonging: could it be said to one who has never seen any world. (1) LOGIC AND PROOF: relations,
placeholders, and, or, not, equal and differ, every, some, at least so many, matching a pattern against facts; proof in Lean.
(2) SETS AND COUNTING: a set, membership, a kind as a set, how many. (3) NUMBER AND ORDER: whole and real numbers, comparison,
arithmetic. (4) STRUCTURE, three pure forms that layer 1 points at: a graph (nodes and links), a sequence (items in order),
geometry (points, distance, direction, region, inside). (5) CHANCE: probability and expectation. (6) OPPOSED CHOICE: choosers, a
set of options, an order of preference over results, and the form "there is a choice such that for every counter-choice", with
choices in turn, choices at once and a choice by chance as its cases. TAKEN OUT of the nine as seen or told: that an object is
the same one as before, time and events, the places a game has (layer 1); who controls what and what is wanted (layer 2 and up).
READING of the main agent for the build (plan 79.30), kept until Kaveh says otherwise: one crate at the bottom that depends on no
other crate of ours and names no game and no thing of a world; the pattern language and its matcher move down into it and layer
1 hands them on under their old names; each of (4) and (6) is an interface with more than one case; the laws of each are stated
in Lean and the tests name the law they mirror; the test uses fixtures of facts from two unlike games, the second said and not
played; for choices at once only pure choices are read (the lower and the upper value), mixed choices are not built and said so.

**Belief about the opponent, and choosing by chance** (Kaveh 2026-11-22, after asking what a mixed choice is: "I actually think
chess does need that randomization ... you want to throw your opponent off balance ... we also need a belief state over our
opponents ... being able to express belief states"). Kaveh's DIRECTION: the engine must be able to say what it believes of its
opponent, and to choose by chance so as not to be read. READING of the main agent, PROPOSED, nothing built: (a) layer 0 gives the
FORM only: a belief is a spread over suppositions; it is changed by what is seen (the spread given a condition, weighed by how
likely each supposition made what was seen); a choice is read against a belief by its expectation; a mixed choice is a spread
over one's own options, and for a choice at once the best such spread and the one value are found. (b) WHAT is believed of a
given opponent (what they know, what they tend to play) is of a world and belongs to the layer of the opponent (ledger rows 29
and 36), not to layer 0. (c) Against a belief that does not change, the best choice is a single option; choosing by chance pays
only when the other side reads us in turn, so the two are separate capabilities. See section 13 item 49.
Kaveh the same day: "even what is an opponent ... my opponent losing means I'm winning. So this is kind of assuming something
about a zero-sum game ... the idea of winning and losing ... belongs to a higher level up." READING of the main agent, PROPOSED:
layer 0 knows choosers, each with an order of its own over results, and DEFINES two cases without assuming either: opposed (one
order is the reverse of the other) and allied (the same order). What the results of a game are and who gets which is layer 2's;
that another chooser is an opponent then follows from layer 2's results and layer 0's definition. As built in 79.30 the forcing
reading assumes nothing of wants, but the value reading takes one order and assumes the other reversed: to be widened to an order
for each chooser (item 49).

**Layer 0 is made right; nothing of a second game is built; it is applied to the game at hand** (Kaveh 2026-11-22: "continue
building and making this layer right and fixing all the bugs that you saw. I don't want you to actually build anything soccer
related ... What I want is the tools and concepts in this layer to be able to handle a game like soccer. Not exactly, but like be
extensible ... I want to directly apply it to chess. So what's coming is chess, but try to keep it generic"). DECIDED: the go on
section 13 item 49 (an order for each chooser, the mixed choice, the belief); the faults of the 79.30 READ are mended; no rules,
player or search of a second game is ever part of this work; layer 0 is judged by its use in the game at hand and by being open
to extension, not by covering the second game exactly. READING of the main agent: the one small fixture of facts of a second
kind stays as test data only (it is what shows a notion is not shaped by one game) and is not grown; say so if it should go.
Plan 79.31.

**The pure search is mathematics and lives in layer 0; what steers it does not** (Kaveh 2026-11-22: "all of this in Rust, because
this is all going to be in the hot path ... the most invoked thing is going to be this layer 0"; then: "the search has to be in
layer 0. Is that because search is a mathematical thing and should live outside? I buy that"). DECIDED as a principle. THIS AMENDS
the line of the block "Layers 0, 1 and 2 by what they hold" that the search and the matcher as running programs are machinery and
not of a layer: that was too coarse. The reading of a tree of choices (there is a choice such that for every counter-choice, by
recursion, with what was already answered kept by the sameness of nodes) names no game and is mathematics: it is layer 0's, as
an algorithm over an interface in which a node gives its options when asked, so that no tree is built. NOT layer 0's: which
options a node has (layer 2), what is asked for (the goal, from above), the order in which options are tried, the narrowing by
what a goal reads, the budget and the clock; these stay in the layer that asks (layer 4). READING of the main agent, GO given the same day
(plan 79.32, after 79.31 is merged): layer 0's forms become interfaces that cost nothing when compiled (choice over a node that gives options
on demand; number as an interface so the hot path uses bare whole numbers); the search of layer 4 becomes a case of layer 0's
algorithm, one copy where there are two; a test of speed for each form; the comparison of plan 79.31 part F is the safety net.
See section 13 item 50.

**What plan 79.32 left of "one algorithm": the walk answers steps, the values are a fold beside it** (main agent 2026-11-22, READING
of the build, mine; to be overturned by Kaveh if wrong). Layer 0's walk answers, for each property, within how many steps it is met.
The readings that weigh numbers (the guarantee, the value, the forcing by chance) are a second recursion over the same node
interface. They are not one algorithm yet. Making them one needs the answer of a walk to be general: a number taken by best at an
own node, by worst at any other, by expectation at a choice by chance, under a window that cuts. The same general answer is what the
remaining walks of the tactical layer would need to become cases. This was proposed in section 13 item 51; Kaveh gave the go on 2026-11-22 ("Do it"), plan 79.33. Also ruled
here: a layer may depend on layer 0 directly; that is the meaning of a base, and it adds no outside crate.

**The direction after the layers: the base is frozen, the loop is the work, and the whole is measured at every read** (Kaveh
2026-11-22: asked "is this project on the right direction", then of the main agent's four steps, "great. do that"). The main agent's
view, which he accepted: plans 79.25 to 79.33 made the layers sound and were each judged a success when no answer changed; none was
meant to change a decision; the bet of the project (what sleep mines, entering through the prior, makes a small search play and
explain like a master) was not tested by them. Decided:
1. Plan 79.33 is capped: one attempt for each walk; what matches is kept, the rest is put back, with no second attempt.
2. Layer 0 is frozen after plan 79.33. It changes again only when a caller above it needs something, and the plan names the caller.
   The same holds for the cut of the layers: no plan is made for tidiness alone.
3. The next plans are chosen from the rows of the loop: sleep mines (rows 25, 31, 32), the prior and the goals use what was mined
   (rows 3, 6, 14, 17, 23, 24), and the wake is measured.
4. Every READ from here on carries one measure of the whole engine on a fixed ladder and a fixed puzzle set, by the same program
   each time, also when the plan is a refactor; where the machine was not quiet the READ says the measure is owed and it is made
   before the next plan starts. The measure is a measure only; nothing is tuned on it and no verdict is written from one run.

**The cut of layers 1 and 2 is carried into the code** (Kaveh 2026-11-22: "make the code follow this and have unit tests for layer 1
and layer 2. So layer 2's got to be the rules that references the pieces ... Layer 2 says how the bishop moves. Layer 1 says this is
a piece, we call it a bishop. These are pawns. There's 8 of them."). Layer 1 says what is there and can be seen with no knowledge of
play: the things, their kinds and names, how many, the board with its places and their two alternating classes, where each thing
stands, and the history as a sequence of such states. Layer 2 says the rules and refers to the things by layer 1's names: how each
kind moves, what a move changes, whose turn it is, what ends the game. READING, mine: a relation between places seen in one state
(next to, on one line, between) is layer 1's; a relation that follows from how things move (bears on, may move to, a line shut, the
line through a unit, the turn) is layer 2's; facts derived from those (an open line, a unit tied to a guard) are layer 3's, and it
is layer 3's vocabulary, not layer 1's, that bounds what goals can be mined. This plan (79.34) changes no decision and is a named
exception to the rule that no plan is made for tidiness alone.
AMENDED the same day (Kaveh: "the arrangement on the pieces on the board is a layer 2 thing, because we want to be able to handle
different variants of chess"): layer 1 does not say which state the game starts from. It says the kinds and the board, and can hold
and describe any state it is shown. The starting arrangement is a rule and is layer 2's. READING, mine: the counts at the start
(eight of one kind, and so on) follow from the starting arrangement and so are layer 2's too; layer 1 can count what stands in any
state shown to it. The builder of 79.34 was told by message; the plan block is not rewritten.
BUILT (plan 79.34, 2026-11-23). The door is a value: layer 2 gives each relation a slot and a word, layer 1 makes the node and its
links and never reads the word. CHOICES OF THE BUILDER, standing until Kaveh says otherwise: the side a state marks as to act is a
value layer 1 holds, layer 2 names it and makes it; the door has four slots on a graph and three on a kept unit; the two classes of
places are derived from the links (a step not made of two other steps changes the class). No variant with another start has a fixture
yet, so that reason for the cut is not tested.

**Layer 2 says variants, with another start and with other rules, on the same layer 1** (Kaveh 2026-11-23: "so variants must be able
to have different starts and it should be in this layer, right?" and "sure lets do these variants. also atomic chess and antichess").
A variant may differ from chess in its starting arrangement, in the counts of its units, in how a move changes the state and in what
ends the game; all of that is layer 2's, and layer 1 (kinds, board, any state shown) does not change for it. The games named:
Chess960 and horde for the start, atomic and antichess for the rules. READING, mine: a game counts as said by layer 2 when it has
its description, the kernel's mode, a certificate that the two agree, and a fixture from an outside library; where the description
language cannot say a rule as it stands, the game is tested against the kernel and the fixture only, and the gap is named, not
hidden. What the layers above layer 2 assume that these games break (one leader a side, for one) is listed by plan 79.35 and is not
mended there; the engine is not asked to play them yet.
BUILT (plan 79.35, 2026-11-23): all four games have description, certificate, fixture and unit tests, and no source of layer 1
changed. A game gives its starts by number; one description serves the 960 starts, a start being a delta of starting facts. CHOICE
OF THE BUILDER, standing until Kaveh says otherwise: across games, bears on is the kind's geometry on the board as it stands, and
may move to is where the kind's step reaches under the game's rule of what may be taken; neither reads compulsion, what else a
taking removes, or the leader left struck. What the layers above assume that these games break is ledger row 42.

**Every unit of a layer is tested before the layer above is taken up** (Kaveh 2026-11-23: "everything that you checked by hand
should become a unit test. All the units need to be tested. And before we move on to layer 3, this is what we need to do."). What
the main agent or a builder checked by reading or by hand is not done until a test holds it. READING, mine: "unit" is taken in both
senses, every public item of a layer's code and every kind of unit of every game the rules say; a gate fails when a public item of
layers 0, 1 or 2 is called by no test; a fault a new test finds is mended in its layer unless the file is protected, and then it is
named. The same rule is applied to layer 3 when it is taken up. Plan 79.36.

**The description is the source of the rules; fast code is only a certified copy** (Kaveh 2026-11-23: "Layer 2 should only talk to
the GDL. We do not want to hard code game rules in layer 2. We want to use game rules provided by other people", then, of three
ways offered, "let's go with option number 2, where GDL is the source, and then we have certified verified copies, and then the unit
test should make sure that all certifications are in place before passing"). Every answer layer 2 gives about a game is defined by
the game's description (rules/chess.gdl and the deltas under rules/variants). Code that answers faster (the kernel over an outside
library, the placed questions bears on, may move to, strikes a place, the state as facts, the names of moves) may stand in for the
description only where a certificate shows the two agree; a fast answer with no fresh certificate fails the tests. No rule of a
game is written by hand in Rust as its own authority: a branch on a game's name is allowed only inside a certified copy.
STATE WHEN DECIDED: the certificates cover the game questions (the side to move, the allowed moves, the next state, the end, the
outcome, the facts) for seven games. NOT covered: the placed questions, which are hand-written over the outside library's tables
(game/src/outside.rs, with branches for atomic and antichess from plan 79.35) and have no definition in the description and no
certificate; the state as facts and the move names added by 79.35. READING, mine: the descriptions in the repo were written in
this project, not by other people (rules/chess.gdl says so); reading descriptions others wrote (Ludii, the Stanford set) is a
separate piece, not decided here. PROPOSED, mine: the plan after 79.36 defines the placed questions in the description, certifies
the fast copies against it for every game, and adds the gate that lists every fast answer of layer 2 with its certificate.

**The crux of layer 2 is a translator from the description to Rust, made fast piece by piece under tests** (Kaveh 2026-11-23: "this
is the crux of layer 2, is basically this GDL to Rust translation ... as part of the ingestion of new game rules, where if the
GDL to Rust translator ends up translating into very slow Rust, then we will feed it some fast, like let's say Shakmati kind of
comparison, and basically say this is possible ... profile the code and get the speed to here, or to 90% of here ... build unit
tests around every little piece and making sure the pieces each provide the same answer, but at a better speed"). Ingesting a game
is: read its description, translate it to Rust by one translator that knows no game, pin every piece of the translated code with
unit tests of its answers, then profile and make the hot pieces faster one at a time, each change passing the same tests. A fast
outside implementation, where one exists (shakmaty for chess), is the mark of what speed is possible and a second check of the
answers; it is not the source. Kaveh's expectation: this method can end faster than the hand-written library.
READING, mine. (i) Two kinds of speed work must be kept apart. A pass of the translator (a relation over a small fixed set of
places kept as a set of bits, a pure function of a small input kept as a table) is general, serves every game, and keeps the
claim that no rule is written by hand. A piece rewritten by hand for one game is a certified copy in the sense of the block
above: allowed, tested against the translated piece, but it is hand-written rules again and each new game pays for it again. The
first kind is preferred and the second is counted and reported. (ii) The target should be set by a measurement of what share of
a decision's time the rules take, not only by the outside library's speed. (iii) NOT MEASURED: how slow the translated code
is, and whether a general pass can reach the library; my expectation is a large gap at first and that beating the library is
unlikely without passes that rediscover its tables. (iv) This is foundation work and changes no decision at chess; it is taken up
at Kaveh's word, as layer 2's own work, before layer 3. PROPOSED, mine: the first step is a measurement (translate the chess
description once, time it against the evaluator and the library), and the plan is written from that number.
THE UNIT OF TEST IS A PIECE OF THE GAME'S LOGIC, NOT A POSITION (Kaveh 2026-11-23: "the test cases are no longer chess positions,
but they're little bits of the chess logic. And so that is much easier to test"). Each relation of the description is a piece:
given these facts, exactly these answers. READING, mine: a relation that does not read the state (a step, two places on one line)
has a small fixed domain and is tested on ALL of it, which is a proof by exhaustion and not a sample; a relation that reads the
state is tested on made states and sampled ones; tests of whole positions (the fixtures, co-play) stay as the check that the
pieces compose.

**Layers 0 to 3 may be mended; layer 0 takes no significant new function** (Kaveh 2026-11-23: "Yes, you can mend anything in layers
0 through 3. We don't want to build significant functionality in layer 0 anymore, but we want to build things that fix bugs or
allow us to support the work that we're doing."). This amends the freeze of 2026-11-22: a fault in layer 0 is mended when found,
with its test, and a small addition is allowed when a named piece of work above needs it; no new large part is built there. The
same holds for layers 1, 2 and 3. The three protected files of the rules (the description reader, the evaluator, the kernel) and
the old player's solver stay protected until Kaveh says otherwise: READING, mine, since their certificates and the yardstick rest
on them.

**Layer 2 answers whose turn it is without asking whether the game has ended** (Kaveh 2026-11-23: "Yes. You go ahead and add this to the
game interface and modify everywhere that uses it, including ledger.rs."). Measured in plan 79.37: more than half of a decision at 0.2 s
is the question whose turn it is, asked about a million times a decision and answered through the end of the game, which makes every
move. The game interface of layer 2 gets one more question, the side to move as the state says it, cheap, with no reading of the end; at
an ended state its answer is the side that would have moved. The old question (the mover, none at an end) stays for whoever needs the
end. Every asker in the layers above that only wants the side uses the new question, the shared search (tactics/src/ledger.rs) included:
this is the one change the old player's search takes, at Kaveh's word, and the wake and the old player get it alike. READING, mine: the
kernel's own answer to the new question is written in kernel.rs, since an interface question needs its answer there; no rule changes.
What must hold: every answer at a fixed count of nodes is identical to before; only the time changes.
MANY ROLES (Kaveh 2026-11-23: "be sure to do this in a way that is compatible with 4-player chess, like Bughouse chess or these other
variants"): the question answers a role among the game's roles, however many; its default assumes no count of roles; the swap adds no
new assumption of two sides. READING, mine: in a game where several roles may act at once (two boards) the question answers what the
mover answers today, one role; a question that answers several roles is not built here and is named under ledger row 42.

**Several boards and several players are owed, later** (Kaveh 2026-11-23: "We just want to not add a two-player assumption ... maybe we
need a node for every player ... every person who can believe things. But that might get too complicated. So let's just keep going the
way we are ... But remember that we need to come back and fix this. So add it to the list of things to do later to add support for
multiple boards for multiple players."). Decided now: nothing new assumes two players; layer 1 is not touched for this now. Owed later:
one game whose state holds several boards linked by the rules, more than two roles, sides that are teams, several roles on the move at
once, and waiting as a move. Kaveh's idea, not decided: a node in the graph for every player who holds a strategy and can believe
things. READING, mine: a game with one board, two roles and units held in hand is the natural first step, since it tests places off the
board alone.

**Two checks at a merge: a lean one off the engine's path, the full one on it** (Kaveh 2026-11-23: "Maybe we want to do this full
verification at the end after we've done speed improvements, and then keep like a leaner verification for middle steps"; then "Yes, please
do it. Amend the rules."). This AMENDS the rule of 2026-11-22 that every READ carries one measure of the whole engine. The line is drawn
by what a plan touches, not by where it stands in a sequence. A source is ON THE ENGINE'S PATH when the engine that plays is built from it
or reads it at play: the sources of the crates, the manifests, the Python of the wake and the measure, the descriptions under rules/ and
the reader. A plan that changes NO such source (the translator and its generated files while they are only in tests, documents, test
files, printing scripts) gets the LEAN CHECK: the tests of the crates it touched, the gates of the plans whose files it touched and its
own, the scan of names (test_plan78), and the count of fresh certificates. Its READ says "no source on the engine's path changed, no
measure" and names the last measure that stands. A plan that changes a source on the path gets the FULL CHECK as before: the tests of the
nine crates, every plan gate, the certificates, the Lean library, the comparisons of 79.32 and 79.33, and the measure of the whole (row
39), on a quiet machine. The day translated code is put on the engine's path is such a plan. The main agent decides which check from the
list of changed files and says which it ran in the READ.

**One lemma machinery, used twice: by the rules and by the layers above** (Kaveh 2026-11-23, on why the kernel stays faster than any
general pass of the translator: "I like the third also, especially since we'll be using lemmas in the context of chess more broadly. So in
the context of rules it makes sense to have them, and in the context of strategy in upper layers. So maybe we need to make it generic so
we can use the lemma machinery twice."). DECIDED, direction: the translator may take proved lemmas as input, and the machinery that holds,
proves and serves a lemma is ONE, the same for a lemma about the rules (layer 2: a cheaper condition that gives the same rows as a relation
of the description, such as the fact about a unit held on a line that lets the kernel skip trying each candidate) and for a lemma about
play (the layers above: frames, pruning, goals). A lemma belongs to a game and lives with its description; the translator and the layers
stay free of any game. Lean stays the source of truth (AGENTS.md rule 11); the description stays the source of the rules and fast code is a
certified copy (row 44). NOT A GO TO BUILD: plan 79.42 first shows how much general passes leave. PROPOSED by the main agent, not decided:
(a) one form for both uses, "under condition C, relation R is the same as R'", stated over the relations of the description, with its proof
in Lean or, until proved, a certificate by co-play that marks it as not proved; (b) the translator uses such a lemma as a rewrite of R and
the harness holds the rewritten piece to the plain one as it does a pass; (c) the layers above read the same store; (d) the store that
exists for learned lemmas (rules/learned/<game>/lemmas, the certificates under discovery/lemmas) is the starting point, to be searched
before anything is added.

**The lower layers are made as fast as the outside library first; the facts stay; the list of allowed moves is made once a state** (Kaveh
2026-11-24: "i don't care about the engine just yet. i want the lower layers to be as fast as shakmati if possible"; "while keeping our
representation in terms of facts"; "yes fix the lower layers in kernel.rs").
(i) ORDER OF WORK. The speed of layer 2 comes before work on the engine above it. This lifts, for speed work only, the freeze of the
rules layer of 2026-11-22. The mark is the outside library's rate at the same questions on the same positions.
(ii) WHAT THE MEASURES OF 2026-11-24 SAID (one run each, a loaded machine, directions only; the printed lines are under
artifacts/experiments/rules_share_79_37/now and in the plan note). The translated module's time is spread over many pieces, none
large enough to be worth a lemma of its own. Its cost is in how it works: it makes every row of every derived relation at every state,
one row at a time, with allocations, and it does the work of the end of the game when only the moves were asked. In the engine, the
rules take about two thirds of a decision, and the largest part is the end of the game asked of the outside library, which makes the
whole list of allowed moves to answer it; the same state is asked two to four times in a row.
(iii) THE FACTS STAY. A state is a set of facts and every rule speaks of facts. HOW the facts are stored is free: a relation over small
sorts known in advance may be stored as machine words, one bit a fact, and a whole set of facts may be made by one operation or one
lookup. Every state can be handed out as a list of facts and made from one. NOT allowed: a layout of the state designed by hand for one
game; a stored field with no fact behind it (such a field is a lemma and belongs in the lemma files, standing and all).
(iv) THE STEPS, none of them a lemma, each in the translator and so for every game: answer only what was asked (the end's own work
only when the end is asked); no allocation on the path of a question; a set of rows made at once from a table the translator builds
from the description's own rule; derived rows made only where the question reads them. What is left after these, the test of each move
against the mover's own safety, is the generic lemma "a test is skipped for a move that touches none of the cells the test reads".
Others' work to read first: demand transformation (magic sets) and incremental view maintenance for logic programs.
(v) THE BRIDGE OVER THE OUTSIDE LIBRARY (kernel.rs, changed at Kaveh's word): the last state read on a thread is kept with its end and
its list of allowed moves, so the list is made once a state. The answers are unchanged. The library's own equality of two states
leaves out the move clock, a variant's count and the units in hand, so the bridge compares them as well.
(vi) FIRST READING OF OTHERS' WORK, 2026-11-24 (a reading agent; numbers are the papers' own, on their machines and in their units,
not ours; not to be set beside our rates without checking what each counts). Regular Boardgames (Kowalski, Mika, Sutowicz, Szykula,
AAAI 2019, arXiv 1706.02462; "Efficient Reasoning in Regular Boardgames", arXiv 2006.08295): a language with the board as a labelled
graph, compiled; its largest single gain is tables that replace a run of board steps by one lookup, and it makes moves by a depth-first
walk that changes the state in place and undoes it, with nothing written out. Propositional networks for descriptions of our kind
(Sironi and Winands, 2016): far above a prover, well below the compiled board language at chess, and their own passes gained a few per
cent each. A compiler of such descriptions to C++ that evaluates bottom-up (Lancucki, arXiv 1401.5813): little gain at chess. No source
shows a general description reaching a hand-written library at chess; the board language's own authors report their compiled code three
to thirteen times behind their hand-written code on simpler games. No source applies demand transformation or incremental view
maintenance to making moves. No source derives blocker-indexed tables from a description; one design note only (H. G. Muller). What the
reading says to copy: find in the description the relations that are walks along a fixed step, and compile them to tables. OPEN: how
far to go (section 13).
(vii) DECIDED 2026-11-24 (Kaveh: "do both tables and the top down alongside bottom up"). TWO WAYS OF ANSWERING, ONE MODULE. The
translator makes, from the one description and over the one state of facts, (1) the bottom-up way that exists: the rows of a relation
written out, used where a whole relation is wanted (the graph's edges); and (2) a top-down way: a question is answered by a depth-first
walk of its rules that reaches only what it needs, stops when the answer is known ("is there one allowed move" stops at the first),
and changes the state in place and undoes the change where a rule asks about the state after a move. TABLES FOR WALKS serve both: the
translator finds, by the form of the rules alone, a relation that is a walk along a fixed step held back by a condition on single
places, and makes its answer one lookup in a table indexed by what stands on the line; the table is filled by running the description's
own rule, so it is no lemma and has no standing of its own. WHICH WAY ANSWERS A QUESTION is chosen by a clock on the same positions,
never by a list written by hand; the choice and its numbers are recorded. Both ways are held to the evaluator by the tests that exist,
and to each other on every question both can answer. The first top-down question is the end of the game, where the engine spends the
most; the others follow only if it wins. Neither way names a thing of any game.
(viii) DECIDED 2026-11-24 (Kaveh: "then lock up these lower layers and move up to upper layers, 3 and above"). When plan 79.53 is
read, merged and its records are fresh, the lower layers are LOCKED: no further speed plan and no tidiness plan below layer 3 without
Kaveh's word, whatever the gap to the outside library then is; the gap is recorded and left. Work moves to layer 3 and above.
(ix) DECIDED 2026-11-25 (Kaveh: "assuming some sort of continuity in time ... we carry a state forward ... what are the allowed
moves right now, and what was the last move that was made, and then from these two calculate the next set"; "maybe the first position
will be slow, but the later positions will be fast"). ANSWERS CARRIED FROM THE LAST STATE. Before the lock of (viii), one more plan
(79.54). The translated module keeps what it derived at the last state it read; at a new state it makes again only what the change can
have touched. HOW. The state is words, so the facts that changed are the difference of the words. The rows of a derived relation are
kept in groups; each group carries its READ SET, the base facts its derivation read (those found true and those found absent). A group
whose read set does not meet the change is carried as it is; any other group is made again whole from the new state, so a rule with a
negation needs no special handling. A new state far from the last one (the change larger than a bound the clock sets) is read in full,
as now. No interface of the module changes: the memory is inside it, as the bridge's memory of one state is. This is the read set of
section 7 (an answer is carried while nothing it read has changed) used one layer down, and the general form of the lemma of (ii) "a
test is skipped for a move that touches none of the cells the test reads". KNOWN LIMITS, said before the build. The side to move changes
every ply, so every relation that reads it is made again every ply; what can be carried is what both sides share. A state reached in a
search is one move from its parent and a few moves from its sibling, so the change stays small there; a state from elsewhere is read in
full. Programs written by hand that kept such tables are remembered to have found them no faster than making them again from words
(from memory, not checked; to be read before part B). SO THE COUNT COMES FIRST: part A counts, between a state and the state after one
move, the share of the derived rows of each relation that are the same, and the share of the time spent in the pieces that made them;
part B is built on that count and the note says what the count allows at most. The lock of (viii) then holds after plan 79.54 is read.

## 12. What exists

The three inventories (the repo against these pieces; the earlier frame and intermezzo work; the October Lean
library against the game-free graph) are in docs/SYSTEM_SPEC_HISTORY.md, sections 9.1, 12.1 and 12.2, as made on
2026-11-14 and 2026-11-15. One correction to the third: Reach.lean is chess-bound, not generic.

THE LIST OF DECISIONS WITH THEIR STATE, one row each, is `docs/SPEC_LEDGER.md` (2026-11-20); read it before this table.

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
| The whole wake, every piece of section 8 called in one decision and judged as one (2026-11-19) | Not built. Until now the player was the search with a count of worths at its end; every other piece was a measurement scored alone as a tie-break and left off. Plan 79.05. Plan 79.05 first part (2026-11-19): BEGUN IN RUST. The wake's interface is written with every piece declared; the readings of the graph onto a goal unit (a line one unit shuts, who bears on that unit and who holds it; the moves that bring a bearer), the verdict of a line, and moves named by position read first and in full by the search are in Rust, off by default. Plan 79.06: the concrete threat list, the line read of every unit and the stake's arithmetic are in Rust (cs_decision::threats), agreeing with the Python on small constructed positions; too slow as it stands because every unit is read without a question. Plan 79.07: the proposer is in Rust (cs_decision::goals): the same facts for every target, subgoals by recursion on what stops a goal, fixes as legal moves, the defence as the same facts from the other side, a budget of questions; near the play clock's slot on one test position, not under it. Plan 79.08: THE WHOLE WAKE IS IN RUST and runs end to end (carry, frame read, threat list, goals and subgoals, named moves, search, a fixed rule, estimates per goal, a log of every candidate), off by default, the heads over whole-position counts left out. First position test, one run: the pieces see the missed puzzle (the answer is a fix under three goals; the quiet refutation is a fix of the subgoal under the goal unit) and the rule that picks which goal's fixes reach the search drops it; the whole costs many times the play clock; the printed reasons are often not a player's. Plan 79.09 repairs these. Plan 79.09: the wake holds its clock, asks in order of stake and prints the deciding claim. On 300 fresh puzzles (one run, a measure) the whole finds a few more answers than the old player at the same clock given and is level with the old player given the wall time the wake uses: the whole does not yet beat the bare search at equal time. The missed puzzle is still lost; their goals must be read after our move, a chain must give its next goal when the nearest is refuted, and names must go to the search without a reading of their own first: plan 79.10. Plan 79.10: built; with ten times the clock the wake now plays the missed puzzle's answer where the old player at equal wall time does not, and at the play clock it is lost to depth alone; on the same 300 puzzles the whole is still level with the bare search at equal time. A named move is read in full beside a reduced rival, which can tilt the comparison. The whole now fits the play clock; play with a log of every decision is plan 79.11. Plan 79.11 (one run a match): in play the whole is level with the old player at equal mean wall a move on both opening sets, reading about one ply less; above a half at the same clock given, where the old player spends about 0.59 of its clock. The wake and the old player play in the match harness (whole_play) with a log of every decision and the game's result; the binding keeps the carried tactics per slot; the search reads again in full the un-named rivals of a named best root move (fair_names); the climb to the far goal is bounded. Read on the board: of twenty differing decisions five clearly better, none clearly worse, eleven a coin; about half the printed reasons are not what separated the moves; the estimates are cut by the clock. The chooser's first fit, from games in which the equal moves are varied, is plan 79.12. Plan 79.12 (one run): the last layer is built in Rust (modes Fixed, Weights, Explore; readable named inputs; only among the moves held equal) and Fixed stays the default; the reason names what separated the move played from the search's first; the moves held equal are always scored. The first fit on results, from games with the choice varied, found no input that tells equal moves apart beyond what shuffled results give; the chooser is not played. The frame read and the threat list take about a quarter of the wake's clock. A chance of each goal counted from play, as the chooser's input, is plan 79.13. Plan 79.13 (one run): the chance that a goal is reached, counted from our own games by the goal's facts, predicts on held-out games and is calibrated below a half (the first estimate per goal that says something; goal_table.rs, given through the binding). Our goals are read after each move as theirs are. Among equal moves neither a fit with it nor the plain gain rule shows against the fixed rule, and the plain rule as written had two faults (progress lowered the gain; only the goal unit's chain was read after a move). Read together with 79.11 and 79.12: the choice among equal moves does not show at this clock; the wake reads a ply less than the old player. The steps made cheaper and the chance spent on what the search reads are plan 79.14. Plan 79.14 (one run a match): the ply did not come back, because the steps before the search run to their share of the clock and a faster step asks more, not less; until this plan the frame read at the play clock built almost no frames (its share ended on the seeds), and the keys without allocation now let it; the carry by the graph does not equal asking afresh and stays off; the names in order of stake times chance and the gain mended are built behind switches; puzzles and play still level with a lean above a half. A FAULT found on the board: the wake plays the search's first move though its own probe has put it lower, about once in a hundred decisions. The wall kept there and the steps' share judged by play are plan 79.15. Plan 79.15 (one run a match): the wall is kept (wall.rs; nought lowered first moves played) and the score did not move; the cause is in the search, which answers with depth d's move when that move fails at d+1 and the clock ends; a line on the mover's goal unit is named as the standing threat; the share of the steps is a scale through the binding and three shares are not told apart. At 0.2 s the frame read still builds half a frame a decision, so the whole with every piece running has not been judged yet. The whole judged at a clock where every piece runs, and the search mended, are plan 79.16. Plan 79.16 (one run a match): the whole is level with the old player at 2.0 s a move as at 0.2 s; the steps still do not always finish at 2.0 s and their cost a node is the game's own; the search asks its first move again when the clock stops it, unused time goes to the search, fixes are ranked by stake times chance. Read on the board, every goal the proposer makes is the taking of a unit one or two moves deep, inside the search's horizon. The main agent has stopped for Kaveh (section 13 items 31 and 34). Plan 79.17 (2026-11-20, at Kaveh's go): a measure of progress is a game-free piece (cs_tactics::measure), mined in sleep from good and bad games of a small ending and checked by October's criterion; given to the wake as a goal it is held from move to move (carry), each fall is logged as a subgoal reached, the first move of a line that lowers it is named first and played among moves held equal, never in a line's value. With one long-moving unit against a lone leader the wake with the goal ends most games where without it it repeats; two faults stand (the fall is counted from the value now and not from the lowest held; the goal is dropped for good when the search claims the end), see the 79.17 READ. Plan 79.18 mended both (the fall is counted against the lowest held; a goal sleeps while the search holds the end and wakes after) and gave the question's search a memo; what stands is the question's cost at a short clock, see the 79.18 READ. Plan 79.19 (layer 1, at Kaveh's go): one form of statement with standing, holder and what it read (cs_memory); the long-term memory is one store partitioned by game and variant, changed only in sleep, every change a new snapshot; the memory of the game at hand is the wake's; the wake reads its tables and the mined measure through them and decides as before (guard). Not yet moved in: the lemmas of the old store and the record of the game; see the 79.19 READ. Plan 79.20: layer 1's tests read fixtures of facts and call nothing of the rules (cs_memory/tests/layer1.rs, with a scan that keeps it so); recreating by moves and the text form are layer 2 tests. Found: the base graph is built from the rules' state, not from facts, and the type of a fact lives in the crate of the rules; see the 79.20 READ and section 13 item 41. Plan 79.21 (at Kaveh's go): layer 1 is the crate cs_rep below the rules; it holds the board read from the game's description, the graph of a position from board and facts, kept step by step, and the history as changes with keys; the rules write the stepping relations and the tags through its door; the wake decides as before (guard). Not yet: the wake reads no tags and builds its root graph more than once; see the 79.21 READ. Plan 79.22: the wake builds its root graph once and lends it; the game's memory keeps the position (history with changes, tags, graph) between decisions; the record of repeats is layer 1's; no word of a game in layer 1, scans widened; see the 79.22 READ. Plan 79.23: layer 2 is cs_game with its interface (cs_game::rules) and a test of being done against an outside reference in three games, no disagreement; the graph's maker and keeper sit in it; see the 79.23 READ. Plan 79.24: layer 3 is cs_value, one question with the value by kinds, the plain count and the measure of progress behind it, tested from fixtures; the search-bound part of the measure stays above; the wake reads no value from it yet; in the sample the value and the measure do not say the same; see the 79.24 READ. Plan 79.25: layer 4 is the search in cs_tactics behind one entry (meansends), tested on budgets of nodes on positions that are not quiet with no false proof; callers still reach past the entry; two faults found and left for plan 79.26; see the 79.25 READ. Plan 79.26: layer 4's test is read with the wake's judge beside the plain count (the fault with units given up and units changing kind was the plain count's); the deepening stops on an end proved by force whatever was given up; the whole decides as before; see the 79.26 READ. Plan 79.27: the entry of layer 4 is meansends::ask, an ordered list of goals (a statement of layer 1, the end, a gain) with one budget and one answer for each; statements are read in one and-or tree on the live graph; tested against a slow reference on two games; no caller but the test yet; see the 79.27 READ. Plan 79.28: reached means held after every reply; touched (held for one ply and undone) is a fourth answer; the pattern language can say two placeholders differ; see the 79.28 READ. Plan 79.29: held and touched are read in one tree (the nodes fell below 79.27's); the test's goals no longer include ones that mean nothing; see the 79.29 READ. Plan 79.30: layer 0 is the crate cs_base below the representation (logic with the pattern language and matcher moved down, sets, number, structure, chance, opposed choice), its laws proved in Lean on models, tested with fixtures of two unlike games; only its logic is used by the engine that plays; mixed choices are not built; see the 79.30 READ. Plan 79.31: layer 0 made right: an order for each chooser with opposed and allied defined, the mixed choice (exact), the belief and its change, exact fractions, no dependency; layer 4's reached agrees with layer 0's can-force on the game at hand to three plies; nothing above uses the new readings; see the 79.31 READ. Plan 79.32: the pure search is layer 0's: cs_base::choice::walk is one walk over a node that gives its options, and layer 4's reach is a case of it (the position and the steering in cs_tactics, the recursion, the kept answers and the marks in cs_base), with every answer and count as before and no rise of time; the explicit tree is a second case; the readings that weigh numbers are a fold beside the walk and the other copies of a walk remain; see the 79.32 READ. Plan 79.33: cs_base::choice::fold is the general walk (best, worst, expectation, a window); the search toward claims, the tail of a line and the other side's gain in the forcing read are cases of it through tactics/stepping.rs, answers identical; layer 0's value fold, the search inside a frame and the steps walk stay apart, each with a measured cause; the law of the window is open in Lean; layer 0 is frozen from here; see the 79.33 READ. Plan 79.34: cs_rep (layer 1) names no relation of the rules and keeps a door (Board::write with a Rel, a slot and a word) for a layer above; cs_game (layer 2) owns bear, may_move, shut, through and the turn's fact word, their making and their fact form (relations.rs, graph_facts.rs, graph.rs); the start is layer 2's; unit tests for both layers; no answer changed; see the 79.34 READ. Plan 79.35: cs_game says Chess960, horde, atomic and antichess as well (descriptions under rules/variants, certificates, fixtures, unit tests); a game gives its starts by number (starts.rs) and a start is a delta of starting facts (rules::restart_delta); no source of cs_rep changed; the engine does not play these games yet; see the 79.35 READ. Plan 79.36: every public item of cs_base, cs_rep and cs_game is called by a unit test, kept so by a gate over scripts/layer_units_print.py; every kind in every game has its table of moves; three faults found, the one in layer 0 mended, the two about a game's rules waiting for the description to be the source; see the 79.36 READ. Plan 79.37: a translator from a description to Rust exists (discovery/gdl_to_rust.py), knows no game, and gives the evaluator's answers on whole domains, fixtures and seeded games for two games; the plain translation is about a thousand times slower than the kernel; the rules are about four fifths of a decision, more than half of it the question whose turn it is answered by making every move; nothing translated is on the engine's path; see the 79.37 READ. Plan 79.38: four general passes of the translator (dense tables, answers on demand, kept rows, new rows first), each held piece by piece to the plain translation with no disagreement; together about two and a half to four times the plain translation, the kernel still some hundreds of times faster; the form of the data, rows in hash sets, is read as the cost; see the 79.38 READ. Plan 79.39: the game interface answers whose turn it is from the state alone, for any number of roles; the shared search and 8 more places ask it, 100 askers keep the old question with a cause; the search makes about three times the nodes a second and both players search more than a ply deeper; answers of layers 1 to 3 identical; against the old player the whole did not gain and set A fell, a repeat on a quiet machine owed; see the 79.39 READ. Plan 79.40: the translator holds a relation over small sorts as words and a clause over them as word operations, finding the sorts from the clauses alone; every pass together is about six times the plain translation in the first game and thirteen in the second, the kernel still about two hundred times faster in the first game and seventy in the second; the state is still rows; nothing translated is on the engine's path; the repeat of the measure on a free machine did not show the fall of set A again; see the 79.40 READ. Plan 79.41: four more passes (a key scanned by the words held, a clause stopped at the first row, a walk closed by whole-word shifts, the position after a candidate read as the base words plus its change); every pass is about fifteen times the plain translation in the first game and twenty-eight in the second, the kernel about eighty-five and thirty-one times faster; the lines are still walked for every candidate; checked by the lean check, no measure; see the 79.41 READ. Plan 79.42: three more passes (a clause's start chosen by the words held, a walk kept as its end words, a word set into other orientations by stride); every pass is about twenty-four times the plain translation in the first game and forty-five in the second, the kernel about fifty-two and nineteen times faster; the gain of a plan of passes is falling; see the 79.42 READ. Plan 79.44: a lemma store beside the descriptions exists (rules/lemmas/<game>/): one form for lemmas about the rules and about play, kinds SAME and IMPLIES, each lemma held by a certificate over fixtures and seeded games and given its Lean obligation over the generated relations, its standing computed; three hand lemmas, among them the cases under which a candidate leaves the leader exposed (certified, not proved) and one proved; nothing reads a lemma yet; see the 79.44 READ. Plan 79.43 (merged after 79.44): three last passes (a walk made only for the keys asked, a literal read from a projection, an orientation of a fixed relation made once); every pass is about thirty-nine times the plain translation in the first game and seventy-seven in the second, about two hundred and ninety times the evaluator, the kernel about thirty-two and twelve times faster; what is left is mostly the form of the module's interface (the state as rows, a move as a term), not a fact about the game; no more plans of passes alone; see the 79.43 READ. Plan 79.46: the translated module is the checker of the fast copy: the kernel's five answers for the first and second games agree with it on about 12.3 and 5.0 million states of seeded random games, no disagreement, kept as records apart from the twelve certificates; the placed questions and five games are not covered; see the 79.46 READ. Plan 79.45: the translator reads a certified SAME lemma as a rewrite behind a flag and a pass rests the pieces the flag shuts; with the lemma of the cases the first game is about 1.03 times, the gain eaten by a walk the lemma reads twice; the lemma agrees on a million states by translated code; a clause no state reached was found and a fixture added; see the 79.45 READ. Plan 79.47: five more games went through the translator unchanged, so seven games have a translated module held to the evaluator and the kernel's record against it with no disagreement; a module can be begun again at other initial facts (`Game::restart`); the pass asked takes a walk read twice, and the first lemma then makes its module about 1.4 times as fast; the kernel's bridge gives three-check's count of checks, the one source on the engine's path that changed; nothing translated is on the engine's path; see the 79.47 READ. Plan 79.48: the placed questions of layer 2 are said over the description (rules/placed) and the kernel's answers are held to the translated module in seven games; one fault found, the first move of two places missing from where a unit may move; a gate lists every fast answer of layer 2 against what holds it, with 22 named exceptions; see the 79.48 READ. Plan 79.50: where a unit may move now has the first move of two places as the description says; the placed records hold no known cause. Plan 79.51: the future channel of section 7 exists in a first form (cs_decision::future: the stop with its cause, goals on a line some unmet facts away, forcing moves, the join, the near read), computed and logged beside a decision and read by no choice; most of what it calls a threat now does not close when read; see the 79.50 and 79.51 READ. Plan 79.49: the prover harness takes a rule lemma's obligation and a found proof is adopted through the lemma's document; no rule lemma has been proved by it; see the 79.49 READ. Was planned as: the pick, the heads, the carried tactics and the wake's loop (79.08). Order of testing, his: Rust unit tests a piece, then integration of the whole, then positions |
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
27. ANSWERED 2026-11-19 ("Yep, you got all of that right."): the six points of section 7 stand, with the fixed rule first and the learned
    chooser after the whole wake has logged games, and estimates per frame and goal. Item 23's proposal is withdrawn by this. Was: whether the
    reading above is right, above all point 2 (estimates per frame and goal, which would replace my proposal of item 23) and point 5
    (a fixed rule first, the learned chooser after the whole wake runs and has logged games).
26. From 2026-11-19, a question you asked and did not settle: "how do we spend effort chasing a mate that is very far off that we can't
    really reason our way around because there's just too many plies out, or do we chase the smaller stake that is more concrete?"
    What the spec already says (section 7, 2026-11-19): the stake orders goals, not the length; a budget of questions; past it only
    where progress is made; too far means break it down. The proposer's own default orders by the fewest unmet facts first, which is
    the opposite. PROPOSED by the main agent, not decided: two passes. The ASKING is in order of stake and costs almost nothing (the
    graph alone: how many things stand in the way of each goal). The READING, which is what costs, goes to the goal with the fewest
    things in its way, the larger stake breaking ties. A far goal of high stake is never read as a line; it is carried from move to
    move and gives the reading one thing only, its nearest stopper as a subgoal, which is small and concrete. So the far goal chooses
    WHICH small goals are worth reading, and the small goals get the reading.
29. From plans 79.11 and 79.12 (2026-11-19). Choices of the builder where the spec is silent, kept until you say otherwise: the reread of
    fair_names is at the root only and only where the best root move is a named one; the carry is kept by the binding in a slot named by
    the caller; the old player is brought to the wake's mean wall by one scale found in a sanity match; a watchdog of 300 s a game.
    PROPOSED by the main agent for the chooser of section 7 point 4, kept until you say otherwise: (a) the chooser picks only among the
    moves the calculation holds equal, so the wall of section 8 stands for it as for the fixed rule; (b) its training games vary the
    choice among equal moves at random and log the propensity, because under the fixed rule the log cannot say what another equal move
    would have brought; (c) its first form is one readable weight a named input, fitted in sleep on the game's result only, on the
    inputs of the move played less the mean of its equal set; (d) it is played only if a weight is clear of nought on held-out games.
30. From plan 79.12 and for plan 79.13 (2026-11-20). PROPOSED by the main agent, kept until you say otherwise: (a) the estimate per goal of
    section 7 point 2 is first a CHANCE THAT THE GOAL IS REACHED, counted from our own logged games: reached when the target is taken
    by the goal's side within some plies without that side falling behind; no oracle. (b) Among equal moves the plain reading of your
    rule is the largest gain in stake times chance over our chains less the same for theirs. (c) A learned chooser is played only when
    its fit beats the same fit on shuffled results, not when one weight happens to be clear. A QUESTION FOR YOU: about half the games at
    0.2 s a move reach the cap of 120 plies with no result; say if training games should run longer or be judged another way.
31. From plan 79.13 and for plan 79.14 (2026-11-20). READINGS of the main agent, kept until you say otherwise: (a) training games run to
    a cap of 200 plies (95.5 pct end on a result against 52.0 pct at 120); judging matches stay at 120. (b) The window in which a goal
    counts as reached is 16 plies in the wake, with 8 and 60 kept beside it in sleep. (c) A goal whose facts are all met is not dropped:
    it is the nearest goal. A QUESTION FOR YOU, not opened by me: three plans show that choosing among equal moves does not move the
    result at this clock. If the ply comes back in plan 79.14 and the whole is still level, the estimates per goal and frame have no
    place left to act except the end of the search's lines, where today only the count of worths stands. Rule 7 and section 8's wall
    say subgoals never enter the value. Say if an estimate per goal may stand at a line's end where the calculation stops, or how
    else you mean the estimate and the calculation to be decided together there.
    ANSWERED BY THE SPEC ITSELF, the main agent's error (2026-11-20): section 6 ("the estimate scores a leaf") and section 8 step 5
    decided on 2026-11-15 and 2026-11-16 that the chances are read where a line ends; what stays out of the value is a subgoal
    BONUS. The wake does not do it (ledger row 5). The question should not have been asked.
32. From plan 79.14 and for plan 79.15 (2026-11-20). Choices of the builder where the spec is silent, kept until you say otherwise: the
    switches carry_graph, chance_order and gain_mend sit in the wake's inputs and are off by default; a carried row counts against the
    question budget as used; a goal is met when no fact is unmet and none open. READINGS of the main agent: (a) the wall of section 8
    covers the probe of the equal reading: a first move the probe put lower is not played while another move's probe holds, verified
    or not; (b) a carry that does not equal asking afresh is not turned on; (c) the share of the clock the steps before the search may
    take is chosen by play of the whole, every piece on at every share; (d) a line standing on the mover's goal unit is a standing
    threat and is named as such in the reason.
33. From plan 79.15 and for plan 79.16 (2026-11-20). Choices of the builder where the spec is silent, kept until you say otherwise: a
    first move the verification dropped counts as lowered like one the probe lowered; where no other move holds the move with the
    highest probe plays; the share of the steps is a scale of the default share. READINGS of the main agent: (a) your decision that
    the whole is judged means judged at a clock where every piece finishes its reading; at 0.2 s it does not, so the judging is done
    also at 2.0 s a move until the steps are fast enough; (b) time a step does not use goes to the search; (c) the search reads on,
    within a bounded share, when its first move fails at the next depth; (d) among fixes the rule ranks by stake times chance when a
    table is given, as the names are. If the whole is level at 2.0 s too, the main agent stops for your answer to item 31.
34. From plan 79.16 (2026-11-20). STOPPED FOR YOU. Six plans (79.11 to 79.16) leave the whole level with the bare search at equal time.
    DIAGNOSIS of the main agent, not proved by a run: (a) every goal the proposer makes is the taking of a unit within one or two
    moves, which the search has already read, so the goals tell it nothing; a player's goals lie beyond what they can calculate; (b) a
    goal reached counts for nothing at a line's end, where only the count of worths stands (item 31); (c) no goal is held from one
    move to the next; (d) the goals a player has were learned from games, and ours are not. PROPOSED, in this order, none begun:
    (1) mine from master games the facts of the graph that become true, stay true and go with the result at equal count of worths
    (78.80 began this and found signal with artefacts), read against the board; (2) a reading toward one such goal: our own moves
    that reach it, their best reply checked at each step; (3) the goal held across moves until refuted; (4) a line's end judged by
    goals reached, as an estimate fitted on results only, which needs your word because of rule 7.
35. From Kaveh's words of 2026-11-20 (section 7, last block before section 8). PROPOSED by the main agent, kept until you say otherwise:
    the ledger `docs/SPEC_LEDGER.md` and its six rules of use; that the next plans are taken from its rows 5, 13, 23, 24, 25 before
    anything else. QUESTIONS: (a) "from our own play" and "from master games": the reading taken is both, human games by rating;
    say if one comes first. (b) What rating counts as master for the accurate work, given that the human file is of all ratings.
36. From Kaveh's go of 2026-11-20 on the small endings (section 7, last block before section 8). READING of the main agent: the go
    covers the endgame table in sleep and in the measure (the label of a position as won; the defender in the test), never in the
    wake. Say if the table may not be used so.
37. From plan 79.17 (2026-11-20). Choices the builder made where the spec is silent, kept until you say otherwise: the defender of
    the test is the table's reply with the largest distance to the next irreversible move; the miner's first test is ranked by the
    share of turns at which the term is lower within the next three turns; the goal's question reads up to three of our moves and
    gets half the share of the steps before the search; a game of the test ends at the third repetition. Two choices the main agent
    holds to be faults and will mend: the fall is counted from the value now, and a dropped goal is not taken up again.
38. From Kaveh's words of 2026-11-20 on the layers (section 11, last block). The main agent's answer, PROPOSED: the crates are
    close to his layers but not the same (game and gdl hold representation and rules together; cs_tactics holds the base graph, the
    frames and the search in one crate; cs_decision holds goals, the rule for choosing and the wake in one crate; memory is its own
    crate; the opponent and the strategic layer have no crate). No layer has a test that says "this layer is done". Missing from his
    list, to be placed: value (results, worth), the estimator, sleep (learning), and the measure of what the whole does. His to decide.
39. From Kaveh's second pass on the layers (2026-11-20, section 11, last block). Two questions of the main agent. (i) "Value is
    the measure of progress": today the value is the chance of the game's result, fitted on results, and a subgoal or a measure may
    order and choose but never enters a line's value (AGENTS rule 7; plan 79.17). PROPOSED reading that keeps both: one layer holds
    every measure of progress, each toward a stated goal; value is the one whose goal is the result of the game; a measure toward a
    subgoal still enters by the order and the choice. Say if instead a subgoal's measure may now enter a line's value. (ii) The
    language of frames (placeholders) is representation; do the frames found by learning live in layer 3 with the statistics, or
    in a layer of their own.
40. From plan 79.19 (2026-11-20). Choices the builder made where the spec is silent, kept until you say otherwise: a counted table
    is a belief with its count, the worth record and the reasons are conjectures on a named sample; a statement's read set is every
    literal of its pattern; the rules' address is computed from the rules' text the caller passes; the wake names the measure it
    holds by an id. Found by the layer's test: the position's text form is not exact for one fact of the rules; the record of moves is.
41. From plan 79.20 (2026-11-20). Two things for you. (i) You said the representation is a graph. Today the base graph is built
    from the rules (their steps and a game's state), not from facts, so it is not layer 1. PROPOSED: the graph is built from the
    facts of a position plus a static part of the board that the rules hand over once (places, lines, steps), so layer 1 owns it
    and the rules only supply facts. (ii) A fact's type lives in the crate of the rules. Moving the representation into a crate
    below costs making the kernel's certificates again, because they are keyed by the lock file. PROPOSED: do it, once, together
    with (i).
42. From plan 79.21 (2026-11-20). Item 41 (i) and (ii) are built. Choices the builder made where the spec is silent, kept until
    you say otherwise: layer 1 uses the parser of descriptions to read the board (taken to be below the rules); the tempo is one
    declared relation and a tag is every other fact that is not a unit; a tag is a statement whose one literal is the fact itself;
    in layer 1's tests a position's key is the content address of its facts, in play it is the rules' key.
43. From plan 79.23 (2026-11-20). For you. (i) PROPOSED by the builder: a position's key reads the right to take in passing only where
    a taking is allowed; the text form writes it always. (ii) A fault in the protected fast kernel, not mended: in one variant an
    inconsistent position makes a counter wrap below nought; not reachable by play. (iii) A repeat has a limit in layer 2 but no
    result for the sides there. (iv) Callers above still reach past layer 2's interface; closing that changes the binding.
44. From plan 79.24 (2026-11-21). For you. (i) Value and measure do not say the same in the sample: the value tells who is ahead
    and gives one number inside a won ending; the measure orders inside its ending and means nothing outside its class. Two classes
    only, so your expectation is neither shown nor refuted. (ii) Along the table's quickest win the measure rises about a quarter of
    the time: the measure and the quickest win are different roads; the measure was mined so that a fall exists, not that the
    quickest win makes it. (iii) The value is worse than a plain count on generated endings (it was fitted on quiet rows of human
    games). (iv) Choices of the builder: the question takes the game and state, not the graph; an ending is at most 8 units.
45. From plan 79.25 (2026-11-21). Choices of the builder where the spec is silent, kept until you say otherwise: a position is not
    quiet when the outside engine's best move beats its second by 150 hundredths at depth 15 and each move of ours in the solution
    is a taking, a threat to the chief unit or an only move; budgets of 5000, 50000 and 500000 nodes; the whole line counts as seen
    only if every move of both sides is equal; the test's judge is the plain count. Found: what in the means-ends crate belongs to
    the layer above (the sheet, the census, the reasons, the scenarios) is listed, not moved.
46. From plan 79.26 and your word of 2026-11-21 ("Layer 4 needs to work towards the sub-goal"). Layer 4's entry takes no goal:
    its one built-in aim is to take units or end the game. DECIDED by Kaveh 2026-11-21 that the entry takes a goal as an input (section 11, last block; plan 79.27); the form is the main agent's reading: the question to layer 4 carries a goal, a
    statement in layer 1's form that the side wants made true; the search reads the moves that serve it and the replies to them
    and answers reached by force, refuted, or not settled; taking a unit and ending the game are two goals among others; the
    value still judges the line's end (rule 7 stands); the layer above chooses which goal to ask. Also open: with the wake's
    judge the layer claims a gain in 9 of 61 positions the outside engine calls quiet; not yet read one by one.
47. From plan 79.27 (2026-11-21). Choices the builder made where the spec is silent, kept until you say otherwise: a goal is
    reached when its pattern binds at any position of the line, the root included, for one ply or more (DECIDED otherwise by Kaveh 2026-11-22, section 11 last block, plan 79.28; as proposed by the main
    agent: reached means it still holds after the other side's best reply, and what holds for one ply only is said apart);
    the budget is shared equally between the statements as one group and each old goal; one tree holds at most 64 goals; "one
    reply meets both" means a reply after which both goals are out of reach within the plies left. Open: the pattern language
    cannot say two different units; the search of 78.36 and the new one share their deepening and memo as two copies.
48. From plan 79.28 (2026-11-22), read on the board by the main agent. Three things for you. (i) The line layer 4 gives for a
    goal is the shortest, whatever it gives up; the answer says what it took, but the question cannot bound it. PROPOSED: the
    question carries, beside each goal, the most the side may give up by the judge, and the answer is the shortest line within
    it; this is a bound on a line, not a goal entering a value. (79.29: the bound must hold on every reply, and the cost said is the worst over the replies, not that of the one line shown.) (ii) A double goal under the new meaning: PROPOSED: two goals
    each touched by one move, such that no one reply undoes both; then one of them is reached a move later. (iii) A goal about
    a unit names the place the unit stood on when the goal was written, so it is lost when the unit moves. PROPOSED: a goal
    names a unit by a mark that follows it through its moves (layer 1's history knows where each unit went).
49. From 2026-11-22. Your direction: a belief about the opponent, and choosing by chance. GO given 2026-11-22 (plan 79.31);
    was proposed as: add to layer 0 (i) the mixed choice for a choice at once (the best spread over one's options and the one value, by
    a small linear program; the law that the two sides' guarantees meet stated in Lean, proved or left open and said), (ii) the
    belief: a spread over suppositions, its change by what is seen, and the best choice against it. What is believed of a given
    opponent is left to the layer of the opponent.
    (iii) An order for each chooser in the value reading, with opposed and allied as named cases, so that layer 0 assumes no
    winning and losing; that the game at hand is one of opposite wants is said by layer 2.
50. From 2026-11-22. You agreed the pure search is layer 0's. GO given 2026-11-22 ("Agreed. Go"); plan 79.32, to be written and
    started when plan 79.31 is merged, since both change the same files. Was proposed as: the search of layer 4
    is rebuilt as a case of layer 0's algorithm over a node that gives its options on demand, with layer 4 keeping the order of
    options, the narrowing and the budget; nodes and answers must not change and the time must not rise (measured, mended if it
    does). It touches the most speed-sensitive code we have.
51. From plan 79.32 (2026-11-22). GO given by Kaveh 2026-11-22 ("Do it"); plan 79.33. Was proposed as: give layer 0's walk a general answer (a number by best, worst and
    expectation, with a window), so that the value readings of layer 0 and the other walks of the tactical layer (toward.rs,
    lines.rs, sheet/frame.rs, sheet/forcing.rs) become cases of the one walk, each moved only with identical answers and no rise of
    time. The old player's solver is never changed and stays a copy. Choices the builder made where the spec is silent, kept until
    you say otherwise: two readings in one pass (strict and loose); cs_base a plain dependency of the tactical layer; a turn with
    no option forces nothing.
28. From plan 79.09 (2026-11-19). Choices the builder made where the spec is silent, kept until you say otherwise: the steps before the
    search get what the search and its equal reading leave of the clock; a line cut by the clock is a gap and never a refutation; an
    estimate that did not reach a move does not decide. Three readings of your words that plan 79.10 builds on, mine: when every fix of
    the nearest goal of a chain is refuted the next goal of the chain is asked; the other side's goals are read after each move of ours
    and not after a pass; a named move is read by the search itself and not by a reading of its own first.
25. From plan 79.06 (2026-11-19). Choices the builder made where the spec is silent, kept until you say otherwise: until the frame read
    is in Rust the rows of the threat list come from the graph (every unit an opposing unit bears on); the stake is read by a function
    the caller gives. Two behaviours carried over from the Python that look wrong and are to be settled at the integration tests: a
    response whose result only equals the threat's is called kept; a quiet first move that ends level is dropped, not an exchange.
23. WITHDRAWN 2026-11-19 by item 27: the next estimator is per frame and per goal, not the same heads on more positions. Was: From plan 79.04 (2026-11-19). A DECISION FOR YOU: how the estimator is fitted next. The heads were fitted on quiet positions only, so
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

- OPEN (2026-11-24, future prediction): what it means that force can arrive at a target, beyond units on a line and the reach of row 16. One example is held (the fixture of 2026-11-24). To be defined from more cases, with Kaveh.

- OPEN (2026-11-24): the game of preparation (each side guessing what the other prepared) is an aim with no design; energy as a cost is named and not measured; the work of others listed in section 7's last block is to be read before a plan is written.

DECIDED 2026-11-24 (section 11 last block, (vii)): (a) and (b) together, the top-down way beside the bottom-up one; (c) stays true of the engine's path until a translated module is certified fast enough. The item as it stood: the two steps of plan 79.52 part B gained a few per cent; the reading says nobody has brought a general
description to a hand-written library's speed at chess. Kaveh to choose: (a) tables for walks, found in the description by the
translator; (b) a new back end that walks depth-first and changes the state in place; (c) the outside library kept as the certified fast
copy where one exists, the translated rules fast enough for games that have none.

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
| Calculation where it is near and estimate by sub-structures where it is not; several estimates; a learned last layer decides from both; a fixed rule first, the chooser trained after the whole wake has logged games | 2026-11-19 | 7 |
| One way of asking for every goal: what stands in the way, can it be removed; a stopper is a target and that is the subgoal; the defence is the same read from the other side; no word special to the goal unit | 2026-11-19 | 7 |
| The chooser's first form and its training games: only among equal moves, varied at random with the propensity logged, one readable weight an input, results only (PROPOSED, main agent) | 2026-11-19 | 13 item 29 |
| The estimate per goal is first the chance that the goal is reached, counted from our own games; a learned chooser is played only if it beats the shuffled fit (PROPOSED, main agent) | 2026-11-20 | 13 item 30 |
| Training games to 200 plies; the goal window 16 plies in the wake; a met goal is the nearest goal (READINGS, main agent); where the estimate may act if not among equal moves (QUESTION) | 2026-11-20 | 13 item 31 |
| The wall covers the probe: a lowered first move is not played; an unequal carry stays off; the steps' share is chosen by play of the whole (READINGS, main agent) | 2026-11-20 | 13 item 32 |
| The whole is judged at a clock where every piece finishes; unused time goes to the search; the search reads on when its first move fails; fixes ranked by stake times chance (READINGS, main agent) | 2026-11-20 | 13 item 33 |
| Stopped for Kaveh: the goals made lie inside the search's horizon; goals mined from games, read toward, held, and counted at a line's end (DIAGNOSIS and PROPOSED, main agent) | 2026-11-20 | 13 item 34 |
| Goals beyond the search's horizon broken down into it; a reached subgoal is a training signal; subgoals mined as the tactics whose existence leads to gain; human data by rating; the ledger against forgetting | 2026-11-20 | 7 (before section 8), 13 item 35, docs/SPEC_LEDGER.md |
| The small endings as the test of means and ends; the October measure of progress was not brought into the new engine (DIRECTION, Kaveh; ANSWER and READING, main agent) | 2026-11-20 | 7 (last block before section 8), ledger row 35 |
| The go on the small endings: the order of the build; the table in sleep and in the measure only (READING) | 2026-11-20 | 7 (last block before section 8), 13 item 36 |
| Plan 79.17: the measure of progress in the new engine, mined, held by the wake; two faults that stand | 2026-11-20 | 12, 13 item 37 |
| The layers, each with its work, interface and test (Kaveh thinking out loud, UNDER DISCUSSION) | 2026-11-20 | 11 (last six blocks; layer 1 DECIDED, plan 79.19), 13 items 38 and 39 |
| A layer's test uses only that layer and what is below it; layer 1's tests call nothing of the rules | 2026-11-20 | 11 (last block) |
| One graph is the representation (board, lines, units, tempo); the rules are the next layer | 2026-11-20 | 11 (last block) |
| Layer 1 holds the history as a sequence of changes; rights and counters are layer 2's tags on the graph | 2026-11-20 | 11 (last block) |
| A slower result is diagnosed and mended, not set aside | 2026-11-20 | 11 (last block) |
| The order of work is layer by layer from the base, each made like layer 1 | 2026-11-20 | 11 (last block) |
| Layer 3 holds two implementations, value and the measure of progress; their agreement is measured | 2026-11-21 | 11 (last block) |
| Layer 4 takes a goal as an input; the layer above chooses it | 2026-11-21 | 11 (last block) |
| Layer 4 is asked with an ordered list of goals and answers each; it does not choose by worth | 2026-11-21 | 11 (last block) |
| Reached means the goal outlasts the reply; held for one ply is touched | 2026-11-22 | 11 (last block) |
| Layer 0 is mathematics and logic; layer 1 what is seen in one state; layer 2 what links states; 1 and 2 by hand for now | 2026-11-22 | 11 (last block) |
| Layer 0 must be able to say two unlike games of opposed sides; the layers above serve the game at hand (contents of layer 0 PROPOSED) | 2026-11-22 | 11 (last block) |
| Layer 0 holds six things (logic and proof, sets, number, structure, chance, opposed choice); the go to build it | 2026-11-22 | 11 (last block) |
| A belief about the opponent and choosing by chance: layer 0 gives the form, the layer of the opponent the content (direction; details PROPOSED) | 2026-11-22 | 11 (last block), 13 item 49 |
| Layer 0 made right and applied to the game at hand; nothing of a second game built; open to extension | 2026-11-22 | 11 (last block) |
| The pure search (the reading of a tree of choices) is layer 0's; options, goal, order and budget are not | 2026-11-22 | 11 (last block), 13 item 50 |
| Plan 79.32 left two recursions in layer 0 (steps, and values); one general answer is proposed | 2026-11-22 | 11 (last block), 13 item 51 |
| The base is frozen after 79.33; the loop is the work; one measure of the whole in every READ | 2026-11-22 | 11 (last block) |
| The cut of layers 1 and 2 (what is there; what the rules say of it) is carried into the crates, with unit tests | 2026-11-22 | 11 (last block) |
| Layer 2 says variants (another start: Chess960, horde; other rules: atomic, antichess) on an unchanged layer 1 | 2026-11-23 | 11 (last block) |
| Every unit of a layer is tested before the layer above is taken up; what was checked by hand becomes a test | 2026-11-23 | 11 (last block) |
| The description is the source of the rules; fast code is only a certified copy, and the tests fail without the certificate | 2026-11-23 | 11 (last block) |
| The crux of layer 2: one translator from the description to Rust, each piece pinned by tests, then made faster piece by piece against an outside mark | 2026-11-23 | 11 (last block) |
| Layers 0 to 3 may be mended; layer 0 takes fixes and small support, no significant new function | 2026-11-23 | 11 (last block) |
| Layer 2 answers the side to move without reading the end; every asker above uses it, the shared search included | 2026-11-23 | 11 (last block) |
| Several boards and several players are owed later; nothing new assumes two players | 2026-11-23 | 11 (last block) |
| Two checks at a merge: lean off the engine's path, full on it; the measure of the whole only when a source on the path changed | 2026-11-23 | 11 (last block) |
| One lemma machinery used twice: lemmas about the rules feed the translator, lemmas about play feed the layers above | 2026-11-23 | 11 (last block) |
| Future prediction is one channel with a near (concrete) and a far (believed, to be made concrete by evidence) sub-layer; a target is left only when both say nothing; a goal some unmet facts away; frames join through forcing moves | 2026-11-24 | 7 (last block) |
| Result first, then least cost; a way that wins is taken and locked, safe by a falling measure and a cheap check; narrowness of their good replies and steering toward it; the game of preparation owed later | 2026-11-24 | 7 (last block) |
| Lower layers as fast as the outside library first; the facts stay, their storage is free; the list of allowed moves made once a state | 2026-11-24 | 11 (last block) |
| Two ways of answering in one translated module: bottom-up rows and a top-down walk; tables for walks found by the form of the rules; the way chosen by a clock | 2026-11-24 | 11 (last block, vii) |
| Answers carried from the last state in the translated module: the change is the difference of the words, a group of rows is carried while its read set does not meet the change; the count first; the lock of the lower layers after it | 2026-11-25 | 11 (last block, ix) |
| What was replaced | 2026-11-16 | 14 |
