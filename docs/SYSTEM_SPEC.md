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
- PROPOSED, built in part: generation zero knows the rules and the end of the game and otherwise moves at random;
  odds games measure each kind's effect on the result; that record is the next generation's judge; repeat. A
  comparison that needs worth is returned as a CONDITION on the unknowns ("good if this kind is worth more than
  that kind"), kept with the claim. The condition in the frame read is not built yet.

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
  - *Whether it is in the future follows the two layers.* Layer one, me playing only, is a count of my own moves
    until the pattern holds, where a quasi-metric fits and the earlier work on one may be reused. Layer two, with
    the opponent's veto, is a chance: the pattern coming true is one more event class of the heads of section 6.
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
| The ladder under each measured worth record | Measured (plan 78.76): the odds record, the self-play record and the human-result record give the same division within noise; the worths are not the main cause of the open surprises. Open ones are positions where nothing is taken inside the reading. CORRECTED by plan 78.77: a unit changing kind IS an event of the ledger (since plan 78.5) and only the print left it out; the line read and the search are two readings at different depths, and the printed line now follows the search's move |
| The goals as a hint among equal moves, scored on the open surprises | Measured (plan 78.78): silent in most tie positions by layer one; no signal above the base rate by stoppers. Missing: goals that stop the other side's goals, and a count for long walks |
| The other side's goals as ours to stop, the long count, the replied stoppers | Measured (plan 78.79): defence and replied stoppers show no gap between accepted and not accepted moves; the long count is undecided |
| Symbols that go with winning as subgoals (fact patterns, two tests, a wider yardstick) | Built as a measurement (plan 78.80). Events of a few hundred patterns pass both tests on unseen games. The pick agrees with the move played a little more than the search's first move and is ahead on the ladder's scored part, both bands clear of zero, one run. The strongest symbols are the other side losing places, carried mostly by takings and thrusts; one reads as a purpose. Open: redundant twins, the taking confound, thin endings, layers one and two (plan 78.81) |
| The menu, compression (rung 4), the chooser | Written, not built |
| Quiet or sharp by two readings; open tactics checked before the estimate; carrying measured | Measured once (plan 78.69): the difference of the readings not shown as a signal with the present heads; open claims of the frame read go with events soon; carrying first measured by a proxy, then truly (plans 78.70, 78.72): a claim whose read set did not change is held again only about half the time |
| The threat list (threats with stake and response, in four bands) | Built concretely (plan 78.72, after a first form in 78.70): each row and each response is a line read to a settled end and judged by the measured worths; exchanges print apart; claims nothing bears on are dropped; the end of the game is a row for either side. Like for like cancels in the verdict and responses carry their own end and are ordered (plan 78.73). Still short of section 8: a clear gain can print as an exchange because the measured worths are coarse, many positions print no row, threats that need a quiet first move are rare, and the top band is not the most dangerous. Carrying by the frame's read set does not hold for a concrete line; the line's own read set is not built. Not wired into play |
| The frame's interface (ports for the intermezzo, the pass, the deflection) | Not built beyond the named causes of plan 78.60 |
| The rolling measure across sleeps and going back to a past learning state | Not built; snapshots of the store exist |
| The opportunity policy, the subgoal vocabulary, holding a subgoal across moves, the rule for choosing a move | The proposer is built as a measurement (plan 78.71 A to C): goals as facts with the cause and the fix, the fix checked as a what-if edit by our own reasoning, a first grading by hindsight. The diagnosis holds. A goal for every unit of the other side and the goal unit, the fix as a set, reach carried in two layers (plan 78.74): the first three goals hold the unit then won more often than chance; the printed order is not yet a player's (goals through several units between rank first); layer one is crude. The rule for choosing a move is built as a sandbox player (plan 78.75) that picks among moves the search holds equal: in matches it was no better than the search's own choice, a random pick or the arrival number, at half a second a decision. The threat half hardly ever decided, since a real threat leaves no tie; the goal half changed most choices without gain; a goal was held about one decision. A decision took about twenty line queries. The policy as a learned part is not built |
| Earlier work on subgoals to be reused, all on older systems | The what-if edits (catspace/incubator/analyst/whatif*.py); the October vocabulary of why a fix fails; the subgoal former and the waypoint decomposer of the summer (docs/SUBGOALFORMER.md, subgoal_cascade), built on a learned embedding; the October what-if edits were bound to the game and to assumed worths, so only their idea was reused in plan 78.71; the subgoal fit and play of plans 78.49 and 78.50. Subgoals themselves were never graded |
| The unknown as a condition in the frame read; the sequence model; the endgame fill from human play | Not built |

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

## 15. Standing limits that apply to every plan

No number of the oracle in a fit or in the wake, unless the last resort of section 9 is taken on his word. No
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
| Subgoals as symbols that go with winning: fact patterns first, two tests on human moves, a graph network with quantized codes only if they fail; in the future or not by the two layers | 2026-11-17 | 7 |
| Human games may be read; the endings risk and the fill | 2026-11-15 | 10 |
| Rust first; game-free layers; Lean; reuse; the real store | 2026-11-14 | 11 |
| Every build checked against the spec; modules with stated interfaces; no quiet departure | 2026-11-16 | 11 |
| The pointing at a weakness and the choice of subgoal are from the graph only; language is the last step, for a reader | 2026-11-16 | 7, 11, 15 |
| What exists and the state of the build | 2026-11-16 | 12 |
| Open questions | 2026-11-16 | 13 |
| What was replaced | 2026-11-16 | 14 |
