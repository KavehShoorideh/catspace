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
