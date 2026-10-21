# Review: "replies as one forced chain; exact vs estimate decided per node" (Kaveh, 2026-10-21)

Status: review only, no build. The idea is recorded in memory (`exact_or_estimate_judge_2026_10_21`). Literature by a search
agent (four papers read in full: Thomsen 2000, Cazenave GAPS 2002, Winands et al. 2008, Hay et al. 2012; the rest from
abstracts). Code facts from `regress.py` and `agenda.py` as of 9d8b0037.

## 1. The idea in four claims

1. A quiet move after which every reply loses is not "unforced": the replies are one chain. Each reply is **immediately
   forced** (a threat is on the board now) or **delayed forced** (a threat we can set up whatever they do).
2. Exact reading of that chain blows up if done everywhere; reserve it for nodes with a handful of lines; beyond, estimate.
3. A per-node **judge** chooses exact vs estimate from the predicted **cost** of exact reading and the predicted
   **accuracy** of the estimate, both known ahead of time.
4. The judge re-asks at every node, so there is no hard time budget: when slow, switch to the estimate ("like AlphaZero's prior").

## 2. Literature: what is established, what is open

**Claim 1 is Thomsen's lambda search** (Thomsen 2000, ICGA J. 23(4), https://www.t-t.dk/publications/lambda_lncs.pdf).
A lambda-n move is one such that if the defender *passes*, a lambda-(n-1) tree wins. lambda-0 = the goal in one move,
lambda-1 = a chain of direct threats (our "immediately forced"), lambda-2 = a move that sets up a lambda-1 chain (our
"delayed forced"); several alternative lower-order threats are allowed. Theorem 1: a lambda tree that returns 1 is a true
minimax win (a proof). Theorem 2 (completeness) needs passing to be harmless, so **zugzwang wins are invisible**: a quiet
move whose replies all lose *with no threat to parry* is not a lambda move. Claim 1 as stated ("the replies form one chain")
is false for pure zugzwang; true otherwise. Cost: the null-move test is paid on every legal move; the saving is on the
defender's width (b of lambda moves against B legal moves).

**"Only the parries need reading"** is threat-space search (Allis, van den Herik, Huntjens 1993/1994; Gomoku) and
Cazenave's Abstract Proof Search / Generalized Threats Search / Iterative Widening (2000-2002,
https://www.lamsade.dauphine.fr/~cazenave/papers/gaps.pdf): at every defender node, if the attacker has an order-k win
after a pass, only the defender moves that prevent it are tried. **Trap** (Allis): a parry that is itself a threat
(intermezzo, counter-threat) must be in the defender set or the sequence re-verified against full counterplay; finding no
threat-sequence proves nothing. Dual lambda search (Soeda, Kaneko, Tanaka 2006, shogi) runs the threat machinery for both
sides; lambda df-pn (Yoshizoe, Kishimoto, Müller 2007) uses lambda order as the controller inside proof-number search.
Relevance-zone search (Shih et al. AAAI 2022, https://arxiv.org/abs/2112.02563) reads the defender only inside the zone
the attacker's proof touched: the modern form of our read set. No closed-form result on the width saving; it is
empirical, qualitatively large.

**Claim 3(a), cost known ahead of time**: proof numbers (Allis et al. 1994) are lower bounds on the remaining work, but
computed from the partly expanded tree, not before expansion. Heuristic initialisation (df-pn+, Nagai 2002; Winands &
Schadd 2010, evaluation-function PNS, 45-85% less solving time) and learned cost heads (Gao, Müller, Hayward 2017
FDFPN-CNN; Wu et al. ICLR 2022 "AlphaZero-based Proof Cost Network", https://iclr.cc/virtual/2022/poster/7121) predict
cost from the board. **Discipline in all of them: a cost estimate orders the work; a node is proved only through
terminals.** That is the same rule Kaveh set on 2026-10-20 (learned parts only order). No paper predicts chess solver
cost from static features; the "king has many flight squares, few pieces" features are a hand-written PCN, plausible,
unmeasured.

**Claim 3(b), proved beside estimated**: MCTS-Solver (Winands, Björnsson, Saito 2008,
https://staff.ru.is/yngvi/pdf/WinandsBS08.pdf) keeps +-inf only from terminals; a proven loss needs *all* children
proven; an estimate never becomes a proof; and §4.2 shows pruning proven-loss children biases the parent's estimate, so
proven losses stay in the bookkeeping but out of the estimate. Implicit minimax backups (Lanctot et al. 2014), PN-MCTS
(Doe et al. 2022), HTPS (Lample et al. NeurIPS 2022: AND-OR hypertree, learned prior orders, exact rules propagate
"solved") all keep two channels. Baum & Smith 1997 (evaluation returns a distribution over its own error; expand the leaf
whose expansion changes the root most) and conspiracy numbers (McAllester 1988: how many leaves must change to change the
root) are the published forms of "the estimate's accuracy known ahead of time". B* (Berliner 1979) is the
optimistic/pessimistic-bound version. Ramanujan, Sabharwal, Selman (ICAPS 2010): chess, unlike Go, has shallow traps
that estimates miss; this is the published reason the exact local reading must stay.

**Claim 4, the judge**: rational metareasoning (Russell & Wefald 1991: expected value of computation minus its cost);
Hay, Russell, Tolpin, Shimony (UAI 2012, https://arxiv.org/abs/1207.5879): search as a selection problem with stopping
in the policy; **the myopic policy "has a tendency to stop too early"**, so a per-node "is one more exact step worth it?"
rule under-reads long forcing chains unless it values several steps ahead (Callaway et al. 2018 bracket the learned
meta-policy between myopic and perfect information value). Conditional performance profiles (Zilberstein & Russell
1993-96): quality(time | input features) compiled offline = "accuracy known ahead of time". Time management: Baier &
Winands 2016 found the STOP rule (stop when the best move cannot change) strongest across five games; "no hard budget"
in this literature means *interruptible at any time*, never the absence of a clock.

**Quiescence** (Beal 1990 null-move bound; Schrüfer 1989) is the classical "estimate only where it is trustworthy": a
static value is trusted when no forcing move changes it. Uncertainty-aware search (Bayes-UCB tree search ICML 2024,
epistemic MCTS ICLR 2025) gains only with calibrated uncertainty; no published calibration study of chess value heads.

**Humans**: Campitelli & Gobet (ICGA J. 2004) measured that in complex positions strong players search more and deeper
(up to 10x): the published form of claim 2, against the folk reading of Chase & Simon that experts barely calculate.

**Verdict on established / open.** Established: claims 1, 2, and the *form* of 3 and 4 (two labels, learned orders
only, EVC judge, widening by threat order). Open: predicting chess solver cost statically; calibrating the estimate;
the exact/estimate switch with interpretable traces as one unit (the pieces exist, the unit does not).

## 3. Cohesion with our machinery

| element | today | verdict |
|---|---|---|
| replies as one chain (AND) | a route holds iff every reply holds (`agenda.py:1649`, all legal replies) | already there |
| immediately forced | `regress._forcing_why_py` 343-362: check, mate threat, capture >= stake, quiet threat >= stake via the null-move test (`pass_claim`) | this is lambda-1 |
| delayed forced | absent; a quiet move with no threat after it is killed by tempo ("a free reply"): RoR8t's Kg3 | the gap: lambda-2 |
| only the parries read | every legal reply is read | the width saving is not taken |
| exact for a handful of lines | the fuse (`FUSE=10` x budget, `DEPTH_FUSE=30`) and the clock; a cut is a GAP, never a kill (2323) | the GAP is the slot an estimate would fill; the fuse is the numbered limit the judge replaces |
| cost ahead of time | nothing counts defences | new |
| accuracy ahead of time | no estimate exists; every claim is proved by a word or a read line | new; the risky part |
| judge re-asked per node | plan 56 task graph + plan 60 policy, both order only | the judge needs a new right: to stop a line and fill it with an estimate |

The idea is three separable pieces of unequal maturity: (1) lambda-2 forcing, a vocabulary word, exact, Lean-able; (2)
parry-only replies, a correctness fact once the threat is named; (3) the judge, a metareasoning layer that needs an
estimator, its calibration and a sound label for estimated claims.

## 4. Correctness

4.1 **AND-node estimates are unsound.** A claim today is a lower bound (every reply read, each closed by a proved word).
Closing a reply by estimate and reporting the route as holding makes the proof wrong. Required: `Claim.proved` (or
`estimated_by`, `confidence`); AND = min, OR = max over the proved component; `_better` prefers proved at equal value;
the decision line prints "estimated" wherever it leaned on one. Under the interpretability rule this is mandatory.

4.2 **The dual trap.** Their quiet replies can be delayed-forcing against us. Estimating their side optimistically is
unsound in the other direction (Allis's counter-threat hole; shallow traps). Any "not losing" that rests on an
estimated reply is itself an estimate. Dual lambda for their agenda is the cure and doubles the cost; the judge prices both sides.

4.3 **Zugzwang** has no threat to parry and is invisible to threat-order search (Thomsen Thm 2). Accept the gap, or detect
"every reply loses by the exchange" separately: RoR8t after Kg3 is partly this (White's rook has only losing squares).

4.4 **"Known ahead of time" is two falsifiable predictions.** Cost: a proof number whose first level is exact (count the
parries of the named threat). Accuracy: P(the estimate holds when read exactly), measurable only by sometimes reading
what was estimated: without that loop "accurate" is an assertion.

4.5 **Quiescence is the stateable accuracy rule.** The status rule (plan 59) already computes their pass threat; a node
is quiet iff both forcing sets are empty or closed by rule. First rule: estimate at quiet nodes only.

4.6 **The myopic judge under-reads.** A one-step "worth one more?" rule stops forcing chains early (Hay et al. 2012). The
judge must value the whole chain (cost of the remaining parries), not one step.

4.7 **The judge affects the decision.** Resolution consistent with the 2026-10-20 rule: the judge's inputs are exact
(parry count, quietness, open-threat flags) and its rule a readable threshold; a learned part may only order which
nodes are estimated first, never whether estimating is permitted. This must be decided explicitly before any build.

4.8 **No hard budget** is right as anytime behaviour; the clock stays as the outermost safety for play, and the depth
guard stays until the chase is iterative (plan 56 stages 3-4).

## 5. Improvements

5.1 **Name the threat, read only the parries (plus counter-threats).** `forcing_why` returns the threat object (target,
claim), not a string; the reply set of a forcing route = replies after which the threat no longer binds, plus replies that
are themselves forcing (Allis); all others are closed by the threat's claim. Lean word `parries(threat)`. Pure
vocabulary, the biggest exact-tree shrink; measure replies read per route on the question log before/after.

5.2 **Lambda-2 as "threat of a threat".** A quiet move is delayed-forcing iff after it, for every reply, some lambda-1
move of ours binds a claim >= stake. One bounded AND level (replies x forcing moves). It is Kg3. Build as a plan 59
tier under the time budget and grade on 85 random puzzles as plan 59 was. Zugzwang detector as a separate word later.

5.3 **Widen by threat order and tree shape instead of the fuse** (Cazenave's iterative/generalized widening): lambda-1
fully, then lambda-2, narrow shapes first. Plan 59's tiers are already this ladder in miniature.

5.4 **The first estimator is our own status, at quiet nodes only.** Material balance minus their pass threat, already
computed and interpretable; its accuracy is measurable from the log today. A learned value head only if the quiet-status
estimate is shown miscalibrated on a named class.

5.5 **Two numbers per node, PNS-style**: (proved value or none, estimated value + confidence). The decision reads proved
first; falls to the estimate only when the status rule says nothing proved keeps status (so most decisions stay fully
proved); the trace prints which. Proven losses stay in the bookkeeping, out of the estimate (MCTS-Solver §4.2).

5.6 **Cost predictor = proof number with an exact first level**: cost(node) = 1 + sum over parries of cost(child),
cost(unread child) = its parry count (one legal-move generation per parry). "Handful or blow-up" from the board alone; a
learned correction (PCN-style) later, and it only orders.

5.7 **Calibration loop**: every estimated leaf the clock later allows is read exactly on the background queue (plan 56
stage 3); disagreements logged with the estimator's name; the judge's threshold moves only from this log. Dual lambda
on their side logged the same way.

5.8 **Run the judge over the chain, not the step**: its cost input is the remaining proof number of the line, so long
forcing chains with a small parry set at every node read to their end (the fix for 2zB89's seven-ply line).

5.9 **STOP rule for the clock side** (Baier & Winands): stop when no remaining line can change the decided move; this is
the status rule's "kept" test applied to the open set, and it is exact.

## 6. Order when it becomes a go

1. 5.1 + 5.2 (vocabulary, exact, Lean words `parries`, `lambda2`): fixes RoR8t-type misses, shrinks width, no learning.
2. 5.5 + 5.4 (the proved/estimated label and the quiet-status estimate): soundness bookkeeping before any estimator.
3. 5.6 + 5.8 (the exact proof-number cost) replacing the fuse: fixes 2zB89-type cuts.
4. 5.7 (calibration loop) and only then a learned accuracy/cost head as a head of plan 60's policy, ordering only.
