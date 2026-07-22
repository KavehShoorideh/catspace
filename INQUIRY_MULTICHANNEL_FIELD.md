# INQUIRY: The multichannel quasimetric (opened 2026-07-23)

Kaveh's proposal: index human-play positions; from each anchor generate branches under multiple
PLAY REGIMES (random walk, optimal attack vs optimal defense, optimal vs graded-strength
defenders, human play itself); tag every state/edge with the regime ("color/flavor") that
produced it; learn a quasimetric whose steps carry both DISTANCE and PER-REGIME PROBABILITY —
"a distance of one that also carries a probability."

## 1. Formal answer: yes, three routes, we own the main one

- **Conditioning (the route we own):** the F-tower is omega-conditioned by design ("condition
  the cone on who generates the dynamics"); the lichess field already conditions on Elo bins.
  Regime tags = NEW OMEGA VALUES: d(F(x; regime), B(y)); B stays board-only. One model,
  channels as tokens. No new architecture — new vocabulary + tagged data.
- **The distance(+)probability algebra:** probability multiplies along paths -> -log p ADDS,
  like plies. Edge weight w_c = lambda*1 + (1-lambda)*(-log p_c); its shortest-path closure is
  a true quasimetric; exp(-d_c) is a discounted reach PROBABILITY (the C-learning /
  discounted-occupancy equivalence: distance and -log probability are one currency). Training
  on regime-c branches bakes likelihood in implicitly (visitation-weighted d_c) even without
  the explicit term — "the probability is automatically wired in through the data."
- **Channel combination semantics:** channels differing by MY style may switch mid-path ->
  the union-graph quasimetric (per-edge min) is valid. Channels differing by OPPONENT may not
  -> keep a VECTOR-valued d per channel, combined per-query (successor-features / GPI
  reasoning: a library of per-policy reachabilities, max/min at decision time).

## 2. The payoff: derived signals become CHANNEL QUERIES on one object

  cooperative reachability        d_union / d_random-ish
  human familiarity               d_human(elo)
  FORCEABILITY / the veto         d_optimal-defense - d_cooperative
  the tactic alarm (veto lapse)   that gap COLLAPSING after the opponent's move
  blunder affordance by strength  d_vs-1500 - d_vs-optimal
  indexed-planner "in my future"  small d under the channel I actually play

The tactics machinery (INQUIRY_TACTICS.md) runs per-channel: a basin's watershed under optimal
defense = the truly-forceable entries; under human-1500 defense = much wider; the DIFFERENCE is
where the spidey-sense hunts.

## 3. Data generation: sensible — it GENERALIZES the week's bespoke datasets

Existing one-offs are channel selections of the unified generator (anchors x regimes -> tagged
branches): contrast tuples = (sf-directed vs filtered-random); veto measurement = (random walks
vs optimal-defense forceability); cooperative-vs-human field pair = two single-channel models.
Machinery exists: the index, uci.py Elo/skill-graded engines, tb play, the branch generator
(~0.7s/branch measured). Caveats: (a) SUPPORT — branches cover near-anchor space only; channel
differences trustworthy only on shared support (measure); (b) COST — linear in |regimes|;
background generation, density-prioritized anchors; (c) AUDIT — engine PLAY as data admitted
(as with continuations), engine EVAL labels still excluded.

## 4. v1 proposal (awaiting Kaveh's go)

Channels: human(elo-binned — the shard stream, already have) · random · sf-optimal/tb-defense ·
sf-vs-weak. Implementation: regime tokens appended to the omega vocabulary of the d=512 field;
unified tagged branch generator extending gen_contrast_mate_tuples. First instruments:
(i) per-channel d_step/d_rand health; (ii) does d_optimal-defense - d_random correlate with
tb-exact deniedness (closes the loop with the 87%/99% veto measurement); (iii) shell resolution
per channel (INQUIRY_TACTICS sec 7).

## 5. The unification math (Kaveh: "two quantities, two ways — how do they unify?")

Distance composes by (min,+) over paths: d = min_P |P|. Rollout counts estimate the discounted
visitation, composing by (+,x): rho_c = sum_P gamma^|P| Pr_c(P). Define path energy
E_c(P) = lambda|P| + sum_e -log p_c(e), lambda = log(1/gamma). Then:

    rho_c(x,y)      = sum_P exp(-E_c(P))            (partition function)
    min_P E_c(P)    = HARD min (the quasimetric)
    -log rho_c      = SOFT min (log-sum-exp)  =  min_P E_c(P) - log N_eff,   N_eff >= 1

**Same functional at two temperatures — no averaging.** The gap Delta = log N_eff (effective
number of near-optimal paths) is a THIRD signal:
  Delta ~ 0        -> FORCED connection (only-move; the tactics narrowness statistic, principled)
  Delta large      -> ROBUST connection (many routes; the density prior formalized)
  d small, -log rho_c huge (strong-opponent channel) -> DENIED (requires their mistake) --
                      the veto as a metric divergence; its collapse = the tactic alarm.

Training: d stays the hard ply-metric (per regime via omega); rho_c = per-regime occupancy /
arrival-time head from tagged rollout counts (the existing distributional bin-head is nearly
this); consistency as an INEQUALITY (-log rho_c <= scale*d, sum >= max term), learned slack =
multiplicity. Scale note: current d is ply-count not energy; near-uniform regimes make them
affinely related (they "unify on their own" there); skewed strong-opponent regimes make them
diverge -- and the divergence is the signal.

## 6. The flavored-energy opponent model (Kaveh 2026-07-23: Boltzmann barriers; "parametrize
the energy, condition on Elo -- how do we frame the training?")

**Model (multidimensional IRT / Boltzmann rationality; Regan's IPR = the K=1 eval-based special
case; Maia = the nonparametric ceiling; ours is learned, multiD, EVAL-FREE):**

    pi_omega(m|x) = softmax_m( -<beta(omega), E(m,x)> )
    E: (x,m) -> R^K_{>=0}   K flavor barriers (a policy-net head: K move-map channels)
    beta(omega) in R^K_{>=0} per cohort: Elo bins, Stockfish, Leela

Flavors are IDENTIFIED by cross-cohort disagreement (who sees what), interpreted post-hoc via
probes (puzzle themes = diagnostic only). Physics: Arrhenius/Boltzmann with Elo as inverse
temperature; "difficulty of seeing the move" = barrier height.

**Training:** masked softmax CE on move-selection triples (position, move played, mover Elo) --
~12M rows in the 4gb prefix alone; engine channels (SF/lc0 played on sampled positions) add
extreme beta profiles; NO eval labels anywhere (audit-clean by construction).
**Identifiability:** nonneg E and beta (NMF-style), per-flavor scale normalization; monotone
beta(elo) optional (or check monotonicity as validation). **Ladder:** K=1 baseline (the naive
scalar) -> K=2,3 by held-out likelihood; Maia-style per-bin policy = ceiling; EXTERNAL
validation = lichess puzzle ratings (empirical human-attempt difficulty): predicted
P(solve|Elo)=0.5 point must map monotonically onto puzzle rating, zero training contact.

**Plugs in:** -log pi_omega = the per-edge -log p_c of sec 5 (one consistent stack: energy ->
edge probs -> rho_c / soft distance -> channel queries); watchlist gets "will THEY see the
defense?" = their barrier-crossing probability; tilt = ONLINE re-estimation of beta from the
opponent's recent moves (opponent model as updatable state). First build on go: K=1 smoke on
one shard prefix, acceptance = held-out LL vs per-bin baseline + puzzle-rating calibration.

## 7. Attention generalization of the energy (Kaveh 2026-07-23: "get rid of the exponential")

Two exponentials, different roles: the categorical softmax over legal moves (bookkeeping, stays)
vs the Boltzmann LINEARITY law log-prob = -<beta,E> (the physics assumption -- KILLED). The
attention form is the strict generalization; bilinear = its rank-K zero-interaction special case.

**Model:** cross-attention from the move to the player's skill profile.
  Query   = demand vector of the move (board encoder, per legal move; heads = demand aspects,
            the flavors reborn)
  Key/Val = the player's skill TOKENS per cohort omega (Elo bins, SF, lc0; tilt = online update
            of the tokens -- a richer opponent state than one beta)
  2-3 layers -> skill INTERACTIONS ("needs tactics x calculation JOINTLY")
  logit(m) = f(demand(m,x), Skills(omega)); same masked CE, same data. Temperature becomes
  implicitly position- and player-dependent.

**Path algebra untouched (sec 5 survives):** it only needs pi_omega(m|x); E(edge) := -log
pi_omega is now a learned interaction-aware energy; rho_c, hardmin/softmin, Delta, veto
divergence, watchlist all consume it unchanged. Attention = richer energy PARAMETRIZATION,
same thermodynamics -- and this IS the multichannel field's edge model.

**Tradeoffs:** interpretability relocates to the explicit low-dim demand BOTTLENECK (+ probes;
puzzle-rating calibration transfers verbatim). Ladder keeps bilinear rungs as baselines:
K=1 -> K=3 bilinear -> 1-layer attn -> 2-3 layers, held-out LL per Elo bin decides; the LL gap
between bilinear and attention rungs = the measured value of skill interactions.

## 8. DECIDED (Kaveh 2026-07-23): candidate-set self-attention (option A); iterate later

The opponent model's final architecture for v1:
  1. legal moves = TOKENS (+ a position token); SELF-ATTENTION among them -> set-contextual
     scores ("likelihood of seeing m depends on what else is going on": distractor suppression,
     threat-load, Einstellung-style competition become learnable move<->move interactions)
  2. CROSS-ATTENTION from move tokens to the cohort's skill tokens (Elo bins / SF / lc0;
     tilt = online token update)
  3. masked softmax over legal moves; CE on (position, move played, mover Elo) triples.
Deferred (iterate later): the seeing x choosing two-stage factorization (sec 7 discussion --
identifiable via engine channels when we want it); history/plan-state conditioning (C).
Ladder unchanged: bilinear K=1/K=3 rungs stay as baselines; external judge = puzzle-rating
monotonicity.

## 9. Unification: multiplicity IS attention-weighted (Kaveh 2026-07-23: "multiplicity might
be high but if I don't see the options then it's effectively lower")

The path algebra (sec 5) never had its own edge probabilities -- the attention opponent model
(sec 8) SUPPLIES them: p(edge) = pi_omega(m|s). Substituting makes every quantity
cohort-indexed: rho_omega = sum_P prod pi_omega, Delta_omega = log N_eff^(omega). Unseen paths
carry ~zero mass under my pi -> they don't count toward MY multiplicity: five objective routes,
four invisible => N_eff^(me) ~ 1 (forced FOR ME, robust on the board). "Objective" multiplicity
= Delta under an all-seeing channel (engine / uniform). RULE: never quote Delta without a
cohort subscript. Physics closure: Arrhenius rate = A*exp(-E/kT); seeing = the pre-exponential
ATTEMPT FREQUENCY A; log-additive with the barrier -- exactly the deferred seeing x choosing
factorization (attention v1 learns them jointly; engine channels separate them later).

**Derived signals:** (1) TRAP POTENTIAL = Delta_objective(their defenses) - Delta_them(their
defenses): the saving defense exists and they won't see it -- swindles/practical chances as a
map quantity, conditioned on their Elo/tilt. (2) METACOGNITIVE SEARCH TRIGGER: Delta_objective
(my options) >> Delta_me -> options exist that I'm not seeing; the map knows -- search more.
(3) COHORT-RELATIVE SHARPNESS: sharp = Delta_omega ~ 0 for the cohort at hand (feeds alpha-dial
and time management).

**Falsifiable prediction (set-contextuality):** adding a loud losing option can LOWER effective
multiplicity (Einstellung suppression) -- impossible for independent-barrier models (they only
renormalize). Test: matched-difficulty lichess puzzles with vs without flashy distractors;
solve-rate gap beyond renormalization = attention model confirmed. **Data note:** the human
channel's visitation-rho inherits the perception filter FOR FREE (humans only traverse seen
paths); the attention model adds counterfactual generalization (any cohort, any position, the
opponent's current tilt state).
