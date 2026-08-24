# gko1 — the unified Koopman-concept oracle-distillation system
### Plan, invariants, unit tests, and review gates (Kaveh's protocol, 2026-08-21)

Approved directives this plan implements: joint training of trunk + concept VQ + move
tokens + per-move-code linear rollforward operators + concepts->alpha/beta intervals +
oracle move-distribution outputs; full SF/Maia labeling authorized FOR THIS LINE;
oracle-QUERY paradigm (no bulk corpus; pool + acquisition + budgets); gciv folded in
(frozen-field run = ablation-before row + bootstrap acquisition).

## 1. Falsifiable hypotheses
H1 (operators): bilinear per-move-code linear operators over the concept state predict
    child concept states with one-ply fidelity within 10% of the MLP dyn, and BEAT it on
    COMPOSED rollouts at 3 AND 6 plies (Kaveh 2026-08-21: "2-3 plies is too few" -- 6-ply
    composition matches the target search regime; MLP chains typically collapse there,
    operator products should merely widen).
H2 (intervals): jointly-trained concept intervals reach coverage within [nominal,
    nominal+0.07] with width narrower than the frozen-field probe by >=25%.
H3 (query efficiency): active acquisition (interval-width x code-rarity x entropy)
    reaches the same ChessBench-test action-value fidelity as random querying with >=5x
    fewer oracle calls.

## 2. Components (tensor contracts)
- Trunk + concept heads: GraphConceptField (gcf4h arch; warm-start gcf4h_latest),
  z_state = zq in R^{d_state}; move tokens: mv_vq codes c(m) in {0..63}.
- KoopmanOps module (v1, INTERACTION-AWARE VIA THE LIFTING -- Kaveh 2026-08-21:
  "if both concepts a and b are present, the evolution of both will be different"):
    z_aug = [z ; phi(z)]  with phi = learned low-rank pairwise interaction features
    (128 dims: bilinear layer over head embeddings, jointly trained). Operators are
    STRICTLY LINEAR on z_aug: z_aug' = clamp_spectral(A + B_{c(m)}) z_aug -- the
    Koopman-orthodox move: lift past nonlinearity, keep the algebra (composition,
    spectra, persistence rulebook all intact on the augmented space).
    A shared; B_c low-rank (U_c V_c^T, r=16) per move-code; spectral clamp via
    power iteration, penalty max(0, rho - 1.02)^2. Null move: B == 0, A init = I.
  v2 FLAG-GATED (switching-linear, Kaveh's literal mechanism): + Sum_h E_h[code_h]
    low-rank code-conditioned terms (rSLDS-style). TRIGGER, pre-registered: one-ply
    residuals stratified by co-active code PAIRS still dominated by specific pairs
    despite phi -- then v2 turns on for those heads; costs trajectory-varying
    operators (codes re-read from predictions at each ply; compounding risk noted).
- Interval head: MLP([zq ; e_code(m)]) -> (dv, softplus lo-off, softplus hi-off)
  (the gciv architecture, now unfrozen and joint).
- Oracle heads: SF action-value distribution per move — HL-Gauss binned classification,
  K=64 bins over win-prob [0,1] (ChessBench + Stop-Regressing recipe); Maia policy head
  — KL to Maia's softmax policy at the queried elo band (maia-1100..1900 as bands).
- Persistence readout (no params): concept h survives m iff argmax-code of P_h z' ==
  argmax-code of P_h z; tabulated per (head, code, move-code) offline.

## 3. Data engine (oracle-query)
- Pool: 2M unlabeled positions: 60% human-corpus states, 30% SF-corpus, 10% random
  legal walks (diversity floor). Pool is free; LABELS are metered.
- Oracle service (oracle_service.py): query(fen) ->
    sf: multipv=all, depth 10 + 50ms cap per position (ChessBench parity), all legal
        moves' cp/wdl -> win-prob targets
    maia(band): lc0 --weights=maia-<band> nodes=1 verbose-move-stats -> raw policy P(m)
  sqlite cache, journal_mode=DELETE (the tb-cache scar), keyed (fen, oracle, band).
- Active loop, PIPELINED (Kaveh 2026-08-21: "pipeline data generation with training
  so you can work in batches"): producer/consumer over the sqlite store --
    PRODUCER (CPU, concurrent with training): oracle worker process drains an
      acquisition-ranked queue; SF capped at Threads=2/Hash=64MB, lc0 nodes=1, so the
      co-tenancy stays inside the one-heavy-job rule's RAM budget; labels commit
      atomically to the store (queue row claimed -> labeled exactly once).
    CONSUMER (GPU): trains continuously; each batch mixes newly-landed labels with
      replay; acquisition scores refresh from a model SNAPSHOT every 500 steps (stale
      scores only need to RANK -- exactness not required; staleness noted, accepted).
  Budgets unchanged: ~5k queries per refresh window, ~8 windows v1 (40k SF + 10k Maia).
  Baseline arm: random acquisition, same budgets (H3's control; part of the hypothesis
  test, not an A/B rollout).

## 4. Losses (weights at launch; interaction review inline)
  L_sf (HL-Gauss CE, all legal moves)          2.0   the distillation target
  L_maia (KL, queried bands)                   1.0
  L_op1 (one-ply: ||K_m z - sg(z_child)||)     1.0   operator grounding (real children)
  L_op3 (3-ply composed vs real, 25% of batch) 0.5   composability trained, not hoped
  L_op6 (6-ply composed vs real, 32-row sub-
         batch, weight ramped in after 2k steps) 0.3   the full-horizon claim; targets
                                                     need the intervening move sequence:
                                                     once-per-corpus game_transitions
                                                     cache (gen_game_transitions.build)
                                                     + live slim encodes of s_{t+3},
                                                     s_{t+6} for the sub-batch
  L_persist (CE code-survival vs truth)        0.5   the rulebook's accuracy
  L_iv (pinball 0.1/0.9 on oracle value gaps)  1.0   NOTE: intervals now supervise on
                                                     SF action-value gaps, not CF pairs
  L_spec (spectral penalty)                    0.1
  + existing field stack (move PL, dyn, VQ commit, decor, phase, vamp) at gcf4h weights
  Interactions reviewed: L_op1 vs dyn (complementary targets, same child codes — dyn
  keeps distributions, ops keep states); L_iv vs L_sf (iv reads value gaps FROM the sf
  head's targets — shared bins, no circularity because iv trains on oracle values not
  the head's predictions); decor vs ops (operators act on the concatenated state —
  decorrelated heads make B_c nearer block-diagonal, GOOD); spectral penalty vs vamp
  (both favor non-explosive slow dynamics, aligned).

## 5. Invariants + unit tests (all must pass BEFORE any long run)
  U1  operator identity: A:=I, B:=0 => z_aug' == z_aug exactly (null-move fixed point)
  U1b phi consistency: the z-part of a composed rollout stays consistent with re-lifting
      (||phi(z'_pred) - phi-part of z_aug'|| tracked; drift bounded on synthetic linear
      systems with known quadratic observables)
  U2  spectral clamp: rho estimate within 5% of torch.linalg.eigvals on random 64x64
  U3  composability algebra: op(op(z,m1),m2) == matmul-composed (A+B2)(A+B1) z (fp32 tol)
  U4  ridge-fit cross-check: closed-form per-code operator fit on synthetic linear data
      recovers the generator; MATCHES kooplearn EDMD estimate on the same data (adopt
      kooplearn as the reference implementation, not a dependency of the model)
  U5  HL-Gauss: bin-encode->decode round-trip bias < 1e-3 across [0,1]; CE minimized at
      the true value on synthetic
  U6  pinball quantile unbiasedness (tonight's verified test, promoted to the suite)
  U7  interval ordering lo<=dv<=hi structural (softplus)
  U8  persistence readout: hand-built operator that permutes two codes yields the
      predicted survival table exactly
  U9  oracle service: FEN round-trip, cache hit determinism, multipv covers ALL legal
      moves (count match python-chess), Maia policy sums to ~1
  U10 acquisition: on a synthetic pool with planted rare codes + wide intervals, the
      scorer ranks planted states first; dedup-by-signature caps duplicates
  U10b queue atomicity: concurrent claim of the same row labels it EXACTLY once; the
      trainer never reads a row before its label commit (WAL-free store, DELETE mode)
  U11 pool/label split hygiene: ChessBench test FENs excluded from the pool (hash check)
  U12 elo-conditioning passthrough: fixed-elo vs per-band Maia conditioning shapes agree
  U13 gradient isolation: oracle targets are stop-grad; no loss reads model predictions
      as its own target (circularity scan)
  U14 loss-zero checks: every loss == 0 on its own perfect target
  U15 the flat-Q lesson: any score-combination path reduces to its base when the added
      signal is constant (searched==prior class of invariants, applied to acquisition
      and to interval-adjusted orderings)
  U16 metrics sanity: every logged metric MOVES when its input is perturbed (no
      constant/disconnected gauges); a planted NaN triggers the pause-alarm path end
      to end in a synthetic dry run

## 6. Evaluation gates + ablations
  - ChessBench test set: action-value Spearman + top1-vs-SF; puzzles solve-rate.
  - Their data-scale curve with our query-count points (H3 headline).
  - Operator table: 1/3/6-ply R2 vs MLP dyn (H1); per-horizon error growth curve;
    INTERACTION RESIDUAL TABLE: one-ply residual by co-active code pair (the v2 trigger)
    (the composability signature: sub-exponential for ops, exponential for MLP chains);
    persistence-table accuracy.
  - Intervals: coverage/width vs the frozen gciv1 "before" row (H2); pruning power
    (survivor counts by phase) via eval_interval_pruning.
  - Play: prune-search arena (intervals gate, committor ranks) vs fixed-beam
    ConceptAlphaBeta — strength-per-node frontier.
  - Concept health: alignment eval + decor + effective-rank gate (standing rule).

## 7. Observability (Kaveh 2026-08-21: "proper logs and metrics so you can check on
## the health")
- TRAINER (every eval, MLflow via scaffold + live_curves jsonl/png/html dashboard --
  the "watchable, not just logged" standard): every loss term; per-horizon operator R2
  (1/3/6); interval coverage + width on the dev slice; distill fidelity (action-value
  Spearman on a FIXED 5k dev set); code-usage entropy per head + EFFECTIVE RANK
  (bootstrapped -- the standing collapse gate); decor; grad-norm; steps/sec.
- PRODUCER (per-minute heartbeat line + per-window summary in producer.log):
  queries/sec, oracle latency p50/p95, queue depth, label-store rows, cache hit rate,
  engine restarts, SF-vs-maia band mix, and the DISTRIBUTION of acquisition scores
  among claimed rows (drift check: is the scorer still discriminating).
- PIPELINE: snapshot staleness (steps since refresh), fresh-label fraction per batch,
  labeled/consumed ratio (starvation and overrun both alarm).
- HEALTH GATES (auto-alarm via watchdog; PAUSE both processes, never silent-kill):
  coverage outside [nominal-0.02, nominal+0.10]; effective rank down >20% from start;
  op R2@1 falling 3 consecutive evals; queue starved >15 min; any NaN anywhere.
- Watchdogs: PID-based on producer AND consumer, swap guard, queue-stall detection,
  30-min heartbeats into the session (the day's watchdog lineage, both scars encoded).
- Files: artifacts/experiments/gko1/{train.log, producer.log, metrics.jsonl,
  dashboard.html}; verdicts land in JOURNAL with printed numbers only (standing rule).

## 8. Risks
  - Hard code-flips may defeat linear ops (captures): measured by L_op1 stratified by
    move type; if capture rows dominate the residual, ops get a capture-conditioned
    rank boost r=32 for capture codes (flag-gated, pre-registered).
  - Oracle latency: 50ms x 40k = ~35 min/SF round — acceptable; Maia nodes=1 is fast.
  - Joint-training instability with 10+ losses: warm-start everything possible
    (gcf4h trunk, gciv-style head init), operator A at identity, ramp L_op3 after 1k.
  - Pool staleness: acquisition re-scores every round on the CURRENT model.

## 9. Build order (each step lands with its unit tests green)
  1. vendored bagz reader + ChessBench test loader (U11); game_transitions cache for
     both corpora (the 6-ply composition targets)
  2. hl_gauss.py (U5) ; ops module (U1-U4) ; interval head reuse (U6-U7)
  3. oracle_service.py (U9) + maia weights fetch
  4. acquisition.py (U10, U15)
  5. gko trainer assembling all (U12-U14) ; observability wiring (U16) ; invariant
     review script that GATES
  6. smoke (crash/scale only) -> round-0 bootstrap (gciv scorer) -> launch producer
     FIRST (warm the label store while units finalize), consumer joins when gates pass

## H1 REFINEMENT (Kaveh 2026-08-21, verbal): search economics + locality are the prize
Composition fidelity is necessary, not sufficient. The real payoffs, in order:
(a) SEARCH SPEED: inside a tree, child evaluation = one linear apply, not a re-encode.
    Measured 2026-08-21 (CPU, batch 64, gcf4h-scale): re-encode 2.766 ms/node vs
    operator apply 59 us/node = 47x more nodes per unit compute. This is the number
    that bootstraps the weak policy via search; report it beside every H1 fidelity row.
(b) LOCALITY ("what move changes what concept," separated IN ADVANCE): a concept
    localized to one region must be touched only by moves connected to it (the
    corner-pawn-vs-mate example). Anti-MuZero: no monolithic latent rollforward.
    Verdict-time instrument: the TOUCH TABLE -- per (move-code c, concept-head h),
    mass of B_c on head h's block; plus square-geometry alignment for the per-square
    part of the state. Interpretability artifact AND a pruning map: zero blocks =
    certified non-interaction = skipped compute.
(c) PRE-REGISTERED LEVER (next run, flag-gated, NOT mid-run): if learned B_c is not
    naturally block-sparse, add group-sparsity (L2 over per-head blocks of U_c V_c)
    -- "don't pay for non-interaction" as a training pressure.
Concepts are mandatory throughout (interpretability is the endpoint); linearity is
what makes them cheap to search through.

## LOCALITY REFINEMENT 2 (Kaveh 2026-08-21, verbal): piece-support ladder + L1
Plain-language rename: the "touch table" is the PIECE-SUPPORT picture -- per concept,
which pieces it reads; per move, which concepts it can change. Kaveh's prior, testable:
a DECREASING LADDER of support sizes (many 1-piece concepts, fewer 2-piece, fewer
3-piece, one or two 4-5-piece structural ones). The pin reads its three pieces only;
pin-fails-to-knight-check is an INTERACTION between two small concepts (carried by
phi / the code-pair table), never a widening of either concept's support.
Levers (flag-gated, NEXT run, never mid-run):
  (a) field-side: group-L1 over per-square attribution mass per concept head
      (concepts become piece-sparse against the mean-pool architecture);
  (b) operator-side: group-L1 over per-head blocks of B_c (moves change few concepts).
Verdict artifacts from gko1c (free, measurement-only): per-concept support-size
histogram judged against the ladder shape; per-move-code changed-concepts histogram;
pin-like small-support exemplars as case studies. Honest architecture note: global
heads mean-pool ALL 64 squares, so support sparsity must be learned against the
pooling; if the penalty fights the architecture, the structural fix is region-gated
heads (next-design decision, Kaveh's call).

## LOCALITY REFINEMENT 3 (Kaveh 2026-08-21, verbal): the 2x2 concept plane
Two locality axes: SPACE (piece-support size) x TIME (persistence; "fast = local in
time, slow = global in time"). Naming discipline: WIDE/NARROW for the spatial axis
(the codebase already uses "global heads" for trunk-level VQ heads -- do not overload).
- WIDE heads: FEW of them; NO piece-sparsity penalty (game phase = canonical
  wide-slow; they may legitimately read many pieces).
- NARROW heads: MANY; the piece-support group-L1 applies; ladder-shaped supports.
- Temporal axis already exists (slow_heads under VAMP lag pressure vs fast + vicreg).
- Quadrant examples: wide-slow phase; wide-fast initiative/tempo; narrow-slow pin,
  structure motifs; narrow-fast checks, hanging pieces. Per-square codes = maximally
  narrow end of the ladder.
Next-run design: head budgets per quadrant (few wide, many narrow) + selective L1
(narrow only). Verdict artifact from gko1c (measurement-only): THE CONCEPT PLANE --
scatter every learned concept on (support size x persistence); Kaveh's prior = mass
in narrow-fast thinning toward a few important wide-slow heads.

## INTERACTION-LIFT v2 (Kaveh 2026-08-21, verbal): transformer-over-concepts BEFORE ops
Ruling: concept interactions (pin x check) are computed by 1-3 transformer layers OVER
THE CONCEPT TOKENS, written back onto the tokens ("a little bit of a rotation"); the
Koopman operator applies AFTER, to the post-interaction state, and rollouts carry the
interaction signal with NO re-encoding and NO re-application of the transformer --
"otherwise I am no better than MuZero." Current gko1c already enforces the ordering
(lift -> ops -> never re-lift; consistency metric = the meter); the bilinear phi is
the weakest instance and becomes the ablation "before" row. v2 = InteractionLift as a
small concept-token transformer feeding the augmentation block; U1/U1b/U3 gates apply
unchanged. Bet named: interactions, once computed, evolve linearly too (Koopman-
orthodox observable enrichment); if code-pair residuals still leak, that is the rSLDS
v2 trigger, unchanged.

## PRODUCT SHAPE (Kaveh 2026-08-21): multiple heads on one concept state
Final model = one checkpoint, two play personalities: (1) BEST-PLAY head -- the SF
action-value head (multipv-distilled win-probs; argmax = best move, magnitudes = the
search eval); (2) HUMAN head(s) -- the Maia-KL move head, elo-CONDITIONED (one head,
continuous dial, bands 1100-1900). Verdict obligation added: PER-BAND top-1-vs-Maia
at each band's own dial setting (pooled maia-top1 hides blur); if bands blur into one
average human, escalate to discrete per-band heads (cheap swap). Play wiring: "best"
mode = search on sf_av; "human@elo" = dialed move head, tempered.

## VOCABULARY SCALE (Kaveh 2026-08-21, verbal): a couple hundred concepts
Ruling: the concept vocabulary should be O(200) nameable units, not 32. Proposed
default (his 2x2 as budget sheet): 8 wide-slow + 8 wide-fast + ~96 narrow-slow +
~96 narrow-fast heads; heads get SHALLOWER as they multiply (narrow = 4-8 codes,
present/absent + mode; wide keep richer books). Cost check: d_state ~3.8k, operator
~17M params, apply ~120us/node = still 20x+ under re-encode; lift tokens 97->~270,
attention still trivial. This is the next FIELD generation (gcf5/gko2), calibrated by
gko1e's concept-plane + duplicate-merge analysis (how many distinct concepts the
current 32x32 already expresses). Composes with: piece-support L1 (narrow only),
per-head diffusion scales (emergent spatial granularity), per-band maia heads.

## VOCABULARY SCALE AMENDMENT (Kaveh, same day): 512 concepts
512 heads, not ~200. Proposed quadrant split 16 wide-slow + 16 wide-fast + 240
narrow-slow + 240 narrow-fast; mixed depth (wide d_code 16, narrow d_code 4-8 --
narrow concepts are present/absent + mode). Cost at narrow-d8: d_state ~6.3k, d_aug
~6.5k, shared operator ~43M params (Adam state ~0.5GB -- the one real jump), apply
~300us/node = still >=10x under re-encode, MORE once piece-support L1 makes the
operator block-sparse (touched-blocks-only application). Lift tokens ~577, attention
still cheap. Verdict-time duplicate-merge analysis on gko1e calibrates how much of
512 the data can actually individuate.

## MOVE CONCEPTS IN THE INTERACTION LAYERS (Kaveh 2026-08-21 late, verbal)
Ruling: moves have concepts (the mv_vq 64-code book IS them; gives_check is a raw
move-token feature, and operators are already indexed by move code -- his tempo
example: check-capture vs quiet-capture = different codes = different operators =
different rolled futures). MISSING piece his ruling adds: the 48 move-concept tokens
join the 3-layer interaction transformer (currently board-state tokens only), so
"the check is available" participates in pre-operator interactions and rolled states
carry tactical texture. Everything-talks-to-everything x3 layers, THEN operators.
Edge-histogram token stays pending redundancy analysis (edges = standing relations;
moves = available actions -- different kinds). VERIFICATION OWED at verdict: move-code
table (code x gives_check/capture/promo MI + exemplars) -- do the 64 codes split
check-ness; widen the move vocabulary if blurred. Timing: fold into gko2d on gko2c's
morning verdict (one change on top of tonight's three-mechanism bundle).

## COMPONENT-REVIEW NOTES (2026-08-21 night, Kaveh-requested)
EMA bugs found+fixed+gated (train()-recursion re-arm; uninitialized-codebook first
forward) -- U18 (bit-frozen under train), U19 (first-update exact equivalence), U19b
(lag inequality) now in the suite. eval_chessbench fixed to honor run-ckpt geometry
(64-head ckpts would have shape-mismatched the morning battery). FORWARD-COMPAT NOTE
for the operator-search engine: mh_vq checkpoints store codebooks as vq_mh[group]
(8 groups), not vq[i] -- the engine's concept-stats access must be group-aware.
Sidecar rewrite race is benign (torn npz read raises; rescan keeps prior coverage).

## GCF5 OPERATOR DESIGN (Kaveh 2026-08-22 evening, verbal): slowness as compute subsidy
Measured baseline (18k ops, printed): A near-copy everywhere (slow-block dev 0.0850 ~=
fast 0.0859 -- NO slow/fast asymmetry despite vamp 30.8/32); move-writes land equally
on slow/fast (1.14 vs 1.15 per-dim), most on phi (1.76). The encoder's timescales have
NOT propagated into operator structure. Pre-registered levers:
 (1) delta-form slow-write penalty: z' = z + Delta, ||Delta_slow|| charged -- silence
     free, movement justified; enables row-skip application (z'_slow = z_slow).
 (2) A_slow-block FROZEN at exact identity (ambient drift structurally zero; slow
     change only through B_c events) -- verify the 0.085 dev is normalization artifact
     vs real macro-drift before committing.
 (3) horizon-matched loss weights: slow dims graded at long horizons, fast at 1-ply.
 PREREQUISITE: kill per-row min-max zq normalization (couples all dims; slow constancy
 unrepresentable today) -> per-block or fixed-global affine.
 Compute: frozen-slow + sparse writes -> apply ~900/2944 dims (~3x); makes the
 token-preserving lift (24k dims) tractable at 10-15 percent active rows. The temporal
 twin of piece-support sparsity -- the two multiply.

## OPERATOR PRIOR AMENDMENT (Kaveh 2026-08-22, asking-not-telling -> adopted):
PRIMARY: global near-identity prior ON THE OPERATOR -- K = I + Delta with Delta
earning magnitude everywhere (explicit ||A - I|| + move-correction norm penalties,
ongoing, not just init). Timescale structure EMERGES in the operator; no encoder
slow/fast labels imported (same philosophy as the check-targeting rejection).
Adjudicating experiment: train under the global prior, rerun the block-structure
measurement -- slow rows shrinking harder ON THEIR OWN retires the label-informed
variant (demoted to fallback). Both variants gated on the normalization prerequisite.

## GCF5 COMPONENT: PIECE CONCEPTS (Kaveh 2026-08-22 late, verbal)
The missing persistence layer: squares = places, board = global state, moves =
actions; the PIECE -- the entity that has activity -- exists only implicitly. Design:
32 piece slots (identity = original square; promotions inherit; captures mark dead),
token = gather(current square hidden) + identity embedding, small per-piece codebook
(8-16 codes: active/passive/trapped/hanging/...), joins trunk attention + the
interaction table (~430 tokens) + a piece block in the operator state.
PRE-REGISTERED PREDICTION + compute subsidy: one move touches <=2 pieces, so under
the I+delta prior the operator's piece-block writes must be <=2 slots/move (skip 30
of 32 at apply). Piece concepts naturally slow -> VAMP split; piece-support ladder
becomes literal slot-sets. Completes the concept type system: piece x square x move
x board, all at one interaction table.

## GCF5 STATE BUDGET (Kaveh 2026-08-22: reallocate, do not grow; powers of two)
d_aug = 2048 EXACTLY (down from 2944): squares 64x16=1024 (the funding cut: 32->16,
squares were 76 percent of state) | board 64x8=512 (2x2 budgets herein) | pieces
32x4=128 | hist 128 -> concept state 1792 + phi 256. VARIANT (preferred, pending the
hist redundancy trial): hist retired, pieces 32x8=256. Wins: operator A halves to
4.2M params, apply ~75us (economics IMPROVE while adding a concept type), power-of-two
blocks make slow-silence + piece-sparsity subsidies clean strided slices.

## VALUE METRICS RULING (Kaveh 2026-08-22 night): distributions, not scalars
Flat vMAE launders decision-relevance (0.1 at +12 meaningless; 0.1 near equality is
the game). Standard form now: STRATIFIED by band (near-equal / advantage / decided)
+ CRPS over the predicted bin distribution. First table (gko2h-4k, k=0, printed):
near-equal 0.072 | ADVANTAGE 0.134 (the weak band = edge-conversion = what the
piece-down diet treats) | decided 0.066 -- pooled 0.095 hid all of it. Retrofit:
W/D/L triples synthesized from stored wp (invertible to cp -> standard WDL model),
no re-querying. GCF5: native W/D/L value heads (reconnects to the committor/basin
founding frame).

## ROLL-ONLY SEARCH PROGRAM (Kaveh 2026-08-23 dawn: "why child encodes at all?")
Target: encode ONCE at root, roll everywhere, re-encode = drift audit only. Current
child encodes pay for: (1) contextual move codes (operator selection), (2) the sharp
per-move head's inputs (the 4pct-vs-62pct gap: rolled ordering 0.27 vs sharp 0.5),
(3) drift resets. Closure path, pre-registered: (a) CROSS-ROLL DISTILLATION -- rolled
move-scorer trained to match the sharp head at encoded states (teacher-student across
the roll; data already flows); (b) MOVE-CODE PREDICTOR from rolled state + move
tokens (closes operator selection); (c) token-preserving state (gcf5 format lever).
(a)+(b) are trainable in the CURRENT lineage. Endgame: ~25ms/move, another 10x on
strength-per-node.

## SEARCH CADENCE CONTRACT (Kaveh 2026-08-23: "happy to re-encode like MuZero after
## every move, but search shouldn't need re-encoding")
Encode ONCE per real game move; zero encodes inside a search. Arithmetic: 1 encode
(23ms) + ~1800 operator applies at the same per-move budget = wide depth 4-6 in
concept space (vs 9 encodes + 180 applies today). Requirement: av_roll (the rolled
scorer -- ALREADY the student) closes ordering 0.27 -> ~0.5 via (i) move-code
embedding input, (ii) sharp-head soft-target distillation at encoded states,
(iii) capacity; structural unlock = gcf5 token-preserving state. MuZero-consistent:
heads trained through the dynamics make latent-only search sound.

## OPERATOR CONDITIONING REDESIGN (Kaveh 2026-08-23: position-free move lookup)
Ruling-question answered YES with one refinement: move representation = the CHEAP-FLAG
token (from,to,piece,captured,promo,CHECK -- all python-chess-derivable, no encoder;
= the existing 19-feature mv token), NOT bare uci (would resurrect tempo blindness).
Operator selection: VQ codes -> BASIS MIXING B(m) = sum_k w_k(mv_emb) B_k, K=64 basis
operators (same params; smooth, no quantization loss; mixture weights = soft move
concepts with discrete readout on demand; U1/U3 gates unchanged). MuZero-precedented:
context lives in the STATE; the move needs identity + cheap flags only. Deletes the
move-code-predictor need; unifies with cross-roll distillation (one move rep
everywhere). Completes the roll-only search stack: rolled nodes need NOTHING from the
encoder.

## CONDITIONING AMENDMENT (Kaveh 2026-08-23: bare move identity, no flags)
The check flag RETRACTED from the move token: check-ness is a property of the ROLLED
STATE, not the move -- the operator (same matrix per move) acting on the state's
king-geometry features writes the check concept exactly when circumstance warrants
(linear-representable via kin/king-line features already in the state). Move token =
bare identity lookup (from,to,piece,promo -- promo is move-intrinsic). Emergence over
injection, MuZero-orthodox, and faster at depth (no per-candidate flag computation,
~1s/move saved at d5-6). PRE-REGISTERED GATE: rolled-check AUC (rolled state's
king-danger readout vs python-chess gives_check truth); the flag returns only on
measured emergence failure.

## GATE CORRECTION (Kaveh, immediately following): NO named-concept gates, ever
The rolled-check AUC is DEMOTED from gate to optional post-hoc description. "The
check concept might be spread across three other concepts. I don't need to see it.
I just need to see it play well." Standing epistemology, third statement of the same
ruling this week: GATES = performance (play strength, outcome prediction, decor
health). PROBES = post-hoc interpretation only, never pass/fail. Named-concept
recovery is a curiosity, not a criterion.

## FINAL GATE FORM (Kaveh 2026-08-23): correctness, then adequacy
"The gate is that it trains correctly, and it works... it doesn't have to play
perfectly well because we can always offset playing strength with search. We're
planning as where we wanna go next. Something that plays a few plies of search as
good as Maia-1100 is good enough. I just want it to be bug free."
gko3a acceptance: (1) BUG-FREE (the U-gate suite + review discipline, the week's real
deliverable), (2) trains/works (curves move, machinery sound), (3) ~Maia-1100 at
shallow search = sufficient substrate. Beyond that: the PLANNER, designed with Kaveh
present. Current engine already clears (3).

---

## Resolved 2026-08-23: where the play gap actually lives

Three generations measured on identical unseen data (5,903 SF positions the models
never trained on; 39-47 real positions for search internals):

    model       ord     lvlMAE  lvlcorr  valcorr | childsep(rel) | arena vs Maia-1100
    gko2i      0.535    0.139   0.864    0.866   |    2.95pct    | 60pct pooled (n=24)
    gko3a @8k  0.537    0.110   0.874    0.853   |    0.48pct    | 38pct (n=24)
    gko3b @8k  0.549    0.100   0.895    0.879   |    1.12pct    | 45pct pooled (n=48)

**The heads are not the problem.** gko3b @8k beats the 62pct-era incumbent on every
head metric. **The search path is.** The quantity that separates them is the
rolled-child value spread -- what alpha-beta consumes at every node it cannot afford
to re-encode. The incumbent prices siblings 5-8x more distinctly (0.00484 vs
0.00058/0.00092). In gen5 the sharp head carries 109-140x more discrimination than
the rolled path, versus 19x in the incumbent, so gen5's search is effectively blind
across most of the tree -- precisely where operators were supposed to pay.

**Mechanism (measured):** decomposing K(m)z = Az + sum_k w_k(m) U_k V_k^T z over 47
real positions, the move-DEPENDENT term is ~0.5-1pct of the shared A term. ~99pct of
what the operator does is identical regardless of which move is played.

**Why no loss caught it:** every k-step target follows the PLAYED line -- one move
per step -- where the shared A term alone predicts the next state. Nothing in the
objective ever required K(m) to depend on m.

**Diet is not the fix (tested).** A full 8,000-step run on a healthy sf diet raised
move-conditioning 2.3x and left it 2.6x below the incumbent. The sf diet governs
HEAD quality (0pct starved them; 36pct produced the lineage's best) and does not
govern play. Do not expect data fixes to close a search-path gap.

**The fix, implemented and gated OFF (`--w-sib`, `--sib-cap`):** roll the parent
through each legal move and price each rolled child against that move's own SF WDL.
SF already labels 100pct of legal moves (mean 32.3/position), so no new labelling is
required -- this is Kaveh's multipv observation as a loss. U21 gates that the
gradient reaches ops.U and ops.mix rather than only sf_v_full.

**Ship it bundled with a reduced U-penalty.** The identity prior's 0.1||U||^2 term
shrinks exactly the component this loss grows; at current weights they may cancel,
and the hard spectral projection sits in the same conflict. Per the no-one-lever
rule these move together, not in sequence.

**Also discarded and recoverable for free:** SF returns a ~10-ply principal variation
for EACH candidate (33 PVs/position); oracle_service keeps only pv[0]. Recording them
would give ~33 trajectories per position instead of the chain's single played line --
the "Stockfish for all children six plies down" data, already computed today and
thrown away. Free for new labels; a re-label for the existing 4.2M.

## Ruling 2026-08-24: the operator is state-conditioned, and composability is spent

`move_weights` now takes the POST-INTERACTION concepts (the phi tail of z_aug, after
the interaction transformer) alongside the move. Rationale, measured: without it K(m)
is a FIXED matrix per move identity -- the same transform in the opening and the
endgame -- so it cannot express that Nf3 is a capture *here*, and the rolled path
emitted a near-constant across siblings (4pct of the true spread; within-position
ordering 0.00 vs the sharp head's 0.532).

COST, ACCEPTED BY KAVEH: K is no longer linear in z, so K(m1)K(m2) is not a matrix
product and the operator is not a composable Koopman operator. His ruling: "we never
want to see exactly n plies ahead in one shot, we're happy to roll forward one ply at
a time." Search applies sequentially and re-derives the mixture at each node, so the
composition property was never being used. Everything that motivated the linear
design is retained: one encode at the root, one matmul per node, no re-encoding
anywhere below it.

FOLLOW-ON worth remembering: several constraints in this stack exist to keep LONG
COMPOSITIONS stable -- the spectral projection (U *= 0.97 when rho > 1.03) and, in
part, the identity prior. If we only ever roll 4-6 plies with weights re-derived each
step, those are more conservative than the use-case requires and are candidate levers
if the operator turns out to need more expressiveness. Not changed yet; flagged.
