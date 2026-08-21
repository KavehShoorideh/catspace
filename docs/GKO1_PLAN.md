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

## 7. Risks
  - Hard code-flips may defeat linear ops (captures): measured by L_op1 stratified by
    move type; if capture rows dominate the residual, ops get a capture-conditioned
    rank boost r=32 for capture codes (flag-gated, pre-registered).
  - Oracle latency: 50ms x 40k = ~35 min/SF round — acceptable; Maia nodes=1 is fast.
  - Joint-training instability with 10+ losses: warm-start everything possible
    (gcf4h trunk, gciv-style head init), operator A at identity, ramp L_op3 after 1k.
  - Pool staleness: acquisition re-scores every round on the CURRENT model.

## 8. Build order (each step lands with its unit tests green)
  1. vendored bagz reader + ChessBench test loader (U11); game_transitions cache for
     both corpora (the 6-ply composition targets)
  2. hl_gauss.py (U5) ; ops module (U1-U4) ; interval head reuse (U6-U7)
  3. oracle_service.py (U9) + maia weights fetch
  4. acquisition.py (U10, U15)
  5. gko trainer assembling all (U12-U14) ; invariant review script that GATES
  6. smoke (crash/scale only) -> round-0 bootstrap (gciv scorer) -> launch producer
     FIRST (warm the label store while units finalize), consumer joins when gates pass
