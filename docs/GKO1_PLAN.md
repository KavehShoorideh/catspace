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
    2-3-ply composed rollouts (composability is the point).
H2 (intervals): jointly-trained concept intervals reach coverage within [nominal,
    nominal+0.07] with width narrower than the frozen-field probe by >=25%.
H3 (query efficiency): active acquisition (interval-width x code-rarity x entropy)
    reaches the same ChessBench-test action-value fidelity as random querying with >=5x
    fewer oracle calls.

## 2. Components (tensor contracts)
- Trunk + concept heads: GraphConceptField (gcf4h arch; warm-start gcf4h_latest),
  z_state = zq in R^{d_state}; move tokens: mv_vq codes c(m) in {0..63}.
- KoopmanOps module: z' = clamp_spectral(A + B_{c(m)}) z
    A in R^{d x d} shared; B_c LOW-RANK (U_c V_c^T, rank r=16) per move-code; spectral
    clamp: power-iteration estimate of rho(A+B_c), penalty max(0, rho - 1.02)^2
    (Koopman Dreamer's stabilizer, penalty form). Null-move operator: B_null == 0 and
    A initialized at I (identity-at-init invariant).
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
- Active loop: round k: score pool on current model -> top-B by acquisition (dedup by
  concept-code signature to enforce diversity) -> query -> add to labeled set ->
  fine-tune -> re-score. Budgets: B=5k/round, 8 rounds v1 (40k SF + 10k Maia calls).
  Baseline arm: random acquisition, same budgets (H3 needs it; approved as part of the
  hypothesis test, not an A/B rollout).

## 4. Losses (weights at launch; interaction review inline)
  L_sf (HL-Gauss CE, all legal moves)          2.0   the distillation target
  L_maia (KL, queried bands)                   1.0
  L_op1 (one-ply: ||K_m z - sg(z_child)||)     1.0   operator grounding (real children)
  L_op3 (3-ply composed vs real, 25% of batch) 0.5   composability trained, not hoped
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
  U1  operator identity: A:=I, B:=0 => z' == z exactly (null-move fixed point)
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
  - Operator table: 1/2/3-ply R2 vs MLP dyn (H1); persistence-table accuracy.
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
  1. vendored bagz reader + ChessBench test loader (U11)
  2. hl_gauss.py (U5) ; ops module (U1-U4) ; interval head reuse (U6-U7)
  3. oracle_service.py (U9) + maia weights fetch
  4. acquisition.py (U10, U15)
  5. gko trainer assembling all (U12-U14) ; invariant review script that GATES
  6. smoke (crash/scale only) -> round-0 bootstrap (gciv scorer) -> launch
