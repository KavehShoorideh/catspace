# Stratified, tablebase-grounded, adversarial quasimetric planning — canonical architecture

*Status: current canonical design (2026-07-20). Supersedes ARCHITECTURE_SYNTHESIS.md for the
field/grounding/inference design; ARCHITECTURE.md remains the standing repo spec for shared
infra. History of how we got here lives in JOURNAL.md.*

---

## 0. One-paragraph statement

Learn a **quasimetric reachability field** over chess positions whose strata boundary is
**piece count** (only captures change it), **ground it bottom-up on the exactly-solved
tablebase** (perfect-play labels), and **extrapolate one-to-few strata above the solved
frontier**, where the capture boundaries serve simultaneously as curriculum levels, planner
subgoals, and error-reset checkpoints. The field is **factored** — a dense *cooperative*
reachability geometry (L1) plus *adversarial* outcome heads (L2) — because the pure adversarial
distance (game-theoretic **remoteness** / attractor rank; DTM is its chess instance) is a valid
quasimetric but **degenerate** (∞ off forced lines, finite only to regions), so it cannot itself
be the plannable geometry. Purpose: a verifiable toy for human-like planning that transfers to
agentic/robust planning; chess is the model system with an exact oracle to validate the
extrapolation.

---

## 1. The three layers (what each is, and why it is separate)

### L1 — cooperative reachability geometry (the quasimetric field)
- **Object.** `d(F(s) → B(g))` = a directed, composable distance = *how few plies of legal play
  (both sides cooperating) could take `s` to `g`*. Policy-independent — a property of the rules.
- **Model.** F/B two-tower encoder → IQE (Interval Quasimetric Embedding) head. Board-only
  (clock/repetition zeroed and carried as separate monotone potentials). GroupNorm (train==eval).
- **Training signal (policy-independent only — NO DTM):**
  - `L_pos`: `d(F(s)→B(s')) ≈ 1` on legal 1-ply edges (QRL local constraint).
  - `L_strata`: `d(F(child)→B(parent)) ≥ 1+margin` for **capture** edges only (piece-count drops
    are one-way — you cannot un-capture). This is the coarse strata / irreversibility.
  - `L_repel`: material-**unreachable** random pairs pushed apart (count-vector reachability).
- **Role.** Dense, plannable, composable (triangle inequality → subgoal stitching); provides the
  retrieval/OOD geometry and the landmark graph. This is the geometry the planner descends.

### L2 — adversarial outcome (reads L1's frozen embedding)
- **Object.** The game-theoretic value under **perfect (minimax) play**. Two heads, because the
  hard adversarial distance is sparse:
  - **Remoteness / DTM-to-region** (categorical over distance bins + a draw class): the *hard*
    forcing-distance to an outcome region (mate set, winning-material classes). = the classical
    **remoteness** function (Smith 1966) / **attractor rank** (reachability games). Sparse: finite
    only on forcible lines.
  - **Committor** `−ln P(W/D/L)` (harmonic, Doob-martingale checkable): the *soft, dense, graded*
    adversarial value that fills in where hard forcing is ∞.
- **Training signal.** Supervised on **exact tablebase labels** (perfect-play WDL + DTM), on the
  labeled nodes only; off-frontier it extrapolates and its entropy flags where it is unsure.
- **Role.** Turns the cooperative geometry into a decision value. Kept **separate** from L1 (frozen
  L1) to avoid the *small-world collapse* — co-training outcome into the geometry at full weight
  folds it into W/D/L clusters that fight the metric (battle-tested failure, JOURNAL).

### L3 — human/opponent playability (deferred)
- Omega-conditioned realized-cost model over actual (sub-optimal, human) play. This is where
  opponent **stochasticity** lives; the field stays perfect-play. Source: lichess. Not built yet.

**Why the factorization is forced (not a hack):** the adversarial forcing-distance is a genuine
quasimetric (triangle inequality by strategy concatenation) but **degenerate** — you can force a
*region*, almost never a *point*, so it is ∞ almost everywhere. A dense, plannable geometry must
therefore be the *cooperative* reachability (L1); the adversary enters as a value on top (L2).

---

## 2. Grounding: the bottom-up stratified perfect-play engine

- **Strata = piece count.** Only captures change it (promotion is count-preserving). The game is a
  strict DAG over piece-count strata; any position is **≤k captures from a solved stratum**.
- **Frontier = the tablebase.** Locally: the KRRvKBP endgame + its capture-descendants (3–6p), all
  in Syzygy. Computing remoteness/attractor by backward iteration *is* retrograde analysis — so
  **the tablebase already IS the exact adversarial forcing-distance for ≤ frontier**.
- **Labels = perfect play (deterministic, both sides optimal).** Exact WDL + DTM below the
  frontier; **negamax/MCTS-into-tablebase** one stratum above (bottoms out at the first capture
  into the solved set). This makes the failing data (draws/losses) *genuine* (lost vs perfect
  play), and the targets exact + reproducible.
- **Coverage = human-distribution starts.** Sample start states from lichess (on-distribution),
  relabel with perfect play. (Coverage from humans; labels from the oracle.)
- **Curriculum = bottom-up.** Introduce strata 3→4→5→6→7, checkpointing at each boundary
  (`_le3 … _le7`): a solved lower stratum is frozen ground for the next (DP in stratum order —
  non-circular by construction).

---

## 3. Inference: uncertainty-gated recursive minimax with retrieval-gated leaves

For an out-of-sample (e.g. 10-piece) position — one recursion, not two methods:
```
value(s):
  if pieces(s) ≤ frontier:                      return tablebase(s)         # exact base case
  nn, spread, nn_dist = retrieve(embed_L1(s))                                # vector-DB kNN
  v_hat = L2_head(embed_L1(s))
  if nn_dist small AND v_hat agrees with a shallow search:                   # confident base case
      return v_hat
  else:                                                                      # descend & re-ground
      children = minimax_search_toward_captures(s)     # reduce piece count toward the frontier
      return minimax_backup(value(child) for child in children)
```
- **L1** provides the OOD gate (nearest labeled-neighbor distance — ProQ's detector, done
  non-parametrically off the exact-frontier reference set) and the subgoal/landmark geometry.
- **L2** provides the leaf value (a non-parametric committor via kNN, or the parametric head).
- **Backup is minimax** (perfect defense we committed to); **averaging lives only at leaves.**
- **Trust gate = nearest-neighbor distance AND retrieval-vs-shallow-search agreement** — not
  neighbor-spread alone (chess value is non-smooth: one tempo / zugzwang / fortress flips WDL, so a
  tight cluster can be confidently wrong). Deepen where they disagree (adaptive horizon).
- **Hard guarantees** come only from reaching the tablebase; retrieval gives *calibrated
  confidence*, not proof. The OOD gate decides which regime a position is in.

---

## 4. Positioning / novelty (what to claim, honestly)

- The **field + landmark planner** machinery (IQE + QRL + repulsion-spread landmarks + directional
  cost + OOD gate) is **not novel** — ProQ (Kobanda et al., 2025, Inria/Ubisoft) independently
  arrived at nearly the same planner. We cite it and *concede* the overlap.
- **Novel (three axes), all where ProQ/QRL are silent:**
  1. **Exact-oracle grounding + extrapolation past the training support** via an exactly-solved
     sub-problem library — precisely ProQ's stated open limitation ("coverage limited to the
     dataset's convex hull").
  2. **Domain-structural boundaries** doing triple duty (curriculum + subgoal + **error-reset**),
     giving *bounded* compounding error (≤k captures from truth).
  3. **The adversarial axis (the one we lead with).** The whole quasimetric-RL line is
     single-agent. We give a **learned, composable quasimetric embedding of the game-theoretic
     remoteness / attractor-distance** — a classical object (Smith 1966; Conway/Berlekamp/Guy;
     parity/reachability-game attractors) that **no one has learned as a generalizing embedding or
     extrapolated off a partially-solved frontier**. The tablebase *is* that object solved exactly;
     we learn + extrapolate it. Generalizes to robust control (adversary = worst-case disturbance).
- **The binding constraint is ε_n** — how fast the field's error compounds per stratum above the
  exact frontier — corroborated by the OOD-value-generalization literature (GOAT; AVI error bounds;
  edge-of-reach). Chess is the one place we can *measure* ε_n against ground truth.

See `relevant_sources.md` for the full prior-art map and citations.

---

## 5. Data flow (one picture in words)

```
lichess games ──(start-state coverage)──┐
                                         ▼
                      sample starts → PERFECT-PLAY RELABEL (tablebase ≤frontier;
                                         negamax-into-TB above) → stratified dataset
                                         (positions + WDL/DTM + edges + capture flags + pairs)
                                         │
              ┌──────────────────────────┼───────────────────────────┐
              ▼                          ▼                            ▼
     L1 cooperative IQE field      L2 adversarial heads         vector DB of labeled
     (edges/strata/repel,          (remoteness-to-region +      embeddings (retrieval /
      NO DTM; bottom-up curric.)   committor; frozen L1)        OOD gate at inference)
              └──────────────┬───────────┘
                             ▼
             INFERENCE: uncertainty-gated recursive minimax descent to the solved frontier
                             ▼
             (later) L3 human playability from lichess modulates for real opponents
```
