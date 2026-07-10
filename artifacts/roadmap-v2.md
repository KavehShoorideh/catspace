# Roadmap v2 — Latent Chess Planner (core thesis re-cut)

**What changed from v1:** the core thesis is now **planning as a trajectory of the cone in a learned embedding, with self-discovered (never hardcoded) concepts** — "a chess engine you can watch think." ω/Elo/clock conditioning demoted to Extension Track. Fatal gates reordered accordingly. Same fail-fast discipline as v1 (pre-registered gates, kill/degrade named in advance, sequential tests, artifact-on-kill, 2× calendar circuit-breaker).

## Core track (fatal gates)

| M | Milestone | Question (gate) | Modules | Est. | Fatal? |
|---|---|---|---|---|---|
| **1** | **Toy-domain learnability** (small-board KRk): learn F/B from played games; rank probe; spectral + VQ concept emergence checked against exact ground truth; greedy cone-steering engine mates | G-M1: (a) rank-d captures reach structure; (b) learned B/F dims + VQ codes correlate with known concepts *without being trained on them*; (c) engine trained from random play beats random baseline decisively | state, cone(F/B, VQ), regions(eval-probes only), planner(greedy), eval | 2–3 wks (starting now) | **YES** — if nothing legible/useful is learnable at toy scale, stop |
| 2 | Full-board FB + VQ plan bottleneck (frozen lc0/Maia trunk, d≈64–128, K≈128–512 EMA+dead-code-reset) | G-M2: codebook perplexity healthy (no collapse); plan tokens carry position semantics (retrieval + probe checks vs. exact python-chess predicates — post-hoc measurement only) | encoder(lc0), cone, learn | 4–5 wks | **YES→pivot** (FSQ fallback; per-stratum F fallback) |
| 3 | Plan-trajectory value A/B: z-bottleneck planner vs same-capacity plan-free net, vs weak fixed opponent | G-M3: pre-registered win-rate edge for the *trajectory mechanism itself*, anytime-valid sequential test, capped budget; target ≈1000-Elo-level play with legible plans | planner, proposer, game, eval | 4–5 wks | **YES** |
| 4 | Legibility: plan-token filmstrips (anchor retrieval), persistence/replan events, region flow graph from the learned model | G-M4: blinded read-test — do token streams read as coherent intent? (Honest possible outcome: wins but thinks illegibly — publishable either way) | viz, regions | 3 wks | no (this is the payoff, not a gate that kills) |

**Stopping point for the current push (per direction): Milestone 1 complete = "we learned something"** — demonstrated end-to-end at toy scale: learned embedding, emergent concepts, engine that plays using them.

## Extension track (formerly core, now optional, unchanged specs from v1)
E1 ω/Elo-conditioned kernels + trap regions (old G0/Phase 3) · E2 clock/SMDP + flagging · E3 quasimetric layer hardening · E4 conditional discovery pipeline (SAE route, gated by trap-region existence). None are fatal; none block M1–M4.

## Milestone 1 spec (executing now)
Domain: KRk on a 5×5 board (~14k white-to-move states — exact everything is computable, so every learned quantity has ground truth). Data: games from **random play only** (nothing optimal, nothing labeled). Learn: empirical transition estimates → successor measure M̂ → F/B via SVD factorization at rank d (the exact analogue of FB), at several data sizes (learning curve). Concepts: (i) spectral — do top B/F dimensions correlate with DTM (depth-to-mate), king proximity, rook-cut/box-area? (ii) VQ — k-means plan tokens on cone shapes F(x); usage perplexity = measured concept count. Engine: greedy cone-steering toward the "near-mate" *region* (z_G = Σ_{g∈G} B(g), G = DTM≤3 — "a mate somewhere in there"). Eval: mate rate + move count vs. random baseline and vs. DTM-optimal ceiling; filmstrip of one game with per-ply plan token + concept readout. Pre-registered G-M1 thresholds: rank-64 relative reach error <5%; ≥3 embedding dims with |ρ|>0.5 to distinct ground-truth concepts; engine mate-rate ≥5× random baseline within 100 plies.
