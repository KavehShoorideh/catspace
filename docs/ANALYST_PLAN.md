# ANALYST_PLAN — a chess analysis + preparation tool that explains, on top of Stockfish

**Status: PLAN BONES (2026-09-04).** Kaveh's pivot: the product is a browser tool that (1) analyzes
a game/position and explains the *ideas* in text, and (2) prepares you against a specific
opponent. Stockfish is conceded as the best searcher and used as a TOOL — at inference and in
training. We read its output (multi-PV, evals, null-move threats) and explain it with concepts;
we do not compete with it on strength. Nothing here is built; sequencing needs Kaveh's go.

---

## 1. The idea, assessed

- **Right frame.** "Better than Stockfish/Leela" means better at *explaining*, not evaluating.
  Every existing explainer either templates rules (DecodeChess) or wraps an LLM around engine
  lines with little grounding (ChessLogix, Chessvia, ChessMindAI). The open gap is
  **faithful, grounded explanation**: structured facts first (PVs, exact board features,
  null-move threats, concept deltas along the PV), text second, every sentence traceable to a
  fact. That is the project's interpretability lineage pointed at a user.
- **Prep is under-served for club players.** Incumbents: ChessBase 18 (paid desktop, pro),
  OpeningTree (free, discontinued 2022, still works), Lichess personal opening explorer (openings
  only). None *explains* what to do against the tendencies they find.
- **Personal-use bar is low; publishable bar is the faithfulness gauge** (§6).

## 2. Market (verified 2026-09-04)

| Product | What it does | Gap we exploit |
|---|---|---|
| DecodeChess ($8.25/mo, alive) | SF NNUE + rule-based "explainable AI": threats, plans, piece functionality; pitched ≤2000 Elo | template explanations, no prep, no learned concepts |
| ChessLogix / Chessvia / ChessMindAI | SF17 + GPT-4/Claude prose, voice coaching | unverified prose; hallucination risk; no structured idea layer |
| Chess.com Game Review + Coach | mass-market review with brief NL notes | shallow; closed data |
| Lichess analysis | free SF in browser; **"show threat" (x key) = null-move analysis**; personal opening explorer; insights | no explanation layer |
| ChessBase 18 | pro opponent prep incl. Lichess-account inference | desktop, paid, no explanation of *ideas* |
| OpeningTree | opponent repertoire tree from lichess/chess.com games | discontinued; openings only |
| Aimchess (→ chess.com), Noctie | weakness stats / human-like sparring | not analysis-explanation |

Research precedents: Jhamtani et al. 2018 (GameKnot, 298k move–commentary pairs, noisy);
ChessGPT dataset (HF Waterhorse/chess_data: games + language + mixed); **CCC-GPT (arXiv
2410.20811): concept-guided chess commentary generation + evaluation** — the closest recipe to
ours; Schut et al. 2023 (arXiv 2310.16410): concepts discovered in AlphaZero and *taught to
grandmasters* — the precedent for "engine ideas → human concepts".

## 3. What the repo already has (reuse map)

| Asset | Where | Role in the product |
|---|---|---|
| Named concept basis, Rust (28 per-square + 100 global: att/least-att/SEE/pinned/pinning/x-ray/hanging/defenders/king-ring; passed/isolated/doubled pawns, material, phase…) 4.5 µs/pos | `rust/catspace-concepts`, `concepts_reference.py` | the deterministic, *understandable* concept layer; feature deltas along a PV = "what the idea changes" |
| Threat vs execution labels | `gen_threat_execution.py` | "threat exists / when it gets cashed" — tension narration |
| SF oracle pipeline (multi-PV, PVs, sqlite cache, producers) | `oracle_service.py`, `gko1_oracle.sqlite` | the labeling backbone for training AND the inference cache |
| Browser UI (chessground + engine server + per-player memory) | `kittychess_server.py`, `player_profile.py` | the product shell; player_profile → opponent memory seed |
| GNN concepts with touch-set locality + concept dynamics | qgt line | stage-4 learned concepts for what the basis misses (plans); "how concepts roll forward" |
| Style-z / opponent modeling designs (M2b) | memory + MILESTONES | prep: predict what *this* opponent plays |
| Syzygy TBs, chessbench puzzles (10k, no themes) | `data/` | exact zugzwang/threat ground truth in endgames; eval set |
| Missing | — | lichess/chess.com game fetchers; Lichess puzzle DB with **themes** (free concept labels, ~4M); annotated corpora |

## 4. The null-move threat trick — checked 2026-09-04 (local SF, depth 18, printed)

Method: eval(us) with us to move vs eval after `Move.null()` (opponent moves twice).
THREAT = gain the opponent gets from a free tempo. ZUGZWANG = passing would score better than
our best move. Illegal when in check.

    mate threat (3.Qh5, black to move)   eval +40 → opp after pass +9999 (Qxf7#)   THREAT +10039
    hanging queen (3.Nc3, black)         eval −71 → opp +757 (Nxd5)               THREAT  +686
    quiet (2.Nf3, black)                 eval −39 → opp +147 (Nxe5)               THREAT  +108

Works, with three refinements the product needs: (a) **baseline** — a free tempo is worth
~+100 cp even with nothing on; report threat *above* that baseline; (b) **"real" threat test** —
a threat matters only if our best move's PV neutralizes it (compare eval after our best move vs
after passing → the residual is the threat's teeth); (c) **enumerate threats** with multi-PV on
the null-move position, not just the top one. Zugzwang is exact in ≤7-man endgames via Syzygy
(WDL to-move < WDL if-pass) and SF-estimated elsewhere. Lichess already ships (a)'s raw form as
the "x" key; the product's edge is narrating it ("if you do nothing: …") and combining it with
concepts.

## 5. Components ("the bones")

1. **Engine service** — SF via python-chess UCI: multi-PV, per-move evals, null-move threat +
   zugzwang probes, sqlite cache (existing). Optional: classical-eval term trace from SF ≤15 for
   named eval terms (NNUE SF has none).
2. **Fact layer** — Rust basis features per position; **feature deltas along each PV** (what
   the idea changes: material, hanging, pins, king ring, passed pawns…); threat/execution;
   tension. Output: a structured **Idea Record** (JSON) per position/move.
3. **Concept layer** — v1: the named basis + puzzle-theme detectors trained on the Lichess
   puzzle DB (free labels: fork, pin, discoveredAttack, zugzwang, …). v2 (stage 4): learned local
   concepts (GNN, touch-set budget = "fewest pieces per concept") for plans the basis misses.
4. **Explanation layer** — LLM (Claude) prompted *only* with the Idea Record + retrieved
   exemplars; a **verifier** rejects any sentence not traceable to a record field (CCC-GPT
   recipe). Retrieval over GameKnot/ChessGPT/lichess-study annotations for style, not truth.
5. **Prep engine** — fetchers (lichess API, chess.com public API; no auth for public games);
   repertoire tree by color/time-control/recency; tendencies (blunder-by-phase, time trouble,
   recurring structures); **weakness miner** = run the analyst over their games and aggregate
   recurring bad concepts; prep report = lines into their weaknesses + explained ideas + model
   games. Later: style-z / Maia "what will they play".
6. **Browser UI** — kittychess shell: board, eval bar, threat overlay, concept chips,
   explanation panel, PGN/URL import, prep dashboard.

## 6. Gauges (big-jump-shaped, printed)

- **Concept precision/recall** on Lichess puzzle themes (free labels, millions of positions).
- **Faithfulness**: fraction of generated sentences whose claim the verifier can ground in the
  Idea Record; target ≥0.95 before anyone reads the prose.
- **Threat detection** vs puzzle "threat"/"defensive" themes and vs Syzygy zugzwangs (exact).
- **Usefulness**: Kaveh's own games — does the explanation name the idea he missed? (weekly).
- Prep: held-out games of the opponent — do predicted lines/tendencies recur?

## 7. Staging (each stage ships something usable)

- **S0 Analyst core (CLI)**: SF multi-PV + null-move threats + basis deltas → Idea Record. Gauge:
  puzzle-theme recall. ~1 week.
- **S1 Explanations**: grounded LLM + verifier + faithfulness gauge; Kaveh reads his own games.
- **S2 Browser**: kittychess shell → analysis view; PGN/lichess/chess.com import.
- **S3 Prep**: fetchers → repertoire + weakness miner → explained prep report.
- **S4 Learned concepts**: GNN local concepts + concept dynamics for plan-level ideas.

## 8. Risks / decisions for Kaveh

- Text supervision (GameKnot) is noisy — use it for retrieval/eval, not as the primary training
  signal; primary signal = free structured labels (SF PVs, null-move, puzzle themes, deltas).
- LLM cost/latency per position; cache Idea Records; explain on demand, not every ply.
- Rust basis vs learned concepts: start with the basis (legible by construction); learned
  concepts must clear a "names itself via puzzle themes" bar before they appear in prose.
- Scope: personal tool first; publishable claim = faithfulness + concept transfer to humans.

---

## 9. Revisions (Kaveh, 2026-09-04, same day)

**Desktop, not a service.** Every person runs their own copy: local Stockfish, local concept
basis, local models. No backend, no accounts, no cloud calls by default. Consequences:
- Packaging: the concept basis is already a Rust crate, the UI is already chessground; a
  **Tauri** shell (Rust) with Stockfish as a bundled sidecar binary is the natural desktop
  target. Iterate the analysis core in Python (CLI) first; port hot paths to Rust when the
  spec is stable. The kittychess localhost pattern is fine for development, not the product.
- Training is done ONCE on Kaveh's box (Stockfish as labeler); users receive weights/tables,
  never a training pipeline.
- Prose is local: **(L1) deterministic verbalization** from the Idea Record via an idea
  grammar (faithful by construction, zero model), then optional **(L2) local LLM polish**
  (Ollama is already on the box: llama3.1:8b, qwen3.6, mistral-small) restricted to restating
  record facts, with the verifier. Cloud LLMs only as an opt-in with a user's own key.

**The core capability, stated precisely.** For a Stockfish line, say (a) *what the idea is*
and (b) *which ideas no longer work, and why*.
- (a) IDEA = the named basis deltas between the position and the end of the PV (what the line
  changes: material, pieces that become hanging/pinned/defended, king-ring pressure, passed
  pawns), plus the source of the terminal advantage. Concept dynamics ("moves predict concept
  changes") is the learned generalization of this.
- (b) KILLED IDEAS = candidates that were near-best in the previous position (multi-PV there)
  and are now refuted (eval drop > threshold in the current position); for each, the
  REFUTATION line and the basis deltas along it name the *why* (e.g. "the knight on d5 is now
  defended by the bishop, so Nxf7 no longer wins material"). This is the refutation →
  precondition-vector idea from the planner design, and the M7 "blocking condition" —
  the same object, read for a human. Also CONTRASTIVE alternatives in the current position
  (top-k lines that fail) with their refutations.
- First prototype run 2026-09-04 (see JOURNAL / session): mechanics work with the existing
  basis and Stockfish multi-PV; the vocabulary (hanging/pinned/see/king_ring/material) already
  produces readable "why"s; plan-level ideas (space, pawn breaks, piece routes) are the gap the
  learned concept layer must fill.

**Prototype verdict (2026-09-04, printed; Fried Liver after 6.d4 Be6, SF depth 16, multi-PV 5):**
the Idea Record found the killed idea exactly — `Nxf7: was +87, now -395 after ...Be6; refutation
Bxf7 (we lose a knight)` — and the current idea `Nxe6 (+100)`. Two lessons for the build:
(1) the per-move fact extractor needs curation: king-ring changes on castling and the mover's
own-square changes are noise; captures, hanging/pinned/defended changes on OTHER pieces, and
material are the signal. (2) **The local 8B LLM (llama3.1) corrupted the facts even when told
not to** ("Black loses another knight with fxe6" — it was White's knight; "gaining movement for
their queen" — invented). So L2 polish is unusable without the verifier, and L1 deterministic
prose is the faithful default. Faithfulness is not a nice-to-have gauge; it is the product.

## 10. S0 status — BUILT 2026-09-04

`catspace/incubator/analyst/` (one module per layer + `engine.py`, `cli.py`, `gauge_puzzles.py`);
tests in `tests/test_analyst_*.py`; Opera Game fixture. Numbers and lessons: JOURNAL 2026-09-04
"ANALYST S0 BUILT". Layer cut as in §9 with the objective quarantined to L5 and all L3 relations
defined through L1 probes (removal test). Next: S1 explanations polish + plan-level concepts;
S2 desktop shell (Tauri + sidecar Stockfish); S3 prep.

## 11. Closest existing work (searched 2026-09-04, after S0) — one neighbour per organ, none for the organism

| Neighbour | What it is | Overlap with us | What it lacks |
|---|---|---|---|
| Lichess puzzle tagger (`ornicar/lichess-puzzler/tagger/cook.py`) | the python-chess program that assigns the theme labels our gauge uses | same probes (attackers/attacks/pin/between); fork/hanging definitions close to ours | solution-conditioned (needs the puzzle's moves), standard-chess-hardcoded (king special-cased, fixed values, ray piece types), absolute pins only via `board.pin`, no position-level relations, no explanation |
| CARA (`pguntermann/CARA`, GPL-3, PyQt6 desktop, active) | desktop review app: 44 rule-based tactical/positional patterns over Stockfish multi-PV, PV piece-trajectory visualisation, LLM chat via cloud API keys | closest PRODUCT shape to S0+S2 | hardcoded chess rules (no variants), no killed-ideas / contrastive layer, no null-move threat narration, prose = cloud LLM chat (no faithfulness contract) |
| ChessGrammar (API) | 10 patterns from FEN/PGN, python-chess heuristics, no engine | L3-like | no dynamics, no engine grounding, API not desktop |
| En Croissant (Tauri, open source) | desktop GUI: multi-engine, game reports (win-prob-loss move classes, best lines), lichess/chess.com import, repertoire + spaced repetition, tablebases | closest SHELL for S2; already covers game import and part of S3 prep | no explanations, no concepts |
| DecodeChess (commercial) | rule-based explanations of threats/plans/piece functionality | closest explanation product | closed, templated, no prep, standard chess |
| CCC-GPT (arXiv 2410.20811), Schut et al. (2310.16410), Ludii concepts | concept-guided commentary; engine concepts taught to humans; rules-generic game concepts | the research frame | none is a tool; Ludii concepts are for game design, not tactics |

Ours alone: rules-generic relations via the removal test + royalty (variant = one-layer change);
arity as locality/level; "what no longer works" from one move-pair ago with refutations;
null-move threats above a tempo baseline, narrated; deterministic prose with a faithfulness
contract; the theme gauge with degenerate baselines as a development instrument. Note the gauge
measures agreement with the tagger's rule set, not human ground truth -- the second instrument
is Kaveh reading his own games.
