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

## 12. Open layer: a PLAN as a sequence of concepts (Kaveh, 2026-09-04 evening)

"We need to be able to express a plan as a sequence of things -- a sequence of concepts -- and I
don't know how to do that yet; I don't even know what a good plan sounds like." Recorded, not
solved. What exists: L4's per-ply effects along a Stockfish line are already a sequence of
concept EVENTS (pin created -> pinned piece attacked -> captured). What is missing is the
abstraction that names such a sequence as one thing ("win the pinned piece", "open the e-file
against the king") and that can be compared across games. Candidate homes: an L4.5 that
compresses effect sequences into named motifs (grammar over relation events), or the learned
concept-dynamics layer (moves predict concept changes; planning subgoal-first). Decide after
Kaveh's own games are on screen.

## 13. Vocabulary completeness is a measurement (2026-09-04)

`motifs.py`: enumerate k-piece tension pattern types over the primitive graph, count them in puzzle
positions, report named coverage and the top unnamed types with examples; enumerate ply-level edge
events along solutions and score each Lichess theme by its best-predicting event or two-ply
conjunction. Re-run after every vocabulary change. Current: arity-2 named 45%, arity-3 6% (the
unnamed mass = alignment through blockers + multi-duty pieces, i.e. precursors -> L3.5); sequence
motifs get data-derived definitions (discovered 0.76, skewer 0.52, sacrifice 0.41, attraction
0.28, deflection recall 0.81). Next: three-ply conjunctions; then promote the frequent unnamed
types and the best sequence definitions into named relations / sequence relations.

## 14. Layer boundary for evaluation changes (Kaveh, 2026-09-05)

At L3-L4.5 a bishop-for-knight trade is materially EQUAL. The engine's eval change after such a
trade is about which plans became available or unavailable -- that belongs to the plan layer (§12).
Rule for the explainer: never invent a tactical reason for a positional cost; say "an even trade;
the cost is positional -- what the trade does to the plans -- which this layer does not explain".
Likewise the refutation's TEETH (its margin over the opponent's second-best reply) decide whether an
error was a tactic suffered or an opportunity missed; and the played move's own PURPOSE (the pin it
broke, the piece it saved) is acknowledged before the cost is stated.

## 15. The exchange scale (Kaveh, 2026-09-05: "piece values should depend on the variant")

`rules.Scale`: values per piece type, royal value, material sign, and the "real piece" threshold.
Declared with the objective in L1 (`Rules.scale`; `set_scale(variant, scale)` swaps it at runtime),
because it is a derivative of the objective, not a rule. Consumed only through `facts.value()` /
`Scale.of_symbol()` by: the valued L3 relations (pin vs skewer vs x-ray, a fork's teeth, attacked by
a lesser piece), capture prices in L4 effects, significance thresholds, even-trade detection, and L5
material deltas (sign). Standard chess 1/3/3/5/9; crazyhouse minors 3.5; antichess sign -1 and no royal
value. Known limit: the prose-side thresholds in L4/L6 use the standard scale unless a board is in hand.

## 16. L3.5 structure (2026-09-05): the positional residue gets a vocabulary

A third of the errors in Kaveh's rapid games have no tactic in the reply, and even trades cost without any
piece changing hands. `structure.py` gives those moves something true to say. Facts, all Relations so L4 diffs
them per move and L6 phrases the diff: isolated_pawn, doubled_pawns (identity = side + file), passed_pawn (tag
= ranks to go; an advance is one "push"), backward_pawn, pawn_islands (speaks only when the count changes),
rook_on_open_file / rook_on_half_open_file, outpost (a minor piece in the enemy half, held by a pawn, that no
enemy pawn can ever attack), bishop_pair (identity = side), bad_bishop (two or more own blocked central pawns
on its colour), king_open_file (only once castling rights are gone), and scope (squares a piece reaches; phrased
only for a piece that did NOT move and gained or lost three squares: "opens a line for the bishop on c8").
Used by: even-trade headlines ("What changes: White gives up the bishop pair"), quiet missed moves ("missed e5
-- opens a line for the black bishop on c8"), the per-move report as the fallback fact, the long render
("Structure after ..."), and a review table (structure changes in your errors vs in the best move at the same
positions). Rule kept from §14: these are observations; causes belong to the plan layer.

## 17. The drill (2026-09-05): your errors as a deck

`drill.py` turns the review's rows into one HTML page (`tmp/drill.html`, published as an artifact). Two question
types: PUNISH (the position after your mistake, from the opponent's side; answer = the refutation the review named;
the ladder shows how often your level plays that mistake) and FIND (your position; answer = the better move; kept
only when Maia 1500+ finds it at least one time in ten, so the question is fair). A right answer moves the position
up a Leitner box (due in 1, 3, 7, 21 days), a wrong one back to now; progress lives in the browser. Grading is exact
against the named move; in FIND positions other moves may hold, which the page says. Regenerate after every review.

## 18. The local app and the flag queue (Kaveh 2026-09-05)

`web.py` + `web_page.html`: one server on the machine (`python -m catspace.incubator.analyst.web`, port 8765),
one page. Paste a Lichess game link or id: the game is fetched, analysed move by move (both sides, depth 18, the
Maia ladder) and shown as a board + move list + a card per move (verdict, cost, headline or the move's fact, best
line, standing threats, structure changes, ladder bars, the full text under a disclosure), stepping with the
arrow keys; partial results stream in while the job runs. Paste a user name: their recent games are fetched and
reviewed with the parallel driver; the page shows the profile block, the errors biggest-first and the games,
each opening the game view at that move. Results are cached as JSON under data/derived/web/.
The "Something is not right" button (floating, on every move card and on the profile) appends what was on
screen plus an optional note to `feedback/flags.jsonl` (tracked in git; a flag's id is its line number).
The coder answers in `feedback/RESOLVED.md` by id. `review.annotate()` is the shared per-move code path, so
the page, the report and the drill say the same thing about the same move.

## 19. Composition (Kaveh 2026-09-05): atoms by probe, compounds by operator, nothing higher by hand

"If we don't compose from the lower-level concepts, then under a rule shift how do we know the higher-level concept
still means the same thing?" The rule: an atom is defined by an L1 probe; a compound is defined by operators over
atoms; nothing above the atoms is named by hand. The operators so far: glue on shared pieces (the composed pins),
the removal test (pins, skewers, discoveries), the exchange (SEE under the variant scale), the null move (threats),
and now EFFECTIVE DEFENDERS (`defense.py`): simulate the cheapest legal capture, keep a defender only if the rules
allow its recapture (absolute pins, checks, atomic adjacency, forced captures -- whatever the variant says) and if
recapturing does not expose more than it wins (the removal test on the defender's origin, priced by SEE; exposures
the capture itself uncovered are not charged to the recapturer). `underdefended` and `hanging_through_pin` are
now READINGS of that trace, and the prose comes from the trace: "its only defender, the black pawn on c7, cannot
legally recapture"; "the white pawn on c3 does not count: it is overloaded (it also guards b4)". No static
exchange is claimed while the owner is in check. Test: the same placement under antichess (no royal king) turns an
undefended pawn into a defended one. Limits: one level deep (no overload chains); the composed four-piece pins
remain as structural readings beside the operator; the enumerator (motifs.py) stays the coverage gauge for what
the operators have not yet named.

## 19b. The attackers' side of the ledger (Kaveh 2026-09-07, a rook ending from a screenshot)

Kg3 Ra4 a3 b4 h5 v Kg6 Rc4 a5 b6 f6, after ...Kxh5: the plan layer said "win the pawn on a5 in 1: bxa5 (attacked 2,
effectively defended 1). The race is ours" while the goal search refuted it at once with bxa5 Rxa4 -- the b4 pawn
shielded the a4 rook from the c4 rook, and Black declines to recapture. The ledger priced a5 by the exchange on
a5 alone. Rule: the two tests the operator applies to defenders apply to attackers too. An attacker counts only
if its capture is legal (an absolutely pinned attacker is out) and its departure does not hand the owner a piece
worth at least the target that the owner may LEGALLY take (a capture with check leaves no time, so the exposure
does not count then). Disqualified attackers are excluded from the exchange like disqualified defenders; the
flip conditions, the planner's removal targets and the search candidates use the attackers that count; the
prose says why ("the white pawn on b4 cannot take: its departure hangs the white rook on a4"). Stored ledgers
carry a version in the store key so the change never reads a stale pickle. After: "held: the black pawn on a5
(1 v 1)", no plan offered for White, the search still gives the goal up (now trying Rxa5+ before bxa5).
One level deep, like the defenders' side: the exposed piece is priced by SEE with nominal pieces elsewhere.

## 20. Reach and the progress map (Kaveh 2026-09-05): the tactics that are not executable yet

`reach.py`, two more operators over the atoms and the defense ledger. FLIP CONDITIONS: for a held target, the
loosest single change that makes the exchange win it (one more attacker worth at most v; one defender gone).
SAFE REACH: the fewest quiet moves for a piece to a square from which it attacks the target, walking the piece's
move graph from the rules on a scratch board and landing only where the defense operator says it survives -- so a
wrong-coloured bishop, a boxed-in piece or a king facing a defended pawn simply never arrive. Composed into the
PROGRESS MAP of a side: each enemy piece is 'falls now', 'held; X joins from S in n moves and it falls',
'held; its defender Y falls first, then it does', or held/out of reach (said once). `verify_plan` gives the side
its hops with null moves for the other and asks the ledger, then lets the engine answer the first hop: the
outcome is 'wins', 'chased' (the target moved: a tempo, not the piece) or 'met'. Shown in the long render
("Progress for White: ...") and in the app's move panel. Transfer test: under antichess a king takes the pawn a
pawn defends. Known limits: reach ignores replies (verify_plan is the check); one level deep; three hops.

## 21. The concept store and the search pathway (Kaveh 2026-09-05)

`memo.py`: a concept is keyed by its FOOTPRINT, not by the position. Each operator that runs through the store
records the squares whose occupancy could change its answer -- the target and the rays and jumps into it, cut at
the first piece that could not take part in an exchange there; a piece's own rays to the first blocker; the pin
line from a piece to its king -- plus cheap predicates recomputed at lookup (who checks a king). An entry holds
the value, the mask and a Zobrist hash of the pieces on the mask; a lookup from any position hits when those
pieces are the same. Sound by declaration, checked by a property test (memo-served == fresh over ~110k
defense/reach checks across four seeds, zero mismatches). Hit rates from a parent to its children: defense 53%,
reach 23%; a second run on the same position 73%; a position two plies later 60% from the store. Persisted in
sqlite (data/derived/analyst_memo.sqlite); the app's analyst loads it, uses it for the progress lines of every
ply and saves after each game. `search.py`: a search that expands candidates (the engine's top-k when given;
always the captures of falling pieces, the joiners' first hops, checks) and evaluates every node by its concepts
(ledgers, both progress maps, material), with the DELTA of concepts on each edge ("cxb6: pawn on a7 now falls;
pawn on c7 now falls; my pawn on b6 now falls; material +1") -- the plan layer's answer to what a move does.
Limits: footprints are conservative (a move that lands on a ray into a target invalidates it even when it could
not take part); entries per key capped at 8; the search has no value and no pruning yet: it is the pathway.

## 22. The planner: suggestions by a chain of reasoning (Kaveh 2026-09-06)

The interior-node experiment (journal 2026-09-06) settled what the ledgers are for: not for ordering the engine's
search -- Stockfish already tries winning captures first, and when its first move fails the cutting move is a quiet
move 71% of the time -- but for the plan layer. `plans.py` is backward chaining over typed goals with the operators
as rewrite rules: win(t) reads the ledger; held targets rewrite through the flip conditions into "bring a piece"
(reach) or "remove the defender" (win(d), recursively); defend(t) rewrites into escape, add a defender, or win the
attacker. Every rewrite is one rule over one object, every leaf a move, the chain is the explanation, and branching
is two or three per step. Each plan is answered by the opponent's COUNTER-GOALS on the board after our first step
(defend the target, or win our joiner) and the race is called by hop counts; multi-step plans can be verified by
the engine (reach.verify_plan). Defensive plans for our own falling pieces come first: on AUjG3sl4 after 16.c4 the
first plan is "hold the queen on d5: Bxc4 -- the attacker on c4 is removed", the move played. In the long render
("Plan for ...") and the app's move panel ("plans for White, by reasoning"). ~10-30 ms per position with the store.
Limits: goals are material only (no positional goals yet: open the file, occupy the outpost -- the quiet 70%);
counter-goals one level; races by hop count ignore tempo gained by checks and captures.

## 23. Goal search: proof-number search toward a material goal (Kaveh 2026-09-06)

`goal_search.py`. "Win the f7 pawn" is a boolean over an AND/OR tree, so the search is proof-number search
(Allis 1994), not Monte Carlo: OR nodes are our moves, AND nodes the opponent's; every legal move is generated on
both sides, so a wrong prior costs time, never the answer, and a proof holds against every reply. The prior is the
planner: our plan steps and the opponent's counter-goals are ordered first at shallow plies (a cheap target-centred
order deeper) and start with lower proof / disproof numbers; the ledger initialises them ("falls now" = 1).
The goal predicate at our turn: the target has been captured, our material is up by its value on the raw count
(in hand: Nxc5 dxc5 is not a pawn won), and the ENGINE, looking to the same absolute horizon from every leaf
(depth 8 at the search horizon, 14 at the root), says the position is not worse than the root by more than 40 cp
-- the engine is the quiescence, so recaptures, counter-attacks and horizon tricks are its job; it is not asked
to have risen, because a root evaluation already anticipates a pawn that is falling; its verdict is final. A leaf
whose count is in hand only after a pending capture (the ledger's settled count) may run up to 4 plies past the
horizon to cash it. Fresh leaves are priced by the work they will need (few open replies after a capture of the
target, most after an unrelated quiet move) so the search goes down the plan move instead of breadth-first over
the root's siblings. Give-up is native: a refuted root (disproof number 0) or an exhausted expansion budget
abandons the goal with the moves tried and the hardest defence found; `pursue` walks the progress map's targets
most valuable first, gives up, moves on, stops at the first proof. Worked transcripts on tPVi7fZF (Kaveh's latest
game) are in docs/ANALYST_GOAL_SEARCH.md: "win the bishop on b4" after 12.a3 proven in 13 expansions (Nxb4; axb4
Rxa1, the b1 knight blocks the queen's defence of a1); "win the pawn on c5" after 15.O-O in 18 (bxc5; the
desperado dxc5 met by Qxe5, the knight the game missed); "win the knight on e5" after 16.dxc5 in 11 (the goal
loop's first target). On AsXfL6OX before 18 for White, "win the pawn on b6" is GIVEN UP, rightly: Black's doomed
g3 pawn cashes h2 (gxh2+), the greedy Nxh2 is refuted by the judge (...Qh7), and after Kh1 the count is a pawn for
a pawn (an earlier flat-depth judge had "proven" it through Nxh2: a false proof, removed by the consistent-horizon
judge). For Black there, "win the knight on f3: given up (tried g4 first...)". Seconds per goal warm, 10-20 s
cold. In the app: the "Search goals" button on the move panel runs the loop for the side to move; the CLI's
`--trace` prints the narration. Limits: material goals only; the hopeless-line floor (down a piece beyond the
target on the settled count) is a stated heuristic; the no-engine material rule exists but is slow and strict;
horizon 6 plies.
History: goal-directed search with hand-written knowledge as the move generator is NSS 1958, CAPS-II 1974,
Pitrat 1977, PARADISE 1980 (plans as the generator: incomplete), Botvinnik's trajectories and Stilman's zones;
completeness with knowledge as a prior is the modern PUCT lesson, and threat-space / proof-number search (Allis
1994) is the exact tool for a boolean goal.

## 24. Sub-games as lemmas: coupling, the refutation as the next sketch, stored proofs, opportunity cost (Kaveh 2026-09-07)

Kaveh: define sub-games as an interplay of several pieces over a few moves; a piece has an opportunity cost across
the sub-games it takes part in; the endgame is a different regime because the vocabulary changes. Literature: sums
of games and temperature (Elkies 1996 for chess endgames; Berlekamp's hotstrat; Muller's decomposition search for Go),
Stilman's zones, Campbell-Berliner chunks, Barthelemy's interaction-graph fragility; and the 2025-26 theorem-proving
agents (DeepSeek-Prover-V2, Seed-Prover, Aristotle, AlphaProof Nexus): sketch, lemmas with holes, a prover per
lemma, a lemma pool keyed on the exact context, the verifier's error as the next sketch. The disanalogy is the
point: a lemma is reused for free, a piece is consumed -- linear logic, not classical.

`subgames.py`. A LEMMA = a target for one side with status, value, PARTICIPANTS (the ledger's attackers and
defenders, what a disqualified piece exposes, the joiner, the piece to remove; after a search the pieces that move
inside the proof) and read footprint (the store records everything read inside a block: `memo.recording()`).
Components = union-find over shared pieces. MEASURED (OBS, 3 games, 192 positions): 1.3-1.7 components per
position, the biggest holding most lemmas; 63-87% of captures made by a piece in >=2 lemmas; endings have almost
no material lemma. The disjunctive sum is the exception, so opportunity cost is accounted, not assumed away.
Built in the order agreed: (1) REFINEMENT (`refutations`, `diagnose`, `secure_prior`, `refine`, `search_goal`):
a refuted plan move's reply is read in ledger terms ("takes my queen on d1", "my knight on f6 now falls"), the
counter-goal's moves (the planner's defend plans, win the attacker, step aside, add a defender) become the root
prior, the horizon grows by the preparation's two plies, the search runs again; the chain is reported. (2) PROOF
STORE: every proof is put in the concept store exactly (all squares + castling / en-passant predicates: a
transposition is served as it stands) and as a SKETCH on the proof's footprint (the target's attack reads, our
moves with pin lines and king squares, the replies that needed work), served as the prior and re-verified by a
bounded search, never believed. (3) THE RESOURCE LEDGER (`holdings`): for every piece, the lemmas that flip when
it is removed -- our attack that stops working, their attack that starts working, the mate in one that appears --
each with its loss; `cost_of_move` replays the mover's holdings on the board after the move (a move that keeps
guarding, or captures the lemma's own attacker, breaks nothing). Plans carry the cost of their first move and the
prose says it ("But the knight on f6 keeps off Qxh7 (mate): not playable"); plans are ordered by net value with
unplayable ones last; "What holds for Black: the knight on f6 keeps off Qxh7 (mate); the pawn on b4 holds the
rook on a4 (5)" is a line of the long render and the app panel. OBS: 2.5 pieces per position hold 2.8 lemmas;
the move played broke a holding in 15/192 positions (mean cost 1.3, never a mate) in these three games.
Limits: lemmas are material and mate-in-one only (the positional vocabulary is still the missing regime for
endings); holdings are one level deep by the removal test; the store's sketch key is a predictor, soundness is the
re-verification; refinement runs one round. MEASURED (`subgames deep`, OBS 2026-09-07): holdings by search
(remove the piece, prove the lemma with and without it) against the static ledger on 91 piece-lemma pairs: 13
agree, 3 search-only of which 2 are search artifacts and 1 a genuine two-link chain (the f3 queen holds f4
through e3), 12 static-only where the judge fails in lopsided removal positions; 3.4-4.2 s per search against 45
ms per position statically. Deeper holdings are a chain problem for the static ledger, not a search problem.

## 25. The board as a sum of sub-games, when it earns it (Kaveh 2026-09-07, the rook ending)

Kaveh: "the rook is out of commission -- leave that side as it is; on the other side I am a pawn up and I know I
win; the king cannot leave because the pawn would queen; every piece has a job in its graph neighbourhood." The
design questions he set before building: decomposition is never the first step and never an assumption -- the
default is one game and a split is an inference that must be earned (a frozen region, a solved region, or a
coupling graph with more than one component); the unit is the lemma, and coupling decides what is valued
jointly (an overload is the coupling of two lemmas through a shared defender, not a sub-game); the boundary is a
light-cone statement -- a piece's cone is the squares it can reach safely within the horizon, shrunk by what it
holds and the races it must win, so the job binds the piece and the bound piece cannot reach the other region.
`regime.py`: `cone` (safe reach as a set, kings by legality), `race` (the rule of the square by computation:
passed pawn's steps, the double step counted, against the fastest catcher's cone, with the tempo of the side to
move), `promotion_lemmas` (a runner is a lemma worth a queen less a pawn; participants = the pawn, the catchers
in time, the pieces covering its path), trapped-piece lemmas (a piece with no safe square and no capture that
counts is out of play), `bound_cone` (first steps that break a holding worth a pawn or more are pruned; a catcher
may not leave the square), `coupling` (lemmas couple when they share a piece that must MOVE for one of them, when
one's target is the other's participant, or when a mover's bound cone reaches the other's participants; holders
shared between regions do not couple), `frozen` (the local fixpoint: no move by either side inside the region
gains, one reply deep with the ledger's attackers' test, two null moves deep; the null moves are the tempo
reservoir; regions with a runner or a king are never constants), `abstracted` + `tablebase_verdict` (the live
remainder with its own participants, shared pieces kept, probed in the syzygy tables when five men or fewer; a
single game of five men or fewer is probed as it stands), `narrate`. The rook ending after ...Kxh5: "The
queenside: locked (the white rook on a4 is out of play). Waiting moves: White 0, Black 5. White: Rxa5+, bxa5, b5
loses; Black: Rc8 ... b5 loses. Leave it as it is. The kingside: the live game (the black rook on c4 serves both
sides) -- with the locked side set aside it is 4 men, tablebase: loss for White (the black pawn on f6, 5 steps to
queen, is stopped only by the white king on g3: it cannot leave). Verdict: loss for White; engine -4.2 agrees."
The middlegame position of tPVi7fZF: one game, nothing said. K+2P v K: "4 men, tablebase: win for White". In the
long render and the app panel as "Sub-games". Checks: the split is stamped with its horizon; the engine judges the
whole; the tablebase abstraction is claimed only when a region is frozen. Limits: the fixpoint is material (a
plan that regroups inside a locked region is not seen); zugzwang between regions is counted only as the
reservoir, not played out; "joinable" lemmas still ignore what the joiner's departure exposes (the b6 pawn
'joining' against a4 by b5 frees the rook: the fixpoint prices it as losing, the lemma text does not).

## 26. Means-ends planning over needs (Kaveh 2026-09-07, the pinned knight and the second attacker)

Kaveh: "the need to get the knight to that square becomes its own goal with its own blockers; keep going down
until I find an answer or not; each line gets a value; I compare; a directed search over the things I need."
`needs.py`: an AND/OR tree over NEEDS (win, defend, reach a square, make a square safe, remove, deflect, block),
each with WAYS (a move now, or sub-needs it requires), every line priced (a piece that falls where it lands, a
capture that loses, holdings broken, the king's shelter or a pawn weakness created, tempo) and compared at the
node where alternatives split; give-up = no way, no way that pays, the depth bound (4) or the budget (80 needs).
Base cases: the planner's derive and reach; reach's failing precondition (the first unsafe square on the geometric
path) becomes 'make it safe', whose coverers (pieces without which the square would be safe; a pinned piece is
not one) are deflected (a lure on a square the coverer attacks, whose taking uncovers the square; poisoned lures
are rejected, declinable ones cost half a pawn; the expected reply is played through), removed (a win need) or
blocked (a reach onto the line). The RACE is judged at the moment the threat is real: the opponent's fastest
defence against the free moves they get, one per preparatory step less the replies our lures forced. The
VERIFIER is the goal search with the line's first move as prior. The IDEA LOOP (`ideas_for`): for a piece with a
reason to move (attacked, undeveloped), each quiet destination, the targets it newly attacks or makes fall, each
planned from the position after the move. Bb5 in the Ruy position: "Win the knight on c6: Nxc6 -- add the knight
on d4 as an attacker. Bring the knight to d4: Nxd4 -- d4 on the way is covered. Make d4 safe: deflect the pawn on
e5: d4 (then exd4), a lure on d4. Gain 1, cost 0.5. Their answer: a6 in 1, with 1 free move: they are in time";
the goal search agrees (given up). With c5 and e5 both covering d4: "given up (no way that pays)". Two ledger
bugs found by it and fixed: the flip cap kept the SMALLEST attacker value that flips (pawns only; now the
largest), and a joiner's path counted a square where it can be traded off as safe (an even trade is no arrival;
reach.py fixed too). Costs: 0.0-1.5 s per position for three ideas. In the long render and the app as "Idea for
<side>". GPS (Newell and Simon 1959) with a prover underneath; PARADISE had the patterns and no prover. Not built:
the motif memory (proofs abstracted over placement, roles, the cast prior), second-level deflection (attack what
the coverer holds), the interference/decoy families, a shelter cost finer than one pawn.

## 27. The agenda, the standing threat, and learning sub-goals (Kaveh 2026-09-07)

Three requests in one evening. (1) "When there is no threat, how do I create one, and what are the sub-goals on the
way? I do not want to program 'develop' or 'take the centre'." `needs.agenda`: the need tree for every enemy
target; the trees that give up name the precondition they died on (`frontier`); those preconditions, ranked by the
gains of the trees they block, are the sub-goals, with no rule behind them. Two planner additions made the agenda
honest: a target that can step away is not winnable by adding attackers (it is PINNED first: `pin_squares`, a
slider of ours on the line through the target to the enemy king or a dearer piece, then the win tree on the board
after the pin: "Win the knight on c6: pin it first (bishop on b5) ... a6 in 1, with 2 free moves: they are in
time"); a reach with no geometric path is 'unreachable', not a sub-goal. (2) "When I create a threat, what must
they do? If they do something else, take the winning move at once." `threats.py`: a threat = a goal our move makes
forced against all but a few replies; the goal search's AND node after our move already sorts the replies into the
ones the judge settled (punished: the proof's next move is stored) and the ones that held (the DEFENCES, described
in ledger terms: defends f7, blocks the line, the king steps away, takes the attacker, counters with check); mate
in one by enumeration. "Qh5 threatens mate (Qxf7#): they must Nh6, Ke7, Qe7, Qf6, Qg5 ...; Nf6 does not: Qxf7#
executes." The reply is judged against the standing threat in the CLI long render and the app. (3) "How do the
sub-goals get learned?" `mine.py`: the atoms stay the rules'; what is learned is which recurring intermediate
needs pay. Every node of every need tree over games becomes a TEMPLATE (kind, the mover's and target's piece
types, the square's class: centre / their camp / by their king / own camp; never coordinates); the root's outcome
is whether its target is taken within k plies of the engine's line; support and lift over executed-versus-not
give the library, credited by the root's gain. The first number is the agenda's own reach before any learning:
of the engine's quiet best moves, how many serve an agenda sub-goal (a step of a way in its tree, or the sub-goal
met after the move). See the JOURNAL for the run.

## 28. The terminal goal, the sure gain, and the threat ledger (Kaveh 2026-09-07, evening)

Three generic repairs after the first puzzle gauge (47/100), each stated in the rules' terms: the TERMINAL GOAL
(`GoalSearch(target=None)`: win the game by `board.outcome()`, whatever ends the game in the variant; no judge; the
proof-number pricing is the only ordering), the SURE GAIN one reply deep (`rank.sure_gain`: min over replies of
what the ledger then gives less what the reply took -- forks, discovered attacks and checks with a threat, no
pattern named), promotion races as scores, rescues and saves. Same 100 puzzles: 47 -> 71 solved, 18 -> 9 silent,
mates 100%, sacrifice 0 -> 86%. Kaveh's definition of a threat -- a free move that improves the position; big when
it wins a piece with no payback; the goal of the game is a threat with no adequate defence -- is the same number:
`threats.free_move_threats` (the null-move scan by the ledger) and `threat_ledger` give both sides' threats at every
position in one currency ("Black threatens: Nxh5 (+9, big); White would, with a free move: Qxf7# (mate)"); the
standing threat (§27) is the proof-based special case with the defences listed. In the long render and the app.
Not done: the created/parried/executed bookkeeping across the game for the ledger version (the proof-based one
has it), and the two-move combinations at 1600+ that one reply deep does not reach.

## 29. Tactics by reasoning, measured on puzzles (Kaveh 2026-09-07/08, "go autonomous until morning")

`rank.py` scores every legal move by what the reasoning says it does, with no engine and no Maia in the ranking
(the engine judges leaves inside proofs only), and `gauge_rank.py` measures it on random Lichess puzzles: the
position after the setup move, the solution's first move as the reference, solved / named / silent, by theme,
by the kind of the solution move, by rating. Sources, each stated in the rules' and the ledger's terms: the
terminal goal (win the game by `board.outcome()`; §28); the SURE GAIN one reply deep and the COMBINATION two
forcing moves deep (an alpha-beta over captures, checks and attacks on a piece worth 3 or more, all replies, the
ledger's best capture at the leaves as the quiescence, a node cap) -- forks, discovered attacks, skewers,
deflections, promotions, captures with a reply all come out of these two with nothing named; proven material
goals (the target taken and materially ahead; the dearest enemy pieces always tried); ideas and plans that win
their race, discounted by length; the agenda's sub-goals at a quarter; rescues; SAVES as an additive term (not
when our move gives check: a piece held only because the enemy is in check is not saved); minus the holdings cost
for the static sources only. Same 100 puzzles rated <=1800: solved 47 -> 71 -> 88, silent 18 -> 9 -> 6; mates 100%,
forks 73%, sacrifices 100%. 100 puzzles rated 1200-2200: 77 solved before the last pricing pass (1200-1599 91%,
1600-1999 72%, 2000-2399 43%; mate and fork 100%, pin 91%, sacrifice 82%; advanced pawns 50%, promotion 33%,
very long 44%). What remains is quiet technique (Kf2, b5, d7) and combinations three of our moves deep. Costs:
7-10 s per position, dominated by the goal searches. The numbers are single runs on 100 puzzles: directional.
Deepened 2026-09-08: the combination search runs three of our moves deep, guided by the reasoning (the ledger's
order for our candidates, replies complete, a transposition table, iterative deepening with the node cap as the
budget). 500 random puzzles rated <=2200: 84% solved outright, 87% named, 9% silent; every mate class 97-100%;
forks, discovered attacks, deflections, skewers, attractions, double checks, trapped pieces above 90%; sacrifices
85%; the misses are pawn endings, zugzwang and quiet technique (the trajectory system's), and 2000+ combinations
beyond three of our moves. See the JOURNAL entry for the full table.

## 30. Explanation by contrast, with the human move at the player's level (Kaveh 2026-09-08)

Kaveh: "a knight hop that is good in another position is bad here because it ALLOWS the opponent a tactic; that
the tactic exists is not interesting, that this move allowed it is. Compare the move played with the best move,
and the best move with the most common human move at my level (1800)." `contrast.py`: two moves from the same
position, both resulting positions through the same analysis, and the difference as the explanation -- the
opponent's threats after A that B does not allow ("Nf6 allows Qxf7#, mate; g6 does not"), the ones A prevents,
our threats A gives up or creates (the partner's own move excluded: a threat the partner executed is not 'given
up'), the material each nets one reply deep ("Nxc5 sure gain 1; Qxe5 sure gain 4"), the holdings each breaks,
the structure each changes, the engine's number for each; only threats worth two pawns or a mate are named; when
nothing the ledger names differs, it says the difference is positional. Partners: the move played against the
best when they differ; the best against Maia-1800's top move when that differs. In the CLI long render ("By
contrast [played against best] ...") and the app ("by contrast"). Not done: the reply that refutes a move that
creates a threat (Qh5 creates Qxf7#, and g6 refutes it with tempo) as its own sentence.


## 31. The plan layer: the interaction graph, focus, and learning on the knowledge (Kaveh 2026-09-09)

The morning's discussion fixed the programme and three design decisions:

1. **The interaction graph is the one object.** Every module's facts are readings of one graph per position, computed
   by the four generic operators (the exchange on a square, counterfactual removal, the free move, shortest paths on
   the move graph). Values and judgment do NOT come from the graph: they are the planner's, learned from outcomes.
   The hand-written residue is the vocabulary of names (pin, fork, skewer), the planner's need kinds (provably edge
   edits of the graph), two geometric shortcuts (the rule of the square, the king zone) and the constants.
   `graph.py`: 43 node answers, 17 pair relations, 39 position indicators; zones = lemma components with a
   temperature (the combinatorial-game answer to 'which sub-game first': the hottest).
2. **Approach B is a transformer whose inputs are the rules' answers and whose output is a sum of named senses.**
   Precedent: KataGo's rule-computed ladder planes, Giraffe's attack maps, Graphormer's node and edge encodings,
   Leela's learned pair bias (here computed from the rules), Backpack language models (output = weighted sum of
   sense vectors, so a representation is decomposable by construction). Approach A is the same network with the
   answers zeroed (identity and square only) -- one code path, a switch, and the required control for free.
   The trajectory loss calibrates d(phi_t -> phi_{t+k}) to k plies against cross-game negatives and never names
   pawns, so the asymmetry test (reverse distance after pawn moves vs piece moves) is honest.
3. **Three experts and a gate blind to the board.** A, B and the rules ranking C score every legal move; the gate sees
   only named context (the position indicators + whether C speaks) so its weights mean 'trust X here'. Before the
   gate: each expert alone, the per-position oracle (the bound on any gate), a fixed blend. After: the scenario
   table (mean weights by indicator).

Measured on day one (OBS): focus -- with two zones or more the engine's best move touches the hottest zone 63%
(largest 60%, random 45%, previous best move's zone 48%, any zone 80%); persistence 69-71%; 29% of quiet best
moves touch no sub-game. Item 6 is sized by this: a zone-first ORDER with a budget (never a filter) in `pursue`,
behind `score_moves(zone_first=True)`; the cross-ply memory helps the static facts only (the free-move ledger's
reply loops are scratch boards), so the incremental gain went into the search itself: the goal search carries the
parent's exchange ledger and recomputes only where the move's squares touch a footprint (3.4x on a quiet
middlegame, same scores). `track.py` is the cross-ply ledger (executed / created / parried / left; the action
stays or moves).

Day-one results of items 3-5 (OBS, single runs; JOURNAL 2026-09-09 16:55-18:55): the trajectory is in the knowledge
(ridge to the regime 20 plies ahead R2 0.45 against persistence -0.13, though the 15 regime indicators alone give
0.42); B beats A on one code path (33.0% against 27.2% top-1, evaluation R2 0.80 against 0.64); the rules-only
variant is the best on quiet moves (26.3%); the geometry finds the arrow in the rules' order (reverse > forward after
captures 69%, pawn moves 56%, piece moves 54%) once the objective pushes the reverse of time apart and the IQE's
scale can move; the three experts are complementary (oracle 56%, with Maia 67%), a fixed blend takes 40% (45%
with Maia), the context gate does not beat it at 560 fitting rows. Open: the geometry against the heads (rank 6
when the geometry forms), the distance-based move scoring, more expert rows.


## 32. The planning layer: the quantized future, its value and sharpness (Kaveh 2026-09-11)

"A future board embedding distribution predicted from the current state, quantized or interpretable; it has to
predict the sharpness and the value, not the details; a signal from the tactics to the planner and back."

The future is a DISTRIBUTION OVER NAMED REGIME CODES (`regimes.py`): cells of a quantizer on the graph's own
indicators, described by their centroids, valued by the mover's result rate in them. The knowledge net predicts
the distribution at +10 and +20 plies, the engine's evaluation there (given the evaluation now), and the window
sharpness (a big threat within ten plies; the engine's best moves forcing). `planner.py` reads off the expected
value and sharpness, the entropy, and the STEERING TARGETS: reachable futures whose value beats the expectation,
turned into NEEDS by their named differences from the present -- the vocabulary is which indicator is which need
kind; the values come from the codes. Tactics -> planner: the ledgers, zones and threats are the prediction's
inputs and define the sharp codes. Planner -> tactics: the needs go to the goal search, the terminal search and
the race (`pursue`); the expected sharpness sets the search budget (`gauge_budget.py` measures it).
Measured first (OBS): a big threat within ten plies is predictable at AUC 0.79 from the present graph (0.72 from
the regime alone, 0.58 from the present threat); the code at +20 at 13.8% top-1 against 9.4% persistence, NLL
3.05 against 4.09 marginal; the details (which ply, which threat) are not predictable, as expected.
Trained and measured (OBS, `kn_future4`, 4000 held-out positions): the code at +20 top-1 13.8% (persistence
9.4%), NLL 3.05 (marginal 4.09); a big threat within ten plies AUC 0.80 (persistence 0.58); the evaluation at
+10 R2 0.58 (persistence 0.56). The budget gate: on 300 game positions all four budgets give the same top move
(the ranking is not budget-limited there); the gate's sorting is real (forcing best moves 36% in 'deep', 13% in
'shallow'); its value on puzzle-like positions is not measured yet.


## 33. Threats, entropy and transition values as the planner's own (Kaveh 2026-09-15)

A THREAT is the loss a cheap observer sees under the null move; the level is the observer's cheapness times the
magnitude; the loss classes are the rules' irreversibles and the terminal condition. Measured: under a threat of
a piece or more the engine's best move lies in the response set 84-93% of the time and the set is a quarter to
a third of the legal moves; under a pawn threat 52%, and counter-threats matter there (17%). Built: the gate in
the ranking (the deep searches confined to the response set above the smallest piece; the second-level threat
as a penalty), flag-gated.
LOW ENTROPY: among lines within a margin of the best value, prefer the one the opponent can least deflect (the
width: how many replies hold us within a pawn of the minimum, free from the ledger's reply nets) and the one
that costs least to prove (the proof's size, not the line's length).
TRANSITION VALUES: the planner does not know that trading when ahead is good, because the outcomes say it is
not (Lichess, every band, within +-0.02; Kaveh's own games the reverse of the textbook, +-0.13); so the value
of moving a regime coordinate is a table from outcomes, conditional on material and level or evaluation, and
the steering supports or avoids targets by it. Nothing chess-specific is written down; the sign is the data's.
