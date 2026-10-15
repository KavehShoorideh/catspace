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

## 34. Per-piece futures and action concepts on pieces (Kaveh 2026-09-18)

"I could plan to trade my light-squared bishop for their kingside knight ... either remembering or calculating that
the line I am heading for tends to have my bishop very inactive (so I want to trade it) and my opponent's
kingside knight very active (so I want to get rid of it)." An action concept is an OPERATOR (trade, remove) on a
PIECE named by identity, kind and attributes (square colour, wing: board coordinates, not rules), justified by
the piece's predicted FUTURE and priced by OUTCOMES.
- Identity: `pieces.py` follows each piece from its origin through the game (castling, en passant, promotion);
  its fate within a window: kept, exchanged (it captured and was captured), lost, promoted.
- Labels: `dataset.py piece-futures` reads, for every piece in the corpus, its own node answers 10 and 20 plies on
  and its fate -- no new graphs, the corpus holds every ply.
- Calculating: the knowledge network's per-piece head predicts each piece's activity answers at +10/+20 and its
  fate, from the present graph; compared per kind with persistence (the piece stays as it is).
- Remembering / pricing: `piece_values.py` -- the mover's result when a piece like it (side, kind, activity,
  king zone, evaluation band) was traded off with the material even, minus when it stayed; game-clustered se.
- The planner names an idea only when the future stands out against the kind in the phase AND the outcomes say
  the trade paid beyond its se; the ranking then serves the moves that carry it out (trades it, offers the
  trade, takes it, goes for it).
Next, planned with Kaveh's go: PATTERNS THAT ALLOW ATTACKS -- pairs/triples of pieces with their graph relations
(two knights side by side on a line a rook reaches, facing pawn chains), mined against the events they allow
(a piece lost, a threat, a passed pawn) by lift; the armed tactics of the old plan (M7) in the graph's vocabulary.

## 35. Invariant tokens from the tablebase, rooted in the rules (Kaveh 2026-09-19)

"Learn invariant tokens that help predict position value, then root them in the graph operations" -- and hand-roll
nothing beyond movement, attack, capture, promotion and check. The REGION (regions.py): a piece's safe closure
through unattacked squares, the opponent passing, and the contested frontier beyond it with its attackers (a
sacrifice is a contested square whose attacker's duties are worth more: a deeper search's question, never
excluded). Relations between regions are ENUMERATED, never named. The tablebase (3-5 men) is the teacher
(endgames.py). The token model (tokens.py): squares as tokens, the rules as the only relations, a binary
bottleneck, the result and its distance from the tokens alone, coherence across a reply. Rooting (root_tokens.py):
the smallest tree over the relations that predicts each token; faithfulness by re-predicting the result from the
rooted tokens; the explanation of two king moves = the tokens that differ, rooted. First numbers: tokens 0.896 vs
plain 0.875 vs majority 0.771; rooted tokens lose on test (0.735) and GAIN on held-out materials (0.709 vs
0.653); Kaveh's Kf8/Kd8 separated by one token. Open: rooting per material and on sibling pairs; the advised
player (Maia's policy reweighted toward moves keeping an advised invariant) as the test of sufficiency.

## 36. Concepts as tokens; the concept bank loop (Kaveh 2026-09-20)

The framework: a rules-based layer produces concepts (the registry, concepts.py: programs over the rules'
vocabulary, versioned, append-only); a shallow transformer over the board's pieces AND the concept instances
decides, with the move score a sum of per-token contributions (concept_net.py); attention over the concepts is
the learned weighting; a generative proposer adds concepts against the residual, admitted only when they make
the decision better on held-out data; every explanation cites the registry version. Canonicality lives in the
concepts (code), not in the weights. First numbers on the tablebase: with concepts 0.928 / 0.913 (test / unseen
materials), board only 0.929 / 0.895, random 0.774; explanations 6 tokens; the top concept is causal in a third
of decisions. Round zero earns transfer and explanation, not decision quality. Next: intervention consistency as
a training constraint; proposal round one against the residual (queen endings, pawn races, KBNvK).

## 37. The lemma loop: search once, remember the result as a lemma, search less (Kaveh 2026-09-21)

> **2026-10-04 (Kaveh):** the "NOT Lean/Coq" line below is reversed. Lean 4 is in -- not for search but for
> targets and pruning: a proved lemma with individually evaluable hypotheses gives the conclusion without search, and
> the one failing hypothesis is the search target. Kernel imported (chess-lean), scaffold in `lean/`; see JOURNAL
> 2026-10-04 and `lean/README.md`.

THE END STATE. A loop: the searchers (our ledger and goal search, Stockfish, the tablebase) explore the space;
where our fast evaluation falls short of what search found, or where search was expensive, we PROPOSE a lemma;
we TEST it against the corpus, PROVE it where it is provable, and ADMIT it to a versioned library only if it
pays; the searchers then use it and spend fewer nodes. The prize is not a better chess number. It is machinery
that transfers to domains where rolling the world model forward is expensive: a lemma is a cached, conditioned
result of rollout, and nothing in the procedure mentions chess.

WHAT A LEMMA IS HERE. A triple: a PATTERN over the rules' vocabulary (concepts.py instances), a CLAIM about what
search would have found, and EVIDENCE. Two claim kinds so far:
  - refutation: "this family of the opponent's replies is answered by this move" -> the family is searched once
    (lemmas.py, the discovered-check schema; the closed angles of angles.py are a second schema, unwired).
  - value: "when this pattern holds, the subtree's value is V (or bounded by V)" -> the subtree is not expanded.
    The relative pin is the worked example: "a piece between an enemy slider and a dearer own piece does not
    move" is the cheap over-general lemma; the exception ("unless a move of it gives check or takes the pinner")
    is what the residual should teach us. Discovering that refinement without writing it is the test of the loop.
Status is tracked like a proof library's: PROVED (a finite geometric claim, settled by exhaustion or Z3) or
CONJECTURED (survived testing, carrying its measured precision). The kernel (proof.py) already separates proven
replies from priced holes; a mined lemma enters as a hole with a number on it.

WHAT WE TAKE FROM THE MATHEMATICS SOCIETY (2026-09-21 review; do not reinvent):
  - THEORY EXPLORATION (QuickSpec, Hipster/HipSpec): the loop's skeleton -- enumerate candidate statements from a
    typed grammar, filter by cheap testing, prove the survivors, add to the library; Hipster's "proof mode" (find
    the lemma that unsticks THIS goal) is our residual-driven proposal.
  - LEMMA MINING (Mizar/Flyspeck, Kaliszyk & Urban): only a tiny fraction of proved statements deserve a name;
    rank candidates by how often they would serve later proofs. Our ranking: applicability x nodes saved.
  - CUT INTRODUCTION: the field's metric for a lemma is PROOF-SIZE REDUCTION -- the same question as ours in
    another vocabulary. It is the headline number, not an afterthought.
  - PREMISE SELECTION (Sledgehammer, Magnushammer): "which of the library's lemmas apply to this goal" is our
    attention over concept tokens; read before tuning it further.
  - LCF KERNEL: taken already (proof.py).
  - Packages, all installable here: z3-solver (promote geometric lemmas from conjectured to proved), python-sat
    (minimal unsatisfiable subsets = the smallest pattern that explains an outcome), clingo (declarative rules and
    subset-minimal models) if the pattern language outgrows hand-written matching. NOT Lean/Coq: encoding chess
    would be its own project and buys no search.

WHERE WE ARE (2026-09-21).
  Rules layer: the interaction graph (43 node answers, 17 pair relations, 39 position indicators), the ledger and
    threats, regions (safe closure + contested frontier with attackers), angles of attack on the king, piece
    identity and per-piece futures.
  Concepts: a versioned, append-only registry (concepts.py): v0 = 118 concepts enumerated from the rules'
    vocabulary incl. the line predicates; v1 adds discovery(A, B). Instances are tokens.
  Learner: concept_net.py -- a shallow transformer over the board's pieces AND the concept instances, the move
    score an exact sum of per-token contributions, trained on the tablebase's partition of moves (keeps the
    result / loses it). Rounds, OBS (test / held-out materials): round 0 concepts 0.928 / 0.913; board only
    0.929 / 0.895; round 1 (+ move consequences) 0.959 / 0.960; round 2 (+ discovery) 0.962 / 0.963; ten linear
    consequences alone 0.916 / 0.926; random legal 0.774 / 0.772. Explanations 3.2 terms; removing the top
    concept changes the choice in 22% / 14%.
  Search: the goal search (proof numbers, OR/AND), the ledger's one-reply minimum, family lemmas represented once,
    proof terms and a rules-only kernel that refuses forgeries.
  Canonical selection: canon.py -- a deterministic L1 path over the fixed vocabulary; cross-run identity and
    bootstrap stability hold; 0.832 test / 0.579 held (the slot-named vocabulary does not transfer).
  Data: 100,000 tablebase positions over 25 materials with every legal move's result; Kaveh's 69k game positions;
    60k Lichess games; Lichess puzzles; STS.
  Honest negatives: the one lemma schema cost time on puzzles (80 solved at 16.8 s vs 81 at 14.6 s, before the
    geometric gate); the concept tokens contribute at the 0.01 level beside the consequence terms; Kaveh's
    KRPvKR position is still played wrong.

THE GAP, AND THE PHASES.
  P1 INSTRUMENTATION (2 d). One node counter across the ledger, the goal search and the engine; traces that emit
    per node: the concept instances present, the subtree's value, the nodes it cost. Baseline deliverable: the
    NODES-TO-VALUE CURVE -- how many nodes our search needs to come within epsilon of the deep value, per
    material and per position class. Nothing today measures this.
  P2 LEMMA SCHEMA + LIBRARY (2 d). Generalize Family into Lemma(pattern, claim, evidence, status, provenance,
    stats); a library file with versions; re-express the discovered-check schema and the closed-angle schema in
    it; the kernel reports proved / conjectured per application. Re-run the puzzle comparison with the gate.
  P3 THE MINER (3 d). From the traces: cluster subtrees that were expensive AND whose result is constant over a
    class; extract the smallest pattern that predicts the result (PySAT minimal subsets, the rooting machinery);
    emit candidates with support and precision. Validation target: rediscover the pin lemma and its relative-pin
    exception without them being written.
  P4 THE PROVER (2 d). Z3 (or exhaustion) for the finite geometric claims; promote conjecture -> proved; the
    library records which is which and the kernel honours it.
  P5 THE LOOP (2 d). Residual -> proposals (grammar enumeration + an LLM against the residual clusters) -> test ->
    admit -> re-measure; one journal entry per round with the numbers below; the registry and library versioned
    per round; explanations cite the version.
  P6 THE PRODUCT (1 d). Lemmas in the analyst's prose: "no search needed here: <lemma>", and the budget the
    lemma saved, so the user sees why the machine did not look.

PROGRESS, AND WHY IT IS THE SAME CURRENCY AS NODES (Kaveh, amendment 2026-09-21). The headline is PROGRESS PER
NODE. Progress needs a POTENTIAL over positions; a move's progress is the potential's change. Potential-based
shaping has the classical guarantee (Ng, Harada & Russell 1999) that it densifies a sparse reward WITHOUT
changing which policy is optimal -- so a lemma can be credited for progress in a game we go on to draw, while
the thing optimized is still the result. The potentials, in order of trust:
  1. the tablebase's distance to conversion (exact for <= 5 men: the whole endgame corpus);
  2. the reachability quasimetric already in the repo (reach_qm: R2 0.83 in plies, the direction of time 0.875)
     -- it failed as a NAVIGATION target and that is precisely the point: as a MEASURE it need only be monotone
     along real games, which it was;
  3. the regime codebook's values, which already come from outcomes, where neither of the first two reaches.
  The repo's scar (z_conditioned_field_two_evaluators) is that a learned field used as the search's TARGET is
  trivially gamed. Using it as a measure is the different purpose; the guard is that admission never rests on the
  potential -- a lemma is admitted on the result at a fixed node budget, and the potential only says where it helped.
  NO-PROGRESS LEMMAS ARE FIRST-CLASS: if no move raises the potential, the position is drawn by our lights, and a
  lemma that establishes "no progress is possible here" (a fortress, opposite-coloured bishops) saves more search
  than any winning one, and the tablebase validates it exactly.
  Consequences for the phases: P1's traces record the POTENTIAL at every node, not only the value; P3's clusters
  are "subtrees where progress stalled" as well as "subtrees that were expensive"; and the search over lemma
  applications ("which lemma raises the potential from here") is the same object as the planner's steering, so
  the planner and the prover share one mechanism.

THE NUMBERS, EVERY ROUND, EACH BESIDE ITS BASELINE.
  1. FIDELITY: the value with lemmas against the deep value (tablebase where exact, Stockfish at a fixed depth
     elsewhere); baseline = the same search without lemmas.
  2. SAVINGS: nodes to reach within epsilon of the deep value, with and without; baseline = without.
  3. DECISION AT A FIXED BUDGET: the share of positions where the chosen move keeps the result at N nodes, for a
     ladder of N; baselines = without lemmas, and a random legal move.
  4. INTERPRETABILITY: the explanation's length and the share of decisions whose top term is causal under
     intervention.
  5. LIBRARY HEALTH: how many lemmas, proved vs conjectured, each one's applicability and measured precision, and
     how many are ever used (the mining literature's warning: most named statements are never reused).
  Kaveh's rule stands: single runs are OBS, never verdicts; two seeds before any claim that turns on a small gap.

RISKS AND THEIR GUARDS.
  - A lemma that memorizes a position rather than a pattern: the pattern must be small, hold across materials, and
    survive the held-out-material split and the bootstrap.
  - A lemma that is wrong: status and precision are recorded; the kernel never reports a conjecture as proven.
  - A lemma that costs more than it saves: P1's node accounting makes that visible before admission (the puzzle
    result above is the warning).
  - The proposer drifting into chess knowledge: proposals are programs over the rules' vocabulary only, and the
    admission test is the decision, not plausibility.

## 38. The tower: frames from the rules, one piece at a time (Kaveh 2026-10-14)

**Status: PLAN (2026-10-14 10:05 PDT), to be built by a cheaper agent, side by side with the surprise loop -- not a
replacement.** Kaveh: "start from a king, and then ask, how can I mate a king? ... if I'm unable, I bring my strongest
piece in. If I'm successful, I bring their pieces in ... a binary search in the space to find the areas where it's edgy ...
those are the sharp positions ... the frames are not really absolute ... relational, like a sliding piece attacking this
from four squares away."

### 38.1 The idea, assessed

- The tower IS a frame. A floor is a set of positions described in the vocabulary (a condition); floor 0 is mate; floor
  n+1 is the positions from which some floor <= n is FORCED within k of our moves against every reply. So the tower is
  one conditional frame per floor (`<material>:if:<cond>`, `horizon:k`) whose measure is the floor number, and the
  claim "the measure falls within k against every reply, or the goal" is exactly `Ideas.Forced.forced_by_measure`
  (lean/ideas/Ideas/ForcedByMeasure.lean). Nothing new is needed to PLAY a tower: the chooser already plays frames.
- The exact search already speaks the tower's language: `goals.rs forced_goal(fen, literals, depth, budget)` asks
  "can we force, within `depth`, a position where these literals hold" -- a floor's literals ARE the goal of the next
  floor. `defence_moves` names what stops it. Nothing new is needed to CHECK a floor.
- What is new is the ORDER (Kaveh: the loop learns in whatever order surprise brings; here the easy frames come first
  by construction) and the GENERATOR of positions: not games, but the material ladder -- our strongest piece in when we
  cannot mate, their piece in when we always can -- and, within a material, the horizon k and the condition size.
  The sharp band (forced from some positions of the material, not from others) is where the conditions live; the
  separator that finds them exists (`explorer.separate`, `explorer.lemma`, the committee, CEGIS, held-out).
- Their pieces enter twice: as DEFENDERS (capture, interpose, flee, counter-threat, stalemate -- `defence_moves`)
  and as BLOCKERS of their own king (the back rank). "The queen alone cannot mate" holds only against a bare king.
- Where the tower is strong and weak: strong for the short window (tactics: every floor is exact, carries its own
  why, and a floor one defence short is an ARMED tactic with its blocker named -- memory `armed_tactics_parked`,
  `planner_tactic_tokens_2026_10_07`); weak for the long manoeuvre (the 15-move K+R v K drive is a loop, not a stack):
  regressing it position by position is the tablebase. The tower must regress over SETS described in words; where
  no word describes the set, that is a vocabulary gap to REPORT (the vocabulary is Kaveh's), never to patch, and the
  frames from play and watching fill the long window.
- Literature (from memory, not re-checked): goal regression (Waldinger 1977); Pitrat 1977 and Wilkins's PARADISE
  (1980) for chess combinations; Minton 1984 (forced wins learned from single examples; the utility problem, 1990);
  Allis's threat-space search (1994); Fawcett & Utgoff's Zenith (1992) -- new features by regressing the goal, which
  is how the vocabulary could propose its own words later. No prior work found deriving endgame strategy from the
  rules alone with machine-checked proofs (docs/LITERATURE.md; memory `related_work_reasoned_discovery`).

### 38.2 Definitions (extend catspace/research/docs/GLOSSARY.md, section "The lemma-discovery vocabulary", no new file)

- FLOOR: a conditional frame `<material>:if:<cond>` with `horizon:k`; its number n is the measure; its claim: from
  every position of the floor some floor < n (or mate) is forced within k of our moves against every reply.
- TOWER: the floors of one material, numbered from 0 (mate); a frame dict in the loop's own format.
- SHARP BAND: a material and horizon where the goal is forced from some drawn positions and not from others.
- A floor is a LEMMA when its claim is proved in Lean or checked exactly on every position it covers (<= 4 men:
  enumerate every placement inside the condition); otherwise a CONJECTURE (checked on a sample and held out).

### 38.3 Implementation (new module `catspace/incubator/analyst/discovery/tower.py`; nothing in the loop changes)

1. `ladder(start="KvK")`: the binary search over material. From a material M and horizon k, draw N positions of M
   (us to move; `explorer.random_positions`), ask `forced_goal` for mate (floor 0) within k. All forced -> add
   THEIR piece (weakest first: P, N, B, R, Q); none forced -> add OUR piece (strongest first: Q, R, B, N, P); mixed ->
   M at k is sharp: build its floors. Horizons k = 1, 2, 3 (then 4) per material. Print the ladder as OBS lines.
   Stay <= 5 men (our tablebases end there: the test oracle).
2. `floor(M, k, n)`: the positions where floor <= n-1 is forced within k (right rows) vs not (wrong rows), both drawn
   inside M; `explorer.separate` / `explorer.lemma` find the condition in words (layer 0 first: no named square;
   relative words -- offset / beyond / next / xray / attackers / defenders / see / pinned / between / nearer -- are the
   "sliding piece attacking from four squares away"); the condition must keep every wrong row out and the most right
   rows in; refine as the explorer does; the committee and CEGIS as the explorer does; held out.
   A floor's literals are then the goal of floor n+1 (`forced_goal` with those literals plus "nothing of ours lost").
3. `canonical(fen)`: the mirror lemma (lean Ideas/Mirror.lean; `explorer.mirror_fen`) and translations
   (`explorer.translations`) so one arrangement is counted once.
4. `defences(M, k, floor)`: for the positions of a floor that stop being forced when THEIR piece is added, the
   `defence_moves` labels -> which defence (capture / interpose / flee / counter-threat / stalemate) -- the floor's
   blocker, named by a word (`pinned_theirs`, `attackers`, `see`, `xray`, `trapped_theirs` ...) when a word fits; when no
   word separates, print `OBS VOCABULARY GAP` with ten example positions and stop that branch.
5. `status(floor)`: <= 4 men -> enumerate every placement inside the condition (all cores, `exchange_law._workers`
   pattern) and check each with `forced_goal`: all forced -> LEMMA; else the failures become new wrong rows (refine
   once, then CONJECTURE). 5 men -> sample 600 + held out 300 -> CONJECTURE.
6. Output: `data/derived/tower_<M>.json` (the frames dict + per-floor evidence: positions drawn, forced/not, the
   defences, status, seconds) through `paths.derived`; `python -m catspace.incubator.analyst.discovery.tower --material
   KQvK --horizons 1,2,3` runs one material; `--ladder` runs the search from K v K.
7. The ablation (Kaveh's metric: lemmas found + what each changed): `surprise_loop._game` on the SAME 100 starts (seeds
   5000-5099, band 1-256, the tablebase defending, CATSPACE_SCRATCH set, the mating frames off) with the tower's
   frames, with none, with the tower minus each floor; print mated / moves to mate / better / worse paired by seed.
8. Tests (root `tests/test_tower.py`): floor 0 of K+Q v K found as a condition that holds on mated positions only;
   `canonical` identifies a position with its mirror; a floor's claim re-checked by `forced_goal` on fresh draws;
   the back-rank mate (K+R v K+P+P+P, their pawns on f7 g7 h7, their king g8) reached by the tower with their pawns as
   blockers -- compare with the proved lemma in docs/LEAN_INDEX.md (`lean_index search back rank`).

### 38.4 Acceptance (directional, one run; numbers only from OBS lines)

- K+Q v K: a tower of >= 3 floors at k <= 3, floor 1 a LEMMA (exhaustive), in < 10 minutes on all cores.
- K+Q v K+P and K+Q v K+R: the ladder marks them sharp; the defences named for >= 1 floor each.
- K+R v K: the tower's frames against none on the same 100 starts, better/worse printed (the watched and play-only
  frames, 2026-10-14 JOURNAL, are the comparison: 89-95 mated of 100).
- Every vocabulary gap printed, none patched.

### 38.5 Rules for the builder

No AI trailers in commits; Lean is the source of truth (prove, never a Python sweep as proof; the finite check is a
check, not a proof); import, don't reinvent (reuse explorer / patterns / goals.rs; no new solver); all cores, no Pool
from stdin (`-m` entry points, spawn); short runs before long ones; numbers only as printed; bug-check negatives;
journal with `date` timestamps, directional; never rewrite dated docs; the vocabulary is Kaveh's -- gaps are reported.
Later, not now: probability where the tower branches too wide (the tactic-density model), Lean proofs per floor.

### 38.6 Deconstruction: a big frame as small frames, and the lift check (Kaveh 2026-10-14, phase 2)

Kaveh: "in a 10 man frame we can deconstruct into multiple partial overlapping or sequential smaller frames."
- SEQUENTIAL: a frame hands off to another at an event (a capture, a promotion, a floor reached) -- `Reach.bind`; a
  game is a path through small towers.
- OVERLAPPING: a position matches several small floors at once, each a partial pattern over a few pieces, the rest
  of the board ignored (the conditional frame's condition names only the pieces it needs).
- THE LIFT CHECK: a floor established for few men applies in a bigger position only if the other pieces supply no
  defence within k -- no capture, interposition, counter-threat, stalemate, no changed flight. Not a theorem in
  general (extra pieces block and defend), so an exact check in the full position: `forced_goal` with the floor's
  literals; when it fails, `defence_moves` names the interfering piece -- that piece is the next SEQUENTIAL frame
  (deal with it first). `lift(floor, fen)` -> holds / the named defence. Output: for the loop's own games (castled
  K+R+R v K+R with pawns, artifacts/experiments/castled_play_r1.json) how many positions some small floor lifts into,
  and which defences stop the lifts most often.
Built after §38.3-38.4 are accepted.
- THE RECURSION ON A NAMED PIECE (Kaveh: "who is the attacker? ... this bishop. What can we do about it? Who is the
  defender? ... this knight"): the roles are generic words (`attackers`, `defenders`, `between`, `pinned`, `see`,
  `xray`); the binding is the piece names (O2-O5, P2-P5); the answers are a fixed menu of goals in the same words --
  capture it, deflect it (its `moves` off the line), pin it (`pinned_theirs`), overload it (two things to defend),
  block it (`between`) -- each a `forced_goal` literal, so each is a new floor whose goal is "that defence is gone",
  and the ladder recurses on it.

## 39. The reasoning ledger: questions, findings, bounds (Kaveh 2026-10-14)

**Status: DESIGN (2026-10-14 14:06 PDT), nothing built; to be handed to a builder on Kaveh's go.** Kaveh: "analyze the
safety of each piece ... each question has to be answered and each question affects some decisions ... track evidence ...
those proofs become new things upon which we reason ... the moment we get rid of the pin, immediately we should know that the
pawn can move ... roll them forward in time ... sometimes we might need to backtrack ... keep asking these questions until we
reach a point where the cost is known ... and we can prune based on that value ... if we don't know the cost, we can
estimate it." The first example: 5k1r/1p1n1p2/p1qb1p1p/4p3/2rP2PP/2PQ1PB1/PP2NK2/1R5R b -- the e5 pawn is pinned because
the only guard of the bishop behind it (the queen) is the only guard of the rook on c4, and from d6 it no longer sees c4.

### 39.1 The objects

1. FACTS -- exact, cheap, recomputed every move: the vocabulary's words on named pieces (attacks, lines, `between`,
   guards, `see`, `without`, `job`). Never stored as findings.
2. QUESTIONS -- typed, about named pieces, each from the CRITICAL-QUESTION MENU of the claim it challenges (Walton's
   argument schemes; the menu comes from the ways the rules let a capture or a guard fail):
   - "X is safe" (the entry point, asked of EVERY piece, ours and theirs: the threats to us, our threats): what does the
     exchange on X give? for each recapturer: does it have another job? is it attacked after recapturing?
   - "X guards Y": after X recaptures, is Y still on X's lines? can X legally move (pinned, in check)? does X have a job
     worth more? can the line X->Y be blocked? is X safe on the new square?
   - "X blocks A->Y": if X leaves, what does A hit -- Y, and what is behind Y (the king: a check)? can X leave with tempo
     (a capture, a check, a threat)?
   - "A attacks Y": is A pinned or attacked itself? does the capture come with check (no in-between moves for them)?
   - "the reply is forced": what are all the replies (king moves, interpositions, captures)?
   - "kick X" (a move that attacks X): what does the kicking piece leave undone (its own jobs)? where can X go?
3. FINDINGS -- answered questions: (claim in words over named pieces; value interval [lo, hi] in material for the side to
   move; status PROVEN -- an exact computation or the solver -- or ESTIMATED; depends-on: the squares read and the findings
   cited). A finding is evidence for the findings that cite it.
4. THE LEDGER -- the findings and their dependency graph (a truth-maintenance system: Doyle 1979, de Kleer 1986):
   - ADD a finding when a question is answered;
   - RETRACT a finding when a square it read changes or a finding it cites is retracted -- the cascade (the pin goes, so
     "the pawn cannot move" goes, at once);
   - ROLL FORWARD on every move: retract only the findings whose squares the move touched and put their questions back
     on the agenda; the rest carry over;
   - BACKTRACK: a defeated finding ("the queen cannot do both" -> it can) retracts only what was built on it
     (dependency-directed backtracking: Stallman & Sussman 1977).
5. THE AGENDA -- the open questions, ordered by STAKE: how much the answer could move a decision (the width of the interval
   it could change, times whether that decision is still open); later the order is LEARNED (which questions flip
   conclusions -- memory `whatif_search_learned_priority`).

### 39.2 Values, estimates, decisions

- Every candidate move of ours gets an interval from the findings it touches (what it wins, what it leaves undefended).
- BOUNDS FROM THE RULES before any answer: a piece left en prise costs at most its value; a pawn kick costs at most a pawn
  plus the kicking pawn's jobs; a check costs nothing unless the checker is lost.
- STOP (Berliner's B*, 1979): when one move's lower bound >= every other move's upper bound, decide -- the remaining
  questions cannot change the choice. A question whose answer cannot separate the bounds is not asked.
- ESTIMATE when a question cannot be answered within its budget: the rule bound's midpoint first, later a learned
  estimate (the long-window model, the tactic density). The finding is marked ESTIMATED and so is every decision resting on
  it. When play later shows an estimate wrong, that is a SURPRISE -- the surprise loop's existing input: the estimate is
  corrected, or a lemma is mined.
- CONFIRM: when material is at stake, the chosen line is checked by the exact solver, a few moves deep -- the solver is the
  checker, never the finder.

### 39.3 Where the parts already are

- Facts: vocab.rs words (`see` = the exchange law, proved in Lean: Ideas.ExchangeLaw; `without`, `job` from 2026-10-14).
- The roll forward: PlanState already carries the game (its line) move by move -- the ledger lives beside it in Rust.
- Lemmas: a PROVEN chain with its piece names generalized is a lemma (explanation-based learning: Minton 1984) -- the same
  claim answered once for a whole class; the ledger answers a question by a lemma when one covers it. The tower (§38)
  builds lemmas bottom-up from the rules; the ledger uses them top-down in a position.
- The explanation shown to the user: the dependency path from the decision down to the facts -- the because-chain.

### 39.4 Phases and tests (each a short run first; numbers only from OBS lines)

1. THE LEDGER ON ONE POSITION: facts, the safety question for every piece, the menu, findings, the cascade. Test: on the
   position above the ledger reaches "e5 is pinned: moving it costs >= bishop - pawn", every finding PROVEN, and the
   solver confirms (…exd4 and …e4 both lose material by force); print the because-chain and the number of questions asked.
2. ROLL FORWARD: through Stockfish's line 1…Qc7 2.Rbd1 Rc6 3.h5 exd4 -- the pin finding must retract when the rook
   reaches c6 (a jobless guard of d6) and …exd4 then be found playable; print how many findings carried over vs were
   re-asked at each move.
3. BOUNDS AND DECISIONS: B* stop and rule bounds on a set of tactical positions with a known best move (the puzzles used
   before); print per position: the move chosen, PROVEN or ESTIMATED, questions asked, and the solver's nodes for the same
   decision (strength per node -- memory `north_star_strength_per_node`).
4. LEARNED ESTIMATES AND QUESTION ORDER from the surprises; lemmas from proven chains (Kaveh's metric: lemmas found and
   what each changed).
Lean: the critical questions' answers are the vocabulary's words (already Lean-defined); new chess claims in a lemma are
proved before they are relied on. The ledger's bookkeeping (retraction, the agenda) is not chess and needs no proof.

### 39.5 Stress test (2026-10-14 14:09 PDT): the holes found, and what was decided

1. WHOSE MOVE. Every question is stamped with the side that acts. "Is our piece X safe" is asked as if we PASSED (the
   null-move threat -- their best capture sequence if they move now); "is their piece safe" as if we move. A candidate move
   of ours is valued by the ledger of the position after it, them to move. The ledger is symmetric: the same code reads
   both sides.
2. CYCLES ("is X guarded -> by Y -> does Y have another job -> guarding Z -> is Z guarded -> by X ..."). A question is asked
   at most once per (type, pieces, position); a question whose answer would cite a finding still open on the stack is
   closed as ESTIMATED with the rule bound ("assume the guard holds") and marked CIRCULAR, so a surprise can find it.
   The dependency graph stays acyclic by construction.
3. BLOW-UP. A budget of questions per decision (printed; start at 200); over budget -> ESTIMATED with the skipped question
   recorded in the finding. The B* stop usually ends long before.
4. WHAT A FINDING READS. The fact evaluator in Rust records its READ SET (every square it looked at: the pieces, and every
   square of every ray it walked). "Qc6 guards c4" reads c6, c5, c4 and d3->c4's ray; a rook arriving on c6 later touches
   the ray into d6 that "guards of d6" walked -- the cascade needs no chess knowledge, only the read sets.
5. VALUES. Material in pawn units (the kernel values) for the side whose decision it is; a mate ±infinity. A candidate
   move's interval: lo = immediate exchange gain − upper bounds of the jobs the moved piece leaves undone − upper bounds of
   the new threats against us it allows; hi = the same with the lower bounds. Intervals combine by worst case for lo,
   best case for hi. This is a bound, not an evaluation: the solver confirms the chosen line when material is at stake.
6. THE KING. "Is the king safe" = no check sequence that wins material or mates; checks and captures of a bigger piece are
   asked before quiet questions (they are the in-between moves). Mate patterns come from the tower's floors (§38) as
   lemmas; phase 1 handles checks as forced replies and leaves mate search to the solver.
7. CANDIDATES. Not every legal move: the moves the findings suggest (winning captures, defences of an unsafe piece, kicks,
   blocks, moves of a piece with no job), plus every check and capture. All legal moves only inside the solver's confirm.
8. DECISION = the root move choice only; sub-choices (whether to recapture) are inside the exchange law. The B* stop is
   at the root.
9. STAKE of a question = the value of the piece it is about, counted only if it sits on a candidate whose interval still
   overlaps the best candidate's; zero otherwise (not asked).
10. ESTIMATES. First the rule bound's midpoint; later a model trained on OUR OWN play's surprises (never Stockfish as the
    teacher of the estimate; Stockfish and the tablebase only check). Every ESTIMATED finding keeps the question it did not
    answer, so a surprise says which question to answer next time.
11. ROLL FORWARD on every ply, theirs too. Findings live within a game; across games only lemmas persist.
12. LEMMAS (phase 4): a PROVEN chain with its piece names replaced by variables and its facts kept as the conditions --
    the tower's arrangement form; checked on fresh positions by the explorer's guards before it is kept.
13. SCOPE OF PHASE 1: a standalone analysis of a position (`ledger(fen)` -> findings, the because-chain, questions
    asked), not wired into the player; wiring is phase 3.
14. ONE VOCABULARY GAP: "after X recaptures on Y, does X still guard Z" and "if X leaves, what does A hit" need the board
    AFTER A MOVE of one piece: `(moved X s T)` -- T read on the board with the piece on X placed on s (whatever stood on s
    taken), the sibling of `without` (2026-10-14). NEEDS KAVEH'S WORD (the vocabulary is his).
15. THE LANGUAGE OF CHANGE AND OF CONDITIONALS (Kaveh 2026-10-14, after the stress test): proposed base words --
    `(if C A B)` (the floor-number term had to fake it with min/max/eq: 38,873 characters), `(after m T)` (T on the board
    after move m; replaces `moved` in item 14), `(delta T)` (T now minus T a ply ago: increased / reduced),
    `(gained S)` / `(lost S)` (squares in S now and not a ply ago, and the reverse: now does / no longer does). One
    family with `ago` (back), `without` / `after` (sideways), `if` (branch). AWAITING KAVEH'S WORD.

### 39.6 Conjectures, listeners, the canonical form (Kaveh 2026-10-14)

Kaveh: "if this pony can be kicked, then we can win this piece. That's a conjecture ... keep it in your back pocket ... a
listener design or indexing design to help you find these other lemmas that will activate when something happens ... a
canonical form of each query ... hash it and sort by it."
- CANONICAL FORM of a question / finding: (type, side that acts, the pieces as SORTED (kind, square) pairs, the extra
  squares) -- pieces by kind and square, never by frame name, so two routes to the same question hash the same. Hashed;
  the ledger's tables are keyed by it. Mirror / translation canonicalisation only for lemmas (across positions).
- STATUS: PROVEN / REFUTED / CONDITIONAL (a premise open: "if the knight is kicked, +3") / UNKNOWN (over budget,
  the skipped question kept). A CONDITIONAL finding keeps its value and its premise; it is the armed tactic
  (memory `armed_tactics_parked`: the tactic + its blocking condition, watch for the blocker's removal).
- LISTENERS: an index premise-key -> the conjectures waiting on it; when a finding is added, retracted or changes status,
  its key wakes exactly its listeners (Rete-style indexing). The read-set index of §39.5 item 4 is the same index keyed
  by squares: ONE dependency graph whose nodes are squares, findings and assumptions (an ATMS: de Kleer 1986).
- MEMORY WITHIN THE GAME: every answer kept (key -> status, value, evidence, ply answered); a repeated question is a
  lookup; retraction marks STALE with the ply, never deletes ("true at ply 20, refuted at 23" stays readable); a REFUTED
  premise can come back and its listeners still fire.
- WHAT IS PRINTED: per decision, the counts by status, the conjectures left in the back pocket and their premises.

### 39.7 Every finding is a goal generator (Kaveh 2026-10-14)

Kaveh: "if something is or is not safe on the new square, can we reason about why it is or is not? So maybe we can make it
safe ... I'm asking in general, this was just an example." Safety was the first instance; the rule is general:
1. WHY: every finding (safe, guards, blocks, pinned, a mate pattern, a tower floor) rests on CONDITIONS -- facts in the
   vocabulary about named pieces -- recorded as its evidence.
2. WHY NOT: a failed claim records which of its conditions failed -- the smallest set of missing conditions (Kaveh's
   "minimum barriers", memory `planner_tactic_tokens_2026_10_07`).
3. WHERE TO ACT: the finding's READ SET (§39.5 item 4) is exactly the squares whose change could flip it -- generic, no
   chess-specific repair table needed for the where.
4. HOW: only the moves that touch those squares (arrive on, leave, capture on, interpose on) are tried; each becomes a
   CONDITIONAL finding "the claim holds if move m" (§39.6) with its own cost asked by the same ledger -- a repair can break
   something else (the repairing piece's own jobs), so the check recurses, bounded by stake and stopped by the B* bounds
   (§39.2). The menus of §39.1 item 2 and §38.6 (capture, kick, pin, block, add a guard, relieve a job, break a pin, play
   with tempo, sacrifice) are the common repairs, used for ORDER and for the explanation, not as the generator.
5. BOTH SIDES: flipping their findings = our attacking plans; flipping ours = their threats; the same code.
6. MEMORY: a repair too expensive now stays in the back pocket; its listener (§39.6) wakes it when the premise comes true
   by any route.
Literature (from memory): means-ends analysis (Newell & Simon, GPS: the difference picks the operator); proof critics
(Ireland & Bundy 1996: the failure pattern suggests the patch). Reuse: the what-if edit kinds (add, guard, remove,
relocate) of catspace/incubator/analyst/whatif_proposer.py (Kaveh 2026-09-28: the what-ifs are the search).
WHEN: after phase 3's measurement -- its miss classification ("a candidate never generated", quiet preparatory moves)
says how many misses this would fix; test: those misses re-run with goal generation on, the repair move generated and
its chain printed.

## 40. The event loop: events, conditions, actions; chains as data; the proposer against puzzles (Kaveh 2026-10-15)

Kaveh, after the move-30 comparison (his reading beat the ledger's: trades when behind, the two-move deflection threat, the
Qc4 discovery, the nearly trapped but unattackable rook): "how do we write the machinery so that the machinery gets this ...
step one, can we get any reasoning going? step two, can we learn the reasoning itself?" Then: "an event condition action
framework, where every move a bunch of events are sent out, conditions are matched, changes are made to some latent state
... lemmas or tactics invalidated or made available ... conditions for things that make something possible or prevent
something ... a whole vocabulary ... a proposer to pick the shape of the events and the conditions ... make the surprise go
away ... puzzles are the way to go." And on 2026-10-14: "I don't want you to add a hardwired exception" -- a higher frame
that needs a guard gone proposes the deflection; nothing is special-cased.
Literature (searched 2026-10-15): the blackboard / Soar lineage, not plain ECA. Events trigger the precondition checks
(polling every rule does not scale: NASA blackboard control); every derived fact carries its justification and retracts
with its premises (a JTMS; our read-set retraction of §39.5 is this); the AGENDA is its own layer, explainable and
learnable (BB1 meta-control; Soar-RL tunes numeric preferences on operators from reward); an IMPASSE opens a subgoal whose
result is compiled into one rule (Soar chunking = explanation-based learning). Chess precedent: PARADISE (Wilkins 1979;
200 productions post concepts, plans guide tens-to-hundreds of nodes, 19-ply combinations) -- its hand-written rules never
covered quiet positions, so it could not play a game: the reason step two matters. Plain ECA fires everything that matches,
no agenda, no stake: the noise of "f7 guards Qg6".

### 40.1 The loop (replaces the fixed script LATENT -> THREATS -> BRANCH a/b -> pokes -> deflection groups -> B*)

The ledger of §39 keeps its objects (facts with read sets, findings with cites and premises, canonical keys, listeners,
STALE). What changes is CONTROL: one loop over an agenda, nothing else decides what is asked next.
- EVENT: a change the loop reacts to. Kinds: `push(move)` (a move arrives: findings whose read set it touches go STALE, their
  listeners wake); `found(finding)` (a finding settled or changed status: its listeners wake); `reason(finding, holder)` (an
  answer that is HELD by something -- a guard, a blocker, a tempo, unreachability -- posts the holder as a thing to ask
  about); `impasse(question)` (no lemma answers it: a chain is opened). Every event is a record: (kind, ply, subject, payload).
- CONDITION: a question's trigger = a pattern over findings and the board in the vocabulary (the premise key of §39.6). A
  question is OPENED only by an event that matches its condition; never by a sweep.
- ACTION: asking the question (a fact read, a lemma lookup, or a chain -- see 40.2) and posting its answer as a finding; a
  finding may post further events. Actions never choose moves; moves are read off the findings at the end (40.4).
- AGENDA: the open questions ordered by STAKE = the value they could change, inherited down a chain ("can Re2 be deflected?"
  is worth the 3 of Bb2 it would unlock). A question is opened only if its stake beats the best value already known for its
  side (the alpha-beta: a settled stake closes the chain -- "the queen is lost, -6" ends the pin chain; "nothing can reach Rh4"
  ends the trapped-rook chain and leaves the rook's mobility as a fact). Ties: the cheaper question (a lookup before a chain).
- BUDGET: per chain, set by its stake (a stake of 1 gets a short chain); the decision's total as now (§39.2).
- BOTH SIDES in one loop: a reason on THEIR finding opens OUR question (deflect their guard = our attack); on OURS, theirs
  (= their threat). Same code, the actor named in the key.

### 40.2 Chains as data (step two needs step one to produce something learnable)

A CHAIN is a sequence of records (question, answer, reason, stake, cost) in the frame language, stored on the ledger and
printed as the trace. Reasoning is data, never Rust control flow: the choice of the next question is a POLICY that reads the
agenda and returns one question; the first policy is "highest stake, then cheapest" (fixed, 40.1); later policies are
learned (40.5) and swap in without touching the kernel.
- Question kinds (the existing menus, now reachable from any reason): SAFE (exchange on a square; reason = the guards or the
  tempo), JOB / RECAP (a guard's other duties), BLOCKS / PIN (a line), FIXERS (the ways out), plus two NEW words, each with a
  Lean def + gloss + Rust mirror (vocabulary rule: tell Kaveh in one line, then add): REACH (can a piece of side c attack
  square s within k moves, each landing square safe by the exchange -- the answer names the route or "unreachable", reason =
  the squares that block it) and MOBILITY (the escape squares of a piece that are safe by the exchange). Kaveh's "the rook is
  almost trapped but cannot be attacked" = MOBILITY 1 and REACH unreachable, two lookups.
- A reason is a first-class edge: finding -> holder (a piece, a square, a line, a tempo). The holder is a THING with a stake;
  the event `reason` carries it. That is "every finding is a goal generator" (§39.7) made the only way a question opens.
- Two-move threats fall out: SAFE(Bb2) held by Re2 -> REMOVE/DEFLECT(Re2) for Black, stake 3 -> FIXERS over Black's moves that
  attack, capture, or pin Re2 -> ...Rfe8 -> re-ask SAFE(Bb2) after it -> 3 -> a CONDITIONAL threat "(if Rfe8) Qb1+ wins 3" with
  its line, stake 3, posted to the agenda of the side to move as a constraint (40.4).
- Nxc6 (the missed fork, 2026-10-14): size_kick's play-out replaced by the chain: SAFE(c6) nets 1, reason "nothing retakes";
  the capture's landing posts `reason(Nc6 attacks Rb8, Qd8)` -> SAFE(Rb8) and SAFE(Qd8) with the knight present -> one of
  them lost -> the chain's value 1 + 5, settled; nothing else of Black's answers both (FIXERS of two things at once = the
  double-threat word already in threat.rs).

### 40.3 Lemmas (chunking): a settled chain whose shape recurs becomes one lookup

When a chain settles, its shape -- the question kinds and the relations among the pieces it touched (guard-of, attacks,
on-line, kind, side), with the squares abstracted -- and its value are recorded as a CANDIDATE LEMMA (canonical form by the
mirror / translation rule of §39.6, kept with its evidence: the positions it settled on). A candidate that settles the same
way on every later chain of that shape (counted; threshold a Kaveh knob, default 20, never below 5) is promoted: thereafter
the question is answered by the lemma in one step, and the chain is run only when the stake is large (the stake knob) or on
the held-out check. A lemma whose answer is ever contradicted by a chain is demoted and the contradiction kept (a surprise).
Lean: a promoted lemma is a conjecture until proved; status shown. This is Soar's chunk and Minton's EBL with the utility
problem handled by the count and the stake knob.

### 40.4 From findings to the move

No branch (a) / (b). At the end of the loop (budget spent or agenda empty above the margin) the findings against US are
CONSTRAINTS on our move (each with its stake: "Qg4 must leave or be traded, 9 as a stake, 0 as a loss at even material";
"Re2 must not be deflectable, 3"; "the c-file must not open with the queen on it, 9"); the findings FOR us are OPPORTUNITIES
(each a line with its stake). Candidates = moves that meet the most expensive constraints or cash the most expensive
opportunity; each candidate is CHECKED by re-running the loop after it with the budget of its stake (the events of the push
do the work: only touched findings are re-asked). Nothing threatened and nothing to win: the opening default (Kaveh
2026-10-15: develop, take the centre, castle, then set up tactics) -- the existing `opening` tie-break rule (commit 1e2bc782)
followed by the move whose push opens the most OPPORTUNITIES for us at the smallest stake against us (steering toward
tactics). The value of a trade is not a rule: the transition table of §33 (trades cost the side far behind) is a stake
source read from data/derived/transitions.json; passed pawns likewise from the pawn words -- stakes, never moves.

### 40.5 The proposer: learning the shapes against puzzles (step two)

The objective is SURPRISE AGAINST PUZZLES, not Stockfish parity (Kaveh: "we can't really expect to match Stockfish ...
puzzles are the way to go"): a Lichess puzzle gives who wins what, how (the line) and when (its length); the loop is run on
the puzzle position; a surprise is (a) the puzzle's move not among our candidates, (b) its value not among our findings, or
(c) our chain longer than the puzzle's line by more than one link. Each surprise is classified by the loop's own records:
which event was missing (no condition matched), which question was never opened (stake too low / budget), which answer was
wrong (a lemma to demote, a word to check), which shape has no lemma (a chain run from scratch).
The PROPOSER (ledger_puzzles.py gains `--propose`): from the surprises of a dev batch, propose (1) new CONDITIONS: a premise
key pattern (from the vocabulary, compositions allowed as in watch.py's lift terms) that would have opened the missing
question -- ranked by how many surprises it fixes against how many questions it would open on the non-surprises (the cost);
(2) new LEMMA shapes: chains that recur across the solved puzzles (40.3 at a lower threshold on puzzles, each still checked);
(3) STAKE / PRIORITY adjustments: a question kind whose opening would have fixed surprises gets a higher base stake
(Soar-RL style numeric preference), learned by the same strict pairwise ranking as watch.py's frames over (opened, fixed)
pairs. Every proposal is checked on the fresh sample before it is kept; what is kept is printed as a lemma count and its
ablation (Kaveh's metric), never a solve rate alone.

### 40.6 Phases (each a short run first; numbers only from OBS lines; no batch before 2-3 positions have readable chains)

1. THE KERNEL (reasoning.rs, a new `agenda` module beside it; Python entry `python -m catspace.incubator.analyst.discovery.ledger
   chain FEN`): events, conditions, the agenda by stake, chains as records, REACH and MOBILITY words (Lean def + gloss +
   Rust), the trace printed as the chain. Acceptance: on move 30 (2r2rk1/5pp1/1p4qp/p1b5/6QR/6P1/1B2RP2/5K2 w - - 6 30) the chain
   reads like Kaveh's paragraph: the queen trade (stake from the transition table), Bb2 held by Re2 -> Rfe8 deflects -> the
   two-move threat, Rh4 MOBILITY 1 and unreachable, Qc4 rejected by the c-file reason; on the Nxc6 position
   (1r1qk2r/p3bpp1/2p1p2p/2p1N1P1/4PPn1/2NPp3/PPP1Q1P1/1R3RK1 w k - 1 16, Black passed) the fork chain settles at 6; the three
   trace positions and the 15.Ne5 test keep their answers. Old script behind --loop script for the comparison only.
2. THE MOVE (40.4) and the puzzle harness on the new loop: ledger_puzzles --set dev --n 60 with --chains 5; the dev number
   beside the old script's (phase 3 of §39: dev 63%) as a directional guide; the miss classes printed.
3. LEMMAS (40.3) on the dev set: the candidates found, how many promoted, the chain length before / after (OBS).
4. THE PROPOSER (40.5): one round on dev, checked on fresh; what it proposed, what survived, the ablation per kept item.
5. Then the game walk (the step-through page) on the new loop, and the opening default.
Rules for the builder: AGENTS.md; no new top-level package; Lean for every new word; every number from a printed OBS line;
short messages; no commit trailers; a step-by-step chain on 2-3 positions before any batch; a vocabulary gap is reported to
Kaveh in one line and then added; nothing chess-specific in the loop (the words carry the chess).

## 41. The learner in layers: observables, windows, the tactical layer, the store, the subgoal layer, the cycle (Kaveh 2026-10-15)

Kaveh's design, recorded 2026-10-15 (memory learning_design_observables_2026_10_15; literature map in the JOURNAL the same day): learning anchored on
OBSERVABLES; watching and playing interleaved; from each observable BACKWARD to the point of inevitability, located by the teacher (Stockfish, strength
conditioned); the window inevitability -> observable explained exactly by the tactical layer and mined by the proposer; an indexed, periodically
COMPRESSED store of winning observables giving directionality and SUBGOALS; planning as two imagined futures; a surprise is a new observable. "Make it
modular ... the sub-goal layer and the tactical layer ... think out the components, then send out an agent to build it." This section is the modular
map: what exists, what is new, how the layers stack, the build order. Nothing here is built yet.

### 41.1 The layers (bottom up) and what each one owns

| layer | owns | exists today | new |
|---|---|---|---|
| L0 Vocabulary | words with a Lean def, a gloss, a Rust mirror; frames and their canonical shapes | base_vocabulary.py, vocab.rs, Vocabulary.lean, chain_lemmas premise shapes | nothing (words are added as gaps appear, Kaveh's rule) |
| L1 Observables | the event stream of a game: capture, promotion, check, castle, mate, resignation; and a SURPRISE (a prediction of ours that failed) as an observable of the same type | ledger.py material_events (captures and promotions, settled), the ledger's push events, watch.py's game stream | `observables.py`: one record type, a domain adapter listing the kinds (chess adapter), check/castle/mate/surprise kinds, a stream over watched and played games |
| L2 Windows (the teacher) | for each observable the point of inevitability: the ply where the teacher's loss probability for the victim crossed the knob (default 1%), conditioned on strength (the Elo band of the players, Maia bins later); whether the opponent CASHED it; the window [inevitability, observable] | watch.py sf_move_loss / threat labels (one ply back only), the Lichess stream with Elo bands | `windows.py`: backward scan with the teacher's eval ladder, the knob, the strength band, the cashed flag; OBS: window lengths, cashed rate per band |
| L3 Tactical layer | explains a window EXACTLY with our own machinery: from the inevitability position the chain (decide_by_chain, the solver, the exchange law, the split, the turn-around) must reach the observable; explained / not explained; the surprise classes of 40.5; quiescence = the forcing line's end (Beal: the bounds close) | the event loop (40), the solver, size_kick, the proposer's classes | `explain.py`: run the chain on the window's start, compare with the observable; a window not explained is a SURPRISE, posted back to L1 |
| L4 Proposer | mines the WINDOWS (not whole puzzles): premise shape at the window's start -> the observable; verbs = the vocabulary's relations over the object graph (squares, lines, pieces); lemmas by chunking, promotion by count, demotion by contradiction; per-lemma utility measured (the EBL utility problem) | proposer.py, chain_lemmas.py (shapes, counts, promotion) | scoping to windows; the utility ledger per lemma (match cost vs savings), pruning |
| L5 Store | the index of every observed position with a winning observable, keyed by canonical shape; a COMPRESSION pass merging entries by shape into counts and outcomes (P(win | shape), P(cashed | shape, band)), bounded size; the compressed table is the directionality | chain_lemmas.json (shapes with counts, no outcomes, no compression), data/derived/transitions.json (the trade cell: one directional table already) | `store.py`: the index, the compression pass, the eviction rule (merge, never LRU), the exemplar kept per shape |
| L6 Subgoal layer | two imagined futures from the present position: the nearest winning shapes for us and for them in the store, each rolled forward by the chain to see if it is reachable / preventable; a subgoal = the shape to reach or deny, with a stake; subgoals enter the decision as a PRIOR on candidates (AGENTS.md rule 7: never the value) | the opening default and the trade stake are two hand-made subgoals; decide_by_chain's candidate tags and the proposer's tag bonus are the prior's slot | `subgoals.py`: retrieval by shape, the two futures, the stake, the prior; the tactical layer then serves it |
| L7 The cycle | watch a batch (L1-L5), play a batch against a strength-banded opponent (the same L1-L5 on our own games), compress, learn lemmas, repeat; one agent state (the store + the lemmas + the words) improves across cycles | ledger walk (games), ledger_puzzles (puzzles), the night's cron | `cycle.py`: the alternation, checkpoints per batch, the watchdog, the report per cycle in Kaveh's metric (lemmas found, each one's ablation) |

### 41.2 The interfaces (one record each; everything downstream reads these, nothing reaches across layers)

- Observable {game, ply, kind, side, delta, fen_before, fen_after, source: watched | played | surprise}
- Window {observable, ply_inevitable, teacher_curve[], band, cashed: bool, length}
- Explanation {window, decision (the chain's record), explained: bool, surprise_class, lines[]}
- Shape {canonical premise (mirror / colour / translation), words[]}; StoreEntry {shape, n, outcomes{win, draw, loss}, cashed_by_band{}, exemplar fen}
- Subgoal {shape, for: us | them, stake, route (the chain's line) | None, status: reachable | deniable | open}
- Lemma {premise shape, outcome, count, contradictions, utility {match_cost, savings}}

### 41.3 Where the two layers meet

The subgoal layer says WHERE to go (a shape with a stake); the tactical layer says WHETHER a move gets there or stops them, exactly, and what it costs.
A subgoal never changes a candidate's value; it changes which candidates are opened first and breaks ties (the prior). A tactical result (a line that
loses material) overrides any subgoal. The opening default (40.4) becomes the first subgoal the store yields when it has nothing better.

### 41.4 The teacher's place

Stockfish locates the windows (L2) and checks explanations (L3); it never scores a decision. The point of inevitability is a knob on the teacher's loss
probability (default 1%, strength-banded); the machinery must then explain each window on its own, and an unexplained window is the surprise that drives
L4. Nothing in L3-L6 reads the teacher.

### 41.5 Build order (each phase: a step-by-step trace on 2-3 windows before any batch; no baselines; numbers only from OBS lines)

A. L1 + L2 on 100 watched Lichess games: the observable stream and the windows; OBS: window lengths, cashed rate per band, the knob's effect.
B. L3 on those windows: how many explained by the chain, the surprise classes; the misses are the fodder.
C. L4 scoped to windows + the utility ledger: lemmas found, per-lemma ablation (Kaveh's metric).
D. L5: the store and its compression pass; OBS: entries, shapes after compression, P(win | shape) on held-out games.
E. L6: the two futures as a prior on candidates; OBS: decisions changed, puzzles and windows affected, per-subgoal ablation.
F. L7: one full watch-play cycle; the report per cycle.
Rules for the builder: AGENTS.md; no new top-level package (all under catspace/incubator/analyst/discovery/); Lean for every new word; nothing
chess-specific below the adapter; -m entry points, never a Pool from stdin; checkpoints + a watchdog on long runs; short messages; no commit trailers.

### 41.6 Kaveh's clarification of the layers (2026-10-15, later the same day)

- **L1 is the base**: the observables AND the object graph (squares, lines, pieces, their relations: the vocabulary) together.
- **L2 has two points per window, not one threshold.** The teacher's probability of THIS outcome falls as one goes back from the observable and as the
  players' strength falls. The INEVITABILITY point is where the outcome is (near) certain under best play; the LIKELIHOOD point is where it is more
  likely than not (> 50%) for the band the game was played in. Both are read off the same curve; the knobs are the two probabilities, not a ply count.
- **L3 and L4 work together on the windows**: the tactical layer looks inside a window, the proposer proposes rules that match what was found there --
  and a rule must match EVERY window seen to date, not only this one. (That global check is the hard part; today's held-out check with demotion on
  contradiction is its first form; the store's compressed shapes are what make it affordable.)
- **L5 the store holds everything seen to date**: positions, windows, lemmas, the vocabulary layers.
- **L6 has a LEARNER of subgoals with more than one source**: (a) memory -- winning shapes remembered that are reachable from here; (b) tactic density
  (the long-window / short-window idea of 2026-10-13); others later. Subgoals feed DOWN to the tactical layer.
- **The tactical layer has an APPLIER**: given the subgoal and the board, it picks which lemma to apply where to get there. Pipeline: PROPOSER finds
  lemmas -> VERIFIER verifies them (the exact solver, Lean) -> STORE keeps them -> APPLIER uses them on the board at hand. (The chain's one-step lemma
  answers of 40.3 are the applier's embryo.)
- **Two phases**: TRAINING (watch, windows, propose, verify, store, learn subgoals) and INFERENCE / play / analysis (set the subgoal, the applier plays it).
- **The board state is carried forward incrementally from the start of the game**, always: a given position is reached by applying its moves one by one
  from move one ("the dumb way, keep it simple for now"). The ledger's push (40.1) is this; decide_by_chain today starts fresh from a FEN and must be
  switched to the pushed-forward ledger.
- **Rules are PROVEN IN LEAN, never re-checked against the windows** (Kaveh, later the same day: "the only way to guarantee it ... is that if the rules are
  mathematically sound and proven in Lean ... I don't want you to go back and check everything"). The windows are the source of conjectures; Lean is the
  verifier; a proved lemma holds in every position satisfying its premise, so no window is revisited. The gap today: the chunked lemmas (40.3) are shapes
  with counts and no Lean statement; the work is shape + outcome -> a Lean statement over the vocabulary's renderings -> an automatic proof attempt (the
  exchange-law theorems first). Unproved = conjecture = at most a prior on which questions to open, never a value. The window points, the store's outcome
  frequencies and the subgoal learner stay probabilistic by design: directionality through the prior, never a rule.
- **A window is the span of a small explanation tree** (Kaveh, later again): going back from the observable, each ply offers a handful of REASONABLE
  decisions (not blunders: the moves the teacher rates within a margin of its best, the margin a knob per band), each leading to a different outcome,
  one of them the observable seen. The explanation is the tree "if this, then that, if this, then that ...". The window extends back only while the tree
  stays small (a handful of branches, ~5-10 reasonable decisions in all): a principled end, not a ply count. Two things in the tree: the BRANCH
  CONSEQUENCES (this move leads to that outcome: forcing lines, the exchange law) are the lemmas, PROVED IN LEAN and composed into the explanation; the
  BRANCH PROBABILITIES (which band takes which branch, how often) are MEASURED from the watched games, never proved, and ride on the proved skeleton
  as its prior. Explanation of a surprise = a proved tree over reasonable alternatives with one leaf the observable, probabilities on its edges.

### 41.7 Phase A build brief: observables and windows, with the window page (Kaveh GO 2026-10-15)

Deliverables (all under catspace/incubator/analyst/discovery/, -m entry points, no new top-level package):
1. `observables.py`: the Observable record (41.2) and `stream(games)`; kinds through a domain adapter (chess: capture, promotion, check, castle, mate,
   resignation; a settled material change as ledger.material_events already computes it); `python -m ...observables --games N` prints OBS counts per kind.
2. `windows.py`: for each material observable, the EXPLANATION TREE backward from it with the teacher (Stockfish, depth and multipv knobs): at each ply
   back, the REASONABLE moves (within a margin of the teacher's best; the margin a knob per Elo band), each followed by the teacher's best play to see
   whether the same observable (the victim's material loss >= delta, or mate) still happens -> branch outcome: SAME, AVOIDED, OTHER (another observable,
   recorded). Going back stops when the tree stops being small (the knob: reasonable branches in all, default 8) or at move one. The inevitability
   point = the furthest ply back at which every reasonable branch still loses; the cashed flag = the loss actually fell in the game; the band of the
   game. Branch PROBABILITIES per band are not in phase A (they need a band-conditioned policy over many games; recorded as the next step). Windows
   sharing a start position are MERGED: their leaves grouped as winning / losing / neutral for the side to move at the start.
3. The WINDOW PAGE: one static HTML page per game (python-chess SVG boards, the step-through page's style): the window's start position, the tree of
   branches ply by ply (the played move marked, each branch's outcome class and the teacher's number as a checker only), the leaves grouped by outcome,
   click a branch to see its board; the inevitability point and the cashed flag shown. Generator: `python -m ...windows --page <id>`.
4. Tests in tests/test_windows.py: the observable kinds on a known game; the tree on a hand-made position where one branch avoids the loss; the stop rule.
5. Stage 1 = items 1-4 traced on 2-3 Lichess games (OBS lines: observables per kind, trees' sizes and depths, inevitability depths), the pages written,
   then STOP and report. Stage 2 (after Kaveh's look) = 100 games on all cores with per-game checkpoints and a watchdog, the same OBS lines aggregated
   per band. Rules: AGENTS.md; Lean for any new word (none expected); no Pool from stdin; numbers only from OBS lines; no commit trailers; short messages.
Amendment (Kaveh, the same hour): the TEACHER is pluggable like the domain adapter -- a protocol (score, reasonable moves within a band's margin, a
rollout of its best play) with Stockfish as one implementation; windows.py uses only the protocol, so another engine, a tablebase, a learned policy or a
non-chess scorer slots in. NO big mining job, now or later: stage 2 (100 games) is dropped; the entry point takes a handful of games. Machinery first; then
the CYCLE (41.1 L7) in small turns: mine a few windows, work on them, prove lemmas, mine again.
Kaveh, after the first window page (2026-10-15): "the windows are too many. we need to focus on important windows leading to important observations." An
IMPORTANT observation: at the end of your turn the material balance has shifted compared to the end of your last turn (kind `shift`, signed delta, through
the adapter's `balance`). Even trades are no observation; a capture sequence counts once, by its net, at the turn where it settled. Only important
observables (shift, mate) get windows.

### 41.8 The self-mining loop (Kaveh 2026-10-15, after the window board)

"I don't want it to use the lemmas we already know and the language you already know to explain the position. I need it to learn: propose, verify,
and if it doesn't work, propose something different until you can explain it and it's provable in Lean. We want it to mine itself." Phase B is this
loop, not an explanation by the existing chain. Allowed at the start: the base vocabulary (words with Lean definitions), the Lean theorems already
proved as proof material (the exchange law), the exact solver as a pre-check, the teacher to label branches. Not allowed: the hand-written frames and
rules, the chunked store, the defence frame, learned priorities. Loop on one window: the target (the played move loses >= delta, the avoiding moves do
not) -> propose small premises over the words that separate SAME from AVOIDED on every branch -> verify (solver on the concrete positions, then a Lean
statement and a proof attempt composing local theorems) -> fail -> propose differently -> done when a proved lemma covers the window. The hard part:
the composing tactic in Lean (a fork = two exchanges + a tempo); it does not exist. First target: the rook window of sHa6yUKv (17.Qg4? Qxc3+ fork).
