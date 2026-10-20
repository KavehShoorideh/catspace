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
Kaveh GO (the same hour): start with the exchange law as the known material; once the machinery works, more of what we know can be added to accelerate
it, and later the start can be moved to more basic things.

### 41.9 Phase B build brief: the self-mining loop on one window (stage 1)

`selfmine.py` under catspace/incubator/analyst/discovery/, -m entry `--game sHa6yUKv --window 3` (reads data/derived/windows/<id>.json). Not allowed
inside: the chain's frames and rules (reasoning.rs decide/size_kick), the chunked store, the defence frame, learned priorities. Allowed: the words
(base_vocabulary, vocab_eval / vocab_values in analyst_core; note their lone-side mirror convention), the Lean theorems in lean/ideas (ExchangeLaw),
goals.forced_moves as the concrete pre-check, the window's teacher labels.
1. THE TARGET from the window: at the start position the played move loses >= delta for the mover (SAME); each AVOIDED branch does not. Examples = the
   SAME branches' positions after the move, counter-examples = the AVOIDED branches' positions after the move (and the OTHER ones set aside).
2. PROPOSE: atoms = vocabulary terms over the OBJECTS of the window (the squares of the pieces that move, capture or are captured in the window's
   lines, plus the kings), including `after`, `see`, `attackers`, `defenders`, `mobility`, `declined`, `attackable`, lines/between; premises = conjunctions
   of 1-3 atoms, canonical under mirror / colour / translation (reuse chain_lemmas' canonical form if it is only a form, not its store). Keep a premise
   when it holds on every example and fails on every counter-example; rank by size then by how many of the window's plies it covers. Print the top 10.
3. VERIFY, concrete: for each kept premise, the solver confirms on each example that the mover loses >= delta within the window's plies (forced_moves
   from the other side); a premise whose outcome the solver does not confirm on every example is dropped.
4. VERIFY, Lean: render each surviving premise and its outcome as a Lean statement over the vocabulary's Lean definitions (base_vocabulary.lean() renders
   terms; the outcome as a `seeOf`-based claim where it is one exchange, or as a forced-loss claim if a definition exists -- if none exists, say so, do
   not invent one outside Lean), write it to lean/ideas/Ideas/Mined/<game>_<window>.lean, run `lake env lean` on it: report TYPECHECKS / PROVES (try
   `decide`-free automation: simp with the ExchangeLaw lemmas, omega, aesop if present) / FAILS with the error. Expect most to fail: the output we want
   is the list of statements and WHY each proof failed (what local theorem is missing).
5. FAIL -> PROPOSE DIFFERENTLY: on a Lean failure, continue down the ranked premises; on exhaustion, widen the atoms (4 atoms, more objects) once.
6. OBS lines: atoms, premises enumerated, kept after separation, kept after the solver, Lean typechecked, Lean proved, the top premises in words, the
   failure reasons. Tests in tests/test_selfmine.py: a hand-made two-branch window where a one-atom premise separates (e.g. the moved piece lands where
   `see` >= 5) and is solver-confirmed. Trace this one window; STOP and report (under 300 words). No batch. Plain commits, no trailers.

## 42. THE ROOT AND ITS LAYERS: explanations from the simplest lemma up (Kaveh 2026-10-15 evening, GO; built by a cheaper agent)

### 42.1 What Kaveh said, and why the loop changes
The goal of the game is mate; mate needs pieces (a lone king cannot give check); so the implicit goals are to win material and to
stop promotions. When a piece hangs you take it, because it is the most valuable thing you can do and nothing bigger threatens you. The
SIMPLEST lemma is therefore the root: an undefended piece is worth its value to the side that can take it. Everything else is a layer
on the root, each layer one named fact that changes the root's prediction: it is defended (the exchange), it is pinned, taking it loses
more elsewhere. An explanation is always the simplest lemma that fits, plus the one deviation when it does not: "normally X, but because
of Y". Lemmas are rewarded for REUSE (the windows they explain beyond their own) and for being SMALL; the atom-combination search of
phase B (selfmine.propose over 4,906 terms) is the wrong source of lemmas -- it finds the shortest sound statement, not the simplest
reusable one. The drill (42.6) stays as the thing that says WHERE a lemma's prediction fails; it no longer proposes.

Values: pawn 1, knight 3, bishop 3, rook 5, queen 9 are declared (a convention), not derived. The one derived fact worth proving: a lone
king cannot give check, so cannot mate (42.3 item 0).

### 42.2 One new word: `worth`
`(worth X)`: the declared value of the piece on X (pawn 1, knight 3, bishop 3, rook 5, queen 9; a king or an empty square 0), whoever
owns it. Lean: `worthOf (p : Position) (s : Square) : Nat := match p.board s with | some q => (pieceValue q.kind).toNat | none => 0`
in Ideas/Base/Vocabulary.lean (pieceValue is Ideas.pieceValue from Fork.lean -- check the import; if Vocabulary.lean cannot import it,
define `worthOf` by a match on the kind with the same five numbers and prove `worthOf_eq` against pieceValue in Root.lean). Python:
base_vocabulary table ("the declared value of the piece on {0}"), `lean()` renderer `(onum (worthOf p) <sq>)`, gloss. Rust vocab.rs:
Term::Worth(Box<Term>) parsed from ("worth", [a]), evaluated from the piece at the square (0 when none or a king), in is_num and reads.
Glossary entry (catspace/research/docs/GLOSSARY.md, the style of `balance`). `lean_index build`.

### 42.3 Lean: Ideas/Root.lean (import Ideas.TreeStep)
Every theorem is small; none is a search. Hypotheses are the WORDS as base_vocabulary.lean renders them (so the generator can discharge
them from a mined statement); conclusions are over the kernel (`applyMove`, `legalMoves`, `materialOf`, `forcingOf`). Proof sketches
given; the builder writes the proofs and keeps `sorry` with a comment where one fails, reporting which.

0. `lone_king_no_check`: if side c has only its king on the board (every square holds none or a c-piece that is a king or a piece of
   the other side), then `inCheckB p (Color.opposite c) = false`. Sketch: unfold inCheckB / isAttackedB to the attacker enumeration;
   every attacker of c is a king; a king attacks only adjacent squares and the two kings cannot be adjacent in a legal position -- if the
   kernel does not give adjacency-freeness for free, state the theorem with the hypothesis `¬ adjacent (kingSq c) (kingSq c')` and say
   so. Corollary `lone_king_no_mate`: with no check there is no mate (mate = in check and no legal move, by the kernel's definition).
1. Material after a move (the kernel facts the root needs). For `m ∈ legalMoves p`:
   a. `material_mover_after`: `materialOf (applyMove p m) p.turn = materialOf p p.turn + promoGain m` where promoGain is 8 for a
      promotion to a queen, 4 to a rook, 2 to a bishop or knight (queen 9 minus pawn 1 etc.), 0 otherwise. Sketch: applyBoard moves one
      piece of the mover; the sum over allSquares changes only at source and destination (and the en-passant captured square, which
      holds the OTHER side's pawn). Use allSquares_nodup and a sum-split lemma (List.sum over a Nodup list with one changed entry);
      the pattern is in TreeStep.lean `sum_le_three_of_one`.
   b. `material_other_after`: `materialOf (applyMove p m) (Color.opposite p.turn) = materialOf p (Color.opposite p.turn) - capturedWorth p m`
      where capturedWorth is `worthOf p m.destination` for a normal move or promotion, 1 for en passant, 0 for castling. Same sketch.
   If (a) and (b) resist, prove the inequalities that the root uses instead: the mover's material does not fall; the other side's
   material falls by at most `worthOf p m.destination` (plus 1 for en passant) -- and say which form was proved.
2. `capture_gains`: our legal move m from x to y onto their piece: `balanceOf (applyMove p m) c = balanceOf p c + worthOf p y` when m is
   a normal move (no promotion; so x holds a non-pawn or a pawn not reaching the last rank). From 1a, 1b.
3. `reply_bound`: for every legal reply r of theirs from position q (their turn), `balanceOf (applyMove q r) c ≥ balanceOf q c -
   maxHit q c - promoThreat q` where `maxHit q c` = the largest `worthOf` over squares holding our pieces that they attack (in words: the
   most `worth` over `(and ours hit)`), and `promoThreat q` = 8 if they have a pawn one step from promotion (a pawn of theirs on its
   seventh rank) else 0. From 1a, 1b and "a capture lands on an attacked square": `m.destination` of a legal capture is attacked by the
   mover -- prove from pseudoLegal (normalPieceGeometryB gives the attack geometry; if the kernel has `legal_capture_attacks` or similar,
   use it; otherwise state it as the one hypothesis of the theorem and report).
4. THE ROOT, `hanging`: p.turn = c; our non-pawn piece on x can legally go to y (the three legs of 41.9/TreeStep: ours, not a pawn,
   `moves`); their piece on y; after the move, the most `worth` over our attacked pieces is w and no pawn of theirs stands one step from
   promotion; then `forcingOf p c (worthOf p y - w) ≠ 0` whenever `worthOf p y - w ≥ 4`... stated as: with `k := worthOf p y - w`,
   `(k : Int) ≥ v` and `k ≥ 4` imply `forcingOf p c v ≠ 0`. Proof: forcingOf_ne_zero with n = 1; step_after (TreeStep) for the move;
   step_theirs over every reply r with the bound of 3; leaf by gainGoal_of_balance. The `k ≥ 4` is the dead-draw guard of
   gainGoal_of_balance; state it and move on. In words the hypotheses are: `(count (and ours (at x))) ≥ 1`, `(count (and ourpawns (at x)))
   ≤ 0`, `(count (and (moves x) (at y))) ≥ 1`, `(count (and theirs (at y))) ≥ 1`, `(worth y) ≥ k + w`, `(after x y (most-over-set ...))`
   -- the "most worth over our attacked pieces" needs a set-maximum word; there is none. Use the composition the vocabulary already
   allows: `(after x y (count (and ours hit)))` = 0 for the first layer (nothing of ours is attacked after the capture: w = 0), and for
   w > 0 the bound per value class: `(after x y (count (and (and ours hit) <pieces worth more than w>)))` = 0 is not expressible either.
   DECISION for this build: the root is proved for w = 0 (`(after x y (count (and ours hit))) ≤ 0`) and for w bounded by a declared
   class via `attackers`/`worth` of each our piece is NOT attempted; the second layer (42.4) handles defended pieces. Report what the
   words cannot say; Kaveh adds words by the standing rule.
5. `root_explains`: the proved root applied to the solver's label on a position: a test, not a theorem (42.7).

### 42.4 The first layer: the defended piece (the exchange, one step)
`defended_once`: as the root, but y has exactly one defender of theirs and after they recapture on y our balance is `+worth(y) - worth(x)`;
if that is ≥ v (we took something worth more than the capturer) the gain is forced in 2 moves (4 plies): our capture, their recapture
(any other reply gives the root), our... no: after their recapture the balance is settled at `worth(y) - worth(x)` only if THEY cannot
be recaptured again -- the exchange sequence. DECISION: this layer is stated and proved only for the two-ply case "they recapture and we
do not continue": conclusion `forcingOf p c v ≠ 0` for `v ≤ worth(y) - worth(x)` requires the balance to hold after EVERY reply, and a
recapture is one reply, so the root's statement with `k := worth(y) - worth(x) - w` where w bounds their best OTHER capture covers it: a
recapture on y is a capture of a piece worth `worth(x)` on an attacked square, so `maxHit` already counts it. So the first layer is the
root with w = worth(x) when y is defended; no new theorem, one new reading. Write that down in Root.lean's docstring; the full exchange
sequence is the exchange law (ExchangeLaw.lean, unproved) and is NOT in this build.

### 42.5 The proposer searches lemma space (selfmine.py, a new mode `--layers`)
Replace the atom-combination search by a ladder:
1. ROOT FIRST. For the window's example position (after the bad move, our turn) list every capture we can make: (x, y) with `moves x y`
   and a piece of theirs on y. For each, read the root's hypotheses in the words (the legs, `worth y`, `(after x y (count (and ours
   hit)))`), and the root's prediction `k = worth(y) - w` with w the most worth among our attacked pieces after the move (computed in
   Python from the words `(after x y (count (and ours hit)))` per square: evaluate `(after x y (count (and (and ours hit) (at s))))` for
   each our square s, take the max worth; that is a Python reading of the words, not a new word). The root's claim: forced gain ≥ k.
2. CHECK against the solver (the labeller): does the solver confirm a forced gain ≥ k within 1 move (n_opp = 1)? If yes and k ≥ delta:
   the window is EXPLAINED BY THE ROOT; emit the Lean statement as an instance of `hanging` (the generator writes `exact
   Ideas.Root.hanging ...` with the hypotheses from the atoms) and STOP. If the solver confirms ≥ delta but the root's k is smaller, the
   root under-predicts: record the deviation "gain larger than the hanging piece: w over-counted or a follow-up" and continue to 3. If
   the root's k ≥ delta but the solver refutes: record "the root over-predicts" with the solver's refuting line -- this is the surprise
   the next layer must explain -- and continue to 3.
3. DEVIATION. Only now run the old proposer (41.9 atoms, the drill of 42.6 as the filter), but SEEDED: the premise must contain the
   root's atoms for the chosen capture, plus at most TWO more atoms (the deviation); rank by size then coverage. Lean as before (the
   generator extended: root atoms discharge by `hanging`'s steps where they can; otherwise the tree proof of 41.9 if the premise has that
   shape; otherwise TACTIC).
4. On the rook window of sHa6yUKv (the example after 17.Qg4, Black to move): the captures are Qxc3 (a pawn, defended by nothing? read it)
   and nothing else; the root predicts +1 (a pawn) or less; the solver confirms +5 forced in 3 moves -- the root UNDER-predicts, deviation
   = "after Qxc3+ the rook on a1 hangs and the king must move": the seeded search should find `(after f6 c3 (least (see a1))) ≥ 5` or the
   check atom as the deviation, which Lean cannot yet prove (the exchange law) -- so the expected outcome here is: root instance NOT
   proved as the explanation, a two-atom conjecture (root + deviation) recorded with its coverage, PLUS the already proved tree lemma
   (41.9) kept as the sound fallback. Print all three clearly.
5. A second window where the root DOES explain: find one. `python -m catspace.incubator.analyst.discovery.windows --pgn-index N --page`
   on games with a shift of 3 or more (scan_shifts in the scratchpad did this; re-implement a tiny scan inside windows.py `--find-shift
   K` listing (game index, ply, delta) over the first 60 downloaded games, no teacher) and pick a window whose example is a plain
   hanging piece (a knight or bishop taken for free). Run `--layers` there: expected OBS "EXPLAINED BY THE ROOT", Lean proves the
   instance.

### 42.6 The drill's new place
`drill()` stays. It is called in 42.5 step 3 only (the deviation search) and in a new report: for a proved lemma, the number of drilled
variants where its premise holds (coverage) and -- the new line -- the number where its premise holds and the solver DISAGREES (must be 0
for a proved lemma: a non-zero count is a bug in the evaluator or the solver; print it and stop).

### 42.7 The score: reuse and size
A table (data/derived/lemmas/score.json, written by `selfmine --layers`): per lemma name (root `hanging`, each mined lemma): size
(atoms), windows explained (list), variants covered per window, and for conjectures the windows where they held. `reuse = windows
explained beyond the window it came from`. Print `OBS score <lemma>: size S, reuse R, coverage C`. No ranking across lemmas is used for
anything yet except the printout; the applier (later) will read it.

### 42.8 Tests (tests/test_root.py, tests/test_selfmine.py additions)
- `worth` on a FEN: queen 9, pawn 1, king 0, empty 0, from both sides' views (the lone-side mirror convention of vocab_eval).
- A hand position where a knight hangs (nothing of ours attacked after the capture): the root's Python reading gives k = 3, the solver
  confirms, the Lean instance proves (run `lake env lean` on the generated file in the test: mark the test slow but keep it).
- A hand position where the "hanging" piece is defended once by a pawn and the capturer is a queen: the root's reading gives w = 9 >
  worth, no claim; the solver refutes a gain -- the over-prediction path is not taken because k < delta; printed.
- `material_other_after` on one concrete move by `decide`/`native_decide`-free evaluation: compare with python-chess material.

### 42.9 Rules for the builder
Read AGENTS.md. No AI trailers in commits. OBS lines for every number. Short runs first (one window, then the second). Builder commits
per piece (word; Root.lean items 0-2; items 3-4; the ladder; the score; tests). Do not edit TreeStep.lean or the 41.9 proof generator's
existing paths except to add the root's discharge. maturin: `cd rust/analyst_core && ../../.venv/bin/maturin develop --release`. Lean:
`cd lean/ideas && lake build Ideas.Root` (add `import Ideas.Root` to lean/ideas/Ideas.lean) then `.venv/bin/python -m
catspace.incubator.analyst.lean_index build`. Where a proof fails, keep the statement with `sorry` and the error in a comment, and say so
in the report; do not weaken a statement silently -- say what was weakened and why. Report under 400 words: numbers from OBS lines only,
what was proved, what was left `sorry`, which words were missing.

## 43. WINDOW BOUNDARIES BY QUIESCENCE (Kaveh 2026-10-15 evening, GO; built by a cheaper agent)

### 43.1 What Kaveh said
A check is usually the middle of a tactical sequence; it should not end a window and usually does not begin one. The END of a window is
the quiescence point. The BEGINNING is where the tactic is set up so that some players fall into it and some do not; the point where
they fall in (or avoid it) is the inevitability point. The split of one exchange into two shifts by an interposed check (MTq15Poj,
20.gxh6 Bxe5 21.Rg1+ Kh8 22.dxe5 became "Black wins 2" + "White wins 3") is the defect this fixes.

### 43.2 Definitions (chess adapter; the protocol stays generic: `quiet(board) -> bool`, `settle(boards, i) -> j`)
- QUIET: the side to move is not in check AND has no capture that wins material by the exchange (for every capture, the exchange value on
  the landing square, read as the vocabulary's `see` from the capturer's side, is <= 0 -- use analyst_core `vocab_eval(fen, "(see X)")`
  per candidate landing square, or the Rust `declined`-style helper if one is exposed; a pure python-chess SEE is acceptable if it is
  tested against `vocab_eval` on 20 positions). A knob `--quiet forcing` switches to the stronger test `forcing_us_1 = 0 and
  forcing_them_1 = 0` (both words exist); default is the exchange test.
- SETTLE(i): from the position after move i, walk FORWARD ALONG THE GAME while the position is not quiet; return the first quiet index j
  (bounded by the game's end). The settled balance of a turn is the material balance at SETTLE.
- SHIFT (observables.ChessDomain.shifts): a turn whose settled balance differs from the previous settled balance of the same side; gain =
  the difference; only the gaining side emits. A sequence 40..44 with a check inside is ONE shift at the first move of the sequence
  whose settled end is the same quiet position; record `settle_ply` on the Observable (new field, default None for other kinds).
- WINDOW (windows.py): the branch rollout follows the teacher's line and then continues until QUIET (not merely through captures); a
  branch's outcome is its settled balance for the window's winner; classes: SAME = the same gain as the game's settled gain (within 0),
  AVOIDED = a smaller gain for the winner, OTHER = a larger gain or a different kind of event (mate). The back-walk is unchanged
  (stop at the branch cap); the inevitability point is the deepest ply back at which every reasonable branch is SAME. Record
  `quiescence_ply` (the game's settle index) on the window. The window's `plies` run from the start to the quiescence ply so the window
  board (window_board.py) shows the whole sequence; the board's cards need no change beyond reading the new field if present.
- The ladder (selfmine --layers) takes the EXAMPLE at the window's start ply (the position where the trap was laid, our turn), not at the
  last ply with both classes; the tree search (41.9) keeps its own choice. Print which position is the example (OBS target ... ply k).

### 43.3 Acceptance (OBS lines, one run each)
1. MTq15Poj: the two shifts at plies 41 and 44 become ONE shift (White, gain 1, settle at the position after 22.dxe5 or the next quiet
   one); print the game's shift list before/after.
2. sHa6yUKv window 2: the quiescence ply is after Kxd7 (or the next quiet position); the window's plies run through it; the proved tree
   lemma's six plies lie inside the window. Run `selfmine --game sHa6yUKv --window <idx> --drill --layers` once; the proved lemma
   still proves (the Mined file is regenerated; report attempted/typechecked/proved).
3. 4pfUO5Kb: the hanging-piece window still EXPLAINED BY THE ROOT.
4. The window pages for sHa6yUKv regenerated (`windows --pgn-index <idx> --page` and `window_board`) -- just report the paths.
Tests (tests/test_windows.py additions): quiet() on three hand positions (quiet; a hanging piece = not quiet; in check = not quiet);
settle() on a hand-made 5-ply sequence with a check in the middle returns the index after the recapture; a shift list on that sequence
has one entry. Numbers only from OBS lines. No AI trailers in commits; one commit per piece (adapter; windows; ladder example; tests).

## 44. LAYERS FOUND BY THE LOOP: diagnose the root, repair it by a generic operator, prove the SCHEMA once, bind it everywhere (Kaveh 2026-10-15 night; written for a cheaper agent, NOT yet a go)

### 44.1 What Kaveh said
"I want the machinery's loop to do this automagically ... so that we get a generic fork lemma." The layers of 42 (defended piece, two
threats) must NOT be written by us: the loop finds them from the words, proves them in Lean and reuses them. What stays hand-written: the
words, the root's hypotheses (they are the diagnostic vocabulary), the Lean step theorems, and a few GENERIC repair operators. What the loop
produces: a SCHEMA = a premise over square variables, proved once as a theorem with the squares universally quantified, then bound to the
squares of any window. Today's tree lemmas name squares (f6-c3, c3-a1) and can never fire again; a schema can, and its REUSE count is the score.

### 44.2 Definitions
- SCHEMA: a premise whose squares are variables X, Y, ... (our move X -> Y, further squares as needed) and whose numbers are variables
  (B0 the balance now, v the gain), with the usual legs. `bind(schema, fen)` enumerates our legal non-pawn-promoting moves X -> Y and
  evaluates the premise in Rust (`vocab_eval`) with the squares substituted; an instance is a binding where every atom holds.
- THE ROOT AS SCHEMA 0: `hanging(X, Y, v)` = the atoms of `root_atoms` with x, y replaced by X, Y. Its Lean theorem is already universal
  (`Ideas.Root.hanging` quantifies x y v): nothing to prove, only to bind. Keep the current code path (root_readings) as its binder.
- DIAGNOSE (new, in `layers`): at the example, for the solver's first move x -> y (the line that forces the gain), evaluate every hypothesis
  of the root with that binding and print which hold and which fail, by NAME (hth: a piece of theirs on y; hv: worth >= v; hhit: nothing of
  ours attacked after; hpr: no promotion threat; hmov: a reply exists; hmat: material >= 4; legs). The failing names are the deviation.
  OBS line: `OBS diagnose <tag>: root at <san> fails {hv, hth}` etc. On sHa6yUKv the expected failures are hth/hv (Qxc3+ takes a pawn, 1 < 6;
  the rook falls a move later); on the gxh6 window of MTq15Poj the expected failure is hhit (the pawn is retaken: our piece is attacked after).
- REPAIR OPERATORS (generic, the hand-written part; apply by failing hypothesis; each produces ONE schema = root + one named deviation):
  * DELAY (for hv/hth: the gain is not now): the root's conclusion holds one move later under every reply. Schema `delayed_root(X, Y, B0, v)`:
      `balance <= B0`,  legs of X -> Y,
      `(after X Y (least (most (if (and (le (count (and ours hit)) 0) (le (count (and theirpawns (rank 7th))) 0)) balance 0)))) >= B0 + v`
    read: after our move X -> Y, whatever they reply, we have a move after which nothing of ours is attacked, no pawn of theirs is about to
    promote, and the balance is up by v. (The `if`, `and`, `le` words exist; check that `le` renders as a Bool for `if`; if the vocabulary has no
    boolean `and`/`le` for the `if` guard, tell Kaveh in one line and add them per the vocabulary rule: Lean def + gloss + Rust mirror + glossary
    + `lean_index build`.) The deviation's name in the explanation: "normally the hanging piece is taken now, but because it hangs only after
    X -> Y (whatever they do), it is taken next move." THIS is the generic fork lemma when the move X -> Y attacks two pieces: no square of the
    targets is named, the `least ... most` says a target survives every reply.
  * DELAY applied to its own result gives depth 2 (the gain after two of our moves). The loop tries depth 1, then 2, bounded by the solver's
    move count for the window. Never deeper than the solver's count.
  * RECAPTURE (for hhit: our capturing piece is attacked after): the same DELAY schema (capture X -> Y, their recapture = `least`, our move
    `most` after which nothing is attacked and the balance is up v). So one operator covers both the fork and the defended piece; the
    diagnosis differs, the schema is the same. Record the diagnosis in the lemma's doc so the explanation names the right deviation.
  * EXCLUDE (for hmat / hmov failing, or for a drill disagreement): add one atom from the words that is false on every counter-example and
    true on the examples (the existing `seeded_premises` with extra=1). Only when DELAY fails to prove. The result is still a schema if the
    added atom names only X, Y or no square; otherwise it is an instance (recorded as such, lowest score).
- SCHEMA PROOF (generator, `schema_proof`): the Lean theorem quantifies `(x y : Square) (B0 v : Int)` and takes the atoms as hypotheses; the
  proof is the existing composition: `forcingOf_ne_zero`, hbase from `balanceOf`, `step_after` (x y), `step_least`, then a NEW step theorem
  `root_tail` (Ideas/TreeStep.lean or Ideas/Root.lean, hand-written by the builder):
      from `mostOver q (fun q' => if (hit-count q' = 0 ∧ promo-count q' = 0) then balanceOf q' c else 0) ≥ 64 + k`, `q.turn = c`, k ≥ 4,
      conclude `forcedWithinB c (gainGoal c base v) q 2 = true` for k ≥ base + v;
    proved by `most_witness` (a move m with the body ≥ 64+k, so the `if` took its then-branch: hit = 0, promo = 0, balance ≥ 64+k), then every
    reply r of theirs: `reply_no_loss` (nothing of ours attacked, no en passant -- state the enPassant hypothesis as in Root.lean or discharge it
    as hanging does) gives materialOf unchanged for us and not raised for them, so balance stays ≥ 64+k, then `gainGoal_of_balance`. The
    promotion count excludes the promotion reply exactly as `hanging` does (reuse its argument; cite the theorems by name, do not re-prove).
    Depth 2 = `step_after`, `step_least` twice then `root_tail`. One schema theorem per (operator, depth) lives in Ideas/Mined/Schemas.lean
    (generated, committed); instances are `exact delayed_root x y B0 v ...` with the window's squares -- the instance file proves in seconds.
- THE LADDER (new order in `layers`): (1) bind every PROVED schema from Schemas.lean, root first, then by size (fewest atoms), at the example;
  the first instance the solver confirms explains the window: print `OBS EXPLAINED BY SCHEMA <name> bound to X=.., Y=..` and count the
  reuse; (2) else diagnose, repair by the operators, drill the instance (disagreements must be 0), send the SCHEMA to Lean, on proof
  append it to Schemas.lean and explain the window by it; (3) else the EXCLUDE operator; (4) else the 41.9 tree search as the fallback
  (unchanged). Reuse in score.json = the number of distinct windows a schema explained; coverage = drilled forced variants the instance holds on.
- Item 0 (pawn captures): the no-pawn leg exists only because `afterMove` promotes to a queen. Replace it by "no pawn of ours on X that
  stands on the rank before promotion" (`(count (and ourpawns (and (at X) (rank 7th)))) <= 0`, 7th read from the mover's side), with a Lean
  sibling of `queenOnly_of_not_pawn` for a pawn not on that rank (`queenOnly_of_not_promoting`), so gxh6 can be the X -> Y of a schema. En
  passant stays excluded (its own leg, or read it as `after` does today -- say which).

### 44.3 Acceptance (OBS lines, one run each; no tuning between them)
1. sHa6yUKv window 1 (`selfmine --game sHa6yUKv --window 1 --drill --layers`): the diagnosis names the failing hypotheses of the root at
   Qxc3+; DELAY depth 1 or 2 produces a schema whose instance the drill accepts with 0 disagreements and the solver confirms; the SCHEMA
   theorem proves in Lean (print its statement in words and the theorem name); the window is EXPLAINED BY SCHEMA. The explanation line reads
   "normally ..., but because ..." from the diagnosis.
2. MTq15Poj, the gxh6 window (W1@41): the root fails on hhit (after item 0 it is a candidate at all); DELAY explains it; the SAME schema
   theorem is reused (reuse = 2 in score.json), no new Lean proof beyond the instance.
3. 4pfUO5Kb window 0: still EXPLAINED BY THE ROOT (schema 0 bound first); the delayed schema is not needed.
4. A fourth game of Kaveh's choice (or the next one in the index): run the ladder and report explained / not, which schema, reuse counts.
Tests: tests/test_selfmine.py additions -- `bind` finds the instance on a hand-made fork (a knight move attacking a queen and a rook); diagnose
names hv on it and hhit on a hand-made defended pawn; `schema_proof` output contains `root_tail` once per depth; the instance proof is `exact`
of the schema. Lean: `root_tail` proved with no sorry; `lean_index build` after. Numbers only from OBS lines; no AI trailers in commits; one
commit per piece (item 0; diagnose + operators; root_tail + schema generator; ladder order + score; tests).

## 45. DEAD-END ATTACKS: every attack on us gets a verdict, proved once (Kaveh 2026-10-16 early morning, GO; built by a cheaper agent)

### 45.1 What Kaveh said
"There's always some piece of us under attack, but those attacks lead to losses for my opponent, so I don't expect them to take ... what
we're looking for is attacks that are dead ends versus attacks that lose material now but gain something later [a sacrifice] ... these
attack lines have been proven to lose material. You have to prove them once, and then that's it." So the root's guard "nothing of ours is
attacked" (never true on a real board, 44) and the max-worth bound (charges us for defended pawns) are replaced by a VERDICT per attack.

### 45.2 Definitions
- At a leaf of a delayed schema it is THEIR turn (after our witness move). Their replies split into CAPTURES of our pieces (the attacks)
  and quiet moves. The sound tail stays `(least (most balance)) ≥ T` (delayed_balance, proved); what 45 adds is its DECOMPOSITION into one
  verdict per reply, each a small lemma, and the EXPLANATION built from them. Lean: `least_of_each` -- if every legal reply m has
  F (applyMove q m) ≥ T then leastOver q F ≥ T (prove it from the definition of leastOver; check TreeStep for an existing form first).
- DEAD END (the one-step exchange, from their side): their piece on s takes our piece on t, and a piece of ours on z takes back on t;
  when worth(s) ≥ worth(t) they gain nothing. Schema `dead_end(s, t, z)`, premise over the words (read with THEM to move):
  their piece on s (count theirs at s ≥ 1), our piece on t (count ours at t ≥ 1), `(worth s) ≥ (worth t)` (worth is the value of what stands
  there, whichever colour), the legs of s -> t for the side to move (legal, theirs, not promoting), and after s -> t the legs of z -> t for us
  (`(after s t (count (and (moves z) (at t))))` ≥ 1, ours at z, not promoting) -- conclusion `(after s t (most balance)) ≥ balance` i.e.
  afterMove p s t (fun q => mostOver q (balanceOf · c)) ≥ balanceOf p c with c = US (the side NOT to move at p). Lean proof: after_of_moves
  for their move (the kernel plays the legal move for whichever side moves; check the hypotheses of after_of_moves / step_after_np and
  write the "their side" variant if it assumes p.turn = c), capture_gains twice (their capture lowers our material by worth t, our
  recapture lowers theirs by worth s), le_mostOver with the witness z -> t. Generated into Ideas/Mined/Schemas.lean by schema_proof as
  the family `dead_end` (depth 1 only), proved once, instances by `exact`.
- VERDICTS (`verdicts(fen, us)` in selfmine.py, fen with THEM to move): for each legal reply s -> t of theirs: if it captures a piece of ours,
  bind `dead_end(s, t, z)` over our pieces z that can recapture (Rust evaluation); on a binding the verdict is
  ("dead end", s, t, z, "taking t loses: we take back with z"); otherwise read `(after s t (most balance))` (Rust) and label it
  ("costs", k) when it is below the balance now (k = the drop) else ("affordable", 0) -- these two are the NEXT LAYER's targets (an attack
  that is not a dead end by the one-step exchange: an escape, a bigger exchange, a pin). A quiet reply gets ("quiet", witness) where witness
  is a move of ours after which balance ≥ balance now (read `(after s t (most balance))` and name the argmax move -- add a cheap
  python-chess argmax, it is explanation, not proof). Print per leaf one OBS line: `OBS verdicts <fen>: attacks N, dead ends by schema D,
  costs C (max drop k), affordable A, quiet Q`.
- THE EXPLANATION of a delayed_balance instance (the EXPLAINED BY SCHEMA line) now ends with the verdict list at the leaf of the solver's
  line (the position after our last witness move along `sol['line']`): "attacks on us after <line>: <t> by <s>: dead end (we take back
  with <z>); <t'> by <s'>: costs 1; ...". For windows explained by the root nothing changes.
- THE CACHE ("prove once, then that's it"): a verdict depends only on (s, t, z, the kinds and colours on them, worth s, worth t, legality
  of the two moves); keep a per-run dict keyed by that tuple -> verdict; across the leaves of one window (several replies at depth 2, the
  drill's variants if cheap) count hits: `OBS verdict cache: M verdicts, H hits`. Reuse by re-binding, never by trusting a stale verdict:
  on a hit the atoms are re-evaluated (Rust, microseconds) and the proof is the schema's; what is saved is the search for z.
- SACRIFICE FLAG (detection only, no learning yet): when a window's winning line (the solver's first move, from the WINNER's side) is a
  capture that `dead_end` labels a dead end for the winner (i.e. bound from the VICTIM's view, the winner's capture looks like it loses by the
  one-step exchange) print `OBS sacrifice candidate <tag>: <san> is a dead end by the exchange yet the solver forces <delta>`. The fork
  Qxc3+ (a queen takes a pawn defended by a pawn) should be flagged; this is the deviation "normally taking here loses, but because of Y
  they gain more later" for a later plan.

### 45.3 Acceptance (OBS lines, one run each; no tuning)
1. `dead_end` schema PROVES in Schemas.lean (print the statement in words). A hand-made test: a rook attacks a defended pawn (dead end),
   a pawn attacks a queen (costs 8 or an escape: "affordable" if the queen can step away keeping the balance), a quiet reply.
2. sHa6yUKv window 1: still EXPLAINED BY SCHEMA delayed_balance_d2; the explanation lists the attacks at the leaf after Qxc3+ Ke2 Qxa1
   <reply>; report the verdict counts and which attacks are dead ends by the schema; the sacrifice flag on Qxc3+.
3. 46LzXjoz window 1: the attack on our pawn after Qxf1 gets its verdict; counts.
4. 4pfUO5Kb window 0: unchanged (root).
5. The verdict cache line on each run.
Tests: tests/test_selfmine.py additions (the hand-made positions of item 1; the explanation string contains "dead end"; the cache counts
hits on a repeated leaf). Lean: `least_of_each` and the schema proved, no sorry; `lean_index build`. Numbers only from OBS lines; no AI
trailers; one commit per piece (Lean; verdicts + explanation; cache + sacrifice flag; tests + acceptance mined files).

## 46. THE SEARCH RULER: an alpha-beta solver with Stockfish's ordering, its trace beside the stream (Kaveh 2026-10-16 afternoon, GO; built by a cheaper agent)

### 46.1 What Kaveh said
"Minimax with alpha-beta ... bake in the heuristics that Stockfish has (checks and captures first; go check what they do) ... measure how
many nodes we visit that we shouldn't ... I don't think we'll ever beat Stockfish; that's not the goal. The good interpretability goal is
the right one: say the logic behind it. I want move numbers on the same order as Stockfish: for every puzzle we solve, see the trace,
which nodes it visited first and how it went by. Our search should visit fewer nodes but go deeper." The solver is the RULER, never the
player (the criterion: progress is never measured by Stockfish; it may label and verify). The tactical-frame search (ours from the frame,
all their replies, to quiescence, every cut with its reason) comes AFTER this ruler exists (plan 47).

### 46.2 Definitions
- `catspace/incubator/analyst/discovery/ruler.py`: `solve(fen, need, max_depth, trace) -> dict(status, line, nodes, depth, trace)`:
  iterative deepening, negamax alpha-beta over ALL legal moves of both sides, a quiescence search over captures (and checks at the
  first quiescence ply) with stand-pat, the leaf = material balance in PIECE_WORTH (mate = +100/-100, stalemate/repetition/50-move = 0,
  the window's `need` is the target: "forced" when the root score ≥ need against every defence). Ordering, Stockfish's own in order:
  the hash move (previous iteration's best), captures by MVV-LVA (most valuable victim, least valuable attacker) with SEE-negative
  captures after the quiet moves, promotions, checks, two killer moves per ply, history heuristic. Read Stockfish's movepick.cpp first
  (`git log` / web); name in the docstring what is taken and what is left out (no null-move pruning, no LMR, no futility: Kaveh wants a
  FULL search with ordering, so every node visited is one a plain search needs).
- TRACE: a list of (ply, san path, kind in {main, quiescence}, reason in {cut: beta, cut: stand-pat, leaf, mate, repeat}) in VISIT
  ORDER, capped by `trace` entries (the first N visited); `nodes` is the total count (main + quiescence, reported separately).
- REFERENCE: real Stockfish (`/opt/homebrew/bin/stockfish`, python-chess UCI) on the same position: the lowest depth whose pv's first
  move is the puzzle's answer and its `nodes` at that depth (iterate depth 1..20, `info` lines). Printed as `OBS stockfish depth d nodes n`.
- The puzzle page / stream gets EV 7: `EV 7 ruler: forced at depth d, nodes M main + Q quiescence, line ...; Stockfish: depth d' nodes n'`
  followed by the first 40 trace lines ("1. Kxc2 .. Qc4+ .. Kd2 .. Qxe4 leaf 62" style) -- the order the search looked.
- Entry point: `python -m catspace.incubator.analyst.discovery.ruler --set dev_20 --index 0 [--trace 200]`; `puzzle_apply` calls
  `ruler.solve` in EV 7 (import, don't duplicate).

### 46.3 Acceptance
1. dev_20 puzzles 000Sa (Bxc2, gain 5) and 00cZ4 (Qh2#) and 0Ag9s (Qxa8+): status forced, the line's first move = the answer; OBS lines with
   nodes and Stockfish's nodes. One run each, numbers directional.
2. The trace of 000Sa shows the check Qc4+ visited before quiet replies after Kxc2 (the ordering works).
3. tests/test_ruler.py: mate in one found at depth 1; a hanging piece found at depth 1; the ordering puts a capture before a quiet move;
   the trace's first entry is the root's first child; a position with no gain returns "stopped".
No Pool; no AI trailers; one commit for the solver + tests, one for the EV 7 hook; the Rust solver `forced_moves` stays the exact labeller.

## 47. THE MATE WORD AND THE MATE-IN-ONE SCHEMA (autonomy night 2026-10-16/17; built by a cheaper agent)

### 47.1 Why
Six of the twenty dev puzzles are mates and the stream says "no lemma reads mate yet" on every one; the goal in Lean already counts a mate
(`gainGoal`'s second disjunct: their turn, no legal move, in check) and `forcedWithinB` checks the goal right after our move, so a mate in one
is a one-step theorem away. One vocabulary gap: there is no word for "how many legal moves the side to move has" (`flights` counts the
KING's squares only). Vocabulary rule (Kaveh 2026-10-14): add it -- Lean def + gloss + Rust mirror + glossary + `lean_index build`.

### 47.2 Definitions
- WORD `mobility` ((), num, "how many legal moves the side to move has"): Lean `def mobilityOf (p : Position) : Nat := (legalMoves p).length`
  in Ideas/Base/Vocabulary.lean next to `maxWorthOf`; Rust `Term::Mobility` in rust/analyst_core/src/vocab.rs (parse "mobility", eval =
  the legal move count of the side to move in q, the dependency mask FULL like MaxWorth); Python PRIMITIVES entry + Lean render lambda
  `"mobility": lambda: "(mobilityOf p)"` in base_vocabulary.py; GLOSSARY.md entry (catspace/research/docs/GLOSSARY.md, the `maxworth` style);
  tests/test_vocab_mobility.py in the shape of test_vocab_maxworth.py (python-chess `legal_moves.count()` on three positions; a mated
  position reads 0). Under `after x y` the word reads the board after our move: THEIR mobility. Rebuild the Rust module the way the repo does
  (look at how `maxworth` was added: `git log -S MaxWorth`), then `lean_index build`.
- THEOREM `mate_in_one` in Ideas/Mined/Schemas.lean (or Ideas/TreeStep.lean if the step belongs there): for p with p.turn = c and a legal
  normal move m of ours from x to y (the same legs the delayed schemas use: `with_legality` renders them), if after it the side to move has
  no legal move (`mobilityOf (applyMove p m) = 0`) and is in check (render as the existing words: `(after x y (count (and (at L) attacked)))
  ≥ 1`, L their king on the board after -- check how `hit`/`kingSqD` are rendered under `after` and use the same for `attacked`; if the
  check cannot be said in the existing words, add the word `check` ((), num, "1 when the side to move is in check") the same way as
  `mobility` and say so in the report), then `forcingOf p c v ≠ 0` for every v. Proof: unfold forcingOf (q = p since p.turn = c), n = 1,
  `forcedWithinB c goal p 2 = goal p || any ...`; pick m; `forcedWithinB ... (applyMove p m) 1 = goal (applyMove p m) || ...`; the goal's
  second disjunct: the turn flipped (find the existing lemma on applyMove's turn; TreeStep/Root proofs use one), legalMoves empty from
  mobility 0 (`List.length_eq_zero`), inCheckB from the check atom. Search the index for the applyMove-turn lemma and inCheckB facts first
  (`python -m catspace.incubator.analyst.lean_index search applyMove turn`, `... inCheckB attacked king`).
- FAMILY `mate` in selfmine.py's ladder (FAMILIES += "mate"; `schema_name(1, "mate") = "mate_in_one"`): `bind(1, fen, mover, need, candidates,
  "mate")` returns the candidates whose two atoms hold (read with `premise_truth`), with prem = the two atoms; no tail, no B/T. The instance
  proof is an application of `mate_in_one` with the squares (mirror `schema_instance_proof`). `ensure_schema` must not try to regenerate it.
- STREAM: puzzle_apply reads the `mate` family first in the bind stage for every puzzle (it is the biggest gain): `bind mate: Qh2# -- their
  mobility 0, their king attacked: mate in one` and the decision takes it over any material gain (k = 100). A puzzle whose window is a mate
  in two keeps "no lemma reads mate in two yet".

### 47.3 Acceptance
1. 00cZ4 (black, Qh2#): bind mate fires on Qh2# and the decision is Qh2#, MATCH. 01IFU (Nf3+, mate in two): bind mate does not fire.
2. Lean: `mate_in_one` and one mined instance (00cZ4, written with the mined-file machinery into Ideas/Mined/) build with no sorry.
3. tests: test_vocab_mobility.py; tests/test_selfmine.py gets `test_mate_family_binds_only_on_mate_in_one` (00cZ4 fen binds Qh2# only; the
   00cZ4 position after a quiet move binds nothing).
No AI trailers; one commit per piece (word; theorem; family + stream; tests); numbers only from OBS lines.

## 48. THE DEEP TAIL: one named move, the rest quantified (autonomy night 2026-10-16/17; built by a cheaper agent after 47)

### 48.1 Why
000Sa: Kxc2 and Bxc2 both read 65 at depth 1 because a one-reply tail cannot see Qc4+ then Qxe4. The depth-2 schemas name BOTH our moves, and
a second named move is almost never legal under every reply (every check of theirs makes it illegal), so "no candidate's legs hold" on real
boards. The reading a puzzle needs is one named move and a DEEPER QUANTIFIED tail: after Bxc2, whatever they do, some move of ours, whatever
they do, keeps the balance. Generic: the depth-1 schema with the minimax tail nested one level more (6 plies = forcingOf's n = 3).

### 48.2 Definitions
- FAMILY `delayed_balance_deep` (FAMILIES += it; ALIVE_NEED like delayed_balance; `tail_family` recognises the path (most, least, most, least)):
  `tail_body` = `(least (most (least (most (least (min balance (sum material_us C)))))))`; `delayed_tail([(x, y)], ...)` wraps it in one `after`.
- THEOREM `tail_four_alive` in Ideas/TreeStep.lean beside `tail_two_alive`: the same hypotheses with the tail nested twice
  (`mostOver p (fun q => leastOver q (fun r => mostOver r (fun s => leastOver s (fun t => min (balanceOf t c) (materialOf t c + C)))))) ≥ T`)
  concluding `forcedWithinB c (gainGoal c base v) p 4 = true`; proof = step_most, step_least, step_most, step_least, leaf, exactly as
  tail_two_alive does once (needs the turn lemmas for four plies: `turn_after_one/two` exist; add `turn_after_three/four` the same way or
  reuse by composing). Then the schema `delayed_balance_deep_alive_d1` is generated and proved by the schema generator the way
  `delayed_balance_alive_d1` is (root_tail_alive-style composition with one `after` leg: look at how `schema_proof` composes the d1 case and
  add the family's case; the window n becomes 3).
- STREAM: puzzle_apply's ORDER gets the family after delayed_balance; `best_bind` works unchanged (one named move per candidate).

### 48.3 Acceptance
1. 000Sa: `bind delayed_balance_deep_d1` reads Bxc2 at 65 and Kxc2 below it (the bishop lost) -- print both candidates' tails (one OBS line
   each) -- so the family alone decides Bxc2 without the shape tiebreaker.
2. The schema theorem and one mined instance (000Sa) prove in Lean, no sorry; `lean_index build`.
3. tests/test_selfmine.py: `test_deep_tail_splits_the_king_and_bishop_capture` (000Sa fen: tail of Bxc2 > tail of Kxc2) and the family's
   `tail_family` recognition.
Time: report the per-candidate seconds on 000Sa and on 04Qx2 (36 legal moves) -- if a candidate takes more than ~10 s, say so (the Rust
evaluator's minimax may need a node cap: budget hit = unknown, never a wrong number).
No AI trailers; one commit per piece (Lean step theorem; family + schema; stream + tests).

## 49. THE MATE IN TWO: a schema whose leaf is a schema (autonomy 2026-10-17; built by a cheaper agent)

### 49.1 Why
Three of the twenty dev puzzles (01IFU Nf3+, 03B3o Qd8+, 06xEN Rh1+) are mates in two and the stream reads "no lemma reads mate in two yet".
Kaveh's goal (2026-10-17): reason instead of searching moves. The reasoning is one sentence: after our move, WHATEVER they reply, some move of
ours mates (the mate-in-one premise of plan 47 at every leaf). Every word exists: `after`, `least`, `most`, `diff`, `check`, `movecount`. No
new word.

### 49.2 Definitions
- TAIL `(after x y (least (most (diff check movecount))))` ≥ 1. Read: after our move x->y, for every reply, the best of our moves reads
  `check - movecount` ≥ 1, which in Nat holds exactly when check = 1 and movecount = 0 (a mate). `least` over no replies reads 0, so a
  stalemate or an immediate mate after x->y does NOT satisfy it (the mate in one is plan 47's job); that is the soundness we want.
- THEOREM `mate_in_two` in Ideas/Mined/Schemas.lean beside `mate_in_one` (written by hand, the generator does not regenerate it): the same
  legs h0 h1 h2 as `mate_in_one`, and
  `(afterMove p (some x) (some y) (fun q => leastOver q (fun r => mostOver r (fun s => checkOf s - moveCountOf s)))) ≥ 1`
  concluding `forcingOf p c v ≠ 0` for every v. Proof: `forcingOf_ne_zero` with n = 2 (`forcedWithinB c _ p 4`), `step_after_np` for our move,
  `step_least` (k = 1) over every reply, `step_most` (k = 1) for our witness move, then the leaf: from `checkOf s - moveCountOf s ≥ 1` get
  `checkOf s = 1` and `moveCountOf s = 0` (omega after unfolding), and close `gainGoal`'s second disjunct exactly as `mate_in_one` does (turn
  flipped: `turn_after_one`-style lemma for the third ply, compose `applyMove_turn` three times). Check how `tail_body`'s `diff` renders to Lean
  (delayed_root_max uses it) and use the same rendering, so the Rust reading and the Lean statement are the same term.
- FAMILY `mate2` in selfmine.py: FAMILIES += "mate2" (after "mate"); `schema_name(1, "mate2") = "mate_in_two"`; `bind_mate2(fen, mover,
  candidates)` evaluates the tail (Rust `eval` of the term under `after`, the way `best_bind` reads tails) for each candidate in the stream's
  candidate order and STOPS AT THE FIRST that reads ≥ 1 (no mate in two beats another; principled stop); returns [(san, pairs, prem)]. The
  instance proof is `exact Ideas.Mined.Schemas.mate_in_two p c hc x y v <legs> <tail>` (mirror `mate_instance_proof`); `write_mate2_instance`
  writes Ideas/Mined/<tag>_mate2.lean and checks it with Lean (the premise_truth of the tail is a `decide`/`native_decide`? -- NO: use whatever
  the mined instances use for `after` tails today (look at puzzle_000Sa_deep.lean and the delayed instances); the instance's hypotheses are
  proved the same way).
- STREAM: puzzle_apply's bind stage reads `mate` then `mate2` before the material ladder; a hit logs
  `bind mate_in_two: Nf3+ -- every reply, then a mate (check 1, movecount 0)` and the decision takes it (k = 100, as a mate). Print the seconds
  per candidate as an OBS line on 01IFU and on 04Qx2 (36 legal moves, no mate): if a candidate costs more than ~10 s say so in the report and
  stop; do not add caps.

### 49.3 Acceptance
1. 01IFU, 03B3o, 06xEN: `bind mate_in_two` fires on the answer and the decision MATCHes. 00cZ4 (mate in one): `mate` fires first, `mate2` is
   not asked. 000Sa: `mate2` binds nothing.
2. Lean: `mate_in_two` and one mined instance (01IFU) build with no sorry; `lean_index build`.
3. tests/test_selfmine.py: `test_mate2_binds_only_the_mate_in_two` (01IFU fen binds Nf3+; the 000Sa fen binds nothing; a stalemate-in-one
   position binds nothing).
4. Rerun the dev_20 batch (the scratchpad's puzzle_batch.py, ~3 min) and report the OBS tally line.
No AI trailers (no Co-Authored-By / Claude-Session lines); one commit per piece (theorem; family + instance; stream + tests); numbers only from
OBS lines; timestamps from `date`.

## 50. THE REMEMBERED STATE: every solve is stored as relations, retrieved by relations, and asked first (Kaveh 2026-10-17, GO for autonomous rounds; built by a cheaper agent after 49)

### 50.1 What Kaveh said (2026-10-17)
"Remember these reasoned states so that next time when you see something similar, you can remember: aha, there is the opportunity for me to
invoke or create this type of scenario which I've seen before, which I know results in this type of thing." Index by RELATIONSHIPS and the
lemma that applied, never by where the pieces stand (2026-10-16). The remembered state is a PRIOR for which lemma to ask first; the lemma's
Lean proof is still what decides (never a value, never a proof).

### 50.2 Definitions
- RECORD (one JSON line per solved puzzle, appended to `paths.derived("reasoned_states.jsonl")`; per-item checkpoint, a rerun skips stored
  ids): `id`, `set`, `fen`, `us`, `decision` (san + uci), `family` (the schema that decided: mate_in_one / mate_in_two / delayed_*), `gain`,
  `instance` (the mined Lean file if one was written), and the RELATIONS:
  - `pieces`: the level-one state's rows (`state_lines` computes them) as tokens WITHOUT squares: `their Q attacked1 guarded0 hanging`,
    `our R attacked1 guarded1`, `our K attacked0`, ... (kind, side, attacked count, guarded count, `hanging` when see > 0, `check` when in check);
  - `roles`: who did what in the decision, by kind not square: `mover=N`, `victim=K|Q|R|...|none` (the occupant of the target square),
    `move=check|capture|quiet` (any subset), `recapturer=<kind>|none` (the biggest defender of the target, from `defenders`);
  - `threat`: the threat stage's biggest threat to us as `threat=<kind of ours attacked> value=<v>` or `threat=none`;
  - `after`: the chosen move's shape (hanging count, guarded, poking, number of their checks), from `shape`.
  A record is written for every puzzle whose decision came from a bound schema (MATCH or not: the decision is what the lemma said; `correct`
  stores the MATCH/MISS so that a wrong memory can be audited later).
- SIGNATURE: the set of `pieces` + `roles` + `threat` tokens. SIMILARITY: Jaccard over the two signatures (generic, no weights; print it).
- RETRIEVAL (`remember.py`, the store's module: `store(record)`, `recall(signature, k=3) -> [(sim, record)]`, `propose(fen, us) -> list of
  (family, mover_kind, victim_kind, move_kind, sim, id)`): before the bind stage, puzzle_apply computes the new puzzle's signature from the same
  functions, recalls the k most similar records of OTHER puzzles (never itself: by id), and logs one `recall` event per hit:
  `#n recall   seen 01IFU (sim 0.62): N gives check, mate_in_two decided -- ask mate2 on knight checks first`.
- THE ORDER OF ASKING: the bind stage asks the recalled families first (in sim order, de-duplicated), each on the candidates that match the
  recalled roles first (same mover kind and move kind), then its other candidates; if a recalled family decides, the stream says so in the
  decision line (`by memory of 01IFU`); otherwise the usual ladder runs unchanged (mate, mate2, ORDER). Soundness is untouched: a family binds
  only by its proved schema.
- MEASURE (OBS lines, one run each, directional): per puzzle `candidates asked`, `families asked`, `seconds`, `decided_by_memory yes/no`,
  `recall_top1_family == deciding family yes/no`; the batch tally prints: matches, how many decided by memory, how many recalls were right, the
  summed candidates asked with memory vs the same batch with `--no-memory`.
- WHERE THE RECORDS COME FROM: store from dev_20 first (the solved ones; it is the short run), then the recall run on `fresh_60_threats_event`
  (60 puzzles, no overlap with dev_*; load it like dev_20). If that run is clean (no crash, tally printed), store from dev_300 in the background
  (per-item checkpoint; check every 5 min; it is ~10 s a puzzle, one labeller outlier ~100 s) and recall again on fresh_60 with the bigger store.
- The record format must also take a WINDOW from a game (plan 41-43: fen at the inevitability point, the window's decision, the lemma), so the
  store serves games later; one field `source=puzzle|window`; do not build the window writer now.

### 50.3 Acceptance
1. `remember.py` + tests/test_remember.py: a record round-trips; a signature has no square names; `recall` never returns the puzzle's own id;
   Jaccard of identical signatures is 1.0, of disjoint 0.0; `propose` on 03B3o after storing 01IFU names mate_in_two with mover Q (whatever
   it is; print and assert the printed value).
2. puzzle_apply: `--no-memory` flag; `recall` events in the stream; the decision line says `by memory of <id>` when it was.
3. OBS: the dev_20 store count; the fresh_60 recall run's tally (matches, decided by memory, recalls right, candidates asked with vs without).
4. puzzle pages: the scratchpad's puzzle_page.py shows the recall events like any event (no change needed if it prints the stream).
No AI trailers; one commit per piece (store + tests; stream + flag; the runs' OBS in the commit message of a journal-free commit, I journal);
numbers only from OBS lines; timestamps from `date`; use `paths.derived`, never a relative path.

## 51. THE REASONED SEARCH: our moves from the reasoning, their replies all, nodes counted against the ruler (autonomy 2026-10-17; built by a cheaper agent after 50)

### 51.1 Why
Plan 50 showed memory cannot save anything on the ladder: the ladder is exhaustive and cheap. Kaveh's goal: "avoid a search of moves as much
as possible by reasoning, and search alongside the moves the reasoning recommends". The ruler (plan 46) is a full alpha-beta with Stockfish's
ordering. The reasoned search is the same alpha-beta where, at a node where WE move, only the moves the reasoning recommends are searched,
every cut recorded with its reason; where THEY move, every legal reply is searched (soundness: a line found is a real forced line; the reasoned
search can miss, never lie). The measure is nodes: the full ruler's against the reasoned search's, puzzle by puzzle, and whether each finds the
answer.

### 51.2 Definitions
- `ruler.solve(..., ours=None)`: an optional `ours(board) -> list[Move] | None` called at every node where the side to move is us (the root's
  side); when it returns a list, only those moves are searched at that node (in the ruler's own order); None = all. Quiescence is unchanged.
  The result gains `cuts` = a list of (ply, fen-free description, n_moves_cut, reason) and `nodes` as before.
- `reasoned.py` (catspace/incubator/analyst/discovery/): `recommend(board, us, memory=True) -> (moves, reasons)`, pure functions of the board
  and the vocabulary (Rust `analyst_core` words where puzzle_apply already uses them), in this order, each a REASON string that goes into cuts:
  1. `mate`: if `bind_mate` / `bind_mate2` fires at the node, return just that move (reason "mate in one/two at this node: nothing beats it").
  2. `in check`: only the king-saving moves (reason "in check: nothing else is legal" -- this is already so; record it anyway as a reason of 0 cuts).
  3. `threat`: when a piece of ours is attacked with see > 0 (the level-one state's `hanging`), the recommended moves are: captures of the
     attacker, moves of the attacked piece, moves that add a defender or interpose (the `defenders` count of the attacked square rises after the
     move), checks, and captures of value ≥ the hanging piece's worth (reason "piece worth v hangs: parry it, or check, or take as much").
  4. `observables`: otherwise the frame's moves: captures, checks, promotions, and moves that create a hanging piece of theirs (after the move
     some piece of theirs has see > 0 under `attackers`/`defenders`), reason "quiet moves that threaten nothing are not searched".
  5. `memory`: `remember.propose` at the ROOT only orders the list (recalled mover kind + move kind first); never adds or removes a move.
  If a rule yields no move, fall to the next; if all yield nothing, return None (search everything, reason "no reasoning applies").
- STREAM: puzzle_apply's search stage logs two lines: the ruler's as today and `search reasoned: <status>, depth d, nodes N (ruler M), cuts K
  moves at our nodes -- reasons: mate 0, threat 12, observables 31 ...`; the line found, when found. Flag `--no-reasoned`.
- BATCH: the scratchpad's puzzle_batch.py tally prints `OBS search <set>: ruler found a/n nodes S1; reasoned found b/n nodes S2; cut moves C`.

### 51.3 Acceptance
1. Trace on three positions first (000Sa, 01IFU, 06J6T), one OBS line each: ruler nodes, reasoned nodes, cuts with reasons, found or not.
   Say what the reasoned search misses and why BEFORE the batch (Kaveh's rule: mechanisms traced on 2-3 positions before any batch).
2. dev_20 and fresh_60_threats_event batches with the ruler on (dev_20 ~3 min; fresh_60 longer with the ruler: check at 1 min then every 5 min).
3. tests/test_reasoned.py: in check -> only king moves; a hanging queen -> every recommended move parries, checks or takes ≥ 9; a quiet
   position -> captures/checks/promotions only; `ours` restricts the ruler's node count on 000Sa (assert reasoned nodes ≤ ruler nodes) and the
   ruler without `ours` is unchanged (test_ruler.py still passes).
No AI trailers; one commit per piece (ruler hook; reasoned.py + tests; stream + batch); numbers only from OBS lines; timestamps from `date`.
A reasoned search that finds FEWER answers than the ruler is expected: report which and the reason that cut the answer -- that is the
finding (the missing reason is the next lemma), not a failure to hide.

## 52. THE CORE AND THE AGENDA: state, goal regression on questions, the generic binder as the verifier (Kaveh 2026-10-17, GO; built by a cheaper agent)

### 52.1 The design (Kaveh's redraw, 2026-10-17 afternoon; restated and agreed in chat)
- CORE. Layer 1 = rules + vocabulary + the board as a graph that answers attack / reach / COUNTERFACTUAL questions (place or remove a piece,
  re-read). Layer 2 = the STATE: all of our reasoning about one position = relations in the order of attention (kings, then by value, ours
  before theirs, pawns last), CLAIMS each backed by a proved lemma, and admitted GAPS. MEMORY = the store of states indexed by relations +
  claims (remember.py is its seed). Two tracks on the core: INFERENCE and TRAINING.
- INFERENCE = GOAL REGRESSION on an AGENDA of QUESTIONS, never a search of moves. Root questions are the goals in order of STAKE: mate their king
  (stake 100), win their queen 9 / rook 5 / minor 3 / pawn 1, save ours (the biggest of ours that hangs). A question is a FUTURE STATE as a
  partial pattern ("our queen on h2, their king on g1"), asked as a counterfactual of layer 1; its answer is a DEFECT LIST read from the rules
  (for mate: checker can be taken / a flight exists / an interposition exists / no check; for material: defended k times, see ≤ 0). Each defect
  GENERATES sub-questions (guard the square: who x-rays it, what blocks the x-ray; remove the blocker; cover or block the flight; lure the king).
  TEMPO ARITHMETIC prunes: a future state with d defects needs d moves of ours; the opponent gets d−1 free replies unless each of our moves is
  FORCING (a check, or a bigger threat); a question whose fixes cannot be made forcing dies WITH THAT REASON. A ROUTE = our move that supplies a
  fix with a forcing constraint; it is VERIFIED by the quantified tail over every reply (the generic binder below). STOP: (a) the best answer's
  stake beats every open question (a mate ends it); (b) tempo arithmetic leaves nothing open; (c) budget hit = a GAP, never a guess. Among equal
  stakes memory orders (NOT in this slice).
- THE GENERIC BINDER = the verifier. A schema is DATA: variables typed by legs, a premise over words, a tail, the Lean theorem's name. Binding:
  legs GENERATE assignments (x over our piece squares, y over x's legal targets ...), atoms prune under one `after`, the tail is read last; a
  surviving assignment is a binding whose guarantee is the theorem's conclusion; the Lean instance is the receipt. Memory never adds or removes
  a binding.
- TRAINING (not in this slice): a surprise = an observation contradicting a claim, or the evaluation moving where no claim was made; the record
  = state+claims, window to the observable, actual line, failed claim or gap; propose, prove, append, rerun inference on the same state.

### 52.2 The first slice (what to build now; nothing else)
New modules in catspace/incubator/analyst/discovery/ (the old strata are NOT refactored; puzzle_apply/selfmine stay as they are):
- `state.py`: `State` (dataclass: fen, us, pieces (the level-one rows as in puzzle_apply.state_lines, reuse that code by import), claims
  (list of Claim(kind, what, value, lemma, line)), gaps (list of str)); `state_of(fen, us) -> State`; `State.tokens()` = remember.py's
  signature tokens so one object serves the stream and the store.
- `graph.py` (layer 1 questions over python-chess + the Rust words where they exist: `attackers`, `defenders`, `see`, `xray`, `without`,
  `mobility`): `attackers_of(board, sq, colour)`, `flights(board, king_sq)`, `xrays_to(board, sq, colour) -> [(piece_sq, blockers)]`,
  `with_piece(board, piece, at, removed_from=None) -> board` (the counterfactual placement), `without(board, sq) -> board`. Pure functions.
- `regress.py`: `Question` (goal, stake, pattern (dict of placements), defects, fixes, parent, status, reason), `mate_defects(board, us,
  placement) -> list[Defect]` read from the rules (check? takeable? flights? interpositions?), `material_defects(board, us, target_sq)`
  (defenders, see), `fixes_of(defect) -> list[Question]` (guard / remove blocker / cover flight / lure, each as a sub-question), `tempo_ok(q)`
  (d defects vs forcing fixes), `routes_of(q, board) -> list[Move]` (our moves supplying a fix with a forcing constraint: a check, or a capture
  / threat worth ≥ the stake of the thing it leaves). Every Question carries `why` strings for the trace.
- `binder.py`: `Schema(name, vars, legs, atoms, tail, need, theorem)`; `SCHEMAS` built from the existing proved families (mate_in_one,
  mate_in_two, delayed_root_d1, delayed_root_max_d1, delayed_balance_d1 -- read their premises from selfmine's existing renderers rather than
  retyping them); `bind(schema, board, us, candidates=None) -> list[Binding(assignment, value, theorem)]` with legs as generators, atoms
  pruning, tail last (evaluated with the Rust `vocab_eval` the way best_bind does today); `verify(route, board, us) -> Binding | None` = try
  the schemas from the smallest up, stop at a mate.
- `agenda.py`: `Agenda(state)`: `questions` ordered by (−stake, depth), `ask()` loop: pop, compute defects, if none -> candidate routes ->
  verify -> a Claim on the state (value, lemma, line) ; else generate sub-questions if tempo_ok else kill with reason; stop rules (a)(b)(c);
  `trace()` = the agenda printed top to bottom: `Q3 mate on h2 (stake 100): defect checker takeable -> Q4 guard h2: Bd6 x-rays through e5 (our N)
  -> Q5 remove e5 with tempo: Nf3+ (check) -> verify mate_in_two binds (3 replies) -> CLAIM mate, lemma mate_in_two, line Nf3+ ...`.
- `analyse.py` CLI: `python -m catspace.incubator.analyst.discovery.analyse --set dev_20 --id 01IFU` prints the state (order of attention), the
  agenda trace, the claims, the gaps, and one OBS line: `OBS agenda <id>: questions asked n, killed by tempo k, claims c, decision <san> by
  <lemma>, answer <san> MATCH|MISS, seconds s`.
Budget: one principled budget only (questions asked ≤ the number of our pieces × 8 squares around their king + our captures), hit = gap.

### 52.3 Acceptance (in this order; Kaveh's rule: trace on 2-3 positions BEFORE any batch)
1. 01IFU: the trace reads the battery analysis Kaveh gave in chat: mate on h2 asked (queen placement), ONE defect (checker takeable), the x-ray
   from d6 through our own knight on e5 found, the tempo constraint (a knight CHECK), Nf3+ the one route, mate_in_two binds over Kh1 / Bxf3 /
   gxf3, decision Nf3+ MATCH. Print the trace verbatim in the report.
2. 000Sa (in check; answer Bxc2): the save-ours goal first (king in check), the two captures as routes, the material schema binds Bxc2, the
   agenda says why Kxc2 is not chosen (their Qc4+ then Qxe4: the opponent's view as a defect of the Kxc2 future) -- if the slice cannot say
   this yet, say so exactly.
3. 06J6T (answer Rxe5): report what the agenda does, found or not, and the reason that stopped it.
4. tests/test_regress.py: mate_defects on the 01IFU queen-on-h2 counterfactual returns exactly [checker takeable]; after `without e5` it returns
   []; flights of the g1 king are [] ; tempo_ok false for a 2-defect pattern with no forcing fix; bind(mate_in_two) on 01IFU returns e5->f3 only.
5. Then the dev_20 batch (one OBS line per puzzle + tally), ~minutes; report which puzzles the agenda decides, which it leaves as gaps, and the
   most common killing reason.
No AI trailers (no Co-Authored-By / Claude-Session lines); one commit per module; numbers only from OBS lines; timestamps from `date`; never a
relative data path; python-chess for counterfactual boards is fine (the rules), the GUARANTEE always comes from a proved schema's binding.

### 52.4 The regression from THEIR side (Kaveh 2026-10-17, GO; second slice)
"We have to do the regression from their side." The same Agenda, run for the opponent, in two places:
- THREATS AGAINST US = their agenda on the CURRENT board as if we passed (turn flipped, en passant cleared; not when we are in check): each
  claim they get (a mate; a material gain bound by a schema from their side) is a THREAT with the claim's value, and each becomes a root of OUR
  agenda: `save our B on e4: their threat Qc4+ then Qxe4 wins 3 (their claim, lemma delayed_balance_d1)` with stake = the threat's value (mate
  threats stake 100). The `hanging_of_ours` save-root stays as the degenerate one-move case. The stream prints their agenda's claims as
  `their view:` lines under the root.
- DEFECTS OF OUR FUTURES = their agenda on the board after each of our ROUTES (their turn, no pass): their claims are defects of that future;
  a route's VALUE = our bound value minus their best claim's value; a route whose future gives them a claim ≥ our stake is killed with the
  reason (`Kxc2: their Qc4+ then Qxe4 wins 3 (lemma delayed_balance_d1) -- the future loses the bishop`). This REPLACES the `opponent_defects`
  heuristic: ties and preferences are now read from bound schemas. Keep `opponent_defects` only as a fallback label when their agenda finds
  nothing (say "no claim of theirs binds").
- DEPTH: their agenda runs with `side_depth = 0` (it does not run our agenda on its routes); one level of the opponent's view, principled by the
  tempo arithmetic, not a count. Mate roots + win roots + their save root; the same budget rule (hit = gap, printed).
- COST: print per puzzle the seconds and the number of their agendas run; if a puzzle exceeds ~60 s say so (no caps; the plan decides later).
Acceptance: (1) 000Sa: Kxc2 is demoted by THEIR CLAIM (bound schema), not by the fork heuristic; trace printed. (2) a puzzle in dev_20 where a
piece of ours is NOT hanging now but falls to a forcing sequence gets a save-root from their agenda (find one; print it). (3) tests: their agenda
on the 000Sa position after Kxc2 claims the bishop; on the position after Bxc2 claims nothing; the pass-threat on a quiet position with a
two-move threat finds it. (4) dev_20 batch OBS line; report decided/MATCH and seconds against the previous 9 / 15 s. No AI trailers; commits per piece.

### 52.5 Their claims by the lemma that explains them: the depth-two delayed schema in the binder, the deep tail out (Kaveh 2026-10-17, GO)
Qc4+ on 000Sa is "check and attack the loose bishop; no king reply saves it" = the DELAY operator applied twice (plan 44): our named first move,
every reply quantified, our named second move (the capture), the exchange reading -- the proved depth-2 delayed schemas (`delayed_root_d2`,
`delayed_root_max_d2`, `delayed_balance_d2`, whichever exist in Schemas.lean; check `proved_schemas()`). Their weakness (a second named move is
illegal under many replies) vanishes after a CHECK: the reply set is the king's moves. So:
- binder.py: add the depth-2 delayed schemas to SCHEMAS, asked only on routes that are CHECKS (legs can hold) with the second move = each capture
  / threat execution available after the check (use puzzle_apply.depth2_candidates's shape or generate: first move the route, second move every
  capture of ours that is legal after EVERY reply); value = T − B as for d1; the binding names both moves in its line (`Qc4+ then Qxe4`).
- Their agenda (side_depth = 0) tries: mate, mate2, d1 families, d2 families on checks. The deep tail is REMOVED from the binder (keep the theorem).
- Our agenda gets the same d2 schemas on checking routes (it is the same binder); report if any dev_20 decision changes.
Acceptance: (1) 000Sa: `verify Kxc2 ... their view: their Qc4+ then Qxe4 wins 3 (lemma delayed_*_d2)`, printed verbatim; Qa4+ does NOT bind at d2
(Kd3 guards e4) -- print its note. (2) tests: Black's agenda after Kxc2 claims the bishop by a d2 lemma; after Bxc2 claims nothing; Qa4+ fails.
(3) dev_20 OBS line (previous: decided 9, MATCH 9, their agendas 22, seconds 22.6). No AI trailers; commits per piece; Lean untouched unless a d2
schema is missing (then say so and stop).

### 52.6 The promotion goal, both sides (autonomy night 2026-10-17; built by a cheaper agent after 52.5)
Kaveh: "if they are two moves from promoting and my king is two moves away, that's forcing: I have to run back. A quick look sees it. We need to
ask: who can stop this?" A root for EACH side's agenda:
- ROOT `promote our P on <sq>` (stake 8 = queen less pawn) for every pawn of the mover with `pushes` ≤ 3 (a principled bound: the mate/material
  stakes dominate farther pawns; say so in the trace). Its DEFECTS, read from the rules on the pawn's path (`ahead`) and promotion square (`promo`):
  `blocked` (a piece on the path), `path attacked` (a square of the path attacked by them more than we defend it: `attackers`/`defenders`),
  `stopper` (a piece of theirs that reaches the promotion square or the path within the pawn's `pushes`: their king by king distance `dist`,
  a slider/knight by `reach` within k -- use the words, not python-chess paths), `promo square attacked`.
- FIXES: `blocked` -> capture/deflect the blocker (sub-question); `stopper` -> cut the stopper's route (interpose / attack it with tempo) or
  push with tempo (a push that checks or attacks); `path attacked` -> add a guard / push past. TEMPO ARITHMETIC as everywhere: the pawn needs
  `pushes` moves; each stopper's distance ≤ pushes is a defect that costs them nothing (they move for free while we push) -- a stopper in time
  kills the question with the reason `their K reaches <promo> in 2, the pawn needs 2: stopped`; no stopper in time = `unstoppable`.
- VERIFIER: the proved lemma. `catspace.incubator.analyst.lean_passer_v2_proto.proved_passers(board) -> [(sq, plies)]` is the Python mirror of
  `Ideas.Passer2.unstoppablePasser2_sound` (bare-king form; read its hypotheses in lean/ideas/Ideas/Passer2.lean and the proto); a claim
  `promote` is made ONLY when it fires (lemma name `unstoppablePasser2_sound`, value 8, line = the pushes). When the defects say `unstoppable`
  but the lemma does not fire (pieces on the board the bare-king form excludes), the question ends as a GAP naming the hypothesis that fails --
  that gap is the next lemma to prove, print it.
- THEIR SIDE: the same root in their agenda; their `unstoppable` claim becomes our `save` root `stop their P on <sq>` with stake 8, whose routes are
  the stoppers' moves (the king runs back) and captures of the pawn; verified by the binder as any save root; if nothing binds, the trace still
  says `forced: every move of ours is spent stopping the pawn` when the tempo arithmetic is exact (distance = pushes).
- TRACE first on two positions: a bare-king race the lemma fires on (make one: white Kf6 Pe6 v black Kc3, white to move -> `unstoppable`), and
  the same with the black king on d4 (`stopped: their K reaches e8 in ... `); then any dev_20 / fresh_60 puzzle whose themes include
  `advancedPawn` or `promotion` (grep the sets' `themes`); print the traces.
Acceptance: (1) the two traces verbatim; (2) tests: unstoppable fires + claim; stopped kills with the distance reason; a blocked pawn names the
blocker; (3) dev_20 and fresh_60 OBS lines (previous dev_20 after 52.5 -- take the builder's number). No AI trailers; commits per piece; no Lean
changes (a failing hypothesis is reported as the gap, not patched).

### 52.7 The interposition that only delays (autonomy night 2026-10-17; after 52.6)
06xEN (Rh1+ Rh2 Rxh2#): the mate question dies at `defect interposition (a piece can step between on h2)` with no fix. The interposition is a
defect of the one-move pattern but it costs THEM the interposer: our checker takes it, with check again, and the pattern may now be a mate. Generic
fix for an `interposition` defect: the sub-question `take the interposer on <sq> with check`, whose future is the counterfactual (the interposer
placed on sq, then our checker captures it) read by `mate_defects` again; it is a forcing fix (a check). The same for a `checker takeable` defect
by a NON-king piece: the sub-question `recapture on <sq>` (our piece takes back) when the recapture is itself a check. VERIFIER: the route is the
root's own move; `verify` already tries `mate_in_two` on it -- the agenda must try the route's verification when every remaining defect has a
forcing fix (it already does for `guard`/`remove` fixes; extend to these). If the counterfactual after the capture still has defects, the question
dies as today with the reason. Also the save-king root in check: when no capture of the checker exists, the evasions are routes already.
Acceptance: 06xEN trace reads `defect interposition on h2 -> take the interposer on h2 with check: Rxh2 -- the future is a mate -> verify Rh1+:
binds mate_in_two (1 reply: Rh2)`, decision Rh1+ MATCH; tests; dev_20 OBS (expect decided 10).

### 52.8 Deciding when nothing gains: the parry by the smallest loss (autonomy night 2026-10-17; after 52.7)
Today a `save` root whose parries bind no gain ends as a GAP and the puzzle is undecided (052Wm Kf8; 06J6T Q3). Kaveh's flow (2026-10-16): in
check or under a threat, read each candidate FROM THE OPPONENT'S VIEW and take the one after which they have the least. We now have their agenda:
a route's `net` = our bound value (0 when nothing binds) less their best claim on the future. So: when a `save` root has routes and none binds a
gain, the agenda DECIDES the route with the best net (the smallest loss of theirs claims), ties broken by the shape key puzzle_apply already uses
(fewest hanging, most guarded, most poking); the claim is `save <san> (net −k)` with `lemma none: the opponent's best claim is k` and it ranks
BELOW any gain claim. Only for `save` roots (a threat or a check is a reason to move); never for `win` roots (no gain = no decision). Print the
reason: `decision Kf8: their best claim after it is 0 (after Kd8: Qxd5 wins 1 ...)`.
Acceptance: 052Wm decided Kf8 by the smallest loss (print the trace); 000Sa unchanged; dev_20 OBS; fresh_60 OBS; tests. No AI trailers; commits
per piece.

### 52.9 Memory orders the agenda among equal stakes (autonomy night 2026-10-17; after 52.8)
The store of states (remember.py, plan 50) found nothing to save on the exhaustive ladder. The agenda is where order matters: among roots of EQUAL
stake and among a question's fixes, ask first what memory says paid off in similar states. Build: (1) every decided puzzle writes a record (the
existing make_record + the agenda's claim: goal kind, the fix kinds along the claim's path, the lemma); (2) `recall` on the new puzzle's state
tokens returns the top-k records; (3) the Agenda's ordering key becomes (−stake, −recall_score, depth) where recall_score = the summed similarity
of recalled records whose goal kind / fix kind matches the question; stakes and stop rules unchanged; memory never adds or removes a question or
a route. Measure (one run each, store from dev_20, recall on fresh_60 with `--no-memory` as the control): questions asked until the deciding claim,
with vs without; decisions must be identical (print `OBS memory agenda: decisions identical yes/no`). Acceptance: the OBS line; a trace where a
recalled record moved a question up, printed. No AI trailers.

### 52.10 The trapped piece, and the leaf read by their agenda (autonomy night 2026-10-17; after 52.9)
0UqOv: `delayed_balance_alive_d1` claimed Qd5 "wins 1"; the queen takes a8 and is trapped two plies beyond the window (fxe6 Qxa8 Qc7). A
one-leaf guard cannot separate it from 000Sa's Bxc2 (both leave a piece attacked; one is defended). Two generic pieces, no new Lean:
- THE TRAPPED PIECE is the depth-two delayed schema on a QUIET first move: our move attacks a piece whose `mobility` (its safe squares, a word) is
  0 after it, so the second named move (the capture) is legal under every reply unless they defend or interpose, which the exchange leaf reads.
  binder: try the d2 schemas also on routes after which the target has `(count (mobility T))` = 0 (not only on checks); the binding line reads
  `Nb6 then Nxa8 (the queen has no safe square)`. regress: a `win` question's defect `can run away` (the piece has safe squares) gets the fix
  `take its squares` (sub-question) besides `attack with tempo`.
- THE LEAF READ BY THEIR AGENDA: for every balance claim of ours, follow the claim's principal line (their reply that minimised, our move that
  maximised -- best_bind can return them: extend its result with the line, or recompute with the words `least`/`most` argmin/argmax in Python over
  python-chess legal moves calling the same tail) to the leaf position; run THEIR agenda there (side_depth 0); if their best claim at the leaf
  is ≥ our claimed gain, the claim is REFUTED WITHIN ONE VIEW: downgrade it to a gap `refuted at the leaf: their <claim> (lemma ...)` and do not
  decide by it. This is not a guarantee (one leaf, one view) but a refutation check: evidence tags together (Kaveh 2026-10-17). Print the count
  of claims downgraded and, for each, Stockfish's view of the old decision vs the answer as `OBS leaf <id>: ...` (as in the transient audit).
Acceptance: 0UqOv no longer decides Qd5 (trace printed); 000Sa still Bxc2; dev_20 and fresh_60 OBS before/after with every changed decision
classified wrong/right by Stockfish; tests. No AI trailers; commits per piece.

### 52.11 A fix must do what it says; the escape note by the rules (Kaveh 2026-10-18 morning, puzzle 00ct0)
Found on a random fresh_60 puzzle (00ct0, Black to move): Q1 "mate on h5" listed `cover the flight g1: quiet fixes (Bh3)`. Bh3 does not cover g1
(the e6 bishop is light-squared, g1 is dark) and it BLOCKS our own check from h5. It passed because `regress._still` tests only "is the named
defect's key absent after the move": after Bh3 the counterfactual's defects collapse to `no check`, so `flight g1` is absent. A proxy, not the fix.
Second finding (profile of the same run, 28.6 s profiled / 10.6 s plain): 24 s is `agenda._escape_note` -- "no mate in one after ..." for every
failed mate route, by `selfmine.bind_mate` over all our moves after every reply (5,030 bind_mate, 197k premise_truth calls), 16 s of it Python
rendering the premise text (`base_vocabulary.sexpr`) before each Rust call; Rust reads 2.4 s. Most calls are inside their agenda and the leaf read,
where the note is never printed.
Build (regress.py, agenda.py):
1. `_still(q, board, us, m, root)` for a mate root becomes TWO conditions, both read on the counterfactual after m (placements made, their move):
   (a) the POSITIVE condition of the fix's goal holds -- `cover`: a piece of ours attacks the flight square `q.pattern["target"]` (rules:
   `cf.is_attacked_by(me, target)`); `guard` (checker takeable): the checker's square is attacked by another piece of ours or the taker is gone;
   `remove` (interposition / blocker): the named piece on `q.pattern["blocker"]` (or the interposer squares in the defect's `by`) is gone or can no
   longer step there; `take the interposer`/`take the checker` keep their existing reads; (b) NO NEW DEFECT: every defect key after m (kind, sq) is
   in the set before m, less the fixed one (`no check` is never allowed: the pattern must still check). A move failing (a) or (b) is not a fix.
   For a `win` root keep the current read but add (b) with `material_defects`. One doc line in the module docstring: "a fix is a move that does
   what the fix names and introduces no new defect; the disappearance of a defect for another reason is not a fix".
2. `_escape_note`: mate in one after a reply is a rules fact -- `any(c2.is_checkmate() after each of our legal moves)` on a copy; keep `bind_mate`
   only to confirm the hit on the ONE line printed (our agenda, `side_depth 0`), never inside their agenda or the leaf read (`_their_agenda`,
   `_leaf_read` pass a flag or the note is skipped when `self.side_depth > 0`). No change of any decision is expected from 2.
3. Cache the rendered premise text: memoize `base_vocabulary.sexpr` on the term's identity/tuple (an `lru_cache` or a dict keyed by the term)
   if the term is hashable; otherwise skip this item and say so.
Acceptance: 00ct0 trace shows Q1 `cover the flight g1: no fix at all` (no Bh3) and the decision is unchanged (Rxf4); `OBS agenda 00ct0 ... seconds`
before/after printed; dev_20 and fresh_60 (`fresh_60_threats_event`) rerun with `python -m ...analyse --set <set> --quiet`, every changed decision
listed; tests: one for `_still` rejecting a move that blocks our own check / does not attack the flight, one for the escape note by the rules
equal to the old reading on dev_20. Commits per item, no AI trailers.

### 52.12 The chase, the fork, the reach, and the read set (Kaveh 2026-10-18 midday; GO)
Fifteen fresh_60 puzzles were read by hand after 52.11. The agenda made no false claim; the misses share one gap: THE FIX EXISTS ONLY AFTER THEIR FORCED
REPLY, and the agenda never asks again after a forcing move. Lines (Stockfish d20, OBS SF lines in the journal): 0Wih8 Ng5+ Kh8 Nxf7+ Kh7 Qg6# (and
hxg5 Qh3+ Nh5 Qxh5#); 08pSj Rf1+ Kh2 Bf4+ (the king is driven onto the bishop's line: check + attack on the queen); 01w7J Bxh2+ Kf2 Qxf6+ (the king is
forced to block its own rook's defence of f6); 0RhU6 Qxf2+ Kh1 Qf1+ Rxf1 Rxf1# (the second check deflects the guard); 0VSUT Qxe5 Rxe5 Rxb1 (Rxb1 Rd8+
Re8 Rxe8#: the recapture pulls the back-rank rook away; the bishop's guard is overloaded); 0DoFB Qd5 (threatens Qxg2#: the b7 bishop covers g2 once
the queen leaves d5 -- the battery; "mate on g2" is never a root because roots are placed ONE move away). A second, systematic proxy bug: 52.10's
"take its squares" read `mobility` with their king IN CHECK, so every check "trapped" every piece (02XyH, 01w7J, 08pSj, 0RhU6, 0DoFB traces).
Kaveh's reframing (chat, 2026-10-18): mate IS the trap of the king; one TRAP question for every piece; the CHASE does not stop at the first check;
after every forced reply ask the WHOLE agenda again (the king's forced path may open a skewer, a fork, a block); CACHE answers by the base
relations a question read ("nothing changed for the rook"); a check that also attacks a piece is NOT a trap but a FORK ("two threats, one move").
Build, in this order, one commit each (builder: deviate minimally where the code disagrees with this text and REPORT the deviation):
1. **Trap vs fork (vocabulary rule).** `regress` 'take its squares' keeps only QUIET routes (the piece has no safe square, read with no check on the
   board: `G.mobility_of` on the board after our move is their move and not in check -- assert it); a CHECK that attacks a piece of theirs becomes the
   fix "fork: check and attack, one reply", verified by the existing depth-two schemas (check, every reply, the capture), binding line
   "Ng5+ then Nxf7 (a fork: they must answer the check)". Vocabulary: Lean `def fork` in lean/ideas/Ideas/Base/Vocabulary.lean with the gloss
   "**fork**: after our move that gives check, a piece of theirs other than the king is attacked by the moved piece: two threats, one reply" (a def,
   no theorem), the same word in Rust vocab.rs if the vocabulary words live there (else say so), one line in catspace/research/docs/GLOSSARY.md for
   `fork` and `trap` ("trap: a piece whose every square is unsafe, read with no check on the board"), `python -m catspace.incubator.analyst.lean_index
   build`. The trace must never again say a check "traps".
2. **The chase.** New fix kind in agenda.py, "chase: make them move", generated for a question whose defects have no forcing fix of ours or whose
   routes all failed to bind: our FORCING moves on the current board (`forcing_why`: check, capture >= stake, threat >= stake) are the routes; for each
   route take EVERY reply; on each reply position run the WHOLE agenda again (an inner `Agenda`, same side, `chase_depth + 1`, memoised by FEN across
   the whole run); the route's value = min over replies of (balance after the route and the reply minus balance now, read by the vocabulary word
   `balance`, plus the inner agenda's best claim value); a reply with no inner claim and no material gain kills the route at once (print which reply);
   the route CLAIMS when its value >= 1 (or the inner claims are mates: value 100). Line = the tree: "Rf1+; after Kh2: win the Q on d6 -> attack d6
   with tempo: Bf4+ binds (value 9)". Natural end: a level with no forcing move, or a failing reply, ends the chase; cycles end by the FEN memo.
   BUDGET = GAP (plan 52 stop rule c): the chase's budget is the root's question budget shared with the inner agendas; when it is hit the open chase
   becomes a GAP line "chase budget: <route> not followed past <reply>", never a kill, and the count is printed. Guarantee: step_after (our named
   move) + step_least (every reply) over the inner claims, each a bound schema; no new Lean theorem in this slice (note the composed statement to
   prove as a TODO in Schemas.lean).
3. **Read sets.** Every Question records `reads`: the squares whose occupancy or attack it read (target, attackers, guards, flight squares, path and
   line squares, the pattern's placement squares). After a move (the chase's route + reply) the inner agenda receives the DIFF of the state (squares
   whose piece or attacker/guard counts changed, from `State.pieces` rows before/after); a question of the outer agenda whose `reads` does not meet
   the diff is INHERITED (same status, claim or kill, one line "inherited: nothing changed for the R on b8"); only questions meeting the diff are asked.
   OBS per puzzle: inner agendas, questions inherited, questions reopened.
4. **The reach.** Mate roots at any distance: for each piece of ours and each square from which it would check their king, if the piece needs more
   than one move to get there (`graph` distance, blocked squares counted as the rules say) the root has the defect "reach: the Q needs 2 moves (via
   d5)" whose fix routes are the first steps of the shortest paths; such a step is FORCING when, after it, the mate pattern has no defect but the
   last step (a mate threat: their every reply must meet it) -- extend `forcing_why` with "threat of mate" read by `mate_defects` on the counterfactual
   after the step (only the reach defect left). The chase then regresses through their replies (0DoFB: Qd5; Qh3 or any parry; Qxg5 wins 3).
   Keep the roots bounded by the rules: distance <= 2 only when the stake is a mate (one prep move), say so in the trace.
Acceptance (traces in chat): 0Wih8 Ng5+, 08pSj Rf1+, 01w7J Bxh2+, 0RhU6 Qxf2+, 0VSUT Qxe5, 0DoFB Qd5 decided (each by the chase or the reach, every
inner line printed); 02XyH no "traps" on a check; 000Sa still Bxc2, 01IFU still Nf3+, 052Wm still Kf8; dev_20 and fresh_60_threats_event OBS
before/after, every changed decision listed and classified right/wrong by Stockfish d18 (OBS SF lines); seconds per set; tests for 1 (a check is not a
trap; a fork binds on 02XyH), 2 (0RhU6 three-check chase), 3 (an inherited question on a constructed position), 4 (0DoFB root). No AI trailers.

### 52.13 Fewer questions, exact caching, their agenda on every static leaf, the hot loop in Rust (Kaveh 2026-10-18 evening; GO "do it all")
Three findings on the 52.12 build, each reproduced (scratch traces tr3_0AgL8*.txt, tr3_0DoFB.txt): (1) the FALSE CLAIM 0AgL8 Nxg2: 35 replies did not
recapture, each was credited "material +1" (balance with us to move) and the inner agenda, told to read for a mate only, asked nothing from THEIR side;
Bc6+ then mates in 4. Caught only on the chosen route by `_static_refuted`. (2) "Rust 28x slower" was false: OBS the Rust evaluator 0.4 us, the Python
balance 2.2 us, the Python glue that copies the board and renders a FEN 15.4 us per call. (3) THE BUDGET (our pieces x 8 + captures: 85 on 0AgL8, 76 on
0DoFB) is spent by ONE inner agenda: it re-asks the whole board (20-25 questions) on one reply position; 0DoFB's first route Bxf2+ Kxf2 eats it and the
answer Qd5 (fourth route) is never followed; GAP routes are not memoised and every root question re-chases them with a fresh budget (894 hits).
Kaveh's two ways out: everything in Rust; a shortlist of their replies, obviously bad ones excluded by a reason that stands while nothing about them
changed, the rest ORDERED by a live ranking, "memoize and cache as much as possible". Correction agreed: exclusion by a probability threshold is out
(absolute logic only); EXCLUSION IS A KILL WITH A READ SET, a learned prior may only ORDER. Build, in this order, one commit each (builder: deviate
minimally where the code disagrees with this text and REPORT every deviation; no AI trailers; OBS lines for every number):
1. **The diff decides what is asked.** An inner agenda (chase_depth >= 1) asks only: (a) the question that sent it there (same `question_key`), (b) every
   outer question whose `reads` meets the diff of the route and the reply, (c) every question that is NEW on this board (a key the outer agenda did not
   have). Everything else is inherited with its status: a KILL as now; a CLAIM too -- a claim records `reads` = the question's reads plus the squares named
   in the bound atoms of its schema (the binder knows them: route from/to, the squares of the terms it evaluated); the inherited claim is re-stamped on the
   inner state with the line "inherited: nothing changed for ...". A question left unasked by the outer agenda (stop (a) ended it) is asked in the inner
   agenda only if its stake >= `need`. OBS per puzzle: inner questions per inner agenda (mean), inherited claims / kills, reopened.
2. **Memoise failures; one budget per root agenda; a route must fit its stake.** (a) A GAP route is memoised in `ctx.routes` with the budget left when it was
   cut; seen again with no more budget it stays a GAP without being re-run (one line). (b) `ctx.left` is set ONCE per root agenda (the chase of a second root
   question continues the same count); the budget is now a FUSE at 10x the old count, printed as "fuse" when hit, and the natural end is printed otherwise:
   "chase ended: every route closed (claim on every reply or a refuting reply)". (c) A question is chased only when its stake > the best claim so far
   (stop (a) would end the agenda anyway). (d) `forcing_why`: a quiet move threatens only if the moved piece is not lost by the exchange on its own square
   (the exchange word, same as `hangs`): "Rc6 threat 9" where the queen takes the rook is NOT forcing (08pSj); the trace says "not a threat: the R on c6
   hangs". The hanging-piece rule is a vocabulary rule: cite the Lean def used (LEAN_INDEX search `exchange`, `hangs`); if none fits, add the def + gloss +
   glossary line + lean_index build.
3. **Their agenda on every static leaf.** In `_chase_route`, a reply after which the gain stands only as a balance (dbal >= 1, inner value < mate) is read
   AT ONCE by their agenda with the move passed to them (memoised by FEN in `ctx`): a claim of theirs of value t makes that reply's value dbal - t (and
   if < 1 the route dies at that reply: "after Bc6+: +1 stands only as a balance and their Qa4+ wins 100 if we pass"). `_static_refuted` is removed (it was
   the chosen-route-only form). The false-claim class is closed: no static balance is a result until their agenda has read it. Test: 0AgL8 never claims
   Nxg2 with the budget at 4x (the scratch reproduction repro_0AgL8.py shows the old claim).
4. **The shortlist of their replies.** Replies are read in the order: the reply with a standing kill from an earlier reply position (same key, reads disjoint
   from the diff: inherited, not read) is skipped and counted; then recaptures on the route's square, checks, captures, the rest; inside a class worst balance
   first (as now). "Obviously bad" = a reply after which the RULES give us a mate in one (is_checkmate over our moves) or a capture whose exchange wins >=
   `need` AND their agenda with the move passed finds no claim (memoised) -- such a reply is closed by that one rule, printed "closed by rule: mate in one /
   exchange +5, nothing of theirs", no inner agenda. Every (route, reply, outcome, reads, reason) is LOGGED to `paths.experiment("chase_log.jsonl")` (one
   row per reply read, appended, per-item) for the plan 53 ordering prior; the prior itself is NOT built in this slice.
5. **The hot loop in Rust.** A `#[pyclass] Pos` in analyst_core (from a FEN once): `push(uci) -> Pos`, `balance(owner)`, `replies() -> [(uci, dbal, check,
   capture, recapture_on_sq, mate_in_one_for_mover_after)]`, `diff(other) -> [squares]` (piece or attacker counts changed), `fen()`. `_chase_route` and
   `diff_squares` read through it (no FEN per reply); `G.balance` stays as the test mirror. Test: Pos.replies and Pos.diff equal python-chess on every
   dev_20 position and every reply of its first legal move. Timing OBS: seconds per set before/after. The Agenda itself stays Python in this slice; the port
   of the agenda is a separate plan (Kaveh's "everything in Rust") and is NOT started here -- say so in the report.
Acceptance (traces in chat): 0DoFB Qd5 decided by the reach (every inner line printed); 0AgL8 no claim of Nxg2 at 4x budget (reproduction script) and its
decision classified by Stockfish d18; 08pSj no "threat 9" route by a hanging rook; 0RhU6 Qxf2+, 0Wih8 Ng5+, 01w7J Bxh2+, 0VSUT Qxe5 still decided;
dev_20 and fresh_60_threats_event OBS before/after (decisions, MATCH, seconds, inner questions per inner agenda, fuse hits -- expected ~0, inherited
claims), every changed decision listed and classified by Stockfish d18 (OBS SF lines); the chase_log row count. Tests for 1 (an inherited claim), 2(d) (a
hanging rook is not a threat), 3 (0AgL8 at 4x), 4 (a reply closed by rule), 5 (Pos equals python-chess). No AI trailers.

### 52.14 Their threat is a save root of the inner agenda (Kaveh 2026-10-19; GO "do it")
52.13 (3) reads a static leaf (a reply after which our gain stands only as the balance, us to move) with THEIR agenda, the move passed to them, and
subtracts their best claim. The subtraction is wrong when we have a parry: at inner levels it refuted 0VSUT's true line ("their Rxb1 wins 4 if we
pass" -- but we are to move), so the builder restricted it to the root chase, and an inner-level static leaf is not read from their side at all.
The root agenda already has the right form (52.4a `_threat_roots`): a claim of theirs with the move passed is a THREAT and becomes a SAVE root of
ours with the claim's value as stake; the save is verified by the binder (a parry, a counter, taking the attacker) or killed. Build, one commit
(builder: deviate minimally and REPORT every deviation; no AI trailers; OBS lines for every number):
1. In `_chase_route`, at EVERY chase depth: a static leaf (dbal >= 1, inner value < mate, not in check) is read by `_their_pass` (memoised as now). If
   their best claim t satisfies dbal - t < 1, the reply is NOT killed: the inner agenda on that position is asked once more WITH their claims as save
   roots (`_inner(fen, need, diff, threats=<their claims>)`: the inner Agenda pushes the save roots exactly as `_threat_roots` does for the root agenda,
   stake = the claim's value, asked regardless of `min_stake`; the inner agenda stays side_depth 0 otherwise). The reply's value = dbal - (the sum of
   the threats whose save root is killed or a gap) + the inner best claim as now. A verified save prints "after Reb5: their Rxb1 wins 4 if we pass;
   parry: Rb8 (lemma ...) keeps +3"; a killed save prints "after Bc6+: +1 stands only as a balance, their Qa4+ wins 100, no parry" and the route dies at
   that reply when the value is below 1. Note `_inner` memoises by position: an answer asked WITH threats serves a later ask without them, not the
   reverse (key the memo by (position, with_threats)).
2. The `closed by rule` exchange closure (52.13 (4)) keeps its "their pass finds nothing" condition (it closes without an inner agenda only when there
   is nothing to parry).
3. OBS per set: static leaves read, parried, unparried (route killed). Test: 0VSUT Qxe5 decided with the inner static leaf read (the parry line printed);
   0AgL8 Nxg2 still never claimed at 4x the fuse (now by "no parry"); a constructed position where a threat is parried and the gain kept.
Acceptance: dev_20 and fresh_60_threats_event OBS before/after, every changed decision classified by Stockfish d18 (OBS SF lines), seconds; the
acceptance puzzles of 52.13 unchanged (0DoFB Qd5, 0AgL8 Qxg2+, 08pSj Rf1+, 0RhU6 Qxf2+, 0Wih8 Ng5+, 01w7J Bxh2+, 0VSUT Qxe5, 0BuQP Kf2, 0UqOv exf7+).

### 52.15 The checker as a win target, the chase for promotion goals, the parry as a bound schema (Kaveh 2026-10-19 "do the next thing"; GO)
The eleven undecided fresh_60 puzzles were read by hand (journal 2026-10-19 09:30): two sit behind a missing question in check, three behind
promotion goals that never chase; and 52.14's parry is the first claim path without a proved schema. Build, in this order, one commit each (builder:
deviate minimally and REPORT every deviation; no AI trailers; OBS lines for every number):
1. **In check, the checker is a win target.** The agenda in check roots only "save our king" (0Zsoa after Rxd6 exd6+: the save claim is Kd7 at 0 and
   Kxd6, a free pawn, "no route binds"; 0SJz3 after Rxd6 Rxd6+: the king takes back a rook). Add, when in check and the checker can be taken, the root
   "win their <checker> on <sq> (it gives check)" with the checker's worth as stake, routes = every legal capture of the checker (the king's included),
   verified by the existing schemas exactly as a win root is; the save root stays. In the chase this makes "after exd6+: no claim of ours" into
   "after exd6+: win the P on d6: Kxd6 (lemma ...) -> +1". Test: 0Zsoa Rxd6 and 0SJz3 Rxd6 decided (classified by Stockfish d18).
2. **Promote roots chase.** `_chase` allows the goals mate and win; add promote. Routes for a promote question: `forcing_why` as now (check, capture,
   threat of mate) PLUS a move that attacks a named STOPPER of the question (the defect's piece: 0Nv29 "b6" attacks the B on a5), read as
   "threat on the stopper"; the inner agenda on each reply asks the promote root again (its reads: the path squares, the stoppers, the promotion
   square). A route claims when the inner promote claim binds on every reply or the reply loses material >= the stake. Test: 0Nv29 b6 decided
   (every inner line printed); 0LRTF and 0GJec traces reported either way with the reason.
3. **The parry as a bound schema.** The Lean theorem `delayed_balance_d1` (Schemas.lean:153) is already general in the balance bound B; the BINDER
   hard-codes `balance <= 0` (`_delayed_atoms`, binder.py:64). Add a binder family `keep` whose atoms are `balance <= B` with B = the balance now and
   the tail `>= B + v` with v = 0 (the parry keeps what we have), bound by the same theorem if its floor `hT : T >= 68` admits it, else prove
   `delayed_balance_keep_d1` in Schemas.lean (the same proof with the floor the parry needs; say what 68 is in the gloss) and rebuild the index
   (`python -m catspace.incubator.analyst.lean_index build`). 52.14's `_parry_keeps` then binds its parry with that family and the claim's lemma is the
   theorem's name; the "their_view" label disappears from every trace (grep). A mate threat is still parried only by taking the threatening piece or
   by mating. Test: the 52.14 parry tests pass with a Lean lemma name in the claim.
Acceptance: dev_20 and fresh_60_threats_event OBS before/after, every changed decision classified by Stockfish d18 (OBS SF lines), seconds; the
52.13/52.14 acceptance puzzles unchanged; the Lean build green (`cd lean/ideas && lake build`) if Lean is touched.

### 52.16 The bishop as a stopper, the fuse order, the sacrifice that still threatens (Kaveh 2026-10-19 GO: "the bishop stopping the pawn and whatever else was left")
Left from 52.15 (journal 10:05): 0Nv29's b6 dies on Bb4 because `unstoppablePasser2_sound` has the hypothesis enemyKingPawns (their bishop is not
allowed); 0LRTF and 0GJec end as fuse GAPs because the mate chases spend the fuse before the promote chase runs; failed promote chases are costly
(02XVK 0.4 -> 5.2 s). From the discovery position Kaveh showed (r3q2k/pQ1nb1pp/6n1/5bP1/3PN3/2N1P3/PPP3B1/R1B3K1 w, SF d20: Nf6 +385, Nd6 +320):
the 52.13 (2d) rule "a hanging piece threatens nothing" reads every sacrifice as no threat ("Nf6: not a threat: the N on f6 hangs"), while the exchange
word already sees the battery (Ng3 / Nf2 / Nd2 are "threat 5": Qxa8 Qxa8 Bxa8 through the vacated e4). Build, one commit each (builder: deviate
minimally and REPORT every deviation; no AI trailers; OBS lines for every number; Lean by reasoning, never a brute-force enumeration of positions):
1. **The passer lemma with their pieces on the board.** Read `unstoppablePasser2_sound` and its hypotheses; prove in Lean a form whose stopper
   condition is per piece: the pawn needs n pushes; their king is outside the square (as now); every other piece of theirs cannot attack any square
   of the remaining path (the squares the pawn still has to cross, the promotion square included) within n - 1 of its moves -- state it with the
   vocabulary's reach / attack words (search LEAN_INDEX for `reach`, `attacks`, `pawnPath`, `passer`); if a bounded form is all that closes (n <= 3, their
   piece's reach read on the static board as the Python defect "stopper ... reaches the path in k" already does), prove that and say so. Gloss, glossary
   line, `lean_index build`, `lake build` green. The binder reads the new lemma where it read the old; the trace names it. Acceptance: 0Nv29 b6 decided
   (Stockfish d18 OBS), "enemyKingPawns" no longer the reason of any promote GAP in dev_20 / fresh_60 (grep the full traces).
2. **The fuse order.** Diagnose 0LRTF and 0GJec: which chase spends the fuse and on what. Fix the CAUSE with a reason, not a bigger number: e.g. a root
   question's chase gets a share of the fuse by stake order so a later root is never starved (say the rule in the trace line), or the check-chain inner
   levels that burn it are closed by a rule (a check chain that returns to a position is a cycle: verify the 52.13 cycle key catches it). Acceptance:
   0LRTF and 0GJec decided or their remaining gap named precisely; OBS fuse hits per set before/after.
3. **A sacrifice that still threatens is forcing.** `forcing_why` (2d): a quiet move by a piece that hangs is a threat when the threat SURVIVES the
   capture of that piece: after the move and their best capture of the moved piece (the exchange's capturer), `threat_value` with us to move is still
   >= the stake. Trace: "Nf6: threat 5 survives Bxf6 (the N hangs, 3)". 08pSj's Rc6 stays "not a threat" (after Qxc6 nothing survives). Test: the
   discovery position decides Nf6 or Nd6 (either is right by SF), printed with the discovery read ("after Nf6 Bxf6: Qxa8 wins 5: Bxa8 behind the Q"),
   and 08pSj still Rf1+.
4. **Failed promote chases.** Diagnose 02XVK (0.4 -> 5.2 s after 52.15 (2)): what the promote chase asks and why it cannot close early. Cut by a reason
   (e.g. a promote route whose reply leaves the stopper defect standing and no new fix is a kill at once; the inner promote root inherits the stopper
   kill by its read set), never by a count. OBS seconds per set before/after, and 02XVK's.
Acceptance: dev_20 and fresh_60_threats_event OBS before/after, every changed decision classified by Stockfish d18 (OBS SF lines); the 52.13-52.15
acceptance puzzles unchanged (0DoFB Qd5, 0AgL8 Qxg2+, 08pSj Rf1+, 0RhU6 Qxf2+, 0Wih8 Ng5+, 01w7J Bxh2+, 0VSUT Qxe5, 0BuQP Kf2, 0UqOv exf7+,
0SJz3 Rxd6, 0PBkh Kxf5); tests for each item.

### 52.17 The region of a piece, the goal that dies by reasoning, cheapest reply first (Kaveh 2026-10-19 on 0LRTF; written on "put this definition there"; NOT a build go)
From the walk through 0LRTF (2k5/2P4R/1K6/8/8/p7/8/2r5 w; SF: Rh8+ Kd7 Rd8+ Ke6 c8=Q Rxc8 Rxc8, a rook up) after 52.16. The agenda misses it
because the mate root's chase follows the first check downward (Re8+ Kd5 Rd8+ Ke4 ...) and spends the fuse before c8=Q+ at depth 2 is read, where
the exchange rule closes it in one reply; every root repeats the walk on a smaller share. Kaveh's reading: the mate goal dies in one look, by
reasoning -- a rook alone cannot mate a king in the open, our king would have to stand next to theirs, and our king cannot get there: it guards c7
and the rook on c1 holds the whole c-file. "The opponent's king can put more distance between us than I can follow"; "I cannot mate with a rook alone,
so there is no point chasing the king"; "it's not that my king is literally stuck -- it has no moves in the direction I need". The promotion then
reasons defect by defect: their king attacks c8 (fix: a check whose every king reply leaves c8 unattacked -- Rh8+ fails it, Kd7 still touches c8;
Rd8+ passes it), their rook controls c8 (fix: the exchange c8=Q Rxc8 Rxc8, the queen guarding d8 so the king cannot take the rook first), then
"do I have time to stop his pawn" (their agenda on the leaf: yes), then STOP, an out is found. Items, in build order (one commit each; Lean by
reasoning; deviations reported; no AI trailers):
1. **The region of a piece (the definition).** For a piece on square s, its REGION is the connected component of s in the graph whose vertices are
   the squares where the piece may STAND (empty, or holding a piece of theirs it may take, and the piece is not lost there by the exchange word --
   for a king: not held by any piece of theirs) and whose edges are the piece's single moves over empty squares. It is a set of squares with a
   distance (moves from s). The piece's move rule makes the walls differ: a king cannot enter a held square, so a rook's line is a wall for it
   (0LRTF: the c1 rook holds c2..c6, our king's region is the a and b files); a slider is stopped by occupied squares; a knight by nothing but
   unsafe landing squares; a bishop's region lives on its colour. The READ SET of a region is the pieces that hold or block its boundary. A goal's
   ZONE is the set of squares the goal needs the piece on: adjacency on the far side of their king for a mating king, the promotion square for a pawn,
   the mating line for a rook, the fork square for a knight. RULE: a goal is KILLED by "region meets no zone square" with the region's read set --
   sound as a kill because they need only keep the wall standing (one defence suffices for them). The CLAIM direction (we do reach the zone) is a
   race and stays with the Passer2 form (our moves against their tempo); never claim reach from the region alone. Lean: def `region` (the closure
   under the piece's step over standable squares), theorem: a rook (or queen) on a line it keeps, every square of which it attacks, is not crossed by
   a king (the king never enters an attacked square); the rook-alone mate: with the kings apart a rook and king never cover the box -- derive from
   `QueenMate.no_queen_mate_in_open` (a rook attacks a subset of what a queen attacks from the same square) or prove beside
   `InsufficientMaterial.no_mate_king_minor` with `king_guards_two_orth`. Gloss, glossary lines ("region", "zone", "wall"), `lean_index build`.
   Python: `region(board, sq)` mirrors the def with a test against the Lean statement; the trace prints the kill as "mate: killed by reasoning -- a
   rook alone never mates with the kings apart (lemma ...), and our K's region {a1..b8 minus ...} never meets the zone (the wall: their r on c1, our K
   bound to c7)". CORRECTION (Kaveh, same day): on 0LRTF our pawn on c7 shields c8 from the rook, so c8 is NOT held and our king's region (b6, b7,
   c8, then the board) DOES meet their king's region: the region kill does not fire there. The second kill, A MEASURE THEY CAN HOLD: the zone is
   reachable but their reply restores the measure every time -- for a mating king, the distance from our king to the zone next to theirs is held
   while their king has ROOM (its region is not boxed): a king never gains on a king with room (Lean: the evasion step -- a king at Chebyshev distance
   >= 2 from ours with a free square farther from ours keeps the distance after any one king move of ours; the negative of `Forced.toward_known`).
   Shrinking the room is the rook's work (K+R v K), and on 0LRTF the rook is also the only piece that stops their a-pawn (promotes in 2 moves: their
   agenda on the leaf), so the mate's tempo count is unbounded against their 2. Read set of the held-measure kill: their king's room, our king's
   distance, the pieces that could shrink the room. Both kills print as "killed by reasoning -- ..." and both are "do not chase" lemmas; a chase of
   the goal is warranted only when neither fires (a forcing move crosses the wall or shrinks the room). Acceptance: 0LRTF's Q1/Q2 killed before any
   chase (no inner agenda under them) by the held-measure kill, printed with its read set; a region kill shown on a puzzle where a wall stands
   (find one in the sets or construct it, SF-checked).
2. **A chase route must address a defect of its own goal.** The chase of a goal follows a forcing move only if the move is a fix of one of the goal's
   defects (covers the flight, drives the king off the square, removes or deflects the attacker, steps the piece toward the zone) or creates a new
   claim of its own (52.16 (3) forcing). A check that fixes nothing of the goal is not a route of that goal (the walk of checks). Name the reason in
   the trace: "Rh8+: not a route of mate h8 (fixes no defect: d7 stays a flight)". Expect fuse hits per set to fall; OBS before/after. The bound
   piece as a kill with a read set (our K must guard c7: "bound") is the reverse lemma Kaveh asked for; print it where it kills.
3. **"Drive the king off a square" as a named fix.** Defect "square X attacked by their king" gets the fix: a check after which every legal king
   reply leaves X unattacked by the king (read set: the king's squares after the check, the checker's safety). 0LRTF: Rh8+ fails (Kd7), Rd8+ after
   Rh8+ Kd7 passes (Ke6, Ke7; Kxd8 illegal, c7 guards d8). With 1-3 the promote root of 0LRTF resolves in two questions and no chase: Rh8+ is then a
   step of the promote goal (it does not fix the defect alone but makes Rd8+ legal: a two-step fix, the chase's business, bounded by rule 2).
4. **Cheapest reply first.** At every level of a chase, before any route opens an inner agenda, read every route's replies at the static leaf and
   close what closes by rule (mate in one, the exchange with nothing of theirs, a standing kill). Only then open inner agendas, in the order of the
   routes still open. A question answered on the board is answered before a question that needs a sub-question. OBS: 0LRTF decided Rh8+ with items
   1-3 off (this item alone), and with all four; dev_20 / fresh_60 seconds and fuse hits before/after.
5. **The budget is time (Kaveh 2026-10-19 GO, during the play server).** `Agenda(..., seconds: float | None)`: a deadline (`time.monotonic()
   + seconds`) kept on the ChaseCtx; every place that tests the fuse (`ctx.left <= 0`) also tests the deadline; past it, the open routes end as
   GAP "out of time: <san> not followed past <san>" (a GAP, never a kill), inner agendas are not opened, and the root agenda's remaining questions
   are asked only for what closes at the static leaf (no chase) so the best bound claim so far decides. The question-count fuse stays as a safety
   when seconds is None (the puzzle sets: unchanged behaviour, OBS identical). The trace prints "budget: 2.5 s, used 2.51 s, N questions"; the
   tally line gets "time outs". The setting is applied from outside: the analyse CLI gets `--seconds`, and the play server (plan 54,
   play_server.py / web.py / web_page.html) gets `--seconds` plus a float field on the Play page sent with `/api/play/new` and changeable between
   moves (`/api/play/settings`); the fallback runs after the agenda returns. Test: a chase that is cut by a 0.05 s budget ends with GAPs and a
   trace line naming the time; the same puzzle with no seconds is unchanged. Kaveh: the seconds bound the LIVE move only; what the clock cut is
   kept, not dropped: every "out of time" GAP is recorded with the question key, the route and the reply it stopped at (the 52.13 GAP memo already
   keeps cut_left; add the deadline mark), so the background tier of 52.19 (3) can resume exactly there on their clock.
Acceptance: 0LRTF decided Rh8+ (SF d18 OBS); 0GJec's gap re-read under rule 2 and named; the 52.13-52.16 acceptance puzzles unchanged (list in 52.16,
plus 0Nv29 b6, 04Qx2 Be8+, 08A6a Bxe5, the discovery position Nf6); tests per item; both sets OBS before/after, every changed decision SF-classified.
Deferred to 52.18 (not written): the landing table (target = (piece, square) whose landing binds a claim with no defect, threat = the same from
their side, the inverted read index), the battery / discovery words in Lean, the x-ray fix on win/promote defects, the fork square as a root,
threat-of-X forcing. Open question Kaveh raised: the region of a knight or a bishop -- answered above as the same closure under each piece's own step;
whether a region should also carry the squares the piece ATTACKS from inside it (its influence, for zones that are lines) is to be decided when 52.18
is written.

### 52.18 The landing table: targets and threats, the battery and the discovery, the fork square as a root, threat-of-X forcing (Kaveh 2026-10-19 GO "build 52.17 and also 52.18"; after 52.17)
Kaveh's definition (2026-10-19, after the discovery position r3q2k/pQ1nb1pp/6n1/5bP1/3PN3/2N1P3/PPP3B1/R1B3K1 w): "the target is a square where if my
piece lands, I win -- mate, a place from which mate is inevitable, a double attack, capturing a piece directly; we need the notion of a threat: if
the piece lands here, I'm lost; if the piece lands here, I win." And: a move creates a new threat from three sources -- the moved piece (fork), the
vacated line (discovery; a battery: the attacker count on a square the move never touches rises), or a broken defence (deflection, overload, pin).
"Keep every threat found and, when evaluating a piece, bring up everything known that involves it." Learned parts only ORDER; every claim and kill
is by rule with a read set (52.13). Items, in build order (one commit each; Lean by reasoning; deviations reported; no AI trailers; OBS lines):
1. **The landing table.** A LANDING is (side, piece, square). A landing is a TARGET of ours when, with the piece placed there and THEM to move (the
   counterfactual placement the roots already use: placement_of / future_board), a claim of ours binds with no defect left (mate in one / two, the
   exchange winning, promotion by the passer lemma, the fork claim of item 4, the trap). A landing is a THREAT of theirs when the same holds from
   their side. The table is not maintained: it is the memo of landing answers (claim or defect list, with the read set of the squares and pieces
   read), keyed by (side, piece, square) + the read set's pieces, filled when a question asks and invalidated by the 52.13 diff. The INVERTED READ
   INDEX: square -> the landings and kills whose read set contains it, so "what is known about their queen on d6" prints every landing that reads
   d6. API: `landing(board, side, piece_sq, to_sq) -> Landing(claim | defects, reads)`; `known_about(sq)`. The agenda's roots become landings
   (the mate roots, win roots, promote roots and the reach roots are each "a landing that binds"); say in the trace when a root is served from
   the table ("landing N on f6: target (win 5 by the exchange on a8 after the discovery), served from the table, reads {e4,f6,a8,b7,g2,e8}").
   No per-square control table (Kaveh: ask when needed; the exchange word orders attackers by value on the fly; key = (piece, square)).
   Test: the landing of the discovery position (N e4 -> f6) is a target with the discovery read; the 52.13-52.17 acceptance puzzles unchanged.
2. **The battery and the discovery, generally.** Today `xray` = squares attacked through one piece, `discovery` is king-only (base_vocabulary.py
   line ~168), `G.xrays_to` finds slider + blocker. Define in Lean with gloss + glossary: BATTERY (two pieces of one side on one line, the front one
   blocking the rear one's attack on a square beyond: `battery s1 s2 t`), DISCOVERY (a move of the front piece of a battery -- OUR piece as the
   blocker, as in Kaveh's position where the knight on e4 blocks Bg2 -> a8 -- that raises the attacker count on t by one: `discovered_attack`),
   ALMOST-A-BATTERY (a line from a slider of ours to a target with exactly one piece of ours on it, i.e. one move away from a battery). Prove: after
   the blocker's move (any move off the line), the rear piece attacks t (`pieceAttacksB_mono` / `empty_geom_reaches` in Ideas.Mate are the tools).
   Python mirrors with tests against the Lean statements; the trace names "discovery: the N leaves e4, the B on g2 joins the Q on a8 (battery)".
   Then the X-RAY FIX, which exists only for the mate defect "checker takeable by the king" (regress.py ~322-352), is offered on WIN and PROMOTE
   defects "defended" / "not attacked enough": the fix = move the blocker with tempo (a check, a capture, or a threat that survives the capture of the
   moved piece, 52.16 (3)). Acceptance: the discovery position decides Nf6 / Nd6 with the fix NAMED as a discovery (not only found by the forcing
   rule), OBS SF unchanged; a second position found in the sets or constructed where the discovery is the only way (SF-checked).
3. **Threat-of-X forcing.** `forcing_why`: a move is forcing when, after it, a PASS by them lets a claim of ours bind with no defect (today only the
   threat of mate and the material threat value): threat of promotion (the passer lemma binds after a pass), threat of a trap, threat of a fork
   (item 4). The reason names the claim: "Rd8+ (check)", "b6 (threat: b7 then b8=Q binds after a pass)". The pass is the null move on the Rust Pos
   (add `Pos.pass` if missing; the position after a pass is never a cycle key of the chase). Their side symmetric: their threats on the static leaf
   (52.14) use the same word. OBS: forcing routes per set before/after; decisions changed, SF-classified.
4. **The fork square as a root.** A root question "fork on s with our <piece>": the landing on s attacks two pieces of theirs of value >= the stake
   (or the king and one piece) and the landing is safe (exchange word); claim = the smaller of the two values (both cannot be saved in one move
   unless one move saves both: a capture of the forking piece, a move of one attacked piece that defends the other, a check -- these are the
   defects, each with a fix as usual); today a fork is only a fix of a win question. Lean: search the index (`knight_fork_same_colour`,
   `Fork.lean`, `localCheck_sound`) and bind the claim to what is proved; prove the two-threats-one-move lemma if missing (after our landing, one
   move of theirs changes at most one of {the attacked piece on a, the attacked piece on b} unless it is a capture of the forking piece, a check,
   or a move that guards the other -- by reasoning over the move's from/to squares). Acceptance: a puzzle in the sets decided by a fork root (find
   one: grep the traces for "fork" fixes), OBS SF; the knight-fork colour lemma kills fork roots on opposite-colour pieces with its read set.
5. **The region's influence (decide, then build if small).** Whether a region carries the squares the piece attacks from inside it (its influence),
   for zones that are lines (the mating line of a rook): if the goal zones of 52.17 need it, add `influence(region)` as the union of attacks from the
   region's squares with the same read set; otherwise record why not in the plan and skip.
   DECIDED (builder, 2026-10-19): NOT built. None of the zones the 52.17 kills use is a line the piece attacks from inside its region: the wall kill's zone is the squares from which OUR KING covers a
   flight (a king's cover is its adjacency, not a region property), and the held-measure kill reads their king's room and the rules, not a rook's line. A rook-line zone would need the influence
   `attacks from the region's squares` with the same read set; the first goal that asks for it (a mating line for a rook behind a wall) is where it gets built, with its Lean statement.
Acceptance for 52.18: dev_20 and fresh_60_threats_event OBS before/after, every changed decision classified by Stockfish d18; every 52.13-52.17
acceptance puzzle unchanged; glossary + LEAN_INDEX rebuilt; tests per item; `known_about` printed in the trace for the deciding move's piece.

### 52.19 The fact store with justifications: stale flags, a stake-ordered queue, two tiers of thinking (Kaveh 2026-10-19 GO "go ahead and build it"; after 52.18)
Kaveh's design (2026-10-19 afternoon): "anytime a decision is rendered we need a listener that listens for changes in the premises it used; that
way anytime the premises change the listener can figure out what needs to recalculate -- memoize more effectively"; "I will get the signal that
something changed and record it, but I don't need to process that change at that moment -- a queue for future processing between turns; two
tiers of thinking, immediate and in the background, by the order of stake"; "the highest stake of the entire chain is the stake of that chain".
What exists: read sets on every kill and claim and inheritance by the move's diff (52.13, pull form, parent to child only); the inverted read
index of 52.18 (1) (square -> facts that read it; the listener registry, push form). Missing: derivation links (a fact knows its squares, not
the facts it was built from), a store that outlives one agenda (siblings in the chase and later root agendas re-derive), lazy recomputation.
Related work to name in the code: Doyle 1979 (justification-based truth maintenance), de Kleer 1986 (ATMS); ours is justification-based with
read sets as the leaf premises. Items, in build order (one commit each; no AI trailers; OBS lines; principled, not numbered):
1. **The fact store.** `FactStore` holds every claim and kill ever derived in a run: id, kind (claim / kill / gap), the question key, the value,
   the read set (squares + the pieces on them at derivation time), the JUSTIFICATIONS (ids of the facts it was derived from: a chase claim from
   its per-reply facts, a kill from the fact that killed it, a parry from the claim it parries, an inherited fact from its parent), the stake of
   the question that asked it, and a STALE flag. The store is keyed by position-independent premises, not by FEN: a fact is valid in ANY position
   whose read-set squares hold the same pieces (that is what the read set means); the FEN memo of 52.13 stays as a fast path. The chase and the
   root agendas consult the store before asking: a fresh fact whose read set matches is served ("served from the store (fact #n, derived under
   Q3 at depth 2)"); the store replaces `_inherit_for`'s parent-only walk with a lookup, so siblings and transpositions share facts. OBS: facts
   derived vs served per set; inner questions before/after.
2. **Stale flags propagate eagerly, up the chain.** On every move (in the chase and between the agenda's root questions) the diff's squares look
   up the inverted read index; every fact that read a changed square is marked stale, and staleness propagates through the justification links to
   every fact derived from it (cheap flag flips, no recomputation). The CHAIN STAKE of a fact = the max stake over every fact that depends on it
   (Kaveh: "the highest stake of the entire chain is the stake of that chain"); keep it on the fact, updated when a dependant is added. Test: a
   kill that read c1 goes stale when the rook leaves c1, and the claim built on it goes stale with it; a fact that read nothing on the changed
   squares stays fresh (0LRTF: "the rook alone cannot mate while our king is walled" survives every check of the walk).
3. **Lazy recomputation, two tiers.** A stale fact is NOT recomputed on invalidation. It goes into a queue ordered by chain stake. FOREGROUND: the
   agenda's heap (already stake-ordered, stop rule (a) when the best claim beats every remaining stake) reads a fact on demand; stale + needed =
   recompute now, and only facts in the live question's justification chain are recomputed (the mate's chain may include a pawn fact: recompute it;
   a stale "wins a pawn" fact outside the chain stays queued). BACKGROUND: `FactStore.drain(budget)` recomputes from the top of the queue; the
   event loop (plan 40 / catspace event loop, the "between turns" slot: find where the loop idles after a decision and call drain there, the
   opponent's clock as the budget if the loop has one, else a call the analyse CLI makes after each puzzle with `--ponder`) so the store is warm
   for the next question. Kaveh (2026-10-19, with the time budget 52.17 (5)): "after our move was made, we can continue analysing whatever was
   left behind" -- the out-of-time GAPs of the live move enter the background queue first (by their chain stake), and the play server keeps
   thinking on the opponent's clock (ponder by lemma: the worker drains the queue until the human moves, its lines streamed to the trace panel
   marked "background"); facts found there are fresh when our next move is asked. The trace prints both tiers: "foreground: recomputed 3 stale
   facts of the mate chain; background queue: 41 (top stake 5)".
   OBS: foreground recomputations per set, queue length at decision, seconds before/after.
4. **Across puzzles (the remember goal).** With `--memory` the store persists between puzzles of a set (the sets are unrelated positions, so expect
   few hits -- say so) and, in the frame-player / event-loop path, between the moves of one game: a fact derived on move 30 whose read set is
   untouched is served on move 31. Test: a two-ply sequence where the second position serves facts from the first; OBS hit counts.
Acceptance: every 52.13-52.18 acceptance puzzle unchanged; dev_20 and fresh_60_threats_event OBS before/after (decisions, inner questions,
served-from-store counts, seconds), every changed decision SF-classified; tests per item; the justification graph of the deciding claim printed in
the trace for one puzzle (the chain with each fact's read set and stake).

## 54. THE PLAY SERVER: play against the agenda, watch it think (Kaveh 2026-10-19 GO "let's get a game server up ... show me its thinking"; built beside 52.17-52.19)
(53 stays reserved for the ordering prior / measure claims.) Kaveh: "have us play against it and have it show me its thinking; I want to see inside
of its brain as it plays against me. Take the lichess UI; one corner of the screen for the trace; auto-scroll inside that element but don't move my
position in the page; colour-coded so we can read the trace; as I hover over parts of the trace, an arrow on the board showing what it's thinking --
if there's a move, I want to see the arrow." Later, NOT now: a vendor-agnostic hosting adapter (Hetzner / Google Cloud / a cheap VM) so people can
play it; first locally.
Inventory (2026-10-19): nothing plays a full game from the agenda (`discovery/agenda.py`; only analyse.py and tests call it). `catspace-server`
(deployment/server/assistant_server.py, :8777) plays MAIA against the human and our old engine only co-analyses; its PAGE paths point at a directory
that no longer exists since the 2026-08-03 restructure. The analyst local app (incubator/analyst/web.py, :8765, web_page.html) has chessground +
chess.js vendored (research/tools/viz/viz/assets/chessground), the "Something is not right" flag button -> feedback/flags.jsonl, and the job +
polling pattern (`/api/job/<id>` with partial results). The agenda's trace is `"\n".join(self.lines)` after `ask()`; `R.reset_ids()` is a module
global (one agenda at a time: a lock); `chase_log` must stay unset in a server. No player has game context (50-move clock, repetition) in the agenda.
Build in `catspace/incubator/analyst/discovery/play_server.py` + the page, one commit per item, no AI trailers, no edits to agenda.py / regress.py
(another builder is editing them; everything here hooks from outside):
1. **The mover.** `AgendaPlayer.move(board) -> (move, trace_lines, source)`: `Agenda(state_of(board.fen(), us), memory=...)`, `ask()`, `decision()`;
   the claim's `move` (UCI of the first move; `what` may be "X then Y") checked legal. When the agenda has NO decision (most quiet positions), a
   FALLBACK that is named in the trace and in the UI: default = the rules-based one-ply `frame_player.choose` (pure Python, no search); flag
   `--fallback stockfish` for a playable opponent, `--fallback random` for the honest baseline; the trace's first line says which played ("no
   decision: fallback one_ply chose Nf3"). Game context: the server owns a python-chess board with the full move stack (50-move clock, threefold
   via `is_repetition`, `is_game_over(claim_draw=True)`); the agenda sees the FEN only (a known gap, [[game_context_in_frames]]; say so in the trace
   when the clock is under 30 or a repetition is available). One lock around every `ask()`.
2. **The live trace.** No edit to agenda.py: right after constructing the Agenda, replace `ag.lines` with a list subclass whose `append` also calls
   a callback (the agenda only ever appends). Each line is emitted as a record {n, text, kind, indent, moves, squares}: `kind` from the line's
   shape (question "Q<n>", claim "-> CLAIM", kill "KILLED" / "killed", gap "GAP", chase "chase:", route "    <san>:", their "their" / "parry",
   stop "STOP", fuse "fuse", note "note:", state "EV"), `indent` = leading spaces, `moves` = every SAN token in the line parsed against the board
   the line is about (the root board; a line under "after <san>:" is parsed after pushing that move -- best effort, never a crash), `squares` =
   every square name in the line. The record goes to a per-game ring buffer; GET `/api/play/trace?since=n` returns the records after n (polling at
   ~300 ms, the existing job pattern); the engine's move runs on a worker thread so the lines stream while it thinks.
3. **The server.** Extend web.py (same port 8765, same assets): POST `/api/play/new {colour, fallback}`, POST `/api/play/move {uci}` (the human's
   move, then the engine's reply starts on the worker), GET `/api/play/state` (fen, legal dests, last move, game over + reason, whose turn,
   thinking: bool, clock/repetition notes), GET `/api/play/trace?since=`, POST `/api/play/flag` reusing `append_flag` with the trace line numbers
   in the context. PGN of the game appended per move to artifacts/experiments/play_games.pgn (paths.experiment) with the trace as comments on the
   engine's moves.
4. **The page.** A "Play" view in web_page.html (lichess look: the vendored chessground with its default board + piece set, the lichess-like
   layout: board left, panel right): new game as white / black, the board movable only on the human's turn, last move highlighted, game-over
   banner. The TRACE PANEL in one corner (right column, fixed height, `overflow-y: auto`): colour-coded by `kind` (question / claim green / kill
   red / gap amber / chase blue / their purple / stop / note grey), indent preserved, monospace; it auto-scrolls to its own bottom ONLY by setting
   `el.scrollTop` and ONLY while the user is already at the bottom (a scroll-up pins it; a "follow" toggle); NEVER `scrollIntoView` or anything
   that moves the page. HOVER a line: chessground `setShapes` draws an arrow per move in the line (first move strong colour, the rest lighter) and
   a circle per square; leaving clears; CLICK pins the shapes until the next click. The engine's chosen move stays drawn as its arrow until the
   human moves. The flag button and dialog kept (per-line flag: the line's n). Phone-width works (panel below the board).
5. **Run and prove it.** `.venv/bin/python -m catspace.incubator.analyst.web --play` (or the existing entry) starts it; a smoke test plays 6 plies
   against itself through the HTTP API (pytest, no browser) and asserts the trace records stream, moves are legal, the fallback is named; one
   game played by the builder through the API against `--fallback random` to the end, OBS: plies, agenda decisions vs fallbacks, seconds per
   engine move (mean/max). Report the URL and the exact command in the hand-back.
6. **Past positions (Kaveh 2026-10-19 GO "I should be able to go to a past position and see its analysis").** The game keeps EVERY trace
   record of every ply (no ring-buffer loss; the per-game store is the whole game), each record carrying its ply and the FEN it was parsed against.
   The page gets a move list under the board (SAN by move number, the current one highlighted, arrow keys and click to navigate, "live" button to
   return). Selecting a ply shows that position on the board (not movable), its last move highlighted, and the trace panel shows the records of
   that ply (the engine's thinking before its move there; for a human ply, the background records if any), with hover arrows drawn on THAT board
   (the records already carry from/to, so the current-ply-only restriction goes), click pins as before. GET `/api/play/trace?ply=k` returns one
   ply's records; `/api/play/state` carries the move list. The live game continues underneath; a new engine move does not yank the view back
   (the panel for the viewed ply stays; a small "engine moved" note appears). Test: after 4 plies via the API, `?ply=2` returns only ply 2's
   records and the state's move list has 4 SANs.
Not now (recorded): the hosting adapter (an `infra/providers/<name>` interface: create VM, open port, deploy the compose; Hetzner and GCP first),
auth / multiple players, the agenda's own game context (52.19 (4) carries facts across moves).

### 52.20 Repeated work in the word reads and the defect scans (Kaveh 2026-10-19 "do a timing profile ... Do it"; GO)
Profile of the two slowest puzzles (0KCeb 18 s, 0UqOv / 07Z0P 19 s; median puzzle 0.2 s), same shape on both. The deep part is the design (their
agenda after every forced reply); these three items are repeated work inside it, each changes no decision.
1. **Word reads memoised.** ~40% of the time: 203k Rust word reads, 149k FEN strings built for 19k distinct boards, 39% of the reads exact
   repeats of (board, term, mover), `install_terms` re-run on every read (383k times). `val_at` caches on (fen, term, mover); `graph.see` /
   `graph.word` take the FEN from a per-board cache keyed by python-chess's transposition key (bitboards + turn + castling + e.p.), and
   `with_turn` copies nothing when the owner is already on move with no e.p. square; `install_terms` runs once. The caches are cleared at every
   fresh root question (a natural end: the root is the unit of attention; the fact store, not these caches, carries knowledge across moves).
2. **The defects before the move computed once.** ~10%: `routes_of` calls `_still` per candidate move and `_still` recomputes the mate defects of
   the unchanged starting board each time (84k scans for 3k route lists). Computed once per `routes_of` and passed in.
3. **`reach_steps` lists the legal moves once** (it walked the generator twice per call; 1.2M generator steps). Small.
Acceptance: both sets rerun, every decision and every tally but seconds unchanged; OBS seconds per set before/after; the three slowest puzzles' seconds.

### 52.21 The fact store serves more: root lists, chase claims across positions; the deadline finishes the route in hand (Kaveh 2026-10-19 GO; after 52.20)
After 52.20 the profile of the slow puzzles is flat and the time is the design: on 0KCeb 353 inner agendas each rebuild their root lists (the
reach roots ~15% of the time) and chase claims are re-derived on siblings because they are filed non-portable. Kaveh: the fact store "could" serve
these; and on the budget: "when time runs out, if we're in the middle of creating a claim, we should finish that one up and then not start any new
ones until we make our move; then it's the opponent's turn and we can continue thinking."
1. **Root lists as facts.** `_roots` / `_reach_roots` read the board's pieces, their attacks and the king's neighbourhood. Record the squares each
   root list read (the piece squares scanned, the squares attacked / attacking that decided a root in or out, the king's box) and file the list as a
   fact (kind `roots`, positional key = the side to move and the goal family; portable) with that read set; an inner agenda whose board matches the
   cells is served the list and pushes it without rescanning. Trace: "roots served from the store (fact #n)"; tally `root lists served`.
   The read set must be sound: a square whose change could add or remove a root is in it (test: a sibling where a piece moved onto such a square is
   NOT served; one where an unread piece moved IS served and its roots agree with a rescan).
2. **Chase claims portable.** A chase claim today is filed `portable=False` because it read every reply. Its read set is the union of the read sets
   of the facts it stands on (its routes' claims and kills, their replies' agendas), which the justification links already hold: compute it, file the
   claim portable with that union, and let `lookup` serve it on a sibling whose cells match. Where a justification is a positional fact (an inner
   agenda's whole position) the union is the whole board and the claim stays non-portable (say so in the trace). Tally `chase claims served`.
3. **The deadline finishes the route in hand.** 52.17 (5) cuts wherever `out_of_time()` is next read, mid-route, leaving a resume point. New rule:
   past the deadline NO new question is pushed and NO new route is opened (at any depth), but the route currently being chased (the innermost unit
   that yields a claim or a refuting reply) runs to its end; then the best bound claim decides and the open routes are GAPs as before. The play
   server's background tier (52.19 (3)) resumes them on the human's clock, out-of-time GAPs first (already so). Trace: "out of time: finishing
   <route>, no new routes". Bound the overshoot by the route, not the root: an inner agenda started before the deadline finishes its CURRENT question
   only. OBS: used seconds vs budget on the sets at --seconds 0.3 and 1.0, before and after; overshoot should fall to the length of one route.
Acceptance: both sets identical in decisions and MATCH (tallies of inner agendas / questions should FALL; print them); 0KCeb inner agendas well below
353, with root lists served and chase claims served printed; the seconds per set; tests for the sound read set of a root list and for a served chase
claim on a transposition; 52.19's fact-store tests still pass.

### 52.22 A reply closes by rule only when the goal is met; the attacker added with tempo; the refuter fix (Kaveh 2026-10-19 on 07Z0P: "build them both"; "check if there's a better way, otherwise build")
07Z0P (2r2rk1/1p4bp/6p1/pP1qpp2/6PP/P1n1BQ2/2B2R2/5RK1 w; Qxd5+ Nxd5 Bb3 wins the pinned knight) is a MISS. Kaveh's walk: down 2; our king
checkable only on the light diagonal and their Q stands on it; Bb3 skewers but Qxb3; so trade the queens: Nxd5 is forced and the piece on d5 is
now a knight, which cannot take the pinner; Bb3 pins; no pawn defends d5; they can add one rook (c5 covered by Be3, d8 holds one) and we add two;
two against one wins it. The trace: standalone after Qxd5+ Nxd5 the agenda finds Bb3 (attack with tempo AND trap) but values it 1: after ...Rfd8 the
reply is "closed by rule: exchange +1 (gxf5 ...)", because `_closed_by_rule` closes when the exchange reaches `need = 1 - dbal` (one pawn ahead),
not what the goal asks; so the root's chase of Qxd5+ under "win the N" (stake 3) gets 1 back and kills the route. Nothing names the trade.
1. **A reply closes by rule only when the chased goal is met.** In `_closed_by_rule` the exchange must reach the question's stake (the goal's
   value), not the route's `need` of 1; below that, if the goal's piece is still on the board and `fixes_of` finds a forcing fix of its defect on
   P2, the reply is NOT closed: the inner agenda is asked (the normal path). Replies with no forcing fix left still close as before (speed kept).
   Print "not closed by rule: the goal (win the N on d5, stake 3) is not met by the exchange (+1); a forcing fix remains (Rd2)".
2. **"Add an attacker with tempo"** as a fix of the `defended` defect, the twin of "remove the defender with tempo": our moves that add an attacker
   of the target and are forcing (a threat: the exchange on the target then wins). Kaveh's count (the defenders they can add vs the attackers we can
   add; c5 covered, d8 one rook) is then read by the chase, not by a special count. Acceptance: after Qxd5+ Nxd5 the agenda claims Bb3 worth 3
   (the ...Rfd8 reply followed by Rd2 and read to its end); then check whether 07Z0P decides Qxd5+ by the existing chase alone (report it).
3. **The refuter fix.** Every killed route names the reply that refuted it (Bb3: "after Qxb3"). The fix: remove or replace the REFUTING piece with
   tempo, the same fix the agenda applies to defenders and blockers, applied to the refuter: a capture of it with a forced recapture (one legal
   recapture, or every recapture puts a piece that cannot make the refuting move) is a route; the original route is then re-asked on the board after
   the exchange, and the claim is filed with the refuter's square in its read set. Trace: "Bb3 fails to Qxb3; exchange the Q on d5 first (Qxd5+ Nxd5
   forced): the N cannot take b3; Bb3 pins the N". Even if 07Z0P decides by (1)+(2), build (3): it names the reason and routes the search (the
   trade is asked because it fixes a refuted route, not because a chase stumbled on it). Lean: a short word `refuter` (the piece whose move killed a
   route) + gloss + glossary + lean_index build; no new theorem needed unless the builder finds the recapture condition needs one.
Acceptance: 07Z0P decided Qxd5+ (SF: +301 d20) with the refuter line printed; both sets rerun one at a time, every other decision unchanged, any
change checked with Stockfish at depth 18 and reported; tallies (inner questions, replies closed by rule, seconds) before/after as OBS.

### 52.23 Unlimited time until a batch is right: random samples, the fuse gone under a time budget, the need-aware close, the in-check chase, the skewer (Kaveh 2026-10-20: "keep fixing and going forward ... leave it unlimited time ... once we get to 5 puzzles correct, then reduce time waste"; "randomly sample puzzles from the set, don't stick to the same puzzles")
The sets dev_20 / fresh_60 were fixed strided draws; `sample_puzzles --seed S --n N` now draws N rows uniformly from the whole lichess file (reservoir
sampling) into `ledger_puzzles_randS_N.json`. Three samples of 10 (seeds 1-3, run with the question fuse): 25 MATCH, 4 undecided, 1 wrong move; every miss
rated 2240+ (9wgF7 2486 Nc5, dOC36 2779 Nf6+ [we chose Rd2: SF -87 vs +436], zBouR 2432 Rhxf2, pogsJ 2434 Qxa4, 4efZ2 2240 Ra5). Items, each on a miss:
1. NO QUESTION-COUNT FUSE UNDER A TIME BUDGET (`--seconds`): `ctx.left = NO_FUSE`; the routes share the TIME fairly (52.17 (5)), never the count (the 20x safety
   fuse still cut zBouR's routes: 28 fuse hits in 9.5 s).
2. THE NEED-AWARE CLOSE: `_closed_by_rule` closes a reply only at a value that can still rescue the route this agenda serves (`dbal + best >= min_stake`); a lower
   bound below the need decides nothing, so the reply is asked. zBouR: Qh1+ Kd2 closed at the rook (5) when 6 was needed; Rxf2+ Kc1 Rxc2+ wins the queen.
3. THE IN-CHECK CHASE: an inner agenda in check (their intermezzo check inside our forcing line) chases the save-king question with every legal move as a route
   (each is forced: the check must be answered); claim kind `save`. ONE intermezzo is followed: an agenda opened by an evasion route does not chase a second
   check of theirs (their chain, not our line). Before: the inner agenda in check verified its evasions by the schemas only and died (zBouR: Rhxf2 cxb7+ "no claim").
4. THE SKEWER (Lean `Ideas/Skewer.lean`: `skewerB`, `skewer_blocked`, `skewer_attack`, the almost-battery with their king as the one piece between; both theorems on
   the standard axioms): a named fix of the win defects 'not attacked' / 'defended' (`regress.skewer_fix`, routes `skewer_routes`), and a check that ADDRESSES them
   (`_addresses`); `_unmet_fix` sees it, so a reply is not closed by rule while the skewer remains. Gloss, glossary, lean index.
5. A WIN QUESTION CHASES AT EVERY DEPTH: the depth-two gate ("below the first inner level only the mate questions chase") was a numbered limit; its checks must address
   a defect (52.17 (2)) and its routes at inner levels are checks, captures and mate threats only, as before. Promote questions keep the gate.
Acceptance: zBouR decided Rhxf2 with the skewer line printed; three random samples of 10 rerun one at a time with a time budget, every decision diffed against the
fuse runs, any change checked with Stockfish; the cost (inner agendas, seconds) as OBS, directional; then the next samples.
Outcome 2026-10-20 (the build, by me, not a builder): (1), (4) built and kept; (2), (3), (5) built but OFF by default (`NEED_AWARE_CLOSE`, `EVASION_CHASE`,
`DEEP_WIN_CHASE`), with three more chain items built the same day and also off: (6) THE GOAL-PROGRESS JUDGE (Kaveh: "mating the king is like mating the bishop"):
tempo arithmetic ends a chain -- a mate or win pattern with d defects allows d prep moves (a two-move fix such as "drive the king onto the skewer line, then skewer"
says `steps` 2), a bigger goal found on the chain starts its own distance, an exchange (a recapture on our route's square) spends no level, a win goal's allowance
starts again after a capture that is not bad for us (`GOAL_JUDGE`); (7) THE NEED PROPAGATES to the inner agendas (what the route must reach, not 1), a gaining reply
carries the inner claim, a chain of two (this question's capture plus the largest other stake when the target can be taken now) meets a need no single stake does, a
bound below the need chases first (`NEED_PROPAGATION`); (8) THE CHASE CLOCK (Kaveh: a capture that is not bad for us is one-way progress; N of our moves without one
end the chase; a hyperparameter, `CHASE_CLOCK`, 4 was his starting value, 99 = off); (9) a kill by the chain's state (the allowance, the clock, a second intermezzo
check) is never filed as a fact of the position and is memoised for that chain state only. ON together they take zBouR from 20 minutes to 8 seconds and read the whole
line (Rhxf2 cxb7+ Kb8 Rxf2 Qh1+ ... Qxc2) but still do not decide it (the last defence read, Qb1, is answered by a mate in three the threat-save inner path does not
find though the root does), and they cost 07Z0P its decision under the count fuse (Bb3's share no longer finished) -- the evasion chase ran through `_verify`
whatever its switch said (fixed). Two soundness fixes kept ON: (10) a SCHEMA claim's read set includes its move's squares and their king (2VDtX: "Ng5+ then Nxe6"
derived after Rf8+ Kxf8 was served at the root, where Ng5 is no check and Bxg5 takes it, SF -697; the store had served it since 52.19); (11) the time regime: a
root's chase gets at most half the time left while another root can beat the best claim (the fuse's rule), and a route cut by time re-runs on a longer slice.
Directional numbers (OBS, one run each, the chain items off): rand1_10 9/10 (9wgF7 2486 none); rand2_10 7/10 under 60 s (BEKFf none: the mate roots re-run the same
four routes on shrinking slices and the win root never gets its turn; under the count fuse 8/10 with dOC36 wrong); rand3_10 8/10 (2VDtX right again). The count fuse
stays the default budget for sets and tests; the time regime wastes the inner work a cut throws away and needs the prior (plan 53) to ask the deciding question first.

### 53 The prior that orders the questions (Kaveh 2026-10-20: "it's time to wire in the prior that orders the questions to be asked ... training data: did the deciding line agree with Stockfish and did we go down it; if we neglected to ask the right question, a negative; never penalise the right question wrongly answered -- that is a bug, it should have been a proven lemma")
1. THE QUESTION LOG (`analyse --qlog [--sf]`): one row per question of the ROOT agenda in the order asked -- what was known of it before it was asked
   (`question_prior.features`: goal, stake, the target's piece and the counts on its square, the check state, the depth; generic words, no square names),
   its outcome (status; the WORD that answered it: the claim's lemma or the kill's reason, so a wrong answer to a right question is seen as a bug or a
   missing lemma, never as the prior's fault), whether it decided (`Claim.qid`), the decision, the puzzle's answer, the STOCKFISH VERDICT
   (`verdict.py`: RIGHT within 50 cp of the best move at depth 14; the grader, never a premise). Written to experiments/question_log.jsonl.
2. THE PRIOR (`question_prior build` -> data/derived/question_prior.json; `analyse --prior`): logistic regression over the feature dict (sklearn at build
   time; JSON weights at run time). Labels: the question that decided a RIGHT decision is 1, every question asked before it 0; when the decision was wrong
   or none, the question whose target is the answer's square is the one that should have been asked (1), those before it 0. At run time the ROOT agenda
   asks the open question with the highest score next (the stops stay by stakes and claims: the prior ORDERS, it never decides; the line "prior: Q7 asked
   first (score +1.3; the stake order would ask Q1)" says so). `question_prior eval`: held out by puzzle, how often the deciding question ranks first
   under the prior vs the stake order (OBS, directional).
3. Data from fresh random samples only (seeds 1-6 for the log, 7-8 held out), under the count fuse; the chain items stay off.
Acceptance: the log written and labelled, the model built with its largest weights printed, the eval OBS, then two held-out samples run with and without
the prior: decisions diffed (the prior must not change a decision, only the order and the questions asked before the deciding one), the count of
questions asked before the deciding one before/after as OBS.
Outcome 2026-10-20: built and committed (the log, the grader as its own pass -- the engine beside an agenda stalled the set runs --, the prior, tests,
glossary). OBS: 60 puzzles (seeds 1-6) logged in 2.5 min, 313 question rows; Stockfish grade RIGHT 43, WRONG 3, none 14; the model from 95 labelled rows
(51 positives, 51 agendas); largest weights target=pawn +0.80, goal=win +0.76, target_ours -0.74, target=queen +0.72, goal=mate -0.61; 5-fold held out
by puzzle: the deciding question ranks first under the prior 9/13, under the stake order 0/13 (by construction). Held-out seeds 7-8 (20 puzzles), with
and without the prior, under the count fuse and under 10 s per puzzle: every decision IDENTICAL (the prior orders, never decides -- as required), but
questions asked rose (18 -> 34, 32 -> 53, 22 -> 48 ...) and time 22 -> 36 s, 23 -> 52 s: a small claim asked first does not stop the agenda (stop (a) is by
stakes), so the higher stakes are still asked after it. The ordering pays only when a budget cuts before the deciding question; on these samples none did.
`--prior` stays OFF by default. Next for it: more data (the log is cheap: 60 puzzles in 2.5 min), the question's defects in the features once asked
(two-stage: a cheap feature pass before the fixes), and a stop rule that reads the prior's confidence -- a decision for Kaveh, since the stops are the
reasoning's and the prior must not become a decider.
