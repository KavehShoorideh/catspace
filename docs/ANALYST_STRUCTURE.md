# The analyst, as built — structure, complexities, risks

Written 2026-09-05 from the code in `catspace/incubator/analyst/` (31 modules, ~5,600 lines, 40 tests), for two
purposes: to let rule changes be made deliberately, and to know what the next concept layer stands on.

## 1. The layer stack and what each module owns

Every layer imports only the ones below it. Nothing above L1 knows what winning is; the exchange scale is the one
thing the lower layers may take from the objective, and only through `value()`.

| Layer | Module | Owns | Depends on |
|---|---|---|---|
| L0 geometry | `geometry.py` | squares, direction families, rays, distances; board size as parameters (unused by most layers, which use python-chess directly) | python-chess |
| L1 rules | `rules.py` | the variant table (python-chess board classes), the `royal` flag, `captures_forced`, the exchange `Scale` per variant (`set_scale` swaps it at runtime), `royal_square`, `is_royal`, `scale_of` | python-chess |
| L2 facts | `facts.py` | `value()` and `material()` under the scale; `see()`, the static exchange evaluation (swap list on a scratch board, sliders revealed behind capturers, a king never captures into a line its departure opens, an exclusion set for disqualified defenders); state facts | L1 |
| L3 relations | `relations.py` | `Relation(name, arity, squares, owner, tags, pieces)`; attacks/defends; hanging, loose, under-defended (via the defense operator), attacked-by-lower; line relations by the removal test (pin, skewer, x-ray, battery, discovered attack, x-ray defence, own alignment) with BITE; forks, double attacks, overload, double duty, king-zone pressure; the composed four-piece pins; hanging-through-pin (a reading of the defense trace); the owner convention `VICTIM_OWNED` + `against()` | L1, L2, defense |
| L5 threats | `threats.py` | a threat with its required defences (from the goal search's AND node after our move, or mate-in-one enumeration) and the punishing move for every other reply; `judge_reply`, `threat_lines` | goal_search, subgames |
| L5 mining | `mine.py` | learning sub-goals: need-tree nodes as templates (roles and relations), credit from the engine line, support/lift library, the quiet-move coverage number | needs, engine |
| L5 needs | `needs.py` | means-ends planning: needs (win, defend, reach, safe, remove, deflect, block) with ways and sub-needs, costs (material, holdings, shelter, weaknesses, tempo), comparison and give-up, the race at the moment of the threat, the goal search as verifier, the idea loop (reasons to move, destinations, what each creates), the AGENDA (the preconditions given-up trees died on, ranked by what they unlock; pin-first ways), `idea_lines`, `agenda_lines` | plans, reach, subgames, structure, goal_search |
| L5 regime | `regime.py` | the board as a sum of sub-games when earned: light cones (safe reach as a set, bound by holdings and races), promotion races (rule of the square by computation), trapped pieces, the coupling graph over lemmas (movers, targets, cones), frozen regions by a local fixpoint with a tempo reservoir, the live remainder probed in syzygy (3-5 men) with shared pieces kept, `regime_lines` | L3, subgames, tablebases |
| L3 graph | `graph.py` | THE INTERACTION GRAPH, one explicit object per position: piece nodes with 43 named answers from the four generic operators (the exchange, counterfactual removal, the free move, paths), 17 named pair relations (attacks/defends that count, line-through with what lies beyond, blocks, holds, wins-with, co-attack, reach in 1-2, distance, aligned, same zone), 39 position indicators; zones = components of the lemma graph with a TEMPERATURE (what a free move there is worth); `changed_squares`; vectors for learning, prose for the product | defense, subgames, regime, threats |
| L5 track | `track.py` | the state across plies: per move the threats it executes / creates / parries / leaves, the zone it touched matched to the previous ply's by piece overlap (the action stays or moves; zones named by pieces) | threats, subgames, gauge_focus |
| L6 dataset | `dataset.py` | the knowledge dataset: graphs from real games + the engine's best move and evaluation at depth 12 + the regime 10 and 20 plies ahead + the result, sharded npz + jsonl | graph, engine |
| learn B/A | `knowledge_net.py` | the transformer over the graph: answers as bucketed token embeddings, relations as per-head attention bias, a Backpack output (phi linear in named senses), IQE poles for W/D/L, cp / regime-ahead / policy heads, trajectory loss calibrated in plies with cross-game negatives (no pawn term); variants full / board (= approach A) / rules; evaluation with degenerate baselines; the pawn asymmetry test | dataset, IQE |
| learn gate | `experts.py` | three experts per move (A, B, the rules ranking C) and a gate blind to the board (named context only): single experts, the per-position oracle bound, a fixed blend, the context gate, the scenario table of mean weights by indicator | knowledge_net, rank, graph |
| L5 regimes | `regimes.py` | the quantized future: 48 named regime codes (k-means on the graph's 15 regime + 6 sharpness indicators, training games only), each described by its centroid and valued by the mover's result rate; sharp codes = a big threat or a mate threat at the centroid | graph, dataset |
| L5 planner | `planner.py` | THE PLANNING LAYER: the distribution over codes at +10 / +20 from the knowledge net, expected value and sharpness, entropy, steering targets (reachable futures whose value beats the expectation) mapped to needs by named differences, the search budget from the window-sharpness head; `pursue` hands the needs to the goal search / terminal search / race; prose in the web 'plan' pill and CLI --long | regimes, knowledge_net, goal_search, regime |
| L5 recall | `recall.py` | memory indexed by the geometry: nearest memories, the FUTURES by d(query -> m) with the rules' reachability beside, steer/avoid contrast; gauges (reachability AUC, value by recall) | knowledge_net, regimes |
| learn reach | `reach_qm.py` | a simple reachability quasimetric conditioned on Elo: board-only piece tokens + an Elo token to a 64-d IQE, the quasimetric objective in plies over one-rating-window batches; gauges by band (calibration, reverse of time, future ordering, wrong Elo) | IQE |
| learn data | `acquire.py`, `dataset.extract_fens` | a labeled random corpus (ChessBench) to the shard format; uncertainty-vs-random acquisition rounds with regret against the teacher | knowledge_net |
| L5 regions | `regions.py`, `endgames.py`, `tokens.py`, `root_tokens.py` | from the rules alone: a piece's safe closure and contested frontier with attackers; enumerated region relations; the tablebase dataset (3-5 men, WDL/DTZ, reply child, king-move siblings); invariant binary tokens predicting the result and its distance, coherent across a reply; each token rooted in a tree over the relations; faithfulness; king-move explanations | -- |
| L5 pieces | `pieces.py`, `piece_values.py` | piece identity across plies and fates (kept / exchanged / lost / promoted); the value of a piece of a kind traded off, from outcomes, by side x kind x activity x king zone x evaluation, with base rates by phase; read by the planner's piece ideas | dataset |
| L5 transitions | `transitions.py` | the value of a regime coordinate falling, from outcomes, in cells of material and level / evaluation; read by the planner's steering (support / avoid) | dataset |
| gauge | `gauge_sts.py`, `gauge_threat.py`, `probe_concepts.py`, `gauge_pieces.py` | the Strategic Test Suite per expert and theme; the threat as a constraint (the best move in the response set, the set's size, counters); what the network learned in the rules' vocabulary (sense atlas, square maps, concept probes, moves by kind) | rank, knowledge_net |
| gauge | `sharpness_check.py`, `gauge_budget.py` | is sharpness predictable ten plies out (AUC vs regime-only and persistence); the ranking at full / normal / shallow / gated budgets against the engine, with seconds | dataset, planner, rank |
| gauge | `gauge_focus.py` | focus accuracy (does the engine's best move touch the hottest zone; baselines largest / random / previous / any) and persistence across plies | graph, engine |
| L5 sub-games | `subgames.py` | lemmas (target, status, value, participants, read footprint), components by shared pieces, `measure` (the sum-assumption test), HOLDINGS = what each piece keeps by the removal test (our attack lost, their attack exposed, a mate appears), `cost_of_move` (exact replay), `holdings_lines`; `goal_search.search_goal` adds refinement (the refutation as the next sketch) and the proof / sketch store | L3, L4.5, memo, goal_search |
| L3 operators | `defense.py` | EFFECTIVE DEFENDERS AND ATTACKERS: simulate the cheapest legal capture; a defender counts only if the rules let it recapture and recapturing does not expose more than it wins (exposure candidates = what the defender guards or shields; priced by SEE; exposures the capture itself opened are not charged); an attacker counts only if its capture is legal and its DEPARTURE does not hand the owner a piece worth at least the target that the owner may legally take (the b4 pawn shielding the a4 rook: bxa5?? Rxa4); reasons read off the geometry afterwards; `effective_attackers`, `attackers_out`, `excluded` | L1, L2, memo |
| L3 operators | `reach.py` | flip conditions (the loosest single ledger change that wins a held target); safe reach (BFS over a piece's legal quiet moves, landing squares checked by the defense operator); the progress map (falls / joinable with path / undermine / held / out of reach) and its prose; `verify_plan` (null-move hops, then the engine's reply: wins / chased / met) | defense, memo |
| L3.5 structure | `structure.py` | pawn structure (isolated, doubled, passed, backward, islands), rooks on open files, outposts, bishop pair, blocked bishops, king files, per-piece scope; identity rules that keep the per-move diff quiet; `structure_diff`, `top_changes`, `structure_tags` | L1 |
| L4 dynamics | `dynamics.py` | the significance filter; `Effect` = what one move created / ended / used / resolved / escaped, plus the structure diff, all in a deterministic order; standing threats by the null move with the cash test; structural threats verified by the engine; zugzwang | L3, structure, engine |
| L4.5 sequences | `sequences.py` | edge events over ≤3 plies: removal of the guard, deflection, attraction, interference, clearance (castling excluded) | L4 |
| L4.5 mates | `mates.py` | mate patterns on the mated position (back rank, smothered, double check, discovered, supported queen, edge) | L1 |
| L5 ideas | `ideas.py` | Stockfish + the objective: idea, killed ideas with refutations, failing alternatives, threats, the played move's verdict in win-probability points, the clean-hash score trajectory | L4, engine |
| L5 causes | `causes.py` | why an error was an error: walked into (the reply exploits a liability the move created, same victim piece), ignored a standing threat, allowed a tactic, missed better, positional (even trade); simple vs compound | L4 |
| L5 horizon | `horizon.py` | per-depth score trajectories (fresh hash per candidate, one thread), trap vs blind spot, `winp` (the Lichess logistic) | engine |
| L5 ladder | `maia.py` | lc0 with Maia weights 1100–1900, policy from verbose move stats at one node, sqlite cache; the "effective rating" = the level whose policy best predicts the player | lc0 |
| L6 verbalize | `verbalize.py` | one phrase per relation, `best_fact` (one fact per ply), terse and long renders, `delta_str`, `purpose`, the headline per cause, `ladder_tag`, motif phrases; the standard scale for prose thresholds | everything below |
| memory | `memo.py` | the concept store: footprint masks (rays cut at exchange non-participants, slider rays to the first blocker, pin lines) + predicates (checkers), Zobrist sub-hash, sqlite persistence | python-chess |
| search | `search.py` | concept state per node (ledgers, both progress maps, material), candidates (engine top-k, falling captures, joiner hops, checks), the concept DELTA per edge | reach, defense, memo, engine |

Services and products (not layers): `engine.py` (Stockfish adapter, root-restricted searches, null-move probe, sqlite
cache in DELETE mode, Fairy-Stockfish for variants when installed), `review.py` (a player's games: `annotate()` per
move, the profile, "most useful to fix"), `parallel.py` (game-sharded workers), `game_report.py` (one game, both
sides, markdown), `prep.py` / `cards.py` (opponent preparation, line cards), `drill.py` (the spaced-repetition
deck), `web.py` + `web_page.html` (the local app and the flag queue), `play.py`, `cli.py`. Measurement:
`gauge_puzzles.py` (Lichess themes with degenerate baselines), `motifs.py` (vocabulary completeness by enumeration),
`miner.py`, `tests/test_analyst_*.py`.

## 2. The pipelines

- **One move** (`review.annotate`, used by the report, the app and the drill): position → `analyse_position`
  (engine lines, threats, killed ideas, played-move trajectory) → mechanisms of the refutation → teeth, the reply's
  exchange net, availability before the move, even trade → `causes.classify` → horizon kind → headline and text →
  the row (also structure tags, refutation, best line).
- **A game**: `game_report.py` runs both players' reviews and the per-position ideas and merges; the app's
  `Analyst.game` does the same in process and streams frames.
- **A player**: `parallel.py` shards games over workers, each running `review.py` with the two-stage filter
  (Lichess evals or a depth-6 check; full analysis only on flagged moves); the profile aggregates rows.
- **A position**: `search.py` expands candidates and reports concept deltas through the store.

## 3. The complexities worth knowing

1. **The removal test** (line relations): S attacks P1; remove P1; whatever S newly sees is what P1 shielded. No
   chess-specific rays; it works for any piece whose attacks python-chess computes.
2. **Bite** (relative pins and skewers): geometry alone is not a pin. A pin binds only if the capture on the back
   piece, with the front piece gone, wins material by SEE; a skewer needs the front piece winnable and the back
   piece winnable once it moves. Shapes without bite are x-rays tagged `pin_shape` / `skewer_shape`.
3. **SEE under the scale**, with the exclusion set the defense operator feeds it.
4. **Effective defenders**: two halves, legality by simulation and economics by the removal test priced with SEE,
   with exposures the capture itself created not charged to the recapturer, and no static exchange while the owner
   is in check. The named four-piece facts are readings of the trace.
5. **The cash test**: a standing threat is real only if its line captures, checks, promotes or mates within two
   moves (the null-move probe also rewards defensive value). A refutation is a tactic only if it cashes or creates a
   loss-naming relation; otherwise the cause is the omission.
6. **Teeth and the exchange net**: a reply with under 40 cp of teeth over the second-best reply is not a tactic
   unless it wins material on its capture square; an even trade is judged after the recapture, and even trades are
   positional unless mate.
7. **Verdicts in win-probability points** (blunder ≥ 20, mistake ≥ 10, inaccuracy ≥ 5) from the Lichess logistic,
   and the per-depth trajectory with a fresh hash per candidate (a shared table hides traps).
8. **Structure identities**: the bishop pair belongs to a side, doubled pawns to a file, a passed pawn's advance is
   one push, a lone pawn moving stays isolated; scope is phrased only for a piece that did not move.
9. **Footprints**: every stored concept declares what it read; soundness is by declaration and checked by the
   property test (memo-served == fresh over ~110k defense/reach checks). Three cuts made it useful: exposure
   candidates only, pin lines + a checkers predicate instead of the king region, rays cut at the first exchange
   non-participant.
10. **Determinism**: relation diffs and structure diffs are sorted with a total order; two runs of a report are
    byte-identical (set differences once followed per-process string hashing).

## 4. Risks for a rule change, ranked

Where chess is baked in, most to least dangerous. "Handled" means a variant test exists.

1. **`memo.py` footprints assume standard movement on 8×8.** Rays, knight jumps and pawn attacks come from
   python-chess bitboard tables; `_can_join` knows the six piece types. A fairy piece, a larger board, or drops
   (crazyhouse: a piece arrives from nowhere) make a footprint UNSOUND, and the store would serve wrong answers
   silently. Rule: for any variant beyond python-chess's, run without the store or extend `attack_reads` /
   `move_reads` and re-run the property test first.
2. **`facts.see` and `defense.py` model captures as moving onto the square.** Atomic explosions, en passant,
   promotion during an exchange, and crazyhouse drops are not modelled. Antichess forced captures are handled by
   legality. `reach` has no notion of a drop (infinite reach in one move).
3. **`structure.py` is pawn- and file-specific**: forward direction per colour, ranks-to-go, king shelter by
   castling rights, outposts by pawn attacks. Variants without pawns or with different pawn moves need their own
   structure facts; nothing breaks, but the facts go quiet or lie.
4. **`mates.py`, `king_zone_pressure`, `royal_square`**: kings-as-royal only; antichess handled by the `royal` flag
   (tested); racing kings / three-check objectives are not represented (the objective is quarantined, so nothing
   reads them; verdicts would still come from the engine).
5. **`maia.py` is chess-only.** No ladder for variants: effective ratings and ladder tags disappear.
6. **`engine.py`** picks Fairy-Stockfish when the variant is not chess; its UCI options and eval scales differ and
   `winp` is calibrated for chess.
7. **Prose thresholds** in `verbalize.py` and `dynamics.significant` use the STANDARD scale (`_STD`) where no board
   is in hand: minor-value cut-offs would be wrong under a custom scale.
8. **The gauges are chess measurements**: Lichess puzzle themes and the motif enumerator's named coverage. A variant
   has no ground truth here; the tests carry the antichess contrasts instead.
9. **The Lichess coupling**: PGN fetches, the `%eval` two-stage filter, opening names. Local PGNs work without it.

General risks, independent of rules:

- **One level deep** in the defense operator (no overload chains) and in reach (replies ignored; `verify_plan` is
  the check). Three hops. Conservative footprints (a landing on any ray invalidates).
- **Thresholds are numbers someone chose**: `TEETH_CP` 40, `REAL_THREAT_MARGIN_CP` 30, `TEMPO_BASELINE_CP` 100,
  `SCOPE_STEP` 3, the verdict cut-offs, `STAGE1_FLAG_CP` 30, `CASHLESS` gone. Each came from a case; none was swept.
- **`review.main` is still a long function** around `annotate()`; `game_report.py` and `web.user_review` parse the
  review's text output with regexes rather than consuming rows.
- **Python speed**: `relations()` runs per ply of every line; ~1 ms static, ~40 ms with the progress map. Fine for
  one game, the reason the parallel driver exists for many.
- **Cache keys**: the engine cache keys on `(variant, fen, depth, multipv, root_moves)` — a Stockfish upgrade does
  not invalidate it; the Maia cache likewise.

## 5. What the next concept layer stands on

The contract for a concept, as every existing one meets it:

1. **A definition by operators over atoms**, never a hand-written shape: an L1 probe (a legal move, an attack, a
   line through a piece) for atoms; glue, the removal test, the exchange, effective defenders, reach, the null move
   for compounds. The rule-shift test is the acceptance test: the same placement under antichess must change the
   answer the way the rules say.
2. **A `Relation` (or a dataclass with the same shape)**: name, arity, squares in role order, owner by the
   `VICTIM_OWNED` convention, tags for what prose needs, piece symbols captured at observation time. `key` makes it
   diffable by `dynamics.move_effects`.
3. **A footprint** if it goes through the store: what it read, declared into `Reads`, verified by the property test.
4. **One phrase** in `verbalize.rel_phrase` (or the module's own phrase function), one priority in `PRIORITY`, one
   family in `causes.FAMILY` if it can be a mechanism, one line in `significant` if it must be filtered.
5. **A gauge**: a Lichess theme, an enumerator coverage number, or at least a test on a real position from a game.

Where a plan layer plugs in: `search.delta` already emits the change of concept state per move (targets won,
lost, reachable, safe); `structure_diff` says what an even trade changed; `reach.progress` gives every target its
status and its joiner; `verify_plan` tells a plan from a fantasy. A plan is a path whose ledger flips, and the plan
layer's job is to say which path a move served, killed, or created — over these objects, through the store.
