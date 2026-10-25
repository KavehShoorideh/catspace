# RUST_BOUNDARY: where Python ends and Rust begins (plan 72.7, research, 2026-10-23)

Direction (Kaveh 2026-10-23): "everything that's not machine learning should live in Rust." This note is research and a plan. No engine code was
written or run. Every number below is from the 2026-10-23 cProfile of a 16 s run (given with the task) or from a grep of the tree today; the gains are
expectations, not results (AGENTS rule 10).

## 0. Where things stand today (read from the tree)

- The Rust core is `rust/analyst_core` (about 13.3k lines, pyo3 0.25, shakmaty 0.27, rayon). It is far more than the words: `regress.rs` (ports of
  `mate_defects`, `material_defects`, `_still`, `forcing_why`), `route.rs` (schema verification), `sched.rs` (scheduler core), `agenda.rs` (event-loop
  kernel), `vocab.rs` (the words, the certifier), `plan.rs`, `threat.rs`, `rules.rs`, `maia.rs`.
- `src/pos.rs` already is a position handle: `#[pyclass] Pos { pos: Chess, last_to }` with `push(uci) -> Pos`, `fen()`, `balance(white)`, `legal_count()`,
  `mate_in_one()`, `is_check()`, `turn_white()`, `replies()` (uci, balance change, check, capture, recapture, mate-in-one-after), `diff(other)`, `exchange()`.
  Its header measures the reply loop at 15.4 us per reply with python-chess against 0.4 us with `Pos` (journal 2026-10-18).
- But the handle is used narrowly. `agenda.py` has about 20 `AC.Pos(...)` sites and most are `AC.Pos(b.fen())` at the top of a function (lines 1142,
  1730, 2409, 2682, 3667, 3905 and others), so a node's FEN is rendered and parsed again at each. `Pos` is passed down as an argument in the fix
  functions (`P2`), which is the right shape, but it is created from a python-chess `Board` each time.
- Every other Rust call is a free function taking a FEN string: `local_check(fen, uci, k)`, `prove_tree`, `frame_line`, `target_profile`, `fact_pool`,
  `threat_claims`, `corner_check`, and the vocabulary reader `selfmine.val_at(fen, term, mover)` (29 call sites in discovery/). `val_at` also renders the
  term to an s-expression (`BV.sexpr(render(term, mover))`) that Rust parses per call; it is memoised on (fen, term, mover), which hides the cost on
  repeats but not on first reads. `graph.fen_of` has a per-board FEN cache for the same reason.
- python-chess is everywhere: `chess.Board` 110 mentions in discovery/. In agenda.py, graph.py and regress.py alone: `board.piece_at` 52, `board.san` 33,
  `board.copy` 29, `board.fen` 17, `gives_check` 9, `is_capture` 8, `attackers` 8, `king` 7. Also `chess.Move` 92, `chess.SquareSet` 32,
  `chess.between` 12, piece and square constants. `regress.py` has 185 board/move/push/legal-move mentions; `with_turn` is used 59 times in the three files.
- The already-Rust twins must reproduce python-chess's legal-move order (`regress.rs` has `py_key`). That is a hidden coupling: Python list order sets
  tie-breaks, so traces depend on it.

## 1. What other people use (sources)

Move generators in Rust:
- shakmaty (niklasf), the core's dependency: full rules and variants, FEN/UCI/SAN, Syzygy, maintained. https://github.com/niklasf/shakmaty
  The Grand Chess Tree leaderboard lists shakmaty 0.27.3 at about 371M nodes/s single thread without cache. https://grandchesstree.com/leaderboard
- cozy-chess (analog-hors): legal movegen only, no SAN; about 602M nodes/s for 0.3.4 on the same leaderboard; PEXT is optional and slower on AMD before
  Zen 3. https://github.com/analog-hors/cozy-chess , https://docs.rs/cozy-chess/latest/cozy_chess/
- `chess` (jordanbray): fast older generator, apparently unmaintained, forks exist. https://github.com/jordanbray/chess
- pleco (Stockfish rewrite): I found no current speed data; treat as unverified. A third-party harness compares them: https://github.com/alex65536/chess_bench
  (its note: perft alone is unfair between legal and pseudo-legal generators).
- GigaChess claims 540M nodes/s with a shakmaty-compatible API; self-reported only. https://github.com/itshak/gigachess-rs

Reading: raw perft differs by about 1.5x between the good ones. Our hot path is not perft; it is Python-to-Rust crossings and allocation, which a faster
generator does not touch. Keep shakmaty: the core is built on it, it has SAN and UCI parsing (the play server and trace text need them), and the Lean
mirrors were checked against it. Revisit cozy-chess only if a profile of Rust-side time after step 3 shows movegen at the top.

Precedent for a Python chess program moving its board to Rust:
- A PGN pipeline replacing python-chess with shakmaty: about 25 min to 15 s for 1M games (one anecdote, parse-bound, not a handle design).
  https://dev.to/chrnx/processing-1m-chess-games-in-15-seconds-with-rust-pe3
- cozy-chess-py: a pyo3 binding with a Pythonic board class (operators, iteration, copy). https://pypi.org/project/cozy-chess-py/
- A report on optimising a Python chess engine, for the Python-side baseline. https://hps.vi4io.org/_media/teaching/autumn_term_2024/stud/nthpda/frederick_hennecke.pdf
- I found no drop-in Rust replacement for python-chess's `Board`. Plan on our own thin wrapper (section 3.3).

pyo3 patterns:
- A `#[pyclass]` is a Rust value owned by a Python object. A method call costs an attribute lookup, argument conversion and a runtime borrow check; the
  per-call overhead is acknowledged upstream. So cut the NUMBER of crossings and the conversions (String/str, Vec/list), not the crossing itself.
  https://codspeed.io/PyO3/pyo3/branches/MatthieuDartiailh%3Amethods-call-overhead
- Return tuples of ints/bools or one packed Vec per batched call; bind a method once outside a loop if a loop must remain.
  https://medium.com/@noahkim0723/pyo3-performance-tips-for-calling-python-objects-methods-146b3cb40b72
- `#[pyclass(frozen)]` over immutable data drops the borrow check and makes the object shareable across threads; release the GIL (`allow_threads`, named
  `detach` in recent pyo3) for batches long enough to matter. The core's `local_check_many` already batches this way.
  https://docs.rs/crate/pyo3/0.29.2/source/guide/src/class/thread-safety.md
- `Pos` is `Clone` and `push` returns a new `Pos`: immutable after construction, which suits a search tree where each Python node owns its position and
  parents stay valid.

## 2. What stays in Python, and why

- torch and the graph policy (`graph_policy.py`, `question_prior.py`) and any training: the machine-learning part.
- Orchestration: scripts, CLI, the play server's HTTP layer, configs, the registry, run logs and OBS lines.
- Tests and fixtures; they assert on the trace text and read results through the Python surface.
- Thin glue turning Rust results into the objects the policy needs (feature vectors for the prior).
- Things read once per run (loading a puzzle set, final reports).

Everything else (words, chase, leaf, agendas, routes, defect/fix/forcing logic, exchange, reach, legal-move work) is a candidate. Order matters more than the list.

## 3. The boundary design

### 3.1 One handle per node
Each Python node (Agenda, Question, Route) holds a `Pos` built ONCE: at the root from a FEN, below it by `parent.pos.push(uci)`. No Python function rebuilds
a position from `board.fen()` afterwards. Extend `Pos` (do not add a second class); additions are read-only on `&self`, so `#[pyclass(frozen)]` becomes possible:
- `fen_exact()` (replaces `graph.exact_fen`), `key()` (u64 hash; replaces `_pos_key(b.fen())` and `board._transposition_key()`), `with_turn(white)`
  (replaces `graph.with_turn`, including its en-passant clearing), `legal_ucis()` in python-chess order (the `py_key` order from `regress.rs`),
  `piece_at(sq)`, `attackers(sq, white) -> u64`, `attacks(sq) -> u64`, `is_capture(uci)`, `gives_check(uci)`, `san(uci)`, `king(white)`, `board_fen()`,
  `ep_square()`, `is_checkmate()`, `is_stalemate()`, `pass_turn()`.
- Squares are ints 0..63 (a1 = 0, the same numbering in python-chess and shakmaty); bitmasks are u64. Python wraps a mask in `chess.SquareSet` only where
  set operations are needed.

### 3.2 Words on the handle, batched
Replace `val_at(fen, term, mover)`, `graph.word`, `graph.see` by handle methods:
- `AC.compile_term(sexpr) -> int` once per term; `pos.word(term_id, white)`. No s-expression is rendered or parsed per read.
- `pos.words(term_ids, white) -> list[int]`: many words on ONE position in a single crossing. Callers that read several words on one board (the leaf,
  the fix functions, `_standing_kill`) collect ids and ask once. A word cache keyed (pos key, term id, side) lives in Rust, so the 39% exact repeats noted in
  52.20 stay free.
- `pos.see(sq)`, `pos.exchange()` as methods.
- The free functions that take `fen` (`local_check`, `prove_tree`, `frame_line`, `target_profile`, `threat_claims`, ...) get `Pos`-taking siblings built on
  the inner functions they already call (`local_check_pos`, `prove_pos`, `corner_check_pos` exist). The FEN versions remain as shims for tests and scripts.
- Many positions at once: `AC.words_many(pos_list, term_ids, side)`, releasing the GIL and using rayon, for the one place with real fan-out (all replies of a
  route). Needs `Pos` frozen.

### 3.3 python-chess Board: bridge, then retire
Do not rewrite 4,000+ lines of Python in one go. Two layers:
1. Bridge (step 1). Python keeps its `chess.Board` for the parts not yet ported, and the node carries `.pos` beside it. One function `Node.child(uci)` pushes
   both (`board.push`, `pos.push`); no FEN is rendered. A debug flag asserts `pos.fen() == board.fen()` after each push (on in the guard run, off otherwise).
2. Retire (steps 2 and 3). A thin Python class `RBoard` wraps a `Pos` and exposes exactly the subset the code uses: `turn`, `legal_moves`, `push`, `copy`
   (free, since `Pos` is immutable), `piece_at`, `attackers`, `attacks`, `is_check`, `gives_check`, `is_capture`, `san`, `fen`, `king`, `ep_square`,
   `board_fen`. The counts in section 0 define the subset; grep-verify before writing it (anything outside is a finding, not a silent gap). Moves stay
   `chess.Move` (a cheap value type, 92 uses) until the ported code stops producing them; ucis are the currency across the border. When no discovery module
   imports `chess.Board`, python-chess remains in tests, puzzle loading and the play server's I/O edge only.

## 4. Migration order, each step behaviour-identical under the trace guard

The guard is `scripts/trace_identity.py`: `snapshot <dir>` writes the traces of a fixed case set (eight acceptance puzzles plus the discovery position and
`pos2`) at the COUNT budget (deterministic, no clock); `compare <before> <after>` diffs them with seconds masked and exits 1 on any difference. Protocol per
step: snapshot at the parent commit, make the step, snapshot, compare, and only then run the test gate once (one file at a time). A step that changes one
trace line is not done, whatever its speed. Each step also gets one negative control (a deliberately wrong build) to show the guard fires; the guard compares
trace text only, so it has not been shown to catch a read-set change (see risk 2 and step 0).

### Step 0. Make the guard sharper (prerequisite)
Extend the snapshot with the read set of every claim; add the SAN cross-check and the legal-order cross-check of section 5 over the case set and a sample of
the puzzle sets. Without this, steps 2 and 3 can break the listener silently.
STATUS: BUILT 2026-10-23 on branch worktree-agent-a506db989ebdb7e32 (journal 2026-10-23 20:43): `<id>.reads.txt` beside each trace, `trace_identity.py crosscheck`, `Pos.san`,
`Pos.legal_ucis`; the cross-check clean on its sample; the negative control (claims reading one square less) caught by the read sets where the traces did not move.

### Step 1. The handle and the word calls on it
Add the `Pos` methods and term-id compile of 3.1/3.2. Switch `graph.word`, `graph.see` and the `selfmine.val_at` callers to the node's `Pos`; give nodes a
`.pos`; replace the `AC.Pos(b.fen())` sites in agenda.py by it; delete `fen_of`/`_FENS` when the last caller is gone. This removes the FEN round trip (the
20% share) and touches no decision logic: the Rust code behind it is the same inner functions. Python still does board copies, pushes and legal moves.
STATUS: BUILT 2026-10-23 on branch worktree-agent-a506db989ebdb7e32 (journal 2026-10-23 21:42): the 3.1 methods, `compile_term` and the words on the handle, the word cache
replaying the read set, the Pos siblings, `Agenda.pos` handed down by push, `graph.Node`, `fen_of`/`_FENS` deleted; guard 0 of 10 traces and 0 of 10 read sets differ. Not
built: a stored `.pos` on Question / Route (their positions are the agenda's handle and `P1 = pos.push(uci)` where they are read), and handles for the four FEN-taking
binders (`bind_mate`, `bind_mate2` x3) that build a python-chess board from the FEN.

### Step 2. The chase's reply loop and the static leaf
Port `_chase_route_body` (agenda.py 2375 on) and its helpers (`_standing_kill`, `_closed_by_rule`, `_principal_captures`, `_their_after_capture`,
`_unmet_fix`, `_promote_stands`, `_refuter_fix`) to Rust over `Pos`, one call per route returning the structure Python needs ({value, killed, gap, parts,
lemmas} plus the replies that still need an inner agenda). Inner agendas stay Python-driven at first, through the "pending" protocol the cheap pass already
uses (closed replies kept, pending ones handed back). Reply order, memo keys and the `ctx.routes` cache key must match exactly; the key becomes
`(pos.key(), uci + suffix)`. The static leaf goes with it: `leaf_value`, `frame_leaf`, `promo_gain` are already Rust; the Python leaf glue and the reach-price
callers (plan 72.1/72.2) move. `RBoard` can replace `chess.Board` in agenda.py at the end of this step.
STATUS 2a: BUILT 2026-10-24 on branch worktree-agent-a501cc77b08a4de59 (journal 2026-10-24 20:20): `rboard.RBoard` (the subset above plus `is_castling`, `is_attacked_by`, `attacks_mask`, `with_turn`;
python-chess's `is_legal` castling aliases; the bridge's python-chess shadow under CATSPACE_BRIDGE_CHECK=1), read by the loop and the leaf where they read boards of their
own; every board a word reads is built as before. Guard 0 of 10 traces and 0 of 10 read sets differ; the negative control (one legal move dropped) moved 1 of 10.
Measured first: the loop's and the leaf's own code is ~3.5% of the profiled count-budget run; the time is in the inner / their agendas and the words (step 3).
STATUS 2b: BUILT 2026-10-24 on branch worktree-agent-a501cc77b08a4de59 (journal 2026-10-24 20:47): `Pos.leaf_replies` / `Pos.leaf_tied`, the static leaf's decomposition as a record (ucis, values, SAN, handles,
the trackers' additions); Python keeps the verdict and the text; their agendas at the leaf and `stuck_threats` (a word) stay Python. The guard now writes the leaf's
verdicts (`<id>.leaf.txt`): two wrong leaf builds had moved no trace and no read set. Guard 0 of 10 traces, read sets and leaf lists differ.
STATUS 2c: BUILT 2026-10-24 on branch worktree-agent-a501cc77b08a4de59 (journal 2026-10-24 21:15): `Pos.route_table` -- one Rust call per route: the replies with their balances, checks, captures, mates in one, the RULE
that closes a reply with no agenda ("dead", "mate") and the handles; the pending replies are read in Python as before (inner agendas, their pass, the leaf). The memo key
is `(pos.tkey(), uci + suffix)` (`Agenda._route_key`): the EXACT transposition key, not the 64-bit `key()`, which could serve a proof to another position on a collision.
Guard 0 of 10 traces, read sets and leaf lists; the negative control (one reply dropped) moved 10 of 10. THE MEASURE: the step-2 surface fell from ~3.5% to ~1.2% of the
profiled count-budget run; the case set ~2% faster; W0VFO at 10 s read the same work. The speed is in the words (step 3): half the profiled run is python-chess called by them.

### Step 3. The words themselves
Make the Python twins of `regress.py` (`mate_defects`, `material_defects`, `_still`, `forcing_why` and the rest of defects/fixes/forcing) call-throughs to
`regress.rs`, then delete them, keeping the Python versions only as test oracles while the guard runs on a wider case set (30 to 50 positions beyond the ten).
Then the Python words in `graph.py` (`piece_moves_to`, reach, ...) and the last python-chess uses in discovery/. The 90-puzzle timed grade stays the
end-of-plan evaluation, not a per-step grade (standing rule).
STATUS 3a: BUILT 2026-10-24 on branch worktree-agent-afa08f337b471dd18 (journal 2026-10-24 22:48): the mate in two in Rust (words.rs `bind_mate2_pos`, `mate2_threat_pos`: the
binder's candidates and the premise of `mate_in_two`, its terms compiled from the s-expressions Python renders, read by the same evaluator and word cache); selfmine.bind_mate2 /
regress.mate2_threat call-throughs, the Python bodies the test oracles. THE GUARD for step 3: `--sets wide` (99 positions), `--tier K` (the count-budget guard runs tier 0, where
the mate in two is never asked: a wrong build moved 0 of 10 there and 10 of 10 at tier 2), PYTHONHASHSEED pinned (2 of the 99 depend on the string-hash order), the Syzygy tables
linked (gitignored: a worktree has none). Guard 0 differences on the ten and the 99 at tiers 0 and 2 (traces, read sets, leaf lists). The profile first: under the clock the mate
in two was a third of W0VFO's run; at the count budget the passer family is the top (3b, 3c).

Do not interleave steps: each diff is revertable alone and the guard locates a regression to one step.

## 5. Risks (what can silently change)

1. Lean mirror citations. Each word's docstring cites a Lean definition in `lean/ideas` (for example `Ideas.ExchangeLaw.seeXB`, `Ideas.Vocabulary`,
   `Reach.reachPrice`; `docs/LEAN_INDEX.md` is the index). A port that renames, merges or "simplifies" loses the citation. Rule: move the docstring with the code
   verbatim; one test per word against the Lean statement it mirrors (AGENTS rule 11); where a Python function and its Rust twin both mirror Lean, compare
   the two on the guard's positions. A mismatch is a bug in one of them, not an accepted difference. Rebuild the index after any new theorem.
2. READ SETS. Every claim records the squares it read, and the listener re-asks when `Pos::diff` says one changed. The Rust dependency trackers (`dep_add`,
   `dep_part`, the `DEP` thread-locals in lib.rs) are per-thread globals. A chase moved into Rust must accumulate the read set on the same path and return
   it WITH the result; the new Rust word cache must replay the read set on a hit, or a claim looks narrower than it was and the listener misses a change.
   This is the largest correctness risk and the reason for step 0.
3. Trace text. Tests assert on it. Keep formatting in Python at first: Rust returns structured records (ints, ucis, squares) and Python formats as today.
   Move formatting only after the records are stable.
4. Order. python-chess's legal-move order sets tie-breaks (`py_key`); shakmaty's native order differs. `legal_ucis()` must use python-chess's. Any Rust
   `HashMap` iteration that reaches output must be sorted or use an ordered map.
5. SAN. python-chess and shakmaty SAN can differ in corners (disambiguation, check/mate suffixes). Names appear in trace text and the play server. Check
   all legal moves of the guard's positions plus a sample from the puzzle sets before relying on it; until clean, SAN stays computed by python-chess from
   the bridge board.
6. Play server and deployment. They consume the engine through the end-to-end approach interface. Keep the Python surface (`Agenda`, `State`, `decision()`,
   `trace()`) unchanged and the server needs no edit. The wheel must include the compiled core; check `scripts/check_approaches.py` and the container
   recipe, since more of the engine now depends on it. AGENTS rule 3 still holds (shipping code never imports `experiments/`).
7. Panics and illegal input. A Rust panic across pyo3 surfaces as `PanicException` (a BaseException) and can kill a long run: return `PyResult`, test
   illegal moves and FENs. `play_unchecked` on a move not taken from `legal_ucis()` corrupts silently.
8. rayon with the thread-local trackers and hints: results must not depend on which thread ran; batched parallel calls are pure reads only.
9. Two sources of truth during the bridge. The debug assertion in 3.3 is the control; do not ship the bridge past step 2.
10. Builds on the shared machine: no `cargo build` while another agent runs a timed run; `rust/analyst_core/target` is never committed.

## 6. Expected gain per step (expectation, not a result)

Shares, from the 2026-10-23 profile of a 16 s run: about 20% FEN string building for the Rust words, about 25% python-chess board copies, pushes and
legal moves, the rest Python words and the chase.

- Step 1: removes most of the 20% FEN share (the FEN cache and the s-expression render go with it) and adds a small per-call handle cost. Expectation: 12 to
  18 points of wall time, roughly 16 s to 13-14 s. The repeats are already memoised, so the gain is on first reads; the profile does not split first reads
  from cache hits, so this is uncertain.
- Step 2: removes most of the 25% board work in the chase (15.4 us vs 0.4 us per reply, measured in 2026-10-18) and its Python glue. Expectation: a further
  15 to 25 points. Python callbacks for inner agendas keep part of the cost until step 3, so the low end is likelier.
- Step 3: the rest of the Python words and defects. Expectation: another 20 to 35 points of the original, leaving Rust search, the policy and I/O. Redo
  the profile afterwards; the remaining Python (policy, orchestration) sets the floor.
- Together: 3x to 5x less wall time is plausible; 2x is the cautious figure. These are guesses. The first timed run after step 1 gives the first real number;
  revise the plan after it.

## 7. Effort (agent-days, one agent, including guard runs and tests)

- Step 0, sharper guard and cross-checks: 1.
- Step 1, handle + word calls + `Pos` siblings of the FEN functions + replacing the FEN sites: 3 to 4.
- Step 2, chase reply loop + static leaf + their agendas (the part of agenda.py with the read sets): 6 to 9.
- Step 3, rest of regress/graph words, `RBoard` retirement: 6 to 8.
- Total about 16 to 22 agent-days; step 2 carries most of the uncertainty (the read-set protocol).

## 8. Recommendation

Keep shakmaty. Extend the existing `Pos` into the per-node handle. Go 0, 1, 2, 3. The open question for the owner is how long python-chess's `Move` and
`SquareSet` types survive before ucis and u64 masks are the only currency; this plan keeps them until step 3 on purpose.
