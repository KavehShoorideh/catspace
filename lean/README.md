# lean/ — the idea basis in Lean 4

Decision (Kaveh, 2026-10-04): Lean is in, for **targets and pruning, never search**. A proved
lemma with individually evaluable hypotheses says (a) the conclusion holds with no search and
(b) when exactly one hypothesis fails, that hypothesis names the search target. The zone search
(`catspace/incubator/analyst/zone_search.py`) stays the searcher. This reverses the one-line
rejection in `docs/ANALYST_PLAN.md` §37 (2026-09-21), which was about search speed.

Source plan: "Lean Chess — Idea Basis & Live Engine: Handoff Plan" (Oct 4, 2026).

## Layout

- `ideas/` — our Lake package.
  - `Ideas/Schema.lean` — the `Idea` record (named hypotheses, protagonist, tier, horizon), `evalUntil`
    (stop at the second failure), and the array-backed board (`boardOfArray`, `memoBoard_eq`).
  - `Ideas/Fast.lean`, `Ideas/FastMate.lean` — fast primitives proved EQUAL to the kernel's (a ray walk for
    `squaresBetween` by kernel `decide` over 4,096 pairs; a per-square attacker table; `isAttackedB'`,
    `legalNormalB'`, `flightOkB'`, `noCaptureB'`, `noBlockB'`).
  - `Ideas/Mate.lean` — `checkmate_of_unmeetable_check`: a check with no flight, no capture of the checker and
    no interposition (each over-approximated) is checkmate. The core under every mate lemma.
  - `Ideas/BackRank.lean` — lemma 1, the back-rank mate (nine hypotheses), `backRankMate_sound`.
  - `Ideas/Passer.lean` — lemma 2, first form: the unstoppable passer against a bare king (rule of the square),
    `unstoppablePasser_sound'`; also `forcedWithinB` (reach a goal within n plies whatever they do) and the
    path geometry.
  - `Ideas/Passer2.lean` — lemma 2, second form: enemy king and pawns (a pawn-distance bound to the bad squares,
    a king-room pigeonhole against self-stalemate), `unstoppablePasser2_sound`.
  - `Main.lean` — the `ideas` binary: `bb` (back rank), `passer`, `passer2` over python-chess bitboards in hex;
    `backrank`/`passerfen`/`legal`/`checkmate` over FEN for the bridge; `bench`.
- Rule kernel: [Arthur742Ramos/chess-lean](https://github.com/Arthur742Ramos/chess-lean)
  (Apache-2.0; Lean 4.32.0, Mathlib v4.32.0), pinned by commit in `ideas/lakefile.toml`. Not vendored.

Every theorem depends only on Lean's standard axioms (`propext`, `Classical.choice`, `Quot.sound`): no
`sorry`, no `native_decide` (chess-lean's `allSquares_nodup` is `native_decide`; `Passer2` reproves it).
Check with `#print axioms` after every theorem.

Our side has no rook, bishop or queen in both passer forms: a push could otherwise uncover a check and break
the no-stalemate argument. The detector reports a pawn PROVED when either form holds.

Like `rust/`, this directory is not part of the `catspace` wheel; Python talks to the built
binary over stdin/stdout (`catspace/incubator/analyst/lean_ideas.py`).

## Build

```bash
curl https://elan.lean-lang.org/elan-init.sh -sSf | sh     # once; installed 2026-10-04
cd lean/ideas && lake exe cache get && lake build            # Mathlib cache: several GB, once
echo "backrank 6k1/5ppp/8/8/8/8/8/3R2K1 w - - 0 1" | .lake/build/bin/ideas
```

The first full build compiles the kernel (about 4.5 minutes, most of it the kernel's own perft
and mate regressions); after that `lake build` takes seconds.

## Tests and gauges

```bash
.venv/bin/python -m pytest tests/test_analyst_lean_ideas.py -q
.venv/bin/python -m catspace.incubator.analyst.lean_backrank_puzzles --bridge 300 --offset 17
.venv/bin/python -m catspace.incubator.analyst.lean_backrank_puzzles --mate 2 --n 300 --offset 17
.venv/bin/python -m catspace.incubator.analyst.frame_puzzles --stride 97 --offset 17 --theme backRankMate --proved-mates 1
.venv/bin/python -m catspace.incubator.analyst.lean_passer_check --form 2 --their-pawns 3 --n 2000   # the race-search bridge
.venv/bin/python -m catspace.incubator.analyst.frame_puzzles --stride 97 --offset 17 --theme pawnEndgame --proved-passer
```

Lean traps met here (details in `WORKLOG.md` 2026-10-05): a docstring cannot precede `set_option`; `subst h`
eliminates the right-hand variable; `cases` on an equation between pieces of different colours fails (use
`simp at h`); `rfl` cannot reduce `c == c` for a variable colour (go through `Bool.eq_iff_iff`); a `let` used
once inside a lambda is floated into it by the compiler (`@[noinline]`); `omega` needs every `Fin` bound in
scope.
