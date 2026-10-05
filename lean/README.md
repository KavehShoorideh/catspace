# lean/ — the idea basis in Lean 4

Decision (Kaveh, 2026-10-04): Lean is in, for **targets and pruning, never search**. A proved
lemma with individually evaluable hypotheses says (a) the conclusion holds with no search and
(b) when exactly one hypothesis fails, that hypothesis names the search target. The zone search
(`catspace/incubator/analyst/zone_search.py`) stays the searcher. This reverses the one-line
rejection in `docs/ANALYST_PLAN.md` §37 (2026-09-21), which was about search speed.

Source plan: "Lean Chess — Idea Basis & Live Engine: Handoff Plan" (Oct 4, 2026).

## Layout

- `ideas/` — our Lake package.
  - `Ideas/Schema.lean` — the `Idea` record (named hypotheses, protagonist, tier, horizon) and
    `memoBoard` (the board read into an array once; `memoBoard_eq` proves it is the same board).
  - `Ideas/Mate.lean` — `checkmate_of_unmeetable_check`: a check whose every answer is ruled out
    (no flight square, no capture of the checker, no interposition, all over-approximated) is
    checkmate. The reusable core for every mate lemma.
  - `Ideas/BackRank.lean` — the back-rank mate as an `Idea` (nine hypotheses: heavy, onBackRank,
    legal, oneKing, check, boxed, rankFlights, noDefender, noInterpose), its one-away `targets`,
    and `backRankMate_sound`: all nine hold ⇒ the move is legal and mates.
  - `Main.lean` — the `ideas` binary: `backrank <fen>`, `legal <fen>`, `checkmate <fen>` over stdin.
- Rule kernel: [Arthur742Ramos/chess-lean](https://github.com/Arthur742Ramos/chess-lean)
  (Apache-2.0; Lean 4.32.0, Mathlib v4.32.0), pinned by commit in `ideas/lakefile.toml`. Not vendored.

Both theorems depend only on Lean's standard axioms (`propext`, `Classical.choice`,
`Quot.sound`): no `sorry`, no `native_decide`.

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
```
