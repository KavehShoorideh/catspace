# PLAN 76.4 (d) THE SECOND GAME, THE ACID TEST -- the brief for its own agent (written by the 76.4c agent, 2026-10-29: (a)-(c) took that run's time)

READ docs/AGENT_BRIEF.md first and follow it. Then docs/ANALYST_PLAN.md 76 and 76.4 (with its 76.4c BUILT note) and the journal entries of 2026-10-29 for 76.4c.

## The task
CHECKERS (English draughts: 8x8, the 32 dark squares, men step diagonally forward, kings both ways, a capture is compulsory and a capturing unit keeps
jumping while it can, a man reaching the far row is crowned and its move ends, the side with no legal move loses; the draw by agreement is no rule -- write
the draw the WCDF rules state, or leave it out and say so, as chess.gdl left threefold out at 76.0) as a GDL rule set `rules/checkers.gdl` with the 76.1 (b)
extensions (sorts, signatures; the fluent bindings, the move binding) -- then:

1. A FAST KERNEL: a small Python kernel (`discovery/checkers_kernel.py`: legal moves, next state, terminal, goal) and CROSS-PLAY against the GDL interpreter
   (gdl.py, unchanged) on a large random sample of positions, as scripts/crossplay.py did for chess (76.0 (b)): every disagreement shrunk and filed as a surprise
   (rules/surprises/, the same schema).
2. THE LEAN GENERATION: `scripts/gdl_to_lean.py` UNCHANGED on rules/checkers.gdl building with no sorry (a new Ideas/Spec/Checkers.lean); an extension the
   translator needs is a FINDING, said so.
3. THE ONTOLOGY: the checkers entries (the location as the shelf's one SUPPLIED product object, the roles, the placement fluent of (location, side, unit), whose
   turn, the opponents) -- terms.Domain finds every sort BY ROLE; it must load checkers with no code change (the generic test of tests/test_plan76_4c.py
   extended to the new game: terms.py, object_miner.py and valuation_miner.py name no object of either game). Decide and say how the shelf holds two games (a
   directory per game, or a `game` field): today ontology.DIR and gdl_chess.SPEC are one game's.
4. MATERIAL (76.4c (c)'s generator, unchanged): men against kings fit only to outcomes -- table-judged endings if a checkers endgame database is available,
   else self-play records of the kernel (name the corpus and its size); the priced search, the mined worths, the fit printed beside a hand reading if one is
   supplied (the usual one: a king worth about one and a half men).
5. AN OBJECT (76.4c (b)'s generator, unchanged): one technique or object found -- the regression's readings need GOAL entries with `gdl` heads for checkers
   (the outcome: a win is the other side with no legal move); the derivations' readings and flips, the obligations' cited sets are generic.

ZERO CODE OUTSIDE rules/checkers.gdl AND THE KERNEL (and the shelf's checkers entries). Every place where chess leaked into "generic" code is a FINDING, printed.

## What the 76.4c agent read as likely leaks (unverified; check each)
- ontology.py: DIR, SPEC, KERNEL_SRC, SAMPLE, readings_* and agreement are chess's; `validate` cross-checks python-chess and the Rust core.
- gdl_chess.py is the chess binding (state_of, the move term, the native core); the regression (regression.py) is generic but its Regressor reads the native
  core, which loads chess.gdl -- a checkers regression is the Python reading (PyRegressor) over a checkers binding.
- lemma_store.py: the GOAL entries, the VALUATION's letters (p n b r q k), names.py's WORDS are chess's; `valuation_weights` maps letters in gdl_chess.
- technique_measures / technique_miner / technique_sample read python-chess boards and the Syzygy table: the KRvK class machinery is chess-only.
- scripts/plan76_4c.py's corpora (the lichess records, the Syzygy endings) and its references (python-chess's between / rays) are chess's by design.

## Acceptance
The chess guard untouched (0 of 218 on all five lists against the plan's parent); `ontology validate`, `lemma_store validate`; tests/test_rules_checkers.py
(new) in the engine gate; the report as the agent brief says, plus: the cross-play agreement and its surprises, whether the Lean generation built unchanged,
the mined material (men, kings) and its fit, the object found, and every leak.
