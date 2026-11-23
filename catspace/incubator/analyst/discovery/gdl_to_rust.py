"""PLAN 79.37 PART B: THE INTERFACE OF THE TRANSLATOR FROM A DESCRIPTION TO RUST (SYSTEM_SPEC 11, the blocks "The description is the source of the rules" and "The crux of
layer 2 is a translator ..."; piece: rules; ledger rows 45, 44). This text was written and committed before any body (AGENT_BRIEF 0); the bodies follow in the next commit.

WHAT IT TAKES. A composed description (the base text and a delta, as `cs_gdl::compose` makes it: one text), read by the reader that exists, discovery/gdl.py (the reader of
the Python mirror of the evaluator: `parse`, `clauses`, `Program` with its strata and each rule's join order; a second reader is not written). The translator holds no game: its
source holds only the fixed words of the description language (role, init, true, does, legal, next, terminal, goal, distinct, not, and the input `record`, and the move that
passes, `noop`, which the evaluator's `GdlGame` also reads by that name). tests/test_plan79_37.py scans the source against every relation name the descriptions under rules/ define.

WHAT IT RETURNS. One Rust source file, as text: `translate(text, module) -> str`. It is a module `pub mod <module> { ... }` with the same public items for every game, so a test
can be written once for all of them. The file begins with a head that names the hash of the description and the hash of the translator; a test fails when either differs.
The file is compiled into tests and the timing programs of cs_game from rust/engine/game/tests/generated/; nothing under rust/engine/*/src names it.

WHO CALLS IT. `python -m catspace.incubator.analyst.discovery.gdl_to_rust <description path> [<delta path>] --module <name> --out <path>` (the command line composes with
`cs_engine.compose`, read only), and tests/test_plan79_37.py. The Rust side calls the generated module only from tests under rust/engine/game/tests.

HOW THE THREE THINGS ARE HELD.
  SYMBOLS. Every constant of the description (every symbol of a clause, of a sort, and `noop`) is a number, a `u32`, numbered in the order of the names; the table of names is a
  constant of the file. A COMPOUND TERM (the moves: `(f t ...)`) is a number too, made at run time by one table (`Terms`) that gives the same number to the same compound; its
  numbers begin where the constants end. So a row of any relation is an array of numbers, and two terms are equal when their numbers are.
  FACTS. A relation of n arguments is a `Rel<n>`: its rows (arrays of n numbers) in the order made, a hash set of them (is this fact true), and for each set of argument places
  the joins ask by, an index from the numbers at those places to the rows. The places an index is built on are read from the join orders by the translator and fixed in the file.
  THE STATE. The set of true facts, one sorted list of rows for each fluent (a relation named by `(true (f ...))`): `State { f_<fluent>: Vec<[u32; n]> ... }`; equality and hash of
  states are those of the lists. The state of a turn is loaded into a `Dy` (the relations of one evaluation: the state, the move, the relations the state decides); the relations
  that do not read the state live in `St` (made once).
HOW A RELATION THAT DOES NOT READ THE STATE IS TOLD FROM ONE THAT DOES. By the dependency graph of the description, as the evaluator does it: a relation reads the state when it is a
`true` fluent, `does` or `record`, or depends on one that is (through any positive or negated literal). The others are STATIC: their strata are evaluated once, in `Game::new`, into
`St`, and never again; the others are evaluated at each question that needs them, into `Dy`.
HOW A RELATION IS MADE A PIECE. The relations are cut into strongly connected sets, each after every set it depends on (the evaluator's strata). Each set is ONE FUNCTION of the
file (a piece): a nonrecursive relation is one function over the rows of the relations below it; a recursive set is evaluated by the semi-naive rule (each round joins one literal
against the rows new in the last round). Each clause of a set is one function written as nested loops over its literals in the join order the reader chose; a literal with places
bound by the loops above it is read through the index on those places; `(not ...)` is a lookup of the whole row in the set; `(distinct ...)` is a comparison. No clause and no
relation is named by the translator: every name comes from the text.

THE QUESTIONS (the public items of the module; every method of `Game` is below; a question loads the state, runs the pieces its answer depends on, in order, and reads one relation):
  Game::new() -> Game                              symbols, the static pieces computed once
  roles() -> &[u32]                                the numbers of the roles, in the order the description lists them
  initial() -> State                               the rows of the `init` facts
  mover(&mut self, &State) -> Option<usize>        the first role with a legal move that is not `noop`; None at an end
  moves(&mut self, &State) -> Vec<u32>             the legal moves of the mover (terms)
  next(&mut self, &State, u32) -> State            the `next` facts under the joint move (the mover's move, `noop` for the rest)
  terminal(&mut self, &State) -> bool              whether `terminal` has a row
  outcome(&mut self, &State) -> Option<Vec<i64>>   the `goal` of each role at an end
  and the readings the tests need: sym, term, render, state_of_facts, facts_of, static_relations, all_relations, and the profile of the pieces (calls, time, rows, joins).
The five questions are the evaluator's (rust/engine/game/src/gdl_game.rs, `GdlGame`) in the same words, on numbers.

WHAT IT DOES NOT DO (plan 79.37 is a measurement): no pass for speed. The only things that are free and done: symbols are numbers, the relations that do not read the state are
computed once, an index is built where a join asks by bound places. A clause or a form the plain translation cannot handle is an error that names it, never a clause left out.
"""
from __future__ import annotations


def translate(text: str, module: str) -> str:
    """The Rust source of the description `text` (composed) as `pub mod <module>`. See the module text for the interface."""
    raise NotImplementedError("plan 79.37 part C")
