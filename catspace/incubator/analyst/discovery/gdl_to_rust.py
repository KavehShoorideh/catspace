"""PLAN 79.37 PART B: THE INTERFACE OF THE TRANSLATOR FROM A DESCRIPTION TO RUST (SYSTEM_SPEC 11, the blocks "The description is the source of the rules" and "The crux of
layer 2 is a translator ..."; piece: rules; ledger rows 45, 44). This text was written and committed before any body (AGENT_BRIEF 0); the bodies follow in the next commit.

WHAT IT TAKES. A composed description (the base text and a delta, as `cs_gdl::compose` makes it: one text), read by the reader that exists, discovery/gdl.py (the reader of
the Python mirror of the evaluator: `parse`, `clauses`, `Program` with its strata and each rule's join order; a second reader is not written). The translator holds no game: its
source holds only the fixed words of the description language (role, init, true, does, legal, next, terminal, goal, distinct, not, and the input `record`, and the turn that
passes, `noop`, which the evaluator's `GdlGame` also reads by that name). tests/test_plan79_37.py scans the source against every relation name the descriptions under rules/ define.

WHAT IT RETURNS. One Rust source document, as text: `translate(text, module) -> str`. It is a module `pub mod <module> { ... }` with the same public items for every game, so a test
can be written once for all of them. The document begins with a head that names the hash of the description and the hash of the translator; a test fails when either differs.
The document is compiled into tests and the timing programs of cs_game from rust/engine/game/tests/generated/; nothing under rust/engine/*/src names it.

WHO CALLS IT. `python -m catspace.incubator.analyst.discovery.gdl_to_rust <description path> [<delta path>] --module <name> --out <path>` (the command line composes with
`cs_engine.compose`, read only), and tests/test_plan79_37.py. The Rust party calls the generated module only from tests under rust/engine/game/tests.

HOW SYMBOLS, FACTS AND THE STATE ARE HELD.
  SYMBOLS. Every constant of the description (every symbol of a clause, of a sort, and `noop`) is a number, a `u32`, numbered in the order of the names; the table of names is a
  constant of the document. A COMPOUND TERM (the moves: `(f t ...)`) is a number too, made at run time by one table (`Terms`) that gives the same number to the same compound; its
  numbers begin where the constants end. So a row of any relation is an array of numbers, and two terms are equal when their numbers are.
  FACTS. A relation of n arguments is a `Rel<n>`: its rows (arrays of n numbers) in the order made, a hash set of them (is this fact true), and for each set of argument places
  the joins ask by, an index from the numbers at those places to the rows. The places an index is built on are read from the join orders by the translator and fixed in the document.
  THE STATE. The set of true facts, one sorted list of rows for each fluent (a relation named by `(true (f ...))`): `State { f_<fluent>: Vec<[u32; n]> ... }`; equality and hash of
  states are those of the lists. The state of a turn is loaded into a `Dy` (the relations of one evaluation: the state, the turn, the relations the state decides); the relations
  that do not read the state live in `St` (made once).
HOW A RELATION THAT DOES NOT READ THE STATE IS TOLD FROM ONE THAT DOES. By the dependency graph of the description, as the evaluator does it: a relation reads the state when it is a
`true` fluent, `does` or `record`, or depends on one that is (through any positive or negated literal). The others are STATIC: their strata are evaluated once, in `Game::new`, into
`St`, and never again; the others are evaluated at each question that needs them, into `Dy`.
HOW A RELATION IS MADE A PIECE. The relations are cut into strongly connected sets, each behind every set it depends on (the evaluator's strata). Each set is ONE FUNCTION of the
document (a piece): a nonrecursive relation is one function over the rows of the relations below it; a recursive set is evaluated by the semi-naive rule (each round joins one literal
against the rows new in the last round). Each clause of a set is one function written as nested loops over its literals in the join order the reader chose; a literal with places
bound by the loops above it is read through the index on those places; `(not ...)` is a lookup of the whole row in the set; `(distinct ...)` is a comparison. No clause and no
relation is named by the translator: every name comes from the text.

THE QUESTIONS (the public items of the module; every method of `Game` is below; a question loads the state, runs the pieces its answer depends on, in order, and reads one relation):
  Game::new() -> Game                              symbols, the static pieces computed once
  roles() -> &[u32]                                the numbers of the roles, in the order the description lists them
  initial() -> State                               the rows of the `init` facts
  mover(&mut self, &State) -> Option<usize>        the first role with a legal turn that is not `noop`; None at an end
  moves(&mut self, &State) -> Vec<u32>             the legal moves of the mover (terms)
  next(&mut self, &State, u32) -> State            the `next` facts under the joint turn (the mover's turn, `noop` for the rest)
  terminal(&mut self, &State) -> bool              whether `terminal` has a row
  outcome(&mut self, &State) -> Option<Vec<i64>>   the `goal` of each role at an end
  and the readings the tests need: sym, term, render, state_of_facts, facts_of, static_relations, all_relations, and the profile of the pieces (calls, time, rows, joins).
The five questions are the evaluator's (rust/engine/game/src/gdl_game.rs, `GdlGame`) in the same words, on numbers.

WHAT IT DOES NOT DO (plan 79.37 is a measurement): no pass for speed. The only things that are free and done: symbols are numbers, the relations that do not read the state are
computed once, an index is built where a join asks by bound places. A clause or a form the plain translation cannot handle is an error that names it, never a clause left out.

PLAN 79.38 PART A/B: THE PASSES (SYSTEM_SPEC 11, "The crux of layer 2 is a translator from the description to Rust, made fast piece by piece under tests"; piece: rules; ledger row 45).
This text is the interface of the passes, written and committed before any body. `translate` grows one argument and nothing else changes for the callers that do not give it.

  translate(text, module, passes=()) -> str
  PASSES = ("dense", "demand", "copy", "delta")   # the names a caller may give; the passes are applied in this order whatever the order given; a name not in PASSES is an error
  (the fourth pass, "delta", was not in the plan: the first time of the hot pieces showed the semi-naive rounds of the rays scanning the rows new in the last round once for every row of a table of steps; it is written below in
  the same form as the others, with its condition, and is one of the choices the report names.)

WITH NO PASS the document is today's, byte for byte, but for the translator hash in its head (a change of this source changes that hash; the files of generated/ are made again by scripts/gdl_to_rust_make.py, and
tests/test_plan79_38.py holds the plain files to differ from those of the commit before the plan (12f17e26) in the line of that hash only). WITH PASSES the head holds one more line, `// passes: <names>`, and the
document has these public items besides the plain ones: `PASSES: &[&str]`, `Game::demanded() -> Vec<(&'static str, Vec<usize>, bool)>` (a relation answered on demand, the places its key binds, and whether the mode only asks
whether a row exists), `Game::ask_many(&mut self, &State, does, &[(rel, key)]) -> Vec<(Vec<Vec<u32>>, Option<bool>)>` (for each query, a demanded relation and a key with FREE at the places not bound: the rows by the way that key is asked, and, where that exact key has a
mode that only asks existence, whether a row exists; one state is loaded for all the queries), so that a test may hold every piece to the plain module. `all_relations` of a module with passes lists the relations made in full; a relation a pass removes or merges is compared by the harness at the first relation
that survives it, and the module says which they are (`Game::demanded`, `COPIED`).

A PASS reads only the shape of the clauses, the sorts, the signatures and the dependency graph of the description (the reader's `Program`: strata, each rule's join order, `signature`, `sorts`); it names no relation, sort member
or constant of any description (tests/test_plan79_37.py's scan and tests/test_plan79_38.py's hold for the whole source); it returns a change of the document only, never of the description; and it is applied only where its CONDITION
holds for the relation or clause, everywhere else the plain translation stands for that piece.

  (1) "dense"   TAKES the signature of each relation and the declared sorts. CONDITION: every place of the relation is of a declared sort whose members are all constants, the product of the sizes of the sorts is at most
                2^22, and the product over the places of each index the joins ask by (read from the clauses that are written) is at most 2^16. RETURNS: that relation held as a bit table over the tuples of its sorts (membership by position,
                no hash) with each index a bucket array by the places' ordinals (no hash), and cleared by its own rows, so that the evaluation of a state is kept and reused instead of made again (`Game` keeps one `Dy`).
                A relation that fails the condition remains a hash relation.
  (2) "demand"  TAKES the dependency graph, the strata and every written clause with its join order. CONDITION: the relation reads the state, is not recursive, is not read by a question (legal, terminal, goal, next) and is
                not an input, and every use of it in a written clause is a literal that binds at least one place of it or only asks that a row exist (a negated literal, an all-bound test, a literal whose unbound variables are
                used nowhere further on); a use that would read the relation whole with no place bound (not an existence test) makes the relation remain in full, and so does a use whose answer, joined from its key, is a loop over a whole relation
                that reads the state (an existence test of no key is exempt: it stops at its first row; a relation that does not read the state, the turn, and a fluent whose sorts allow at most SMALL_SCAN rows are not "whole").
                RETURNS: the relation is not made; each use is answered for its key by the relation's own clauses joined again from the key (`JoinRule`: the reader's choice of order with the key's variables taken as bound, and a demanded
                relation no bound variable reaches taken last), stopping at the first row where only existence is asked; the modes (relation, places bound, exists or rows) are found by closing the uses over the clauses of the demanded
                relations. Where the answer is rows, rows may repeat; the consumers keep a set. The rules that are made in full are joined with the same `JoinRule`, so a demanded relation is never the first literal of a clause that has
                another to start from. Why a relation remains in full is kept (`Plan.refused`) with the clause that defeats it.
  (3) "copy"    TAKES the clauses of each next-state relation of a fluent. CONDITION (general, on the clause): a clause whose head arguments are distinct variables, whose body has exactly one positive literal, of the fluent's
                own `true` relation, with the same variables in the same places, and every other literal a `distinct` or a negated literal whose arguments are variables of that row or constants. Such a clause says: the next
                rows are the old rows that no negated relation names. SOUNDNESS: its rows are a function of one old row alone, so the filter of the old rows by those literals equals the clause's rows; any other clause of the same
                fluent is unaffected and made as before; a clause not of this shape is made as before. RETURNS: the old rows are kept and the filter applied to them in `next` (the removals are the rows the negated relations name),
                the changes (the other clauses) are added; the clause is not run as a join and the next relation is not written again by it. THE GENERAL CONDITION UNDER WHICH IT HOLDS, stated once: the clause is a function of one old
                row alone (one positive literal, the head the same row, the other literals tests of that row by relations the clause does not define); where it does not hold the pass refuses, clause by clause (tests/test_plan79_38.py).
  (4) "delta"   TAKES each rule of a recursive set. CONDITION: the rule has a positive literal of its own set. RETURNS: that rule once for each such literal, written with that literal first, read from the rows new in the last
                round (borrowed, not copied and filtered), the other literals joined behind it in the usual order and read from the whole relation; the rule joined as before for the first round. SOUNDNESS: a conjunction does not
                depend on the order of its literals, the safety of the negated literals and tests is kept by the same rule as before (each as soon as its variables are bound), and the semi-naive rule (one literal of the set from the
                new rows, the others from all) is unchanged.
  A pass that makes a piece slower or gives a different row is a fault of the pass: it is mended, never set aside.

PLAN 79.40 PARTS A, C, D: THE FORM OF A RELATION OVER SMALL SORTS (SYSTEM_SPEC 11, "The crux of layer 2 is a translator ... made fast piece by piece under tests"; piece: rules; ledger row 45).
This text is the interface of 3 more passes, written and committed before any body. PASSES = ("dense", "demand", "copy", "delta", "bits", "order", "keep"); `translate(text, module, passes)` is as above; with no pass the
document is still the plain one byte for byte but for the translator hash. The new passes add no public item to a module: the questions and the items of 79.38 are the same, so the harness holds them to the plain module as before.

  (5) "bits"   TAKES the signature of each relation, the declared sorts, and every written clause with its join order. IT FINDS, from the clauses and signatures alone (told no sort and no product):
                TILES. A sort of constants of at most 64 members is a tile. Two sorts (A, B) are one tile, a PRODUCT, when in one clause the same two variables stand next to each other, in that order, at places of sorts A and B in at least two
                literals (head included) and |A|*|B| <= 64: the pair travels together, so it is one number (a*|B|+b). The places of a relation are cut into tiles from the left, a product where one is found.
                HELD AS BITS. A relation is held as bits when every place has a declared sort of constants, the cut into tiles is made, and for every way a written clause asks it (which tiles are bound) the tile that is left free has at most
                64 members and the product of the sizes of the other tiles is at most BIT_KEYS (2^20: the first writing of this text had 2^16, the size of the keys of an index of the pass dense; the relations that hold what a turn changes are larger, and a word that is never set is never made resident). Its form is then, for each such way (an ORIENTATION: which tile is free), an array of words with one word for each combination of the other tiles'
                numbers, and bit t of the word is set when the row with the free tile's number t is there; a list of the words that are not nought keeps the clearing and the scan proportional to what is held. A row is set in every orientation.
                A state is loaded by setting these bits; the evaluation is kept between questions and cleared by its lists of words (no row is made, hashed or freed: Part D below states what else is kept).
                CLAUSES AS WORD OPERATIONS. A clause whose literals are asked in the join order is written as nested loops over what is NOT yet bound: a positive literal whose tiles but one are bound reads one word (and, for the places of the free
                tile that are bound or constants, ands it with a mask); a literal with no tile bound is scanned by its list of words; a literal all bound is a bit test; a negated literal is a bit test, or an and-not when it stands over the same
                variables as the free tile of a literal before it; a literal over the same variables as that free tile is anded in before the bits are looped over (every later literal of the clause that stands over the same variables of the same
                tiles is folded in so); a literal read only to know that a row exists is `word != 0`; where the variables of the free tile are used by nothing but the head's last tile, the word is or-ed into the head whole, with no loop over its bits.
                A RECURSIVE SET of such relations is evaluated by the semi-naive rule on words: the words new in the last round are the first literal of each rule (scanned by their list), the rest read the whole relation; a round adds its words
                to the relation and to the next round's list. The shape recognised is exactly: every relation of the set is held as bits; this is the walk along a table of successors until a test fails, found as such because it is a clause that
                joins the set's relation, tests the same variables as its free tile, and reads a relation that takes those variables to the next ones, and is compiled by the one rule above, not by a pattern of any game.
                WHAT IT REFUSES, naming the clause: a relation with a place of a sort that has a compound or a symbol that is not a member of the sort at the place; a free tile of more than 64 members, or other tiles with more than BIT_KEYS (2^20: the first writing of this text had 2^16, the size of the keys of an index of the pass dense; the relations that hold what a turn changes are larger, and a word that is never set is never made resident)
                combinations, in any way a clause asks it; a place whose term is a compound; a tile whose two places carry one variable. A relation refused remains as it is (a hash relation, or dense if the pass dense accepts it) and the clauses that
                read it through its rows continue as today: a clause is written in the word form when it reads or makes a relation held as bits, and otherwise as before. A relation answered on demand (pass demand) is not held as bits.
                `Plan.bits_refused` holds each refusal with its clause.
  (6) "order" TAKES the written clauses of a stratum and the number of rows of each relation that does not read the state, known at translation by evaluating those strata once with the reader's evaluator. CONDITION: a clause has two or more
                positive literals that start equally well (the same number of bound places); RETURNS: the clause joined choosing, among those, the literal of the fewest rows (static relations by their number of rows, relations that read the state
                by the size of the product of the sorts of the places that remain unbound, the same everywhere; a literal of the recursive set the clause belongs to counts as half a row, for the rows new in a round are the fewest and are read first), otherwise the reader's order. The reader's is kept where the two have the same count of bound places; the order is a different order of the same conjunction,
                with the same safety rule, and so the same rows.
  (7) "keep"  TAKES the pieces and the questions. RETURNS: the working memory of a question kept between questions where it was made at each: the lists a clause makes, the terms of a compound built again, and the state that a question returns
                built in the vector of a former one. Any place that makes and frees memory in a question where the profile shows it is named in the commit with its share before and then.
  A pass that makes a piece slower or gives a different row is a fault of the pass: it is mended, never set aside. The harness of 79.38 holds each of them to the plain module (tile numbers are an inside of the module: rows are compared as text).
  AS BUILT, where the body differs from the text above (said here because the text was committed first):
    (5) BIT_KEYS is 2^20 words for an orientation (the text had 2^16). A way of asking that would need an orientation outside the limits is answered by the last tile's orientation, the tiles not bound looped over and the last tile tested (the relation is
        dropped only when that orientation itself is outside the limits). The state of a game remains a list of rows (`State` is as in the plain module); a question loads its rows into the words and clears them through their lists when it is done, so no row is
        made or hashed for a relation held as bits, but the state is not itself a fixed array of words (the plan's part A (2) asked for that; not built). A relation is held as bits only when a written clause reads it (a relation only the questions read would
        be made as words to be turned back into rows). The pass bits is decided before the pass demand and the pass dense: what it holds is neither answered on demand nor dense. Recursive sets held as bits are evaluated by their own rounds
        (the pass delta has nothing to do for them). Every refusal and its reason is in `Plan.bits_refused`.
    (6) a literal of the set the clause belongs to counts as half a row (the first version did not, and made the plain translation slower: the semi-naive rule read the rows new in a round only behind a loop over a table).
    (7) the lookup of a compound builds its key on the stack; `Game` keeps the reading of a state, the joint turn and, in a `Dw`, the rows a clause gives and the rows new in a round of a set not held as bits.

PLAN 79.41 PARTS A, B, C, D: AN ALTERNATIVE'S ANSWER AS THE BASE ANSWER PLUS WHAT THE ALTERNATIVE CHANGED, AND WALKS BY WHOLE WORDS (SYSTEM_SPEC 11, "The crux of layer 2 is a translator ... made fast piece by piece under tests"; piece: rules; ledger row 45).
This text is the interface of the passes below, written and committed before any body. PASSES gains the names "base", "shift", "scan" and "cut" (each is added to PASSES in the commit of its body); `translate(text, module, passes)` is as above; with no pass the
document is still the plain one byte for byte but for the translator hash; the passes add no public item to a module but `UNFOLDED: &[&str]` (below), so the harness of 79.38 holds each of them to the plain module piece by piece as before.
WHY (the profile of 79.40, every pass, the first game): two pieces held two thirds of the time, one a set of lines walked again for every alternative, the other a test of each alternative over those lines; the form of their joins was the cost (a key
whose tiles are not all bound was looped over whatever it held), and the board that results from each alternative was made whole as rows for each alternative.

  (8) "base"   (part A) TAKES the written clauses of every stratum (once the pass order has run) and their dependency graph. A VIEW is a nonrecursive relation S, not an input, not a fluent, not read by a question and not in a recursive set, none of whose readers
              reads it negated, with at least one clause that is a PRODUCT: its positive literals fall in two groups that share no variable, each group binding a variable of the head (so the rows S has for each value of what the first group binds
              are as many as the second group's rows: the base relation read whole for each key), and whose head is a list of distinct variables. RETURNS: S unfolded into every clause that reads it: each such clause is replaced by one clause for each clause of S,
              S's literal replaced by that clause's body with its variables renamed to names not used and its head variables replaced by the literal's terms; S's own strata stay in the document and run in `all_relations` only (`UNFOLDED` lists them), never in the cones
              of the questions, so no row of S is made for the questions. An alternative's board is then read as it is: the base relation's words (the same for every key), the words of what the key removes anded out (the negated literals, folded into the word
              by the pass bits) and the words of what the key adds or-ed in (the other clauses of S), and no row is made for each key.
              THE ARGUMENT THAT IT GIVES THE SAME ROWS, under stratified negation. (i) S is nonrecursive, so its rows are exactly the union over its clauses of the rows of the clause's body (the least model of a nonrecursive relation is the union of its clauses);
              (ii) a clause C with S in its body is a conjunction in which S's literal stands for that union, and a conjunction with a union is the union of the conjunctions (distribution), so C is the same as the clauses made by replacing the literal by each
              clause's body; the variables of S's clause are renamed to names not used so that none of C's is captured, and the head's distinct variables are replaced by the literal's terms, which is the unifier of the literal with the head; (iii) the new body holds the literals
              of S's body, which are of relations in strata at or below S's, hence below C's, because S is not in C's recursive set (S is nonrecursive and C reads it, so a cycle through S would make S recursive); a negated literal in S's body remains a
              negated literal over variables that S's positive literals bind, and those literals are in C's body now, so safety and stratification are unchanged; (iv) S is read positively only (the condition): the negation of a union is not a conjunction of
              literals and is not unfolded. A relation that is not covered is REFUSED with the clause that defeats it (`Plan.base_refused`): read negated (the clause named), read by a question, in a recursive set, a head that is not distinct variables, no clause a
              product (nothing is gained: unfolding would repeat S's rows in each reader). THE RECURSIVE SETS: a relation in a recursive set is never unfolded, and a view's reader inside a recursive set is unfolded (its own set is not changed); the argument does not
              cover making the rows of a recursive set from the base answer plus changes (a row of the base that a removed row supported must be made again, through the cycle), so that is refused.
  (9) "shift" (part B) TAKES a recursive set held as bits (pass bits) that is a walk (the shape the pass bits already finds: a clause that joins the set's relation, tests the same variables as its free tile, and reads a relation that takes them to the next ones)
              and the relation that takes them to the next ones, when it does not read the state. RECOGNISES from its rows at translation, for each value of the tiles it is keyed by, the displacements of the free tile's number from a member to each of its next
              ones: the set of displacements, at most SHIFT_MAX of them, and for each the mask of the members that have that next one; it then TESTS the table against every row (the rows rebuilt from the displacements and masks are the rows, exactly) and refuses
              the relation, naming the key and the member, where they are not. RETURNS: a round that, for each key, moves the whole frontier word by one shift and a mask for each displacement (a word and the mask of the members that have that next one, shifted),
              stops at the words with no bit left, and tests the stopping literal as an and of the word with the word of the stopping relation, one word for every member of a frontier; the rounds are one loop over keys, not a pass over lists of words.
              A relation whose rows have no such form (more than SHIFT_MAX displacements for a key, or a member with a next one that is not a displacement of a mask) remains as the pass bits builds it.
  (10) "scan" (part D) TAKES a literal over a relation held as bits some of whose key tiles are bound and some are not. Today the tiles that are not bound are looped over, every combination, whatever the relation holds. RETURNS: the loop is over the words the
              relation holds (its list of non-nought words, checked at the key tiles that are bound) when that list is shorter than the number of combinations, and over the combinations otherwise; one body, the word's number made from the loop in either case.
  (11) "cut"  (part D) TAKES a clause whose head is a row of the relation whose variables are all bound at some literal k (the head is complete there). RETURNS: the literals that follow k as a block that is left at the first row they make (the head row is
              the same for every later row: a relation is a set), so a head row that exists is not made again by every way of making it.
  (12) THE STATE AS WORDS (part C). TAKES the relations held as bits that a state holds (the fluents). RETURNS `State` that carries, besides the lists of rows by which it is equal and hashed and by which a caller gives and reads it, the words of those relations of the
              state as they stand when a question has loaded them; a question on a state whose words are those of the last question (the same state, as `next` follows `moves`) does not load or clear them; the public items are the same. WHERE IT DOES NOT HELP (the
              load of a state is a few hundred nanoseconds of tens of microseconds) it is measured and said, and not built.
  A pass that makes a piece slower or gives a different row is a fault of the pass: it is mended, never set aside. The harness of 79.38 holds each of them to the plain module (the tiles and the displacements are an inside of the module: rows are compared as text).
  AS BUILT, where the body differs from the text above (said here because the text was committed first); PASSES = ("dense", "demand", "copy", "delta", "bits", "order", "keep", "scan", "cut", "shift", "base"):
    (8) is the unfolding of the text, not the form "the base answer made once, and for each key the derivations that read a removed or added row made again" that the plan's words suggest: the argument above covers the unfolding (a union is
        distributed), it does not cover the other form for a recursive set, and it was not needed for the others, since once the view is unfolded the base relation is read as its words, the same for every key, with the words of what the key removes anded
        out and the words of what it adds or-ed in. The pass takes the views of both games it finds (the number of them is printed by the test of the renamed description) and refuses one relation in each, naming the clause: a view read negated (the stopping test of a walk). A relation in a recursive set
        is never unfolded and the walk's own clause is not changed. The views stay in the document and run in `all_relations` only (`UNFOLDED`); the cones of the questions are made from the clauses as unfolded (`Plan.cone`). `base` brings `scan` with it:
        on top of bits alone the unfolded clauses ask words with key tiles unbound, and the module was half as fast as without the pass (diagnosed; scan answers it, and `bits+base` names both).
    (9) is one function for the whole set (the closure of each key's frontier, run behind the base clauses), not the rounds of the pass bits; it is taken for a set of one relation with one recursive clause of the form in the text, and the other walks of
        a description remain as the pass bits builds them. The table is TESTED against every row at translation (`shift_table`); a relation whose rows are not that form is refused, naming the relation, the key and the rows that differ.
    (10) the loop over combinations is kept as the other branch of one body; the number of the word is made from the loop in either case and the tiles are read from it, so a literal with some key tile bound and some not costs the same lines in either case.
    (11) the block is a labeled block of the rest of the clause (a `continue` may not leave one, so a failing test in it is a `break` of its label); a head row is complete when every variable of the head is bound.
    (12) NOT BUILT: the pieces together are most of the wall of a position in the clocks (the profile prints the share), the rest being the load of a state, its clearing and the reading of the answers; the text of (12) let the measurement end it.

PLAN 79.42 PARTS A, B, C, D: THE MACHINERY LEFT IN THE HOT PIECES (SYSTEM_SPEC 11, "The crux of layer 2 is a translator ... made fast piece by piece under tests"; piece: rules; ledger row 45).
This text is the interface of the passes below, written and committed before any body. PASSES gains "start", "orient", "ends" and (only if its argument below is completed) "lines", each added to PASSES in the commit of its body; `translate(text, module, passes)` is as above; with no pass the document is
still the plain one byte for byte but for the translator hash; the passes add no public item to a module but `ENDS: &[&str]` (below), so the harness of 79.38 holds each of them to the plain module piece by piece as before.
WHY (the profile of 79.41, every pass): a few pieces hold two thirds of the time. In the largest the clauses start from the literal with the most places bound and then loop over every word of a relation for each alternative; the walks set each row into every orientation
bit by bit and make rows that no reader looks at. Each cause is machinery, none is a rule of a game.

  (13) "start"  (part A) TAKES a clause that is made as words (the head held as bits, not static), its written order (the reader's, or the pass order's) and the relations held as bits. THE STARTS of a clause are its positive literals over relations held as bits;
              for each, the order that the rule of `JoinRule` gives when that literal is taken first (the others as before, the safety rule unchanged). RETURNS: where two or more starts give different orders, one function for each admitted order and a chooser that is called in its place
              (same name, same arguments): it computes, for each admitted order, an ESTIMATE from the lists of words held now, and runs the order of the least estimate (the first of them when equal). The estimate of an order is the sum, over its literals, of the product of the factors of the literals up to
              and including it, where the factor of a literal is the number of words its relation holds in the orientation read (its list of words) when the literal is looped over the words held, the number of combinations of the tiles not bound when it is looped over those, the number of words held or of combinations whichever is fewer when
              the pass scan chooses between them, 2 for a literal read by key that binds a variable (a loop over the bits of a word) and 1 for any other. ADMITTED: an order whose clause, written, asks no orientation that the clause in the reader's order, and the orders admitted before it, did not already ask (an orientation is paid by every row set in the relation, in every question);
              at most START_MAX orders, those of the fewest literals read by loops over words first. SOUNDNESS: a clause is a conjunction, a conjunction does not depend on the order of its literals, each order is joined with the safety rule of `JoinRule` (a negation or a distinct as soon as its variables are bound), so every order makes the same set of head rows
              (the rows are a set: a row made twice is one); the estimate only chooses which order runs and can make no row differ; the harness holds it to the plain module. A clause with one order, or whose other orders ask a new orientation, is as before, and `Plan.start_refused` names the clause and the orientation.
  (14) "orient" (part B) TAKES a relation held as bits that a function sets by whole words (a walk made by the pass shift, or a clause that ors a word into its head) and the orientations the written clauses ask of it. RETURNS: the word is or-ed into the orientation the function itself makes (the last tile's), and each other orientation that a written clause reads
              is made from the new bits of that word by a function written for that orientation: the part of the word's number that does not change with the bit is computed once for the word, the part that does is a product of the bit and a constant, and the list of words is pushed once for each word first set; an orientation that no written clause reads is not made (as before). SOUNDNESS: the rows set in an
              orientation are those of the word, bit by bit, in the same places as the generic `set` puts them (the index of a row in an orientation is the mixed radix of its tiles; the function is that index with the constant parts moved out of the loop); it is held to the generic function by the harness on every orientation of every relation it is written for.
  (15) "ends"  (part C) TAKES a walk made by the pass shift (the relation W, its key tiles, the free tile X, the base clauses, the one recursive clause with its stop literals, one literal that takes a member to the next ones) and every clause of the description that reads W. CONDITION, on each reader clause (all of them, in every stratum, as written
              once the other passes have run): W is read by a positive literal, once, with a variable at every place and the places of the free tile X variables of the clause; W is not a relation a question reads; and the clause ENTAILS a stop of the walk at the member it reads, by a homomorphism: for one stop literal Z of the recursive clause (the same for every reader), a negated literal `not Z(k, x)` over the key variables k and the free tile's variables x, there is a clause
              `Z(h) <= B` of Z and a substitution s of its variables with s(h) the arguments the reader gives Z at W's places (key places and free tile) and every literal of s(B) (positive, negated, distinct, by the sign) among the literals of the reader other than W's. WHY THAT HOLDS. (i) A literal of the reader's body holds at every row the reader accepts, so every row it accepts holds s(B), and by the clause of Z, Z(k, x) for W's row (k, x).
              (ii) The walk reaches a member x of key k only from members of its own key; it goes on from x only if its stop literals pass at x (the recursive clause is `W(k, y) <= W(k, x), not Z(k, x), onward(x, y)`); so the rows of W at which Z(k, x) holds are the first members at which the walk stops, the ENDS; the rest of W's rows are members it passed. (iii) By (i) the rows a reader accepts are rows of the ends; dropping from W's relation the rows that
              are not ends removes no row any reader accepts, and the walk itself does not read W's relation (it keeps its own words), so no row of any relation changes. RETURNS: a walk made by whole-word fill and only the word of the ends kept (`W & !live`, where live is the and of the stop literals' words): for a table with one displacement and one mask for each key the fill is the doubling (generate and propagate) form, a constant number of rounds (the
              base two logarithm of the size of the free tile, rounded up) of one shift and one and each; for another table the closure by rounds as the pass shift writes it, the ends kept. `Dw.full` (set by `all_relations` and `ask_many`, clear in every question) makes the function keep the whole word, so that the harness compares W in full; `ENDS: &[&str]` lists the relations made as ends in the questions.
              WHY THE DOUBLING FORM GIVES THE SAME WORD. Let S be the start word (the rows of the base clauses), P the word of members that pass (live and with a next one), shift(w, n) the bits of w displaced by n displacements. The walk's closure R is the least word with S in R and shift(R and P, 1) in R. Writing Q_0 = shift(P, 1) (the destinations whose predecessor passes), Q_(i+1) = Q_i and shift(Q_i, 2^i), and
              G_0 = S, G_(i+1) = G_i or (Q_i and shift(G_i, 2^i)): by induction G_i holds exactly the members reached from S by at most 2^i - 1 hops (a member t is reached in n hops when its n predecessors on the chain are in P, and Q_i holds t exactly when the 2^i predecessors on the chain pass), so G_r with 2^r >= the size of the free tile holds every member any chain of the table can attain. A refusal names the
              reader clause (`Plan.ends_refused`): a negated reader, a reader read by a question, a reader that does not entail a stop, a walk not made by the pass shift.
  (16) "lines" (part D) TAKES a walk made by the pass ends (or shift) and the relation its stop literal reads. THE ARGUMENT, in the one shape it covers. Let live_K be the word of members at which the stop literals of the walk pass for the key K, and live_0 the same for another key 0 (the base); let C be the word of members where they differ, live_K xor live_0; let S_K and S_0 be the start words of the two keys; and R_0 the closure of S_0 under live_0. If S_K = S_0 and
              R_0 and C = 0 then the closure of S_K under live_K is R_0. PROOF: by induction on the rounds the two frontiers are equal; they are equal at the first round (S_K = S_0); if the frontiers are equal and inside R_0, the members of the frontier pass the stop literals in one key exactly when they pass in the other (they are not in C), so the next frontiers are equal and inside R_0 (the next members of a member of R_0 are in R_0 by the closure of R_0); so every round is the same and the closures
              are equal. So a line of a walk that passes through no member of C is the base line, word for word, and a line through a member of C is walked again by the walk as it stands. WHAT IS NOT YET AN ARGUMENT. That the key K's live word and start word are "the base's with a bounded change" is a fact of the clauses of the relations they are made from, and a rule that reads it from the clauses (which literal is the state's relation, which negated literals are the key's removals,
              which clauses add) with a proof that the words differ only inside C, and the place where the base's closures are kept for the question and the test that the key's start word is the base's, are not written here; they are the pieces of work this plan builds only if the profile once the pass ends has run still shows the walks (the number is printed) and the rule is proved here first; otherwise the pass is not built and the module text of the commit says which piece is missing.
  A pass that makes a piece slower or gives a different row is a fault of the pass: it is mended, never set aside. The harness of 79.38 holds each of them to the plain module (the tiles and the displacements are an inside of the module: rows are compared as text).
  AS BUILT 79.42, where the body differs from the text above (said here because the text was committed first); PASSES = ("dense", "demand", "copy", "delta", "bits", "order", "keep", "scan", "cut", "shift", "base", "start"):
    (13) is written as the text says, with these choices: the orders are those of `JoinRule` with each positive literal over a relation held as bits first (`start_orders`), written to the same function text as the reader's order (`emit_full_clause`); the factors of the estimate are noted by the writer
         as it writes each literal (`BitWriter.prof`), a literal read by a loop over words counting the words its relation holds in the orientation read, 2 times that when it binds a variable; a clause of a static relation is made once and has one order; an order is also not admitted when it reads a relation
         that is not held as bits by places that no clause asks it by (its indexes are fixed: a ValueError of the writer, named in `Plan.start_refused`); the decision is made in the dry run of the last round of `find_bits` and kept (`Plan.start_choice`) so the final writing is the same text.
         An order is admitted when it asks no orientation that the clauses in the reader's order ask altogether (`Plan.start_base`, found by a first writing of every clause in the reader's order, `Plan.start_off`), and not only none that the clauses written before it ask:
         the first writing of this part decided by the clauses written before, which depend on the order the strata are written in, which depends on the names, and the description with every name replaced gave other refusals (found by tests/test_plan79_42.py; mended).
    (15) is written as the text says, with these choices: the readers are the clauses of every stratum as written (`Plan.rules`, once every other pass has run), and the entailment is a homomorphism of the body of a clause of the stop relation into the literals of the reader found by backtracking
         (`entail_body`: positive and negated literals by relation and sign, a distinct test only against a distinct test); the stop literal is a negated literal of the recursive clause whose places are key variables, free tile variables or constants (a walk whose stop literals are all positive, as a walk over
         places that must hold something, is refused: `Plan.ends_refused`); the doubling form is written when every key of the displacement table has at most one displacement, the closure by rounds otherwise (the ends kept in both); and the base clauses of the walk (the start words) are made into words
         of the working memory (`Dw.sb_<relation>`, the last tile's orientation only) and not into the relation, which is then given the ends by one word for each key (the whole word when `Dw.full`), so that the relation of an ends walk does not hold the rows of the first places that are no end (the first writing kept
         them, and they were most of the rows the walk made).
    (14) is written as the text says for the one function it covers, `or_word` of a relation held as bits (the walks and every clause that ors a word into its head): with the pass, the index of each other orientation is made once for the word (`bf`, with the bit at nought, and the tile that is free there, `af`)
         and each new bit moves it by a constant (`p * stride`); a relation with one orientation, or no new bit, is as before. The rows set one by one by `set_t` (a clause that makes a row, not a word) are not changed: the pass does not cover them.
    (16) NOT BUILT. The lemma and its proof above are complete for the test (a line of a walk whose closure in the base meets no place that differs between the key and the base is the base closure; tests/test_plan79_42.py runs the lemma on random tables, with each condition left out in turn to show it is needed). What is not complete, and so what is missing, is the
         piece of work that makes the lemma a rule of the translator: (i) a rule that reads from the clauses that the stop relation and the start relation of a walk are, for a key, the relation of the state with the key's removals taken out and the key's additions put in (so that live_0 and S_0 can be named and C bounded by the words of the key's removals and additions): in the first game the clauses of the stop relation are one
         clause that adds (a relation of the key) and one that copies the state's relation less two relations of the key, under a guard literal of the key, and to name the base the guard must be shown true for every key the walk is made for (it is the literal the start relation's clause holds too); the pass base unfolds a view into its readers and does not give this form, and no pass finds it; (ii) the closures of the base,
         kept for the question in the working memory and their invalidation by the next state, and the test of the key's start word against the base's, which need (i) for the start relation. The profile once the pass ends has run shows the piece of the walk with the lines of an alternative as a small share of the pieces' time, and a walk that is keyed by no alternative (it is made for every place) is outside the shape of the
         lemma; so the piece of work above would buy at most that share, and it is left for a plan that first writes the form (i) as a pass of its own with its proof.

PLAN 79.43 PARTS A, B, C: THE LAST PASSES, A WALK MADE ONLY FOR THE KEYS ASKED, A LITERAL READ FROM A WORD WITH TILES DROPPED, THE ORIENTATION AN ORDER NEEDS WHERE IT PAYS (SYSTEM_SPEC 11, "The crux of layer 2 is a translator ... made fast piece by piece under tests"; piece: rules; ledger row 45).
This text is the interface of the passes below, written and committed before any body. PASSES gains "asked", "project" and "worth", each added to PASSES in the commit of its body; `translate(text, module, passes)` is as above; with no pass the document is still the plain one byte for byte but for the
translator hash; the passes add no public item to a module but `ASKED: &[&str]` (below), so the harness of 79.38 holds each of them to the plain module piece by piece as before.
WHY (the profile of 79.42, every pass): the largest piece is a walk made for every key of its sorts, though its readers ask it for the places a unit stands on; the third and the fifth hold a relation row by row where one word of places, with the tiles that are not asked
for dropped, would do; the orders the pass start refuses for want of an orientation are never tried. Each cause is machinery, none is a rule of a game.

  (17) "asked"   TAKES a relation W that reads the state and is the one relation of its recursive set (a walk), the clauses with W as head (the BASE clauses, which do not read W, and the RECURSIVE clauses), and every clause outside W's set that reads W (the READERS), as written once the other passes
               have run. THE KEY of W is its places but the last tile's. CONDITION on W: (a) every recursive clause reads W once and keeps the key: the places of the key of its head and of that literal are the same distinct variables; (b) W is not read by a question, is not a state, an input, a `next` or an `init` relation, and no reader reads it negated;
               (c) every base clause has distinct variables at the places of the key of its head. CONDITION on each reader R: W's literal is read once, with distinct variables at the places of the key, and R has positive literals before W's, in the order R is written, whose relations stand in strata before W's, are not W,
               and which together hold every one of those variables (the PREFIX of R: the positive literals of R before W's, in R's written order, that stand in strata before W's); a prefix that the base clause already holds (every literal of it maps into the base clause's body, by a homomorphism that is the identity on the key) adds nothing and the pass refuses W, naming R.
               RETURNS: W's base clauses are replaced by one clause for each base clause B and each reader R: B with the prefix of R added, the key variables of R's W-literal taken as the variables of B's head at the same places, every other variable of the prefix named anew (the union of the clauses stands for the union over the readers).
               The recursive clauses are not changed. A module with this pass keeps the base clauses as they were as well and runs them, in place of the new ones, when `Dw.full` is set (as `all_relations` and `ask_many` set it), so that the harness compares W in full; `ASKED: &[&str]` lists the relations made so in the questions.
               SOUNDNESS. Let K be the set of keys that some reader's prefix gives (some valuation of R's prefix literals, which are made before W, has the key as the values of its key variables): the union over readers of the keys for which the prefix holds. (i) A row that a reader accepts is accepted under a valuation of all its literals, so under one of its prefix, so
               its W-literal's key is in K: the reader reads from W only rows whose key is in K. (ii) The rows of W are those derived from the base clauses by the recursive clauses; a recursive clause keeps the key, so every row derived from a base row has that base row's key, and the rows of W of a key k are exactly those derived from the base rows of k (by induction on the derivation).
               (iii) The new base clauses give the base rows of B whose key is in K (B's head's key places are the variables the prefix is joined on, and the prefix holds for them), so the rows of W then made are the rows of W whose key is in K, and nothing else (by (ii)). By (i) no reader sees the difference, so no row of any relation but W changes. W itself differs, which is why `Dw.full` runs the base clauses
               as they were. A refusal names the reader or the clause: a reader that asks W with a key place that is no variable or is repeated, one that has no prefix literal for a key variable ("asks it unbound"), one read negated, a recursive clause that changes the key, a relation read by a question.
  (18) "project" TAKES a relation P that reads the state, is held as bits, is not answered on demand, and whose clauses are all the one clause `P(h_1 .. h_m) <= R(a_1 .. a_n)` with R held as bits, the h distinct variables, the a distinct variables, every h one of the a (P is then the PROJECTION of R at the places the h stand at, the tiles
               of the other places dropped: P(x) holds if and only if R(y) holds for some y with the h-places equal to x), and the clauses that read R. CONDITION on a reader C of R: R's literal is positive, its places are variables, each variable at a place that P does not keep stands once in the whole clause (not in another literal, not in the head, not twice in the literal); P's stratum is before the head's stratum of C (P is made before C
               runs; a head of the set of R or of P is therefore never rewritten). RETURNS: in C the literal over R is replaced by the literal over P with the variables at the places P keeps, in the order of P's head; the clause is then written as before (a literal of P with its free tile unbound is read as one word, and a literal later in the clause over the same variables is folded into that word, as the pass bits does).
               SOUNDNESS. C is `H <= L_1 .. R(y) .. L_k` and the variables at the places not kept occur only in R(y); so C holds under a valuation v exactly when P's kept places of y under v are a row of P, whatever the dropped places hold (those places have no other use), and P is the set of those rows (it has no other clause). The change is of the literal alone; the stratification holds because P is made before C. Nothing
               of the description changes. THE PROJECTION AS A BY-PRODUCT (the plan's form in which P is set when a row of R is set and never made as rows) is NOT built here: P is made by its own clause, a loop over the words R holds, as before; the module text of the commit that builds it, if one does, says so.
  (19) "worth" TAKES the orders the pass start refuses (`Plan.start_refused`: an order whose clause asks an orientation that no clause in the reader's orders asks). THE COST of an orientation of a relation X is paid by each row set in X: one bit set more, and X's words of that orientation cleared at the end of the question. THE SAVING of an order is the loops over words that
               the order does not make. RETURNS: an order that asks new orientations is admitted when the estimate of the chooser (13) for it, over the sizes the translator knows (the number of words the orientation can hold, the sorts' sizes), is under that of the reader's order by more than the cost of the orientations it asks, in units of one bit set; otherwise it is still refused, the refusal naming the orientation, its cost and its saving. SOUNDNESS: an order is a different order of the same
               conjunction (13); an orientation is a copy of the same bits (the harness holds every orientation to the generic `set`); so admitting an order or an orientation changes no row, only the time.
  A pass that makes a piece slower or gives a different row is a fault of the pass: it is mended, never set aside. The harness of 79.38 holds each of them to the plain module (the tiles are an inside of the module: rows are compared as text).
  AS BUILT 79.43, where the body differs from the text above (said here because the text was committed first); PASSES = ("dense", "demand", "copy", "delta", "bits", "order", "keep", "scan", "cut", "shift", "base", "start", "ends", "orient", "asked"):
    (17) is written as the text says, with these choices: the pass is decided in `Plan.__init__` (`find_asked`) once the relations that may be held as bits are known, so that the clauses a walk is made by (and the orientations they ask) are those the passes that follow it see; a walk that the pass bits then drops is made as written again (`Plan.undo_asked`).
         The prefix of a reader is the positive literals before W's in the order the reader is joined in (`JoinRule.order`, the reader's own choice or the pass order's), over relations that are made before W in every order of the strata: relations that do not read the state, the state's own, and those W reads, closed (the order of the strata of two relations that do not depend on each other is by name, and a decision may not rest on it; found by the test of the renamed description); it need bind only some of the key variables (the others are looped over as before: a guard on part of the key is a guard), and
         a reader that binds none of them refuses the walk, naming the reader. The text's rule that a prefix the base clause already holds adds nothing is applied by two tests, either of which keeps the base clause as it was for that reader: the prefix is a homomorphic image in the base clause's body with the key variables fixed (the text's test), or it binds no key variable through
         a relation that reads the state that the base clause does not already bind through one (the keys are then those the state already gives: the base clause makes its keys from the state, as the base clause of a walk over a table of places made for every key does not). The second is the rule that tells a walk made for every key of its sorts from one made for the keys of the moves of a position;
         it is a choice of the translator, not a condition of the argument (the argument holds for any prefix; the rule only chooses where the added literals pay). A walk where every base clause is kept so is refused, naming the reader. `Dw.full` is the one flag of the passes ends and asked (set by `all_relations` and `ask_many`, clear in every question).
    (18) is written as the text says (PASSES gains "project"), with these choices: a relation is a projection when it is not static, is held as bits, has the one clause, a positive literal over a relation of the state that is held as bits with distinct variables, and its head keeps some of them and not all (`Plan.find_project`; a projection that keeps every place is a copy and is not used);
         a reader's literal is rewritten when every place whose term is no variable that occurs only there is among the places a projection keeps, and the projection that keeps the fewest places of those that do is chosen; the places that are kept but are such variables stay variables of the literal (they are not used elsewhere, so they stand for "some"). THE ORDER OF THE STRATA: the projection must be made before the reader in every order of
         the strata, which only a dependency decides (the order of strata of relations that do not depend on each other is by name): it is rewritten only where the head of the reader already reads the projection, closed (the relations the head reads), and otherwise the clause is left as written and named in `Plan.project_refused`. The projection is made by its own clause as before; THE BY-PRODUCT FORM (set when a row of R is set) is NOT built:
         it would put the words of P inside the struct of R and send every reader of P through R, a change of every place that names a relation's words, and the pass here takes the reads that cost (the loops over R's words that the dropped tiles forced) and not the making of P, which is one loop over the words R holds. A clause of a pass asked keeps its place in the sets of the clauses run when everything is asked for, or only what is. The entailment of the pass ends (15) is shown on the clauses of the stop relation before this pass has read a projection in place of a relation (`Plan.rules_unread`): a literal of a relation entails the one of its projection, which the homomorphism does not see (found by the test of the renamed description: a walk was no longer made as its ends).
    (13) MENDED by the pass project: the estimate of the order of a clause counted the bits of a word read by a key as 2 whatever the word held and a word that is or-ed into the head whole as a loop over its bits; the first made the chooser take an order that looped over the one word of a relation with no key (the places held) with a loop inside it over the candidates, the second made it count a word
         that costs one operation as two. A literal read as a word that is or-ed into the head counts 1, and a literal that reads the one word of a relation with no key counts the bits that word holds (`count_ones` at the time of the question); every other literal counts as before.
    (19) is written as the text says in the one case where the cost of an orientation is nothing by the sizes: the orientation of a relation that does not read the state is set once, with the relation, in `Game::new`, and a question pays nothing for it, so an order that asks only such new orientations is admitted by the chooser of (13) (`Plan.worth_taken`).
         An order that asks the new orientation of a relation that reads the state is still refused, the refusal naming the orientation and "a relation that reads the state: each row set would set it too" (`Plan.start_refused`). A version that admitted those too was written and timed as a trial (every such order admitted, the orientation set with every row): the module with it was not faster than the one without it, so the estimate by sizes of the plan's text
         (the saving against the cost, in bit sets) was not built; the orientation made only when the chooser picks the order that reads it (kept for the question, made from the words of the relation) is the form that would be tried next, and is not built here. `Plan.worth_taken` lists, for each order the pass admitted, the orientations it made.

PLAN 79.45 PARTS A, B: A LEMMA AS A REWRITE, AND A RELATION NOT MADE WHERE EVERY READER IS SHUT OFF (SYSTEM_SPEC 11, "One lemma machinery, used twice" and "The crux of layer 2 is a translator ..."; piece: rules; ledger rows 49, 45).
This text is the interface, written and committed before any body. `translate` grows one argument and nothing else changes for the callers that do not give it:

  translate(text, module, passes=(), lemmas=()) -> str
  LEMMAS ARE DATA. `lemmas` is a list of documents, each {"name": str, "form": "SAME" | "IMPLIES", "left": {"relation": str, "places": int}, "right": {"relation": str, "places": int}, "condition": [{"relation": str, "args": [str], "neg": bool}],
  "standing": {"is": "CERTIFIED", "key": str} | {"is": "PROVED", "theorem": str}} (the form of rules/lemmas/<game>/<name>.json that rule_lemma.py reads; the translator imports no reader of it and reads no folder). `text` is the composed description, the base with the lemma's delta
  (`rule_lemma.composed_text`), so that the right relation is a relation of the text. WHO PASSES THEM: scripts/gdl_to_rust_make.py (which may name a game) loads the folder of the game it makes, computes each standing and passes only the lemmas whose computed standing is CERTIFIED or PROVED,
  printing the others as refused; the translator refuses again, naming the lemma, a form that is not SAME (a lemma of inclusion is no rewrite), a standing that is neither of the two, a relation that is not in the text or has other places than the document says, and the cases below.
  WHAT IT RETURNS: the module of the text read with the rewrite, `pub mod <module>`, with the same public items as without, and these: `LEMMAS: &[Lm]` (for each lemma: name, form, the left relation, the right relation, the relation that replaces the left one, the flag, the evidence it stands on, and its condition as data so that a test
  may read it from the rows of the plain module), and, with the pass gated, `GATED: &[&str]`. The head lines of the document carry, for each lemma, `// lemma: <name> <form> <left> -> <replacement>, flag <flag>, <CERTIFIED key | PROVED theorem>`; tests/test_plan79_45.py fails a kept document whose lemma line differs from what the lemma's certificate now says.
  THE REWRITE, in the text before anything else reads it (with no lemma the text is unchanged, byte for byte). Let L be the left relation, M the right one, c_1 .. c_n the literals of the condition, x_1 .. x_k the places of L. The text gets, with new names that no relation of the text has:
      a FLAG of no places                  F  <=  c_1 .. c_n                      (a literal over a state fluent is written as the fluent; a `differ` literal is a `distinct`)
      a replacement of L                   L' (x_1 .. x_k)  <=  F, M (x_1 .. x_k)
                                           L' (x_1 .. x_k)  <=  (not F), L (x_1 .. x_k)
      and in every clause whose head is not L, every literal over L, positive or negated, is a literal over L' with the same terms.
  ARGUMENT, in two sentences. At a state where F holds the second clause of L' is silent, so L' has the rows of M, and by the lemma M has the rows of L there; at a state where F does not hold the first clause is silent and L' has the rows of L; so L' has the rows of L at every state, and every clause that read L read the same rows, so no
  relation but the new ones changes. WHAT THE ARGUMENT NEEDS, and the translator tests it from the clauses, refusing with the lemma and the reason named: (i) L, M and the flag read the state only (no `does`, no `record`, none of the fluents of a turn: the lemma was certified at states), because F is evaluated in the same question as its readers; (ii) L is not read
  by a question (`legal`, `terminal`, `goal`, a `next`, an `init`, a fluent) and is not in a recursive set, and neither M nor F reads L (else L' would read itself and the rows of L' would not be those of L); (iii) the text with the new clauses is stratified, safe and well sorted by the reader (the flag is below L' and so is L, M, the condition's relations). The lemma is as sure as its standing: the
  clauses give L' the rows of L only where the lemma holds; the pieces test (translated_pieces.rs, `--cfg singles`) holds L' to the plain L at every state it visits, and the lemma's certificate by translated code (translated_lemma.rs) holds L to M where F holds on many more states than the certificate of 79.44 made.
  THE PASSES AND THE REWRITE. The rewrite comes before the reader sees the text, so every pass reads the rewritten clauses as it would read any text; a flag of no places that a pass could answer on demand is made in full (it is read once by two clauses of L' and by whatever the lemma's other readers are; it costs one test).

  (20) "gated" (part B; general, it needs no lemma to be stated) TAKES the clauses of every stratum as written once the other passes have run. A FLAG is a relation of no places that reads the state, is made in full, is no question and no input and is not in a recursive set. A READER of a stratum S is a clause outside S with a literal over a relation of S. A clause is
              SHUT by a flag literal when it holds a positive literal over a flag or a negated one, and that literal is false in the state. CONDITION: S is gated when it is made as a piece (no relation of it answered on demand), holds no question relation, no fluent, no input and no flag, has at least one reader, and every reader is shut by a flag literal that it holds
              or has its head in a stratum that is gated (the largest set of strata for which this holds, found by removing the strata that fail until none does); the flags it relies on are made before S (the piece order is the stratum order with the cone of each such flag moved before S when it came once, which no clause forbids because the flag does not read S: the translator
              refuses, naming the stratum, when a flag reads a gated stratum or the order has a cycle). RETURNS: the piece of S is not run in a question where, for every reader, the flag literal that shuts it is false, or the head's piece is itself not run (`off`, a function of the flags' rows alone); `all_relations` and `ask_many` run every piece (a flag `Dy::whole`),
              so that the harness compares every relation in full. SOUNDNESS. A piece that is not run leaves its relations void; every clause that reads them is shut, so it makes no row whatever the relation holds (a clause with a false literal makes no row), and a clause whose head is not made makes none; so every relation that is made has the rows it had, and the relations of S have none in
              that question, where no clause that is run reads them. A stratum for which the condition cannot be shown is not gated, and `Plan.gated_refused` names the stratum and the reader that is not shut.

PLAN 79.47 PART A: THE WALK READ MORE THAN ONCE IN ONE CLAUSE (SYSTEM_SPEC 11, "The crux of layer 2 is a translator ..."; piece: rules; ledger rows 44, 45).
This text is the interface of the change, written and committed before its body. `translate`, PASSES and the public items of a module are as above and nothing is added to them; only the pass asked (17) takes more than it did, and `Plan.asked_refused` names the same things.
WHY (79.45 READ): the lemma of 79.45 reads the line walk twice in one clause; the pass asked refused a walk read twice, so with the lemma the walk was made for every key again and the gain of the lemma went into it.

  (17, widened) "asked"  TAKES what (17) takes. THE READS OF A READER. A reader clause R may now hold more than one positive literal over W (a negated one is still refused, naming the clause). Each read is taken by itself: its key is the places of W's key, distinct variables, as before; the GUARD of a read is the
               positive literals of R, in R's written order, that stand BEFORE that read, over relations made before W (the prefix of (17): relations that do not read the state, the state's own, the inputs, and those W reads, closed), and the guard that a relation made LATER THAN W gives (below); a read that stands later than
               another read of W in R takes the earlier read's literal out of its guard (W's own rows are not made yet when W's base clauses run), the earlier read binds nothing a guard may use. CONDITION on each read, as before: its guard binds at least one variable of its key ("asks it unbound": a read whose key is bound only by an earlier read of W has no guard and the pass refuses W, naming the reader).
               THE GUARD THAT A RELATION MADE LATER THAN W GIVES. Let a positive literal Q(t_1 .. t_n) of R stand before the read, Q not W, not made before W (it reads W, directly or through others), not in a recursive set, and let a variable v of the read's key stand at the place p of the literal as t_p. Every row of Q is made by a clause C of Q whose head has at p a variable h
               (a clause with a constant or a compound at p gives no guard); a row of Q whose place p holds a was made under a valuation of C's body with h = a, so the positive literals of C's body that are over relations made before W (and, through the same rule, those that Q' gives, to a depth of 3), and that are joined to h by variables, all hold at a valuation with h = a. The guard of Q is the union over the clauses C of Q of those literals (the head variable h replaced by t_p, every other variable of
               C named anew): one alternative for each clause. A clause of Q that gives no literal makes Q give no guard (the union would be every value).
  RETURNS: as (17): W's base clauses replaced by one clause for each base clause B, for each read of each reader, and for each alternative of the guards of that read: B with the guard added, the key variables of the read taken as B's head variables at the same places, every other variable named anew. The recursive clauses are not changed; `Dw.full` runs the base clauses as written, as before.
  SOUNDNESS, the argument of (17) with one more addition. Let K_j be the keys for which some valuation satisfies the guards of the j-th read of R, and K the union over all reads of all readers. (i) A row that R accepts is accepted under a valuation of all its literals; under it the guard of each read holds (the literals of the guard are among R's, or follow from R's literals by the clauses of the Q that R reads: a row of Q at that valuation was
  made by a clause C whose body held, so C's guard literals hold at the head value), so the key of each read of W in R is in K_j. R reads from W only rows whose key is in K. (ii) and (iii) as in (17): the rows of W of a key are those derived from the base rows of that key (the recursive clauses keep the key); the new base clauses give the base rows of every key in K and no other; so the rows of W then made are the rows of W whose key is in K,
  and no reader sees the difference. A guard may be wider than the keys a read needs (it is a part of what must hold, never more); it is never narrower, so no key a read asks for is missing. A relation Q whose clause gives no guard, a literal at a place that holds no variable of the key, a Q in a recursive set: the guard of that literal is left out (the read may still be guarded by the other literals before it); a read with no guard refuses W.
  A pass that makes a piece slower or gives a different row is a fault of the pass: it is mended, never set aside. The harness of 79.38 holds each of them to the plain module (the tiles are an inside of the module: rows are compared as text).
  AS BUILT 79.47 A, where the body differs from the text above (said here because the text was committed first):
    (a) THE FIRST BODY of the text (an earlier read of W standing out of its guard) was built and clocked: with the lemma of 79.45 the walk was made for the keys of the second read of a clause as every place the key's guard admitted (the squares that hold a unit, in every way), 625,285 rows a 1,920
        positions against 91,215 for the module with no lemma, and the module with the lemma was about 0.6 times as fast as the module without (OBS 79.47 A, first body). The guard was valid and far too wide: it dropped what the earlier read says of the key (the second read's key lies on a line that the first read walked).
    (b) THE STATIC OVER-APPROXIMATION (`over_approximations`, a rewrite of the text before the reader sees it, as the lemma's rewrite is; made only for a walk that a clause outside it reads more than once). For a walk W the relation S, named `<W>_over`, has W's clauses with every literal over a relation that reads the state taken out
        and W's own literal read over S. ARGUMENT: every row of W is made by clauses whose bodies hold, and the clauses of S are those clauses with some literals left out and W's literal read over S; by induction on the rounds of W's closure every row of W is a row of S (a conjunction with fewer literals holds wherever the whole holds, and a positive recursion is monotone), so S has every row of W and the text with S added gives
        every other relation the rows it gave. S reads no state: it is made once, with the static relations. A walk with a negated literal of itself, an `or` in a clause, no declared sorts, or a clause that is unsafe once the literals are out, has none (`over_approximations` then leaves the text as it is) and its earlier reads give no guard.
        THE GUARD OF A READ then holds, for each earlier read of W in the reader (in its written order) a literal over S at the same terms, besides the literals of (17): the key of the second read lies on the rows of the first (W at the first's key), these are rows of S, so the guard holds at the second's key. It is still never narrower than the keys the read asks for (S has every row of W), and
        it is a part of what must hold, as before. The soundness argument of (17, widened) stands with this addition: the literal over S is among the literals that hold at every valuation under which the reader accepts a row.
    (c) A READER THAT ASKS THE WALK UNBOUND IN ITS ORDER (module text above, "asks it unbound") is joined again with the walk taken last (`JoinRule` with the walk in `avoid`), once, and the pass goes on with that order, where a relation that reads the walk binds the key and gives a guard (the guard of a relation made later than the walk): a conjunction does not depend on the order of its literals. The reader is
        written in that order when the pass is taken (`Plan.asked_reorder`); where the pass refuses, nothing is changed.
    The module with `<W>_over` has one relation more that the plain module has not, a static one; the harness of the lemma (translated_lemma.rs, `View`) keeps it back as it keeps back the relations the rewrite and the delta add.

PLAN 79.47 PART B: A MODULE BEGUN AT OTHER INITIAL FACTS (SYSTEM_SPEC 11, "The crux of layer 2 is a translator ..."; piece: rules; ledger rows 44, 45). This text is the interface of the one item the module gains, written and committed before its body.
WHY. A description may read its initial facts in its rules (a relation made from `(init ...)` that other relations read: a variant with a start by number reads where its leader and its corner units begin). Such a relation reads no state, so a module makes it once, in `Game::new`, from the clauses' own initial facts, and answers
for that start alone; the evaluator is begun at another start by composing a delta that gives the facts as the initial ones (`rules::restart_delta`), and the kernel of that game is compared at such starts. The module has no such way, so a game whose rules read its start was compared from its own start only (found by the first run of the five more games).
  Game::restart(&mut self, facts: &[(String, Vec<String>)]) -> Result<(), String>        (every module has it; a game whose clauses do not read an `init` relation changes nothing but its initial facts)
  TAKES the facts of a state as `state_of_facts` takes them (a relation name and its arguments as text). DOES: the rows of each relation `init.<fluent>` become the rows of that fluent among the facts (a fluent with no fact has none, as a delta that retracts the initial clauses and gives the facts says); every relation that does not read the state and
  that reads an `init` relation, directly or through others (found by the dependency graph, as the static relations are told from the others), is made again from them, in the order of its strata; `roles` is read again and the reading kept by the last question is dropped. RETURNS Err naming the fact where `state_of_facts` does.
  ARGUMENT. A relation that does not read the state is a function of the clauses' static rows and of the `init` rows alone (that is what makes it static); the rows `init.<fluent>` of a description restarted at facts are exactly those facts (the delta retracts every initial clause and gives them), so making the relations that read them again, from them, in stratum order, gives the rows the evaluator has for the restarted description;
  the relations that read no `init` relation have the same rows whatever the facts and are not made again. Once `restart` has run, `initial()` is the state of those facts. WHO CALLS IT: the tests of cs_game (a position of a fixture or of a game's list of starts is begun with it, as `evaluator_at` begins the evaluator); `impl_tr!` forwards it as `restart_t`.

PLAN 79.52 PART B: ANSWER ONLY WHAT WAS ASKED, AND NO ALLOCATION ON THE PATH OF A QUESTION (SYSTEM_SPEC 11, last block "The lower layers are made as fast as the outside library first; the facts stay ..."; piece: rules; ledger rows 55, 45).
This text is the interface of 2 more passes, written before any body. PASSES = (... "gated", "cones", "words"); both bring "keep" with them (they change the questions of a module that keeps its working memory). With no pass the document is still the plain one byte for byte but for the translator hash.
  (21) "cones" TAKES the dependency graph (as written, with the passes that unfold) and the relations the questions read in full (the fixed words of the language: legal, terminal, goal). RETURNS: the pieces of the cone of each subset that holds at least one of those relations, in the order
              of the strata (gated pieces ordered as before), as `CONES[mask]` (bit 0 legal, bit 1 terminal, bit 2 goal); a question runs the cone of what it needs and of what the state's cached reading already holds, and keeps the reading with the mask it holds.
              WHAT EACH QUESTION NEEDS: mover and moves: legal and terminal (a state that ends has no mover); terminal: terminal; outcome: terminal and goal; `legal_moves_into` (new): legal alone, the relation's rows for the first role that has a turn that is not
              the turn that passes, with no test of the end; next: legal alone (its state is one the caller found not at an end, by `moves`). The decision is the dependency graph's and no name is read.
  (22) "words" TAKES each fluent's signature and the declared sorts. CONDITION: every place of the fluent is of a declared sort of constants and the product of the sizes is at most WORDS_MAX (2^13) bits. RETURNS: `State` holds that fluent as `Wd<W>`, one bit a fact (the mixed
              radix of the ordinals of its places in their sorts, the first place the most significant), W words; the other fluents stay lists of rows. A state is then a value that is copied, compared and hashed by its words and made without allocation. HANDED OUT AS FACTS:
              `facts_of`, `state_of_facts`, `initial`, `restart` read and make the facts through the same tables (a fact outside its sorts is an error of `state_of_facts`, never a silent drop). `next` is made from the old words by clearing the rows a copy clause removes and setting the
              rows the next relations give; no list is made. The hash relations that a question keeps (`Rel`) keep the lists of their indexes when cleared. New items of a module with the pass: `Game::moves_into(&mut self, &State, &mut Vec<u32>)` (the permitted turns of the mover, with the test of the end, into a buffer the caller owns),
              `Game::legal_moves_into` (the same without the test of the end). The tests call them through `Tr::moves_into_t` and `Tr::legal_into_t`, which every module has (the plain module answers them by `moves`).

PLAN 79.53 PART A: TABLES FOR WALKS (SYSTEM_SPEC 11, last block, (vi) and (vii); piece: rules; ledger rows 55, 45). Interface written before the body. PASSES = (... "cones", "words", "tables"); it brings "bits" and "shift" (it is a form of a walk made by the pass shift).
  (23) "tables" TAKES a recursive set made by the pass shift (one relation W held as bits, one recursive clause of the form of the pass shift: a literal of W, literals that stop the walk and test the member the walk stands on, and one literal of a relation that does not read the state that takes a member to the next ones), its onward relation read at translation.
              THE FORM IT FINDS, by the clauses and the rows of the onward relation alone: for each combination of the onward relation's other places (a line), the onward link is ONE displacement of the member's number with a mask of the members that have a next one (no branching), and following it from any member ends (no cycle), inside the free tile, and the chains
              fit at most TABLE_WORDS words. RETURNS: a table CH[line][member] of the members reached from a member by following the link again and again, the member included, as a word (the table is read from the rows of the onward relation, which the pass shift has tested against every row; it is no lemma and has no standing of its own);
              the closure of a walk then is, for each member the base clauses made: the chain word, and of it the members up to and including the first member that the stop literals shut (the lowest bit of the shut members on the chain when the link raises the number, the highest when it lowers it), taken by one count of trailing or leading zeros; the readers of the walk read the same words as before.
              WHAT IT REFUSES, naming why (`Plan.tables_refused`): a line with two displacements, a chain that returns to a member, a chain that leaves the tile, a tile of more than a word, a table of more than TABLE_WORDS words; a walk so refused keeps the shifts of the pass shift.
  HELD TO THE PLAIN MODULE: by the harness of 79.38 on every relation at every state (the walk's relation is compared as text), and by tests/test_plan79_53.py on planted cases.
PLAN 79.53 PART B: THE TOP-DOWN WAY BESIDE THE BOTTOM-UP ONE (SYSTEM_SPEC 11, last block, (vii)). PASSES = (... "tables", "topdown"); "topdown" brings "words" and "cones". Interface written before the body.
  (24) "topdown" TAKES the clauses of the description as written, the relations of the cone of the end (terminal) and of the legal rows (legal), the state as the words of pass (22), and the rows of the relations that do not read the state. RETURNS, in the same module and over the same State,
              Game::terminal_td(&State) -> bool           whether the end holds: a depth-first walk of the clauses of the end that stops at the first rule that holds
              Game::has_turn_td(&State) -> bool           whether a turn that is not the turn that passes is legal (the mover exists; the end is not asked): the walk stops at the first legal row
              Game::ask_td(&State, relation) -> Option<Vec<Vec<u32>>>   the rows of a relation of the cone with no place bound, by the same walk (for the tests that hold each relation to the bottom-up rows); a place where a clause gives a compound term holds FREE - 1
              Game::terminal_bu, Game::has_turn, Game::terminal  the bottom-up end, the entry for the question, and the end by the way recorded for it; `WAYS: &[(&str, &str)]` records for each question ("end", "legal-turn") the way that answers it, "bottom-up" or "top-down", given to `translate(..., ways=)`
              by the script that makes the files from the clocks of plan 79.53 (the translator chooses nothing); the default is bottom-up.
              HOW A WALK RUNS. A literal of a relation that does not read the state or of a fact of the state is a loop over the rows of a table read by the places already bound (the tables are Dense relations made at `Game::new` and from the state at each question); a literal of a derived relation is a CALL with the places bound and the rest of the clause as the
              continuation: a closure that is given each row and says whether to stop; a negated literal or a literal read only for existence is a call whose closure stops at the first row. Clauses are joined in the reader's order with the places the caller binds taken as bound (`JoinRule`), a relation of a walk taken once the places it keeps are bound.
              HOW IT TERMINATES. A relation not in a recursive set calls only relations below it. A relation of ONE linear recursive clause (one literal of its own relation) is answered for one call as a WORKLIST: the rows of the clauses without the literal (the seeds, made for the places the walk keeps), then each row found once (a row already found is not made again, so a cycle of the link relation ends) is joined with the recursive clause; the rows found are
              at most TD_CAP: when more are found the question is given to the bottom-up way and the module counts it (`td_over`). A relation in a recursive set of more than one relation, a clause that reads its own relation twice, and a literal with a compound term in the cone are REFUSED (`Plan.td_refused`, naming the relation or the clause) and the module has no top-down way.
              HOW A NEGATED CONDITION IS ANSWERED. By the call that asks whether a row exists with every place bound (the reader's order takes a negation only when its variables are bound, as the evaluator does); the negation of a relation of the cone is the failure of that walk, which has finished (the relation is below the clause in the strata).
              A place that a clause gives as a compound term is not computed: it holds FREE - 1, permitted where the caller's variable is used nowhere else (else refused). The facts, the rows and the questions of the bottom-up way are unchanged; every old entry point keeps working.
  HELD TO THE BOTTOM-UP WAY: rust/engine/game/tests/translated_topdown.rs (the end and the legal turn at every position of the fixture of each of the seven games and at every state of seeded random games; each relation of the cone asked with no place bound) and tests/test_plan79_53.py (planted cases).

PLAN 79.54 PART B: ANSWERS CARRIED FROM THE LAST STATE (SYSTEM_SPEC 11, last block, (ix); piece: rules; ledger rows 55, 45). Interface written before the body. PASSES = (... "tables", "topdown", "carry"); "carry" brings "cones" (and so "keep").
  (25) "carry" TAKES the pieces of the cones of the questions (pass 21), the state as `State` (a value that is compared fluent by fluent), and from the dependency graph of the clauses as written, for each piece, the fluents its cone reads (`PIECE_FL[k]`: a bit for each fluent in the order of `FLUENTS`).
              RETURNS, in the same module and with no change to any entry (`Game::new`, `initial`, `mover`, `moves`, `moves_into`, `legal_moves_into`, `terminal`, `outcome`, `next`, `facts_of`, `state_of_facts`, `restart`, `all_relations`, `ask_many` keep their signatures and their answers): the module keeps, besides its pooled evaluation, the
              evaluation of the last state it READ for the questions of the cones (`Game::cd`, a `Dy` not cleared between questions) and, for each piece, whether the rows in it are those of that state (`Game::cvalid`).
              B1, BY PIECE. At a new state the fluents that differ from the kept state (`State::fluent_diff`) are read again into the kept evaluation (the rows of each changed fluent cleared and loaded), each valid piece whose cone reads a changed fluent is cleared (the relations it makes: `clear_piece`) and marked not valid, and the question runs the pieces of its cone that are not valid, in the order of the cone, a piece left out by
              the gates of pass (20) being left not valid (it made no rows). A piece whose cone reads no changed fluent is carried as it is. SOUNDNESS. The rows of a piece are a function of the rows of the relations it reads, which are rows of its cone; so they are a function of the fluents its cone reads and of the rows that do not read the state; a piece is carried only when none of those fluents differs
              from the state its rows were made at; a piece is made again whole from the new state otherwise, so a rule with a negation needs no special handling; a piece in the cone of a question that has never run (not valid) is run. THE STATE THAT IS NOT NEAR is read in full: a module with no kept state (`new`, `restart`) starts with every piece not valid.
              WHAT IT REFUSES: nothing is refused by form (any piece may be carried); the timing decides whether a module has the pass (the make script keeps the modules with it beside the modules without it).
  B2, BY GROUP (NOT BUILT; the count of rust/engine/game/tests/translated_carry_count.rs decides it; see the note of plan 79.54 in docs/ANALYST_PLAN.md for what B2 would need and why it was not built).
  AS BUILT, where the body differs from the text above (said here because the text was committed first): (a) the evaluation kept is `cd` and the validity `cvalid` as said; the pooled evaluation of `next` and of `all_relations` is not changed (the next state reads the turn, which the kept evaluation never holds).
    (b) A piece whose cone reads the turn (`does`, `record`) has every fluent in its mask. (c) The cone of a piece is that of the clauses AS WRITTEN (`Plan.cone`), so a walk made for the keys its readers ask (pass asked) reads, in its mask, what the guards of its readers read.
    (d) Both modules have `carried_relations` (the rows the questions of the cones made, for the tests); the module with the pass hands out only the relations of valid pieces. (e) `scripts/gdl_to_rust_make.py` keeps the module with the pass as `<game>_carry.rs` beside `<game>_fast.rs` (every pass but `carry`);
    the lemma module and the placed modules are made without it.
  HELD TO THE MODULE WITHOUT THE PASS: rust/engine/game/tests/translated_carry.rs at every state of seeded games (each state following the last), across siblings, at jumps to an unrelated state, and every relation of the kept evaluation against the plain module; tests/test_plan79_54.py (planted cases).
PLAN 79.56 PART B: THE SUPPORT FILTER (SYSTEM_SPEC 11, blocks "Layer 2 unlocked for lemmas" and "The support filter"; LAYER 2, piece: rules; ledger rows 55, 56, 57). Interface written before the body. PASSES = (... "topdown", "carry", "support"); "support" brings "topdown" (and so "words", "cones", "keep"); with no "support" every module is the module of the commit before, byte for byte but for the translator hash line.
  THE LEMMA (lean/ideas/Ideas/Support.lean, `Test.run_congr`, `Test.run_of_disjoint`, no game in it): a test that asks inputs one at a time and branches on the answers gives the same result in two worlds that agree on every input it asked in the first.
  (26) "support" TAKES the clauses as written (`Description`: strata, signatures, the dependency graph), the passes before it, and the top-down functions of pass (24). IT FINDS, from the clauses alone (told no relation, sort or constant):
              A STATE RELATION B: a fluent `true.F` of p places.
              A HYPOTHETICAL RELATION H of m+p places (m >= 1, a KEY K at the first m places, a ROW at the last p), not recursive, every clause of which is a VIEW (one positive literal of a relation V with the head's variables, distinct, in the same order); every view V has clauses all of one of two forms:
                COPY: a positive literal of B whose p places are the head's p row variables in order; besides it, positive literals that bind no row variable ("guards": they say whether the key is in play), and negated literals whose places are variables of the key or of the row and that name no other variable; no test `distinct`.
                ADD: a clause with no positive literal of B at the row variables; the rows it gives at the key are the places the key changes (the guards it reads are real facts).
              H IS "B OUTSIDE THE PLACES THE KEY TOUCHES, GIVEN ROWS AT THOSE PLACES": H(K, .) differs from B(.) only at the rows of the ADD views at K and at the rows of B that a negated literal of a COPY clause names at K. D(K), the DIFFERING PLACES of K, is that set, written as patterns over the row (a number at a place the literal fixes, FREE at the others), and is read off the clauses: the rows of each ADD view at K, and for each negated literal of a COPY clause the rows of its relation at K, each turned into a pattern; a superset is safe.
              THE RELATIONS MADE FROM H (HD): the relations that depend on H. R, A TEST OF THE KEY: a relation of HD with exactly the key's m places that a relation outside HD reads. R CONFORMS when, for every clause of every relation of HD it reaches (the key of a relation is the places that hold the key variables of the clause that calls it, found by following the calls from R, whose key is its places): each literal is a literal of H with the key at its first m places in the order of the clause's key variables, or a literal of an HD relation with the whole key at some places in that order, or a GUARD (a relation outside HD that reads the state and has a key variable); no literal over a static relation, no `distinct`, no compound term, holds a key variable. A GUARD DOMAIN: a guard G* such that every clause of R reaches a positive literal of G* with the key through relations whose every clause holds one (R has no row at a key that G* has no row at). THE GUARD CLASS of a key: the rows of every guard of the cone at the key, the key's places removed (the guards' values are what the test reads besides H).
              A DESCRIPTION WITHOUT THIS SHAPE IS LEFT UNCHANGED: `Plan.support` is blank and `Plan.support_refused` names, for each relation that came near, the reason (the head lines print both).
  RETURNS, for each conforming R, in the same module: R's piece is not run in the cone of a question: `Game::run` runs `run_support` in its place, and the relations made only to answer R (those of HD that no other piece of the cones reads) are out of the cones (the cones are the written clauses' graph with R's reads taken to be the guard domain's, the fluents and the relations that do not read the state). In ONE QUESTION at ONE STATE, `run_support` does this: it takes the keys of the guard domain (the rows of the guard's relation made by its piece, the key places); for each key K, in the order of the rows: computes D(K) (top-down, the key bound); takes the guard class of K; if the class has a REFERENCE (S0, answer) and no pattern of D(K) is met by a pattern of S0, K takes the reference's answer, NO TEST; otherwise K is TESTED: the top-down function of R at K, with every call of a function of H recorded (its bound places, FREE elsewhere) in a word S; if D(K) is met by no pattern of S and the class has no reference, (S, answer) becomes the class's reference. Every other key is tested as before. No reference found: every key is tested as before. The references die with the state (they are made in the question). The rows of R are the keys whose answer is true, put into R's relation at the end. A td worklist that overflowed (`td.overflow`) makes the question run the pieces the module without the pass runs (the relations of HD in order and R's plain piece): the answer is then the plain one.
  A COUNTER, on under the profile switch only: `Game::support_counts() -> (states, keys, tests, answered_from_reference, no_reference_classes)`; `reset_support_counts()`.
  SOUNDNESS (Ideas.Support.Test.run_of_disjoint; the easy way only): R(K) is a test whose inputs are the facts of H(K, .) it consults (every consult is recorded as a pattern: a superset of the facts read), the facts of the guards at K (equal for two keys of one class), and static facts. H(K0, .) and H(K1, .) both agree with B outside D(K0) and D(K1); the reference's S0 meets neither; so both worlds agree on every input the reference's test asked, and the answer of K1 is the reference's. Nothing is proved about a game.
  HELD TO THE MODULE WITHOUT THE PASS: rust/engine/game/tests/translated_support.rs (the rows of every relation that the module with the pass makes equal those of the module `fast`, and the permitted moves, at every state of the seeded games used for the carry; a planted wrong reference is caught by the equality test); tests/test_plan79_56.py (the pass finds the shape in the clauses of a description whose every name is changed, and refuses where a clause reads the key outside H).
  THE KEPT MODULE `<game>_support.rs` is made by scripts/gdl_to_rust_make.py for every game where the shape is found; the modules without the pass stay as they are.
  AS BUILT 79.56, where the body differs from the text above (said here because the text was committed first):
    (a) "support" is NOT one of PASSES: it is in OPT_PASSES, a caller asks for it by name; "every pass" (PASSES, which the gates of the earlier plans take) is still the twenty-3, and `Plan.passes` lists it behind them. The head of a module with it holds `// passes: ...,support` and one line `// support: test ...` for each test found and `// support refused: <relation>: <reason>` for each relation that came near (a relation that no question reads is refused with that reason: the shape is there and nothing is changed).
    (b) D(K) is a set of ROWS, not of places alone: a negated literal of a copy clause names places (a pattern with FREE at the others); the rows of B at those places are read through an index of the top-down tables and put into D(K) one by one (a row of an add view is a row already). The first writing took the place alone and the pattern then met every read of a unit of a form, so no key ever became a reference.
    (c) ORDER. The clauses of the relations made from H are joined with the literals of H LAST (`JoinRule(late=)`, only in a module with the pass; the rows are the same, the order is another order of the same conjunction): joined by the reader's rule, a literal of H with a constant form was read first, which reads every unit of that form, and S held `(* * * *)`. Found by the log of a question (`Game::sp_log`, printed by translated_support.rs `the_support_words_chess`): 0 of 17987 keys were ever answered from a reference in the first game before this order, and about half behind it (see the OBS lines of translated_support.rs).
    (d) A key is in the DOMAIN when the guard domain G* has a row at it; the guard class is the sorted rows of every guard of the cone at the key, with the key's places removed, taken as one list; a class has at most one reference (the first key of the class, in the order of the sorted rows, whose test ends with D(K0) and S0 disjoint). The guard rows are read from the relations made by the pieces of the guards (they are in the cones: `Plan.support_reads`), the state is read as the top-down tables (loaded and cleared inside the piece, `Game::sp_s` holds the state of `load`).
    (e) `Game::sp_plant` turns the answer a reference keeps round (the test that plants a wrong reference); `Game::sp_log: Option<Vec<SpLog>>` keeps, for each key of a question, D(K), S and the answer (the material of the stop rule). Both are test hooks in every module with the pass.
    (f) The pass is an interpreter of the test by the top-down functions: the guards and the places D(K) are read from are made again top-down, key by key. It is NOT claimed to be faster than the bulk piece of the module without the pass: the part C and D tests print tests and nanoseconds.
PLAN 79.57 PART B: THE READS OF AN ANSWER (SYSTEM_SPEC 11, blocks "A divergence found and named" and "The support filter"; LAYER 2, piece: rules, on layer 1's kept graph; ledger rows 58, 57). Interface written before the body.
  THE LEMMA IS THE ONE OF 79.56 (lean/ideas/Ideas/Support.lean, `Test.run_of_disjoint`): an answer made by reading facts is the same in two worlds that agree on every fact that was read. The pass gives the facts read; the caller (the keeper of the graph) keeps them.
  (27) "reads" TAKES the clauses as written, the passes before it (it brings "topdown"), and a list `reads` of (relation, k) given by the CALLER of `translate` (the make script names the relations; this source names none): each relation is asked with its first k places bound (its KEY) and the rest free.
  IT FINDS, from the clauses alone: nothing to search for; for each (relation, k) it makes the top-down function of the relation with the first k places bound (the functions of the relations it reads are made as the way of pass (24) makes them), and REFUSES, naming the reason, a relation that is not derived, has fewer than k places, is not in the description, or whose walk reads a fluent of more places than the record holds.
  RETURNS, in the same module: `Game::ask_reads(&mut self, s: &State, rel: &str, key: &[u32]) -> Option<(Vec<Vec<u32>>, Vec<(u16, Vec<u32>)>)>`: TAKES a state, one of the relations given and a key of its k places; RETURNS the rows of the relation at the key (all places, sorted, no duplicate) and THE READS: every read of a fluent of the state made by the walk, in the order made, as (the fluent's number, its places as asked: a number where the read binds the place, FREE where it does not), a read of a row that is not there counted like the read of a row (it is recorded before it is made); None when the relation is not one given or the walk gave a question to the bottom-up way (the worklist overflowed).
    `READS_FLUENTS: &[&str]` the fluents in the order of their numbers (as the clauses name them, without the word `true.`), `READS: &[(&str, usize)]` the relations given with their k.
  WHO CALLS IT: the keeper's second rule (rust/engine/game/src/keeper.rs, through the test adapter tests/described: the places a read binds are the places of the support of an answer); the tests of the modules made with it (tests/translated_reads.rs). Nothing on the engine's path.
  SOUNDNESS: the walk is the top-down way of pass (24), which tests hold equal to the bottom-up rows; every consult of a fluent goes through a recorded statement (each form a literal of a fluent takes: a negated literal, a literal with no place left free, a literal with places left free); relations that do not read the state are not recorded. The module without "reads" is the module of the commit before, byte for byte but for the translator hash line.
"""
from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

from catspace.incubator.analyst.discovery import gdl as GDL

NOOP = "noop"
INPUT_RELS = ("does", "record")
PASSES = ("dense", "demand", "copy", "delta", "bits", "order", "keep", "scan", "cut", "shift", "base", "start", "ends", "orient", "asked", "project", "worth", "gated", "cones", "words", "tables", "topdown", "carry")
OPT_PASSES = ("support", "reads", "explain")                              # passes a caller asks for by name; "every pass" (PASSES) does not hold them (pass 26: AS BUILT, the module text says why)
QUESTION_RELS = ("legal", "terminal", "goal")           # read in full by the questions (as are the `next` relations)
TABLE_WORDS = 1 << 13                                   # pass (23): a walk gets a table when its chains are at most this many words
WORDS_MAX = 1 << 13                                     # pass (22): a fluent is held as the bits of a state when the product of the sizes of its sorts is at most this
DENSE_SPACE = 1 << 22                                   # a dense relation: at most this many tuples of its sorts ...
DENSE_KEYS = 1 << 16                                    # ... and at most this many keys for any index the joins ask by
SMALL_SCAN = 8                                          # a loop over a whole fluent is cheap when its sorts allow no more rows than this
START_MAX = 4                                           # pass (13): at most this many orders of one clause are written (the reader's and the best others)
BOUND = "\0"                                            # the mark of a constant that stands for a variable bound by the key of a demanded relation


def render_term(t) -> str:
    """A term of a clause as written: a variable or constant as it is, a compound as a list."""
    if isinstance(t, tuple):
        return "(" + " ".join(render_term(x) for x in t) + ")"
    return t.replace(BOUND, "?") if isinstance(t, str) else str(t)


def render_literal(x) -> str:
    """Pass (28): a literal of a clause as it is written in the description (a literal of a state fluent as `(true (f ...))`, a negation as `(not ...)`)."""
    if x[0] == "distinct":
        return f"(distinct {render_term(x[1])} {render_term(x[2])})"
    rel, args = x[1], x[2]
    body = " ".join(render_term(a) for a in args)
    inner = f"(true ({rel[len('true.'):]}{' ' + body if body else ''}))" if rel.startswith("true.") else f"({rel}{' ' + body if body else ''})"
    return f"(not {inner})" if x[0] == "neg" else inner


class Untranslatable(Exception):
    pass


class AskedRefuse(Exception):
    """A walk that the pass asked cannot restrict to the keys its readers ask, with the reason (the clause or the reader is named in it)."""


def count_vars(t, counts: dict):
    """The number of times each variable occurs in a term, added to `counts`."""
    if isinstance(t, tuple):
        for y in t[1:]:
            count_vars(y, counts)
    elif GDL.is_var(t):
        counts[t] = counts.get(t, 0) + 1


def subst_term(t, m: dict):
    """A term with its variables replaced as `m` says (the others stay)."""
    if isinstance(t, tuple):
        return (t[0],) + tuple(subst_term(y, m) for y in t[1:])
    return m.get(t, t) if GDL.is_var(t) else t


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def translator_hash() -> str:
    return sha(Path(__file__).read_bytes())


def ident(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "_", name)


# ------------------------------------------------------------------------------------------------------------------------------------------ the runtime text

RUNTIME = r'''
use std::collections::{HashMap, HashSet};
use std::hash::{BuildHasherDefault, Hasher};

/// The mark of an unbound place in a key. No term has this number.
pub const FREE: u32 = u32::MAX;

/// A small multiplicative hash (the one rustc uses), so that a lookup of a row of numbers costs a few instructions.
#[derive(Default, Clone, Copy)]
pub struct Fx(u64);
impl Fx {
    #[inline]
    fn add(&mut self, w: u64) { self.0 = (self.0.rotate_left(5) ^ w).wrapping_mul(0x517c_c1b7_2722_0a95); }
}
impl Hasher for Fx {
    #[inline]
    fn finish(&self) -> u64 { self.0 }
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        for c in bytes.chunks(8) { let mut w = [0u8; 8]; w[..c.len()].copy_from_slice(c); self.add(u64::from_le_bytes(w)); }
    }
    #[inline]
    fn write_u32(&mut self, i: u32) { self.add(i as u64); }
    #[inline]
    fn write_usize(&mut self, i: usize) { self.add(i as u64); }
}
type B = BuildHasherDefault<Fx>;

/// A relation of N arguments: its rows in the order made, a set of them, and for each set of argument places the joins ask by, an index from the numbers at those places to the rows.
pub struct Rel<const N: usize> {
    pub order: Vec<[u32; N]>,
    set: HashSet<[u32; N], B>,
    kp: &'static [&'static [usize]],
    ix: Vec<HashMap<[u32; N], Vec<[u32; N]>, B>>,
}
impl<const N: usize> Rel<N> {
    pub fn new(kp: &'static [&'static [usize]]) -> Rel<N> { Rel { order: vec![], set: HashSet::default(), kp, ix: kp.iter().map(|_| HashMap::default()).collect() } }
    #[inline]
    pub fn has(&self, r: &[u32; N]) -> bool { self.set.contains(r) }
    /// Add a row; true when it was not there.
    pub fn add(&mut self, r: [u32; N]) -> bool {
        if !self.set.insert(r) { return false; }
        self.order.push(r);
        for (i, k) in self.kp.iter().enumerate() {
            let mut key = [FREE; N];
            for &p in k.iter() { key[p] = r[p]; }
            self.ix[i].entry(key).or_default().push(r);
        }
        true
    }
    /// The rows whose places of the i-th index equal `key`'s.
    #[inline]
    pub fn get(&self, i: usize, key: &[u32; N]) -> &[[u32; N]] { match self.ix[i].get(key) { Some(v) => v.as_slice(), None => &[] } }
}

/// The rows of `rows` that equal `key` at the places `kp` (a scan: the rows new in a round of the semi-naive rule have no index).
pub fn scan<const N: usize>(rows: &[[u32; N]], kp: &[usize], key: &[u32; N]) -> Vec<[u32; N]> { rows.iter().filter(|r| kp.iter().all(|&p| r[p] == key[p])).copied().collect() }

/// What a clause gives: its rows and the work it did (rows visited by its loops, indexes asked).
pub struct Out<const N: usize> { pub rows: Vec<[u32; N]>, pub visits: u64, pub probes: u64 }
impl<const N: usize> Out<N> { pub fn new() -> Out<N> { Out { rows: vec![], visits: 0, probes: 0 } } }

/// The compound terms of a run, one number each: the same compound is the same number. Numbers begin where the constants end.
pub struct Terms { data: Vec<Vec<u32>>, ids: HashMap<Vec<u32>, u32, B> }
impl Terms {
    pub fn new() -> Terms { Terms { data: vec![], ids: HashMap::default() } }
    pub fn fun(&mut self, f: u32, args: &[u32]) -> u32 {
        let mut k = Vec::with_capacity(args.len() + 1);
        k.push(f);
        k.extend_from_slice(args);
        if let Some(&i) = self.ids.get(&k) { return i; }
        let i = (SYMS.len() + self.data.len()) as u32;
        self.data.push(k.clone());
        self.ids.insert(k, i);
        i
    }
    /// The functor and arguments of a compound (None: a constant).
    #[inline]
    pub fn compound(&self, t: u32) -> Option<(u32, &[u32])> {
        if (t as usize) < SYMS.len() || t == FREE { None } else { let v = &self.data[t as usize - SYMS.len()]; Some((v[0], &v[1..])) }
    }
    /// The name of a constant; of a compound, its functor's.
    pub fn name(&self, t: u32) -> &'static str { match self.compound(t) { Some((f, _)) => SYMS[f as usize], None => SYMS[t as usize] } }
    pub fn render(&self, t: u32) -> String {
        match self.compound(t) {
            Some((f, a)) => format!("({} {})", SYMS[f as usize], a.iter().map(|x| self.render(*x)).collect::<Vec<_>>().join(" ")),
            None => SYMS[t as usize].to_string(),
        }
    }
}

/// The work of one piece: calls, nanoseconds (when profiling), rows made, rows visited by its loops, indexes asked.
#[derive(Clone, Default, Debug)]
pub struct Stat { pub calls: u64, pub nanos: u64, pub rows: u64, pub visits: u64, pub probes: u64 }

/// What one evaluation of the state gave: the legal rows (role, turn), whether the end holds, the goal rows (role, value).
#[derive(Clone)]
pub struct Reading { pub legal: Vec<(u32, u32)>, pub terminal: bool, pub goals: Vec<(u32, u32)> }
'''
# PASS (7), the one change of the runtime text: a compound that is there is found by a key made on the stack, not in a vector made for each question.
RUNTIME_KEEP = RUNTIME.replace("""        let mut k = Vec::with_capacity(args.len() + 1);
        k.push(f);
        k.extend_from_slice(args);
        if let Some(&i) = self.ids.get(&k) { return i; }
""", """        if args.len() < 15 {
            let mut buf = [0u32; 16];
            buf[0] = f;
            buf[1..=args.len()].copy_from_slice(args);
            if let Some(&i) = self.ids.get(&buf[..=args.len()]) { return i; }
        }
        let mut k = Vec::with_capacity(args.len() + 1);
        k.push(f);
        k.extend_from_slice(args);
        if let Some(&i) = self.ids.get(&k) { return i; }
""")
assert RUNTIME_KEEP != RUNTIME
# The text of the pieces of a term: a symbol of a term, kept separate so the names scan sees the language's words only.


# ------------------------------------------------------------------------------------------------------------------------------------------ reading the description

class Description:
    """The description read: strata, arities, symbols, which relations read the state."""

    def __init__(self, text: str):
        raw = GDL.clauses(GDL.parse(text))                      # (read first: a relation used with two numbers of arguments is named here, not by the reader's own error)
        self.arity: dict = {}
        syms: set = {NOOP}

        def note_term(t):
            if isinstance(t, tuple):
                syms.add(t[0])
                for y in t[1:]:
                    note_term(y)
            elif not GDL.is_var(t):
                syms.add(t)

        def note_arity(rel, n, where):
            if self.arity.setdefault(rel, n) != n:
                raise Untranslatable(f"{rel} is used with {self.arity[rel]} and with {n} arguments: {where}")

        for c in raw:
            note_arity(c.head[0], len(c.head[1]), c.text)
            for t in c.head[1]:
                note_term(t)
            for lit in c.body:
                if lit[0] == "distinct":
                    note_term(lit[1])
                    note_term(lit[2])
                else:
                    note_arity(lit[1], len(lit[2]), c.text)
                    for t in lit[2]:
                        note_term(t)
        self.prog = GDL.Program(text)
        p = self.prog
        for members in p.sorts.values():
            for m in members:
                if isinstance(m, tuple):
                    syms.update(x for x in m)
                else:
                    syms.add(m)
        self.syms = sorted(s for s in syms if not GDL.is_var(s))
        self.sym_id = {s: i for i, s in enumerate(self.syms)}
        self.nodes = sorted(p.nodes)
        self.static = {r: not p.reads_input.get(r, GDL._is_input(r)) for r in self.nodes}
        self.fluents = sorted(r[len("true."):] for r in self.nodes if r.startswith("true."))
        for f in self.fluents:
            if f"true.{f}" not in self.arity:
                raise Untranslatable(f"the fluent {f} has no arity")
        names: dict = {}
        self.rid: dict = {}
        for r in self.nodes:
            i = "r_" + ident(r)
            if i in names:
                raise Untranslatable(f"relations {r} and {names[i]} have one Rust name {i}")
            names[i] = r
            self.rid[r] = i
        self.strata = p.strata
        self.flags: set = set()                                  # the relations of no places that a lemma's rewrite made (plan 79.45): made in full, never answered on demand
        for comp, rules in self.strata:
            for r in rules:
                for st in r.steps:
                    if st[0] != "distinct" and len(r.head) > 8:
                        raise Untranslatable(f"too many arguments: {r.clause.text}")

    def cone(self, wanted) -> set:
        visited: set = set()
        todo = [w for w in wanted if w in self.prog.nodes]
        while todo:
            r = todo.pop()
            if r in visited:
                continue
            visited.add(r)
            todo.extend(self.prog.deps.get(r, ()))
        return visited

    def dynamic_meeting(self, cone: set) -> list:
        return [k for k, (comp, _) in enumerate(self.strata) if not self.static[sorted(comp)[0]] and comp & cone]

    def index_places(self) -> dict:
        """{relation: sorted list of the argument-place tuples the joins ask by}."""
        out: dict = {r: set() for r in self.nodes}
        for _, rules in self.strata:
            for rule in rules:
                for st in rule.steps:
                    if st[0] == "pos" and st[3]:
                        out[st[1]].add(tuple(st[3]))
        return {r: sorted(v) for r, v in out.items()}


# ------------------------------------------------------------------------------------------------------------------------------------------ one clause as nested loops

def comp_vars(p) -> set:
    """The variable numbers of a compiled term."""
    if p[0] == "v":
        return {p[1]}
    if p[0] == "c":
        return set()
    out: set = set()
    for q in p[2]:
        out |= comp_vars(q)
    return out


def bind_term(t, names: set):
    """A term with the variables in `names` taken as bound (each becomes a constant marked BOUND)."""
    if isinstance(t, tuple):
        return (t[0],) + tuple(bind_term(y, names) for y in t[1:])
    if GDL.is_var(t) and t in names:
        return BOUND + t[1:]
    return t


def bind_clause(c, names: set):
    head = (c.head[0], tuple(bind_term(a, names) for a in c.head[1]))
    body = []
    for lit in c.body:
        if lit[0] == "distinct":
            body.append(("distinct", bind_term(lit[1], names), bind_term(lit[2], names)))
        else:
            body.append((lit[0], lit[1], tuple(bind_term(a, names) for a in lit[2])))
    return GDL.Clause(head, body, c.text)


def only_exists(rule, k: int) -> bool:
    """Whether the positive literal at `k` of a rule is read only for whether a row exists: its places not bound by the loops above are distinct variables that nothing later it (a later literal, the head) uses."""
    st = rule.steps[k]
    args, keypos = st[2], st[3]
    fresh: set = set()
    for i, a in enumerate(args):
        if i in keypos:
            continue
        if a[0] != "v" or a[1] in fresh:
            return False
        fresh.add(a[1])
    later: set = set()
    for h in rule.head:
        later |= comp_vars(h)
    for s2 in rule.steps[k + 1:]:
        if s2[0] == "distinct":
            later |= comp_vars(s2[1]) | comp_vars(s2[2])
        else:
            for a in s2[2]:
                later |= comp_vars(a)
    return not (fresh & later)


def uses_of(rule, demanded: set) -> list:
    """The demanded relations a written rule asks, each as (relation, places bound, mode): mode "x" when only the existence of a row is asked (a negated literal, an all-bound test, a literal read only for being non-vacant), else "e" (the rows)."""
    out = []
    for k, st in enumerate(rule.steps):
        if st[0] == "neg" and st[1] in demanded:
            out.append((st[1], tuple(range(len(st[2]))), "x"))
        elif st[0] == "pos" and st[1] in demanded:
            out.append((st[1], tuple(st[3]), "x" if only_exists(rule, k) else "e"))
    return out


class JoinRule:
    """A clause joined in an order chosen here: the reader's own choice (`GDL._Rule`: the next positive literal with the most bound arguments, a negation or a `distinct` as soon as its variables are bound), except that a literal of a
    relation in `avoid` that no variable already bound reaches (constants do not count) is taken last (pass 2: a demanded relation is joined from a key, never read whole), and the literal `first` (an element of the clause's body) is taken before every other (pass 4). With `avoid` vacant the steps are the reader's, to the tuple (tests/test_plan79_38.py pins this on every clause
    of every description). The fields are the reader's `_Rule`: clause, nvars, head_rel, head, steps, rec_steps."""

    def __init__(self, clause, scc: set, avoid=(), first=None, est=None, need=None, late=()):
        self.clause = clause
        names: dict = {}

        def comp(t):
            if GDL.is_var(t):
                if t not in names:
                    names[t] = len(names)
                return ("v", names[t])
            if isinstance(t, tuple):
                return ("f", t[0], tuple(comp(y) for y in t[1:]))
            return ("c", t)

        pos = [x for x in clause.body if x[0] == "pos"]
        rest = [x for x in clause.body if x[0] != "pos"]
        bound: set = set()
        order = []
        pending = list(pos)
        waiting = list(rest)

        def lit_vars(x):
            if x[0] == "distinct":
                return GDL.term_vars(x[1], set()) | GDL.term_vars(x[2], set())
            acc: set = set()
            for a in x[2]:
                GDL.term_vars(a, acc)
            return acc

        def flush():
            for x in list(waiting):
                if lit_vars(x) <= bound:
                    order.append(x)
                    waiting.remove(x)

        flush()
        while pending:
            def ranking(x):
                if first is not None and x is first:
                    return (2,)
                nb = sum(1 for a in x[2] if GDL.term_vars(a, set()) <= bound)
                keyed = sum(1 for a in x[2] if GDL.term_vars(a, set()) and GDL.term_vars(a, set()) <= bound)
                waits = bool(need) and x[1] in need and not all(GDL.term_vars(x[2][p], set()) <= bound for p in need[x[1]])        # pass (24): a relation asked for the places it keeps (a walk) is taken when they are bound
                return (not waits, x[1] not in late, not (x[1] in avoid and keyed == 0 and len(x[2]) > 0), nb == len(x[2]), nb, -est(x, bound, scc) if est else 0, -len(x[2]))
            best = max(pending, key=ranking)
            pending.remove(best)
            order.append(best)
            bound |= lit_vars(best)
            flush()
        if waiting:
            raise GDL.GDLError(f"unsafe clause (a variable of a negation, a distinct or the head is bound by no positive literal): {clause.text}")
        hv: set = set()
        for a in clause.head[1]:
            GDL.term_vars(a, hv)
        if not hv <= bound:
            raise GDL.GDLError(f"unsafe clause (head variable unbound): {clause.text}")
        self.steps = []
        bound_now: set = set()
        for x in order:
            if x[0] == "distinct":
                self.steps.append(("distinct", comp(x[1]), comp(x[2])))
                continue
            args = tuple(comp(a) for a in x[2])
            if x[0] == "neg":
                self.steps.append(("neg", x[1], args))
                continue
            keypos = tuple(i for i, a in enumerate(x[2]) if GDL.term_vars(a, set()) <= bound_now)
            self.steps.append(("pos", x[1], args, keypos))
            bound_now |= lit_vars(x)
        self.order = order                                      # the body's literals in the order written (pass (17) reads which literals come before a literal)
        self.nvars = len(names)
        self.head_rel = clause.head[0]
        self.head = tuple(comp(a) for a in clause.head[1])
        self.rec_steps = [i for i, s in enumerate(self.steps) if s[0] == "pos" and s[1] in scc]


class Sizes:
    """Pass (6): what is known of the size of a relation at translation, to choose among literals that start equally well. A relation that does not read the state has the rows the reader's evaluator made once from the description; a relation that reads the
    state is as large as the product of the sizes of its sorts; one without a signature (an input) is taken as one row; a relation of the set the clause belongs to is taken as half a row, as the rows new in a round are the fewest. The estimate of a literal is the rows of its relation over the sizes of the sorts of the places that are bound."""

    def __init__(self, d: "Description"):
        self.d = d

    def size_of(self, sort) -> int:
        ms = self.d.prog.sorts.get(sort)
        return max(1, len(ms)) if ms else 64

    def rows(self, rel: str) -> float:
        d = self.d
        r = d.prog.static.get(rel) if d.static.get(rel) else None
        if r is not None:
            return float(max(1, len(r.facts)))
        sig = d.prog.signature(rel)
        if sig is None:
            return 1.0
        n = 1.0
        for sort in sig:
            n *= self.size_of(sort)
        return n

    def __call__(self, lit, bound: set, scc=()) -> float:
        if lit[1] in scc:
            return 0.5                                         # the rows of the set being made that are new in a round: the fewest, and the semi-naive rule reads them first
        sig = self.d.prog.signature(lit[1])
        n = self.rows(lit[1])
        if sig is not None and len(sig) == len(lit[2]):
            for sort, a in zip(sig, lit[2]):
                if GDL.term_vars(a, set()) <= bound:
                    n /= self.size_of(sort)
        return max(n, 1e-9)


class Plan:
    """What the passes decided for one description, from its clauses, sorts, signatures and dependency graph alone (no name of a relation is known here).
    dense:    {relation: its sorts, one for each place}                  (pass "dense")
    demanded: the relations not made; modes: {relation: [(places bound, mode)]} the ways they are asked, test_modes: the extra ways the tests ask; fname: {(relation, places, mode): Rust function name};
    ask_rules: {(relation, places): [(compiled clause with the key's variables taken as bound, the clause)]}   (pass "demand")
    copy:     {next relation: [(rule, {variable number: place})]}        (pass "copy")"""

    def __init__(self, d: Description, passes):
        for p in passes:
            if p not in PASSES and p not in OPT_PASSES:
                raise Untranslatable(f"no pass named {p} (the passes: {', '.join(PASSES + OPT_PASSES)})")
        self.d = d
        if "base" in passes and "scan" not in passes:
            passes = tuple(passes) + ("scan",)                 # the clauses pass (8) makes ask words with their key tiles unbound, which only pass (10) answers by the words held (on top of bits alone it made the module half as fast: diagnosed, 79.41 part A)
        if "ends" in passes or "tables" in passes:
            passes = tuple(passes) + tuple(p for p in ("bits", "shift") if p not in passes)     # the pass ends is a form of a walk made by the pass shift, over relations held as bits
        if "worth" in passes and "start" not in passes:
            passes = tuple(passes) + ("start",)                 # the pass worth admits orders the pass start would refuse
        if ("asked" in passes or "project" in passes or "worth" in passes) and "bits" not in passes:
            passes = tuple(passes) + ("bits",)                  # the passes asked and project read relations held as bits
        if ("support" in passes or "reads" in passes or "explain" in passes) and "topdown" not in passes:
            passes = tuple(passes) + ("topdown",)               # the support filter tests a key by the top-down functions (pass 26); the reads of an answer are the top-down walk's (pass 27)
        if "topdown" in passes:
            passes = tuple(passes) + tuple(p for p in ("words", "cones") if p not in passes)         # the top-down way reads the facts of a state as the bits of words (pass 22) and falls back on the legal rows alone (pass 21)
        if "carry" in passes and "cones" not in passes:
            passes = tuple(passes) + ("cones",)                 # the carry keeps the evaluation of a state for the questions of the cones (pass 21)
        if ("cones" in passes or "words" in passes) and "keep" not in passes:
            passes = tuple(passes) + ("keep",)                  # the passes cones and words change the questions of a module that keeps its working memory (pass 7)
        self.passes = tuple(p for p in PASSES if p in passes) + tuple(p for p in OPT_PASSES if p in passes)
        self.support: list = []                                  # pass (26): the tests found, each a dict (SupportFinder.shape); decided in `translate`, which knows whether the top-down way was made
        self.support_refused: dict = {}                          # pass (26): relation -> why it is not a test of the shape
        self.support_h: list = []                                # pass (26): the hypothetical relations of the tests, in order
        self.support_reads: dict = {}                            # pass (26): test -> the relations its piece reads (the guards): the cones are made with these in place of the clauses' reads
        self.reads: list = []                                    # pass (27): the (relation, key places) pairs whose answers come with the fluent facts they read; decided in `translate`
        self.reads_refused: dict = {}                            # pass (27): relation -> why it is not made
        self.cones: bool = "cones" in self.passes                # pass (21)
        self.topdown: bool = "topdown" in self.passes            # pass (24)
        self.carry: bool = "carry" in self.passes                # pass (25)
        self.td_refused: str = ""                               # why pass (24) made no top-down way (blank: it was made)
        self.tables: bool = "tables" in self.passes              # pass (23)
        self.tables_found: dict = {}                            # relation of a walk -> (the chains of its table, the words of its table)   (pass "tables")
        self.tables_refused: dict = {}                          # relation of a walk -> why pass (23) keeps its shifts
        self.chain_used: dict = {}                              # table id -> (chains, number of keys, tile size)   (pass "tables")
        self.words: dict = {}                                   # fluent -> (its sorts, the number of words)   (pass "words")
        self.copy: dict = {}
        self.demanded: set = set()
        self.modes: dict = {}
        self.test_modes: dict = {}
        self.fname: dict = {}
        self.ask_rules: dict = {}
        self.dense: dict = {}
        self.delta: dict = {}                                   # stratum -> [(rule written in full, [(the delta literal's relation, the rule joined from the delta rows)])]   (pass "delta")
        self.bits: dict = {}                                    # relation -> BitInfo   (pass "bits")
        self.bits_refused: dict = {}                            # relation -> why pass (5) does not hold it as bits
        self.sids: dict = {}
        self.tids: dict = {}
        self.bit_rec: list = []
        self.keep: bool = "keep" in self.passes                  # pass (7)
        self.keep_out: set = set()                               # the arities of the rows a clause that is not held as words gives (their buffers are kept)
        self.keep_rec: list = []                                 # the relations of a recursive set that is not held as words (the rows new in a round are kept)
        self.shift_tables: dict = {}                            # (onward relation, tile read at, next tile) -> (number, displacements by key, keys, tiles that are keys)   (pass "shift")
        self.shift_used: set = set()                            # the tables the last writing of the clauses used
        self.shift_refused: dict = {}                           # relation of a recursive set -> why pass (9) does not compile it by shifts
        self.start_refused: dict = {}                           # (clause text, order) -> why pass (13) does not write that order of the clause
        self.start_choice: dict = {}                            # clause key -> the orders (indexes) admitted, decided in the dry run of the last round of find_bits and kept for the final writing
        self.ends: dict = {}                                    # relation of a walk made as its ends -> the stop literal its readers entail   (pass "ends")
        self.ends_refused: dict = {}                            # relation of a walk -> why pass (15) does not make it as its ends, the reader clause named
        self.asked: dict = {}                                   # relation of a walk -> {"k": its stratum, "plain": ids of the base clauses run when Dw.full, "guarded": ids of the clauses with the prefix of a reader added}   (pass "asked")
        self.asked_refused: dict = {}                           # relation of a walk -> why pass (17) does not restrict its keys, the reader or the clause named
        self.projected: list = []                               # [(clause text, [(the relation a literal read, the projection it reads now)])]   (pass "project")
        self.project_refused: dict = {}                         # clause text -> why a literal that could be a projection remains as written
        self.gated: dict = {}                                   # stratum -> the Rust test that its piece is not run in this question (a function of the rows of flags alone)   (pass "gated")
        self.gated_flags: dict = {}                             # stratum -> the flags (relations) its test reads, those of the strata it calls included
        self.gated_refused: dict = {}                           # stratum -> why pass (20) does not gate it: the reader that no flag literal shuts
        self.worth_taken: dict = {}                             # (clause text, order) -> the orientations of relations that do not read the state that pass (19) made for the order   (pass "worth")
        self.start_off: bool = False                            # pass (13): while the orientations the reader's orders ask are being found, no other order is written
        self.start_base: dict | None = None                     # relation -> the orientations asked by the clauses in the reader's orders, all of them (an order is admitted when it asks no other: the same whatever the order the clauses are written in)
        self.start_written: dict = {}                           # clause key -> (the number of orders written, the number of orders found)
        self.units: dict = {}                                   # the products of two sorts found (pass bits), with their weights
        self.refused: dict = {}                                # relation -> why pass (2) keeps it in full: the (relation, places) chain and the clause that defeats it
        self.est = Sizes(d) if "order" in self.passes else None                              # pass (6): what is known of the size of a relation, for the order of joining
        self.rules: dict = {k: rules for k, (comp, rules) in enumerate(d.strata)}          # the rules of each stratum as they are written (the reader's, or joined in another order by pass (2) or (6))
        if self.est is not None:
            self.rules = {k: [JoinRule(x.clause, comp, (), None, self.est) for x in rules] for k, (comp, rules) in enumerate(d.strata)}
        self.unfolded: list = []                                # the views unfolded into their readers, in order   (pass "base")
        self.base_refused: dict = {}                            # relation with a product clause -> why pass (8) does not unfold it
        self.ub = 0
        if "base" in self.passes:
            self.find_base()
        self.pre_info: dict = self.bit_candidates() if "bits" in self.passes else {}
        self.pre_bits: set = set(self.pre_info)                  # pass (5) comes before pass (2): a relation that can be held as bits is not answered on demand
        if "asked" in self.passes:
            self.find_asked()
        self.rules_unread: dict = {k: list(rules) for k, rules in self.rules.items()}      # the clauses before pass (18) reads a projection in place of a relation: the entailment of pass (15) is shown on these (a literal of the relation entails the one of its projection, which the homomorphism does not see)
        if "project" in self.passes:
            self.find_project()
        if "demand" in self.passes:
            self.find_demand()
        if "copy" in self.passes:
            self.find_copy()
        if "delta" in self.passes:
            self.find_delta()
        self.places = self.index_places()
        if "bits" in self.passes:
            self.find_bits()
        if "dense" in self.passes:
            self.find_dense()
        if "gated" in self.passes:
            self.find_gated()
        if "words" in self.passes:
            self.find_words()

    # ---------------------------------------------------------------------------------------------- pass (22): the state's facts as the bits of words

    def find_words(self):
        """The fluents held as the bits of a state: every place of a declared sort of constants, the product of the sizes at most WORDS_MAX."""
        d = self.d
        sorts = d.prog.sorts
        for f in d.fluents:
            r = "true." + f
            n = d.arity.get(r, 0)
            sig = d.prog.signature(r)
            if n == 0 or sig is None or len(sig) != n:
                continue
            if not all(s in sorts and sorts[s] and all(isinstance(m, str) for m in sorts[s]) and len(sorts[s]) < NONE for s in sig):
                continue
            space = 1
            for s in sig:
                space *= len(sorts[s])
            if space <= WORDS_MAX:
                self.words[f] = (tuple(sig), (space + 63) // 64)

    # ---------------------------------------------------------------------------------------------- pass (20): a piece not run where every reader is shut off

    def find_gated(self):
        """Pass (20) (module text): the strata whose readers are all shut by a flag literal that is false in the state, or are heads of such strata; `gated[k]` is the Rust test that the piece of k is not to be run. The strata that fail are named in `gated_refused` with the reader."""
        d = self.d
        sidx = {h: k for k, (comp, _r) in enumerate(d.strata) for h in comp}
        flags = {}
        for k, (comp, _r) in enumerate(d.strata):
            if len(comp) != 1:
                continue
            r = next(iter(comp))
            if d.arity.get(r, 0) == 0 and not d.static[r] and r not in QUESTION_RELS and r not in INPUT_RELS and not r.startswith(("true.", "next.", "init.")) and not (comp & self.demanded) and self.rules[k] and not any(x.rec_steps for x in self.rules[k]):
                flags[r] = k
        reads: dict = {}
        for k2 in range(len(d.strata)):
            for rule in self.rules[k2]:
                met: set = set()
                for st in rule.steps:
                    k = sidx.get(st[1]) if st[0] != "distinct" else None
                    if k is not None and k != k2 and k not in met:
                        met.add(k)
                        reads.setdefault(k, []).append((k2, rule))

        def flag_lits(rule):
            return [(st[1], st[0] == "pos") for st in rule.steps if st[0] in ("pos", "neg") and st[1] in flags]
        cand: set = set()
        for k, (comp, _r) in enumerate(d.strata):
            if d.static[sorted(comp)[0]] or (comp & self.demanded) or not self.rules[k] or k not in reads:
                continue
            if any(h in QUESTION_RELS or h in INPUT_RELS or h.startswith(("true.", "next.", "init.")) or h in flags for h in comp):
                continue
            cand.add(k)
        while True:
            changed = False
            for k in sorted(cand):
                for k2, rule in reads[k]:
                    if not flag_lits(rule) and k2 not in cand:
                        cand.discard(k)
                        self.gated_refused[k] = f"{'+'.join(sorted(d.strata[k][0]))}: read by {rule.clause.text.replace(chr(10), ' ')}, which no flag literal shuts and whose head is not gated"
                        changed = True
                        break
            used = {f for k in cand for _k2, rule in reads[k] for f, _pos in flag_lits(rule)}
            need = self.cone(sorted(used)) if used else set()
            for k in sorted(cand):
                if d.strata[k][0] & need:
                    cand.discard(k)
                    self.gated_refused[k] = f"{'+'.join(sorted(d.strata[k][0]))}: it is made for a flag that shuts a reader"
                    changed = True
            if not changed:
                break
        for k in sorted(cand):
            parts = []
            fl: set = set()
            for k2, rule in reads[k]:
                alts = []
                for f, pos in flag_lits(rule):
                    alts.append(f"d.{d.rid[f]}.order.is_empty()" if pos else f"!d.{d.rid[f]}.order.is_empty()")
                    fl.add(f)
                if k2 in cand:
                    alts.append(f"off(d, {k2})")
                text = "(" + " || ".join(alts) + ")"
                if text not in parts:
                    parts.append(text)
            self.gated[k] = " && ".join(parts)
            self.gated_flags[k] = fl
        for k in sorted(cand):                                   # the flags of the strata a test calls are read too
            met, todo = set(self.gated_flags[k]), [k]
            while todo:
                x = todo.pop()
                for k2, rule in reads[x]:
                    if k2 in cand and k2 not in todo:
                        todo.append(k2)
                        met |= self.gated_flags[k2]
            self.gated_flags[k] = met
        self.flag_strata = flags

    def gate_order(self, ks: list) -> list:
        """The pieces `ks` (in the order of the strata) with, for each gated piece, the cone of every flag its test reads moved before it where that cone came later it (the cone does not read the piece: no flag reads a gated stratum)."""
        out = list(ks)
        for k in ks:
            if k not in self.gated:
                continue
            for f in sorted(self.gated_flags[k]):
                cone = {j for j, (comp, _r) in enumerate(self.d.strata) if comp & self.cone([f])}
                at = out.index(k)
                later = [x for x in out[at + 1:] if x in cone]
                if later:
                    out = [x for x in out[:at] if x not in later] + later + [k] + [x for x in out[at + 1:] if x not in later]
        return out

    # ---------------------------------------------------------------------------------------------- pass (8): a view unfolded into its readers

    @staticmethod
    def is_product(clause) -> bool:
        """Whether the positive literals of a clause fall in two or more groups that share no variable, each binding a variable of the head (so the rows of the head are as many as the product of the groups' rows)."""
        pos = [x for x in clause.body if x[0] == "pos"]
        hv: set = set()
        for a in clause.head[1]:
            GDL.term_vars(a, hv)
        groups: list = []
        for lit in pos:
            vs: set = set()
            for a in lit[2]:
                GDL.term_vars(a, vs)
            if not vs:
                continue
            merged = [g for g in groups if g & vs]
            for g in merged:
                groups.remove(g)
                vs |= g
            groups.append(vs)
        return sum(1 for g in groups if g & hv) >= 2

    def reads_of(self, rel: str) -> list:
        """[(stratum, rule)] of every written rule that has a literal over `rel`."""
        return [(k, r) for k in range(len(self.d.strata)) for r in self.rules[k] if any(st[0] != "distinct" and st[1] == rel for st in r.steps)]

    def find_base(self):
        """Pass (8): every view (see the module text) unfolded into the clauses that read it; repeated, for a reader that gains a product clause by it is a view in turn. Refusals, with the clause, in `base_refused`."""
        d = self.d
        while True:
            progress = False
            for k, (comp, _r) in enumerate(d.strata):
                if len(comp) != 1:
                    continue
                s = next(iter(comp))
                rules = self.rules[k]
                if s in self.unfolded or d.static[s] or s in QUESTION_RELS or s in INPUT_RELS or s.startswith(("true.", "next.", "init.")) or not rules or any(x.rec_steps for x in rules):
                    continue
                prod = [x for x in rules if self.is_product(x.clause)]
                readers = self.reads_of(s)
                if not prod or not readers:
                    continue
                why = None
                for x in rules:
                    if any(h[0] != "v" for h in x.head) or len({h[1] for h in x.head}) != len(x.head):
                        why = f"a head that is not a list of distinct variables: {x.clause.text}"
                for _k2, rd in readers:
                    for st in rd.steps:
                        if st[0] == "neg" and st[1] == s:
                            why = why or f"read negated, and the negation of a union is no conjunction of literals: {rd.clause.text}"
                if why:
                    self.base_refused[s] = why.replace("\n", " ")
                    continue
                self.base_refused.pop(s, None)
                self.unfold(k, s)
                progress = True
            if not progress:
                break

    def unfold(self, ks: int, s: str):
        """Replace every positive literal over `s` in the written rules by the body of each clause of `s` (variables of that clause renamed to names not used, its head variables replaced by the literal's terms)."""
        d = self.d
        s_clauses = [x.clause for x in self.rules[ks]]

        def subst(t, m):
            if isinstance(t, tuple):
                return (t[0],) + tuple(subst(y, m) for y in t[1:])
            return m.get(t, t) if GDL.is_var(t) else t

        def inline(c, j, sc):
            self.ub += 1
            m = {h: c.body[j][2][i] for i, h in enumerate(sc.head[1])}
            for lit in sc.body:
                vs: set = set()
                for a in (lit[1:] if lit[0] == "distinct" else lit[2]):
                    GDL.term_vars(a, vs)
                for v in vs:
                    m.setdefault(v, f"?_u{self.ub}_{v[1:]}")
            body = []
            for lit in sc.body:
                if lit[0] == "distinct":
                    body.append(("distinct", subst(lit[1], m), subst(lit[2], m)))
                else:
                    body.append((lit[0], lit[1], tuple(subst(a, m) for a in lit[2])))
            return GDL.Clause(c.head, list(c.body[:j]) + body + list(c.body[j + 1:]), c.text.replace("\n", " ") + f" (with {s} unfolded)")
        for k2, (comp, _r) in enumerate(d.strata):
            if not any(st[0] != "distinct" and st[1] == s for r in self.rules[k2] for st in r.steps):
                continue
            new = []
            for r in self.rules[k2]:
                todo = [r.clause]
                while True:
                    at = next(((i, j) for i, c in enumerate(todo) for j, lit in enumerate(c.body) if lit[0] == "pos" and lit[1] == s), None)
                    if at is None:
                        break
                    i, j = at
                    c = todo.pop(i)
                    todo[i:i] = [inline(c, j, sc) for sc in s_clauses]
                new += [JoinRule(c, comp, (), None, self.est) for c in todo]
            self.rules[k2] = new
        self.unfolded.append(s)

    # ---------------------------------------------------------------------------------------------- pass (17): a walk made only for the keys asked

    def find_asked(self):
        """Pass (17): for each walk that the module text's conditions cover, its base clauses replaced by the base clauses with the prefix of each reader added (`Plan.asked`); a walk that does not meet them remains as it is, the clause or the reader named in `Plan.asked_refused`."""
        d = self.d
        for k, (comp, _r) in enumerate(d.strata):
            if len(comp) != 1:
                continue
            w = next(iter(comp))
            rules = self.rules[k]
            if w not in self.pre_bits or d.static[w] or w in QUESTION_RELS or w in INPUT_RELS or w.startswith(("true.", "next.", "init.")) or not any(x.rec_steps for x in rules):
                continue
            try:
                new, plain, guarded = self.asked_rules(k, w, comp, rules)
            except AskedRefuse as e:
                self.asked_refused[w] = str(e)
                continue
            self.asked[w] = {"k": k, "plain": plain, "guarded": guarded}
            self.rules[k] = new
            for (k2, ident_x), x2 in self.asked_reorder.items():
                self.rules[k2] = [x2 if id(x) == ident_x else x for x in self.rules[k2]]

    def asked_rules(self, k: int, w: str, comp: set, rules: list):
        """The clauses of the walk `w` (stratum k) with the prefix of each reader added to each base clause (module text, (17)): -> (the rules, the ids of the clauses run when everything is asked for, the ids of the clauses with a prefix); AskedRefuse, naming the clause or the reader, where a condition fails."""
        d = self.d
        before = d.cone([w]) - {w}                               # the relations W reads, closed: made before W in every order of the strata (the order of strata of relations that do not meet is by name and decides nothing here)
        info = self.pre_info[w]
        keyp = [p for i, t in enumerate(info.tiles) if i != info.canon for p in t.places]
        if not keyp:
            raise AskedRefuse(f"the relation has no key before its last tile: {rules[0].clause.text}".replace("\n", " "))
        for x in rules:
            text = x.clause.text.replace("\n", " ")
            if x.rec_steps:
                if len(x.rec_steps) != 1:
                    raise AskedRefuse(f"a recursive clause reads the relation {len(x.rec_steps)} times: {text}")
                la = x.steps[x.rec_steps[0]][2]
                if any(la[p][0] != "v" or x.head[p][0] != "v" or la[p][1] != x.head[p][1] for p in keyp) or len({la[p][1] for p in keyp}) != len(keyp):
                    raise AskedRefuse(f"a recursive clause that does not keep the key, in distinct variables: {text}")
            else:
                hv = [x.clause.head[1][p] for p in keyp]
                if any(not GDL.is_var(a) for a in hv) or len(set(hv)) != len(hv):
                    raise AskedRefuse(f"a base clause with no distinct variables at the places of the key: {text}")
        readers = []
        for k2, (c2, _r2) in enumerate(d.strata):
            if k2 == k:
                continue
            for x in self.rules[k2]:
                lits = [lit for lit in x.clause.body if lit[0] in ("pos", "neg") and lit[1] == w]
                if lits:
                    text = x.clause.text.replace("\n", " ")
                    if any(lit[0] == "neg" for lit in lits):
                        raise AskedRefuse(f"the relation is read negated: {text}")
                    readers.append((x if hasattr(x, "order") else JoinRule(x.clause, c2, (), None, self.est), lits, text, k2, c2))
        if not readers:
            raise AskedRefuse("no clause reads it")
        admitted = lambda rel: rel != w and (rel in before or d.static.get(rel, False) or rel.startswith("true.") or rel in INPUT_RELS)
        ws = getattr(d, "over", {}).get(w)
        over = lambda lit: ("pos", ws, lit[2]) if (lit[1] == w and ws) else None            # a read of the walk, in a guard, stands as its static over-approximation (module text, (17, widened))
        prefixes = []
        self.asked_reorder = {}
        for x, wls, text, k2, c2 in readers:
            try:
                got = self.asked_reads(x, wls, text, keyp, admitted, over)
            except AskedRefuse as e:
                if "asks it unbound" not in str(e):
                    raise
                x2 = JoinRule(x.clause, c2, (w,), None, self.est)       # the reader joined with the walk taken last where no variable reaches it: a relation that reads the walk may bind its key (module text, (17, widened))
                got = self.asked_reads(x2, [lit for lit in x2.clause.body if lit[0] == "pos" and lit[1] == w], text, keyp, admitted, over)
                self.asked_reorder[(k2, id(x))] = x2
            prefixes += got
        nothing_for = ""
        out, plain, guarded = [], set(), set()
        for x in rules:
            if x.rec_steps:
                out.append(x)
                continue
            c = x.clause
            hv = [c.head[1][p] for p in keyp]
            held: set = set()                                   # the key variables the base clause binds through a relation that reads the state: its keys are those of the state already
            for lit in c.body:
                if lit[0] == "pos" and not d.static.get(lit[1], False):
                    for a in lit[2]:
                        GDL.term_vars(a, held)
            made = []
            for kv, alts, text in prefixes:
                for pre in alts:
                    m = {v: h for v, h in zip(kv, hv)}
                    self.ub += 1
                    lits = []
                    for lit in pre:
                        for a in lit[2]:
                            for v in GDL.term_vars(a, set()):
                                m.setdefault(v, f"?_a{self.ub}_{v[1:]}")
                        lits.append((lit[0], lit[1], tuple(subst_term(a, m) for a in lit[2])))
                    bound: set = set()                          # the key variables the prefix binds through a relation that reads the state
                    for lit in lits:
                        if not d.static.get(lit[1], False):
                            for a in lit[2]:
                                GDL.term_vars(a, bound)
                    if not ((bound & set(hv)) - held) or entail_body(lits, 0, {h: h for h in hv}, list(c.body)):
                        made = None                             # this reader's keys are those the base clause itself makes (it binds them through the state already): nothing is gained for it
                        nothing_for = nothing_for or text
                        break
                    made.append(GDL.Clause(c.head, list(c.body) + lits, c.text.replace("\n", " ") + f" (asked by: {text})"))
                if made is None:
                    break
            out.append(x)
            if made is None:
                continue
            plain.add(id(c))
            for nc in made:
                out.append(JoinRule(nc, comp, (), None, self.est))
                guarded.add(id(nc))
        if not guarded:
            raise AskedRefuse("every base clause already holds the prefix of a reader, or no literal that reads the state binds a key variable of a read (a read whose key an earlier read of the walk alone binds is one), so the keys are those it makes: " + (nothing_for or "; ".join(t for _kv, _p, t in prefixes[:2])))
        return out, plain, guarded

    def asked_reads(self, x, wls: list, text: str, keyp: list, admitted, over) -> list:
        """The reads of the walk in the reader `x` (joined as it is), each with its key variables and the alternatives of its guard (module text, (17, widened)): -> [(key variables, [alternative: literals], reader's text)]; AskedRefuse naming the reader."""
        out = []
        for wl in wls:
            kv = [wl[2][p] for p in keyp]
            if any(not GDL.is_var(a) for a in kv) or len(set(kv)) != len(kv):
                raise AskedRefuse(f"a reader asks the relation with a key place that is no variable, or one variable twice: {text}")
            j = next(i for i, lit in enumerate(x.order) if lit is wl)
            pre = [g for lit in x.order[:j] if lit[0] == "pos" for g in [lit if admitted(lit[1]) else over(lit)] if g is not None]      # an earlier read of the walk stands as its static over-approximation (its own rows are not made when the base clauses run)
            alts = [pre]
            for lit in x.order[:j]:                             # the guard that a relation made later than the walk gives
                if lit[0] == "pos" and wl[1] != lit[1] and not admitted(lit[1]) and over(lit) is None and len(alts) <= 8:
                    got = self.asked_guards(lit[1], lit[2], set(kv), admitted, over, 0)
                    if got:
                        alts = [a + g for a in alts for g in got]
            covered: set = set()
            for lit in pre:
                for a in lit[2]:
                    GDL.term_vars(a, covered)
            alts = [a for a in alts if set(kv) & (covered | {v for lit in a for t in lit[2] for v in GDL.term_vars(t, set())})]
            if not alts:
                raise AskedRefuse(f"a reader asks the relation with no key variable bound by a literal before it (it asks it unbound): {text}")
            out.append((kv, alts, text))
        return out

    def asked_guards(self, rel: str, args, need: set, admitted, over, depth: int):
        """The guard that the relation `rel` (made later than the walk, not in a recursive set) gives to a read whose key variables `need` stand at some places of its literal `args` (module text, (17, widened)): -> a list of alternatives, one for each clause of `rel`, each a list of positive
        literals over relations made before the walk (the head variable at a place that holds a key variable taken as the literal's term there, every other variable named anew), or None where a clause gives none, where the relation is recursive, or beyond a depth of 3."""
        d = self.d
        if depth > 3 or rel in INPUT_RELS or rel in QUESTION_RELS or rel.startswith(("true.", "next.", "init.")):
            return None
        mine = [x for k2, (c2, _r) in enumerate(d.strata) if rel in c2 and len(c2) == 1 for x in self.rules[k2]]
        if not mine or any(x.rec_steps or x.head_rel != rel for x in mine):
            return None
        out: list = []
        for x in mine:
            c = x.clause
            hm: dict = {}
            for i, a in enumerate(args):
                h = c.head[1][i]
                if GDL.is_var(a) and a in need and GDL.is_var(h):
                    hm.setdefault(h, a)
            if not hm:
                return None
            self.ub += 1
            ren: dict = dict(hm)
            def rn(t):
                return subst_term(t, {v: ren.setdefault(v, f"?_g{self.ub}_{v[1:]}") for v in GDL.term_vars(t, set())})
            body = [(lit[0], lit[1], tuple(rn(a) for a in lit[2])) for lit in c.body if lit[0] == "pos"]
            reached: set = set(hm.values())
            taken: list = []
            grew = True
            while grew:
                grew = False
                for lit in body:
                    if lit in taken:
                        continue
                    lv = {v for a in lit[2] for v in GDL.term_vars(a, set())}
                    if lv & reached:
                        taken.append(lit)
                        reached |= lv
                        grew = True
            direct = [g for lit in taken for g in [lit if admitted(lit[1]) else over(lit)] if g is not None]
            subs = [[]]
            for lit in taken:
                if not admitted(lit[1]) and over(lit) is None and lit[1] != rel:
                    got = self.asked_guards(lit[1], lit[2], {v for a in lit[2] for v in GDL.term_vars(a, set())} & reached, admitted, over, depth + 1)
                    if got:
                        subs = [a + g for a in subs for g in got]
            alts = [direct + extra for extra in subs]
            if not any(alts):
                return None
            out += alts
            if len(out) > 8:
                return None
        return out

    # ---------------------------------------------------------------------------------------------- pass (18): a literal read from a word with tiles dropped

    def find_project(self):
        """Pass (18): the relations that are the projection of one relation (one clause, a literal of that relation, the head's variables some of the literal's), and every clause that reads that relation with places that nothing else uses, rewritten to read the projection (module text, (18));
        the clauses it leaves as they are, and why, in `Plan.project_refused`."""
        d = self.d
        projections: dict = {}                                  # relation -> [(the places it keeps, in its head's order, the projection, the clause that makes it)]
        defining: set = set()
        for k, (comp, _r) in enumerate(d.strata):
            if len(comp) != 1:
                continue
            p = next(iter(comp))
            rules = self.rules[k]
            if p not in self.pre_bits or d.static[p] or len(rules) != 1 or rules[0].rec_steps or len(rules[0].steps) != 1 or rules[0].steps[0][0] != "pos":
                continue
            x = rules[0]
            r, args = x.steps[0][1], x.steps[0][2]
            if r == p or r not in self.pre_bits or d.static[r] or any(a[0] != "v" for a in args) or len({a[1] for a in args}) != len(args):
                continue
            if any(h[0] != "v" for h in x.head) or len({h[1] for h in x.head}) != len(x.head):
                continue
            at = {a[1]: i for i, a in enumerate(args)}
            if any(h[1] not in at for h in x.head) or len(x.head) >= len(args):
                continue
            projections.setdefault(r, []).append((tuple(at[h[1]] for h in x.head), p, x.clause))
            defining.add(id(x.clause))
        if not projections:
            return
        for k, (comp, _r) in enumerate(d.strata):
            head_deps = None
            out = []
            for x in self.rules[k]:
                c = x.clause
                if id(c) in defining or not any(lit[0] == "pos" and lit[1] in projections for lit in c.body):
                    out.append(x)
                    continue
                counts: dict = {}
                for t in c.head[1]:
                    count_vars(t, counts)
                for lit in c.body:
                    for t in ((lit[1], lit[2]) if lit[0] == "distinct" else lit[2]):
                        count_vars(t, counts)
                body, replaced = [], []
                for lit in c.body:
                    got = None
                    if lit[0] == "pos" and lit[1] in projections:
                        need = {i for i, a in enumerate(lit[2]) if not (GDL.is_var(a) and counts.get(a, 0) == 1)}
                        if len(need) < len(lit[2]):
                            for kept, p, _pc in sorted(projections[lit[1]], key=lambda o: (len(o[0]), o[0])):
                                if need <= set(kept):
                                    if head_deps is None:
                                        head_deps = d.cone([c.head[0]]) - {c.head[0]}
                                    if p in comp or p not in head_deps:
                                        self.project_refused[c.text.replace("\n", " ")] = f"{p} is not made before this clause in every order of the strata (it is not among the relations the head reads)"
                                        continue
                                    got = ("pos", p, tuple(lit[2][i] for i in kept))
                                    replaced.append((lit[1], p))
                                    break
                    body.append(got or lit)
                if not replaced:
                    out.append(x)
                    continue
                nc = GDL.Clause(c.head, body, c.text.replace("\n", " ") + " (with a projection read)")
                self.projected.append((c.text.replace("\n", " "), replaced))
                for info in self.asked.values():                  # the clause of a walk that the pass asked wrote keeps its place among those run when everything is asked for or only what is
                    for which in ("plain", "guarded"):
                        if id(c) in info[which]:
                            info[which].discard(id(c))                  # (an id of a clause that is no longer written could name another clause)
                            info[which].add(id(nc))
                out.append(JoinRule(nc, comp, (), None, self.est))
            self.rules[k] = out

    def cone(self, wanted) -> set:
        """The relations the written rules need to make `wanted` (the dependency graph of the rules as written, which a pass that unfolds has changed)."""
        deps: dict = {}
        for k, (comp, _r) in enumerate(self.d.strata):
            reads = {st[1] for r in self.rules[k] for st in r.steps if st[0] != "distinct"}
            for h in comp:
                deps.setdefault(h, set()).update(self.support_reads.get(h, reads))
        reached: set = set()
        todo = [x for x in wanted if x in self.d.prog.nodes]
        while todo:
            r = todo.pop()
            if r in reached:
                continue
            reached.add(r)
            todo.extend(deps.get(r, ()))
        return reached

    # ---------------------------------------------------------------------------------------------- pass (3): what a turn does not change

    def copy_shape(self, rule, old_rel: str):
        """The condition of pass (3) on one clause (see the module text): -> {variable number: place} or None."""
        steps = rule.steps
        pos = [s for s in steps if s[0] == "pos"]
        if len(pos) != 1 or pos[0][1] != old_rel:
            return None
        head = rule.head
        if any(h[0] != "v" for h in head) or len({h[1] for h in head}) != len(head) or tuple(pos[0][2]) != tuple(head):
            return None
        index = {h[1]: i for i, h in enumerate(head)}
        for s in steps:
            if s[0] == "pos":
                continue
            terms = [s[1], s[2]] if s[0] == "distinct" else list(s[2])
            for t in terms:
                if not (t[0] == "c" or (t[0] == "v" and t[1] in index)):
                    return None
        return index

    def find_copy(self):
        d = self.d
        for k, (comp, _rules) in enumerate(d.strata):
            if len(comp) != 1:
                continue
            r = next(iter(comp))
            if not r.startswith("next."):
                continue
            found = []
            for rule in self.rules[k]:
                if rule.rec_steps:
                    continue
                ix = self.copy_shape(rule, "true." + r[len("next."):])
                if ix is not None:
                    found.append((rule, ix))
            if found:
                self.copy[r] = found

    def is_copy_rule(self, rule) -> bool:
        return any(rule is x[0] for found in self.copy.values() for x in found)

    # ---------------------------------------------------------------------------------------------- pass (2): answered on demand

    def written_rules(self, demanded: set, fresh: bool = True) -> list:
        """The rules that are made as joins over all the facts: every rule of a relation not demanded, joined so that no demanded relation is read whole first (a copy clause is kept: its negated literals are uses)."""
        out = []
        for k, (comp, _rules) in enumerate(self.d.strata):
            if comp & demanded:
                continue
            out += [JoinRule(x.clause, comp, demanded, est=self.est) for x in self.rules[k]] if fresh else self.rules[k]
        return out

    def compile_ask(self, rel: str, places: tuple, demanded: set) -> list:
        out = []
        k = next(k for k, (comp, _r) in enumerate(self.d.strata) if rel in comp)
        for c in [x.clause for x in self.rules[k]]:                # (the clauses as written: a view unfolded into this relation's clauses by pass (8) is read as it stands now)
            names: set = set()
            for j in places:
                GDL.term_vars(c.head[1][j], names)
            out.append((JoinRule(bind_clause(c, names), set(), demanded, est=self.est), c))
        return out

    def scans(self, rule) -> bool:
        """Whether a rule reads a relation of the state with no place bound, other than to ask that a row exist (a loop over every row of it)."""
        return any(st[0] == "pos" and not st[3] and not self.small(st[1]) and not only_exists(rule, k) for k, st in enumerate(rule.steps))

    def small(self, rel: str) -> bool:
        """Whether a loop over the whole relation is cheap whatever the state: it does not read the state (made once), or it is the turn (one row for each role), or a fluent whose sorts allow at most SMALL_SCAN rows."""
        d = self.d
        if d.static[rel] or rel == "does":
            return True
        sig = d.prog.signature(rel) if rel.startswith("true.") else None
        if sig is None or not all(s in d.prog.sorts and all(isinstance(m, str) for m in d.prog.sorts[s]) for s in sig):
            return False
        n = 1
        for s in sig:
            n *= len(d.prog.sorts[s])
        return n <= SMALL_SCAN

    def find_demand(self):
        d = self.d
        roots = set(QUESTION_RELS) | {r for r in d.nodes if r.startswith("next.")}
        cand: set = set()
        for k, (comp, _rules) in enumerate(d.strata):
            rules = self.rules[k]
            if len(comp) != 1:
                continue
            r = next(iter(comp))
            if d.static[r] or r in roots or r in INPUT_RELS or r.startswith("true.") or not rules or any(x.rec_steps for x in rules) or r in self.pre_bits or r in self.unfolded or r in d.flags:
                continue
            cand.add(r)
        demanded = set(cand)
        while True:
            noted: dict = {}                                   # (relation, places) -> set of modes
            roots: set = set()                                # the (relation, places) a rule that is made in full asks
            todo: list = []

            def note(u):
                rel, places, mode = u
                if (rel, places) not in noted:
                    noted[(rel, places)] = set()
                    todo.append((rel, places))
                noted[(rel, places)].add(mode)
            for rule in self.written_rules(demanded):
                for u in uses_of(rule, demanded):
                    note(u)
                    roots.add((u[0], u[1]))
            asks: dict = {}
            while todo:
                rel, places = todo.pop()
                asks[(rel, places)] = self.compile_ask(rel, places, demanded)
                for rule, _c in asks[(rel, places)]:
                    for u in uses_of(rule, demanded):
                        note(u)

            def chain_scans(key, path=()):
                """The clause that makes answering (relation, places) a loop over a whole relation of the state, itself or in what it asks in turn (None when none does)."""
                for rule, c in asks[key]:
                    if self.scans(rule) and not (key[1] == () and noted[key] == {"x"}):          # (a relation asked only whether a row exists, with no key, stops at its first row: a loop in its clause is cut short)
                        return path + (key, c.text.replace("\n", " "))
                    for (r2, p2, _m) in uses_of(rule, demanded):
                        got = chain_scans((r2, p2), path + (key,))
                        if got:
                            return got
                return None
            # a relation remains in full when a rule made in full reads it whole, or asks it in a way that is a loop over a whole relation (the blame is on the relation the made rule asks, not on what that asks in turn)
            bad = {}
            for (rel, places) in sorted(roots):
                if "e" in noted[(rel, places)] and places == ():
                    bad.setdefault(rel, ((rel, places), "read whole by a rule that is made in full"))
                else:
                    got = chain_scans((rel, places))
                    if got:
                        bad.setdefault(rel, got)
            if not bad:
                break
            # what is certain goes first (read whole); then, of the chains that defeat a relation, the relations whose chain runs through another relation that is blamed too (the others may be blamed only for the way the first was asked)
            whole = {rel for rel, why in bad.items() if why[1] == "read whole by a rule that is made in full"}
            outer = {rel for rel, why in bad.items() if any(x[0] in bad and x[0] != rel for x in why[:-1])}
            chosen = whole or outer or set(bad)
            for rel in chosen:
                self.refused.setdefault(rel, bad[rel])
            demanded -= chosen
        self.demanded = demanded
        for k, (comp, _rules) in enumerate(d.strata):
            if not (comp & demanded):
                self.rules[k] = [JoinRule(x.clause, comp, demanded, est=self.est) for x in self.rules[k]]
        self.ask_rules = asks
        prod = {(rel, places, m) for (rel, places), ms in noted.items() for m in ms}
        # the ways the tests ask: every demanded relation whole, and what that asks in turn
        extra: set = set()
        scanned: set = set()
        work = [(rel, ()) for rel in sorted(demanded)]
        extra |= {(rel, (), "e") for rel in demanded if (rel, (), "e") not in prod}
        while work:
            rel, places = work.pop()
            if (rel, places) in scanned:
                continue
            scanned.add((rel, places))
            if (rel, places) not in self.ask_rules:
                self.ask_rules[(rel, places)] = self.compile_ask(rel, places, demanded)
            for rule, _c in self.ask_rules[(rel, places)]:
                for (r2, p2, m2) in uses_of(rule, demanded):
                    if (r2, p2, m2) not in prod:
                        extra.add((r2, p2, m2))
                    work.append((r2, p2))
        for (rel, places, m) in extra:
            if (rel, places) not in self.ask_rules:
                self.ask_rules[(rel, places)] = self.compile_ask(rel, places, demanded)
        for rel, places, m in sorted(prod):
            self.modes.setdefault(rel, []).append((places, m))
        for rel, places, m in sorted(extra - prod):
            self.test_modes.setdefault(rel, []).append((places, m))
        for n, key in enumerate(sorted(prod | extra)):
            self.fname[key] = f"q{n}"

    # ---------------------------------------------------------------------------------------------- pass (4): the rows new in a round first

    def find_delta(self):
        """Each rule of a recursive set, once for each of its literals of the set, joined from the rows new in the last round (a literal that is the smallest relation in the semi-naive rule), the rest in the usual order."""
        for k, (comp, _rules) in enumerate(self.d.strata):
            if comp & self.demanded:
                continue
            for rule in self.rules[k]:
                if not rule.rec_steps:
                    continue
                lits = [x for x in rule.clause.body if x[0] == "pos" and x[1] in comp]
                got = []
                for lit in lits:
                    got.append((lit[1], JoinRule(rule.clause, comp, self.demanded, lit, est=self.est)))
                self.delta.setdefault(k, []).append((rule, got))

    # ---------------------------------------------------------------------------------------------- the index places, and pass (1)

    def index_places(self) -> dict:
        """{relation: sorted list of the argument-place tuples the written joins ask by} (the rules of the demanded relations, joined from their keys, count)."""
        d = self.d
        out: dict = {r: set() for r in d.nodes}
        rules = [x for x in self.written_rules(self.demanded, False) if not self.is_copy_rule(x)]
        for lst in self.ask_rules.values():
            rules += [x[0] for x in lst]
        for lst in self.delta.values():
            for _rule, got in lst:
                rules += [x[1] for x in got]
        for rule in rules:
            for st in rule.steps:
                if st[0] == "pos" and st[3] and st[1] not in self.demanded:
                    out[st[1]].add(tuple(st[3]))
        return {r: sorted(v) for r, v in out.items()}

    def find_dense(self):
        d = self.d
        sorts = d.prog.sorts
        for r in d.nodes:
            n = d.arity.get(r, 0)
            sig = d.prog.signature(r)
            if r in self.demanded or r in self.bits or n == 0 or sig is None or len(sig) != n:
                continue
            if not all(s in sorts and sorts[s] and all(isinstance(m, str) for m in sorts[s]) and len(sorts[s]) < 65535 for s in sig):
                continue
            size = [len(sorts[s]) for s in sig]
            space = 1
            for x in size:
                space *= x
            if space > DENSE_SPACE:
                continue
            ok = True
            for pl in self.places[r]:
                keys = 1
                for p in pl:
                    keys *= size[p]
                ok = ok and keys <= DENSE_KEYS
            if ok:
                self.dense[r] = tuple(sig)


    # ---------------------------------------------------------------------------------------------- pass (5): relations held as bits

    def recursive_bits(self) -> list:
        """The relations held as bits that are in a recursive set (each has the words new in the last round)."""
        return sorted(h for k, (comp, _r) in enumerate(self.d.strata) for h in comp if h in self.bits and any(x.rec_steps for x in self.rules[k]) and not (comp & self.demanded))

    def bit_candidates(self) -> dict:
        """The relations that may be held as bits before any clause is written: every place of a declared sort of constants, cut into tiles, the last tile of at most TILE_MAX members and the others of at most DENSE_KEYS combinations, not an input,
        not answered on demand, and read by a written clause (a relation only the questions read would be made as words to be turned into rows)."""
        d = self.d
        prog = d.prog
        units = self.units = find_units(d)
        reads: set = set()
        for k, (comp, _rules) in enumerate(d.strata):
            if comp & self.demanded:
                continue
            for rule in self.rules[k]:
                if self.is_copy_rule(rule):
                    continue
                for st in rule.steps:
                    if st[0] in ("pos", "neg"):
                        reads.add(st[1])
        cand: dict = {}
        for r in d.nodes:
            n = d.arity.get(r, 0)
            sig = prog.signature(r)
            if r in self.demanded or r in INPUT_RELS or n == 0 or sig is None or len(sig) != n:
                continue
            tiles = tile_relation(prog, r, sig, units)
            if tiles is None:
                continue
            if r not in reads:
                self.bits_refused[r] = "no written clause reads it: only the questions do, so words would be made to be turned into rows"
                continue
            info = BitInfo(r, sig, tiles)
            if info.tiles[info.canon].size > TILE_MAX or info.key_size(info.canon) > BIT_KEYS:
                self.bits_refused[r] = f"its last tile has {info.tiles[info.canon].size} members and the others {info.key_size(info.canon)} combinations"
                continue
            cand[r] = info
        return cand

    def undo_asked(self, rel: str):
        """A walk that is not held as bits in the end is made as it was written: the clauses with a prefix are taken out again (pass (17))."""
        info = self.asked.pop(rel, None)
        if info is not None:
            k = info["k"]
            self.rules[k] = [x for x in self.rules[k] if id(x.clause) not in info["guarded"]]
            self.asked_refused[rel] = "the relation is not held as bits"

    def find_bits(self):
        """Which relations are held as bits (see the module text): the candidates, then the refusals found by writing every clause with them and dropping, one round at a time, each relation a clause or an orientation defeats."""
        d = self.d
        cand = self.bit_candidates()
        self.sids = {s: n for n, s in enumerate(sorted({s for i in cand.values() for t in i.tiles for s in t.sorts}))}
        self.tids = {s: n for n, s in enumerate(sorted({t.sorts for i in cand.values() for t in i.tiles}))}
        self.bits = cand
        while True:
            bad: dict = {}
            for k, (comp, rules) in enumerate(d.strata):
                if comp & self.demanded:
                    continue
                inside = [h for h in comp if h in self.bits]
                if inside and len(inside) != len(comp) and any(x.rec_steps for x in self.rules[k]):
                    for h in inside:
                        bad.setdefault(h, "it is in a recursive set with a relation that is not held as bits: " + "; ".join(sorted(comp)))
            if not bad:
                self.start_choice = {}
                self.start_refused = {}
                self.start_base = None
                for i in self.bits.values():
                    i.orients = {i.canon: "the membership test"}
                self.bit_rec = self.recursive_bits()
                if "start" in self.passes:
                    self.start_off = True
                    emit_strata(lambda text: None, d, self, self.places, {})
                    self.start_off = False
                    self.start_base = {r: set(i.orients) for r, i in self.bits.items()}
                emit_strata(lambda text: None, d, self, self.places, bad)
                for i in self.bits.values():
                    for f, text in sorted(i.orients.items()):
                        if i.tiles[f].size > TILE_MAX or i.key_size(f) > BIT_KEYS:
                            bad.setdefault(i.rel, f"asked with tile {f} free ({i.tiles[f].size} members, and {i.key_size(f)} combinations of the others) by: {text}")
            if not bad:
                break
            for rel, why in bad.items():
                self.bits_refused.setdefault(rel, why)
                self.bits.pop(rel, None)
                self.undo_asked(rel)
        self.bit_rec = self.recursive_bits()


class ClauseWriter:
    """One clause as nested loops. `mode`: "full" (the clause of a relation made in full: its rows into `o`), "rows" or "exists" (a clause of a demanded relation joined from a key: its rows into `o`, or whether a row exists)."""

    def __init__(self, d: Description, places: dict, cid, rule, scc: set, plan: "Plan | None" = None, mode: str = "full", clause=None, key_places: tuple = ()):
        self.d, self.places, self.cid, self.rule, self.scc = d, places, cid, rule, scc
        self.plan, self.mode, self.clause, self.key_places = plan, mode, clause, key_places
        self.lines: list = []
        self.n = 0
        self.depth = 0
        self.loops = 0
        self.bound: set = set()
        self.quiet = False

    def out(self, s: str):
        self.lines.append("    " * (self.depth + 1) + s)

    def fresh(self) -> int:
        self.n += 1
        return self.n

    def fail(self) -> str:
        if self.loops:
            return "continue;"
        return "return false;" if self.mode == "exists" else "return;"

    def holder(self, rel: str) -> str:
        return "s" if self.d.static[rel] else "d"

    def is_bit(self, rel: str) -> bool:
        return bool(self.plan) and rel in self.plan.bits

    def sym(self, name: str) -> str:
        """A constant of the clause: its number; or the variable the key of a demanded relation bound."""
        if name.startswith(BOUND):
            return "b_" + ident(name[1:])
        return str(self.d.sym_id[name])

    def prologue(self, t, e: str, noted: set):
        """The key's value `e` at a place of the head, matched against the head's term (binding its variables as the key gives them)."""
        if isinstance(t, tuple):
            u = self.fresh()
            self.out(f"let Some((f{u}, a{u})) = tm.compound({e}) else {{ {self.fail()} }};")
            self.out(f"if f{u} != {self.d.sym_id[t[0]]} || a{u}.len() != {len(t) - 1} {{ {self.fail()} }}")
            if len(t) > 1:
                self.out(f"let ({', '.join(f'x{u}_{j}' for j in range(len(t) - 1))},) = ({', '.join(f'a{u}[{j}]' for j in range(len(t) - 1))},);")
            for j, q in enumerate(t[1:]):
                self.prologue(q, f"x{u}_{j}", noted)
        elif GDL.is_var(t):
            name = "b_" + ident(t[1:])
            if t in noted:
                self.out(f"if {e} != {name} {{ {self.fail()} }}")
            else:
                self.out(f"let {name} = {e};")
                noted.add(t)
        else:
            self.out(f"if {e} != {self.d.sym_id[t]} {{ {self.fail()} }}")

    def build(self, p) -> str:
        if p[0] == "c":
            return self.sym(p[1])
        if p[0] == "v":
            if p[1] not in self.bound:
                raise Untranslatable(f"a variable is read before it is bound: {self.rule.clause.text}")
            return f"v{p[1]}"
        subs = [self.build(q) for q in p[2]]
        u = self.fresh()
        self.out(f"let t{u} = tm.fun({self.d.sym_id[p[1]]}, &[{', '.join(subs)}]);")
        return f"t{u}"

    def match(self, p, e: str):
        if p[0] == "c":
            self.out(f"if {e} != {self.sym(p[1])} {{ {self.fail()} }}")
        elif p[0] == "v":
            if p[1] in self.bound:
                self.out(f"if {e} != v{p[1]} {{ {self.fail()} }}")
            else:
                self.out(f"let v{p[1]} = {e};")
                self.bound.add(p[1])
        else:
            u = self.fresh()
            n = len(p[2])
            self.out(f"let Some((f{u}, a{u})) = tm.compound({e}) else {{ {self.fail()} }};")
            self.out(f"if f{u} != {self.d.sym_id[p[1]]} || a{u}.len() != {n} {{ {self.fail()} }}")
            if n:
                self.out(f"let ({', '.join(f'x{u}_{j}' for j in range(n))},) = ({', '.join(f'a{u}[{j}]' for j in range(n))},);")
            for j, q in enumerate(p[2]):
                self.match(q, f"x{u}_{j}")

    def steps(self, k: int):
        r = self.rule
        counting = self.mode != "exists" and not self.quiet
        demanded = self.plan.demanded if self.plan else ()
        if k == len(r.steps):
            if self.mode == "exists":
                self.out("return true;")
                return
            row = ", ".join(self.build(h) for h in r.head)
            self.out(f"o.rows.push([{row}]);")
            return
        st = r.steps[k]
        if st[0] == "distinct":
            a, b = self.build(st[1]), self.build(st[2])
            self.out(f"if {a} == {b} {{ {self.fail()} }}")
            self.steps(k + 1)
        elif st[0] == "neg":
            args = [self.build(a) for a in st[2]]
            if st[1] in demanded:
                fn = self.plan.fname[(st[1], tuple(range(len(st[2]))), "x")]
                self.out(f"if {fn}(s, d, tm, &[{', '.join(args)}]) {{ {self.fail()} }}")
            else:
                self.out(f"if {self.holder(st[1])}.{self.d.rid[st[1]]}.has(&[{', '.join(args)}]) {{ {self.fail()} }}")
            self.steps(k + 1)
        elif st[1] in demanded:
            _, rel, args, keypos = st
            n = len(args)
            kv = {i: self.build(args[i]) for i in keypos}
            self.out(f"let key{k}: [u32; {n}] = [" + ", ".join(kv.get(i, "FREE") for i in range(n)) + "];")
            if only_exists(r, k):
                self.out(f"if !{self.plan.fname[(rel, tuple(keypos), 'x')]}(s, d, tm, &key{k}) {{ {self.fail()} }}")
                self.steps(k + 1)
                return
            fn = self.plan.fname[(rel, tuple(keypos), "e")]
            self.out(f"let mut oo{k} = Out::<{n}>::new();")
            self.out(f"{fn}(s, d, tm, &key{k}, &mut oo{k});")
            self.out(f"let rows{k}: &[[u32; {n}]] = &oo{k}.rows;")
            self.out(f"for row{k} in rows{k}.iter() {{")
            self.depth += 1
            self.loops += 1
            if counting:
                self.out("o.visits += 1;")
            saved = set(self.bound)
            for i, a in enumerate(args):
                self.match(a, f"row{k}[{i}]")
            self.steps(k + 1)
            self.bound = saved
            self.loops -= 1
            self.depth -= 1
            self.out("}")
        else:
            _, rel, args, keypos = st
            n = len(args)
            rid = self.d.rid[rel]
            src = self.holder(rel)
            if keypos:
                kv = {i: self.build(args[i]) for i in keypos}
                key = "[" + ", ".join(kv.get(i, "FREE") for i in range(n)) + "]"
                self.out(f"let key{k}: [u32; {n}] = {key};")
                ix = self.places[rel].index(tuple(keypos))
                get = f"{'&' if self.is_bit(rel) else ''}{src}.{rid}.get({ix}, &key{k})"
                kp = "&[" + ", ".join(str(i) for i in keypos) + "]"
            else:
                get = f"&{src}.{rid}.rows_all()" if self.is_bit(rel) else f"&{src}.{rid}.order"
                kp = "&[]"
                self.out(f"let key{k}: [u32; {n}] = [FREE; {n}];")
            if counting:
                self.out("o.probes += 1;")
            if self.mode == "delta" and k == 0:
                if keypos:
                    self.out(f"let tmp{k};")
                    self.out(f"let rows{k}: &[[u32; {n}]] = {{ tmp{k} = scan(&dl.{rid}, {kp}, &key{k}); &tmp{k} }};")
                else:
                    self.out(f"let rows{k}: &[[u32; {n}]] = &dl.{rid};")
            elif k in r.rec_steps and self.mode != "delta":
                self.out(f"let tmp{k};")
                self.out(f"let rows{k}: &[[u32; {n}]] = if di == {k} {{ tmp{k} = scan(&dl.{rid}, {kp}, &key{k}); &tmp{k} }} else {{ {get} }};")
            elif self.is_bit(rel) and self.plan is not None and "words" in self.plan.passes:
                pass                                            # pass (22): the rows of a relation held as bits are walked by an iterator over its words, no list is made
            else:
                self.out(f"let rows{k}: &[[u32; {n}]] = {get};")
            self.out(f"for row{k} in {src}.{rid}.iter(&key{k}) {{" if (self.is_bit(rel) and self.plan is not None and "words" in self.plan.passes and not (self.mode == "delta" and k == 0) and not (k in r.rec_steps and self.mode != "delta")) else f"for row{k} in rows{k}.iter() {{")
            self.depth += 1
            self.loops += 1
            if counting:
                self.out("o.visits += 1;")
            saved = set(self.bound)
            for i, a in enumerate(args):
                self.match(a, f"row{k}[{i}]")
            self.steps(k + 1)
            self.bound = saved
            self.loops -= 1
            self.depth -= 1
            self.out("}")

    def write(self) -> str:
        r = self.rule
        n = len(r.head)
        text = r.clause.text.replace("\n", " ")
        self.lines.append(f"// {text}")
        if self.mode == "full":
            self.lines.append(f"fn c{self.cid}(s: &St, d: &Dy, tm: &mut Terms, di: usize, dl: &Dl, o: &mut Out<{n}>) {{")
        elif self.mode == "delta":
            self.lines.append(f"fn {self.cid}(s: &St, d: &Dy, tm: &mut Terms, dl: &Dl, o: &mut Out<{n}>) {{")
        elif self.mode == "rows":
            self.lines.append(f"fn {self.cid}(s: &St, d: &Dy, tm: &mut Terms, key: &[u32; {n}], o: &mut Out<{n}>) {{")
        else:
            self.lines.append(f"fn {self.cid}(s: &St, d: &Dy, tm: &mut Terms, key: &[u32; {n}]) -> bool {{")
        if self.mode != "full":
            noted: set = set()
            for j in self.key_places:
                self.prologue(self.clause.head[1][j], f"key[{j}]", noted)
        self.steps(0)
        if self.mode == "exists":
            self.out("false")
        self.lines.append("}")
        return "\n".join(self.lines)


# ------------------------------------------------------------------------------------------------------------------------------------------ pass (5): the form of a relation over small sorts

NONE = 65535                                            # the number of "not a member" in a table of symbols to ordinals
TILE_MAX = 64                                           # a tile is held as the bits of one word
BIT_KEYS = 1 << 20                                      # a relation held as bits has at most this many words for any orientation (8 MB; the words never set are never made resident)
RUNTIME_BITS = r'''
/// A set of words (pass "bits"): one word for each combination of the key tiles of an orientation of a relation; `nz` lists the words that are not nought, so that a scan and a clearing cost what the relation holds.
pub struct Bw { pub w: Vec<u64>, pub nz: Vec<u32> }
impl Bw {
    pub fn new(n: usize) -> Bw { Bw { w: vec![0u64; n], nz: Vec::new() } }
    /// Set bit `b` of word `i`; true when it was not set.
    #[inline]
    pub fn set(&mut self, i: usize, b: usize) -> bool { let x = self.w[i]; let m = 1u64 << b; if x & m != 0 { return false; } if x == 0 { self.nz.push(i as u32); } self.w[i] = x | m; true }
    /// Or `v` into word `i`; the bits that were not set.
    #[inline]
    pub fn or(&mut self, i: usize, v: u64) -> u64 { let x = self.w[i]; let n = v & !x; if n != 0 { if x == 0 { self.nz.push(i as u32); } self.w[i] = x | n; } n }
    pub fn clear(&mut self) { for &i in self.nz.iter() { self.w[i as usize] = 0; } self.nz.clear(); }
}
/// The ordinal of a symbol in a sort (NONE when it is no member of the sort, or is a compound).
pub const NONE: usize = 65535;
#[inline]
pub fn ordof(t: &[u16], e: u32) -> usize { match t.get(e as usize) { Some(&o) => o as usize, None => NONE } }
'''


class BitRefuse(Exception):
    """A relation that the pass bits cannot hold soundly, with the reason (the clause is named in it)."""

    def __init__(self, rel: str, why: str):
        super().__init__(why)
        self.rel, self.why = rel, why


class Tile:
    """One or two places of a relation held as one number: a sort of constants, or a product of two (the first place's ordinal times the second sort's size plus the second's)."""

    def __init__(self, places: tuple, sorts: tuple, sizes: tuple):
        self.places, self.sorts, self.sizes = places, sorts, sizes
        self.size = 1
        for x in sizes:
            self.size *= x


class BitInfo:
    """A relation held as bits: its tiles, and the orientations asked (the index of the tile left free by a way a clause asks it; the last tile is always held)."""

    def __init__(self, rel: str, sig: tuple, tiles: list):
        self.rel, self.sig, self.tiles = rel, tuple(sig), tiles
        self.n, self.m = len(sig), len(tiles)
        self.canon = self.m - 1
        self.orients: dict = {self.canon: "the membership test"}          # orientation -> the clause that first asked it

    def request(self, f: int, text: str):
        self.orients.setdefault(f, text)

    def key_tiles(self, f: int) -> list:
        return [i for i in range(self.m) if i != f]

    def key_size(self, f: int) -> int:
        n = 1
        for i in self.key_tiles(f):
            n *= self.tiles[i].size
        return n


def sort_ok(prog, s: str) -> bool:
    ms = prog.sorts.get(s)
    return bool(ms) and all(isinstance(m, str) for m in ms) and len(ms) < NONE


def find_units(d: "Description") -> dict:
    """The products of two sorts, found from the clauses and signatures, each with its weight: in one clause the same two distinct variables stand next to each other, in that order, at places of the sorts (A, B) in at least two literals
    (head included), and the product is at most TILE_MAX; the weight is the number of those literals, summed. Of (A, B) and (B, A) the one of more weight is kept."""
    prog = d.prog
    weight: dict = {}
    for rel, clauses in prog.rules_of.items():
        for c in clauses:
            lits = [(c.head[0], c.head[1])] + [(x[1], x[2]) for x in c.body if x[0] in ("pos", "neg")]
            pairs: dict = {}
            for li, (r, args) in enumerate(lits):
                sig = prog.signature(r)
                if sig is None or len(sig) != len(args):
                    continue
                for i in range(len(args) - 1):
                    a, b = args[i], args[i + 1]
                    if GDL.is_var(a) and GDL.is_var(b) and a != b and sig[i] != sig[i + 1] and sort_ok(prog, sig[i]) and sort_ok(prog, sig[i + 1]):
                        pairs.setdefault((a, b, sig[i], sig[i + 1]), set()).add(li)
            for (a, b, sa, sb), ls in pairs.items():
                if len(ls) >= 2 and len(prog.sorts[sa]) * len(prog.sorts[sb]) <= TILE_MAX:
                    weight[(sa, sb)] = weight.get((sa, sb), 0) + len(ls)
    return {u: w for u, w in weight.items() if w > weight.get((u[1], u[0]), 0)}


def tile_relation(prog, rel: str, sig: tuple, units: dict):
    """The cut of a relation's places into tiles that has the most weight of products (the cut with fewer tiles where equal); None when a place's sort is not of constants."""
    if not all(sort_ok(prog, s) for s in sig):
        return None
    n = len(sig)
    best = {n: (0, 0, [])}                                  # from place i on: (weight, -tiles, cut)
    for i in range(n - 1, -1, -1):
        opts = []
        w1, t1, c1 = best[i + 1]
        opts.append((w1, t1 - 1, [(i,)] + c1))
        if i + 1 < n and (sig[i], sig[i + 1]) in units:
            w2, t2, c2 = best[i + 2]
            opts.append((w2 + units[(sig[i], sig[i + 1])], t2 - 1, [(i, i + 1)] + c2))
        best[i] = max(opts, key=lambda o: (o[0], o[1]))
    return [Tile(pl, tuple(sig[p] for p in pl), tuple(len(prog.sorts[sig[p]]) for p in pl)) for pl in best[0][2]]


class BitWriter(ClauseWriter):
    """One clause as nested loops over what is not yet bound, over relations held as bits (pass bits; see the module text). Modes: "full" and "delta" (the functions of ClauseWriter: the rows of a head that is not held as bits into `o`), and
    "bfull" and "bdelta" (a head held as bits: its bits are set in the head's relation; "bdelta" reads the first literal from the words new in the last round of the recursive set). A literal of a relation not held as bits is written as ClauseWriter does."""

    def __init__(self, d, places, cid, rule, scc, plan, mode="full", clause=None, key_places=(), record=False, alias=None):
        super().__init__(d, places, cid, rule, scc, plan, mode, clause, key_places)
        self.alias = alias                                             # pass (15): the Rust place of a set of words that stands for the head's relation (the start words of a walk made as its ends); only the last tile's orientation
        self.bi = plan.bits
        self.done: set = set()
        self.record = record
        self.head_bit = mode in ("bfull", "bdelta")
        self.quiet = self.head_bit
        self.cut = "cut" in plan.passes and self.head_bit            # pass (11): a block left at the first row once the head is complete
        self.cut_label = None
        self.cut_loops = 0
        self.prof: list = []                                           # pass (13): for each literal read by a loop, the Rust text of how many iterations the loop is estimated to make (the number of words held, the combinations, or 1)
        self.head_vars: set = set()
        for h in rule.head:
            self.head_vars |= comp_vars(h)
        self.text = rule.clause.text.replace("\n", " ")
        vs: dict = {}

        def note(rel, args):
            info = self.bi.get(rel)
            if info is not None:
                for i, a in enumerate(args):
                    if a[0] == "v":
                        vs.setdefault(a[1], set()).add(info.sig[i])
        for st in rule.steps:
            if st[0] in ("pos", "neg"):
                note(st[1], st[2])
        note(rule.head_rel, rule.head)
        self.vsort = {n: next(iter(s)) for n, s in vs.items() if len(s) == 1}

    # ---------------------------------------------------------------------------------------------- the pieces of a clause

    def fail(self) -> str:
        if self.loops:
            if self.cut_label is not None and self.loops == self.cut_loops:
                return f"break {self.cut_label};"                  # a `continue` may not leave a labeled block: the block is the rest of the loop's body
            return "continue;"
        if self.head_bit:
            return "return made;"
        return "return false;" if self.mode == "exists" else "return;"

    def sid(self, s: str) -> int:
        return self.plan.sids[s]

    def is_ord(self, n: int, sort: str) -> bool:
        return self.vsort.get(n) == sort

    def build(self, p) -> str:
        if p[0] == "v" and p[1] in self.vsort:
            if p[1] not in self.bound:
                raise Untranslatable(f"a variable is read before it is bound: {self.rule.clause.text}")
            return f"SS{self.sid(self.vsort[p[1]])}[v{p[1]}]"
        return super().build(p)

    def match(self, p, e: str):
        if p[0] == "v" and p[1] in self.vsort:
            n, si = p[1], self.sid(self.vsort[p[1]])
            if n in self.bound:
                self.out(f"if ordof(&OS{si}, {e}) != v{n} {{ {self.fail()} }}")
            else:
                self.out(f"let v{n} = ordof(&OS{si}, {e});")
                self.out(f"if v{n} == NONE {{ {self.fail()} }}")
                self.bound.add(n)
        else:
            super().match(p, e)

    def const_ord(self, rel: str, name: str, sort: str) -> int:
        ms = self.d.prog.sorts[sort]
        if name not in ms:
            raise BitRefuse(rel, f"the constant {name} is no member of the sort {sort} at its place: {self.text}")
        return ms.index(name)

    def var_ord(self, n: int, sort: str) -> str:
        """The ordinal, in `sort`, of the bound variable n (a number, and a test that it is a member where the variable is held as a symbol)."""
        if self.is_ord(n, sort):
            return f"v{n}"
        u = self.fresh()
        self.out(f"let o{u} = ordof(&OS{self.sid(sort)}, v{n});")
        self.out(f"if o{u} == NONE {{ {self.fail()} }}")
        return f"o{u}"

    def term_ord(self, rel: str, a, sort: str) -> str:
        if a[0] == "c":
            return str(self.const_ord(rel, a[1], sort))
        if a[0] == "v":
            if a[1] not in self.bound:
                raise Untranslatable(f"a variable is read before it is bound: {self.rule.clause.text}")
            return self.var_ord(a[1], sort)
        raise BitRefuse(rel, f"a compound term at a place of a relation held as bits: {self.text}")

    def tile_expr(self, rel: str, tile: Tile, args) -> str:
        comps = [self.term_ord(rel, args[p], tile.sorts[c]) for c, p in enumerate(tile.places)]
        if len(comps) == 1:
            return comps[0]
        return f"({comps[0]}) * {tile.sizes[1]} + {comps[1]}"

    def index_of(self, info: BitInfo, f: int, tiles: dict) -> str:
        """The number of the word of orientation f for the tiles' numbers (expressions) given."""
        kt = info.key_tiles(f)
        if not kt:
            return "0"
        acc = tiles[kt[0]]
        for i in kt[1:]:
            acc = f"({acc}) * {info.tiles[i].size} + {tiles[i]}"
        return acc

    def member(self, info: BitInfo, rel: str, args) -> str:
        """A Rust test that the row is in the relation (all its places bound)."""
        h, rid = self.holder(rel), self.d.rid[rel]
        if any(a[0] == "v" and not self.is_ord(a[1], info.sig[p]) for p, a in enumerate(args)):
            return f"{h}.{rid}.has(&[{', '.join(self.build(a) for a in args)}])"
        tiles = {i: self.tile_expr(rel, t, args) for i, t in enumerate(info.tiles)}
        c = info.canon
        return f"(({h}.{rid}.o{c}.w[{self.index_of(info, c, tiles)}] >> ({tiles[c]})) & 1) == 1"

    def mask(self, tile: Tile, c: int, e: str) -> str:
        if len(tile.places) == 1:
            return f"(1u64 << ({e}))"
        return f"MK{self.plan.tids[tile.sorts]}_{c}[{e}]"

    def comp_of(self, tile: Tile, c: int, b: str) -> str:
        """The component c of the tile's number b."""
        if len(tile.places) == 1:
            return b
        return f"{b} / {tile.sizes[1]}" if c == 0 else f"{b} % {tile.sizes[1]}"

    def bind(self, n: int, sort: str, e: str):
        """Bind variable n to the ordinal e (a number) in `sort`."""
        if self.is_ord(n, sort):
            self.out(f"let v{n} = {e};")
        else:
            self.out(f"let v{n} = SS{self.sid(sort)}[{e}];")
        self.bound.add(n)

    def bind_or_check(self, rel: str, a, sort: str, e: str):
        if a[0] == "c":
            self.out(f"if {e} != {self.const_ord(rel, a[1], sort)} {{ {self.fail()} }}")
        elif a[1] in self.bound:
            if self.is_ord(a[1], sort):
                self.out(f"if v{a[1]} != {e} {{ {self.fail()} }}")
            else:
                self.out(f"if v{a[1]} != SS{self.sid(sort)}[{e}] {{ {self.fail()} }}")
        else:
            self.bind(a[1], sort, e)

    def fits(self, info: BitInfo, f: int) -> bool:
        return info.tiles[f].size <= TILE_MAX and info.key_size(f) <= BIT_KEYS

    def open(self, header: str):
        self.out(header)
        self.depth += 1
        self.loops += 1

    def close(self, n: int):
        for _ in range(n):
            self.loops -= 1
            self.depth -= 1
            self.out("}")

    # ---------------------------------------------------------------------------------------------- the steps

    def steps(self, k: int):
        r = self.rule
        while k in self.done:
            k += 1
        if k == len(r.steps):
            self.finish()
            return
        if self.cut and self.cut_label is None and self.head_vars <= self.bound and any(j not in self.done and r.steps[j][0] == "pos" for j in range(k, len(r.steps))):
            u = self.fresh()
            self.cut_label = f"'h{u}"
            self.cut_loops = self.loops
            self.out(f"{self.cut_label}: {{")
            self.depth += 1
            self.steps(k)
            self.depth -= 1
            self.out("}")
            self.cut_label = None
            return
        st = r.steps[k]
        if st[0] == "distinct":
            a, b = st[1], st[2]
            if a[0] == "c" and b[0] == "v":
                a, b = b, a
            if a[0] == "v" and a[1] in self.vsort and b[0] == "v" and self.vsort.get(b[1]) == self.vsort[a[1]] and a[1] in self.bound and b[1] in self.bound:
                self.out(f"if v{a[1]} == v{b[1]} {{ {self.fail()} }}")
            elif a[0] == "v" and a[1] in self.vsort and b[0] == "c" and a[1] in self.bound:
                ms = self.d.prog.sorts[self.vsort[a[1]]]
                if b[1] in ms:
                    self.out(f"if v{a[1]} == {ms.index(b[1])} {{ {self.fail()} }}")
            else:
                self.out(f"if {self.build(st[1])} == {self.build(st[2])} {{ {self.fail()} }}")
            self.steps(k + 1)
            return
        if st[0] in ("pos", "neg") and st[1] in self.bi:
            self.bit_literal(k)
            return
        super().steps(k)

    def finish(self):
        r = self.rule
        if not self.head_bit:
            super().steps(len(r.steps))
            return
        info = self.bi[r.head_rel]
        rid = self.d.rid[r.head_rel]
        h = self.holder(r.head_rel)
        tiles = [self.tile_expr(r.head_rel, t, r.head) for t in info.tiles]
        u = self.fresh()
        self.out(f"let tt{u}: [usize; {info.m}] = [{', '.join(tiles)}];")
        rec = ""
        if self.alias:
            ix = self.index_of(info, info.canon, {i: f"tt{u}[{i}]" for i in range(info.m)})
            self.out(f"if {self.alias}.set({ix}, tt{u}[{info.canon}]) {{ made += 1; }}")
            if self.cut_label is not None:
                self.out(f"break {self.cut_label};")
            return
        if self.record and r.head_rel in self.scc:
            ix = self.index_of(info, info.canon, {i: f"tt{u}[{i}]" for i in range(info.m)})
            rec = f" dw.nw_{rid}.set({ix}, tt{u}[{info.canon}]);"
        self.out(f"if {h}.{rid}.set_t(tt{u}) {{ made += 1;{rec} }}")
        if self.cut_label is not None:
            self.out(f"break {self.cut_label};")

    def fold(self, k: int, ftile: Tile, fargs, vvars: set, wname: str) -> bool:
        """Literals later than k over the same variables of the same sort of tile as the free tile of literal k are anded into its word (and-not when negated); -> whether any was."""
        r = self.rule
        any_ = False
        for j in range(k + 1, len(r.steps)):
            if j in self.done:
                continue
            sj = r.steps[j]
            if sj[0] not in ("pos", "neg") or sj[1] not in self.bi:
                continue
            im, args = self.bi[sj[1]], sj[2]
            if any(a[0] == "f" for a in args):
                continue
            unb = {a[1] for a in args if a[0] == "v" and a[1] not in self.bound}
            if not unb or not unb <= vvars:
                continue
            hit = [i for i, t in enumerate(im.tiles) if any(args[p][0] == "v" and args[p][1] in unb for p in t.places)]
            if len(hit) != 1:
                continue
            fm = hit[0]
            if not self.fits(im, fm):
                continue
            tm_ = im.tiles[fm]
            if tm_.sorts != ftile.sorts or tuple(args[p] for p in tm_.places) != tuple(fargs[p] for p in ftile.places):
                continue
            if any(a[0] == "v" and a[1] not in self.bound and i not in tm_.places for i, a in enumerate(args)):
                continue
            im.request(fm, self.text)
            tiles = {i: self.tile_expr(sj[1], t, args) for i, t in enumerate(im.tiles) if i != fm}
            wm = f"{self.holder(sj[1])}.{self.d.rid[sj[1]]}.o{fm}.w[{self.index_of(im, fm, tiles)}]"
            self.out(f"{wname} &= {'!' if sj[0] == 'neg' else ''}{wm};")
            self.done.add(j)
            any_ = True
        return any_

    def or_word_ok(self, k: int, info: BitInfo, f: int, args, vvars: set) -> bool:
        r = self.rule
        if not self.head_bit:
            return False
        if any(j not in self.done for j in range(k + 1, len(r.steps))):
            return False
        hi = self.bi[r.head_rel]
        ht, ft = hi.tiles[hi.canon], info.tiles[f]
        if ht.sorts != ft.sorts or tuple(r.head[p] for p in ht.places) != tuple(args[p] for p in ft.places):
            return False
        hv: set = set()
        for t in hi.tiles[:-1]:
            for p in t.places:
                hv |= comp_vars(r.head[p])
        return not (vvars & hv) and hv <= self.bound

    def decode_key(self, u: int, kt: list, info: BitInfo, rel: str, args):
        """The tiles of the key of the word `ix{u}` (the number of a word of an orientation), bound to the variables at their places or tested against the ones bound."""
        if not kt:
            return
        self.out(f"let mut rm{u} = ix{u};")
        names = {}
        for i in reversed(kt):
            names[i] = f"t{u}_{i}"
            last = i == kt[0]
            self.out(f"let {names[i]} = rm{u} % {info.tiles[i].size};" if not last else f"let {names[i]} = rm{u};")
            if not last:
                self.out(f"rm{u} /= {info.tiles[i].size};")
        for i in kt:
            t = info.tiles[i]
            for c, p in enumerate(t.places):
                self.bind_or_check(rel, args[p], t.sorts[c], self.comp_of(t, c, names[i]))

    def bit_literal(self, k: int):
        r = self.rule
        st = r.steps[k]
        neg = st[0] == "neg"
        rel, args = st[1], st[2]
        info = self.bi[rel]
        h, rid = self.holder(rel), self.d.rid[rel]
        for a in args:
            if a[0] == "f":
                raise BitRefuse(rel, f"a compound term at a place of a relation held as bits: {self.text}")
        unb = lambda a: a[0] == "v" and a[1] not in self.bound
        if neg:
            self.out(f"if {self.member(info, rel, args)} {{ {self.fail()} }}")
            self.steps(k + 1)
            return
        free = [i for i, t in enumerate(info.tiles) if any(unb(args[p]) for p in t.places)]
        if not free:
            self.out(f"if !({self.member(info, rel, args)}) {{ {self.fail()} }}")
            self.steps(k + 1)
            return
        delta = self.mode == "bdelta" and k == 0
        f = info.canon if delta else free[-1]
        if f != info.canon and not self.fits(info, f):
            f = info.canon                                   # that way of asking would need too many words: the tiles not bound are looped over and the last tile is tested
        info.request(f, self.text)
        saved = set(self.bound)
        opened = 0
        u = self.fresh()
        kt = info.key_tiles(f)
        w = f"w{u}"
        all_unbound = bool(kt) and all(unb(args[p]) for i in kt for p in info.tiles[i].places)
        slots = [(i, c, p) for i in kt for c, p in enumerate(info.tiles[i].places) if unb(args[p])]
        scan_ok = "scan" in self.plan.passes and bool(kt) and not delta and not all_unbound and bool(slots) and len({args[p][1] for _i, _c, p in slots}) == len(slots)
        factor = "1"
        if delta or all_unbound:
            src = f"dw.dl_{rid}" if delta else f"{h}.{rid}.o{f}"
            factor = f"{src}.nz.len()"
            self.open(f"for ti{u} in 0..{src}.nz.len() {{")
            opened += 1
            self.out(f"let ix{u} = {src}.nz[ti{u}] as usize;")
            self.out(f"let mut {w} = {src}.w[ix{u}];")
            self.decode_key(u, kt, info, rel, args)
        elif scan_ok:
            # pass (10): the tiles not bound are looped over as combinations, or the words the relation holds are looped over and tested at the tiles that are bound, whichever is fewer
            vals = {}
            for i in kt:
                t = info.tiles[i]
                row = []
                for c, p in enumerate(t.places):
                    a = args[p]
                    if a[0] == "c":
                        row.append(str(self.const_ord(rel, a[1], t.sorts[c])))
                    elif a[1] in self.bound:
                        row.append(self.var_ord(a[1], t.sorts[c]))
                    else:
                        row.append(None)
                vals[i] = row
            nc = " * ".join(str(info.tiles[i].sizes[c]) for i, c, _p in slots)
            factor = f"({h}.{rid}.o{f}.nz.len()).min({nc})"
            self.out(f"let nc{u}: usize = {nc};")
            self.out(f"let sc{u} = {h}.{rid}.o{f}.nz.len() < nc{u};")
            self.open(f"for ti{u} in 0..(if sc{u} {{ {h}.{rid}.o{f}.nz.len() }} else {{ nc{u} }}) {{")
            opened += 1
            self.out(f"let ix{u}: usize = if sc{u} {{ {h}.{rid}.o{f}.nz[ti{u}] as usize }} else {{")
            self.depth += 1
            self.out(f"let mut cm{u} = ti{u};")
            names = {}
            for j, (i, c, _p) in enumerate(reversed(slots)):
                nm = f"c{u}_{i}_{c}"
                names[(i, c)] = nm
                if j < len(slots) - 1:
                    self.out(f"let {nm} = cm{u} % {info.tiles[i].sizes[c]}; cm{u} /= {info.tiles[i].sizes[c]};")
                else:
                    self.out(f"let {nm} = cm{u};")
            tiles = {}
            for i in kt:
                t = info.tiles[i]
                comps = [names[(i, c)] if e is None else e for c, e in enumerate(vals[i])]
                tiles[i] = comps[0] if len(comps) == 1 else f"({comps[0]}) * {t.sizes[1]} + {comps[1]}"
            self.out(self.index_of(info, f, tiles))
            self.depth -= 1
            self.out("};")
            self.out(f"let mut {w} = {h}.{rid}.o{f}.w[ix{u}];")
            self.decode_key(u, kt, info, rel, args)
        else:
            tiles = {}
            for i in kt:
                t = info.tiles[i]
                comps = []
                for c, p in enumerate(t.places):
                    a = args[p]
                    if a[0] == "c":
                        comps.append(str(self.const_ord(rel, a[1], t.sorts[c])))
                    elif a[1] in self.bound:
                        comps.append(self.var_ord(a[1], t.sorts[c]))
                    else:
                        n = a[1]
                        factor = f"{t.sizes[c]}" if factor == "1" else f"{factor} * {t.sizes[c]}"
                        if self.is_ord(n, t.sorts[c]):
                            self.open(f"for v{n} in 0..{t.sizes[c]}usize {{")
                            comps.append(f"v{n}")
                        else:
                            u2 = self.fresh()
                            self.open(f"for o{u2} in 0..{t.sizes[c]}usize {{")
                            self.out(f"let v{n} = SS{self.sid(t.sorts[c])}[o{u2}];")
                            comps.append(f"o{u2}")
                        opened += 1
                        self.bound.add(n)
                tiles[i] = comps[0] if len(comps) == 1 else f"({comps[0]}) * {t.sizes[1]} + {comps[1]}"
            self.out(f"let mut {w} = {h}.{rid}.o{f}.w[{self.index_of(info, f, tiles)}];")
        # the free tile: masks for the places already bound or constants
        ft = info.tiles[f]
        vlist = []
        masked = False
        for c, p in enumerate(ft.places):
            a = args[p]
            if a[0] == "c":
                self.out(f"{w} &= {self.mask(ft, c, str(self.const_ord(rel, a[1], ft.sorts[c])))};")
                masked = True
            elif a[1] in self.bound:
                self.out(f"{w} &= {self.mask(ft, c, self.var_ord(a[1], ft.sorts[c]))};")
                masked = True
            else:
                if any(a[1] == x[1] for x in vlist):
                    raise BitRefuse(rel, f"one variable at two places of one tile: {self.text}")
                vlist.append((c, a[1]))
        vvars = {n for _, n in vlist}
        folded = self.fold(k, ft, args, vvars, w) if vlist else False
        by_word = bool(vlist) and self.or_word_ok(k, info, f, args, vvars)
        if by_word:
            self.prof.append(f"({factor})")                              # (pass (13), the estimate) the word is or-ed into the head whole: no loop over its bits
        elif vlist and not kt:
            self.prof.append(f"({h}.{rid}.o{f}.w[0].count_ones() as usize)")      # the one word of a relation with no key: its bits are the loop
        else:
            self.prof.append(f"({factor}) * 2" if vlist else f"({factor})")
        if masked or folded or delta or opened:
            self.out(f"if {w} == 0 {{ {self.fail()} }}")
        if by_word:
            hi = self.bi[r.head_rel]
            hrid = self.d.rid[r.head_rel]
            hh = self.holder(r.head_rel)
            tiles = {i: self.tile_expr(r.head_rel, t, r.head) for i, t in enumerate(hi.tiles) if i != hi.canon}
            self.out(f"let hk{u} = {self.index_of(hi, hi.canon, tiles)};")
            self.out(f"let nb{u} = {self.alias}.or(hk{u}, {w});" if self.alias else f"let nb{u} = {hh}.{hrid}.or_word(hk{u}, {w});")
            rec = f" dw.nw_{hrid}.or(hk{u}, nb{u});" if (self.record and r.head_rel in self.scc) else ""
            self.out(f"if nb{u} != 0 {{ made += nb{u}.count_ones() as u64;{rec} }}")
        else:
            used: set = set()
            for j in range(k + 1, len(r.steps)):
                if j in self.done:
                    continue
                s2 = r.steps[j]
                if s2[0] == "distinct":
                    used |= comp_vars(s2[1]) | comp_vars(s2[2])
                else:
                    for a2 in s2[2]:
                        used |= comp_vars(a2)
            for a2 in r.head:
                used |= comp_vars(a2)
            if vlist and (vvars & used):
                u3 = self.fresh()
                self.out(f"let mut m{u3} = {w};")
                self.open(f"while m{u3} != 0 {{")
                opened += 1
                self.out(f"let b{u3} = m{u3}.trailing_zeros() as usize;")
                self.out(f"m{u3} &= m{u3} - 1;")
                for c, n in vlist:
                    self.bind(n, ft.sorts[c], self.comp_of(ft, c, f"b{u3}"))
                self.steps(k + 1)
            else:
                if vlist and not masked and not folded and not delta and not opened:
                    self.out(f"if {w} == 0 {{ {self.fail()} }}")
                self.steps(k + 1)
        self.close(opened)
        self.bound = saved

    def write(self) -> str:
        if not self.head_bit:
            return super().write()
        r = self.rule
        self.lines.append(f"// {self.text}")
        if self.d.static[r.head_rel]:
            self.lines.append(f"fn {self.cid}(s: &mut St, d: &Dy, tm: &mut Terms, dw: &mut Dw) -> u64 {{")
        else:
            self.lines.append(f"fn {self.cid}(s: &St, d: &mut Dy, tm: &mut Terms, dw: &mut Dw) -> u64 {{")
        self.out("let mut made = 0u64;")
        self.steps(0)
        self.out("made")
        self.lines.append("}")
        return "\n".join(self.lines)


# ------------------------------------------------------------------------------------------------------------------------------------------ pass (9): walks by whole words

SHIFT_MAX = 8                                           # a onward with more displacements than this for one key is no shift


class ShiftRefuse(Exception):
    """A recursive set that the pass shift cannot compile, with the reason (the clause or the relation is named in it)."""


def tile_roles(info: BitInfo, args, xs: tuple, ys: tuple, keys: set, text: str) -> list:
    """For each tile of a literal's relation: "X" (its places are the variables of the free tile of the recursive literal, in order), "Y" (those of the head's free tile), "K" (variables of the key and constants only); anything else refuses."""
    out = []
    for t in info.tiles:
        here = tuple(args[p][1] if args[p][0] == "v" else None for p in t.places)
        if here == xs:
            out.append("X")
        elif here == ys:
            out.append("Y")
        elif all(a is None or a in keys for a in here):
            out.append("K")
        else:
            raise ShiftRefuse(f"a tile of {info.rel} that mixes the free tile's variables with others: {text}")
    return out


def shift_table(d: "Description", plan: "Plan", rel: str, info: BitInfo, fin: int, fout: int, text: str):
    """The displacements of the onward relation `rel` (read from its rows at translation, which do not read the state): for each combination of the tiles that are neither the one a member is read at (`fin`) nor the next one (`fout`), the list of
    (displacement of the tile's number, mask of the members that have a next one at that displacement). It is TESTED against every row: the rows made again from the masks and displacements are the rows, exactly (else the walk is refused naming the relation)."""
    key = (rel, fin, fout)
    if key in plan.shift_tables:
        return plan.shift_tables[key]
    prog = d.prog
    rows = prog.static[rel].facts if rel in prog.static else None
    if rows is None or not d.static[rel]:
        raise ShiftRefuse(f"{rel} reads the state, its rows are not known at translation: {text}")
    sel = [i for i in range(info.m) if i not in (fin, fout)]
    nsel = 1
    for i in sel:
        nsel *= info.tiles[i].size

    def tile_val(row, t: Tile) -> int:
        cs = [prog.sorts[t.sorts[c]].index(row[p]) for c, p in enumerate(t.places)]
        return cs[0] if len(cs) == 1 else cs[0] * t.sizes[1] + cs[1]
    by: list = [dict() for _ in range(nsel)]
    want = set()
    for row in rows:
        tv = [tile_val(row, t) for t in info.tiles]
        s = 0
        for i in sel:
            s = s * info.tiles[i].size + tv[i]
        dl = tv[fout] - tv[fin]
        by[s][dl] = by[s].get(dl, 0) | (1 << tv[fin])
        want.add(tuple(tv))
    for s, dd in enumerate(by):
        if len(dd) > SHIFT_MAX:
            raise ShiftRefuse(f"{rel} has {len(dd)} displacements for the key {s} (at most {SHIFT_MAX}): {text}")
    got = set()
    for s, dd in enumerate(by):
        rest, vals = s, {}
        for i in reversed(sel):
            vals[i] = rest % info.tiles[i].size
            rest //= info.tiles[i].size
        for dl, m in dd.items():
            for b in range(info.tiles[fin].size):
                if (m >> b) & 1:
                    tv = [vals.get(i, 0) for i in range(info.m)]
                    tv[fin], tv[fout] = b, b + dl
                    got.add(tuple(tv))
    if got != want:
        raise ShiftRefuse(f"the rows of {rel} are not displacements of the free tile's number with masks of the members: {len(want ^ got)} rows differ, e.g. {sorted(want ^ got)[0]}: {text}")
    plan.shift_tables[key] = (len(plan.shift_tables), by, nsel, sel)
    return plan.shift_tables[key]


def shift_shape(d: "Description", plan: "Plan", comp: set, rule, places) -> dict:
    """The recursive clause of a walk, taken to pieces (module text, (9)): the relation of the set, its key places and free tile, the literals that stop a walk (words anded into the frontier), and the one literal that takes a member to the next ones (its relation, tiles and displacement table);
    ShiftRefuse, naming the clause, where the set is not that shape."""
    bits = plan.bits
    text = rule.clause.text.replace("\n", " ")
    if len(comp) != 1:
        raise ShiftRefuse(f"the set has {len(comp)} relations: {text}")
    h = next(iter(comp))
    info = bits[h]
    if len(rule.rec_steps) != 1:
        raise ShiftRefuse(f"the clause reads its own set {len(rule.rec_steps)} times: {text}")
    ri = rule.rec_steps[0]
    largs, hargs = rule.steps[ri][2], rule.head
    if any(a[0] != "v" for a in largs) or any(a[0] != "v" for a in hargs):
        raise ShiftRefuse(f"a place of the set that is not a variable: {text}")
    ct = info.tiles[info.canon]
    keyp = [p for i, t in enumerate(info.tiles) if i != info.canon for p in t.places]
    kv = [largs[p][1] for p in keyp]
    if kv != [hargs[p][1] for p in keyp] or len(set(kv)) != len(kv):
        raise ShiftRefuse(f"the key places of the head are not those of the literal of the set, one variable each: {text}")
    xs, ys = tuple(largs[p][1] for p in ct.places), tuple(hargs[p][1] for p in ct.places)
    if len(set(xs)) != len(xs) or len(set(ys)) != len(ys) or set(xs) & set(ys) or (set(xs) | set(ys)) & set(kv):
        raise ShiftRefuse(f"the free tile's variables of the head and of the literal are not distinct from each other and from the key: {text}")
    keys = set(kv)
    stops, onward = [], None
    for j, st in enumerate(rule.steps):
        if j == ri:
            continue
        if st[0] == "distinct":
            raise ShiftRefuse(f"a distinct test among the literals: {text}")
        rel, args = st[1], st[2]
        if rel not in bits:
            raise ShiftRefuse(f"{rel} is not held as bits: {text}")
        if any(a[0] == "f" for a in args):
            raise ShiftRefuse(f"a compound term at a place of {rel}: {text}")
        im = bits[rel]
        vs = {a[1] for a in args if a[0] == "v"}
        if not vs <= keys | set(xs) | set(ys):
            raise ShiftRefuse(f"a variable of {rel} that is neither of the key nor of the free tiles: {text}")
        roles = tile_roles(im, args, xs, ys, keys, text)
        if "Y" in roles:
            if onward is not None or st[0] != "pos" or roles.count("Y") != 1 or roles.count("X") != 1:
                raise ShiftRefuse(f"{rel} is not the one positive literal that takes a member to the next ones: {text}")
            fin, fout = roles.index("X"), roles.index("Y")
            if im.tiles[fin].sorts != ct.sorts or im.tiles[fout].sorts != ct.sorts:
                raise ShiftRefuse(f"the tiles of {rel} are not of the sorts of the free tile: {text}")
            onward = (rel, args, im, fin, fout)
        else:
            if roles.count("X") != 1 or im.tiles[roles.index("X")].sorts != ct.sorts:
                raise ShiftRefuse(f"{rel} is not a test of the members of the free tile (one tile of its sorts): {text}")
            fm = roles.index("X")
            if im.tiles[fm].size > TILE_MAX or im.key_size(fm) > BIT_KEYS:
                raise ShiftRefuse(f"{rel} asked with tile {fm} free needs too many words: {text}")
            stops.append((st[0] == "neg", rel, args, im, fm))
    if onward is None:
        raise ShiftRefuse(f"no literal takes a member to the next ones: {text}")
    return {"h": h, "info": info, "kv": kv, "keyp": keyp, "largs": largs, "stops": stops, "onward": onward, "text": text}


class EndsRefuse(Exception):
    """A walk that the pass ends cannot make as its ends, with the reason (the reader clause is named in it)."""


def unify_term(a, b, sig: dict) -> bool:
    """`a`, a term of a clause of the relation read for its stop (its variables are its own), made equal to `b`, a term of the reader (taken as it stands); extends `sig`."""
    if GDL.is_var(a):
        if a in sig:
            return sig[a] == b
        sig[a] = b
        return True
    if isinstance(a, tuple):
        return isinstance(b, tuple) and a[0] == b[0] and len(a) == len(b) and all(unify_term(x, y, sig) for x, y in zip(a[1:], b[1:]))
    return a == b


def entail_body(body: list, i: int, sig: dict, lits: list) -> bool:
    """Whether every literal of `body` from the i-th on, under `sig` (extended as needed), is a literal of `lits` of the same sign (a homomorphism of the body into the literals)."""
    if i == len(body):
        return True
    lit = body[i]
    if lit[0] == "distinct":
        for x in lits:
            if x[0] != "distinct":
                continue
            for p, q in ((x[1], x[2]), (x[2], x[1])):
                s2 = dict(sig)
                if unify_term(lit[1], p, s2) and unify_term(lit[2], q, s2) and entail_body(body, i + 1, s2, lits):
                    return True
        return False
    for x in lits:
        if x[0] != lit[0] or x[1] != lit[1] or len(x[2]) != len(lit[2]):
            continue
        s2 = dict(sig)
        if all(unify_term(a, b, s2) for a, b in zip(lit[2], x[2])) and entail_body(body, i + 1, s2, lits):
            return True
    return False


def entails_stop(plan: "Plan", zrel: str, target: list, reader, wlit) -> bool:
    """Whether the literals of the reader clause other than `wlit` entail `zrel(target)`: a clause of `zrel` (as written) whose head unifies with the target and whose body maps into those literals (module text, (15))."""
    kz = next(k for k, (c, _r) in enumerate(plan.d.strata) if zrel in c)
    lits = [x for x in reader.body if x is not wlit]
    for zr in plan.rules_unread[kz]:
        zc = zr.clause
        if zc.head[0] != zrel or len(zc.head[1]) != len(target):
            continue
        sig: dict = {}
        if all(unify_term(a, b, sig) for a, b in zip(zc.head[1], target)) and entail_body(list(zc.body), 0, sig, lits):
            return True
    return False


def find_ends(d: "Description", plan: "Plan", comp: set, rec_rule, shape: dict):
    """Pass (15): the stop literal (a negated literal of the walk's recursive clause) that every clause reading the walk entails at the member it reads; EndsRefuse, naming the clause, where there is none."""
    h = shape["h"]
    info = plan.bits[h]
    if h in QUESTION_RELS or h in INPUT_RELS or h.startswith(("next.", "init.")):
        raise EndsRefuse(f"{h} is read by a question or is a state or an input")
    rc = rec_rule.clause
    reclit = next(x for x in rc.body if x[0] == "pos" and x[1] in comp)
    keyp, ct = shape["keyp"], info.tiles[info.canon]
    kvars = [reclit[2][p] for p in keyp]
    xvars = [reclit[2][p] for p in ct.places]
    readers = []
    for k2, (c2, _r) in enumerate(d.strata):
        if c2 == comp:
            continue
        for r2 in plan.rules[k2]:
            pos = [x for x in r2.clause.body if x[0] == "pos" and x[1] == h]
            if any(x[0] == "neg" and x[1] == h for x in r2.clause.body):
                raise EndsRefuse(f"read negated: {r2.clause.text.replace(chr(10), ' ')}")
            if len(pos) > 1:
                raise EndsRefuse(f"read twice by one clause: {r2.clause.text.replace(chr(10), ' ')}")
            if pos:
                readers.append((r2.clause, pos[0]))
    if not readers:
        raise EndsRefuse("no clause reads it")
    why = []
    for z in [x for x in rc.body if x[0] == "neg"]:
        zrel, zargs = z[1], z[2]
        if any(isinstance(a, tuple) or (GDL.is_var(a) and a not in kvars and a not in xvars) for a in zargs):
            why.append(f"the stop literal over {zrel} has a place that is not a key variable, a free tile variable or a constant")
            continue
        failed = None
        for clause, wl in readers:
            text = clause.text.replace("\n", " ")
            if any(not GDL.is_var(a) for a in wl[2]):
                failed = f"a place of the walk is read with a term that is no variable: {text}"
                break
            theta = {v: wl[2][p] for v, p in zip(kvars, keyp)}
            theta.update({v: wl[2][p] for v, p in zip(xvars, ct.places)})
            if not entails_stop(plan, zrel, [theta.get(a, a) for a in zargs], clause, wl):
                failed = f"the reader does not entail a stop at {zrel}: {text}"
                break
        if failed is None:
            return z
        why.append(failed)
    raise EndsRefuse("; ".join(why) or "the recursive clause has no negated stop literal")


RUNTIME_ENDS = r'''
/// Displace the bits of `x` by `n` places (to the higher numbers when n is positive); nothing is left when the displacement is a word or more.
#[inline]
pub fn shw(x: u64, n: i32) -> u64 { if n >= 0 { if n >= 64 { 0 } else { x << n } } else if n <= -64 { 0 } else { x >> (-n) } }
'''


def write_shift(w, d, plan, places, name: str, comp, rule, shape: dict, ends: bool = False):
    """The function of a walk compiled by whole words (module text, (9)): for each key word of the set, the closure of the walk from the words the base clauses made, by shifts; one function, no rounds over lists of words."""
    bw = BitWriter(d, places, name, rule, comp, plan, "bfull", None, (), False)
    info, h, text = shape["info"], shape["h"], shape["text"]
    rid = d.rid[h]
    rel, sargs, sim, fin, fout = shape["onward"]
    tab_id, by, nsel, sel = shift_table(d, plan, rel, sim, fin, fout, text)
    bw.lines.append(f"// {text}")
    bw.lines.append(f"fn {name}(s: &St, d: &mut Dy, tm: &mut Terms, dw: &mut Dw) -> u64 {{")
    bw.out("let mut made = 0u64;")
    u = bw.fresh()
    src = f"dw.sb_{rid}" if ends else f"d.{rid}.o{info.canon}"
    bw.open(f"for ti{u} in 0..{src}.nz.len() {{")
    bw.out(f"let ix{u} = {src}.nz[ti{u}] as usize;")
    bw.out(f"let mut fr{u} = {src}.w[ix{u}];")
    bw.out(f"let mut acc{u} = fr{u};")
    kt = info.key_tiles(info.canon)
    bw.decode_key(u, kt, info, h, shape["largs"])
    bw.out(f"let mut live{u} = u64::MAX;")
    for neg, srel, sargs2, sim2, fm in shape["stops"]:
        tiles = {i: bw.tile_expr(srel, t, sargs2) for i, t in enumerate(sim2.tiles) if i != fm}
        sim2.request(fm, bw.text)
        bw.out(f"live{u} &= {'!' if neg else ''}{bw.holder(srel)}.{d.rid[srel]}.o{fm}.w[{bw.index_of(sim2, fm, tiles)}];")
    acc_sel = "0"
    for n, i in enumerate(sel):
        e = bw.tile_expr(rel, sim.tiles[i], sargs)
        acc_sel = e if n == 0 else f"({acc_sel}) * {sim.tiles[i].size} + {e}"
    bw.out(f"let tab{u} = SH{tab_id}[{acc_sel}];")
    chains = None
    if plan.tables:
        try:
            if nsel * info.tiles[info.canon].size > TABLE_WORDS:
                raise ShiftRefuse(f"the table would hold {nsel * info.tiles[info.canon].size} words (at most {TABLE_WORDS})")
            chains, _inc = chain_table(by, info.tiles[info.canon].size)
            plan.tables_found[h] = (nsel, nsel * info.tiles[info.canon].size)
            plan.tables_refused.pop(h, None)
        except ShiftRefuse as e:
            plan.tables_refused[h] = str(e)
            plan.tables_found.pop(h, None)
    if chains is not None:
        # pass (23): the chain of each member as a word; the walk from a member stops at the first member the stop literals shut (that member included), found as the lowest or highest bit of the shut members on the chain
        plan.chain_used[tab_id] = (chains, nsel, info.tiles[info.canon].size)
        bw.out(f"if let Some(&(dl, _)) = tab{u}.first() {{")
        bw.depth += 1
        bw.out(f"let chains{u} = &CH{tab_id}[{acc_sel}];")
        bw.out(f"let mut f{u} = fr{u};")
        bw.out(f"while f{u} != 0 {{")
        bw.depth += 1
        bw.out(f"let b{u} = f{u}.trailing_zeros() as usize; f{u} &= f{u} - 1;")
        bw.out(f"let ch{u} = chains{u}[b{u}];")
        bw.out(f"let blk{u} = ch{u} & !live{u};")
        bw.out(f"acc{u} |= if blk{u} == 0 {{ ch{u} }} else if dl > 0 {{ let low = blk{u} & blk{u}.wrapping_neg(); ch{u} & (low | (low - 1)) }} else {{ let hi = 1u64 << (63 - blk{u}.leading_zeros()); ch{u} & !(hi - 1) }};")
        bw.depth -= 1
        bw.out("}")
        bw.depth -= 1
        bw.out("}")
    elif ends and all(len(dd) <= 1 for dd in by):
        # pass (15): the doubling form (generate and propagate), a constant number of rounds
        rounds = max(1, (info.tiles[info.canon].size - 1).bit_length())
        bw.out(f"if let Some(&(dl, m)) = tab{u}.first() {{")
        bw.depth += 1
        bw.out(f"let mut q = shw(live{u} & m, dl);")
        bw.out("let mut s = dl;")
        bw.out(f"for _ in 0..{rounds} {{ acc{u} |= q & shw(acc{u}, s); q &= shw(q, s); s = s.saturating_mul(2); }}")
        bw.depth -= 1
        bw.out("}")
    else:
        bw.out("loop {")
        bw.depth += 1
        bw.out(f"let x{u} = fr{u} & live{u};")
        bw.out(f"if x{u} == 0 {{ break; }}")
        bw.out(f"let mut nx{u} = 0u64;")
        bw.out(f"for &(dl, m) in tab{u}.iter() {{ let y = x{u} & m; nx{u} |= if dl >= 0 {{ y << dl }} else {{ y >> (-dl) }}; }}")
        bw.out(f"nx{u} &= !acc{u};")
        bw.out(f"if nx{u} == 0 {{ break; }}")
        bw.out(f"acc{u} |= nx{u};")
        bw.out(f"fr{u} = nx{u};")
        bw.depth -= 1
        bw.out("}")
    if ends:
        bw.out(f"let out{u} = if dw.full {{ acc{u} }} else {{ acc{u} & !live{u} }};")
        bw.out(f"let nb{u} = d.{rid}.or_word(ix{u}, out{u});")
    else:
        bw.out(f"let nb{u} = d.{rid}.or_word(ix{u}, acc{u});")
    bw.out(f"if nb{u} != 0 {{ made += nb{u}.count_ones() as u64; }}")
    bw.close(1)
    if ends:
        bw.out(f"dw.sb_{rid}.clear();")
    bw.out("made")
    bw.lines.append("}")
    plan.shift_used.add(key_of_table(rel, fin, fout))
    w("\n".join(bw.lines))


def chain_table(by: list, size: int):
    """Pass (23): from the displacements of a walk's onward relation (`by`: for each key, {displacement: mask of the members that have a next one}) the CHAINS: for each key and member, the set of members reached by following the onward link again and again, the member included, as a word.
    Returns (chains, inc) or raises ShiftRefuse naming why not: a key with two displacements (the line branches), a chain that returns to a member it has left (a cycle), or members that do not fit a word."""
    if size > TILE_MAX:
        raise ShiftRefuse(f"the free tile has {size} members, more than a word")
    chains, inc = [], []
    for s, dd in enumerate(by):
        if len(dd) > 1:
            raise ShiftRefuse(f"the key {s} has {len(dd)} displacements: the line branches")
        row = [1 << b for b in range(size)]
        if dd:
            (dl, m), = dd.items()
            for b in range(size):
                cur, word = b, 1 << b
                while (m >> cur) & 1:
                    cur += dl
                    if cur < 0 or cur >= size:
                        raise ShiftRefuse(f"the key {s}: a member with a next one whose next is outside the tile")
                    if (word >> cur) & 1:
                        raise ShiftRefuse(f"the key {s}: the chain from member {b} returns to member {cur}")
                    word |= 1 << cur
                row[b] = word
        chains.append(row)
        inc.append(next(iter(dd)) > 0 if dd else True)
    return chains, inc


def key_of_table(rel: str, fin: int, fout: int) -> tuple:
    return (rel, fin, fout)


def w_shift_tables(w, plan):
    """The tables of the pass shift that the document uses: for each combination of the onward's other tiles, its (displacement, mask) pairs."""
    for key in sorted(plan.shift_used):
        tid, by, nsel, sel = plan.shift_tables[key]
        rows = ", ".join("&[" + ", ".join(f"({dl}, 0x{m:x}u64)" for dl, m in sorted(dd.items())) + "]" for dd in by)
        w(f"static SH{tid}: [&[(i32, u64)]; {nsel}] = [{rows}];")
        if tid in plan.chain_used:
            chains, nk, size = plan.chain_used[tid]
            w(f"static CH{tid}: [[u64; {size}]; {nk}] = [{', '.join('[' + ', '.join('0x%xu64' % x for x in row) + ']' for row in chains)}];")


# ------------------------------------------------------------------------------------------------------------------------------------------ what the passes add to the document

RUNTIME_DENSE = r'''
/// A relation of N arguments over small sorts of constants (pass "dense"): the rows in the order made, a table of bits over the tuples of the sorts (membership by position, no hash), and for each set of places the joins
/// ask by a bucket for each combination of the places' ordinals. A symbol's ordinal in a sort is read from a table; a symbol outside the sort is no member (and may not be added).
pub struct Dense<const N: usize> {
    pub order: Vec<[u32; N]>,
    bits: Vec<u64>,
    ord: &'static [&'static [u16]],
    radix: &'static [usize],
    kp: &'static [&'static [usize]],
    ix: Vec<Vec<Vec<[u32; N]>>>,
}
impl<const N: usize> Dense<N> {
    pub fn new(ord: &'static [&'static [u16]], radix: &'static [usize], kp: &'static [&'static [usize]]) -> Dense<N> {
        let space: usize = radix.iter().product();
        let ix = kp.iter().map(|k| { let n: usize = k.iter().map(|&p| radix[p]).product(); vec![Vec::new(); n] }).collect();
        Dense { order: vec![], bits: vec![0u64; (space + 63) / 64], ord, radix, kp, ix }
    }
    #[inline]
    fn slot(&self, p: usize, v: u32) -> Option<usize> { match self.ord[p].get(v as usize) { Some(&o) if o != u16::MAX => Some(o as usize), _ => None } }
    #[inline]
    fn pos(&self, r: &[u32; N]) -> Option<usize> { let mut x = 0usize; for p in 0..N { x = x * self.radix[p] + self.slot(p, r[p])?; } Some(x) }
    #[inline]
    fn bucket(&self, i: usize, key: &[u32; N]) -> Option<usize> { let mut x = 0usize; for &p in self.kp[i].iter() { x = x * self.radix[p] + self.slot(p, key[p])?; } Some(x) }
    #[inline]
    pub fn has(&self, r: &[u32; N]) -> bool { match self.pos(r) { Some(p) => (self.bits[p >> 6] >> (p & 63)) & 1 == 1, None => false } }
    /// Add a row; true when it was not there.
    pub fn add(&mut self, r: [u32; N]) -> bool {
        let p = self.pos(&r).expect("a row outside the sorts of its relation");
        let (w, m) = (p >> 6, 1u64 << (p & 63));
        if self.bits[w] & m != 0 { return false; }
        self.bits[w] |= m;
        self.order.push(r);
        for i in 0..self.kp.len() { let b = self.bucket(i, &r).unwrap(); self.ix[i][b].push(r); }
        true
    }
    /// The rows whose places of the i-th index equal `key`'s.
    #[inline]
    pub fn get(&self, i: usize, key: &[u32; N]) -> &[[u32; N]] { match self.bucket(i, key) { Some(b) => self.ix[i][b].as_slice(), None => &[] } }
    /// Clear it by its own rows (the cost is that of the rows it holds).
    pub fn clear(&mut self) {
        let mut rows = std::mem::take(&mut self.order);
        for r in rows.iter() {
            let p = self.pos(r).unwrap();
            self.bits[p >> 6] &= !(1u64 << (p & 63));
            for i in 0..self.kp.len() { let b = self.bucket(i, r).unwrap(); self.ix[i][b].clear(); }
        }
        rows.clear();
        self.order = rows;
    }
}
'''

RUNTIME_WORDS = r'''
/// A state's fluent held as bits (pass 22): `W` words, bit i set when the fact whose places' ordinals, in the mixed radix of the sorts, number i is true.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Wd<const W: usize>(pub [u64; W]);
impl<const W: usize> Default for Wd<W> { fn default() -> Self { Wd([0u64; W]) } }
/// How the facts of a fluent are numbered: for each place the ordinal of every symbol in the sort (65535: not a member), the radix of each place, and the row of each number (a table, so that a number is turned into its row by one read).
pub struct Tab<const N: usize> { pub ord: &'static [&'static [u16]], pub rad: &'static [usize], pub dec: &'static [[u32; N]] }
/// The number of a row, None when a place holds what is not a member of its sort.
#[inline]
pub fn widx<const N: usize>(t: &Tab<N>, r: &[u32; N]) -> Option<usize> {
    let mut i = 0usize;
    for p in 0..N {
        let o = *t.ord[p].get(r[p] as usize)?;
        if o == 65535 { return None; }
        i = i * t.rad[p] + o as usize;
    }
    Some(i)
}
impl<const W: usize> Wd<W> {
    #[inline] pub fn set(&mut self, i: usize) { self.0[i >> 6] |= 1u64 << (i & 63); }
    #[inline] pub fn unset(&mut self, i: usize) { self.0[i >> 6] &= !(1u64 << (i & 63)); }
    #[inline] pub fn has(&self, i: usize) -> bool { (self.0[i >> 6] >> (i & 63)) & 1 == 1 }
    pub fn rows<const N: usize>(&self, t: &'static Tab<N>) -> WdRows<'_, W, N> { WdRows { w: &self.0, t, k: 0, cur: self.0[0] } }
}
/// The true facts of a fluent held as bits, in the order of their numbers; no memory is made.
pub struct WdRows<'a, const W: usize, const N: usize> { w: &'a [u64; W], t: &'static Tab<N>, k: usize, cur: u64 }
impl<'a, const W: usize, const N: usize> Iterator for WdRows<'a, W, N> {
    type Item = [u32; N];
    #[inline]
    fn next(&mut self) -> Option<[u32; N]> {
        while self.cur == 0 { self.k += 1; if self.k >= W { return None; } self.cur = self.w[self.k]; }
        let b = self.cur.trailing_zeros() as usize;
        self.cur &= self.cur - 1;
        Some(self.t.dec[self.k * 64 + b])
    }
}
'''

RUNTIME_REL_KEEP_INDEX = r'''
impl<const N: usize> Rel<N> {
    /// Clear it, keeping the lists of its indexes (pass 22): the rows made again at the next question are put in lists that are there.
    pub fn clear(&mut self) { self.order.clear(); self.set.clear(); for m in self.ix.iter_mut() { for v in m.values_mut() { v.clear(); } } }
}
'''

RUNTIME_REL_CLEAR = r'''
impl<const N: usize> Rel<N> {
    /// Clear it (the hash relations of a state evaluation that is kept).
    pub fn clear(&mut self) { self.order.clear(); self.set.clear(); for m in self.ix.iter_mut() { m.clear(); } }
}
'''


def copy_condition(d: Description, plan: Plan, rel: str, st_name: str, d_name: str, args: str, row: str) -> str:
    """The Rust condition under which an old row `row` of a fluent is kept by a copy clause of `rel` (pass 3): the guards of that clause, over the row alone."""
    alts = []
    for rule, index in plan.copy[rel]:
        def term(t):
            return str(d.sym_id[t[1]]) if t[0] == "c" else f"{row}[{index[t[1]]}]"
        parts = []
        for s in rule.steps:
            if s[0] == "pos":
                continue
            if s[0] == "distinct":
                parts.append(f"{term(s[1])} != {term(s[2])}")
            else:
                a = ", ".join(term(x) for x in s[2])
                if s[1] in plan.demanded:
                    parts.append(f"!{plan.fname[(s[1], tuple(range(len(s[2]))), 'x')]}({args}, &[{a}])")
                else:
                    parts.append(f"!{st_name if d.static[s[1]] else d_name}.{d.rid[s[1]]}.has(&[{a}])")
        alts.append("(" + (" && ".join(parts) or "true") + ")")
    return " || ".join(alts)


def w_asks(w, d: Description, plan: Plan):
    """The functions that answer the demanded relations: one for each way they are asked (relation, places bound, mode), each clause of the relation joined from the key."""
    for (rel, places, mode), fn in sorted(plan.fname.items(), key=lambda kv: int(kv[1][1:])):
        n = d.arity.get(rel, 0)
        w(f"// {fn}: {rel} answered for a key at the places {list(places)}, {'whether a row exists' if mode == 'x' else 'its rows'}")
        names = []
        for j, (rule, c) in enumerate(plan.ask_rules[(rel, places)]):
            cname = f"{fn}c{j}"
            w(ClauseWriter(d, plan.places, cname, rule, set(), plan, "exists" if mode == "x" else "rows", c, places).write())
            names.append(cname)
        if mode == "x":
            w(f"fn {fn}(s: &St, d: &Dy, tm: &mut Terms, key: &[u32; {n}]) -> bool {{ " + " || ".join(f"{c}(s, d, tm, key)" for c in names) + " }")
        else:
            w(f"fn {fn}(s: &St, d: &Dy, tm: &mut Terms, key: &[u32; {n}], o: &mut Out<{n}>) {{ " + " ".join(f"{c}(s, d, tm, key, o);" for c in names) + " }")


def w_api(w, d: Description, plan: Plan, pooled: bool):
    """The items of a module with passes that the tests hold it to the plain module by (inside `impl Game`)."""
    items = ", ".join(f'("{rel}", vec![{", ".join(map(str, places))}], {"true" if m == "x" else "false"})' for rel in sorted(plan.demanded) for places, m in (plan.modes.get(rel) or [((), "e")]))
    w(f"    /// The demanded relations and how each is asked: (relation, places its key binds, whether only existence is asked).")
    w(f"    pub fn demanded() -> Vec<(&'static str, Vec<usize>, bool)> {{ vec![{items}] }}")
    w("    /// For each query (a demanded relation, a key with FREE at the places not bound): the rows by the way of asking that key, and, where that exact key has a mode that asks existence, whether a row exists.")
    w("    pub fn ask_many(&mut self, s: &State, does: &[(u32, u32)], queries: &[(String, Vec<u32>)]) -> Vec<(Vec<Vec<u32>>, Option<bool>)> {")
    w("        let mut d = self.load(s, does);")
    if plan.gated:
        w("        d.whole = true;")
    if plan.ends or plan.asked:
        w("        self.dw.full = true;")
    w("        self.run(&mut d, ALL_DYNAMIC);")
    if plan.ends or plan.asked:
        w("        self.dw.full = false;")
    w("        let mut out = vec![];")
    w("        for (rel, key) in queries {")
    w("            let at: Vec<usize> = key.iter().enumerate().filter(|(_, k)| **k != FREE).map(|(i, _)| i).collect();")
    w("            out.push(match rel.as_str() {")
    for rel in sorted(plan.demanded):
        n = d.arity.get(rel, 0)
        allm = sorted(plan.modes.get(rel, []) + plan.test_modes.get(rel, []))
        w(f"                \"{rel}\" => {{")
        w(f"                    assert!(key.len() <= {n});")
        w(f"                    let mut k = [FREE; {n}]; k[..key.len()].copy_from_slice(key);")
        w(f"                    let mut o = Out::<{n}>::new();")
        w("                    match at.as_slice() {")
        for places, m in allm:
            if m == "e" and places:
                w(f"                        [{', '.join(map(str, places))}] => {plan.fname[(rel, places, 'e')]}(&self.st, &d, &mut self.tm, &k, &mut o),")
        w(f"                        _ => {plan.fname[(rel, (), 'e')]}(&self.st, &d, &mut self.tm, &k, &mut o),")
        w("                    }")
        w("                    let rows: Vec<Vec<u32>> = o.rows.iter().filter(|r| at.iter().all(|&p| r[p] == k[p])).map(|r| r.to_vec()).collect();")
        w("                    let ex = match at.as_slice() {")
        for places, m in allm:
            if m == "x":
                w(f"                        [{', '.join(map(str, places))}] => Some({plan.fname[(rel, places, 'x')]}(&self.st, &d, &mut self.tm, &k)),")
        w("                        _ => None,")
        w("                    };")
        w("                    (rows, ex)")
        w("                }")
    w("                other => panic!(\"{other} is not a demanded relation\"),")
    w("            });")
    w("        }")
    if pooled:
        w("        d.clear();")
        w("        self.pool = Some(d);")
    w("        out")
    w("    }")


# ------------------------------------------------------------------------------------------------------------------------------------------ what the pass bits adds to the document

def write_clause(w, writer, refusals):
    """The text of one clause; a relation the pass bits cannot hold for this clause is recorded in `refusals` (the dry run that decides which relations are held) and the clause is left out of that run."""
    try:
        w(writer.write())
    except BitRefuse as e:
        if refusals is None:
            raise
        refusals.setdefault(e.rel, e.why)


def clause_writer(d, places, name, rule, comp, plan, mode="full", record=False):
    if plan and plan.bits and (any(st[0] in ("pos", "neg") and st[1] in plan.bits for st in rule.steps) or (mode in ("bfull", "bdelta"))):
        return BitWriter(d, places, name, rule, comp, plan, mode, None, (), record)
    return ClauseWriter(d, places, name, rule, comp, plan, mode)


def start_orders(plan, rule, comp) -> list:
    """Pass (13): the distinct orders a clause is joined in when each of its positive literals over a relation held as bits is taken first (the reader's order first)."""
    have = {repr(rule.steps)}
    out = [rule]
    for lit in rule.clause.body:
        if lit[0] != "pos" or lit[1] not in plan.bits:
            continue
        jr = JoinRule(rule.clause, comp, plan.demanded, lit, est=plan.est)
        if repr(jr.steps) not in have:
            have.add(repr(jr.steps))
            out.append(jr)
    return out


def emit_full_clause(w, d, plan, places, name, rule, comp, recursive, refusals, alias=None) -> str:
    """A clause whose head is held as bits, made once from the words held (mode bfull). With the pass start and two or more admitted orders (module text, (13)): one function for each order, `{name}v<j>`, and a chooser `{name}` that runs the order of the least
    estimate; otherwise `{name}` is the clause in the reader's order, as before. -> the name to call."""
    def one(order, fname):
        return BitWriter(d, places, fname, order, comp, plan, "bfull", None, (), recursive, alias)
    if "start" not in plan.passes or d.static[rule.head_rel] or plan.start_off or plan.start_base is None:
        write_clause(w, one(rule, name), refusals)
        return name
    orders = start_orders(plan, rule, comp)
    if len(orders) < 2:
        write_clause(w, one(rule, name), refusals)
        return name
    key = (name, rule.clause.text)
    base = one(orders[0], f"{name}v0")
    try:
        text0 = base.write()
    except BitRefuse as e:
        if refusals is None:
            raise
        refusals.setdefault(e.rel, e.why)
        return name
    memo = plan.start_choice.get(key)
    admitted = [(0, text0, base.prof)]
    others: list = []
    for j in (range(1, len(orders)) if memo is None else [x for x in memo if x]):
        saved = {r: dict(i.orients) for r, i in plan.bits.items()}
        wr = one(orders[j], f"{name}v{len(admitted) + len(others)}")
        try:
            txt = wr.write()
        except (BitRefuse, ValueError) as e:                  # (a ValueError: the order asks a relation not held as bits by places no clause asks it by, and its indexes are fixed)
            for r, i in plan.bits.items():
                i.orients = saved[r]
            plan.start_refused[(rule.clause.text.replace("\n", " "), j)] = e.why if isinstance(e, BitRefuse) else f"it reads a relation not held as bits by places no clause before it reads it by: {orders[j].clause.text}".replace("\n", " ")
            continue
        new = sorted((r, f) for r, i in plan.bits.items() for f in i.orients if f not in plan.start_base.get(r, ()))
        dear = [x for x in new if not (d.static[x[0]] and "worth" in plan.passes)]       # pass (19): an orientation of a relation that does not read the state is made once, with the relation, and costs a question nothing
        if dear:
            for r, i in plan.bits.items():
                i.orients = saved[r]
            plan.start_refused[(rule.clause.text.replace("\n", " "), j)] = f"it asks the orientation {dear[0][1]} of {dear[0][0]}, which no clause in the reader's order asks" + (" (a relation that reads the state: each row set would set it too)" if "worth" in plan.passes else "") + f": {orders[j].clause.text.replace(chr(10), ' ')}"
            continue
        here = sorted((r, f) for r, i in plan.bits.items() for f in i.orients if f not in saved[r] and f not in plan.start_base.get(r, ()))
        if here:
            plan.worth_taken[(rule.clause.text.replace("\n", " "), j)] = here
        others.append((j, txt, wr.prof))
    if memo is None:
        others.sort(key=lambda x: (sum(1 for p in x[2] if "nz.len()" in p), x[0]))
        others = others[:START_MAX - 1]
    admitted += others
    plan.start_choice[key] = [x[0] for x in admitted]
    plan.start_written[key] = (len(admitted), len(orders))
    if len(admitted) == 1:
        w(text0.replace(f"fn {name}v0(", f"fn {name}(", 1))
        return name
    # the functions are named by their place among those admitted
    for n, (j, txt, prof) in enumerate(admitted):
        m = re.match(r"(?s).*?\nfn (" + re.escape(name) + r"v\d+)\(", "\n" + txt)
        w(txt.replace(f"fn {m.group(1)}(", f"fn {name}v{n}(", 1) if m else txt)
    lines = [f"fn {name}(s: &St, d: &mut Dy, tm: &mut Terms, dw: &mut Dw) -> u64 {{", "    let mut best = 0usize; let mut best_e = u64::MAX;"]
    for n, (j, txt, prof) in enumerate(admitted):
        lines.append("    { let mut p = 1u64; let mut e = 0u64;")
        for fct in prof:
            lines.append(f"      p = p.saturating_mul((({fct}) as u64).max(1)); e = e.saturating_add(p);")
        lines.append(f"      if e < best_e {{ best_e = e; best = {n}; }} }}")
    lines.append("    match best {")
    for n in range(len(admitted) - 1):
        lines.append(f"        {n} => {name}v{n}(s, d, tm, dw),")
    lines.append(f"        _ => {name}v{len(admitted) - 1}(s, d, tm, dw),")
    lines.append("    }")
    lines.append("}")
    w("\n".join(lines))
    return name


def asked_call(plan, rel: str, rule, name: str) -> str:
    """The statement that runs a base clause of a walk: always, or (pass (17)) only when everything is asked for (`dw.full`) or only when it is not."""
    info = plan.asked.get(rel)
    if info is not None and id(rule.clause) in info["plain"]:
        return f"if dw.full {{ made += {name}(s, d, tm, dw); }}"
    if info is not None and id(rule.clause) in info["guarded"]:
        return f"if !dw.full {{ made += {name}(s, d, tm, dw); }}"
    return f"made += {name}(s, d, tm, dw);"


def emit_shift_stratum(w, d, plan, places, k, comp, rules, cid, refusals):
    """Pass (9): a recursive set held as bits that is a walk, as one function that closes each key's frontier by shifts (once the base clauses have run); None (and the reason in `Plan.shift_refused`) where it is not that shape, the set then being made as above."""
    recs = [x for x in rules if x.rec_steps]
    bases = [x for x in rules if not x.rec_steps]
    buf: list = []
    h = sorted(comp)[0]
    try:
        if len(recs) != 1:
            raise ShiftRefuse(f"the set has {len(recs)} recursive clauses: " + "; ".join(x.clause.text.replace("\n", " ") for x in recs))
        shape = shift_shape(d, plan, comp, recs[0], places)
        ends = False
        if "ends" in plan.passes:
            try:
                plan.ends[h] = find_ends(d, plan, comp, recs[0], shape)
                ends = True
            except EndsRefuse as e:
                plan.ends_refused[h] = str(e)
        write_shift(buf.append, d, plan, places, f"c{cid + len(bases)}", comp, recs[0], shape, ends)
    except ShiftRefuse as e:
        plan.shift_refused[h] = str(e)
        plan.ends.pop(h, None)
        return None
    except BitRefuse as e:
        plan.shift_refused[h] = e.why
        plan.ends.pop(h, None)
        return None
    names = []
    for rule in bases:
        name = f"c{cid}"
        emit_full_clause(w, d, plan, places, name, rule, comp, False, refusals, f"dw.sb_{d.rid[h]}" if h in plan.ends else None)
        names.append(asked_call(plan, h, rule, name))
        cid += 1
    for t in buf:
        w(t)
    names.append(f"made += c{cid}(s, d, tm, dw);")
    cid += 1
    w(f"fn st{k}(s: &St, d: &mut Dy, tm: &mut Terms, dw: &mut Dw) -> (u64, u64, u64) {{")
    w("    let mut made = 0u64;")
    for c in names:
        w(f"    {c}")
    w("    (made, 0, 0)")
    w("}")
    return cid


def emit_bit_stratum(w, d, plan, places, k, comp, rules, cid, refusals):
    """The functions of a stratum all of whose relations are held as bits: each clause (a recursive one once for each of its literals of the set, that literal first and read from the words new in the last round) and the piece."""
    static = d.static[sorted(comp)[0]]
    recursive = any(x.rec_steps for x in rules)
    if "shift" in plan.passes and recursive and not static:
        got = emit_shift_stratum(w, d, plan, places, k, comp, rules, cid, refusals)
        if got is not None:
            return got
    base, rec = [], []
    for rule in rules:
        if rule.rec_steps:
            for j, lit in enumerate([x for x in rule.clause.body if x[0] == "pos" and x[1] in comp]):
                name = f"c{cid}d{j}"
                write_clause(w, BitWriter(d, places, name, JoinRule(rule.clause, comp, plan.demanded, lit, est=plan.est), comp, plan, "bdelta", None, (), True), refusals)
                rec.append(name)
        else:
            name = f"c{cid}"
            emit_full_clause(w, d, plan, places, name, rule, comp, recursive, refusals)
            base.append(name if static else asked_call(plan, sorted(comp)[0], rule, name))
        cid += 1
    if static:
        w(f"fn st{k}(s: &mut St, tm: &mut Terms, dy: &Dy, dw: &mut Dw) -> (u64, u64, u64) {{")
        call = lambda c: f"{c}(s, dy, tm, dw)"
    else:
        w(f"fn st{k}(s: &St, d: &mut Dy, tm: &mut Terms, dw: &mut Dw) -> (u64, u64, u64) {{")
        call = lambda c: f"{c}(s, d, tm, dw)"
    w("    let mut made = 0u64;")
    for c in base:
        w(f"    made += {call(c)};" if static else f"    {c}")
    if recursive:
        rels = [d.rid[h] for h in sorted(comp)]
        w("    loop {")
        for r in rels:
            w(f"        std::mem::swap(&mut dw.dl_{r}, &mut dw.nw_{r}); dw.nw_{r}.clear();")
        w("        if " + " && ".join(f"dw.dl_{r}.nz.is_empty()" for r in rels) + " { break; }")
        for c in rec:
            w(f"        made += {call(c)};")
        w("    }")
    w("    (made, 0, 0)")
    w("}")
    return cid


def emit_strata(w, d, plan, places, refusals=None) -> list:
    """Every clause and the piece of every stratum, to `w`; -> [(stratum number, whether it reads no state, its relations)]."""
    demanded = plan.demanded if plan else set()
    bits = plan.bits if plan else {}
    keep = bool(plan and plan.keep)
    if keep:
        plan.keep_out, plan.keep_rec = set(), []
    if plan:
        plan.shift_used = set()
        plan.ends, plan.ends_refused = {}, {}
    cid = 0
    stratum_fns = []
    for k, (comp, rules) in enumerate(d.strata):
        static = d.static[sorted(comp)[0]]
        if comp & demanded:
            stratum_fns.append((k, static, comp))              # answered on demand: no piece runs it
            continue
        if plan:
            rules = [x for x in plan.rules[k] if not plan.is_copy_rule(x)]
        if bits and all(h in bits for h in comp):
            cid = emit_bit_stratum(w, d, plan, places, k, comp, rules, cid, refusals)
            stratum_fns.append((k, static, comp))
            continue
        cids = []
        dl_fns: dict = {}
        for rule in rules:
            write_clause(w, clause_writer(d, places, cid, rule, comp, plan), refusals)
            cids.append((cid, rule))
            if plan and plan.delta.get(k):
                for r0, got in plan.delta[k]:
                    if r0 is rule:
                        for j, (rel, jr) in enumerate(got):
                            name = f"c{cid}d{j}"
                            write_clause(w, clause_writer(d, places, name, jr, comp, plan, "delta"), refusals)
                            dl_fns.setdefault(cid, []).append((name, rel))
            cid += 1
        heads = sorted({r.head_rel for r in rules})
        recursive = any(r.rec_steps for r in rules)
        X = "s" if static else "d"
        dwp = ", dw: &mut Dw" if (bits or keep) else ""
        if static:
            w(f"fn st{k}(s: &mut St, tm: &mut Terms, dy: &Dy{dwp}) -> (u64, u64, u64) {{")
            call = lambda c, di: f"c{c}(&*s, dy, tm, {di}, &dl, &mut o);"
            calld = lambda name: f"{name}(&*s, dy, tm, &dl, &mut o);"
        else:
            w(f"fn st{k}(s: &St, d: &mut Dy, tm: &mut Terms{dwp}) -> (u64, u64, u64) {{")
            call = lambda c, di: f"c{c}(s, &*d, tm, {di}, &dl, &mut o);"
            calld = lambda name: f"{name}(s, &*d, tm, &dl, &mut o);"
        new_out = (lambda n: f"let mut o = Out::<{n}> {{ rows: std::mem::take(&mut dw.o{n}), visits: 0, probes: 0 }};") if keep else (lambda n: f"let mut o = Out::<{n}>::new();")
        rows_of = "o.rows.drain(..)" if keep else "o.rows"
        end_out = (lambda n: f" dw.o{n} = o.rows;") if keep else (lambda n: "")
        w("    let (mut made, mut visits, mut probes) = (0u64, 0u64, 0u64);")
        w("    let mut dl = std::mem::take(&mut dw.dl);" if keep else "    let mut dl = Dl::default();")
        for c, rule in cids:
            h = rule.head_rel
            n = len(rule.head)
            if keep:
                plan.keep_out.add(n)
            w(f"    {{ {new_out(n)} {call(c, 'usize::MAX')} visits += o.visits; probes += o.probes;")
            if recursive:
                w(f"      for row in {rows_of} {{ if {X}.{d.rid[h]}.add(row) {{ made += 1; dl.{d.rid[h]}.push(row); }} }}{end_out(n)} }}")
            else:
                w(f"      for row in {rows_of} {{ if {X}.{d.rid[h]}.add(row) {{ made += 1; }} }}{end_out(n)} }}")
        if recursive:
            w("    loop {")
            w("        if " + " && ".join(f"dl.{d.rid[h]}.is_empty()" for h in sorted(comp)) + " { break; }")
            for h in sorted(comp):
                n = d.arity.get(h, 0)
                if keep:
                    if h not in plan.keep_rec:
                        plan.keep_rec.append(h)
                    w(f"        let mut nw{d.rid[h]}: Vec<[u32; {n}]> = std::mem::take(&mut dw.nw{d.rid[h]}); nw{d.rid[h]}.clear(); let mut sn{d.rid[h]}: HashSet<[u32; {n}], B> = std::mem::take(&mut dw.sn{d.rid[h]}); sn{d.rid[h]}.clear();")
                else:
                    w(f"        let mut nw{d.rid[h]}: Vec<[u32; {n}]> = vec![]; let mut sn{d.rid[h]}: HashSet<[u32; {n}], B> = HashSet::default();")
            for c, rule in cids:
                h = rule.head_rel
                n = len(rule.head)
                if c in dl_fns:
                    todo = [(d.rid[rel], calld(name)) for name, rel in dl_fns[c]]
                else:
                    todo = [(d.rid[rule.steps[i][1]], call(c, str(i))) for i in rule.rec_steps]
                for rr, text_call in todo:
                    w(f"        if !dl.{rr}.is_empty() {{ {new_out(n)} {text_call} visits += o.visits; probes += o.probes;")
                    w(f"            for row in {rows_of} {{ if !{X}.{d.rid[h]}.has(&row) && sn{d.rid[h]}.insert(row) {{ nw{d.rid[h]}.push(row); }} }}{end_out(n)} }}")
            for h in sorted(comp):
                w(f"        for row in &nw{d.rid[h]} {{ {X}.{d.rid[h]}.add(*row); made += 1; }}")
                if keep:
                    w(f"        std::mem::swap(&mut dl.{d.rid[h]}, &mut nw{d.rid[h]}); dw.nw{d.rid[h]} = nw{d.rid[h]}; dw.sn{d.rid[h]} = sn{d.rid[h]};")
                else:
                    w(f"        dl.{d.rid[h]} = nw{d.rid[h]};")
            w("    }")
        if keep:
            w("    dw.dl = dl;")
        w("    (made, visits, probes)")
        w("}")
        stratum_fns.append((k, static, comp))
    return stratum_fns


def w_bit_tables(w, d, plan):
    """The tables of the pass bits: for each sort a held relation uses, the ordinal of each symbol and the symbol of each ordinal; for each sort of tile, the masks of the tiles whose component has one value."""
    p = d.prog
    by_id = sorted(plan.sids.items(), key=lambda kv: kv[1])
    used = {s for i in plan.bits.values() for t in i.tiles for s in t.sorts}
    for s, n in by_id:
        if s not in used:
            continue
        table = [NONE] * len(d.syms)
        for j, m in enumerate(p.sorts[s]):
            table[d.sym_id[m]] = j
        w(f"static OS{n}: [u16; {len(d.syms)}] = [{', '.join(map(str, table))}];")
        w(f"static SS{n}: [u32; {len(p.sorts[s])}] = [{', '.join(str(d.sym_id[m]) for m in p.sorts[s])}];")
    kinds = sorted({t.sorts for i in plan.bits.values() for t in i.tiles if len(t.sorts) == 2})
    for sorts in kinds:
        n = plan.tids[sorts]
        a, b = len(p.sorts[sorts[0]]), len(p.sorts[sorts[1]])
        w(f"static MK{n}_0: [u64; {a}] = [{', '.join(str(sum(1 << (x * b + y) for y in range(b))) + 'u64' for x in range(a))}];")
        w(f"static MK{n}_1: [u64; {b}] = [{', '.join(str(sum(1 << (x * b + y) for x in range(a))) + 'u64' for y in range(b))}];")


def w_bit_relation(w, d, plan, info: BitInfo):
    """The struct of a relation held as bits: an array of words for each orientation asked, `set_t` (every orientation), `or_word` (a word into the last tile's orientation), and the items the rest of the document reads a relation by (`add`, `has`, `get`, `rows_all`, `is_empty`, `clear`)."""
    rid = d.rid[info.rel]
    name = "B" + rid[1:]
    n, m, c = info.n, info.m, info.canon
    ors = sorted(info.orients)
    strides: dict = {}
    tt = {i: f"t[{i}]" for i in range(m)}
    w(f"pub struct {name} {{ " + ", ".join(f"o{f}: Bw" for f in ors) + " }")
    w(f"impl {name} {{")
    w(f"    pub fn new() -> {name} {{ {name} {{ " + ", ".join(f"o{f}: Bw::new({info.key_size(f)})" for f in ors) + " } }")
    ix = lambda f, t: IndexExpr(info, f, t)
    w(f"    #[inline] pub fn set_t(&mut self, t: [usize; {m}]) -> bool {{")
    w(f"        if !self.o{c}.set({ix(c, tt)}, t[{c}]) {{ return false; }}")
    for f in ors:
        if f != c:
            w(f"        self.o{f}.set({ix(f, tt)}, t[{f}]);")
    w("        true")
    w("    }")
    w(f"    #[inline] pub fn has_t(&self, t: [usize; {m}]) -> bool {{ (self.o{c}.w[{ix(c, tt)}] >> t[{c}]) & 1 == 1 }}")
    w(f"    /// Or the word `v` into the word `key` of the last tile's orientation (the tiles before it numbered as that orientation numbers them); the bits that were not set.")
    w("    pub fn or_word(&mut self, key: usize, v: u64) -> u64 {")
    w(f"        let nb = self.o{c}.or(key, v);")
    if len(ors) > 1:
        w(f"        if nb != 0 {{")
        w(f"            let mut t = [0usize; {m}];")
        w("            let mut rm = key;")
        kt = info.key_tiles(c)
        for i in reversed(kt):
            w(f"            t[{i}] = rm % {info.tiles[i].size}; rm /= {info.tiles[i].size};")
        if "orient" in plan.passes:
            # pass (14): the other orientations are set from the new bits of the word; what does not change with the bit (the index with the bit at nought, the tile that is free there) is made once, the bit moves the index by a constant
            for f in ors:
                if f != c:
                    kf = info.key_tiles(f)
                    stride = 1
                    for i in kf[kf.index(c) + 1:]:
                        stride *= info.tiles[i].size
                    w(f"            let (bf{f}, af{f}) = ({IndexExpr(info, f, {i: ('0' if i == c else f't[{i}]') for i in range(m)})}, t[{f}]);")
                    strides[f] = stride
            w("            let mut b = nb;")
            w("            while b != 0 { let p = b.trailing_zeros() as usize; b &= b - 1;")
            for f in ors:
                if f != c:
                    w(f"                self.o{f}.set(bf{f} + p * {strides[f]}, af{f});")
            w("            }")
        else:
            w("            let mut b = nb;")
            w(f"            while b != 0 {{ t[{c}] = b.trailing_zeros() as usize; b &= b - 1;")
            for f in ors:
                if f != c:
                    w(f"                self.o{f}.set({ix(f, tt)}, t[{f}]);")
            w("            }")
        w("        }")
    w("        nb")
    w("    }")
    # tiles of a row, and the row of tiles
    w(f"    pub fn tiles_of(r: &[u32; {n}]) -> Option<[usize; {m}]> {{")
    parts = []
    for i, t in enumerate(info.tiles):
        cs = [f"ordof(&OS{plan.sids[t.sorts[x]]}, r[{p}])" for x, p in enumerate(t.places)]
        w(f"        let a{i} = [{', '.join(cs)}];")
        w(f"        if a{i}.iter().any(|&x| x == NONE) {{ return None; }}")
        parts.append(f"a{i}[0]" if len(cs) == 1 else f"a{i}[0] * {t.sizes[1]} + a{i}[1]")
    w(f"        Some([{', '.join(parts)}])")
    w("    }")
    w(f"    pub fn row_of(t: [usize; {m}]) -> [u32; {n}] {{")
    w(f"        let mut r = [0u32; {n}];")
    for i, t in enumerate(info.tiles):
        for x, p in enumerate(t.places):
            comp = f"t[{i}]" if len(t.places) == 1 else (f"t[{i}] / {t.sizes[1]}" if x == 0 else f"t[{i}] % {t.sizes[1]}")
            w(f"        r[{p}] = SS{plan.sids[t.sorts[x]]}[{comp}];")
    w("        r")
    w("    }")
    w(f"    pub fn add(&mut self, r: [u32; {n}]) -> bool {{ let t = Self::tiles_of(&r).expect(\"a row outside the sorts of its relation\"); self.set_t(t) }}")
    w(f"    pub fn has(&self, r: &[u32; {n}]) -> bool {{ match Self::tiles_of(r) {{ Some(t) => self.has_t(t), None => false }} }}")
    w(f"    pub fn rows_all(&self) -> Vec<[u32; {n}]> {{")
    w("        let mut out = vec![];")
    w(f"        for &key in self.o{c}.nz.iter() {{")
    w(f"            let mut t = [0usize; {m}];")
    w("            let mut rm = key as usize;")
    for i in reversed(info.key_tiles(c)):
        w(f"            t[{i}] = rm % {info.tiles[i].size}; rm /= {info.tiles[i].size};")
    w(f"            let mut b = self.o{c}.w[key as usize];")
    w(f"            while b != 0 {{ t[{c}] = b.trailing_zeros() as usize; b &= b - 1; out.push(Self::row_of(t)); }}")
    w("        }")
    w("        out")
    w("    }")
    if "words" in plan.passes:
        w(f"    /// The rows that equal `key` at the places that are not FREE, one by one, with no list made (pass 22).")
        w(f"    pub fn iter<'a>(&'a self, key: &'a [u32; {n}]) -> impl Iterator<Item = [u32; {n}]> + 'a {{")
        w(f"        let ws = &self.o{c};")
        w(f"        let (mut wi, mut b, mut t) = (0usize, 0u64, [0usize; {m}]);")
        w("        std::iter::from_fn(" + "mo" + "ve || {")         # the keyword of a closure that owns its state, spelled in two parts: the scan of names (tests/test_plan79_37.py) reads words, and the word is a relation of a description
        w("            loop {")
        w("                if b != 0 {")
        w(f"                    t[{c}] = b.trailing_zeros() as usize; b &= b - 1;")
        w("                    let r = Self::row_of(t);")
        w(f"                    if (0..{n}).all(|p| key[p] == FREE || r[p] == key[p]) {{ return Some(r); }}")
        w("                    continue;")
        w("                }")
        w("                if wi >= ws.nz.len() { return None; }")
        w("                let kw = ws.nz[wi] as usize; wi += 1;")
        w("                let mut rm = kw;")
        for i in reversed(info.key_tiles(c)):
            w(f"                t[{i}] = rm % {info.tiles[i].size}; rm /= {info.tiles[i].size};")
        w("                b = ws.w[kw];")
        w("            }")
        w("        })")
        w("    }")
        w(f"    /// Each row, by value, with no list made (pass 22).")
        w(f"    pub fn each(&self, mut f: impl FnMut([u32; {n}])) {{")
        w(f"        for &key in self.o{c}.nz.iter() {{")
        w(f"            let mut t = [0usize; {m}];")
        w("            let mut rm = key as usize;")
        for i in reversed(info.key_tiles(c)):
            w(f"            t[{i}] = rm % {info.tiles[i].size}; rm /= {info.tiles[i].size};")
        w(f"            let mut b = self.o{c}.w[key as usize];")
        w(f"            while b != 0 {{ t[{c}] = b.trailing_zeros() as usize; b &= b - 1; f(Self::row_of(t)); }}")
        w("        }")
        w("    }")
    w(f"    pub fn get(&self, _i: usize, key: &[u32; {n}]) -> Vec<[u32; {n}]> {{ self.rows_all().into_iter().filter(|r| (0..{n}).all(|p| key[p] == FREE || r[p] == key[p])).collect() }}")
    w(f"    pub fn is_empty(&self) -> bool {{ self.o{c}.nz.is_empty() }}")
    w("    pub fn clear(&mut self) { " + " ".join(f"self.o{f}.clear();" for f in ors) + " }")
    w("}")


class IndexExpr:
    """The Rust text of the number of the word of orientation f for tile numbers given as text (str() of it)."""

    def __init__(self, info: BitInfo, f: int, tiles: dict):
        kt = info.key_tiles(f)
        if not kt:
            self.text = "0"
        else:
            acc = tiles[kt[0]]
            for i in kt[1:]:
                acc = f"({acc}) * {info.tiles[i].size} + {tiles[i]}"
            self.text = acc

    def __str__(self):
        return self.text


def w_dw(w, d, plan):
    """The working memory the questions share (pass bits: the words new in the last round and in this one, for each relation of a recursive set held as bits, the last tile's orientation; pass keep: the buffers of the rows a clause that is not held as words gives,
    and for a recursive set that is not held as words the rows new in a round and the set that dedupes them), kept between questions."""
    fields, inits = [], []
    if plan.ends or plan.asked:
        fields.append("pub full: bool")
        inits.append("full: false")
    if plan.ends:
        for r in sorted(plan.ends):
            fields.append(f"pub sb_{d.rid[r]}: Bw")
            inits.append(f"sb_{d.rid[r]}: Bw::new({plan.bits[r].key_size(plan.bits[r].canon)})")
    for r in plan.bit_rec:
        n = plan.bits[r].key_size(plan.bits[r].canon)
        fields += [f"pub dl_{d.rid[r]}: Bw", f"pub nw_{d.rid[r]}: Bw"]
        inits += [f"dl_{d.rid[r]}: Bw::new({n})", f"nw_{d.rid[r]}: Bw::new({n})"]
    if plan.keep:
        fields.append("pub dl: Dl")
        inits.append("dl: Dl::default()")
        for n in sorted(plan.keep_out):
            fields.append(f"pub o{n}: Vec<[u32; {n}]>")
            inits.append(f"o{n}: Vec::new()")
        for h in plan.keep_rec:
            n = d.arity.get(h, 0)
            fields += [f"pub nw{d.rid[h]}: Vec<[u32; {n}]>", f"pub sn{d.rid[h]}: HashSet<[u32; {n}], B>"]
            inits += [f"nw{d.rid[h]}: Vec::new()", f"sn{d.rid[h]}: HashSet::default()"]
    w("pub struct Dw { " + ", ".join(fields) + (" " if fields else "") + "}")
    w("impl Dw { pub fn new() -> Dw { Dw { " + ", ".join(inits) + " } } }")


# ------------------------------------------------------------------------------------------------------------------------------------------ the document

# ------------------------------------------------------------------------------------------------------------------------------------------ a lemma as a rewrite (plan 79.45 part A)

LEMMA_FORM = "SAME"                                     # the one form that is a rewrite: under the condition the left relation has the rows of the right one
LEMMA_STANDINGS = ("CERTIFIED", "PROVED")               # the standings a lemma may be given to the translator with (the make script computes them; the translator refuses the others)
NO_ROWS = "differ"                                      # the one literal of a condition that is no row: two terms that differ (the form of rule_lemma.py)


def lemma_atom(prog, rel: str, args: list):
    """The s-expression of a literal over `rel` (a relation of the text, or the fluent of that name) with `args`, and the relation's name in the text."""
    if rel in prog.nodes:
        name = rel
    elif ("true." + rel) in prog.nodes:
        name = "true." + rel
    else:
        raise Untranslatable(f"the relation {rel} is not in the text")
    sig = prog.signature(name)
    if sig is not None and len(sig) != len(args):
        raise Untranslatable(f"{rel} takes {len(sig)} places and the literal gives {len(args)}")
    if name.startswith("true."):
        return ["true", ([name[len("true."):]] + list(args)) if args else name[len("true."):]], name
    return (([name] + list(args)) if args else name), name


def lemma_problems(lem: dict) -> list:
    """What is wrong with the form of a lemma given as data (the keys, the form, the standing); the relations are read against the text by `rewrite_lemmas`."""
    keys = ("name", "form", "left", "right", "condition", "standing")
    miss = [k for k in keys if k not in lem]
    if miss:
        return [f"the lemma has no {miss}"]
    bad = []
    nm = lem["name"]
    if lem["form"] != LEMMA_FORM:
        bad.append(f"lemma {nm}: the form is {lem['form']}; only {LEMMA_FORM} is a rewrite (a lemma of inclusion changes no relation)")
    st = lem["standing"]
    if not isinstance(st, dict) or st.get("is") not in LEMMA_STANDINGS:
        bad.append(f"lemma {nm}: the standing is {st}; only {' and '.join(LEMMA_STANDINGS)} lemmas are given to the translator")
    elif (st["is"] == "CERTIFIED" and not st.get("key")) or (st["is"] == "PROVED" and not st.get("theorem")):
        bad.append(f"lemma {nm}: a {st['is']} lemma names its evidence ({'key' if st['is'] == 'CERTIFIED' else 'theorem'})")
    for end_of in ("left", "right"):
        r = lem[end_of]
        if not isinstance(r, dict) or not isinstance(r.get("relation"), str) or not isinstance(r.get("places"), int):
            bad.append(f"lemma {nm}: {end_of} is {{relation, places}}")
    for i, c in enumerate(lem["condition"]):
        if not (isinstance(c, dict) and isinstance(c.get("relation"), str) and isinstance(c.get("args"), list) and all(isinstance(a, str) for a in c["args"])):
            bad.append(f"lemma {nm}: the condition's literal {i} is not {{relation, args, neg}}")
    return bad


def swap_atom(x, old: str, new: str):
    """A body literal of a clause with the atoms over `old` read over `new` (through `not` and `or`; the terms of an atom are not entered)."""
    if isinstance(x, list) and x and x[0] in ("not", "or"):
        return [x[0]] + [swap_atom(y, old, new) for y in x[1:]]
    if isinstance(x, list) and x and x[0] == "distinct":
        return x
    if isinstance(x, list) and x and x[0] == old:
        return [new] + x[1:]
    if x == old:
        return new
    return x


def rewrite_lemmas(text: str, lemmas) -> tuple:
    """Plan 79.45 part A (the module text states the rewrite and its argument): -> (the text with each SAME lemma read as a rewrite, [the facts of each lemma for the document]). Untranslatable, naming the lemma and the reason, where the argument's needs (i) to (iii) are not met."""
    cur = text
    infos = []
    for n, lem in enumerate(lemmas, 1):
        bad = lemma_problems(lem)
        if bad:
            raise Untranslatable("; ".join(bad))
        nm = lem["name"]
        try:
            prog = GDL.Program(cur)
        except GDL.GDLError as e:
            raise Untranslatable(f"lemma {nm}: the text does not read: {e}")
        left, right = lem["left"]["relation"], lem["right"]["relation"]
        for rel in (left, right):
            if rel not in prog.nodes:
                raise Untranslatable(f"lemma {nm}: the relation {rel} is not in the text (the composed text holds the lemma's delta)")
        arity = lem["left"]["places"]
        for rel, places in ((left, arity), (right, lem["right"]["places"])):
            sig = prog.signature(rel)
            if sig is not None and len(sig) != places:
                raise Untranslatable(f"lemma {nm}: {rel} has {len(sig)} places in the text and the document says {places}")
        if lem["right"]["places"] != arity:
            raise Untranslatable(f"lemma {nm}: the two relations have different places")
        if left in QUESTION_RELS or left in INPUT_RELS or left.startswith(("true.", "next.", "init.")):
            raise Untranslatable(f"lemma {nm}: {left} is read by a question or is a state, an input or a next: its rows are not for a rewrite")
        taken = set(prog.nodes) | set(prog.signatures)
        flag, rep = f"lemma{n}_flag", f"lemma{n}_rep"
        while flag in taken or rep in taken:
            flag, rep = flag + "_", rep + "_"
        cond, resolved = [], []
        for c in lem["condition"]:
            neg = bool(c.get("neg", False))
            if c["relation"] == NO_ROWS:
                if len(c["args"]) != 2 or neg:
                    raise Untranslatable(f"lemma {nm}: a {NO_ROWS} literal takes two terms and is positive")
                cond.append(["distinct", c["args"][0], c["args"][1]])
                resolved.append((NO_ROWS, list(c["args"]), False))
                continue
            atom, name = lemma_atom(prog, c["relation"], c["args"])
            cond.append(["not", atom] if neg else atom)
            resolved.append((name, list(c["args"]), neg))
        xs = [f"?x{i}" for i in range(1, arity + 1)]
        atom_of = lambda rel: ([rel] + xs) if xs else rel
        out = []
        for x in GDL.parse(cur):
            if isinstance(x, list) and x and x[0] == "<=" and GDL.to_atom(x[1])[0] != left:
                x = [x[0], x[1]] + [swap_atom(y, left, rep) for y in x[2:]]
            out.append(x)
        sig = prog.signature(left)
        if sig is not None:
            out.append(["signature", rep] + list(sig))
            out.append(["signature", flag])
        out.append(["<=", flag] + cond)
        out.append(["<=", atom_of(rep), flag, atom_of(right)])
        out.append(["<=", atom_of(rep), ["not", flag], atom_of(left)])
        new = "\n".join(GDL._unparse(x) for x in out) + "\n"
        try:
            d2 = Description(new)
        except GDL.GDLError as e:
            raise Untranslatable(f"lemma {nm}: the text with the rewrite is not a program: {e}")
        if len(next(comp for comp, _r in d2.strata if left in comp)) > 1 or any(r.rec_steps for comp, rules in d2.strata if left in comp for r in rules):
            raise Untranslatable(f"lemma {nm}: {left} is in a recursive set: its replacement would read itself")
        for who, rel in (("the right relation", right), ("the flag", flag)):
            if left in d2.cone([rel]):
                raise Untranslatable(f"lemma {nm}: {who} reads {left}: the rewrite would read itself")
        turn = sorted(set(INPUT_RELS) & d2.cone([left, right, flag]))
        if turn:
            raise Untranslatable(f"lemma {nm}: {left}, {right} or the condition reads the turn ({turn}): the lemma is about states")
        ev = lem["standing"]
        infos.append({"name": nm, "form": lem["form"], "left": left, "right": right, "rep": rep, "flag": flag, "evidence": f"{ev['is']} {ev.get('key') or ev.get('theorem')}", "condition": resolved})
        cur = new
    return cur, infos


def over_approximations(text: str) -> tuple:
    """Plan 79.47 part A (module text, (17, widened)): -> (the text, and for each walk that a clause outside it reads more than once, a relation that is its STATIC OVER-APPROXIMATION added to the text, {walk: its over-approximation}). The over-approximation `S` of a
    walk `W` has W's clauses with every literal over a relation that reads the state taken out and W's own literal read over S: S holds every row of W (a clause with fewer literals gives at least the rows of the clause, and a positive program is monotone), and S reads no state, so it
    is made once and a guard of the pass asked may read it in place of an earlier read of W. A walk for which the clauses cannot be so written (a literal of the walk negated, an `or` in a clause, no declared sorts, an unsafe clause) has none: its earlier reads give no guard."""
    d = Description(text)
    sidx = {h: k for k, (comp, _r) in enumerate(d.strata) for h in comp}
    twice: set = set()
    for k, (comp, rules) in enumerate(d.strata):
        for x in rules:
            count: dict = {}
            for lit in x.clause.body:
                if lit[0] == "pos" and lit[1] not in comp:
                    count[lit[1]] = count.get(lit[1], 0) + 1
            twice |= {r for r, n in count.items() if n > 1}
    taken = set(d.prog.nodes) | set(d.prog.signatures)
    sexprs = GDL.parse(text)
    added: list = []
    over: dict = {}
    for w in sorted(twice):
        k = sidx.get(w)
        if k is None:
            continue
        comp, rules = d.strata[k]
        sig = d.prog.signature(w)
        if len(comp) != 1 or d.static[w] or w in QUESTION_RELS or w in INPUT_RELS or w.startswith(("true.", "next.", "init.")) or not any(x.rec_steps for x in rules) or sig is None:
            continue
        ws = f"{w}_over"
        while ws in taken:
            ws += "_"
        new: list = []
        ok = True
        for x in sexprs:
            if not (isinstance(x, list) and x and x[0] == "<=" and not isinstance(x[1], str) and GDL.to_atom(x[1])[0] == w):
                continue
            body = []
            for item in x[2:]:
                alts = GDL._literals(item)
                if len(alts) != 1 or len(alts[0]) != 1:
                    ok = False
                    break
                sign, rel = alts[0][0][0], alts[0][0][1]
                if rel == w and sign == "neg":
                    ok = False
                    break
                if sign == "distinct" or rel == w or d.static.get(rel, False):
                    body.append(swap_atom(item, w, ws))
            if not ok:
                break
            new.append(["<=", [ws] + x[1][1:]] + body)
        if not ok or not new:
            continue
        new.append(["signature", ws] + list(sig))
        try:
            Description(text + "\n" + "\n".join(GDL._unparse(c) for c in added + new) + "\n")
        except GDL.GDLError:
            continue
        added += new
        taken.add(ws)
        over[w] = ws
    if not added:
        return text, {}
    return text + "\n" + "\n".join(GDL._unparse(c) for c in added) + "\n", over


# ------------------------------------------------------------------------------------------------------------------------------------------ pass (24): the top-down way

TD_CAP = 64                                             # pass (24): the rows a walk of one relation keeps for one question, at most this many (more: the question is answered by the bottom-up way)


class TopDownRefuse(Exception):
    """The top-down way cannot be made for this description, with the reason (the relation or the clause is named in it)."""


class TopDown:
    """PASS (24), the top-down way (module text, (24)): the functions that answer the questions "is there a legal turn" and "does the end hold" by a depth-first walk of the clauses, from the question's relation down to the facts of the state and to the relations that do not read it,
    stopping at the first answer. One function for each (relation, the places a caller binds); a literal of a derived relation is a call with the rest of the clause as the continuation (a closure that is given each row and says whether to stop); a negated literal is a call that
    asks whether a row exists, its places bound; a relation of one linear recursive clause is made for one call as a worklist of the rows found (each once, at most TD_CAP of them: more and the question is given to the bottom-up way). Nothing here names a relation: every name comes from the text."""

    def __init__(self, d: "Description", plan: "Plan"):
        self.d, self.plan = d, plan
        self.texts: list = []
        self.fns: dict = {}                             # (relation, bound places) -> function name
        self.todo: list = []
        self.pats: dict = {}                            # relation -> sorted list of the bound-place tuples a literal asks it by (static and fluent relations)
        self.used: set = set()                          # the static and fluent relations a function reads
        self.head_compound: dict = {}                   # relation -> places where some clause has a compound head term
        self.nvar = 0
        self.cone: set = set()
        self.late: dict = {}                            # pass (26): relation made from a hypothetical relation -> the hypothetical relations its clauses read last (behind the literals that bind the places they are read at)
        self.rec: dict = {}                             # pass (26): hypothetical relation -> (its number, key places, row places): its functions record the places they are asked about
        self.extra: list = []                           # pass (26): (relation, bound places) functions made besides those the cone of the questions asks
        self.fl_rec: bool = False                       # pass (27): every read of a fluent of the state is recorded (`rd_note`)
        self.fl_no: dict = {}                           # pass (27): fluent relation -> its number, in the order of the first read emitted
        self.fl_ex: bool = False                        # pass (28): every recorded read also carries the clause and the literal of the description that made it
        self.ex_clause: dict = {}                       # pass (28): clause text -> its number, in the order the clauses are first met
        self.ex_lits: dict = {}                         # pass (28): clause number -> the texts of its literals in the order written
        self.ex_func: dict = {}                         # pass (28): (clause number, head places bound, join order, a worklist clause) -> the number of the function-clause, in the order made
        self.cur_fi: int = 0                            # pass (28): the number of the function-clause being written
        self.keeps: dict = {}                           # relation of one linear recursive clause -> the places the walk keeps (the head's variable is the literal's): a literal of it is taken once they are bound

    # ---- what is a relation here
    def how(self, rel: str) -> str:
        d = self.d
        if rel.startswith("true."):
            if rel[len("true."):] not in self.plan.words:
                raise TopDownRefuse(f"the fluent {rel} is not held as words")
            return "fluent"
        if rel in INPUT_RELS:
            return "input"
        if d.static[rel]:
            return "static"
        return "derived"

    def clauses_of(self, rel: str) -> list:
        k = next(k for k, (comp, _r) in enumerate(self.d.strata) if rel in comp)
        comp, rules = self.d.strata[k]
        if len(comp) != 1 and any(x.rec_steps for x in rules):
            raise TopDownRefuse(f"{rel} is in a recursive set of {len(comp)} relations")
        return [x.clause for x in rules if x.clause.head[0] == rel]

    def is_recursive(self, rel: str) -> bool:
        k = next(k for k, (comp, _r) in enumerate(self.d.strata) if rel in comp)
        return any(x.rec_steps for x in self.d.strata[k][1])

    def pat(self, rel: str, keypos: tuple) -> int:
        lst = self.pats.setdefault(rel, [])
        if tuple(keypos) not in lst:
            lst.append(tuple(keypos))
        self.used.add(rel)
        return lst.index(tuple(keypos))

    def fname(self, rel: str, mask: tuple) -> str:
        key = (rel, tuple(mask))
        if key not in self.fns:
            n = self.d.arity.get(rel, 0)
            self.fns[key] = f"td_{self.d.rid[rel][2:]}_m{''.join('1' if j in mask else '0' for j in range(n))}"
            self.todo.append(key)
        return self.fns[key]

    # ---- terms
    def expr(self, t) -> str:
        if t[0] == "v":
            return f"v{t[1]}"
        if t[0] == "c":
            if t[1].startswith(BOUND):
                return f"u_{ident(t[1][len(BOUND):])}"
            return str(self.d.sym_id[t[1]])
        raise TopDownRefuse("a compound term in the body of a clause in the cone of the question")

    def make(self, rel: str, mask: tuple) -> str:
        """The text of the function for (relation, bound places)."""
        d = self.d
        n = d.arity.get(rel, 0)
        name = self.fname(rel, mask)
        rec = self.is_recursive(rel)
        cl = self.clauses_of(rel)
        out = [f"fn {name}(td: &Td, a: [u32; {n}], k: &mut dyn FnMut(&[u32; {n}]) -> bool) -> bool {{"]
        if rel in self.rec:
            hi, hm, hp = self.rec[rel]
            out.append(f"    if SP_ON{hi}.with(|o| o.load(std::sync::atomic::Ordering::Relaxed)) {{ SP_REC{hi}.with(|r| {{ let mut q = [FREE; {hp}]; q.copy_from_slice(&a[{hm}..]); r.lock().unwrap().push(q); }}); }}")
        if rec:
            out.append(f"    let mut got = [[0u32; {n}]; {TD_CAP}];")
            out.append("    let mut ns = 0usize;")
            test = " && ".join(f"(a[{j}] == FREE || r[{j}] == a[{j}])" for j in range(n)) or "true"
            out.append(f"    macro_rules! out {{ ($r:expr) => {{{{ let r: [u32; {n}] = $r; if !got[..ns].contains(&r) {{ if ns == {TD_CAP} {{ td.overflow.store(true, std::sync::atomic::Ordering::Relaxed); return true; }} got[ns] = r; ns += 1; if {test} && k(&r) {{ return true; }} }} }}}}; }}")
        sink = "out!(ROW);" if rec else "if k(&ROW) { return true; }"
        inv = set(range(n))
        for c in cl:
            lits = [x for x in c.body if x[0] == "pos" and x[1] == rel]
            if rec and len(lits) > 1:
                raise TopDownRefuse(f"{rel} reads itself twice in a clause: {c.text}")
            for lit in lits:
                inv &= {j for j in range(n) if GDL.is_var(c.head[1][j]) and lit[2][j] == c.head[1][j]}      # the places the walk does not change: the head's variable is the literal's
        for c in cl:
            lits = [x for x in c.body if x[0] == "pos" and x[1] == rel]
            if lits:
                continue
            # the seeds of a walk are made for the places the walk keeps; the others are asked of each row found
            out += self.clause(rel, c, tuple(j for j in mask if (not rec or j in inv)), None, sink)
        if rec:
            out.append("    let mut hd = 0usize;")
            out.append("    while hd < ns {")
            out.append("        let w = got[hd]; hd += 1;")
            for c in cl:
                lits = [x for x in c.body if x[0] == "pos" and x[1] == rel]
                if lits:
                    out += self.clause(rel, c, (), lits[0], sink)
            out.append("    }")
        out.append("    false")
        out.append("}")
        return "\n".join(out)

    def clause(self, rel: str, c, mask: tuple, first, sink: str) -> list:
        """One clause as text, its head places in `mask` bound by the call (`a`), or, with `first` (the clause's literal of its own relation), bound by the row `w` taken from the worklist."""
        d = self.d
        names: set = set()
        conds, binds = [], []
        seenvar: dict = {}
        for j in mask:
            t = c.head[1][j]
            if isinstance(t, tuple):
                raise TopDownRefuse(f"a compound term at a bound place of {rel}: {c.text}")
            if GDL.is_var(t):
                if t in seenvar:
                    conds.append(f"a[{j}] == a[{seenvar[t]}]")
                else:
                    seenvar[t] = j
                    names.add(t)
                    binds.append(f"let u_{ident(t[1:])} = a[{j}];")
            else:
                conds.append(f"a[{j}] == {d.sym_id[t]}")
        rule = JoinRule(bind_clause(c, names) if names else c, {rel} if first is not None else set(), (), first, None, self.keeps, self.late.get(rel, ()))
        if self.fl_ex:
            self.cur_fi = self.explain_func(c, mask, first, rule)
        lines = [f"    if {' && '.join(conds) if conds else 'true'} {{"]
        self.ind = 2
        self.fail = ["continue"]
        self.bound: set = set()
        self.lab = 0
        body: list = []
        self.emit = body.append
        for b in binds:
            self.emit(b)
        self.emit("'cl: {")
        self.fail = ["break 'cl"]
        self.ind += 1
        steps = rule.steps
        # the head row
        def head_row():
            xs = []
            for j, t in enumerate(rule.head):
                if t[0] == "f":
                    xs.append("FREE - 1")
                else:
                    xs.append(self.expr(t))
            return "[" + ", ".join(xs) + "]"
        self.steps(rule, 0, head_row, sink, first)
        self.ind -= 1
        self.emit("}")
        lines += ["        " + x for x in body]
        lines.append("    }")
        return lines

    def line(self, s: str):
        self.emit("    " * (self.ind - 1) + s)

    def steps(self, rule, i: int, head_row, sink: str, first):
        d = self.d
        if i == len(rule.steps):
            self.line(sink.replace("ROW", head_row()))
            return
        st = rule.steps[i]
        fail = self.fail[-1]
        if st[0] == "distinct":
            self.line(f"if {self.expr(st[1])} == {self.expr(st[2])} {{ {fail}; }}")
            self.steps(rule, i + 1, head_row, sink, first)
            return
        sign, rel, args = st[0], st[1], st[2]
        n = len(args)
        way = self.how(rel)
        if any(a[0] == "f" for a in args):
            raise TopDownRefuse(f"a compound term in a literal of {rel} in the cone of the question")
        if sign == "neg":
            key = "[" + ", ".join(self.expr(a) for a in args) + "]"
            if way == "input":
                pass
            elif way == "derived":
                if self.fl_ex:
                    self.line(f"ex_push({self.cur_fi}, {i}); let ex_r = {self.fname(rel, tuple(range(n)))}(td, {key}, &mut |_| true); ex_pop();")
                    self.line(f"if ex_r {{ {fail}; }}")
                else:
                    self.line(f"if {self.fname(rel, tuple(range(n)))}(td, {key}, &mut |_| true) {{ {fail}; }}")
            else:
                self.pat(rel, tuple(range(n)))
                self.note_read(rel, key, rule, i)
                self.line(f"if td.{self.holder(rel)}.has(&{key}) {{ {fail}; }}")
            self.steps(rule, i + 1, head_row, sink, first)
            return
        keypos = tuple(st[3])
        is_first = first is not None and i == self.first_index(rule, first)
        # a literal read from the worklist row
        if is_first:
            self.bind_row(args, "w", fail)
            self.steps(rule, i + 1, head_row, sink, first)
            return
        key = "[" + ", ".join(self.expr(args[j]) if j in keypos else "FREE" for j in range(n)) + "]"
        if way == "input":
            self.line(f"{fail};")
            return
        fresh = [j for j in range(n) if j not in keypos]
        if way in ("static", "fluent"):
            if not fresh:
                self.pat(rel, tuple(range(n)))
                self.note_read(rel, key, rule, i)
                self.line(f"if !td.{self.holder(rel)}.has(&{key}) {{ {fail}; }}")
                self.steps(rule, i + 1, head_row, sink, first)
                return
            ix = self.pat(rel, keypos)
            self.note_read(rel, key, rule, i)
            src = f"td.{self.holder(rel)}.get({ix}, &{key})" if keypos else f"td.{self.holder(rel)}.order.as_slice()"
            self.line(f"for row{i} in {src}.iter() {{")
            self.fail.append("continue")
            self.ind += 1
            self.bind_row(args, f"row{i}", "continue", only=fresh)
            self.steps(rule, i + 1, head_row, sink, first)
            self.ind -= 1
            self.fail.pop()
            self.line("}")
            return
        # a derived relation: a call, the rest of the clause the continuation (or, where only existence is asked, a test)
        fn = self.fname(rel, keypos)
        for j in self.head_compound.get(rel, ()):
            if j in keypos or args[j][0] != "v" or self.occurs(rule, args[j][1]) > 1:
                raise TopDownRefuse(f"{rel} gives a compound term at place {j}, which a clause that reads it uses or binds")
        if only_exists(rule, i) or not fresh:
            if self.fl_ex:
                self.line(f"ex_push({self.cur_fi}, {i}); let ex_r = {fn}(td, {key}, &mut |_| true); ex_pop();")
                self.line(f"if !ex_r {{ {fail}; }}")
            else:
                self.line(f"if !{fn}(td, {key}, &mut |_| true) {{ {fail}; }}")
            self.steps(rule, i + 1, head_row, sink, first)
            return
        if self.fl_ex:
            self.line(f"let ex_d = ex_push({self.cur_fi}, {i});")
            self.line(f"let ex_r = {fn}(td, {key}, &mut |row{i}| {{")
            self.fail.append("return false")
            self.ind += 1
            self.line("let _ex_g = ExCut::new(ex_d);")
        else:
            self.line(f"if {fn}(td, {key}, &mut |row{i}| {{")
            self.fail.append("return false")
            self.ind += 1
        self.bind_row(args, f"row{i}", "return false", only=fresh)
        self.steps(rule, i + 1, head_row, sink, first)
        self.line("false")
        self.ind -= 1
        self.fail.pop()
        if self.fl_ex:
            self.line("});")
            self.line("ex_pop(); if ex_r { return true; }")
        else:
            self.line("}) { return true; }")

    def note_read(self, rel: str, key: str, rule=None, i: int = 0):
        """Pass (27): before a read of a fluent of the state, the statement that records it (the fluent's number and the places as asked); nothing for a relation that does not read the state.
        Pass (28): the statement also carries the number of the function-clause that makes the read and the step of its join (which names the clause, the literal and the order the clause was joined in)."""
        if not self.fl_rec or not rel.startswith("true."):
            return
        no = self.fl_no.setdefault(rel, len(self.fl_no))
        if self.fl_ex and rule is not None:
            self.line(f"rd_note({no}, &{key}, {self.cur_fi}, {i});")
            return
        self.line(f"rd_note({no}, &{key});")

    def explain_func(self, c, mask: tuple, first, rule) -> int:
        """Pass (28): the number of the function-clause written next: a clause of the description, the head places a call binds, the order its literals are joined in (the places of the literals in the order written) and whether it is a clause of a worklist.
        The clause is numbered by its text, in the order the clauses are first met; the tables of the texts and of the function-clauses are constants of the module."""
        text = rule.clause.text
        if text not in self.ex_clause:
            self.ex_clause[text] = len(self.ex_clause)
            self.ex_lits[self.ex_clause[text]] = [render_literal(x) for x in rule.clause.body]
        order = tuple(next(j for j, x in enumerate(rule.clause.body) if x is y) for y in rule.order)
        n = self.d.arity.get(rule.head_rel, 0)
        key = (self.ex_clause[text], "".join("1" if j in mask else "0" for j in range(n)), order, first is not None)
        if key not in self.ex_func:
            self.ex_func[key] = len(self.ex_func)
        return self.ex_func[key]

    def occurs(self, rule, v: int) -> int:
        """How many times the variable of this number stands in the head and the literals of the rule."""
        n = sum(1 for t in rule.head if v in comp_vars(t))
        for s in rule.steps:
            if s[0] == "distinct":
                n += (v in comp_vars(s[1])) + (v in comp_vars(s[2]))
            else:
                n += sum(1 for a in s[2] if v in comp_vars(a))
        return n

    def first_index(self, rule, first) -> int:
        return rule.rec_steps[0]

    def bind_row(self, args, row: str, fail: str, only=None):
        """Take the places of a row into the variables of a literal: a variable met for the first time is set, one already set (in the key, or earlier in this literal) is compared; a constant is compared."""
        for j, a in enumerate(args):
            if only is not None and j not in only:
                continue
            if a[0] == "v":
                if a[1] in self.bound:
                    self.line(f"if {row}[{j}] != v{a[1]} {{ {fail}; }}")
                else:
                    self.bound.add(a[1])
                    self.line(f"let v{a[1]} = {row}[{j}];")
            elif a[0] == "c":
                self.line(f"if {row}[{j}] != {self.expr(a)} {{ {fail}; }}")

    def holder(self, rel: str) -> str:
        return ("fl_" if rel.startswith("true.") else "st_") + self.d.rid[rel][2:]

    # ---- the whole
    def build(self, roots: list) -> dict:
        d = self.d
        # the relations of the cone, and where a clause gives a compound term
        cone = d.cone(roots)
        for rel in sorted(cone):
            if self.how(rel) != "derived":
                continue
            for c in self.clauses_of(rel):
                for j, t in enumerate(c.head[1]):
                    if isinstance(t, tuple):
                        self.head_compound.setdefault(rel, set()).add(j)
        for rel in sorted(cone):
            if self.how(rel) == "derived" and self.is_recursive(rel):
                n = d.arity.get(rel, 0)
                keep = set(range(n))
                for c in self.clauses_of(rel):
                    for lit in (x for x in c.body if x[0] == "pos" and x[1] == rel):
                        keep &= {j for j in range(n) if GDL.is_var(c.head[1][j]) and lit[2][j] == c.head[1][j]}
                self.keeps[rel] = tuple(sorted(keep))
        for rel in roots:
            self.fname(rel, ())
        for rel in sorted(cone):                         # every relation of the cone asked with no place bound too, for the tests (`ask_td`)
            if self.how(rel) == "derived":
                self.fname(rel, ())
        for rel, mask in self.extra:                     # pass (26): the functions the support filter calls
            self.fname(rel, mask)
        while self.todo:
            rel, mask = self.todo.pop(0)
            if self.how(rel) != "derived":
                continue
            self.texts.append(self.make(rel, mask))
        return {"cone": cone}


def w_topdown_tables(w, d, plan, td: "TopDown"):
    """The relations the functions read as Dense tables with the places they are asked by, the struct `Td` that holds them, and the functions."""
    sorts = d.prog.sorts
    used = sorted(td.used)
    tabs = sorted({s for r in used for s in (d.prog.signature(r) or ())})
    for r in used:
        sig = d.prog.signature(r)
        n = d.arity.get(r, 0)
        if sig is None or len(sig) != n or not all(s in sorts and sorts[s] and all(isinstance(m, str) for m in sorts[s]) and len(sorts[s]) < NONE for s in sig):
            raise TopDownRefuse(f"{r} has no signature of sorts of constants")
        size = [len(sorts[s]) for s in sig]
        space = 1
        for x in size:
            space *= x
        if space > DENSE_SPACE:
            raise TopDownRefuse(f"{r} has {space} possible rows")
        for pl in td.pats[r]:
            keys = 1
            for p in pl:
                keys *= size[p]
            if keys > DENSE_KEYS:
                raise TopDownRefuse(f"{r} asked by {pl} needs {keys} buckets")
    for si, s in enumerate(tabs):
        table = [65535] * len(d.syms)
        for j, m in enumerate(sorts[s]):
            table[d.sym_id[m]] = j
        w(f"static TDO{si}: [u16; {len(d.syms)}] = [{', '.join(map(str, table))}];")
    for r in used:
        sig = d.prog.signature(r)
        w(f"static TDS_{d.rid[r][2:]}: &[&[u16]] = &[{', '.join('&TDO%d' % tabs.index(s) for s in sig)}];")
        w(f"const TDR_{d.rid[r][2:]}: &[usize] = &[{', '.join(str(len(sorts[s])) for s in sig)}];")
        w(f"const TDK_{d.rid[r][2:]}: &[&[usize]] = &[{', '.join('&[' + ', '.join(map(str, pl)) + ']' for pl in td.pats[r])}];")
    w("pub struct Td { pub overflow: std::sync::atomic::AtomicBool, " + ", ".join(f"pub {td.holder(r)}: Dense<{d.arity[r]}>" for r in used) + " }")
    w("impl Td { pub fn new() -> Td { Td { overflow: std::sync::atomic::AtomicBool::new(false), " + ", ".join(f"{td.holder(r)}: Dense::new(TDS_{d.rid[r][2:]}, TDR_{d.rid[r][2:]}, TDK_{d.rid[r][2:]})" for r in used) + " } } }")
    for t in td.texts:
        w(t)


class SupportRefuse(Exception):
    """The shape of pass (26) is not found for a relation, with the reason."""


class SupportFinder:
    """PASS (26), the finding (module text, (26)). Reads the clauses as written and the dependency graph; names no relation. `found`: one dict for each conforming test R (`shape`); `refused`: {relation: why}."""

    def __init__(self, d: "Description", demanded=()):
        self.d = d
        self.demanded = set(demanded)
        self.cl: dict = {}
        for comp, rules in d.strata:
            for r in rules:
                self.cl.setdefault(r.clause.head[0], []).append(r.clause)
        self.found: list = []
        self.refused: dict = {}

    # ---- small helpers over the clauses
    def derived(self, rel: str) -> bool:
        return rel in self.cl and not rel.startswith("true.") and rel not in INPUT_RELS

    @staticmethod
    def flat(args) -> bool:
        return all(not isinstance(a, tuple) for a in args)

    def view_of(self, h: str, c):
        """The relation a clause of h is a view of (one positive literal, the head's variables distinct and the same, in the same order), else None."""
        if len(c.body) != 1 or c.body[0][0] != "pos":
            return None
        _s, v, args = c.body[0]
        hd = c.head[1]
        if v == h or tuple(args) != tuple(hd) or len(set(hd)) != len(hd) or not all(GDL.is_var(a) for a in hd):
            return None
        return v

    def copy_clause(self, c, m: int):
        """-> (B, guards, negs) when the clause is a COPY clause with a key of m places, else None; SupportRefuse when it has the form but breaks a condition."""
        hd = c.head[1]
        key, row = hd[:m], hd[m:]
        B = None
        for lit in c.body:
            if lit[0] == "pos" and lit[1].startswith("true.") and tuple(lit[2]) == tuple(row) and len(row) == self.d.arity.get(lit[1], -1) and len(set(row)) == len(row) and all(GDL.is_var(a) for a in row):
                B = lit[1]
                break
        if B is None:
            return None
        if len(set(key)) != m or not all(GDL.is_var(a) for a in key):
            raise SupportRefuse(f"a copy clause whose key places are not distinct variables: {c.text}")
        guards, negs = [], []
        for lit in c.body:
            if lit[0] == "distinct":
                raise SupportRefuse(f"a test distinct in a copy clause: {c.text}")
            sign, rel, args = lit
            if not self.flat(args):
                raise SupportRefuse(f"a compound term in a copy clause: {c.text}")
            if sign == "pos" and rel == B and tuple(args) == tuple(row):
                continue
            avars = {a for a in args if GDL.is_var(a)}
            if sign == "pos":
                if avars & set(row):
                    raise SupportRefuse(f"a positive literal besides the state relation holds a row variable: {c.text}")
                if not avars & set(key):
                    raise SupportRefuse(f"a positive literal of a copy clause holds no key variable: {c.text}")
                guards.append((rel, tuple(args)))
            else:
                if avars - set(key) - set(row):
                    raise SupportRefuse(f"a negated literal of a copy clause names a variable that is neither key nor row: {c.text}")
                negs.append((rel, tuple(args)))
        return B, guards, negs

    def hypothetical(self, h: str) -> dict:
        """-> {views: [(V, 'copy' | 'add', [clauses])], B, m, p} when h has the shape H (module text, (26)); SupportRefuse otherwise."""
        d = self.d
        if not self.derived(h) or d.static[h]:
            raise SupportRefuse("not a relation that reads the state")
        n = d.arity[h]
        views = []
        for c in self.cl[h]:
            v = self.view_of(h, c)
            if v is None:
                raise SupportRefuse(f"a clause that is not a view: {c.text}")
            if not self.derived(v) or d.arity.get(v) != n:
                raise SupportRefuse(f"the view {v} is not a relation of the same places")
            views.append(v)
        info = {"views": [], "B": None, "m": None, "p": None}
        for v in sorted(set(views)):
            kinds = set()
            for c in self.cl[v]:
                got = None
                for m in sorted({n - d.arity[lit[1]] for lit in c.body if lit[0] == "pos" and lit[1].startswith("true.") and 0 < d.arity[lit[1]] < n}):
                    got = self.copy_clause(c, m)
                    if got is not None:
                        if info["m"] not in (None, m) or info["B"] not in (None, got[0]):
                            raise SupportRefuse(f"two ways of copying a state relation: {c.text}")
                        info["m"], info["p"], info["B"] = m, n - m, got[0]
                        break
                kinds.add("copy" if got is not None else "add")
            if len(kinds) != 1:
                raise SupportRefuse(f"the view {v} has clauses of both forms")
            info["views"].append((v, kinds.pop(), self.cl[v]))
        if info["B"] is None:
            raise SupportRefuse("no clause copies a state relation")
        m = info["m"]
        for v, form, cls in info["views"]:
            for c in cls:
                if form == "copy":
                    if self.copy_clause(c, m) is None:
                        raise SupportRefuse(f"a copy clause that does not copy {info['B']} at the row: {c.text}")
                else:
                    if any(isinstance(t, tuple) for t in c.head[1]):
                        raise SupportRefuse(f"a compound term in an add clause: {c.text}")
                    if any(lit[0] != "distinct" and lit[1] == info["B"] and tuple(lit[2]) == tuple(c.head[1][m:]) for lit in c.body):
                        raise SupportRefuse(f"an add clause that reads the state relation at its row: {c.text}")
        return info

    def dependents(self, h: str) -> set:
        """The relations that depend on h (the relations made from it), h not included."""
        deps = self.d.prog.deps
        out: set = set()
        grew = True
        while grew:
            grew = False
            for r, ds in deps.items():
                if r not in out and r != h and (h in ds or ds & out):
                    out.add(r)
                    grew = True
        return out

    def conform(self, R: str, h: str, m: int, hd: set):
        """Follow the calls from R with the key found as the module text says; -> (key places of each relation reached, the guards [(relation, places, sign)]); SupportRefuse when a clause reads the key outside H."""
        d = self.d
        kp = {R: tuple(range(m))}
        queue = [R]
        met = {R}
        guards: list = []
        while queue:
            x = queue.pop(0)
            for c in self.cl[x]:
                kv = [c.head[1][j] for j in kp[x]]
                if len(set(kv)) != len(kv) or not all(GDL.is_var(a) for a in kv):
                    raise SupportRefuse(f"the key places of {x} are not distinct variables: {c.text}")
                kvs = set(kv)
                for lit in c.body:
                    if lit[0] == "distinct":
                        if any(isinstance(a, tuple) or a in kvs for a in (lit[1], lit[2])):
                            raise SupportRefuse(f"a test distinct holds a key variable: {c.text}")
                        continue
                    sign, rel, args = lit
                    if not self.flat(args):
                        raise SupportRefuse(f"a compound term in a literal of the cone: {c.text}")
                    at = tuple(j for j, a in enumerate(args) if a in kvs)
                    got = [args[j] for j in at]
                    if rel == h:
                        if at != tuple(range(m)) or got != kv:
                            raise SupportRefuse(f"a literal of {h} whose first places are not the key in order: {c.text}")
                    elif rel in hd:
                        if got != kv:
                            raise SupportRefuse(f"a literal of {rel}, made from {h}, that does not hold the whole key in order: {c.text}")
                        if kp.setdefault(rel, at) != at:
                            raise SupportRefuse(f"{rel} holds the key at two sets of places")
                        if rel not in met:
                            met.add(rel)
                            queue.append(rel)
                    elif at:
                        if d.static[rel]:
                            raise SupportRefuse(f"a relation that does not read the state holds a key variable: {c.text}")
                        if got != kv:
                            raise SupportRefuse(f"a guard {rel} that does not hold the whole key in order: {c.text}")
                        if (rel, at, sign) not in guards:
                            guards.append((rel, at, sign))
        return kp, guards

    def needs_guard(self, x: str, g: tuple, hd: set, kp: dict, memo: dict, active: frozenset) -> bool:
        """Whether every clause of x holds a positive literal of the guard g = (relation, key places) or of a relation of hd that needs it (a clause that reads x itself is taken to need it where the other clauses of x do)."""
        if x in memo:
            return memo[x]
        if x in active:
            return True
        act = active | {x}
        ok = True
        for c in self.cl[x]:
            kvs = {c.head[1][j] for j in kp[x]}
            good = False
            for lit in c.body:
                if lit[0] != "pos":
                    continue
                if lit[1] == g[0] and tuple(j for j, a in enumerate(lit[2]) if a in kvs) == g[1]:
                    good = True
                    break
                if lit[1] == x:
                    good = True                 # the recursive clause: its rows come from the rows of the clauses that do not read x
                    break
                if lit[1] in hd and self.needs_guard(lit[1], g, hd, kp, memo, act):
                    good = True
                    break
            if not good:
                ok = False
                break
        memo[x] = ok
        return ok

    def shape(self, R: str, h: str, info: dict, hd: set, kp: dict, guards: list) -> dict:
        d = self.d
        m, p = info["m"], info["p"]
        # the differing places D(K): the add views, and the negated literals of the copy clauses
        dparts = []
        gstars = None
        for v, form, cls in info["views"]:
            if form == "add":
                dparts.append(("add", v))
                continue
            for c in cls:
                B, cguards, negs = self.copy_clause(c, m)
                key, row = c.head[1][:m], c.head[1][m:]
                mine = {(rel, tuple(j for j, a in enumerate(args) if a in set(key))) for rel, args in cguards if [a for a in args if a in set(key)] == list(key)}
                gstars = mine if gstars is None else gstars & mine
                for rel, args in negs:
                    if not self.derived(rel):
                        raise SupportRefuse(f"a negated literal of a copy clause over {rel}, which is not made by clauses: {c.text}")
                    amap = []
                    for a in args:
                        if a in key:
                            amap.append(("k", key.index(a)))
                        elif a in row:
                            amap.append(("r", row.index(a)))
                        else:
                            amap.append(("c", a))
                    dparts.append(("neg", rel, tuple(amap)))
        cands = sorted(g for g in (gstars or set()) if not d.static[g[0]] and g[0] not in self.demanded)
        memo: dict = {}
        gstar = None
        for g in cands:
            memo.clear()
            if self.needs_guard(R, g, hd, kp, memo, frozenset()):
                gstar = g
                break
        if gstar is None:
            raise SupportRefuse("no guard that every copy clause and every clause of the test hold: the keys of the test are not known")
        for rel, at, sign in guards:
            if rel in self.demanded:
                raise SupportRefuse(f"the guard {rel} is answered on demand, no rows of it are made")
        uniq = []
        for rel, at, sign in guards:
            if (rel, at) not in uniq:
                uniq.append((rel, at))
        if gstar not in uniq:
            uniq.append(gstar)
        return {"R": R, "H": h, "B": info["B"], "m": m, "p": p, "kp": kp, "hd": sorted(kp), "views": [(v, form) for v, form, _c in info["views"]],
                "dparts": dparts, "guards": uniq, "gstar": gstar}

    def run(self) -> None:
        d = self.d
        for h in sorted(d.nodes):
            if not self.derived(h) or not self.cl.get(h):
                continue
            vs = [self.view_of(h, c) for c in self.cl[h]]
            if not any(v is not None and self.derived(v) for v in vs):
                continue                                # no clause is a view of a relation made by clauses: not a contender for H
            if not all(v is not None for v in vs):
                self.refused[h] = "a clause that is not a view: " + next(c.text for c, v in zip(self.cl[h], vs) if v is None)
                continue
            try:
                info = self.hypothetical(h)
            except SupportRefuse as e:
                self.refused[h] = str(e)
                continue
            m = info["m"]
            hd = self.dependents(h)
            tops = sorted(r for r in hd if d.arity.get(r) == m and r in self.cl)
            if not tops:
                self.refused[h] = "no relation made from it has exactly the places of the key"
                continue
            got = []
            for R in tops:
                try:
                    kp, guards = self.conform(R, h, m, hd)
                    got.append(self.shape(R, h, info, hd, kp, guards))
                except SupportRefuse as e:
                    self.refused[R] = str(e)
            # a test that another test of the same key reaches is part of that one: only the outermost are kept
            for s in got:
                if not any(s["R"] != o["R"] and s["R"] in o["kp"] for o in got):
                    self.found.append(s)


def prepare_reads(d, plan, tdg, reads) -> None:
    """Pass (27), before the top-down functions are made: for each (relation, k) the caller gave, ask the top-down maker for the function with the first k places bound, and switch the recording of fluent reads on; a relation that cannot be made is refused with the reason."""
    for rel, k in reads:
        if rel not in d.arity or rel not in d.prog.nodes:
            plan.reads_refused[rel] = "the description has no such relation"
        elif tdg.how(rel) != "derived":
            plan.reads_refused[rel] = "not a relation derived from clauses"
        elif not (0 <= k <= d.arity[rel]):
            plan.reads_refused[rel] = f"the key of {k} places does not fit its {d.arity[rel]} places"
        else:
            plan.reads.append((rel, k))
            tdg.extra.append((rel, tuple(range(k))))
    tdg.fl_rec = bool(plan.reads)
    tdg.fl_ex = tdg.fl_rec and "explain" in plan.passes


def explain_prelude(d, plan, tdg) -> list:
    """Pass (28): the buffer where a read of a fluent is recorded WITH the function-clause and the step that made it and the chain of calls above it, the stack of calls, and the tables that name them: the text of each clause, the text of each of its literals in the order written, and for each function-clause its clause,
    the head places a call binds and the order its literals are joined in."""
    width = max([d.arity.get(r, 0) for r in tdg.fl_no] + [1])
    names = sorted(tdg.fl_no, key=lambda r: tdg.fl_no[r])
    texts = sorted(tdg.ex_clause, key=lambda t: tdg.ex_clause[t])
    funcs = sorted(tdg.ex_func, key=lambda k: tdg.ex_func[k])
    out = [f"pub const RD_W: usize = {width};",
           "type ExFrame = (u16, u16);",
           "thread_local! { static RD_ON: std::sync::atomic::AtomicBool = const { std::sync::atomic::AtomicBool::new(false) }; static RD_REC: std::sync::Mutex<Vec<(u16, [u32; RD_W], u16, u16, u32)>> = const { std::sync::Mutex::new(Vec::new()) }; static EX_STACK: std::cell::RefCell<Vec<ExFrame>> = const { std::cell::RefCell::new(Vec::new()) }; static EX_CHAINS: std::cell::RefCell<(Vec<Vec<ExFrame>>, HashMap<Vec<ExFrame>, u32>)> = std::cell::RefCell::new((Vec::new(), HashMap::new())); }",
           "/// Pass (28): a call of a derived relation from a literal (function-clause, step) puts a frame on the stack and takes it off when the call returns; the rest of the clause after the literal runs inside the callee (a closure), so it first cuts the stack back to the depth of its own call (`ExCut`) and puts it back on leaving.",
           "#[inline] fn ex_push(fi: u16, st: u16) -> usize { EX_STACK.with(|s| { let mut s = s.borrow_mut(); let d = s.len(); s.push((fi, st)); d }) }",
           "#[inline] fn ex_pop() { EX_STACK.with(|s| { s.borrow_mut().pop(); }) }",
           "struct ExCut(Vec<ExFrame>);",
           "impl ExCut { #[inline] fn new(d: usize) -> ExCut { ExCut(EX_STACK.with(|s| { let mut s = s.borrow_mut(); if s.len() > d { s.split_off(d) } else { Vec::new() } })) } }",
           "impl Drop for ExCut { #[inline] fn drop(&mut self) { EX_STACK.with(|s| s.borrow_mut().extend(self.0.drain(..))); } }",
           "/// Pass (28): the number of the chain of calls above the read being made now (the chains are kept once each, by number, in this thread).",
           "fn ex_chain() -> u32 { EX_STACK.with(|s| { let s = s.borrow(); EX_CHAINS.with(|c| { let mut c = c.borrow_mut(); if let Some(&i) = c.1.get(&*s) { return i; } let i = c.0.len() as u32; c.0.push(s.clone()); c.1.insert(s.clone(), i); i }) }) }",
           "/// Pass (28): the chain of calls of a number (the function-clause and step of each call, outermost first); the numbers are those of this thread.",
           "pub fn expl_chain(id: u32) -> Vec<(u16, u16)> { EX_CHAINS.with(|c| c.borrow().0[id as usize].clone()) }",
           "/// Pass (28): record one read of a fluent of the state (its number, the places as asked, FREE where the read leaves a place free) with the function-clause and the step that make it and the chain of calls above it, while a question asks for the reads.",
           "#[inline] fn rd_note(f: u16, key: &[u32], fi: u16, st: u16) { if RD_ON.with(|o| o.load(std::sync::atomic::Ordering::Relaxed)) { let mut q = [FREE; RD_W]; q[..key.len()].copy_from_slice(key); let ch = ex_chain(); RD_REC.with(|r| r.lock().unwrap().push((f, q, fi, st, ch))); } }",
           "/// The fluents in the order of their numbers, and the number of places of each.",
           "pub const READS_FLUENTS: &[&str] = &[" + ", ".join(f'"{r[len("true."):]}"' for r in names) + "];",
           "pub const READS_ARITY: &[usize] = &[" + ", ".join(str(d.arity[r]) for r in names) + "];",
           "/// The relations whose answers come with their reads, each with the number of places of its key.",
           "pub const READS: &[(&str, usize)] = &[" + ", ".join(f'("{r}", {k})' for r, k in plan.reads) + "];",
           "/// Pass (28): the text of each clause that made a read, by its number.",
           "pub const EXPL_CLAUSES: &[&str] = &[" + ", ".join(rs_str(t) for t in texts) + "];",
           "/// Pass (28): for each clause, the text of its literals in the order written.",
           "pub const EXPL_LITS: &[&[&str]] = &[" + ", ".join("&[" + ", ".join(rs_str(x) for x in tdg.ex_lits[i]) + "]" for i in range(len(texts))) + "];",
           "/// Pass (28): for each function-clause (the number a read and a call carry): the number of its clause, which head places the call binds (1 bound), the places of its literals, in the order written, in the order they are joined, and whether it is a clause of a worklist (a walk).",
           "pub const EXPL_FUNCS: &[(u16, &str, &[u16], bool)] = &[" + ", ".join(f'({k[0]}, "{k[1]}", &[{", ".join(str(x) for x in k[2])}], {"true" if k[3] else "false"})' for k in funcs) + "];"]
    return out


def reads_prelude(d, plan, tdg) -> list:
    """Pass (27): the buffer where a read of a fluent is recorded, the recording function, the numbers of the fluents and the relations given."""
    if tdg.fl_ex:
        return explain_prelude(d, plan, tdg)
    width = max([d.arity.get(r, 0) for r in tdg.fl_no] + [1])
    names = sorted(tdg.fl_no, key=lambda r: tdg.fl_no[r])
    return [f"pub const RD_W: usize = {width};",
            "thread_local! { static RD_ON: std::sync::atomic::AtomicBool = const { std::sync::atomic::AtomicBool::new(false) }; static RD_REC: std::sync::Mutex<Vec<(u16, [u32; RD_W])>> = const { std::sync::Mutex::new(Vec::new()) }; }",
            "/// Pass (27): record one read of a fluent of the state (its number and the places as asked, FREE where the read leaves a place free), while a question asks for the reads.",
            "#[inline] fn rd_note(f: u16, key: &[u32]) { if RD_ON.with(|o| o.load(std::sync::atomic::Ordering::Relaxed)) { let mut q = [FREE; RD_W]; q[..key.len()].copy_from_slice(key); RD_REC.with(|r| r.lock().unwrap().push((f, q))); } }",
            "/// The fluents in the order of their numbers, and the number of places of each.",
            "pub const READS_FLUENTS: &[&str] = &[" + ", ".join(f'"{r[len("true."):]}"' for r in names) + "];",
            "pub const READS_ARITY: &[usize] = &[" + ", ".join(str(d.arity[r]) for r in names) + "];",
            "/// The relations whose answers come with their reads, each with the number of places of its key.",
            "pub const READS: &[(&str, usize)] = &[" + ", ".join(f'("{r}", {k})' for r, k in plan.reads) + "];"]


def explain_method(d, plan, tdg) -> list:
    """Pass (28): `Game::ask_explain`, the rows of a relation at a key with every read of a fluent the walk made, each with the function-clause and step that made it and the number of the chain of calls above it: a small proof of the answer. `ask_reads` is the same walk with the tags dropped."""
    out = ["    /// Pass (28): the rows of one of the relations given at a key (as `ask_reads`) with, for each read of a fluent, (the fluent's number, the places as asked, the function-clause, the step, the number of the chain of calls above the read: `expl_chain`); None as `ask_reads`.",
           "    pub fn ask_explain(&mut self, s: &State, rel: &str, key: &[u32]) -> Option<(Vec<Vec<u32>>, Vec<(u16, Vec<u32>, u16, u16, u32)>)> {",
           "        self.load_td(s);",
           "        RD_REC.with(|r| r.lock().unwrap().clear());",
           "        EX_STACK.with(|s| s.borrow_mut().clear());",
           "        RD_ON.with(|o| o.store(true, std::sync::atomic::Ordering::Relaxed));",
           "        let mut v: Vec<Vec<u32>> = vec![];",
           "        let known = match rel {"]
    for rel, k in plan.reads:
        n = d.arity[rel]
        out.append(f'            "{rel}" if key.len() == {k} => {{ let mut a = [FREE; {n}]; a[..{k}].copy_from_slice(key); {tdg.fname(rel, tuple(range(k)))}(&self.td, a, &mut |row| {{ v.push(row.to_vec()); false }}); true }}')
    out += ["            _ => false,",
            "        };",
            "        RD_ON.with(|o| o.store(false, std::sync::atomic::Ordering::Relaxed));",
            "        let rec: Vec<(u16, [u32; RD_W], u16, u16, u32)> = RD_REC.with(|r| std::mem::take(&mut *r.lock().unwrap()));",
            "        let over = self.td.overflow.load(std::sync::atomic::Ordering::Relaxed);",
            "        self.clear_td();",
            "        if !known || over { return None; }",
            "        v.sort(); v.dedup();",
            "        Some((v, rec.into_iter().map(|(f, q, c, l, ch)| (f, q[..READS_ARITY[f as usize]].to_vec(), c, l, ch)).collect()))",
            "    }",
            "    /// Pass (27), the same walk: `ask_explain` with the tags left off.",
            "    pub fn ask_reads(&mut self, s: &State, rel: &str, key: &[u32]) -> Option<(Vec<Vec<u32>>, Vec<(u16, Vec<u32>)>)> {",
            "        let (v, r) = self.ask_explain(s, rel, key)?;",
            "        Some((v, r.into_iter().map(|(f, q, _, _, _)| (f, q)).collect()))",
            "    }"]
    return out


def reads_method(d, plan, tdg) -> list:
    """Pass (27): `Game::ask_reads`, the rows of a relation at a key with the fluent facts the walk read."""
    if tdg.fl_ex:
        return explain_method(d, plan, tdg)
    out = ["    /// Pass (27): the rows of one of the relations given at a key (its first places bound), by the top-down way, with every read of a fluent of the state the walk made (module text, (27)); None when the relation is not one given or the walk overflowed.",
           "    pub fn ask_reads(&mut self, s: &State, rel: &str, key: &[u32]) -> Option<(Vec<Vec<u32>>, Vec<(u16, Vec<u32>)>)> {",
           "        self.load_td(s);",
           "        RD_REC.with(|r| r.lock().unwrap().clear());",
           "        RD_ON.with(|o| o.store(true, std::sync::atomic::Ordering::Relaxed));",
           "        let mut v: Vec<Vec<u32>> = vec![];",
           "        let known = match rel {"]
    for rel, k in plan.reads:
        n = d.arity[rel]
        out.append(f'            "{rel}" if key.len() == {k} => {{ let mut a = [FREE; {n}]; a[..{k}].copy_from_slice(key); {tdg.fname(rel, tuple(range(k)))}(&self.td, a, &mut |row| {{ v.push(row.to_vec()); false }}); true }}')
    out += ["            _ => false,",
            "        };",
            "        RD_ON.with(|o| o.store(false, std::sync::atomic::Ordering::Relaxed));",
            "        let rec: Vec<(u16, [u32; RD_W])> = RD_REC.with(|r| std::mem::take(&mut *r.lock().unwrap()));",
            "        let over = self.td.overflow.load(std::sync::atomic::Ordering::Relaxed);",
            "        self.clear_td();",
            "        if !known || over { return None; }",
            "        v.sort(); v.dedup();",
            "        Some((v, rec.into_iter().map(|(f, q)| (f, q[..READS_ARITY[f as usize]].to_vec())).collect()))",
            "    }"]
    return out


def support_prelude(plan) -> list:
    """The thread-local buffers where a function of a hypothetical relation records the places it was asked about (pass 26), one for each hypothetical relation, and the counters of the Game."""
    out = ["#[derive(Clone, Copy, Default, Debug)]",
           "pub struct SpCount { pub states: u64, pub keys: u64, pub tests: u64, pub answered: u64, pub noref: u64 }",
           "/// One key of one question (for the tests that read the support words): the key, the places it differs at (patterns, FREE: any), the places its test read (blank for a key answered from a reference), the answer, and how (0 tested, 1 tested and kept as the reference, 2 answered from the reference).",
           "#[derive(Clone, Debug)]",
           "pub struct SpLog { pub key: Vec<u32>, pub d: Vec<Vec<u32>>, pub s: Vec<Vec<u32>>, pub ans: bool, pub how: u8 }",
           "fn sp_meet<const P: usize>(a: &[u32; P], b: &[u32; P]) -> bool { (0..P).all(|j| a[j] == FREE || b[j] == FREE || a[j] == b[j]) }",
           "fn sp_disjoint<const P: usize>(d: &[[u32; P]], s: &[[u32; P]]) -> bool { !d.iter().any(|x| s.iter().any(|y| sp_meet(x, y))) }"]
    for hi, h in enumerate(plan.support_h):
        p = next(s["p"] for s in plan.support if s["H"] == h)
        out.append(f"thread_local! {{ static SP_ON{hi}: std::sync::atomic::AtomicBool = const {{ std::sync::atomic::AtomicBool::new(false) }}; static SP_REC{hi}: std::sync::Mutex<Vec<[u32; {p}]>> = const {{ std::sync::Mutex::new(Vec::new()) }}; }}")
    return out


def support_free(d, plan, tdg) -> list:
    """For each conforming test, the function that gives D(K) as patterns over the row (pass 26)."""
    out = []
    for i, s in enumerate(plan.support):
        m, p = s["m"], s["p"]
        out.append(f"/// Pass (26): the places where the world of the key differs from the state, as patterns over the row (FREE: any), for the test {s['R']}.")
        out.append(f"fn sp_d{i}(td: &Td, key: &[u32; {m}], out: &mut Vec<[u32; {p}]>) {{")
        for part in s["dparts"]:
            if part[0] == "add":
                fn = tdg.fname(part[1], tuple(range(m)))
                a = ", ".join([f"key[{j}]" for j in range(m)] + ["FREE"] * p)
                out.append(f"    {fn}(td, [{a}], &mut |row| {{ let mut q = [FREE; {p}]; q.copy_from_slice(&row[{m}..]); out.push(q); false }});")
            else:
                _t, rel, amap = part
                mask = tuple(j for j, (form, _v) in enumerate(amap) if form in ("k", "c"))
                fn = tdg.fname(rel, mask)
                a = ", ".join(f"key[{v}]" if form == "k" else (str(d.sym_id[v]) if form == "c" else "FREE") for form, v in amap)
                body = []
                for j, (form, v) in enumerate(amap):
                    if form == "r":
                        body.append(f"{{ let x = row[{j}]; if q[{v}] != FREE && q[{v}] != x {{ ok = false; }} else {{ q[{v}] = x; }} }}")
                rb = tuple(sorted({v for form, v in amap if form == "r"}))
                hold = tdg.holder(s["B"])
                if rb:
                    grab = f"for brow in td.{hold}.get({tdg.pat(s['B'], rb)}, &q).iter() {{ out.push(*brow); }}"
                else:
                    grab = f"for brow in td.{hold}.order.as_slice().iter() {{ out.push(*brow); }}"
                out.append(f"    {fn}(td, [{a}], &mut |row| {{ let mut q = [FREE; {p}]; let mut ok = true; {' '.join(body)} if ok {{ {grab} }} false }});")
        out.append("}")
    return out


def support_methods(d, plan, tdg, ro, gated, has_dw) -> list:
    """For each conforming test, `Game::run_support<i>` (pass 26), the piece that is run in place of the test's own piece, and the counters."""
    out = ["    /// Pass (26): the counters of the support filter (on under the profile switch only): (states, keys, tests run, keys answered from a reference, guard classes with no reference).",
           "    pub fn support_counts(&self) -> (u64, u64, u64, u64, u64) { let c = self.sp; (c.states, c.keys, c.tests, c.answered, c.noref) }",
           "    pub fn reset_support_counts(&mut self) { self.sp = SpCount::default(); }"]
    dwa = ", &mut self.dw" if has_dw else ""
    for i, s in enumerate(plan.support):
        m, p = s["m"], s["p"]
        hi = plan.support_h.index(s["H"])
        kR = next(k for k, (comp, _r) in enumerate(d.strata) if s["R"] in comp)
        fb = sorted({next(k for k, (comp, _r) in enumerate(d.strata) if x in comp) for x in s["kp"]})
        gs = s["guards"].index(s["gstar"])
        out.append(f"    /// Pass (26): the test {s['R']} for every key of the guard domain, answered once for the keys that leave the places its reference test read alone (module text, (26)).")
        out.append(f"    fn run_support{i}(&mut self, d: &mut Dy) -> (u64, u64, u64) {{")
        if kR in gated:
            out.append(f"        if !d.whole && off(d, {kR}) {{ return (0, 0, 0); }}")
        out.append(f"        let mut ent: Vec<([u32; {m}], u8, Vec<u32>)> = Vec::new();")
        for gi, (rel, at) in enumerate(s["guards"]):
            n = d.arity[rel]
            rest = [j for j in range(n) if j not in at]
            out.append(f"        for r in {ro('d', rel)}.iter() {{ ent.push(([{', '.join(f'r[{j}]' for j in at)}], {gi}, vec![{', '.join(f'r[{j}]' for j in rest)}])); }}")
        out.append("        ent.sort_unstable();")
        out.append("        let s = std::mem::take(&mut self.sp_s);")
        out.append("        self.clear_td(); self.load_td(&s);")
        out.append("        self.sp_s = s;")
        out.append("        let mut classes: HashMap<Vec<u32>, usize> = HashMap::new();")
        out.append(f"        let mut refs: Vec<Option<(Vec<[u32; {p}]>, bool)>> = Vec::new();")
        out.append(f"        let mut rows: Vec<[u32; {m}]> = Vec::new();")
        out.append("        let (mut tests, mut answered, mut keys) = (0u64, 0u64, 0u64);")
        out.append(f"        let mut dk: Vec<[u32; {p}]> = Vec::new();")
        out.append("        let (mut i0, mut over) = (0usize, false);")
        out.append("        while i0 < ent.len() {")
        out.append("            let key = ent[i0].0;")
        out.append("            let (mut i1, mut sig, mut has) = (i0, Vec::<u32>::new(), false);")
        out.append(f"            while i1 < ent.len() && ent[i1].0 == key {{ sig.push(ent[i1].1 as u32); sig.push(ent[i1].2.len() as u32); sig.extend_from_slice(&ent[i1].2); if ent[i1].1 == {gs} {{ has = true; }} i1 += 1; }}")
        out.append("            i0 = i1;")
        out.append("            if !has { continue; }")
        out.append("            keys += 1;")
        out.append("            let cn = classes.len();")
        out.append("            let c = *classes.entry(sig).or_insert(cn);")
        out.append("            if c == refs.len() { refs.push(None); }")
        out.append(f"            dk.clear(); sp_d{i}(&self.td, &key, &mut dk);")
        out.append("            if let Some((s0, ans)) = &refs[c] { if sp_disjoint(&dk, s0) { answered += 1; if let Some(l) = self.sp_log.as_mut() { l.push(SpLog { key: key.to_vec(), d: dk.iter().map(|x| x.to_vec()).collect(), s: vec![], ans: *ans, how: 2 }); } if *ans { rows.push(key); } continue; } }")
        out.append(f"            SP_REC{hi}.with(|r| r.lock().unwrap().clear()); SP_ON{hi}.with(|o| o.store(true, std::sync::atomic::Ordering::Relaxed));")
        fn = tdg.fname(s["R"], tuple(range(m)))
        out.append(f"            let ans = {fn}(&self.td, [{', '.join(f'key[{j}]' for j in range(m))}], &mut |_| true);")
        out.append(f"            SP_ON{hi}.with(|o| o.store(false, std::sync::atomic::Ordering::Relaxed));")
        out.append("            tests += 1;")
        out.append("            if self.td.overflow.load(std::sync::atomic::Ordering::Relaxed) { over = true; break; }")
        out.append(f"            let mut how = 0u8;")
        out.append(f"            if refs[c].is_none() || self.sp_log.is_some() {{ let s0: Vec<[u32; {p}]> = SP_REC{hi}.with(|r| r.lock().unwrap().clone()); if let Some(l) = self.sp_log.as_mut() {{ l.push(SpLog {{ key: key.to_vec(), d: dk.iter().map(|x| x.to_vec()).collect(), s: s0.iter().map(|x| x.to_vec()).collect(), ans, how: 0 }}); }} if refs[c].is_none() && sp_disjoint(&dk, &s0) {{ how = 1; refs[c] = Some((s0, if self.sp_plant {{ !ans }} else {{ ans }})); }} }}")
        out.append("            if how == 1 { if let Some(l) = self.sp_log.as_mut() { if let Some(x) = l.last_mut() { x.how = 1; } } }")
        out.append("            if ans { rows.push(key); }")
        out.append("        }")
        out.append("        self.clear_td();")
        out.append("        if over {")
        out.append("            self.td.overflow.store(false, std::sync::atomic::Ordering::Relaxed);")
        out.append(f"            for &k in SP_FALLBACK{i} {{ run_piece(&self.st, d, &mut self.tm{dwa}, k); }}")
        out.append("            return (0, 0, 0);")
        out.append("        }")
        out.append("        let mut made = 0u64;")
        out.append(f"        for k in rows.iter() {{ if d.{d.rid[s['R']]}.add(*k) {{ made += 1; }} }}")
        out.append("        if self.profiling { let c = &mut self.sp; c.states += 1; c.keys += keys; c.tests += tests; c.answered += answered; c.noref += refs.iter().filter(|r| r.is_none()).count() as u64; }")
        out.append("        (made, tests, answered)")
        out.append("    }")
    return out


def support_consts(d, plan) -> list:
    out = []
    for i, s in enumerate(plan.support):
        kR = next(k for k, (comp, _r) in enumerate(d.strata) if s["R"] in comp)
        fb = sorted({next(k for k, (comp, _r) in enumerate(d.strata) if x in comp) for x in s["kp"]})
        out.append(f"const SP_K{i}: usize = {kR};")
        out.append(f"/// The pieces the module without the pass runs for this test, in order (the fallback of a worklist that overflowed).")
        out.append(f"const SP_FALLBACK{i}: &[usize] = &[{', '.join(map(str, fb))}];")
    out.append("/// Pass (26): the tests the pass found, with the hypothetical relation each reads the key through and the state relation it is made from.")
    out.append("pub const SUPPORT: &[(&str, &str, &str)] = &[" + ", ".join(f'("{s["R"]}", "{s["H"]}", "{s["B"]}")' for s in plan.support) + "];")
    return out


def prepare_support(d, plan, tdg) -> None:
    """Pass (26), before the top-down functions are made: find the tests (SupportFinder), keep those a question reads, and tell the top-down maker which functions record and which it must make besides."""
    f = SupportFinder(d, plan.demanded)
    f.run()
    plan.support_refused = dict(f.refused)
    wanted = [x for x in QUESTION_RELS if x in d.prog.nodes] + [f"next.{x}" for x in d.fluents if f"next.{x}" in d.prog.nodes]
    attain = d.cone(wanted)
    keep = []
    for s in f.found:
        if s["R"] in attain:
            keep.append(s)
        else:
            plan.support_refused[s["R"]] = "no question reads it (the shape is there; nothing is changed)"
    plan.support = keep
    plan.support_h = sorted({s["H"] for s in keep})
    plan.support_reads = {s["R"]: {g[0] for g in s["guards"]} for s in keep}
    for s in keep:
        m = s["m"]
        for x in s["kp"]:
            tdg.late[x] = tuple(sorted(set(tdg.late.get(x, ())) | {s["H"]}))
        tdg.rec[s["H"]] = (plan.support_h.index(s["H"]), m, s["p"])
        tdg.extra.append((s["R"], tuple(range(m))))
        for part in s["dparts"]:
            if part[0] == "add":
                tdg.extra.append((part[1], tuple(range(m))))
            else:
                tdg.extra.append((part[1], tuple(j for j, (form, _v) in enumerate(part[2]) if form in ("k", "c"))))
                rb = tuple(sorted({v for form, v in part[2] if form == "r"}))
                if rb:
                    tdg.pat(s["B"], rb)                 # the rows of the state relation a negated literal names are read by an index on those places
                tdg.used.add(s["B"])


def rs_str(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def translate(text: str, module: str, passes=(), lemmas=(), ways=None, reads=()) -> str:
    """The Rust source of the description `text` (composed) as `pub mod <module>`, with the passes named and the lemmas read as rewrites (see the module text for the interface); with none, the plain translation."""
    src_text = text
    infos: list = []
    if lemmas:
        text, infos = rewrite_lemmas(text, lemmas)
    over: dict = {}
    if "asked" in passes:
        text, over = over_approximations(text)
    d = Description(text)
    d.flags = {i["flag"] for i in infos}
    d.over = over
    p = d.prog
    for needed in ("role",):
        if needed not in p.nodes:
            raise Untranslatable(f"the description has no `{needed}`")
    plan = Plan(d, passes) if passes else None
    demanded = plan.demanded if plan else set()
    dense = plan.dense if plan else {}
    bits = plan.bits if plan else {}
    keep = bool(plan and plan.keep)
    gated = plan.gated if plan else {}
    wd = plan.words if plan else {}
    cones = bool(plan and plan.cones)
    carry = bool(plan and plan.carry)
    has_dw = bool(bits) or keep
    pooled = bool(dense) or bool(bits) or keep
    out: list = []
    w = out.append
    w(f"// GENERATED by catspace/incubator/analyst/discovery/gdl_to_rust.py (plan 79.37). Do not edit: make it again.")
    w(f"// description-sha256: {sha(src_text.encode())}")
    w(f"// translator-sha256: {translator_hash()}")
    if plan:
        w(f"// passes: {','.join(plan.passes)}")
    head_at = len(out)
    for walk, ow in sorted(over.items()):
        w(f"// over-approximation: {ow} of {walk}")
    for i in infos:
        w(f"// lemma: {i['name']} {i['form']} {i['left']} -> {i['rep']}, flag {i['flag']}, {i['evidence']}")
    w(f"#[allow(dead_code, unused_variables, unused_mut, unused_imports, unused_parens, non_snake_case, non_upper_case_globals, clippy::all{', unreachable_code' if plan else ''})]")
    w(f"pub mod {module} {{")
    w((RUNTIME_KEEP.replace("pub goals: Vec<(u32, u32)> }", "pub goals: Vec<(u32, u32)>, pub have: u8 }") if cones else RUNTIME_KEEP) if keep else RUNTIME)
    if dense or (plan and plan.topdown):
        w(RUNTIME_DENSE)
    if pooled or any(r.startswith("init.") for r in d.nodes):
        w(RUNTIME_REL_KEEP_INDEX if wd else RUNTIME_REL_CLEAR)
    if wd:
        w(RUNTIME_WORDS)
    if bits:
        w(RUNTIME_BITS)
        w_bit_tables(w, d, plan)
    w(f"pub const SYMS: [&str; {len(d.syms)}] = [{', '.join(repr(s).replace(chr(39), chr(34)) for s in d.syms)}];")
    places = plan.places if plan else d.index_places()
    for r in d.nodes:
        if r in demanded or r in bits:
            continue
        pl = places[r]
        w(f"const KP{d.rid[r][1:]}: &[&[usize]] = &[{', '.join('&[' + ', '.join(map(str, t)) + ']' for t in pl)}];")
    if dense:
        used = sorted({s for sig in dense.values() for s in sig})
        for si, s in enumerate(used):
            table = [65535] * len(d.syms)
            for j, m in enumerate(p.sorts[s]):
                table[d.sym_id[m]] = j
            w(f"static ORD{si}: [u16; {len(d.syms)}] = [{', '.join(map(str, table))}];")
        for r in sorted(dense):
            sig = dense[r]
            w(f"static DS{d.rid[r][1:]}: &[&[u16]] = &[{', '.join('&ORD%d' % used.index(s) for s in sig)}];")
            w(f"const RX{d.rid[r][1:]}: &[usize] = &[{', '.join(str(len(p.sorts[s])) for s in sig)}];")
    # the state
    wsorts = sorted({s for sig, _w in wd.values() for s in sig})
    for si, s in enumerate(wsorts):
        table = [65535] * len(d.syms)
        for j, m in enumerate(p.sorts[s]):
            table[d.sym_id[m]] = j
        w(f"static WO{si}: [u16; {len(d.syms)}] = [{', '.join(map(str, table))}];")
    for f, (sig, nw) in sorted(wd.items()):
        # the row of each number of the fluent: the mixed radix of the ordinals of its places, the first place the most significant
        sizes = [len(p.sorts[s]) for s in sig]
        space = 1
        for x in sizes:
            space *= x
        rows = []
        for i in range(space):
            r, j = [0] * len(sig), i
            for q in range(len(sig) - 1, -1, -1):
                r[q] = d.sym_id[p.sorts[sig[q]][j % sizes[q]]]
                j //= sizes[q]
            rows.append("[" + ", ".join(map(str, r)) + "]")
        w(f"static DEC_{ident(f)}: [[u32; {len(sig)}]; {space}] = [{', '.join(rows)}];")
        w(f"static TAB_{ident(f)}: Tab<{len(sig)}> = Tab {{ ord: &[{', '.join('&WO%d' % wsorts.index(s) for s in sig)}], rad: &[{', '.join(map(str, sizes))}], dec: &DEC_{ident(f)} }};")
    n_of = lambda f: d.arity['true.' + f]
    rows_of = lambda f, e: f"{e}.rows::<{n_of(f)}>(&TAB_{ident(f)})" if f in wd else f"&{e}"
    deref = lambda f: "" if f in wd else "*"
    def each_rows(holder, rel, body):
        """Rust statement that runs `body` (over `r`, a row by value) for each row of the relation `rel` of `holder`, with no list made."""
        if rel in bits:
            return f"{holder}.{d.rid[rel]}.each(|r| {{ {body} }});"
        return f"for r in {holder}.{d.rid[rel]}.order.iter().copied() {{ {body} }}"
    w("/// The state: the true facts, one sorted list of rows for each fluent" + (" (a fluent held as the bits of words, pass 22, is a `Wd`: one bit a fact)." if wd else "."))
    w("#[derive(Clone, PartialEq, Eq, Hash, Debug, Default)]")
    w("pub struct State {")
    for f in d.fluents:
        w(f"    pub f_{ident(f)}: Wd<{wd[f][1]}>," if f in wd else f"    pub f_{ident(f)}: Vec<[u32; {n_of(f)}]>,")
    w("}")
    w("pub const FLUENTS: [&str; %d] = [%s];" % (len(d.fluents), ", ".join('"%s"' % f for f in d.fluents)))
    # the relations: static ones in St, the others in Dy
    for holder, name in ((True, "St"), (False, "Dy")):
        rels = [r for r in d.nodes if d.static[r] == holder and r not in demanded]
        w(f"pub struct {name} {{")
        for r in rels:
            w(f"    pub {d.rid[r]}: {('B' + d.rid[r][1:]) if r in bits else ('Dense<%d>' % d.arity.get(r, 0) if r in dense else 'Rel<%d>' % d.arity.get(r, 0))},")
        if gated and not holder:
            w("    /// Set by the questions that ask for every relation in full (`all_relations`, `ask_many`): no piece is left out.")
            w("    pub whole: bool,")
        w("}")
        new = lambda r: f"B{d.rid[r][1:]}::new()" if r in bits else (f"Dense::new(DS{d.rid[r][1:]}, RX{d.rid[r][1:]}, KP{d.rid[r][1:]})" if r in dense else f"Rel::new(KP{d.rid[r][1:]})")
        w(f"impl {name} {{ pub fn new() -> {name} {{ {name} {{ " + ", ".join([f"{d.rid[r]}: {new(r)}" for r in rels] + (["whole: false"] if gated and not holder else [])) + " } } }")
        if pooled and not holder:
            w(f"impl {name} {{ pub fn clear(&mut self) {{ " + " ".join(f"self.{d.rid[r]}.clear();" for r in rels) + " } }")
    # the rows new in a round, for the relations of a recursive set
    rec_rels = []
    for comp, rules in d.strata:
        if any(r.rec_steps for r in rules):
            rec_rels += sorted(comp)
    w("#[derive(Default)]")
    w("pub struct Dl {")
    for r in rec_rels:
        w(f"    pub {d.rid[r]}: Vec<[u32; {d.arity.get(r, 0)}]>,")
    w("}")
    if bits:
        for r in sorted(bits):
            w_bit_relation(w, d, plan, bits[r])
    # the clauses and the pieces
    stratum_fns = emit_strata(w, d, plan, places)
    if plan:
        w_shift_tables(w, plan)
        if plan.ends:
            w(RUNTIME_ENDS)
    if has_dw:
        w_dw(w, d, plan)
    if plan:
        w_asks(w, d, plan)
    nst = len(d.strata)
    td_on = False
    if plan and plan.topdown:
        tdg = TopDown(d, plan)
        try:
            roots = [x for x in ("terminal", "legal") if x in d.prog.nodes]
            if not roots:
                raise TopDownRefuse("the description has no terminal and no legal")
            if "support" in plan.passes:
                prepare_support(d, plan, tdg)
            if "reads" in plan.passes:
                prepare_reads(d, plan, tdg, reads)
            tdg.cone = tdg.build(roots)["cone"]
            td_tabs: list = []
            w_topdown_tables(td_tabs.append, d, plan, tdg)
            td_on = True
        except TopDownRefuse as e:
            plan.td_refused = str(e)
            plan.support, plan.support_h, plan.support_reads = [], [], {}
            plan.support_refused["(all)"] = f"the top-down way was not made: {e}"
            plan.reads = []
            if "reads" in plan.passes:
                plan.reads_refused["(all)"] = f"the top-down way was not made: {e}"
    if plan and "support" in plan.passes:
        heads = [f"// support: test {x['R']}, hypothetical relation {x['H']} made from {x['B']} (key {x['m']} places, row {x['p']}), guard domain {x['gstar'][0]}, relations made from it {','.join(x['hd'])}" for x in plan.support]
        heads += [f"// support refused: {k}: {v[:160]}" for k, v in sorted(plan.support_refused.items())]
        if not plan.support and not plan.support_refused:
            heads.append("// support: no hypothetical relation found")
        out[head_at:head_at] = heads
    if plan and "reads" in plan.passes:
        heads = [f"// reads: {r} with its first {k} places as the key" for r, k in plan.reads]
        if tdg.fl_ex:
            heads.append(f"// explain: every read carries its clause and literal ({len(tdg.ex_clause)} clauses)")
        heads += [f"// reads refused: {k}: {v[:160]}" for k, v in sorted(plan.reads_refused.items())]
        out[head_at:head_at] = heads
    sup = bool(td_on and plan and plan.support)
    ro = lambda holder, rel: f"{holder}.{d.rid[rel]}.rows_all()" if rel in bits else f"{holder}.{d.rid[rel]}.order"
    if td_on:
        w("\n".join(td_tabs))
        if plan.support:
            w("\n".join(support_prelude(plan) + support_consts(d, plan) + support_free(d, plan, tdg)))
        if tdg.fl_rec:
            w("\n".join(reads_prelude(d, plan, tdg)))
        ways = dict(ways or {})
        for q in ways:
            if q not in ("end", "legal-turn") or ways[q] not in ("bottom-up", "top-down"):
                raise Untranslatable(f"no question {q} with way {ways[q]}")
        w(f"/// For each question that has both ways, the way that answers it (decided by the clocks of plan 79.53, tests/translated_speed_bits.rs, and recorded by scripts/gdl_to_rust_make.py).")
        w(f"pub const WAYS: &[(&str, &str)] = &[(\"end\", \"{ways.get('end', 'bottom-up')}\"), (\"legal-turn\", \"{ways.get('legal-turn', 'bottom-up')}\")];")
        w(f"const WAY_END_TOP_DOWN: bool = {'true' if ways.get('end') == 'top-down' else 'false'};")
        w(f"const WAY_TURN_TOP_DOWN: bool = {'true' if ways.get('legal-turn') == 'top-down' else 'false'};")
    dwp = ", dw: &mut Dw" if has_dw else ""
    dwa = ", dw" if has_dw else ""
    w(f"pub const PIECES: [(&str, bool); {nst}] = [" + ", ".join('("%s", %s)' % ("+".join(sorted(c)), "true" if s else "false") for _, s, c in stratum_fns) + "];")
    if plan:
        w(f"pub const PASSES: &[&str] = &[{', '.join(chr(34) + x + chr(34) for x in plan.passes)}];")
        w(f"pub const COPIED: &[&str] = &[{', '.join(chr(34) + x + chr(34) for x in sorted(plan.copy))}];")
        w(f"pub const UNFOLDED: &[&str] = &[{', '.join(chr(34) + x + chr(34) for x in plan.unfolded)}];")
        w(f"pub const ENDS: &[&str] = &[{', '.join(chr(34) + x + chr(34) for x in sorted(plan.ends))}];")
        w(f"pub const ASKED: &[&str] = &[{', '.join(chr(34) + x + chr(34) for x in sorted(plan.asked))}];")
        if "gated" in plan.passes:
            w(f"pub const GATED: &[&str] = &[{', '.join(chr(34) + x + chr(34) for x in sorted('+'.join(sorted(d.strata[k][0])) for k in gated))}];")
    if infos:
        w("pub struct Lm { pub name: &'static str, pub form: &'static str, pub left: &'static str, pub right: &'static str, pub replacement: &'static str, pub flag: &'static str, pub evidence: &'static str, pub condition: &'static [(&'static str, &'static [&'static str], bool)] }")
        w("pub const LEMMAS: &[Lm] = &[" + ", ".join(
            f"Lm {{ name: {rs_str(i['name'])}, form: {rs_str(i['form'])}, left: {rs_str(i['left'])}, right: {rs_str(i['right'])}, replacement: {rs_str(i['rep'])}, flag: {rs_str(i['flag'])}, evidence: {rs_str(i['evidence'])}, condition: &[" + ", ".join(
                f"({rs_str(r)}, &[{', '.join(rs_str(a) for a in args)}], {'true' if ng else 'false'})" for r, args, ng in i["condition"]) + "] }" for i in infos) + "];")
    dyn_all = [k for k, s, c in stratum_fns if not s and not (c & demanded)]
    coned = plan.cone if (plan and (plan.unfolded or plan.asked or plan.projected or plan.support)) else d.cone
    legal_cone = [k for k in d.dynamic_meeting(coned([x for x in ("legal", "terminal", "goal")])) if not (d.strata[k][0] & demanded)]
    next_pairs = [f for f in d.fluents if f"next.{f}" in d.prog.nodes]
    next_cone = [k for k in d.dynamic_meeting(coned([f"next.{f}" for f in next_pairs])) if not (d.strata[k][0] & demanded)]
    if gated:
        legal_cone, next_cone = plan.gate_order(legal_cone), plan.gate_order(next_cone)
        w("/// Whether the piece of stratum k is left out of a question: every clause that reads it is shut by a flag literal that is false, or has its head in a piece that is left out.")
        w("fn off(d: &Dy, k: usize) -> bool {")
        w("    match k {")
        for k in sorted(gated):
            w(f"        {k} => {gated[k]},")
        w("        _ => false,")
        w("    }")
        w("}")
    w(f"fn run_piece(s: &St, d: &mut Dy, tm: &mut Terms{dwp}, k: usize) -> (u64, u64, u64) {{")
    w("    match k {")
    for k in dyn_all:
        if k in gated:
            w(f"        {k} => {{ if !d.whole && off(d, {k}) {{ return (0, 0, 0); }} st{k}(s, d, tm{dwa}) }}")
        else:
            w(f"        {k} => st{k}(s, d, tm{dwa}),")
    w("        _ => panic!(\"not a piece of the state\"),")
    w("    }")
    w("}")
    lst = lambda v: "&[" + ", ".join(map(str, v)) + "]"
    w(f"const ALL_DYNAMIC: &[usize] = {lst(dyn_all)};")
    w(f"const READ_CONE: &[usize] = {lst(legal_cone)};")
    if cones:
        arrs = []
        for mask in range(8):
            rs = [x for b, x in ((1, "legal"), (2, "terminal"), (4, "goal")) if mask & b and x in d.prog.nodes]
            ks = [k for k in d.dynamic_meeting(coned(rs)) if not (d.strata[k][0] & demanded)] if rs else []
            arrs.append(lst(plan.gate_order(ks) if gated else ks))
        w(f"const CONES: [&[usize]; 8] = [{', '.join(arrs)}];")
    w(f"const NEXT_CONE: &[usize] = {lst(next_cone)};")
    if carry:
        # pass (25): the fluents each piece's cone reads (a bit for each fluent of FLUENTS) and the relations each piece makes
        fl_of = []
        clears = {}
        for k, (comp, _r) in enumerate(d.strata):
            if d.static[sorted(comp)[0]] or (comp & demanded):
                fl_of.append(0)
                continue
            reached = coned(sorted(comp))
            mask = 0
            for i, f in enumerate(d.fluents):
                if f"true.{f}" in reached:
                    mask |= 1 << i
            if any(r in reached for r in INPUT_RELS):
                mask = (1 << len(d.fluents)) - 1            # a piece that reads the turn is never carried
            fl_of.append(mask)
            clears[k] = [r for r in sorted(comp) if not r.startswith("true.")]
        w("/// Pass (25): for each piece, the fluents its cone reads (a bit for each fluent of FLUENTS).")
        w(f"const PIECE_FL: [u32; {nst}] = [{', '.join(str(x) for x in fl_of)}];")
        w("/// Pass (25): the relations of the piece of stratum k that its question made (the facts loaded from the state are not the piece's).")
        w("fn clear_piece(d: &mut Dy, k: usize) {")
        w("    match k {")
        for k, rs in sorted(clears.items()):
            if rs:
                w(f"        {k} => {{ " + " ".join(f"d.{d.rid[r]}.clear();" for r in rs) + " }")
        w("        _ => {}")
        w("    }")
        w("}")
    # the game
    w("pub struct Game { st: St, pub tm: Terms, roles: Vec<u32>, noop: u32, last: Option<(State, Reading)>, pub prof: Vec<Stat>, pub profiling: bool" + (", pool: Option<Dy>" if pooled else "") + (", dw: Dw" if has_dw else "") + (", does_buf: Vec<(u32, u32)>" if keep else "") + (", td: Td, pub td_over: u64" if td_on else "") + (", sp: SpCount, sp_s: State, pub sp_plant: bool, pub sp_log: Option<Vec<SpLog>>" if sup else "") + (", cd: Option<Dy>, cvalid: Vec<bool>" if carry else "") + " }")
    w("impl Game {")
    w(f"    pub fn new() -> Game {{")
    w("        let mut st = St::new();")
    w("        let mut tm = Terms::new();")
    w("        let dy = Dy::new();")
    if has_dw:
        w("        let mut dw = Dw::new();")
    for k, s, _ in stratum_fns:
        if s:
            w(f"        st{k}(&mut st, &mut tm, &dy{', &mut dw' if has_dw else ''});")
    w(f"        let roles: Vec<u32> = {ro('st', 'role')}.iter().map(|r| r[0]).collect();")
    if td_on:
        w(f"        let mut g = Game {{ st, tm, roles, noop: {d.sym_id[NOOP]}, last: None, prof: vec![Stat::default(); {nst}], profiling: false" + (", pool: Some(dy)" if pooled else "") + (", dw" if has_dw else "") + (", does_buf: Vec::new()" if keep else "") + ", td: Td::new(), td_over: 0" + (", sp: SpCount::default(), sp_s: State::default(), sp_plant: false, sp_log: None" if sup else "") + (f", cd: None, cvalid: vec![false; {nst}]" if carry else "") + " };")
        w("        g.fill_td();")
        w("        g")
    else:
        w(f"        Game {{ st, tm, roles, noop: {d.sym_id[NOOP]}, last: None, prof: vec![Stat::default(); {nst}], profiling: false" + (", pool: Some(dy)" if pooled else "") + (", dw" if has_dw else "") + (", does_buf: Vec::new()" if keep else "") + (f", cd: None, cvalid: vec![false; {nst}]" if carry else "") + " }")
    w("    }")
    if td_on:
        statics_td = [r for r in sorted(tdg.used) if not r.startswith("true.")]
        fluents_td = [r for r in sorted(tdg.used) if r.startswith("true.")]
        w("    /// The relations that do not read the state, as the tables the top-down functions ask (made again by `restart`).")
        w("    fn fill_td(&mut self) {")
        for r in statics_td:
            w(f"        self.td.{tdg.holder(r)}.clear();")
            w(f"        for r in {ro('self.st', r)}.iter().copied() {{ self.td.{tdg.holder(r)}.add(r); }}")
        w("    }")
        w("    fn load_td(&mut self, s: &State) {")
        w("        self.td.overflow.store(false, std::sync::atomic::Ordering::Relaxed);")
        for r in fluents_td:
            f = r[len("true."):]
            w(f"        for r in s.f_{ident(f)}.rows::<{d.arity[r]}>(&TAB_{ident(f)}) {{ self.td.{tdg.holder(r)}.add(r); }}")
        w("    }")
        w("    /// Only the load and the clearing of the tables the top-down functions read, for the timing that says what they cost.")
        w("    pub fn td_load_only(&mut self, s: &State) { self.load_td(s); self.clear_td(); }")
        w("    fn clear_td(&mut self) {")
        for r in fluents_td:
            w(f"        self.td.{tdg.holder(r)}.clear();")
        w("    }")
        w("    /// The rows of a relation of the cone of the questions, every place free, by the top-down way (for the tests that hold each relation of it to the bottom-up way); None when the name is not such a relation or the worklist overflowed. A place where a clause gives a compound term holds `FREE - 1`.")
        w("    pub fn ask_td(&mut self, s: &State, rel: &str) -> Option<Vec<Vec<u32>>> {")
        w("        self.load_td(s);")
        w("        let mut v: Vec<Vec<u32>> = vec![];")
        w("        let known = match rel {")
        for r in sorted(tdg.cone):
            if tdg.how(r) == "derived":
                n = d.arity.get(r, 0)
                w(f"            \"{r}\" => {{ {tdg.fname(r, ())}(&self.td, [FREE; {n}], &mut |row| {{ v.push(row.to_vec()); false }}); true }}")
        w("            _ => false,")
        w("        };")
        w("        let over = self.td.overflow.load(std::sync::atomic::Ordering::Relaxed);")
        w("        self.clear_td();")
        w("        if !known || over { return None; }")
        w("        v.sort(); v.dedup();")
        w("        Some(v)")
        w("    }")
        if tdg.fl_rec:
            w("\n".join(reads_method(d, plan, tdg)))
        if "terminal" in d.prog.nodes:
            w("    /// The end of the game, answered top-down: the first rule that holds ends the walk (module text, (24)).")
            w("    pub fn terminal_td(&mut self, s: &State) -> bool {")
            w("        self.load_td(s);")
            w(f"        let r = {tdg.fname('terminal', ())}(&self.td, [], &mut |_| true);")
            w("        let over = self.td.overflow.load(std::sync::atomic::Ordering::Relaxed);")
            w("        self.clear_td();")
            w("        if over { self.td_over += 1; return self.terminal_bu(s); }")
            w("        r")
            w("    }")
        if "legal" in d.prog.nodes:
            w("    /// Whether a turn that is not the one that passes is legal at the state (the mover exists), answered top-down: the first legal row ends the walk.")
            w("    pub fn has_turn_td(&mut self, s: &State) -> bool {")
            w("        self.load_td(s);")
            w("        let noop = self.noop;")
            w(f"        let r = {tdg.fname('legal', ())}(&self.td, [FREE; 2], &mut |row| row[1] != noop);")
            w("        let over = self.td.overflow.load(std::sync::atomic::Ordering::Relaxed);")
            w("        self.clear_td();")
            w("        if over { self.td_over += 1; return self.legal_mover(s).is_some(); }")
            w("        r")
            w("    }")
    w("    pub fn roles(&self) -> &[u32] { &self.roles }")
    w("    pub fn initial(&self) -> State {")
    w("        let mut s = State::default();")
    for f in d.fluents:
        if f"init.{f}" in d.prog.nodes:
            if f in wd:
                w(f"        for r in {ro('self.st', 'init.' + f)}.iter() {{ s.f_{ident(f)}.set(widx(&TAB_{ident(f)}, r).expect(\"an initial fact outside the sorts of its relation\")); }}")
            else:
                w(f"        s.f_{ident(f)} = {ro('self.st', 'init.' + f)}.clone();")
    w("        s.sort();")
    w("        s")
    w("    }")
    inits = [f for f in d.fluents if f"init.{f}" in d.prog.nodes]
    init_rels = {f"init.{f}" for f in inits}
    redo = [(k, c) for k, s, c in stratum_fns if s and not (c & init_rels) and (d.cone(sorted(c)) & init_rels)]
    w("    /// Begin the game again at the facts given as the initial ones (plan 79.47): the `init` relations hold them and the relations that read no state and read an `init` relation are made again.")
    w("    pub fn restart(&mut self, facts: &[(String, Vec<String>)]) -> Result<(), String> {")
    w("        let s = self.state_of_facts(facts)?;")
    for f in inits:
        rel = f"self.st.{d.rid['init.' + f]}"
        w(f"        {rel}.clear();")
        w(f"        for r in {rows_of(f, 's.f_' + ident(f))} {{ {rel}.add({deref(f)}r); }}")
    for k, c in redo:
        for r in sorted(c):
            w(f"        self.st.{d.rid[r]}.clear();")
    if redo:
        w("        let dy = Dy::new();")
    for k, c in redo:
        w(f"        st{k}(&mut self.st, &mut self.tm, &dy{', &mut self.dw' if has_dw else ''});")
    w(f"        self.roles = {ro('self.st', 'role')}.iter().map(|r| r[0]).collect();")
    w("        self.last = None;")
    if carry:
        w("        self.cd = None;")
    if td_on:
        w("        self.fill_td();")
    w("        Ok(())")
    w("    }")
    w("    /// The state's rows and the turn's into a fresh evaluation.")
    if pooled:
        w("    fn load(&mut self, s: &State, does: &[(u32, u32)]) -> Dy {")
        w("        let mut d = match self.pool.take() { Some(d) => d, None => Dy::new() };")
        if sup:
            w("        self.sp_s.assign(s);")
        if gated:
            w("        d.whole = false;")
    else:
        w("    fn load(&self, s: &State, does: &[(u32, u32)]) -> Dy {")
        w("        let mut d = Dy::new();")
    for f in d.fluents:
        w(f"        for r in {rows_of(f, 's.f_' + ident(f))} {{ d.{d.rid['true.' + f]}.add({deref(f)}r); }}")
    if "does" in d.prog.nodes:
        w(f"        for (r, m) in does {{ d.{d.rid['does']}.add([*r, *m]); }}")
    w("        d")
    w("    }")
    w("    fn run(&mut self, d: &mut Dy, which: &[usize]) {")
    w("        for &k in which {")
    if sup:
        pc = "match k { " + " ".join(f"SP_K{i} => self.run_support{i}(d)," for i in range(len(plan.support))) + f" _ => run_piece(&self.st, d, &mut self.tm{', &mut self.dw' if has_dw else ''}, k) }}"
    else:
        pc = f"run_piece(&self.st, d, &mut self.tm{', &mut self.dw' if has_dw else ''}, k)"
    w("            if self.profiling {")
    w("                let t = std::time::Instant::now();")
    w(f"                let (a, b, c) = {pc};")
    w("                let p = &mut self.prof[k];")
    w("                p.calls += 1; p.nanos += t.elapsed().as_nanos() as u64; p.rows += a; p.visits += b; p.probes += c;")
    w("            } else {")
    w(f"                {pc};")
    w("            }")
    w("        }")
    w("    }")
    if sup:
        w("\n".join(support_methods(d, plan, tdg, ro, gated, has_dw)))
    if cones:
        lg = f"lr.legal.extend({ro('d', 'legal')}.iter().map(|x| (x[0], x[1])));" if "legal" in d.prog.nodes else ""
        tm_ = f"lr.terminal = !{ro('d', 'terminal')}.is_empty();" if "terminal" in d.prog.nodes else "lr.terminal = false;"
        gl = f"lr.goals.extend({ro('d', 'goal')}.iter().map(|x| (x[0], x[1])));" if "goal" in d.prog.nodes else ""
        w("    /// The reading of a state that holds what `need` asks (bit 0 the legal rows, bit 1 whether the end holds, bit 2 the goal rows), kept with the mask it holds: the cone of what is needed and of what the kept reading of the same state holds is run (pass 21).")
        w("    fn ensure(&mut self, s: &State, need: u8) {")
        w("        let mut want = need;")
        if carry:
            w("        let mut same = false;")
            w("        if let Some((k, r)) = &self.last { if k == s { if r.have & need == need { return; } want = r.have | need; same = true; } }")
            w("        let mut d;")
            w("        if self.cd.is_none() || self.last.is_none() {")
            w("            d = self.cd.take().unwrap_or_else(Dy::new);")
            w("            d.clear();")
            w("            for v in self.cvalid.iter_mut() { *v = false; }")
            for f in d.fluents:
                w(f"            for r in {rows_of(f, 's.f_' + ident(f))} {{ d.{d.rid['true.' + f]}.add({deref(f)}r); }}")
            if gated:
                w("            d.whole = false;")
            w("        } else {")
            w("            d = self.cd.take().unwrap();")
            w("            if !same {")
            w("                let diff = self.last.as_ref().unwrap().0.fluent_diff(s);")
            w("                for k in 0..self.cvalid.len() { if self.cvalid[k] && PIECE_FL[k] & diff != 0 { clear_piece(&mut d, k); self.cvalid[k] = false; } }")
            for i, f in enumerate(d.fluents):
                w(f"                if diff & {1 << i} != 0 {{ d.{d.rid['true.' + f]}.clear(); for r in {rows_of(f, 's.f_' + ident(f))} {{ d.{d.rid['true.' + f]}.add({deref(f)}r); }} }}")
            w("            }")
            w("        }")
            w("        self.run_carried(&mut d, CONES[want as usize]);")
        else:
            w("        if let Some((k, r)) = &self.last { if k == s { if r.have & need == need { return; } want = r.have | need; } }")
            w("        let mut d = self.load(s, &[]);")
            w("        self.run(&mut d, CONES[want as usize]);")
        w("        let (mut ls, mut lr) = match self.last.take() { Some(x) => x, None => (State::default(), Reading { legal: vec![], terminal: false, goals: vec![], have: 0 }) };")
        w("        ls.assign(s);")
        w(f"        if want & 1 != 0 {{ lr.legal.clear(); {lg} }}")
        w(f"        if want & 2 != 0 {{ {tm_} }}")
        w(f"        if want & 4 != 0 {{ lr.goals.clear(); {gl} }}")
        w("        lr.have = want;")
        w("        self.last = Some((ls, lr));")
        if carry:
            w("        self.cd = Some(d);")
            w("    }")
            w("    /// Pass (25): the pieces of the cone that are not valid at the kept state, run in the order of the cone; a piece left out by the gates remains not valid.")
            w("    fn run_carried(&mut self, d: &mut Dy, which: &[usize]) {")
            w("        for &k in which {")
            w("            if self.cvalid[k] { continue; }")
            if gated:
                w("            if off(d, k) { continue; }")
            w("            self.run(d, &[k]);")
            w("            self.cvalid[k] = true;")
            w("        }")
            w("    }")
            w("    /// The last state read for the questions of the cones, and what the pass keeps of it (for the tests): how many pieces are valid.")
            w("    pub fn carried_valid(&self) -> usize { self.cvalid.iter().filter(|v| **v).count() }")
            w("    /// Pass (25), for the tests: the state is read for every question of the cones, and the rows the kept evaluation holds of each relation of a valid piece of those cones are handed out (to be held to `all_relations`).")
            w("    pub fn carried_relations(&mut self, s: &State) -> Vec<(&'static str, Vec<Vec<u32>>)> {")
            w("        self.ensure(s, 7);")
            w("        let d = self.cd.as_ref().unwrap();")
            w("        let mut out: Vec<(&'static str, Vec<Vec<u32>>)> = vec![];")
            cone7 = [k for k in d.dynamic_meeting(coned([x for x in QUESTION_RELS if x in d.prog.nodes])) if not (d.strata[k][0] & demanded)]
            for k in cone7:
                for r in sorted(d.strata[k][0]):
                    if r.startswith("true.") or r in demanded or (plan.copy and r in plan.copy):
                        continue
                    w(f"        if self.cvalid[{k}] {{ out.push((\"{r}\", {ro('d', r)}.iter().map(|x| x.to_vec()).collect())); }}")
            w("        out")
            w("    }")
        else:
            w("        d.clear();")
            w("        self.pool = Some(d);")
            w("    }")
            w("    /// For the tests (the same item as the modules with the pass `carry`): the state is read for every question of the cones as a state not read before, and the rows made of each relation of those cones are handed out.")
            w("    pub fn carried_relations(&mut self, s: &State) -> Vec<(&'static str, Vec<Vec<u32>>)> {")
            w("        let mut d = self.load(s, &[]);")
            w("        self.run(&mut d, CONES[7]);")
            w("        let mut out: Vec<(&'static str, Vec<Vec<u32>>)> = vec![];")
            for k in [k for k in d.dynamic_meeting(coned([x for x in QUESTION_RELS if x in d.prog.nodes])) if not (d.strata[k][0] & demanded)]:
                for r in sorted(d.strata[k][0]):
                    if r.startswith("true.") or r in demanded or (plan.copy and r in plan.copy):
                        continue
                    w(f"        out.push((\"{r}\", {ro('d', r)}.iter().map(|x| x.to_vec()).collect()));")
            w("        d.clear();")
            w("        self.pool = Some(d);")
            w("        out")
            w("    }")
            w("    pub fn carried_valid(&self) -> usize { 0 }")
        w("    /// The first role that has a turn that is not the one that passes, by the legal rows alone (no test of the end).")
        w("    fn legal_mover(&mut self, s: &State) -> Option<usize> {")
        w("        self.ensure(s, 1);")
        w("        let rd = &self.last.as_ref().unwrap().1;")
        w("        self.roles.iter().position(|r| rd.legal.iter().any(|(q, m)| q == r && *m != self.noop))")
        w("    }")
        w("    pub fn mover(&mut self, s: &State) -> Option<usize> {")
        w("        self.ensure(s, 3);")
        w("        let rd = &self.last.as_ref().unwrap().1;")
        w("        if rd.terminal { return None; }")
        w("        self.roles.iter().position(|r| rd.legal.iter().any(|(q, m)| q == r && *m != self.noop))")
        w("    }")
        w("    pub fn moves(&mut self, s: &State) -> Vec<u32> { let mut v = Vec::new(); self.moves_into(s, &mut v); v }")
        w("    /// The permitted turns of the mover (none at an end), into a buffer the caller owns.")
        w("    pub fn moves_into(&mut self, s: &State, out: &mut Vec<u32>) {")
        w("        out.clear();")
        w("        let Some(r) = self.mover(s) else { return };")
        w("        let rd = &self.last.as_ref().unwrap().1;")
        w("        out.extend(rd.legal.iter().filter(|(q, _)| *q == self.roles[r]).map(|(_, m)| *m));")
        w("    }")
        w("    /// The legal rows of the first role that has a turn, with no test of the end and none of the work that only the end needs.")
        w("    pub fn legal_moves_into(&mut self, s: &State, out: &mut Vec<u32>) {")
        w("        out.clear();")
        w("        let Some(r) = self.legal_mover(s) else { return };")
        w("        let rd = &self.last.as_ref().unwrap().1;")
        w("        out.extend(rd.legal.iter().filter(|(q, _)| *q == self.roles[r]).map(|(_, m)| *m));")
        w("    }")
        if td_on:
            w("    /// The end of the game, by the way the module records for that question (`WAYS`, module text, (24)).")
            w("    pub fn terminal(&mut self, s: &State) -> bool { if WAY_END_TOP_DOWN { self.terminal_td(s) } else { self.terminal_bu(s) } }")
            w("    /// Whether a turn that is not the one that passes is legal, by the way the module records for that question.")
            w("    pub fn has_turn(&mut self, s: &State) -> bool { if WAY_TURN_TOP_DOWN { self.has_turn_td(s) } else { self.legal_mover(s).is_some() } }")
            w("    pub fn terminal_bu(&mut self, s: &State) -> bool { self.ensure(s, 2); self.last.as_ref().unwrap().1.terminal }")
        else:
            w("    pub fn terminal(&mut self, s: &State) -> bool { self.ensure(s, 2); self.last.as_ref().unwrap().1.terminal }")
        w("    pub fn outcome(&mut self, s: &State) -> Option<Vec<i64>> {")
        w("        self.ensure(s, 6);")
        w("        let rd = &self.last.as_ref().unwrap().1;")
        w("        if !rd.terminal { return None; }")
        w("        Some(self.roles.iter().map(|r| rd.goals.iter().find(|(q, _)| q == r).and_then(|(_, v)| self.tm.name(*v).parse().ok()).unwrap_or(0)).collect())")
        w("    }")
    elif keep:
        w("    fn ensure(&mut self, s: &State) {")
        w("        if let Some((k, _)) = &self.last { if k == s { return; } }")
        w("        let mut d = self.load(s, &[]);")
        w("        self.run(&mut d, READ_CONE);")
        w("        let (mut ls, mut lr) = match self.last.take() { Some(x) => x, None => (State::default(), Reading { legal: vec![], terminal: false, goals: vec![] }) };")
        w("        ls.assign(s);")
        lg = f"lr.legal.extend({ro('d', 'legal')}.iter().map(|x| (x[0], x[1])));" if "legal" in d.prog.nodes else ""
        tm_ = f"lr.terminal = !{ro('d', 'terminal')}.is_empty();" if "terminal" in d.prog.nodes else "lr.terminal = false;"
        gl = f"lr.goals.extend({ro('d', 'goal')}.iter().map(|x| (x[0], x[1])));" if "goal" in d.prog.nodes else ""
        w("        lr.legal.clear(); lr.goals.clear();")
        w(f"        {lg} {tm_} {gl}")
        w("        self.last = Some((ls, lr));")
        w("        d.clear();")
        w("        self.pool = Some(d);")
        w("    }")
        w("    pub fn mover(&mut self, s: &State) -> Option<usize> {")
        w("        self.ensure(s);")
        w("        let rd = &self.last.as_ref().unwrap().1;")
        w("        if rd.terminal { return None; }")
        w("        self.roles.iter().position(|r| rd.legal.iter().any(|(q, m)| q == r && *m != self.noop))")
        w("    }")
        w("    pub fn moves(&mut self, s: &State) -> Vec<u32> {")
        w("        let Some(r) = self.mover(s) else { return vec![] };")
        w("        let rd = &self.last.as_ref().unwrap().1;")
        w("        rd.legal.iter().filter(|(q, _)| *q == self.roles[r]).map(|(_, m)| *m).collect()")
        w("    }")
        w("    pub fn terminal(&mut self, s: &State) -> bool { self.ensure(s); self.last.as_ref().unwrap().1.terminal }")
        w("    pub fn outcome(&mut self, s: &State) -> Option<Vec<i64>> {")
        w("        self.ensure(s);")
        w("        let rd = &self.last.as_ref().unwrap().1;")
        w("        if !rd.terminal { return None; }")
        w("        Some(self.roles.iter().map(|r| rd.goals.iter().find(|(q, _)| q == r).and_then(|(_, v)| self.tm.name(*v).parse().ok()).unwrap_or(0)).collect())")
        w("    }")
    else:
        w("    fn read(&mut self, s: &State) -> Reading {")
        w("        if let Some((k, r)) = &self.last { if k == s { return r.clone(); } }")
        w("        let mut d = self.load(s, &[]);")
        w("        self.run(&mut d, READ_CONE);")
        lg = f"{ro('d', 'legal')}.iter().map(|x| (x[0], x[1])).collect()" if "legal" in d.prog.nodes else "vec![]"
        tm_ = f"!{ro('d', 'terminal')}.is_empty()" if "terminal" in d.prog.nodes else "false"
        gl = f"{ro('d', 'goal')}.iter().map(|x| (x[0], x[1])).collect()" if "goal" in d.prog.nodes else "vec![]"
        w(f"        let rd = Reading {{ legal: {lg}, terminal: {tm_}, goals: {gl} }};")
        w("        self.last = Some((s.clone(), rd.clone()));")
        if pooled:
            w("        d.clear();")
            w("        self.pool = Some(d);")
        w("        rd")
        w("    }")
        w("    pub fn mover(&mut self, s: &State) -> Option<usize> {")
        w("        let rd = self.read(s);")
        w("        if rd.terminal { return None; }")
        w("        self.roles.iter().position(|r| rd.legal.iter().any(|(q, m)| q == r && *m != self.noop))")
        w("    }")
        w("    pub fn moves(&mut self, s: &State) -> Vec<u32> {")
        w("        let Some(r) = self.mover(s) else { return vec![] };")
        w("        let rd = self.read(s);")
        w("        rd.legal.iter().filter(|(q, _)| *q == self.roles[r]).map(|(_, m)| *m).collect()")
        w("    }")
        w("    pub fn terminal(&mut self, s: &State) -> bool { self.read(s).terminal }")
        w("    pub fn outcome(&mut self, s: &State) -> Option<Vec<i64>> {")
        w("        let rd = self.read(s);")
        w("        if !rd.terminal { return None; }")
        w("        Some(self.roles.iter().map(|r| rd.goals.iter().find(|(q, _)| q == r).and_then(|(_, v)| self.tm.name(*v).parse().ok()).unwrap_or(0)).collect())")
        w("    }")
    w("    pub fn next(&mut self, s: &State, mv: u32) -> State {")
    w("        let me = self.legal_mover(s);" if cones else "        let me = self.mover(s);")
    if keep:
        w("        let mut does = std::mem::take(&mut self.does_buf);")
        w("        does.clear();")
        w("        does.extend(self.roles.iter().enumerate().map(|(i, r)| (*r, if Some(i) == me { mv } else { self.noop })));")
    else:
        w("        let does: Vec<(u32, u32)> = self.roles.iter().enumerate().map(|(i, r)| (*r, if Some(i) == me { mv } else { self.noop })).collect();")
    w("        let mut d = self.load(s, &does);")
    w("        self.run(&mut d, NEXT_CONE);")
    w("        let mut n = State::default();")
    for f in next_pairs:
        if f in wd:
            tb = f"TAB_{ident(f)}"
            setr = f"v.set(widx(&{tb}, &r).expect(\"a next fact outside the sorts of its relation\"));"
            if plan and f"next.{f}" in plan.copy:
                cond = copy_condition(d, plan, f'next.{f}', 'self.st', 'd', '&self.st, &d, &mut self.tm', 'r')
                w(f"        n.f_{ident(f)} = {{ let mut v = s.f_{ident(f)}; for r in s.f_{ident(f)}.rows::<{n_of(f)}>(&{tb}) {{ if !({cond}) {{ v.unset(widx(&{tb}, &r).unwrap()); }} }} {each_rows('d', 'next.' + f, setr)} v }};")
            else:
                w(f"        n.f_{ident(f)} = {{ let mut v = Wd::default(); {each_rows('d', 'next.' + f, setr)} v }};")
        elif plan and f"next.{f}" in plan.copy:
            w(f"        n.f_{ident(f)} = {{ let mut v: Vec<[u32; {d.arity['true.' + f]}]> = vec![]; for r in &s.f_{ident(f)} {{ if {copy_condition(d, plan, f'next.{f}', 'self.st', 'd', '&self.st, &d, &mut self.tm', 'r')} {{ v.push(*r); }} }} v.extend_from_slice(&{ro('d', 'next.' + f)}); v }};")
        else:
            w(f"        n.f_{ident(f)} = {ro('d', 'next.' + f)}.clone();")
    w("        n.sort();")
    if pooled:
        w("        d.clear();")
        w("        self.pool = Some(d);")
    if keep:
        w("        self.does_buf = does;")
    w("        n")
    w("    }")
    w("    pub fn sym(&self, name: &str) -> Option<u32> { SYMS.binary_search(&name).ok().map(|i| i as u32) }")
    w("    pub fn render(&self, t: u32) -> String { self.tm.render(t) }")
    w("    /// A term from its text: a constant, or (functor arg ...).")
    w("    pub fn term(&mut self, text: &str) -> Result<u32, String> {")
    w("        let spaced = text.replace('(', \" ( \").replace(')', \" ) \");")
    w("        let toks: Vec<&str> = spaced.split_whitespace().collect();")
    w("        let mut i = 0;")
    w("        let t = self.term_at(&toks, &mut i)?;")
    w("        if i != toks.len() { return Err(format!(\"trailing text in a term: {text}\")); }")
    w("        Ok(t)")
    w("    }")
    w("    fn term_at(&mut self, toks: &[&str], i: &mut usize) -> Result<u32, String> {")
    w("        let tok = *toks.get(*i).ok_or(\"a term ends too soon\")?;")
    w("        *i += 1;")
    w("        if tok != \"(\" { return self.sym(tok).ok_or_else(|| format!(\"no symbol {tok} in the description\")); }")
    w("        let f = *toks.get(*i).ok_or(\"a term ends too soon\")?;")
    w("        *i += 1;")
    w("        let ft = self.sym(f).ok_or_else(|| format!(\"no symbol {f} in the description\"))?;")
    w("        let mut args = vec![];")
    w("        while toks.get(*i) != Some(&\")\") { if *i >= toks.len() { return Err(\"a term is not closed\".into()); } args.push(self.term_at(toks, i)?); }")
    w("        *i += 1;")
    w("        Ok(self.tm.fun(ft, &args))")
    w("    }")
    w("    /// The state from its facts: (fluent, arguments as text).")
    w("    pub fn state_of_facts(&mut self, facts: &[(String, Vec<String>)]) -> Result<State, String> {")
    w("        let mut s = State::default();")
    w("        for (rel, args) in facts {")
    w("            let mut ts = Vec::with_capacity(args.len());")
    w("            for a in args { ts.push(self.term(a)?); }")
    w("            match rel.as_str() {")
    for f in d.fluents:
        n = d.arity["true." + f]
        if f in wd:
            w(f"                \"{f}\" => {{ if ts.len() != {n} {{ return Err(format!(\"{f} takes {n} arguments\")); }} let mut r = [0u32; {n}]; r.copy_from_slice(&ts); s.f_{ident(f)}.set(widx(&TAB_{ident(f)}, &r).ok_or_else(|| format!(\"a fact of {f} outside the sorts of its places\"))?); }}")
        else:
            w(f"                \"{f}\" => {{ if ts.len() != {n} {{ return Err(format!(\"{f} takes {n} arguments\")); }} let mut r = [0u32; {n}]; r.copy_from_slice(&ts); s.f_{ident(f)}.push(r); }}")
    w("                other => return Err(format!(\"the description has no fluent {other}\")),")
    w("            }")
    w("        }")
    w("        s.sort();")
    w("        Ok(s)")
    w("    }")
    w("    /// The facts of a state: (fluent, arguments as text), sorted.")
    w("    pub fn facts_of(&self, s: &State) -> Vec<(String, Vec<String>)> {")
    w("        let mut out = vec![];")
    for f in d.fluents:
        w(f"        for r in {rows_of(f, 's.f_' + ident(f))} {{ out.push((\"{f}\".to_string(), r.iter().map(|t| self.tm.render(*t)).collect())); }}")
    w("        out.sort();")
    w("        out")
    w("    }")
    w("    /// Every relation that does not read the state, with its rows (the extension that was computed once).")
    w("    pub fn static_relations(&self) -> Vec<(&'static str, Vec<Vec<u32>>)> {")
    w("        vec![")
    for r in d.nodes:
        if d.static[r]:
            w(f"            (\"{r}\", {ro('self.st', r)}.iter().map(|x| x.to_vec()).collect()),")
    w("        ]")
    w("    }")
    w("    /// Every relation that reads the state, evaluated in full on a state and a joint turn (the turn's rows void when `does` is void).")
    w("    pub fn all_relations(&mut self, s: &State, does: &[(u32, u32)]) -> Vec<(&'static str, Vec<Vec<u32>>)> {")
    w("        let mut d = self.load(s, does);")
    if gated:
        w("        d.whole = true;")
    if plan and (plan.ends or plan.asked):
        w("        self.dw.full = true;")
    w("        self.run(&mut d, ALL_DYNAMIC);")
    if plan and (plan.ends or plan.asked):
        w("        self.dw.full = false;")
    if not plan:
        w("        vec![")
        for r in d.nodes:
            if not d.static[r]:
                w(f"            (\"{r}\", {ro('d', r)}.iter().map(|x| x.to_vec()).collect()),")
        w("        ]")
    else:
        w("        let out = vec![")
        for r in d.nodes:
            if d.static[r] or r in demanded:
                continue
            if r in plan.copy:
                f = r[len("next."):]
                w(f"            (\"{r}\", {{ let mut v: Vec<Vec<u32>> = {ro('d', r)}.iter().map(|x| x.to_vec()).collect(); for r in {rows_of(f, 's.f_' + ident(f))} {{ if {copy_condition(d, plan, r, 'self.st', 'd', '&self.st, &d, &mut self.tm', 'r')} {{ v.push(r.to_vec()); }} }} v.sort(); v.dedup(); v }}),")
            else:
                w(f"            (\"{r}\", {ro('d', r)}.iter().map(|x| x.to_vec()).collect()),")
        w("        ];")
        if pooled:
            w("        d.clear();")
            w("        self.pool = Some(d);")
        w("        out")
    w("    }")
    if plan:
        w_api(w, d, plan, pooled)
    w("    pub fn noop(&self) -> u32 { self.noop }")
    w("    pub fn reset_profile(&mut self) { for p in self.prof.iter_mut() { *p = Stat::default(); } }")
    w("}")
    if keep:
        w("impl State {")
        w("    /// Copy `o` into this state, keeping the memory of the lists it holds.")
        w("    pub fn assign(&mut self, o: &State) {")
        for f in d.fluents:
            if f in wd:
                w(f"        self.f_{ident(f)} = o.f_{ident(f)};")
            else:
                w(f"        self.f_{ident(f)}.clear(); self.f_{ident(f)}.extend_from_slice(&o.f_{ident(f)});")
        w("    }")
        w("}")
    if carry:
        w("impl State {")
        w("    /// Pass (25): a bit for each fluent (the order of FLUENTS) whose facts differ between this state and `o`.")
        w("    pub fn fluent_diff(&self, o: &State) -> u32 {")
        w("        let mut m = 0u32;")
        for i, f in enumerate(d.fluents):
            w(f"        if self.f_{ident(f)} != o.f_{ident(f)} {{ m |= {1 << i}; }}")
        w("        m")
        w("    }")
        w("}")
    w("impl State {")
    w("    pub fn sort(&mut self) {")
    for f in d.fluents:
        if f not in wd:
            w(f"        self.f_{ident(f)}.sort(); self.f_{ident(f)}.dedup();")
    w("    }")
    w("}")
    w("}")
    translate.last_plan = plan                              # what the passes found, for the tests and the printing scripts that ask (a diagnostic, not part of the document)
    return "\n".join(out) + "\n"


def compose_text(paths: list) -> str:
    """The base and its deltas composed as `cs_gdl::compose` does (the shared module's, read only)."""
    import cs_engine as CE
    text = Path(paths[0]).read_text()
    for q in paths[1:]:
        text = CE.compose(text, Path(q).read_text())
    return text


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="translate a description to Rust")
    ap.add_argument("sources", nargs="+", help="the base description and its deltas, in order")
    ap.add_argument("--module", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    text = compose_text(a.sources)
    src = translate(text, a.module)
    Path(a.out).write_text(src)
    print(f"OBS 79.37 translate {a.module}: description {len(text)} bytes {len(GDL.clauses(GDL.parse(text)))} clauses, Rust {len(src)} bytes {src.count(chr(10))} lines")
    return 0


if __name__ == "__main__":
    sys.exit(main())
