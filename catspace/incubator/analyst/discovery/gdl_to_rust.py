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
PASSES = ("dense", "demand", "copy", "delta", "bits", "order", "keep", "scan", "cut", "shift", "base", "start")
QUESTION_RELS = ("legal", "terminal", "goal")           # read in full by the questions (as are the `next` relations)
DENSE_SPACE = 1 << 22                                   # a dense relation: at most this many tuples of its sorts ...
DENSE_KEYS = 1 << 16                                    # ... and at most this many keys for any index the joins ask by
SMALL_SCAN = 8                                          # a loop over a whole fluent is cheap when its sorts allow no more rows than this
START_MAX = 4                                           # pass (13): at most this many orders of one clause are written (the reader's and the best others)
BOUND = "\0"                                            # the mark of a constant that stands for a variable bound by the key of a demanded relation


class Untranslatable(Exception):
    pass


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

    def __init__(self, clause, scc: set, avoid=(), first=None, est=None):
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
                return (not (x[1] in avoid and keyed == 0 and len(x[2]) > 0), nb == len(x[2]), nb, -est(x, bound, scc) if est else 0, -len(x[2]))
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
            if p not in PASSES:
                raise Untranslatable(f"no pass named {p} (the passes: {', '.join(PASSES)})")
        self.d = d
        if "base" in passes and "scan" not in passes:
            passes = tuple(passes) + ("scan",)                 # the clauses pass (8) makes ask words with their key tiles unbound, which only pass (10) answers by the words held (on top of bits alone it made the module half as fast: diagnosed, 79.41 part A)
        self.passes = tuple(p for p in PASSES if p in passes)
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
        self.pre_bits: set = set(self.bit_candidates()) if "bits" in self.passes else set()          # pass (5) comes before pass (2): a relation that can be held as bits is not answered on demand
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

    def cone(self, wanted) -> set:
        """The relations the written rules need to make `wanted` (the dependency graph of the rules as written, which a pass that unfolds has changed)."""
        deps: dict = {}
        for k, (comp, _r) in enumerate(self.d.strata):
            reads = {st[1] for r in self.rules[k] for st in r.steps if st[0] != "distinct"}
            for h in comp:
                deps.setdefault(h, set()).update(reads)
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
            if d.static[r] or r in roots or r in INPUT_RELS or r.startswith("true.") or not rules or any(x.rec_steps for x in rules) or r in self.pre_bits or r in self.unfolded:
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
                for i in self.bits.values():
                    i.orients = {i.canon: "the membership test"}
                self.bit_rec = self.recursive_bits()
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
            else:
                self.out(f"let rows{k}: &[[u32; {n}]] = {get};")
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

    def __init__(self, d, places, cid, rule, scc, plan, mode="full", clause=None, key_places=(), record=False):
        super().__init__(d, places, cid, rule, scc, plan, mode, clause, key_places)
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
        self.prof.append(f"({factor}) * 2" if vlist else f"({factor})")
        folded = self.fold(k, ft, args, vvars, w) if vlist else False
        if masked or folded or delta or opened:
            self.out(f"if {w} == 0 {{ {self.fail()} }}")
        if vlist and self.or_word_ok(k, info, f, args, vvars):
            hi = self.bi[r.head_rel]
            hrid = self.d.rid[r.head_rel]
            hh = self.holder(r.head_rel)
            tiles = {i: self.tile_expr(r.head_rel, t, r.head) for i, t in enumerate(hi.tiles) if i != hi.canon}
            self.out(f"let hk{u} = {self.index_of(hi, hi.canon, tiles)};")
            self.out(f"let nb{u} = {hh}.{hrid}.or_word(hk{u}, {w});")
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


def write_shift(w, d, plan, places, name: str, comp, rule, shape: dict):
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
    src = f"d.{rid}.o{info.canon}"
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
    bw.out(f"let nb{u} = d.{rid}.or_word(ix{u}, acc{u});")
    bw.out(f"if nb{u} != 0 {{ made += nb{u}.count_ones() as u64; }}")
    bw.close(1)
    bw.out("made")
    bw.lines.append("}")
    plan.shift_used.add(key_of_table(rel, fin, fout))
    w("\n".join(bw.lines))


def key_of_table(rel: str, fin: int, fout: int) -> tuple:
    return (rel, fin, fout)


def w_shift_tables(w, plan):
    """The tables of the pass shift that the document uses: for each combination of the onward's other tiles, its (displacement, mask) pairs."""
    for key in sorted(plan.shift_used):
        tid, by, nsel, sel = plan.shift_tables[key]
        rows = ", ".join("&[" + ", ".join(f"({dl}, 0x{m:x}u64)" for dl, m in sorted(dd.items())) + "]" for dd in by)
        w(f"static SH{tid}: [&[(i32, u64)]; {nsel}] = [{rows}];")


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
    w("        self.run(&mut d, ALL_DYNAMIC);")
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


def emit_full_clause(w, d, plan, places, name, rule, comp, recursive, refusals) -> str:
    """A clause whose head is held as bits, made once from the words held (mode bfull). With the pass start and two or more admitted orders (module text, (13)): one function for each order, `{name}v<j>`, and a chooser `{name}` that runs the order of the least
    estimate; otherwise `{name}` is the clause in the reader's order, as before. -> the name to call."""
    def one(order, fname):
        return BitWriter(d, places, fname, order, comp, plan, "bfull", None, (), recursive)
    if "start" not in plan.passes or d.static[rule.head_rel]:
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
        new = sorted((r, f) for r, i in plan.bits.items() for f in i.orients if f not in saved[r])
        if new:
            for r, i in plan.bits.items():
                i.orients = saved[r]
            plan.start_refused[(rule.clause.text.replace("\n", " "), j)] = f"it asks the orientation {new[0][1]} of {new[0][0]}, which no clause before it asks: {orders[j].clause.text.replace(chr(10), ' ')}"
            continue
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
        write_shift(buf.append, d, plan, places, f"c{cid + len(bases)}", comp, recs[0], shape)
    except ShiftRefuse as e:
        plan.shift_refused[h] = str(e)
        return None
    except BitRefuse as e:
        plan.shift_refused[h] = e.why
        return None
    names = []
    for rule in bases:
        name = f"c{cid}"
        emit_full_clause(w, d, plan, places, name, rule, comp, False, refusals)
        names.append(name)
        cid += 1
    for t in buf:
        w(t)
    names.append(f"c{cid}")
    cid += 1
    w(f"fn st{k}(s: &St, d: &mut Dy, tm: &mut Terms, dw: &mut Dw) -> (u64, u64, u64) {{")
    w("    let mut made = 0u64;")
    for c in names:
        w(f"    made += {c}(s, d, tm, dw);")
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
            base.append(name)
        cid += 1
    if static:
        w(f"fn st{k}(s: &mut St, tm: &mut Terms, dy: &Dy, dw: &mut Dw) -> (u64, u64, u64) {{")
        call = lambda c: f"{c}(s, dy, tm, dw)"
    else:
        w(f"fn st{k}(s: &St, d: &mut Dy, tm: &mut Terms, dw: &mut Dw) -> (u64, u64, u64) {{")
        call = lambda c: f"{c}(s, d, tm, dw)"
    w("    let mut made = 0u64;")
    for c in base:
        w(f"    made += {call(c)};")
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

def translate(text: str, module: str, passes=()) -> str:
    """The Rust source of the description `text` (composed) as `pub mod <module>`, with the passes named (see the module text for the interface); with none, the plain translation."""
    d = Description(text)
    p = d.prog
    for needed in ("role",):
        if needed not in p.nodes:
            raise Untranslatable(f"the description has no `{needed}`")
    plan = Plan(d, passes) if passes else None
    demanded = plan.demanded if plan else set()
    dense = plan.dense if plan else {}
    bits = plan.bits if plan else {}
    keep = bool(plan and plan.keep)
    has_dw = bool(bits) or keep
    pooled = bool(dense) or bool(bits) or keep
    out: list = []
    w = out.append
    w(f"// GENERATED by catspace/incubator/analyst/discovery/gdl_to_rust.py (plan 79.37). Do not edit: make it again.")
    w(f"// description-sha256: {sha(text.encode())}")
    w(f"// translator-sha256: {translator_hash()}")
    if plan:
        w(f"// passes: {','.join(plan.passes)}")
    w(f"#[allow(dead_code, unused_variables, unused_mut, unused_imports, unused_parens, non_snake_case, non_upper_case_globals, clippy::all{', unreachable_code' if plan else ''})]")
    w(f"pub mod {module} {{")
    w(RUNTIME_KEEP if keep else RUNTIME)
    if dense:
        w(RUNTIME_DENSE)
    if pooled:
        w(RUNTIME_REL_CLEAR)
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
    w("/// The state: the true facts, one sorted list of rows for each fluent.")
    w("#[derive(Clone, PartialEq, Eq, Hash, Debug, Default)]")
    w("pub struct State {")
    for f in d.fluents:
        w(f"    pub f_{ident(f)}: Vec<[u32; {d.arity['true.' + f]}]>,")
    w("}")
    w("pub const FLUENTS: [&str; %d] = [%s];" % (len(d.fluents), ", ".join('"%s"' % f for f in d.fluents)))
    # the relations: static ones in St, the others in Dy
    for holder, name in ((True, "St"), (False, "Dy")):
        rels = [r for r in d.nodes if d.static[r] == holder and r not in demanded]
        w(f"pub struct {name} {{")
        for r in rels:
            w(f"    pub {d.rid[r]}: {('B' + d.rid[r][1:]) if r in bits else ('Dense<%d>' % d.arity.get(r, 0) if r in dense else 'Rel<%d>' % d.arity.get(r, 0))},")
        w("}")
        new = lambda r: f"B{d.rid[r][1:]}::new()" if r in bits else (f"Dense::new(DS{d.rid[r][1:]}, RX{d.rid[r][1:]}, KP{d.rid[r][1:]})" if r in dense else f"Rel::new(KP{d.rid[r][1:]})")
        w(f"impl {name} {{ pub fn new() -> {name} {{ {name} {{ " + ", ".join(f"{d.rid[r]}: {new(r)}" for r in rels) + " } } }")
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
    if has_dw:
        w_dw(w, d, plan)
    if plan:
        w_asks(w, d, plan)
    nst = len(d.strata)
    ro = lambda holder, rel: f"{holder}.{d.rid[rel]}.rows_all()" if rel in bits else f"{holder}.{d.rid[rel]}.order"
    dwp = ", dw: &mut Dw" if has_dw else ""
    dwa = ", dw" if has_dw else ""
    w(f"pub const PIECES: [(&str, bool); {nst}] = [" + ", ".join('("%s", %s)' % ("+".join(sorted(c)), "true" if s else "false") for _, s, c in stratum_fns) + "];")
    if plan:
        w(f"pub const PASSES: &[&str] = &[{', '.join(chr(34) + x + chr(34) for x in plan.passes)}];")
        w(f"pub const COPIED: &[&str] = &[{', '.join(chr(34) + x + chr(34) for x in sorted(plan.copy))}];")
        w(f"pub const UNFOLDED: &[&str] = &[{', '.join(chr(34) + x + chr(34) for x in plan.unfolded)}];")
    dyn_all = [k for k, s, c in stratum_fns if not s and not (c & demanded)]
    coned = plan.cone if (plan and plan.unfolded) else d.cone
    legal_cone = [k for k in d.dynamic_meeting(coned([x for x in ("legal", "terminal", "goal")])) if not (d.strata[k][0] & demanded)]
    next_pairs = [f for f in d.fluents if f"next.{f}" in d.prog.nodes]
    next_cone = [k for k in d.dynamic_meeting(coned([f"next.{f}" for f in next_pairs])) if not (d.strata[k][0] & demanded)]
    w(f"fn run_piece(s: &St, d: &mut Dy, tm: &mut Terms{dwp}, k: usize) -> (u64, u64, u64) {{")
    w("    match k {")
    for k in dyn_all:
        w(f"        {k} => st{k}(s, d, tm{dwa}),")
    w("        _ => panic!(\"not a piece of the state\"),")
    w("    }")
    w("}")
    lst = lambda v: "&[" + ", ".join(map(str, v)) + "]"
    w(f"const ALL_DYNAMIC: &[usize] = {lst(dyn_all)};")
    w(f"const READ_CONE: &[usize] = {lst(legal_cone)};")
    w(f"const NEXT_CONE: &[usize] = {lst(next_cone)};")
    # the game
    w("pub struct Game { st: St, pub tm: Terms, roles: Vec<u32>, noop: u32, last: Option<(State, Reading)>, pub prof: Vec<Stat>, pub profiling: bool" + (", pool: Option<Dy>" if pooled else "") + (", dw: Dw" if has_dw else "") + (", does_buf: Vec<(u32, u32)>" if keep else "") + " }")
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
    w(f"        Game {{ st, tm, roles, noop: {d.sym_id[NOOP]}, last: None, prof: vec![Stat::default(); {nst}], profiling: false" + (", pool: Some(dy)" if pooled else "") + (", dw" if has_dw else "") + (", does_buf: Vec::new()" if keep else "") + " }")
    w("    }")
    w("    pub fn roles(&self) -> &[u32] { &self.roles }")
    w("    pub fn initial(&self) -> State {")
    w("        let mut s = State::default();")
    for f in d.fluents:
        if f"init.{f}" in d.prog.nodes:
            w(f"        s.f_{ident(f)} = {ro('self.st', 'init.' + f)}.clone();")
    w("        s.sort();")
    w("        s")
    w("    }")
    w("    /// The state's rows and the turn's into a fresh evaluation.")
    if pooled:
        w("    fn load(&mut self, s: &State, does: &[(u32, u32)]) -> Dy {")
        w("        let mut d = match self.pool.take() { Some(d) => d, None => Dy::new() };")
    else:
        w("    fn load(&self, s: &State, does: &[(u32, u32)]) -> Dy {")
        w("        let mut d = Dy::new();")
    for f in d.fluents:
        w(f"        for r in &s.f_{ident(f)} {{ d.{d.rid['true.' + f]}.add(*r); }}")
    if "does" in d.prog.nodes:
        w(f"        for (r, m) in does {{ d.{d.rid['does']}.add([*r, *m]); }}")
    w("        d")
    w("    }")
    w("    fn run(&mut self, d: &mut Dy, which: &[usize]) {")
    w("        for &k in which {")
    w("            if self.profiling {")
    w("                let t = std::time::Instant::now();")
    w(f"                let (a, b, c) = run_piece(&self.st, d, &mut self.tm{', &mut self.dw' if has_dw else ''}, k);")
    w("                let p = &mut self.prof[k];")
    w("                p.calls += 1; p.nanos += t.elapsed().as_nanos() as u64; p.rows += a; p.visits += b; p.probes += c;")
    w("            } else {")
    w(f"                run_piece(&self.st, d, &mut self.tm{', &mut self.dw' if has_dw else ''}, k);")
    w("            }")
    w("        }")
    w("    }")
    if keep:
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
    w("        let me = self.mover(s);")
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
        if plan and f"next.{f}" in plan.copy:
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
        w(f"        for r in &s.f_{ident(f)} {{ out.push((\"{f}\".to_string(), r.iter().map(|t| self.tm.render(*t)).collect())); }}")
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
    w("        self.run(&mut d, ALL_DYNAMIC);")
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
                w(f"            (\"{r}\", {{ let mut v: Vec<Vec<u32>> = {ro('d', r)}.iter().map(|x| x.to_vec()).collect(); for r in &s.f_{ident(f)} {{ if {copy_condition(d, plan, r, 'self.st', 'd', '&self.st, &d, &mut self.tm', 'r')} {{ v.push(r.to_vec()); }} }} v.sort(); v.dedup(); v }}),")
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
            w(f"        self.f_{ident(f)}.clear(); self.f_{ident(f)}.extend_from_slice(&o.f_{ident(f)});")
        w("    }")
        w("}")
    w("impl State {")
    w("    pub fn sort(&mut self) {")
    for f in d.fluents:
        w(f"        self.f_{ident(f)}.sort(); self.f_{ident(f)}.dedup();")
    w("    }")
    w("}")
    w("}")
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
