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
                64 members and the product of the sizes of the other tiles is at most DENSE_KEYS. Its form is then, for each such way (an ORIENTATION: which tile is free), an array of words with one word for each combination of the other tiles'
                numbers, and bit t of the word is set when the row with the free tile's number t is there; a list of the words that are not nought keeps the clearing and the scan proportional to what is held. A row is set in every orientation.
                A state is loaded by setting these bits; the evaluation is kept between questions and cleared by its lists of words (no row is made, hashed or freed: Part D below states what else is kept).
                CLAUSES AS WORD OPERATIONS. A clause whose literals are asked in the join order is written as nested loops over what is NOT yet bound: a positive literal whose tiles but one are bound reads one word (and, for the places of the free
                tile that are bound or constants, ands it with a mask); a literal with no tile bound is scanned by its list of words; a literal all bound is a bit test; a negated literal is a bit test, or an and-not when it stands over the same
                variables as the free tile of a literal before it; a literal over the same variables as that free tile is anded in before the bits are looped over (every later literal of the clause that stands over the same variables of the same
                tiles is folded in so); a literal read only to know that a row exists is `word != 0`; where the variables of the free tile are used by nothing but the head's last tile, the word is or-ed into the head whole, with no loop over its bits.
                A RECURSIVE SET of such relations is evaluated by the semi-naive rule on words: the words new in the last round are the first literal of each rule (scanned by their list), the rest read the whole relation; a round adds its words
                to the relation and to the next round's list. The shape recognised is exactly: every relation of the set is held as bits; this is the walk along a table of successors until a test fails, found as such because it is a clause that
                joins the set's relation, tests the same variables as its free tile, and reads a relation that takes those variables to the next ones, and is compiled by the one rule above, not by a pattern of any game.
                WHAT IT REFUSES, naming the clause: a relation with a place of a sort that has a compound or a symbol that is not a member of the sort at the place; a free tile of more than 64 members, or other tiles with more than DENSE_KEYS
                combinations, in any way a clause asks it; a place whose term is a compound; a tile whose two places carry one variable. A relation refused remains as it is (a hash relation, or dense if the pass dense accepts it) and the clauses that
                read it through its rows continue as today: a clause is written in the word form when it reads or makes a relation held as bits, and otherwise as before. A relation answered on demand (pass demand) is not held as bits.
                `Plan.bits_refused` holds each refusal with its clause.
  (6) "order" TAKES the written clauses of a stratum and the number of rows of each relation that does not read the state, known at translation by evaluating those strata once with the reader's evaluator. CONDITION: a clause has two or more
                positive literals that start equally well (the same number of bound places); RETURNS: the clause joined choosing, among those, the literal of the fewest rows (static relations by their number of rows, relations that read the state
                by the size of the product of the sorts of the places that remain unbound, the same everywhere), otherwise the reader's order. The reader's is kept where the two have the same count of bound places; the order is a different order of the same conjunction,
                with the same safety rule, and so the same rows.
  (7) "keep"  TAKES the pieces and the questions. RETURNS: the working memory of a question kept between questions where it was made at each: the lists a clause makes, the terms of a compound built again, and the state that a question returns
                built in the vector of a former one. Any place that makes and frees memory in a question where the profile shows it is named in the commit with its share before and then.
  A pass that makes a piece slower or gives a different row is a fault of the pass: it is mended, never set aside. The harness of 79.38 holds each of them to the plain module (tile numbers are an inside of the module: rows are compared as text).
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
PASSES = ("dense", "demand", "copy", "delta")   # (79.40 adds "bits", "order", "keep": the bodies follow the interface text above)
QUESTION_RELS = ("legal", "terminal", "goal")           # read in full by the questions (as are the `next` relations)
DENSE_SPACE = 1 << 22                                   # a dense relation: at most this many tuples of its sorts ...
DENSE_KEYS = 1 << 16                                    # ... and at most this many keys for any index the joins ask by
SMALL_SCAN = 8                                          # a loop over a whole fluent is cheap when its sorts allow no more rows than this
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

    def __init__(self, clause, scc: set, avoid=(), first=None):
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
                return (not (x[1] in avoid and keyed == 0 and len(x[2]) > 0), nb == len(x[2]), nb, -len(x[2]))
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
        self.passes = tuple(p for p in PASSES if p in passes)
        self.copy: dict = {}
        self.demanded: set = set()
        self.modes: dict = {}
        self.test_modes: dict = {}
        self.fname: dict = {}
        self.ask_rules: dict = {}
        self.dense: dict = {}
        self.delta: dict = {}                                   # stratum -> [(rule written in full, [(the delta literal's relation, the rule joined from the delta rows)])]   (pass "delta")
        self.refused: dict = {}                                # relation -> why pass (2) keeps it in full: the (relation, places) chain and the clause that defeats it
        self.rules: dict = {k: rules for k, (comp, rules) in enumerate(d.strata)}          # the rules of each stratum as they are written (the reader's, or joined in another order by pass (2))
        if "demand" in self.passes:
            self.find_demand()
        if "copy" in self.passes:
            self.find_copy()
        if "delta" in self.passes:
            self.find_delta()
        self.places = self.index_places()
        if "dense" in self.passes:
            self.find_dense()

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
        for k, (comp, rules) in enumerate(self.d.strata):
            if comp & demanded:
                continue
            out += [JoinRule(x.clause, comp, demanded) for x in rules] if fresh else self.rules[k]
        return out

    def compile_ask(self, rel: str, places: tuple, demanded: set) -> list:
        out = []
        for c in self.d.prog.rules_of[rel]:
            names: set = set()
            for j in places:
                GDL.term_vars(c.head[1][j], names)
            out.append((JoinRule(bind_clause(c, names), set(), demanded), c))
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
        for comp, rules in d.strata:
            if len(comp) != 1:
                continue
            r = next(iter(comp))
            if d.static[r] or r in roots or r in INPUT_RELS or r.startswith("true.") or not rules or any(x.rec_steps for x in rules):
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
        for k, (comp, rules) in enumerate(d.strata):
            if not (comp & demanded):
                self.rules[k] = [JoinRule(x.clause, comp, demanded) for x in rules]
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
                    got.append((lit[1], JoinRule(rule.clause, comp, self.demanded, lit)))
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
            if r in self.demanded or n == 0 or sig is None or len(sig) != n:
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
        counting = self.mode != "exists"
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
                get = f"{src}.{rid}.get({ix}, &key{k})"
                kp = "&[" + ", ".join(str(i) for i in keypos) + "]"
            else:
                get = f"&{src}.{rid}.order"
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
    pooled = bool(dense)
    out: list = []
    w = out.append
    w(f"// GENERATED by catspace/incubator/analyst/discovery/gdl_to_rust.py (plan 79.37). Do not edit: make it again.")
    w(f"// description-sha256: {sha(text.encode())}")
    w(f"// translator-sha256: {translator_hash()}")
    if plan:
        w(f"// passes: {','.join(plan.passes)}")
    w(f"#[allow(dead_code, unused_variables, unused_mut, unused_imports, unused_parens, non_snake_case, non_upper_case_globals, clippy::all{', unreachable_code' if plan else ''})]")
    w(f"pub mod {module} {{")
    w(RUNTIME)
    if pooled:
        w(RUNTIME_DENSE)
    w(f"pub const SYMS: [&str; {len(d.syms)}] = [{', '.join(repr(s).replace(chr(39), chr(34)) for s in d.syms)}];")
    places = plan.places if plan else d.index_places()
    for r in d.nodes:
        if r in demanded:
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
            w(f"    pub {d.rid[r]}: {'Dense' if r in dense else 'Rel'}<{d.arity.get(r, 0)}>,")
        w("}")
        new = lambda r: f"Dense::new(DS{d.rid[r][1:]}, RX{d.rid[r][1:]}, KP{d.rid[r][1:]})" if r in dense else f"Rel::new(KP{d.rid[r][1:]})"
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
    # the clauses and the pieces
    cid = 0
    stratum_fns = []
    for k, (comp, rules) in enumerate(d.strata):
        static = d.static[sorted(comp)[0]]
        if comp & demanded:
            stratum_fns.append((k, static, comp))              # answered on demand: no piece runs it
            continue
        if plan:
            rules = [x for x in plan.rules[k] if not plan.is_copy_rule(x)]
        cids = []
        dl_fns: dict = {}
        for rule in rules:
            w(ClauseWriter(d, places, cid, rule, comp, plan).write())
            cids.append((cid, rule))
            if plan and plan.delta.get(k):
                for r0, got in plan.delta[k]:
                    if r0 is rule:
                        for j, (rel, jr) in enumerate(got):
                            name = f"c{cid}d{j}"
                            w(ClauseWriter(d, places, name, jr, comp, plan, "delta").write())
                            dl_fns.setdefault(cid, []).append((name, rel))
            cid += 1
        heads = sorted({r.head_rel for r in rules})
        recursive = any(r.rec_steps for r in rules)
        X = "s" if static else "d"
        if static:
            w(f"fn st{k}(s: &mut St, tm: &mut Terms, dy: &Dy) -> (u64, u64, u64) {{")
            call = lambda c, di: f"c{c}(&*s, dy, tm, {di}, &dl, &mut o);"
            calld = lambda name: f"{name}(&*s, dy, tm, &dl, &mut o);"
        else:
            w(f"fn st{k}(s: &St, d: &mut Dy, tm: &mut Terms) -> (u64, u64, u64) {{")
            call = lambda c, di: f"c{c}(s, &*d, tm, {di}, &dl, &mut o);"
            calld = lambda name: f"{name}(s, &*d, tm, &dl, &mut o);"
        w("    let (mut made, mut visits, mut probes) = (0u64, 0u64, 0u64);")
        w("    let mut dl = Dl::default();")
        for c, rule in cids:
            h = rule.head_rel
            n = len(rule.head)
            w(f"    {{ let mut o = Out::<{n}>::new(); {call(c, 'usize::MAX')} visits += o.visits; probes += o.probes;")
            if recursive:
                w(f"      for row in o.rows {{ if {X}.{d.rid[h]}.add(row) {{ made += 1; dl.{d.rid[h]}.push(row); }} }} }}")
            else:
                w(f"      for row in o.rows {{ if {X}.{d.rid[h]}.add(row) {{ made += 1; }} }} }}")
        if recursive:
            w("    loop {")
            w("        if " + " && ".join(f"dl.{d.rid[h]}.is_empty()" for h in sorted(comp)) + " { break; }")
            for h in sorted(comp):
                n = d.arity.get(h, 0)
                w(f"        let mut nw{d.rid[h]}: Vec<[u32; {n}]> = vec![]; let mut sn{d.rid[h]}: HashSet<[u32; {n}], B> = HashSet::default();")
            for c, rule in cids:
                h = rule.head_rel
                n = len(rule.head)
                if c in dl_fns:
                    todo = [(d.rid[rel], calld(name)) for name, rel in dl_fns[c]]
                else:
                    todo = [(d.rid[rule.steps[i][1]], call(c, str(i))) for i in rule.rec_steps]
                for rr, text_call in todo:
                    w(f"        if !dl.{rr}.is_empty() {{ let mut o = Out::<{n}>::new(); {text_call} visits += o.visits; probes += o.probes;")
                    w(f"            for row in o.rows {{ if !{X}.{d.rid[h]}.has(&row) && sn{d.rid[h]}.insert(row) {{ nw{d.rid[h]}.push(row); }} }} }}")
            for h in sorted(comp):
                w(f"        for row in &nw{d.rid[h]} {{ {X}.{d.rid[h]}.add(*row); made += 1; }}")
                w(f"        dl.{d.rid[h]} = nw{d.rid[h]};")
            w("    }")
        w("    (made, visits, probes)")
        w("}")
        stratum_fns.append((k, static, comp))
    if plan:
        w_asks(w, d, plan)
    nst = len(d.strata)
    w(f"pub const PIECES: [(&str, bool); {nst}] = [" + ", ".join('("%s", %s)' % ("+".join(sorted(c)), "true" if s else "false") for _, s, c in stratum_fns) + "];")
    if plan:
        w(f"pub const PASSES: &[&str] = &[{', '.join(chr(34) + x + chr(34) for x in plan.passes)}];")
        w(f"pub const COPIED: &[&str] = &[{', '.join(chr(34) + x + chr(34) for x in sorted(plan.copy))}];")
    dyn_all = [k for k, s, c in stratum_fns if not s and not (c & demanded)]
    legal_cone = [k for k in d.dynamic_meeting(d.cone([x for x in ("legal", "terminal", "goal")])) if not (d.strata[k][0] & demanded)]
    next_pairs = [f for f in d.fluents if f"next.{f}" in d.prog.nodes]
    next_cone = [k for k in d.dynamic_meeting(d.cone([f"next.{f}" for f in next_pairs])) if not (d.strata[k][0] & demanded)]
    w("fn run_piece(s: &St, d: &mut Dy, tm: &mut Terms, k: usize) -> (u64, u64, u64) {")
    w("    match k {")
    for k in dyn_all:
        w(f"        {k} => st{k}(s, d, tm),")
    w("        _ => panic!(\"not a piece of the state\"),")
    w("    }")
    w("}")
    lst = lambda v: "&[" + ", ".join(map(str, v)) + "]"
    w(f"const ALL_DYNAMIC: &[usize] = {lst(dyn_all)};")
    w(f"const READ_CONE: &[usize] = {lst(legal_cone)};")
    w(f"const NEXT_CONE: &[usize] = {lst(next_cone)};")
    # the game
    w("pub struct Game { st: St, pub tm: Terms, roles: Vec<u32>, noop: u32, last: Option<(State, Reading)>, pub prof: Vec<Stat>, pub profiling: bool" + (", pool: Option<Dy>" if pooled else "") + " }")
    w("impl Game {")
    w(f"    pub fn new() -> Game {{")
    w("        let mut st = St::new();")
    w("        let mut tm = Terms::new();")
    w("        let dy = Dy::new();")
    for k, s, _ in stratum_fns:
        if s:
            w(f"        st{k}(&mut st, &mut tm, &dy);")
    w(f"        let roles: Vec<u32> = st.{d.rid['role']}.order.iter().map(|r| r[0]).collect();")
    w(f"        Game {{ st, tm, roles, noop: {d.sym_id[NOOP]}, last: None, prof: vec![Stat::default(); {nst}], profiling: false" + (", pool: Some(dy)" if pooled else "") + " }")
    w("    }")
    w("    pub fn roles(&self) -> &[u32] { &self.roles }")
    w("    pub fn initial(&self) -> State {")
    w("        let mut s = State::default();")
    for f in d.fluents:
        if f"init.{f}" in d.prog.nodes:
            w(f"        s.f_{ident(f)} = self.st.{d.rid['init.' + f]}.order.clone();")
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
    w("                let (a, b, c) = run_piece(&self.st, d, &mut self.tm, k);")
    w("                let p = &mut self.prof[k];")
    w("                p.calls += 1; p.nanos += t.elapsed().as_nanos() as u64; p.rows += a; p.visits += b; p.probes += c;")
    w("            } else {")
    w("                run_piece(&self.st, d, &mut self.tm, k);")
    w("            }")
    w("        }")
    w("    }")
    w("    fn read(&mut self, s: &State) -> Reading {")
    w("        if let Some((k, r)) = &self.last { if k == s { return r.clone(); } }")
    w("        let mut d = self.load(s, &[]);")
    w("        self.run(&mut d, READ_CONE);")
    lg = f"d.{d.rid['legal']}.order.iter().map(|x| (x[0], x[1])).collect()" if "legal" in d.prog.nodes else "vec![]"
    tm_ = f"!d.{d.rid['terminal']}.order.is_empty()" if "terminal" in d.prog.nodes else "false"
    gl = f"d.{d.rid['goal']}.order.iter().map(|x| (x[0], x[1])).collect()" if "goal" in d.prog.nodes else "vec![]"
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
    w("        let does: Vec<(u32, u32)> = self.roles.iter().enumerate().map(|(i, r)| (*r, if Some(i) == me { mv } else { self.noop })).collect();")
    w("        let mut d = self.load(s, &does);")
    w("        self.run(&mut d, NEXT_CONE);")
    w("        let mut n = State::default();")
    for f in next_pairs:
        if plan and f"next.{f}" in plan.copy:
            w(f"        n.f_{ident(f)} = {{ let mut v: Vec<[u32; {d.arity['true.' + f]}]> = vec![]; for r in &s.f_{ident(f)} {{ if {copy_condition(d, plan, f'next.{f}', 'self.st', 'd', '&self.st, &d, &mut self.tm', 'r')} {{ v.push(*r); }} }} v.extend_from_slice(&d.{d.rid['next.' + f]}.order); v }};")
        else:
            w(f"        n.f_{ident(f)} = d.{d.rid['next.' + f]}.order.clone();")
    w("        n.sort();")
    if pooled:
        w("        d.clear();")
        w("        self.pool = Some(d);")
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
            w(f"            (\"{r}\", self.st.{d.rid[r]}.order.iter().map(|x| x.to_vec()).collect()),")
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
                w(f"            (\"{r}\", d.{d.rid[r]}.order.iter().map(|x| x.to_vec()).collect()),")
        w("        ]")
    else:
        w("        let out = vec![")
        for r in d.nodes:
            if d.static[r] or r in demanded:
                continue
            if r in plan.copy:
                f = r[len("next."):]
                w(f"            (\"{r}\", {{ let mut v: Vec<Vec<u32>> = d.{d.rid[r]}.order.iter().map(|x| x.to_vec()).collect(); for r in &s.f_{ident(f)} {{ if {copy_condition(d, plan, r, 'self.st', 'd', '&self.st, &d, &mut self.tm', 'r')} {{ v.push(r.to_vec()); }} }} v.sort(); v.dedup(); v }}),")
            else:
                w(f"            (\"{r}\", d.{d.rid[r]}.order.iter().map(|x| x.to_vec()).collect()),")
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
