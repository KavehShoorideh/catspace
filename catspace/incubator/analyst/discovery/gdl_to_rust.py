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

class ClauseWriter:
    def __init__(self, d: Description, places: dict, cid: int, rule, scc: set):
        self.d, self.places, self.cid, self.rule, self.scc = d, places, cid, rule, scc
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
        return "continue;" if self.loops else "return;"

    def holder(self, rel: str) -> str:
        return "s" if self.d.static[rel] else "d"

    def build(self, p) -> str:
        if p[0] == "c":
            return str(self.d.sym_id[p[1]])
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
            self.out(f"if {e} != {self.d.sym_id[p[1]]} {{ {self.fail()} }}")
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
        if k == len(r.steps):
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
            self.out(f"if {self.holder(st[1])}.{self.d.rid[st[1]]}.has(&[{', '.join(args)}]) {{ {self.fail()} }}")
            self.steps(k + 1)
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
            self.out("o.probes += 1;")
            if k in r.rec_steps:
                self.out(f"let tmp{k};")
                self.out(f"let rows{k}: &[[u32; {n}]] = if di == {k} {{ tmp{k} = scan(&dl.{rid}, {kp}, &key{k}); &tmp{k} }} else {{ {get} }};")
            else:
                self.out(f"let rows{k}: &[[u32; {n}]] = {get};")
            self.out(f"for row{k} in rows{k}.iter() {{")
            self.depth += 1
            self.loops += 1
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
        self.lines.append(f"fn c{self.cid}(s: &St, d: &Dy, tm: &mut Terms, di: usize, dl: &Dl, o: &mut Out<{n}>) {{")
        self.steps(0)
        self.lines.append("}")
        return "\n".join(self.lines)


# ------------------------------------------------------------------------------------------------------------------------------------------ the document

def translate(text: str, module: str) -> str:
    """The Rust source of the description `text` (composed) as `pub mod <module>`. See the module text for the interface."""
    d = Description(text)
    p = d.prog
    for needed in ("role",):
        if needed not in p.nodes:
            raise Untranslatable(f"the description has no `{needed}`")
    out: list = []
    w = out.append
    w(f"// GENERATED by catspace/incubator/analyst/discovery/gdl_to_rust.py (plan 79.37). Do not edit: make it again.")
    w(f"// description-sha256: {sha(text.encode())}")
    w(f"// translator-sha256: {translator_hash()}")
    w(f"#[allow(dead_code, unused_variables, unused_mut, unused_imports, unused_parens, non_snake_case, non_upper_case_globals, clippy::all)]")
    w(f"pub mod {module} {{")
    w(RUNTIME)
    w(f"pub const SYMS: [&str; {len(d.syms)}] = [{', '.join(repr(s).replace(chr(39), chr(34)) for s in d.syms)}];")
    places = d.index_places()
    for r in d.nodes:
        pl = places[r]
        w(f"const KP{d.rid[r][1:]}: &[&[usize]] = &[{', '.join('&[' + ', '.join(map(str, t)) + ']' for t in pl)}];")
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
        rels = [r for r in d.nodes if d.static[r] == holder]
        w(f"pub struct {name} {{")
        for r in rels:
            w(f"    pub {d.rid[r]}: Rel<{d.arity.get(r, 0)}>,")
        w("}")
        w(f"impl {name} {{ pub fn new() -> {name} {{ {name} {{ " + ", ".join(f"{d.rid[r]}: Rel::new(KP{d.rid[r][1:]})" for r in rels) + " } } }")
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
        cids = []
        for rule in rules:
            w(ClauseWriter(d, places, cid, rule, comp).write())
            cids.append((cid, rule))
            cid += 1
        heads = sorted({r.head_rel for r in rules})
        recursive = any(r.rec_steps for r in rules)
        X = "s" if static else "d"
        if static:
            w(f"fn st{k}(s: &mut St, tm: &mut Terms, dy: &Dy) -> (u64, u64, u64) {{")
            call = lambda c, di: f"c{c}(&*s, dy, tm, {di}, &dl, &mut o);"
        else:
            w(f"fn st{k}(s: &St, d: &mut Dy, tm: &mut Terms) -> (u64, u64, u64) {{")
            call = lambda c, di: f"c{c}(s, &*d, tm, {di}, &dl, &mut o);"
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
                for i in rule.rec_steps:
                    rr = d.rid[rule.steps[i][1]]
                    w(f"        if !dl.{rr}.is_empty() {{ let mut o = Out::<{n}>::new(); {call(c, str(i))} visits += o.visits; probes += o.probes;")
                    w(f"            for row in o.rows {{ if !{X}.{d.rid[h]}.has(&row) && sn{d.rid[h]}.insert(row) {{ nw{d.rid[h]}.push(row); }} }} }}")
            for h in sorted(comp):
                w(f"        for row in &nw{d.rid[h]} {{ {X}.{d.rid[h]}.add(*row); made += 1; }}")
                w(f"        dl.{d.rid[h]} = nw{d.rid[h]};")
            w("    }")
        w("    (made, visits, probes)")
        w("}")
        stratum_fns.append((k, static, comp))
    nst = len(d.strata)
    w(f"pub const PIECES: [(&str, bool); {nst}] = [" + ", ".join('("%s", %s)' % ("+".join(sorted(c)), "true" if s else "false") for _, s, c in stratum_fns) + "];")
    dyn_all = [k for k, s, _ in stratum_fns if not s]
    legal_cone = d.dynamic_meeting(d.cone([x for x in ("legal", "terminal", "goal")]))
    next_pairs = [f for f in d.fluents if f"next.{f}" in d.prog.nodes]
    next_cone = d.dynamic_meeting(d.cone([f"next.{f}" for f in next_pairs]))
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
    w("pub struct Game { st: St, pub tm: Terms, roles: Vec<u32>, noop: u32, last: Option<(State, Reading)>, pub prof: Vec<Stat>, pub profiling: bool }")
    w("impl Game {")
    w(f"    pub fn new() -> Game {{")
    w("        let mut st = St::new();")
    w("        let mut tm = Terms::new();")
    w("        let dy = Dy::new();")
    for k, s, _ in stratum_fns:
        if s:
            w(f"        st{k}(&mut st, &mut tm, &dy);")
    w(f"        let roles: Vec<u32> = st.{d.rid['role']}.order.iter().map(|r| r[0]).collect();")
    w(f"        Game {{ st, tm, roles, noop: {d.sym_id[NOOP]}, last: None, prof: vec![Stat::default(); {nst}], profiling: false }}")
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
        w(f"        n.f_{ident(f)} = d.{d.rid['next.' + f]}.order.clone();")
    w("        n.sort();")
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
    w("        vec![")
    for r in d.nodes:
        if not d.static[r]:
            w(f"            (\"{r}\", d.{d.rid[r]}.order.iter().map(|x| x.to_vec()).collect()),")
    w("        ]")
    w("    }")
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
