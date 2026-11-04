//! PLAN 76.4b (a) THE SPEC IN RUST: a GDL description read at run time from its text (rules/chess.gdl: the file is the source -- no clause of it is written
//! here) -- the mirror of discovery/gdl.py. Nothing in this module knows a game.
//!
//! THE LANGUAGE READ (gdl.py's): facts `(rel a b)` and rules `(<= head lit ...)`; a literal is an atom, `(not atom)`, `(distinct t1 t2)` or `(or lit ...)`
//! (expanded into one clause per branch); variables are `?x`; terms are constants or compound `(f t ...)`; `(true (f ...))`, `(next (f ...))` and
//! `(init (f ...))` are stored flattened by the fluent's functor (relation `true.f` ...). The NAMED EXTENSIONS of 76.1 (b): `(sort ...)` and
//! `(signature ...)` are declarations, not clauses (kept: a relation's signature types its slots); `record` is an input relation like `does`.
//!
//! THE EVALUATION (gdl.py `Program.run`): the dependency graph cut into strongly connected components (a negative edge inside one is an error: the
//! description must be stratified), evaluated bottom-up in topological order, a recursive component by SEMI-NAIVE iteration (each round joins one body
//! literal against the last round's new facts, the others against everything known); the STATIC relations (reading neither `true`, `does` nor `record`)
//! once, at load. Why not the crate's `ascent` Datalog: ascent compiles its clauses into Rust at build time -- the spec would have to be transcribed into
//! the macro by hand, and a description loaded from a file at run time cannot be.
//!
//! Terms are interned (`Interner`): a ground term is a u32 -- a constant or a compound (functor, args). A row is a fixed array of 8 term ids (the widest
//! relation of chess.gdl has 8 arguments; FREE pads the rest).
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

pub type T = u32;
pub const FREE: T = u32::MAX;
pub const ARITY_MAX: usize = 8;
pub type Row = [T; ARITY_MAX];
pub type Rel = u16;

#[derive(Debug)]
pub struct GdlError(pub String);
impl std::fmt::Display for GdlError { fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { f.write_str(&self.0) } }

fn err<X>(s: String) -> Result<X, GdlError> { Err(GdlError(s)) }

// ------------------------------------------------------------------------------------------------------------------------------------------ terms

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum TermData { Sym(String), Fun(T, Box<[T]>) }

/// Every ground term of an evaluation, interned: a constant (a symbol) or a compound (functor symbol, argument terms).
#[derive(Default, Clone)]
pub struct Interner {
    data: Vec<TermData>,
    ids: HashMap<TermData, T>,
}

impl Interner {
    pub fn sym(&mut self, s: &str) -> T {
        let k = TermData::Sym(s.to_string());
        if let Some(&i) = self.ids.get(&k) { return i; }
        let i = self.data.len() as T;
        self.data.push(k.clone());
        self.ids.insert(k, i);
        i
    }
    pub fn fun(&mut self, f: T, args: &[T]) -> T {
        let k = TermData::Fun(f, args.to_vec().into_boxed_slice());
        if let Some(&i) = self.ids.get(&k) { return i; }
        let i = self.data.len() as T;
        self.data.push(k.clone());
        self.ids.insert(k, i);
        i
    }
    pub fn len(&self) -> usize { self.data.len() }
    /// The compound's functor and arguments (None: a constant).
    pub fn compound(&self, t: T) -> Option<(T, &[T])> {
        match &self.data[t as usize] { TermData::Fun(f, a) => Some((*f, a)), TermData::Sym(_) => None }
    }
    pub fn name(&self, t: T) -> &str {
        match &self.data[t as usize] { TermData::Sym(s) => s, TermData::Fun(f, _) => self.name(*f) }
    }
}

/// A term of a clause: a variable (by name), a constant, or a compound.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Term { Var(String), Const(T), Fun(T, Vec<Term>) }

impl Term {
    pub fn vars(&self, acc: &mut BTreeSet<String>) {
        match self { Term::Var(v) => { acc.insert(v.clone()); } Term::Fun(_, a) => { for x in a { x.vars(acc); } } Term::Const(_) => {} }
    }
}

#[derive(Clone, Debug)]
pub enum Lit { Pos(Rel, Vec<Term>), Neg(Rel, Vec<Term>), Distinct(Term, Term) }

#[derive(Clone, Debug)]
pub struct Clause { pub head: (Rel, Vec<Term>), pub body: Vec<Lit>, pub text: String }

// ------------------------------------------------------------------------------------------------------------------------------------------ reading

#[derive(Clone, Debug)]
enum Sx { Sym(String), List(Vec<Sx>) }

fn tokenize(text: &str) -> Vec<String> {
    let mut out = vec![];
    for line in text.lines() {
        let line = line.split(';').next().unwrap_or("");
        let mut cur = String::new();
        for ch in line.chars() {
            if ch == '(' || ch == ')' || ch.is_whitespace() {
                if !cur.is_empty() { out.push(std::mem::take(&mut cur)); }
                if ch == '(' || ch == ')' { out.push(ch.to_string()); }
            } else {
                cur.push(ch);
            }
        }
        if !cur.is_empty() { out.push(cur); }
    }
    out
}

fn parse(text: &str) -> Result<Vec<Sx>, GdlError> {
    let mut out = vec![];
    let mut stack: Vec<Vec<Sx>> = vec![];
    for t in tokenize(text) {
        if t == "(" { stack.push(vec![]); }
        else if t == ")" {
            let x = stack.pop().ok_or_else(|| GdlError("unbalanced ')'".into()))?;
            match stack.last_mut() { Some(top) => top.push(Sx::List(x)), None => out.push(Sx::List(x)) }
        } else {
            let s = Sx::Sym(t.to_lowercase());
            match stack.last_mut() { Some(top) => top.push(s), None => out.push(s) }
        }
    }
    if !stack.is_empty() { return err("unbalanced '('".into()); }
    Ok(out)
}

fn unparse(x: &Sx) -> String {
    match x { Sx::Sym(s) => s.clone(), Sx::List(v) => format!("({})", v.iter().map(unparse).collect::<Vec<_>>().join(" ")) }
}

/// plan 78.1: A DESCRIPTION COMPOSED FROM A BASE AND A DELTA -- a variant's rules written as the base's (rules/chess.gdl) less the clauses the delta retracts
/// (`(retract <clause>)`: it must match a top-level form of the base exactly, as unparsed, or the composition fails -- a drifted base is caught) plus the delta's
/// own forms. -> the composed text (one form per line).
pub fn compose(base: &str, delta: &str) -> Result<String, GdlError> {
    let mut forms: Vec<String> = parse(base)?.iter().map(unparse).collect();
    let mut added = vec![];
    for x in parse(delta)? {
        if let Sx::List(v) = &x {
            if matches!(v.first(), Some(Sx::Sym(h)) if h == "retract") {
                if v.len() != 2 { return err(format!("(retract <clause>) takes one form: {}", unparse(&x))); }
                let target = unparse(&v[1]);
                let before = forms.len();
                forms.retain(|f| *f != target);
                if forms.len() == before { return err(format!("retract: no form of the base reads {target}")); }
                continue;
            }
        }
        added.push(unparse(&x));
    }
    forms.extend(added);
    Ok(forms.join("\n"))
}

pub const FLAT: [&str; 3] = ["true", "next", "init"];
pub const INPUTS: [&str; 2] = ["does", "record"];

pub fn is_input(rel: &str) -> bool { rel.starts_with("true.") || INPUTS.contains(&rel) }

/// A sort's member: a constant, or a compound pattern (functor, argument sorts) -- kept as gdl.py keeps them (the type check stays gdl.py's).
#[derive(Clone, Debug)]
#[allow(dead_code)]
pub enum SortMember { Const(String), Pattern(Vec<String>) }

/// A loaded description: its relations (names <-> ids), clauses, strata, signatures and the static relations evaluated once.
pub struct Program {
    pub terms: Interner,
    pub rel_names: Vec<String>,
    pub rel_ids: HashMap<String, Rel>,
    pub arity: Vec<Option<usize>>,
    pub clauses: Vec<Clause>,
    pub rules_of: Vec<Vec<usize>>,                     // rel -> its clauses' indices (file order)
    pub deps: Vec<BTreeSet<Rel>>,
    pub neg: Vec<BTreeSet<Rel>>,
    pub nodes: BTreeSet<Rel>,
    pub strata: Vec<(BTreeSet<Rel>, Vec<CRule>)>,
    pub reads_input: Vec<bool>,
    pub statics: HashMap<Rel, Relation>,              // the static relations' facts (every relation of a stratum that reads no input)
    pub sorts: BTreeMap<String, Vec<SortMember>>,
    pub signatures: HashMap<String, Vec<String>>,
}

impl Program {
    fn rel(&mut self, name: &str) -> Rel {
        if let Some(&r) = self.rel_ids.get(name) { return r; }
        let r = self.rel_names.len() as Rel;
        self.rel_names.push(name.to_string());
        self.rel_ids.insert(name.to_string(), r);
        self.arity.push(None);
        self.rules_of.push(vec![]);
        self.deps.push(BTreeSet::new());
        self.neg.push(BTreeSet::new());
        self.reads_input.push(false);
        r
    }
    pub fn rel_id(&self, name: &str) -> Option<Rel> { self.rel_ids.get(name).copied() }
    pub fn rel_name(&self, r: Rel) -> &str { &self.rel_names[r as usize] }
    pub fn nrels(&self) -> usize { self.rel_names.len() }

    fn note_arity(&mut self, r: Rel, n: usize, text: &str) -> Result<(), GdlError> {
        if n > ARITY_MAX { return err(format!("{} takes {n} arguments (at most {ARITY_MAX} are read): {text}", self.rel_names[r as usize])); }
        match self.arity[r as usize] {
            None => { self.arity[r as usize] = Some(n); Ok(()) }
            Some(m) if m == n => Ok(()),
            Some(m) => err(format!("{} is used with {m} and with {n} arguments: {text}", self.rel_names[r as usize])),
        }
    }

    fn to_term(&mut self, x: &Sx) -> Result<Term, GdlError> {
        match x {
            Sx::Sym(s) => Ok(if s.starts_with('?') { Term::Var(s.clone()) } else { Term::Const(self.terms.sym(s)) }),
            Sx::List(v) => {
                let Some(Sx::Sym(f)) = v.first() else { return err(format!("empty term or a compound term without a functor symbol: {}", unparse(x))) };
                if f.starts_with('?') { return err(format!("a variable functor: {}", unparse(x))); }
                let f = self.terms.sym(f);
                let args = v[1..].iter().map(|y| self.to_term(y)).collect::<Result<Vec<_>, _>>()?;
                Ok(Term::Fun(f, args))
            }
        }
    }

    fn to_atom(&mut self, x: &Sx, text: &str) -> Result<(Rel, Vec<Term>), GdlError> {
        match x {
            Sx::Sym(s) => { let r = self.rel(s); self.note_arity(r, 0, text)?; Ok((r, vec![])) }
            Sx::List(v) => {
                let Some(Sx::Sym(rel)) = v.first() else { return err(format!("relation name is not a symbol: {}", unparse(x))) };
                if FLAT.contains(&rel.as_str()) {
                    if v.len() != 2 { return err(format!("({rel} ...) takes one fluent: {}", unparse(x))); }
                    return match &v[1] {
                        Sx::Sym(f) => {
                            if f.starts_with('?') { return err(format!("a variable fluent ({rel} {f}) is not supported")); }
                            let r = self.rel(&format!("{rel}.{f}"));
                            self.note_arity(r, 0, text)?;
                            Ok((r, vec![]))
                        }
                        Sx::List(w) => {
                            let Some(Sx::Sym(f)) = w.first() else { return err(format!("a fluent without a functor: {}", unparse(x))) };
                            let r = self.rel(&format!("{rel}.{f}"));
                            let args = w[1..].iter().map(|y| self.to_term(y)).collect::<Result<Vec<_>, _>>()?;
                            self.note_arity(r, args.len(), text)?;
                            Ok((r, args))
                        }
                    };
                }
                let r = self.rel(rel);
                let args = v[1..].iter().map(|y| self.to_term(y)).collect::<Result<Vec<_>, _>>()?;
                self.note_arity(r, args.len(), text)?;
                Ok((r, args))
            }
        }
    }

    /// One body literal -> its alternatives (each a list of plain literals): `or` expands into one alternative per branch.
    fn literals(&mut self, x: &Sx, text: &str) -> Result<Vec<Vec<Lit>>, GdlError> {
        if let Sx::List(v) = x {
            if let Some(Sx::Sym(h)) = v.first() {
                if h == "or" {
                    let mut alts = vec![];
                    for y in &v[1..] { alts.extend(self.literals(y, text)?); }
                    return Ok(alts);
                }
                if h == "not" {
                    if v.len() != 2 { return err(format!("(not ...) takes one atom: {}", unparse(x))); }
                    if let Sx::List(w) = &v[1] {
                        if let Some(Sx::Sym(k)) = w.first() { if k == "or" || k == "not" || k == "distinct" { return err(format!("(not {k} ...) is not supported: {}", unparse(x))); } }
                    }
                    let (r, a) = self.to_atom(&v[1], text)?;
                    return Ok(vec![vec![Lit::Neg(r, a)]]);
                }
                if h == "distinct" {
                    if v.len() != 3 { return err(format!("(distinct ...) takes two terms: {}", unparse(x))); }
                    return Ok(vec![vec![Lit::Distinct(self.to_term(&v[1])?, self.to_term(&v[2])?)]]);
                }
            }
        }
        let (r, a) = self.to_atom(x, text)?;
        Ok(vec![vec![Lit::Pos(r, a)]])
    }

    /// Load a description from its text: the declarations kept, the clauses read, the strata cut, the static relations evaluated.
    pub fn load(text: &str) -> Result<Program, GdlError> {
        let mut p = Program {
            terms: Interner::default(), rel_names: vec![], rel_ids: HashMap::new(), arity: vec![], clauses: vec![], rules_of: vec![], deps: vec![], neg: vec![],
            nodes: BTreeSet::new(), strata: vec![], reads_input: vec![], statics: HashMap::new(), sorts: BTreeMap::new(), signatures: HashMap::new(),
        };
        for x in parse(text)? {
            if let Sx::List(v) = &x {
                if let Some(Sx::Sym(h)) = v.first() {
                    if h == "sort" || h == "signature" {
                        let Some(Sx::Sym(name)) = v.get(1) else { return err(format!("a declaration names what it declares: {}", unparse(&x))) };
                        if h == "sort" {
                            if p.sorts.contains_key(name) { return err(format!("sort {name} declared twice")); }
                            let mut mem = vec![];
                            for y in &v[2..] {
                                match y {
                                    Sx::List(w) if matches!(w.first(), Some(Sx::Sym(r)) if r == "range") => {
                                        let num = |i: usize| -> Result<i64, GdlError> {
                                            match w.get(i) { Some(Sx::Sym(s)) => s.parse::<i64>().map_err(|_| GdlError(format!("a range bound is a numeral: {}", unparse(y)))),
                                                             _ => err(format!("(range lo hi): {}", unparse(y))) }
                                        };
                                        for n in num(1)?..=num(2)? { mem.push(SortMember::Const(n.to_string())); }
                                    }
                                    Sx::List(w) => mem.push(SortMember::Pattern(w.iter().map(|z| match z { Sx::Sym(s) => s.clone(), _ => unparse(z) }).collect())),
                                    Sx::Sym(s) => mem.push(SortMember::Const(s.clone())),
                                }
                            }
                            p.sorts.insert(name.clone(), mem);
                        } else {
                            if p.signatures.contains_key(name) { return err(format!("signature of {name} declared twice")); }
                            p.signatures.insert(name.clone(), v[2..].iter().map(|z| match z { Sx::Sym(s) => s.clone(), _ => unparse(z) }).collect());
                        }
                        continue;
                    }
                }
            }
            let text = unparse(&x);
            match &x {
                Sx::List(v) if matches!(v.first(), Some(Sx::Sym(h)) if h == "<=") => {
                    if v.len() < 2 { return err(format!("a rule without a head: {text}")); }
                    let head = p.to_atom(&v[1], &text)?;
                    let mut bodies: Vec<Vec<Lit>> = vec![vec![]];
                    for lit in &v[2..] {
                        let alts = p.literals(lit, &text)?;
                        let mut nb = vec![];
                        for b in &bodies { for a in &alts { let mut c = b.clone(); c.extend(a.iter().cloned()); nb.push(c); } }
                        bodies = nb;
                    }
                    for b in bodies { p.clauses.push(Clause { head: head.clone(), body: b, text: text.clone() }); }
                }
                _ => { let head = p.to_atom(&x, &text)?; p.clauses.push(Clause { head, body: vec![], text }); }
            }
        }
        for (i, c) in p.clauses.iter().enumerate() {
            let h = c.head.0;
            p.nodes.insert(h);
            p.rules_of[h as usize].push(i);
            for l in &c.body {
                match l {
                    Lit::Pos(r, _) => { p.nodes.insert(*r); p.deps[h as usize].insert(*r); }
                    Lit::Neg(r, _) => { p.nodes.insert(*r); p.deps[h as usize].insert(*r); p.neg[h as usize].insert(*r); }
                    Lit::Distinct(..) => {}
                }
            }
        }
        // the strongly connected components (Tarjan), each after every component it depends on
        let comps = p.sccs();
        let mut strata = vec![];
        for comp in comps {
            for &h in &comp {
                let bad: Vec<&str> = p.neg[h as usize].iter().filter(|r| comp.contains(r)).map(|r| p.rel_name(*r)).collect();
                if !bad.is_empty() { return err(format!("not stratified: {} depends negatively on {bad:?} in its own component", p.rel_name(h))); }
            }
            let mut rules = vec![];
            for &h in &comp { for &ci in &p.rules_of[h as usize] { rules.push(CRule::compile(&p.clauses[ci], ci, &comp)?); } }
            strata.push((comp, rules));
        }
        for (comp, _) in &strata {
            let r = comp.iter().any(|h| is_input(p.rel_name(*h)))
                || comp.iter().any(|h| p.deps[*h as usize].iter().any(|d| if comp.contains(d) { false } else { p.reads_input[*d as usize] || is_input(p.rel_name(*d)) }));
            for &h in comp { p.reads_input[h as usize] = r; }
        }
        p.strata = strata;
        // the static relations, evaluated once
        let mut db = Db::new(p.nrels());
        for si in 0..p.strata.len() {
            let comp = p.strata[si].0.clone();
            if comp.iter().any(|h| p.reads_input[*h as usize]) { continue; }
            p.eval_stratum(&mut db, si);
        }
        for (r, rel) in db.rels.into_iter().enumerate() {
            if let Some(rel) = rel { p.statics.insert(r as Rel, rel); }
        }
        Ok(p)
    }

    fn sccs(&self) -> Vec<BTreeSet<Rel>> {
        struct St { index: HashMap<Rel, usize>, low: HashMap<Rel, usize>, on: HashSet<Rel>, stack: Vec<Rel>, out: Vec<BTreeSet<Rel>>, counter: usize }
        fn visit(p: &Program, v: Rel, s: &mut St) {
            s.index.insert(v, s.counter);
            s.low.insert(v, s.counter);
            s.counter += 1;
            s.stack.push(v);
            s.on.insert(v);
            for &w in &p.deps[v as usize] {
                if !s.index.contains_key(&w) {
                    visit(p, w, s);
                    let lw = s.low[&w];
                    let lv = s.low.get_mut(&v).unwrap();
                    *lv = (*lv).min(lw);
                } else if s.on.contains(&w) {
                    let iw = s.index[&w];
                    let lv = s.low.get_mut(&v).unwrap();
                    *lv = (*lv).min(iw);
                }
            }
            if s.low[&v] == s.index[&v] {
                let mut comp = BTreeSet::new();
                loop {
                    let w = s.stack.pop().unwrap();
                    s.on.remove(&w);
                    comp.insert(w);
                    if w == v { break; }
                }
                s.out.push(comp);
            }
        }
        let mut s = St { index: HashMap::new(), low: HashMap::new(), on: HashSet::new(), stack: vec![], out: vec![], counter: 0 };
        let mut names: Vec<Rel> = self.nodes.iter().copied().collect();
        names.sort_by(|a, b| self.rel_name(*a).cmp(self.rel_name(*b)));
        for v in names { if !s.index.contains_key(&v) { visit(self, v, &mut s); } }
        s.out
    }

    /// A relation's declared signature (a fluent's `true` / `next` / `init` forms read the fluent's).
    pub fn signature(&self, rel: &str) -> Option<&Vec<String>> {
        for k in FLAT { if let Some(rest) = rel.strip_prefix(&format!("{k}.")) { return self.signatures.get(rest); } }
        self.signatures.get(rel)
    }

    // ---------------------------------------------------------------------------------------------- the bottom-up evaluation

    /// Evaluate the description on a state (`true.*` facts), the joint move `does` and the other inputs, every relation computed (gdl.py `run`, wanted None).
    pub fn run(&mut self, facts: &[(Rel, Row)]) -> Db {
        let mut db = Db::new(self.nrels());
        for (r, rel) in &self.statics { db.rels[*r as usize] = Some(rel.clone()); }
        for (r, row) in facts { db.writable(*r).add(*row); }
        for si in 0..self.strata.len() {
            let comp = &self.strata[si].0;
            if !comp.iter().any(|h| self.reads_input[*h as usize]) { continue; }
            self.eval_stratum(&mut db, si);
        }
        db
    }

    fn eval_stratum(&mut self, db: &mut Db, si: usize) {
        let comp = self.strata[si].0.clone();
        for &h in &comp { db.writable(h); }
        let rules = std::mem::take(&mut self.strata[si].1);
        let recursive = rules.iter().any(|r| !r.rec_steps.is_empty());
        let mut delta: BTreeMap<Rel, Vec<Row>> = BTreeMap::new();
        for r in &rules {
            for t in self.fire(db, r, None, &BTreeMap::new()) {
                if db.writable(r.head_rel).add(t) { delta.entry(r.head_rel).or_default().push(t); }
            }
        }
        if recursive {
            while delta.values().any(|v| !v.is_empty()) {
                let dview: BTreeMap<Rel, Relation> = delta.iter().filter(|(_, v)| !v.is_empty()).map(|(k, v)| (*k, Relation::from_rows(v))).collect();
                let mut new: BTreeMap<Rel, Vec<Row>> = BTreeMap::new();
                for r in &rules {
                    for &i in &r.rec_steps {
                        let CStep::Pos(rel, _, _) = &r.steps[i] else { continue };
                        if !dview.contains_key(rel) { continue; }
                        for t in self.fire(db, r, Some(i), &dview) {
                            if !db.rels[r.head_rel as usize].as_ref().is_some_and(|x| x.facts.contains(&t)) {
                                let e = new.entry(r.head_rel).or_default();
                                if !e.contains(&t) { e.push(t); }
                            }
                        }
                    }
                }
                for (rel, ts) in &new { let w = db.writable(*rel); for t in ts { w.add(*t); } }
                delta = new;
            }
        }
        self.strata[si].1 = rules;
    }

    fn fire(&mut self, db: &mut Db, rule: &CRule, delta_at: Option<usize>, dview: &BTreeMap<Rel, Relation>) -> Vec<Row> {
        let mut out = vec![];
        let mut env = vec![FREE; rule.nvars];
        self.go(db, rule, 0, &mut env, delta_at, dview, &mut out);
        out
    }

    #[allow(clippy::too_many_arguments)]
    fn go(&mut self, db: &mut Db, rule: &CRule, k: usize, env: &mut [T], delta_at: Option<usize>, dview: &BTreeMap<Rel, Relation>, out: &mut Vec<Row>) {
        if k == rule.steps.len() {
            let mut row = [FREE; ARITY_MAX];
            for (i, p) in rule.head.iter().enumerate() { row[i] = build(&mut self.terms, p, env); }
            out.push(row);
            return;
        }
        match &rule.steps[k] {
            CStep::Pos(rel, args, keypos) => {
                let key: Vec<T> = keypos.iter().map(|&i| build(&mut self.terms, &args[i], env)).collect();
                let rows: Vec<Row> = if delta_at == Some(k) {
                    match dview.get(rel) { Some(r) => r.scan(keypos, &key), None => return }
                } else {
                    match db.rels[*rel as usize].as_mut() { Some(r) => r.lookup_rows(keypos, &key), None => return }
                };
                for t in rows {
                    let mut trail = vec![];
                    let mut ok = true;
                    for (i, a) in args.iter().enumerate() { if !mtch(&self.terms, a, t[i], env, &mut trail) { ok = false; break; } }
                    if ok { self.go(db, rule, k + 1, env, delta_at, dview, out); }
                    for v in trail { env[v as usize] = FREE; }
                }
            }
            CStep::Neg(rel, args) => {
                let mut row = [FREE; ARITY_MAX];
                for (i, a) in args.iter().enumerate() { row[i] = build(&mut self.terms, a, env); }
                let has = db.rels[*rel as usize].as_ref().is_some_and(|r| r.facts.contains(&row));
                if !has { self.go(db, rule, k + 1, env, delta_at, dview, out); }
            }
            CStep::Distinct(a, b) => {
                if build(&mut self.terms, a, env) != build(&mut self.terms, b, env) { self.go(db, rule, k + 1, env, delta_at, dview, out); }
            }
        }
    }
}

// ------------------------------------------------------------------------------------------------------------------------------------------ compiled terms

/// A compiled term: a variable (by number), a constant, a compound.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum P { V(u16), C(T), F(T, Vec<P>) }

pub fn comp(t: &Term, names: &mut Vec<String>) -> P {
    match t {
        Term::Var(v) => {
            let i = match names.iter().position(|n| n == v) { Some(i) => i, None => { names.push(v.clone()); names.len() - 1 } };
            P::V(i as u16)
        }
        Term::Const(c) => P::C(*c),
        Term::Fun(f, a) => P::F(*f, a.iter().map(|x| comp(x, names)).collect()),
    }
}

pub fn pvars(p: &P, acc: &mut BTreeSet<u16>) {
    match p { P::V(i) => { acc.insert(*i); } P::F(_, a) => { for x in a { pvars(x, acc); } } P::C(_) => {} }
}

/// The ground term of a compiled term under `env` (FREE when a variable in it is unbound).
pub fn build(it: &mut Interner, p: &P, env: &[T]) -> T {
    match p {
        P::C(c) => *c,
        P::V(i) => env[*i as usize],
        P::F(f, a) => {
            let mut args = Vec::with_capacity(a.len());
            for q in a { let v = build(it, q, env); if v == FREE { return FREE; } args.push(v); }
            it.fun(*f, &args)
        }
    }
}

/// Match a compiled term against a ground term, binding unbound variables (recorded on `trail`).
pub fn mtch(it: &Interner, p: &P, v: T, env: &mut [T], trail: &mut Vec<u16>) -> bool {
    match p {
        P::C(c) => *c == v,
        P::V(i) => {
            let cur = env[*i as usize];
            if cur == FREE { env[*i as usize] = v; trail.push(*i); true } else { cur == v }
        }
        P::F(f, a) => {
            if v == FREE { return false; }
            match it.compound(v) {
                Some((g, ga)) if g == *f && ga.len() == a.len() => {
                    let ga: Vec<T> = ga.to_vec();
                    for (q, x) in a.iter().zip(ga) { if !mtch(it, q, x, env, trail) { return false; } }
                    true
                }
                _ => false,
            }
        }
    }
}

#[derive(Clone, Debug)]
pub enum CStep { Pos(Rel, Vec<P>, Vec<usize>), Neg(Rel, Vec<P>), Distinct(P, P) }

/// A clause compiled for the bottom-up join (gdl.py `_Rule`): variables numbered, the body in its join order (the next positive literal with the most
/// bound arguments; a negation or a distinct as soon as its variables are bound), each positive literal with the argument positions bound when reached.
#[derive(Clone, Debug)]
#[allow(dead_code)]
pub struct CRule { pub clause: usize, pub nvars: usize, pub head_rel: Rel, pub head: Vec<P>, pub steps: Vec<CStep>, pub rec_steps: Vec<usize> }

fn term_vars(t: &Term) -> BTreeSet<String> { let mut s = BTreeSet::new(); t.vars(&mut s); s }

fn lit_vars(l: &Lit) -> BTreeSet<String> {
    let mut s = BTreeSet::new();
    match l { Lit::Pos(_, a) | Lit::Neg(_, a) => { for x in a { x.vars(&mut s); } } Lit::Distinct(a, b) => { a.vars(&mut s); b.vars(&mut s); } }
    s
}

impl CRule {
    fn compile(c: &Clause, ci: usize, scc: &BTreeSet<Rel>) -> Result<CRule, GdlError> {
        let pos: Vec<&Lit> = c.body.iter().filter(|l| matches!(l, Lit::Pos(..))).collect();
        let rest: Vec<&Lit> = c.body.iter().filter(|l| !matches!(l, Lit::Pos(..))).collect();
        let mut bound: BTreeSet<String> = BTreeSet::new();
        let mut order: Vec<&Lit> = vec![];
        let mut pending = pos.clone();
        let mut waiting = rest.clone();
        fn flush<'c>(waiting: &mut Vec<&'c Lit>, order: &mut Vec<&'c Lit>, bound: &BTreeSet<String>) {
            let mut i = 0;
            while i < waiting.len() {
                if lit_vars(waiting[i]).is_subset(bound) { order.push(waiting.remove(i)); } else { i += 1; }
            }
        }
        flush(&mut waiting, &mut order, &bound);
        while !pending.is_empty() {
            let score = |l: &Lit| -> (bool, usize, i64) {
                let Lit::Pos(_, a) = l else { unreachable!() };
                let nb = a.iter().filter(|x| term_vars(x).is_subset(&bound)).count();
                (nb == a.len(), nb, -(a.len() as i64))
            };
            let mut bi = 0;
            for i in 1..pending.len() { if score(pending[i]) > score(pending[bi]) { bi = i; } }
            let best = pending.remove(bi);
            order.push(best);
            bound.extend(lit_vars(best));
            flush(&mut waiting, &mut order, &bound);
        }
        if !waiting.is_empty() { return err(format!("unsafe clause (a variable of a negation, a distinct or the head is bound by no positive literal): {}", c.text)); }
        let mut hv = BTreeSet::new();
        for a in &c.head.1 { a.vars(&mut hv); }
        if !hv.is_subset(&bound) { return err(format!("unsafe clause (head variable unbound): {}", c.text)); }
        let mut names: Vec<String> = vec![];
        let mut steps = vec![];
        let mut bound_now: BTreeSet<String> = BTreeSet::new();
        for l in order {
            match l {
                Lit::Distinct(a, b) => { steps.push(CStep::Distinct(comp(a, &mut names), comp(b, &mut names))); }
                Lit::Neg(r, a) => { steps.push(CStep::Neg(*r, a.iter().map(|x| comp(x, &mut names)).collect())); }
                Lit::Pos(r, a) => {
                    let keypos: Vec<usize> = (0..a.len()).filter(|&i| term_vars(&a[i]).is_subset(&bound_now)).collect();
                    steps.push(CStep::Pos(*r, a.iter().map(|x| comp(x, &mut names)).collect(), keypos));
                    bound_now.extend(lit_vars(l));
                }
            }
        }
        let head: Vec<P> = c.head.1.iter().map(|x| comp(x, &mut names)).collect();
        let rec_steps = steps.iter().enumerate().filter(|(_, s)| matches!(s, CStep::Pos(r, _, _) if scc.contains(r))).map(|(i, _)| i).collect();
        Ok(CRule { clause: ci, nvars: names.len(), head_rel: c.head.0, head, steps, rec_steps })
    }
}

// ------------------------------------------------------------------------------------------------------------------------------------------ the store of facts

/// A relation's facts in insertion order, with hash indexes on the bound argument positions (built when first asked, dropped when a fact is added).
#[derive(Clone, Default)]
pub struct Relation {
    pub facts: HashSet<Row>,
    pub order: Vec<Row>,
    index: HashMap<Vec<usize>, HashMap<Vec<T>, Vec<Row>>>,
}

impl Relation {
    pub fn from_rows(rows: &[Row]) -> Relation { let mut r = Relation::default(); for t in rows { r.add(*t); } r }
    pub fn add(&mut self, t: Row) -> bool {
        if !self.facts.insert(t) { return false; }
        self.order.push(t);
        self.index.clear();
        true
    }
    pub fn lookup_rows(&mut self, keypos: &[usize], key: &[T]) -> Vec<Row> {
        if keypos.is_empty() { return self.order.clone(); }
        if !self.index.contains_key(keypos) {
            let mut ix: HashMap<Vec<T>, Vec<Row>> = HashMap::new();
            for t in &self.order { ix.entry(keypos.iter().map(|&i| t[i]).collect()).or_default().push(*t); }
            self.index.insert(keypos.to_vec(), ix);
        }
        self.index[keypos].get(key).cloned().unwrap_or_default()
    }
    /// The rows matching `key` at `keypos`, by a scan (no index: the semi-naive round's deltas).
    pub fn scan(&self, keypos: &[usize], key: &[T]) -> Vec<Row> {
        self.order.iter().filter(|t| keypos.iter().zip(key).all(|(&i, &k)| t[i] == k)).copied().collect()
    }
}

/// The facts of one evaluation: relation id -> its facts (None: never written).
pub struct Db { pub rels: Vec<Option<Relation>> }

impl Db {
    pub fn new(n: usize) -> Db { Db { rels: (0..n).map(|_| None).collect() } }
    pub fn writable(&mut self, r: Rel) -> &mut Relation { self.rels[r as usize].get_or_insert_with(Relation::default) }
    pub fn facts(&self, r: Rel) -> Vec<Row> { self.rels[r as usize].as_ref().map(|x| x.order.clone()).unwrap_or_default() }
}
