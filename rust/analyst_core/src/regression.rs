//! PLAN 76.4b (b) THE REGRESSION IN RUST: the mirror of discovery/regression.py (plan 76.4a) -- a goal's DEFECTS and FIXES derived from the rule spec. GENERIC:
//! this module names no game object. It reads a loaded description (`gdl::Program`), a state given through a BINDING (the game's implementation of the spec's
//! state, its certified relations and its moves: the `Binding` trait; chess's is gdl_chess.rs) and a GOAL (a relation head of the spec with arguments), and
//! answers, as regression.py answers (its Python reading is the TEST ORACLE: equal on every question of the guard positions, scripts/plan76_4a.py compare):
//!   (1) THE DERIVATION -- which clause of the head applies and which body literals are TRUE / FALSE in the state, evaluated top-down and tabled (a recursive
//!       relation -- a ray -- is a table iterated to its fixpoint; negation is negation as failure over a lower stratum);
//!   (2) THE DEFECTS -- the FALSE positive literals (with the deepest atom whose failure is a single point of the clause) and the TRUE negated literals with
//!       their WITNESSES (an action of a role, applied by the certified next state, with the LEAVES of the goal's positive literals it flips beyond the pass);
//!       or the goal's VALUE short of its need;
//!   (3) THE NAMES -- the store's DEFECT_FIX bindings (passed in from Python, as data) applied to the defects: which entry names which defect, its location and
//!       the locations involved (the TEXT stays Python's: RUST_BOUNDARY risk 3 -- the record carries the template's fields);
//!   (4) THE FIXES -- for a named defect, the actions after which (in the goal's future of the next state) it is gone and nothing named is new.
//! THE READ SET of a regression (`Regression::reads`: the literals, their deepest false atoms, the leaves, the witnesses' bodies, the valuation's reads) is
//! returned WITH the result and replayed from the cache on a hit (RUST_BOUNDARY risk 2: no dependency tracker of lib.rs is written -- regression.py writes none);
//! every set that reaches an output is ordered (BTreeSet) or sorted (risk 4); nothing here runs on another thread (risk 8).
//! SOUNDNESS is regression.py's: the certified relations are read through the binding (their RULE entries PROVED equal to the kernel: Ideas.LegalEquiv.legal_iff,
//! attackBy_iff; the next state Ideas.NextEquiv.applyMoveSpec_eq_applyMove); every other atom is the spec's own clause evaluated here.
use crate::gdl::{build, comp, mtch, pvars, Interner, Lit, Program, Rel, Row, T, ARITY_MAX, FREE, P};
use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::{BTreeMap, BTreeSet};

pub const DISTINCT: Rel = Rel::MAX;            // the pseudo-relation of a `distinct` atom (regression.py's ("distinct", (t1, t2)))

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kind { Certified, Static, Input, Derived }

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash, PartialOrd, Ord)]
pub enum LK { Pos, Neg, Distinct }

/// One literal of a compiled clause: (kind, relation (DISTINCT for distinct), args, its index in the written body).
#[derive(Clone, Debug)]
#[allow(dead_code)]
pub struct Step { pub kind: LK, pub rel: Rel, pub args: Vec<P>, pub j: usize }

/// A clause compiled for one call pattern (regression.py `Spec._plan`): its index, its head, the number of variables, the body in its join order, the body as written.
#[derive(Clone, Debug)]
pub struct Plan { pub idx: usize, pub head: Vec<P>, pub nvars: usize, pub steps: Vec<Step>, pub written: Vec<Step> }

/// THE SPEC read top-down (regression.py `Spec`): the relations' classes and the plans of every call pattern (computed once, at load).
pub struct Spec {
    pub prog: Program,
    pub kinds: Vec<Kind>,
    pub recursive: Vec<bool>,
    pub has_rules: Vec<bool>,
    plans: FxHashMap<(Rel, u16), Vec<Plan>>,
    statics: FxHashMap<(Rel, u16), FxHashMap<Vec<T>, Vec<Row>>>,
    pub arity: Vec<usize>,
    pub legal: Option<Rel>,                    // GDL's keyword `legal` (a role's moves): a witness reading a certified `legal` row is an action
}

fn mask_of(pat: &Row, n: usize) -> u16 { let mut m = 0u16; for i in 0..n { if pat[i] != FREE { m |= 1 << i; } } m }

impl Spec {
    /// The spec over a loaded program; `certified`: the relations read through the binding (their RULE entries PROVED).
    pub fn new(prog: Program, certified: &[Rel], terms: &mut Interner) -> Result<Spec, String> {
        let n = prog.nrels();
        let mut kinds = vec![Kind::Derived; n];
        for r in 0..n {
            let name = prog.rel_name(r as Rel).to_string();
            kinds[r] = if certified.contains(&(r as Rel)) { Kind::Certified }
                       else if !prog.reads_input[r] && prog.statics.contains_key(&(r as Rel)) { Kind::Static }
                       else if crate::gdl::is_input(&name) { Kind::Input } else { Kind::Derived };
        }
        let mut recursive = vec![false; n];
        for (comp, rules) in &prog.strata {
            if rules.iter().any(|r| !r.rec_steps.is_empty()) { for &h in comp { recursive[h as usize] = true; } }
        }
        let arity: Vec<usize> = prog.arity.iter().map(|a| a.unwrap_or(0)).collect();
        let has_rules: Vec<bool> = (0..n).map(|r| !prog.rules_of[r].is_empty()).collect();
        let legal = prog.rel_id("legal");
        let mut sp = Spec { prog, kinds, recursive, has_rules, plans: FxHashMap::default(), statics: FxHashMap::default(), arity, legal };
        // every call pattern's plans, once
        for r in 0..n {
            if sp.prog.rules_of[r].is_empty() { continue; }
            let a = sp.arity[r];
            for m in 0..(1u16 << a) {
                let mut ps = vec![];
                for (i, &ci) in sp.prog.rules_of[r].iter().enumerate() { ps.push(sp.plan(i, ci, m)?); }
                sp.plans.insert((r as Rel, m), ps);
            }
        }
        // every static relation's index per bound pattern, once
        let statics = std::mem::take(&mut sp.prog.statics);
        for (&r, rel) in &statics {
            let a = sp.arity[r as usize];
            for m in 0..(1u16 << a) {
                let keypos: Vec<usize> = (0..a).filter(|i| m & (1 << i) != 0).collect();
                let mut ix: FxHashMap<Vec<T>, Vec<Row>> = FxHashMap::default();
                for t in &rel.order { ix.entry(keypos.iter().map(|&i| t[i]).collect()).or_default().push(*t); }
                sp.statics.insert((r, m), ix);
            }
        }
        sp.prog.statics = statics;
        let _ = terms;
        Ok(sp)
    }

    pub fn kind(&self, r: Rel) -> Kind { self.kinds[r as usize] }
    fn base(&self, r: Rel) -> bool { self.kind(r) != Kind::Derived }

    pub fn plans(&self, r: Rel, m: u16) -> &[Plan] { self.plans.get(&(r, m)).map(|v| v.as_slice()).unwrap_or(&[]) }

    fn plan(&self, idx: usize, ci: usize, bound: u16) -> Result<Plan, String> {
        let clause = &self.prog.clauses[ci];
        let mut names: Vec<String> = vec![];
        let head: Vec<P> = clause.head.1.iter().map(|a| comp(a, &mut names)).collect();
        let mut have: BTreeSet<u16> = BTreeSet::new();
        for (i, h) in head.iter().enumerate() { if bound & (1 << i) != 0 { pvars(h, &mut have); } }
        let mut lits: Vec<Step> = vec![];
        for (j, l) in clause.body.iter().enumerate() {
            match l {
                Lit::Distinct(a, b) => { let pa = comp(a, &mut names); let pb = comp(b, &mut names); lits.push(Step { kind: LK::Distinct, rel: DISTINCT, args: vec![pa, pb], j }); }
                Lit::Pos(r, a) => { let args = a.iter().map(|x| comp(x, &mut names)).collect(); lits.push(Step { kind: LK::Pos, rel: *r, args, j }); }
                Lit::Neg(r, a) => { let args = a.iter().map(|x| comp(x, &mut names)).collect(); lits.push(Step { kind: LK::Neg, rel: *r, args, j }); }
            }
        }
        let written = lits.clone();
        let lv = |l: &Step| -> BTreeSet<u16> { let mut s = BTreeSet::new(); for a in &l.args { pvars(a, &mut s); } s };
        let mut order: Vec<Step> = vec![];
        let mut pending: Vec<Step> = lits.iter().filter(|l| l.kind == LK::Pos).cloned().collect();
        let mut waiting: Vec<Step> = lits.iter().filter(|l| l.kind != LK::Pos).cloned().collect();
        let flush = |waiting: &mut Vec<Step>, order: &mut Vec<Step>, have: &BTreeSet<u16>| {
            let mut i = 0;
            while i < waiting.len() { if lv(&waiting[i]).is_subset(have) { order.push(waiting.remove(i)); } else { i += 1; } }
        };
        flush(&mut waiting, &mut order, &have);
        while !pending.is_empty() {
            // the most bound arguments first (a ray read back from its bound end), a base relation on a tie (regression.py's score; max keeps the first)
            let score = |l: &Step| -> (bool, usize, bool, i64) {
                let nb = l.args.iter().filter(|a| { let mut s = BTreeSet::new(); pvars(a, &mut s); s.is_subset(&have) }).count();
                (nb == l.args.len(), nb, self.base(l.rel), -(l.args.len() as i64))
            };
            let mut bi = 0;
            for i in 1..pending.len() { if score(&pending[i]) > score(&pending[bi]) { bi = i; } }
            let best = pending.remove(bi);
            have.extend(lv(&best));
            order.push(best);
            flush(&mut waiting, &mut order, &have);
        }
        if !waiting.is_empty() { return Err(format!("unsafe clause for the top-down reading: {}", clause.text)); }
        Ok(Plan { idx, head, nvars: names.len(), steps: order, written })
    }

    pub fn static_rows(&self, r: Rel, pat: &Row) -> &[Row] {
        let a = self.arity[r as usize];
        let m = mask_of(pat, a);
        let key: Vec<T> = (0..a).filter(|i| m & (1 << i) != 0).map(|i| pat[i]).collect();
        self.statics.get(&(r, m)).and_then(|ix| ix.get(&key)).map(|v| v.as_slice()).unwrap_or(&[])
    }
}

// ------------------------------------------------------------------------------------------------------------------------------------------ the binding

/// What the regression needs of a game (regression.py's binding contract): the state's input rows and the certified relations' rows by pattern, an action of
/// a certified `legal` row, the certified next state, the turn passed, a state given to a role, the valuation laws, and the reading of terms as locations.
pub trait Binding {
    type X: Clone;
    type A: Clone + PartialEq + std::fmt::Debug;
    fn input(&self, x: &Self::X, rel: Rel, pat: &Row) -> Vec<Row>;
    fn certified(&self, x: &Self::X, rel: Rel, pat: &Row) -> Vec<Row>;
    fn action(&self, row: &Row) -> Option<Self::A>;
    fn apply(&self, x: &Self::X, a: &Self::A) -> Self::X;
    fn pass(&self, x: &Self::X) -> Self::X;
    fn to_move(&self, x: &Self::X, role: T) -> Self::X;
    fn value(&self, x: &Self::X, law: &str, args: &[T]) -> Option<i64>;
    /// The location of the first ground (location) slot pair of an atom (`slots`: its relation's slot pairs from the signature).
    fn loc(&self, slots: &[usize], args: &Row) -> Option<u8>;
    /// An action's two locations (from, to).
    fn ends(&self, a: &Self::A) -> (Option<u8>, Option<u8>);
    /// The unit standing at a location, as its letter ('?' for none).
    fn unit_letter(&self, x: &Self::X, loc: Option<u8>) -> char;
}

// ------------------------------------------------------------------------------------------------------------------------------------------ the prover

struct Table { rows: Vec<Row>, seen: FxHashSet<Row>, done: bool, active: bool, depth: usize }

/// THE INTERPRETER READ TOP-DOWN on one state (regression.py `Prover`): `ans(rel, pattern)` -> the ground rows of `rel` matching `pattern` (FREE where free).
pub struct Prover<'a, B: Binding> {
    pub sp: &'a Spec,
    pub b: &'a B,
    pub x: &'a B::X,
    pub terms: &'a mut Interner,
    tables: FxHashMap<(Rel, Row), usize>,
    tabs: Vec<Table>,
    stack: Vec<usize>,
    low: usize,
    added: u64,
    pending: Vec<usize>,
    inp: FxHashMap<(Rel, Row), Vec<Row>>,
    cert: FxHashMap<(Rel, Row), Vec<Row>>,
}

impl<'a, B: Binding> Prover<'a, B> {
    pub fn new(sp: &'a Spec, b: &'a B, x: &'a B::X, terms: &'a mut Interner) -> Self {
        Prover { sp, b, x, terms, tables: FxHashMap::default(), tabs: vec![], stack: vec![], low: 1 << 30, added: 0, pending: vec![],
                 inp: FxHashMap::default(), cert: FxHashMap::default() }
    }

    pub fn ans(&mut self, rel: Rel, pat: &Row) -> Vec<Row> {
        match self.sp.kind(rel) {
            Kind::Static => self.sp.static_rows(rel, pat).to_vec(),
            Kind::Input => {
                if let Some(v) = self.inp.get(&(rel, *pat)) { return v.clone(); }
                let v = self.b.input(self.x, rel, pat);
                self.inp.insert((rel, *pat), v.clone());
                v
            }
            Kind::Certified => {
                if let Some(v) = self.cert.get(&(rel, *pat)) { return v.clone(); }
                let v = self.b.certified(self.x, rel, pat);
                self.cert.insert((rel, *pat), v.clone());
                v
            }
            Kind::Derived => {
                let key = (rel, *pat);
                let ti = match self.tables.get(&key) {
                    Some(&ti) => {
                        if self.tabs[ti].done { return self.tabs[ti].rows.clone(); }
                        if self.tabs[ti].active { self.low = self.low.min(self.tabs[ti].depth); return self.tabs[ti].rows.clone(); }
                        ti
                    }
                    None => {
                        let ti = self.tabs.len();
                        self.tabs.push(Table { rows: vec![], seen: FxHashSet::default(), done: false, active: false, depth: 0 });
                        self.tables.insert(key, ti);
                        if !self.sp.recursive[rel as usize] {                // a relation of no recursive component: one pass is its whole answer
                            let m = mask_of(pat, self.sp.arity[rel as usize]);
                            let np = self.sp.plans(rel, m).len();
                            for pi in 0..np {
                                let rows = self.fire(rel, m, pi, pat);
                                let t = &mut self.tabs[ti];
                                for row in rows { if t.seen.insert(row) { t.rows.push(row); } }
                            }
                            self.tabs[ti].done = true;
                            return self.tabs[ti].rows.clone();
                        }
                        ti
                    }
                };
                self.eval(rel, pat, ti)
            }
        }
    }

    pub fn holds(&mut self, rel: Rel, args: &Row) -> bool { !self.ans(rel, args).is_empty() }

    fn eval(&mut self, rel: Rel, pat: &Row, ti: usize) -> Vec<Row> {
        let depth = self.stack.len();
        self.stack.push(ti);
        self.tabs[ti].active = true;
        self.tabs[ti].depth = depth;
        let outer = self.low;
        let mark = self.pending.len();
        let m = mask_of(pat, self.sp.arity[rel as usize]);
        let np = self.sp.plans(rel, m).len();
        let mut mine;
        loop {
            self.low = depth + 1;
            let n0 = self.added;
            for pi in 0..np {
                let rows = self.fire(rel, m, pi, pat);
                for row in rows {
                    let t = &mut self.tabs[ti];
                    if t.seen.insert(row) { t.rows.push(row); self.added += 1; }
                }
            }
            mine = self.low;
            if mine > depth || self.added == n0 { break; }
        }
        self.stack.pop();
        self.tabs[ti].active = false;
        if mine >= depth {
            self.tabs[ti].done = true;
            for p in self.pending.drain(mark..).collect::<Vec<_>>() { self.tabs[p].done = true; }
        } else {
            self.pending.push(ti);
        }
        self.low = outer.min(mine);
        self.tabs[ti].rows.clone()
    }

    fn fire(&mut self, rel: Rel, m: u16, pi: usize, pat: &Row) -> Vec<Row> {
        let sp = self.sp;
        let plan = &sp.plans(rel, m)[pi];
        let mut env = vec![FREE; plan.nvars];
        let mut trail = vec![];
        for i in 0..plan.head.len() {
            if pat[i] != FREE && !mtch(self.terms, &plan.head[i], pat[i], &mut env, &mut trail) { return vec![]; }
        }
        let mut out = vec![];
        self.go(plan, 0, &mut env, &mut out);
        out
    }

    fn go(&mut self, plan: &Plan, k: usize, env: &mut Vec<T>, out: &mut Vec<Row>) {
        if k == plan.steps.len() {
            let mut row = [FREE; ARITY_MAX];
            for (i, h) in plan.head.iter().enumerate() { row[i] = build(self.terms, h, env); }
            out.push(row);
            return;
        }
        let s = &plan.steps[k];
        match s.kind {
            LK::Pos => {
                let callpat = self.callpat(&s.args, env);
                for row in self.ans(s.rel, &callpat) {
                    let mut tr = vec![];
                    let mut ok = true;
                    for (i, a) in s.args.iter().enumerate() { if !mtch(self.terms, a, row[i], env, &mut tr) { ok = false; break; } }
                    if ok { self.go(plan, k + 1, env, out); }
                    for v in tr { env[v as usize] = FREE; }
                }
            }
            LK::Neg => {
                let callpat = self.callpat(&s.args, env);
                if self.ans(s.rel, &callpat).is_empty() { self.go(plan, k + 1, env, out); }
            }
            LK::Distinct => {
                if build(self.terms, &s.args[0], env) != build(self.terms, &s.args[1], env) { self.go(plan, k + 1, env, out); }
            }
        }
    }

    pub fn callpat(&mut self, args: &[P], env: &[T]) -> Row {
        let mut row = [FREE; ARITY_MAX];
        for (i, a) in args.iter().enumerate() { row[i] = build(self.terms, a, env); }
        row
    }

    // ----------------------------------------------------------------------------------- explaining an atom

    /// The clause instances that derive the ground atom (rel, args) here: [(clause index, [(kind, rel, ground args) in the written order])].
    pub fn derivations(&mut self, rel: Rel, args: &Row) -> Vec<(usize, Vec<(LK, Rel, Row)>)> {
        let sp = self.sp;
        let a = sp.arity[rel as usize];
        let m: u16 = if a == 0 { 0 } else { (1u16 << a) - 1 };
        let mut out = vec![];
        for plan in sp.plans(rel, m) {
            let mut env = vec![FREE; plan.nvars];
            let mut trail = vec![];
            if !(0..plan.head.len()).all(|i| mtch(self.terms, &plan.head[i], args[i], &mut env, &mut trail)) { continue; }
            let mut found: Vec<Vec<(LK, Rel, Row)>> = vec![];
            self.dgo(plan, 0, &mut env, &mut found);
            for lits in found { out.push((plan.idx, lits)); }
        }
        out
    }

    fn dgo(&mut self, plan: &Plan, k: usize, env: &mut Vec<T>, out: &mut Vec<Vec<(LK, Rel, Row)>>) {
        if k == plan.steps.len() {
            let mut lits = vec![];
            for s in &plan.written { let r = self.callpat(&s.args, env); lits.push((s.kind, s.rel, r)); }
            out.push(lits);
            return;
        }
        let s = &plan.steps[k];
        match s.kind {
            LK::Pos => {
                let callpat = self.callpat(&s.args, env);
                for row in self.ans(s.rel, &callpat) {
                    let mut tr = vec![];
                    if s.args.iter().enumerate().all(|(i, a)| mtch(self.terms, a, row[i], env, &mut tr)) { self.dgo(plan, k + 1, env, out); }
                    for v in tr { env[v as usize] = FREE; }
                }
            }
            LK::Neg => {
                let callpat = self.callpat(&s.args, env);
                if self.ans(s.rel, &callpat).is_empty() { self.dgo(plan, k + 1, env, out); }
            }
            LK::Distinct => {
                if build(self.terms, &s.args[0], env) != build(self.terms, &s.args[1], env) { self.dgo(plan, k + 1, env, out); }
            }
        }
    }

    /// THE LEAVES of every derivation of the ground TRUE atom (rel, args): the fluents of the state and the negated atoms its derivations rest on; static facts
    /// are no leaves; a certified atom with no clauses is a leaf of its own.
    pub fn leaves(&mut self, rel: Rel, args: &Row, seen: &BTreeSet<(Rel, Row)>) -> BTreeSet<Leaf> {
        let mut seen2 = seen.clone();
        seen2.insert((rel, *args));
        let mut out = BTreeSet::new();
        for (idx, lits) in self.derivations(rel, args) {
            let mut sub = BTreeSet::new();
            let mut ok = true;
            for (j, (kind, r2, av)) in lits.iter().enumerate() {
                if *kind == LK::Distinct { continue; }
                let k = self.sp.kind(*r2);
                let at = (rel, idx as u16, j as u16);
                if *kind == LK::Neg { sub.insert(Leaf { at, atom: Atom::new(*r2, *av, self.sp.arity[*r2 as usize]), neg: true }); }
                else if k == Kind::Static { continue; }
                else if k == Kind::Input || (k == Kind::Certified && !self.sp.has_rules[*r2 as usize]) {
                    sub.insert(Leaf { at, atom: Atom::new(*r2, *av, self.sp.arity[*r2 as usize]), neg: false });
                } else if seen2.contains(&(*r2, *av)) { ok = false; break; }           // a derivation through the atom itself is not minimal
                else { sub.extend(self.leaves(*r2, av, &seen2)); }
            }
            if ok { out.extend(sub); }
        }
        out
    }

    /// THE DEEPEST FALSE ATOM of the false atom (rel, args) (regression.py `why_not`).
    pub fn why_not(&mut self, rel: Rel, args: &Row) -> (Rel, Row) {
        let mut cur = (rel, *args);
        for _ in 0..64 {
            let (r, a) = cur;
            if self.sp.kind(r) != Kind::Derived { return cur; }
            let m = mask_of(&a, self.sp.arity[r as usize]);
            let sp = self.sp;
            let plans = sp.plans(r, m);
            if plans.len() != 1 { return cur; }
            let plan = &plans[0];
            let mut env = vec![FREE; plan.nvars];
            let mut trail = vec![];
            if !(0..plan.head.len()).filter(|&i| a[i] != FREE).all(|i| mtch(self.terms, &plan.head[i], a[i], &mut env, &mut trail)) { return cur; }
            let mut nxt = None;
            for s in &plan.written {
                if s.kind == LK::Distinct {
                    let x = build(self.terms, &s.args[0], &env);
                    let y = build(self.terms, &s.args[1], &env);
                    if x != FREE && y != FREE && x == y { return cur; }
                    continue;
                }
                let callpat = self.callpat(&s.args, &env);
                if s.kind == LK::Neg {
                    if (0..s.args.len()).any(|i| callpat[i] == FREE) || !self.ans(s.rel, &callpat).is_empty() { return cur; }
                    continue;
                }
                let rows = self.ans(s.rel, &callpat);
                if rows.is_empty() { nxt = Some((s.rel, callpat)); break; }
                if rows.len() > 1 { return cur; }
                let mut tr = vec![];
                if !s.args.iter().enumerate().all(|(i, x)| mtch(self.terms, x, rows[0][i], &mut env, &mut tr)) { return cur; }
            }
            match nxt { None => return cur, Some(n) => cur = n }
        }
        cur
    }
}

// ------------------------------------------------------------------------------------------------------------------------------------------ results

/// An atom: a relation and its arguments (FREE where a pattern leaves one free); `n` its arity.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Atom { pub rel: Rel, pub args: Row, pub n: u8 }
impl Atom { pub fn new(rel: Rel, args: Row, n: usize) -> Atom { Atom { rel, args, n: n as u8 } } }

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Leaf { pub at: (Rel, u16, u16), pub atom: Atom, pub neg: bool }

#[derive(Clone, Debug)]
pub struct Witness<A> { pub atom: Atom, pub body: Vec<(LK, Rel, Row)>, pub action: Option<A>, pub flips: BTreeSet<Leaf> }

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DK { False, Witness, Value }

#[derive(Clone, Debug)]
pub struct Derived<A> {
    pub kind: DK, pub clause: i32, pub lit: i32, pub atom: Atom, pub deep: Option<Atom>, pub witness: Option<Witness<A>>,
    pub value: Option<(i64, i64, Vec<Atom>)>,
}

#[derive(Clone, Debug)]
pub struct Regression<A> {
    pub holds: bool,
    pub clauses: Vec<(usize, Vec<(usize, LK, Option<Atom>, bool)>)>,
    pub defects: Vec<Derived<A>>,
    pub reads: BTreeSet<Atom>,
    pub rows: Vec<Row>,
    pub value: Option<i64>,
    pub leaves: BTreeSet<Leaf>,
}

/// A goal (regression.py `Goal`, its parameters substituted): the head (relation, args: FREE for a variable), the role on move (FREE: as the state is), the
/// value (law, args, need, the pattern the valuation reads).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Goal { pub name: String, pub head: (Rel, Row), pub on_move: T, pub value: Option<(String, Vec<T>, i64, (Rel, Row))> }

// ------------------------------------------------------------------------------------------------------------------------------------------ the names

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ND { False, Witness, Value, Holds, Leaf }

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Field { Leaf, To, From }

/// A place of the spec a name binds: (relation, clause index or any, literal index).
pub type Place = (Rel, Option<u16>, u16);

/// A DEFECT_FIX entry's `gdl` binding (the store's, passed in as data -- regression.py `name` reads the same fields).
#[derive(Clone, Debug)]
pub struct NameEntry {
    pub name: String, pub goal: String, pub theirs: bool, pub defect: ND, pub order: i64,
    pub leaf: Option<Place>, pub unless: Option<Place>, pub sq: Field, pub by: Field, pub group_sq: bool, pub sort_sq: bool,
    pub clause: Option<(i32, i32)>, pub text_any: bool, pub text_rels: Vec<String>,
}

/// A named defect: the entry (its index in the list passed in), its location, the locations involved, and the template's fields (the text is Python's).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Named {
    pub entry: usize, pub sq: Option<u8>, pub by: Vec<Option<u8>>, pub frm: Option<u8>, pub to: Option<u8>, pub unit: char, pub value: Option<i64>,
    pub rel: String, pub src: Vec<usize>,
}

fn at_matches(at: &(Rel, u16, u16), p: &Place) -> bool { at.0 == p.0 && p.1.is_none_or(|c| c == at.1) && at.2 == p.2 }

fn okey(s: Option<u8>) -> i64 { match s { None => -1, Some(x) => x as i64 } }

/// Python's repr of an atom (the tie-break `sorted(hit, key=repr(l.atom))` reads it): ('rel', ('a', 'b')).
pub fn atom_repr(prog: &Program, it: &Interner, a: &Atom) -> String {
    fn term(it: &Interner, t: T) -> String {
        if t == FREE { return "None".into(); }
        match it.compound(t) {
            None => format!("'{}'", it.name(t)),
            Some((f, args)) => {
                let mut parts = vec![format!("'{}'", it.name(f))];
                for x in args { parts.push(term(it, *x)); }
                if parts.len() == 1 { format!("({},)", parts[0]) } else { format!("({})", parts.join(", ")) }
            }
        }
    }
    let name = if a.rel == DISTINCT { "distinct".to_string() } else { prog.rel_name(a.rel).to_string() };
    let args: Vec<String> = (0..a.n as usize).map(|i| term(it, a.args[i])).collect();
    let tup = match args.len() { 0 => "()".to_string(), 1 => format!("({},)", args[0]), _ => format!("({})", args.join(", ")) };
    format!("('{name}', {tup})")
}

// ------------------------------------------------------------------------------------------------------------------------------------------ the regressor

/// THE REGRESSION over one spec and one binding (regression.py `Regressor`, its reading): `regress(x, goal)` read afresh; the cache (by the state's key and the
/// goal) is the caller's (gdl_chess.rs: the result is a function of the state's reading and the goal alone).
pub struct Regressor<'a, B: Binding> {
    pub sp: &'a Spec,
    pub b: &'a B,
    pub terms: &'a mut Interner,
    pub slots: &'a [Vec<usize>],
    leaves_acc: BTreeSet<Leaf>,
}

impl<'a, B: Binding> Regressor<'a, B> {
    pub fn new(sp: &'a Spec, b: &'a B, terms: &'a mut Interner, slots: &'a [Vec<usize>]) -> Self { Regressor { sp, b, terms, slots, leaves_acc: BTreeSet::new() } }

    /// The regression of `goal` on the state `x` (the goal's role put on move first, as regression.py's `regress`).
    pub fn regress(&mut self, x0: &B::X, goal: &Goal) -> (B::X, Regression<B::A>) {
        let x = if goal.on_move != FREE { self.b.to_move(x0, goal.on_move) } else { x0.clone() };
        let reg = {
            let sp = self.sp;
            let b = self.b;
            let mut pr = Prover::new(sp, b, &x, self.terms);
            regress_on(&mut pr, &x, goal, self.slots, &mut self.leaves_acc)
        };
        (x, reg)
    }
}

fn regress_on<B: Binding>(pr: &mut Prover<'_, B>, x: &B::X, goal: &Goal, slots: &[Vec<usize>], leaves_acc: &mut BTreeSet<Leaf>) -> Regression<B::A> {
    let (rel, hpat) = goal.head;
    let sp = pr.sp;
    let ar = sp.arity[rel as usize];
    let mut reads: BTreeSet<Atom> = BTreeSet::new();
    let rows = pr.ans(rel, &hpat);
    let m = mask_of(&hpat, ar);
    if !rows.is_empty() {
        for r in &rows { reads.insert(Atom::new(rel, *r, ar)); }
        match &goal.value {
            None => {
                *leaves_acc = BTreeSet::new();          // (a holding rule head: its clause is read for the leaves its true literals rest on -- what a keeper must flip)
                if sp.kind(rel) == Kind::Derived {
                    for pi in 0..sp.plans(rel, m).len() {
                        let plan = &sp.plans(rel, m)[pi];
                        let mut env = vec![FREE; plan.nvars];
                        let mut tr = vec![];
                        if (0..plan.head.len()).filter(|&i| hpat[i] != FREE).all(|i| mtch(pr.terms, &plan.head[i], hpat[i], &mut env, &mut tr)) {
                            let _ = read_clause(pr, x, plan, &mut env, &mut reads, slots, leaves_acc);
                        }
                    }
                }
                Regression { holds: true, clauses: vec![], defects: vec![], reads, rows, value: None, leaves: leaves_acc.clone() }
            }
            Some((law, vargs, need, (vrel, vpat))) => {
                let v = pr.b.value(x, law, vargs);
                let rr: BTreeSet<Atom> = pr.ans(*vrel, vpat).into_iter().map(|r| Atom::new(*vrel, r, sp.arity[*vrel as usize])).collect();
                reads.extend(rr.iter().copied());
                match v {
                    None => Regression { holds: true, clauses: vec![], defects: vec![], reads, rows, value: None, leaves: BTreeSet::new() },
                    Some(v) if v >= *need => Regression { holds: true, clauses: vec![], defects: vec![], reads, rows, value: Some(v), leaves: BTreeSet::new() },
                    Some(v) => {
                        let d = Derived { kind: DK::Value, clause: -1, lit: -1, atom: Atom::new(rel, hpat, ar), deep: None, witness: None,
                                          value: Some((v, *need, rr.iter().copied().collect())) };
                        Regression { holds: false, clauses: vec![], defects: vec![d], reads, rows, value: Some(v), leaves: BTreeSet::new() }
                    }
                }
            }
        }
    } else {
        if sp.kind(rel) != Kind::Derived || !sp.has_rules[rel as usize] {
            let (dr, da) = pr.why_not(rel, &hpat);
            let deep = Atom::new(dr, da, sp.arity[dr as usize]);
            reads.insert(deep);
            let d = Derived { kind: DK::False, clause: -1, lit: -1, atom: Atom::new(rel, hpat, ar), deep: Some(deep), witness: None, value: None };
            return Regression { holds: false, clauses: vec![], defects: vec![d], reads, rows: vec![], value: None, leaves: BTreeSet::new() };
        }
        *leaves_acc = BTreeSet::new();
        let mut clauses = vec![];
        let mut defects = vec![];
        for pi in 0..sp.plans(rel, m).len() {
            let plan = &sp.plans(rel, m)[pi];
            let mut env = vec![FREE; plan.nvars];
            let mut tr = vec![];
            if !(0..plan.head.len()).filter(|&i| hpat[i] != FREE).all(|i| mtch(pr.terms, &plan.head[i], hpat[i], &mut env, &mut tr)) { continue; }
            let (lits, ds) = read_clause(pr, x, plan, &mut env, &mut reads, slots, leaves_acc);
            clauses.push((plan.idx, lits));
            defects.extend(ds);                    // (a disjunctive head: each clause's defects are an alternative)
        }
        Regression { holds: false, clauses, defects, reads, rows: vec![], value: None, leaves: leaves_acc.clone() }
    }
}

#[allow(clippy::type_complexity)]
fn read_clause<B: Binding>(pr: &mut Prover<'_, B>, x: &B::X, plan: &Plan, env: &mut Vec<T>, reads: &mut BTreeSet<Atom>, slots: &[Vec<usize>],
                           leaves_acc: &mut BTreeSet<Leaf>) -> (Vec<(usize, LK, Option<Atom>, bool)>, Vec<Derived<B::A>>) {
    let sp = pr.sp;
    let ar = |r: Rel| if r == DISTINCT { 2 } else { sp.arity[r as usize] };
    let mut lits = vec![];
    let mut defects = vec![];
    let mut positives: Vec<(Rel, Row)> = vec![];
    let mut negs: Vec<(usize, Rel, Row)> = vec![];
    for (j, s) in plan.written.iter().enumerate() {
        if s.kind == LK::Distinct {
            let a = build(pr.terms, &s.args[0], env);
            let b = build(pr.terms, &s.args[1], env);
            let ok = a != b;
            lits.push((j, LK::Distinct, None, ok));
            if !ok {
                let mut row = [FREE; ARITY_MAX];
                row[0] = a;
                row[1] = b;
                defects.push(Derived { kind: DK::False, clause: plan.idx as i32, lit: j as i32, atom: Atom::new(DISTINCT, row, 2), deep: None, witness: None, value: None });
                return (lits, defects);
            }
            continue;
        }
        let callpat = pr.callpat(&s.args, env);
        if s.kind == LK::Pos {
            let rows = pr.ans(s.rel, &callpat);
            if rows.is_empty() {
                let (dr, da) = pr.why_not(s.rel, &callpat);
                let deep = Atom::new(dr, da, ar(dr));
                reads.insert(Atom::new(s.rel, callpat, ar(s.rel)));
                reads.insert(deep);
                lits.push((j, LK::Pos, Some(Atom::new(s.rel, callpat, ar(s.rel))), false));
                defects.push(Derived { kind: DK::False, clause: plan.idx as i32, lit: j as i32, atom: Atom::new(s.rel, callpat, ar(s.rel)), deep: Some(deep), witness: None, value: None });
                return (lits, defects);
            }
            let row = rows[0];
            let mut tr = vec![];
            for (i, a) in s.args.iter().enumerate() { let _ = mtch(pr.terms, a, row[i], env, &mut tr); }
            lits.push((j, LK::Pos, Some(Atom::new(s.rel, row, ar(s.rel))), true));
            reads.insert(Atom::new(s.rel, row, ar(s.rel)));
            positives.push((s.rel, row));
        } else {
            negs.push((j, s.rel, callpat));
            let t = pr.ans(s.rel, &callpat).is_empty();
            lits.push((j, LK::Neg, Some(Atom::new(s.rel, callpat, ar(s.rel))), t));
        }
    }
    let mut leaves: BTreeSet<Leaf> = BTreeSet::new();
    for (r, row) in &positives { leaves.extend(pr.leaves(*r, row, &BTreeSet::new())); }
    for lf in &leaves { reads.insert(lf.atom); }
    leaves_acc.extend(leaves.iter().copied());
    let mut base_flips: Option<BTreeSet<Leaf>> = None;
    for (j, r, callpat) in negs {
        if pr.ans(r, &callpat).is_empty() { continue; }
        for mut w in witnesses(pr, r, &callpat) {
            for (k, r2, a) in &w.body { if *k != LK::Distinct { reads.insert(Atom::new(*r2, *a, ar(*r2))); } }      // (a distinct reads no relation)
            if let Some(act) = w.action.clone() {
                if !leaves.is_empty() {
                    if base_flips.is_none() { let xp = pr.b.pass(x); base_flips = Some(flips(pr, &xp, &leaves)); }
                    let xa = pr.b.apply(x, &act);
                    let f = flips(pr, &xa, &leaves);
                    w.flips = f.difference(base_flips.as_ref().unwrap()).copied().collect();
                }
            }
            defects.push(Derived { kind: DK::Witness, clause: plan.idx as i32, lit: j as i32, atom: Atom::new(r, callpat, ar(r)), deep: None, witness: Some(w), value: None });
        }
    }
    let _ = slots;
    (lits, defects)
}

/// The witnesses of the true atom (rel, args): its derivations one level down; an action when the derivation reads a certified `legal` row (GDL's keyword).
fn witnesses<B: Binding>(pr: &mut Prover<'_, B>, rel: Rel, args: &Row) -> Vec<Witness<B::A>> {
    let mut out = vec![];
    let atom = Atom::new(rel, *args, pr.sp.arity[rel as usize]);
    for (_idx, lits) in pr.derivations(rel, args) {
        let mut action = None;
        for (k, r2, av) in &lits {
            if *k == LK::Pos && Some(*r2) == pr.sp.legal && pr.sp.kind(*r2) == Kind::Certified { action = pr.b.action(av); }
        }
        out.push(Witness { atom, body: lits, action, flips: BTreeSet::new() });
    }
    out
}

/// The leaves false in the state `x2` (true where they were read): a fluent leaf read off the state, a negated leaf by its atom read there.
fn flips<B: Binding>(pr: &mut Prover<'_, B>, x2: &B::X, leaves: &BTreeSet<Leaf>) -> BTreeSet<Leaf> {
    let sp = pr.sp;
    let b = pr.b;
    let mut p2 = Prover::new(sp, b, x2, &mut *pr.terms);
    let mut out = BTreeSet::new();
    for lf in leaves {
        if p2.holds(lf.atom.rel, &lf.atom.args) == !lf.neg { continue; }
        out.insert(*lf);
    }
    out
}

// ------------------------------------------------------------------------------------------------------------------------------------------ naming

/// THE NAMES of a regression's defects (regression.py `name`, its binding logic; the text is formatted in Python from the fields): every entry of `entries`
/// whose `gdl` names a defect of the goal, in their `order`.
pub fn name<B: Binding>(sp: &Spec, b: &B, it: &Interner, slots: &[Vec<usize>], reg: &Regression<B::A>, goal: &Goal, entries: &[NameEntry], x: &B::X,
                        theirs: bool) -> Vec<Named> {
    let loc = |a: &Atom| -> Option<u8> { if a.rel == DISTINCT { None } else { b.loc(&slots[a.rel as usize], &a.args) } };
    let mut mine: Vec<usize> = (0..entries.len()).filter(|&i| entries[i].goal == goal.name && entries[i].theirs == theirs).collect();
    mine.sort_by_key(|&i| entries[i].order);                 // (stable: the entries' own order on a tie, as Python's sorted)
    let mut out = vec![];
    let ar = |r: Rel| if r == DISTINCT { 2 } else { sp.arity[r as usize] };
    for &ei in &mine {
        let g = &entries[ei];
        match g.defect {
            ND::Holds => {
                if reg.holds && !reg.rows.is_empty() {
                    let sq = loc(&Atom::new(goal.head.0, goal.head.1, ar(goal.head.0)));
                    let mut by: Vec<u8> = reg.rows.iter().filter_map(|r| loc(&Atom::new(goal.head.0, *r, ar(goal.head.0)))).collect::<BTreeSet<u8>>().into_iter().collect();
                    by.sort_by_key(|s| okey(Some(*s)));
                    out.push(Named { entry: ei, sq, by: by.into_iter().map(Some).collect(), frm: None, to: None, unit: '?', value: reg.value, rel: String::new(), src: vec![] });
                }
                continue;
            }
            ND::Leaf => {
                let Some(pat) = g.leaf else { continue };
                let mut ls: Vec<&Leaf> = reg.leaves.iter().filter(|lf| at_matches(&lf.at, &pat)).collect();
                ls.sort_by_key(|lf| okey(loc(&lf.atom)));
                for lf in ls {
                    let sq = loc(&lf.atom);
                    out.push(Named { entry: ei, sq, by: vec![], frm: None, to: None, unit: b.unit_letter(x, sq), value: None, rel: String::new(), src: vec![] });
                }
                continue;
            }
            _ => {}
        }
        // grouped per witness or per location
        let mut groups: BTreeMap<(Option<u8>, Option<usize>), Named> = BTreeMap::new();
        let mut order: Vec<((i64, usize), (Option<u8>, Option<usize>))> = vec![];
        for (i, d) in reg.defects.iter().enumerate() {
            let want = match g.defect { ND::False => DK::False, ND::Witness => DK::Witness, ND::Value => DK::Value, _ => unreachable!() };
            if d.kind != want { continue; }
            match d.kind {
                DK::Witness => {
                    let (Some(w), Some(lp)) = (d.witness.as_ref(), g.leaf) else { continue };
                    let mut hit: Vec<&Leaf> = w.flips.iter().filter(|lf| at_matches(&lf.at, &lp)).collect();
                    if hit.is_empty() { continue; }
                    if let Some(u) = g.unless { if w.flips.iter().any(|lf| at_matches(&lf.at, &u)) { continue; } }
                    if hit.len() > 1 {
                        let mut keyed: Vec<(String, &Leaf)> = hit.iter().map(|lf| (atom_repr(&sp.prog, it, &lf.atom), *lf)).collect();
                        keyed.sort_by(|p, q| p.0.cmp(&q.0));
                        hit = keyed.into_iter().map(|(_, lf)| lf).collect();
                    }
                    let lf = hit[0];
                    let Some(act) = w.action.as_ref() else { continue };          // (a witness flips a leaf only through its action)
                    let (fr, to) = b.ends(act);
                    let leaf_loc = loc(&lf.atom);
                    let pick = |f: Field| match f { Field::Leaf => leaf_loc, Field::To => to, Field::From => fr };
                    let sq = pick(g.sq);
                    let by = pick(g.by);
                    let gk = if g.group_sq { (sq, None) } else { (sq, Some(i)) };
                    if !groups.contains_key(&gk) {
                        groups.insert(gk, Named { entry: ei, sq, by: vec![], frm: None, to: None, unit: '?', value: None, rel: String::new(), src: vec![] });
                        order.push(((if g.group_sq || g.sort_sq { okey(sq) } else { 0 }, i), gk));
                    }
                    let nd = groups.get_mut(&gk).unwrap();
                    nd.by.push(by);
                    nd.src.push(i);
                    nd.frm = fr;
                    nd.to = to;
                    nd.unit = b.unit_letter(x, fr);
                }
                DK::False => {
                    if let Some((c, l)) = g.clause { if (d.clause, d.lit) != (c, l) { continue; } }
                    let rel = match &d.deep { Some(a) => a.rel, None => d.atom.rel };
                    let rname = if rel == DISTINCT { "distinct".to_string() } else { sp.prog.rel_name(rel).to_string() };
                    if !g.text_any && !g.text_rels.contains(&rname) { continue; }
                    let sq = match &d.deep { Some(a) => loc(a), None => loc(&d.atom) };
                    let gk = (sq, Some(i));
                    groups.insert(gk, Named { entry: ei, sq, by: vec![], frm: None, to: None, unit: '?', value: None, rel: rname, src: vec![i] });
                    order.push(((0, i), gk));
                }
                DK::Value => {
                    let sq = loc(&d.atom);
                    let Some((v, _need, ats)) = d.value.as_ref() else { continue };
                    let v = *v;
                    let mut by: Vec<u8> = ats.iter().filter_map(|a| loc(a)).collect::<BTreeSet<u8>>().into_iter().collect();
                    by.sort_by_key(|s| okey(Some(*s)));
                    let gk = (sq, Some(i));
                    groups.insert(gk, Named { entry: ei, sq, by: by.into_iter().map(Some).collect(), frm: None, to: None, unit: '?', value: Some(v), rel: String::new(), src: vec![i] });
                    order.push(((0, i), gk));
                }
            }
        }
        order.sort_by(|p, q| p.0.cmp(&q.0));
        for (_, gk) in order { out.push(groups.remove(&gk).unwrap()); }
    }
    out
}

/// The (entry, location) keys of a list of names (regression.py `fixes`' keys: (n.kind, n.sq)).
pub fn keys_of(ns: &[Named], entries: &[NameEntry]) -> Vec<(String, Option<u8>)> { ns.iter().map(|n| (entries[n.entry].name.clone(), n.sq)).collect() }

