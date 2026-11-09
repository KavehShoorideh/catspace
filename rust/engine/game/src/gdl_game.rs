//! ANY GDL TEXT AS A GAME (plan 78.1): the spec evaluator (cs_gdl) behind the `Game` trait -- the state is its `true` fluents, the initial state the `init`
//! facts, the legal moves the `legal` rows of the role to move, the next state the `next` fluents under the joint move (`does`: the mover's move, every other
//! role's `noop`, GDL's convention for alternating games), the outcome the `goal` rows at a `terminal` state. Slow; the oracle of every faster binding. Nothing
//! here knows a game: the relation names are GDL's own.
use crate::{Fact, Game, Role, Vocabulary};
use cs_gdl::{Db, Program, Rel, Row, ARITY_MAX, FREE, T};
use std::cell::RefCell;
use std::collections::HashMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// A state: its `true` facts, sorted.
pub type GdlState = Vec<(Rel, Row)>;

#[derive(Clone)]
struct Reading { legal: Vec<(T, T)>, terminal: bool, goals: Vec<(T, T)> }

pub struct GdlGame {
    prog: RefCell<Program>,
    label: String,
    roles: Vec<String>,
    role_t: Vec<T>,
    init: GdlState,
    legal: Option<Rel>,
    does: Rel,
    terminal: Option<Rel>,
    goal: Option<Rel>,
    next_true: Vec<(Rel, Rel)>,
    noop: T,
    last: RefCell<Option<(GdlState, Reading)>>,
    vocab: Vocabulary,
    unit_rel: Option<Rel>,
}

impl GdlGame {
    /// The game a GDL text describes (its `role`, `init`, `legal`, `next`, `terminal` and `goal` relations).
    pub fn load(label: &str, text: &str) -> Result<GdlGame, String> {
        let mut p = Program::load(text).map_err(|e| e.0)?;
        let role = p.rel_id("role").ok_or("the description has no `role`")?;
        let rows = p.statics.get(&role).map(|r| r.order.clone()).unwrap_or_default();
        let role_t: Vec<T> = rows.iter().map(|r| r[0]).collect();
        let roles = role_t.iter().map(|t| p.terms.name(*t).to_string()).collect();
        let names: Vec<String> = p.rel_names.clone();
        let mut init = vec![];
        let mut next_true = vec![];
        for (i, n) in names.iter().enumerate() {
            if let Some(f) = n.strip_prefix("init.") {
                if let Some(tr) = p.rel_id(&format!("true.{f}")) {
                    for row in p.statics.get(&(i as Rel)).map(|r| r.order.clone()).unwrap_or_default() { init.push((tr, row)); }
                }
            }
            if let Some(f) = n.strip_prefix("next.") {
                if let Some(tr) = p.rel_id(&format!("true.{f}")) { next_true.push((i as Rel, tr)); }
            }
        }
        init.sort();
        init.dedup();
        let does = p.rel_id("does").ok_or("the description has no `does`")?;
        let noop = p.terms.sym("noop");
        Ok(GdlGame { legal: p.rel_id("legal"), terminal: p.rel_id("terminal"), goal: p.rel_id("goal"), prog: RefCell::new(p), label: label.to_string(), roles, role_t,
                     init, does, next_true, noop, last: RefCell::new(None), vocab: Vocabulary::default(), unit_rel: None })
    }

    /// The game with its fact vocabulary: how its state holds units (so the layers that count units read it) and which relations belong to the fact layer.
    pub fn with_vocabulary(mut self, v: Vocabulary) -> Result<GdlGame, String> {
        if let Some(u) = &v.units {
            let p = self.prog.borrow();
            self.unit_rel = Some(p.rel_id(&format!("true.{}", u.relation)).ok_or_else(|| format!("units: the description has no fluent {}", u.relation))?);
        }
        self.vocab = v;
        Ok(self)
    }

    /// The units of `s` with their places: (owner, kind, place), the place the unit's other arguments joined by `:`.
    pub fn units_placed(&self, s: &GdlState) -> Vec<(Role, String, String)> {
        let (Some(rel), Some(u)) = (self.unit_rel, self.vocab.units.as_ref()) else { return vec![] };
        let mut out = vec![];
        for (r, row) in s.iter().filter(|(r, _)| *r == rel) {
            let _ = r;
            let args = row_args(row);
            let owner = self.render(args[u.owner]);
            let Some(o) = self.roles.iter().position(|n| *n == owner) else { continue };
            let kind = u.kind.map(|k| self.render(args[k])).unwrap_or_else(|| "unit".to_string());
            let place: Vec<String> = args.iter().enumerate().filter(|(i, _)| *i != u.owner && Some(*i) != u.kind).map(|(_, a)| self.render(*a)).collect();
            out.push((o, kind, place.join(":")));
        }
        out
    }

    fn run(&self, s: &GdlState, does: &[(T, T)]) -> Db {
        let mut facts = s.clone();
        for (r, m) in does {
            let mut row = [FREE; ARITY_MAX];
            row[0] = *r;
            row[1] = *m;
            facts.push((self.does, row));
        }
        self.prog.borrow_mut().run(&facts)
    }

    fn read(&self, s: &GdlState) -> Reading {
        if let Some((k, r)) = self.last.borrow().as_ref() { if k == s { return r.clone(); } }
        let db = self.run(s, &[]);
        let rows = |r: Option<Rel>| r.map(|r| db.facts(r)).unwrap_or_default();
        let rd = Reading { legal: rows(self.legal).iter().map(|x| (x[0], x[1])).collect(), terminal: !rows(self.terminal).is_empty(),
                           goals: rows(self.goal).iter().map(|x| (x[0], x[1])).collect() };
        *self.last.borrow_mut() = Some((s.clone(), rd.clone()));
        rd
    }

    /// A ground term's text: a constant, or (functor arg ...).
    pub fn render(&self, t: T) -> String { render_term(&self.prog.borrow(), t) }
}

impl Game for GdlGame {
    type State = GdlState;
    type Move = T;
    fn name(&self) -> String { self.label.clone() }
    fn roles(&self) -> Vec<String> { self.roles.clone() }
    fn initial(&self) -> GdlState { self.init.clone() }
    fn mover(&self, s: &GdlState) -> Option<Role> {
        let rd = self.read(s);
        if rd.terminal { return None; }
        self.role_t.iter().position(|r| rd.legal.iter().any(|(q, m)| q == r && *m != self.noop))
    }
    fn moves(&self, s: &GdlState) -> Vec<T> {
        let Some(r) = self.mover(s) else { return vec![] };
        let rd = self.read(s);
        rd.legal.iter().filter(|(q, _)| *q == self.role_t[r]).map(|(_, m)| *m).collect()
    }
    fn apply(&self, s: &GdlState, m: &T) -> GdlState {
        let me = self.mover(s);
        let does: Vec<(T, T)> = self.role_t.iter().enumerate().map(|(i, r)| (*r, if Some(i) == me { *m } else { self.noop })).collect();
        let db = self.run(s, &does);
        let mut out: GdlState = vec![];
        for (nr, tr) in &self.next_true { for row in db.facts(*nr) { out.push((*tr, row)); } }
        out.sort();
        out.dedup();
        out
    }
    fn outcome(&self, s: &GdlState) -> Option<Vec<i64>> {
        let rd = self.read(s);
        if !rd.terminal { return None; }
        let p = self.prog.borrow();
        Some(self.role_t.iter().map(|r| rd.goals.iter().find(|(q, _)| q == r).and_then(|(_, v)| p.terms.name(*v).parse().ok()).unwrap_or(0)).collect())
    }
    fn move_name(&self, _s: &GdlState, m: &T) -> String { self.render(*m) }
    fn units(&self, s: &GdlState) -> Vec<(Role, String)> { self.units_placed(s).into_iter().map(|(o, k, _)| (o, k)).collect() }
    fn taken_at(&self, s: &GdlState, m: &T) -> Vec<(Role, String, String)> {
        let me = self.mover(s);
        let mut left: Vec<(Role, String, String)> = self.units_placed(&self.apply(s, m)).into_iter().filter(|(o, _, _)| Some(*o) != me).collect();
        let mut out = vec![];
        for u in self.units_placed(s).into_iter().filter(|(o, _, _)| Some(*o) != me) {
            if let Some(i) = left.iter().position(|x| *x == u) { left.swap_remove(i); } else { out.push(u); }
        }
        out
    }
    fn base_facts(&self, s: &GdlState) -> Vec<Fact> {
        let p = self.prog.borrow();
        let mut out: Vec<Fact> = s.iter().map(|(r, row)| Fact {
            relation: p.rel_name(*r).strip_prefix("true.").unwrap_or(p.rel_name(*r)).to_string(),
            args: row_args(row).iter().map(|t| render_term(&p, *t)).collect(),
        }).collect();
        out.sort();
        out
    }
    fn key(&self, s: &GdlState) -> u64 {
        let mut h = DefaultHasher::new();
        s.hash(&mut h);
        h.finish()
    }
}

/// A ground term's text: a constant, or (functor arg ...).
pub fn render_term(p: &Program, t: T) -> String {
    match p.terms.compound(t) {
        Some((f, args)) => format!("({} {})", p.terms.name(f), args.iter().map(|a| render_term(p, *a)).collect::<Vec<_>>().join(" ")),
        None => p.terms.name(t).to_string(),
    }
}

/// The arguments of a row (up to its first FREE).
pub fn row_args(row: &Row) -> Vec<T> { row.iter().copied().take_while(|t| *t != FREE).collect() }

/// A term from its text: a constant, or (functor arg ...) with arguments in the same form.
pub fn parse_term(p: &mut Program, text: &str) -> Result<T, String> {
    fn go(p: &mut Program, toks: &[String], i: &mut usize) -> Result<T, String> {
        let tok = toks.get(*i).ok_or("a term ends too soon")?.clone();
        *i += 1;
        if tok != "(" { return Ok(p.terms.sym(&tok)); }
        let f = toks.get(*i).ok_or("a term ends too soon")?.clone();
        *i += 1;
        let ft = p.terms.sym(&f);
        let mut args = vec![];
        while toks.get(*i).map(|x| x.as_str()) != Some(")") {
            if *i >= toks.len() { return Err("a term is not closed".into()); }
            args.push(go(p, toks, i)?);
        }
        *i += 1;
        Ok(p.terms.fun(ft, &args))
    }
    let spaced = text.replace('(', " ( ").replace(')', " ) ");
    let toks: Vec<String> = spaced.split_whitespace().map(|x| x.to_string()).collect();
    let mut i = 0;
    let t = go(p, &toks, &mut i)?;
    if i != toks.len() { return Err(format!("trailing text in a term: {text}")); }
    Ok(t)
}

/// THE FACT VIEW (plan 78.13): a game's description evaluated over a state's base facts, giving the fact layer -- the base facts and the derived relations the
/// game's vocabulary names (attack, check, ... whatever its description derives). Built from the description text and the vocabulary; knows no game. The base
/// facts come from any `Game::base_facts`: a fast binding's are certified against the oracle's by co-play, so both give the same view.
pub struct FactView {
    prog: RefCell<Program>,
    vocab: Vocabulary,
    derived: Vec<Rel>,
    demand: bool,
    memo: RefCell<Memo>,
}

/// A3-lite: what the view has already read or rendered -- a base fact's argument text -> its interned term, a term -> its text, a fluent name -> its relation.
/// The interner only grows, so an entry never goes stale.
#[derive(Default)]
struct Memo { term_of: HashMap<String, T>, text_of: HashMap<T, String>, fluent: HashMap<String, Option<Rel>> }

impl FactView {
    pub fn new(text: &str, vocab: Vocabulary) -> Result<FactView, String> {
        let p = Program::load(text).map_err(|e| e.0)?;
        let mut derived = vec![];
        for d in &vocab.derived { derived.push(p.rel_id(d).ok_or_else(|| format!("the description has no relation {d} (the vocabulary names it)"))?); }
        Ok(FactView { prog: RefCell::new(p), vocab, derived, demand: true, memo: RefCell::new(Memo::default()) })
    }
    /// The view with DEMAND on (default: only the strata the vocabulary's relations depend on are evaluated) or off (the full `Program::run`, the oracle).
    pub fn with_demand(mut self, on: bool) -> FactView { self.demand = on; self }
    pub fn demand(&self) -> bool { self.demand }
    pub fn vocabulary(&self) -> &Vocabulary { &self.vocab }
    /// (relations, clauses) of the dependency cone of the vocabulary's derived relations, and the description's totals (relations, clauses).
    pub fn cone_size(&self) -> ((usize, usize), (usize, usize)) {
        let p = self.prog.borrow();
        let c = p.cone(&self.derived);
        ((c.len(), p.cone_rules(&c)), (p.nodes.len(), p.clauses.len()))
    }

    /// The base facts as interned rows (memoised: a repeated argument is parsed once). A base fact the description has no fluent for is an error.
    fn rows_of(&self, p: &mut Program, base: &[Fact]) -> Result<Vec<(Rel, Row)>, String> {
        let mut m = self.memo.borrow_mut();
        let mut rows = Vec::with_capacity(base.len());
        for f in base {
            let r = match m.fluent.get(&f.relation) {
                Some(r) => *r,
                None => { let r = p.rel_id(&format!("true.{}", f.relation)); m.fluent.insert(f.relation.clone(), r); r }
            };
            let r = r.ok_or_else(|| format!("the description has no fluent {}", f.relation))?;
            if f.args.len() > ARITY_MAX { return Err(format!("{f}: too many arguments")); }
            let mut row = [FREE; ARITY_MAX];
            for (i, a) in f.args.iter().enumerate() {
                row[i] = match m.term_of.get(a) {
                    Some(t) => *t,
                    None => { let t = parse_term(p, a)?; m.term_of.insert(a.clone(), t); t }
                };
            }
            rows.push((r, row));
        }
        Ok(rows)
    }

    fn evaluate(&self, p: &mut Program, rows: &[(Rel, Row)]) -> Db {
        if self.demand { p.run_for(rows, &self.derived) } else { p.run(rows) }
    }

    /// The derived rows of a state as interned rows, relation by relation in the vocabulary's order (what a later stage reads without strings).
    pub fn rows(&self, base: &[Fact]) -> Result<Vec<(Rel, Row)>, String> {
        let mut p = self.prog.borrow_mut();
        let rows = self.rows_of(&mut p, base)?;
        let db = self.evaluate(&mut p, &rows);
        let mut out = vec![];
        for &r in &self.derived { for row in db.facts(r) { out.push((r, row)); } }
        Ok(out)
    }

    /// The fact layer of a state: its base facts and the derived relations of the vocabulary, sorted. A base fact the description has no fluent for is an error.
    pub fn facts(&self, base: &[Fact]) -> Result<Vec<Fact>, String> {
        let mut p = self.prog.borrow_mut();
        let rows = self.rows_of(&mut p, base)?;
        let db = self.evaluate(&mut p, &rows);
        let mut m = self.memo.borrow_mut();
        let mut out: Vec<Fact> = base.to_vec();
        for (d, &r) in self.vocab.derived.iter().zip(&self.derived) {
            for row in db.facts(r) {
                let args = row_args(&row).iter().map(|t| m.text_of.entry(*t).or_insert_with(|| render_term(&p, *t)).clone()).collect();
                out.push(Fact { relation: d.clone(), args });
            }
        }
        out.sort();
        out.dedup();
        Ok(out)
    }
}
