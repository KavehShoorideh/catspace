//! ANY GDL TEXT AS A GAME (plan 78.1): the spec evaluator (cs_gdl) behind the `Game` trait -- the state is its `true` fluents, the initial state the `init`
//! facts, the legal moves the `legal` rows of the role to move, the next state the `next` fluents under the joint move (`does`: the mover's move, every other
//! role's `noop`, GDL's convention for alternating games), the outcome the `goal` rows at a `terminal` state. Slow; the oracle of every faster binding. Nothing
//! here knows a game: the relation names are GDL's own.
use crate::{Game, Role};
use cs_gdl::{Db, Program, Rel, Row, ARITY_MAX, FREE, T};
use std::cell::RefCell;
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
                     init, does, next_true, noop, last: RefCell::new(None) })
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
    pub fn render(&self, t: T) -> String {
        let p = self.prog.borrow();
        fn go(p: &Program, t: T) -> String {
            match p.terms.compound(t) {
                Some((f, args)) => format!("({} {})", p.terms.name(f), args.iter().map(|a| go(p, *a)).collect::<Vec<_>>().join(" ")),
                None => p.terms.name(t).to_string(),
            }
        }
        go(&p, t)
    }
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
    fn key(&self, s: &GdlState) -> u64 {
        let mut h = DefaultHasher::new();
        s.hash(&mut h);
        h.finish()
    }
}
