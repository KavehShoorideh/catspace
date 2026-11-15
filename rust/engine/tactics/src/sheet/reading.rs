//! ONE POSITION READ ONCE, LIGHT (the Python sheet's `Reading` with `light=True`): the units, the strike tables and the move tables of a unit standing on ANY place (asked of the
//! game on demand and kept), the worth of each kind, the goal's kind, the stakes and the first order of the questions.
use super::goal_report::{self, Report};
use super::medium::Medium;
use super::{Params, Side, Unit};
use cs_game::placed::Placed;
use cs_game::{Game, Role};
use std::cell::{OnceCell, RefCell};
use std::collections::{BTreeSet, HashMap};
use std::rc::Rc;

pub type Places = Rc<BTreeSet<String>>;
type Key = (String, Side, String);

pub struct Reading<'a, G: Game + Placed> {
    pub g: &'a G,
    pub s: G::State,
    pub mover: Role,
    pub units: Vec<Unit>,
    pub medium: Option<&'a Medium>,
    pub can_go: usize,
    pub params_k: u32,
    worth: &'a HashMap<String, f64>,
    goal_kind: Option<String>,
    names: [String; 2],
    index: HashMap<String, usize>,
    weights: Vec<f64>,
    unit_strikes: Vec<Places>,
    into: OnceCell<HashMap<String, Vec<usize>>>,
    acts: RefCell<HashMap<Key, Places>>,
    one: RefCell<HashMap<Key, Places>>,
    within: RefCell<HashMap<(usize, Key), Places>>,
    scenario: RefCell<HashMap<Side, Rc<Report>>>,
    exits: RefCell<HashMap<String, (BTreeSet<String>, &'static str)>>,
    passed: RefCell<Option<Option<G::State>>>,
}

impl<'a, G: Game + Placed> Reading<'a, G> {
    /// Read `s`. `worth` is the worth of every kind of the game by the game's own word; `goal_kind` the kind the goal clause names; `can_go` how many move relations the game's
    /// vocabulary names (the tables of one move, within two, within three).
    pub fn new(g: &'a G, s: G::State, medium: Option<&'a Medium>, worth: &'a HashMap<String, f64>, goal_kind: Option<String>, can_go: usize) -> Result<Reading<'a, G>, String> {
        let mover = match g.mover(&s) { Some(m) => m, None => return Err("the position is over or has no legal moves".into()) };
        if g.moves(&s).is_empty() { return Err("the position is over or has no legal moves".into()); }
        let roles = g.roles();
        let side_of = |owner: Role| if owner == mover { Side::Us } else { Side::Them };
        let mut units: Vec<Unit> = g.cells(&s).into_iter().map(|(o, kind, at)| Unit { side: side_of(o), kind, at }).collect();
        units.sort();
        let names = [roles[mover].clone(), roles[g.other(mover)].clone()];
        let index: HashMap<String, usize> = units.iter().enumerate().map(|(i, u)| (u.at.clone(), i)).collect();
        let worth_of = |kind: &str| worth.get(kind).copied().unwrap_or(0.0);
        let top = units.iter().map(|x| worth_of(&x.kind)).fold(1.0, f64::max) * Params::default().goal_stake_multiple;
        let goal_of = |side: Side| -> Option<usize> {
            let k = goal_kind.as_deref()?;
            let mut found = units.iter().enumerate().filter(|(_, u)| u.side == side && u.kind == k).map(|(i, _)| i);
            let first = found.next()?;
            if found.next().is_some() { None } else { Some(first) }
        };
        let goals = [goal_of(Side::Us), goal_of(Side::Them)];
        let weights: Vec<f64> = units.iter().enumerate().map(|(i, u)| if goals.contains(&Some(i)) { top } else { worth_of(&u.kind) }).collect();
        let owner_of = |side: Side| if side == Side::Us { mover } else { g.other(mover) };
        let unit_strikes: Vec<Places> = units.iter().map(|u| Rc::new(g.bearing_from(&s, owner_of(u.side), &u.kind, &u.at).into_iter().collect::<BTreeSet<String>>())).collect();
        let acts: HashMap<Key, Places> = units.iter().zip(unit_strikes.iter()).map(|(u, p)| ((u.kind.clone(), u.side, u.at.clone()), p.clone())).collect();
        Ok(Reading { g, s, mover, units, medium, can_go, params_k: 2, worth, goal_kind, names, index, weights, unit_strikes, into: OnceCell::new(),
                     acts: RefCell::new(acts), one: RefCell::new(HashMap::new()), within: RefCell::new(HashMap::new()),
                     scenario: RefCell::new(HashMap::new()), exits: RefCell::new(HashMap::new()), passed: RefCell::new(None) })
    }

    pub fn owner(&self, side: Side) -> Role { if side == Side::Us { self.mover } else { self.g.other(self.mover) } }

    /// The side as the game names it (the same name in every position of a game).
    pub fn name(&self, side: Side) -> &str { &self.names[if side == Side::Us { 0 } else { 1 }] }

    pub fn worth_of(&self, kind: &str) -> f64 { self.worth.get(kind).copied().unwrap_or(0.0) }

    pub fn goal_kind(&self) -> Option<&str> { self.goal_kind.as_deref() }

    /// The one unit of `side` of the goal's kind, None when there is none or several.
    pub fn goal_unit(&self, side: Side) -> Option<&Unit> {
        let k = self.goal_kind.as_deref()?;
        let mut found = self.units.iter().filter(|u| u.side == side && u.kind == k);
        let first = found.next()?;
        if found.next().is_some() { None } else { Some(first) }
    }

    /// What losing the unit costs: its worth, and for a goal unit the top stake.
    pub fn weight(&self, u: &Unit) -> f64 {
        match self.index.get(&u.at) { Some(i) if self.units[*i] == *u => self.weights[*i], _ => self.worth_of(&u.kind) }
    }

    /// The index in `units` of the unit standing on `place`.
    pub fn unit_at(&self, place: &str) -> Option<usize> { self.index.get(place).copied() }

    /// The units (by index) that bear on `place`, in the order of `units`.
    pub fn bearers_of(&self, place: &str) -> &[usize] {
        let map = self.into.get_or_init(|| {
            let mut m: HashMap<String, Vec<usize>> = HashMap::new();
            for (i, ps) in self.unit_strikes.iter().enumerate() { for p in ps.iter() { m.entry(p.clone()).or_default().push(i); } }
            m
        });
        map.get(place).map(|v| v.as_slice()).unwrap_or(&[])
    }

    fn top_worth(&self) -> f64 { self.units.iter().map(|x| self.worth_of(&x.kind)).fold(1.0, f64::max) }

    /// The places a unit of `kind` on `side` standing on `at` bears on, the board as it stands.
    pub fn acts_from(&self, kind: &str, side: Side, at: &str) -> Places {
        let key = (kind.to_string(), side, at.to_string());
        if let Some(p) = self.acts.borrow().get(&key) { return p.clone(); }
        let p: Places = Rc::new(self.g.bearing_from(&self.s, self.owner(side), kind, at).into_iter().collect());
        self.acts.borrow_mut().insert(key, p.clone());
        p
    }

    pub fn strikes_of(&self, u: &Unit) -> Places {
        match self.index.get(&u.at) { Some(i) if self.units[*i] == *u => self.unit_strikes[*i].clone(), _ => self.acts_from(&u.kind, u.side, &u.at) }
    }

    /// The places a unit standing on `at` may move to in one move.
    pub fn one_from(&self, kind: &str, side: Side, at: &str) -> Places {
        let key = (kind.to_string(), side, at.to_string());
        if let Some(p) = self.one.borrow().get(&key) { return p.clone(); }
        let p: Places = Rc::new(self.g.reach_from(&self.s, self.owner(side), kind, at).into_iter().collect());
        self.one.borrow_mut().insert(key, p.clone());
        p
    }

    /// `go[j - 1]`: the places reached in at least one and at most `j` moves (j = 1: one move).
    pub fn go_from(&self, j: usize, kind: &str, side: Side, at: &str) -> Places {
        if j <= 1 { return self.one_from(kind, side, at); }
        let key = (j, (kind.to_string(), side, at.to_string()));
        if let Some(p) = self.within.borrow().get(&key) { return p.clone(); }
        let first = self.one_from(kind, side, at);
        let mut reach: BTreeSet<String> = (*first).clone();
        for _ in 1..j {
            let mut next: BTreeSet<String> = (*first).clone();
            for p in &reach { for y in self.one_from(kind, side, p).iter() { next.insert(y.clone()); } }
            reach = next;
        }
        let p: Places = Rc::new(reach);
        self.within.borrow_mut().insert(key, p.clone());
        p
    }

    pub fn exits_of(&self, t: &Unit) -> Places { self.one_from(&t.kind, t.side, &t.at) }

    /// The units of `asker` (other than `target`) that bear on the target's place.
    pub fn strikers_of(&self, target: &Unit, asker: Side) -> Vec<Unit> {
        self.units.iter().filter(|u| u.side == asker && *u != target && self.strikes_of(u).contains(&target.at)).cloned().collect()
    }

    /// Every place a unit of `side` bears on.
    pub fn struck_by(&self, side: Side) -> BTreeSet<String> {
        let mut out = BTreeSet::new();
        for u in self.units.iter().filter(|u| u.side == side) { out.extend(self.strikes_of(u).iter().cloned()); }
        out
    }

    /// The position with the other side to move (the game's pass), None when refused.
    fn passed_state(&self) -> Option<G::State> {
        if self.passed.borrow().is_none() { *self.passed.borrow_mut() = Some(self.g.pass(&self.s)); }
        self.passed.borrow().as_ref().unwrap().clone()
    }

    /// (the places the goal unit `t` may move to, where they were read from): the game's own legal moves of the unit on the position (the side to move) or on the passed position;
    /// when the pass is refused, the places its kind goes from where it stands.
    pub fn goal_exits(&self, t: &Unit) -> (BTreeSet<String>, &'static str) {
        if let Some(e) = self.exits.borrow().get(&t.at) { return e.clone(); }
        let moves: Option<(G::State, Vec<G::Move>)> = if t.side == Side::Us { Some((self.s.clone(), self.g.moves(&self.s))) } else {
            self.passed_state().map(|st| { let ms = self.g.moves(&st); (st, ms) })
        };
        let got = match moves {
            Some((st, ms)) => (ms.iter().map(|m| self.g.move_ends(&st, m)).filter(|(from, _)| *from == t.at).map(|(_, to)| to).collect::<BTreeSet<String>>(), "moves"),
            None => ((*self.exits_of(t)).clone(), "zone"),
        };
        self.exits.borrow_mut().insert(t.at.clone(), got.clone());
        got
    }

    /// The goal unit's report as `role` ("us": the side to move wants the goal; "them": the other), cached.
    pub fn scenario(&self, asker: Side) -> Rc<Report> {
        if let Some(r) = self.scenario.borrow().get(&asker) { return r.clone(); }
        let r = Rc::new(goal_report::report(self, asker));
        self.scenario.borrow_mut().insert(asker, r.clone());
        r
    }

    /// {unit: stake}: the goal unit of each side the top, the rest by worth.
    pub fn stakes(&self, p: &Params) -> Vec<(Unit, f64)> {
        let top = self.top_worth() * p.goal_stake_multiple;
        let goals = [self.goal_unit(Side::Us), self.goal_unit(Side::Them)];
        self.units.iter().map(|u| (u.clone(), if goals.iter().any(|g| *g == Some(u)) { top } else { self.worth_of(&u.kind).max(p.min_stake) })).collect()
    }

    /// The questions' targets in the first order: our unit at the top stake, theirs at the top stake, theirs at the next, ours at the next, then ours, theirs, theirs, ours, ...
    /// -> [(unit, stake, goal, rank)].
    pub fn first_order(&self, p: &Params) -> Vec<(Unit, f64, bool, usize)> {
        let st = self.stakes(p);
        let by = |side: Side| {
            let mut v: Vec<(Unit, f64)> = st.iter().filter(|(u, _)| u.side == side).cloned().collect();
            v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap().then_with(|| a.0.kind.cmp(&b.0.kind)).then_with(|| a.0.at.cmp(&b.0.at)));
            v
        };
        let (mine, theirs) = (by(Side::Us), by(Side::Them));
        let mut order: Vec<(Unit, f64)> = vec![];
        for i in 0..mine.len().max(theirs.len()) {
            let (a, b) = (mine.get(i), theirs.get(i));
            let pair = if i % 2 == 1 { [b, a] } else { [a, b] };
            for x in pair.into_iter().flatten() { order.push(x.clone()); }
        }
        let goals = [self.goal_unit(Side::Us), self.goal_unit(Side::Them)];
        order.into_iter().enumerate().map(|(n, (u, s))| { let goal = goals.iter().any(|g| *g == Some(&u)); (u, s, goal, n + 1) }).collect()
    }
}
