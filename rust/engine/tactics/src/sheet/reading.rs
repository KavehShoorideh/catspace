//! ONE POSITION READ ONCE, LIGHT (the Python sheet's `Reading` with `light=True`): the units, the strike tables and the move tables of a unit standing on ANY place (asked of the
//! game on demand and kept), the worth of each kind, the goal's kind, the stakes and the first order of the questions. Places and kinds are numbers; the tables are arrays
//! indexed by (kind, side, place), filled on first use.
use super::goal_report::{self, Report};
use super::{FxMap, Kind, Model, Params, Place, PlaceSet, Side, Unit};
use cs_game::placed::Placed;
use cs_game::{Game, Role};
use std::cell::{OnceCell, RefCell};
use std::rc::Rc;

pub type Places = Rc<PlaceSet>;

pub struct Reading<'a, G: Game + Placed> {
    pub g: &'a G,
    pub s: G::State,
    pub mover: Role,
    pub units: Vec<Unit>,
    pub m: &'a Model,
    pub params_k: u32,
    np: usize,
    /// place -> index in `units` + 1 (0: nothing stands there)
    occupant: Vec<u16>,
    weights: Vec<f64>,
    unit_strikes: Vec<Places>,
    into: OnceCell<Vec<Vec<usize>>>,
    acts: RefCell<Vec<Option<Places>>>,
    one: RefCell<Vec<Option<Places>>>,
    within: RefCell<FxMap<(u32, usize), Places>>,
    scenario: RefCell<[Option<Rc<Report>>; 2]>,
    exits: RefCell<Vec<(Place, Places, &'static str)>>,
    passed: RefCell<Option<Option<G::State>>>,
}

impl<'a, G: Game + Placed> Reading<'a, G> {
    /// Read `s` under the model of the game.
    pub fn new(g: &'a G, s: G::State, m: &'a Model) -> Result<Reading<'a, G>, String> {
        let mover = match g.mover(&s) { Some(x) => x, None => return Err("the position is over or has no legal moves".into()) };
        if g.moves(&s).is_empty() { return Err("the position is over or has no legal moves".into()); }
        let np = m.names.places.len();
        let side_of = |owner: Role| if owner == mover { Side::Us } else { Side::Them };
        let mut units: Vec<Unit> = g.cells(&s).into_iter().map(|(o, kind, at)| Unit { side: side_of(o), kind, at }).collect();
        units.sort();
        let mut occupant = vec![0u16; np];
        for (i, u) in units.iter().enumerate() { occupant[u.at as usize] = i as u16 + 1; }
        let worth_of = |kind: Kind| m.worth[kind as usize];
        let top = units.iter().map(|x| worth_of(x.kind)).fold(1.0, f64::max) * Params::default().goal_stake_multiple;
        let goal_of = |side: Side| -> Option<usize> {
            let k = m.goal_kind?;
            let mut found = units.iter().enumerate().filter(|(_, u)| u.side == side && u.kind == k).map(|(i, _)| i);
            let first = found.next()?;
            if found.next().is_some() { None } else { Some(first) }
        };
        let goals = [goal_of(Side::Us), goal_of(Side::Them)];
        let weights: Vec<f64> = units.iter().enumerate().map(|(i, u)| if goals.contains(&Some(i)) { top } else { worth_of(u.kind) }).collect();
        let owner_of = |side: Side| if side == Side::Us { mover } else { g.other(mover) };
        let unit_strikes: Vec<Places> = units.iter().map(|u| Rc::new(PlaceSet::from_places(np, g.bearing_from(&s, owner_of(u.side), u.kind, u.at)))).collect();
        let nk = m.names.kinds.len();
        let mut acts: Vec<Option<Places>> = vec![None; nk * 2 * np];
        for (u, p) in units.iter().zip(unit_strikes.iter()) { acts[(u.kind as usize * 2 + u.side.ix()) * np + u.at as usize] = Some(p.clone()); }
        Ok(Reading { g, s, mover, units, m, params_k: 2, np, occupant, weights, unit_strikes, into: OnceCell::new(), acts: RefCell::new(acts), one: RefCell::new(vec![None; nk * 2 * np]),
                     within: RefCell::new(FxMap::default()), scenario: RefCell::new([None, None]), exits: RefCell::new(vec![]), passed: RefCell::new(None) })
    }

    pub fn np(&self) -> usize { self.np }

    pub fn owner(&self, side: Side) -> Role { if side == Side::Us { self.mover } else { self.g.other(self.mover) } }

    pub fn worth_of(&self, kind: Kind) -> f64 { self.m.worth[kind as usize] }

    pub fn goal_kind(&self) -> Option<Kind> { self.m.goal_kind }

    /// The one unit of `side` of the goal's kind, None when there is none or several.
    pub fn goal_unit(&self, side: Side) -> Option<Unit> {
        let k = self.m.goal_kind?;
        let mut found = self.units.iter().filter(|u| u.side == side && u.kind == k);
        let first = found.next()?;
        if found.next().is_some() { None } else { Some(*first) }
    }

    /// What losing the unit costs: its worth, and for a goal unit the top stake.
    pub fn weight(&self, u: &Unit) -> f64 {
        match self.unit_at(u.at) { Some(i) if self.units[i] == *u => self.weights[i], _ => self.worth_of(u.kind) }
    }

    /// The index in `units` of the unit standing on `place`.
    pub fn unit_at(&self, place: Place) -> Option<usize> { match self.occupant[place as usize] { 0 => None, i => Some(i as usize - 1) } }

    /// The units (by index) that bear on `place`, in the order of `units`.
    pub fn bearers_of(&self, place: Place) -> &[usize] {
        let map = self.into.get_or_init(|| {
            let mut m: Vec<Vec<usize>> = vec![vec![]; self.np];
            for (i, ps) in self.unit_strikes.iter().enumerate() { for p in ps.iter() { m[p as usize].push(i); } }
            m
        });
        &map[place as usize]
    }

    pub fn top_worth(&self) -> f64 { self.units.iter().map(|x| self.worth_of(x.kind)).fold(1.0, f64::max) }

    fn slot(&self, kind: Kind, side: Side, at: Place) -> usize { (kind as usize * 2 + side.ix()) * self.np + at as usize }

    /// The places a unit of `kind` on `side` standing on `at` bears on, the board as it stands.
    pub fn acts_from(&self, kind: Kind, side: Side, at: Place) -> Places {
        let i = self.slot(kind, side, at);
        if let Some(p) = &self.acts.borrow()[i] { return p.clone(); }
        let p: Places = Rc::new(PlaceSet::from_places(self.np, self.g.bearing_from(&self.s, self.owner(side), kind, at)));
        self.acts.borrow_mut()[i] = Some(p.clone());
        p
    }

    pub fn strikes_of(&self, u: &Unit) -> Places {
        match self.unit_at(u.at) { Some(i) if self.units[i] == *u => self.unit_strikes[i].clone(), _ => self.acts_from(u.kind, u.side, u.at) }
    }

    /// Whether the unit bears on `place`.
    pub fn strikes(&self, u: &Unit, place: Place) -> bool {
        match self.unit_at(u.at) { Some(i) if self.units[i] == *u => self.unit_strikes[i].contains(place), _ => self.acts_from(u.kind, u.side, u.at).contains(place) }
    }

    /// The places a unit standing on `at` may move to in one move.
    pub fn one_from(&self, kind: Kind, side: Side, at: Place) -> Places {
        let i = self.slot(kind, side, at);
        if let Some(p) = &self.one.borrow()[i] { return p.clone(); }
        let p: Places = Rc::new(PlaceSet::from_places(self.np, self.g.reach_from(&self.s, self.owner(side), kind, at)));
        self.one.borrow_mut()[i] = Some(p.clone());
        p
    }

    /// `go[j - 1]`: the places reached in at least one and at most `j` moves (j = 1: one move).
    pub fn go_from(&self, j: usize, kind: Kind, side: Side, at: Place) -> Places {
        if j <= 1 { return self.one_from(kind, side, at); }
        let key = (j as u32, self.slot(kind, side, at));
        if let Some(p) = self.within.borrow().get(&key) { return p.clone(); }
        let first = self.one_from(kind, side, at);
        let mut reach: PlaceSet = (*first).clone();
        for _ in 1..j {
            let mut next: PlaceSet = (*first).clone();
            for p in reach.iter() { next.union_with(&self.one_from(kind, side, p)); }
            reach = next;
        }
        let p: Places = Rc::new(reach);
        self.within.borrow_mut().insert(key, p.clone());
        p
    }

    pub fn exits_of(&self, t: &Unit) -> Places { self.one_from(t.kind, t.side, t.at) }

    /// The units of `asker` (other than `target`) that bear on the target's place.
    pub fn strikers_of(&self, target: &Unit, asker: Side) -> Vec<Unit> {
        self.units.iter().filter(|u| u.side == asker && *u != target && self.strikes(u, target.at)).copied().collect()
    }

    /// Every place a unit of `side` bears on.
    pub fn struck_by(&self, side: Side) -> PlaceSet {
        let mut out = PlaceSet::new(self.np);
        for u in self.units.iter().filter(|u| u.side == side) { out.union_with(&self.strikes_of(u)); }
        out
    }

    /// The position with the other side to move (the game's pass), None when refused.
    pub fn passed_state(&self) -> Option<G::State> {
        if self.passed.borrow().is_none() { *self.passed.borrow_mut() = Some(self.g.pass(&self.s)); }
        self.passed.borrow().as_ref().unwrap().clone()
    }

    /// (the places the goal unit `t` may move to, where they were read from): the game's own legal moves of the unit on the position (the side to move) or on the passed position;
    /// when the pass is refused, the places its kind goes from where it stands.
    pub fn goal_exits(&self, t: &Unit) -> (Places, &'static str) {
        if let Some(e) = self.exits.borrow().iter().find(|e| e.0 == t.at) { return (e.1.clone(), e.2); }
        let moves: Option<(G::State, Vec<G::Move>)> = if t.side == Side::Us { Some((self.s.clone(), self.g.moves(&self.s))) } else {
            self.passed_state().map(|st| { let ms = self.g.moves(&st); (st, ms) })
        };
        let got: (Places, &'static str) = match moves {
            Some((st, ms)) => (Rc::new(PlaceSet::from_places(self.np, ms.iter().map(|m| self.g.move_ends(&st, m)).filter(|(from, _)| *from == t.at).map(|(_, to)| to))), "moves"),
            None => (self.exits_of(t), "zone"),
        };
        self.exits.borrow_mut().push((t.at, got.0.clone(), got.1));
        got
    }

    /// The goal unit's report as `asker` wants the goal, cached.
    pub fn scenario(&self, asker: Side) -> Rc<Report> {
        if let Some(r) = &self.scenario.borrow()[asker.ix()] { return r.clone(); }
        let r = Rc::new(goal_report::report(self, asker));
        self.scenario.borrow_mut()[asker.ix()] = Some(r.clone());
        r
    }

    /// {unit: stake}: the goal unit of each side the top, the rest by worth.
    pub fn stakes(&self, p: &Params) -> Vec<(Unit, f64)> {
        let top = self.top_worth() * p.goal_stake_multiple;
        let goals = [self.goal_unit(Side::Us), self.goal_unit(Side::Them)];
        self.units.iter().map(|u| (*u, if goals.iter().any(|g| *g == Some(*u)) { top } else { self.worth_of(u.kind).max(p.min_stake) })).collect()
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
            for x in pair.into_iter().flatten() { order.push(*x); }
        }
        let goals = [self.goal_unit(Side::Us), self.goal_unit(Side::Them)];
        order.into_iter().enumerate().map(|(n, (u, s))| { let goal = goals.iter().any(|g| *g == Some(u)); (u, s, goal, n + 1) }).collect()
    }
}
