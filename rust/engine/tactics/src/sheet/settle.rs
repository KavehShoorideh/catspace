//! THE DEEPER LEVELS OF THE QUESTION SHEET, game-free (the Python sheet's `level_deep`, `level_frames`, `level_search`, `level_census`, `settle_way`, `_settle_way`,
//! `exchange_stands`, `exchange_after`, `cornering`, `ask_obstacle`, `cost_of_leaving`, `goal_sign`, `_duty_reading`, `apply_duties`, `_refute_counterfactual`), on the light reading, the
//! LIGHT path only. A `Settler` holds one reading and the answers the Python sheet keeps on its reading (`_frames`: the exchanges as they stand, the exchanges after a way's first
//! move, the cost of leaving, the follow-up on an obstacle, the settled ways; `_settled`: the search on a child) -- the caches are kept with the keys the Python sheet keys them
//! by, so asking in the same order gives the same answers. Where the Python sheet calls the kernel's search, census or pass, the same engine functions are called here directly.
//! The verdict of a way is DATA: CONFIRMED, REFUTED, WAITING or NOT SETTLED, with the obstacles, the stops and what it waits on; words are made at the boundary.
//! Not ported: the claims graph, the pool of waiting ways, the chance table, the time obstacle (a note on a confirmed way read from the other questions).
use super::cf::{counterfactual_obstacles, CfObstacle};
use super::frame::{solve_real, solve_static, static_state, Solved};
use super::levels::{self, Level1, Route, What, Way};
use super::reading::Reading;
use super::walk::{search_way, Stop, Walk, BIG, V};
use super::{Params, Place, PlaceSet, Side, Status, Unit};
use crate::census::{self, Tactic};
use crate::ledger::Worth;
use crate::reason;
use cs_game::placed::Placed;
use cs_game::Game;
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

/// A question as the deeper levels see it: its target, its asker, its stake, whether the target is the goal unit, the blocked steps the counterfactual has examined for it so far,
/// and its level 1 (the ways and the routes the deeper levels settle).
#[derive(Clone)]
pub struct Q { pub t: Unit, pub asker: Side, pub stake: f64, pub goal: bool, pub blocked: usize, pub l1: Rc<Level1> }

impl Q { pub fn owner(&self) -> Side { self.asker.other() } }

/// A duty a travelling unit holds where it stands.
#[derive(Debug, Clone)]
pub struct Duty { pub unit: Unit, pub stake: f64, pub kind: &'static str, pub value: f64 }

#[derive(Debug, Clone)]
pub struct Leaving { pub read: bool, pub duties: Vec<Duty> }

/// The follow-up question on one obstacle: movable | takeable | dead | not read.
#[derive(Debug, Clone)]
pub struct Ask { pub verdict: &'static str, pub cost: f64, pub more: u32, pub duties: Vec<Duty>, pub mv: Option<String> }

fn dead() -> Ask { Ask { verdict: "dead", cost: 0.0, more: 0, duties: vec![], mv: None } }

/// The verdict of one way at level 2, as a result entry (the first entry of a frame's results is the exchange as it stands: `way` None).
#[derive(Debug, Clone)]
pub struct Entry {
    pub way: Option<Way>,
    pub status: V,
    pub value: f64,
    pub line: Vec<String>,
    pub source: &'static str,
    pub open: bool,
    pub nodes: Option<u64>,
    pub mv: Option<String>,
    pub route: Option<Route>,
    pub obstacles: Vec<(CfObstacle, Ask)>,
    pub first_obstacles: Option<Vec<(Option<Side>, &'static str, bool)>>,
    pub stops: Vec<Stop>,
    pub alts: Vec<(&'static str, Vec<Place>)>,
    pub breaks: Vec<Stop>,
    pub duties: Option<Vec<Duty>>,
    pub held_back: bool,
    pub skipped: bool,
    pub dead_routes: Option<usize>,
    pub unread_routes: Option<usize>,
    pub without_stopper: Option<(Unit, V, f64)>,
    pub tried: Option<usize>,
    pub cut: Option<bool>,
    pub routes_tried: Option<usize>,
    pub answer: Option<String>,
}

impl Entry {
    fn blank(way: Option<Way>, status: V) -> Entry {
        Entry { way, status, value: 0.0, line: vec![], source: "route", open: false, nodes: None, mv: None, route: None, obstacles: vec![], first_obstacles: None, stops: vec![], alts: vec![],
                breaks: vec![], duties: None, held_back: false, skipped: false, dead_routes: None, unread_routes: None, without_stopper: None, tried: None, cut: None, routes_tried: None, answer: None }
    }
    fn from_solved(way: Option<Way>, s: &Solved, with_nodes: bool) -> Entry {
        let mut e = Entry::blank(way, if s.value > 0.0 { V::Confirmed } else { V::Refuted });
        e.value = s.value;
        e.line = s.line.clone();
        e.source = s.source;
        e.open = s.open;
        if with_nodes { e.nodes = Some(s.nodes); }
        e
    }
}

#[derive(Debug, Clone)]
pub struct Cornering { pub cornered: bool, pub lost_now: bool, pub places: Vec<(Place, bool)>, pub open_exits: usize }

#[derive(Debug, Clone)]
pub struct FrameDetail {
    pub results: Vec<Entry>,
    pub confirmed: usize,
    pub refuted: usize,
    pub waiting: usize,
    pub open: usize,
    pub lost_where_it_stands: bool,
    pub cornering: Option<Cornering>,
    pub value: Option<f64>,
    pub line: Option<Vec<String>>,
    pub sources: Vec<&'static str>,
}

/// The result of playing a way's first move and reading the child by the reasoning search: confirmed | refuted | open.
#[derive(Debug, Clone)]
pub struct SearchRes { pub mv: Option<String>, pub status: &'static str, pub value: f64, pub reply: Option<String> }

#[derive(Debug, Clone)]
pub struct SearchDetail { pub depth: usize, pub ways: Vec<Way>, pub results: Vec<SearchRes>, pub confirmed: usize, pub refuted: usize, pub open: usize, pub value: Option<f64>, pub line: Option<String> }

#[derive(Debug, Clone)]
pub struct CensusDetail { pub depth: usize, pub tactics: usize, pub value: Option<i64>, pub line: Option<Vec<String>> }

#[derive(Debug, Clone)]
pub enum Detail { Frame(FrameDetail), Search(SearchDetail), Census(CensusDetail) }

#[derive(Debug, Clone)]
pub struct Finding { pub status: Status, pub detail: Detail }

pub struct Settler<'a, 'g, G: Game + Placed> {
    pub r: &'a Reading<'g, G>,
    pub p: &'a Params,
    stands: RefCell<HashMap<(Unit, Side), Rc<Solved>>>,
    after: RefCell<HashMap<(Unit, Side, Way), Rc<Solved>>>,
    duty: RefCell<HashMap<(Side, Place, Option<String>), Rc<Leaving>>>,
    obstacle: RefCell<HashMap<(Side, Place, Vec<Place>, &'static str), Rc<Ask>>>,
    settled: RefCell<HashMap<(Unit, Side, Way), Rc<Entry>>>,
    sign: RefCell<HashMap<Side, bool>>,
    sign_child: RefCell<HashMap<u64, bool>>,
    searched: RefCell<HashMap<(String, usize), Rc<SearchRes>>>,
    census: RefCell<HashMap<(Side, usize), Rc<Vec<Tactic>>>>,
}

impl<'a, 'g, G: Game + Placed> Settler<'a, 'g, G> {
    pub fn new(r: &'a Reading<'g, G>, p: &'a Params) -> Self {
        Settler { r, p, stands: Default::default(), after: Default::default(), duty: Default::default(), obstacle: Default::default(), settled: Default::default(), sign: Default::default(),
                  sign_child: Default::default(), searched: Default::default(), census: Default::default() }
    }

    /// The position the asker reads from: the position as it stands when the asker is to move, else the game's pass of it (None when refused).
    fn base(&self, asker: Side) -> Option<G::State> { if asker == Side::Us { Some(self.r.s.clone()) } else { self.r.passed_state() } }

    fn worth(&self, kind: super::Kind) -> f64 { self.r.worth_of(kind) }

    fn lowest_worth(&self) -> f64 { self.r.m.worth.iter().copied().filter(|w| *w > 0.0).fold(None, |m: Option<f64>, w| Some(m.map_or(w, |x| x.min(w)))).unwrap_or(0.0) }

    /// A goal unit, or a unit whose kind is worth more than the lowest worth.
    fn weighty(&self, u: &Unit) -> bool { Some(*u) == self.r.goal_unit(u.side) || self.worth(u.kind) > self.lowest_worth() }

    // ------------------------------------------------------------------------------------------------------------------------------------------ the exchange frames

    fn exchange_static(&self, t: &Unit, asker: Side, strikers: Option<Vec<Unit>>, at: Option<Place>) -> Solved {
        let r = self.r;
        let at = at.unwrap_or(t.at);
        let strikers = strikers.unwrap_or_else(|| r.strikers_of(t, asker));
        let guards: Vec<Unit> = r.units.iter().filter(|u| u.side == t.side && *u != t && r.strikes(u, at)).copied().collect();
        solve_static(r, &static_state(r, &strikers, &guards, self.worth(t.kind)), self.p.frame_budget)
    }

    /// The exchange frame started by the asker on the target as it stands (cached): the real legal takings when the asker is to move or its pass is given, else the static form.
    pub fn exchange_stands(&self, t: &Unit, asker: Side) -> Rc<Solved> {
        if let Some(s) = self.stands.borrow().get(&(*t, asker)) { return s.clone(); }
        let s = Rc::new(match self.base(asker) {
            Some(b) => solve_real(self.r, t.at, &b, self.p.frame_budget, "real"),
            None => self.exchange_static(t, asker, None, None),
        });
        self.stands.borrow_mut().insert((*t, asker), s.clone());
        s
    }

    /// The (origin, destination) of the first move of a way: the strike itself for a unit already striking, the step to the arrival place, or to a place from which the arrival place is one move on.
    fn first_moves(&self, t: &Unit, way: &Way) -> Vec<(Place, Place)> {
        let (u, to) = (way.unit, way.arrival);
        match way.d {
            0 => vec![(u.at, t.at)],
            1 => vec![(u.at, to)],
            _ => self.r.one_from(u.kind, u.side, u.at).iter().filter(|m| self.r.one_from(u.kind, u.side, *m).contains(to)).map(|m| (u.at, m)).collect(),
        }
    }

    /// The exchange frame after the way's first move (a unit of the asker comes to strike): the first move is played for real, the owner answers with every legal move (the target followed
    /// where it goes), and the asker starts the real frame from there; the value is the least over the answers. The static form with the supplier placed when the pass is refused.
    fn exchange_after(&self, t: &Unit, asker: Side, way: &Way) -> Rc<Solved> {
        let key = (*t, asker, *way);
        if let Some(s) = self.after.borrow().get(&key) { return s.clone(); }
        let (r, g) = (self.r, self.r.g);
        let mut out: Option<Solved> = None;
        if let Some(base) = self.base(asker) {
            let mut mv = None;
            for (o, d) in self.first_moves(t, way) {
                if let Some(m) = g.find_step(&base, o, d) { mv = Some(m); break; }
            }
            if let Some(mv) = mv {
                let name = g.move_name(&base, &mv);
                let child = g.apply(&base, &mv);
                if let Some(o) = g.outcome(&child) {
                    out = Some(Solved { value: if o.contains(&100) { BIG } else { 0.0 }, line: vec![name], source: "given", open: false, nodes: 0 });
                } else {
                    let mut worst: Option<(f64, Vec<String>)> = None;
                    for m2 in g.moves(&child) {
                        let nxt = g.apply(&child, &m2);
                        let (from_h, to_h) = g.move_ends(&child, &m2);
                        let moved_to = if from_h == t.at { to_h } else { t.at };
                        let n2 = g.move_name(&child, &m2);
                        let v: (f64, Vec<String>) = if g.outcome(&nxt).is_some() || g.kind_at(&nxt, moved_to).is_none() { (0.0, vec![name.clone(), n2]) } else {
                            let s = solve_real(r, moved_to, &nxt, self.p.frame_budget, "given");
                            let mut l = vec![name.clone(), n2];
                            l.extend(s.line);
                            (s.value, l)
                        };
                        if worst.as_ref().is_none_or(|w| v.0 < w.0) { worst = Some(v); }
                        if worst.as_ref().unwrap().0 <= 0.0 { break; }
                    }
                    out = Some(match worst { Some((value, line)) => Solved { value, line, source: "given", open: false, nodes: 0 }, None => Solved { value: 0.0, line: vec![name], source: "given", open: false, nodes: 0 } });
                }
            }
        }
        let out = out.unwrap_or_else(|| {
            let mut strikers: Vec<Unit> = r.strikers_of(t, asker).into_iter().filter(|v| *v != way.unit).collect();
            if way.what == What::Strike { strikers.push(way.unit); }
            self.exchange_static(t, asker, Some(strikers), Some(t.at))
        });
        let out = Rc::new(out);
        self.after.borrow_mut().insert(key, out.clone());
        out
    }

    /// Is the unit cornered: lost where it stands and lost on every place it can go to (the asker's exchange frames say so).
    pub fn cornering(&self, t: &Unit, asker: Side) -> Cornering {
        let (r, g) = (self.r, self.r.g);
        let now = self.exchange_stands(t, asker);
        let mut places: Vec<(Place, bool)> = vec![];
        let real = |base: &G::State, places: &mut Vec<(Place, bool)>| {
            for m in g.moves(base) {
                let (from, dest) = g.move_ends(base, &m);
                if from != t.at { continue; }
                let child = g.apply(base, &m);
                if g.outcome(&child).is_some() { places.push((dest, false)); continue; }
                let v = solve_real(r, dest, &child, self.p.frame_budget, "real");
                places.push((dest, v.value > 0.0));
            }
        };
        if asker == Side::Them {
            real(&r.s, &mut places);
        } else if let Some(base) = r.passed_state() {
            real(&base, &mut places);
        } else {
            let own: PlaceSet = PlaceSet::from_places(r.np(), r.units.iter().filter(|u| u.side == t.side && *u != t).map(|u| u.at));
            for to in r.exits_of(t).iter().filter(|p| !own.contains(*p)) {
                let hit: Vec<Unit> = r.units.iter().filter(|u| u.side == asker && u.at != to && r.strikes(u, to)).copied().collect();
                let gd: Vec<Unit> = r.units.iter().filter(|u| u.side == t.side && *u != t && r.strikes(u, to)).copied().collect();
                let v = solve_static(r, &static_state(r, &hit, &gd, self.worth(t.kind)), self.p.frame_budget);
                places.push((to, v.value > 0.0));
            }
        }
        let lost_now = now.value > 0.0;
        Cornering { cornered: lost_now && places.iter().all(|p| p.1), lost_now, open_exits: places.iter().filter(|p| !p.1).count(), places }
    }

    // ------------------------------------------------------------------------------------------------------------------------------------------ the cost of leaving

    /// The value of the other side's exchange on `v`'s place in `child` when that is a newly lost unit (it was not lost where it stands before), else None.
    fn lost_after(&self, q: &Q, v: &Unit, child: &G::State) -> Option<f64> {
        self.r.g.kind_at(child, v.at)?;
        if self.exchange_stands(v, q.owner()).value > 0.0 { return None; }
        let ex = solve_real(self.r, v.at, child, self.p.frame_budget, "given");
        if ex.value > 0.0 { Some(ex.value) } else { None }
    }

    /// The goal unit of `side` is under the sign: struck now (level 0 live) or at most `max_open_exits` exits open with an asker unit near (level 1 live).
    fn goal_sign(&self, r: &Reading<G>, side: Side) -> bool {
        let Some(gu) = r.goal_unit(side) else { return false };
        let asker = side.other();
        levels::level0(r, &gu, asker, true).status == Status::Live || levels::level1(r, &gu, asker, true, self.p).status == Status::Live
    }

    fn goal_newly_under_sign(&self, q: &Q, child: &G::State) -> bool {
        let key = self.r.g.key(child);
        let after = match self.sign_child.borrow().get(&key) { Some(b) => Some(*b), None => None };
        let after = match after {
            Some(b) => b,
            None => {
                let b = match Reading::new(self.r.g, child.clone(), self.r.m) { Ok(ar) => self.goal_sign(&ar, Side::Them), Err(_) => return false };
                self.sign_child.borrow_mut().insert(key, b);
                b
            }
        };
        let before = match self.sign.borrow().get(&q.asker) { Some(b) => Some(*b), None => None };
        let before = match before { Some(b) => b, None => { let b = self.goal_sign(self.r, q.asker); self.sign.borrow_mut().insert(q.asker, b); b } };
        after && !before
    }

    /// (units of the asker's side whose standing the leaving can change, whether the goal unit's sign may change): the leaving empties the unit's place and fills its new place.
    fn leaving_candidates(&self, q: &Q, u: &Unit, new: Option<Place>) -> (Vec<Unit>, bool) {
        let (r, med) = (self.r, &self.r.m.medium);
        let origin = u.at;
        let own: Vec<Unit> = r.units.iter().filter(|v| v.side == q.asker && *v != u).copied().collect();
        let foes: Vec<Unit> = r.units.iter().filter(|w| w.side != q.asker).copied().collect();
        let strikes = r.strikes_of(u);
        let mut got: Vec<Unit> = vec![];
        for v in &own { if strikes.contains(v.at) { got.push(*v); } }
        for v in &own {
            if foes.iter().any(|w| med.over(w.kind, r.owner(w.side), w.at, v.at).contains(&origin)) { if !got.contains(v) { got.push(*v); } }
            if let Some(nw) = new {
                if !got.contains(v) && own.iter().any(|g| g != v && r.strikes(g, v.at) && med.over(g.kind, r.owner(g.side), g.at, v.at).contains(&nw)) { got.push(*v); }
            }
        }
        let gu = r.goal_unit(q.asker);
        let mut sign = false;
        if let Some(gu) = gu.filter(|gu| gu != u) {
            let near: Vec<Place> = med.hops(gu.kind, r.owner(gu.side), gu.at).iter().map(|x| x.0).collect();
            let side_u = r.owner(u.side);
            let strikes_near = strikes.iter().any(|x| near.contains(&x)) || new.is_some_and(|nw| near.iter().any(|p| med.strike_places(u.kind, side_u, *p).contains(&nw)));
            let touched: Vec<Place> = std::iter::once(origin).chain(new).collect();
            sign = got.contains(&gu) || strikes_near || touched.iter().any(|x| near.contains(x));
        }
        (got, sign)
    }

    /// What changes for the asker's side when `u` is no longer on its place. Its first step `mv` is played for real from `base` (the other side then to move); when the step is not legal
    /// the board without the unit is used and the other side is given the move by the game's pass. In that position the exchange frame is read for the asker's side's units that the
    /// leaving can change, and the goal unit's sign when it may have changed.
    pub fn cost_of_leaving(&self, q: &Q, u: &Unit, mv: Option<&G::Move>, base: &G::State) -> Rc<Leaving> {
        let (r, g) = (self.r, self.r.g);
        let key = (q.asker, u.at, mv.map(|m| g.move_name(base, m)));
        if let Some(l) = self.duty.borrow().get(&key) { return l.clone(); }
        let child: Option<G::State> = match mv {
            Some(m) => { let c = g.apply(base, m); if g.outcome(&c).is_none() { Some(c) } else { None } }
            None => g.without(base, u.at).and_then(|pos| g.pass(&pos)),
        };
        let out = match child {
            None => Leaving { read: false, duties: vec![] },
            Some(child) => {
                let new = mv.map(|m| g.move_ends(base, m).1);
                let (cands, sign) = self.leaving_candidates(q, u, new);
                let goal = r.goal_unit(q.asker);
                let mut duties = vec![];
                for v in &r.units {
                    if !cands.contains(v) || !self.weighty(v) || Some(*v) == goal { continue; }
                    if let Some(value) = self.lost_after(q, v, &child) { duties.push(Duty { unit: *v, stake: r.weight(v), kind: "newly lost", value }); }
                }
                if let Some(gu) = goal {
                    if gu != *u && sign && self.goal_newly_under_sign(q, &child) { duties.push(Duty { unit: gu, stake: r.weight(&gu), kind: "goal unit newly under the sign", value: 0.0 }); }
                }
                Leaving { read: true, duties }
            }
        };
        let out = Rc::new(out);
        self.duty.borrow_mut().insert(key, out.clone());
        out
    }

    /// The first question about a unit that is to join: what changes if it leaves, read after its first step on the route (or on the board without it when the step is not legal).
    fn duty_reading(&self, q: &Q, way: &Way, route: &Route, base: &G::State) -> Rc<Leaving> {
        let g = self.r.g;
        let u = way.unit;
        let mv = g.find_step(base, u.at, route.places[0]);
        self.cost_of_leaving(q, &u, mv.as_ref(), base)
    }

    // ------------------------------------------------------------------------------------------------------------------------------------------ the follow-up on an obstacle

    /// The follow-up question on one obstacle, by the game's real moves from the base position (the asker to move). An own unit in the way: does it have a move that leaves the route
    /// clear, and what does leaving cost? The other side's unit: will not move for us; can it be taken?
    fn ask_obstacle(&self, q: &Q, route: &Route, ob: &CfObstacle, base: Option<&G::State>) -> Rc<Ask> {
        if ob.dead || ob.unit.is_none() { return Rc::new(dead()); }
        let Some(base) = base else { return Rc::new(Ask { verdict: "not read", cost: 0.0, more: 0, duties: vec![], mv: None }) };
        let key = (q.asker, ob.place, route.places.clone(), ob.kind);
        if let Some(a) = self.obstacle.borrow().get(&key) { return a.clone(); }
        let (r, g) = (self.r, self.r.g);
        let u = ob.unit.unwrap();
        let out = if u.side == q.asker {
            let mut keep: Vec<Place> = route.places.clone();
            for ov in &route.over { keep.extend_from_slice(ov); }
            keep.push(route.unit.at);
            let cands: Vec<G::Move> = g.moves(base).into_iter().filter(|m| { let (f, t) = g.move_ends(base, m); f == ob.place && !keep.contains(&t) }).collect();
            if cands.is_empty() { dead() } else {
                let guarded: Vec<Unit> = r.units.iter().filter(|x| x.side == u.side && **x != u && r.strikes(&u, x.at)).copied().collect();
                let mut best: Option<(f64, G::Move)> = None;
                for m in cands.iter().take(self.p.max_obstacle_moves) {
                    let child = g.apply(base, m);
                    let mut cost = 0.0f64;
                    if g.outcome(&child).is_none() {
                        let mut ps = vec![g.move_ends(base, m).1];
                        ps.extend(guarded.iter().map(|x| x.at));
                        for place in ps {
                            if g.kind_at(&child, place).is_none() { continue; }
                            cost = cost.max(solve_real(r, place, &child, self.p.frame_budget, "given").value);
                        }
                    }
                    if best.as_ref().is_none_or(|b| cost < b.0) { best = Some((cost, m.clone())); }
                    if cost <= 0.0 { break; }
                }
                let (cost, m) = best.unwrap();
                let duties = self.cost_of_leaving(q, &u, Some(&m), base).duties.clone();
                Ask { verdict: "movable", cost, more: 1, duties, mv: Some(g.move_name(base, &m)) }
            }
        } else if Some(u) == r.goal_unit(u.side) {
            dead()
        } else {
            let ex = solve_real(r, ob.place, base, self.p.frame_budget, "given");
            if ex.value > 0.0 { Ask { verdict: "takeable", cost: 0.0, more: 1, duties: vec![], mv: None } } else { dead() }
        };
        let out = Rc::new(out);
        self.obstacle.borrow_mut().insert(key, out.clone());
        out
    }

    // ------------------------------------------------------------------------------------------------------------------------------------------ the verdict of a way

    fn after_entry(&self, q: &Q, way: &Way, routes: &[Route]) -> Entry {
        let v = self.exchange_after(&q.t, q.asker, way);
        let mut e = Entry::from_solved(Some(*way), &v, v.source == "static");
        e.mv = if v.source == "given" { v.line.first().cloned() } else { None };
        e.route = routes.first().cloned();
        e
    }

    /// The verdict of one way at level 2. A clear route of one move is read after its first move. Otherwise every route (the best-ordered first) has its obstacles found by the
    /// counterfactual and asked about, one follow-up question each; a route with a dead obstacle is dead, and the routes left are followed whole by the route follower. A way is
    /// never refuted for what was not read.
    fn settle_inner(&self, q: &mut Q, way: &Way) -> Entry {
        let r = self.r;
        let (t, asker, j) = (q.t, q.asker, way.d);
        let routes: Vec<Route> = q.l1.routes.iter().find(|(w, _)| w == way).map(|(_, v)| v.to_vec()).unwrap_or_default();
        if (j == 1 && (routes.is_empty() || routes.iter().any(|x| x.obstacles.is_empty()))) || (j >= 1 && routes.is_empty()) {
            return self.after_entry(q, way, &routes);
        }
        let base = self.base(asker);
        if j >= 1 {
            if let (Some(b), Some(rt0)) = (&base, routes.first()) {
                if !rt0.places.is_empty() {
                    let early = self.duty_reading(q, way, rt0, b);
                    let mut big: Option<&Duty> = None;
                    for d in &early.duties { if big.is_none_or(|b| d.stake > b.stake) { big = Some(d); } }
                    if let Some(big) = big {
                        if big.stake > q.stake {
                            let mut e = Entry::blank(Some(*way), V::Waiting);
                            e.route = Some(rt0.clone());
                            e.alts = early.duties.iter().map(|d| ("duty", vec![d.unit.at])).collect();
                            e.duties = Some(early.duties.clone());
                            e.held_back = true;
                            e.skipped = true;
                            e.first_obstacles = Some(vec![]);
                            e.tried = Some(0);
                            e.cut = Some(false);
                            e.nodes = Some(0);
                            return e;
                        }
                    }
                }
            }
        }
        let (mut walk, mut dead_rs, mut unread, mut followed): (Vec<Route>, Vec<Route>, Vec<Route>, Vec<(CfObstacle, Ask)>) = (vec![], vec![], vec![], vec![]);
        let mut first_obs: Option<Vec<(Option<Side>, &'static str, bool)>> = None;
        for rt in routes.iter().take(self.p.max_routes) {
            let (obs_list, cut) = match &base {
                Some(b) => counterfactual_obstacles(r, asker, rt, self.p, b, &mut q.blocked),
                None => (rt.obstacles.iter().map(|o| CfObstacle { kind: o.kind, unit: o.unit, place: o.place, hop: o.hop, dead: o.dead, joint: vec![], aside: None }).collect(), false),
            };
            if first_obs.is_none() { first_obs = Some(obs_list.iter().map(|o| (o.unit.map(|u| u.side), o.kind, o.dead)).collect()); }
            let mut answers: Vec<(CfObstacle, Ask)> = vec![];
            for ob in &obs_list {
                if !ob.joint.is_empty() && answers.iter().any(|(o, a)| o.joint == ob.joint && a.verdict == "dead") { continue; }
                let ans = self.ask_obstacle(q, rt, ob, base.as_ref());
                answers.push((ob.clone(), (*ans).clone()));
                followed.push((ob.clone(), (*ans).clone()));
            }
            if cut { answers.push((CfObstacle { kind: "other", unit: None, place: 0, hop: rt.places.len(), dead: false, joint: vec![], aside: None }, Ask { verdict: "not read", cost: 0.0, more: 0, duties: vec![], mv: None })); }
            if answers.iter().any(|(_, a)| a.verdict == "dead") { dead_rs.push(rt.clone()); }
            else if answers.iter().any(|(_, a)| a.verdict == "not read") { unread.push(rt.clone()); }
            else { walk.push(rt.clone()); }
        }
        let mut entry: Entry;
        if j == 0 {                                                          // it bears on the target in place, but a unit stands in the line
            let (verdict, _text) = if !dead_rs.is_empty() { (V::Refuted, ()) } else if !unread.is_empty() { (V::Unsettled, ()) } else { (V::Waiting, ()) };
            let mut e = Entry::blank(Some(*way), verdict);
            e.open = verdict == V::Unsettled;
            e.route = routes.first().cloned();
            if verdict == V::Waiting {
                e.alts = followed.iter().filter_map(|(ob, a)| ob.unit.map(|_| (if a.verdict == "movable" { "vacated" } else { "taken" }, vec![ob.place]))).collect();
            }
            e.obstacles = followed.clone();
            e.nodes = Some(0);
            entry = e;
        } else {
            let v = if walk.is_empty() { None } else { search_way(r, &t, asker, &walk, self.p) };
            if !walk.is_empty() && v.is_none() {                              // the asker not to move and no pass: read after the first move, marked
                let mut e = self.after_entry(q, way, &routes);
                e.mv = None;
                return e;
            }
            let mut v = v;
            let synthetic = v.is_none();
            if v.is_none() {
                if !unread.is_empty() {
                    v = Some(super::walk::WayVerdict { verdict: V::Unsettled, value: 0.0, line: vec![], route: Some(unread[0].clone()), answer: None, alts: vec![], breaks: vec![], nodes: 0, tried: 0, cut: false, routes_tried: 0, stops: vec![] });
                } else {
                    let rt = &dead_rs[0];
                    v = Some(super::walk::WayVerdict { verdict: V::Refuted, value: 0.0, line: vec![], route: Some(rt.clone()), answer: None, alts: vec![], breaks: vec![], nodes: 0, tried: 0, cut: false, routes_tried: 0, stops: vec![] });
                }
            } else if v.as_ref().unwrap().verdict == V::Refuted && !unread.is_empty() {
                v.as_mut().unwrap().verdict = V::Unsettled;
            }
            let v = v.unwrap();
                        let mut e = Entry::blank(Some(*way), v.verdict);
            e.value = v.value;
            e.mv = v.line.first().cloned();
            e.line = v.line.clone();
            e.open = v.verdict == V::Unsettled;
            e.route = v.route.clone();
            e.nodes = Some(v.nodes);
            e.stops = v.stops.clone();
            e.alts = v.alts.clone();
            e.breaks = v.breaks.clone();
            e.answer = v.answer.clone();
            e.routes_tried = Some(v.routes_tried);
            if !synthetic { e.tried = Some(v.tried); e.cut = Some(v.cut); }
            e.obstacles = followed.clone();
            e.dead_routes = Some(dead_rs.len());
            e.unread_routes = Some(unread.len());
            entry = e;
        }
        entry.first_obstacles = Some(first_obs.unwrap_or_default());
        self.refute_counterfactual(q, &mut entry, base.as_ref());
        entry
    }

    /// A refuted route: the refuting answer's unit is the obstacle. Would the route be confirmed without that unit (the route's search rerun on the board without it, the same node cap)?
    fn refute_counterfactual(&self, q: &Q, entry: &mut Entry, base: Option<&G::State>) {
        let (r, g) = (self.r, self.r.g);
        if !(entry.status == V::Refuted && base.is_some() && entry.route.is_some() && !entry.stops.is_empty()) { return; }
        let Some(hard) = entry.stops.iter().find(|x| x.hard && x.mover.is_some()) else { return };
        let place = hard.mover.unwrap().0;
        let Some(u) = r.unit_at(place).map(|i| r.units[i]) else { return };
        if place == q.t.at { return; }
        let Some(pos) = g.without(base.unwrap(), place) else { return };
        let res = Walk::new(r, q.t, q.asker, entry.route.clone().unwrap(), self.p, pos, self.p.route_nodes).run();
        entry.without_stopper = Some((u, res.verdict, res.value));
    }

    /// THE COST OF LEAVING for a confirmed way: the unit that travels may hold something where it stands. The route's value is the target's frame less the largest stake among the
    /// duties dropped; if that is not above 0 the way is WAITING (something else holds the unit).
    fn apply_duties(&self, q: &Q, way: &Way, entry: &mut Entry) {
        if entry.status != V::Confirmed || way.d < 1 { return; }
        let Some(base) = self.base(q.asker) else { return };
        let routes: Vec<Route> = q.l1.routes.iter().find(|(w, _)| w == way).map(|(_, v)| v.to_vec()).unwrap_or_default();
        let route = entry.route.clone().or_else(|| routes.first().cloned());
        let Some(route) = route else { return };
        if route.places.is_empty() { return; }
        let cost = self.duty_reading(q, way, &route, &base);
        entry.duties = Some(cost.duties.clone());
        if cost.duties.is_empty() { return; }
        let mut biggest = &cost.duties[0];
        for d in &cost.duties { if d.stake > biggest.stake { biggest = d; } }
        if entry.value - biggest.stake > 0.0 { return; }
        entry.status = V::Waiting;
        for d in &cost.duties { entry.alts.push(("duty", vec![d.unit.at])); }
        entry.breaks = vec![];
        entry.held_back = true;
    }

    pub fn settle_way(&self, q: &mut Q, way: &Way) -> Rc<Entry> {
        let key = (q.t, q.asker, *way);
        if let Some(e) = self.settled.borrow().get(&key) { return e.clone(); }
        let mut e = self.settle_inner(q, way);
        self.apply_duties(q, way, &mut e);
        let e = Rc::new(e);
        self.settled.borrow_mut().insert(key, e.clone());
        e
    }

    // ------------------------------------------------------------------------------------------------------------------------------------------ the levels

    /// Level 2 of a non-goal unit by the exchange frame: the asker's frame on the target as it stands, then each way that passed the three links, judged by `settle_way`.
    pub fn level_frames(&self, q: &mut Q) -> Finding {
        let l1 = q.l1.clone();
        let rank = |w: &Way| l1.rank.iter().find(|(x, _)| x == w).map_or(0, |x| x.1);
        let mut ways: Vec<Way> = l1.ways.clone();
        ways.sort_by_key(|w| (w.d, rank(w), w.unit.kind, w.unit.at, w.arrival));
        ways.truncate(self.p.max_ways);
        let stands = self.exchange_stands(&q.t, q.asker);
        let mut results: Vec<Entry> = vec![Entry::from_solved(None, &stands, true)];
        for way in &ways {
            if way.d == 0 && !l1.routes.iter().any(|(w, _)| w == way) { continue; }
            results.push((*self.settle_way(q, way)).clone());
        }
        let (conf, wait, unset): (Vec<&Entry>, Vec<&Entry>, Vec<&Entry>) = (
            results.iter().filter(|x| x.status == V::Confirmed).collect(), results.iter().filter(|x| x.status == V::Waiting).collect(), results.iter().filter(|x| x.status == V::Unsettled).collect());
        let mut sources: Vec<&'static str> = results.iter().map(|x| x.source).collect();
        sources.sort();
        sources.dedup();
        let lost = stands.value > 0.0;
        let mut d = FrameDetail { confirmed: conf.len(), refuted: results.len() - conf.len() - wait.len() - unset.len(), waiting: wait.len(), open: unset.len(), lost_where_it_stands: lost, cornering: None,
                                  value: None, line: None, sources, results: vec![] };
        if lost { d.cornering = Some(self.cornering(&q.t, q.asker)); }
        let status = if !conf.is_empty() {
            let mut best = conf[0];
            for x in &conf { if x.value > best.value { best = x; } }
            d.value = Some(best.value);
            d.line = Some(best.line.clone());
            Status::Live
        } else if !wait.is_empty() || !unset.is_empty() { Status::Idle } else { Status::Closed };
        d.results = results;
        Finding { status, detail: Detail::Frame(d) }
    }

    /// Play `mv` and read the child by the reasoning search at `depth`: the asker's value over the line = what the move takes + a threat nobody can parry - the best reply.
    fn settle_move(&self, mv: &G::Move, depth: usize) -> Rc<SearchRes> {
        let (r, g) = (self.r, self.r.g);
        let name = g.move_name(&r.s, mv);
        let key = (name.clone(), depth);
        if let Some(s) = self.searched.borrow().get(&key) { return s.clone(); }
        let child = g.apply(&r.s, mv);
        let out = if g.outcome(&child).is_some() {
            SearchRes { mv: Some(name.clone()), status: "confirmed", value: BIG, reply: None }
        } else {
            let judge = Worth { name: String::new(), table: r.m.ledger_worth.clone(), ours_first: false };
            let rep = reason::reason(g, &child, &judge, depth as i32, self.p.reason_fuse);
            let gain = self.gain(&child);
            let mut replies: Vec<(i64, &String)> = vec![];
            for a in &rep.answers { if let Some(l) = a.line.first() { replies.push((a.net, l)); } }
            for o in &rep.ours { if let Some(l) = o.line.first() { replies.push((o.value, l)); } }
            let mut top: (i64, Option<&String>) = (0, None);
            for (i, rp) in replies.iter().enumerate() { if i == 0 || rp.0 > top.0 { top = (rp.0, Some(rp.1)); } }
            let bonus = if !rep.threats.is_empty() && rep.answers.is_empty() { rep.threats.iter().map(|t| t.stake).max().unwrap_or(0) } else { 0 };
            let value = gain + bonus - top.0.max(0);
            let reply = if top.0 > 0 || rep.answers.is_empty() { top.1.cloned() } else { rep.answers[0].line.first().cloned() };
            if value > 0 { SearchRes { mv: Some(name.clone()), status: "confirmed", value: value as f64, reply: None } }
            else if !rep.threats.is_empty() || gain > 0 { SearchRes { mv: Some(name.clone()), status: "refuted", value: value as f64, reply } }
            else { SearchRes { mv: Some(name.clone()), status: "open", value: 0.0, reply: None } }
        };
        let out = Rc::new(out);
        self.searched.borrow_mut().insert(key, out.clone());
        out
    }

    /// What the move into `child` takes from the other side, in worth: the units it has fewer of than the position before.
    fn gain(&self, child: &G::State) -> i64 {
        let (r, g) = (self.r, self.r.g);
        let nk = r.m.names.kinds.len();
        let count = |s: &G::State| { let mut c = vec![0i64; nk]; for (o, k, _) in g.cells(s) { if o != r.mover { c[k as usize] += 1; } } c };
        let (before, after) = (count(&r.s), count(child));
        (0..nk).filter(|k| before[*k] > after[*k]).map(|k| (r.m.worth[k] as i64) * (before[k] - after[k])).sum()
    }

    /// Level 2 and on of the goal unit asked by the side to move: the first move of each way is played and the child read by the reasoning search.
    pub fn level_search(&self, q: &Q, level: usize) -> Finding {
        let (r, g) = (self.r, self.r.g);
        let depth = level.min(self.p.search_depth_cap);
        let mut ways: Vec<Way> = q.l1.ways.clone();
        ways.sort_by_key(|w| (w.d, w.unit.kind, w.unit.at, w.arrival));
        ways.truncate(self.p.max_ways);
        let mut results: Vec<SearchRes> = vec![];
        for way in &ways {
            let mut mv: Option<G::Move> = None;
            for (o, d) in self.first_moves(&q.t, way) {
                if let Some(m) = g.find_step(&r.s, o, d) { mv = Some(m); break; }
            }
            results.push(match mv { Some(m) => (*self.settle_move(&m, depth)).clone(), None => SearchRes { mv: None, status: "open", value: 0.0, reply: None } });
        }
        let conf: Vec<&SearchRes> = results.iter().filter(|x| x.status == "confirmed").collect();
        let refu = results.iter().filter(|x| x.status == "refuted").count();
        let mut d = SearchDetail { depth, ways: ways.clone(), confirmed: conf.len(), refuted: refu, open: results.len() - conf.len() - refu, value: None, line: None, results: vec![] };
        let status = if results.is_empty() { Status::Idle } else if !conf.is_empty() {
            let mut best = conf[0];
            for x in &conf { if x.value > best.value { best = x; } }
            d.value = Some(best.value);
            d.line = best.mv.clone();
            Status::Live
        } else if refu == results.len() { Status::Closed } else { Status::Idle };
        d.results = results;
        Finding { status, detail: Detail::Search(d) }
    }

    fn tactics(&self, side: Side, depth: usize) -> Rc<Vec<Tactic>> {
        if let Some(t) = self.census.borrow().get(&(side, depth)) { return t.clone(); }
        let (r, g) = (self.r, self.r.g);
        let state = if side == Side::Us { Some(r.s.clone()) } else { r.passed_state() };
        let tactics = match state {
            Some(s) => {
                let judge = Worth { name: String::new(), table: r.m.ledger_worth.clone(), ours_first: false };
                census::census(g, &s, &judge, 1, depth as i32, self.p.census_fuse).tactics
            }
            None => vec![],
        };
        let t = Rc::new(tactics);
        self.census.borrow_mut().insert((side, depth), t.clone());
        t
    }

    /// Level 2 and on of the goal unit asked by the other side: the exact tactics census of the asking side at depth = level, kept to the lines whose targets include this unit.
    pub fn level_census(&self, q: &Q, level: usize) -> Finding {
        let r = self.r;
        let all = self.tactics(q.asker, level);
        let name = format!("{}@{}", r.m.letters[q.t.kind as usize], r.m.names.place_text(q.t.at));
        let tactics: Vec<&Tactic> = all.iter().filter(|x| (q.goal && x.targets.iter().any(|t| t == "goal")) || x.targets.iter().any(|t| *t == name)).collect();
        if tactics.is_empty() {
            let status = if level >= self.p.max_level { Status::Closed } else { Status::Idle };
            return Finding { status, detail: Detail::Census(CensusDetail { depth: level, tactics: 0, value: None, line: None }) };
        }
        let mut best = tactics[0];
        for x in &tactics { if (x.value, std::cmp::Reverse(x.line.len())) > (best.value, std::cmp::Reverse(best.line.len())) { best = x; } }
        Finding { status: Status::Live, detail: Detail::Census(CensusDetail { depth: level, tactics: tactics.len(), value: Some(best.value), line: Some(best.line.clone()) }) }
    }

    /// One deeper step of a question at `level` (2 and on): a non-goal unit by the exchange frame; the goal unit by the search when the asker is to move, else by the census.
    pub fn deep(&self, q: &mut Q, level: usize) -> Finding {
        if !q.goal { self.level_frames(q) } else if q.asker == Side::Us { self.level_search(q, level) } else { self.level_census(q, level) }
    }
}

