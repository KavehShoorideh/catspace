//! LEVELS 0 AND 1 OF THE QUESTION SHEET (the Python sheet's `level0`, `_level1`, `_level1_geometry`, `geometric_routes`, `_stand`, `_link_check`), on the light reading.
//! A question is (target unit, the asking side = the side that does not own the target). Level 0: the board as it stands -- what bears on the target, what guards it, how many of its
//! exits are open (the goal unit by the goal report). Level 1: the asking side's moves alone -- the goal unit by the report's scenarios; any other unit by the medium of movement: for
//! each unit of the asker, the places that would bear on the target (units ignored) and the hops to each, every shortest route with the obstacles on it as the board stands, and the
//! three links (reach, a place to stand, a strike that gains).
use super::reading::Reading;
use super::{Params, Side, Status, Unit};
use cs_game::placed::Placed;
use cs_game::Game;
use std::collections::{BTreeSet, HashSet};

// ---------------------------------------------------------------------------------------------------------------------------------------------- level 0

#[derive(Debug, Clone)]
pub struct Level0 {
    pub status: Status,
    pub strikers: Vec<Unit>,
    pub guards: Vec<Unit>,
    pub open_exits: usize,
    pub exits: usize,
    /// For the goal unit read by the report: (struck, cornered).
    pub goal: Option<(bool, bool)>,
}

pub fn level0<G: Game + Placed>(r: &Reading<G>, t: &Unit, asker: Side, is_goal: bool) -> Level0 {
    let guards: Vec<Unit> = r.units.iter().filter(|u| u.side == t.side && *u != t && r.strikes_of(u).contains(&t.at)).cloned().collect();
    if is_goal {
        let rep = r.scenario(asker);
        if rep.readable() {
            let strikers: Vec<Unit> = rep.covered.iter().find(|c| c.own).map(|c| c.by.clone()).unwrap_or_default();
            let open_exits = rep.holes.iter().filter(|h| h.1 == "exit").count();
            let total = open_exits + rep.covered.iter().filter(|c| !c.own).count();
            let live = !strikers.is_empty();
            let goal = Some((live, live && open_exits == 0));
            return Level0 { status: if live { Status::Live } else { Status::Idle }, strikers, guards, open_exits, exits: total, goal };
        }
    }
    let strikers = r.strikers_of(t, asker);
    let exits = r.exits_of(t);
    let covered = r.struck_by(asker);
    let open_exits = exits.iter().filter(|e| !covered.contains(*e)).count();
    let cheapest = strikers.iter().map(|u| r.worth_of(&u.kind)).fold(None, |m: Option<f64>, w| Some(m.map_or(w, |x| x.min(w))));
    let live = !strikers.is_empty() && (is_goal || guards.is_empty() || cheapest.unwrap() < r.worth_of(&t.kind));
    Level0 { status: if live { Status::Live } else { Status::Idle }, strikers, guards, open_exits, exits: exits.len(), goal: None }
}

// ---------------------------------------------------------------------------------------------------------------------------------------------- the pieces of level 1

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum What { Strike, Cover }

impl What {
    pub fn word(self) -> &'static str { match self { What::Strike => "strike", What::Cover => "cover an exit" } }
}

/// A way: the unit, the place it would stand on, the hops it takes, and what it does there.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Way { pub d: u32, pub unit: Unit, pub arrival: String, pub what: What }

/// Anything whose absence would let a step happen (the Python sheet's `Obstacle`): the describing word, the unit, its place, the number of the step, and whether no unit accounts for it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Obstacle { pub kind: &'static str, pub unit: Option<Unit>, pub place: String, pub hop: usize, pub dead: bool }

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Route {
    pub unit: Unit,
    pub places: Vec<String>,
    pub stands: Vec<(bool, String)>,
    pub what: What,
    pub over: Vec<Vec<String>>,
    pub obstacles: Vec<Obstacle>,
    pub goal: String,
}

/// (true, "") when a unit `u` of the asking side could stand on `to` (the target's side does not bear on it with a cheaper unit, nor bear on it while nothing of the asking side
/// guards it); else (false, why).
pub fn stand<G: Game + Placed>(r: &Reading<G>, t: &Unit, u: &Unit, to: &str) -> (bool, String) {
    let hit: Vec<&Unit> = r.bearers_of(to).iter().map(|i| &r.units[*i]).filter(|v| v.side == t.side).collect();
    let cheaper = hit.iter().find(|v| r.weight(v) < r.weight(u));
    let backed = r.bearers_of(to).iter().map(|i| &r.units[*i]).any(|w| w.side == u.side && w != u);
    if let Some(c) = cheaper { return (false, format!("{} would stand on {}, struck by {}, which is worth less", u.words(), to, c.words())); }
    if !hit.is_empty() && !backed { return (false, format!("{} would stand on {}, struck by {} and not guarded", u.words(), to, hit[0].words())); }
    (true, String::new())
}

/// (the link that fails, or 0; why) for a unit `u` of the asking side that would stand on `to` and bear on the target (what = strike) or cover one of its exits.
pub fn link_test<G: Game + Placed>(r: &Reading<G>, t: &Unit, u: &Unit, to: &str, moved: bool, what: What, struck_now: bool) -> (u8, String) {
    if moved {
        let (ok, why) = stand(r, t, u, to);
        if !ok { return (2, why); }
    }
    if what != What::Strike && !struck_now {
        return (3, format!("{} would only cover an exit of {} and nothing strikes it yet", u.words(), t.words()));
    }
    let guard = r.units.iter().find(|v| v.side == t.side && *v != t && r.strikes_of(v).contains(&t.at));
    if let Some(g) = guard {
        if r.weight(t) <= r.weight(u) { return (3, format!("{} is guarded by {} and is worth no more than {}", t.words(), g.words(), u.words())); }
    }
    (0, String::new())
}

/// The units standing on places, but for the travelling unit.
struct Standing<'x, 'a, G: Game + Placed> { r: &'x Reading<'a, G>, u: &'x Unit }

impl<'x, 'a, G: Game + Placed> Standing<'x, 'a, G> {
    fn get(&self, place: &str) -> Option<&'x Unit> { self.r.unit_at(place).map(|i| &self.r.units[i]).filter(|v| *v != self.u) }
}

/// The shortest geometric routes of `u` to `arrival` (the medium, units ignored), each with its obstacles as the board stands and the place-to-stand reading of every landing; the
/// routes with the fewest obstacles first.
pub fn geometric_routes<G: Game + Placed>(r: &Reading<G>, t: &Unit, u: &Unit, arrival: &str, goal_place: &str, what: What) -> Vec<Route> {
    let med = r.medium.expect("the medium of movement");
    let (kind, side, a) = (u.kind.as_str(), r.name(u.side), u.at.as_str());
    let at = Standing { r, u };
    let takes = med.takes(kind, side);
    let paths: Vec<Vec<String>> = if arrival != a { med.routes(kind, side, a, arrival, 200) } else { vec![vec![]] };
    let mut out: Vec<Route> = vec![];
    for path in paths {
        let (mut obs, mut over, mut prev): (Vec<Obstacle>, Vec<Vec<String>>, String) = (vec![], vec![], a.to_string());
        for (i, landing) in path.iter().enumerate() {
            let ov = med.over(kind, side, &prev, landing);
            for x in &ov { if let Some(v) = at.get(&x) { obs.push(Obstacle { kind: "stands on a place passed over", unit: Some((*v).clone()), place: x.clone(), hop: i, dead: false }); } }
            over.push(ov);
            let take_only = med.hops(kind, side, &prev).get(landing).copied().unwrap_or(false);
            match at.get(landing) {
                None => { if take_only { obs.push(Obstacle { kind: "no unit to take there", unit: None, place: landing.clone(), hop: i, dead: true }); } }
                Some(v) if v.side == u.side => obs.push(Obstacle { kind: "stands on the landing place", unit: Some((*v).clone()), place: landing.clone(), hop: i, dead: false }),
                Some(v) if takes && !take_only => obs.push(Obstacle { kind: "stands on the landing place and this step may not take it", unit: Some((*v).clone()), place: landing.clone(), hop: i, dead: true }),
                Some(v) => obs.push(Obstacle { kind: "stands on the landing place", unit: Some((*v).clone()), place: landing.clone(), hop: i, dead: false }),
            }
            prev = landing.clone();
        }
        if !takes {
            for x in med.over(kind, side, arrival, goal_place) {
                if let Some(v) = at.get(&x) { obs.push(Obstacle { kind: "stands on a place passed over", unit: Some((*v).clone()), place: x.clone(), hop: path.len(), dead: false }); }
            }
        }
        let stands: Vec<(bool, String)> = path.iter().map(|p| stand(r, t, u, p)).collect();
        out.push(Route { unit: u.clone(), places: path, stands, what, over, obstacles: obs, goal: goal_place.to_string() });
    }
    out.sort_by(|x, y| {
        let key = |z: &Route| (z.obstacles.iter().filter(|o| o.dead).count(), z.obstacles.len(), z.stands.iter().filter(|s| !s.0).count());
        key(x).cmp(&key(y)).then_with(|| x.places.cmp(&y.places))
    });
    out
}

// ---------------------------------------------------------------------------------------------------------------------------------------------- level 1

#[derive(Debug, Clone, Default)]
pub struct GoalHeld {
    pub open_exits: usize,
    pub near: bool,
    pub scenarios: usize,
    pub moves: Option<u32>,
}

#[derive(Debug, Clone)]
pub struct Level1 {
    pub status: Status,
    /// The ways the level left standing (the Python detail's `supply`).
    pub ways: Vec<Way>,
    pub routes: Vec<(Way, Vec<Route>)>,
    /// (the link that failed, why, what) of every way that failed a link.
    pub failed: Vec<(u8, String, What)>,
    pub failed_at: Option<u8>,
    /// [ways that reach, ways past the first two links, ways that pass all three]
    pub links: Option<[usize; 3]>,
    pub reach: Option<usize>,
    pub reach_geo: BTreeSet<(String, String)>,
    /// (hops of its nearest strike place, unit) of the asker's units that are too far in time.
    pub far: Vec<(u32, Unit)>,
    pub too_far: Option<u32>,
    /// Some for the goal unit read by the report.
    pub goal: Option<GoalHeld>,
}

impl Level1 {
    fn blank(status: Status) -> Level1 {
        Level1 { status, ways: vec![], routes: vec![], failed: vec![], failed_at: None, links: None, reach: None, reach_geo: BTreeSet::new(), far: vec![], too_far: None, goal: None }
    }
}

/// At least one unit of the asking side bears on the target's place or one of its exits now, or can come to a place that does within `k` moves.
fn asker_near<G: Game + Placed>(r: &Reading<G>, t: &Unit, asker: Side, k: usize) -> bool {
    let mut goals: BTreeSet<String> = (*r.exits_of(t)).clone();
    goals.insert(t.at.clone());
    for u in r.units.iter().filter(|u| u.side == asker) {
        if r.strikes_of(u).iter().any(|p| goals.contains(p)) { return true; }
        for j in 1..=k.min(r.can_go) {
            for to in r.go_from(j, &u.kind, u.side, &u.at).iter() {
                if r.acts_from(&u.kind, u.side, to).iter().any(|p| goals.contains(p)) { return true; }
            }
        }
    }
    false
}

fn goal_level1<G: Game + Placed>(r: &Reading<G>, t: &Unit, asker: Side, p: &Params) -> Option<Level1> {
    let rep = r.scenario(asker);
    if !rep.readable() { return None; }
    let k = p.reach_k as usize;
    let open_exits = rep.holes.iter().filter(|h| h.1 == "exit").count();
    let mut out = Level1::blank(Status::Idle);
    if rep.safe == Some(true) {
        out.goal = Some(GoalHeld { open_exits, near: false, scenarios: 0, moves: None });
        if open_exits <= p.max_open_exits { return Some(out); }
        out.status = Status::Closed;
        out.failed_at = Some(1);
        return Some(out);
    }
    let near = asker_near(r, t, asker, k);
    let mut held = GoalHeld { open_exits, near, scenarios: rep.scenarios.len(), moves: None };
    if !rep.scenarios.is_empty() {
        let best = rep.scenarios.iter().map(|s| s.0).min().unwrap();
        let mut sorted: Vec<&(u32, Vec<(usize, String, u32)>)> = rep.scenarios.iter().collect();
        sorted.sort_by_key(|s| s.0);
        let mut ways: Vec<Way> = vec![];
        for sc in sorted.into_iter().take(3) {
            for (i, to, moves) in &sc.1 {
                let from = &rep.units[*i];
                let w = Way { d: *moves, unit: Unit { side: asker, kind: from.kind.clone(), at: from.at.clone() }, arrival: to.clone(), what: What::Strike };
                if !ways.contains(&w) && *moves > 0 { ways.push(w); }
            }
        }
        held.moves = Some(best);
        out.ways = ways;
        out.status = if open_exits <= p.max_open_exits && near { Status::Live } else { Status::Idle };
        out.goal = Some(held);
        return Some(out);
    }
    out.goal = Some(held);
    if open_exits <= p.max_open_exits {
        out.status = if near { Status::Live } else { Status::Idle };
        return Some(out);
    }
    out.status = Status::Closed;
    out.failed_at = Some(3);
    Some(out)
}

pub fn level1<G: Game + Placed>(r: &Reading<G>, t: &Unit, asker: Side, is_goal: bool, p: &Params) -> Level1 {
    if is_goal { if let Some(l) = goal_level1(r, t, asker, p) { return l; } }
    geometry(r, t, asker, p)
}

fn geometry<G: Game + Placed>(r: &Reading<G>, t: &Unit, asker: Side, p: &Params) -> Level1 {
    let k = p.reach_k;
    let med = r.medium.expect("the medium of movement");
    let now = r.strikers_of(t, asker);
    let exits = r.exits_of(t);
    let covered = r.struck_by(asker);
    let open_exits: Vec<&String> = exits.iter().filter(|e| !covered.contains(*e)).collect();
    let mut goals: Vec<(String, What)> = vec![(t.at.clone(), What::Strike)];
    if !now.is_empty() { for e in &open_exits { goals.push(((*e).clone(), What::Cover)); } }
    let mut supply: Vec<Way> = now.iter().map(|u| Way { d: 0, unit: u.clone(), arrival: u.at.clone(), what: What::Strike }).collect();
    let mut routes: Vec<(Way, Vec<Route>)> = vec![];
    let mut far: Vec<(u32, Unit)> = vec![];
    let mut reach_geo: BTreeSet<(String, String)> = BTreeSet::new();
    let mut seen: HashSet<(Unit, String, What)> = HashSet::new();
    for u in r.units.iter().filter(|u| u.side == asker && !now.contains(u)) {
        let side_abs = r.name(u.side);
        let mut nearest: Option<u32> = None;
        for (goal, what) in &goals {
            for place in med.strike_places(&u.kind, side_abs, goal) {
                let Some(d) = med.distance(&u.kind, side_abs, &u.at, &place) else { continue };
                nearest = Some(nearest.map_or(d, |n| n.min(d)));
                if d > k || seen.contains(&(u.clone(), place.clone(), *what)) || (*what == What::Cover && seen.contains(&(u.clone(), place.clone(), What::Strike))) { continue; }
                seen.insert((u.clone(), place.clone(), *what));
                let found = geometric_routes(r, t, u, &place, goal, *what);
                if found.is_empty() { continue; }
                let way = Way { d, unit: u.clone(), arrival: place.clone(), what: *what };
                supply.push(way.clone());
                routes.push((way, found));
                reach_geo.insert((u.kind.clone(), u.at.clone()));
            }
        }
        if let Some(n) = nearest { if n > k { far.push((n, u.clone())); } }
    }
    far.sort_by(|a, b| (a.0, &a.1.kind, &a.1.at).cmp(&(b.0, &b.1.kind, &b.1.at)));
    let route_of = |w: &Way| routes.iter().find(|(x, _)| x == w).map(|(_, v)| v);
    let obstacles_of = |w: &Way| route_of(w).map_or(0, |v| v[0].obstacles.len());
    supply.sort_by(|a, b| (a.d, obstacles_of(a), &a.unit.kind, &a.unit.at, &a.arrival).cmp(&(b.d, obstacles_of(b), &b.unit.kind, &b.unit.at, &b.arrival)));
    let mut out = Level1::blank(Status::Closed);
    out.reach_geo = reach_geo;
    out.far = far;
    if supply.len() == now.len() {
        if !now.is_empty() {
            out.status = Status::Idle;
            out.ways = supply;
            return out;
        }
        if let Some((hops, _)) = out.far.first() { out.too_far = Some(*hops); }
        out.failed_at = Some(1);
        return out;
    }
    let (mut passed, mut failed): (Vec<Way>, Vec<(u8, String, What)>) = (vec![], vec![]);
    for w in &supply {
        if w.d == 0 && route_of(w).is_none() { passed.push(w.clone()); continue; }
        let (link, why) = link_test(r, t, &w.unit, &w.arrival, w.d > 0, w.what, !now.is_empty());
        if link > 0 { failed.push((link, why, w.what)); } else { passed.push(w.clone()); }
    }
    out.reach = Some(supply.len());
    out.links = Some([supply.len(), failed.iter().filter(|f| f.0 > 2).count() + passed.len(), passed.len()]);
    out.routes = passed.iter().filter_map(|w| route_of(w).map(|v| (w.clone(), v.clone()))).collect();
    if passed.is_empty() {
        let strikes: Vec<&(u8, String, What)> = failed.iter().filter(|f| f.2 == What::Strike).collect();
        let top = if strikes.is_empty() { failed.iter().map(|f| f.0).max() } else { strikes.iter().map(|f| f.0).max() };
        out.failed_at = top;
        out.failed = failed;
        return out;
    }
    out.status = Status::Live;
    out.ways = passed;
    out.failed = failed;
    out
}

// ---------------------------------------------------------------------------------------------------------------------------------------------- the whole reading of one position

/// Everything stage A reads of one position: the questions in their first order (target, stake, is the goal unit, rank), each with its level 0 and level 1.
pub struct Sheet {
    pub order: Vec<(Unit, f64, bool, usize)>,
    pub level0: Vec<Level0>,
    pub level1: Vec<Level1>,
}

pub fn read<G: Game + Placed>(r: &Reading<G>, p: &Params) -> Sheet {
    let order = r.first_order(p);
    let (mut l0, mut l1) = (vec![], vec![]);
    for (t, _stake, goal, _rank) in &order {
        let asker = t.side.other();
        l0.push(level0(r, t, asker, *goal));
        l1.push(level1(r, t, asker, *goal, p));
    }
    Sheet { order, level0: l0, level1: l1 }
}
