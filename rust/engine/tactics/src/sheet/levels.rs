//! LEVELS 0 AND 1 OF THE QUESTION SHEET (the Python sheet's `level0`, `_level1`, `_level1_geometry`, `geometric_routes`, `_stand`, `_link_check`), on the light reading.
//! A question is (target unit, the asking side = the side that does not own the target). Level 0: the board as it stands -- what bears on the target, what guards it, how many of its
//! exits are open (the goal unit by the goal report). Level 1: the asking side's moves alone -- the goal unit by the report's scenarios; any other unit by the medium of movement: for
//! each unit of the asker, the places that would bear on the target (units ignored) and the hops to each, every shortest route with the obstacles on it as the board stands, and the
//! three links (reach, a place to stand, a strike that gains). The reasons a link fails are DATA (`Why`), made into words only at the boundary.
use super::reading::Reading;
use super::{FxMap, FxSet, Params, Place, PlaceSet, Side, Status, Unit};
use cs_game::placed::Placed;
use cs_game::Game;
use std::collections::BTreeSet;

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
    let guards: Vec<Unit> = r.units.iter().filter(|u| u.side == t.side && *u != t && r.strikes(u, t.at)).copied().collect();
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
    let cheapest = strikers.iter().map(|u| r.worth_of(u.kind)).fold(None, |m: Option<f64>, w| Some(m.map_or(w, |x| x.min(w))));
    let live = !strikers.is_empty() && (is_goal || guards.is_empty() || cheapest.unwrap() < r.worth_of(t.kind));
    Level0 { status: if live { Status::Live } else { Status::Idle }, strikers, guards, open_exits, exits: exits.len(), goal: None }
}

// ---------------------------------------------------------------------------------------------------------------------------------------------- the pieces of level 1

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum What { Strike, Cover }

impl What {
    pub fn word(self) -> &'static str { match self { What::Strike => "strike", What::Cover => "cover an exit" } }
}

/// A way: the unit, the place it would stand on, the hops it takes, and what it does there.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Way { pub d: u32, pub unit: Unit, pub arrival: Place, pub what: What }

/// Anything whose absence would let a step happen (the Python sheet's `Obstacle`): the describing word, the unit, its place, the number of the step, and whether no unit accounts for it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Obstacle { pub kind: &'static str, pub unit: Option<Unit>, pub place: Place, pub hop: usize, pub dead: bool }

/// Why a link fails, as data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Why {
    /// `u` would stand on `to`, struck by `by`, which is worth less
    Cheaper { u: Unit, to: Place, by: Unit },
    /// `u` would stand on `to`, struck by `by` and not guarded
    Unguarded { u: Unit, to: Place, by: Unit },
    /// `u` would only cover an exit of `t` and nothing strikes it yet
    OnlyCover { u: Unit, t: Unit },
    /// `t` is guarded by `guard` and is worth no more than `u`
    Guarded { t: Unit, guard: Unit, u: Unit },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Route {
    pub unit: Unit,
    pub places: Vec<Place>,
    pub stands: Vec<Option<Why>>,
    pub what: What,
    pub over: Vec<Vec<Place>>,
    pub obstacles: Vec<Obstacle>,
    pub goal: Place,
}

/// None when a unit `u` of the asking side could stand on `to` (the target's side does not bear on it with a cheaper unit, nor bear on it while nothing of the asking side
/// guards it); else why not.
pub fn stand<G: Game + Placed>(r: &Reading<G>, t: &Unit, u: &Unit, to: Place) -> Option<Why> {
    let bearers = r.bearers_of(to);
    let wu = r.weight(u);
    let mut first_hit: Option<Unit> = None;
    for i in bearers {
        let v = &r.units[*i];
        if v.side != t.side { continue; }
        if r.weight(v) < wu { return Some(Why::Cheaper { u: *u, to, by: *v }); }
        if first_hit.is_none() { first_hit = Some(*v); }
    }
    if let Some(h) = first_hit {
        let backed = bearers.iter().map(|i| &r.units[*i]).any(|w| w.side == u.side && w != u);
        if !backed { return Some(Why::Unguarded { u: *u, to, by: h }); }
    }
    None
}

/// (the link that fails, why), or None when the three links hold, for a unit `u` of the asking side that would stand on `to` and bear on the target (what = strike) or cover one of
/// its exits.
pub fn link_test<G: Game + Placed>(r: &Reading<G>, t: &Unit, u: &Unit, to: Place, moved: bool, what: What, struck_now: bool) -> Option<(u8, Why)> {
    if moved {
        if let Some(w) = stand(r, t, u, to) { return Some((2, w)); }
    }
    if what != What::Strike && !struck_now { return Some((3, Why::OnlyCover { u: *u, t: *t })); }
    let guard = r.units.iter().find(|v| v.side == t.side && *v != t && r.strikes(v, t.at));
    if let Some(g) = guard {
        if r.weight(t) <= r.weight(u) { return Some((3, Why::Guarded { t: *t, guard: *g, u: *u })); }
    }
    None
}

/// The units standing on places, but for the travelling unit.
struct Standing<'x, 'a, G: Game + Placed> { r: &'x Reading<'a, G>, u: &'x Unit }

impl<G: Game + Placed> Standing<'_, '_, G> {
    fn get(&self, place: Place) -> Option<Unit> { self.r.unit_at(place).map(|i| self.r.units[i]).filter(|v| v != self.u) }
}

/// The shortest geometric routes of `u` to `arrival` (the medium, units ignored), each with its obstacles as the board stands and the place-to-stand reading of every landing; the
/// routes with the fewest obstacles first.
pub fn geometric_routes<G: Game + Placed>(r: &Reading<G>, t: &Unit, u: &Unit, arrival: Place, goal_place: Place, what: What) -> Vec<Route> {
    let med = &r.m.medium;
    let (kind, role, a) = (u.kind, r.owner(u.side), u.at);
    let at = Standing { r, u };
    let takes = med.takes(kind, role);
    let one_path = Rc::new(vec![vec![]]);
    let paths = if arrival != a { med.routes(kind, role, a, arrival, 200) } else { one_path };
    let mut out: Vec<Route> = Vec::with_capacity(paths.len());
    for path in paths.iter() {
        let (mut obs, mut over, mut prev): (Vec<Obstacle>, Vec<Vec<Place>>, Place) = (vec![], Vec::with_capacity(path.len()), a);
        for (i, landing) in path.iter().enumerate() {
            let ov = med.over(kind, role, prev, *landing);
            for x in ov { if let Some(v) = at.get(*x) { obs.push(Obstacle { kind: "stands on a place passed over", unit: Some(v), place: *x, hop: i, dead: false }); } }
            over.push(ov.to_vec());
            let take_only = med.hop_take(kind, role, prev, *landing).unwrap_or(false);
            match at.get(*landing) {
                None => { if take_only { obs.push(Obstacle { kind: "no unit to take there", unit: None, place: *landing, hop: i, dead: true }); } }
                Some(v) if v.side == u.side => obs.push(Obstacle { kind: "stands on the landing place", unit: Some(v), place: *landing, hop: i, dead: false }),
                Some(v) if takes && !take_only => obs.push(Obstacle { kind: "stands on the landing place and this step may not take it", unit: Some(v), place: *landing, hop: i, dead: true }),
                Some(v) => obs.push(Obstacle { kind: "stands on the landing place", unit: Some(v), place: *landing, hop: i, dead: false }),
            }
            prev = *landing;
        }
        if !takes {
            for x in med.over(kind, role, arrival, goal_place) {
                if let Some(v) = at.get(*x) { obs.push(Obstacle { kind: "stands on a place passed over", unit: Some(v), place: *x, hop: path.len(), dead: false }); }
            }
        }
        let stands: Vec<Option<Why>> = path.iter().map(|p| stand(r, t, u, *p)).collect();
        out.push(Route { unit: *u, places: path.clone(), stands, what, over, obstacles: obs, goal: goal_place });
    }
    out.sort_by(|x, y| {
        let key = |z: &Route| (z.obstacles.iter().filter(|o| o.dead).count(), z.obstacles.len(), z.stands.iter().filter(|s| s.is_some()).count());
        key(x).cmp(&key(y)).then_with(|| x.places.cmp(&y.places))
    });
    out
}

use std::rc::Rc;

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
    pub routes: Vec<(Way, Rc<Vec<Route>>)>,
    /// (the link that failed, why, what) of every way that failed a link.
    pub failed: Vec<(u8, Why, What)>,
    pub failed_at: Option<u8>,
    /// [ways that reach, ways past the first two links, ways that pass all three]
    pub links: Option<[usize; 3]>,
    pub reach: Option<usize>,
    pub reach_geo: BTreeSet<(super::Kind, Place)>,
    /// (hops of its nearest strike place, unit) of the asker's units that are too far in time.
    pub far: Vec<(u32, Unit)>,
    pub too_far: Option<u32>,
    /// Some for the goal unit read by the report.
    pub goal: Option<GoalHeld>,
    /// The place of each way in the whole supply (the Python detail's `rank`): the order the deeper levels take the ways in.
    pub rank: Vec<(Way, usize)>,
}

impl Level1 {
    fn blank(status: Status) -> Level1 {
        Level1 { status, ways: vec![], routes: vec![], failed: vec![], failed_at: None, links: None, reach: None, reach_geo: BTreeSet::new(), far: vec![], too_far: None, goal: None, rank: vec![] }
    }
}

/// At least one unit of the asking side bears on the target's place or one of its exits now, or can come to a place that does within `k` moves.
pub fn asker_near<G: Game + Placed>(r: &Reading<G>, t: &Unit, asker: Side, k: usize) -> bool {
    let mut goals: PlaceSet = (*r.exits_of(t)).clone();
    goals.insert(t.at);
    for u in r.units.iter().filter(|u| u.side == asker) {
        if r.strikes_of(u).intersects(&goals) { return true; }
        for j in 1..=k.min(r.m.can_go) {
            for to in r.go_from(j, u.kind, u.side, u.at).iter() {
                if r.acts_from(u.kind, u.side, to).intersects(&goals) { return true; }
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
        let mut sorted: Vec<&(u32, Vec<(usize, Place, u32)>)> = rep.scenarios.iter().collect();
        sorted.sort_by_key(|s| s.0);
        let mut ways: Vec<Way> = vec![];
        for sc in sorted.into_iter().take(3) {
            for (i, to, moves) in &sc.1 {
                let from = &rep.units[*i];
                let w = Way { d: *moves, unit: Unit { side: asker, kind: from.kind, at: from.at }, arrival: *to, what: What::Strike };
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
    let med = &r.m.medium;
    let now = r.strikers_of(t, asker);
    let exits = r.exits_of(t);
    let covered = r.struck_by(asker);
    let mut goals: Vec<(Place, What)> = vec![(t.at, What::Strike)];
    if !now.is_empty() { for e in exits.iter().filter(|e| !covered.contains(*e)) { goals.push((e, What::Cover)); } }
    let mut supply: Vec<Way> = now.iter().map(|u| Way { d: 0, unit: *u, arrival: u.at, what: What::Strike }).collect();
    let mut routes: Vec<(Way, Rc<Vec<Route>>)> = vec![];
    let mut route_ix: FxMap<Way, usize> = FxMap::default();
    let mut far: Vec<(u32, Unit)> = vec![];
    let mut reach_geo: BTreeSet<(super::Kind, Place)> = BTreeSet::new();
    let mut seen: FxSet<(Unit, Place, What)> = FxSet::default();
    for u in r.units.iter().filter(|u| u.side == asker && !now.contains(u)) {
        let role = r.owner(u.side);
        let mut nearest: Option<u32> = None;
        for (goal, what) in &goals {
            for place in med.strike_places(u.kind, role, *goal).iter().copied() {
                let Some(d) = med.distance(u.kind, role, u.at, place) else { continue };
                nearest = Some(nearest.map_or(d, |n| n.min(d)));
                if d > k || seen.contains(&(*u, place, *what)) || (*what == What::Cover && seen.contains(&(*u, place, What::Strike))) { continue; }
                seen.insert((*u, place, *what));
                let found = geometric_routes(r, t, u, place, *goal, *what);
                if found.is_empty() { continue; }
                let way = Way { d, unit: *u, arrival: place, what: *what };
                supply.push(way);
                route_ix.insert(way, routes.len());
                routes.push((way, Rc::new(found)));
                reach_geo.insert((u.kind, u.at));
            }
        }
        if let Some(n) = nearest { if n > k { far.push((n, *u)); } }
    }
    far.sort_by_key(|f| (f.0, f.1.kind, f.1.at));
    let obstacles_of = |w: &Way| route_ix.get(w).map_or(0, |i| routes[*i].1[0].obstacles.len());
    supply.sort_by_key(|w| (w.d, obstacles_of(w), w.unit.kind, w.unit.at, w.arrival));
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
    let (mut passed, mut failed): (Vec<Way>, Vec<(u8, Why, What)>) = (vec![], vec![]);
    for w in &supply {
        if w.d == 0 && !route_ix.contains_key(w) { passed.push(*w); continue; }
        match link_test(r, t, &w.unit, w.arrival, w.d > 0, w.what, !now.is_empty()) {
            Some((link, why)) => failed.push((link, why, w.what)),
            None => passed.push(*w),
        }
    }
    out.reach = Some(supply.len());
    out.links = Some([supply.len(), failed.iter().filter(|f| f.0 > 2).count() + passed.len(), passed.len()]);
    out.routes = passed.iter().filter_map(|w| route_ix.get(w).map(|i| (*w, routes[*i].1.clone()))).collect();
    out.rank = supply.iter().enumerate().map(|(n, w)| (*w, n)).collect();
    if passed.is_empty() {
        let strikes: Vec<&(u8, Why, What)> = failed.iter().filter(|f| f.2 == What::Strike).collect();
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

// ---------------------------------------------------------------------------------------------------------------------------------------------- the goal unit's weak places

/// A weak place of the goal unit (the Python sheet's `weak_places`): its own place or a place next to it where the guards are not fewer than the strikers, with the units of the asker that
/// could be brought to strike it.
#[derive(Debug, Clone)]
pub struct WeakPlace {
    pub place: Place,
    pub guards: Vec<Unit>,
    pub strikers: Vec<Unit>,
    /// guards - strikers + 1: the units that must be brought
    pub need: usize,
    /// (hops, unit, the place it would strike from), nearest first
    pub candidates: Vec<(u32, Unit, Place)>,
}

/// The goal unit `t`'s own place and each place next to it (where it could move on the empty board, not held by its own side): the units of its side that guard the place and the units of the
/// asker that bear on it now. Where the guards are not fewer than the strikers, the question is whether `need` more units can be brought within `reach_k` hops.
pub fn weak_places<G: Game + Placed>(r: &Reading<G>, t: &Unit, asker: Side, p: &Params) -> Vec<WeakPlace> {
    let med = &r.m.medium;
    let own: Vec<Place> = r.units.iter().filter(|u| u.side == t.side && *u != t).map(|u| u.at).collect();
    let mut near: Vec<Place> = vec![t.at];
    let mut hops: Vec<Place> = med.hops(t.kind, r.owner(t.side), t.at).iter().map(|x| x.0).collect();
    hops.sort();
    near.extend(hops);
    let mut out = vec![];
    for pl in near {
        if own.contains(&pl) { continue; }
        let guards: Vec<Unit> = r.units.iter().filter(|u| u.side == t.side && *u != t && r.strikes(u, pl)).copied().collect();
        let strikers: Vec<Unit> = r.units.iter().filter(|u| u.side == asker && r.strikes(u, pl)).copied().collect();
        if guards.len() < strikers.len() { continue; }
        let mut cands: Vec<(u32, Unit, Place)> = vec![];
        for u in r.units.iter().filter(|u| u.side == asker && !strikers.contains(u)) {
            let role = r.owner(u.side);
            let mut best: Option<(u32, Place)> = None;
            let mut places: Vec<Place> = med.strike_places(u.kind, role, pl).to_vec();
            places.sort();
            for a in places {
                if let Some(d) = med.distance(u.kind, role, u.at, a) {
                    if d > 0 && d <= p.reach_k && best.is_none_or(|b| d < b.0) { best = Some((d, a)); }
                }
            }
            if let Some((d, a)) = best { cands.push((d, *u, a)); }
        }
        cands.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.kind.cmp(&b.1.kind)).then_with(|| a.1.at.cmp(&b.1.at)));
        out.push(WeakPlace { place: pl, need: guards.len() - strikers.len() + 1, guards, strikers, candidates: cands });
    }
    out
}
