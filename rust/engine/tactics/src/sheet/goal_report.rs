//! THE GOAL UNIT'S REPORT from the light reading (the Python sheet's `scenario_light`): the holes (the goal unit's own place and the places it could move to that no unit of the
//! asking side bears on), the covered places and who covers them, the supply (which unit could come to bear on a hole, from where, in how many moves) and the scenarios found by
//! the same backtracking as the kernel's report (`crate::scenario`). The board is read as it stands. Same caps.
use super::reading::Reading;
use super::{Place, PlaceSet, Side, Unit};
use cs_game::placed::Placed;
use cs_game::Game;
use std::collections::BTreeSet;

pub const MAX_HOLES: usize = 8;
pub const MAX_UNITS: usize = 16;
pub const MAX_SCENARIOS: usize = 20;
pub const MAX_NODES: u64 = 200_000;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Covered { pub at: Place, pub own: bool, pub by: Vec<Unit> }

/// A placement of a supplying unit: (its index in `Report::units`, the place it goes to, the moves it takes).
pub type Supplier = (usize, Place, u32);

#[derive(Debug, Clone, Default)]
pub struct Report {
    pub asker: Option<Side>,
    pub target: Option<Unit>,
    pub unreadable: Option<&'static str>,
    pub exits_from: &'static str,
    /// (place, "own" | "exit")
    pub holes: Vec<(Place, &'static str)>,
    pub covered: Vec<Covered>,
    pub units: Vec<Unit>,
    pub supply: Vec<Vec<Supplier>>,
    /// (total moves, placements) in the order of the sorted plans.
    pub scenarios: Vec<(u32, Vec<Supplier>)>,
    pub safe: Option<bool>,
}

impl Report {
    pub fn readable(&self) -> bool { self.target.is_some() && self.unreadable.is_none() }
}

struct Search<'s> { supply: &'s [Vec<Supplier>], found: Vec<Vec<Supplier>>, nodes: u64, cut: bool }

impl Search<'_> {
    fn run(&mut self, h: usize, chosen: &mut Vec<Supplier>) {
        if self.cut { return; }
        self.nodes += 1;
        if self.nodes > MAX_NODES || self.found.len() >= MAX_SCENARIOS { self.cut = true; return; }
        if h == self.supply.len() { self.found.push(chosen.clone()); return; }
        if self.supply[h].iter().any(|x| chosen.iter().any(|c| x.0 == c.0 && x.1 == c.1)) { self.run(h + 1, chosen); return; }
        let options = self.supply[h].clone();
        for x in options {
            if chosen.iter().any(|c| c.0 == x.0) { continue; }
            chosen.push(x);
            self.run(h + 1, chosen);
            chosen.pop();
            if self.cut { return; }
        }
    }
}

/// The report with `asker` as the side that wants the goal.
pub fn report<G: Game + Placed>(r: &Reading<G>, asker: Side) -> Report {
    let owner = asker.other();
    let mut rep = Report { asker: Some(asker), exits_from: "none", ..Default::default() };
    let Some(kind) = r.goal_kind() else {
        rep.unreadable = Some("no goal unit on the board");
        return rep;
    };
    let targets: Vec<&Unit> = r.units.iter().filter(|u| u.side == owner && u.kind == kind).collect();
    if targets.len() != 1 {
        rep.unreadable = Some(if targets.is_empty() { "no goal unit on the board" } else { "several goal units: the target is not one unit" });
        return rep;
    }
    let t = *targets[0];
    rep.target = Some(t);
    let np = r.np();
    let k = (r.params_k.max(1) as usize).min(r.m.can_go);
    let mine: Vec<&Unit> = r.units.iter().filter(|u| u.side == asker).collect();
    let units: Vec<Unit> = mine.iter().take(MAX_UNITS).map(|u| **u).collect();
    let mut zone: PlaceSet = (*r.exits_of(&t)).clone();
    let (exits, from) = r.goal_exits(&t);
    rep.exits_from = from;
    zone.union_with(&exits);
    let acted_by = |place: Place| -> Vec<Unit> { mine.iter().filter(|u| r.strikes(u, place)).map(|u| **u).collect() };
    let acting = |u: &Unit, at: Place, place: Place| r.acts_from(u.kind, u.side, at).contains(place);
    let mut holes: Vec<(Place, &'static str)> = vec![];
    let mut covered: Vec<Covered> = vec![];
    let own_by = acted_by(t.at);
    if own_by.is_empty() { holes.push((t.at, "own")); } else { covered.push(Covered { at: t.at, own: true, by: own_by }); }
    for place in zone.iter() {
        let by = acted_by(place);
        if by.is_empty() {
            if exits.contains(place) { holes.push((place, "exit")); }
        } else {
            covered.push(Covered { at: place, own: false, by });
        }
    }
    let mut reach: Vec<Vec<(Place, u32)>> = vec![];
    for u in &units {
        let mut m: Vec<u32> = vec![u32::MAX; np];
        m[u.at as usize] = 0;
        for j in 1..=k {
            for d in r.go_from(j, u.kind, u.side, u.at).iter() { if m[d as usize] == u32::MAX { m[d as usize] = j as u32; } }
        }
        let alone: Vec<Place> = covered.iter().filter(|c| c.by.len() == 1 && c.by[0] == *u).map(|c| c.at).collect();
        reach.push(m.into_iter().enumerate().filter(|(q, j)| *j != u32::MAX && (*j == 0 || alone.iter().all(|a| acting(u, *q as Place, *a)))).map(|(q, j)| (q as Place, j)).collect());
    }
    let mut supply: Vec<Vec<Supplier>> = vec![];
    for (at, _) in &holes {
        let mut sup: Vec<(u32, usize, Place)> = vec![];
        for (i, u) in units.iter().enumerate() {
            for (q, j) in &reach[i] { if acting(u, *q, *at) { sup.push((*j, i, *q)); } }
        }
        sup.sort();
        supply.push(sup.into_iter().map(|(j, i, q)| (i, q, j)).collect());
    }
    if holes.len() <= MAX_HOLES {
        let mut search = Search { supply: &supply, found: vec![], nodes: 0, cut: false };
        search.run(0, &mut vec![]);
        let cut = search.cut;
        let mut found = search.found;
        if !holes.is_empty() {
            found.retain(|sc| (0..sc.len()).all(|skip| {
                let all_covered = (0..holes.len()).all(|hh| sc.iter().enumerate().any(|(i, c)| i != skip && supply[hh].iter().any(|x| x.0 == c.0 && x.1 == c.1)));
                !all_covered
            }));
        }
        let plans: BTreeSet<(u32, Vec<Supplier>)> = found.into_iter().map(|sc| {
            let mut sorted = sc.clone();
            sorted.sort();
            (sc.iter().map(|c| c.2).sum(), sorted)
        }).collect();
        rep.scenarios = plans.into_iter().collect();
        rep.safe = if cut { None } else { Some(rep.scenarios.is_empty()) };
    }
    rep.holes = holes;
    rep.covered = covered;
    rep.units = units;
    rep.supply = supply;
    rep
}
