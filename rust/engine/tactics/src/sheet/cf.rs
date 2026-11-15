//! THE OBSTACLES FOUND BY THE COUNTERFACTUAL (the Python sheet's `counterfactual_obstacles`, `_move_aside`, `_without_all`), not listed by shape. Each step of a route (a hop, then the
//! final strike on the goal place) is asked of the game on the base position (for a later step on a hypothetical board with the unit already on the step's first place). A step the
//! game offers is not blocked. For a step it does not offer, every other unit is taken off in turn (a hypothetical board, never a position of the game) and the game is asked again:
//! each unit whose removal lets the step be made is an obstacle, whatever the reason. A unit the game will not read the board without (the goal unit) is moved aside instead; when no
//! single removal does it and the units named on the step are several, all of them are removed together (a JOINT obstacle). The counterfactual boards are the game's to make
//! (`Placed::without`, `Placed::relocate`).
use super::levels::Route;
use super::reading::Reading;
use super::{Params, Place, PlaceSet, Side, Unit};
use cs_game::placed::Placed;
use cs_game::Game;

/// An obstacle found by the counterfactual: the describing word, the unit, its place, the number of the step, whether it was read on a board without the unit (always), whether no
/// unit accounts for it (`dead`), and for one of several units that must all go, the places of the whole set.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CfObstacle {
    pub kind: &'static str,
    pub unit: Option<Unit>,
    pub place: Place,
    pub hop: usize,
    pub dead: bool,
    pub joint: Vec<Place>,
    /// the place the unit was moved aside to, when it was moved aside rather than removed
    pub aside: Option<Place>,
}

/// The first place a unit that the game will not read the board without can be stood on instead: the empty places its own hops reach, none of them in `avoid`; those the other side
/// does not bear on come first; at most `aside_places` tried. -> (the place, the board).
fn move_aside<G: Game + Placed>(r: &Reading<G>, pos: &G::State, h: Place, avoid: &[Place], v: Option<Unit>, params: &Params) -> Option<(Place, G::State)> {
    let v = v?;
    if params.aside_places < 1 { return None; }
    let role = r.owner(v.side);
    let struck: PlaceSet = r.struck_by(v.side.other());
    let mut cands: Vec<Place> = r.m.medium.hops(v.kind, role, h).iter().map(|x| x.0).filter(|p| !avoid.contains(p) && r.g.kind_at(pos, *p).is_none()).collect();
    cands.sort_by_key(|p| (struck.contains(*p), *p));
    for p in cands.into_iter().take(params.aside_places) {
        if let Some(moved) = r.g.relocate(pos, h, p) {
            if r.g.outcome(&moved).is_none() { return Some((p, moved)); }
        }
    }
    None
}

/// The board with every unit of `group` out of the way: each is removed, and a unit whose removal the game refuses as not a position is moved aside instead.
/// -> (the board or None when one cannot be got out of the way, [(place, where it was moved aside)]).
fn without_all<G: Game + Placed>(r: &Reading<G>, pos: &G::State, group: &[Place], avoid: &[Place], params: &Params) -> (Option<G::State>, Vec<(Place, Place)>) {
    let mut pos = pos.clone();
    let mut aside = vec![];
    for h in group {
        if let Some(gone) = r.g.without(&pos, *h) { pos = gone; continue; }
        let mut av = avoid.to_vec();
        av.extend_from_slice(group);
        let by = r.unit_at(*h).map(|i| r.units[i]);
        match move_aside(r, &pos, *h, &av, by, params) {
            None => return (None, aside),
            Some((p, st)) => { aside.push((*h, p)); pos = st; }
        }
    }
    (Some(pos), aside)
}

fn offers<G: Game + Placed>(r: &Reading<G>, s: &G::State, a: Place, b: Place) -> bool { r.g.outcome(s).is_none() && r.g.find_step(s, a, b).is_some() }

/// The obstacles of one route, found by the counterfactual. `blocked` counts the blocked steps this question has examined so far (the cap is per question). -> (obstacles, cut).
pub fn counterfactual_obstacles<G: Game + Placed>(r: &Reading<G>, asker: Side, route: &Route, params: &Params, base: &G::State, blocked: &mut usize) -> (Vec<CfObstacle>, bool) {
    let med = &r.m.medium;
    let (origin, unit) = (route.unit.at, route.unit);
    let role = r.owner(unit.side);
    let goal = route.goal;
    let arrival = route.places.last().copied().unwrap_or(origin);
    let mut steps: Vec<(Place, Place, Vec<Place>, usize)> = vec![];
    for (i, p) in route.places.iter().enumerate() {
        steps.push((if i == 0 { origin } else { route.places[i - 1] }, *p, route.over.get(i).cloned().unwrap_or_default(), i));
    }
    steps.push((arrival, goal, if med.takes(unit.kind, role) { vec![] } else { med.over(unit.kind, role, arrival, goal).to_vec() }, route.places.len()));
    let (mut out, mut cut) = (vec![], false);
    let asker_goal = r.goal_unit(asker);
    for (a_place, b_place, over, idx) in steps {
        let pos = if a_place == origin { base.clone() } else { match r.g.relocate(base, origin, a_place) { Some(p) => p, None => continue } };
        if offers(r, &pos, a_place, b_place) || r.g.outcome(&pos).is_some() { continue; }       // the game offers the step: nothing blocks it
        if *blocked >= params.max_blocked_steps { cut = true; continue; }
        *blocked += 1;
        let mut found: Vec<Place> = vec![];
        let mut aside: Vec<(Place, Place)> = vec![];
        let mut route_places: Vec<Place> = route.places.clone();
        for ov in &route.over { route_places.extend_from_slice(ov); }
        route_places.extend_from_slice(&[origin, a_place, b_place, goal]);
        for (_o, _k, h) in r.g.cells(&pos) {
            if h == a_place { continue; }
            match r.g.without(&pos, h) {
                None => {                                                  // the game rejects the board without it (the goal unit): not a position
                    let by = r.unit_at(h).map(|i| r.units[i]);
                    if let Some((p, st3)) = move_aside(r, &pos, h, &route_places, by, params) {
                        if offers(r, &st3, a_place, b_place) { found.push(h); aside.push((h, p)); }
                    }
                }
                Some(st2) => { if offers(r, &st2, a_place, b_place) { found.push(h); } }
            }
        }
        if found.is_empty() {
            let mut named: Vec<Place> = vec![];
            let mut cand = over.clone();
            if idx < route.places.len() { cand.push(b_place); }
            for h in cand { if !named.contains(&h) && r.unit_at(h).is_some() && h != a_place { named.push(h); } }
            if named.len() >= 2 {
                if named.len() > params.max_joint {
                    out.push(CfObstacle { kind: "more units in the way than the cap", unit: None, place: b_place, hop: idx, dead: true, joint: vec![], aside: None });
                    continue;
                }
                let (gone, moved) = without_all(r, &pos, &named, &route_places, params);
                if let Some(gp) = gone {
                    if offers(r, &gp, a_place, b_place) {
                        let mut members = named.clone();
                        members.sort_by(|x, y| {
                            let (ux, uy) = (r.units[r.unit_at(*x).unwrap()], r.units[r.unit_at(*y).unwrap()]);
                            ((ux.side != asker), r.worth_of(ux.kind), *x).partial_cmp(&((uy.side != asker), r.worth_of(uy.kind), *y)).unwrap()
                        });
                        for h in &members {
                            let v = r.units[r.unit_at(*h).unwrap()];
                            out.push(CfObstacle { kind: "one of several units, each of which must go", unit: Some(v), place: *h, hop: idx, dead: false, joint: members.clone(),
                                                  aside: moved.iter().find(|m| m.0 == *h).map(|m| m.1) });
                        }
                        continue;
                    }
                }
            }
        }
        let mut words = 0;
        for h in &found {
            let Some(v) = r.unit_at(*h).map(|i| r.units[i]) else { continue };
            let kind = if over.contains(h) { "stands on a place passed over" } else if *h == b_place { "stands on the landing place" }
                       else if asker_goal.is_some_and(|ag| v != ag && r.strikes(&v, ag.at)) { "strikes the asker's goal unit" } else { "other" };
            words += 1;
            out.push(CfObstacle { kind, unit: Some(v), place: *h, hop: idx, dead: false, joint: vec![], aside: aside.iter().find(|a| a.0 == *h).map(|a| a.1) });
        }
        if words == 0 {
            out.push(CfObstacle { kind: "none", unit: None, place: b_place, hop: idx, dead: true, joint: vec![], aside: None });
        }
    }
    (out, cut)
}
