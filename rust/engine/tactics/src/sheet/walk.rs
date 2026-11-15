//! THE ROUTE FOLLOWER (the Python sheet's `_Walk`, `_conditions`, `search_way`): follow one route for real from a base position (the asker to move): the asker plays each step of the
//! route, the owner answers every legal move (the likely stoppers first), and after the last step and one more answer the asker's real exchange frame runs on the target (followed
//! where it goes). Value to the asker: the least over the owner's answers; a step that cannot be made after an answer is a DELAY (kept in `stops`, not counted in the least); an
//! answer that brings the value to 0 or below is a HARD stop and ends that branch. The verdict of a way is data: CONFIRMED, REFUTED, WAITING or NOT SETTLED, with its stops.
use super::frame::solve_real;
use super::levels::Route;
use super::reading::Reading;
use super::{Params, Place, Side, Unit};
use cs_game::placed::Placed;
use cs_game::Game;

/// A ledger value at or above this is a line that reaches the goal.
pub const BIG: f64 = 1_000_000_000.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum V { Confirmed, Refuted, Waiting, Unsettled }

impl V {
    pub fn word(self) -> &'static str { match self { V::Confirmed => "confirmed", V::Refuted => "refuted", V::Waiting => "waiting", V::Unsettled => "not settled" } }
}

/// One answer of the owner (or the board as it stands) that stops a route. `hard`: it costs the owner nothing; a delay: the step cannot be made now but nothing is lost.
#[derive(Debug, Clone)]
pub struct Stop {
    pub hard: bool,
    /// the owner's move (its name); None: the board as it stands
    pub answer: Option<String>,
    pub after: usize,
    /// delay: the number of the step that cannot be made
    pub step: Option<usize>,
    pub reason: &'static str,
    pub places: Vec<Place>,
    /// (origin, destination) of the answer
    pub mover: Option<(Place, Place)>,
}

#[derive(Debug, Clone)]
pub struct RouteResult {
    pub route: Route,
    pub verdict: V,
    pub value: f64,
    pub line: Vec<String>,
    pub stops: Vec<Stop>,
    pub nodes: u64,
    pub tried: usize,
    pub capped: bool,
}

struct Capped;

/// The positions' moves ordered so that the likely stoppers of a route come first: the move that takes the travelling unit, then one that lands on a place of the route, then one
/// after which that side bears on the unit's place or a place of the route, then one that moves the target or newly guards it, then the rest (each group in the names' order).
pub fn rank_answers<G: Game + Placed>(g: &G, s: &G::State, moves: Vec<G::Move>, unit_place: Place, route_places: &[Place], target: Place) -> Vec<G::Move> {
    let Some(me) = g.mover(s) else { return moves };
    let guarded_before = g.strikes_place(s, me, target);
    let mut scored: Vec<(u8, String, G::Move)> = Vec::with_capacity(moves.len());
    for m in moves {
        let (from, to) = g.move_ends(s, &m);
        let score = if to == unit_place { 0 } else if route_places.contains(&to) { 1 } else {
            let after = g.apply(s, &m);
            if std::iter::once(unit_place).chain(route_places.iter().copied()).any(|p| g.strikes_place(&after, me, p)) { 2 }
            else if from == target || (!guarded_before && g.strikes_place(&after, me, target)) { 3 } else { 4 }
        };
        scored.push((score, g.move_name(s, &m), m));
    }
    scored.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
    scored.into_iter().map(|x| x.2).collect()
}

pub struct Walk<'a, 'g, G: Game + Placed> {
    r: &'a Reading<'g, G>,
    t: Unit,
    route: Route,
    params: &'a Params,
    base: G::State,
    cap: u64,
    places: Vec<Place>,
    origin: Place,
    unit_worth: f64,
    stops: Vec<Stop>,
    nodes: u64,
    tried: usize,
}

type StepOut = (Option<f64>, Vec<String>);

impl<'a, 'g, G: Game + Placed> Walk<'a, 'g, G> {
    pub fn new(r: &'a Reading<'g, G>, t: Unit, _asker: Side, route: Route, params: &'a Params, base: G::State, cap: u64) -> Self {
        let places = route.places.clone();
        let origin = route.unit.at;
        let unit_worth = r.worth_of(route.unit.kind);
        Walk { r, t, route, params, base, cap, places, origin, unit_worth, stops: vec![], nodes: 0, tried: 0 }
    }

    fn tick(&mut self, n: u64) -> Result<(), Capped> {
        self.nodes += n;
        if self.nodes > self.cap { Err(Capped) } else { Ok(()) }
    }

    fn solve(&self, place: Place, s: &G::State) -> super::frame::Solved { solve_real(self.r, place, s, self.params.frame_budget, "given") }

    fn hard(&mut self, pos: &G::State, mv: &G::Move, after: usize, reason: &'static str) {
        let g = self.r.g;
        self.stops.push(Stop { hard: true, answer: Some(g.move_name(pos, mv)), after, step: None, reason, places: vec![], mover: Some(g.move_ends(pos, mv)) });
    }

    pub fn run(mut self) -> RouteResult {
        let (mut capped, mut value, mut line) = (false, None, vec![]);
        let base = self.base.clone();
        match self.step(&base, 0, self.origin, self.t.at, None, vec![]) {
            Ok((v, l)) => { value = v; line = l; }
            Err(Capped) => capped = true,
        }
        let hard = self.stops.iter().any(|x| x.hard);
        let verdict = if hard { V::Refuted } else if capped { V::Unsettled } else if !self.stops.is_empty() { V::Waiting } else { V::Confirmed };
        let value = if hard { 0.0 } else { value.unwrap_or(0.0) };
        RouteResult { route: self.route, verdict, value, line, stops: self.stops, nodes: self.nodes, tried: self.tried, capped }
    }

    /// -> (value to the asker or None when the step cannot be made, the line). `pos`: the asker to move, its unit on `at`, the target on `where_`; `by`: (position, move) of the
    /// answer that came before.
    fn step(&mut self, pos: &G::State, i: usize, at: Place, where_: Place, by: Option<(G::State, G::Move)>, line: Vec<String>) -> Result<StepOut, Capped> {
        let (r, g) = (self.r, self.r.g);
        let dest = self.places[i];
        let Some(mv) = g.find_step(pos, at, dest) else {
            let (reason, named) = g.why_not(pos, at, dest);
            let (answer, mover) = match &by { Some((bp, bm)) => (Some(g.move_name(bp, bm)), Some(g.move_ends(bp, bm))), None => (None, None) };
            self.stops.push(Stop { hard: false, answer, after: i, step: Some(i + 1), reason: reason.word(), places: named, mover });
            return Ok((None, line));
        };
        let mut line = line;
        line.push(g.move_name(pos, &mv));
        let child = g.apply(pos, &mv);
        if let Some(out) = g.outcome(&child) {
            return Ok((Some(if out.contains(&100) { BIG } else { 0.0 }), line));
        }
        if dest == where_ {                                                  // the step itself takes the target: the owner may take back
            let ex = self.solve(where_, &child);
            self.tick(ex.nodes)?;
            line.extend(ex.line.iter().cloned());
            return Ok((Some((r.worth_of(self.t.kind) - ex.value).max(0.0)), line));
        }
        let mut worst: Option<(f64, Vec<String>)> = None;
        let answers = rank_answers(g, &child, g.moves(&child), dest, &self.places[i + 1..], where_);
        for a in answers {
            self.tried += 1;
            self.tick(1)?;
            let nxt = g.apply(&child, &a);
            let (ofrom, oto) = g.move_ends(&child, &a);
            let (v, ln): (f64, Vec<String>);
            if g.outcome(&nxt).is_some() {
                self.hard(&child, &a, i + 1, "game_over");
                let mut l = line.clone();
                l.push(g.move_name(&child, &a));
                return Ok((Some(0.0), l));
            }
            if oto == dest {                                                 // the travelling unit is taken: what does it cost the owner
                let ex = self.solve(dest, &nxt);
                self.tick(ex.nodes)?;
                v = ex.value - self.unit_worth;
                let mut l = vec![g.move_name(&child, &a)];
                l.extend(ex.line.iter().cloned());
                ln = l;
                if v <= 0.0 {
                    self.hard(&child, &a, i + 1, "unit_taken");
                    let mut full = line.clone();
                    full.extend(ln);
                    return Ok((Some(0.0), full));
                }
            } else {
                let where2 = if ofrom == where_ { oto } else { where_ };
                if i + 1 < self.places.len() {
                    let mut l1 = line.clone();
                    l1.push(g.move_name(&child, &a));
                    let (sv, sl) = self.step(&nxt, i + 1, dest, where2, Some((child.clone(), a.clone())), l1)?;
                    let Some(sv) = sv else { continue };
                    if sv <= 0.0 { return Ok((Some(0.0), sl)); }
                    v = sv;
                    ln = sl[line.len()..].to_vec();
                } else if g.kind_at(&nxt, where2).is_none() {
                    self.hard(&child, &a, i + 1, "exchange_stopped");
                    let mut l = line.clone();
                    l.push(g.move_name(&child, &a));
                    return Ok((Some(0.0), l));
                } else {
                    let ex = self.solve(where2, &nxt);
                    self.tick(ex.nodes)?;
                    v = ex.value;
                    let mut l = vec![g.move_name(&child, &a)];
                    l.extend(ex.line.iter().cloned());
                    ln = l;
                    if v <= 0.0 {
                        let (mut pos0, mut mv0, mut after0) = (child.clone(), a.clone(), i + 1);
                        if let Some((bp, bm)) = &by {                         // was the exchange already dead before this last answer (the game's pass gives the asker the move)? then the answer before it is the one that bites
                            if let Some(got) = g.pass(&child) {
                                let ex0 = self.solve(where_, &got);
                                self.tick(ex0.nodes)?;
                                if ex0.value <= 0.0 { pos0 = bp.clone(); mv0 = bm.clone(); after0 = i; }
                            }
                        }
                        self.hard(&pos0, &mv0, after0, "exchange_stopped");
                        let mut full = line.clone();
                        full.extend(ln);
                        return Ok((Some(0.0), full));
                    }
                }
            }
            if worst.as_ref().is_none_or(|w| v < w.0) { worst = Some((v, ln)); }
        }
        match worst {
            None => Ok((None, line)),
            Some((v, l)) => { let mut full = line; full.extend(l); Ok((Some(v), full)) }
        }
    }
}

/// What a waiting route waits on, from the delays the search saw: [(kind, places)]. A place vacated, a line cleared, the unit able to move again, and for a later step the steps
/// before it made.
pub fn conditions(stops: &[Stop], route: &Route) -> Vec<(&'static str, Vec<Place>)> {
    let mut out: Vec<(&'static str, Vec<Place>)> = vec![];
    for x in stops.iter().filter(|x| !x.hard) {
        let cond = match x.reason { "place_taken" => ("vacated", x.places.clone()), "line_shut" => ("cleared", x.places.clone()), _ => ("freed", x.places.clone()) };
        let mut cs: Vec<(&'static str, Vec<Place>)> = vec![];
        if let Some(s) = x.step { if s > 1 { cs.push(("step", vec![route.places[s - 2]])); } }
        cs.push(cond);
        for c in cs { if !out.contains(&c) { out.push(c); } }
    }
    out
}

/// The verdict of a way: the route picked among those followed, with what it waits on and what stops it.
#[derive(Debug, Clone)]
pub struct WayVerdict {
    pub verdict: V,
    pub value: f64,
    pub line: Vec<String>,
    pub route: Option<Route>,
    pub answer: Option<String>,
    pub alts: Vec<(&'static str, Vec<Place>)>,
    pub breaks: Vec<Stop>,
    pub nodes: u64,
    pub tried: usize,
    pub cut: bool,
    pub routes_tried: usize,
    pub stops: Vec<Stop>,
}

/// Follow the routes of a way (at most `max_routes`, the best-ordered first) from the base position, within `route_nodes` nodes over all of them. None when the asker is not to
/// move and the game refuses its pass (the caller reads the way after its first move, marked).
pub fn search_way<G: Game + Placed>(r: &Reading<G>, t: &Unit, asker: Side, routes: &[Route], params: &Params) -> Option<WayVerdict> {
    let base = if asker == Side::Us { Some(r.s.clone()) } else { r.passed_state() }?;
    let (mut used, mut results): (u64, Vec<RouteResult>) = (0, vec![]);
    for route in routes.iter().take(params.max_routes) {
        let res = Walk::new(r, *t, asker, route.clone(), params, base.clone(), params.route_nodes.saturating_sub(used)).run();
        used += res.nodes;
        let stop = matches!(res.verdict, V::Confirmed | V::Unsettled);
        results.push(res);
        if stop { break; }
    }
    let pick = results.iter().find(|x| x.verdict == V::Confirmed).or_else(|| results.iter().find(|x| x.verdict == V::Waiting))
        .or_else(|| results.iter().find(|x| x.verdict == V::Unsettled)).or(results.first());
    let Some(pick) = pick else {
        return Some(WayVerdict { verdict: V::Refuted, value: 0.0, line: vec![], route: None, answer: None, alts: vec![], breaks: vec![], nodes: 0, tried: 0, cut: false, routes_tried: 0, stops: vec![] });
    };
    let stops = pick.stops.clone();
    let alts = if pick.verdict == V::Waiting { conditions(&stops, &pick.route) } else { vec![] };
    let breaks: Vec<Stop> = stops.iter().filter(|x| x.hard || pick.verdict == V::Waiting).cloned().collect();
    let answer = stops.iter().find(|x| x.hard).map(|x| x.answer.clone()).or_else(|| if pick.verdict == V::Waiting { stops.first().map(|x| x.answer.clone()) } else { None }).flatten();
    Some(WayVerdict { verdict: pick.verdict, value: pick.value, line: pick.line.clone(), route: Some(pick.route.clone()), answer, alts, breaks, nodes: used,
                      tried: results.iter().map(|x| x.tried).sum(), cut: results.iter().any(|x| x.capped), routes_tried: results.len(), stops })
}
