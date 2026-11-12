//! PLAN 78.39 STAGE 1: A GOAL TAKEN APART BACKWARD, as a REPORT (not yet a step of the decision). The goal's TARGET is a unit of the other side (read from the description:
//! cs_game::goal_target); the side that wants the goal is ROLE R. The reading:
//!   HOLES     the places where the goal does not yet bind: (1) the target's own place, when no unit of R acts on it; (2) every place the target itself could move to that no unit of R
//!             acts on. The target's moves are the game's own (`Game::moves` of the state, or of the passed state when it is R's turn; a state with no pass has its exits read from the
//!             target's zone instead -- the places its kind can go, `exits_from` says which), a move is the target's when `reason::effect` shows the target's fact leaving one place and
//!             arriving at another, so legality is the game's. The places of the target's zone that ARE already acted on are reported with the units that act on them (`covered`).
//!   SUPPLY    for each hole, the units of R that could act on the hole's place from some place they can get to within k of their own moves, with that place and the move count --
//!             read through the fact view from two relations named by the caller, by the roles they play (acts-on: a unit of a kind and owner standing at a place bears on a place; can-go:
//!             where such a unit may move in 1, 2, .. moves, one relation per count): this module names no relation of any game.
//!   SCENARIOS assignments of supplying units to ALL holes, one unit covering several holes only when a single place of it acts on all of them; found by backtracking, in the least
//!             total of moves first, a scenario that has a unit it does not need dropped. None: the goal is out of reach of the units present at depth k, which read from the other side
//!             says the target is SAFE (from that direction, until the units change).
//!   CONTEST   not read yet (`contest` is None).
//! THE BOARD IS READ AS IT STANDS: a blocker that could move away is not considered, the place a unit leaves is not freed for its own later steps, the units of the other side do not
//! answer (a unit that is the only one acting on a place now is kept there: it may move only to a place that acts on it too) (no interposing, no taking of the unit that acts), and the units of R do not unblock each other. So the answer is an over- and an under-estimate in known ways: a scenario is a
//! candidate and "none" is "none with the board as it stands".
use crate::won;
use cs_game::gdl_game::FactView;
use cs_game::goal_target::GoalTarget;
use cs_game::{Fact, Game, Role};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

/// A place: the arguments of a unit's place, as the game writes them.
pub type Loc = Vec<String>;

pub fn show_loc(l: &Loc) -> String { l.join(":") }

/// A relation of the fact view by the roles of its arguments (argument positions): the unit's kind, its owner, the place it stands at, the place it acts on or goes to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RelSpec { pub relation: String, pub kind: usize, pub owner: usize, pub from: Vec<usize>, pub to: Vec<usize> }

/// The two relations of the reading: `acts_on`; `can_go[j - 1]` = where a unit may go in at most j moves (so k <= can_go.len()).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Relations { pub acts_on: RelSpec, pub can_go: Vec<RelSpec> }

#[derive(Debug, Clone)]
pub struct Params {
    /// Moves of its own a supplying unit may use.
    pub k: u32,
    /// More holes than this: no scenario is searched (the report says so).
    pub max_holes: usize,
    /// Only this many units of R are read (in the order of their kind and place).
    pub max_units: usize,
    /// The search stops at this many scenarios.
    pub max_scenarios: usize,
    /// The search stops at this many nodes.
    pub max_nodes: u64,
}
impl Default for Params { fn default() -> Params { Params { k: 2, max_holes: 8, max_units: 16, max_scenarios: 20, max_nodes: 200_000 } } }

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct UnitAt { pub kind: String, pub at: Loc }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HoleKind { Own, Exit }
impl HoleKind { pub fn word(&self) -> &'static str { match self { HoleKind::Own => "own", HoleKind::Exit => "exit" } } }

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Hole { pub at: Loc, pub kind: HoleKind }

/// A place of the target's zone already acted on, and by whom.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Covered { pub at: Loc, pub own: bool, pub by: Vec<UnitAt> }

/// A unit of R (by its index in `Report::units`) that could act on a hole from `to`, in `moves` of its own.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Supplier { pub unit: usize, pub to: Loc, pub moves: u32 }

/// One placement of a scenario: the unit (kind, where it stands now), where it goes, in how many moves.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Placement { pub unit: UnitAt, pub to: Loc, pub moves: u32 }

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Scenario { pub placements: Vec<Placement>, pub total_moves: u32 }

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TargetAt { pub kind: String, pub owner: String, pub at: Loc }

#[derive(Debug, Clone, Default)]
pub struct Report {
    pub role: String,
    pub target: Option<TargetAt>,
    /// Why nothing could be read (no target on the board, the state has no mover, ...). None: read.
    pub unreadable: Option<String>,
    /// The state is terminal and R's goal is reached: no holes.
    pub goal_holds: bool,
    /// Where the target's exits were read from: "moves" (the game's own), "zone" (no pass: the places its kind can go) or "none".
    pub exits_from: String,
    pub holes: Vec<Hole>,
    pub covered: Vec<Covered>,
    /// The units of R read.
    pub units: Vec<UnitAt>,
    /// Per hole (same order), who could act on it.
    pub supply: Vec<Vec<Supplier>>,
    pub scenarios: Vec<Scenario>,
    /// Some(true): no scenario exists (the search was complete): the target is safe from this direction at depth k. Some(false): one exists. None: not decided (a cap was hit).
    pub safe: Option<bool>,
    /// Caps hit and other things the reader must know.
    pub notes: Vec<String>,
    /// The CONTEST (what of theirs denies the supply): not read yet.
    pub contest: Option<String>,
    pub nodes: u64,
    pub seconds: f64,
}

fn take(args: &[String], at: &[usize]) -> Loc { at.iter().map(|i| args[*i].clone()).collect() }

/// The rows of a relation (by name) as (kind, owner, from, to).
fn rows_of(facts: &[Fact], spec: &RelSpec) -> Vec<(String, String, Loc, Loc)> {
    facts.iter().filter(|f| f.relation == spec.relation).map(|f| (f.args[spec.kind].clone(), f.args[spec.owner].clone(), take(&f.args, &spec.from), take(&f.args, &spec.to))).collect()
}

/// A role to read for: "mover" (the side to move), "other" (the side not to move), or a role's own name.
pub fn resolve_role<G: Game>(g: &G, s: &G::State, word: &str) -> Result<Role, String> {
    let roles = g.roles();
    match word {
        "mover" => g.mover(s).ok_or_else(|| "no side is to move (a terminal state): name the role".to_string()),
        "other" => g.mover(s).map(|m| g.other(m)).ok_or_else(|| "no side is to move (a terminal state): name the role".to_string()),
        w => roles.iter().position(|r| r == w).ok_or_else(|| format!("{w} is no role of the game")),
    }
}

/// THE REPORT of role `me` at `s` (see the module). `view` derives the two relations from the state's base facts; `target` is the goal's target read from the description.
pub fn scenario<G: Game>(g: &G, s: &G::State, view: &FactView, target: &GoalTarget, rels: &Relations, me: Role, p: &Params) -> Result<Report, String> {
    let t0 = std::time::Instant::now();
    if p.k as usize > rels.can_go.len() { return Err(format!("k = {} but only {} can-go relations are named", p.k, rels.can_go.len())); }
    if p.k == 0 { return Err("k is at least 1".into()); }
    let roles = g.roles();
    if roles.len() != 2 { return Err("the report reads games of two roles".into()); }
    let (mine, theirs) = (roles[me].clone(), roles[g.other(me)].clone());
    let mut rep = Report { role: mine.clone(), exits_from: "none".into(), ..Default::default() };
    let base = g.base_facts(s);
    if base.is_empty() { return Err("the game gives no facts".into()); }
    let facts = view.facts(&base)?;

    // the units, as the description's fluent holds them
    let unit_of = |f: &Fact| -> Option<(String, UnitAt)> {
        if f.relation != target.fluent || f.args.len() != target.arity { return None; }
        Some((f.args[target.owner_arg].clone(), UnitAt { kind: f.args[target.kind_arg].clone(), at: take(&f.args, &target.place_args) }))
    };
    let all: Vec<(String, UnitAt)> = base.iter().filter_map(unit_of).collect();
    let targets: Vec<&UnitAt> = all.iter().filter(|(o, u)| *o == theirs && u.kind == target.kind).map(|(_, u)| u).collect();
    let mut units: Vec<UnitAt> = all.iter().filter(|(o, _)| *o == mine).map(|(_, u)| u.clone()).collect();
    units.sort();
    if units.len() > p.max_units { rep.notes.push(format!("{} units of {mine}: the first {} read", units.len(), p.max_units)); units.truncate(p.max_units); }
    rep.units = units.clone();
    let tu = match targets.as_slice() {
        [t] => (*t).clone(),
        [] => { rep.unreadable = Some(format!("{theirs} has no {} on the board", target.kind)); rep.seconds = t0.elapsed().as_secs_f64(); return Ok(rep); }
        _ => { rep.unreadable = Some(format!("{theirs} has several {}: the target is not one unit", target.kind)); rep.seconds = t0.elapsed().as_secs_f64(); return Ok(rep); }
    };
    rep.target = Some(TargetAt { kind: tu.kind.clone(), owner: theirs.clone(), at: tu.at.clone() });

    // the goal already holds
    if let Some(v) = g.outcome(s) {
        rep.goal_holds = won(&v, me);
        rep.exits_from = "none".into();
        rep.safe = Some(!rep.goal_holds);
        if rep.goal_holds { rep.scenarios.push(Scenario { placements: vec![], total_moves: 0 }); }
        rep.seconds = t0.elapsed().as_secs_f64();
        return Ok(rep);
    }

    // the relations
    let acts: HashSet<(String, String, Loc, Loc)> = rows_of(&facts, &rels.acts_on).into_iter().collect();
    let go: Vec<HashMap<(String, String, Loc), Vec<Loc>>> = rels.can_go.iter().take(p.k as usize).map(|spec| {
        let mut m: HashMap<(String, String, Loc), Vec<Loc>> = HashMap::new();
        for (k, o, f, t) in rows_of(&facts, spec) { m.entry((k, o, f)).or_default().push(t); }
        m
    }).collect();
    let acting = |kind: &str, at: &Loc, place: &Loc| acts.contains(&(kind.to_string(), mine.clone(), at.clone(), place.clone()));
    let acted_by = |place: &Loc| -> Vec<UnitAt> { all.iter().filter(|(o, u)| *o == mine && acting(&u.kind, &u.at, place)).map(|(_, u)| u.clone()).collect() };

    // the target's zone: where its kind can go from where it stands (one move)
    let mut zone: BTreeSet<Loc> = BTreeSet::new();
    if let Some(ds) = go.first().and_then(|m| m.get(&(tu.kind.clone(), theirs.clone(), tu.at.clone()))) { zone.extend(ds.iter().cloned()); }
    // the target's exits: the game's own legal moves of the target
    let st = match g.mover(s) { Some(m) if m == g.other(me) => Some(s.clone()), Some(_) => g.pass(s), None => None };
    let mut exits: BTreeSet<Loc> = BTreeSet::new();
    match &st {
        Some(st) => {
            rep.exits_from = "moves".into();
            for m in g.moves(st) {
                let (gone, arrived) = crate::reason::effect(g, st, &m);
                let left = gone.iter().filter_map(unit_of).any(|(o, u)| o == theirs && u == tu);
                if !left { continue; }
                for f in &arrived { if let Some((o, u)) = unit_of(f) { if o == theirs && u.kind == tu.kind && u.at != tu.at { exits.insert(u.at); } } }
            }
        }
        None => { rep.exits_from = "zone".into(); rep.notes.push("no pass: the target's exits are read from its zone, not from the game's moves".into()); exits = zone.clone(); }
    }
    // holes and covered places
    let own_by = acted_by(&tu.at);
    if own_by.is_empty() { rep.holes.push(Hole { at: tu.at.clone(), kind: HoleKind::Own }); } else { rep.covered.push(Covered { at: tu.at.clone(), own: true, by: own_by }); }
    let mut places: BTreeSet<Loc> = zone.clone();
    places.extend(exits.iter().cloned());
    for l in places {
        let by = acted_by(&l);
        if by.is_empty() { if exits.contains(&l) { rep.holes.push(Hole { at: l, kind: HoleKind::Exit }); } } else { rep.covered.push(Covered { at: l, own: false, by }); }
    }

    // supply: per unit the places it reaches (least moves), per hole the places that act on it
    let mut reach: Vec<BTreeMap<Loc, u32>> = vec![];
    for u in &units {
        let mut m: BTreeMap<Loc, u32> = BTreeMap::new();
        m.insert(u.at.clone(), 0);
        for (j, rel) in go.iter().enumerate() {
            if let Some(ds) = rel.get(&(u.kind.clone(), mine.clone(), u.at.clone())) { for d in ds { m.entry(d.clone()).or_insert(j as u32 + 1); } }
        }
        // a unit that moves must keep acting where it is the only unit that acts now (else the move opens a hole the report did not count)
        let alone: Vec<&Loc> = rep.covered.iter().filter(|c| c.by.len() == 1 && c.by[0] == *u).map(|c| &c.at).collect();
        m.retain(|q, j| *j == 0 || alone.iter().all(|a| acting(&u.kind, q, a)));
        reach.push(m);
    }
    for h in &rep.holes {
        let mut sup = vec![];
        for (i, u) in units.iter().enumerate() {
            for (q, j) in &reach[i] { if acting(&u.kind, q, &h.at) { sup.push(Supplier { unit: i, to: q.clone(), moves: *j }); } }
        }
        sup.sort_by(|a, b| (a.moves, a.unit, &a.to).cmp(&(b.moves, b.unit, &b.to)));
        rep.supply.push(sup);
    }

    // scenarios
    if rep.holes.len() > p.max_holes {
        rep.notes.push(format!("{} holes: more than {}: no scenario searched", rep.holes.len(), p.max_holes));
    } else {
        let mut found: Vec<Vec<(usize, Loc, u32)>> = vec![];
        let mut chosen: Vec<(usize, Loc, u32)> = vec![];
        let mut cut = false;
        search(0, &rep.supply, &mut chosen, &mut found, &mut rep.nodes, p, &mut cut);
        // a unit not needed makes a scenario no scenario
        found.retain(|sc| (0..sc.len()).all(|skip| !(0..rep.holes.len()).all(|h| sc.iter().enumerate().any(|(i, c)| i != skip && rep.supply[h].iter().any(|x| x.unit == c.0 && x.to == c.1)))) || rep.holes.is_empty());
        let mut sset: BTreeSet<(u32, Vec<(usize, Loc, u32)>)> = BTreeSet::new();
        for mut sc in found { sc.sort(); sset.insert((sc.iter().map(|c| c.2).sum(), sc)); }
        for (total, sc) in sset {
            rep.scenarios.push(Scenario { placements: sc.into_iter().map(|(u, to, j)| Placement { unit: units[u].clone(), to, moves: j }).collect(), total_moves: total });
        }
        if cut { rep.notes.push("a cap on scenarios or nodes was hit: the search is not complete".into()); rep.safe = None; }
        else { rep.safe = Some(rep.scenarios.is_empty()); }
    }
    rep.seconds = t0.elapsed().as_secs_f64();
    Ok(rep)
}

fn search(h: usize, supply: &[Vec<Supplier>], chosen: &mut Vec<(usize, Loc, u32)>, found: &mut Vec<Vec<(usize, Loc, u32)>>, nodes: &mut u64, p: &Params, cut: &mut bool) {
    if *cut { return; }
    *nodes += 1;
    if *nodes > p.max_nodes || found.len() >= p.max_scenarios { *cut = true; return; }
    if h == supply.len() { found.push(chosen.clone()); return; }
    // already covered by a placement made: no branch
    if supply[h].iter().any(|x| chosen.iter().any(|c| c.0 == x.unit && c.1 == x.to)) { search(h + 1, supply, chosen, found, nodes, p, cut); return; }
    for x in &supply[h] {
        if chosen.iter().any(|c| c.0 == x.unit) { continue; }                 // (a used unit stands at one place only)
        chosen.push((x.unit, x.to.clone(), x.moves));
        search(h + 1, supply, chosen, found, nodes, p, cut);
        chosen.pop();
        if *cut { return; }
    }
}

/// The report in plain lines, for a reader.
pub fn explain(r: &Report) -> Vec<String> {
    let mut out = vec![];
    if let Some(u) = &r.unreadable { out.push(format!("unreadable: {u}")); return out; }
    if let Some(t) = &r.target { out.push(format!("{} reads the {} of {} on {}", r.role, t.kind, t.owner, show_loc(&t.at))); }
    if r.goal_holds { out.push("the goal already holds: no holes".into()); return out; }
    out.push(format!("holes ({}; exits from {}): {}", r.holes.len(), r.exits_from, r.holes.iter().map(|h| format!("{}@{}", h.kind.word(), show_loc(&h.at))).collect::<Vec<_>>().join(" ")));
    for c in &r.covered { out.push(format!("covered {}{}: by {}", show_loc(&c.at), if c.own { " (the target itself)" } else { "" }, c.by.iter().map(|u| format!("{}@{}", u.kind, show_loc(&u.at))).collect::<Vec<_>>().join(" "))); }
    for (h, sup) in r.holes.iter().zip(&r.supply) {
        out.push(format!("supply for {}: {}", show_loc(&h.at), if sup.is_empty() { "none".to_string() } else {
            sup.iter().map(|x| format!("{}@{}->{} in {}", r.units[x.unit].kind, show_loc(&r.units[x.unit].at), show_loc(&x.to), x.moves)).collect::<Vec<_>>().join(", ") }));
    }
    if r.scenarios.is_empty() {
        out.push(match r.safe { Some(true) => "no scenario: the goal is out of reach of the units present at this depth (the target is safe from this side)".to_string(), _ => "no scenario read".to_string() });
    }
    for (i, s) in r.scenarios.iter().enumerate() {
        out.push(format!("scenario {} ({} moves): {}", i + 1, s.total_moves, s.placements.iter().map(|p| format!("{}@{}->{} ({})", p.unit.kind, show_loc(&p.unit.at), show_loc(&p.to), p.moves)).collect::<Vec<_>>().join(", ")));
    }
    for n in &r.notes { out.push(format!("note: {n}")); }
    out
}
