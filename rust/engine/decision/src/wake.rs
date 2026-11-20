//! PLAN 79.05 THE WHOLE WAKE (SYSTEM_SPEC 8 "Wake", "The whole is judged, not the piece"; 4 reasoning by questions; 7 the proposer; 11 Rust in the hot paths). The wake is ONE decision made of the steps of section 8; today only step 5 (the search with the count of worths at its end)
//! plays. This module is that decision's interface and the graph readings already ported into it. Game-free: it reads a game only through `Game` and `Placed`, and a unit, a kind, a place, a move is a number or a name the game gave; nothing of any game is named here.
//! No oracle, no external model, no dependence on the learning crate. Behind `WakeParams`: with every piece off the decision is the old player's (the ledger search), unchanged.
//!
//! RULES OF ENTRY (section 8): the threat list and the goals ORDER what the search reads and say what must be READ IN FULL (`Search::named`: read first, unreduced); they never add to the judge's value (rule 7). The reasons' pick and the heads choose only among the moves the
//! reading holds equal. The wall: no oracle number is read here.
//!
//! INTERFACE (written before the bodies; who calls: the binding `cs_engine.wake_*` and the thin Python callers of the measures).
//!   `Pieces`            which pieces are on. `Pieces::old_player()` all off; `Pieces::all()` all on. A piece not yet ported is listed in `WakeDecision::unported` when asked for, never silently skipped.
//!   `WakeParams`        the clock, the pieces, the share of the clock the deepening has.
//!   `decide`            takes the game, the position, the history (the rules' draw), the judge and the params; returns the `WakeDecision`: the move, the chain of claims with the reason each line ended, what each piece contributed (step, seconds, moves named), the pieces asked for and not ported.
//!   `lines_onto`        (ported, step 3) the lines of a side's units onto a place that exactly ONE unit shuts, that unit, who bears on its place and who holds it (the graph's relations only).
//!   `goal_unit_lines`   (ported, step 3) `lines_onto` for the goal unit of each side, the lines of the side not to move read after the game's pass.
//!   `new_bearer_moves`  (ported, step 3) the legal moves after which a unit of the mover bears on a given place where fewer did before (the bearer a move brings, a discovered line included; no edited position is needed).
use cs_game::placed::{Kind, Place, Placed};
use cs_game::{Game, Role};
use cs_tactics::ledger::{Judge, Search};
use cs_tactics::sheet::base_graph::{PositionPart, Static, UnitRec};
use std::collections::HashSet;

/// Which pieces of the wake are on (section 8's steps): `carried` step 1, `frames` step 2, `threats` and `goals` and `two_threat_goals` step 3, `graph_reads` step 3 (the shut lines and the bearers a move brings), `reasons_pick` and `heads` step 6.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Pieces { pub carried: bool, pub frames: bool, pub threats: bool, pub goals: bool, pub two_threat_goals: bool, pub graph_reads: bool, pub reasons_pick: bool, pub heads: bool }

impl Pieces {
    /// The old player: the search with the count of worths at its end, nothing else.
    pub fn old_player() -> Pieces { Pieces::default() }
    /// Every piece on.
    pub fn all() -> Pieces { Pieces { carried: true, frames: true, threats: true, goals: true, two_threat_goals: true, graph_reads: true, reasons_pick: true, heads: true } }
    /// The names of the pieces asked for that are NOT ported into this module yet (the Python mirrors exist; see docs/ANALYST_PLAN.md 79.05 BUILT).
    pub fn unported(&self) -> Vec<&'static str> {
        let mut v = vec![];
        for (on, name) in [(self.carried, "carried open tactics"), (self.frames, "frames and the question read"), (self.reasons_pick, "reasons' pick"), (self.heads, "heads")] {
            if on { v.push(name); }
        }
        v
    }
}

/// The parameters of one wake decision.
#[derive(Debug, Clone)]
pub struct WakeParams {
    /// The clock of the whole decision, in seconds.
    pub seconds: f64,
    /// The deepest ledger reading (a named cap; the clock is the real bound).
    pub depth: i32,
    /// The pieces that are on.
    pub pieces: Pieces,
    /// The places the graph reads ask about (the places of the unmet facts of the goals; none: the graph reads name no move). Filled by the goals when that piece is ported.
    pub places: Vec<Place>,
}

impl Default for WakeParams { fn default() -> WakeParams { WakeParams { seconds: 0.2, depth: crate::LEDGER_DEPTH, pieces: Pieces::old_player(), places: vec![] } } }

/// One line of the chain of claims: the moves, and the reason it ended (section 4's four reasons, in the words of this build), and the piece that made it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Claim { pub line: Vec<String>, pub ended: String, pub by: &'static str }

/// What one piece contributed to the decision: the step of section 8 it serves, its seconds, the moves it named (read first and unreduced).
#[derive(Debug, Clone, PartialEq)]
pub struct Contribution { pub piece: &'static str, pub step: u8, pub seconds: f64, pub named: Vec<String> }

/// The wake's answer.
#[derive(Debug, Clone)]
pub struct WakeDecision { pub mv: Option<String>, pub value: i64, pub depth: i32, pub nodes: u64, pub chain: Vec<Claim>, pub contributions: Vec<Contribution>, pub unported: Vec<&'static str> }

/// A unit as the graph names it: owner, kind, place (numbers the game gave).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct UnitRef { pub owner: Role, pub kind: Kind, pub at: Place }

fn uref(u: &UnitRec) -> UnitRef { UnitRef { owner: u.owner, kind: u.kind, at: u.at } }

/// A line of a unit onto a place that exactly one unit shuts: the unit, the blocker, the units on the other side of the blocker that bear on its place, the units on its side that bear on it (the blocker itself aside).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LineOnto { pub unit: UnitRef, pub blocker: UnitRef, pub bearers: Vec<UnitRef>, pub holders: Vec<UnitRef> }

/// The lines of `side`'s units onto `place` that exactly one unit shuts, read from a position's graph. The graph computes `shuts` for the side to move, so for the other side the caller hands the position after the game's pass (`goal_unit_lines`).
pub fn lines_onto(part: &PositionPart, place: Place, side: Role) -> Vec<LineOnto> {
    let bears: HashSet<(usize, Place)> = part.bears.iter().map(|r| (r.unit, r.place)).collect();
    let mut out = vec![];
    for r in &part.shuts {
        if r.place != place || part.units[r.unit].owner != side || r.between.len() != 1 { continue; }
        let bi = r.between[0];
        let b = &part.units[bi];
        let mut bearers: Vec<UnitRef> = part.units.iter().enumerate().filter(|(j, u)| u.owner != b.owner && bears.contains(&(*j, b.at))).map(|(_, u)| uref(u)).collect();
        let mut holders: Vec<UnitRef> = part.units.iter().enumerate().filter(|(j, u)| u.owner == b.owner && *j != bi && bears.contains(&(*j, b.at))).map(|(_, u)| uref(u)).collect();
        bearers.sort();
        holders.sort();
        out.push(LineOnto { unit: uref(&part.units[r.unit]), blocker: uref(b), bearers, holders });
    }
    out.sort_by_key(|l| (l.unit, l.blocker));
    out
}

/// For the goal unit (the kind `goal`) of each role: (its owner, its place, the lines of the other side onto it that exactly one unit shuts, or None where the game gives no pass and the other side is not to move).
pub fn goal_unit_lines<G: Game + Placed>(st: &Static, g: &G, s: &G::State, goal: Kind) -> Vec<(Role, Place, Option<Vec<LineOnto>>)> {
    let part = st.position(g, s);
    let passed = g.pass(s).map(|p| st.position(g, &p));
    let mut out = vec![];
    for u in part.units.iter().filter(|u| u.kind == goal) {
        let attacker = (0..g.roles().len()).find(|r| *r != u.owner);
        let lines = attacker.and_then(|a| if part.mover == Some(a) { Some(lines_onto(&part, u.at, a)) } else { passed.as_ref().map(|pp| lines_onto(pp, u.at, a)) });
        out.push((u.owner, u.at, lines));
    }
    out.sort_by_key(|x| x.0);
    out
}

/// How many units of `owner` bear on each of `places` in `s` (the graph's `bear` relation is `Placed::bearing_from` of each unit, so this is the same count without building the graph).
fn bearer_counts<G: Game + Placed>(g: &G, s: &G::State, owner: Role, places: &[Place]) -> Vec<usize> {
    let mut n = vec![0usize; places.len()];
    for (o, k, at) in g.cells(s) {
        if o != owner { continue; }
        let b = g.bearing_from(s, o, k, at);
        for (i, p) in places.iter().enumerate() { if b.contains(p) { n[i] += 1; } }
    }
    n
}

/// The legal moves of the mover after which more units of the mover bear on one of `places` than before: (the move's name, the places it adds a bearer on). A bearer the move brings and a line it opens both count: who bears is read after the move (the relation of each unit, as the graph
/// builds it, without building the graph: PLAN 79.07 made it cheap, the answer is the same). This needs no edited position, so a bearer brought by a move that attacks the other side's goal unit is read like any other.
pub fn new_bearer_moves<G: Game + Placed>(_st: &Static, g: &G, s: &G::State, places: &[Place]) -> Vec<(String, Vec<Place>)> {
    let Some(mover) = g.mover(s) else { return vec![] };
    if places.is_empty() { return vec![]; }
    let before = bearer_counts(g, s, mover, places);
    let mut out = vec![];
    for m in g.moves(s) {
        let after = bearer_counts(g, &g.apply(s, &m), mover, places);
        let added: Vec<Place> = places.iter().zip(&before).zip(&after).filter(|((_, b), a)| a > b).map(|((p, _), _)| *p).collect();
        if !added.is_empty() { out.push((g.move_name(s, &m), added)); }
    }
    out
}

/// THE WHOLE DECISION (see the module doc). Ported so far: the graph reads (`new_bearer_moves` over `params.places`) that name moves read first and unreduced; the search is `cs_tactics::ledger::Search`. A piece asked for and not ported is listed in `unported`.
pub fn decide<G: Game + Placed, J: Judge>(st: &Static, g: &G, s: &G::State, history: &[u64], judge: &J, params: &WakeParams) -> WakeDecision {
    let me = g.mover(s).unwrap_or(0);
    let mut contributions = vec![];
    let mut named: Vec<String> = vec![];
    if params.pieces.graph_reads && !params.places.is_empty() {
        let t0 = std::time::Instant::now();
        named = new_bearer_moves(st, g, s, &params.places).into_iter().map(|(m, _)| m).collect();
        contributions.push(Contribution { piece: "graph reads: bearers a move brings", step: 3, seconds: t0.elapsed().as_secs_f64(), named: named.clone() });
    }
    let t0 = std::time::Instant::now();
    let mut ls = Search::new(g, judge, me, params.seconds);
    ls.history = history.iter().copied().collect();
    if !named.is_empty() { ls.named.insert(g.key(s), named); }
    let ln = ls.run(s, params.depth);
    contributions.push(Contribution { piece: "search with the count of worths at its end", step: 5, seconds: t0.elapsed().as_secs_f64(), named: vec![] });
    let ended = if ln.value.abs() >= 1_000_000 { "the game itself ends in the line (concrete)".to_string() } else { format!("the settled end: no capture is left to either side after the depth read in full ({} plies)", ln.depth) };
    WakeDecision { mv: ln.first.clone(), value: ln.value, depth: ln.depth, nodes: ln.nodes, chain: vec![Claim { line: ln.line.clone(), ended, by: "search" }], contributions, unported: params.pieces.unported() }
}

// ------------------------------------------------------------------------------------------------------------------------------ THE INTERFACES OF THE PIECES STILL TO BE PORTED (written before their bodies; each body answers `Unported`)
// Each is the Rust form of a Python piece that stays as its mirror (docs/ANALYST_PLAN.md 79.05 BUILT lists size and order). Section 8's steps: 1 carried tactics, 2-3 frames and questions, the threat list, the proposer, 6 the pick.

/// A piece asked for and not ported yet: its name.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Unported(pub &'static str);

/// STEP 3, the threat list (PLAN 79.06, ported): the rows, the list and `concrete_threats` live in `crate::threats` (one piece per module) and are named here as the wake's interface. Their types were declared here thin (`unit: UnitRef`, `responses: Vec<String>`) and are widened there with the reason stated in the 79.06 BUILT block: the Python row carries the line, outcome, verdict and kept responses a proposer reads.
pub use crate::threats::{concrete_threats, ConcreteList, ThreatRow};

/// STEP 3, the proposer (PLAN 79.07, ported): the facts, causes, fixes and goals, the goals of two threats, the moves that make two threats at once and `named_moves` live in `crate::goals` (one piece per module) and are named here as the wake's interface. Their types were declared here thin
/// (`Fix`, `Fact`, `Goal` with strings and a count of moves) and are widened there with the reason stated in the 79.07 BUILT block: a fix must carry the legal moves that do it, a fact its cause and the units it is about, a goal its side, source and the fix set.
pub use crate::goals::{double_threats, goals, named_moves, propose, two_threat_goals, DoubleThreat, Fact, Fix, FixKind, Goal, GoalParams, Proposal, TwoThreats};

/// STEP 6, the reasons' pick among the moves the search holds equal (symbols_play.ReasonsReader.pick: the pattern counter is Rust already; the weights are a JSON record).
pub fn reasons_pick<G: Game + Placed>(_st: &Static, _g: &G, _s: &G::State, _equal: &[String]) -> Result<Option<String>, Unported> { Err(Unported("reasons' pick")) }

/// STEP 6, the heads read at the equal moves (heads_read.py: the kinds, the 26 statements and the concepts of the position after each move, then the fitted head from the record).
pub fn heads_pick<G: Game + Placed>(_st: &Static, _g: &G, _s: &G::State, _equal: &[String]) -> Result<Option<String>, Unported> { Err(Unported("heads")) }

/// STEP 1, the open tactics carried from the last decision (threat_concrete.same_read_set: a claim whose read set did not change is kept).
pub fn carry(_previous: &ConcreteList, _now_read_sets: &[Vec<Place>]) -> Result<ConcreteList, Unported> { Err(Unported("carried open tactics")) }
