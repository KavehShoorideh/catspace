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
use crate::log::{DecisionLog, GoalLog, MoveLog, RowLog};
use crate::pick::MoveReasons;
use crate::rule;
use crate::threat_verdict::Record;
use crate::threats::{RespState, ThreatParams};
use cs_tactics::ledger::{Judge, Search, WIN};
use cs_tactics::lines::LineParams;
use cs_tactics::sheet::base_graph::{PositionPart, Static, UnitRec};
use std::collections::HashSet;

/// Which pieces of the wake are on (section 8's steps): `carried` step 1, `frames` step 2, `threats` and `goals` and `two_threat_goals` step 3, `graph_reads` step 3 (the shut lines and the bearers a move brings), `reasons_pick` and `heads` step 6, `after_move` step 4 (PLAN 79.10 A).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Pieces { pub carried: bool, pub frames: bool, pub threats: bool, pub goals: bool, pub two_threat_goals: bool, pub graph_reads: bool, pub reasons_pick: bool, pub heads: bool,
    /// PLAN 79.10 A: their goals are read after each move of ours (`after`), their fixes named at that child position, the fact about the move logged
    pub after_move: bool }

impl Pieces {
    /// The old player: the search with the count of worths at its end, nothing else.
    pub fn old_player() -> Pieces { Pieces::default() }
    /// Every piece on.
    pub fn all() -> Pieces { Pieces { carried: true, frames: true, threats: true, goals: true, two_threat_goals: true, graph_reads: true, reasons_pick: true, heads: true, after_move: true } }
    /// The names of the pieces asked for that are NOT ported: only the heads that read counts summed over the whole position, which the owner's decision of 2026-11-19 (SYSTEM_SPEC 7, point 2) says not to port and not to call; the estimates of the wake are per goal (estimate.rs) and per move (pick.rs).
    pub fn unported(&self) -> Vec<&'static str> {
        let mut v = vec![];
        if self.heads { v.push("heads over whole-position counts (not ported, not called: SYSTEM_SPEC 7)"); }
        v
    }
}

/// What the whole wake reads besides the position (all of it data the caller gives; none of it learned here): the measured record of worths, the goal kind, the budgets, the learned seeds of the frame read, the last decision's open tactics, the table of the chance of a goal, the reasons and their weights.
#[derive(Debug, Clone)]
pub struct WakeInputs {
    pub record: Record, pub goal: Option<Kind>,
    /// the budget of questions of the proposer (0: `goals::BUDGET`) and of line reads for naming moves (0: `rule::NAMED_BUDGET`)
    pub budget: usize, pub named_budget: usize,
    /// the frame read's learned seeds and parameters (None: the threat list's rows come from the graph)
    pub frame: Option<FrameRead>,
    /// the open tactics of the last decision (None: nothing to carry)
    pub previous: Option<Carried>,
    /// the table of the chance of a goal by what stops it, and the reasons with their weights (None: no estimate of that kind, and the log says so)
    pub chances: Option<ChanceTable>, pub reasons: Option<Reasons>,
    /// the lines read from one position share a table (repair 2)
    pub share_table: bool,
    /// the share of the clock the search keeps whatever the steps before it cost, and from which the equal reading's clock follows (None: `LEDGER_SHARE`); the old player's share when the two are compared at one clock
    pub ledger_share: Option<f64>,
    /// PLAN 79.09 B: the share of the clock the steps before the search (1 to 4) may take together (None: what the search and its equal reading do not keep, less `TAIL_SHARE`: `pre_share_of`). The steps ask in order of stake and stop at it; what they did not ask is listed.
    pub pre_share: Option<f64>,
}

impl Default for WakeInputs { fn default() -> WakeInputs { WakeInputs { record: Record::default(), goal: None, budget: 0, named_budget: 0, frame: None, previous: None, chances: None, reasons: None, share_table: true, ledger_share: None, pre_share: None } } }

/// The parameters of one wake decision.
#[derive(Debug, Clone)]
pub struct WakeParams {
    /// The clock of the whole decision, in seconds.
    pub seconds: f64,
    /// The deepest ledger reading (a named cap; the clock is the real bound).
    pub depth: i32,
    /// The pieces that are on.
    pub pieces: Pieces,
    /// The places the graph reads ask about (the places of the unmet facts of the goals; none: the graph reads name no move). Used by the search-only path.
    pub places: Vec<Place>,
    /// What the whole wake reads (see `WakeInputs`).
    pub inputs: WakeInputs,
}

impl Default for WakeParams { fn default() -> WakeParams { WakeParams { seconds: 0.2, depth: crate::LEDGER_DEPTH, pieces: Pieces::old_player(), places: vec![], inputs: WakeInputs::default() } } }

/// One line of the chain of claims: the moves, and the reason it ended (section 4's four reasons, in the words of this build), and the piece that made it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Claim { pub line: Vec<String>, pub ended: String, pub by: &'static str }

/// What one piece contributed to the decision: the step of section 8 it serves, its seconds, the moves it named (read first and unreduced).
#[derive(Debug, Clone, PartialEq)]
pub struct Contribution { pub piece: &'static str, pub step: u8, pub seconds: f64, pub named: Vec<String>, pub note: String }

/// The wake's answer: the move, the search's value, depth and nodes, the chain of claims with the reason each line ended, what each piece contributed (step, seconds, moves named), the pieces asked for and not ported, and, for the whole wake, the decision log, the proposal (threat list and goals), the
/// open tactics to carry to the next decision, the moves named to the search (ours at the root, theirs at the ply after ours) and the moves the search held equal.
#[derive(Debug, Clone)]
pub struct WakeDecision {
    pub mv: Option<String>, pub value: i64, pub depth: i32, pub nodes: u64, pub chain: Vec<Claim>, pub contributions: Vec<Contribution>, pub unported: Vec<&'static str>,
    pub log: DecisionLog, pub proposal: Option<Proposal>, pub carried: Option<Carried>, pub named_mine: Vec<String>, pub named_theirs: Vec<String>, pub equal: Vec<String>,
}

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
fn decide_search<G: Game + Placed, J: Judge>(st: &Static, g: &G, s: &G::State, history: &[u64], judge: &J, params: &WakeParams) -> WakeDecision {
    let me = g.mover(s).unwrap_or(0);
    let mut contributions = vec![];
    let mut named: Vec<String> = vec![];
    if params.pieces.graph_reads && !params.places.is_empty() {
        let t0 = std::time::Instant::now();
        named = new_bearer_moves(st, g, s, &params.places).into_iter().map(|(m, _)| m).collect();
        contributions.push(Contribution { piece: "graph reads: bearers a move brings", step: 3, seconds: t0.elapsed().as_secs_f64(), named: named.clone(), note: String::new() });
    }
    let t0 = std::time::Instant::now();
    let mut ls = Search::new(g, judge, me, params.seconds);
    ls.history = history.iter().copied().collect();
    if !named.is_empty() { ls.named.insert(g.key(s), named); }
    let ln = ls.run(s, params.depth);
    contributions.push(Contribution { piece: "search with the count of worths at its end", step: 5, seconds: t0.elapsed().as_secs_f64(), named: vec![], note: String::new() });
    let ended = if ln.value.abs() >= 1_000_000 { "the game itself ends in the line (concrete)".to_string() } else { format!("the settled end: no capture is left to either side after the depth read in full ({} plies)", ln.depth) };
    WakeDecision { mv: ln.first.clone(), value: ln.value, depth: ln.depth, nodes: ln.nodes, chain: vec![Claim { line: ln.line.clone(), ended, by: "search" }], contributions, unported: params.pieces.unported(), log: DecisionLog::default(), proposal: None, carried: None, named_mine: vec![], named_theirs: vec![], equal: vec![] }
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
pub use crate::goals::{answers_against, goals, named_moves, propose, shared_fix_moves, two_threat_goals, Answer, SharedMove, Fact, Fix, FixKind, Goal, GoalParams, Proposal, TwoThreats};

/// STEP 6, the reasons' pick (PLAN 79.08, ported): `pick::scores` and `pick::pick` (the pattern counter is Rust; the weights are a parameter). STEP 1, the carry (PLAN 79.08, ported): `carry::carry`. STEP 2, the frame read as the source of the rows: `frame_rows::frame_read`. The estimates: `estimate::ChanceTable`.
pub use crate::carry::{carry, Carried, Carry};
pub use crate::estimate::{ChanceTable, Estimate};
pub use crate::frame_rows::{frame_read, FrameRead, FrameRows};
pub use crate::pick::{pick, scores, Reasons};

// ------------------------------------------------------------------------------------------------------------------------------ THE WHOLE DECISION (PLAN 79.08 C)

/// The share of the decision's clock that the equal reading keeps back from the search, as the old player keeps it (`LEDGER_SHARE`, `TIE_SHARE`): the search is never given less than the old player's share of the whole clock, whatever the steps before it cost.
fn tie_seconds(total: f64, share: f64) -> f64 { (total * (1.0 - share) * crate::TIE_SHARE).max(0.01) }

/// PLAN 79.09 B (SYSTEM_SPEC 7, the budget; piece: the wake): THE CLOCK IS HELD. The whole decision's clock is shared as: the search keeps `ledger_share`, the equal reading keeps what the old player gives it (`TIE_SHARE` of the rest), the choice and the log keep `TAIL_SHARE`, and what is left is the share of
/// the steps before the search (`pre_share_of`). A named share of the clock, not a number tuned to a position.
pub const TAIL_SHARE: f64 = 0.08;
/// Of what the steps before the search may take: the frame read at most `FRAMES_OF_PRE` of what is left, the threat list and goals at most `LIST_OF_PRE` of what is then left; the reading of the named moves has the rest.
pub const FRAMES_OF_PRE: f64 = 0.3;
pub const LIST_OF_PRE: f64 = 0.7;
/// PLAN 79.10 A: of what is left of the steps before the search, their goals after our moves may take at most this much (the reading of the named moves has the rest).
pub const AFTER_OF_PRE: f64 = 0.5;

/// The share of the clock the steps before the search may take.
pub fn pre_share_of(share: f64, given: Option<f64>) -> f64 { given.unwrap_or(((1.0 - share) * (1.0 - crate::TIE_SHARE) - TAIL_SHARE).max(0.0)) }

/// The instant a step may run to: `frac` of what is left of the time up to `end`.
fn until(end: std::time::Instant, frac: f64) -> std::time::Instant { let now = std::time::Instant::now(); if end <= now { now } else { now + (end - now).mul_f64(frac) } }

/// THE WHOLE DECISION, by the steps of SYSTEM_SPEC 8 and the rule for choosing a move (rule.rs): (1) the carry, (2) the frames and the rows of the threat list, (3) the threat list and the goals of both sides in order of stake within the budget, (4) THEIR goals that work: our answers; OUR goal with the fewest things in its way:
/// its fixes; only the moves whose own line was read and not refuted are named, (5) the search reads with the named moves first and in full, ours at the root and theirs at the ply after ours; the judge's value is untouched, (6) among the moves the search holds equal: an answer to their largest standing threat, then a fix of
/// our chosen goal, then the estimates, (7) the chain of claims and the decision log. With no piece of the whole on this is `decide_search`, the old player. The clock: steps 1 to 4 take what they take (their cost is reported by step); the search is given what is left of `params.seconds` and never less than the old player's share of it.
pub fn decide<G: Game + Placed, J: Judge>(st: &Static, g: &G, s: &G::State, history: &[u64], judge: &J, params: &WakeParams) -> WakeDecision {
    let p = params.pieces;
    if !(p.carried || p.frames || p.threats || p.goals || p.two_threat_goals || p.reasons_pick) { return decide_search(st, g, s, history, judge, params); }
    let t_all = std::time::Instant::now();
    let me = g.mover(s).unwrap_or(0);
    let other = g.other(me);
    let inp = &params.inputs;
    let rec = &inp.record;
    let (kinds, roles, places) = (g.kind_names(), g.roles(), g.place_names());
    let share = inp.ledger_share.unwrap_or(crate::LEDGER_SHARE);
    let pre_end = t_all + std::time::Duration::from_secs_f64(params.seconds * pre_share_of(share, inp.pre_share));
    let wall_end = t_all + std::time::Duration::from_secs_f64(params.seconds);
    let text = |u: &UnitRef| format!("{} {} on {}", roles[u.owner], kinds[u.kind as usize], places[u.at as usize]);
    let mut contributions: Vec<Contribution> = vec![];
    let mut log = DecisionLog::default();
    // ---- steps 1 and 2: the frames and the carry
    let mut frames: Option<FrameRows> = None;
    if p.frames {
        match &inp.frame {
            Some(fr) => { let mut fr = fr.clone(); fr.params.deadline = Some(until(pre_end, FRAMES_OF_PRE)); let f = frame_read(st, g, s, inp.goal, &fr, None); contributions.push(Contribution { piece: "frames and the question read", step: 2, seconds: f.seconds, named: vec![], note: format!("{} frames, {} rows, {} questions, {} frames not reached by the clock", f.frames, f.rows.len(), f.questions, f.unread_frames) }); frames = Some(f); }
            None => log.notes.push("frames asked for and no learned seeds given: the rows of the threat list are the graph's".into()),
        }
    }
    let mut carried_out: Option<Carry> = None;
    if p.carried {
        let t0 = std::time::Instant::now();
        match (&inp.previous, &frames) {
            (Some(prev), Some(f)) => { let c = carry(prev, f); contributions.push(Contribution { piece: "carried open tactics", step: 1, seconds: t0.elapsed().as_secs_f64(), named: vec![], note: format!("{} rows held, {} read again", c.held.len(), c.read.len()) }); carried_out = Some(c); }
            (None, _) => log.notes.push("carry asked for and there is no last decision: nothing is carried".into()),
            (_, None) => log.notes.push("carry asked for and the frame read gave no read sets: nothing is carried".into()),
        }
    }
    // ---- step 3: the threat list and the goals
    let t0 = std::time::Instant::now();
    let rows = frames.as_ref().map(|f| carried_out.as_ref().map_or_else(|| f.rows.clone(), |c| c.read.clone()));
    let tp = ThreatParams { goal: inp.goal, rows, share_table: inp.share_table, deadline: Some(until(pre_end, LIST_OF_PRE)), ..Default::default() };
    let gp = GoalParams { budget: if inp.budget > 0 { inp.budget } else { crate::goals::BUDGET }, held: carried_out.as_ref().map(|c| c.held.clone()).unwrap_or_default(), ..Default::default() };
    let prop = propose(st, g, s, rec, &tp, &gp);
    contributions.push(Contribution { piece: "threat list of both sides", step: 3, seconds: prop.list.seconds(), named: vec![], note: format!("{} rows against us, {} against them, {} dropped, {} questions; not asked: {}", prop.list.mover.len(), prop.list.other.len(), prop.list.dropped.len(), prop.questions, if prop.list.not_asked.is_empty() { "nothing".to_string() } else { prop.list.not_asked.join("; ") }) });
    contributions.push(Contribution { piece: "goals, subgoals, answers to their goals", step: 3, seconds: t0.elapsed().as_secs_f64() - prop.list.seconds(), named: vec![], note: format!("{} goals", prop.goals.len()) });
    // ---- step 4: what is named
    let t0 = std::time::Instant::now();
    let their = rule::their_threats(&prop.list);
    let mut answers_largest: Vec<String> = vec![];
    let mut mine: Vec<String> = vec![];
    let put = |v: &mut Vec<String>, m: &String| { if !v.contains(m) { v.push(m.clone()); } };
    for (k, r) in their.iter().enumerate() {
        for x in r.responses.iter().filter(|x| x.state == RespState::Kept) { if let Some(n) = x.names.first() { put(&mut mine, n); if k == 0 { put(&mut answers_largest, n); } } }
    }
    // PLAN 79.09 A: for each side the fixes are named chain by chain in order of stake, the nearest goal of each chain
    let stakes = rule::Stakes::read(g, s, rec, &prop.goals, inp.goal);
    let chains_mine = rule::named_fixes(&prop.goals, me, &stakes);
    let chains_theirs = rule::named_fixes(&prop.goals, other, &stakes);
    let count = if inp.named_budget > 0 { inp.named_budget } else { rule::NAMED_BUDGET };
    for m in rule::names_of(&chains_mine, count) { put(&mut mine, &m); }
    for (k, r) in their.iter().enumerate() {
        let Some(u) = r.unit else { continue };
        for (t, ans) in &prop.defence { if *t == u { for a in ans { put(&mut mine, &a.mv); if k == 0 { put(&mut answers_largest, &a.mv); } } } }
    }
    let mut theirs: Vec<String> = vec![];
    // PLAN 79.10 A: with `after_move` their fixes are named at each child position (below), not at the ply after ours on a pass
    if !p.after_move { for m in rule::names_of(&chains_theirs, count) { put(&mut theirs, &m); } }
    for r in prop.list.other.iter().filter(|r| r.verdict == crate::threat_verdict::VerdictKind::Confirmed && !r.two_moves) {
        for x in r.responses.iter().filter(|x| x.state == RespState::Kept) { if let Some(n) = x.names.first() { put(&mut theirs, n); } }
    }
    let budget = count;
    // PLAN 79.10 A: their goals on the position after each move of ours, in the search's own order, until the clock of the step; their fixes named at that child
    let mut after = crate::after::AfterRead::default();
    if p.after_move {
        let order = Search::new(g, judge, me, 0.1).deepening_order(s);
        after = crate::after::their_goals_after(st, g, s, rec, inp.goal, &order, budget, Some(until(pre_end, AFTER_OF_PRE)));
        contributions.push(Contribution { piece: "their goals after each move of ours", step: 4, seconds: after.seconds, named: vec![], note: format!("{} positions read, {} not reached by the clock", after.rows.len(), after.not_asked.len()) });
    }
    let kept = rule::keep_read_by(g, s, rec, &mine, &theirs, budget, LineParams { depth: tp.sweep_depth, fuse: tp.sweep_fuse }, inp.share_table, Some(pre_end));
    let named_mine: Vec<String> = kept.mine.iter().map(|k| k.name.clone()).collect();
    let named_theirs: Vec<String> = kept.theirs.iter().map(|k| k.name.clone()).collect();
    // PLAN 79.09 C: a goal is the subgoal of the moment only if a named move serves it: the first chain (in order of stake) whose nearest goal has a move that was named
    let chosen_chain = chains_mine.iter().position(|c| c.all_moves().iter().any(|m| named_mine.contains(m)));
    let chosen_ix = chosen_chain.map(|k| chains_mine[k].goal);
    if chosen_ix.is_none() && !chains_mine.is_empty() { log.notes.push("no goal of ours is the subgoal of the moment: no move that fixes one was named (refuted or not read)".into()); }
    contributions.push(Contribution { piece: "named moves: their lines read, the refuted left out", step: 4, seconds: t0.elapsed().as_secs_f64(), named: named_mine.iter().chain(named_theirs.iter()).cloned().collect(), note: format!("{} lines read, {} refuted, {} not read (over the budget)", kept.read, kept.refuted.len(), kept.unlisted.len()) });
    // ---- step 5: the search, with the named moves first and in full
    let t0 = std::time::Instant::now();
    // the search keeps its share of the clock and gets what the steps before it did not spend of theirs
    let search_secs = params.seconds * share + pre_end.saturating_duration_since(std::time::Instant::now()).as_secs_f64();
    let mut ls = Search::new(g, judge, me, search_secs);
    ls.history = history.iter().copied().collect();
    ls.verify_ties = true;
    if !named_mine.is_empty() { ls.named.insert(g.key(s), named_mine.clone()); }
    if !named_theirs.is_empty() { ls.named_ply.insert(1, named_theirs.clone()); }
    for r in &after.rows { if !r.names.is_empty() { ls.named.insert(r.key, r.names.clone()); } }
    let ln = ls.run(s, params.depth);
    let moves = g.moves(s);
    let own: Option<usize> = ln.first_index;
    let decided = ln.value.abs() >= WIN - 1000;
    let mut probes: Vec<(usize, i64)> = vec![];
    let mut equal_ix: Vec<usize> = vec![];
    if !decided && own.is_some() {
        ls.resume(tie_seconds(params.seconds, share));
        probes = ls.probes(s, ln.depth, ln.value, None);
        let first: Vec<usize> = probes.iter().filter(|(_, v)| *v >= ln.value).map(|(i, _)| *i).collect();
        equal_ix = if ls.cut() || first.is_empty() { if ls.cut() { vec![] } else { first } } else { ls.verify(s, ln.depth, ln.value, &first) };
    }
    if let Some(o) = own { equal_ix.retain(|i| *i != o); equal_ix.insert(0, o); }
    let equal: Vec<String> = equal_ix.iter().map(|i| g.move_name(s, &moves[*i])).collect();
    contributions.push(Contribution { piece: "search with the count of worths at its end, and the equal reading", step: 5, seconds: t0.elapsed().as_secs_f64(), named: vec![], note: format!("depth {}, {} nodes, {} held equal", ln.depth, ls.nodes, equal.len()) });
    // ---- step 6: among the moves held equal
    let t0 = std::time::Instant::now();
    let all_names: Vec<String> = moves.iter().map(|m| g.move_name(s, m)).collect();
    let sc: Option<Vec<MoveReasons>> = match (&inp.reasons, p.reasons_pick) {
        // the moves held equal are scored first (they decide); the others only while the clock of the whole decision lasts (they are logged); one not reached has no score
        (Some(r), true) => { let order: Vec<String> = equal.iter().cloned().chain(all_names.iter().filter(|n| !equal.contains(n)).cloned()).collect(); Some(crate::pick::scores_by(st, g, s, inp.goal, r, &order, Some(wall_end))) }
        (None, true) => { log.notes.push("the reasons' pick asked for and no reasons given: no estimate of a move, the search's own first move decides among the rest".into()); None }
        _ => None,
    };
    let tie_digits = inp.reasons.as_ref().map_or(9, |r| r.tie_digits);
    let (fixes, fix_chain) = rule::fixes_among(&chains_mine, &equal);
    // the estimates decide only if every move held equal was scored
    let scored_equal = sc.as_ref().is_some_and(|v| equal.iter().all(|e| v.iter().any(|x| &x.mv == e)));
    if sc.as_ref().is_some_and(|v| v.len() < all_names.len()) { log.notes.push(format!("the clock ended before the reasons' score of {} moves: they have none", all_names.len() - sc.as_ref().map_or(0, |v| v.len()))); }
    if sc.is_some() && !scored_equal { log.notes.push("the clock ended before every move held equal was scored: the estimates do not decide".into()); }
    let chosen = rule::choose(&equal, &answers_largest, &fixes, &|set: &[String]| if scored_equal { sc.as_ref().and_then(|v| crate::pick::pick(v, set, tie_digits)) } else { None });
    contributions.push(Contribution { piece: "the choice among the moves held equal (the rule, the estimates)", step: 6, seconds: t0.elapsed().as_secs_f64(), named: vec![], note: String::new() });
    let mv = chosen.as_ref().map(|c| c.mv.clone()).or_else(|| ln.first.clone());
    // ---- step 7: the chain of claims and the log
    // THE CLAIM THAT DECIDED THE MOVE (PLAN 79.09 C): the goals the move is a fix of, the goal of the largest stake first
    let serves_of = |name: &String| -> Vec<(f64, usize, String)> {
        let mut v: Vec<(f64, usize, String)> = vec![];
        for (i, gl) in prop.goals.iter().enumerate().filter(|(_, x)| x.side == me) {
            for f in &gl.facts { if f.fixes.iter().any(|fx| fx.moves.contains(name)) {
                let w = format!("goal on {} ({}){}", text(&gl.target), f.name, gl.parent.as_ref().map_or(String::new(), |par| format!(", the stopper of the goal on {}", text(par))));
                if !v.iter().any(|x| x.2 == w) { v.push((rule::stake_of(&prop.goals, i, &stakes), i, w)); }
            } }
        }
        v.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal).then(rule::things_in_the_way(&prop.goals[a.1]).cmp(&rule::things_in_the_way(&prop.goals[b.1]))));
        v
    };
    let mut decided_by: Option<rule::Decided> = None;
    let nonempty = |x: &str, fallback: &str| if x.trim().is_empty() { fallback.to_string() } else { x.to_string() };
    let mut chain: Vec<Claim> = vec![];
    if let (Some(c), Some(m)) = (&chosen, &mv) {
        let own = kept.mine.iter().find(|k| &k.name == m);
        let standing: Option<String> = their.first().map(|r| r.unit.map_or_else(|| "the end of the game forced against us".to_string(), |u| text(&u)));
        let answers = answers_largest.contains(m);
        let sv: Vec<String> = serves_of(m).into_iter().map(|x| x.2).collect();
        let (kind, why) = rule::why_of(&rule::WhyIn { chosen: c, value: ln.value, depth: ln.depth, own, standing: standing.clone(), answers, serves: sv });
        decided_by = Some(kind);
        // the search's own line of the move chosen (the table's line from it, not the line of its first move when another equal move is played)
        let (mline, _) = if Some(m) == ln.first.as_ref() { (ln.line.clone(), vec![]) } else { moves.iter().position(|x| &g.move_name(s, x) == m).map_or((vec![], vec![]), |i| ls.line_from(s, i, ln.depth.max(1) as usize + 2)) };
        let search_end = if decided { format!("the game itself ends in the line (concrete): {}", if ln.value > 0 { "won by us" } else { "lost by us" }) } else { format!("the settled end: no capture is left to either side after the depth read in full ({} plies)", ln.depth) };
        let by_search = if decided { "search: the end of the game forced" } else { "search" };
        chain.push(Claim { line: mline, ended: search_end, by: by_search });
        match kind {
            rule::Decided::OwnGain => if let Some(k) = own { chain.push(Claim { line: k.line.clone(), ended: nonempty(&k.ended, "its own line was read; how it ended was not recorded"), by: "its own line read: it wins by its own reading" }); },
            rule::Decided::AnswersThreat => {
                if let Some(x) = their.first().and_then(|r| r.responses.iter().find(|x| x.names.first() == Some(m))) { chain.push(Claim { line: x.names.clone(), ended: nonempty(&x.end_reason, &x.why), by: "threat list: an answer to a threat of theirs that stands" }); }
                else if let Some(k) = own { chain.push(Claim { line: k.line.clone(), ended: nonempty(&k.ended, "its own line was read; how it ended was not recorded"), by: "threat list: an answer to a threat of theirs that stands" }); }
                else { chain.push(Claim { line: vec![m.clone()], ended: "the move is among the answers the goal of theirs names; its own line was not read, so no end is claimed".into(), by: "threat list: an answer to a threat of theirs that stands" }); }
            }
            rule::Decided::FixesGoal => { if let Some(k) = own { chain.push(Claim { line: k.line.clone(), ended: nonempty(&k.ended, "its own line was read; how it ended was not recorded"), by: "goal: a fix of a goal" }); } }
            rule::Decided::Estimate => chain.push(Claim { line: vec![m.clone()], ended: "the reasons' score decides among the moves held equal (an estimate, never a value)".into(), by: "estimate: the reasons' pick" }),
            _ => {}
        }
        log.why = why;
    } else { log.why = "no move held equal: the search's own move".to_string(); chain.push(Claim { line: ln.line.clone(), ended: if decided { "the game itself ends in the line (concrete)".to_string() } else { format!("the settled end: no capture is left to either side after the depth read in full ({} plies)", ln.depth) }, by: "search" }); }
    let _ = (decided_by, fix_chain);
    log.chosen = mv.clone();
    let goal_log = |x: &Goal, chosen: bool| GoalLog {
        side: roles[x.side].clone(), target: text(&x.target), source: x.source.clone(), depth: x.depth, parent: x.parent.as_ref().map(&text), via: x.via.map(|v| v.to_string()), unmet: x.unmet.iter().map(|u| u.to_string()).collect(), stoppers: x.stoppers.iter().map(&text).collect(),
        fixes: x.fixes.iter().map(|f| (format!("{:?}", f.kind), f.moves.iter().take(8).cloned().collect())).collect(), in_the_way: rule::things_in_the_way(x),
        chance: inp.chances.as_ref().map(|t| { let e = t.chance(x); (e.chance, e.n, e.source.to_string()) }), chosen, goal_unit: x.goal_unit,
    };
    log.goals = prop.goals.iter().enumerate().map(|(i, x)| goal_log(x, Some(i) == chosen_ix)).collect();
    if inp.chances.is_none() && p.goals { log.notes.push("no table of the chance of a goal given: no estimate per goal".into()); }
    log.notes.push("no table of frames given: no estimate per frame".into());
    log.chosen_goal = chosen_ix.map(|i| goal_log(&prop.goals[i], true));
    if let Some(i) = chosen_ix { let mut x = &prop.goals[i]; while let Some(par) = x.parent { match prop.goals.iter().find(|y| y.target == par && y.side == x.side) { Some(y) => x = y, None => break } } if !std::ptr::eq(x, &prop.goals[i]) { log.far_goal = Some(format!("{} ({})", text(&x.target), x.unmet.join(", "))); } }
    log.their_threat = their.first().map(|r| r.unit.map_or_else(|| "the end of the game".to_string(), |u| text(&u)));
    let row_log = |name: &str, r: &ThreatRow| RowLog { list: name.to_string(), unit: r.unit.map_or_else(|| "the end of the game".to_string(), |u| text(&u)), class: r.class.clone(), verdict: format!("{:?}", r.verdict), net: r.net, ended: r.ended.to_string(), line: r.line.clone(),
        responses: r.responses.iter().map(|x| (x.names.first().cloned().unwrap_or_default(), x.state.word().to_string(), x.net)).collect(), carried: carried_out.as_ref().is_some_and(|c| c.held.iter().any(|h| h.row.unit == r.unit && r.unit.is_some())) };
    for (n, v) in [("against us", &prop.list.mover), ("against them", &prop.list.other), ("exchange", &prop.list.exchange), ("dropped", &prop.list.dropped)] { log.rows.extend(v.iter().take(12).map(|r| row_log(n, r))); }
    log.named_mine = kept.mine.iter().map(|k| (k.name.clone(), format!("{:?}", k.verdict), k.net, k.ended.clone())).collect();
    log.named_theirs = kept.theirs.iter().map(|k| (k.name.clone(), format!("{:?}", k.verdict), k.net, k.ended.clone())).collect();
    log.refuted = kept.refuted.clone();
    log.unlisted = kept.unlisted.clone();
    // the candidate moves at the root: what was calculated, what each serves or answers, each estimate
    let mut mlog: Vec<MoveLog> = vec![];
    for (i, m) in moves.iter().enumerate() {
        let name = all_names[i].clone();
        let (line, _lost) = if std::time::Instant::now() >= wall_end && !equal.contains(&name) { (vec![], vec![]) } else { ls.line_from(s, i, ln.depth.max(1) as usize + 2) };
        let value = probes.iter().find(|(j, _)| *j == i).map(|(_, v)| *v);
        let mut serves: Vec<String> = vec![];
        let mut chance: Option<f64> = None;
        for gl in prop.goals.iter().filter(|x| x.side == me) {
            for f in &gl.facts { if f.fixes.iter().any(|fx| fx.moves.contains(&name)) { serves.push(format!("fix of {} ({})", text(&gl.target), f.name)); if let Some(t) = &inp.chances { let c = t.chance(gl).chance; chance = Some(chance.map_or(c, |o: f64| o.max(c))); } } }
        }
        serves.dedup();
        let mut ans: Vec<String> = vec![];
        for r in their.iter() {
            let by_resp = r.responses.iter().any(|x| x.state == RespState::Kept && x.names.first() == Some(&name));
            let by_fact = r.unit.is_some_and(|u| prop.defence.iter().any(|(t, a)| *t == u && a.iter().any(|x| x.mv == name)));
            if by_resp || by_fact { ans.push(format!("answers the threat on {}", r.unit.map_or_else(|| "the end of the game".to_string(), |u| text(&u)))); }
        }
        let rs = sc.as_ref().and_then(|v| v.iter().find(|x| x.mv == name));
        let _ = m;
        mlog.push(MoveLog { mv: name.clone(), value, equal: equal.contains(&name), line, ended: if value.is_some_and(|v| v.abs() >= WIN - 1000) { "the game itself ends in the line".into() } else { format!("the settled end after the depth read ({} plies)", ln.depth) },
            named: named_mine.contains(&name), own_read: kept.mine.iter().find(|k| k.name == name).map(|k| (format!("{:?}", k.verdict), k.net, k.ended.clone())), serves, answers: ans, chance, reasons_score: rs.map(|x| x.score), reasons_carried: rs.map(|x| x.carried.len()),
            their_after: after.rows.iter().find(|r| r.mv == name).map(|r| their_after_log(r, &text)) });
    }
    mlog.sort_by_key(|m| (!m.equal, m.value.map_or(i64::MAX, |v| -v)));
    log.moves = mlog;
    log.after_not_asked = after.not_asked.clone();
    let carried_next = frames.as_ref().map(|f| Carried::of(&prop.list, &f.read_sets));
    contributions.push(Contribution { piece: "the whole decision", step: 0, seconds: t_all.elapsed().as_secs_f64(), named: vec![], note: String::new() });
    WakeDecision { mv, value: ln.value, depth: ln.depth, nodes: ln.nodes, chain, contributions, unported: params.pieces.unported(), log, proposal: Some(prop), carried: carried_next, named_mine, named_theirs, equal }
}

/// PLAN 79.10 A: the fact about one move of ours in words (the units as the game names them): a thing it took out of the way of their chain, a thing it put in the way, or no change. It says what happened; it adds to no value.
fn their_after_log(r: &crate::after::TheirAfter, text: &dyn Fn(&UnitRef) -> String) -> crate::log::TheirAfterLog {
    let list = |v: &[UnitRef]| v.iter().map(|u| text(u)).collect::<Vec<_>>();
    let Some(f) = &r.fact else { return crate::log::TheirAfterLog { names: r.names.clone(), said: "after it no chain of theirs stands".into(), ..Default::default() } };
    let (took_out, put_in) = (list(&f.took_out), list(&f.put_in));
    let mut said = format!("their chain on {}: {} things in its way after it", text(&f.top), f.in_the_way_after);
    match f.in_the_way_before { Some(b) => said.push_str(&format!(" ({b} before)")), None => said.push_str(" (no such chain stood before)") }
    if !took_out.is_empty() { said.push_str(&format!("; it took out of the way: {}", took_out.join(", "))); }
    if !put_in.is_empty() { said.push_str(&format!("; it put in the way: {}", put_in.join(", "))); }
    if took_out.is_empty() && put_in.is_empty() { said.push_str("; nothing of ours was taken out of the way or put in it"); }
    crate::log::TheirAfterLog { top: Some(text(&f.top)), before: f.in_the_way_before, after: f.in_the_way_after, took_out, put_in, names: r.names.clone(), said }
}

impl WakeDecision {
    /// The decision as JSON, in words (the caller writes it where it wants): the move, the search's value, depth and nodes, the chain of claims with the reason each line ended, the time of each step, the pieces not ported, the moves named and held equal, and the decision log.
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "move": self.mv, "value": self.value, "depth": self.depth, "nodes": self.nodes, "unported": self.unported, "named_mine": self.named_mine, "named_theirs": self.named_theirs, "equal": self.equal,
            "chain": self.chain.iter().map(|c| serde_json::json!({"line": c.line, "ended": c.ended, "by": c.by})).collect::<Vec<_>>(),
            "steps": self.contributions.iter().map(|c| serde_json::json!({"piece": c.piece, "step": c.step, "seconds": c.seconds, "note": c.note, "named": c.named})).collect::<Vec<_>>(),
            "log": self.log.to_json(),
        })
    }
}
