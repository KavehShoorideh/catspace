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
use cs_game::relations::{PartRelations};
use cs_game::placed::{Kind, Place, Placed};
use cs_game::{Game, Role};
use crate::log::{DecisionLog, GoalLog, MoveLog, RowLog};
use crate::pick::MoveReasons;
use crate::rule;
use crate::threat_verdict::Record;
use cs_memory::store::GameMemory;
use crate::threats::{RespState, ThreatParams};
use cs_tactics::ledger::{Judge, Search, WIN};
use cs_tactics::lines::LineParams;
use cs_rep::history::Seen;
use cs_tactics::sheet::base_graph::{PositionPart, Root, Static, UnitRec};
use std::collections::HashSet;

/// Which pieces of the wake are on (section 8's steps): `carried` step 1, `frames` step 2, `threats` and `goals` and `two_threat_goals` step 3, `graph_reads` step 3 (the shut lines and the bearers a move brings), `reasons_pick` and `heads` step 6, `after_move` step 4 (PLAN 79.10 A), `direct_names` step 4 (79.10 C).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Pieces { pub carried: bool, pub frames: bool, pub threats: bool, pub goals: bool, pub two_threat_goals: bool, pub graph_reads: bool, pub reasons_pick: bool, pub heads: bool,
    /// PLAN 79.10 A: their goals are read after each move of ours (`after`), their fixes named at that child position, the fact about the move logged
    pub after_move: bool,
    /// PLAN 79.10 C: the names go to the search directly, a small count a position, in order of stake; no line of a named move is read before the search (the reading for the chain of claims comes after it, where the clock allows)
    pub direct_names: bool }

impl Pieces {
    /// The old player: the search with the count of worths at its end, nothing else.
    pub fn old_player() -> Pieces { Pieces::default() }
    /// Every piece on.
    pub fn all() -> Pieces { Pieces { carried: true, frames: true, threats: true, goals: true, two_threat_goals: true, graph_reads: true, reasons_pick: true, heads: true, after_move: true, direct_names: true } }
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
    /// PLAN 79.19 C: the long-term memory of this game (the worth record, the table of the chance of a goal by what stops it, the reasons and their weights, the table of the chance of a goal counted from play, the mined measure as a statement) and the memory of the game at hand (the open tactics of the last decision and the goal held, in its slot): `memory.rs`. These replace the fields `record`, `chances`, `reasons`, `goal_table`, `measure` and `previous`; nothing is read of them but through the two memories.
    pub memory: crate::memory::WakeMemory,
    pub goal: Option<Kind>,
    /// the budget of questions of the proposer (0: `goals::BUDGET`) and of line reads for naming moves (0: `rule::NAMED_BUDGET`)
    pub budget: usize, pub named_budget: usize,
    /// the frame read's learned seeds and parameters (None: the threat list's rows come from the graph)
    pub frame: Option<FrameRead>,
    /// the lines read from one position share a table (repair 2)
    pub share_table: bool,
    /// the share of the clock the search keeps whatever the steps before it cost, and from which the equal reading's clock follows (None: `LEDGER_SHARE`); the old player's share when the two are compared at one clock
    pub ledger_share: Option<f64>,
    /// PLAN 79.09 B: the share of the clock the steps before the search (1 to 4) may take together (None: what the search and its equal reading do not keep, less `TAIL_SHARE`: `pre_share_of`). The steps ask in order of stake and stop at it; what they did not ask is listed.
    pub pre_share: Option<f64>,
    /// PLAN 79.12 B: the last layer among the moves held equal (None: the rule of rule.rs decides, as before). With a spec the rows of the candidates are written to the log in every mode; Weights and Explore choose.
    pub chooser: Option<crate::chooser::ChooserSpec>,
    /// PLAN 79.14 B: the carry asks again only what the graph says the two plies since touched (the seeds, the frames, the rows of the threat list); every answer carried is marked carried in the log. false (the default): the carry of 79.08, the frame read asked as it was.
    pub carry_graph: bool,
    /// PLAN 79.14 C: with a goal table, the named moves go to the search in the order of stake times the table's chance of the goal they fix, ours and theirs, a goal whose every fact is met named first (rule::named_fixes_by); no table, or false: the order of stake alone, as on main.
    pub chance_order: bool,
    /// PLAN 79.14 D: the gain of mode Gain counts a goal the move meets as reached (its stake in full) and reads the chains after a move on every target (gain.rs, after.rs); false: the gain of 79.13.
    pub gain_mend: bool,
    /// PLAN 79.15 A: after the decision, a lowered first move is asked again unreduced and at the next depths (the log's `first_diag`); a measure, false in play
    pub diag_first: bool,
    /// PLAN 79.16 A: the search does not stop on a failed first move (`Search::fail_extend`): the clock it may read on is `FAIL_OF_TIE` of the equal reading's share, taken from it; false (the default): the search as it was.
    pub read_on: bool,
    /// PLAN 79.22 PARTS A AND B: where the game's facts hold its units and its tempo (data the game gives). With it the memory of the game at hand keeps its position between decisions: the root graph is the kept one, changed by the moves played, and the decision's own move is played in the memory at its end (`keep::advance`). None: nothing is kept, the root graph is built once a decision.
    pub keep: Option<cs_tactics::sheet::base_graph::FactDecl>,
}

/// PLAN 79.17 D: the term of the measure (cs_tactics::measure) the wake is asked to hold and the depth `d`, in our moves, within which a line that lowers it against every reply is asked for.
#[derive(Debug, Clone)]
pub struct MeasureGoal { pub term: cs_tactics::measure::Term, pub d: u32 }

impl Default for WakeInputs { fn default() -> WakeInputs { WakeInputs { memory: crate::memory::WakeMemory::default(), goal: None, budget: 0, named_budget: 0, frame: None, share_table: true, ledger_share: None, pre_share: None, chooser: None, carry_graph: false, chance_order: false, gain_mend: false, diag_first: false, read_on: false, keep: None } } }

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
    pub log: DecisionLog, pub proposal: Option<Proposal>,
    /// PLAN 79.19 C: the memory of the game at hand after this decision: the open tactics to carry and the goal held are its slot `carried`, written by the wake (`WakeDecision::carried()` reads it); the caller hands it to the next decision
    pub game: GameMemory, pub named_mine: Vec<String>, pub named_theirs: Vec<String>, pub equal: Vec<String>,
    /// PLAN 79.15 B: the search's own first move (the first of its line), which is no longer always the head of `equal`
    pub first: Option<String>,
}

impl WakeDecision {
    /// The open tactics and the goal held that this decision left in the memory of the game (None: none).
    pub fn carried(&self) -> Option<&Carried> { self.game.slot::<Carried>(crate::memory::CARRIED) }
}

/// The record when none is given: the empty one (read, never built per decision).
fn empty_record() -> &'static Record { static EMPTY: std::sync::OnceLock<Record> = std::sync::OnceLock::new(); EMPTY.get_or_init(Record::default) }

/// A unit as the graph names it: owner, kind, place (numbers the game gave).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct UnitRef { pub owner: Role, pub kind: Kind, pub at: Place }

fn uref(u: &UnitRec) -> UnitRef { UnitRef { owner: u.owner, kind: u.kind, at: u.at } }

/// A line of a unit onto a place that exactly one unit shuts: the unit, the blocker, the units on the other side of the blocker that bear on its place, the units on its side that bear on it (the blocker itself aside).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LineOnto { pub unit: UnitRef, pub blocker: UnitRef, pub bearers: Vec<UnitRef>, pub holders: Vec<UnitRef> }

/// The lines of `side`'s units onto `place` that exactly one unit shuts, read from a position's graph. The graph computes `shuts` for the side to move, so for the other side the caller hands the position after the game's pass (`goal_unit_lines`).
pub fn lines_onto(part: &PositionPart, place: Place, side: Role) -> Vec<LineOnto> {
    let bears: HashSet<(usize, Place)> = part.bears().iter().map(|r| (r.unit, r.place)).collect();
    let mut out = vec![];
    for r in part.shuts() {
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

/// PLAN 79.22 PART B: the game's memory after the decision, with the move the decision chose played in it (the position at hand, the tags and the history one tempo on) when the game keeps its position (`WakeInputs::keep`).
fn kept_after<G: Game + Placed>(st: &Static, g: &G, s: &G::State, inp: &WakeInputs, mv: Option<&String>, mut game: cs_memory::store::GameMemory) -> cs_memory::store::GameMemory {
    if let (Some(decl), Some(name)) = (&inp.keep, mv) {
        if let Some(m) = g.find_move(s, name) { let after = g.apply(s, &m); crate::keep::advance(st, g, decl, &mut game, &after); }
    }
    game
}

/// THE WHOLE DECISION (see the module doc). Ported so far: the graph reads (`new_bearer_moves` over `params.places`) that name moves read first and unreduced; the search is `cs_tactics::ledger::Search`. A piece asked for and not ported is listed in `unported`.
fn decide_search<G: Game + Placed, J: Judge>(st: &Static, g: &G, s: &G::State, seen: &std::sync::Arc<Seen>, judge: &J, params: &WakeParams) -> WakeDecision {
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
    ls.history = seen.clone();
    if !named.is_empty() { ls.named.insert(g.key(s), named); }
    let ln = ls.run(s, params.depth);
    contributions.push(Contribution { piece: "search with the count of worths at its end", step: 5, seconds: t0.elapsed().as_secs_f64(), named: vec![], note: String::new() });
    let ended = if ln.value.abs() >= 1_000_000 { "the game itself ends in the line (concrete)".to_string() } else { format!("the settled end: no capture is left to either side after the depth read in full ({} plies)", ln.depth) };
    WakeDecision { mv: ln.first.clone(), value: ln.value, depth: ln.depth, nodes: ln.nodes, chain: vec![Claim { line: ln.line.clone(), ended, by: "search" }], contributions, unported: params.pieces.unported(), log: DecisionLog::default(), proposal: None, game: kept_after(st, g, s, &params.inputs, ln.first.as_ref(), params.inputs.memory.game.clone()), named_mine: vec![], named_theirs: vec![], equal: vec![], first: ln.first.clone() }
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
/// PLAN 79.10 C: with direct names the tail of the clock (what is left after the search and the equal reading) is shared between the reasons' score of the moves held equal and the reading of the named moves for the claims: the score may take this share of what is left, the claims the rest.
pub const CLAIMS_OF_TAIL: f64 = 0.5;
/// PLAN 79.15 A: the clock of each extra reading of the diagnosis of a lowered first move (a measure, asked only by the piece `diag_first`, after the decision).
pub const DIAG_SECONDS: f64 = 20.0;
/// PLAN 79.16 A (SYSTEM_SPEC 7, the budget): THE SEARCH'S READING ON a failed first move may take this share of the equal reading's clock (`tie_seconds`), and the equal reading has that much less to do; the decision's total clock is what it was given. A named share (half: the equal reading and the reading on a first move are the
/// two readings that ask whether the search's first move holds, and each gets half of what the one had), printed in the log with what it spent.
pub const FAIL_OF_TIE: f64 = 0.5;
/// PLAN 79.16 B: the clock of the search is the whole decision's clock less what the equal reading and the tail keep (`search_seconds`).

/// PLAN 79.16 B (SYSTEM_SPEC 7, the budget; 8 the wall): TIME NOT USED GOES TO THE SEARCH. The search is given what is left of the decision's clock `total` (`left` seconds from now) after the equal reading's clock `tie` and the tail (`TAIL_SHARE` of the total) are kept, and never less than its own share of the total: so where the steps before it end
/// under their share (or the share is scaled down) the search's deadline is moved by what is left, and where they overrun it the search still has its share. With the steps spending exactly their share this is `total * share`, as before.
pub fn search_seconds(total: f64, share: f64, left: f64, tie: f64) -> f64 { (left - tie - TAIL_SHARE * total).max(total * share) }

/// PLAN 79.16 B: the clock of the equal reading: its share less what the reading on of a failed first move spent (`spent`), and what is left of the decision's clock after the tail is kept if that is more (the clock the steps and the search did not use is the equal reading's, which the clock cuts three times in four); never less than 0.01 s.
pub fn equal_seconds(total: f64, tie: f64, spent: f64, left: f64) -> f64 { (tie - spent).max(left - TAIL_SHARE * total).max(0.01) }

/// The share of the clock the steps before the search may take.
pub fn pre_share_of(share: f64, given: Option<f64>) -> f64 { given.unwrap_or(((1.0 - share) * (1.0 - crate::TIE_SHARE) - TAIL_SHARE).max(0.0)) }

/// PLAN 79.17 D: of the share of the clock the steps before the search have, the held goal's question may take this much (half: the other steps read in what is left, and the search keeps its share whatever they spend).
pub const MEASURE_OF_PRE: f64 = 0.5;
/// PLAN 79.17 D: the fuse on the nodes of the held goal's question (a named cap, the clock is the real bound; a cut is a gap, never a verdict).
pub const MEASURE_NODES: u64 = 20_000_000;

/// PLAN 79.17 PART D, MENDED BY PLAN 79.18 B AND C (SYSTEM_SPEC 7, the measure of progress; 8 the rule for choosing a move; piece: the wake). THE HELD GOAL, one decision of it. INTERFACE: takes the game, the position, the history keys, the side to move (the side the measure is for), the goal asked for (`MeasureGoal`), the goal held from the last decision
/// (`Carried::held`, None at the first) and the instant its question may run to; returns the goal held now, the log of what it did and the first move of the line the question found (None: none, or the clock cut the question). Called by `decide`, first, inside the share of the steps before the search.
/// THE RULES: the goal is taken up at the first decision at the value read there; a value lower than the lowest held is a SUBGOAL REACHED (the event of ledger row 24, logged with the position's key) and the lowest held moves to it. THE QUESTION (79.18 B, `cs_tactics::measure::question`) is counted against the LOWEST held: a line within `d` of our moves after which the term is below the lowest held
/// against every reply (a fall); where none exists, read to its end, and the value now is above the lowest, a line that brings it below the value now (a RECOVERY: logged as one, played, never counted as a subgoal reached); a position met before in the game is never the end of a line that counts. THE GOAL'S STATES (79.18 C): it is DROPPED for good only when the class changed (the units of each side by kind are not
/// the units it was taken up with); a question read to its end that found no line leaves the goal held and it is asked again at the next decision (the log says how many decisions in a row without a line); a goal the search's end of the game puts to SLEEP (`decide`) stays in what is carried and is woken at the next decision in which the search no longer holds the end.
pub fn measure_step<G: Game + Placed>(st: &Static, g: &G, s: &G::State, history: &Seen, me: Role, mg: &MeasureGoal, prev: Option<&crate::carry::HeldGoal>, end: std::time::Instant) -> (crate::carry::HeldGoal, crate::log::MeasureLog, Option<String>) {
    use cs_tactics::measure::{lower, question, Geometry};
    use cs_value::{Assess, Question};
    let geo = Geometry::new(st);
    // PLAN 79.24: the numbers of the held term are read through layer 3's one question (the same code as the value of a position is asked by), not by a copy
    let progress = cs_value::progress::Progress::given(mg.term.clone(), mg.d, &geo, cs_memory::statement::Standing::Conjecture("held as given by the caller of the wake".into()));
    let now = progress.assess(&Question { game: g, state: s, side: me }).ok().and_then(|a| a.raw).unwrap_or_default();
    let mut class: Vec<(bool, u8)> = g.cells(s).iter().map(|(o, k, _)| (*o != me, *k)).collect();
    class.sort();
    let text = mg.term.text();
    let mut log = crate::log::MeasureLog { term: text.clone(), d: mg.d, value_now: now.clone(), key: g.key(s), ..Default::default() };
    let fresh = prev.is_none_or(|p| p.term != text);
    let mut held = if fresh { crate::carry::HeldGoal { term: text.clone(), value: now.clone(), since: 0, reached: 0, class: class.clone(), dropped: None, asleep: None, slept: 0, woken: 0, without_line: 0, recovered: 0 } } else { prev.unwrap().clone() };
    let totals = |log: &mut crate::log::MeasureLog, h: &crate::carry::HeldGoal| { log.since = h.since; log.reached_total = h.reached; log.without_line = h.without_line; log.recovered_total = h.recovered; log.slept_total = h.slept; log.woken_total = h.woken; log.asleep = h.asleep.clone(); };
    if fresh { log.said = format!("the goal is taken up: the measure {text} reads {now:?} here"); }
    else {
        log.held_before = Some(held.value.clone());
        if let Some(c) = held.dropped.clone() { log.dropped = Some(c.clone()); log.said = format!("the goal was dropped earlier: {c}"); totals(&mut log, &held); return (held, log, None); }
        if held.class != class {
            let c = "the class changed: the units of the two sides are no longer those the goal was taken up with".to_string();
            held.dropped = Some(c.clone()); log.dropped = Some(c.clone()); log.said = format!("the goal is dropped: {c}"); totals(&mut log, &held);
            return (held, log, None);
        }
        if lower(&now, &held.value) { log.reached = true; log.said = format!("a subgoal is reached: the measure fell from {:?} to {now:?}", held.value); held.value = now.clone(); held.since = 0; held.reached += 1; }
        else { held.since += 2; log.said = format!("the measure is {now:?}, the lowest held {:?}, {} plies since it was lowered", held.value, held.since); }
    }
    log.held_after = Some(held.value.clone());
    let t0 = std::time::Instant::now();
    let mut b = cs_tactics::Budget::new(end.saturating_duration_since(t0).as_secs_f64(), MEASURE_NODES);
    let (found, cut) = question(g, &geo, s, me, &mg.term, &held.value, &now, mg.d, history, &mut b);
    log.seconds = t0.elapsed().as_secs_f64();
    log.nodes = b.nodes;
    log.cut = cut;
    let mut first = None;
    match found {
        Some(f) => {
            let l = f.lowering();
            let name = g.move_name(s, &g.moves(s)[l.first]);
            let ends = if l.game_end { " (the end of the game is found in it)" } else { "" };
            if f.is_fall() { log.said.push_str(&format!("; a line of {} of our moves lowers it below the lowest held against every reply{ends}, its first move {name}", l.moves)); }
            else { log.recovery = true; held.recovered += 1; log.said.push_str(&format!("; no line within {} of our moves goes below the lowest held; a RECOVERY, not a subgoal reached: a line of {} of our moves brings it below the value now {now:?} against every reply{ends}, its first move {name}", mg.d, l.moves)); }
            log.first = Some(name.clone()); log.moves = Some(l.moves); log.game_end = l.game_end;
            held.without_line = 0;
            first = Some(name);
        }
        None if !cut => {
            held.without_line += 1;
            log.said.push_str(&format!("; no line of at most {} of our moves goes below the lowest held or below the value now (the question was read to its end): the goal stays and is asked again at the next decision, {} decisions in a row without a line", mg.d, held.without_line));
        }
        None => log.said.push_str("; the clock cut the question: nothing is named (a gap, the goal stays)"),
    }
    totals(&mut log, &held);
    (held, log, first)
}

/// The instant a step may run to: `frac` of what is left of the time up to `end`.
fn until(end: std::time::Instant, frac: f64) -> std::time::Instant { let now = std::time::Instant::now(); if end <= now { now } else { now + (end - now).mul_f64(frac) } }

/// THE WHOLE DECISION, by the steps of SYSTEM_SPEC 8 and the rule for choosing a move (rule.rs): (1) the carry, (2) the frames and the rows of the threat list, (3) the threat list and the goals of both sides in order of stake within the budget, (4) THEIR goals that work: our answers; OUR goal with the fewest things in its way:
/// its fixes; only the moves whose own line was read and not refuted are named, (5) the search reads with the named moves first and in full, ours at the root and theirs at the ply after ours; the judge's value is untouched, (6) among the moves the search holds equal: an answer to their largest standing threat, then a fix of
/// our chosen goal, then the estimates, (7) the chain of claims and the decision log. With no piece of the whole on this is `decide_search`, the old player. The clock: steps 1 to 4 take what they take (their cost is reported by step); the search is given what is left of `params.seconds` and never less than the old player's share of it.
pub fn decide<G: Game + Placed, J: Judge>(st: &Static, g: &G, s: &G::State, history: &[u64], judge: &J, params: &WakeParams) -> WakeDecision { decide_seen(st, g, s, &Seen::of(history), judge, params) }

/// PLAN 79.22 PART B: `decide` with the record as layer 1 keeps it (`History::seen`): every reader of a repeat in the decision (the search, the measure's lines) borrows this one set by reference count; nothing is copied from a list. `decide` is the way in of a caller that holds the keys as a list (a test, a measure).
pub fn decide_seen<G: Game + Placed, J: Judge>(st: &Static, g: &G, s: &G::State, history: &std::sync::Arc<Seen>, judge: &J, params: &WakeParams) -> WakeDecision {
    let p = params.pieces;
    if !(p.carried || p.frames || p.threats || p.goals || p.two_threat_goals || p.reasons_pick) { return decide_search(st, g, s, history, judge, params); }
    let t_all = std::time::Instant::now();
    let me = g.mover(s).unwrap_or(0);
    let other = g.other(me);
    let inp = &params.inputs;
    // PLAN 79.22 PART A: THE ROOT'S GRAPH, MADE ONCE FOR THIS DECISION and lent to every part that reads it (the frame read, the threat list, the goals, the reasons' pick, the carry's digest); a part that reads another position builds that one only
    let root = crate::keep::root_of(st, g, s, &inp.memory.game);
    // PLAN 79.19 C: what the wake reads comes from the two memories: the tables from the reader of the long-term memory, the open tactics from the memory of the game at hand
    let mem = &inp.memory;
    let rec: &Record = match mem.record() { Some(r) => r, None => empty_record() };
    let (goal_table, chances, reasons, previous) = (mem.goal_table(), mem.chances(), mem.reasons(), mem.previous());
    let measure = mem.measure();
    let (kinds, roles, places) = (g.kind_names(), g.roles(), g.place_names());
    let share = inp.ledger_share.unwrap_or(crate::LEDGER_SHARE);
    let pre_end = t_all + std::time::Duration::from_secs_f64(params.seconds * pre_share_of(share, inp.pre_share));
    let wall_end = t_all + std::time::Duration::from_secs_f64(params.seconds);
    let text = |u: &UnitRef| format!("{} {} on {}", roles[u.owner], kinds[u.kind as usize], places[u.at as usize]);
    let mut contributions: Vec<Contribution> = vec![];
    let mut log = DecisionLog::default();
    // ---- PLAN 79.17 D, before every other step: the held goal of the measure, inside half of the share of the steps before the search (SYSTEM_SPEC 7)
    let mut held_next: Option<crate::carry::HeldGoal> = None;
    let mut measure_log: Option<crate::log::MeasureLog> = None;
    let mut toward_first: Option<String> = None;
    if let Some(mg) = &measure {
        let t0 = std::time::Instant::now();
        let (h, ml, first) = measure_step(st, g, s, history, me, mg, previous.and_then(|c| c.held.as_ref()), until(pre_end, MEASURE_OF_PRE));
        contributions.push(Contribution { piece: "the held goal of the measure: the question of a lowering line", step: 3, seconds: t0.elapsed().as_secs_f64(), named: first.iter().cloned().collect(), note: ml.said.clone() });
        held_next = Some(h);
        measure_log = Some(ml);
        toward_first = first;
    }
    // ---- PLAN 79.10 A, first, because it needs nothing of the steps after it and costs about a thousandth of a second a position: their goals on the position after each move of ours, in the search's own order, within its share of the steps before the search; their fixes named at that child
    let count = if inp.named_budget > 0 { inp.named_budget } else { rule::NAMED_BUDGET };
    // PLAN 79.13 C: the chance of a goal: the table counted from play where given, else the old table; a goal's rank is the rank of its target's stake among the record's worths
    let worths: Vec<i64> = rec.worth.values().copied().collect();
    let table_chance = |gl: &Goal, sk: &rule::Stakes| -> Option<Estimate> { goal_table.map(|t| t.chance(&crate::goal_table::GoalFacts::of(gl, gl.side == me, crate::goal_table::rank_of(sk.of(gl.target), &worths)))) };
    let goal_chance = |gl: &Goal, sk: &rule::Stakes| -> Option<Estimate> { table_chance(gl, sk).or_else(|| chances.map(|t| t.chance(gl))) };
    let gain_on = p.after_move && goal_table.is_some() && inp.chooser.is_some();
    let mut after = crate::after::AfterRead::default();
    if p.after_move {
        let order = Search::new(g, judge, me, 0.1).deepening_order(s);
        let value = |tg: &crate::after::TheirGoals, side: cs_game::Role| { let ch = |gl: &Goal, sk: &rule::Stakes| table_chance(gl, sk).map_or(0.0, |e| e.chance); if inp.gain_mend { crate::gain::chain_value_mended(&tg.goals, side, &tg.stakes, &ch) } else { crate::gain::chain_value(&tg.goals, side, &tg.stakes, &ch) } };
        after = crate::after::goals_after_in(st, Some(&root), g, s, rec, inp.goal, &order, count, Some(until(pre_end, AFTER_OF_PRE)), if gain_on { Some(&value) } else { None }, inp.gain_mend && gain_on);
        contributions.push(Contribution { piece: "their goals after each move of ours", step: 3, seconds: after.seconds, named: vec![], note: format!("{} positions read, {} not reached by the clock{}", after.rows.len(), after.not_asked.len(), if gain_on { format!("; ours read too, {:.4} s of it", after.ours_seconds) } else { String::new() }) });
    }
    // ---- PLAN 79.14 B: what the two plies since the last decision touched, from the graph (the digest of this position is kept for the next decision)
    let digest_now: Option<(usize, cs_tactics::sheet::delta::Digest)> = if inp.carry_graph { let part = st.position_in(&root, g, s); let b = cs_tactics::sheet::patterns::Board::new(st, &part, inp.goal); Some((me, cs_tactics::sheet::delta::Digest::of(&b, &part))) } else { None };
    let touch_now: Option<cs_tactics::sheet::delta::Touch> = match (&digest_now, previous.and_then(|c| c.digest.as_ref())) { (Some((m, now)), Some((pm, prev))) if m == pm => Some(cs_tactics::sheet::delta::Touch::between(prev, now)), _ => None };
    // ---- steps 1 and 2: the frames and the carry
    let mut frames: Option<FrameRows> = None;
    if p.frames {
        match &inp.frame {
            Some(fr) => { let mut fr = fr.clone(); fr.params.deadline = Some(until(pre_end, FRAMES_OF_PRE)); let f = crate::frame_rows::frame_read_in(st, Some(&root), g, s, inp.goal, &fr, None, previous.and_then(|c| c.memo.as_ref()), inp.carry_graph); contributions.push(Contribution { piece: "frames and the question read", step: 2, seconds: f.seconds, named: vec![], note: format!("{} frames, {} rows, {} questions, {} frames not reached by the clock{}", f.frames, f.rows.len(), f.questions, f.unread_frames, if inp.carry_graph { format!("; carried: {} frames, {} seeds{}", f.parts.carried, f.parts.seeds_carried, if f.touch.is_none() { " (nothing of the last decision to carry from)" } else { "" }) } else { String::new() }) });
                // PLAN 79.14 A: the cost of the frame read by part (these are parts of the step above, not steps of their own: step 20 and over)
                let pt = &f.parts;
                for (name, secs, note) in [("part of the frame read: the seeds on the board", pt.seeds_s, format!("{} seeds found", pt.seeds)), ("part of the frame read: the frames built", pt.build_s, format!("{} frames built", pt.built)), ("part of the frame read: the claims' questions", pt.claims_s, format!("{} frames asked, {} carried", pt.asked, pt.carried)), ("part of the frame read: the answers' questions", pt.answers_s, String::new())] {
                    contributions.push(Contribution { piece: name, step: 22, seconds: secs, named: vec![], note });
                }
                frames = Some(f); }
            None => log.notes.push("frames asked for and no learned seeds given: the rows of the threat list are the graph's".into()),
        }
    }
    let mut carried_out: Option<Carry> = None;
    if p.carried {
        let t0 = std::time::Instant::now();
        match (&previous, &frames) {
            (Some(prev), Some(f)) => { let c = if inp.carry_graph { crate::carry::carry_graph(prev, f, touch_now.as_ref()) } else { carry(prev, f) }; contributions.push(Contribution { piece: "carried open tactics", step: 1, seconds: t0.elapsed().as_secs_f64(), named: vec![], note: format!("{} rows held, {} read again", c.held.len(), c.read.len()) }); carried_out = Some(c); }
            (None, _) => log.notes.push("carry asked for and there is no last decision: nothing is carried".into()),
            (_, None) => log.notes.push("carry asked for and the frame read gave no read sets: nothing is carried".into()),
        }
    }
    // ---- step 3: the threat list and the goals
    let t0 = std::time::Instant::now();
    let rows = frames.as_ref().map(|f| carried_out.as_ref().map_or_else(|| f.rows.clone(), |c| c.read.clone()));
    let tp = ThreatParams { goal: inp.goal, rows, share_table: inp.share_table, deadline: Some(until(pre_end, LIST_OF_PRE)), root: Some(&root), ..Default::default() };
    // PLAN 79.14 B: the units of the last decision's every-unit reading whose neighbourhood nothing touched are not asked again
    let held_units = match (inp.carry_graph && p.carried, &previous, &touch_now) { (true, Some(prev), Some(t)) => Some((prev.rows.iter().filter(|r| r.row.every).cloned().collect::<Vec<_>>(), t.clone())), _ => None };
    let gp = GoalParams { budget: if inp.budget > 0 { inp.budget } else { crate::goals::BUDGET }, held: carried_out.as_ref().map(|c| c.held.clone()).unwrap_or_default(), held_units, keep_met: inp.chance_order && goal_table.is_some(), ..Default::default() };
    let prop = propose(st, g, s, rec, &tp, &gp);
    contributions.push(Contribution { piece: "threat list of both sides", step: 3, seconds: prop.list.seconds(), named: vec![], note: format!("{} rows against us, {} against them, {} dropped, {} questions; not asked: {}", prop.list.mover.len(), prop.list.other.len(), prop.list.dropped.len(), prop.questions, if prop.list.not_asked.is_empty() { "nothing".to_string() } else { prop.list.not_asked.join("; ") }) });
    for c in &prop.list.cost { contributions.push(Contribution { piece: "part of the threat list: a round of line reads", step: 23, seconds: c.seconds, named: vec![], note: format!("{}: {} queries, {} nodes", c.name, c.queries, c.nodes) }); }
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
    let by_chance = |gl: &Goal| table_chance(gl, &stakes).map_or(0.0, |e| e.chance);
    let ordered: Option<&dyn Fn(&Goal) -> f64> = if inp.chance_order && goal_table.is_some() { Some(&by_chance) } else { None };
    let chains_mine = rule::named_fixes_by(&prop.goals, me, &stakes, ordered);
    let chains_theirs = rule::named_fixes_by(&prop.goals, other, &stakes, ordered);
    // (with direct names the count is applied once, below, so that what it cuts is listed)
    let chain_cap = if p.direct_names { usize::MAX } else { count };
    for m in rule::names_of(&chains_mine, chain_cap) { put(&mut mine, &m); }
    for (k, r) in their.iter().enumerate() {
        let Some(u) = r.unit else { continue };
        for (t, ans) in &prop.defence { if *t == u { for a in ans { put(&mut mine, &a.mv); if k == 0 { put(&mut answers_largest, &a.mv); } } } }
    }
    // PLAN 79.15 C: where a line stands on the mover's goal unit that is their largest standing threat (the end of the game forced against us aside): the moves that answer it are the answers, and the reason says so before any fix of a goal of ours is named
    let line = if let Some(first) = their.first().filter(|r| r.goal) { let _ = first; None } else { crate::goals::standing_line(st, g, s, inp.goal) };
    let line_stands = line.is_some();
    let line_answers: Vec<String> = line.as_ref().map(|l| l.answers.clone()).unwrap_or_default();
    let line_text: Option<String> = line.as_ref().map(|l| format!("{} (a line of theirs stands on it)", text(&l.unit)));
    // among the answers to the largest threat only those that also answer the rows' largest threat where there are any (as before), else the answers to the line
    let answers_largest: Vec<String> = if line_stands { let both: Vec<String> = answers_largest.iter().filter(|m| line_answers.contains(m)).cloned().collect(); if both.is_empty() { line_answers.clone() } else { both } } else { answers_largest };
    let mut theirs: Vec<String> = vec![];
    // PLAN 79.10 A: with `after_move` their fixes are named at each child position (below), not at the ply after ours on a pass
    if !p.after_move { for m in rule::names_of(&chains_theirs, chain_cap) { put(&mut theirs, &m); } }
    for r in prop.list.other.iter().filter(|r| r.verdict == crate::threat_verdict::VerdictKind::Confirmed && !r.two_moves) {
        for x in r.responses.iter().filter(|x| x.state == RespState::Kept) { if let Some(n) = x.names.first() { put(&mut theirs, n); } }
    }
    let budget = count;
    let t0 = std::time::Instant::now();
    let (mut kept, named_mine, named_theirs): (rule::KeptNames, Vec<String>, Vec<String>);
    if p.direct_names {
        // PLAN 79.10 C: the names go to the search as they are, the answers to their largest threat first, then in order of stake; a count, not a reading. What the count cut is listed (a gap).
        let first: Vec<String> = answers_largest.iter().cloned().chain(mine.iter().filter(|m| !answers_largest.contains(m)).cloned()).collect();
        named_mine = first.iter().take(count).cloned().collect();
        named_theirs = theirs.iter().take(count).cloned().collect();
        kept = rule::KeptNames::default();
        kept.unlisted = first.iter().skip(count).map(|m| (m.clone(), false)).chain(theirs.iter().skip(count).map(|m| (m.clone(), true))).collect();
        contributions.push(Contribution { piece: "named moves: given to the search directly, in order of stake", step: 4, seconds: t0.elapsed().as_secs_f64(), named: named_mine.iter().chain(named_theirs.iter()).cloned().collect(), note: format!("{} named for the root, {} for the ply after, {} over the count", named_mine.len(), named_theirs.len(), kept.unlisted.len()) });
    } else {
        kept = rule::keep_read_by(g, s, rec, &mine, &theirs, budget, LineParams { depth: tp.sweep_depth, fuse: tp.sweep_fuse }, inp.share_table, Some(pre_end));
        named_mine = kept.mine.iter().map(|k| k.name.clone()).collect();
        named_theirs = kept.theirs.iter().map(|k| k.name.clone()).collect();
    }
    // PLAN 79.17 D: the first move of the line that lowers the held measure is named to the search first (read in full beside its rivals); it orders, it never adds to a value
    let (named_mine, named_theirs) = (named_mine, named_theirs);
    let mut named_mine = named_mine;
    if let Some(m) = &toward_first { named_mine.retain(|x| x != m); named_mine.insert(0, m.clone()); }
    // PLAN 79.09 C: a goal is the subgoal of the moment only if a named move serves it: the first chain (in order of stake) whose nearest goal has a move that was named
    let chosen_chain = chains_mine.iter().position(|c| c.all_moves().iter().any(|m| named_mine.contains(m)));
    let chosen_ix = chosen_chain.map(|k| chains_mine[k].goal);
    if chosen_ix.is_none() && !chains_mine.is_empty() { log.notes.push("no goal of ours is the subgoal of the moment: no move that fixes one was named (refuted or not read)".into()); }
    if !p.direct_names { contributions.push(Contribution { piece: "named moves: their lines read, the refuted left out", step: 4, seconds: t0.elapsed().as_secs_f64(), named: named_mine.iter().chain(named_theirs.iter()).cloned().collect(), note: format!("{} lines read, {} refuted, {} not read (over the budget)", kept.read, kept.refuted.len(), kept.unlisted.len()) }); }
    // ---- step 5: the search, with the named moves first and in full
    let t0 = std::time::Instant::now();
    // the search keeps its share of the clock and gets what the steps before it did not spend of theirs
    let tie = tie_seconds(params.seconds, share);
    let search_secs = search_seconds(params.seconds, share, wall_end.saturating_duration_since(std::time::Instant::now()).as_secs_f64(), tie);
    let mut ls = Search::new(g, judge, me, search_secs);
    if inp.read_on { ls.fail_extend = FAIL_OF_TIE * tie; }
    ls.lean = true;          // PLAN 79.16 C: the same answers with one ask of the game's mover a node
    ls.history = history.clone();
    ls.verify_ties = true;
    ls.fair_names = p.direct_names;          // PLAN 79.11 A: a named move and an un-named rival are compared by the same reading
    if !named_mine.is_empty() { ls.named.insert(g.key(s), named_mine.clone()); }
    if !named_theirs.is_empty() { ls.named_ply.insert(1, named_theirs.clone()); }
    for r in &after.rows { if !r.names.is_empty() { ls.named.insert(r.key, r.names.clone()); } }
    let ln = ls.run(s, params.depth);
    let moves = g.moves(s);
    let own: Option<usize> = ln.first_index;
    let decided = ln.value.abs() >= WIN - 1000;
    let mut probes: Vec<(usize, i64)> = vec![];
    let mut equal_ix: Vec<usize> = vec![];
    let mut verified = false;
    if !decided && own.is_some() {
        ls.resume(equal_seconds(params.seconds, tie, ls.failed_first.as_ref().map_or(0.0, |f| f.seconds), wall_end.saturating_duration_since(std::time::Instant::now()).as_secs_f64()));
        probes = ls.probes(s, ln.depth, ln.value, None);
        let first: Vec<usize> = probes.iter().filter(|(_, v)| *v >= ln.value).map(|(i, _)| *i).collect();
        equal_ix = if ls.cut() || first.is_empty() { if ls.cut() { vec![] } else { first } } else { ls.verify(s, ln.depth, ln.value, &first) };
        verified = !ls.cut();
    }
    // PLAN 79.15 B (the wall kept): a first move the equal reading put lower is not a candidate while any other move holds (wall.rs); main put it at the head of the set whether the reading kept it or not
    let cand = crate::wall::candidates(own, ln.value, &probes, &equal_ix, verified);
    let first_lowered = cand.lowered != crate::wall::Lowered::No;
    // PLAN 79.12 A: did the equal reading keep the search's first move (None: the reading did not run; Some(false) now also where its probe alone put it lower and the verification was cut)
    let first_held: Option<bool> = if first_lowered { Some(false) } else if verified { own.map(|o| equal_ix.contains(&o)) } else { None };
    let first_lowered_by: Option<i64> = if first_held == Some(false) { own.and_then(|o| probes.iter().find(|(i, _)| *i == o)).map(|(_, v)| ln.value - v).filter(|d| *d > 0) } else { None };
    let equal_ix: Vec<usize> = cand.moves.clone();
    let equal: Vec<String> = equal_ix.iter().map(|i| g.move_name(s, &moves[*i])).collect();
    contributions.push(Contribution { piece: "search with the count of worths at its end, and the equal reading", step: 5, seconds: t0.elapsed().as_secs_f64(), named: vec![], note: format!("depth {}, {} nodes, {} held equal", ln.depth, ls.nodes, equal.len()) });
    // ---- step 6: among the moves held equal
    let t0 = std::time::Instant::now();
    let all_names: Vec<String> = moves.iter().map(|m| g.move_name(s, m)).collect();
    let sc: Option<Vec<MoveReasons>> = match (&reasons, p.reasons_pick) {
        // the moves held equal are scored first (they decide); the others only while the clock of the whole decision lasts (they are logged); one not reached has no score
        (Some(r), true) => { let order: Vec<String> = equal.iter().cloned().chain(all_names.iter().filter(|n| !equal.contains(n)).cloned()).collect(); Some(crate::pick::scores_must_in(st, Some(&root), g, s, inp.goal, r, &order, Some(if p.direct_names { until(wall_end, CLAIMS_OF_TAIL) } else { wall_end }), equal.len())) }
        (None, true) => { log.notes.push("the reasons' pick asked for and no reasons given: no estimate of a move, the search's own first move decides among the rest".into()); None }
        _ => None,
    };
    let tie_digits = reasons.map_or(9, |r| r.tie_digits);
    let (fixes, fix_chain) = rule::fixes_among(&chains_mine, &equal);
    // the estimates decide only if every move held equal was scored
    let scored_equal = sc.as_ref().is_some_and(|v| equal.iter().all(|e| v.iter().any(|x| &x.mv == e)));
    if sc.as_ref().is_some_and(|v| v.len() < all_names.len()) { log.notes.push(format!("the clock ended before the reasons' score of {} moves: they have none", all_names.len() - sc.as_ref().map_or(0, |v| v.len()))); }
    if sc.is_some() && !scored_equal { log.notes.push("the clock ended before every move held equal was scored: the estimates do not decide".into()); }
    let chosen = rule::choose(&equal, &answers_largest, &fixes, &|set: &[String]| if scored_equal { sc.as_ref().and_then(|v| crate::pick::pick(v, set, tie_digits)) } else { None });
    contributions.push(Contribution { piece: "the choice among the moves held equal (the rule, the estimates)", step: 6, seconds: t0.elapsed().as_secs_f64(), named: vec![], note: String::new() });
    let mut mv = chosen.as_ref().map(|c| c.mv.clone()).or_else(|| ln.first.clone());
    // ---- PLAN 79.12 B: the last layer among the moves held equal (the wall: a first move the equal reading dropped is not a candidate)
    let mut override_why: Option<(rule::Decided, String)> = None;
    // ---- PLAN 79.17 D: among the moves the calculation holds equal, the first move of the line that lowers the held measure (the wall stands: a move the calculation holds worse is never played for the measure; their largest threat is answered first, as the rule does)
    // ---- PLAN 79.18 C: the goal sleeps while the search holds the end of the game (it is played) and is woken at the next decision in which the search no longer holds it; only a class that changed ends it for good
    if let (Some(h), Some(ml)) = (held_next.as_mut(), measure_log.as_mut()) {
        if h.dropped.is_none() {
            if decided && h.asleep.is_none() { h.slept += 1; h.asleep = Some("the end of the game is found by the search, which plays it".to_string()); ml.said.push_str("; the goal goes to sleep: the search holds the end of the game"); }
            else if !decided && h.asleep.is_some() { h.asleep = None; h.woken += 1; ml.woke = true; ml.said.push_str("; the goal is woken: the search no longer holds the end of the game"); }
            ml.asleep = h.asleep.clone(); ml.slept_total = h.slept; ml.woken_total = h.woken;
        }
    }
    if let (Some(tf), Some(ml)) = (&toward_first, measure_log.as_mut()) {
        let held_by_probe: Vec<String> = probes.iter().filter(|(_, v)| *v >= ln.value).map(|(i, _)| g.move_name(s, &moves[*i])).collect();
        let allowed: Vec<String> = equal.iter().cloned().chain(held_by_probe.into_iter().filter(|m| !equal.contains(m))).collect();
        let answering: Vec<String> = allowed.iter().filter(|m| answers_largest.contains(m)).cloned().collect();
        let pool = if answering.is_empty() { allowed.clone() } else { answering };
        if decided {
            ml.said.push_str("; the search finds the end of the game: it is played");
        } else if pool.contains(tf) {
            ml.played_toward = true;
            override_why = Some((rule::Decided::Measure, if ml.recovery { format!("toward the held goal, a recovery: the measure is above the lowest held and a line of {} of our moves brings it below {:?} against every reply, and the calculation holds this first move equal to the best", ml.moves.unwrap_or(0), ml.value_now) } else { format!("toward the held goal: the measure falls from {:?} within {} of our moves against every reply, and the calculation holds this first move equal to the best", ml.value_now, ml.moves.unwrap_or(0)) }));
            mv = Some(tf.clone());
        } else {
            ml.said.push_str("; the move toward the goal is not among those the calculation holds equal (or answers their threat): it is not played, the wall stands");
        }
    }
    if let Some(spec) = &inp.chooser {
        let t0 = std::time::Instant::now();
        // (PLAN 79.15 B: a first move the equal reading put lower is no longer in `equal` at all, so nothing is dropped here)
        let cands: Vec<String> = equal.clone();
        if cands.len() >= 2 {
            let cells = g.cells(s);
            let standing_row = their.first();
            let threat_stake = if line_stands { f64::INFINITY } else { standing_row.map_or(0.0, |r| r.unit.map_or(f64::INFINITY, |u| stakes.of(u))) };
            let facts: Vec<crate::chooser::MoveFacts> = cands.iter().map(|name| {
                let i = moves.iter().position(|m| &g.move_name(s, m) == name);
                let fix = prop.goals.iter().enumerate().filter(|(_, x)| x.side == me && !x.unmet.is_empty() && rule::fix_moves(x).contains(name))
                    .min_by_key(|(_, x)| (rule::things_in_the_way(x), std::cmp::Reverse(x.depth), x.target.at))
                    .map(|(gi, x)| crate::chooser::FixFacts::new(rule::stake_of(&prop.goals, gi, &stakes), rule::things_in_the_way(x), rule::things_in_the_way(&prop.goals[rule::chain_top(&prop.goals, gi)]), goal_chance(x, &stakes).map(|e| e.chance)));
                let taking = i.and_then(|i| { let to = g.move_ends(s, &moves[i]).1; cells.iter().find(|(o, _, at)| *at == to && *o != me).map(|(o, k, at)| { let u = UnitRef { owner: *o, kind: *k, at: *at }; if Some(*k) == inp.goal { f64::INFINITY } else { crate::threats::kind_worths(g, s, rec, &[u]).get(k).copied().unwrap_or(0.0) } }) });
                let af = after.rows.iter().find(|r| &r.mv == name).and_then(|r| r.fact.as_ref()).map(|f| crate::chooser::AfterFacts { before: f.in_the_way_before, after: f.in_the_way_after, took_out: f.took_out.len(), put_in: f.put_in.len() });
                let gain = after.rows.iter().find(|r| &r.mv == name).and_then(|r| r.gain).zip(after.before).map(|(a, b)| { let (o, t) = (a.ours.less(&b.ours), a.theirs.less(&b.theirs)); crate::chooser::GainFacts { ours: o.finite, theirs: t.finite, ours_goal: o.goal, theirs_goal: t.goal } });
                crate::chooser::MoveFacts { is_first: Some(name) == ln.first.as_ref(), first_held: first_held != Some(false), answers_threat: answers_largest.contains(name), threat_stake, fix, reasons_score: sc.as_ref().and_then(|v| v.iter().find(|x| &x.mv == name)).map(|x| x.score), after: af, taking, gain }
            }).collect();
            let rows: Vec<Vec<f64>> = facts.iter().map(crate::chooser::inputs_of).collect();
            let fixed_ix = cands.iter().position(|c| Some(c) == mv.as_ref());
            let ap = crate::chooser::apply(spec, &rows, fixed_ix);
            let played = cands[ap.index].clone();
            let moved = spec.mode != crate::chooser::Mode::Fixed;
            if moved {
                override_why = Some(match spec.mode {
                    crate::chooser::Mode::Explore => (rule::Decided::Chooser, format!("explore: one of the {} moves the search holds equal, drawn at random from the seed (propensity 1 in {}); the rule would pick {}", cands.len(), cands.len(), mv.clone().unwrap_or_default())),
                    crate::chooser::Mode::Gain => (rule::Decided::Chooser, format!("the gain: among the {} moves the search holds equal{}, the largest gain of our chains' stake times chance less theirs ({:.4}); the rule would pick {}", cands.len(), if cands.iter().any(|c| answers_largest.contains(c)) { " that answer the standing threat" } else { "" }, ap.sums[ap.index], mv.clone().unwrap_or_default())),
                    _ => (rule::Decided::Chooser, format!("the chooser: the largest weighted sum of its inputs among the {} moves the search holds equal ({:.4}); the rule would pick {}", cands.len(), ap.sums[ap.index], mv.clone().unwrap_or_default())),
                });
            }
            log.chooser = Some(crate::log::ChooserLog { mode: format!("{:?}", spec.mode), seed: spec.seed, inputs: crate::chooser::INPUTS.iter().map(|x| x.to_string()).collect(),
                candidates: cands.iter().cloned().zip(rows.into_iter().zip(ap.sums.iter().copied())).map(|(c, (r, x))| (c, r, x)).collect(), fixed_pick: mv.clone(), played: Some(played.clone()), propensity: ap.propensity, applied: moved });
            if moved { mv = Some(played); }
        } else {
            log.notes.push("the last layer: fewer than two moves held equal, nothing to choose among".into());
        }
        contributions.push(Contribution { piece: "the last layer among the moves held equal", step: 6, seconds: t0.elapsed().as_secs_f64(), named: vec![], note: String::new() });
    }
    // PLAN 79.10 C: the reading of the named moves for the chain of claims comes AFTER the search, where the clock of the whole decision allows: the move played first, then the others named, in order; what the clock did not reach is listed (a gap)
    if p.direct_names {
        let t0 = std::time::Instant::now();
        let mut order: Vec<String> = mv.iter().cloned().collect();
        order.extend(named_mine.iter().filter(|n| Some(*n) != mv.as_ref()).cloned());
        let r = rule::keep_read_by(g, s, rec, &order, &[], count, LineParams { depth: tp.sweep_depth, fuse: tp.sweep_fuse }, inp.share_table, Some(wall_end));
        contributions.push(Contribution { piece: "the lines of the named moves read for the claims, after the search", step: 7, seconds: t0.elapsed().as_secs_f64(), named: vec![], note: format!("{} lines read, {} refuted, {} not reached by the clock", r.read, r.refuted.len(), r.unlisted.len()) });
        kept.unlisted.extend(r.unlisted);
        kept.read = r.read; kept.nodes = r.nodes; kept.seconds = r.seconds; kept.mine = r.mine; kept.refuted = r.refuted;
    }
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
        // PLAN 79.16 D: with the table of play the goals a move serves are ranked by stake times chance (`rule::rank_cmp`, the one order of the names and of the fixes among the moves held equal); without it by stake alone, as before
        if inp.chance_order && goal_table.is_some() { v.sort_by(|a, b| rule::rank_cmp((a.0, by_chance(&prop.goals[a.1])), (b.0, by_chance(&prop.goals[b.1])), stakes.ceiling).then(rule::things_in_the_way(&prop.goals[a.1]).cmp(&rule::things_in_the_way(&prop.goals[b.1])))); }
        else { v.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal).then(rule::things_in_the_way(&prop.goals[a.1]).cmp(&rule::things_in_the_way(&prop.goals[b.1])))); }
        v
    };
    let mut decided_by: Option<rule::Decided> = None;
    let nonempty = |x: &str, fallback: &str| if x.trim().is_empty() { fallback.to_string() } else { x.to_string() };
    let mut chain: Vec<Claim> = vec![];
    if let (Some(c), Some(m)) = (&chosen, &mv) {
        let own = kept.mine.iter().find(|k| &k.name == m);
        let standing: Option<String> = if line_stands { line_text.clone() } else { their.first().map(|r| r.unit.map_or_else(|| "the end of the game forced against us".to_string(), |u| text(&u))) };
        let answers = answers_largest.contains(m);
        let sv: Vec<String> = serves_of(m).into_iter().map(|x| x.2).collect();
        let wi = rule::WhyIn { chosen: c, value: ln.value, depth: ln.depth, own, standing: standing.clone(), answers, serves: sv };
        let (kind, why) = match (&override_why, &ln.first) {
            (Some(o), _) => o.clone(),
            (None, Some(f)) if f != m => rule::why_against_first(&wi, &rule::FirstFacts { name: f.clone(), held: first_held, lowered_by: first_lowered_by, answers: answers_largest.contains(f), fixes: fixes.contains(f), own_gain: kept.mine.iter().any(|k| &k.name == f && k.verdict == crate::threat_verdict::VerdictKind::Confirmed && k.net > 0) }),
            _ => rule::why_of(&wi),
        };
        decided_by = Some(kind);
        // the search's own line of the move chosen (the table's line from it, not the line of its first move when another equal move is played)
        let (mline, _) = if Some(m) == ln.first.as_ref() { (ln.line.clone(), vec![]) } else { moves.iter().position(|x| &g.move_name(s, x) == m).map_or((vec![], vec![]), |i| ls.line_from(s, i, ln.depth.max(1) as usize + 2)) };
        let search_end = if decided { format!("the game itself ends in the line (concrete): {}", if ln.value > 0 { "won by us" } else { "lost by us" }) } else { format!("the settled end: no capture is left to either side after the depth read in full ({} plies)", ln.depth) };
        let by_search = if decided { "search: the end of the game forced" } else { "search" };
        chain.push(Claim { line: mline, ended: search_end, by: by_search });
        match kind {
            rule::Decided::OwnGain => if let Some(k) = own { chain.push(Claim { line: k.line.clone(), ended: nonempty(&k.ended, "its own line was read; how it ended was not recorded"), by: "its own line read: it wins by its own reading" }); },
            rule::Decided::AnswersThreat => {
                if line_stands && answers { chain.push(Claim { line: vec![m.clone()], ended: "after it no unit of theirs bears on our goal unit (the graph and the legal moves)".into(), by: "threat: a line of theirs stands on our goal unit and this move answers it" }); }
                else if let Some(x) = their.first().and_then(|r| r.responses.iter().find(|x| x.names.first() == Some(m))) { chain.push(Claim { line: x.names.clone(), ended: nonempty(&x.end_reason, &x.why), by: "threat list: an answer to a threat of theirs that stands" }); }
                else if let Some(k) = own { chain.push(Claim { line: k.line.clone(), ended: nonempty(&k.ended, "its own line was read; how it ended was not recorded"), by: "threat list: an answer to a threat of theirs that stands" }); }
                else { chain.push(Claim { line: vec![m.clone()], ended: "the move is among the answers the goal of theirs names; its own line was not read, so no end is claimed".into(), by: "threat list: an answer to a threat of theirs that stands" }); }
            }
            rule::Decided::FixesGoal => { if let Some(k) = own { chain.push(Claim { line: k.line.clone(), ended: nonempty(&k.ended, "its own line was read; how it ended was not recorded"), by: "goal: a fix of a goal" }); } }
            rule::Decided::Measure => chain.push(Claim { line: vec![m.clone()], ended: "the first move of a line within d of our moves after which the held measure is lower against every reply (the search of toward.rs); the calculation holds it equal to the best, so it is played (never a value)".into(), by: "measure: toward the held goal" }),
            rule::Decided::Chooser => chain.push(Claim { line: vec![m.clone()], ended: "the last layer chose among the moves held equal (their inputs and weighted sums are in the log; never a value)".into(), by: "chooser: the last layer" }),
            rule::Decided::Estimate => chain.push(Claim { line: vec![m.clone()], ended: "the reasons' score decides among the moves held equal (an estimate, never a value)".into(), by: "estimate: the reasons' pick" }),
            _ => {}
        }
        log.why = why;
    } else { log.why = match &override_why { Some((rule::Decided::Measure, w)) => w.clone(), _ => "no move held equal: the search's own move".to_string() }; chain.push(Claim { line: ln.line.clone(), ended: if decided { "the game itself ends in the line (concrete)".to_string() } else { format!("the settled end: no capture is left to either side after the depth read in full ({} plies)", ln.depth) }, by: "search" }); }
    let _ = (decided_by, fix_chain);
    log.chosen = mv.clone();
    let goal_log = |x: &Goal, chosen: bool| GoalLog {
        side: roles[x.side].clone(), target: text(&x.target), source: x.source.clone(), depth: x.depth, parent: x.parent.as_ref().map(&text), via: x.via.map(|v| v.to_string()), unmet: x.unmet.iter().map(|u| u.to_string()).collect(), stoppers: x.stoppers.iter().map(&text).collect(),
        fixes: x.fixes.iter().map(|f| (format!("{:?}", f.kind), f.moves.iter().take(8).cloned().collect())).collect(), in_the_way: rule::things_in_the_way(x),
        chance: goal_chance(x, &stakes).map(|e| (e.chance, e.n, e.source.to_string())), chosen, goal_unit: x.goal_unit,
    };
    log.goals = prop.goals.iter().enumerate().map(|(i, x)| goal_log(x, Some(i) == chosen_ix)).collect();
    if chances.is_none() && goal_table.is_none() && p.goals { log.notes.push("no table of the chance of a goal given: no estimate per goal".into()); }
    log.notes.push("no table of frames given: no estimate per frame".into());
    log.chosen_goal = chosen_ix.map(|i| goal_log(&prop.goals[i], true));
    // (PLAN 79.11: the climb to the far goal takes at most one step a goal: two goals that name each other as the parent made it loop for ever, and a game in play hung on it)
    if let Some(i) = chosen_ix { let mut x = &prop.goals[i]; for _ in 0..prop.goals.len() { let Some(par) = x.parent else { break }; match prop.goals.iter().find(|y| y.target == par && y.side == x.side) { Some(y) => x = y, None => break } } if !std::ptr::eq(x, &prop.goals[i]) { log.far_goal = Some(format!("{} ({})", text(&x.target), x.unmet.join(", "))); } }
    log.their_threat = if line_stands { line_text.clone() } else { their.first().map(|r| r.unit.map_or_else(|| "the end of the game".to_string(), |u| text(&u))) };
    let row_log = |name: &str, r: &ThreatRow| RowLog { list: name.to_string(), unit: r.unit.map_or_else(|| "the end of the game".to_string(), |u| text(&u)), class: r.class.clone(), verdict: format!("{:?}", r.verdict), net: r.net, ended: r.ended.to_string(), line: r.line.clone(),
        responses: r.responses.iter().map(|x| (x.names.first().cloned().unwrap_or_default(), x.state.word().to_string(), x.net)).collect(), carried: r.carried || carried_out.as_ref().is_some_and(|c| c.held.iter().any(|h| h.row.unit == r.unit && r.unit.is_some())) };
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
            for f in &gl.facts { if f.fixes.iter().any(|fx| fx.moves.contains(&name)) { serves.push(format!("fix of {} ({})", text(&gl.target), f.name)); if let Some(e) = goal_chance(gl, &stakes) { let c = e.chance; chance = Some(chance.map_or(c, |o: f64| o.max(c))); } } }
        }
        serves.dedup();
        let mut ans: Vec<String> = vec![];
        for r in their.iter() {
            let by_resp = r.responses.iter().any(|x| x.state == RespState::Kept && x.names.first() == Some(&name));
            let by_fact = r.unit.is_some_and(|u| prop.defence.iter().any(|(t, a)| *t == u && a.iter().any(|x| x.mv == name)));
            if by_resp || by_fact { ans.push(format!("answers the threat on {}", r.unit.map_or_else(|| "the end of the game".to_string(), |u| text(&u)))); }
        }
        if line_stands && line_answers.contains(&name) { ans.push("answers the line of theirs standing on our goal unit".to_string()); }
        let rs = sc.as_ref().and_then(|v| v.iter().find(|x| x.mv == name));
        let _ = m;
        mlog.push(MoveLog { mv: name.clone(), value, equal: equal.contains(&name), line, ended: if value.is_some_and(|v| v.abs() >= WIN - 1000) { "the game itself ends in the line".into() } else { format!("the settled end after the depth read ({} plies)", ln.depth) },
            named: named_mine.contains(&name), own_read: kept.mine.iter().find(|k| k.name == name).map(|k| (format!("{:?}", k.verdict), k.net, k.ended.clone())), serves, answers: ans, chance, reasons_score: rs.map(|x| x.score), reasons_carried: rs.map(|x| x.carried.len()),
            their_after: after.rows.iter().find(|r| r.mv == name).map(|r| their_after_log(r, &text)) });
    }
    mlog.sort_by_key(|m| (!m.equal, m.value.map_or(i64::MAX, |v| -v)));
    log.moves = mlog;
    log.after_not_asked = after.not_asked.clone();
    let mut carried_next = if inp.carry_graph && p.carried {
        // the neighbourhood of every unit the every-unit reading read, in this position's atoms
        let mut every_sets: std::collections::BTreeMap<UnitRef, crate::frame_rows::ReadSet> = Default::default();
        if let Some((_, dg)) = &digest_now { for r in prop.list.mover.iter().chain(&prop.list.other).chain(&prop.list.exchange).chain(&prop.list.dropped).filter(|r| r.every && !r.goal) { if let Some(u) = r.unit { let (cells, places) = dg.neighbourhood((u.owner, u.kind, u.at), u.at); every_sets.insert(u, crate::frame_rows::ReadSet { mover: me, members: cells, places }); } } }
        let empty = Default::default();
        Some(Carried::of_every(&prop.list, frames.as_ref().map_or(&empty, |f| &f.read_sets), &every_sets, digest_now.clone()).with_memo(frames.as_ref().and_then(|f| f.memo.clone())))
    } else { frames.as_ref().map(|f| Carried::of(&prop.list, &f.read_sets)) };
    contributions.push(Contribution { piece: "the whole decision", step: 0, seconds: t_all.elapsed().as_secs_f64(), named: vec![], note: String::new() });
    log.first_held = first_held; log.first_lowered_by = first_lowered_by;
    // PLAN 79.15 B: how the first move stood, and what the set of moves held equal was made from
    log.first_lowered = match cand.lowered { crate::wall::Lowered::No => None, crate::wall::Lowered::ByProbe => Some("probe".to_string()), crate::wall::Lowered::ByVerification => Some("verification".to_string()) };
    log.equal_unverified = cand.unverified;
    log.equal_from = match cand.held { crate::wall::Held::AsBefore => "as_before", crate::wall::Held::Verified => "verified", crate::wall::Held::ProbeOnly => "probe_only", crate::wall::Held::Highest => "highest_probe" }.to_string();
    if first_lowered { log.notes.push(format!("the search's first move was put lower by the equal reading ({}): it is not a candidate{}", log.first_lowered.clone().unwrap_or_default(), if cand.unverified { "; the clock cut the verification, so the moves whose probe held are the candidates, unverified" } else if cand.held == crate::wall::Held::Highest { "; no other move holds, so the move with the highest probe is the candidate" } else { "" })); }
    log.failed_first = ls.failed_first.as_ref().map(|f| crate::log::FailedFirstLog { from_depth: f.depth - 1, depth: f.depth, first_value: f.first_value, goal: f.goal, outcome: f.outcome.to_string(), seconds: f.seconds, nodes: f.nodes, allowed: FAIL_OF_TIE * tie });
    log.depth_reads = ls.reads.iter().map(|r| (r.depth, r.complete, r.nodes, r.seconds)).collect();
    // PLAN 79.15 A (a measure, never in play): why the probe and the search disagree about a lowered first move
    if inp.diag_first && first_lowered {
        if let Some(o) = own {
            if let Some(pv) = probes.iter().find(|(i, _)| *i == o).map(|(_, v)| *v) {
                ls.resume(DIAG_SECONDS);
                let un = ls.verify(s, ln.depth, ln.value, &[o]);
                let unreduced_holds = if ls.cut() { None } else { Some(!un.is_empty()) };
                let mut deeper = vec![];
                for k in 1..=2 { ls.resume(DIAG_SECONDS); let pr = ls.probes(s, ln.depth + k, ln.value, Some(&[o])); deeper.push((ln.depth + k, if ls.cut() { None } else { pr.first().map(|x| x.1) })); }
                log.first_diag = Some(crate::log::FirstDiag { probe: pv, value: ln.value, unreduced_holds, deeper });
            }
        }
    }
    // PLAN 79.17 D: the held goal is carried to the next decision with the open tactics (a carry is made for it where the frames made none)
    if let Some(h) = held_next { carried_next.get_or_insert_with(Carried::default).held = Some(h); }
    log.measure = measure_log;
    log.step_seconds = contributions.iter().map(|c| (c.step, c.piece.to_string(), c.seconds)).collect();
    WakeDecision { mv: mv.clone(), value: ln.value, depth: ln.depth, nodes: ln.nodes, chain, contributions, unported: params.pieces.unported(), log, proposal: Some(prop), game: { let mut game = kept_after(st, g, s, inp, mv.as_ref(), inp.memory.game.clone()); game.put_slot(crate::memory::CARRIED, carried_next.map(|c| std::sync::Arc::new(c) as std::sync::Arc<dyn cs_memory::store::Kept>), "the open tactics and the goal held after this decision"); game }, named_mine, named_theirs, equal, first: ln.first.clone() }
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
            "move": self.mv, "value": self.value, "depth": self.depth, "nodes": self.nodes, "unported": self.unported, "named_mine": self.named_mine, "named_theirs": self.named_theirs, "equal": self.equal, "first": self.first,
            "chain": self.chain.iter().map(|c| serde_json::json!({"line": c.line, "ended": c.ended, "by": c.by})).collect::<Vec<_>>(),
            "steps": self.contributions.iter().map(|c| serde_json::json!({"piece": c.piece, "step": c.step, "seconds": c.seconds, "note": c.note, "named": c.named})).collect::<Vec<_>>(),
            "log": self.log.to_json(),
        })
    }
}
