//! PLAN 79.06 THE CONCRETE THREAT LIST IN RUST (SYSTEM_SPEC 8 "The threat list" steps 2 and 3; 4 how a line ends and the four reasons; 11 Rust in the hot paths; piece: the wake). The Rust form of catspace/incubator/analyst/discovery/threat_concrete.py
//! (`concrete_lists`, `response_of`), of subgoal_units.py (`every_unit`, `stake_closure`) and of threat_measure.stake_of_kinds; the Python stays as the mirror. Game-free: a unit, a kind, a place is a number or a word the game gave; nothing of any game is named here.
//! No oracle, no external model, no worth assumed: the only worths are the caller's measured `Record`. The list ORDERS and SAYS what is read in full; it never adds to the judge's value (SYSTEM_SPEC 8, rule 7).
//!
//! INTERFACE (written before the bodies).
//!   `concrete_threats(st, g, s, record, params) -> ConcreteList`   takes the graph's static part, the game, the position, the measured record and `ThreatParams`; returns the list of BOTH sides: `mover` (rows against the mover's units), `other` (against the other
//!       side's), `exchange` apart, `dropped` (kept for the proposer, each with the code of why), the goal rows (the end of the game forced, either side), and the cost of each round. Every row is one LINE read to a settled end (`cs_tactics::lines`), judged
//!       by `threat_verdict`; a CONFIRMED row owes responses, each read two ways and judged by `response_of`. Called by: the wake (step 3) and the unit tests. No player calls it.
//!   `every_unit(st, g, s, list, sides, record, params)`            adds one row for every unit of the other side that the list has no row for (the line read the same way); changes `list` in place. `concrete_threats` calls it for `params.every_sides`.
//!   `response_of(after, passed, record, how, source, row_net)`    one response read two ways and judged: KEPT, STILL WON, WORSE THAN PASSING, COSTS MORE THAN THE THREAT or UNREAD.
//!   `stake_of_kinds(read, k)`                                      the stake of each kind of each owner from a kinds head the CALLER reads (`read(K) -> us less them`); the head itself is the heads piece, not this one.
//!   `ThreatParams`                                                 the depths and fuses (named, as in Python), the goal kind (no row for it in `every_unit`), the candidate rows of the frame read if the caller has them (`rows`), the sides `every_unit` serves, the stake function.
//! BATCHES. The line reads run in the Python's rounds, each round one list of queries over all the rows (never one call a row): round 1 the taking lines of every row and the forced end of each side; round 2 the quiet moves that make a threat (rows with no taking),
//! the best line of the taker once a side, the responses the frame read named, the answers to a forced end; round 3 the sweep of every legal move of the owner once for every place; round 4 the sweep's cut moves again with the fuse of a line. The reads run in one process; no thread is spawned here (the caller's `batch` spreads positions over threads).
//! DIFFERENCES FROM THE PYTHON (listed in the 79.06 BUILT block): the candidate rows come from the frame read when the caller supplies them (`rows`) and otherwise from the graph (every unit an opposing unit bears on, no frame class, no named answers: the sweep reads every
//! legal move anyway); the read of the game's pass is `Game::pass`; the kind-change event is always in the judge (the Python's `kind_change=False` ablation is not ported).
use crate::threat_verdict::{best_taking, verdict_of, Record, Verdict, VerdictKind};
use crate::wake::UnitRef;
use cs_game::placed::{Kind, Place, Placed};
use cs_game::{Game, Role};
use cs_tactics::ledger::{Search, Worth};
use cs_tactics::lines::{self as L, After, Forced, LineParams, LineRead};
use cs_tactics::sheet::base_graph::{PositionPart, Root, Static};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::time::Instant;

/// The depth in plies after the first move and the node fuse of one line; the sweep's smaller ones; the plies and fuse of the search for the end of the game (named fuses: a line one of them cuts is UNREAD).
pub const DEPTH: i32 = 6;
pub const FUSE: u64 = 40_000;
pub const SWEEP_DEPTH: i32 = 4;
pub const SWEEP_FUSE: u64 = 12_000;
pub const END_PLIES: u32 = 7;
pub const END_FUSE: u64 = 1_500_000;
/// The four reasons a line ended (spec 4), as this build maps them: concrete, quiet, lost beyond doubt (not produced here), out of budget.
pub const ENDED: [&str; 4] = ["concrete", "quiet", "lost beyond doubt", "out of budget"];
/// How a response was found, by the code of the frame read's answers.
pub const WHY: [&str; 5] = ["a taking", "the unit moves", "a guard is added", "a counter threat", "a unit is put between"];

/// A candidate row from the frame read (threat_list.threat_rows without the verdict): the unit, its class word, the units that bear on it, and the answers the read named as (from, to, how, the unit moves).
#[derive(Debug, Clone)]
pub struct FrameRow { pub unit: UnitRef, pub class: String, pub by: Vec<UnitRef>, pub response_ix: Vec<(Place, Place, u8, bool)>, pub stake: Option<f64> }

/// The parameters of the list. `rows` None: the candidates are read from the graph. `every_sides` None: the mover's goals are wanted (the targets are the units of the other side). `stake`: the caller's stake of a kind (word, owner is the mover).
pub struct ThreatParams<'a> {
    pub depth: i32, pub fuse: u64, pub sweep_depth: i32, pub sweep_fuse: u64, pub end_plies: u32, pub end_fuse: u64,
    pub goal: Option<Kind>, pub rows: Option<Vec<FrameRow>>, pub every_sides: Option<Vec<Role>>, pub stake: Option<&'a dyn Fn(&str, bool) -> Option<f64>>,
    /// PLAN 79.08 repair 2: the lines read from one position share one table (`cs_tactics::lines::*_in`), one search a (position, side). false (the default) is each line with a table of its own, as 79.06 read them.
    pub share_table: bool,
    /// PLAN 79.09 B (SYSTEM_SPEC 7, the budget is questions; piece: the wake): the clock of this step (None: the fuses only, as before). With it every line read goes through a shared search that stops at it (a line it cuts is UNREAD, a gap, never a refutation), the rows are read in the order given (the caller's order of stake),
    /// the forced end of the game, the unit of the largest stake, is read first, and what the clock did not reach is listed (`ConcreteList::unasked`, `not_asked`, `ThreatRow::responses_unasked`).
    pub deadline: Option<Instant>,
    /// PLAN 79.22 PART A: the decision's root graph, lent by the wake (None: the position's graph is built where it is read, as before).
    pub root: Option<&'a Root>,
}

/// The graph of `s` for a part that may be lent the decision's root (PLAN 79.22 PART A).
fn graph_of<G: Game + Placed>(st: &Static, root: Option<&Root>, g: &G, s: &G::State) -> std::rc::Rc<PositionPart> {
    match root { Some(r) => st.position_in(r, g, s), None => std::rc::Rc::new(st.position(g, s)) }
}

impl<'a> Default for ThreatParams<'a> {
    fn default() -> Self { ThreatParams { depth: DEPTH, fuse: FUSE, sweep_depth: SWEEP_DEPTH, sweep_fuse: SWEEP_FUSE, end_plies: END_PLIES, end_fuse: END_FUSE, goal: None, rows: None, every_sides: None, stake: None, share_table: false, deadline: None, root: None } }
}

/// What each side is left with after a line (the taker's units lost and taken, and what is left once like for like cancels).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Outcome { pub taker_lost: Vec<String>, pub taker_won: Vec<String>, pub left_lost: Vec<String>, pub left_won: Vec<String>, pub cancelled: usize }

/// The state of a response after it is read.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum RespState { Kept, StillWon, WorseThanPassing, CostsMoreThanTheThreat, Unread }
impl RespState {
    pub fn word(self) -> &'static str { match self { RespState::Kept => "KEPT", RespState::StillWon => "STILL WON", RespState::WorseThanPassing => "WORSE THAN PASSING", RespState::CostsMoreThanTheThreat => "COSTS MORE THAN THE THREAT", RespState::Unread => "UNREAD" } }
}

/// Where a response came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Source { FrameRead, Sweep, ForcingRead }

/// One response of the owner, a line read the same way: the whole-board line of the move and the taking of the unit after it.
#[derive(Debug, Clone)]
pub struct Response {
    pub first: (Place, Place), pub names: Vec<String>, pub net: i64, pub taker_lost: Vec<String>, pub taker_won: Vec<String>, pub resp_takes: Vec<String>, pub resp_loses: Vec<String>, pub cancelled: usize, pub loses_nothing: bool,
    pub ended: &'static str, pub end_reason: String, pub depth: i32, pub settled: usize, pub end: u8, pub nodes: u64, pub state: RespState, pub why: String, pub how: String, pub source: Source, pub after: Option<(VerdictKind, Vec<String>, i64)>,
}

/// One row of the concrete threat list: a unit of one side that the other side takes, with the verdict of the best taking line, the stake and the kept responses; or a row of the goal (`goal`, `unit` None).
#[derive(Debug, Clone)]
pub struct ThreatRow {
    pub owner: Role, pub unit: Option<UnitRef>, pub goal: bool, pub every: bool, pub class: String, pub by: Vec<UnitRef>, pub stake: Option<f64>,
    pub verdict: VerdictKind, pub ended: &'static str, pub reason: String, pub net: Option<i64>, pub line: Vec<String>, pub lead: Option<(Place, Place)>, pub first: Option<(Place, Place)>, pub two_moves: bool, pub depth: i32,
    pub outcome: Option<Outcome>, pub resolution: f64, pub pv_disagrees: bool, pub nodes: u64,
    pub responses: Vec<Response>, pub resp_read: usize, pub resp_unread: usize, pub resp_unread_moves: Vec<String>, pub resp_states: BTreeMap<&'static str, usize>, pub deferred: bool, pub response_ix: Vec<(Place, Place, u8, bool)>,
    pub why: Option<&'static str>, pub dropped_verdict: Option<VerdictKind>,
    /// PLAN 79.09 B: the clock stopped before the responses of this row were read (they are not known, which is not the same as none)
    pub responses_unasked: bool,
    /// PLAN 79.14 B: the row was carried from the last decision (its read set untouched by the two plies since), not read again
    pub carried: bool,
}

impl ThreatRow {
    pub fn new(owner: Role, unit: Option<UnitRef>, class: &str, by: Vec<UnitRef>, stake: Option<f64>) -> ThreatRow {
        ThreatRow { owner, unit, goal: false, every: false, class: class.to_string(), by, stake, verdict: VerdictKind::Unread, ended: ENDED[3], reason: String::new(), net: None, line: vec![], lead: None, first: None, two_moves: false, depth: 0, outcome: None, resolution: 0.0, pv_disagrees: false, nodes: 0,
            responses: vec![], resp_read: 0, resp_unread: 0, resp_unread_moves: vec![], resp_states: BTreeMap::new(), deferred: false, response_ix: vec![], why: None, dropped_verdict: None, responses_unasked: false, carried: false }
    }
}

/// The time and work of one round of line reads.
#[derive(Debug, Clone, Default)]
pub struct RoundCost { pub name: &'static str, pub seconds: f64, pub queries: u64, pub nodes: u64 }

/// The concrete threat list of a position, both sides (rows against the mover's units, rows against the other side's, exchanges apart, dropped rows kept for the proposer, the rows of the goal), the counts Python keeps, and the cost of each round.
#[derive(Debug, Clone, Default)]
pub struct ConcreteList {
    pub mover: Vec<ThreatRow>, pub other: Vec<ThreatRow>, pub exchange: Vec<ThreatRow>, pub dropped: Vec<ThreatRow>, pub goal: Vec<ThreatRow>,
    /// PLAN 79.07: the units the budget of questions did not ask (no row: a gap, never a refutation).
    pub unasked: Vec<UnitRef>, pub not_asked: Vec<String>, pub dropped_counts: BTreeMap<String, usize>, pub resp_read: usize, pub resp_dropped: usize, pub resp_sweeps: usize, pub resp_reread: usize, pub unread_rows: usize, pub cost: Vec<RoundCost>,
}

impl ConcreteList {
    pub fn seconds(&self) -> f64 { self.cost.iter().map(|c| c.seconds).sum() }
    fn add_cost(&mut self, name: &'static str, t0: Instant, queries: u64, nodes: u64) { self.cost.push(RoundCost { name, seconds: t0.elapsed().as_secs_f64(), queries, nodes }); }
}

// ------------------------------------------------------------------------------------------------------------------------------ the readings (one reader per position)

/// The line reader of ONE position: the position, its pass, the judge made from the record. Every call is one query of the Python's modes; the counters are the cost. With `share` the lines read for one (position, side) go through one `Search` and so one table (PLAN 79.08 repair 2).
struct Rd<'a, G: Game + Placed> { g: &'a G, s: &'a G::State, passed: Option<G::State>, judge: &'a Worth, mover: Role, queries: u64, nodes: u64, share: bool, shared: HashMap<(bool, Role), Search<'a, G, Worth>>, stop: Option<Instant> }

impl<'a, G: Game + Placed> Rd<'a, G> {
    fn new(g: &'a G, s: &'a G::State, judge: &'a Worth, share: bool, stop: Option<Instant>) -> Rd<'a, G> {
        Rd { g, s, passed: g.pass(s), judge, mover: g.mover(s).unwrap_or(0), queries: 0, nodes: 0, share: share || stop.is_some(), shared: HashMap::new(), stop }
    }
    /// the clock of the step has run out
    fn expired(&self) -> bool { self.stop.is_some_and(|t| Instant::now() >= t) }
    /// The position the query is asked at: the position itself, or the one after the mover passes once (None where the game gives no pass).
    fn at(&self, pass: bool) -> Option<&G::State> { if pass { self.passed.as_ref() } else { Some(self.s) } }
    fn lines(&mut self, v: &[LineRead]) { self.queries += 1; self.nodes += v.iter().map(|r| r.nodes).sum::<u64>(); }
    fn takings(&mut self, pass: bool, place: Place, p: LineParams) -> Option<Vec<LineRead>> {
        let p0 = self.at(pass)?; let me = self.g.mover(p0)?;
        let r = if self.share { let (g, j) = (self.g, self.judge); let stop = self.stop; let sr = self.shared.entry((pass, me)).or_insert_with(|| { let mut x = Search::lean_with_fuse(g, j, me, 1); if let Some(t) = stop { x.stop_at(t); } x }); let p0 = if pass { self.passed.as_ref()? } else { self.s }; L::read_takings_in(sr, p0, place, me, p) } else { L::read_takings(self.g, self.judge, p0, place, me, p) };
        self.lines(&r); Some(r)
    }
    fn two_moves(&mut self, pass: bool, place: Place, p: LineParams) -> Option<Vec<LineRead>> {
        let p0 = self.at(pass)?; let me = self.g.mover(p0)?;
        let r = if self.share { let (g, j) = (self.g, self.judge); let stop = self.stop; let sr = self.shared.entry((pass, me)).or_insert_with(|| { let mut x = Search::lean_with_fuse(g, j, me, 1); if let Some(t) = stop { x.stop_at(t); } x }); let p0 = if pass { self.passed.as_ref()? } else { self.s }; L::read_two_moves_in(sr, p0, place, me, p) } else { L::read_two_moves(self.g, self.judge, p0, place, me, p) };
        self.lines(&r); Some(r)
    }
    fn best(&mut self, pass: bool, p: LineParams) -> Option<LineRead> {
        let p0 = self.at(pass)?; let me = self.g.mover(p0)?;
        let r = L::read_best_by(self.g, self.judge, p0, me, p, self.stop); self.lines(std::slice::from_ref(&r)); Some(r)
    }
    /// The moves of the mover of the position (after the pass or not) that start and end as `ends` (all moves when None), each read two ways for the OTHER side, as the taker of the unit on `place`.
    fn afters(&mut self, pass: bool, ends: Option<(Place, Place)>, place: Place, p: LineParams) -> Option<Vec<After>> {
        let p0 = self.at(pass)?; let me = self.g.other(self.g.mover(p0)?);
        let g = self.g;
        let moves: Vec<G::Move> = g.moves(p0).into_iter().filter(|m| ends.map_or(true, |e| g.move_ends(p0, m) == e)).collect();
        let r: Vec<After> = if self.share {
            let j = self.judge;
            let p0 = if pass { self.passed.as_ref()? } else { self.s };
            let stop = self.stop;
            let sr = self.shared.entry((pass, me)).or_insert_with(|| { let mut x = Search::lean_with_fuse(g, j, me, 1); if let Some(t) = stop { x.stop_at(t); } x });
            moves.iter().map(|m| L::read_after_in(sr, p0, m, place, me, p)).collect()
        } else { moves.iter().map(|m| L::read_after(g, self.judge, p0, m, place, me, p)).collect() };
        self.queries += 1; self.nodes += r.iter().map(|a| a.whole.nodes + a.takes.iter().map(|t| t.nodes).sum::<u64>()).sum::<u64>(); Some(r)
    }
    fn forced(&mut self, pass: bool, end_plies: u32, end_fuse: u64, until: Option<Instant>) -> Option<Forced> {
        let p0 = self.at(pass)?; let me = self.g.mover(p0)?;
        let r = L::forced_end_by(self.g, p0, me, end_plies, end_fuse, until); self.queries += 1; Some(r)
    }
    fn answers(&mut self, pass: bool, end_plies: u32, end_fuse: u64) -> Option<Vec<((Place, Place), String, u8)>> {
        let p0 = self.at(pass)?; let me = self.g.mover(p0)?;
        let r = L::end_answers_by(self.g, p0, self.g.other(me), end_plies, end_fuse, self.stop); self.queries += 1; Some(r)
    }
}

// ------------------------------------------------------------------------------------------------------------------------------ verdict words and the response

/// How the line ended and why, in the words of the Python (`verdict_of`'s `ended` and `reason`).
pub(crate) fn describe(v: &Verdict) -> (&'static str, String) {
    if v.kind == VerdictKind::Unread { return (ENDED[3], "out of budget: the fuse cut the line; no forced claim is made from it".into()); }
    let tail = if v.end_in_tail { " (reached in the tail of the line)" } else { "" };
    if v.end == L::END_WON { return (ENDED[0], format!("the end of the game, won by the taker, decides it{tail}")); }
    if v.end == L::END_LOST { return (ENDED[0], format!("the end of the game, lost by the taker, decides it{tail}")); }
    (ENDED[1], format!("the settled end: no capture is left to either side after {} plies{}", v.depth, if v.cancelled > 0 { format!("; {} like for like cancelled", v.cancelled) } else { String::new() }))
}

/// Fill a row from the verdict of its best taking line (`threat_concrete._fill`): with `two` the line is the quiet move that makes the threat and what follows.
fn fill(row: &mut ThreatRow, b: &Verdict, two: bool, nodes_all: u64) {
    let (ended, reason) = describe(b);
    row.verdict = b.kind; row.ended = ended; row.reason = reason; row.net = Some(b.net);
    row.line = if two { b.prefix.iter().chain(b.names.iter()).cloned().collect() } else { b.names.clone() };
    row.lead = if two { b.lead } else { None };
    row.outcome = Some(Outcome { taker_lost: b.taker_lost.clone(), taker_won: b.taker_won.clone(), left_lost: b.left_lost.clone(), left_won: b.left_won.clone(), cancelled: b.cancelled });
    row.resolution = b.resolution; row.depth = b.depth; row.first = Some(b.first); row.pv_disagrees = b.net != b.net_search; row.two_moves = two; row.nodes = nodes_all;
}

/// ONE RESPONSE read two ways and judged (threat_concrete.response_of). KEPT only if after it the taker's best taking of the unit is no longer CONFIRMED (or none is legal) and nothing larger is lost by it: the taker's whole-board line is not better for it
/// than what it could count on if the owner passed (`passed`), nor than what the taking of the unit would have won (`row_net`). The response carries its own end (what it takes and loses after like for like cancels).
pub fn response_of(after: &After, passed: Option<&LineRead>, rec: &Record, how: String, source: Source, row_net: Option<i64>) -> Response {
    let whole = &after.whole;
    let wv = verdict_of(whole, rec);
    let tk = best_taking(&after.takes, rec);
    let pnet = passed.map(|p| verdict_of(p, rec).net);
    let tk_unread = tk.as_ref().is_some_and(|t| t.kind == VerdictKind::Unread);
    let tk_confirmed = tk.as_ref().is_some_and(|t| t.kind == VerdictKind::Confirmed);
    let (state, why) = if whole.cut || tk_unread || passed.is_none() || passed.is_some_and(|p| p.cut) { (RespState::Unread, "a fuse cut a line: the response is not judged".to_string()) }
        else if whole.end == 1 || tk_confirmed { (RespState::StillWon, if tk.is_some() { "after it the taker still ends ahead by taking the unit".to_string() } else { "after it the game is lost".to_string() }) }
        else if whole.end != 2 && pnet.is_some_and(|p| wv.net > p) { (RespState::WorseThanPassing, "it saves the unit and the taker ends further ahead elsewhere than if the owner had passed".to_string()) }
        else if whole.end != 2 && row_net.is_some_and(|r| wv.net > r) { (RespState::CostsMoreThanTheThreat, "it saves the unit and loses more than the taking of the unit would have cost".to_string()) }
        else { (RespState::Kept, match &tk { None => "no taking of the unit is legal after it".to_string(), Some(t) => format!("the taking of the unit is {}", format!("{:?}", t.kind).to_uppercase()) }) };
    let (ended, end_reason) = describe(&wv);
    Response { first: whole.first, names: whole.names.clone(), net: wv.net, taker_lost: wv.taker_lost.clone(), taker_won: wv.taker_won.clone(), resp_takes: wv.left_lost.clone(), resp_loses: wv.left_won.clone(), cancelled: wv.cancelled, loses_nothing: wv.left_won.is_empty(),
        ended, end_reason, depth: whole.depth, settled: whole.settled, end: whole.end, nodes: whole.nodes + after.takes.iter().map(|x| x.nodes).sum::<u64>(), state, why, how, source,
        after: tk.map(|t| (t.kind, t.names.clone(), t.net)) }
}

// ------------------------------------------------------------------------------------------------------------------------------ the candidates

/// The candidates from the graph when the caller has no frame read: every unit of either side (not the goal kind) that a unit of the other side bears on.
pub fn graph_rows<G: Game + Placed>(st: &Static, g: &G, s: &G::State, goal: Option<Kind>) -> Vec<FrameRow> { graph_rows_in(st, None, g, s, goal) }

/// `graph_rows` that reads the position's graph from the decision's root when it is lent (PLAN 79.22 PART A).
pub fn graph_rows_in<G: Game + Placed>(st: &Static, root: Option<&Root>, g: &G, s: &G::State, goal: Option<Kind>) -> Vec<FrameRow> {
    let part = graph_of(st, root, g, s);
    let mut out = vec![];
    for (i, u) in part.units.iter().enumerate() {
        if Some(u.kind) == goal { continue; }
        let mut by: Vec<UnitRef> = part.bears.iter().filter(|r| r.place == u.at && part.units[r.unit].owner != u.owner).map(|r| { let b = &part.units[r.unit]; UnitRef { owner: b.owner, kind: b.kind, at: b.at } }).collect();
        by.sort(); by.dedup();
        if by.is_empty() { continue; }
        let _ = i;
        out.push(FrameRow { unit: UnitRef { owner: u.owner, kind: u.kind, at: u.at }, class: "contested".into(), by, response_ix: vec![], stake: None });
    }
    out.sort_by_key(|r| (r.unit.owner, r.unit.at));
    out
}

enum St { Pending, Dropped, Done, Skipped }
struct Work { row: ThreatRow, taker_p: bool, owner_p: bool, st: St, notake: bool, pending: Vec<(u8, Vec<After>)>, resp: Vec<Response>, place: Place }

fn side_is_mover(owner: Role, mover: Role) -> bool { owner == mover }

/// The order of a list: the goal rows first, then the greater stake, then the place.
pub(crate) fn sort_rows(v: &mut [ThreatRow]) {
    v.sort_by(|a, b| {
        let key = |r: &ThreatRow| (if r.goal { 0 } else { 1 }, -(r.stake.unwrap_or(-9.0)), r.unit.map_or(-1i64, |u| u.at as i64));
        let (x, y) = (key(a), key(b));
        x.0.cmp(&y.0).then(x.1.partial_cmp(&y.1).unwrap_or(std::cmp::Ordering::Equal)).then(x.2.cmp(&y.2))
    });
}

fn drop_row(list: &mut ConcreteList, why: &str, row: Option<(&ThreatRow, &'static str)>) {
    *list.dropped_counts.entry(why.to_string()).or_insert(0) += 1;
    if let Some((r, code)) = row { let mut r = r.clone(); r.why = Some(code); list.dropped.push(r); }
}

/// The share of what is left of the clock that one forced-end read of a side may take (a named share, not tuned: two reads leave the rows more than two fifths).
pub const FORCED_SHARE: f64 = 1.0 / 3.0;

/// The forced end of the game for each side (the mover's reading first): a search cut by the clock is a gap and is listed as one.
fn read_forced<G: Game + Placed>(rd: &mut Rd<G>, list: &mut ConcreteList, params: &ThreatParams, hits: &mut Vec<(bool, Forced)>) {
    for p in [false, true] {
        let until = params.deadline.map(|d| { let now = Instant::now(); if d <= now { now } else { now + (d - now).mul_f64(FORCED_SHARE) } });
        match rd.forced(p, params.end_plies, params.end_fuse, until) {
            Some(f @ (Forced::Proved { .. } | Forced::Suspected { .. })) => hits.push((p, f)),
            Some(Forced::Gap { .. }) => drop_row(list, "a forced end of the game read out of budget (a gap, not a refutation)", None),
            _ => {}
        }
    }
}

/// THE CONCRETE THREAT LIST of both sides (see the module doc). Rounds of line reads over all the rows; the rows come from `params.rows` (the frame read) or from the graph; then `every_unit` for `params.every_sides`.
pub fn concrete_threats<G: Game + Placed>(st: &Static, g: &G, s: &G::State, rec: &Record, params: &ThreatParams) -> ConcreteList {
    let mut list = ConcreteList::default();
    let judge = Worth { name: "record".into(), table: rec.worth.clone(), ours_first: false };
    let mut rd = Rd::new(g, s, &judge, params.share_table, params.deadline);
    let mover = rd.mover;
    let kinds = g.kind_names();
    let lp = LineParams { depth: params.depth, fuse: params.fuse };
    let sp = LineParams { depth: params.sweep_depth, fuse: params.sweep_fuse };
    let cands = match &params.rows { Some(r) => r.clone(), None => graph_rows_in(st, params.root, g, s, params.goal) };
    let mut works: Vec<Work> = cands.iter().map(|c| {
        let owner_is_mover = c.unit.owner == mover;
        let stake = c.stake.or_else(|| params.stake.and_then(|f| f(&kinds[c.unit.kind as usize], owner_is_mover)));
        let mut row = ThreatRow::new(c.unit.owner, Some(c.unit), &c.class, c.by.clone(), stake);
        row.response_ix = c.response_ix.clone();
        Work { row, taker_p: owner_is_mover, owner_p: !owner_is_mover, st: St::Pending, notake: false, pending: vec![], resp: vec![], place: c.unit.at }
    }).collect();

    // ---- round 1: the taking lines of every row and the forced end of the game for each side (with a clock: the forced end first, it is the unit of the largest stake; then the rows in the order given, the clock stopping the reading: the rows not reached are listed, never refuted)
    let t0 = Instant::now(); let (q0, n0) = (rd.queries, rd.nodes);
    let mut goal_hits: Vec<(bool, Forced)> = vec![];
    if params.deadline.is_some() { read_forced(&mut rd, &mut list, params, &mut goal_hits); }
    for w in works.iter_mut() {
        if rd.expired() { w.st = St::Skipped; if let Some(u) = w.row.unit { list.unasked.push(u); } continue; }
        match rd.takings(w.taker_p, w.place, lp) {
            None => { w.row.verdict = VerdictKind::Unread; w.row.ended = ENDED[3]; w.row.reason = "the side to move must answer: the game gives no pass, so what the other side would take cannot be read".into(); w.row.net = None; w.st = St::Done; }
            Some(reads) => match best_taking(&reads, rec) {
                None => w.notake = true,
                Some(b) => { let all = reads.iter().map(|r| r.nodes).sum(); fill(&mut w.row, &b, false, all); w.st = St::Done; }
            },
        }
    }
    if params.deadline.is_none() { read_forced(&mut rd, &mut list, params, &mut goal_hits); }
    list.add_cost("round 1: the taking lines and the forced end", t0, rd.queries - q0, rd.nodes - n0);

    // ---- round 2: threats in two moves (no taking move), the responses of CONFIRMED rows, the answers to a forced end of the game
    let t0 = Instant::now(); let (q0, n0) = (rd.queries, rd.nodes);
    let mut passed_value: HashMap<bool, Option<LineRead>> = HashMap::new();
    for w in works.iter_mut() {
        if w.notake && rd.expired() { w.st = St::Skipped; if let Some(u) = w.row.unit { list.unasked.push(u); } continue; }
        if w.notake {
            let best = rd.two_moves(w.taker_p, w.place, lp).and_then(|rs| { let all: u64 = rs.iter().map(|r| r.nodes).sum(); best_taking(&rs, rec).map(|b| (b, all)) });
            match best {
                Some((b, all)) if matches!(b.kind, VerdictKind::Confirmed | VerdictKind::Unread) => { fill(&mut w.row, &b, true, all); w.st = St::Done; }
                other => {
                    let code = if other.is_some() { "two_move_not_ahead" } else if w.row.by.is_empty() { "no_bearer_no_move" } else { "no_taking_legal" };
                    if let Some((b, all)) = other { fill(&mut w.row, &b, true, all); w.row.dropped_verdict = Some(w.row.verdict); }
                    w.st = St::Dropped;
                    let why = if w.row.by.is_empty() { "no unit bears on it and no move of the taker makes a taking stand" } else { "a unit bears on it but no taking is legal" };
                    drop_row(&mut list, why, Some((&w.row, code)));
                }
            }
        } else if matches!(w.st, St::Done) && w.row.verdict == VerdictKind::Confirmed && !w.row.two_moves {
            if rd.expired() { w.row.responses_unasked = true; continue; }
            if !passed_value.contains_key(&w.taker_p) { let b = rd.best(w.taker_p, lp); passed_value.insert(w.taker_p, b); }
            for (fr, to, how, _um) in w.row.response_ix.clone() {
                if let Some(afters) = rd.afters(w.owner_p, Some((fr, to)), w.place, LineParams { depth: params.depth, fuse: params.fuse }) { w.pending.push((how, afters)); }
            }
        }
    }
    let mut goal_resp: Vec<(bool, Forced, Option<Vec<((Place, Place), String, u8)>>)> = vec![];
    for (p, f) in goal_hits { let ans = rd.answers(!p, params.end_plies, params.end_fuse); goal_resp.push((p, f, ans)); }
    for w in works.iter_mut() {
        for (how, afters) in std::mem::take(&mut w.pending) {
            for af in &afters {
                let r = response_of(af, passed_value.get(&w.taker_p).and_then(|x| x.as_ref()), rec, WHY[(how as usize).min(4)].to_string(), Source::FrameRead, w.row.net);
                w.row.nodes += r.nodes; w.resp.push(r);
            }
        }
    }
    list.add_cost("round 2: quiet threats, responses, answers to the forced end", t0, rd.queries - q0, rd.nodes - n0);

    // ---- round 3: the sweep of every legal move of the owner, once for every place (the frame read stops at its first sound answers, so its list is not complete)
    let t0 = Instant::now(); let (q0, n0) = (rd.queries, rd.nodes);
    let mut sweeps: HashMap<(bool, Place), Option<Vec<After>>> = HashMap::new();
    let mut base: HashMap<bool, Option<LineRead>> = HashMap::new();
    let mut shared: HashMap<(bool, Place), u64> = HashMap::new();
    let mut plan3: Vec<usize> = vec![];
    for (i, w) in works.iter().enumerate() {
        if !matches!(w.st, St::Done) || w.row.verdict != VerdictKind::Confirmed || w.row.two_moves { continue; }
        plan3.push(i);
        *shared.entry((w.owner_p, w.place)).or_insert(0) += 1;
    }
    for &i in &plan3 {
        if rd.expired() { works[i].row.responses_unasked = true; continue; }
        let (owner_p, taker_p, place) = (works[i].owner_p, works[i].taker_p, works[i].place);
        if !sweeps.contains_key(&(owner_p, place)) { let a = rd.afters(owner_p, None, place, sp); sweeps.insert((owner_p, place), a); }
        if !base.contains_key(&taker_p) { let b = rd.best(taker_p, sp); base.insert(taker_p, b); }
        list.resp_sweeps += 1;
        let how = format!("found by the sweep of every legal move (read to {} plies)", params.sweep_depth);
        let bs = base[&taker_p].as_ref();
        let n = shared[&(owner_p, place)].max(1);
        let w = &mut works[i];
        for af in sweeps[&(owner_p, place)].iter().flatten() {
            let r = response_of(af, bs, rec, how.clone(), Source::Sweep, w.row.net);
            w.row.nodes += (af.whole.nodes + af.takes.iter().map(|x| x.nodes).sum::<u64>()) / n;
            w.resp.push(r);
        }
    }
    list.add_cost("round 3: the sweep of every legal move", t0, rd.queries - q0, rd.nodes - n0);

    // ---- round 4: a move the sweep's small fuse cut is read again with the fuse of a line; one still cut stays UNREAD and is listed by name
    let t0 = Instant::now(); let (q0, n0) = (rd.queries, rd.nodes);
    let mut again_at: HashMap<(bool, (Place, Place), Place), Option<Vec<After>>> = HashMap::new();
    for &i in &plan3 {
        if rd.expired() { works[i].row.responses_unasked = true; continue; }
        let (owner_p, taker_p, place) = (works[i].owner_p, works[i].taker_p, works[i].place);
        for k in 0..works[i].resp.len() {
            if works[i].resp[k].source != Source::Sweep || works[i].resp[k].state != RespState::Unread { continue; }
            let first = works[i].resp[k].first;
            if !again_at.contains_key(&(owner_p, first, place)) { let a = rd.afters(owner_p, Some(first), place, LineParams { depth: params.sweep_depth, fuse: params.fuse }); again_at.insert((owner_p, first, place), a); }
            let old = works[i].resp[k].clone();
            let bs = base[&taker_p].as_ref();
            let row_net = works[i].row.net;
            let how = format!("{}; read again with the fuse of a line ({})", old.how, params.fuse);
            let again: Vec<Response> = again_at[&(owner_p, first, place)].iter().flatten().map(|af| response_of(af, bs, rec, how.clone(), Source::Sweep, row_net)).filter(|a| a.names.first() == old.names.first()).collect();
            if let Some(a) = again.into_iter().next() { works[i].resp[k] = a; }
            list.resp_reread += 1;
        }
    }
    list.add_cost("round 4: the cut moves read again", t0, rd.queries - q0, rd.nodes - n0);

    let works_unasked_responses = works.iter().filter(|w| w.row.responses_unasked).count();
    // ---- assemble
    for w in works {
        let Work { mut row, st, resp, .. } = w;
        if matches!(st, St::Dropped | St::Skipped) { continue; }
        if matches!(st, St::Pending) { drop_row(&mut list, "no reading", None); continue; }
        let v = row.verdict;
        if v == VerdictKind::Confirmed && row.two_moves { row.responses = vec![]; row.deferred = true; }
        else if v == VerdictKind::Confirmed {
            let firsts: HashSet<(Place, Place)> = resp.iter().map(|x| x.first).collect();
            let mut seen = HashSet::new();
            let mut ks: Vec<Response> = vec![];
            for x in &resp { if seen.insert(x.first) && x.state == RespState::Kept { ks.push(x.clone()); } }
            ks.sort_by(|a, b| (!a.loses_nothing).cmp(&!b.loses_nothing).then(a.net.cmp(&b.net)).then(a.names.cmp(&b.names)));
            row.resp_read = firsts.len();
            row.resp_unread = resp.iter().filter(|x| x.state == RespState::Unread).map(|x| x.first).collect::<HashSet<_>>().len();
            let mut um: Vec<String> = resp.iter().filter(|x| x.state == RespState::Unread).filter_map(|x| x.names.first().cloned()).collect::<HashSet<_>>().into_iter().collect();
            um.sort();
            row.resp_unread_moves = um;
            for stt in [RespState::Kept, RespState::StillWon, RespState::WorseThanPassing, RespState::CostsMoreThanTheThreat, RespState::Unread] {
                let n = resp.iter().filter(|x| x.state == stt).map(|x| x.first).collect::<HashSet<_>>().len();
                if n > 0 { row.resp_states.insert(stt.word(), n); }
            }
            list.resp_read += firsts.len();
            list.resp_dropped += firsts.len() - ks.len();
            row.responses = ks;
        }
        match v {
            VerdictKind::Exchange => list.exchange.push(row),
            VerdictKind::Refuted | VerdictKind::Quiet => drop_row(&mut list, "the taker ends behind: the claim is refuted", Some((&row, "refuted"))),
            _ => { if v == VerdictKind::Unread { list.unread_rows += 1; } if side_is_mover(row.owner, mover) { list.mover.push(row) } else { list.other.push(row) } }
        }
    }
    if works_unasked_responses > 0 || !list.unasked.is_empty() {
        let skipped = list.unasked.len();
        let cut = works_unasked_responses;
        list.not_asked.push(format!("the clock of the step ran out: {skipped} units not asked, the responses of {cut} rows not read"));
    }
    // ---- the goal rows (the end of the game forced, for either side)
    for (p, f, ans) in goal_resp {
        let attacker = if !p { mover } else { g.other(mover) };
        let owner = g.other(attacker);
        let (proved, depth, first, line) = match f { Forced::Proved { depth, first, line } => (true, depth, first, line), Forced::Suspected { depth, first, line } => (false, depth, first, line), _ => continue };
        let all = ans.unwrap_or_default();
        let mut row = ThreatRow::new(owner, None, "goal", vec![], Some(f64::INFINITY));
        row.goal = true; row.verdict = if proved { VerdictKind::Confirmed } else { VerdictKind::Unread }; row.ended = if proved { ENDED[0] } else { ENDED[3] };
        row.reason = if proved { format!("the end of the game within {depth} moves of the taker: found by the search and confirmed by the exact forcing read") } else { format!("the search finds a forced end within {depth} moves, the exact forcing read could not confirm it within its fuse") };
        row.line = line; row.depth = depth as i32; row.first = Some(first);
        row.resp_read = all.len(); row.resp_unread = all.iter().filter(|a| a.2 == 2).count();
        row.responses = all.iter().filter(|a| a.2 == 0).map(|a| Response { first: a.0, names: vec![a.1.clone()], net: 0, taker_lost: vec![], taker_won: vec![], resp_takes: vec![], resp_loses: vec![], cancelled: 0, loses_nothing: true, ended: ENDED[0], end_reason: String::new(), depth: 0, settled: 0, end: 0, nodes: 0, state: RespState::Kept,
            why: "after it the forcing read finds no forced end".into(), how: "after it the forcing read finds no forced end".into(), source: Source::ForcingRead, after: None }).collect();
        list.resp_read += row.resp_read;
        list.goal.push(row.clone());
        if side_is_mover(owner, mover) { list.mover.push(row) } else { list.other.push(row) }
    }
    sort_rows(&mut list.mover);
    sort_rows(&mut list.other);
    let sides: Vec<Role> = params.every_sides.clone().unwrap_or_else(|| vec![mover]);
    every_unit(st, g, s, &mut list, &sides, rec, params);
    list
}

// ------------------------------------------------------------------------------------------------------------------------------ every unit of the other side

/// THE LINE READ OF EVERY UNIT OF THE OTHER SIDE THAT THE LIST HAS NO ROW FOR (subgoal_units.every_unit; SYSTEM_SPEC 7 "a goal for every target"). For each side in `sides` the targets are the units of the other side (the goal kind has no row: it cannot be taken). One
/// row is added for each, read as the list reads a row (the taking lines; where none is legal the quiet move that makes a taking stand) and put where the list would put it (a taking that works or is cut: the side's list; an exchange: `exchange`; a failure: `dropped`
/// with the code of why). No responses are read here: the goal reads them. Changes `list` in place; returns the count of units read. Every unit is asked: the old behaviour (`Ask::all`); the proposer asks through `every_unit_ask`.
pub fn every_unit<G: Game + Placed>(st: &Static, g: &G, s: &G::State, list: &mut ConcreteList, sides: &[Role], rec: &Record, params: &ThreatParams) -> usize {
    every_unit_ask(st, g, s, list, sides, rec, params, &Ask::all()).read
}

/// THE WORTH OF A KIND FOR THE ORDER OF QUESTIONS (PLAN 79.07): the measured record's worth of the kind of each unit, where the record is keyed by the words the game's ledger uses (the words of `Game::units`, which are not always the words of `kind_names`): the word is read from what taking the unit off the
/// board (`Placed::without`) removes from the game's units, so no word of any game is written here. A unit the game cannot take off, or whose word the record does not hold, is worth 0.
pub fn kind_worths<G: Game + Placed>(g: &G, s: &G::State, rec: &Record, units: &[UnitRef]) -> HashMap<Kind, f64> {
    let mut out: HashMap<Kind, f64> = HashMap::new();
    for u in units {
        if out.contains_key(&u.kind) { continue; }
        let word = g.without(s, u.at).and_then(|after| { let mut b = g.units(s); for x in g.units(&after) { if let Some(i) = b.iter().position(|y| *y == x) { b.swap_remove(i); } } b.into_iter().next().map(|x| x.1) });
        out.insert(u.kind, word.and_then(|w| rec.worth.get(&w).copied()).unwrap_or(0) as f64);
    }
    out
}

/// PLAN 79.07 (SYSTEM_SPEC 4 "a budget of questions", 7 the proposer): THE BUDGET OF QUESTIONS of `every_unit_ask`. A question is one unit asked (its taking lines, and where none is legal the quiet moves that make one stand). `budget` None: every unit is asked. Some(n): the units are asked
/// in the order of their stake (the greater first; the record's worth of the kind where the caller gave no stake), the first `n - used` of them; past the budget a unit is asked only while the LAST question changed a fact (`progress`: a line was read for the unit before it, so that
/// the facts of its goal are known). A unit not asked has no row and is listed in `unasked`: a gap, never a refutation.
#[derive(Debug, Clone, Copy)]
pub struct Ask { pub budget: Option<usize>, pub used: usize, pub progress: bool }
impl Ask { pub fn all() -> Ask { Ask { budget: None, used: 0, progress: true } } }

/// What `every_unit_ask` did: the units read, the units asked (with whether the quiet-move read was needed), the units not asked, the questions, the distinct lines read, their nodes.
#[derive(Debug, Clone, Default)]
pub struct EveryOut { pub read: usize, pub asked: Vec<(UnitRef, bool)>, pub unasked: Vec<UnitRef>, pub questions: usize, pub line_reads: usize, pub nodes: u64,
    /// PLAN 79.14 B: the units whose row was carried (counted against the budget as a question answered, not asked again)
    pub carried: Vec<UnitRef> }

/// PLAN 79.14 B: the rows of the last decision's every-unit reading and what the two plies since touched (`cs_tactics::sheet::delta::Touch`): a unit whose row was read under a read set that nothing touched keeps its row.
#[derive(Clone, Copy)]
pub struct Held<'a> { pub rows: &'a [crate::carry::CarriedRow], pub touch: &'a cs_tactics::sheet::delta::Touch }

/// Whether a question that ended as this row ended leaves the next unit worth asking past the budget (the `progress` of the reading): a taking was found, or a two-move threat was; not where nothing could be taken or the game gives no pass.
fn progress_of(row: &ThreatRow) -> bool { !(matches!(row.why, Some("no_bearer_no_move") | Some("no_taking_legal")) || (row.verdict == VerdictKind::Unread && row.net.is_none() && row.line.is_empty())) }

/// ONE PASS OVER THE QUIET MOVES SERVES EVERY UNIT (PLAN 79.07): for each taker the quiet moves are listed once with the places a taking would stand on after the other side's pass; the whole line of a move is read at most once however many units it serves
/// (the Python reads it again for each unit: the same lines, the same order, so the rows agree).
type QuietList<M> = Vec<(M, (Place, Place), HashSet<Place>)>;

fn quiet_list<G: Game + Placed>(g: &G, p0: &G::State) -> QuietList<G::Move> {
    let mut out = vec![];
    for m in g.moves(p0) {
        if !g.captured(p0, &m).is_empty() { continue; }
        let c = g.apply(p0, &m);
        if g.outcome(&c).is_some() { continue; }
        let Some(cq) = g.pass(&c) else { continue };
        let ends: HashSet<Place> = g.moves(&cq).iter().filter(|x| !g.captured(&cq, x).is_empty()).map(|x| g.move_ends(&cq, x).1).collect();
        if ends.is_empty() { continue; }
        let e = g.move_ends(p0, &m);
        out.push((m, e, ends));
    }
    out
}

pub fn every_unit_ask<G: Game + Placed>(st: &Static, g: &G, s: &G::State, list: &mut ConcreteList, sides: &[Role], rec: &Record, params: &ThreatParams, ask: &Ask) -> EveryOut { every_unit_ask_with(st, g, s, list, sides, rec, params, ask, None) }

/// PLAN 79.14 B: `every_unit_ask` that does not ask again a unit whose row of the last decision stands (`held`). The unit counts as a question used, as if asked, so that the budget is the same; its row is the row the last decision read, marked carried.
#[allow(clippy::too_many_arguments)]
pub fn every_unit_ask_with<G: Game + Placed>(st: &Static, g: &G, s: &G::State, list: &mut ConcreteList, sides: &[Role], rec: &Record, params: &ThreatParams, ask: &Ask, held: Option<Held>) -> EveryOut {
    let t0 = Instant::now();
    let judge = Worth { name: "record".into(), table: rec.worth.clone(), ours_first: false };
    let mut rd = Rd::new(g, s, &judge, params.share_table, params.deadline);
    let mover = rd.mover;
    let kinds = g.kind_names();
    let part = graph_of(st, params.root, g, s);
    let lp = LineParams { depth: params.depth, fuse: params.fuse };
    let have: HashSet<(Role, Place)> = list.mover.iter().chain(&list.other).chain(&list.exchange).chain(&list.dropped).filter(|r| !r.goal).filter_map(|r| r.unit.map(|u| (r.owner, u.at))).collect();
    let mut rows: Vec<(ThreatRow, bool)> = vec![];
    for &side in sides {
        let owner = g.other(side);
        let mut units: Vec<_> = part.units.iter().filter(|u| u.owner == owner && Some(u.kind) != params.goal && !have.contains(&(owner, u.at))).collect();
        units.sort_by_key(|u| u.at);
        for u in units {
            let stake = params.stake.and_then(|f| f(&kinds[u.kind as usize], owner == mover));
            let mut by: Vec<UnitRef> = part.bears.iter().filter(|r| r.place == u.at && part.units[r.unit].owner == side).map(|r| { let b = &part.units[r.unit]; UnitRef { owner: b.owner, kind: b.kind, at: b.at } }).collect();
            by.sort(); by.dedup();
            let mut row = ThreatRow::new(owner, Some(UnitRef { owner, kind: u.kind, at: u.at }), "unit", by, stake);
            row.every = true;
            rows.push((row, owner == mover));
        }
    }
    let n = rows.len();
    // the order of the questions: by stake (the stake of the caller where every unit has one, else the record's worth of the kind), the greater first
    let mut order: Vec<usize> = (0..n).collect();
    if ask.budget.is_some() {
        let by_stake = rows.iter().all(|r| r.0.stake.is_some());
        let us: Vec<UnitRef> = rows.iter().filter_map(|r| r.0.unit).collect();
        let worth = kind_worths(g, s, rec, &us);
        let key = |i: usize| -> f64 { let r = &rows[i].0; if by_stake { r.stake.unwrap_or(0.0) } else { r.unit.map_or(0.0, |u| worth.get(&u.kind).copied().unwrap_or(0.0)) } };
        order.sort_by(|a, b| key(*b).partial_cmp(&key(*a)).unwrap_or(std::cmp::Ordering::Equal).then(rows[*a].0.unit.map(|u| u.at).cmp(&rows[*b].0.unit.map(|u| u.at))));
    }
    let mut state: Vec<u8> = vec![0; n];                                   // 0 pending, 1 done, 2 dropped, 3 not asked
    let mut out = EveryOut::default();
    let mut quiet: HashMap<bool, Option<QuietList<G::Move>>> = HashMap::new();
    let mut lines: HashMap<(bool, usize), LineRead> = HashMap::new();
    let (mut used, mut progress) = (ask.used, ask.progress);
    for &i in &order {
        let allowed = (ask.budget.map_or(true, |b| used < b) || progress) && !rd.expired();
        let (row, taker_p) = &mut rows[i];
        let uref = row.unit.expect("a unit row");
        if !allowed { state[i] = 3; out.unasked.push(uref); continue; }
        // a row of the last decision whose read set nothing touched stands: it is not asked again
        if let Some(h) = held {
            if let Some(c) = h.rows.iter().find(|c| c.row.every && c.row.unit == Some(uref) && !h.touch.hits(&c.read_set.members, &c.read_set.places)) {
                let mut r = c.row.clone();
                r.carried = true;
                state[i] = if r.why.is_some() { 2 } else { 1 };
                progress = progress_of(&r);
                *row = r;
                used += 1; out.carried.push(uref);
                continue;
            }
        }
        used += 1; out.questions += 1;
        let place = uref.at;
        match rd.takings(*taker_p, place, lp) {
            None => { row.verdict = VerdictKind::Unread; row.ended = ENDED[3]; row.reason = "the side to move must answer: the game gives no pass, so what the other side would take cannot be read".into(); state[i] = 1; progress = false; out.asked.push((uref, false)); }
            Some(reads) => match best_taking(&reads, rec) {
                Some(b) => { let all = reads.iter().map(|r| r.nodes).sum(); fill(row, &b, false, all); state[i] = 1; progress = true; out.asked.push((uref, false)); }
                None => {
                    // the quiet moves that make a taking stand: the pass over them is made once per taker, the line of a move once
                    out.asked.push((uref, true));
                    let mut fresh = 0u64;
                    let best = match rd.at(*taker_p).and_then(|p0| g.mover(p0).map(|me| (p0, me))) {
                        None => None,
                        Some((p0, me)) => {
                            let ql = quiet.entry(*taker_p).or_insert_with(|| Some(quiet_list(g, p0))).as_ref().unwrap();
                            let mut rs: Vec<LineRead> = vec![];
                            let mut sr = Search::lean_with_fuse(g, rd.judge, me, 1);
                            if let Some(t) = params.deadline { sr.stop_at(t); }
                            for (k, (m, e, ends)) in ql.iter().enumerate() {
                                if !ends.contains(&place) { continue; }
                                if rd.stop.is_some_and(|t| Instant::now() >= t) { break; }
                                if !lines.contains_key(&(*taker_p, k)) { let r = if params.deadline.is_some() { L::read_line_in(&mut sr, p0, m, me, lp) } else { L::read_line(g, rd.judge, p0, m, me, lp) }; fresh += r.nodes; out.line_reads += 1; lines.insert((*taker_p, k), r); }
                                let mut r = lines[&(*taker_p, k)].clone();
                                r.lead = Some(*e);
                                rs.push(r);
                            }
                            rs.sort_by_key(|r| -r.value);
                            rd.queries += 1;
                            let all: u64 = rs.iter().map(|r| r.nodes).sum();
                            best_taking(&rs, rec).map(|b| (b, all))
                        }
                    };
                    rd.nodes += fresh;
                    progress = best.is_some();
                    match best {
                        Some((b, all)) => { fill(row, &b, true, all); state[i] = 1; if !matches!(b.kind, VerdictKind::Confirmed | VerdictKind::Unread) { row.dropped_verdict = Some(row.verdict); row.why = Some("two_move_not_ahead"); state[i] = 2; } }
                        None => { row.why = Some(if row.by.is_empty() { "no_bearer_no_move" } else { "no_taking_legal" }); row.line = vec![]; row.net = None; row.outcome = None; state[i] = 2; }
                    }
                }
            },
        }
    }
    for (i, (row, _)) in rows.into_iter().enumerate() {
        let v = row.verdict;
        if state[i] == 3 { continue; }
        if state[i] == 2 { list.dropped.push(row); }
        else if matches!(v, VerdictKind::Refuted | VerdictKind::Quiet) { let mut r = row; r.why = Some("refuted"); list.dropped.push(r); }
        else if v == VerdictKind::Exchange { list.exchange.push(row); }
        else if side_is_mover(row.owner, mover) { list.mover.push(row) } else { list.other.push(row) }
    }
    list.unasked.extend(out.unasked.iter().copied());
    if rd.expired() && !out.unasked.is_empty() { list.not_asked.push(format!("the clock of the step ran out in the reading of every unit: {} units not asked", out.unasked.len())); }
    out.read = n - out.unasked.len();
    out.nodes = rd.nodes;
    list.add_cost("every unit of the other side", t0, rd.queries, rd.nodes);
    out
}

// ------------------------------------------------------------------------------------------------------------------------------ the stake

/// THE STAKE OF EACH KIND (threat_measure.stake_of_kinds): the change in the owner's chance of the result, by the kinds reading, if a unit of that kind of that owner is lost. `read(K)` is the CALLER's kinds head on a vector of kinds (ours less theirs) and returns the
/// chance of the side to move less the other side's (`us` less `them`); this function is the arithmetic around it. Returns, per kind column, `[owner 0 (the other side), owner 1 (the side to move)]`. Signed: a loss that raises the owner's chance is negative.
pub fn stake_of_kinds(read: &dyn Fn(&[f64]) -> f64, k: &[f64]) -> Vec<[f64; 2]> {
    let r0 = read(k);
    let mut out = vec![[0.0; 2]; k.len()];
    for owner in [1usize, 0] {
        for col in 0..k.len() {
            let mut kv = k.to_vec();
            kv[col] += if owner == 1 { -1.0 } else { 1.0 };
            let r1 = read(&kv);
            out[col][owner] = if owner == 1 { r0 - r1 } else { r1 - r0 };
        }
    }
    out
}

/// The kinds of a position, ours less theirs, over the head's kind columns (`cols`, the game's words): the units of the side to move less the units of the other side.
pub fn kinds_ours_less_theirs<G: Game + Placed>(st: &Static, g: &G, s: &G::State, cols: &[String]) -> Vec<f64> {
    let part = st.position(g, s);
    let mover = g.mover(s).unwrap_or(0);
    let kinds = g.kind_names();
    cols.iter().map(|c| part.units.iter().filter(|u| &kinds[u.kind as usize] == c).map(|u| if u.owner == mover { 1.0 } else { -1.0 }).sum()).collect()
}

/// The stake function a position's table gives (`stake_closure`): the stake of a kind word for the owner (the side to move or not); None for a kind the table does not have.
pub fn stake_closure(table: Vec<[f64; 2]>, cols: Vec<String>) -> impl Fn(&str, bool) -> Option<f64> {
    move |kind, own| cols.iter().position(|c| c == kind).map(|i| table[i][if own { 1 } else { 0 }])
}
