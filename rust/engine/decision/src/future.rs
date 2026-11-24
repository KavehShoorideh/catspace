//! PLAN 79.51 FUTURE PREDICTION, FIRST STEP (SYSTEM_SPEC 7 "Future prediction: one channel, two sub-layers, and the decision to stop"; piece: tactical reasoning and the opportunity policy; ledger row 50, also 6, 9, 14, 16).
//! Game-free: a unit, a kind, a place, a move is a number or a name the game gave; nothing of any game is named here. No oracle, no external model, no worth assumed (the only worths are the caller's measured `Record`).
//!
//! WHAT THE CHANNEL SAYS. Everything said here is about what will happen, and every statement says which sub-layer made it. The NEAR sub-layer is what the rules and a line read settle (the line read, the taking lines of the
//! threat list's first round, the forced end). The FAR sub-layer is the belief that a goal can still come true: a goal some unmet facts away, its unmet facts the units between our unit and its target on a line, ALIVE when each can
//! be met by a move of ours and a THREAT NOW when all but the first can be met by forcing moves. The far sub-layer orders, keeps and stops; IT NEVER ADDS TO THE VALUE (AGENTS rule 7): nothing here is read by the rule for choosing a
//! move or by the search, and a gain is counted only where the near sub-layer closes a goal (`Near`). In this plan the statements are computed and logged beside a decision and read by nobody.
//!
//! INTERFACE (written before the bodies).
//!   `read(st, g, s, rec, params) -> FutureRead`   TAKES the graph's static part, the game, the position (the side to move is the side that wants the goals), the measured record and `FutureParams` (the goal kind, the clock, the depths of the
//!       line reads). RETURNS one `TargetStmt` for each unit of the other side in the order of its stake (the unit whose loss ends the game first, then by the record's worth; an order, not a cut), each a STOP statement, KEPT with its goals or LEFT
//!       with its cause (near: no line gains; far: no goal alive) and the forcing moves kept; the `LineGoal`s (a goal of one of our units on a line to a target, its unmet facts, for each the moves that MEET it, each said FORCING or not and whether
//!       the moved unit can be taken where it lands, ALIVE, THREAT NOW with the move that starts it, the NEAR READ of that move by the line read that exists); the `JoinStmt`s (a forcing move kept from a left target, offered to the facts of
//!       other goals it meets, by the unit and the place it shares); what the clock did not reach. Every statement carries what it read (`read`) and the sub-layer that made it (`Sub`).
//!   `FutureRead::said() -> Vec<String>`  the statements in words (the game's own words for units and places); `FutureRead::to_json()` the same for the decision log.
//!   WHO CALLS IT: `wake::decide_seen`, after the decision is complete, only where `WakeInputs::future` is given (a measure, like `diag_first`: it spends its own clock and writes into `DecisionLog::future`), and the tests.
//!   WHAT IT USES, NOT REBUILT: the graph's `shut` relation (what the proposer of goals.rs reads for a unit that would bear if the units between left), `wake::new_bearer_moves` (the moves that bring a bearer), the taking lines and the
//!   quiet moves that make a threat (`cs_tactics::lines`, the lines of the threat list's first round) judged by `threat_verdict` (the list's own verdict), the line read of a move (`lines::read_line`), the forced end (`lines::forced_end_by`),
//!   the game's pass. The proposer's own goal on a blocked line names only the first line and one fix for all the units between; this module reads every line and every unit between, which that goal does not.
use crate::threat_verdict::{best_taking, verdict_of, Record, VerdictKind};
use crate::wake::{new_bearer_moves, UnitRef};
use cs_game::placed::{Kind, Place, Placed};
use cs_game::relations::PartRelations;
use cs_game::{Game, Role};
use cs_tactics::ledger::Worth;
use cs_tactics::lines::{self as L, Forced, LineParams};
use cs_tactics::sheet::base_graph::Static;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// The sub-layer that made a statement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sub { Near, Far, Both }
impl Sub { pub fn word(self) -> &'static str { match self { Sub::Near => "near", Sub::Far => "far", Sub::Both => "near and far" } } }

/// The parameters of one reading. `goal`: the kind whose loss ends the game (None: the game has none). `seconds`: the clock of the whole reading; a target or a move the clock did not reach is listed, a gap and never a refutation.
/// `line`: the depth and fuse of the line reads of the near sub-layer; `force`: of the taking reads that tell whether a move is forcing; `end_plies` and `end_fuse`: of the forced end (the threat list's names, smaller fuse).
#[derive(Debug, Clone, Copy)]
pub struct FutureParams { pub goal: Option<Kind>, pub seconds: f64, pub line: LineParams, pub force: LineParams, pub end_plies: u32, pub end_fuse: u64 }
impl Default for FutureParams {
    fn default() -> FutureParams { FutureParams { goal: None, seconds: 2.0, line: LineParams { depth: crate::threats::DEPTH, fuse: crate::threats::FUSE }, force: LineParams { depth: crate::threats::SWEEP_DEPTH, fuse: crate::threats::SWEEP_FUSE },
        end_plies: crate::threats::END_PLIES, end_fuse: 200_000 } }
}

/// A unit and its words.
#[derive(Debug, Clone, PartialEq)]
pub struct Said { pub r: UnitRef, pub text: String }

/// Whether a move is forcing: after it the other side faces something it must answer, read as the threat list reads it. `None` in `forcing`: not read (the clock, or the fuse of a line, cut it).
#[derive(Debug, Clone, PartialEq)]
pub struct Forcing { pub forcing: Option<bool>, pub why: String, pub net: Option<i64>, pub line: Vec<String> }

/// A move of ours that meets an unmet fact: the unit that plays it, where it lands, whether the unit can be taken there (a unit of theirs bears on the place in the position after the move) and whether it is held (a unit of ours bears on
/// the place as well), whether the move is forcing, and where it came from (`from_left`: a forcing move kept from a target that was left, offered by the join).
#[derive(Debug, Clone, PartialEq)]
pub struct Meet { pub mv: String, pub mover: Said, pub lands: Place, pub lands_text: String, pub how: &'static str, pub taken_there: bool, pub held: bool, pub forcing: Forcing, pub from_left: Option<String> }

impl Meet {
    /// The unit that moved can be taken where it lands and nothing of ours holds the place.
    pub fn lost_there(&self) -> bool { self.taken_there && !self.held }
}

/// One unmet fact of a goal: a unit stands on the line, ours or theirs, and must leave it.
#[derive(Debug, Clone, PartialEq)]
pub struct FactStmt { pub between: Said, pub theirs: bool, pub meets: Vec<Meet>, at: Place }

/// What starts a threat now: the move that meets the first fact, and the forcing moves that meet the others, each read in the position after the one before it (the other side passing after a move that is not forcing).
#[derive(Debug, Clone, PartialEq)]
pub struct ThreatNow { pub first_fact: usize, pub first: Meet, pub rest: Vec<(usize, Meet)>, pub worth: f64, pub verified: bool }

/// The near read of a goal that is a threat now: the line read of its first move (the other side's best replies), its printed line and the verdict of the threat list's own judge. A gain is counted only here: `closes` is true only for a
/// confirmed verdict.
#[derive(Debug, Clone, PartialEq)]
pub struct NearRead { pub line: Vec<String>, pub net: i64, pub verdict: &'static str, pub closes: bool, pub nodes: u64, pub cut: bool }

/// A goal on a line, made by the far sub-layer: one of our units that would bear on the target along a line if the units between left.
#[derive(Debug, Clone, PartialEq)]
pub struct LineGoal { pub sub: Sub, pub target: Said, pub stake: f64, pub bearer: Said, pub facts: Vec<FactStmt>, pub alive: bool, pub threat_now: Option<ThreatNow>, pub near: Option<NearRead>, pub read: Vec<String>, pub said: String,
    /// a goal that would be a threat now but for this: when the line opens it opens for the other side as well, a unit of theirs bears on our unit that bears on the target, nothing of ours holds it, and they move first (the mend of the first count: 147 of 210 threats now were not closed by the near read, 56 of them by a reply that takes another unit of ours, 32 by the target moving away)
    pub opened_for_them: Option<String> }

/// The near sub-layer's statement about a target: a line that gains, or none.
#[derive(Debug, Clone, PartialEq)]
pub struct NearStmt { pub gains: bool, pub line: Vec<String>, pub net: Option<i64>, pub read: Vec<String> }

/// The stop decision: KEPT (with the goals that keep it and the sub-layer that did) or LEFT (the cause from each sub-layer and the forcing moves kept for the join).
#[derive(Debug, Clone, PartialEq)]
pub enum Stop { Kept { by: Sub, goals: Vec<usize> }, Left { near: String, far: String, kept: Vec<Meet> } }

/// A target of the other side: its stake, what the near sub-layer found, and the stop statement.
#[derive(Debug, Clone, PartialEq)]
pub struct TargetStmt { pub sub: Sub, pub target: Said, pub stake: f64, pub near: NearStmt, pub stop: Stop, pub read: Vec<String>, pub said: String }

/// A forcing move kept from a left target, offered to a fact of another goal it meets.
#[derive(Debug, Clone, PartialEq)]
pub struct JoinStmt { pub from: Said, pub mv: String, pub goal: usize, pub fact: usize, pub by: String, pub said: String }

/// Everything one reading said: the targets in their order, the goals, the joins, what the clock did not reach, the seconds and the counts of the reads.
#[derive(Debug, Clone, Default)]
pub struct FutureRead { pub targets: Vec<TargetStmt>, pub goals: Vec<LineGoal>, pub joins: Vec<JoinStmt>, pub not_asked: Vec<String>, pub seconds: f64, pub forcing_reads: usize, pub forcing_unread: usize, pub line_reads: usize }

struct Words { roles: Vec<String>, kinds: Vec<String>, places: Vec<String> }
impl Words {
    fn of<G: Game + Placed>(g: &G) -> Words { Words { roles: g.roles(), kinds: g.kind_names(), places: g.place_names() } }
    fn unit(&self, u: &UnitRef) -> Said { Said { r: *u, text: format!("{} {} on {}", self.roles[u.owner], self.kinds[u.kind as usize], self.places[u.at as usize]) } }
    fn place(&self, p: Place) -> &str { &self.places[p as usize] }
}

/// The shared state of one reading: the clock, the judge, the cache of forcing answers by (position key, move name).
struct Cx<'a, G: Game + Placed> {
    st: &'a Static, g: &'a G, rec: &'a Record, p: &'a FutureParams, w: Words, judge: Worth, me: Role, other: Role, stop_at: Instant,
    forcing: HashMap<(u64, String), Forcing>, standing: HashMap<u64, Vec<Place>>, forcing_reads: usize, forcing_unread: usize, line_reads: usize,
}

impl<'a, G: Game + Placed> Cx<'a, G> {
    fn expired(&self) -> bool { Instant::now() >= self.stop_at }

    /// The places of their units on which a taking of ours stands confirmed at `s` (we to move): a threat that stands already, which a move does not make.
    fn standing_of(&mut self, s: &G::State) -> Vec<Place> {
        let key = self.g.key(s);
        if let Some(v) = self.standing.get(&key) { return v.clone(); }
        let mut out = vec![];
        for (o, _k, at) in self.g.cells(s) {
            if o == self.me || !self.g.strikes_place(s, self.me, at) { continue; }
            let reads = L::read_takings(self.g, &self.judge, s, at, self.me, self.p.force);
            if best_taking(&reads, self.rec).is_some_and(|b| b.kind == VerdictKind::Confirmed) { out.push(at); }
        }
        self.standing.insert(key, out.clone());
        out
    }

    /// IS THE MOVE `m` OF `s` FORCING (the side to move is us): the game ends by it, or the other side cannot pass (it is under the sign of the end of the game: it must answer), or after it a taking of ours stands confirmed, by the threat
    /// list's own first round, on a unit of theirs that was not already a confirmed threat. Read once for a position and a move.
    fn forcing_of(&mut self, s: &G::State, mv: &str) -> Forcing {
        let key = (self.g.key(s), mv.to_string());
        if let Some(f) = self.forcing.get(&key) { return f.clone(); }
        let f = self.forcing_read(s, mv);
        self.forcing_reads += 1;
        if f.forcing.is_none() { self.forcing_unread += 1; }
        self.forcing.insert(key, f.clone());
        f
    }

    fn forcing_read(&mut self, s: &G::State, mv: &str) -> Forcing {
        let no = |why: &str| Forcing { forcing: Some(false), why: why.to_string(), net: None, line: vec![] };
        let Some(m) = self.g.find_move(s, mv) else { return Forcing { forcing: None, why: "the move is not legal here".into(), net: None, line: vec![] } };
        let child = self.g.apply(s, &m);
        if self.g.outcome(&child).is_some() { return Forcing { forcing: Some(true), why: "the move ends the game".into(), net: None, line: vec![] }; }
        let Some(passed) = self.g.pass(&child) else { return Forcing { forcing: Some(true), why: "after it the other side is under the sign of the end of the game: it must answer".into(), net: None, line: vec![] } };
        let already = self.standing_of(s);
        let mut cut = false;
        let mut targets: Vec<Place> = self.g.cells(&passed).into_iter().filter(|(o, _, at)| *o != self.me && self.g.strikes_place(&passed, self.me, *at)).map(|c| c.2).collect();
        targets.sort();
        for at in targets {
            if self.expired() { cut = true; break; }
            if already.contains(&at) { continue; }
            let reads = L::read_takings(self.g, &self.judge, &passed, at, self.me, self.p.force);
            match best_taking(&reads, self.rec) {
                Some(b) if b.kind == VerdictKind::Confirmed => {
                    let unit = self.g.kind_at(&passed, at).map(|k| self.w.unit(&UnitRef { owner: self.other, kind: k, at }).text).unwrap_or_default();
                    return Forcing { forcing: Some(true), why: format!("after it a taking of ours on {unit} stands confirmed (the other side must answer it)"), net: Some(b.net), line: b.names.clone() };
                }
                Some(b) if b.kind == VerdictKind::Unread => cut = true,
                _ => {}
            }
        }
        if cut { Forcing { forcing: None, why: "not read: the clock or a fuse cut a taking line".into(), net: None, line: vec![] } } else { no("after it no taking of ours stands confirmed on a unit of theirs") }
    }

    /// The legal moves of `s` (we to move) that meet the fact "the unit on `at` leaves the line from `bearer` to `target`": the unit between moves off the line, or our bearer takes it. Each is said by what it does. The reading is by
    /// the place: the place of the unit between is empty in the position after the move, or our bearer lands on it by taking the unit that stood there. A move that only puts a unit on the line, or moves the bearer elsewhere, is no meeting.
    fn meets_at(&self, s: &G::State, bearer: &UnitRef, between: &UnitRef, theirs: bool) -> Vec<Meet> {
        let mut out = vec![];
        for m in self.g.moves(s) {
            let (from, to) = self.g.move_ends(s, &m);
            let after = self.g.apply(s, &m);
            let (how, ok) = if from == bearer.at {
                if to == between.at && theirs { ("takes the unit between", true) } else { ("", false) }
            } else {
                let still = self.g.cells(&after).iter().any(|c| c.2 == between.at);
                if !still && !theirs && from == between.at { ("leaves the line", true) } else if !still && theirs { ("takes the unit between", true) } else { ("", false) }
            };
            if !ok { continue; }
            let mover_kind = self.g.kind_at(s, from).unwrap_or(0);
            let mover = self.w.unit(&UnitRef { owner: self.me, kind: mover_kind, at: from });
            let name = self.g.move_name(s, &m);
            let (taken_there, held) = (self.g.strikes_place(&after, self.other, to), self.g.strikes_place(&after, self.me, to));
            out.push(Meet { mv: name, mover, lands: to, lands_text: self.w.place(to).to_string(), how, taken_there, held, forcing: Forcing { forcing: None, why: "not read yet".into(), net: None, line: vec![] }, from_left: None });
        }
        out
    }

    /// Reads whether each meeting move is forcing (within the clock), the safe ones and the takings first.
    fn read_forcing(&mut self, s: &G::State, meets: &mut [Meet]) {
        for m in meets.iter_mut() {
            if self.expired() { m.forcing = Forcing { forcing: None, why: "not read: the clock ran out".into(), net: None, line: vec![] }; continue; }
            m.forcing = self.forcing_of(s, &m.mv);
        }
    }

    /// THE LINE OPENS FOR THEM TOO: in `after` (the other side to move) a LEGAL move of theirs takes our bearer on the place it stands and nothing of ours holds that place (the other side moves first, and the bearer, which the line made a
    /// bearer, is taken before it takes). Read by the rules' own legal moves, so a side that must answer a move of ours (it is under the sign of the end of the game) is held to the moves that answer it. The same test as "can be taken where it lands",
    /// for the bearer where it stands.
    fn exposed(&self, after: &G::State, bearer: &UnitRef) -> Option<String> {
        if self.g.strikes_place(after, self.me, bearer.at) { return None; }
        let takes = self.g.moves(after).into_iter().find(|m| self.g.move_ends(after, m).1 == bearer.at && !self.g.captured(after, m).is_empty())?;
        Some(format!("when the line opens the other side may take {} by {} and nothing of ours holds it: they move first", self.w.unit(bearer).text, self.g.move_name(after, &takes)))
    }

    /// Does `u` (the bearer, as it stands) bear on the target in `after`, both still standing?
    fn bears_after(&self, after: &G::State, bearer: &UnitRef, target: &UnitRef) -> bool {
        self.g.kind_at(after, bearer.at) == Some(bearer.kind) && self.g.kind_at(after, target.at) == Some(target.kind) && self.g.bearing_from(after, bearer.owner, bearer.kind, bearer.at).contains(&target.at)
    }
}

fn uref(o: Role, k: Kind, at: Place) -> UnitRef { UnitRef { owner: o, kind: k, at } }

/// THE NEAR STATEMENT OF A TARGET: does a line gain on it from `s`? For the unit whose loss ends the game: the forced end. For any other: the taking lines on it and, where none is legal, the quiet moves that make a threat on it, judged
/// as the threat list judges them. A cut line is no refutation; it is said.
fn near_of<G: Game + Placed>(cx: &mut Cx<G>, s: &G::State, t: &UnitRef) -> NearStmt {
    let mut read = vec![];
    if Some(t.kind) == cx.p.goal {
        let f = L::forced_end_by(cx.g, s, cx.me, cx.p.end_plies, cx.p.end_fuse, Some(cx.stop_at));
        cx.line_reads += 1;
        return match f {
            Forced::Proved { depth, line, .. } => { read.push(format!("the forced end of the game within {depth} moves of ours: proved")); NearStmt { gains: true, line, net: None, read } }
            Forced::Suspected { depth, line, .. } => { read.push(format!("the forced end of the game within {depth} moves of ours: suspected, not confirmed by every reply")); NearStmt { gains: false, line, net: None, read } }
            Forced::None { depth } => { read.push(format!("the forced end of the game: none within {depth} plies read")); NearStmt { gains: false, line: vec![], net: None, read } }
            Forced::Gap { depth } => { read.push(format!("the forced end of the game: not read (a fuse cut it at {depth} plies)")); NearStmt { gains: false, line: vec![], net: None, read } }
        };
    }
    let reads = L::read_takings(cx.g, &cx.judge, s, t.at, cx.me, cx.p.line);
    cx.line_reads += reads.len();
    read.push(format!("{} taking lines of ours on it read", reads.len()));
    let mut best = best_taking(&reads, cx.rec);
    if best.is_none() {
        let two = L::read_two_moves(cx.g, &cx.judge, s, t.at, cx.me, cx.p.line);
        cx.line_reads += two.len();
        read.push(format!("no taking is legal; {} quiet moves that make a threat on it read", two.len()));
        best = best_taking(&two, cx.rec);
    }
    match best {
        Some(b) => {
            read.push(format!("best line {} ({:?}), net {}", b.names.join(" "), b.kind, b.net));
            let line = b.prefix.iter().chain(b.names.iter()).cloned().collect();
            NearStmt { gains: b.kind == VerdictKind::Confirmed, line, net: Some(b.net), read }
        }
        None => { read.push("no line of ours reaches it".into()); NearStmt { gains: false, line: vec![], net: None, read } }
    }
}

/// THE GOALS ON LINES onto a target: every line of ours onto its place that units shut (the graph's own relation), each with its facts and, for each fact, the moves of ours that meet it (not yet read for force).
fn goals_on<G: Game + Placed>(cx: &Cx<G>, s: &G::State, part: &cs_tactics::sheet::base_graph::PositionPart, t: &UnitRef, stake: f64) -> Vec<LineGoal> {
    let mut out = vec![];
    for r in part.shuts().iter().filter(|r| r.place == t.at && part.units[r.unit].owner == cx.me && !r.between.is_empty()) {
        let bu = &part.units[r.unit];
        let bearer = uref(bu.owner, bu.kind, bu.at);
        let mut facts = vec![];
        for &bi in &r.between {
            let b = &part.units[bi];
            let between = uref(b.owner, b.kind, b.at);
            let theirs = b.owner != cx.me;
            facts.push(FactStmt { between: cx.w.unit(&between), theirs, meets: cx.meets_at(s, &bearer, &between, theirs), at: b.at });
        }
        let read = vec![format!("the shut relation of the position: {} would bear on {} if {} left the line", cx.w.unit(&bearer).text, cx.w.unit(t).text, r.between.iter().map(|i| cx.w.unit(&uref(part.units[*i].owner, part.units[*i].kind, part.units[*i].at)).text).collect::<Vec<_>>().join(" and ")),
                        format!("{} legal moves of ours read for what they do to each unit between", cx.g.moves(s).len())];
        out.push(LineGoal { sub: Sub::Far, target: cx.w.unit(t), stake, bearer: cx.w.unit(&bearer), facts, alive: false, threat_now: None, near: None, read, said: String::new(), opened_for_them: None });
    }
    out
}

fn safe(m: &Meet) -> bool { !m.lost_there() }

/// ALIVE: each unmet fact has a move of ours that meets it.
fn alive_of(goal: &LineGoal) -> bool { goal.facts.iter().all(|f| !f.meets.is_empty()) }

/// THREAT NOW: a first move that meets one fact and cannot be taken for nothing (it is not taken where it lands, or it is held), and for every other fact a forcing move in the position that follows (the other side having passed after a move
/// that is not forcing), the last of them leaving the bearer on the target. The reading is verified by playing the moves (`verified`).
fn threat_now_of<G: Game + Placed>(cx: &mut Cx<G>, s: &G::State, goal: &LineGoal) -> (Option<ThreatNow>, Option<String>) {
    let mut opened: Option<String> = None;
    let (bearer, target) = (goal.bearer.r, goal.target.r);
    let n = goal.facts.len();
    for i in 0..n {
        let mut firsts: Vec<&Meet> = goal.facts[i].meets.iter().filter(|m| safe(m)).collect();
        firsts.sort_by_key(|m| (m.taken_there, m.mv.clone()));
        for first in firsts {
            if cx.expired() { return (None, opened); }
            let Some(m1) = cx.g.find_move(s, &first.mv) else { continue };
            let child1 = cx.g.apply(s, &m1);
            if cx.g.outcome(&child1).is_some() { continue; }
            let moved = if cx.g.move_ends(s, &m1).0 == bearer.at { UnitRef { at: cx.g.move_ends(s, &m1).1, ..bearer } } else { bearer };
            if n == 1 {
                if cx.bears_after(&child1, &moved, &target) {
                    if let Some(why) = cx.exposed(&child1, &moved) { opened.get_or_insert(why); continue; }
                    return (Some(ThreatNow { first_fact: i, first: first.clone(), rest: vec![], worth: goal.stake, verified: true }), None);
                }
                continue;
            }
            let Some(mut cur) = cx.g.pass(&child1) else { continue };
            let mut rest: Vec<(usize, Meet)> = vec![];
            let others: Vec<usize> = (0..n).filter(|j| *j != i).collect();
            let mut ok = true;
            for (k, &j) in others.iter().enumerate() {
                let f = &goal.facts[j];
                let between = f.between.r;
                let mut cands = cx.meets_at(&cur, &moved, &between, f.theirs);
                // the moves the join offered (kept from a target that was left) are tried first, then those that cannot be taken, by name
                let joined = |name: &str| f.meets.iter().any(|m| m.mv == name && m.from_left.is_some());
                cands.sort_by_key(|m| (!joined(&m.mv), m.taken_there, m.mv.clone()));
                let mut found: Option<(Meet, G::State)> = None;
                for mut c in cands {
                    if cx.expired() { break; }
                    c.forcing = cx.forcing_of(&cur, &c.mv);
                    if c.forcing.forcing != Some(true) { continue; }
                    let Some(mm) = cx.g.find_move(&cur, &c.mv) else { continue };
                    let nx = cx.g.apply(&cur, &mm);
                    if cx.g.outcome(&nx).is_some() { continue; }
                    let last = k + 1 == others.len();
                    if last { if cx.bears_after(&nx, &moved, &target) { found = Some((c, nx)); break; } else { continue; } }
                    match cx.g.pass(&nx) { Some(p) => { found = Some((c, p)); break; } None => continue }
                }
                match found { Some((c, nx)) => { rest.push((j, c)); cur = nx; } None => { ok = false; break; } }
            }
            if ok {
                if let Some(why) = cx.exposed(&cur, &moved) { opened.get_or_insert(why); continue; }
                return (Some(ThreatNow { first_fact: i, first: first.clone(), rest, worth: goal.stake, verified: true }), None);
            }
        }
    }
    (None, opened)
}

/// THE NEAR READ of a goal that is a threat now: the line read of its first move, the printed line and the verdict of the threat list's own judge. No gain is counted before this.
fn near_read<G: Game + Placed>(cx: &mut Cx<G>, s: &G::State, first: &str) -> Option<NearRead> {
    let m = cx.g.find_move(s, first)?;
    let r = L::read_line(cx.g, &cx.judge, s, &m, cx.me, cx.p.line);
    cx.line_reads += 1;
    let v = verdict_of(&r, cx.rec);
    Some(NearRead { line: r.names.clone(), net: v.net, verdict: match v.kind { VerdictKind::Confirmed => "confirmed", VerdictKind::Exchange => "exchange", VerdictKind::Refuted => "refuted", VerdictKind::Unread => "unread", VerdictKind::Unsettled => "unsettled", VerdictKind::Quiet => "quiet" }, closes: v.kind == VerdictKind::Confirmed, nodes: r.nodes, cut: r.cut })
}

fn meet_said(m: &Meet) -> String {
    let f = match m.forcing.forcing { Some(true) => "FORCING", Some(false) => "not forcing", None => "force not read" };
    let t = if m.lost_there() { "can be taken there and nothing holds it" } else if m.taken_there { "can be taken there but is held" } else { "cannot be taken there" };
    format!("{} ({} {}, lands on {}; {}; {}{})", m.mv, m.mover.text, m.how, m.lands_text, f, t, m.from_left.as_ref().map(|x| format!("; kept from the target {x}")).unwrap_or_default())
}

/// READS THE FUTURE CHANNEL of `s` (see the module doc).
pub fn read<G: Game + Placed>(st: &Static, g: &G, s: &G::State, rec: &Record, p: &FutureParams) -> FutureRead {
    let t0 = Instant::now();
    let mut out = FutureRead::default();
    let Some(me) = g.mover(s) else { return out };
    let other = g.other(me);
    let judge = Worth { name: "record".into(), table: rec.worth.clone(), ours_first: false };
    let mut cx = Cx { st, g, rec, p, w: Words::of(g), judge, me, other, stop_at: t0 + Duration::from_secs_f64(p.seconds), forcing: HashMap::new(), standing: HashMap::new(), forcing_reads: 0, forcing_unread: 0, line_reads: 0 };
    let part = cx.st.position(g, s);
    let mut targets: Vec<UnitRef> = part.units.iter().filter(|u| u.owner == other).map(|u| uref(u.owner, u.kind, u.at)).collect();
    let worths = crate::threats::kind_worths(g, s, rec, &targets);
    let stake = |u: &UnitRef| -> f64 { if Some(u.kind) == p.goal { f64::INFINITY } else { worths.get(&u.kind).copied().unwrap_or(0.0) } };
    targets.sort_by(|a, b| stake(b).partial_cmp(&stake(a)).unwrap_or(std::cmp::Ordering::Equal).then(a.at.cmp(&b.at)));
    // ---- pass one: for each target in the order of its stake, the near statement, its goals with their facts and the moves that meet them, and the forcing moves on it
    let mut kept_by_target: Vec<Vec<Meet>> = vec![];
    for t in &targets {
        if cx.expired() { out.not_asked.push(cx.w.unit(t).text); kept_by_target.push(vec![]); continue; }
        let near = near_of(&mut cx, s, t);
        let first_goal = out.goals.len();
        let mut gs = goals_on(&cx, s, &part, t, stake(t));
        for gl in gs.iter_mut() { for f in gl.facts.iter_mut() { cx.read_forcing(s, &mut f.meets); } gl.alive = alive_of(gl); }
        out.goals.extend(gs);
        let alive: Vec<usize> = (first_goal..out.goals.len()).filter(|i| out.goals[*i].alive).collect();
        // the moves that bring a bearer onto the target, forcing ones: kept where the target is left
        let mut kept: Vec<Meet> = vec![];
        if alive.is_empty() && !near.gains {
            for (name, _) in new_bearer_moves(cx.st, g, s, &[t.at]) {
                if cx.expired() { break; }
                let f = cx.forcing_of(s, &name);
                if f.forcing != Some(true) { continue; }
                let Some(m) = g.find_move(s, &name) else { continue };
                let (from, to) = g.move_ends(s, &m);
                let after = g.apply(s, &m);
                let mover = cx.w.unit(&uref(me, g.kind_at(s, from).unwrap_or(0), from));
                kept.push(Meet { mv: name, mover, lands: to, lands_text: cx.w.place(to).to_string(), how: "brings a bearer onto the target", taken_there: g.strikes_place(&after, other, to), held: g.strikes_place(&after, me, to), forcing: f, from_left: None });
            }
        }
        let tw = cx.w.unit(t);
        let stop = if near.gains && alive.is_empty() { Stop::Kept { by: Sub::Near, goals: vec![] } }
            else if !alive.is_empty() { Stop::Kept { by: if near.gains { Sub::Both } else { Sub::Far }, goals: alive.clone() } }
            else {
                let n_lines = out.goals.len() - first_goal;
                let nearc = format!("near: no line gains on it ({})", near.read.join("; "));
                let farc = if n_lines == 0 { "far: no unit of ours is on a line onto it with units between, so no goal on it is alive".to_string() } else { format!("far: {n_lines} line(s) onto it, each with an unmet fact that no move of ours meets, so no goal on it is alive") };
                Stop::Left { near: nearc, far: farc, kept: kept.clone() }
            };
        let said = match &stop {
            Stop::Kept { by, goals } => format!("KEPT {} (stake {}), by the {} sub-layer: {}", tw.text, fmt_stake(stake(t)), by.word(), if goals.is_empty() { format!("a line gains: {}", near.line.join(" ")) } else { format!("{} goal(s) on a line alive", goals.len()) }),
            Stop::Left { near: n, far: f, kept } => format!("LEFT {} (stake {}): {}; {}; forcing moves kept: {}", tw.text, fmt_stake(stake(t)), n, f, if kept.is_empty() { "none".to_string() } else { kept.iter().map(meet_said).collect::<Vec<_>>().join(" | ") }),
        };
        kept_by_target.push(kept);
        out.targets.push(TargetStmt { sub: Sub::Both, target: tw, stake: stake(t), near, stop, read: vec![format!("the graph's shut relation and {} legal moves of ours", g.moves(s).len())], said });
    }
    // ---- pass two, the join: a forcing move kept from a left target is offered to every fact of the goals of the other targets it meets, by the unit and the place it shares
    for (ti, kept) in kept_by_target.iter().enumerate() {
        for km in kept {
            let Some(mm) = g.find_move(s, &km.mv) else { continue };
            let (from, _to) = g.move_ends(s, &mm);
            for gi in 0..out.goals.len() {
                if out.goals[gi].target.r == targets[ti] { continue; }
                for fi in 0..out.goals[gi].facts.len() {
                    let (between, theirs, fat) = (out.goals[gi].facts[fi].between.r, out.goals[gi].facts[fi].theirs, out.goals[gi].facts[fi].at);
                    let bearer = out.goals[gi].bearer.r;
                    let ms = cx.meets_at(s, &bearer, &between, theirs);
                    if out.joins.iter().any(|j| j.mv == km.mv && j.goal == gi && j.fact == fi) { continue; }          // offered already by a target of a larger stake
                    if let Some(hit) = ms.into_iter().find(|m| m.mv == km.mv) {
                        let by = if from == fat { format!("the unit it moves is the unit between ({})", out.goals[gi].facts[fi].between.text) } else { format!("it empties the place {} of the unit between", cx.w.place(fat)) };
                        let from_t = cx.w.unit(&targets[ti]);
                        let said = format!("JOIN: the forcing move {} kept from the left target {} meets the unmet fact \"{} leaves the line\" of the goal on {} ({})", km.mv, from_t.text, out.goals[gi].facts[fi].between.text, out.goals[gi].target.text, by);
                        out.joins.push(JoinStmt { from: from_t.clone(), mv: km.mv.clone(), goal: gi, fact: fi, by, said });
                        let mut h = hit; h.forcing = km.forcing.clone(); h.from_left = Some(from_t.text.clone());
                        match out.goals[gi].facts[fi].meets.iter_mut().find(|m| m.mv == h.mv) { Some(m) => { m.forcing = h.forcing.clone(); if m.from_left.is_none() { m.from_left = h.from_left.clone(); } } None => out.goals[gi].facts[fi].meets.push(h) }
                    }
                }
            }
        }
    }
    // ---- pass three: ALIVE and THREAT NOW with the joins in, and the near read of each threat now
    for gi in 0..out.goals.len() {
        let mut gl = out.goals[gi].clone();
        gl.alive = alive_of(&gl);
        if gl.alive && !cx.expired() { let (tn, op) = threat_now_of(&mut cx, s, &gl); gl.threat_now = tn; gl.opened_for_them = op; }
        if let Some(tn) = &gl.threat_now { gl.near = near_read(&mut cx, s, &tn.first.mv); }
        gl.said = goal_said(&gl);
        out.goals[gi] = gl;
    }
    out.seconds = t0.elapsed().as_secs_f64();
    out.forcing_reads = cx.forcing_reads;
    out.forcing_unread = cx.forcing_unread;
    out.line_reads = cx.line_reads;
    out
}

fn fmt_stake(x: f64) -> String { if x.is_infinite() { "the end of the game".to_string() } else { format!("{x}") } }

fn goal_said(gl: &LineGoal) -> String {
    let mut s = format!("[far] GOAL: {} takes {} (stake {}) with {} unmet fact(s): ", gl.bearer.text, gl.target.text, fmt_stake(gl.stake), gl.facts.len());
    for f in &gl.facts {
        s += &format!("\"{} leaves the line\" ({}; {} move(s) meet it: {}); ", f.between.text, if f.theirs { "theirs" } else { "ours" }, f.meets.len(), f.meets.iter().map(meet_said).collect::<Vec<_>>().join(" | "));
    }
    s += &format!("ALIVE: {}.", if gl.alive { "yes" } else { "no" });
    if let Some(tn) = &gl.threat_now {
        s += &format!(" THREAT NOW (worth {}): first {}", fmt_stake(tn.worth), meet_said(&tn.first));
        for (j, m) in &tn.rest { s += &format!("; then, for the fact of {}, {}", gl.facts[*j].between.text, meet_said(m)); }
        s += &format!("; verified by playing the moves: {}.", tn.verified);
        if let Some(n) = &gl.near { s += &format!(" NEAR READ of {}: line {}, net {}, {}{}.", tn.first.mv, n.line.join(" "), n.net, n.verdict, if n.closes { " (closes: a gain is counted here)" } else { " (does not close: no gain counted)" }); }
    } else if let Some(why) = &gl.opened_for_them { s += &format!(" Not a threat now: {why}."); }
    else if gl.alive { s += " Not a threat now: no first move with forcing moves for the others was found."; }
    s
}

impl FutureRead {
    /// The statements in words, in the order of the stop statements of the targets, then the goals, then the joins, then what the clock did not reach.
    pub fn said(&self) -> Vec<String> {
        let mut v: Vec<String> = vec![];
        for t in &self.targets { v.push(format!("[{}] {}", t.sub.word(), t.said)); for (i, g) in self.goals.iter().enumerate() { if g.target == t.target { v.push(format!("  goal {i}: {}", g.said)); } } }
        for j in &self.joins { v.push(format!("[far] {}", j.said)); }
        for n in &self.not_asked { v.push(format!("not reached by the clock (a gap, not a refutation): {n}")); }
        v
    }
    /// The same as JSON for the decision log.
    pub fn to_json(&self) -> Value {
        let meet = |m: &Meet| json!({"move": m.mv, "mover": m.mover.text, "how": m.how, "lands": m.lands_text, "taken_there": m.taken_there, "held": m.held, "forcing": m.forcing.forcing, "why": m.forcing.why, "from_left": m.from_left});
        json!({
            "targets": self.targets.iter().map(|t| json!({"sub": t.sub.word(), "target": t.target.text, "stake": if t.stake.is_infinite() { Value::Null } else { json!(t.stake) }, "goal_kind": t.stake.is_infinite(), "said": t.said,
                "near": {"gains": t.near.gains, "line": t.near.line, "net": t.near.net, "read": t.near.read},
                "stop": match &t.stop { Stop::Kept { by, goals } => json!({"kept": true, "by": by.word(), "goals": goals}), Stop::Left { near, far, kept } => json!({"kept": false, "near": near, "far": far, "forcing_kept": kept.iter().map(&meet).collect::<Vec<_>>()}) }})).collect::<Vec<_>>(),
            "goals": self.goals.iter().map(|g| json!({"sub": g.sub.word(), "target": g.target.text, "bearer": g.bearer.text, "alive": g.alive, "said": g.said, "read": g.read, "opened_for_them": g.opened_for_them,
                "facts": g.facts.iter().map(|f| json!({"between": f.between.text, "theirs": f.theirs, "meets": f.meets.iter().map(&meet).collect::<Vec<_>>()})).collect::<Vec<_>>(),
                "threat_now": g.threat_now.as_ref().map(|t| json!({"first": meet(&t.first), "first_fact": t.first_fact, "rest": t.rest.iter().map(|(j, m)| json!({"fact": j, "meet": meet(m)})).collect::<Vec<_>>(), "verified": t.verified})),
                "near_read": g.near.as_ref().map(|n| json!({"line": n.line, "net": n.net, "verdict": n.verdict, "closes": n.closes, "nodes": n.nodes, "cut": n.cut}))})).collect::<Vec<_>>(),
            "joins": self.joins.iter().map(|j| json!({"from": j.from.text, "move": j.mv, "goal": j.goal, "fact": j.fact, "by": j.by})).collect::<Vec<_>>(),
            "not_asked": self.not_asked, "seconds": self.seconds, "forcing_reads": self.forcing_reads, "forcing_unread": self.forcing_unread, "line_reads": self.line_reads,
        })
    }
}
