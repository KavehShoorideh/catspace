//! PLAN 79.08 C, THE RULE FOR CHOOSING A MOVE (SYSTEM_SPEC 8 "the rule for choosing a move", 7 "Calculation and estimate", 13 items 26 and 27; piece: the wake). The FIXED rule that decides until a learned chooser exists (the chooser comes after the whole wake has logged games). The parts that are pure functions
//! of the goals, the list and the search's equal set live here so that each is tested alone; the loop (`wake::decide`) calls them in the order of section 8. Game-free: a unit is a number, a move a name.
//!
//! THE RULE (his words, section 8): evaluate the threats of the other side; where there is no threat advance toward the subgoal; where there is a threat defend and advance toward the subgoal; a big threat of theirs makes refuting it the subgoal; among the answers those that also serve our subgoal first.
//!   (a) THEIR threats that work now and are above the spread (a CONFIRMED row: its taking ends ahead by more than the record's resolution, or the end of the game) against OUR units: our ANSWERS are named: the responses the line read kept, and the moves the facts of their goal give (`answers_against`).
//!   (b) OUR goals: the fixes of the goal with the FEWEST THINGS IN ITS WAY, the larger stake breaking a tie (PROPOSED order, section 13 item 26), are named; a far goal of high stake is never read as a line, it gives the reading its nearest stopper as the subgoal (the stopper is itself a goal of the same form, built by the
//!       recursion of the proposer, so the goal with the fewest things in its way is often that stopper).
//!   (c) among the moves the search holds equal: one that answers their largest standing threat, then one that is a fix of our chosen goal, then the estimates (the reasons' pick, else the search's own first move). The estimates only choose among what the reading holds equal; nothing here adds to a value (rule 7).
//!   (d) a named move is one whose LINE WAS READ and not refuted, within a budget of line reads (a cut is a gap, never a refutation): `keep_read`.
//!
//! INTERFACE (written before the bodies).
//!   `things_in_the_way(goal) -> usize`            the distinct stoppers of its unmet facts, plus one for each unmet fact with no stopper named (something is missing).
//!   `chosen_goal(goals, side, worth, kinds, goal_kind) -> Option<usize>`   the index of the goal of `side` with the fewest things in its way that has a move among its fixes; ties by the larger stake (the row's stake, else the record's worth of the target's kind, the goal kind the largest), then nearer the root.
//!   `their_threats(list, mover) -> Vec<&ThreatRow>`   the rows against the mover that work now, the largest net first (the end of the game first).
//!   `fix_moves(goal) -> Vec<String>`              the moves of a goal's fix set, then of its other fixes, each once, in that order.
//!   `keep_read(g, s, rec, mine, theirs, budget, p, share) -> KeptNames`   reads the line of each named move (ours at the position, theirs after the game's pass), in order, within `budget` reads a side; keeps those not refuted.
//!   `choose(equal, answers, fixes, picked) -> Chosen`   the rule (c).
use crate::goals::{Goal, Status};
use crate::threat_verdict::{verdict_of, Record, VerdictKind};
use crate::threats::{describe, ThreatRow};
use crate::wake::UnitRef;
use cs_game::placed::{Kind, Placed};
use cs_game::{Game, Role};
use cs_tactics::ledger::{Search, Worth};
use cs_tactics::lines::{self as L, LineParams};
use std::collections::HashSet;

/// The default number of line reads a side may spend on naming moves (a named fuse of the question budget, not tuned): a question is one move's line read.
pub const NAMED_BUDGET: usize = 12;

/// How many things stand in the way of a goal: the distinct stoppers of its unmet facts, and one for each unmet fact that names none (a unit is missing, or the end is not found).
pub fn things_in_the_way(goal: &Goal) -> usize {
    let mut stoppers: HashSet<UnitRef> = HashSet::new();
    let mut bare = 0;
    for f in goal.facts.iter().filter(|f| matches!(f.status, Status::Unmet | Status::Unread)) {
        match &f.cause { Some(c) if !c.stoppers.is_empty() => { for u in &c.stoppers { stoppers.insert(*u); } } _ => bare += 1 }
    }
    stoppers.len() + bare
}

/// The moves of a goal's fix set, then of its other fixes, each once, in that order.
pub fn fix_moves(goal: &Goal) -> Vec<String> {
    let mut out: Vec<String> = vec![];
    let mut put = |m: &String| { if !out.contains(m) { out.push(m.clone()); } };
    for e in &goal.fix_set { for m in &e.moves { put(m); } }
    for f in &goal.fixes { for m in &f.moves { put(m); } }
    out
}

/// The stake used ONLY to break a tie of the order of goals: the row's stake; else the record's worth of the target's kind (a number the record measured); the goal kind, which the record does not hold, is the end of the game and the largest.
fn tie_stake(goal: &Goal, rec: &Record, kinds: &[String], goal_kind: Option<Kind>) -> f64 {
    if let Some(s) = goal.stake { return s; }
    if Some(goal.target.kind) == goal_kind { return f64::INFINITY; }
    rec.worth.get(&kinds[goal.target.kind as usize]).map_or(0.0, |w| *w as f64)
}

/// THE CHOSEN GOAL of a side (the order of section 13 item 26, PROPOSED): among its goals that have a move among their fixes, the fewest things in the way, then the larger stake, then the nearer the root of the recursion, then the place.
pub fn chosen_goal(goals: &[Goal], side: Role, rec: &Record, kinds: &[String], goal_kind: Option<Kind>) -> Option<usize> {
    let mut best: Option<(usize, (usize, f64, u8, u16))> = None;
    for (i, gl) in goals.iter().enumerate() {
        if gl.side != side || gl.unmet.is_empty() || fix_moves(gl).is_empty() { continue; }
        let key = (things_in_the_way(gl), -tie_stake(gl, rec, kinds, goal_kind), gl.depth, gl.target.at);
        let better = match &best { None => true, Some((_, k)) => key.0 < k.0 || (key.0 == k.0 && (key.1 < k.1 || (key.1 == k.1 && (key.2, key.3) < (k.2, k.3)))) };
        if better { best = Some((i, key)); }
    }
    best.map(|b| b.0)
}

/// THEIR THREATS that work now (a confirmed row against the mover's units, the end of the game forced against the mover included), the end of the game first, then the larger net.
pub fn their_threats(list: &crate::threats::ConcreteList) -> Vec<&ThreatRow> {
    let mut v: Vec<&ThreatRow> = list.mover.iter().filter(|r| r.verdict == VerdictKind::Confirmed && !r.two_moves && !r.deferred).collect();
    v.sort_by(|a, b| (!a.goal, -a.net.unwrap_or(0)).cmp(&(!b.goal, -b.net.unwrap_or(0))));
    v
}

/// One named move whose line was read: the verdict of its line for the side that plays it, the net, the line and how it ended.
#[derive(Debug, Clone)]
pub struct Kept { pub name: String, pub verdict: VerdictKind, pub net: i64, pub line: Vec<String>, pub ended: String }

/// The named moves after their lines were read: ours and theirs that stand, the ones refuted (name, whether theirs), the reads made, the nodes and seconds.
#[derive(Debug, Clone, Default)]
pub struct KeptNames { pub mine: Vec<Kept>, pub theirs: Vec<Kept>, pub refuted: Vec<(String, bool, String)>, pub unlisted: Vec<(String, bool)>, pub read: usize, pub nodes: u64, pub seconds: f64 }

/// A NAMED MOVE IS ONE WHOSE LINE WAS READ AND NOT REFUTED, within a budget of reads a side (the first `budget` of the order given; the rest are `unlisted`: not read, a gap). Ours are read at the position by the mover; theirs after the game's pass by the other side (none where the game gives no pass). A line the fuse
/// cut is not refuted (a gap, never a kill). With `share` the reads of one side go through one search and one table.
pub fn keep_read<G: Game + Placed>(g: &G, s: &G::State, rec: &Record, mine: &[String], theirs: &[String], budget: usize, p: LineParams, share: bool) -> KeptNames {
    let t0 = std::time::Instant::now();
    let judge = Worth { name: "record".into(), table: rec.worth.clone(), ours_first: false };
    let mut out = KeptNames::default();
    let Some(mover) = g.mover(s) else { return out };
    let passed = g.pass(s);
    for (names, who) in [(mine, false), (theirs, true)] {
        let (pos, me) = if who { match &passed { Some(q) => (q, g.other(mover)), None => { for n in names { out.unlisted.push((n.clone(), true)); } continue; } } } else { (s, mover) };
        let mut shared = Search::with_fuse(g, &judge, me, 1);
        let mut seen: HashSet<&String> = HashSet::new();
        let mut n = 0;
        for name in names {
            if !seen.insert(name) { continue; }
            let Some(m) = g.find_move(pos, name) else { continue };
            if n >= budget { out.unlisted.push((name.clone(), who)); continue; }
            n += 1;
            let r = if share { L::read_line_in(&mut shared, pos, &m, me, p) } else { L::read_line(g, &judge, pos, &m, me, p) };
            out.read += 1; out.nodes += r.nodes;
            let v = verdict_of(&r, rec);
            let (_, ended) = describe(&v);
            if v.kind == VerdictKind::Refuted { out.refuted.push((name.clone(), who, ended)); continue; }
            let k = Kept { name: name.clone(), verdict: v.kind, net: v.net, line: r.names.clone(), ended };
            if who { out.theirs.push(k) } else { out.mine.push(k) }
        }
    }
    out.seconds = t0.elapsed().as_secs_f64();
    out
}

/// What rule (c) chose and why, in words and as the three ladders it went down.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Chosen { pub mv: String, pub answers_their_threat: bool, pub fixes_our_goal: bool, pub why: String, pub left_after_threat: usize, pub left_after_goal: usize }

/// RULE (c): among the moves the search holds equal (`equal`, the search's own first move first): those that answer their largest standing threat (`answers`) if any; then of those the ones that are a fix of our chosen goal (`fixes`) if any; then `picked` (the estimates' pick among what is left,
/// None when no estimate decides) or the search's own first move. `equal` empty gives None.
pub fn choose(equal: &[String], answers: &[String], fixes: &[String], picked: &dyn Fn(&[String]) -> Option<String>) -> Option<Chosen> {
    if equal.is_empty() { return None; }
    let a: Vec<String> = equal.iter().filter(|m| answers.contains(m)).cloned().collect();
    let (set1, by_threat) = if a.is_empty() { (equal.to_vec(), false) } else { (a, true) };
    let f: Vec<String> = set1.iter().filter(|m| fixes.contains(m)).cloned().collect();
    let (set2, by_goal) = if f.is_empty() { (set1.clone(), false) } else { (f, true) };
    let (mv, by_estimate) = if set2.len() == 1 { (set2[0].clone(), false) } else { match picked(&set2) { Some(m) => (m, true), None => (set2[0].clone(), false) } };
    let why = format!("{}{}{}", if by_threat { "answers their largest standing threat; " } else { "no equal move answers a standing threat of theirs; " },
        if by_goal { "a fix of our chosen goal; " } else { "no equal move of these is a fix of our chosen goal; " },
        if set2.len() == 1 { "the only one left" } else if by_estimate { "the estimates pick among the rest" } else { "the search's own first move among the rest" });
    Some(Chosen { mv, answers_their_threat: by_threat, fixes_our_goal: by_goal, why, left_after_threat: set1.len(), left_after_goal: set2.len() })
}
