//! PLAN 79.08 B, STEP 1 OF THE WAKE (SYSTEM_SPEC 8 "bring forward the open tactics carried from the last move, and update only those whose facts changed"; 4; piece: the wake). THE CARRY: a claim whose READ SET did not change is held and not read again; the others are read again.
//! The rule is threat_concrete.same_read_set / threat_list.read_set_of ported: the read set of a row is the members and places of the frame that explains it and the side to move. Measured earlier (SYSTEM_SPEC 12, plan 78.72): a claim whose read set did not change is held again only about half the time, so
//! the carry saves the readings and is itself a claim to be judged; it is off unless the wake's `carried` piece is on. Game-free.
//!
//! INTERFACE (written before the bodies).
//!   `Carried`                           the rows of the last decision's list with the read set of each (`Carried::of(list, read_sets)`); what `WakeDecision` hands the next decision.
//!   `carry(previous, now) -> Carry`     takes the last decision's `Carried` and this position's frame rows with their read sets; returns the rows held (taken from the last list, unread) and the frame rows that must be read.
//!   `carry_graph(previous, now) -> Carry`  (PLAN 79.14 B) the same with the graph's word on what was touched: a held row's read set is also untouched.
//!   `Carry::merge_into(list)`           puts the held rows into the list the reading of the others made (each into the bucket it was in), in the list's own order.
//!   Called by: the wake's loop (`wake::decide`) and the unit tests.
use crate::frame_rows::{FrameRows, ReadSet};
use crate::threats::{sort_rows, ConcreteList, FrameRow, ThreatRow};
use crate::wake::UnitRef;
use std::collections::BTreeMap;

/// Which list of the concrete threat list a row was in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Bucket { Mover, Other, Exchange, Dropped }

/// One row of the last list with the read set it was read under.
#[derive(Debug, Clone)]
pub struct CarriedRow { pub bucket: Bucket, pub row: ThreatRow, pub read_set: ReadSet }

/// The open tactics of the last decision: its rows with their read sets (a row of the end of the game, or one with no read set, is never carried).
#[derive(Debug, Clone, Default)]
pub struct Carried { pub rows: Vec<CarriedRow>,
    /// PLAN 79.14 B: the side to move and the atoms of the position that decision read (a position's digest), so that what the two plies since touched can be read from the graph
    pub digest: Option<(usize, cs_tactics::sheet::delta::Digest)>,
    /// PLAN 79.14 B: what the frame read of that decision left for the next (the seeds found, the frames read to the end), with the carry on
    pub memo: Option<cs_tactics::sheet::threat_frames::FrameMemo>,
    /// PLAN 79.17 D: the measure held as a goal beyond the search's horizon (SYSTEM_SPEC 7): carried from move to move with the open tactics
    pub held: Option<HeldGoal> }

/// PLAN 79.17 PART D (SYSTEM_SPEC 7, the measure of progress as a goal held across moves; ledger rows 13, 24, 35): THE HELD GOAL. The term (its text), the lowest value held (the value it had when it was last lowered), the plies since, the number of subgoals reached so far (each a lowering of the value below the lowest), the class of the position
/// it was taken up in (the units of each side by kind, as numbers, sorted: a change of class ends the goal) and, once dropped, the cause. It is a goal and never a value: it orders what the search reads and chooses among moves the calculation holds equal.
/// PLAN 79.18 C (the goal's states): `dropped` is for good and only for a class that changed; `asleep` is the cause the goal sleeps for (the search holds the end of the game: it is played) and ends at the next decision in which the search no longer holds it; `slept` and `woken` count the times it went to sleep and was
/// woken; `without_line` counts the decisions in a row in which the question was read to its end and found no line within d (the goal stays and is asked again); `recovered` counts the decisions in which the move named was a recovery (below the value now, not below the lowest held).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HeldGoal { pub term: String, pub value: Vec<i64>, pub since: u32, pub reached: u32, pub class: Vec<(bool, u8)>,
    /// the cause the goal was dropped for (None: held); a dropped goal is not taken up again in the game
    pub dropped: Option<String>,
    pub asleep: Option<String>, pub slept: u32, pub woken: u32, pub without_line: u32, pub recovered: u32 }

impl Carried {
    /// The rows of `list` that have a read set (the frame read made them).
    pub fn of(list: &ConcreteList, read_sets: &BTreeMap<UnitRef, ReadSet>) -> Carried {
        let mut rows = vec![];
        for (b, v) in [(Bucket::Mover, &list.mover), (Bucket::Other, &list.other), (Bucket::Exchange, &list.exchange), (Bucket::Dropped, &list.dropped)] {
            for r in v.iter().filter(|r| !r.goal && !r.every) {
                if let Some(rs) = r.unit.and_then(|u| read_sets.get(&u)) { rows.push(CarriedRow { bucket: b, row: r.clone(), read_set: rs.clone() }); }
            }
        }
        Carried { rows, digest: None, memo: None, held: None }
    }
    /// PLAN 79.14 B: the rows of the last list INCLUDING the rows of the every-unit reading, each with its read set (`every_sets`: the unit's neighbourhood in the position's atoms), and the digest of the position.
    pub fn of_every(list: &ConcreteList, read_sets: &BTreeMap<UnitRef, ReadSet>, every_sets: &BTreeMap<UnitRef, ReadSet>, digest: Option<(usize, cs_tactics::sheet::delta::Digest)>) -> Carried {
        let mut c = Carried::of(list, read_sets);
        for (b, v) in [(Bucket::Mover, &list.mover), (Bucket::Other, &list.other), (Bucket::Exchange, &list.exchange), (Bucket::Dropped, &list.dropped)] {
            for r in v.iter().filter(|r| !r.goal && r.every) {
                if let Some(rs) = r.unit.and_then(|u| every_sets.get(&u)) { c.rows.push(CarriedRow { bucket: b, row: r.clone(), read_set: rs.clone() }); }
            }
        }
        c.digest = digest;
        c
    }
    /// The same, with the memo of the frame read (PLAN 79.14 B).
    pub fn with_memo(mut self, memo: Option<cs_tactics::sheet::threat_frames::FrameMemo>) -> Carried { self.memo = memo; self }
}

/// What the carry decided: the rows held, the frame rows to read again, and the count of each.
#[derive(Debug, Clone, Default)]
pub struct Carry { pub held: Vec<CarriedRow>, pub read: Vec<FrameRow> }

/// THE CARRY: a frame row of this position whose unit has a row in the last list read under the same read set is held; every other frame row is read again.
pub fn carry(previous: &Carried, now: &FrameRows) -> Carry {
    let mut out = Carry::default();
    for fr in &now.rows {
        let held = now.read_sets.get(&fr.unit).and_then(|rs| previous.rows.iter().find(|p| p.row.unit == Some(fr.unit) && &p.read_set == rs));
        match held { Some(h) => out.held.push(h.clone()), None => out.read.push(fr.clone()) }
    }
    out
}

/// PLAN 79.14 B: THE CARRY BY WHAT THE GRAPH SAYS WAS TOUCHED. A frame row of this position is held if its unit has a row in the last list read under the same read set AND nothing in that read set was touched by the two plies since (`now.touch`: no member moved, was taken or appeared, no
/// relation onto a place of the frame was opened or shut, no bearer came or went). With no touch given (no last decision of this side, or the other side to move) nothing is held. The rows held are the rows the last decision read; the test of whole_79_14 checks them against the rows asked afresh.
pub fn carry_graph(previous: &Carried, now: &FrameRows, touch: Option<&cs_tactics::sheet::delta::Touch>) -> Carry {
    let mut out = Carry::default();
    for fr in &now.rows {
        let held = now.read_sets.get(&fr.unit).filter(|rs| touch.is_some_and(|t| !t.hits(&rs.members, &rs.places)))
            .and_then(|rs| previous.rows.iter().find(|p| !p.row.every && p.row.unit == Some(fr.unit) && &p.read_set == rs));
        match held { Some(h) => out.held.push(h.clone()), None => out.read.push(fr.clone()) }
    }
    out
}

/// Put held rows into a list, each into the list it was in; the lists keep their order (the goal's rows are never held).
pub fn merge_held(list: &mut ConcreteList, held: &[CarriedRow]) {
    if held.is_empty() { return; }
    for h in held {
        match h.bucket { Bucket::Mover => list.mover.push(h.row.clone()), Bucket::Other => list.other.push(h.row.clone()), Bucket::Exchange => list.exchange.push(h.row.clone()), Bucket::Dropped => list.dropped.push(h.row.clone()) }
    }
    for v in [&mut list.mover, &mut list.other, &mut list.exchange] { sort_rows(v); }
}

impl Carry {
    /// Put the held rows into the list the reading of the others made.
    pub fn merge_into(&self, list: &mut ConcreteList) { merge_held(list, &self.held); }
}
