//! PLAN 79.08 B, STEP 1 OF THE WAKE (SYSTEM_SPEC 8 "bring forward the open tactics carried from the last move, and update only those whose facts changed"; 4; piece: the wake). THE CARRY: a claim whose READ SET did not change is held and not read again; the others are read again.
//! The rule is threat_concrete.same_read_set / threat_list.read_set_of ported: the read set of a row is the members and places of the frame that explains it and the side to move. Measured earlier (SYSTEM_SPEC 12, plan 78.72): a claim whose read set did not change is held again only about half the time, so
//! the carry saves the readings and is itself a claim to be judged; it is off unless the wake's `carried` piece is on. Game-free.
//!
//! INTERFACE (written before the bodies).
//!   `Carried`                           the rows of the last decision's list with the read set of each (`Carried::of(list, read_sets)`); what `WakeDecision` hands the next decision.
//!   `carry(previous, now) -> Carry`     takes the last decision's `Carried` and this position's frame rows with their read sets; returns the rows held (taken from the last list, unread) and the frame rows that must be read.
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
pub struct Carried { pub rows: Vec<CarriedRow> }

impl Carried {
    /// The rows of `list` that have a read set (the frame read made them).
    pub fn of(list: &ConcreteList, read_sets: &BTreeMap<UnitRef, ReadSet>) -> Carried {
        let mut rows = vec![];
        for (b, v) in [(Bucket::Mover, &list.mover), (Bucket::Other, &list.other), (Bucket::Exchange, &list.exchange), (Bucket::Dropped, &list.dropped)] {
            for r in v.iter().filter(|r| !r.goal && !r.every) {
                if let Some(rs) = r.unit.and_then(|u| read_sets.get(&u)) { rows.push(CarriedRow { bucket: b, row: r.clone(), read_set: rs.clone() }); }
            }
        }
        Carried { rows }
    }
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
