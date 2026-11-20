//! PLAN 79.05 (SYSTEM_SPEC 8 step 3 "the threat list", 4 "how a line ends"; piece: the concrete threat list, its JUDGMENT of a line): the PORT of `verdict_of`, `cancel_like`, `kind_change_net`, `best_taking` and the record's resolution from
//! catspace/incubator/analyst/discovery/threat_concrete.py (the Python stays as the mirror). Game-free: a kind is the game's word; the judge is the caller's measured record of the kinds (no worth is assumed here). No oracle.
//!
//! INTERFACE. `Record { worth, half }` (the measured record: the worth of each kind and the half width of its confidence interval); `Record::resolution(kinds)` the least gap two ledgers of these units can be told apart by; `Record::change_of(from, to)` what a unit changing kind adds.
//! `verdict_of(read, record) -> Verdict` the verdict of ONE line read for the side that takes (`cs_tactics::lines::LineRead`); `best_taking(reads, record) -> Option<Verdict>` the line of a row from its taking lines. Called by the wake's concrete threat list (not yet ported) and by tests.
use cs_tactics::lines::{LineRead, END_LOST, END_WON};
use std::collections::HashMap;

/// The verdicts of the Python module, in its words.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VerdictKind { Confirmed, Exchange, Refuted, Unread, Unsettled, Quiet }

/// The measured record of the kinds: the worth of each (integers, as the line search's judge has them) and the half width of its confidence interval.
#[derive(Debug, Clone, Default)]
pub struct Record { pub worth: HashMap<String, i64>, pub half: HashMap<String, f64> }

impl Record {
    /// The least gap the record can tell apart in a ledger of these units: the half widths added in quadrature.
    pub fn resolution(&self, kinds: &[String]) -> f64 { kinds.iter().map(|k| self.half.get(k).copied().unwrap_or(0.0).powi(2)).sum::<f64>().sqrt() }
    /// What one of the mover's own units changing kind adds: the worth of the new kind less the old one's (a unit that arrives from nothing, `from` empty, is the worth of the new kind). A kind the record does not hold is worth 0.
    pub fn change_of(&self, from: &str, to: &str) -> i64 { self.worth.get(to).copied().unwrap_or(0) - if from.is_empty() { 0 } else { self.worth.get(from).copied().unwrap_or(0) } }
    fn w(&self, k: &str) -> i64 { self.worth.get(k).copied().unwrap_or(0) }
}

/// The verdict of one line, for the side that takes it.
#[derive(Debug, Clone)]
pub struct Verdict { pub kind: VerdictKind, pub taker_lost: Vec<String>, pub taker_won: Vec<String>, pub left_lost: Vec<String>, pub left_won: Vec<String>, pub cancelled: usize, pub net: i64, pub resolution: f64, pub first: (u16, u16), pub end: u8 }

/// Units of the same kind taken and lost cancel exactly: (the kinds the taker is left having lost, the kinds it is left having taken, the pairs cancelled).
pub fn cancel_like(lost: &[String], won: &[String]) -> (Vec<String>, Vec<String>, usize) {
    let mut left_won: Vec<String> = won.to_vec();
    let (mut left_lost, mut pairs) = (vec![], 0);
    for k in lost {
        if let Some(i) = left_won.iter().position(|x| x == k) { left_won.remove(i); pairs += 1; } else { left_lost.push(k.clone()); }
    }
    (left_lost, left_won, pairs)
}

/// What the kind changes along a line add for the side it was read for: its own units changing kind add, the other side's subtract.
pub fn kind_change_net(read: &LineRead, rec: &Record) -> i64 { read.changes.iter().map(|(mine, a, b)| if *mine { 1 } else { -1 } * rec.change_of(a, b)).sum() }

/// The verdict of one line read for the side that takes (the line was read with that side as `me`).
pub fn verdict_of(read: &LineRead, rec: &Record) -> Verdict {
    let lost: Vec<String> = read.events.iter().filter(|(m, _)| *m).map(|(_, k)| k.clone()).collect();
    let won: Vec<String> = read.events.iter().filter(|(m, _)| !*m).map(|(_, k)| k.clone()).collect();
    let (left_lost, left_won, pairs) = cancel_like(&lost, &won);
    let net = left_won.iter().map(|k| rec.w(k)).sum::<i64>() - left_lost.iter().map(|k| rec.w(k)).sum::<i64>() + kind_change_net(read, rec);
    let both: Vec<String> = left_lost.iter().chain(left_won.iter()).cloned().collect();
    let res = rec.resolution(&both);
    let end = if read.end != 0 { read.end } else { read.tail_end };
    let kind = if read.cut { VerdictKind::Unread }
        else if end == END_WON { VerdictKind::Confirmed }
        else if end == END_LOST { VerdictKind::Refuted }
        else if (net as f64) > res { VerdictKind::Confirmed }
        else if (net as f64) < -res { if read.unsettled { VerdictKind::Unsettled } else { VerdictKind::Refuted } }
        else if lost.is_empty() && won.is_empty() && read.changes.is_empty() && net == 0 { VerdictKind::Quiet }
        else if (!lost.is_empty() && !won.is_empty()) || (!left_lost.is_empty() && !left_won.is_empty()) { VerdictKind::Exchange }
        else if !left_won.is_empty() { VerdictKind::Confirmed } else { VerdictKind::Refuted };
    Verdict { kind, taker_lost: lost, taker_won: won, left_lost, left_won, cancelled: pairs, net, resolution: res, first: read.first, end }
}

/// The line of a row from its taking lines: the best for the taker; a line that was cut stands only when no line that was read is CONFIRMED.
pub fn best_taking(reads: &[LineRead], rec: &Record) -> Option<Verdict> {
    let vs: Vec<Verdict> = reads.iter().map(|r| verdict_of(r, rec)).collect();
    // (the first of the greatest net, as Python's `max` returns it)
    let best = |f: &dyn Fn(&Verdict) -> bool| vs.iter().filter(|v| f(v)).fold(None::<&Verdict>, |b, v| match b { Some(x) if x.net >= v.net => Some(x), _ => Some(v) }).cloned();
    best(&|v| v.kind == VerdictKind::Confirmed).or_else(|| best(&|v| v.kind == VerdictKind::Unread)).or_else(|| best(&|v| v.kind != VerdictKind::Unread))
}
