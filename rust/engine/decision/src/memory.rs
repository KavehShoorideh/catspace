//! PLAN 79.19 PART C: WHAT THE WAKE READS COMES FROM THE TWO MEMORIES (SYSTEM_SPEC 11 "two memories", "the go on layer 1"; ledger rows 13, 34; piece: the memory, read by the wake). cs_memory owns the statement, the long-term memory (one store for every game, partitioned by the game's key) and the memory of the game at hand;
//! this module is where the TYPES OF THIS CRATE are made things those memories keep: each is `Kept` (its kind, its standing, what it takes and returns, its text form) and is reached as its own type through the reader, never copied into the memory (cs_memory may not depend on this crate).
//! INTERFACE (written before the bodies; who calls: `wake::decide` reads, the binding and the tests make).
//!   `impl Kept for` Record ("worth_record"), ChanceTable ("chance_table"), Reasons ("reasons"), GoalTable ("goal_table"), Carried ("carried": the open tactics and the goal held; a slot of the game's memory, its text form a summary, not decoded back).
//!   `decoders()`                      the decoders of the kinds that are written to a partition (all but "carried").
//!   `Parts { record, chances, reasons, goal_table, measure }`   what a caller that is given the tables of a game hands over; `WakeMemory::of(key, parts)` makes the reader of one decision from them, in place, with no hashing.
//!   `WakeMemory { long, game, hold }`   what `WakeInputs` holds in place of the fields it lost: the reader of the long-term memory (None: nothing known of this game), the memory of the game at hand, and the id of the statement whose term is held as the goal (None: no goal given).
//!   `MeasureStatement`                the mined measure as a statement: `Says::Lower { term, within }`, standing a conjecture with its sample, holder ours; `WakeMemory::measure()` reads it back as `MeasureGoal`.
//!   `WakeMemory::previous()`          the open tactics of the last decision, read from the slot "carried" of the game's memory.
//! The kinds and names: entries "worth_record", "chance_table", "reasons", "goal_table" of the long-term partition; the slot "carried" and the statement "measure" are named by the constants below.
use crate::carry::Carried;
use crate::estimate::ChanceTable;
use crate::goal_table::{Field, GoalTable, Level};
use crate::pick::{Dir, Reason, Reasons, Side};
use crate::threat_verdict::Record;
use crate::wake::MeasureGoal;
use cs_memory::statement::{Holder, Says, Standing, Statement};
use cs_memory::store::{Decoders, GameKey, GameMemory, Kept, Partition, Reader};
use serde_json::{json, Value};
use std::any::Any;
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

pub const RECORD: &str = "worth_record";
pub const CHANCES: &str = "chance_table";
pub const REASONS: &str = "reasons";
pub const GOAL_TABLE: &str = "goal_table";
pub const CARRIED: &str = "carried";
pub const MEASURE: &str = "measure";

fn num(v: &Value, k: &str) -> Result<f64, String> { v.get(k).and_then(|x| x.as_f64()).ok_or_else(|| format!("{k} is a number")) }
fn int(v: &Value, k: &str) -> Result<u64, String> { v.get(k).and_then(|x| x.as_u64()).ok_or_else(|| format!("{k} is a whole number")) }
fn list<'a>(v: &'a Value, k: &str) -> Result<&'a Vec<Value>, String> { v.get(k).and_then(|x| x.as_array()).ok_or_else(|| format!("{k} is a list")) }
fn sorted_cells(m: &HashMap<String, (u32, u32)>) -> Value { let b: BTreeMap<&String, &(u32, u32)> = m.iter().collect(); json!(b.iter().map(|(k, v)| json!([k, v.0, v.1])).collect::<Vec<_>>()) }
fn cells_from(v: &Value) -> Result<HashMap<String, (u32, u32)>, String> {
    v.as_array().ok_or("cells are a list")?.iter().map(|c| Ok((c[0].as_str().ok_or("a cell has a key")?.to_string(), (c[1].as_u64().ok_or("n")? as u32, c[2].as_u64().ok_or("hits")? as u32)))).collect()
}

// ---------------------------------------------------------------------------------------------------------------------------------------- the worth record

impl Kept for Record {
    fn kind(&self) -> &'static str { RECORD }
    fn standing(&self) -> Standing { Standing::Conjecture("the measured record of worths, as the caller gave it".into()) }
    fn describe(&self) -> (&'static str, &'static str) { ("a kind of unit", "its worth and the half width of its interval") }
    fn to_json(&self) -> Value {
        let w: BTreeMap<&String, &i64> = self.worth.iter().collect();
        let h: BTreeMap<&String, &f64> = self.half.iter().collect();
        json!({"worth": w, "half": h})
    }
    fn as_any(&self) -> &dyn Any { self }
}
fn decode_record(v: &Value) -> Result<Arc<dyn Kept>, String> {
    let worth = v.get("worth").and_then(|x| x.as_object()).ok_or("worth")?.iter().map(|(k, x)| x.as_i64().map(|n| (k.clone(), n)).ok_or("a worth is a whole number")).collect::<Result<HashMap<_, _>, _>>()?;
    let half = v.get("half").and_then(|x| x.as_object()).ok_or("half")?.iter().map(|(k, x)| x.as_f64().map(|n| (k.clone(), n)).ok_or("a half width is a number")).collect::<Result<HashMap<_, _>, _>>()?;
    Ok(Arc::new(Record { worth, half }))
}

// ---------------------------------------------------------------------------------------------------------------------------------------- the table of the chance of a goal by what stops it (79.08)

impl Kept for ChanceTable {
    fn kind(&self) -> &'static str { CHANCES }
    fn standing(&self) -> Standing { Standing::Belief { probability: (self.total.1 as f64 + 1.0) / (self.total.0 as f64 + 2.0), count: self.total.0 as u64 } }
    fn describe(&self) -> (&'static str, &'static str) { ("a goal (its key and the names of its unmet facts)", "the chance it is reached, the count behind it, the cell that spoke") }
    fn to_json(&self) -> Value { json!({"by_key": sorted_cells(&self.by_key), "by_facts": sorted_cells(&self.by_facts), "total": [self.total.0, self.total.1], "floor": self.floor}) }
    fn as_any(&self) -> &dyn Any { self }
}
fn decode_chances(v: &Value) -> Result<Arc<dyn Kept>, String> {
    let t = v.get("total").and_then(|x| x.as_array()).ok_or("total")?;
    Ok(Arc::new(ChanceTable { by_key: cells_from(v.get("by_key").ok_or("by_key")?)?, by_facts: cells_from(v.get("by_facts").ok_or("by_facts")?)?, total: (t[0].as_u64().ok_or("total")? as u32, t[1].as_u64().ok_or("total")? as u32), floor: int(v, "floor")? as u32 }))
}

// ---------------------------------------------------------------------------------------------------------------------------------------- the reasons and their weights (78.81)

impl Kept for Reasons {
    fn kind(&self) -> &'static str { REASONS }
    fn standing(&self) -> Standing { Standing::Conjecture("weights fitted on the choice of the move played among equal moves".into()) }
    fn describe(&self) -> (&'static str, &'static str) { ("a move", "the sum of the weights of the reasons it carries") }
    fn to_json(&self) -> Value {
        json!({"keys": self.keys, "reasons": self.reasons.iter().map(|r| json!({"column": r.column, "side": if r.side == Side::Mover { "mover" } else { "other" }, "dir": if r.dir == Dir::Up { "up" } else { "down" }, "weight": r.weight})).collect::<Vec<_>>(),
               "max_size": self.max_size, "levels": self.levels, "tie_digits": self.tie_digits})
    }
    fn as_any(&self) -> &dyn Any { self }
}
fn decode_reasons(v: &Value) -> Result<Arc<dyn Kept>, String> {
    let reasons = list(v, "reasons")?.iter().map(|r| Ok(Reason { column: int(r, "column")? as usize, side: if r.get("side").and_then(|x| x.as_str()) == Some("mover") { Side::Mover } else { Side::Other }, dir: if r.get("dir").and_then(|x| x.as_str()) == Some("up") { Dir::Up } else { Dir::Down }, weight: num(r, "weight")? })).collect::<Result<Vec<_>, String>>()?;
    Ok(Arc::new(Reasons { keys: list(v, "keys")?.iter().map(|x| x.as_u64().ok_or("a key is a number")).collect::<Result<_, _>>()?, reasons, max_size: int(v, "max_size")? as usize,
                          levels: list(v, "levels")?.iter().map(|x| x.as_u64().map(|n| n as u8).ok_or("a level is a number")).collect::<Result<_, _>>()?, tie_digits: v.get("tie_digits").and_then(|x| x.as_i64()).ok_or("tie_digits")? as i32 }))
}

// ---------------------------------------------------------------------------------------------------------------------------------------- the table of the chance of a goal counted from play (79.13)

const FIELDS: [(&str, Field); 7] = [("source", Field::Source), ("unmet", Field::Unmet), ("in_way", Field::InWay), ("depth", Field::Depth), ("rank", Field::Rank), ("fix", Field::Fix), ("mine", Field::Mine)];
fn field_name(f: Field) -> &'static str { FIELDS.iter().find(|(_, x)| *x == f).map(|(n, _)| *n).unwrap_or("?") }

impl Kept for GoalTable {
    fn kind(&self) -> &'static str { GOAL_TABLE }
    fn standing(&self) -> Standing { Standing::Belief { probability: (self.total.1 as f64 + 1.0) / (self.total.0 as f64 + 2.0), count: self.total.0 as u64 } }
    fn describe(&self) -> (&'static str, &'static str) { ("the facts of a goal (its key on each level's fields)", "the chance its target is taken within the table's plies, the count behind it, the level that spoke") }
    fn to_json(&self) -> Value {
        json!({"k": self.k, "floor": self.floor, "total": [self.total.0, self.total.1],
               "levels": self.levels.iter().map(|l| json!({"fields": l.fields.iter().map(|f| field_name(*f)).collect::<Vec<_>>(), "cells": sorted_cells(&l.cells)})).collect::<Vec<_>>()})
    }
    fn as_any(&self) -> &dyn Any { self }
}
fn decode_goal_table(v: &Value) -> Result<Arc<dyn Kept>, String> {
    let t = v.get("total").and_then(|x| x.as_array()).ok_or("total")?;
    let levels = list(v, "levels")?.iter().map(|l| Ok(Level { fields: list(l, "fields")?.iter().map(|f| f.as_str().and_then(Field::named).ok_or("a field of the table")).collect::<Result<_, _>>()?, cells: cells_from(l.get("cells").ok_or("cells")?)? })).collect::<Result<Vec<_>, String>>()?;
    Ok(Arc::new(GoalTable { k: int(v, "k")? as u32, floor: int(v, "floor")? as u32, levels, total: (t[0].as_u64().ok_or("total")? as u32, t[1].as_u64().ok_or("total")? as u32) }))
}

// ---------------------------------------------------------------------------------------------------------------------------------------- the open tactics and the goal held: a slot of the game's memory

impl Kept for Carried {
    fn kind(&self) -> &'static str { CARRIED }
    fn standing(&self) -> Standing { Standing::Fact }
    fn describe(&self) -> (&'static str, &'static str) { ("the position now and what the last decision read", "the rows held unread, the frames carried, the goal held") }
    fn to_json(&self) -> Value {
        json!({"rows": self.rows.len(), "digest": self.digest.is_some(), "memo": self.memo.is_some(),
               "held": self.held.as_ref().map(|h| json!({"term": h.term, "value": h.value, "since": h.since, "reached": h.reached, "dropped": h.dropped, "asleep": h.asleep, "slept": h.slept, "woken": h.woken, "without_line": h.without_line, "recovered": h.recovered}))})
    }
    fn as_any(&self) -> &dyn Any { self }
}

/// The decoders of the kinds a partition holds.
pub fn decoders() -> Decoders { Decoders::new().with(RECORD, decode_record).with(CHANCES, decode_chances).with(REASONS, decode_reasons).with(GOAL_TABLE, decode_goal_table) }

// ---------------------------------------------------------------------------------------------------------------------------------------- the mined measure as a statement

/// The mined measure as a statement of the long-term memory: a term to be lowered within `d` of our moves, a conjecture on the sample it was mined from, ours.
pub fn measure_statement(term: &str, d: u32, sample: &str) -> Statement {
    Statement { id: MEASURE.to_string(), pattern: vec![], says: Says::Lower { term: term.to_string(), within: d }, standing: Standing::Conjecture(sample.to_string()), holder: Holder::Ours, also: vec![] }
}

/// What a caller that is given the tables of a game hands over (each None: not given).
#[derive(Clone, Default)]
pub struct Parts { pub record: Option<Record>, pub chances: Option<ChanceTable>, pub reasons: Option<Reasons>, pub goal_table: Option<GoalTable>,
    /// (term text, d, the sample it was mined on)
    pub measure: Option<(String, u32, String)>,
    /// more mined measures kept in the partition without being held: (id, term text, d, sample); the wake holds one by naming its id (`WakeMemory::hold`)
    pub more_measures: Vec<(String, String, u32, String)> }

/// What `WakeInputs` holds of the two memories: the reader of the long-term memory, the memory of the game at hand, and the id of the statement whose term is held as the goal.
#[derive(Debug, Clone, Default)]
pub struct WakeMemory { pub long: Option<Reader>, pub game: GameMemory, pub hold: Option<String> }

impl WakeMemory {
    /// The memory of one decision from the tables a caller gives, for the game `key`: the long-term partition is made in place (no snapshot, no hash), the game's memory is the one given (or a fresh one).
    pub fn of(key: GameKey, parts: Parts, game: Option<GameMemory>) -> WakeMemory {
        let mut p = Partition::new(key.clone());
        let mut hold = None;
        if let Some(r) = parts.record { p = p.with_entry(RECORD, Arc::new(r)); }
        if let Some(c) = parts.chances { p = p.with_entry(CHANCES, Arc::new(c)); }
        if let Some(r) = parts.reasons { p = p.with_entry(REASONS, Arc::new(r)); }
        if let Some(t) = parts.goal_table { p = p.with_entry(GOAL_TABLE, Arc::new(t)); }
        if let Some((term, d, sample)) = parts.measure { p = p.with_statement(measure_statement(&term, d, &sample)); hold = Some(MEASURE.to_string()); }
        for (id, term, d, sample) in parts.more_measures { let mut st = measure_statement(&term, d, &sample); st.id = id; p = p.with_statement(st); }
        WakeMemory { long: Some(Reader::new(p)), game: game.unwrap_or_else(|| GameMemory::new(key)), hold }
    }
    /// The same for a caller that has no game's key to give (the unit tests): the partition is the one of a game named by nothing but this call.
    pub fn given(parts: Parts, game: Option<GameMemory>) -> WakeMemory { WakeMemory::of(GameKey::of("given", "given"), parts, game) }
    /// The reader of the long-term memory, if it is a reader of THIS game: a game's memory that knows its key refuses a reader opened for another game (what was learned in one game is not read in another).
    pub fn long(&self) -> Option<&Reader> { let r = self.long.as_ref()?; match self.game.key() { Some(k) if k != r.key() => None, _ => Some(r) } }
    /// The worth record (None: not given: the empty record).
    pub fn record(&self) -> Option<&Record> { self.long().and_then(|r| r.get::<Record>(RECORD)) }
    pub fn chances(&self) -> Option<&ChanceTable> { self.long().and_then(|r| r.get::<ChanceTable>(CHANCES)) }
    pub fn reasons(&self) -> Option<&Reasons> { self.long().and_then(|r| r.get::<Reasons>(REASONS)) }
    pub fn goal_table(&self) -> Option<&GoalTable> { self.long().and_then(|r| r.get::<GoalTable>(GOAL_TABLE)) }
    /// The open tactics of the last decision, from the slot of the game's memory.
    pub fn previous(&self) -> Option<&Carried> { self.game.slot::<Carried>(CARRIED) }
    /// The measure to hold: the term and depth of the statement `hold` names in the long-term memory (None: no goal given, or a statement that does not say a term to lower, or a term that does not parse).
    pub fn measure(&self) -> Option<MeasureGoal> {
        let st = self.long()?.statement(self.hold.as_deref()?)?;
        let Says::Lower { term, within } = &st.says else { return None };
        Some(MeasureGoal { term: cs_tactics::measure::Term::parse(term).ok()?, d: *within })
    }
}
