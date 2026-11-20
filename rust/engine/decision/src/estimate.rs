//! PLAN 79.08 B, THE ESTIMATES (SYSTEM_SPEC 7 "Calculation and estimate, decided together", points 2 and 3; 13 item 27; piece: the wake). What is not near enough to be calculated is estimated PER GOAL (and per frame, where a table of frames exists), never as one number for the whole position.
//! TWO ESTIMATES, each a number that is an INPUT to the last layer and NEVER a value added to the judge (rule 7; a test says a line's value is the same with and without them):
//!   (1) THE CHANCE OF A GOAL: the chance that a goal's unmet facts come true, read from a TABLE keyed by what stops it (the table of plans 78.71 and 78.74: subgoal_measure.table_of / chance_of; the table is a PARAMETER, given by the caller from the record of those runs); absent, there is no estimate and the log says so.
//!   (2) THE REASONS' SCORE OF A MOVE: the sum of the weights of the reasons (fact patterns whose count a move changes) that the move carries (symbols_play.ReasonsReader.pick; the pattern counter is cs_tactics::sheet::patterns, in Rust; the weights are a PARAMETER). See pick.rs.
//! The heads that read counts summed over the whole position (heads_read.py) are NOT ported and NOT called: point 2 of the section says not to do that. There is no table of FRAMES here: a per-frame estimate needs a table the sleep has not made, and the log says "none" for it.
//!
//! INTERFACE (written before the bodies).
//!   `ChanceTable`                   the table: counts by the goal's key, by the names of its unmet testable facts, and in all; `floor` the least count a cell needs to speak (the Python's 20).
//!   `ChanceTable::chance(goal)`     -> `Estimate` { key, chance, n, source }: (hits + 1) / (n + 2) of the first cell that has at least `floor` goals (key, then facts, then all). Called by the wake's loop for every goal it logs.
use crate::goals::Goal;
use std::collections::HashMap;

/// The names of the facts the table counts (subgoal_measure.TESTABLE): the facts the graph can show a plan hence.
pub const TESTABLE: [&str; 3] = ["bears", "may_take", "unguarded"];

/// The chance of reaching within four plies (the unmet testable facts all hold two own moves on) by the goal's key (subgoal_measure.table_of), as counts: (goals, reached).
#[derive(Debug, Clone, Default)]
pub struct ChanceTable { pub by_key: HashMap<String, (u32, u32)>, pub by_facts: HashMap<String, (u32, u32)>, pub total: (u32, u32), pub floor: u32 }

/// One goal's estimate: its key, the chance, the count behind it and which cell spoke.
#[derive(Debug, Clone, PartialEq)]
pub struct Estimate { pub key: String, pub chance: f64, pub n: u32, pub source: &'static str }

/// The names of a goal's unmet testable facts, sorted and joined (the table's second key).
pub fn facts_key(goal: &Goal) -> String {
    let mut v: Vec<&str> = goal.facts.iter().filter(|f| matches!(f.status, crate::goals::Status::Unmet | crate::goals::Status::Unread) && TESTABLE.contains(&f.name)).map(|f| f.name).collect();
    v.sort();
    v.join("|")
}

impl ChanceTable {
    /// THE CHANCE of a goal (subgoal_measure.chance_of): the first cell with at least `floor` goals speaks; else the whole table's rate; all smoothed (hits + 1) / (n + 2).
    pub fn chance(&self, goal: &Goal) -> Estimate {
        let fk = facts_key(goal);
        for (name, tab, k) in [("key", &self.by_key, goal.key.clone()), ("facts", &self.by_facts, fk)] {
            if let Some(x) = tab.get(&k) { if x.0 >= self.floor { return Estimate { key: goal.key.clone(), chance: (x.1 as f64 + 1.0) / (x.0 as f64 + 2.0), n: x.0, source: name }; } }
        }
        Estimate { key: goal.key.clone(), chance: (self.total.1 as f64 + 1.0) / (self.total.0 as f64 + 2.0), n: self.total.0, source: "total" }
    }
}
