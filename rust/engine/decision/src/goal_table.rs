//! PLAN 79.13 C, THE TABLE OF THE CHANCE OF A GOAL, READ IN THE WAKE (SYSTEM_SPEC 7 "Calculation and estimate, decided together" point 2: what is not near is estimated per goal; 6 the event heads; piece: the estimate per goal). The table is COUNTED IN SLEEP from the logged games (catspace/incubator/analyst/discovery/goal_table.py); nothing is counted here. This module only
//! builds a goal's key from the facts the wake already holds for it and reads the cell. The chance is that the goal's target is taken by the goal's side within `k` plies without the side falling behind (the event of goal_reach.py); it is an INPUT to the last layer and NEVER a value added to the judge (rule 7).
//! Game-free: no field names an object of any game; a kind is a number whose worth the record gives.
//!
//! INTERFACE (written before the bodies; who calls: `wake::decide`, behind `WakeInputs::goal_table`, and the unit tests).
//!   `Field`                          the fields a key is made of: source, unmet, in_way, depth, rank, fix, mine (`Field::named`).
//!   `GoalFacts::of(goal, mine, rank)`  the facts of a goal the key reads: its class word, the set of its unmet facts, the things in its way, its depth in its chain, the rank of its target's stake, whether a fix with a move exists, whether it is the mover's.
//!   `GoalFacts::key(&[Field])`       the values joined by "|" (the unmet facts sorted, distinct, joined by "+"; fix and mine as 0 or 1): the text the sleep counts under.
//!   `rank_of(stake, worths)`         the number of distinct worths of the record strictly under the stake (the goal kind's stake is above every worth).
//!   `GoalTable`                      the levels, each a list of fields and the cells (key -> goals, reached) that held at least `floor` goals, from the full key to the fewest, and the whole's counts.
//!   `GoalTable::chance(&GoalFacts)` -> `Estimate` { key, chance, n, source }: (reached + 1) / (n + 2) of the first level whose cell speaks, else of the whole.
use crate::estimate::Estimate;
use crate::goals::Goal;
use std::collections::HashMap;

/// The fields a key is made of.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Field { Source, Unmet, InWay, Depth, Rank, Fix, Mine }

impl Field {
    /// The field of a name (the names the sleep writes in the table).
    pub fn named(n: &str) -> Option<Field> {
        Some(match n { "source" => Field::Source, "unmet" => Field::Unmet, "in_way" => Field::InWay, "depth" => Field::Depth, "rank" => Field::Rank, "fix" => Field::Fix, "mine" => Field::Mine, _ => return None })
    }
}

/// What a key reads of a goal.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct GoalFacts { pub source: String, pub unmet: String, pub in_way: usize, pub depth: u8, pub rank: usize, pub fix: bool, pub mine: bool }

impl GoalFacts {
    /// The facts of a goal; `mine`: the goal is the mover's of the decision; `rank`: `rank_of` the stake of its target.
    pub fn of(goal: &Goal, mine: bool, rank: usize) -> GoalFacts {
        let mut u: Vec<&str> = goal.unmet.to_vec();
        u.sort();
        u.dedup();
        GoalFacts { source: goal.source.clone(), unmet: u.join("+"), in_way: crate::rule::things_in_the_way(goal), depth: goal.depth, rank, fix: goal.fixes.iter().any(|f| !f.moves.is_empty()), mine }
    }
    /// The value of one field as text.
    pub fn value(&self, f: Field) -> String {
        match f {
            Field::Source => self.source.clone(), Field::Unmet => self.unmet.clone(), Field::InWay => self.in_way.to_string(), Field::Depth => self.depth.to_string(), Field::Rank => self.rank.to_string(),
            Field::Fix => (self.fix as u8).to_string(), Field::Mine => (self.mine as u8).to_string(),
        }
    }
    /// The key of these facts on a list of fields.
    pub fn key(&self, fields: &[Field]) -> String { fields.iter().map(|f| self.value(*f)).collect::<Vec<_>>().join("|") }
}

/// The rank of a stake among the kinds: how many distinct worths of the record are strictly under it. The goal kind's stake is infinite and ranks above every worth.
pub fn rank_of(stake: f64, worths: &[i64]) -> usize {
    let mut v: Vec<i64> = worths.to_vec();
    v.sort();
    v.dedup();
    v.iter().filter(|w| (**w as f64) < stake).count()
}

/// One level of the table: the fields of its key and the cells that held at least the floor.
#[derive(Debug, Clone, Default)]
pub struct Level { pub fields: Vec<Field>, pub cells: HashMap<String, (u32, u32)> }

/// The table: `k` the plies of the event, `floor` the least count a cell needs to speak, the levels from the full key to the fewest, and the counts of all goals counted (goals, reached).
#[derive(Debug, Clone, Default)]
pub struct GoalTable { pub k: u32, pub floor: u32, pub levels: Vec<Level>, pub total: (u32, u32) }

const LEVEL_NAMES: [&str; 9] = ["level0", "level1", "level2", "level3", "level4", "level5", "level6", "level7", "level8"];

impl GoalTable {
    /// THE CHANCE of a goal: the first level with a cell of at least `floor` goals speaks; else the whole's rate; all smoothed (reached + 1) / (n + 2) as the table of 78.74 is.
    pub fn chance(&self, f: &GoalFacts) -> Estimate {
        for (i, lv) in self.levels.iter().enumerate() {
            let k = f.key(&lv.fields);
            if let Some(x) = lv.cells.get(&k) {
                if x.0 >= self.floor { return Estimate { key: k, chance: (x.1 as f64 + 1.0) / (x.0 as f64 + 2.0), n: x.0, source: LEVEL_NAMES[i.min(LEVEL_NAMES.len() - 1)] }; }
            }
        }
        Estimate { key: String::new(), chance: (self.total.1 as f64 + 1.0) / (self.total.0 as f64 + 2.0), n: self.total.0, source: "total" }
    }
}
