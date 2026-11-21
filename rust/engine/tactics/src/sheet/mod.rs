//! PLAN 78.43 STAGE A: THE QUESTION SHEET'S LIGHT READING AND ITS LEVELS 0 AND 1 BY GEOMETRY, in the engine, game-free.
//!
//! The Python sheet (catspace/incubator/analyst/discovery/question_sheet.py) is the readable mirror and the ORACLE of this module. A position is read once (`reading`): its units,
//! what each bears on and where each may move as the board stands (asked of the game through `cs_game::placed::Placed`, from ANY place), the worth of each kind (a table
//! supplied from outside) and the goal's kind. The MEDIUM OF MOVEMENT (`medium`: the one-move hops on the empty board, the places passed over, which kinds take only) is a data
//! table supplied from outside, read by the caller from the game's description. `goal_report` is the goal unit's report (holes, covered places, supply, scenarios) made from
//! the light reading; `levels` holds level 0 (the board as it stands) and level 1 (the asking side's moves alone: the goal unit by the report, any other unit by the medium
//! of movement, the routes of each way with the obstacles on them as the board stands). No object of any game is named here.
//!
//! STAGE B0: places and kinds are NUMBERS (`Place`, `Kind`: the game's own lists of names, in the order of the text), sets of places are bit sets; the text is made only at the
//! boundary (`Names`). Units, ways and routes are small copyable or cheaply cloned values, nothing is hashed by text.
pub mod base_graph;
pub mod base_statements;
pub mod cf;
pub mod delta;
pub mod exchange;
pub mod fates;
pub mod forcing;
pub mod frame;
pub mod frames;
pub mod keeper;
pub mod questions;
pub mod goal_report;
pub mod levels;
pub mod medium;
pub mod mv;
pub mod patterns;
pub mod reading;
pub mod run;
pub mod settle;
pub mod subgoal;
pub mod threat_frames;
pub mod walk;

// plan 79.21: the places, kinds, sides, units, names and place sets are layer 1's (cs_rep::names); re-exported so every path here keeps resolving
pub use cs_rep::names::{Kind, Names, Place, PlaceSet, Side, Unit};
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status { Live, Idle, Closed }

impl Status {
    pub fn word(self) -> &'static str { match self { Status::Live => "live", Status::Idle => "idle", Status::Closed => "closed" } }
}

/// The named numbers of the levels (the Python sheet's `Params`, the ones levels 0 and 1 read).
#[derive(Debug, Clone)]
pub struct Params {
    /// Moves of its own a supplying unit may use.
    pub reach_k: u32,
    /// The goal unit, no full plan read: live only when at most this many of its exits are open.
    pub max_open_exits: usize,
    /// The goal unit's stake: this many times the largest worth.
    pub goal_stake_multiple: f64,
    /// No unit's stake is lower.
    pub min_stake: f64,
    // ---- stage B: the deeper levels (the Python sheet's own defaults)
    /// Ways settled per step.
    pub max_ways: usize,
    /// Nodes one frame search may expand.
    pub frame_budget: u64,
    /// Routes of a way the route follower takes, the best-ordered first.
    pub max_routes: usize,
    /// Nodes one way may use over all its routes; past it the way is NOT SETTLED, never refuted.
    pub route_nodes: u64,
    /// Blocked steps one question may examine by the counterfactual.
    pub max_blocked_steps: usize,
    /// Places tried when a unit the game will not read the board without is moved aside instead of removed.
    pub aside_places: usize,
    /// Units in a joint set; more is recorded as "more than the cap in the way".
    pub max_joint: usize,
    /// Moves of an own unit in the way tried when asking whether it can move away.
    pub max_obstacle_moves: usize,
    /// The search is not deepened past this many plies.
    pub search_depth_cap: usize,
    /// Nodes the reasoning search may use on one child.
    pub reason_fuse: u64,
    /// Nodes the census may use.
    pub census_fuse: u64,
    /// The deepest level; a question that finds nothing there is closed.
    pub max_level: usize,
}

impl Default for Params {
    fn default() -> Params { Params { reach_k: 2, max_open_exits: 1, goal_stake_multiple: 2.0, min_stake: 0.1, max_ways: 4, frame_budget: 20_000, max_routes: 3, route_nodes: 2500, max_blocked_steps: 12,
                    aside_places: 4, max_joint: 3, max_obstacle_moves: 8, search_depth_cap: 3, reason_fuse: 50_000, census_fuse: 2_000_000, max_level: 4 } }
}

/// What belongs to the game and not to the position, made once per model: the names, the medium of movement, the worth of each kind (by number), the goal's kind and how many move
/// relations the vocabulary names.
pub struct Model {
    pub names: Names,
    pub medium: medium::Medium,
    pub worth: Vec<f64>,
    pub goal_kind: Option<Kind>,
    pub can_go: usize,
    /// The table the tactics search is judged by (its worth by the game's own key of a kind) and the key of each kind in it and in the tactics' names.
    pub ledger_worth: HashMap<String, i64>,
    pub letters: Vec<String>,
}

/// A cheap hasher for the small keys of the layer (numbers, units, ways): no text is ever hashed here.
#[derive(Default, Clone, Copy)]
pub struct Fx(u64);

impl std::hash::Hasher for Fx {
    fn write(&mut self, bytes: &[u8]) { for b in bytes { self.add(*b as u64); } }
    fn write_u8(&mut self, n: u8) { self.add(n as u64); }
    fn write_u16(&mut self, n: u16) { self.add(n as u64); }
    fn write_u32(&mut self, n: u32) { self.add(n as u64); }
    fn write_u64(&mut self, n: u64) { self.add(n); }
    fn write_usize(&mut self, n: usize) { self.add(n as u64); }
    fn finish(&self) -> u64 { self.0 }
}

impl Fx {
    fn add(&mut self, n: u64) { self.0 = (self.0.rotate_left(5) ^ n).wrapping_mul(0x517c_c1b7_2722_0a95); }
}

pub type FxMap<K, V> = HashMap<K, V, std::hash::BuildHasherDefault<Fx>>;
pub type FxSet<K> = std::collections::HashSet<K, std::hash::BuildHasherDefault<Fx>>;
