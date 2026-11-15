//! PLAN 78.43 STAGE A: THE QUESTION SHEET'S LIGHT READING AND ITS LEVELS 0 AND 1 BY GEOMETRY, in the engine, game-free.
//!
//! The Python sheet (catspace/incubator/analyst/discovery/question_sheet.py) is the readable mirror and the ORACLE of this module. A position is read once (`reading`): its units,
//! what each bears on and where each may move as the board stands (asked of the game through `cs_game::placed::Placed`, from ANY place), the worth of each kind (a table
//! supplied from outside) and the goal's kind. The MEDIUM OF MOVEMENT (`medium`: the one-move hops on the empty board, the places passed over, which kinds take only) is a data
//! table supplied from outside, read by the caller from the game's description. `goal_report` is the goal unit's report (holes, covered places, supply, scenarios) made from
//! the light reading; `levels` holds level 0 (the board as it stands) and level 1 (the asking side's moves alone: the goal unit by the report, any other unit by the medium
//! of movement, the routes of each way with the obstacles on them as the board stands). No object of any game is named here.
pub mod goal_report;
pub mod levels;
pub mod medium;
pub mod reading;

/// The two sides as the reading names them: the side to move and the other.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Side { Us, Them }

impl Side {
    pub fn other(self) -> Side { match self { Side::Us => Side::Them, Side::Them => Side::Us } }
    pub fn word(self) -> &'static str { match self { Side::Us => "us", Side::Them => "them" } }
}

/// A unit: its side, the game's word for its kind, the place it stands on. Ordered as the Python sheet orders units: our side first, then by kind, then by place.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Unit { pub side: Side, pub kind: String, pub at: String }

impl Unit {
    /// "my <kind> on <place>" / "their <kind> on <place>".
    pub fn words(&self) -> String { format!("{} {} on {}", if self.side == Side::Us { "my" } else { "their" }, self.kind, self.at) }
}

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
}

impl Default for Params {
    fn default() -> Params { Params { reach_k: 2, max_open_exits: 1, goal_stake_multiple: 2.0, min_stake: 0.1 } }
}
