//! PLAN 79.25 PART A, LAYER 4, MEANS-ENDS: THE SEARCH THAT ADVANCES THE VALUE (SYSTEM_SPEC 11 "the layers", 4 tactical reasoning, 7 the measure of progress; ledger row 36; piece: tactical reasoner).
//!
//! THE WORK OF LAYER 4. Given a position and a budget, to find by search the line that gains by force, and to say what it found: the move, the line, the worth at the line's end, and whether the end is
//! proved (won, lost: the game's own end was reached on every reply) or only estimated (the line ends where the budget or the horizon did, with the units taken counted). It uses layers 1 to 3 and nothing above: no goal, no
//! frame, no memory of the game beyond layer 1's history (the record of positions already met), no wake, no clock that decides (a clock may only stop it).
//!
//! WHAT IT USES, AND HOW. Layer 2 (cs_game): the game through `Game` (`moves`, `apply`, `outcome`, `key`, `captured`, `changed`, `move_name`). Layer 1 (cs_rep): the history, `cs_rep::history::Seen` (a position met again is the rules' repeat), and the line being walked, `cs_rep::history::Path`.
//! Layer 3 (cs_value): the worth at a line's end is the sum of a JUDGE over the units the line takes (`ledger::Judge`); the judge's change for a unit taken is the change of layer 3's material count (`cs_value::worth::material`), which tests/layer3_one_copy.rs shows. The search does NOT
//! call layer 3's `Assess` inside a line: a measure orders and names, it never enters a line's value (the rule of section 8 and 15), and reading the value of the kinds at a leaf would change what the wake decides. That is stated again under "not right yet" in the plan's block.
//!
//! INTERFACE (what it takes, what it returns, who calls it). ONE ENTRY:
//!   `search(&Ask) -> Found`
//!   `Ask`      the game (layer 2), the state, the judge (a table of what a unit taken is worth to a side; `ledger::Count` counts no worth), the side whose gain is sought, the BUDGET in nodes (`nodes`, the bound that decides: the same answer on any machine), optionally a clock in seconds (`seconds`: it
//!              may stop the search earlier, never later, and a result that depends on it is not repeatable), the deepest depth asked (`max_depth`, plies), and the history (layer 1's `Seen`, shared by reference count, never copied).
//!   `Found`    the first move (an index into the state's legal moves, and its name), the line (the names of the moves of both sides to the line's end), the units the line takes (owner, kind), the value of the line for the side asked (the judge's sum; the game's own end is above or below every sum), the ENDING
//!              (`Won`, `Lost`: the line ends the game and the search read every reply; `Estimated`: it did not), the nodes spent, the deepest depth read in full, and whether the budget cut the reading (a cut is a gap, never a verdict: the answer is the deepest depth finished).
//!   Called by: this layer's test (rust/engine/tactics/tests/layer4.rs); the wake, the old player's solver (`cs_decision::wake`, the binding's `solve_batch` and `best_move`), the operators (`operators.rs`) and the frames' reads (`lines.rs`) reach the SAME `ledger::Search` directly, past this entry: they are listed in the plan's block and not changed here
//!   (changing them is a matter of the wake's decisions). One copy of the search: this entry builds a `ledger::Search` with its defaults and runs it; it adds nothing to what it decides.
use crate::ledger::{Judge, Search, PROVED};
use cs_game::{Game, Role};
use cs_rep::history::Seen;
use std::sync::Arc;
use std::time::Instant;

/// What the search is asked.
pub struct Ask<'a, G: Game, J: Judge> {
    pub game: &'a G,
    pub state: &'a G::State,
    pub judge: &'a J,
    /// The side whose gain is sought.
    pub side: Role,
    /// The budget in nodes: the bound that decides.
    pub nodes: u64,
    /// A clock that may only stop the search earlier.
    pub seconds: Option<f64>,
    /// The deepest depth asked, in plies.
    pub max_depth: i32,
    /// The record of positions already met (layer 1).
    pub history: Arc<Seen>,
}

/// How the line ends.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ending {
    /// The line ends the game in the asked side's favour and every reply was read.
    Won,
    /// The line ends the game against the asked side.
    Lost,
    /// The line ends where the budget or the horizon did; the units taken are counted.
    Estimated,
}

/// What the search found.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Found {
    pub first: Option<usize>,
    pub first_name: Option<String>,
    pub line: Vec<String>,
    pub taken: Vec<(Role, String)>,
    pub value: i64,
    pub ending: Ending,
    pub nodes: u64,
    pub depth: i32,
    pub cut: bool,
}

/// Whether a value is the game's own end. The ledger's sums are small and the game's end is `WIN` less its ply, with the sums of the line's takings added: a sacrifice on the way makes `WIN - ply - 1000`, so the end is read as
/// at or above `PROVED` (half of `WIN`), the ledger's one constant, which is also where the deepening stops (plan 79.26 part B: it stopped at `WIN - 1000` and so did not stop on a line that gives up a unit).
pub fn ending_of(value: i64) -> Ending {
    if value >= PROVED { Ending::Won } else if value <= -PROVED { Ending::Lost } else { Ending::Estimated }
}

/// THE ENTRY OF LAYER 4 (see the file's doc).
pub fn search<G: Game, J: Judge>(ask: &Ask<G, J>) -> Found {
    let mut s = Search::with_fuse(ask.game, ask.judge, ask.side, ask.nodes);
    s.history = ask.history.clone();
    if let Some(secs) = ask.seconds { s.stop_at(Instant::now() + std::time::Duration::from_secs_f64(secs.max(0.0))); }
    let line = s.run(ask.state, ask.max_depth);
    Found {
        first_name: line.first.clone(), first: line.first_index.filter(|_| line.first.is_some()), ending: ending_of(line.value), value: line.value, line: line.line, taken: line.lost,
        nodes: s.nodes, depth: line.depth, cut: s.cut(),
    }
}
