//! PLAN 79.24 LAYER 3, HOW GOOD A POSITION IS: THE VALUE AND THE MEASURE OF PROGRESS BEHIND ONE INTERFACE (SYSTEM_SPEC 11 "the layers", "layer 3 holds two implementations: value and the measure of progress", 6 the estimator, 7 the measure of progress;
//! 15 the standing limits; ledger rows 35 and 36; piece: the value and the measure of progress). Game-free: it names no object of any game. It sits over layer 1 (cs_rep: the statement, the two memories, the board's names) and layer 2 (cs_game: the
//! game's interface and its placed reading), and below the means-ends crate (cs_tactics), which calls it. It depends on no crate above itself and on no learning crate.
//!
//! THE WORK. To say, of a position and a side, how good it is, from statistics and from the graph, WITHOUT SEARCHING. What needs a search to be computed is not here (see "what stays above").
//!
//! THE INTERFACE (written before the bodies).
//!   `Question { game, state, side }`   what both implementations are asked: the game through layer 2 (`Game`; the measure also reads the game's placed reading, `Placed`), the position (the game's state, the one layer 2 keeps and layer 1's graph is made
//!                                      from), and the side the answer is for. A statement of layer 1, a table or a model is NOT in the question: an implementation is made once from the long-term memory of ONE game, through that memory's reader
//!                                      (`Value::from_memory`, `Count::from_memory`, `Progress::from_memory`); it has no way to write to it and it never sees another game's partition (layer 1's reader is of one game).
//!   `trait Assess<G>`                  `name()` and `assess(&Question) -> Result<Answer, Unanswered>`. The one question.
//!   `Answer`                           `rank`: an ordered tuple, compared from its first number, HIGHER IS BETTER FOR THE SIDE ASKED (a value's expected score; a measure's tuple negated, since a measure FALLS as a goal comes nearer);
//!                                      `raw`: the numbers as the implementation reads them (the measure's own tuple, lower is nearer; None for a value); `chance`: the chances of the side asked (win, draw, loss) where the implementation gives chances (the
//!                                      value does, the measure does not); `sure`: how sure it is, as layer 1's `Standing` of the table it read (a belief counted from results with its count, or a conjecture on a named sample), where it has one;
//!                                      `read`: what it read, in words, for the reasons the wake logs.
//!   `Unanswered(why)`                  the answer is not had: the table is not in the memory, or the position is of a class the table does not speak of. A cut is a gap, never a number.
//!   `Chance`                           win, draw and loss of a side, summing to one; `score()` is win plus half the draw.
//!
//! THE TWO IMPLEMENTATIONS.
//!   `Value`      (module `worth`)     fitted on RESULTS ONLY (human games, generated games; never a number of an outside engine; no worth assumed). A trained model has its own interface: what a table must give to be loaded is `worth::KindModel`: the kinds it
//!                                     reads by name, a per-class intercept and one coefficient per kind and class, the count of positions it was fitted on and its source; it takes the count of each kind of the side to move less the other side's and
//!                                     returns (win, draw, loss) of the side to move. It is `Kept`: written to a partition in sleep, read back in the wake, its standing a conjecture on its sample.
//!   `Count`      (module `worth`)     the plain count of material the value was fitted beside: every unit the same; a table of its own (`worth::CountModel`) of the same form and the same source, one coefficient. It is the baseline the value is held to.
//!   `Progress`   (module `progress`)  the measure of progress: a term over the graph (`Term`, atoms read from the board and the units) that falls as a goal comes nearer. A statement of the long-term memory (`Says::Lower { term, within }`) says which term
//!                                     and within how many of our moves; the board's geometry is made once from layer 2's graph. A measure ORDERS AND NAMES; it never enters a line's value (SYSTEM_SPEC 7, AGENTS rule 7).
//!   They stay two things (Kaveh 2026-11-21); whether they say the same is measured by the layer's test, not assumed.
//!
//! WHO CALLS. The wake (`cs_decision::wake::measure_step`) reads the held goal's numbers through `Progress::assess`; the search's material count (`cs_planning::measure`) is `worth::material`; the means-ends crate re-exports the progress atoms and terms
//! (`cs_tactics::measure`) and adds the questions that need a search. The binding builds the memory's reader; sleep writes the tables (the miners are in the learning crate, which this crate does not name).
//!
//! WHAT STAYS ABOVE, BECAUSE IT NEEDS A SEARCH (named, not moved): `cs_tactics::measure::{lowering, question, Found, Lowering}` (is there a line of our moves that lowers the term against every reply: a game tree is walked); `cs_tactics::ledger` (the line's
//! ledger: the units taken along a line, judged by a table of worths; the line is searched); the chance that a GOAL is reached (`cs_decision::estimate`, `goal_table`: a goal is a thing the questions of the frames make); the fit of every table
//! (`cs_learn::valuation`, the maker scripts; sleep). The value of a position that is not quiet is a thing the search settles first: this layer's value reads the position as it stands.
//!
//! THE TEST OF BEING DONE (tests/layer3.rs, reading fixtures made once by catspace/incubator/analyst/discovery/layer3_fixtures_print.py; uses layers 1 to 3 only and no search, no outside program).
pub mod progress;
pub mod worth;

use cs_game::{Game, Role};
use cs_rep::statement::Standing;
use std::cmp::Ordering;

/// What both implementations are asked.
pub struct Question<'a, G: Game> {
    pub game: &'a G,
    pub state: &'a G::State,
    /// the side the answer is for
    pub side: Role,
}

/// The chances of a side.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Chance { pub win: f64, pub draw: f64, pub loss: f64 }

impl Chance {
    /// win plus half the draw
    pub fn score(&self) -> f64 { self.win + self.draw / 2.0 }
    /// the same chances seen from the other side
    pub fn flip(&self) -> Chance { Chance { win: self.loss, draw: self.draw, loss: self.win } }
}

/// What an implementation answers.
#[derive(Debug, Clone, PartialEq)]
pub struct Answer {
    /// an ordered tuple, higher is better for the side asked, compared from its first number
    pub rank: Vec<f64>,
    /// the numbers as the implementation reads them (the measure's tuple: lower is nearer); None for a value
    pub raw: Option<Vec<i64>>,
    pub chance: Option<Chance>,
    /// how sure it is: the standing of the table it read
    pub sure: Option<Standing>,
    /// what it read, in words
    pub read: Vec<String>,
}

impl Answer {
    /// The order of two answers: by rank, first number first (a total order on the numbers).
    pub fn order(&self, other: &Answer) -> Ordering {
        for (a, b) in self.rank.iter().zip(&other.rank) {
            let c = a.total_cmp(b);
            if c != Ordering::Equal { return c; }
        }
        self.rank.len().cmp(&other.rank.len())
    }
}

/// The answer is not had (and why). Never a number in its place.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Unanswered(pub String);

impl std::fmt::Display for Unanswered {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { write!(f, "unanswered: {}", self.0) }
}

/// THE ONE QUESTION: how good is this position for this side?
pub trait Assess<G: Game> {
    /// a stable name of the implementation and of the table it reads
    fn name(&self) -> String;
    fn assess(&self, q: &Question<G>) -> Result<Answer, Unanswered>;
}

/// Rank agreement of two lists of numbers over the same positions: Kendall's tau-b (ties counted as the usual correction does), in [-1, 1]; None where one list is constant or has fewer than two entries.
pub fn kendall_tau(a: &[f64], b: &[f64]) -> Option<f64> {
    let n = a.len().min(b.len());
    if n < 2 { return None; }
    let (mut conc, mut disc, mut ta, mut tb) = (0i64, 0i64, 0i64, 0i64);
    for i in 0..n {
        for j in i + 1..n {
            let (da, db) = ((a[i] - a[j]).partial_cmp(&0.0).unwrap_or(Ordering::Equal), (b[i] - b[j]).partial_cmp(&0.0).unwrap_or(Ordering::Equal));
            match (da, db) {
                (Ordering::Equal, Ordering::Equal) => { ta += 1; tb += 1; }
                (Ordering::Equal, _) => ta += 1,
                (_, Ordering::Equal) => tb += 1,
                _ if da == db => conc += 1,
                _ => disc += 1,
            }
        }
    }
    let pairs = (n * (n - 1) / 2) as i64;
    let denom = (((pairs - ta) as f64) * ((pairs - tb) as f64)).sqrt();
    if denom == 0.0 { None } else { Some((conc - disc) as f64 / denom) }
}
