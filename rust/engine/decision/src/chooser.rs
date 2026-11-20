//! PLAN 79.12 B, THE LAST LAYER: THE CHOOSER AMONG THE MOVES THE SEARCH HOLDS EQUAL (SYSTEM_SPEC 7 "Calculation and estimate, decided together" points 4 to 6, 8 "the rule for choosing a move" and the wall; piece: the wake's last layer).
//! The wall stands: only the moves the calculation holds equal are candidates, so this layer never picks a move the search holds worse, and it adds nothing to any value (rule 7). It is a function of NAMED INPUTS, every one of them a number the wake already logs; no input names an object of the game and no input is a count over the whole board.
//!
//! INTERFACE (written before the bodies).
//!   `INPUTS`                       the names of the inputs, in the order of a row. A weight is by name; a name not in the list is an error (`ChooserSpec::new`).
//!   `MoveFacts`                    what the wake knows of one candidate move (the facts, each already in the decision log: whether it is the search's first, whether the equal reading kept the first, what it answers and the stake of that threat, the goal it fixes and the stake, the things
//!                                  in the way of that goal and of the top of its chain, the goal's chance, the reasons' score, their chain after it (in the way after, the change, taken out, put in), whether it is a taking and the stake of what is taken). A stake that is the goal kind is
//!                                  `f64::INFINITY` in the facts and becomes a flag in the row (an input is always a finite number).
//!   `inputs_of(&MoveFacts) -> Vec<f64>`  the row, in the order of `INPUTS`.
//!   `Mode::{Fixed, Weights, Explore}`    Fixed: the rule of rule.rs decides (the default; this layer returns the rule's pick untouched); Weights: the largest weighted sum plays (a tie goes to the rule's pick when it is among the tied, else the first); Explore: one of the candidates uniformly at random from the seed
//!                                  given, the propensity (1 over the number of candidates) returned for the log.
//!   `ChooserSpec { mode, weights, seed }`, `ChooserSpec::new(mode, &[(name, weight)], seed) -> Result`.
//!   `apply(&ChooserSpec, rows, fixed) -> Applied`   takes the rows of the candidates (the search's first among them where it is held), the index of the rule's pick (None: the rule's pick is not a candidate); returns the index played, the weighted sum of every candidate (zero with no weights), the propensity.
//!   Called by: `wake::decide` (step 6), which builds the facts and writes the rows, sums and propensity into the decision log; the unit tests. Nothing here reads a game, a clock or a file.

/// The names of the inputs, in the order of a row.
pub const INPUTS: [&str; 19] = [
    "is_first", "first_held",
    "answers_threat", "answers_threat_stake", "answers_end_threat",
    "fixes_goal", "fix_stake", "fix_goal_kind", "fix_in_way", "fix_top_in_way", "fix_chance",
    "reasons_score",
    "their_in_way_after", "their_in_way_change", "their_took_out", "their_put_in",
    "is_taking", "taking_stake", "taking_goal_kind",
];

/// The goal fixed by a move: the stake of the chain it belongs to (INFINITY: the goal kind), the things in the way of the goal and of the goal at the top of the chain, the goal's chance (None: no table).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct FixFacts { pub stake: f64, pub in_way: usize, pub top_in_way: usize, pub chance: Option<f64> }

impl FixFacts { pub fn new(stake: f64, in_way: usize, top_in_way: usize, chance: Option<f64>) -> FixFacts { FixFacts { stake, in_way, top_in_way, chance } } }

/// Their chain after the move: the things in the way of its nearest goal before (None: no such chain stood) and after, the number of our units taken out of its way and put in its way.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct AfterFacts { pub before: Option<usize>, pub after: usize, pub took_out: usize, pub put_in: usize }

/// What the wake knows of one candidate move (see the module doc). A stake of `f64::INFINITY` is the goal kind.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct MoveFacts {
    pub is_first: bool, pub first_held: bool,
    pub answers_threat: bool, pub threat_stake: f64,
    pub fix: Option<FixFacts>, pub reasons_score: Option<f64>, pub after: Option<AfterFacts>,
    pub taking: Option<f64>,
}

fn finite(x: f64) -> f64 { if x.is_finite() { x } else { 0.0 } }
fn flag(b: bool) -> f64 { if b { 1.0 } else { 0.0 } }

/// The row of inputs of a move, in the order of `INPUTS`. Every value is finite.
pub fn inputs_of(f: &MoveFacts) -> Vec<f64> {
    let fix = f.fix.as_ref();
    let after = f.after.as_ref();
    vec![
        flag(f.is_first), flag(f.is_first && f.first_held),
        flag(f.answers_threat), if f.answers_threat { finite(f.threat_stake) } else { 0.0 }, flag(f.answers_threat && !f.threat_stake.is_finite()),
        flag(fix.is_some()), fix.map_or(0.0, |x| finite(x.stake)), flag(fix.is_some_and(|x| !x.stake.is_finite())), fix.map_or(0.0, |x| x.in_way as f64), fix.map_or(0.0, |x| x.top_in_way as f64), fix.and_then(|x| x.chance).unwrap_or(0.0),
        f.reasons_score.unwrap_or(0.0),
        after.map_or(0.0, |a| a.after as f64), after.map_or(0.0, |a| a.before.map_or(0.0, |b| a.after as f64 - b as f64)), after.map_or(0.0, |a| a.took_out as f64), after.map_or(0.0, |a| a.put_in as f64),
        flag(f.taking.is_some()), f.taking.map_or(0.0, finite), flag(f.taking.is_some_and(|x| !x.is_finite())),
    ]
}

/// How the choice among the equal moves is made.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Mode { #[default] Fixed, Weights, Explore }

/// The mode, the weights (a weight a name of `INPUTS`; the rest are nought) and the seed of Explore.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ChooserSpec { pub mode: Mode, pub weights: Vec<f64>, pub seed: u64 }

impl ChooserSpec {
    /// A spec from named weights; a name that is no input is an error.
    pub fn new(mode: Mode, named: &[(String, f64)], seed: u64) -> Result<ChooserSpec, String> {
        let mut w = vec![0.0; INPUTS.len()];
        for (n, x) in named {
            let Some(i) = INPUTS.iter().position(|k| k == n) else { return Err(format!("no such input of the chooser: {n}")) };
            w[i] = *x;
        }
        Ok(ChooserSpec { mode, weights: w, seed })
    }
    /// The weighted sum of a row.
    pub fn sum(&self, row: &[f64]) -> f64 { row.iter().zip(&self.weights).map(|(x, w)| x * w).sum() }
}

/// What the layer returns: the index played among the candidates, the weighted sum of each, the propensity of the move played (1 where the choice was not random).
#[derive(Debug, Clone, PartialEq)]
pub struct Applied { pub index: usize, pub sums: Vec<f64>, pub propensity: f64 }

fn splitmix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

/// THE LAYER (see the module doc). `rows` empty gives index 0 and no sums; one candidate gives it.
pub fn apply(spec: &ChooserSpec, rows: &[Vec<f64>], fixed: Option<usize>) -> Applied {
    let sums: Vec<f64> = rows.iter().map(|r| spec.sum(r)).collect();
    let n = rows.len();
    let keep = fixed.filter(|i| *i < n).unwrap_or(0);
    if n < 2 { return Applied { index: 0, sums, propensity: 1.0 }; }
    match spec.mode {
        Mode::Fixed => Applied { index: keep, sums, propensity: 1.0 },
        Mode::Explore => Applied { index: (splitmix(spec.seed) % n as u64) as usize, sums, propensity: 1.0 / n as f64 },
        Mode::Weights => {
            let top = sums.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let tied: Vec<usize> = (0..n).filter(|i| (sums[*i] - top).abs() <= 1e-12 * top.abs().max(1.0)).collect();
            let index = if tied.contains(&keep) { keep } else { tied[0] };
            Applied { index, sums, propensity: 1.0 }
        }
    }
}
