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
//! PLAN 79.27 (SYSTEM_SPEC 11 "Layer 4 takes a goal as an input" and its amendment; ledger rows 36, 14): THE QUESTION TO THIS LAYER CARRIES AN ORDERED LIST OF GOALS and one budget; the answer holds one entry per goal. THE ENTRY IS NOW `ask` (see "THE ENTRY WITH GOALS" at the end of this file); `search` with its `Ask` and `Found`
//! are what `ask` calls for the two goals the layer had before (a gain by the judge, the game's end), unchanged, and they stay public because the test of plan 79.26 reads them as the reference of "unchanged". The text below is the interface of that reading.
//!
//! INTERFACE of the reading of the two old goals (what it takes, what it returns, who calls it). ONE ENTRY:
//!   `search(&Ask) -> Found`
//!   `Ask`      the game (layer 2), the state, the judge (a table of what a unit taken is worth to a side; `ledger::Count` counts no worth), the side whose gain is sought, the BUDGET in nodes (`nodes`, the bound that decides: the same answer on any machine), optionally a clock in seconds (`seconds`: it
//!              may stop the search earlier, never later, and a result that depends on it is not repeatable), the deepest depth asked (`max_depth`, plies), and the history (layer 1's `Seen`, shared by reference count, never copied).
//!   `Found`    the first move (an index into the state's legal moves, and its name), the line (the names of the moves of both sides to the line's end), the units the line takes (owner, kind), the value of the line for the side asked (the judge's sum; the game's own end is above or below every sum), the ENDING
//!              (`Won`, `Lost`: the line ends the game and the search read every reply; `Estimated`: it did not), the nodes spent, the deepest depth read in full, and whether the budget cut the reading (a cut is a gap, never a verdict: the answer is the deepest depth finished).
//!   Called by: this layer's test (rust/engine/tactics/tests/layer4.rs); the wake, the old player's solver (`cs_decision::wake`, the binding's `solve_batch` and `best_move`), the operators (`operators.rs`) and the frames' reads (`lines.rs`) reach the SAME `ledger::Search` directly, past this entry: they are listed in the plan's block and not changed here
//!   (changing them is a matter of the wake's decisions). One copy of the search: this entry builds a `ledger::Search` with its defaults and runs it; it adds nothing to what it decides.
use crate::ledger::{Judge, Search, PROVED};
use crate::reach;
use cs_game::graph::Static;
use cs_game::placed::Placed;
use cs_rep::matcher::Binding;
use cs_rep::statement::Statement;
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

// ------------------------------------------------------------------------------------------------------------------------------ THE ENTRY WITH GOALS (plan 79.27)

/// THE ENTRY WITH GOALS (plan 79.27 parts A and B; SYSTEM_SPEC 11 "Layer 4 takes a goal as an input" and its amendment; ledger rows 36, 14; piece: tactical reasoner).
/// WHAT IT TAKES: `Question { game, board, state, judge, side, goals, nodes, seconds, max_depth, history }`: the game (layer 2: `Game` and `Placed`, with layer 2's `Static` for the board of layer 1), the state, the judge (layer 3's count through the ledger's `Judge`), the side that wants the goals,
/// an ORDERED LIST OF GOALS and ONE budget of nodes (the bound that decides; a clock may only stop it earlier), the deepest depth in plies and the history (layer 1's record). A GOAL is `Goal::Holds(Statement)` (layer 1's form: a pattern over the facts of the graph, `cs_rep::graph_facts` says which, that the side wants to
/// hold), `Goal::End` (the game ends in the side's favour) or `Goal::Gain` (a net gain by the judge). The layer does not choose the goals and does not choose among them by worth: it says what is reachable and at what cost.
/// WHAT IT RETURNS: `Reply { answers, both, nodes }`, one `Answer` per goal in the list's order: `Reached` (by force: the line of both sides' moves, the bindings where a pattern holds, and what the judge says the line took and cost, read after the goal holds, so a goal reached by giving a unit away is said as such), `OutOfReach { depth }`
/// (every line of at most that many plies read, the goal held in none: a statement about that depth only; only the statements can say it), `NotSettled { depth, best }` (the budget cut it; a cut is a gap, never a verdict; for the two old goals `best` is what the ledger search found). `both` lists the pairs of statement goals one line reaches at once, with the line, and, where
/// the search read it in full, whether one reply of the other side leaves both out of reach (None otherwise: nothing is claimed).
/// HOW THE BUDGET IS SPENT: in the order of the list, in groups: the statement goals form one group (one tree serves them all, `reach.rs`) placed where the first of them stands; each of the two old goals is a group of its own. Each group in turn gets an equal share of what is left (what it does not spend goes on to the later ones), and inside the tree the first open goal in the
/// list steers the order of moves. The upper layer's opinion enters as that order, never as a value. A list of one answers exactly as the question of that goal alone.
/// Called by: this layer's tests; no caller above is moved onto it in plan 79.27 (the wake, the question's search of 79.17 and the goals' what-ifs reach the searches directly: they are listed in the plan's block).
///
/// PLAN 79.28 PARTS A AND B (SYSTEM_SPEC 11 "Reached means the goal outlasts the reply"; ledger rows 36, 14; piece: tactical reasoner): THE MEANING OF AN ANSWER CHANGES, THE SHAPE OF THE QUESTION DOES NOT. A statement is HELD at a position when it binds there and the side asking is to move there or the game is over there. REACHED: the side has a
/// way of moving such that, on every reply, a position where the statement is held is met within the depth (the position asked counts, at nought plies: a statement that binds at the root with the side to move is reached at once). So a statement that binds only right after a move of the side, and that some reply unbinds, is not reached by that move.
/// TOUCHED (a fourth answer, said apart, never counted as reached): the statement is not reached within the depth, but the side can force it to bind right after a move of its own (the other side to move, the game not over) in a way that some reply of the other side then unbinds; the answer holds the line to that position, the bindings there and
/// the reply that unbinds it (read in the depth, so the line is one ply shorter than the depth). OUT OF REACH: every line of the depth read, the statement neither reached nor touched. NOT SETTLED: the budget cut the reading, of either kind (a statement not reached but whose touching was not read in full is not settled, not out of reach).
/// The answer of a reached goal says the plies and the bindings at the position where it is held. The two old goals answer as before. `both` lists the pairs one line reaches in this meaning (each held at some position of the line; each reached).
pub enum Goal {
    /// a statement the side wants to hold
    Holds(Statement),
    /// the game ends in the side's favour
    End,
    /// a net gain by the judge
    Gain,
}

pub struct Question<'a, G: Game + Placed, J: Judge> {
    pub game: &'a G,
    pub board: &'a Static,
    pub state: &'a G::State,
    pub judge: &'a J,
    pub side: Role,
    pub goals: &'a [Goal],
    /// the one budget of nodes
    pub nodes: u64,
    pub seconds: Option<f64>,
    pub max_depth: i32,
    pub history: Arc<Seen>,
}

/// A goal reached by force.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Reached {
    pub first: Option<usize>,
    pub first_name: Option<String>,
    pub line: Vec<String>,
    pub plies: i32,
    /// where a statement binds at the line's end (empty for the two old goals)
    pub bindings: Vec<Binding>,
    /// the units the line takes (owner, kind) and the judge's sum for the side over the line
    pub taken: Vec<(Role, String)>,
    pub value: i64,
    /// the ledger's own answer for the two old goals
    pub found: Option<Found>,
}

/// A goal touched (plan 79.28): the pattern binds after a move of the side and a reply unbinds it, and the goal is not reached within the depth. `line` ends at the position where it binds (`bindings` are read there), `reply` is a reply that leaves it unbound.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Touched {
    pub first: Option<usize>,
    pub first_name: Option<String>,
    pub line: Vec<String>,
    pub plies: i32,
    pub reply: String,
    pub bindings: Vec<Binding>,
    pub taken: Vec<(Role, String)>,
    pub value: i64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Answer {
    Reached(Reached),
    Touched(Touched),
    OutOfReach { depth: i32 },
    NotSettled { depth: i32, best: Option<Found> },
}

impl Answer {
    pub fn reached(&self) -> Option<&Reached> { if let Answer::Reached(r) = self { Some(r) } else { None } }
    pub fn touched(&self) -> Option<&Touched> { if let Answer::Touched(r) = self { Some(r) } else { None } }
}

/// One line that reaches two goals of the list at once (both patterns bind somewhere along it).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Both {
    pub goals: (usize, usize),
    pub line: Vec<String>,
    /// whether one reply of the other side, after the line's first move, leaves both out of reach (read in full); None: not read in full
    pub one_reply_meets_both: Option<bool>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Reply {
    pub answers: Vec<Answer>,
    pub both: Vec<Both>,
    pub nodes: u64,
    /// where the cost of the statements' tree went: positions made on the live graph, rows matched
    pub makes: u64,
    pub matches: u64,
    /// plan 79.29 part A: where the calls of the statements' one tree went, by what they were asked (summed over the groups)
    pub diag: reach::Diag,
}

/// THE ENTRY WITH GOALS, with the narrowing of the statements' tree given (the entry of the layer is `ask`, which narrows; the test reads the tree without it to measure the narrowing).
pub fn ask_with<G: Game + Placed, J: Judge>(q: &Question<G, J>, narrow: reach::Narrow) -> Reply {
    let n = q.goals.len();
    let mut answers: Vec<Option<Answer>> = vec![None; n];
    let statements: Vec<usize> = (0..n).filter(|&i| matches!(q.goals[i], Goal::Holds(_))).collect();
    // the groups in the list's order: the statements' tree where its first goal stands, each old goal alone
    let mut groups: Vec<Vec<usize>> = vec![];
    for i in 0..n { if let Goal::Holds(_) = q.goals[i] { if statements.first() == Some(&i) { groups.push(statements.clone()); } } else { groups.push(vec![i]); } }
    let mut left = q.nodes;
    let mut spent = 0u64;
    let (mut makes, mut matches) = (0, 0);
    let mut diag = reach::Diag::default();
    let mut reads: Vec<reach::Read> = vec![];
    let mut tree_goals: Vec<Statement> = vec![];
    for (gi, grp) in groups.iter().enumerate() {
        let share = left / (groups.len() - gi) as u64;
        match &q.goals[grp[0]] {
            Goal::Holds(_) => {
                tree_goals = grp.iter().map(|&i| if let Goal::Holds(s) = &q.goals[i] { s.clone() } else { unreachable!() }).collect();
                let r = reach::read(&reach::Ask { game: q.game, board: q.board, state: q.state, judge: q.judge, side: q.side, goals: &tree_goals, nodes: share, seconds: q.seconds, max_depth: q.max_depth, history: q.history.clone(), narrow });
                left -= r.nodes.min(left); spent += r.nodes; makes += r.makes; matches += r.matches; 
                diag.calls_held += r.diag.calls_held; diag.calls_touched += r.diag.calls_touched; diag.calls_both += r.diag.calls_both;
                for (k, &i) in grp.iter().enumerate() {
                    answers[i] = Some(match &r.outcomes[k] {
                        reach::Outcome::Reached(l) => Answer::Reached(Reached { first: l.first, first_name: l.names.first().cloned(), line: l.names.clone(), plies: l.plies, bindings: l.bindings.clone(), taken: l.taken.clone(), value: l.value, found: None }),
                        reach::Outcome::Touched(t) => Answer::Touched(Touched { first: t.line.first, first_name: t.line.names.first().cloned(), line: t.line.names.clone(), plies: t.line.plies, reply: t.reply.clone(), bindings: t.line.bindings.clone(), taken: t.line.taken.clone(), value: t.line.value }),
                        reach::Outcome::Out { depth } => Answer::OutOfReach { depth: *depth },
                        reach::Outcome::Unsettled { depth } => Answer::NotSettled { depth: *depth, best: None },
                    });
                }
                reads.push(r);
            }
            old => {
                let a = Ask { game: q.game, state: q.state, judge: q.judge, side: q.side, nodes: share, seconds: q.seconds, max_depth: q.max_depth, history: q.history.clone() };
                let f = search(&a);
                left -= f.nodes.min(left); spent += f.nodes;
                let got = match old { Goal::Gain => f.value > 0, _ => f.ending == Ending::Won };
                let i = grp[0];
                answers[i] = Some(if got {
                    Answer::Reached(Reached { first: f.first, first_name: f.first_name.clone(), line: f.line.clone(), plies: f.line.len() as i32, bindings: vec![], taken: f.taken.clone(), value: f.value, found: Some(f.clone()) })
                } else { Answer::NotSettled { depth: f.depth, best: Some(f) } });
            }
        }
    }
    // the pairs of statement goals one line reaches at once
    let mut both = vec![];
    if let Some(r) = reads.first() {
        for a in 0..statements.len() { for b in a + 1..statements.len() {
            let (la, lb) = match (&r.outcomes[a], &r.outcomes[b]) { (reach::Outcome::Reached(x), reach::Outcome::Reached(y)) => (x, y), _ => continue };
            let line = if la.meets & (1 << b) != 0 { la } else if lb.meets & (1 << a) != 0 { lb } else { continue };
            let one = if q.game.mover(q.state) == Some(q.side) && !line.names.is_empty() {
                match q.game.find_move(q.state, &line.names[0]) {
                    Some(m) => {
                        let after = q.game.apply(q.state, &m);
                        let ask = reach::Ask { game: q.game, board: q.board, state: &after, judge: q.judge, side: q.side, goals: &tree_goals, nodes: left, seconds: q.seconds, max_depth: q.max_depth, history: q.history.clone(), narrow };
                        let (v, used) = reach::one_reply_meets(&ask, &after, (a, b), la.plies.max(lb.plies));
                        left -= used.min(left); spent += used;
                        v
                    }
                    None => None,
                }
            } else { None };
            both.push(Both { goals: (statements[a], statements[b]), line: line.names.clone(), one_reply_meets_both: one });
        } }
    }
    Reply { answers: answers.into_iter().map(|a| a.expect("every goal is answered")).collect(), both, nodes: spent, makes, matches, diag }
}

/// THE ENTRY OF LAYER 4 (see the block above).
pub fn ask<G: Game + Placed, J: Judge>(q: &Question<G, J>) -> Reply { ask_with(q, reach::NARROWED) }
