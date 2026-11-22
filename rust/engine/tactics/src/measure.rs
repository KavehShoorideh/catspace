//! PLAN 79.17 PART B, THE MEASURE OF PROGRESS AS A PIECE (SYSTEM_SPEC 7, "the small endings as the test of means and ends, and a measure of progress"; 8 the rule for choosing a move; piece: the opportunity policy, read by the wake).
//! PLAN 79.24 PART B: THE ATOMS, THE TERMS AND THEIR READING OF A POSITION (what needs no search) MOVED DOWN TO LAYER 3, cs_value::progress, and are re-exported here so every old path resolves (one copy). WHAT STAYS HERE BECAUSE IT NEEDS A SEARCH: the question
//! of October's criterion, `lowering` (is there a move of ours after which, within d of our moves and against every reply, the term is lower than held, or the game's end is found: the search of toward.rs with the leaf "the term is below the held value" read where it is
//! our turn) and `question` (79.18 B: the same counted against the lowest value held, a Fall or a Recovery). The measure never enters a line's value: it orders and names.
//! Called by: the binding (`measure_atoms`, `measure_lowers`), the wake (`cs_decision::wake`, the held goal), the miner's driver in Python.
pub use cs_value::progress::*;
use crate::toward::{toward_memo, Class, Verdict};
use cs_game::placed::Placed;
use cs_game::{Game, Role};
use cs_operator::Budget;

/// What October's question found: the first move (an index into the root's legal moves), how many of our moves the line takes, and whether the game's own end is what was found.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Lowering { pub first: usize, pub moves: u32, pub game_end: bool, pub nodes: u64 }

/// Is there a move of ours at `s` (we are to move; `side` is the measure's side and the side to move) after which, within `d` of our moves and against every reply, the term is below `held` where it is our turn, or the game's end is ours? The search of toward.rs, deepened ply by ply to 2d - 1 plies and
/// one more for the reply; a leaf where it is our turn answers "win" when the term there is below `held`. None: no such line in the budget (a cut is a gap, never a result: `Err` of the budget is None as well, `cut` tells which).
pub fn lowering<G: Game + Placed>(g: &G, geo: &Geometry, s: &G::State, side: Role, term: &Term, held: &[i64], d: u32, history: &cs_rep::history::Seen, b: &mut Budget) -> (Option<Lowering>, bool) {
    if g.mover(s) != Some(side) || d == 0 { return (None, false); }
    let oracle = |gm: &G, st: &G::State| -> Option<Verdict> {
        if gm.mover(st) != Some(side) { return None; }
        let r = Reader::new(gm, geo, st, side);
        if term.atoms().iter().any(|a| match a { Atom::Region(x) | Atom::Span(x, _) | Atom::Edge(x) | Atom::Corner(x) | Atom::AxisEdge(x, _) => !r.present(*x), Atom::Dist(x, y) | Atom::Axis(x, y, _) => !r.present(*x) || !r.present(*y), Atom::Turn => false }) { return None; }
        if lower(&value(term, &r), held) { Some(Verdict::Win) } else { None }
    };
    let t = toward_memo(g, s, side, &oracle, history, 2 * d, None, b);          // PLAN 79.18: the same answers as `toward_with`, a position reached by another order of moves read once
    let best = t.of_class(Class::Win).into_iter().next();
    let found = best.map(|i| Lowering { first: i, moves: (t.depth + 1) / 2, game_end: t.root.iter().find(|r| r.0 == i).is_some_and(|r| crate::toward::is_game_win(r.2)), nodes: t.nodes });
    (found, t.cut)
}

/// PLAN 79.18 B (SYSTEM_SPEC 7, the measure of progress; piece: the measure): what the question of the held goal found. A FALL goes below the lowest value held and is the subgoal reached; a RECOVERY goes only below the value now (the term had risen above the lowest) and is not a subgoal reached.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Found { Fall(Lowering), Recovery(Lowering) }

impl Found {
    pub fn lowering(&self) -> &Lowering { match self { Found::Fall(l) | Found::Recovery(l) => l }
    }
    pub fn is_fall(&self) -> bool { matches!(self, Found::Fall(_)) }
}

/// PLAN 79.18 B: THE QUESTION OF THE HELD GOAL, counted against what was held. `lowest` is the lowest value held so far and `now` the value in the position (never below `lowest`: a value below it is registered as an event before the question is asked). The first reading asks for a line of at most `d` of our moves after which
/// the term is below `lowest` against every reply; only where that was read to its end and found nothing, and `now` is above `lowest`, a second asks for a line below `now`. A cut (the budget) is a gap: it is returned as `cut` and nothing is said of what was not read. The history keys are the game's record: a position met before is a draw for the search, so it is never the end of a line that counts.
#[allow(clippy::too_many_arguments)]
pub fn question<G: Game + Placed>(g: &G, geo: &Geometry, s: &G::State, side: Role, term: &Term, lowest: &[i64], now: &[i64], d: u32, history: &cs_rep::history::Seen, b: &mut Budget) -> (Option<Found>, bool) {
    let (fall, cut) = lowering(g, geo, s, side, term, lowest, d, history, b);
    if let Some(l) = fall { return (Some(Found::Fall(l)), cut); }
    if cut || !lower(lowest, now) { return (None, cut); }
    let (rec, cut) = lowering(g, geo, s, side, term, now, d, history, b);
    (rec.map(Found::Recovery), cut)
}
