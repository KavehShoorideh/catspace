//! PLAN 78.2 THE PLANNING LAYER: "where should we go when nothing is forced?" -- from the `Game` trait alone. Approach `keep` (the thin first one, a HEURISTIC,
//! labelled so): the search law (Kaveh 2026-10-27) at its shallowest -- among the candidate moves, the one whose WORST reply leaves our measure highest; the measure
//! is our units less theirs by the store's valuation (a table from kind names to worths: the game reports its units, the layer never names one); the other side's
//! fewest replies break a tie (it keeps them busy); then the move's name (a stable order). Subgoals (goal regression, the trajectory model from game data) are the
//! approaches that come after this one.
use cs_game::{Game, Role};
use std::collections::HashMap;

/// Our units less theirs by the valuation (a kind the table does not name counts 0).
pub fn measure<G: Game>(g: &G, s: &G::State, me: Role, worth: &HashMap<String, i64>) -> i64 {
    g.units(s).iter().map(|(owner, kind)| { let w = *worth.get(kind).unwrap_or(&0); if *owner == me { w } else { -w } }).sum()
}

/// The value of a terminal state for `me` on the measure's scale: a win above every measure, a loss below, anything else level.
fn terminal_value(v: &[i64], me: Role) -> i64 {
    let best_other = v.iter().enumerate().filter(|(i, _)| *i != me).map(|(_, x)| *x).max().unwrap_or(0);
    if v[me] > best_other { i64::MAX / 4 } else if v[me] < best_other { i64::MIN / 4 } else { 0 }
}

/// One candidate's reading: (its worst reply's measure, the other side's reply count, the worst reply's name).
pub fn worst_after<G: Game>(g: &G, s: &G::State, m: &G::Move, me: Role, worth: &HashMap<String, i64>) -> (i64, usize, String) {
    let q = g.apply(s, m);
    if let Some(v) = g.outcome(&q) { return (terminal_value(&v, me), 0, String::new()); }
    let replies = g.moves(&q);
    let mut worst = (i64::MAX, String::new());
    for r in &replies {
        let q2 = g.apply(&q, r);
        let v = match g.outcome(&q2) { Some(v) => terminal_value(&v, me), None => measure(g, &q2, me, worth) };
        if v < worst.0 { worst = (v, g.move_name(&q, r)); }
    }
    (worst.0, replies.len(), worst.1)
}

/// THE `keep` APPROACH: the index of the chosen candidate and its reading (worst measure, their replies, the worst reply).
pub fn keep<G: Game>(g: &G, s: &G::State, me: Role, cands: &[G::Move], worth: &HashMap<String, i64>) -> Option<(usize, (i64, usize, String))> {
    let mut best: Option<(usize, (i64, usize, String), String)> = None;
    for (i, m) in cands.iter().enumerate() {
        let r = worst_after(g, s, m, me, worth);
        let name = g.move_name(s, m);
        let better = match &best {
            None => true,
            Some((_, b, bn)) => (r.0 > b.0) || (r.0 == b.0 && (r.1 < b.1 || (r.1 == b.1 && name < *bn))),
        };
        if better { best = Some((i, r, name)); }
    }
    best.map(|(i, r, _)| (i, r))
}
