//! PLAN 78.2 THE PLANNING LAYER: "where should we go when nothing is forced?" -- from the `Game` trait alone. Approach `keep` (the thin first one, a HEURISTIC,
//! labelled so): the search law (Kaveh 2026-10-27) at its shallowest -- among the candidate moves, the one whose WORST reply leaves our measure highest; the measure
//! is our units less theirs by the store's valuation (a table from kind names to worths: the game reports its units, the layer never names one); the other side's
//! fewest replies break a tie (it keeps them busy); then the move's name (a stable order). Subgoals (goal regression, the trajectory model from game data) are the
//! approaches that come after this one.
pub mod operators;
pub mod mcts;

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

/// PLAN 78.7 THE DENSITY PRIOR (Kaveh 2026-11-04: "use ... tactic densities ... as a policy prior to say we want to navigate to areas of more tactics"): among moves the
/// ledger search reads as equally good, the one after which OUR tactics most outnumber THEIRS -- the census (cs_tactics::census) after the move: ours read as if they
/// passed, theirs as they stand; compared first at 0 defects, then at 1 (the order the human-games study found: the 0-defect advantage carries the outcome; the 1-defect
/// one turns into it). A prior over ties, never a verdict: what the ledger reads stays decided by the ledger. Read while its clock lasts; an unread move ranks last.
pub mod density {
    use cs_game::{Game, Role};
    use cs_tactics::census::census;
    use cs_tactics::ledger::Judge;
    use std::time::Instant;

    pub const DEPTH: i32 = 2;
    pub const FUSE: u64 = 50_000;

    /// One move's density reading: (ours less theirs at 0 defects, at 1 defect), and the nodes the census readings spent.
    pub fn advantage<G: Game, J: Judge>(g: &G, s: &G::State, m: &G::Move, me: Role, judge: &J, depth: i32, fuse: u64, defects: u32) -> Option<((i64, i64), u64)> {
        let q = g.apply(s, m);
        if g.outcome(&q).is_some() { return Some(((0, 0), 0)); }
        let mut nodes = 0u64;
        let mut count = |st: &G::State| -> [i64; 2] {
            let r = census(g, st, judge, defects, depth, fuse);
            nodes += r.nodes;
            let mut c = [0i64; 2];
            for t in r.tactics { if (t.defects as usize) < 2 { c[t.defects as usize] += 1; } }
            c
        };
        let theirs = count(&q);
        let ours = match g.pass(&q) { Some(p) if g.mover(&p) == Some(me) => count(&p), _ => [0, 0] };
        Some(((ours[0] - theirs[0], ours[1] - theirs[1]), nodes))
    }

    /// THE PRIOR's answer: the chosen index, its reading, how many were read, the nodes spent, and what each candidate read cost (index, nodes, seconds).
    #[derive(Debug, Clone)]
    pub struct Prior { pub index: usize, pub advantage: (i64, i64), pub read: usize, pub nodes: u64, pub reads: Vec<(usize, u64, f64)> }

    /// THE PRIOR among `cands` (indices into `g.moves(s)`), until `deadline`.
    pub fn prior<G: Game, J: Judge>(g: &G, s: &G::State, me: Role, cands: &[usize], judge: &J, deadline: Instant, depth: i32, fuse: u64, defects: u32) -> Option<Prior> {
        let ms = g.moves(s);
        let mut best: Option<(usize, (i64, i64))> = None;
        let mut reads = vec![];
        let mut nodes = 0u64;
        for &i in cands {
            if Instant::now() >= deadline && best.is_some() { break; }
            let t0 = Instant::now();
            let Some((a, n)) = advantage(g, s, &ms[i], me, judge, depth, fuse, defects) else { continue };
            reads.push((i, n, t0.elapsed().as_secs_f64()));
            nodes += n;
            if best.as_ref().map_or(true, |(_, b)| a > *b) { best = Some((i, a)); }
        }
        best.map(|(i, a)| Prior { index: i, advantage: a, read: reads.len(), nodes, reads })
    }
}
