//! PLAN 78.2 THE TACTICS LAYER: "can this side force its goal from here?" -- answered from the `Game` trait alone (cs_game), so it serves every game the trait
//! holds. Approach `forcing` (the thin first one): the terminal goal by the every-reply clause composed (77.1 (a)/(d)) -- some move of ours after which every
//! legal reply leaves the goal within n - 1 more moves of ours, the outcome read from the game (a win: our goal value above every other role's). Deepened
//! n = 1, 2, ... while the budget lasts; a cut is a GAP, never a refutation. The order of our moves: those that end the game first, then those that leave the other
//! side the fewest replies (the forcing ones) -- an order, never a verdict.
pub mod census;
pub mod choice_audit;
pub mod equal_audit;
pub mod gap_audit;
pub mod ledger;
pub mod lines;
pub mod operators;
pub mod reason;
pub mod scenario;
pub mod sheet;
pub mod toward;

use cs_game::{Game, Role};

pub use cs_operator::{Budget, Cut};

/// A side's goal reached at a terminal state: its goal value above every other role's.
pub fn won(v: &[i64], me: Role) -> bool { v.iter().enumerate().all(|(i, x)| i == me || v[me] > *x) }

/// The answer of `force`.
#[derive(Debug, Clone)]
pub enum Answer<M> {
    /// `me` forces its goal within `depth` of its moves, starting with `first`.
    Proved { first: M, depth: u32 },
    /// No such line within `depth` (every line read: exact at that depth).
    None { depth: u32 },
    /// The budget cut the reading at `depth` (the depths below it read in full: None there).
    Gap { depth: u32 },
}

fn ordered<G: Game>(g: &G, s: &G::State) -> Vec<(G::Move, G::State)> {
    let mut v: Vec<(usize, String, G::Move, G::State)> = g.moves(s).into_iter().map(|m| {
        let q = g.apply(s, &m);
        let k = if g.outcome(&q).is_some() { 0 } else { 1 + g.moves(&q).len() };
        (k, g.move_name(s, &m), m, q)
    }).collect();
    v.sort_by(|a, b| (a.0, &a.1).cmp(&(b.0, &b.1)));
    v.into_iter().map(|(_, _, m, q)| (m, q)).collect()
}

/// `me` to move at `s`: a move after which `me`'s goal binds within `n` of its moves against every reply.
pub fn wins_in<G: Game>(g: &G, s: &G::State, me: Role, n: u32, b: &mut Budget) -> Result<Option<G::Move>, Cut> {
    for (m, q) in ordered(g, s) {
        b.spend()?;
        match g.outcome(&q) {
            Some(v) => { if won(&v, me) { return Ok(Some(m)); } continue; }
            None => {}
        }
        if n <= 1 { continue; }
        let replies = g.moves(&q);
        if replies.is_empty() { continue; }
        let mut all = true;
        for r in replies {
            b.spend()?;
            let q2 = g.apply(&q, &r);
            let ok = match g.outcome(&q2) {
                Some(v) => won(&v, me),
                None => g.mover(&q2) == Some(me) && wins_in(g, &q2, me, n - 1, b)?.is_some(),
            };
            if !ok { all = false; break; }
        }
        if all { return Ok(Some(m)); }
    }
    Ok(None)
}

/// One deepening step of `force`, as measured: the depth asked, the nodes it spent, its seconds.
#[derive(Debug, Clone, Copy)]
pub struct ForceRead { pub depth: u32, pub nodes: u64, pub seconds: f64 }

/// THE `forcing` APPROACH: `me` (to move at `s`) forces its goal? Deepened from one move to `max_depth` while the budget lasts.
pub fn force<G: Game>(g: &G, s: &G::State, me: Role, max_depth: u32, b: &mut Budget) -> Answer<G::Move> {
    force_traced(g, s, me, max_depth, b).0
}

/// `force` with the cost of each deepening step.
pub fn force_traced<G: Game>(g: &G, s: &G::State, me: Role, max_depth: u32, b: &mut Budget) -> (Answer<G::Move>, Vec<ForceRead>) {
    let mut done = 0;
    let mut reads = vec![];
    for n in 1..=max_depth {
        let (n0, t0) = (b.nodes, std::time::Instant::now());
        let r = wins_in(g, s, me, n, b);
        reads.push(ForceRead { depth: n, nodes: b.nodes - n0, seconds: t0.elapsed().as_secs_f64() });
        match r {
            Ok(Some(m)) => return (Answer::Proved { first: m, depth: n }, reads),
            Ok(None) => done = n,
            Err(Cut) => return (Answer::Gap { depth: n }, reads),
        }
        if !b.left() { return (if done == max_depth { Answer::None { depth: done } } else { Answer::Gap { depth: n + 1 } }, reads); }
    }
    (Answer::None { depth: done }, reads)
}
