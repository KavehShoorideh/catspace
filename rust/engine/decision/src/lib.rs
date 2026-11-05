//! PLAN 78.2 / 78.5 THE DECISION LAYER: one move within the clock, from the `Game` trait alone. The anytime loop, each step traced: (1) a PROVED goal -- tactics
//! `forcing` for us (the first move of a forced line of the game's own outcome is played); (2) THE LEDGER SEARCH (tactics `ledger`, plan 78.5): the move whose line
//! loses least of ours and wins most of theirs, by the judge -- the variable: no values by default (a unit lost is a unit lost, ours first), a valuation when one is
//! plugged in; the minimax reads the game's outcome too (a line into their forced goal scores below every ledger: don't lose). Planning `keep` (78.2) stays an
//! approach of its own, no longer the default. The clock is split: a share for the proof, the rest for the ledger search (named shares, not tuned numbers).
use cs_game::Game;
use cs_tactics::{self as tactics, Answer, Budget};
use std::time::Instant;

pub const PROOF_SHARE: f64 = 0.2;          // the share of the clock the proof of the game's own outcome may take (the ledger search reads the outcome too)
pub const MAX_DEPTH: u32 = 8;              // the deepest forced line asked (a named fuse: the clock is the real bound)
pub const DANGER_DEPTH: u32 = 2;           // their forced goal looked for after each candidate, within this many of their moves
pub const NODE_FUSE: u64 = 50_000_000;

pub struct Choice<M> {
    pub mv: M,
    pub proved: bool,
    pub trace: Vec<String>,
    pub nodes: u64,
}

pub const LEDGER_DEPTH: i32 = 64;
pub const LEDGER_SHARE: f64 = 0.6;
pub const TIE_SHARE: f64 = 0.3;            // of what is left: the share the tie reading may take (the rest is the prior's census)         // with the density prior on: the share of the clock the ledger search may take (the prior reads ties with the rest)          // the ledger search's deepest iteration (a named fuse: the clock is the real bound)

/// The move for the side to move at `s` within `seconds` (None at a terminal state). `judge`: the ledger's judge (the variable: `Count` -- no values -- or `Worth`).
pub fn decide<G: Game, J: tactics::ledger::Judge>(g: &G, s: &G::State, seconds: f64, judge: &J, prior: bool) -> Option<Choice<G::Move>> {
    let me = g.mover(s)?;
    let t0 = Instant::now();
    let moves = g.moves(s);
    if moves.is_empty() { return None; }
    let mut trace = vec![format!("decide: {} to move, {} legal moves, {:.2} s", g.roles()[me], moves.len(), seconds)];
    // (1) our goal, proved
    let mut b = Budget::new(seconds * PROOF_SHARE, NODE_FUSE);
    match tactics::force(g, s, me, MAX_DEPTH, &mut b) {
        Answer::Proved { first, depth } => {
            trace.push(format!("tactics forcing: PROVED -- {} forces the goal within {depth} move(s) against every reply ({} nodes)", g.move_name(s, &first), b.nodes));
            return Some(Choice { mv: first, proved: true, trace, nodes: b.nodes });
        }
        Answer::None { depth } => trace.push(format!("tactics forcing: no forced goal within {depth} move(s) ({} nodes)", b.nodes)),
        Answer::Gap { depth } => trace.push(format!("tactics forcing: GAP at depth {depth} -- the clock cut the reading ({} nodes)", b.nodes)),
    }
    // (2) THE LEDGER SEARCH (plan 78.5): the move whose line loses least of ours and wins most of theirs by the judge (the variable: no values by default)
    let left = (seconds * if prior { LEDGER_SHARE } else { 1.0 } - t0.elapsed().as_secs_f64()).max(0.01);
    let mut ls = tactics::ledger::Search::new(g, judge, me, left);
    let ln = ls.run(s, LEDGER_DEPTH);
    let nodes = b.nodes + ln.nodes;
    let Some(i) = ln.first_index else {
        let mv = moves[0].clone();
        trace.push("ledger search: no line read within the clock -- the first legal move".into());
        return Some(Choice { mv, proved: false, trace, nodes });
    };
    let mut i = i;
    let mut prior_line = None;
    if prior && ln.value.abs() < tactics::ledger::WIN - 1000 {
        // (3) THE DENSITY PRIOR (plan 78.7): among the moves as good as the ledger's, the one after which our tactics most outnumber theirs
        let rest = (seconds - t0.elapsed().as_secs_f64()).max(0.0);
        ls.resume(rest * TIE_SHARE);
        let ties = ls.ties(s, ln.depth, ln.value);
        if ties.len() > 1 {
            let until = t0 + std::time::Duration::from_secs_f64(seconds);
            if let Some((j, (a0, a1), read)) = cs_planning::density::prior(g, s, me, &ties, judge, until) {
                prior_line = Some(format!("planning density: {} of {} equal moves read; {} -- our tactics less theirs after it: {a0:+} at 0 defects, {a1:+} at 1 [PRIOR]",
                                          read, ties.len(), g.move_name(s, &g.moves(s)[j])));
                i = j;
            }
        }
    }
    let mv = g.moves(s)[i].clone();
    let who = |r: cs_game::Role| if r == me { "we" } else { "they" };
    let lost: Vec<String> = ln.lost.iter().map(|(o, k)| format!("{} {k}", who(*o))).collect();
    trace.push(format!("ledger search [{}]: {} at depth {} ({} nodes): the line {} -- {}; value {}", judge.name(), g.move_name(s, &g.moves(s)[ln.first_index.unwrap_or(0)]),
                       ln.depth, ln.nodes, ln.line.join(" "), if lost.is_empty() { "nobody loses a unit".to_string() } else { lost.join(", ") }, ln.value));
    if let Some(p) = prior_line { trace.push(p); }
    Some(Choice { mv, proved: false, trace, nodes })
}
