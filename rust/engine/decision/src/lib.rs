//! PLAN 78.2 THE DECISION LAYER: one move within the clock, from the `Game` trait alone. The anytime loop, each step traced: (1) a PROVED goal -- tactics
//! `forcing` for us (the first move of a forced line is played); (2) DON'T LOSE -- every candidate after which the other side forces its goal within the danger
//! depth is dropped (the outcome first: don't lose, then win); (3) the PLANNER's move among what is left (planning `keep`); with nothing left, every move is a
//! candidate again (each loses: the planner picks the one that loses least by its measure). The clock is split: a share for our proof, the rest for the danger
//! reads (named shares, not tuned numbers).
use cs_game::Game;
use cs_planning as planning;
use cs_tactics::{self as tactics, Answer, Budget};
use std::collections::HashMap;
use std::time::Instant;

pub const PROOF_SHARE: f64 = 0.5;          // the share of the clock our own proof may take
pub const MAX_DEPTH: u32 = 8;              // the deepest forced line asked (a named fuse: the clock is the real bound)
pub const DANGER_DEPTH: u32 = 2;           // their forced goal looked for after each candidate, within this many of their moves
pub const NODE_FUSE: u64 = 50_000_000;

pub struct Choice<M> {
    pub mv: M,
    pub proved: bool,
    pub trace: Vec<String>,
    pub nodes: u64,
}

/// The move for the side to move at `s` within `seconds` (None at a terminal state).
pub fn decide<G: Game>(g: &G, s: &G::State, seconds: f64, worth: &HashMap<String, i64>) -> Option<Choice<G::Move>> {
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
    let mut nodes = b.nodes;
    // (2) don't lose: drop the candidates after which they force their goal
    let left = (seconds - t0.elapsed().as_secs_f64()).max(0.0);
    let mut db = Budget::new(left * 0.8, NODE_FUSE);
    let (mut safe, mut lost, mut unread) = (vec![], vec![], 0usize);
    for m in &moves {
        let q = g.apply(s, m);
        let bad = match g.outcome(&q) {
            Some(v) => !tactics::won(&v, me) && v[me] < *v.iter().max().unwrap_or(&0),
            None => match g.mover(&q) {
                Some(them) if them != me => match tactics::wins_in(g, &q, them, DANGER_DEPTH, &mut db) {
                    Ok(Some(_)) => true,
                    Ok(None) => false,
                    Err(_) => { unread += 1; false }          // (cut: a GAP -- the move stays a candidate, unread)
                },
                _ => false,
            },
        };
        if bad { lost.push(g.move_name(s, m)); } else { safe.push(m.clone()); }
    }
    nodes += db.nodes;
    trace.push(format!("don't lose: {} candidate(s) keep us from their forced goal within {DANGER_DEPTH}; dropped {}{}{}", safe.len(), lost.len(),
                       if lost.is_empty() { String::new() } else { format!(" ({})", lost.iter().take(6).cloned().collect::<Vec<_>>().join(", ")) },
                       if unread > 0 { format!("; {unread} unread (GAP)") } else { String::new() }));
    let cands = if safe.is_empty() { trace.push("don't lose: every move loses -- all are candidates again".into()); moves.clone() } else { safe };
    // (3) the planner
    let (i, (v, n, worst)) = planning::keep(g, s, me, &cands, worth)?;
    let mv = cands[i].clone();
    trace.push(format!("planning keep: {} -- its worst reply{} leaves our measure {v} ({n} replies) [HEURISTIC]", g.move_name(s, &mv),
                       if worst.is_empty() { String::new() } else { format!(" {worst}") }));
    Some(Choice { mv, proved: false, trace, nodes })
}
