//! PLAN 79.08 B, THE REASONS' PICK (SYSTEM_SPEC 8 step 6 "the chances decide among what was reached", 7 estimates; piece: the wake). The Rust form of symbols_play.ReasonsReader.pick (mode "reasons") over symbols.deltas_of: for each move, the REASONS it carries are the fact patterns whose count the move changes
//! (the mover's count goes up or down, the other side's goes up or down); the score of a move is the sum of the weights of its reasons, added in the reasons' own order. The pattern counter is cs_tactics::sheet::patterns (Rust, as the Python's `model.pattern_count`); the keys and WEIGHTS ARE A PARAMETER (the fit of
//! plans 78.81 and 78.82 is read by the caller from its record: no new fit here, no weight made up). The score is an ESTIMATE of how the move serves a purpose and it only CHOOSES among the moves the search holds equal: it never enters a value (rule 7). Game-free: a pattern is a number key, a move a name.
//!
//! INTERFACE (written before the bodies).
//!   `Reasons { keys, reasons, max_size, levels, tie_digits }`   the pattern keys the weights are over (in the order the columns were laid out) and the reasons: (column, side, direction, weight) in the fit's own order.
//!   `Side::{Mover, Other}`, `Dir::{Up, Down}`                    the events of a move.
//!   `scores(st, g, s, goal, &Reasons, moves) -> Vec<MoveReasons>` for each named legal move: the reasons it carries (indices into `reasons`) and the score. One count of the position and one of each position after a move.
//!   `pick(scores, tied) -> Option<String>`                       among `tied` (the equal moves, the search's first first): the move with the top score, the first of the tied if it is among them, the first of the top otherwise; None when fewer than two moves are tied.
//!   Called by: the wake's loop (`wake::decide`, step 6) and the unit tests.
use cs_game::placed::{Kind, Placed};
use cs_game::Game;
use cs_tactics::sheet::base_graph::Static;
use cs_tactics::sheet::patterns::{self, Board, FxHashMap};
use std::collections::HashSet;

/// Whose count the event is about, and which way it went.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Side { Mover, Other }
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Dir { Up, Down }

/// One reason: a pattern column, whose count, which way, and its weight in the fit.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Reason { pub column: usize, pub side: Side, pub dir: Dir, pub weight: f64 }

/// The reasons and what they are over. `keys`: the pattern keys, column i is keys[i]; `reasons` in the fit's own order (the sums are added in it); `max_size`, `levels`: the seeds of the count; `tie_digits`: two scores that agree to this many places are equal.
#[derive(Debug, Clone, Default)]
pub struct Reasons { pub keys: Vec<u64>, pub reasons: Vec<Reason>, pub max_size: usize, pub levels: Vec<u8>, pub tie_digits: i32 }

/// The reasons one move carries and its score.
#[derive(Debug, Clone, PartialEq)]
pub struct MoveReasons { pub mv: String, pub carried: Vec<usize>, pub score: f64 }

/// The counts (side to move, other side) of every pattern column on the board of `s`.
fn counts<G: Game + Placed>(st: &Static, g: &G, s: &G::State, goal: Option<Kind>, r: &Reasons, want: &FxHashMap<u64, usize>) -> FxHashMap<usize, (u32, u32)> {
    let part = st.position(g, s);
    let b = Board::new(st, &part, goal);
    let mut m: FxHashMap<usize, (u32, u32)> = FxHashMap::default();
    patterns::count_board(&b, r.max_size, &r.levels, want, &mut m);
    m
}

/// The events of a move (symbols.deltas_matrix): the mover's count after (the side to move there is the other side, so its `them`) less the mover's before (`us`), and the other side's after (`us`) less its before (`them`).
fn events(before: &FxHashMap<usize, (u32, u32)>, after: &FxHashMap<usize, (u32, u32)>, cols: usize) -> HashSet<(usize, Side, Dir)> {
    let mut out = HashSet::new();
    for c in 0..cols {
        let (ub, tb) = before.get(&c).copied().unwrap_or((0, 0));
        let (ua, ta) = after.get(&c).copied().unwrap_or((0, 0));
        let (mover, other) = (ta as i64 - ub as i64, ua as i64 - tb as i64);
        if mover > 0 { out.insert((c, Side::Mover, Dir::Up)); }
        if mover < 0 { out.insert((c, Side::Mover, Dir::Down)); }
        if other > 0 { out.insert((c, Side::Other, Dir::Up)); }
        if other < 0 { out.insert((c, Side::Other, Dir::Down)); }
    }
    out
}

/// THE REASONS OF EACH NAMED MOVE and its score (the sum of the weights of the reasons it carries, in the reasons' own order).
pub fn scores<G: Game + Placed>(st: &Static, g: &G, s: &G::State, goal: Option<Kind>, r: &Reasons, moves: &[String]) -> Vec<MoveReasons> {
    let want: FxHashMap<u64, usize> = r.keys.iter().enumerate().map(|(i, k)| (*k, i)).collect();
    let before = counts(st, g, s, goal, r, &want);
    moves.iter().map(|name| {
        let Some(m) = g.find_move(s, name) else { return MoveReasons { mv: name.clone(), carried: vec![], score: 0.0 } };
        let a = g.apply(s, &m);
        let after = counts(st, g, &a, goal, r, &want);
        let ev = events(&before, &after, r.keys.len());
        let carried: Vec<usize> = r.reasons.iter().enumerate().filter(|(_, x)| ev.contains(&(x.column, x.side, x.dir))).map(|(i, _)| i).collect();
        let score = carried.iter().map(|i| r.reasons[*i].weight).sum();
        MoveReasons { mv: name.clone(), carried, score }
    }).collect()
}

/// THE PICK among the tied moves (symbols_play.ReasonsReader.pick, mode "reasons"): the top rounded score; the first of the tied when it is among the top, else the first of the top. None with fewer than two tied.
pub fn pick(sc: &[MoveReasons], tied: &[String], tie_digits: i32) -> Option<String> {
    if tied.len() < 2 { return None; }
    let k = 10f64.powi(tie_digits);
    let round = |x: f64| (x * k).round() / k;
    let of = |m: &String| sc.iter().find(|x| &x.mv == m).map_or(0.0, |x| round(x.score));
    let top = tied.iter().map(of).fold(f64::NEG_INFINITY, f64::max);
    let best: Vec<&String> = tied.iter().filter(|m| of(m) == top).collect();
    Some(if best.contains(&&tied[0]) { tied[0].clone() } else { best[0].clone() })
}
