//! PLAN 79.03: the instrument of the gap between the deepening's value and the best root move's own value at the same depth (SYSTEM_SPEC 4, the tactical reasoner's deepening and its prunings; SYSTEM_SPEC 12, the debt of plan 78.84). Game-free: the game is read through the `Game` trait alone.
//! A MEASURE: it changes no decision and no default. The same readings as `equal_audit` gives for `exact` and for the deepening's value, without its six probes (it costs two searches, not nine), and with the switch `root_full` as a parameter so that the mend can be read apart.
//!
//! INTERFACE. `gap_audit(g, judge, s, history, run_nodes, max_depth, big, root_full)` -> `Option<Gap>`. Takes a position, the keys of the positions already met (the rules' draw), a node budget for the deepening (no clock: the result repeats), the deepest depth,
//! a node fuse for the exact readings and the switch. Returns the depth D the deepening finished, its value v, the move it chose, the nodes it spent, and for every root move its gain and its `exact` value: the value of the move at D read with the FULL window on a fresh table
//! (the prunings on below the root, no reduction at the root), None where the fuse cut it. Called by: rust/engine/py/src/audit_py.rs (the measure's entry) and the tests of this crate.
use crate::ledger::{Judge, Search};
use cs_game::Game;

#[derive(Debug, Clone)]
pub struct GapRow { pub index: usize, pub gain: i64, pub exact: Option<i64> }

#[derive(Debug, Clone)]
pub struct Gap { pub depth: i32, pub value: i64, pub first: Option<usize>, pub run_nodes: u64, pub exact_cut: bool, pub rows: Vec<GapRow> }

#[allow(clippy::too_many_arguments)]
pub fn gap_audit<G: Game, J: Judge>(g: &G, judge: &J, s: &G::State, history: &[u64], run_nodes: u64, max_depth: i32, big: u64, root_full: bool) -> Option<Gap> {
    let me = g.mover(s)?;
    let mut a = Search::with_fuse(g, judge, me, run_nodes);
    a.history = cs_rep::history::Seen::of(&history);
    a.root_full = root_full;
    let ln = a.run(s, max_depth);
    if ln.depth == 0 { return None; }
    let mut x = Search::with_fuse(g, judge, me, big);
    x.history = cs_rep::history::Seen::of(&history);
    let ex: std::collections::HashMap<usize, i64> = x.root_values(s, ln.depth).into_iter().collect();
    let exact_cut = x.cut();
    let ms = g.moves(s);
    let rows = (0..ms.len()).map(|i| GapRow { index: i, gain: a.gain(s, &ms[i]), exact: ex.get(&i).copied() }).collect();
    Some(Gap { depth: ln.depth, value: ln.value, first: ln.first_index, run_nodes: ln.nodes, exact_cut, rows })
}
