//! PLAN 78.85 PART ONE: the audit of the equal reading (SYSTEM_SPEC 5, the tactical reasoner decides; SYSTEM_SPEC 12, the ledger search). Game-free: the game is read through the `Game` trait alone.
//!
//! INTERFACE. `audit(g, judge, s, history, run_nodes, tie_nodes, max_depth, big)` -> `Audit`. It takes a position, the keys of the positions already met (the rules' draw), a node budget for the
//! deepening (no clock, so the result repeats) and one for the equal reading, and returns for the depth D the deepening finished and its value v: for every root move its answer to "at least v at
//! depth D?" in six readings. They differ in one thing each, so a difference names a cause:
//!   warm probe       the null window on the table the deepening left (what `Search::ties` asks, the engine's reading);
//!   cold probe       the same null window on a fresh table;
//!   no-null probe    fresh table, the null move off; no-late probe: fresh table, the late-move reduction off;
//!   plain probe      fresh table, both off: the question answered by alpha-beta and the table alone (the reference);
//!   exact            the full window on a fresh table, pruning on.
//! Called by: rust/engine/py/src/audit_py.rs (the measure's entry) and the tests of this crate.
use crate::ledger::{Judge, Search};
use cs_game::Game;
use std::collections::HashMap;

/// One root move: its index in the game's list of legal moves, its gain, and the readings (see the module doc).
#[derive(Debug, Clone)]
pub struct Row { pub index: usize, pub gain: i64, pub warm_probe: i64, pub cold_probe: Option<i64>, pub no_null_probe: Option<i64>, pub no_late_probe: Option<i64>, pub plain_probe: Option<i64>, pub exact: Option<i64> }

#[derive(Debug, Clone)]
pub struct Audit { pub depth: i32, pub value: i64, pub run_nodes: u64, pub tie_nodes: u64, pub tie_cut: bool, pub plain_nodes: u64, pub first: Vec<usize>, pub verified: Vec<usize>, pub verify_nodes: u64, pub verify_cut: bool, pub verified_late: Vec<usize>, pub verify_late_nodes: u64, pub rows: Vec<Row> }

fn fresh<'a, G: Game, J: Judge>(g: &'a G, judge: &'a J, me: cs_game::Role, history: &[u64], fuse: u64, null_move: bool, late_reduce: bool) -> Search<'a, G, J> {
    let mut x = Search::with_fuse(g, judge, me, fuse);
    x.history = history.iter().copied().collect();
    x.null_move = null_move;
    x.late_reduce = late_reduce;
    x
}

fn probes_of<G: Game, J: Judge>(x: &mut Search<G, J>, s: &G::State, d: i32, v: i64) -> HashMap<usize, i64> { x.probes(s, d, v, None).into_iter().collect() }

#[allow(clippy::too_many_arguments)]
pub fn audit<G: Game, J: Judge>(g: &G, judge: &J, s: &G::State, history: &[u64], run_nodes: u64, tie_nodes: u64, max_depth: i32, big: u64) -> Option<Audit> {
    let me = g.mover(s)?;
    let mut a = fresh(g, judge, me, history, run_nodes, true, true);
    let ln = a.run(s, max_depth);
    if ln.depth == 0 { return None; }
    let (d, v) = (ln.depth, ln.value);
    let run_used = a.nodes;
    a.rebudget(tie_nodes);
    let n0 = a.nodes;
    let warm = a.probes(s, d, v, None);
    let tie_used = a.nodes - n0;
    let tie_cut = a.cut();
    let first: Vec<usize> = warm.iter().filter(|(_, p)| *p >= v).map(|(i, _)| *i).collect();
    let n1 = a.nodes;
    let verified = a.verify(s, d, v, &first);                                   // the second reading of `ties` (fresh table, late-move reduction and null move off)
    let verify_nodes = a.nodes - n1;
    let verify_cut = a.cut();
    let mut vb = fresh(g, judge, me, history, big, true, false);
    let verified_late: Vec<usize> = vb.probes(s, d, v, Some(&first)).into_iter().filter(|(_, p)| *p >= v).map(|(i, _)| i).collect();   // the same with the late-move reduction off and the null move on (the first, lighter variant)
    let verify_late_nodes = vb.nodes;
    let cold_p = probes_of(&mut fresh(g, judge, me, history, big, true, true), s, d, v);
    let nn_p = probes_of(&mut fresh(g, judge, me, history, big, false, true), s, d, v);
    let nl_p = probes_of(&mut fresh(g, judge, me, history, big, true, false), s, d, v);
    let mut pl = fresh(g, judge, me, history, big, false, false);
    let pl_p = probes_of(&mut pl, s, d, v);
    let plain_nodes = pl.nodes;
    let ex_v: HashMap<usize, i64> = fresh(g, judge, me, history, big, true, true).root_values(s, d).into_iter().collect();
    let ms = g.moves(s);
    let rows = warm.iter().map(|(i, p)| Row { index: *i, gain: ms.get(*i).map_or(0, |m| a.gain(s, m)), warm_probe: *p, cold_probe: cold_p.get(i).copied(), no_null_probe: nn_p.get(i).copied(), no_late_probe: nl_p.get(i).copied(), plain_probe: pl_p.get(i).copied(), exact: ex_v.get(i).copied() }).collect();
    Some(Audit { depth: d, value: v, run_nodes: run_used, tie_nodes: tie_used, tie_cut, plain_nodes, first, verified, verify_nodes, verify_cut, verified_late, verify_late_nodes, rows })
}
