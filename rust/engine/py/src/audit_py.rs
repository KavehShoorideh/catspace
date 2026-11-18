//! PLAN 78.85: the Python entry of the equal reading's audit (cs_tactics::equal_audit). `equal_audit(variant, moves, worth, run_nodes, tie_nodes, max_depth=64, big=20000000, kind_change=True)` -> a dict
//! {depth, value, run_nodes, tie_nodes, tie_cut, rows: [{index, name, gain, warm_probe, cold_probe, no_null_probe, no_late_probe, plain_probe, exact}]} or None. No clock: the budgets are node counts, so the result repeats.
use cs_game::Game;
use cs_tactics::equal_audit as EA;
use cs_tactics::ledger::{NoChange, Worth};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::collections::HashMap;

#[pyfunction]
#[pyo3(signature = (variant, moves, worth, run_nodes, tie_nodes, max_depth=64, big=20_000_000, kind_change=true))]
#[allow(clippy::too_many_arguments)]
pub fn equal_audit(py: Python<'_>, variant: &str, moves: Vec<String>, worth: HashMap<String, i64>, run_nodes: u64, tie_nodes: u64, max_depth: i32, big: u64, kind_change: bool) -> PyResult<PyObject> {
    let (g, s, keys) = crate::record_of(variant, None, 518, &moves)?;
    let judge = Worth { name: "record".into(), table: worth, ours_first: false };
    let r = if kind_change { EA::audit(&g, &judge, &s, &keys, run_nodes, tie_nodes, max_depth, big) } else { EA::audit(&g, &NoChange(judge), &s, &keys, run_nodes, tie_nodes, max_depth, big) };
    let Some(a) = r else { return Ok(py.None()) };
    let d = PyDict::new(py);
    d.set_item("depth", a.depth)?;
    d.set_item("value", a.value)?;
    d.set_item("run_nodes", a.run_nodes)?;
    d.set_item("tie_nodes", a.tie_nodes)?;
    d.set_item("tie_cut", a.tie_cut)?;
    d.set_item("plain_nodes", a.plain_nodes)?;
    d.set_item("first", a.first.clone())?;
    d.set_item("verified", a.verified.clone())?;
    d.set_item("verify_nodes", a.verify_nodes)?;
    d.set_item("verify_cut", a.verify_cut)?;
    d.set_item("verified_late", a.verified_late.clone())?;
    d.set_item("verify_late_nodes", a.verify_late_nodes)?;
    let ms = g.moves(&s);
    let rows = PyList::empty(py);
    for r in &a.rows {
        let e = PyDict::new(py);
        e.set_item("index", r.index)?;
        e.set_item("name", ms.get(r.index).map(|m| g.move_name(&s, m)))?;
        e.set_item("gain", r.gain)?;
        e.set_item("warm_probe", r.warm_probe)?;
        e.set_item("cold_probe", r.cold_probe)?;
        e.set_item("exact", r.exact)?;
        e.set_item("no_null_probe", r.no_null_probe)?;
        e.set_item("no_late_probe", r.no_late_probe)?;
        e.set_item("plain_probe", r.plain_probe)?;
        rows.append(e)?;
    }
    d.set_item("rows", rows)?;
    Ok(d.into_any().unbind())
}

/// PLAN 78.85, the trace: `value_at(variant, moves, root_plies, worth, depth, ply, alpha, beta, null_move=True, late_reduce=True, fuse=20000000)` -> (the value of the position after `moves` read as a node at `ply`
/// of a search whose side is the mover of the position after the first `root_plies` moves, the nodes spent, the names of the legal moves there). A fresh table; the record of positions is the moves before the root.
#[pyfunction]
#[pyo3(signature = (variant, moves, root_plies, worth, depth, ply, alpha, beta, null_move=true, late_reduce=true, fuse=20_000_000))]
#[allow(clippy::too_many_arguments)]
pub fn value_at(variant: &str, moves: Vec<String>, root_plies: usize, worth: HashMap<String, i64>, depth: i32, ply: i32, alpha: i64, beta: i64, null_move: bool, late_reduce: bool, fuse: u64) -> PyResult<(i64, u64, Vec<String>)> {
    let (g, s, keys) = crate::record_of(variant, None, 518, &moves)?;
    let k = root_plies.min(moves.len());
    let (_, root, _) = crate::record_of(variant, None, 518, &moves[..k])?;
    let me = g.mover(&root).ok_or_else(|| pyo3::exceptions::PyValueError::new_err("the root is over"))?;
    let judge = Worth { name: "record".into(), table: worth, ours_first: false };
    let mut x = cs_tactics::ledger::Search::with_fuse(&g, &judge, me, fuse);
    x.history = keys[..k].iter().copied().collect();
    x.null_move = null_move;
    x.late_reduce = late_reduce;
    let v = x.value_at(&s, depth, ply, alpha, beta);
    let names = g.moves(&s).iter().map(|m| g.move_name(&s, m)).collect();
    Ok((v, x.nodes, names))
}

/// PLAN 78.92: the Python entry of the choice audit (cs_tactics::choice_audit). `choice_audit(variant, moves, worth, run_nodes, max_depth=64, big=4000000, extra=2, kind_change=True, other=None)` -> a dict
/// {depth, value, played, played_name, run_nodes, verify_probe, verify_nodes, readings: [{name, depth, best, best_index, best_name, played, other, nodes}]} or None. No clock: the budgets are node counts.
#[pyfunction]
#[pyo3(signature = (variant, moves, worth, run_nodes, max_depth=64, big=4_000_000, extra=2, kind_change=true, other=None, repair=false))]
#[allow(clippy::too_many_arguments)]
pub fn choice_audit(py: Python<'_>, variant: &str, moves: Vec<String>, worth: HashMap<String, i64>, run_nodes: u64, max_depth: i32, big: u64, extra: i32, kind_change: bool, other: Option<String>, repair: bool) -> PyResult<PyObject> {
    let (g, s, keys) = crate::record_of(variant, None, 518, &moves)?;
    let judge = Worth { name: "record".into(), table: worth, ours_first: false };
    let r = if kind_change { cs_tactics::choice_audit::choice_audit(&g, &judge, &s, &keys, run_nodes, max_depth, big, extra, other.as_deref(), repair) } else { cs_tactics::choice_audit::choice_audit(&g, &NoChange(judge), &s, &keys, run_nodes, max_depth, big, extra, other.as_deref(), repair) };
    let Some(a) = r else { return Ok(py.None()) };
    let d = PyDict::new(py);
    d.set_item("depth", a.depth)?;
    d.set_item("value", a.value)?;
    d.set_item("played", a.played)?;
    d.set_item("played_name", a.played_name.clone())?;
    d.set_item("run_nodes", a.run_nodes)?;
    d.set_item("verify_probe", a.verify_probe)?;
    d.set_item("verify_nodes", a.verify_nodes)?;
    d.set_item("repair_state", a.repair_state.label())?;
    d.set_item("repair_nodes", a.repair_nodes)?;
    let rows = PyList::empty(py);
    for r in &a.readings {
        let e = PyDict::new(py);
        e.set_item("name", r.name)?;
        e.set_item("depth", r.depth)?;
        e.set_item("best", r.best)?;
        e.set_item("best_index", r.best_index)?;
        e.set_item("best_name", r.best_name.clone())?;
        e.set_item("played", r.played)?;
        e.set_item("other", r.other)?;
        e.set_item("nodes", r.nodes)?;
        rows.append(e)?;
    }
    d.set_item("readings", rows)?;
    Ok(d.into_any().unbind())
}
