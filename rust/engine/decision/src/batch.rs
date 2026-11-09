//! PLAN 78.12 THE BATCH SOLVER: many positions in ONE call, spread over a thread pool, one summary per position, every step of every decision appended to a trace log.
//! Game-free: the caller builds each position's game and state inside `solve` (so nothing needs to cross threads but the items), and `row` turns a decision into a
//! summary and a log row. The log is JSON Lines, one row per position with its steps nested (operator, goal, status, depth, value, measured nodes and seconds, the
//! facts found), tagged with the SNAPSHOT id the caller names -- the id of the store state the batch read -- so a row can always be tied to what the engine knew.
//! (A stand-in for the architecture document's columnar files: the schema is flat enough to load as columns, and a conversion to Parquet is a job for the harness.)
use cs_game::Game;
use cs_operator::Decision;
use rayon::prelude::*;
use serde_json::{json, Value};
use std::io::Write;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::Path;

/// One position's result, as the caller reads it.
#[derive(Debug, Clone)]
pub struct Summary {
    pub index: usize,
    pub mv: Option<String>,
    pub proved: bool,
    pub nodes: u64,
    pub seconds: f64,
    pub error: Option<String>,
}

/// A decision as a summary and a log row. `label` names the position for whoever reads the log (an id, a description).
pub fn row<G: Game>(index: usize, snapshot: &str, label: &str, g: &G, s: &G::State, d: &Decision<G::Move>) -> (Summary, Value) {
    let mv = d.mv.as_ref().map(|m| g.move_name(s, m));
    let steps: Vec<Value> = d.steps.iter().map(|st| json!({
        "operator": st.operator, "goal": st.goal.describe(), "status": format!("{:?}", st.outcome.status),
        "depth": st.outcome.depth, "value": st.outcome.value, "candidates": st.outcome.candidates.len(),
        "nodes": st.outcome.cost.nodes, "seconds": st.outcome.cost.seconds,
        "parts": st.outcome.parts.iter().map(|p| json!({"name": p.name, "nodes": p.cost.nodes, "seconds": p.cost.seconds})).collect::<Vec<_>>(),
        "line": st.outcome.line, "events": st.outcome.events.iter().map(|(r, e)| json!([r, e])).collect::<Vec<_>>(),
    })).collect();
    let facts: Vec<Value> = d.steps.iter().flat_map(|st| st.outcome.facts.iter().map(|f| json!({"relation": f.relation, "args": f.args}))).collect();
    let v = json!({
        "kind": "position", "snapshot": snapshot, "index": index, "label": label, "move": mv, "proved": d.proved,
        "nodes": d.cost.nodes, "seconds": d.cost.seconds, "steps": steps, "facts": facts, "trace": d.trace, "error": null,
    });
    (Summary { index, mv, proved: d.proved, nodes: d.cost.nodes, seconds: d.cost.seconds, error: None }, v)
}

/// A position that could not be solved (it did not parse, it is over, a worker panicked): the batch goes on, the row says why.
pub fn error_row(index: usize, snapshot: &str, label: &str, why: &str) -> (Summary, Value) {
    let v = json!({"kind": "position", "snapshot": snapshot, "index": index, "label": label, "move": null, "proved": false, "nodes": 0, "seconds": 0.0,
                   "steps": [], "facts": [], "trace": [], "error": why});
    (Summary { index, mv: None, proved: false, nodes: 0, seconds: 0.0, error: Some(why.to_string()) }, v)
}

/// THE BATCH: `f(index, item)` for every item on `threads` workers (0: one per core), results in item order. A panic inside one position is that position's error
/// row, not the batch's.
pub fn solve<T: Sync, F>(items: &[T], threads: usize, snapshot: &str, f: F) -> Vec<(Summary, Value)>
where F: Fn(usize, &T) -> (Summary, Value) + Sync {
    let run = || items.par_iter().enumerate().map(|(i, it)| {
        catch_unwind(AssertUnwindSafe(|| f(i, it))).unwrap_or_else(|_| error_row(i, snapshot, "", "the worker panicked"))
    }).collect::<Vec<_>>();
    match rayon::ThreadPoolBuilder::new().num_threads(threads).build() {
        Ok(pool) => pool.install(run),
        Err(_) => run(),
    }
}

/// Append the rows to the log (created if missing), one JSON object a line.
pub fn append(path: &Path, rows: &[Value]) -> std::io::Result<()> {
    let mut f = std::fs::OpenOptions::new().create(true).append(true).open(path)?;
    let mut buf = String::new();
    for r in rows { buf.push_str(&r.to_string()); buf.push('\n'); }
    f.write_all(buf.as_bytes())
}
