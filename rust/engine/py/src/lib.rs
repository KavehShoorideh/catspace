//! plan 78: THE END-TO-END SYSTEM's Python entry -- the harness calls in once per move (or once per certification) and reads the result; no per-node crossing.
use cs_game::{certify, gdl_game::GdlGame, kernel, Game};
use pyo3::prelude::*;
use std::collections::HashMap;

fn state_of(variant: &str, fen: Option<&str>, start960: u32, moves: &[String]) -> PyResult<(kernel::KernelGame, shakmaty_state::S)> {
    let (g, s, _h) = record_of(variant, fen, start960, moves)?;
    Ok((g, s))
}

/// The game, the state after `moves`, and the INSTANCE's record: the keys of every position before it (plan 78.8).
fn record_of(variant: &str, fen: Option<&str>, start960: u32, moves: &[String]) -> PyResult<(kernel::KernelGame, shakmaty_state::S, Vec<u64>)> {
    let g = match fen { Some(f) => kernel::KernelGame::from_fen(variant, f), None => kernel::KernelGame::new(variant, start960) }
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    let mut s = g.initial();
    let mut keys = vec![];
    for u in moves {
        let m = g.parse(&s, u).ok_or_else(|| pyo3::exceptions::PyValueError::new_err(format!("illegal move {u}")))?;
        keys.push(g.key(&s));
        s = g.apply(&s, &m);
    }
    Ok((g, s, keys))
}

mod sheet_py;
mod shakmaty_state { pub type S = <cs_game::kernel::KernelGame as cs_game::Game>::State; }

/// best_move(variant, fen=None, moves=[], seconds=1.0, worth={}, start960=518) -> (move or None, proved, [trace], nodes, fen after the moves): THE END-TO-END
/// DECISION (plan 78.5: tactics `forcing`, then the ledger search) for the side to move after `moves` from the start (or from `fen`). The ledger's JUDGE is the
/// store's valuation record in use, passed in: `worth` (kind name -> worth), `ours_first`, `valuation` (its name, for the trace); an empty table: the built-in Count.
#[pyfunction]
#[pyo3(signature = (variant, fen=None, moves=vec![], seconds=1.0, worth=HashMap::new(), start960=518, ours_first=false, valuation=String::new(), prior=false, record=true, prior_depth=2, ledger_share=0.6, prior_defects=1, ordering=true, prune=true, extend=true, reason=false, wdl=false, wdl_table=None))]
#[allow(clippy::too_many_arguments)]
fn best_move(variant: &str, fen: Option<&str>, moves: Vec<String>, seconds: f64, worth: HashMap<String, i64>, start960: u32, ours_first: bool, valuation: String,
             prior: bool, record: bool, prior_depth: i32, ledger_share: f64, prior_defects: u32, ordering: bool, prune: bool, extend: bool, reason: bool, wdl: bool, wdl_table: Option<String>)
             -> PyResult<(Option<String>, bool, Vec<String>, u64, String)> {
    let (g, s, keys) = record_of(variant, fen, start960, &moves)?;
    let hist: Vec<u64> = if record { keys } else { vec![] };          // (record=False: the game's record not read -- the comparison)
    let f = g.fen(&s);
    let table = if wdl { load_wdl_table(wdl_table.as_deref())? } else { None };
    let params = cs_decision::Params { prior, history: hist.clone(), prior_depth, ledger_share, prior_defects, ordering, prune, extend, reason, wdl, wdl_table: table, ..cs_decision::Params::default() };
    let choice = if reason || wdl {
        // (plan 78.25 / 78.26: the reasoning chain or the win / draw / loss path on: the decision with its extra steps; both off, the call below is exactly what it was)
        (if worth.is_empty() { cs_decision::decide_params(&g, &s, seconds, &cs_tactics::ledger::Count, &params) }
         else { cs_decision::decide_params(&g, &s, seconds, &cs_tactics::ledger::Worth { name: valuation, table: worth.clone(), ours_first }, &params) })
            .map(|d| cs_decision::Choice { mv: d.mv.expect("a decision with a move"), proved: d.proved, trace: d.trace, nodes: d.cost.nodes })
    } else if worth.is_empty() { cs_decision::decide_with(&g, &s, seconds, &cs_tactics::ledger::Count, prior, &hist, prior_depth, ledger_share, prior_defects, ordering, prune, extend) }
                 else { cs_decision::decide_with(&g, &s, seconds, &cs_tactics::ledger::Worth { name: valuation, table: worth.clone(), ours_first }, prior, &hist, prior_depth, ledger_share, prior_defects, ordering, prune, extend) };
    Ok(match choice {
        Some(c) => (Some(g.move_name(&s, &c.mv)), c.proved, c.trace, c.nodes, f),
        None => (None, false, vec!["the game is over".into()], 0, f),
    })
}

/// The format string a results table file must carry.
const WDL_TABLE_FORMAT: &str = "wdl-table-1";

/// Load a results table file (JSON: {"format": "wdl-table-1", "game", "provenance", "rows": {key: [mover wins, draws, mover losses]}}), once; None when no path is given.
fn load_wdl_table(path: Option<&str>) -> PyResult<Option<std::sync::Arc<cs_planning::wdl::Table>>> {
    let Some(path) = path else { return Ok(None) };
    let err = |m: String| pyo3::exceptions::PyValueError::new_err(format!("wdl table {path}: {m}"));
    let text = std::fs::read_to_string(path).map_err(|e| pyo3::exceptions::PyOSError::new_err(format!("wdl table {path}: {e}")))?;
    let v: serde_json::Value = serde_json::from_str(&text).map_err(|e| err(e.to_string()))?;
    match v.get("format").and_then(|f| f.as_str()) {
        Some(WDL_TABLE_FORMAT) => {}
        other => return Err(err(format!("format {other:?} is not \"{WDL_TABLE_FORMAT}\""))),
    }
    let rows = v.get("rows").and_then(|r| r.as_object()).ok_or_else(|| err("no rows".into()))?;
    let mut t = cs_planning::wdl::Table { name: v.get("game").and_then(|g| g.as_str()).unwrap_or("").to_string(), rows: HashMap::new() };
    for (k, row) in rows {
        let a = row.as_array().filter(|a| a.len() == 3).ok_or_else(|| err(format!("row {k}: three counts expected")))?;
        let f = |i: usize| a[i].as_f64().ok_or_else(|| err(format!("row {k}: counts must be numbers")));
        t.rows.insert(k.clone(), [f(0)?, f(1)?, f(2)?]);
    }
    Ok(Some(std::sync::Arc::new(t)))
}

/// The win / draw / loss operator read at `s` under `judge`: the careful (or greedy) playouts, the balance leaf at the wdl scale, the table when given.
fn wdl_read<G: Game, J: cs_tactics::ledger::Judge + Clone>(g: &G, s: &G::State, me: usize, judge: J, table: Option<std::sync::Arc<cs_planning::wdl::Table>>, among: Option<Vec<usize>>,
                                                           seed: u64, max_playout: usize, careful: bool, budget: cs_operator::Budget) -> cs_operator::Outcome<G::Move> {
    use cs_operator::Operator;
    let scale = cs_planning::wdl::leaf_scale(g, s, &judge);
    let goal = match among { Some(a) => cs_operator::Goal::Prefer { role: me, among: a, defects: 0 }, None => cs_operator::Goal::Estimate { role: me } };
    if careful {
        cs_planning::wdl::WdlOp::new(cs_planning::mcts::EXPLORATION, max_playout, cs_planning::rollout::Careful::new(judge.clone(), cs_decision::WDL_EPS, seed),
                                     cs_planning::rollout::Balance::new(judge, scale), table).run(g, s, &goal, budget)
    } else {
        cs_planning::wdl::WdlOp::new(cs_planning::mcts::EXPLORATION, max_playout, cs_planning::rollout::Greedy::new(judge.clone(), cs_decision::WDL_EPS, seed),
                                     cs_planning::rollout::Balance::new(judge, scale), table).run(g, s, &goal, budget)
    }
}

/// wdl(variant, fen=None, moves=[], seconds=0.2, max_nodes=10_000_000, worth={}, ours_first=False, valuation="", table=None, among=None, seed=1, max_playout=8,
/// rollout="careful") -> dict:
/// plan 78.26 THE WIN / DRAW / LOSS of the position for the side to move (cs_planning::wdl): playouts (`rollout`: "careful", a move's gain less what it leaves the next
/// mover to take, or "greedy", the gain alone; a cut one valued by the units' balance at 3
/// times the smallest worth) pooled with the results table's row for the position when `table` (a file path) has one. {win, draw, loss, n (mass), score, sharpness (1 less
/// the draw probability), sharp (True / False / None: unknown), table_key, table_counts, playouts, cut, nodes, seconds, evidence, moves: [{move, win, draw, loss, n, score,
/// visits}]}. `among` (move names) asks the probabilities to pick among those moves instead: `moves` then holds them, best first, and the numbers are the best one's.
#[pyfunction]
#[pyo3(signature = (variant, fen=None, moves=vec![], seconds=0.2, max_nodes=10_000_000, worth=HashMap::new(), ours_first=false, valuation=String::new(), table=None, among=None, seed=1, max_playout=cs_planning::wdl::WDL_PLAYOUT, rollout="careful"))]
#[allow(clippy::too_many_arguments)]
fn wdl(py: Python<'_>, variant: &str, fen: Option<&str>, moves: Vec<String>, seconds: f64, max_nodes: u64, worth: HashMap<String, i64>, ours_first: bool, valuation: String,
       table: Option<String>, among: Option<Vec<String>>, seed: u64, max_playout: usize, rollout: &str) -> PyResult<PyObject> {
    let careful = match rollout { "careful" => true, "greedy" => false, other => return Err(pyo3::exceptions::PyValueError::new_err(format!("rollout: {other} is not careful or greedy"))) };
    let (g, s) = state_of(variant, fen, 518, &moves)?;
    let Some(me) = g.mover(&s) else { return Err(pyo3::exceptions::PyValueError::new_err("the game is over")) };
    let table = load_wdl_table(table.as_deref())?;
    let legal = g.moves(&s);
    let among_ix = match &among {
        Some(names) => Some(names.iter().map(|n| legal.iter().position(|m| g.move_name(&s, m) == *n).ok_or_else(|| pyo3::exceptions::PyValueError::new_err(format!("among: {n} is not legal")))).collect::<PyResult<Vec<usize>>>()?),
        None => None,
    };
    let o = py.allow_threads(|| {
        let budget = cs_operator::Budget::new(seconds, max_nodes);
        if worth.is_empty() { wdl_read(&g, &s, me, cs_tactics::ledger::Count, table, among_ix, seed, max_playout, careful, budget) }
        else { wdl_read(&g, &s, me, cs_tactics::ledger::Worth { name: valuation, table: worth.clone(), ours_first }, table, among_ix, seed, max_playout, careful, budget) }
    });
    let fact = |name: &str| o.facts.iter().find(|f| f.relation == name);
    let num = |name: &str, i: usize| -> f64 { fact(name).and_then(|f| f.args.get(i)).and_then(|v| v.parse().ok()).unwrap_or(0.0) };
    let d = pyo3::types::PyDict::new(py);
    d.set_item("status", format!("{:?}", o.status))?;
    d.set_item("win", num("wdl", 0))?;
    d.set_item("draw", num("wdl", 1))?;
    d.set_item("loss", num("wdl", 2))?;
    d.set_item("n", num("wdl", 3))?;
    d.set_item("sources", fact("wdl").and_then(|f| f.args.get(4)).cloned().unwrap_or_default())?;
    d.set_item("score", o.value.map(|v| v as f64 / cs_planning::wdl::SCORE_SCALE))?;
    d.set_item("sharpness", num("sharpness", 0))?;
    let word = fact("sharpness").and_then(|f| f.args.get(1)).cloned().unwrap_or_default();
    d.set_item("sharp", match word.as_str() { "sharp" => Some(true), "quiet" => Some(false), _ => None })?;
    d.set_item("table_key", fact("table_hit").and_then(|f| f.args.first()).cloned())?;
    d.set_item("table_counts", fact("table_hit").map(|f| (0..3).map(|i| f.args[i + 1].parse::<f64>().unwrap_or(0.0)).collect::<Vec<f64>>()))?;
    d.set_item("playouts", num("playouts", 0) as u64)?;
    d.set_item("cut", num("playouts", 1) as u64)?;
    d.set_item("nodes", o.cost.nodes)?;
    d.set_item("seconds", o.cost.seconds)?;
    d.set_item("evidence", o.evidence.clone())?;
    d.set_item("move", o.mv.as_ref().map(|m| g.move_name(&s, m)))?;
    let rows = pyo3::types::PyList::empty(py);
    for f in o.facts.iter().filter(|f| f.relation == "move_wdl") {
        let m = pyo3::types::PyDict::new(py);
        let p = |i: usize| f.args[i].parse::<f64>().unwrap_or(0.0);
        m.set_item("move", f.args[0].clone())?;
        m.set_item("win", p(1))?;
        m.set_item("draw", p(2))?;
        m.set_item("loss", p(3))?;
        m.set_item("n", p(4))?;
        m.set_item("score", p(1) + p(2) / 2.0)?;
        m.set_item("visits", p(4).round() as u64)?;
        rows.append(m)?;
    }
    d.set_item("moves", rows)?;
    Ok(d.into())
}

/// The keys of a position in the results table, most specific first (cs_planning::wdl::Table::keys).
fn keys_of(g: &kernel::KernelGame, s: &shakmaty_state::S) -> Vec<String> { cs_planning::wdl::Table::keys(g, s) }

/// wdl_keys(variant, fen=None, moves=[]) -> [str]: the table keys of the position after `moves`, most specific first.
#[pyfunction]
#[pyo3(signature = (variant, fen=None, moves=vec![]))]
fn wdl_keys(variant: &str, fen: Option<&str>, moves: Vec<String>) -> PyResult<Vec<String>> {
    let (g, s) = state_of(variant, fen, 518, &moves)?;
    Ok(keys_of(&g, &s))
}

/// wdl_keys_along(variant, fen=None, moves=[]) -> [[str]]: the keys of every position along the line, the start included (len(moves) + 1 lists), in one call.
#[pyfunction]
#[pyo3(signature = (variant, fen=None, moves=vec![]))]
fn wdl_keys_along(variant: &str, fen: Option<&str>, moves: Vec<String>) -> PyResult<Vec<Vec<String>>> {
    let g = match fen { Some(f) => kernel::KernelGame::from_fen(variant, f), None => kernel::KernelGame::new(variant, 518) }.map_err(pyo3::exceptions::PyValueError::new_err)?;
    let mut s = g.initial();
    let mut out = vec![keys_of(&g, &s)];
    for u in &moves {
        let m = g.parse(&s, u).ok_or_else(|| pyo3::exceptions::PyValueError::new_err(format!("illegal move {u}")))?;
        s = g.apply(&s, &m);
        out.push(keys_of(&g, &s));
    }
    Ok(out)
}

/// state(variant, fen=None, moves=[], start960=518) -> (fen, the mover's name or None, [legal moves], the outcome per role or None): the harness's view of a state.
#[pyfunction]
#[pyo3(signature = (variant, fen=None, moves=vec![], start960=518))]
fn state(variant: &str, fen: Option<&str>, moves: Vec<String>, start960: u32) -> PyResult<(String, Option<String>, Vec<String>, Option<Vec<i64>>, u64)> {
    let (g, s) = state_of(variant, fen, start960, &moves)?;
    let ms = g.moves(&s).iter().map(|m| g.move_name(&s, m)).collect();
    Ok((g.fen(&s), g.mover(&s).map(|r| g.roles()[r].clone()), ms, g.outcome(&s), g.key(&s)))
}

/// passed(variant, fen=None, moves=[], start960=518) -> the `state` tuple of the position after the side to move passes (the other side to move, the en-passant right
/// gone), by the kernel's own `pass` (the one `census(side="other")` reads); None when it refuses (the mover in check, the game over, a game with no pass).
/// The fen of the tuple is a position string the other calls take with no moves.
#[pyfunction]
#[pyo3(signature = (variant, fen=None, moves=vec![], start960=518))]
fn passed(variant: &str, fen: Option<&str>, moves: Vec<String>, start960: u32) -> PyResult<Option<(String, Option<String>, Vec<String>, Option<Vec<i64>>, u64)>> {
    let (g, s0) = state_of(variant, fen, start960, &moves)?;
    Ok(g.pass(&s0).map(|s| {
        let ms = g.moves(&s).iter().map(|m| g.move_name(&s, m)).collect();
        (g.fen(&s), g.mover(&s).map(|r| g.roles()[r].clone()), ms, g.outcome(&s), g.key(&s))
    }))
}

/// The variants the kernels play (lichess's names).
#[pyfunction]
fn variants() -> Vec<String> { kernel::VARIANTS.iter().map(|s| s.to_string()).collect() }

/// certify_coplay(variant, gdl_text, games, seed, base=None) -> (games, plies, states, [disagreements]): the chess kernel of `variant` against the GDL text (a
/// delta composed on `base` when given), by co-play.
#[pyfunction]
#[pyo3(signature = (variant, gdl_text, games, seed, base=None))]
fn certify_coplay(variant: &str, gdl_text: &str, games: usize, seed: u64, base: Option<&str>) -> PyResult<(usize, usize, usize, Vec<String>)> {
    let k = kernel::KernelGame::new(variant, 518).map_err(pyo3::exceptions::PyValueError::new_err)?;
    let text = match base { Some(b) => cs_gdl::compose(b, gdl_text).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.0))?, None => gdl_text.to_string() };
    let g = GdlGame::load(variant, &text).map_err(pyo3::exceptions::PyValueError::new_err)?;
    let r = certify::coplay(&k, &g, &|n| kernel::uci_to_spec_term(n), games, seed);
    Ok((r.games, r.plies, r.states, r.disagreements))
}

/// watch(variant, engine, games, random_plies=6, movetime_ms=20, seed=78, threads=11, options={}) -> [(start, [moves], outcome)]: plan 78.4 -- instances by
/// WATCHING an outside UCI engine play itself (seeded random openings; Chess960: each game its own start, seeded); the game judged by the kernel.
#[pyfunction]
#[pyo3(signature = (variant, engine, games, random_plies=6, movetime_ms=20, seed=78, threads=11, options=HashMap::new()))]
fn watch(variant: &str, engine: &str, games: usize, random_plies: usize, movetime_ms: u64, seed: u64, threads: usize, options: HashMap<String, String>)
         -> PyResult<Vec<(String, Vec<String>, Vec<i64>)>> {
    let mut opts: Vec<(String, String)> = options.into_iter().collect();
    if variant == "chess960" { opts.push(("UCI_Chess960".into(), "true".into())); } else if variant != "chess" { opts.push(("UCI_Variant".into(), variant.into())); }
    kernel::KernelGame::new(variant, 518).map_err(pyo3::exceptions::PyValueError::new_err)?;
    let v = variant.to_string();
    struct T(cs_game::outside::UciTeacher);
    impl cs_learn::watch::Teacher for T {
        fn name(&self) -> String { "uci".into() }
        fn start(&mut self, start: &str) { self.0.begin(start) }
        fn best(&mut self, moves: &[String]) -> Option<String> { self.0.answer(moves) }
    }
    let recs = cs_learn::watch::watch(|i| {
        if v == "chess960" {
            let n = (cs_game::rng::Rng::new(seed ^ (0x960 + i as u64)).next_u64() % 960) as u32;
            let f = kernel::chess960_fen(n);
            (kernel::KernelGame::from_fen("chess960", &f).unwrap(), format!("fen {f}"), f)
        } else {
            (kernel::KernelGame::new(&v, 518).unwrap(), "startpos".to_string(), String::new())
        }
    }, || cs_game::outside::UciTeacher::new(engine, &opts, movetime_ms).map(T), games, random_plies, seed, threads);
    Ok(recs.into_iter().map(|r| (r.start, r.moves, r.outcome)).collect())
}

/// mine_valuation(variant, records, from_ply=10, fuse=64) -> {kinds, constant, beta, band, logloss_best, se_best, worth, price, logloss, excess, se, tried,
/// positions, games, groups}: plan 78.4 -- THE VALUATION MINER on the records [(start, [moves], outcome)] (cs_learn::valuation).
#[pyfunction]
#[pyo3(signature = (variant, records, from_ply=10, fuse=64))]
fn mine_valuation(py: Python<'_>, variant: &str, records: Vec<(String, Vec<String>, Vec<i64>)>, from_ply: usize, fuse: u32) -> PyResult<PyObject> {
    let recs: Vec<cs_learn::Record> = records.into_iter().map(|(start, moves, outcome)| cs_learn::Record { start, moves, outcome }).collect();
    let v = variant.to_string();
    let c = cs_learn::valuation::corpus(|r: &cs_learn::Record| {
        if r.start.is_empty() { kernel::KernelGame::new(&v, 518).ok() } else { kernel::KernelGame::from_fen(&v, &r.start).ok() }
    }, &recs, from_ply);
    let f = cs_learn::valuation::search(&c, fuse);
    let d = pyo3::types::PyDict::new(py);
    d.set_item("kinds", f.kinds)?; d.set_item("constant", f.constant)?; d.set_item("beta", f.beta)?; d.set_item("band", f.band)?;
    d.set_item("logloss_best", f.logloss_best)?; d.set_item("se_best", f.se_best)?; d.set_item("worth", f.worth)?; d.set_item("price", f.price)?;
    d.set_item("logloss", f.logloss)?; d.set_item("excess", f.excess)?; d.set_item("se", f.se)?; d.set_item("tried", f.tried)?;
    d.set_item("positions", f.positions)?; d.set_item("games", f.games)?; d.set_item("groups", f.groups)?;
    Ok(d.into())
}

/// census(variant, fen=None, moves=[], max_defects=1, depth=3, fuse=200000, side="mover", worth={}, ours_first=false, valuation="") -> (tactics, stats):
/// plan 78.6 THE CENSUS OF TACTICS BY THEIR DEFECTS (cs_tactics::census) of the side to move ("mover") or of the other side ("other": read after a pass) --
/// tactics: [{defects, free, line, wins, ledger, value}]; stats: {readings, cut, nodes, frontier}. The judge: the store's record, as best_move takes it.
#[pyfunction]
#[pyo3(signature = (variant, fen=None, moves=vec![], max_defects=1, depth=3, fuse=200000, side="mover", worth=HashMap::new(), ours_first=false, valuation=String::new()))]
#[allow(clippy::too_many_arguments)]
fn census(py: Python<'_>, variant: &str, fen: Option<&str>, moves: Vec<String>, max_defects: u32, depth: i32, fuse: u64, side: &str, worth: HashMap<String, i64>,
          ours_first: bool, valuation: String) -> PyResult<(Vec<PyObject>, PyObject)> {
    let (g, s0) = state_of(variant, fen, 518, &moves)?;
    let s = if side == "other" { match g.pass(&s0) { Some(q) => q, None => return Ok((vec![], pyo3::types::PyDict::new(py).into())) } } else { s0 };
    let rep = if worth.is_empty() { cs_tactics::census::census(&g, &s, &cs_tactics::ledger::Count, max_defects, depth, fuse) }
              else { cs_tactics::census::census(&g, &s, &cs_tactics::ledger::Worth { name: valuation, table: worth, ours_first }, max_defects, depth, fuse) };
    let mut out = vec![];
    for t in rep.tactics {
        let d = pyo3::types::PyDict::new(py);
        d.set_item("defects", t.defects)?; d.set_item("free", t.free)?; d.set_item("line", t.line)?; d.set_item("wins", t.wins)?; d.set_item("targets", t.targets)?;
        d.set_item("ledger", t.ledger)?; d.set_item("value", t.value)?;
        out.push(d.into());
    }
    let st = pyo3::types::PyDict::new(py);
    st.set_item("readings", rep.readings)?; st.set_item("cut", rep.cut)?; st.set_item("nodes", rep.nodes)?; st.set_item("frontier", rep.frontier)?;
    Ok((out, st.into()))
}

type RelArg = (String, usize, usize, Vec<usize>, Vec<usize>);

fn relspec_json(v: &serde_json::Value) -> Result<cs_tactics::scenario::RelSpec, String> {
    let ix = |k: &str| -> Result<Vec<usize>, String> { v.get(k).and_then(|x| x.as_array()).ok_or(format!("scenario relation: {k} is a list"))?.iter().map(|x| x.as_u64().map(|n| n as usize).ok_or(format!("{k} holds numbers"))).collect() };
    let n = |k: &str| -> Result<usize, String> { v.get(k).and_then(|x| x.as_u64()).map(|n| n as usize).ok_or(format!("scenario relation: {k} is a number")) };
    Ok(cs_tactics::scenario::RelSpec { relation: v.get("relation").and_then(|x| x.as_str()).ok_or("scenario relation: relation is a name")?.to_string(), kind: n("kind")?, owner: n("owner")?, from: ix("from")?, to: ix("to")? })
}

fn relspec_arg(a: RelArg) -> cs_tactics::scenario::RelSpec { cs_tactics::scenario::RelSpec { relation: a.0, kind: a.1, owner: a.2, from: a.3, to: a.4 } }

thread_local! {
    /// The goal's target per description text (reading the description's goal costs a load of the program).
    static GOAL_TARGETS: std::cell::RefCell<HashMap<String, Result<cs_game::goal_target::GoalTarget, String>>> = std::cell::RefCell::new(HashMap::new());
}

fn goal_target_for(spec: &str) -> Result<cs_game::goal_target::GoalTarget, String> {
    let key = cs_memory::sha::sha256_hex(spec.as_bytes());
    GOAL_TARGETS.with(|m| m.borrow_mut().entry(key).or_insert_with(|| cs_game::goal_target::goal_target(spec)).clone())
}

fn loc_text(l: &[String]) -> String { cs_tactics::scenario::show_loc(&l.to_vec()) }

/// scenario(variant, spec, vocabulary, fen=None, moves=[], role="mover", k=2, acts_on=None, can_go=None, max_holes=8, max_units=16, max_scenarios=20, max_nodes=200000, demand=True) -> dict:
/// plan 78.39 stage 1, A GOAL TAKEN APART BACKWARD (cs_tactics::scenario), as a report. `spec` is the description text (composed with the vocabulary's concepts), `vocabulary` its JSON.
/// `role`: "mover", "other" or a role's name -- the side that wants the goal. The target is read from the description's goal (unreadable: {"unreadable": why}). The two relations are
/// named by the roles of their arguments, as (relation, kind, owner, [from], [to]) positions: `acts_on` and `can_go` (one per move count, 1.. in order); left out, they are read from
/// the vocabulary JSON's "scenario" entry ({"acts_on": {relation, kind, owner, from, to}, "can_go": [..]}). -> {role, target: {kind, owner, at}, unreadable, goal_holds, exits_from,
/// holes: [{at, kind}], covered: [{at, own, by: [{kind, at}]}], units: [{kind, at}], supply: [[{unit, kind, from, to, moves}]] (per hole), scenarios: [{moves, placements: [{kind, from,
/// to, moves}]}], safe (True: no scenario exists; False: one does; None: not decided), notes, contest (None: unread), evidence: [lines], nodes, seconds}. A report, not a verdict.
#[pyfunction]
#[pyo3(signature = (variant, spec, vocabulary, fen=None, moves=vec![], role="mover", k=2, acts_on=None, can_go=None, max_holes=8, max_units=16, max_scenarios=20, max_nodes=200000, demand=true))]
#[allow(clippy::too_many_arguments)]
fn scenario(py: Python<'_>, variant: &str, spec: &str, vocabulary: &str, fen: Option<&str>, moves: Vec<String>, role: &str, k: u32, acts_on: Option<RelArg>, can_go: Option<Vec<RelArg>>,
            max_holes: usize, max_units: usize, max_scenarios: usize, max_nodes: u64, demand: bool) -> PyResult<PyObject> {
    let bad = |e: String| pyo3::exceptions::PyValueError::new_err(e);
    let (g, s) = state_of(variant, fen, 518, &moves)?;
    let view = facts_view_for(spec, vocabulary, demand)?;
    let d = pyo3::types::PyDict::new(py);
    let target = match goal_target_for(spec) { Ok(t) => t, Err(e) => { d.set_item("unreadable", e)?; return Ok(d.into()); } };
    let vj: serde_json::Value = serde_json::from_str(vocabulary).map_err(|e| bad(format!("vocabulary: {e}")))?;
    let named = vj.get("scenario");
    let acts = match acts_on { Some(a) => relspec_arg(a), None => relspec_json(named.and_then(|n| n.get("acts_on")).ok_or_else(|| bad("no acts_on: name it, or give the vocabulary a scenario entry".into()))?).map_err(bad)? };
    let go = match can_go {
        Some(v) => v.into_iter().map(relspec_arg).collect(),
        None => named.and_then(|n| n.get("can_go")).and_then(|x| x.as_array()).ok_or_else(|| bad("no can_go: name them, or give the vocabulary a scenario entry".into()))?.iter().map(relspec_json).collect::<Result<Vec<_>, _>>().map_err(bad)?,
    };
    let me = cs_tactics::scenario::resolve_role(&g, &s, role).map_err(bad)?;
    let p = cs_tactics::scenario::Params { k, max_holes, max_units, max_scenarios, max_nodes };
    let r = cs_tactics::scenario::scenario(&g, &s, &view, &target, &cs_tactics::scenario::Relations { acts_on: acts, can_go: go }, me, &p).map_err(bad)?;
    let unit = |u: &cs_tactics::scenario::UnitAt| -> PyResult<PyObject> { let x = pyo3::types::PyDict::new(py); x.set_item("kind", u.kind.clone())?; x.set_item("at", loc_text(&u.at))?; Ok(x.into()) };
    d.set_item("role", r.role.clone())?;
    match &r.target {
        Some(t) => { let x = pyo3::types::PyDict::new(py); x.set_item("kind", t.kind.clone())?; x.set_item("owner", t.owner.clone())?; x.set_item("at", loc_text(&t.at))?; d.set_item("target", x)?; }
        None => d.set_item("target", py.None())?,
    }
    d.set_item("unreadable", r.unreadable.clone())?;
    d.set_item("goal_holds", r.goal_holds)?;
    d.set_item("exits_from", r.exits_from.clone())?;
    let mut holes = vec![];
    for h in &r.holes { let x = pyo3::types::PyDict::new(py); x.set_item("at", loc_text(&h.at))?; x.set_item("kind", h.kind.word())?; holes.push(x); }
    d.set_item("holes", holes)?;
    let mut covered = vec![];
    for c in &r.covered {
        let x = pyo3::types::PyDict::new(py);
        x.set_item("at", loc_text(&c.at))?; x.set_item("own", c.own)?;
        x.set_item("by", c.by.iter().map(&unit).collect::<PyResult<Vec<_>>>()?)?;
        covered.push(x);
    }
    d.set_item("covered", covered)?;
    d.set_item("units", r.units.iter().map(&unit).collect::<PyResult<Vec<_>>>()?)?;
    let mut supply = vec![];
    for per in &r.supply {
        let mut row = vec![];
        for x in per {
            let y = pyo3::types::PyDict::new(py);
            y.set_item("unit", x.unit)?; y.set_item("kind", r.units[x.unit].kind.clone())?; y.set_item("from", loc_text(&r.units[x.unit].at))?; y.set_item("to", loc_text(&x.to))?; y.set_item("moves", x.moves)?;
            row.push(y);
        }
        supply.push(row);
    }
    d.set_item("supply", supply)?;
    let mut scenarios = vec![];
    for sc in &r.scenarios {
        let x = pyo3::types::PyDict::new(py);
        x.set_item("moves", sc.total_moves)?;
        let mut pl = vec![];
        for q in &sc.placements {
            let y = pyo3::types::PyDict::new(py);
            y.set_item("kind", q.unit.kind.clone())?; y.set_item("from", loc_text(&q.unit.at))?; y.set_item("to", loc_text(&q.to))?; y.set_item("moves", q.moves)?;
            pl.push(y);
        }
        x.set_item("placements", pl)?;
        scenarios.push(x);
    }
    d.set_item("scenarios", scenarios)?;
    d.set_item("safe", r.safe)?;
    d.set_item("notes", r.notes.clone())?;
    d.set_item("contest", r.contest.clone())?;
    d.set_item("evidence", cs_tactics::scenario::explain(&r))?;
    d.set_item("nodes", r.nodes)?; d.set_item("seconds", r.seconds)?;
    Ok(d.into())
}

/// reason(variant, fen=None, moves=[], depth=2, fuse=200000, worth={}, ours_first=false, valuation="") -> dict: plan 78.25 THE REASONING CHAIN (cs_tactics::reason) of the
/// side to move -- {must_answer, threats: [{stake, goal, targets, line, frame}], answers: [{index, move, kind, net, line}], irrelevant: [{facts, moves}], unread: [move
/// names], leaving, ours: [{defects, free, line, targets, value}], candidates: [indices], candidate_moves: [names], evidence: [lines], readings, cut, nodes}. The judge:
/// the store's record, as census takes it. An order and an explanation, never a verdict.
#[pyfunction]
#[pyo3(signature = (variant, fen=None, moves=vec![], depth=cs_tactics::reason::REASON_DEPTH, fuse=cs_tactics::reason::REASON_FUSE, worth=HashMap::new(), ours_first=false, valuation=String::new()))]
#[allow(clippy::too_many_arguments)]
fn reason(py: Python<'_>, variant: &str, fen: Option<&str>, moves: Vec<String>, depth: i32, fuse: u64, worth: HashMap<String, i64>, ours_first: bool, valuation: String) -> PyResult<PyObject> {
    let (g, s) = state_of(variant, fen, 518, &moves)?;
    let rep = if worth.is_empty() { cs_tactics::reason::reason(&g, &s, &cs_tactics::ledger::Count, depth, fuse) }
              else { cs_tactics::reason::reason(&g, &s, &cs_tactics::ledger::Worth { name: valuation, table: worth, ours_first }, depth, fuse) };
    let facts = |fs: &[cs_game::Fact]| fs.iter().map(|f| f.to_string()).collect::<Vec<String>>();
    let d = pyo3::types::PyDict::new(py);
    d.set_item("must_answer", rep.must_answer)?;
    let mut threats = vec![];
    for t in &rep.threats {
        let x = pyo3::types::PyDict::new(py);
        x.set_item("stake", t.stake)?; x.set_item("goal", t.goal)?; x.set_item("targets", t.targets.clone())?; x.set_item("line", t.line.clone())?; x.set_item("frame", facts(&t.frame))?;
        threats.push(x);
    }
    d.set_item("threats", threats)?;
    let mut answers = vec![];
    for a in &rep.answers {
        let x = pyo3::types::PyDict::new(py);
        x.set_item("index", a.index)?; x.set_item("move", a.name.clone())?; x.set_item("kind", a.kind.word())?; x.set_item("net", a.net)?; x.set_item("line", a.line.clone())?;
        x.set_item("in_frame", a.in_frame)?;
        x.set_item("if_executed", a.if_executed)?; x.set_item("executed_line", a.executed_line.clone())?;
        answers.push(x);
    }
    d.set_item("answers", answers)?;
    let mut irr = vec![];
    for u in &rep.irrelevant {
        let x = pyo3::types::PyDict::new(py);
        x.set_item("facts", facts(&u.facts))?; x.set_item("moves", u.moves.iter().map(|i| rep.names[*i].clone()).collect::<Vec<String>>())?;
        irr.push(x);
    }
    d.set_item("irrelevant", irr)?;
    d.set_item("unread", rep.unread.iter().map(|i| rep.names[*i].clone()).collect::<Vec<String>>())?;
    d.set_item("leaving", rep.leaving)?;
    let mut ours = vec![];
    for t in &rep.ours {
        let x = pyo3::types::PyDict::new(py);
        x.set_item("defects", t.defects)?; x.set_item("free", t.free.clone())?; x.set_item("line", t.line.clone())?; x.set_item("targets", t.targets.clone())?; x.set_item("value", t.value)?;
        ours.push(x);
    }
    d.set_item("ours", ours)?;
    d.set_item("candidates", rep.candidates.clone())?;
    d.set_item("candidate_moves", rep.candidates.iter().map(|i| rep.names[*i].clone()).collect::<Vec<String>>())?;
    d.set_item("evidence", cs_tactics::reason::explain(&rep))?;
    d.set_item("readings", rep.readings)?; d.set_item("cut", rep.cut)?; d.set_item("nodes", rep.nodes)?;
    Ok(d.into())
}

type BatchItem = (String, Option<String>, Vec<String>);
type BatchRow = (Option<String>, bool, u64, f64, Option<String>);

/// The snapshot a batch reads and the descriptions its lemmas are bound through (per variant: the text and its content id).
struct Store { snapshot: cs_memory::snapshot::Snapshot, specs: HashMap<String, (String, String)> }

thread_local! {
    /// One fact view per (variant, description, snapshot) on each worker: compiling a description is costly, reading a position through it is not.
    static VIEWS: std::cell::RefCell<HashMap<String, std::rc::Rc<cs_game::gdl_game::FactView>>> = std::cell::RefCell::new(HashMap::new());
}

fn view_for(store: &Store, variant: &str) -> Option<std::rc::Rc<cs_game::gdl_game::FactView>> {
    let (text, id) = store.specs.get(variant)?;
    let key = format!("{variant}|{id}|{}", store.snapshot.id);
    VIEWS.with(|v| {
        if let Some(x) = v.borrow().get(&key) { return Some(x.clone()); }
        let view = std::rc::Rc::new(cs_game::gdl_game::FactView::new(text, store.snapshot.vocabulary.clone()).ok()?);
        v.borrow_mut().insert(key, view.clone());
        Some(view)
    })
}

/// One position of a batch: the game and state after the item's moves (its record read unless `record` is off), then the fixed controller's decision -- with the
/// snapshot's lemma operator when the batch has a store and the variant a description; the row for the log and the summary for the caller. A position that does
/// not parse, or is over, is an error row; the batch goes on.
fn batch_one<J: cs_tactics::ledger::Judge>(i: usize, it: &BatchItem, label: &str, judge: &J, seconds: f64, snapshot: &str, params: &cs_decision::Params, record: bool,
                                           store: Option<&Store>, ablate: &std::collections::BTreeSet<String>) -> (cs_decision::batch::Summary, serde_json::Value) {
    use cs_decision::batch;
    let label = if label.is_empty() { format!("{} {} {}", it.0, it.1.as_deref().unwrap_or("startpos"), it.2.join(" ")) } else { label.to_string() };
    let (g, s, keys) = match record_of(&it.0, it.1.as_deref(), 518, &it.2) {
        Ok(x) => x,
        Err(e) => return batch::error_row(i, snapshot, &label, &format!("{e}")),
    };
    let mut p = params.clone();
    p.history = if record { keys } else { vec![] };
    // (plan 78.36: the claims of the snapshot are read by the claim search when the batch asks for it; a snapshot of response rules only reads exactly as it did)
    let with_claims = p.claims && store.map_or(false, |st| !st.snapshot.claims.is_empty());
    let view = store.and_then(|st| if st.snapshot.lemmas.is_empty() && !with_claims { None } else { view_for(st, &it.0) });
    let decision = match (store, &view) {
        (Some(st), Some(v)) => {
            let mut op = cs_memory::op::LemmaOp::new(&st.snapshot, v);
            op.exclude = ablate.clone();
            let mut claims = cs_memory::claim::Claims::new(&st.snapshot, v);
            claims.exclude = ablate.clone();
            let lemma: Option<&dyn cs_operator::Operator<kernel::KernelGame>> = if st.snapshot.lemmas.is_empty() { None } else { Some(&op) };
            cs_decision::decide_with_store(&g, &s, seconds, judge, &p, lemma, if with_claims { Some(&claims) } else { None })
        }
        _ => cs_decision::decide_params(&g, &s, seconds, judge, &p),
    };
    match decision {
        Some(d) => batch::row(i, snapshot, &label, &g, &s, &d),
        None => batch::error_row(i, snapshot, &label, "the game is over"),
    }
}

fn load_store(path: &str, specs: HashMap<String, String>) -> PyResult<Store> {
    let snapshot = cs_memory::snapshot::Snapshot::load(std::path::Path::new(path)).map_err(pyo3::exceptions::PyValueError::new_err)?;
    let specs = specs.into_iter().map(|(k, v)| { let id = cs_memory::sha::content_id(v.as_bytes()); (k, (v, id)) }).collect();
    Ok(Store { snapshot, specs })
}

/// solve_batch(positions, seconds=1.0, worth={}, ours_first=False, valuation="", snapshot="", trace_path=None, threads=0, prior=False, record=True, prior_depth=2,
/// ledger_share=0.6, prior_defects=1, ordering=True, prune=True, extend=True, store=None, specs={}, labels=None, ablate=[], ties=False, reason=False, wdl=False,
/// wdl_table=None, claims=True)
/// -> [(move or None, proved, nodes, seconds, error or None)]: plan 78.12 / 78.14 THE BATCH SOLVER -- `positions` is [(variant, fen or None, moves)], all decided in ONE
/// call on `threads` workers (0: one per core) with the Python lock released, one summary per position in order. `seconds` is each position's own clock (positions run
/// side by side: keep `threads` within the cores, or the clocks measure the contention). `store` is a snapshot file (<id>.json, which must hash to its name): its id
/// tags the log, its judge is used when `worth` is empty, and its lemmas are bound through `specs` ({variant: description text}) by the lemma operator, whose proposals
/// the ledger search reads first (`ablate`: lemma names left out). `labels` names the positions in the log. With `trace_path` every decision's steps are appended to
/// that JSON Lines log. Plan 78.36: `claims` (default True) lets the snapshot's CLAIM rules be read by the search toward claims (operator "claim"); False leaves it out (the A / B
/// measurement; a snapshot with no claim rule decides the same either way).
#[pyfunction]
#[pyo3(signature = (positions, seconds=1.0, worth=HashMap::new(), ours_first=false, valuation=String::new(), snapshot=String::new(), trace_path=None, threads=0, prior=false, record=true, prior_depth=2, ledger_share=0.6, prior_defects=1, ordering=true, prune=true, extend=true, store=None, specs=HashMap::new(), labels=None, ablate=vec![], ties=false, reason=false, wdl=false, wdl_table=None, claims=true))]
#[allow(clippy::too_many_arguments)]
fn solve_batch(py: Python<'_>, positions: Vec<BatchItem>, seconds: f64, worth: HashMap<String, i64>, ours_first: bool, valuation: String, snapshot: String,
               trace_path: Option<String>, threads: usize, prior: bool, record: bool, prior_depth: i32, ledger_share: f64, prior_defects: u32, ordering: bool, prune: bool,
               extend: bool, store: Option<String>, specs: HashMap<String, String>, labels: Option<Vec<String>>, ablate: Vec<String>, ties: bool, reason: bool, wdl: bool, wdl_table: Option<String>, claims: bool) -> PyResult<Vec<BatchRow>> {
    let table = if wdl { load_wdl_table(wdl_table.as_deref())? } else { None };
    let params = cs_decision::Params { prior, ties, reason, wdl, wdl_table: table, claims, prior_depth, ledger_share, prior_defects, ordering, prune, extend, ..cs_decision::Params::default() };
    let store = match store { Some(p) => Some(load_store(&p, specs)?), None => None };
    let snapshot = store.as_ref().map(|s| s.snapshot.id.clone()).unwrap_or(snapshot);
    if let Some(l) = &labels { if l.len() != positions.len() { return Err(pyo3::exceptions::PyValueError::new_err("labels: one per position")); } }
    let ablate: std::collections::BTreeSet<String> = ablate.into_iter().collect();
    // the snapshot's judge stands in when the caller gives none
    let judge_worth: Option<(HashMap<String, i64>, bool, String)> = if worth.is_empty() {
        store.as_ref().and_then(|s| s.snapshot.judge.as_ref()).map(|j| (j.worth.iter().map(|(k, v)| (k.clone(), *v)).collect(), j.ours_first, j.name.clone()))
    } else { Some((worth.clone(), ours_first, valuation.clone())) };
    let lab = |i: usize| labels.as_ref().map(|l| l[i].clone()).unwrap_or_default();
    let done = py.allow_threads(|| match &judge_worth {
        None => cs_decision::batch::solve(&positions, threads, &snapshot, |i, it| batch_one(i, it, &lab(i), &cs_tactics::ledger::Count, seconds, &snapshot, &params, record, store.as_ref(), &ablate)),
        Some((w, of, name)) => {
            let j = cs_tactics::ledger::Worth { name: name.clone(), table: w.clone(), ours_first: *of };
            cs_decision::batch::solve(&positions, threads, &snapshot, |i, it| batch_one(i, it, &lab(i), &j, seconds, &snapshot, &params, record, store.as_ref(), &ablate))
        }
    });
    if let Some(path) = trace_path {
        let rows: Vec<serde_json::Value> = done.iter().map(|(_, v)| v.clone()).collect();
        cs_decision::batch::append(std::path::Path::new(&path), &rows).map_err(|e| pyo3::exceptions::PyOSError::new_err(format!("{path}: {e}")))?;
    }
    Ok(done.into_iter().map(|(s, _)| (s.mv, s.proved, s.nodes, s.seconds, s.error)).collect())
}

/// What `solve_equals_batch` returns per position: (the move the search plays, proved, nodes, seconds, error, the names of the root moves the search holds EQUAL to its best,
/// their indices in the game's own list of legal moves, the depth the ledger finished, its value, the seconds the equal reading took, the number of legal moves, whether the
/// clock cut the equal reading).
type EqualsRow = (Option<String>, bool, u64, f64, Option<String>, Vec<String>, Vec<usize>, Option<i64>, Option<i64>, f64, usize, bool);

/// PLAN 78.50: the ledger step of the decision with the equal reading kept apart and its completeness told. It is `cs_tactics::operators::LedgerOp` read again: the same `Search`
/// (history, ordering, pruning, extension, root order), the same deepening, the same move and value; afterwards the root moves are each asked "at least the best?" by a null window at
/// the finished depth with the table of the run warm (`Search::ties`, unchanged), within `share` of what is left of the decision's clock, and `Search::cut` says whether that
/// clock stopped it. A part named "ties (cut)" instead of "ties" marks a cut reading. Nothing in the search is changed.
struct EqualsLedger<'a, J: cs_tactics::ledger::Judge> { judge: &'a J, history: Vec<u64>, params: &'a cs_decision::Params }

impl<'a, G: Game, J: cs_tactics::ledger::Judge> cs_operator::Operator<G> for EqualsLedger<'a, J> {
    fn name(&self) -> String { "ledger".into() }
    fn accepts(&self, goal: &cs_operator::Goal) -> bool { matches!(goal, cs_operator::Goal::Ledger { .. }) }
    fn run(&self, g: &G, s: &G::State, goal: &cs_operator::Goal, b: cs_operator::Budget) -> cs_operator::Outcome<G::Move> {
        use cs_operator::{Outcome, Status};
        let cs_operator::Goal::Ledger { role, ties, prefer, avoid } = goal else { return Outcome::unsupported("ledger", goal) };
        let t0 = std::time::Instant::now();
        let mut ls = cs_tactics::ledger::Search::new(g, self.judge, *role, b.seconds_left());
        ls.set_fuse(b.max_nodes);
        ls.history = self.history.iter().copied().collect();
        ls.ordering = self.params.ordering;
        ls.prune = self.params.prune;
        ls.extend = self.params.extend;
        ls.root_first = prefer.clone();
        ls.root_last = avoid.clone();
        let ln = ls.run(s, b.max_depth.map_or(cs_decision::LEDGER_DEPTH, |d| d as i32));
        let moves = g.moves(s);
        let Some(i) = ln.first_index else {
            let mut o = Outcome::new(Status::Gap);
            o.evidence.push("ledger search: no line read within the clock -- the first legal move".into());
            o.cost.nodes = ln.nodes;
            o.cost.seconds = t0.elapsed().as_secs_f64();
            return o;
        };
        let mut o = Outcome::new(Status::Read);
        o.mv = Some(moves[i].clone());
        o.value = Some(ln.value);
        o.depth = Some(ln.depth as i64);
        o.line = ln.line.clone();
        o.events = ln.lost.clone();
        let mut nodes = ln.nodes;
        if let Some(share) = ties {
            if ln.value.abs() < cs_tactics::ledger::WIN - 1000 {
                ls.resume(b.rest_of_decision() * share);
                let n0 = ls.nodes;
                let t1 = std::time::Instant::now();
                o.candidates = ls.ties(s, ln.depth, ln.value);
                nodes += ls.nodes - n0;
                o.part(if ls.cut() { "ties (cut)" } else { "ties" }, ls.nodes - n0, t1.elapsed().as_secs_f64());
            }
        }
        o.cost.nodes = nodes;
        o.cost.seconds = t0.elapsed().as_secs_f64();
        o
    }
}

/// The fixed controller with one change: where it asks the ledger for its ties, the equal reading may take all that is left of the clock after the deepening (the ledger's own
/// clock being `ledger_share` of the decision's). The proof, the claim step and the lemma order are the fixed controller's, untouched.
struct EqualsController<'p>(cs_decision::Fixed<'p>);

impl<'p, G: Game> cs_operator::Controller<G> for EqualsController<'p> {
    fn choose(&self, ctx: &cs_operator::Context<G>) -> cs_operator::Action<G::Move> {
        match self.0.choose(ctx) {
            cs_operator::Action::Run { operator, goal: cs_operator::Goal::Ledger { role, ties: Some(_), prefer, avoid }, seconds, max_nodes, max_depth } =>
                cs_operator::Action::Run { operator, goal: cs_operator::Goal::Ledger { role, ties: Some(1.0), prefer, avoid }, seconds, max_nodes, max_depth },
            other => other,
        }
    }
}

/// The decision of `batch_one` (proof, claims, lemma order, then the ledger) with the equal reading on and its candidates handed back by NAME and INDEX. The move played is the
/// ledger's move, as the engine plays it. The moves listed are those whose probe held the best value and the ledger's own move (its value IS the best); where the line decides the
/// game the reading is skipped and the list is that move alone.
fn equals_one<J: cs_tactics::ledger::Judge>(i: usize, it: &BatchItem, judge: &J, seconds: f64, params: &cs_decision::Params, record: bool, store: Option<&Store>)
                                            -> (cs_decision::batch::Summary, serde_json::Value) {
    use cs_decision::batch;
    let label = format!("{} {} {}", it.0, it.1.as_deref().unwrap_or("startpos"), it.2.join(" "));
    let (g, s, keys) = match record_of(&it.0, it.1.as_deref(), 518, &it.2) {
        Ok(x) => x,
        Err(e) => return batch::error_row(i, "", &label, &format!("{e}")),
    };
    let mut p = params.clone();
    p.history = if record { keys } else { vec![] };
    let moves = Game::moves(&g, &s);
    if moves.is_empty() { return batch::error_row(i, "", &label, "the game is over"); }
    let me = match Game::mover(&g, &s) { Some(m) => m, None => return batch::error_row(i, "", &label, "the game is over") };
    let trace = vec![format!("decide: {} to move, {} legal moves, {:.2} s", Game::roles(&g)[me], moves.len(), seconds)];
    let with_claims = p.claims && store.map_or(false, |st| !st.snapshot.claims.is_empty());
    let view = store.and_then(|st| if st.snapshot.lemmas.is_empty() && !with_claims { None } else { view_for(st, &it.0) });
    let forcing = cs_tactics::operators::ForcingOp;
    let ledger = EqualsLedger { judge, history: p.history.clone(), params: &p };
    let d = match (store, &view) {
        (Some(st), Some(v)) => {
            let op = cs_memory::op::LemmaOp::new(&st.snapshot, v);
            let claims = cs_memory::claim::Claims::new(&st.snapshot, v);
            let lemma: Option<&dyn cs_operator::Operator<kernel::KernelGame>> = if st.snapshot.lemmas.is_empty() { None } else { Some(&op) };
            let claim_op = if with_claims && !claims.is_empty() { Some(cs_decision::claim_op::ClaimOp { claims: &claims, history: p.history.clone() }) } else { None };
            let mut ops: Vec<&dyn cs_operator::Operator<kernel::KernelGame>> = vec![&forcing, &ledger];
            if let Some(l) = lemma { ops.push(l); }
            if let Some(c) = &claim_op { ops.push(c); }
            cs_operator::drive(&g, &s, seconds, &ops, &EqualsController(cs_decision::Fixed { params: &p, lemmas: lemma.is_some(), claims: claim_op.is_some() }), trace)
        }
        _ => {
            let ops: Vec<&dyn cs_operator::Operator<kernel::KernelGame>> = vec![&forcing, &ledger];
            cs_operator::drive(&g, &s, seconds, &ops, &EqualsController(cs_decision::Fixed { params: &p, lemmas: false, claims: false }), trace)
        }
    };
    if d.mv.is_none() { return batch::error_row(i, "", &label, "no move"); }
    let (summary, mut v) = batch::row(i, "", &label, &g, &s, &d);
    let step = d.steps.iter().rev().find(|st| st.operator == "ledger");
    let mut idx: Vec<usize> = step.map(|st| st.outcome.candidates.clone()).unwrap_or_default();
    if let Some(m) = &d.mv { if let Some(k) = moves.iter().position(|x| x == m) { idx.push(k); } }
    idx.sort();
    idx.dedup();
    let names: Vec<String> = idx.iter().filter_map(|k| moves.get(*k)).map(|m| Game::move_name(&g, &s, m)).collect();
    let tie_parts: Vec<&cs_operator::Part> = step.map(|st| st.outcome.parts.iter().filter(|p| p.name.starts_with("ties")).collect()).unwrap_or_default();
    v["equals"] = serde_json::json!(names);
    v["equal_indices"] = serde_json::json!(idx);
    v["legal"] = serde_json::json!(moves.len());
    v["ledger_depth"] = serde_json::json!(step.and_then(|st| st.outcome.depth));
    v["ledger_value"] = serde_json::json!(step.and_then(|st| st.outcome.value));
    v["ties_seconds"] = serde_json::json!(tie_parts.iter().map(|p| p.cost.seconds).sum::<f64>());
    v["ties_cut"] = serde_json::json!(tie_parts.iter().any(|p| p.name == "ties (cut)"));
    (summary, v)
}

/// solve_equals_batch(positions, seconds=0.2, worth={}, ours_first=False, valuation="", threads=0, store=None, specs={}, record=True, ledger_share=0.8, claims=True)
/// -> [(move or None, proved, nodes, seconds, error or None, equal move names, equal move indices, ledger depth, ledger value, equal-reading seconds, legal moves, cut)]. Plan 78.50:
/// the batch decision (as `solve_batch`: proof, claims, lemma order, the ledger search, the same move) with the root moves the search holds EQUAL to its best handed back. The ledger
/// deepens within `ledger_share` of the clock and the equal reading takes what is left of it (so the whole decision stays within `seconds`); `cut` says the clock stopped the reading
/// (the list is then only part of the equals: every move in it held the best). `positions`, `seconds`, `worth`, `store`, `specs` and `threads` as in `solve_batch`.
#[pyfunction]
#[pyo3(signature = (positions, seconds=0.2, worth=HashMap::new(), ours_first=false, valuation=String::new(), threads=0, store=None, specs=HashMap::new(), record=true, ledger_share=0.8, claims=true))]
#[allow(clippy::too_many_arguments)]
fn solve_equals_batch(py: Python<'_>, positions: Vec<BatchItem>, seconds: f64, worth: HashMap<String, i64>, ours_first: bool, valuation: String, threads: usize, store: Option<String>,
                      specs: HashMap<String, String>, record: bool, ledger_share: f64, claims: bool) -> PyResult<Vec<EqualsRow>> {
    let params = cs_decision::Params { ties: true, claims, ledger_share, ..cs_decision::Params::default() };
    let store = match store { Some(p) => Some(load_store(&p, specs)?), None => None };
    let judge_worth: Option<(HashMap<String, i64>, bool, String)> = if worth.is_empty() {
        store.as_ref().and_then(|s| s.snapshot.judge.as_ref()).map(|j| (j.worth.iter().map(|(k, v)| (k.clone(), *v)).collect(), j.ours_first, j.name.clone()))
    } else { Some((worth.clone(), ours_first, valuation.clone())) };
    let done = py.allow_threads(|| match &judge_worth {
        None => cs_decision::batch::solve(&positions, threads, "", |i, it| equals_one(i, it, &cs_tactics::ledger::Count, seconds, &params, record, store.as_ref())),
        Some((w, of, name)) => {
            let j = cs_tactics::ledger::Worth { name: name.clone(), table: w.clone(), ours_first: *of };
            cs_decision::batch::solve(&positions, threads, "", |i, it| equals_one(i, it, &j, seconds, &params, record, store.as_ref()))
        }
    });
    Ok(done.into_iter().map(|(s, v)| {
        let names: Vec<String> = v["equals"].as_array().map(|a| a.iter().filter_map(|x| x.as_str().map(|t| t.to_string())).collect()).unwrap_or_default();
        let idx: Vec<usize> = v["equal_indices"].as_array().map(|a| a.iter().filter_map(|x| x.as_u64().map(|t| t as usize)).collect()).unwrap_or_default();
        (s.mv, s.proved, s.nodes, s.seconds, s.error, names, idx, v["ledger_depth"].as_i64(), v["ledger_value"].as_i64(), v["ties_seconds"].as_f64().unwrap_or(0.0), v["legal"].as_u64().unwrap_or(0) as usize,
         v["ties_cut"].as_bool().unwrap_or(false))
    }).collect())
}

fn vocab_of(json: &str) -> PyResult<cs_game::Vocabulary> {
    let v: serde_json::Value = serde_json::from_str(json).map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("vocabulary: {e}")))?;
    cs_memory::snapshot::vocabulary_from_json(&v).map_err(pyo3::exceptions::PyValueError::new_err)
}

/// facts(variant, description, vocabulary, fen=None, moves=[], perspective=True) -> [[relation, arg ...]]: the FACT LAYER of a position -- the kernel's base facts
/// (certified against the description by co-play) and the derived relations the vocabulary (JSON) names, evaluated through the description; in the mover's
/// perspective (`us` / `them`) by default.
#[pyfunction]
#[pyo3(signature = (variant, spec, vocabulary, fen=None, moves=vec![], perspective=true, demand=true))]
fn facts(variant: &str, spec: &str, vocabulary: &str, fen: Option<&str>, moves: Vec<String>, perspective: bool, demand: bool) -> PyResult<Vec<Vec<String>>> {
    let (g, s) = state_of(variant, fen, 518, &moves)?;
    let view = facts_view_for(spec, vocabulary, demand)?;
    let base = g.base_facts(&s);
    if base.is_empty() { return Err(pyo3::exceptions::PyValueError::new_err(format!("{variant} gives no facts (no description is certified for it)"))); }
    let mut fs = view.facts(&base).map_err(pyo3::exceptions::PyValueError::new_err)?;
    if perspective { if let Some(me) = g.mover(&s) { fs = cs_memory::perspective(&fs, &g.roles(), me); } }
    Ok(fs.into_iter().map(|f| { let mut v = vec![f.relation]; v.extend(f.args); v }).collect())
}

thread_local! {
    /// One fact view per (description, vocabulary, demand) for `facts`: loading a description (its statics and their indexes) costs more than reading a position through it (plan 78.19 A).
    static FACT_VIEWS: std::cell::RefCell<HashMap<String, std::rc::Rc<cs_game::gdl_game::FactView>>> = std::cell::RefCell::new(HashMap::new());
}

fn facts_view_for(spec: &str, vocabulary: &str, demand: bool) -> PyResult<std::rc::Rc<cs_game::gdl_game::FactView>> {
    let key = format!("{}|{vocabulary}|{demand}", cs_memory::sha::sha256_hex(spec.as_bytes()));
    FACT_VIEWS.with(|v| {
        if let Some(x) = v.borrow().get(&key) { return Ok(x.clone()); }
        let view = std::rc::Rc::new(cs_game::gdl_game::FactView::new(spec, vocab_of(vocabulary)?).map_err(pyo3::exceptions::PyValueError::new_err)?.with_demand(demand));
        v.borrow_mut().insert(key, view.clone());
        Ok(view)
    })
}

/// facts_audit(variant, spec, vocabulary, games, plies, seed, demand=True) -> dict: PLAN 78.19 A's AUDIT. `games` seeded random games of the variant's kernel, at most
/// `plies` plies each; at every position the fact view with demand evaluation (`demand`, else the full run on both sides) is compared with the full evaluation (the
/// oracle). {positions, equal, unequal, ms_per_position_full, ms_per_position_demand, cone_relations, cone_clauses, relations, clauses}.
#[pyfunction]
#[pyo3(signature = (variant, spec, vocabulary, games, plies, seed, demand=true))]
fn facts_audit(variant: &str, spec: &str, vocabulary: &str, games: usize, plies: usize, seed: u64, demand: bool) -> PyResult<HashMap<String, f64>> {
    let g = kernel::KernelGame::new(variant, 518).map_err(pyo3::exceptions::PyValueError::new_err)?;
    let vocab = vocab_of(vocabulary)?;
    let mk = |d: bool| cs_game::gdl_game::FactView::new(spec, vocab.clone()).map(|v| v.with_demand(d)).map_err(pyo3::exceptions::PyValueError::new_err);
    let (full, dem) = (mk(false)?, mk(demand)?);
    let pos = cs_game::fact_audit::positions(&g, games, plies, seed);
    let r = cs_game::fact_audit::audit(&full, &dem, &pos).map_err(pyo3::exceptions::PyValueError::new_err)?;
    let ((cr, cc), (tr, tc)) = dem.cone_size();
    Ok(HashMap::from([("positions".to_string(), r.positions as f64), ("equal".to_string(), r.equal as f64), ("unequal".to_string(), r.unequal as f64),
        ("ms_per_position_full".to_string(), r.ms_per_position_full()), ("ms_per_position_demand".to_string(), r.ms_per_position_demand()),
        ("cone_relations".to_string(), cr as f64), ("cone_clauses".to_string(), cc as f64), ("relations".to_string(), tr as f64), ("clauses".to_string(), tc as f64)]))
}

/// The product of two priors' weights (the gain prior and the lemma prior together).
#[derive(Clone)]
struct ProductPrior<A: cs_planning::rollout::Prior + Clone, B: cs_planning::rollout::Prior + Clone>(A, B);
impl<A: cs_planning::rollout::Prior + Clone, B: cs_planning::rollout::Prior + Clone> cs_planning::rollout::Prior for ProductPrior<A, B> {
    fn weights<G: Game>(&self, g: &G, s: &G::State, moves: &[G::Move]) -> Vec<f64> {
        self.0.weights(g, s, moves).into_iter().zip(self.1.weights(g, s, moves)).map(|(a, b)| a * b).collect()
    }
    fn name(&self) -> String { format!("{} x {}", self.0.name(), self.1.name()) }
}

/// The rollout and leaf names the operator's evidence line states ("... rollout <name>, leaf <name>, value for role ...").
fn playout_names(evidence: &[String]) -> (String, String) {
    for e in evidence {
        if let (Some(a), Some(b), Some(c)) = (e.find(", rollout "), e.find(", leaf "), e.find(", value for role")) {
            if a < b && b < c { return (e[a + 10..b].to_string(), e[b + 7..c].to_string()); }
        }
    }
    (String::new(), String::new())
}

/// What `estimate` was asked of the playouts, as read from its keywords.
struct PlayoutSpec<'a> { rollout: &'a str, leaf: &'a str, eps: f64, scale: f64, temperature: f64, floor: f64, prior: bool, boost: f64, seed: u64, exploration: f64, max_playout: usize }

fn estimate_leaf<G: Game, R: cs_planning::mcts::Rollout + Clone, J: cs_tactics::ledger::Judge + Clone>(g: &G, s: &G::State, me: usize, p: &PlayoutSpec, rollout: R, judge: J,
                                                                                                       budget: cs_operator::Budget) -> Result<cs_operator::Outcome<G::Move>, String> {
    use cs_operator::Operator;
    let goal = cs_operator::Goal::Estimate { role: me };
    match p.leaf {
        "draw" => Ok(cs_planning::mcts::MctsOp::with_policy(p.exploration, p.max_playout, rollout, cs_planning::rollout::Draw).run(g, s, &goal, budget)),
        "balance" => Ok(cs_planning::mcts::MctsOp::with_policy(p.exploration, p.max_playout, rollout, cs_planning::rollout::Balance::new(judge, p.scale)).run(g, s, &goal, budget)),
        other => Err(format!("leaf: \"{other}\" is not draw or balance")),
    }
}

fn estimate_with<G: Game, J: cs_tactics::ledger::Judge + Clone>(g: &G, s: &G::State, me: usize, p: &PlayoutSpec, judge: J, lemmas: Option<cs_decision::LemmaPrior>,
                                                                budget: cs_operator::Budget) -> Result<cs_operator::Outcome<G::Move>, String> {
    use cs_planning::rollout::{GainPrior, Greedy, Weighted};
    match (p.rollout, p.prior, lemmas) {
        ("uniform", false, _) => estimate_leaf(g, s, me, p, cs_planning::mcts::Uniform::new(p.seed), judge, budget),
        ("greedy", false, _) => estimate_leaf(g, s, me, p, Greedy::new(judge.clone(), p.eps, p.seed), judge, budget),
        ("gain", false, _) => estimate_leaf(g, s, me, p, Weighted::new(GainPrior { judge: judge.clone(), temperature: p.temperature }, p.floor, p.seed), judge, budget),
        ("lemma", _, Some(lp)) | ("uniform", true, Some(lp)) => estimate_leaf(g, s, me, p, Weighted::new(lp, p.floor, p.seed), judge, budget),
        ("gain", true, Some(lp)) => estimate_leaf(g, s, me, p, Weighted::new(ProductPrior(GainPrior { judge: judge.clone(), temperature: p.temperature }, lp), p.floor, p.seed), judge, budget),
        ("greedy", true, _) => Err("prior: a lemma prior combines with rollout \"uniform\" or \"gain\" (or use rollout \"lemma\"), not \"greedy\"".into()),
        ("lemma", _, None) | (_, true, None) => Err("the lemma prior needs store and spec (the snapshot path and the variant's description text)".into()),
        (other, _, _) => Err(format!("rollout: \"{other}\" is not uniform, greedy, gain or lemma")),
    }
}

/// estimate(variant, fen, moves, seconds, max_nodes, seed, exploration, max_playout, rollout="uniform", leaf="draw", eps=0.1, scale=3.0, temperature=1.0, store=None,
/// prior=False, floor=0.05, spec=None, boost=4.0) -> {move, value, candidates, visits, playouts, cut, depth, nodes, seconds, rollout, leaf}: the
/// Monte Carlo tree search operator's ESTIMATE of the position for the side to move (sampled playouts: an estimate, never a proof). `candidates` are the legal moves in
/// visit order (most visited first), `visits` one row per root move (name, visits, mean outcome in hundredths of an outcome unit), `value` the root's mean for the
/// mover in the same units. One source of a ranking among others (the teacher is another); game-free behind the Operator interface.
/// PLAYOUTS: `rollout` is "uniform" (random moves), "greedy" (with probability `eps` random, else the move of best immediate gain), "gain" (sampled by exp(gain /
/// `temperature`) plus `floor`) or "lemma" (sampled by the snapshot's lemma prior: `boost` on the moves a bound lemma names, 1 on the rest, plus `floor`). `leaf` is what
/// a playout cut at `max_playout` scores: "draw" (50 for every role) or "balance" (the units' balance by the judge, squashed by `scale` worth units). The judge is the
/// `store` snapshot's judge record (its worths) or, without one, a count (every unit worth 1). `prior=True` (with `store` and `spec`, the variant's description text)
/// multiplies the weights of "uniform" or "gain" by the lemma prior's.
#[pyfunction]
#[pyo3(signature = (variant, fen=None, moves=vec![], seconds=0.2, max_nodes=10_000_000, seed=1, exploration=cs_planning::mcts::EXPLORATION, max_playout=cs_planning::mcts::MAX_PLAYOUT,
                    rollout="uniform", leaf="draw", eps=0.1, scale=3.0, temperature=1.0, store=None, prior=false, floor=0.05, spec=None, boost=4.0))]
#[allow(clippy::too_many_arguments)]
fn estimate(py: Python<'_>, variant: &str, fen: Option<&str>, moves: Vec<String>, seconds: f64, max_nodes: u64, seed: u64, exploration: f64, max_playout: usize, rollout: &str,
            leaf: &str, eps: f64, scale: f64, temperature: f64, store: Option<&str>, prior: bool, floor: f64, spec: Option<&str>, boost: f64) -> PyResult<PyObject> {
    let (g, s) = state_of(variant, fen, 518, &moves)?;
    let Some(me) = g.mover(&s) else { return Err(pyo3::exceptions::PyValueError::new_err("the game is over")) };
    let p = PlayoutSpec { rollout, leaf, eps, scale, temperature, floor, prior, boost, seed, exploration, max_playout };
    let snapshot = match store { Some(path) => Some(cs_memory::snapshot::Snapshot::load(std::path::Path::new(path)).map_err(pyo3::exceptions::PyValueError::new_err)?), None => None };
    let o = py.allow_threads(|| -> Result<_, String> {
        let budget = cs_operator::Budget::new(seconds, max_nodes);
        // (the fact view is built here: it is not sent between threads)
        let view = match (&snapshot, spec) { (Some(sn), Some(text)) if p.prior || p.rollout == "lemma" => Some(cs_game::gdl_game::FactView::new(text, sn.vocabulary.clone())?), _ => None };
        let lemmas = match (&snapshot, &view) { (Some(sn), Some(v)) => Some(cs_decision::LemmaPrior::new(sn, v, p.boost)), _ => None };
        match snapshot.as_ref().and_then(|sn| sn.judge.as_ref()) {
            Some(j) => estimate_with(&g, &s, me, &p, cs_tactics::ledger::Worth { name: j.name.clone(), table: j.worth.iter().map(|(k, v)| (k.clone(), *v)).collect(), ours_first: j.ours_first }, lemmas, budget),
            None => estimate_with(&g, &s, me, &p, cs_tactics::ledger::Count, lemmas, budget),
        }
    }).map_err(pyo3::exceptions::PyValueError::new_err)?;
    let legal = g.moves(&s);
    let name = |i: usize| g.move_name(&s, &legal[i]);
    let d = pyo3::types::PyDict::new(py);
    d.set_item("status", format!("{:?}", o.status))?;
    let (rollout_name, leaf_name) = playout_names(&o.evidence);
    d.set_item("rollout", rollout_name)?;
    d.set_item("leaf", leaf_name)?;
    d.set_item("move", o.mv.as_ref().map(|m| g.move_name(&s, m)))?;
    d.set_item("value", o.value)?;
    d.set_item("depth", o.depth)?;
    d.set_item("candidates", o.candidates.iter().map(|&i| name(i)).collect::<Vec<_>>())?;
    let mut visits = vec![];
    let (mut playouts, mut cut) = (0i64, 0i64);
    for f in &o.facts {
        match f.relation.as_str() {
            "visits" => if let (Ok(i), Ok(v), Ok(m)) = (f.args[0].parse::<usize>(), f.args[1].parse::<i64>(), f.args[2].parse::<i64>()) { visits.push((name(i), v, m)); },
            "playouts" => { playouts = f.args[0].parse().unwrap_or(0); cut = f.args[1].parse().unwrap_or(0); }
            _ => {}
        }
    }
    d.set_item("visits", visits)?;
    d.set_item("playouts", playouts)?;
    d.set_item("cut", cut)?;
    d.set_item("nodes", o.cost.nodes)?;
    d.set_item("seconds", o.cost.seconds)?;
    Ok(d.into())
}

/// One playout from `s` to a terminal state (or `max_plies`) under `policy`: the move names played, the outcome per role, whether the cap was hit.
fn play_out<G: Game, R: cs_planning::mcts::Rollout>(g: &G, s: &G::State, policy: &mut R, max_plies: usize) -> (Vec<String>, Option<Vec<i64>>, bool) {
    let mut s = s.clone();
    let mut names = vec![];
    loop {
        if let Some(o) = g.outcome(&s) { return (names, Some(o), false); }
        let moves = g.moves(&s);
        if moves.is_empty() { return (names, None, false); }
        if names.len() >= max_plies { return (names, None, true); }
        let i = policy.pick(g, &s, &moves).min(moves.len() - 1);
        names.push(g.move_name(&s, &moves[i]));
        s = g.apply(&s, &moves[i]);
    }
}

/// playout_game(variant, fen=None, moves=[], seed=1, max_plies=400, rollout="careful", store=None, spec=None, worth={}, valuation="", eps=0.1, boost=2.0) ->
/// {moves, outcome, capped, named, plies}: ONE playout from the position after `moves` to a terminal state (`outcome`: the per-role list, None when `capped` -- a
/// guard in plies -- or the game has no move), its moves kept. `rollout` "careful" is cs_planning's two-ply look; "rules" plays one of the moves the store's lemmas name
/// (uniformly among them) when any lemma speaks, else the careful look (cs_decision::RulesFirst; needs `store`, the snapshot path, and `spec`, the variant's description
/// text); with probability `eps` a uniform move. `named` counts the plies where a rule spoke. The judge is `worth` when given, else the snapshot's judge, else a count.
#[pyfunction]
#[pyo3(signature = (variant, fen=None, moves=vec![], seed=1, max_plies=400, rollout="careful", store=None, spec=None, worth=HashMap::new(), valuation=String::new(), eps=cs_decision::WDL_EPS, boost=2.0))]
#[allow(clippy::too_many_arguments)]
fn playout_game(py: Python<'_>, variant: &str, fen: Option<&str>, moves: Vec<String>, seed: u64, max_plies: usize, rollout: &str, store: Option<&str>, spec: Option<&str>,
                worth: HashMap<String, i64>, valuation: String, eps: f64, boost: f64) -> PyResult<PyObject> {
    if rollout != "careful" && rollout != "rules" { return Err(pyo3::exceptions::PyValueError::new_err(format!("rollout: {rollout} is not careful or rules"))); }
    let (g, s) = state_of(variant, fen, 518, &moves)?;
    let snapshot = match store { Some(path) => Some(cs_memory::snapshot::Snapshot::load(std::path::Path::new(path)).map_err(pyo3::exceptions::PyValueError::new_err)?), None => None };
    let judge: Option<cs_tactics::ledger::Worth> = if !worth.is_empty() { Some(cs_tactics::ledger::Worth { name: valuation, table: worth.clone(), ours_first: false }) }
        else { snapshot.as_ref().and_then(|sn| sn.judge.as_ref()).map(|j| cs_tactics::ledger::Worth { name: j.name.clone(), table: j.worth.iter().map(|(k, v)| (k.clone(), *v)).collect(), ours_first: j.ours_first }) };
    let rules = rollout == "rules";
    if rules && (snapshot.is_none() || spec.is_none()) { return Err(pyo3::exceptions::PyValueError::new_err("rollout \"rules\" needs store and spec (the snapshot path and the variant's description text)")); }
    let r = py.allow_threads(|| -> Result<_, String> {
        fn run<G: Game, J: cs_tactics::ledger::Judge + Clone>(g: &G, s: &G::State, judge: J, rules: Option<(&cs_memory::snapshot::Snapshot, &str)>, boost: f64, eps: f64, seed: u64, max_plies: usize)
            -> Result<(Vec<String>, Option<Vec<i64>>, bool, u64), String> {
            let careful = cs_planning::rollout::Careful::new(judge, eps, seed);
            match rules {
                None => { let mut p = careful; let (m, o, c) = play_out(g, s, &mut p, max_plies); Ok((m, o, c, 0)) }
                Some((sn, text)) => {
                    let view = cs_game::gdl_game::FactView::new(text, sn.vocabulary.clone())?;
                    let mut p = cs_decision::RulesFirst::new(cs_decision::LemmaPrior::new(sn, &view, boost), careful, eps, seed);
                    let (m, o, c) = play_out(g, s, &mut p, max_plies);
                    Ok((m, o, c, p.named))
                }
            }
        }
        let rules_in = match (&snapshot, spec) { (Some(sn), Some(t)) if rules => Some((sn, t)), _ => None };
        match judge { Some(j) => run(&g, &s, j, rules_in, boost.max(1.5), eps, seed, max_plies), None => run(&g, &s, cs_tactics::ledger::Count, rules_in, boost.max(1.5), eps, seed, max_plies) }
    }).map_err(pyo3::exceptions::PyValueError::new_err)?;
    let (names, outcome, capped, named) = r;
    let d = pyo3::types::PyDict::new(py);
    d.set_item("plies", names.len())?;
    d.set_item("moves", names)?;
    d.set_item("outcome", outcome)?;
    d.set_item("capped", capped)?;
    d.set_item("named", named)?;
    Ok(d.into())
}

/// A corpus of cards (positions with their facts and the exact value of every legal move) lemmas are verified on.
#[pyclass]
struct CardSet { inner: std::sync::Arc<cs_memory::cards::CardSet> }

fn lemma_of(text: &str) -> PyResult<cs_memory::lemma::Lemma> {
    let v: serde_json::Value = serde_json::from_str(text).map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("lemma: {e}")))?;
    cs_memory::lemma::Lemma::from_json(&v).map_err(pyo3::exceptions::PyValueError::new_err)
}

fn eval_dict(py: Python<'_>, ev: &cs_memory::cards::Eval) -> PyResult<PyObject> {
    let d = pyo3::types::PyDict::new(py);
    d.set_item("fires", ev.fires)?; d.set_item("holds", ev.holds)?; d.set_item("refuted", ev.refuted)?; d.set_item("unknown", ev.unknown)?;
    d.set_item("groups", ev.groups)?; d.set_item("groups_held", ev.groups_held)?; d.set_item("precision", ev.precision)?; d.set_item("expected", ev.expected)?;
    d.set_item("excess", ev.excess)?; d.set_item("selectivity", ev.selectivity)?; d.set_item("truncated", ev.truncated)?; d.set_item("stopped", ev.stopped)?; d.set_item("steps", ev.steps)?;
    d.set_item("cases", ev.cases.clone())?; d.set_item("trivial", ev.trivial)?;
    Ok(d.into())
}

impl CardSet {
    fn check_indices(&self, indices: &Option<Vec<usize>>) -> PyResult<()> {
        if let Some(ix) = indices { if ix.iter().any(|i| *i >= self.inner.len()) { return Err(pyo3::exceptions::PyIndexError::new_err("a card index is out of range")); } }
        Ok(())
    }
}

#[pymethods]
impl CardSet {
    fn __len__(&self) -> usize { self.inner.len() }
    /// The card at `i` as JSON text.
    fn card(&self, i: usize) -> PyResult<String> {
        self.inner.cards.get(i).map(|c| c.to_json().to_string()).ok_or_else(|| pyo3::exceptions::PyIndexError::new_err("no such card"))
    }
    /// eval(lemma_json, indices=None, max_bindings=64, max_refuted=None) -> {fires, holds, refuted, unknown, groups, groups_held, precision, expected, excess, selectivity, truncated, stopped, steps, cases}; with `max_refuted` it stops at the first card past that many refutations (partial counts, `stopped`).
    #[pyo3(signature = (lemma, indices=None, max_bindings=64, max_refuted=None, all_named=false))]
    fn eval(&self, py: Python<'_>, lemma: &str, indices: Option<Vec<usize>>, max_bindings: usize, max_refuted: Option<usize>, all_named: bool) -> PyResult<PyObject> {
        let l = lemma_of(lemma)?;
        self.check_indices(&indices)?;
        let set = self.inner.clone();
        let ev = py.allow_threads(|| set.eval_with(&l, indices.as_deref(), max_bindings, max_refuted, all_named));
        eval_dict(py, &ev)
    }
    /// eval_many([lemma_json], indices=None, max_bindings=64) -> [the eval dict of each]: the lemmas read side by side on the cores, in one call (the climb reads every
    /// candidate step at once). A lemma that is not well formed is an error for the whole call.
    #[pyo3(signature = (lemmas, indices=None, max_bindings=64, max_refuted=None, all_named=false))]
    fn eval_many(&self, py: Python<'_>, lemmas: Vec<String>, indices: Option<Vec<usize>>, max_bindings: usize, max_refuted: Option<usize>, all_named: bool) -> PyResult<Vec<PyObject>> {
        let ls = lemmas.iter().map(|l| lemma_of(l)).collect::<PyResult<Vec<_>>>()?;
        self.check_indices(&indices)?;
        let set = self.inner.clone();
        let evs: Vec<cs_memory::cards::Eval> = py.allow_threads(|| {
            use rayon::prelude::*;
            ls.par_iter().map(|l| set.eval_with(l, indices.as_deref(), max_bindings, max_refuted, all_named)).collect()
        });
        evs.iter().map(|e| eval_dict(py, e)).collect()
    }
    /// binds(rules, indices=None, max_bindings=64) -> [{at: [card index], truncated, steps}] (plan 78.36): where each rule's BODY binds on the cards (a claim or a lemma; only the body is read),
    /// by the same matcher and fuse as `eval`, side by side on the cores. The claim drill reads the corpus row's exact class at each of these cards.
    #[pyo3(signature = (rules, indices=None, max_bindings=64))]
    fn binds(&self, py: Python<'_>, rules: Vec<String>, indices: Option<Vec<usize>>, max_bindings: usize) -> PyResult<Vec<PyObject>> {
        let bodies = rules.iter().map(|r| {
            let v: serde_json::Value = serde_json::from_str(r).map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("rule: {e}")))?;
            match cs_memory::claim::Rule::from_json(&v).map_err(pyo3::exceptions::PyValueError::new_err)? { cs_memory::claim::Rule::Response(l) => Ok(l.body), cs_memory::claim::Rule::Claim(c) => Ok(c.body) }
        }).collect::<PyResult<Vec<_>>>()?;
        self.check_indices(&indices)?;
        let set = self.inner.clone();
        let out: Vec<cs_memory::cards::Binds> = py.allow_threads(|| {
            use rayon::prelude::*;
            bodies.par_iter().map(|b| set.binds(b, indices.as_deref(), max_bindings)).collect()
        });
        out.iter().map(|b| { let d = pyo3::types::PyDict::new(py); d.set_item("at", b.at.clone())?; d.set_item("truncated", b.truncated)?; d.set_item("steps", b.steps)?; Ok(d.into()) }).collect()
    }
    /// Append the cards to a JSON Lines file.
    fn save(&self, path: &str) -> PyResult<()> {
        let rows: Vec<serde_json::Value> = self.inner.cards.iter().map(|c| c.to_json()).collect();
        cs_decision::batch::append(std::path::Path::new(path), &rows).map_err(|e| pyo3::exceptions::PyOSError::new_err(format!("{path}: {e}")))
    }
    #[staticmethod]
    fn load(path: &str) -> PyResult<CardSet> {
        let text = std::fs::read_to_string(path).map_err(|e| pyo3::exceptions::PyOSError::new_err(format!("{path}: {e}")))?;
        let cards = text.lines().filter(|l| !l.trim().is_empty()).map(|l| {
            let v: serde_json::Value = serde_json::from_str(l).map_err(|e| e.to_string())?;
            cs_memory::cards::Card::from_json(&v)
        }).collect::<Result<Vec<_>, String>>().map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(CardSet { inner: std::sync::Arc::new(cs_memory::cards::CardSet::new(cards)) })
    }
    /// revalued(values) -> CardSet: the same cards in the same order; for card i with Some({move name: value}) every move's value is replaced by the map's (a move the map
    /// lacks makes the card unusable, so it is marked `cut` rather than given a guessed value); None keeps the card. The list must have one entry per card.
    fn revalued(&self, values: Vec<Option<HashMap<String, i64>>>) -> PyResult<CardSet> {
        if values.len() != self.inner.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!("revalued: {} value maps for {} cards", values.len(), self.inner.len())));
        }
        let cards = self.inner.cards.iter().zip(values.iter()).map(|(c, v)| {
            let mut c = c.clone();
            if let Some(map) = v {
                for m in c.moves.iter_mut() {
                    match map.get(&m.name) { Some(x) => m.value = *x, None => c.cut = true }
                }
            }
            c
        }).collect();
        Ok(CardSet { inner: std::sync::Arc::new(cs_memory::cards::CardSet::new(cards)) })
    }
    /// The cards of several sets, in order, as one.
    #[staticmethod]
    fn merge(sets: Vec<PyRef<CardSet>>) -> CardSet {
        let cards = sets.iter().flat_map(|s| s.inner.cards.iter().cloned()).collect();
        CardSet { inner: std::sync::Arc::new(cs_memory::cards::CardSet::new(cards)) }
    }
}

/// build_cards(items, specs, vocabulary, depth=3, fuse=50_000_000, threads=0, worth={}, ours_first=False) -> CardSet: the cards of the positions [(variant, fen or None,
/// moves, id, group)] -- facts in the mover's perspective, every legal move's effect and exact value at `depth` (the plain minimax of the judge, no pruning), the nodes
/// of one card's reading capped by `fuse` (a cut card says so and is never used to refute). `specs` is {variant: description text}.
#[pyfunction]
#[pyo3(signature = (items, specs, vocabulary, depth=3, fuse=50_000_000, threads=0, worth=HashMap::new(), ours_first=false))]
#[allow(clippy::too_many_arguments)]
fn build_cards(py: Python<'_>, items: Vec<(String, Option<String>, Vec<String>, String, String)>, specs: HashMap<String, String>, vocabulary: &str, depth: i32, fuse: u64,
               threads: usize, worth: HashMap<String, i64>, ours_first: bool) -> PyResult<CardSet> {
    let vocab = vocab_of(vocabulary)?;
    let judged = !worth.is_empty();
    let w = cs_tactics::ledger::Worth { name: "cards".into(), table: worth, ours_first };
    let one = |it: &(String, Option<String>, Vec<String>, String, String)| -> Result<Option<cs_memory::cards::Card>, String> {
        let spec = specs.get(&it.0).ok_or_else(|| format!("no description for {}", it.0))?;
        let (g, s) = state_of(&it.0, it.1.as_deref(), 518, &it.2).map_err(|e| e.to_string())?;
        thread_local! { static V: std::cell::RefCell<HashMap<String, std::rc::Rc<cs_game::gdl_game::FactView>>> = std::cell::RefCell::new(HashMap::new()); }
        let key = format!("{}|{}|{:?}", it.0, cs_memory::sha::content_id(spec.as_bytes()), vocab);
        let view = V.with(|m| -> Result<_, String> {
            if let Some(x) = m.borrow().get(&key) { return Ok(x.clone()); }
            let x = std::rc::Rc::new(cs_game::gdl_game::FactView::new(spec, vocab.clone())?);
            m.borrow_mut().insert(key, x.clone());
            Ok(x)
        })?;
        if judged { cs_decision::cards::card_of(&g, &view, &w, &s, depth, fuse, &it.3, &it.4) } else { cs_decision::cards::card_of(&g, &view, &cs_tactics::ledger::Count, &s, depth, fuse, &it.3, &it.4) }
    };
    let results: Vec<Result<Option<cs_memory::cards::Card>, String>> = py.allow_threads(|| {
        use rayon::prelude::*;
        let run = || items.par_iter().map(|it| std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| one(it))).unwrap_or_else(|_| Err("a worker panicked".into()))).collect::<Vec<_>>();
        match rayon::ThreadPoolBuilder::new().num_threads(threads).build() { Ok(pool) => pool.install(run), Err(_) => run() }
    });
    let mut cards = vec![];
    for r in results { if let Some(c) = r.map_err(pyo3::exceptions::PyValueError::new_err)? { cards.push(c); } }
    Ok(CardSet { inner: std::sync::Arc::new(cs_memory::cards::CardSet::new(cards)) })
}

/// snapshot_info(path) -> {id, game, parent, lemmas (the response rules), claims (plan 78.36: the claim rules), suggestions (the response rules marked `suggest`), teacher_used}: a snapshot read the way the engine reads it (it must hash to its name).
#[pyfunction]
fn snapshot_info(py: Python<'_>, path: &str) -> PyResult<PyObject> {
    let s = cs_memory::snapshot::Snapshot::load(std::path::Path::new(path)).map_err(pyo3::exceptions::PyValueError::new_err)?;
    let d = pyo3::types::PyDict::new(py);
    d.set_item("id", s.id)?; d.set_item("game", s.game)?; d.set_item("parent", s.parent)?;
    d.set_item("lemmas", s.lemmas.iter().map(|l| l.name.clone()).collect::<Vec<_>>())?; d.set_item("teacher_used", s.teacher_used)?;
    d.set_item("claims", s.claims.iter().map(|c| c.name.clone()).collect::<Vec<_>>())?;
    d.set_item("suggestions", s.lemmas.iter().filter(|l| l.suggest).map(|l| l.name.clone()).collect::<Vec<_>>())?;
    Ok(d.into())
}

/// claim_toward(variant, store, specs, fen=None, moves=[], seconds=0.2, max_depth=None, max_nodes=50_000_000, worth={}, ours_first=False, valuation="") -> dict: plan 78.36 THE SEARCH TOWARD
/// CLAIMS on its own (the operator "claim" of the decision, without the rest): the snapshot `store` (a file) holds the claim rules, `specs` the descriptions they are bound through. {status, move
/// (the proved win's), depth (the deepest completed iteration), exact, cut, nodes, evals, seconds, win, draw, unknown, loss (root moves by class), candidates, refuted (move names), spoke
/// [(rule, count)], claim_evaluations, claim_seconds, evidence}. A measurement and a debugging door; the wake reaches the same operator through solve_batch.
#[pyfunction]
#[pyo3(signature = (variant, store, specs, fen=None, moves=vec![], seconds=0.2, max_depth=None, max_nodes=50_000_000))]
#[allow(clippy::too_many_arguments)]
fn claim_toward(py: Python<'_>, variant: &str, store: &str, specs: HashMap<String, String>, fen: Option<&str>, moves: Vec<String>, seconds: f64, max_depth: Option<u32>, max_nodes: u64) -> PyResult<PyObject> {
    use cs_operator::Operator;
    let (g, s, keys) = record_of(variant, fen, 518, &moves)?;
    let Some(me) = g.mover(&s) else { return Err(pyo3::exceptions::PyValueError::new_err("the game is over")) };
    let st = load_store(store, specs)?;
    let view = view_for(&st, variant).ok_or_else(|| pyo3::exceptions::PyValueError::new_err(format!("no description for {variant}")))?;
    let (o, stats, legal) = (|| {
        let claims = cs_memory::claim::Claims::new(&st.snapshot, &view);
        let op = cs_decision::claim_op::ClaimOp { claims: &claims, history: keys.clone() };
        let mut b = cs_operator::Budget::new(seconds, max_nodes);
        b.max_depth = max_depth;
        let o = op.run(&g, &s, &cs_operator::Goal::Toward { role: me }, b);
        (o, claims.stats(), g.moves(&s))
    })();
    let d = pyo3::types::PyDict::new(py);
    let names = |ix: &[usize]| ix.iter().map(|i| g.move_name(&s, &legal[*i])).collect::<Vec<String>>();
    d.set_item("status", format!("{:?}", o.status))?;
    d.set_item("move", o.mv.as_ref().map(|m| g.move_name(&s, m)))?;
    d.set_item("depth", o.depth)?;
    d.set_item("nodes", o.cost.nodes)?;
    d.set_item("seconds", o.cost.seconds)?;
    d.set_item("claim_evaluations", stats.evaluations)?;
    d.set_item("claim_seconds", stats.seconds)?;
    d.set_item("candidates", names(&o.candidates))?;
    d.set_item("refuted", o.facts.iter().filter(|f| f.relation == "refuted").map(|f| f.args[1].clone()).collect::<Vec<_>>())?;
    d.set_item("spoke", o.facts.iter().filter(|f| f.relation == "spoke").map(|f| (f.args[0].clone(), f.args[1].parse::<u64>().unwrap_or(0))).collect::<Vec<_>>())?;
    d.set_item("evidence", o.evidence.clone())?;
    d.set_item("parts", o.parts.iter().map(|p| (p.name.clone(), p.cost.nodes, p.cost.seconds)).collect::<Vec<_>>())?;
    Ok(d.into())
}

/// compose(base, delta) -> the description with the delta applied (retractions and additions), as the certificate reads a variant's text.
#[pyfunction]
fn compose(base: &str, delta: &str) -> PyResult<String> { cs_gdl::compose(base, delta).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.0)) }

/// content_id(bytes) -> the first 16 hex digits of the SHA-256: a snapshot's id.
#[pyfunction]
fn content_id(data: &[u8]) -> String { cs_memory::sha::content_id(data) }

/// check_lemma(lemma_json) -> None: raises ValueError if the rule is not well formed (the engine's own reading of it). A rule carries a `response` (a lemma) or a `claim` (plan 78.36), not both.
#[pyfunction]
fn check_lemma(lemma: &str) -> PyResult<()> {
    let v: serde_json::Value = serde_json::from_str(lemma).map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("lemma: {e}")))?;
    cs_memory::claim::Rule::from_json(&v).map(|_| ()).map_err(pyo3::exceptions::PyValueError::new_err)
}

// ---------------------------------------------------------------------------------------------------------------- plan 78.21: the typed fact graph and the graph prior, in Rust

thread_local! {
    /// One graph spec per (description, graph vocabulary): loading the description (statics, fact view) and prebuilding the static graph costs far more than a position.
    static GRAPH_SPECS: std::cell::RefCell<HashMap<String, std::rc::Rc<cs_decision::graph::GraphSpec>>> = std::cell::RefCell::new(HashMap::new());
}

fn graph_spec_for(spec: &str, graph_vocabulary: &str) -> PyResult<std::rc::Rc<cs_decision::graph::GraphSpec>> {
    let key = format!("{}|{graph_vocabulary}", cs_memory::sha::sha256_hex(spec.as_bytes()));
    GRAPH_SPECS.with(|v| {
        if let Some(x) = v.borrow().get(&key) { return Ok(x.clone()); }
        let gs = std::rc::Rc::new(cs_decision::graph::GraphSpec::new(spec, graph_vocabulary).map_err(pyo3::exceptions::PyValueError::new_err)?);
        v.borrow_mut().insert(key, gs.clone());
        Ok(gs)
    })
}

/// graph_vocab(spec, graph_vocabulary) -> {"term": [token ...], "rel": [name ...], "max_parts", "max_arity"}: the CONSTANT vocabularies of the description's graph
/// (UNK = id 0 is implicit: an id is the index + 1).
#[pyfunction]
fn graph_vocab<'py>(py: Python<'py>, spec: &str, graph_vocabulary: &str) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    let gs = graph_spec_for(spec, graph_vocabulary)?;
    let d = pyo3::types::PyDict::new(py);
    d.set_item("term", gs.term_vocab().to_vec())?;
    d.set_item("rel", gs.rel_vocab().to_vec())?;
    d.set_item("max_parts", gs.max_parts())?;
    d.set_item("max_arity", cs_decision::graph::MAX_ARITY)?;
    Ok(d)
}

/// graph(variant, spec, graph_vocabulary, fen=None, moves=[]) -> dict: the typed fact graph of the position after `moves`, built in Rust: {"terms": [token], "term_ids",
/// "facts": [[relation, kind, present]], "rel_ids" (per fact), "moves": [name] (the kernel's order), "edges": {"arg0": [[src], [dst]], ... "part<i>", "removes", "adds",
/// "binds"}, "n_static_terms", "n_static_facts", "move_rel_id", "claim_rel_id"} (forward edge types; the reverses are the swapped lists).
#[pyfunction]
#[pyo3(signature = (variant, spec, graph_vocabulary, fen=None, moves=vec![]))]
fn graph<'py>(py: Python<'py>, variant: &str, spec: &str, graph_vocabulary: &str, fen: Option<&str>, moves: Vec<String>) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    let (g, s) = state_of(variant, fen, 518, &moves)?;
    let gs = graph_spec_for(spec, graph_vocabulary)?;
    let gr = gs.graph(&g, &s).map_err(pyo3::exceptions::PyValueError::new_err)?;
    let d = pyo3::types::PyDict::new(py);
    let rels = gs.rel_vocab();
    d.set_item("terms", &gr.terms)?;
    d.set_item("term_ids", &gr.term_ids)?;
    d.set_item("facts", gr.facts.iter().map(|f| (rels[f[0] as usize - 1].clone(), f[1], f[2])).collect::<Vec<_>>())?;
    d.set_item("rel_ids", gr.facts.iter().map(|f| f[0]).collect::<Vec<_>>())?;
    d.set_item("moves", &gr.moves)?;
    let e = pyo3::types::PyDict::new(py);
    for (n, a, b) in &gr.edges { e.set_item(n, vec![a.clone(), b.clone()])?; }
    d.set_item("edges", e)?;
    d.set_item("n_static_terms", gr.n_static_terms)?;
    d.set_item("n_static_facts", gr.n_static_facts)?;
    d.set_item("move_rel_id", gs.move_rel_id())?;
    d.set_item("claim_rel_id", gs.claim_rel_id())?;
    Ok(d)
}

/// GraphModel(weights_path, specs, graph_vocabulary): THE GRAPH PRIOR ENTIRELY IN RUST (plan 78.21): the exported graph net over the typed fact graph of each variant's
/// description. `rank(variant, moves, fen=None)` -> [(move, score)] best first (ties by name); `scores` -> (names in the kernel's move order, scores); `timing()` ->
/// mean milliseconds a position: facts, graph building, forward pass, total (replaying the moves included).
#[pyclass(unsendable)]
struct GraphModel {
    priors: HashMap<String, cs_decision::GraphPrior>,
    total: std::cell::Cell<f64>,
    positions: std::cell::Cell<u64>,
}

#[pymethods]
impl GraphModel {
    #[new]
    fn new(weights_path: &str, specs: HashMap<String, String>, graph_vocabulary: &str) -> PyResult<GraphModel> {
        let net = std::rc::Rc::new(cs_decision::graph_net::GraphNet::load(weights_path).map_err(pyo3::exceptions::PyValueError::new_err)?);
        let mut priors = HashMap::new();
        for (variant, text) in &specs {
            let gs = graph_spec_for(text, graph_vocabulary)?;
            let p = cs_decision::GraphPrior::new(gs, net.clone()).map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{variant}: {e}")))?;
            priors.insert(variant.clone(), p);
        }
        Ok(GraphModel { priors, total: std::cell::Cell::new(0.0), positions: std::cell::Cell::new(0) })
    }

    #[pyo3(signature = (variant, moves=vec![], fen=None))]
    fn scores(&self, variant: &str, moves: Vec<String>, fen: Option<&str>) -> PyResult<(Vec<String>, Vec<f32>)> {
        let t0 = std::time::Instant::now();
        let p = self.priors.get(variant).ok_or_else(|| pyo3::exceptions::PyValueError::new_err(format!("no description for variant {variant}")))?;
        let (g, s) = state_of(variant, fen, 518, &moves)?;
        if g.moves(&s).is_empty() { return Ok((vec![], vec![])); }
        let r = p.scores(&g, &s).map_err(pyo3::exceptions::PyValueError::new_err)?;
        self.total.set(self.total.get() + t0.elapsed().as_secs_f64());
        self.positions.set(self.positions.get() + 1);
        Ok(r)
    }

    #[pyo3(signature = (variant, moves=vec![], fen=None))]
    fn rank(&self, variant: &str, moves: Vec<String>, fen: Option<&str>) -> PyResult<Vec<(String, f32)>> {
        let (names, scores) = self.scores(variant, moves, fen)?;
        let mut out: Vec<(String, f32)> = names.into_iter().zip(scores).collect();
        out.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal).then(a.0.cmp(&b.0)));
        Ok(out)
    }

    fn timing(&self) -> HashMap<String, f64> {
        let n = self.positions.get().max(1) as f64;
        let (mut f, mut g, mut w) = (0.0, 0.0, 0.0);
        for p in self.priors.values() { let t = p.timing(); f += t.1; g += t.2; w += t.3; }
        HashMap::from([("positions".to_string(), self.positions.get() as f64), ("ms_facts".to_string(), 1000.0 * f / n), ("ms_graph".to_string(), 1000.0 * g / n),
                       ("ms_forward".to_string(), 1000.0 * w / n), ("ms_total".to_string(), 1000.0 * self.total.get() / n)])
    }
}


#[pymodule]
fn cs_engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(variants, m)?)?;
    m.add_function(wrap_pyfunction!(certify_coplay, m)?)?;
    m.add_function(wrap_pyfunction!(best_move, m)?)?;
    m.add_function(wrap_pyfunction!(playout_game, m)?)?;
    m.add_function(wrap_pyfunction!(solve_batch, m)?)?;
    m.add_function(wrap_pyfunction!(solve_equals_batch, m)?)?;
    m.add_function(wrap_pyfunction!(facts, m)?)?;
    m.add_function(wrap_pyfunction!(facts_audit, m)?)?;
    m.add_function(wrap_pyfunction!(build_cards, m)?)?;
    m.add_function(wrap_pyfunction!(estimate, m)?)?;
    m.add_function(wrap_pyfunction!(wdl, m)?)?;
    m.add_function(wrap_pyfunction!(wdl_keys, m)?)?;
    m.add_function(wrap_pyfunction!(wdl_keys_along, m)?)?;
    m.add_function(wrap_pyfunction!(snapshot_info, m)?)?;
    m.add_function(wrap_pyfunction!(claim_toward, m)?)?;
    m.add_function(wrap_pyfunction!(content_id, m)?)?;
    m.add_function(wrap_pyfunction!(compose, m)?)?;
    m.add_function(wrap_pyfunction!(check_lemma, m)?)?;
    m.add_class::<CardSet>()?;
    m.add_class::<GraphModel>()?;
    m.add_function(wrap_pyfunction!(graph, m)?)?;
    m.add_function(wrap_pyfunction!(graph_vocab, m)?)?;
    m.add_function(wrap_pyfunction!(state, m)?)?;
    m.add_function(wrap_pyfunction!(passed, m)?)?;
    m.add_function(wrap_pyfunction!(watch, m)?)?;
    m.add_function(wrap_pyfunction!(census, m)?)?;
    m.add_function(wrap_pyfunction!(reason, m)?)?;
    m.add_function(wrap_pyfunction!(scenario, m)?)?;
    m.add_function(wrap_pyfunction!(mine_valuation, m)?)?;
    m.add_class::<sheet_py::SheetModel>()?;
    Ok(())
}
