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

mod shakmaty_state { pub type S = <cs_game::kernel::KernelGame as cs_game::Game>::State; }

/// best_move(variant, fen=None, moves=[], seconds=1.0, worth={}, start960=518) -> (move or None, proved, [trace], nodes, fen after the moves): THE END-TO-END
/// DECISION (plan 78.5: tactics `forcing`, then the ledger search) for the side to move after `moves` from the start (or from `fen`). The ledger's JUDGE is the
/// store's valuation record in use, passed in: `worth` (kind name -> worth), `ours_first`, `valuation` (its name, for the trace); an empty table: the built-in Count.
#[pyfunction]
#[pyo3(signature = (variant, fen=None, moves=vec![], seconds=1.0, worth=HashMap::new(), start960=518, ours_first=false, valuation=String::new(), prior=false, record=true, prior_depth=2, ledger_share=0.6))]
#[allow(clippy::too_many_arguments)]
fn best_move(variant: &str, fen: Option<&str>, moves: Vec<String>, seconds: f64, worth: HashMap<String, i64>, start960: u32, ours_first: bool, valuation: String,
             prior: bool, record: bool, prior_depth: i32, ledger_share: f64)
             -> PyResult<(Option<String>, bool, Vec<String>, u64, String)> {
    let (g, s, keys) = record_of(variant, fen, start960, &moves)?;
    let hist: Vec<u64> = if record { keys } else { vec![] };          // (record=False: the game's record not read -- the comparison)
    let f = g.fen(&s);
    let choice = if worth.is_empty() { cs_decision::decide_with(&g, &s, seconds, &cs_tactics::ledger::Count, prior, &hist, prior_depth, ledger_share) }
                 else { cs_decision::decide_with(&g, &s, seconds, &cs_tactics::ledger::Worth { name: valuation, table: worth.clone(), ours_first }, prior, &hist, prior_depth, ledger_share) };
    Ok(match choice {
        Some(c) => (Some(g.move_name(&s, &c.mv)), c.proved, c.trace, c.nodes, f),
        None => (None, false, vec!["the game is over".into()], 0, f),
    })
}

/// state(variant, fen=None, moves=[], start960=518) -> (fen, the mover's name or None, [legal moves], the outcome per role or None): the harness's view of a state.
#[pyfunction]
#[pyo3(signature = (variant, fen=None, moves=vec![], start960=518))]
fn state(variant: &str, fen: Option<&str>, moves: Vec<String>, start960: u32) -> PyResult<(String, Option<String>, Vec<String>, Option<Vec<i64>>, u64)> {
    let (g, s) = state_of(variant, fen, start960, &moves)?;
    let ms = g.moves(&s).iter().map(|m| g.move_name(&s, m)).collect();
    Ok((g.fen(&s), g.mover(&s).map(|r| g.roles()[r].clone()), ms, g.outcome(&s), g.key(&s)))
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

#[pymodule]
fn cs_engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(variants, m)?)?;
    m.add_function(wrap_pyfunction!(certify_coplay, m)?)?;
    m.add_function(wrap_pyfunction!(best_move, m)?)?;
    m.add_function(wrap_pyfunction!(state, m)?)?;
    m.add_function(wrap_pyfunction!(watch, m)?)?;
    m.add_function(wrap_pyfunction!(census, m)?)?;
    m.add_function(wrap_pyfunction!(mine_valuation, m)?)?;
    Ok(())
}
