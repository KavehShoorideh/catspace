//! plan 78: THE END-TO-END SYSTEM's Python entry -- the harness calls in once per move (or once per certification) and reads the result; no per-node crossing.
use cs_game::{certify, gdl_game::GdlGame, kernel, Game};
use pyo3::prelude::*;
use std::collections::HashMap;

fn state_of(variant: &str, fen: Option<&str>, start960: u32, moves: &[String]) -> PyResult<(kernel::KernelGame, shakmaty_state::S)> {
    let g = match fen { Some(f) => kernel::KernelGame::from_fen(variant, f), None => kernel::KernelGame::new(variant, start960) }
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    let mut s = g.initial();
    for u in moves {
        let m = g.parse(&s, u).ok_or_else(|| pyo3::exceptions::PyValueError::new_err(format!("illegal move {u}")))?;
        s = g.apply(&s, &m);
    }
    Ok((g, s))
}

mod shakmaty_state { pub type S = <cs_game::kernel::KernelGame as cs_game::Game>::State; }

/// best_move(variant, fen=None, moves=[], seconds=1.0, worth={}, start960=518) -> (move or None, proved, [trace], nodes, fen after the moves): THE END-TO-END
/// DECISION (plan 78.2: tactics `forcing`, don't lose, planning `keep`) for the side to move after `moves` from the start (or from `fen`). `worth`: the store's
/// valuation (kind name -> worth) the planner's measure counts.
#[pyfunction]
#[pyo3(signature = (variant, fen=None, moves=vec![], seconds=1.0, worth=HashMap::new(), start960=518))]
fn best_move(variant: &str, fen: Option<&str>, moves: Vec<String>, seconds: f64, worth: HashMap<String, i64>, start960: u32)
             -> PyResult<(Option<String>, bool, Vec<String>, u64, String)> {
    let (g, s) = state_of(variant, fen, start960, &moves)?;
    let f = g.fen(&s);
    Ok(match cs_decision::decide(&g, &s, seconds, &worth) {
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

#[pymodule]
fn cs_engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(variants, m)?)?;
    m.add_function(wrap_pyfunction!(certify_coplay, m)?)?;
    m.add_function(wrap_pyfunction!(best_move, m)?)?;
    m.add_function(wrap_pyfunction!(state, m)?)?;
    Ok(())
}
