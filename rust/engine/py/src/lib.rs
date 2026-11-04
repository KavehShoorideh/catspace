//! plan 78: THE END-TO-END SYSTEM's Python entry -- the harness calls in once per move (or once per certification) and reads the result; no per-node crossing.
use cs_game::{certify, gdl_game::GdlGame, kernel};
use pyo3::prelude::*;

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
    Ok(())
}
