//! Python bindings (feature "python", built with maturin): `maia_chess.Maia(path).policy(fen, temp=1.0)`,
//! `.wdl(fen)`.
use pyo3::prelude::*;
use shakmaty::{fen::Fen, CastlingMode, Chess};

fn parse(fen: &str) -> PyResult<Chess> {
    let f: Fen = fen.parse().map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{e}")))?;
    f.into_position(CastlingMode::Standard).map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{e}")))
}

#[pyclass(name = "Maia")]
struct PyMaia { inner: crate::Maia }

#[pymethods]
impl PyMaia {
    #[new]
    fn new(path: String) -> PyResult<Self> {
        Ok(PyMaia { inner: crate::Maia::load(path).map_err(pyo3::exceptions::PyRuntimeError::new_err)? })
    }
    #[pyo3(signature = (fen, temp=1.0))]
    fn policy(&self, fen: &str, temp: f32) -> PyResult<Vec<(String, f32)>> {
        self.inner.policy(&parse(fen)?, temp).map_err(pyo3::exceptions::PyRuntimeError::new_err)
    }
    fn wdl(&self, fen: &str) -> PyResult<(f32, f32, f32)> {
        let w = self.inner.wdl(&parse(fen)?).map_err(pyo3::exceptions::PyRuntimeError::new_err)?;
        Ok((w.win, w.draw, w.loss))
    }
}

#[pymodule]
fn maia_chess(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyMaia>()?;
    Ok(())
}
