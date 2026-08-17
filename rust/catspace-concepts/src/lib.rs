//! catspace-concepts -- named board probes for concept-program distillation.
//!
//! Concepts today are directions in a learned 192-d trunk that costs 2.347 ms/position on one CPU
//! thread -- about 2.6 s/move at 800 nodes, which is why concepts have never been affordable
//! inside search. This crate computes a basis of ~1,892 NAMED primitives directly from the corpus
//! arrays, so a concept can be distilled into a program over things a chess player has words for,
//! and evaluated in microseconds.
//!
//! The Python module `concepts_reference.py` is the specification, not a fallback: every number
//! here is held to elementwise equality with it by `test_probe_features.py`.

use numpy::ndarray::{Array2, Array3};
use numpy::{IntoPyArray, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;

pub mod board;
pub mod concepts;
pub mod names;
pub mod probes;

use names::{N_GLOB, N_SQ};

/// Extract the basis for a batch of positions.
///
/// `tok` is (N, 64) uint8, `glob` is (N, 6) uint8 -- the arrays the corpus already holds.
/// `repetition` and `halfmove` are (N,) float32 and come from the GAME, not the position; they
/// cannot be recovered from a single board and are passed through unchanged.
///
/// Returns `(sq_feat (N, 64, 28), glob_feat (N, 100))` float32, plus the number of rows whose
/// position could not be constructed (which should be zero on a real corpus, and is returned
/// rather than logged so a caller cannot ignore it).
#[pyfunction]
#[pyo3(signature = (tok, glob, repetition=None, halfmove=None))]
fn extract<'py>(
    py: Python<'py>,
    tok: PyReadonlyArray2<'py, u8>,
    glob: PyReadonlyArray2<'py, u8>,
    repetition: Option<PyReadonlyArray1<'py, f32>>,
    halfmove: Option<PyReadonlyArray1<'py, f32>>,
) -> PyResult<(Bound<'py, PyArray3<f32>>, Bound<'py, PyArray2<f32>>, usize)> {
    let tok = tok.as_array();
    let glob = glob.as_array();
    let n = tok.shape()[0];
    if glob.shape()[0] != n {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "tok and glob must have the same number of rows",
        ));
    }
    if tok.shape()[1] != 64 || glob.shape()[1] != 6 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "expected tok (N,64) and glob (N,6)",
        ));
    }
    let rep: Vec<f32> = repetition
        .map(|a| a.as_array().to_vec())
        .unwrap_or_else(|| vec![0.0; n]);
    let hm: Vec<f32> = halfmove
        .map(|a| a.as_array().to_vec())
        .unwrap_or_else(|| vec![0.0; n]);

    let toks: Vec<Vec<u8>> = (0..n).map(|r| tok.row(r).to_vec()).collect();
    let globs: Vec<Vec<u8>> = (0..n).map(|r| glob.row(r).to_vec()).collect();

    let mut sq_out = vec![0f32; n * 64 * N_SQ];
    let mut gl_out = vec![0f32; n * N_GLOB];
    let failed: usize = py.detach(|| {
        sq_out
            .par_chunks_mut(64 * N_SQ)
            .zip(gl_out.par_chunks_mut(N_GLOB))
            .enumerate()
            .map(|(r, (sq_row, gl_row))| match board::position_from_row(&toks[r], &globs[r]) {
                Some(pos) => {
                    probes::sq_features(&pos, sq_row);
                    concepts::glob_features(&pos, rep[r], hm[r], gl_row);
                    0
                }
                None => 1,
            })
            .sum()
    });

    let sq = Array3::from_shape_vec((n, 64, N_SQ), sq_out)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    let gl = Array2::from_shape_vec((n, N_GLOB), gl_out)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok((sq.into_pyarray(py), gl.into_pyarray(py), failed))
}

/// Column names, in order. Asserted equal to the Python oracle's lists before any number is
/// compared -- a name that drifts from its column turns every printed program into a quiet lie.
#[pyfunction]
fn feature_names() -> (Vec<String>, Vec<String>) {
    (names::sq_names(), names::glob_names())
}

#[pymodule]
fn catspace_concepts(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(extract, m)?)?;
    m.add_function(wrap_pyfunction!(feature_names, m)?)?;
    m.add("N_SQ", N_SQ)?;
    m.add("N_GLOB", N_GLOB)?;
    Ok(())
}
