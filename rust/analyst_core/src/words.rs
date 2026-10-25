//! PLAN 72.7 STEP 3: THE PYTHON WORDS IN RUST (docs/RUST_BOUNDARY.md section 4 step 3). Each word here is the port of a Python word of discovery/ (its docstring
//! there, with its Lean citation, is the specification and stays with the Python call-through); the Python body is kept as the TEST ORACLE in
//! tests/test_plan72_7_step3.py, compared on random positions and on the guard's. The words read the position on its handle (`Pos`, no FEN), in python-chess's
//! move order (`pos::py_order`, RUST_BOUNDARY risk 4) where a list order reaches an output, and return the structure Python formats (risk 3). Every pyo3 function
//! returns PyResult (risk 7); no rayon (risk 8); the dependency trackers are written exactly where the Python words wrote them (risk 2) -- by the vocabulary's
//! own reader (`vocab::read_word`, the word cache replaying what it recorded) and nowhere else.
use super::*;
use pyo3::exceptions::PyValueError;

fn err(s: &str) -> PyErr { PyValueError::new_err(s.to_string()) }
fn colour(white: bool) -> Color { if white { Color::White } else { Color::Black } }

/// A square named from the mover's side, as selfmine.view names it (black reads the ranks upside down).
fn view(s: Square, mover_white: bool) -> String { (if mover_white { s } else { s.flip_vertical() }).to_string() }

fn square_of(name: &str) -> PyResult<Square> { name.parse::<Square>().map_err(|_| err(&format!("no square {name}"))) }

// ------------------------------------------------------------------------------------------------------------------ 3a: the mate in two
// selfmine.bind_mate2 (plan 49; Lean `Ideas.Mined.Schemas.mate_in_two`): a move x -> y of the mover after which every reply is met by a mate, read as the PREMISE
// the binder writes for it -- the tail `(after x y (least (most (min check (eq movecount 0)))))` >= 1 and its three legality legs (selfmine.legs_of: the piece on x
// can legally go to y; a piece of ours stands on x; no pawn of ours on x about to promote, read <= 0) -- each term compiled from the s-expression Python renders
// for it (selfmine.render: squares named from the mover's side; the no-pawn leg's rank `a7` from either side) and read by `vocab::read_word` as `Pos.values`
// reads it (the word cache included). The candidates in python-chess's order, promotions and castling left out, stopping at the FIRST that binds.

/// (installation of the compositions, x, y, mover) -> the compiled ids [legal leg, ours leg, no-pawn leg, tail].
fn mate2_ids_cache() -> &'static Mutex<HashMap<(u64, u8, u8, bool), [u32; 4]>> {
    static C: OnceLock<Mutex<HashMap<(u64, u8, u8, bool), [u32; 4]>>> = OnceLock::new();
    C.get_or_init(|| Mutex::new(HashMap::new()))
}

fn mate2_ids(x: Square, y: Square, mover_white: bool) -> PyResult<[u32; 4]> {
    let key = (crate::vocab::macro_gen(), x as u8, y as u8, mover_white);
    if let Some(ids) = mate2_ids_cache().lock().unwrap().get(&key) { return Ok(*ids); }
    let (vx, vy) = (view(x, mover_white), view(y, mover_white));
    let ids = [crate::vocab::compile_term(&format!("(count (and (moves {vx}) (at {vy})))"))?,
               crate::vocab::compile_term(&format!("(count (and ours (at {vx})))"))?,
               crate::vocab::compile_term(&format!("(count (and ourpawns (and (at {vx}) (rank a7))))"))?,
               crate::vocab::compile_term(&format!("(after {vx} {vy} (least (most (min check (eq movecount 0)))))"))?];
    mate2_ids_cache().lock().unwrap().insert(key, ids);
    Ok(ids)
}

/// Does the premise of `mate_in_two` at x -> y hold on `p` as it stands (`selfmine.premise_truth` of the binder's atoms)? Every atom is read (the legs, then the
/// tail), as `Pos.values` read them in one crossing.
fn mate2_holds(p: &Chess, x: Square, y: Square, mover_white: bool) -> PyResult<bool> {
    let ids = mate2_ids(x, y, mover_white)?;
    let mut v = [0u32; 4];
    for (i, id) in ids.iter().enumerate() {
        let c = crate::vocab::compiled(*id)?;
        v[i] = crate::vocab::read_word(p, *id, &c, false);
    }
    Ok(v[0] >= 1 && v[1] >= 1 && v[2] == 0 && v[3] >= 1)
}

/// The candidates `bind_mate2` generates: every legal move but promotions and castling, python-chess's order, each (from, to) once.
fn mate2_candidates(p: &Chess) -> Vec<(Square, Square)> {
    let mut seen = std::collections::HashSet::new();
    let mut out = vec![];
    for m in crate::pos::py_order(p) {
        if m.promotion().is_some() || m.is_castle() { continue; }
        let pr = (m.from().unwrap_or(m.to()), m.to());
        if seen.insert(pr) { out.push(pr); }
    }
    out
}

/// The first candidate (its index in the list read, x, y) that binds `mate_in_two` on `p`.
fn mate2_first(p: &Chess, mover_white: bool, cands: &[(Square, Square)]) -> PyResult<Option<(usize, Square, Square)>> {
    for (i, &(x, y)) in cands.iter().enumerate() {
        if mate2_holds(p, x, y, mover_white)? { return Ok(Some((i, x, y))); }
    }
    Ok(None)
}

/// bind_mate2_pos(pos, mover_white, candidates=None) -> (index, x, y) of the FIRST candidate that binds `mate_in_two` on the handle as it stands (the squares named
/// as the binder names them, absolute), or None. `candidates`: [(x, y)] absolute names (default: the binder's own -- every legal move but promotions and castling,
/// python-chess's order). ValueError on a square name that is not one.
#[pyfunction]
#[pyo3(signature = (pos, mover_white, candidates=None))]
pub fn bind_mate2_pos(pos: &crate::pos::Pos, mover_white: bool, candidates: Option<Vec<(String, String)>>) -> PyResult<Option<(usize, String, String)>> {
    let cands = match candidates {
        Some(v) => v.iter().map(|(x, y)| Ok((square_of(x)?, square_of(y)?))).collect::<PyResult<Vec<_>>>()?,
        None => mate2_candidates(&pos.pos),
    };
    Ok(mate2_first(&pos.pos, mover_white, &cands)?.map(|(i, x, y)| (i, x.to_string(), y.to_string())))
}

/// regress.mate2_threat after our move: `c` = the board after it (their turn). False when it checks (no pass is legal), when we would have no legal move after a
/// pass, or when the board after the pass is no position shakmaty holds (the FEN reader read nothing there); else whether `mate_in_two` binds for us after the pass.
fn mate2_after(c: &Chess, me: bool) -> PyResult<bool> {
    if c.is_check() { return Ok(false); }
    let Some(w) = crate::pos::with_turn_strict(c, colour(me)) else { return Ok(false) };
    if w.legal_moves().is_empty() { return Ok(false); }
    let cands = mate2_candidates(&w);
    Ok(mate2_first(&*w, me, &cands)?.is_some())
}

/// mate2_threat_pos(pos, uci, me_white) -> regress.mate2_threat: after our legal move `uci` and a pass, a forced mate in two for us (`mate_in_two` binds).
/// ValueError on an illegal move.
#[pyfunction]
pub fn mate2_threat_pos(pos: &crate::pos::Pos, uci: &str, me_white: bool) -> PyResult<bool> {
    let c = pos.push_move(uci)?;
    mate2_after(&c.pos, me_white)
}

/// mate2_after_pos(pos, me_white) -> `mate2_threat` read on the board AFTER our move (the handle `pos`, their turn): for a board whose own handle shakmaty does
/// not hold before the move.
#[pyfunction]
pub fn mate2_after_pos(pos: &crate::pos::Pos, me_white: bool) -> PyResult<bool> { mate2_after(&pos.pos, me_white) }

