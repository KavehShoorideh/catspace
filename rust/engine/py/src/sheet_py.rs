//! PLAN 78.43 STAGE A, the Python entry of the question sheet's light reading (cs_tactics::sheet): a `SheetModel` holds what is the game's and not the position's -- the medium of
//! movement, the worth of each kind, the goal's kind, how many move relations the vocabulary names -- and `read(position)` returns the questions of one position with their level 0
//! and level 1 as plain data. The Python sheet (question_sheet.py) is the oracle it is compared with.
use cs_game::kernel::KernelGame;
use cs_tactics::sheet::levels::{self, Level0, Level1, Obstacle, Route, Way};
use cs_tactics::sheet::medium::Medium;
use cs_tactics::sheet::reading::Reading;
use cs_tactics::sheet::{Params, Unit};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::IntoPyObjectExt;
use std::collections::HashMap;

type Hop = (String, String, String, String, bool);
type Over = (String, String, String, String, String);

#[pyclass(unsendable)]
pub struct SheetModel {
    medium: Option<Medium>,
    worth: HashMap<String, f64>,
    goal_kind: Option<String>,
    can_go: usize,
}

fn unit3(u: &Unit) -> (String, String, String) { (u.side.word().to_string(), u.kind.clone(), u.at.clone()) }
fn unit2(u: &Unit) -> (String, String) { (u.kind.clone(), u.at.clone()) }
fn way_of(w: &Way) -> (u32, String, String, String, String) { (w.d, w.unit.kind.clone(), w.unit.at.clone(), w.arrival.clone(), w.what.word().to_string()) }
fn obstacle_of(o: &Obstacle) -> (String, Option<(String, String, String)>, String, usize, bool) { (o.kind.to_string(), o.unit.as_ref().map(unit3), o.place.clone(), o.hop, o.dead) }

type RouteWire = (String, String, Vec<String>, Vec<(bool, String)>, String, Vec<Vec<String>>, Vec<(String, Option<(String, String, String)>, String, usize, bool)>, String);
fn route_of(r: &Route) -> RouteWire {
    (r.unit.kind.clone(), r.unit.at.clone(), r.places.clone(), r.stands.clone(), r.what.word().to_string(), r.over.clone(), r.obstacles.iter().map(obstacle_of).collect(), r.goal.clone())
}

fn level0_dict(py: Python<'_>, l: &Level0) -> PyResult<PyObject> {
    let d = PyDict::new(py);
    d.set_item("status", l.status.word())?;
    d.set_item("strikers", l.strikers.iter().map(unit2).collect::<Vec<_>>())?;
    d.set_item("guards", l.guards.iter().map(unit2).collect::<Vec<_>>())?;
    d.set_item("open_exits", l.open_exits)?;
    d.set_item("exits", l.exits)?;
    match l.goal { Some((struck, cornered)) => { d.set_item("struck", struck)?; d.set_item("cornered", cornered)?; } None => {} }
    d.into_py_any(py)
}

fn level1_dict(py: Python<'_>, l: &Level1) -> PyResult<PyObject> {
    let d = PyDict::new(py);
    d.set_item("status", l.status.word())?;
    d.set_item("ways", l.ways.iter().map(way_of).collect::<Vec<_>>())?;
    d.set_item("routes", l.routes.iter().map(|(w, rs)| (way_of(w), rs.iter().map(route_of).collect::<Vec<_>>())).collect::<Vec<_>>())?;
    d.set_item("failed", l.failed.iter().map(|(k, why, what)| (*k, why.clone(), what.word().to_string())).collect::<Vec<_>>())?;
    d.set_item("failed_at", l.failed_at)?;
    d.set_item("links", l.links.map(|x| x.to_vec()))?;
    d.set_item("reach", l.reach)?;
    d.set_item("reach_geo", l.reach_geo.iter().cloned().collect::<Vec<_>>())?;
    d.set_item("far", l.far.iter().map(|(h, u)| (*h, u.kind.clone(), u.at.clone())).collect::<Vec<_>>())?;
    d.set_item("too_far", l.too_far)?;
    if let Some(g) = &l.goal {
        d.set_item("open_exits", g.open_exits)?; d.set_item("near", g.near)?; d.set_item("scenarios", g.scenarios)?; d.set_item("moves", g.moves)?;
    }
    d.into_py_any(py)
}

#[pymethods]
impl SheetModel {
    /// SheetModel(hops, overs, takes, worth, goal_kind, can_go): the medium (hops (kind, side, from, to, take-only), places passed over (kind, side, from, to, over), the (kind, side)
    /// pairs with take-only hops; empty hops: no medium), the worth of each kind by the game's word, the goal's kind or None, the number of move relations the vocabulary names.
    #[new]
    fn new(hops: Vec<Hop>, overs: Vec<Over>, takes: Vec<(String, String)>, worth: HashMap<String, f64>, goal_kind: Option<String>, can_go: usize) -> SheetModel {
        let medium = if hops.is_empty() { None } else { Some(Medium::new(hops, overs, takes)) };
        SheetModel { medium, worth, goal_kind, can_go }
    }

    /// read(variant, position, reach_k=2, max_open_exits=1) -> {mover, seconds (the engine's own, before the data is handed over), units: [(side, kind, place, worth)], order: [(side, kind, place, stake, goal, rank)], level0: [dict per question
    /// in the order], level1: [dict per question]}. Raises ValueError for a position that is over.
    #[pyo3(signature = (variant, position, reach_k=2, max_open_exits=1))]
    fn read(&self, py: Python<'_>, variant: &str, position: &str, reach_k: u32, max_open_exits: usize) -> PyResult<PyObject> {
        let t0 = std::time::Instant::now();
        let g = KernelGame::from_fen(variant, position).map_err(pyo3::exceptions::PyValueError::new_err)?;
        let s = cs_game::Game::initial(&g);
        let r = Reading::new(&g, s, self.medium.as_ref(), &self.worth, self.goal_kind.clone(), self.can_go).map_err(pyo3::exceptions::PyValueError::new_err)?;
        let p = Params { reach_k, max_open_exits, ..Params::default() };
        let sheet = levels::read(&r, &p);
        let compute = t0.elapsed().as_secs_f64();
        let d = PyDict::new(py);
        d.set_item("mover", r.name(cs_tactics::sheet::Side::Us))?;
        d.set_item("seconds", compute)?;
        d.set_item("units", r.units.iter().map(|u| (u.side.word().to_string(), u.kind.clone(), u.at.clone(), r.worth_of(&u.kind))).collect::<Vec<_>>())?;
        d.set_item("order", sheet.order.iter().map(|(u, stake, goal, rank)| (u.side.word().to_string(), u.kind.clone(), u.at.clone(), *stake, *goal, *rank)).collect::<Vec<_>>())?;
        d.set_item("level0", sheet.level0.iter().map(|l| level0_dict(py, l)).collect::<PyResult<Vec<_>>>()?)?;
        d.set_item("level1", sheet.level1.iter().map(|l| level1_dict(py, l)).collect::<PyResult<Vec<_>>>()?)?;
        d.into_py_any(py)
    }
}
