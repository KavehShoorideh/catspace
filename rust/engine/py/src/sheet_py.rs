//! PLAN 78.43 STAGE A, the Python entry of the question sheet's light reading (cs_tactics::sheet): a `SheetModel` holds what is the game's and not the position's -- the names of
//! its places and kinds, the medium of movement, the worth of each kind, the goal's kind, how many move relations the vocabulary names -- and `read(position)` returns the questions
//! of one position with their level 0 and level 1 as plain data. The Python sheet (question_sheet.py) is the oracle it is compared with. Text is made here and only here.
use cs_game::kernel::KernelGame;
use cs_game::placed::Placed;
use cs_tactics::sheet::levels::{self, Level0, Level1, Obstacle, Route, Way, Why};
use cs_tactics::sheet::medium::Medium;
use cs_tactics::sheet::reading::Reading;
use cs_tactics::sheet::{Model, Names, Params, Unit};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::IntoPyObjectExt;
use std::collections::HashMap;

type Hop = (String, String, String, String, bool);
type Over = (String, String, String, String, String);

#[pyclass(unsendable)]
pub struct SheetModel {
    variant: String,
    model: Model,
}

type Unit3 = (String, String, String);

fn unit3(n: &Names, u: &Unit) -> Unit3 { (u.side.word().to_string(), n.kind_text(u.kind).to_string(), n.place_text(u.at).to_string()) }
fn unit2(n: &Names, u: &Unit) -> (String, String) { (n.kind_text(u.kind).to_string(), n.place_text(u.at).to_string()) }
fn way_of(n: &Names, w: &Way) -> (u32, String, String, String, String) { (w.d, n.kind_text(w.unit.kind).to_string(), n.place_text(w.unit.at).to_string(), n.place_text(w.arrival).to_string(), w.what.word().to_string()) }
fn obstacle_of(n: &Names, o: &Obstacle) -> (String, Option<Unit3>, String, usize, bool) { (o.kind.to_string(), o.unit.as_ref().map(|u| unit3(n, u)), n.place_text(o.place).to_string(), o.hop, o.dead) }

pub fn why_words(n: &Names, w: &Why) -> String {
    match w {
        Why::Cheaper { u, to, by } => format!("{} would stand on {}, struck by {}, which is worth less", n.words(u), n.place_text(*to), n.words(by)),
        Why::Unguarded { u, to, by } => format!("{} would stand on {}, struck by {} and not guarded", n.words(u), n.place_text(*to), n.words(by)),
        Why::OnlyCover { u, t } => format!("{} would only cover an exit of {} and nothing strikes it yet", n.words(u), n.words(t)),
        Why::Guarded { t, guard, u } => format!("{} is guarded by {} and is worth no more than {}", n.words(t), n.words(guard), n.words(u)),
    }
}

type RouteWire = (String, String, Vec<String>, Vec<(bool, String)>, String, Vec<Vec<String>>, Vec<(String, Option<Unit3>, String, usize, bool)>, String);
fn route_of(n: &Names, r: &Route) -> RouteWire {
    (n.kind_text(r.unit.kind).to_string(), n.place_text(r.unit.at).to_string(), r.places.iter().map(|p| n.place_text(*p).to_string()).collect(),
     r.stands.iter().map(|s| match s { None => (true, String::new()), Some(w) => (false, why_words(n, w)) }).collect(), r.what.word().to_string(),
     r.over.iter().map(|o| o.iter().map(|p| n.place_text(*p).to_string()).collect()).collect(), r.obstacles.iter().map(|o| obstacle_of(n, o)).collect(), n.place_text(r.goal).to_string())
}

fn level0_dict(py: Python<'_>, n: &Names, l: &Level0) -> PyResult<PyObject> {
    let d = PyDict::new(py);
    d.set_item("status", l.status.word())?;
    d.set_item("strikers", l.strikers.iter().map(|u| unit2(n, u)).collect::<Vec<_>>())?;
    d.set_item("guards", l.guards.iter().map(|u| unit2(n, u)).collect::<Vec<_>>())?;
    d.set_item("open_exits", l.open_exits)?;
    d.set_item("exits", l.exits)?;
    if let Some((struck, cornered)) = l.goal { d.set_item("struck", struck)?; d.set_item("cornered", cornered)?; }
    d.into_py_any(py)
}

fn level1_dict(py: Python<'_>, n: &Names, l: &Level1) -> PyResult<PyObject> {
    let d = PyDict::new(py);
    d.set_item("status", l.status.word())?;
    d.set_item("ways", l.ways.iter().map(|w| way_of(n, w)).collect::<Vec<_>>())?;
    d.set_item("routes", l.routes.iter().map(|(w, rs)| (way_of(n, w), rs.iter().map(|r| route_of(n, r)).collect::<Vec<_>>())).collect::<Vec<_>>())?;
    d.set_item("failed", l.failed.iter().map(|(k, why, what)| (*k, why_words(n, why), what.word().to_string())).collect::<Vec<_>>())?;
    d.set_item("failed_at", l.failed_at)?;
    d.set_item("links", l.links.map(|x| x.to_vec()))?;
    d.set_item("reach", l.reach)?;
    d.set_item("reach_geo", l.reach_geo.iter().map(|(k, p)| (n.kind_text(*k).to_string(), n.place_text(*p).to_string())).collect::<Vec<_>>())?;
    d.set_item("far", l.far.iter().map(|(h, u)| (*h, n.kind_text(u.kind).to_string(), n.place_text(u.at).to_string())).collect::<Vec<_>>())?;
    d.set_item("too_far", l.too_far)?;
    if let Some(g) = &l.goal {
        d.set_item("open_exits", g.open_exits)?; d.set_item("near", g.near)?; d.set_item("scenarios", g.scenarios)?; d.set_item("moves", g.moves)?;
    }
    d.into_py_any(py)
}

#[pymethods]
impl SheetModel {
    /// SheetModel(variant, hops, overs, takes, worth, goal_kind, can_go): the game by its variant (its names of places, kinds and roles), the medium (hops (kind, side, from, to,
    /// take-only), places passed over (kind, side, from, to, over), the (kind, side) pairs with take-only hops), the worth of each kind by the game's word, the goal's kind or None,
    /// the number of move relations the vocabulary names.
    #[new]
    fn new(variant: &str, hops: Vec<Hop>, overs: Vec<Over>, takes: Vec<(String, String)>, worth: HashMap<String, f64>, goal_kind: Option<String>, can_go: usize) -> PyResult<SheetModel> {
        let bad = |e: String| pyo3::exceptions::PyValueError::new_err(e);
        if hops.is_empty() { return Err(bad("the game's vocabulary names no medium of movement".into())); }
        let g = KernelGame::new(variant, 518).map_err(bad)?;
        let names = Names::new(g.place_names(), g.kind_names(), cs_game::Game::roles(&g));
        let medium = Medium::new(&names, &hops, &overs, &takes).map_err(bad)?;
        let w: Vec<f64> = names.kinds.iter().map(|k| worth.get(k).copied().unwrap_or(0.0)).collect();
        let goal_kind = match goal_kind { Some(k) => Some(names.kind(&k).map_err(bad)?), None => None };
        Ok(SheetModel { variant: variant.to_string(), model: Model { names, medium, worth: w, goal_kind, can_go } })
    }

    /// read(position, reach_k=2, max_open_exits=1) -> {mover, seconds (the engine's own, before the data is handed over), units: [(side, kind, place, worth)], order: [(side, kind,
    /// place, stake, goal, rank)], level0: [dict per question in the order], level1: [dict per question]}. Raises ValueError for a position that is over.
    #[pyo3(signature = (position, reach_k=2, max_open_exits=1))]
    fn read(&self, py: Python<'_>, position: &str, reach_k: u32, max_open_exits: usize) -> PyResult<PyObject> {
        let t0 = std::time::Instant::now();
        let g = KernelGame::from_fen(&self.variant, position).map_err(pyo3::exceptions::PyValueError::new_err)?;
        let s = cs_game::Game::initial(&g);
        let r = Reading::new(&g, s, &self.model).map_err(pyo3::exceptions::PyValueError::new_err)?;
        let p = Params { reach_k, max_open_exits, ..Params::default() };
        let sheet = levels::read(&r, &p);
        let compute = t0.elapsed().as_secs_f64();
        let n = &self.model.names;
        let d = PyDict::new(py);
        d.set_item("mover", n.roles[r.mover].as_str())?;
        d.set_item("seconds", compute)?;
        d.set_item("units", r.units.iter().map(|u| (u.side.word().to_string(), n.kind_text(u.kind).to_string(), n.place_text(u.at).to_string(), r.worth_of(u.kind))).collect::<Vec<_>>())?;
        d.set_item("order", sheet.order.iter().map(|(u, stake, goal, rank)| (u.side.word().to_string(), n.kind_text(u.kind).to_string(), n.place_text(u.at).to_string(), *stake, *goal, *rank)).collect::<Vec<_>>())?;
        d.set_item("level0", sheet.level0.iter().map(|l| level0_dict(py, n, l)).collect::<PyResult<Vec<_>>>()?)?;
        d.set_item("level1", sheet.level1.iter().map(|l| level1_dict(py, n, l)).collect::<PyResult<Vec<_>>>()?)?;
        d.into_py_any(py)
    }
}
