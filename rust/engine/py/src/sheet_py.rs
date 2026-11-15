//! PLAN 78.43 STAGE A, the Python entry of the question sheet's light reading (cs_tactics::sheet): a `SheetModel` holds what is the game's and not the position's -- the names of
//! its places and kinds, the medium of movement, the worth of each kind, the goal's kind, how many move relations the vocabulary names -- and `read(position)` returns the questions
//! of one position with their level 0 and level 1 as plain data. The Python sheet (question_sheet.py) is the oracle it is compared with. Text is made here and only here.
use cs_game::kernel::KernelGame;
use cs_game::placed::Placed;
use cs_tactics::sheet::cf::CfObstacle;
use cs_tactics::sheet::levels::{self, Level0, Level1, Obstacle, Route, Way, Why};
use cs_tactics::sheet::medium::Medium;
use cs_tactics::sheet::mv::{self, Event, Loss, MoveAnswer};
use cs_tactics::sheet::reading::Reading;
use cs_tactics::sheet::settle::{Ask, Cornering, Detail, Duty, Entry, Finding, Q, Settler};
use cs_tactics::sheet::walk::Stop;
use cs_tactics::sheet::{Model, Names, Params, Side, Unit};
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
    #[pyo3(signature = (variant, hops, overs, takes, worth, goal_kind, can_go, ledger_worth=HashMap::new(), letters=HashMap::new()))]
    #[allow(clippy::too_many_arguments)]
    fn new(variant: &str, hops: Vec<Hop>, overs: Vec<Over>, takes: Vec<(String, String)>, worth: HashMap<String, f64>, goal_kind: Option<String>, can_go: usize,
           ledger_worth: HashMap<String, i64>, letters: HashMap<String, String>) -> PyResult<SheetModel> {
        let bad = |e: String| pyo3::exceptions::PyValueError::new_err(e);
        if hops.is_empty() { return Err(bad("the game's vocabulary names no medium of movement".into())); }
        let g = KernelGame::new(variant, 518).map_err(bad)?;
        let names = Names::new(g.place_names(), g.kind_names(), cs_game::Game::roles(&g));
        let medium = Medium::new(&names, &hops, &overs, &takes).map_err(bad)?;
        let w: Vec<f64> = names.kinds.iter().map(|k| worth.get(k).copied().unwrap_or(0.0)).collect();
        let goal_kind = match goal_kind { Some(k) => Some(names.kind(&k).map_err(bad)?), None => None };
        let letters: Vec<String> = names.kinds.iter().map(|k| letters.get(k).cloned().unwrap_or_else(|| k.clone())).collect();
        Ok(SheetModel { variant: variant.to_string(), model: Model { names, medium, worth: w, goal_kind, can_go, ledger_worth, letters } })
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

    /// settle(position, steps, reach_k=2, max_open_exits=1) -> {seconds, steps: [dict per step]}: the deeper levels of the light sheet (stage B), driven by `steps`, a list of
    /// (side, kind, place, level) in the order the Python sheet took them (level 2 and on). A question keeps what the deeper levels keep of it across its steps; the answers the
    /// sheet keeps on its reading are kept across questions, so the order matters. Each step's dict: {status, mode, ...} (see `finding_dict`).
    #[pyo3(signature = (position, steps, reach_k=2, max_open_exits=1))]
    fn settle(&self, py: Python<'_>, position: &str, steps: Vec<(String, String, String, usize)>, reach_k: u32, max_open_exits: usize) -> PyResult<PyObject> {
        let bad = |e: String| pyo3::exceptions::PyValueError::new_err(e);
        let t0 = std::time::Instant::now();
        let g = KernelGame::from_fen(&self.variant, position).map_err(bad)?;
        let s = cs_game::Game::initial(&g);
        let r = Reading::new(&g, s, &self.model).map_err(bad)?;
        let p = Params { reach_k, max_open_exits, ..Params::default() };
        let settler = Settler::new(&r, &p);
        let order = r.first_order(&p);
        let n = &self.model.names;
        let mut qs: HashMap<Unit, Q> = HashMap::new();
        let mut done: Vec<Finding> = vec![];
        for (side, kind, place, level) in &steps {
            let side = if side == "us" { Side::Us } else { Side::Them };
            let u = Unit { side, kind: n.kind(kind).map_err(bad)?, at: n.place(place).map_err(bad)? };
            if !qs.contains_key(&u) {
                let (_, stake, goal, _) = order.iter().find(|o| o.0 == u).ok_or_else(|| bad(format!("no such unit on the board: {kind} on {place}")))?;
                let l1 = levels::level1(&r, &u, side.other(), *goal, &p);
                qs.insert(u, Q { t: u, asker: side.other(), stake: *stake, goal: *goal, blocked: 0, l1: std::rc::Rc::new(l1) });
            }
            let q = qs.get_mut(&u).unwrap();
            done.push(settler.deep(q, *level));
        }
        let compute = t0.elapsed().as_secs_f64();
        let d = PyDict::new(py);
        d.set_item("seconds", compute)?;
        d.set_item("steps", done.iter().map(|f| finding_dict(py, n, f)).collect::<PyResult<Vec<_>>>()?)?;
        d.into_py_any(py)
    }

    /// move_question(position, mv, reach_k=2, max_open_exits=1) -> the question of one move of the side to move as plain data (see `answer_dict`); ValueError when the move is not
    /// a legal move of a unit of the side to move.
    #[pyo3(signature = (position, mv, reach_k=2, max_open_exits=1))]
    fn move_question(&self, py: Python<'_>, position: &str, mv: &str, reach_k: u32, max_open_exits: usize) -> PyResult<PyObject> {
        let bad = |e: String| pyo3::exceptions::PyValueError::new_err(e);
        let t0 = std::time::Instant::now();
        let g = KernelGame::from_fen(&self.variant, position).map_err(bad)?;
        let s = cs_game::Game::initial(&g);
        let r = Reading::new(&g, s, &self.model).map_err(bad)?;
        let p = Params { reach_k, max_open_exits, ..Params::default() };
        let settler = Settler::new(&r, &p);
        let m = cs_game::Game::moves(&g, &r.s).into_iter().find(|m| cs_game::Game::move_name(&g, &r.s, m) == mv).ok_or_else(|| bad(format!("{mv} is not a legal move")))?;
        let a = settler.move_question(&m).map_err(bad)?;
        let d = answer_dict(py, &self.model.names, &a)?;
        d.set_item("seconds", t0.elapsed().as_secs_f64())?;
        d.into_py_any(py)
    }

    /// move_sheet(position, reach_k=2, max_open_exits=1, max_moves=None) -> {order, unread, results: [answer dict per move asked], seconds}: the question of the move for each legal move.
    #[pyo3(signature = (position, reach_k=2, max_open_exits=1, max_moves=None))]
    fn move_sheet(&self, py: Python<'_>, position: &str, reach_k: u32, max_open_exits: usize, max_moves: Option<usize>) -> PyResult<PyObject> {
        let bad = |e: String| pyo3::exceptions::PyValueError::new_err(e);
        let t0 = std::time::Instant::now();
        let g = KernelGame::from_fen(&self.variant, position).map_err(bad)?;
        let s = cs_game::Game::initial(&g);
        let r = Reading::new(&g, s, &self.model).map_err(bad)?;
        let p = Params { reach_k, max_open_exits, ..Params::default() };
        let settler = Settler::new(&r, &p);
        let sheet = settler.move_sheet(max_moves);
        let compute = t0.elapsed().as_secs_f64();
        let d = PyDict::new(py);
        d.set_item("seconds", compute)?;
        d.set_item("order", sheet.order.clone())?;
        d.set_item("unread", sheet.unread.clone())?;
        d.set_item("results", sheet.results.iter().map(|a| answer_dict(py, &self.model.names, a)).collect::<PyResult<Vec<_>>>()?)?;
        d.into_py_any(py)
    }

    /// unit_loss(events, unit_id, start, horizon) -> None | {status, balance, place, ply, run, arrival, charged}: the loss of a unit followed through a game's real moves (`events`: one
    /// (mover id, mover side, mover kind, destination, taken (id, side, kind) | None, taken_at | None, arrives | None) per move). `charged` is the move a lost unit is charged to.
    fn unit_loss(&self, py: Python<'_>, events: Vec<EventWire>, unit_id: i64, start: usize, horizon: usize) -> PyResult<PyObject> {
        let evs: Vec<Event> = events.into_iter().map(event_of).collect();
        let m = &self.model;
        let worth = |k: &str| mv::worth_by_word(m, k);
        match mv::unit_loss(&evs, unit_id, start, horizon, &worth) {
            None => Ok(py.None()),
            Some(l) => loss_dict(py, &evs, &l),
        }
    }
}

type EventWire = (i64, String, String, String, Option<(i64, String, String)>, Option<String>, Option<String>);

fn event_of(e: EventWire) -> Event { Event { mover: (e.0, e.1, e.2), dest: e.3, taken: e.4, taken_at: e.5, arrives: e.6 } }

fn loss_dict(py: Python<'_>, evs: &[Event], l: &Loss) -> PyResult<PyObject> {
    let d = PyDict::new(py);
    d.set_item("status", l.status.word())?;
    d.set_item("balance", l.balance)?;
    d.set_item("place", l.place.clone())?;
    d.set_item("ply", l.ply)?;
    d.set_item("run", l.run.clone())?;
    d.set_item("arrival", l.arrival)?;
    d.set_item("charged", if l.status == mv::LossStatus::Lost { mv::charged_ply(evs, l) } else { None })?;
    d.into_py_any(py)
}

type ThreatWire = (Unit3, String, String, String, f64, Vec<String>, bool, String);

fn threat_of(n: &Names, t: &mv::Threat) -> ThreatWire {
    (unit3(n, &t.unit), n.place_text(t.place).to_string(), t.part.to_string(), t.kind.to_string(), t.value, t.line.clone(), t.open, t.words.clone())
}

fn answer_dict<'py>(py: Python<'py>, n: &Names, a: &MoveAnswer) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("move", a.mv.clone())?;
    d.set_item("unit", unit3(n, &a.unit))?;
    d.set_item("dest", n.place_text(a.dest))?;
    d.set_item("took", a.took)?;
    d.set_item("credit", a.credit)?;
    d.set_item("took_units", a.took_units.iter().map(|u| unit3(n, u)).collect::<Vec<_>>())?;
    d.set_item("answers", a.answers.to_vec())?;
    d.set_item("threats", a.threats.iter().map(|t| threat_of(n, t)).collect::<Vec<_>>())?;
    d.set_item("unsettled", a.unsettled.iter().map(|x| (unit3(n, &x.unit), x.part.to_string(), x.words.clone(), x.stake)).collect::<Vec<_>>())?;
    d.set_item("verdict", a.verdict)?;
    d.set_item("named", a.named.as_ref().map(|x| (unit3(n, &x.unit), n.place_text(x.place).to_string(), x.part.to_string(), x.words.clone(), x.kind.map(|k| k.to_string()), x.value)))?;
    d.set_item("low", a.low)?;
    d.set_item("high", a.high)?;
    d.set_item("line", a.line.clone())?;
    d.set_item("because", a.because.clone())?;
    Ok(d)
}

type Wire3 = (String, String, String);
type DutyWire = (Wire3, f64, String, f64);
type StopWire = (bool, Option<String>, usize, Option<usize>, String, Vec<String>, Option<(String, String)>);

fn side_text(s: Side) -> String { s.word().to_string() }

fn duty_of(n: &Names, d: &Duty) -> DutyWire { (unit3(n, &d.unit), d.stake, d.kind.to_string(), d.value) }

fn stop_of(n: &Names, s: &Stop) -> StopWire {
    (s.hard, s.answer.clone(), s.after, s.step, s.reason.to_string(), s.places.iter().map(|p| n.place_text(*p).to_string()).collect(),
     s.mover.map(|(a, b)| (n.place_text(a).to_string(), n.place_text(b).to_string())))
}

fn cf_of(n: &Names, o: &CfObstacle) -> (String, Option<Wire3>, String, usize, bool, Vec<String>, Option<String>) {
    (o.kind.to_string(), o.unit.as_ref().map(|u| unit3(n, u)), n.place_text(o.place).to_string(), o.hop, o.dead, o.joint.iter().map(|p| n.place_text(*p).to_string()).collect(), o.aside.map(|p| n.place_text(p).to_string()))
}

fn ask_of(n: &Names, a: &Ask) -> (String, f64, u32, Vec<DutyWire>, Option<String>) {
    (a.verdict.to_string(), a.cost, a.more, a.duties.iter().map(|d| duty_of(n, d)).collect(), a.mv.clone())
}

fn entry_dict(py: Python<'_>, n: &Names, e: &Entry) -> PyResult<PyObject> {
    let d = PyDict::new(py);
    d.set_item("way", e.way.as_ref().map(|w| way_of(n, w)))?;
    d.set_item("status", e.status.word())?;
    d.set_item("value", e.value)?;
    d.set_item("line", e.line.clone())?;
    d.set_item("source", e.source)?;
    d.set_item("open", e.open)?;
    d.set_item("nodes", e.nodes)?;
    d.set_item("move", e.mv.clone())?;
    d.set_item("route", e.route.as_ref().map(|r| r.places.iter().map(|p| n.place_text(*p).to_string()).collect::<Vec<_>>()))?;
    d.set_item("obstacles", e.obstacles.iter().map(|(o, a)| (cf_of(n, o), ask_of(n, a))).collect::<Vec<_>>())?;
    d.set_item("first_obstacles", e.first_obstacles.as_ref().map(|v| v.iter().map(|(s, k, dd)| (s.map(side_text), k.to_string(), *dd)).collect::<Vec<_>>()))?;
    d.set_item("stops", e.stops.iter().map(|s| stop_of(n, s)).collect::<Vec<_>>())?;
    d.set_item("alts", e.alts.iter().map(|(k, ps)| (k.to_string(), ps.iter().map(|p| n.place_text(*p).to_string()).collect::<Vec<_>>())).collect::<Vec<_>>())?;
    d.set_item("breaks", e.breaks.iter().map(|s| stop_of(n, s)).collect::<Vec<_>>())?;
    d.set_item("duties", e.duties.as_ref().map(|v| v.iter().map(|x| duty_of(n, x)).collect::<Vec<_>>()))?;
    d.set_item("held_back", e.held_back)?;
    d.set_item("skipped", e.skipped)?;
    d.set_item("dead_routes", e.dead_routes)?;
    d.set_item("unread_routes", e.unread_routes)?;
    d.set_item("without_stopper", e.without_stopper.as_ref().map(|(u, v, val)| (unit3(n, u), v.word().to_string(), *val)))?;
    d.set_item("tried", e.tried)?;
    d.set_item("cut", e.cut)?;
    d.set_item("routes_tried", e.routes_tried)?;
    d.set_item("answer", e.answer.clone())?;
    d.into_py_any(py)
}

fn cornering_dict(py: Python<'_>, n: &Names, c: &Cornering) -> PyResult<PyObject> {
    let d = PyDict::new(py);
    d.set_item("cornered", c.cornered)?;
    d.set_item("lost_now", c.lost_now)?;
    d.set_item("places", c.places.iter().map(|(p, lost)| (n.place_text(*p).to_string(), *lost)).collect::<Vec<_>>())?;
    d.set_item("open_exits", c.open_exits)?;
    d.into_py_any(py)
}

fn finding_dict(py: Python<'_>, n: &Names, f: &Finding) -> PyResult<PyObject> {
    let d = PyDict::new(py);
    d.set_item("status", f.status.word())?;
    match &f.detail {
        Detail::Frame(x) => {
            d.set_item("mode", "frame")?;
            d.set_item("results", x.results.iter().map(|e| entry_dict(py, n, e)).collect::<PyResult<Vec<_>>>()?)?;
            d.set_item("confirmed", x.confirmed)?; d.set_item("refuted", x.refuted)?; d.set_item("waiting", x.waiting)?; d.set_item("open", x.open)?;
            d.set_item("lost_where_it_stands", x.lost_where_it_stands)?;
            d.set_item("value", x.value)?;
            d.set_item("line", x.line.clone())?;
            d.set_item("sources", x.sources.clone())?;
            if let Some(c) = &x.cornering { d.set_item("cornering", cornering_dict(py, n, c)?)?; }
        }
        Detail::Search(x) => {
            d.set_item("mode", "search")?;
            d.set_item("depth", x.depth)?;
            d.set_item("ways", x.ways.iter().map(|w| way_of(n, w)).collect::<Vec<_>>())?;
            d.set_item("results", x.results.iter().map(|s| (s.mv.clone(), s.status.to_string(), s.value, s.reply.clone())).collect::<Vec<_>>())?;
            d.set_item("confirmed", x.confirmed)?; d.set_item("refuted", x.refuted)?; d.set_item("open", x.open)?;
            d.set_item("value", x.value)?; d.set_item("line", x.line.clone())?;
        }
        Detail::Census(x) => {
            d.set_item("mode", "census")?;
            d.set_item("depth", x.depth)?; d.set_item("tactics", x.tactics)?; d.set_item("value", x.value)?; d.set_item("line", x.line.clone())?;
        }
    }
    d.into_py_any(py)
}
