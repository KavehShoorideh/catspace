//! PLAN 78.43 STAGE A, the Python entry of the question sheet's light reading (cs_tactics::sheet): a `SheetModel` holds what is the game's and not the position's -- the names of
//! its places and kinds, the medium of movement, the worth of each kind, the goal's kind, how many move relations the vocabulary names -- and `read(position)` returns the questions
//! of one position with their level 0 and level 1 as plain data. The Python sheet (question_sheet.py) is the oracle it is compared with. Text is made here and only here.
use cs_game::kernel::KernelGame;
use cs_game::placed::Placed;
use cs_tactics::sheet::cf::CfObstacle;
use cs_tactics::sheet::forcing::{self, ClaimOut, ForcingParams, ReplyOut, ThreatOut};
use cs_tactics::sheet::levels::{self, Level0, Level1, Obstacle, Route, WeakPlace, Way, Why};
use cs_tactics::sheet::medium::Medium;
use cs_tactics::sheet::mv::{self, Event, Loss, MoveAnswer};
use cs_tactics::sheet::reading::Reading;
use cs_tactics::sheet::run::{play_game, Chances, Clock, Costing, Fnd, Found, PlySummary, Qn, Runner, Schedule, DANGER};
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
    spec: Spec,
}

/// What the model is made from, kept as plain data so that every thread of `games` can make its own (a `Model` holds shared caches that do not cross threads).
#[derive(Clone)]
struct Spec {
    variant: String,
    hops: Vec<Hop>,
    overs: Vec<Over>,
    takes: Vec<(String, String)>,
    worth: HashMap<String, f64>,
    goal_kind: Option<String>,
    can_go: usize,
    ledger_worth: HashMap<String, i64>,
    letters: HashMap<String, String>,
}

fn build(sp: &Spec) -> Result<Model, String> {
    if sp.hops.is_empty() { return Err("the game's vocabulary names no medium of movement".into()); }
    let g = KernelGame::new(&sp.variant, 518)?;
    let names = Names::new(g.place_names(), g.kind_names(), cs_game::Game::roles(&g));
    let medium = Medium::new(&names, &sp.hops, &sp.overs, &sp.takes)?;
    let w: Vec<f64> = names.kinds.iter().map(|k| sp.worth.get(k).copied().unwrap_or(0.0)).collect();
    let goal_kind = match &sp.goal_kind { Some(k) => Some(names.kind(k)?), None => None };
    let letters: Vec<String> = names.kinds.iter().map(|k| sp.letters.get(k).cloned().unwrap_or_else(|| k.clone())).collect();
    Ok(Model { names, medium, worth: w, goal_kind, can_go: sp.can_go, ledger_worth: sp.ledger_worth.clone(), letters })
}

/// The schedule and its numbers as one plain value (it crosses threads).
struct Config { p: Params, s: Schedule, table: Option<Chances> }

type GameWire = (Option<String>, Vec<String>);
type GameRes = Result<Vec<Option<PlySummary>>, String>;

fn play_one(sp: &Spec, model: &Model, c: &Config, game: &GameWire) -> GameRes {
    let g = match &game.0 { Some(f) => KernelGame::from_fen(&sp.variant, f)?, None => KernelGame::new(&sp.variant, 518)? };
    play_game(&g, &game.1, model, &c.p, &c.s, c.table.as_ref())
}

/// Many games across `threads` std threads (each makes its own model once and takes the next game from a shared counter); the results are in the order of the games.
fn run_games(sp: &Spec, c: &Config, games: &[GameWire], threads: usize) -> Vec<GameRes> {
    let next = std::sync::atomic::AtomicUsize::new(0);
    let slots: std::sync::Mutex<Vec<Option<GameRes>>> = std::sync::Mutex::new((0..games.len()).map(|_| None).collect());
    let workers = threads.max(1).min(games.len().max(1));
    std::thread::scope(|sc| {
        for _ in 0..workers {
            sc.spawn(|| {
                let model = build(sp);
                loop {
                    let i = next.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    if i >= games.len() { break; }
                    let r = match &model { Ok(m) => play_one(sp, m, c, &games[i]), Err(e) => Err(e.clone()) };
                    slots.lock().unwrap()[i] = Some(r);
                }
            });
        }
    });
    slots.into_inner().unwrap().into_iter().map(|x| x.unwrap_or_else(|| Err("not run".into()))).collect()
}

/// PLAN 78.48 A. One side's forcing read of one position, kept small: per threat move its verdict, value, level, the kinds and worths of its claims, what it took, the longest line.
struct LiteThreat { verdict: &'static str, value: f64, level: u32, claims: Vec<(&'static str, f64)>, took_worth: f64, plies: usize }
struct LiteRead { threats: Vec<LiteThreat>, nodes: u64, cut: bool, seconds: f64, goal_stake: f64 }
/// A position of a game: the read of the side to move and, where the game gives a pass, the read of the other side on the position after the pass.
struct LitePly { ours: LiteRead, theirs: Option<LiteRead>, mover: usize }
type LiteRes = Result<Vec<Option<LitePly>>, String>;

fn lite_read(g: &KernelGame, st: <KernelGame as cs_game::Game>::State, model: &Model, fp: &ForcingParams) -> Option<(LiteRead, usize)> {
    let r = Reading::new(g, st, model).ok()?;
    let t0 = std::time::Instant::now();
    let read = forcing::Forcing::new(&r, fp.clone()).read();
    let seconds = t0.elapsed().as_secs_f64();
    let goal_stake = r.top_worth() * Params::default().goal_stake_multiple;
    let threats = read.threats.iter().map(|t| LiteThreat { verdict: t.verdict.word(), value: t.value, level: t.level, claims: t.claims.iter().map(|c| (c.kind, c.value)).collect(), took_worth: t.took_worth, plies: t.plies }).collect();
    Some((LiteRead { threats, nodes: read.nodes, cut: read.cut, seconds, goal_stake }, r.mover))
}

fn lite_game(sp: &Spec, model: &Model, game: &GameWire, fp: &ForcingParams) -> LiteRes {
    let g = match &game.0 { Some(f) => KernelGame::from_fen(&sp.variant, f)?, None => KernelGame::new(&sp.variant, 518)? };
    let mut st = cs_game::Game::initial(&g);
    let mut out = Vec::with_capacity(game.1.len() + 1);
    for ply in 0..=game.1.len() {
        out.push(lite_read(&g, st.clone(), model, fp).map(|(ours, mover)| {
            let theirs = cs_game::Game::pass(&g, &st).and_then(|p| lite_read(&g, p, model, fp)).map(|x| x.0);
            LitePly { ours, theirs, mover }
        }));
        if ply < game.1.len() {
            let mv = cs_game::Game::moves(&g, &st).into_iter().find(|x| cs_game::Game::move_name(&g, &st, x) == game.1[ply]).ok_or_else(|| format!("ply {ply}: {} is not a legal move", game.1[ply]))?;
            st = cs_game::Game::apply(&g, &st, &mv);
        }
    }
    Ok(out)
}

fn run_lite(sp: &Spec, games: &[GameWire], threads: usize, fp: &ForcingParams) -> Vec<LiteRes> {
    let next = std::sync::atomic::AtomicUsize::new(0);
    let slots: std::sync::Mutex<Vec<Option<LiteRes>>> = std::sync::Mutex::new((0..games.len()).map(|_| None).collect());
    let workers = threads.max(1).min(games.len().max(1));
    std::thread::scope(|sc| {
        for _ in 0..workers {
            sc.spawn(|| {
                let model = build(sp);
                loop {
                    let i = next.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    if i >= games.len() { break; }
                    let r = match &model { Ok(m) => lite_game(sp, m, &games[i], fp), Err(e) => Err(e.clone()) };
                    slots.lock().unwrap()[i] = Some(r);
                }
            });
        }
    });
    slots.into_inner().unwrap().into_iter().map(|x| x.unwrap_or_else(|| Err("not run".into()))).collect()
}

fn lite_dict(py: Python<'_>, x: &LiteRead) -> PyResult<PyObject> {
    let d = PyDict::new(py);
    d.set_item("nodes", x.nodes)?;
    d.set_item("cut", x.cut)?;
    d.set_item("seconds", x.seconds)?;
    d.set_item("goal_stake", x.goal_stake)?;
    d.set_item("threats", x.threats.iter().map(|t| (t.verdict, t.value, t.level, t.claims.clone(), t.took_worth, t.plies)).collect::<Vec<_>>())?;
    d.into_py_any(py)
}

#[allow(clippy::too_many_arguments)]
fn config(reach_k: u32, max_open_exits: usize, clock: &str, max_steps: Option<usize>, budget_cost: Option<f64>, cost_model: Option<Vec<f64>>, chances: Option<Vec<(Vec<String>, Vec<(u32, (i64, i64))>)>>,
          chance_horizon: u32, risk: f64) -> Result<Config, String> {
    let costing = match cost_model { Some(v) if !v.is_empty() => Costing::Fixed(v), Some(_) => return Err("an empty cost model".into()), None => Costing::Measured };
    Ok(Config { p: Params { reach_k, max_open_exits, ..Params::default() }, s: Schedule { clock: Clock::parse(clock)?, risk, chance_horizon, costing, max_steps, budget_cost, ..Schedule::default() },
                table: chances.map(Chances::new) })
}

fn summary_dict(py: Python<'_>, n: &Names, x: &PlySummary) -> PyResult<PyObject> {
    let d = PyDict::new(py);
    d.set_item("ply", x.ply)?;
    d.set_item("mover", n.roles[x.mover].as_str())?;
    d.set_item("steps", x.steps)?;
    d.set_item("targets", x.targets.iter().map(|t| (t.owner.word().to_string(), n.kind_text(t.kind).to_string(), n.place_text(t.at).to_string(), t.worth, t.verdict.clone(), t.danger, t.stake, t.level, t.key.to_vec(), t.live0)).collect::<Vec<_>>())?;
    d.set_item("ways_us", x.ways[0].to_vec())?;
    d.set_item("ways_them", x.ways[1].to_vec())?;
    d.into_py_any(py)
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
        let spec = Spec { variant: variant.to_string(), hops, overs, takes, worth, goal_kind, can_go, ledger_worth, letters };
        let model = build(&spec).map_err(bad)?;
        Ok(SheetModel { variant: variant.to_string(), model, spec })
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

    /// sheet(position, reach_k=2, max_open_exits=1, clock="present", max_steps=None, budget_cost=None, cost_model=None, chances=None, chance_horizon=4, risk=1.0, only=None, finish=True,
    /// moves=False) -> the whole sheet of one position (stage C): the schedule `run` under the clock (`present`, `chance`, `plain`), then, with `finish`, `finish_levels(1)` and `settle_flagged`
    /// (what the Python `read_position` does). `cost_model`: seconds per level (the last for every deeper level) in place of the measured time of each step, which makes the order a function
    /// of the position; `chances`: the table of the Python `Chances` as [(key, [(horizon, (taken, total))])]; `only`: [(side, kind, place)] to ask about. Returns {mover, seconds, questions,
    /// steps, moves}: each question {target, asker, stake, rank, goal, quiet, level, state, status, urgency, spent, live_by_level1, closed_at, best_value, waiting, key, danger, bounds, at_risk,
    /// findings: [{level, status, new, value, cost, mode, detail}]}; steps [(question, level, status, mode, cost)].
    #[pyo3(signature = (position, reach_k=2, max_open_exits=1, clock="present", max_steps=None, budget_cost=None, cost_model=None, chances=None, chance_horizon=4, risk=1.0, only=None, finish=true, moves=false))]
    #[allow(clippy::too_many_arguments)]
    fn sheet(&self, py: Python<'_>, position: &str, reach_k: u32, max_open_exits: usize, clock: &str, max_steps: Option<usize>, budget_cost: Option<f64>, cost_model: Option<Vec<f64>>,
             chances: Option<Vec<(Vec<String>, Vec<(u32, (i64, i64))>)>>, chance_horizon: u32, risk: f64, only: Option<Vec<Unit3>>, finish: bool, moves: bool) -> PyResult<PyObject> {
        let bad = |e: String| pyo3::exceptions::PyValueError::new_err(e);
        let t0 = std::time::Instant::now();
        let g = KernelGame::from_fen(&self.variant, position).map_err(bad)?;
        let s = cs_game::Game::initial(&g);
        let r = Reading::new(&g, s, &self.model).map_err(bad)?;
        let n = &self.model.names;
        let p = Params { reach_k, max_open_exits, ..Params::default() };
        let costing = match cost_model { Some(v) if !v.is_empty() => Costing::Fixed(v), Some(_) => return Err(bad("an empty cost model".into())), None => Costing::Measured };
        let sched = Schedule { clock: Clock::parse(clock).map_err(bad)?, risk, chance_horizon, costing, max_steps, budget_cost, ..Schedule::default() };
        let table = chances.map(Chances::new);
        let only_units: Option<Vec<Unit>> = match only {
            Some(v) => Some(v.iter().map(|(side, kind, place)| Ok(Unit { side: if side == "us" { Side::Us } else { Side::Them }, kind: n.kind(kind)?, at: n.place(place)? })).collect::<Result<Vec<_>, String>>().map_err(bad)?),
            None => None,
        };
        let mut run = Runner::new(&r, &p, &sched, table.as_ref());
        run.run(only_units.as_deref());
        if finish { run.finish_levels(1); run.settle_flagged(); }
        let mv = if moves { Some(run.settler.move_sheet(None)) } else { None };
        let compute = t0.elapsed().as_secs_f64();
        let d = PyDict::new(py);
        d.set_item("mover", n.roles[r.mover].as_str())?;
        d.set_item("seconds", compute)?;
        d.set_item("units", r.units.iter().map(|u| (u.side.word().to_string(), n.kind_text(u.kind).to_string(), n.place_text(u.at).to_string(), r.worth_of(u.kind))).collect::<Vec<_>>())?;
        let mut qs = vec![];
        for q in &run.qs { qs.push(question_dict(py, n, q, sched.risk)?); }
        d.set_item("questions", qs)?;
        d.set_item("steps", run.steps.iter().map(|s| (s.question, s.level, s.status.word().to_string(), s.mode.to_string(), s.cost)).collect::<Vec<_>>())?;
        if let Some(m) = mv {
            d.set_item("moves", m.results.iter().map(|a| answer_dict(py, n, a)).collect::<PyResult<Vec<_>>>()?)?;
            d.set_item("move_order", m.order.clone())?;
        }
        d.into_py_any(py)
    }

    /// game(start, moves, reach_k=2, max_open_exits=1, clock="present", max_steps=None, budget_cost=None, cost_model=None, chances=None, chance_horizon=4, risk=1.0) -> [None | summary per
    /// ply]: the whole sheet of every position of a game (the start, then after each move), made in the engine without returning to the caller between plies; the same sheet as `sheet` makes
    /// at that position (`run`, `finish_levels(1)`, `settle_flagged`) in brief: {ply, mover, steps, targets: [(owner, kind, place, worth, verdict kind, danger, stake, level reached, the record key as a list of four words, whether level 0 read it live)],
    /// ways_us / ways_them: [confirmed, waiting, refuted, not settled] of the ways the deepest level settled, by the side that asks}. None for a position that is over. `start` None: the game's start.
    #[pyo3(signature = (start, moves, reach_k=2, max_open_exits=1, clock="present", max_steps=None, budget_cost=None, cost_model=None, chances=None, chance_horizon=4, risk=1.0))]
    #[allow(clippy::too_many_arguments)]
    fn game(&self, py: Python<'_>, start: Option<String>, moves: Vec<String>, reach_k: u32, max_open_exits: usize, clock: &str, max_steps: Option<usize>, budget_cost: Option<f64>,
            cost_model: Option<Vec<f64>>, chances: Option<Vec<(Vec<String>, Vec<(u32, (i64, i64))>)>>, chance_horizon: u32, risk: f64) -> PyResult<PyObject> {
        let bad = |e: String| pyo3::exceptions::PyValueError::new_err(e);
        let c = config(reach_k, max_open_exits, clock, max_steps, budget_cost, cost_model, chances, chance_horizon, risk).map_err(bad)?;
        let res = play_one(&self.spec, &self.model, &c, &(start, moves)).map_err(bad)?;
        let n = &self.model.names;
        res.iter().map(|x| match x { Some(s) => summary_dict(py, n, s), None => Ok(py.None()) }).collect::<PyResult<Vec<_>>>()?.into_py_any(py)
    }

    /// games(games, threads, reach_k=2, ...) -> [the result of `game` per (start, moves)]: many games across `threads` std threads, the GIL released; each thread makes its own model.
    #[pyo3(signature = (games, threads, reach_k=2, max_open_exits=1, clock="present", max_steps=None, budget_cost=None, cost_model=None, chances=None, chance_horizon=4, risk=1.0))]
    #[allow(clippy::too_many_arguments)]
    fn games(&self, py: Python<'_>, games: Vec<GameWire>, threads: usize, reach_k: u32, max_open_exits: usize, clock: &str, max_steps: Option<usize>, budget_cost: Option<f64>,
             cost_model: Option<Vec<f64>>, chances: Option<Vec<(Vec<String>, Vec<(u32, (i64, i64))>)>>, chance_horizon: u32, risk: f64) -> PyResult<PyObject> {
        let bad = |e: String| pyo3::exceptions::PyValueError::new_err(e);
        let c = config(reach_k, max_open_exits, clock, max_steps, budget_cost, cost_model, chances, chance_horizon, risk).map_err(bad)?;
        let spec = self.spec.clone();
        let res = py.allow_threads(move || run_games(&spec, &c, &games, threads));
        let n = &self.model.names;
        let mut out = vec![];
        for r in res {
            let plies = r.map_err(bad)?;
            out.push(plies.iter().map(|x| match x { Some(s) => summary_dict(py, n, s), None => Ok(py.None()) }).collect::<PyResult<Vec<_>>>()?);
        }
        out.into_py_any(py)
    }

    /// forcing(position, fuse=None, goal_nodes=None) -> the FORCING READ of the side to move (plan 78.46), plain data: {mover, legal, found, confirmed, unsettled, no_claim,
    /// unread, nodes, fuse, cut, level, seconds, threats: [{move, verdict, value, took, took_worth, claims, replies, level, plies}]}; see `forcing.rs`. ValueError when the position is over.
    /// `shortcut=False` reads every reply in full (the read as built before plan 78.47 B); the default answers a reply that touches nothing the read looked at from the stored result of an earlier one.
    #[pyo3(signature = (position, fuse=None, goal_nodes=None, shortcut=None))]
    fn forcing(&self, py: Python<'_>, position: &str, fuse: Option<u64>, goal_nodes: Option<u64>, shortcut: Option<bool>) -> PyResult<PyObject> {
        let bad = |e: String| pyo3::exceptions::PyValueError::new_err(e);
        let t0 = std::time::Instant::now();
        let g = KernelGame::from_fen(&self.variant, position).map_err(bad)?;
        let s = cs_game::Game::initial(&g);
        let r = Reading::new(&g, s, &self.model).map_err(bad)?;
        let mut fp = ForcingParams::default();
        if let Some(f) = fuse { fp.fuse = f; }
        if let Some(n) = goal_nodes { fp.goal_nodes = n; }
        if let Some(b) = shortcut { fp.shortcut = b; }
        let read = forcing::Forcing::new(&r, fp).read();
        let d = PyDict::new(py);
        d.set_item("mover", self.model.names.roles[r.mover].as_str())?;
        d.set_item("seconds", t0.elapsed().as_secs_f64())?;
        for (k, v) in [("legal", read.legal), ("found", read.found), ("confirmed", read.confirmed), ("not_borne_out", read.not_borne_out), ("unsettled", read.unsettled), ("no_claim", read.no_claim), ("unread", read.unread)] { d.set_item(k, v)?; }
        d.set_item("nodes", read.nodes)?;
        d.set_item("fuse", read.fuse)?;
        d.set_item("cut", read.cut)?;
        d.set_item("level", read.level)?;
        d.set_item("cost", cost_dict(py, &read.cost)?)?;
        d.set_item("threats", read.threats.iter().map(|t| threat_dict(py, t)).collect::<PyResult<Vec<_>>>()?)?;
        d.into_py_any(py)
    }

    /// forcing_games(games, threads, fuse=None, goal_nodes=None) -> per game, per position (None where over): {mover, ours, theirs}: the forcing read of the side to move, and of the other
    /// side on the position after the game's pass (None where the game gives none). Each read: {nodes, cut, seconds, goal_stake, threats: [(verdict, value, level, [(claim kind, worth)], took_worth, plies)]}.
    /// Many games across `threads` std threads, the GIL released (plan 78.48 A).
    #[pyo3(signature = (games, threads, fuse=None, goal_nodes=None))]
    fn forcing_games(&self, py: Python<'_>, games: Vec<GameWire>, threads: usize, fuse: Option<u64>, goal_nodes: Option<u64>) -> PyResult<PyObject> {
        let bad = |e: String| pyo3::exceptions::PyValueError::new_err(e);
        let mut fp = ForcingParams::default();
        if let Some(f) = fuse { fp.fuse = f; }
        if let Some(n) = goal_nodes { fp.goal_nodes = n; }
        let spec = self.spec.clone();
        let res = py.allow_threads(move || run_lite(&spec, &games, threads, &fp));
        let n = &self.model.names;
        let mut out = vec![];
        for r in res {
            let plies = r.map_err(bad)?;
            out.push(plies.iter().map(|x| match x {
                Some(p) => {
                    let d = PyDict::new(py);
                    d.set_item("mover", n.roles[p.mover].as_str())?;
                    d.set_item("ours", lite_dict(py, &p.ours)?)?;
                    d.set_item("theirs", match &p.theirs { Some(t) => lite_dict(py, t)?, None => py.None() })?;
                    d.into_py_any(py)
                }
                None => Ok(py.None()),
            }).collect::<PyResult<Vec<_>>>()?);
        }
        out.into_py_any(py)
    }

    /// forcing_check(position, mv, depth, nodes) -> {value, depth, cut, line, nodes, win}: the ledger search (the sheet's worth table, no pruning) from the position after the move `mv`, deepened to
    /// `depth` plies under `nodes`; `value` is the future for the mover (the move's own taking not counted), `win` the ledger's number of a won game.
    fn forcing_check(&self, py: Python<'_>, position: &str, mv: &str, depth: i32, nodes: u64) -> PyResult<PyObject> {
        let bad = |e: String| pyo3::exceptions::PyValueError::new_err(e);
        let g = KernelGame::from_fen(&self.variant, position).map_err(bad)?;
        let s = cs_game::Game::initial(&g);
        let r = Reading::new(&g, s, &self.model).map_err(bad)?;
        let m = cs_game::Game::moves(&g, &r.s).into_iter().find(|m| cs_game::Game::move_name(&g, &r.s, m) == mv).ok_or_else(|| bad(format!("{mv} is not a legal move")))?;
        let (value, depth, cut, line, n) = forcing::ledger_check(&r, &r.s, &m, depth, nodes);
        let d = PyDict::new(py);
        d.set_item("value", value)?;
        d.set_item("depth", depth)?;
        d.set_item("cut", cut)?;
        d.set_item("line", line)?;
        d.set_item("nodes", n)?;
        d.set_item("win", forcing::win_value())?;
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

fn weak_of(n: &Names, w: &WeakPlace) -> (String, Vec<Unit3>, Vec<Unit3>, usize, Vec<(u32, Unit3, String)>) {
    (n.place_text(w.place).to_string(), w.guards.iter().map(|u| unit3(n, u)).collect(), w.strikers.iter().map(|u| unit3(n, u)).collect(), w.need,
     w.candidates.iter().map(|(h, u, a)| (*h, unit3(n, u), n.place_text(*a).to_string())).collect())
}

fn fnd_dict(py: Python<'_>, n: &Names, f: &Fnd) -> PyResult<PyObject> {
    let d = PyDict::new(py);
    d.set_item("level", f.level)?;
    d.set_item("status", f.status.word())?;
    d.set_item("new", f.new)?;
    d.set_item("value", f.value)?;
    d.set_item("cost", f.cost)?;
    d.set_item("mode", f.mode())?;
    match &f.found {
        Found::Quiet { unit, strikes, places } => {
            let q = PyDict::new(py);
            q.set_item("unit", unit3(n, unit))?; q.set_item("strikes", *strikes)?; q.set_item("places", *places)?;
            d.set_item("detail", q)?;
        }
        Found::L0(l) => d.set_item("detail", level0_dict(py, n, l)?)?,
        Found::L1(l, weak) => {
            let dd = level1_dict(py, n, l)?;
            if let Some(w) = weak { dd.bind(py).downcast::<PyDict>()?.set_item("weak", w.iter().map(|x| weak_of(n, x)).collect::<Vec<_>>())?; }
            d.set_item("detail", dd)?;
        }
        Found::Deep(x) => d.set_item("detail", finding_dict(py, n, x)?)?,
    }
    d.into_py_any(py)
}

fn question_dict(py: Python<'_>, n: &Names, q: &Qn, risk: f64) -> PyResult<PyObject> {
    let d = PyDict::new(py);
    d.set_item("target", q.target.as_ref().map(|u| unit3(n, u)))?;
    d.set_item("asker", q.asker.word())?;
    d.set_item("stake", q.stake)?;
    d.set_item("rank", q.rank)?;
    d.set_item("goal", q.goal)?;
    d.set_item("quiet", q.quiet)?;
    d.set_item("level", q.level)?;
    d.set_item("state", if q.open { "open" } else { "closed" })?;
    d.set_item("status", q.status.word())?;
    d.set_item("urgency", q.urgency)?;
    d.set_item("spent", q.spent)?;
    d.set_item("live_by_level1", q.live_by_level1)?;
    d.set_item("closed_at", q.closed_at)?;
    d.set_item("best_value", q.best_value)?;
    d.set_item("waiting", q.waiting)?;
    let key = q.question_key();
    d.set_item("danger", DANGER.contains(&key[1].as_str()))?;
    d.set_item("key", key.to_vec())?;
    d.set_item("bounds", q.stake_bounds())?;
    d.set_item("at_risk", q.stake_at_risk(risk))?;
    d.set_item("findings", q.findings.iter().map(|f| fnd_dict(py, n, f)).collect::<PyResult<Vec<_>>>()?)?;
    d.into_py_any(py)
}

type EventWire =(i64, String, String, String, Option<(i64, String, String)>, Option<String>, Option<String>);

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

fn claim_dict<'py>(py: Python<'py>, c: &ClaimOut) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("target", c.target.clone())?;
    d.set_item("kind", c.kind)?;
    d.set_item("value", c.value)?;
    d.set_item("line", c.line.clone())?;
    Ok(d)
}

fn reply_dict<'py>(py: Python<'py>, r: &ReplyOut) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("reply", r.reply.clone())?;
    d.set_item("took", r.took.clone())?;
    d.set_item("took_worth", r.took_worth)?;
    d.set_item("stands", r.stands.as_ref().map(|c| claim_dict(py, c)).transpose()?)?;
    d.set_item("deeper", r.deeper.clone())?;
    d.set_item("line", r.line.clone())?;
    d.set_item("value", r.value)?;
    d.set_item("open", r.open)?;
    d.set_item("ends", r.ends)?;
    Ok(d)
}

/// The cost of a forcing read per kind of work (plan 78.47 A): {work: {positions, frame_nodes, goal_nodes, memory, seconds}} (a kind read inside a deeper threat move of ours is named "... (deeper)"), and the replies.
fn cost_dict(py: Python<'_>, c: &forcing::Cost) -> PyResult<PyObject> {
    let d = PyDict::new(py);
    for i in 0..14 {
        let e = PyDict::new(py);
        e.set_item("positions", c.positions[i])?;
        e.set_item("frame_nodes", c.frame_nodes[i])?;
        e.set_item("goal_nodes", c.goal_nodes[i])?;
        e.set_item("memory", c.memory[i])?;
        e.set_item("seconds", c.seconds[i])?;
        d.set_item(format!("{}{}", forcing::WORK[i % 7], if i >= 7 { " (deeper)" } else { "" }), e)?;
    }
    d.set_item("replies_took", c.replies[0])?;
    d.set_item("replies_quiet", c.replies[1])?;
    d.set_item("unchanged_took", c.replies_unchanged[0])?;
    d.set_item("unchanged_quiet", c.replies_unchanged[1])?;
    d.set_item("untouched", c.untouched)?;
    d.set_item("why", c.why.to_vec())?;
    d.into_py_any(py)
}

fn threat_dict(py: Python<'_>, t: &ThreatOut) -> PyResult<PyObject> {
    let d = PyDict::new(py);
    d.set_item("move", t.mv.clone())?;
    d.set_item("verdict", t.verdict.word())?;
    d.set_item("value", t.value)?;
    d.set_item("took", t.took.clone())?;
    d.set_item("took_worth", t.took_worth)?;
    d.set_item("claims", t.claims.iter().map(|c| claim_dict(py, c)).collect::<PyResult<Vec<_>>>()?)?;
    d.set_item("replies", t.replies.iter().map(|r| reply_dict(py, r)).collect::<PyResult<Vec<_>>>()?)?;
    d.set_item("level", t.level)?;
    d.set_item("plies", t.plies)?;
    d.set_item("search", match &t.searched {
        None => py.None(),
        Some(c) => {
            let e = PyDict::new(py);
            e.set_item("status", c.status)?;
            e.set_item("claimed", c.claimed)?;
            e.set_item("value", c.value)?;
            e.set_item("depth", c.depth)?;
            e.set_item("depth_read", c.depth_read)?;
            e.set_item("cut", c.cut)?;
            e.set_item("line", c.line.clone())?;
            e.set_item("nodes", c.nodes)?;
            e.into_py_any(py)?
        }
    })?;
    d.into_py_any(py)
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
    d.set_item("time", e.time.as_ref().map(|(m, u, side)| (*m, unit3(n, u), side.word().to_string())))?;
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
