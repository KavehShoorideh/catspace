//! PLAN 78.52: the Python entry of the layer-1 graph (cs_tactics::sheet::base_graph and base_statements). A `BaseGraph` holds the static part of one game (built once from the rules'
//! description through `Placed::steps`); `position(text)` returns the per-position part as plain node and link lists. Game-free: every name arrives as data.
use cs_game::kernel::KernelGame;
use cs_game::placed::Placed;
use cs_game::Game;
use cs_tactics::sheet::base_graph::{LinkRec, LinkType, NodeRec, NodeType, PositionPart, Static};
use cs_tactics::sheet::base_statements;
use cs_tactics::sheet::fates::{self, Cell, RowBoard};
use cs_tactics::sheet::patterns::{self, Board};
use cs_tactics::sheet::patterns::{FxHashMap, FxHashSet};
use cs_tactics::sheet::Names;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::IntoPyObjectExt;
use std::collections::HashMap;

#[pyclass(unsendable)]
pub struct BaseGraph {
    variant: String,
    st: Static,
    /// PLAN 79.17: what the board gives once for the measure of progress (cs_tactics::measure)
    geo: cs_tactics::measure::Geometry,
    /// 78.58: the worth of each kind (by kind index) the exchange reads, set from the estimator's record; None: no exchange.
    kv: Option<Vec<i64>>,
    /// 78.60: the band (low, high) of the worth of each kind, from the estimator record; and the rules of a line's end (all off until set).
    bands: Option<Vec<(i64, i64)>>,
    rules: cs_tactics::sheet::questions::Rules,
    /// PLAN 79.11 B, MOVED BY PLAN 79.19 C: the memory of the game at hand of a caller's slot (a name the caller gives; its open tactics and the goal held are its slot `carried`), taken by the next `wake_decide` of the same slot and put back after it. The binding holds them so that the caller carries nothing across the language boundary.
    games: std::cell::RefCell<HashMap<String, cs_memory::store::GameMemory>>,
    /// PLAN 79.19: the key of this game's partition of the long-term memory: what the game reports (its name) and the content address of its rules (the description it was built from)
    key: cs_memory::store::GameKey,
}


// PLAN 79.19 C: the tables a caller gives, as the types of the decision crate (the one conversion, used by `wake_decide` and `memory_write`).
type ChancesWire = (Vec<(String, u32, u32)>, Vec<(String, u32, u32)>, (u32, u32), u32);
type ReasonsWire = (Vec<u64>, Vec<(usize, String, String, f64)>, usize, Vec<u8>, i32);
type GoalTableWire = (u32, u32, Vec<(Vec<String>, Vec<(String, u32, u32)>)>, (u32, u32));

fn chances_of(c: Option<ChancesWire>) -> Option<cs_decision::estimate::ChanceTable> {
    c.map(|c| cs_decision::estimate::ChanceTable { by_key: c.0.into_iter().map(|x| (x.0, (x.1, x.2))).collect(), by_facts: c.1.into_iter().map(|x| (x.0, (x.1, x.2))).collect(), total: c.2, floor: c.3 })
}

fn reasons_of(r: Option<ReasonsWire>) -> Option<cs_decision::pick::Reasons> {
    use cs_decision::pick::{Dir, Reason, Reasons, Side};
    r.map(|r| Reasons { keys: r.0, reasons: r.1.into_iter().map(|x| Reason { column: x.0, side: if x.1 == "mover" { Side::Mover } else { Side::Other }, dir: if x.2 == "up" { Dir::Up } else { Dir::Down }, weight: x.3 }).collect(), max_size: r.2, levels: r.3, tie_digits: r.4 })
}

/// PLAN 79.13 C: the table of the chance of a goal counted in sleep: (the plies of its event, the floor, the levels as (the names of the fields, the cells (key, goals, reached)), the counts of all goals (goals, reached)); None: the wake reads no such table
fn goal_table_of(t: Option<GoalTableWire>) -> Result<Option<cs_decision::goal_table::GoalTable>, String> {
    use cs_decision::goal_table::{Field, GoalTable, Level};
    let Some((k, floor, levels, total)) = t else { return Ok(None) };
    let mut ls = vec![];
    for (fields, cells) in levels {
        let fs: Result<Vec<Field>, String> = fields.iter().map(|n| Field::named(n).ok_or_else(|| format!("no such field of the table of the chance of a goal: {n}"))).collect();
        ls.push(Level { fields: fs?, cells: cells.into_iter().map(|c| (c.0, (c.1, c.2))).collect() });
    }
    Ok(Some(GoalTable { k, floor, levels: ls, total }))
}

type NodeWire = (String, String);
type LinkWire = (String, u32, u32, i32, i32);

fn nodes_of(v: &[NodeRec]) -> Vec<NodeWire> { v.iter().map(|n| (n.kind.word().to_string(), n.id.clone())).collect() }
fn links_of(v: &[LinkRec]) -> Vec<LinkWire> { v.iter().map(|l| (l.kind.word().to_string(), l.a, l.b, l.via, l.value)).collect() }
pub(crate) fn bad(e: String) -> PyErr { pyo3::exceptions::PyValueError::new_err(e) }

pub(crate) type GameWire = (Option<String>, Vec<String>);

/// The boards of the wanted plies of one game: (row number within the call, the board, the position part's units by (owner is the side to move, kind) counts are made by the callers).
fn walk_rows<T>(variant: &str, st: &Static, goal: Option<u8>, game: &GameWire, wanted: &[usize], mut f: impl FnMut(usize, &Board, &cs_tactics::sheet::base_graph::PositionPart) -> T) -> Result<Vec<Option<T>>, String> {
    let g = match &game.0 { Some(text) => KernelGame::from_fen(variant, text)?, None => KernelGame::new(variant, 518)? };
    let mut s = g.initial();
    let last = wanted.iter().copied().max().map_or(0, |m| m + 1);
    let mut out: Vec<Option<T>> = (0..wanted.len()).map(|_| None).collect();
    for ply in 0..last.min(game.1.len() + 1) {
        for (i, _) in wanted.iter().enumerate().filter(|(_, w)| **w == ply) {
            let p = st.position(&g, &s);
            if p.mover.is_some() { let b = Board::new(st, &p, goal); out[i] = Some(f(i, &b, &p)); }
        }
        if ply < game.1.len() {
            let mv = g.find_move(&s, &game.1[ply]).ok_or_else(|| format!("ply {ply}: {} is not a legal move", game.1[ply]))?;
            s = g.apply(&s, &mv);
        }
    }
    Ok(out)
}

fn run_pool<T: Send>(n: usize, threads: usize, f: impl Fn(usize) -> Result<T, String> + Sync) -> Result<Vec<T>, String> {
    let next = std::sync::atomic::AtomicUsize::new(0);
    let slots: std::sync::Mutex<Vec<Option<Result<T, String>>>> = std::sync::Mutex::new((0..n).map(|_| None).collect());
    std::thread::scope(|sc| {
        for _ in 0..threads.max(1).min(n.max(1)) {
            sc.spawn(|| loop {
                let i = next.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                if i >= n { break; }
                let r = f(i);
                slots.lock().unwrap()[i] = Some(r);
            });
        }
    });
    slots.into_inner().unwrap().into_iter().map(|x| x.unwrap_or_else(|| Err("not run".into()))).collect()
}

/// One game walked for the fates (78.54): the sorted units of every ply, the mover of every ply, and the boards of the wanted rows (those with a side to move).
type Walked = (Vec<Vec<Cell>>, Vec<Option<usize>>, Vec<RowBoard>);

fn walk_game(variant: &str, st: &Static, goal: Option<u8>, game: &GameWire, wanted: &[usize]) -> Result<Walked, String> {
    let g = match &game.0 { Some(text) => KernelGame::from_fen(variant, text)?, None => KernelGame::new(variant, 518)? };
    let mut s = g.initial();
    let n = game.1.len();
    let (mut plies, mut movers, mut boards) = (vec![], vec![], vec![]);
    for ply in 0..=n {
        let mut cells = g.cells(&s);
        cells.sort_by_key(|(o, k, p)| (*o, *k, *p));
        if wanted.contains(&ply) {
            let p = st.position(&g, &s);
            if p.mover.is_some() { let b = Board::new(st, &p, goal); boards.push(RowBoard::of(ply, b, &p)); }
        }
        plies.push(cells);
        movers.push(g.mover(&s));
        if ply < n {
            let mv = g.find_move(&s, &game.1[ply]).ok_or_else(|| format!("ply {ply}: {} is not a legal move", game.1[ply]))?;
            s = g.apply(&s, &mv);
        }
    }
    Ok((plies, movers, boards))
}

/// Threads each keeping its own accumulator of the games they pull; the accumulators are returned for the caller to merge.
pub(crate) fn run_acc<A: Send>(n: usize, threads: usize, make: impl Fn() -> A + Sync, f: impl Fn(usize, &mut A) -> Result<(), String> + Sync) -> Result<Vec<A>, String> {
    let next = std::sync::atomic::AtomicUsize::new(0);
    let out: std::sync::Mutex<Vec<Result<A, String>>> = std::sync::Mutex::new(vec![]);
    std::thread::scope(|sc| {
        for _ in 0..threads.max(1).min(n.max(1)) {
            sc.spawn(|| {
                let mut acc = make();
                loop {
                    let i = next.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    if i >= n { break; }
                    if let Err(e) = f(i, &mut acc) { out.lock().unwrap().push(Err(e)); return; }
                }
                out.lock().unwrap().push(Ok(acc));
            });
        }
    });
    out.into_inner().unwrap().into_iter().collect()
}

impl BaseGraph {
    fn goal_of(&self, goal_kind: &Option<String>) -> PyResult<Option<u8>> {
        match goal_kind { Some(k) => Ok(Some(self.st.names.kind(k).map_err(bad)?)), None => Ok(None) }
    }

    fn game(&self, text: &str) -> PyResult<(KernelGame, <KernelGame as Game>::State)> {
        let g = KernelGame::from_fen(&self.variant, text).map_err(bad)?;
        let s = g.initial();
        Ok((g, s))
    }
}

#[pymethods]
impl BaseGraph {
    /// BaseGraph(variant, description, static_relations, objects): the game by its variant (its places, kinds and sides), the description's text, the static relations and the objects
    /// (name -> component sorts) its graph vocabulary binds.
    #[new]
    fn new(variant: &str, description: &str, static_relations: Vec<String>, objects: Vec<(String, Vec<String>)>) -> PyResult<BaseGraph> {
        let g = KernelGame::new(variant, 518).map_err(bad)?;
        let names = Names::new(g.place_names(), g.kind_names(), g.roles());
        let steps = g.steps(description, &static_relations, &objects).map_err(bad)?;
        let st = Static::new(&steps, &names).map_err(bad)?;
        let geo = cs_tactics::measure::Geometry::new(&st);
        Ok(BaseGraph { variant: variant.to_string(), st, geo, kv: None, bands: None, rules: Default::default(), games: Default::default(), key: cs_memory::store::GameKey::of(variant, description) })
    }

    /// kinds() -> the game's own words for its kinds, in the order of the numbers the measure's groups use.
    fn kinds(&self) -> Vec<String> { self.st.names.kinds.clone() }

    /// PLAN 79.17 (the measure of progress, cs_tactics::measure). measure_groups(text, side) -> [(theirs, kind)]: the groups of units present (the measure's side is the role number `side`). The kinds are numbers, the names are the game's own (`kinds()`).
    fn measure_groups(&self, text: &str, side: usize) -> PyResult<Vec<(bool, u8)>> {
        let (g, s) = self.game(text)?;
        Ok(cs_tactics::measure::groups_of(&g, &s, side).into_iter().map(|x| (x.theirs, x.kind)).collect())
    }

    /// measure_library(groups) -> [atom name]: every atom the groups and the board's line families give.
    fn measure_library(&self, groups: Vec<(bool, u8)>) -> Vec<String> {
        let gs: Vec<cs_tactics::measure::Group> = groups.into_iter().map(|(theirs, kind)| cs_tactics::measure::Group { theirs, kind }).collect();
        cs_tactics::measure::library(&gs, &self.geo).iter().map(|a| a.name()).collect()
    }

    /// measure_atoms(texts, side, atoms) -> [[value of each atom]] a position (the batch is read on the calling thread, a Pool of processes spreads it).
    fn measure_atoms(&self, py: Python<'_>, texts: Vec<String>, side: usize, atoms: Vec<String>) -> PyResult<Vec<Vec<i64>>> {
        let at: Vec<cs_tactics::measure::Atom> = atoms.iter().map(|a| cs_tactics::measure::Atom::parse(a)).collect::<Result<_, _>>().map_err(bad)?;
        let games: Vec<(KernelGame, <KernelGame as Game>::State)> = texts.iter().map(|t| self.game(t)).collect::<PyResult<_>>()?;
        let geo = &self.geo;
        Ok(py.allow_threads(|| games.iter().map(|(g, s)| cs_tactics::measure::atom_values(&at, &cs_tactics::measure::Reader::new(g, geo, s, side))).collect()))
    }

    /// measure_value(text, side, term) -> the tuple of numbers of a term in a position.
    fn measure_value(&self, text: &str, side: usize, term: &str) -> PyResult<Vec<i64>> {
        let t = cs_tactics::measure::Term::parse(term).map_err(bad)?;
        let (g, s) = self.game(text)?;
        Ok(cs_tactics::measure::value(&t, &cs_tactics::measure::Reader::new(&g, &self.geo, &s, side)))
    }

    /// measure_lowers(text, side, term, held, d, seconds, history) -> (first move or None, our moves the line takes, the end of the game is what was found, the clock cut the question, nodes): October's criterion at one position: a move of ours after which, within `d` of our moves and against every reply, the term is below `held` (or the end of the game is ours).
    #[pyo3(signature = (text, side, term, held, d, seconds, history=None))]
    fn measure_lowers(&self, py: Python<'_>, text: &str, side: usize, term: &str, held: Vec<i64>, d: u32, seconds: f64, history: Option<Vec<u64>>) -> PyResult<(Option<String>, u32, bool, bool, u64)> {
        let t = cs_tactics::measure::Term::parse(term).map_err(bad)?;
        let (g, s) = self.game(text)?;
        let hist: std::collections::HashSet<u64> = history.unwrap_or_default().into_iter().collect();
        let geo = &self.geo;
        let r = py.allow_threads(|| { let mut b = cs_tactics::Budget::new(seconds, 400_000_000); let (f, cut) = cs_tactics::measure::lowering(&g, geo, &s, side, &t, &held, d, &hist, &mut b); (f, cut, b.nodes) });
        let name = r.0.as_ref().map(|l| g.move_name(&s, &g.moves(&s)[l.first]));
        Ok((name, r.0.as_ref().map_or(0, |l| l.moves), r.0.as_ref().is_some_and(|l| l.game_end), r.1, r.2))
    }

    /// static_part() -> {nodes: [(type, id)], links: [(type, a, b, via, value)], directions, lines: [[places]]}: the part that is the game's and not the position's.
    fn static_part(&self, py: Python<'_>) -> PyResult<PyObject> {
        let d = PyDict::new(py);
        d.set_item("nodes", nodes_of(&self.st.nodes))?;
        d.set_item("links", links_of(&self.st.links))?;
        d.set_item("directions", self.st.directions.clone())?;
        d.set_item("lines", self.st.lines.iter().map(|l| l.places.iter().map(|p| self.st.names.places[*p as usize].clone()).collect::<Vec<_>>()).collect::<Vec<_>>())?;
        d.into_py_any(py)
    }

    /// position(text) -> {mover, offset (the number of static nodes: the indices of these nodes and links continue after them), nodes, links}.
    fn position(&self, py: Python<'_>, text: &str) -> PyResult<PyObject> {
        let (g, s) = self.game(text)?;
        let p = self.st.position(&g, &s);
        let d = PyDict::new(py);
        d.set_item("mover", p.mover.map(|m| self.st.names.roles[m].clone()))?;
        d.set_item("offset", self.st.nodes.len())?;
        d.set_item("nodes", nodes_of(&p.nodes))?;
        d.set_item("links", links_of(&p.links))?;
        d.into_py_any(py)
    }

    /// PLAN 79.05 (the wake, cs_decision::wake): wake_graph_read(text, goal_kind, places) -> {"lines": [(owner, place, None | [{"unit", "blocker", "bearers", "holders"}])], "new_bearer_moves": [(move name, [place names])]}. A unit is (role, kind, place) by the game's words.
    /// `lines`: for the goal unit of each side, the lines of the other side onto it that exactly one unit shuts (None where the game gives no pass for the side not to move); `new_bearer_moves`: the legal moves after which more units of the mover bear on one of `places`.
    fn wake_graph_read(&self, py: Python<'_>, text: &str, goal_kind: String, places: Vec<String>) -> PyResult<PyObject> {
        use cs_decision::wake;
        let (g, s) = self.game(text)?;
        let n = &self.st.names;
        let goal = n.kind(&goal_kind).map_err(bad)?;
        let ps: Vec<u16> = places.iter().map(|p| n.place(p)).collect::<Result<_, _>>().map_err(bad)?;
        let uw = |u: &wake::UnitRef| (n.roles[u.owner].clone(), n.kind_text(u.kind).to_string(), n.place_text(u.at).to_string());
        let lines: Vec<(String, String, Option<Vec<PyObject>>)> = wake::goal_unit_lines(&self.st, &g, &s, goal).into_iter().map(|(owner, at, ls)| {
            let ls = ls.map(|v| v.into_iter().map(|l| {
                let d = PyDict::new(py);
                let _ = d.set_item("unit", uw(&l.unit));
                let _ = d.set_item("blocker", uw(&l.blocker));
                let _ = d.set_item("bearers", l.bearers.iter().map(&uw).collect::<Vec<_>>());
                let _ = d.set_item("holders", l.holders.iter().map(&uw).collect::<Vec<_>>());
                d.into_any().unbind()
            }).collect());
            (n.roles[owner].clone(), n.place_text(at).to_string(), ls)
        }).collect();
        let nb: Vec<(String, Vec<String>)> = wake::new_bearer_moves(&self.st, &g, &s, &ps).into_iter().map(|(m, a)| (m, a.iter().map(|p| n.place_text(*p).to_string()).collect())).collect();
        let d = PyDict::new(py);
        d.set_item("lines", lines)?;
        d.set_item("new_bearer_moves", nb)?;
        d.into_py_any(py)
    }

    /// PLAN 79.06 (the wake's threat list, cs_decision::threats): wake_threats(text, worth, half, goal_kind, rows, every_sides) -> {"mover","other","exchange","dropped","goal": [row], "dropped_counts", "cost": [(round, seconds, queries, nodes)]}.
    /// `worth`/`half`: the measured record (the game's words for the kinds). `rows`: None (the graph's candidates) or the frame read's rows as (role, kind, place, class, by [(role, kind, place)], response_ix [(from, to, how, unit moves)], stake).
    /// `every_sides`: None (the mover) or the roles whose goals are wanted. A unit is (role, kind, place) by the game's words. Exists for the unit check of the Rust list against its Python mirror; the wake calls the Rust function.
    /// PLAN 79.13 C: the chooser's mode may also be "gain" (the owner's rule read plainly: among the moves held equal the largest gain of our chains' stake times chance less theirs); `goal_table` (None: none) is the table of the chance of a goal counted in sleep, (plies of its event, floor, [(field names, [(key, goals, reached)])], (goals, reached) in all), read in place of `chances` wherever a goal's chance is read.
    /// PLAN 79.08 THE WHOLE WAKE, ONE BINDING (SYSTEM_SPEC 8): `wake_decide(text, history, seconds, depth, worth, half, goal_kind, pieces, budget, named_budget, keys, frame_fuses, chances, reasons)` -> the decision as a JSON string (the move, the chain of claims, the time of each step, the moves named and held equal, the decision
    /// log in words). `pieces`: any of "carried", "frames", "threats", "goals", "two_threat_goals", "reasons_pick", "heads", and (PLAN 79.10) "after_move", "direct_names"; none is the old player. `keys`: the learned seeds of the frame read (None: the rows are the graph's); `frame_fuses`: (largest seed, fuse, row fuse, answer fuse); `chances`: (by key, by facts, (goals,
    /// reached) in all, floor), each cell (name, goals, reached); `reasons`: (pattern keys, [(column, side "mover"/"other", direction "up"/"down", weight)], largest seed, levels, tie digits). The frame read reads the kind values, bands and rules this graph was given (`set_kind_values`, `set_kind_bands`, `set_rules`) and its learned profile.
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    #[pyo3(signature = (text, history, seconds, depth, worth, half, goal_kind, pieces, budget, named_budget, keys, frame_fuses, chances, reasons, ledger_share=None, carry_slot=None, chooser=None, goal_table=None, pre_share_scale=None, measure=None, long_term=None, hold=None))]
    fn wake_decide(&self, py: Python<'_>, text: &str, history: Vec<u64>, seconds: f64, depth: i32, worth: HashMap<String, i64>, half: HashMap<String, f64>, goal_kind: Option<String>, pieces: Vec<String>, budget: usize, named_budget: usize,
                   keys: Option<Vec<u64>>, frame_fuses: Option<(usize, u64, u64, u64)>, chances: Option<(Vec<(String, u32, u32)>, Vec<(String, u32, u32)>, (u32, u32), u32)>,
                   reasons: Option<(Vec<u64>, Vec<(usize, String, String, f64)>, usize, Vec<u8>, i32)>, ledger_share: Option<f64>, carry_slot: Option<String>, chooser: Option<(String, HashMap<String, f64>, u64)>, goal_table: Option<(u32, u32, Vec<(Vec<String>, Vec<(String, u32, u32)>)>, (u32, u32))>, pre_share_scale: Option<f64>, measure: Option<(String, u32)>, long_term: Option<String>, hold: Option<String>) -> PyResult<String> {
        use cs_decision::estimate::ChanceTable;
        use cs_decision::frame_rows::FrameRead;
        use cs_decision::pick::{Dir, Reason, Reasons, Side};
        use cs_decision::threat_verdict::Record;
        use cs_decision::memory::{Parts, WakeMemory};
        use cs_decision::wake::{decide, Pieces, WakeInputs, WakeParams};
        use cs_memory::store::{GameMemory, Partition, Reader};
        let (g, s) = self.game(text)?;
        let goal = self.goal_of(&goal_kind)?;
        let mut pc = Pieces::old_player();
        let (mut carry_graph, mut chance_order, mut gain_mend, mut diag_first, mut read_on) = (false, false, false, false, false);
        for w in &pieces {
            match w.as_str() {
                "carried" => pc.carried = true, "frames" => pc.frames = true, "threats" => pc.threats = true, "goals" => pc.goals = true, "two_threat_goals" => pc.two_threat_goals = true, "reasons_pick" => pc.reasons_pick = true, "heads" => pc.heads = true, "graph_reads" => pc.graph_reads = true, "after_move" => pc.after_move = true, "direct_names" => pc.direct_names = true, "carry_graph" => carry_graph = true, "chance_order" => chance_order = true, "gain_mend" => gain_mend = true, "diag_first" => diag_first = true, "read_on" => read_on = true,
                other => return Err(bad(format!("no such piece of the wake: {other}"))),
            }
        }
        let frame = keys.map(|k| { let f = frame_fuses.unwrap_or((3, 1500, 30_000, 1500)); FrameRead { want: k, params: cs_tactics::sheet::threat_frames::FrameReadParams { max_size: f.0, fuse: f.1, row_fuse: f.2, answer_fuse: f.3, conditional: true, kv: self.kv.clone(), bands: self.bands.clone(), rules: self.rules, deadline: None } } });
        let (chances, reasons) = (chances_of(chances), reasons_of(reasons));
        // PLAN 79.12 B: the last layer among the moves held equal: (mode "fixed" | "weights" | "explore", weights by the name of an input, the seed of one decision); None: the rule decides and no row is logged
        let chooser = match chooser {
            None => None,
            Some((mode, w, seed)) => {
                use cs_decision::chooser::{ChooserSpec, Mode};
                let m = match mode.as_str() { "fixed" => Mode::Fixed, "weights" => Mode::Weights, "explore" => Mode::Explore, "gain" => Mode::Gain, other => return Err(bad(format!("no such mode of the chooser: {other}"))) };
                let named: Vec<(String, f64)> = w.into_iter().collect();
                Some(ChooserSpec::new(m, &named, seed).map_err(bad)?)
            }
        };
        // PLAN 79.13 C: the table of the chance of a goal counted in sleep: (the plies of its event, the floor, the levels as (the names of the fields, the cells (key, goals, reached)), the counts of all goals (goals, reached)); None: the wake reads no such table
        let goal_table = goal_table_of(goal_table).map_err(bad)?;
        let judge = cs_tactics::ledger::Worth { name: "record".into(), table: worth.clone(), ours_first: false };
        // PLAN 79.19 C: the wake reads the tables and the goal through the two memories. The long-term memory is the partition of THIS game made from the tables given (in place, no snapshot made), or, with `long_term` (the path of a partition file written by `memory_write`), read from that file, which must be this game's; the memory of the game at hand is the slot's
        let measure = match measure { Some((term, d)) => { cs_tactics::measure::Term::parse(&term).map_err(bad)?; Some((term, d, "given by the caller of the wake".to_string())) } None => None };
        let game = carry_slot.as_ref().and_then(|k| self.games.borrow_mut().remove(k));
        let memory = match long_term {
            None => WakeMemory::of(self.key.clone(), Parts { record: Some(Record { worth, half }), chances, reasons, goal_table, measure, ..Default::default() }, game),
            Some(path) => {
                let part = Partition::load(std::path::Path::new(&path), &cs_decision::memory::decoders()).map_err(bad)?;
                if part.key != self.key { return Err(bad(format!("the partition at {path} is of another game ({} @ {}), not of {} @ {}", part.key.name, part.key.rules, self.key.name, self.key.rules))); }
                WakeMemory { long: Some(Reader::new(part)), game: game.unwrap_or_else(|| GameMemory::new(self.key.clone())), hold }
            }
        };
        let params = WakeParams { seconds, depth, pieces: pc, places: vec![], inputs: WakeInputs { memory, goal, budget, named_budget, frame, share_table: true, ledger_share, pre_share: pre_share_scale.map(|k| cs_decision::wake::pre_share_of(ledger_share.unwrap_or(cs_decision::LEDGER_SHARE), None) * k), chooser, carry_graph, chance_order, gain_mend, diag_first, read_on } };
        let st = &self.st;
        let d = py.allow_threads(|| decide(st, &g, &s, &history, &judge, &params));
        if let Some(k) = &carry_slot { self.games.borrow_mut().insert(k.clone(), d.game.clone()); }
        Ok(d.to_json().to_string())
    }

    /// PLAN 79.19 C: memory_write(root, worth, half, chances, reasons, goal_table, measure, sample) -> (path, id, parent): the tables the wake reads today, written as the partition of THIS game of the long-term memory under `root` (one file named by its content, in the folder of the game's key); the mined measure (term, d) as a statement that is a conjecture on `sample`. Nothing of the real store is written: the caller names `root`.
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    #[pyo3(signature = (root, worth, half, chances=None, reasons=None, goal_table=None, measure=None, sample=None, measures=None))]
    fn memory_write(&self, root: &str, worth: HashMap<String, i64>, half: HashMap<String, f64>, chances: Option<ChancesWire>, reasons: Option<ReasonsWire>, goal_table: Option<GoalTableWire>, measure: Option<(String, u32)>, sample: Option<String>, measures: Option<Vec<(String, String, u32, String)>>) -> PyResult<(String, String, Option<String>)> {
        use cs_decision::memory::{Parts, WakeMemory};
        use cs_decision::threat_verdict::Record;
        if let Some((term, _)) = &measure { cs_tactics::measure::Term::parse(term).map_err(bad)?; }
        let parts = Parts { record: Some(Record { worth, half }), chances: chances_of(chances), reasons: reasons_of(reasons), goal_table: goal_table_of(goal_table).map_err(bad)?, measure: measure.map(|(t, d)| (t, d, sample.unwrap_or_default())), more_measures: { let m = measures.unwrap_or_default(); for x in &m { cs_tactics::measure::Term::parse(&x.1).map_err(bad)?; } m } };
        let m = WakeMemory::of(self.key.clone(), parts, None);
        let r = m.long.as_ref().expect("the memory of a call has a reader");
        let path = r.write(std::path::Path::new(root)).map_err(bad)?;
        Ok((path.to_string_lossy().to_string(), r.id().to_string(), r.parent().map(|x| x.to_string())))
    }

    #[pyo3(signature = (text, worth, half, goal_kind=None, rows=None, every_sides=None))]
    #[allow(clippy::type_complexity)]
    fn wake_threats(&self, py: Python<'_>, text: &str, worth: HashMap<String, i64>, half: HashMap<String, f64>, goal_kind: Option<String>, rows: Option<Vec<(String, String, String, String, Vec<(String, String, String)>, Vec<(String, String, u8, bool)>, Option<f64>)>>, every_sides: Option<Vec<String>>) -> PyResult<PyObject> {
        use cs_decision::threat_verdict::Record;
        use cs_decision::threats::{concrete_threats, FrameRow, ThreatParams, ThreatRow};
        use cs_decision::wake::UnitRef;
        let (g, s) = self.game(text)?;
        let n = &self.st.names;
        let u = |t: &(String, String, String)| -> Result<UnitRef, String> { Ok(UnitRef { owner: n.role(&t.0)?, kind: n.kind(&t.1)?, at: n.place(&t.2)? }) };
        let mut p = ThreatParams::default();
        p.goal = match &goal_kind { Some(k) => Some(n.kind(k).map_err(bad)?), None => None };
        if let Some(rs) = rows {
            let mut v = vec![];
            for r in &rs {
                let unit = u(&(r.0.clone(), r.1.clone(), r.2.clone())).map_err(bad)?;
                let by = r.4.iter().map(&u).collect::<Result<Vec<_>, _>>().map_err(bad)?;
                let ix = r.5.iter().map(|a| Ok((n.place(&a.0)?, n.place(&a.1)?, a.2, a.3))).collect::<Result<Vec<_>, String>>().map_err(bad)?;
                v.push(FrameRow { unit, class: r.3.clone(), by, response_ix: ix, stake: r.6 });
            }
            p.rows = Some(v);
        }
        if let Some(sd) = every_sides { p.every_sides = Some(sd.iter().map(|r| n.role(r)).collect::<Result<Vec<_>, _>>().map_err(bad)?); }
        let rec = Record { worth, half };
        let l = concrete_threats(&self.st, &g, &s, &rec, &p);
        let uw = |x: &UnitRef| (n.roles[x.owner].clone(), n.kind_text(x.kind).to_string(), n.place_text(x.at).to_string());
        let row_d = |r: &ThreatRow| -> PyResult<PyObject> {
            let d = PyDict::new(py);
            d.set_item("owner", n.roles[r.owner].clone())?;
            d.set_item("unit", r.unit.as_ref().map(&uw))?;
            d.set_item("goal", r.goal)?; d.set_item("every", r.every)?; d.set_item("class", r.class.clone())?;
            d.set_item("verdict", format!("{:?}", r.verdict).to_uppercase())?;
            d.set_item("net", r.net)?; d.set_item("line", r.line.clone())?; d.set_item("two_moves", r.two_moves)?;
            d.set_item("lead", r.lead.map(|x| (n.place_text(x.0).to_string(), n.place_text(x.1).to_string())))?;
            d.set_item("ended", r.ended)?; d.set_item("why", r.why)?; d.set_item("deferred", r.deferred)?;
            d.set_item("resp_read", r.resp_read)?; d.set_item("resp_unread", r.resp_unread)?;
            d.set_item("outcome", r.outcome.as_ref().map(|o| (o.taker_lost.clone(), o.taker_won.clone(), o.left_lost.clone(), o.left_won.clone(), o.cancelled)))?;
            let resp: Vec<(Vec<String>, i64, &'static str, String)> = r.responses.iter().map(|x| (x.names.clone(), x.net, x.state.word(), format!("{:?}", x.source))).collect();
            d.set_item("responses", resp)?;
            d.into_py_any(py)
        };
        let out = PyDict::new(py);
        for (k, v) in [("mover", &l.mover), ("other", &l.other), ("exchange", &l.exchange), ("dropped", &l.dropped), ("goal", &l.goal)] {
            out.set_item(k, v.iter().map(&row_d).collect::<PyResult<Vec<_>>>()?)?;
        }
        out.set_item("dropped_counts", l.dropped_counts.clone())?;
        out.set_item("cost", l.cost.iter().map(|c| (c.name, c.seconds, c.queries, c.nodes)).collect::<Vec<_>>())?;
        out.into_py_any(py)
    }

    /// PLAN 79.07 (the wake's proposer, cs_decision::goals): wake_goals(text, worth, half, goal_kind, sides, rows, budget) -> {"goals": [goal], "two": [(role, [pair])], "shared": [(role, move, [(unit, fact)])], "defence": [(unit, [(move, [fact])])], "asked", "unasked", "questions", "line_reads"}.
    /// A unit is (role, kind, place) by the game's words; a goal is {"side", "target", "source", "unmet", "key", "stake", "two_moves", "lead", "n_unmet", "every", "goal_unit", "stoppers", "own_moves", "facts": [{"name", "status", "cause": None | {"type", "stoppers", "victim", "basis"}, "a", "place", "fixes": [(kind, [moves])]}],
    /// "fix_set": [{"unit", "kind", "removable", "resolves", "left", "left_stoppers", "moves"}], "steps", "escapes"}. `sides`: the roles whose goals are wanted (None: both); `budget`: the questions (None: the default). Exists for the unit check of the Rust proposer against its Python mirror; the wake calls the Rust function.
    #[pyo3(signature = (text, worth, half, goal_kind=None, sides=None, rows=None, budget=None))]
    #[allow(clippy::type_complexity)]
    fn wake_goals(&self, py: Python<'_>, text: &str, worth: HashMap<String, i64>, half: HashMap<String, f64>, goal_kind: Option<String>, sides: Option<Vec<String>>, rows: Option<Vec<(String, String, String, String, Vec<(String, String, String)>, Vec<(String, String, u8, bool)>, Option<f64>)>>, budget: Option<usize>) -> PyResult<PyObject> {
        use cs_decision::goals::{propose, FixKind, GoalParams, Status};
        use cs_decision::threat_verdict::Record;
        use cs_decision::threats::{FrameRow, ThreatParams};
        use cs_decision::wake::UnitRef;
        let (g, s) = self.game(text)?;
        let n = &self.st.names;
        let u = |t: &(String, String, String)| -> Result<UnitRef, String> { Ok(UnitRef { owner: n.role(&t.0)?, kind: n.kind(&t.1)?, at: n.place(&t.2)? }) };
        let mut p = ThreatParams::default();
        p.goal = match &goal_kind { Some(k) => Some(n.kind(k).map_err(bad)?), None => None };
        if let Some(rs) = rows {
            let mut v = vec![];
            for r in &rs {
                let unit = u(&(r.0.clone(), r.1.clone(), r.2.clone())).map_err(bad)?;
                let by = r.4.iter().map(&u).collect::<Result<Vec<_>, _>>().map_err(bad)?;
                let ix = r.5.iter().map(|a| Ok((n.place(&a.0)?, n.place(&a.1)?, a.2, a.3))).collect::<Result<Vec<_>, String>>().map_err(bad)?;
                v.push(FrameRow { unit, class: r.3.clone(), by, response_ix: ix, stake: r.6 });
            }
            p.rows = Some(v);
        }
        let mut gp = GoalParams::default();
        if let Some(b) = budget { gp.budget = b; }
        if let Some(sd) = sides { gp.sides = Some(sd.iter().map(|r| n.role(r)).collect::<Result<Vec<_>, _>>().map_err(bad)?); }
        let rec = Record { worth, half };
        let pr = propose(&self.st, &g, &s, &rec, &p, &gp);
        let uw = |x: &UnitRef| (n.roles[x.owner].clone(), n.kind_text(x.kind).to_string(), n.place_text(x.at).to_string());
        let kind_word = |k: FixKind| match k { FixKind::Remove => "remove", FixKind::Bring => "bring", FixKind::FirstMove => "first_move" };
        let mut goals_py = vec![];
        for gl in &pr.goals {
            let d = PyDict::new(py);
            d.set_item("side", n.roles[gl.side].clone())?; d.set_item("target", uw(&gl.target))?; d.set_item("source", gl.source.clone())?;
            d.set_item("unmet", gl.unmet.clone())?; d.set_item("key", gl.key.clone())?; d.set_item("stake", gl.stake)?; d.set_item("two_moves", gl.two_moves)?; d.set_item("lead", gl.lead.clone())?;
            d.set_item("depth", gl.depth)?; d.set_item("parent", gl.parent.as_ref().map(&uw))?; d.set_item("via", gl.via)?; d.set_item("n_unmet", gl.n_unmet)?; d.set_item("every", gl.every)?; d.set_item("goal_unit", gl.goal_unit)?; d.set_item("own_moves", gl.own_moves)?;
            d.set_item("stoppers", gl.stoppers.iter().map(&uw).collect::<Vec<_>>())?;
            let mut facts = vec![];
            for f in &gl.facts {
                let fd = PyDict::new(py);
                fd.set_item("name", f.name)?; fd.set_item("status", f.status.word())?;
                let _ = Status::Met;
                match &f.cause {
                    Some(c) => { let cd = PyDict::new(py); cd.set_item("type", c.kind)?; cd.set_item("stoppers", c.stoppers.iter().map(&uw).collect::<Vec<_>>())?; cd.set_item("victim", c.victim.as_ref().map(&uw))?; cd.set_item("basis", c.basis)?; fd.set_item("cause", cd)?; }
                    None => fd.set_item("cause", py.None())?,
                }
                fd.set_item("a", f.a.as_ref().map(&uw))?; fd.set_item("place", f.place.map(|p| n.place_text(p).to_string()))?; 
                fd.set_item("fixes", f.fixes.iter().map(|x| (kind_word(x.kind), x.moves.clone())).collect::<Vec<_>>())?;
                facts.push(fd);
            }
            d.set_item("facts", facts)?;
            let fs: Vec<PyObject> = gl.fix_set.iter().map(|e| {
                let ed = PyDict::new(py);
                let _ = ed.set_item("unit", e.unit.as_ref().map(&uw)); let _ = ed.set_item("kind", kind_word(e.kind)); let _ = ed.set_item("removable", e.removable); let _ = ed.set_item("resolves", e.resolves.clone());
                let _ = ed.set_item("left", e.left); let _ = ed.set_item("left_stoppers", e.left_stoppers); let _ = ed.set_item("moves", e.moves.clone());
                ed.into_any().unbind()
            }).collect();
            d.set_item("fix_set", fs)?;
            d.set_item("steps", gl.steps.iter().map(|x| (x.step, x.moves)).collect::<Vec<_>>())?;
            d.set_item("escapes", gl.escapes.iter().map(|e| (n.place_text(e.place).to_string(), e.covered, e.by.iter().map(&uw).collect::<Vec<_>>())).collect::<Vec<_>>())?;
            goals_py.push(d);
        }
        let out = PyDict::new(py);
        out.set_item("goals", goals_py)?;
        out.set_item("two", pr.two.iter().map(|(r, t)| (n.roles[*r].clone(), t.goals.iter().map(|x| (x.kind, x.stake, x.threats.iter().map(|y| (uw(&y.0), y.1, y.2)).collect::<Vec<_>>())).collect::<Vec<_>>())).collect::<Vec<_>>())?;
        out.set_item("shared", pr.shared.iter().map(|d| (n.roles[d.side].clone(), d.mv.clone(), d.under.iter().map(|x| (uw(&x.0), x.1)).collect::<Vec<_>>())).collect::<Vec<_>>())?;
        out.set_item("defence", pr.defence.iter().map(|(u, a)| (uw(u), a.iter().map(|x| (x.mv.clone(), x.facts.clone())).collect::<Vec<_>>())).collect::<Vec<_>>())?;
        out.set_item("asked", pr.asked.iter().map(&uw).collect::<Vec<_>>())?;
        out.set_item("unasked", pr.unasked.iter().map(&uw).collect::<Vec<_>>())?;
        out.set_item("questions", pr.questions)?;
        out.set_item("line_reads", pr.line_reads)?;
        out.into_py_any(py)
    }

    /// dump(text) -> text: the position's nodes, one per line, with their links.
    fn dump(&self, text: &str) -> PyResult<String> {
        let (g, s) = self.game(text)?;
        Ok(self.st.position(&g, &s).dump(&self.st))
    }

    /// bearings(text) -> [(owner, kind, place, [places borne on])]: what `Placed::bearing_from` reports for each unit, straight from the trait (for the check that the graph holds each once).
    fn bearings(&self, text: &str) -> PyResult<Vec<(String, String, String, Vec<String>)>> {
        let (g, s) = self.game(text)?;
        let n = &self.st.names;
        Ok(g.cells(&s).into_iter().map(|(o, k, at)| (n.roles[o].clone(), n.kinds[k as usize].clone(), n.places[at as usize].clone(),
            g.bearing_from(&s, o, k, at).into_iter().map(|p| n.places[p as usize].clone()).collect())).collect())
    }

    /// statement_names() -> ([(name, text)] sided, [(name, text)] of the board).
    fn statement_names(&self) -> (Vec<NodeWire>, Vec<NodeWire>) {
        (base_statements::names(&self.st), base_statements::BOARD.iter().map(|(a, b)| (a.to_string(), b.to_string())).collect())
    }

    /// statements(text, worth, goal_kind) -> (us, them, board): the statements over the graph of the position, the side to move first.
    fn statements(&self, text: &str, worth: HashMap<String, f64>, goal_kind: Option<String>) -> PyResult<(Vec<f64>, Vec<f64>, Vec<f64>)> {
        let (g, s) = self.game(text)?;
        let n = &self.st.names;
        let w: Vec<f64> = n.kinds.iter().map(|k| worth.get(k).copied().unwrap_or(0.0)).collect();
        let gk = match goal_kind { Some(k) => Some(n.kind(&k).map_err(bad)?), None => None };
        let p = self.st.position(&g, &s);
        if p.mover.is_none() { return Err(bad("the position is over".into())); }
        let ([a, b], board) = base_statements::statements(&self.st, &p, &w, gk);
        Ok((a, b, board))
    }

    /// open_lengths(text) -> [(line id, the longest run of places with no unit)].
    fn open_lengths(&self, text: &str) -> PyResult<Vec<(String, usize)>> {
        let (g, s) = self.game(text)?;
        let p = self.st.position(&g, &s);
        Ok((0..self.st.lines.len()).map(|l| (self.st.nodes[self.st.line_node[l] as usize].id.clone(), p.open_length(&self.st, l))).collect())
    }

    /// measure(texts) -> {positions, seconds (the graph build alone, the states parsed beforehand), nodes: {type: count}, links: {type: count}, static_nodes, static_links} over the
    /// positions (summed).
    fn measure(&self, py: Python<'_>, texts: Vec<String>) -> PyResult<PyObject> {
        let mut states = vec![];
        for f in &texts { states.push(self.game(f)?); }
        let mut parts: Vec<PositionPart> = Vec::with_capacity(states.len());
        let t0 = std::time::Instant::now();
        for (g, s) in &states { parts.push(self.st.position(g, s)); }
        let seconds = t0.elapsed().as_secs_f64();
        let mut nodes: HashMap<&str, usize> = HashMap::new();
        let mut links: HashMap<&str, usize> = HashMap::new();
        for p in &parts {
            for n in &p.nodes { *nodes.entry(n.kind.word()).or_default() += 1; }
            for l in &p.links { *links.entry(l.kind.word()).or_default() += 1; }
        }
        let d = PyDict::new(py);
        d.set_item("positions", parts.len())?;
        d.set_item("seconds", seconds)?;
        d.set_item("nodes", nodes)?;
        d.set_item("links", links)?;
        let (mut sn, mut sl): (HashMap<&str, usize>, HashMap<&str, usize>) = (HashMap::new(), HashMap::new());
        for n in &self.st.nodes { *sn.entry(n.kind.word()).or_default() += 1; }
        for l in &self.st.links { *sl.entry(l.kind.word()).or_default() += 1; }
        d.set_item("static_nodes", sn)?;
        d.set_item("static_links", sl)?;
        d.set_item("node_types", NodeType::ALL.iter().map(|t| t.word()).collect::<Vec<_>>())?;
        d.set_item("link_types", LinkType::ALL.iter().map(|t| t.word()).collect::<Vec<_>>())?;
        d.into_py_any(py)
    }

    /// pattern_freq(games, rows, size, levels, allowed, goal_kind, threads, keep) -> (found per level, [(hash, level, support, total, bytes, parts, free_parent)] the `keep` most supported per level):
    /// the keys of the patterns of `size` relation nodes over the wanted plies (`rows[game]`) of the games. `allowed`, for a size above 1, the kept hashes of the size below.
    #[allow(clippy::too_many_arguments)]
    fn pattern_freq(&self, py: Python<'_>, games: Vec<GameWire>, rows: Vec<Vec<usize>>, size: usize, levels: Vec<u8>, allowed: Option<Vec<u64>>, goal_kind: Option<String>, threads: usize, keep: usize) -> PyResult<PyObject> {
        let goal = self.goal_of(&goal_kind)?;
        let st = self.st.clone();
        let variant = self.variant.clone();
        let allowed: Option<FxHashSet<u64>> = allowed.map(|v| v.into_iter().collect());
        let mut base = vec![0u32; games.len() + 1];
        for i in 0..games.len() { base[i + 1] = base[i] + rows[i].len() as u32; }
        let res = py.allow_threads(|| run_pool(games.len(), threads, |i| {
            let mut m: FxHashMap<u64, patterns::Seen> = FxHashMap::default();
            walk_rows(&variant, &st, goal, &games[i], &rows[i], |ri, b, _| patterns::freq_board(b, size, &levels, allowed.as_ref(), base[i] + ri as u32, &mut m))?;
            Ok(m)
        })).map_err(bad)?;
        let mut all: FxHashMap<u64, patterns::Seen> = FxHashMap::default();
        for m in res {
            for (h, e) in m {
                match all.get_mut(&h) {
                    Some(x) => { x.total += e.total; x.support += e.support; }
                    None => { all.insert(h, e); }
                }
            }
        }
        let mut per: Vec<Vec<(u64, u32, u64)>> = vec![vec![]; patterns::LEVELS as usize];
        for (h, e) in &all { per[e.bytes[0] as usize].push((*h, e.support, e.total)); }
        let mut found = vec![];
        let mut kept = vec![];
        for (lv, v) in per.iter_mut().enumerate() {
            found.push(v.len());
            v.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
            for (h, sup, tot) in v.iter().take(keep) {
                let e = &all[h];
                kept.push((*h, lv, *sup, *tot, pyo3::types::PyBytes::new(py, &e.bytes).unbind(), e.parts.clone(), e.free_parent));
            }
        }
        (found, kept).into_py_any(py)
    }

    /// pattern_count(games, rows, keys, max_size, levels, goal_kind, threads) -> bytes: for each wanted row a record (row u32, column u32, count for the side to move u16, count for the other u16)
    /// for every wanted key found on it, the rows numbered in the order of `rows` across the games.
    #[allow(clippy::too_many_arguments)]
    fn pattern_count(&self, py: Python<'_>, games: Vec<GameWire>, rows: Vec<Vec<usize>>, keys: Vec<u64>, max_size: usize, levels: Vec<u8>, goal_kind: Option<String>, threads: usize) -> PyResult<PyObject> {
        let goal = self.goal_of(&goal_kind)?;
        let st = self.st.clone();
        let variant = self.variant.clone();
        let want: FxHashMap<u64, usize> = keys.iter().enumerate().map(|(i, k)| (*k, i)).collect();
        let mut base = vec![0u32; games.len() + 1];
        for i in 0..games.len() { base[i + 1] = base[i] + rows[i].len() as u32; }
        let res = py.allow_threads(|| run_pool(games.len(), threads, |i| {
            let mut out: Vec<u8> = vec![];
            walk_rows(&variant, &st, goal, &games[i], &rows[i], |ri, b, _| {
                let mut m: FxHashMap<usize, (u32, u32)> = FxHashMap::default();
                patterns::count_board(b, max_size, &levels, &want, &mut m);
                let mut v: Vec<_> = m.into_iter().collect();
                v.sort();
                for (c, (us, them)) in v {
                    out.extend_from_slice(&(base[i] + ri as u32).to_le_bytes());
                    out.extend_from_slice(&(c as u32).to_le_bytes());
                    out.extend_from_slice(&(us.min(65535) as u16).to_le_bytes());
                    out.extend_from_slice(&(them.min(65535) as u16).to_le_bytes());
                }
            })?;
            Ok(out)
        })).map_err(bad)?;
        let bytes: Vec<u8> = res.into_iter().flatten().collect();
        pyo3::types::PyBytes::new(py, &bytes).into_py_any(py)
    }

    /// unit_tables(games, rows, horizon, threads) -> (kind names, bytes): per wanted row 1 + 4 * kinds i16: [valid, then the units of the side to move by kind, the other side's, at the row; the same
    /// `horizon` plies later (the side to move of the row stays "us")]; valid 0 (and zeros) where the position is over or the horizon runs past the game.
    fn unit_tables(&self, py: Python<'_>, games: Vec<GameWire>, rows: Vec<Vec<usize>>, horizon: usize, threads: usize) -> PyResult<PyObject> {
        use cs_game::Game as _;
        let variant = self.variant.clone();
        let names = self.st.names.clone();
        let nk = names.kinds.len();
        let res = py.allow_threads(|| run_pool(games.len(), threads, |i| {
            let g = match &games[i].0 { Some(text) => KernelGame::from_fen(&variant, text)?, None => KernelGame::new(&variant, 518)? };
            let mut s = g.initial();
            let n = games[i].1.len();
            // per ply: the mover and the units by owner and kind
            let mut at: Vec<(Option<usize>, Vec<Vec<i16>>)> = vec![];
            for ply in 0..=n {
                let mut c = vec![vec![0i16; nk]; names.roles.len()];
                for (o, k, _) in g.cells(&s) { c[o][k as usize] += 1; }
                at.push((g.mover(&s), c));
                if ply < n {
                    let mv = g.find_move(&s, &games[i].1[ply]).ok_or_else(|| format!("ply {ply}: {} is not a legal move", games[i].1[ply]))?;
                    s = g.apply(&s, &mv);
                }
            }
            let mut out: Vec<i16> = vec![];
            for &r in &rows[i] {
                match at.get(r).and_then(|x| x.0) {
                    Some(m) if r + horizon <= n => {
                        out.push(1);
                        for ply in [r, r + horizon] {
                            out.extend(at[ply].1[m].iter());
                            for k in 0..nk { out.push((0..names.roles.len()).filter(|o| *o != m).map(|o| at[ply].1[o][k]).sum()); }
                        }
                    }
                    _ => { out.push(0); out.extend(std::iter::repeat(0).take(4 * nk)); }
                }
            }
            Ok(out)
        })).map_err(bad)?;
        let bytes: Vec<u8> = res.into_iter().flatten().flat_map(|v| v.to_le_bytes()).collect();
        (names.kinds.clone(), pyo3::types::PyBytes::new(py, &bytes).unbind()).into_py_any(py)
    }

    /// position_table(games, threads) -> (kind names, bytes): PLAN 78.58 A. For every ply 0..=n of every game, in order, 2 + 2 * kinds i16: [valid, quiet, the units of the side to move by kind, the other side's].
    /// valid 0 (and zeros) where there is no side to move. QUIET 1 where nothing stands to be lost by inclusion for either side: no unit is borne on by the other side and not by its own side
    /// (the question of 78.57 Q0 asked of both sides at once). A cheap quiet for positions that have no forcing read.
    fn position_table(&self, py: Python<'_>, games: Vec<GameWire>, threads: usize) -> PyResult<PyObject> {
        let variant = self.variant.clone();
        let names = self.st.names.clone();
        let nk = names.kinds.len();
        let res = py.allow_threads(|| run_pool(games.len(), threads, |i| {
            let g = match &games[i].0 { Some(text) => KernelGame::from_fen(&variant, text)?, None => KernelGame::new(&variant, 518)? };
            let mut s = g.initial();
            let n = games[i].1.len();
            let mut out: Vec<i16> = Vec::with_capacity((n + 1) * (2 + 2 * nk));
            for ply in 0..=n {
                match g.mover(&s) {
                    Some(m) => {
                        let cells = g.cells(&s);
                        let mut c = vec![vec![0i16; nk]; names.roles.len()];
                        let mut quiet = true;
                        for &(o, k, at) in &cells {
                            c[o][k as usize] += 1;
                            let other = if o == 0 { 1 } else { 0 };
                            if g.strikes_place(&s, other, at) && !g.strikes_place(&s, o, at) { quiet = false; }
                        }
                        out.push(1);
                        out.push(quiet as i16);
                        out.extend(c[m].iter());
                        for k in 0..nk { out.push((0..names.roles.len()).filter(|o| *o != m).map(|o| c[o][k]).sum()); }
                    }
                    None => { out.push(0); out.extend(std::iter::repeat(0).take(1 + 2 * nk)); }
                }
                if ply < n {
                    let mv = g.find_move(&s, &games[i].1[ply]).ok_or_else(|| format!("ply {ply}: {} is not a legal move", games[i].1[ply]))?;
                    s = g.apply(&s, &mv);
                }
            }
            Ok(out)
        })).map_err(bad)?;
        let bytes: Vec<u8> = res.into_iter().flatten().flat_map(|v| v.to_le_bytes()).collect();
        (names.kinds.clone(), pyo3::types::PyBytes::new(py, &bytes).unbind()).into_py_any(py)
    }

    /// first_blocker_check(text) -> (the shut relations of the side to move with exactly one unit between, as `why_not` gives them; the same set found by the first-blocker form through `without`; whether the two sets are equal).
    fn first_blocker_check(&self, text: &str) -> PyResult<(usize, usize, bool)> {
        let (g, s) = self.game(text)?;
        let mut p = self.st.position(&g, &s);
        let ns = self.st.nodes.len() as u32;
        let Some(m) = p.mover else { return Ok((0, 0, true)) };
        let key = |p: &cs_tactics::sheet::base_graph::PositionPart, from: usize| -> std::collections::BTreeSet<(usize, u16)> {
            p.shuts[from..].iter().filter(|r| r.between.len() == 1 && p.units[r.unit].owner == m).map(|r| (r.unit, r.place)).collect()
        };
        let a = key(&p, 0);
        let before = p.shuts.len();
        for ui in 0..p.units.len() { if p.units[ui].owner == m { self.st.first_blocker_shuts(&g, &s, ui, &mut p, ns); } }
        let b = key(&p, before);
        Ok((a.len(), b.len(), a == b))
    }

    /// learn_profile(games) -> [(kind, direction, forward)]: learn which kinds continue along which line families from the bear nodes of every position of the given games (78.54 A);
    /// from then on every position part also holds its `through` relation nodes. Sequential, a sample.
    fn learn_profile(&mut self, games: Vec<GameWire>) -> PyResult<Vec<(String, String, bool)>> {
        for game in &games {
            let g = match &game.0 { Some(text) => KernelGame::from_fen(&self.variant, text).map_err(bad)?, None => KernelGame::new(&self.variant, 518).map_err(bad)? };
            let mut s = g.initial();
            for ply in 0..=game.1.len() {
                let p = self.st.position(&g, &s);
                self.st.learn(&p);
                if ply < game.1.len() {
                    let mv = g.find_move(&s, &game.1[ply]).ok_or_else(|| bad(format!("ply {ply}: {} is not a legal move", game.1[ply])))?;
                    s = g.apply(&s, &mv);
                }
            }
        }
        Ok(self.profile())
    }

    /// profile() -> [(kind word, line direction word, forward)]: what has been learned to continue.
    fn profile(&self) -> Vec<(String, String, bool)> {
        let (nd, n) = (self.st.directions.len(), &self.st.names);
        let mut out = vec![];
        for (i, c) in self.st.cont.iter().enumerate() { if *c { out.push((n.kinds[i / 2 / nd].clone(), self.st.directions[i / 2 % nd].clone(), i % 2 == 1)); } }
        out
    }

    /// through_check(text) -> (through relations of the side to move, shut relations of the side to move, in both, through only, shut only): THE AGREEMENT of 78.54 A with the rules' own answer for the side
    /// to move. A shut node (`why_not` says the line is shut) with exactly one unit between, whose end place holds a unit of the other side, is compared with a through node of the same unit and end place
    /// (the rules say nothing of an end place held by the unit's own side, so those are left out of both).
    fn through_check(&self, text: &str) -> PyResult<(usize, usize, usize, usize, usize)> {
        let (g, s) = self.game(text)?;
        let p = self.st.position(&g, &s);
        let Some(m) = p.mover else { return Ok((0, 0, 0, 0, 0)) };
        let other = |r: &cs_tactics::sheet::base_graph::RelRec| p.units[r.unit].owner == m && match p.occupant[r.place as usize] { 0 => false, i => p.units[i as usize - 1].owner != m };
        let t: std::collections::BTreeSet<(usize, u16)> = p.throughs.iter().filter(|r| other(r)).map(|r| (r.unit, r.place)).collect();
        let sh: std::collections::BTreeSet<(usize, u16)> = p.shuts.iter().filter(|r| r.between.len() == 1 && other(r)).map(|r| (r.unit, r.place)).collect();
        Ok((t.len(), sh.len(), t.intersection(&sh).count(), t.difference(&sh).count(), sh.difference(&t).count()))
    }

    /// unit_fates(game, horizons) -> (kinds, [per ply: (mover, [(owner, kind, place, [fate class at each horizon])])]): the tracker on one game.
    #[allow(clippy::type_complexity)]
    fn unit_fates(&self, game: GameWire, horizons: Vec<usize>) -> PyResult<(Vec<String>, Vec<(Option<String>, Vec<(String, String, String, Vec<u8>)>)>)> {
        let (plies, movers, _) = walk_game(&self.variant, &self.st, None, &game, &[]).map_err(bad)?;
        let tr = fates::track(&plies);
        let n = &self.st.names;
        let out = (0..plies.len()).map(|t| (movers[t].map(|m| n.roles[m].clone()), plies[t].iter().enumerate().map(|(i, c)| (n.roles[c.0].clone(), n.kinds[c.1 as usize].clone(), n.places[c.2 as usize].clone(),
            horizons.iter().map(|h| fates::fate_code(&tr, &plies, t, i, *h)).collect())).collect())).collect();
        Ok((n.kinds.clone(), out))
    }

    /// pattern_mine(games, rows, size, levels, allowed, goal_kind, threads, min_support) -> (found per level, [(hash, level, support, total, bytes, parts, maps)] those with at least `min_support`
    /// rows): `pattern_freq` by a minimum support, a candidate when every connected part is in `allowed` (78.54 D); `maps` says where each variable of the pattern stands in each part.
    #[allow(clippy::too_many_arguments)]
    fn pattern_mine(&self, py: Python<'_>, games: Vec<GameWire>, rows: Vec<Vec<usize>>, size: usize, levels: Vec<u8>, allowed: Option<Vec<u64>>, goal_kind: Option<String>, threads: usize, min_support: u32) -> PyResult<PyObject> {
        let goal = self.goal_of(&goal_kind)?;
        let st = self.st.clone();
        let variant = self.variant.clone();
        let allowed: Option<FxHashSet<u64>> = allowed.map(|v| v.into_iter().collect());
        let mut base = vec![0u32; games.len() + 1];
        for i in 0..games.len() { base[i + 1] = base[i] + rows[i].len() as u32; }
        let res = py.allow_threads(|| run_pool(games.len(), threads, |i| {
            let mut m: FxHashMap<u64, patterns::Mined> = FxHashMap::default();
            walk_rows(&variant, &st, goal, &games[i], &rows[i], |ri, b, _| patterns::mine_board(b, size, &levels, allowed.as_ref(), base[i] + ri as u32, &mut m))?;
            Ok(m)
        })).map_err(bad)?;
        let mut all: FxHashMap<u64, patterns::Mined> = FxHashMap::default();
        for m in res {
            for (h, e) in m {
                match all.get_mut(&h) { Some(x) => { x.total += e.total; x.support += e.support; } None => { all.insert(h, e); } }
            }
        }
        let mut found = vec![0usize; patterns::LEVELS as usize];
        let mut kept = vec![];
        let mut keys: Vec<&u64> = all.keys().collect();
        keys.sort();
        for h in keys {
            let e = &all[h];
            let lv = e.bytes[0] as usize;
            found[lv] += 1;
            if e.support >= min_support { kept.push((*h, lv, e.support, e.total, pyo3::types::PyBytes::new(py, &e.bytes).unbind(), e.parts.clone(), e.maps.clone())); }
        }
        (found, kept).into_py_any(py)
    }

    /// fate_base(games, rows, group, groups, horizons, threads) -> (kinds, counts): per group of games, horizon, kind and side-to-move state the number of units of each fate class and the number with a fate
    /// (u64, laid out [group][horizon][kind][state][classes then total]). `group[i]` the group of game i.
    #[allow(clippy::too_many_arguments)]
    fn fate_base(&self, py: Python<'_>, games: Vec<GameWire>, rows: Vec<Vec<usize>>, group: Vec<usize>, groups: usize, horizons: Vec<usize>, threads: usize) -> PyResult<PyObject> {
        let st = self.st.clone();
        let variant = self.variant.clone();
        let spec = fates::BaseSpec::new(st.names.kinds.len(), horizons.len());
        let size = groups * horizons.len() * spec.nk * 2 * (spec.ncls + 1);
        let res = py.allow_threads(|| run_acc(games.len(), threads, || vec![0u64; size], |i, acc| {
            let (plies, movers, _) = walk_game(&variant, &st, None, &games[i], &[])?;
            let tr = fates::track(&plies);
            fates::add_base(&spec, group[i], &horizons, &tr, &plies, &movers, &rows[i], acc);
            Ok(())
        })).map_err(bad)?;
        let mut tot = vec![0u64; size];
        for v in res { for (a, b) in tot.iter_mut().zip(v) { *a += b; } }
        (self.st.names.kinds.clone(), tot).into_py_any(py)
    }

    /// fate_cells(games, rows, keys, rates, horizons, max_size, goal_kind, threads) -> (classes, [f64]) rows of 7 + 2 classes numbers: [pattern index, variable, horizon index, n, games, sum of the games'
    /// residuals, sum of their squares, counts by class, expected by class]; `rates` the base rates [horizon][kind][state][class] (78.54 C).
    #[allow(clippy::too_many_arguments)]
    fn fate_cells(&self, py: Python<'_>, games: Vec<GameWire>, rows: Vec<Vec<usize>>, keys: Vec<u64>, rates: Vec<f64>, horizons: Vec<usize>, max_size: usize, goal_kind: Option<String>, threads: usize) -> PyResult<PyObject> {
        let goal = self.goal_of(&goal_kind)?;
        let st = self.st.clone();
        let variant = self.variant.clone();
        let spec = fates::BaseSpec::new(st.names.kinds.len(), horizons.len());
        let want: FxHashMap<u64, u32> = keys.iter().enumerate().map(|(i, k)| (*k, i as u32)).collect();
        let levels: Vec<u8> = (0..patterns::LEVELS).collect();
        let res = py.allow_threads(|| run_acc(games.len(), threads, fates::Cells::default, |i, acc| {
            let (plies, _movers, boards) = walk_game(&variant, &st, goal, &games[i], &rows[i])?;
            let tr = fates::track(&plies);
            fates::add_cells(&spec, &horizons, &rates, &want, max_size, &levels, &tr, &plies, &boards, acc);
            Ok(())
        })).map_err(bad)?;
        let mut all = fates::Cells::default();
        for c in res { fates::merge_cells(&mut all, c); }
        let nh = horizons.len();
        let mut cells: Vec<(&u32, &Vec<f64>)> = all.iter().collect();
        cells.sort_by_key(|x| *x.0);
        let mut out: Vec<f64> = Vec::with_capacity(cells.len() * (7 + 2 * spec.ncls));
        for (c, v) in cells {
            let c = *c as usize;
            let (hi, rest) = (c % nh, c / nh);
            out.extend([(rest / fates::MAXV) as f64, (rest % fates::MAXV) as f64, hi as f64]);
            out.extend(v.iter());
        }
        (spec.ncls, out).into_py_any(py)
    }

    /// track_counts(games, threads) -> [plies walked, unit steps, alive, taken, other, units paired among several of one owner and kind]: what the tracker made of every step of every unit of the games.
    fn track_counts(&self, py: Python<'_>, games: Vec<GameWire>, threads: usize) -> PyResult<Vec<u64>> {
        let st = self.st.clone();
        let variant = self.variant.clone();
        let res = py.allow_threads(|| run_acc(games.len(), threads, || vec![0u64; 6], |i, acc| {
            let (plies, _m, _b) = walk_game(&variant, &st, None, &games[i], &[])?;
            let tr = fates::track(&plies);
            acc[0] += plies.len() as u64;
            acc[5] += tr.ambiguous as u64;
            for s in tr.steps.iter().flatten() {
                acc[1] += 1;
                match s { fates::Step::Alive(_) => acc[2] += 1, fates::Step::Taken(_) => acc[3] += 1, fates::Step::Other => acc[4] += 1 }
            }
            Ok(())
        })).map_err(bad)?;
        let mut tot = vec![0u64; 6];
        for v in res { for (a, b) in tot.iter_mut().zip(v) { *a += b; } }
        Ok(tot)
    }

    /// unit_histories(game, horizons) -> (kinds, [per ply: [(owner, kind, place, [(class, [kinds it took]) at each horizon])]]): the tracker's history of 78.55 stage 0 on one game (class numbers: see
    /// `fates::History`).
    #[allow(clippy::type_complexity)]
    fn unit_histories(&self, game: GameWire, horizons: Vec<usize>) -> PyResult<(Vec<String>, Vec<Vec<(String, String, String, Vec<(u8, Vec<String>)>)>>)> {
        let (plies, _m, _) = walk_game(&self.variant, &self.st, None, &game, &[]).map_err(bad)?;
        let tr = fates::track(&plies);
        let n = &self.st.names;
        let nk = n.kinds.len();
        let out = (0..plies.len()).map(|t| plies[t].iter().enumerate().map(|(i, c)| (n.roles[c.0].clone(), n.kinds[c.1 as usize].clone(), n.places[c.2 as usize].clone(),
            horizons.iter().map(|h| { let hist = fates::fate_history(&tr, &plies, t, i, *h); (fates::history_class(&hist, nk), hist.took.iter().map(|k| n.kinds[*k as usize].clone()).collect()) }).collect())).collect()).collect();
        Ok((n.kinds.clone(), out))
    }

    /// frame_seeds(text, keys, goal_kind, max_size) -> [(concept number, unit ids, place ids)]: the units and empty places each match of the concepts (by key) binds on a position.
    fn frame_seeds(&self, text: &str, keys: Vec<u64>, goal_kind: Option<String>, max_size: usize) -> PyResult<Vec<(u32, Vec<String>, Vec<String>)>> {
        let goal = self.goal_of(&goal_kind)?;
        let (g, s) = self.game(text)?;
        let part = self.st.position(&g, &s);
        let b = Board::new(&self.st, &part, goal);
        let want: FxHashMap<u64, u32> = keys.iter().enumerate().map(|(i, k)| (*k, i as u32)).collect();
        let levels: Vec<u8> = (0..patterns::LEVELS).collect();
        Ok(cs_tactics::sheet::frames::seeds_of_board(&b, &part, max_size, &levels, &want).into_iter()
            .map(|(c, sd)| (c, sd.units.iter().map(|u| cs_tactics::sheet::frames::unit_id(&self.st, u)).collect(), sd.places.iter().map(|p| format!("place:{}", self.st.names.places[*p as usize])).collect())).collect())
    }

    /// frame_read(text, units, places, conditional, full_tempo, table_passes, fuse) -> the frame of the seed on the position, its claims with their status, the tempo table, the outcomes table
    /// (when `table_passes` is given), the layer 3 nodes and links, and what the reading cost (78.55 A to E).
    #[allow(clippy::too_many_arguments)]
    fn frame_read(&self, py: Python<'_>, text: &str, units: Vec<String>, places: Vec<String>, conditional: bool, full_tempo: bool, table_passes: Option<u8>, fuse: u64) -> PyResult<PyObject> {
        let seed = crate::frames_py::parse_seed(&self.st, &units, &places).map_err(bad)?;
        crate::frames_py::frame_read(py, &self.variant, &self.st, text, &seed, conditional, full_tempo, table_passes, fuse)
    }

    /// frame_join(text, seeds) -> {frames, refused}: the frames of the seeds [(unit ids, place ids)] on a position, those in contact joined.
    fn frame_join(&self, py: Python<'_>, text: &str, seeds: Vec<(Vec<String>, Vec<String>)>) -> PyResult<PyObject> {
        let sds: Vec<_> = seeds.iter().map(|(u, p)| crate::frames_py::parse_seed(&self.st, u, p)).collect::<Result<_, _>>().map_err(bad)?;
        crate::frames_py::frame_join(py, &self.variant, &self.st, text, &sds)
    }

    /// frame_compose(text, seed_a, seed_b, fuse) -> the composition of two frames and the joint read of the union frame, for both sides (78.55 F: checked, not proved).
    fn frame_compose(&self, py: Python<'_>, text: &str, a: (Vec<String>, Vec<String>), b: (Vec<String>, Vec<String>), fuse: u64) -> PyResult<PyObject> {
        let sa = crate::frames_py::parse_seed(&self.st, &a.0, &a.1).map_err(bad)?;
        let sb = crate::frames_py::parse_seed(&self.st, &b.0, &b.1).map_err(bad)?;
        crate::frames_py::frame_compose(py, &self.variant, &self.st, text, &sa, &sb, fuse)
    }

    /// set_kind_values([(kind, worth)]): PLAN 78.58 the worth of each kind in whole numbers, read from the estimator's record (`kind_values`); every kind of the game must be named. With it `question_read` and
    /// `questions_measure` prune a taking whose exchange on its place loses by the estimate and read one that wins or is level as one question. An empty list clears it.
    fn set_kind_values(&mut self, vals: Vec<(String, i64)>) -> PyResult<()> {
        if vals.is_empty() { self.kv = None; return Ok(()); }
        let mut kv = vec![i64::MIN; self.st.names.kinds.len()];
        for (k, v) in vals { let ix = self.st.names.kind(&k).map_err(bad)? as usize; kv[ix] = v; }
        if let Some(i) = kv.iter().position(|&v| v == i64::MIN) { return Err(bad(format!("no worth for kind {}", self.st.names.kinds[i]))); }
        self.kv = Some(kv);
        Ok(())
    }

    /// set_kind_bands([(kind, low, high)]): PLAN 78.60 D the band of the worth of each kind (whole numbers; from the estimator record's order of kinds); a kind not named has no width (low = high = its worth). With the
    /// rule `band` on, a taking is pruned by the estimate only if it loses at every corner of the bands. An empty list clears it.
    fn set_kind_bands(&mut self, vals: Vec<(String, i64, i64)>) -> PyResult<()> {
        if vals.is_empty() { self.bands = None; return Ok(()); }
        let kv = self.kv.clone().ok_or_else(|| bad("set_kind_values first".to_string()))?;
        let mut b: Vec<(i64, i64)> = kv.iter().map(|&v| (v, v)).collect();
        for (k, lo, hi) in vals { let ix = self.st.names.kind(&k).map_err(bad)? as usize; if lo > hi { return Err(bad(format!("band of {k}: low above high"))); } b[ix] = (lo, hi); }
        self.bands = Some(b);
        Ok(())
    }

    /// set_rules(settle, progress, cause, band, root_cap): PLAN 78.60 the rules of a line's end for `question_read` and `questions_measure` (A settled, B progress, C named cause, D band) and the cap of members at the
    /// root of a frame (0 or 10: as before; below 10: the seed units and what bears on them). All false and 0: the 78.58 read (and with no kind values, the 78.57 read).
    fn set_rules(&mut self, settle: bool, progress: bool, cause: bool, band: bool, root_cap: usize) {
        self.rules = cs_tactics::sheet::questions::Rules { settle, progress, cause, band, root_cap };
    }

    /// exchange_taking(text, taker, place) -> None or (estimate, the units that take in order, (bearers of the taker's side, bearers of the other)): the estimated exchange on `place` when the unit `taker`
    /// takes what stands there (78.58 C), with the worth set by `set_kind_values`.
    fn exchange_taking(&self, text: &str, taker: &str, place: &str) -> PyResult<Option<(i64, Vec<String>, (usize, usize))>> {
        let kv = self.kv.clone().ok_or_else(|| bad("set_kind_values first".to_string()))?;
        let (g, s) = self.game(text)?;
        let part = self.st.position(&g, &s);
        let cell = crate::frames_py::parse_unit(&self.st, taker).map_err(bad)?;
        let ti = part.units.iter().position(|u| (u.owner, u.kind, u.at) == cell).ok_or_else(|| bad(format!("no unit {taker} in the position")))?;
        let to = self.st.names.place(place.strip_prefix("place:").unwrap_or(place)).map_err(bad)?;
        Ok(cs_tactics::sheet::exchange::estimate_taking(&part, to, ti, &kv).map(|e| (e.value, e.line.iter().map(|&i| cs_tactics::sheet::frames::unit_id(&self.st, &(part.units[i].owner, part.units[i].kind, part.units[i].at))).collect(), e.bearers)))
    }

    /// exchange_plain(on_s, mine, theirs) -> (cheapest, value): the two reads of lean/ideas/Ideas/Exchange.lean on plain lists (78.58 C): the static exchange and the minimax of the capture game.
    #[staticmethod]
    fn exchange_plain(on_s: i64, mine: Vec<i64>, theirs: Vec<i64>) -> (i64, i64) {
        (cs_tactics::sheet::exchange::cheapest(on_s, &mine, &theirs), cs_tactics::sheet::exchange::value(on_s, &mine, &theirs))
    }

    /// exchange_rays(white, on_s, mine, theirs) -> (cheapestXF, the brute read of every order): the Lean `cheapestXF` on rays (value, [(owner, value)]) and, beside it, the brute read.
    #[staticmethod]
    fn exchange_rays(white: bool, on_s: i64, mine: Vec<(i64, Vec<(bool, i64)>)>, theirs: Vec<(i64, Vec<(bool, i64)>)>) -> (i64, i64) {
        use cs_tactics::sheet::exchange as ex;
        let mk = |v: Vec<(i64, Vec<(bool, i64)>)>| -> Vec<ex::Ray> { v.into_iter().map(|(value, b)| ex::Ray { value, id: 0, behind: b.into_iter().map(|(o, x)| (o, x, 0u16)).collect() }).collect() };
        let (m, t) = (mk(mine), mk(theirs));
        let fuel = 8;
        (ex::cheapest_xf(fuel, white, on_s, &m, &t), ex::brute_xf(fuel, white, on_s, &m, &t))
    }

    /// question_read(text, units, places, conditional, fuse) -> the frame of the seed (bound to its units), its claims with their status and marks, what the questions cost, the pruned root moves, the layer 3 nodes (78.57).
    fn question_read(&self, py: Python<'_>, text: &str, units: Vec<String>, places: Vec<String>, conditional: bool, fuse: u64) -> PyResult<PyObject> {
        let seed = crate::frames_py::parse_seed(&self.st, &units, &places).map_err(bad)?;
        crate::questions_py::question_read(py, &self.variant, &self.st, text, &seed, conditional, fuse, self.kv.clone(), self.rules, self.bands.clone())
    }

    /// question_trap(text, unit, fuse) -> the trap question of one unit on quiet routes (78.57).
    fn question_trap(&self, py: Python<'_>, text: &str, unit: &str, fuse: u64) -> PyResult<PyObject> {
        crate::questions_py::question_trap(py, &self.variant, &self.st, text, unit, fuse)
    }

    /// question_compose(text, seed_a, seed_b, fuse) -> the tempo-charged composition of two frames and the joint read, for the side to move (78.57 D: checked, not proved).
    fn question_compose(&self, py: Python<'_>, text: &str, a: (Vec<String>, Vec<String>), b: (Vec<String>, Vec<String>), fuse: u64) -> PyResult<PyObject> {
        let sa = crate::frames_py::parse_seed(&self.st, &a.0, &a.1).map_err(bad)?;
        let sb = crate::frames_py::parse_seed(&self.st, &b.0, &b.1).map_err(bad)?;
        crate::questions_py::question_compose(py, &self.variant, &self.st, text, &sa, &sb, fuse)
    }

    /// question_reopen(text, seeds, fuse) -> the frames of the seeds, their reports, and the pruned root moves reopened by claims of other frames (78.57 E').
    fn question_reopen(&self, py: Python<'_>, text: &str, seeds: Vec<(Vec<String>, Vec<String>)>, fuse: u64) -> PyResult<PyObject> {
        let sds = crate::questions_py::seeds_from(&self.st, &seeds).map_err(bad)?;
        crate::questions_py::question_reopen(py, &self.variant, &self.st, text, &sds, fuse)
    }

    /// questions_measure(games, rows, keys, goal_kind, horizons, fuse, row_fuse, compose_every, max_size, conditional, trap_fuse, threads) -> {obs, count, secs, dump}: stage F of 78.57.
    #[allow(clippy::too_many_arguments)]
    fn questions_measure(&self, py: Python<'_>, games: Vec<GameWire>, rows: Vec<Vec<usize>>, keys: Vec<u64>, goal_kind: Option<String>, horizons: Vec<usize>, fuse: u64, row_fuse: u64, compose_every: usize, max_size: usize, conditional: bool, trap_fuse: u64, threads: usize) -> PyResult<PyObject> {
        let goal = self.goal_of(&goal_kind)?;
        let params = crate::questions_py::MeasureParams { horizons, fuse, row_fuse, compose_every, max_size, conditional, trap_fuse, kv: self.kv.clone(), rules: self.rules, bands: self.bands.clone() };
        crate::questions_py::questions_measure(py, &self.variant, &self.st, goal, games, rows, keys, params, threads)
    }

    /// questions_state(games, rows, keys, goal_kind, fuse, row_fuse, max_size, conditional, trap_fuse, horizon, threads) -> a list, {kinds, roles, places, rows}: one dict per wanted row (game, ply, mover, board, questions, unread, frames, units):
    /// PLAN 78.62 A the units of every frame read at the row with their class, basis, trap word and observed fate within `horizon` plies (the codes are in questions_py.rs).
    #[allow(clippy::too_many_arguments)]
    fn questions_state(&self, py: Python<'_>, games: Vec<GameWire>, rows: Vec<Vec<usize>>, keys: Vec<u64>, goal_kind: Option<String>, fuse: u64, row_fuse: u64, max_size: usize, conditional: bool, trap_fuse: u64, horizon: usize, threads: usize) -> PyResult<PyObject> {
        let goal = self.goal_of(&goal_kind)?;
        let params = crate::questions_py::MeasureParams { horizons: vec![horizon], fuse, row_fuse, compose_every: 1, max_size, conditional, trap_fuse, kv: self.kv.clone(), rules: self.rules, bands: self.bands.clone() };
        crate::questions_py::questions_state(py, &self.variant, &self.st, goal, games, rows, keys, params, horizon, threads)
    }

    /// questions_threats(games, rows, keys, goal_kind, fuse, row_fuse, max_size, conditional, answer_fuse, threads) -> {kinds, roles, places, rows}: PLAN 78.70 A. Per wanted row, every frame read with its members and
    /// places (the read set of its claims) and, per claim, the units that bear on the unit, its class and basis (as questions_state) and the answers of its owner with their reason (codes in questions_py.rs).
    #[allow(clippy::too_many_arguments)]
    fn questions_threats(&self, py: Python<'_>, games: Vec<GameWire>, rows: Vec<Vec<usize>>, keys: Vec<u64>, goal_kind: Option<String>, fuse: u64, row_fuse: u64, max_size: usize, conditional: bool, answer_fuse: u64, threads: usize) -> PyResult<PyObject> {
        let goal = self.goal_of(&goal_kind)?;
        let params = crate::questions_py::MeasureParams { horizons: vec![0], fuse, row_fuse, compose_every: 1, max_size, conditional, trap_fuse: 0, kv: self.kv.clone(), rules: self.rules, bands: self.bands.clone() };
        crate::questions_py::questions_threats(py, &self.variant, &self.st, goal, games, rows, keys, params, answer_fuse, threads)
    }

    /// frames_measure(games, rows, keys, goal_kind, horizons, fuse, row_fuse, compose_every, max_size, conditional, threads) -> {obs, count, secs, dump}: stage G over the wanted rows of the games.
    #[allow(clippy::too_many_arguments)]
    fn frames_measure(&self, py: Python<'_>, games: Vec<GameWire>, rows: Vec<Vec<usize>>, keys: Vec<u64>, goal_kind: Option<String>, horizons: Vec<usize>, fuse: u64, row_fuse: u64, compose_every: usize, max_size: usize, conditional: bool, threads: usize) -> PyResult<PyObject> {
        let goal = self.goal_of(&goal_kind)?;
        let params = crate::frames_py::MeasureParams { horizons, fuse, row_fuse, compose_every, max_size, conditional };
        crate::frames_py::frames_measure(py, &self.variant, &self.st, goal, games, rows, keys, params, threads)
    }
}
