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
    /// 78.58: the worth of each kind (by kind index) the exchange reads, set from the estimator's record; None: no exchange.
    kv: Option<Vec<i64>>,
    /// 78.60: the band (low, high) of the worth of each kind, from the estimator record; and the rules of a line's end (all off until set).
    bands: Option<Vec<(i64, i64)>>,
    rules: cs_tactics::sheet::questions::Rules,
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
        Ok(BaseGraph { variant: variant.to_string(), st, kv: None, bands: None, rules: Default::default() })
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
