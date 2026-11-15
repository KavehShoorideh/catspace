//! PLAN 78.52: the Python entry of the layer-1 graph (cs_tactics::sheet::base_graph and base_statements). A `BaseGraph` holds the static part of one game (built once from the rules'
//! description through `Placed::steps`); `position(fen)` returns the per-position part as plain node and link lists. Game-free: every name arrives as data.
use cs_game::kernel::KernelGame;
use cs_game::placed::Placed;
use cs_game::Game;
use cs_tactics::sheet::base_graph::{LinkRec, LinkType, NodeRec, NodeType, PositionPart, Static};
use cs_tactics::sheet::base_statements;
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
}

type NodeWire = (String, String);
type LinkWire = (String, u32, u32, i32, i32);

fn nodes_of(v: &[NodeRec]) -> Vec<NodeWire> { v.iter().map(|n| (n.kind.word().to_string(), n.id.clone())).collect() }
fn links_of(v: &[LinkRec]) -> Vec<LinkWire> { v.iter().map(|l| (l.kind.word().to_string(), l.a, l.b, l.via, l.value)).collect() }
fn bad(e: String) -> PyErr { pyo3::exceptions::PyValueError::new_err(e) }

type GameWire = (Option<String>, Vec<String>);

/// The boards of the wanted plies of one game: (row number within the call, the board, the position part's units by (owner is the side to move, kind) counts are made by the callers).
fn walk_rows<T>(variant: &str, st: &Static, goal: Option<u8>, game: &GameWire, wanted: &[usize], mut f: impl FnMut(usize, &Board, &cs_tactics::sheet::base_graph::PositionPart) -> T) -> Result<Vec<Option<T>>, String> {
    let g = match &game.0 { Some(fen) => KernelGame::from_fen(variant, fen)?, None => KernelGame::new(variant, 518)? };
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

impl BaseGraph {
    fn goal_of(&self, goal_kind: &Option<String>) -> PyResult<Option<u8>> {
        match goal_kind { Some(k) => Ok(Some(self.st.names.kind(k).map_err(bad)?)), None => Ok(None) }
    }

    fn game(&self, fen: &str) -> PyResult<(KernelGame, <KernelGame as Game>::State)> {
        let g = KernelGame::from_fen(&self.variant, fen).map_err(bad)?;
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
        Ok(BaseGraph { variant: variant.to_string(), st })
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

    /// position(fen) -> {mover, offset (the number of static nodes: the indices of these nodes and links continue after them), nodes, links}.
    fn position(&self, py: Python<'_>, fen: &str) -> PyResult<PyObject> {
        let (g, s) = self.game(fen)?;
        let p = self.st.position(&g, &s);
        let d = PyDict::new(py);
        d.set_item("mover", p.mover.map(|m| self.st.names.roles[m].clone()))?;
        d.set_item("offset", self.st.nodes.len())?;
        d.set_item("nodes", nodes_of(&p.nodes))?;
        d.set_item("links", links_of(&p.links))?;
        d.into_py_any(py)
    }

    /// dump(fen) -> text: the position's nodes, one per line, with their links.
    fn dump(&self, fen: &str) -> PyResult<String> {
        let (g, s) = self.game(fen)?;
        Ok(self.st.position(&g, &s).dump(&self.st))
    }

    /// bearings(fen) -> [(owner, kind, place, [places borne on])]: what `Placed::bearing_from` reports for each unit, straight from the trait (for the check that the graph holds each once).
    fn bearings(&self, fen: &str) -> PyResult<Vec<(String, String, String, Vec<String>)>> {
        let (g, s) = self.game(fen)?;
        let n = &self.st.names;
        Ok(g.cells(&s).into_iter().map(|(o, k, at)| (n.roles[o].clone(), n.kinds[k as usize].clone(), n.places[at as usize].clone(),
            g.bearing_from(&s, o, k, at).into_iter().map(|p| n.places[p as usize].clone()).collect())).collect())
    }

    /// statement_names() -> ([(name, text)] sided, [(name, text)] of the board).
    fn statement_names(&self) -> (Vec<NodeWire>, Vec<NodeWire>) {
        (base_statements::names(&self.st), base_statements::BOARD.iter().map(|(a, b)| (a.to_string(), b.to_string())).collect())
    }

    /// statements(fen, worth, goal_kind) -> (us, them, board): the statements over the graph of the position, the side to move first.
    fn statements(&self, fen: &str, worth: HashMap<String, f64>, goal_kind: Option<String>) -> PyResult<(Vec<f64>, Vec<f64>, Vec<f64>)> {
        let (g, s) = self.game(fen)?;
        let n = &self.st.names;
        let w: Vec<f64> = n.kinds.iter().map(|k| worth.get(k).copied().unwrap_or(0.0)).collect();
        let gk = match goal_kind { Some(k) => Some(n.kind(&k).map_err(bad)?), None => None };
        let p = self.st.position(&g, &s);
        if p.mover.is_none() { return Err(bad("the position is over".into())); }
        let ([a, b], board) = base_statements::statements(&self.st, &p, &w, gk);
        Ok((a, b, board))
    }

    /// open_lengths(fen) -> [(line id, the longest run of places with no unit)].
    fn open_lengths(&self, fen: &str) -> PyResult<Vec<(String, usize)>> {
        let (g, s) = self.game(fen)?;
        let p = self.st.position(&g, &s);
        Ok((0..self.st.lines.len()).map(|l| (self.st.nodes[self.st.line_node[l] as usize].id.clone(), p.open_length(&self.st, l))).collect())
    }

    /// measure(fens) -> {positions, seconds (the graph build alone, the states parsed beforehand), nodes: {type: count}, links: {type: count}, static_nodes, static_links} over the
    /// positions (summed).
    fn measure(&self, py: Python<'_>, fens: Vec<String>) -> PyResult<PyObject> {
        let mut states = vec![];
        for f in &fens { states.push(self.game(f)?); }
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
            let g = match &games[i].0 { Some(fen) => KernelGame::from_fen(&variant, fen)?, None => KernelGame::new(&variant, 518)? };
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

    /// first_blocker_check(fen) -> (the shut relations of the side to move with exactly one unit between, as `why_not` gives them; the same set found by the first-blocker form through `without`; whether the two sets are equal).
    fn first_blocker_check(&self, fen: &str) -> PyResult<(usize, usize, bool)> {
        let (g, s) = self.game(fen)?;
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
}
