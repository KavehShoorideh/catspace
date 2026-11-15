//! PLAN 78.52: the Python entry of the layer-1 graph (cs_tactics::sheet::base_graph and base_statements). A `BaseGraph` holds the static part of one game (built once from the rules'
//! description through `Placed::steps`); `position(fen)` returns the per-position part as plain node and link lists. Game-free: every name arrives as data.
use cs_game::kernel::KernelGame;
use cs_game::placed::Placed;
use cs_game::Game;
use cs_tactics::sheet::base_graph::{LinkRec, LinkType, NodeRec, NodeType, PositionPart, Static};
use cs_tactics::sheet::base_statements;
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

impl BaseGraph {
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
}
