//! PLAN 78.21 THE GRAPH PRIOR, ALL IN RUST: the position's typed fact graph (graph.rs) scored by the exported graph net (graph_net.rs). It ranks the legal moves and
//! claims nothing: a `Prior` for playouts (weights = softmax of the scores times the number of moves; uniform on any error) and the engine of the rank source "gnn".
//! Timing counters: seconds in the fact layer, in graph building (without the fact layer) and in the forward pass, and the positions scored.
use crate::graph::GraphSpec;
use crate::graph_net::GraphNet;
use cs_game::Game;
use cs_planning::rollout::Prior;
use std::cell::Cell;
use std::collections::HashMap;
use std::rc::Rc;
use std::time::Instant;

#[derive(Clone)]
pub struct GraphPrior {
    pub spec: Rc<GraphSpec>,
    pub net: Rc<GraphNet>,
    positions: Cell<u64>,
    secs_facts: Cell<f64>,
    secs_graph: Cell<f64>,
    secs_forward: Cell<f64>,
}

impl GraphPrior {
    /// The prior of a net over a spec; an error unless the net's vocabularies are the spec's.
    pub fn new(spec: Rc<GraphSpec>, net: Rc<GraphNet>) -> Result<GraphPrior, String> {
        net.validate(&spec)?;
        Ok(GraphPrior { spec, net, positions: Cell::new(0), secs_facts: Cell::new(0.0), secs_graph: Cell::new(0.0), secs_forward: Cell::new(0.0) })
    }

    /// The legal moves' names (the game's move order) and their scores.
    pub fn scores<G: Game>(&self, g: &G, s: &G::State) -> Result<(Vec<String>, Vec<f32>), String> {
        let t0 = Instant::now();
        let graph = self.spec.graph(g, s)?;
        let built = t0.elapsed().as_secs_f64();
        let t1 = Instant::now();
        let scores = self.net.forward(&graph);
        self.positions.set(self.positions.get() + 1);
        self.secs_facts.set(self.secs_facts.get() + graph.secs_facts);
        self.secs_graph.set(self.secs_graph.get() + (built - graph.secs_facts).max(0.0));
        self.secs_forward.set(self.secs_forward.get() + t1.elapsed().as_secs_f64());
        Ok((graph.moves, scores))
    }

    /// (positions, seconds in the fact layer, seconds building graphs, seconds in the forward pass).
    pub fn timing(&self) -> (u64, f64, f64, f64) { (self.positions.get(), self.secs_facts.get(), self.secs_graph.get(), self.secs_forward.get()) }
}

/// softmax(scores) * n (the mean weight is 1).
pub fn softmax_weights(scores: &[f32]) -> Vec<f64> {
    let top = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
    let e: Vec<f64> = scores.iter().map(|s| (*s as f64 - top).exp()).collect();
    let z: f64 = e.iter().sum();
    e.iter().map(|x| x / z * scores.len() as f64).collect()
}

impl Prior for GraphPrior {
    fn weights<G: Game>(&self, g: &G, s: &G::State, moves: &[G::Move]) -> Vec<f64> {
        let uniform = vec![1.0; moves.len()];
        let Ok((names, scores)) = self.scores(g, s) else { return uniform };
        if scores.is_empty() { return uniform; }
        let w = softmax_weights(&scores);
        let by_name: HashMap<&str, f64> = names.iter().map(|n| n.as_str()).zip(w.iter().copied()).collect();
        moves.iter().map(|m| by_name.get(g.move_name(s, m).as_str()).copied().unwrap_or(1.0)).collect()
    }
    fn name(&self) -> String { "graph".into() }
}
