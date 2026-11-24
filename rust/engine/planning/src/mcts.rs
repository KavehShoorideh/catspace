//! PLAN 78 MONTE CARLO TREE SEARCH: one operator among the others (the architecture document: "inference has lemmas and MCTS"). UCT over the `Game` trait alone,
//! multi-role: a terminal outcome is a value per role, every node on the path takes each role's share, and the side to move maximizes ITS OWN mean. Playouts come
//! from a `Rollout` (a small trait: pick one of the moves offered), so a policy from a lemma prior or a shallow search can replace the uniform default later.
//! What it answers is an ESTIMATE by sampling (`Goal::Estimate`), never a proof: the status is `Read`, the value is the mean outcome for the role in
//! HUNDREDTHS of an outcome unit (`VALUE_SCALE`; outcomes are 0..100, so a value of 5000 is 50.00), the candidates are the root moves most visited first.
use cs_game::rng::Rng;
use cs_game::{Fact, Game, Role};
use crate::rollout::{Draw, Leaf};
use crate::wdl::{split, Wdl};
use cs_operator::{Budget, Goal, Operator, Outcome, Status};
use std::time::Instant;

/// The game's outcome units run 0..OUTCOME_MAX (GDL's goal): the scale UCT normalizes by.
pub const OUTCOME_MAX: f64 = 100.0;
/// What a playout cut at its cap scores for every role: a draw (the middle of the outcome units).
pub const DRAW: f64 = 50.0;
/// The reported value is the mean outcome times this (an integer in hundredths of an outcome unit).
pub const VALUE_SCALE: f64 = 100.0;
/// One part of the cost is recorded per this many playouts.
pub const BATCH: u64 = 64;
/// The default UCT exploration constant (outcomes are normalized to 0..1).
pub const EXPLORATION: f64 = 1.41;
/// The default cap on a playout's length, in plies.
pub const MAX_PLAYOUT: usize = 200;

/// How a playout chooses among the legal moves. Small on purpose: another policy (a lemma prior, a shallow search) implements this and plugs into `MctsOp`.
pub trait Rollout {
    /// The index into `moves` (never empty) of the move to play at `s`.
    fn pick<G: Game>(&mut self, g: &G, s: &G::State, moves: &[G::Move]) -> usize;
    /// The policy's name, for the operator's evidence.
    fn name(&self) -> String;
}

/// Uniformly random legal moves from the engine's seeded generator: the same seed, the same playouts. Cloning starts the stream again from the seed (each run of an
/// operator is reproducible on its own).
pub struct Uniform { seed: u64, rng: Rng }
impl Uniform { pub fn new(seed: u64) -> Uniform { Uniform { seed, rng: Rng::new(seed) } } }
impl Clone for Uniform { fn clone(&self) -> Uniform { Uniform::new(self.seed) } }
impl Rollout for Uniform {
    fn pick<G: Game>(&mut self, _g: &G, _s: &G::State, moves: &[G::Move]) -> usize { self.rng.below(moves.len()) }
    fn name(&self) -> String { "uniform".into() }
}

struct Node<G: Game> {
    state: G::State,
    outcome: Option<Vec<i64>>,
    moves: Vec<G::Move>,
    /// The child reached by each move (by index), once expanded.
    children: Vec<Option<usize>>,
    /// Moves are expanded in order: the next one not yet tried.
    next: usize,
    mover: Role,
    visits: u64,
    sums: Vec<f64>,
    depth: i64,
}

/// A finished search: the root's moves with their visits and mean outcomes, and what was measured.
#[derive(Debug, Clone, Default)]
pub struct Search {
    /// Per root move (by index): (visits, mean outcome per role in outcome units).
    pub root: Vec<(u64, Vec<f64>)>,
    pub playouts: u64,
    pub cut: u64,
    pub depth: i64,
    /// Budget nodes spent and seconds, per batch of `BATCH` playouts (the last batch may be short); the nodes sum to `nodes`.
    pub batches: Vec<(u64, f64)>,
    pub nodes: u64,
    /// The roles' mean outcomes at the root over all playouts.
    pub root_mean: Vec<f64>,
    /// Plan 78.26: per role, the win / draw / loss MASS of every playout (a cut playout's leaf value split by `wdl::split`); the mass is the playouts.
    pub root_wdl: Vec<Wdl>,
    /// Per root move (by index), per role: the mass of the playouts that went through that move.
    pub move_wdl: Vec<Vec<Wdl>>,
}

/// UCT from `s` until the budget's fuse or clock says stop (an iteration it interrupts is dropped, not scored). Every step -- a selection, an expansion, a playout ply
/// -- spends one node.
pub fn search<G: Game, R: Rollout, L: Leaf>(g: &G, s: &G::State, budget: &mut Budget, exploration: f64, max_playout: usize, rollout: &mut R, leaf: &L) -> Search {
    let n_roles = g.roles().len();
    let make = |st: G::State, depth: i64| -> Node<G> {
        let outcome = g.outcome(&st);
        let moves = if outcome.is_some() { vec![] } else { g.moves(&st) };
        let mover = if outcome.is_some() { 0 } else { g.turn(&st) };
        Node { children: vec![None; moves.len()], state: st, outcome, moves, next: 0, mover, visits: 0, sums: vec![0.0; n_roles], depth }
    };
    let mut tree: Vec<Node<G>> = vec![make(s.clone(), 0)];
    let mut res = Search::default();
    res.root_wdl = vec![Wdl::default(); n_roles];
    res.move_wdl = vec![vec![Wdl::default(); n_roles]; tree[0].moves.len()];
    let (mut mark_nodes, mut mark_t) = (budget.nodes, Instant::now());
    while budget.left() {
        // selection and expansion
        let mut path = vec![0usize];
        let mut cur = 0usize;
        let mut scored: Option<Vec<f64>> = None;
        let mut stopped = false;
        loop {
            if let Some(o) = &tree[cur].outcome { scored = Some(o.iter().map(|x| *x as f64).collect()); break; }
            if tree[cur].moves.is_empty() { res.cut += 1; scored = Some(leaf.value(g, &tree[cur].state)); break; }
            if budget.spend().is_err() { stopped = true; break; }
            let next = if tree[cur].next < tree[cur].moves.len() {
                let i = tree[cur].next;
                tree[cur].next += 1;
                let st = g.apply(&tree[cur].state, &tree[cur].moves[i]);
                let d = tree[cur].depth + 1;
                tree.push(make(st, d));
                let id = tree.len() - 1;
                tree[cur].children[i] = Some(id);
                res.depth = res.depth.max(d);
                (id, true)
            } else {
                let node = &tree[cur];
                let ln = (node.visits.max(1) as f64).ln();
                let mut best = (f64::NEG_INFINITY, 0usize);
                for (i, c) in node.children.iter().enumerate() {
                    let c = &tree[c.expect("all moves expanded")];
                    let u = c.sums[node.mover] / (c.visits.max(1) as f64) / OUTCOME_MAX + exploration * (ln / c.visits.max(1) as f64).sqrt();
                    if u > best.0 { best = (u, i); }
                }
                (node.children[best.1].unwrap(), false)
            };
            cur = next.0;
            path.push(cur);
            if next.1 { break; }
        }
        if stopped { break; }
        // playout from the new node (or the terminal one the selection reached)
        let value = match scored {
            Some(v) => v,
            None => {
                let mut st = tree[cur].state.clone();
                let mut plies = 0usize;
                let mut result: Option<Vec<f64>> = None;
                loop {
                    if let Some(o) = g.outcome(&st) { result = Some(o.iter().map(|x| *x as f64).collect()); break; }
                    if plies >= max_playout { res.cut += 1; result = Some(leaf.value(g, &st)); break; }
                    let ms = g.moves(&st);
                    if ms.is_empty() { res.cut += 1; result = Some(leaf.value(g, &st)); break; }
                    if budget.spend().is_err() { break; }
                    let i = rollout.pick(g, &st, &ms);
                    st = g.apply(&st, &ms[i]);
                    plies += 1;
                }
                match result { Some(v) => v, None => break }
            }
        };
        // backup: every node on the path takes each role's share
        for &id in &path {
            let n = &mut tree[id];
            n.visits += 1;
            for r in 0..n_roles { n.sums[r] += value[r]; }
        }
        // (the win / draw / loss tallies: the root's, and the root move this playout went through)
        let first = path.get(1).and_then(|id| tree[0].children.iter().position(|c| *c == Some(*id)));
        for r in 0..n_roles {
            let m = split(value[r]);
            res.root_wdl[r].add_mass(m);
            if let Some(i) = first { res.move_wdl[i][r].add_mass(m); }
        }
        res.playouts += 1;
        if res.playouts % BATCH == 0 {
            res.batches.push((budget.nodes - mark_nodes, mark_t.elapsed().as_secs_f64()));
            mark_nodes = budget.nodes;
            mark_t = Instant::now();
        }
    }
    if budget.nodes > mark_nodes || res.batches.is_empty() || res.playouts % BATCH != 0 {
        res.batches.push((budget.nodes - mark_nodes, mark_t.elapsed().as_secs_f64()));
    }
    res.nodes = budget.nodes;
    let mean = |n: &Node<G>| -> Vec<f64> { n.sums.iter().map(|x| x / (n.visits.max(1) as f64)).collect() };
    res.root_mean = mean(&tree[0]);
    res.root = tree[0].children.iter().map(|c| match c { Some(id) => (tree[*id].visits, mean(&tree[*id])), None => (0, vec![DRAW; n_roles]) }).collect();
    res
}

/// THE OPERATOR: MCTS as one tool of the shared interface. Reads `Goal::Estimate` only. The playout policy (`rollout`) and the valuation of a playout cut short
/// (`leaf`) are the two seams.
pub struct MctsOp<R: Rollout + Clone = Uniform, L: Leaf + Clone = Draw> {
    pub exploration: f64,
    pub max_playout: usize,
    pub rollout: R,
    pub leaf: L,
}

impl MctsOp<Uniform, Draw> {
    pub fn new(exploration: f64, max_playout: usize, seed: u64) -> MctsOp<Uniform, Draw> { MctsOp { exploration, max_playout, rollout: Uniform::new(seed), leaf: Draw } }
}
impl<R: Rollout + Clone> MctsOp<R, Draw> {
    /// The same search with another playout policy (a cut playout still scores a draw).
    pub fn with_rollout(exploration: f64, max_playout: usize, rollout: R) -> MctsOp<R, Draw> { MctsOp { exploration, max_playout, rollout, leaf: Draw } }
}
impl<R: Rollout + Clone, L: Leaf + Clone> MctsOp<R, L> {
    /// Another playout policy and another valuation of a cut playout.
    pub fn with_policy(exploration: f64, max_playout: usize, rollout: R, leaf: L) -> MctsOp<R, L> { MctsOp { exploration, max_playout, rollout, leaf } }
}

impl<G: Game, R: Rollout + Clone, L: Leaf + Clone> Operator<G> for MctsOp<R, L> {
    fn name(&self) -> String { "mcts".into() }
    fn accepts(&self, goal: &Goal) -> bool { matches!(goal, Goal::Estimate { .. }) }
    fn run(&self, g: &G, s: &G::State, goal: &Goal, mut budget: Budget) -> Outcome<G::Move> {
        let Goal::Estimate { role } = goal else { return Outcome::unsupported("mcts", goal) };
        let role = *role;
        let t0 = Instant::now();
        let mut out = Outcome::new(Status::Read);
        let mut rollout = self.rollout.clone();
        let r = search(g, s, &mut budget, self.exploration, self.max_playout, &mut rollout, &self.leaf);
        let moves = g.moves(s);
        let scaled = |m: f64| (m * VALUE_SCALE).round() as i64;
        if r.playouts == 0 {
            out.status = Status::Gap;
            out.evidence.push("mcts: the budget ended before one playout finished".into());
        } else {
            out.value = Some(scaled(r.root_mean.get(role).copied().unwrap_or(DRAW)));
            let mut order: Vec<usize> = (0..r.root.len()).collect();
            order.sort_by(|a, b| r.root[*b].0.cmp(&r.root[*a].0).then(a.cmp(b)));
            if let Some(&top) = order.first() { out.mv = moves.get(top).cloned(); }
            out.candidates = order;
        }
        out.depth = Some(r.depth);
        for (i, (v, mean)) in r.root.iter().enumerate() {
            let m = scaled(mean.get(role).copied().unwrap_or(DRAW));
            out.facts.push(Fact { relation: "visits".into(), args: vec![i.to_string(), v.to_string(), m.to_string()] });
        }
        out.facts.push(Fact { relation: "playouts".into(), args: vec![r.playouts.to_string(), r.cut.to_string()] });
        for (i, (n, sec)) in r.batches.iter().enumerate() { out.part(format!("playouts {}..{}", i as u64 * BATCH, ((i as u64 + 1) * BATCH).min(r.playouts.max(1))), *n, *sec); }
        out.cost.nodes = r.nodes;
        out.cost.seconds = t0.elapsed().as_secs_f64();
        out.evidence.push(format!("mcts: {} playouts ({} cut at {} plies), depth {}, rollout {}, leaf {}, value for role {role} = mean outcome x {VALUE_SCALE} (an estimate by sampling, not a proof)", r.playouts, r.cut, self.max_playout, r.depth, self.rollout.name(), self.leaf.name()));
        out
    }
}
