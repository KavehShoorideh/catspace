//! PLAN 78.12 THE PLANNING LAYER'S OPERATORS: `keep` and the `density` prior behind the shared interface (cs_operator). Each wraps the function the engine has always
//! called, unchanged. Game-free.
use crate::{density, keep};
use cs_game::Game;
use cs_operator::{Budget, Goal, Operator, Outcome, Status};
use cs_tactics::ledger::Judge;
use std::collections::HashMap;

/// `keep`: the cheapest move that keeps our state by the measure (the worst reply leaves our units less theirs highest). A heuristic, labelled so.
pub struct KeepOp<'a> { pub worth: &'a HashMap<String, i64> }

impl<'a, G: Game> Operator<G> for KeepOp<'a> {
    fn name(&self) -> String { "keep".into() }
    fn accepts(&self, goal: &Goal) -> bool { matches!(goal, Goal::Keep { .. }) }
    fn run(&self, g: &G, s: &G::State, goal: &Goal, _b: Budget) -> Outcome<G::Move> {
        let Goal::Keep { role } = goal else { return Outcome::unsupported("keep", goal) };
        let t0 = std::time::Instant::now();
        let moves = g.moves(s);
        let Some((i, (worst, replies, reply))) = keep(g, s, *role, &moves, self.worth) else { return Outcome::new(Status::Gap) };
        let mut o = Outcome::new(Status::Read);
        o.evidence.push(format!("planning keep: {} -- the worst reply {reply} leaves the measure {worst}, they have {replies} replies", g.move_name(s, &moves[i])));
        o.mv = Some(moves[i].clone());
        o.value = Some(worst);
        o.cost.nodes = moves.len() as u64;
        o.part("candidates", moves.len() as u64, t0.elapsed().as_secs_f64());
        o.cost.seconds = t0.elapsed().as_secs_f64();
        o
    }
}

/// `density`: among moves another operator found equally good, the one after which our tactics most outnumber theirs. A prior over ties, never a verdict. It reads
/// while the budget's clock lasts (an unread move ranks last); `max_nodes` is the census fuse of each reading, `max_depth` its plies.
pub struct DensityOp<'a, J: Judge> { pub judge: &'a J }

impl<'a, G: Game, J: Judge> Operator<G> for DensityOp<'a, J> {
    fn name(&self) -> String { "density".into() }
    fn accepts(&self, goal: &Goal) -> bool { matches!(goal, Goal::Prefer { .. }) }
    fn run(&self, g: &G, s: &G::State, goal: &Goal, b: Budget) -> Outcome<G::Move> {
        let Goal::Prefer { role, among, defects } = goal else { return Outcome::unsupported("density", goal) };
        let t0 = std::time::Instant::now();
        let depth = b.max_depth.map_or(density::DEPTH, |d| d as i32);
        let Some(pr) = density::prior(g, s, *role, among, self.judge, b.deadline, depth, b.max_nodes, *defects) else {
            return Outcome::new(Status::Gap);
        };
        let (j, (adv0, adv1), read, nodes) = (pr.index, pr.advantage, pr.read, pr.nodes);
        let moves = g.moves(s);
        let mut o = Outcome::new(Status::Read);
        o.evidence.push(format!("planning density: {} of {} equal moves read; {} -- our tactics less theirs after it: {adv0:+} at 0 defects, {adv1:+} at 1 [PRIOR]",
                                read, among.len(), g.move_name(s, &moves[j])));
        o.mv = Some(moves[j].clone());
        o.candidates = vec![j];
        o.value = Some(adv0);
        o.cost.nodes = nodes;
        for (i, n, sec) in &pr.reads { o.part(format!("read {}", g.move_name(s, &moves[*i])), *n, *sec); }
        o.cost.seconds = t0.elapsed().as_secs_f64();
        o
    }
}
