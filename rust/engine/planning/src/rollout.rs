//! THE STRONGER RANK SOURCE for MCTS, as small pluggable pieces (game-free: only the `Game` trait and the ledger's `Judge`). Two seams:
//!   `Leaf`     the value per role of a NON-terminal state where a playout is cut (`Draw`: today's behaviour; `Balance`: the units standing, by the judge's worths);
//!   `Rollout`  how a playout picks its moves (`Uniform` in mcts.rs; `Greedy`: take the best immediate gain; `Careful`: the best gain less what the move leaves the
//!              next mover to take; `Weighted<P>`: sample by a `Prior`'s weights).
//! `Greedy` and `Weighted<GainPrior>` are two policies over ONE notion of gain (`gain`): the units a move takes plus the worth its own units gain by changing kind.
//! A learned prior or a shallow search plugs in by implementing `Prior` / `Rollout` / `Leaf`; nothing here is wired into `search`.
use cs_game::rng::Rng;
use cs_game::Game;
use cs_tactics::ledger::Judge;

use crate::mcts::{Rollout, DRAW};

/// The value per role (outcome units 0..100) of a NON-terminal state where a playout is cut.
pub trait Leaf {
    fn value<G: Game>(&self, g: &G, s: &G::State) -> Vec<f64>;
    fn name(&self) -> String;
}

/// Every role gets `DRAW`: a cut playout scores as a draw (the behaviour before leaf valuation).
#[derive(Clone, Copy, Default)]
pub struct Draw;
impl Leaf for Draw {
    fn value<G: Game>(&self, g: &G, _s: &G::State) -> Vec<f64> { vec![DRAW; g.roles().len()] }
    fn name(&self) -> String { "draw".into() }
}

/// The material balance of each role, squashed to outcome units: balance_r = sum over the units of worth(kind) x (+1 if owned by r, else -1/(n_roles-1)); value_r =
/// 100 / (1 + exp(-balance_r / scale)). `scale` is in worth units, chosen by the caller: with a `Count` judge a scale of about 3 units means 3 units up is about 73%.
/// A balance of 0 is exactly 50.
#[derive(Clone)]
pub struct Balance<J: Judge> { pub judge: J, pub scale: f64 }
impl<J: Judge> Balance<J> {
    pub fn new(judge: J, scale: f64) -> Balance<J> { Balance { judge, scale } }
}
impl<J: Judge> Leaf for Balance<J> {
    fn value<G: Game>(&self, g: &G, s: &G::State) -> Vec<f64> {
        let n = g.roles().len();
        let others = if n > 1 { (n - 1) as f64 } else { 1.0 };
        let mut bal = vec![0.0f64; n];
        for (owner, kind) in g.units(s) {
            let w = self.judge.worth(&kind) as f64;
            for (r, b) in bal.iter_mut().enumerate() { *b += if owner == r { w } else { -w / others }; }
        }
        bal.iter().map(|b| if *b == 0.0 { 50.0 } else { 100.0 / (1.0 + (-b / self.scale).exp()) }).collect()
    }
    fn name(&self) -> String { format!("balance({}, {})", self.judge.name(), self.scale) }
}

/// A move's immediate gain for the mover: the worth of the units it takes, plus (worth(to) - worth(from)) for each of the mover's own units changing kind.
pub fn gain<G: Game, J: Judge>(g: &G, judge: &J, s: &G::State, m: &G::Move) -> i64 {
    let taken: i64 = g.captured(s, m).iter().map(|(_, k)| judge.worth(k)).sum();
    let changed: i64 = g.changed(s, m).iter().map(|(from, to)| judge.worth(to) - if from.is_empty() { 0 } else { judge.worth(from) }).sum();
    taken + changed
}

/// A uniform float in [0, 1) from the engine's generator.
fn unit(rng: &mut Rng) -> f64 { (rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64 }

/// A ONE-PLY (greedy) look: with probability `eps` a uniform move; otherwise the move of best immediate `gain` (uniformly among ties) if that gain is above 0, else
/// uniform. It sees nothing past the move: the leaf valuation is what covers what the move gives away. Cloning restarts the stream from the seed.
pub struct Greedy<J: Judge> { pub judge: J, pub eps: f64, seed: u64, rng: Rng }
impl<J: Judge> Greedy<J> {
    pub fn new(judge: J, eps: f64, seed: u64) -> Greedy<J> { Greedy { judge, eps, seed, rng: Rng::new(seed) } }
}
impl<J: Judge + Clone> Clone for Greedy<J> { fn clone(&self) -> Greedy<J> { Greedy::new(self.judge.clone(), self.eps, self.seed) } }
impl<J: Judge> Rollout for Greedy<J> {
    fn pick<G: Game>(&mut self, g: &G, s: &G::State, moves: &[G::Move]) -> usize {
        if unit(&mut self.rng) < self.eps { return self.rng.below(moves.len()); }
        let gains: Vec<i64> = moves.iter().map(|m| gain(g, &self.judge, s, m)).collect();
        let best = gains.iter().copied().max().unwrap_or(0);
        if best <= 0 { return self.rng.below(moves.len()); }
        let ties: Vec<usize> = (0..moves.len()).filter(|i| gains[*i] == best).collect();
        ties[self.rng.below(ties.len())]
    }
    fn name(&self) -> String { "greedy".into() }
}

/// What a terminal state reached by a playout's own move counts for in `Careful` (above any gain the judge can give).
pub const SURE: i64 = 1_000_000;

/// A TWO-PLY (careful) look: with probability `eps` a uniform move; otherwise the move of best NET, uniformly among ties, where a move's net is its immediate `gain`
/// less the best immediate gain it leaves the next mover (nothing when the next mover has no gaining move). A move that ends the game counts by its outcome for the
/// mover (`SURE` times the distance from a draw). It gives nothing away for free, which the one-ply look does. Cloning restarts the stream from the seed.
pub struct Careful<J: Judge> { pub judge: J, pub eps: f64, seed: u64, rng: Rng }
impl<J: Judge> Careful<J> {
    pub fn new(judge: J, eps: f64, seed: u64) -> Careful<J> { Careful { judge, eps, seed, rng: Rng::new(seed) } }
}
impl<J: Judge + Clone> Clone for Careful<J> { fn clone(&self) -> Careful<J> { Careful::new(self.judge.clone(), self.eps, self.seed) } }
impl<J: Judge> Rollout for Careful<J> {
    fn pick<G: Game>(&mut self, g: &G, s: &G::State, moves: &[G::Move]) -> usize {
        if unit(&mut self.rng) < self.eps { return self.rng.below(moves.len()); }
        let Some(me) = g.mover(s) else { return self.rng.below(moves.len()) };
        let nets: Vec<i64> = moves.iter().map(|m| {
            let q = g.apply(s, m);
            if let Some(o) = g.outcome(&q) { return (o.get(me).copied().unwrap_or(DRAW as i64) - DRAW as i64) * SURE; }
            let theirs = g.moves(&q).iter().map(|r| gain(g, &self.judge, &q, r)).max().unwrap_or(0).max(0);
            gain(g, &self.judge, s, m) - theirs
        }).collect();
        let best = nets.iter().copied().max().unwrap_or(0);
        let ties: Vec<usize> = (0..moves.len()).filter(|i| nets[*i] == best).collect();
        ties[self.rng.below(ties.len())]
    }
    fn name(&self) -> String { "careful".into() }
}

/// A prior over the moves of a state: a non-negative weight per move (same length as `moves`).
pub trait Prior {
    fn weights<G: Game>(&self, g: &G, s: &G::State, moves: &[G::Move]) -> Vec<f64>;
    fn name(&self) -> String;
}

/// Sample a move with probability proportional to (weight + `floor`); `floor` > 0 keeps every move possible. Cloning restarts the stream from the seed.
pub struct Weighted<P: Prior + Clone> { pub prior: P, pub floor: f64, seed: u64, rng: Rng }
impl<P: Prior + Clone> Weighted<P> {
    pub fn new(prior: P, floor: f64, seed: u64) -> Weighted<P> { Weighted { prior, floor, seed, rng: Rng::new(seed) } }
}
impl<P: Prior + Clone> Clone for Weighted<P> { fn clone(&self) -> Weighted<P> { Weighted::new(self.prior.clone(), self.floor, self.seed) } }
impl<P: Prior + Clone> Rollout for Weighted<P> {
    fn pick<G: Game>(&mut self, g: &G, s: &G::State, moves: &[G::Move]) -> usize {
        let w: Vec<f64> = self.prior.weights(g, s, moves).into_iter().map(|x| if x.is_finite() && x > 0.0 { x + self.floor } else { self.floor }).collect();
        let total: f64 = w.iter().sum();
        if w.len() != moves.len() || total <= 0.0 { return self.rng.below(moves.len()); }
        let mut x = unit(&mut self.rng) * total;
        for (i, wi) in w.iter().enumerate() { if x < *wi { return i; } x -= wi; }
        moves.len() - 1
    }
    fn name(&self) -> String { format!("weighted({})", self.prior.name()) }
}

/// The weight exp(gain / temperature) of each move, by the same `gain` as `Greedy` (a low temperature approaches greedy, a high one uniform).
#[derive(Clone)]
pub struct GainPrior<J: Judge> { pub judge: J, pub temperature: f64 }
impl<J: Judge> Prior for GainPrior<J> {
    fn weights<G: Game>(&self, g: &G, s: &G::State, moves: &[G::Move]) -> Vec<f64> {
        let gains: Vec<i64> = moves.iter().map(|m| gain(g, &self.judge, s, m)).collect();
        let top = gains.iter().copied().max().unwrap_or(0);
        // shifted by the best gain (the same ratios, no overflow)
        gains.iter().map(|x| ((*x - top) as f64 / self.temperature).exp()).collect()
    }
    fn name(&self) -> String { format!("gain({}, T={})", self.judge.name(), self.temperature) }
}

