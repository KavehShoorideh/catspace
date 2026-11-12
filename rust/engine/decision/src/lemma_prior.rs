//! A LEMMA PRIOR for playouts (the `Prior` of cs_planning::rollout): at a state, the snapshot's lemmas whose body binds in the mover's perspective facts name the
//! moves they respond with; those moves get weight `boost`, every other move 1.0. It matches exactly as the lemma operator does (cs_memory::op::LemmaOp: the same
//! fact view, perspective, step fuse, binding cap, delta relations and `response_matches`). COST: every call rebuilds the facts and, if some lemma binds, the delta
//! of EVERY legal move (one `apply` and one fact reading each) -- far dearer than a uniform pick; acceptable for now, a cache by state key is the next step. A
//! state with no facts (a game that gives none), a failing fact view, or no bound lemma gives uniform weights.
use cs_game::gdl_game::FactView;
use cs_game::Game;
use cs_memory::cards::{response_matches, STEP_FUSE};
use cs_memory::matcher::{bindings_fused, Binding, FactIndex};
use cs_memory::op::{Effects, MAX_BINDINGS};
use cs_memory::perspective;
use cs_memory::snapshot::Snapshot;
use cs_game::rng::Rng;
use cs_planning::rollout::Prior;

#[derive(Clone, Copy)]
pub struct LemmaPrior<'a> {
    pub snapshot: &'a Snapshot,
    pub view: &'a FactView,
    /// The weight of a move some bound lemma responds with (the others weigh 1.0).
    pub boost: f64,
}

impl<'a> LemmaPrior<'a> {
    pub fn new(snapshot: &'a Snapshot, view: &'a FactView, boost: f64) -> LemmaPrior<'a> { LemmaPrior { snapshot, view, boost } }
}

impl<'a> Prior for LemmaPrior<'a> {
    fn weights<G: Game>(&self, g: &G, s: &G::State, moves: &[G::Move]) -> Vec<f64> {
        let uniform = vec![1.0; moves.len()];
        let base = g.base_facts(s);
        let Some(mover) = g.mover(s) else { return uniform };
        if base.is_empty() { return uniform; }
        let roles = g.roles();
        let Ok(facts) = self.view.facts(&base).map(|f| perspective(&f, &roles, mover)) else { return uniform };
        let idx = FactIndex::new(&facts);
        let mut deltas: Option<Vec<(FactIndex, FactIndex)>> = None;
        let mut out = uniform;
        let mut steps = 0u64;
        for lemma in self.snapshot.lemmas.iter().filter(|l| !l.suggest) {          // (plan 78.36: a suggestion never acts unverified: the playout prior leaves it out)
            let (bs, _cut) = bindings_fused(&lemma.body, &Binding::new(), &idx, MAX_BINDINGS, &mut steps, STEP_FUSE);
            if bs.is_empty() { continue; }
            if deltas.is_none() {
                let built = Effects::at(self.view, &self.snapshot.vocabulary.delta, &base).and_then(|ef| moves.iter().map(|m| {
                    let (r, a) = ef.of(&g.base_facts(&g.apply(s, m)))?;
                    Ok((FactIndex::new(&perspective(&r, &roles, mover)), FactIndex::new(&perspective(&a, &roles, mover))))
                }).collect::<Result<Vec<_>, String>>());
                let Ok(d) = built else { return out };          // (a failing fact view gives uniform weights, as above)
                deltas = Some(d);
            }
            let ds = deltas.as_ref().unwrap();
            for (i, (r, a)) in ds.iter().enumerate() {
                if out[i] != self.boost && bs.iter().any(|b| response_matches(lemma, b, r, a, &mut steps)) { out[i] = self.boost; }
            }
        }
        out
    }
    fn name(&self) -> String { format!("lemma(boost {})", self.boost) }
}

/// A PLAYOUT POLICY "RULES FIRST, THE FALLBACK OTHERWISE" (a `Rollout` of cs_planning): with probability `eps` a uniform move; else, when the prior names some moves
/// (a weight above the neutral 1.0: `LemmaPrior` weighs a named move `boost` > 1 and every other 1.0), one of the named moves, uniformly among them; else whatever
/// the fallback policy picks. `named` counts the picks where a rule spoke (not the uniform ones), `picks` all picks. Cloning restarts both streams and counters.
pub struct RulesFirst<P: Prior + Clone, R: cs_planning::mcts::Rollout + Clone> { pub prior: P, pub fallback: R, pub eps: f64, pub named: u64, pub picks: u64, seed: u64, rng: Rng }

impl<P: Prior + Clone, R: cs_planning::mcts::Rollout + Clone> RulesFirst<P, R> {
    pub fn new(prior: P, fallback: R, eps: f64, seed: u64) -> RulesFirst<P, R> { RulesFirst { prior, fallback, eps, named: 0, picks: 0, seed, rng: Rng::new(seed ^ 0x9e37_79b9_7f4a_7c15) } }
}
impl<P: Prior + Clone, R: cs_planning::mcts::Rollout + Clone> Clone for RulesFirst<P, R> {
    fn clone(&self) -> RulesFirst<P, R> { RulesFirst::new(self.prior.clone(), self.fallback.clone(), self.eps, self.seed) }
}
impl<P: Prior + Clone, R: cs_planning::mcts::Rollout + Clone> cs_planning::mcts::Rollout for RulesFirst<P, R> {
    fn pick<G: Game>(&mut self, g: &G, s: &G::State, moves: &[G::Move]) -> usize {
        self.picks += 1;
        if ((self.rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64) < self.eps { return self.rng.below(moves.len()); }
        let w = self.prior.weights(g, s, moves);
        let named: Vec<usize> = if w.len() == moves.len() { (0..moves.len()).filter(|i| w[*i].is_finite() && w[*i] > 1.0).collect() } else { vec![] };
        if named.is_empty() { return self.fallback.pick(g, s, moves); }
        self.named += 1;
        named[self.rng.below(named.len())]
    }
    fn name(&self) -> String { format!("rules first ({}), else {}", self.prior.name(), self.fallback.name()) }
}

#[cfg(test)]
mod rules_first_tests {
    use super::*;
    use cs_game::Game;
    use cs_planning::mcts::Rollout;

    /// A tiny counting game: the state is a number, a move adds 1..=3, the game ends at 10 or more.
    #[derive(Clone)]
    struct Count10;
    impl Game for Count10 {
        type State = u32;
        type Move = u32;
        fn name(&self) -> String { "count10".into() }
        fn move_name(&self, _s: &u32, m: &u32) -> String { m.to_string() }
        fn roles(&self) -> Vec<String> { vec!["a".into(), "b".into()] }
        fn initial(&self) -> u32 { 0 }
        fn mover(&self, s: &u32) -> Option<usize> { if *s >= 10 { None } else { Some((*s % 2) as usize) } }
        fn moves(&self, s: &u32) -> Vec<u32> { if *s >= 10 { vec![] } else { vec![1, 2, 3] } }
        fn apply(&self, s: &u32, m: &u32) -> u32 { s + m }
        fn outcome(&self, s: &u32) -> Option<Vec<i64>> { if *s >= 10 { Some(vec![50, 50]) } else { None } }
        fn key(&self, s: &u32) -> u64 { *s as u64 }
    }
    #[derive(Clone)]
    struct Names(u32);
    impl Prior for Names {
        fn weights<G: Game>(&self, _g: &G, _s: &G::State, moves: &[G::Move]) -> Vec<f64> { moves.iter().enumerate().map(|(i, _)| if i as u32 + 1 == self.0 { 2.0 } else { 1.0 }).collect() }
        fn name(&self) -> String { "names".into() }
    }
    #[derive(Clone)]
    struct First;
    impl Rollout for First {
        fn pick<G: Game>(&mut self, _g: &G, _s: &G::State, _m: &[G::Move]) -> usize { 0 }
        fn name(&self) -> String { "first".into() }
    }

    #[test]
    fn a_named_move_is_played_when_a_rule_speaks_else_the_fallback() {
        let g = Count10;
        let mut p = RulesFirst::new(Names(3), First, 0.0, 1);
        for _ in 0..20 { assert_eq!(p.pick(&g, &0, &g.moves(&0)), 2); }
        assert_eq!((p.named, p.picks), (20, 20));
        let mut q = RulesFirst::new(Names(9), First, 0.0, 1);      // no move named: the fallback's pick, no rule counted
        for _ in 0..5 { assert_eq!(q.pick(&g, &0, &g.moves(&0)), 0); }
        assert_eq!((q.named, q.picks), (0, 5));
    }

    #[test]
    fn eps_one_is_uniform_and_the_stream_is_reproducible() {
        let g = Count10;
        let run = |eps: f64| { let mut p = RulesFirst::new(Names(3), First, eps, 7); (0..300).map(|_| p.pick(&g, &0, &g.moves(&0))).collect::<Vec<_>>() };
        let all = run(1.0);
        for i in 0..3 { assert!(all.contains(&i)); }
        assert_eq!(run(0.3), run(0.3));
    }
}
