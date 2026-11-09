//! A LEMMA PRIOR for playouts (the `Prior` of cs_planning::rollout): at a state, the snapshot's lemmas whose body binds in the mover's perspective facts name the
//! moves they respond with; those moves get weight `boost`, every other move 1.0. It matches exactly as the lemma operator does (cs_memory::op::LemmaOp: the same
//! fact view, perspective, step fuse, binding cap, delta relations and `response_matches`). COST: every call rebuilds the facts and, if some lemma binds, the delta
//! of EVERY legal move (one `apply` and one fact reading each) -- far dearer than a uniform pick; acceptable for now, a cache by state key is the next step. A
//! state with no facts (a game that gives none), a failing fact view, or no bound lemma gives uniform weights.
use cs_game::gdl_game::FactView;
use cs_game::Game;
use cs_memory::cards::{response_matches, STEP_FUSE};
use cs_memory::matcher::{bindings_fused, Binding, FactIndex};
use cs_memory::op::{delta, MAX_BINDINGS};
use cs_memory::perspective;
use cs_memory::snapshot::Snapshot;
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
        for lemma in &self.snapshot.lemmas {
            let (bs, _cut) = bindings_fused(&lemma.body, &Binding::new(), &idx, MAX_BINDINGS, &mut steps, STEP_FUSE);
            if bs.is_empty() { continue; }
            let ds = deltas.get_or_insert_with(|| moves.iter().map(|m| {
                let after = g.base_facts(&g.apply(s, m));
                let (r, a) = delta(&base, &after, &self.snapshot.vocabulary.delta);
                (FactIndex::new(&perspective(&r, &roles, mover)), FactIndex::new(&perspective(&a, &roles, mover)))
            }).collect());
            for (i, (r, a)) in ds.iter().enumerate() {
                if out[i] != self.boost && bs.iter().any(|b| response_matches(lemma, b, r, a, &mut steps)) { out[i] = self.boost; }
            }
        }
        out
    }
    fn name(&self) -> String { format!("lemma(boost {})", self.boost) }
}
