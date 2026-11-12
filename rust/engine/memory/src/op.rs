//! THE LEMMA OPERATOR (plan 78.14): binds a snapshot's lemmas to the position and proposes the moves they name -- the architecture document's lemma matcher and
//! response generator in one. It answers `Goal::Respond`. What it returns is a PRIOR: candidate move indices a search may read first. It never says a move is best
//! or a line is lost, so a wrong lemma costs reading time and nothing else. Every row the matcher examines and the seconds each phase takes are reported.
use crate::lemma::Lemma;
use crate::matcher::{bindings_fused, Binding, FactIndex};
use crate::cards::response_matches;
use crate::perspective;
use crate::snapshot::Snapshot;
use cs_game::gdl_game::FactView;
use cs_game::{Fact, Game};
use cs_operator::{Budget, Goal, Operator, Outcome, Status};
use std::collections::BTreeSet;

/// The cap on the bindings one lemma may produce at one position (a named fuse).
pub const MAX_BINDINGS: usize = 64;

pub struct LemmaOp<'a> {
    pub snapshot: &'a Snapshot,
    pub view: &'a FactView,
    /// Lemmas left out (the ablation: credit a lemma by removing it and measuring again).
    pub exclude: BTreeSet<String>,
}

impl<'a> LemmaOp<'a> {
    pub fn new(snapshot: &'a Snapshot, view: &'a FactView) -> Self { LemmaOp { snapshot, view, exclude: BTreeSet::new() } }

    pub fn lemmas(&self) -> impl Iterator<Item = &Lemma> { self.snapshot.lemmas.iter().filter(|l| !self.exclude.contains(&l.name)) }
}

/// The delta of a move on the delta relations: (removed, added).
pub fn delta(before: &[Fact], after: &[Fact], relations: &[String]) -> (Vec<Fact>, Vec<Fact>) {
    let keep = |f: &&Fact| relations.contains(&f.relation);
    let (b, a): (BTreeSet<&Fact>, BTreeSet<&Fact>) = (before.iter().filter(keep).collect(), after.iter().filter(keep).collect());
    (b.difference(&a).map(|f| (*f).clone()).collect(), a.difference(&b).map(|f| (*f).clone()).collect())
}

/// THE EFFECT OF A MOVE, read one way everywhere (the cards, the lemma operator, the prior, the graph). `relations` are the vocabulary's delta relations. When all of them
/// are state relations the effect is the delta of the BASE facts (cost and behaviour as ever); when at least one is a derived relation of the view's vocabulary it is the delta of
/// the FULL fact views (state + derived, `FactView::facts`) before and after, restricted to the delta relations. Not in a perspective: the caller applies `perspective` with the
/// mover BEFORE the move. A fact view failing after the move is an error, never an empty delta.
pub struct Effects<'a> {
    view: &'a FactView,
    relations: &'a [String],
    full: bool,
    before: Vec<Fact>,
}

impl<'a> Effects<'a> {
    /// The reader of the effects of moves from the position whose base facts are `base`.
    pub fn at(view: &'a FactView, relations: &'a [String], base: &[Fact]) -> Result<Effects<'a>, String> {
        let full = relations.iter().any(|r| view.vocabulary().derived.contains(r));
        let before = if full { view.facts(base)? } else { base.to_vec() };
        Ok(Effects { view, relations, full, before })
    }
    /// Whether the effect is read on the full fact views (a delta relation is derived).
    pub fn on_full_views(&self) -> bool { self.full }
    /// (removed, added) of the move that leads to the position whose base facts are `after_base`.
    pub fn of(&self, after_base: &[Fact]) -> Result<(Vec<Fact>, Vec<Fact>), String> {
        if self.full {
            let after = self.view.facts(after_base)?;
            Ok(delta(&self.before, &after, self.relations))
        } else {
            Ok(delta(&self.before, after_base, self.relations))
        }
    }
}

impl<'a, G: Game> Operator<G> for LemmaOp<'a> {
    fn name(&self) -> String { "lemma".into() }
    fn accepts(&self, goal: &Goal) -> bool { matches!(goal, Goal::Respond { .. }) }
    fn run(&self, g: &G, s: &G::State, goal: &Goal, _b: Budget) -> Outcome<G::Move> {
        let Goal::Respond { .. } = goal else { return Outcome::unsupported("lemma", goal) };
        let t0 = std::time::Instant::now();
        let mut o = Outcome::new(Status::NoLine);
        let base = g.base_facts(s);
        if base.is_empty() {
            o.evidence.push("lemma: the game gives no facts of this state".into());
            return o;
        }
        let roles = g.roles();
        let Some(mover) = g.mover(s) else { o.evidence.push("lemma: the game is over".into()); return o; };
        let facts = match self.view.facts(&base).map(|f| perspective(&f, &roles, mover)) {
            Ok(f) => f,
            Err(e) => { o.status = Status::Gap; o.evidence.push(format!("lemma: the fact view failed: {e}")); return o; }
        };
        o.part("facts", facts.len() as u64, t0.elapsed().as_secs_f64());
        let idx = FactIndex::new(&facts);
        let moves = g.moves(s);
        let mut deltas: Option<Vec<(FactIndex, FactIndex)>> = None;
        let mut steps_total = 0u64;
        let mut hit: BTreeSet<usize> = BTreeSet::new();
        let mut suggested: Vec<(String, BTreeSet<usize>)> = vec![];          // plan 78.36: the moves of SUGGESTION rules, apart from the candidates
        for lemma in self.lemmas() {
            let t1 = std::time::Instant::now();
            let mut steps = 0u64;
            let (bs, cut) = bindings_fused(&lemma.body, &Binding::new(), &idx, MAX_BINDINGS, &mut steps, crate::cards::STEP_FUSE);
            if cut { o.evidence.push(format!("lemma {}: the match was cut by the step fuse", lemma.name)); }
            if !bs.is_empty() {
                if deltas.is_none() {
                    let built = Effects::at(self.view, &self.snapshot.vocabulary.delta, &base).and_then(|ef| moves.iter().map(|m| {
                        let (r, a) = ef.of(&g.base_facts(&g.apply(s, m)))?;
                        Ok((FactIndex::new(&perspective(&r, &roles, mover)), FactIndex::new(&perspective(&a, &roles, mover))))
                    }).collect::<Result<Vec<_>, String>>());
                    match built {
                        Ok(d) => deltas = Some(d),
                        Err(e) => { o.status = Status::Gap; o.evidence.push(format!("lemma: the fact view failed on a move's effect: {e}")); return o; }
                    }
                }
                let ds = deltas.as_ref().unwrap();
                let mut mine: BTreeSet<usize> = BTreeSet::new();
                for (i, (r, a)) in ds.iter().enumerate() {
                    if bs.iter().any(|b| response_matches(lemma, b, r, a, &mut steps)) { mine.insert(i); }
                }
                o.evidence.push(format!("lemma {}: {} binding(s), {} move(s) {}", lemma.name, bs.len(), mine.len(), if lemma.suggest { "suggested (a suggestion: used only where a claim search classes the move a win)" } else { "proposed" }));
                o.facts.push(Fact { relation: "fired".into(), args: vec![lemma.name.clone(), bs.len().to_string(), mine.len().to_string()] });
                if lemma.suggest { suggested.push((lemma.name.clone(), mine)); } else { hit.extend(mine); }
            }
            o.part(format!("match {}", lemma.name), steps, t1.elapsed().as_secs_f64());
            steps_total += steps;
        }
        for (rule, moves) in &suggested { for i in moves { o.facts.push(Fact { relation: "suggested".into(), args: vec![rule.clone(), i.to_string()] }); } }
        o.candidates = hit.into_iter().collect();
        if !o.candidates.is_empty() { o.status = Status::Read; }
        o.cost.nodes = steps_total + facts.len() as u64;
        o.cost.seconds = t0.elapsed().as_secs_f64();
        o
    }
}
