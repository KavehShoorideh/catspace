//! PLAN 78.19 A: THE AUDIT of the fast fact layer against its oracle. Seeded random play of any `Game` visits positions; at each the fact view with demand
//! evaluation (`FactView::with_demand(true)`) must give the same facts as the view with full evaluation (the oracle). Counts and the time of each path are
//! reported; a disagreement is kept (the first few) so it can be read. Nothing here knows a game.
use crate::gdl_game::FactView;
use crate::rng::Rng;
use crate::{Fact, Game};
use std::time::Instant;

#[derive(Debug, Default, Clone)]
pub struct AuditReport {
    pub positions: usize,
    pub equal: usize,
    pub unequal: usize,
    pub ms_full: f64,
    pub ms_demand: f64,
    /// the first disagreements: (the base facts, the oracle's extra facts, the demand view's extra facts)
    pub examples: Vec<(Vec<Fact>, Vec<Fact>, Vec<Fact>)>,
}

impl AuditReport {
    pub fn ms_per_position_full(&self) -> f64 { if self.positions == 0 { 0.0 } else { self.ms_full / self.positions as f64 } }
    pub fn ms_per_position_demand(&self) -> f64 { if self.positions == 0 { 0.0 } else { self.ms_demand / self.positions as f64 } }
}

/// The positions of `games` seeded random games of at most `plies` plies each (the start included), as base facts.
pub fn positions<G: Game>(g: &G, games: usize, plies: usize, seed: u64) -> Vec<Vec<Fact>> {
    let mut rng = Rng::new(seed);
    let mut out = vec![];
    for _ in 0..games {
        let mut s = g.initial();
        for _ in 0..=plies {
            let b = g.base_facts(&s);
            if b.is_empty() { break; }
            out.push(b);
            let ms = g.moves(&s);
            if g.outcome(&s).is_some() || ms.is_empty() { break; }
            s = g.apply(&s, &ms[rng.below(ms.len())]);
        }
    }
    out
}

/// `full` and `demand` are views of the same description and vocabulary; the audit compares them over `positions`.
pub fn audit(full: &FactView, demand: &FactView, positions: &[Vec<Fact>]) -> Result<AuditReport, String> {
    let mut r = AuditReport::default();
    for base in positions {
        let t = Instant::now();
        let a = full.facts(base)?;
        r.ms_full += t.elapsed().as_secs_f64() * 1e3;
        let t = Instant::now();
        let b = demand.facts(base)?;
        r.ms_demand += t.elapsed().as_secs_f64() * 1e3;
        r.positions += 1;
        if a == b { r.equal += 1; } else {
            r.unequal += 1;
            if r.examples.len() < 3 {
                r.examples.push((base.clone(), a.iter().filter(|f| !b.contains(f)).cloned().collect(), b.iter().filter(|f| !a.contains(f)).cloned().collect()));
            }
        }
    }
    Ok(r)
}
