//! PLAN 78.36 THE CLAIM RULE: a second kind of rule, a statement about a POSITION rather than about a move. "IF body THEN the position's class for the side to move is win /
//! draw / loss." The body is written in the mover's perspective (`us` / `them`) exactly as a lemma's is, and is bound by the same matcher over the same fact view; what
//! differs is the head: a lemma (this crate's `lemma`) names the effect of a move, a claim names a class. A rule carries EITHER a `response` or a `claim`, never both.
//!
//! JSON: {"name", "body": [lit...], "claim": {"class": "win" | "draw" | "loss"}, "stats": {...}, "meta": {...}}.
//!
//! `Claims` is the READER the decision's search uses at its leaves: `class_at(g, s)` binds every claim of a snapshot on the position's facts (the mover's perspective, one binding
//! suffices) and answers the class and the rule that spoke; claims that disagree make no answer (and count a conflict). It counts what it evaluated and how long it took.
//! A claim enters a store only as a rule admitted by the training side's drill (like a response rule); nothing of that drill is written into the rule.
use crate::matcher::{bindings_fused, Binding, FactIndex};
use crate::perspective;
use crate::snapshot::Snapshot;
use cs_game::gdl_game::FactView;
use cs_game::Game;
use std::cell::Cell;
use std::collections::BTreeSet;

pub use cs_rep::claim::{Class, Claim, Rule};

/// The rows one claim's matching may examine at one position (the same named fuse the lemma operator uses).
pub use crate::cards::STEP_FUSE;

/// THE CLAIM READER: the snapshot's claims bound to a position through the fact view.
pub struct Claims<'a> {
    pub snapshot: &'a Snapshot,
    pub view: &'a FactView,
    /// Claims left out (the ablation).
    pub exclude: BTreeSet<String>,
    evaluations: Cell<u64>,
    spoke: Cell<u64>,
    conflicts: Cell<u64>,
    seconds: Cell<f64>,
}

/// What the reader has spent.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ClaimStats { pub evaluations: u64, pub spoke: u64, pub conflicts: u64, pub seconds: f64 }

impl<'a> Claims<'a> {
    pub fn new(snapshot: &'a Snapshot, view: &'a FactView) -> Claims<'a> {
        Claims { snapshot, view, exclude: BTreeSet::new(), evaluations: Cell::new(0), spoke: Cell::new(0), conflicts: Cell::new(0), seconds: Cell::new(0.0) }
    }
    /// The claims in use.
    pub fn rules(&self) -> impl Iterator<Item = &Claim> { self.snapshot.claims.iter().filter(|c| !self.exclude.contains(&c.name)) }
    pub fn is_empty(&self) -> bool { self.rules().next().is_none() }
    pub fn stats(&self) -> ClaimStats { ClaimStats { evaluations: self.evaluations.get(), spoke: self.spoke.get(), conflicts: self.conflicts.get(), seconds: self.seconds.get() } }

    /// The class of the position for its side to move, and the rule that spoke (the first of those that agree): None where no claim binds, where claims disagree (a conflict is
    /// counted), where the game gives no facts, or at a terminal state.
    pub fn class_at<G: Game>(&self, g: &G, s: &G::State) -> Option<(Class, String)> {
        let t0 = std::time::Instant::now();
        self.evaluations.set(self.evaluations.get() + 1);
        let out = self.read(g, s);
        self.seconds.set(self.seconds.get() + t0.elapsed().as_secs_f64());
        if out.is_some() { self.spoke.set(self.spoke.get() + 1); }
        out
    }

    fn read<G: Game>(&self, g: &G, s: &G::State) -> Option<(Class, String)> {
        let mover = g.mover(s)?;
        let base = g.base_facts(s);
        if base.is_empty() { return None; }
        let roles = g.roles();
        let facts = self.view.facts(&base).ok().map(|f| perspective(&f, &roles, mover))?;
        let idx = FactIndex::new(&facts);
        let mut said: Option<(Class, String)> = None;
        for c in self.rules() {
            let mut steps = 0u64;
            let (bs, _cut) = bindings_fused(&c.body, &Binding::new(), &idx, 1, &mut steps, STEP_FUSE);
            if bs.is_empty() { continue; }
            match &said {
                None => said = Some((c.class, c.name.clone())),
                Some((k, _)) if *k == c.class => {}
                Some(_) => { self.conflicts.set(self.conflicts.get() + 1); return None; }
            }
        }
        said
    }
}

