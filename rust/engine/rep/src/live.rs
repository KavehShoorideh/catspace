//! PLAN 79.21 STAGE B/C, THE GRAPH KEPT STEP BY STEP (SYSTEM_SPEC 11 "one graph is the representation", "layer 1 holds the history"; Kaveh 2026-11-20, speed: "the graph of a position must be kept step by step in make and undo, never rebuilt from facts at each node"; piece:
//! representation). A `LiveGraph` is the graph of the position the game is at, changed by one tempo at a time: `make(Edit)` and `undo()`. An `Edit` is what one tempo changed: the tempo, the units removed and added, and the places a layer above wrote anew, in slots, for the
//! units they changed for. Layer 1 knows no rule and names no relation: which units' relations a tempo changes, and what they now are, is layer 2's to say; layer 1 only keeps them, in the canonical order, and can give the bare graph of the position at any moment (`bare_part`);
//! layer 2 writes its relations onto it through the door (`cs_game::graph::Static::part_of`), exactly equal to the graph built from the units and the relations written afresh (plan 79.34).
//!
//! INTERFACE.
//!   `LiveGraph::new(mover, units: Vec<LiveUnit>) -> LiveGraph`       the graph of a first position: units with the places layer 2 wrote for them, in slots. Called by layer 2's keeper.
//!   `LiveGraph::make(&mut self, Edit)`                                one tempo: apply the edit and remember what it replaced.
//!   `LiveGraph::undo(&mut self) -> bool`                              take the last tempo back; false at the first position.
//!   `LiveGraph::key_of / units() / mover()`                           what stands, in the canonical order (owner, kind, place); the tempo.
//!   `LiveGraph::bare_part(&self, &Board) -> PositionPart`             the graph of the position as it now is, the units only (the nodes, ids and links are made here from what is kept).
use crate::graph::{Board, PositionPart};
use crate::names::{Kind, Place, Role};

/// How many lists of places a layer above keeps for one unit.
pub const LIVE_SLOTS: usize = 3;

/// A unit as the graph keeps it: who, what, where, and the places layer 2 wrote for it, in slots (each in the order layer 2 wrote them). Layer 1 does not say what a slot holds.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LiveUnit { pub owner: Role, pub kind: Kind, pub at: Place, pub slots: [Vec<Place>; LIVE_SLOTS] }

impl LiveUnit {
    pub fn new(owner: Role, kind: Kind, at: Place) -> LiveUnit { LiveUnit { owner, kind, at, slots: Default::default() } }
    pub fn key(&self) -> (Role, Kind, Place) { (self.owner, self.kind, self.at) }
}

/// What one tempo changed. `removed` and `added` are units by (owner, kind, place); `rewrite` replaces the first two slots of the units listed (an added unit must be listed, or it has none); `last` replaces the last slot of the units listed, and with `clear_last` every other unit's last slot is emptied first
/// (the slot a layer above keeps for the side marked as to act, which changes with the tempo).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Edit {
    pub tempo: Option<Role>,
    pub removed: Vec<(Role, Kind, Place)>,
    pub added: Vec<(Role, Kind, Place)>,
    pub rewrite: Vec<((Role, Kind, Place), [Vec<Place>; 2])>,
    pub clear_last: bool,
    pub last: Vec<((Role, Kind, Place), Vec<Place>)>,
}

#[derive(Debug, Clone)]
struct Undo { mover: Option<Role>, units: Vec<LiveUnit> }

#[derive(Debug, Clone)]
pub struct LiveGraph { mover: Option<Role>, units: Vec<LiveUnit>, log: Vec<Undo> }

impl LiveGraph {
    pub fn new(mover: Option<Role>, mut units: Vec<LiveUnit>) -> LiveGraph {
        units.sort_by_key(|u| u.key());
        LiveGraph { mover, units, log: vec![] }
    }
    pub fn tempo(&self) -> Option<Role> { self.mover }
    pub fn units(&self) -> &[LiveUnit] { &self.units }
    pub fn depth(&self) -> usize { self.log.len() }
    /// PLAN 79.22: the graph is kept between decisions: what it could be taken back to is let go (a game's graph carries no log of the whole game).
    pub fn settle(&mut self) { self.log.clear(); }

    /// One tempo. The log keeps the whole unit list before it (a clone of a few dozen small units: cheaper than the nodes and ids a rebuild makes, and exact).
    pub fn make(&mut self, e: Edit) {
        self.log.push(Undo { mover: self.mover, units: self.units.clone() });
        self.mover = e.tempo;
        if e.clear_last { for u in &mut self.units { u.slots[LIVE_SLOTS - 1].clear(); } }
        for k in &e.removed { if let Ok(i) = self.units.binary_search_by_key(k, |u| u.key()) { self.units.remove(i); } }
        for k in &e.added {
            if let Err(i) = self.units.binary_search_by_key(k, |u| u.key()) { self.units.insert(i, LiveUnit::new(k.0, k.1, k.2)); }
        }
        for (k, [first, second]) in e.rewrite {
            if let Ok(i) = self.units.binary_search_by_key(&k, |u| u.key()) { self.units[i].slots[0] = first; self.units[i].slots[1] = second; }
        }
        for (k, last) in e.last {
            if let Ok(i) = self.units.binary_search_by_key(&k, |u| u.key()) { self.units[i].slots[LIVE_SLOTS - 1] = last; }
        }
    }

    pub fn undo(&mut self) -> bool {
        match self.log.pop() { Some(u) => { self.mover = u.mover; self.units = u.units; true } None => false }
    }

    /// Who stands where, as `PositionPart::occupant` says it: place -> index in `units()` + 1 (0: nothing stands there). For a question asked of the graph as it is, with no node made.
    pub fn occupant(&self, b: &Board) -> Vec<u16> {
        let mut o = vec![0u16; b.names.places.len()];
        for (i, u) in self.units.iter().enumerate() { o[u.at as usize] = i as u16 + 1; }
        o
    }

    /// The units (indices into `units()`) standing strictly between two places on the lines they share, in the order a relation node lists them (line by line).
    pub fn between(&self, b: &Board, occupant: &[u16], from: Place, to: Place) -> Vec<usize> {
        let mut out = vec![];
        for l in b.common_lines(from, to) { for q in b.strictly_between(l, from, to) { let o = occupant[q as usize]; if o != 0 { out.push(o as usize - 1); } } }
        out
    }

    /// THE GRAPH OF THE POSITION AS IT NOW IS, THE UNITS ONLY: by `Board::position_from_units`. The layer above writes its relations onto it through the door, from the slots (`units()`).
    pub fn bare_part(&self, b: &Board) -> PositionPart {
        b.position_from_units(self.mover, self.units.iter().map(|u| (u.owner, u.kind, u.at)).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn u(o: usize, k: u8, at: u16) -> LiveUnit { LiveUnit::new(o, k, at) }

    #[test]
    fn make_and_undo_return_the_same_units_and_tempo() {
        let mut g = LiveGraph::new(Some(0), vec![u(1, 0, 3), u(0, 0, 1)]);
        let before = (g.tempo(), g.units().to_vec());
        assert_eq!(g.units()[0].key(), (0, 0, 1));                       // canonical order
        g.make(Edit { tempo: Some(1), removed: vec![(0, 0, 1)], added: vec![(0, 0, 2)], rewrite: vec![((0, 0, 2), [vec![3], vec![]])], clear_last: true, last: vec![] });
        assert_eq!((g.tempo(), g.units().len(), g.units()[0].slots[0].clone()), (Some(1), 2, vec![3]));
        assert!(g.undo());
        assert_eq!((g.tempo(), g.units().to_vec()), before);
        assert!(!g.undo());
    }
}
