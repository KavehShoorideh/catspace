//! PLAN 79.21 STAGE B/C, THE GRAPH KEPT STEP BY STEP (SYSTEM_SPEC 11 "one graph is the representation", "layer 1 holds the history"; Kaveh 2026-11-20, speed: "the graph of a position must be kept step by step in make and undo, never rebuilt from facts at each node"; piece:
//! representation). A `LiveGraph` is the graph of the position the game is at, changed by one tempo at a time: `make(Edit)` and `undo()`. An `Edit` is what one tempo changed: the tempo, the units removed and added, and the relations (bear, may_move, shut) that layer 2 wrote anew for the
//! units they changed for. Layer 1 knows no rule: which units' relations a tempo changes, and what they now are, is layer 2's to say (through the door of change); layer 1 only keeps them, in the canonical order, and can give the graph of the position at any moment (`part`),
//! exactly equal to the graph built from the units and the relations written afresh.
//!
//! INTERFACE.
//!   `LiveGraph::new(mover, units: Vec<LiveUnit>) -> LiveGraph`       the graph of a first position: units with their relations as layer 2 wrote them. Called by layer 2's keeper.
//!   `LiveGraph::make(&mut self, Edit)`                                one tempo: apply the edit and remember what it replaced.
//!   `LiveGraph::undo(&mut self) -> bool`                              take the last tempo back; false at the first position.
//!   `LiveGraph::key_of / units() / mover()`                           what stands, in the canonical order (owner, kind, place); the tempo.
//!   `LiveGraph::part(&self, &Board) -> PositionPart`                  the graph of the position as it now is (the nodes, ids and links are made here from what is kept).
use crate::graph::{Board, NodeType, PositionPart};
use crate::names::{Kind, Place, Role};

/// A unit as the graph keeps it: who, what, where, and the relations layer 2 wrote for it (each in the order layer 2 wrote them).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LiveUnit { pub owner: Role, pub kind: Kind, pub at: Place, pub bears: Vec<Place>, pub reach: Vec<Place>, pub shuts: Vec<Place> }

impl LiveUnit {
    pub fn key(&self) -> (Role, Kind, Place) { (self.owner, self.kind, self.at) }
}

/// What one tempo changed. `removed` and `added` are units by (owner, kind, place); `rewrite` replaces the relations of the units listed (an added unit must be listed, or it has none); `shuts` replaces the shut lists of the units listed, and with `clear_shuts` every other unit's shut list is emptied first
/// (the tempo moved, so the shut lines are those of the new side to move).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Edit {
    pub tempo: Option<Role>,
    pub removed: Vec<(Role, Kind, Place)>,
    pub added: Vec<(Role, Kind, Place)>,
    pub rewrite: Vec<((Role, Kind, Place), Vec<Place>, Vec<Place>)>,
    pub clear_shuts: bool,
    pub shuts: Vec<((Role, Kind, Place), Vec<Place>)>,
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

    /// One tempo. The log keeps the whole unit list before it (a clone of a few dozen small units: cheaper than the nodes and ids a rebuild makes, and exact).
    pub fn make(&mut self, e: Edit) {
        self.log.push(Undo { mover: self.mover, units: self.units.clone() });
        self.mover = e.tempo;
        if e.clear_shuts { for u in &mut self.units { u.shuts.clear(); } }
        for k in &e.removed { if let Ok(i) = self.units.binary_search_by_key(k, |u| u.key()) { self.units.remove(i); } }
        for k in &e.added {
            if let Err(i) = self.units.binary_search_by_key(k, |u| u.key()) { self.units.insert(i, LiveUnit { owner: k.0, kind: k.1, at: k.2, bears: vec![], reach: vec![], shuts: vec![] }); }
        }
        for (k, bears, moves) in e.rewrite {
            if let Ok(i) = self.units.binary_search_by_key(&k, |u| u.key()) { self.units[i].bears = bears; self.units[i].reach = moves; }
        }
        for (k, shuts) in e.shuts {
            if let Ok(i) = self.units.binary_search_by_key(&k, |u| u.key()) { self.units[i].shuts = shuts; }
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

    /// THE GRAPH OF THE POSITION AS IT NOW IS: units by `Board::position_from_units`, then every relation written through the door, in the order a fresh build writes them (per unit its bears then its moves; then the shut lists; then the lines through).
    pub fn part(&self, b: &Board) -> PositionPart {
        let mut p = b.position_from_units(self.mover, self.units.iter().map(|u| (u.owner, u.kind, u.at)).collect());
        for (ui, u) in self.units.iter().enumerate() {
            for &q in &u.bears { b.write(&mut p, NodeType::Bear, ui, q); }
            for &q in &u.reach { b.write(&mut p, NodeType::MayMove, ui, q); }
        }
        for (ui, u) in self.units.iter().enumerate() { for &q in &u.shuts { b.write(&mut p, NodeType::Shut, ui, q); } }
        b.derive_throughs(&mut p);
        p
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn u(o: usize, k: u8, at: u16) -> LiveUnit { LiveUnit { owner: o, kind: k, at, bears: vec![], reach: vec![], shuts: vec![] } }

    #[test]
    fn make_and_undo_return_the_same_units_and_tempo() {
        let mut g = LiveGraph::new(Some(0), vec![u(1, 0, 3), u(0, 0, 1)]);
        let before = (g.tempo(), g.units().to_vec());
        assert_eq!(g.units()[0].key(), (0, 0, 1));                       // canonical order
        g.make(Edit { tempo: Some(1), removed: vec![(0, 0, 1)], added: vec![(0, 0, 2)], rewrite: vec![((0, 0, 2), vec![3], vec![])], clear_shuts: true, shuts: vec![] });
        assert_eq!((g.tempo(), g.units().len(), g.units()[0].bears.clone()), (Some(1), 2, vec![3]));
        assert!(g.undo());
        assert_eq!((g.tempo(), g.units().to_vec()), before);
        assert!(!g.undo());
    }
}
