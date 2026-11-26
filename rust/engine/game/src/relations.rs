//! PLAN 79.34 PART C, LAYER 2 OWNS THE RELATIONS THAT FOLLOW FROM HOW THINGS MOVE (SYSTEM_SPEC 11 "The cut of layers 1 and 2 is carried into the code"; ledger rows 37, 36; piece: rules).
//! Layer 1 (cs_rep) keeps a position's graph and a door through which a layer above writes relations of its own (`Board::write` with a `Rel`: a slot and a word), and knows no relation's name or meaning. THIS FILE NAMES THEM: a unit bears on a place (a place its own side holds is
//! guarded by it), it may move there in one move, a line before a place is shut by the units between, a unit is on a line through another (made by `graph::Static::derive_throughs`); and it reads them back off a position's graph and off a unit as kept.
//!
//! INTERFACE (written first).
//!   `BEAR, MAY_MOVE, SHUT, THROUGH: Rel`                  slots 0 to 3 of a position's graph, and the words of the ids of their nodes (`bear(<unit id>,<place id>)`, ...). The first three are also the slots (0, 1, 2) of a kept unit (`UnitRelations`); the last is made from them and is not kept.
//!   `PartRelations` for `PositionPart`                    `bears() / moves() / shuts() / throughs() -> &Vec<RelRec>`: the relations written, in the order written (inline: one index into an array, as cheap as the fields they replace); `struck_by(np, side) -> PlaceSet`: the places borne on by some unit of the side.
//!   `UnitRelations` for `LiveUnit`                        `bears() / reach() / shuts() -> &Vec<Place>`.
//!   `node_words() -> Vec<&'static str>`                   the words of every node type of a graph, layer 1's then these.
//! PLAN 79.58 PART B1 (SYSTEM_SPEC 11, blocks "A divergence found and named" and "The support filter"; LAYER 2, piece: rules; ledger rows 58, 57, 44): a FIFTH relation.
//!   `ALLOWED: Rel`                                         slot 4 of a position's graph, the word `allowed`: a unit of the side to act may be played to a place (the two ends of an allowed move, a promotion's kind not told). Written onto the graph by `keeper::AllowedKeeper::write` through the door; a graph that is not given the rows has none (slot 4 is empty and nothing of the graph differs from before).
//!   `PartRelations::allowed(&self) -> &Vec<RelRec>`        the rows written in slot 4. `node_words()` is not changed (it lists the four relations of the graphs made before this one).
//! Called by: the layers above that read the graph (the tactical reading, the wake, the question sheet) and by `graph::Static`, `keeper::GraphKeeper`, `graph_facts::FactReader` in this crate.
use cs_rep::graph::{NodeType, PositionPart, Rel, RelRec};
use cs_rep::live::LiveUnit;
use cs_rep::names::{Place, PlaceSet, Side};

pub const BEAR: Rel = Rel::new(0, "bear");
pub const MAY_MOVE: Rel = Rel::new(1, "may_move");
pub const SHUT: Rel = Rel::new(2, "shut");
pub const THROUGH: Rel = Rel::new(3, "through");
pub const ALLOWED: Rel = Rel::new(4, "allowed");

pub trait PartRelations {
    fn bears(&self) -> &Vec<RelRec>;
    fn moves(&self) -> &Vec<RelRec>;
    fn shuts(&self) -> &Vec<RelRec>;
    fn throughs(&self) -> &Vec<RelRec>;
    fn allowed(&self) -> &Vec<RelRec>;
    /// Places borne on by some unit of `side`.
    fn struck_by(&self, np: usize, side: Side) -> PlaceSet;
}

impl PartRelations for PositionPart {
    #[inline(always)]
    fn bears(&self) -> &Vec<RelRec> { &self.rels[BEAR.slot as usize] }
    #[inline(always)]
    fn moves(&self) -> &Vec<RelRec> { &self.rels[MAY_MOVE.slot as usize] }
    #[inline(always)]
    fn shuts(&self) -> &Vec<RelRec> { &self.rels[SHUT.slot as usize] }
    #[inline(always)]
    fn throughs(&self) -> &Vec<RelRec> { &self.rels[THROUGH.slot as usize] }
    #[inline(always)]
    fn allowed(&self) -> &Vec<RelRec> { &self.rels[ALLOWED.slot as usize] }
    fn struck_by(&self, np: usize, side: Side) -> PlaceSet {
        let mut out = PlaceSet::new(np);
        for r in self.bears() { if self.units[r.unit].side == side { out.insert(r.place); } }
        out
    }
}

pub trait UnitRelations {
    fn bears(&self) -> &Vec<Place>;
    fn reach(&self) -> &Vec<Place>;
    fn shuts(&self) -> &Vec<Place>;
}

impl UnitRelations for LiveUnit {
    #[inline(always)]
    fn bears(&self) -> &Vec<Place> { &self.slots[BEAR.slot as usize] }
    #[inline(always)]
    fn reach(&self) -> &Vec<Place> { &self.slots[MAY_MOVE.slot as usize] }
    #[inline(always)]
    fn shuts(&self) -> &Vec<Place> { &self.slots[SHUT.slot as usize] }
}

/// The words of every node type of a graph: layer 1's (place, direction, line, edge, kind, side, unit) then the relations of this layer.
pub fn node_words() -> Vec<&'static str> {
    [NodeType::Place, NodeType::Direction, NodeType::Line, NodeType::Edge, NodeType::Kind, NodeType::Side, NodeType::Unit, NodeType::Relation(BEAR), NodeType::Relation(MAY_MOVE), NodeType::Relation(SHUT), NodeType::Relation(THROUGH)]
        .iter().map(|t| t.word()).collect()
}
