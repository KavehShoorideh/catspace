//! PLAN 79.21 LAYER 1, THE REPRESENTATION, IN A CRATE BELOW THE RULES (SYSTEM_SPEC 11 "the layers", "what the representation layer must be able to say", "two memories", "one graph is the representation", "layer 1 holds the history"; ledger row 36;
//! piece: representation). This crate depends on NO crate that holds the rules (not cs_game); of the description language it uses only cs_gdl's parser, to read a game's BOARD once. It names no object of any game.
//!
//! STAGE A (what moved here from cs_game and cs_memory; every old path still resolves through a re-export):
//!   `fact`      the fact (`Fact`), the game's fact vocabulary (`Vocabulary`) and how a state holds its units (`UnitDecl`).
//!   `sha`       the content address.
//!   `lemma`     the lemma language (restricted Datalog over the fact vocabulary) and its JSON.
//!   `matcher`   the one join that binds a pattern to facts.
//!   `statement` the one form of statement with its standing, holder and what it read; `Change`, the unit of change.
//!   `store`     the two memories (the long-term store partitioned by game, the memory of the game at hand) and the four changes.
//!   `snapshot`  the immutable, content-addressed state of the store.
//!   `claim`     the claim rule as data (the reader that binds it to a game stays in cs_memory).
//! What applies lemmas to a game (the lemma operator, the claim reader) stays above, in cs_memory, which depends on this crate.
//!
//! STAGE B and C are documented where they are written: `board` (the board read once, the graph of a position from the board and facts) and `history` (the game at hand as a sequence of changes).
//!
//! PLAN 79.34 PART B, THE INTERFACE AFTER THE CUT OF LAYERS 1 AND 2 (SYSTEM_SPEC 11 "The cut of layers 1 and 2 is carried into the code"; ledger rows 37, 36; piece: representation). WRITTEN BEFORE ANY BODY; the bodies follow in the next commits.
//! Layer 1 says what is seen in one state with no knowledge of play: the kinds and their names, the board (places, plain links: next, on a line, on an edge), the TWO CLASSES of places, the units with kind and side and where they stand, any state it is shown, and the history. It
//! names NO relation that follows from how things move, and it says nothing of which state a game starts from (the start is layer 2's). It keeps one DOOR by which a layer above writes relations of its own onto a position's graph without layer 1 knowing their names or meaning.
//! THE DOOR (static dispatch: a relation is a VALUE, no boxed function, no map or text compare per node).
//!   `graph::REL_SLOTS: usize = 4`                          how many kinds of relation a layer above may write onto one position's graph.
//!   `graph::Rel { slot: u8, word: &'static str }`          a relation as the layer above names it: its slot and the word the ids of its nodes begin with; `Rel::new(slot, word)` is const.
//!   `graph::NodeType::{Place, Direction, Line, Edge, Kind, Side, Unit, Relation(Rel)}`   no relation of the rules is a node type of layer 1.
//!   `Board::write(&self, part: &mut PositionPart, rel: Rel, unit: usize, to: Place)`   writes one relation of a unit: the node `<word>(<unit id>,<place id>)` and its links (by, at_place, at_unit, along, between).
//!   `PositionPart { mover, nodes, links, units, rels: [Vec<RelRec>; REL_SLOTS], occupant }` and `PositionPart::written(&self, Rel) -> &Vec<RelRec>`, `relation_count()`, `count(owner, kind) -> usize` (how many units of a kind a side has in the state).
//!   `live::LIVE_SLOTS: usize = 3`, `live::LiveUnit { owner, kind, at, slots: [Vec<Place>; LIVE_SLOTS] }`: the places a layer above wrote for a unit, kept in slots that layer 1 does not name.
//!   `live::Edit { tempo, removed, added, rewrite: Vec<(key, [Vec<Place>; 2])>, clear_last: bool, last: Vec<(key, Vec<Place>)> }`   what one tempo changed; the last slot is the one a layer above keeps for the side that is marked.
//!   `LiveGraph::bare_part(&self, &Board) -> PositionPart`   the graph of the position as it now is, the units only; the layer above writes its relations onto it (this replaces `LiveGraph::part`).
//! WHAT IS NEW SEEN: `Board::classes(&self) -> Option<Classes>` with `Classes { class: Vec<u8>, flips: Vec<bool> }` (per place its class, 0 or 1; per direction whether a step along it changes the class), derived from the board's own links with no game named;
//! `Board::state_text(&self, &PositionPart)` and `Board::part_from_text(&self, marked, &[(side, kind, place)])` write a state as names and read it back.
//! WHAT LEAVES (to layer 2, `cs_game::relations`, `cs_game::graph::Static`, `cs_game::graph_facts`): the names of the relations the rules make and their node types, the readers of them on a position and on a unit, the line through a unit and the profile of the kinds learned from the relations, the
//! rows of the relations as facts (`graph_facts`), the making of a position's graph from the live graph. HOW THE TURN IS HELD: layer 1 carries the one side a layer above marked as to act (`mover`, `Edit::tempo`, `FactDecl::tempo` as data the game declares) because a unit's side as seen from it and the declared fact need it; layer 2 names and makes it (the
//! fact `tempo(owner)` is layer 2's).
pub mod claim;
pub mod fact;
pub mod graph;
pub mod graph_facts;
pub mod history;
pub mod lemma;
pub mod live;
pub mod matcher;
pub mod names;
pub mod sha;
pub mod snapshot;
pub mod statement;
pub mod steps;
pub mod store;

pub use fact::{Fact, UnitDecl, Vocabulary};
// plan 79.23: the one JSON of the layers above (a game's declaration file is read with it; no crate adds the dependency of its own)
pub use serde_json;
