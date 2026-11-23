//! PLAN 79.21 STAGE A (SYSTEM_SPEC 11 "the layers", "one graph is the representation"; piece: representation). The three plain data types that every layer writes and reads, moved down from the crate of the rules (cs_game re-exports them, so no
//! user of `cs_game::Fact` changes). They call nothing of any game.

// plan 79.30: the fact moved down to layer 0 (cs_base::logic); the old path resolves here.
pub use cs_base::logic::Fact;

/// Where a game keeps its UNITS in the state: the fluent `relation`, the argument holding the owner (a role's name) and, if any, the one holding the kind; the other
/// arguments are the unit's place. A game's declaration of what a unit is -- nothing else in the layers names a unit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnitDecl { pub relation: String, pub owner: usize, pub kind: Option<usize> }

/// A game's FACT VOCABULARY (rules ingestion's second half): which derived relations of its description belong to the fact layer beside the state's own fluents,
/// which relations a move's effect is read in (`delta`), and how a state holds its units. The learned lemmas are written in this language.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Vocabulary { pub derived: Vec<String>, pub delta: Vec<String>, pub units: Option<UnitDecl> }

/// PLAN 79.34: THE WORDS OF THE FACTS OF WHAT IS SEEN IN ONE STATE (the rows a pattern reads over a graph, layer 2's reader makes them with its own beside): a unit stands on a place, a place lies on a line, a place is on the edge under a direction.
pub const UNIT: &str = "unit";
pub const ON_LINE: &str = "on_line";
pub const ON_EDGE: &str = "on_edge";
