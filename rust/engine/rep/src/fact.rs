//! PLAN 79.21 STAGE A (SYSTEM_SPEC 11 "the layers", "one graph is the representation"; piece: representation). The three plain data types that every layer writes and reads, moved down from the crate of the rules (cs_game re-exports them, so no
//! user of `cs_game::Fact` changes). They call nothing of any game.

/// A ground fact as text: a relation name and its arguments (the store's language; the layers know no game).
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Fact { pub relation: String, pub args: Vec<String> }

impl Fact {
    pub fn new(relation: &str, args: &[&str]) -> Fact { Fact { relation: relation.to_string(), args: args.iter().map(|a| a.to_string()).collect() } }
}

impl std::fmt::Display for Fact {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.args.is_empty() { write!(f, "{}", self.relation) } else { write!(f, "({} {})", self.relation, self.args.join(" ")) }
    }
}

/// Where a game keeps its UNITS in the state: the fluent `relation`, the argument holding the owner (a role's name) and, if any, the one holding the kind; the other
/// arguments are the unit's place. A game's declaration of what a unit is -- nothing else in the layers names a unit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnitDecl { pub relation: String, pub owner: usize, pub kind: Option<usize> }

/// A game's FACT VOCABULARY (rules ingestion's second half): which derived relations of its description belong to the fact layer beside the state's own fluents,
/// which relations a move's effect is read in (`delta`), and how a state holds its units. The learned lemmas are written in this language.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Vocabulary { pub derived: Vec<String>, pub delta: Vec<String>, pub units: Option<UnitDecl> }
