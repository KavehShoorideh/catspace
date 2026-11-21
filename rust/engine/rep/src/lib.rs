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
pub mod claim;
pub mod fact;
pub mod graph;
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
