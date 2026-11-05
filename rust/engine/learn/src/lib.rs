//! PLAN 78.4 THE LEARNING LAYER: knowledge of a GAME mined from INSTANCES of it -- the records of games played (watched or our own). Game-free: it reads only the
//! `Game` trait (cs_game). Two scopes of knowledge (Kaveh 2026-11-04): what is known about the GAME as a whole (its valuation, its lemmas: the store's entries
//! scoped to the game) and what is known about one INSTANCE of it (its record, what was learned while it was played); the miners here turn instances into game
//! knowledge.
//!   `Record`      one instance: the move names from the start, and the outcome (the goal per role).
//!   `watch`       instances by watching: an outside engine (any UCI engine) played against itself, seeded random openings, the game judged by the Game trait.
//!   `valuation`   THE VALUATION MINER (plan 76.4c's definition, its loops here): unit-kind counts fit to outcomes by a symmetric ordinal logistic, then the cheapest
//!                 integer worths that predict as well (one paired standard error, games as clusters) -- signed: a game whose goal is to shed units values them below 0.
pub mod valuation;
pub mod watch;

/// One instance of a game: where it started (an opaque descriptor the game's binding reads; empty: the game's own initial state), its moves from there (the game's
/// own names) and its outcome (the goal per role; empty when unfinished).
#[derive(Clone, Debug)]
pub struct Record { pub start: String, pub moves: Vec<String>, pub outcome: Vec<i64> }
