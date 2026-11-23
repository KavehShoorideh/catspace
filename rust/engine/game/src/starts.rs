//! PLAN 79.35 PART B: A GAME WHOSE START IS A PARAMETER (SYSTEM_SPEC 11 "Layer 2 says variants, with another start and with other rules, on the same layer 1"; ledger rows 41, 40; piece: rules).
//! Layer 1 holds no start; the start is layer 2's. A game with one start answers `start_count() == 1`; a game with many (the arrangement of its units is drawn from a numbered list) answers how many, and
//! `with_start(n)` is the same game begun from start number `n`: its `Game::initial` is that start, nothing else of it changes. Nothing here names an object of a game; a binding implements it beside its kernel (outside.rs),
//! and `rules::Ruled` delegates it (with the description's start turned to the same one by `rules::restart_delta`).
//! Called by: the certify stage (which plays the kernel and the description from several starts), the tests.
use crate::Game;

pub trait Starts: Game + Sized {
    /// How many starts the game has (1 when it has one).
    fn start_count(&self) -> usize;
    /// The same game begun from start number `n`; Err when `n >= start_count()`.
    fn with_start(&self, n: usize) -> Result<Self, String>;
}
