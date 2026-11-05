//! PLAN 78.1 THE GAME LAYER (docs/ANALYST_PLAN.md 78): the one interface every layer above reads -- `Game` -- and its implementations.
//!
//! A game is its roles, its states, the side to move, the legal moves, the next state, the terminal outcome (a goal value per role, 0..100 as GDL's `goal`), a
//! state key and the names of its moves. `observe(state, role)` is what a role sees of a state: the state itself for a game of perfect information -- the
//! extension point of GDL-II's `sees` (an imperfect-information game answers a belief there). Nothing in this module knows a game.
//!
//! Implementations:
//!   `gdl_game::GdlGame`   any GDL text, read by the spec evaluator (cs_gdl): slow, and the oracle of every faster binding;
//!   `kernel::KernelGame`  shakmaty's positions for the perfect-information chess variants (chess, Chess960, crazyhouse, atomic, antichess, king of the hill,
//!                         three-check, racing kings, horde): fast, and the ONLY chess module of the layer stack -- rules ingestion, certified against its GDL
//!                         text where one is written (`certify`).
pub mod certify;
pub mod gdl_game;
pub mod kernel;
pub mod rng;
pub mod outside;      // the binding to outside engines (their protocol), beside the kernel

/// A role, by its index in `Game::roles`.
pub type Role = usize;

/// THE GAME (see the module docstring).
pub trait Game {
    type State: Clone;
    type Move: Clone;
    /// The game's name.
    fn name(&self) -> String;
    /// The roles, in the game's order.
    fn roles(&self) -> Vec<String>;
    /// The initial state.
    fn initial(&self) -> Self::State;
    /// The role to move; None at a terminal state.
    fn mover(&self, s: &Self::State) -> Option<Role>;
    /// The mover's legal moves (empty at a terminal state).
    fn moves(&self, s: &Self::State) -> Vec<Self::Move>;
    /// The state after the mover plays `m` (a legal move).
    fn apply(&self, s: &Self::State, m: &Self::Move) -> Self::State;
    /// At a terminal state, the goal value of every role (0..100); None while the game goes on.
    fn outcome(&self, s: &Self::State) -> Option<Vec<i64>>;
    /// The move's name (the game's own notation).
    fn move_name(&self, s: &Self::State, m: &Self::Move) -> String;
    /// A key of the state (equal states, equal keys).
    fn key(&self, s: &Self::State) -> u64;
    /// What `role` sees of the state: the state itself under perfect information (the GDL-II extension point).
    fn observe(&self, s: &Self::State, _role: Role) -> Self::State { s.clone() }
    /// The units standing in the state as (owner, kind name): what a valuation from the store counts (none for a game without units). The kind names are the
    /// game's own; the layers read them only through a valuation table.
    fn units(&self, _s: &Self::State) -> Vec<(Role, String)> { vec![] }
    /// The role that is not `r` in a game of two roles (the opponent).
    fn other(&self, r: Role) -> Role { 1 - r }
    /// THE UNITS A MOVE TAKES (plan 78.5): what the other sides lose by it, as (owner, kind name) -- by default the units that disappear from every side but the
    /// mover's (a side's own units change on its own move only by its rules, e.g. a promotion: that is no loss). A faster binding may override it.
    fn captured(&self, s: &Self::State, m: &Self::Move) -> Vec<(Role, String)> {
        let me = self.mover(s);
        let before = self.units(s);
        let after = self.units(&self.apply(s, m));
        let mut left: Vec<(Role, String)> = after.into_iter().filter(|(o, _)| Some(*o) != me).collect();
        let mut out = vec![];
        for u in before.into_iter().filter(|(o, _)| Some(*o) != me) {
            if let Some(i) = left.iter().position(|x| *x == u) { left.swap_remove(i); } else { out.push(u); }
        }
        out
    }
    /// The legal move named `name` on `s` (the game's own notation), or None.
    fn find_move(&self, s: &Self::State, name: &str) -> Option<Self::Move> { self.moves(s).into_iter().find(|m| self.move_name(s, m) == name) }
}
