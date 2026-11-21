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
pub mod fact_audit;   // plan 78.19: the demand fact view against the full one, over seeded random play
pub mod gdl_game;
pub mod goal_target;   // plan 78.39: the goal's target unit, read from the description
pub mod graph;         // plan 79.23: the rules' writing onto layer 1's graph (how a unit steps), moved down from cs_tactics
pub mod keeper;        // plan 79.23: what a tempo changes in layer 1's graph, asked of the rules, moved down from cs_tactics
pub mod kernel;
pub mod rng;
pub mod rules;         // plan 79.23: layer 2's interface: the game's rules' key, declaration, repeat limit and text door, `Ruled`
pub mod tags;         // plan 79.21: layer 2 writes what the rules read off the history as tags (rights, counters, repeats), kept step by step in make and undo
pub mod outside;      // the kernel's placed reading, text door and description notation (rules ingestion); the outside engine's driver is cs_learn::outside (plan 79.23)
pub mod placed;       // plan 78.43: what stands where, what bears on what, from any place (the question sheet's light reading)

/// A role, by its index in `Game::roles`.
pub type Role = usize;

// the fact, the vocabulary and the unit's declaration live in the representation crate below the rules (plan 79.21); re-exported so every `cs_game::Fact` keeps resolving
pub use cs_rep::{Fact, UnitDecl, Vocabulary};

/// THE GAME (see the module docstring).
pub trait Game {
    type State: Clone;
    type Move: Clone + Eq + std::hash::Hash;          // (comparable and hashable: a search remembers moves -- killers, history)
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
    /// THE PASS (plan 78.6): the same state with the next role to move and nothing played -- a free move for the side that just moved (None where the game has no
    /// pass: a side that must answer cannot pass; a game without one says None).
    fn pass(&self, _s: &Self::State) -> Option<Self::State> { None }
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
    /// THE UNITS A MOVE TAKES, WITH THEIR PLACES (plan 78.7): (owner, kind name, the place it stood: the game's own name of a location; "" for a game that has none)
    /// -- what names a tactic's TARGET. By default the captured units with no place.
    fn taken_at(&self, s: &Self::State, m: &Self::Move) -> Vec<(Role, String, String)> {
        self.captured(s, m).into_iter().map(|(o, k)| (o, k, String::new())).collect()
    }
    /// THE MOVER'S OWN UNITS CHANGING KIND by a move (plan 78.5: a promotion -- a unit of one kind leaves, one of another arrives; a placement from outside the board
    /// arrives from nothing: from ""): [(from kind, to kind)]. By default read off the mover's units before and after.
    fn changed(&self, s: &Self::State, m: &Self::Move) -> Vec<(String, String)> where Self: Sized { units_changed(self, s, m) }
    /// THE STATE AS GROUND FACTS (plan 78.13): the game's own state relations, one fact per true fluent -- what a fact view derives its relations from, and what the
    /// co-play certificate compares. Empty for a game that gives none.
    fn base_facts(&self, _s: &Self::State) -> Vec<Fact> { vec![] }
    /// The legal move named `name` on `s` (the game's own notation), or None.
    fn find_move(&self, s: &Self::State, name: &str) -> Option<Self::Move> { self.moves(s).into_iter().find(|m| self.move_name(s, m) == name) }
}

/// The default reading of `Game::changed`: the mover's units before and after the move, the kinds that left paired with those that arrived (an arrival with no
/// departure arrives from nothing: "").
pub fn units_changed<G: Game>(g: &G, s: &G::State, m: &G::Move) -> Vec<(String, String)> {
    let Some(me) = g.mover(s) else { return vec![] };
    let mine = |st: &G::State| -> Vec<String> { g.units(st).into_iter().filter(|(o, _)| *o == me).map(|(_, k)| k).collect() };
    let (mut before, mut after) = (mine(s), mine(&g.apply(s, m)));
    before.retain(|k| if let Some(i) = after.iter().position(|x| x == k) { after.swap_remove(i); false } else { true });
    let mut out: Vec<(String, String)> = before.iter().zip(after.iter()).map(|(a, b)| (a.clone(), b.clone())).collect();
    for b in after.iter().skip(before.len()) { out.push((String::new(), b.clone())); }
    out
}
