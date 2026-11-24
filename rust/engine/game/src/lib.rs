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
//!
//! PLAN 79.34 PART B, WHAT LAYER 2 OWNS AFTER THE CUT (SYSTEM_SPEC 11 "The cut of layers 1 and 2 is carried into the code"; ledger rows 37, 36; piece: rules). WRITTEN BEFORE ANY BODY. Layer 2 says the rules and refers to the kinds and places by layer 1's names: how each kind moves, what a move changes, whose turn it is, where the
//! game starts, what ends it. The relations that FOLLOW FROM HOW THINGS MOVE (bear, may_move, shut, through, the turn), their making and their fact form are named, made and owned here, and are written onto layer 1's graph through its door (`cs_rep::graph::Board::write` with a `Rel`).
//!   `relations::{BEAR, MAY_MOVE, SHUT, THROUGH}: Rel`      slots 0 to 3 and the words bear, may_move, shut, through (layer 1 sees a slot and a word, never the meaning).
//!   `relations::PartRelations` (for `PositionPart`)        `bears() / moves() / shuts() / throughs() -> &Vec<RelRec>`, `struck_by(np, side) -> PlaceSet`; inline, one array index.
//!   `relations::UnitRelations` (for `LiveUnit`)            `bears() / reach() / shuts() -> &Vec<Place>`.
//!   `relations::node_words() -> Vec<&'static str>`          the words of the node types of a graph, layer 1's and the rules'.
//!   `graph::Static`                                        the board plus the rules' reading; it owns `cont` (the profile of the kinds learned from the bearing relations) with `learn(&mut self, &PositionPart)`, `continues(&self, Kind, line, forward) -> bool`, `derive_throughs(&self, &mut PositionPart)` (the line through a unit), and
//!                                                          `part_of(&self, &LiveGraph) -> PositionPart` (layer 1's bare graph with the relations written in the order a fresh build writes them).
//!   `graph_facts::FactReader`                              the rows of a graph as facts, as a pattern reads them (unit, bear, may_move, tempo, on_line, on_edge); layer 1 keeps only the words of what is seen (unit, on_line, on_edge).
//!   `Game::initial`                                        the state the game starts from (unchanged): layer 1 reads and holds no starting arrangement.
//!
//! PLAN 79.35 PART B, LAYER 2 SAYS FOUR MORE GAMES ON THE SAME LAYER 1 (SYSTEM_SPEC 11 "Layer 2 says variants, with another start and with other rules, on the same layer 1"; ledger rows 41, 40; pieces: rules, representation). WRITTEN BEFORE ANY BODY. Layer 1 (cs_rep) does not change.
//! THE START IS A PARAMETER OF A GAME (a game with 960 starts):
//!   `starts::Starts: Game`   `start_count(&self) -> usize` (1 for a game with one start; 960 for the game with a start by number) and `with_start(&self, n: usize) -> Result<Self, String>` (the same game begun from start `n`; Err when `n >= start_count`). `Game::initial` of the result is that start. Implemented beside the kernel (outside.rs), delegated by `Ruled`. Called by: the certify stage, the tests.
//! A DESCRIPTION GIVES ANOTHER START BY A DELTA, GENERATED FROM THE START'S FACTS (nothing of cs_gdl's evaluator, nor of gdl_game.rs, is touched):
//!   `rules::restart_delta(text: &str, start: &[Fact]) -> Result<String, String>`   the delta that takes the initial facts of the description `text` away (`(retract (init <fact>))`, one each) and gives `start` instead (`(init <fact>)`, one each): composed by `cs_gdl::compose` on `text`. A game that has a fixed other start (a delta file under rules/variants) says it in its own delta file; a game with a start by number gets it from here, the facts being the ones the kernel gives for that start.
//!   `rules::Ruled::open_variant(game, name, rules_root, declaration_path, delta: Option<&str>, start: Option<&[Fact]>) -> Result<Ruled<G>, String>`   like `Ruled::open`, except that the description is the declaration's base with the delta file (a path under the root) composed, and then the start's delta when a start is given; the declaration file is read only for its base, its units and its board, and is not changed (the files of rules/learned are not written).
//! WHAT `bear` AND `may_move` MEAN IN THESE GAMES (the smaller thing; the spec is silent): `bear(unit, place)` is the geometry of the unit's kind on the board as it stands (a place held by its own side counts: it guards it), whatever the rules do with a taking. `may_move(unit, place)` is the places the unit may step onto by its kind's step and the game's rule of what may be taken (a place of its own side never; a unit of the other side by the game's rule: the leader only in a game where a leader may be taken; in a game where the leader may not take, no place held by the other side), not read: whether the taking is compulsory, what a taking removes besides the unit taken, whether the leader is left attacked, the two-unit move of the leader, the taking in passing, the two-row step of the smallest kind. So a game's ALLOWED moves (`Game::moves`) are a subset of what `may_move` says, except the moves just listed. The test states it: for every position of every fixture, `bear` and `may_move` of every unit equal sets written by a program in the test that walks the board by hand (never by the rules), and every allowed move that is not one of the listed exceptions lies in `may_move`.
//! WHAT A GAME'S RULES OF ENDING ARE, in the same terms: `Game::outcome` is the end the state decides, as before (a side with no unit, a side with no allowed move, a leader removed, material that cannot give an end, the counter); the kernel's mode is the first reader, the description the second, the fixture the outside's.
pub mod certify;
pub mod fact_audit;   // plan 78.19: the demand fact view against the full one, over seeded random play
pub mod gdl_game;
pub mod goal_target;   // plan 78.39: the goal's target unit, read from the description
pub mod graph_facts;   // plan 79.34: the facts of a graph as a pattern reads them, moved up from layer 1 (the relations the rules make are named here)
pub mod graph;         // plan 79.23: the rules' writing onto layer 1's graph (how a unit steps), moved down from cs_tactics
pub mod keeper;        // plan 79.23: what a tempo changes in layer 1's graph, asked of the rules, moved down from cs_tactics
pub mod kernel;
pub mod relations;     // plan 79.34: layer 2 names and reads the relations that follow from how things move (bear, may_move, shut, through), written through layer 1's door
pub mod rng;
pub mod rules;         // plan 79.23: layer 2's interface: the game's rules' key, declaration, repeat limit and text door, `Ruled`
pub mod starts;        // plan 79.35: a game whose start is a parameter
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
    /// PLAN 79.39 (SYSTEM_SPEC 11; piece: rules; ledger row 46). THE SIDE TO MOVE AS THE STATE SAYS IT, with NO reading of the end of the game: the role whose turn it
    /// is; at an ended state, the role that would have moved. Takes a state, returns an index among `roles()` (any number of roles: two, four, two boards), never None; at a
    /// state not ended it equals `mover`. Called by every asker that only wants the side (the shared
    /// search, the layers above); an asker that needs to learn that the game ended keeps `mover` or `outcome`. The default keeps every game correct: it is `mover`,
    /// and at an end the first role, index 0 (the one fallback: no arithmetic on roles, no `other`, no assumption of two roles; an ended state's answer is not to be relied on by an asker, and a game that knows the side that would have moved overrides it) (a game whose end is cheap to read need not override it; a game that can read its turn off the state does, and pays nothing for the end).
    fn turn(&self, s: &Self::State) -> Role { self.mover(s).unwrap_or(0) }
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
