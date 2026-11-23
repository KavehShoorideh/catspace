//! PLAN 79.23 PART A, LAYER 2, THE RULES: ITS WORK, ITS INTERFACE (SYSTEM_SPEC 11 "the layers", "the order of work"; ledger row 36; piece: rules).
//!
//! THE WORK OF LAYER 2 (Kaveh 2026-11-20): one move a tempo; which moves are allowed and which are not; the position after a move; when the game is over and for whom. It writes what it knows onto layer 1's graph (the relations of stepping, the tags) and reads nothing above itself.
//! It sits on layer 1 (`cs_rep`: the fact, the board and its graph, the history, the two memories) and on the description parser; no crate above it is read.
//!
//! WHAT A GAME GIVES (the trait `Game`, lib.rs, and what this file adds), who calls it, what it takes and returns:
//!   `Game::initial / mover / moves / apply`      the start; the side to move (None where the game is over); the allowed moves (empty where it is over); the state after a move (a state, never an edit).   Called by the search, the wake, the tests.
//!   `Game::outcome`                              None while the game goes on, else the value of every side (0..100). The end the STATE decides: a side beaten, no move left, material that cannot give an end, the counter, a variant's own end.
//!   `Game::key`                                  the state's key: equal states, equal keys; two states that differ only in what no rule can ever read have one key (the game's own idea of "the same position" for a repeat).
//!   `Game::base_facts`                           the state as facts (every fluent of the description: the units, the tempo, the tags).   Called by layer 1's graph (units and tempo), by `TagKeeper` (tags).
//!   `Game::move_name / find_move`                the game's own name of a move and the move of a name.
//!   `Game::units / captured / taken_at / changed / pass`   what a move takes and changes; the free move.
//!   `tags::TagKeeper`                            what the rules read off the history (a right still held, a taking in passing, the counters, a repeat) as tags on layer 1's history, kept in make and undo.
//!   `graph::Static`, `keeper::GraphKeeper`       THE POSITION AFTER A MOVE AS A CHANGE TO LAYER 1'S GRAPH: `GraphKeeper::make(&Static, g, after)` writes the change (units gone and come, the tempo, the relations of stepping that the change touches) onto the kept graph, `undo` takes it back; `Static::position` the graph made afresh (the reference of the keeper).
//!   `placed::Placed`                             the board as the rules read it: what stands where, what bears on what, from any place, why a step is refused, the hypothetical boards.
//!   `certify::coplay`                            two readers of one game (the fast kernel and the description's evaluator) played side by side: moves, ends, facts, equal at every state.
//! WHAT THIS FILE ADDS (written first, then the bodies):
//!   `Rules: Game`                                `rules_key() -> GameKey` (the game's name and the content address of its own rules: what the long-term memory is partitioned by); `declaration() -> &Declaration` (where the state holds its units and its tempo, and which relations of the description are its board: what layer 1's graph is built from); `fact_decl()`; `repetition_limit() -> Option<usize>` (read from the description: how many earlier occurrences of a position end the game); `ends_by_record(repeats)` (the repeat end: layer 1's history says how many, the rules say whether that ends the game).
//!   `StateText: Game`                            the state written as the game's text form and read back: `state_text(&s) -> String`, `read_state_text(&str) -> Result<State, String>`. The writer keeps every fact of the state (see the decision below).
//!   `DescriptionNames: Game`                     `description_name(&native) -> String`: a move's name in the notation of the description (what `coplay` maps by).
//!   `Ruled<G>`                                 a game with its rules' text, address and declaration in hand: `Ruled::open(game, name, rules_root, declaration_path)`; it is a `Game`, a `Placed`, a `Rules` and a `StateText` whenever `G` is, by delegation (nothing of `G`'s own overrides is lost). `Ruled::board() -> Static` is layer 1's board of the game, read from its own description.
//! Who calls what is above: nothing above layer 2 may reach past this interface (callers that still do are listed in docs/ANALYST_PLAN.md, "79.23 BUILT").
//!
//! THE TEXT DOOR AND THE DROPPED FACT (decided here, plan 79.23). The description's state keeps the square a unit passed over on a two-step advance after EVERY such advance (the description says so in its own words: the taking there is legal only where a unit can take). The text form as the game's kernel writes it keeps that square only where a taking is legal, so it drops a fact of the description's state in the positions where no taking is legal (79.19 READ). By the rules both are right for what they are for: the KEY must not tell apart two positions that no rule can tell apart (a repeat counts positions with the same board, side, rights and a passing-square at which a taking is LEGAL: the game's key reads the legal one), and the TEXT must keep the state whole, because a state written and read back must have the facts and the moves it had. So the key reads the legal square and the text door writes the passed-over square always; both are tested against a reference that shares no code with the engine.
use crate::gdl_game::GdlGame;
use crate::graph::Static;
use crate::placed::{NoStep, Placed};
use crate::starts::Starts;
use crate::{Fact, Game, Role};
use cs_gdl::Program;
use cs_rep::graph::FactDecl;
use cs_rep::names::{Kind, Names, Place};
use cs_rep::store::GameKey;
use cs_rep::UnitDecl;
use cs_rep::serde_json::Value;
use std::path::Path;

/// WHAT THE GAME DECLARES OF ITSELF for layer 1: where its state holds the units and the tempo, and which relations of its description are its board (the static relations that say which place is next to which, and the objects, each a name and the sorts a place is made of).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Declaration { pub fact_decl: FactDecl, pub static_relations: Vec<String>, pub objects: Vec<(String, Vec<String>)> }

impl Declaration {
    /// The declaration as the game's declaration file writes it (the keys `fact_decl`, `static`, `objects`).
    pub fn from_json(text: &str) -> Result<Declaration, String> {
        let v: Value = cs_rep::serde_json::from_str(text).map_err(|e| e.to_string())?;
        let d = v.get("fact_decl").ok_or("the declaration has no fact_decl")?;
        let u = d.get("units").ok_or("fact_decl has no units")?;
        let units = UnitDecl {
            relation: u.get("relation").and_then(|x| x.as_str()).ok_or("units has no relation")?.to_string(),
            owner: u.get("owner").and_then(|x| x.as_u64()).ok_or("units has no owner")? as usize,
            kind: u.get("kind").and_then(|x| x.as_u64()).map(|k| k as usize),
        };
        let strings = |v: &Value| -> Vec<String> { v.as_array().map(|a| a.iter().filter_map(|x| x.as_str().map(|s| s.to_string())).collect()).unwrap_or_default() };
        let objects = v.get("objects").and_then(|o| o.as_object()).map(|o| o.iter().map(|(k, c)| (k.clone(), strings(c))).collect()).unwrap_or_default();
        Ok(Declaration {
            fact_decl: FactDecl { units, tempo: d.get("tempo").and_then(|x| x.as_str()).map(|s| s.to_string()) },
            static_relations: v.get("static").map(strings).unwrap_or_default(),
            objects,
        })
    }
}

/// THE RULES OF A GAME AS A LAYER ABOVE LAYER 1 SEES THEM (see the file's doc).
pub trait Rules: Game {
    /// The game's name and the content address of its rules' text.
    fn rules_key(&self) -> GameKey;
    fn declaration(&self) -> &Declaration;
    /// Where the state holds its units and its tempo.
    fn fact_decl(&self) -> FactDecl { self.declaration().fact_decl.clone() }
    /// How many earlier occurrences of a position (by key) end the game, read from the description (None where it has no such rule).
    fn repetition_limit(&self) -> Option<usize>;
    /// Whether a position that has `repeats` earlier occurrences in the record is over by the repeat rule. The record is layer 1's (`History::repeats`).
    fn ends_by_record(&self, repeats: usize) -> bool { self.repetition_limit().is_some_and(|n| repeats >= n) }
}

/// THE STATE AS THE GAME'S TEXT FORM, and back. The writer keeps every fact of the state; the reader gives a state equal to the one written in its facts, its moves, its end and its key.
pub trait StateText: Game {
    fn state_text(&self, s: &Self::State) -> String;
    fn read_state_text(&self, text: &str) -> Result<Self::State, String>;
}

/// THE SAME MOVE IN THE OTHER READER'S NOTATION: the game's own name of a move turned into the term the description plays it by (the notation `certify::coplay` compares across the two readers of one game).
pub trait DescriptionNames: Game {
    fn description_name(&self, name: &str) -> String;
}

/// A game with its rules in hand (the text of its description, the address of it, its declaration).
pub struct Ruled<G> { game: G, text: String, key: GameKey, decl: Declaration, limit: Option<usize> }

impl<G: Game> Ruled<G> {
    /// A game `game` named `name` whose rules lie under `rules_root`: the declaration file is read at `declaration_path` (relative to the root; its keys `spec` and `variants` say which description is the game's: the base file, and for a variant the delta composed onto it, `variants[name]`), the description is composed, its address made. One input says where the game's own declaration lies; the rest is read from it.
    pub fn open(game: G, name: &str, rules_root: &Path, declaration_path: &str) -> Result<Ruled<G>, String> {
        let read = |rel: &str| std::fs::read_to_string(rules_root.join(rel)).map_err(|e| format!("{rel}: {e}"));
        let decl_text = read(declaration_path)?;
        let v: Value = cs_rep::serde_json::from_str(&decl_text).map_err(|e| e.to_string())?;
        let spec = v.get("spec").and_then(|x| x.as_str()).ok_or("the declaration names no spec")?;
        let base = read(spec)?;
        let text = match v.get("variants").and_then(|m| m.get(name)) {
            Some(Value::String(delta)) => cs_gdl::compose(&base, &read(delta)?).map_err(|e| e.0)?,
            Some(Value::Null) => base,
            _ => return Err(format!("the declaration has no game named {name}")),
        };
        Ruled::new(game, name, &text, Declaration::from_json(&decl_text)?)
    }

    /// The same from a description text and a declaration already in hand.
    pub fn new(game: G, name: &str, text: &str, decl: Declaration) -> Result<Ruled<G>, String> { Ruled::with_key(game, GameKey::of(name, text), text, decl) }

    fn with_key(game: G, key: GameKey, text: &str, decl: Declaration) -> Result<Ruled<G>, String> {
        let p = Program::load(text).map_err(|e| e.0)?;
        let limit = p.rel_id("repetition_limit").and_then(|r| p.statics.get(&r).and_then(|rel| rel.order.first().map(|row| p.terms.name(row[0]).to_string()))).and_then(|n| n.parse().ok());
        Ok(Ruled { game, text: text.to_string(), key, decl, limit })
    }

    /// PLAN 79.35 PART B. A game by its name, the delta file of its rules (a path under `rules_root`; None for the base alone) and, when its start is not the description's, the facts of the start. The declaration file is read for its
    /// base, its units, its tempo and its board only (its `variants` list is not read, and no file of the rules' folder is written). The rules' address is the address of the base with the delta composed, WITHOUT the start: two starts of one game are one game.
    pub fn open_variant(game: G, name: &str, rules_root: &Path, declaration_path: &str, delta: Option<&str>, start: Option<&[Fact]>) -> Result<Ruled<G>, String> {
        let read = |rel: &str| std::fs::read_to_string(rules_root.join(rel)).map_err(|e| format!("{rel}: {e}"));
        let decl_text = read(declaration_path)?;
        let v: Value = cs_rep::serde_json::from_str(&decl_text).map_err(|e| e.to_string())?;
        let spec = v.get("spec").and_then(|x| x.as_str()).ok_or("the declaration names no spec")?;
        let base = read(spec)?;
        let rules = match delta { Some(d) => cs_gdl::compose(&base, &read(d)?).map_err(|e| e.0)?, None => base };
        let key = GameKey::of(name, &rules);
        let text = match start { Some(f) => cs_gdl::compose(&rules, &restart_delta(&rules, f)?).map_err(|e| e.0)?, None => rules };
        Ruled::with_key(game, key, &text, Declaration::from_json(&decl_text)?)
    }

    pub fn game(&self) -> &G { &self.game }
    /// The text of the description the game's rules are.
    pub fn description(&self) -> &str { &self.text }
}

impl<G: Game + Placed> Ruled<G> {
    /// Layer 1's board of the game, read once from the game's own description through its own declaration.
    pub fn board(&self) -> Result<Static, String> {
        let steps = self.game.steps(&self.text, &self.decl.static_relations, &self.decl.objects)?;
        let names = Names::new(self.game.place_names(), self.game.kind_names(), self.game.roles());
        Ok(Static::from_board(cs_rep::graph::Board::new(&steps, &names)?.with_decl(self.decl.fact_decl.clone())))
    }
}

impl<G: Game> Rules for Ruled<G> {
    fn rules_key(&self) -> GameKey { self.key.clone() }
    fn declaration(&self) -> &Declaration { &self.decl }
    fn repetition_limit(&self) -> Option<usize> { self.limit }
}

impl<G: StateText> StateText for Ruled<G> {
    fn state_text(&self, s: &G::State) -> String { self.game.state_text(s) }
    fn read_state_text(&self, text: &str) -> Result<G::State, String> { self.game.read_state_text(text) }
}

impl<G: DescriptionNames> DescriptionNames for Ruled<G> {
    fn description_name(&self, name: &str) -> String { self.game.description_name(name) }
}

impl<G: Game> Game for Ruled<G> {
    type State = G::State;
    type Move = G::Move;
    fn name(&self) -> String { self.game.name() }
    fn roles(&self) -> Vec<String> { self.game.roles() }
    fn initial(&self) -> G::State { self.game.initial() }
    fn mover(&self, s: &G::State) -> Option<Role> { self.game.mover(s) }
    fn moves(&self, s: &G::State) -> Vec<G::Move> { self.game.moves(s) }
    fn apply(&self, s: &G::State, m: &G::Move) -> G::State { self.game.apply(s, m) }
    fn outcome(&self, s: &G::State) -> Option<Vec<i64>> { self.game.outcome(s) }
    fn move_name(&self, s: &G::State, m: &G::Move) -> String { self.game.move_name(s, m) }
    fn key(&self, s: &G::State) -> u64 { self.game.key(s) }
    fn observe(&self, s: &G::State, role: Role) -> G::State { self.game.observe(s, role) }
    fn units(&self, s: &G::State) -> Vec<(Role, String)> { self.game.units(s) }
    fn pass(&self, s: &G::State) -> Option<G::State> { self.game.pass(s) }
    fn other(&self, r: Role) -> Role { self.game.other(r) }
    fn captured(&self, s: &G::State, m: &G::Move) -> Vec<(Role, String)> { self.game.captured(s, m) }
    fn taken_at(&self, s: &G::State, m: &G::Move) -> Vec<(Role, String, String)> { self.game.taken_at(s, m) }
    fn changed(&self, s: &G::State, m: &G::Move) -> Vec<(String, String)> { self.game.changed(s, m) }
    fn base_facts(&self, s: &G::State) -> Vec<Fact> { self.game.base_facts(s) }
    fn find_move(&self, s: &G::State, name: &str) -> Option<G::Move> { self.game.find_move(s, name) }
}

/// PLAN 79.35 PART B. THE DELTA THAT GIVES A DESCRIPTION ANOTHER START: the initial facts of the description `text` taken away (`(retract (init <fact>))`, one each, as the description's own forms read) and `start` given instead
/// (`(init <fact>)`, one each). Composed by `cs_gdl::compose` on `text`. The facts of a start are whatever the caller holds: the facts of a kernel's start, or of any state shown.
pub fn restart_delta(text: &str, start: &[Fact]) -> Result<String, String> {
    let term = |f: &Fact| if f.args.is_empty() { f.relation.clone() } else { format!("({} {})", f.relation, f.args.join(" ")) };
    let g = GdlGame::load("restart", text)?;
    let mut out = String::new();
    for f in g.base_facts(&g.initial()) { out.push_str(&format!("(retract (init {}))\n", term(&f))); }
    for f in start { out.push_str(&format!("(init {})\n", term(f))); }
    Ok(out)
}

/// A game with a start by number, kept with its rules: the same game from start `n`, its description given the start's facts as the kernel gives them (a game whose kernel gives no facts keeps its description).
impl<G: Starts> Starts for Ruled<G> {
    fn start_count(&self) -> usize { self.game.start_count() }
    fn with_start(&self, n: usize) -> Result<Ruled<G>, String> {
        let game = self.game.with_start(n)?;
        let facts = game.base_facts(&game.initial());
        let text = if facts.is_empty() { self.text.clone() } else { cs_gdl::compose(&self.text, &restart_delta(&self.text, &facts)?).map_err(|e| e.0)? };
        Ruled::with_key(game, self.key.clone(), &text, self.decl.clone())
    }
}

impl<G: Placed> Placed for Ruled<G> {
    fn place_names(&self) -> Vec<String> { self.game.place_names() }
    fn kind_names(&self) -> Vec<String> { self.game.kind_names() }
    fn cells(&self, s: &G::State) -> Vec<(Role, Kind, Place)> { self.game.cells(s) }
    fn bearing_from(&self, s: &G::State, owner: Role, kind: Kind, at: Place) -> Vec<Place> { self.game.bearing_from(s, owner, kind, at) }
    fn reach_from(&self, s: &G::State, owner: Role, kind: Kind, at: Place) -> Vec<Place> { self.game.reach_from(s, owner, kind, at) }
    fn move_ends(&self, s: &G::State, m: &G::Move) -> (Place, Place) { self.game.move_ends(s, m) }
    fn kind_at(&self, s: &G::State, at: Place) -> Option<Kind> { self.game.kind_at(s, at) }
    fn find_step(&self, s: &G::State, from: Place, to: Place) -> Option<G::Move> { self.game.find_step(s, from, to) }
    fn why_not(&self, s: &G::State, from: Place, to: Place) -> (NoStep, Vec<Place>) { self.game.why_not(s, from, to) }
    fn move_info(&self, s: &G::State, m: &G::Move) -> (Kind, Option<Kind>) { self.game.move_info(s, m) }
    fn strikes_place(&self, s: &G::State, owner: Role, at: Place) -> bool { self.game.strikes_place(s, owner, at) }
    fn without(&self, s: &G::State, at: Place) -> Option<G::State> { self.game.without(s, at) }
    fn relocate(&self, s: &G::State, from: Place, to: Place) -> Option<G::State> { self.game.relocate(s, from, to) }
}
