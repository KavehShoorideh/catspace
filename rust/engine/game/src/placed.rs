//! PLAN 78.43 THE PLACED GAME: the one small interface the question sheet's light reading asks of a game beyond `Game` -- what stands where, what a unit bears on and
//! where it may move FROM ANY PLACE (a unit standing there or not), as the board stands, and where a move starts and ends. The game names its places and its kinds ONCE
//! (`place_names`, `kind_names`: text, each list in the order of the text) and every question after that is asked and answered by NUMBER: a place is its index in
//! `place_names`, a kind its index in `kind_names`, so the layer above never hashes or compares text and orders places and kinds as the text would. Nothing here knows a game:
//! a binding implements it beside its kernel (outside.rs, beside the rules ingestion).
//!
//! STAGE B adds what the obstacles, the frames and the route follower ask: the kind on a place, the legal step from one place to another and why there is none, what a move
//! plays and takes, whether a side bears on a place, and the two COUNTERFACTUAL BOARDS (a unit taken off, a unit put elsewhere) -- the board edits are the game's to make.
use crate::{Game, Role};

/// A place: its index in `Placed::place_names`.
pub type Place = u16;
/// A kind of unit: its index in `Placed::kind_names`.
pub type Kind = u8;

/// Why the unit on one place may not make one step to another: nothing of the side to move stands there, the place holds a unit it may not take, units stand on the line
/// between, or the unit may not move (anything else the rules say).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NoStep { UnitGone, PlaceTaken, LineShut, MayNotMove }

impl NoStep {
    pub fn word(self) -> &'static str { match self { NoStep::UnitGone => "unit_gone", NoStep::PlaceTaken => "place_taken", NoStep::LineShut => "line_shut", NoStep::MayNotMove => "may_not_move" } }
}

pub trait Placed: Game {
    /// The game's own names of its places, sorted by text; a place is an index into this list.
    fn place_names(&self) -> Vec<String>;
    /// The game's own words for its kinds of unit, sorted by text; a kind is an index into this list.
    fn kind_names(&self) -> Vec<String>;
    /// Every unit standing in `s` as (owner, kind, place), in the game's own order of places.
    fn cells(&self, s: &Self::State) -> Vec<(Role, Kind, Place)>;
    /// The places a unit of `owner` and `kind` standing on `at` would bear on, the board as it stands (a place held by its own side counts: it guards it), ascending.
    fn bearing_from(&self, s: &Self::State, owner: Role, kind: Kind, at: Place) -> Vec<Place>;
    /// The places a unit of `owner` and `kind` standing on `at` may move to in one move, the board as it stands and the side's own goal unit's safety not read, ascending.
    fn reach_from(&self, s: &Self::State, owner: Role, kind: Kind, at: Place) -> Vec<Place>;
    /// (the place a move starts from, the place it ends on), by the move's own name.
    fn move_ends(&self, s: &Self::State, m: &Self::Move) -> (Place, Place);
    /// The kind of the unit standing on `at` in `s`, None when none does.
    fn kind_at(&self, s: &Self::State, at: Place) -> Option<Kind>;
    /// The legal move of the side to move that takes the unit on `from` to `to` (the game's preferred one when several do), None when there is none.
    fn find_step(&self, s: &Self::State, from: Place, to: Place) -> Option<Self::Move>;
    /// Why the unit on `from` may not step to `to` in `s` (the side to move's), with the places involved, in the game's own order of places.
    fn why_not(&self, s: &Self::State, from: Place, to: Place) -> (NoStep, Vec<Place>);
    /// (the kind of the unit that plays `m`, the kind of the unit it takes if any).
    fn move_info(&self, s: &Self::State, m: &Self::Move) -> (Kind, Option<Kind>);
    /// Whether some unit of `owner` bears on `at` in `s` (a unit's own safety not read).
    fn strikes_place(&self, s: &Self::State, owner: Role, at: Place) -> bool;
    /// HYPOTHETICAL: `s` with the unit on `at` taken off the board (the side to move kept, what depended on the unit made consistent; a place holding nothing is left as it is).
    /// None when the game refuses the board as a position.
    fn without(&self, s: &Self::State, at: Place) -> Option<Self::State>;
    /// HYPOTHETICAL: `s` with the unit on `from` standing on `to` instead (a unit there is gone). None when nothing stands on `from` or the game refuses the board as a position.
    fn relocate(&self, s: &Self::State, from: Place, to: Place) -> Option<Self::State>;
}
