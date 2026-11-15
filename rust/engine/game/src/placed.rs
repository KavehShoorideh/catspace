//! PLAN 78.43 THE PLACED GAME: the one small interface the question sheet's light reading asks of a game beyond `Game` -- what stands where, what a unit bears on and
//! where it may move FROM ANY PLACE (a unit standing there or not), as the board stands, and where a move starts and ends. The game names its places and its kinds ONCE
//! (`place_names`, `kind_names`: text, each list in the order of the text) and every question after that is asked and answered by NUMBER: a place is its index in
//! `place_names`, a kind its index in `kind_names`, so the layer above never hashes or compares text and orders places and kinds as the text would. Nothing here knows a game:
//! a binding implements it beside its kernel (outside.rs, beside the rules ingestion).
use crate::{Game, Role};

/// A place: its index in `Placed::place_names`.
pub type Place = u16;
/// A kind of unit: its index in `Placed::kind_names`.
pub type Kind = u8;

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
}
