//! PLAN 78.43 THE PLACED GAME: the one small interface the question sheet's light reading asks of a game beyond `Game` -- what stands where, what a unit bears on and
//! where it may move FROM ANY PLACE (a unit standing there or not), as the board stands, and where a move starts and ends. Places are the game's own names for locations
//! (opaque text); kinds are the game's own words. Nothing here knows a game: a binding implements it beside its kernel (kernel.rs, beside the rules ingestion).
use crate::{Game, Role};

pub trait Placed: Game {
    /// Every unit standing in `s` as (owner, kind, place).
    fn cells(&self, s: &Self::State) -> Vec<(Role, String, String)>;
    /// The places a unit of `owner` and `kind` standing on `at` would bear on, the board as it stands (a place held by its own side counts: it guards it).
    fn bearing_from(&self, s: &Self::State, owner: Role, kind: &str, at: &str) -> Vec<String>;
    /// The places a unit of `owner` and `kind` standing on `at` may move to in one move, the board as it stands and the side's own goal unit's safety not read.
    fn reach_from(&self, s: &Self::State, owner: Role, kind: &str, at: &str) -> Vec<String>;
    /// (the place a move starts from, the place it ends on), by the move's own name.
    fn move_ends(&self, s: &Self::State, m: &Self::Move) -> (String, String);
}
