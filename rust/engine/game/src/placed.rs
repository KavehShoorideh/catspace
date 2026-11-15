//! PLAN 78.43 THE PLACED GAME: the one small interface the question sheet's light reading asks of a game beyond `Game` -- what stands where, what a unit bears on and
//! where it may move FROM ANY PLACE (a unit standing there or not), as the board stands, and where a move starts and ends. The game names its places and its kinds ONCE
//! (`place_names`, `kind_names`: text, each list in the order of the text) and every question after that is asked and answered by NUMBER: a place is its index in
//! `place_names`, a kind its index in `kind_names`, so the layer above never hashes or compares text and orders places and kinds as the text would. Nothing here knows a game:
//! a binding implements it beside its kernel (outside.rs, beside the rules ingestion).
//!
//! STAGE B adds what the obstacles, the frames and the route follower ask: the kind on a place, the legal step from one place to another and why there is none, what a move
//! plays and takes, whether a side bears on a place, and the two COUNTERFACTUAL BOARDS (a unit taken off, a unit put elsewhere) -- the board edits are the game's to make.
use crate::gdl_game::{render_term, row_args};
use crate::{Game, Role};
use cs_gdl::{Program, SortMember};

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

    /// PLAN 78.52 A. THE STATIC STEP RELATION OF THE RULES, by direction, as place numbers: asked of the game's description (its text), through the static relations and the
    /// objects (a name and the sorts it is made of) its graph vocabulary binds. Position-free: no state is read. See `read_steps`.
    fn steps(&self, description: &str, static_relations: &[String], objects: &[(String, Vec<String>)]) -> Result<Steps, String> {
        read_steps(description, static_relations, objects, &self.place_names())
    }
}

/// The rules' step relation as numbers: the directions (the rules' own words, sorted by text) and every row (direction, from place, to place), sorted.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Steps { pub directions: Vec<String>, pub rows: Vec<(u16, Place, Place)>, pub relation: String, pub object: String }

/// PLAN 78.52 A. Read the places and the steps by direction from a game description, with no name of any game written here. The BOUND NAMES come from outside (the game's graph
/// vocabulary): `static_relations` (the relations of the description that do not read the state) and `objects` (a name and the component sorts that make a place). A place is the
/// product of the component sorts of an object; the STEP RELATION is the one static relation whose signature is a direction sort followed by two places (the object's components
/// twice); it must be exactly one. The description's places are matched to `place_names` by the components joined, and the two sets must be equal, else an error: the engine's
/// numbering and the rules' board are the same board or this refuses.
pub fn read_steps(description: &str, static_relations: &[String], objects: &[(String, Vec<String>)], place_names: &[String]) -> Result<Steps, String> {
    let prog = Program::load(description).map_err(|e| e.0)?;
    let mut found: Vec<(String, String, Vec<String>)> = vec![];
    for r in static_relations {
        let Some(sig) = prog.signature(r) else { continue };
        for (name, comps) in objects {
            let n = comps.len();
            if sig.len() == 1 + 2 * n && sig[1..1 + n] == comps[..] && sig[1 + n..] == comps[..] && !comps.contains(&sig[0]) { found.push((r.clone(), name.clone(), comps.clone())); }
        }
    }
    let (relation, object, comps) = match found.len() { 1 => found.remove(0), 0 => return Err("no static relation is a step relation (a direction sort, then two places)".into()),
        n => return Err(format!("{n} static relations look like a step relation: {:?}", found.iter().map(|f| f.0.clone()).collect::<Vec<_>>())) };
    let members = |sort: &String| -> Result<Vec<String>, String> {
        Ok(prog.sorts.get(sort).ok_or_else(|| format!("the description has no sort {sort}"))?.iter().filter_map(|m| if let SortMember::Const(c) = m { Some(c.clone()) } else { None }).collect())
    };
    let mut product: Vec<String> = vec![String::new()];
    for c in &comps { let ms = members(c)?; product = product.iter().flat_map(|p| ms.iter().map(move |m| format!("{p}{m}"))).collect(); }
    let (mut a, mut b) = (product.clone(), place_names.to_vec());
    a.sort(); b.sort();
    if a != b { return Err(format!("the rules' places ({}) and the game's places ({}) are not the same set", a.len(), b.len())); }
    let index: std::collections::HashMap<&str, Place> = place_names.iter().enumerate().map(|(i, s)| (s.as_str(), i as Place)).collect();
    let rid = prog.rel_id(&relation).ok_or("the step relation is not in the description")?;
    let rel = prog.statics.get(&rid).ok_or_else(|| format!("{relation} is not a static relation"))?;
    let n = comps.len();
    let mut raw: Vec<(String, Place, Place)> = vec![];
    for row in rel.order.iter() {
        let args: Vec<String> = row_args(row).iter().map(|t| render_term(&prog, *t)).collect();
        if args.len() != 1 + 2 * n { return Err("a step row has the wrong width".into()); }
        let (f, t) = (args[1..1 + n].concat(), args[1 + n..].concat());
        raw.push((args[0].clone(), *index.get(f.as_str()).ok_or("a step starts on no place")?, *index.get(t.as_str()).ok_or("a step ends on no place")?));
    }
    let mut directions: Vec<String> = raw.iter().map(|r| r.0.clone()).collect();
    directions.sort(); directions.dedup();
    let mut rows: Vec<(u16, Place, Place)> = raw.into_iter().map(|(d, f, t)| (directions.iter().position(|x| *x == d).unwrap() as u16, f, t)).collect();
    rows.sort(); rows.dedup();
    Ok(Steps { directions, rows, relation, object })
}
