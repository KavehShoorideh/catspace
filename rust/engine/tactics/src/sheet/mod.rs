//! PLAN 78.43 STAGE A: THE QUESTION SHEET'S LIGHT READING AND ITS LEVELS 0 AND 1 BY GEOMETRY, in the engine, game-free.
//!
//! The Python sheet (catspace/incubator/analyst/discovery/question_sheet.py) is the readable mirror and the ORACLE of this module. A position is read once (`reading`): its units,
//! what each bears on and where each may move as the board stands (asked of the game through `cs_game::placed::Placed`, from ANY place), the worth of each kind (a table
//! supplied from outside) and the goal's kind. The MEDIUM OF MOVEMENT (`medium`: the one-move hops on the empty board, the places passed over, which kinds take only) is a data
//! table supplied from outside, read by the caller from the game's description. `goal_report` is the goal unit's report (holes, covered places, supply, scenarios) made from
//! the light reading; `levels` holds level 0 (the board as it stands) and level 1 (the asking side's moves alone: the goal unit by the report, any other unit by the medium
//! of movement, the routes of each way with the obstacles on them as the board stands). No object of any game is named here.
//!
//! STAGE B0: places and kinds are NUMBERS (`Place`, `Kind`: the game's own lists of names, in the order of the text), sets of places are bit sets; the text is made only at the
//! boundary (`Names`). Units, ways and routes are small copyable or cheaply cloned values, nothing is hashed by text.
pub mod cf;
pub mod frame;
pub mod goal_report;
pub mod levels;
pub mod medium;
pub mod mv;
pub mod reading;
pub mod settle;
pub mod walk;

pub use cs_game::placed::{Kind, Place};
use std::collections::HashMap;

/// The two sides as the reading names them: the side to move and the other.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Side { Us, Them }

impl Side {
    pub fn other(self) -> Side { match self { Side::Us => Side::Them, Side::Them => Side::Us } }
    pub fn word(self) -> &'static str { match self { Side::Us => "us", Side::Them => "them" } }
    pub fn ix(self) -> usize { match self { Side::Us => 0, Side::Them => 1 } }
}

/// A unit: its side, its kind, the place it stands on. Ordered as the Python sheet orders units: our side first, then by kind, then by place (the numbers run in the order of the text).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Unit { pub side: Side, pub kind: Kind, pub at: Place }

/// The game's names, made once per model: the places, the kinds and the roles by number, and back.
#[derive(Debug, Clone, Default)]
pub struct Names {
    pub places: Vec<String>,
    pub kinds: Vec<String>,
    pub roles: Vec<String>,
    place_ix: HashMap<String, Place>,
    kind_ix: HashMap<String, Kind>,
}

impl Names {
    pub fn new(places: Vec<String>, kinds: Vec<String>, roles: Vec<String>) -> Names {
        let place_ix = places.iter().enumerate().map(|(i, s)| (s.clone(), i as Place)).collect();
        let kind_ix = kinds.iter().enumerate().map(|(i, s)| (s.clone(), i as Kind)).collect();
        Names { places, kinds, roles, place_ix, kind_ix }
    }
    pub fn place(&self, s: &str) -> Result<Place, String> { self.place_ix.get(s).copied().ok_or_else(|| format!("no such place: {s}")) }
    pub fn kind(&self, s: &str) -> Result<Kind, String> { self.kind_ix.get(s).copied().ok_or_else(|| format!("no such kind: {s}")) }
    pub fn role(&self, s: &str) -> Result<usize, String> { self.roles.iter().position(|r| r == s).ok_or_else(|| format!("no such role: {s}")) }
    pub fn place_text(&self, p: Place) -> &str { &self.places[p as usize] }
    pub fn kind_text(&self, k: Kind) -> &str { &self.kinds[k as usize] }
    /// "my <kind> on <place>" / "their <kind> on <place>".
    pub fn words(&self, u: &Unit) -> String { format!("{} {} on {}", if u.side == Side::Us { "my" } else { "their" }, self.kind_text(u.kind), self.place_text(u.at)) }
}

/// A set of places as a bit set: iteration runs in the order of the numbers, which is the order of the text.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub struct PlaceSet { w: Vec<u64> }

impl PlaceSet {
    pub fn new(np: usize) -> PlaceSet { PlaceSet { w: vec![0; np.div_ceil(64)] } }
    pub fn from_places(np: usize, it: impl IntoIterator<Item = Place>) -> PlaceSet { let mut s = PlaceSet::new(np); for p in it { s.insert(p); } s }
    pub fn insert(&mut self, p: Place) { self.w[(p >> 6) as usize] |= 1u64 << (p & 63); }
    pub fn contains(&self, p: Place) -> bool { self.w.get((p >> 6) as usize).is_some_and(|x| (x >> (p & 63)) & 1 == 1) }
    pub fn is_empty(&self) -> bool { self.w.iter().all(|x| *x == 0) }
    pub fn len(&self) -> usize { self.w.iter().map(|x| x.count_ones() as usize).sum() }
    pub fn union_with(&mut self, o: &PlaceSet) { for (a, b) in self.w.iter_mut().zip(o.w.iter()) { *a |= *b; } }
    pub fn intersects(&self, o: &PlaceSet) -> bool { self.w.iter().zip(o.w.iter()).any(|(a, b)| a & b != 0) }
    pub fn iter(&self) -> impl Iterator<Item = Place> + '_ {
        self.w.iter().enumerate().flat_map(|(i, &word)| {
            let mut x = word;
            std::iter::from_fn(move || {
                if x == 0 { return None; }
                let b = x.trailing_zeros();
                x &= x - 1;
                Some((i as u32 * 64 + b) as Place)
            })
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status { Live, Idle, Closed }

impl Status {
    pub fn word(self) -> &'static str { match self { Status::Live => "live", Status::Idle => "idle", Status::Closed => "closed" } }
}

/// The named numbers of the levels (the Python sheet's `Params`, the ones levels 0 and 1 read).
#[derive(Debug, Clone)]
pub struct Params {
    /// Moves of its own a supplying unit may use.
    pub reach_k: u32,
    /// The goal unit, no full plan read: live only when at most this many of its exits are open.
    pub max_open_exits: usize,
    /// The goal unit's stake: this many times the largest worth.
    pub goal_stake_multiple: f64,
    /// No unit's stake is lower.
    pub min_stake: f64,
    // ---- stage B: the deeper levels (the Python sheet's own defaults)
    /// Ways settled per step.
    pub max_ways: usize,
    /// Nodes one frame search may expand.
    pub frame_budget: u64,
    /// Routes of a way the route follower takes, the best-ordered first.
    pub max_routes: usize,
    /// Nodes one way may use over all its routes; past it the way is NOT SETTLED, never refuted.
    pub route_nodes: u64,
    /// Blocked steps one question may examine by the counterfactual.
    pub max_blocked_steps: usize,
    /// Places tried when a unit the game will not read the board without is moved aside instead of removed.
    pub aside_places: usize,
    /// Units in a joint set; more is recorded as "more than the cap in the way".
    pub max_joint: usize,
    /// Moves of an own unit in the way tried when asking whether it can move away.
    pub max_obstacle_moves: usize,
    /// The search is not deepened past this many plies.
    pub search_depth_cap: usize,
    /// Nodes the reasoning search may use on one child.
    pub reason_fuse: u64,
    /// Nodes the census may use.
    pub census_fuse: u64,
    /// The deepest level; a question that finds nothing there is closed.
    pub max_level: usize,
}

impl Default for Params {
    fn default() -> Params { Params { reach_k: 2, max_open_exits: 1, goal_stake_multiple: 2.0, min_stake: 0.1, max_ways: 4, frame_budget: 20_000, max_routes: 3, route_nodes: 2500, max_blocked_steps: 12,
                    aside_places: 4, max_joint: 3, max_obstacle_moves: 8, search_depth_cap: 3, reason_fuse: 50_000, census_fuse: 2_000_000, max_level: 4 } }
}

/// What belongs to the game and not to the position, made once per model: the names, the medium of movement, the worth of each kind (by number), the goal's kind and how many move
/// relations the vocabulary names.
pub struct Model {
    pub names: Names,
    pub medium: medium::Medium,
    pub worth: Vec<f64>,
    pub goal_kind: Option<Kind>,
    pub can_go: usize,
    /// The table the tactics search is judged by (its worth by the game's own key of a kind) and the key of each kind in it and in the tactics' names.
    pub ledger_worth: HashMap<String, i64>,
    pub letters: Vec<String>,
}

/// A cheap hasher for the small keys of the layer (numbers, units, ways): no text is ever hashed here.
#[derive(Default, Clone, Copy)]
pub struct Fx(u64);

impl std::hash::Hasher for Fx {
    fn write(&mut self, bytes: &[u8]) { for b in bytes { self.add(*b as u64); } }
    fn write_u8(&mut self, n: u8) { self.add(n as u64); }
    fn write_u16(&mut self, n: u16) { self.add(n as u64); }
    fn write_u32(&mut self, n: u32) { self.add(n as u64); }
    fn write_u64(&mut self, n: u64) { self.add(n); }
    fn write_usize(&mut self, n: usize) { self.add(n as u64); }
    fn finish(&self) -> u64 { self.0 }
}

impl Fx {
    fn add(&mut self, n: u64) { self.0 = (self.0.rotate_left(5) ^ n).wrapping_mul(0x517c_c1b7_2722_0a95); }
}

pub type FxMap<K, V> = HashMap<K, V, std::hash::BuildHasherDefault<Fx>>;
pub type FxSet<K> = std::collections::HashSet<K, std::hash::BuildHasherDefault<Fx>>;
