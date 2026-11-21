//! PLAN 79.21 STAGE B (SYSTEM_SPEC 11 "one graph is the representation"; piece: representation). The small value types the graph speaks in, moved down from cs_tactics' sheet (which re-exports them): a place is its index in the game's list of place names, a kind its index in
//! the list of kind names; a side is the side to move or the other; `Names` holds the three lists once and answers by number or by text; `PlaceSet` is a bit set of places. Nothing here names an object of any game.
use std::collections::HashMap;

/// A place: its index in the game's names of its places (sorted by text).
pub type Place = u16;
/// A kind of unit: its index in the game's names of its kinds (sorted by text).
pub type Kind = u8;
/// A role, by its index in the game's roles.
pub type Role = usize;

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
