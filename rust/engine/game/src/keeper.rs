//! PLAN 79.21 STAGE B, LAYER 2'S KEEPER OF THE GRAPH (SYSTEM_SPEC 11 "one graph is the representation", "layer 1 holds the history"; Kaveh 2026-11-20: "the graph of a position must be kept step by step in make and undo, never rebuilt from facts at each node"; piece: representation, the rules above it).
//! Layer 1 keeps the graph (`cs_rep::live::LiveGraph`: units, tempo, the relations written through the door, make and undo). WHAT A TEMPO CHANGES IN THE RELATIONS is a question of the rules, so it is asked here: after a tempo the places whose holder changed are `C`; a unit's relations (what
//! it bears on, where it may move) can have changed only if the unit is new, or a place of `C` lies on a line it stands on, or a place of `C` is one of its relations' places; those units are asked again of the rules (`bearing_from`, `reach_from`) and every other unit keeps what it had. The shut
//! lines belong to the side to move, which changes every tempo, so they are asked again for that side's units (`Static::shut_places`). The rule is a SUPERSET of what can change, tested equal to a fresh build at every position of the fixture games and on undo.
//!
//! INTERFACE.
//!   `GraphKeeper::new(&Static, g, s) -> GraphKeeper`        the graph of a first position, every relation asked of the rules once.
//!   `GraphKeeper::make(&mut self, &Static, g, after)`        one tempo: `after` is the state reached; only the units the tempo can have changed are asked again.
//!   `GraphKeeper::undo(&mut self) -> bool`                   the last tempo taken back (no rule is asked).
//!   `GraphKeeper::part(&self, &Static) -> PositionPart`      the graph as it now is, equal to `Static::position` on the state (the lines through included once a profile is learned).
//!   `GraphKeeper::asked() -> (units asked again, units in all)`   what the last make cost in questions to the rules.
//!   PLAN 79.22 PART A: `GraphKeeper::from_live(LiveGraph)` and `into_live(self)`: the graph is the game's, kept in the game's memory between decisions (`cs_rep::store::AtHand`); the keeper is taken up from it, brought to the position at hand by `make`, and given back.
//!
//! PLAN 79.57 PART B, THE SECOND RULE: EACH ANSWER WITH ITS SUPPORT (SYSTEM_SPEC 11, blocks "A divergence found and named" and "The support filter"; LAYER 2, piece: rules; ledger rows 58, 57). WRITTEN BEFORE THE BODY.
//! The hand rule above is a guess at what an answer depends on. The second rule is the thing itself: the rules' own answer for a unit comes with the PLACES OF THE STATE IT READ (its support); a unit is asked again after a tempo only when it is new or a place the tempo changed is in its support.
//! The licence is the lemma proved in lean/ideas/Ideas/Support.lean (`Test.run_of_disjoint`): an answer made by reading facts is the same in two states that agree on every fact that was read. The supports are layer 2's and live HERE beside the graph, with their own undo; layer 1 is not changed.
//!   `Support { places: Vec<Place>, all: bool }`   the places an answer read, ascending; `all`: the answer cannot bound its reads (it is asked again after every tempo). `Support::meets(&self, changed: &[Place]) -> bool`.
//!   `trait Supported: Placed`                    `bearing_with_support(&self, s, owner, kind, at) -> (Vec<Place>, Support)` and `reach_with_support(..)`: what `bearing_from` / `reach_from` return, with the support of the answer. Implemented by a game whose rules can tell what they read (the description's compiled module through tests/described, never the bridge).
//!   `GraphKeeper::new_supported(&Static, g, s) -> GraphKeeper`        like `new`, every unit's two answers asked with their supports (a unit's support is the union of its two).
//!   `GraphKeeper::make_supported(&mut self, &Static, g, after)`        like `make`, but a unit is asked again only when it is new or a changed place is in its support. Only on a keeper made by `new_supported` (the two rules are not mixed in one keeper: asserted).
//!   `GraphKeeper::undo` takes the supports back with the graph. `GraphKeeper::rows_again() -> (rows asked again by the last make, rows held in all)`: the relation rows (bear and may_move) of the units asked again, and of every unit, for the measure (counted by both rules).
//!   `GraphKeeper::support_of(&self, unit key) -> Option<&Support>` and `supports_mut()`: for the tests (a support with one place struck out by hand must be caught by the equality with the fresh graph).
//! Called by: the tests, and the layers that walk a sequence of positions (a game being played, a line being read) instead of building the graph of each from nothing.
use crate::graph::Static;
use crate::placed::Placed;
use crate::relations::{UnitRelations, BEAR, MAY_MOVE, SHUT};
use crate::{Game, Role};
use cs_rep::names::Kind;
use cs_rep::live::{Edit, LiveGraph, LiveUnit};
use cs_rep::graph::PositionPart;
use cs_rep::names::Place;

/// The places an answer read (plan 79.57 part B); `all`: no bound is known, so the answer is asked again after every tempo.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Support { pub places: Vec<Place>, pub all: bool }

impl Support {
    pub fn all() -> Support { Support { places: vec![], all: true } }
    pub fn of(mut places: Vec<Place>) -> Support { places.sort_unstable(); places.dedup(); Support { places, all: false } }
    pub fn union(&self, o: &Support) -> Support { if self.all || o.all { return Support::all(); } Support::of(self.places.iter().chain(o.places.iter()).copied().collect()) }
    /// Whether a place of `changed` is in the support.
    pub fn meets(&self, changed: &[Place]) -> bool { self.all || changed.iter().any(|c| self.places.binary_search(c).is_ok()) }
}

/// A game whose rules answer a unit's bearing and reach together with the places of the state the answer read (plan 79.57 part B).
pub trait Supported: Placed {
    fn bearing_with_support(&self, s: &Self::State, owner: Role, kind: Kind, at: Place) -> (Vec<Place>, Support);
    fn reach_with_support(&self, s: &Self::State, owner: Role, kind: Kind, at: Place) -> (Vec<Place>, Support);
}

type UnitKey = (usize, u8, Place);

pub struct GraphKeeper {
    live: LiveGraph, asked: (usize, usize), rows: (usize, usize),
    /// plan 79.57: the support of each unit's answers (only on a keeper made by `new_supported`), and for each make the old values of the keys it changed (for undo)
    supported: bool, sup: std::collections::BTreeMap<UnitKey, Support>, sup_undo: Vec<Vec<(UnitKey, Option<Support>)>>,
}

impl GraphKeeper {
    #[track_caller]
    pub fn new<G: Game + Placed>(st: &Static, g: &G, s: &G::State) -> GraphKeeper {
        if crate::graph::builds::on() { crate::graph::builds::note(std::panic::Location::caller(), g.key(s)); }          // a graph asked of the rules from nothing counts as a build
        let mover = g.mover(s);
        let mut units: Vec<LiveUnit> = g.cells(s).into_iter().map(|(owner, kind, at)| { let mut u = LiveUnit::new(owner, kind, at); u.slots[BEAR.slot as usize] = g.bearing_from(s, owner, kind, at); u.slots[MAY_MOVE.slot as usize] = g.reach_from(s, owner, kind, at); u }).collect();
        units.sort_by_key(|u| u.key());
        if let Some(m) = mover { for u in units.iter_mut().filter(|u| u.owner == m) { let shut = st.shut_places(g, s, u.at, u.bears()); u.slots[SHUT.slot as usize] = shut; } }
        let n = units.len();
        GraphKeeper { live: LiveGraph::new(mover, units), asked: (n, n), rows: (0, 0), supported: false, sup: Default::default(), sup_undo: vec![] }
    }

    /// The keeper taken up again from the graph the game's memory kept.
    pub fn from_live(live: LiveGraph) -> GraphKeeper { let n = live.units().len(); GraphKeeper { live, asked: (n, n), rows: (0, 0), supported: false, sup: Default::default(), sup_undo: vec![] } }
    /// The graph, to go back into the game's memory.
    pub fn into_live(self) -> LiveGraph { self.live }

    pub fn make<G: Game + Placed>(&mut self, st: &Static, g: &G, after: &G::State) {
        let mover = g.mover(after);
        let mut cells = g.cells(after);
        cells.sort_by_key(|(o, k, p)| (*o, *k, *p));
        let before: Vec<(usize, u8, Place)> = self.live.units().iter().map(|u| u.key()).collect();
        let removed: Vec<_> = before.iter().filter(|k| cells.binary_search(k).is_err()).copied().collect();
        let added: Vec<_> = cells.iter().filter(|k| before.binary_search(k).is_err()).copied().collect();
        let changed: Vec<Place> = removed.iter().chain(added.iter()).map(|k| k.2).collect();
        let mut rewrite = vec![];
        let (mut asked, mut rows_again, mut rows_all) = (0, 0usize, 0usize);
        // the relations of every unit of the state reached, as they will stand: the old ones kept, the ones the tempo can have changed asked again
        let mut bears_of: Vec<((usize, u8, Place), Vec<Place>)> = Vec::with_capacity(cells.len());
        for &(owner, kind, at) in &cells {
            let key = (owner, kind, at);
            let old = self.live.units().binary_search_by_key(&key, |u| u.key()).ok().map(|i| &self.live.units()[i]);
            let again = match old {
                None => true,
                Some(u) => changed.iter().any(|&c| !st.common_lines(at, c).is_empty() || u.bears().contains(&c) || u.reach().contains(&c)),
            };
            if again {
                asked += 1;
                let (bears, moves) = (g.bearing_from(after, owner, kind, at), g.reach_from(after, owner, kind, at));
                rows_again += bears.len() + moves.len();
                rows_all += bears.len() + moves.len();
                bears_of.push((key, bears.clone()));
                rewrite.push((key, [bears, moves]));
            } else { let u = old.unwrap(); rows_all += u.bears().len() + u.reach().len(); bears_of.push((key, u.bears().clone())); }
        }
        self.rows = (rows_again, rows_all);
        let mut shuts = vec![];
        if let Some(m) = mover {
            for (key, bears) in &bears_of { if key.0 == m { shuts.push((*key, st.shut_places(g, after, key.2, bears))); } }
        }
        self.asked = (asked, cells.len());
        self.live.make(Edit { tempo: mover, removed, added, rewrite, clear_last: true, last: shuts });
    }

    pub fn undo(&mut self) -> bool {
        let ok = self.live.undo();
        if ok && self.supported {
            for (k, old) in self.sup_undo.pop().unwrap_or_default().into_iter().rev() { match old { Some(v) => { self.sup.insert(k, v); } None => { self.sup.remove(&k); } } }
        }
        ok
    }
    /// (the relation rows of the units the last make asked again, the relation rows of every unit after it): what the last tempo cost in rows (plan 79.57 part C; counted by both rules).
    pub fn rows_again(&self) -> (usize, usize) { self.rows }
    pub fn support_of(&self, key: &(usize, u8, Place)) -> Option<&Support> { self.sup.get(key) }
    /// The supports by the units' keys, for a test that strikes one out.
    pub fn supports_mut(&mut self) -> &mut std::collections::BTreeMap<(usize, u8, Place), Support> { &mut self.sup }

    /// The graph of a first position with every unit's answers asked WITH THEIR SUPPORTS (plan 79.57 part B).
    pub fn new_supported<G: Game + Supported>(st: &Static, g: &G, s: &G::State) -> GraphKeeper {
        let mover = g.mover(s);
        let mut sup = std::collections::BTreeMap::new();
        let mut units: Vec<LiveUnit> = g.cells(s).into_iter().map(|(owner, kind, at)| {
            let mut u = LiveUnit::new(owner, kind, at);
            let ((bears, sb), (moves, sr)) = (g.bearing_with_support(s, owner, kind, at), g.reach_with_support(s, owner, kind, at));
            u.slots[BEAR.slot as usize] = bears;
            u.slots[MAY_MOVE.slot as usize] = moves;
            sup.insert(u.key(), sb.union(&sr));
            u
        }).collect();
        units.sort_by_key(|u| u.key());
        if let Some(m) = mover { for u in units.iter_mut().filter(|u| u.owner == m) { let shut = st.shut_places(g, s, u.at, u.bears()); u.slots[SHUT.slot as usize] = shut; } }
        let n = units.len();
        let rows = units.iter().map(|u| u.bears().len() + u.reach().len()).sum();
        GraphKeeper { live: LiveGraph::new(mover, units), asked: (n, n), rows: (rows, rows), supported: true, sup, sup_undo: vec![] }
    }

    /// One tempo by the SECOND RULE: a unit is asked again only when it is new or a place the tempo changed is in its support.
    pub fn make_supported<G: Game + Supported>(&mut self, st: &Static, g: &G, after: &G::State) {
        assert!(self.supported, "make_supported on a keeper not made by new_supported");
        let mover = g.mover(after);
        let mut cells = g.cells(after);
        cells.sort_by_key(|(o, k, p)| (*o, *k, *p));
        let before: Vec<UnitKey> = self.live.units().iter().map(|u| u.key()).collect();
        let removed: Vec<_> = before.iter().filter(|k| cells.binary_search(k).is_err()).copied().collect();
        let added: Vec<_> = cells.iter().filter(|k| before.binary_search(k).is_err()).copied().collect();
        let changed: Vec<Place> = removed.iter().chain(added.iter()).map(|k| k.2).collect();
        let mut undo: Vec<(UnitKey, Option<Support>)> = removed.iter().map(|k| (*k, self.sup.remove(k))).collect();
        let (mut rewrite, mut asked, mut rows_again, mut rows_all) = (vec![], 0, 0usize, 0usize);
        let mut bears_of: Vec<(UnitKey, Vec<Place>)> = Vec::with_capacity(cells.len());
        for &(owner, kind, at) in &cells {
            let key = (owner, kind, at);
            let old = self.live.units().binary_search_by_key(&key, |u| u.key()).ok().map(|i| &self.live.units()[i]);
            let again = match old { None => true, Some(_) => self.sup.get(&key).map(|s| s.meets(&changed)).unwrap_or(true) };
            if again {
                asked += 1;
                let ((bears, sb), (moves, sr)) = (g.bearing_with_support(after, owner, kind, at), g.reach_with_support(after, owner, kind, at));
                rows_again += bears.len() + moves.len();
                rows_all += bears.len() + moves.len();
                undo.push((key, self.sup.insert(key, sb.union(&sr))));
                bears_of.push((key, bears.clone()));
                rewrite.push((key, [bears, moves]));
            } else {
                let u = old.unwrap();
                rows_all += u.bears().len() + u.reach().len();
                bears_of.push((key, u.bears().clone()));
            }
        }
        let mut shuts = vec![];
        if let Some(m) = mover {
            for (key, bears) in &bears_of { if key.0 == m { shuts.push((*key, st.shut_places(g, after, key.2, bears))); } }
        }
        self.asked = (asked, cells.len());
        self.rows = (rows_again, rows_all);
        self.sup_undo.push(undo);
        self.live.make(Edit { tempo: mover, removed, added, rewrite, clear_last: true, last: shuts });
    }
    pub fn asked(&self) -> (usize, usize) { self.asked }
    pub fn depth(&self) -> usize { self.live.depth() }
    /// The graph kept, for a question asked of it as it is (no node made).
    pub fn live(&self) -> &LiveGraph { &self.live }

    /// The graph as it now is. The lines through are derived from the bears written and the profile of the kinds (geometry, no rule).
    pub fn part(&self, st: &Static) -> PositionPart { st.part_of(&self.live) }
}
