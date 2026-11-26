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
//!
//! PLAN 79.58 PART B, THE ALLOWED MOVES AS FACTS WITH SUPPORTS, AND THE LAG OF TWO (SYSTEM_SPEC 11, blocks "A divergence found and named", "The support filter" and the last block (ix); LAYER 2, piece: rules; ledger rows 58, 57, 44). WRITTEN BEFORE THE BODY.
//! The fourth relation of this layer (`relations::ALLOWED`): a unit of the side that is to act may be played to a place. It belongs to the side to act only (the other side's units have no rows). A unit's rows come from the description with the places and the
//! fluents the answer read: the places of the unit fluent are the support's places, a fluent that is read as a whole is kept with a stamp of its facts (`whole_stamp`), compared across the move; the fluent that says whose turn it is is fixed by the question (the unit is of the side to act).
//!   `ReadPat { places, owner, kind }` and `AllowedSupport { places, reads, wholes, all }`   what an allowed answer read: each read of the unit fluent as the units it matches (those of the owner and kind, or any, on the places it binds: a read that leaves the place free names every place), the places of all the reads together (ascending), (fluent number, stamp of that fluent's facts when read) for a fluent read as a whole; `all`: no bound is known (the unit is asked again every time it is taken up).
//!                                    TWO RULES FROM ONE SUPPORT: the PLACE rule (`meets`: a place of the support changed) and the ROW rule (`reads_differ`: some read matches other units than it matched, what stood there and not only where). The keeper uses the row rule, the finer, and counts what the place rule alone would have asked (`asked_by_places`). The reason for the second rule: the allowed answer finds the goal unit by reading the unit fluent with its kind and side bound and the place free, which as a place is the whole board and as a row is the one place it stands on.
//!   `trait Allowing: Placed`        `allowed_with_support(&self, s, owner, kind, at) -> (Vec<Place>, AllowedSupport)`: where the unit (of the side to act at `s`) may be played, with what the answer read; `whole_stamp(&self, s, fluent) -> u64`: the stamp of a fluent's facts at `s`.
//!                                    Implemented by a game whose rules tell what they read (the description's compiled module through tests/described), never by the bridge.
//!   `AllowedKeeper::new(g, s)`       the rows of the side to act at `s`, every unit asked once.
//!   `AllowedKeeper::make(g, after)`  one tempo. THE LAG OF TWO: the side now to act took its rows at the last state at which it acted, two tempi ago, and the rows are kept by side; a unit's rows are taken up again, with no question, when no place of its support changed between that state and this
//!                                    one (the two moves between: the units removed or added, by key, between the two states) and no fluent it read as a whole has another stamp; every other unit of the side is asked again. A side that never acted before has all its units asked.
//!   `AllowedKeeper::undo()`          the tempo taken back, with the records of both sides (no question).
//!   `AllowedKeeper::rows() -> &BTreeMap<(Role, Kind, Place), Vec<Place>>`   the rows of the side to act (a unit with no move has an empty list); `side() -> Option<Role>`.
//!   `AllowedKeeper::asked() -> (units asked again by the last make, units of the side)`, `asked_units()` which they are,, `rows_again() -> (rows of the units asked again, rows of the side)`: what the last make cost; `support_of(key)` and `supports_mut()`: for a test that strikes one place out of a support.
//!   `AllowedKeeper::write(&self, &Static, &mut PositionPart)`   the rows through layer 1's door (`Board::write`, slot 4), for each unit of the graph in its order, each unit's ends ascending.
//! Called by: the tests (tests/translated_reads.rs), and later the layers that read the allowed moves off the graph instead of asking the bridge for them.
use crate::graph::Static;
use crate::placed::Placed;
use crate::relations::{UnitRelations, ALLOWED, BEAR, MAY_MOVE, SHUT};
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

// ------------------------------------------------------------------------------------------------------------------------------ PLAN 79.58 PART B: the allowed moves

/// One read of the unit fluent, as the units it matches: those of `owner` (any when None) and `kind` (any when None) standing on one of `places` (a read that leaves the file and the rank free names every place).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReadPat { pub places: Vec<Place>, pub owner: Option<Role>, pub kind: Option<Kind> }

impl ReadPat {
    /// The units of `cells` (sorted keys) that the read matches.
    pub fn matching(&self, cells: &[UnitKey]) -> Vec<UnitKey> {
        cells.iter().filter(|c| self.owner.map_or(true, |o| c.0 == o) && self.kind.map_or(true, |k| c.1 == k) && self.places.binary_search(&c.2).is_ok()).copied().collect()
    }
}

/// What an allowed answer read (plan 79.58 part B): every read of the unit fluent (`reads`) with the places they bind together (`places`, ascending), the fluents read as a whole with the stamp of their facts when read, and `all` when no bound is known.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct AllowedSupport { pub places: Vec<Place>, pub reads: Vec<ReadPat>, pub wholes: Vec<(u16, u64)>, pub all: bool }

impl AllowedSupport {
    pub fn all() -> AllowedSupport { AllowedSupport { places: vec![], reads: vec![], wholes: vec![], all: true } }
    /// THE PLACE RULE: whether a place of `changed` is in the support.
    pub fn meets(&self, changed: &[Place]) -> bool { self.all || changed.iter().any(|c| self.places.binary_search(c).is_ok()) }
    /// THE ROW RULE: whether some read matches other units in the two states (what stood there, not only where): the answer is the same where none does (Ideas.Support.Test.run_of_disjoint).
    pub fn reads_differ(&self, before: &[UnitKey], after: &[UnitKey]) -> bool { self.all || self.reads.iter().any(|r| r.matching(before) != r.matching(after)) }
}

/// A game whose rules answer where a unit of the side to act may be played, with what the answer read (plan 79.58 part B).
pub trait Allowing: Placed {
    fn allowed_with_support(&self, s: &Self::State, owner: Role, kind: Kind, at: Place) -> (Vec<Place>, AllowedSupport);
    /// The stamp of the facts of a fluent (by the number the supports use) at a state: equal stamps are equal facts.
    fn whole_stamp(&self, s: &Self::State, fluent: u16) -> u64;
}

/// What a side took at the last state it acted at: the units (by key) and, for each unit of the side, its allowed ends and its support; what taking it cost.
#[derive(Clone, Debug, Default)]
struct SideRecord {
    cells: Vec<UnitKey>, rows: std::collections::BTreeMap<UnitKey, Vec<Place>>, sup: std::collections::BTreeMap<UnitKey, AllowedSupport>,
    /// the units asked again, how many the place rule alone would have asked, the rows of the units asked again
    asked: Vec<UnitKey>, asked_by_places: usize, rows_again: usize,
}

pub struct AllowedKeeper {
    recs: Vec<Option<std::rc::Rc<SideRecord>>>, cur: Option<Role>,
    /// for each make: (the side that acted before, the side whose record it replaced, that record)
    stack: Vec<(Option<Role>, Option<Role>, Option<std::rc::Rc<SideRecord>>)>,
}

impl AllowedKeeper {
    fn take<G: Game + Allowing>(g: &G, s: &G::State, old: Option<&SideRecord>) -> SideRecord {
        let m = g.mover(s).expect("a side to act");
        let mut cells = g.cells(s);
        cells.sort_by_key(|(o, k, p)| (*o, *k, *p));
        let changed: Vec<Place> = match old {
            None => vec![],
            Some(r) => { let mut c: Vec<Place> = r.cells.iter().filter(|k| cells.binary_search(k).is_err()).map(|k| k.2).collect(); c.extend(cells.iter().filter(|k| r.cells.binary_search(k).is_err()).map(|k| k.2)); c }
        };
        let mut stamps: std::collections::BTreeMap<u16, u64> = Default::default();
        let mut rec = SideRecord { cells: cells.clone(), ..Default::default() };
        for &key in cells.iter().filter(|k| k.0 == m) {
            let mut place_rule = true;
            let kept = old.and_then(|r| match (r.rows.get(&key), r.sup.get(&key)) {
                (Some(rows), Some(sup)) => {
                    place_rule = sup.meets(&changed);
                    let wholes_same = sup.wholes.iter().all(|&(f, h)| *stamps.entry(f).or_insert_with(|| g.whole_stamp(s, f)) == h);
                    if wholes_same && !sup.all && (!place_rule || !sup.reads_differ(&r.cells, &cells)) { Some((rows.clone(), sup.clone())) } else { None }
                }
                _ => None,
            });
            if place_rule { rec.asked_by_places += 1; }
            let (rows, sup) = match kept {
                Some(x) => x,
                None => { rec.asked.push(key); let (r, sp) = g.allowed_with_support(s, key.0, key.1, key.2); rec.rows_again += r.len(); (r, sp) }
            };
            rec.rows.insert(key, rows);
            rec.sup.insert(key, sup);
        }
        rec
    }

    /// The rows of the side to act at `s`, every unit asked once.
    pub fn new<G: Game + Allowing>(g: &G, s: &G::State) -> AllowedKeeper {
        let mut recs = vec![None; g.roles().len()];
        let cur = g.mover(s);
        if let Some(m) = cur { recs[m] = Some(std::rc::Rc::new(Self::take(g, s, None))); }
        AllowedKeeper { recs, cur, stack: vec![] }
    }

    /// One tempo: the lag of two (see the header).
    pub fn make<G: Game + Allowing>(&mut self, g: &G, after: &G::State) {
        let mover = g.mover(after);
        let mut replaced = None;
        if let Some(m) = mover {
            let r = Self::take(g, after, self.recs[m].as_deref());
            replaced = Some(m);
            let old = self.recs[m].replace(std::rc::Rc::new(r));
            self.stack.push((self.cur, replaced, old));
        } else { self.stack.push((self.cur, None, None)); }
        self.cur = mover;
    }

    pub fn undo(&mut self) -> bool {
        let Some((cur, side, rec)) = self.stack.pop() else { return false };
        if let Some(m) = side { self.recs[m] = rec; }
        self.cur = cur;
        true
    }

    fn rec(&self) -> Option<&SideRecord> { self.cur.and_then(|m| self.recs[m].as_deref()) }
    pub fn side(&self) -> Option<Role> { self.cur }
    /// The allowed ends of each unit of the side to act (none when no side is to act).
    pub fn rows(&self) -> std::collections::BTreeMap<(usize, u8, Place), Vec<Place>> { self.rec().map(|r| r.rows.clone()).unwrap_or_default() }
    /// (units asked again by the last make, units of the side to act).
    pub fn asked(&self) -> (usize, usize) { self.rec().map(|r| (r.asked.len(), r.rows.len())).unwrap_or((0, 0)) }
    /// How many units the PLACE rule alone would have asked again at the last make (the row rule asks fewer or as many).
    pub fn asked_by_places(&self) -> usize { self.rec().map(|r| r.asked_by_places).unwrap_or(0) }
    /// (rows of the units asked again, rows of the side to act).
    pub fn rows_again(&self) -> (usize, usize) { self.rec().map(|r| (r.rows_again, r.rows.values().map(|v| v.len()).sum())).unwrap_or((0, 0)) }
    /// The units of the side to act that the last make asked again (all of them when the side had no record).
    pub fn asked_units(&self) -> Vec<(usize, u8, Place)> { self.rec().map(|r| r.asked.clone()).unwrap_or_default() }
    pub fn support_of(&self, key: &(usize, u8, Place)) -> Option<&AllowedSupport> { self.rec().and_then(|r| r.sup.get(key)) }
    /// The supports of the side to act, to strike one out (a test of the planted fault); the record is copied first when it is shared with an earlier tempo.
    pub fn supports_mut(&mut self) -> Option<&mut std::collections::BTreeMap<(usize, u8, Place), AllowedSupport>> {
        let m = self.cur?;
        self.recs[m].as_mut().map(|r| &mut std::rc::Rc::make_mut(r).sup)
    }

    /// The rows through layer 1's door onto a graph that holds the units: for each unit of the graph, in its order, its ends ascending (slot 4).
    pub fn write(&self, st: &Static, part: &mut PositionPart) {
        let Some(r) = self.rec() else { return };
        for ui in 0..part.units.len() {
            let u = part.units[ui];
            if let Some(ends) = r.rows.get(&(u.owner, u.kind, u.at)) { for &p in ends { st.board().write(part, ALLOWED, ui, p); } }
        }
    }
}
