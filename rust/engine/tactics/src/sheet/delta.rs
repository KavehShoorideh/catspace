//! PLAN 79.14 B (SYSTEM_SPEC 8 step 1 "bring forward the open tactics carried from the last move, and update only those whose facts changed"; 4 the order of questions; 11 Rust in the hot paths; piece: the wake). WHAT THE LAST DECISION'S POSITION AND THIS ONE DIFFER IN, read from the graph alone, so that
//! what was answered about a part of the position the two plies since did not touch is carried and not asked again. Game-free: a unit is a cell (owner, kind, place) the game gave; nothing of any game is named here.
//!
//! INTERFACE (written before the bodies).
//!   `Digest::of(board, part) -> Digest`                    the atoms of a position as signatures (a relation node with the cells it is about and the attributes of its units the matcher's key reads) and its units.
//!   `Touch::between(prev, now) -> Touch`                   the cells and the places that any atom or unit that is in one position and not the other is about: what the two plies since TOUCHED (a unit moved, was taken, appeared; a line onto a place was opened or shut; a bearer came or went).
//!   `Touch::hits(cells, places) -> bool`                   whether a read set (its members and its places) holds anything touched; false means: nothing the read looked at has changed, so an answer given under it stands.
//!   `seeds_carried(board, part, max_size, levels, want, stop, prev) -> SeedRead`   the seeds of `frames::seeds_of_board_until`, found by asking only about the connected sets that hold an atom the last position did not have; the matches of the last position whose atoms are all still there are
//!       carried. The same seeds as asked afresh (the test of tests/delta.rs and the carry test of whole_79_14 check it); `SeedRead::memo` is what the next position carries.
//!   Called by: `threat_frames` (the frame read), the wake's carry (cs_decision::carry), the unit tests.
use super::base_graph::PositionPart;
use super::fates::Cell;
use super::frames::{seeds_of_set, Seed};
use super::patterns::{Board, FxHashMap, FxHashSet, Obj};
use cs_game::placed::Place;

/// A unit as the matcher reads it: its cell and the attributes its key reads (the side to move's, the goal unit's, on the edge).
pub type UDesc = (Cell, bool, bool, bool);
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TDesc { Unit(UDesc), Place(Place) }
/// One atom of a position by the cells it is about (never by an index of the position): the type of the relation, the actor, the target, the line and the unit between.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AtomSig { pub ty: u8, pub actor: UDesc, pub target: TDesc, pub line: i32, pub between: Option<UDesc> }

/// The atoms and the units of a position, by signature.
#[derive(Debug, Clone, Default)]
pub struct Digest { pub atoms: FxHashSet<AtomSig>, pub units: FxHashSet<UDesc> }

fn udesc(b: &Board, part: &PositionPart, i: usize) -> UDesc { let u = &part.units[i]; ((u.owner, u.kind, u.at), b.mine[i], b.goal[i], b.edge[i]) }

/// The signature of every atom of the board, in the board's order.
pub fn signatures(b: &Board, part: &PositionPart) -> Vec<AtomSig> {
    b.items.iter().map(|it| AtomSig {
        ty: it.ty, actor: udesc(b, part, it.actor as usize),
        target: match it.target { Obj::Unit(u) => TDesc::Unit(udesc(b, part, u as usize)), Obj::Place(p) => TDesc::Place(p) },
        line: it.line, between: if it.between >= 0 { Some(udesc(b, part, it.between as usize)) } else { None },
    }).collect()
}

impl Digest {
    pub fn of(b: &Board, part: &PositionPart) -> Digest { Digest::from_sigs(&signatures(b, part), b, part) }
    pub fn from_sigs(sigs: &[AtomSig], b: &Board, part: &PositionPart) -> Digest {
        Digest { atoms: sigs.iter().cloned().collect(), units: (0..part.units.len()).map(|i| udesc(b, part, i)).collect() }
    }
}

impl Digest {
    /// THE READ SET OF A UNIT'S ROW: the unit's cell and place and the cells and places of every atom of this position that is about the unit or its place (who bears on it, what it bears on, the lines through it, who stands between): what a reading of the unit looks at first. The cells and places, sorted.
    pub fn neighbourhood(&self, cell: Cell, place: Place) -> (Vec<Cell>, Vec<Place>) {
        let (mut cells, mut places): (FxHashSet<Cell>, FxHashSet<Place>) = (FxHashSet::default(), FxHashSet::default());
        cells.insert(cell);
        places.insert(place);
        for a in &self.atoms {
            let mut objs: Vec<Cell> = vec![a.actor.0];
            let mut pl: Vec<Place> = vec![];
            match &a.target { TDesc::Unit(u) => objs.push(u.0), TDesc::Place(p) => pl.push(*p) }
            if let Some(b) = &a.between { objs.push(b.0); }
            if objs.contains(&cell) || pl.contains(&place) || objs.iter().any(|c| c.2 == place) {
                for c in objs { cells.insert(c); places.insert(c.2); }
                for p in pl { places.insert(p); }
            }
        }
        let (mut c, mut p): (Vec<Cell>, Vec<Place>) = (cells.into_iter().collect(), places.into_iter().collect());
        c.sort();
        p.sort();
        (c, p)
    }
}

/// What two positions differ in: the cells and the places something that is in one and not the other is about.
#[derive(Debug, Clone, Default)]
pub struct Touch { pub cells: FxHashSet<Cell>, pub places: FxHashSet<Place> }

impl Touch {
    fn unit(&mut self, u: &UDesc) { self.cells.insert(u.0); self.places.insert(u.0 .2); }
    fn target(&mut self, t: &TDesc) { match t { TDesc::Unit(u) => self.unit(u), TDesc::Place(p) => { self.places.insert(*p); } } }
    fn atom(&mut self, a: &AtomSig) { self.unit(&a.actor); self.target(&a.target); if let Some(b) = &a.between { self.unit(b); } }
    /// The atoms and units in exactly one of the two digests.
    pub fn between(prev: &Digest, now: &Digest) -> Touch {
        let mut t = Touch::default();
        for a in prev.atoms.symmetric_difference(&now.atoms) { t.atom(a); }
        for u in prev.units.symmetric_difference(&now.units) { t.unit(u); }
        t
    }
    /// Whether a read set (the cells and the places it looked at) holds anything touched. A place counts through the unit standing on it and a unit through its place: both are in the set.
    pub fn hits(&self, cells: &[Cell], places: &[Place]) -> bool {
        cells.iter().any(|c| self.cells.contains(c) || self.places.contains(&c.2)) || places.iter().any(|p| self.places.contains(p))
    }
}

/// One concept match of a position: the concept, the seed and the signatures of the atoms of the embedding.
#[derive(Debug, Clone)]
pub struct SeedMatch { pub concept: u32, pub seed: Seed, pub atoms: Vec<AtomSig> }
/// What a position leaves for the next: its digest and its matches.
#[derive(Debug, Clone, Default)]
pub struct SeedMemo { pub digest: Digest, pub matches: Vec<SeedMatch> }
/// The seeds of a position, whether the clock cut the matching, how many were carried and how many asked, and the memo.
#[derive(Debug, Clone, Default)]
pub struct SeedRead { pub seeds: Vec<(u32, Seed)>, pub cut: bool, pub carried: usize, pub asked: usize, pub memo: SeedMemo, pub sigs: Vec<AtomSig> }

/// The connected sets of at most `max` atoms that hold the atom `root` and no new atom before it (so that each set is visited at the lowest new atom it holds): grown from `root` by the atoms that share an object with a member.
fn sets_with(b: &Board, root: usize, new: &[bool], max: usize, visit: &mut impl FnMut(&[usize])) {
    let mut level: Vec<Vec<usize>> = vec![vec![root]];
    for size in 1..=max {
        for s in &level { if !s.iter().any(|&j| j < root && new[j]) { visit(s); } }
        if size == max { break; }
        let mut seen: FxHashSet<Vec<usize>> = FxHashSet::default();
        let mut next: Vec<Vec<usize>> = vec![];
        for s in &level {
            for &m in s {
                for &n in &b.nbrs[m] {
                    let n = n as usize;
                    if s.contains(&n) { continue; }
                    let mut t = s.clone();
                    t.push(n);
                    t.sort_unstable();
                    if seen.insert(t.clone()) { next.push(t); }
                }
            }
        }
        level = next;
    }
}

/// THE SEEDS OF CONCEPT MATCHES, asking only what the last position did not answer. With no `prev` every connected set is asked (`seeds_of_board_until`'s own enumeration, the atoms recorded); with it the matches whose atoms are all in this position are carried and only the connected sets
/// that hold an atom the last position lacked are asked. `stop` cuts the asking as the clock of the step does (a gap, never a refutation: `cut`).
pub fn seeds_carried(b: &Board, part: &PositionPart, max_size: usize, levels: &[u8], want: &FxHashMap<u64, u32>, stop: Option<std::time::Instant>, prev: Option<&SeedMemo>) -> SeedRead {
    let sigs = signatures(b, part);
    let digest = Digest::from_sigs(&sigs, b, part);
    let mut out = SeedRead::default();
    let mut seen: FxHashSet<(u32, Vec<Cell>, Vec<Place>)> = FxHashSet::default();
    let mut found: Vec<(u32, Seed)> = vec![];
    let mut matches: Vec<SeedMatch> = vec![];
    let expired = || stop.is_some_and(|t| std::time::Instant::now() >= t);
    let take = |its: &[usize], seen: &mut FxHashSet<(u32, Vec<Cell>, Vec<Place>)>, found: &mut Vec<(u32, Seed)>, matches: &mut Vec<SeedMatch>| {
        for (c, sd) in seeds_of_set(b, part, its, levels, want, seen, found) { matches.push(SeedMatch { concept: c, seed: sd, atoms: its.iter().map(|&i| sigs[i].clone()).collect() }); }
    };
    match prev {
        None => {
            for size in 1..=max_size {
                b.connected_sets(size, &mut |its| { if out.cut || expired() { out.cut = true; return; } out.asked += 1; take(its, &mut seen, &mut found, &mut matches); });
            }
        }
        Some(p) => {
            // carried: a match of the last position whose every atom is still here (its set is connected here as it was there: the atoms share the same objects)
            for m in &p.matches {
                if m.atoms.iter().all(|a| digest.atoms.contains(a)) {
                    if seen.insert((m.concept, m.seed.units.clone(), m.seed.places.clone())) { found.push((m.concept, m.seed.clone())); }
                    matches.push(m.clone());
                    out.carried += 1;
                }
            }
            let new: Vec<bool> = sigs.iter().map(|a| !p.digest.atoms.contains(a)).collect();
            'roots: for root in (0..sigs.len()).filter(|&i| new[i]) {
                let mut stop_now = false;
                sets_with(b, root, &new, max_size, &mut |its| { if stop_now || expired() { stop_now = true; return; } out.asked += 1; take(its, &mut seen, &mut found, &mut matches); });
                if stop_now { out.cut = true; break 'roots; }
            }
        }
    }
    // one order whatever was carried (the frames are built in the order of the seeds, and joined in it): by concept, then units and places
    found.sort_by(|a, b| (a.0, &a.1.units, &a.1.places).cmp(&(b.0, &b.1.units, &b.1.places)));
    out.seeds = found;
    out.memo = SeedMemo { digest, matches };
    out.sigs = sigs;
    out
}
