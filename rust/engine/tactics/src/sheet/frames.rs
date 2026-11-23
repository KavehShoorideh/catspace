//! PLAN 78.55 THE FRAME, ITS SEQUENCES AND FATE RECORDS, THE PASS, AND WHAT EACH SIDE CAN FORCE, on the layer-1 graph. The game-free form of the October work (fates.py `square_frame`, `outcomes_table`,
//! `with_passes`, `compose`; frame_search.py; waiting.rs; the Lean `withPass` / `frame_sound_pass` / `far_stays_off`), over `Placed` and the graph, with NO worth table: no number of any kind is made
//! of an outcome. Nothing here names an object of any game: units are (owner, kind, place) as the game gives them, a kind is a raw index, a place a number.
//!
//! A. THE FRAME. From a SEED (units and places) the frame is its set of PLACES P (the seed's places and the places of the seed's units) and its MEMBERS: the units standing on a place of P, every unit that
//! bears on a place of P (a layer-1 `bear` node), and, to a fixed point, every unit that bears on a place of P THROUGH a member (a layer-1 `through` node: the unit behind a unit of the frame, which the
//! October `square_frame` found by lifting the frame's units off the board). Closure is to a fixed point; the stated bound is `MAX_MEMBERS` (a bigger frame is refused and counted, never cut). The
//! members are the units tracked along every line. A frame is a graph node `frame:<hash of its sorted members and places>`. LIMIT: the through relation exists only once the profile of the kinds is
//! learned (`Static::learn`), it sees the unit behind a unit, not behind two, and not onto an EMPTY place. Two frames are IN CONTACT when they share a unit or a place or a unit of one bears on a place of
//! the other; frames in contact are joined (`join_in_contact`) and read as one.
//!
//! B. SEQUENCES. A move of the frame is a legal move of the game (asked through `Game::moves`) by a member onto a place of P ("inside"), or a PASS. Sides alternate. HOW A PASS IS MADE on a real position:
//! `Game::pass` (the same position with the other side to move, nothing played; None in a position the game refuses a pass in), and it must also be REALISABLE: the side has a legal move outside the
//! frame that leaves the frame's projection (the member cells on P and the units bearing on P, through members included) as it was -- the Lean hypothesis `htheirs` (a reply leaves the projection as it
//! was) and `excluded_is_pass`, checked on the real position instead of assumed. WHAT A PASS CANNOT REPRESENT: a side that must answer a threat to its goal unit, or a position the game ends (no `Game::pass`), a side whose every legal move
//! changes the frame's projection (a zugzwang: the pass is then not realisable and is not offered), a side's move that changes the projection from outside (an arrival on or onto the frame), and
//! what the game drops by passing (a right that lapses with a turn, by the game's own pass). The moves the lemma needs among the REPLIES and that are neither inside nor a pass (a member leaving the frame, an
//! outsider arriving on it) are one LEAVE option, offered to the side that is NOT the one whose claims are being read (soundness: the lemma needs every reply of theirs; our own moves may be fewer): the
//! line then ends with the units as they stand (nothing further is lost to the leaver, nothing is guessed). A line is bounded by a DEPTH of one ply per member plus one (counted as BOUNDED, a stated bound
//! and not a fuse) and by the FUSE: positions read. A line the fuse cut ends UNREAD and is counted: its value is the worst for the side whose claim is read, so no claim is ever made on it.
//!
//! C. THE OUTCOME is a FATE RECORD per member: alive (standing where) or taken (by which member, at which ply). `outcomes_table` lists the distinct outcomes with the first line reaching each.
//!
//! D. WHAT EACH SIDE CAN FORCE, WITH NO WORTH. An outcome is the pair (the members the reading side loses, the members the other side loses). Outcomes are compared only by INCLUSION: one is at least as good
//! for the reading side X as another when X loses no member the other keeps and the other side loses every member it loses there (`Pair::at_least`). What X can force is the set of its GUARANTEED outcomes
//! that no other guaranteed outcome beats (an antichain of pairs, kept as a set, never collapsed to a number): at a node of X, the unbeaten among its options' guarantees; at a node of the other side, the
//! unbeaten among the meets (the worst of each, by union of X's losses and intersection of the other's) of one guarantee from each reply. A CLAIM is a statement over such pairs: "X can force an outcome
//! in which member u is taken and none of X's is" (`Claim::TakeFree`), "... u is taken" (`TakeAny`), "X can force an outcome in which its member u is not taken" (`Safe`).
//!
//! E. THE PASS COUNT AND THE CONDITIONAL CLAIM. The forceable set is a function of who is to move at the root and of how many passes each side may spend (`bud`, `ALL` = every ply). Soundness form of the Lean
//! lemma: X spends none, the other side may pass at every ply (`bud = [X: 0, other: ALL]`). A claim that is not forced there, X not to move, and holds with the other side to move (it passed once) is a
//! THREAT; one that holds with X to move and ONE pass of its own is a WAITING claim; a claim that is not forced and holds after ONE move of the side to move that ARRIVES on or onto the frame (a unit that
//! was not a member and, after the move, is) is a CONDITIONAL claim: the condition is that move's arrival (kind, place), read by trying each such move. Layer-3 nodes: `claim:<frame>:<statement>`,
//! `condition:<claim>:<change>` (`L3`).
//!
//! F. TWO FRAMES. `compose` computes the forceable set of two frames not in contact from the two frames' sets (each side's moves in the other frame are the other side's passes there: the sound form of E
//! for each frame, the guarantees unioned) and `compose_check` compares it with the joint read of the union frame. NO LEAN THEOREM COVERS THIS COMPOSITION (SYSTEM_SPEC 12.1): it is CHECKED per
//! position, not proved, and nothing may treat it as proved.
use cs_game::relations::{PartRelations};
use super::base_graph::{PositionPart, Static};
use super::fates::{step_between, Cell, Step};
use super::patterns::{Board, FxHashMap, FxHashSet, Obj};
use super::{Kind, Place, PlaceSet};
use cs_game::placed::Placed;
use cs_game::{Game, Role};
use std::hash::{Hash, Hasher};
use std::rc::Rc;

/// The most members a frame may have (the ids are the bits of a `u16` mask, with room for the one unit a conditional claim lets arrive).
pub const MAX_MEMBERS: usize = 12;
/// The fuse: positions read in one reading (the same size as the forcing read's, `forcing::FUSE_POSITIONS`).
pub const FUSE_POSITIONS: u64 = 20_000;
/// A pass budget that is never spent up: every ply may be a pass.
pub const ALL: u8 = u8::MAX;
/// The most guarantees a set may hold; a set that would grow past it ends that line UNREAD (counted), never cut quietly.
pub const MAX_PAIRS: usize = 128;
/// The most distinct outcomes the table keeps.
pub const MAX_OUTCOMES: usize = 400;

// ------------------------------------------------------------------------------------------------------------------------------------------------------------------ outcomes compared by inclusion

/// An outcome by inclusion: the members (bits of the frame's ids) the reading side loses and the members the other side loses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Pair { pub lx: u16, pub lo: u16 }

impl Pair {
    pub const ZERO: Pair = Pair { lx: 0, lo: 0 };
    /// The worst outcome there is for the reading side (an unread line is valued so): it loses every member and the other loses none.
    pub const BOTTOM: Pair = Pair { lx: 0xFFFF, lo: 0 };
    /// This outcome is at least as good for the reading side as `o`: it loses no member `o` keeps and the other side loses every member it loses in `o`.
    pub fn at_least(self, o: Pair) -> bool { self.lx & !o.lx == 0 && o.lo & !self.lo == 0 }
    /// The worst of two (what is guaranteed when either may happen): the reading side loses what it loses in either, the other side only what it loses in both.
    pub fn meet(self, o: Pair) -> Pair { Pair { lx: self.lx | o.lx, lo: self.lo & o.lo } }
    pub fn plus(self, lx: u16, lo: u16) -> Pair { Pair { lx: self.lx | lx, lo: self.lo | lo } }
}

/// The unbeaten of a set of outcomes: those no other outcome of the set strictly beats (equal ones once), sorted.
pub fn unbeaten(mut v: Vec<Pair>) -> Vec<Pair> {
    v.sort();
    v.dedup();
    let keep: Vec<Pair> = v.iter().copied().filter(|a| !v.iter().any(|b| b != a && b.at_least(*a))).collect();
    keep
}

/// The guarantees at a node of the other side: the unbeaten among the meets of one guarantee from each reply. None when a set grows past `MAX_PAIRS`.
fn meet_all(sets: &[Vec<Pair>]) -> Option<Vec<Pair>> {
    let mut acc = sets[0].clone();
    for s in &sets[1..] {
        let mut next = Vec::with_capacity(acc.len() * s.len());
        for a in &acc { for b in s { next.push(a.meet(*b)); } }
        acc = unbeaten(next);
        if acc.len() > MAX_PAIRS { return None; }
    }
    Some(acc)
}

// ------------------------------------------------------------------------------------------------------------------------------------------------------------------ the seed and the frame

/// A seed: units as the game gives them (owner, kind, place) and places.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Seed { pub units: Vec<Cell>, pub places: Vec<Place> }

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FrameErr { TooLarge(usize), UnitGone(Cell), NoProfile }

impl FrameErr {
    pub fn word(&self) -> String { match self { FrameErr::TooLarge(n) => format!("too large: {n} members"), FrameErr::UnitGone(c) => format!("a seed unit is not on the board: {c:?}"), FrameErr::NoProfile => "no profile learned".into() } }
}

/// A frame: its places, its members at the root (sorted cells; the index in this list is the member's id), the seed it was made from and its node id.
#[derive(Debug, Clone)]
pub struct Frame { pub seed: Seed, pub places: Vec<Place>, pub members: Vec<Cell>, pub id: String }

fn closure(part: &PositionPart, pset: &PlaceSet, init: &[usize]) -> Vec<usize> {
    let mut inm = vec![false; part.units.len()];
    for &u in init { inm[u] = true; }
    for p in pset.iter() { let o = part.occupant[p as usize]; if o != 0 { inm[o as usize - 1] = true; } }
    for r in part.bears() { if pset.contains(r.place) { inm[r.unit] = true; } }
    loop {
        let mut changed = false;
        for r in part.throughs() {
            if !inm[r.unit] && pset.contains(r.place) && r.between.iter().any(|&b| inm[b]) { inm[r.unit] = true; changed = true; }
        }
        if !changed { break; }
    }
    (0..inm.len()).filter(|&i| inm[i]).collect()
}

fn cell_of(part: &PositionPart, i: usize) -> Cell { let u = &part.units[i]; (u.owner, u.kind, u.at) }

/// The places of a seed: its own and its units' places.
fn places_of_seed(np: usize, seed: &Seed) -> PlaceSet {
    let mut ps = PlaceSet::new(np);
    for &p in &seed.places { ps.insert(p); }
    for c in &seed.units { ps.insert(c.2); }
    ps
}

fn frame_id(st: &Static, members: &[Cell], places: &[Place]) -> String {
    let mut text: Vec<String> = members.iter().map(|c| unit_id(st, c)).collect();
    text.extend(places.iter().map(|p| st.names.places[*p as usize].clone()));
    let mut h = rustc_hash::FxHasher::default();
    for t in &text { h.write(t.as_bytes()); h.write_u8(0); }
    format!("frame:{:016x}", h.finish())
}

pub fn unit_id(st: &Static, c: &Cell) -> String { format!("unit:{}:{}:{}", st.names.roles[c.0], st.names.kinds[c.1 as usize], st.names.places[c.2 as usize]) }

impl Frame {
    /// The frame of a seed on a position (see the module text, A).
    pub fn new<G: Game + Placed>(st: &Static, g: &G, s: &G::State, seed: &Seed) -> Result<Frame, FrameErr> {
        let part = st.position(g, s);
        Frame::from_part(st, &part, seed)
    }

    pub fn from_part(st: &Static, part: &PositionPart, seed: &Seed) -> Result<Frame, FrameErr> {
        let np = st.names.places.len();
        let mut init = vec![];
        for c in &seed.units {
            match part.units.iter().position(|u| (u.owner, u.kind, u.at) == *c) { Some(i) => init.push(i), None => return Err(FrameErr::UnitGone(*c)) }
        }
        let pset = places_of_seed(np, seed);
        let m = closure(part, &pset, &init);
        if m.len() > MAX_MEMBERS { return Err(FrameErr::TooLarge(m.len())); }
        let mut members: Vec<Cell> = m.iter().map(|&i| cell_of(part, i)).collect();
        members.sort();
        let places: Vec<Place> = pset.iter().collect();
        let id = frame_id(st, &members, &places);
        Ok(Frame { seed: seed.clone(), places, members, id })
    }

    pub fn id_of(&self, c: &Cell) -> Option<u8> { self.members.binary_search(c).ok().map(|i| i as u8) }
}

/// Two frames are in contact: they share a unit or a place, or a unit of one bears on a place of the other.
pub fn in_contact(a: &Frame, b: &Frame, part: &PositionPart) -> bool {
    if a.members.iter().any(|c| b.members.contains(c)) || a.places.iter().any(|p| b.places.contains(p)) { return true; }
    let bears_on = |x: &Frame, y: &Frame| x.members.iter().any(|c| {
        part.units.iter().position(|u| (u.owner, u.kind, u.at) == *c).is_some_and(|i| part.bears().iter().any(|r| r.unit == i && y.places.contains(&r.place)))
    });
    bears_on(a, b) || bears_on(b, a)
}

/// Frames in contact are joined (their seeds united, the closure taken again) until no two are in contact.
pub fn join_in_contact(st: &Static, part: &PositionPart, frames: Vec<Frame>) -> Vec<Result<Frame, FrameErr>> {
    let mut frames = frames;
    'again: loop {
        for i in 0..frames.len() {
            for j in i + 1..frames.len() {
                if in_contact(&frames[i], &frames[j], part) {
                    let seed = union_seed(&frames[i].seed, &frames[j].seed);
                    match Frame::from_part(st, part, &seed) {
                        Ok(f) => { frames[i] = f; frames.remove(j); continue 'again; }
                        Err(e) => { let mut out: Vec<Result<Frame, FrameErr>> = frames.into_iter().enumerate().filter(|(k, _)| *k != i && *k != j).map(|(_, f)| Ok(f)).collect(); out.push(Err(e)); return out; }
                    }
                }
            }
        }
        break;
    }
    frames.into_iter().map(Ok).collect()
}

pub fn union_seed(a: &Seed, b: &Seed) -> Seed {
    let mut units = a.units.clone();
    for c in &b.units { if !units.contains(c) { units.push(*c); } }
    units.sort();
    let mut places = a.places.clone();
    for p in &b.places { if !places.contains(p) { places.push(*p); } }
    places.sort();
    Seed { units, places }
}

/// THE SEEDS OF CONCEPT MATCHES on one board of the matcher: for each embedding of 1..=`max_size` relation nodes whose key (at any of `levels`) is in `want` (hash -> concept number), the units and the
/// empty places the embedding binds. A key whose embedding is its own swap (sign 0) is skipped as 78.54 skipped it.
pub fn seeds_of_board(b: &Board, part: &PositionPart, max_size: usize, levels: &[u8], want: &FxHashMap<u64, u32>) -> Vec<(u32, Seed)> { seeds_of_board_until(b, part, max_size, levels, want, None).0 }

/// PLAN 79.09 B: `seeds_of_board` that stops matching keys at `stop` (the clock of the step); the second value says the clock cut the matching (the seeds after it were not looked for: a gap).
/// PLAN 79.14 B: the concept matches of ONE connected set of atoms `its` (the body of `seeds_of_board_until`'s loop, one place): every key of the set at the `levels` that is wanted and not its own swap gives (concept, seed); the seeds not seen before go into `out`, and every match of the set is returned
/// (so that a caller that carries matches keeps the ones another embedding of the same seed already found).
pub fn seeds_of_set(b: &Board, part: &PositionPart, its: &[usize], levels: &[u8], want: &FxHashMap<u64, u32>, seen: &mut FxHashSet<(u32, Vec<Cell>, Vec<Place>)>, out: &mut Vec<(u32, Seed)>) -> Vec<(u32, Seed)> {
    let mut all = vec![];
    // (the hash and the sign without a heap allocation when the levels are at most four, the same values as `keys`)
    let quick = levels.len() <= 4;
    let (arr, slow): ([(u64, i8); 4], Vec<(u64, i8)>) = if quick { (b.key_hashes(its, levels), vec![]) } else { ([(0, 0); 4], b.keys(its, levels).iter().map(|k| (k.hash, k.sign)).collect()) };
    let hs: &[(u64, i8)] = if quick { &arr[..levels.len()] } else { &slow };
    for &(hash, sign) in hs {
        if sign == 0 { continue; }
        let Some(&c) = want.get(&hash) else { continue };
        let (mut us, mut ps): (Vec<usize>, Vec<Place>) = (vec![], vec![]);
        for &ix in its {
            let it = &b.items[ix];
            us.push(it.actor as usize);
            match it.target { Obj::Unit(u) => us.push(u as usize), Obj::Place(p) => ps.push(p) }
            if it.between >= 0 { us.push(it.between as usize); }
        }
        us.sort_unstable(); us.dedup(); ps.sort_unstable(); ps.dedup();
        let cells: Vec<Cell> = { let mut v: Vec<Cell> = us.iter().map(|&i| cell_of(part, i)).collect(); v.sort(); v };
        let sd = Seed { units: cells.clone(), places: ps.clone() };
        if seen.insert((c, cells, ps)) { out.push((c, sd.clone())); }
        all.push((c, sd));
    }
    all
}

/// PLAN 79.09 B: `seeds_of_board` that stops matching keys at `stop` (the clock of the step); the second value says the clock cut the matching (the seeds after it were not looked for: a gap).
pub fn seeds_of_board_until(b: &Board, part: &PositionPart, max_size: usize, levels: &[u8], want: &FxHashMap<u64, u32>, stop: Option<std::time::Instant>) -> (Vec<(u32, Seed)>, bool) {
    let mut cut = false;
    let mut out: Vec<(u32, Seed)> = vec![];
    let mut seen: FxHashSet<(u32, Vec<Cell>, Vec<Place>)> = FxHashSet::default();
    for size in 1..=max_size {
        b.connected_sets(size, &mut |its| {
            if cut || stop.is_some_and(|t| std::time::Instant::now() >= t) { cut = true; return; }
            seeds_of_set(b, part, its, levels, want, &mut seen, &mut out);
        });
    }
    (out, cut)
}

// ------------------------------------------------------------------------------------------------------------------------------------------------------------------ the reading

#[derive(Debug, Clone, Copy, Default)]
pub struct Stats {
    /// Positions read (the fuse's count).
    pub nodes: u64,
    /// Lines the fuse (or a set past `MAX_PAIRS`) ended UNREAD.
    pub unread: u64,
    /// Lines the depth bound ended.
    pub bounded: u64,
    /// Nodes where the side to move had neither a move inside the frame nor a pass.
    pub stuck: u64,
    /// Lines ended by a LEAVE option.
    pub left: u64,
    /// Passes refused because no move outside the frame leaves its projection as it was.
    pub pass_unrealisable: u64,
    /// Passes the game itself refused (a goal unit under threat, game over).
    pub pass_refused: u64,
    /// Passes a side made because it had no frame move and its budget allowed none (a side must move).
    pub forced_pass: u64,
    /// Moves of a unit that was not tracked (a taking that leaves the victim's place, or a change of kind) read as OTHER by the tracker: not guessed.
    pub others: u64,
}

impl Stats {
    pub fn add(&mut self, o: &Stats) {
        self.nodes += o.nodes; self.unread += o.unread; self.bounded += o.bounded; self.stuck += o.stuck; self.left += o.left;
        self.pass_unrealisable += o.pass_unrealisable; self.pass_refused += o.pass_refused; self.forced_pass += o.forced_pass; self.others += o.others;
    }
}

type Tracked = Vec<(Cell, u8)>;

struct Child<G: Game> { state: G::State, tracked: Tracked, lost: u16, ends: (Place, Place), taken: Vec<(u8, u8)> }

struct Expansion<G: Game> { mover: Option<Role>, inside: Vec<Rc<Child<G>>>, pass: Option<G::State>, leaves: bool }

type MemoKey = (u64, u64, Role, [u8; 2], u8);

/// The forceable set of one reading: the unbeaten guaranteed outcomes of the reading side and what the reading cost.
#[derive(Debug, Clone)]
pub struct Forceable { pub pairs: Vec<Pair>, pub stats: Stats }

pub struct Reader<'a, G: Game + Placed> {
    g: &'a G,
    st: &'a Static,
    pub frame: &'a Frame,
    pset: PlaceSet,
    pub owner_mask: [u16; 2],
    pub depth: u8,
    pub fuse: u64,
    exp: FxHashMap<(u64, u64), Rc<Expansion<G>>>,
    proj: FxHashMap<u64, Rc<Vec<Cell>>>,
    memo: FxHashMap<MemoKey, Vec<Pair>>,
    stats: Stats,
    nodes_left: u64,
}

fn hash_tracked(t: &Tracked) -> u64 {
    let mut h = rustc_hash::FxHasher::default();
    for (c, id) in t { c.hash(&mut h); h.write_u8(*id); }
    h.finish()
}

fn sorted_cells<G: Placed>(g: &G, s: &G::State) -> Vec<Cell> { let mut c = g.cells(s); c.sort(); c }

impl<'a, G: Game + Placed> Reader<'a, G> {
    pub fn new(g: &'a G, st: &'a Static, frame: &'a Frame) -> Reader<'a, G> {
        let pset = PlaceSet::from_places(st.names.places.len(), frame.places.iter().copied());
        let mut owner_mask = [0u16; 2];
        for (i, c) in frame.members.iter().enumerate() { owner_mask[c.0.min(1)] |= 1 << i; }
        Reader { g, st, frame, pset, owner_mask, depth: (frame.members.len() + 1) as u8, fuse: FUSE_POSITIONS, exp: FxHashMap::default(), proj: FxHashMap::default(), memo: FxHashMap::default(), stats: Stats::default(), nodes_left: FUSE_POSITIONS }
    }

    pub fn root_tracked(&self) -> Tracked { self.frame.members.iter().enumerate().map(|(i, c)| (*c, i as u8)).collect() }

    /// The frame's projection on a position: the cells of its members, from the closure of its places on that position.
    fn projection(&mut self, s: &G::State) -> Rc<Vec<Cell>> {
        let k = self.g.key(s);
        if let Some(p) = self.proj.get(&k) { return p.clone(); }
        let part = self.st.position(self.g, s);
        let m = closure(&part, &self.pset, &[]);
        let mut cells: Vec<Cell> = m.iter().map(|&i| cell_of(&part, i)).collect();
        cells.sort();
        let r = Rc::new(cells);
        self.proj.insert(k, r.clone());
        r
    }

    fn tracked_after(&mut self, prev: &[Cell], next: &[Cell], tracked: &Tracked) -> (Tracked, u16, Vec<(u8, u8)>) {
        let (steps, _) = step_between(prev, next);
        let (mut out, mut lost, mut taken) = (Vec::with_capacity(tracked.len()), 0u16, vec![]);
        let mut victims: Vec<(u8, Place)> = vec![];
        for (c, id) in tracked {
            let i = prev.binary_search(c).expect("a tracked unit stands on the board");
            match steps[i] {
                Step::Alive(j) => out.push((next[j as usize], *id)),
                Step::Taken(_) => { lost |= 1 << id; victims.push((*id, c.2)); }
                Step::Other => { self.stats.others += 1; }
            }
        }
        for (v, at) in victims {
            let captor = out.iter().find(|(c, _)| c.2 == at).map_or(255, |(_, id)| *id);
            taken.push((v, captor));
        }
        out.sort();
        (out, lost, taken)
    }

    fn expand(&mut self, s: &G::State, tracked: &Tracked) -> Rc<Expansion<G>> {
        let key = (self.g.key(s), hash_tracked(tracked));
        if let Some(e) = self.exp.get(&key) { return e.clone(); }
        let g = self.g;
        let Some(mover) = g.mover(s) else {
            let e = Rc::new(Expansion { mover: None, inside: vec![], pass: None, leaves: false });
            self.exp.insert(key, e.clone());
            return e;
        };
        let prev = sorted_cells(g, s);
        let mut inside: Vec<Rc<Child<G>>> = vec![];
        let mut leaves = false;
        let mut outsiders: Vec<G::Move> = vec![];
        for m in g.moves(s) {
            let (from, to) = g.move_ends(s, &m);
            let tr = tracked.iter().find(|(c, _)| c.0 == mover && c.2 == from);
            match tr {
                Some(_) => {
                    if self.pset.contains(to) {
                        let s2 = g.apply(s, &m);
                        let next = sorted_cells(g, &s2);
                        let (t2, lost, taken) = self.tracked_after(&prev, &next, tracked);
                        inside.push(Rc::new(Child { state: s2, tracked: t2, lost, ends: (from, to), taken }));
                    } else { leaves = true; }
                }
                None => outsiders.push(m),
            }
        }
        // the moves of units that are not tracked: a pass needs one that leaves the projection as it was; an arrival on the frame is a LEAVE
        let here = self.projection(s);
        let (mut neutral, mut changing) = (false, leaves);
        for m in &outsiders {
            if neutral && changing { break; }
            let (from, to) = g.move_ends(s, m);
            let is_neutral = if self.pset.contains(to) || self.pset.contains(from) { false } else {
                let s2 = g.apply(s, m);
                *self.projection(&s2) == *here
            };
            if is_neutral { neutral = true } else { changing = true }
        }
        let pass = if neutral { g.pass(s) } else { None };
        if !neutral { self.stats.pass_unrealisable += 1; }
        if neutral && pass.is_none() { self.stats.pass_refused += 1; }
        let e = Rc::new(Expansion { mover: Some(mover), inside, pass, leaves: changing });
        self.exp.insert(key, e.clone());
        e
    }

    /// The guarantees from a node: the unbeaten outcomes of the reading side `xr` from here on (losses incurred from here, not before).
    fn solve(&mut self, s: &G::State, tracked: &Tracked, xr: Role, bud: [u8; 2], depth: u8) -> Vec<Pair> {
        if self.nodes_left == 0 { self.stats.unread += 1; return vec![Pair::BOTTOM]; }
        let tk = hash_tracked(tracked);
        let mk: MemoKey = (self.g.key(s), tk, xr, bud, depth);
        if let Some(v) = self.memo.get(&mk) { return v.clone(); }
        self.nodes_left -= 1;
        self.stats.nodes += 1;
        let before_unread = self.stats.unread;
        let e = self.expand(s, tracked);
        let Some(mover) = e.mover else { return vec![Pair::ZERO]; };
        if depth == 0 { self.stats.bounded += 1; return vec![Pair::ZERO]; }
        let other = 1 - xr.min(1);
        let (own_mask, opp_mask) = (self.owner_mask[xr.min(1)], self.owner_mask[other]);
        let mut sets: Vec<Vec<Pair>> = vec![];
        for ch in e.inside.iter() {
            let sub = self.solve(&ch.state, &ch.tracked, xr, bud, depth - 1);
            sets.push(sub.into_iter().map(|p| if p == Pair::BOTTOM { p } else { p.plus(ch.lost & own_mask, ch.lost & opp_mask) }).collect());
        }
        if bud[mover.min(1)] > 0 {
            if let Some(sp) = &e.pass {
                let mut left = bud;
                if left[mover.min(1)] != ALL { left[mover.min(1)] -= 1; }
                sets.push(self.solve(sp, tracked, xr, left, depth - 1));
            }
        }
        if mover != xr && e.leaves { self.stats.left += 1; sets.push(vec![Pair::ZERO]); }
        if sets.is_empty() {
            // a side with no move inside the frame and no pass its budget allows must still move: it PASSES if it can (a forced pass, counted), else the line is STUCK. A stuck line of the side read ends
            // with the worst outcome for it (it has no frame move, no claim is made on it); a stuck line of the other side ends as it stands
            if let Some(sp) = &e.pass {
                self.stats.forced_pass += 1;
                sets.push(self.solve(sp, tracked, xr, bud, depth - 1));
            }
        }
        let result = if sets.is_empty() {
            self.stats.stuck += 1;
            if mover == xr { vec![Pair::BOTTOM] } else { vec![Pair::ZERO] }
        } else if mover == xr {
            let all: Vec<Pair> = sets.into_iter().flatten().collect();
            let u = unbeaten(all);
            if u.len() > MAX_PAIRS { self.stats.unread += 1; vec![Pair::BOTTOM] } else { u }
        } else {
            match meet_all(&sets) { Some(v) => v, None => { self.stats.unread += 1; vec![Pair::BOTTOM] } }
        };
        if self.stats.unread == before_unread { self.memo.insert(mk, result.clone()); }
        result
    }

    /// What `xr` can force from `s` (the root position or the position after a pass), with `bud` passes for each side.
    pub fn forceable_at(&mut self, s: &G::State, tracked: &Tracked, xr: Role, bud: [u8; 2]) -> Forceable {
        self.stats = Stats::default();
        self.nodes_left = self.fuse;
        let d = self.depth;
        let pairs = self.solve(s, tracked, xr, bud, d);
        Forceable { pairs, stats: self.stats }
    }

    /// The root position as it stands, or the position after a pass of the side to move (None when it cannot pass there, see the module text B).
    pub fn root(&mut self, s: &G::State, flipped: bool) -> Option<G::State> {
        if !flipped { return Some(s.clone()); }
        let t = self.root_tracked();
        self.expand(s, &t).pass.clone()
    }

    pub fn forceable(&mut self, s: &G::State, xr: Role, flipped: bool, bud: [u8; 2]) -> Option<Forceable> {
        let root = self.root(s, flipped)?;
        let t = self.root_tracked();
        Some(self.forceable_at(&root, &t, xr, bud))
    }

    /// The frame's units that stand and bear in the position, as the legal moves of the side to move that start from a member and end outside the frame (a LEAVE) -- member ids.
    pub fn leaves_of(&mut self, s: &G::State) -> Vec<u8> {
        let g = self.g;
        let Some(mover) = g.mover(s) else { return vec![] };
        let t = self.root_tracked();
        let mut out = vec![];
        for m in g.moves(s) {
            let (from, to) = g.move_ends(s, &m);
            if let Some((_, id)) = t.iter().find(|(c, _)| c.0 == mover && c.2 == from) { if !self.pset.contains(to) && !out.contains(id) { out.push(*id); } }
        }
        out
    }

    pub fn stats(&self) -> Stats { self.stats }
}

// ------------------------------------------------------------------------------------------------------------------------------------------------------------------ claims

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ClaimKind { TakeFree, TakeAny, Safe }

impl ClaimKind { pub fn word(self) -> &'static str { match self { ClaimKind::TakeFree => "taken, none of its own lost", ClaimKind::TakeAny => "taken", ClaimKind::Safe => "not taken" } } }

/// Whether `pairs` (the guarantees of the reading side) carry the claim about member `u`.
pub fn holds(pairs: &[Pair], kind: ClaimKind, u: u8) -> bool {
    let bit = 1u16 << u;
    pairs.iter().any(|p| match kind { ClaimKind::TakeFree => p.lo & bit != 0 && p.lx == 0, ClaimKind::TakeAny => p.lo & bit != 0, ClaimKind::Safe => p.lx & bit == 0 })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status { Forced, Threat, Waiting, Conditional, Not }

impl Status { pub fn word(self) -> &'static str { match self { Status::Forced => "forced", Status::Threat => "threat", Status::Waiting => "waiting", Status::Conditional => "conditional", Status::Not => "no" } } }

/// One row of the tempo table: who is to move at the root (`flipped`: the side that was not to move, the real mover having passed), the passes each side may spend, the side read and its guarantees.
#[derive(Debug, Clone)]
pub struct TempoRow { pub flipped: bool, pub bud: [u8; 2], pub reading: Role, pub pairs: Vec<Pair>, pub stats: Stats }

/// A claim and what makes it hold.
#[derive(Debug, Clone)]
pub struct ClaimRow { pub reading: Role, pub kind: ClaimKind, pub unit: u8, pub status: Status, pub condition: Option<Condition> }

/// A condition: a unit of the real mover arrives on a place (the move that makes it so is read from `from` to `to`; `kind` the arriving unit's kind, `arrivals` how many such moves make the claim hold).
#[derive(Debug, Clone)]
pub struct Condition { pub owner: Role, pub kind: Kind, pub from: Place, pub to: Place, pub arrivals: u32 }

#[derive(Debug, Clone)]
pub struct Report {
    pub frame: Frame,
    pub mover: Role,
    pub tempo: Vec<TempoRow>,
    pub claims: Vec<ClaimRow>,
    pub arrivals_tried: u32,
    pub stats: Stats,
    pub leaves: Vec<u8>,
}

/// The sound form of the Lean lemma for the side `xr`: it spends no pass, the other side may pass at every ply.
pub fn sound_budget(xr: Role) -> [u8; 2] { let mut b = [ALL; 2]; b[xr.min(1)] = 0; b }

/// THE TEMPO TABLE of a frame on a position: every reading side, who moves first at the root and the passes each side may spend, from the set `budgets` of counts (`ALL` included).
pub fn tempo_table<G: Game + Placed>(rd: &mut Reader<G>, s: &G::State, budgets: &[u8]) -> Vec<TempoRow> {
    let mut out = vec![];
    for reading in 0..2usize {
        for flipped in [false, true] {
            for &x0 in budgets { for &x1 in budgets {
                let bud = [x0, x1];
                if let Some(f) = rd.forceable(s, reading, flipped, bud) { out.push(TempoRow { flipped, bud, reading, pairs: f.pairs, stats: f.stats }); }
            } }
        }
    }
    out
}

/// THE CLAIMS of a frame on a position, with their status (module text, E). `conditional`: also try each move of the side to move that arrives on or onto the frame.
pub fn claims_of<G: Game + Placed>(g: &G, st: &Static, frame: &Frame, s: &G::State, conditional: bool, fuse: u64) -> Report {
    let mut rd = Reader::new(g, st, frame);
    rd.fuse = fuse;
    let mover = g.mover(s).unwrap_or(0);
    let mut stats = Stats::default();
    let mut tempo = vec![];
    let mut claims: Vec<ClaimRow> = vec![];
    for reading in 0..2usize {
        let sound = sound_budget(reading);
        let now = rd.forceable(s, reading, false, sound);
        let flip = if reading != mover { rd.forceable(s, reading, true, sound) } else { None };
        let wait = if reading == mover { let mut b = sound; b[reading] = 1; rd.forceable(s, reading, false, b) } else { None };
        for (f, flipped, bud) in [(&now, false, sound), (&flip, true, sound)] { if let Some(f) = f { tempo.push(TempoRow { flipped, bud, reading, pairs: f.pairs.clone(), stats: f.stats }); stats.add(&f.stats); } }
        if let Some(f) = &wait { let mut b = sound; b[reading] = 1; tempo.push(TempoRow { flipped: false, bud: b, reading, pairs: f.pairs.clone(), stats: f.stats }); stats.add(&f.stats); }
        for (id, c) in frame.members.iter().enumerate() {
            let own = c.0.min(1) == reading;
            let kinds: &[ClaimKind] = if own { &[ClaimKind::Safe] } else { &[ClaimKind::TakeFree, ClaimKind::TakeAny] };
            for &kind in kinds {
                let h = |f: &Option<Forceable>| f.as_ref().is_some_and(|f| holds(&f.pairs, kind, id as u8));
                let status = if h(&now) { Status::Forced } else if h(&flip) { Status::Threat } else if h(&wait) { Status::Waiting } else { Status::Not };
                claims.push(ClaimRow { reading, kind, unit: id as u8, status, condition: None });
            }
        }
    }
    let leaves = rd.leaves_of(s);
    // conditional claims: the side to move tries each move that arrives on or onto the frame (a unit that was not a member and is after it)
    let mut arrivals_tried = 0;
    if conditional {
        if let Some(mv) = g.mover(s) {
            let here_cells: Vec<Cell> = frame.members.clone();
            let prev = sorted_cells(g, s);
            let mut seen: FxHashSet<(Kind, Place)> = FxHashSet::default();
            for m in g.moves(s) {
                let (from, to) = g.move_ends(s, &m);
                let Some(unit) = prev.iter().find(|c| c.0 == mv && c.2 == from).copied() else { continue };
                if here_cells.contains(&unit) { continue; }                              // a member's own move is inside or a leave, not an arrival
                let s2 = g.apply(s, &m);
                let part2 = st.position(g, &s2);
                let pset = PlaceSet::from_places(st.names.places.len(), frame.places.iter().copied());
                let m2 = closure(&part2, &pset, &[]);
                let mut cells2: Vec<Cell> = m2.iter().map(|&i| cell_of(&part2, i)).collect();
                cells2.sort();
                // is it an arrival? the unit stands among the members after the move
                let next = sorted_cells(g, &s2);
                let arrived = next.iter().find(|c| c.0 == mv && c.2 == to && c.1 == unit.1).copied();
                let Some(arrived) = arrived else { continue };
                if !cells2.contains(&arrived) { continue; }
                if cells2.len() > MAX_MEMBERS { continue; }
                if !seen.insert((unit.1, to)) {
                    // the same arrival by another unit of the kind is the same condition: counted, not read again
                    for cr in claims.iter_mut() { if let Some(c) = &mut cr.condition { if c.kind == unit.1 && c.to == to && c.owner == mv { c.arrivals += 1; } } }
                    continue;
                }
                // the frame after the arrival: the places as they were, the members as they now stand
                let fn_ = Frame { seed: frame.seed.clone(), places: frame.places.clone(), members: cells2.clone(), id: frame.id.clone() };
                arrivals_tried += 1;
                let (steps, _) = step_between(&prev, &next);
                // original member -> its id in the frame after the move (None when taken)
                let map: Vec<Option<u8>> = frame.members.iter().map(|c| {
                    let i = prev.binary_search(c).ok()?;
                    match steps[i] { Step::Alive(j) => fn_.id_of(&next[j as usize]), _ => None }
                }).collect();
                let mut rd2 = Reader::new(g, st, &fn_);
                rd2.fuse = fuse;
                let t2: Tracked = fn_.members.iter().enumerate().map(|(i, c)| (*c, i as u8)).collect();
                let sound = sound_budget(mv);
                let f = rd2.forceable_at(&s2, &t2, mv, sound);
                stats.add(&f.stats);
                for cr in claims.iter_mut() {
                    if cr.reading != mv || cr.status != Status::Not || cr.condition.is_some() { continue; }
                    let Some(nid) = map[cr.unit as usize] else { continue };       // taken by the move itself: not a claim the condition makes
                    if holds(&f.pairs, cr.kind, nid) { cr.status = Status::Conditional; cr.condition = Some(Condition { owner: mv, kind: unit.1, from, to, arrivals: 1 }); }
                }
            }
        }
    }
    Report { frame: frame.clone(), mover, tempo, claims, arrivals_tried, stats, leaves }
}

// ------------------------------------------------------------------------------------------------------------------------------------------------------------------ layer 3

/// LAYER 3 NODES of one report (SYSTEM_SPEC 11): node kinds `frame`, `claim`, `condition`, ids that are their definitions, and links to the layer-1 nodes (units and places by their ids).
///   frame:<hash>                                         has_unit -> unit:<role>:<kind>:<place>, has_place -> place:<text>
///   claim:<frame>:<role>:<claim>:<unit id>               in_frame -> frame, about_unit -> unit, at_place -> the place the unit stands on     (only a claim that holds in some form)
///   condition:<claim>:arrive:<kind>@<place>              condition_of -> claim, arrives_at -> place, arriving_kind -> kind:<word>
/// A claim's own `status` word is carried as the third element of its `status` link-less record in `claim_status`.
#[derive(Debug, Clone, Default)]
pub struct L3 { pub nodes: Vec<(&'static str, String)>, pub links: Vec<(&'static str, String, String)>, pub claim_status: Vec<(String, &'static str)> }

pub fn claim_kind_word(k: ClaimKind) -> &'static str { match k { ClaimKind::TakeFree => "take_free", ClaimKind::TakeAny => "take_any", ClaimKind::Safe => "safe" } }

pub fn layer3(st: &Static, r: &Report) -> L3 {
    let mut out = L3::default();
    let f = &r.frame;
    out.nodes.push(("frame", f.id.clone()));
    for c in &f.members { out.links.push(("has_unit", f.id.clone(), unit_id(st, c))); }
    for p in &f.places { out.links.push(("has_place", f.id.clone(), format!("place:{}", st.names.places[*p as usize]))); }
    for c in &r.claims {
        if c.status == Status::Not { continue; }
        let m = &f.members[c.unit as usize];
        let cid = format!("claim:{}:{}:{}:{}", f.id, st.names.roles[c.reading], claim_kind_word(c.kind), unit_id(st, m));
        out.nodes.push(("claim", cid.clone()));
        out.claim_status.push((cid.clone(), c.status.word()));
        out.links.push(("in_frame", cid.clone(), f.id.clone()));
        out.links.push(("about_unit", cid.clone(), unit_id(st, m)));
        out.links.push(("at_place", cid.clone(), format!("place:{}", st.names.places[m.2 as usize])));
        if let Some(cd) = &c.condition {
            let did = format!("condition:{}:arrive:{}@{}", cid, st.names.kinds[cd.kind as usize], st.names.places[cd.to as usize]);
            out.nodes.push(("condition", did.clone()));
            out.links.push(("condition_of", did.clone(), cid.clone()));
            out.links.push(("arrives_at", did.clone(), format!("place:{}", st.names.places[cd.to as usize])));
            out.links.push(("arriving_kind", did, format!("kind:{}", st.names.kinds[cd.kind as usize])));
        }
    }
    out
}

// ------------------------------------------------------------------------------------------------------------------------------------------------------------------ the table of outcomes

/// A member's fate in one outcome: alive standing where, or taken by a member (255: not a member) at a ply.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Fate { Alive(Place), Taken { by: u8, ply: u8 } }

#[derive(Debug, Clone)]
pub struct OutcomeRow { pub pair: Pair, pub fates: Vec<Fate>, pub line: Vec<String>, pub end: &'static str }

/// THE TABLE OF OUTCOMES of the frame with the line that reaches each: every sequence of moves inside the frame and passes (each side at most `passes` of them), the side to move first at the root, read
/// by the same fuse; the distinct outcomes (by fate record) with the first line that reaches each. `us` is the side whose losses are `lx` in the pairs. An unread line is counted, not listed.
pub fn outcomes_table<G: Game + Placed>(g: &G, st: &Static, frame: &Frame, s: &G::State, us: Role, passes: u8, fuse: u64) -> (Vec<OutcomeRow>, Stats) {
    let mut rd = Reader::new(g, st, frame);
    rd.nodes_left = fuse;
    let mut rows: Vec<OutcomeRow> = vec![];
    let n = frame.members.len();
    let t = rd.root_tracked();
    let mut fates: Vec<Fate> = frame.members.iter().map(|c| Fate::Alive(c.2)).collect();
    let mut line: Vec<String> = vec![];
    let depth = rd.depth;
    let mut bud = [passes; 2];
    walk(&mut rd, s, &t, us, &mut bud, depth, 0, &mut fates, &mut line, &mut rows, n);
    (rows, rd.stats)
}

#[allow(clippy::too_many_arguments)]
fn walk<G: Game + Placed>(rd: &mut Reader<G>, s: &G::State, tracked: &Tracked, us: Role, bud: &mut [u8; 2], depth: u8, ply: u8, fates: &mut Vec<Fate>, line: &mut Vec<String>, rows: &mut Vec<OutcomeRow>, n: usize) {
    let record = |rd: &Reader<G>, fates: &Vec<Fate>, line: &Vec<String>, rows: &mut Vec<OutcomeRow>, end: &'static str| {
        if rows.len() >= MAX_OUTCOMES { return; }
        let (mut lx, mut lo) = (0u16, 0u16);
        for (i, f) in fates.iter().enumerate() { if matches!(f, Fate::Taken { .. }) { if rd.owner_mask[us.min(1)] & (1 << i) != 0 { lx |= 1 << i } else { lo |= 1 << i } } }
        if !rows.iter().any(|r| r.fates == *fates) { rows.push(OutcomeRow { pair: Pair { lx, lo }, fates: fates.clone(), line: line.clone(), end }); }
    };
    if rd.nodes_left == 0 { rd.stats.unread += 1; return; }
    rd.nodes_left -= 1;
    rd.stats.nodes += 1;
    let e = rd.expand(s, tracked);
    let Some(mover) = e.mover else { record(rd, fates, line, rows, "ended"); return; };
    if depth == 0 { rd.stats.bounded += 1; record(rd, fates, line, rows, "bounded"); return; }
    let mut any = false;
    for ch in e.inside.iter() {
        any = true;
        let saved = fates.clone();
        for (v, by) in &ch.taken { fates[*v as usize] = Fate::Taken { by: *by, ply: ply + 1 }; }
        for (c, id) in &ch.tracked { if let Fate::Alive(_) = fates[*id as usize] { fates[*id as usize] = Fate::Alive(c.2); } }
        line.push(rd.g.move_name(s, &rd.g.find_step(s, ch.ends.0, ch.ends.1).expect("the move is legal")));
        walk(rd, &ch.state, &ch.tracked, us, bud, depth - 1, ply + 1, fates, line, rows, n);
        line.pop();
        *fates = saved;
    }
    if bud[mover.min(1)] > 0 {
        if let Some(sp) = e.pass.clone() {
            any = true;
            if bud[mover.min(1)] != ALL { bud[mover.min(1)] -= 1; }
            line.push("pass".into());
            walk(rd, &sp, tracked, us, bud, depth - 1, ply + 1, fates, line, rows, n);
            line.pop();
            if bud[mover.min(1)] != ALL { bud[mover.min(1)] += 1; }
        }
    }
    if e.leaves { any = true; line.push("leaves the frame".into()); record(rd, fates, line, rows, "left"); line.pop(); }
    if !any {
        match e.pass.clone() {
            Some(sp) => { rd.stats.forced_pass += 1; line.push("pass (forced)".into()); walk(rd, &sp, tracked, us, bud, depth - 1, ply + 1, fates, line, rows, n); line.pop(); }
            None => { rd.stats.stuck += 1; record(rd, fates, line, rows, "stuck"); }
        }
    }
}

// ------------------------------------------------------------------------------------------------------------------------------------------------------------------ two frames

/// The forceable set of two frames NOT in contact, from each frame's own set (the sound form for each: the side read spends no pass in a frame, the other may pass at every ply -- its moves in the
/// other frame are its passes here), the guarantees united. The pairs are over the ids of the two frames side by side: frame `a`'s ids as they are, frame `b`'s shifted by `a`'s member count. A
/// COMPOSITION THAT IS CHECKED, NOT PROVED (module text, F).
pub fn compose<G: Game + Placed>(g: &G, st: &Static, a: &Frame, b: &Frame, s: &G::State, reading: Role, fuse: u64) -> Option<Vec<Pair>> {
    let na = a.members.len();
    if na + b.members.len() > 16 { return None; }
    let mut ra = Reader::new(g, st, a); ra.fuse = fuse;
    let mut rb = Reader::new(g, st, b); rb.fuse = fuse;
    let bud = sound_budget(reading);
    let fa = ra.forceable(s, reading, false, bud)?;
    let fb = rb.forceable(s, reading, false, bud)?;
    let mut v = vec![];
    for p in &fa.pairs { for q in &fb.pairs { v.push(Pair { lx: p.lx | (q.lx << na), lo: p.lo | (q.lo << na) }); } }
    Some(unbeaten(v))
}

/// How two sets of guarantees compare.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cmp { Equal, FirstMore, SecondMore, Incomparable }

impl Cmp { pub fn word(self) -> &'static str { match self { Cmp::Equal => "equal", Cmp::FirstMore => "composed more", Cmp::SecondMore => "joint more", Cmp::Incomparable => "incomparable" } } }

fn dominates(a: &[Pair], b: &[Pair]) -> bool { b.iter().all(|q| a.iter().any(|p| p.at_least(*q))) }

pub fn compare_sets(a: &[Pair], b: &[Pair]) -> Cmp {
    match (dominates(a, b), dominates(b, a)) { (true, true) => Cmp::Equal, (true, false) => Cmp::FirstMore, (false, true) => Cmp::SecondMore, _ => Cmp::Incomparable }
}

/// The union frame's members in the composed id space (a's ids, then b's shifted), as a map from the joint frame's ids; None when the joint frame holds a member of neither.
pub fn joint_ids(a: &Frame, b: &Frame, joint: &Frame) -> Option<Vec<u8>> {
    joint.members.iter().map(|c| a.id_of(c).or_else(|| b.id_of(c).map(|i| i + a.members.len() as u8))).collect()
}

/// Re-index a pair of the joint frame's ids into the composed id space.
pub fn reindex(p: Pair, map: &[u8]) -> Pair {
    let (mut lx, mut lo) = (0u16, 0u16);
    for (i, &m) in map.iter().enumerate() { if p.lx & (1 << i) != 0 { lx |= 1 << m } if p.lo & (1 << i) != 0 { lo |= 1 << m } }
    Pair { lx, lo }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(lx: u16, lo: u16) -> Pair { Pair { lx, lo } }

    #[test]
    fn outcomes_compare_by_inclusion_and_never_by_a_number() {
        // losing nothing while taking a unit beats losing that unit while taking the same unit; one that takes more and loses the same is better; a trade of two different units is neither
        assert!(p(0b0, 0b1).at_least(p(0b1, 0b1)));
        assert!(!p(0b1, 0b1).at_least(p(0b0, 0b1)));
        assert!(p(0b1, 0b11).at_least(p(0b1, 0b1)));
        let (a, b) = (p(0b01, 0b0), p(0b10, 0b0));
        assert!(!a.at_least(b) && !b.at_least(a));
        assert_eq!(unbeaten(vec![a, b, p(0b11, 0b0)]), vec![a, b]);
        assert!(Pair::ZERO.at_least(Pair::BOTTOM) && !Pair::BOTTOM.at_least(Pair::ZERO));
    }

    #[test]
    fn the_guarantee_at_the_other_sides_node_is_the_meet_of_its_replies() {
        // the other side may answer with (we lose a, they lose nothing) or (we lose nothing, they lose b): we are guaranteed only what holds in both
        let sets = vec![vec![p(0b1, 0b0)], vec![p(0b0, 0b10)]];
        assert_eq!(meet_all(&sets).unwrap(), vec![p(0b1, 0b0)]);
        // two choices of ours on one reply: both guarantees survive when neither beats the other
        let sets2 = vec![vec![p(0b1, 0b1), p(0b0, 0b0)], vec![p(0b1, 0b1), p(0b0, 0b0)]];
        let m = meet_all(&sets2).unwrap();
        assert!(m.contains(&p(0b1, 0b1)) && m.contains(&p(0b0, 0b0)) && m.len() == 2);
    }

    #[test]
    fn a_claim_is_a_statement_over_the_guaranteed_outcomes() {
        let pairs = vec![p(0b0, 0b10), p(0b1, 0b10)];
        assert!(holds(&pairs, ClaimKind::TakeFree, 1));
        assert!(holds(&pairs, ClaimKind::TakeAny, 1));
        assert!(!holds(&[p(0b1, 0b10)], ClaimKind::TakeFree, 1));
        assert!(holds(&[p(0b1, 0b10)], ClaimKind::TakeAny, 1));
        assert!(holds(&[p(0b1, 0b0)], ClaimKind::Safe, 1) && !holds(&[p(0b1, 0b0)], ClaimKind::Safe, 0));
        assert!(!holds(&[Pair::BOTTOM], ClaimKind::Safe, 0) && !holds(&[Pair::BOTTOM], ClaimKind::TakeAny, 0));
    }

    #[test]
    fn two_sets_of_guarantees_compare() {
        assert_eq!(compare_sets(&[p(0, 1)], &[p(0, 1)]), Cmp::Equal);
        assert_eq!(compare_sets(&[p(0, 1)], &[p(1, 1)]), Cmp::FirstMore);
        assert_eq!(compare_sets(&[p(1, 1)], &[p(0, 1)]), Cmp::SecondMore);
        assert_eq!(compare_sets(&[p(1, 0)], &[p(0b10, 0)]), Cmp::Incomparable);
    }

    #[test]
    fn ids_are_reindexed_into_the_composed_space() {
        assert_eq!(reindex(p(0b101, 0b010), &[0, 3, 1]), p(0b0011, 0b1000));
    }
}
