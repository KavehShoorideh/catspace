//! PLAN 78.57 THE FRAME READ BY QUESTIONS, on the layer-1 graph. The repair of the 78.55 read (frames.rs, which stays as the exhaustive reference). Game-free: units are (owner, kind, place) as the game
//! gives them; a kind is a raw index, a place a number; no worth table, no number is made of an outcome.
//!
//! A'. THE FRAME IS BOUND TO UNITS. A frame is made from a seed: its identity is the seed's UNITS (`frame:<hash of the seed units>`). Every unit of the board has an identity from the root position (its
//! index among the root's sorted cells) that follows it through every capture and move. At any position the frame's PLACES are derived again from where the seed's units stand and bear NOW (their own
//! places, and the seed's empty places that a seed unit still bears on), and its MEMBERS are the seed's living units, the units standing on those places, the units that bear on them, and, to a fixed
//! point, the units that bear on them through a member (the layer-1 `through` relation). A member that moves takes the frame with it; a unit that now bears on the frame joins. A frame whose derived
//! members pass `MAXQ` ends that line as BOUND (counted), never cut quietly.
//!
//! B'. THE READ BY QUESTIONS, in place of the table of all sequences. At a node the side to move is asked, in this order:
//!   Q0 what stands to be lost: its members that the other side bears on now.
//!   Q1 its candidate moves, which are only: a taking by or of a member (or onto the frame); a move of a threatened member (it steps away: the read follows it); a move that adds a bearing on a
//!      threatened member; a move that makes a new threat on a member of the other side; a move onto a place of the frame (put a unit between, or arrive). Every other legal move is NOT ASKED (counted).
//!   Q2 for each candidate the other side's answer, asked the same way from its side (a static balance is never a result until the other side has been asked from there). A candidate that loses a
//!      member and gains none against one reply is PRUNED at once (that reply is its reason, kept), and the rest of the replies are not read.
//!   STOP: at a node of the side read, the first candidate that keeps every member of that side (and a taking that gains and loses nothing is the best there is) ends the questions at that node. At a node
//!   of the other side every candidate is asked and met (a claim of ours is only sound if every answer of theirs is read, as in the Lean `frame_sound_pass`); none of theirs is dropped, except that
//!   the reading stops at their first reply that loses a member of ours for nothing (the node is then a pure loss, marked PRUNED, and grounds no claim).
//!   The side not read may also PASS (a move elsewhere that leaves the frame's derived members and places as they were: the 78.55 realisability test, unchanged); the side read passes only when it has
//!   no candidate (a forced pass, counted).
//!   THE END OF A LINE IS A REASON, NOT ONLY A DEPTH: a taking is progress and restarts the count of quiet plies; after `QUIET_PLIES` quiet plies, or at `DEPTH` plies, only takings are read on (a quiet
//!   leaf: the mover may stand pat) up to `EXT` more plies; a line ended so is marked BOUND in its pairs. The fuse is `fuse` questions per reading; a line it ends is UNREAD (the worst for the side read).
//!
//! COMPARISON stays by inclusion (frames.rs `Pair` rules, here over 64 identity bits): a pure loss is losing a member and gaining none, a pure gain the converse. An UNEQUAL trade of kinds is neither;
//! it is reported as it stands (the pair) and left open. STUCK is its own mark (`F_STUCK`), not an outcome: a stuck line never carries a claim.
//!
//! SOUNDNESS, stated against the Lean `frame_sound_pass`: the lemma reads the frame with the other side allowed every reply; it does not follow a unit that leaves the frame. A claim whose supporting
//! lines all contain a member moving off the places of the root frame carries `F_FOLLOW`; one reached while a pure-loss candidate of the side read was dropped carries `F_PRUNE_USED`; one reached through a
//! line ended by a bound carries `F_BOUND`. These are written into the claim's layer-3 node. They are claims about a LARGER read than the lemma covers and are not proved by it.
//!
//! E'. THE PRUNED BRANCH AND THE OTHER FRAME (`reopen`): a pruned root candidate is reopened only by a claim of another frame that the same move serves (the move is an arrival of that claim's condition),
//! and only if that claim still holds forced after the other side has taken here (the pruned move, its reason reply, then the side to move again). Written as layer-3 nodes `prune:<frame>:<move>:<reason>`
//! and a link `reopened_by` to the claim. NOT BUILT: a claim whose gain is a named change of structure (it needs a third coordinate in the outcome).
//!
//! D. COMPOSITION WITH THE TEMPO CHARGED (`compose_q`, for the side to move): X spends c moves in the frame it plays first; in the other frame those are c passes of X while the other side moves (`owed`).
//! Both orders are read. Checked against the joint read, not proved (SYSTEM_SPEC 12.1).
//!
//! TRAP (`trap`): one question for any unit, on quiet routes only: has it a place to go that is not a pure loss. Read with the owner to move and no forcing threat on it (the game refuses a pass otherwise:
//! then the answer is NOT QUIET, never "trapped").
use super::base_graph::{PositionPart, Static};
use super::exchange::{estimate_taking, Val};
use super::fates::{step_between, Cell, Step};
use super::frames::{claim_kind_word, unit_id, ClaimKind, FrameErr, Seed, Status};
use super::patterns::{FxHashMap, FxHashSet};
use super::{Kind, Place, PlaceSet};
use cs_game::placed::Placed;
use cs_game::{Game, Role};
use std::hash::{Hash, Hasher};
use std::rc::Rc;

/// The most members a frame may hold at a position.
pub const MAXQ: usize = 16;
/// The most members a frame may have when it is made from a seed or by joining two (a few units, not the board); a bigger one is split.
pub const ROOT_CAP: usize = 10;
/// Quiet plies (no taking) a line may run before it is read on by takings only.
pub const QUIET_PLIES: u8 = 2;
/// Plies after which only takings are read on.
pub const DEPTH: i8 = 4;
/// More plies of takings read after the bound; then the mover stands pat and the line is BOUND.
pub const EXT: i8 = 2;
/// The fuse: questions asked in one reading.
pub const FUSE_QUESTIONS: u64 = 1_500;
/// The most questions one frame may spend over all its readings (a frame past it has its remaining readings UNREAD).
pub const FRAME_FUSE: u64 = 8_000;
/// 78.60 A: the most questions the trap question may spend on one member when a line is being settled (the same as the measure's TRAP_FUSE).
pub const SETTLE_TRAP_FUSE: u64 = 400;
/// The most guarantees a set may hold.
pub const MAX_QP: usize = 128;

/// The marks of a guarantee are a bit set.
pub type Fl = u16;
pub const F_FOLLOW: Fl = 1;
pub const F_BOUND: Fl = 2;
pub const F_PRUNE_USED: Fl = 4;
pub const F_STUCK: Fl = 8;
pub const F_PRUNED: Fl = 16;
pub const F_UNREAD: Fl = 32;
/// The line was cut: the other side found a reply after which the goal asked cannot hold (the replies not read could only make it worse).
pub const F_CUT: Fl = 64;
/// A member was taken by a unit that is not a member of the frame (the frame did not hold the taker).
pub const F_OUTSIDE: Fl = 128;
/// PLAN 78.58 D: a taking was dropped because the estimated exchange on its place loses (the estimate stood in for the questions below it).
pub const F_EST_PRUNE: Fl = 256;
/// An exchange on one place was read by the estimate in place of the takings below it.
pub const F_EST_READ: Fl = 512;
/// PLAN 78.60 B: the line came back to a set of bearings on the frame it had already seen (no progress); no claim rests on it.
pub const F_CIRCLE: Fl = 1024;
/// The marks that make a claim a READ and not forced: it rests on a bound, a prune or an estimate.
pub const READ_MARKS: Fl = F_BOUND | F_PRUNE_USED | F_EST_PRUNE | F_EST_READ;
/// Marks that make a guarantee no ground for a claim.
const BAD: Fl = F_STUCK | F_PRUNED | F_UNREAD | F_CUT | F_CIRCLE;

/// PLAN 78.60 E: what a claim rests on. FORCED within the frame: no bound, no prune, no estimate on any line that supports it. BOUND: some line was cut (a depth, the quiet plies). ESTIMATE: it rests on the
/// exchange or on a prune by the estimate (lines otherwise ended for a reason). PRUNED: a pure-loss candidate dropped by inclusion (no estimate, no cut). A higher rank rests on more.
pub fn basis_rank(fl: Fl) -> u8 {
    if fl & (F_BOUND | F_CIRCLE) != 0 { 3 } else if fl & (F_EST_PRUNE | F_EST_READ) != 0 { 2 } else if fl & F_PRUNE_USED != 0 { 1 } else { 0 }
}

pub fn basis_word(rank: u8) -> &'static str { match rank { 0 => "forced", 1 => "pruned", 2 => "estimate", _ => "bound" } }

/// PLAN 78.60: the rules of a line's end, each switched on by the caller. All off: the read is exactly that of 78.58 (with no kind values, that of 78.57). `root_cap`: 0, or at least ROOT_CAP, means the cap of
/// 78.57; a smaller one makes frames of the seed units and what bears on them, no more than that many.
#[derive(Debug, Clone, Copy, Default)]
pub struct Rules { pub settle: bool, pub progress: bool, pub cause: bool, pub band: bool, pub root_cap: usize }

/// A named cause for an exchange to be read on below it: kind 0, a unit of the exchange was the only guard of another member; kind 1, its leaving uncovers a line onto a member.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CauseRec { pub kind: u8, pub target: Cell, pub named: Cell, pub ends: (Place, Place) }

/// WHAT IS BEING ASKED. `All` reads the whole set of guarantees (for the composition and the comparison with the exhaustive read); the others ask ONE question and cut the reading when it is answered:
/// `Hold` the side read loses no member; `Gain` it loses none and takes one; `Safe(u)` unit u is not taken; `TakeFree(u)` unit u is taken and none of the side read's is; `TakeAny(u)` unit u is taken.
/// Each is monotone under the meet of replies, so a reply after which it cannot hold ends the reading of that node (a cut).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Goal { All, Hold, Gain, Safe(u8), TakeFree(u8), TakeAny(u8) }

impl Goal {
    fn code(self) -> u16 { match self { Goal::All => 0xFFFF, Goal::Hold => 0xFFFE, Goal::Gain => 0xFFFD, Goal::Safe(u) => u as u16, Goal::TakeFree(u) => 0x100 | u as u16, Goal::TakeAny(u) => 0x200 | u as u16 } }
    fn sat(self, p: QPair) -> bool {
        match self {
            Goal::All => true, Goal::Hold => p.lx == 0, Goal::Gain => p.lx == 0 && p.lo != 0,
            Goal::Safe(u) => p.lx & (1u64 << u) == 0, Goal::TakeFree(u) => p.lx == 0 && p.lo & (1u64 << u) != 0, Goal::TakeAny(u) => p.lo & (1u64 << u) != 0,
        }
    }
    /// The goals that a loss of the side read's members never drops: pure-loss candidates are dropped only when asking these (the whole frame's question).
    fn frame_level(self) -> bool { matches!(self, Goal::All | Goal::Hold | Goal::Gain) }
}

// ------------------------------------------------------------------------------------------------------------------------------------------------------------------ outcomes over identity bits

/// An outcome by inclusion over the identity bits of all units of the root position: what the reading side loses and what the other side loses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct QPair { pub lx: u64, pub lo: u64 }

impl QPair {
    pub const ZERO: QPair = QPair { lx: 0, lo: 0 };
    pub const BOTTOM: QPair = QPair { lx: u64::MAX, lo: 0 };
    pub fn at_least(self, o: QPair) -> bool { self.lx & !o.lx == 0 && o.lo & !self.lo == 0 }
    pub fn meet(self, o: QPair) -> QPair { QPair { lx: self.lx | o.lx, lo: self.lo & o.lo } }
}

/// A guarantee: the outcome, the marks of how it was reached, and the number of moves the reading side spent in the frame to reach it (the tempo charge of `compose_q`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QP { pub p: QPair, pub fl: Fl, pub c: u8 }

fn dominates(b: &QP, a: &QP) -> bool { b.p.at_least(a.p) && b.c <= a.c && (b.fl & BAD == 0 || a.fl & BAD != 0) }

/// The unbeaten guarantees (by inclusion and by cost); equal ones are merged keeping the cheaper and the marks both carry.
pub fn qunbeaten(mut v: Vec<QP>) -> Vec<QP> {
    v.sort_by_key(|q| (q.p.lx, q.p.lo, q.fl & BAD, q.c));
    let mut merged: Vec<QP> = Vec::with_capacity(v.len());
    for q in v {
        match merged.last_mut() {
            Some(l) if l.p == q.p && l.fl & BAD == q.fl & BAD => { l.c = l.c.min(q.c); l.fl = (l.fl & q.fl & !BAD) | (l.fl & BAD); }
            _ => merged.push(q),
        }
    }
    let keep: Vec<QP> = merged.iter().copied().filter(|a| !merged.iter().any(|b| b != a && dominates(b, a) && !dominates(a, b))).collect();
    keep
}

fn meet_all(sets: &[Rc<Vec<QP>>]) -> Option<Vec<QP>> {
    let mut acc: Vec<QP> = sets[0].to_vec();
    for s in &sets[1..] {
        let mut next = Vec::with_capacity(acc.len() * s.len());
        for a in &acc { for b in s.iter() { next.push(QP { p: a.p.meet(b.p), fl: a.fl | b.fl, c: a.c.max(b.c) }); } }
        acc = qunbeaten(next);
        if acc.len() > MAX_QP { return None; }
    }
    Some(acc)
}

/// Whether the guarantees carry the claim about the unit with identity `id` (guarantees marked STUCK, PRUNED or UNREAD ground nothing).
pub fn holds_q(pairs: &[QP], kind: ClaimKind, id: u8) -> bool { support(pairs, kind, id).is_some() }

/// The marks every supporting guarantee carries (a mark is on the claim only if all its supports have it), None when the claim does not hold.
pub fn support(pairs: &[QP], kind: ClaimKind, id: u8) -> Option<Fl> {
    let bit = 1u64 << id;
    let mut fl: Option<Fl> = None;
    for q in pairs.iter().filter(|q| q.fl & BAD == 0) {
        let ok = match kind { ClaimKind::TakeFree => q.p.lo & bit != 0 && q.p.lx == 0, ClaimKind::TakeAny => q.p.lo & bit != 0, ClaimKind::Safe => q.p.lx & bit == 0 };
        if ok { fl = Some(fl.map_or(q.fl, |f| f & q.fl)); }
    }
    fl
}

pub fn marks_word(fl: Fl) -> Vec<&'static str> {
    let mut v = vec![];
    if fl & F_FOLLOW != 0 { v.push("followed_off_frame"); }
    if fl & F_BOUND != 0 { v.push("ended_by_bound"); }
    if fl & F_PRUNE_USED != 0 { v.push("prune_used"); }
    if fl & F_OUTSIDE != 0 { v.push("taken_by_a_unit_off_frame"); }
    if fl & F_EST_PRUNE != 0 { v.push("pruned_by_estimate"); }
    if fl & F_EST_READ != 0 { v.push("exchange_by_estimate"); }
    if fl & F_CIRCLE != 0 { v.push("circling"); }
    v
}

// ------------------------------------------------------------------------------------------------------------------------------------------------------------------ the frame derived at a position

type Ident = Rc<Vec<(Cell, u8)>>;

/// How many units of each owner bear on a place.
fn bearer_counts(part: &PositionPart, place: Place) -> [u32; 2] {
    let mut c = [0u32; 2];
    for r in part.bears.iter().filter(|r| r.place == place) { c[part.units[r.unit].owner.min(1)] += 1; }
    c
}

/// The units that bear on a place and took no part in the exchange (their cell is not one of the line's, and none stands on the exchange's place).
fn outside_bearers(part: &PositionPart, place: Place, to: Place, line: &[Cell]) -> Vec<Cell> {
    let mut v: Vec<Cell> = part.bears.iter().filter(|r| r.place == place).map(|r| cell_of(part, r.unit)).filter(|c| c.2 != to && !line.contains(c)).collect();
    v.sort();
    v
}

/// 78.58 D: whether the exchange changed a bearing off its place, on the frame's places: a unit that stands there still and is borne on by another number of units than before (a unit drawn away from
/// what it guarded, or one that now bears on it from the place of the exchange), or a place the exchange vacated or left empty that a unit outside the exchange now bears on (something uncovered).
fn bearings_changed(before: &PositionPart, after: &PositionPart, places: &[Place], to: Place, line: &[Cell]) -> bool {
    places.iter().filter(|&&q| q != to).any(|&q| {
        let (ob, oa) = (before.occupant[q as usize], after.occupant[q as usize]);
        if ob != 0 && oa != 0 && cell_of(before, ob as usize - 1) == cell_of(after, oa as usize - 1) { bearer_counts(before, q) != bearer_counts(after, q) }
        else { outside_bearers(before, q, to, line) != outside_bearers(after, q, to, line) }
    })
}

/// 78.60 D. The best the taker can hope for when each kind's worth may stand anywhere in its band: the estimate at the corners of the box of the kinds on the board (all of them when there are at most six
/// with a band of some width, else only the all-low and all-high corners). A taking is pruned by the estimate only when this is still negative.
fn band_best(part: &PositionPart, to: Place, taker: usize, kv: &[Val], bands: &[(Val, Val)], point: Val) -> Val {
    let mut ks: Vec<usize> = part.units.iter().map(|u| u.kind as usize).filter(|&k| k < bands.len() && bands[k].0 != bands[k].1).collect();
    ks.sort_unstable();
    ks.dedup();
    let corners: Vec<u32> = if ks.len() <= 6 { (0..(1u32 << ks.len())).collect() } else { vec![0, (1u32 << ks.len().min(31)) - 1] };
    let mut best = point;
    for m in corners {
        let mut kv2 = kv.to_vec();
        for (j, &k) in ks.iter().enumerate() { kv2[k] = if (m >> j) & 1 == 1 { bands[k].1 } else { bands[k].0 }; }
        if let Some(e) = estimate_taking(part, to, taker, &kv2) { best = best.max(e.value); }
    }
    best
}

/// 78.60 C. The named cause for the line to go on below an exchange: (0, member, guard) a unit that took part was the only guard of a member that the other side now bears on; (1, unit behind, member) a
/// taker's leaving uncovers a line, from the `through` relation, onto a member (a unit outside the exchange that bore on that place through units of the exchange only). None: nothing named, the exchange is the result.
fn named_cause(before: &PositionPart, after: &PositionPart, places: &[Place], to: Place, taking_line: &[Cell]) -> Option<(u8, Cell, Cell)> {
    // the units that took part: those that take in the line and the one that stood on the place
    let mut line: Vec<Cell> = taking_line.to_vec();
    let victim = before.occupant[to as usize];
    if victim != 0 { line.push(cell_of(before, victim as usize - 1)); }
    let line = &line[..];
    for &q in places.iter().filter(|&&q| q != to) {
        let (ob, oa) = (before.occupant[q as usize], after.occupant[q as usize]);
        if ob == 0 || oa == 0 { continue; }
        let cb = cell_of(before, ob as usize - 1);
        if cb != cell_of(after, oa as usize - 1) || line.contains(&cb) { continue; }
        let own = cb.0.min(1);
        let guards: Vec<Cell> = before.bears.iter().filter(|r| r.place == q && before.units[r.unit].owner.min(1) == own).map(|r| cell_of(before, r.unit)).collect();
        let kept = after.bears.iter().filter(|r| r.place == q && after.units[r.unit].owner.min(1) == own).count();
        let attacked = after.bears.iter().any(|r| r.place == q && after.units[r.unit].owner.min(1) != own);
        if !guards.is_empty() && kept == 0 && attacked && guards.iter().all(|c| line.contains(c)) { return Some((0, cb, guards[0])); }
    }
    for r in &after.bears {
        let q = r.place;
        if q == to || !places.contains(&q) { continue; }
        let o = after.occupant[q as usize];
        if o == 0 { continue; }
        let w = cell_of(after, r.unit);
        if line.contains(&w) { continue; }
        if before.bears.iter().any(|b| b.place == q && cell_of(before, b.unit) == w) { continue; }
        let via = before.throughs.iter().any(|t| t.place == q && cell_of(before, t.unit) == w && !t.between.is_empty() && t.between.iter().all(|&i| line.contains(&cell_of(before, i))));
        if via { return Some((1, w, cell_of(after, o as usize - 1))); }
    }
    None
}

fn cell_of(part: &PositionPart, i: usize) -> Cell { let u = &part.units[i]; (u.owner, u.kind, u.at) }

fn sorted_cells<G: Placed>(g: &G, s: &G::State) -> Vec<Cell> { let mut c = g.cells(s); c.sort(); c }

fn hash_ident(t: &[(Cell, u8)]) -> u64 {
    let mut h = rustc_hash::FxHasher::default();
    for (c, id) in t { c.hash(&mut h); h.write_u8(*id); }
    h.finish()
}

fn id_at(ident: &[(Cell, u8)], c: &Cell) -> Option<u8> { ident.binary_search_by_key(c, |x| x.0).ok().map(|i| ident[i].1) }

fn cell_of_id(ident: &[(Cell, u8)], id: u8) -> Option<Cell> { ident.iter().find(|x| x.1 == id).map(|x| x.0) }

/// The members and places of a frame at a position, derived from where its seed units stand and bear now.
struct View { members: Vec<usize>, pset: PlaceSet, mask: u64 }

/// `mode` 0: the seed's units, its places and what bears on them; 1: the same, to a fixed point through the `through` relation; 2 (78.60 F): the seed's units and what bears on them, no empty place of the seed.
fn view(part: &PositionPart, np: usize, ident: &[(Cell, u8)], seed_ids: &[u8], seed_places: &[Place], mode: u8) -> Option<View> {
    let through = mode == 1;
    let mut alive: Vec<usize> = vec![];
    for &sid in seed_ids {
        let Some(c) = cell_of_id(ident, sid) else { continue };
        let o = part.occupant[c.2 as usize];
        if o != 0 && cell_of(part, o as usize - 1) == c { alive.push(o as usize - 1); }
    }
    if alive.is_empty() { return None; }
    let mut pset = PlaceSet::new(np);
    for &i in &alive { pset.insert(part.units[i].at); }
    if mode != 2 {
        for &q in seed_places {
            if part.bears.iter().any(|r| r.place == q && alive.contains(&r.unit)) { pset.insert(q); }
        }
    }
    let mut inm = vec![false; part.units.len()];
    for &i in &alive { inm[i] = true; }
    for p in pset.iter() { let o = part.occupant[p as usize]; if o != 0 { inm[o as usize - 1] = true; } }
    for r in &part.bears { if pset.contains(r.place) { inm[r.unit] = true; } }
    if through {
        loop {
            let mut changed = false;
            for r in &part.throughs {
                if !inm[r.unit] && pset.contains(r.place) && r.between.iter().any(|&b| inm[b]) { inm[r.unit] = true; changed = true; }
            }
            if !changed { break; }
        }
    }
    let members: Vec<usize> = (0..inm.len()).filter(|&i| inm[i]).collect();
    let mut mask = 0u64;
    for &i in &members { if let Some(id) = id_at(ident, &cell_of(part, i)) { mask |= 1u64 << id; } }
    Some(View { members, pset, mask })
}

/// A frame made from a seed at a position: its identity is the seed's units.
#[derive(Debug, Clone)]
pub struct QFrame { pub seed: Seed, pub members: Vec<Cell>, pub places: Vec<Place>, pub id: String, pub split: bool, pub mode: u8 }

fn qframe_id(st: &Static, seed: &Seed) -> String {
    let mut h = rustc_hash::FxHasher::default();
    for c in &seed.units { h.write(unit_id(st, c).as_bytes()); h.write_u8(0); }
    format!("frame:{:016x}", h.finish())
}

impl QFrame {
    pub fn new<G: Game + Placed>(st: &Static, g: &G, s: &G::State, seed: &Seed) -> Result<QFrame, FrameErr> { QFrame::new_cap(st, g, s, seed, ROOT_CAP) }

    /// 78.60 F: with `cap` below ROOT_CAP the frame is the seed's units and what bears on them (mode 2, at every node of a read too); more than `cap` members is TooLarge (the builder then splits the seed).
    pub fn new_cap<G: Game + Placed>(st: &Static, g: &G, s: &G::State, seed: &Seed, cap: usize) -> Result<QFrame, FrameErr> {
        let cap = if cap == 0 { ROOT_CAP } else { cap };
        let part = st.position(g, s);
        let cells = sorted_cells(g, s);
        if cells.len() > 64 { return Err(FrameErr::TooLarge(cells.len())); }
        let ident: Vec<(Cell, u8)> = cells.iter().enumerate().map(|(i, c)| (*c, i as u8)).collect();
        let mut sids = vec![];
        for c in &seed.units { match id_at(&ident, c) { Some(i) => sids.push(i), None => return Err(FrameErr::UnitGone(*c)) } }
        let np = st.names.places.len();
        let mut split = false;
        let small = cap < ROOT_CAP;
        let mut v = view(&part, np, &ident, &sids, &seed.places, if small { 2 } else { 1 }).ok_or(FrameErr::NoProfile)?;
        if small {
            if v.members.len() > cap { return Err(FrameErr::TooLarge(v.members.len())); }
        } else if v.members.len() > cap {
            split = true;
            v = view(&part, np, &ident, &sids, &seed.places, 0).ok_or(FrameErr::NoProfile)?;
            if v.members.len() > cap { return Err(FrameErr::TooLarge(v.members.len())); }
        }
        let members: Vec<Cell> = v.members.iter().map(|&i| cell_of(&part, i)).collect();
        let mut ms = members;
        ms.sort();
        Ok(QFrame { seed: seed.clone(), members: ms, places: v.pset.iter().collect(), id: qframe_id(st, seed), split, mode: if small { 2 } else { 1 } })
    }
}

/// Two frames are in contact when they share a member or a place, or a member of one bears on a place of the other.
pub fn q_in_contact(a: &QFrame, b: &QFrame, part: &PositionPart) -> bool {
    if a.members.iter().any(|c| b.members.contains(c)) || a.places.iter().any(|p| b.places.contains(p)) { return true; }
    let bears_on = |x: &QFrame, y: &QFrame| x.members.iter().any(|c| part.units.iter().position(|u| (u.owner, u.kind, u.at) == *c).is_some_and(|i| part.bears.iter().any(|r| r.unit == i && y.places.contains(&r.place))));
    bears_on(a, b) || bears_on(b, a)
}

/// FRAMES OF SEVERAL SEEDS, joined where in contact, split where a join is too large, and none read with a unit in common: [(frame, seeds joined)], with the count of frames dropped for a shared unit
/// and of joins refused (split instead).
pub struct Built { pub frames: Vec<QFrame>, pub refused: Vec<String>, pub split_joins: u32, pub dropped_shared: u32 }

pub fn build_frames<G: Game + Placed>(st: &Static, g: &G, s: &G::State, seeds: &[Seed]) -> Built { build_frames_cap(st, g, s, seeds, ROOT_CAP) }

/// `build_frames` with the cap of members at the root (78.60 F). Below ROOT_CAP a seed too large for the cap is SPLIT into one frame per seed unit (each read with what bears on it, or refused as too large).
pub fn build_frames_cap<G: Game + Placed>(st: &Static, g: &G, s: &G::State, seeds: &[Seed], cap: usize) -> Built {
    let part = st.position(g, s);
    let mut frames: Vec<QFrame> = vec![];
    let mut refused = vec![];
    let small = cap != 0 && cap < ROOT_CAP;
    for sd in seeds {
        match QFrame::new_cap(st, g, s, sd, cap) {
            Ok(f) => { if !frames.iter().any(|x| x.id == f.id) { frames.push(f) } }
            Err(FrameErr::TooLarge(n)) if small && sd.units.len() > 1 => {
                refused.push(format!("split:{n}"));
                for c in &sd.units {
                    let one = Seed { units: vec![*c], places: vec![] };
                    match QFrame::new_cap(st, g, s, &one, cap) { Ok(f) => { if !frames.iter().any(|x| x.id == f.id) { frames.push(f) } } Err(e) => refused.push(e.word()) }
                }
            }
            Err(e) => refused.push(e.word()),
        }
    }
    let (mut split_joins, mut dropped_shared) = (0u32, 0u32);
    let mut stuck_pairs: FxHashSet<(String, String)> = FxHashSet::default();
    'again: loop {
        for i in 0..frames.len() {
            for j in i + 1..frames.len() {
                if stuck_pairs.contains(&(frames[i].id.clone(), frames[j].id.clone())) { continue; }
                if q_in_contact(&frames[i], &frames[j], &part) {
                    let mut seed = frames[i].seed.clone();
                    for c in &frames[j].seed.units { if !seed.units.contains(c) { seed.units.push(*c); } }
                    seed.units.sort();
                    for p in &frames[j].seed.places { if !seed.places.contains(p) { seed.places.push(*p); } }
                    seed.places.sort();
                    match QFrame::new_cap(st, g, s, &seed, cap).ok().filter(|f| !f.split) {
                        Some(f) => { frames[i] = f; frames.remove(j); continue 'again; }
                        None => { split_joins += 1; stuck_pairs.insert((frames[i].id.clone(), frames[j].id.clone())); continue 'again; }
                    }
                }
            }
        }
        break;
    }
    // frames left in contact after a refused join are split: the bigger is read, the other is dropped when it shares a member (never both read with a unit in common)
    frames.sort_by_key(|f| std::cmp::Reverse(f.members.len()));
    let mut out: Vec<QFrame> = vec![];
    for f in frames {
        if out.iter().any(|o| o.members.iter().any(|c| f.members.contains(c))) { dropped_shared += 1; } else { out.push(f); }
    }
    Built { frames: out, refused, split_joins, dropped_shared }
}

// ------------------------------------------------------------------------------------------------------------------------------------------------------------------ the reading

#[derive(Debug, Clone, Copy, Default)]
pub struct QStats {
    /// Questions asked: one node of a reading, the side to move asked what it can do (memo hits are not asked again).
    pub questions: u64,
    /// Positions made by playing a move (the cost the old read counted).
    pub positions: u64,
    /// Candidate moves asked about, and legal moves not asked about.
    pub candidates: u64,
    pub not_asked: u64,
    /// Candidates pruned: a member lost and none gained against one reply; by whether the moved unit was the one lost.
    pub pruned_moved: u64,
    pub pruned_other: u64,
    /// Nodes where the questions stopped at the first candidate that holds.
    pub stops: u64,
    /// Lines ended by the bound (quiet plies, depth, extension), lines the fuse ended, lines stuck, forced passes, passes not realisable, passes the game refused, frames past MAXQ, frame gone.
    pub bounded: u64,
    pub unread: u64,
    pub stuck: u64,
    pub forced_pass: u64,
    pub pass_unrealisable: u64,
    pub pass_refused: u64,
    pub too_big: u64,
    pub gone: u64,
    /// Candidate moves of a member to a place outside the root frame's places (the frame follows it).
    pub follows: u64,
    pub others: u64,
    /// PLAN 78.58 D: takings dropped because the exchange on their place loses by the estimate; takings whose exchange was read by the estimate as one question; of those, the ones whose line went on below because a bearing off the place changed; and lines of takings the game could not play.
    pub est_pruned: u64,
    pub est_read: u64,
    pub est_cont: u64,
    pub est_unplayed: u64,
    /// PLAN 78.60. Lines ended SETTLED (nothing stands to be lost on the frame for either side, no member trapped); the ones of them that had a contact an estimate read as no loss; lines that CIRCLED; taking
    /// pruned by the estimate that the band of the estimate held back and so was read; exchanges read on below for a named cause (the only guard of a member, a line uncovered) and exchanges that stopped for want of one;
    /// quiet moves not read for want of progress; the trap questions asked to settle a line (their questions, positions) and the members they found trapped.
    pub end_settled: u64,
    pub end_settled_est: u64,
    pub end_circle: u64,
    pub band_held: u64,
    pub cause_guard: u64,
    pub cause_line: u64,
    pub est_stop: u64,
    pub quiet_skipped: u64,
    pub settle_trap_q: u64,
    pub settle_trap_pos: u64,
    pub settle_trapped: u64,
    /// Questions by the class of the move that led to the node (taking, answer, defence, threat, block; the last is the root or a pass), candidates made and asked by class, questions by ply below the root.
    pub by_class: [u64; 6],
    pub gen_class: [u64; 5],
    pub ask_class: [u64; 5],
    pub by_depth: [u64; 16],
}

macro_rules! stat_fields { ($m:ident) => { $m!(questions positions candidates not_asked pruned_moved pruned_other stops bounded unread stuck forced_pass pass_unrealisable pass_refused too_big gone follows others est_pruned est_read est_cont est_unplayed end_settled end_settled_est end_circle band_held cause_guard cause_line est_stop quiet_skipped settle_trap_q settle_trap_pos settle_trapped) }; }

impl QStats {
    pub fn add(&mut self, o: &QStats) { macro_rules! a { ($($f:ident)*) => { $( self.$f += o.$f; )* } } stat_fields!(a); for i in 0..6 { self.by_class[i] += o.by_class[i]; } for i in 0..5 { self.gen_class[i] += o.gen_class[i]; self.ask_class[i] += o.ask_class[i]; } for i in 0..16 { self.by_depth[i] += o.by_depth[i]; } }
    pub fn since(&self, b: &QStats) -> QStats { let mut r = QStats::default(); macro_rules! a { ($($f:ident)*) => { $( r.$f = self.$f - b.$f; )* } } stat_fields!(a); for i in 0..6 { r.by_class[i] = self.by_class[i] - b.by_class[i]; } for i in 0..5 { r.gen_class[i] = self.gen_class[i] - b.gen_class[i]; r.ask_class[i] = self.ask_class[i] - b.ask_class[i]; } for i in 0..16 { r.by_depth[i] = self.by_depth[i] - b.by_depth[i]; } r }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Class { Capture, Answer, Defend, Threat, Block }

/// The estimate of the exchange on the place a taking is made on (78.58): its worth to the taker's side, and the units (cells at the node) that take in the line it reads, the taker first.
struct Est { value: Val, best: Val, line: Vec<Cell> }

struct Cand<G: Game> { state: G::State, ident: Ident, jm: u64, lost: u64, capture: bool, follow: bool, outside: bool, ends: (Place, Place), moved: u8, class: Class, est: Option<Est>, kind_change: bool }

fn class_ix(c: Class) -> usize { match c { Class::Capture => 0, Class::Answer => 1, Class::Defend => 2, Class::Threat => 3, Class::Block => 4 } }

struct Exp<G: Game> { frame_places: Vec<Place>, att: u64, mask: u64, mover: Option<Role>, gone: bool, big: bool, cands: Vec<Rc<Cand<G>>>, pass: Option<G::State>, att_o: u64, contacts: Vec<Cell>, contact_pos: bool, sig: u64 }

type MKey = (u64, u64, u8, i8, u8, u8, u64, u16, u64);

fn credit_key(cr: QPair) -> u64 { cr.lx.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ cr.lo.rotate_left(17) }

/// The root candidate pruned, with its reason.
#[derive(Debug, Clone)]
pub struct PruneRec { pub mover: Role, pub ends: (Place, Place), pub lost: Cell, pub moved_lost: bool, pub reply: Option<(Place, Place)> }

pub struct QReader<'a, G: Game + Placed> {
    g: &'a G,
    st: &'a Static,
    pub frame: &'a QFrame,
    seed_ids: Vec<u8>,
    seed_places: Vec<Place>,
    root_p: PlaceSet,
    omask: [u64; 2],
    pub root_ident: Ident,
    pub fuse: u64,
    pub frame_fuse: u64,
    spent: u64,
    pub depth: i8,
    pub quiet: u8,
    pub ext: i8,
    nodes_left: u64,
    parts: FxHashMap<u64, Rc<PositionPart>>,
    exp: FxHashMap<(u64, u64, u64), Rc<Exp<G>>>,
    memo: FxHashMap<MKey, Rc<Vec<QP>>>,
    wit: FxHashMap<MKey, ((Place, Place), u64)>,
    pub stats: QStats,
    pub prunes: Vec<PruneRec>,
    via_next: usize,
    np: usize,
    /// The worth of each kind (indexed by kind) read from the estimator's record: the input of the exchange (78.58). None: no exchange is read (the 78.57 read).
    pub kv: Option<Rc<Vec<Val>>>,
    /// PLAN 78.60: the rules of a line's end, the band (low, high) of the worth of each kind (indexed by kind; the estimator's record), how the frame's members are derived at a node, the bearings seen on the
    /// line being read (circling), and the named causes found for an exchange to go on below.
    pub rules: Rules,
    pub bands: Option<Rc<Vec<(Val, Val)>>>,
    vmode: u8,
    path: Vec<u64>,
    pub causes: Vec<CauseRec>,
}

/// What one reading found.
#[derive(Debug, Clone)]
pub struct QForce { pub pairs: Vec<QP>, pub stats: QStats }

impl<'a, G: Game + Placed> QReader<'a, G> {
    pub fn new(g: &'a G, st: &'a Static, frame: &'a QFrame, s: &G::State) -> QReader<'a, G> {
        let cells = sorted_cells(g, s);
        let ident: Vec<(Cell, u8)> = cells.iter().enumerate().map(|(i, c)| (*c, i as u8)).collect();
        let np = st.names.places.len();
        let mut omask = [0u64; 2];
        for (c, id) in &ident { omask[c.0.min(1)] |= 1u64 << id; }
        let seed_ids: Vec<u8> = frame.seed.units.iter().filter_map(|c| id_at(&ident, c)).collect();
        let root_p = PlaceSet::from_places(np, frame.places.iter().copied());
        QReader { g, st, frame, seed_ids, seed_places: frame.seed.places.clone(), root_p, omask, root_ident: Rc::new(ident), fuse: FUSE_QUESTIONS, frame_fuse: FRAME_FUSE, spent: 0, depth: DEPTH, quiet: QUIET_PLIES, ext: EXT, nodes_left: FUSE_QUESTIONS, parts: FxHashMap::default(), exp: FxHashMap::default(), memo: FxHashMap::default(), wit: FxHashMap::default(), stats: QStats::default(), prunes: vec![], via_next: 5, np, kv: None, rules: Rules::default(), bands: None, vmode: frame.mode, path: vec![], causes: vec![] }
    }

    /// The identity of a root member (index into `frame.members`).
    pub fn member_id(&self, i: usize) -> u8 { id_at(&self.root_ident, &self.frame.members[i]).expect("a member stands at the root") }

    fn part(&mut self, s: &G::State) -> Rc<PositionPart> {
        let k = self.g.key(s);
        if let Some(p) = self.parts.get(&k) { return p.clone(); }
        let p = Rc::new(self.st.position(self.g, s));
        self.parts.insert(k, p.clone());
        p
    }

    fn child_ident(&mut self, prev: &[Cell], next: &[Cell], ident: &[(Cell, u8)], member_mask: u64) -> (Ident, u64) {
        let (steps, _) = step_between(prev, next);
        let (mut out, mut lost) = (Vec::with_capacity(ident.len()), 0u64);
        for (i, (_c, id)) in ident.iter().enumerate() {
            match steps[i] {
                Step::Alive(j) => out.push((next[j as usize], *id)),
                Step::Taken(_) => { if member_mask & (1u64 << id) != 0 { lost |= 1u64 << id; } }
                Step::Other => { self.stats.others += 1; }
            }
        }
        out.sort();
        (Rc::new(out), lost)
    }

    fn expand(&mut self, s: &G::State, ident: &Ident, jm: u64) -> Rc<Exp<G>> {
        let key = (self.g.key(s), hash_ident(ident), jm);
        if let Some(e) = self.exp.get(&key) { return e.clone(); }
        let g = self.g;
        let e = Rc::new(self.expand_new(g, s, ident, jm));
        self.exp.insert(key, e.clone());
        e
    }

    fn expand_new(&mut self, g: &'a G, s: &G::State, ident: &Ident, jm: u64) -> Exp<G> {
        let Some(mover) = g.mover(s) else { return Exp { frame_places: vec![], att: 0, mask: 0, mover: None, gone: false, big: false, cands: vec![], pass: None, att_o: 0, contacts: vec![], contact_pos: false, sig: 0 } };
        let part = self.part(s);
        let Some(v) = view(&part, self.np, ident, &self.seed_ids, &self.seed_places, self.vmode) else { self.stats.gone += 1; return Exp { frame_places: vec![], att: 0, mask: 0, mover: Some(mover), gone: true, big: false, cands: vec![], pass: None, att_o: 0, contacts: vec![], contact_pos: false, sig: 0 } };
        // the units that were members earlier in the line and still stand stay members (the frame is bound to its units: a taker does not leave it when what it took is gone)
        let (mut members, mut mask) = (v.members.clone(), v.mask);
        for (c, id) in ident.iter() {
            if jm & (1u64 << id) != 0 && mask & (1u64 << id) == 0 {
                let o = part.occupant[c.2 as usize];
                if o != 0 && cell_of(&part, o as usize - 1) == *c { members.push(o as usize - 1); mask |= 1u64 << id; }
            }
        }
        if members.len() > MAXQ { return Exp { frame_places: vec![], att: 0, mask: 0, mover: Some(mover), gone: false, big: true, cands: vec![], pass: None, att_o: 0, contacts: vec![], contact_pos: false, sig: 0 }; }
        let other = 1 - mover.min(1);
        let prev: Vec<Cell> = ident.iter().map(|x| x.0).collect();
        // Q0: the members of the side to move that the other side bears on now
        let attacked: Vec<Place> = members.iter().map(|&i| &part.units[i]).filter(|u| u.owner == mover && g.strikes_place(s, other, u.at) && !g.strikes_place(s, mover, u.at)).map(|u| u.at).collect();
        // the units of theirs that stand to be lost: members that nothing of theirs guards (a guarded unit is an exchange, not a pure gain)
        let att: u64 = attacked.iter().filter_map(|&q| ident.iter().find(|x| x.0 .2 == q && x.0 .0 == mover)).fold(0u64, |a, x| a | (1u64 << x.1));
        let opp_member_places: Vec<Place> = members.iter().map(|&i| &part.units[i]).filter(|u| u.owner != mover && !g.strikes_place(s, other, u.at)).map(|u| u.at).collect();
        // 78.60 A: what stands to be lost for the OTHER side (its members that nothing of theirs guards and that the mover bears on), and the contacts: members of either side that the other bears on while
        // their own side guards them. A contact stands to be lost when some taking on it is worth something by the estimate (or at all when no estimate is set: it cannot be told).
        let (mut att_o, mut contacts, mut contact_pos) = (0u64, Vec::<Cell>::new(), false);
        if self.rules.settle {
            for &i in &members {
                let u = &part.units[i];
                let o = u.owner.min(1);
                if !g.strikes_place(s, 1 - o, u.at) { continue; }
                let cell = cell_of(&part, i);
                if !g.strikes_place(s, o, u.at) {
                    if o != mover.min(1) { if let Some(x) = ident.iter().find(|x| x.0 == cell) { att_o |= 1u64 << x.1; } }
                } else {
                    contacts.push(cell);
                    match &self.kv {
                        None => contact_pos = true,
                        Some(kv) => {
                            for r in part.bears.iter().filter(|r| r.place == u.at && part.units[r.unit].owner.min(1) != o) {
                                if estimate_taking(&part, u.at, r.unit, kv).is_some_and(|e| e.value > 0) { contact_pos = true; }
                            }
                        }
                    }
                }
            }
        }
        let mut cands: Vec<Rc<Cand<G>>> = vec![];
        let mut outsiders: Vec<G::Move> = vec![];
        for m in g.moves(s) {
            let (from, to) = g.move_ends(s, &m);
            let Some(kind) = g.kind_at(s, from) else { continue };
            let Some(mid) = id_at(ident, &(mover, kind, from)) else { continue };
            let is_member = mask & (1u64 << mid) != 0;
            let (_k, cap) = g.move_info(s, &m);
            let capture = cap.is_some();
            let victim_member = capture && ident.iter().find(|x| x.0 .2 == to && x.0 .0 != mover).is_some_and(|x| mask & (1u64 << x.1) != 0);
            let in_p = v.pset.contains(to);
            let threatened = is_member && attacked.contains(&from);
            let mut class: Option<Class> = None;
            if capture && (is_member || victim_member || in_p) { class = Some(Class::Capture); }
            else if threatened && !capture { class = Some(Class::Answer); }
            else if in_p && !capture { class = Some(Class::Block); }
            if class.is_none() {
                // a bearing added on a threatened member of ours (a defence), or a new threat on a member of theirs that nothing of theirs guards: read from where the unit would stand
                let bear = g.bearing_from(s, mover, kind, to);
                let defends = attacked.iter().any(|&q| q != from && bear.contains(&q) && !g.strikes_place(s, mover, q));
                let threatens = opp_member_places.iter().any(|&q| bear.contains(&q) && !g.strikes_place(s, mover, q));
                // a unit that stands between one of ours and a unit of theirs that nothing guards: moving it uncovers that bearing
                let me_ix = part.occupant[from as usize];
                let uncovers = me_ix != 0 && part.throughs.iter().any(|r| r.between.contains(&(me_ix as usize - 1)) && part.units[r.unit].owner == mover && opp_member_places.contains(&r.place));
                if defends { class = Some(Class::Defend) } else if threatens || uncovers { class = Some(Class::Threat) }
            }
            let child: Option<(G::State, Vec<Cell>)> = if class.is_some() {
                let s2 = g.apply(s, &m);
                self.stats.positions += 1;
                let next = sorted_cells(g, &s2);
                Some((s2, next))
            } else { None };
            match (class, child) {
                (Some(c), Some((s2, next))) => {
                    let (nid, lost) = self.child_ident(&prev, &next, ident, mask);
                    let follow = is_member && !self.root_p.contains(to);
                    if follow { self.stats.follows += 1; }
                    let outside = capture && !is_member;
                    self.stats.gen_class[class_ix(c)] += 1;
                    let (band_on, bands) = (self.rules.band, self.bands.clone());
                    let est = if capture && c == Class::Capture {
                        self.kv.clone().and_then(|kv| {
                            let ti = part.occupant[from as usize] as usize;
                            if ti == 0 { return None; }
                            estimate_taking(&part, to, ti - 1, &kv).map(|e| {
                                let best = if e.value < 0 && band_on { match &bands { Some(b) => band_best(&part, to, ti - 1, &kv, b, e.value), None => e.value } } else { e.value };
                                Est { value: e.value, best, line: e.line.iter().map(|&i| cell_of(&part, i)).collect() }
                            })
                        })
                    } else { None };
                    let kind_change = !capture && g.kind_at(&s2, to) != Some(kind);
                    cands.push(Rc::new(Cand { state: s2, ident: nid, jm: mask & !lost, lost, capture, follow, outside, ends: (from, to), moved: mid, class: c, est, kind_change }));
                }
                _ => { self.stats.not_asked += 1; if !is_member { outsiders.push(m); } }
            }
        }
        cands.sort_by_key(|c| match c.class { Class::Capture => 0, Class::Answer => 1, Class::Defend => 2, Class::Threat => 3, Class::Block => 4 });
        // the pass: the game's pass, and some move of a unit that is not a member leaves the frame's derived members and places as they were
        let here: (Vec<Cell>, Vec<Place>) = (v.members.iter().map(|&i| cell_of(&part, i)).collect(), v.pset.iter().collect());
        let mut neutral: Option<G::State> = None;
        for m in &outsiders {
            let (from, to) = g.move_ends(s, m);
            if v.pset.contains(to) || v.pset.contains(from) { continue; }
            let s2 = g.apply(s, m);
            self.stats.positions += 1;
            let next = sorted_cells(g, &s2);
            let (nid, _) = self.child_ident(&prev, &next, ident, mask);
            let part2 = self.part(&s2);
            if let Some(v2) = view(&part2, self.np, &nid, &self.seed_ids, &self.seed_places, self.vmode) {
                let mut cells_two: Vec<Cell> = v2.members.iter().map(|&i| cell_of(&part2, i)).collect();
                cells_two.sort();
                let mut here_two = here.0.clone();
                here_two.sort();
                if cells_two == here_two && v2.pset.iter().collect::<Vec<_>>() == here.1 { neutral = Some(s2); break; }
            }
        }
        // the pass is the game's pass; where the game refuses it (the side to move is under a forcing threat) the neutral move itself stands for it: it leaves the frame as it was
        let pass = match &neutral { Some(sn) => match g.pass(s) { Some(sp) => Some(sp), None => { self.stats.pass_refused += 1; Some(sn.clone()) } }, None => None };
        if neutral.is_none() { self.stats.pass_unrealisable += 1; }
        let mut frame_places: Vec<Place> = v.pset.iter().collect();
        for &i in &members { frame_places.push(part.units[i].at); }
        frame_places.sort_unstable();
        frame_places.dedup();
        // 78.60 B: the bearings on the frame as they stand (who stands where in the frame and what bears on its places), by the side to move: a line that comes back to one has made no progress
        let mut sig = 0u64;
        if self.rules.progress {
            let mut cs: Vec<Cell> = members.iter().map(|&i| cell_of(&part, i)).collect();
            cs.sort();
            let mut bs: Vec<(Cell, Place)> = part.bears.iter().filter(|r| frame_places.binary_search(&r.place).is_ok()).map(|r| (cell_of(&part, r.unit), r.place)).collect();
            bs.sort();
            let mut h = rustc_hash::FxHasher::default();
            h.write_u8(mover as u8);
            cs.hash(&mut h);
            bs.hash(&mut h);
            sig = h.finish();
        }
        Exp { frame_places, att, mask, mover: Some(mover), gone: false, big: false, cands, pass, att_o, contacts, contact_pos, sig }
    }

    /// PLAY THE LINE of an estimate: the takings after the first, each side with the unit the estimate names, on the place of the first taking. None when the game cannot play one of them (the estimate then
    /// stays unused for this candidate, counted).
    fn play_line(&mut self, c: &Cand<G>, est: &Est) -> Option<(G::State, Ident, u64, u64)> {
        let g = self.g;
        let (mut state, mut ident, mut jm, mut lost) = (c.state.clone(), c.ident.clone(), c.jm, c.lost);
        let to = c.ends.1;
        for cell in est.line.iter().skip(1) {
            let mv = g.moves(&state).into_iter().find(|m| { let (f, t) = g.move_ends(&state, m); f == cell.2 && t == to && g.kind_at(&state, f) == Some(cell.1) })?;
            let prev: Vec<Cell> = ident.iter().map(|x| x.0).collect();
            let s2 = g.apply(&state, &mv);
            self.stats.positions += 1;
            let next = sorted_cells(g, &s2);
            let (nid, l) = self.child_ident(&prev, &next, &ident, jm);
            lost |= l;
            jm &= !l;
            state = s2;
            ident = nid;
        }
        Some((state, ident, jm, lost))
    }

    /// One question and what it opens. The bearings seen on the line being read (circling, 78.60 B) are kept for the line only: what a node pushed is taken off when it returns.
    fn solve(&mut self, s: &G::State, ident: &Ident, jm: u64, xr: Role, left: i8, quiet: u8, owed: u8, cr: QPair, goal: Goal) -> Rc<Vec<QP>> {
        let n = self.path.len();
        let r = self.solve_inner(s, ident, jm, xr, left, quiet, owed, cr, goal);
        self.path.truncate(n);
        r
    }

    fn solve_inner(&mut self, s: &G::State, ident: &Ident, jm: u64, xr: Role, left: i8, quiet: u8, owed: u8, cr: QPair, goal: Goal) -> Rc<Vec<QP>> {
        let mk: MKey = (self.g.key(s), hash_ident(ident), xr as u8, left, quiet, owed, credit_key(cr), goal.code(), jm);
        if let Some(v) = self.memo.get(&mk) { return v.clone(); }
        if self.nodes_left == 0 { self.stats.unread += 1; return Rc::new(vec![QP { p: QPair::BOTTOM, fl: F_UNREAD, c: 0 }]); }
        self.nodes_left -= 1;
        self.spent += 1;
        self.stats.questions += 1;
        let via = std::mem::replace(&mut self.via_next, 5);
        self.stats.by_class[via] += 1;
        self.stats.by_depth[((self.depth - left).max(0) as usize).min(15)] += 1;
        let e = self.expand(s, ident, jm);
        let Some(mover) = e.mover else { return Rc::new(vec![QP { p: QPair::ZERO, fl: 0, c: 0 }]) };
        if e.gone { return Rc::new(vec![QP { p: QPair::ZERO, fl: 0, c: 0 }]); }
        if e.big { self.stats.too_big += 1; return Rc::new(vec![QP { p: QPair::ZERO, fl: F_BOUND, c: 0 }]); }
        // 78.60 A. SETTLED: past the plies where quiet threats are still read, a position where neither side has a member that stands to be lost, and no member is trapped, ends the line for that reason. With no contact
        // between the sides on the frame the end rests on nothing; with a contact that the estimate read as no loss it carries the estimate mark.
        if self.rules.settle && left < self.depth - 1 && e.att == 0 && e.att_o == 0 && !e.contact_pos {
            let mut ok = true;
            for cell in e.contacts.clone() {
                let ta = trap(self.g, self.st, s, cell, SETTLE_TRAP_FUSE);
                self.stats.settle_trap_q += ta.stats.questions;
                self.stats.settle_trap_pos += ta.stats.positions;
                if ta.word == TrapWord::Trapped { self.stats.settle_trapped += 1; }
                if matches!(ta.word, TrapWord::Trapped | TrapWord::NotQuiet | TrapWord::Unread) { ok = false; break; }
            }
            if ok {
                let fl = if e.contacts.is_empty() { 0 } else { F_EST_READ };
                if fl == 0 { self.stats.end_settled += 1 } else { self.stats.end_settled_est += 1 }
                let r = Rc::new(vec![QP { p: QPair::ZERO, fl, c: 0 }]);
                self.memo.insert(mk, r.clone());
                return r;
            }
        }
        // 78.60 B. CIRCLING: the line is at a set of bearings on the frame it has already been at, with the same side to move; no claim rests on it
        if self.rules.progress && e.sig != 0 {
            if self.path.contains(&e.sig) { self.stats.end_circle += 1; return Rc::new(vec![QP { p: QPair::ZERO, fl: F_CIRCLE, c: 0 }]); }
            self.path.push(e.sig);
        }
        let is_x = mover == xr;
        let leaf = left <= 0 || quiet == 0;
        let hard_end = left <= -self.ext;
        let (own_mask, opp_mask) = (self.omask[xr.min(1)], self.omask[1 - xr.min(1)]);
        let mut options: Vec<Rc<Vec<QP>>> = vec![];
        let mut used_prune = false;
        let mut used_est = false;
        let mut saw_circle = false;
        let mut early: Option<(Place, Place)> = None;
        let mut cut: Option<((Place, Place), bool)> = None;
        let mut asked = 0usize;
        let owed_now = is_x && owed > 0;
        if !owed_now && !hard_end {
            for c in e.cands.clone().iter() {
                if leaf && !c.capture { continue; }
                // a quiet threat is read only on the first ply of each side (an in-between move or a double threat); deeper, quiet moves are answers (a step away, a defence, a block)
                if c.class == Class::Threat && left < self.depth - 1 { continue; }
                // 78.58 D: a taking whose exchange on its place loses by the estimate is pruned (pruned by estimate); one that wins or is level is read as ONE question with its result
                // 78.60 B: past the first plies a quiet move is read only as a reply to a standing loss (a taking, or a move that changes a unit's kind, is progress)
                if self.rules.progress && left < self.depth - 1 && !c.capture && !c.kind_change && e.att == 0 { self.stats.quiet_skipped += 1; continue; }
                // 78.60 D: with a band, the estimate prunes only when the taking loses at every corner of the band; otherwise the taking is read (held back), not as one question
                let mut est_c: Option<&Est> = c.est.as_ref();
                if let Some(est) = est_c {
                    if est.value < 0 {
                        if est.best < 0 { self.stats.est_pruned += 1; used_est = true; continue; }
                        self.stats.band_held += 1;
                        est_c = None;
                    }
                }
                asked += 1;
                self.stats.candidates += 1;
                self.stats.ask_class[class_ix(c.class)] += 1;
                self.via_next = class_ix(c.class);
                let nq = if c.capture { self.quiet } else { quiet.saturating_sub(1) };
                let mut played: Option<(G::State, Ident, u64, u64)> = None;
                let est_line: Vec<Cell> = est_c.map(|x| x.line.clone()).unwrap_or_default();
                if let Some(est) = est_c { played = self.play_line(c, est); if played.is_none() { self.stats.est_unplayed += 1; } }
                let (c_state, c_ident, c_jm, c_lost) = match &played { Some((st_a, id_a, jm_a, lost_a)) => (st_a.clone(), id_a.clone(), *jm_a, *lost_a), None => (c.state.clone(), c.ident.clone(), c.jm, c.lost) };
                let child_cr = if is_x { QPair { lx: c_lost & own_mask, lo: c_lost & opp_mask } } else { QPair::ZERO };
                let sub: Rc<Vec<QP>> = if played.is_some() {
                    self.stats.est_read += 1;
                    let (before, after) = (self.part(s), self.part(&c_state));
                    let to = c.ends.1;
                    // 78.58: any change of a bearing on the frame lets the line go on below; 78.60 C: only a named cause does (kept with the question)
                    let changed = if self.rules.cause {
                        match named_cause(&before, &after, &e.frame_places, to, &est_line) {
                            Some((k, target, named)) => {
                                if k == 0 { self.stats.cause_guard += 1 } else { self.stats.cause_line += 1 }
                                if self.causes.len() < 512 { self.causes.push(CauseRec { kind: k, target, named, ends: c.ends }); }
                                true
                            }
                            None => { self.stats.est_stop += 1; false }
                        }
                    } else { bearings_changed(&before, &after, &e.frame_places, to, &est_line) };
                    if changed {
                        self.stats.est_cont += 1;
                        let r = self.solve(&c_state, &c_ident, c_jm, xr, left - 1, nq, owed, child_cr, goal);
                        Rc::new(r.iter().map(|q| QP { fl: q.fl | F_EST_READ, ..*q }).collect())
                    } else {
                        Rc::new(vec![QP { p: QPair::ZERO, fl: F_EST_READ, c: 0 }])
                    }
                } else { self.solve(&c_state, &c_ident, c_jm, xr, left - 1, nq, owed, child_cr, goal) };
                let (lx, lo) = (c_lost & own_mask, c_lost & opp_mask);
                if sub.iter().any(|q| q.fl & F_CIRCLE != 0) { saw_circle = true; }
                let shifted: Vec<QP> = sub.iter().map(|q| QP { p: if q.p == QPair::BOTTOM { q.p } else { QPair { lx: q.p.lx | lx, lo: q.p.lo | lo } }, fl: q.fl | if c.follow { F_FOLLOW } else { 0 } | if c.outside { F_OUTSIDE } else { 0 }, c: q.c.saturating_add(is_x as u8) }).collect();
                                if is_x {
                    let pure_loss = shifted.iter().all(|q| q.fl & F_PRUNED != 0 || (q.fl & F_UNREAD == 0 && q.p != QPair::BOTTOM && q.p.lx != 0 && q.p.lo == 0));
                    let hold = shifted.iter().any(|q| q.fl & BAD == 0 && q.p.lx == 0);
                    let satisfied = if goal == Goal::All { hold } else { shifted.iter().any(|q| q.fl & BAD == 0 && goal.sat(q.p)) };
                    if pure_loss && !hold {
                        // a pure loss: a member lost against a reply and none gained
                        let wk: MKey = (self.g.key(&c_state), hash_ident(&c_ident), xr as u8, left - 1, nq, owed, credit_key(child_cr), goal.code(), c_jm);
                        let w = self.wit.get(&wk).copied();
                        let moved_lost = shifted.iter().any(|q| q.p.lx & (1u64 << c.moved) != 0 && q.p != QPair::BOTTOM) || shifted.iter().any(|q| q.p == QPair::BOTTOM);
                        if moved_lost { self.stats.pruned_moved += 1 } else { self.stats.pruned_other += 1 }
                        if left == self.depth && owed == 0 && goal == Goal::Hold {
                            let first = shifted.iter().filter(|q| q.p != QPair::BOTTOM).map(|q| q.p.lx).find(|x| *x != 0);
                            let lb = first.map_or(c.moved, |f| f.trailing_zeros() as u8);
                            if let Some(lc) = cell_of_id(&self.root_ident, lb) { self.prunes.push(PruneRec { mover, ends: c.ends, lost: lc, moved_lost, reply: w.map(|x| x.0) }); }
                        }
                        if goal.frame_level() {
                            used_prune = true;
                            options.push(Rc::new(shifted.into_iter().map(|q| QP { fl: q.fl | F_PRUNED, ..q }).collect()));
                            continue;
                        }
                    }
                    options.push(Rc::new(shifted));
                    if satisfied { self.stats.stops += 1; break; }
                } else {
                    let pure_loss_for_x = shifted.iter().all(|q| q.fl & F_UNREAD == 0 && (q.p.lx | cr.lx) != 0 && (q.p.lo | cr.lo) == 0);
                    let no_unread = shifted.iter().all(|q| q.fl & F_UNREAD == 0);
                    let dead = goal != Goal::All && no_unread && !shifted.iter().any(|q| q.fl & BAD == 0 && goal.sat(QPair { lx: q.p.lx | cr.lx, lo: q.p.lo | cr.lo }));
                    options.push(Rc::new(shifted));
                    if dead { cut = Some((c.ends, pure_loss_for_x)); break; }
                    if goal == Goal::All && pure_loss_for_x && asked >= 1 { early = Some(c.ends); break; }
                }
            }
        }
        if let Some((ends, pure)) = cut {
            // the other side has a reply after which the goal cannot hold: the node answers no (a pure loss when the reply loses a member of ours and gains none)
            let r = Rc::new(vec![QP { p: QPair::BOTTOM, fl: F_CUT | if pure { F_PRUNED } else { 0 }, c: 0 }]);
            if pure { self.wit.insert(mk, (ends, 0)); }
            if !saw_circle { self.memo.insert(mk, r.clone()); }
            return r;
        }
        // the pass and the stand-pat
        if !hard_end || owed_now {
            if is_x {
                if (asked == 0 || owed_now) && !leaf {
                    if let Some(sp) = e.pass.clone() {
                        if !owed_now { self.stats.forced_pass += 1; }
                        let sub = self.solve(&sp, ident, e.mask, xr, left - 1, quiet.saturating_sub(1), if owed_now { owed - 1 } else { owed }, QPair::ZERO, goal);
                        options.push(sub);
                    }
                }
            } else if !leaf && early.is_none() {
                if let Some(sp) = e.pass.clone() { let sub = self.solve(&sp, ident, e.mask, xr, left - 1, quiet.saturating_sub(1), owed, QPair::ZERO, goal); options.push(sub); }
            }
        }
        if leaf || hard_end {
            self.stats.bounded += 1;
            // standing pat is not free when the mover has members that the other side takes for nothing: the other side is asked (a static balance is no result). Pessimistic for the side read (all of them
            // are lost), for the other side only what is certain (a single attacked member is lost)
            let pat = if e.att == 0 { QPair::ZERO } else if is_x { QPair { lx: e.att, lo: 0 } } else { QPair { lx: 0, lo: if e.att.count_ones() == 1 { e.att } else { 0 } } };
            options.push(Rc::new(vec![QP { p: pat, fl: F_BOUND, c: 0 }]));
        }
        let mut result: Vec<QP> = if options.is_empty() {
            // a side with no frame move and no pass: no claim rests on its line when it is the side read; the other side's line stands as it is (its moves can only be outside the frame), marked as bound
            self.stats.stuck += 1;
            vec![QP { p: QPair::ZERO, fl: if is_x { F_STUCK } else { F_BOUND }, c: 0 }]
        } else if is_x {
            let clean_exists = options.iter().any(|o| o.iter().any(|q| q.fl & F_PRUNED == 0));
            let all: Vec<QP> = options.iter().flat_map(|o| o.iter().copied()).filter(|q| !clean_exists || q.fl & F_PRUNED == 0).collect();
            let u = qunbeaten(all);
            if u.len() > MAX_QP { self.stats.unread += 1; vec![QP { p: QPair::BOTTOM, fl: F_UNREAD, c: 0 }] } else { u }
        } else {
            match meet_all(&options) { Some(v) => v, None => { self.stats.unread += 1; vec![QP { p: QPair::BOTTOM, fl: F_UNREAD, c: 0 }] } }
        };
        if let Some(ends) = early {
            // an adversary reply that loses a member of the side read and gains none: the node is a pure loss for it (the replies not read can only make it worse for X)
            for q in result.iter_mut() { q.fl |= F_PRUNED; }
            self.wit.insert(mk, (ends, 0));
        }
        if used_prune { for q in result.iter_mut() { q.fl |= F_PRUNE_USED; } }
        if used_est { for q in result.iter_mut() { q.fl |= F_EST_PRUNE; } }
        let r = Rc::new(result);
        if !saw_circle && !r.iter().any(|q| q.fl & (F_UNREAD | F_CIRCLE) != 0) { self.memo.insert(mk, r.clone()); }
        r
    }

    /// What `xr` can force from `s` (or from the position after the side to move passes, `flipped`), the other side may pass anywhere; `owed` moves of `xr` are forced passes first (the tempo charge).
    pub fn forceable(&mut self, s: &G::State, xr: Role, flipped: bool, owed: u8, goal: Goal) -> Option<QForce> {
        let ident = self.root_ident.clone();
        let jm0 = self.expand(s, &ident, 0).mask;
        let root = if flipped { self.expand(s, &ident, jm0).pass.clone()? } else { s.clone() };
        let before = self.stats;
        self.nodes_left = self.fuse.min(self.frame_fuse.saturating_sub(self.spent));
        let (d0, q0) = (self.depth, self.quiet);
        let pairs = self.solve(&root, &ident, jm0, xr, d0, q0, owed, QPair::ZERO, goal);
        Some(QForce { pairs: pairs.to_vec(), stats: self.stats.since(&before) })
    }

    pub fn root_state_after_pass(&mut self, s: &G::State) -> Option<G::State> { let ident = self.root_ident.clone(); let jm0 = self.expand(s, &ident, 0).mask; self.expand(s, &ident, jm0).pass.clone() }

    /// The moves of `who` (to move at `s`) by units that are not members and that are members after the move: the arrivals that change the frame.
    fn arrivals(&mut self, s: &G::State, who: Role) -> Vec<(Place, Place, Kind, G::State, Ident, u64)> {
        let g = self.g;
        let ident = self.root_ident.clone();
        let part = self.part(s);
        let Some(v) = view(&part, self.np, &ident, &self.seed_ids, &self.seed_places, self.vmode) else { return vec![] };
        if g.mover(s) != Some(who) { return vec![]; }
        let prev: Vec<Cell> = ident.iter().map(|x| x.0).collect();
        let mut out = vec![];
        for m in g.moves(s) {
            let (from, to) = g.move_ends(s, &m);
            let Some(kind) = g.kind_at(s, from) else { continue };
            let Some(mid) = id_at(&ident, &(who, kind, from)) else { continue };
            if v.mask & (1u64 << mid) != 0 { continue; }
            let s2 = g.apply(s, &m);
            self.stats.positions += 1;
            let next = sorted_cells(g, &s2);
            let (nid, _) = self.child_ident(&prev, &next, &ident, v.mask);
            let part2 = self.part(&s2);
            let Some(v2) = view(&part2, self.np, &nid, &self.seed_ids, &self.seed_places, self.vmode) else { continue };
            if v2.mask & (1u64 << mid) == 0 || v2.members.len() > MAXQ { continue; }
            out.push((from, to, kind, s2, nid, v2.mask));
        }
        out
    }

    fn solve_at(&mut self, s2: &G::State, nid: &Ident, jm: u64, xr: Role, goal: Goal) -> QForce {
        let before = self.stats;
        self.nodes_left = self.fuse.min(self.frame_fuse.saturating_sub(self.spent));
        let (d0, q0) = (self.depth, self.quiet);
        let pairs = self.solve(s2, nid, jm, xr, d0, q0, 0, QPair::ZERO, goal);
        QForce { pairs: pairs.to_vec(), stats: self.stats.since(&before) }
    }
}

// ------------------------------------------------------------------------------------------------------------------------------------------------------------------ claims

/// A condition: a move of `owner` by a unit that was not a member, which makes it one; `moves` lists every (from, to) of such a move that makes the claim hold.
#[derive(Debug, Clone)]
pub struct QCond { pub owner: Role, pub kind: Kind, pub from: Place, pub to: Place, pub arrivals: u32, pub moves: Vec<(Place, Place)> }

#[derive(Debug, Clone)]
pub struct QClaim { pub reading: Role, pub kind: ClaimKind, pub unit: u8, pub status: Status, pub flags: Fl, pub condition: Option<QCond> }

impl QClaim {
    /// A claim that holds but rests on a bound, a prune or an estimate is READ, never forced (78.58 D).
    pub fn is_read(&self) -> bool { self.status != Status::Not && self.flags & READ_MARKS != 0 }
    pub fn status_word(&self) -> &'static str { if self.is_read() { "read" } else { self.status.word() } }
}

#[derive(Debug, Clone)]
pub struct QReport {
    pub frame: QFrame,
    pub mover: Role,
    pub claims: Vec<QClaim>,
    /// THE FRAME'S OWN QUESTION for the side to move: can it lose no member (hold), can it lose none and take one (gain). None when the fuse ended the reading.
    pub holds: Option<bool>,
    pub gains: Option<bool>,
    pub arrivals_tried: u32,
    pub stats: QStats,
    pub prunes: Vec<PruneRec>,
    /// 78.60 C: the named causes found for an exchange to go on below (at most 512 kept).
    pub causes: Vec<CauseRec>,
}

/// The marks every clean guarantee satisfying the goal carries (an AND), None when none does.
fn goal_flags(f: &Option<QForce>, goal: Goal) -> Option<Fl> { f.as_ref().and_then(|f| f.pairs.iter().filter(|q| q.fl & BAD == 0 && goal.sat(q.p)).map(|q| q.fl).reduce(|a, b| a & b)) }

fn goal_holds(f: &Option<QForce>, kind: ClaimKind, id: u8) -> Option<Fl> { f.as_ref().and_then(|f| support(&f.pairs, kind, id)) }

/// THE CLAIMS of a frame on a position. Each claim is one question (a goal) read with the cut: forced (holds with the side read spending no pass), threat (holds if the side to move passes once),
/// conditional (after one arrival of the side read, for either side). The frame's own question (hold, gain) is asked first for the side to move: its pruned candidates are the prunes of the report.
pub fn claims_q<G: Game + Placed>(g: &G, st: &Static, frame: &QFrame, s: &G::State, conditional: bool, fuse: u64) -> QReport { claims_q_with(g, st, frame, s, conditional, fuse, None) }

/// `claims_q` with the worth of the kinds (78.58): with it the read prunes a taking whose exchange loses by the estimate and reads one that wins or is level as one question.
pub fn claims_q_with<G: Game + Placed>(g: &G, st: &Static, frame: &QFrame, s: &G::State, conditional: bool, fuse: u64, kv: Option<Rc<Vec<Val>>>) -> QReport { claims_q_rules(g, st, frame, s, conditional, fuse, kv, Rules::default(), None) }

/// `claims_q_with` and the rules of 78.60 (settled ends, progress, named causes, the band of the estimate). With every rule off this is the 78.58 read.
#[allow(clippy::too_many_arguments)]
pub fn claims_q_rules<G: Game + Placed>(g: &G, st: &Static, frame: &QFrame, s: &G::State, conditional: bool, fuse: u64, kv: Option<Rc<Vec<Val>>>, rules: Rules, bands: Option<Rc<Vec<(Val, Val)>>>) -> QReport {
    let mut rd = QReader::new(g, st, frame, s);
    rd.fuse = fuse;
    rd.kv = kv;
    rd.rules = rules;
    rd.bands = bands;
    let mover = g.mover(s).unwrap_or(0);
    let ids: Vec<u8> = (0..frame.members.len()).map(|i| rd.member_id(i)).collect();
    let hold_read = rd.forceable(s, mover, false, 0, Goal::Hold);
    let prunes = rd.prunes.clone();
    let holds = hold_read.as_ref().filter(|f| f.stats.unread == 0).map(|f| f.pairs.iter().any(|q| q.fl & BAD == 0 && q.p.lx == 0));
    let gain_read = rd.forceable(s, mover, false, 0, Goal::Gain);
    let gains = gain_read.as_ref().filter(|f| f.stats.unread == 0).map(|f| f.pairs.iter().any(|q| q.fl & BAD == 0 && Goal::Gain.sat(q.p)));
    // Q0 at the root: the members that the other side bears on. A claim about a member that nothing bears on is not asked unless a frame question names it.
    let attacked: Vec<bool> = frame.members.iter().map(|c| g.strikes_place(s, 1 - c.0.min(1), c.2)).collect();
    let mut claims: Vec<QClaim> = vec![];
    for reading in 0..2usize {
        // the frame's questions for this side, now and (for the side not to move) after the side to move has passed
        let hold_now = if reading == mover { hold_read.clone() } else { rd.forceable(s, reading, false, 0, Goal::Hold) };
        let gain_now = if reading == mover { gain_read.clone() } else { rd.forceable(s, reading, false, 0, Goal::Gain) };
        let (hold_flip, gain_flip) = if reading != mover { (rd.forceable(s, reading, true, 0, Goal::Hold), rd.forceable(s, reading, true, 0, Goal::Gain)) } else { (None, None) };
        for (i, c) in frame.members.iter().enumerate() {
            let own = c.0.min(1) == reading;
            let bit = 1u64 << ids[i];
            let kinds: &[ClaimKind] = if own { &[ClaimKind::Safe] } else { &[ClaimKind::TakeFree, ClaimKind::TakeAny] };
            let mut free: (Option<Fl>, Option<Fl>) = (None, None);
            for &kind in kinds {
                let goal = match kind { ClaimKind::Safe => Goal::Safe(ids[i]), ClaimKind::TakeFree => Goal::TakeFree(ids[i]), ClaimKind::TakeAny => Goal::TakeAny(ids[i]) };
                // forced: named by the frame's own question, or asked on its own when the unit stands attacked
                let named = |hold: &Option<QForce>, gain: &Option<QForce>| -> Option<Fl> {
                    match kind {
                        ClaimKind::Safe => goal_flags(hold, Goal::Hold),
                        ClaimKind::TakeFree => gain.as_ref().and_then(|f| f.pairs.iter().filter(|q| q.fl & BAD == 0 && Goal::Gain.sat(q.p) && q.p.lo & bit != 0).map(|q| q.fl).reduce(|a, b| a & b)),
                        ClaimKind::TakeAny => None,
                    }
                };
                let ask = |rd: &mut QReader<G>, flipped: bool| -> Option<Fl> { let f = rd.forceable(s, reading, flipped, 0, goal); goal_holds(&f, kind, ids[i]) };
                let mut got = if kind == ClaimKind::TakeAny && free.0.is_some() { free.0 } else { named(&hold_now, &gain_now).or_else(|| if attacked[i] { ask(&mut rd, false) } else { None }) };
                if kind == ClaimKind::TakeFree { free.0 = got; }
                let mut status = if got.is_some() { Status::Forced } else { Status::Not };
                if got.is_none() && reading != mover {
                    let gt = if kind == ClaimKind::TakeAny && free.1.is_some() { free.1 } else { named(&hold_flip, &gain_flip).or_else(|| if attacked[i] { ask(&mut rd, true) } else { None }) };
                    if kind == ClaimKind::TakeFree { free.1 = gt; }
                    if gt.is_some() { status = Status::Threat; got = gt; }
                }
                claims.push(QClaim { reading, kind, unit: i as u8, status, flags: got.unwrap_or(0), condition: None });
            }
        }
    }
    let mut arrivals_tried = 0u32;
    if conditional {
        let root_pass = rd.root_state_after_pass(s);
        let mut sides: Vec<(Role, G::State)> = vec![];
        if g.mover(s).is_some() { sides.push((mover, s.clone())); }
        if let Some(sp) = root_pass { sides.push((1 - mover.min(1), sp)); }
        for (who, st0) in sides {
            let mut seen: FxHashSet<(Kind, Place)> = FxHashSet::default();
            for (from, to, kind, s2, nid, jm2) in rd.arrivals(&st0, who) {
                if !seen.insert((kind, to)) {
                    for cr in claims.iter_mut() { if let Some(c) = &mut cr.condition { if c.kind == kind && c.to == to && c.owner == who { c.arrivals += 1; c.moves.push((from, to)); } } }
                    continue;
                }
                arrivals_tried += 1;
                for cr in claims.iter_mut() {
                    if cr.reading != who || cr.status != Status::Not || cr.condition.is_some() || cr.kind != ClaimKind::TakeFree { continue; }
                    let f = rd.solve_at(&s2, &nid, jm2, who, Goal::TakeFree(ids[cr.unit as usize]));
                    let Some(fl) = support(&f.pairs, cr.kind, ids[cr.unit as usize]) else { continue };
                    cr.status = Status::Conditional;
                    cr.flags = fl;
                    cr.condition = Some(QCond { owner: who, kind, from, to, arrivals: 1, moves: vec![(from, to)] });
                }
            }
        }
    }
    QReport { frame: frame.clone(), mover, claims, holds, gains, arrivals_tried, stats: rd.stats, prunes, causes: rd.causes.clone() }
}

// ------------------------------------------------------------------------------------------------------------------------------------------------------------------ layer 3

#[derive(Debug, Clone, Default)]
pub struct L3Q { pub nodes: Vec<(&'static str, String)>, pub links: Vec<(&'static str, String, String)>, pub claim_status: Vec<(String, &'static str)>, pub claim_marks: Vec<(String, Vec<&'static str>)> }

pub fn claim_id(st: &Static, f: &QFrame, c: &QClaim) -> String {
    format!("claim:{}:{}:{}:{}", f.id, st.names.roles[c.reading], claim_kind_word(c.kind), unit_id(st, &f.members[c.unit as usize]))
}

pub fn prune_id(st: &Static, f: &QFrame, p: &PruneRec) -> String {
    let mv = format!("{}->{}", st.names.places[p.ends.0 as usize], st.names.places[p.ends.1 as usize]);
    let to = p.reply.map_or("a reply".to_string(), |r| format!("{}->{}", st.names.places[r.0 as usize], st.names.places[r.1 as usize]));
    format!("prune:{}:{}:loses:{}:to:{}", f.id, mv, unit_id(st, &p.lost), to)
}

pub fn layer3_q(st: &Static, r: &QReport) -> L3Q {
    let mut out = L3Q::default();
    let f = &r.frame;
    out.nodes.push(("frame", f.id.clone()));
    for c in &f.seed.units { out.links.push(("has_unit", f.id.clone(), unit_id(st, c))); }
    for p in &f.places { out.links.push(("has_place", f.id.clone(), format!("place:{}", st.names.places[*p as usize]))); }
    for c in &r.claims {
        if c.status == Status::Not { continue; }
        let m = &f.members[c.unit as usize];
        let cid = claim_id(st, f, c);
        out.nodes.push(("claim", cid.clone()));
        out.claim_status.push((cid.clone(), c.status_word()));
        out.claim_marks.push((cid.clone(), marks_word(c.flags)));
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
    for p in &r.prunes {
        let pid = prune_id(st, f, p);
        out.nodes.push(("prune", pid.clone()));
        out.links.push(("pruned_in", pid.clone(), f.id.clone()));
        out.links.push(("loses_unit", pid, unit_id(st, &p.lost)));
    }
    out
}

/// A pruned root move and the claim of another frame that the same move serves.
#[derive(Debug, Clone)]
pub struct Reopen { pub prune: String, pub claim: String, pub frame_of_claim: String, pub reopened: bool, pub why: &'static str }

/// Where the cells of `cells` stand after the move from `s` to `s2` (None when one is taken).
fn follow_cells<G: Game + Placed>(g: &G, s: &G::State, s2: &G::State, cells: &[Cell]) -> Option<Vec<Cell>> {
    let (before, after) = (sorted_cells(g, s), sorted_cells(g, s2));
    let (steps, _) = step_between(&before, &after);
    let mut out = vec![];
    for c in cells {
        let ix = before.binary_search(c).ok()?;
        let Step::Alive(j) = steps[ix] else { return None };
        out.push(after[j as usize]);
    }
    out.sort();
    Some(out)
}

/// E'. For each root candidate pruned in one frame (the frame's own question, hold), every claim of the side to move in ANOTHER frame that the same move serves: after the move that claim holds forced
/// (the other frame derived again at the new position, its units followed through the move). It is reopened only if the claim still holds forced after the other side has taken here (the pruned
/// move, its reason reply, then the side to move again). The tempo is charged by that second reading: the taking costs the other side its move.
pub fn reopen<G: Game + Placed>(g: &G, st: &Static, reports: &[QReport], s: &G::State, fuse: u64) -> Vec<Reopen> {
    let mut out = vec![];
    let Some(mover) = g.mover(s) else { return out };
    for (i, ri) in reports.iter().enumerate() {
        let mut seen_moves: FxHashSet<(Place, Place)> = FxHashSet::default();
        for p in ri.prunes.iter().filter(|p| p.mover == mover) {
            if !seen_moves.insert(p.ends) { continue; }
            let Some(m1) = g.find_step(s, p.ends.0, p.ends.1) else { continue };
            let s1 = g.apply(s, &m1);
            let s2 = p.reply.and_then(|rp| g.find_step(&s1, rp.0, rp.1)).map(|m2| g.apply(&s1, &m2));
            for (j, rj) in reports.iter().enumerate() {
                if i == j { continue; }
                // the frame of the other claim as it stands after the pruned move
                let Some(seed_cells) = follow_cells(g, s, &s1, &rj.frame.seed.units) else { continue };
                let seed1 = Seed { units: seed_cells, places: rj.frame.seed.places.clone() };
                let Ok(frame_one) = QFrame::new(st, g, &s1, &seed1) else { continue };
                // the move serves a claim of that frame only if the moved unit belongs to it after the move
                let Some(moved_to) = g.kind_at(&s1, p.ends.1).map(|k| (mover, k, p.ends.1)) else { continue };
                if !frame_one.members.contains(&moved_to) { continue; }
                let mut rd1 = QReader::new(g, st, &frame_one, &s1);
                rd1.fuse = fuse;
                for cj in rj.claims.iter().filter(|c| c.reading == mover && c.status != Status::Forced && matches!(c.kind, ClaimKind::TakeFree | ClaimKind::TakeAny)) {
                    let Some(tc) = follow_cells(g, s, &s1, &[rj.frame.members[cj.unit as usize]]).map(|v| v[0]) else { continue };
                    let Some(ix) = frame_one.members.iter().position(|c| *c == tc) else { continue };
                    let id = rd1.member_id(ix);
                    let goal = if cj.kind == ClaimKind::TakeFree { Goal::TakeFree(id) } else { Goal::TakeAny(id) };
                    let serves = rd1.forceable(&s1, mover, false, 0, goal).is_some_and(|f| holds_q(&f.pairs, cj.kind, id));
                    if !serves { continue; }
                    let pid = prune_id(st, &ri.frame, p);
                    let cid = claim_id(st, &rj.frame, cj);
                    let (reopened, why) = (|| {
                        let Some(s2) = &s2 else { return (false, "no reply to take here is known") };
                        if g.mover(s2) != Some(mover) { return (false, "the side to move is not ours again"); }
                        let Some(seed_cells) = follow_cells(g, &s1, s2, &seed1.units) else { return (false, "a seed unit of the other frame is taken") };
                        let seed2 = Seed { units: seed_cells, places: rj.frame.seed.places.clone() };
                        let Ok(frame_two) = QFrame::new(st, g, s2, &seed2) else { return (false, "the other frame is not readable after the taking") };
                        let Some(tc2) = follow_cells(g, &s1, s2, &[tc]).map(|v| v[0]) else { return (false, "the claimed unit is taken after the taking") };
                        let Some(ix2) = frame_two.members.iter().position(|c| *c == tc2) else { return (false, "the claimed unit is not a member after the taking") };
                        let mut rd2 = QReader::new(g, st, &frame_two, s2);
                        rd2.fuse = fuse;
                        let id2 = rd2.member_id(ix2);
                        let goal2 = if cj.kind == ClaimKind::TakeFree { Goal::TakeFree(id2) } else { Goal::TakeAny(id2) };
                        if rd2.forceable(s2, mover, false, 0, goal2).is_some_and(|f| holds_q(&f.pairs, cj.kind, id2)) { (true, "still forced after the other side has taken here") } else { (false, "not forced once the other side has taken here") }
                    })();
                    out.push(Reopen { prune: pid, claim: cid, frame_of_claim: rj.frame.id.clone(), reopened, why });
                }
            }
        }
    }
    out
}

// ------------------------------------------------------------------------------------------------------------------------------------------------------------------ two frames

/// COMPOSITION WITH THE TEMPO CHARGED, for the side to move: it plays the first frame for c moves (c passes of it in the second frame, in which the other side moves), both orders, guarantees united.
/// Pairs are over the identity bits of the root position, the same space for both frames. None when the side to move cannot read (game over). CHECKED, NOT PROVED.
pub fn compose_q<G: Game + Placed>(g: &G, st: &Static, a: &QFrame, b: &QFrame, s: &G::State, fuse: u64) -> Option<Vec<QP>> {
    let x = g.mover(s)?;
    let mut ra = QReader::new(g, st, a, s); ra.fuse = fuse;
    let mut rb = QReader::new(g, st, b, s); rb.fuse = fuse;
    let mut all: Vec<QP> = vec![];
    compose_order(&mut ra, &mut rb, s, x, &mut all)?;
    compose_order(&mut rb, &mut ra, s, x, &mut all)?;
    Some(qunbeaten(all))
}

fn compose_order<G: Game + Placed>(first: &mut QReader<G>, second: &mut QReader<G>, s: &G::State, x: Role, all: &mut Vec<QP>) -> Option<()> {
    let fr_one = first.forceable(s, x, false, 0, Goal::All)?;
    let mut by_cost: FxHashMap<u8, Vec<QP>> = FxHashMap::default();
    for q in fr_one.pairs.iter().filter(|q| q.fl & BAD == 0) {
        let owed = q.c;
        let fr_two = match by_cost.get(&owed) { Some(v) => v.clone(), None => { let v = second.forceable(s, x, false, owed, Goal::All).map(|f| f.pairs).unwrap_or_default(); by_cost.insert(owed, v.clone()); v } };
        for r in fr_two.iter().filter(|r| r.fl & BAD == 0) { all.push(QP { p: QPair { lx: q.p.lx | r.p.lx, lo: q.p.lo | r.p.lo }, fl: q.fl | r.fl, c: q.c.saturating_add(r.c) }); }
    }
    Some(())
}

pub use super::frames::Cmp;

fn dom_sets(a: &[QP], b: &[QP]) -> bool { b.iter().filter(|q| q.fl & BAD == 0).all(|q| a.iter().filter(|p| p.fl & BAD == 0).any(|p| p.p.at_least(q.p))) }

pub fn compare_q(a: &[QP], b: &[QP]) -> Cmp {
    match (dom_sets(a, b), dom_sets(b, a)) { (true, true) => Cmp::Equal, (true, false) => Cmp::FirstMore, (false, true) => Cmp::SecondMore, _ => Cmp::Incomparable }
}

// ------------------------------------------------------------------------------------------------------------------------------------------------------------------ the trap question

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrapWord { Free, Trapped, Immobile, NotQuiet, Ended, Unread }

impl TrapWord { pub fn word(self) -> &'static str { match self { TrapWord::Free => "free", TrapWord::Trapped => "trapped", TrapWord::Immobile => "immobile", TrapWord::NotQuiet => "not_quiet", TrapWord::Ended => "ended", TrapWord::Unread => "unread" } } }

#[derive(Debug, Clone)]
pub struct TrapAnswer { pub word: TrapWord, pub free: Vec<Place>, pub bad: Vec<Place>, pub stats: QStats }

/// ONE QUESTION FOR ANY UNIT: on quiet routes, has it a place to go that is not a pure loss. The owner is put to move (the game's pass when it is not); where the game refuses that pass the owner
/// is under a forcing threat and the answer is NOT QUIET (read under a forcing move everything looks trapped); where the owner has no move at all the game itself has ended. A unit that no side bears on
/// is Free (it may stay). A place is bad when, after the unit goes there, the other side can force it taken, none of its own lost.
pub fn trap<G: Game + Placed>(g: &G, st: &Static, s: &G::State, unit: Cell, fuse: u64) -> TrapAnswer {
    let mut ans = TrapAnswer { word: TrapWord::Free, free: vec![], bad: vec![], stats: QStats::default() };
    let owner = unit.0;
    let Some(mv) = g.mover(s) else { ans.word = TrapWord::Ended; return ans };
    let so = if mv == owner { s.clone() } else { match g.pass(s) { Some(x) => x, None => { ans.word = TrapWord::NotQuiet; return ans; } } };
    if g.moves(&so).is_empty() { ans.word = TrapWord::Ended; return ans; }
    if g.pass(&so).is_none() { ans.word = TrapWord::NotQuiet; return ans; }
    let moves: Vec<G::Move> = g.moves(&so).into_iter().filter(|m| g.move_ends(&so, m).0 == unit.2).collect();
    if moves.is_empty() { ans.word = TrapWord::Immobile; return ans; }
    if !g.strikes_place(&so, 1 - owner.min(1), unit.2) { ans.word = TrapWord::Free; return ans; }
    for m in &moves {
        let (_f, to) = g.move_ends(&so, m);
        let s1 = g.apply(&so, m);
        ans.stats.positions += 1;
        let Some(k1) = g.kind_at(&s1, to) else { ans.free.push(to); continue };
        let seed = Seed { units: vec![(owner, k1, to)], places: vec![] };
        let Ok(f) = QFrame::new(st, g, &s1, &seed) else { ans.word = TrapWord::Unread; continue };
        let mut rd = QReader::new(g, st, &f, &s1);
        rd.fuse = fuse;
        let id = rd.member_id(f.members.iter().position(|c| *c == (owner, k1, to)).unwrap_or(0));
        let other = 1 - owner.min(1);
        let r = rd.forceable(&s1, other, false, 0, Goal::TakeFree(id));
        ans.stats.add(&rd.stats);
        match r { Some(r) if holds_q(&r.pairs, ClaimKind::TakeFree, id) => ans.bad.push(to), _ => ans.free.push(to) }
    }
    if ans.word == TrapWord::Unread { return ans; }
    ans.word = if ans.free.is_empty() { TrapWord::Trapped } else { TrapWord::Free };
    ans
}

#[cfg(test)]
mod tests {
    use super::*;

    fn qp(lx: u64, lo: u64, c: u8) -> QP { QP { p: QPair { lx, lo }, fl: 0, c } }

    #[test]
    fn a_cheaper_guarantee_is_kept_beside_a_better_dearer_one() {
        let v = qunbeaten(vec![qp(0, 1, 2), qp(0, 0, 0), qp(0, 1, 1)]);
        assert!(v.contains(&qp(0, 1, 1)) && v.contains(&qp(0, 0, 0)) && v.len() == 2);
    }

    #[test]
    fn a_marked_guarantee_grounds_no_claim_and_never_beats_a_clean_one() {
        let bad = QP { p: QPair::ZERO, fl: F_STUCK, c: 0 };
        assert!(!holds_q(&[bad], ClaimKind::Safe, 3));
        let v = qunbeaten(vec![bad, qp(0, 0, 0)]);
        assert_eq!(v, vec![qp(0, 0, 0)]);
    }

    #[test]
    fn the_follow_mark_is_on_a_claim_only_if_every_support_has_it() {
        let a = QP { p: QPair { lx: 0, lo: 2 }, fl: F_FOLLOW, c: 0 };
        let b = qp(0, 2, 1);
        assert_eq!(support(&[a, b], ClaimKind::TakeFree, 1), Some(0));
        assert_eq!(support(&[a], ClaimKind::TakeFree, 1), Some(F_FOLLOW));
        assert_eq!(support(&[a], ClaimKind::TakeFree, 2), None);
    }

    #[test]
    fn what_a_claim_rests_on_is_ranked_and_a_circling_line_grounds_nothing() {
        assert_eq!(basis_rank(0), 0);
        assert!(basis_rank(F_PRUNE_USED) < basis_rank(F_EST_READ) && basis_rank(F_EST_READ) == basis_rank(F_EST_PRUNE) && basis_rank(F_EST_PRUNE) < basis_rank(F_BOUND));
        assert_eq!(basis_rank(F_CIRCLE), basis_rank(F_BOUND));
        assert_eq!(basis_rank(F_EST_READ | F_BOUND), 3);
        let c = QP { p: QPair::ZERO, fl: F_CIRCLE, c: 0 };
        assert!(!holds_q(&[c], ClaimKind::Safe, 3));
        assert_eq!(qunbeaten(vec![c, qp(0, 0, 0)]), vec![qp(0, 0, 0)]);
    }

    #[test]
    fn the_meet_of_two_replies_keeps_what_both_give() {
        let s = vec![Rc::new(vec![qp(1, 0, 1)]), Rc::new(vec![qp(0, 2, 2)])];
        assert_eq!(meet_all(&s).unwrap(), vec![qp(1, 0, 2)]);
    }
}
