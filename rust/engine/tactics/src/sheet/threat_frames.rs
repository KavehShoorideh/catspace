//! PLAN 79.08 B (SYSTEM_SPEC 8 step 2, "split the position into frames and ask the questions inside them"): the FRAME READ OF ONE POSITION for the threat list, in process. The body of the binding's `questions_threats` (plan 78.70 A) moved here so that the wake calls it without Python and the
//! binding calls the same code (one piece, one place). Game-free: a unit is (owner, kind, place) numbers, a frame is the units the learned seeds name and what bears on them.
//! INTERFACE. `threat_frames(st, g, s, goal, want, p) -> ThreatFrames`: takes the static graph (with its learned profile), the game and position, the goal kind, the learned seeds' keys (`want`: key to its number) and `FrameReadParams` (fuses, the kind values and bands the read uses, the rules of a
//! line's end); returns, per frame read, its members as (the mover's own, kind, place), its places (the READ SET of its claims) and, per member the other side bears on (or the read holds lost, both, a threat or conditional), the class and basis of its claim, the units that bear on it and the answers of its
//! owner. `unit_class` is the class and basis of one member (shared with the binding's `questions_state`). Called by the wake (cs_decision::frame_rows) and by the binding.
use super::base_graph::Static;
use super::frames::{ClaimKind, Seed, Status};
use super::patterns::{self, Board, FxHashMap};
use super::questions::{self as q, QFrame, QReader};
use cs_game::placed::{Kind, Place, Placed};
use cs_game::Game;
use std::rc::Rc;

pub fn rank(a: Status, b: Status) -> Status {
    let r = |s: Status| match s { Status::Forced => 4, Status::Threat => 3, Status::Waiting => 2, Status::Conditional => 1, Status::Not => 0 };
    if r(b) > r(a) { b } else { a }
}

/// The class and basis of member `id` (of the frame read in `rep`) owned by `owner`, as the unit record of `StateRec` states them. Class: 0 lost for its owner (the other side forces its taking), 1 safe, 2 both, 3 unread, 4 threat, 5 conditional, 6 contested. Basis (of a lost, safe or both class): 0 forced, 1 pruned,
/// 2 estimate, 3 bound; 255 none.
pub fn unit_class(rep: &q::QReport, id: usize, owner: usize) -> (u8, u8) {
    let status_of = |id: usize, reading: usize, kinds: &[ClaimKind]| -> (Status, q::Fl) {
        let mine: Vec<&q::QClaim> = rep.claims.iter().filter(|c| c.reading == reading && c.unit as usize == id && kinds.contains(&c.kind)).collect();
        let best = mine.iter().map(|c| c.status).fold(Status::Not, rank);
        (best, mine.iter().filter(|c| c.status == best).fold(0xFFFFu16, |a, c| a & c.flags))
    };
    let (free, ffl) = status_of(id, 1 - owner, &[ClaimKind::TakeFree]);
    let (safe, sfl) = status_of(id, owner, &[ClaimKind::Safe]);
    let class: u8 = if free == Status::Forced && safe == Status::Forced { 2 } else if free == Status::Forced { 0 } else if safe == Status::Forced { 1 }
        else if rep.stats.unread > 0 { 3 } else if matches!(free, Status::Threat | Status::Waiting) { 4 } else if free == Status::Conditional { 5 } else { 6 };
    let basis: u8 = match class { 0 => q::basis_rank(ffl), 1 => q::basis_rank(sfl), 2 => q::basis_rank(ffl).max(q::basis_rank(sfl)), _ => 255 };
    (class, basis)
}

/// The parameters of the frame read: the largest seed (`max_size`), the fuse of a frame's questions and of a row (`fuse`, `row_fuse`), of the answers (`answer_fuse`), whether conditional claims are read, the learned kind values and bands (None: no exchange), the rules of a line's end.
#[derive(Debug, Clone, Default)]
pub struct FrameReadParams { pub max_size: usize, pub fuse: u64, pub row_fuse: u64, pub answer_fuse: u64, pub conditional: bool, pub kv: Option<Vec<i64>>, pub bands: Option<Vec<(i64, i64)>>, pub rules: q::Rules }

/// One claim of the threat list: a member of a frame that the other side bears on (or that the read holds lost, both, a threat or conditional): [member index, class, basis], the take basis, the units that bear on it as (kind, place), the answers found as (from, to, how, the unit moves), the candidates tried and the unread.
#[derive(Debug, Clone)]
pub struct ThreatClaim { pub member: u16, pub class: u8, pub basis: u8, pub take_basis: u8, pub bearers: Vec<(Kind, Place)>, pub answers: Vec<(Place, Place, u8, bool)>, pub tried: u32, pub unread: u32 }
/// One frame of a position: its identity, its members as (the mover's own, kind, place), its places (the read set with the members), the unread lines of its claims, and its claims.
#[derive(Debug, Clone)]
pub struct ThreatFrame { pub id: String, pub members: Vec<(bool, Kind, Place)>, pub places: Vec<Place>, pub unread: u64, pub claims: Vec<ThreatClaim> }
/// The frames read at one position and the questions asked.
#[derive(Debug, Clone, Default)]
pub struct ThreatFrames { pub frames: Vec<ThreatFrame>, pub questions: u64 }

/// THE FRAME READ OF ONE POSITION (the body of `questions_threats`, plan 78.70 A and B: the frame as built, and the smallest frame of the same seed when that is a strict part of it). Empty when the learned seeds name nothing on the board.
pub fn threat_frames<G: Game + Placed>(st: &Static, g: &G, s: &G::State, goal: Option<Kind>, want: &FxHashMap<u64, u32>, p: &FrameReadParams) -> ThreatFrames {
    let mut out = ThreatFrames::default();
    let Some(mover) = g.mover(s) else { return out };
    let part = st.position(g, s);
    let b = Board::new(st, &part, goal);
    let levels: Vec<u8> = (0..patterns::LEVELS).collect();
    let seeds: Vec<Seed> = super::frames::seeds_of_board(&b, &part, p.max_size, &levels, want).into_iter().map(|x| x.1).collect();
    if seeds.is_empty() { return out; }
    let built = q::build_frames_cap(st, g, s, &seeds, p.rules.root_cap);
    let mut row_q = 0u64;
    for f0 in &built.frames {
        // THE WIDENING (78.70 B): the frame as built, and the SMALLEST frame of the same seed (its units and what bears on them, the 78.60 F mode) when that is a strict part of it
        let mut todo: Vec<(QFrame, bool)> = vec![(f0.clone(), false)];
        if let Ok(sm) = QFrame::new_cap(st, g, s, &f0.seed, q::ROOT_CAP - 1) {
            if sm.members.len() < f0.members.len() && sm.members.iter().all(|c| f0.members.contains(c)) { todo.push((sm, true)); }
        }
        for (f, small) in &todo {
            if row_q >= p.row_fuse { continue; }
            let kv = p.kv.clone().map(Rc::new);
            let bands = p.bands.clone().map(Rc::new);
            let rep = q::claims_q_rules(g, st, f, s, p.conditional, p.fuse, kv.clone(), p.rules, bands.clone());
            row_q += rep.stats.questions;
            let mut rd = QReader::new(g, st, f, s);
            rd.fuse = p.answer_fuse;
            rd.kv = kv;
            rd.rules = p.rules;
            rd.bands = bands;
            let mut tf = ThreatFrame { id: format!("{}{}", f.id, if *small { "/small" } else { "" }), members: f.members.iter().map(|c| (c.0 == mover, c.1, c.2)).collect(), places: f.places.clone(), unread: rep.stats.unread, claims: vec![] };
            for (i, c) in f.members.iter().enumerate() {
                let owner = c.0.min(1);
                let (class, basis) = unit_class(&rep, i, owner);
                // what the taking of this unit by the other side rests on, whatever the class (255: no such claim holds)
                let take_basis = rep.claims.iter().filter(|x| x.reading == 1 - owner && x.unit as usize == i && x.kind == ClaimKind::TakeFree && x.status != Status::Not).map(|x| q::basis_rank(x.flags)).max().unwrap_or(255);
                let mut bearers: Vec<(Kind, Place)> = part.bears.iter().filter(|r| r.place == c.2).filter(|r| part.units[r.unit].owner.min(1) != owner).map(|r| (part.units[r.unit].kind, part.units[r.unit].at)).collect();
                bearers.sort();
                if bearers.is_empty() && !matches!(class, 0 | 2 | 4 | 5) { continue; }
                let mut ans = (vec![], 0u32, 0u32);
                if class != 0 && row_q < p.row_fuse {
                    let before = rd.stats.questions;
                    if let Some(a) = rd.answers(s, owner, rd.member_id(i)) {
                        ans = (a.moves.iter().map(|m| (m.ends.0, m.ends.1, m.how, m.unit_moves)).collect(), a.tried, a.unread);
                    }
                    row_q += rd.stats.questions - before;
                }
                tf.claims.push(ThreatClaim { member: i as u16, class, basis, take_basis, bearers, answers: ans.0, tried: ans.1, unread: ans.2 });
            }
            out.questions += rep.stats.questions + rd.stats.questions;
            out.frames.push(tf);
        }
    }
    out
}
