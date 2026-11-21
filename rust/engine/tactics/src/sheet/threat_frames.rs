//! PLAN 79.08 B (SYSTEM_SPEC 8 step 2, "split the position into frames and ask the questions inside them"): the FRAME READ OF ONE POSITION for the threat list, in process. The body of the binding's `questions_threats` (plan 78.70 A) moved here so that the wake calls it without Python and the
//! binding calls the same code (one piece, one place). Game-free: a unit is (owner, kind, place) numbers, a frame is the units the learned seeds name and what bears on them.
//! INTERFACE. `threat_frames(st, g, s, goal, want, p) -> ThreatFrames`: takes the static graph (with its learned profile), the game and position, the goal kind, the learned seeds' keys (`want`: key to its number) and `FrameReadParams` (fuses, the kind values and bands the read uses, the rules of a
//! line's end); returns, per frame read, its members as (the mover's own, kind, place), its places (the READ SET of its claims) and, per member the other side bears on (or the read holds lost, both, a threat or conditional), the class and basis of its claim, the units that bear on it and the answers of its
//! owner. `unit_class` is the class and basis of one member (shared with the binding's `questions_state`). Called by the wake (cs_decision::frame_rows) and by the binding.
use super::base_graph::{Root, Static};
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
pub struct FrameReadParams { pub max_size: usize, pub fuse: u64, pub row_fuse: u64, pub answer_fuse: u64, pub conditional: bool, pub kv: Option<Vec<i64>>, pub bands: Option<Vec<(i64, i64)>>, pub rules: q::Rules,
    /// PLAN 79.09 B: the clock of this step (None: the fuses only). With it the frames are read in the order of the learned worth of their largest member (`kv`), the greater first, and the read stops at it: what it did not reach is listed in `ThreatFrames::unread_frames` (a gap, never a refutation).
    pub deadline: Option<std::time::Instant> }

/// One claim of the threat list: a member of a frame that the other side bears on (or that the read holds lost, both, a threat or conditional): [member index, class, basis], the take basis, the units that bear on it as (kind, place), the answers found as (from, to, how, the unit moves), the candidates tried and the unread.
#[derive(Debug, Clone)]
pub struct ThreatClaim { pub member: u16, pub class: u8, pub basis: u8, pub take_basis: u8, pub bearers: Vec<(Kind, Place)>, pub answers: Vec<(Place, Place, u8, bool)>, pub tried: u32, pub unread: u32 }
/// One frame of a position: its identity, its members as (the mover's own, kind, place), its places (the read set with the members), the unread lines of its claims, and its claims.
#[derive(Debug, Clone)]
pub struct ThreatFrame { pub id: String, pub members: Vec<(bool, Kind, Place)>, pub places: Vec<Place>, pub unread: u64, pub claims: Vec<ThreatClaim> }
/// PLAN 79.14 A (the cost by part of the frame read): the seconds of the seeds, the frames built, the claims' questions and the answers' questions; the seeds found, the frames built, the frames whose claims were asked, the frames whose claims were carried.
#[derive(Debug, Clone, Default)]
pub struct FrameParts { pub seeds_s: f64, pub build_s: f64, pub claims_s: f64, pub answers_s: f64, pub seeds: usize, pub built: usize, pub asked: usize, pub carried: usize,
    /// PLAN 79.14 B: the matches of the last position carried (not asked again)
    pub seeds_carried: usize }

/// PLAN 79.14 B: what a frame read leaves for the next decision of the same side: the seeds found (with the atoms of each) and the frames whose claims and answers were read to the end, by id.
#[derive(Debug, Clone, Default)]
pub struct FrameMemo { pub mover: usize, pub seeds: super::delta::SeedMemo, pub frames: Vec<ThreatFrame> }

/// The frames read at one position and the questions asked.
#[derive(Debug, Clone, Default)]
pub struct ThreatFrames { pub frames: Vec<ThreatFrame>, pub questions: u64,
    /// PLAN 79.14 A: where the seconds of the read went (seeds found on the board, frames built, the claims' questions, the answers' questions) and the counts
    pub parts: FrameParts,
    /// PLAN 79.09 B: the frames the clock did not reach (their ids), and the members of the frames it cut part way (claims read after it are unread)
    pub unread_frames: Vec<String>, pub unbuilt_seeds: usize,
    /// PLAN 79.14 B: with the carry on, what the next decision may carry (None where the clock cut the reading), and what this position differs from the last in (None: no last, or another side to move)
    pub memo: Option<FrameMemo>, pub touch: Option<super::delta::Touch> }

/// THE FRAME READ OF ONE POSITION (the body of `questions_threats`, plan 78.70 A and B: the frame as built, and the smallest frame of the same seed when that is a strict part of it). Empty when the learned seeds name nothing on the board.
pub fn threat_frames<G: Game + Placed>(st: &Static, g: &G, s: &G::State, goal: Option<Kind>, want: &FxHashMap<u64, u32>, p: &FrameReadParams) -> ThreatFrames { threat_frames_with(st, g, s, goal, want, p, None, false) }

/// PLAN 79.14 B: THE FRAME READ THAT ASKS ONLY WHAT CHANGED. `keep` false: the read of `threat_frames`, nothing carried or kept. `keep` true: the seeds are found by asking only about the connected sets that hold an atom the last position did not have (`delta::seeds_carried`), and a frame whose id,
/// members and places are those of a frame the last decision read to the end, and that nothing the two plies since did touches (`delta::Touch::hits`), is carried with its claims and its answers (counted in `parts.carried`), not asked again; what the read leaves for the next decision is in `memo`.
/// `prev` is the last decision's memo (ignored when it was made with the other side to move).
#[allow(clippy::too_many_arguments)]
pub fn threat_frames_with<G: Game + Placed>(st: &Static, g: &G, s: &G::State, goal: Option<Kind>, want: &FxHashMap<u64, u32>, p: &FrameReadParams, prev: Option<&FrameMemo>, keep: bool) -> ThreatFrames { threat_frames_in(st, None, g, s, goal, want, p, prev, keep) }

/// `threat_frames_with` on the decision's root graph when it is lent (PLAN 79.22 PART A): the seeds, the frames, every frame's reader and answers read the position's graph once, the root's.
#[allow(clippy::too_many_arguments)]
pub fn threat_frames_in<G: Game + Placed>(st: &Static, root: Option<&Root>, g: &G, s: &G::State, goal: Option<Kind>, want: &FxHashMap<u64, u32>, p: &FrameReadParams, prev: Option<&FrameMemo>, keep: bool) -> ThreatFrames {
    let mut out = ThreatFrames::default();
    let Some(mover) = g.mover(s) else { return out };
    let part = match root { Some(r) => st.position_in(r, g, s), None => Rc::new(st.position(g, s)) };
    let b = Board::new(st, &part, goal);
    let levels: Vec<u8> = (0..patterns::LEVELS).collect();
    let t_seeds = std::time::Instant::now();
    let prev = prev.filter(|m| m.mover == mover);
    let (found, cut, seed_memo) = if keep {
        let r = super::delta::seeds_carried(&b, &part, p.max_size, &levels, want, p.deadline, prev.map(|m| &m.seeds));
        out.parts.seeds_carried = r.carried;
        out.touch = prev.map(|m| super::delta::Touch::between(&m.seeds.digest, &r.memo.digest));
        let cut = r.cut;
        (r.seeds, cut, Some(r.memo))
    } else { let (f, c) = super::frames::seeds_of_board_until(&b, &part, p.max_size, &levels, want, p.deadline); (f, c, None) };
    out.parts.seeds_s = t_seeds.elapsed().as_secs_f64();
    out.parts.seeds = found.len();
    out.unbuilt_seeds = usize::from(cut);
    let seeds: Vec<Seed> = found.into_iter().map(|x| x.1).collect();
    if seeds.is_empty() { if let (true, Some(sm)) = (keep, seed_memo) { if !cut { out.memo = Some(FrameMemo { mover, seeds: sm, frames: vec![] }); } } return out; }
    let mut seeds = seeds;
    let worth = |c: &(usize, cs_game::placed::Kind, Place)| p.kv.as_ref().and_then(|kv| kv.get(c.1 as usize).copied()).unwrap_or(0);
    if p.deadline.is_some() { seeds.sort_by_key(|sd| std::cmp::Reverse(sd.units.iter().map(&worth).max().unwrap_or(0))); }
    let t_build = std::time::Instant::now();
    let mut built = q::build_frames_in(st, root, g, s, &seeds, p.rules.root_cap, p.deadline);
    out.parts.build_s = t_build.elapsed().as_secs_f64();
    out.parts.built = built.frames.len();
    out.unbuilt_seeds += built.unbuilt;
    if p.deadline.is_some() { built.frames.sort_by_key(|f| std::cmp::Reverse(f.members.iter().map(&worth).max().unwrap_or(0))); }
    let expired = || p.deadline.is_some_and(|d| std::time::Instant::now() >= d);
    let mut row_q = 0u64;
    let mut kept: Vec<ThreatFrame> = vec![];
    let mut kept_whole = true;
    // a frame the last decision read to the end, the same in members and places and with nothing touched in its read set: carried
    let carried_of = |out: &ThreatFrames, f: &QFrame, small: bool| -> Option<ThreatFrame> {
        let (m, tc) = (prev?, out.touch.as_ref()?);
        let id = format!("{}{}", f.id, if small { "/small" } else { "" });
        let mm: Vec<(bool, Kind, Place)> = f.members.iter().map(|c| (c.0 == mover, c.1, c.2)).collect();
        let pf = m.frames.iter().find(|x| x.id == id && x.members == mm && x.places == f.places)?;
        if tc.hits(&f.members, &f.places) { None } else { Some(pf.clone()) }
    };
    for f0 in &built.frames {
        if expired() && carried_of(&out, f0, false).is_none() { out.unread_frames.push(f0.id.clone()); kept_whole = false; continue; }
        // THE WIDENING (78.70 B): the frame as built, and the SMALLEST frame of the same seed (its units and what bears on them, the 78.60 F mode) when that is a strict part of it
        let mut todo: Vec<(QFrame, bool)> = vec![(f0.clone(), false)];
        if let Ok(sm) = QFrame::new_cap_in(st, root, g, s, &f0.seed, q::ROOT_CAP - 1) {
            if sm.members.len() < f0.members.len() && sm.members.iter().all(|c| f0.members.contains(c)) { todo.push((sm, true)); }
        }
        for (f, small) in &todo {
            if row_q >= p.row_fuse { kept_whole = false; continue; }
            if let Some(pf) = carried_of(&out, f, *small) { out.parts.carried += 1; kept.push(pf.clone()); out.frames.push(pf); continue; }
            let kv = p.kv.clone().map(Rc::new);
            let bands = p.bands.clone().map(Rc::new);
            let t_claims = std::time::Instant::now();
            let rep = q::claims_q_rules_in(g, st, root, f, s, p.conditional, p.fuse, kv.clone(), p.rules, bands.clone(), p.deadline);
            out.parts.claims_s += t_claims.elapsed().as_secs_f64();
            out.parts.asked += 1;
            row_q += rep.stats.questions;
            let mut rd = QReader::new_in(g, st, f, s, root);
            rd.fuse = p.answer_fuse;
            rd.deadline = p.deadline;
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
                if class != 0 && row_q < p.row_fuse && !expired() {
                    let before = rd.stats.questions;
                    let t_ans = std::time::Instant::now();
                    if let Some(a) = rd.answers(s, owner, rd.member_id(i)) {
                        ans = (a.moves.iter().map(|m| (m.ends.0, m.ends.1, m.how, m.unit_moves)).collect(), a.tried, a.unread);
                    }
                    out.parts.answers_s += t_ans.elapsed().as_secs_f64();
                    row_q += rd.stats.questions - before;
                }
                tf.claims.push(ThreatClaim { member: i as u16, class, basis, take_basis, bearers, answers: ans.0, tried: ans.1, unread: ans.2 });
            }
            out.questions += rep.stats.questions + rd.stats.questions;
            if expired() { kept_whole = false; } else { kept.push(tf.clone()); }
            out.frames.push(tf);
        }
    }
    if let (true, Some(sm)) = (keep, seed_memo) { if !cut && kept_whole && out.unbuilt_seeds == 0 { out.memo = Some(FrameMemo { mover, seeds: sm, frames: kept }); } }
    out
}
