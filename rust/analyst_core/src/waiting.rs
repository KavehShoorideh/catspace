//! WAITING FRAMES (Kaveh 2026-10-09: "keep the frames and the facts in the frames, don't discard them ... a live index
//! of who's waiting on what kind of condition ... a language to communicate what condition"; "if there were another
//! attacker I'd win -- how likely can I bring one?").
//!
//! A FRAME is a seed move with a focus and a goal: a check against the king (goal: mate), a new threat or fork, a
//! cornered or trapped piece (goal: the rule layer's gain). Its certificate is the value search (`certified_value`:
//! our moves restricted to the frame's answers, EVERY reply of theirs searched).
//!   active    the certificate holds now
//!   dormant   it fails, but ONE change to the board inside the frame makes it hold -- the MISSING CONDITION
//!   far       it fails and no single change fixes it
//! THE MISSING CONDITION, found generically (no extractor per tactic family): the smallest perturbation of the board
//! inside the frame under which the certificate passes -- add one of our pieces on a square bearing on the frame,
//! remove one of theirs (a defender, a blocker), vacate one of ours (a blocker). Each condition is an ATOM over one
//! square; a frame's condition is the OR of the atoms that work. The certificates run in parallel on every core.
//! THE POOL keeps frames across moves and reports what changed (new, dormant -> active, active -> broken, dead). By
//! default every frame is recomputed every move (`exact`): reuse of frames whose footprint no move touched (occupancy
//! and attack changes) was audited on two games at 1.0% stale statuses, six missed tactics among them, for 14% of the
//! work saved (2026-10-09) -- kept as an option, off.
//! THE REVERSE LOOKUP: which of our moves makes a dormant frame of ours hold on the null move (it ARMS a threat), and
//! which makes a dormant frame of theirs hold (it HANDS them a tactic).
use super::*;
use rayon::prelude::*;
use shakmaty::{Bitboard, EnPassantMode};

pub const MATE_GOAL: i64 = 500;
/// WAITING CONDITIONS ONE MOVE AWAY (2026-10-09, measured: 85% of the frame work was ~1,000 completed refutations a
/// position -- "this hypothetical change would NOT make the frame hold" -- which a stored refutation cannot shorten):
/// an added piece of ours must be able to reach its square in one move. Conditions further away are the reachability
/// model's to estimate, not the search's to enumerate.
pub static ONE_MOVE_WAITS: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);
#[pyfunction]
pub fn set_one_move_waits(flag: bool) -> PyResult<()> { ONE_MOVE_WAITS.store(flag, Ordering::Relaxed); Ok(()) }
/// The condition and breaker searches use the 2-ply pre-check only (false) or the pre-check then the frame's depth
/// (true). Measured before choosing a default (2026-10-09).
pub static DEEP_CONDITIONS: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

#[pyfunction]
pub fn set_deep_conditions(flag: bool) -> PyResult<()> { DEEP_CONDITIONS.store(flag, Ordering::Relaxed); Ok(()) }

/// THE PROOF QUESTION of a frame: does the seed certify its goal on `p` (its owner to move)? The 2-ply pre-check, then
/// the frame's depth (unless DEEP_CONDITIONS is off), each a null-window search at the goal run with the stored proof
/// `hints` tried first -> (holds, the proof or refutation it recorded).
pub fn prove(p: &Chess, uci: &str, focus: &[Square], goal: i64, hints: Option<std::sync::Arc<Hints>>) -> (bool, Hints) {
    let (ok, h, _, _) = prove_n(p, uci, focus, goal, hints);
    (ok, h)
}

pub fn prove_n(p: &Chess, uci: &str, focus: &[Square], goal: i64, hints: Option<std::sync::Arc<Hints>>) -> (bool, Hints, u64, bool) {
    let (ok, h, n, spent, _) = prove_t(p, uci, focus, goal, hints, DEEP_CONDITIONS.load(Ordering::Relaxed));
    (ok, h, n, spent)
}

/// `prove` with its cost: (holds, proof, nodes, the last tier ran out of budget, the tier that proved it: 1 = 2 plies,
/// 2 = the frame's depth, 0 = none); `deep` off stops after the 2-ply tier.
pub fn prove_t(p: &Chess, uci: &str, focus: &[Square], goal: i64, hints: Option<std::sync::Arc<Hints>>, deep: bool) -> (bool, Hints, u64, bool, u8) {
    let Ok(u) = uci.parse::<shakmaty::uci::UciMove>() else { return (false, Hints::new(), 0, false, 0) };
    let Ok(m) = u.to_move(p) else { return (false, Hints::new(), 0, false, 0) };
    let n0 = PROVE_NODES.with(|c| c.get());
    let mut last = Hints::new();
    let mut spent = false;
    for (d, b) in [(2u32, 10_000u64), (PURSUIT_PLIES, 50_000u64)] {
        if d > 2 && !deep { break; }
        let (r, rec) = with_hints(hints.clone(), || prove_pos(p, &m, focus, d, b, goal));
        last = rec;
        spent = r.is_none();
        if r == Some(true) { return (true, last, PROVE_NODES.with(|c| c.get()) - n0, false, if d > 2 { 2 } else { 1 }); }
    }
    (false, last, PROVE_NODES.with(|c| c.get()) - n0, spent, 0)
}

// THE CALCULATION AUDIT (Kaveh 2026-10-09: "what are the calculations that are useless and lost?"): every proof search
// counted by purpose and outcome, with its nodes. Index = 2 * slot (count) and 2 * slot + 1 (nodes); slots:
pub const A_SEED_HOLD: usize = 0; pub const A_SEED_FAIL: usize = 1;
pub const A_COND_FOUND: usize = 2; pub const A_COND_REFUTED: usize = 3; pub const A_COND_SPENT: usize = 4;
pub const A_COND_FOUND_ON: usize = 5; pub const A_COND_FOUND_OFF: usize = 6;   // on / off the base refutation's squares
pub const A_COND_MISS_ON: usize = 7; pub const A_COND_MISS_OFF: usize = 8;     // refuted or spent, on / off
pub const A_BRK_BREAKS: usize = 9; pub const A_BRK_HOLDS: usize = 10; pub const A_BRK_SPENT: usize = 11;
pub const A_COND_CONTROL: usize = 12;                                            // refused by the control before any search
pub const A_FAR_AGAIN: usize = 13;                                               // a far frame that was far last move too
pub const A_ACTIVE: usize = 14; pub const A_DORMANT: usize = 15; pub const A_FAR: usize = 16;
pub const A_GATED: usize = 17;                                                   // frames whose condition search the gate skipped
pub const A_SHALLOW: usize = 18;                                                 // frames searched at 2 plies only (gate 2)
pub static AUDIT: [AtomicU64; 40] = [const { AtomicU64::new(0) }; 40];
pub fn tally(slot: usize, nodes: u64) { AUDIT[2 * slot].fetch_add(1, Ordering::Relaxed); AUDIT[2 * slot + 1].fetch_add(nodes, Ordering::Relaxed); }
#[pyfunction]
pub fn audit() -> Vec<u64> { AUDIT.iter().map(|a| a.swap(0, Ordering::Relaxed)).collect() }

/// The squares a refutation (or proof) used: the from- and to-squares of every move it recorded.
pub fn touched(h: &Hints) -> Bitboard {
    let mut b = Bitboard::EMPTY;
    for &c in h.values() {
        if c == 0x1F_FFFF { continue; }
        let from = c & 127; let to = (c >> 7) & 127;
        if from < 64 { b.add(Square::new(from)); }
        if to < 64 { b.add(Square::new(to)); }
    }
    b
}

/// A frame's stored proofs: one for the seed on the board (key None) and one for each candidate change (key Some).
pub type Cache = HashMap<Option<Atom>, std::sync::Arc<Hints>>;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Kind { AddOurs, AddTheirs, RemoveOurs, RemoveTheirs }

/// One condition over one square, relative to the frame's owner: our / their `role` stands on `sq` (Add), or no
/// longer stands there (Remove). A dormant frame WAITS on atoms (add ours, remove theirs, remove ours); an active
/// frame is INVALIDATED by atoms (Kaveh 2026-10-09: "each frame also needs to know what is invalidated, so that it can
/// re-fire if something happens") -- its BREAKERS: remove ours, add theirs, remove theirs (a flight square freed).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Atom { pub kind: Kind, pub role: Role, pub sq: Square }

impl Atom {
    pub fn holds(&self, pos: &Chess, owner: Color) -> bool {
        let p = pos.board().piece_at(self.sq);
        match self.kind {
            Kind::AddOurs => p == Some(Piece { color: owner, role: self.role }),
            Kind::AddTheirs => p == Some(Piece { color: !owner, role: self.role }),
            Kind::RemoveOurs => p != Some(Piece { color: owner, role: self.role }),
            Kind::RemoveTheirs => p != Some(Piece { color: !owner, role: self.role }),
        }
    }
    pub fn row(&self) -> (String, String, String) {
        let k = match self.kind { Kind::AddOurs => "add ours", Kind::AddTheirs => "add theirs", Kind::RemoveOurs => "remove ours", Kind::RemoveTheirs => "remove theirs" };
        (k.to_string(), self.role.upper_char().to_string(), self.sq.to_string())
    }
}

#[derive(Clone)]
pub struct Seed { pub mv: Move, pub uci: String, pub name: &'static str, pub focus: Vec<Square>, pub goal: i64 }

impl Seed {
    fn key(&self) -> String { format!("{}|{}", self.uci, self.focus.iter().map(|s| s.to_string()).collect::<Vec<_>>().join(",")) }
}

/// ONE CORNERING FRAME FOR EVERY PIECE (Kaveh 2026-10-09: "king frames should use the same cornering mechanism as every
/// other cornering mechanism -- so that should be checkmate, basically"): a piece is being cornered when our move
/// attacks it (for the king: gives check) and leaves it at most CORNER_ESC safe escapes, or leaves it (unattacked) with
/// none. The goal is the piece -- for the king, mate. Its conditions are read off its KEY SQUARES (the target, its safe
/// escapes, the attacker's square): what would cover an escape, what defends one, what could take the attacker. Off
/// (UNIFIED false): the older families -- check, single threat, cornered piece, trapped piece -- with conditions
/// enumerated over the whole zone.
pub static UNIFIED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);
#[pyfunction]
pub fn set_unified(flag: bool) -> PyResult<()> { UNIFIED.store(flag, Ordering::Relaxed); Ok(()) }
/// One change (one piece added or removed) cannot close more escapes than this: a target with more open escapes after
/// the seed gets no condition search -- far by count, not by search.
pub const CORNER_ESC: u32 = 3;

/// The safe escapes of the piece on `t` from `us`'s attacks: the squares it can MOVE to -- for a pawn its pushes and
/// captures, for a pinned piece only along the pin line (Kaveh 2026-10-09: "there could be a pin and also a cornering"
/// -- a pin is what takes escapes away) -- and a destination is unsafe if a cheaper piece of ours attacks it, or ours
/// attack it and theirs do not defend it (with the piece lifted off its square). A momentary check on their king is
/// NOT counted against other pieces (it would make every attacked piece look cornered). The king: its legal moves.
pub fn escapes(q: &Chess, us: Color, t: Square) -> Bitboard {
    let b = q.board();
    let Some(tp) = b.piece_at(t) else { return Bitboard::EMPTY };
    let them = tp.color;
    let occ = b.occupied();
    if tp.role == Role::King {
        let mover = if q.turn() == them { Some(q.clone()) } else { pass(q) };
        if let Some(m) = mover {
            return m.legal_moves().into_iter().filter(|mv| mv.from() == Some(t) && !matches!(mv, Move::Castle { .. })).map(|mv| mv.to()).collect();
        }
        return escape_set(q, us, t).0;
    }
    let mut dests = if tp.role == Role::Pawn {
        let caps = shakmaty::attacks::pawn_attacks(them, t) & b.by_color(us);
        let step: i32 = if them == Color::White { 8 } else { -8 };
        let mut pushes = Bitboard::EMPTY;
        let one = t as i32 + step;
        if (0..64).contains(&one) && !occ.contains(Square::new(one as u32)) {
            pushes.add(Square::new(one as u32));
            let start = if them == Color::White { shakmaty::Rank::Second } else { shakmaty::Rank::Seventh };
            let two = one + step;
            if t.rank() == start && (0..64).contains(&two) && !occ.contains(Square::new(two as u32)) { pushes.add(Square::new(two as u32)); }
        }
        caps | pushes
    } else {
        shakmaty::attacks::attacks(t, tp, occ) & !b.by_color(them)
    };
    if let Some(k) = b.king_of(them) {                                   // the pin: only along the line to the pinner
        for sl in b.attacks_to(k, us, occ.without(t)) {
            if shakmaty::attacks::between(sl, k).contains(t) {
                let mut line = shakmaty::attacks::between(sl, k); line.add(sl);
                dests &= line;
            }
        }
    }
    let occ_wo = occ.without(t);
    let tv = kernel_value(tp.role).max(1);
    dests.into_iter().filter(|e| {
        let ours = b.attacks_to(*e, us, occ_wo);
        let theirs = b.attacks_to(*e, them, occ_wo);
        let cheaper = ours.into_iter().any(|a| b.piece_at(a).map_or(false, |p| p.role != Role::King && kernel_value(p.role) < tv));
        !(cheaper || (ours.any() && !theirs.any()))
    }).collect()
}

/// The cornering seeds of move m (q = the board after it): see UNIFIED.
fn cornering(pos: &Chess, m: &Move, q: &Chess, now: &HashMap<u8, i64>, before: &HashMap<u8, i64>,
             push: &mut dyn FnMut(&'static str, Vec<Square>, i64)) {
    let us = pos.turn();
    for t in q.board().by_color(!us) {
        let Some(p) = q.board().piece_at(t) else { continue };
        let king = p.role == Role::King;
        let attacked = if king { q.is_check() } else { now.get(&(t as u8)).map_or(false, |g| *g > 0) };
        // a NEW attack: none before, or a bigger gain than before (2026-10-09 parity audit: pawns and grown gains were
        // dropped -- 38 of 119 positions' tactics kept)
        let newly = if king { attacked } else { attacked && before.get(&(t as u8)).map_or(true, |g0| now[&(t as u8)] > *g0) };
        let esc = escapes(q, us, t).count() as u32;
        let esc0 = if m.to() == t { 99 } else { escapes(pos, us, t).count() as u32 };
        let goal = if king { MATE_GOAL } else if attacked { now[&(t as u8)] } else { kernel_value(p.role) };
        let name = if king { "cornering king" } else { "cornering piece" };
        // every new attack opens the frame (2026-10-09: gating frames on <= 2 escapes lost 99 of 119 positions' tactics
        // on 10 games -- a pinned or overloaded piece, a check that forks); the escape count gates only the condition search
        // ...and a CHECK gives an attacked piece no time to escape this move: the fork with check (2026-10-09: 10
        // positions' tactics, 6 confirmed by Stockfish, were lost when the check stopped counting against escapes) --
        // two frames from one move, the king's and the piece's
        if attacked && (newly || esc < esc0 || (!king && q.is_check())) { push(name, vec![t], goal); }
        else if !attacked && esc == 0 && esc0 > 0 && !king && kernel_value(p.role) >= 3 { push(name, vec![t], goal); }
        else if !attacked && king && esc == 0 && esc0 > 0 { push(name, vec![t], goal); }   // a quiet move taking the last flight
    }
}

/// The frames the side to move could open: every check (focus: their king, goal: mate), every move making new threats
/// (fork: the second-best gain; single threat: its gain), every move newly cornering or trapping a piece. Rule-layer
/// matching only, no certificate.
pub fn seeds(pos: &Chess) -> Vec<Seed> {
    let us = pos.turn();
    let before = rules::threats_of(pos, us);
    let passed = pass(pos);
    let corner_before: std::collections::HashSet<u8> = passed.as_ref().map(|p| rules::cornered(p, us).into_iter().map(|(t, _)| t).collect()).unwrap_or_default();
    let trapped_before: std::collections::HashSet<u8> = passed.as_ref().map(|p| rules::trapped_unattacked(p, us).into_iter().map(|(t, _)| t).collect()).unwrap_or_default();
    let mut out: Vec<Seed> = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for m in pos.legal_moves() {
        let q = after(pos, &m);
        if q.is_game_over() { continue; }
        let uci = m.to_uci(CastlingMode::Standard).to_string();
        let king = q.board().king_of(!us);
        let mut push = |name: &'static str, focus: Vec<Square>, goal: i64| {
            let s = Seed { mv: m.clone(), uci: uci.clone(), name, focus, goal };
            if seen.insert(s.key()) { out.push(s); }
        };
        let now = rules::threats_of(&q, us);
        let mut new: Vec<(u8, i64)> = now.iter().filter(|(t, g)| before.get(t).map_or(true, |g0| **g > *g0)).map(|(t, g)| (*t, *g)).collect();
        new.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        let sqs: Vec<Square> = new.iter().map(|(t, _)| Square::new(*t as u32)).collect();
        if UNIFIED.load(Ordering::Relaxed) {
            if new.len() >= 2 { push("fork", sqs.clone(), new[1].1); }
            cornering(pos, &m, &q, &now, &before, &mut push);
            continue;
        }
        if q.is_check() { if let Some(k) = king { push("check", vec![k], MATE_GOAL); } }
        if new.len() >= 2 { push("fork", sqs.clone(), new[1].1); }
        else if new.len() == 1 && new[0].1 > 0 { push("single threat", sqs.clone(), new[0].1); }
        for (t, g) in rules::cornered(&q, us) {
            if Some(Square::new(t as u32)) != king && !corner_before.contains(&t) && g > 0 {
                push("cornered piece", vec![Square::new(t as u32)], g);
            }
        }
        if !q.is_check() {
            for (t, v) in rules::trapped_unattacked(&q, us) {
                if !trapped_before.contains(&t) && !corner_before.contains(&t) && v > 0 {
                    push("trapped piece", vec![Square::new(t as u32)], v);
                }
            }
        }
    }
    out
}

/// The certified value of the seed on `p` (its owner to move), or -1000: the 2-ply pre-check, then the frame's depth.
pub fn certify(p: &Chess, uci: &str, focus: &[Square], goal: i64) -> i64 {
    let Ok(u) = uci.parse::<shakmaty::uci::UciMove>() else { return -1000 };
    let Ok(m) = u.to_move(p) else { return -1000 };
    let a = certified_value(p, &m, focus, 2, 10_000).unwrap_or(-1000);
    if a >= goal { return a; }
    let b = certified_value(p, &m, focus, PURSUIT_PLIES, 50_000).unwrap_or(-1000);
    a.max(b)
}

fn perturb(pos: &Chess, a: &Atom, owner: Color) -> Option<Chess> {
    let mut s = pos.clone().into_setup(EnPassantMode::Legal);
    let here = s.board.piece_at(a.sq);
    match a.kind {
        Kind::AddOurs | Kind::AddTheirs => {
            if here.is_some() { return None; }
            let c = if a.kind == Kind::AddOurs { owner } else { !owner };
            s.board.set_piece_at(a.sq, Piece { color: c, role: a.role });
        }
        Kind::RemoveTheirs => { if here != Some(Piece { color: !owner, role: a.role }) { return None; } s.board.discard_piece_at(a.sq); }
        Kind::RemoveOurs => { if here != Some(Piece { color: owner, role: a.role }) { return None; } s.board.discard_piece_at(a.sq); }
    }
    Chess::from_setup(s, CastlingMode::Standard)
        .or_else(|e| e.ignore_invalid_castling_rights())
        .or_else(|e| e.ignore_too_much_material())
        .ok()
}

/// The frame's zone after the seed move: the focus, its reach, our pieces bearing on it and the lines between.
fn zone(pos: &Chess, s: &Seed) -> Bitboard {
    let q = after(pos, &s.mv);
    let mut z = frame_zone(&q, pos.turn(), &s.focus);
    for f in &s.focus { z.add(*f); }
    z
}

/// CAN A PIECE OF `c`'s `role` ARRIVE ON `sq`? (2026-10-09, measured on 40 census positions: 72 breakers per active
/// frame, mostly "their queen appears on some square", and conditions asking for a piece type we do not own -- dead
/// weight that also swelled every footprint until any move re-evaluated the frame.) A waiting condition (`one_move`
/// false) needs a piece of that role on the board -- a pawn on the square's file or an adjacent one, behind it; a
/// breaker (`one_move` true) is a change THEIR NEXT MOVE can make: a piece of that role that moves to `sq` (a pawn
/// pushing or capturing there).
fn can_arrive(pos: &Chess, c: Color, role: Role, sq: Square, one_move: bool) -> bool {
    let b = pos.board();
    let mine = b.by_color(c) & b.by_role(role);
    if mine.is_empty() { return false; }
    let fwd: i32 = if c == Color::White { 1 } else { -1 };
    if role == Role::Pawn {
        let (f, r) = (sq.file() as i32, sq.rank() as i32);
        return mine.into_iter().any(|p| {
            let (pf, pr) = (p.file() as i32, p.rank() as i32);
            let ahead = (r - pr) * fwd;
            if one_move {
                (pf == f && (ahead == 1 || (ahead == 2 && pr == if c == Color::White { 1 } else { 6 })))
                    || ((pf - f).abs() == 1 && ahead == 1)
            } else {
                (pf - f).abs() <= 1 && ahead >= 1
            }
        });
    }
    if !one_move { return true; }
    let occ = b.occupied();
    mine.into_iter().any(|p| shakmaty::attacks::attacks(p, Piece { color: c, role }, occ).contains(sq))
}

/// A frame's KEY SQUARES (after its seed move): each target, its safe escapes, and the square our moving piece lands
/// on -- where a cornering or a fork fails (an escape still open, a target defended, the attacker taken).
fn key_squares(pos: &Chess, s: &Seed) -> Bitboard {
    let q = after(pos, &s.mv);
    let mut k = Bitboard::EMPTY;
    for &t in &s.focus {
        k.add(t);
        let e = escapes(&q, pos.turn(), t);
        if e.count() as u32 <= CORNER_ESC { k |= e; }
    }
    k.add(s.mv.to());
    k
}

/// The candidate conditions read off the key squares (UNIFIED): our piece (one move away) added where it covers or
/// stands on a key square; their piece removed when it stands on or bears on one (a defender, a guard of an escape, an
/// attacker of our piece); our piece removed when it blocks one of our sliders' lines to one.
fn key_candidates(pos: &Chess, s: &Seed) -> Vec<Atom> {
    let us = pos.turn();
    let b = pos.board();
    let q = after(pos, &s.mv);
    let occ = q.board().occupied();
    let k = key_squares(pos, s);
    let sliders = q.board().by_color(us) & (q.board().by_role(Role::Bishop) | q.board().by_role(Role::Rook) | q.board().by_role(Role::Queen));
    let mut out = Vec::new();
    for i in 0..64u32 {
        let sq = Square::new(i);
        if sq == s.mv.to() || Some(sq) == s.mv.from() || s.focus.contains(&sq) { continue; }
        match b.piece_at(sq) {
            None => {
                for role in [Role::Queen, Role::Rook, Role::Bishop, Role::Knight, Role::Pawn] {
                    if !can_arrive(pos, us, role, sq, true) { continue; }
                    let reach = shakmaty::attacks::attacks(sq, Piece { color: us, role }, occ);
                    if k.contains(sq) || (reach & k).any() { out.push(Atom { kind: Kind::AddOurs, role, sq }); }
                }
            }
            Some(p) if p.role == Role::King => {}
            Some(p) if p.color != us => {
                let reach = shakmaty::attacks::attacks(sq, p, occ);
                if k.contains(sq) || (reach & k).any() { out.push(Atom { kind: Kind::RemoveTheirs, role: p.role, sq }); }
            }
            Some(p) => {
                let blocks = sliders.into_iter().any(|sl| k.into_iter().any(|t| shakmaty::attacks::between(sl, t).contains(sq)));
                if blocks { out.push(Atom { kind: Kind::RemoveOurs, role: p.role, sq }); }
            }
        }
    }
    out
}

/// The candidate atoms of a frame: pieces added on empty squares bearing on (or standing in) the zone -- ours when
/// looking for what is missing, theirs when looking for what breaks it -- their non-king pieces removed in or bearing
/// on it, our non-king pieces removed from it.
fn candidates(pos: &Chess, s: &Seed, z: Bitboard, breakers: bool) -> Vec<Atom> {
    let us = pos.turn();
    let b = pos.board();
    let occ = after(pos, &s.mv).board().occupied();
    let mut out = Vec::new();
    for i in 0..64u32 {
        let sq = Square::new(i);
        if sq == s.mv.to() || Some(sq) == s.mv.from() || s.focus.contains(&sq) { continue; }
        match b.piece_at(sq) {
            None => {
                let (c, kind) = if breakers { (!us, Kind::AddTheirs) } else { (us, Kind::AddOurs) };
                let near = breakers || ONE_MOVE_WAITS.load(Ordering::Relaxed);
                for role in [Role::Queen, Role::Rook, Role::Bishop, Role::Knight, Role::Pawn] {
                    if !can_arrive(pos, c, role, sq, near) { continue; }
                    let reach = shakmaty::attacks::attacks(sq, Piece { color: c, role }, occ);
                    if z.contains(sq) || (reach & z).any() { out.push(Atom { kind, role, sq }); }
                }
            }
            Some(p) if p.role == Role::King => {}
            Some(p) if p.color != us => {
                let reach = shakmaty::attacks::attacks(sq, p, occ);
                if z.contains(sq) || (reach & z).any() { out.push(Atom { kind: Kind::RemoveTheirs, role: p.role, sq }); }
            }
            Some(p) => { if z.contains(sq) { out.push(Atom { kind: Kind::RemoveOurs, role: p.role, sq }); } }
        }
    }
    out
}

type Found = (Vec<(Atom, i64)>, Vec<(Option<Atom>, Hints)>);

/// The single changes under which the seed's certificate reaches its goal: [(atom, the gain it earns)], best first,
/// with every candidate's recorded proof or refutation (for the next move's re-check).
/// THE CONTROL (bug 2026-10-09, the castling position: "add our pawn on e7" made four unrelated frames hold -- the pawn
/// promotes on d8 by itself): the seed must earn its goal BEYOND what the change alone wins by captures and promotions
/// (the resolved captures with us to move, before the seed): the proof asks for goal + that, and a mate goal is refused
/// when the change alone wins a promotion or a queen (review 2026-10-09: at 9 "add our pawn on e7" with e8 empty
/// slipped through and e8=Q# "proved" the mate).
pub fn conditions(pos: &Chess, s: &Seed, cache: &Cache, base: &Hints, deep: bool) -> (Found, u64, u32) {
    let us = pos.turn();
    let z = zone(pos, s);
    let used = touched(base);
    let occ = pos.board().occupied();
    let on_ref = |a: &Atom| -> bool {
        if used.contains(a.sq) { return true; }
        let piece = match a.kind { Kind::AddOurs | Kind::RemoveOurs => Piece { color: us, role: a.role }, _ => Piece { color: !us, role: a.role } };
        (shakmaty::attacks::attacks(a.sq, piece, occ) & used).any()
    };
    let unified = UNIFIED.load(Ordering::Relaxed);
    if unified && s.name.starts_with("cornering") {
        let q = after(pos, &s.mv);
        if s.focus.iter().any(|t| escapes(&q, us, *t).count() as u32 > CORNER_ESC) {
            tally(A_COND_CONTROL, 0);                                  // too many escapes for one change: far by count
            return ((vec![], vec![]), 0, 0);
        }
    }
    let cands = if unified { key_candidates(pos, s) } else { candidates(pos, s, z, false) };
    let rows: Vec<(Atom, bool, i64, Hints, u64, u8)> = cands.par_iter().filter_map(|a| {
        let p2 = perturb(pos, a, us)?;
        BUDGET.store(50_000, Ordering::Relaxed);
        let alone = settle_d(&p2, -2000, 2000, true, SETTLE_DEPTH).max(0);
        let spent = BUDGET.load(Ordering::Relaxed) == 0;
        BUDGET.store(u64::MAX, Ordering::Relaxed);
        if spent { tally(A_COND_CONTROL, 0); return None; }            // unknown what the change wins alone: refuse it
        let mate = s.goal >= MATE_GOAL;
        if mate && alone >= 8 { tally(A_COND_CONTROL, 0); return None; } // a promotion (8) or a queen: the change, not the seed
        let goal = if mate { s.goal } else { s.goal + alone };
        let (ok, rec, n, spent, tier) = prove_t(&p2, &s.uci, &s.focus, goal, cache.get(&Some(*a)).cloned(), deep);
        let on = on_ref(a);
        tally(if ok { A_COND_FOUND } else if spent { A_COND_SPENT } else { A_COND_REFUTED }, n);
        tally(match (ok, on) { (true, true) => A_COND_FOUND_ON, (true, false) => A_COND_FOUND_OFF, (false, true) => A_COND_MISS_ON, _ => A_COND_MISS_OFF }, n);
        Some((*a, ok, if mate { 1000 } else { goal - alone }, rec, n, tier))
    }).collect();
    let nodes = rows.iter().map(|r| r.4).sum();
    let shallow = rows.iter().filter(|r| r.5 == 1).count() as u32;
    let mut found: Vec<(Atom, i64)> = rows.iter().filter(|r| r.1).map(|r| (r.0, r.2)).collect();
    found.sort_by(|a, b| b.1.cmp(&a.1).then((a.0.sq as u8).cmp(&(b.0.sq as u8))));
    ((found, rows.into_iter().map(|r| (Some(r.0), r.3)).collect()), nodes, shallow)
}

/// What invalidates an active frame: the single changes (their next move can make) under which its certificate no
/// longer reaches the goal -- [(atom, goal - 1: "below the goal")], with every candidate's recorded proof.
pub fn breakers(pos: &Chess, s: &Seed, cache: &Cache) -> (Found, u64) {
    let us = pos.turn();
    let z = zone(pos, s);
    let rows: Vec<(Atom, bool, Hints, u64)> = candidates(pos, s, z, true).par_iter().filter_map(|a| {
        let p2 = perturb(pos, a, us)?;
        let (ok, rec, n, spent) = prove_n(&p2, &s.uci, &s.focus, s.goal, cache.get(&Some(*a)).cloned());
        tally(if spent { A_BRK_SPENT } else if ok { A_BRK_HOLDS } else { A_BRK_BREAKS }, n);
        Some((*a, ok, rec, n))
    }).collect();
    let nodes = rows.iter().map(|r| r.3).sum();
    let mut found: Vec<(Atom, i64)> = rows.iter().filter(|r| !r.1).map(|r| (r.0, s.goal - 1)).collect();
    found.sort_by_key(|a| a.0.sq as u8);
    ((found, rows.into_iter().map(|r| (Some(r.0), r.2)).collect()), nodes)
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Status { Active, Dormant, Far, Unsearched }

impl Status {
    fn name(&self) -> &'static str { match self { Status::Active => "active", Status::Dormant => "dormant", Status::Far => "far", Status::Unsearched => "unsearched" } }
}

#[derive(Clone)]
pub struct Frame {
    pub owner: Color, pub seed: Seed, pub value: i64, pub status: Status,
    pub alts: Vec<(Atom, i64)>,      // dormant: what it waits on (any one of them makes it hold)
    pub breaks: Vec<(Atom, i64)>,    // active: what invalidates it (any one of them makes it fail)
    pub footprint: Bitboard, pub born: u32,
    pub cache: Cache,                // the stored proofs, re-checked on the next move
    pub stale: bool,                 // kept from the last ply unevaluated (its owner had no null move: the other side in check)
    pub nodes: u64,                  // what evaluating it cost this move
    pub last_searched: Option<(Status, usize)>,   // its status and conditions when last searched (carried through skips)
}

// THE GATES (Kaveh 2026-10-09: "a cheap gate followed by a more thorough gate"; the audit: 96% of the nodes went to
// condition searches that found nothing). Gate 1, before any condition search: a linear score over features that cost
// nothing (GATE_FEATURES) -- below `threshold` the frame is left UNSEARCHED (the seed proof still ran: a tactic is never
// missed, only a waiting condition). Gate 2: frames scoring below `deep_threshold` are searched at 2 plies only. The
// weights are trained offline (frame_gate.py) on the logged frames and set with `set_gate`.
pub const GATE_FEATURES: [&str; 18] = ["family cornering king", "family cornering piece", "family fork", "goal",
    "target value", "escapes", "targets", "candidate changes", "seed is a check", "seed is a capture",
    "seed proof nodes (log2)", "dormant last move", "far last move", "new this move", "conditions last move",
    "on the real board (not the null move)", "material for the owner", "pieces on the board"];
#[derive(Clone)]
pub struct Gate { pub w: Vec<f32>, pub b: f32, pub threshold: f32, pub deep_threshold: f32 }
pub static GATE: std::sync::RwLock<Option<Gate>> = std::sync::RwLock::new(None);
#[pyfunction]
pub fn set_gate(w: Vec<f32>, b: f32, threshold: f32, deep_threshold: f32) -> PyResult<()> {
    *GATE.write().unwrap() = Some(Gate { w, b, threshold, deep_threshold }); Ok(())
}
#[pyfunction]
pub fn clear_gate() -> PyResult<()> { *GATE.write().unwrap() = None; Ok(()) }
#[pyfunction]
pub fn gate_feature_names() -> Vec<String> { GATE_FEATURES.iter().map(|s| s.to_string()).collect() }

/// The logged frames (when LOG_FRAMES is on): (features, conditions found, found within 2 plies, condition-search nodes)
pub static LOG_FRAMES: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
pub static FRAME_LOG: Mutex<Vec<(Vec<f32>, u32, u32, u64)>> = Mutex::new(Vec::new());
#[pyfunction]
pub fn log_frames(flag: bool) -> PyResult<()> { LOG_FRAMES.store(flag, Ordering::Relaxed); Ok(()) }
#[pyfunction]
pub fn frame_log() -> Vec<(Vec<f32>, u32, u32, u64)> { std::mem::take(&mut *FRAME_LOG.lock().unwrap()) }

fn gate_features(p: &Chess, s: &Seed, prev: Option<(Status, usize)>, real: bool, n_seed: u64) -> Vec<f32> {
    let us = p.turn();
    let q = after(p, &s.mv);
    let king = s.name == "cornering king";
    let val = |t: &Square| p.board().piece_at(*t).map_or(0, |pc| if pc.role == Role::King { 10 } else { kernel_value(pc.role) });
    let esc = s.focus.iter().map(|t| escapes(&q, us, *t).count()).min().unwrap_or(0).min(8);
    let ncand = if UNIFIED.load(Ordering::Relaxed) { key_candidates(p, s).len() } else { 0 };
    vec![king as u8 as f32, (s.name == "cornering piece") as u8 as f32, (s.name == "fork") as u8 as f32,
         s.goal.min(10) as f32, s.focus.iter().map(val).max().unwrap_or(0) as f32, esc as f32, s.focus.len() as f32,
         ncand as f32, q.is_check() as u8 as f32, s.mv.is_capture() as u8 as f32, ((n_seed + 1) as f32).log2(),
         matches!(prev, Some((Status::Dormant, _))) as u8 as f32,
         matches!(prev, Some((Status::Far, _)) | Some((Status::Unsearched, _))) as u8 as f32,
         prev.is_none() as u8 as f32, prev.map_or(0, |x| x.1) as f32, real as u8 as f32,
         kernel_material(p, us) as f32, p.board().occupied().count() as f32]
}

/// Evaluate one seed on `p` (its owner to move): active with its breakers if it certifies, else dormant with its
/// missing conditions (or far). With `cache` (the frame's stored proofs from the last move) every proof question is
/// re-checked with its stored proof tried first -- the same answers as a fresh search, at the cost of checking. The
/// value is a lower bound: the goal when active, -1000 otherwise. The footprint holds the zone, the seed's squares and
/// every atom's square.
pub fn evaluate(p: &Chess, s: &Seed, born: u32, cache: &Cache, prev: Option<(Status, usize)>, real: bool) -> Frame {
    let (holds, rec, n0, _) = prove_n(p, &s.uci, &s.focus, s.goal, cache.get(&None).cloned());
    tally(if holds { A_SEED_HOLD } else { A_SEED_FAIL }, n0);
    let mut fp = zone(p, s);
    for f in &s.focus { fp.add(*f); }
    if let Some(f) = s.mv.from() { fp.add(f); }
    fp.add(s.mv.to());
    let (status, alts, breaks, recs, n1) = if holds {
        let ((b, r), n) = breakers(p, s, cache);
        (Status::Active, vec![], b, r, n)
    } else {
        let gate = GATE.read().unwrap().clone();
        let logging = LOG_FRAMES.load(Ordering::Relaxed);
        let feats = if gate.is_some() || logging { gate_features(p, s, prev, real, n0) } else { vec![] };
        let score = gate.as_ref().map(|g| g.b + g.w.iter().zip(&feats).map(|(w, x)| w * x).sum::<f32>());
        if let (Some(g), Some(sc)) = (&gate, score) {
            if sc < g.threshold {
                tally(A_GATED, 0);
                // (the skip carries the last SEARCHED status forward -- 2026-10-09: a skipped frame read as "far last move"
                // was skipped again, and the held-out games kept 84% of the conditions against the 90% chosen offline)
                return Frame { owner: p.turn(), seed: s.clone(), value: -1000, status: Status::Unsearched, alts: vec![],
                               breaks: vec![], footprint: fp, born, cache: { let mut c = Cache::new(); c.insert(None, std::sync::Arc::new(rec)); c },
                               stale: false, nodes: n0, last_searched: prev };
            }
        }
        let deep = match (&gate, score) { (Some(g), Some(sc)) => sc >= g.deep_threshold, _ => DEEP_CONDITIONS.load(Ordering::Relaxed) };
        if !deep { tally(A_SHALLOW, 0); }
        let ((a, r), n, shallow) = conditions(p, s, cache, &rec, deep);
        if logging {
            FRAME_LOG.lock().unwrap().push((feats, a.len() as u32, shallow, n));
        }
        (if a.is_empty() { Status::Far } else { Status::Dormant }, a, vec![], r, n)
    };
    tally(match status { Status::Active => A_ACTIVE, Status::Dormant => A_DORMANT, _ => A_FAR }, n0 + n1);
    for (a, _) in alts.iter().chain(breaks.iter()) { fp.add(a.sq); }
    let mut new_cache: Cache = recs.into_iter().map(|(k, h)| (k, std::sync::Arc::new(h))).collect();
    new_cache.insert(None, std::sync::Arc::new(rec));
    let nalts = alts.len();
    Frame { owner: p.turn(), seed: s.clone(), value: if holds { s.goal } else { -1000 }, status, alts, breaks,
            footprint: fp, born, cache: new_cache, stale: false, nodes: n0 + n1, last_searched: Some((status, nalts)) }
}

/// Both sides' frames: the side to move's on the position, the other side's on the null move (none when in check).
pub fn frames_both(pos: &Chess, born: u32) -> Vec<Frame> {
    let empty = Cache::new();
    let mut out: Vec<Frame> = seeds(pos).par_iter().map(|s| evaluate(pos, s, born, &empty, None, true)).collect();
    if let Some(p) = pass(pos) {
        out.extend(seeds(&p).par_iter().map(|s| evaluate(&p, s, born, &empty, None, false)).collect::<Vec<_>>());
    }
    out
}

type FrameRow = (String, String, String, Vec<String>, i64, i64, String, Vec<(String, String, String, i64)>, Vec<(String, String, String, i64)>, u32, bool);

fn row(f: &Frame) -> FrameRow {
    (if f.owner == Color::White { "w" } else { "b" }.to_string(), f.seed.uci.clone(), f.seed.name.to_string(),
     f.seed.focus.iter().map(|s| s.to_string()).collect(), f.seed.goal, f.value, f.status.name().to_string(),
     f.alts.iter().map(|(a, v)| { let (k, r, s) = a.row(); (k, r, s, *v) }).collect(),
     f.breaks.iter().map(|(a, v)| { let (k, r, s) = a.row(); (k, r, s, *v) }).collect(), f.born, f.stale)
}

/// waiting_frames(fen) -> [(owner, seed uci, name, focus, goal, certified value, status, waits on [(kind, role,
/// square, value)], broken by [(kind, role, square, value left)], born, stale)] for both sides.
#[pyfunction]
pub fn waiting_frames(fen: String) -> PyResult<Vec<FrameRow>> {
    let Some(pos) = parse(&fen) else { return Ok(vec![]) };
    Ok(frames_both(&pos, 0).iter().map(row).collect())
}

type ArmRow = (String, String, String, Vec<String>, i64, Vec<(String, String, String)>);

/// The reverse lookup on `pos` (side to move = us): for each legal move M, the dormant frames of ours that M's
/// changes satisfy AND that then certify on the null move (M arms a threat), and the dormant frames of theirs that M
/// satisfies and that then certify with them to move (M hands them a tactic). Checking moves are not armed here (no
/// null move after a check). And DEFUSES: the active frames of theirs (threats on the null move) that M breaks -- one
/// of its breakers holds after M and the certificate, with them to move, falls below the goal.
pub fn arming_core(pos: &Chess, frames: &[Frame]) -> (Vec<ArmRow>, Vec<ArmRow>, Vec<ArmRow>) {
    let us = pos.turn();
    let moves: Vec<Move> = pos.legal_moves().into_iter().collect();
    let res: Vec<(Vec<ArmRow>, Vec<ArmRow>, Vec<ArmRow>)> = moves.par_iter().map(|mm| {
        let q = after(pos, mm);
        let muci = mm.to_uci(CastlingMode::Standard).to_string();
        let mut arms = Vec::new();
        let mut hands = Vec::new();
        let mut defuses = Vec::new();
        for f in frames.iter().filter(|f| f.status == Status::Active && f.owner != us) {
            let hit: Vec<(String, String, String)> = f.breaks.iter()
                .filter(|(a, _)| a.holds(&q, f.owner) && !a.holds(pos, f.owner)).map(|(a, _)| a.row()).collect();
            if hit.is_empty() { continue; }
            let v = certify(&q, &f.seed.uci, &f.seed.focus, f.seed.goal);
            if v < f.seed.goal { defuses.push((muci.clone(), f.seed.uci.clone(), f.seed.name.to_string(), f.seed.focus.iter().map(|s| s.to_string()).collect(), v, hit)); }
        }
        for f in frames.iter().filter(|f| f.status == Status::Dormant) {
            let met: Vec<(String, String, String)> = f.alts.iter()
                .filter(|(a, _)| a.holds(&q, f.owner) && !a.holds(pos, f.owner)).map(|(a, _)| a.row()).collect();
            if met.is_empty() { continue; }
            if f.owner == us {
                if q.is_check() { continue; }
                let Some(p) = pass(&q) else { continue };
                let v = certify(&p, &f.seed.uci, &f.seed.focus, f.seed.goal);
                if v >= f.seed.goal { arms.push((muci.clone(), f.seed.uci.clone(), f.seed.name.to_string(), f.seed.focus.iter().map(|s| s.to_string()).collect(), v, met)); }
            } else {
                let v = certify(&q, &f.seed.uci, &f.seed.focus, f.seed.goal);
                if v >= f.seed.goal { hands.push((muci.clone(), f.seed.uci.clone(), f.seed.name.to_string(), f.seed.focus.iter().map(|s| s.to_string()).collect(), v, met)); }
            }
        }
        (arms, hands, defuses)
    }).collect();
    let mut a = Vec::new(); let mut h = Vec::new(); let mut d = Vec::new();
    for (x, y, z) in res { a.extend(x); h.extend(y); d.extend(z); }
    (a, h, d)
}

/// arming_moves(fen) -> (arms, hands, defuses): [(our move, seed uci, name, focus, certified value, [atoms met])].
#[pyfunction]
pub fn arming_moves(fen: String) -> PyResult<(Vec<ArmRow>, Vec<ArmRow>, Vec<ArmRow>)> {
    let Some(pos) = parse(&fen) else { return Ok((vec![], vec![], vec![])) };
    let frames = frames_both(&pos, 0);
    Ok(arming_core(&pos, &frames))
}

/// THE LIVE POOL: frames kept across the moves of a game, both sides. `observe(fen)`: the seeds the move CREATED are
/// evaluated from scratch; the seeds it DESTROYED are dropped; every other frame is RE-CHECKED from its stored proofs
/// (`proofs`, default; off = every frame from scratch, for the audit). (Reuse by footprint without re-checking was
/// audited at 1.0% stale statuses for 14% saved and removed.) Returns the events: (owner, seed uci, name, focus, goal, value, status
/// before ("new" if unseen), status after ("dead" if dropped), born ply).
#[pyclass(unsendable)]
pub struct FramePool {
    frames: Vec<Frame>,
    last: Option<Chess>,
    ply: u32,
    proofs: bool,
    #[pyo3(get)] rechecked: u64,
    #[pyo3(get)] created: u64,
    #[pyo3(get)] nodes: u64,
}

type Event = (String, String, String, Vec<String>, i64, i64, String, String, u32);

fn event(f: &Frame, before: &str, after_: &str) -> Event {
    (if f.owner == Color::White { "w" } else { "b" }.to_string(), f.seed.uci.clone(), f.seed.name.to_string(),
     f.seed.focus.iter().map(|s| s.to_string()).collect(), f.seed.goal, f.value, before.to_string(), after_.to_string(), f.born)
}

#[pymethods]
impl FramePool {
    #[new]
    #[pyo3(signature = (proofs=true))]
    fn new(proofs: bool) -> Self { FramePool { frames: vec![], last: None, ply: 0, proofs, rechecked: 0, created: 0, nodes: 0 } }

    fn observe(&mut self, fen: String) -> PyResult<Vec<Event>> {
        let Some(pos) = parse(&fen) else { return Ok(vec![]) };
        let n0 = NODES.load(Ordering::Relaxed);
        let ply = self.ply;
        let mut events = Vec::new();
        let mut next: Vec<Frame> = Vec::new();
        let sides: Vec<Chess> = std::iter::once(pos.clone()).chain(pass(&pos)).collect();
        let mut alive_owners = Vec::new();
        for (si, p) in sides.iter().enumerate() {
            let owner = p.turn();
            let real = si == 0;
            alive_owners.push(owner);
            let ss = seeds(p);
            let old: HashMap<String, &Frame> = self.frames.iter().filter(|f| f.owner == owner).map(|f| (f.seed.key(), f)).collect();
            let proofs = self.proofs;
            let empty = Cache::new();
            let done: Vec<(Frame, Option<Status>, bool)> = ss.par_iter().map(|s| {
                match old.get(&s.key()) {
                    Some(f) => {
                        let prev = if f.status == Status::Unsearched { f.last_searched } else { Some((f.status, f.alts.len())) };
                        (evaluate(p, s, f.born, if proofs { &f.cache } else { &empty }, prev, real), Some(f.status), true)
                    }
                    None => (evaluate(p, s, ply, &empty, None, real), None, false),
                }
            }).collect();
            let keys: std::collections::HashSet<String> = ss.iter().map(|s| s.key()).collect();
            for (f, before, reused) in done {
                if reused { self.rechecked += 1 } else { self.created += 1 }
                if before == Some(Status::Far) && f.status == Status::Far { tally(A_FAR_AGAIN, f.nodes); }
                match before {
                    None => events.push(event(&f, "new", f.status.name())),
                    Some(b) if b != f.status => events.push(event(&f, b.name(), f.status.name())),
                    _ => {}
                }
                next.push(f);
            }
            for f in self.frames.iter().filter(|f| f.owner == owner && !keys.contains(&f.seed.key())) {
                events.push(event(f, f.status.name(), "dead"));
            }
        }
        // a side that could not be evaluated (the side to move is in check: no null move for the other) keeps its frames
        for f in self.frames.iter().filter(|f| !alive_owners.contains(&f.owner)) { let mut g = f.clone(); g.stale = true; next.push(g); }
        self.frames = next;
        self.nodes += NODES.load(Ordering::Relaxed).wrapping_sub(n0);
        self.last = Some(pos);
        self.ply += 1;
        Ok(events)
    }

    fn frames(&self) -> Vec<FrameRow> { self.frames.iter().map(row).collect() }

    /// The reverse lookup on the last observed position, from the pool's frames.
    fn arming(&self) -> (Vec<ArmRow>, Vec<ArmRow>, Vec<ArmRow>) {
        match &self.last { Some(p) => arming_core(p, &self.frames), None => (vec![], vec![], vec![]) }
    }
}


/// THE TRACE OF ONE ANALYSIS (Kaveh 2026-10-09: "walk me through one analysis"): for every seed of both sides (the
/// side to move on the board, the other on the null move) -- its seed proof (holds?, nodes, ms), the gate's score and
/// decision, the candidate changes tried, what was found (conditions or breakers), the nodes and ms of that search.
#[pyfunction]
pub fn frame_trace(fen: String) -> PyResult<Vec<((String, String, String, Vec<String>, i64), (bool, u64, f64, Option<f32>, String), (usize, Vec<String>, u64, f64))>> {
    let Some(pos) = parse(&fen) else { return Ok(vec![]) };
    let gate = GATE.read().unwrap().clone();
    let mut out = Vec::new();
    let sides: Vec<(Chess, bool)> = std::iter::once((pos.clone(), true)).chain(pass(&pos).map(|p| (p, false))).collect();
    for (p, real) in sides {
        for s in seeds(&p) {
            let t0 = std::time::Instant::now();
            let (holds, rec, n0, _) = prove_n(&p, &s.uci, &s.focus, s.goal, None);
            let ms0 = t0.elapsed().as_secs_f64() * 1000.0;
            let feats = gate_features(&p, &s, None, real, n0);
            let score = gate.as_ref().map(|g| g.b + g.w.iter().zip(&feats).map(|(w, x)| w * x).sum::<f32>());
            let t1 = std::time::Instant::now();
            let (decision, ncand, found, n1) = if holds {
                let ((b, r), n) = breakers(&p, &s, &Cache::new());
                ("active: breakers searched".to_string(), r.len(), b.iter().map(|(a, _)| { let (k, ro, sq) = a.row(); format!("{k} {ro}@{sq}") }).collect::<Vec<_>>(), n)
            } else if matches!((&gate, score), (Some(g), Some(sc)) if sc < g.threshold) {
                ("skipped by gate 1".to_string(), 0, vec![], 0)
            } else {
                let deep = match (&gate, score) { (Some(g), Some(sc)) => sc >= g.deep_threshold, _ => true };
                let ((a, r), n, _) = conditions(&p, &s, &Cache::new(), &rec, deep);
                (format!("conditions searched ({})", if deep { "2 plies, then 4" } else { "2 plies only: gate 2" }), r.len(),
                 a.iter().map(|(a, _)| { let (k, ro, sq) = a.row(); format!("{k} {ro}@{sq}") }).collect::<Vec<_>>(), n)
            };
            let ms1 = t1.elapsed().as_secs_f64() * 1000.0;
            out.push(((if p.turn() == Color::White { "w" } else { "b" }.to_string(), s.uci.clone(), s.name.to_string(),
                       s.focus.iter().map(|x| x.to_string()).collect::<Vec<String>>(), s.goal),
                      (holds, n0, ms0, score, decision), (ncand, found, n1, ms1)));
        }
    }
    Ok(out)
}
