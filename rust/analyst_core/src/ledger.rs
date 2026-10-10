//! THE PIECE LEDGER (Kaveh 2026-10-10: "if you think about the jobs of the pieces and we keep track of the jobs of the
//! pieces, then we can see how can a piece do more than one thing ... usually we want to consider changes ... I'll only
//! recalculate it if something changed regarding my bishop"; "the pin is suddenly removed ... the knight becomes
//! available to hop ... so the queen that was doing something is now not doing anything ... that's guiding the starting
//! point, the nucleus of search").
//!
//! The ledger of a position: every piece's JOBS (threat.rs: the threats that appear when it is lifted) and the PINS on
//! the board (a slider, the piece it pins, what that piece shields). Kept across a game's moves (plan.rs PlanState); the
//! DIFF against the previous position is a list of EVENTS -- a piece that lost a job (freed), a piece that gained one,
//! a pin released (the pinning piece and the pinned one both freed). The planner starts from them: our freed pieces are
//! asked "what can you do now?" first (moves making two threats at once, a mate threat among them), their freed pieces
//! "what can you do?" (named in the trace -- "it can do nothing" is Kaveh's step that frees our queen).
//! (Measured 2026-10-10, job_ledger_measure.py: 94.8% of jobs survive a move -- the events are the few that do not.)
use super::*;
use crate::threat::{jobs_with, new_threats, threats, Threat};

#[derive(Clone, Default)]
pub struct Ledger {
    pub jobs: HashMap<Square, (Piece, Vec<Threat>)>,
    pub pins: Vec<(Square, Square, Square)>,               // (slider, pinned piece, what it shields)
}

#[derive(Clone, Debug)]
pub struct Event {
    pub sq: Square,
    pub piece: Piece,
    pub freed: bool,                                        // lost a job or a pin (true) / gained a job (false)
    pub text: String,
}

fn job_text(j: &Threat) -> String {
    if j.mate { format!("guarding against mate on {}", j.target) } else { format!("guarding {} (worth {})", j.target, j.value) }
}

fn name(p: Piece) -> String { format!("{:?} {:?}", p.color, p.role).to_lowercase() }

/// The ledger of a position.
pub fn ledger(pos: &Chess) -> Ledger {
    let b = pos.board();
    let now_w = threats(pos, Color::White);
    let now_b = threats(pos, Color::Black);
    let mut jobs = HashMap::new();
    for sq in b.occupied() {
        let Some(pc) = b.piece_at(sq) else { continue };
        if pc.role == Role::King { continue; }
        let atk_now = if pc.color == Color::White { &now_b } else { &now_w };
        jobs.insert(sq, (pc, jobs_with(pos, sq, atk_now)));
    }
    let mut pins = Vec::new();
    for c in [Color::White, Color::Black] {
        let zone = b.by_color(!c) & (b.by_role(Role::King) | b.by_role(Role::Queen) | b.by_role(Role::Rook));
        for p in crate::paths::attack_paths(pos, c, zone) {
            if p.kind == "pin" { if let Some(bl) = p.via { pins.push((p.piece, bl, p.target)); } }
        }
    }
    Ledger { jobs, pins }
}

/// The events from `prev` to `now` (pieces compared where they stand; a piece that moved is new).
pub fn diff(prev: &Ledger, now: &Ledger) -> Vec<Event> {
    let mut ev = Vec::new();
    let same = |a: &Threat, b: &Threat| a.target == b.target && a.mate == b.mate;
    for (sq, (pc, j)) in &now.jobs {
        let Some((pp, j0)) = prev.jobs.get(sq) else { continue };
        if pp != pc { continue; }
        let lost: Vec<&Threat> = j0.iter().filter(|a| !j.iter().any(|b| same(a, b))).collect();
        let gained: Vec<&Threat> = j.iter().filter(|a| !j0.iter().any(|b| same(a, b))).collect();
        if !lost.is_empty() {
            ev.push(Event { sq: *sq, piece: *pc, freed: true, text: format!("the {} on {sq} is no longer {}", name(*pc),
                lost.iter().map(|t| job_text(t)).collect::<Vec<_>>().join(", ")) });
        }
        if !gained.is_empty() {
            ev.push(Event { sq: *sq, piece: *pc, freed: false, text: format!("the {} on {sq} is now {}", name(*pc),
                gained.iter().map(|t| job_text(t)).collect::<Vec<_>>().join(", ")) });
        }
    }
    for (sl, bl, tg) in &prev.pins {
        if now.pins.iter().any(|p| p == &(*sl, *bl, *tg)) { continue; }
        // released: both pieces still where they were -- the pinning piece's hold is gone, the pinned piece is free
        let (Some((ps, _)), Some((pb, _))) = (now.jobs.get(sl).or_else(|| None), now.jobs.get(bl)) else { continue };
        if prev.jobs.get(sl).map(|x| x.0) != Some(*ps) || prev.jobs.get(bl).map(|x| x.0) != Some(*pb) { continue; }
        ev.push(Event { sq: *sl, piece: *ps, freed: true, text: format!("the {} on {sl} no longer pins the {} on {bl} (it shielded {tg})", name(*ps), name(*pb)) });
        ev.push(Event { sq: *bl, piece: *pb, freed: true, text: format!("the {} on {bl} is unpinned", name(*pb)) });
    }
    ev.sort_by_key(|e| (e.sq as u32, !e.freed));
    ev
}

/// WHAT CAN THIS PIECE DO NOW? Its quiet moves (no capture, no check -- those are proved elsewhere) after which its side
/// has two or more new threats (material, promotion, trapped piece, mate by checks): -> [(move, the lesser threat's
/// target, its value, whether the bigger one is a mate)] best first. For the side to move.
pub fn double_threats_of(pos: &Chess, sq: Square) -> Vec<(Move, Square, i64, bool)> {
    let us = pos.turn();
    let before = threats(pos, us);
    let mut out = Vec::new();
    for m in pos.legal_moves() {
        if m.from() != Some(sq) || m.is_capture() || m.is_promotion() { continue; }
        let q = after(pos, &m);
        if q.is_check() { continue; }
        // (the moving piece must not simply be lost: their gain on it below the lesser threat)
        let risk = rules::threats_of(&q, !us).get(&(m.to() as u8)).copied().unwrap_or(0).max(0);
        let now = threats(&q, us);
        let mut new: Vec<Threat> = new_threats(&now, &before).into_iter().cloned().collect();
        if new.len() < 2 { continue; }
        new.sort_by(|a, b| b.value.cmp(&a.value));
        let lesser = new[1].clone();
        let v = lesser.value - risk;
        if v > 0 { out.push((m, lesser.target, v, new[0].mate)); }
    }
    out.sort_by(|a, b| b.2.cmp(&a.2));
    out
}

/// WHAT CAN THEIR PIECE DO? (the side NOT to move) -- its moves after which its side gains a new threat, if we passed:
/// -> [(move uci, the threat's target, value)].
pub fn threats_by_piece(pos: &Chess, sq: Square) -> Vec<(String, Square, i64)> {
    let Some(p) = pass(pos) else { return vec![] };
    let them = p.turn();
    let before = threats(&p, them);
    let mut out = Vec::new();
    for m in p.legal_moves() {
        if m.from() != Some(sq) { continue; }
        let q = after(&p, &m);
        for t in new_threats(&threats(&q, them), &before) {
            out.push((m.to_uci(CastlingMode::Standard).to_string(), t.target, t.value));
        }
    }
    out
}
