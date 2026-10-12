//! THE GAME CONTEXT (Kaveh 2026-10-12: "in every frame that we analyze, we need to carry into that frame the move counter
//! -- the 50-move clock -- and the triple position ... without that information there's no way the frame could know
//! that we can draw ... it is forced to be global ... a class that represents this instance of this game ... when we do
//! puzzles, we don't have it").
//!
//! The fifty-move clock rides on every position already (shakmaty keeps the half-move counter from the FEN through every
//! move). The repetitions do not: the game's positions since the last irreversible move (a capture or a pawn move -- no
//! position before it can come back) are counted by the game's object (`plan::PlanState`) and installed here, read-only,
//! for one planning call; a stateless call (a puzzle, `plan_trace` on a FEN) sees an empty history.
//!
//! What the rules of the game draw (the Lean library states them: `Chess.History` -- repetitionKey, threefoldClaimableB,
//! fiftyMoveClaimable; and the proved `Ideas.InsufficientMaterial.insufficient_material`):
//!   CLAIMABLE   a position reached in analysis for the third time (the history's count plus this one), or with 100
//!               plies since the last capture or pawn move -- either side may claim it, so no claim to win survives it
//!   INSUFFICIENT two kings and at most one knight or bishop: no checkmate exists (proved, static)
//!   STALEMATE   no legal move and not in check
use super::*;
use std::cell::RefCell;
use std::collections::HashMap;

thread_local! {
    /// the repetition keys (zobrist: board, side, castling, a LEGAL en passant -- the library's repetitionKey) of the game's
    /// positions since the last irreversible move -> how often each occurred, the position being planned included
    static SEEN: RefCell<HashMap<u64, u8>> = RefCell::new(HashMap::new());
}

pub fn install(seen: HashMap<u64, u8>) { SEEN.with(|s| *s.borrow_mut() = seen); }
pub fn clear() { SEEN.with(|s| s.borrow_mut().clear()); }

/// How often the game has already seen this position (0 in a puzzle).
pub fn seen(q: &Chess) -> u8 {
    SEEN.with(|s| { let s = s.borrow(); if s.is_empty() { 0 } else { s.get(&crate::threat::zkey(q)).copied().unwrap_or(0) } })
}

/// A position reached in ANALYSIS (not the root) that either side may claim as a draw: its third occurrence, or the
/// fifty-move clock run out (a checkmate on that move comes first).
pub fn claimable_draw(q: &Chess) -> Option<&'static str> {
    if q.halfmoves() >= 100 && !q.is_checkmate() { return Some("the fifty-move rule"); }
    if seen(q) >= 2 { return Some("threefold repetition"); }
    None
}

/// INSUFFICIENT MATERIAL, the proved static rule (Lean Ideas.InsufficientMaterial.insufficient_material): two kings and
/// at most one more piece, a knight or a bishop of either side -- no checkmate exists. (That no continuation adds
/// material -- no pawn, so no promotion -- is the dynamic step, not yet proved in Lean.)
pub fn insufficient(q: &Chess) -> bool {
    let b = q.board();
    let others = b.occupied() & !b.kings();
    match others.count() {
        0 => true,
        1 => (b.knights() | b.bishops()).contains(others.first().unwrap()),
        _ => false,
    }
}

/// A DRAWN position for a decision: stalemate, a claimable draw, insufficient material.
pub fn drawn(q: &Chess) -> Option<&'static str> {
    if let Some(w) = claimable_draw(q) { return Some(w); }
    if insufficient(q) { return Some("insufficient material"); }
    if !q.is_check() && q.legal_moves().is_empty() { return Some("stalemate"); }
    None
}

/// What a draw is worth to the side to move in `pos`, in material: it turns the balance into nothing -- minus what it is
/// ahead (a draw costs the side ahead its lead and gives the side behind its deficit back).
pub fn draw_net(pos: &Chess) -> i64 { -kernel_material(pos, pos.turn()) }

/// game_draw_check(fen, seen_fens) -> for every legal move of the side to move, the draw it walks into, if any, with the
/// given positions as the game's history -- the context's verification hook.
#[pyfunction]
pub fn game_draw_check(fen: String, history: Vec<String>) -> Vec<(String, String)> {
    let Some(pos) = parse(&fen) else { return vec![] };
    let mut seen: HashMap<u64, u8> = HashMap::new();
    for f in history.iter().chain(std::iter::once(&fen)) {
        if let Some(p) = parse(f) { *seen.entry(crate::threat::zkey(&p)).or_insert(0) += 1; }
    }
    install(seen);
    let out = pos.legal_moves().into_iter().filter_map(|m| {
        let q = after(&pos, &m);
        drawn(&q).map(|w| (m.to_uci(CastlingMode::Standard).to_string(), w.to_string()))
    }).collect();
    clear();
    out
}
