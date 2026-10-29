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
//!   INSUFFICIENT the kernel's rule (shakmaty / python-chess): the bare king, the lone knight, one-colour bishops --
//!               a dead position (Lean Ideas.DeadPosition for the bishops; plan 76.1 (e))
//!   STALEMATE   no legal move and not in check
use super::*;
use std::cell::RefCell;
use std::collections::HashMap;

thread_local! {
    /// the repetition keys (zobrist: board, side, castling, a LEGAL en passant -- the library's repetitionKey) of the game's
    /// positions since the last irreversible move -> how often each occurred, the position being planned included
    static SEEN: RefCell<HashMap<u64, u8>> = RefCell::new(HashMap::new());
}

thread_local! {
    /// THE GAME'S LINE (the vocabulary's time word, (ago k T)): its positions in order, the one being planned last
    static LINE: RefCell<Vec<shakmaty::Chess>> = RefCell::new(Vec::new());
}
pub fn install_line(line: Vec<shakmaty::Chess>) { LINE.with(|l| *l.borrow_mut() = line); }
/// The game's position `plies` before the one being planned (None before the game's start or with no line installed).
pub fn back(plies: u32) -> Option<shakmaty::Chess> {
    LINE.with(|l| { let l = l.borrow(); l.len().checked_sub(1 + plies as usize).and_then(|i| l.get(i).cloned()) })
}
/// set_vocab_line(fens) -> the game's line for (ago k T) in analysis (the planner installs its own in play).
#[pyo3::pyfunction]
pub fn set_vocab_line(fens: Vec<String>) { install_line(fens.iter().filter_map(|f| crate::parse(f)).collect()); }

pub fn install(seen: HashMap<u64, u8>) { SEEN.with(|s| *s.borrow_mut() = seen); }
pub fn clear() { SEEN.with(|s| s.borrow_mut().clear()); }

/// How often the game has already seen this position (0 in a puzzle).
pub fn seen(q: &Chess) -> u8 {
    SEEN.with(|s| { let s = s.borrow(); if s.is_empty() { 0 } else { s.get(&crate::threat::zkey(q)).copied().unwrap_or(0) } })
}

/// THE RULES' ENDS, ONE DEFINITION EACH (plan 76.2 (c)): the Rust core's implementation of the Game surface's `terminal` (game.py; its spec rules/chess.gdl,
/// proved the kernel's in Lean Ideas.Spec.LegalEquiv / NextEquiv): each rule read once from the kernel, here and nowhere else in the crate. `terminal` composes
/// them in Game.terminal's order (Lean Chess.gameStatus: a mate on the move comes first) and is certified against Game.terminal and the spec by the oracle test
/// (tests/test_rules_shelf.py, through `game_terminal`); `claimable_draw` and `drawn` are readings of the same rules for the searches, in their own order.
pub fn mate(q: &Chess) -> bool { q.is_checkmate() }
/// No legal move and not in check.
pub fn stalemate(q: &Chess) -> bool { !q.is_check() && q.legal_moves().is_empty() }
/// The third occurrence: the game's record holds the position's identity twice before it (`seen`: the history installed).
pub fn threefold(q: &Chess) -> bool { seen(q) >= 2 }
/// The fifty-move clock at its limit (a hundred half-moves with no capture and no pawn move).
pub fn fifty(q: &Chess) -> bool { q.halfmoves() >= 100 }

/// THE TERMINAL, in Game.terminal's kinds and order: 'mate', 'stalemate', 'draw' (insufficient material), 'repetition', 'fifty-move'; None: the game goes on.
pub fn terminal(q: &Chess) -> Option<&'static str> {
    if mate(q) { return Some("mate"); }
    if stalemate(q) { return Some("stalemate"); }
    if insufficient(q) { return Some("draw"); }
    if threefold(q) { return Some("repetition"); }
    if fifty(q) { return Some("fifty-move"); }
    None
}

/// A position reached in ANALYSIS (not the root) that either side may claim as a draw: its third occurrence, or the
/// fifty-move clock run out (a checkmate on that move comes first) -- the claimable ends of `terminal`, read by the searches.
pub fn claimable_draw(q: &Chess) -> Option<&'static str> {
    if fifty(q) && !mate(q) { return Some("the fifty-move rule"); }
    if threefold(q) { return Some("threefold repetition"); }
    None
}

/// game_terminal(fen, history) -> the Game surface's terminal of the position `fen` with the game's positions BEFORE it (`history`, the game record's FENs) --
/// the Rust implementation read for the oracle test (plan 76.2 (c)).
#[pyfunction]
pub fn game_terminal(fen: String, history: Vec<String>) -> Option<String> {
    let pos = parse(&fen)?;
    let mut seen: HashMap<u64, u8> = HashMap::new();
    for f in history.iter() {
        if let Some(p) = parse(f) { *seen.entry(crate::threat::zkey(&p)).or_insert(0) += 1; }
    }
    install(seen);
    let out = terminal(&pos).map(|s| s.to_string());
    clear();
    out
}

/// INSUFFICIENT MATERIAL, the kernel's rule (plan 76.1 (e)): shakmaty's `is_insufficient_material` -- both sides lack
/// mating material by their material alone, python-chess's `has_insufficient_material` (the rule Game.terminal reads and
/// the spec's `draw`, rules/chess.gdl): the bare king, a king and a knight against kings and queens, and EVERY BISHOP OF
/// THE BOARD ON ONE COLOUR with no pawn and no knight. A dead position (FIDE 5.2.2): the bishops case and the bare kings
/// for every continuation (Lean Ideas.DeadPosition.one_colour_dead, dead_of_spec_draw), a king and one minor piece in the
/// position itself (Ideas.InsufficientMaterial.insufficient_material). It read only "two kings and at most one knight or
/// bishop" before, which missed the one-colour bishops (the nine surprises of rules/surprises/, plan 76.0).
pub fn insufficient(q: &Chess) -> bool {
    q.is_insufficient_material()
}

/// A DRAWN position for a decision: stalemate, a claimable draw, insufficient material.
pub fn drawn(q: &Chess) -> Option<&'static str> {
    if let Some(w) = claimable_draw(q) { return Some(w); }
    if insufficient(q) { return Some("insufficient material"); }
    if stalemate(q) { return Some("stalemate"); }
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
