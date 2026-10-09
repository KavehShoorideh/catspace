//! lc0's classical input (INPUT_CLASSICAL_112_PLANE): 8 history slots of 13 planes (the side to move's P N B R Q K,
//! the opponent's P N B R Q K, a repetition plane), then: our O-O-O, our O-O, their O-O-O, their O-O, black to move,
//! the rule-50 count, zeros, ones. Seen from the side to move (the board mirrored for Black).
use shakmaty::{Board, CastlingSide, Chess, Color, EnPassantMode, Position, Rank, Role, Square};

pub const PLANES: usize = 112;

fn put(x: &mut [f32], slot: usize, b: &Board, us: Color) {
    let flip = |sq: Square| -> usize { let i = sq as usize; if us == Color::Black { i ^ 56 } else { i } };
    let roles = [Role::Pawn, Role::Knight, Role::Bishop, Role::Rook, Role::Queen, Role::King];
    for (k, role) in roles.iter().enumerate() {
        for sq in b.by_piece(role.of(us)) { x[(slot * 13 + k) * 64 + flip(sq)] = 1.0; }
        for sq in b.by_piece(role.of(!us)) { x[(slot * 13 + 6 + k) * 64 + flip(sq)] = 1.0; }
    }
}

fn aux(x: &mut [f32], pos: &Chess) {
    let us = pos.turn();
    let castles = pos.castles();
    let fill = |x: &mut [f32], plane: usize, v: f32| { for i in 0..64 { x[plane * 64 + i] = v; } };
    if castles.has(us, CastlingSide::QueenSide) { fill(x, 104, 1.0); }
    if castles.has(us, CastlingSide::KingSide) { fill(x, 105, 1.0); }
    if castles.has(!us, CastlingSide::QueenSide) { fill(x, 106, 1.0); }
    if castles.has(!us, CastlingSide::KingSide) { fill(x, 107, 1.0); }
    if us == Color::Black { fill(x, 108, 1.0); }
    fill(x, 109, pos.halfmoves() as f32);
    fill(x, 111, 1.0);
}

/// A bare position, with lc0's history fill for a FEN: no fill at the standard start; with an en-passant square the
/// earlier slots hold the position before the double step (the pawn back); otherwise copies of the position.
pub fn encode(pos: &Chess) -> Vec<f32> {
    let mut x = vec![0f32; PLANES * 64];
    let us = pos.turn();
    let b = pos.board();
    let start = b == Chess::default().board() && us == Color::White;
    let mut prev = b.clone();
    if let Some(ep) = pos.ep_square(EnPassantMode::Always) {
        let (now_sq, back_sq) = if us == Color::White {
            (Square::from_coords(ep.file(), Rank::Fifth), Square::from_coords(ep.file(), Rank::Seventh))
        } else {
            (Square::from_coords(ep.file(), Rank::Fourth), Square::from_coords(ep.file(), Rank::Second))
        };
        if let Some(p) = prev.remove_piece_at(now_sq) { prev.set_piece_at(back_sq, p); }
    }
    let slots = if start { 1 } else { 8 };
    for slot in 0..slots { put(&mut x, slot, if slot == 0 { b } else { &prev }, us); }
    aux(&mut x, pos);
    x
}

/// With the real game history (oldest first, current last): the last 8 positions, most recent in slot 0, each seen
/// from the CURRENT side to move; repetition planes set where a position occurred before; earlier slots empty when
/// the game is shorter than 8 plies.
pub fn encode_history(history: &[Chess]) -> Vec<f32> {
    let mut x = vec![0f32; PLANES * 64];
    let pos = history.last().expect("non-empty history");
    let us = pos.turn();
    let n = history.len();
    for slot in 0..n.min(8) {
        let idx = n - 1 - slot;
        let p = &history[idx];
        put(&mut x, slot, p.board(), us);
        // repetition: the position occurred EARLIER in the game (up to this point), not anywhere in the history
        let reps = history[..=idx].iter().filter(|q| q.board() == p.board() && q.turn() == p.turn()).count();
        if reps > 1 { for i in 0..64 { x[(slot * 13 + 12) * 64 + i] = 1.0; } }
    }
    aux(&mut x, pos);
    x
}
