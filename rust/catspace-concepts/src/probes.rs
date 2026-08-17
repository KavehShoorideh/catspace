//! probes.rs -- the per-square half of the basis, and SEE.
//!
//! Every function here has exactly one counterpart in `concepts_reference.py`, and the pair is
//! held to elementwise equality by `test_probe_features.py`. Where a definition could reasonably
//! go two ways, the Python is the specification and this file follows it -- including tie-breaks,
//! which is where a port of a "standard" algorithm usually diverges without anyone noticing.

use shakmaty::attacks;
use shakmaty::{Bitboard, Board, Chess, Color, Piece, Position, Role, Square};

use crate::names::N_SQ;

/// Matches `control.SEE_VALUES` exactly. The king's 20000 is what makes a king recapture into a
/// defended square lose by construction.
#[inline]
pub fn see_value(r: Role) -> i32 {
    match r {
        Role::Pawn => 100,
        Role::Knight => 300,
        Role::Bishop => 300,
        Role::Rook => 500,
        Role::Queen => 900,
        Role::King => 20000,
    }
}

/// Used for `least_att_*`, which is a different (coarse) scale from SEE -- mirrors
/// `concepts_reference.PIECE_VAL`.
#[inline]
pub fn piece_val(r: Role) -> f32 {
    match r {
        Role::Pawn => 1.0,
        Role::Knight => 3.0,
        Role::Bishop => 3.0,
        Role::Rook => 5.0,
        Role::Queen => 9.0,
        Role::King => 10.0,
    }
}

/// Static exchange evaluation, transcribed from `control._see_square`.
///
/// TIE-BREAK, which is the whole ballgame for matching: the Python builds the attacker list by
/// iterating `board.attackers(color, sq)` (ascending square index) and then does a STABLE sort by
/// value. So among equally-valuable attackers the one on the LOWEST square index moves first.
/// `min_by_key` over an ascending iteration reproduces that; picking any other equal-valued
/// attacker gives a different exchange order and a silently different number.
pub fn see_square(board: &Board, square: Square, side: Color) -> i32 {
    let target = match board.piece_at(square) {
        Some(p) => p,
        None => return 0,
    };
    let mut b = board.clone();
    let mut gain: Vec<i32> = vec![see_value(target.role)];
    let mut attacker_color = side;
    let mut depth: usize = 0;

    loop {
        let occ = b.occupied();
        let atts = b.attacks_to(square, attacker_color, occ);
        // least valuable attacker, lowest square index breaking ties
        let mut best: Option<(i32, Square, Role)> = None;
        for from in atts {
            if let Some(p) = b.piece_at(from) {
                let v = see_value(p.role);
                if best.map_or(true, |(bv, _, _)| v < bv) {
                    best = Some((v, from, p.role));
                }
            }
        }
        let (val, from_sq, role) = match best {
            Some(t) => t,
            None => break,
        };
        if role == Role::King {
            // a king may not capture into a square the other side still defends
            let occ2 = b.occupied();
            if !b.attacks_to(square, attacker_color.other(), occ2).is_empty() {
                break;
            }
        }
        depth += 1;
        gain.push(val - gain[depth - 1]);
        let moving = b.remove_piece_at(from_sq);
        if let Some(mp) = moving {
            b.set_piece_at(square, mp);
        }
        attacker_color = attacker_color.other();
        if role == Role::King {
            break;
        }
    }
    // backward negamax fold: each side continues only if it improves. depth == 1 folds zero
    // times, leaving an uncontested capture at its full value -- the subtle case the Python
    // comments call out.
    let mut i = depth;
    while i > 1 {
        i -= 1;
        gain[i - 1] = -std::cmp::max(-gain[i - 1], gain[i]);
    }
    gain[0]
}

#[inline]
fn token_index(p: Piece) -> usize {
    let base = match p.role {
        Role::Pawn => 0,
        Role::Knight => 1,
        Role::Bishop => 2,
        Role::Rook => 3,
        Role::Queen => 4,
        Role::King => 5,
    };
    base + if p.color == Color::White { 0 } else { 6 }
}

/// Fills `out` (64 * N_SQ, row-major by square) for one position.
pub fn sq_features(pos: &Chess, out: &mut [f32]) {
    let board = pos.board();
    let occ = board.occupied();
    let ring_w = board
        .king_of(Color::White)
        .map(attacks::king_attacks)
        .unwrap_or(Bitboard::EMPTY);
    let ring_b = board
        .king_of(Color::Black)
        .map(attacks::king_attacks)
        .unwrap_or(Bitboard::EMPTY);

    for sq_i in 0..64u32 {
        let sq = Square::new(sq_i);
        let o = (sq_i as usize) * N_SQ;
        let aw = board.attacks_to(sq, Color::White, occ);
        let ab = board.attacks_to(sq, Color::Black, occ);

        if let Some(p) = board.piece_at(sq) {
            out[o + token_index(p)] = 1.0;
        }
        out[o + 12] = aw.count() as f32;
        out[o + 13] = ab.count() as f32;
        out[o + 14] = aw
            .into_iter()
            .filter_map(|s| board.piece_at(s).map(|p| piece_val(p.role)))
            .fold(f32::INFINITY, f32::min);
        if out[o + 14].is_infinite() {
            out[o + 14] = 0.0;
        }
        out[o + 15] = ab
            .into_iter()
            .filter_map(|s| board.piece_at(s).map(|p| piece_val(p.role)))
            .fold(f32::INFINITY, f32::min);
        if out[o + 15].is_infinite() {
            out[o + 15] = 0.0;
        }

        if let Some(p) = board.piece_at(sq) {
            // "what does capturing here win?" is only a question for the side that does not own
            // the piece -- same convention as the oracle.
            out[o + 16] = see_square(board, sq, p.color.other()) as f32;

            // PINNED: the occupant shields its own king from an enemy slider. python-chess
            // answers this with `is_pinned`; shakmaty has no equivalent, so it is spelled out --
            // an enemy slider aligned with our king, with this piece the ONLY man between.
            let mut pinned = 0.0f32;
            if let Some(k) = board.king_of(p.color) {
                if k != sq {
                    for enemy in board.by_color(p.color.other()) {
                        if let Some(ep) = board.piece_at(enemy) {
                            if !matches!(ep.role, Role::Bishop | Role::Rook | Role::Queen) {
                                continue;
                            }
                            // the slider must actually attack along this line once cleared
                            let cleared = occ & !Bitboard::from_square(sq);
                            if attacks::attacks(enemy, ep, cleared).contains(k)
                                && attacks::between(enemy, k).contains(sq)
                                && (attacks::between(enemy, k) & occ).count() == 1
                            {
                                pinned = 1.0;
                                break;
                            }
                        }
                    }
                }
            }
            out[o + 17] = pinned;

            // PINNING: the mirror -- this piece holds an enemy man against the enemy king.
            let mut pinning = 0.0f32;
            if matches!(p.role, Role::Bishop | Role::Rook | Role::Queen) {
                if let Some(ek) = board.king_of(p.color.other()) {
                    let btw = attacks::between(sq, ek);
                    if (btw & occ).count() == 1 {
                        let mid = (btw & occ).into_iter().next().unwrap();
                        if let Some(mp) = board.piece_at(mid) {
                            let cleared = occ & !Bitboard::from_square(mid);
                            if mp.color != p.color && attacks::attacks(sq, p, cleared).contains(ek)
                            {
                                pinning = 1.0;
                            }
                        }
                    }
                }
            }
            out[o + 18] = pinning;

            let own = if p.color == Color::White { aw } else { ab };
            let foe = if p.color == Color::White { ab } else { aw };
            out[o + 23] = if !foe.is_empty() && own.is_empty() { 1.0 } else { 0.0 };
            out[o + 24] = own.count() as f32;
            out[o + 25] = attacks::attacks(sq, p, occ).count() as f32;
        }

        // X-RAY through exactly one blocker: ask who would attack this square if nothing blocked,
        // drop those already attacking, keep the ones with a single man in the way. Removing the
        // ATTACKERS from occupancy (the obvious-looking version) is backwards -- it deletes the
        // very pieces whose sight-lines are in question. That bug was caught by the f7 case in
        // the oracle's tests: Qf3 behind Nf6.
        for (col, direct, idx) in [(Color::White, aw, 19usize), (Color::Black, ab, 20usize)] {
            let unblocked = board.attacks_to(sq, col, Bitboard::EMPTY);
            let hidden = unblocked & !direct;
            out[o + idx] = hidden
                .into_iter()
                .filter(|&s2| (attacks::between(s2, sq) & occ).count() == 1)
                .count() as f32;
        }

        out[o + 21] = if aw
            .into_iter()
            .any(|s| board.piece_at(s).map_or(false, |p| p.role == Role::Pawn))
        {
            1.0
        } else {
            0.0
        };
        out[o + 22] = if ab
            .into_iter()
            .any(|s| board.piece_at(s).map_or(false, |p| p.role == Role::Pawn))
        {
            1.0
        } else {
            0.0
        };
        out[o + 26] = if ring_w.contains(sq) { 1.0 } else { 0.0 };
        out[o + 27] = if ring_b.contains(sq) { 1.0 } else { 0.0 };
    }
}
