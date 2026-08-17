//! concepts.rs -- the position-level half of the basis.
//!
//! Counterpart of `concepts_reference.glob_features`. Column order is fixed by
//! `names::glob_names()` and must not drift.

use shakmaty::attacks;
use shakmaty::{Bitboard, Chess, Color, Piece, Position, Role, Square};

use crate::names::N_GLOB;
use crate::probes::piece_val;

const ROLES: [Role; 6] = [
    Role::Pawn,
    Role::Knight,
    Role::Bishop,
    Role::Rook,
    Role::Queen,
    Role::King,
];

/// (passed, isolated, doubled, backward), each an 8-vector over files, for one colour.
fn pawn_structure(pos: &Chess, color: Color) -> [[f32; 8]; 4] {
    let board = pos.board();
    let ours = board.by_piece(Piece { color, role: Role::Pawn });
    let theirs = board.by_piece(Piece { color: color.other(), role: Role::Pawn });
    let fwd: i32 = if color == Color::White { 1 } else { -1 };

    let mut files_ours = [0i32; 8];
    for s in ours {
        files_ours[u32::from(s.file()) as usize] += 1;
    }
    let mut passed = [0f32; 8];
    let mut iso = [0f32; 8];
    let mut dbl = [0f32; 8];
    let mut back = [0f32; 8];

    for s in ours {
        let f = u32::from(s.file()) as i32;
        let r = u32::from(s.rank()) as i32;
        let blocked = theirs.into_iter().any(|t| {
            let tf = u32::from(t.file()) as i32;
            let tr = u32::from(t.rank()) as i32;
            (tf == f - 1 || tf == f || tf == f + 1) && (tr - r) * fwd > 0
        });
        if !blocked {
            passed[f as usize] = 1.0;
        }
        let has_neighbour = [f - 1, f + 1]
            .iter()
            .any(|&g| (0..8).contains(&g) && files_ours[g as usize] > 0);
        if !has_neighbour {
            iso[f as usize] = 1.0;
        }
        // BACKWARD: no friendly pawn on an adjacent file at or behind this rank, so it cannot be
        // supported by one. Distinct from isolated, and deliberately not the same predicate.
        let supported = ours.into_iter().any(|t| {
            if t == s {
                return false;
            }
            let tf = u32::from(t.file()) as i32;
            let tr = u32::from(t.rank()) as i32;
            (tf == f - 1 || tf == f + 1) && (tr - r) * fwd <= 0
        });
        if !supported {
            back[f as usize] = 1.0;
        }
    }
    for f in 0..8 {
        dbl[f] = (files_ours[f] - 1).max(0) as f32;
    }
    [passed, iso, dbl, back]
}

pub fn glob_features(pos: &Chess, repetition: f32, halfmove: f32, out: &mut [f32]) {
    let board = pos.board();
    let mut i = 0usize;

    let mut counts = [[0f32; 6]; 2];
    for (ci, col) in [Color::White, Color::Black].iter().enumerate() {
        for (ri, r) in ROLES.iter().enumerate() {
            let n = board
                .by_piece(Piece { color: *col, role: *r })
                .count() as f32;
            counts[ci][ri] = n;
            out[i + ci * 6 + ri] = n;
        }
    }
    i += 12;

    let mat = |ci: usize| -> f32 {
        ROLES
            .iter()
            .enumerate()
            .filter(|(_, r)| **r != Role::King)
            .map(|(ri, r)| piece_val(*r) * counts[ci][ri])
            .sum()
    };
    let (matw, matb) = (mat(0), mat(1));
    out[i] = matw - matb;
    out[i + 1] = (matw + matb) - (counts[0][0] + counts[1][0]);
    out[i + 2] = board.occupied().count() as f32;
    i += 3;

    // pawn structure: kind-major, then colour, then file -- matching glob_names()
    let ps = [
        pawn_structure(pos, Color::White),
        pawn_structure(pos, Color::Black),
    ];
    for kind in 0..4 {
        for c in 0..2 {
            out[i..i + 8].copy_from_slice(&ps[c][kind]);
            i += 8;
        }
    }

    let occ = board.occupied();
    for kind in 0..4 {
        for col in [Color::White, Color::Black] {
            let k = match board.king_of(col) {
                Some(k) => k,
                None => {
                    i += 1;
                    continue;
                }
            };
            let ring = attacks::king_attacks(k);
            out[i] = match kind {
                0 => ring
                    .into_iter()
                    .map(|s| board.attacks_to(s, col.other(), occ).count() as f32)
                    .sum(),
                1 => ring
                    .into_iter()
                    .filter(|&s| {
                        board
                            .piece_at(s)
                            .map_or(false, |p| p.color == col && p.role == Role::Pawn)
                    })
                    .count() as f32,
                2 => ring
                    .into_iter()
                    .filter(|&s| {
                        board.piece_at(s).is_none()
                            && board.attacks_to(s, col.other(), occ).is_empty()
                    })
                    .count() as f32,
                _ => {
                    let kf = u32::from(k.file()) as i32;
                    let mut files: Vec<i32> = vec![(kf - 1).max(0), kf, (kf + 1).min(7)];
                    files.sort_unstable();
                    files.dedup();
                    files
                        .iter()
                        .filter(|&&f| {
                            !(0..8).any(|r| {
                                board
                                    .piece_at(Square::new((r * 8 + f) as u32))
                                    .map_or(false, |p| p.role == Role::Pawn && p.color == col)
                            })
                        })
                        .count() as f32
                }
            };
            i += 1;
        }
    }

    out[i] = if pos.turn() == Color::White { 1.0 } else { 0.0 };
    let cr: Bitboard = pos.castles().castling_rights();
    out[i + 1] = if cr.contains(Square::H1) { 1.0 } else { 0.0 };
    out[i + 2] = if cr.contains(Square::A1) { 1.0 } else { 0.0 };
    out[i + 3] = if cr.contains(Square::H8) { 1.0 } else { 0.0 };
    out[i + 4] = if cr.contains(Square::A8) { 1.0 } else { 0.0 };
    out[i + 5] = pos
        .maybe_ep_square()
        .map_or(0.0, |s| (u32::from(s.file()) + 1) as f32);
    out[i + 6] = if pos.is_check() { 1.0 } else { 0.0 };
    out[i + 7] = pos.checkers().count() as f32;
    i += 8;

    let legal = pos.legal_moves();
    out[i] = legal.iter().filter(|m| m.is_capture()).count() as f32;
    out[i + 1] = legal.len() as f32;
    out[i + 2] = legal
        .iter()
        .filter(|m| {
            let mut p2 = pos.clone();
            p2.play_unchecked(**m);
            p2.is_check()
        })
        .count() as f32;
    i += 3;

    out[i] = repetition;
    out[i + 1] = halfmove;
    debug_assert_eq!(i + 2, N_GLOB);
}
