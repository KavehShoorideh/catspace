//! board.rs -- rebuild a position from the corpus arrays, with no FEN round-trip.
//!
//! The corpus already stores a complete position as `tok (64,) uint8` (jepa vocab: 0 empty,
//! 1-6 white PNBRQK, 7-12 black pnbrqk, indexed by square a1=0..h8=63) plus
//! `glob (6,) = [turn, K, Q, k, q, ep_file+1]`. Parsing FENs here would be both slower and a
//! second place for the encoding to be got wrong, so we read the arrays directly -- exactly as
//! `concepts_reference.board_from_row` does on the Python side.

use shakmaty::{Bitboard, Board, CastlingMode, Chess, Color, FromSetup, Piece, Role, Setup, Square};

pub fn piece_of_token(t: u8) -> Option<Piece> {
    if t == 0 || t > 12 {
        return None;
    }
    let (role, color) = match t {
        1 => (Role::Pawn, Color::White),
        2 => (Role::Knight, Color::White),
        3 => (Role::Bishop, Color::White),
        4 => (Role::Rook, Color::White),
        5 => (Role::Queen, Color::White),
        6 => (Role::King, Color::White),
        7 => (Role::Pawn, Color::Black),
        8 => (Role::Knight, Color::Black),
        9 => (Role::Bishop, Color::Black),
        10 => (Role::Rook, Color::Black),
        11 => (Role::Queen, Color::Black),
        _ => (Role::King, Color::Black),
    };
    Some(Piece { color, role })
}

/// `tok` is 64 entries, `glob` is 6. Returns None only if the position cannot be made legal
/// enough to generate moves from; the caller counts those rather than silently zero-filling.
pub fn position_from_row(tok: &[u8], glob: &[u8]) -> Option<Chess> {
    let mut b = Board::empty();
    for sq_i in 0..64u32 {
        if let Some(p) = piece_of_token(tok[sq_i as usize]) {
            b.set_piece_at(Square::new(sq_i), p);
        }
    }
    let turn = if glob[0] != 0 { Color::White } else { Color::Black };

    // Castling rights are stored as ROOK squares in shakmaty, not as a 4-bit code.
    let mut rights = Bitboard::EMPTY;
    if glob[1] != 0 { rights |= Bitboard::from_square(Square::H1); }
    if glob[2] != 0 { rights |= Bitboard::from_square(Square::A1); }
    if glob[3] != 0 { rights |= Bitboard::from_square(Square::H8); }
    if glob[4] != 0 { rights |= Bitboard::from_square(Square::A8); }

    // The ep TARGET sits behind the pawn that just double-pushed: rank 6 when White is to move,
    // rank 3 when Black is. Same convention as the Python oracle.
    let ep = if glob[5] != 0 {
        let file = (glob[5] - 1) as u32;
        let rank = if turn == Color::White { 5u32 } else { 2u32 };
        Some(Square::new(rank * 8 + file))
    } else {
        None
    };

    let mut s = Setup::empty();
    s.board = b;
    s.turn = turn;
    s.castling_rights = rights;
    s.ep_square = ep;
    s.halfmoves = 0;
    s.fullmoves = std::num::NonZeroU32::new(1).unwrap();

    // python-chess is permissive about positions that a strict rules engine rejects (an ep
    // square with no pawn to take, castling rights whose rook has moved, and so on), and the
    // corpus contains real game positions rather than curated ones. Mirror that permissiveness
    // instead of dropping rows: a dropped row would show up later as a mysterious feature
    // mismatch rather than as an error.
    Chess::from_setup(s, CastlingMode::Standard)
        .or_else(|e| e.ignore_impossible_check())
        .or_else(|e| e.ignore_invalid_castling_rights())
        .or_else(|e| e.ignore_invalid_ep_square())
        .or_else(|e| e.ignore_too_much_material())
        .ok()
}
