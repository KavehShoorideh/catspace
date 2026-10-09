//! lc0's policy table: 1,858 moves, written from the side to move's point of view (Black's moves mirrored), the index
//! of each in the network's policy output. Read out of lc0's own `--verbose-move-stats` (each legal move printed with
//! its index) over thousands of positions; the eight entries no position produced follow from the table's ordering.
//! Conventions: castling is the king taking its own rook (e1h1, e1a1); a pawn's plain move to the last rank is a KNIGHT
//! promotion; queen, rook and bishop promotions carry their suffix.
use shakmaty::{Color, Move, Role, Square};
use std::collections::HashMap;
use std::sync::OnceLock;

static TABLE: OnceLock<HashMap<String, usize>> = OnceLock::new();

/// The table: move (from the side to move's view) -> policy index.
pub fn policy_index() -> &'static HashMap<String, usize> {
    TABLE.get_or_init(|| include_str!("lc0_policy.txt").lines().enumerate().map(|(i, u)| (u.to_string(), i)).collect())
}

/// The policy index of a legal move for the side `us`.
pub fn move_index(m: &Move, us: Color) -> Option<usize> {
    let mirror = |sq: Square| -> Square { if us == Color::Black { Square::new((sq as u32) ^ 56) } else { sq } };
    let (from, to) = match m {
        Move::Castle { king, rook } => (*king, *rook),
        _ => (m.from()?, m.to()),
    };
    let mut s = format!("{}{}", mirror(from), mirror(to));
    if let Some(p) = m.promotion() {
        if p != Role::Knight { s.push(p.char()); }
    }
    policy_index().get(&s).copied()
}
