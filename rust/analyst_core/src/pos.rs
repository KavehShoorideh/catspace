//! THE CHASE'S POSITION (ANALYST_PLAN 52.13 (5)): a position built from a FEN ONCE, moves played on it without a FEN per reply. The chase of the agenda (agenda.py
//! `_chase_route`) reads every reply of every forcing route; the python-chess board copy + FEN render per reply cost 15.4 us against 0.4 us for this evaluator
//! (OBS, journal 2026-10-18). The words are the vocabulary's, not new ones:
//!   balance   the owner's material less the other's, plus 64 (vocab.rs Term::Balance; kings out, p1 n3 b3 r5 q9)
//!   replies   every legal reply: (uci, balance change for the side that just moved, gives check, captures, captures on the square the last move went to, we then
//!             have a mate in one (rules: some legal move of the side to move checkmates))
//!   diff      the squares whose piece or whose attacker counts (either colour) differ between two positions (regress.diff_squares, the read-set diff)
//!   exchange  the best static exchange (rules::see, the mirror of Lean Ideas.ExchangeLaw.seeXB) the side to move has on a piece of the other side
use super::*;
use pyo3::exceptions::PyValueError;
use shakmaty::Bitboard;

fn bal(p: &Chess, owner: Color) -> i64 {
    crate::vocab::material(p.board(), owner) - crate::vocab::material(p.board(), !owner) + 64
}

#[pyclass]
#[derive(Clone)]
pub struct Pos {
    pos: Chess,
    last_to: Option<Square>,
}

#[pymethods]
impl Pos {
    #[new]
    fn new(fen: &str) -> PyResult<Pos> {
        parse(fen).map(|pos| Pos { pos, last_to: None }).ok_or_else(|| PyValueError::new_err(format!("illegal or unparsable fen: {fen}")))
    }

    /// The position after the legal move `uci` (the square it went to is remembered for `replies`).
    fn push(&self, uci: &str) -> PyResult<Pos> {
        let m = shakmaty::uci::UciMove::from_ascii(uci.as_bytes()).ok().and_then(|u| u.to_move(&self.pos).ok())
            .ok_or_else(|| PyValueError::new_err(format!("illegal move {uci}")))?;
        let mut q = self.pos.clone();
        q.play_unchecked(&m);
        Ok(Pos { pos: q, last_to: Some(m.to()) })
    }

    fn fen(&self) -> String { Fen::from_position(self.pos.clone(), shakmaty::EnPassantMode::Legal).to_string() }

    /// `owner`'s material less the other's, plus 64 (white = true).
    fn balance(&self, white: bool) -> i64 { bal(&self.pos, if white { Color::White } else { Color::Black }) }

    fn legal_count(&self) -> usize { self.pos.legal_moves().len() }
    /// The side to move has a mate in one (the rules: some legal move checkmates).
    fn mate_in_one(&self) -> bool { self.pos.legal_moves().iter().any(|m| { let mut r = self.pos.clone(); r.play_unchecked(m); r.is_checkmate() }) }
    fn is_check(&self) -> bool { self.pos.is_check() }
    fn turn_white(&self) -> bool { self.pos.turn() == Color::White }

    /// Every legal reply of the side to move: (uci, balance change for the side that just moved, check, capture, capture on the last move's square, mate in one for the
    /// side that just moved afterwards).
    fn replies(&self) -> Vec<(String, i64, bool, bool, bool, bool)> {
        let me = !self.pos.turn();
        let base = bal(&self.pos, me);
        self.pos.legal_moves().iter().map(|m| {
            let mut q = self.pos.clone();
            q.play_unchecked(m);
            let mate1 = q.legal_moves().iter().any(|m2| { let mut r = q.clone(); r.play_unchecked(m2); r.is_checkmate() });
            (m.to_uci(CastlingMode::Standard).to_string(), bal(&q, me) - base, q.is_check(), m.is_capture(), m.is_capture() && Some(m.to()) == self.last_to, mate1)
        }).collect()
    }

    /// The squares whose piece or attacker counts (either colour) differ between this position and `other`.
    fn diff(&self, other: &Pos) -> Vec<u8> {
        let (a, b) = (self.pos.board(), other.pos.board());
        let mut out = vec![];
        for i in 0..64u32 {
            let s = Square::new(i);
            let occ = (a.occupied(), b.occupied());
            let cnt = |bd: &shakmaty::Board, o: Bitboard, c: Color| bd.attacks_to(s, c, o).count();
            if a.piece_at(s) != b.piece_at(s)
                || cnt(a, occ.0, Color::White) != cnt(b, occ.1, Color::White)
                || cnt(a, occ.0, Color::Black) != cnt(b, occ.1, Color::Black) {
                out.push(i as u8);
            }
        }
        out
    }

    /// The best static exchange the SIDE TO MOVE has on a piece of the other side (a non-king piece it attacks); 0 when none.
    fn exchange(&self) -> i64 {
        let b = self.pos.board();
        let us = self.pos.turn();
        let mut best = 0;
        for s in b.by_color(!us) {
            if b.role_at(s) == Some(Role::King) || b.attacks_to(s, us, b.occupied()).is_empty() { continue; }
            best = best.max(crate::rules::see(b, s, us));
        }
        best
    }
}
