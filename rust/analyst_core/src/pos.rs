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

pub(crate) fn bal(p: &Chess, owner: Color) -> i64 {
    crate::vocab::material(p.board(), owner) - crate::vocab::material(p.board(), !owner) + 64
}

#[pyclass]
#[derive(Clone)]
pub struct Pos {
    pub(crate) pos: Chess,
    pub(crate) last_to: Option<Square>,
}

impl Pos {
    pub(crate) fn from_fen(fen: &str) -> PyResult<Pos> {
        parse(fen).map(|pos| Pos { pos, last_to: None }).ok_or_else(|| PyValueError::new_err(format!("illegal or unparsable fen: {fen}")))
    }
    pub(crate) fn push_move(&self, uci: &str) -> PyResult<Pos> {
        let m = shakmaty::uci::UciMove::from_ascii(uci.as_bytes()).ok().and_then(|u| u.to_move(&self.pos).ok())
            .ok_or_else(|| PyValueError::new_err(format!("illegal move {uci}")))?;
        let mut q = self.pos.clone();
        q.play_unchecked(&m);
        Ok(Pos { pos: q, last_to: Some(m.to()) })
    }
    /// The LEGAL move `uci` on this position (ValueError when it is not one: `play_unchecked` on anything else corrupts silently, RUST_BOUNDARY risk 7).
    pub(crate) fn legal_move(&self, uci: &str) -> PyResult<Move> {
        shakmaty::uci::UciMove::from_ascii(uci.as_bytes()).ok().and_then(|u| u.to_move(&self.pos).ok())
            .ok_or_else(|| PyValueError::new_err(format!("illegal move {uci}")))
    }
}

/// `graph.with_turn` on a shakmaty position: the same board with `c` to move, NO en passant square, the castling rights and the clocks kept -- STRICT (what the
/// FEN reader would accept: None when that side to move is not a legal position, e.g. the other side in check). The position itself when nothing changes.
pub(crate) fn with_turn_strict(p: &Chess, c: Color) -> Option<std::borrow::Cow<'_, Chess>> {
    if p.turn() == c && p.maybe_ep_square().is_none() { return Some(std::borrow::Cow::Borrowed(p)); }
    let mut s = p.clone().into_setup(shakmaty::EnPassantMode::Always);
    s.turn = c;
    s.ep_square = None;
    Chess::from_setup(s, CastlingMode::Standard).ok().map(std::borrow::Cow::Owned)
}

fn square(sq: u8) -> PyResult<Square> {
    if sq < 64 { Ok(Square::new(sq as u32)) } else { Err(PyValueError::new_err(format!("no square {sq} (squares are 0..63, a1 = 0)"))) }
}

fn colour(white: bool) -> Color { if white { Color::White } else { Color::Black } }

/// python-chess's piece types: pawn 1, knight 2, bishop 3, rook 4, queen 5, king 6.
fn piece_type(r: Role) -> u8 { match r { Role::Pawn => 1, Role::Knight => 2, Role::Bishop => 3, Role::Rook => 4, Role::Queen => 5, Role::King => 6 } }

/// PYTHON-CHESS'S ORDER of the legal moves (`Board.generate_legal_moves`, python-chess 1.11; RUST_BOUNDARY risk 4: Python list order sets tie-breaks, so traces
/// depend on it). Not in check, the pseudo-legal order: the pieces other than pawns by from-square descending, each by to-square descending; then castling, the
/// rook's square descending (the king side first); then the pawn captures (from, to descending; promotions Q R B N); the single pushes (to descending; Q R B N); the
/// double pushes (to descending); en passant. In check, the evasions: the king's moves first (to descending), then the rest in the same order. (regress.rs `py_key`
/// files castling among the king's moves; every user of it there drops castling, so it never reached an output.)
pub(crate) fn py_order(p: &Chess) -> Vec<Move> {
    let evasion = p.is_check();
    let mut v: Vec<Move> = p.legal_moves().into_iter().collect();
    v.sort_by_key(|m| py_order_key(m, evasion));
    v
}

fn py_order_key(m: &Move, evasion: bool) -> (u8, u8, i32, i32, u8) {
    let kf = if evasion && m.role() == Role::King { 0 } else { 1 };
    let promo = match m.promotion() { Some(Role::Rook) => 1, Some(Role::Bishop) => 2, Some(Role::Knight) => 3, _ => 0 };
    match *m {
        Move::Castle { rook, .. } => (kf, 1, -(rook as i32), 0, 0),
        Move::EnPassant { from, to } => (kf, 5, -(from as i32), -(to as i32), 0),
        Move::Normal { role: Role::Pawn, from, to, capture, .. } => {
            if capture.is_some() { (kf, 2, -(from as i32), -(to as i32), promo) }
            else if (to as i32 - from as i32).abs() == 8 { (kf, 3, -(to as i32), 0, promo) }
            else { (kf, 4, -(to as i32), 0, 0) }
        }
        Move::Normal { from, to, .. } => (kf, 0, -(from as i32), -(to as i32), 0),
        Move::Put { to, .. } => (kf, 6, -(to as i32), 0, 0),
    }
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
    fn diff(&self, other: &Pos) -> Vec<u8> { pos_diff(&self.pos, &other.pos) }

    /// The best static exchange the SIDE TO MOVE has on a piece of the other side (a non-king piece it attacks); 0 when none.
    fn exchange(&self) -> i64 { pos_exchange(&self.pos) }

    /// The SAN of the legal move `uci`, check and mate suffixes included (shakmaty's `SanPlus`; RUST_BOUNDARY risk 5: the trace and the play server print names, so
    /// the guard cross-checks this against python-chess's `board.san` before anything relies on it). ValueError on an illegal move.
    fn san(&self, uci: &str) -> PyResult<String> {
        let m = self.legal_move(uci)?;
        Ok(shakmaty::san::SanPlus::from_move(self.pos.clone(), &m).to_string())
    }

    /// The legal moves as UCI in PYTHON-CHESS'S ORDER (`[m.uci() for m in board.legal_moves]`; see `py_order`).
    fn legal_ucis(&self) -> PyResult<Vec<String>> { Ok(py_order(&self.pos).iter().map(|m| m.to_uci(CastlingMode::Standard).to_string()).collect()) }

    // ------------------------------------------------------------------------------------------------------------ plan 72.7 step 1: the handle of a node (RUST_BOUNDARY 3.1)

    /// The handle of a python-chess board from its parts, NO FEN STRING (graph.pos_of): the eight bitboards (pawns, knights, bishops, rooks, queens, kings, white,
    /// black), the side to move, the castling rights as the rooks' squares (python-chess `clean_castling_rights()`), the en passant square when a legal en passant
    /// capture exists (else None), the clocks -- exactly the fields `board.fen()` prints, so the position is the one `Pos(board.fen())` reads. ValueError when the
    /// parts are inconsistent or shakmaty does not hold the position (as the FEN reader).
    #[staticmethod]
    #[pyo3(signature = (pawns, knights, bishops, rooks, queens, kings, white, black, turn_white, castling, ep, halfmoves, fullmoves))]
    #[allow(clippy::too_many_arguments)]
    fn from_parts(pawns: u64, knights: u64, bishops: u64, rooks: u64, queens: u64, kings: u64, white: u64, black: u64, turn_white: bool, castling: u64, ep: Option<u8>,
                  halfmoves: u32, fullmoves: u32) -> PyResult<Pos> {
        let roles = [pawns, knights, bishops, rooks, queens, kings];
        let union = roles.iter().fold(0u64, |a, r| a | r);
        if roles.iter().map(|r| r.count_ones()).sum::<u32>() != union.count_ones() || union != (white | black) || white & black != 0 {
            return Err(PyValueError::new_err("inconsistent bitboards"));
        }
        let board = shakmaty::Board::from_bitboards(
            shakmaty::ByRole { pawn: Bitboard(pawns), knight: Bitboard(knights), bishop: Bitboard(bishops), rook: Bitboard(rooks), queen: Bitboard(queens), king: Bitboard(kings) },
            shakmaty::ByColor { white: Bitboard(white), black: Bitboard(black) });
        let mut s = shakmaty::Setup::empty();
        s.board = board;
        s.turn = colour(turn_white);
        s.castling_rights = Bitboard(castling);
        s.ep_square = match ep { Some(e) => Some(square(e)?), None => None };
        s.halfmoves = halfmoves;
        s.fullmoves = std::num::NonZeroU32::new(fullmoves).ok_or_else(|| PyValueError::new_err("fullmove number 0"))?;
        Chess::from_setup(s, CastlingMode::Standard).map(|pos| Pos { pos, last_to: None }).map_err(|_| PyValueError::new_err("illegal position"))
    }

    /// `board.fen()` exactly: every field python-chess prints, the clocks included (replaces graph.exact_fen; the bridge asserts the two are equal).
    fn fen_exact(&self) -> PyResult<String> { Ok(self.fen()) }

    /// The TRANSPOSITION KEY (zobrist: board, side to move, castling rights, a LEGAL en passant square; not the clocks) -- python-chess's `_transposition_key()`.
    fn key(&self) -> PyResult<u64> { Ok(crate::threat::zkey(&self.pos)) }

    /// `graph.with_turn(board, white)`: the same board with that side to move and no en passant square (castling rights and clocks kept). None when shakmaty
    /// does not hold it (the side not to move would stand in check), where python-chess would keep an illegal board.
    fn with_turn(&self, white: bool) -> PyResult<Option<Pos>> {
        Ok(with_turn_strict(&self.pos, colour(white)).map(|p| Pos { pos: p.into_owned(), last_to: None }))
    }

    /// The PASS (`game.pass_`: `graph.with_turn(board, not board.turn)`): the other side to move, en passant dropped, the clocks as they were. None in check.
    fn pass_turn(&self) -> PyResult<Option<Pos>> { self.with_turn(self.pos.turn() != Color::White) }

    /// `board.piece_at(sq)` as (python-chess piece type: pawn 1 .. king 6, white?) or None. Squares are 0..63, a1 = 0.
    fn piece_at(&self, sq: u8) -> PyResult<Option<(u8, bool)>> {
        Ok(self.pos.board().piece_at(square(sq)?).map(|p| (piece_type(p.role), p.color == Color::White)))
    }

    /// `board.attackers_mask(white, sq)`: the squares of that side's pieces attacking `sq` on this board (a u64 mask).
    fn attackers(&self, sq: u8, white: bool) -> PyResult<u64> {
        let b = self.pos.board();
        let c = colour(white);
        Ok((b.attacks_to(square(sq)?, c, b.occupied()) & b.by_color(c)).0)
    }

    /// `board.attacks_mask(sq)`: the squares the piece on `sq` attacks (0 for an empty square).
    fn attacks(&self, sq: u8) -> PyResult<u64> { Ok(self.pos.board().attacks_from(square(sq)?).0) }

    /// `board.is_capture(move)` for the legal move `uci` (en passant included, castling not). ValueError on an illegal move.
    fn is_capture(&self, uci: &str) -> PyResult<bool> { Ok(self.legal_move(uci)?.is_capture()) }

    /// `board.gives_check(move)` for the legal move `uci`. ValueError on an illegal move.
    fn gives_check(&self, uci: &str) -> PyResult<bool> {
        let m = self.legal_move(uci)?;
        let mut q = self.pos.clone();
        q.play_unchecked(&m);
        Ok(q.is_check())
    }

    /// `board.king(white)`: that side's king's square, or None.
    fn king(&self, white: bool) -> PyResult<Option<u8>> {
        let b = self.pos.board();
        Ok((b.kings() & b.by_color(colour(white))).last().map(|s| s as u8))
    }

    /// `board.board_fen()`: the placement field of the FEN.
    fn board_fen(&self) -> PyResult<String> { Ok(self.pos.board().to_string()) }

    /// `board.ep_square`: the en passant square after a double push (whether or not a capture there is legal), or None.
    fn ep_square(&self) -> PyResult<Option<u8>> { Ok(self.pos.maybe_ep_square().map(|s| s as u8)) }

    fn is_checkmate(&self) -> PyResult<bool> { Ok(self.pos.is_checkmate()) }

    fn is_stalemate(&self) -> PyResult<bool> { Ok(self.pos.is_stalemate()) }

    // ------------------------------------------------------------------------------------------------------------ plan 72.7 step 1: the words on the handle (RUST_BOUNDARY 3.2)

    /// `graph.word`: the compiled term `term` (an id of `compile_term`, rendered from the mover's side) read with `white` as the MOVER -- the board given to that
    /// side first, en passant cleared (`graph.with_turn`: the evaluator reads from the side to move and reads 0 / mirrored squares otherwise) -- as `vocab_eval`
    /// reads it (a number; a set term as its count). None when shakmaty does not hold the board with that side to move (the FEN reader answered None there).
    fn word(&self, term: u32, white: bool) -> PyResult<Option<u32>> {
        let c = crate::vocab::compiled(term)?;
        Ok(with_turn_strict(&self.pos, colour(white)).map(|p| crate::vocab::read_word(&p, term, &c, true)))
    }

    /// Many words on ONE position in a single crossing (`word` for each id, the board given to `white` once).
    fn words(&self, terms: Vec<u32>, white: bool) -> PyResult<Vec<Option<u32>>> {
        let cs = terms.iter().map(|&t| crate::vocab::compiled(t)).collect::<PyResult<Vec<_>>>()?;
        Ok(match with_turn_strict(&self.pos, colour(white)) {
            Some(p) => terms.iter().zip(cs.iter()).map(|(&t, c)| Some(crate::vocab::read_word(&p, t, c, true))).collect(),
            None => vec![None; terms.len()],
        })
    }

    /// `selfmine.val_at`: the compiled term read on this board AS IT STANDS (its side to move the mover), as `vocab_eval` reads it.
    fn read(&self, term: u32) -> PyResult<u32> {
        let c = crate::vocab::compiled(term)?;
        Ok(crate::vocab::read_word(&self.pos, term, &c, true))
    }

    /// `selfmine.evaluate` (`vocab_values`): the compiled number terms on this board as it stands, in one crossing (a set term reads 0, as there).
    fn values(&self, terms: Vec<u32>) -> PyResult<Vec<u32>> {
        let cs = terms.iter().map(|&t| crate::vocab::compiled(t)).collect::<PyResult<Vec<_>>>()?;
        Ok(terms.iter().zip(cs.iter()).map(|(&t, c)| crate::vocab::read_word(&self.pos, t, c, false)).collect())
    }

    /// `graph.see`: the word `(see sq)` read with the mover = the side to move (vocab.rs Term::See: what the side opposite the piece's owner wins by the exchange
    /// started by a capture on `sq`, rules::see, the mirror of Lean Ideas.ExchangeLaw.seeXB; 0 below zero and on an empty square).
    fn see(&self, sq: u8) -> PyResult<u32> {
        let s = square(sq)?;
        let us = self.pos.turn();
        let named = if us == Color::White { s } else { s.flip_vertical() };          // the square named from the mover's side, as `render` names it
        Ok(crate::vocab::num(&crate::vocab::Term::See(Box::new(crate::vocab::Term::Named(named))), &self.pos, !us))
    }
}

/// The squares whose piece or attacker counts (either colour) differ between two positions.
pub(crate) fn pos_diff(pa: &Chess, pb: &Chess) -> Vec<u8> {
    let (a, b) = (pa.board(), pb.board());
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

/// The best static exchange the side to move has on a piece of the other side; 0 when none.
pub(crate) fn pos_exchange(p: &Chess) -> i64 {
    let b = p.board();
    let us = p.turn();
    let mut best = 0;
    for s in b.by_color(!us) {
        if b.role_at(s) == Some(Role::King) || b.attacks_to(s, us, b.occupied()).is_empty() { continue; }
        best = best.max(crate::rules::see(b, s, us));
    }
    best
}

/// The side to move has a mate in one.
pub(crate) fn pos_mate_in_one(p: &Chess) -> bool { p.legal_moves().iter().any(|m| { let mut r = p.clone(); r.play_unchecked(m); r.is_checkmate() }) }
