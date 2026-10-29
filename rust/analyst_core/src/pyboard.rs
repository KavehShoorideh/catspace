//! PLAN 76.4b THE BOARD AS PYTHON-CHESS HOLDS IT: the regression's states are python-chess boards -- counterfactuals among them (a unit placed, the side to move
//! given away, the side NOT to move possibly in check, a king missing), boards shakmaty's `Chess` refuses -- and the binding of 76.4a read them through
//! python-chess (its legal moves in its order, its attacks, its `push`). So the Rust binding reads a board by python-chess 1.11's rules, mirrored here line by line
//! (`generate_legal_moves` with `_slider_blockers`, `_generate_evasions`, `_is_safe`, `pin_mask`, `_ep_skewered`, `generate_castling_moves` with
//! `clean_castling_rights`; `push` with `_to_chess960`, the castling rights, the e.p. square set on every double push, the clocks). RUST_BOUNDARY risk 4: the
//! order of the moves is python-chess's (the regression's witnesses inherit it). Standard chess only (chess960 off, no promoted mask: a king is never promoted).
use shakmaty::{attacks, Bitboard, Color, Square};

pub const PAWN: u8 = 1;
pub const KNIGHT: u8 = 2;
pub const BISHOP: u8 = 3;
pub const ROOK: u8 = 4;
pub const QUEEN: u8 = 5;
pub const KING: u8 = 6;

const RANK_1: u64 = 0xFF;
const RANK_8: u64 = 0xFF << 56;
const BACK: u64 = RANK_1 | RANK_8;
const RANK_3: u64 = 0xFF << 16;
const RANK_4: u64 = 0xFF << 24;
const RANK_5: u64 = 0xFF << 32;
const RANK_6: u64 = 0xFF << 40;
const ALL: u64 = u64::MAX;
const A1: u8 = 0;
const C1: u8 = 2;
const D1: u8 = 3;
const E1: u8 = 4;
const F1: u8 = 5;
const G1: u8 = 6;
const H1: u8 = 7;
const A8: u8 = 56;
const C8: u8 = 58;
const D8: u8 = 59;
const E8: u8 = 60;
const F8: u8 = 61;
const G8: u8 = 62;
const H8: u8 = 63;

#[inline] fn bb(s: u8) -> u64 { 1u64 << s }
#[inline] fn sqr(s: u8) -> Square { Square::new(s as u32) }
#[inline] fn msb(b: u64) -> u8 { 63 - b.leading_zeros() as u8 }
#[inline] fn lsb(b: u64) -> u8 { b.trailing_zeros() as u8 }
fn rank_of(s: u8) -> u8 { s >> 3 }
fn file_of(s: u8) -> u8 { s & 7 }
fn col(white: bool) -> Color { if white { Color::White } else { Color::Black } }

/// Squares in python-chess's `scan_reversed` order (descending).
fn rev(mut b: u64) -> impl Iterator<Item = u8> {
    std::iter::from_fn(move || { if b == 0 { None } else { let s = msb(b); b ^= bb(s); Some(s) } })
}
/// Squares in `scan_forward` order (ascending: a SquareSet's iteration).
pub fn fwd(mut b: u64) -> impl Iterator<Item = u8> {
    std::iter::from_fn(move || { if b == 0 { None } else { let s = lsb(b); b &= b - 1; Some(s) } })
}

fn ray(a: u8, b: u8) -> u64 { attacks::ray(sqr(a), sqr(b)).0 }
fn between(a: u8, b: u8) -> u64 { attacks::between(sqr(a), sqr(b)).0 }
fn rook_att(s: u8, occ: u64) -> u64 { attacks::rook_attacks(sqr(s), Bitboard(occ)).0 }
fn bishop_att(s: u8, occ: u64) -> u64 { attacks::bishop_attacks(sqr(s), Bitboard(occ)).0 }
fn knight_att(s: u8) -> u64 { attacks::knight_attacks(sqr(s)).0 }
fn king_att(s: u8) -> u64 { attacks::king_attacks(sqr(s)).0 }
fn pawn_att(white: bool, s: u8) -> u64 { attacks::pawn_attacks(col(white), sqr(s)).0 }
fn rank_mask(s: u8) -> u64 { RANK_1 << (8 * rank_of(s)) }
fn file_mask(s: u8) -> u64 { 0x0101_0101_0101_0101u64 << file_of(s) }

/// A move as python-chess holds it: from, to, promotion (a piece type 2..5).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PMove { pub from: u8, pub to: u8, pub promo: u8 }
impl PMove {
    pub fn new(from: u8, to: u8, promo: u8) -> PMove { PMove { from, to, promo } }
    pub fn uci(&self) -> String {
        let n = |s: u8| format!("{}{}", (b'a' + file_of(s)) as char, rank_of(s) + 1);
        let p = match self.promo { KNIGHT => "n", BISHOP => "b", ROOK => "r", QUEEN => "q", _ => "" };
        format!("{}{}{}", n(self.from), n(self.to), p)
    }
}

/// A python-chess board's fields (chess.Board with its move stack irrelevant: `clean_castling_rights` is computed -- after a push the raw rights already are clean).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PBoard {
    pub roles: [u64; 6],          // pawns, knights, bishops, rooks, queens, kings
    pub white: u64,
    pub black: u64,
    pub turn: bool,               // white to move
    pub castling: u64,            // castling_rights (raw: the rooks' squares)
    pub ep: Option<u8>,           // ep_square (raw: set after every double push)
    pub halfmove: u32,
    pub fullmove: u32,
}

impl PBoard {
    pub fn occupied(&self) -> u64 { self.white | self.black }
    pub fn co(&self, white: bool) -> u64 { if white { self.white } else { self.black } }
    fn co_mut(&mut self, white: bool) -> &mut u64 { if white { &mut self.white } else { &mut self.black } }
    pub fn pawns(&self) -> u64 { self.roles[0] }
    pub fn knights(&self) -> u64 { self.roles[1] }
    pub fn bishops(&self) -> u64 { self.roles[2] }
    pub fn rooks(&self) -> u64 { self.roles[3] }
    pub fn queens(&self) -> u64 { self.roles[4] }
    pub fn kings(&self) -> u64 { self.roles[5] }

    pub fn piece_type_at(&self, s: u8) -> u8 {
        let m = bb(s);
        for (i, r) in self.roles.iter().enumerate() { if r & m != 0 { return i as u8 + 1; } }
        0
    }
    /// The piece on `s`: (type, white) or None.
    pub fn piece_at(&self, s: u8) -> Option<(u8, bool)> {
        let t = self.piece_type_at(s);
        if t == 0 { None } else { Some((t, self.white & bb(s) != 0)) }
    }

    pub fn attacks_mask(&self, s: u8) -> u64 {
        let m = bb(s);
        if m & self.pawns() != 0 { return pawn_att(m & self.white != 0, s); }
        if m & self.knights() != 0 { return knight_att(s); }
        if m & self.kings() != 0 { return king_att(s); }
        let mut a = 0;
        if m & self.bishops() != 0 || m & self.queens() != 0 { a = bishop_att(s, self.occupied()); }
        if m & self.rooks() != 0 || m & self.queens() != 0 { a |= rook_att(s, self.occupied()); }
        a
    }

    pub fn attackers_mask(&self, white: bool, s: u8, occupied: u64) -> u64 {
        let qr = self.queens() | self.rooks();
        let qb = self.queens() | self.bishops();
        let a = (king_att(s) & self.kings()) | (knight_att(s) & self.knights()) | (rook_att(s, occupied) & qr) | (bishop_att(s, occupied) & qb)
            | (pawn_att(!white, s) & self.pawns());
        a & self.co(white)
    }
    pub fn is_attacked_by(&self, white: bool, s: u8) -> bool { self.attackers_mask(white, s, self.occupied()) != 0 }

    pub fn king(&self, white: bool) -> Option<u8> { let k = self.co(white) & self.kings(); if k == 0 { None } else { Some(msb(k)) } }

    pub fn clean_castling_rights(&self) -> u64 {
        let castling = self.castling & self.rooks();
        let mut w = castling & RANK_1 & self.white;
        let mut b = castling & RANK_8 & self.black;
        w &= bb(A1) | bb(H1);
        b &= bb(A8) | bb(H8);
        if self.white & self.kings() & bb(E1) == 0 { w = 0; }
        if self.black & self.kings() & bb(E8) == 0 { b = 0; }
        w | b
    }

    // ---------------------------------------------------------------------------------------------- the moves (python-chess's generators)

    fn pseudo(&self, from_mask: u64, to_mask: u64, out: &mut Vec<PMove>) {
        let our = self.co(self.turn);
        for from in rev(our & !self.pawns() & from_mask) {
            for to in rev(self.attacks_mask(from) & !our & to_mask) { out.push(PMove::new(from, to, 0)); }
        }
        if from_mask & self.kings() != 0 { self.castling_moves(from_mask, to_mask, out); }
        let pawns = self.pawns() & our & from_mask;
        if pawns == 0 { return; }
        for from in rev(pawns) {
            for to in rev(pawn_att(self.turn, from) & self.co(!self.turn) & to_mask) {
                if bb(to) & BACK != 0 { for p in [QUEEN, ROOK, BISHOP, KNIGHT] { out.push(PMove::new(from, to, p)); } } else { out.push(PMove::new(from, to, 0)); }
            }
        }
        let occ = self.occupied();
        let (mut single, mut double) = if self.turn {
            let s = (pawns << 8) & !occ;
            (s, (s << 8) & !occ & (RANK_3 | RANK_4))
        } else {
            let s = (pawns >> 8) & !occ;
            (s, (s >> 8) & !occ & (RANK_6 | RANK_5))
        };
        single &= to_mask;
        double &= to_mask;
        for to in rev(single) {
            let from = if self.turn { to - 8 } else { to + 8 };
            if bb(to) & BACK != 0 { for p in [QUEEN, ROOK, BISHOP, KNIGHT] { out.push(PMove::new(from, to, p)); } } else { out.push(PMove::new(from, to, 0)); }
        }
        for to in rev(double) {
            let from = if self.turn { to - 16 } else { to + 16 };
            out.push(PMove::new(from, to, 0));
        }
        if self.ep.is_some_and(|e| e != 0) { self.pseudo_ep(from_mask, to_mask, out); }
    }

    fn pseudo_ep(&self, from_mask: u64, to_mask: u64, out: &mut Vec<PMove>) {
        let Some(ep) = self.ep else { return };
        if ep == 0 || bb(ep) & to_mask == 0 { return; }
        if bb(ep) & self.occupied() != 0 { return; }
        let rank = if self.turn { 0xFFu64 << 32 } else { 0xFFu64 << 24 };
        let capturers = self.pawns() & self.co(self.turn) & from_mask & pawn_att(!self.turn, ep) & rank;
        for c in rev(capturers) { out.push(PMove::new(c, ep, 0)); }
    }

    fn attacked_for_king(&self, path: u64, occupied: u64) -> bool { rev(path).any(|s| self.attackers_mask(!self.turn, s, occupied) != 0) }

    fn castling_moves(&self, from_mask: u64, to_mask: u64, out: &mut Vec<PMove>) {
        let backrank = if self.turn { RANK_1 } else { RANK_8 };
        let mut king = self.co(self.turn) & self.kings() & backrank & from_mask;
        king &= king.wrapping_neg();
        if king == 0 { return; }
        let ksq = msb(king);
        let (bc, bd, bf, bg) = if self.turn { (C1, D1, F1, G1) } else { (C8, D8, F8, G8) };
        for cand in rev(self.clean_castling_rights() & backrank & to_mask) {
            let rook = bb(cand);
            let a_side = rook < king;
            let king_to = if a_side { bc } else { bg };
            let rook_to = if a_side { bd } else { bf };
            let king_path = between(ksq, king_to);
            let rook_path = between(cand, rook_to);
            let occ = self.occupied();
            if !(((occ ^ king ^ rook) & (king_path | rook_path | bb(king_to) | bb(rook_to))) != 0
                || self.attacked_for_king(king_path | king, occ ^ king)
                || self.attacked_for_king(bb(king_to), occ ^ king ^ rook ^ bb(rook_to))) {
                out.push(self.from_chess960(ksq, cand));
            }
        }
    }

    fn from_chess960(&self, from: u8, to: u8) -> PMove {
        if from == E1 && self.kings() & bb(E1) != 0 {
            if to == H1 { return PMove::new(E1, G1, 0); } else if to == A1 { return PMove::new(E1, C1, 0); }
        } else if from == E8 && self.kings() & bb(E8) != 0 {
            if to == H8 { return PMove::new(E8, G8, 0); } else if to == A8 { return PMove::new(E8, C8, 0); }
        }
        PMove::new(from, to, 0)
    }

    fn to_chess960(&self, m: PMove) -> PMove {
        if m.from == E1 && self.kings() & bb(E1) != 0 {
            if m.to == G1 && self.rooks() & bb(G1) == 0 { return PMove::new(E1, H1, m.promo); }
            if m.to == C1 && self.rooks() & bb(C1) == 0 { return PMove::new(E1, A1, m.promo); }
        } else if m.from == E8 && self.kings() & bb(E8) != 0 {
            if m.to == G8 && self.rooks() & bb(G8) == 0 { return PMove::new(E8, H8, m.promo); }
            if m.to == C8 && self.rooks() & bb(C8) == 0 { return PMove::new(E8, A8, m.promo); }
        }
        m
    }

    fn slider_blockers(&self, king: u8) -> u64 {
        let snipers = (rook_att(king, 0) & (self.rooks() | self.queens())) | (bishop_att(king, 0) & (self.bishops() | self.queens()));
        let mut blockers = 0;
        for s in rev(snipers & self.co(!self.turn)) {
            let b = between(king, s) & self.occupied();
            if b != 0 && b.count_ones() == 1 { blockers |= b; }
        }
        blockers & self.co(self.turn)
    }

    pub fn pin_mask(&self, white: bool, s: u8) -> u64 {
        let Some(king) = self.king(white) else { return ALL };
        let m = bb(s);
        let fam: [(u64, u64); 3] = [(file_mask(king) & !bb(king), self.rooks() | self.queens()), (rank_mask(king) & !bb(king), self.rooks() | self.queens()),
                                    (bishop_att(king, 0), self.bishops() | self.queens())];
        for (rays, sliders) in fam {
            if rays & m != 0 {
                for sniper in rev(rays & sliders & self.co(!white)) {
                    if between(sniper, king) & (self.occupied() | m) == m { return ray(king, sniper); }
                }
                break;
            }
        }
        ALL
    }

    fn ep_skewered(&self, king: u8, capturer: u8) -> bool {
        let ep = self.ep.unwrap();
        let last_double = if self.turn { ep - 8 } else { ep + 8 };
        let occupancy = (self.occupied() & !bb(last_double) & !bb(capturer)) | bb(ep);
        let horizontal = self.co(!self.turn) & (self.rooks() | self.queens());
        if rook_att(king, occupancy) & rank_mask(king) & horizontal != 0 { return true; }
        let diagonal = self.co(!self.turn) & (self.bishops() | self.queens());
        bishop_att(king, occupancy) & diagonal != 0
    }

    pub fn is_castling(&self, m: PMove) -> bool {
        if self.kings() & bb(m.from) != 0 {
            let diff = file_of(m.from) as i32 - file_of(m.to) as i32;
            return diff.abs() > 1 || self.rooks() & self.co(self.turn) & bb(m.to) != 0;
        }
        false
    }

    pub fn is_en_passant(&self, m: PMove) -> bool {
        self.ep == Some(m.to) && self.pawns() & bb(m.from) != 0 && matches!((m.to as i32 - m.from as i32).abs(), 7 | 9) && self.occupied() & bb(m.to) == 0
    }

    fn is_safe(&self, king: u8, blockers: u64, m: PMove) -> bool {
        if m.from == king {
            if self.is_castling(m) { return true; }
            return !self.is_attacked_by(!self.turn, m.to);
        }
        if self.is_en_passant(m) {
            return self.pin_mask(self.turn, m.from) & bb(m.to) != 0 && !self.ep_skewered(king, m.from);
        }
        blockers & bb(m.from) == 0 || ray(m.from, m.to) & bb(king) != 0
    }

    fn evasions(&self, king: u8, checkers: u64, from_mask: u64, to_mask: u64, out: &mut Vec<PMove>) {
        let sliders = checkers & (self.bishops() | self.rooks() | self.queens());
        let mut attacked = 0;
        for c in rev(sliders) { attacked |= ray(king, c) & !bb(c); }
        if bb(king) & from_mask != 0 {
            for to in rev(king_att(king) & !self.co(self.turn) & !attacked & to_mask) { out.push(PMove::new(king, to, 0)); }
        }
        let checker = msb(checkers);
        if bb(checker) == checkers {
            let target = between(king, checker) | checkers;
            self.pseudo(!self.kings() & from_mask, target & to_mask, out);
            if let Some(ep) = self.ep {
                if ep != 0 && bb(ep) & target == 0 {
                    let last_double = if self.turn { ep as i32 - 8 } else { ep as i32 + 8 };
                    if last_double == checker as i32 { self.pseudo_ep(from_mask, to_mask, out); }
                }
            }
        }
    }

    /// python-chess `generate_legal_moves()`, in its order.
    pub fn legal_moves(&self) -> Vec<PMove> { self.legal_masked(ALL, ALL) }

    fn legal_masked(&self, from_mask: u64, to_mask: u64) -> Vec<PMove> {
        let mut gen = vec![];
        let km = self.kings() & self.co(self.turn);
        if km == 0 { self.pseudo(from_mask, to_mask, &mut gen); return gen; }
        let king = msb(km);
        let blockers = self.slider_blockers(king);
        let checkers = self.attackers_mask(!self.turn, king, self.occupied());
        if checkers != 0 { self.evasions(king, checkers, from_mask, to_mask, &mut gen); } else { self.pseudo(from_mask, to_mask, &mut gen); }
        gen.retain(|m| self.is_safe(king, blockers, *m));
        gen
    }

    fn is_into_check(&self, m: PMove) -> bool {
        let Some(king) = self.king(self.turn) else { return false };
        let checkers = self.attackers_mask(!self.turn, king, self.occupied());
        if checkers != 0 {
            let mut ev = vec![];
            self.evasions(king, checkers, bb(m.from), bb(m.to), &mut ev);
            if !ev.contains(&m) { return true; }
        }
        !self.is_safe(king, self.slider_blockers(king), m)
    }

    /// python-chess `has_legal_en_passant()`.
    pub fn has_legal_en_passant(&self) -> bool {
        if self.ep.is_none() { return false; }
        let mut v = vec![];
        self.pseudo_ep(ALL, ALL, &mut v);
        v.into_iter().any(|m| !self.is_into_check(m))
    }

    /// python-chess `is_pseudo_legal(move)` (no drops, no null move).
    pub fn is_pseudo_legal(&self, m: PMove) -> bool {
        let piece = self.piece_type_at(m.from);
        if piece == 0 { return false; }
        let (fm, tm) = (bb(m.from), bb(m.to));
        if self.co(self.turn) & fm == 0 { return false; }
        if m.promo != 0 {
            if piece != PAWN { return false; }
            if self.turn && rank_of(m.to) != 7 { return false; }
            if !self.turn && rank_of(m.to) != 0 { return false; }
        }
        if piece == KING {
            let m2 = self.from_chess960(m.from, m.to);
            let mut cs = vec![];
            self.castling_moves(ALL, ALL, &mut cs);
            if cs.contains(&m2) { return true; }
        }
        if self.co(self.turn) & tm != 0 { return false; }
        if piece == PAWN {
            let mut v = vec![];
            self.pseudo(fm, tm, &mut v);
            return v.contains(&m);
        }
        self.attacks_mask(m.from) & tm != 0
    }

    // ---------------------------------------------------------------------------------------------- the board changed

    pub fn remove_piece_at(&mut self, s: u8) -> u8 {
        let t = self.piece_type_at(s);
        if t == 0 { return 0; }
        let m = bb(s);
        self.roles[t as usize - 1] ^= m;
        self.white &= !m;
        self.black &= !m;
        t
    }
    pub fn set_piece_at(&mut self, s: u8, t: u8, white: bool) {
        self.remove_piece_at(s);
        if t == 0 { return; }
        let m = bb(s);
        self.roles[t as usize - 1] |= m;
        *self.co_mut(white) |= m;
    }

    /// python-chess `push(move)` (the move pseudo-legal; castling written as the king's two-file move or as the king onto its rook).
    pub fn push(&mut self, m: PMove) {
        let m = self.to_chess960(m);
        self.castling = self.clean_castling_rights();
        let ep_square = self.ep;
        self.ep = None;
        self.halfmove += 1;
        if !self.turn { self.fullmove += 1; }
        let touched = bb(m.from) ^ bb(m.to);
        if touched & self.pawns() != 0 || touched & self.co(!self.turn) != 0 { self.halfmove = 0; }
        let (from_bb, to_bb) = (bb(m.from), bb(m.to));
        let mut piece_type = self.remove_piece_at(m.from);
        let mut captured = self.piece_type_at(m.to);
        self.castling &= !to_bb & !from_bb;
        if piece_type == KING {
            self.castling &= if self.turn { !RANK_1 } else { !RANK_8 };
        } else if captured == KING {
            if self.turn && rank_of(m.to) == 7 { self.castling &= !RANK_8; } else if !self.turn && rank_of(m.to) == 0 { self.castling &= !RANK_1; }
        }
        if piece_type == PAWN {
            let diff = m.to as i32 - m.from as i32;
            if diff == 16 && rank_of(m.from) == 1 { self.ep = Some(m.from + 8); }
            else if diff == -16 && rank_of(m.from) == 6 { self.ep = Some(m.from - 8); }
            else if Some(m.to) == ep_square && (diff.abs() == 7 || diff.abs() == 9) && captured == 0 {
                let e = ep_square.unwrap() as i32;
                let cs = if self.turn { e - 8 } else { e + 8 };
                captured = self.remove_piece_at(cs as u8);
            }
        }
        let _ = captured;
        if m.promo != 0 { piece_type = m.promo; }
        let castling = piece_type == KING && self.co(self.turn) & to_bb != 0;
        if castling {
            let a_side = file_of(m.to) < file_of(m.from);
            self.remove_piece_at(m.from);
            self.remove_piece_at(m.to);
            let t = self.turn;
            if a_side {
                self.set_piece_at(if t { C1 } else { C8 }, KING, t);
                self.set_piece_at(if t { D1 } else { D8 }, ROOK, t);
            } else {
                self.set_piece_at(if t { G1 } else { G8 }, KING, t);
                self.set_piece_at(if t { F1 } else { F8 }, ROOK, t);
            }
        } else {
            let t = self.turn;
            self.set_piece_at(m.to, piece_type, t);
        }
        self.turn = !self.turn;
    }

    /// graph.with_turn / the binding's `to_move`: a copy with `white` to move, the e.p. square dropped (always a copy in python-chess; the fields are the board's).
    pub fn with_turn(&self, white: bool) -> PBoard { let mut c = *self; c.turn = white; c.ep = None; c }
}

/// shakmaty's position of a board read as graph.pos_of reads it (`Pos.from_parts`: the bitboards, the side to move, the CLEAN castling rights, the e.p. square only
/// when a legal en passant exists, the clocks) -- None where shakmaty does not hold it (graph.pos_of answers None there).
pub fn to_chess(b: &PBoard) -> Option<shakmaty::Chess> {
    use shakmaty::{CastlingMode, FromSetup};
    let roles = b.roles;
    let union = roles.iter().fold(0u64, |a, r| a | r);
    if roles.iter().map(|r| r.count_ones()).sum::<u32>() != union.count_ones() || union != (b.white | b.black) || b.white & b.black != 0 { return None; }
    let board = shakmaty::Board::from_bitboards(
        shakmaty::ByRole { pawn: Bitboard(roles[0]), knight: Bitboard(roles[1]), bishop: Bitboard(roles[2]), rook: Bitboard(roles[3]), queen: Bitboard(roles[4]), king: Bitboard(roles[5]) },
        shakmaty::ByColor { white: Bitboard(b.white), black: Bitboard(b.black) });
    let mut s = shakmaty::Setup::empty();
    s.board = board;
    s.turn = col(b.turn);
    s.castling_rights = Bitboard(b.clean_castling_rights());
    s.ep_square = match b.ep { Some(e) if b.has_legal_en_passant() => Some(sqr(e)), _ => None };
    s.halfmoves = b.halfmove;
    s.fullmoves = std::num::NonZeroU32::new(b.fullmove)?;
    shakmaty::Chess::from_setup(s, CastlingMode::Standard).ok()
}
