//! THE BASE VOCABULARY, EVALUATED (Kaveh 2026-10-12: "the base has to be deterministic and interpretable"). A term over
//! the base -- each primitive the mirror of a definition in Lean `Ideas.Vocabulary` -- read on a board, and THE
//! CERTIFIER: how many stuck positions a frame's measure, extended by a candidate term, certifies (progress against every
//! reply). The proposer (catspace/incubator/analyst/proposer.py) builds the terms and reads the winners back in words.
//!
//! Terms are S-expressions. Squares: W (our king), L (the lone king), P (our other piece), O (their other piece -- each
//! the first in the board's square order, a1 b1 ... h8). Sets: edge, corner, attacked (the squares we attack, their king
//! lifted), free (the rest), hit (the squares they attack, our king lifted), unhit (the rest), ours / theirs (the squares
//! of each side's pieces other than the king), (box S), (rank S), (file S), (line S S), (between S S), (nearer S S),
//! (and A B), (or A B), (not A), (reach S A). Numbers: (count A), (dist S S), (sum N N). (Kaveh 2026-10-12: "base vocab
//! should be complete with words for their pieces" -- O, hit, unhit, ours, theirs.)
use super::*;
use shakmaty::Bitboard;

#[derive(Clone, Debug)]
pub enum Term {
    W, L, P, O,
    Edge, Corner, Attacked, Free, Hit, Unhit, Ours, Theirs,
    Box1(Box<Term>), Rank(Box<Term>), File(Box<Term>),
    Line(Box<Term>, Box<Term>), Between(Box<Term>, Box<Term>), Nearer(Box<Term>, Box<Term>),
    And(Box<Term>, Box<Term>), Or(Box<Term>, Box<Term>), Not(Box<Term>), Reach(Box<Term>, Box<Term>),
    Count(Box<Term>), Dist(Box<Term>, Box<Term>), Sum(Box<Term>, Box<Term>),
    // the second batch (Kaveh 2026-10-12): geometry -- a rook's and a bishop's lines, a segment, a knight's one and two
    // hops, the centre, a named square up to the board's symmetry -- and per piece: what the piece on a square attacks
    Lines(Box<Term>), Diagonals(Box<Term>), Segment(Box<Term>, Box<Term>), Knight1(Box<Term>), Knight2(Box<Term>), Center,
    Attacks(Box<Term>), Cls(Square),
    // a square's colour (Kaveh 2026-10-13: "go on both" -- the light and dark squares; the squares of a square's colour:
    // K+B+N v K's corner of the bishop's colour, which no word could say)
    Light, Dark, Colour(Box<Term>),
    // every square by its name, from our side, and the square itself as a set (Kaveh 2026-10-13: "a mirroring lemma
    // separately, and then ... the squares individually available")
    Named(Square), At(Box<Term>),
    // RELATIVE GEOMETRY (Kaveh 2026-10-13: "whatever it took for you to find that these are the right lemmas ... needs to
    // become machinery ... the vocabulary needs to be complemented"): a piece's file and rank counted from our side, and
    // how many files / ranks one piece stands from another (plus 7, so 7 is level) -- an arrangement of the pieces, its
    // shift across the board, "ahead of", "the opposition" become compositions
    FileOf(Box<Term>), RankOf(Box<Term>), DFile(Box<Term>, Box<Term>), DRank(Box<Term>, Box<Term>),
    // THE VOCABULARY COMPLETED (Kaveh 2026-10-13: "Yes, please add all of these ... compositional vocab"):
    // every piece named (strongest first, pawns the most advanced first) and every kind as a set
    Piece(bool, u8), Kind(bool, Role),
    // a pawn's direction (its owner's): the squares ahead, the square in front, the promotion square, the ranks to go,
    // the passed-pawn zone
    Ahead(Box<Term>), Front(Box<Term>), Promo(Box<Term>), Pushes(Box<Term>), Passer(Box<Term>),
    // a literal number; the square at an offset from a piece (from our side)
    Int(i32), Offset(Box<Term>, i32, i32),
    // lines through pieces: the squares past Y on the ray from X, the first piece there, what X attacks through one piece
    Beyond(Box<Term>, Box<Term>), Next(Box<Term>, Box<Term>), Xray(Box<Term>),
    // control and exchanges: how many of ours / theirs attack a square, the exchange on it, a piece's legal moves
    Attackers(Box<Term>), Defenders(Box<Term>), See(Box<Term>), Moves(Box<Term>), Worth(Box<Term>), MaxWorth(Box<Term>),
    // the short window's tactics (exact): the fewest moves forcing a gain of v or more (ours; theirs as if we passed)
    Forcing(bool, i64),
    // material and the game's context; time
    MaterialUs, MaterialThem, Balance, Halfmoves, Seen, Ago(u32, Box<Term>),
    // a number read on the board with the non-king pieces on a set's squares taken off (Kaveh 2026-10-14: "what each piece's job is")
    Without(Box<Term>, Box<Term>),
    // THE LANGUAGE OF CHANGE AND OF CONDITIONALS (Kaveh 2026-10-14): a branch, the board after a move, a change since the last ply
    If(Box<Term>, Box<Term>, Box<Term>), After(Box<Term>, Box<Term>, Box<Term>), Least(Box<Term>), Most(Box<Term>), Delta(Box<Term>), Gained(Box<Term>), Lost(Box<Term>),
    // number combinators
    Min(Box<Term>, Box<Term>), Max(Box<Term>, Box<Term>), Diff(Box<Term>, Box<Term>), Eq(Box<Term>, Box<Term>),
    // MOBILITY and ATTACKABLE (Kaveh 2026-10-15, ANALYST_PLAN 40.2): the escape squares of a piece (legal, safe by the exchange); whether the other
    // side can attack a piece's square within k of its moves, every landing square safe by the exchange
    Mobility(Box<Term>), Attackable(Box<Term>, u32), Declined(Box<Term>),
}

/// THE PIECES OF A SIDE, RANKED -- the order the names P, P2, P3, ... (O, O2, ...) follow: the strongest kind first (queen,
/// rook, bishop, knight, pawn), pawns the most advanced first, then the lower square.
pub fn ranked(b: &shakmaty::Board, c: Color) -> Vec<Square> {
    let kr = |r: Role| match r { Role::Queen => 5, Role::Rook => 4, Role::Bishop => 3, Role::Knight => 2, Role::Pawn => 1, Role::King => 0 };
    let adv = |s: Square| if c == Color::White { s.rank() as u32 } else { 7 - s.rank() as u32 };
    let mut v: Vec<Square> = (b.by_color(c) & !b.kings()).into_iter().collect();
    v.sort_by_key(|s| { let r = b.role_at(*s).unwrap_or(Role::King);
        (std::cmp::Reverse(kr(r)), std::cmp::Reverse(if r == Role::Pawn { adv(*s) } else { 0 }), *s as u8) });
    v
}

/// THE COMPOSITIONS (Kaveh 2026-10-13: "regular chess terms like pin ... in terms of simpler vocabulary so that we can
/// have compositional vocab"): name -> (parameters, body over the base words); a term naming one is expanded before
/// it is read, so every chess term unfolds into the base (set from Python: discovery/base_vocabulary.py CHESS_TERMS).
static MACROS: std::sync::RwLock<Option<std::collections::HashMap<String, (Vec<String>, String)>>> = std::sync::RwLock::new(None);
#[pyfunction]
pub fn set_macros(macros: Vec<(String, Vec<String>, String)>) -> PyResult<()> {
    *MACROS.write().unwrap() = Some(macros.into_iter().map(|(n, ps, b)| (n, (ps, b))).collect());
    Ok(())
}
#[derive(Clone, Debug)]
enum Sx { A(String), L(Vec<Sx>) }
fn sx_parse(t: &[String], i: &mut usize) -> Option<Sx> {
    let tok = t.get(*i)?.clone(); *i += 1;
    if tok != "(" { return Some(Sx::A(tok)); }
    let mut v = Vec::new();
    while t.get(*i)? != ")" { v.push(sx_parse(t, i)?); }
    *i += 1;
    Some(Sx::L(v))
}
fn sx_text(x: &Sx) -> String { match x { Sx::A(a) => a.clone(), Sx::L(v) => format!("({})", v.iter().map(sx_text).collect::<Vec<_>>().join(" ")) } }
fn sx_subst(x: &Sx, env: &std::collections::HashMap<String, Sx>) -> Sx {
    match x { Sx::A(a) => env.get(a).cloned().unwrap_or_else(|| x.clone()), Sx::L(v) => Sx::L(v.iter().map(|y| sx_subst(y, env)).collect()) }
}
fn sx_expand(x: &Sx, m: &std::collections::HashMap<String, (Vec<String>, String)>, depth: u32) -> Option<Sx> {
    if depth > 16 { return None; }
    match x {
        Sx::A(a) => match m.get(a) {                                      // a composition without parameters
            Some((ps, body)) if ps.is_empty() => { let t = toks(body); let mut i = 0; sx_expand(&sx_parse(&t, &mut i)?, m, depth + 1) }
            _ => Some(x.clone()),
        },
        Sx::L(v) => {
            let v: Vec<Sx> = v.iter().map(|y| sx_expand(y, m, depth)).collect::<Option<_>>()?;
            if let Some(Sx::A(h)) = v.first() {
                if let Some((ps, body)) = m.get(h) {
                    if ps.len() + 1 != v.len() { return None; }
                    let env: std::collections::HashMap<String, Sx> = ps.iter().cloned().zip(v[1..].iter().cloned()).collect();
                    let t = toks(body); let mut i = 0;
                    return sx_expand(&sx_subst(&sx_parse(&t, &mut i)?, &env), m, depth + 1);
                }
            }
            Some(Sx::L(v))
        }
    }
}
fn toks(s: &str) -> Vec<String> { s.replace('(', " ( ").replace(')', " ) ").split_whitespace().map(|x| x.to_string()).collect() }

/// The strongest piece of `c` other than its king -- by kind (queen, rook, bishop, knight, pawn), ties by the lower square.
fn strongest(b: &shakmaty::Board, c: Color) -> Option<Square> {
    let rank = |r: Role| match r { Role::Queen => 5, Role::Rook => 4, Role::Bishop => 3, Role::Knight => 2, Role::Pawn => 1, Role::King => 0 };
    (b.by_color(c) & !b.kings()).into_iter().max_by_key(|s| (b.role_at(*s).map(rank).unwrap_or(0), std::cmp::Reverse(*s as u8)))
}

/// Parse an S-expression term.
pub fn parse_term(s: &str) -> Option<Term> {
    let mut toks: Vec<String> = toks(s);
    if let Some(m) = MACROS.read().unwrap().as_ref() {
        if !m.is_empty() && toks.iter().any(|t| m.contains_key(t)) {
            let mut i = 0;
            let x = sx_expand(&sx_parse(&toks, &mut i)?, m, 0)?;
            toks = self::toks(&sx_text(&x));
        }
    }
    let mut i = 0;
    let t = parse_at(&toks, &mut i)?;
    (i == toks.len()).then_some(t)
}
fn parse_at(t: &[String], i: &mut usize) -> Option<Term> {
    let tok = t.get(*i)?.clone();
    *i += 1;
    if tok != "(" {
        return match tok.as_str() { "W" => Some(Term::W), "L" => Some(Term::L), "P" => Some(Term::P), "O" => Some(Term::O),
            "hit" => Some(Term::Hit), "unhit" => Some(Term::Unhit), "ours" => Some(Term::Ours), "theirs" => Some(Term::Theirs), "edge" => Some(Term::Edge),
            "corner" => Some(Term::Corner), "attacked" => Some(Term::Attacked), "free" => Some(Term::Free), "center" => Some(Term::Center),
            "light" => Some(Term::Light), "dark" => Some(Term::Dark),
            "material_us" => Some(Term::MaterialUs), "material_them" => Some(Term::MaterialThem), "balance" => Some(Term::Balance),
            "halfmoves" => Some(Term::Halfmoves), "seen" => Some(Term::Seen),
            _ => {
                let kinds = [("queens", Role::Queen), ("rooks", Role::Rook), ("bishops", Role::Bishop), ("knights", Role::Knight), ("pawns", Role::Pawn)];
                for (side, ours) in [("our", true), ("their", false)] {
                    for (k, r) in kinds { if tok == format!("{side}{k}") { return Some(Term::Kind(ours, r)); } }
                }
                if let Some(rest) = tok.strip_prefix("forcing_") {
                    let (side, v) = rest.split_once('_')?;
                    let v = if v == "mate" { crate::waiting::MATE_GOAL } else { v.parse::<i64>().ok()? };
                    return Some(Term::Forcing(side == "us", v));
                }
                if let (Some(c), Ok(n)) = (tok.chars().next(), tok[1..].parse::<u8>()) {
                    if (c == 'P' || c == 'O') && (2..=7).contains(&n) { return Some(Term::Piece(c == 'P', n)); }
                }
                if let Ok(n) = tok.parse::<i32>() { return Some(Term::Int(n)); }
                tok.parse::<Square>().ok().map(Term::Named)
            } };
    }
    // (cls <square>): a literal square name, named from our side
    if t.get(*i).map(|h| h.as_str()) == Some("cls") {
        let name = t.get(*i + 1)?.clone();
        if t.get(*i + 2)? != ")" { return None; }
        *i += 3;
        return name.parse::<Square>().ok().map(Term::Cls);
    }
    let head = t.get(*i)?.clone();
    *i += 1;
    let mut args = Vec::new();
    while t.get(*i)? != ")" { args.push(parse_at(t, i)?); }
    *i += 1;
    let b = |x: &Term| Box::new(x.clone());
    Some(match (head.as_str(), args.as_slice()) {
        ("box", [a]) => Term::Box1(b(a)), ("rank", [a]) => Term::Rank(b(a)), ("file", [a]) => Term::File(b(a)),
        ("line", [a, c]) => Term::Line(b(a), b(c)), ("between", [a, c]) => Term::Between(b(a), b(c)),
        ("nearer", [a, c]) => Term::Nearer(b(a), b(c)), ("and", [a, c]) => Term::And(b(a), b(c)), ("or", [a, c]) => Term::Or(b(a), b(c)),
        ("not", [a]) => Term::Not(b(a)), ("reach", [a, c]) => Term::Reach(b(a), b(c)), ("count", [a]) => Term::Count(b(a)),
        ("dist", [a, c]) => Term::Dist(b(a), b(c)), ("sum", [a, c]) => Term::Sum(b(a), b(c)),
        ("lines", [a]) => Term::Lines(b(a)), ("diagonals", [a]) => Term::Diagonals(b(a)), ("segment", [a, c]) => Term::Segment(b(a), b(c)),
        ("knight1", [a]) => Term::Knight1(b(a)), ("knight2", [a]) => Term::Knight2(b(a)), ("attacks", [a]) => Term::Attacks(b(a)),
        ("colour", [a]) => Term::Colour(b(a)), ("at", [a]) => Term::At(b(a)),
        ("fileof", [a]) => Term::FileOf(b(a)), ("rankof", [a]) => Term::RankOf(b(a)),
        ("dfile", [a, c]) => Term::DFile(b(a), b(c)), ("drank", [a, c]) => Term::DRank(b(a), b(c)),
        ("ahead", [a]) => Term::Ahead(b(a)), ("front", [a]) => Term::Front(b(a)), ("promo", [a]) => Term::Promo(b(a)),
        ("pushes", [a]) => Term::Pushes(b(a)), ("passer", [a]) => Term::Passer(b(a)),
        ("offset", [a, Term::Int(df), Term::Int(dr)]) => Term::Offset(b(a), *df, *dr),
        ("beyond", [a, c]) => Term::Beyond(b(a), b(c)), ("next", [a, c]) => Term::Next(b(a), b(c)), ("xray", [a]) => Term::Xray(b(a)),
        ("attackers", [a]) => Term::Attackers(b(a)), ("defenders", [a]) => Term::Defenders(b(a)), ("see", [a]) => Term::See(b(a)), ("worth", [a]) => Term::Worth(b(a)), ("maxworth", [a]) => Term::MaxWorth(b(a)),
        ("moves", [a]) => Term::Moves(b(a)),
        ("ago", [Term::Int(k), a]) if *k > 0 => Term::Ago(*k as u32, b(a)),
        ("without", [x, a]) => Term::Without(b(x), b(a)),
        ("if", [c, a, d]) => Term::If(b(c), b(a), b(d)),
        ("after", [f, t2, a]) => Term::After(b(f), b(t2), b(a)),
        ("least", [a]) => Term::Least(b(a)), ("most", [a]) => Term::Most(b(a)),
        ("delta", [a]) => Term::Delta(b(a)), ("gained", [a]) => Term::Gained(b(a)), ("lost", [a]) => Term::Lost(b(a)),
        ("mobility", [a]) => Term::Mobility(b(a)),
        ("attackable", [a, Term::Int(k)]) if *k > 0 => Term::Attackable(b(a), *k as u32),
        ("declined", [a]) => Term::Declined(b(a)),
        ("min", [a, c]) => Term::Min(b(a), b(c)), ("max", [a, c]) => Term::Max(b(a), b(c)),
        ("diff", [a, c]) => Term::Diff(b(a), b(c)), ("eq", [a, c]) => Term::Eq(b(a), b(c)),
        _ => return None,
    })
}

fn kd(a: Square, b: Square) -> u32 { a.distance(b) }

/// **without X T**: T read on the board with every NON-KING piece on X's squares taken off (kings are never removed); the side to
/// move, `lone` (the us/lone reading) and the other settings unchanged. THE CHOICES: (1) the names in X (P2, O, theirrooks ...) are
/// resolved on the ORIGINAL board, the names inside T (W, L, P, O, P2 ...) on the MODIFIED one, `ranked()` recomputed there -- "T if X
/// were not on the board" speaks of the board that is left; (2) taking a blocker off can leave a king attacked by the side NOT to
/// move, a position shakmaty's strict constructor refuses: it is built RELAXED (`ignore_impossible_check`, invalid castling rights
/// and en passant dropped), the words read the board as it stands; a board that still cannot be built reads 0.
fn without(x: &Term, a: &Term, q: &Chess, lone: Color) -> u32 {
    match without_pos(x, q, lone) { Some(p) => num(a, &p, lone), None => 0 }
}

/// The board of `without X`: None when nothing is taken off (the board as it is is the same) -- or an unbuildable one, see `without_board`.
pub(crate) fn without_pos(x: &Term, q: &Chess, lone: Color) -> Option<Chess> {
    let gone = set(x, q, lone) & !q.board().kings();
    if gone.is_empty() { return Some(q.clone()); }
    let mut setup = q.clone().into_setup(shakmaty::EnPassantMode::Legal);
    for s in gone { setup.board.discard_piece_at(s); }
    relaxed(setup)
}

/// The position of a setup built RELAXED: en passant and castling rights dropped, a king left attacked by the side not to move allowed
/// (a board with a king missing, or otherwise impossible, is None).
pub(crate) fn relaxed(mut setup: shakmaty::Setup) -> Option<Chess> {
    setup.ep_square = None;
    setup.castling_rights = Bitboard::EMPTY;
    let build = |st: shakmaty::Setup| Chess::from_setup(st, shakmaty::CastlingMode::Standard).or_else(|e| e.ignore_impossible_check());
    match build(setup.clone()) {
        Ok(p) => Some(p),
        Err(e) if e.kinds().contains(shakmaty::PositionErrorKinds::OPPOSITE_CHECK) => {
            // THE SIDE NOT TO MOVE IS IN CHECK (shakmaty cannot ignore this): the board is kept and the checked side is made the mover -- every
            // board word (attacks, see, between ...) reads the same; only the words that ask whose turn it is (moves, forcing) read the other side
            setup.turn = !setup.turn;
            build(setup).ok()
        }
        Err(_) => None,
    }
}

/// **delta T**: signed, carried on an offset (numbers are unsigned): T now - T one ply ago + DELTA_ZERO; DELTA_ZERO when there is no
/// earlier position in the game line.
const DELTA_ZERO: u32 = 64;

/// **after F T P**: T read on the board after the piece on square F moves to square P (whatever stood on P taken -- one's own piece
/// too: "after the bishop is gone, the queen recaptures there"; a pawn reaching the last rank becomes a queen). The mover is whoever
/// stands on F; the board is then THEIR opponent's turn; `lone` (who is "us") is unchanged. Legality is not asked: built relaxed.
/// Names in F and P resolve on the board as it is, names in T on the board after the move.
fn after_move(f: &Term, t2: &Term, a: &Term, q: &Chess, lone: Color) -> u32 {
    match after_pos(f, t2, q, lone) { Some(p) => num(a, &p, lone), None => 0 }
}

/// Where the moved piece lands: a castling move names the rook's square in shakmaty, the king lands two files over.
fn move_dest(m: &shakmaty::Move) -> Square {
    match m {
        shakmaty::Move::Castle { king, rook } => Square::from_coords(if rook.file() > king.file() { shakmaty::File::G } else { shakmaty::File::C }, king.rank()),
        _ => m.to(),
    }
}

pub(crate) fn after_pos(f: &Term, t2: &Term, q: &Chess, lone: Color) -> Option<Chess> {
    let (from, to) = (sq(f, q, lone)?, sq(t2, q, lone)?);
    let mut pc = q.board().piece_at(from)?;
    // as Lean afterMove: when the piece on `from` is the side to move's and from->to is legal, the kernel's move (rights, en passant, clocks as
    // the rules say; a promotion only to a queen); else the relaxed hand-built board below
    if let Some(m) = q.legal_moves().into_iter().find(|m| m.from() == Some(from) && move_dest(m) == to && m.promotion().map_or(true, |r| r == shakmaty::Role::Queen)) {
        return Some(after(q, &m));
    }
    let mut setup = q.clone().into_setup(shakmaty::EnPassantMode::Legal);
    setup.board.discard_piece_at(from);
    if pc.role == shakmaty::Role::Pawn && matches!(to.rank(), shakmaty::Rank::First | shakmaty::Rank::Eighth) { pc.role = shakmaty::Role::Queen; }
    setup.board.set_piece_at(to, pc);
    setup.turn = !pc.color;
    relaxed(setup)
}

/// A square term on the board (`lone` the lone king's colour).
pub(crate) fn sq(t: &Term, q: &Chess, lone: Color) -> Option<Square> {
    let b = q.board();
    match t {
        Term::W => b.king_of(!lone), Term::L => b.king_of(lone),
        // "our / their other piece": the STRONGEST other piece by kind (queen, rook, bishop, knight, pawn), ties by square
        // (2026-10-13: it was the first by square -- in K+B+N v K the bishop or the knight by where they stood, so "the
        // colour of P" meant nothing half the time)
        Term::P => ranked(b, !lone).first().copied(),
        Term::O => ranked(b, lone).first().copied(),
        Term::Piece(ours, n) => ranked(b, if *ours { !lone } else { lone }).get(*n as usize - 1).copied(),
        Term::Front(a) => { let x = sq(a, q, lone)?; let c = b.color_at(x)?;
            let r = x.rank() as i32 + if c == Color::White { 1 } else { -1 };
            (0..8).contains(&r).then(|| Square::from_coords(x.file(), shakmaty::Rank::new(r as u32))) }
        Term::Promo(a) => { let x = sq(a, q, lone)?; let c = b.color_at(x)?;
            Some(Square::from_coords(x.file(), if c == Color::White { shakmaty::Rank::Eighth } else { shakmaty::Rank::First })) }
        Term::Offset(a, df, dr) => { let x = sq(a, q, lone)?;
            let (f, r) = (x.file() as i32 + df, x.rank() as i32 + if !lone == Color::White { *dr } else { -dr });
            ((0..8).contains(&f) && (0..8).contains(&r)).then(|| Square::from_coords(shakmaty::File::new(f as u32), shakmaty::Rank::new(r as u32))) }
        Term::Next(a, c) => { let (x, y) = (sq(a, q, lone)?, sq(c, q, lone)?);
            if x == y || !shakmaty::attacks::queen_attacks(x, Bitboard::EMPTY).contains(y) { return None; }
            ray_past(x, y).into_iter().find(|s| b.occupied().contains(*s)) }
        // a named square, from our side: our first rank is rank 1 (the mirror of Lean `named`)
        Term::Named(x) => Some(if !lone == Color::White { *x } else { x.flip_vertical() }),
        _ => None,
    }
}

/// A set term on the board (the mirror of Lean `Ideas.Vocabulary`).
pub fn set(t: &Term, q: &Chess, lone: Color) -> Bitboard {
    let b = q.board();
    let s1 = |x: &Term| sq(x, q, lone);
    match t {
        Term::Edge => Bitboard::from(shakmaty::Rank::First) | Bitboard::from(shakmaty::Rank::Eighth) | Bitboard::from(shakmaty::File::A) | Bitboard::from(shakmaty::File::H),
        Term::Corner => [Square::A1, Square::A8, Square::H1, Square::H8].into_iter().collect(),
        Term::Attacked | Term::Free => {
            let Some(k) = b.king_of(lone) else { return Bitboard::EMPTY };
            let occ = b.occupied().without(k);
            let at: Bitboard = Bitboard::FULL.into_iter().filter(|s| b.attacks_to(*s, !lone, occ).any()).collect();
            if matches!(t, Term::Attacked) { at } else { !at }
        }
        Term::Hit | Term::Unhit => {
            let Some(k) = b.king_of(!lone) else { return Bitboard::EMPTY };
            let occ = b.occupied().without(k);
            let at: Bitboard = Bitboard::FULL.into_iter().filter(|s| b.attacks_to(*s, lone, occ).any()).collect();
            if matches!(t, Term::Hit) { at } else { !at }
        }
        Term::Ours => b.by_color(!lone),
        Term::Theirs => b.by_color(lone),
        // the second batch (mirrors of Lean Ideas.Vocabulary lines / diagonals / segment / knight1 / knight2 / center /
        // attacksFrom / cls)
        Term::Lines(a) => s1(a).map_or(Bitboard::EMPTY, |s| (Bitboard::from(s.rank()) | Bitboard::from(s.file())).without(s)),
        Term::Diagonals(a) => s1(a).map_or(Bitboard::EMPTY, |s| shakmaty::attacks::bishop_attacks(s, Bitboard::EMPTY)),
        Term::Segment(a, c) => match (s1(a), s1(c)) {
            (Some(x), Some(y)) if x != y && shakmaty::attacks::queen_attacks(x, Bitboard::EMPTY).contains(y) =>
                shakmaty::attacks::between(x, y).with(x).with(y),
            _ => Bitboard::EMPTY },
        Term::Knight1(a) => s1(a).map_or(Bitboard::EMPTY, shakmaty::attacks::knight_attacks),
        Term::Knight2(a) => s1(a).map_or(Bitboard::EMPTY, |s| shakmaty::attacks::knight_attacks(s).into_iter()
            .fold(Bitboard::EMPTY, |m, u| m | shakmaty::attacks::knight_attacks(u))),
        Term::Center => [Square::D4, Square::E4, Square::D5, Square::E5].into_iter().collect(),
        Term::Light => Bitboard::LIGHT_SQUARES, Term::Dark => Bitboard::DARK_SQUARES,
        Term::At(a) => s1(a).map_or(Bitboard::EMPTY, Bitboard::from_square),
        Term::Kind(ours, r) => b.by_color(if *ours { !lone } else { lone }) & b.by_role(*r),
        Term::Ahead(a) | Term::Passer(a) => {
            let Some(x) = s1(a) else { return Bitboard::EMPTY };
            let Some(c) = b.color_at(x) else { return Bitboard::EMPTY };
            let wide = matches!(t, Term::Passer(_));
            Bitboard::FULL.into_iter().filter(|s| {
                let ahead = if c == Color::White { s.rank() > x.rank() } else { s.rank() < x.rank() };
                let df = (s.file() as i32 - x.file() as i32).abs();
                ahead && (df == 0 || (wide && df == 1))
            }).collect()
        }
        Term::Beyond(a, c) => match (s1(a), s1(c)) {
            (Some(x), Some(y)) if x != y && shakmaty::attacks::queen_attacks(x, Bitboard::EMPTY).contains(y) => ray_past(x, y).into_iter().collect(),
            _ => Bitboard::EMPTY },
        Term::Xray(a) => {
            let Some(x) = s1(a) else { return Bitboard::EMPTY };
            let Some(p) = b.piece_at(x) else { return Bitboard::EMPTY };
            let dirs: &[(i32, i32)] = match p.role {
                Role::Rook => &[(1, 0), (-1, 0), (0, 1), (0, -1)], Role::Bishop => &[(1, 1), (1, -1), (-1, 1), (-1, -1)],
                Role::Queen => &[(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)],
                _ => return shakmaty::attacks::attacks(x, p, b.occupied()),
            };
            let mut out = Bitboard::EMPTY;
            for (df, dr) in dirs {
                let mut seen_blocker = 0;
                let (mut f, mut r) = (x.file() as i32 + df, x.rank() as i32 + dr);
                while (0..8).contains(&f) && (0..8).contains(&r) {
                    let s = Square::from_coords(shakmaty::File::new(f as u32), shakmaty::Rank::new(r as u32));
                    out.add(s);
                    if b.occupied().contains(s) { seen_blocker += 1; if seen_blocker == 2 { break; } }
                    f += df; r += dr;
                }
            }
            out
        }
        Term::Moves(a) => {
            let Some(x) = s1(a) else { return Bitboard::EMPTY };
            let Some(c) = b.color_at(x) else { return Bitboard::EMPTY };
            let pos = if c == q.turn() { Some(q.clone()) } else if q.is_check() { None } else { q.clone().swap_turn().ok() };
            pos.map_or(Bitboard::EMPTY, |p| p.legal_moves().into_iter().filter(|m| m.from() == Some(x)).map(|m| m.to()).collect())
        }
        Term::Colour(x) => match sq(x, q, lone) {
            Some(s) if Bitboard::LIGHT_SQUARES.contains(s) => Bitboard::LIGHT_SQUARES,
            Some(_) => Bitboard::DARK_SQUARES,
            None => Bitboard::EMPTY,
        },
        Term::Attacks(a) => s1(a).and_then(|s| b.piece_at(s).map(|p| shakmaty::attacks::attacks(s, p, b.occupied()))).unwrap_or(Bitboard::EMPTY),
        Term::Cls(x) => {
            let us = !lone;
            let abs = if us == Color::White { *x } else { x.flip_vertical() };
            if b.pawns().any() { Bitboard::from_square(abs).with(abs.flip_horizontal()) }
            else {
                [abs, abs.flip_horizontal(), abs.flip_vertical(), abs.flip_horizontal().flip_vertical(),
                 abs.flip_diagonal(), abs.flip_diagonal().flip_horizontal(), abs.flip_diagonal().flip_vertical(),
                 abs.flip_diagonal().flip_horizontal().flip_vertical()].into_iter().collect()
            }
        }
        Term::Box1(a) => s1(a).map_or(Bitboard::EMPTY, |s| shakmaty::attacks::king_attacks(s).with(s)),
        Term::Rank(a) => s1(a).map_or(Bitboard::EMPTY, |s| Bitboard::from(s.rank())),
        Term::File(a) => s1(a).map_or(Bitboard::EMPTY, |s| Bitboard::from(s.file())),
        Term::Line(a, c) => match (s1(a), s1(c)) { (Some(x), Some(y)) if x != y => shakmaty::attacks::ray(x, y), _ => Bitboard::EMPTY },
        Term::Between(a, c) => match (s1(a), s1(c)) { (Some(x), Some(y)) => shakmaty::attacks::between(x, y), _ => Bitboard::EMPTY },
        Term::Nearer(a, c) => match (s1(a), s1(c)) {
            (Some(x), Some(y)) => Bitboard::FULL.into_iter().filter(|u| kd(*u, x) < kd(*u, y)).collect(), _ => Bitboard::EMPTY },
        Term::Mobility(a) => { let Some(x) = s1(a) else { return Bitboard::EMPTY }; mobility(q, x) }
        Term::And(a, c) => set(a, q, lone) & set(c, q, lone),
        Term::Or(a, c) => set(a, q, lone) | set(c, q, lone),
        Term::Not(a) => !set(a, q, lone),
        Term::Reach(a, c) => {
            let Some(s) = s1(a) else { return Bitboard::EMPTY };
            let allowed = set(c, q, lone);
            let mut seen = Bitboard::from_square(s);
            let mut frontier = vec![s];
            while let Some(x) = frontier.pop() {
                for y in shakmaty::attacks::king_attacks(x) & allowed & !seen { seen.add(y); frontier.push(y); }
            }
            seen
        }
        Term::Gained(a) => crate::game::back(1).map_or(Bitboard::EMPTY, |p| set(a, q, lone) & !set(a, &p, lone)),
        Term::Lost(a) => crate::game::back(1).map_or(Bitboard::EMPTY, |p| set(a, &p, lone) & !set(a, q, lone)),
        _ => Bitboard::EMPTY,
    }
}

/// A number term on the board.
pub fn num(t: &Term, q: &Chess, lone: Color) -> u32 {
    match t {
        Term::Count(a) => set(a, q, lone).count() as u32,
        Term::Dist(a, c) => match (sq(a, q, lone), sq(c, q, lone)) { (Some(x), Some(y)) => kd(x, y), _ => 0 },
        Term::Sum(a, c) => num(a, q, lone) + num(c, q, lone),
        // (from our side: our first rank is 0; the files as they stand -- the mirror of Lean `fileOf`, `rankOf`)
        Term::FileOf(a) => sq(a, q, lone).map_or(0, |x| x.file() as u32),
        Term::RankOf(a) => sq(a, q, lone).map_or(0, |x| our_rank(x, !lone)),
        Term::DFile(a, c) => match (sq(a, q, lone), sq(c, q, lone)) { (Some(x), Some(y)) => (x.file() as i32 - y.file() as i32 + 7) as u32, _ => 0 },
        Term::DRank(a, c) => match (sq(a, q, lone), sq(c, q, lone)) {
            (Some(x), Some(y)) => (our_rank(x, !lone) as i32 - our_rank(y, !lone) as i32 + 7) as u32, _ => 0 },
        Term::Pushes(a) => sq(a, q, lone).and_then(|x| q.board().color_at(x).map(|c| 7 - our_rank(x, c))).unwrap_or(0),
        Term::Attackers(a) | Term::Defenders(a) => sq(a, q, lone).map_or(0, |x| {
            let side = if matches!(t, Term::Attackers(_)) { !lone } else { lone };
            (q.board().attacks_to(x, side, q.board().occupied()) & q.board().by_color(side)).count() as u32 }),
        Term::See(a) => sq(a, q, lone).and_then(|x| q.board().color_at(x).map(|c| crate::rules::see(q.board(), x, !c).max(0) as u32)).unwrap_or(0),
        // the declared value of the piece on a square, whoever owns it (pawn 1, minor 3, rook 5, queen 9; a king or nothing 0)
        Term::Worth(a) => sq(a, q, lone).and_then(|x| q.board().role_at(x)).map_or(0, |r| if r == Role::King { 0 } else { crate::rules::val(r).max(0) as u32 }),
        // the largest declared value among the pieces on a set of squares (0 for an empty set or kings only)
        Term::MaxWorth(a) => set(a, q, lone).into_iter().filter_map(|x| q.board().role_at(x))
            .map(|r| if r == Role::King { 0 } else { crate::rules::val(r).max(0) as u32 }).max().unwrap_or(0),
        Term::Forcing(ours, v) => crate::goals::forcing(q, *ours, *v),
        Term::MaterialUs => material(q.board(), !lone).max(0) as u32,
        // the material balance on an offset of 64 (Kaveh 2026-10-15: THE observable; one number for a tree's leaf)
        Term::Balance => (material(q.board(), !lone).max(0) as u32 + 64).saturating_sub(material(q.board(), lone).max(0) as u32),
        Term::MaterialThem => material(q.board(), lone).max(0) as u32,
        Term::Halfmoves => q.halfmoves(),
        Term::Seen => crate::game::seen(q) as u32,
        Term::Ago(k, a) => crate::game::back(2 * *k).map_or(0, |p| num(a, &p, !p.turn())),
        Term::Without(x, a) => without(x, a, q, lone),
        Term::If(c, a, d) => if num(c, q, lone) != 0 { num(a, q, lone) } else { num(d, q, lone) },
        Term::After(f, t2, a) => after_move(f, t2, a, q, lone),
        // least / most (Lean leastOver / mostOver): the body on the board after each legal move of the side to move, `lone` unchanged
        Term::Least(a) => q.legal_moves().iter().map(|m| num(a, &after(q, m), lone)).min().unwrap_or(0),
        Term::Most(a) => q.legal_moves().iter().map(|m| num(a, &after(q, m), lone)).max().unwrap_or(0),
        Term::Delta(a) => crate::game::back(1).map_or(DELTA_ZERO, |p| (num(a, q, lone) + DELTA_ZERO).saturating_sub(num(a, &p, lone))),
        Term::Attackable(a, k) => sq(a, q, lone).and_then(|x| q.board().color_at(x).map(|c| reach_route(q, !c, x, *k).is_ok() as u32)).unwrap_or(0),
        Term::Declined(a) => sq(a, q, lone).map_or(0, |x| declined(q, x) as u32),
        Term::Min(a, c) => num(a, q, lone).min(num(c, q, lone)),
        Term::Max(a, c) => num(a, q, lone).max(num(c, q, lone)),
        Term::Diff(a, c) => num(a, q, lone).abs_diff(num(c, q, lone)),
        Term::Eq(a, c) => (num(a, q, lone) == num(c, q, lone)) as u32,
        Term::Int(n) => (*n).max(0) as u32,
        _ => 0,
    }
}

/// The same board with `c` to move (built relaxed: the other side's king may stand attacked).
fn with_turn(q: &Chess, c: Color) -> Option<Chess> {
    if q.turn() == c { return Some(q.clone()); }
    let mut st = q.clone().into_setup(shakmaty::EnPassantMode::Legal);
    st.turn = c;
    relaxed(st).filter(|p| p.turn() == c)
}

/// A move lands safely: after it the exchange on its destination wins nothing for the other side (`see` after the move, as the word `see`).
fn lands_safe(after: &Chess, to: Square, mover: Color) -> bool { crate::rules::see(after.board(), to, !mover) == 0 }

/// **mobility X**: the squares the piece on `x` can legally move to (its own side to move, as `moves`) where it is safe by the exchange. Castling is not a move
/// of the piece in this word.
pub(crate) fn mobility(q: &Chess, x: Square) -> Bitboard {
    let Some(c) = q.board().color_at(x) else { return Bitboard::EMPTY };
    let Some(p) = with_turn(q, c) else { return Bitboard::EMPTY };
    let mut out = Bitboard::EMPTY;
    for m in p.legal_moves() {
        if m.from() != Some(x) || m.is_castle() { continue; }
        let mut a = p.clone(); a.play_unchecked(&m);
        if lands_safe(&a, m.to(), c) { out.add(m.to()); }
    }
    out
}

/// **declined X** (Kaveh 2026-10-15, "if they don't"): what the side to move can win by the exchange on a square OTHER than `x` -- its best capture if it
/// declines to take on `x`. The split of a sacrifice: if they take, the exact solver's forced line; if they don't, this. Lean `declinedOf`.
pub(crate) fn declined(q: &Chess, x: Square) -> i64 {
    let b = q.board();
    b.by_color(!q.turn()).into_iter().filter(|t| *t != x).map(|t| crate::rules::see(b, t, q.turn())).max().unwrap_or(0)
}

/// **attackable / REACH** (ANALYST_PLAN 40.2): can side `c` attack square `s` within `k` of its moves (the other side passes), every landing square safe by the
/// exchange? -> Ok(the route: the moves (from, to) in order; empty when `s` is attacked already) or Err(the blocking squares: the landing squares from which a
/// piece of `c` would attack `s` but where it is lost by the exchange). The shortest route (iterative deepening); a node cap bounds the search.
pub(crate) fn reach_route(q: &Chess, c: Color, s: Square, k: u32) -> Result<Vec<(Square, Square)>, Bitboard> {
    let attacked_now = |b: &shakmaty::Board| (b.attacks_to(s, c, b.occupied()) & b.by_color(c)).any();
    if attacked_now(q.board()) { return Ok(vec![]); }
    let mut blockers = Bitboard::EMPTY;
    let Some(start) = with_turn(q, c) else { return Err(blockers) };
    fn go(p: &Chess, c: Color, s: Square, d: u32, route: &mut Vec<(Square, Square)>, blockers: &mut Bitboard, nodes: &mut u32, seen: &mut std::collections::HashSet<(u64, u32)>) -> bool {
        if !seen.insert((crate::threat::zkey(p), d)) { return false; }
        for m in p.legal_moves() {
            *nodes += 1;
            if *nodes > 30000 { return false; }
            if m.is_castle() { continue; }
            let mut a = p.clone(); a.play_unchecked(&m);
            let hits = (a.board().attacks_to(s, c, a.board().occupied()) & a.board().by_color(c)).any();
            if !lands_safe(&a, m.to(), c) { if hits { blockers.add(m.to()); } continue; }
            route.push((m.from().unwrap_or(m.to()), m.to()));
            if hits && d == 1 { return true; }
            if d > 1 && !hits {
                if let Some(n) = with_turn(&a, c) { if go(&n, c, s, d - 1, route, blockers, nodes, seen) { return true; } }
            }
            if hits { return true; }
            route.pop();
        }
        false
    }
    for d in 1..=k.max(1) {
        let mut route = vec![]; let mut nodes = 0; let mut seen = std::collections::HashSet::new();
        if go(&start, c, s, d, &mut route, &mut blockers, &mut nodes, &mut seen) { return Ok(route); }
    }
    Err(blockers)
}

/// Is this term a number (else a set, read as its count)?
pub fn is_num(t: &Term) -> bool {
    matches!(t, Term::Count(_) | Term::Dist(_, _) | Term::Sum(_, _) | Term::FileOf(_) | Term::RankOf(_) | Term::DFile(_, _) | Term::DRank(_, _)
        | Term::Pushes(_) | Term::Attackers(_) | Term::Defenders(_) | Term::See(_) | Term::Worth(_) | Term::MaxWorth(_) | Term::Forcing(_, _) | Term::MaterialUs | Term::MaterialThem | Term::Balance
        | Term::Halfmoves | Term::Seen | Term::Ago(_, _) | Term::Without(_, _) | Term::If(_, _, _) | Term::After(_, _, _) | Term::Least(_) | Term::Most(_) | Term::Delta(_) | Term::Attackable(_, _) | Term::Declined(_) | Term::Min(_, _) | Term::Max(_, _) | Term::Diff(_, _) | Term::Eq(_, _) | Term::Int(_))
}

/// The squares past `y` on the ray from `x` through `y`, nearest first.
fn ray_past(x: Square, y: Square) -> Vec<Square> {
    let (df, dr) = ((y.file() as i32 - x.file() as i32).signum(), (y.rank() as i32 - x.rank() as i32).signum());
    let mut out = Vec::new();
    let (mut f, mut r) = (y.file() as i32 + df, y.rank() as i32 + dr);
    while (0..8).contains(&f) && (0..8).contains(&r) {
        out.push(Square::from_coords(shakmaty::File::new(f as u32), shakmaty::Rank::new(r as u32)));
        f += df; r += dr;
    }
    out
}

/// A square's rank counted from `us`'s first rank (0-7).
fn our_rank(x: Square, us: Color) -> u32 { if us == Color::White { x.rank() as u32 } else { 7 - x.rank() as u32 } }

/// THE CERTIFIER: the base measure of the frame (the rook's, as the discoverer derived it: wall, opposition) extended by
/// `cand` under `schema` -- "append" (a third component), "sum" (added to the opposition), "replace" (in place of it);
/// a position is certified when some safe move makes the measure fall within `k` moves against every reply.
fn extended(q: &Chess, lone: Color, cand: &Term, schema: &str) -> Vec<u32> {
    let (w, o, c) = (crate::mating::wall_rows(q, lone), crate::mating::kr_opposition(q, lone), num(cand, q, lone));
    match schema { "sum" => vec![w, o + c], "replace" => vec![w, c], "none" => vec![w, o], _ => vec![w, o, c] }
}

fn worst(q: &Chess, lone: Color, cand: &Term, schema: &str) -> Vec<u32> {
    q.legal_moves().iter().map(|r| extended(&after(q, r), lone, cand, schema)).max().unwrap_or_else(|| vec![0; 3])
}

fn within(q: &Chess, before: &[u32], us: Color, k: u32, cand: &Term, schema: &str) -> bool {
    if q.is_checkmate() { return true; }
    if k <= 1 { return worst(q, !us, cand, schema).as_slice() < before; }
    q.legal_moves().iter().all(|r| {
        let x = after(q, r);
        x.legal_moves().iter().any(|m2| {
            let y = after(&x, m2);
            y.is_checkmate() || (crate::mating::safe_step(&y, us, Role::Rook) && within(&y, before, us, k - 1, cand, schema))
        })
    })
}

/// certified(pos): some safe move of ours makes the extended measure fall within `k` moves against every reply.
pub fn certified(pos: &Chess, cand: &Term, schema: &str, k: u32) -> bool {
    let us = pos.turn();
    let before = extended(pos, !us, cand, schema);
    pos.legal_moves().iter().any(|m| {
        let q = after(pos, m);
        q.is_checkmate() || (crate::mating::safe_step(&q, us, Role::Rook) && within(&q, &before, us, k, cand, schema))
    })
}

/// vocab_eval(fen, term) -> the term's value on the board (a number; a set as its count) -- the evaluator's hook.
#[pyfunction]
pub fn vocab_eval(fen: String, term: String) -> PyResult<Option<u32>> {
    let (Some(p), Some(t)) = (parse(&fen), parse_term(&term)) else { return Ok(None) };
    let lone = !p.turn();
    Ok(Some(if is_num(&t) { num(&t, &p, lone) } else { set(&t, &p, lone).count() as u32 }))
}

/// vocab_values(fens, term) -> the term's number on each board (the proposer's observational test).
#[pyfunction]
pub fn vocab_values(fens: Vec<String>, term: String) -> PyResult<Vec<u32>> {
    use rayon::prelude::*;
    let Some(t) = parse_term(&term) else { return Ok(vec![]) };
    Ok(fens.par_iter().map(|f| parse(f).map_or(0, |p| { let lone = !p.turn(); num(&t, &p, lone) })).collect())
}

/// vocab_certify(fens, term, schema, k) -> how many of the boards (the attacker to move) the extended measure certifies.
#[pyfunction]
pub fn vocab_certify(fens: Vec<String>, term: String, schema: String, k: u32) -> PyResult<u32> {
    use rayon::prelude::*;
    let Some(t) = parse_term(&term) else { return Ok(0) };
    Ok(fens.par_iter().filter(|f| parse(f).map_or(false, |p| certified(&p, &t, &schema, k))).count() as u32)
}

// ---------------------------------------------------------------------------------------------------------------------
// THE VERIFIER, GENERAL (Kaveh 2026-10-12: proposers -> a filter -> the verifier, which checks everything the filter
// kept): a measure given as components -- a named procedure of a frame ("fn:wall", "fn:opposition", "fn:chase",
// "fn:kq_offedge", "fn:kq_approach", "fn:kq_region") or a term over the base -- extended by a candidate term under a
// schema, certified on boards against every reply within `k` moves, the frame's piece kept safe.

enum Comp { F(fn(&Chess, Color) -> u32), T(Term) }

fn comp(spec: &str) -> Option<Comp> {
    Some(match spec {
        "fn:wall" => Comp::F(crate::mating::wall_rows), "fn:opposition" => Comp::F(crate::mating::kr_opposition),
        "fn:chase" => Comp::F(crate::mating::kr_chase), "fn:kq_offedge" => Comp::F(crate::mating::off_edge),
        "fn:kq_approach" => Comp::F(crate::mating::approach), "fn:kq_region" => Comp::F(crate::mating::region_size),
        "fn:kr_generated" => Comp::F(crate::mating::kr_generated),
        _ => Comp::T(parse_term(spec)?),
    })
}
fn cval(c: &Comp, q: &Chess, lone: Color) -> u32 { match c { Comp::F(f) => f(q, lone), Comp::T(t) => num(t, q, lone) } }

/// What the frame keeps safe: one piece from the lone king (the mating frames), or ALL our material from every piece of
/// theirs (the generic frame below).
#[derive(Clone, Copy)]
enum Safety { Piece(Role), Material }

struct Spec { base: Vec<Comp>, cand: Option<Term>, schema: String, safety: Safety,
              // THE FRAME'S HORIZON (Kaveh 2026-10-13: loops are not a kind of frame -- a frame whose measure falls once a
              // cycle has a longer horizon): "horizon:k" among its terms; none -- CLAIM_K
              horizon: Option<u32> }

/// A frame's horizon, if its terms carry one ("horizon:k").
fn horizon_of(terms: &[String]) -> Option<u32> { terms.iter().find_map(|t| t.strip_prefix("horizon:").and_then(|x| x.parse::<u32>().ok())) }

impl Spec {
    fn measure(&self, q: &Chess, lone: Color) -> Vec<u32> {
        let mut v: Vec<u32> = self.base.iter().map(|c| cval(c, q, lone)).collect();
        if let Some(t) = &self.cand {
            let x = num(t, q, lone);
            match self.schema.as_str() {
                "sum" => { if let Some(l) = v.last_mut() { *l += x } else { v.push(x) } }
                "replace" => { if let Some(l) = v.last_mut() { *l = x } else { v.push(x) } }
                "append" => v.push(x),
                "prepend" => v.insert(0, x),                  // ranked above the frame (Kaveh: first don't lose)
                _ => {}
            }
        }
        v
    }
    /// Our move `q` is safe for the frame (the full check: the move we play).
    fn safe(&self, q: &Chess, us: Color) -> bool {
        match self.safety { Safety::Piece(r) => crate::mating::safe_step(q, us, r), Safety::Material => material_safe(q, us) }
    }
    /// The same inside the certificate's search (cheaper for the material: nothing en prise now).
    fn safe_quick(&self, q: &Chess, us: Color) -> bool {
        match self.safety {
            Safety::Piece(r) => crate::mating::safe_step(q, us, r),
            Safety::Material => q.is_checkmate() || (!q.legal_moves().is_empty() && crate::game::claimable_draw(q).is_none() && static_safe(q.board(), us)),
        }
    }
    fn worst(&self, q: &Chess, lone: Color) -> Vec<u32> {
        q.legal_moves().iter().map(|r| self.measure(&after(q, r), lone)).max().unwrap_or_default()
    }
    fn within(&self, q: &Chess, before: &[u32], us: Color, k: u32) -> bool {
        if q.is_checkmate() { return true; }
        if k <= 1 { return self.worst(q, !us).as_slice() < before; }
        q.legal_moves().iter().all(|r| {
            let x = after(q, r);
            x.legal_moves().iter().any(|m2| {
                let y = after(&x, m2);
                y.is_checkmate() || (self.safe_quick(&y, us) && self.within(&y, before, us, k - 1))
            })
        })
    }
    fn certified(&self, pos: &Chess, k: u32) -> bool {
        let us = pos.turn();
        let before = self.measure(pos, !us);
        pos.legal_moves().iter().any(|m| {
            let q = after(pos, m);
            q.is_checkmate() || (self.safe(&q, us) && self.within(&q, &before, us, k))
        })
    }
}

fn spec(base: &[String], cand: &Option<String>, schema: &str, piece: &str) -> Option<Spec> {
    let horizon = horizon_of(base);
    let base: Vec<Comp> = base.iter().filter(|s| !s.starts_with("horizon:")).map(|s| comp(s)).collect::<Option<_>>()?;
    let cand = match cand { Some(c) => Some(parse_term(c)?), None => None };
    let safety = match piece { "Q" | "q" => Safety::Piece(Role::Queen), "*" => Safety::Material, _ => Safety::Piece(Role::Rook) };
    Some(Spec { base, cand, schema: schema.to_string(), safety, horizon })
}

/// verify_mask(fens, base, cand, schema, k, piece) -> for each board (the attacker to move): certified or not.
#[pyfunction]
#[pyo3(signature = (fens, base, cand, schema, k, piece))]
pub fn verify_mask(fens: Vec<String>, base: Vec<String>, cand: Option<String>, schema: String, k: u32, piece: String) -> PyResult<Vec<bool>> {
    use rayon::prelude::*;
    let Some(sp) = spec(&base, &cand, &schema, &piece) else { return Ok(vec![]) };
    Ok(fens.par_iter().map(|f| parse(f).map_or(false, |p| sp.certified(&p, k))).collect())
}

/// verify_count(...) -> how many boards are certified (the verifier's score).
#[pyfunction]
#[pyo3(signature = (fens, base, cand, schema, k, piece))]
pub fn verify_count(fens: Vec<String>, base: Vec<String>, cand: Option<String>, schema: String, k: u32, piece: String) -> PyResult<u32> {
    Ok(verify_mask(fens, base, cand, schema, k, piece)?.into_iter().filter(|x| *x).count() as u32)
}

/// component_values(fens, specs) -> each spec's value on each board (the lone king = the side not to move): the hole's
/// own measure, read for the learned proposers' context.
#[pyfunction]
pub fn component_values(fens: Vec<String>, specs: Vec<String>) -> PyResult<Vec<Vec<u32>>> {
    use rayon::prelude::*;
    let comps: Vec<Comp> = specs.iter().filter_map(|s| comp(s)).collect();
    Ok(fens.par_iter().map(|f| parse(f).map_or(vec![0; comps.len()], |p| { let lone = !p.turn(); comps.iter().map(|c| cval(c, &p, lone)).collect() })).collect())
}

/// component_values_us(fens, specs, us_white) -> the same, but "us" is NAMED: the evaluator's lone king is the other colour whatever the side to move is.
/// For a board read from one side's point of view with the OTHER side to move (the true board after S's move: us = S, O to move).
#[pyfunction]
pub fn component_values_us(fens: Vec<String>, specs: Vec<String>, us_white: bool) -> PyResult<Vec<Vec<u32>>> {
    use rayon::prelude::*;
    let comps: Vec<Comp> = specs.iter().filter_map(|s| comp(s)).collect();
    let lone = if us_white { Color::Black } else { Color::White };
    Ok(fens.par_iter().map(|f| parse(f).map_or(vec![0; comps.len()], |p| comps.iter().map(|c| cval(c, &p, lone)).collect())).collect())
}

// ---------------------------------------------------------------------------------------------------------------------
// THE VERIFIER, STAGE 2 -- CLOSED LOOP (2026-10-12: stage 1 is gameable -- "how many squares we attack" certified 100% of
// every hole's stuck boards: a measure we lower ourselves at will, whatever the defender does): the candidate measure
// PLAYS from the hole's boards -- our move chosen by it (against every reply, then a tempo move within two), the
// defender's reply the one that raises it most and never walks into mate in one; a repetition (a position the third
// time) or no mate within the limit is a failure. The defender here is the measure's adversary, no engine.

impl Spec {
    /// -> (the move, whether it carries a certificate: the measure falls against every reply, at once or within two --
    /// or, for the material, the goal: a safe capture); `seen` counts the positions of the game so far.
    fn choose_cert(&self, pos: &Chess, seen: &dyn Fn(&Chess) -> u8, allow: &dyn Fn(&Move) -> bool) -> Option<(Move, bool)> {
        let us = pos.turn();
        let before = self.measure(pos, !us);
        let mut cands: Vec<(Vec<u32>, Vec<u32>, bool, Move)> = Vec::new();
        for m in pos.legal_moves() {
            let q = after(pos, &m);
            if q.is_checkmate() { return Some((m, true)); }
            if !self.safe(&q, us) || seen(&q) >= 2 || !allow(&m) { continue; }
            // THE GOAL of the material's frame: a capture we keep (it wins material, or trades into a smaller material)
            if matches!(self.safety, Safety::Material) && m.is_capture() { return Some((m, true)); }
            cands.push((self.worst(&q, !us), self.measure(&q, !us), seen(&q) > 0, m));
        }
        cands.sort_by(|a, b| (&a.0, &a.1, a.2).cmp(&(&b.0, &b.1, b.2)));
        let first = cands.first()?;
        if first.0 < before { return Some((first.3.clone(), true)); }
        for (_, _, _, m) in &cands {
            if self.within(&after(pos, m), &before, us, self.horizon.unwrap_or_else(|| CLAIM_K.load(std::sync::atomic::Ordering::Relaxed))) { return Some((m.clone(), true)); }
        }
        Some((first.3.clone(), false))
    }
    fn defend(&self, q: &Chess) -> Option<Move> {
        let lone = q.turn();
        // the measure's adversary: material first (what it takes from us), then the measure as high as it can make it,
        // never into a mate in one
        q.legal_moves().into_iter().filter(|r| {
            let x = after(q, r);
            !x.legal_moves().iter().any(|m| after(&x, m).is_checkmate())
        }).max_by_key(|r| { let x = after(q, r); (-material(x.board(), !lone), self.measure(&x, lone), crate::threat::zkey(&x)) })
            .or_else(|| q.legal_moves().into_iter().next())
    }
    /// -> (Some(moves to mate) or None -- no mate within the limit, a repetition, a stalemate, the piece lost; our moves
    /// played; how many of them carried a certificate)
    fn play(&self, start: &Chess, limit: u32) -> (Option<u32>, u32, u32) {
        let mut p = start.clone();
        let mut seen: std::collections::HashMap<u64, u8> = std::collections::HashMap::new();
        let (mut played, mut cert) = (0u32, 0u32);
        let us = start.turn();
        let (theirs0, lead0) = (material(start.board(), !us), material(start.board(), us) - material(start.board(), !us));
        for n in 1..=limit {
            *seen.entry(crate::threat::zkey(&p)).or_insert(0) += 1;
            let pick = { let s = &seen; self.choose_cert(&p, &|q: &Chess| s.get(&crate::threat::zkey(q)).copied().unwrap_or(0), &|_| true) };
            let Some((m, c)) = pick else { return (None, played, cert) };
            played += 1; cert += c as u32;
            p = after(&p, &m);
            if p.is_checkmate() { return (Some(n), played, cert); }
            if matches!(self.safety, Safety::Material) {
                let lead = material(p.board(), us) - material(p.board(), !us);
                // FIRST DON'T LOSE (Kaveh 2026-10-12): ground lost after our move is a failure, whatever follows
                if lead < lead0 { return (None, played, cert); }
                // then the goal: theirs smaller, our lead no smaller -- the next frame's material (Reach.bind)
                if material(p.board(), !us) < theirs0 { return (Some(n), played, cert); }
            }
            if p.legal_moves().is_empty() { return (None, played, cert); }
            *seen.entry(crate::threat::zkey(&p)).or_insert(0) += 1;
            if seen[&crate::threat::zkey(&p)] >= 3 { return (None, played, cert); }
            let Some(r) = self.defend(&p) else { return (None, played, cert) };
            p = after(&p, &r);
            if matches!(self.safety, Safety::Piece(_)) && (p.board().by_color(p.turn()) & !p.board().kings()).is_empty() { return (None, played, cert); }   // our piece taken
            if seen.get(&crate::threat::zkey(&p)).copied().unwrap_or(0) >= 2 { return (None, played, cert); }
        }
        (None, played, cert)
    }
}

/// frame_choose(fen, base, cand, schema, piece, history) -> the frame's move and whether it carries a certificate -- ONE
/// move of the closed loop, for a defender outside Rust (verifier.py: the tablebase, Stockfish); `history` the game's
/// positions so far (a position seen twice is not played into).
#[pyfunction]
#[pyo3(signature = (fen, base, cand, schema, piece, history))]
pub fn frame_choose(fen: String, base: Vec<String>, cand: Option<String>, schema: String, piece: String, history: Vec<String>) -> PyResult<Option<(String, bool)>> {
    let (Some(sp), Some(p)) = (spec(&base, &cand, &schema, &piece), parse(&fen)) else { return Ok(None) };
    let mut seen: std::collections::HashMap<u64, u8> = std::collections::HashMap::new();
    for f in &history { if let Some(q) = parse(f) { *seen.entry(crate::threat::zkey(&q)).or_insert(0) += 1; } }
    Ok(sp.choose_cert(&p, &|q: &Chess| seen.get(&crate::threat::zkey(q)).copied().unwrap_or(0), &|_| true)
        .map(|(m, c)| (m.to_uci(CastlingMode::Standard).to_string(), c)))
}

/// verify_play(fens, base, cand, schema, piece, limit) -> (games mated, total moves of the mated games, our moves played,
/// our moves certified): stage 2 -- the closed loop.
#[pyfunction]
#[pyo3(signature = (fens, base, cand, schema, piece, limit))]
pub fn verify_play(fens: Vec<String>, base: Vec<String>, cand: Option<String>, schema: String, piece: String, limit: u32) -> PyResult<(u32, u32, u32, u32)> {
    use rayon::prelude::*;
    let Some(sp) = spec(&base, &cand, &schema, &piece) else { return Ok((0, 0, 0, 0)) };
    let r: Vec<(Option<u32>, u32, u32)> = fens.par_iter().map(|f| parse(f).map_or((None, 0, 0), |p| sp.play(&p, limit))).collect();
    Ok((r.iter().filter(|x| x.0.is_some()).count() as u32, r.iter().filter_map(|x| x.0).sum(),
        r.iter().map(|x| x.1).sum(), r.iter().map(|x| x.2).sum()))
}

/// term_dynamics(fens, term) -> per board (mean |change| of the term over our moves, over their replies after a pass):
/// WHO CONTROLS the quantity -- a measure only we move is gameable (stage 1 cannot tell), one both move is not.
#[pyfunction]
pub fn term_dynamics(fens: Vec<String>, term: String) -> PyResult<Vec<(f32, f32)>> {
    use rayon::prelude::*;
    let Some(t) = parse_term(&term) else { return Ok(vec![]) };
    Ok(fens.par_iter().map(|f| {
        let Some(p) = parse(f) else { return (0.0, 0.0) };
        let lone = !p.turn();
        let v0 = num(&t, &p, lone) as f32;
        let ours: Vec<f32> = p.legal_moves().iter().map(|m| (num(&t, &after(&p, m), lone) as f32 - v0).abs()).collect();
        let theirs: Vec<f32> = pass(&p).map(|q| q.legal_moves().iter().map(|r| (num(&t, &after(&q, r), lone) as f32 - v0).abs()).collect()).unwrap_or_default();
        let mean = |v: &[f32]| if v.is_empty() { 0.0 } else { v.iter().sum::<f32>() / v.len() as f32 };
        (mean(&ours), mean(&theirs))
    }).collect())
}

// ---------------------------------------------------------------------------------------------------------------------
// THE GENERIC FRAME, ANY MATERIAL (Kaveh 2026-10-12: "the KRR frame is the one that is new ... we can use the earlier
// lemmas to our advantage"): a material's measure given at run time as terms over the base -- the lemma mined from the
// games' surprises; its SAFETY the material's: nothing of ours lost to an exchange now (the static exchange, the mirror
// of Lean Ideas.ExchangeLaw.seeXB) nor after any reply (the skewer, the double attack: we keep a move after which
// nothing is en prise); its GOAL mate, or a capture we keep -- the material changes and the next frame takes over (a
// trade into king and rook is a material we already win: Reach.bind). Played when the planner proves nothing.

/// Our material (the kernel values, kings out).
pub(crate) fn material(b: &shakmaty::Board, c: Color) -> i64 {
    (b.by_color(c) & !b.kings()).into_iter().filter_map(|s| b.role_at(s)).map(crate::rules::val).sum()
}

/// The material from `us`'s side, "KRRvKR".
pub fn material_key(b: &shakmaty::Board, us: Color) -> String {
    let side = |c: Color| {
        let mut s = String::from("K");
        for r in [Role::Queen, Role::Rook, Role::Bishop, Role::Knight, Role::Pawn] {
            for _ in 0..(b.by_role(r) & b.by_color(c)).count() { s.push(r.upper_char()); }
        }
        s
    };
    format!("{}v{}", side(us), side(!us))
}

/// They win nothing by capturing: the static exchange for them on every piece of ours is zero.
fn static_safe(b: &shakmaty::Board, us: Color) -> bool {
    (b.by_color(us) & !b.kings()).into_iter().all(|s| crate::rules::see(b, s, !us) == 0)
}

/// After our move `q` (they to move): no stalemate, no claimable draw; nothing of ours en prise; and against every reply
/// (none mating us, none a claimable draw) we keep a move after which nothing of ours is en prise.
pub(crate) fn material_safe(q: &Chess, us: Color) -> bool {
    if q.is_checkmate() { return true; }
    if q.legal_moves().is_empty() || crate::game::claimable_draw(q).is_some() || !static_safe(q.board(), us) { return false; }
    q.legal_moves().iter().all(|r| {
        let x = after(q, r);
        if x.is_checkmate() || crate::game::claimable_draw(&x).is_some() { return false; }
        // our answer keeps the material AND the game (2026-10-12, round 1 game 1001: after Kf2 Re2+ the only answer
        // keeping everything was Rxe2 -- stalemate)
        x.legal_moves().iter().any(|m| {
            let y = after(&x, m);
            static_safe(y.board(), us) && (y.is_checkmate() || (!y.legal_moves().is_empty() && crate::game::claimable_draw(&y).is_none()))
        })
    })
}

/// material_safe_moves(fen) -> the moves the generic frame counts safe (a debugging view).
#[pyfunction]
pub fn material_safe_moves(fen: String) -> PyResult<Vec<String>> {
    let Some(p) = parse(&fen) else { return Ok(vec![]) };
    let us = p.turn();
    Ok(p.legal_moves().iter().filter(|m| material_safe(&after(&p, m), us)).map(|m| m.to_uci(CastlingMode::Standard).to_string()).collect())
}

pub static GENERIC_FRAME: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
static MATERIAL_FRAMES: std::sync::RwLock<Option<std::collections::HashMap<String, Vec<String>>>> = std::sync::RwLock::new(None);

/// The generic frame on (the planner plays it when it proves nothing), or off.
#[pyfunction]
pub fn set_generic_frame(flag: bool) -> PyResult<()> { GENERIC_FRAME.store(flag, std::sync::atomic::Ordering::Relaxed); Ok(()) }

/// set_material_frame(material, terms) -> whether every term parses: the material's measure (ranked); [] clears it.
#[pyfunction]
pub fn set_material_frame(material: String, terms: Vec<String>) -> PyResult<bool> {
    if !terms.iter().all(|t| t.starts_with("horizon:") || parse_term(t).is_some()) { return Ok(false); }
    let mut g = MATERIAL_FRAMES.write().unwrap();
    let map = g.get_or_insert_with(Default::default);
    if terms.is_empty() { map.remove(&material); } else { map.insert(material, terms); }
    Ok(true)
}

/// material_frames() -> {material: terms}.
#[pyfunction]
pub fn material_frames() -> PyResult<std::collections::HashMap<String, Vec<String>>> {
    Ok(MATERIAL_FRAMES.read().unwrap().clone().unwrap_or_default())
}

/// A material we KNOW how to win (from the side to move's view): a mating frame's, or one with a frame of the run (a
/// learned measure whose claims we rely on) -- the forced entry's goal (threat.rs known_entry).
pub fn is_known(key: &str) -> bool {
    matches!(key, "KQvK" | "KRvK") || MATERIAL_FRAMES.read().unwrap().as_ref().map_or(false, |m| m.contains_key(key))
}

/// The frame a material is known by (the run's mined frames, else the mating frames' named measures).
fn known_frame(key: &str) -> Vec<String> {
    if let Some(t) = MATERIAL_FRAMES.read().unwrap().as_ref().and_then(|m| m.get(key).cloned()) { return t; }
    match key {
        "KRvK" => vec!["fn:wall".into(), "fn:kr_generated".into()],
        "KQvK" => vec!["fn:kq_offedge".into(), "fn:kq_approach".into(), "fn:kq_region".into()],
        _ => vec![],
    }
}

/// THE HOLD FRAME (Kaveh 2026-10-12: "first, don't lose, and then try to win"; "when losing, drag the loss out as long
/// as possible ... same thing with drawing"): not ahead, the frame's moves are the audited ones (nothing lost; a draw
/// is worth our deficit), and among them the one keeping THEIR frame's measure highest against their best reply -- the
/// mirror of the frame they would play (an earlier lemma, from their side); then a position not yet seen.
fn hold_move(pos: &Chess, key: &str, net: &dyn Fn(&Move) -> i64) -> Option<(Move, String)> {
    let us = pos.turn();
    let mirror = material_key(pos.board(), !us);
    let specs = known_frame(&mirror);
    let comps: Vec<Comp> = specs.iter().filter_map(|s| comp(s)).collect();
    let theirs = |q: &Chess| -> Vec<u32> { comps.iter().map(|c| cval(c, q, us)).collect() };
    let mut cands: Vec<(Vec<u32>, bool, Move)> = Vec::new();
    for m in pos.legal_moves() {
        if net(&m) < 0 { continue; }
        let q = after(pos, &m);
        let held = q.legal_moves().iter().map(|r| theirs(&after(&q, r))).min().unwrap_or_default();
        cands.push((held, crate::game::seen(&q) == 0, m));
    }
    cands.sort_by(|a, b| (&b.0, b.1).cmp(&(&a.0, a.1)));
    let (held, _, m) = cands.into_iter().next()?;
    let before = theirs(pos);
    let claim = if specs.is_empty() {
        format!("hold ({key}, not ahead): nothing lost within {FRAME_K} moves (no frame known for their side, {mirror})")
    } else {
        format!("hold ({key}, not ahead): nothing lost within {FRAME_K} moves; their frame's measure {specs:?} ({mirror}) kept high -- {before:?} now, {held:?} after their best reply")
    };
    Some((m, claim))
}

// ---------------------------------------------------------------------------------------------------------------------
// THE FRAME'S MATE (Kaveh 2026-10-13: "the mate search has to be purposeful and guided by a need shown by the frames or
// lemmas so it's understandable"; "search inside the frame is ok"). Asked when the frame's goal is mate -- their king
// stands alone; the frame's claim is "the measure falls, or the goal", and this asks the goal -- a proof search for a
// forced mate whose moves of ours stay INSIDE THE FRAME:
// a check, or a statically safe move that does not raise the frame's measure; every reply of theirs; at most
// FRAME_MATE_DEPTH moves of ours. Proved, the line is played and explained move by move by their king's flight squares.
// (2026-10-12/13: from K+B+N v K starts with mate in five or fewer the planner wandered -- it had no quiet mate search,
// and the frames plateaued at 54-76% with the rest fifty-move draws in the final net.)
pub static FRAME_MATE_DEPTH: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(5);
#[pyfunction]
pub fn set_frame_mate_depth(d: u32) -> PyResult<()> { FRAME_MATE_DEPTH.store(d, std::sync::atomic::Ordering::Relaxed); Ok(()) }
pub const FRAME_MATE_BUDGET: u64 = 30_000;
pub static FRAME_MATE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);
/// (for digging: false lifts the inside-the-frame filter -- every safe move or check of ours)
pub static FRAME_MATE_INSIDE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);
#[pyfunction]
pub fn set_frame_mate(flag: bool) -> PyResult<()> { FRAME_MATE.store(flag, std::sync::atomic::Ordering::Relaxed); Ok(()) }

#[derive(Clone)]
pub struct MState { pos: Chess, mv: Option<Move>, ours: u32 }
struct FrameMate<'a> { by: Color, base: &'a [Comp], depth: u32 }
impl FrameMate<'_> {
    fn meas(&self, q: &Chess) -> Vec<u32> { self.base.iter().map(|c| cval(c, q, !self.by)).collect() }
}
/// Their king's flight squares (their turn on `q`).
fn flights(q: &Chess) -> usize { q.legal_moves().iter().filter(|m| m.role() == Role::King).count() }
impl crate::pns::Problem for FrameMate<'_> {
    type State = MState;
    fn judge(&self, s: &MState, or: bool, root: bool) -> Option<bool> {
        if !root && crate::game::claimable_draw(&s.pos).is_some() { return Some(false); }
        if or {
            if s.pos.legal_moves().is_empty() || s.ours >= self.depth { return Some(false); }
            None
        } else if s.pos.legal_moves().is_empty() { Some(s.pos.is_check()) } else { None }
    }
    fn expand(&self, s: &MState, or: bool) -> Vec<(MState, f64)> {
        if or {
            let m0 = self.meas(&s.pos);
            s.pos.legal_moves().into_iter().filter_map(|m| {
                let q = after(&s.pos, &m);
                let inside = q.is_check() || (static_safe(q.board(), self.by)
                    && (!FRAME_MATE_INSIDE.load(std::sync::atomic::Ordering::Relaxed) || self.meas(&q) <= m0));
                inside.then(|| { let w = 1.0 + 0.5 * flights(&q) as f64; (MState { pos: q, mv: Some(m), ours: s.ours + 1 }, w) })
            }).collect()
        } else {
            s.pos.legal_moves().into_iter().map(|m| (MState { pos: after(&s.pos, &m), mv: Some(m), ours: s.ours }, 1.0)).collect()
        }
    }
    fn and_size(&self, s: &MState) -> f64 { s.pos.legal_moves().len().max(1) as f64 }
    fn key(&self, s: &MState) -> u64 { crate::threat::zkey(&s.pos) }
}

/// frame_mate_of(fen, terms, inside) -> the frame's mate (the first move, the line) or None -- for digging.
#[pyfunction]
pub fn frame_mate_of(fen: String, terms: Vec<String>, inside: bool) -> Option<(String, String)> {
    let pos = parse(&fen)?;
    let base: Vec<Comp> = terms.iter().filter_map(|t| parse_term(t).map(Comp::T)).collect();
    let was = FRAME_MATE_INSIDE.swap(inside, std::sync::atomic::Ordering::Relaxed);
    let r = frame_mate(&pos, &base).map(|(m, w)| (m.to_uci(CastlingMode::Standard).to_string(), w));
    FRAME_MATE_INSIDE.store(was, std::sync::atomic::Ordering::Relaxed);
    r
}

/// The frame's mate from `pos` (us to move) with the frame `base`: the first move and the line in words, or None.
fn frame_mate(pos: &Chess, base: &[Comp]) -> Option<(Move, String)> {
    if !FRAME_MATE.load(std::sync::atomic::Ordering::Relaxed) { return None; }
    let by = pos.turn();
    // THE SHORTEST MATE FIRST -- mate in 1, then 2, ... (2026-10-13, the dig into seed 5006: proving "a mate within 5"
    // and playing any proof's first move, it proved "mate in 5" again by another line for thirty moves, their king
    // shuttling a8-b8 with mate 3-4 moves away; the shortest proof's continuation proves the next one shorter: every
    // move closes in)
    let mut found = None;
    for depth in 1..=FRAME_MATE_DEPTH.load(std::sync::atomic::Ordering::Relaxed) {
        let (r, lines) = crate::pns::search_proof_lines(&FrameMate { by, base, depth }, MState { pos: pos.clone(), mv: None, ours: 0 }, FRAME_MATE_BUDGET, 1);
        if r == Some(true) { found = lines.into_iter().next(); break; }
    }
    let line = found?;
    let first = line.get(1)?.mv.clone()?;
    let mut words = Vec::new();
    for w in line.windows(2) {
        let (a, b) = (&w[0], &w[1]);
        let Some(m) = &b.mv else { continue };
        let mut t = crate::plan::san(&a.pos, &m.to_uci(CastlingMode::Standard).to_string());
        if a.pos.turn() == by {
            t += &if b.pos.is_checkmate() { "#".to_string() }
                else { format!(" ({} flight square{} left{})", flights(&b.pos), if flights(&b.pos) == 1 { "" } else { "s" }, if b.pos.is_check() { ", check" } else { "" }) };
        }
        words.push(t);
    }
    let n = line.iter().filter(|s| s.pos.turn() != by).count();
    Some((first, format!("the frame's mate in {n}, every move inside the frame (a check, or its measure not rising): {}", words.join(" "))))
}


/// The generic frame's move for the side to move -> (the move, its trace line: the claim and whether it is certified).
/// Ahead: the WIN frame (its material's terms; none known -- the empty frame, whose only claim is the goal). Not ahead:
/// the HOLD frame.
pub fn material_frame_move(pos: &Chess) -> Option<(Move, String)> {
    let us = pos.turn();
    let key = material_key(pos.board(), us);
    if material(pos.board(), us) <= material(pos.board(), !us) {
        let net = |m: &Move| crate::plan::audit_net(pos, m);
        if let Some(x) = hold_move(pos, &key, &net) { return Some(x); }
        let m = least_loss(pos, &net)?;
        return Some((m.clone(), format!("hold ({key}, not ahead): no move keeps everything -- the least loss (net {})", net(&m))));
    }
    // A FRAME PER EPISODE KIND (Kaveh 2026-10-12): in check -- their threat on our king -- the material's CHECK frame
    // when one is known ("<material>:check"), its own measure of progress; else the material's frame
    let ckey = format!("{key}:check");
    // CONDITIONAL FRAMES (the challenger's answer; formerly "stages", after Sadikov & Bratko): "<material>:if:<term>>=<theta>" is played
    // while its condition holds (a trap the challenger found: its own measure, mined to LEAVE it); the check frame first
    let situational = || -> Option<(String, Vec<String>)> {
        let g = MATERIAL_FRAMES.read().unwrap();
        let m = g.as_ref()?;
        let prefix = format!("{key}:if:");
        let mut ks: Vec<&String> = m.keys().filter(|k| k.starts_with(&prefix)).collect();
        ks.sort();
        for k in ks {
            let cond = &k[prefix.len()..];
            // ("term>=theta", or "term<=theta" -- the surprise explorer's conditions go either way -- and several joined
            // by "&&": an exception refined, Ripple-Down style, by what tells it apart from the cases it broke)
            let holds = |c: &str| -> Option<bool> {
                let (t, th, at_most) = match c.rsplit_once(">=") { Some((t, th)) => (t, th, false),
                    None => { let (t, th) = c.rsplit_once("<=")?; (t, th, true) } };
                let (Some(term), Ok(th)) = (parse_term(t), th.parse::<u32>()) else { return None };
                let v = num(&term, pos, !us);
                Some(if at_most { v <= th } else { v >= th })
            };
            let all: Option<Vec<bool>> = cond.split("&&").map(holds).collect();
            if all.map_or(false, |v| v.iter().all(|x| *x)) { return Some((k.clone(), m[k].clone())); }
        }
        None
    };
    let (key, terms) = {
        let g = MATERIAL_FRAMES.read().unwrap();
        let m = g.as_ref();
        if pos.is_check() && m.map_or(false, |m| m.contains_key(&ckey)) { (ckey.clone(), m.unwrap()[&ckey].clone()) }
        else {
            drop(g);
            match situational() {
                Some(x) => x,
                None => { let g = MATERIAL_FRAMES.read().unwrap(); (key.clone(), g.as_ref().and_then(|m| m.get(&key).cloned()).unwrap_or_default()) }
            }
        }
    };
    let base: Vec<Comp> = terms.iter().filter_map(|t| parse_term(t).map(Comp::T)).collect();
    let horizon = horizon_of(&terms);
    let sp = Spec { base, cand: None, schema: "none".to_string(), safety: Safety::Material, horizon };
    let k_claim = horizon.unwrap_or(FRAME_K);
    // in play, a move must also pass the planner's own audit (its reasoning for them: nothing of ours lost, no draw);
    // none does -- the least loss, said so
    let audited = |m: &Move| crate::plan::audit_net(pos, m) >= 0;
    let chosen = sp.choose_cert(pos, &|q: &Chess| crate::game::seen(q), &audited);
    // THE NEED SHOWN BY THE FRAME: its claim is "the measure falls within FRAME_K moves, OR THE GOAL" -- with their king
    // alone the goal is mate, and whether the goal is provable inside the frame is the claim's own second half; asked
    // first (2026-10-13: asked only when no move certified progress, it ran from 1 of 20 starts within 10 plies of mate,
    // where it proves the mate from 12 -- the frame "certifies" a falling measure long after a mate is there)
    if (pos.board().by_color(!us) & !pos.board().kings()).is_empty() {
        if let Some(x) = frame_mate(pos, &sp.base) { return Some(x); }
    }
    let Some((m, cert)) = chosen else {
        let m = least_loss(pos, &|m: &Move| crate::plan::audit_net(pos, m))?;
        let net = crate::plan::audit_net(pos, &m);
        return Some((m, format!("no safe move for the {key} frame: the least loss (net {net}) -- the claim is the goal within {k_claim} moves")));
    };
    let before = sp.measure(pos, !us);
    let q = after(pos, &m);
    let claim = if terms.is_empty() {
        format!("the empty frame (no measure known for {key}): the claim is the goal within {} moves (mate, or a capture we keep)", crate::vocab::FRAME_K)
    } else {
        format!("the {key} frame {:?}: the measure {:?} falls within {} moves against every reply, or the goal (mate, or a capture we keep) -- {}",
                terms.iter().filter(|t| !t.starts_with("horizon:")).collect::<Vec<_>>(), before, k_claim,
                if cert { format!("certified (worst after it {:?})", sp.worst(&q, !us)) } else { "NOT certified".to_string() })
    };
    Some((m, claim))
}
/// THE LEAST LOSS when no move keeps everything: the best net; among equal nets the move handing over the least material,
/// then one that does NOT end the game (Kaveh: "when losing, drag the loss out as long as possible ... same thing with
/// drawing" -- 2026-10-12, the clean run: four games stalemated by the tie, Rxd7 stalemating where giving up the rook
/// played on in K+R v K+R; the carry2 run then showed playing on a rook down in K+R v K+R LOSES -- "first, don't lose"
/// ranks before dragging it out, so a stalemate now beats a rook handed over at an equal net).
fn least_loss(pos: &Chess, net: &dyn Fn(&Move) -> i64) -> Option<Move> {
    // FIRST, DON'T LOSE (Kaveh 2026-10-12: "first, don't lose, and then try to win"; then "drag the loss out"): among
    // equal nets, the move handing over the least material (their biggest static exchange on a piece of ours after it),
    // then a move not ending the game. A draw they may claim keeps our pieces; a piece handed over plays on a piece down
    // and can lose (2026-10-12, the carry2 run, seed 2097: Kd2 dropped the d5 rook rather than allow Rb4's threefold,
    // both "net -5" -- K+R v K+R then lost; the same principle was applied before only at the fifty-move clock's end,
    // where five rooks were thrown to reset it). A move ending the game hands over nothing.
    let us = pos.turn();
    pos.legal_moves().into_iter().max_by_key(|m| {
        let q = after(pos, m);
        let ends = !q.is_checkmate() && (q.legal_moves().is_empty() || crate::game::drawn(&q).is_some());
        let risk = if ends { 0 } else {
            (q.board().by_color(us) & !q.board().kings()).into_iter().map(|s| crate::rules::see(q.board(), s, !us)).max().unwrap_or(0)
        };
        (net(m), -risk, !ends)
    })
}
/// The horizon of a generic frame's claim (moves of ours).
pub const FRAME_K: u32 = 2;
/// The horizon a move's claim is certified within in play (moves of ours; 2 = FRAME_K) -- settable, to measure it
/// (Kaveh 2026-10-13: "why did we pick two ... the effect of a deeper claim horizon").
pub static CLAIM_K: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(2);
#[pyfunction]
pub fn set_claim_k(k: u32) -> PyResult<()> { CLAIM_K.store(k, std::sync::atomic::Ordering::Relaxed); Ok(()) }

// ---------------------------------------------------------------------------------------------------------------------
// THE LEMMA-STATE (Kaveh 2026-10-12: "moving to a new place where new lemmas are applying is progress. Cycling back to a
// place where the same lemmas apply is a loop ... detect a loop and decide that we are looping"): which base facts hold
// -- each named square (W our king, L theirs, P our other piece, O theirs) on the edge, in a corner, attacked by us,
// attacked by them; who stands next to whom; a piece of ours attacked, a piece of theirs attacked. Each fact is a
// membership in a base set (Lean Ideas.Base.Vocabulary: edge, corner, attackedBy, box, piecesOf) -- nothing learned,
// nothing chess beyond the base. The side to move is "us" (the lone side is the other).

const NAMED: [(&str, Term); 4] = [("our king", Term::W), ("their king", Term::L), ("our other piece", Term::P), ("their other piece", Term::O)];

/// -> [(the fact in words, whether it holds)] on the board.
pub fn lemma_facts(q: &Chess) -> Vec<(String, bool)> {
    let lone = !q.turn();
    let mut out = Vec::new();
    let sets = [("on the edge", Term::Edge), ("in a corner", Term::Corner), ("attacked by us", Term::Attacked), ("attacked by them", Term::Hit)];
    let sq_of = |t: &Term| sq(t, q, lone);
    for (name, t) in NAMED.iter() {
        let s = sq_of(t);
        for (what, set_t) in sets.iter() {
            out.push((format!("{name} {what}"), s.map_or(false, |x| set(set_t, q, lone).contains(x))));
        }
    }
    for i in 0..NAMED.len() {
        for j in i + 1..NAMED.len() {
            let (a, b) = (sq_of(&NAMED[i].1), sq_of(&NAMED[j].1));
            let near = match (a, b) { (Some(x), Some(y)) => x.distance(y) <= 1, _ => false };
            out.push((format!("{} next to {}", NAMED[i].0, NAMED[j].0), near));
        }
    }
    out.push(("a piece of ours attacked by them".to_string(), (set(&Term::Ours, q, lone) & set(&Term::Hit, q, lone)).any()));
    out.push(("a piece of theirs attacked by us".to_string(), (set(&Term::Theirs, q, lone) & set(&Term::Attacked, q, lone)).any()));
    out
}

/// lemma_state(fens) -> per board, the facts that hold as a bitmask (the order of lemma_facts).
#[pyfunction]
pub fn lemma_state(fens: Vec<String>) -> PyResult<Vec<u64>> {
    Ok(fens.iter().map(|f| parse(f).map_or(0, |p| lemma_facts(&p).iter().enumerate().fold(0u64, |m, (i, (_, v))| if *v { m | (1 << i) } else { m }))).collect())
}

/// lemma_facts_of(fen) -> the facts in words with their values (for reading a state).
#[pyfunction]
pub fn lemma_facts_of(fen: String) -> PyResult<Vec<(String, bool)>> {
    Ok(parse(&fen).map_or(vec![], |p| lemma_facts(&p)))
}

// ---------------------------------------------------------------------------------------------------------------------
// THE CHALLENGER'S CHECK (Kaveh 2026-10-12: "you need your reasoner to reason both sides"; the literature, red-teamed: a
// frame is a ranking certificate, a counter-frame its dual -- a TRAP, checked LOCALLY, never by replaying an oracle): R =
// {term >= theta} is a one-step trap against the frame's measure at a position of R when EVERY safe move of ours (nothing
// left to the static exchange) has a reply landing back in R with the measure no lower than here -- the attacker can
// neither make progress nor leave; a mate or a capture of ours that the exchange keeps escapes it. (Lean hold_blocks is
// the trap's theorem; this is its check on boards.)

/// trap_check(fens, frame_terms, region_term, theta) -> per board: None (not in R), Some(the trap holds here).
#[pyfunction]
pub fn trap_check(fens: Vec<String>, frame_terms: Vec<String>, region_term: String, theta: u32) -> PyResult<Vec<Option<bool>>> {
    use rayon::prelude::*;
    let Some(rt) = parse_term(&region_term) else { return Ok(vec![]) };
    let base: Vec<Comp> = frame_terms.iter().filter_map(|t| comp(t)).collect();
    let meas = |q: &Chess, lone: Color| -> Vec<u32> { base.iter().map(|c| cval(c, q, lone)).collect() };
    Ok(fens.par_iter().map(|f| {
        let p = parse(f)?;
        let us = p.turn();
        if num(&rt, &p, !us) < theta { return None; }
        let m0 = meas(&p, !us);
        let theirs0 = material(p.board(), !us);
        let safe: Vec<Move> = p.legal_moves().into_iter().filter(|m| {
            let q = after(&p, m);
            q.is_checkmate() || (!q.legal_moves().is_empty() && static_safe(q.board(), us))
        }).collect();
        for m in safe {
            let q = after(&p, &m);
            if q.is_checkmate() { return Some(false); }                       // the attacker escapes: mate
            if material(q.board(), !us) < theirs0 { return Some(false); }      // a capture kept (static exchange): escapes
            let held = q.legal_moves().iter().any(|r| {
                let x = after(&q, r);
                num(&rt, &x, !us) >= theta && !(meas(&x, !us) < m0)
            });
            if !held { return Some(false); }
        }
        Some(true)
    }).collect())
}


// ---------------------------------------------------------------------------------------------------------------------
// THE READ SET (ANALYST_PLAN 39.5 item 4): every square a term's value can depend on. The reasoning ledger retracts a finding when a
// move touches a square in its read set, with no chess knowledge. A conservative cover: a slider's attack reads its attack set (the
// squares up to and including the first piece in each direction); an exchange on x reads the walk of every line from x through the
// pieces that could join it (x-ray batteries), the knight squares and the king's neighbours; `without` / `after` read the board that
// is left. A word not listed reads the whole board.

/// The squares an exchange on `x` can look at.
pub(crate) fn exchange_reads(b: &shakmaty::Board, x: Square) -> Bitboard {
    let mut out = Bitboard::from(x) | shakmaty::attacks::knight_attacks(x) | shakmaty::attacks::king_attacks(x);
    for (df, dr) in [(1i32, 0i32), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)] {
        let diag = df != 0 && dr != 0;
        let (mut f, mut r) = (x.file() as i32 + df, x.rank() as i32 + dr);
        while (0..8).contains(&f) && (0..8).contains(&r) {
            let s = Square::from_coords(shakmaty::File::new(f as u32), shakmaty::Rank::new(r as u32));
            out |= Bitboard::from(s);
            if let Some(p) = b.piece_at(s) {
                let joins = match p.role { shakmaty::Role::Queen => true, shakmaty::Role::Rook => !diag, shakmaty::Role::Bishop | shakmaty::Role::Pawn => diag, _ => false };
                if !joins { break; }
            }
            f += df; r += dr;
        }
    }
    out
}

pub(crate) fn reads(t: &Term, q: &Chess, lone: Color) -> Bitboard {
    let b = q.board();
    let r = |x: &Term| reads(x, q, lone);
    let at = |x: &Term| sq(x, q, lone).map_or(Bitboard::EMPTY, Bitboard::from);
    match t {
        Term::Named(_) | Term::Cls(_) => at(t),
        Term::Int(_) | Term::Edge | Term::Corner | Term::Center | Term::Light | Term::Dark => Bitboard::EMPTY,
        Term::At(a) => r(a) | at(a),
        Term::Attacks(a) => r(a) | at(a) | sq(a, q, lone).map_or(Bitboard::EMPTY, |s| b.attacks_from(s)),
        Term::Between(a, c) => r(a) | r(c) | at(a) | at(c) | match (sq(a, q, lone), sq(c, q, lone)) {
            (Some(x), Some(y)) => shakmaty::attacks::between(x, y), _ => Bitboard::EMPTY },
        Term::See(a) | Term::Attackers(a) | Term::Defenders(a) => r(a) | sq(a, q, lone).map_or(Bitboard::EMPTY, |s| exchange_reads(b, s)),
        Term::Moves(a) => r(a) | sq(a, q, lone).map_or(Bitboard::EMPTY, |s| {
            let own_king = b.piece_at(s).and_then(|p| b.king_of(p.color)).map_or(Bitboard::EMPTY, |k| exchange_reads(b, k));
            exchange_reads(b, s) | own_king }),
        Term::Without(x, a) => r(x) | set(x, q, lone) | without_pos(x, q, lone).map_or(Bitboard::FULL, |p| reads(a, &p, lone)),
        Term::After(f, t2, a) => r(f) | r(t2) | at(f) | at(t2) | after_pos(f, t2, q, lone).map_or(Bitboard::FULL, |p| reads(a, &p, lone)),
        Term::Least(_) | Term::Most(_) => Bitboard::FULL,                                            // (every reply's board)
        Term::Mobility(a) | Term::Attackable(a, _) => r(a) | at(a) | Bitboard::FULL,
        Term::Declined(a) => r(a) | Bitboard::FULL,                                                  // (every exchange but one)        // (the escape squares and the routes read most of the board)
        Term::Worth(a) => r(a) | at(a),
        Term::MaxWorth(a) => r(a) | Bitboard::FULL,                                                 // (every piece of the set is read)
        Term::Count(a) | Term::Not(a) | Term::Delta(a) | Term::Gained(a) | Term::Lost(a) => r(a),
        Term::And(a, c) | Term::Or(a, c) | Term::Sum(a, c) | Term::Min(a, c) | Term::Max(a, c) | Term::Diff(a, c) | Term::Eq(a, c) => r(a) | r(c),
        Term::If(c, a, d) => r(c) | r(a) | r(d),
        _ => Bitboard::FULL,
    }
}

/// vocab_reads(fen, term) -> (the value, the squares read as a list of names, absolute) -- a fact with its read set
#[pyfunction]
pub fn vocab_reads(fen: String, term: String) -> PyResult<(u32, Vec<String>)> {
    let (Some(p), Some(t)) = (parse(&fen), parse_term(&term)) else { return Err(pyo3::exceptions::PyValueError::new_err("bad fen or term")) };
    let lone = !p.turn();
    Ok((num(&t, &p, lone), reads(&t, &p, lone).into_iter().map(|s| s.to_string()).collect()))
}
