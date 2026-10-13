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
}

/// The strongest piece of `c` other than its king -- by kind (queen, rook, bishop, knight, pawn), ties by the lower square.
fn strongest(b: &shakmaty::Board, c: Color) -> Option<Square> {
    let rank = |r: Role| match r { Role::Queen => 5, Role::Rook => 4, Role::Bishop => 3, Role::Knight => 2, Role::Pawn => 1, Role::King => 0 };
    (b.by_color(c) & !b.kings()).into_iter().max_by_key(|s| (b.role_at(*s).map(rank).unwrap_or(0), std::cmp::Reverse(*s as u8)))
}

/// Parse an S-expression term.
pub fn parse_term(s: &str) -> Option<Term> {
    let toks: Vec<String> = s.replace('(', " ( ").replace(')', " ) ").split_whitespace().map(|x| x.to_string()).collect();
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
            _ => tok.parse::<Square>().ok().map(Term::Named) };
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
        _ => return None,
    })
}

fn kd(a: Square, b: Square) -> u32 { a.distance(b) }

/// A square term on the board (`lone` the lone king's colour).
fn sq(t: &Term, q: &Chess, lone: Color) -> Option<Square> {
    let b = q.board();
    match t {
        Term::W => b.king_of(!lone), Term::L => b.king_of(lone),
        // "our / their other piece": the STRONGEST other piece by kind (queen, rook, bishop, knight, pawn), ties by square
        // (2026-10-13: it was the first by square -- in K+B+N v K the bishop or the knight by where they stood, so "the
        // colour of P" meant nothing half the time)
        Term::P => strongest(b, !lone),
        Term::O => strongest(b, lone),
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
        Term::Ours => b.by_color(!lone) & !b.kings(),
        Term::Theirs => b.by_color(lone) & !b.kings(),
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
        _ => Bitboard::EMPTY,
    }
}

/// A number term on the board.
pub fn num(t: &Term, q: &Chess, lone: Color) -> u32 {
    match t {
        Term::Count(a) => set(a, q, lone).count() as u32,
        Term::Dist(a, c) => match (sq(a, q, lone), sq(c, q, lone)) { (Some(x), Some(y)) => kd(x, y), _ => 0 },
        Term::Sum(a, c) => num(a, q, lone) + num(c, q, lone),
        _ => 0,
    }
}

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
    Ok(Some(match t { Term::Count(_) | Term::Dist(_, _) | Term::Sum(_, _) => num(&t, &p, lone), _ => set(&t, &p, lone).count() as u32 }))
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

struct Spec { base: Vec<Comp>, cand: Option<Term>, schema: String, safety: Safety }

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
    let base: Vec<Comp> = base.iter().map(|s| comp(s)).collect::<Option<_>>()?;
    let cand = match cand { Some(c) => Some(parse_term(c)?), None => None };
    let safety = match piece { "Q" | "q" => Safety::Piece(Role::Queen), "*" => Safety::Material, _ => Safety::Piece(Role::Rook) };
    Some(Spec { base, cand, schema: schema.to_string(), safety })
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
            if self.within(&after(pos, m), &before, us, CLAIM_K.load(std::sync::atomic::Ordering::Relaxed)) { return Some((m.clone(), true)); }
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
    if !terms.iter().all(|t| parse_term(t).is_some()) { return Ok(false); }
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
    // SITUATIONAL FRAMES (the challenger's answer, a stage -- Sadikov & Bratko): "<material>:if:<term>>=<theta>" is played
    // while its condition holds (a trap the challenger found: its own measure, mined to LEAVE it); the check frame first
    let situational = || -> Option<(String, Vec<String>)> {
        let g = MATERIAL_FRAMES.read().unwrap();
        let m = g.as_ref()?;
        let prefix = format!("{key}:if:");
        let mut ks: Vec<&String> = m.keys().filter(|k| k.starts_with(&prefix)).collect();
        ks.sort();
        for k in ks {
            let cond = &k[prefix.len()..];
            let (t, th) = cond.rsplit_once(">=")?;
            let (Some(term), Ok(th)) = (parse_term(t), th.parse::<u32>()) else { continue };
            if num(&term, pos, !us) >= th { return Some((k.clone(), m[k].clone())); }
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
    let sp = Spec { base, cand: None, schema: "none".to_string(), safety: Safety::Material };
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
        return Some((m, format!("no safe move for the {key} frame: the least loss (net {net}) -- the claim is the goal within {FRAME_K} moves")));
    };
    let before = sp.measure(pos, !us);
    let q = after(pos, &m);
    let claim = if terms.is_empty() {
        format!("the empty frame (no measure known for {key}): the claim is the goal within {} moves (mate, or a capture we keep)", crate::vocab::FRAME_K)
    } else {
        format!("the {key} frame {:?}: the measure {:?} falls within {} moves against every reply, or the goal (mate, or a capture we keep) -- {}",
                terms, before, FRAME_K, if cert { format!("certified (worst after it {:?})", sp.worst(&q, !us)) } else { "NOT certified".to_string() })
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
