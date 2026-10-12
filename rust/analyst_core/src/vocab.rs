//! THE BASE VOCABULARY, EVALUATED (Kaveh 2026-10-12: "the base has to be deterministic and interpretable"). A term over
//! the base -- each primitive the mirror of a definition in Lean `Ideas.Vocabulary` -- read on a board, and THE
//! CERTIFIER: how many stuck positions a frame's measure, extended by a candidate term, certifies (progress against every
//! reply). The proposer (catspace/incubator/analyst/proposer.py) builds the terms and reads the winners back in words.
//!
//! Terms are S-expressions. Squares: W (our king), L (the lone king), P (our other piece). Sets: edge, corner, attacked
//! (the squares we attack, the lone king lifted), free (the rest), (box S), (rank S), (file S), (line S S), (between S S),
//! (nearer S S), (and A B), (or A B), (not A), (reach S A). Numbers: (count A), (dist S S), (sum N N).
use super::*;
use shakmaty::Bitboard;

#[derive(Clone, Debug)]
pub enum Term {
    W, L, P,
    Edge, Corner, Attacked, Free,
    Box1(Box<Term>), Rank(Box<Term>), File(Box<Term>),
    Line(Box<Term>, Box<Term>), Between(Box<Term>, Box<Term>), Nearer(Box<Term>, Box<Term>),
    And(Box<Term>, Box<Term>), Or(Box<Term>, Box<Term>), Not(Box<Term>), Reach(Box<Term>, Box<Term>),
    Count(Box<Term>), Dist(Box<Term>, Box<Term>), Sum(Box<Term>, Box<Term>),
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
        return match tok.as_str() { "W" => Some(Term::W), "L" => Some(Term::L), "P" => Some(Term::P), "edge" => Some(Term::Edge),
            "corner" => Some(Term::Corner), "attacked" => Some(Term::Attacked), "free" => Some(Term::Free), _ => None };
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
        _ => return None,
    })
}

fn kd(a: Square, b: Square) -> u32 { a.distance(b) }

/// A square term on the board (`lone` the lone king's colour).
fn sq(t: &Term, q: &Chess, lone: Color) -> Option<Square> {
    let b = q.board();
    match t {
        Term::W => b.king_of(!lone), Term::L => b.king_of(lone),
        Term::P => (b.by_color(!lone) & !b.kings()).first(),
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
