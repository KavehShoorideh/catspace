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

struct Spec { base: Vec<Comp>, cand: Option<Term>, schema: String, piece: Role }

impl Spec {
    fn measure(&self, q: &Chess, lone: Color) -> Vec<u32> {
        let mut v: Vec<u32> = self.base.iter().map(|c| cval(c, q, lone)).collect();
        if let Some(t) = &self.cand {
            let x = num(t, q, lone);
            match self.schema.as_str() {
                "sum" => { if let Some(l) = v.last_mut() { *l += x } else { v.push(x) } }
                "replace" => { if let Some(l) = v.last_mut() { *l = x } else { v.push(x) } }
                "append" => v.push(x),
                _ => {}
            }
        }
        v
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
                y.is_checkmate() || (crate::mating::safe_step(&y, us, self.piece) && self.within(&y, before, us, k - 1))
            })
        })
    }
    fn certified(&self, pos: &Chess, k: u32) -> bool {
        let us = pos.turn();
        let before = self.measure(pos, !us);
        pos.legal_moves().iter().any(|m| {
            let q = after(pos, m);
            q.is_checkmate() || (crate::mating::safe_step(&q, us, self.piece) && self.within(&q, &before, us, k))
        })
    }
}

fn spec(base: &[String], cand: &Option<String>, schema: &str, piece: &str) -> Option<Spec> {
    let base: Vec<Comp> = base.iter().map(|s| comp(s)).collect::<Option<_>>()?;
    let cand = match cand { Some(c) => Some(parse_term(c)?), None => None };
    let piece = match piece { "Q" | "q" => Role::Queen, _ => Role::Rook };
    Some(Spec { base, cand, schema: schema.to_string(), piece })
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
    fn choose(&self, pos: &Chess, seen: &std::collections::HashMap<u64, u8>) -> Option<Move> { self.choose_cert(pos, seen).map(|x| x.0) }
    /// -> (the move, whether it carries a certificate: the measure falls against every reply, at once or within two)
    fn choose_cert(&self, pos: &Chess, seen: &std::collections::HashMap<u64, u8>) -> Option<(Move, bool)> {
        let us = pos.turn();
        let before = self.measure(pos, !us);
        let fresh = |q: &Chess| seen.get(&crate::threat::zkey(q)).copied().unwrap_or(0) < 2;
        let mut cands: Vec<(Vec<u32>, Vec<u32>, bool, Move)> = Vec::new();
        for m in pos.legal_moves() {
            let q = after(pos, &m);
            if q.is_checkmate() { return Some((m, true)); }
            if !crate::mating::safe_step(&q, us, self.piece) || !fresh(&q) { continue; }
            cands.push((self.worst(&q, !us), self.measure(&q, !us), seen.contains_key(&crate::threat::zkey(&q)), m));
        }
        cands.sort_by(|a, b| (&a.0, &a.1, a.2).cmp(&(&b.0, &b.1, b.2)));
        let first = cands.first()?;
        if first.0 < before { return Some((first.3.clone(), true)); }
        for (_, _, _, m) in &cands {
            if self.within(&after(pos, m), &before, us, 2) { return Some((m.clone(), true)); }
        }
        Some((first.3.clone(), false))
    }
    fn defend(&self, q: &Chess) -> Option<Move> {
        let lone = q.turn();
        q.legal_moves().into_iter().filter(|r| {
            let x = after(q, r);
            !x.legal_moves().iter().any(|m| after(&x, m).is_checkmate())
        }).max_by_key(|r| (self.measure(&after(q, r), lone), crate::threat::zkey(&after(q, r))))
            .or_else(|| q.legal_moves().into_iter().next())
    }
    /// -> (Some(moves to mate) or None -- no mate within the limit, a repetition, a stalemate, the piece lost; our moves
    /// played; how many of them carried a certificate)
    fn play(&self, start: &Chess, limit: u32) -> (Option<u32>, u32, u32) {
        let mut p = start.clone();
        let mut seen: std::collections::HashMap<u64, u8> = std::collections::HashMap::new();
        let (mut played, mut cert) = (0u32, 0u32);
        for n in 1..=limit {
            *seen.entry(crate::threat::zkey(&p)).or_insert(0) += 1;
            let Some((m, c)) = self.choose_cert(&p, &seen) else { return (None, played, cert) };
            played += 1; cert += c as u32;
            p = after(&p, &m);
            if p.is_checkmate() { return (Some(n), played, cert); }
            if p.legal_moves().is_empty() { return (None, played, cert); }
            *seen.entry(crate::threat::zkey(&p)).or_insert(0) += 1;
            if seen[&crate::threat::zkey(&p)] >= 3 { return (None, played, cert); }
            let Some(r) = self.defend(&p) else { return (None, played, cert) };
            p = after(&p, &r);
            if (p.board().by_color(p.turn()) & !p.board().kings()).is_empty() { return (None, played, cert); }   // our piece taken
            if seen.get(&crate::threat::zkey(&p)).copied().unwrap_or(0) >= 2 { return (None, played, cert); }
        }
        (None, played, cert)
    }
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
