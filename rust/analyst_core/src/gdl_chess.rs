//! PLAN 76.4b THE CHESS BINDING OF THE REGRESSION, IN RUST (the mirror of discovery/gdl_chess.py `Binding`, plan 76.4a): a board's reading as the spec's facts,
//! the certified relations read through the kernel, the next state, the valuation laws -- and THE ENGINE: rules/chess.gdl loaded ONCE from its text (handed in
//! by gdl_chess.py at its first use: the file is the source, no clause is written here), the store's DEFECT_FIX bindings and the GOAL entries registered as data,
//! the regressions cached per (the board's key, the goal).
//!
//! THE BINDING TABLE is the ontology shelf's (ontology.py: each fluent / certified relation of the spec has one entry with its `rust` binding): Python hands
//! (relation, the entry's `rust` name) pairs and every one is looked up in READERS below -- a relation of the spec with no reader is an error at load. The
//! readers read the board as python-chess holds it (pyboard.rs; gdl_chess.py reads python-chess):
//!   rules.rs:cell        (cell ?f ?r ?c ?k)    the units, python-chess's `piece_map()` order (squares descending) or the one square asked
//!   pos.rs:turn_white    (control ?c)          the side to move
//!   pos.rs:tkey          (castling_right ?c ?side)   `clean_castling_rights()`, its squares ascending (a1 h1 a8 h8 -> queenside / kingside)
//!   pos.rs:ep_square     (ep_square ?f ?r)     the RAW square (set after every double push)
//!   game.rs:halfmoves    (halfmove ?n)         min(the clock, the limit)
//!   game.rs:seen         (record ?n)           no game record in a regression: 0
//!   pos.rs:push          (does ?c ?m)          a state reading has no joint move: no rows
//!   pos.rs:legal_ucis    (legal ?c ?m)         CERTIFIED (Ideas.LegalEquiv.legal_iff): python-chess's legal moves in its order, then the other side's noop
//!   pos.rs:attacks       (attack_by ?k ?c ?f1 ?r1 ?f2 ?r2)   CERTIFIED (Ideas.LegalEquiv.attackBy_iff): `attackers` / `attacks`, squares ascending
//! the next state (Ideas.NextEquiv.applyMoveSpec_eq_applyMove): python-chess's `push`; the valuation laws: 'exchange' (the word `see`, the mirror of Lean
//! Ideas.ExchangeLaw, on graph.pos_of's position: None where shakmaty does not hold it) and 'exchange from' (the target's worth by the store's valuation less
//! what the other side wins back after that capture).
use crate::gdl::{Interner, Program, Rel, Row, T, ARITY_MAX, FREE};
use crate::pyboard::{fwd, to_chess, PBoard, PMove, BISHOP, KING, KNIGHT, PAWN, QUEEN, ROOK};
use crate::regression::{self as RG, Atom, Binding, Field, Goal, Leaf, Named, NameEntry, Regression, Regressor, Spec, DISTINCT, LK, ND};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rustc_hash::FxHashMap;
use std::collections::BTreeSet;
use std::sync::Mutex;

fn verr(s: String) -> PyErr { PyValueError::new_err(s) }

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Reader { Cell, Control, Castling, Ep, Halfmove, Record, NoMove, Legal, AttackBy }

const READERS: [(&str, Reader); 9] = [
    ("rules.rs:cell", Reader::Cell), ("pos.rs:turn_white", Reader::Control), ("pos.rs:tkey", Reader::Castling), ("pos.rs:ep_square", Reader::Ep),
    ("game.rs:halfmoves", Reader::Halfmove), ("game.rs:seen", Reader::Record), ("pos.rs:push", Reader::NoMove), ("pos.rs:legal_ucis", Reader::Legal),
    ("pos.rs:attacks", Reader::AttackBy),
];

/// An action of the chess binding: a role's move (the term and the move it names).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Act { pub white: bool, pub term: T, pub m: PMove }

/// The chess binding (gdl_chess.py `Binding`): term ids of the spec's constants, the move terms, the readers by relation.
pub struct ChessBinding {
    files: [T; 8], ranks: [T; 8], white: T, black: T, kinds: [T; 7], noop: T, kingside: T, queenside: T,
    clock: Vec<T>, zero: T,
    file_of: Vec<i8>, rank_of: Vec<i8>,
    move_term: Vec<T>,                          // (from * 64 + to) * 5 + promotion index (0 none, 1 knight .. 4 queen)
    move_of: FxHashMap<T, PMove>,
    readers: Vec<Option<Reader>>,
    clock_limit: u32,
}                                               // (plan 76.5: the 'exchange from' law's worths are the valuation's, valuation.rs)

fn promo_index(p: u8) -> usize { match p { KNIGHT => 1, BISHOP => 2, ROOK => 3, QUEEN => 4, _ => 0 } }

impl ChessBinding {
    fn new(terms: &mut Interner, prog: &Program, bindings: &[(String, String)], clock_limit: u32) -> Result<ChessBinding, String> {
        let files: [T; 8] = std::array::from_fn(|i| terms.sym(&((b'a' + i as u8) as char).to_string()));
        let ranks: [T; 8] = std::array::from_fn(|i| terms.sym(&(i + 1).to_string()));
        let kind_names = ["", "pawn", "knight", "bishop", "rook", "queen", "king"];
        let kinds: [T; 7] = std::array::from_fn(|i| if i == 0 { FREE } else { terms.sym(kind_names[i]) });
        let (white, black, noop) = (terms.sym("white"), terms.sym("black"), terms.sym("noop"));
        let (kingside, queenside) = (terms.sym("kingside"), terms.sym("queenside"));
        let clock: Vec<T> = (0..=clock_limit.max(2)).map(|n| terms.sym(&n.to_string())).collect();
        let zero = clock[0];
        let (fm, fp) = (terms.sym("move"), terms.sym("promote"));
        let mut move_term = vec![FREE; 64 * 64 * 5];
        let mut move_of = FxHashMap::default();
        for from in 0..64u8 {
            for to in 0..64u8 {
                let (f1, r1, f2, r2) = (files[(from & 7) as usize], ranks[(from >> 3) as usize], files[(to & 7) as usize], ranks[(to >> 3) as usize]);
                let t = terms.fun(fm, &[f1, r1, f2, r2]);
                move_term[(from as usize * 64 + to as usize) * 5] = t;
                move_of.insert(t, PMove::new(from, to, 0));
                for p in [KNIGHT, BISHOP, ROOK, QUEEN] {
                    let t = terms.fun(fp, &[f1, r1, f2, r2, kinds[p as usize]]);
                    move_term[(from as usize * 64 + to as usize) * 5 + promo_index(p)] = t;
                    move_of.insert(t, PMove::new(from, to, p));
                }
            }
        }
        let n = terms.len();
        let mut file_of = vec![-1i8; n];
        let mut rank_of = vec![-1i8; n];
        for i in 0..8 { file_of[files[i] as usize] = i as i8; rank_of[ranks[i] as usize] = i as i8; }
        let mut readers = vec![None; prog.nrels()];
        for (rel, rust) in bindings {
            let Some(r) = prog.rel_id(rel) else { return Err(format!("the shelf binds {rel}, no relation of the spec")) };
            let Some((_, rd)) = READERS.iter().find(|(nm, _)| nm == rust) else { return Err(format!("{rel}: its Rust binding {rust} has no reader in gdl_chess.rs")) };
            readers[r as usize] = Some(*rd);
        }
        Ok(ChessBinding { files, ranks, white, black, kinds, noop, kingside, queenside, clock, zero, file_of, rank_of, move_term, move_of, readers, clock_limit })
    }

    fn colour(&self, white: bool) -> T { if white { self.white } else { self.black } }
    fn sq_terms(&self, s: u8) -> (T, T) { (self.files[(s & 7) as usize], self.ranks[(s >> 3) as usize]) }
    fn square(&self, f: T, r: T) -> Option<u8> {
        let fi = *self.file_of.get(f as usize)?;
        let ri = *self.rank_of.get(r as usize)?;
        if fi < 0 || ri < 0 { None } else { Some((ri as u8) * 8 + fi as u8) }
    }
    pub fn move_term_of(&self, m: PMove) -> T { self.move_term[(m.from as usize * 64 + m.to as usize) * 5 + promo_index(m.promo)] }
    fn role_white(&self, t: T) -> Option<bool> { if t == self.white { Some(true) } else if t == self.black { Some(false) } else { None } }

    fn cell_row(&self, s: u8, t: u8, w: bool) -> Row {
        let (f, r) = self.sq_terms(s);
        let mut row = [FREE; ARITY_MAX];
        row[0] = f; row[1] = r; row[2] = self.colour(w); row[3] = self.kinds[t as usize];
        row
    }

    fn see_on(&self, x: &PBoard, s: u8) -> Option<i64> {
        let p = to_chess(x)?;
        let us = x.turn;
        let sq = shakmaty::Square::new(s as u32);
        let named = if us { sq } else { sq.flip_vertical() };
        let c = if us { shakmaty::Color::White } else { shakmaty::Color::Black };
        Some(crate::vocab::num(&crate::vocab::Term::See(Box::new(crate::vocab::Term::Named(named))), &p, !c) as i64)
    }
}

fn row1(a: T) -> Row { let mut r = [FREE; ARITY_MAX]; r[0] = a; r }
fn row2(a: T, b: T) -> Row { let mut r = [FREE; ARITY_MAX]; r[0] = a; r[1] = b; r }
fn fits(row: &Row, pat: &Row, n: usize) -> bool { (0..n).all(|i| pat[i] == FREE || pat[i] == row[i]) }

impl Binding for ChessBinding {
    type X = PBoard;
    type A = Act;

    fn input(&self, x: &PBoard, rel: Rel, pat: &Row) -> Vec<Row> {
        match self.readers.get(rel as usize).copied().flatten() {
            Some(Reader::Cell) => {
                if pat[0] != FREE && pat[1] != FREE {
                    let Some(s) = self.square(pat[0], pat[1]) else { return vec![] };
                    let Some((t, w)) = x.piece_at(s) else { return vec![] };
                    let row = self.cell_row(s, t, w);
                    return if fits(&row, pat, 4) { vec![row] } else { vec![] };
                }
                let mut out = vec![];
                let occ = x.occupied();
                for s in (0..64u8).rev() {
                    if occ & (1u64 << s) == 0 { continue; }
                    let Some((t, w)) = x.piece_at(s) else { continue };
                    let row = self.cell_row(s, t, w);
                    if fits(&row, pat, 4) { out.push(row); }
                }
                out
            }
            Some(Reader::Control) => { let row = row1(self.colour(x.turn)); if fits(&row, pat, 1) { vec![row] } else { vec![] } }
            Some(Reader::Castling) => {
                let mut out = vec![];
                for s in fwd(x.clean_castling_rights()) {
                    let row = match s { 7 => row2(self.white, self.kingside), 0 => row2(self.white, self.queenside), 63 => row2(self.black, self.kingside),
                                        56 => row2(self.black, self.queenside), _ => continue };
                    if fits(&row, pat, 2) { out.push(row); }
                }
                out
            }
            Some(Reader::Ep) => {
                let Some(e) = x.ep else { return vec![] };
                let (f, r) = self.sq_terms(e);
                let row = row2(f, r);
                if fits(&row, pat, 2) { vec![row] } else { vec![] }
            }
            Some(Reader::Halfmove) => {
                let row = row1(self.clock[x.halfmove.min(self.clock_limit) as usize]);
                if fits(&row, pat, 1) { vec![row] } else { vec![] }
            }
            Some(Reader::Record) => if pat[0] == FREE || pat[0] == self.zero { vec![row1(self.zero)] } else { vec![] },
            _ => vec![],
        }
    }

    fn certified(&self, x: &PBoard, rel: Rel, pat: &Row) -> Vec<Row> {
        match self.readers.get(rel as usize).copied().flatten() {
            Some(Reader::Legal) => {
                let mover = self.colour(x.turn);
                let mut out = vec![];
                for m in x.legal_moves() {
                    let row = row2(mover, self.move_term_of(m));
                    if fits(&row, pat, 2) { out.push(row); }
                }
                let row = row2(self.colour(!x.turn), self.noop);
                if fits(&row, pat, 2) { out.push(row); }
                out
            }
            Some(Reader::AttackBy) => {
                let (k, c, f1, r1, f2, r2) = (pat[0], pat[1], pat[2], pat[3], pat[4], pat[5]);
                let mut out = vec![];
                let mk = |kind: u8, w: bool, s: u8, t: u8| -> Row {
                    let (fs, rs) = self.sq_terms(s);
                    let (ft, rt) = self.sq_terms(t);
                    let mut row = [FREE; ARITY_MAX];
                    row[0] = self.kinds[kind as usize]; row[1] = self.colour(w); row[2] = fs; row[3] = rs; row[4] = ft; row[5] = rt;
                    row
                };
                if f2 != FREE && r2 != FREE {
                    let Some(t) = self.square(f2, r2) else { return vec![] };
                    let colours: Vec<bool> = if c != FREE { vec![c == self.white] } else { vec![true, false] };
                    for w in colours {
                        for s in fwd(x.attackers_mask(w, t, x.occupied())) {
                            let kind = x.piece_type_at(s);
                            let (fs, rs) = self.sq_terms(s);
                            if (k == FREE || k == self.kinds[kind as usize]) && (f1 == FREE || f1 == fs) && (r1 == FREE || r1 == rs) { out.push(mk(kind, w, s, t)); }
                        }
                    }
                    return out;
                }
                let srcs: Vec<u8> = if f1 != FREE && r1 != FREE { match self.square(f1, r1) { Some(s) => vec![s], None => vec![] } } else { fwd(x.occupied()).collect() };
                for s in srcs {
                    let Some((kind, w)) = x.piece_at(s) else { continue };
                    if (c != FREE && self.colour(w) != c) || (k != FREE && self.kinds[kind as usize] != k) { continue; }
                    let (fs, rs) = self.sq_terms(s);
                    for t in fwd(x.attacks_mask(s)) {
                        let (ft, rt) = self.sq_terms(t);
                        if (f2 == FREE || f2 == ft) && (r2 == FREE || r2 == rt) {
                            let mut row = [FREE; ARITY_MAX];
                            row[0] = self.kinds[kind as usize]; row[1] = self.colour(w); row[2] = fs; row[3] = rs; row[4] = ft; row[5] = rt;
                            out.push(row);
                        }
                    }
                }
                out
            }
            _ => vec![],
        }
    }

    fn action(&self, row: &Row) -> Option<Act> {
        if row[1] == self.noop { return None; }
        let white = self.role_white(row[0])?;
        let m = *self.move_of.get(&row[1])?;
        Some(Act { white, term: row[1], m })
    }

    fn apply(&self, x: &PBoard, a: &Act) -> PBoard { let mut c = *x; c.push(a.m); c }
    fn pass(&self, x: &PBoard) -> PBoard { x.with_turn(!x.turn) }
    fn to_move(&self, x: &PBoard, role: T) -> PBoard {
        let w = role == self.white;
        if x.turn == w { *x } else { x.with_turn(w) }
    }

    fn value(&self, x: &PBoard, law: &str, args: &[T]) -> Option<i64> {
        match law {
            "exchange" => {
                let t = self.square(args[0], args[1])?;
                let y = self.to_move(x, args[2]);
                self.see_on(&y, t)
            }
            "exchange from" => {
                let t = self.square(args[0], args[1])?;
                let s = self.square(args[2], args[3])?;
                let y = self.to_move(x, args[4]);
                let m = PMove::new(s, t, 0);
                let (tt, _) = y.piece_at(t)?;
                y.piece_at(s)?;
                if !y.is_pseudo_legal(m) { return None; }
                let gain = crate::valuation::worth_pt(tt);
                let mut after = y;
                after.push(m);
                let back = self.see_on(&after, t)?;
                Some(gain - back.max(0))
            }
            _ => None,
        }
    }

    fn loc(&self, slots: &[usize], args: &Row) -> Option<u8> {
        for &i in slots {
            if i + 1 < ARITY_MAX && args[i] != FREE && args[i + 1] != FREE { return self.square(args[i], args[i + 1]); }
        }
        None
    }
    fn ends(&self, a: &Act) -> (Option<u8>, Option<u8>) { (Some(a.m.from), Some(a.m.to)) }
    fn unit_letter(&self, x: &PBoard, loc: Option<u8>) -> char {
        let Some(s) = loc else { return '?' };
        match x.piece_type_at(s) { PAWN => 'P', KNIGHT => 'N', BISHOP => 'B', ROOK => 'R', QUEEN => 'Q', KING => 'K', _ => '?' }
    }
}

// ------------------------------------------------------------------------------------------------------------------------------------------ the engine

/// A regression's cached answer: the names of both sides' readings, the read set (replayed on a hit), the head's truth.
#[derive(Clone)]
struct Cached { ours: Vec<Named>, theirs: Vec<Named>, reads: Vec<Atom>, holds: bool }

type Key = ([u64; 8], bool, u64, Option<u8>, u32);

pub struct Engine {
    pub spec: Spec,
    pub terms: Interner,
    pub b: ChessBinding,
    pub slots: Vec<Vec<usize>>,
    entries: Vec<NameEntry>,
    goals: Vec<Goal>,
    goal_ids: FxHashMap<Goal, u32>,
    cache: FxHashMap<(Key, u32), Cached>,
    stats: [u64; 2],                            // regressions read, served from the cache
}

/// A fuse on the memory the cache may hold (a count, named as one: cleared whole when reached, never a cut of an answer -- regression.py's CACHE_MAX).
const CACHE_MAX: usize = 400_000;

static ENGINE: Mutex<Option<Engine>> = Mutex::new(None);

fn with_engine<R>(f: impl FnOnce(&mut Engine) -> PyResult<R>) -> PyResult<R> {
    let mut g = ENGINE.lock().map_err(|_| verr("the regression engine's lock is poisoned".into()))?;
    let e = g.as_mut().ok_or_else(|| verr("the regression engine is not loaded (gdl_chess.native() loads rules/chess.gdl first)".into()))?;
    f(e)
}

fn key_of(x: &PBoard, limit: u32) -> Key {
    ([x.roles[0], x.roles[1], x.roles[2], x.roles[3], x.roles[4], x.roles[5], x.white, x.black], x.turn, x.clean_castling_rights(), x.ep, x.halfmove.min(limit))
}

pub(crate) fn board_of(bbs: [u64; 8], turn: bool, castling: u64, ep: Option<u8>, clocks: (u32, u32)) -> PyResult<PBoard> {
    let roles = [bbs[0], bbs[1], bbs[2], bbs[3], bbs[4], bbs[5]];
    let union = roles.iter().fold(0u64, |a, r| a | r);
    if roles.iter().map(|r| r.count_ones()).sum::<u32>() != union.count_ones() || union != (bbs[6] | bbs[7]) || bbs[6] & bbs[7] != 0 {
        return Err(verr("inconsistent bitboards".into()));
    }
    if let Some(e) = ep { if e >= 64 { return Err(verr(format!("no square {e}"))); } }
    Ok(PBoard { roles, white: bbs[6], black: bbs[7], turn, castling, ep, halfmove: clocks.0, fullmove: clocks.1 })
}

pub(crate) fn board_of_pos(p: &shakmaty::Chess) -> PBoard {
    use shakmaty::Position;
    let b = p.board();
    PBoard { roles: [b.pawns().0, b.knights().0, b.bishops().0, b.rooks().0, b.queens().0, b.kings().0], white: b.white().0, black: b.black().0,
             turn: p.turn() == shakmaty::Color::White, castling: p.castles().castling_rights().0, ep: p.maybe_ep_square().map(|s| s as u8),
             halfmove: p.halfmoves(), fullmove: p.fullmoves().get() }
}

/// A placement of the future: (piece symbol, at, lifted from).
type Placement = (String, u8, Option<u8>);

fn piece_of(sym: &str) -> PyResult<(u8, bool)> {
    let c = sym.chars().next().ok_or_else(|| verr("an empty piece symbol".into()))?;
    let t = match c.to_ascii_lowercase() { 'p' => PAWN, 'n' => KNIGHT, 'b' => BISHOP, 'r' => ROOK, 'q' => QUEEN, 'k' => KING, _ => return Err(verr(format!("no piece {sym}"))) };
    Ok((t, c.is_ascii_uppercase()))
}

/// regress.future_board (the counterfactual: the placements made -- graph.with_piece -- and `turn` to move, e.p. dropped: graph.with_turn); None = the state itself.
fn future(x: &PBoard, fut: &Option<(Vec<(u8, bool, u8, Option<u8>)>, bool)>) -> PBoard {
    match fut {
        None => *x,
        Some((pl, turn)) => {
            let mut c = *x;
            for &(t, w, at, frm) in pl {
                if let Some(f) = frm { c.remove_piece_at(f); }
                c.set_piece_at(at, t, w);
            }
            c.with_turn(*turn)
        }
    }
}

#[allow(clippy::type_complexity)]
fn parse_future(fut: Option<(Vec<Placement>, bool)>) -> PyResult<Option<(Vec<(u8, bool, u8, Option<u8>)>, bool)>> {
    match fut {
        None => Ok(None),
        Some((pl, turn)) => {
            let mut v = vec![];
            for (sym, at, frm) in pl {
                let (t, w) = piece_of(&sym)?;
                if at >= 64 || frm.is_some_and(|f| f >= 64) { return Err(verr("a square out of range (0..63)".into())); }
                v.push((t, w, at, frm));
            }
            Ok(Some((v, turn)))
        }
    }
}

impl Engine {
    fn goal(&self, g: u32) -> PyResult<Goal> { self.goals.get(g as usize).cloned().ok_or_else(|| verr(format!("no goal {g}"))) }

    /// The regression of `goal` on `x` read afresh (the goal's role on move first): (the state read, the regression).
    fn regress_fresh(&mut self, x: &PBoard, goal: &Goal) -> (PBoard, Regression<Act>) {
        let Engine { spec, terms, b, slots, .. } = self;
        let mut rg = Regressor::new(&*spec, &*b, terms, &slots[..]);
        rg.regress(x, goal)
    }

    /// The cached reading of `goal` on `x`: the names of both sides and the read set (computed once per (key, goal)).
    fn cached(&mut self, x: &PBoard, gi: u32) -> PyResult<&Cached> {
        let goal = self.goal(gi)?;
        let x0 = if goal.on_move != FREE { self.b.to_move(x, goal.on_move) } else { *x };
        let k = (key_of(&x0, self.b.clock_limit), gi);
        if self.cache.contains_key(&k) {
            self.stats[1] += 1;
            return Ok(&self.cache[&k]);
        }
        let (xs, reg) = self.regress_fresh(&x0, &goal);
        let ours = RG::name(&self.spec, &self.b, &self.terms, &self.slots, &reg, &goal, &self.entries, &xs, false);
        let theirs = RG::name(&self.spec, &self.b, &self.terms, &self.slots, &reg, &goal, &self.entries, &xs, true);
        self.stats[0] += 1;
        if self.cache.len() >= CACHE_MAX { self.cache.clear(); }
        self.cache.insert(k, Cached { ours, theirs, reads: reg.reads.iter().copied().collect(), holds: reg.holds });
        Ok(&self.cache[&k])
    }

    fn keys(&self, ns: &[Named]) -> Vec<(String, Option<u8>)> { RG::keys_of(ns, &self.entries) }

    // ------------------------------------------------------------------------------------- the Python values of terms and atoms
    fn term_py(&self, py: Python<'_>, t: T) -> PyObject {
        if t == FREE { return py.None(); }
        match self.terms.compound(t) {
            None => self.terms.name(t).into_pyobject(py).unwrap().into_any().unbind(),
            Some((f, args)) => {
                let mut v: Vec<PyObject> = vec![self.terms.name(f).into_pyobject(py).unwrap().into_any().unbind()];
                for a in args { v.push(self.term_py(py, *a)); }
                pyo3::types::PyTuple::new(py, v).unwrap().into_any().unbind()
            }
        }
    }
    fn row_py(&self, py: Python<'_>, row: &Row, n: usize) -> PyObject {
        let v: Vec<PyObject> = (0..n).map(|i| self.term_py(py, row[i])).collect();
        pyo3::types::PyTuple::new(py, v).unwrap().into_any().unbind()
    }
    fn rel_py(&self, r: Rel) -> String { if r == DISTINCT { "distinct".into() } else { self.spec.prog.rel_name(r).to_string() } }
    fn atom_py(&self, py: Python<'_>, a: &Atom) -> PyObject {
        (self.rel_py(a.rel), self.row_py(py, &a.args, a.n as usize)).into_pyobject(py).unwrap().into_any().unbind()
    }
    fn leaf_py(&self, py: Python<'_>, lf: &Leaf) -> PyObject {
        ((self.rel_py(lf.at.0), lf.at.1 as i64, lf.at.2 as i64), self.atom_py(py, &lf.atom), lf.neg).into_pyobject(py).unwrap().into_any().unbind()
    }
    fn lk(k: LK) -> &'static str { match k { LK::Pos => "pos", LK::Neg => "neg", LK::Distinct => "distinct" } }

    /// The whole regression as Python values (regression.py's Regression, field by field: its dataclasses are built from it in gdl_chess.py).
    fn regression_py(&self, py: Python<'_>, reg: &Regression<Act>, goal: &Goal) -> PyResult<PyObject> {
        let clauses: Vec<PyObject> = reg.clauses.iter().map(|(idx, lits)| {
            let ls: Vec<PyObject> = lits.iter().map(|(j, k, a, t)| {
                let at = match a { Some(a) => self.atom_py(py, a), None => py.None() };
                (*j as i64, Self::lk(*k), at, *t).into_pyobject(py).unwrap().into_any().unbind()
            }).collect();
            (*idx as i64, ls).into_pyobject(py).unwrap().into_any().unbind()
        }).collect();
        let defects: Vec<PyObject> = reg.defects.iter().map(|d| {
            let kind = match d.kind { RG::DK::False => "false", RG::DK::Witness => "witness", RG::DK::Value => "value" };
            let deep = match &d.deep { Some(a) => self.atom_py(py, a), None => py.None() };
            let w = match &d.witness {
                None => py.None(),
                Some(w) => {
                    let body: Vec<PyObject> = w.body.iter().map(|(k, r, row)| {
                        let n = if *r == DISTINCT { 2 } else { self.spec.arity[*r as usize] };
                        let rel: PyObject = if *r == DISTINCT { py.None() } else { self.rel_py(*r).into_pyobject(py).unwrap().into_any().unbind() };
                        (Self::lk(*k), rel, self.row_py(py, row, n)).into_pyobject(py).unwrap().into_any().unbind()
                    }).collect();
                    let action = match &w.action {
                        None => py.None(),
                        Some(a) => (self.term_py(py, self.b.colour(a.white)), self.term_py(py, a.term)).into_pyobject(py).unwrap().into_any().unbind(),
                    };
                    let flips: Vec<PyObject> = w.flips.iter().map(|lf| self.leaf_py(py, lf)).collect();
                    (self.atom_py(py, &w.atom), body, action, flips).into_pyobject(py).unwrap().into_any().unbind()
                }
            };
            let value = match &d.value {
                None => py.None(),
                Some((v, need, ats)) => {
                    let a: Vec<PyObject> = ats.iter().map(|a| self.atom_py(py, a)).collect();
                    (*v, *need, a).into_pyobject(py).unwrap().into_any().unbind()
                }
            };
            (kind, d.clause as i64, d.lit as i64, self.atom_py(py, &d.atom), deep, w, value).into_pyobject(py).unwrap().into_any().unbind()
        }).collect();
        let reads: Vec<PyObject> = reg.reads.iter().map(|a| self.atom_py(py, a)).collect();
        let n = self.spec.arity[goal.head.0 as usize];
        let rows: Vec<PyObject> = reg.rows.iter().map(|r| self.row_py(py, r, n)).collect();
        let leaves: Vec<PyObject> = reg.leaves.iter().map(|lf| self.leaf_py(py, lf)).collect();
        Ok((reg.holds, clauses, defects, reads, rows, reg.value, leaves).into_pyobject(py)?.into_any().unbind())
    }
}

fn named_py(n: &Named) -> (usize, Option<u8>, Vec<Option<u8>>, Option<u8>, Option<u8>, String, Option<i64>, String) {
    (n.entry, n.sq, n.by.clone(), n.frm, n.to, n.unit.to_string(), n.value, n.rel.clone())
}

// ------------------------------------------------------------------------------------------------------------------------------------------ the Python surface

/// spec_load(text, bindings, certified, location_sorts, clock_limit) -> (clauses, relations, strata, static relations): THE SPEC LOADED from its text
/// (rules/chess.gdl, read by gdl_chess.py: the file is the source) with the shelf's binding table (relation, its entry's `rust` name), the certified relations,
/// the location sorts (a file and a rank slot of a signature name a location) and the clock's limit (plan 76.5: the worths are the valuation's). Replaces a
/// loaded spec (the cache and the goals cleared). ValueError on a description that does not parse, is not stratified, or binds a relation no reader reads.
#[pyfunction]
pub fn spec_load(text: &str, bindings: Vec<(String, String)>, certified: Vec<String>, location_sorts: (String, String), clock_limit: u32)
                 -> PyResult<(usize, usize, usize, usize)> {
    let mut prog = Program::load(text).map_err(|e| verr(e.0))?;
    let mut terms = std::mem::take(&mut prog.terms);
    let b = ChessBinding::new(&mut terms, &prog, &bindings, clock_limit).map_err(verr)?;
    let cert: Vec<Rel> = certified.iter().map(|c| prog.rel_id(c).ok_or_else(|| verr(format!("certified {c}: no relation of the spec")))).collect::<PyResult<_>>()?;
    // every input relation and every certified one has its reader
    for r in 0..prog.nrels() {
        let name = prog.rel_name(r as Rel);
        if (crate::gdl::is_input(name) || cert.contains(&(r as Rel))) && prog.nodes.contains(&(r as Rel)) && b.readers[r].is_none() {
            return Err(verr(format!("{name}: no binding in the shelf's table (its entry's `rust` name)")));
        }
    }
    let slots: Vec<Vec<usize>> = (0..prog.nrels()).map(|r| {
        let sig = prog.signature(prog.rel_name(r as Rel)).cloned().unwrap_or_default();
        (0..sig.len().saturating_sub(1)).filter(|&i| sig[i] == location_sorts.0 && sig[i + 1] == location_sorts.1).collect()
    }).collect();
    let counts = (prog.clauses.len(), prog.nrels(), prog.strata.len(), prog.statics.len());
    let spec = Spec::new(prog, &cert, &mut terms).map_err(verr)?;
    let e = Engine { spec, terms, b, slots, entries: vec![], goals: vec![], goal_ids: FxHashMap::default(), cache: FxHashMap::default(), stats: [0, 0] };
    *ENGINE.lock().map_err(|_| verr("lock poisoned".into()))? = Some(e);
    Ok(counts)
}

/// spec_loaded() -> whether a spec is loaded in this process.
#[pyfunction]
pub fn spec_loaded() -> PyResult<bool> { Ok(ENGINE.lock().map_err(|_| verr("lock poisoned".into()))?.is_some()) }

type EntryIn = (String, String, bool, String, i64, Option<(String, Option<u16>, u16)>, Option<(String, Option<u16>, u16)>, String, String, String, bool, Option<(i32, i32)>);

/// spec_names(entries, text_keys) -> how many: THE STORE'S DEFECT_FIX BINDINGS as data (lemma_store.defect_names, in its order): each (name, goal, for theirs,
/// defect, order, leaf place, unless place, sq field, by field, group, sort by sq, (clause, lit)) and its template's keys (None: a plain text; else the
/// relations a dict template names). The text itself stays in Python.
#[pyfunction]
pub fn spec_names(entries: Vec<EntryIn>, text_keys: Vec<Option<Vec<String>>>) -> PyResult<usize> {
    with_engine(|e| {
        let place = |p: &Option<(String, Option<u16>, u16)>| -> PyResult<Option<RG::Place>> {
            match p { None => Ok(None), Some((r, c, l)) => Ok(Some((e.spec.prog.rel_id(r).ok_or_else(|| verr(format!("{r}: no relation of the spec")))?, *c, *l))) }
        };
        let field = |s: &str| -> PyResult<Field> { match s { "leaf" => Ok(Field::Leaf), "to" => Ok(Field::To), "from" => Ok(Field::From), "" => Ok(Field::Leaf), _ => Err(verr(format!("no field {s}"))) } };
        let mut out = vec![];
        for (i, (name, goal, theirs, defect, order, leaf, unless, sq, by, group, sort_sq, clause)) in entries.iter().enumerate() {
            let d = match defect.as_str() { "false" => ND::False, "witness" => ND::Witness, "value" => ND::Value, "holds" => ND::Holds, "leaf" => ND::Leaf,
                                            _ => return Err(verr(format!("{name}: defect {defect}"))) };
            let tk = text_keys.get(i).cloned().flatten();
            out.push(NameEntry { name: name.clone(), goal: goal.clone(), theirs: *theirs, defect: d, order: *order, leaf: place(leaf)?, unless: place(unless)?,
                                 sq: field(sq)?, by: field(by)?, group_sq: group == "sq", sort_sq: *sort_sq, clause: *clause, text_any: tk.is_none(),
                                 text_rels: tk.unwrap_or_default() });
        }
        e.entries = out;
        e.cache.clear();
        Ok(e.entries.len())
    })
}

type ValueIn = (String, Vec<String>, i64, String, Vec<Option<String>>);

/// spec_goal(name, head_rel, head_args, on_move, value) -> the goal's id: a GOAL entry's `gdl` with its parameters substituted (regression.goal_of): the head
/// (a relation and its arguments, None for a variable), the role on move (None: as the state is), the value (law, args, need, the read pattern's relation and
/// arguments). The same goal is one id. ValueError on a relation or a law the engine does not know.
#[pyfunction]
pub fn spec_goal(name: String, head_rel: String, head_args: Vec<Option<String>>, on_move: Option<String>, value: Option<ValueIn>) -> PyResult<u32> {
    with_engine(|e| {
        let rel = e.spec.prog.rel_id(&head_rel).ok_or_else(|| verr(format!("goal {name}: {head_rel} is no relation of the spec")))?;
        let pat = |args: &[Option<String>], terms: &mut Interner| -> PyResult<Row> {
            if args.len() > ARITY_MAX { return Err(verr("too many arguments".into())); }
            let mut row = [FREE; ARITY_MAX];
            for (i, a) in args.iter().enumerate() { if let Some(s) = a { row[i] = terms.sym(s); } }
            Ok(row)
        };
        let head = pat(&head_args, &mut e.terms)?;
        let om = match on_move { Some(s) => e.terms.sym(&s), None => FREE };
        let val = match value {
            None => None,
            Some((law, args, need, vrel, vargs)) => {
                if law != "exchange" && law != "exchange from" { return Err(verr(format!("goal {name}: no valuation law {law:?} in the chess binding"))); }
                let vr = e.spec.prog.rel_id(&vrel).ok_or_else(|| verr(format!("goal {name}: {vrel} is no relation of the spec")))?;
                let a: Vec<T> = args.iter().map(|s| e.terms.sym(s)).collect();
                Some((law, a, need, (vr, pat(&vargs, &mut e.terms)?)))
            }
        };
        let g = Goal { name, head: (rel, head), on_move: om, value: val };
        if let Some(&i) = e.goal_ids.get(&g) { return Ok(i); }
        let i = e.goals.len() as u32;
        e.goals.push(g.clone());
        e.goal_ids.insert(g, i);
        Ok(i)
    })
}

type Parts = ([u64; 8], bool, u64, Option<u8>, (u32, u32));

/// regress_bb(parts, goal) -> the whole regression of the goal on the board (the parts of a python-chess board: the eight bitboards, white to move, the raw
/// castling rights, the raw e.p. square, (halfmove, fullmove)), read afresh: (holds, clauses, defects, reads, rows, value, leaves) as regression.py's fields.
#[pyfunction]
pub fn regress_bb(py: Python<'_>, parts: Parts, goal: u32) -> PyResult<PyObject> {
    let x = board_of(parts.0, parts.1, parts.2, parts.3, parts.4)?;
    with_engine(|e| {
        let g = e.goal(goal)?;
        let (_xs, reg) = e.regress_fresh(&x, &g);
        e.regression_py(py, &reg, &g)
    })
}

type NamedOut = (usize, Option<u8>, Vec<Option<u8>>, Option<u8>, Option<u8>, String, Option<i64>, String);

/// names_bb(parts, goal, theirs, future) -> the NAMED DEFECTS of the goal on the board's future (`future`: None -- the board; or (placements [(symbol, at,
/// lifted from)], white to move): regress.future_board), cached per (key, goal): each (entry index, location, locations involved, from, to, the unit's letter,
/// value, the false atom's relation) -- the fields of the entry's template (the text is formatted in Python).
#[pyfunction]
pub fn names_bb(parts: Parts, goal: u32, theirs: bool, future_spec: Option<(Vec<Placement>, bool)>) -> PyResult<Vec<NamedOut>> {
    let x = board_of(parts.0, parts.1, parts.2, parts.3, parts.4)?;
    let fut = parse_future(future_spec)?;
    with_engine(|e| {
        let y = future(&x, &fut);
        let c = e.cached(&y, goal)?;
        Ok((if theirs { &c.theirs } else { &c.ours }).iter().map(named_py).collect())
    })
}

/// reads_bb(parts, goal) -> (the head holds, the read set) of the cached reading (the test hook of the cache's replay: a hit gives the read set the fresh reading gave).
#[pyfunction]
pub fn reads_bb(py: Python<'_>, parts: Parts, goal: u32) -> PyResult<(bool, Vec<PyObject>)> {
    let x = board_of(parts.0, parts.1, parts.2, parts.3, parts.4)?;
    with_engine(|e| {
        let c = e.cached(&x, goal)?.clone();
        Ok((c.holds, c.reads.iter().map(|a| e.atom_py(py, a)).collect()))
    })
}

type FixOut = (bool, Vec<(String, Option<u8>)>);

/// fixes_bb(parts, goal, key, moves, future, before) -> per move (mended, new): regression.fixes in one crossing -- for each move (from, to, promotion type 0 /
/// 2..5) of the board's side, its certified next state (python-chess's push), the goal's future of it, the goal regressed and named there; `mended`: no named
/// defect with the key (name, location) is left; `new`: the named defects whose key was not a defect before (`before`: None -- the names on the board's future less
/// the key). A FIX is mended and nothing new. ValueError on a move that is not pseudo-legal there.
#[pyfunction]
pub fn fixes_bb(parts: Parts, goal: u32, key: (String, Option<u8>), moves: Vec<(u8, u8, u8)>, future_spec: Option<(Vec<Placement>, bool)>,
                before: Option<Vec<(String, Option<u8>)>>) -> PyResult<Vec<FixOut>> {
    let x = board_of(parts.0, parts.1, parts.2, parts.3, parts.4)?;
    let fut = parse_future(future_spec)?;
    with_engine(|e| {
        let before: BTreeSet<(String, Option<u8>)> = match before {
            Some(b) => b.into_iter().collect(),
            None => {
                let y = future(&x, &fut);
                let ns = e.cached(&y, goal)?.ours.clone();
                e.keys(&ns).into_iter().filter(|k| *k != key).collect()
            }
        };
        let mut out = vec![];
        for (f, t, p) in moves {
            let m = PMove::new(f, t, p);
            if f >= 64 || t >= 64 || x.piece_type_at(f) == 0 { return Err(verr(format!("move {} has no unit on its from-square (python-chess push asserts one)", m.uci()))); }
            let mut x1 = x;
            x1.push(m);
            let y = future(&x1, &fut);
            let ns = e.cached(&y, goal)?.ours.clone();
            let keys = e.keys(&ns);
            out.push((!keys.contains(&key), keys.into_iter().filter(|k| !before.contains(k)).collect()));
        }
        Ok(out)
    })
}

/// spec_cache_clear(): the regression cache emptied (graph.CLEAR_HOOKS: with every fresh root).
#[pyfunction]
pub fn spec_cache_clear() -> PyResult<()> { with_engine(|e| { e.cache.clear(); Ok(()) }) }

/// plan 76.5: the regression's cache emptied when the valuation changes (set_valuation), if a spec is loaded (no error when none is).
pub(crate) fn cache_clear_if_loaded() { if let Ok(mut g) = ENGINE.lock() { if let Some(e) = g.as_mut() { e.cache.clear(); } } }

/// spec_stats() -> (regressions read, served from the cache, cached now).
#[pyfunction]
pub fn spec_stats() -> PyResult<(u64, u64, usize)> { with_engine(|e| Ok((e.stats[0], e.stats[1], e.cache.len()))) }

/// spec_run_bb(parts, does) -> {relation: [rows]}: (a)'s CROSS-CHECK -- the bottom-up semi-naive evaluation of every relation on the board's facts (the readers'
/// rows of every input relation, every pattern free) and, with `does` (a legal uci of the side to move), the joint move (the mover's move, the other's noop).
#[pyfunction]
pub fn spec_run_bb(py: Python<'_>, parts: Parts, does: Option<String>) -> PyResult<PyObject> {
    let x = board_of(parts.0, parts.1, parts.2, parts.3, parts.4)?;
    with_engine(|e| {
        let mut facts: Vec<(Rel, Row)> = vec![];
        for r in 0..e.spec.prog.nrels() {
            let name = e.spec.prog.rel_name(r as Rel).to_string();
            if !crate::gdl::is_input(&name) || name == "does" { continue; }
            for row in e.b.input(&x, r as Rel, &[FREE; ARITY_MAX]) { facts.push((r as Rel, row)); }
        }
        if let Some(u) = does {
            let m = x.legal_moves().into_iter().find(|m| m.uci() == u).ok_or_else(|| verr(format!("{u} is not legal on the board")))?;
            let d = e.spec.prog.rel_id("does").ok_or_else(|| verr("the spec has no `does`".into()))?;
            facts.push((d, row2(e.b.colour(x.turn), e.b.move_term_of(m))));
            facts.push((d, row2(e.b.colour(!x.turn), e.b.noop)));
        }
        let mut prog_terms = std::mem::take(&mut e.terms);
        std::mem::swap(&mut e.spec.prog.terms, &mut prog_terms);
        let db = e.spec.prog.run(&facts);
        std::mem::swap(&mut e.spec.prog.terms, &mut prog_terms);
        e.terms = prog_terms;
        let out = pyo3::types::PyDict::new(py);
        for r in 0..e.spec.prog.nrels() {
            let n = e.spec.arity[r];
            let rows: Vec<PyObject> = db.facts(r as Rel).iter().map(|row| e.row_py(py, row, n)).collect();
            out.set_item(e.spec.prog.rel_name(r as Rel), rows)?;
        }
        Ok(out.into_any().unbind())
    })
}

/// spec_ans_bb(parts, rel, pattern, certified) -> the top-down prover's rows of `rel` matching the pattern (None free) on the board -- with the certified relations
/// read through the binding (`certified` true, the regression's reading) or every relation from its clauses (false): (a)'s cross-check of the regression's evaluator.
#[pyfunction]
pub fn spec_ans_bb(py: Python<'_>, parts: Parts, rel: String, pattern: Vec<Option<String>>, certified: bool) -> PyResult<Vec<PyObject>> {
    let x = board_of(parts.0, parts.1, parts.2, parts.3, parts.4)?;
    with_engine(|e| {
        let r = e.spec.prog.rel_id(&rel).ok_or_else(|| verr(format!("{rel}: no relation of the spec")))?;
        let mut pat = [FREE; ARITY_MAX];
        for (i, a) in pattern.iter().enumerate() { if let Some(s) = a { pat[i] = e.terms.sym(s); } }
        let n = e.spec.arity[r as usize];
        let rows = if certified {
            let Engine { spec, terms, b, .. } = &mut *e;
            let mut p = RG::Prover::new(spec, b, &x, terms);
            p.ans(r, &pat)
        } else {
            let Engine { spec, terms, b, .. } = &mut *e;
            let mut kinds = spec.kinds.clone();
            for (i, k) in spec.kinds.iter().enumerate() { if *k == RG::Kind::Certified { kinds[i] = RG::Kind::Derived; } }
            let saved = std::mem::replace(&mut spec.kinds, kinds);
            let rows = { let mut p = RG::Prover::new(spec, b, &x, terms); p.ans(r, &pat) };
            spec.kinds = saved;
            rows
        };
        Ok(rows.iter().map(|row| e.row_py(py, row, n)).collect())
    })
}

/// The handle's regression (`Pos.regress`): the position as shakmaty holds it read as a board (its e.p. square shakmaty's).
pub(crate) fn regress_pos(py: Python<'_>, p: &shakmaty::Chess, goal: u32) -> PyResult<PyObject> {
    let x = board_of_pos(p);
    with_engine(|e| {
        let g = e.goal(goal)?;
        let (_xs, reg) = e.regress_fresh(&x, &g);
        e.regression_py(py, &reg, &g)
    })
}

/// The handle's fixes (`Pos.fixes`): fixes_bb on the handle's position.
pub(crate) fn fixes_pos(p: &shakmaty::Chess, goal: u32, key: (String, Option<u8>), moves: Vec<(u8, u8, u8)>, future_spec: Option<(Vec<Placement>, bool)>,
                        before: Option<Vec<(String, Option<u8>)>>) -> PyResult<Vec<FixOut>> {
    let x = board_of_pos(p);
    fixes_bb(([x.roles[0], x.roles[1], x.roles[2], x.roles[3], x.roles[4], x.roles[5], x.white, x.black], x.turn, x.castling, x.ep, (x.halfmove, x.fullmove)),
             goal, key, moves, future_spec, before)
}

/// py_legal_full_bb(parts) -> the legal moves as UCI in python-chess's order on the board (pyboard.rs's mirror: the test hook, castling and e.p. included).
#[pyfunction]
pub fn py_legal_full_bb(parts: Parts) -> PyResult<Vec<String>> {
    let x = board_of(parts.0, parts.1, parts.2, parts.3, parts.4)?;
    Ok(x.legal_moves().iter().map(|m| m.uci()).collect())
}

/// py_push_bb(parts, uci) -> the parts after python-chess's push of the (pseudo-legal) move (pyboard.rs's mirror: the test hook). ValueError when not pseudo-legal.
#[pyfunction]
pub fn py_push_bb(parts: Parts, uci: String) -> PyResult<Parts> {
    let x = board_of(parts.0, parts.1, parts.2, parts.3, parts.4)?;
    let m = x.legal_moves().into_iter().find(|m| m.uci() == uci).or_else(|| {
        let b = uci.as_bytes();
        if b.len() < 4 { return None; }
        let sq = |f: u8, r: u8| -> Option<u8> { if (b'a'..=b'h').contains(&f) && (b'1'..=b'8').contains(&r) { Some((r - b'1') * 8 + (f - b'a')) } else { None } };
        let p = match b.get(4) { None => 0, Some(b'n') => KNIGHT, Some(b'b') => BISHOP, Some(b'r') => ROOK, Some(b'q') => QUEEN, _ => return None };
        let m = PMove::new(sq(b[0], b[1])?, sq(b[2], b[3])?, p);
        if x.is_pseudo_legal(m) { Some(m) } else { None }
    }).ok_or_else(|| verr(format!("{uci} is not pseudo-legal on the board")))?;
    let mut y = x;
    y.push(m);
    Ok(([y.roles[0], y.roles[1], y.roles[2], y.roles[3], y.roles[4], y.roles[5], y.white, y.black], y.turn, y.castling, y.ep, (y.halfmove, y.fullmove)))
}

