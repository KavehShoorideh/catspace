//! PLAN 78.36 THE SEARCH TOWARD CLAIMS: a depth-limited AND-OR search, deepened one ply at a time inside its budget, whose LEAVES are
//!   (1) the game's own outcome (a win, draw or loss for `role`),
//!   (2) what a LEAF ORACLE says of a position that is not the root (a claim of the store: the oracle answers for the side to move there; here it is turned into `role`'s view),
//!   (3) a position met again on the line (or in the game's record) -- a draw, the rules' repetition,
//!   (4) otherwise UNKNOWN, at the depth limit.
//! A WIN is of two kinds, kept apart in the value: a GAME win (every leaf of the proof is the game's own outcome) stands above a CLAIMED win (some leaf is a claim): a node's value is its
//! best (or worst) child's, so a claim anywhere in a proof makes the whole a claimed win. Only a game win is certain; a claimed win is a statement of the store.
//! A node's value for `role` (the maximiser) is its best child's, for the other side (the minimiser) its worst child's, in the order  LOSS < UNKNOWN < DRAW < WIN.
//! UNKNOWN stays distinct from DRAW on purpose: unknown is not a result, and the side against us prefers to leave us in the unknown over a claimed draw (it has not conceded
//! the draw), so a claimed draw we can force is better than an unknown, and an unknown is better than a loss. Inside a class the value carries the ply (a nearer win is better,
//! a later loss is better), so the shortest line to a win is known where the leaves are. A move's class is read at the DEEPEST COMPLETED iteration; an iteration the budget cut
//! is dropped (a cut is a gap, never a result).
//!
//! The oracle is a trait so that this crate needs to know nothing of rules or stores: the claim-backed oracle is built where both are visible (cs_decision). Game-free.
use crate::{won, Budget, Cut};
use crate::stepping::Stepped;
use cs_base::choice::fold::{Bounds, Fold, Kind, Node, Pre};
use cs_game::{Game, Role};
use std::collections::HashMap;
use std::time::Instant;

/// What an oracle says of a position, for the SIDE TO MOVE there.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict { Win, Draw, Loss }

/// Something that may know the result of a (non-root) position: a verdict for the side to move, or None. It is a claim, read as evidence; it is not computed by the search.
pub trait LeafOracle<G: Game> {
    fn verdict(&self, g: &G, s: &G::State) -> Option<Verdict>;
}

impl<G: Game, F: Fn(&G, &G::State) -> Option<Verdict>> LeafOracle<G> for F {
    fn verdict(&self, g: &G, s: &G::State) -> Option<Verdict> { self(g, s) }
}

/// The class of a root move for `role`, in the order loss < unknown < draw < win.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Class { Loss, Unknown, Draw, Win }

impl Class {
    pub fn word(&self) -> &'static str { match self { Class::Loss => "loss", Class::Unknown => "unknown", Class::Draw => "draw", Class::Win => "win" } }
}

// A value is an integer so that the order is plain: a class owns a band, and inside the band the ply orders (win: nearer is higher; loss: later is higher).
const LOSS_BASE: i32 = 0;           // loss: LOSS_BASE + ply      (below UNKNOWN: plies stay under 1000)
const UNKNOWN: i32 = 1000;
const DRAW: i32 = 2000;
const CLAIM_WIN_BASE: i32 = 3500;   // a CLAIMED win: CLAIM_WIN_BASE - ply   (3001..3499: above a draw, below every game win)
const GAME_WIN_BASE: i32 = 4500;    // a GAME win:    GAME_WIN_BASE - ply    (4001..4499)
const GAME_WIN_FLOOR: i32 = 4000;
/// The depth at which the search stops when nothing has spoken: if the iteration of this depth completes, no claim has answered at any leaf so far and no game outcome was met, deeper
/// reading of a position the store says nothing about only costs time (a named constant, not tuned).
pub const SILENT_DEPTH: u32 = 2;
const INF: i32 = i32::MAX / 2;

/// A value that rests on the game's own outcome alone.
pub fn is_game_win(v: i32) -> bool { v >= GAME_WIN_FLOOR }

pub fn class_of(v: i32) -> Class { if v >= 3000 { Class::Win } else if v >= DRAW { Class::Draw } else if v >= UNKNOWN { Class::Unknown } else { Class::Loss } }

fn scored(c: Verdict, ply: i32) -> i32 { match c { Verdict::Win => CLAIM_WIN_BASE - ply, Verdict::Draw => DRAW, Verdict::Loss => LOSS_BASE + ply } }

/// One deepening step, as measured.
#[derive(Debug, Clone, Copy)]
pub struct TowardRead { pub depth: u32, pub nodes: u64, pub evals: u64, pub seconds: f64, pub complete: bool }

/// What the search found.
#[derive(Debug, Clone, Default)]
pub struct Toward {
    /// The deepest COMPLETED iteration (0: none completed -- every root move is unknown).
    pub depth: u32,
    /// Per root move (by its index in the game's legal moves): its class and value at that depth.
    pub root: Vec<(usize, Class, i32)>,
    pub nodes: u64,
    /// The oracle's calls (leaf positions it was asked about, counting repeats).
    pub evals: u64,
    /// The budget cut an iteration.
    pub cut: bool,
    /// The last completed iteration met no unknown leaf: every line was read to a result, deeper reading changes nothing.
    pub exact: bool,
    pub reads: Vec<TowardRead>,
    /// The search stopped at `SILENT_DEPTH` because no claim had spoken and no game outcome had been met (every root move is then unknown).
    pub silent: bool,
}

impl Toward {
    /// The root moves whose win rests on the game's own outcome alone, best (nearest) first.
    pub fn game_wins(&self) -> Vec<usize> {
        let mut v: Vec<(usize, i32)> = self.root.iter().filter(|r| is_game_win(r.2) && r.1 == Class::Win).map(|r| (r.0, r.2)).collect();
        v.sort_by_key(|(i, val)| (-*val, *i));
        v.into_iter().map(|(i, _)| i).collect()
    }
    /// The best class among the root moves (Unknown when there are none).
    pub fn best_class(&self) -> Class { self.root.iter().map(|r| r.1).max().unwrap_or(Class::Unknown) }
    /// The root moves of `class`, best value first (ties by index).
    pub fn of_class(&self, class: Class) -> Vec<usize> {
        let mut v: Vec<(usize, i32)> = self.root.iter().filter(|r| r.1 == class).map(|r| (r.0, r.2)).collect();
        v.sort_by_key(|(i, val)| (-*val, *i));
        v.into_iter().map(|(i, _)| i).collect()
    }
    pub fn count(&self, class: Class) -> usize { self.root.iter().filter(|r| r.1 == class).count() }
}

/// PLAN 79.33 (SYSTEM_SPEC 11 "What plan 79.32 left of one algorithm", 13 item 51; piece: the tactical reasoner as a case of the base). THE WALK IS NOT HERE: the and-or recursion with its window is layer 0's (`cs_base::choice::fold::Fold`). What this file hands it: `stepping::Stepped`, the node (the positions of the line being walked, the
/// game's moves as its options), and `Run`, the kind of answer (the class-and-ply whole number, best at the side's turn and worst at any other, in a window): the leaf (the game's own outcome, the record's draw, the line's draw, the oracle's claim, the depth limit), the budget and the clock, the answers kept by position and plies
/// left (the memo) with the bound a cut makes of them, and the marks of the iteration (`horizon`, `spoke`, `outcome`, the taint of a value that rests on the line).
struct Run<'a, G: Game, O: LeafOracle<G>> {
    g: &'a G,
    role: Role,
    oracle: &'a O,
    history: &'a cs_rep::history::Seen,
    path: Vec<u64>,
    evals: u64,
    horizon: bool,
    /// Whether any leaf was answered by the oracle, or was the game's own outcome (over the whole search so far).
    spoke: bool,
    outcome: bool,
    /// PLAN 79.18: the answers already read in this iteration, by (position, plies left): the value and what it is a bound of; None: no memo (the search as it was)
    memo: Option<HashMap<(u64, u32), (i32, Bound)>>,
    /// a value read in this subtree rests on the path (a position met again on the line is a draw): such a value is never stored
    path_taint: bool,
    b: &'a mut Budget,
}

/// PLAN 79.18: what a stored value is of the position's true value inside its iteration: the value itself, or a bound from a cut (the window's edge).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Bound { Exact, Lower, Upper }

impl<'a, G: Game, O: LeafOracle<G>> Run<'a, G, O> {
    fn terminal(&self, v: &[i64], ply: i32) -> i32 {
        if won(v, self.role) { GAME_WIN_BASE - ply }
        else if v.iter().enumerate().any(|(i, x)| i != self.role && *x > v[self.role]) { LOSS_BASE + ply }
        else { DRAW }
    }
}

impl<'a, G: Game, O: LeafOracle<G>> Kind<Stepped<'a, G>> for Run<'a, G, O> {
    type Val = i32;
    type Err = Cut;
    type Edge = ();
    /// the position's key, the taint of the node that opened this one, and the plies left (the memo's key)
    type Token = (u64, bool, u32);
    const WINDOWED: bool = true;
    const CHANCE: bool = false;

    fn before(&mut self, space: &Stepped<'a, G>, left: u32, ply: u32, win: &mut Bounds<i32>) -> Result<Pre<i32, (u64, bool, u32)>, Cut> {
        let s = space.top();
        let ply = ply as i32;
        if let Some(v) = self.g.outcome(s) { self.outcome = true; return Ok(Pre::Done(self.terminal(&v, ply))); }
        let key = self.g.key(s);
        if self.history.contains(&key) { return Ok(Pre::Done(DRAW)); }          // a position met again in the game's record: the rules' draw
        if self.path.contains(&key) { self.path_taint = true; return Ok(Pre::Done(DRAW)); }          // met again on the line
        // PLAN 79.18: within one iteration a position and the plies left fix the ply, so an answer read once stands for every way of reaching it (the memo is emptied at each iteration)
        if let Some(m) = &self.memo {
            if let Some(&(v, bound)) = m.get(&(key, left)) {
                match bound {
                    Bound::Exact => return Ok(Pre::Done(v)),
                    Bound::Lower => { if v >= win.hi { return Ok(Pre::Done(v)); } win.lo = win.lo.max(v); }
                    Bound::Upper => { if v <= win.lo { return Ok(Pre::Done(v)); } win.hi = win.hi.min(v); }
                }
            }
        }
        let taint_before = self.path_taint;
        self.path_taint = false;
        self.b.spend()?;
        if self.b.nodes & 15 == 0 && Instant::now() >= self.b.deadline { return Err(Cut); }
        let mover = self.g.mover(s);
        if let Some(v) = { self.evals += 1; self.oracle.verdict(self.g, s) } {
            self.spoke = true;
            let v = if mover == Some(self.role) { v } else { match v { Verdict::Win => Verdict::Loss, Verdict::Loss => Verdict::Win, Verdict::Draw => Verdict::Draw } };
            self.path_taint |= taint_before;
            return Ok(Pre::Done(scored(v, ply)));
        }
        Ok(Pre::Go((key, taint_before, left)))
    }

    fn horizon(&mut self, _space: &Stepped<'a, G>, token: (u64, bool, u32), _ply: u32) -> Result<i32, Cut> { self.horizon = true; self.path_taint |= token.1; Ok(UNKNOWN) }

    #[inline(always)]
    fn edge(&mut self, _space: &Stepped<'a, G>, _o: &G::Move) -> Option<()> { Some(()) }

    #[inline(always)]
    fn open(&mut self, token: &(u64, bool, u32)) { self.path.push(token.0); }
    #[inline(always)]
    fn close(&mut self) { self.path.pop(); }

    fn finish(&mut self, _space: &Stepped<'a, G>, token: (u64, bool, u32), got: Option<i32>, entry: &Bounds<i32>) -> Result<i32, Cut> {
        let out = match got { Some(v) => v, None => UNKNOWN };
        if !self.path_taint {
            if let Some(m) = self.memo.as_mut() {
                let bound = if out <= entry.lo { Bound::Upper } else if out >= entry.hi { Bound::Lower } else { Bound::Exact };
                m.insert((token.0, token.2), (out, bound));
            }
        }
        self.path_taint |= token.1;
        Ok(out)
    }
}

/// `role` to move at `s` (the root): the class of every root move, deepened from one ply to `max_depth` while the budget lasts. `history`: the keys of the positions the game's
/// record already holds (a position met again is the rules' draw). The root itself is never asked of the oracle: a claim speaks of positions the line REACHES.
pub fn toward<G: Game, O: LeafOracle<G>>(g: &G, s: &G::State, role: Role, oracle: &O, history: &cs_rep::history::Seen, max_depth: u32, b: &mut Budget) -> Toward {
    toward_with(g, s, role, oracle, history, max_depth, Some(SILENT_DEPTH), b)
}

/// PLAN 79.17 (SYSTEM_SPEC 7, the measure of progress; piece: the measure): the same search with the stop at a silent depth as a parameter. `silent_at` None: the search is never stopped for having heard nothing (a leaf oracle that speaks only at some leaves, such as "the measure is
/// lower here", is silent at depth two and must still be read deeper); `Some(d)`: as `toward`. Nothing else differs.
#[allow(clippy::too_many_arguments)]
pub fn toward_with<G: Game, O: LeafOracle<G>>(g: &G, s: &G::State, role: Role, oracle: &O, history: &cs_rep::history::Seen, max_depth: u32, silent_at: Option<u32>, b: &mut Budget) -> Toward {
    toward_run(g, s, role, oracle, history, max_depth, silent_at, false, b)
}

/// PLAN 79.18 (SYSTEM_SPEC 7 the measure of progress, 11 hot paths in Rust; the cost of the held goal's question): `toward_with` with a MEMO of the answers read in an iteration, by position and plies left, so that the same position reached by another order of moves is read once. The
/// answers are the same as `toward_with`'s (a test compares the roots on constructed positions): a value that rests on the path (a position met again on the line is a draw) is never stored, a bound from a cut is stored as a bound, the memo is emptied at each iteration (the ply is then fixed by the position and the plies left).
#[allow(clippy::too_many_arguments)]
pub fn toward_memo<G: Game, O: LeafOracle<G>>(g: &G, s: &G::State, role: Role, oracle: &O, history: &cs_rep::history::Seen, max_depth: u32, silent_at: Option<u32>, b: &mut Budget) -> Toward {
    toward_run(g, s, role, oracle, history, max_depth, silent_at, true, b)
}

#[allow(clippy::too_many_arguments)]
fn toward_run<G: Game, O: LeafOracle<G>>(g: &G, s: &G::State, role: Role, oracle: &O, history: &cs_rep::history::Seen, max_depth: u32, silent_at: Option<u32>, memo: bool, b: &mut Budget) -> Toward {
    let mut out = Toward::default();
    let moves = g.moves(s);
    out.root = (0..moves.len()).map(|i| (i, Class::Unknown, UNKNOWN)).collect();
    if moves.is_empty() { out.exact = true; return out; }
    let root_key = g.key(s);
    let run = Run { g, role, oracle, history, path: vec![root_key], evals: 0, horizon: false, spoke: false, outcome: false, memo: if memo { Some(HashMap::new()) } else { None }, path_taint: false, b };
    let mut w = Fold::new(Stepped::new(g, role, s), run);
    for d in 1..=max_depth {
        let (n0, e0, t0) = (w.kind.b.nodes, w.kind.evals, Instant::now());
        w.kind.horizon = false;
        if let Some(m) = w.kind.memo.as_mut() { m.clear(); }
        let mut now = Vec::with_capacity(moves.len());
        let mut cut = false;
        for (i, m) in moves.iter().enumerate() {
            w.space.enter(m);
            let r = w.go(d - 1, 1, Bounds { lo: -INF, hi: INF });
            w.space.leave();
            match r {
                Ok(v) => now.push((i, class_of(v), v)),
                Err(Cut) => { cut = true; break; }
            }
        }
        out.reads.push(TowardRead { depth: d, nodes: w.kind.b.nodes - n0, evals: w.kind.evals - e0, seconds: t0.elapsed().as_secs_f64(), complete: !cut });
        if cut { out.cut = true; break; }
        out.depth = d;
        out.root = now;
        out.exact = !w.kind.horizon;
        if out.exact || out.root.iter().any(|x| x.1 == Class::Win) { break; }
        if silent_at == Some(d) && !w.kind.spoke && !w.kind.outcome { out.silent = true; break; }          // every line read to a result, or a win is found (the earlier iteration was the shorter)
        if !w.kind.b.left() { out.cut = true; break; }
    }
    out.nodes = w.kind.b.nodes;
    out.evals = w.kind.evals;
    out
}
