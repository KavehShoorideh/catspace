//! PLAN 78.5 THE LEDGER SEARCH (Kaveh 2026-11-04: "I don't want us to have value for the material yet ... consider material lost as an outcome of a tactic: bishop
//! lost, queen lost ... later we will consider what is the value of the thing that's lost ... for now it's just a piece lost. And we want to avoid piece losses").
//!
//! A line's OUTCOME is its LEDGER: the units each side lost along it (owner, kind -- read through `Game::captured`, so the layer names no unit), or the game's own
//! outcome where the line ends the game. A JUDGE turns a ledger into a number; the judge is the VARIABLE: `Count` (no values: every unit the same, and a unit of ours
//! lost outweighs every unit of theirs won -- losses are avoided first) is the default; `Worth` (a valuation plugged in: a table from kind names to worths, read from
//! the store when we have one) is the other. Both are additive over captures, so the search is a minimax of the judge's sum: alpha-beta, deepened while the clock
//! lasts, the line's captures resolved at its horizon (each side may stop or go on capturing), a transposition table on the future part. A line ending the game
//! scores above (or below) every ledger. Within its horizon the answer is exact; beyond it, unread.
use cs_game::{Game, Role};
use rustc_hash::FxHashMap;
use std::collections::HashMap;
use std::time::Instant;

/// The JUDGE of a ledger: what one captured unit adds for `me` (positive: theirs lost; negative: ours lost), and what one of a side's own units changing kind adds
/// (a promotion: worth of the new kind less the old).
pub trait Judge {
    fn delta(&self, me: Role, owner: Role, kind: &str) -> i64;
    fn change(&self, me: Role, owner: Role, from: &str, to: &str) -> i64;
    fn name(&self) -> String;
    /// The plain worth of one unit of `kind` (no loss weighting: what `delta` carries for a unit of theirs).
    fn worth(&self, kind: &str) -> i64;
}

/// The built-in fallback when no valuation record is given: each unit the same (+1 theirs lost, -LOSS ours lost); a change of kind is no gain.
#[derive(Clone, Copy)]
pub struct Count;
pub const LOSS: i64 = 1000;
impl Judge for Count {
    fn delta(&self, me: Role, owner: Role, _kind: &str) -> i64 { if owner == me { -LOSS } else { 1 } }
    fn change(&self, _me: Role, _owner: Role, _from: &str, _to: &str) -> i64 { 0 }
    fn worth(&self, _kind: &str) -> i64 { 1 }
    fn name(&self) -> String { "count (no record: a unit lost is a unit lost; ours first)".into() }
}

/// A VALUATION RECORD plugged in (the store's entry in use: its worths by kind name, `ours_first` -- our losses weighted by LOSS: avoided before any gain).
#[derive(Clone)]
pub struct Worth { pub name: String, pub table: HashMap<String, i64>, pub ours_first: bool }
impl Worth {
    fn w(&self, kind: &str) -> i64 { *self.table.get(kind).unwrap_or(&0) }
}
impl Judge for Worth {
    fn delta(&self, me: Role, owner: Role, kind: &str) -> i64 {
        let w = self.w(kind);
        if owner == me { -(w * if self.ours_first { LOSS } else { 1 }) } else { w }
    }
    fn change(&self, me: Role, owner: Role, from: &str, to: &str) -> i64 {
        let d = self.w(to) - if from.is_empty() { 0 } else { self.w(from) };
        if owner == me { d } else { -d }
    }
    fn worth(&self, kind: &str) -> i64 { self.w(kind) }
    fn name(&self) -> String { format!("{} {:?}{}", self.name, self.table, if self.ours_first { ", ours first" } else { "" }) }
}

pub const WIN: i64 = 1 << 40;
const NULL_MIN_DEPTH: i32 = 3;          // the null move at nodes this deep or deeper
const NULL_R: i32 = 2;                  // ... read this much shallower
const ZUGZWANG_UNITS: usize = 3;        // ... only while the mover keeps more units than this
const LMR_MIN_DEPTH: i32 = 3;           // late quiet moves reduced at nodes this deep or deeper
const LMR_AFTER: usize = 3;             // ... after this many moves read in full
const FEW_REPLIES: usize = 2;           // a node with this few legal moves is a forced answer: the line is not shortened there
pub const NODE_FUSE: u64 = 200_000_000;
const QUIET_FUSE: u32 = 16;            // the deepest capture run resolved at a horizon (a named fuse: captures end sooner)

#[derive(Clone, Copy)]
enum Bound { Exact, Lower, Upper }

struct Tt { depth: i32, value: i64, bound: Bound, best: usize }

pub struct Search<'a, G: Game, J: Judge> {
    pub g: &'a G,
    pub judge: &'a J,
    pub me: Role,
    pub nodes: u64,
    fuse: u64,
    deadline: Instant,
    tt: FxHashMap<u64, Tt>,
    stopped: bool,
    /// INSTANCE KNOWLEDGE (plan 78.8): the keys of the positions the game's record already holds -- reached again, the rules' draw (repetition) is the outcome
    pub history: std::collections::HashSet<u64>,
    path: Vec<u64>,
    /// MOVE ORDERING (plan 78.9, game-free): the quiet moves that cut a sibling off at the same ply (killers) and those that cut off anywhere (history, by the
    /// square of the depth) are read early; `ordering` off reads the old order (the comparison)
    pub ordering: bool,
    /// PRUNING (plan 78.10, game-free): the pass read shallow -- if passing already holds the bound, the node is cut (null move; never where the game has no
    /// pass or the mover has few units: a zugzwang guard); late quiet moves read two plies shallower first (re-read in full when they look better)
    pub prune: bool,
    /// THE FEW-REPLY EXTENSION (plan 78.11): a forced answer does not shorten the line
    pub extend: bool,
    killers: Vec<[Option<G::Move>; 2]>,
    hist: FxHashMap<(Role, G::Move), i64>,
    /// ROOT PRIOR (plan 78.14): indices of the root's legal moves a learned lemma proposes; read first (after the table's best), in this order. An order, never a
    /// verdict: an exact search returns the same value whatever the order.
    pub root_first: Vec<usize>,
    /// The deepening steps of the runs so far, as measured (the granular cost of the search).
    pub reads: Vec<DepthRead>,
}

/// One deepening step of `run`, as measured: the depth, the nodes it spent, its seconds, and whether it finished (the last one may have been cut by the clock).
#[derive(Debug, Clone, Copy)]
pub struct DepthRead { pub depth: i32, pub nodes: u64, pub seconds: f64, pub complete: bool }

/// The answer: the move, its value, the line (move names) and its ledger, the depth read in full, the nodes.
#[derive(Debug, Clone)]
pub struct Line { pub first: Option<String>, pub value: i64, pub line: Vec<String>, pub lost: Vec<(Role, String)>, pub depth: i32, pub nodes: u64, pub first_index: Option<usize> }

impl<'a, G: Game, J: Judge> Search<'a, G, J> {
    pub fn new(g: &'a G, judge: &'a J, me: Role, seconds: f64) -> Self {
        Search { g, judge, me, nodes: 0, fuse: NODE_FUSE, deadline: Instant::now() + std::time::Duration::from_secs_f64(seconds.max(0.0)), tt: FxHashMap::default(), stopped: false, history: Default::default(), path: vec![], ordering: true, prune: true, extend: true, killers: vec![], hist: FxHashMap::default(), root_first: vec![], reads: vec![] }
    }

    /// A search bounded by a node fuse alone (no clock): the census reads every position with the same budget.
    pub fn with_fuse(g: &'a G, judge: &'a J, me: Role, nodes: u64) -> Self {
        Search { g, judge, me, nodes: 0, fuse: nodes, deadline: Instant::now() + std::time::Duration::from_secs(1 << 30), tt: FxHashMap::default(), stopped: false, history: Default::default(), path: vec![], ordering: true, prune: true, extend: true,
                 killers: vec![], hist: FxHashMap::default(), root_first: vec![], reads: vec![] }
    }

    /// The node fuse (a named cap; the interface's budget sets it).
    pub fn set_fuse(&mut self, nodes: u64) { self.fuse = nodes; }

    /// Did the fuse or the clock cut the last reading (its values are then lower bounds of the work, not exact)?
    pub fn cut(&self) -> bool { self.stopped }

    /// EVERY FIRST MOVE'S VALUE at `depth` (each read with the full window: exact within the horizon) -> [(the move's index in `moves`, its value)], best first.
    pub fn root_values(&mut self, s: &G::State, depth: i32) -> Vec<(usize, i64)> {
        let maxing = self.g.mover(s) == Some(self.me);
        let mut out = vec![];
        for (i, m, d) in self.ordered(s, maxing, None) {
            let q = self.g.apply(s, &m);
            let v = d + self.value(&q, depth - 1, 1, i64::MIN / 2, i64::MAX / 2);
            if self.stopped { break; }
            out.push((i, v));
        }
        out.sort_by_key(|(i, v)| (-*v, *i));
        out
    }

    /// A fresh node budget for the next reading, the table kept (the census reads many neighbouring positions with one search: their sub-positions are shared).
    pub fn rebudget(&mut self, nodes: u64) { self.fuse = self.nodes + nodes; self.stopped = false; }

    /// Go on reading after the clock stopped the search: a fresh clock of `seconds` (the table kept).
    pub fn resume(&mut self, seconds: f64) {
        self.stopped = false;
        self.deadline = Instant::now() + std::time::Duration::from_secs_f64(seconds.max(0.0));
    }

    /// THE FIRST MOVES AS GOOD AS `best` at `depth` (each asked "at least best?" by a null window; the table of the run warms it) -> their indices in `moves`.
    pub fn ties(&mut self, s: &G::State, depth: i32, best: i64) -> Vec<usize> {
        let maxing = self.g.mover(s) == Some(self.me);
        let mut out = vec![];
        for (i, m, d) in self.ordered(s, maxing, None) {
            let q = self.g.apply(s, &m);
            let probe = d + self.value(&q, depth - 1, 1, best - 1 - d, best - d);
            if self.stopped { break; }
            if probe >= best { out.push(i); }
        }
        out
    }

    /// THE FIRST MOVES THAT GAIN at `depth` -- each first asked only "above nothing?" (a null window: cheap), the gaining ones then read exact -> [(index, value)],
    /// best first. The census's reading: every move that gains, none that does not.
    pub fn gainers(&mut self, s: &G::State, depth: i32) -> Vec<(usize, i64)> {
        let maxing = self.g.mover(s) == Some(self.me);
        let mut out = vec![];
        for (i, m, d) in self.ordered(s, maxing, None) {
            let q = self.g.apply(s, &m);
            let probe = d + self.value(&q, depth - 1, 1, -d, -d + 1);         // (fails high iff the move's value is at least 1)
            if self.stopped { break; }
            if probe >= 1 {
                let v = d + self.value(&q, depth - 1, 1, i64::MIN / 2, i64::MAX / 2);
                if self.stopped { break; }
                if v >= 1 { out.push((i, v)); }
            }
        }
        out.sort_by_key(|(i, v)| (-*v, *i));
        out
    }

    /// The line that starts with the move numbered `first` on `s`, then the table's best moves, with its ledger events.
    pub fn line_from(&self, s: &G::State, first: usize, n: usize) -> (Vec<String>, Vec<(Role, String)>) {
        let ms = self.g.moves(s);
        let Some(m) = ms.get(first) else { return (vec![], vec![]) };
        let mover = self.g.mover(s).unwrap_or(self.me);
        let mut ev: Vec<(Role, String)> = self.g.captured(s, m).into_iter().map(|(o, k)| (o, format!("lose {k}"))).collect();
        ev.extend(self.g.changed(s, m).into_iter().map(|(a, b)| (mover, format!("promote {} to {b}", if a.is_empty() { "-" } else { a.as_str() }))));
        let q = self.g.apply(s, m);
        let (mut names, rest) = self.line_of(&q, n.saturating_sub(1));
        names.insert(0, self.g.move_name(s, m));
        ev.extend(rest);
        (names, ev)
    }

    fn terminal(&self, v: &[i64], ply: i32) -> i64 {
        let best_other = v.iter().enumerate().filter(|(i, _)| *i != self.me).map(|(_, x)| *x).max().unwrap_or(0);
        if v[self.me] > best_other { WIN - ply as i64 } else if v[self.me] < best_other { -WIN + ply as i64 } else { 0 }
    }

    fn gain(&self, s: &G::State, m: &G::Move) -> i64 {
        let took: i64 = self.g.captured(s, m).iter().map(|(o, k)| self.judge.delta(self.me, *o, k)).sum();
        let mover = self.g.mover(s).unwrap_or(self.me);
        took + self.g.changed(s, m).iter().map(|(a, b)| self.judge.change(self.me, mover, a, b)).sum::<i64>()
    }

    fn tick(&mut self) -> bool {
        self.nodes += 1;
        if self.nodes >= self.fuse || (self.nodes & 1023 == 0 && Instant::now() >= self.deadline) { self.stopped = true; }
        self.stopped
    }

    /// The moves in their reading order: the previous best first, then the captures (by what they take for the side to move), then the rest; with each move's gain.
    fn ordered(&self, s: &G::State, maxing: bool, hint: Option<usize>) -> Vec<(usize, G::Move, i64)> {
        let mut v: Vec<(usize, G::Move, i64)> = self.g.moves(s).into_iter().enumerate().map(|(i, m)| { let d = self.gain(s, &m); (i, m, d) }).collect();
        v.sort_by_key(|(i, _, d)| (Some(*i) != hint, if maxing { -*d } else { *d }, *i));
        v
    }

    /// `ordered` at a node of the main search (ply `ply`): the previous best first, then the moves that gain (best gain first), then the killers of this ply, then
    /// the quiet moves by their history.
    fn ordered_at(&self, s: &G::State, maxing: bool, hint: Option<usize>, ply: i32) -> Vec<(usize, G::Move, i64)> {
        if !self.ordering { return self.ordered(s, maxing, hint); }
        let mover = self.g.mover(s).unwrap_or(self.me);
        let k = self.killers.get(ply as usize);
        let mut v: Vec<(usize, G::Move, i64, i64)> = self.g.moves(s).into_iter().enumerate().map(|(i, m)| {
            let d = self.gain(s, &m);
            let rank = if Some(i) == hint { i64::MIN } else if ply == 0 && self.root_first.contains(&i) { i64::MIN / 2 + self.root_first.iter().position(|x| *x == i).unwrap() as i64 } else if d != 0 { -(1 << 40) + if maxing { -d } else { d } }
                       else if k.is_some_and(|k| k.iter().any(|x| x.as_ref() == Some(&m))) { -(1 << 30) }
                       else { -*self.hist.get(&(mover, m.clone())).unwrap_or(&0) };
            (i, m, d, rank)
        }).collect();
        v.sort_by_key(|(i, _, _, r)| (*r, *i));
        v.into_iter().map(|(i, m, d, _)| (i, m, d)).collect()
    }

    fn cutoff(&mut self, s: &G::State, m: &G::Move, d: i64, ply: i32, depth: i32) {
        if !self.ordering || d != 0 { return; }
        let p = ply as usize;
        if self.killers.len() <= p { self.killers.resize(p + 1, [None, None]); }
        if self.killers[p][0].as_ref() != Some(m) { self.killers[p][1] = self.killers[p][0].take(); self.killers[p][0] = Some(m.clone()); }
        let mover = self.g.mover(s).unwrap_or(self.me);
        *self.hist.entry((mover, m.clone())).or_insert(0) += (depth as i64) * (depth as i64);
    }

    /// The capture run at a horizon: the side to move may stand (the ledger as it is) or go on capturing.
    fn quiet(&mut self, s: &G::State, ply: i32, mut alpha: i64, mut beta: i64, depth: u32) -> i64 {
        if let Some(v) = self.g.outcome(s) { return self.terminal(&v, ply); }
        if self.tick() || depth >= QUIET_FUSE { return 0; }
        let maxing = self.g.mover(s) == Some(self.me);
        let mut best = 0i64;                                   // standing: no further capture
        if maxing { if best >= beta { return best; } alpha = alpha.max(best); } else { if best <= alpha { return best; } beta = beta.min(best); }
        for (_i, m, d) in self.ordered(s, maxing, None) {
            if d == 0 { continue; }                            // (captures only)
            let q = self.g.apply(s, &m);
            let v = d + self.quiet(&q, ply + 1, alpha - d, beta - d, depth + 1);
            if self.stopped { return best; }
            if maxing { if v > best { best = v; } alpha = alpha.max(v); } else { if v < best { best = v; } beta = beta.min(v); }
            if alpha >= beta { break; }
        }
        best
    }

    /// Minimax of the judge's sum over the FUTURE of `s` (what is lost from here on), `depth` plies, alpha-beta.
    fn value(&mut self, s: &G::State, depth: i32, ply: i32, mut alpha: i64, mut beta: i64) -> i64 {
        if let Some(v) = self.g.outcome(s) { return self.terminal(&v, ply); }
        let key = self.g.key(s);
        if ply > 0 && (self.history.contains(&key) || self.path.contains(&key)) { return 0; }      // (a position met again: the rules' draw -- nothing more is lost or won)
        if depth <= 0 { return self.quiet(s, ply, alpha, beta, 0); }
        if self.tick() { return 0; }
        let (a0, b0) = (alpha, beta);
        let mut hint = None;
        if let Some(e) = self.tt.get(&key) {
            hint = Some(e.best);
            if e.depth >= depth {
                match e.bound { Bound::Exact => return e.value, Bound::Lower => alpha = alpha.max(e.value), Bound::Upper => beta = beta.min(e.value) }
                if alpha >= beta { return e.value; }
            }
        }
        let maxing = self.g.mover(s) == Some(self.me);
        if self.prune && ply > 0 && depth >= NULL_MIN_DEPTH && beta < WIN - 1000 && alpha > -WIN + 1000 {
            let mover = self.g.mover(s);
            let mine = self.g.units(s).iter().filter(|(o, _)| Some(*o) == mover).count();
            if mine > ZUGZWANG_UNITS {
                if let Some(p) = self.g.pass(s) {
                    let v = self.value(&p, depth - 1 - NULL_R, ply + 1, alpha, beta);
                    if self.stopped { return 0; }
                    if (maxing && v >= beta) || (!maxing && v <= alpha) { return v; }
                }
            }
        }
        let mut best = if maxing { i64::MIN } else { i64::MAX };
        let mut best_i = 0;
        self.path.push(key);
        let order = self.ordered_at(s, maxing, hint, ply);
        // THE FEW-REPLY EXTENSION (plan 78.11, game-free): a node with at most FEW_REPLIES legal moves (a forced answer) does not shorten the line
        let ext = if self.extend && order.len() <= FEW_REPLIES && ply < 2 * depth + 8 { 1 } else { 0 };
        let depth = depth + ext;
        for (n_read, (i, m, d)) in order.into_iter().enumerate() {
            let q = self.g.apply(s, &m);
            let late = self.prune && depth >= LMR_MIN_DEPTH && n_read >= LMR_AFTER && d == 0 && Some(i) != hint;
            let mut v = if late { d + self.value(&q, depth - 3, ply + 1, alpha - d, beta - d) } else { 0 };
            if !late || (maxing && v > alpha) || (!maxing && v < beta) {
                v = d + self.value(&q, depth - 1, ply + 1, alpha - d, beta - d);
            }
            if self.stopped { self.path.pop(); return if best == i64::MIN || best == i64::MAX { 0 } else { best }; }
            if (maxing && v > best) || (!maxing && v < best) { best = v; best_i = i; }
            if maxing { alpha = alpha.max(v); } else { beta = beta.min(v); }
            if alpha >= beta { self.cutoff(s, &m, d, ply, depth); break; }
        }
        self.path.pop();
        let bound = if best <= a0 { Bound::Upper } else if best >= b0 { Bound::Lower } else { Bound::Exact };
        self.tt.insert(key, Tt { depth, value: best, bound, best: best_i });
        best
    }

    /// The principal line from `s` (by the table's best moves) with its ledger.
    fn line_of(&self, s: &G::State, n: usize) -> (Vec<String>, Vec<(Role, String)>) {
        // (the ledger's events: "lose <kind>" for a unit taken, "promote <from> to <to>" for a change of kind)
        let (mut names, mut lost) = (vec![], vec![]);
        let mut cur = s.clone();
        for _ in 0..n {
            if self.g.outcome(&cur).is_some() { break; }
            let Some(e) = self.tt.get(&self.g.key(&cur)) else { break };
            let ms = self.g.moves(&cur);
            let Some(m) = ms.get(e.best) else { break };
            names.push(self.g.move_name(&cur, m));
            lost.extend(self.g.captured(&cur, m).into_iter().map(|(o, k)| (o, format!("lose {k}"))));
            let mover = self.g.mover(&cur).unwrap_or(self.me);
            lost.extend(self.g.changed(&cur, m).into_iter().map(|(a, b)| (mover, format!("promote {} to {b}", if a.is_empty() { "-" } else { a.as_str() }))));
            cur = self.g.apply(&cur, m);
        }
        (names, lost)
    }

    /// Iterative deepening to `max_depth` while the clock lasts -> the deepest line read in full.
    pub fn run(&mut self, s: &G::State, max_depth: i32) -> Line {
        let mut out = Line { first: None, value: 0, line: vec![], lost: vec![], depth: 0, nodes: 0, first_index: None };
        for d in 1..=max_depth {
            let (n0, t0) = (self.nodes, Instant::now());
            let v = self.value(s, d, 0, i64::MIN / 2, i64::MAX / 2);
            self.reads.push(DepthRead { depth: d, nodes: self.nodes - n0, seconds: t0.elapsed().as_secs_f64(), complete: !self.stopped });
            if self.stopped { break; }
            let (line, lost) = self.line_of(s, d as usize + 8);
            let fi = self.tt.get(&self.g.key(s)).map(|e| e.best);
            out = Line { first: line.first().cloned(), value: v, line, lost, depth: d, nodes: self.nodes, first_index: fi };
            if v.abs() >= WIN - 1000 { break; }                // the game's own outcome is read: no deeper line changes it
        }
        out.nodes = self.nodes;
        out
    }
}
