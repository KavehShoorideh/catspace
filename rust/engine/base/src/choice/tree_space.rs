//! PLAN 79.32 PART A (THE EXPLICIT TREE AS ONE CASE OF THE INTERFACE; SYSTEM_SPEC 11 "The pure search is mathematics and lives in layer 0", 13 item 50): a finite `Tree` seen as a `Space`, so that the forcing reading of `choice::can_force_set` is `walk::Walk` and the readings that weigh numbers
//! (`guarantee`, `value`, `force_chance`: what a set can make sure of) are one fold over the same stepping (`TreeSpace::value`). No tree is copied: the space holds the line of nodes being walked, as references into the tree.
//! INTERFACE. `TreeSpace::forcing(tree, set, holds)`: property 0 is "the result holds `holds`"; a chooser in `set` picks as the asker, any other as the rest; a choice by chance is read as every outcome of positive chance (the rest pick); a choice at once is read as the two parties' picks in turn:
//!   both parties the set's own: some cell; neither: every cell; the row party only: some row, then every column; the column party only: some column, then every row (pure choices; a party counts as the set's own only when all its members are in the set).
//!   `TreeSpace::valuing(tree, set)` the same stepping with a choice by chance kept in full (every outcome with its chance), for `value`. Options are numbered in the tree's own order; a node is its tree node and, inside a choice at once, the stage (the whole choice, a row picked, a column picked).
//!   Called by: `choice::can_force_set`, `choice::secure` (and so `guarantee`, `value`, `value_between`, `force_chance_set`).
use super::walk::{Cut, Look, Node, Space, Steer};
use super::{Party, Tree};
use crate::number::{Num, NumError};

/// A node's stage inside a choice at once.
#[derive(Clone, Copy)]
enum Stage { Whole, Row(usize), Col(usize) }

struct Frame<'t, R> { tree: &'t Tree<R>, stage: Stage }

/// An explicit tree as a node that gives its options when asked.
pub struct TreeSpace<'t, R, F: Fn(&R) -> bool> {
    stack: Vec<Frame<'t, R>>,
    set: &'t Party,
    holds: F,
    /// a choice by chance is read as every outcome of positive chance (forcing), or in full (valuing)
    forcing: bool,
}

impl<'t, R, F: Fn(&R) -> bool> TreeSpace<'t, R, F> {
    /// The forcing reading: property 0 is "the result holds `holds`".
    pub fn forcing(tree: &'t Tree<R>, set: &'t Party, holds: F) -> Self { TreeSpace { stack: vec![Frame { tree, stage: Stage::Whole }], set, holds, forcing: true } }

    fn top(&self) -> &Frame<'t, R> { self.stack.last().expect("a node is current") }

    /// Whether the row party and the column party of a choice at once are the set's own.
    fn own_parties(&self, t: &Tree<R>) -> (bool, bool) {
        match t { Tree::Table { row_party, col_party, .. } => (row_party.is_part_of(self.set), col_party.is_part_of(self.set)), _ => (false, false) }
    }

    /// How many options the current node has, and the tree node option `ix` leads to with the stage it is entered at.
    fn count(&self) -> usize {
        let f = self.top();
        match (f.tree, f.stage) {
            (Tree::Result(_), _) => 0,
            (Tree::Turn { options, .. }, _) => options.len(),
            (Tree::Chance(s), _) => s.outcomes().len(),
            (Tree::Table { rows, cols, .. }, Stage::Whole) => match self.own_parties(f.tree) { (true, true) | (false, false) => rows * cols, (true, false) => *rows, (false, true) => *cols },
            (Tree::Table { cols, .. }, Stage::Row(_)) => *cols,
            (Tree::Table { rows, .. }, Stage::Col(_)) => *rows,
        }
    }

    /// The value of the tree for the set, in the score of a result: at the turn of a member the best option, at any other turn the worst, by chance the expectation, at once (pure choices) by whether the parties are the set's own. The first overflow is said.
    pub fn value(&mut self, score: &mut impl FnMut(&R) -> Result<Num, NumError>) -> Result<Num, NumError> {
        if let Tree::Result(r) = self.top().tree { return score(r); }
        let n = self.count();
        if let (Tree::Chance(_), Stage::Whole) = (self.top().tree, self.top().stage) {
            let mut acc = Num::whole(0);
            for ix in 0..n {
                let w = if let Tree::Chance(s) = self.top().tree { s.outcomes()[ix].1 } else { unreachable!() };
                self.enter(&ix);
                let v = self.value(score);
                self.leave();
                acc = acc.add(&v?.mul(&w)?)?;
            }
            return Ok(acc);
        }
        let best = self.own_turn();
        let mut got: Option<Num> = None;
        for ix in 0..n {
            self.enter(&ix);
            let v = self.value(score);
            self.leave();
            let v = v?;
            got = Some(match got { None => v, Some(a) => if best { a.max(v) } else { a.min(v) } });
        }
        Ok(got.expect("a choice has an option"))
    }
}

impl<'t, R> TreeSpace<'t, R, fn(&R) -> bool> {
    /// The same stepping for the readings that weigh numbers: no property, a choice by chance kept in full.
    pub fn valuing(tree: &'t Tree<R>, set: &'t Party) -> Self { TreeSpace { stack: vec![Frame { tree, stage: Stage::Whole }], set, holds: |_| false, forcing: false } }
}

impl<'t, R, F: Fn(&R) -> bool> Node for TreeSpace<'t, R, F> {
    type Opt = usize;

    fn own_turn(&self) -> bool {
        let f = self.top();
        match (f.tree, f.stage) {
            (Tree::Turn { by, .. }, _) => self.set.contains(by),
            (Tree::Table { .. }, Stage::Whole) => !matches!(self.own_parties(f.tree), (false, false)),
            _ => false,
        }
    }

    fn options(&mut self, out: &mut Vec<usize>) {
        let n = self.count();
        let skip_null = self.forcing && matches!(self.top().tree, Tree::Chance(_));
        for ix in 0..n {
            if skip_null { if let Tree::Chance(s) = self.top().tree { if s.outcomes()[ix].1 <= Num::whole(0) { continue; } } }
            out.push(ix);
        }
    }

    fn enter(&mut self, ix: &usize) {
        let ix = *ix;
        let f = self.top();
        let next = match (f.tree, f.stage) {
            (Tree::Turn { options, .. }, _) => Frame { tree: &options[ix], stage: Stage::Whole },
            (Tree::Chance(s), _) => Frame { tree: &s.outcomes()[ix].0, stage: Stage::Whole },
            (Tree::Table { cells, .. }, Stage::Whole) => match self.own_parties(f.tree) {
                (true, true) | (false, false) => Frame { tree: &cells[ix], stage: Stage::Whole },
                (true, false) => Frame { tree: f.tree, stage: Stage::Row(ix) },
                (false, true) => Frame { tree: f.tree, stage: Stage::Col(ix) },
            },
            (Tree::Table { cols, cells, .. }, Stage::Row(i)) => Frame { tree: &cells[i * cols + ix], stage: Stage::Whole },
            (Tree::Table { cols, cells, .. }, Stage::Col(j)) => Frame { tree: &cells[ix * cols + j], stage: Stage::Whole },
            (Tree::Result(_), _) => unreachable!("a result has no option"),
        };
        self.stack.push(next);
    }

    fn leave(&mut self) { self.stack.pop(); }
}

impl<'t, R, F: Fn(&R) -> bool> Space for TreeSpace<'t, R, F> {
    const CYCLES: bool = false;

    fn key(&self) -> u64 { self.top().tree as *const Tree<R> as u64 }

    fn look(&mut self) -> Look {
        match self.top().tree {
            Tree::Result(r) => { let m = (self.holds)(r) as u64; Look { bound: m, held: m, ended: true } }
            _ => Look { bound: 0, held: 0, ended: false },
        }
    }

    fn met_before(&self, _key: u64) -> bool { false }

    fn breaks(&mut self, _cand: u64, _opts: &[usize]) -> u64 { 0 }
}

/// What steers a walk of an explicit tree: nothing. No budget, no order, no kept answers.
pub struct Free;

impl<S: Space> Steer<S> for Free {
    #[inline(always)] fn spend(&mut self, _asked: super::walk::Ask) -> Result<(), Cut> { Ok(()) }
    #[inline(always)] fn order(&mut self, _space: &S, _opts: &mut Vec<S::Opt>, _first: usize) {}
    #[inline(always)] fn keeps(&self) -> bool { false }
}
