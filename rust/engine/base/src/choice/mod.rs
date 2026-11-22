//! LAYER 0, MODULE 6: OPPOSED CHOICE (plan 79.30; plan 79.31 parts A, B and C: choosers with wants of their own, the mixed choice, the choice against a belief; SYSTEM_SPEC 11 "Layer 0 holds six things" item 6, "Layer 0 is made right", the block on belief and
//! choosing by chance, section 13 item 49). The logic of choosers: "there is a choice such that for every counter-choice". THE MEANING that the search of layer 4 is held against (plan 79.31 part F, a test in cs_tactics); the search is not moved here.
//!
//! WHO WANTS WHAT IS NOT ASSUMED. A CHOOSER is an identity (`Chooser::new(id)`), any number of them. Each has an ORDER of its own over results (`Wants`). Two choosers are OPPOSED when one order is the reverse of the other on the results of a tree, ALLIED when
//! the orders agree on them (both hold when every result is a tie). That one chooser is set against another is never read off the form of a tree: it is asked of the orders (`Wants::opposed`, `Wants::allied`), and a reading that needs it says so and refuses a tree whose orders do not give it.
//!
//! INTERFACE (callers: tests of layer 0 now; layer 4's test as the meaning to be held against; layers above that say who wants what).
//!   `Chooser`, `Party = Set<Chooser>` (choosers that pick as one). `Chooser::A`, `Chooser::B` are the first two identities, kept for the two-chooser calls.
//!   `Tree<R>`: a finite tree whose nodes are a result `Tree::Result(R)`; a choice IN TURN by one chooser `Tree::Turn { by, options }` (at least one option); a choice AT ONCE between two PARTIES `Tree::Table { row_party, col_party, rows, cols, cells }` (the row party
//!     picks a row, the column party a column, cells in row-major order, each a tree; the two parties are not empty and share no chooser); a choice BY CHANCE `Tree::Chance(Spread<Tree<R>>)`. Made by `Tree::turn(by, options)`, `Tree::table(rows, cols, cells)` (the
//!     parties are `A` and `B`), `Tree::table_between(row_party, col_party, rows, cols, cells)`, `Tree::chance(Vec<(Tree, Num)>)`. `choosers() -> Party`, `results() -> Vec<&R>`, `has_chance()`, `has_table()`, `size()`.
//!   `Order<R>`: an order of preference over results by a score (more is better): `Order::by(impl Fn(&R) -> Num)`, `score`, `prefers`, `compare`, `reversed()`; cheap to clone. `Wants<R>`: `Wants::new().with(chooser, order)`, `order(chooser)`,
//!     `opposed(a, b, results)`, `allied(a, b, results)`, `zero_sum(a, b, results)` (the two scores add to nought on every result: what an expectation over chance needs).
//!   READING ONE, FORCING. `can_force_set(&Tree, who: &Party, holds) -> bool`: the set `who` can make the property hold of the result against every counter-choice of the rest: at a turn of a member some option, at a turn of any other every option; at once the
//!     row party and the column party each count as one: one the set holds picks, one it does not hold picks against it (a party counts as the set's own only when all its members are in the set); by chance every outcome of positive chance.
//!     `can_force(&Tree, who: Chooser, holds)` is the case of a set of one. `force_chance_set`, `force_chance`: the least chance with which the set can make it hold (chance nodes weighed by their chances; exact on exact chances).
//!   READING TWO, WHAT ALLIED CHOOSERS CAN MAKE SURE OF. `guarantee(&Tree, side: &Party, order: &Order<R>) -> Result<Num, NumError>`: in the order, what the set can make sure of whatever the rest do (the rest are taken as acting against the set; no want is read for them);
//!     `guarantee_allied(&Tree, side, &Wants<R>) -> Result<Num, ChoiceError>` first asks that the members' orders agree on the results.
//!   READING THREE, THE VALUE OF TWO OPPOSED PARTIES. `value_between(&Tree, &Wants, p: &Party, q: &Party) -> Result<Bounds, ChoiceError>`: asks that every chooser of the tree is in `p` or `q`, that each side is allied and the two are opposed on the results (and, with a chance
//!     node, that the scores add to nought); the lower value is what `p` can make sure of with pure choices, the upper what `q` holds `p` down to with pure choices. `value(&Tree, &Order) -> Result<Bounds, NumError>` is the case of `A` against `B` with B's order the reverse of A's.
//!   READING FOUR, THE MIXED CHOICE. `mixed_value(&Tree, &Wants, p, q) -> Result<Num, ChoiceError>` and `mixed_table(...) -> Result<MixedTable, ChoiceError>`: at a choice at once between the two sides each side picks a SPREAD over its own options; the value is the one number
//!     both sides can hold to, found EXACTLY by the linear program of `lp` (`solve_matrix_game`); the value of a cell is the value of its subtree. Needs the scores to add to nought. The value lies between the lower and the upper of pure choices; a table with a pure answer gives it with chance one.
//!   READING FIVE, A CHOICE AGAINST A BELIEF. `choose_against(&[O], &Belief<S>, payoff) -> Result<Choice, ChoiceError>`: the expectation of each option under the belief and the first best option. Against a belief that does not change a single option is best (no spread of options does
//!     better: `mix_expectation` is a weighted mean and no mean is above its best part); choosing by chance pays only against a side that reads the chooser in turn, which is the mixed choice, not this.
//!   `ChoiceError::{NoOption, BadTable, BadParty, NoOrder(Chooser), NotCovered(Chooser), SplitParty, NotAllied, NotOpposed, NotZeroSum, Chance(ChanceError), Number(NumError), Lp(LpError)}`.
//! LAWS: lean/ideas/Ideas/Layer0/Choice.lean (in a finite tree of turns without chance exactly one of "A can force P" and "B can force not-P" holds; the lower value is not above the upper; the guarantees of two choosers who hold each other to a value are
//! that value; against a fixed belief no spread of options beats the best option). The minimax theorem itself (that every table has such spreads) is OPEN in Lean and said so there; the program here finds them and the certificate is tested.
pub mod lp;

use crate::chance::{Belief, ChanceError, Spread};
use crate::number::{Num, NumError};
use crate::sets::Set;
use lp::{solve_matrix_game, LpError};
use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::rc::Rc;

/// A chooser: an identity, nothing more. Any number of them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Chooser(u32);

impl Chooser {
    /// The first two identities, for the calls of two choosers.
    pub const A: Chooser = Chooser(0);
    pub const B: Chooser = Chooser(1);
    pub fn new(id: u32) -> Chooser { Chooser(id) }
    pub fn id(self) -> u32 { self.0 }
}

/// Choosers that pick as one.
pub type Party = Set<Chooser>;

/// A party of the given choosers.
pub fn party(members: &[Chooser]) -> Party { Set::from_items(members.iter().copied()) }

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChoiceError {
    NoOption, BadTable, BadParty,
    /// a chooser the reading needs has no order
    NoOrder(Chooser),
    /// a chooser of the tree is on neither side
    NotCovered(Chooser),
    /// a party at a choice at once has members on both sides
    SplitParty,
    NotAllied, NotOpposed, NotZeroSum,
    Chance(ChanceError), Number(NumError), Lp(LpError),
}

impl std::fmt::Display for ChoiceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ChoiceError::NoOption => f.write_str("a choice has at least one option"),
            ChoiceError::BadTable => f.write_str("a table has rows times columns cells, and at least one of each"),
            ChoiceError::BadParty => f.write_str("the two parties of a table are not empty and share no chooser"),
            ChoiceError::NoOrder(c) => write!(f, "chooser {} has no order", c.id()),
            ChoiceError::NotCovered(c) => write!(f, "chooser {} is on neither side", c.id()),
            ChoiceError::SplitParty => f.write_str("a party of a choice at once has members on both sides"),
            ChoiceError::NotAllied => f.write_str("the orders of one side do not agree on the results"),
            ChoiceError::NotOpposed => f.write_str("the orders of the two sides are not the reverse of each other on the results"),
            ChoiceError::NotZeroSum => f.write_str("the scores of the two sides do not add to nought on every result"),
            ChoiceError::Chance(e) => write!(f, "{e}"),
            ChoiceError::Number(e) => write!(f, "{e}"),
            ChoiceError::Lp(e) => write!(f, "{e}"),
        }
    }
}

impl From<NumError> for ChoiceError { fn from(e: NumError) -> ChoiceError { ChoiceError::Number(e) } }
impl From<ChanceError> for ChoiceError { fn from(e: ChanceError) -> ChoiceError { ChoiceError::Chance(e) } }
impl From<LpError> for ChoiceError { fn from(e: LpError) -> ChoiceError { ChoiceError::Lp(e) } }

/// A finite tree of choices.
#[derive(Debug, Clone, PartialEq)]
pub enum Tree<R> {
    Result(R),
    /// A choice in turn by one chooser.
    Turn { by: Chooser, options: Vec<Tree<R>> },
    /// A choice at once between two parties: the row party picks a row, the column party a column; the cells in row-major order.
    Table { row_party: Party, col_party: Party, rows: usize, cols: usize, cells: Vec<Tree<R>> },
    /// A choice by chance.
    Chance(Spread<Tree<R>>),
}

impl<R> Tree<R> {
    pub fn turn(by: Chooser, options: Vec<Tree<R>>) -> Result<Tree<R>, ChoiceError> {
        if options.is_empty() { Err(ChoiceError::NoOption) } else { Ok(Tree::Turn { by, options }) }
    }
    /// A choice at once between `A` (the rows) and `B` (the columns).
    pub fn table(rows: usize, cols: usize, cells: Vec<Tree<R>>) -> Result<Tree<R>, ChoiceError> { Tree::table_between(party(&[Chooser::A]), party(&[Chooser::B]), rows, cols, cells) }
    /// A choice at once between two parties.
    pub fn table_between(row_party: Party, col_party: Party, rows: usize, cols: usize, cells: Vec<Tree<R>>) -> Result<Tree<R>, ChoiceError> {
        if rows == 0 || cols == 0 || cells.len() != rows * cols { return Err(ChoiceError::BadTable); }
        if row_party.is_empty() || col_party.is_empty() || !row_party.is_disjoint(&col_party) { return Err(ChoiceError::BadParty); }
        Ok(Tree::Table { row_party, col_party, rows, cols, cells })
    }
    pub fn chance(outcomes: Vec<(Tree<R>, Num)>) -> Result<Tree<R>, ChoiceError> { Spread::new(outcomes).map(Tree::Chance).map_err(ChoiceError::Chance) }

    /// How many nodes the tree has.
    pub fn size(&self) -> usize {
        1 + match self {
            Tree::Result(_) => 0,
            Tree::Turn { options, .. } => options.iter().map(Tree::size).sum(),
            Tree::Table { cells, .. } => cells.iter().map(Tree::size).sum(),
            Tree::Chance(s) => s.outcomes().iter().map(|(t, _)| t.size()).sum(),
        }
    }
    /// Every chooser that picks anywhere in the tree.
    pub fn choosers(&self) -> Party {
        match self {
            Tree::Result(_) => Party::new(),
            Tree::Turn { by, options } => options.iter().fold(party(&[*by]), |a, o| a.union(&o.choosers())),
            Tree::Table { row_party, col_party, cells, .. } => cells.iter().fold(row_party.union(col_party), |a, o| a.union(&o.choosers())),
            Tree::Chance(s) => s.outcomes().iter().fold(Party::new(), |a, (o, _)| a.union(&o.choosers())),
        }
    }
    /// Every result at the end of the tree (a result met twice is listed twice).
    pub fn results(&self) -> Vec<&R> {
        match self {
            Tree::Result(r) => vec![r],
            Tree::Turn { options, .. } => options.iter().flat_map(|o| o.results()).collect(),
            Tree::Table { cells, .. } => cells.iter().flat_map(|o| o.results()).collect(),
            Tree::Chance(s) => s.outcomes().iter().flat_map(|(o, _)| o.results()).collect(),
        }
    }
    pub fn has_chance(&self) -> bool {
        match self { Tree::Result(_) => false, Tree::Turn { options, .. } => options.iter().any(Tree::has_chance), Tree::Table { cells, .. } => cells.iter().any(Tree::has_chance), Tree::Chance(_) => true }
    }
    pub fn has_table(&self) -> bool {
        match self { Tree::Result(_) => false, Tree::Turn { options, .. } => options.iter().any(Tree::has_table), Tree::Table { .. } => true, Tree::Chance(s) => s.outcomes().iter().any(|(o, _)| o.has_table()) }
    }
}

// ------------------------------------------------------------------------------------------------------------------------------ orders and wants

/// An order of preference over results, by a score: more is better.
pub struct Order<R> { score: Rc<dyn Fn(&R) -> Num>, flipped: bool }

impl<R> Clone for Order<R> { fn clone(&self) -> Order<R> { Order { score: self.score.clone(), flipped: self.flipped } } }

impl<R: 'static> Order<R> {
    pub fn by(score: impl Fn(&R) -> Num + 'static) -> Order<R> { Order { score: Rc::new(score), flipped: false } }
    /// The score of a result in this order (the reverse order scores minus the score); overflow is said.
    pub fn score(&self, r: &R) -> Result<Num, NumError> { let s = (self.score)(r); if self.flipped { s.neg() } else { Ok(s) } }
    pub fn compare(&self, a: &R, b: &R) -> Ordering { let o = (self.score)(a).cmp(&(self.score)(b)); if self.flipped { o.reverse() } else { o } }
    /// The first is better than the second.
    pub fn prefers(&self, a: &R, b: &R) -> bool { self.compare(a, b) == Ordering::Greater }
    /// The order of one who wants the opposite: the same scores, reversed.
    pub fn reversed(self) -> Order<R> { Order { score: self.score, flipped: !self.flipped } }
}

/// What each chooser wants: an order of its own for each.
pub struct Wants<R> { orders: BTreeMap<Chooser, Order<R>> }

impl<R: 'static> Default for Wants<R> { fn default() -> Wants<R> { Wants::new() } }

impl<R: 'static> Wants<R> {
    pub fn new() -> Wants<R> { Wants { orders: BTreeMap::new() } }
    pub fn with(mut self, who: Chooser, order: Order<R>) -> Wants<R> { self.orders.insert(who, order); self }
    pub fn order(&self, who: Chooser) -> Option<&Order<R>> { self.orders.get(&who) }
    fn need(&self, who: Chooser) -> Result<&Order<R>, ChoiceError> { self.orders.get(&who).ok_or(ChoiceError::NoOrder(who)) }
    /// The two choosers' orders agree on every pair of the results (a tie in one is a tie in the other).
    pub fn allied(&self, a: Chooser, b: Chooser, results: &[&R]) -> Result<bool, ChoiceError> {
        let (oa, ob) = (self.need(a)?, self.need(b)?);
        Ok(results.iter().all(|r| results.iter().all(|s| oa.compare(r, s) == ob.compare(r, s))))
    }
    /// The two choosers' orders are each the reverse of the other on every pair of the results.
    pub fn opposed(&self, a: Chooser, b: Chooser, results: &[&R]) -> Result<bool, ChoiceError> {
        let (oa, ob) = (self.need(a)?, self.need(b)?);
        Ok(results.iter().all(|r| results.iter().all(|s| oa.compare(r, s) == ob.compare(r, s).reverse())))
    }
    /// The two scores add to nought on every result (so that what one gains the other loses, in numbers and not only in order).
    pub fn zero_sum(&self, a: Chooser, b: Chooser, results: &[&R]) -> Result<bool, ChoiceError> {
        let (oa, ob) = (self.need(a)?, self.need(b)?);
        for r in results { if oa.score(r)?.add(&ob.score(r)?)? != Num::whole(0) { return Ok(false); } }
        Ok(true)
    }
}

// ------------------------------------------------------------------------------------------------------------------------------ reading one: forcing

/// The two sides of a choice at once as the set sees them: whether the row party and the column party are the set's own.
fn own(side: &Party, row_party: &Party, col_party: &Party) -> (bool, bool) { (row_party.is_part_of(side), col_party.is_part_of(side)) }

/// READING ONE. Whether the set `who` can make the property hold of the result against every counter-choice of the rest. At once, pure choices only.
pub fn can_force_set<R>(t: &Tree<R>, who: &Party, holds: &dyn Fn(&R) -> bool) -> bool {
    match t {
        Tree::Result(r) => holds(r),
        Tree::Turn { by, options } => if who.contains(by) { options.iter().any(|o| can_force_set(o, who, holds)) } else { options.iter().all(|o| can_force_set(o, who, holds)) },
        Tree::Table { row_party, col_party, rows, cols, cells } => {
            let c = |i: usize, j: usize| can_force_set(&cells[i * cols + j], who, holds);
            match own(who, row_party, col_party) {
                (true, true) => (0..*rows).any(|i| (0..*cols).any(|j| c(i, j))),
                (true, false) => (0..*rows).any(|i| (0..*cols).all(|j| c(i, j))),
                (false, true) => (0..*cols).any(|j| (0..*rows).all(|i| c(i, j))),
                (false, false) => (0..*rows).all(|i| (0..*cols).all(|j| c(i, j))),
            }
        }
        Tree::Chance(s) => s.outcomes().iter().all(|(o, p)| *p <= Num::whole(0) || can_force_set(o, who, holds)),
    }
}

/// READING ONE for one chooser: the case of a set of one.
pub fn can_force<R>(t: &Tree<R>, who: Chooser, holds: &dyn Fn(&R) -> bool) -> bool { can_force_set(t, &party(&[who]), holds) }

fn pick(vals: impl IntoIterator<Item = Num>, best: bool) -> Num {
    vals.into_iter().reduce(|a, b| if best { a.max(b) } else { a.min(b) }).expect("a choice has an option")
}

/// What the set can make sure of in a score it wants more of, the rest acting against it: at the turn of a member the best option, at any other turn the worst, by chance the expectation, at once (pure choices) by whether the parties are the set's own.
fn secure<R>(t: &Tree<R>, side: &Party, score: &dyn Fn(&R) -> Result<Num, NumError>) -> Result<Num, NumError> {
    match t {
        Tree::Result(r) => score(r),
        Tree::Turn { by, options } => {
            let vals = options.iter().map(|o| secure(o, side, score)).collect::<Result<Vec<_>, _>>()?;
            Ok(pick(vals, side.contains(by)))
        }
        Tree::Table { row_party, col_party, rows, cols, cells } => {
            let cell = |i: usize, j: usize| secure(&cells[i * cols + j], side, score);
            let grid = (0..*rows).map(|i| (0..*cols).map(|j| cell(i, j)).collect::<Result<Vec<_>, _>>()).collect::<Result<Vec<_>, _>>()?;
            Ok(pure_choice(&grid, own(side, row_party, col_party)))
        }
        Tree::Chance(s) => {
            let mut acc = Num::whole(0);
            for (o, p) in s.outcomes() { acc = acc.add(&secure(o, side, score)?.mul(p)?)?; }
            Ok(acc)
        }
    }
}

/// The value of a grid of cells for the set when each party that is its own picks the best and the others the worst (pure choices).
fn pure_choice(grid: &[Vec<Num>], own: (bool, bool)) -> Num {
    let (rows, cols) = (grid.len(), grid[0].len());
    match own {
        (true, true) => pick(grid.iter().flatten().copied(), true),
        (true, false) => pick((0..rows).map(|i| pick(grid[i].iter().copied(), false)), true),
        (false, true) => pick((0..cols).map(|j| pick((0..rows).map(|i| grid[i][j]), false)), true),
        (false, false) => pick(grid.iter().flatten().copied(), false),
    }
}

/// The least chance with which the set can make the property hold (chance nodes weighed by their chances).
pub fn force_chance_set<R>(t: &Tree<R>, who: &Party, holds: &dyn Fn(&R) -> bool) -> Result<Num, NumError> { secure(t, who, &|r| Ok(Num::whole(holds(r) as i64))) }

/// The least chance with which one chooser can make the property hold.
pub fn force_chance<R>(t: &Tree<R>, who: Chooser, holds: &dyn Fn(&R) -> bool) -> Result<Num, NumError> { force_chance_set(t, &party(&[who]), holds) }

// ------------------------------------------------------------------------------------------------------------------------------ reading two: what allied choosers can make sure of

/// READING TWO. In the order given, what the set can make sure of whatever the rest do (the rest are taken as acting against it). Pure choices at once.
pub fn guarantee<R: 'static>(t: &Tree<R>, side: &Party, order: &Order<R>) -> Result<Num, NumError> { secure(t, side, &|r| order.score(r)) }

/// The members of the set must be allied on the results of the tree; the guarantee is in their one order.
pub fn guarantee_allied<R: 'static>(t: &Tree<R>, side: &Party, wants: &Wants<R>) -> Result<Num, ChoiceError> {
    let first = *side.iter().next().ok_or(ChoiceError::BadParty)?;
    let results = t.results();
    for m in side.iter() { if !wants.allied(first, *m, &results)? { return Err(ChoiceError::NotAllied); } }
    Ok(guarantee(t, side, wants.need(first)?)?)
}

// ------------------------------------------------------------------------------------------------------------------------------ reading three: the value of two opposed parties

/// The two values of a tree for the first side.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Bounds { pub lower: Num, pub upper: Num }

/// THE CASE OF A AGAINST B. The values for A when A and B want opposite things (B's order is A's reversed). Without a choice at once lower equals upper. Choosers of the tree other than A and B act against whoever is asking.
pub fn value<R: 'static>(t: &Tree<R>, a_order: &Order<R>) -> Result<Bounds, NumError> {
    let lower = secure(t, &party(&[Chooser::A]), &|r| a_order.score(r))?;
    // what B can make sure of, in the reversed score, is minus what B can hold A down to
    let b_secure = secure(t, &party(&[Chooser::B]), &|r| a_order.score(r)?.neg())?;
    Ok(Bounds { lower, upper: b_secure.neg()? })
}

/// What a reading of two opposed sides asks of the tree, and the order of the first side's first member.
fn two_sides<'a, R: 'static>(t: &Tree<R>, wants: &'a Wants<R>, p: &Party, q: &Party, zero_sum: bool) -> Result<&'a Order<R>, ChoiceError> {
    if p.is_empty() || q.is_empty() || !p.is_disjoint(q) { return Err(ChoiceError::BadParty); }
    if let Some(c) = t.choosers().iter().find(|c| !p.contains(c) && !q.contains(c)) { return Err(ChoiceError::NotCovered(*c)); }
    fn parties<R>(t: &Tree<R>, p: &Party, q: &Party) -> bool {
        match t {
            Tree::Result(_) => true,
            Tree::Turn { options, .. } => options.iter().all(|o| parties(o, p, q)),
            Tree::Table { row_party, col_party, cells, .. } => (row_party.is_part_of(p) || row_party.is_part_of(q)) && (col_party.is_part_of(p) || col_party.is_part_of(q)) && cells.iter().all(|o| parties(o, p, q)),
            Tree::Chance(s) => s.outcomes().iter().all(|(o, _)| parties(o, p, q)),
        }
    }
    if !parties(t, p, q) { return Err(ChoiceError::SplitParty); }
    let (fp, fq) = (*p.iter().next().expect("not empty"), *q.iter().next().expect("not empty"));
    let results = t.results();
    for side in [p, q] {
        let first = *side.iter().next().expect("not empty");
        for m in side.iter() { if !wants.allied(first, *m, &results)? { return Err(ChoiceError::NotAllied); } }
    }
    if !wants.opposed(fp, fq, &results)? { return Err(ChoiceError::NotOpposed); }
    if zero_sum && !wants.zero_sum(fp, fq, &results)? { return Err(ChoiceError::NotZeroSum); }
    wants.need(fp)
}

/// READING THREE. The lower and the upper value of the tree for `p`, `p` and `q` being opposed (asked of the orders, not assumed). Pure choices at once. A chance node needs the scores to add to nought.
pub fn value_between<R: 'static>(t: &Tree<R>, wants: &Wants<R>, p: &Party, q: &Party) -> Result<Bounds, ChoiceError> {
    let order = two_sides(t, wants, p, q, t.has_chance())?;
    let lower = secure(t, p, &|r| order.score(r))?;
    let q_secure = secure(t, q, &|r| order.score(r)?.neg())?;
    Ok(Bounds { lower, upper: q_secure.neg()? })
}

// ------------------------------------------------------------------------------------------------------------------------------ reading four: the mixed choice

/// The value of a choice at once with each side picking a spread, and the spreads.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MixedTable {
    /// the value for `p`
    pub value: Num,
    /// the best spread of `p` over its own options at this table (its rows or its columns), and of `q` over its
    pub p_spread: Vec<Num>,
    pub q_spread: Vec<Num>,
    /// the cells as `p` reads them: `p`'s options down, `q`'s across, each the mixed value of the cell's subtree
    pub cells: Vec<Vec<Num>>,
}

/// The mixed value of a tree for `p`: turns by the best or the worst option, chance by expectation, a choice at once between the sides by the exact linear program over the mixed values of its cells.
fn mixed<R: 'static>(t: &Tree<R>, p: &Party, q: &Party, order: &Order<R>) -> Result<Num, ChoiceError> {
    match t {
        Tree::Result(r) => Ok(order.score(r)?),
        Tree::Turn { by, options } => {
            let vals = options.iter().map(|o| mixed(o, p, q, order)).collect::<Result<Vec<_>, _>>()?;
            Ok(pick(vals, p.contains(by)))
        }
        Tree::Chance(s) => {
            let mut acc = Num::whole(0);
            for (o, w) in s.outcomes() { acc = acc.add(&mixed(o, p, q, order)?.mul(w)?)?; }
            Ok(acc)
        }
        Tree::Table { .. } => Ok(mixed_here(t, p, q, order)?.value),
    }
}

fn mixed_here<R: 'static>(t: &Tree<R>, p: &Party, q: &Party, order: &Order<R>) -> Result<MixedTable, ChoiceError> {
    let Tree::Table { row_party, col_party, rows, cols, cells } = t else { return Err(ChoiceError::BadTable) };
    let grid = (0..*rows).map(|i| (0..*cols).map(|j| mixed(&cells[i * cols + j], p, q, order)).collect::<Result<Vec<_>, _>>()).collect::<Result<Vec<_>, _>>()?;
    let (rp, cp) = (row_party.is_part_of(p), col_party.is_part_of(p));
    match (rp, cp) {
        (true, false) => { let g = solve_matrix_game(&grid)?; Ok(MixedTable { value: g.value, p_spread: g.row, q_spread: g.col, cells: grid }) }
        (false, true) => {
            let by_col: Vec<Vec<Num>> = (0..*cols).map(|j| (0..*rows).map(|i| grid[i][j]).collect()).collect();
            let g = solve_matrix_game(&by_col)?;
            Ok(MixedTable { value: g.value, p_spread: g.row, q_spread: g.col, cells: by_col })
        }
        // both parties on one side: that side picks the best (or the worst) cell, a pure choice
        (true, true) => { let v = pick(grid.iter().flatten().copied(), true); Ok(MixedTable { value: v, p_spread: vec![], q_spread: vec![], cells: grid }) }
        (false, false) => { let v = pick(grid.iter().flatten().copied(), false); Ok(MixedTable { value: v, p_spread: vec![], q_spread: vec![], cells: grid }) }
    }
}

/// READING FOUR. The value of the tree for `p` when every choice at once is a choice of spreads. The scores must add to nought (a spread is a chance).
pub fn mixed_value<R: 'static>(t: &Tree<R>, wants: &Wants<R>, p: &Party, q: &Party) -> Result<Num, ChoiceError> {
    let order = two_sides(t, wants, p, q, true)?;
    mixed(t, p, q, order)
}

/// READING FOUR for the choice at once that is the tree's own root: the value and the best spread of each side.
pub fn mixed_table<R: 'static>(t: &Tree<R>, wants: &Wants<R>, p: &Party, q: &Party) -> Result<MixedTable, ChoiceError> {
    let order = two_sides(t, wants, p, q, true)?;
    mixed_here(t, p, q, order)
}

// ------------------------------------------------------------------------------------------------------------------------------ reading five: a choice against a belief

/// The expectation of each option and the best option.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Choice { /// the first option whose expectation is greatest
    pub best: usize, pub expectations: Vec<Num> }

/// READING FIVE. Each option's expectation under the belief (the payoff read off the option and the supposition), and the first best option.
pub fn choose_against<O, S: Clone>(options: &[O], belief: &Belief<S>, payoff: &dyn Fn(&O, &S) -> Num) -> Result<Choice, ChoiceError> {
    if options.is_empty() { return Err(ChoiceError::NoOption); }
    let mut expectations = Vec::with_capacity(options.len());
    for o in options { expectations.push(belief.expectation(&|s| payoff(o, s))?); }
    let mut best = 0;
    for (i, e) in expectations.iter().enumerate() { if *e > expectations[best] { best = i; } }
    Ok(Choice { best, expectations })
}

/// The expectation of a spread over the options (chances summing to one) against a fixed belief: the weighted mean of the options' expectations.
pub fn mix_expectation(mix: &Spread<usize>, expectations: &[Num]) -> Result<Num, NumError> { mix.expectation(&|i| expectations[*i]) }

#[cfg(test)]
mod tests {
    use super::*;
    use crate::maker::{random_tree, Maker, Shape};

    fn res(v: i64) -> Tree<i64> { Tree::Result(v) }
    fn n(v: i64) -> Num { Num::whole(v) }
    fn q(a: i64, b: i64) -> Num { Num::frac(a, b).unwrap() }
    fn order() -> Order<i64> { Order::by(|r: &i64| Num::whole(*r)) }
    fn two() -> Wants<i64> { Wants::new().with(Chooser::A, order()).with(Chooser::B, order().reversed()) }
    fn pa() -> Party { party(&[Chooser::A]) }
    fn pb() -> Party { party(&[Chooser::B]) }

    #[test]
    fn a_choice_without_an_option_and_a_ragged_table_are_refused() {
        assert_eq!(Tree::<i64>::turn(Chooser::A, vec![]).unwrap_err(), ChoiceError::NoOption);
        assert_eq!(Tree::table(2, 2, vec![res(1), res(2), res(3)]).unwrap_err(), ChoiceError::BadTable);
        assert_eq!(Tree::table(0, 2, Vec::<Tree<i64>>::new()).unwrap_err(), ChoiceError::BadTable);
        assert!(matches!(Tree::chance(vec![(res(1), q(1, 2))]).unwrap_err(), ChoiceError::Chance(_)));
        assert_eq!(Tree::table_between(pa(), pa(), 1, 1, vec![res(1)]).unwrap_err(), ChoiceError::BadParty);
        assert_eq!(Tree::table_between(Party::new(), pb(), 1, 1, vec![res(1)]).unwrap_err(), ChoiceError::BadParty);
    }

    /// Mirrors Ideas.Layer0.Choice.exactly_one: in a finite tree of turns without chance, A can force P exactly when B cannot force not-P.
    #[test]
    fn in_a_tree_of_turns_exactly_one_of_a_forces_p_and_b_forces_not_p() {
        let mut m = Maker::new(41);
        for _ in 0..3000 {
            let t = random_tree(&mut m, 5, Shape { tables: false, chance: false });
            let k = m.range(-9, 9);
            let a = can_force(&t, Chooser::A, &|r| *r >= k);
            let b = can_force(&t, Chooser::B, &|r| *r < k);
            assert!(a ^ b, "exactly one of the two: {a} {b} on {t:?}");
        }
    }

    /// Mirrors Ideas.Layer0.Choice.low_le_up and Ideas.Layer0.Choice.low_eq_up_of_tableFree.
    #[test]
    fn the_lower_value_is_not_above_the_upper_and_they_meet_without_a_choice_at_once() {
        let mut m = Maker::new(42);
        let (mut apart, mut tables) = (0, 0);
        for _ in 0..3000 {
            let with = m.chance();
            let ch = m.chance();
            let t = random_tree(&mut m, 5, Shape { tables: with, chance: ch });
            let b = value(&t, &order()).unwrap();
            assert!(b.lower <= b.upper, "{t:?}");
            if !with { assert_eq!(b.lower, b.upper, "no choice at once: one value"); } else { tables += 1; if b.lower < b.upper { apart += 1; } }
        }
        assert!(apart > 0 && tables > apart, "the maker makes tables, some with a saddle and some without ({apart} of {tables} apart)");
    }

    /// Worked examples with the numbers of the Lean module: Mirrors Ideas.Layer0.Choice.example_pennies_low, example_pennies_up, example_saddle_low, example_saddle_up, example_coin_low.
    #[test]
    fn matching_pennies_has_no_pure_saddle_and_the_mixed_choice_is_worth_nought() {
        // A wins (1) when the two picks match, loses (-1) when they differ
        let t = Tree::table(2, 2, vec![res(1), res(-1), res(-1), res(1)]).unwrap();
        let b = value(&t, &order()).unwrap();
        assert_eq!((b.lower, b.upper), (n(-1), n(1)));
        assert!(!can_force(&t, Chooser::A, &|r| *r > 0) && !can_force(&t, Chooser::B, &|r| *r < 0), "neither can force it at once, with pure choices");
        let m = mixed_table(&t, &two(), &pa(), &pb()).unwrap();
        assert_eq!((m.value, m.p_spread, m.q_spread), (n(0), vec![q(1, 2), q(1, 2)], vec![q(1, 2), q(1, 2)]));
        // a table with a saddle: row 0 is best against every column
        let s = Tree::table(2, 2, vec![res(3), res(2), res(1), res(0)]).unwrap();
        let b = value(&s, &order()).unwrap();
        assert_eq!((b.lower, b.upper), (n(2), n(2)));
        assert!(can_force(&s, Chooser::A, &|r| *r >= 2));
        let m = mixed_table(&s, &two(), &pa(), &pb()).unwrap();
        assert_eq!((m.value, m.p_spread), (n(2), vec![n(1), n(0)]), "a pure answer is given with chance one");
    }

    /// Mirrors Ideas.Layer0.Choice.value_ge_iff_canForce. The value of a tree of turns against the forcing reading: A makes sure of at least k exactly when A can force "at least k".
    #[test]
    fn the_value_of_a_tree_of_turns_agrees_with_the_forcing_reading() {
        let mut m = Maker::new(43);
        for _ in 0..2000 {
            let t = random_tree(&mut m, 5, Shape { tables: false, chance: false });
            let v = value(&t, &order()).unwrap().lower;
            for k in -9..=9 {
                assert_eq!(v >= n(k), can_force(&t, Chooser::A, &|r| *r >= k), "k {k} on {t:?}");
            }
        }
        let mut m = Maker::new(44);
        for _ in 0..2000 {
            let t = random_tree(&mut m, 4, Shape { tables: true, chance: false });
            let v = value(&t, &order()).unwrap().lower;
            for k in -9..=9 { assert_eq!(v >= n(k), can_force(&t, Chooser::A, &|r| *r >= k), "k {k} on {t:?}"); }
        }
    }

    /// Mirrors Ideas.Layer0.Choice.example_coin_low (the coin or the sure one is worth 2).
    #[test]
    fn a_tree_with_chance_is_weighed_by_the_chances_and_forcing_says_the_least_chance() {
        // A picks between a coin (half for the result 4, half for 0) and a sure 1; B has no say
        let coin = Tree::chance(vec![(res(4), q(1, 2)), (res(0), q(1, 2))]).unwrap();
        let t = Tree::turn(Chooser::A, vec![coin.clone(), res(1)]).unwrap();
        assert_eq!(value(&t, &order()).unwrap().lower, n(2));
        assert_eq!(force_chance(&coin, Chooser::A, &|r| *r == 4).unwrap(), q(1, 2));
        assert_eq!(force_chance(&t, Chooser::A, &|r| *r == 4).unwrap(), q(1, 2));
        assert_eq!(force_chance(&t, Chooser::A, &|r| *r >= 1).unwrap(), n(1), "the sure option forces it with chance one");
        // B moves after the coin: the coin's worse half is chosen for A by B in the second level
        let hard = Tree::chance(vec![(Tree::turn(Chooser::B, vec![res(3), res(-1)]).unwrap(), q(3, 4)), (res(5), q(1, 4))]).unwrap();
        assert_eq!(value(&hard, &order()).unwrap().lower, q(1, 2), "3/4 * -1 + 1/4 * 5, exactly");
    }

    #[test]
    fn force_with_chance_agrees_with_can_force_where_the_chance_is_one() {
        let mut m = Maker::new(45);
        for _ in 0..3000 {
            let t = random_tree(&mut m, 4, Shape { tables: true, chance: true });
            let k = m.range(-9, 9);
            let who = if m.chance() { Chooser::A } else { Chooser::B };
            let sure = can_force(&t, who, &|r| *r >= k);
            let c = force_chance(&t, who, &|r| *r >= k).unwrap();
            assert!(c >= n(0) && c <= n(1));
            assert_eq!(sure, c == n(1), "can force exactly when the least chance is one: {c} on {t:?}");
        }
    }

    #[test]
    fn an_order_and_its_reverse_want_opposite_things() {
        let o = order();
        assert!(o.prefers(&3, &2) && !o.prefers(&2, &3));
        let r = order().reversed();
        assert!(r.prefers(&2, &3) && r.compare(&5, &5) == Ordering::Equal);
        assert!(res(1).size() == 1 && Tree::turn(Chooser::A, vec![res(1), res(2)]).unwrap().size() == 3);
    }

    // ---- part A: choosers and wants

    #[test]
    fn opposed_and_allied_are_defined_by_the_orders_and_neither_is_assumed() {
        let rs = [1i64, 2, 3];
        let results: Vec<&i64> = rs.iter().collect();
        let (a, b, c, d) = (Chooser::A, Chooser::B, Chooser::new(2), Chooser::new(3));
        let w = Wants::new().with(a, order()).with(b, order().reversed()).with(c, Order::by(|r: &i64| Num::whole(r * 10))).with(d, Order::by(|r: &i64| Num::whole(if *r == 2 { 5 } else { *r })));
        assert!(w.opposed(a, b, &results).unwrap() && !w.allied(a, b, &results).unwrap());
        assert!(w.allied(a, c, &results).unwrap() && !w.opposed(a, c, &results).unwrap(), "the same order on another scale is allied");
        assert!(!w.allied(a, d, &results).unwrap() && !w.opposed(a, d, &results).unwrap(), "two that are neither");
        assert!(!w.zero_sum(a, c, &results).unwrap() && w.zero_sum(a, b, &results).unwrap());
        let ties = [7i64, 7];
        let t: Vec<&i64> = ties.iter().collect();
        assert!(w.opposed(a, b, &t).unwrap() && w.allied(a, b, &t).unwrap(), "when every result is a tie both hold");
        assert_eq!(w.allied(a, Chooser::new(9), &results).unwrap_err(), ChoiceError::NoOrder(Chooser::new(9)));
    }

    #[test]
    fn a_set_of_one_is_the_old_reading_and_a_set_of_allied_choosers_can_force_what_none_can_alone() {
        let mut m = Maker::new(51);
        for _ in 0..1500 {
            let shape = Shape { tables: m.chance(), chance: m.chance() };
            let t = random_tree(&mut m, 5, shape);
            let k = m.range(-9, 9);
            for who in [Chooser::A, Chooser::B] { assert_eq!(can_force(&t, who, &|r| *r >= k), can_force_set(&t, &party(&[who]), &|r| *r >= k)); }
        }
        // three choosers: C and A each move once, in turn; the property needs both to pick the 1
        let c = Chooser::new(2);
        let leaf = |v: i64| res(v);
        let t = Tree::turn(c, vec![Tree::turn(Chooser::A, vec![leaf(1), leaf(0)]).unwrap(), leaf(0)]).unwrap();
        let wins = |r: &i64| *r == 1;
        assert!(!can_force(&t, Chooser::A, &wins) && !can_force(&t, c, &wins));
        assert!(can_force_set(&t, &party(&[Chooser::A, c]), &wins), "allied choosers pick together");
        // a choice at once between {A, C} and {B}: the pair picks as one
        let tb = Tree::table_between(party(&[Chooser::A, c]), pb(), 2, 1, vec![leaf(1), leaf(0)]).unwrap();
        assert!(can_force_set(&tb, &party(&[Chooser::A, c]), &wins));
        assert!(!can_force_set(&tb, &party(&[Chooser::A]), &wins), "a party counts as the set's own only when all its members are in the set");
        assert_eq!(tb.choosers(), party(&[Chooser::A, Chooser::B, c]));
    }

    /// Mirrors Ideas.Layer0.Choice.coalition_exactly_one: with any number of choosers in turn, for any set of them exactly one of "the set can force P" and "the rest can force not-P" holds.
    #[test]
    fn the_set_and_the_rest_are_determined_with_any_number_of_choosers() {
        fn made(m: &mut Maker, depth: u32) -> Tree<i64> {
            if depth == 0 || m.below(4) == 0 { return Tree::Result(m.range(-9, 9)); }
            let n = 1 + m.below(3);
            Tree::turn(Chooser::new(m.below(4) as u32), (0..n).map(|_| made(m, depth - 1)).collect()).unwrap()
        }
        let mut m = Maker::new(53);
        let mut both_sides = 0;
        for _ in 0..3000 {
            let t = made(&mut m, 5);
            let set: Party = (0..4).filter(|_| m.chance()).map(Chooser::new).collect();
            let rest: Party = (0..4).map(Chooser::new).filter(|c| !set.contains(c)).collect();
            let k = m.range(-9, 9);
            let (a, b) = (can_force_set(&t, &set, &|r| *r >= k), can_force_set(&t, &rest, &|r| *r < k));
            assert!(a ^ b, "exactly one of the two: {a} {b} on {t:?}");
            if a { both_sides += 1; }
        }
        assert!(both_sides > 300 && both_sides < 2700, "the set wins sometimes and the rest sometimes ({both_sides})");
    }

    #[test]
    fn what_allied_choosers_can_make_sure_of_is_asked_of_their_orders() {
        let c = Chooser::new(2);
        let t = Tree::turn(c, vec![Tree::turn(Chooser::B, vec![res(4), res(2)]).unwrap(), Tree::turn(Chooser::A, vec![res(1), res(3)]).unwrap()]).unwrap();
        let w = Wants::new().with(Chooser::A, order()).with(c, Order::by(|r: &i64| Num::whole(r * 2))).with(Chooser::B, order().reversed());
        // A and C are allied (same order); B is against them: the pair gets max(min(4, 2), max(1, 3)) = 3
        assert_eq!(guarantee_allied(&t, &party(&[Chooser::A, c]), &w).unwrap(), n(3));
        // B alone, in B's order, the rest acting against it: C picks the worse for B, which is A's turn, where A picks 3: minus 3
        assert_eq!(guarantee_allied(&t, &pb(), &w).unwrap(), n(-3));
        assert_eq!(guarantee_allied(&t, &party(&[Chooser::A, Chooser::B]), &w).unwrap_err(), ChoiceError::NotAllied);
        let b = value_between(&t, &w, &party(&[Chooser::A, c]), &pb()).unwrap();
        assert_eq!((b.lower, b.upper), (n(3), n(3)));
    }

    #[test]
    fn a_reading_that_needs_opposed_sides_refuses_a_tree_whose_orders_do_not_give_them() {
        let t = Tree::turn(Chooser::A, vec![Tree::turn(Chooser::B, vec![res(1), res(2)]).unwrap(), res(0)]).unwrap();
        let allied = Wants::new().with(Chooser::A, order()).with(Chooser::B, order());
        assert_eq!(value_between(&t, &allied, &pa(), &pb()).unwrap_err(), ChoiceError::NotOpposed);
        let c = Chooser::new(2);
        assert_eq!(value_between(&Tree::turn(c, vec![res(1)]).unwrap(), &two(), &pa(), &pb()).unwrap_err(), ChoiceError::NotCovered(c));
        assert_eq!(value_between(&t, &Wants::new().with(Chooser::A, order()), &pa(), &pb()).unwrap_err(), ChoiceError::NoOrder(Chooser::B));
        // the scores of two opposed orders that do not add to nought: fine for pure choices, refused with a chance
        let scaled = Wants::new().with(Chooser::A, order()).with(Chooser::B, Order::by(|r: &i64| Num::whole(-3 * r)));
        assert!(value_between(&t, &scaled, &pa(), &pb()).is_ok());
        let coin = Tree::turn(Chooser::A, vec![Tree::chance(vec![(res(1), q(1, 2)), (res(0), q(1, 2))]).unwrap()]).unwrap();
        assert_eq!(value_between(&coin, &scaled, &pa(), &pb()).unwrap_err(), ChoiceError::NotZeroSum);
        assert_eq!(mixed_value(&t, &scaled, &pa(), &pb()).unwrap_err(), ChoiceError::NotZeroSum);
        // a party with members on both sides at a choice at once
        let split = Tree::table_between(party(&[Chooser::A, Chooser::B]), party(&[c]), 1, 1, vec![res(1)]).unwrap();
        let w3 = Wants::new().with(Chooser::A, order()).with(Chooser::B, order().reversed()).with(c, order().reversed());
        assert_eq!(value_between(&split, &w3, &pa(), &party(&[Chooser::B, c])).unwrap_err(), ChoiceError::SplitParty);
    }

    #[test]
    fn the_value_between_two_parties_of_one_is_the_two_chooser_value_on_made_trees() {
        let mut m = Maker::new(52);
        for _ in 0..1500 {
            let shape = Shape { tables: m.chance(), chance: m.chance() };
            let t = random_tree(&mut m, 5, shape);
            assert_eq!(value_between(&t, &two(), &pa(), &pb()).unwrap(), value(&t, &order()).unwrap());
        }
    }

    // ---- part B: the mixed choice

    /// A tree of one choice at once over made whole results, or in a few turns around it.
    fn made_mixed(m: &mut Maker) -> Tree<i64> {
        let (r, c) = (1 + m.below(3) as usize, 1 + m.below(3) as usize);
        let cells = (0..r * c).map(|_| random_tree(m, 2, Shape { tables: false, chance: true })).collect();
        let t = Tree::table(r, c, cells).unwrap();
        if m.chance() { Tree::turn(Chooser::A, vec![t, Tree::Result(m.range(-9, 9))]).unwrap() } else { t }
    }

    /// The value lies between the lower and the upper value of pure choices, and a tree without a choice at once has one value.
    #[test]
    fn the_mixed_value_lies_between_the_lower_and_the_upper_value() {
        let mut m = Maker::new(61);
        let mut between_strictly = 0;
        for _ in 0..1500 {
            let t = made_mixed(&mut m);
            let v = mixed_value(&t, &two(), &pa(), &pb()).unwrap();
            let b = value(&t, &order()).unwrap();
            assert!(b.lower <= v && v <= b.upper, "{t:?}: {} {v} {}", b.lower, b.upper);
            if b.lower < v && v < b.upper { between_strictly += 1; }
        }
        assert!(between_strictly > 100, "tables without a pure answer ({between_strictly})");
        let mut m = Maker::new(62);
        for _ in 0..1000 {
            let shape = Shape { tables: false, chance: m.chance() };
            let t = random_tree(&mut m, 5, shape);
            assert_eq!(mixed_value(&t, &two(), &pa(), &pb()).unwrap(), value(&t, &order()).unwrap().lower);
        }
    }

    /// The two sides' guarantees meet at a table: the spread of each side gives the value, as the program's certificate says.
    #[test]
    fn the_guarantees_of_the_two_sides_meet_and_a_pure_answer_has_chance_one() {
        let mut m = Maker::new(63);
        let mut pure = 0;
        for _ in 0..1500 {
            let (r, c) = (1 + m.below(3) as usize, 1 + m.below(3) as usize);
            let t = Tree::table(r, c, (0..r * c).map(|_| res(m.range(-6, 6))).collect()).unwrap();
            let mt = mixed_table(&t, &two(), &pa(), &pb()).unwrap();
            let g = lp::MatrixGame { value: mt.value, row: mt.p_spread.clone(), col: mt.q_spread.clone() };
            assert!(g.guarantees_meet(&mt.cells));
            let b = value(&t, &order()).unwrap();
            if b.lower == b.upper { pure += 1; assert_eq!(mt.value, b.lower); assert!(mt.p_spread.contains(&n(1)) || mt.p_spread.iter().any(|x| *x > n(0))); }
        }
        assert!(pure > 100 && pure < 1500, "{pure}");
    }

    #[test]
    fn a_mixed_choice_inside_a_tree_takes_the_value_of_each_cell() {
        // the cell (0, 0) is itself a choice of B between 3 and -1; matching pennies around it
        let inner = Tree::turn(Chooser::B, vec![res(3), res(-1)]).unwrap();
        let t = Tree::table(2, 2, vec![inner, res(-1), res(-1), res(1)]).unwrap();
        // the cells read as: [[-1, -1], [-1, 1]]: row 1 and column 0 give -1 at best against... the value is -1 (row 0 is -1 everywhere)
        assert_eq!(mixed_value(&t, &two(), &pa(), &pb()).unwrap(), n(-1));
        // a choice of A after a coin: a half of matching pennies and a half of a sure 2
        let pennies = Tree::table(2, 2, vec![res(1), res(-1), res(-1), res(1)]).unwrap();
        let coin = Tree::chance(vec![(pennies, q(1, 2)), (res(2), q(1, 2))]).unwrap();
        assert_eq!(mixed_value(&coin, &two(), &pa(), &pb()).unwrap(), n(1));
        // the sides the other way round: B picks the rows, A the columns, of matching pennies skewed to favour B's pick of the first
        let skew = Tree::table_between(pb(), pa(), 2, 2, vec![res(2), res(-1), res(-1), res(1)]).unwrap();
        let mt = mixed_table(&skew, &two(), &pa(), &pb()).unwrap();
        // from A's side (columns): A's rows are B's columns; A wants high; the matrix by columns [[2, -1], [-1, 1]] has value 1/5
        assert_eq!(mt.value, q(1, 5));
        assert_eq!(mt.p_spread, vec![q(2, 5), q(3, 5)]);
    }

    // ---- part C: the choice against a belief

    /// Mirrors Ideas.Layer0.Choice.no_mix_beats_best: against a belief that does not change, no spread of the options has a greater expectation than the best option.
    #[test]
    fn against_a_fixed_belief_a_single_option_is_best_and_no_mixing_gains() {
        let mut m = Maker::new(71);
        for _ in 0..1500 {
            let k = 1 + m.below(3) as usize;
            let belief = Belief::new(Spread::uniform((0..k as i64).collect()).unwrap());
            let opts = 1 + m.below(4) as usize;
            let table: Vec<Vec<i64>> = (0..opts).map(|_| (0..k).map(|_| m.range(-9, 9)).collect()).collect();
            let idx: Vec<usize> = (0..opts).collect();
            let c = choose_against(&idx, &belief, &|o, s| n(table[*o][*s as usize])).unwrap();
            let best = c.expectations[c.best];
            assert!(c.expectations.iter().all(|e| *e <= best));
            // a made spread over the options (quarters)
            let mut left = 4i64;
            let mut mix = vec![];
            for i in 0..opts { let w = if i + 1 == opts { left } else { m.range(0, left) }; left -= w; mix.push((i, q(w, 4))); }
            let mix = Spread::new(mix).unwrap();
            assert!(mix_expectation(&mix, &c.expectations).unwrap() <= best, "no mean is above its best part");
        }
        assert_eq!(choose_against::<i64, i64>(&[], &Belief::new(Spread::certain(0)), &|_, _| n(0)).unwrap_err(), ChoiceError::NoOption);
    }

    #[test]
    fn a_belief_changed_by_what_is_seen_changes_the_best_option() {
        // two suppositions about the other side: it plays left or right; the options are a pick of the left or the right door
        let belief = Belief::new(Spread::new(vec![("left", q(1, 2)), ("right", q(1, 2))]).unwrap());
        let payoff = |o: &&str, s: &&str| if o == s { n(1) } else { n(0) };
        let opts = ["left", "right"];
        assert_eq!(choose_against(&opts, &belief, &payoff).unwrap().best, 0, "a tie goes to the first");
        let seen = belief.after_seeing(&|s| if *s == "right" { n(1) } else { q(1, 4) }).unwrap();
        let c = choose_against(&opts, &seen, &payoff).unwrap();
        assert_eq!((c.best, c.expectations), (1, vec![q(1, 5), q(4, 5)]));
    }
}
