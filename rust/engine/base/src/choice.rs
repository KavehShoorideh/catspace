//! LAYER 0, MODULE 6: OPPOSED CHOICE (plan 79.30; SYSTEM_SPEC 11 "Layer 0 holds six things" item 6). The logic of two choosers who want opposite things: "there is a choice such that for every counter-choice". THE MEANING that the search of layer 4 must agree with;
//! the search is not moved or changed here, and a comparison with it is not part of this plan.
//!
//! INTERFACE (callers: tests of layer 0 now; layer 4 later, as the meaning to be held against).
//!   `Chooser::{A, B}`, `Chooser::other()`.
//!   `Tree<R>`: a finite tree whose nodes are a result `Tree::Result(R)`; a choice IN TURN by one chooser `Tree::Turn { by, options }` (at least one option); a choice AT ONCE by two choosers `Tree::Table { rows, cols, cells }` (A picks a row, B a column, cells in
//!     row-major order, each a tree); a choice BY CHANCE `Tree::Chance(Spread<Tree<R>>)`. Made by `Tree::turn(by, options) -> Result<Tree, ChoiceError>`, `Tree::table(rows, cols, cells) -> Result<Tree, ChoiceError>`, `Tree::chance(Vec<(Tree, f64)>) -> Result<Tree, ChoiceError>`.
//!   `Order<R>`: an order of preference over results for a chooser, by a score (more is better): `Order::by(impl Fn(&R) -> Num)`, `prefers(&R, &R) -> bool`, `compare(&R, &R) -> Ordering`, `reversed() -> Order<R>`. Two choosers who want opposite things: B's order is A's reversed.
//!   READING ONE, CAN FORCE: `can_force(&Tree, who: Chooser, holds: &dyn Fn(&R) -> bool) -> bool` (`who` can make the property hold of the result against every counter-choice; in turn: some option of its own, every option of the other; at once, PURE choices only:
//!     a row of its own such that every column of the other; by chance: every outcome of positive chance); `force_chance(&Tree, who, holds) -> Result<Num, NumError>` (the least chance with which `who` can make it hold: chance nodes are weighed by their chances).
//!   READING TWO, VALUE: `value(&Tree, a_order: &Order<R>) -> Result<Bounds, NumError>`, `Bounds { lower, upper }`, the values for A when the two want opposite things. Without a choice at once lower equals upper. For a choice at once only PURE choices are read:
//!     the LOWER value is what A can make sure of (best row against worst column), the UPPER is what B can hold A down to (best column for B against A's best row). MIXED CHOICES (a chance over rows or columns) ARE NOT BUILT; a table with no pure saddle has lower below upper and says only that.
//!   LAWS: lean/ideas/Ideas/Layer0/Choice.lean (in a finite tree of turns without chance exactly one of "A can force P" and "B can force not-P" holds; the lower value is not above the upper).
use crate::chance::{ChanceError, Spread};
use crate::number::{Num, NumError};
use std::cmp::Ordering;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Chooser { A, B }

impl Chooser {
    pub fn other(self) -> Chooser { match self { Chooser::A => Chooser::B, Chooser::B => Chooser::A } }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChoiceError { NoOption, BadTable, Chance(ChanceError) }

impl std::fmt::Display for ChoiceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self { ChoiceError::NoOption => f.write_str("a choice has at least one option"), ChoiceError::BadTable => f.write_str("a table has rows times columns cells, and at least one of each"), ChoiceError::Chance(e) => write!(f, "{e}") }
    }
}

/// A finite tree of choices.
#[derive(Debug, Clone, PartialEq)]
pub enum Tree<R> {
    Result(R),
    /// A choice in turn by one chooser.
    Turn { by: Chooser, options: Vec<Tree<R>> },
    /// A choice at once: A picks a row, B a column; the cells in row-major order.
    Table { rows: usize, cols: usize, cells: Vec<Tree<R>> },
    /// A choice by chance.
    Chance(Spread<Tree<R>>),
}

impl<R> Tree<R> {
    pub fn turn(by: Chooser, options: Vec<Tree<R>>) -> Result<Tree<R>, ChoiceError> {
        if options.is_empty() { Err(ChoiceError::NoOption) } else { Ok(Tree::Turn { by, options }) }
    }
    pub fn table(rows: usize, cols: usize, cells: Vec<Tree<R>>) -> Result<Tree<R>, ChoiceError> {
        if rows == 0 || cols == 0 || cells.len() != rows * cols { Err(ChoiceError::BadTable) } else { Ok(Tree::Table { rows, cols, cells }) }
    }
    pub fn chance(outcomes: Vec<(Tree<R>, f64)>) -> Result<Tree<R>, ChoiceError> { Spread::new(outcomes).map(Tree::Chance).map_err(ChoiceError::Chance) }

    /// How many nodes the tree has.
    pub fn size(&self) -> usize {
        1 + match self {
            Tree::Result(_) => 0,
            Tree::Turn { options, .. } => options.iter().map(Tree::size).sum(),
            Tree::Table { cells, .. } => cells.iter().map(Tree::size).sum(),
            Tree::Chance(s) => s.outcomes().iter().map(|(t, _)| t.size()).sum(),
        }
    }
}

/// An order of preference over results, by a score: more is better.
pub struct Order<R> { score: Box<dyn Fn(&R) -> Num>, flipped: bool }

impl<R: 'static> Order<R> {
    pub fn by(score: impl Fn(&R) -> Num + 'static) -> Order<R> { Order { score: Box::new(score), flipped: false } }
    /// The score of a result in this order (the reverse order scores minus the score); overflow is said.
    pub fn score(&self, r: &R) -> Result<Num, NumError> { let s = (self.score)(r); if self.flipped { s.neg() } else { Ok(s) } }
    pub fn compare(&self, a: &R, b: &R) -> Ordering { let o = (self.score)(a).cmp(&(self.score)(b)); if self.flipped { o.reverse() } else { o } }
    /// The first is better than the second.
    pub fn prefers(&self, a: &R, b: &R) -> bool { self.compare(a, b) == Ordering::Greater }
    /// The order of one who wants the opposite.
    pub fn reversed(self) -> Order<R> { Order { score: self.score, flipped: !self.flipped } }
}

/// READING ONE. Whether `who` can make the property hold of the result against every counter-choice. At once, pure choices only.
pub fn can_force<R>(t: &Tree<R>, who: Chooser, holds: &dyn Fn(&R) -> bool) -> bool {
    match t {
        Tree::Result(r) => holds(r),
        Tree::Turn { by, options } => if *by == who { options.iter().any(|o| can_force(o, who, holds)) } else { options.iter().all(|o| can_force(o, who, holds)) },
        Tree::Table { rows, cols, cells } => match who {
            Chooser::A => (0..*rows).any(|i| (0..*cols).all(|j| can_force(&cells[i * cols + j], who, holds))),
            Chooser::B => (0..*cols).any(|j| (0..*rows).all(|i| can_force(&cells[i * cols + j], who, holds))),
        },
        Tree::Chance(s) => s.outcomes().iter().all(|(o, p)| *p <= 0.0 || can_force(o, who, holds)),
    }
}

/// What `who` can make sure of in a score it wants more of, the other wanting less: at its own turn the best option, at the other's the worst, by chance the expectation, at once (pure choices) the best of its own
/// against the worst of the other's.
fn security<R>(t: &Tree<R>, who: Chooser, score: &dyn Fn(&R) -> Result<Num, NumError>) -> Result<Num, NumError> {
    let pick = |vals: Vec<Num>, best: bool| -> Num { vals.into_iter().reduce(|a, b| if best { a.max(b) } else { a.min(b) }).expect("a choice has an option") };
    match t {
        Tree::Result(r) => score(r),
        Tree::Turn { by, options } => {
            let vals = options.iter().map(|o| security(o, who, score)).collect::<Result<Vec<_>, _>>()?;
            Ok(pick(vals, *by == who))
        }
        Tree::Table { rows, cols, cells } => {
            let cell = |i: usize, j: usize| security(&cells[i * cols + j], who, score);
            let mut per_own = vec![];
            match who {
                Chooser::A => for i in 0..*rows { per_own.push(pick((0..*cols).map(|j| cell(i, j)).collect::<Result<Vec<_>, _>>()?, false)); },
                Chooser::B => for j in 0..*cols { per_own.push(pick((0..*rows).map(|i| cell(i, j)).collect::<Result<Vec<_>, _>>()?, false)); },
            }
            Ok(pick(per_own, true))
        }
        Tree::Chance(s) => {
            let mut acc = Num::whole(0);
            for (o, p) in s.outcomes() { acc = acc.add(&security(o, who, score)?.mul(&Num::real(*p)?)?)?; }
            Ok(acc)
        }
    }
}

/// The least chance with which `who` can make the property hold (chance nodes weighed by their chances).
pub fn force_chance<R>(t: &Tree<R>, who: Chooser, holds: &dyn Fn(&R) -> bool) -> Result<Num, NumError> {
    security(t, who, &|r| Ok(Num::whole(holds(r) as i64)))
}

/// The two values of a tree for A.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Bounds { pub lower: Num, pub upper: Num }

/// READING TWO. The values for A when A and B want opposite things (B's order is A's reversed). The lower value is what A can make sure of; the upper is what B can hold A down to. Pure choices only.
pub fn value<R: 'static>(t: &Tree<R>, a_order: &Order<R>) -> Result<Bounds, NumError> {
    let lower = security(t, Chooser::A, &|r| a_order.score(r))?;
    // what B can make sure of, in the reversed score, is minus what B can hold A down to
    let b_secure = security(t, Chooser::B, &|r| a_order.score(r)?.neg())?;
    Ok(Bounds { lower, upper: b_secure.neg()? })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::maker::{random_tree, Maker, Shape};

    fn res(v: i64) -> Tree<i64> { Tree::Result(v) }
    fn n(v: i64) -> Num { Num::whole(v) }
    fn order() -> Order<i64> { Order::by(|r: &i64| Num::whole(*r)) }

    #[test]
    fn a_choice_without_an_option_and_a_ragged_table_are_refused() {
        assert_eq!(Tree::<i64>::turn(Chooser::A, vec![]).unwrap_err(), ChoiceError::NoOption);
        assert_eq!(Tree::table(2, 2, vec![res(1), res(2), res(3)]).unwrap_err(), ChoiceError::BadTable);
        assert_eq!(Tree::table(0, 2, Vec::<Tree<i64>>::new()).unwrap_err(), ChoiceError::BadTable);
        assert!(matches!(Tree::chance(vec![(res(1), 0.5)]).unwrap_err(), ChoiceError::Chance(_)));
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

    /// Mirrors Ideas.Layer0.Choice.lower_le_upper.
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

    #[test]
    fn matching_pennies_has_no_pure_saddle() {
        // A wins (1) when the two picks match, loses (-1) when they differ
        let t = Tree::table(2, 2, vec![res(1), res(-1), res(-1), res(1)]).unwrap();
        let b = value(&t, &order()).unwrap();
        assert_eq!((b.lower, b.upper), (n(-1), n(1)));
        assert!(!can_force(&t, Chooser::A, &|r| *r > 0) && !can_force(&t, Chooser::B, &|r| *r < 0), "neither can force it at once, with pure choices");
        // a table with a saddle: row 0 is best against every column
        let s = Tree::table(2, 2, vec![res(3), res(2), res(1), res(0)]).unwrap();
        let b = value(&s, &order()).unwrap();
        assert_eq!((b.lower, b.upper), (n(2), n(2)));
        assert!(can_force(&s, Chooser::A, &|r| *r >= 2));
    }

    /// The value of a tree of turns against the forcing reading: A makes sure of at least k exactly when A can force "at least k".
    #[test]
    fn the_value_of_a_tree_of_turns_agrees_with_the_forcing_reading() {
        let mut m = Maker::new(43);
        for _ in 0..2000 {
            let t = random_tree(&mut m, 5, Shape { tables: false, chance: false });
            let v = value(&t, &order()).unwrap().lower;
            for k in -9..=9 {
                assert_eq!(v >= n(k), can_force(&t, Chooser::A, &|r| *r >= k), "k {k} on {t:?}");
            }
            // and the same with the table cells read as chances of one: with tables the lower value is what A can force with pure choices
        }
        let mut m = Maker::new(44);
        for _ in 0..2000 {
            let t = random_tree(&mut m, 4, Shape { tables: true, chance: false });
            let v = value(&t, &order()).unwrap().lower;
            for k in -9..=9 { assert_eq!(v >= n(k), can_force(&t, Chooser::A, &|r| *r >= k), "k {k} on {t:?}"); }
        }
    }

    #[test]
    fn a_tree_with_chance_is_weighed_by_the_chances_and_forcing_says_the_least_chance() {
        // A picks between a coin (half for the result 4, half for 0) and a sure 1; B has no say
        let coin = Tree::chance(vec![(res(4), 0.5), (res(0), 0.5)]).unwrap();
        let t = Tree::turn(Chooser::A, vec![coin.clone(), res(1)]).unwrap();
        assert_eq!(value(&t, &order()).unwrap().lower, n(2));
        assert_eq!(force_chance(&coin, Chooser::A, &|r| *r == 4).unwrap(), Num::real(0.5).unwrap());
        assert_eq!(force_chance(&t, Chooser::A, &|r| *r == 4).unwrap(), Num::real(0.5).unwrap());
        assert_eq!(force_chance(&t, Chooser::A, &|r| *r >= 1).unwrap(), n(1), "the sure option forces it with chance one");
        // B moves after the coin: the coin's worse half is chosen for A by B in the second level
        let hard = Tree::chance(vec![(Tree::turn(Chooser::B, vec![res(3), res(-1)]).unwrap(), 0.75), (res(5), 0.25)]).unwrap();
        assert_eq!(value(&hard, &order()).unwrap().lower, Num::real(0.5).unwrap(), "0.75 * -1 + 0.25 * 5");
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
        assert_eq!(Chooser::A.other(), Chooser::B);
        assert!(res(1).size() == 1 && Tree::turn(Chooser::A, vec![res(1), res(2)]).unwrap().size() == 3);
    }
}
