//! LAYER 0, MODULE 5: CHANCE (plan 79.30; SYSTEM_SPEC 11 "Layer 0 holds six things" item 5).
//!
//! INTERFACE (callers: choice's chance nodes; any layer above that reads an uncertain act).
//!   `Spread<T>`: a finite spread of outcomes with their chances. The chances are `f64`, each between nought and one, and sum to one within `SUM_TOLERANCE` (1e-9); the spread is refused when made if not.
//!     `Spread::new(Vec<(T, f64)>) -> Result<Spread<T>, ChanceError>`; `Spread::certain(T)`; `Spread::uniform(Vec<T>) -> Result<Spread<T>, ChanceError>`; `outcomes() -> &[(T, f64)]`; `total() -> f64`;
//!     `chance_of(&dyn Fn(&T) -> bool) -> f64`; `expectation(&dyn Fn(&T) -> Num) -> Result<Num, NumError>`; `map(&dyn Fn(&T) -> U) -> Spread<U>` (outcomes are kept, not merged);
//!     `given(&dyn Fn(&T) -> bool) -> Result<Spread<T>, ChanceError>` (the spread given a condition: the outcomes that hold, their chances divided by the chance of the condition; a condition of chance nought is `ImpossibleCondition`);
//!     `product(&Spread<A>, &Spread<B>) -> Spread<(A, B)>` (two that do not depend on each other: the chances multiply).
//!   `ChanceError::{BadChance, DoNotSumToOne, Empty, ImpossibleCondition}`.
//! LAWS: lean/ideas/Ideas/Layer0/Chance.lean (the chances of a product and of a spread given a condition sum to one; expectation is linear).
use crate::number::{Num, NumError};

/// How far from one the chances of a spread may sum (rounding of a real number, nothing more).
pub const SUM_TOLERANCE: f64 = 1e-9;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChanceError { BadChance, DoNotSumToOne, Empty, ImpossibleCondition }

impl std::fmt::Display for ChanceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self { ChanceError::BadChance => "a chance is not between nought and one", ChanceError::DoNotSumToOne => "the chances do not sum to one",
                                 ChanceError::Empty => "a spread has an outcome", ChanceError::ImpossibleCondition => "the condition has chance nought" })
    }
}

/// A finite spread of outcomes with their chances.
#[derive(Debug, Clone, PartialEq)]
pub struct Spread<T> { outcomes: Vec<(T, f64)> }

impl<T> Spread<T> {
    pub fn new(outcomes: Vec<(T, f64)>) -> Result<Spread<T>, ChanceError> {
        if outcomes.is_empty() { return Err(ChanceError::Empty); }
        if outcomes.iter().any(|(_, p)| !(0.0..=1.0).contains(p)) { return Err(ChanceError::BadChance); }       // a not-a-number is outside the range
        let sum: f64 = outcomes.iter().map(|(_, p)| p).sum();
        if (sum - 1.0).abs() > SUM_TOLERANCE { return Err(ChanceError::DoNotSumToOne); }
        Ok(Spread { outcomes })
    }
    pub fn certain(t: T) -> Spread<T> { Spread { outcomes: vec![(t, 1.0)] } }
    pub fn uniform(items: Vec<T>) -> Result<Spread<T>, ChanceError> {
        if items.is_empty() { return Err(ChanceError::Empty); }
        let p = 1.0 / items.len() as f64;
        Spread::new(items.into_iter().map(|t| (t, p)).collect())
    }
    pub fn outcomes(&self) -> &[(T, f64)] { &self.outcomes }
    /// The sum of the chances (one, within the tolerance).
    pub fn total(&self) -> f64 { self.outcomes.iter().map(|(_, p)| p).sum() }
    /// The chance that the property holds of the outcome.
    pub fn chance_of(&self, holds: &dyn Fn(&T) -> bool) -> f64 { self.outcomes.iter().filter(|(t, _)| holds(t)).map(|(_, p)| p).sum() }
    /// The expectation of a number read off each outcome.
    pub fn expectation(&self, value: &dyn Fn(&T) -> Num) -> Result<Num, NumError> {
        let mut acc = Num::whole(0);
        for (t, p) in &self.outcomes { acc = acc.add(&value(t).mul(&Num::real(*p)?)?)?; }
        Ok(acc)
    }
    /// The same chances over what each outcome is read as (outcomes are kept, not merged).
    pub fn map<U>(&self, f: &dyn Fn(&T) -> U) -> Spread<U> { Spread { outcomes: self.outcomes.iter().map(|(t, p)| (f(t), *p)).collect() } }
}

impl<T: Clone> Spread<T> {
    /// The spread given a condition: the outcomes where it holds, each chance divided by the chance of the condition.
    pub fn given(&self, holds: &dyn Fn(&T) -> bool) -> Result<Spread<T>, ChanceError> {
        let c = self.chance_of(holds);
        if c <= 0.0 { return Err(ChanceError::ImpossibleCondition); }
        let kept: Vec<(T, f64)> = self.outcomes.iter().filter(|(t, _)| holds(t)).map(|(t, p)| (t.clone(), (p / c).min(1.0))).collect();
        Spread::new(kept)
    }
}

/// Two spreads that do not depend on each other: every pair, with the product of the chances.
pub fn product<A: Clone, B: Clone>(a: &Spread<A>, b: &Spread<B>) -> Spread<(A, B)> {
    Spread { outcomes: a.outcomes.iter().flat_map(|(x, p)| b.outcomes.iter().map(move |(y, q)| ((x.clone(), y.clone()), p * q))).collect() }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::maker::Maker;

    /// A spread of quarters over 1..=4 outcomes made by the seeded maker (exact in `f64`).
    fn make(m: &mut Maker) -> Spread<i64> {
        let n = 1 + m.below(4) as usize;
        let mut left = 4u64;
        Spread::new((0..n).map(|i| { let q = if i + 1 == n { left } else { m.below(left + 1) }; left -= q; (m.range(-6, 6), q as f64 / 4.0) }).collect()).unwrap()
    }
    fn n(v: i64) -> Num { Num::whole(v) }

    #[test]
    fn a_spread_that_is_not_one_is_refused_when_made() {
        assert_eq!(Spread::new(vec![(1, 0.5), (2, 0.4)]).unwrap_err(), ChanceError::DoNotSumToOne);
        assert_eq!(Spread::new(vec![(1, 1.5), (2, -0.5)]).unwrap_err(), ChanceError::BadChance);
        assert_eq!(Spread::new(vec![(1, f64::NAN)]).unwrap_err(), ChanceError::BadChance);
        assert_eq!(Spread::<i64>::new(vec![]).unwrap_err(), ChanceError::Empty);
        assert!(Spread::new(vec![(1, 0.25), (2, 0.75)]).is_ok());
        assert!((Spread::uniform(vec![1, 2, 3]).unwrap().total() - 1.0).abs() < SUM_TOLERANCE);
    }

    /// Mirrors Ideas.Layer0.Chance.expectation_add, expectation_smul (linearity).
    #[test]
    fn expectation_is_linear() {
        let mut m = Maker::new(31);
        for _ in 0..1500 {
            let s = make(&mut m);
            let (a, b) = (m.range(-5, 5), m.range(-5, 5));
            let f = |x: &i64| n(*x * 3 - 1);
            let g = |x: &i64| n(x * x);
            let lhs = s.expectation(&|x| n(a).mul(&f(x)).unwrap().add(&n(b).mul(&g(x)).unwrap()).unwrap()).unwrap();
            let rhs = n(a).mul(&s.expectation(&f).unwrap()).unwrap().add(&n(b).mul(&s.expectation(&g).unwrap()).unwrap()).unwrap();
            assert_eq!(lhs, rhs);
        }
    }

    /// Mirrors Ideas.Layer0.Chance.product_sums_to_one and given_sums_to_one.
    #[test]
    fn a_product_and_a_spread_given_a_condition_sum_to_one() {
        let mut m = Maker::new(32);
        for _ in 0..1500 {
            let (a, b) = (make(&mut m), make(&mut m));
            assert!((product(&a, &b).total() - 1.0).abs() < SUM_TOLERANCE);
            let t = m.range(-6, 6);
            match a.given(&|x| *x >= t) {
                Ok(g) => { assert!((g.total() - 1.0).abs() < SUM_TOLERANCE); assert!(g.outcomes().iter().all(|(x, _)| *x >= t)); }
                Err(e) => assert_eq!(e, ChanceError::ImpossibleCondition),
            }
        }
    }

    #[test]
    fn independence_the_chance_of_a_pair_is_the_product_of_the_chances() {
        let (a, b) = (Spread::new(vec![(0, 0.5), (1, 0.5)]).unwrap(), Spread::new(vec![(7, 0.25), (8, 0.75)]).unwrap());
        let p = product(&a, &b);
        assert_eq!(p.chance_of(&|(x, y)| *x == 1 && *y == 8), 0.375);
        assert_eq!(p.chance_of(&|(x, _)| *x == 1), a.chance_of(&|x| *x == 1));
    }

    #[test]
    fn expectation_by_hand_and_the_condition_of_chance_nought() {
        let s = Spread::new(vec![(2, 0.5), (4, 0.25), (10, 0.25)]).unwrap();
        assert_eq!(s.expectation(&|x| n(*x)).unwrap(), Num::real(4.5).unwrap());                        // 1 + 1 + 2.5
        let g = s.given(&|x| *x >= 4).unwrap();
        assert_eq!(g.expectation(&|x| n(*x)).unwrap(), n(7));                                           // (4 + 10) / 2
        assert_eq!(s.given(&|x| *x > 100).unwrap_err(), ChanceError::ImpossibleCondition);
        assert_eq!(s.map(&|x| x * 2).outcomes().len(), 3);
    }
}
