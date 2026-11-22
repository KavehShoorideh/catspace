//! LAYER 0, MODULE 5: CHANCE (plan 79.30; plan 79.31 parts C and D: exact chances and the belief; SYSTEM_SPEC 11 "Layer 0 holds six things" item 5, the block on belief and choosing by chance, "Layer 0 is made right").
//!
//! INTERFACE (callers: choice's chance nodes, its mixed choices and its choice against a belief; any layer above that reads an uncertain act or says what it believes).
//!   `Spread<T>`: a finite spread of outcomes with their chances. A chance is a `Num`, between nought and one. When every chance is EXACT (whole or fraction) the chances must sum to EXACTLY one; the tolerance `SUM_TOLERANCE` (1e-9) is used only when a real
//!     number enters a chance. Refused when made if not.
//!     `Spread::new(Vec<(T, Num)>) -> Result<Spread<T>, ChanceError>`; `Spread::from_reals(Vec<(T, f64)>)` (the chances are reals, so the tolerance applies); `Spread::certain(T)`; `Spread::uniform(Vec<T>)` (each chance one over how many, exact);
//!     `outcomes() -> &[(T, Num)]`; `total() -> Result<Num, NumError>`; `chance_of(&dyn Fn(&T) -> bool) -> Result<Num, NumError>`; `expectation(&dyn Fn(&T) -> Num) -> Result<Num, NumError>` (exact on exact chances and values);
//!     `map(&dyn Fn(&T) -> U) -> Spread<U>` (outcomes are kept, not merged);
//!     `given(&dyn Fn(&T) -> bool) -> Result<Spread<T>, ChanceError>` (the spread given a condition: the outcomes that hold, their chances divided by the chance of the condition; a condition of chance nought is `ImpossibleCondition`);
//!     `product(&Spread<A>, &Spread<B>) -> Result<Spread<(A, B)>, ChanceError>` (two that do not depend on each other: the chances multiply).
//!   `Belief<S>`: a belief is a spread over SUPPOSITIONS (the type `S` is whatever the caller supposes; layer 0 says nothing of what is believed of anyone).
//!     `Belief::new(Spread<S>)`, `spread() -> &Spread<S>`, `expectation(&dyn Fn(&S) -> Num) -> Result<Num, NumError>`;
//!     `after_seeing(&dyn Fn(&S) -> Num) -> Result<Belief<S>, BeliefError>`: what is seen changes the belief. The function gives, for each supposition, how likely it made what was seen (a number from nought to one). Each chance is weighed by that and the
//!     spread is made whole again; REFUSED (`Impossible`) when no supposition with a chance could have made what was seen.
//!   `ChanceError::{BadChance, DoNotSumToOne, Empty, ImpossibleCondition, Number(NumError)}`; `BeliefError::{BadLikelihood, Impossible, Number(NumError)}`.
//! LAWS: lean/ideas/Ideas/Layer0/Chance.lean (the chances of a product and of a spread given a condition sum to one; expectation is linear; a belief changed by what is seen sums to one; the change by two sightings is the change by both together).
use crate::number::{Num, NumError};

/// How far from one the chances of a spread may sum when a real number enters a chance (rounding of a real number, nothing more).
pub const SUM_TOLERANCE: f64 = 1e-9;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChanceError { BadChance, DoNotSumToOne, Empty, ImpossibleCondition, Number(NumError) }

impl std::fmt::Display for ChanceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ChanceError::BadChance => f.write_str("a chance is not between nought and one"),
            ChanceError::DoNotSumToOne => f.write_str("the chances do not sum to one"),
            ChanceError::Empty => f.write_str("a spread has an outcome"),
            ChanceError::ImpossibleCondition => f.write_str("the condition has chance nought"),
            ChanceError::Number(e) => write!(f, "{e}"),
        }
    }
}

impl From<NumError> for ChanceError { fn from(e: NumError) -> ChanceError { ChanceError::Number(e) } }

/// A finite spread of outcomes with their chances.
#[derive(Debug, Clone, PartialEq)]
pub struct Spread<T> { outcomes: Vec<(T, Num)> }

fn one() -> Num { Num::whole(1) }
fn nought() -> Num { Num::whole(0) }

impl<T> Spread<T> {
    pub fn new(outcomes: Vec<(T, Num)>) -> Result<Spread<T>, ChanceError> {
        if outcomes.is_empty() { return Err(ChanceError::Empty); }
        if outcomes.iter().any(|(_, p)| *p < nought() || *p > one()) { return Err(ChanceError::BadChance); }
        let mut sum = nought();
        for (_, p) in &outcomes { sum = sum.add(p)?; }
        let exact = outcomes.iter().all(|(_, p)| p.is_exact());
        let ok = if exact { sum == one() } else { (sum.as_f64() - 1.0).abs() <= SUM_TOLERANCE };
        if !ok { return Err(ChanceError::DoNotSumToOne); }
        Ok(Spread { outcomes })
    }
    /// Chances given as reals: each is a real `Num`, so the sum is read within the tolerance.
    pub fn from_reals(outcomes: Vec<(T, f64)>) -> Result<Spread<T>, ChanceError> {
        let mut out = Vec::with_capacity(outcomes.len());
        for (t, p) in outcomes { out.push((t, Num::real(p).map_err(|_| ChanceError::BadChance)?)); }
        Spread::new(out)
    }
    pub fn certain(t: T) -> Spread<T> { Spread { outcomes: vec![(t, one())] } }
    pub fn uniform(items: Vec<T>) -> Result<Spread<T>, ChanceError> {
        if items.is_empty() { return Err(ChanceError::Empty); }
        let p = Num::frac(1, items.len() as i64)?;
        Spread::new(items.into_iter().map(|t| (t, p)).collect())
    }
    pub fn outcomes(&self) -> &[(T, Num)] { &self.outcomes }
    /// The sum of the chances (one: exactly when the chances are exact).
    pub fn total(&self) -> Result<Num, NumError> { let mut s = nought(); for (_, p) in &self.outcomes { s = s.add(p)?; } Ok(s) }
    /// The chance that the property holds of the outcome.
    pub fn chance_of(&self, holds: &dyn Fn(&T) -> bool) -> Result<Num, NumError> {
        let mut s = nought();
        for (t, p) in &self.outcomes { if holds(t) { s = s.add(p)?; } }
        Ok(s)
    }
    /// The expectation of a number read off each outcome.
    pub fn expectation(&self, value: &dyn Fn(&T) -> Num) -> Result<Num, NumError> {
        let mut acc = nought();
        for (t, p) in &self.outcomes { acc = acc.add(&value(t).mul(p)?)?; }
        Ok(acc)
    }
    /// The same chances over what each outcome is read as (outcomes are kept, not merged).
    pub fn map<U>(&self, f: &dyn Fn(&T) -> U) -> Spread<U> { Spread { outcomes: self.outcomes.iter().map(|(t, p)| (f(t), *p)).collect() } }
}

impl<T: Clone> Spread<T> {
    /// The spread given a condition: the outcomes where it holds, each chance divided by the chance of the condition.
    pub fn given(&self, holds: &dyn Fn(&T) -> bool) -> Result<Spread<T>, ChanceError> {
        let c = self.chance_of(holds)?;
        if c <= nought() { return Err(ChanceError::ImpossibleCondition); }
        let mut kept = vec![];
        for (t, p) in &self.outcomes { if holds(t) { kept.push((t.clone(), p.div(&c)?)); } }
        Spread::new(kept)
    }
}

/// Two spreads that do not depend on each other: every pair, with the product of the chances.
pub fn product<A: Clone, B: Clone>(a: &Spread<A>, b: &Spread<B>) -> Result<Spread<(A, B)>, ChanceError> {
    let mut out = vec![];
    for (x, p) in &a.outcomes { for (y, q) in &b.outcomes { out.push(((x.clone(), y.clone()), p.mul(q)?)); } }
    Spread::new(out)
}

// ------------------------------------------------------------------------------------------------------------------------------ the belief (plan 79.31 part C)

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BeliefError { BadLikelihood, Impossible, Number(NumError) }

impl std::fmt::Display for BeliefError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BeliefError::BadLikelihood => f.write_str("a likelihood is not between nought and one"),
            BeliefError::Impossible => f.write_str("no supposition could have made what was seen"),
            BeliefError::Number(e) => write!(f, "{e}"),
        }
    }
}

impl From<NumError> for BeliefError { fn from(e: NumError) -> BeliefError { BeliefError::Number(e) } }

/// A belief: a spread over suppositions. Layer 0 gives the form only; what is supposed, and of whom, is not said here.
#[derive(Debug, Clone, PartialEq)]
pub struct Belief<S> { spread: Spread<S> }

impl<S> Belief<S> {
    pub fn new(spread: Spread<S>) -> Belief<S> { Belief { spread } }
    pub fn spread(&self) -> &Spread<S> { &self.spread }
    /// The expectation of a number read off each supposition.
    pub fn expectation(&self, value: &dyn Fn(&S) -> Num) -> Result<Num, NumError> { self.spread.expectation(value) }
}

impl<S: Clone> Belief<S> {
    /// The belief after what was seen: each chance is weighed by how likely its supposition made what was seen, and the result is made whole again. `Impossible` when the weights are all nought.
    pub fn after_seeing(&self, likelihood: &dyn Fn(&S) -> Num) -> Result<Belief<S>, BeliefError> {
        let mut weighed = vec![];
        let mut total = nought();
        for (s, p) in self.spread.outcomes() {
            let l = likelihood(s);
            if l < nought() || l > one() { return Err(BeliefError::BadLikelihood); }
            let w = p.mul(&l)?;
            total = total.add(&w)?;
            weighed.push((s.clone(), w));
        }
        if total <= nought() { return Err(BeliefError::Impossible); }
        let mut out = vec![];
        for (s, w) in weighed { out.push((s, w.div(&total)?)); }
        // a weight over the total is at most one; the sum is one up to the tolerance when a real took part, and exactly one when none did
        Spread::new(out).map(Belief::new).map_err(|e| match e { ChanceError::Number(n) => BeliefError::Number(n), _ => BeliefError::BadLikelihood })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::maker::Maker;

    /// A spread of quarters over 1..=4 outcomes made by the seeded maker (exact).
    fn make(m: &mut Maker) -> Spread<i64> {
        let n = 1 + m.below(4) as usize;
        let mut left = 4u64;
        Spread::new((0..n).map(|i| { let q = if i + 1 == n { left } else { m.below(left + 1) }; left -= q; (m.range(-6, 6), Num::frac(q as i64, 4).unwrap()) }).collect()).unwrap()
    }
    fn n(v: i64) -> Num { Num::whole(v) }
    fn q(a: i64, b: i64) -> Num { Num::frac(a, b).unwrap() }

    #[test]
    fn a_spread_that_is_not_one_is_refused_when_made() {
        assert_eq!(Spread::new(vec![(1, q(1, 2)), (2, q(2, 5))]).unwrap_err(), ChanceError::DoNotSumToOne);
        assert_eq!(Spread::new(vec![(1, q(3, 2)), (2, q(-1, 2))]).unwrap_err(), ChanceError::BadChance);
        assert_eq!(Spread::<i64>::new(vec![]).unwrap_err(), ChanceError::Empty);
        assert!(Spread::new(vec![(1, q(1, 4)), (2, q(3, 4))]).is_ok());
        assert_eq!(Spread::from_reals(vec![(1, f64::NAN)]).unwrap_err(), ChanceError::BadChance);
        assert_eq!(Spread::from_reals(vec![(1, 0.5), (2, 0.4)]).unwrap_err(), ChanceError::DoNotSumToOne);
        assert!(Spread::from_reals(vec![(1, 0.1), (2, 0.2), (3, 0.7)]).is_ok(), "reals sum within the tolerance only");
        assert_eq!(Spread::uniform(vec![1, 2, 3]).unwrap().total().unwrap(), n(1), "a third three times is exactly one");
        // exact inputs ask equality: a sum that is one in reals within the tolerance, but is not one as fractions, is refused
        assert_eq!(Spread::new(vec![(1, q(1, 3)), (2, q(1, 3)), (3, Num::frac(333_333_333, 1_000_000_000).unwrap())]).unwrap_err(), ChanceError::DoNotSumToOne);
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

    /// Mirrors Ideas.Layer0.Chance.product_sums_to_one and given_sums_to_one: exactly one now.
    #[test]
    fn a_product_and_a_spread_given_a_condition_sum_to_exactly_one() {
        let mut m = Maker::new(32);
        for _ in 0..1500 {
            let (a, b) = (make(&mut m), make(&mut m));
            assert_eq!(product(&a, &b).unwrap().total().unwrap(), n(1));
            let t = m.range(-6, 6);
            match a.given(&|x| *x >= t) {
                Ok(g) => { assert_eq!(g.total().unwrap(), n(1)); assert!(g.outcomes().iter().all(|(x, _)| *x >= t)); }
                Err(e) => assert_eq!(e, ChanceError::ImpossibleCondition),
            }
        }
    }

    #[test]
    fn independence_the_chance_of_a_pair_is_the_product_of_the_chances() {
        let (a, b) = (Spread::new(vec![(0, q(1, 2)), (1, q(1, 2))]).unwrap(), Spread::new(vec![(7, q(1, 4)), (8, q(3, 4))]).unwrap());
        let p = product(&a, &b).unwrap();
        assert_eq!(p.chance_of(&|(x, y)| *x == 1 && *y == 8).unwrap(), q(3, 8));
        assert_eq!(p.chance_of(&|(x, _)| *x == 1).unwrap(), a.chance_of(&|x| *x == 1).unwrap());
    }

    /// Mirrors Ideas.Layer0.Chance.example_expectation (the spread 2: a half, 4: a quarter, 10: a quarter is worth 9/2).
    #[test]
    fn expectation_by_hand_and_the_condition_of_chance_nought() {
        let s = Spread::new(vec![(2, q(1, 2)), (4, q(1, 4)), (10, q(1, 4))]).unwrap();
        assert_eq!(s.expectation(&|x| n(*x)).unwrap(), q(9, 2));                                       // 1 + 1 + 2.5, exactly
        let g = s.given(&|x| *x >= 4).unwrap();
        assert_eq!(g.expectation(&|x| n(*x)).unwrap(), n(7));                                           // (4 + 10) / 2
        assert_eq!(s.given(&|x| *x > 100).unwrap_err(), ChanceError::ImpossibleCondition);
        assert_eq!(s.map(&|x| x * 2).outcomes().len(), 3);
        let thirds = Spread::uniform(vec![1, 2, 3]).unwrap();
        assert_eq!(thirds.expectation(&|x| n(*x)).unwrap(), n(2));
        assert_eq!(thirds.given(&|x| *x != 2).unwrap().outcomes()[0].1, q(1, 2));
    }

    /// Mirrors Ideas.Layer0.Chance.belief_sums_to_one, example_belief_first, example_belief_second: a belief changed by what is seen is a spread again, exactly.
    #[test]
    fn a_belief_changed_by_what_is_seen_is_weighed_by_the_likelihoods_and_sums_to_one() {
        let prior = Belief::new(Spread::new(vec![("a", q(1, 2)), ("b", q(1, 3)), ("c", q(1, 6))]).unwrap());
        let seen = prior.after_seeing(&|s| match *s { "a" => q(1, 10), "b" => q(1, 2), _ => n(1) }).unwrap();
        // weights: 1/20, 1/6, 1/6 ; total 1/20 + 1/3 = 23/60
        assert_eq!(seen.spread().total().unwrap(), n(1));
        assert_eq!(seen.spread().outcomes()[0].1, q(3, 23));
        assert_eq!(seen.spread().outcomes()[1].1, q(10, 23));
        assert_eq!(seen.spread().outcomes()[2].1, q(10, 23));
        let mut m = Maker::new(81);
        for _ in 0..1500 {
            let b = Belief::new(make(&mut m));
            let lik = |s: &i64| q((s.rem_euclid(5)) + 0, 4).min(n(1));
            match b.after_seeing(&lik) { Ok(c) => assert_eq!(c.spread().total().unwrap(), n(1)), Err(e) => assert_eq!(e, BeliefError::Impossible) }
        }
    }

    /// Mirrors Ideas.Layer0.Chance.belief_two_sightings: the change by two sightings is the change by both together (the likelihoods multiply), in either order.
    #[test]
    fn two_sightings_change_a_belief_as_both_together_and_in_either_order() {
        let mut m = Maker::new(82);
        let mut met = 0;
        for _ in 0..1500 {
            let b = Belief::new(make(&mut m));
            let (k1, k2) = (m.range(1, 4), m.range(1, 4));
            let l1 = move |s: &i64| q((s.rem_euclid(4)) + 1, 5 * k1);
            let l2 = move |s: &i64| q((s.rem_euclid(3)) + 1, 4 * k2);
            let both = |s: &i64| l1(s).mul(&l2(s)).unwrap();
            let one_then_two = b.after_seeing(&l1).and_then(|x| x.after_seeing(&l2));
            let two_then_one = b.after_seeing(&l2).and_then(|x| x.after_seeing(&l1));
            let together = b.after_seeing(&both);
            let norm = |r: Result<Belief<i64>, BeliefError>| r.map(|x| x.spread().outcomes().iter().map(|(s, p)| (*s, *p)).collect::<Vec<_>>());
            assert_eq!(norm(one_then_two.clone()), norm(together.clone()));
            assert_eq!(norm(two_then_one), norm(together));
            if one_then_two.is_ok() { met += 1; }
        }
        assert!(met > 500);
    }

    #[test]
    fn nothing_supposed_could_have_made_what_was_seen_is_refused() {
        let b = Belief::new(Spread::new(vec![(1, q(1, 2)), (2, q(1, 2))]).unwrap());
        assert_eq!(b.after_seeing(&|_| n(0)).unwrap_err(), BeliefError::Impossible);
        assert_eq!(b.after_seeing(&|_| n(2)).unwrap_err(), BeliefError::BadLikelihood);
        // a supposition with chance nought is never made likely by what is seen
        let c = Belief::new(Spread::new(vec![(1, n(1)), (2, n(0))]).unwrap());
        assert_eq!(c.after_seeing(&|s| if *s == 2 { n(1) } else { n(0) }).unwrap_err(), BeliefError::Impossible);
        assert_eq!(b.expectation(&|s| n(*s)).unwrap(), q(3, 2));
    }
}
