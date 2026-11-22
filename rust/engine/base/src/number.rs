//! LAYER 0, MODULE 3: NUMBER AND ORDER (plan 79.30; SYSTEM_SPEC 11 "Layer 0 holds six things" item 3).
//!
//! INTERFACE (callers: sequence differences, geometry, chance's expectation, choice's values; any layer above that adds, compares or scales).
//!   `Num`: ONE type for whole and real numbers with ONE total order. A real that is a whole number within the range of `i64` is stored as the whole number, so two equal numbers are one representation and `==`, `Ord` and `Hash` agree.
//!     `Num::whole(i64) -> Num`; `Num::real(f64) -> Result<Num, NumError>` (a value that is not a number, or is not finite, is REFUSED here, when made); `as_f64() -> f64`; `as_whole() -> Option<i64>`; `is_whole() -> bool`.
//!     `Ord`/`PartialOrd`/`Eq`/`Hash`: a whole and a real are compared exactly (no rounding of the whole).
//!   `NumError::{NotANumber, Overflow, DivideByZero}`.
//!   `add`, `sub`, `mul`, `div(&Num, &Num) -> Result<Num, NumError>`, `neg() -> Result<Num, NumError>`, `abs()`, `sqrt() -> Result<Num, NumError>` (of a negative: NotANumber).
//!     Whole with whole stays whole and exact, and says `Overflow` when it leaves `i64`; a real result that is not finite says `Overflow`; a whole with a real goes through `f64` (the whole is read as the nearest `f64`: said, not hidden).
//!     `div`: whole by whole that divides exactly stays whole, else real.
//!   `min` and `max` are `Ord`'s; `Display`.
//! LAWS: lean/ideas/Ideas/Layer0/Number.lean (the order is total, antisymmetric and transitive; addition is monotone in the order).
use std::cmp::Ordering;
use std::hash::{Hash, Hasher};

/// Why a number could not be made or an operation could not be done.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NumError { NotANumber, Overflow, DivideByZero }

impl std::fmt::Display for NumError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self { NumError::NotANumber => "not a number", NumError::Overflow => "overflow", NumError::DivideByZero => "division by nought" })
    }
}

#[derive(Debug, Clone, Copy)]
enum Repr { Whole(i64), Real(f64) }

/// A whole or a real number. The real is always finite and never a whole number within the range of `i64` (it is stored as the whole number then).
#[derive(Debug, Clone, Copy)]
pub struct Num(Repr);

/// 2^63 as a real: the least real above the range of `i64`.
const TWO63: f64 = 9_223_372_036_854_775_808.0;

impl Num {
    pub fn whole(v: i64) -> Num { Num(Repr::Whole(v)) }

    /// A real number; a value that is not a number or is not finite is refused. A whole value within the range of `i64` becomes the whole number.
    pub fn real(v: f64) -> Result<Num, NumError> {
        if v.is_nan() { return Err(NumError::NotANumber); }
        if v.is_infinite() { return Err(NumError::Overflow); }
        if v.fract() == 0.0 && (-TWO63..TWO63).contains(&v) { return Ok(Num::whole(v as i64)); }
        Ok(Num(Repr::Real(v)))
    }

    pub fn as_f64(&self) -> f64 { match self.0 { Repr::Whole(w) => w as f64, Repr::Real(r) => r } }
    pub fn as_whole(&self) -> Option<i64> { match self.0 { Repr::Whole(w) => Some(w), Repr::Real(_) => None } }
    pub fn is_whole(&self) -> bool { matches!(self.0, Repr::Whole(_)) }

    pub fn add(&self, o: &Num) -> Result<Num, NumError> {
        match (self.0, o.0) { (Repr::Whole(a), Repr::Whole(b)) => a.checked_add(b).map(Num::whole).ok_or(NumError::Overflow), _ => Num::real(self.as_f64() + o.as_f64()) }
    }
    pub fn sub(&self, o: &Num) -> Result<Num, NumError> {
        match (self.0, o.0) { (Repr::Whole(a), Repr::Whole(b)) => a.checked_sub(b).map(Num::whole).ok_or(NumError::Overflow), _ => Num::real(self.as_f64() - o.as_f64()) }
    }
    pub fn mul(&self, o: &Num) -> Result<Num, NumError> {
        match (self.0, o.0) { (Repr::Whole(a), Repr::Whole(b)) => a.checked_mul(b).map(Num::whole).ok_or(NumError::Overflow), _ => Num::real(self.as_f64() * o.as_f64()) }
    }
    pub fn div(&self, o: &Num) -> Result<Num, NumError> {
        if *o == Num::whole(0) { return Err(NumError::DivideByZero); }
        match (self.0, o.0) {
            (Repr::Whole(a), Repr::Whole(b)) => match a.checked_rem(b) {
                Some(0) => a.checked_div(b).map(Num::whole).ok_or(NumError::Overflow),
                Some(_) => Num::real(a as f64 / b as f64),
                None => Err(NumError::Overflow),
            },
            _ => Num::real(self.as_f64() / o.as_f64()),
        }
    }
    pub fn neg(&self) -> Result<Num, NumError> {
        match self.0 { Repr::Whole(a) => a.checked_neg().map(Num::whole).ok_or(NumError::Overflow), Repr::Real(r) => Num::real(-r) }
    }
    pub fn abs(&self) -> Result<Num, NumError> { if *self < Num::whole(0) { self.neg() } else { Ok(*self) } }
    /// The square root of a number that is not negative; of a negative, `NotANumber`.
    pub fn sqrt(&self) -> Result<Num, NumError> { if *self < Num::whole(0) { Err(NumError::NotANumber) } else { Num::real(self.as_f64().sqrt()) } }
}

/// A whole number against a real, exactly (the whole is never rounded to compare).
fn whole_against_real(w: i64, r: f64) -> Ordering {
    if r >= TWO63 { return Ordering::Less; }
    if r < -TWO63 { return Ordering::Greater; }
    let f = r.floor();
    match w.cmp(&(f as i64)) { Ordering::Equal => if r > f { Ordering::Less } else { Ordering::Equal }, o => o }
}

impl Ord for Num {
    fn cmp(&self, o: &Num) -> Ordering {
        match (self.0, o.0) {
            (Repr::Whole(a), Repr::Whole(b)) => a.cmp(&b),
            (Repr::Real(a), Repr::Real(b)) => a.partial_cmp(&b).expect("a real is never not-a-number"),
            (Repr::Whole(a), Repr::Real(b)) => whole_against_real(a, b),
            (Repr::Real(a), Repr::Whole(b)) => whole_against_real(b, a).reverse(),
        }
    }
}
impl PartialOrd for Num { fn partial_cmp(&self, o: &Num) -> Option<Ordering> { Some(self.cmp(o)) } }
impl PartialEq for Num { fn eq(&self, o: &Num) -> bool { self.cmp(o) == Ordering::Equal } }
impl Eq for Num {}
impl Hash for Num {
    fn hash<H: Hasher>(&self, h: &mut H) { match self.0 { Repr::Whole(w) => w.hash(h), Repr::Real(r) => r.to_bits().hash(h) } }
}
impl std::fmt::Display for Num {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { match self.0 { Repr::Whole(w) => write!(f, "{w}"), Repr::Real(r) => write!(f, "{r}") } }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::maker::Maker;

    fn n(v: i64) -> Num { Num::whole(v) }
    fn r(v: f64) -> Num { Num::real(v).unwrap() }

    #[test]
    fn a_value_that_is_not_a_number_is_refused_when_made() {
        assert_eq!(Num::real(f64::NAN), Err(NumError::NotANumber));
        assert_eq!(Num::real(f64::INFINITY), Err(NumError::Overflow));
        assert_eq!(Num::real(f64::NEG_INFINITY), Err(NumError::Overflow));
        assert!(Num::real(1.5).is_ok());
    }

    #[test]
    fn a_whole_real_is_the_whole_number_so_equal_numbers_are_one() {
        assert_eq!(r(2.0), n(2));
        assert!(r(2.0).is_whole());
        assert_eq!(r(-0.0), n(0));
        assert!(!r(0.5).is_whole());
        assert_eq!(r(1e30).as_whole(), None, "a real beyond the whole range stays a real");
    }

    /// Mirrors Ideas.Layer0.Number.order_total, order_antisymm, order_trans on cases made by the seeded maker (whole and real, near each other and far).
    #[test]
    fn the_order_is_total_antisymmetric_and_transitive() {
        let mut m = Maker::new(11);
        let make = |m: &mut Maker| -> Num {
            match m.below(3) { 0 => n(m.range(-20, 20)), 1 => r(m.range(-40, 40) as f64 / 2.0), _ => r((m.fraction() - 0.5) * 1e19) }
        };
        for _ in 0..4000 {
            let (a, b, c) = (make(&mut m), make(&mut m), make(&mut m));
            assert!(a <= b || b <= a, "total");
            if a <= b && b <= a { assert_eq!(a, b, "antisymmetric"); }
            if a <= b && b <= c { assert!(a <= c, "transitive"); }
            assert_eq!(a.cmp(&b), b.cmp(&a).reverse());
        }
    }

    #[test]
    fn a_whole_and_a_real_are_compared_exactly() {
        let big = i64::MAX;
        assert!(n(big) < r(9.3e18), "above the whole range");
        assert!(n(i64::MIN) > r(-9.4e18));
        assert!(n(3) < r(3.5) && n(4) > r(3.5) && n(-4) < r(-3.5) && n(-3) > r(-3.5));
        // a whole that a real cannot hold stays itself: 2^53 + 1 is above 2^53
        assert!(n((1i64 << 53) + 1) > r((1u64 << 53) as f64));
        assert!(n(big) < r(TWO63 * 2.0));
    }

    /// Mirrors Ideas.Layer0.Number.add_monotone.
    #[test]
    fn addition_is_monotone_in_the_order() {
        let mut m = Maker::new(12);
        for _ in 0..2000 {
            let (a, b, c) = (n(m.range(-1000, 1000)), r(m.range(-2000, 2000) as f64 / 8.0), n(m.range(-50, 50)));
            if a <= b { assert!(a.add(&c).unwrap() <= b.add(&c).unwrap()); }
        }
    }

    #[test]
    fn arithmetic_says_when_it_overflows() {
        assert_eq!(n(i64::MAX).add(&n(1)), Err(NumError::Overflow));
        assert_eq!(n(i64::MIN).sub(&n(1)), Err(NumError::Overflow));
        assert_eq!(n(i64::MAX).mul(&n(2)), Err(NumError::Overflow));
        assert_eq!(n(i64::MIN).neg(), Err(NumError::Overflow));
        assert_eq!(n(i64::MIN).div(&n(-1)), Err(NumError::Overflow));
        assert_eq!(r(1e308).mul(&r(10.0)), Err(NumError::Overflow));
        assert_eq!(n(1).div(&n(0)), Err(NumError::DivideByZero));
        assert_eq!(n(-4).sqrt(), Err(NumError::NotANumber));
    }

    #[test]
    fn whole_stays_whole_and_exact_and_a_real_joins_when_it_must() {
        assert_eq!(n(6).div(&n(3)).unwrap().as_whole(), Some(2));
        assert_eq!(n(1).div(&n(4)).unwrap(), r(0.25));
        assert_eq!(n(2).add(&r(0.5)).unwrap(), r(2.5));
        assert_eq!(r(0.5).add(&r(0.5)).unwrap().as_whole(), Some(1));
        assert_eq!(n(9).sqrt().unwrap().as_whole(), Some(3));
        assert_eq!(n(-5).abs().unwrap(), n(5));
        assert_eq!(n(3).min(r(2.5)), r(2.5));
        assert_eq!(n(3).max(r(2.5)), n(3));
    }
}
