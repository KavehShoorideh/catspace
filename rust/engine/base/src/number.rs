//! LAYER 0, MODULE 3: NUMBER AND ORDER (plan 79.30; plan 79.31 part D: exact fractions; SYSTEM_SPEC 11 "Layer 0 holds six things" item 3, "Layer 0 is made right").
//!
//! INTERFACE (callers: sequence differences, geometry, chance's spreads and expectations, choice's values and the exact linear program; any layer above that adds, compares or scales).
//!   `Num`: ONE type for whole numbers, exact fractions and reals, with ONE total order. A whole number and a fraction are EXACT; a real is used only when it is asked for (`Num::real`) or when an operation takes a real in.
//!     `Num::whole(i64) -> Num`; `Num::frac(numer: i64, denom: i64) -> Result<Num, NumError>` (a fraction in lowest terms with a positive denominator; a denominator of one is the whole number; a denominator of nought is `DivideByZero`);
//!     `Num::real(f64) -> Result<Num, NumError>` (a value that is not a number, or is not finite, is REFUSED here, when made; a real that is a whole number within the range of `i64` is stored as that whole number);
//!     `Num::exact(f64) -> Result<Num, NumError>` (the real as the exact whole or fraction it is, or `Overflow` when that does not fit);
//!     `as_f64() -> f64`; `as_whole() -> Option<i64>`; `as_ratio() -> Option<(i64, i64)>` (numerator and denominator of an exact number; `None` for a real); `is_whole()`, `is_exact()`.
//!   `Ord`/`PartialOrd`/`Eq`/`Hash`: the order is the order of the values the numbers are, EXACTLY: a whole, a fraction and a real are compared with no rounding (a real is the exact dyadic number it holds), so two equal numbers are
//!     equal whatever their form, and `Hash` agrees with `==`.
//!   `NumError::{NotANumber, Overflow, DivideByZero}`.
//!   `add`, `sub`, `mul`, `div(&Num, &Num) -> Result<Num, NumError>`, `neg()`, `abs()`, `sqrt()` (of a negative: NotANumber).
//!     Whole and fraction with whole and fraction is EXACT (the result is a whole or a fraction in lowest terms) and says `Overflow` when the numerator or the denominator leaves `i64`: it never falls back to a real, silently.
//!     When a real takes part the result is a real (through `f64`), and that is said here, not hidden. `div`: exact by exact is exact.
//!   `min` and `max` are `Ord`'s; `Display` (`3`, `1/4`, `0.3`).
//! LAWS: lean/ideas/Ideas/Layer0/Number.lean (the order is total, antisymmetric and transitive; addition is monotone in the order; a real that is a whole number is that whole number; fractions add exactly).
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
enum Repr { Whole(i64), Frac(i64, i64), Real(f64) }

/// A whole number, an exact fraction or a real. The fraction is in lowest terms with a denominator of at least two; the real is finite and is never a whole number within the range of `i64`.
#[derive(Debug, Clone, Copy)]
pub struct Num(Repr);

/// 2^63 as a real: the least real above the range of `i64`.
const TWO63: f64 = 9_223_372_036_854_775_808.0;

fn gcd(mut a: i128, mut b: i128) -> i128 { a = a.abs(); b = b.abs(); while b != 0 { let t = a % b; a = b; b = t; } a }

/// A ratio of two wide whole numbers made into a number, in lowest terms; `Overflow` when it does not fit.
fn from_ratio(n: i128, d: i128) -> Result<Num, NumError> {
    if d == 0 { return Err(NumError::DivideByZero); }
    let g = gcd(n, d).max(1);
    let (mut n, mut d) = (n / g, d / g);
    if d < 0 { n = -n; d = -d; }
    let (n, d) = (i64::try_from(n).map_err(|_| NumError::Overflow)?, i64::try_from(d).map_err(|_| NumError::Overflow)?);
    Ok(if d == 1 { Num(Repr::Whole(n)) } else { Num(Repr::Frac(n, d)) })
}

/// The value of a real as a ratio of whole numbers, when it is one that fits (the real is a dyadic number).
fn real_as_ratio(r: f64) -> Option<(i64, i64)> {
    let bits = r.to_bits();
    let exp = ((bits >> 52) & 0x7ff) as i32;
    let frac = bits & ((1u64 << 52) - 1);
    let (mut m, mut e) = if exp == 0 { (frac, -1074) } else { (frac | (1u64 << 52), exp - 1075) };
    if m == 0 { return Some((0, 1)); }
    while m & 1 == 0 { m >>= 1; e += 1; }
    let neg = r < 0.0;
    let m = i64::try_from(m).ok()?;
    let m = if neg { -m } else { m };
    if e >= 0 { if e <= 62 { m.checked_mul(1i64 << e).map(|v| (v, 1)) } else { None } } else if -e <= 62 { Some((m, 1i64 << (-e))) } else { None }
}

/// A real (finite, not zero) against the ratio `n / d` (`d` positive), exactly.
fn real_against_ratio(r: f64, n: i64, d: i64) -> Ordering {
    let rneg = r < 0.0;
    let nneg = n < 0;
    if n == 0 { return if rneg { Ordering::Less } else if r > 0.0 { Ordering::Greater } else { Ordering::Equal }; }
    if rneg != nneg { return if rneg { Ordering::Less } else { Ordering::Greater }; }
    // same sign: compare magnitudes, then turn the answer round for a negative pair
    let bits = r.abs().to_bits();
    let exp = ((bits >> 52) & 0x7ff) as i32;
    let frac = bits & ((1u64 << 52) - 1);
    let (m, e) = if exp == 0 { (frac, -1074) } else { (frac | (1u64 << 52), exp - 1075) };
    let (lhs, rhs) = (m as u128 * d as u128, n.unsigned_abs() as u128);                    // |r| = m 2^e against |n| / d: m d 2^e against |n|
    let mag = if e >= 0 {
        if lhs == 0 { Ordering::Less } else if (lhs.leading_zeros() as i32) <= e { Ordering::Greater } else { (lhs << e).cmp(&rhs) }
    } else {
        let k = -e;
        if (rhs.leading_zeros() as i32) <= k { Ordering::Less } else { lhs.cmp(&(rhs << k)) }
    };
    if rneg { mag.reverse() } else { mag }
}

impl Num {
    pub fn whole(v: i64) -> Num { Num(Repr::Whole(v)) }

    /// An exact fraction in lowest terms; a denominator of one is the whole number, of nought is refused.
    pub fn frac(numer: i64, denom: i64) -> Result<Num, NumError> { from_ratio(numer as i128, denom as i128) }

    /// A real number; a value that is not a number or is not finite is refused. A whole value within the range of `i64` becomes the whole number.
    pub fn real(v: f64) -> Result<Num, NumError> {
        if v.is_nan() { return Err(NumError::NotANumber); }
        if v.is_infinite() { return Err(NumError::Overflow); }
        if v.fract() == 0.0 && (-TWO63..TWO63).contains(&v) { return Ok(Num::whole(v as i64)); }
        Ok(Num(Repr::Real(v)))
    }

    /// The exact value of a real as a whole number or a fraction (every real is such a number); `Overflow` when the numerator or the denominator does not fit.
    pub fn exact(v: f64) -> Result<Num, NumError> {
        let r = Num::real(v)?;
        match r.0 { Repr::Real(x) => { let (n, d) = real_as_ratio(x).ok_or(NumError::Overflow)?; Num::frac(n, d) } _ => Ok(r) }
    }

    pub fn as_f64(&self) -> f64 { match self.0 { Repr::Whole(w) => w as f64, Repr::Frac(n, d) => n as f64 / d as f64, Repr::Real(r) => r } }
    pub fn as_whole(&self) -> Option<i64> { match self.0 { Repr::Whole(w) => Some(w), _ => None } }
    /// Numerator and denominator (positive, in lowest terms) of a whole number or a fraction; `None` for a real.
    pub fn as_ratio(&self) -> Option<(i64, i64)> { match self.0 { Repr::Whole(w) => Some((w, 1)), Repr::Frac(n, d) => Some((n, d)), Repr::Real(_) => None } }
    pub fn is_whole(&self) -> bool { matches!(self.0, Repr::Whole(_)) }
    /// A whole number or a fraction (not a real).
    pub fn is_exact(&self) -> bool { !matches!(self.0, Repr::Real(_)) }

    fn exact_pair(&self, o: &Num) -> Option<((i128, i128), (i128, i128))> {
        let (a, b) = (self.as_ratio()?, o.as_ratio()?);
        Some(((a.0 as i128, a.1 as i128), (b.0 as i128, b.1 as i128)))
    }

    pub fn add(&self, o: &Num) -> Result<Num, NumError> {
        match self.exact_pair(o) { Some(((a, b), (c, d))) => from_ratio(a * d + c * b, b * d), None => Num::real(self.as_f64() + o.as_f64()) }
    }
    pub fn sub(&self, o: &Num) -> Result<Num, NumError> {
        match self.exact_pair(o) { Some(((a, b), (c, d))) => from_ratio(a * d - c * b, b * d), None => Num::real(self.as_f64() - o.as_f64()) }
    }
    pub fn mul(&self, o: &Num) -> Result<Num, NumError> {
        match self.exact_pair(o) { Some(((a, b), (c, d))) => from_ratio(a * c, b * d), None => Num::real(self.as_f64() * o.as_f64()) }
    }
    pub fn div(&self, o: &Num) -> Result<Num, NumError> {
        if *o == Num::whole(0) { return Err(NumError::DivideByZero); }
        match self.exact_pair(o) { Some(((a, b), (c, d))) => from_ratio(a * d, b * c), None => Num::real(self.as_f64() / o.as_f64()) }
    }
    pub fn neg(&self) -> Result<Num, NumError> {
        match self.0 {
            Repr::Whole(a) => a.checked_neg().map(Num::whole).ok_or(NumError::Overflow),
            Repr::Frac(n, d) => n.checked_neg().map(|m| Num(Repr::Frac(m, d))).ok_or(NumError::Overflow),
            Repr::Real(r) => Num::real(-r),
        }
    }
    pub fn abs(&self) -> Result<Num, NumError> { if *self < Num::whole(0) { self.neg() } else { Ok(*self) } }
    /// The root of a number that is not negative; of a negative, `NotANumber`. The root is a real unless it is a whole number or a fraction both of whose terms are perfect squares.
    pub fn sqrt(&self) -> Result<Num, NumError> {
        if *self < Num::whole(0) { return Err(NumError::NotANumber); }
        if let Some((n, d)) = self.as_ratio() {
            let (rn, rd) = ((n as f64).sqrt().round() as i64, (d as f64).sqrt().round() as i64);
            if rn.checked_mul(rn) == Some(n) && rd.checked_mul(rd) == Some(d) { return Num::frac(rn, rd); }
        }
        Num::real(self.as_f64().sqrt())
    }
}

/// The order of two numbers: exact on every pair of forms.
fn compare(a: &Num, b: &Num) -> Ordering {
    match (a.0, b.0) {
        (Repr::Real(x), Repr::Real(y)) => x.partial_cmp(&y).expect("a real is never not-a-number"),
        (Repr::Real(x), _) => { let (n, d) = b.as_ratio().expect("exact"); real_against_ratio(x, n, d) }
        (_, Repr::Real(y)) => { let (n, d) = a.as_ratio().expect("exact"); real_against_ratio(y, n, d).reverse() }
        _ => { let ((x, y), (z, w)) = a.exact_pair(b).expect("exact"); (x * w).cmp(&(z * y)) }
    }
}

impl Ord for Num { fn cmp(&self, o: &Num) -> Ordering { compare(self, o) } }
impl PartialOrd for Num { fn partial_cmp(&self, o: &Num) -> Option<Ordering> { Some(self.cmp(o)) } }
impl PartialEq for Num { fn eq(&self, o: &Num) -> bool { self.cmp(o) == Ordering::Equal } }
impl Eq for Num {}
impl Hash for Num {
    /// A real that is exactly a whole number or a fraction that fits hashes as that number, so equal numbers hash alike.
    fn hash<H: Hasher>(&self, h: &mut H) {
        match self.0 {
            Repr::Whole(w) => { 0u8.hash(h); w.hash(h) }
            Repr::Frac(n, d) => { 1u8.hash(h); n.hash(h); d.hash(h) }
            Repr::Real(r) => match real_as_ratio(r) {
                Some((n, 1)) => { 0u8.hash(h); n.hash(h) }
                Some((n, d)) => { 1u8.hash(h); n.hash(h); d.hash(h) }
                None => { 2u8.hash(h); r.to_bits().hash(h) }
            },
        }
    }
}
impl std::fmt::Display for Num {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.0 { Repr::Whole(w) => write!(f, "{w}"), Repr::Frac(n, d) => write!(f, "{n}/{d}"), Repr::Real(r) => write!(f, "{r}") }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::maker::Maker;

    fn n(v: i64) -> Num { Num::whole(v) }
    fn r(v: f64) -> Num { Num::real(v).unwrap() }
    fn q(a: i64, b: i64) -> Num { Num::frac(a, b).unwrap() }

    #[test]
    fn a_value_that_is_not_a_number_is_refused_when_made() {
        assert_eq!(Num::real(f64::NAN), Err(NumError::NotANumber));
        assert_eq!(Num::real(f64::INFINITY), Err(NumError::Overflow));
        assert_eq!(Num::real(f64::NEG_INFINITY), Err(NumError::Overflow));
        assert!(Num::real(1.5).is_ok());
        assert_eq!(Num::frac(1, 0), Err(NumError::DivideByZero));
    }

    /// Mirrors Ideas.Layer0.Number.real_whole_same.
    #[test]
    fn a_whole_real_is_the_whole_number_so_equal_numbers_are_one() {
        assert_eq!(r(2.0), n(2));
        assert!(r(2.0).is_whole());
        assert_eq!(r(-0.0), n(0));
        assert!(!r(0.5).is_whole());
        assert_eq!(r(1e30).as_whole(), None, "a real beyond the whole range stays a real");
    }

    #[test]
    fn a_fraction_is_in_lowest_terms_and_a_denominator_of_one_is_the_whole_number() {
        assert_eq!(q(2, 4).as_ratio(), Some((1, 2)));
        assert_eq!(q(6, 3).as_whole(), Some(2));
        assert_eq!(q(3, -6).as_ratio(), Some((-1, 2)));
        assert_eq!(q(0, 5).as_whole(), Some(0));
        assert!(q(1, 3).is_exact() && !q(1, 3).is_whole() && !r(0.5).is_exact());
        assert_eq!(Num::exact(0.75).unwrap(), q(3, 4));
        assert_eq!(Num::exact(0.75).unwrap().as_ratio(), Some((3, 4)));
        assert_eq!(Num::exact(0.1).unwrap().as_ratio().unwrap().1, 1i64 << 55, "0.1 as a real is a dyadic number, and exactly that is asked for");
        assert_eq!(Num::exact(1e-30), Err(NumError::Overflow));
    }

    /// Mirrors Ideas.Layer0.Number.order_total, order_antisymm, order_trans on cases made by the seeded maker (whole, fraction and real, near each other and far).
    #[test]
    fn the_order_is_total_antisymmetric_and_transitive() {
        let mut m = Maker::new(11);
        let make = |m: &mut Maker| -> Num {
            match m.below(4) { 0 => n(m.range(-20, 20)), 1 => r(m.range(-40, 40) as f64 / 2.0), 2 => q(m.range(-30, 30), m.range(1, 7)), _ => r((m.fraction() - 0.5) * 1e19) }
        };
        for _ in 0..6000 {
            let (a, b, c) = (make(&mut m), make(&mut m), make(&mut m));
            assert!(a <= b || b <= a, "total");
            if a <= b && b <= a { assert_eq!(a, b, "antisymmetric"); }
            if a <= b && b <= c { assert!(a <= c, "transitive"); }
            assert_eq!(a.cmp(&b), b.cmp(&a).reverse());
        }
    }

    #[test]
    fn a_whole_a_fraction_and_a_real_are_compared_exactly() {
        let big = i64::MAX;
        assert!(n(big) < r(9.3e18), "above the whole range");
        assert!(n(i64::MIN) > r(-9.4e18));
        assert!(n(3) < r(3.5) && n(4) > r(3.5) && n(-4) < r(-3.5) && n(-3) > r(-3.5));
        // a whole that a real cannot hold stays itself: 2^53 + 1 is above 2^53
        assert!(n((1i64 << 53) + 1) > r((1u64 << 53) as f64));
        assert!(n(big) < r(TWO63 * 2.0));
        // a fraction that a real cannot hold exactly: a third is below the real nearest to it or above it, never equal
        assert!(q(1, 3) != r(1.0 / 3.0) && (q(1, 3) < r(1.0 / 3.0) || q(1, 3) > r(1.0 / 3.0)));
        assert_eq!(q(1, 2), r(0.5));
        assert_eq!(q(3, 8), r(0.375));
        assert!(q(-1, 3) > r(-0.34) && q(-1, 3) < r(-0.33));
        assert!(r(1e-300) > q(0, 1) && r(-1e-300) < q(1, 1000000007) && !(r(-1e-300) < q(-1, 1000000007)));
        assert!(r(1e300) > q(i64::MAX, 2) && r(-1e300) < q(i64::MIN, 3));
    }

    /// The hash of equal numbers is the hash of one number, whatever the form.
    #[test]
    fn equal_numbers_hash_alike() {
        use std::collections::hash_map::DefaultHasher;
        let h = |x: Num| { let mut s = DefaultHasher::new(); x.hash(&mut s); s.finish() };
        assert_eq!(h(q(1, 2)), h(r(0.5)));
        assert_eq!(h(n(2)), h(r(2.0)));
        assert_eq!(h(q(5, 4)), h(r(1.25)));
        assert_ne!(h(q(1, 3)), h(q(1, 2)));
    }

    /// Mirrors Ideas.Layer0.Number.add_monotone.
    #[test]
    fn addition_is_monotone_in_the_order() {
        let mut m = Maker::new(12);
        for _ in 0..2000 {
            let (a, b, c) = (n(m.range(-1000, 1000)), r(m.range(-2000, 2000) as f64 / 8.0), q(m.range(-50, 50), m.range(1, 9)));
            if a <= b { assert!(a.add(&c).unwrap() <= b.add(&c).unwrap()); }
            if a <= c { assert!(a.add(&b).unwrap() <= c.add(&b).unwrap()); }
        }
    }

    /// Mirrors Ideas.Layer0.Number.frac_add: fractions add exactly (a/b + c/d = (a d + c b) / (b d)), and the four operations undo each other with no rounding.
    #[test]
    fn fractions_are_exact_in_the_four_operations() {
        assert_eq!(q(1, 3).add(&q(1, 6)).unwrap(), q(1, 2));
        assert_eq!(q(1, 10).add(&q(2, 10)).unwrap(), q(3, 10), "a tenth and two tenths are three tenths, with no real number in between");
        assert_eq!(q(1, 3).mul(&n(3)).unwrap(), n(1));
        assert_eq!(n(1).div(&n(3)).unwrap(), q(1, 3));
        assert_eq!(q(2, 3).div(&q(4, 9)).unwrap(), q(3, 2));
        assert_eq!(q(1, 4).sub(&q(1, 2)).unwrap(), q(-1, 4));
        let mut m = Maker::new(13);
        for _ in 0..3000 {
            let (a, b) = (q(m.range(-40, 40), m.range(1, 12)), q(m.range(-40, 40), m.range(1, 12)));
            let s = a.add(&b).unwrap();
            assert_eq!(s.sub(&b).unwrap(), a);
            assert_eq!(a.add(&b).unwrap(), b.add(&a).unwrap());
            assert_eq!(a.mul(&b).unwrap(), b.mul(&a).unwrap());
            assert!(s.is_exact());
            if b != n(0) { assert_eq!(a.mul(&b).unwrap().div(&b).unwrap(), a); }
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
        // a fraction whose terms leave the whole range is said, not rounded
        assert_eq!(q(1, i64::MAX).mul(&q(1, 2)), Err(NumError::Overflow));
        assert_eq!(q(1, i64::MAX).add(&q(1, i64::MAX - 1)), Err(NumError::Overflow));
    }

    #[test]
    fn whole_stays_whole_and_exact_and_a_real_joins_only_when_it_is_given() {
        assert_eq!(n(6).div(&n(3)).unwrap().as_whole(), Some(2));
        assert_eq!(n(1).div(&n(4)).unwrap().as_ratio(), Some((1, 4)));
        assert_eq!(n(1).div(&n(4)).unwrap(), r(0.25));
        assert_eq!(n(2).add(&r(0.5)).unwrap(), r(2.5));
        assert!(!n(2).add(&r(0.5)).unwrap().is_exact(), "a real in, a real out");
        assert_eq!(r(0.5).add(&r(0.5)).unwrap().as_whole(), Some(1));
        assert_eq!(n(9).sqrt().unwrap().as_whole(), Some(3));
        assert_eq!(q(9, 4).sqrt().unwrap(), q(3, 2));
        assert!(!n(2).sqrt().unwrap().is_exact());
        assert_eq!(n(-5).abs().unwrap(), n(5));
        assert_eq!(n(3).min(r(2.5)), r(2.5));
        assert_eq!(n(3).max(r(2.5)), n(3));
        assert_eq!(format!("{} {} {}", n(3), q(1, 4), q(-3, 2)), "3 1/4 -3/2");
    }
}
