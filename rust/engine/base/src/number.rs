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
//!   `min`, `max`, `Display`.
//! LAWS: lean/ideas/Ideas/Layer0/Number.lean (the order is total, antisymmetric and transitive; addition is monotone in the order).
