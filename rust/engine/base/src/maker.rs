//! TEST SUPPORT (plan 79.30; not a seventh thing of layer 0): a seeded maker of cases, so that each test of a law tries it on many cases and the same cases come back on every run. No outside dependency.
//!
//! INTERFACE (callers: the tests of every module and the tests in base/tests).
//!   `Maker::new(seed: u64)`, `next_u64()`, `below(n: u64) -> u64`, `range(lo: i64, hi: i64) -> i64` (both ends in), `fraction() -> f64` (in [0, 1)), `pick<T: Clone>(&[T]) -> T`, `chance() -> bool`.
//!   `random_tree(&mut Maker, depth: u32, shape: Shape) -> Tree<i64>`: a tree of choices with whole results; `Shape { tables: bool, chance: bool }` says whether choices at once and by chance may be made.
