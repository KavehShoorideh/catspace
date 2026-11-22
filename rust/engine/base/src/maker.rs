//! TEST SUPPORT (plan 79.30; not a seventh thing of layer 0): a seeded maker of cases, so that each test of a law tries it on many cases and the same cases come back on every run. No outside dependency.
//!
//! INTERFACE (callers: the tests of every module and the tests in base/tests).
//!   `Maker::new(seed: u64)`, `next_u64()`, `below(n: u64) -> u64`, `range(lo: i64, hi: i64) -> i64` (both ends in), `fraction() -> f64` (in [0, 1)), `pick<T: Clone>(&[T]) -> T`, `chance() -> bool`.
//!   `random_tree(&mut Maker, depth: u32, shape: Shape) -> Tree<i64>`: a tree of choices with whole results; `Shape { tables: bool, chance: bool }` says whether choices at once and by chance may be made.
use crate::choice::{Chooser, Tree};
use crate::number::Num;

/// SplitMix64: small, fast, the same on every machine.
#[derive(Debug, Clone)]
pub struct Maker { state: u64 }

impl Maker {
    pub fn new(seed: u64) -> Maker { Maker { state: seed } }
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// A number below `n` (nought when `n` is nought).
    pub fn below(&mut self, n: u64) -> u64 { if n == 0 { 0 } else { self.next_u64() % n } }
    /// A whole number between `lo` and `hi`, both in.
    pub fn range(&mut self, lo: i64, hi: i64) -> i64 { lo + self.below((hi - lo + 1) as u64) as i64 }
    /// A real in [0, 1).
    pub fn fraction(&mut self) -> f64 { (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64 }
    pub fn pick<T: Clone>(&mut self, items: &[T]) -> T { items[self.below(items.len() as u64) as usize].clone() }
    pub fn chance(&mut self) -> bool { self.next_u64() & 1 == 1 }
}

/// Which kinds of node a made tree may hold besides results and turns.
#[derive(Debug, Clone, Copy)]
pub struct Shape { pub tables: bool, pub chance: bool }

/// A tree of choices with whole results between -9 and 9, `depth` levels deep at most. Chances are quarters (exact fractions), so sums are exactly one.
pub fn random_tree(m: &mut Maker, depth: u32, shape: Shape) -> Tree<i64> {
    if depth == 0 || m.below(5) == 0 { return Tree::Result(m.range(-9, 9)); }
    let mut kinds = vec![0u8, 1];                            // 0 and 1: a turn of A or of B; 2: a table; 3: a chance
    if shape.tables { kinds.push(2); }
    if shape.chance { kinds.push(3); }
    let which = m.pick(&kinds);
    match which {
        0 | 1 => {
            let by = if which == 0 { Chooser::A } else { Chooser::B };
            let n = 1 + m.below(3);
            Tree::turn(by, (0..n).map(|_| random_tree(m, depth - 1, shape)).collect()).expect("at least one option")
        }
        2 => {
            let (r, c) = (1 + m.below(3) as usize, 1 + m.below(3) as usize);
            Tree::table(r, c, (0..r * c).map(|_| random_tree(m, depth - 1, shape)).collect()).expect("a table of cells")
        }
        _ => {
            let n = 1 + m.below(3) as usize;
            // quarters that sum to one: n outcomes share four quarters, each at least none
            let mut left = 4u64;
            let outs: Vec<(Tree<i64>, Num)> = (0..n).map(|i| {
                let q = if i + 1 == n { left } else { m.below(left + 1) };
                left -= q;
                (random_tree(m, depth - 1, shape), Num::frac(q as i64, 4).expect("a quarter"))
            }).collect();
            Tree::chance(outs).expect("chances sum to one")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn the_same_seed_gives_the_same_cases() {
        let (mut a, mut b) = (Maker::new(5), Maker::new(5));
        for _ in 0..50 { assert_eq!(a.next_u64(), b.next_u64()); }
        let mut m = Maker::new(6);
        for _ in 0..500 { let v = m.range(-3, 3); assert!((-3..=3).contains(&v)); let f = m.fraction(); assert!((0.0..1.0).contains(&f)); }
    }
}
