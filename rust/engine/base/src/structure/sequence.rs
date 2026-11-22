//! STRUCTURE, FORM TWO: A SEQUENCE (plan 79.30; SYSTEM_SPEC 11 "Layer 0 holds six things" item 4). Items in order, before and after, the difference of two neighbours.
//! Interface: see the module doc of `structure`.
use crate::number::{Num, NumError};

#[derive(Debug, Clone, PartialEq, Default)]
pub struct Sequence<T> { items: Vec<T> }

impl<T> Sequence<T> {
    pub fn new(items: Vec<T>) -> Sequence<T> { Sequence { items } }
    pub fn len(&self) -> usize { self.items.len() }
    pub fn is_empty(&self) -> bool { self.items.is_empty() }
    pub fn get(&self, i: usize) -> Option<&T> { self.items.get(i) }
    pub fn items(&self) -> &[T] { &self.items }
    /// Whether the item at `i` comes before the item at `j` (both must be in the sequence).
    pub fn before(&self, i: usize, j: usize) -> bool { i < j && j < self.items.len() }
    /// Whether the item at `i` comes after the item at `j` (both must be in the sequence).
    pub fn after(&self, i: usize, j: usize) -> bool { self.before(j, i) }
    /// Each item with the next one.
    pub fn pairs(&self) -> Vec<(&T, &T)> { self.items.windows(2).map(|w| (&w[0], &w[1])).collect() }
}

impl<T: PartialEq> Sequence<T> {
    /// The position of the first item equal to this one.
    pub fn position_of(&self, t: &T) -> Option<usize> { self.items.iter().position(|x| x == t) }
}

impl Sequence<Num> {
    /// Each item less the one before it (one fewer than the items).
    pub fn differences(&self) -> Result<Vec<Num>, NumError> { self.items.windows(2).map(|w| w[1].sub(&w[0])).collect() }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::maker::Maker;

    #[test]
    fn before_after_pairs_and_position() {
        let s = Sequence::new(vec!["x", "y", "z", "y"]);
        assert!(s.before(0, 2) && s.after(2, 0) && !s.before(2, 0) && !s.before(1, 1) && !s.before(0, 9));
        assert_eq!(s.position_of(&"y"), Some(1));
        assert_eq!(s.pairs(), vec![(&"x", &"y"), (&"y", &"z"), (&"z", &"y")]);
        assert_eq!(s.len(), 4);
        assert_eq!(s.get(3), Some(&"y"));
    }

    #[test]
    fn differences_of_whole_and_real_items_and_overflow() {
        let s = Sequence::new(vec![Num::whole(1), Num::real(2.5).unwrap(), Num::whole(2)]);
        assert_eq!(s.differences().unwrap(), vec![Num::real(1.5).unwrap(), Num::real(-0.5).unwrap()]);
        assert_eq!(Sequence::new(vec![Num::whole(i64::MIN), Num::whole(1)]).differences(), Err(NumError::Overflow));
        assert!(Sequence::<Num>::new(vec![]).differences().unwrap().is_empty());
    }

    /// The differences add up to the last item less the first (a law of a sequence of numbers; Ideas.Layer0.Structure.differences_sum is the same statement over the integers).
    #[test]
    fn the_differences_add_up_to_the_last_less_the_first() {
        let mut m = Maker::new(53);
        for _ in 0..500 {
            let items: Vec<Num> = (0..2 + m.below(8)).map(|_| Num::whole(m.range(-100, 100))).collect();
            let d = Sequence::new(items.clone()).differences().unwrap();
            let total = d.iter().fold(Num::whole(0), |a, x| a.add(x).unwrap());
            assert_eq!(total, items.last().unwrap().sub(&items[0]).unwrap());
        }
    }
}
