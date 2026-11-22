//! LAYER 0, MODULE 2: SETS AND COUNTING (plan 79.30; SYSTEM_SPEC 11 "Layer 0 holds six things" item 2).
//!
//! INTERFACE (callers: logic's `Domain::Set`, structure's node sets, choice's option sets, and any layer above that needs "the things of one kind").
//!   `Set<T: Ord + Clone>`: `new()`, `from_items(impl IntoIterator<Item = T>)`, `insert(T) -> bool` (true when new), `remove(&T) -> bool` (true when it was there; plan 79.31 part G, for the one who keeps a set that changes), `contains(&T) -> bool`, `len() -> usize` (how many), `is_empty()`, `iter()` (in order),
//!     `union(&Set) -> Set`, `meet(&Set) -> Set` (the members of both), `minus(&Set) -> Set`, `is_part_of(&Set) -> bool` (every member of this one is in the other), `is_disjoint(&Set) -> bool`.
//!   `Kinds<T: Ord + Clone, K: Ord + Clone>`: a kind as a set of the things that are of it. `new()`, `add(K, T)` (the thing is of the kind), `members(&K) -> Set<T>`, `kinds_of(&T) -> Set<K>`, `is_a(&T, &K) -> bool`,
//!     `count(&K) -> usize`, `kinds() -> Set<K>`.
//! LAWS: lean/ideas/Ideas/Layer0/Sets.lean (union and meet are commutative, associative, idempotent and distribute; part-of is the order of union and meet; how many of a union is the sum less how many of the meet; a part is no bigger than the whole).
use std::collections::{BTreeMap, BTreeSet};

/// A finite set.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Set<T: Ord + Clone> { items: BTreeSet<T> }

impl<T: Ord + Clone> Default for Set<T> { fn default() -> Set<T> { Set::new() } }

impl<T: Ord + Clone> Set<T> {
    pub fn new() -> Set<T> { Set { items: BTreeSet::new() } }
    pub fn from_items(items: impl IntoIterator<Item = T>) -> Set<T> { Set { items: items.into_iter().collect() } }
    /// Add a member; true when it was not there.
    pub fn insert(&mut self, t: T) -> bool { self.items.insert(t) }
    /// Take a member away; true when it was there.
    pub fn remove(&mut self, t: &T) -> bool { self.items.remove(t) }
    pub fn contains(&self, t: &T) -> bool { self.items.contains(t) }
    /// How many members.
    pub fn len(&self) -> usize { self.items.len() }
    pub fn is_empty(&self) -> bool { self.items.is_empty() }
    pub fn iter(&self) -> impl Iterator<Item = &T> { self.items.iter() }
    pub fn union(&self, o: &Set<T>) -> Set<T> { Set { items: self.items.union(&o.items).cloned().collect() } }
    /// The members of both.
    pub fn meet(&self, o: &Set<T>) -> Set<T> { Set { items: self.items.intersection(&o.items).cloned().collect() } }
    /// The members of this one that are not in the other.
    pub fn minus(&self, o: &Set<T>) -> Set<T> { Set { items: self.items.difference(&o.items).cloned().collect() } }
    /// Every member of this one is in the other.
    pub fn is_part_of(&self, o: &Set<T>) -> bool { self.items.is_subset(&o.items) }
    pub fn is_disjoint(&self, o: &Set<T>) -> bool { self.items.is_disjoint(&o.items) }
}

impl<T: Ord + Clone> IntoIterator for Set<T> { type Item = T; type IntoIter = std::collections::btree_set::IntoIter<T>; fn into_iter(self) -> Self::IntoIter { self.items.into_iter() } }

impl<T: Ord + Clone> FromIterator<T> for Set<T> { fn from_iter<I: IntoIterator<Item = T>>(i: I) -> Set<T> { Set::from_items(i) } }

/// Kinds as sets: each kind is the set of the things that are of it; a thing may be of several kinds.
#[derive(Debug, Clone)]
pub struct Kinds<T: Ord + Clone, K: Ord + Clone> { of_kind: BTreeMap<K, Set<T>> }

impl<T: Ord + Clone, K: Ord + Clone> Default for Kinds<T, K> { fn default() -> Kinds<T, K> { Kinds::new() } }

impl<T: Ord + Clone, K: Ord + Clone> Kinds<T, K> {
    pub fn new() -> Kinds<T, K> { Kinds { of_kind: BTreeMap::new() } }
    /// The thing is of the kind.
    pub fn add(&mut self, kind: K, thing: T) { self.of_kind.entry(kind).or_default().insert(thing); }
    /// The set of things of the kind (empty when the kind is unknown).
    pub fn members(&self, kind: &K) -> Set<T> { self.of_kind.get(kind).cloned().unwrap_or_default() }
    pub fn kinds_of(&self, thing: &T) -> Set<K> { self.of_kind.iter().filter(|(_, s)| s.contains(thing)).map(|(k, _)| k.clone()).collect() }
    pub fn is_a(&self, thing: &T, kind: &K) -> bool { self.of_kind.get(kind).is_some_and(|s| s.contains(thing)) }
    /// How many things are of the kind.
    pub fn count(&self, kind: &K) -> usize { self.of_kind.get(kind).map_or(0, |s| s.len()) }
    pub fn kinds(&self) -> Set<K> { self.of_kind.keys().cloned().collect() }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::maker::Maker;

    fn make(m: &mut Maker) -> Set<i64> { Set::from_items((0..m.below(9)).map(|_| m.range(0, 11))) }

    /// Mirrors Ideas.Layer0.Sets.union_comm, union_assoc, meet_comm, meet_assoc, union_idem, meet_idem, meet_union_distrib, union_meet_distrib.
    #[test]
    fn union_and_meet_are_commutative_associative_idempotent_and_distribute() {
        let mut m = Maker::new(21);
        for _ in 0..2000 {
            let (a, b, c) = (make(&mut m), make(&mut m), make(&mut m));
            assert_eq!(a.union(&b), b.union(&a));
            assert_eq!(a.meet(&b), b.meet(&a));
            assert_eq!(a.union(&b).union(&c), a.union(&b.union(&c)));
            assert_eq!(a.meet(&b).meet(&c), a.meet(&b.meet(&c)));
            assert_eq!(a.union(&a), a);
            assert_eq!(a.meet(&a), a);
            assert_eq!(a.meet(&b.union(&c)), a.meet(&b).union(&a.meet(&c)));
            assert_eq!(a.union(&b.meet(&c)), a.union(&b).meet(&a.union(&c)));
        }
    }

    /// Mirrors Ideas.Layer0.Sets.part_iff_union, part_iff_meet, card_union_add_card_meet, card_le_of_part.
    #[test]
    fn part_of_is_the_order_of_union_and_meet_and_how_many_adds_up() {
        let mut m = Maker::new(22);
        for _ in 0..2000 {
            let (a, b) = (make(&mut m), make(&mut m));
            assert_eq!(a.is_part_of(&b), a.union(&b) == b);
            assert_eq!(a.is_part_of(&b), a.meet(&b) == a);
            assert_eq!(a.union(&b).len() + a.meet(&b).len(), a.len() + b.len());
            if a.is_part_of(&b) { assert!(a.len() <= b.len()); }
            assert!(a.meet(&b).is_part_of(&a) && a.is_part_of(&a.union(&b)));
            assert_eq!(a.minus(&b).union(&a.meet(&b)), a);
            assert!(a.minus(&b).is_disjoint(&b));
        }
    }

    #[test]
    fn a_member_taken_away_is_gone_and_a_set_is_read_member_by_member_in_order() {
        let mut s = Set::from_items([3, 1, 2]);
        assert!(s.remove(&2) && !s.remove(&2) && !s.contains(&2) && s.len() == 2);
        assert!(s.insert(2) && s.contains(&2));
        assert_eq!(s.into_iter().collect::<Vec<_>>(), vec![1, 2, 3]);
    }

    #[test]
    fn a_kind_is_the_set_of_what_is_of_it() {
        let mut k: Kinds<&str, &str> = Kinds::new();
        k.add("round", "x");
        k.add("round", "y");
        k.add("red", "y");
        assert_eq!(k.count(&"round"), 2);
        assert!(k.is_a(&"y", &"red") && !k.is_a(&"x", &"red"));
        assert_eq!(k.kinds_of(&"y"), Set::from_items(["red", "round"]));
        assert_eq!(k.members(&"round").meet(&k.members(&"red")), Set::from_items(["y"]));
        assert_eq!(k.count(&"unknown"), 0);
        assert_eq!(k.kinds().len(), 2);
    }
}
