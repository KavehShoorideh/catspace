//! LAYER 0, MODULE 2: SETS AND COUNTING (plan 79.30; SYSTEM_SPEC 11 "Layer 0 holds six things" item 2).
//!
//! INTERFACE (callers: logic's `Domain::Set`, structure's node sets, choice's option sets, and any layer above that needs "the things of one kind").
//!   `Set<T: Ord + Clone>`: `new()`, `from_items(impl IntoIterator<Item = T>)`, `insert(T) -> bool` (true when new), `contains(&T) -> bool`, `len() -> usize` (how many), `is_empty()`, `iter()` (in order),
//!     `union(&Set) -> Set`, `meet(&Set) -> Set` (the members of both), `minus(&Set) -> Set`, `is_part_of(&Set) -> bool` (every member of this one is in the other), `is_disjoint(&Set) -> bool`.
//!   `Kinds<T: Ord + Clone, K: Ord + Clone>`: a kind as a set of the things that are of it. `new()`, `add(K, T)` (the thing is of the kind), `members(&K) -> Set<T>`, `kinds_of(&T) -> Set<K>`, `is_a(&T, &K) -> bool`,
//!     `count(&K) -> usize`, `kinds() -> Set<K>`.
//! LAWS: lean/ideas/Ideas/Layer0/Sets.lean (union and meet are commutative, associative, idempotent and distribute; part-of is the order of union and meet; how many of a union is the sum less how many of the meet; a part is no bigger than the whole).
