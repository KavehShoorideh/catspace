//! PLAN 79.30 LAYER 0, MATHEMATICS AND LOGIC, AS ONE CRATE AT THE BOTTOM (SYSTEM_SPEC 11 "Layers 0, 1 and 2 by what they hold", "Layer 0 is wide enough for two unlike games", "Layer 0 holds six things"; part served: the base under the representation).
//!
//! The test of belonging (Kaveh 2026-11-22): could it be said to one who has never seen any world. This crate depends on NO crate at all (plan 79.31 part E), names no game and no thing of a world, and is given once: it is never extracted or learned.
//! Six modules, each an interface with more than one case, each with its laws stated in Lean (lean/ideas/Ideas/Layer0/):
//!   `logic`      relations, placeholders, and, or, not, differ, some, every, at least so many; the fact, the index of facts and the matcher. (Moved here from cs_rep, which hands them on under their old paths.)
//!   `sets`       a finite set, membership, a kind as a set, how many; union, meet, part-of.
//!   `number`     whole numbers, exact fractions and reals in one type with one total order; arithmetic that is exact on fractions and says when it overflows (plan 79.31).
//!   `structure`  a graph, a sequence, geometry; one interface of distance that the graph and the plane both give.
//!   `chance`     a finite spread of outcomes with their chances (exact numbers); expectation, product, given a condition; a belief over suppositions and how it changes by what is seen.
//!   `choice`     a finite tree of results, choices in turn, choices at once between two parties and choices by chance; choosers with an order of their own (opposed and allied are asked, not assumed); the readings: can force, what allied choosers can make sure of, the value of two opposed parties, the mixed choice (an exact linear program in `choice::lp`), a choice against a belief.
//!   `maker`      a seeded maker of cases for tests (no outside dependency). It is test support, not a seventh thing of layer 0.
//! The search of layer 4 is NOT here and is not changed: `choice` is the meaning it must agree with.
pub mod chance;
pub mod choice;
pub mod logic;
pub mod maker;
pub mod number;
pub mod sets;
pub mod structure;
