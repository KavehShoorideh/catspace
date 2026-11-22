//! PLAN 79.30 LAYER 0, MATHEMATICS AND LOGIC, AS ONE CRATE AT THE BOTTOM (SYSTEM_SPEC 11 "Layers 0, 1 and 2 by what they hold", "Layer 0 is wide enough for two unlike games", "Layer 0 holds six things"; part served: the base under the representation).
//!
//! The test of belonging (Kaveh 2026-11-22): could it be said to one who has never seen any world. This crate depends on no crate of ours, names no game and no thing of a world, and is given once: it is never extracted or learned.
//! Six modules, each an interface with more than one case, each with its laws stated in Lean (lean/ideas/Ideas/Layer0/):
//!   `logic`      relations, placeholders, and, or, not, differ, some, every, at least so many; the fact, the index of facts and the matcher. (Moved here from cs_rep, which hands them on under their old paths.)
//!   `sets`       a finite set, membership, a kind as a set, how many; union, meet, part-of.
//!   `number`     whole and real numbers in one type with one total order; arithmetic that says when it overflows.
//!   `structure`  a graph, a sequence, geometry; one interface of distance that the graph and the plane both give.
//!   `chance`     a finite spread of outcomes with their chances; expectation, product, given a condition.
//!   `choice`     a finite tree of results, choices in turn, choices at once and choices by chance; two readings (can force, value).
//!   `maker`      a seeded maker of cases for tests (no outside dependency). It is test support, not a seventh thing of layer 0.
//! The search of layer 4 is NOT here and is not changed: `choice` is the meaning it must agree with.
pub mod chance;
pub mod choice;
pub mod logic;
pub mod maker;
pub mod number;
pub mod sets;
pub mod structure;
