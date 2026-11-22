//! LAYER 0, MODULE 5: CHANCE (plan 79.30; SYSTEM_SPEC 11 "Layer 0 holds six things" item 5).
//!
//! INTERFACE (callers: choice's chance nodes; any layer above that reads an uncertain act).
//!   `Spread<T>`: a finite spread of outcomes with their chances. The chances are `f64`, each between nought and one, and sum to one within `SUM_TOLERANCE` (1e-9); the spread is refused when made if not.
//!     `Spread::new(Vec<(T, f64)>) -> Result<Spread<T>, ChanceError>`; `Spread::certain(T)`; `Spread::uniform(Vec<T>) -> Result<Spread<T>, ChanceError>`; `outcomes() -> &[(T, f64)]`; `total() -> f64`;
//!     `chance_of(&dyn Fn(&T) -> bool) -> f64`; `expectation(&dyn Fn(&T) -> Num) -> Result<Num, NumError>`; `map(&dyn Fn(&T) -> U) -> Spread<U>` (outcomes are kept, not merged);
//!     `given(&dyn Fn(&T) -> bool) -> Result<Spread<T>, ChanceError>` (the spread given a condition: the outcomes that hold, their chances divided by the chance of the condition; a condition of chance nought is `ImpossibleCondition`);
//!     `product(&Spread<A>, &Spread<B>) -> Spread<(A, B)>` (two that do not depend on each other: the chances multiply).
//!   `ChanceError::{BadChance, DoNotSumToOne, Empty, ImpossibleCondition}`.
//! LAWS: lean/ideas/Ideas/Layer0/Chance.lean (the chances of a product and of a spread given a condition sum to one; expectation is linear).
