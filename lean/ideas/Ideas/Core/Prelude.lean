import Mathlib.Data.List.Forall2
import Mathlib.Order.Basic
import Mathlib.Order.Nat
import Mathlib.Order.MinMax
import Mathlib.Tactic.ByContra

/-!
# What the core files take from Mathlib (PLAN 78.56)

The tactics and order lemmas the game-free files use (`by_contra`, `le_refl`, `List.Forall₂`, ...). They came in transitively through the chess kernel before; the core imports
them by name instead, so that no file of `Ideas/Core/` imports (even transitively) a chess file. `Ideas/Core/ForcedByMeasure.lean` needs nothing and imports nothing.
-/
