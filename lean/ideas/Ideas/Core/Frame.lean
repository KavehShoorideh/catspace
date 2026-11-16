import Ideas.Core.Game

/-!
# The neglect theorem, game-free (PLAN 78.56 stage B; moved out of `Ideas/Frame.lean`, proofs unchanged)

`ReadsOnly`, `IsDiff`, `neglect`, `neglect_step`, `frame_of_diff`: over any `Game`, any type of locations and any view of a state at a location. No chess file is imported; the
namespace is the one they had (`Ideas.Frame`). The chess view and diff, and `chess_neglect`, stay in `Ideas/Frame.lean` (story: ANALYST_PLAN 72.10).
-/

namespace Ideas.Frame

section Generic

variable {State Move Loc V α : Type}

/-- A word `w` READS the locations `R` under `view`: two states that look the same on `R` give it the same value. -/
def ReadsOnly (view : State → Loc → V) (w : State → α) (R : Loc → Prop) : Prop :=
  ∀ p q, (∀ l, R l → view p l = view q l) → w p = w q

/-- `D` is a DIFF of the game's moves under `view`: every location whose view a legal move changes lies in it. -/
def IsDiff (g : Ideas.Game.Game State Move) (view : State → Loc → V) (D : State → Move → Loc → Prop) : Prop :=
  ∀ p m l, m ∈ g.moves p → view (g.apply p m) l ≠ view p l → D p m l

/-- THE NEGLECT THEOREM: a word whose value differs after a legal move has reads meeting the move's diff. -/
theorem neglect (g : Ideas.Game.Game State Move) (view : State → Loc → V) (D : State → Move → Loc → Prop) (hD : IsDiff g view D)
    (w : State → α) (R : Loc → Prop) (hw : ReadsOnly view w R) {p : State} {m : Move} (hm : m ∈ g.moves p)
    (hne : w (g.apply p m) ≠ w p) : ∃ l, R l ∧ D p m l := by
  by_contra h
  apply hne
  apply hw
  intro l hl
  by_contra hv
  exact h ⟨l, hl, hD p m l hm hv⟩

/-- THE NEGLECT THEOREM over `Game.step`: a word whose value differs after one step of the game has reads meeting the diff of the move
that made the step. -/
theorem neglect_step (g : Ideas.Game.Game State Move) (view : State → Loc → V) (D : State → Move → Loc → Prop) (hD : IsDiff g view D)
    (w : State → α) (R : Loc → Prop) (hw : ReadsOnly view w R) {p q : State} (hs : g.step p q) (hne : w q ≠ w p) :
    ∃ m, m ∈ g.moves p ∧ q = g.apply p m ∧ ∃ l, R l ∧ D p m l := by
  obtain ⟨m, hm, rfl⟩ := hs
  exact ⟨m, hm, rfl, neglect g view D hD w R hw hm hne⟩

/-- The frame form (what the agenda reads first): a step whose diff misses every read of `w` leaves `w`'s value. -/
theorem frame_of_diff (g : Ideas.Game.Game State Move) (view : State → Loc → V) (D : State → Move → Loc → Prop) (hD : IsDiff g view D)
    (w : State → α) (R : Loc → Prop) (hw : ReadsOnly view w R) {p : State} {m : Move} (hm : m ∈ g.moves p)
    (hmiss : ∀ l, R l → ¬ D p m l) : w (g.apply p m) = w p := by
  by_contra hne
  obtain ⟨l, hl, hd⟩ := neglect g view D hD w R hw hm hne
  exact hmiss l hl hd

end Generic

end Ideas.Frame
