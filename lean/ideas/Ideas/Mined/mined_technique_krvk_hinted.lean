import Ideas.Mined.mined_technique_krvk

/-! AUTO-WRITTEN by catspace/incubator/analyst/discovery/prover.py (record) from prover run 20261027_165835_mined_technique_krvk_hinted (hinted) on Ideas.Mined.mined_technique_krvk -- do not edit by
hand. The SUB-GOALS the tools proposed on the way (plan 73.6 (4)): each PROVED one with the proof the tools found (accepted on its own: no sorry,
the standard axioms only, no enumeration); each OPEN one with `sorry` and the prover's last state above it. Nobody wrote these proofs. -/

set_option linter.unusedVariables false

namespace Ideas.Mined.mined_technique_krvk

open Chess hiding between
open Ideas.Vocabulary

/-- OPEN sub-goal of hprog (run 20261027_165835_mined_technique_krvk_hinted); the prover's last state: error(lean.projNonPropFromProp): Invalid projection: Cannot project a value of non-propositional type
  Square
from the expression
  hp.left.2
which has propositional type
  ∃ P L,
    p.board ⋯.1 = some { color := p.turn, kind := PieceKind.king } ∧
      p.board P = some { color := p.turn, kind := PieceKind.rook } ∧
        p.board L = some { color := p.turn.opposite, kind := PieceKind.king } ∧
          ∀ (s : Square), s ≠ ⋯.1 → s ≠ P → s ≠ L → p.board s = none -/
theorem conjecture_hprog_hinted_s1 (p : Position) (hp : inClass p) (r : Position)
  (hr : inClass r ∧ mu r < mu p) (h2 : inClass r) (h3 : mu r < mu p) :
  Forced.Reach ours theirs mate (fun r => inClass r ∧ mu r < mu p) 3 p := by
  sorry

/-- OPEN sub-goal of hprog (run 20261027_165835_mined_technique_krvk_hinted); the prover's last state: error: Tactic `tauto` failed

p : Position
hp h3 : inClass p
h4 : ∀ (r : Position), inClass r ∧ mu r < mu p → Forced.Reach ours theirs mate (fun r => inClass r ∧ mu r < mu p) 3 p
h5 : inClass p ∧ mu p < mu p → Forced.Reach ours theirs mate (fun r => inClass r ∧ mu r < mu p) 3 p
h6 : inClass p ∧ mu p < mu p → False
h8 : ¬(inClass p ∧ mu p < mu p)
h10 : ∀ (r : Position), inClass r ∧ mu r < mu p → Forced.Reach ours theirs mate (fun r => inClass r ∧ mu r < mu p) 3 p
h11 : inClass p
h12 : ¬(inClass p ∧ mu p < mu p)
h14 : inClass p ∧ mu p < mu p → False
⊢ Forced.Reach ours theirs mate (fun r => inCl -/
theorem conjecture_hprog_hinted_s3 (p : Position) (hp h3 : inClass p)
  (h4 :
    ∀ (r : Position), inClass r ∧ mu r < mu p → Forced.Reach ours theirs mate (fun r => inClass r ∧ mu r < mu p) 3 p) :
  Forced.Reach ours theirs mate (fun r => inClass r ∧ mu r < mu p) 3 p := by
  sorry

end Ideas.Mined.mined_technique_krvk
