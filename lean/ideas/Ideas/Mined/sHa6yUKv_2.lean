import Ideas.Base.Vocabulary

/-! MINED by catspace/incubator/analyst/discovery/selfmine.py (sHa6yUKv_2): each theorem is a conjectured PREMISE over the base vocabulary with the OUTCOME
`forcingOf p c 5 != 0` (Ideas.Vocabulary.forcingOf: side c, to move, forces a gain of 5 or more in 1 to 3 moves against every defence).
Squares are named from c's side. A proof that failed stays as `sorry` with the error in the comment above it: the statement is the finding. -/

set_option linter.unusedVariables false
set_option linter.unusedSimpArgs false
set_option maxHeartbeats 100000

namespace Ideas.Mined.sHa6yUKv_2

open Chess hiding between
open Ideas.Vocabulary

-- PROOF FAILED: unsolved goals p : Position h : 1 ≤ onum (attackerCount p p.turn.opposite) (some (named p.turn (3, 1))) a : forcingOf p p.turn 5 = 0 ⊢ False
/-- how many of their pieces attack d2 >= 1 (squares named from the mover's side; covers plies k=[0, 1, 2, 3]) -/
theorem premise_01 (p : Position) (c : Color) (hc : p.turn = c) :
    ((onum (attackerCount p (Color.opposite c)) (some (named c (((3 : Fin 8)), ((1 : Fin 8)))))) ≥ 1) → forcingOf p c 5 ≠ 0 := by
  sorry

-- PROOF FAILED: unsolved goals p : Position h : onum (attackerCount p p.turn.opposite) (some (named p.turn (2, 5))) = 0 a : forcingOf p p.turn 5 = 0 ⊢ False
/-- how many of their pieces attack c6 <= 0 (squares named from the mover's side; covers plies k=[1, 2, 3]) -/
theorem premise_02 (p : Position) (c : Color) (hc : p.turn = c) :
    ((onum (attackerCount p (Color.opposite c)) (some (named c (((2 : Fin 8)), ((5 : Fin 8)))))) ≤ 0) → forcingOf p c 5 ≠ 0 := by
  sorry

-- PROOF FAILED: unsolved goals p : Position h : 1 ≤ onum (fun s => attackableOf p s 1) (some (named p.turn (0, 7))) a : forcingOf p p.turn 5 = 0 ⊢ False
/-- whether the other side can attack a8 within 1 of its moves, every landing square safe by the exchange (1 or 0) >= 1 (squares named from the mover's side; covers plies k=[1, 2, 3]) -/
theorem premise_03 (p : Position) (c : Color) (hc : p.turn = c) :
    ((onum (fun s => attackableOf p s 1) (some (named c (((0 : Fin 8)), ((7 : Fin 8)))))) ≥ 1) → forcingOf p c 5 ≠ 0 := by
  sorry

-- PROOF FAILED: unsolved goals p : Position h : 1 ≤ afterMove p (some (named p.turn (6, 7))) (some (named p.turn (5, 5))) fun p_1 => onum (attackerCount p_1 p.turn) (some (named p.turn (5, 5))) a : forcingOf p p.turn 5 = 0 ⊢ False
/-- how many of our pieces attack f6 after the piece on g8 moves to f6 >= 1 (squares named from the mover's side; covers plies k=[0, 2, 3]) -/
theorem premise_04 (p : Position) (c : Color) (hc : p.turn = c) :
    ((afterMove p (some (named c (((6 : Fin 8)), ((7 : Fin 8))))) (some (named c (((5 : Fin 8)), ((5 : Fin 8))))) (fun p => (onum (attackerCount p c) (some (named c (((5 : Fin 8)), ((5 : Fin 8)))))))) ≥ 1) → forcingOf p c 5 ≠ 0 := by
  sorry

-- PROOF FAILED: unsolved goals p : Position h : 1 ≤ onum (fun s => attackableOf p s 1) (some (named p.turn (4, 7))) a : forcingOf p p.turn 5 = 0 ⊢ False
/-- whether the other side can attack e8 within 1 of its moves, every landing square safe by the exchange (1 or 0) >= 1 (squares named from the mover's side; covers plies k=[2, 3]) -/
theorem premise_05 (p : Position) (c : Color) (hc : p.turn = c) :
    ((onum (fun s => attackableOf p s 1) (some (named c (((4 : Fin 8)), ((7 : Fin 8)))))) ≥ 1) → forcingOf p c 5 ≠ 0 := by
  sorry

-- PROOF FAILED: unsolved goals p : Position h : 1 ≤ onum (fun s => attackableOf p s 1) (some (named p.turn (6, 4))) a : forcingOf p p.turn 5 = 0 ⊢ False
/-- whether the other side can attack g5 within 1 of its moves, every landing square safe by the exchange (1 or 0) >= 1 (squares named from the mover's side; covers plies k=[1, 3]) -/
theorem premise_06 (p : Position) (c : Color) (hc : p.turn = c) :
    ((onum (fun s => attackableOf p s 1) (some (named c (((6 : Fin 8)), ((4 : Fin 8)))))) ≥ 1) → forcingOf p c 5 ≠ 0 := by
  sorry

-- PROOF FAILED: unsolved goals p : Position h : 2 ≤ afterMove p (some (named p.turn (6, 4))) (some (named p.turn (3, 7))) fun p_1 => onum (attackerCount p_1 p.turn.opposite) (some (named p.turn (3, 7))) a : forcingOf p p.turn 5 = 0 ⊢ False
/-- how many of their pieces attack d8 after the piece on g5 moves to d8 >= 2 (squares named from the mover's side; covers plies k=[1, 3]) -/
theorem premise_07 (p : Position) (c : Color) (hc : p.turn = c) :
    ((afterMove p (some (named c (((6 : Fin 8)), ((4 : Fin 8))))) (some (named c (((3 : Fin 8)), ((7 : Fin 8))))) (fun p => (onum (attackerCount p (Color.opposite c)) (some (named c (((3 : Fin 8)), ((7 : Fin 8)))))))) ≥ 2) → forcingOf p c 5 ≠ 0 := by
  sorry

-- PROOF FAILED: unsolved goals p : Position h : 3 ≤ onum (attackerCount p p.turn.opposite) (some (named p.turn (3, 7))) a : forcingOf p p.turn 5 = 0 ⊢ False
/-- how many of their pieces attack d8 >= 3 (squares named from the mover's side; covers plies k=[3]) -/
theorem premise_08 (p : Position) (c : Color) (hc : p.turn = c) :
    ((onum (attackerCount p (Color.opposite c)) (some (named c (((3 : Fin 8)), ((7 : Fin 8)))))) ≥ 3) → forcingOf p c 5 ≠ 0 := by
  sorry

-- PROOF FAILED: unsolved goals p : Position h : 1 ≤ onum (seeOf p) (some (named p.turn (2, 5))) a : forcingOf p p.turn 5 = 0 ⊢ False
/-- what the other side wins by the exchange on c6 >= 1 (squares named from the mover's side; covers plies k=[3]) -/
theorem premise_09 (p : Position) (c : Color) (hc : p.turn = c) :
    ((onum (seeOf p) (some (named c (((2 : Fin 8)), ((5 : Fin 8)))))) ≥ 1) → forcingOf p c 5 ≠ 0 := by
  sorry

-- PROOF FAILED: unsolved goals p : Position left : onum (declinedOf p) (some (named p.turn (0, 7))) ≤ 1 right : onum (attackerCount p p.turn.opposite) (some (named p.turn (2, 7))) ≤ 1 a : forcingOf p p.turn 5 = 0 ⊢ False
/-- what the side to move can win by the exchange on a square other than a8: its best capture if it declines to take on a8 <= 1 AND how many of their pieces attack c8 <= 1 (squares named from the mover's side; covers plies k=[0, 1, 2, 3]) -/
theorem premise_10 (p : Position) (c : Color) (hc : p.turn = c) :
    ((onum (declinedOf p) (some (named c (((0 : Fin 8)), ((7 : Fin 8)))))) ≤ 1) ∧ ((onum (attackerCount p (Color.opposite c)) (some (named c (((2 : Fin 8)), ((7 : Fin 8)))))) ≤ 1) → forcingOf p c 5 ≠ 0 := by
  sorry

-- PROOF FAILED: unsolved goals p : Position left : onum (declinedOf p) (some (named p.turn (0, 7))) ≤ 1 right : onum (attackerCount p p.turn.opposite) (some (named p.turn (6, 6))) ≤ 1 a : forcingOf p p.turn 5 = 0 ⊢ False
/-- what the side to move can win by the exchange on a square other than a8: its best capture if it declines to take on a8 <= 1 AND how many of their pieces attack g7 <= 1 (squares named from the mover's side; covers plies k=[0, 1, 2, 3]) -/
theorem premise_11 (p : Position) (c : Color) (hc : p.turn = c) :
    ((onum (declinedOf p) (some (named c (((0 : Fin 8)), ((7 : Fin 8)))))) ≤ 1) ∧ ((onum (attackerCount p (Color.opposite c)) (some (named c (((6 : Fin 8)), ((6 : Fin 8)))))) ≤ 1) → forcingOf p c 5 ≠ 0 := by
  sorry

-- PROOF FAILED: unsolved goals p : Position left : onum (attackerCount p p.turn.opposite) (some (named p.turn (2, 7))) ≤ 1 right : (afterMove p (some (named p.turn (2, 5))) (some (named p.turn (0, 7))) fun p_1 => onum (attackerCount p_1 p.turn.opposite) (some (named p.turn (0, 7)))) = 0 a : forcingOf p p.turn 5 = 0
/-- how many of their pieces attack c8 <= 1 AND how many of their pieces attack a8 after the piece on c6 moves to a8 <= 0 (squares named from the mover's side; covers plies k=[0, 1, 2, 3]) -/
theorem premise_12 (p : Position) (c : Color) (hc : p.turn = c) :
    ((onum (attackerCount p (Color.opposite c)) (some (named c (((2 : Fin 8)), ((7 : Fin 8)))))) ≤ 1) ∧ ((afterMove p (some (named c (((2 : Fin 8)), ((5 : Fin 8))))) (some (named c (((0 : Fin 8)), ((7 : Fin 8))))) (fun p => (onum (attackerCount p (Color.opposite c)) (some (named c (((0 : Fin 8)), ((7 : Fin 8)))))))) ≤ 0) → forcingOf p c 5 ≠ 0 := by
  sorry

-- PROOF FAILED: unsolved goals p : Position left : onum (attackerCount p p.turn.opposite) (some (named p.turn (6, 6))) ≤ 1 right : (afterMove p (some (named p.turn (2, 5))) (some (named p.turn (0, 7))) fun p_1 => onum (attackerCount p_1 p.turn.opposite) (some (named p.turn (0, 7)))) = 0 a : forcingOf p p.turn 5 = 0
/-- how many of their pieces attack g7 <= 1 AND how many of their pieces attack a8 after the piece on c6 moves to a8 <= 0 (squares named from the mover's side; covers plies k=[0, 1, 2, 3]) -/
theorem premise_13 (p : Position) (c : Color) (hc : p.turn = c) :
    ((onum (attackerCount p (Color.opposite c)) (some (named c (((6 : Fin 8)), ((6 : Fin 8)))))) ≤ 1) ∧ ((afterMove p (some (named c (((2 : Fin 8)), ((5 : Fin 8))))) (some (named c (((0 : Fin 8)), ((7 : Fin 8))))) (fun p => (onum (attackerCount p (Color.opposite c)) (some (named c (((0 : Fin 8)), ((7 : Fin 8)))))))) ≤ 0) → forcingOf p c 5 ≠ 0 := by
  sorry

-- PROOF FAILED: unsolved goals p : Position left : 1 ≤ onum (attackerCount p p.turn) (some (named p.turn (4, 2))) right : onum (attackerCount p p.turn.opposite) (some (named p.turn (2, 7))) ≤ 1 a : forcingOf p p.turn 5 = 0 ⊢ False
/-- how many of our pieces attack e3 >= 1 AND how many of their pieces attack c8 <= 1 (squares named from the mover's side; covers plies k=[0, 2, 3]) -/
theorem premise_14 (p : Position) (c : Color) (hc : p.turn = c) :
    ((onum (attackerCount p c) (some (named c (((4 : Fin 8)), ((2 : Fin 8)))))) ≥ 1) ∧ ((onum (attackerCount p (Color.opposite c)) (some (named c (((2 : Fin 8)), ((7 : Fin 8)))))) ≤ 1) → forcingOf p c 5 ≠ 0 := by
  sorry

-- PROOF FAILED: unsolved goals p : Position left : 1 ≤ onum (attackerCount p p.turn) (some (named p.turn (4, 2))) right : onum (attackerCount p p.turn.opposite) (some (named p.turn (4, 5))) ≤ 1 a : forcingOf p p.turn 5 = 0 ⊢ False
/-- how many of our pieces attack e3 >= 1 AND how many of their pieces attack e6 <= 1 (squares named from the mover's side; covers plies k=[0, 2, 3]) -/
theorem premise_15 (p : Position) (c : Color) (hc : p.turn = c) :
    ((onum (attackerCount p c) (some (named c (((4 : Fin 8)), ((2 : Fin 8)))))) ≥ 1) ∧ ((onum (attackerCount p (Color.opposite c)) (some (named c (((4 : Fin 8)), ((5 : Fin 8)))))) ≤ 1) → forcingOf p c 5 ≠ 0 := by
  sorry

-- PROOF FAILED: unsolved goals p : Position left : 1 ≤ onum (attackerCount p p.turn) (some (named p.turn (4, 2))) right : onum (attackerCount p p.turn.opposite) (some (named p.turn (6, 6))) ≤ 1 a : forcingOf p p.turn 5 = 0 ⊢ False
/-- how many of our pieces attack e3 >= 1 AND how many of their pieces attack g7 <= 1 (squares named from the mover's side; covers plies k=[0, 2, 3]) -/
theorem premise_16 (p : Position) (c : Color) (hc : p.turn = c) :
    ((onum (attackerCount p c) (some (named c (((4 : Fin 8)), ((2 : Fin 8)))))) ≥ 1) ∧ ((onum (attackerCount p (Color.opposite c)) (some (named c (((6 : Fin 8)), ((6 : Fin 8)))))) ≤ 1) → forcingOf p c 5 ≠ 0 := by
  sorry

-- PROOF FAILED: unsolved goals p : Position left : onum (declinedOf p) (some (named p.turn (0, 7))) ≤ 1 right : onum (attackerCount p p.turn.opposite) (some (named p.turn (4, 5))) ≤ 1 a : forcingOf p p.turn 5 = 0 ⊢ False
/-- what the side to move can win by the exchange on a square other than a8: its best capture if it declines to take on a8 <= 1 AND how many of their pieces attack e6 <= 1 (squares named from the mover's side; covers plies k=[0, 2, 3]) -/
theorem premise_17 (p : Position) (c : Color) (hc : p.turn = c) :
    ((onum (declinedOf p) (some (named c (((0 : Fin 8)), ((7 : Fin 8)))))) ≤ 1) ∧ ((onum (attackerCount p (Color.opposite c)) (some (named c (((4 : Fin 8)), ((5 : Fin 8)))))) ≤ 1) → forcingOf p c 5 ≠ 0 := by
  sorry

-- PROOF FAILED: unsolved goals p : Position left : onum (attackerCount p p.turn.opposite) (some (named p.turn (2, 7))) ≤ 1 right : 1 ≤ onum (fun s => attackableOf p s 1) (some (named p.turn (0, 3))) a : forcingOf p p.turn 5 = 0 ⊢ False
/-- how many of their pieces attack c8 <= 1 AND whether the other side can attack a4 within 1 of its moves, every landing square safe by the exchange (1 or 0) >= 1 (squares named from the mover's side; covers plies k=[1, 2, 3]) -/
theorem premise_18 (p : Position) (c : Color) (hc : p.turn = c) :
    ((onum (attackerCount p (Color.opposite c)) (some (named c (((2 : Fin 8)), ((7 : Fin 8)))))) ≤ 1) ∧ ((onum (fun s => attackableOf p s 1) (some (named c (((0 : Fin 8)), ((3 : Fin 8)))))) ≥ 1) → forcingOf p c 5 ≠ 0 := by
  sorry

-- PROOF FAILED: unsolved goals p : Position left : onum (attackerCount p p.turn.opposite) (some (named p.turn (2, 7))) ≤ 1 right : 1 ≤ onum (fun s => attackableOf p s 1) (some (named p.turn (4, 5))) a : forcingOf p p.turn 5 = 0 ⊢ False
/-- how many of their pieces attack c8 <= 1 AND whether the other side can attack e6 within 1 of its moves, every landing square safe by the exchange (1 or 0) >= 1 (squares named from the mover's side; covers plies k=[1, 2, 3]) -/
theorem premise_19 (p : Position) (c : Color) (hc : p.turn = c) :
    ((onum (attackerCount p (Color.opposite c)) (some (named c (((2 : Fin 8)), ((7 : Fin 8)))))) ≤ 1) ∧ ((onum (fun s => attackableOf p s 1) (some (named c (((4 : Fin 8)), ((5 : Fin 8)))))) ≥ 1) → forcingOf p c 5 ≠ 0 := by
  sorry

-- PROOF FAILED: unsolved goals p : Position left : onum (attackerCount p p.turn.opposite) (some (named p.turn (6, 6))) ≤ 1 right : 1 ≤ onum (fun s => attackableOf p s 1) (some (named p.turn (0, 3))) a : forcingOf p p.turn 5 = 0 ⊢ False
/-- how many of their pieces attack g7 <= 1 AND whether the other side can attack a4 within 1 of its moves, every landing square safe by the exchange (1 or 0) >= 1 (squares named from the mover's side; covers plies k=[1, 2, 3]) -/
theorem premise_20 (p : Position) (c : Color) (hc : p.turn = c) :
    ((onum (attackerCount p (Color.opposite c)) (some (named c (((6 : Fin 8)), ((6 : Fin 8)))))) ≤ 1) ∧ ((onum (fun s => attackableOf p s 1) (some (named c (((0 : Fin 8)), ((3 : Fin 8)))))) ≥ 1) → forcingOf p c 5 ≠ 0 := by
  sorry

-- PROOF FAILED: unsolved goals p : Position h : (afterMove p (some (named p.turn (2, 0))) (some (named p.turn (2, 5))) fun p_1 => onum (seeOf p_1) (some (named p.turn (2, 5)))) = 0 a : forcingOf p p.turn 5 = 0 ⊢ False
/-- what the other side wins by the exchange on c6 after the piece on c1 moves to c6 <= 0 (squares named from the mover's side; covers plies k=[0, 1, 2, 3]) -/
theorem premise_21 (p : Position) (c : Color) (hc : p.turn = c) :
    ((afterMove p (some (named c (((2 : Fin 8)), ((0 : Fin 8))))) (some (named c (((2 : Fin 8)), ((5 : Fin 8))))) (fun p => (onum (seeOf p) (some (named c (((2 : Fin 8)), ((5 : Fin 8)))))))) ≤ 0) → forcingOf p c 5 ≠ 0 := by
  sorry

-- PROOF FAILED: unsolved goals p : Position h : 1 ≤ onum (attackerCount p p.turn) (some (named p.turn (5, 6))) a : forcingOf p p.turn 5 = 0 ⊢ False
/-- how many of our pieces attack f7 >= 1 (squares named from the mover's side; covers plies k=[0, 2, 3]) -/
theorem premise_22 (p : Position) (c : Color) (hc : p.turn = c) :
    ((onum (attackerCount p c) (some (named c (((5 : Fin 8)), ((6 : Fin 8)))))) ≥ 1) → forcingOf p c 5 ≠ 0 := by
  sorry

-- PROOF FAILED: unsolved goals p : Position h : (afterMove p (some (named p.turn (0, 7))) (some (named p.turn (2, 5))) fun p_1 => onum (attackerCount p_1 p.turn.opposite) (some (named p.turn (2, 5)))) = 0 a : forcingOf p p.turn 5 = 0 ⊢ False
/-- how many of their pieces attack c6 after the piece on a8 moves to c6 <= 0 (squares named from the mover's side; covers plies k=[1, 2, 3]) -/
theorem premise_23 (p : Position) (c : Color) (hc : p.turn = c) :
    ((afterMove p (some (named c (((0 : Fin 8)), ((7 : Fin 8))))) (some (named c (((2 : Fin 8)), ((5 : Fin 8))))) (fun p => (onum (attackerCount p (Color.opposite c)) (some (named c (((2 : Fin 8)), ((5 : Fin 8)))))))) ≤ 0) → forcingOf p c 5 ≠ 0 := by
  sorry

-- PROOF FAILED: unsolved goals p : Position h : (afterMove p (some (named p.turn (4, 5))) (some (named p.turn (2, 5))) fun p_1 => onum (attackerCount p_1 p.turn.opposite) (some (named p.turn (2, 5)))) = 0 a : forcingOf p p.turn 5 = 0 ⊢ False
/-- how many of their pieces attack c6 after the piece on e6 moves to c6 <= 0 (squares named from the mover's side; covers plies k=[1, 2, 3]) -/
theorem premise_24 (p : Position) (c : Color) (hc : p.turn = c) :
    ((afterMove p (some (named c (((4 : Fin 8)), ((5 : Fin 8))))) (some (named c (((2 : Fin 8)), ((5 : Fin 8))))) (fun p => (onum (attackerCount p (Color.opposite c)) (some (named c (((2 : Fin 8)), ((5 : Fin 8)))))))) ≤ 0) → forcingOf p c 5 ≠ 0 := by
  sorry

-- PROOF FAILED: unsolved goals p : Position h : 1 ≤ afterMove p (some (named p.turn (6, 4))) (some (named p.turn (2, 7))) fun p_1 => onum (attackerCount p_1 p.turn.opposite) (some (named p.turn (2, 7))) a : forcingOf p p.turn 5 = 0 ⊢ False
/-- how many of their pieces attack c8 after the piece on g5 moves to c8 >= 1 (squares named from the mover's side; covers plies k=[1, 2, 3]) -/
theorem premise_25 (p : Position) (c : Color) (hc : p.turn = c) :
    ((afterMove p (some (named c (((6 : Fin 8)), ((4 : Fin 8))))) (some (named c (((2 : Fin 8)), ((7 : Fin 8))))) (fun p => (onum (attackerCount p (Color.opposite c)) (some (named c (((2 : Fin 8)), ((7 : Fin 8)))))))) ≥ 1) → forcingOf p c 5 ≠ 0 := by
  sorry

-- PROOF FAILED: unsolved goals p : Position h : 2 ≤ onum (attackerCount p p.turn.opposite) (some (named p.turn (2, 4))) a : forcingOf p p.turn 5 = 0 ⊢ False
/-- how many of their pieces attack c5 >= 2 (squares named from the mover's side; covers plies k=[2, 3]) -/
theorem premise_26 (p : Position) (c : Color) (hc : p.turn = c) :
    ((onum (attackerCount p (Color.opposite c)) (some (named c (((2 : Fin 8)), ((4 : Fin 8)))))) ≥ 2) → forcingOf p c 5 ≠ 0 := by
  sorry

-- PROOF FAILED: unsolved goals p : Position h : 1 ≤ onum (fun s => attackableOf p s 1) (some (named p.turn (5, 6))) a : forcingOf p p.turn 5 = 0 ⊢ False
/-- whether the other side can attack f7 within 1 of its moves, every landing square safe by the exchange (1 or 0) >= 1 (squares named from the mover's side; covers plies k=[1, 3]) -/
theorem premise_27 (p : Position) (c : Color) (hc : p.turn = c) :
    ((onum (fun s => attackableOf p s 1) (some (named c (((5 : Fin 8)), ((6 : Fin 8)))))) ≥ 1) → forcingOf p c 5 ≠ 0 := by
  sorry

-- PROOF FAILED: unsolved goals p : Position h : 1 ≤ afterMove p (some (named p.turn (0, 3))) (some (named p.turn (2, 5))) fun p_1 => onum (seeOf p_1) (some (named p.turn (2, 5))) a : forcingOf p p.turn 5 = 0 ⊢ False
/-- what the other side wins by the exchange on c6 after the piece on a4 moves to c6 >= 1 (squares named from the mover's side; covers plies k=[0, 3]) -/
theorem premise_28 (p : Position) (c : Color) (hc : p.turn = c) :
    ((afterMove p (some (named c (((0 : Fin 8)), ((3 : Fin 8))))) (some (named c (((2 : Fin 8)), ((5 : Fin 8))))) (fun p => (onum (seeOf p) (some (named c (((2 : Fin 8)), ((5 : Fin 8)))))))) ≥ 1) → forcingOf p c 5 ≠ 0 := by
  sorry

-- PROOF FAILED: unsolved goals p : Position h : 1 ≤ afterMove p (some (named p.turn (6, 7))) (some (named p.turn (4, 4))) fun p_1 => onum (attackerCount p_1 p.turn.opposite) (some (named p.turn (4, 4))) a : forcingOf p p.turn 5 = 0 ⊢ False
/-- how many of their pieces attack e5 after the piece on g8 moves to e5 >= 1 (squares named from the mover's side; covers plies k=[1, 3]) -/
theorem premise_29 (p : Position) (c : Color) (hc : p.turn = c) :
    ((afterMove p (some (named c (((6 : Fin 8)), ((7 : Fin 8))))) (some (named c (((4 : Fin 8)), ((4 : Fin 8))))) (fun p => (onum (attackerCount p (Color.opposite c)) (some (named c (((4 : Fin 8)), ((4 : Fin 8)))))))) ≥ 1) → forcingOf p c 5 ≠ 0 := by
  sorry

-- PROOF FAILED: unsolved goals p : Position h : 5 ≤ afterMove p (some (named p.turn (0, 7))) (some (named p.turn (2, 5))) fun p_1 => onum (seeOf p_1) (some (named p.turn (2, 5))) a : forcingOf p p.turn 5 = 0 ⊢ False
/-- what the other side wins by the exchange on c6 after the piece on a8 moves to c6 >= 5 (squares named from the mover's side; covers plies k=[3]) -/
theorem premise_30 (p : Position) (c : Color) (hc : p.turn = c) :
    ((afterMove p (some (named c (((0 : Fin 8)), ((7 : Fin 8))))) (some (named c (((2 : Fin 8)), ((5 : Fin 8))))) (fun p => (onum (seeOf p) (some (named c (((2 : Fin 8)), ((5 : Fin 8)))))))) ≥ 5) → forcingOf p c 5 ≠ 0 := by
  sorry

-- PROOF FAILED: unsolved goals p : Position h : 4 ≤ afterMove p (some (named p.turn (0, 7))) (some (named p.turn (4, 6))) fun p_1 => onum (attackerCount p_1 p.turn.opposite) (some (named p.turn (4, 6))) a : forcingOf p p.turn 5 = 0 ⊢ False
/-- how many of their pieces attack e7 after the piece on a8 moves to e7 >= 4 (squares named from the mover's side; covers plies k=[3]) -/
theorem premise_31 (p : Position) (c : Color) (hc : p.turn = c) :
    ((afterMove p (some (named c (((0 : Fin 8)), ((7 : Fin 8))))) (some (named c (((4 : Fin 8)), ((6 : Fin 8))))) (fun p => (onum (attackerCount p (Color.opposite c)) (some (named c (((4 : Fin 8)), ((6 : Fin 8)))))))) ≥ 4) → forcingOf p c 5 ≠ 0 := by
  sorry

-- PROOF FAILED: unsolved goals p : Position h : 4 ≤ afterMove p (some (named p.turn (4, 5))) (some (named p.turn (4, 6))) fun p_1 => onum (attackerCount p_1 p.turn.opposite) (some (named p.turn (4, 6))) a : forcingOf p p.turn 5 = 0 ⊢ False
/-- how many of their pieces attack e7 after the piece on e6 moves to e7 >= 4 (squares named from the mover's side; covers plies k=[3]) -/
theorem premise_32 (p : Position) (c : Color) (hc : p.turn = c) :
    ((afterMove p (some (named c (((4 : Fin 8)), ((5 : Fin 8))))) (some (named c (((4 : Fin 8)), ((6 : Fin 8))))) (fun p => (onum (attackerCount p (Color.opposite c)) (some (named c (((4 : Fin 8)), ((6 : Fin 8)))))))) ≥ 4) → forcingOf p c 5 ≠ 0 := by
  sorry

-- PROOF FAILED: unsolved goals p : Position left : onum (declinedOf p) (some (named p.turn (0, 7))) ≤ 1 right : 1 ≤ afterMove p (some (named p.turn (0, 7))) (some (named p.turn (3, 5))) fun p_1 => onum (attackerCount p_1 p.turn.opposite) (some (named p.turn (3, 5))) a : forcingOf p p.turn 5 = 0 ⊢ False
/-- what the side to move can win by the exchange on a square other than a8: its best capture if it declines to take on a8 <= 1 AND how many of their pieces attack d6 after the piece on a8 moves to d6 >= 1 (squares named from the mover's side; covers plies k=[0, 1, 2, 3]) -/
theorem premise_33 (p : Position) (c : Color) (hc : p.turn = c) :
    ((onum (declinedOf p) (some (named c (((0 : Fin 8)), ((7 : Fin 8)))))) ≤ 1) ∧ ((afterMove p (some (named c (((0 : Fin 8)), ((7 : Fin 8))))) (some (named c (((3 : Fin 8)), ((5 : Fin 8))))) (fun p => (onum (attackerCount p (Color.opposite c)) (some (named c (((3 : Fin 8)), ((5 : Fin 8)))))))) ≥ 1) → forcingOf p c 5 ≠ 0 := by
  sorry

-- PROOF FAILED: unsolved goals p : Position left : onum (declinedOf p) (some (named p.turn (0, 7))) ≤ 1 right : (afterMove p (some (named p.turn (0, 7))) (some (named p.turn (6, 4))) fun p_1 => onum (attackerCount p_1 p.turn.opposite) (some (named p.turn (6, 4)))) = 0 a : forcingOf p p.turn 5 = 0 ⊢ False
/-- what the side to move can win by the exchange on a square other than a8: its best capture if it declines to take on a8 <= 1 AND how many of their pieces attack g5 after the piece on a8 moves to g5 <= 0 (squares named from the mover's side; covers plies k=[0, 1, 2, 3]) -/
theorem premise_34 (p : Position) (c : Color) (hc : p.turn = c) :
    ((onum (declinedOf p) (some (named c (((0 : Fin 8)), ((7 : Fin 8)))))) ≤ 1) ∧ ((afterMove p (some (named c (((0 : Fin 8)), ((7 : Fin 8))))) (some (named c (((6 : Fin 8)), ((4 : Fin 8))))) (fun p => (onum (attackerCount p (Color.opposite c)) (some (named c (((6 : Fin 8)), ((4 : Fin 8)))))))) ≤ 0) → forcingOf p c 5 ≠ 0 := by
  sorry

-- PROOF FAILED: unsolved goals p : Position left : onum (attackerCount p p.turn.opposite) (some (named p.turn (2, 7))) ≤ 1 right : (afterMove p (some (named p.turn (5, 2))) (some (named p.turn (3, 5))) fun p_1 => onum (seeOf p_1) (some (named p.turn (3, 5)))) ≤ 6 a : forcingOf p p.turn 5 = 0 ⊢ False
/-- how many of their pieces attack c8 <= 1 AND what the other side wins by the exchange on d6 after the piece on f3 moves to d6 <= 6 (squares named from the mover's side; covers plies k=[0, 1, 2, 3]) -/
theorem premise_35 (p : Position) (c : Color) (hc : p.turn = c) :
    ((onum (attackerCount p (Color.opposite c)) (some (named c (((2 : Fin 8)), ((7 : Fin 8)))))) ≤ 1) ∧ ((afterMove p (some (named c (((5 : Fin 8)), ((2 : Fin 8))))) (some (named c (((3 : Fin 8)), ((5 : Fin 8))))) (fun p => (onum (seeOf p) (some (named c (((3 : Fin 8)), ((5 : Fin 8)))))))) ≤ 6) → forcingOf p c 5 ≠ 0 := by
  sorry

-- PROOF FAILED: unsolved goals p : Position left : onum (attackerCount p p.turn.opposite) (some (named p.turn (6, 6))) ≤ 1 right : (afterMove p (some (named p.turn (5, 2))) (some (named p.turn (3, 5))) fun p_1 => onum (seeOf p_1) (some (named p.turn (3, 5)))) ≤ 6 a : forcingOf p p.turn 5 = 0 ⊢ False
/-- how many of their pieces attack g7 <= 1 AND what the other side wins by the exchange on d6 after the piece on f3 moves to d6 <= 6 (squares named from the mover's side; covers plies k=[0, 1, 2, 3]) -/
theorem premise_36 (p : Position) (c : Color) (hc : p.turn = c) :
    ((onum (attackerCount p (Color.opposite c)) (some (named c (((6 : Fin 8)), ((6 : Fin 8)))))) ≤ 1) ∧ ((afterMove p (some (named c (((5 : Fin 8)), ((2 : Fin 8))))) (some (named c (((3 : Fin 8)), ((5 : Fin 8))))) (fun p => (onum (seeOf p) (some (named c (((3 : Fin 8)), ((5 : Fin 8)))))))) ≤ 6) → forcingOf p c 5 ≠ 0 := by
  sorry

-- PROOF FAILED: unsolved goals p : Position left : 1 ≤ afterMove p (some (named p.turn (0, 7))) (some (named p.turn (3, 5))) fun p_1 => onum (attackerCount p_1 p.turn.opposite) (some (named p.turn (3, 5))) right : (afterMove p (some (named p.turn (2, 5))) (some (named p.turn (0, 7))) fun p_1 => onum (attackerCount 
/-- how many of their pieces attack d6 after the piece on a8 moves to d6 >= 1 AND how many of their pieces attack a8 after the piece on c6 moves to a8 <= 0 (squares named from the mover's side; covers plies k=[0, 1, 2, 3]) -/
theorem premise_37 (p : Position) (c : Color) (hc : p.turn = c) :
    ((afterMove p (some (named c (((0 : Fin 8)), ((7 : Fin 8))))) (some (named c (((3 : Fin 8)), ((5 : Fin 8))))) (fun p => (onum (attackerCount p (Color.opposite c)) (some (named c (((3 : Fin 8)), ((5 : Fin 8)))))))) ≥ 1) ∧ ((afterMove p (some (named c (((2 : Fin 8)), ((5 : Fin 8))))) (some (named c (((0 : Fin 8)), ((7 : Fin 8))))) (fun p => (onum (attackerCount p (Color.opposite c)) (some (named c (((0 : Fin 8)), ((7 : Fin 8)))))))) ≤ 0) → forcingOf p c 5 ≠ 0 := by
  sorry

-- PROOF FAILED: unsolved goals p : Position left : 1 ≤ afterMove p (some (named p.turn (0, 7))) (some (named p.turn (3, 5))) fun p_1 => onum (attackerCount p_1 p.turn.opposite) (some (named p.turn (3, 5))) right : (afterMove p (some (named p.turn (5, 2))) (some (named p.turn (3, 5))) fun p_1 => onum (seeOf p_1) (so
/-- how many of their pieces attack d6 after the piece on a8 moves to d6 >= 1 AND what the other side wins by the exchange on d6 after the piece on f3 moves to d6 <= 6 (squares named from the mover's side; covers plies k=[0, 1, 2, 3]) -/
theorem premise_38 (p : Position) (c : Color) (hc : p.turn = c) :
    ((afterMove p (some (named c (((0 : Fin 8)), ((7 : Fin 8))))) (some (named c (((3 : Fin 8)), ((5 : Fin 8))))) (fun p => (onum (attackerCount p (Color.opposite c)) (some (named c (((3 : Fin 8)), ((5 : Fin 8)))))))) ≥ 1) ∧ ((afterMove p (some (named c (((5 : Fin 8)), ((2 : Fin 8))))) (some (named c (((3 : Fin 8)), ((5 : Fin 8))))) (fun p => (onum (seeOf p) (some (named c (((3 : Fin 8)), ((5 : Fin 8)))))))) ≤ 6) → forcingOf p c 5 ≠ 0 := by
  sorry

-- PROOF FAILED: unsolved goals p : Position left : (afterMove p (some (named p.turn (0, 7))) (some (named p.turn (6, 4))) fun p_1 => onum (attackerCount p_1 p.turn.opposite) (some (named p.turn (6, 4)))) = 0 right : (afterMove p (some (named p.turn (2, 5))) (some (named p.turn (0, 7))) fun p_1 => onum (attackerCoun
/-- how many of their pieces attack g5 after the piece on a8 moves to g5 <= 0 AND how many of their pieces attack a8 after the piece on c6 moves to a8 <= 0 (squares named from the mover's side; covers plies k=[0, 1, 2, 3]) -/
theorem premise_39 (p : Position) (c : Color) (hc : p.turn = c) :
    ((afterMove p (some (named c (((0 : Fin 8)), ((7 : Fin 8))))) (some (named c (((6 : Fin 8)), ((4 : Fin 8))))) (fun p => (onum (attackerCount p (Color.opposite c)) (some (named c (((6 : Fin 8)), ((4 : Fin 8)))))))) ≤ 0) ∧ ((afterMove p (some (named c (((2 : Fin 8)), ((5 : Fin 8))))) (some (named c (((0 : Fin 8)), ((7 : Fin 8))))) (fun p => (onum (attackerCount p (Color.opposite c)) (some (named c (((0 : Fin 8)), ((7 : Fin 8)))))))) ≤ 0) → forcingOf p c 5 ≠ 0 := by
  sorry

-- PROOF FAILED: unsolved goals p : Position left : (afterMove p (some (named p.turn (0, 7))) (some (named p.turn (6, 4))) fun p_1 => onum (attackerCount p_1 p.turn.opposite) (some (named p.turn (6, 4)))) = 0 right : (afterMove p (some (named p.turn (5, 2))) (some (named p.turn (3, 5))) fun p_1 => onum (seeOf p_1) (
/-- how many of their pieces attack g5 after the piece on a8 moves to g5 <= 0 AND what the other side wins by the exchange on d6 after the piece on f3 moves to d6 <= 6 (squares named from the mover's side; covers plies k=[0, 1, 2, 3]) -/
theorem premise_40 (p : Position) (c : Color) (hc : p.turn = c) :
    ((afterMove p (some (named c (((0 : Fin 8)), ((7 : Fin 8))))) (some (named c (((6 : Fin 8)), ((4 : Fin 8))))) (fun p => (onum (attackerCount p (Color.opposite c)) (some (named c (((6 : Fin 8)), ((4 : Fin 8)))))))) ≤ 0) ∧ ((afterMove p (some (named c (((5 : Fin 8)), ((2 : Fin 8))))) (some (named c (((3 : Fin 8)), ((5 : Fin 8))))) (fun p => (onum (seeOf p) (some (named c (((3 : Fin 8)), ((5 : Fin 8)))))))) ≤ 6) → forcingOf p c 5 ≠ 0 := by
  sorry

end Ideas.Mined.sHa6yUKv_2
