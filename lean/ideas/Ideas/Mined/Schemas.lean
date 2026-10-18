import Ideas.Base.Vocabulary
import Ideas.TreeStep
import Ideas.Root

/-! SCHEMAS found by the loop (ANALYST_PLAN 44), written by catspace/incubator/analyst/discovery/selfmine.py (`ensure_schema`): each theorem is a premise
over square VARIABLES, proved once from the step theorems (`step_after_np`, `step_least`) and the one hand-written tail `Ideas.Root.root_tail`. A mined window
instance is `exact` of one of them with its squares and numbers. A schema is appended only when Lean proves it. -/

set_option linter.unusedVariables false
set_option linter.unusedSimpArgs false
set_option maxHeartbeats 400000

namespace Ideas.Mined.Schemas

open Chess hiding between
open Ideas.Vocabulary

/-- THE DELAYED ROOT, depth 1 (plan 44.2, the operators DELAY / RECAPTURE). Our moves x1 -> y1 are named (legal, a piece of ours that is not about to promote); whatever they reply to the last, some move of ours leaves a position in which nothing of ours is attacked, they have no pawn about to promote or to take en passant, they have a reply, and the balance is at least T; the balance now is at most B, and T is at least B + v and at least 68: then the gain v is forced. The squares are variables: bind them to any position. -/
theorem delayed_root_d1 (p : Position) (c : Color) (W : Square) (hc : p.turn = c) (hW : kingSq p c = some W)
    (x1 y1 : Square) (B T : Nat) (v : Int) (hTB : (T : Int) ≥ (B : Int) + v) (hT : T ≥ 68)
    (h0 : (count (fun u => (oset (movesOf p) (some x1)) u && (oset atSq (some y1)) u)) ≥ 1)
    (h1 : (count (fun u => (kindSet p c .pawn) u && (fun u => (oset atSq (some x1)) u && (oset rankOf (some (named c (((0 : Fin 8)), ((6 : Fin 8)))))) u) u)) ≤ 0)
    (h2 : (count (fun u => (piecesOf p c) u && (oset atSq (some x1)) u)) ≥ 1)
    (h3 : (balanceOf p c) ≤ B)
    (h4 : (afterMove p (some x1) (some y1) (fun p => (leastOver p (fun p => (mostOver p (fun p => (if (if ((((count (fun u => (piecesOf p c) u && (attackedBy p (Color.opposite c) (kingSqD p c)) u)) + (count (fun u => (kindSet p (Color.opposite c) .pawn) u && (oset rankOf (some (named c (((0 : Fin 8)), ((1 : Fin 8)))))) u))) + (count (fun u => (kindSet p (Color.opposite c) .pawn) u && (oset rankOf (some (named c (((0 : Fin 8)), ((3 : Fin 8)))))) u))) + (if (leastOver p (fun p => (1))) = (0) then 1 else 0)) = (0) then 1 else 0) ≠ 0 then (balanceOf p c) else (0)))))))) ≥ T) :
    forcingOf p c v ≠ 0 := by
  refine Ideas.TreeStep.forcingOf_ne_zero p c v hc 2 (by simp) ?_
  show Ideas.forcedWithinB c _ p 4 = true
  have hbase : ((materialOf p c : Int) - materialOf p (Color.opposite c)) ≤ (B : Int) - 64 := by unfold balanceOf at h3; omega
  refine Ideas.TreeStep.step_after_np c _ (p) 3 hc _ _ h2 h1 h0 ?_
  intro m1 hm2 haf3
  rw [haf3] at h4
  beta_reduce at h4
  refine Ideas.TreeStep.step_least c _ (applyMove (p) m1) 2 (by simp only [applyMove_turn, hc] <;> (cases c <;> first | rfl | decide) : (applyMove (p) m1).turn ≠ c) _ T (by omega) h4 ?_
  intro r4 hr5
  have h4' := Ideas.TreeStep.least_floor _ _ T h4 r4 hr5
  beta_reduce at h4'
  exact Ideas.Root.root_tail c _ v (applyMove (applyMove (p) m1) r4) (by simp only [applyMove_turn, hc] <;> (cases c <;> first | rfl | decide) : (applyMove (applyMove (p) m1) r4).turn = c) (T - 64) (by omega) (by omega) (by omega)

/-- THE DELAYED ROOT, depth 2 (plan 44.2, the operators DELAY / RECAPTURE). Our moves x1 -> y1 and x2 -> y2 are named (legal, a piece of ours that is not about to promote); whatever they reply to the last, some move of ours leaves a position in which nothing of ours is attacked, they have no pawn about to promote or to take en passant, they have a reply, and the balance is at least T; the balance now is at most B, and T is at least B + v and at least 68: then the gain v is forced. The squares are variables: bind them to any position. -/
theorem delayed_root_d2 (p : Position) (c : Color) (W : Square) (hc : p.turn = c) (hW : kingSq p c = some W)
    (x1 y1 x2 y2 : Square) (B T : Nat) (v : Int) (hTB : (T : Int) ≥ (B : Int) + v) (hT : T ≥ 68)
    (h0 : (afterMove p (some x1) (some y1) (fun p => (leastOver p (fun p => (count (fun u => (oset (movesOf p) (some x2)) u && (oset atSq (some y2)) u)))))) ≥ 1)
    (h1 : (afterMove p (some x1) (some y1) (fun p => (leastOver p (fun p => (count (fun u => (piecesOf p c) u && (oset atSq (some x2)) u)))))) ≥ 1)
    (h2 : (afterMove p (some x1) (some y1) (fun p => (mostOver p (fun p => (count (fun u => (kindSet p c .pawn) u && (fun u => (oset atSq (some x2)) u && (oset rankOf (some (named c (((0 : Fin 8)), ((6 : Fin 8)))))) u) u)))))) ≤ 0)
    (h3 : (count (fun u => (oset (movesOf p) (some x1)) u && (oset atSq (some y1)) u)) ≥ 1)
    (h4 : (count (fun u => (kindSet p c .pawn) u && (fun u => (oset atSq (some x1)) u && (oset rankOf (some (named c (((0 : Fin 8)), ((6 : Fin 8)))))) u) u)) ≤ 0)
    (h5 : (count (fun u => (piecesOf p c) u && (oset atSq (some x1)) u)) ≥ 1)
    (h6 : (balanceOf p c) ≤ B)
    (h7 : (afterMove p (some x1) (some y1) (fun p => (leastOver p (fun p => (afterMove p (some x2) (some y2) (fun p => (leastOver p (fun p => (mostOver p (fun p => (if (if ((((count (fun u => (piecesOf p c) u && (attackedBy p (Color.opposite c) (kingSqD p c)) u)) + (count (fun u => (kindSet p (Color.opposite c) .pawn) u && (oset rankOf (some (named c (((0 : Fin 8)), ((1 : Fin 8)))))) u))) + (count (fun u => (kindSet p (Color.opposite c) .pawn) u && (oset rankOf (some (named c (((0 : Fin 8)), ((3 : Fin 8)))))) u))) + (if (leastOver p (fun p => (1))) = (0) then 1 else 0)) = (0) then 1 else 0) ≠ 0 then (balanceOf p c) else (0)))))))))))) ≥ T) :
    forcingOf p c v ≠ 0 := by
  refine Ideas.TreeStep.forcingOf_ne_zero p c v hc 3 (by simp) ?_
  show Ideas.forcedWithinB c _ p 6 = true
  have hbase : ((materialOf p c : Int) - materialOf p (Color.opposite c)) ≤ (B : Int) - 64 := by unfold balanceOf at h6; omega
  refine Ideas.TreeStep.step_after_np c _ (p) 5 hc _ _ h5 h4 h3 ?_
  intro m1 hm2 haf3
  rw [haf3] at h0
  beta_reduce at h0
  rw [haf3] at h1
  beta_reduce at h1
  rw [haf3] at h2
  beta_reduce at h2
  rw [haf3] at h7
  beta_reduce at h7
  refine Ideas.TreeStep.step_least c _ (applyMove (p) m1) 4 (by simp only [applyMove_turn, hc] <;> (cases c <;> first | rfl | decide) : (applyMove (p) m1).turn ≠ c) _ 1 (by omega) h0 ?_
  intro r4 hr5
  have h0' := Ideas.TreeStep.least_floor _ _ 1 h0 r4 hr5
  beta_reduce at h0'
  have h1' := Ideas.TreeStep.least_floor _ _ 1 h1 r4 hr5
  beta_reduce at h1'
  have h2' := le_trans (Ideas.Vocabulary.le_mostOver _ _ r4 hr5) h2
  beta_reduce at h2'
  have h7' := Ideas.TreeStep.least_floor _ _ T h7 r4 hr5
  beta_reduce at h7'
  refine Ideas.TreeStep.step_after_np c _ (applyMove (applyMove (p) m1) r4) 3 (by simp only [applyMove_turn, hc] <;> (cases c <;> first | rfl | decide) : (applyMove (applyMove (p) m1) r4).turn = c) _ _ h1' h2' h0' ?_
  intro m6 hm7 haf8
  rw [haf8] at h7'
  beta_reduce at h7'
  refine Ideas.TreeStep.step_least c _ (applyMove (applyMove (applyMove (p) m1) r4) m6) 2 (by simp only [applyMove_turn, hc] <;> (cases c <;> first | rfl | decide) : (applyMove (applyMove (applyMove (p) m1) r4) m6).turn ≠ c) _ T (by omega) h7' ?_
  intro r9 hr10
  have h7'' := Ideas.TreeStep.least_floor _ _ T h7' r9 hr10
  beta_reduce at h7''
  exact Ideas.Root.root_tail c _ v (applyMove (applyMove (applyMove (applyMove (p) m1) r4) m6) r9) (by simp only [applyMove_turn, hc] <;> (cases c <;> first | rfl | decide) : (applyMove (applyMove (applyMove (applyMove (p) m1) r4) m6) r9).turn = c) (T - 64) (by omega) (by omega) (by omega)

/-- THE DELAYED ROOT, WEAKENED (they take back at most one piece), depth 1 (plan 44.2, the operators DELAY / RECAPTURE). Our moves x1 -> y1 are named (legal, a piece of ours that is not about to promote); whatever they reply to the last, some move of ours leaves a position in which they have no pawn about to promote or to take en passant, they have a reply, and the balance less the worth of the most valuable piece of ours that is attacked is at least T; the balance now is at most B, and T is at least B + v and at least 68: then the gain v is forced. The squares are variables: bind them to any position. -/
theorem delayed_root_max_d1 (p : Position) (c : Color) (W : Square) (hc : p.turn = c) (hW : kingSq p c = some W)
    (x1 y1 : Square) (B T : Nat) (v : Int) (hTB : (T : Int) ≥ (B : Int) + v) (hT : T ≥ 68)
    (h0 : (count (fun u => (oset (movesOf p) (some x1)) u && (oset atSq (some y1)) u)) ≥ 1)
    (h1 : (count (fun u => (kindSet p c .pawn) u && (fun u => (oset atSq (some x1)) u && (oset rankOf (some (named c (((0 : Fin 8)), ((6 : Fin 8)))))) u) u)) ≤ 0)
    (h2 : (count (fun u => (piecesOf p c) u && (oset atSq (some x1)) u)) ≥ 1)
    (h3 : (balanceOf p c) ≤ B)
    (h4 : (afterMove p (some x1) (some y1) (fun p => (leastOver p (fun p => (mostOver p (fun p => (if (if (((count (fun u => (kindSet p (Color.opposite c) .pawn) u && (oset rankOf (some (named c (((0 : Fin 8)), ((1 : Fin 8)))))) u)) + (count (fun u => (kindSet p (Color.opposite c) .pawn) u && (oset rankOf (some (named c (((0 : Fin 8)), ((3 : Fin 8)))))) u))) + (if (leastOver p (fun p => (1))) = (0) then 1 else 0)) = (0) then 1 else 0) ≠ 0 then (nDiff (balanceOf p c) (maxWorthOf p (fun u => (piecesOf p c) u && (attackedBy p (Color.opposite c) (kingSqD p c)) u))) else (0)))))))) ≥ T) :
    forcingOf p c v ≠ 0 := by
  refine Ideas.TreeStep.forcingOf_ne_zero p c v hc 2 (by simp) ?_
  show Ideas.forcedWithinB c _ p 4 = true
  have hbase : ((materialOf p c : Int) - materialOf p (Color.opposite c)) ≤ (B : Int) - 64 := by unfold balanceOf at h3; omega
  refine Ideas.TreeStep.step_after_np c _ (p) 3 hc _ _ h2 h1 h0 ?_
  intro m1 hm2 haf3
  rw [haf3] at h4
  beta_reduce at h4
  refine Ideas.TreeStep.step_least c _ (applyMove (p) m1) 2 (by simp only [applyMove_turn, hc] <;> (cases c <;> first | rfl | decide) : (applyMove (p) m1).turn ≠ c) _ T (by omega) h4 ?_
  intro r4 hr5
  have h4' := Ideas.TreeStep.least_floor _ _ T h4 r4 hr5
  beta_reduce at h4'
  exact Ideas.Root.root_tail_max c _ v (applyMove (applyMove (p) m1) r4) (by simp only [applyMove_turn, hc] <;> (cases c <;> first | rfl | decide) : (applyMove (applyMove (p) m1) r4).turn = c) (T - 64) (by omega) (by omega) (by omega)

/-- THE DELAYED ROOT, WEAKENED (they take back at most one piece), depth 2 (plan 44.2, the operators DELAY / RECAPTURE). Our moves x1 -> y1 and x2 -> y2 are named (legal, a piece of ours that is not about to promote); whatever they reply to the last, some move of ours leaves a position in which they have no pawn about to promote or to take en passant, they have a reply, and the balance less the worth of the most valuable piece of ours that is attacked is at least T; the balance now is at most B, and T is at least B + v and at least 68: then the gain v is forced. The squares are variables: bind them to any position. -/
theorem delayed_root_max_d2 (p : Position) (c : Color) (W : Square) (hc : p.turn = c) (hW : kingSq p c = some W)
    (x1 y1 x2 y2 : Square) (B T : Nat) (v : Int) (hTB : (T : Int) ≥ (B : Int) + v) (hT : T ≥ 68)
    (h0 : (afterMove p (some x1) (some y1) (fun p => (leastOver p (fun p => (count (fun u => (oset (movesOf p) (some x2)) u && (oset atSq (some y2)) u)))))) ≥ 1)
    (h1 : (afterMove p (some x1) (some y1) (fun p => (leastOver p (fun p => (count (fun u => (piecesOf p c) u && (oset atSq (some x2)) u)))))) ≥ 1)
    (h2 : (afterMove p (some x1) (some y1) (fun p => (mostOver p (fun p => (count (fun u => (kindSet p c .pawn) u && (fun u => (oset atSq (some x2)) u && (oset rankOf (some (named c (((0 : Fin 8)), ((6 : Fin 8)))))) u) u)))))) ≤ 0)
    (h3 : (count (fun u => (oset (movesOf p) (some x1)) u && (oset atSq (some y1)) u)) ≥ 1)
    (h4 : (count (fun u => (kindSet p c .pawn) u && (fun u => (oset atSq (some x1)) u && (oset rankOf (some (named c (((0 : Fin 8)), ((6 : Fin 8)))))) u) u)) ≤ 0)
    (h5 : (count (fun u => (piecesOf p c) u && (oset atSq (some x1)) u)) ≥ 1)
    (h6 : (balanceOf p c) ≤ B)
    (h7 : (afterMove p (some x1) (some y1) (fun p => (leastOver p (fun p => (afterMove p (some x2) (some y2) (fun p => (leastOver p (fun p => (mostOver p (fun p => (if (if (((count (fun u => (kindSet p (Color.opposite c) .pawn) u && (oset rankOf (some (named c (((0 : Fin 8)), ((1 : Fin 8)))))) u)) + (count (fun u => (kindSet p (Color.opposite c) .pawn) u && (oset rankOf (some (named c (((0 : Fin 8)), ((3 : Fin 8)))))) u))) + (if (leastOver p (fun p => (1))) = (0) then 1 else 0)) = (0) then 1 else 0) ≠ 0 then (nDiff (balanceOf p c) (maxWorthOf p (fun u => (piecesOf p c) u && (attackedBy p (Color.opposite c) (kingSqD p c)) u))) else (0)))))))))))) ≥ T) :
    forcingOf p c v ≠ 0 := by
  refine Ideas.TreeStep.forcingOf_ne_zero p c v hc 3 (by simp) ?_
  show Ideas.forcedWithinB c _ p 6 = true
  have hbase : ((materialOf p c : Int) - materialOf p (Color.opposite c)) ≤ (B : Int) - 64 := by unfold balanceOf at h6; omega
  refine Ideas.TreeStep.step_after_np c _ (p) 5 hc _ _ h5 h4 h3 ?_
  intro m1 hm2 haf3
  rw [haf3] at h0
  beta_reduce at h0
  rw [haf3] at h1
  beta_reduce at h1
  rw [haf3] at h2
  beta_reduce at h2
  rw [haf3] at h7
  beta_reduce at h7
  refine Ideas.TreeStep.step_least c _ (applyMove (p) m1) 4 (by simp only [applyMove_turn, hc] <;> (cases c <;> first | rfl | decide) : (applyMove (p) m1).turn ≠ c) _ 1 (by omega) h0 ?_
  intro r4 hr5
  have h0' := Ideas.TreeStep.least_floor _ _ 1 h0 r4 hr5
  beta_reduce at h0'
  have h1' := Ideas.TreeStep.least_floor _ _ 1 h1 r4 hr5
  beta_reduce at h1'
  have h2' := le_trans (Ideas.Vocabulary.le_mostOver _ _ r4 hr5) h2
  beta_reduce at h2'
  have h7' := Ideas.TreeStep.least_floor _ _ T h7 r4 hr5
  beta_reduce at h7'
  refine Ideas.TreeStep.step_after_np c _ (applyMove (applyMove (p) m1) r4) 3 (by simp only [applyMove_turn, hc] <;> (cases c <;> first | rfl | decide) : (applyMove (applyMove (p) m1) r4).turn = c) _ _ h1' h2' h0' ?_
  intro m6 hm7 haf8
  rw [haf8] at h7'
  beta_reduce at h7'
  refine Ideas.TreeStep.step_least c _ (applyMove (applyMove (applyMove (p) m1) r4) m6) 2 (by simp only [applyMove_turn, hc] <;> (cases c <;> first | rfl | decide) : (applyMove (applyMove (applyMove (p) m1) r4) m6).turn ≠ c) _ T (by omega) h7' ?_
  intro r9 hr10
  have h7'' := Ideas.TreeStep.least_floor _ _ T h7' r9 hr10
  beta_reduce at h7''
  exact Ideas.Root.root_tail_max c _ v (applyMove (applyMove (applyMove (applyMove (p) m1) r4) m6) r9) (by simp only [applyMove_turn, hc] <;> (cases c <;> first | rfl | decide) : (applyMove (applyMove (applyMove (applyMove (p) m1) r4) m6) r9).turn = c) (T - 64) (by omega) (by omega) (by omega)

/-- THE DELAYED BALANCE (the minimax tail), depth 1 (plan 44.2, the operators DELAY / RECAPTURE). Our moves x1 -> y1 are named (legal, a piece of ours that is not about to promote); whatever they reply to the last, some move of ours, then whatever they reply, leaves the balance at least T; the balance now is at most B, and T is at least B + v and at least 68: then the gain v is forced. The squares are variables: bind them to any position. -/
theorem delayed_balance_d1 (p : Position) (c : Color) (hc : p.turn = c)
    (x1 y1 : Square) (B T : Nat) (v : Int) (hTB : (T : Int) ≥ (B : Int) + v) (hT : T ≥ 68)
    (h0 : (count (fun u => (oset (movesOf p) (some x1)) u && (oset atSq (some y1)) u)) ≥ 1)
    (h1 : (count (fun u => (kindSet p c .pawn) u && (fun u => (oset atSq (some x1)) u && (oset rankOf (some (named c (((0 : Fin 8)), ((6 : Fin 8)))))) u) u)) ≤ 0)
    (h2 : (count (fun u => (piecesOf p c) u && (oset atSq (some x1)) u)) ≥ 1)
    (h3 : (balanceOf p c) ≤ B)
    (h4 : (afterMove p (some x1) (some y1) (fun p => (leastOver p (fun p => (mostOver p (fun p => (leastOver p (fun p => (balanceOf p c))))))))) ≥ T) :
    forcingOf p c v ≠ 0 := by
  refine Ideas.TreeStep.forcingOf_ne_zero p c v hc 2 (by simp) ?_
  show Ideas.forcedWithinB c _ p 4 = true
  have hbase : ((materialOf p c : Int) - materialOf p (Color.opposite c)) ≤ (B : Int) - 64 := by unfold balanceOf at h3; omega
  refine Ideas.TreeStep.step_after_np c _ (p) 3 hc _ _ h2 h1 h0 ?_
  intro m1 hm2 haf3
  rw [haf3] at h4
  beta_reduce at h4
  refine Ideas.TreeStep.step_least c _ (applyMove (p) m1) 2 (by simp only [applyMove_turn, hc] <;> (cases c <;> first | rfl | decide) : (applyMove (p) m1).turn ≠ c) _ T (by omega) h4 ?_
  intro r4 hr5
  have h4' := Ideas.TreeStep.least_floor _ _ T h4 r4 hr5
  beta_reduce at h4'
  exact Ideas.TreeStep.tail_two c _ v (applyMove (applyMove (p) m1) r4) (by simp only [applyMove_turn, hc] <;> (cases c <;> first | rfl | decide) : (applyMove (applyMove (p) m1) r4).turn = c) (T - 64) (by omega) (by omega) (by omega)

/-- THE DELAYED BALANCE (the minimax tail), depth 2 (plan 44.2, the operators DELAY / RECAPTURE). Our moves x1 -> y1 and x2 -> y2 are named (legal, a piece of ours that is not about to promote); whatever they reply to the last, some move of ours, then whatever they reply, leaves the balance at least T; the balance now is at most B, and T is at least B + v and at least 68: then the gain v is forced. The squares are variables: bind them to any position. -/
theorem delayed_balance_d2 (p : Position) (c : Color) (hc : p.turn = c)
    (x1 y1 x2 y2 : Square) (B T : Nat) (v : Int) (hTB : (T : Int) ≥ (B : Int) + v) (hT : T ≥ 68)
    (h0 : (afterMove p (some x1) (some y1) (fun p => (leastOver p (fun p => (count (fun u => (oset (movesOf p) (some x2)) u && (oset atSq (some y2)) u)))))) ≥ 1)
    (h1 : (afterMove p (some x1) (some y1) (fun p => (leastOver p (fun p => (count (fun u => (piecesOf p c) u && (oset atSq (some x2)) u)))))) ≥ 1)
    (h2 : (afterMove p (some x1) (some y1) (fun p => (mostOver p (fun p => (count (fun u => (kindSet p c .pawn) u && (fun u => (oset atSq (some x2)) u && (oset rankOf (some (named c (((0 : Fin 8)), ((6 : Fin 8)))))) u) u)))))) ≤ 0)
    (h3 : (count (fun u => (oset (movesOf p) (some x1)) u && (oset atSq (some y1)) u)) ≥ 1)
    (h4 : (count (fun u => (kindSet p c .pawn) u && (fun u => (oset atSq (some x1)) u && (oset rankOf (some (named c (((0 : Fin 8)), ((6 : Fin 8)))))) u) u)) ≤ 0)
    (h5 : (count (fun u => (piecesOf p c) u && (oset atSq (some x1)) u)) ≥ 1)
    (h6 : (balanceOf p c) ≤ B)
    (h7 : (afterMove p (some x1) (some y1) (fun p => (leastOver p (fun p => (afterMove p (some x2) (some y2) (fun p => (leastOver p (fun p => (mostOver p (fun p => (leastOver p (fun p => (balanceOf p c))))))))))))) ≥ T) :
    forcingOf p c v ≠ 0 := by
  refine Ideas.TreeStep.forcingOf_ne_zero p c v hc 3 (by simp) ?_
  show Ideas.forcedWithinB c _ p 6 = true
  have hbase : ((materialOf p c : Int) - materialOf p (Color.opposite c)) ≤ (B : Int) - 64 := by unfold balanceOf at h6; omega
  refine Ideas.TreeStep.step_after_np c _ (p) 5 hc _ _ h5 h4 h3 ?_
  intro m1 hm2 haf3
  rw [haf3] at h0
  beta_reduce at h0
  rw [haf3] at h1
  beta_reduce at h1
  rw [haf3] at h2
  beta_reduce at h2
  rw [haf3] at h7
  beta_reduce at h7
  refine Ideas.TreeStep.step_least c _ (applyMove (p) m1) 4 (by simp only [applyMove_turn, hc] <;> (cases c <;> first | rfl | decide) : (applyMove (p) m1).turn ≠ c) _ 1 (by omega) h0 ?_
  intro r4 hr5
  have h0' := Ideas.TreeStep.least_floor _ _ 1 h0 r4 hr5
  beta_reduce at h0'
  have h1' := Ideas.TreeStep.least_floor _ _ 1 h1 r4 hr5
  beta_reduce at h1'
  have h2' := le_trans (Ideas.Vocabulary.le_mostOver _ _ r4 hr5) h2
  beta_reduce at h2'
  have h7' := Ideas.TreeStep.least_floor _ _ T h7 r4 hr5
  beta_reduce at h7'
  refine Ideas.TreeStep.step_after_np c _ (applyMove (applyMove (p) m1) r4) 3 (by simp only [applyMove_turn, hc] <;> (cases c <;> first | rfl | decide) : (applyMove (applyMove (p) m1) r4).turn = c) _ _ h1' h2' h0' ?_
  intro m6 hm7 haf8
  rw [haf8] at h7'
  beta_reduce at h7'
  refine Ideas.TreeStep.step_least c _ (applyMove (applyMove (applyMove (p) m1) r4) m6) 2 (by simp only [applyMove_turn, hc] <;> (cases c <;> first | rfl | decide) : (applyMove (applyMove (applyMove (p) m1) r4) m6).turn ≠ c) _ T (by omega) h7' ?_
  intro r9 hr10
  have h7'' := Ideas.TreeStep.least_floor _ _ T h7' r9 hr10
  beta_reduce at h7''
  exact Ideas.TreeStep.tail_two c _ v (applyMove (applyMove (applyMove (applyMove (p) m1) r4) m6) r9) (by simp only [applyMove_turn, hc] <;> (cases c <;> first | rfl | decide) : (applyMove (applyMove (applyMove (applyMove (p) m1) r4) m6) r9).turn = c) (T - 64) (by omega) (by omega) (by omega)

/-- THE DEAD END, the one-step exchange from the capturer's side (plan 45.2). The side to move (c) has a piece on s, not about to promote, that can take a piece of theirs on t, and the piece on s is worth at least W while what stands on t is worth at most W; after that capture a piece of theirs on z, not about to promote, can take back on t. Then whatever they reply after the capture, the capturer's balance is at most what it is now: the capture gains nothing, an attack that is a dead end for the attacker. The squares are variables: bind them to any position. -/
theorem dead_end (p : Position) (c : Color) (hc : p.turn = c) (s t z : Square) (W : Nat)
    (hmv : (count (fun u => (oset (movesOf p) (some s)) u && (oset atSq (some t)) u)) ≥ 1)
    (hown : (count (fun u => (piecesOf p c) u && (oset atSq (some s)) u)) ≥ 1)
    (hnp : (count (fun u => (kindSet p c .pawn) u && (fun u => (oset atSq (some s)) u && (oset rankOf (some (named c (((0 : Fin 8)), ((6 : Fin 8)))))) u) u)) ≤ 0)
    (hth : (count (fun u => (piecesOf p (Color.opposite c)) u && (oset atSq (some t)) u)) ≥ 1)
    (hw1 : (onum (worthOf p) (some s)) ≥ W)
    (hw2 : (onum (worthOf p) (some t)) ≤ W)
    (hzm : (afterMove p (some s) (some t) (fun p => (count (fun u => (oset (movesOf p) (some z)) u && (oset atSq (some t)) u)))) ≥ 1)
    (hzo : (afterMove p (some s) (some t) (fun p => (count (fun u => (piecesOf p (Color.opposite c)) u && (oset atSq (some z)) u)))) ≥ 1)
    (hznp : (afterMove p (some s) (some t) (fun p => (count (fun u => (kindSet p (Color.opposite c) .pawn) u && (fun u => (oset atSq (some z)) u && (oset rankOf (some (named c (((0 : Fin 8)), ((1 : Fin 8)))))) u) u)))) ≤ 0) :
    (afterMove p (some s) (some t) (fun p => (leastOver p (fun p => (balanceOf p c))))) ≤ (balanceOf p c) := by
  obtain ⟨q0, hs, hq0c⟩ := Ideas.TreeStep.ours_at p c s hown
  obtain ⟨r, ht, hrc⟩ := Ideas.TreeStep.ours_at p (Color.opposite c) t hth
  have hnp' := Ideas.TreeStep.not_promoting_at p c s hc hnp q0 hs hq0c
  obtain ⟨hm, haf⟩ := Ideas.Root.after_capture_normal p s t q0 r hs (by rw [hq0c, hc]) hnp' ht (Ideas.TreeStep.moves_of_count p s t hmv)
  rw [haf] at hzm hzo hznp ⊢
  beta_reduce at hzm hzo hznp ⊢
  obtain ⟨q1, hz, hc1⟩ := Ideas.TreeStep.ours_at _ (Color.opposite c) z hzo
  have hnp1 := Ideas.Root.not_promoting_at_opp _ c z (by rw [applyMove_turn, hc]) hznp q1 hz hc1
  exact Ideas.Root.dead_end_core p c hc s t z q0 r hs hq0c ht hrc W hw1 hw2 q1 hz hc1 hnp1 (Ideas.TreeStep.moves_of_count _ z t hzm)

/-- THE DELAYED ROOT, depth 1 (plan 44.2, the operators DELAY / RECAPTURE). Our moves x1 -> y1 are named (legal, a piece of ours that is not about to promote); whatever they reply to the last, some move of ours leaves a position in which nothing of ours is attacked, they have no pawn about to promote or to take en passant, they have a reply, and the balance is at least T; the balance now is at most B, and T is at least B + v, with C = T - 4 (the tail reads the balance AND our material, at least 4, so the game is alive; no floor on the balance): then the gain v is forced. The squares are variables: bind them to any position. -/
theorem delayed_root_alive_d1 (p : Position) (c : Color) (W : Square) (hc : p.turn = c) (hW : kingSq p c = some W)
    (x1 y1 : Square) (B T C : Nat) (v : Int) (hTB : (T : Int) ≥ (B : Int) + v) (hCT : C + 4 = T)
    (h0 : (count (fun u => (oset (movesOf p) (some x1)) u && (oset atSq (some y1)) u)) ≥ 1)
    (h1 : (count (fun u => (kindSet p c .pawn) u && (fun u => (oset atSq (some x1)) u && (oset rankOf (some (named c (((0 : Fin 8)), ((6 : Fin 8)))))) u) u)) ≤ 0)
    (h2 : (count (fun u => (piecesOf p c) u && (oset atSq (some x1)) u)) ≥ 1)
    (h3 : (balanceOf p c) ≤ B)
    (h4 : (afterMove p (some x1) (some y1) (fun p => (leastOver p (fun p => (mostOver p (fun p => (min (if (if ((((count (fun u => (piecesOf p c) u && (attackedBy p (Color.opposite c) (kingSqD p c)) u)) + (count (fun u => (kindSet p (Color.opposite c) .pawn) u && (oset rankOf (some (named c (((0 : Fin 8)), ((1 : Fin 8)))))) u))) + (count (fun u => (kindSet p (Color.opposite c) .pawn) u && (oset rankOf (some (named c (((0 : Fin 8)), ((3 : Fin 8)))))) u))) + (if (leastOver p (fun p => (1))) = (0) then 1 else 0)) = (0) then 1 else 0) ≠ 0 then (balanceOf p c) else (0)) ((materialOf p c) + C)))))))) ≥ T) :
    forcingOf p c v ≠ 0 := by
  refine Ideas.TreeStep.forcingOf_ne_zero p c v hc 2 (by simp) ?_
  show Ideas.forcedWithinB c _ p 4 = true
  have hbase : ((materialOf p c : Int) - materialOf p (Color.opposite c)) ≤ (B : Int) - 64 := by unfold balanceOf at h3; omega
  refine Ideas.TreeStep.step_after_np c _ (p) 3 hc _ _ h2 h1 h0 ?_
  intro m1 hm2 haf3
  rw [haf3] at h4
  beta_reduce at h4
  refine Ideas.TreeStep.step_least c _ (applyMove (p) m1) 2 (by simp only [applyMove_turn, hc] <;> (cases c <;> first | rfl | decide) : (applyMove (p) m1).turn ≠ c) _ T (by omega) h4 ?_
  intro r4 hr5
  have h4' := Ideas.TreeStep.least_floor _ _ T h4 r4 hr5
  beta_reduce at h4'
  exact Ideas.Root.root_tail_alive c _ v (applyMove (applyMove (p) m1) r4) (by simp only [applyMove_turn, hc] <;> (cases c <;> first | rfl | decide) : (applyMove (applyMove (p) m1) r4).turn = c) T C (by omega) (by omega) (by omega)

/-- THE DELAYED ROOT, depth 2 (plan 44.2, the operators DELAY / RECAPTURE). Our moves x1 -> y1 and x2 -> y2 are named (legal, a piece of ours that is not about to promote); whatever they reply to the last, some move of ours leaves a position in which nothing of ours is attacked, they have no pawn about to promote or to take en passant, they have a reply, and the balance is at least T; the balance now is at most B, and T is at least B + v, with C = T - 4 (the tail reads the balance AND our material, at least 4, so the game is alive; no floor on the balance): then the gain v is forced. The squares are variables: bind them to any position. -/
theorem delayed_root_alive_d2 (p : Position) (c : Color) (W : Square) (hc : p.turn = c) (hW : kingSq p c = some W)
    (x1 y1 x2 y2 : Square) (B T C : Nat) (v : Int) (hTB : (T : Int) ≥ (B : Int) + v) (hCT : C + 4 = T)
    (h0 : (afterMove p (some x1) (some y1) (fun p => (leastOver p (fun p => (count (fun u => (oset (movesOf p) (some x2)) u && (oset atSq (some y2)) u)))))) ≥ 1)
    (h1 : (afterMove p (some x1) (some y1) (fun p => (leastOver p (fun p => (count (fun u => (piecesOf p c) u && (oset atSq (some x2)) u)))))) ≥ 1)
    (h2 : (afterMove p (some x1) (some y1) (fun p => (mostOver p (fun p => (count (fun u => (kindSet p c .pawn) u && (fun u => (oset atSq (some x2)) u && (oset rankOf (some (named c (((0 : Fin 8)), ((6 : Fin 8)))))) u) u)))))) ≤ 0)
    (h3 : (count (fun u => (oset (movesOf p) (some x1)) u && (oset atSq (some y1)) u)) ≥ 1)
    (h4 : (count (fun u => (kindSet p c .pawn) u && (fun u => (oset atSq (some x1)) u && (oset rankOf (some (named c (((0 : Fin 8)), ((6 : Fin 8)))))) u) u)) ≤ 0)
    (h5 : (count (fun u => (piecesOf p c) u && (oset atSq (some x1)) u)) ≥ 1)
    (h6 : (balanceOf p c) ≤ B)
    (h7 : (afterMove p (some x1) (some y1) (fun p => (leastOver p (fun p => (afterMove p (some x2) (some y2) (fun p => (leastOver p (fun p => (mostOver p (fun p => (min (if (if ((((count (fun u => (piecesOf p c) u && (attackedBy p (Color.opposite c) (kingSqD p c)) u)) + (count (fun u => (kindSet p (Color.opposite c) .pawn) u && (oset rankOf (some (named c (((0 : Fin 8)), ((1 : Fin 8)))))) u))) + (count (fun u => (kindSet p (Color.opposite c) .pawn) u && (oset rankOf (some (named c (((0 : Fin 8)), ((3 : Fin 8)))))) u))) + (if (leastOver p (fun p => (1))) = (0) then 1 else 0)) = (0) then 1 else 0) ≠ 0 then (balanceOf p c) else (0)) ((materialOf p c) + C)))))))))))) ≥ T) :
    forcingOf p c v ≠ 0 := by
  refine Ideas.TreeStep.forcingOf_ne_zero p c v hc 3 (by simp) ?_
  show Ideas.forcedWithinB c _ p 6 = true
  have hbase : ((materialOf p c : Int) - materialOf p (Color.opposite c)) ≤ (B : Int) - 64 := by unfold balanceOf at h6; omega
  refine Ideas.TreeStep.step_after_np c _ (p) 5 hc _ _ h5 h4 h3 ?_
  intro m1 hm2 haf3
  rw [haf3] at h0
  beta_reduce at h0
  rw [haf3] at h1
  beta_reduce at h1
  rw [haf3] at h2
  beta_reduce at h2
  rw [haf3] at h7
  beta_reduce at h7
  refine Ideas.TreeStep.step_least c _ (applyMove (p) m1) 4 (by simp only [applyMove_turn, hc] <;> (cases c <;> first | rfl | decide) : (applyMove (p) m1).turn ≠ c) _ 1 (by omega) h0 ?_
  intro r4 hr5
  have h0' := Ideas.TreeStep.least_floor _ _ 1 h0 r4 hr5
  beta_reduce at h0'
  have h1' := Ideas.TreeStep.least_floor _ _ 1 h1 r4 hr5
  beta_reduce at h1'
  have h2' := le_trans (Ideas.Vocabulary.le_mostOver _ _ r4 hr5) h2
  beta_reduce at h2'
  have h7' := Ideas.TreeStep.least_floor _ _ T h7 r4 hr5
  beta_reduce at h7'
  refine Ideas.TreeStep.step_after_np c _ (applyMove (applyMove (p) m1) r4) 3 (by simp only [applyMove_turn, hc] <;> (cases c <;> first | rfl | decide) : (applyMove (applyMove (p) m1) r4).turn = c) _ _ h1' h2' h0' ?_
  intro m6 hm7 haf8
  rw [haf8] at h7'
  beta_reduce at h7'
  refine Ideas.TreeStep.step_least c _ (applyMove (applyMove (applyMove (p) m1) r4) m6) 2 (by simp only [applyMove_turn, hc] <;> (cases c <;> first | rfl | decide) : (applyMove (applyMove (applyMove (p) m1) r4) m6).turn ≠ c) _ T (by omega) h7' ?_
  intro r9 hr10
  have h7'' := Ideas.TreeStep.least_floor _ _ T h7' r9 hr10
  beta_reduce at h7''
  exact Ideas.Root.root_tail_alive c _ v (applyMove (applyMove (applyMove (applyMove (p) m1) r4) m6) r9) (by simp only [applyMove_turn, hc] <;> (cases c <;> first | rfl | decide) : (applyMove (applyMove (applyMove (applyMove (p) m1) r4) m6) r9).turn = c) T C (by omega) (by omega) (by omega)

/-- THE DELAYED ROOT, WEAKENED (they take back at most one piece), depth 1 (plan 44.2, the operators DELAY / RECAPTURE). Our moves x1 -> y1 are named (legal, a piece of ours that is not about to promote); whatever they reply to the last, some move of ours leaves a position in which they have no pawn about to promote or to take en passant, they have a reply, and the balance less the worth of the most valuable piece of ours that is attacked is at least T; the balance now is at most B, and T is at least B + v, with C = T - 13 (the tail reads the balance AND our material, at least 13, so the game is alive; no floor on the balance): then the gain v is forced. The squares are variables: bind them to any position. -/
theorem delayed_root_max_alive_d1 (p : Position) (c : Color) (W : Square) (hc : p.turn = c) (hW : kingSq p c = some W)
    (x1 y1 : Square) (B T C : Nat) (v : Int) (hTB : (T : Int) ≥ (B : Int) + v) (hCT : C + 13 = T)
    (h0 : (count (fun u => (oset (movesOf p) (some x1)) u && (oset atSq (some y1)) u)) ≥ 1)
    (h1 : (count (fun u => (kindSet p c .pawn) u && (fun u => (oset atSq (some x1)) u && (oset rankOf (some (named c (((0 : Fin 8)), ((6 : Fin 8)))))) u) u)) ≤ 0)
    (h2 : (count (fun u => (piecesOf p c) u && (oset atSq (some x1)) u)) ≥ 1)
    (h3 : (balanceOf p c) ≤ B)
    (h4 : (afterMove p (some x1) (some y1) (fun p => (leastOver p (fun p => (mostOver p (fun p => (min (if (if (((count (fun u => (kindSet p (Color.opposite c) .pawn) u && (oset rankOf (some (named c (((0 : Fin 8)), ((1 : Fin 8)))))) u)) + (count (fun u => (kindSet p (Color.opposite c) .pawn) u && (oset rankOf (some (named c (((0 : Fin 8)), ((3 : Fin 8)))))) u))) + (if (leastOver p (fun p => (1))) = (0) then 1 else 0)) = (0) then 1 else 0) ≠ 0 then (nDiff (balanceOf p c) (maxWorthOf p (fun u => (piecesOf p c) u && (attackedBy p (Color.opposite c) (kingSqD p c)) u))) else (0)) ((materialOf p c) + C)))))))) ≥ T) :
    forcingOf p c v ≠ 0 := by
  refine Ideas.TreeStep.forcingOf_ne_zero p c v hc 2 (by simp) ?_
  show Ideas.forcedWithinB c _ p 4 = true
  have hbase : ((materialOf p c : Int) - materialOf p (Color.opposite c)) ≤ (B : Int) - 64 := by unfold balanceOf at h3; omega
  refine Ideas.TreeStep.step_after_np c _ (p) 3 hc _ _ h2 h1 h0 ?_
  intro m1 hm2 haf3
  rw [haf3] at h4
  beta_reduce at h4
  refine Ideas.TreeStep.step_least c _ (applyMove (p) m1) 2 (by simp only [applyMove_turn, hc] <;> (cases c <;> first | rfl | decide) : (applyMove (p) m1).turn ≠ c) _ T (by omega) h4 ?_
  intro r4 hr5
  have h4' := Ideas.TreeStep.least_floor _ _ T h4 r4 hr5
  beta_reduce at h4'
  exact Ideas.Root.root_tail_max_alive c _ v (applyMove (applyMove (p) m1) r4) (by simp only [applyMove_turn, hc] <;> (cases c <;> first | rfl | decide) : (applyMove (applyMove (p) m1) r4).turn = c) T C (by omega) (by omega) (by omega)

/-- THE DELAYED ROOT, WEAKENED (they take back at most one piece), depth 2 (plan 44.2, the operators DELAY / RECAPTURE). Our moves x1 -> y1 and x2 -> y2 are named (legal, a piece of ours that is not about to promote); whatever they reply to the last, some move of ours leaves a position in which they have no pawn about to promote or to take en passant, they have a reply, and the balance less the worth of the most valuable piece of ours that is attacked is at least T; the balance now is at most B, and T is at least B + v, with C = T - 13 (the tail reads the balance AND our material, at least 13, so the game is alive; no floor on the balance): then the gain v is forced. The squares are variables: bind them to any position. -/
theorem delayed_root_max_alive_d2 (p : Position) (c : Color) (W : Square) (hc : p.turn = c) (hW : kingSq p c = some W)
    (x1 y1 x2 y2 : Square) (B T C : Nat) (v : Int) (hTB : (T : Int) ≥ (B : Int) + v) (hCT : C + 13 = T)
    (h0 : (afterMove p (some x1) (some y1) (fun p => (leastOver p (fun p => (count (fun u => (oset (movesOf p) (some x2)) u && (oset atSq (some y2)) u)))))) ≥ 1)
    (h1 : (afterMove p (some x1) (some y1) (fun p => (leastOver p (fun p => (count (fun u => (piecesOf p c) u && (oset atSq (some x2)) u)))))) ≥ 1)
    (h2 : (afterMove p (some x1) (some y1) (fun p => (mostOver p (fun p => (count (fun u => (kindSet p c .pawn) u && (fun u => (oset atSq (some x2)) u && (oset rankOf (some (named c (((0 : Fin 8)), ((6 : Fin 8)))))) u) u)))))) ≤ 0)
    (h3 : (count (fun u => (oset (movesOf p) (some x1)) u && (oset atSq (some y1)) u)) ≥ 1)
    (h4 : (count (fun u => (kindSet p c .pawn) u && (fun u => (oset atSq (some x1)) u && (oset rankOf (some (named c (((0 : Fin 8)), ((6 : Fin 8)))))) u) u)) ≤ 0)
    (h5 : (count (fun u => (piecesOf p c) u && (oset atSq (some x1)) u)) ≥ 1)
    (h6 : (balanceOf p c) ≤ B)
    (h7 : (afterMove p (some x1) (some y1) (fun p => (leastOver p (fun p => (afterMove p (some x2) (some y2) (fun p => (leastOver p (fun p => (mostOver p (fun p => (min (if (if (((count (fun u => (kindSet p (Color.opposite c) .pawn) u && (oset rankOf (some (named c (((0 : Fin 8)), ((1 : Fin 8)))))) u)) + (count (fun u => (kindSet p (Color.opposite c) .pawn) u && (oset rankOf (some (named c (((0 : Fin 8)), ((3 : Fin 8)))))) u))) + (if (leastOver p (fun p => (1))) = (0) then 1 else 0)) = (0) then 1 else 0) ≠ 0 then (nDiff (balanceOf p c) (maxWorthOf p (fun u => (piecesOf p c) u && (attackedBy p (Color.opposite c) (kingSqD p c)) u))) else (0)) ((materialOf p c) + C)))))))))))) ≥ T) :
    forcingOf p c v ≠ 0 := by
  refine Ideas.TreeStep.forcingOf_ne_zero p c v hc 3 (by simp) ?_
  show Ideas.forcedWithinB c _ p 6 = true
  have hbase : ((materialOf p c : Int) - materialOf p (Color.opposite c)) ≤ (B : Int) - 64 := by unfold balanceOf at h6; omega
  refine Ideas.TreeStep.step_after_np c _ (p) 5 hc _ _ h5 h4 h3 ?_
  intro m1 hm2 haf3
  rw [haf3] at h0
  beta_reduce at h0
  rw [haf3] at h1
  beta_reduce at h1
  rw [haf3] at h2
  beta_reduce at h2
  rw [haf3] at h7
  beta_reduce at h7
  refine Ideas.TreeStep.step_least c _ (applyMove (p) m1) 4 (by simp only [applyMove_turn, hc] <;> (cases c <;> first | rfl | decide) : (applyMove (p) m1).turn ≠ c) _ 1 (by omega) h0 ?_
  intro r4 hr5
  have h0' := Ideas.TreeStep.least_floor _ _ 1 h0 r4 hr5
  beta_reduce at h0'
  have h1' := Ideas.TreeStep.least_floor _ _ 1 h1 r4 hr5
  beta_reduce at h1'
  have h2' := le_trans (Ideas.Vocabulary.le_mostOver _ _ r4 hr5) h2
  beta_reduce at h2'
  have h7' := Ideas.TreeStep.least_floor _ _ T h7 r4 hr5
  beta_reduce at h7'
  refine Ideas.TreeStep.step_after_np c _ (applyMove (applyMove (p) m1) r4) 3 (by simp only [applyMove_turn, hc] <;> (cases c <;> first | rfl | decide) : (applyMove (applyMove (p) m1) r4).turn = c) _ _ h1' h2' h0' ?_
  intro m6 hm7 haf8
  rw [haf8] at h7'
  beta_reduce at h7'
  refine Ideas.TreeStep.step_least c _ (applyMove (applyMove (applyMove (p) m1) r4) m6) 2 (by simp only [applyMove_turn, hc] <;> (cases c <;> first | rfl | decide) : (applyMove (applyMove (applyMove (p) m1) r4) m6).turn ≠ c) _ T (by omega) h7' ?_
  intro r9 hr10
  have h7'' := Ideas.TreeStep.least_floor _ _ T h7' r9 hr10
  beta_reduce at h7''
  exact Ideas.Root.root_tail_max_alive c _ v (applyMove (applyMove (applyMove (applyMove (p) m1) r4) m6) r9) (by simp only [applyMove_turn, hc] <;> (cases c <;> first | rfl | decide) : (applyMove (applyMove (applyMove (applyMove (p) m1) r4) m6) r9).turn = c) T C (by omega) (by omega) (by omega)

/-- THE DELAYED BALANCE (the minimax tail), depth 1 (plan 44.2, the operators DELAY / RECAPTURE). Our moves x1 -> y1 are named (legal, a piece of ours that is not about to promote); whatever they reply to the last, some move of ours, then whatever they reply, leaves the balance at least T; the balance now is at most B, and T is at least B + v, with C = T - 4 (the tail reads the balance AND our material, at least 4, so the game is alive; no floor on the balance): then the gain v is forced. The squares are variables: bind them to any position. -/
theorem delayed_balance_alive_d1 (p : Position) (c : Color) (hc : p.turn = c)
    (x1 y1 : Square) (B T C : Nat) (v : Int) (hTB : (T : Int) ≥ (B : Int) + v) (hCT : C + 4 = T)
    (h0 : (count (fun u => (oset (movesOf p) (some x1)) u && (oset atSq (some y1)) u)) ≥ 1)
    (h1 : (count (fun u => (kindSet p c .pawn) u && (fun u => (oset atSq (some x1)) u && (oset rankOf (some (named c (((0 : Fin 8)), ((6 : Fin 8)))))) u) u)) ≤ 0)
    (h2 : (count (fun u => (piecesOf p c) u && (oset atSq (some x1)) u)) ≥ 1)
    (h3 : (balanceOf p c) ≤ B)
    (h4 : (afterMove p (some x1) (some y1) (fun p => (leastOver p (fun p => (mostOver p (fun p => (leastOver p (fun p => (min (balanceOf p c) ((materialOf p c) + C)))))))))) ≥ T) :
    forcingOf p c v ≠ 0 := by
  refine Ideas.TreeStep.forcingOf_ne_zero p c v hc 2 (by simp) ?_
  show Ideas.forcedWithinB c _ p 4 = true
  have hbase : ((materialOf p c : Int) - materialOf p (Color.opposite c)) ≤ (B : Int) - 64 := by unfold balanceOf at h3; omega
  refine Ideas.TreeStep.step_after_np c _ (p) 3 hc _ _ h2 h1 h0 ?_
  intro m1 hm2 haf3
  rw [haf3] at h4
  beta_reduce at h4
  refine Ideas.TreeStep.step_least c _ (applyMove (p) m1) 2 (by simp only [applyMove_turn, hc] <;> (cases c <;> first | rfl | decide) : (applyMove (p) m1).turn ≠ c) _ T (by omega) h4 ?_
  intro r4 hr5
  have h4' := Ideas.TreeStep.least_floor _ _ T h4 r4 hr5
  beta_reduce at h4'
  exact Ideas.TreeStep.tail_two_alive c _ v (applyMove (applyMove (p) m1) r4) (by simp only [applyMove_turn, hc] <;> (cases c <;> first | rfl | decide) : (applyMove (applyMove (p) m1) r4).turn = c) T C (by omega) (by omega) (by omega)

/-- THE DELAYED BALANCE (the minimax tail), depth 2 (plan 44.2, the operators DELAY / RECAPTURE). Our moves x1 -> y1 and x2 -> y2 are named (legal, a piece of ours that is not about to promote); whatever they reply to the last, some move of ours, then whatever they reply, leaves the balance at least T; the balance now is at most B, and T is at least B + v, with C = T - 4 (the tail reads the balance AND our material, at least 4, so the game is alive; no floor on the balance): then the gain v is forced. The squares are variables: bind them to any position. -/
theorem delayed_balance_alive_d2 (p : Position) (c : Color) (hc : p.turn = c)
    (x1 y1 x2 y2 : Square) (B T C : Nat) (v : Int) (hTB : (T : Int) ≥ (B : Int) + v) (hCT : C + 4 = T)
    (h0 : (afterMove p (some x1) (some y1) (fun p => (leastOver p (fun p => (count (fun u => (oset (movesOf p) (some x2)) u && (oset atSq (some y2)) u)))))) ≥ 1)
    (h1 : (afterMove p (some x1) (some y1) (fun p => (leastOver p (fun p => (count (fun u => (piecesOf p c) u && (oset atSq (some x2)) u)))))) ≥ 1)
    (h2 : (afterMove p (some x1) (some y1) (fun p => (mostOver p (fun p => (count (fun u => (kindSet p c .pawn) u && (fun u => (oset atSq (some x2)) u && (oset rankOf (some (named c (((0 : Fin 8)), ((6 : Fin 8)))))) u) u)))))) ≤ 0)
    (h3 : (count (fun u => (oset (movesOf p) (some x1)) u && (oset atSq (some y1)) u)) ≥ 1)
    (h4 : (count (fun u => (kindSet p c .pawn) u && (fun u => (oset atSq (some x1)) u && (oset rankOf (some (named c (((0 : Fin 8)), ((6 : Fin 8)))))) u) u)) ≤ 0)
    (h5 : (count (fun u => (piecesOf p c) u && (oset atSq (some x1)) u)) ≥ 1)
    (h6 : (balanceOf p c) ≤ B)
    (h7 : (afterMove p (some x1) (some y1) (fun p => (leastOver p (fun p => (afterMove p (some x2) (some y2) (fun p => (leastOver p (fun p => (mostOver p (fun p => (leastOver p (fun p => (min (balanceOf p c) ((materialOf p c) + C)))))))))))))) ≥ T) :
    forcingOf p c v ≠ 0 := by
  refine Ideas.TreeStep.forcingOf_ne_zero p c v hc 3 (by simp) ?_
  show Ideas.forcedWithinB c _ p 6 = true
  have hbase : ((materialOf p c : Int) - materialOf p (Color.opposite c)) ≤ (B : Int) - 64 := by unfold balanceOf at h6; omega
  refine Ideas.TreeStep.step_after_np c _ (p) 5 hc _ _ h5 h4 h3 ?_
  intro m1 hm2 haf3
  rw [haf3] at h0
  beta_reduce at h0
  rw [haf3] at h1
  beta_reduce at h1
  rw [haf3] at h2
  beta_reduce at h2
  rw [haf3] at h7
  beta_reduce at h7
  refine Ideas.TreeStep.step_least c _ (applyMove (p) m1) 4 (by simp only [applyMove_turn, hc] <;> (cases c <;> first | rfl | decide) : (applyMove (p) m1).turn ≠ c) _ 1 (by omega) h0 ?_
  intro r4 hr5
  have h0' := Ideas.TreeStep.least_floor _ _ 1 h0 r4 hr5
  beta_reduce at h0'
  have h1' := Ideas.TreeStep.least_floor _ _ 1 h1 r4 hr5
  beta_reduce at h1'
  have h2' := le_trans (Ideas.Vocabulary.le_mostOver _ _ r4 hr5) h2
  beta_reduce at h2'
  have h7' := Ideas.TreeStep.least_floor _ _ T h7 r4 hr5
  beta_reduce at h7'
  refine Ideas.TreeStep.step_after_np c _ (applyMove (applyMove (p) m1) r4) 3 (by simp only [applyMove_turn, hc] <;> (cases c <;> first | rfl | decide) : (applyMove (applyMove (p) m1) r4).turn = c) _ _ h1' h2' h0' ?_
  intro m6 hm7 haf8
  rw [haf8] at h7'
  beta_reduce at h7'
  refine Ideas.TreeStep.step_least c _ (applyMove (applyMove (applyMove (p) m1) r4) m6) 2 (by simp only [applyMove_turn, hc] <;> (cases c <;> first | rfl | decide) : (applyMove (applyMove (applyMove (p) m1) r4) m6).turn ≠ c) _ T (by omega) h7' ?_
  intro r9 hr10
  have h7'' := Ideas.TreeStep.least_floor _ _ T h7' r9 hr10
  beta_reduce at h7''
  exact Ideas.TreeStep.tail_two_alive c _ v (applyMove (applyMove (applyMove (applyMove (p) m1) r4) m6) r9) (by simp only [applyMove_turn, hc] <;> (cases c <;> first | rfl | decide) : (applyMove (applyMove (applyMove (applyMove (p) m1) r4) m6) r9).turn = c) T C (by omega) (by omega) (by omega)

/-- THE MATE IN ONE (plan 47, written by hand: the schema generator does not regenerate it). Our piece on x can legally go to y (and is not a pawn about to promote); after that move, the side to move (theirs) has no legal move (`movecount` 0) and is in check (`check` at least 1): then a gain of any v is forced, because the goal counts a mate. The squares are variables: bind them to any position. -/
theorem mate_in_one (p : Position) (c : Color) (hc : p.turn = c) (x y : Square) (v : Int)
    (h0 : (count (fun u => (oset (movesOf p) (some x)) u && (oset atSq (some y)) u)) ≥ 1)
    (h1 : (count (fun u => (kindSet p c .pawn) u && (fun u => (oset atSq (some x)) u && (oset rankOf (some (named c (((0 : Fin 8)), ((6 : Fin 8)))))) u) u)) ≤ 0)
    (h2 : (count (fun u => (piecesOf p c) u && (oset atSq (some x)) u)) ≥ 1)
    (h3 : (afterMove p (some x) (some y) (fun p => (moveCountOf p))) ≤ 0)
    (h4 : (afterMove p (some x) (some y) (fun p => (checkOf p))) ≥ 1) :
    forcingOf p c v ≠ 0 := by
  refine Ideas.TreeStep.forcingOf_ne_zero p c v hc 1 (by simp) ?_
  show Ideas.forcedWithinB c _ p 2 = true
  refine Ideas.TreeStep.step_after_np c _ p 1 hc x y h2 h1 h0 ?_
  intro m hm haf
  rw [haf] at h3 h4
  beta_reduce at h3 h4
  have hturn : (applyMove p m).turn = Color.opposite c := by rw [applyMove_turn, hc]
  have hne : ((applyMove p m).turn != c) = true := by
    rw [hturn]; cases c <;> decide
  have hempty : (legalMoves (applyMove p m)).isEmpty = true := by
    have h3' : (legalMoves (applyMove p m)).length = 0 := by unfold moveCountOf at h3; omega
    exact List.isEmpty_iff_length_eq_zero.mpr h3'
  have hchk : inCheckB (applyMove p m) (applyMove p m).turn = true := by
    unfold checkOf at h4
    by_contra hn
    rw [if_neg hn] at h4
    omega
  show Ideas.forcedWithinB c _ (applyMove p m) (0 + 1) = true
  simp only [Ideas.forcedWithinB, gainGoal, Bool.or_eq_true]
  left
  right
  simp only [Bool.and_eq_true]
  exact ⟨⟨hne, hempty⟩, hchk⟩

/-- THE MATE IN TWO (plan 49, written by hand: the schema generator does not regenerate it). Our piece on x can legally go to y (and is not a pawn about to promote); after that move, whatever they reply, some move of ours leaves `(min check (eq movecount 0))` at least 1 (check 1 and movecount 0, a mate; the plan's `diff check movecount` is |check - movecount| in the words, which a non-mate with replies also satisfies, so it is not used): then a gain of any v is forced, because the goal counts a mate. A stalemate or an immediate mate after x -> y reads 0 (`least` over no replies), so it does not satisfy the tail. The squares are variables: bind them to any position. -/
theorem mate_in_two (p : Position) (c : Color) (hc : p.turn = c) (x y : Square) (v : Int)
    (h0 : (count (fun u => (oset (movesOf p) (some x)) u && (oset atSq (some y)) u)) ≥ 1)
    (h1 : (count (fun u => (kindSet p c .pawn) u && (fun u => (oset atSq (some x)) u && (oset rankOf (some (named c (((0 : Fin 8)), ((6 : Fin 8)))))) u) u)) ≤ 0)
    (h2 : (count (fun u => (piecesOf p c) u && (oset atSq (some x)) u)) ≥ 1)
    (h3 : (afterMove p (some x) (some y) (fun p => (leastOver p (fun p => (mostOver p (fun p => (min (checkOf p) (if (moveCountOf p) = 0 then 1 else 0)))))))) ≥ 1) :
    forcingOf p c v ≠ 0 := by
  refine Ideas.TreeStep.forcingOf_ne_zero p c v hc 2 (by simp) ?_
  show Ideas.forcedWithinB c _ p 4 = true
  refine Ideas.TreeStep.step_after_np c _ p 3 hc x y h2 h1 h0 ?_
  intro m1 hm1 haf
  rw [haf] at h3
  beta_reduce at h3
  have ht1 : (applyMove p m1).turn ≠ c := Ideas.TreeStep.turn_after_one p c hc m1
  refine Ideas.TreeStep.step_least c _ (applyMove p m1) 2 ht1 _ 1 (by omega) h3 ?_
  intro r hr
  have h3' := Ideas.TreeStep.least_floor _ _ 1 h3 r hr
  beta_reduce at h3'
  have ht2 : (applyMove (applyMove p m1) r).turn = c := Ideas.TreeStep.turn_after_two p c hc m1 r
  refine Ideas.TreeStep.step_most c _ (applyMove (applyMove p m1) r) 1 ht2 _ 1 (by omega) h3' ?_
  intro m3 hm3 h3''
  beta_reduce at h3''
  have hturn : (applyMove (applyMove (applyMove p m1) r) m3).turn ≠ c :=
    Ideas.TreeStep.turn_after_one _ c ht2 m3
  have hchk1 : checkOf (applyMove (applyMove (applyMove p m1) r) m3) = 1 := by
    have hle : checkOf (applyMove (applyMove (applyMove p m1) r) m3) ≤ 1 := by unfold checkOf; split <;> omega
    have hge : checkOf (applyMove (applyMove (applyMove p m1) r) m3) ≥ 1 := by omega
    omega
  have hmc : moveCountOf (applyMove (applyMove (applyMove p m1) r) m3) = 0 := by
    by_contra hn
    rw [if_neg hn] at h3''
    omega
  have hno : legalMoves (applyMove (applyMove (applyMove p m1) r) m3) = [] := by
    unfold moveCountOf at hmc
    exact List.length_eq_zero_iff.mp hmc
  have hchk : inCheckB (applyMove (applyMove (applyMove p m1) r) m3) (applyMove (applyMove (applyMove p m1) r) m3).turn = true := by
    unfold checkOf at hchk1
    by_contra hn
    rw [if_neg hn] at hchk1
    omega
  exact Ideas.TreeStep.leaf c _ _ 1 (Ideas.TreeStep.gainGoal_of_mate c _ v _ hturn hno hchk)

/-- THE DEEP DELAYED BALANCE (the minimax tail, one level deeper; plan 48), depth 1 (plan 44.2, the operators DELAY / RECAPTURE). Our moves x1 -> y1 are named (legal, a piece of ours that is not about to promote); whatever they reply to the last, some move of ours, whatever they reply, some move of ours, then whatever they reply, leaves the balance at least T; the balance now is at most B, and T is at least B + v, with C = T - 4 (the tail reads the balance AND our material, at least 4, so the game is alive; no floor on the balance): then the gain v is forced. The squares are variables: bind them to any position. -/
theorem delayed_balance_deep_alive_d1 (p : Position) (c : Color) (hc : p.turn = c)
    (x1 y1 : Square) (B T C : Nat) (v : Int) (hTB : (T : Int) ≥ (B : Int) + v) (hCT : C + 4 = T)
    (h0 : (count (fun u => (oset (movesOf p) (some x1)) u && (oset atSq (some y1)) u)) ≥ 1)
    (h1 : (count (fun u => (kindSet p c .pawn) u && (fun u => (oset atSq (some x1)) u && (oset rankOf (some (named c (((0 : Fin 8)), ((6 : Fin 8)))))) u) u)) ≤ 0)
    (h2 : (count (fun u => (piecesOf p c) u && (oset atSq (some x1)) u)) ≥ 1)
    (h3 : (balanceOf p c) ≤ B)
    (h4 : (afterMove p (some x1) (some y1) (fun p => (leastOver p (fun p => (mostOver p (fun p => (leastOver p (fun p => (mostOver p (fun p => (leastOver p (fun p => (min (balanceOf p c) ((materialOf p c) + C)))))))))))))) ≥ T) :
    forcingOf p c v ≠ 0 := by
  refine Ideas.TreeStep.forcingOf_ne_zero p c v hc 3 (by simp) ?_
  show Ideas.forcedWithinB c _ p 6 = true
  have hbase : ((materialOf p c : Int) - materialOf p (Color.opposite c)) ≤ (B : Int) - 64 := by unfold balanceOf at h3; omega
  refine Ideas.TreeStep.step_after_np c _ (p) 5 hc _ _ h2 h1 h0 ?_
  intro m1 hm2 haf3
  rw [haf3] at h4
  beta_reduce at h4
  refine Ideas.TreeStep.step_least c _ (applyMove (p) m1) 4 (by simp only [applyMove_turn, hc] <;> (cases c <;> first | rfl | decide) : (applyMove (p) m1).turn ≠ c) _ T (by omega) h4 ?_
  intro r4 hr5
  have h4' := Ideas.TreeStep.least_floor _ _ T h4 r4 hr5
  beta_reduce at h4'
  exact Ideas.TreeStep.tail_four_alive c _ v (applyMove (applyMove (p) m1) r4) (by simp only [applyMove_turn, hc] <;> (cases c <;> first | rfl | decide) : (applyMove (applyMove (p) m1) r4).turn = c) T C (by omega) (by omega) (by omega)

-- TODO (plan 52.12, the chase): prove the COMPOSED statement the agenda's chase claims. For our named first move x -> y: whatever they reply (`step_least`), the inner claim at the
-- position after the reply is one of the schemas above (its instance is the receipt), and the balance after the reply plus the inner gain is at least 1. The chain is
-- `step_after` (our named move) + `step_least` (every reply) over the inner claims; no new theorem in that slice. Until it is proved a chase claim is evidence, tagged `chase: ...`.

end Ideas.Mined.Schemas
