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
  obtain ⟨q0, hs, hq0c, -⟩ := Ideas.TreeStep.ours_at p c s hown
  obtain ⟨r, ht, hrc, -⟩ := Ideas.TreeStep.ours_at p (Color.opposite c) t hth
  have hnp' := Ideas.TreeStep.not_promoting_at p c s hc hnp q0 hs hq0c
  obtain ⟨hm, haf⟩ := Ideas.Root.after_capture_normal p s t q0 r hs (by rw [hq0c, hc]) hnp' ht (Ideas.TreeStep.moves_of_count p s t hmv)
  rw [haf] at hzm hzo hznp ⊢
  beta_reduce at hzm hzo hznp ⊢
  obtain ⟨q1, hz, hc1, -⟩ := Ideas.TreeStep.ours_at _ (Color.opposite c) z hzo
  have hnp1 := Ideas.Root.not_promoting_at_opp _ c z (by rw [applyMove_turn, hc]) hznp q1 hz hc1
  exact Ideas.Root.dead_end_core p c hc s t z q0 r hs hq0c ht hrc W hw1 hw2 q1 hz hc1 hnp1 (Ideas.TreeStep.moves_of_count _ z t hzm)

/-- THE DEAD END WITH THEIR KING TAKING BACK, the one-step exchange from the capturer's side (plan 45, the sibling of `dead_end`: the words `theirs`/`ours` leave the king out, so a king recapture is named by the object word L, their king's square). The side to move (c) has a piece on s, not about to promote, that can take a piece of theirs on t, worth at least W while what stands on t is worth at most W; after that capture their king L can go to t. Then whatever they reply after the capture, the capturer's balance is at most what it is now. The king is the king by construction (`kingSq`), so there is no ownership leg and no pawn leg for it. The squares are variables: bind them to any position. -/
theorem dead_end_king (p : Position) (c : Color) (hc : p.turn = c) (s t : Square) (W : Nat)
    (hmv : (count (fun u => (oset (movesOf p) (some s)) u && (oset atSq (some t)) u)) ≥ 1)
    (hown : (count (fun u => (piecesOf p c) u && (oset atSq (some s)) u)) ≥ 1)
    (hnp : (count (fun u => (kindSet p c .pawn) u && (fun u => (oset atSq (some s)) u && (oset rankOf (some (named c (((0 : Fin 8)), ((6 : Fin 8)))))) u) u)) ≤ 0)
    (hth : (count (fun u => (piecesOf p (Color.opposite c)) u && (oset atSq (some t)) u)) ≥ 1)
    (hw1 : (onum (worthOf p) (some s)) ≥ W)
    (hw2 : (onum (worthOf p) (some t)) ≤ W)
    (hzm : (afterMove p (some s) (some t) (fun p => (count (fun u => (oset (movesOf p) (kingSq p (Color.opposite c))) u && (oset atSq (some t)) u)))) ≥ 1) :
    (afterMove p (some s) (some t) (fun p => (leastOver p (fun p => (balanceOf p c))))) ≤ (balanceOf p c) := by
  obtain ⟨q0, hs, hq0c, -⟩ := Ideas.TreeStep.ours_at p c s hown
  obtain ⟨r, ht, hrc, hrk⟩ := Ideas.TreeStep.ours_at p (Color.opposite c) t hth
  have hnp' := Ideas.TreeStep.not_promoting_at p c s hc hnp q0 hs hq0c
  obtain ⟨hm, haf⟩ := Ideas.Root.after_capture_normal p s t q0 r hs (by rw [hq0c, hc]) hnp' ht (Ideas.TreeStep.moves_of_count p s t hmv)
  rw [haf] at hzm ⊢
  beta_reduce at hzm ⊢
  obtain ⟨K, hK, hmK⟩ := Ideas.Root.king_moves_of_count _ (Color.opposite c) t hzm
  exact Ideas.Root.dead_end_core p c hc s t K q0 r hs hq0c ht hrc W hw1 hw2 ⟨Color.opposite c, PieceKind.king⟩ hK rfl (by intro hk; cases hk) hmK

end Ideas.Mined.Schemas
