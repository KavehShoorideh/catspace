import Chess
import Ideas.PathLemma
import Ideas.FrameLemma
import Ideas.ExchangeLaw

/-!
# The exchange law is local (2026-10-11)

The exchange law's value on a square `s` (Ideas.ExchangeLaw.seeXB: the capture game with x-rays) reads the piece on
`s`, who attacks `s`, their values, and what stands behind each attacker on its line. Every one of those is a piece on a
square that BEARS on `s` (Ideas.FrameLemma.reachB: `s` itself, a line through it, a knight's move from it). So two
positions that agree on the bearing squares of `s`, with the same side to move, give the same value:
`seeXB_local`. (Attacks are local by `attack_local`, the two-board form of the path lemma.)

What it licenses in the planner (rust/analyst_core): the incremental index (rules.rs exchange_affected: recompute the
exchange on `t` only when a touched square bears on `t`) is exact BY THEOREM, not only by audit; and a refutation
certificate for a chase (lemma.rs) may ignore every move that touches no bearing square of the frame: the exchange on
the target after it is what it was.
-/

namespace Ideas.ExchangeLocal

open Chess
open Ideas.PathLemma
open Ideas.FrameLemma
open Ideas.ExchangeLaw

/-- Two positions agree on every square bearing on `t`. -/
def agreeOn (p p' : Position) (t : Square) : Prop := ∀ x, reachB t x = true → p'.board x = p.board x

/-- A square beyond `q` on the line from `s` (so that `q` is strictly between) lies on a line through `s`. -/
theorem beyond_reach (s q u : Square) (h : betweenB s q u = true) : reachB s u = true := by
  simp only [betweenB, reachB, queenGeometryB, rookGeometryB, bishopGeometryB, sameDiagonalB, sameFileB,
    sameRankB, betweenCoordB, ltB, Bool.or_eq_true, Bool.and_eq_true, bne_iff_ne, ne_eq, beq_iff_eq,
    decide_eq_true_eq, Prod.ext_iff, Fin.ext_iff, absDiff_def] at *
  split_ifs at * <;> omega

/-- A clear line between `s` and `t` reads only squares bearing on `t`. -/
theorem clear_local (p p' : Position) (s t : Square) (h : agreeOn p p' t) :
    clearBetweenB p'.board s t = clearBetweenB p.board s t := by
  have hu : ∀ u, between s u t → p'.board u = p.board u := fun u hb => h u (between_reach s u t hb)
  have h1 : clearBetweenB p'.board s t = true → clearBetweenB p.board s t = true :=
    clear_mono p.board p'.board s t (fun u hb h0 => by rw [← hu u hb]; exact h0)
  have h2 : clearBetweenB p.board s t = true → clearBetweenB p'.board s t = true :=
    clear_mono p'.board p.board s t (fun u hb h0 => by rw [hu u hb]; exact h0)
  cases a : clearBetweenB p'.board s t <;> cases b : clearBetweenB p.board s t <;> simp_all

/-- ATTACKS ARE LOCAL, two boards: whether `s` attacks `t` depends only on the squares bearing on `t`. -/
theorem attack_local (p p' : Position) (c : Color) (s t : Square) (h : agreeOn p p' t) :
    pieceAttacksB p' c s t = pieceAttacksB p c s t := by
  cases hr : reachB t s with
  | false =>
      have h1 : pieceAttacksB p c s t = false := by
        cases e : pieceAttacksB p c s t with
        | false => rfl
        | true => rw [attack_reach p c s t e] at hr; exact absurd hr (by decide)
      have h2 : pieceAttacksB p' c s t = false := by
        cases e : pieceAttacksB p' c s t with
        | false => rfl
        | true => rw [attack_reach p' c s t e] at hr; exact absurd hr (by decide)
      rw [h1, h2]
  | true =>
      have hs := h s hr
      have hc := clear_local p p' s t h
      unfold pieceAttacksB sliderAttackB
      rw [hs, hc]

theorem attackersOf_local (p p' : Position) (c : Color) (s : Square) (h : agreeOn p p' s) :
    attackersOf p' c s = attackersOf p c s := by
  unfold attackersOf
  apply List.filter_congr
  intro q _
  exact attack_local p p' c q s h

theorem valueAt_local (p p' : Position) (s q : Square) (h : agreeOn p p' s) (hq : reachB s q = true) :
    valueAt p' q = valueAt p q := by
  unfold valueAt
  rw [h q hq]

theorem self_reach (s : Square) : reachB s s = true := by simp [reachB]

theorem valuesOf_local (p p' : Position) (c : Color) (s : Square) (h : agreeOn p p' s) :
    valuesOf p' c s = valuesOf p c s := by
  unfold valuesOf
  rw [attackersOf_local p p' c s h]
  congr 1
  apply List.map_congr_left
  intro q hq
  have hq' : pieceAttacksB p c q s = true := (List.mem_filter.mp hq).2
  exact valueAt_local p p' s q h (attack_reach p c q s hq')

/-- The plain exchange is local. -/
theorem seeB_local (p p' : Position) (s : Square) (ht : p'.turn = p.turn) (h : agreeOn p p' s) :
    seeB p' s = seeB p s := by
  unfold seeB
  rw [ht, valuesOf_local p p' p.turn s h, valuesOf_local p p' (Color.opposite p.turn) s h,
    valueAt_local p p' s s h (self_reach s)]

theorem sliderLineB_local (p p' : Position) (s u : Square) (h : agreeOn p p' s) (hu : reachB s u = true) :
    sliderLineB p' u s = sliderLineB p u s := by
  unfold sliderLineB
  rw [h u hu]

theorem takeChain_local (p p' : Position) (s : Square) (h : agreeOn p p' s) (l : List Square)
    (hl : ∀ u ∈ l, reachB s u = true) :
    chainBehind.takeChain p' s l = chainBehind.takeChain p s l := by
  induction l with
  | nil => rfl
  | cons u rest ih =>
      have hu : reachB s u = true := hl u (List.mem_cons_self ..)
      have ih' := ih (fun v hv => hl v (List.mem_cons_of_mem u hv))
      simp only [chainBehind.takeChain]
      rw [sliderLineB_local p p' s u h hu, h u hu, valueAt_local p p' s u h hu, ih']

theorem chainBehind_local (p p' : Position) (s q : Square) (h : agreeOn p p' s) :
    chainBehind p' s q = chainBehind p s q := by
  unfold chainBehind
  have hf : allSquares.filter (fun u => (p'.board u).isSome && betweenB s q u) =
      allSquares.filter (fun u => (p.board u).isSome && betweenB s q u) := by
    apply List.filter_congr
    intro u _
    cases hb : betweenB s q u with
    | false => simp
    | true => rw [h u (beyond_reach s q u hb)]
  rw [hf]
  apply takeChain_local p p' s h
  intro u hu
  have hm := (List.mem_filter.mp (List.mem_mergeSort.mp hu)).2
  simp only [Bool.and_eq_true] at hm
  exact beyond_reach s q u hm.2

theorem raysOf_local (p p' : Position) (c : Color) (s : Square) (h : agreeOn p p' s) :
    raysOf p' c s = raysOf p c s := by
  unfold raysOf
  rw [attackersOf_local p p' c s h]
  congr 1
  apply List.map_congr_left
  intro q hq
  have hq' : pieceAttacksB p c q s = true := (List.mem_filter.mp hq).2
  have hr := attack_reach p c q s hq'
  rw [valueAt_local p p' s q h hr, chainBehind_local p p' s q h]

/-- THE EXCHANGE LAW IS LOCAL: two positions with the same side to move that agree on the squares bearing on `s` give
the same exchange value on `s`, x-rays included. -/
theorem seeXB_local (p p' : Position) (s : Square) (ht : p'.turn = p.turn) (h : agreeOn p p' s) :
    seeXB p' s = seeXB p s := by
  have hr : ∀ c, raysOf p' c s = raysOf p c s := fun c => raysOf_local p p' c s h
  simp only [seeXB, ht, hr, valueAt_local p p' s s h (self_reach s)]

/-- The move form, for the index: a move touching no square bearing on `s`, followed by any move touching none either,
leaves the exchange on `s` as it was (two plies: the same side to move). -/
theorem seeXB_unchanged_two_plies (p : Position) (m r : Move) (s : Square)
    (hm : ∀ x, x ∈ touched p m → reachB s x = false)
    (hr : ∀ x, x ∈ touched (applyMove p m) r → reachB s x = false)
    (ht : (applyMove (applyMove p m) r).turn = p.turn) :
    seeXB (applyMove (applyMove p m) r) s = seeXB p s := by
  apply seeXB_local p _ s ht
  intro x hx
  have hx1 : x ∉ touched p m := fun hin => by rw [hm x hin] at hx; exact absurd hx (by decide)
  have hx2 : x ∉ touched (applyMove p m) r := fun hin => by rw [hr x hin] at hx; exact absurd hx (by decide)
  rw [applyMove_board (applyMove p m) r, applyBoard_off (applyMove p m) r x hx2,
    applyMove_board p m, applyBoard_off p m x hx1]

end Ideas.ExchangeLocal
