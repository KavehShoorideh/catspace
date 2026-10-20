import Ideas.Battery

/-!
# The skewer (ANALYST_PLAN 52.23 (4), 2026-10-20)

A check by a slider of ours whose line runs THROUGH their king to a piece of theirs behind it. The king must answer the check;
when it steps off the line, the slider attacks the piece behind (zBouR: Rhxf2 cxb7+ Kb8 Rxf2 Qh1+ Ke2 Rxf2+ Kxf2 Qh2+ Ke1 Qxc2 --
the check Qh2+ runs through e1 ... no: through the king on f2 to the queen on c2; the king steps to e1 and the queen falls).

The shape is the ALMOST-BATTERY of `Ideas.Battery` with their KING as the one piece between: `skewerB p c s1 s2 t` -- a slider of
`c` on `s2` aims at `t`, the king of the other colour stands on `s1` strictly between, every other square between is empty.

* `skewer_blocked`: before the king moves the slider does not attack `t` (the king is in the way) -- the attack is not there yet,
  it is the CHECK that makes it come;
* `skewer_attack`: after any move of the king off `s1` (the board otherwise keeping the slider on its square and the squares between
  empty), the slider attacks `t`.

What the proof does NOT say (read by the chase with the rules): that the king must move (it may take the checker or interpose: an
interposition stands between `s2` and `s1`, so it blocks the line to `t` as well; the chase reads the exchange on the interposer),
and what the exchange on `t` is worth.
-/

namespace Ideas.Skewer

open Chess
open Ideas.Battery

/-- The king of the colour other than `c` stands on `u`. -/
def theirKingOnB (p : Position) (c : Color) (u : Square) : Bool :=
  match p.board u with
  | some q => (q.color != c) && (q.kind == .king)
  | none => false

/-- SKEWER: a slider of `c` on `s2` aims at `t`; their king stands on `s1` strictly between; every other square between is empty. -/
def skewerB (p : Position) (c : Color) (s1 s2 t : Square) : Bool :=
  aimsB p c s2 t && betweenB s2 s1 t && theirKingOnB p c s1 &&
    allSquares.all (fun u => !betweenB s2 u t || decide (u = s1) || (p.board u).isNone)

/-- Before the king moves the slider does not attack `t`: the king is in the way. -/
theorem skewer_blocked {p : Position} {c : Color} {s1 s2 t : Square} (h : skewerB p c s1 s2 t = true) :
    pieceAttacksB p c s2 t = false := by
  simp only [skewerB, aimsB, theirKingOnB, Bool.and_eq_true, Bool.or_eq_true, Bool.not_eq_true'] at h
  obtain ⟨⟨⟨ha, hb⟩, hk⟩, _⟩ := h
  unfold pieceAttacksB sliderAttackB clearBetweenB
  cases hq : p.board s2 with
  | none => simp
  | some q =>
    have hocc : p.board s1 ≠ none := by
      intro hn; rw [hn] at hk; simp at hk
    have hmem : s1 ∈ squaresBetween s2 t := by
      rw [squaresBetween_spec]; exact ⟨mem_allSquares s1, hb⟩
    have hnc : ¬ ((squaresBetween s2 t).all (fun u => p.board u == none)) = true := by
      intro hall
      have := (List.all_eq_true.mp hall) s1 hmem
      simp only [beq_iff_eq] at this
      exact hocc this
    obtain ⟨col, kind⟩ := q
    simp only [hq, aimsB] at ha
    simp only [sliderKindB, Bool.and_eq_true, Bool.or_eq_true, beq_iff_eq] at ha
    rcases ha with ⟨⟨hcol, hk'⟩, _⟩
    simp only [hq] at *
    rcases hk' with (hk' | hk') | hk' <;> simp_all

/-- The king steps off the line: the slider attacks `t` (the board otherwise keeping the slider on its square and every square
between that was empty still empty). -/
theorem skewer_attack {p p' : Position} {c : Color} {s1 s2 t : Square} (h : skewerB p c s1 s2 t = true)
    (hrear : p'.board s2 = p.board s2) (hleft : p'.board s1 = none)
    (hrest : ∀ u, betweenB s2 u t = true → u ≠ s1 → p.board u = none → p'.board u = none) :
    pieceAttacksB p' c s2 t = true := by
  have hq : pieceAttacksB (withEmpty p s1) c s2 t = true := by
    simp only [skewerB, aimsB, Bool.and_eq_true, Bool.or_eq_true, Bool.not_eq_true', decide_eq_true_eq] at h
    obtain ⟨⟨⟨ha, hb⟩, _⟩, honly⟩ := h
    have hs12 : s2 ≠ s1 := fun e => by subst e; simp [betweenB] at hb
    unfold pieceAttacksB sliderAttackB clearBetweenB
    have hw : (withEmpty p s1).board s2 = p.board s2 := by simp [withEmpty, boardUpdate, hs12]
    rw [hw]
    cases hq : p.board s2 with
    | none => simp [hq, aimsB] at ha
    | some q =>
      obtain ⟨col, kind⟩ := q
      simp only [hq, aimsB, sliderKindB, Bool.and_eq_true, Bool.or_eq_true, beq_iff_eq] at ha
      have hall : ((squaresBetween s2 t).all (fun u => (withEmpty p s1).board u == none)) = true := by
        refine List.all_eq_true.mpr fun u hu => ?_
        rw [squaresBetween_spec] at hu
        obtain ⟨_, hbu'⟩ := hu
        have hbu : betweenB s2 u t = true := hbu'
        by_cases hu1 : u = s1
        · subst hu1; simp [withEmpty, boardUpdate]
        · have := (List.all_eq_true.mp honly) u (mem_allSquares u)
          simp only [hbu, Bool.not_true, Bool.false_or, hu1, decide_false, Bool.or_false, Option.isNone_iff_eq_none] at this
          simp [withEmpty, boardUpdate, hu1, this]
      rcases ha with ⟨⟨hcol, hk⟩, hg⟩
      simp only [hcol, true_and]
      rcases hk with (hk | hk) | hk <;> subst hk <;> simp_all
  apply pieceAttacksB_mono hq
  · simpa [withEmpty, boardUpdate, show s2 ≠ s1 from fun e => by
      subst e
      simp only [skewerB, Bool.and_eq_true] at h
      obtain ⟨⟨⟨_, hb⟩, _⟩, _⟩ := h
      simp [betweenB] at hb] using hrear
  · intro u hu hn
    by_cases hu1 : u = s1
    · subst hu1; exact hleft
    · apply hrest u hu hu1
      simpa [withEmpty, boardUpdate, hu1] using hn

end Ideas.Skewer
