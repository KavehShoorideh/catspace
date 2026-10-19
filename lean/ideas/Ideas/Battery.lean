import Ideas.Mate

/-!
# The battery and the discovery (ANALYST_PLAN 52.18 (2), Kaveh 2026-10-19)

"A move creates a new threat from three sources: the moved piece (fork), the VACATED LINE (discovery; a battery: the attacker
count on a square the move never touches rises), or a broken defence." Today `xray` is squares attacked through one piece and
`discovery` is king-only; here the general words, with the one fact the fix rests on.

* BATTERY (`battery`): a slider of ours on `s2` aims along a line at the square `t`; the squares strictly between hold only
  pieces of ours (at least the one on `s1`) or nothing -- our front pieces ride in front of the rear slider, all aimed at `t`
  (Kaveh's position: Bg2 behind Qb7, with the Ne4 in front of the Q, aimed at a8).
* ALMOST A BATTERY (`almostBattery`): the same with `s1` the ONLY piece between -- one move of that piece away from the direct
  attack.
* DISCOVERY: a move of the front piece off the line. `almostBattery_blocked`: before it, the rear slider does not attack `t`;
  `discovered_attack`: after any move of `s1` off the line (the board otherwise keeping the rear on its square and the squares
  between empty), it does. `battery_persists`: the other pieces of the chain stay ours-or-empty, so the rear still rides
  behind them (the x-ray of the exchange law).
-/

namespace Ideas.Battery

open Chess

/-- A slider (rook, bishop, queen). -/
def sliderKindB (k : PieceKind) : Bool := k == .rook || k == .bishop || k == .queen

/-- A slider of colour `c` stands on `s2` and its geometry reaches `t` (blockers aside). -/
def aimsB (p : Position) (c : Color) (s2 t : Square) : Bool :=
  match p.board s2 with
  | some q => (q.color == c) && sliderKindB q.kind &&
      ((q.kind == .rook && rookGeometryB s2 t) || (q.kind == .bishop && bishopGeometryB s2 t) || (q.kind == .queen && queenGeometryB s2 t))
  | none => false

/-- The square `u` is empty or holds a piece of colour `c`. -/
def ownOrEmptyB (p : Position) (c : Color) (u : Square) : Bool :=
  match p.board u with
  | none => true
  | some q => q.color == c

/-- BATTERY: a slider of `c` on `s2` aims at `t`; `s1` lies strictly between and holds a piece of `c`; every square between holds a
piece of `c` or nothing. -/
def batteryB (p : Position) (c : Color) (s1 s2 t : Square) : Bool :=
  aimsB p c s2 t && betweenB s2 s1 t && (p.board s1).isSome && ownOrEmptyB p c s1 &&
    allSquares.all (fun u => !betweenB s2 u t || ownOrEmptyB p c u)

/-- ALMOST A BATTERY: a battery whose only piece between is `s1`. -/
def almostBatteryB (p : Position) (c : Color) (s1 s2 t : Square) : Bool :=
  batteryB p c s1 s2 t && allSquares.all (fun u => !betweenB s2 u t || decide (u = s1) || (p.board u).isNone)

/-- Before the front piece moves the rear slider does not attack `t`: the piece on `s1` is in the way. -/
theorem almostBattery_blocked {p : Position} {c : Color} {s1 s2 t : Square} (h : almostBatteryB p c s1 s2 t = true) :
    pieceAttacksB p c s2 t = false := by
  simp only [almostBatteryB, batteryB, aimsB, Bool.and_eq_true, Bool.or_eq_true, Bool.not_eq_true'] at h
  obtain ⟨⟨⟨⟨⟨ha, hb⟩, hs1⟩, _⟩, _⟩, _⟩ := h
  unfold pieceAttacksB sliderAttackB clearBetweenB
  cases hq : p.board s2 with
  | none => simp
  | some q =>
    have hocc : p.board s1 ≠ none := by
      intro hn; rw [hn] at hs1; simp at hs1
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
    rcases ha with ⟨⟨hcol, hk⟩, _⟩
    simp only [hq] at *
    rcases hk with (hk | hk) | hk <;> simp_all

/-- The front piece leaves: the rear slider attacks `t` (a move of `s1` off the line, the board otherwise keeping the rear on its
square and every square between that was empty still empty). The vacated line is the threat. -/
theorem discovered_attack {p p' : Position} {c : Color} {s1 s2 t : Square} (h : almostBatteryB p c s1 s2 t = true)
    (hrear : p'.board s2 = p.board s2) (hleft : p'.board s1 = none)
    (hrest : ∀ u, betweenB s2 u t = true → u ≠ s1 → p.board u = none → p'.board u = none) :
    pieceAttacksB p' c s2 t = true := by
  have hq : pieceAttacksB (withEmpty p s1) c s2 t = true := by
    simp only [almostBatteryB, batteryB, aimsB, Bool.and_eq_true, Bool.or_eq_true, Bool.not_eq_true', decide_eq_true_eq] at h
    obtain ⟨⟨⟨⟨⟨ha, hb⟩, hs1⟩, _⟩, _⟩, honly⟩ := h
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
      simp only [almostBatteryB, batteryB, Bool.and_eq_true] at h
      obtain ⟨⟨⟨⟨⟨_, hb⟩, _⟩, _⟩, _⟩, _⟩ := h
      simp [betweenB] at hb] using hrear
  · intro u hu hn
    by_cases hu1 : u = s1
    · subst hu1; exact hleft
    · apply hrest u hu hu1
      simpa [withEmpty, boardUpdate, hu1] using hn

/-- THE REAR STAYS BEHIND: when the piece on `s1` leaves, every square that was between is still empty or ours, so what is left of
the chain is still a chain (the x-ray of the exchange law counts the rear slider behind the front pieces that remain). -/
theorem battery_persists {p p' : Position} {c : Color} {s1 s2 t : Square} (h : batteryB p c s1 s2 t = true)
    (hleft : p'.board s1 = none) (hrest : ∀ u, u ≠ s1 → p'.board u = p.board u) :
    ∀ u, betweenB s2 u t = true → ownOrEmptyB p' c u = true := by
  simp only [batteryB, Bool.and_eq_true, Bool.or_eq_true, Bool.not_eq_true', decide_eq_true_eq] at h
  obtain ⟨_, hall⟩ := h
  intro u hu
  by_cases hu1 : u = s1
  · subst hu1; simp [ownOrEmptyB, hleft]
  · have := (List.all_eq_true.mp hall) u (mem_allSquares u)
    simp only [hu, Bool.not_true, Bool.false_or] at this
    simpa [ownOrEmptyB, hrest u hu1] using this

end Ideas.Battery
